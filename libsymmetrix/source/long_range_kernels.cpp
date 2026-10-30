#include "long_range_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "KokkosBlas.hpp"

namespace symmetrix::execution::long_range {
namespace {

constexpr double pi = 3.14159265358979323846;

// Fixed unit-convention constants (the design note's Unit convention): the
// rationalized-epsilon_0 Coulomb constant of the package's (e, V, Angstrom)
// internal system and the cubic Madelung constant of the molecular box term.
constexpr double field_constant = 1.0 / (5.526349406e-3);
constexpr double cubic_madelung = 2.837297;
// Below eight atoms launch latency dominates and the canonical serial order
// retains stricter FP64 behavior for pathological coincident pairs.  From
// eight atoms onward sender-parallel team reductions are already faster on
// sm120 and avoid the serial O(N) walk per receiver channel.
constexpr std::size_t realspace_cuda_team_min_nodes = 8;

KOKKOS_FUNCTION double realspace_erf(const double value)
{
    return Kokkos::erf(value);
}

KOKKOS_FUNCTION double realspace_norm_accumulate(
    const double left, const double right, const double accumulator)
{
    // The canonical real-space policy evaluates the Cartesian dot product in
    // FP64. Keep each multiply/add materialization boundary despite the
    // translation unit's fast-math flags so CPU and device order is explicit.
    volatile double product = left * right;
    volatile double sum = product + accumulator;
    return sum;
}

KOKKOS_FUNCTION double realspace_sqrt(const double value)
{
    return Kokkos::sqrt(value);
}

KOKKOS_FUNCTION double realspace_displaced_difference(
    const double source_position,
    const double source_offset,
    const double receiver_position,
    const double receiver_offset)
{
    // The exact coincident replica is the only hazardous case: an O(ulp)
    // reassociation residual is amplified by the singular r -> 0 derivative.
    // Keep the upstream four-term spelling everywhere else so near-pair FP64
    // semantics are unchanged.
    if (source_position == receiver_position
        && source_offset == receiver_offset)
        return 0.0;
    return source_position + source_offset - receiver_position - receiver_offset;
}

KOKKOS_FUNCTION double realspace_kernel(
    const double distance, const double width)
{
    constexpr double epsilon = 1.0e-6;
    return realspace_erf(0.5 * distance / width) / (distance + epsilon);
}

// Return (dK/dr)/r so multiplying by one Cartesian displacement component
// gives grad_d K. The explicit zero branch is the pinned Torch norm
// convention for exactly coincident distinct replicas.
KOKKOS_FUNCTION double realspace_kernel_gradient_factor(
    const double distance, const double width)
{
    if (distance == 0.0)
        return 0.0;
    constexpr double epsilon = 1.0e-6;
    constexpr double sqrt_pi = 1.7724538509055160273;
    const double scaled = 0.5 * distance / width;
    const double shifted = distance + epsilon;
    const double derivative =
        (Kokkos::exp(-scaled * scaled) * shifted / (width * sqrt_pi)
            - realspace_erf(scaled))
        / (shifted * shifted);
    return derivative / distance;
}

KOKKOS_FUNCTION double realspace_replica_transform_entry(
    const int max_l,
    const std::size_t replica,
    const std::size_t multipole,
    const double displacement)
{
    if (max_l == 0)
        return replica == 0 && multipole == 0 ? 1.0 : 0.0;
    if (replica == 0)
        return multipole == 0 ? 1.0 : -1.0 / displacement;
    if (replica == 1)
        return multipole == 3 ? 1.0 / displacement : 0.0;
    if (replica == 2)
        return multipole == 1 ? 1.0 / displacement : 0.0;
    return multipole == 2 ? 1.0 / displacement : 0.0;
}

template <typename MultipoleView>
KOKKOS_FUNCTION double realspace_replica_charge(
    const MultipoleView& multipoles,
    const std::size_t node,
    const std::size_t replica,
    const int max_l,
    const double displacement)
{
    if (max_l == 0)
        return static_cast<double>(multipoles(node, 0));
    const double qx =
        static_cast<double>(multipoles(node, 3)) / displacement;
    const double qy =
        static_cast<double>(multipoles(node, 1)) / displacement;
    const double qz =
        static_cast<double>(multipoles(node, 2)) / displacement;
    if (replica == 0)
        return static_cast<double>(multipoles(node, 0)) - qx - qy - qz;
    if (replica == 1)
        return qx;
    if (replica == 2)
        return qy;
    return qz;
}

// e3nn integral-normalized real spherical harmonics for l <= 1: Y_0 =
// 1/sqrt(4 pi) and Y_1 = sqrt(3/(4 pi)) * (y, z, x)/|k| (the basis permutes
// cartesian (x, y, z) into the e3nn m = -1, 0, +1 phase order, and the
// upstream module normalizes its input direction).
inline double harmonic_l0() { return 0.5 / std::sqrt(pi); }
inline double harmonic_l1() { return std::sqrt(3.0 / (4.0 * pi)); }

// The column of the density's e3nn (y, z, x) l=1 block that carries one
// cartesian component: x -> 3, y -> 1, z -> 2, the order the multipole
// gathers and the dipole readout both use.
inline std::size_t dipole_column(const int cartesian)
{
    static const std::size_t columns[3] = {3, 1, 2};
    return columns[cartesian];
}

// The reciprocal cell 2 pi inv(cell)^T through the cofactor inverse -- exact
// enough for the well-conditioned cells the layer serves, and identical to
// the oracle's LU inverse to reassociation noise.  The minors keep the
// remaining rows and columns in ascending order: the cyclic (i+1, i+2)
// spelling flips the 2x2 determinant sign wherever exactly one of the two
// index pairs runs "backwards", which corrupts rcell for triclinic cells
// (the diagonal fixtures never see it).  With ascending orders the cofactor
// matrix is the adjugate's transpose and rcell[i][j] = 2 pi C_ij/det is
// exactly 2 pi inv(A)^T, so k.r stays 2 pi n.frac under any deformation.
void reciprocal_cell(const double* cell, double* rcell)
{
    double cofactors[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // The rows and columns remaining after deleting row i and
            // column j, in ascending order.
            const int i0 = i == 0 ? 1 : 0;
            const int i1 = i == 2 ? 1 : 2;
            const int j0 = j == 0 ? 1 : 0;
            const int j1 = j == 2 ? 1 : 2;
            cofactors[i][j] = ((i + j) % 2 == 0 ? 1.0 : -1.0)
                * (cell[3 * i0 + j0] * cell[3 * i1 + j1]
                    - cell[3 * i0 + j1] * cell[3 * i1 + j0]);
        }
    }
    double determinant = 0.0;
    for (int j = 0; j < 3; ++j)
        determinant += cell[j] * cofactors[0][j];
    const double scale = 2.0 * pi / determinant;
    // inv(cell)^T: rcell[i][j] = 2 pi * cofactors[i][j]/det.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            rcell[3 * i + j] = cofactors[i][j] * scale;
}

double cell_determinant(const double* cell)
{
    return cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
}

// C_l of the two basis normalizations, exactly the oracle's get_Cl_sigma.
double cl_sigma(const int l, const double sigma, const bool receiver)
{
    if (receiver)
        return 1.0 / (std::pow(2.0, (l + 1) / 2.0) * std::tgamma((l + 3) / 2.0)
            * std::pow(sigma, l + 3));
    return 1.0 / (std::sqrt(4.0 * pi / (2 * l + 1))
        * std::pow(2.0, (2 * l + 1) / 2.0) * std::tgamma((2 * l + 3) / 2.0)
        * std::pow(sigma, 2 * l + 3));
}

// Regularized lower incomplete gamma P(l + 3/2, x) for l in {0, 1} in terms
// of erf -- the closed forms of the scipy gammainc the oracle's integral_F1
// evaluates (its second gammainc term carries gammainc(a, 0) = 0).
double gammainc_3_2(const double x)
{
    return std::erf(std::sqrt(x))
        - 2.0 * std::sqrt(x / pi) * std::exp(-x);
}

double gammainc_5_2(const double x)
{
    const double root_pi = std::sqrt(pi);
    const double root_x = std::sqrt(x);
    const double e = std::exp(-x);
    const double g_3_2 = 0.5 * root_pi * std::erf(root_x) - root_x * e;
    const double g_5_2 = 1.5 * g_3_2 - x * root_x * e;
    return g_5_2 / (0.75 * root_pi);
}

// (2 pi)^3 and (2 pi)^6 with the upstream evaluation order.
inline double two_pi_cubed() { return (2.0 * pi) * (2.0 * pi) * (2.0 * pi); }

// The staging layer between the kind group's host-typed public I/O and its
// kernel-space tables and workspaces: uploads copy host buffers into kernel
// views, downloads copy kernel views back into host views.  The helpers
// accept any host-space view (managed or unmanaged, const or not) and use
// Kokkos's own mirror machinery, so on a host-only build the two spaces
// coincide, the mirror aliases the caller's buffer, and no byte moves: the
// host behavior is unchanged bit for bit.
template <typename T, typename HostT>
Kokkos::View<T*, Kokkos::LayoutRight, LongRangeKernelMemory>
stage_vector(
    const Kokkos::View<const HostT*, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>& host,
    const char* label)
{
    Kokkos::View<T*, Kokkos::LayoutRight, LongRangeKernelMemory> view(
        Kokkos::view_alloc(std::string(label)), host.extent(0));
    auto mirror = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < host.extent(0); ++i)
        mirror(i) = static_cast<T>(host(i));
    Kokkos::deep_copy(view, mirror);
    return view;
}

template <typename T, typename HostT>
Kokkos::View<T*, Kokkos::LayoutRight, LongRangeKernelMemory>
stage_vector(
    const Kokkos::View<HostT*, Kokkos::LayoutRight, Kokkos::HostSpace>& host,
    const char* label)
{
    Kokkos::View<T*, Kokkos::LayoutRight, LongRangeKernelMemory> view(
        Kokkos::view_alloc(std::string(label)), host.extent(0));
    auto mirror = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < host.extent(0); ++i)
        mirror(i) = static_cast<T>(host(i));
    Kokkos::deep_copy(view, mirror);
    return view;
}

template <typename T, typename HostT>
Kokkos::View<T**, Kokkos::LayoutRight, LongRangeKernelMemory>
stage_matrix(
    const Kokkos::View<const HostT**, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>& host,
    const char* label)
{
    Kokkos::View<T**, Kokkos::LayoutRight, LongRangeKernelMemory> view(
        Kokkos::view_alloc(std::string(label)), host.extent(0), host.extent(1));
    auto mirror = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < host.extent(0); ++i)
        for (std::size_t j = 0; j < host.extent(1); ++j)
            mirror(i, j) = static_cast<T>(host(i, j));
    Kokkos::deep_copy(view, mirror);
    return view;
}

template <typename T, typename HostT>
void stage_vector_into(
    const Kokkos::View<const HostT*, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>& host,
    const Kokkos::View<T*, Kokkos::LayoutRight, LongRangeKernelMemory>& view)
{
    if (view.extent(0) != host.extent(0))
        throw std::invalid_argument("long-range vector staging extent mismatch.");
    auto mirror = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < host.extent(0); ++i)
        mirror(i) = static_cast<T>(host(i));
    Kokkos::deep_copy(view, mirror);
}

template <typename T, typename HostT>
void stage_matrix_into(
    const Kokkos::View<const HostT**, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>& host,
    const Kokkos::View<T**, Kokkos::LayoutRight, LongRangeKernelMemory>& view)
{
    if (view.extent(0) != host.extent(0) || view.extent(1) != host.extent(1))
        throw std::invalid_argument("long-range matrix staging extent mismatch.");
    auto mirror = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < host.extent(0); ++i)
        for (std::size_t j = 0; j < host.extent(1); ++j)
            mirror(i, j) = static_cast<T>(host(i, j));
    Kokkos::deep_copy(view, mirror);
}

template <typename T, typename HostT>
Kokkos::View<T**, Kokkos::LayoutRight, LongRangeKernelMemory>
stage_matrix(
    const Kokkos::View<HostT**, Kokkos::LayoutRight, Kokkos::HostSpace>& host,
    const char* label)
{
    Kokkos::View<T**, Kokkos::LayoutRight, LongRangeKernelMemory> view(
        Kokkos::view_alloc(std::string(label)), host.extent(0), host.extent(1));
    auto mirror = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < host.extent(0); ++i)
        for (std::size_t j = 0; j < host.extent(1); ++j)
            mirror(i, j) = static_cast<T>(host(i, j));
    Kokkos::deep_copy(view, mirror);
    return view;
}

template <typename HostT, typename T>
Kokkos::View<HostT*, Kokkos::LayoutRight, Kokkos::HostSpace>
download_vector(
    const Kokkos::View<T*, Kokkos::LayoutRight, LongRangeKernelMemory>&
        kernel,
    const char* label)
{
    Kokkos::View<HostT*, Kokkos::LayoutRight, Kokkos::HostSpace> view(
        Kokkos::view_alloc(std::string(label)), kernel.extent(0));
    const auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), kernel);
    for (std::size_t i = 0; i < kernel.extent(0); ++i)
        view(i) = static_cast<HostT>(mirror(i));
    return view;
}

template <typename HostT, typename T>
Kokkos::View<HostT**, Kokkos::LayoutRight, Kokkos::HostSpace>
download_matrix(
    const Kokkos::View<T**, Kokkos::LayoutRight, LongRangeKernelMemory>&
        kernel,
    const char* label)
{
    Kokkos::View<HostT**, Kokkos::LayoutRight, Kokkos::HostSpace> view(
        Kokkos::view_alloc(std::string(label)), kernel.extent(0),
        kernel.extent(1));
    const auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), kernel);
    for (std::size_t i = 0; i < kernel.extent(0); ++i)
        for (std::size_t j = 0; j < kernel.extent(1); ++j)
            view(i, j) = static_cast<HostT>(mirror(i, j));
    return view;
}

template <typename HostT, typename T>
Kokkos::View<HostT***, Kokkos::LayoutRight, Kokkos::HostSpace>
download_tensor3(
    const Kokkos::View<T***, Kokkos::LayoutRight, LongRangeKernelMemory>&
        kernel,
    const char* label)
{
    Kokkos::View<HostT***, Kokkos::LayoutRight, Kokkos::HostSpace> view(
        Kokkos::view_alloc(std::string(label)), kernel.extent(0),
        kernel.extent(1), kernel.extent(2));
    const auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), kernel);
    for (std::size_t i = 0; i < kernel.extent(0); ++i)
        for (std::size_t j = 0; j < kernel.extent(1); ++j)
            for (std::size_t k = 0; k < kernel.extent(2); ++k)
                view(i, j, k) = static_cast<HostT>(mirror(i, j, k));
    return view;
}

template <typename T>
void accumulate_kernel_matrix(
    const LongRangeKernelMatrix<T>& target,
    const LongRangeKernelMatrix<T>& contribution,
    const char* what)
{
    if (target.extent(0) != contribution.extent(0)
        || target.extent(1) != contribution.extent(1))
        throw std::invalid_argument(std::string(what)
            + ": the device accumulator shape does not match the table seed.");
    const std::size_t size = target.size();
    auto target_view = target;
    auto contribution_view = contribution;
    Kokkos::parallel_for(what,
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, size),
        KOKKOS_LAMBDA(const std::size_t index) {
            target_view.data()[index] += contribution_view.data()[index];
        });
}

template <typename T>
void accumulate_kernel_vector(
    const LongRangeKernelVector<T>& target,
    const LongRangeKernelVector<T>& contribution,
    const char* what)
{
    if (target.extent(0) != contribution.extent(0))
        throw std::invalid_argument(std::string(what)
            + ": the device accumulator shape does not match the seed.");
    const std::size_t size = target.size();
    auto target_view = target;
    auto contribution_view = contribution;
    Kokkos::parallel_for(what,
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, size),
        KOKKOS_LAMBDA(const std::size_t index) {
            target_view(index) += contribution_view(index);
        });
}

} // namespace

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::set_realspace_reverse_strategy(
    const std::string_view strategy)
{
    if (strategy != "materialized_forward")
        release_realspace_full_materialized_tables();
    if (strategy == "auto") {
        realspace_reverse_strategy_request_ =
            RealspaceReverseStrategyRequest::automatic;
        realspace_reverse_strategy_ = RealspaceReverseStrategy::owner;
        realspace_reverse_strategy_resolved_ = false;
        realspace_reverse_strategy_reason_ = "reverse_not_requested";
    } else if (strategy == "owner") {
        realspace_reverse_strategy_request_ =
            RealspaceReverseStrategyRequest::owner;
        realspace_reverse_strategy_ = RealspaceReverseStrategy::owner;
        realspace_reverse_strategy_resolved_ = true;
        realspace_reverse_strategy_reason_ = "forced_override";
        realspace_reverse_launch_diagnostics_.atomic_accumulator_bytes = 0;
    } else if (strategy == "pair_atomic") {
        realspace_reverse_strategy_request_ =
            RealspaceReverseStrategyRequest::pair_atomic;
        realspace_reverse_strategy_ = RealspaceReverseStrategy::pair_atomic;
        realspace_reverse_strategy_resolved_ = true;
        realspace_reverse_strategy_reason_ = "forced_override";
    } else if (strategy == "materialized_forward") {
        if (realspace_full_materialized_reducer_
            != RealspaceFullMaterializedReducer::segmented)
            release_realspace_full_materialized_tables();
        realspace_full_materialized_reducer_ =
            RealspaceFullMaterializedReducer::segmented;
        realspace_reverse_strategy_request_ =
            RealspaceReverseStrategyRequest::materialized_forward;
        realspace_reverse_strategy_ =
            RealspaceReverseStrategy::materialized_forward;
        realspace_reverse_strategy_resolved_ = true;
        realspace_reverse_strategy_reason_ = "forced_override";
    } else
        throw std::invalid_argument(
            "real-space reverse strategy must be 'auto', 'owner', "
            "'pair_atomic', or 'materialized_forward'.");
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::set_realspace_full_materialized_reducer(
    const std::string_view reducer)
{
    RealspaceFullMaterializedReducer selected;
    if (reducer == "serial")
        selected = RealspaceFullMaterializedReducer::serial;
    else if (reducer == "atomic_scatter")
        selected = RealspaceFullMaterializedReducer::atomic_scatter;
    else if (reducer == "segmented")
        selected = RealspaceFullMaterializedReducer::segmented;
    else
        throw std::invalid_argument(
            "full-materialized reducer must be 'serial', 'atomic_scatter', "
            "or 'segmented'.");
    if (selected != realspace_full_materialized_reducer_)
        release_realspace_full_materialized_tables();
    realspace_full_materialized_reducer_ = selected;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::set_realspace_pair_tile_width(
    const std::size_t width)
{
    if (width != 8 && width != 16 && width != 32 && width != 64)
        throw std::invalid_argument(
            "real-space pair tile width must be 8, 16, 32, or 64.");
    realspace_pair_tile_width_ = width;
    realspace_reverse_launch_diagnostics_.tile_width = width;
}

template <typename Scalar>
std::size_t PolarLongRangeOpsT<Scalar>::
checked_realspace_pair_tile_scratch_bytes() const
{
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (density_width_ > maximum - 3)
        throw std::overflow_error(
            "real-space pair tile scratch width overflows size_t.");
    const std::size_t lane_values = density_width_ + 3;
    if (lane_values != 0
        && realspace_pair_tile_width_ > maximum / lane_values)
        throw std::overflow_error(
            "real-space pair tile scratch elements overflow size_t.");
    const std::size_t elements = realspace_pair_tile_width_ * lane_values;
    if (elements > maximum / sizeof(double))
        throw std::overflow_error(
            "real-space pair tile scratch bytes overflow size_t.");
    return elements * sizeof(double);
}

template <typename Scalar>
std::size_t PolarLongRangeOpsT<Scalar>::
checked_realspace_pair_tile_feature_league_extent(
    const std::size_t n_nodes) const
{
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t tile_width = realspace_pair_tile_width_;
    const std::size_t tiles = n_nodes / tile_width
        + static_cast<std::size_t>(n_nodes % tile_width != 0);
    if (tiles != 0 && n_nodes > maximum / tiles)
        throw std::overflow_error(
            "real-space tiled feature league overflows size_t.");
    const std::size_t pairs = n_nodes * tiles;
    if (feature_smearing_widths_.size() != 0
        && pairs > maximum / feature_smearing_widths_.size())
        throw std::overflow_error(
            "real-space tiled feature league overflows size_t.");
    return pairs * feature_smearing_widths_.size();
}

template <typename Scalar>
std::size_t PolarLongRangeOpsT<Scalar>::
checked_realspace_pair_tile_energy_league_extent(
    const std::size_t n_nodes) const
{
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t tile_width = realspace_pair_tile_width_;
    const std::size_t tiles = n_nodes / tile_width
        + static_cast<std::size_t>(n_nodes % tile_width != 0);
    if (tiles != 0 && n_nodes > maximum / tiles)
        throw std::overflow_error(
            "real-space tiled energy league overflows size_t.");
    return n_nodes * tiles;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::validate_realspace_pair_tile_backend() const
{
    const std::size_t backend_capacity = static_cast<std::size_t>(
        LongRangeKernelSpace().concurrency());
    if (realspace_pair_tile_width_ > backend_capacity)
        throw std::invalid_argument("real-space pair tile team width "
            + std::to_string(realspace_pair_tile_width_)
            + " exceeds backend team capacity "
            + std::to_string(backend_capacity) + ".");
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::set_realspace_materialized_budget_bytes(
    const std::size_t budget_bytes)
{
    realspace_materialized_budget_bytes_ = budget_bytes;
    if (realspace_reverse_strategy_request_
        == RealspaceReverseStrategyRequest::materialized_forward) {
        realspace_reverse_strategy_ =
            RealspaceReverseStrategy::materialized_forward;
        realspace_reverse_strategy_reason_ = "forced_override";
    }
}

template <typename Scalar>
std::size_t PolarLongRangeOpsT<Scalar>::
checked_realspace_full_materialized_bytes(
    const std::size_t n_nodes, const bool include_gradients) const
{
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const auto multiply = [maximum] (const std::size_t left,
                              const std::size_t right) {
        if (right != 0 && left > maximum / right)
            throw std::overflow_error(
                "real-space full materialized byte count overflows size_t.");
        return left * right;
    };
    const auto add = [maximum] (const std::size_t left,
                         const std::size_t right) {
        if (left > maximum - right)
            throw std::overflow_error(
                "real-space full materialized byte count overflows size_t.");
        return left + right;
    };
    const std::size_t directed_atoms = n_nodes == 0
        ? 0
        : multiply(n_nodes, n_nodes - 1);
    const std::size_t source_replicas = density_max_l_ == 0 ? 1 : 4;
    const std::size_t receiver_replicas = feature_max_l_ == 0 ? 1 : 4;
    const std::size_t feature_edges = multiply(directed_atoms,
        multiply(source_replicas, receiver_replicas));
    const std::size_t energy_edges = multiply(directed_atoms,
        multiply(source_replicas, source_replicas));
    const std::size_t widths = feature_smearing_widths_.size();
    // Each edge owns uint64 endpoints, validity, FP64 displacement, norm, and
    // K plus optional G. Only the serial diagnostic baseline retains forward
    // contributions; parallel reducers recompute them from charges and K.
    const bool retain_contributions = realspace_full_materialized_reducer_
        == RealspaceFullMaterializedReducer::serial;
    std::size_t feature_edge_bytes = 2 * sizeof(std::uint64_t)
        + sizeof(std::uint8_t) + 4 * sizeof(double);
    feature_edge_bytes = add(feature_edge_bytes,
        multiply(widths,
            (1 + static_cast<std::size_t>(include_gradients)
                + static_cast<std::size_t>(retain_contributions))
                * sizeof(double)));
    const std::size_t energy_edge_bytes = 2 * sizeof(std::uint64_t)
        + sizeof(std::uint8_t) + 4 * sizeof(double)
        + (1 + static_cast<std::size_t>(include_gradients)
              + static_cast<std::size_t>(retain_contributions))
            * sizeof(double);
    std::size_t bytes = add(multiply(feature_edges, feature_edge_bytes),
        multiply(energy_edges, energy_edge_bytes));
    const std::size_t feature_replicas = multiply(
        n_nodes, std::max(source_replicas, receiver_replicas));
    const std::size_t energy_replicas = feature_replicas;
    bytes = add(bytes, multiply(feature_replicas,
        3 * sizeof(double) + sizeof(double)));
    bytes = add(bytes, multiply(energy_replicas,
        3 * sizeof(double) + sizeof(double)));
    // Admission reserve: allocator alignment and scan/reduction workspace.
    bytes = add(bytes, bytes / 8);
    return bytes;
}

template <typename Scalar>
std::size_t PolarLongRangeOpsT<Scalar>::
checked_realspace_materialized_table_bytes(const std::size_t n_nodes) const
{
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (n_nodes > 0 && n_nodes - 1 > maximum / n_nodes)
        throw std::overflow_error(
            "real-space materialized radial pair count overflows size_t.");
    const std::size_t pair_count = n_nodes * (n_nodes - (n_nodes != 0)) / 2;
    const std::size_t source_replicas = density_max_l_ == 0 ? 1 : 4;
    const std::size_t receiver_replicas = feature_max_l_ == 0 ? 1 : 4;
    const std::size_t feature_position_replicas =
        std::max(source_replicas, receiver_replicas);
    const std::size_t energy_replicas = source_replicas;
    std::size_t records_per_pair = 2;
    const auto checked_multiply = [maximum] (const std::size_t left,
                                      const std::size_t right) {
        if (right != 0 && left > maximum / right)
            throw std::overflow_error(
                "real-space materialized radial table size overflows size_t.");
        return left * right;
    };
    records_per_pair = checked_multiply(
        records_per_pair, feature_smearing_widths_.size());
    records_per_pair = checked_multiply(records_per_pair, source_replicas);
    records_per_pair = checked_multiply(records_per_pair, receiver_replicas);
    const std::size_t energy_records =
        checked_multiply(energy_replicas, energy_replicas);
    if (records_per_pair > maximum - energy_records)
        throw std::overflow_error(
            "real-space materialized radial record count overflows size_t.");
    const std::size_t records = checked_multiply(
        pair_count, records_per_pair + energy_records);
    const std::size_t bytes_per_record = 2 * sizeof(double);
    return checked_multiply(records, bytes_per_record);
}

template <typename Scalar>
bool PolarLongRangeOpsT<Scalar>::admit_realspace_materialized(
    const std::size_t n_nodes) const
{
    const std::size_t bytes =
        checked_realspace_materialized_table_bytes(n_nodes);
    if (bytes <= realspace_materialized_budget_bytes_)
        return true;
    materialized_feature_k_ = LongRangeKernelVector<double>{};
    materialized_feature_g_ = LongRangeKernelVector<double>{};
    materialized_energy_k_ = LongRangeKernelVector<double>{};
    materialized_energy_g_ = LongRangeKernelVector<double>{};
    materialized_table_revision_ = std::numeric_limits<std::size_t>::max();
    realspace_reverse_launch_diagnostics_.materialized_radial_entries = 0;
    realspace_reverse_launch_diagnostics_.materialized_radial_bytes = 0;
    realspace_reverse_strategy_ = RealspaceReverseStrategy::pair_atomic;
    realspace_reverse_strategy_reason_ = "materialized_budget_rejected";
    ++realspace_reverse_launch_diagnostics_
          .materialized_admission_rejection_count;
    return false;
}

template <typename Scalar>
bool PolarLongRangeOpsT<Scalar>::admit_realspace_full_materialized(
    const std::size_t n_nodes, const bool include_gradients) const
{
    const std::size_t bytes =
        checked_realspace_full_materialized_bytes(n_nodes, include_gradients);
    realspace_reverse_launch_diagnostics_.full_materialized_admission_bytes =
        bytes;
    if (bytes <= realspace_materialized_budget_bytes_)
        return true;
    release_realspace_full_materialized_tables();
    realspace_reverse_strategy_ = RealspaceReverseStrategy::pair_atomic;
    realspace_reverse_strategy_reason_ = "full_materialized_budget_rejected";
    ++realspace_reverse_launch_diagnostics_
          .full_materialized_admission_rejection_count;
    return false;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::
release_realspace_full_materialized_tables() const
{
    full_feature_sender_ = {};
    full_feature_receiver_ = {};
    full_feature_valid_ = {};
    full_feature_difference_ = {};
    full_feature_norm_ = {};
    full_feature_k_ = {};
    full_feature_g_ = {};
    full_feature_contribution_ = {};
    full_feature_replica_positions_ = {};
    full_feature_replica_charges_ = {};
    full_energy_sender_ = {};
    full_energy_receiver_ = {};
    full_energy_valid_ = {};
    full_energy_difference_ = {};
    full_energy_norm_ = {};
    full_energy_k_ = {};
    full_energy_g_ = {};
    full_energy_contribution_ = {};
    full_energy_replica_positions_ = {};
    full_energy_replica_charges_ = {};
    full_materialized_table_revision_ = std::numeric_limits<std::size_t>::max();
    full_materialized_has_gradients_ = false;
    realspace_reverse_launch_diagnostics_.full_materialized_retained_bytes = 0;
    realspace_reverse_launch_diagnostics_.full_materialized_has_gradients = false;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::ensure_realspace_full_materialized_tables(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const bool include_gradients) const
{
    if (full_materialized_table_revision_ == realspace_geometry_revision_
        && (!include_gradients || full_materialized_has_gradients_)) {
        ++realspace_reverse_launch_diagnostics_
              .full_materialized_table_reuse_count;
        return;
    }
    if (full_materialized_table_revision_
        != std::numeric_limits<std::size_t>::max())
        ++realspace_reverse_launch_diagnostics_
              .full_materialized_table_invalidation_count;

    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t source_replicas = density_max_l_ == 0 ? 1 : 4;
    const std::size_t receiver_replicas = feature_max_l_ == 0 ? 1 : 4;
    const std::size_t feature_position_replicas =
        std::max(source_replicas, receiver_replicas);
    const std::size_t n_sigma = feature_smearing_widths_.size();
    const std::size_t directed_atoms = n_nodes * (n_nodes - (n_nodes != 0));
    const std::size_t feature_edges =
        directed_atoms * source_replicas * receiver_replicas;
    const std::size_t energy_edges =
        directed_atoms * source_replicas * source_replicas;
    try {
        full_feature_sender_ = decltype(full_feature_sender_)(
            "long_range_full_feature_sender", feature_edges);
        full_feature_receiver_ = decltype(full_feature_receiver_)(
            "long_range_full_feature_receiver", feature_edges);
        full_feature_valid_ = decltype(full_feature_valid_)(
            "long_range_full_feature_valid", feature_edges);
        full_feature_difference_ = decltype(full_feature_difference_)(
            "long_range_full_feature_difference", feature_edges, 3);
        full_feature_norm_ = decltype(full_feature_norm_)(
            "long_range_full_feature_norm", feature_edges);
        full_feature_k_ = decltype(full_feature_k_)(
            "long_range_full_feature_k", feature_edges, n_sigma);
        full_feature_g_ = include_gradients
            ? LongRangeKernelMatrix<double>(
                  "long_range_full_feature_g", feature_edges, n_sigma)
            : LongRangeKernelMatrix<double>{};
        full_feature_contribution_ = realspace_full_materialized_reducer_
                == RealspaceFullMaterializedReducer::serial
            ? LongRangeKernelMatrix<double>(
                  "long_range_full_feature_contribution", feature_edges, n_sigma)
            : LongRangeKernelMatrix<double>{};
        full_feature_replica_positions_ = decltype(
            full_feature_replica_positions_)("long_range_full_feature_replicas",
            n_nodes * feature_position_replicas, 3);
        full_feature_replica_charges_ = decltype(
            full_feature_replica_charges_)("long_range_full_feature_charges",
            n_nodes * source_replicas);
        full_energy_sender_ = decltype(full_energy_sender_)(
            "long_range_full_energy_sender", energy_edges);
        full_energy_receiver_ = decltype(full_energy_receiver_)(
            "long_range_full_energy_receiver", energy_edges);
        full_energy_valid_ = decltype(full_energy_valid_)(
            "long_range_full_energy_valid", energy_edges);
        full_energy_difference_ = decltype(full_energy_difference_)(
            "long_range_full_energy_difference", energy_edges, 3);
        full_energy_norm_ = decltype(full_energy_norm_)(
            "long_range_full_energy_norm", energy_edges);
        full_energy_k_ = decltype(full_energy_k_)(
            "long_range_full_energy_k", energy_edges);
        full_energy_g_ = include_gradients
            ? LongRangeKernelVector<double>(
                  "long_range_full_energy_g", energy_edges)
            : LongRangeKernelVector<double>{};
        full_energy_contribution_ = realspace_full_materialized_reducer_
                == RealspaceFullMaterializedReducer::serial
            ? LongRangeKernelVector<double>(
                  "long_range_full_energy_contribution", energy_edges)
            : LongRangeKernelVector<double>{};
        full_energy_replica_positions_ = decltype(
            full_energy_replica_positions_)("long_range_full_energy_replicas",
            n_nodes * source_replicas, 3);
        full_energy_replica_charges_ = decltype(
            full_energy_replica_charges_)("long_range_full_energy_charges",
            n_nodes * source_replicas);
    } catch (const std::exception&) {
        release_realspace_full_materialized_tables();
        realspace_reverse_strategy_ = RealspaceReverseStrategy::pair_atomic;
        realspace_reverse_strategy_reason_ = "full_materialized_allocation_failed";
        ++realspace_reverse_launch_diagnostics_
              .full_materialized_admission_rejection_count;
        realspace_reverse_launch_diagnostics_.full_materialized_retained_bytes = 0;
        return;
    }

    auto positions = cache.realspace_positions;
    auto node_graph = cache.node_graph_index;
    auto effective_widths = realspace_effective_widths_;
    auto feature_replica_positions = full_feature_replica_positions_;
    auto energy_replica_positions = full_energy_replica_positions_;
    constexpr double feature_displacement = 0.1;
    constexpr double energy_displacement = 0.02;
    Kokkos::parallel_for("long_range full materialized feature replica positions",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(
            0, n_nodes * feature_position_replicas * 3),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t cartesian = flat % 3;
            const std::size_t replica_flat = flat / 3;
            const std::size_t replica =
                replica_flat % feature_position_replicas;
            const std::size_t node = replica_flat / feature_position_replicas;
            const double offset = replica > 0 && cartesian == replica - 1
                ? 1.0 : 0.0;
            feature_replica_positions(replica_flat, cartesian) =
                positions(node, cartesian) + offset * feature_displacement;
        });
    Kokkos::parallel_for("long_range full materialized energy replica positions",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes * source_replicas * 3),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t cartesian = flat % 3;
            const std::size_t replica_flat = flat / 3;
            const std::size_t replica = replica_flat % source_replicas;
            const std::size_t node = replica_flat / source_replicas;
            const double offset = replica > 0 && cartesian == replica - 1
                ? 1.0 : 0.0;
            energy_replica_positions(replica_flat, cartesian) =
                positions(node, cartesian) + offset * energy_displacement;
        });

    auto feature_sender = full_feature_sender_;
    auto feature_receiver = full_feature_receiver_;
    auto feature_valid = full_feature_valid_;
    auto feature_difference = full_feature_difference_;
    auto feature_norm = full_feature_norm_;
    auto feature_k = full_feature_k_;
    auto feature_g = full_feature_g_;
    Kokkos::parallel_for("long_range full materialized feature edges",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, feature_edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            std::size_t value = edge;
            const std::size_t source_replica = value % source_replicas;
            value /= source_replicas;
            std::size_t source = value % (n_nodes - 1);
            value /= n_nodes - 1;
            const std::size_t receiver_replica = value % receiver_replicas;
            const std::size_t receiver = value / receiver_replicas;
            if (source >= receiver)
                ++source;
            feature_sender(edge) = static_cast<std::uint64_t>(
                source * source_replicas + source_replica);
            feature_receiver(edge) = static_cast<std::uint64_t>(
                receiver * receiver_replicas + receiver_replica);
            const bool valid = node_graph(source) == node_graph(receiver);
            feature_valid(edge) = static_cast<std::uint8_t>(valid);
            double distance_squared = 0.0;
            for (std::size_t cartesian = 0; cartesian < 3; ++cartesian) {
                const double difference = realspace_displaced_difference(
                    positions(source, cartesian),
                    source_replica > 0 && cartesian == source_replica - 1
                        ? feature_displacement : 0.0,
                    positions(receiver, cartesian),
                    receiver_replica > 0 && cartesian == receiver_replica - 1
                        ? feature_displacement : 0.0);
                feature_difference(edge, cartesian) = difference;
                distance_squared = realspace_norm_accumulate(
                    difference, difference, distance_squared);
            }
            const double distance = realspace_sqrt(distance_squared);
            feature_norm(edge) = distance;
            for (std::size_t sigma = 0; sigma < n_sigma; ++sigma) {
                feature_k(edge, sigma) = valid
                    ? realspace_kernel(distance, effective_widths(sigma)) : 0.0;
                if (include_gradients)
                    feature_g(edge, sigma) = valid
                        ? realspace_kernel_gradient_factor(
                              distance, effective_widths(sigma)) : 0.0;
            }
        });

    auto energy_sender = full_energy_sender_;
    auto energy_receiver = full_energy_receiver_;
    auto energy_valid = full_energy_valid_;
    auto energy_difference = full_energy_difference_;
    auto energy_norm = full_energy_norm_;
    auto energy_k = full_energy_k_;
    auto energy_g = full_energy_g_;
    const double density_width = density_smearing_width_;
    Kokkos::parallel_for("long_range full materialized energy edges",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, energy_edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            std::size_t value = edge;
            const std::size_t receiver_replica = value % source_replicas;
            value /= source_replicas;
            const std::size_t source_replica = value % source_replicas;
            const std::size_t directed = value / source_replicas;
            const std::size_t source = directed / (n_nodes - 1);
            std::size_t receiver = directed - source * (n_nodes - 1);
            if (receiver >= source)
                ++receiver;
            energy_sender(edge) = static_cast<std::uint64_t>(
                source * source_replicas + source_replica);
            energy_receiver(edge) = static_cast<std::uint64_t>(
                receiver * source_replicas + receiver_replica);
            const bool valid = node_graph(source) == node_graph(receiver);
            energy_valid(edge) = static_cast<std::uint8_t>(valid);
            double distance_squared = 0.0;
            for (std::size_t cartesian = 0; cartesian < 3; ++cartesian) {
                const double difference = realspace_displaced_difference(
                    positions(source, cartesian),
                    source_replica > 0 && cartesian == source_replica - 1
                        ? energy_displacement : 0.0,
                    positions(receiver, cartesian),
                    receiver_replica > 0 && cartesian == receiver_replica - 1
                        ? energy_displacement : 0.0);
                energy_difference(edge, cartesian) = difference;
                distance_squared = realspace_norm_accumulate(
                    difference, difference, distance_squared);
            }
            const double distance = realspace_sqrt(distance_squared);
            energy_norm(edge) = distance;
            energy_k(edge) = valid
                ? realspace_kernel(distance, density_width) : 0.0;
            if (include_gradients)
                energy_g(edge) = valid
                    ? realspace_kernel_gradient_factor(distance, density_width)
                    : 0.0;
        });
    full_materialized_table_revision_ = realspace_geometry_revision_;
    full_materialized_has_gradients_ = include_gradients;
    realspace_reverse_launch_diagnostics_.full_materialized_has_gradients =
        include_gradients;
    ++realspace_reverse_launch_diagnostics_.full_materialized_table_build_count;
    realspace_reverse_launch_diagnostics_.full_materialized_feature_directed_edges =
        feature_edges;
    realspace_reverse_launch_diagnostics_.full_materialized_energy_directed_edges =
        energy_edges;
    std::size_t retained_bytes = 0;
    const auto add_retained = [&retained_bytes] (const auto& view) {
        retained_bytes += static_cast<std::size_t>(view.span())
            * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    add_retained(full_feature_sender_);
    add_retained(full_feature_receiver_);
    add_retained(full_feature_valid_);
    add_retained(full_feature_difference_);
    add_retained(full_feature_norm_);
    add_retained(full_feature_k_);
    add_retained(full_feature_g_);
    add_retained(full_feature_contribution_);
    add_retained(full_feature_replica_positions_);
    add_retained(full_feature_replica_charges_);
    add_retained(full_energy_sender_);
    add_retained(full_energy_receiver_);
    add_retained(full_energy_valid_);
    add_retained(full_energy_difference_);
    add_retained(full_energy_norm_);
    add_retained(full_energy_k_);
    add_retained(full_energy_g_);
    add_retained(full_energy_contribution_);
    add_retained(full_energy_replica_positions_);
    add_retained(full_energy_replica_charges_);
    realspace_reverse_launch_diagnostics_.full_materialized_retained_bytes =
        retained_bytes;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::ensure_realspace_materialized_tables(
    const LongRangeGeometryCacheT<Scalar>& cache) const
{
    if (materialized_table_revision_ == realspace_geometry_revision_) {
        ++realspace_reverse_launch_diagnostics_
              .materialized_table_reuse_count;
        return;
    }
    if (materialized_table_revision_
        != std::numeric_limits<std::size_t>::max())
        ++realspace_reverse_launch_diagnostics_
              .materialized_table_invalidation_count;

    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t pair_count = n_nodes * (n_nodes - (n_nodes != 0)) / 2;
    const std::size_t n_sigma = feature_smearing_widths_.size();
    const std::size_t source_replicas = density_max_l_ == 0 ? 1 : 4;
    const std::size_t receiver_replicas = feature_max_l_ == 0 ? 1 : 4;
    const std::size_t energy_replicas = source_replicas;
    const std::size_t feature_records_per_pair =
        2 * n_sigma * source_replicas * receiver_replicas;
    const std::size_t energy_records_per_pair =
        energy_replicas * energy_replicas;
    const std::size_t feature_records =
        pair_count * feature_records_per_pair;
    const std::size_t energy_records = pair_count * energy_records_per_pair;

    LongRangeKernelVector<double> feature_k;
    LongRangeKernelVector<double> feature_g;
    LongRangeKernelVector<double> energy_k;
    LongRangeKernelVector<double> energy_g;
    try {
        feature_k = LongRangeKernelVector<double>(
            "long_range_realspace_materialized_feature_k", feature_records);
        feature_g = LongRangeKernelVector<double>(
            "long_range_realspace_materialized_feature_g", feature_records);
        energy_k = LongRangeKernelVector<double>(
            "long_range_realspace_materialized_energy_k", energy_records);
        energy_g = LongRangeKernelVector<double>(
            "long_range_realspace_materialized_energy_g", energy_records);
    } catch (const std::exception&) {
        materialized_feature_k_ = LongRangeKernelVector<double>{};
        materialized_feature_g_ = LongRangeKernelVector<double>{};
        materialized_energy_k_ = LongRangeKernelVector<double>{};
        materialized_energy_g_ = LongRangeKernelVector<double>{};
        materialized_table_revision_ = std::numeric_limits<std::size_t>::max();
        realspace_reverse_strategy_ = RealspaceReverseStrategy::pair_atomic;
        realspace_reverse_strategy_reason_ = "materialized_allocation_failed";
        ++realspace_reverse_launch_diagnostics_
              .materialized_admission_rejection_count;
        realspace_reverse_launch_diagnostics_.materialized_radial_entries = 0;
        realspace_reverse_launch_diagnostics_.materialized_radial_bytes = 0;
        return;
    }
    auto positions = cache.realspace_positions;
    auto node_graph = cache.node_graph_index;
    auto effective_widths = realspace_effective_widths_;
    constexpr double feature_displacement = 0.1;
    constexpr double energy_displacement = 0.02;
    const double density_width = density_smearing_width_;
    const std::size_t feature_extent =
        n_nodes * n_nodes * n_sigma * source_replicas * receiver_replicas;
    Kokkos::parallel_for(
        "long_range realspace materialized feature radial tables",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, feature_extent),
        KOKKOS_LAMBDA(const std::size_t flat) {
            std::size_t value = flat;
            const std::size_t receiver_replica = value % receiver_replicas;
            value /= receiver_replicas;
            const std::size_t source_replica = value % source_replicas;
            value /= source_replicas;
            const std::size_t sigma = value % n_sigma;
            value /= n_sigma;
            const std::size_t receiver = value % n_nodes;
            const std::size_t source = value / n_nodes;
            if (source == receiver
                || node_graph(source) != node_graph(receiver))
                return;
            const std::size_t left = source < receiver ? source : receiver;
            const std::size_t right = source < receiver ? receiver : source;
            const std::size_t pair = left * n_nodes
                - left * (left + 1) / 2 + (right - left - 1);
            const std::size_t orientation = source == left ? 0 : 1;
            const std::size_t record =
                ((((pair * 2 + orientation) * n_sigma + sigma)
                      * source_replicas
                    + source_replica)
                      * receiver_replicas
                    + receiver_replica);
            double distance_squared = 0.0;
            for (std::size_t cartesian = 0; cartesian < 3; ++cartesian) {
                const double source_offset = source_replica > 0
                        && cartesian == source_replica - 1
                    ? feature_displacement
                    : 0.0;
                const double receiver_offset = receiver_replica > 0
                        && cartesian == receiver_replica - 1
                    ? feature_displacement
                    : 0.0;
                const double difference = realspace_displaced_difference(
                    positions(source, cartesian), source_offset,
                    positions(receiver, cartesian), receiver_offset);
                distance_squared = realspace_norm_accumulate(
                    difference, difference, distance_squared);
            }
            const double distance = realspace_sqrt(distance_squared);
            feature_k(record) =
                realspace_kernel(distance, effective_widths(sigma));
            feature_g(record) = realspace_kernel_gradient_factor(
                distance, effective_widths(sigma));
        });
    const std::size_t energy_extent =
        n_nodes * n_nodes * energy_records_per_pair;
    Kokkos::parallel_for(
        "long_range realspace materialized energy radial tables",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, energy_extent),
        KOKKOS_LAMBDA(const std::size_t flat) {
            std::size_t value = flat;
            const std::size_t right_replica = value % energy_replicas;
            value /= energy_replicas;
            const std::size_t left_replica = value % energy_replicas;
            value /= energy_replicas;
            const std::size_t right = value % n_nodes;
            const std::size_t left = value / n_nodes;
            if (right <= left || node_graph(left) != node_graph(right))
                return;
            const std::size_t pair = left * n_nodes
                - left * (left + 1) / 2 + (right - left - 1);
            const std::size_t record =
                (pair * energy_replicas + left_replica) * energy_replicas
                + right_replica;
            double distance_squared = 0.0;
            for (std::size_t cartesian = 0; cartesian < 3; ++cartesian) {
                const double left_offset = left_replica > 0
                        && cartesian == left_replica - 1
                    ? energy_displacement
                    : 0.0;
                const double right_offset = right_replica > 0
                        && cartesian == right_replica - 1
                    ? energy_displacement
                    : 0.0;
                const double difference = realspace_displaced_difference(
                    positions(right, cartesian), right_offset,
                    positions(left, cartesian), left_offset);
                distance_squared = realspace_norm_accumulate(
                    difference, difference, distance_squared);
            }
            const double distance = realspace_sqrt(distance_squared);
            energy_k(record) = realspace_kernel(distance, density_width);
            energy_g(record) =
                realspace_kernel_gradient_factor(distance, density_width);
        });
    materialized_feature_k_ = feature_k;
    materialized_feature_g_ = feature_g;
    materialized_energy_k_ = energy_k;
    materialized_energy_g_ = energy_g;
    materialized_table_revision_ = realspace_geometry_revision_;
    ++realspace_reverse_launch_diagnostics_.materialized_table_build_count;
    realspace_reverse_launch_diagnostics_.materialized_radial_entries =
        feature_records + energy_records;
    realspace_reverse_launch_diagnostics_.materialized_radial_bytes =
        (feature_records + energy_records) * 2 * sizeof(double);
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::resolve_realspace_reverse_strategy() const
{
    if (realspace_reverse_strategy_resolved_)
        return;
#if defined(KOKKOS_ENABLE_CUDA)
    realspace_reverse_strategy_ = RealspaceReverseStrategy::pair_atomic;
    realspace_reverse_strategy_reason_ = "cuda_qualified_backend";
#else
    realspace_reverse_strategy_ = RealspaceReverseStrategy::owner;
    realspace_reverse_strategy_reason_ = "cpu_or_unqualified_backend";
    realspace_reverse_launch_diagnostics_.atomic_accumulator_bytes = 0;
#endif
    realspace_reverse_strategy_resolved_ = true;
}

template <typename Scalar>
std::size_t PolarLongRangeOpsT<Scalar>::
checked_realspace_atomic_accumulator_bytes(const std::size_t n_nodes) const
{
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (density_width_ != 0 && n_nodes > maximum / density_width_)
        throw std::overflow_error(
            "real-space atomic accumulator element count overflows size_t.");
    const std::size_t elements = n_nodes * density_width_;
    if (elements > maximum / sizeof(double))
        throw std::overflow_error(
            "real-space atomic accumulator byte count overflows size_t.");
    return elements * sizeof(double);
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::record_input_h2d(
    const std::size_t elements,const std::size_t element_size,
    const char* what) const
{
    if(element_size!=0
        &&elements>std::numeric_limits<std::size_t>::max()/element_size)
        throw std::overflow_error(
            std::string("long-range ")+what+" input byte count overflows size_t.");
    const std::size_t bytes=elements*element_size;
    if(input_h2d_bytes_>std::numeric_limits<std::size_t>::max()-bytes)
        throw std::overflow_error(
            std::string("long-range cumulative ")+what+
            " input byte count overflows size_t.");
    input_h2d_bytes_+=bytes;
}

template <typename Scalar>
PolarLongRangeOpsT<Scalar>::PolarLongRangeOpsT(double kspace_cutoff,
    int density_max_l,
    double density_smearing_width,
    int feature_max_l,
    std::vector<double> feature_smearing_widths)
    : kspace_cutoff_(kspace_cutoff),
      density_max_l_(density_max_l),
      density_smearing_width_(density_smearing_width),
      feature_max_l_(feature_max_l),
      feature_smearing_widths_(std::move(feature_smearing_widths))
{
    const auto positive = [] (const double value) {
        return std::isfinite(value) && value > 0.0;
    };
    if (!positive(kspace_cutoff_))
        throw std::invalid_argument(
            "PolarLongRangeOps requires a positive finite kspace_cutoff.");
    if (density_max_l_ < 0 || density_max_l_ > 1)
        throw std::invalid_argument(
            "PolarLongRangeOps density_max_l must be 0 or 1 (the analytic "
            "radial Fourier factors reject l > 1).");
    if (feature_max_l_ < 0 || feature_max_l_ > 1)
        throw std::invalid_argument(
            "PolarLongRangeOps feature_max_l must be 0 or 1 (the analytic "
            "radial Fourier factors reject l > 1).");
    if (!positive(density_smearing_width_))
        throw std::invalid_argument(
            "PolarLongRangeOps requires a positive density smearing width.");
    if (feature_smearing_widths_.empty())
        throw std::invalid_argument(
            "PolarLongRangeOps requires at least one feature smearing width.");
    for (const double width : feature_smearing_widths_) {
        if (!positive(width))
            throw std::invalid_argument(
                "PolarLongRangeOps feature smearing widths must be positive.");
    }
    density_width_ = static_cast<std::size_t>(density_max_l_ + 1)
        * static_cast<std::size_t>(density_max_l_ + 1);
    feature_width_ = static_cast<std::size_t>(feature_max_l_ + 1)
        * static_cast<std::size_t>(feature_max_l_ + 1);
    feature_dim_ = feature_width_ * feature_smearing_widths_.size();

    // The e3nn-sorted feature spelling: one multiplicity block per l,
    // ascending (features.py:345-356).
    for (int l = 0; l <= feature_max_l_; ++l) {
        for (std::size_t c = 0; c < feature_smearing_widths_.size(); ++c) {
            const std::size_t offset = c * feature_width_;
            for (std::size_t m = static_cast<std::size_t>(l)
                    * static_cast<std::size_t>(l);
                m < static_cast<std::size_t>(l + 1)
                    * static_cast<std::size_t>(l + 1);
                ++m)
                output_permutation_.push_back(m + offset);
        }
    }
    output_permutation_kernel_ = LongRangeKernelVector<std::size_t>(
        "long_range_output_permutation", feature_dim_);
    {
        Kokkos::View<std::size_t*, Kokkos::LayoutRight, Kokkos::HostSpace>
            host("long_range_output_permutation_host", feature_dim_);
        for (std::size_t j = 0; j < feature_dim_; ++j)
            host(j) = output_permutation_[j];
        Kokkos::deep_copy(output_permutation_kernel_, host);
    }
    const auto make_fourier_parameters = [] (
        const std::vector<double>& sigmas, const bool normalize_receiver,
        const char* label) {
        const double prefactor_constant = (4.0 * pi) * std::sqrt(pi / 2.0);
        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> host(
            Kokkos::view_alloc(std::string(label)), sigmas.size(), 5);
        for (std::size_t i = 0; i < sigmas.size(); ++i) {
            const double sigma2 = sigmas[i] * sigmas[i];
            const double pref0 = prefactor_constant * sigma2 * sigmas[i];
            host(i, 0) = pref0;
            host(i, 1) = pref0 * sigma2;
            host(i, 2) = sigma2;
            host(i, 3) = cl_sigma(0, sigmas[i], normalize_receiver);
            host(i, 4) = cl_sigma(1, sigmas[i], normalize_receiver);
        }
        return stage_matrix<Scalar>(host, label);
    };
    density_fourier_parameters_ = make_fourier_parameters(
        {density_smearing_width_}, false,
        "long_range_density_basis_parameters");
    feature_fourier_parameters_ = make_fourier_parameters(
        feature_smearing_widths_, true,
        "long_range_feature_basis_parameters");
    {
        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> host(
            "long_range_feature_basis_derivatives_host",
            feature_smearing_widths_.size(), 3);
        for (std::size_t s = 0; s < feature_smearing_widths_.size(); ++s) {
            const double sigma = feature_smearing_widths_[s];
            const double sigma2 = sigma * sigma;
            const double pref0 =
                (4.0 * pi) * std::sqrt(pi / 2.0) * sigma2 * sigma;
            host(s, 0) = sigma2;
            host(s, 1) =
                pref0 * cl_sigma(0, sigma, true) * harmonic_l0();
            host(s, 2) = pref0 * sigma2
                * cl_sigma(1, sigma, true) * harmonic_l1();
        }
        feature_fourier_derivatives_ = stage_matrix<Scalar>(
            host, "long_range_feature_basis_derivatives");
    }

    build_si_constants(density_max_l_, density_smearing_width_, feature_max_l_,
        feature_smearing_widths_, true, si_feature_constants_,
        si_feature_select_);
    build_si_constants(density_max_l_, density_smearing_width_, density_max_l_,
        {density_smearing_width_}, false, si_energy_constants_,
        si_energy_select_);

    // Real-space widths and finite-displacement reconstruction constants are
    // immutable model data. Upload them once so production forward calls move
    // only the changing compact inputs.
    {
        const std::size_t n_sigma = feature_smearing_widths_.size();
        Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
            effective_widths_host("long_range_realspace_effective_widths_host",
                n_sigma);
        Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
            l0_factors_host("long_range_realspace_l0_factors_host", n_sigma);
        Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
            l1_factors_host("long_range_realspace_l1_factors_host", n_sigma);
        for (std::size_t c = 0; c < n_sigma; ++c) {
            const double sigma = feature_smearing_widths_[c];
            effective_widths_host(c) = std::sqrt(
                (density_smearing_width_ * density_smearing_width_
                    + sigma * sigma)
                / 2.0);
            l0_factors_host(c) =
                cl_sigma(0, sigma, true) / cl_sigma(0, sigma, false);
            l1_factors_host(c) = std::sqrt(3.0) * sigma * sigma
                * cl_sigma(1, sigma, true) / cl_sigma(0, sigma, false) / 0.1;
        }
        realspace_effective_widths_ = stage_vector<double>(
            effective_widths_host, "long_range_realspace_effective_widths");
        realspace_l0_factors_ = stage_vector<double>(
            l0_factors_host, "long_range_realspace_l0_factors");
        realspace_l1_factors_ = stage_vector<double>(
            l1_factors_host, "long_range_realspace_l1_factors");
    }
    si_feature_constants_kernel_ = stage_vector<double>(
        si_feature_constants_, "long_range_si_feature_constants_kernel");
    si_energy_constants_kernel_ = stage_vector<double>(
        si_energy_constants_, "long_range_si_energy_constants_kernel");
    si_feature_select_kernel_ = LongRangeKernelVector<std::size_t>(
        "long_range_si_feature_select_kernel", si_feature_select_.size());
    si_energy_select_kernel_ = LongRangeKernelVector<std::size_t>(
        "long_range_si_energy_select_kernel", si_energy_select_.size());
    {
        Kokkos::View<std::size_t*, Kokkos::LayoutRight, Kokkos::HostSpace>
            feature_host("long_range_si_feature_select_host",
                si_feature_select_.size());
        Kokkos::View<std::size_t*, Kokkos::LayoutRight, Kokkos::HostSpace>
            energy_host("long_range_si_energy_select_host",
                si_energy_select_.size());
        for (std::size_t i = 0; i < si_feature_select_.size(); ++i)
            feature_host(i) = si_feature_select_[i];
        for (std::size_t i = 0; i < si_energy_select_.size(); ++i)
            energy_host(i) = si_energy_select_[i];
        Kokkos::deep_copy(si_feature_select_kernel_, feature_host);
        Kokkos::deep_copy(si_energy_select_kernel_, energy_host);
    }

    // The constant [features_dim, 4] matrix of the external-field and
    // internal-field blocks (l=0 and l=1 analytic entries), right-multiplied
    // by the permutation that reorders the columns to [V, E_z, E_x, E_y].
    external_field_matrix_ = LongRangeMatrix<double>(
        "long_range_field_matrix", feature_dim_, 4);
    auto matrix = external_field_matrix_;
    // The matrix is a host-space view, so its zero fill is a host fill: nvcc
    // refuses an extended lambda enclosed by a constructor, and this needs no
    // kernel.
    Kokkos::deep_copy(matrix, 0.0);
    for (std::size_t i_s = 0; i_s < feature_smearing_widths_.size(); ++i_s) {
        const double sigma = feature_smearing_widths_[i_s];
        matrix(i_s, 0) = cl_sigma(0, sigma, true) * std::sqrt(8.0 * pi)
            * std::tgamma(1.5) * sigma * sigma * sigma;
    }
    if (feature_max_l_ >= 1) {
        for (std::size_t i_s = 0; i_s < feature_smearing_widths_.size(); ++i_s) {
            const double sigma = feature_smearing_widths_[i_s];
            const double magnitude = cl_sigma(1, sigma, true) * std::sqrt(1.5)
                * std::pow(sigma, 5) * 2.0 * pi;
            for (int m = 0; m < 3; ++m)
                matrix(feature_smearing_widths_.size() + i_s * 3 + m, 1 + m)
                    = magnitude;
        }
    }
    for (std::size_t i = 0; i < feature_dim_; ++i) {
        // Right-multiplication by [[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]]:
        // new column 0 is the old column 0, new column 1 the old column 2,
        // new column 2 the old column 3, new column 3 the old column 1 --
        // the exact factor the oracle applies before its forward permutes
        // node-field columns by [0, 3, 1, 2].
        const double old_1 = matrix(i, 1);
        const double old_2 = matrix(i, 2);
        const double old_3 = matrix(i, 3);
        matrix(i, 1) = old_2;
        matrix(i, 2) = old_3;
        matrix(i, 3) = old_1;
    }
    // The kernels and the adjoint GEMM consume the matrix in the kernel
    // space; the host copy above stays the accessor's own.
    external_field_kernel_ = stage_matrix<Scalar>(
        external_field_matrix_, "long_range_field_matrix_kernel");
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::prepare_workspace(
    const std::size_t n_k,
    const std::size_t n_nodes,
    const std::size_t n_graphs) const
{
    if (workspace_n_k_ == n_k && workspace_nodes_ == n_nodes
        && workspace_graphs_ == n_graphs)
        return;
    matrix_workspace_.clear();
    vector_workspace_.clear();
    realspace_compute_vector_workspace_.clear();
    realspace_compute_matrix_workspace_.clear();
    realspace_compute_tensor3_workspace_.clear();
    int_vector_workspace_.clear();
    tensor3_workspace_.clear();
    workspace_n_k_ = n_k;
    workspace_nodes_ = n_nodes;
    workspace_graphs_ = n_graphs;
}

template <typename Scalar>
LongRangeAllocationUsage
PolarLongRangeOpsT<Scalar>::workspace_allocation_usage() const
{
    LongRangeAllocationUsage usage;
    const auto add = [&usage] (const auto& view) {
        const std::size_t elements = static_cast<std::size_t>(view.span());
        usage.elements += elements;
        usage.bytes += elements
            * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    for (const auto& [label, view] : matrix_workspace_) {
        static_cast<void>(label);
        add(view);
    }
    for (const auto& [label, view] : vector_workspace_) {
        static_cast<void>(label);
        add(view);
    }
    for (const auto& [label, view] : realspace_compute_vector_workspace_) {
        static_cast<void>(label);
        add(view);
    }
    for (const auto& [label, view] : realspace_compute_matrix_workspace_) {
        static_cast<void>(label);
        add(view);
    }
    for (const auto& [label, view] : realspace_compute_tensor3_workspace_) {
        static_cast<void>(label);
        add(view);
    }
    for (const auto& [label, view] : int_vector_workspace_) {
        static_cast<void>(label);
        add(view);
    }
    for (const auto& [label, view] : tensor3_workspace_) {
        static_cast<void>(label);
        add(view);
    }
    add(materialized_feature_k_);
    add(materialized_feature_g_);
    add(materialized_energy_k_);
    add(materialized_energy_g_);
    add(full_feature_sender_);
    add(full_feature_receiver_);
    add(full_feature_valid_);
    add(full_feature_difference_);
    add(full_feature_norm_);
    add(full_feature_k_);
    add(full_feature_g_);
    add(full_feature_contribution_);
    add(full_feature_replica_positions_);
    add(full_feature_replica_charges_);
    add(full_energy_sender_);
    add(full_energy_receiver_);
    add(full_energy_valid_);
    add(full_energy_difference_);
    add(full_energy_norm_);
    add(full_energy_k_);
    add(full_energy_g_);
    add(full_energy_contribution_);
    add(full_energy_replica_positions_);
    add(full_energy_replica_charges_);
    return usage;
}

template <typename Scalar>
LongRangeKernelMatrix<Scalar> PolarLongRangeOpsT<Scalar>::workspace_matrix(
    const std::string& label,
    const std::size_t rows,
    const std::size_t columns) const
{
    auto& view = matrix_workspace_[label];
    if (view.extent(0) != rows || view.extent(1) != columns
        || view.data() == nullptr) {
        view = LongRangeKernelMatrix<Scalar>(
            Kokkos::view_alloc(label), rows, columns);
        ++arena_view_allocation_count_;
    }
    return view;
}

template <typename Scalar>
LongRangeKernelVector<Scalar> PolarLongRangeOpsT<Scalar>::workspace_vector(
    const std::string& label, const std::size_t size) const
{
    auto& view = vector_workspace_[label];
    if (view.extent(0) != size || view.data() == nullptr) {
        view = LongRangeKernelVector<Scalar>(Kokkos::view_alloc(label), size);
        ++arena_view_allocation_count_;
    }
    return view;
}

template <typename Scalar>
LongRangeKernelVector<double>
PolarLongRangeOpsT<Scalar>::workspace_realspace_compute_vector(
    const std::string& label, const std::size_t size) const
{
    auto& view = realspace_compute_vector_workspace_[label];
    if (view.extent(0) != size || view.data() == nullptr) {
        view = LongRangeKernelVector<double>(Kokkos::view_alloc(label), size);
        ++arena_view_allocation_count_;
    }
    return view;
}

template <typename Scalar>
LongRangeKernelMatrix<double>
PolarLongRangeOpsT<Scalar>::workspace_realspace_compute_matrix(
    const std::string& label,
    const std::size_t rows,
    const std::size_t columns) const
{
    auto& view = realspace_compute_matrix_workspace_[label];
    if (view.extent(0) != rows || view.extent(1) != columns
        || view.data() == nullptr) {
        view = LongRangeKernelMatrix<double>(
            Kokkos::view_alloc(label), rows, columns);
        ++arena_view_allocation_count_;
    }
    return view;
}

template <typename Scalar>
LongRangeKernelTensor3<double>
PolarLongRangeOpsT<Scalar>::workspace_realspace_compute_tensor3(
    const std::string& label,
    const std::size_t first,
    const std::size_t second,
    const std::size_t third) const
{
    auto& view = realspace_compute_tensor3_workspace_[label];
    if (view.extent(0) != first || view.extent(1) != second
        || view.extent(2) != third || view.data() == nullptr) {
        view = LongRangeKernelTensor3<double>(
            Kokkos::view_alloc(label), first, second, third);
        ++arena_view_allocation_count_;
    }
    return view;
}

template <typename Scalar>
LongRangeKernelVector<int> PolarLongRangeOpsT<Scalar>::workspace_int_vector(
    const std::string& label, const std::size_t size) const
{
    auto& view = int_vector_workspace_[label];
    if (view.extent(0) != size || view.data() == nullptr) {
        view = LongRangeKernelVector<int>(Kokkos::view_alloc(label), size);
        ++arena_view_allocation_count_;
    }
    return view;
}

template <typename Scalar>
LongRangeKernelTensor3<Scalar> PolarLongRangeOpsT<Scalar>::workspace_tensor3(
    const std::string& label,
    const std::size_t first,
    const std::size_t second,
    const std::size_t third) const
{
    auto& view = tensor3_workspace_[label];
    if (view.extent(0) != first || view.extent(1) != second
        || view.extent(2) != third || view.data() == nullptr) {
        view = LongRangeKernelTensor3<Scalar>(
            Kokkos::view_alloc(label), first, second, third);
        ++arena_view_allocation_count_;
    }
    return view;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::build_si_constants(
    const int l_source,
    const double sigma_source,
    const int l_receive,
    const std::vector<double>& sigmas_receive,
    const bool normalize_receiver,
    LongRangeVector<double>& constants,
    std::vector<std::size_t>& select) const
{
    const int l_max = std::min(l_source, l_receive);
    const std::size_t n_sigma = sigmas_receive.size();
    constants = LongRangeVector<double>(
        "long_range_si_constants", n_sigma * static_cast<std::size_t>(l_max + 1)
            * static_cast<std::size_t>(l_max + 1));
    Kokkos::deep_copy(constants, 0.0);
    select.clear();

    // Fixed init-time quadrature mirroring gto_utils.py:233-315: the same
    // deterministic 10 000-point grid, the analytic integrand, one trapezoid
    // per (l, sigma).  The forward stays a gather plus broadcast scale -- no
    // runtime quadrature anywhere.
    constexpr std::size_t grid_points = 10000;
    const double grid_start = 0.0001;
    for (int l = 0; l <= l_max; ++l) {
        for (std::size_t i_s = 0; i_s < n_sigma; ++i_s) {
            const double sigma_receive = sigmas_receive[i_s];
            const double grid_stop = 10.0 * std::max(sigma_receive, sigma_source);
            const double step = (grid_stop - grid_start)
                / static_cast<double>(grid_points - 1);
            double accumulator = 0.0;
            double previous = 0.0;
            for (std::size_t i = 0; i < grid_points; ++i) {
                // numpy linspace semantics: the last point is the stop
                // exactly, all earlier points are start + i*step.
                const double r = i + 1 == grid_points
                    ? grid_stop
                    : grid_start + static_cast<double>(i) * step;
                const double x = 0.5 * r * r / (sigma_receive * sigma_receive);
                const double f1 = std::pow(2.0, l + 0.5)
                    * std::pow(sigma_receive, 2 * l + 3) * std::tgamma(l + 1.5)
                    * (l == 0 ? gammainc_3_2(x) : gammainc_5_2(x))
                    * std::pow(r, -(l + 1));
                const double f2 = sigma_receive * sigma_receive
                    * std::pow(r, l)
                    * std::exp(-0.5 * r * r / (sigma_receive * sigma_receive));
                const double integrand = std::pow(r, l + 2)
                    * std::exp(-0.5 * r * r / (sigma_source * sigma_source))
                    * (f1 + f2);
                if (i > 0)
                    accumulator += step * (previous + integrand);
                previous = integrand;
            }
            const double value = 0.5 * accumulator;
            const double prefactor = field_constant / (2 * l + 1);
            const double coefficient = cl_sigma(l, sigma_source, false)
                * cl_sigma(l, sigma_receive, normalize_receiver) * prefactor
                * value;
            for (int m = 0; m < 2 * l + 1; ++m) {
                constants(n_sigma * static_cast<std::size_t>(l)
                        * static_cast<std::size_t>(l)
                    + static_cast<std::size_t>(m) + i_s * (2 * l + 1))
                    = coefficient;
                select.push_back(static_cast<std::size_t>(l)
                    * static_cast<std::size_t>(l) + static_cast<std::size_t>(m));
            }
        }
    }
}

template <typename Scalar>
LongRangeGeometryCacheT<Scalar>
PolarLongRangeOpsT<Scalar>::realspace_geometry_cache(
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index) const
{
    LongRangeGeometryCacheT<Scalar> cache;
    realspace_geometry_cache_into(positions, cell, node_graph_index, cache);
    return cache;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::realspace_geometry_cache_into(
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index,
    LongRangeGeometryCacheT<Scalar>& cache) const
{
    const std::size_t n_nodes = positions.extent(0);
    const std::size_t n_graphs = cell.extent(0);
    if (positions.extent(1) != 3)
        throw std::invalid_argument(
            "realspace_geometry_cache: positions must have three columns.");
    if (cell.extent(1) != 3 || cell.extent(2) != 3 || n_graphs == 0)
        throw std::invalid_argument(
            "realspace_geometry_cache: cell must be [graphs, 3, 3].");
    if (node_graph_index.extent(0) != n_nodes)
        throw std::invalid_argument(
            "realspace_geometry_cache: node_graph_index must match the "
            "position rows.");
    for (std::size_t n = 0; n < n_nodes; ++n)
        if (node_graph_index(n) < 0
            || static_cast<std::size_t>(node_graph_index(n)) >= n_graphs)
            throw std::invalid_argument(
                "realspace_geometry_cache: node_graph_index must address the "
                "cell's graphs.");

    // The operator and cache have the same evaluation owner.  Selecting
    // real space invalidates reciprocal scratch even when this cache itself
    // is already reusable, so a direct -> real transition cannot retain the
    // previous evaluation's dense arena views.
    prepare_workspace(0, n_nodes, n_graphs);

    const bool exact_reuse = cache.mode == LongRangeGeometryMode::realspace
        && cache.n_nodes == n_nodes && cache.n_graphs == n_graphs
        && cache.positions_host_values.size() == n_nodes * 3
        && cache.node_graph_host_values.size() == n_nodes
        && std::equal(cache.positions_host_values.begin(),
            cache.positions_host_values.end(), positions.data())
        && std::equal(cache.node_graph_host_values.begin(),
            cache.node_graph_host_values.end(), node_graph_index.data());
    if (exact_reuse)
        return;

    if (realspace_geometry_revision_
        == std::numeric_limits<std::size_t>::max())
        realspace_geometry_revision_ = 0;
    else
        ++realspace_geometry_revision_;

    const bool replace = cache.mode != LongRangeGeometryMode::realspace
        || cache.n_nodes != n_nodes || cache.n_graphs != n_graphs
        || cache.realspace_positions.data() == nullptr
        || cache.node_graph_index.data() == nullptr;
    if (replace) {
        // Transitioning from direct reciprocal must release every k-space
        // table rather than merely leaving it unreachable.
        cache = LongRangeGeometryCacheT<Scalar>{};
        cache.realspace_positions = LongRangeKernelMatrix<double>(
            "long_range_realspace_positions", n_nodes, 3);
        cache.node_graph_index = LongRangeKernelVector<int>(
            "long_range_realspace_node_graph", n_nodes);
    }
    cache.mode = LongRangeGeometryMode::realspace;
    cache.n_k = 0;
    cache.n_nodes = n_nodes;
    cache.n_graphs = n_graphs;
    stage_matrix_into<double>(positions, cache.realspace_positions);
    record_input_h2d(n_nodes,3*sizeof(double),"real-space position");
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_node_graph("long_range_realspace_node_graph_host", n_nodes);
    for (std::size_t n = 0; n < n_nodes; ++n)
        host_node_graph(n) = node_graph_index(n);
    Kokkos::deep_copy(cache.node_graph_index, host_node_graph);
    record_input_h2d(n_nodes,sizeof(int),"real-space node graph");
    cache.node_graph_index_host = host_node_graph;
    cache.positions_host_values.assign(
        positions.data(), positions.data() + n_nodes * 3);
    cache.node_graph_host_values.assign(
        node_graph_index.data(), node_graph_index.data() + n_nodes);
    // Cell is admitted only for graph-count validation and API compatibility;
    // fixed-Cartesian real-space electrostatics has no cell identity/operand.
    cache.cell_host_values.clear();
}

template <typename Scalar>
LongRangeKernelMatrix<Scalar>
PolarLongRangeOpsT<Scalar>::realspace_features_kernel(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& multipoles,
    const bool include_self_interaction,
    const bool use_workspace) const
{
    if (cache.mode != LongRangeGeometryMode::realspace)
        throw std::invalid_argument(
            "realspace_features requires a real-space geometry cache.");
    if (multipoles.extent(0) != cache.n_nodes
        || multipoles.extent(1) != density_width_)
        throw std::invalid_argument(
            "realspace_features: multipoles must be [nodes, density_width].");
    if (feature_dim_ != 0
        && cache.n_nodes > std::numeric_limits<std::size_t>::max() / feature_dim_)
        throw std::overflow_error(
            "realspace_features: flattened output range overflows size_t.");

    auto output = use_workspace
        ? workspace_matrix(
              "long_range_realspace_features", cache.n_nodes, feature_dim_)
        : LongRangeKernelMatrix<Scalar>(
              "long_range_realspace_features", cache.n_nodes, feature_dim_);
    const std::size_t output_size = cache.n_nodes * feature_dim_;
    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t feature_dim = feature_dim_;
    const std::size_t n_sigma = feature_smearing_widths_.size();
    const int density_max_l = density_max_l_;
    constexpr std::size_t pair_tile = realspace_pair_tile();
    constexpr double displacement = 0.1;
    constexpr double epsilon = 1.0e-6;
    constexpr double four_pi = 4.0 * pi;
    auto positions = cache.realspace_positions;
    auto node_graph = cache.node_graph_index;
    auto effective_widths = realspace_effective_widths_;
    auto l0_factors = realspace_l0_factors_;
    auto l1_factors = realspace_l1_factors_;
    auto self_constants = si_feature_constants_kernel_;
    auto self_select = si_feature_select_kernel_;
    if (realspace_reverse_strategy_request_
        == RealspaceReverseStrategyRequest::materialized_forward) {
        const bool include_gradients =
            realspace_full_materialized_gradients_required_;
        if (admit_realspace_full_materialized(n_nodes, include_gradients))
            ensure_realspace_full_materialized_tables(cache, include_gradients);
        if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward) {
            const std::size_t source_replicas =
                density_max_l == 0 ? 1 : 4;
            const std::size_t receiver_replicas =
                feature_max_l_ == 0 ? 1 : 4;
            const std::size_t edge_extent = full_feature_sender_.extent(0);
            auto samples = use_workspace
                ? workspace_realspace_compute_matrix(
                      "long_range_full_feature_samples",
                      n_nodes * receiver_replicas, n_sigma)
                : LongRangeKernelMatrix<double>(
                      "long_range_full_feature_samples",
                      n_nodes * receiver_replicas, n_sigma);
            Kokkos::deep_copy(samples, 0.0);
            auto charges = full_feature_replica_charges_;
            auto edge_sender = full_feature_sender_;
            auto edge_receiver = full_feature_receiver_;
            auto edge_valid = full_feature_valid_;
            auto edge_k = full_feature_k_;
            auto contributions = full_feature_contribution_;
            Kokkos::parallel_for(
                "long_range full materialized feature replica charges",
                Kokkos::RangePolicy<LongRangeKernelSpace,
                    Kokkos::IndexType<std::size_t>>(
                    0, n_nodes * source_replicas),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const std::size_t node = flat / source_replicas;
                    const std::size_t replica = flat % source_replicas;
                    charges(flat) = realspace_replica_charge(multipoles, node,
                        replica, density_max_l, displacement);
                });
            if (realspace_full_materialized_reducer_
                == RealspaceFullMaterializedReducer::serial)
                Kokkos::parallel_for(
                    "long_range full materialized feature contributions scatter",
                    Kokkos::RangePolicy<LongRangeKernelSpace,
                        Kokkos::IndexType<std::size_t>>(
                        0, edge_extent * n_sigma),
                    KOKKOS_LAMBDA(const std::size_t flat) {
                        const std::size_t sigma = flat % n_sigma;
                        const std::size_t edge = flat / n_sigma;
                        const double value = edge_valid(edge)
                            ? field_constant * charges(edge_sender(edge))
                                * edge_k(edge, sigma) / four_pi
                            : 0.0;
                        contributions(edge, sigma) = value;
                    });
            if (realspace_full_materialized_reducer_
                == RealspaceFullMaterializedReducer::atomic_scatter) {
                Kokkos::parallel_for(
                    "long_range full materialized feature atomic scatter",
                    Kokkos::RangePolicy<LongRangeKernelSpace,
                        Kokkos::IndexType<std::size_t>>(
                        0, edge_extent * n_sigma),
                    KOKKOS_LAMBDA(const std::size_t flat) {
                        const std::size_t sigma = flat % n_sigma;
                        const std::size_t edge = flat / n_sigma;
                        if (edge_valid(edge))
                            Kokkos::atomic_add(
                                &samples(edge_receiver(edge), sigma),
                                field_constant * charges(edge_sender(edge))
                                    * edge_k(edge, sigma) / four_pi);
                    });
                ++realspace_reverse_launch_diagnostics_
                      .full_materialized_feature_atomic_selected_count;
                realspace_reverse_launch_diagnostics_
                    .full_materialized_feature_parallel_edge_extent = edge_extent;
            } else if (realspace_full_materialized_reducer_
                == RealspaceFullMaterializedReducer::segmented) {
                const std::size_t segment_length = n_nodes > 0
                    ? (n_nodes - 1) * source_replicas
                    : 0;
                const std::size_t league_extent =
                    n_nodes * receiver_replicas * n_sigma;
                using FeatureReductionPolicy =
                    Kokkos::TeamPolicy<LongRangeKernelSpace>;
                FeatureReductionPolicy policy(league_extent, Kokkos::AUTO);
                Kokkos::parallel_for(
                    "long_range full materialized feature segmented reduction",
                    policy,
                    KOKKOS_LAMBDA(
                        const FeatureReductionPolicy::member_type& team) {
                        const std::size_t flat = team.league_rank();
                        const std::size_t sigma = flat % n_sigma;
                        const std::size_t receiver_replica_flat = flat / n_sigma;
                        const std::size_t segment_begin =
                            receiver_replica_flat * segment_length;
                        double total = 0.0;
                        Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(team, segment_length),
                            [=](const std::size_t offset, double& value) {
                                const std::size_t edge = segment_begin + offset;
                                if (edge_valid(edge))
                                    value += field_constant
                                        * charges(edge_sender(edge))
                                        * edge_k(edge, sigma) / four_pi;
                            },
                            total);
                        Kokkos::single(Kokkos::PerTeam(team), [=]() {
                            samples(receiver_replica_flat, sigma) = total;
                        });
                    });
                ++realspace_reverse_launch_diagnostics_
                      .full_materialized_feature_segmented_selected_count;
                realspace_reverse_launch_diagnostics_
                    .full_materialized_feature_parallel_edge_extent = edge_extent;
            } else {
                Kokkos::parallel_for(
                    "long_range full materialized feature scatter reduction",
                    Kokkos::RangePolicy<LongRangeKernelSpace,
                        Kokkos::IndexType<std::size_t>>(
                        0, n_nodes * receiver_replicas * n_sigma),
                    KOKKOS_LAMBDA(const std::size_t flat) {
                        const std::size_t sigma = flat % n_sigma;
                        const std::size_t receiver_replica_flat = flat / n_sigma;
                        const std::size_t receiver =
                            receiver_replica_flat / receiver_replicas;
                        const std::size_t receiver_replica =
                            receiver_replica_flat % receiver_replicas;
                        double total = 0.0;
                        const std::size_t segment_length = n_nodes > 0
                            ? (n_nodes - 1) * source_replicas
                            : 0;
                        const std::size_t segment_begin =
                            (receiver * receiver_replicas + receiver_replica)
                            * segment_length;
                        for (std::size_t offset = 0; offset < segment_length;
                            ++offset)
                            total += contributions(segment_begin + offset, sigma);
                        samples(receiver_replica_flat, sigma) = total;
                    });
                ++realspace_reverse_launch_diagnostics_
                      .full_materialized_feature_serial_selected_count;
            }
            Kokkos::parallel_for(
                "long_range full materialized feature reconstruction",
                Kokkos::RangePolicy<LongRangeKernelSpace,
                    Kokkos::IndexType<std::size_t>>(0, output_size),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const std::size_t receiver = flat / feature_dim;
                    const std::size_t output_channel = flat % feature_dim;
                    const bool scalar_channel = output_channel < n_sigma;
                    const std::size_t sigma = scalar_channel
                        ? output_channel : (output_channel - n_sigma) / 3;
                    const std::size_t component = scalar_channel
                        ? 0 : (output_channel - n_sigma) % 3;
                    const std::size_t displaced = component == 0
                        ? 2 : (component == 1 ? 3 : 1);
                    const std::size_t center_flat =
                        receiver * receiver_replicas;
                    double value = scalar_channel
                        ? l0_factors(sigma) * samples(center_flat, sigma)
                        : l1_factors(sigma)
                            * (samples(center_flat + displaced, sigma)
                                - samples(center_flat, sigma));
                    if (include_self_interaction
                        && output_channel < self_constants.extent(0))
                        value += self_constants(output_channel)
                            * static_cast<double>(multipoles(receiver,
                                self_select(output_channel)));
                    output(receiver, output_channel) =
                        static_cast<Scalar>(value);
                });
            return output;
        }
    }
#ifdef KOKKOS_ENABLE_CUDA
    // The scalar and three vector outputs for one sigma are reconstructed
    // from the same four receiver-replica samples.  Form each sample once and
    // parallelize its sender/replica reduction across a CUDA team instead of
    // assigning one thread an O(N) serial walk for every output channel.
    if (n_nodes >= realspace_cuda_team_min_nodes) {
        const std::size_t source_replicas = density_max_l == 0 ? 1 : 4;
        const std::size_t receiver_replicas = feature_max_l_ == 0 ? 1 : 4;
        auto samples = use_workspace
            ? workspace_realspace_compute_matrix(
                  "long_range_realspace_feature_samples",
                  n_nodes * receiver_replicas, n_sigma)
            : LongRangeKernelMatrix<double>(
                  "long_range_realspace_feature_samples",
                  n_nodes * receiver_replicas, n_sigma);
        const std::size_t sample_extent =
            n_nodes * receiver_replicas * n_sigma;
        using FeatureSamplePolicy = Kokkos::TeamPolicy<LongRangeKernelSpace>;
        FeatureSamplePolicy sample_policy(sample_extent, Kokkos::AUTO);
        Kokkos::parallel_for(
            "long_range realspace feature sample reductions", sample_policy,
            KOKKOS_LAMBDA(const FeatureSamplePolicy::member_type& team) {
                const std::size_t flat = team.league_rank();
                const std::size_t sigma = flat % n_sigma;
                const std::size_t receiver_replica_flat = flat / n_sigma;
                const std::size_t receiver =
                    receiver_replica_flat / receiver_replicas;
                const std::size_t receiver_replica =
                    receiver_replica_flat % receiver_replicas;
                double total = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(
                        team, n_nodes * source_replicas),
                    [=](const std::size_t pair_flat, double& value) {
                        const std::size_t sender =
                            pair_flat / source_replicas;
                        if (sender == receiver
                            || node_graph(sender) != node_graph(receiver))
                            return;
                        const std::size_t source_replica =
                            pair_flat % source_replicas;
                        const double charge = realspace_replica_charge(
                            multipoles, sender, source_replica, density_max_l,
                            displacement);
                        double distance_squared = 0.0;
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian) {
                            const double source_offset = source_replica > 0
                                    && cartesian == source_replica - 1
                                ? displacement
                                : 0.0;
                            const double receiver_offset = receiver_replica > 0
                                    && cartesian == receiver_replica - 1
                                ? displacement
                                : 0.0;
                            const double difference =
                                positions(sender, cartesian) + source_offset
                                - positions(receiver, cartesian)
                                - receiver_offset;
                            distance_squared = realspace_norm_accumulate(
                                difference, difference, distance_squared);
                        }
                        const double distance =
                            realspace_sqrt(distance_squared);
                        const double scaled_distance =
                            0.5 * distance / effective_widths(sigma);
                        value += charge * realspace_erf(scaled_distance)
                            / (distance + epsilon);
                    },
                    total);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    samples(receiver_replica_flat, sigma) =
                        field_constant * total / four_pi;
                });
            });
        Kokkos::parallel_for(
            "long_range realspace feature sample reconstruction",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(0, output_size),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t receiver = flat / feature_dim;
                const std::size_t output_channel = flat % feature_dim;
                const bool scalar_channel = output_channel < n_sigma;
                const std::size_t sigma = scalar_channel
                    ? output_channel
                    : (output_channel - n_sigma) / 3;
                const std::size_t component = scalar_channel
                    ? 0
                    : (output_channel - n_sigma) % 3;
                // Receiver replica storage is [0,x,y,z], output is e3nn
                // [y,z,x].
                const std::size_t displaced = component == 0
                    ? 2
                    : (component == 1 ? 3 : 1);
                const std::size_t center_flat =
                    receiver * receiver_replicas;
                double value = scalar_channel
                    ? l0_factors(sigma) * samples(center_flat, sigma)
                    : l1_factors(sigma)
                        * (samples(center_flat + displaced, sigma)
                            - samples(center_flat, sigma));
                if (include_self_interaction
                    && output_channel < self_constants.extent(0))
                    value += self_constants(output_channel)
                        * static_cast<double>(multipoles(receiver,
                            self_select(output_channel)));
                output(receiver, output_channel) =
                    static_cast<Scalar>(value);
            });
        return output;
    }
#endif
    Kokkos::parallel_for(
        "long_range realspace features",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, output_size),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t receiver = flat / feature_dim;
            const std::size_t output_channel = flat % feature_dim;
            const bool scalar_channel = output_channel < n_sigma;
            const std::size_t sigma_channel = scalar_channel
                ? output_channel
                : (output_channel - n_sigma) / 3;
            const std::size_t vector_component = scalar_channel
                ? 0
                : (output_channel - n_sigma) % 3;
            // Receiver replica storage is [0,x,y,z], output is e3nn [y,z,x].
            const std::size_t displaced_receiver = vector_component == 0
                ? 2
                : (vector_component == 1 ? 3 : 1);
            const std::size_t source_replicas = density_max_l == 0 ? 1 : 4;
            double center_sum = 0.0;
            double displaced_sum = 0.0;
            for (std::size_t tile_begin = 0; tile_begin < n_nodes;
                tile_begin += pair_tile) {
                const std::size_t tile_end = tile_begin + pair_tile < n_nodes
                    ? tile_begin + pair_tile
                    : n_nodes;
                for (std::size_t sender = tile_begin; sender < tile_end; ++sender) {
                    if (sender == receiver
                        || node_graph(sender) != node_graph(receiver))
                        continue;
                    for (std::size_t source_replica = 0;
                        source_replica < source_replicas; ++source_replica) {
                        double charge =
                            static_cast<double>(multipoles(sender, 0));
                        if (density_max_l == 1) {
                            const double qx = static_cast<double>(
                                                  multipoles(sender, 3))
                                / displacement;
                            const double qy = static_cast<double>(
                                                  multipoles(sender, 1))
                                / displacement;
                            const double qz = static_cast<double>(
                                                  multipoles(sender, 2))
                                / displacement;
                            const double displaced_charge = qx + qy + qz;
                            charge = source_replica == 0
                                ? static_cast<double>(multipoles(sender, 0))
                                    - displaced_charge
                                : (source_replica == 1
                                          ? qx
                                          : (source_replica == 2 ? qy : qz));
                        }
                        const std::size_t sample_count = scalar_channel ? 1 : 2;
                        for (std::size_t sample = 0; sample < sample_count;
                            ++sample) {
                            const std::size_t receiver_replica = scalar_channel
                                ? 0
                                : (sample == 0 ? displaced_receiver : 0);
                            double distance_squared = 0.0;
                            for (std::size_t cartesian = 0; cartesian < 3;
                                ++cartesian) {
                                const double source_offset = source_replica > 0
                                        && cartesian == source_replica - 1
                                    ? displacement
                                    : 0.0;
                                const double receiver_offset =
                                    receiver_replica > 0
                                        && cartesian == receiver_replica - 1
                                    ? displacement
                                    : 0.0;
                                const double source_position =
                                    positions(sender, cartesian) + source_offset;
                                const double receiver_position =
                                    positions(receiver, cartesian)
                                    + receiver_offset;
                                const double difference =
                                    source_position - receiver_position;
                                distance_squared = realspace_norm_accumulate(
                                    difference, difference, distance_squared);
                            }
                            const double distance =
                                realspace_sqrt(distance_squared);
                            const double scaled_distance = 0.5 * distance
                                / effective_widths(sigma_channel);
                            const double numerator =
                                realspace_erf(scaled_distance);
                            const double radial =
                                numerator / (distance + epsilon);
                            const double contribution = charge * radial;
                            if (scalar_channel || sample == 1)
                                center_sum += contribution;
                            else
                                displaced_sum += contribution;
                        }
                    }
                }
            }
            const double center_potential =
                field_constant * center_sum / four_pi;
            const double displaced_potential =
                field_constant * displaced_sum / four_pi;
            double value = scalar_channel
                ? l0_factors(sigma_channel) * center_potential
                : l1_factors(sigma_channel)
                    * (displaced_potential - center_potential);
            if (include_self_interaction
                && output_channel < self_constants.extent(0)) {
                const double self_value =
                    self_constants(output_channel)
                    * static_cast<double>(
                        multipoles(receiver, self_select(output_channel)));
                value += self_value;
            }
            output(receiver, output_channel) = static_cast<Scalar>(value);
        });
    return output;
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::realspace_features(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& multipoles,
    const bool include_self_interaction,
    const bool use_workspace) const
{
    if (use_workspace)
        prepare_workspace(0, cache.n_nodes, cache.n_graphs);
    auto make_matrix = [this, use_workspace] (const char* label,
                           const std::size_t rows,
                           const std::size_t columns) {
        return use_workspace ? workspace_matrix(label, rows, columns)
                             : LongRangeKernelMatrix<Scalar>(
                                   Kokkos::view_alloc(std::string(label)), rows,
                                   columns);
    };
    auto multipoles_kernel = make_matrix("long_range_realspace_multipoles",
        multipoles.extent(0), multipoles.extent(1));
    stage_matrix_into<Scalar>(multipoles, multipoles_kernel);
    return download_matrix<double>(
        realspace_features_kernel(cache, multipoles_kernel,
            include_self_interaction, use_workspace),
        "long_range_realspace_features_host");
}

template <typename Scalar>
LongRangeKernelVector<Scalar>
PolarLongRangeOpsT<Scalar>::realspace_energy_kernel(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& multipoles,
    const bool include_self_interaction,
    const bool use_workspace) const
{
    if (cache.mode != LongRangeGeometryMode::realspace)
        throw std::invalid_argument(
            "realspace_energy requires a real-space geometry cache.");
    if (multipoles.extent(0) != cache.n_nodes
        || multipoles.extent(1) != density_width_)
        throw std::invalid_argument(
            "realspace_energy: multipoles must be [nodes, density_width].");
    const std::size_t replicas = density_max_l_ == 0 ? 1 : 4;
    if (cache.n_nodes > std::numeric_limits<std::size_t>::max() / replicas)
        throw std::overflow_error(
            "realspace_energy: receiver-replica range overflows size_t.");
    const std::size_t partial_count = cache.n_nodes * replicas;
    auto output = use_workspace
        ? workspace_vector("long_range_realspace_energy", cache.n_graphs)
        : LongRangeKernelVector<Scalar>(
              "long_range_realspace_energy", cache.n_graphs);
    auto partials = use_workspace
        ? workspace_realspace_compute_vector(
              "long_range_realspace_energy_partials", partial_count)
        : LongRangeKernelVector<double>(
              "long_range_realspace_energy_partials", partial_count);
    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t density_width = density_width_;
    const int density_max_l = density_max_l_;
    constexpr std::size_t pair_tile = realspace_pair_tile();
    constexpr double displacement = 0.02;
    constexpr double epsilon = 1.0e-6;
    const double density_smearing_width = density_smearing_width_;
    constexpr double four_pi = 4.0 * pi;
    auto positions = cache.realspace_positions;
    auto node_graph = cache.node_graph_index;
    auto self_constants = si_energy_constants_kernel_;
    auto self_select = si_energy_select_kernel_;

    ++realspace_energy_launch_diagnostics_.selected_count;
    realspace_energy_launch_diagnostics_.receiver_replica_league_extent =
        partial_count;
    realspace_energy_launch_diagnostics_.graph_reduction_league_extent =
        cache.n_graphs;
    realspace_energy_launch_diagnostics_.partial_elements = partial_count;

    if (realspace_reverse_strategy_request_
        == RealspaceReverseStrategyRequest::materialized_forward) {
        const bool include_gradients =
            realspace_full_materialized_gradients_required_;
        if (admit_realspace_full_materialized(n_nodes, include_gradients))
            ensure_realspace_full_materialized_tables(cache, include_gradients);
        if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward) {
            const std::size_t edge_extent = full_energy_sender_.extent(0);
            auto charges = full_energy_replica_charges_;
            auto edge_sender = full_energy_sender_;
            auto edge_receiver = full_energy_receiver_;
            auto edge_valid = full_energy_valid_;
            auto edge_k = full_energy_k_;
            auto contributions = full_energy_contribution_;
            Kokkos::parallel_for(
                "long_range full materialized energy replica charges",
                Kokkos::RangePolicy<LongRangeKernelSpace,
                    Kokkos::IndexType<std::size_t>>(0, n_nodes * replicas),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const std::size_t node = flat / replicas;
                    const std::size_t replica = flat % replicas;
                    charges(flat) = realspace_replica_charge(multipoles, node,
                        replica, density_max_l, displacement);
                });
            if (realspace_full_materialized_reducer_
                == RealspaceFullMaterializedReducer::serial)
                Kokkos::parallel_for(
                    "long_range full materialized energy contributions scatter",
                    Kokkos::RangePolicy<LongRangeKernelSpace,
                        Kokkos::IndexType<std::size_t>>(0, edge_extent),
                    KOKKOS_LAMBDA(const std::size_t edge) {
                        const std::size_t source_replica_flat = edge_sender(edge);
                        const double value = edge_valid(edge)
                            ? 0.5 * field_constant * charges(source_replica_flat)
                                * charges(edge_receiver(edge)) * edge_k(edge)
                                / four_pi
                            : 0.0;
                        contributions(edge) = value;
                    });
            if (realspace_full_materialized_reducer_
                != RealspaceFullMaterializedReducer::serial) {
                auto graph_sums = use_workspace
                    ? workspace_realspace_compute_vector(
                          "long_range_full_energy_graph_sums", cache.n_graphs)
                    : LongRangeKernelVector<double>(
                          "long_range_full_energy_graph_sums", cache.n_graphs);
                for (std::size_t graph = 0; graph < cache.n_graphs; ++graph) {
                    Kokkos::parallel_reduce(
                        "long_range full materialized energy parallel reduction",
                        Kokkos::RangePolicy<LongRangeKernelSpace,
                            Kokkos::IndexType<std::size_t>>(0, edge_extent),
                        KOKKOS_LAMBDA(
                            const std::size_t edge, double& value) {
                            if (!edge_valid(edge))
                                return;
                            const std::size_t source =
                                edge_sender(edge) / replicas;
                            if (static_cast<std::size_t>(node_graph(source))
                                == graph) {
                                const std::size_t sender = edge_sender(edge);
                                value += 0.5 * field_constant * charges(sender)
                                    * charges(edge_receiver(edge)) * edge_k(edge)
                                    / four_pi;
                            }
                        },
                        Kokkos::subview(graph_sums, graph));
                }
                Kokkos::parallel_for(
                    "long_range full materialized energy parallel self and cast",
                    Kokkos::RangePolicy<LongRangeKernelSpace,
                        Kokkos::IndexType<std::size_t>>(0, cache.n_graphs),
                    KOKKOS_LAMBDA(const std::size_t graph) {
                        double value = graph_sums(graph);
                        if (include_self_interaction)
                            for (std::size_t node = 0; node < n_nodes; ++node) {
                                if (static_cast<std::size_t>(node_graph(node))
                                    != graph)
                                    continue;
                                double node_self = 0.0;
                                for (std::size_t m = 0; m < density_width; ++m)
                                    node_self += static_cast<double>(
                                                     multipoles(node, m))
                                        * self_constants(m)
                                        * static_cast<double>(multipoles(
                                            node, self_select(m)));
                                value += 0.5 * node_self;
                            }
                        output(graph) = static_cast<Scalar>(value);
                    });
                ++realspace_energy_launch_diagnostics_
                      .full_materialized_parallel_selected_count;
                realspace_energy_launch_diagnostics_
                    .full_materialized_parallel_edge_extent = edge_extent;
                return output;
            }
            ++realspace_energy_launch_diagnostics_
                  .full_materialized_serial_selected_count;
            Kokkos::parallel_for(
                "long_range full materialized energy self and cast",
                Kokkos::RangePolicy<LongRangeKernelSpace,
                    Kokkos::IndexType<std::size_t>>(0, cache.n_graphs),
                KOKKOS_LAMBDA(const std::size_t graph) {
                    double value = 0.0;
                    for (std::size_t edge = 0; edge < edge_extent; ++edge) {
                        if (!edge_valid(edge))
                            continue;
                        const std::size_t source =
                            edge_sender(edge) / replicas;
                        if (static_cast<std::size_t>(node_graph(source))
                            == graph)
                            value += contributions(edge);
                    }
                    if (include_self_interaction)
                        for (std::size_t node = 0; node < n_nodes; ++node) {
                            if (static_cast<std::size_t>(node_graph(node))
                                != graph)
                                continue;
                            double node_self = 0.0;
                            for (std::size_t m = 0; m < density_width; ++m)
                                node_self += static_cast<double>(
                                                 multipoles(node, m))
                                    * self_constants(m)
                                    * static_cast<double>(multipoles(
                                        node, self_select(m)));
                            value += 0.5 * node_self;
                        }
                    output(graph) = static_cast<Scalar>(value);
                });
            return output;
        }
    }

#ifdef KOKKOS_ENABLE_CUDA
    if (n_nodes >= realspace_cuda_team_min_nodes) {
        using EnergyPartialPolicy = Kokkos::TeamPolicy<LongRangeKernelSpace>;
        EnergyPartialPolicy partial_policy(partial_count, Kokkos::AUTO);
        Kokkos::parallel_for(
            "long_range realspace energy receiver replica reductions",
            partial_policy,
            KOKKOS_LAMBDA(const EnergyPartialPolicy::member_type& team) {
                const std::size_t receiver_replica_flat = team.league_rank();
                const std::size_t receiver =
                    receiver_replica_flat / replicas;
                const std::size_t receiver_replica =
                    receiver_replica_flat % replicas;
                const std::size_t graph =
                    static_cast<std::size_t>(node_graph(receiver));
                const double receiver_charge = realspace_replica_charge(
                    multipoles, receiver, receiver_replica, density_max_l,
                    displacement);
                double node_energy = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, n_nodes * replicas),
                    [=](const std::size_t pair_flat, double& value) {
                        const std::size_t sender = pair_flat / replicas;
                        if (sender == receiver
                            || static_cast<std::size_t>(node_graph(sender))
                                != graph)
                            return;
                        const std::size_t source_replica =
                            pair_flat % replicas;
                        const double source_charge = realspace_replica_charge(
                            multipoles, sender, source_replica, density_max_l,
                            displacement);
                        double distance_squared = 0.0;
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian) {
                            const double source_offset = source_replica > 0
                                    && cartesian == source_replica - 1
                                ? displacement
                                : 0.0;
                            const double receiver_offset = receiver_replica > 0
                                    && cartesian == receiver_replica - 1
                                ? displacement
                                : 0.0;
                            const double difference =
                                positions(sender, cartesian) + source_offset
                                - positions(receiver, cartesian)
                                - receiver_offset;
                            distance_squared = realspace_norm_accumulate(
                                difference, difference, distance_squared);
                        }
                        const double distance =
                            realspace_sqrt(distance_squared);
                        const double scaled_distance =
                            0.5 * distance / density_smearing_width;
                        const double radial = realspace_erf(scaled_distance)
                            / (distance + epsilon);
                        value += 0.5 * field_constant * radial * source_charge
                            * receiver_charge / four_pi;
                    },
                    node_energy);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    partials(receiver_replica_flat) = node_energy;
                });
            });
    } else
#endif
    Kokkos::parallel_for(
        "long_range realspace energy receiver replicas",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, partial_count),
        KOKKOS_LAMBDA(const std::size_t receiver_replica_flat) {
            const std::size_t receiver = receiver_replica_flat / replicas;
            const std::size_t receiver_replica =
                receiver_replica_flat % replicas;
            const std::size_t graph =
                static_cast<std::size_t>(node_graph(receiver));
            const double receiver_qx = density_max_l == 1
                ? static_cast<double>(multipoles(receiver, 3)) / displacement
                : 0.0;
            const double receiver_qy = density_max_l == 1
                ? static_cast<double>(multipoles(receiver, 1)) / displacement
                : 0.0;
            const double receiver_qz = density_max_l == 1
                ? static_cast<double>(multipoles(receiver, 2)) / displacement
                : 0.0;
            const double receiver_displaced_charge =
                receiver_qx + receiver_qy + receiver_qz;
            const double receiver_charge = density_max_l == 0
                ? static_cast<double>(multipoles(receiver, 0))
                : (receiver_replica == 0
                          ? static_cast<double>(multipoles(receiver, 0))
                                - receiver_displaced_charge
                          : (receiver_replica == 1
                                    ? receiver_qx
                                    : (receiver_replica == 2 ? receiver_qy
                                                              : receiver_qz)));
            double node_energy = 0.0;
            for (std::size_t tile_begin = 0; tile_begin < n_nodes;
                tile_begin += pair_tile) {
                const std::size_t tile_end = tile_begin + pair_tile < n_nodes
                    ? tile_begin + pair_tile
                    : n_nodes;
                for (std::size_t sender = tile_begin; sender < tile_end;
                    ++sender) {
                    if (sender == receiver
                        || static_cast<std::size_t>(node_graph(sender)) != graph)
                        continue;
                    const double source_qx = density_max_l == 1
                        ? static_cast<double>(multipoles(sender, 3))
                            / displacement
                        : 0.0;
                    const double source_qy = density_max_l == 1
                        ? static_cast<double>(multipoles(sender, 1))
                            / displacement
                        : 0.0;
                    const double source_qz = density_max_l == 1
                        ? static_cast<double>(multipoles(sender, 2))
                            / displacement
                        : 0.0;
                    const double source_displaced_charge =
                        source_qx + source_qy + source_qz;
                    for (std::size_t source_replica = 0;
                        source_replica < replicas; ++source_replica) {
                        const double source_charge = density_max_l == 0
                            ? static_cast<double>(multipoles(sender, 0))
                            : (source_replica == 0
                                      ? static_cast<double>(multipoles(sender, 0))
                                            - source_displaced_charge
                                      : (source_replica == 1
                                                ? source_qx
                                                : (source_replica == 2
                                                          ? source_qy
                                                          : source_qz)));
                        double distance_squared = 0.0;
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian) {
                            const double source_offset = source_replica > 0
                                    && cartesian == source_replica - 1
                                ? displacement
                                : 0.0;
                            const double receiver_offset = receiver_replica > 0
                                    && cartesian == receiver_replica - 1
                                ? displacement
                                : 0.0;
                            const double source_position =
                                positions(sender, cartesian) + source_offset;
                            const double receiver_position =
                                positions(receiver, cartesian) + receiver_offset;
                            const double difference =
                                source_position - receiver_position;
                            distance_squared = realspace_norm_accumulate(
                                difference, difference, distance_squared);
                        }
                        const double distance =
                            realspace_sqrt(distance_squared);
                        const double scaled_distance =
                            0.5 * distance / density_smearing_width;
                        const double radial = realspace_erf(scaled_distance)
                            / (distance + epsilon);
                        double edge = 0.5 * field_constant;
                        edge *= radial;
                        edge *= source_charge;
                        edge *= receiver_charge;
                        edge /= four_pi;
                        node_energy += edge;
                    }
                }
            }
            partials(receiver_replica_flat) = node_energy;
        });

    Kokkos::parallel_for(
        "long_range realspace energy graph reduction",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, cache.n_graphs),
        KOKKOS_LAMBDA(const std::size_t graph) {
            double value = 0.0;
            for (std::size_t receiver = 0; receiver < n_nodes; ++receiver) {
                if (static_cast<std::size_t>(node_graph(receiver)) != graph)
                    continue;
                for (std::size_t receiver_replica = 0;
                    receiver_replica < replicas; ++receiver_replica) {
                    value += partials(
                        receiver * replicas + receiver_replica);
                }
            }
            if (include_self_interaction) {
                for (std::size_t node = 0; node < n_nodes; ++node) {
                    if (static_cast<std::size_t>(node_graph(node)) != graph)
                        continue;
                    double node_self = 0.0;
                    for (std::size_t m = 0; m < density_width; ++m) {
                        const double multipole =
                            static_cast<double>(multipoles(node, m));
                        const double self_field =
                            self_constants(m)
                            * static_cast<double>(
                                multipoles(node, self_select(m)));
                        node_self += multipole * self_field;
                    }
                    value += 0.5 * node_self;
                }
            }
            output(graph) = static_cast<Scalar>(value);
        });
    return output;
}

template <typename Scalar>
LongRangeVector<double> PolarLongRangeOpsT<Scalar>::realspace_energy(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& multipoles,
    const bool include_self_interaction,
    const bool use_workspace) const
{
    if (use_workspace)
        prepare_workspace(0, cache.n_nodes, cache.n_graphs);
    auto multipoles_kernel = use_workspace
        ? workspace_matrix("long_range_realspace_energy_multipoles",
              multipoles.extent(0), multipoles.extent(1))
        : LongRangeKernelMatrix<Scalar>("long_range_realspace_energy_multipoles",
              multipoles.extent(0), multipoles.extent(1));
    stage_matrix_into<Scalar>(multipoles, multipoles_kernel);
    return download_vector<double>(
        realspace_energy_kernel(cache, multipoles_kernel,
            include_self_interaction, use_workspace),
        "long_range_realspace_energy_host");
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::RealspaceAdjointKernelResult
PolarLongRangeOpsT<Scalar>::adjoint_realspace_features_kernel(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& multipoles,
    const LongRangeKernelMatrix<Scalar>& grad_features,
    const bool include_self_interaction,
    const bool use_workspace) const
{
    if (cache.mode != LongRangeGeometryMode::realspace)
        throw std::invalid_argument(
            "adjoint_realspace_features requires a real-space geometry cache.");
    if (multipoles.extent(0) != cache.n_nodes
        || multipoles.extent(1) != density_width_)
        throw std::invalid_argument(
            "adjoint_realspace_features: multipoles must be "
            "[nodes, density_width].");
    if (grad_features.extent(0) != cache.n_nodes
        || grad_features.extent(1) != feature_dim_)
        throw std::invalid_argument(
            "adjoint_realspace_features: feature seeds must be "
            "[nodes, feature_dim].");
    if (density_width_ != 0
        && cache.n_nodes
            > std::numeric_limits<std::size_t>::max() / density_width_)
        throw std::overflow_error(
            "adjoint_realspace_features: multipole owner range overflows "
            "size_t.");
    if (cache.n_nodes > std::numeric_limits<std::size_t>::max() / 3)
        throw std::overflow_error(
            "adjoint_realspace_features: position owner range overflows "
            "size_t.");

    resolve_realspace_reverse_strategy();
    const bool delegate_full_materialized_reverse =
        realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward
        && realspace_full_materialized_reducer_
            != RealspaceFullMaterializedReducer::serial;

    if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::pair_materialized
        && !admit_realspace_materialized(cache.n_nodes)) {
        // A rejected forced experiment stays in real space and falls back to
        // the bounded pair-atomic implementation.
    }
    if (realspace_reverse_strategy_
        == RealspaceReverseStrategy::pair_materialized)
        ensure_realspace_materialized_tables(cache);
    if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward
        && !delegate_full_materialized_reverse
        && !admit_realspace_full_materialized(cache.n_nodes, true)) {
        // A rejected forced experiment remains on bounded real-space work.
    }
    if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward
        && !delegate_full_materialized_reverse)
        ensure_realspace_full_materialized_tables(cache, true);
    const bool pair_parallel =
        realspace_reverse_strategy_ != RealspaceReverseStrategy::owner;
    const std::size_t atomic_accumulator_bytes = pair_parallel
        ? checked_realspace_atomic_accumulator_bytes(cache.n_nodes)
        : 0;

    RealspaceAdjointKernelResult result;
    result.grad_multipoles = use_workspace
        ? workspace_matrix("long_range_realspace_feature_grad_multipoles",
              cache.n_nodes, density_width_)
        : LongRangeKernelMatrix<Scalar>(
              "long_range_realspace_feature_grad_multipoles",
              cache.n_nodes, density_width_);
    result.grad_positions = use_workspace
        ? workspace_realspace_compute_matrix(
              "long_range_realspace_feature_grad_positions", cache.n_nodes, 3)
        : LongRangeKernelMatrix<double>(
              "long_range_realspace_feature_grad_positions", cache.n_nodes, 3);
    result.grad_cell = use_workspace
        ? workspace_realspace_compute_tensor3(
              "long_range_realspace_feature_grad_cell", cache.n_graphs, 3, 3)
        : LongRangeKernelTensor3<double>(
              "long_range_realspace_feature_grad_cell", cache.n_graphs, 3, 3);
    Kokkos::deep_copy(result.grad_cell, 0.0);

    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t density_width = density_width_;
    const std::size_t n_sigma = feature_smearing_widths_.size();
    const std::size_t source_replicas = density_max_l_ == 0 ? 1 : 4;
    const std::size_t receiver_replicas = feature_max_l_ == 0 ? 1 : 4;
    const int density_max_l = density_max_l_;
    const int feature_max_l = feature_max_l_;
    const std::size_t self_count = si_feature_constants_kernel_.extent(0);
    constexpr std::size_t pair_tile = realspace_pair_tile();
    constexpr double displacement = 0.1;
    constexpr double constant = field_constant / (4.0 * pi);
    auto positions = cache.realspace_positions;
    auto node_graph = cache.node_graph_index;
    auto effective_widths = realspace_effective_widths_;
    auto l0_factors = realspace_l0_factors_;
    auto l1_factors = realspace_l1_factors_;
    auto self_constants = si_feature_constants_kernel_;
    auto self_select = si_feature_select_kernel_;
    auto grad_multipoles = result.grad_multipoles;
    auto grad_positions = result.grad_positions;

    ++realspace_reverse_launch_diagnostics_.feature_selected_count;
    if (realspace_reverse_strategy_ == RealspaceReverseStrategy::pair_tile) {
        validate_realspace_pair_tile_backend();
        const std::size_t tile_width = realspace_pair_tile_width_;
        const std::size_t tiles_per_source = n_nodes / tile_width
            + static_cast<std::size_t>(n_nodes % tile_width != 0);
        const std::size_t league_extent =
            checked_realspace_pair_tile_feature_league_extent(n_nodes);
        const std::size_t lane_values = density_width + 3;
        const std::size_t scratch_bytes =
            checked_realspace_pair_tile_scratch_bytes();
        auto atomic_grad_multipoles = use_workspace
            ? workspace_realspace_compute_matrix(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width)
            : LongRangeKernelMatrix<double>(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width);
        Kokkos::deep_copy(atomic_grad_multipoles, 0.0);
        Kokkos::deep_copy(grad_positions, 0.0);
        ++realspace_reverse_launch_diagnostics_
              .feature_pair_tile_selected_count;
        realspace_reverse_launch_diagnostics_.feature_pair_tile_league_extent =
            league_extent;
        realspace_reverse_launch_diagnostics_.tile_width = tile_width;
        realspace_reverse_launch_diagnostics_.partial_accumulator_bytes =
            scratch_bytes;
        realspace_reverse_launch_diagnostics_.atomic_accumulator_bytes =
            atomic_accumulator_bytes;

        using TeamPolicy = Kokkos::TeamPolicy<LongRangeKernelSpace>;
        TeamPolicy policy(league_extent, tile_width);
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
        Kokkos::parallel_for(
            "long_range realspace feature team tiled directed pairs", policy,
            KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
                const std::size_t league = team.league_rank();
                const std::size_t sigma = league % n_sigma;
                const std::size_t source_tile = league / n_sigma;
                const std::size_t source = source_tile / tiles_per_source;
                const std::size_t tile =
                    source_tile - source * tiles_per_source;
                const std::size_t lane = team.team_rank();
                const std::size_t receiver = tile * tile_width + lane;
                double* scratch = static_cast<double*>(
                    team.team_shmem().get_shmem(scratch_bytes));
                double* lane_scratch = scratch + lane * lane_values;
                for (std::size_t value = 0; value < lane_values; ++value)
                    lane_scratch[value] = 0.0;
                if (receiver < n_nodes && receiver != source
                    && node_graph(source) == node_graph(receiver)) {
                    double sample_seeds[4] = {0.0, 0.0, 0.0, 0.0};
                    sample_seeds[0] =
                        static_cast<double>(grad_features(receiver, sigma))
                        * l0_factors(sigma);
                    if (feature_max_l == 1) {
                        for (std::size_t component = 0; component < 3;
                            ++component) {
                            const std::size_t output =
                                n_sigma + 3 * sigma + component;
                            const double seed = static_cast<double>(
                                                    grad_features(
                                                        receiver, output))
                                * l1_factors(sigma);
                            const std::size_t receiver_replica = component == 0
                                ? 2
                                : (component == 1 ? 3 : 1);
                            sample_seeds[receiver_replica] += seed;
                            sample_seeds[0] -= seed;
                        }
                    }
                    for (std::size_t source_replica = 0;
                        source_replica < source_replicas; ++source_replica) {
                        const double charge = realspace_replica_charge(
                            multipoles, source, source_replica, density_max_l,
                            displacement);
                        for (std::size_t receiver_replica = 0;
                            receiver_replica < receiver_replicas;
                            ++receiver_replica) {
                            const double sample_seed =
                                sample_seeds[receiver_replica];
                            if (sample_seed == 0.0)
                                continue;
                            double difference[3];
                            double distance_squared = 0.0;
                            for (std::size_t cartesian = 0; cartesian < 3;
                                ++cartesian) {
                                const double source_offset = source_replica > 0
                                        && cartesian == source_replica - 1
                                    ? displacement
                                    : 0.0;
                                const double receiver_offset =
                                    receiver_replica > 0
                                        && cartesian == receiver_replica - 1
                                    ? displacement
                                    : 0.0;
                                difference[cartesian] =
                                    realspace_displaced_difference(
                                        positions(source, cartesian),
                                        source_offset,
                                        positions(receiver, cartesian),
                                        receiver_offset);
                                distance_squared = realspace_norm_accumulate(
                                    difference[cartesian],
                                    difference[cartesian], distance_squared);
                            }
                            const double distance =
                                realspace_sqrt(distance_squared);
                            const double weighted_seed = sample_seed * constant;
                            const double radial = realspace_kernel(
                                distance, effective_widths(sigma));
                            for (std::size_t multipole = 0;
                                multipole < density_width; ++multipole)
                                lane_scratch[multipole] += weighted_seed
                                    * realspace_replica_transform_entry(
                                        density_max_l, source_replica,
                                        multipole, displacement)
                                    * radial;
                            const double position_factor =
                                weighted_seed * charge
                                * realspace_kernel_gradient_factor(
                                    distance, effective_widths(sigma));
                            for (std::size_t cartesian = 0; cartesian < 3;
                                ++cartesian)
                                lane_scratch[density_width + cartesian] +=
                                    position_factor * difference[cartesian];
                        }
                    }
                    for (std::size_t cartesian = 0; cartesian < 3;
                        ++cartesian)
                        Kokkos::atomic_add(
                            &grad_positions(receiver, cartesian),
                            -lane_scratch[density_width + cartesian]);
                }
                team.team_barrier();
                if (lane == 0) {
                    for (std::size_t other = 0; other < tile_width; ++other) {
                        const double* other_scratch =
                            scratch + other * lane_values;
                        for (std::size_t multipole = 0;
                            multipole < density_width; ++multipole)
                            Kokkos::atomic_add(
                                &atomic_grad_multipoles(source, multipole),
                                other_scratch[multipole]);
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian)
                            Kokkos::atomic_add(
                                &grad_positions(source, cartesian),
                                other_scratch[density_width + cartesian]);
                    }
                }
            });
        Kokkos::parallel_for(
            "long_range realspace feature tile narrow and self",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(
                0, n_nodes * density_width),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t source = flat / density_width;
                const std::size_t multipole = flat % density_width;
                double total = atomic_grad_multipoles(source, multipole);
                if (include_self_interaction) {
                    for (std::size_t output = 0; output < self_count; ++output)
                        if (self_select(output) == multipole)
                            total += self_constants(output)
                                * static_cast<double>(
                                    grad_features(source, output));
                }
                grad_multipoles(source, multipole) =
                    static_cast<Scalar>(total);
            });
        return result;
    }
    if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward
        && !delegate_full_materialized_reverse) {
        const std::size_t edge_extent = full_feature_sender_.extent(0);
        auto atomic_grad_multipoles = use_workspace
            ? workspace_realspace_compute_matrix(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width)
            : LongRangeKernelMatrix<double>(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width);
        Kokkos::deep_copy(atomic_grad_multipoles, 0.0);
        Kokkos::deep_copy(grad_positions, 0.0);
        ++realspace_reverse_launch_diagnostics_
              .feature_materialized_selected_count;
        realspace_reverse_launch_diagnostics_.feature_pair_league_extent =
            edge_extent;
        realspace_reverse_launch_diagnostics_.atomic_accumulator_bytes =
            atomic_accumulator_bytes;
        auto edge_sender = full_feature_sender_;
        auto edge_receiver = full_feature_receiver_;
        auto edge_valid = full_feature_valid_;
        auto edge_difference = full_feature_difference_;
        auto edge_k = full_feature_k_;
        auto edge_g = full_feature_g_;
        Kokkos::parallel_for(
            "long_range realspace full materialized feature edges",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(0, edge_extent * n_sigma),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t sigma = flat % n_sigma;
                const std::size_t edge = flat / n_sigma;
                if (!edge_valid(edge))
                    return;
                const std::size_t sender_replica_flat = edge_sender(edge);
                const std::size_t receiver_replica_flat = edge_receiver(edge);
                const std::size_t source =
                    sender_replica_flat / source_replicas;
                const std::size_t source_replica =
                    sender_replica_flat % source_replicas;
                const std::size_t receiver =
                    receiver_replica_flat / receiver_replicas;
                const std::size_t receiver_replica =
                    receiver_replica_flat % receiver_replicas;
                double sample_seed = 0.0;
                if (receiver_replica == 0) {
                    sample_seed += static_cast<double>(
                        grad_features(receiver, sigma)) * l0_factors(sigma);
                    if (feature_max_l == 1)
                        for (std::size_t component = 0; component < 3;
                            ++component)
                            sample_seed -= static_cast<double>(grad_features(
                                receiver, n_sigma + 3 * sigma + component))
                                * l1_factors(sigma);
                } else if (feature_max_l == 1) {
                    const std::size_t component = receiver_replica == 2
                        ? 0 : (receiver_replica == 3 ? 1 : 2);
                    sample_seed += static_cast<double>(grad_features(receiver,
                        n_sigma + 3 * sigma + component)) * l1_factors(sigma);
                }
                if (sample_seed == 0.0)
                    return;
                const double weighted_seed = sample_seed * constant;
                const double radial = edge_k(edge, sigma);
                for (std::size_t multipole = 0;
                    multipole < density_width; ++multipole)
                    Kokkos::atomic_add(
                        &atomic_grad_multipoles(source, multipole),
                        weighted_seed
                            * realspace_replica_transform_entry(density_max_l,
                                source_replica, multipole, displacement)
                            * radial);
                const double charge = realspace_replica_charge(multipoles,
                    source, source_replica, density_max_l, displacement);
                const double factor = weighted_seed * charge
                    * edge_g(edge, sigma);
                for (std::size_t cartesian = 0; cartesian < 3; ++cartesian) {
                    const double dx = factor * edge_difference(edge, cartesian);
                    Kokkos::atomic_add(
                        &grad_positions(source, cartesian), dx);
                    Kokkos::atomic_add(
                        &grad_positions(receiver, cartesian), -dx);
                }
            });
        Kokkos::parallel_for(
            "long_range realspace full materialized feature narrow and self",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(
                0, n_nodes * density_width),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t source = flat / density_width;
                const std::size_t multipole = flat % density_width;
                double total = atomic_grad_multipoles(source, multipole);
                if (include_self_interaction)
                    for (std::size_t output = 0; output < self_count; ++output)
                        if (self_select(output) == multipole)
                            total += self_constants(output)
                                * static_cast<double>(
                                    grad_features(source, output));
                grad_multipoles(source, multipole) =
                    static_cast<Scalar>(total);
            });
        return result;
    }
    if (realspace_reverse_strategy_ == RealspaceReverseStrategy::pair_atomic
        || realspace_reverse_strategy_
            == RealspaceReverseStrategy::pair_materialized
        || delegate_full_materialized_reverse) {
        const bool use_materialized = realspace_reverse_strategy_
            == RealspaceReverseStrategy::pair_materialized;
        if (n_nodes != 0
            && n_nodes > std::numeric_limits<std::size_t>::max() / n_nodes)
            throw std::overflow_error(
                "adjoint_realspace_features: pair range overflows size_t.");
        const std::size_t pair_count = n_nodes * n_nodes;
        if (n_sigma != 0
            && pair_count
                > std::numeric_limits<std::size_t>::max() / n_sigma)
            throw std::overflow_error(
                "adjoint_realspace_features: pair-width range overflows "
                "size_t.");
        const std::size_t pair_extent = pair_count * n_sigma;
        auto atomic_grad_multipoles = use_workspace
            ? workspace_realspace_compute_matrix(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width)
            : LongRangeKernelMatrix<double>(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width);
        Kokkos::deep_copy(atomic_grad_multipoles, 0.0);
        Kokkos::deep_copy(grad_positions, 0.0);
        if (use_materialized)
            ++realspace_reverse_launch_diagnostics_
                  .feature_materialized_selected_count;
        else {
            ++realspace_reverse_launch_diagnostics_
                  .feature_pair_atomic_selected_count;
            if (delegate_full_materialized_reverse)
                ++realspace_reverse_launch_diagnostics_
                      .full_materialized_feature_pair_atomic_reverse_count;
        }
        realspace_reverse_launch_diagnostics_.feature_pair_league_extent =
            pair_extent;
        realspace_reverse_launch_diagnostics_.atomic_accumulator_bytes =
            atomic_accumulator_bytes;
        auto materialized_k = materialized_feature_k_;
        auto materialized_g = materialized_feature_g_;

        Kokkos::parallel_for(
            "long_range realspace feature fused directed pairs",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(0, pair_extent),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t sigma = flat % n_sigma;
                const std::size_t pair = flat / n_sigma;
                const std::size_t source = pair / n_nodes;
                const std::size_t receiver = pair - source * n_nodes;
                if (source == receiver
                    || node_graph(source) != node_graph(receiver))
                    return;
                double sample_seeds[4] = {0.0, 0.0, 0.0, 0.0};
                sample_seeds[0] =
                    static_cast<double>(grad_features(receiver, sigma))
                    * l0_factors(sigma);
                if (feature_max_l == 1) {
                    for (std::size_t component = 0; component < 3;
                        ++component) {
                        const std::size_t output =
                            n_sigma + 3 * sigma + component;
                        const double seed = static_cast<double>(
                                                grad_features(receiver, output))
                            * l1_factors(sigma);
                        const std::size_t receiver_replica = component == 0
                            ? 2
                            : (component == 1 ? 3 : 1);
                        sample_seeds[receiver_replica] += seed;
                        sample_seeds[0] -= seed;
                    }
                }
                double dm[4] = {0.0, 0.0, 0.0, 0.0};
                double dx[3] = {0.0, 0.0, 0.0};
                for (std::size_t source_replica = 0;
                    source_replica < source_replicas; ++source_replica) {
                    const double charge = realspace_replica_charge(multipoles,
                        source, source_replica, density_max_l, displacement);
                    for (std::size_t receiver_replica = 0;
                        receiver_replica < receiver_replicas;
                        ++receiver_replica) {
                        const double sample_seed =
                            sample_seeds[receiver_replica];
                        if (sample_seed == 0.0)
                            continue;
                        double difference[3];
                        double distance_squared = 0.0;
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian) {
                            const double source_offset = source_replica > 0
                                    && cartesian == source_replica - 1
                                ? displacement
                                : 0.0;
                            const double receiver_offset = receiver_replica > 0
                                    && cartesian == receiver_replica - 1
                                ? displacement
                                : 0.0;
                            difference[cartesian] =
                                realspace_displaced_difference(
                                    positions(source, cartesian), source_offset,
                                    positions(receiver, cartesian),
                                    receiver_offset);
                            distance_squared = realspace_norm_accumulate(
                                difference[cartesian], difference[cartesian],
                                distance_squared);
                        }
                        const double distance =
                            realspace_sqrt(distance_squared);
                        const double weighted_seed = sample_seed * constant;
                        double radial;
                        double gradient_factor;
                        if (use_materialized) {
                            const std::size_t left = source < receiver
                                ? source
                                : receiver;
                            const std::size_t right = source < receiver
                                ? receiver
                                : source;
                            const std::size_t material_pair = left * n_nodes
                                - left * (left + 1) / 2
                                + (right - left - 1);
                            const std::size_t orientation =
                                source == left ? 0 : 1;
                            const std::size_t record =
                                ((((material_pair * 2 + orientation) * n_sigma
                                      + sigma)
                                      * source_replicas
                                    + source_replica)
                                      * receiver_replicas
                                    + receiver_replica);
                            radial = materialized_k(record);
                            gradient_factor = materialized_g(record);
                        } else {
                            radial = realspace_kernel(
                                distance, effective_widths(sigma));
                            gradient_factor =
                                realspace_kernel_gradient_factor(
                                    distance, effective_widths(sigma));
                        }
                        for (std::size_t multipole = 0;
                            multipole < density_width; ++multipole)
                            dm[multipole] += weighted_seed
                                * realspace_replica_transform_entry(
                                    density_max_l, source_replica, multipole,
                                    displacement)
                                * radial;
                        const double position_factor =
                            weighted_seed * charge * gradient_factor;
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian)
                            dx[cartesian] +=
                                position_factor * difference[cartesian];
                    }
                }
                for (std::size_t multipole = 0;
                    multipole < density_width; ++multipole)
                    Kokkos::atomic_add(
                        &atomic_grad_multipoles(source, multipole),
                        dm[multipole]);
                for (std::size_t cartesian = 0; cartesian < 3; ++cartesian) {
                    Kokkos::atomic_add(
                        &grad_positions(source, cartesian), dx[cartesian]);
                    Kokkos::atomic_add(
                        &grad_positions(receiver, cartesian), -dx[cartesian]);
                }
            });
        Kokkos::parallel_for(
            "long_range realspace feature atomic narrow and self",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(
                0, n_nodes * density_width),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t source = flat / density_width;
                const std::size_t multipole = flat % density_width;
                double total = atomic_grad_multipoles(source, multipole);
                if (include_self_interaction) {
                    for (std::size_t output = 0; output < self_count; ++output)
                        if (self_select(output) == multipole)
                            total += self_constants(output)
                                * static_cast<double>(
                                    grad_features(source, output));
                }
                grad_multipoles(source, multipole) =
                    static_cast<Scalar>(total);
            });
        return result;
    }

    realspace_reverse_launch_diagnostics_.feature_multipole_league_extent =
        n_nodes * density_width;
    realspace_reverse_launch_diagnostics_.feature_position_league_extent =
        n_nodes * 3;
    ++realspace_reverse_launch_diagnostics_.feature_owner_selected_count;

    Kokkos::parallel_for(
        "long_range realspace feature multipole adjoint owners",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes * density_width),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t source = flat / density_width;
            const std::size_t multipole = flat % density_width;
            double total = 0.0;
            for (std::size_t tile_begin = 0; tile_begin < n_nodes;
                tile_begin += pair_tile) {
                const std::size_t tile_end = tile_begin + pair_tile < n_nodes
                    ? tile_begin + pair_tile
                    : n_nodes;
                for (std::size_t receiver = tile_begin; receiver < tile_end;
                    ++receiver) {
                    if (receiver == source
                        || node_graph(receiver) != node_graph(source))
                        continue;
                    for (std::size_t sigma = 0; sigma < n_sigma; ++sigma) {
                        double sample_seeds[4] = {0.0, 0.0, 0.0, 0.0};
                        sample_seeds[0] =
                            static_cast<double>(grad_features(receiver, sigma))
                            * l0_factors(sigma);
                        if (feature_max_l == 1) {
                            for (std::size_t component = 0; component < 3;
                                ++component) {
                                const std::size_t output =
                                    n_sigma + 3 * sigma + component;
                                const double seed = static_cast<double>(
                                                        grad_features(
                                                            receiver, output))
                                    * l1_factors(sigma);
                                const std::size_t receiver_replica =
                                    component == 0
                                    ? 2
                                    : (component == 1 ? 3 : 1);
                                sample_seeds[receiver_replica] += seed;
                                sample_seeds[0] -= seed;
                            }
                        }
                        for (std::size_t source_replica = 0;
                            source_replica < source_replicas;
                            ++source_replica) {
                            const double transform =
                                realspace_replica_transform_entry(
                                    density_max_l, source_replica, multipole,
                                    displacement);
                            if (transform == 0.0)
                                continue;
                            for (std::size_t receiver_replica = 0;
                                receiver_replica < receiver_replicas;
                                ++receiver_replica) {
                                if (sample_seeds[receiver_replica] == 0.0)
                                    continue;
                                double distance_squared = 0.0;
                                for (std::size_t cartesian = 0; cartesian < 3;
                                    ++cartesian) {
                                    const double source_offset =
                                        source_replica > 0
                                            && cartesian == source_replica - 1
                                        ? displacement
                                        : 0.0;
                                    const double receiver_offset =
                                        receiver_replica > 0
                                            && cartesian
                                                == receiver_replica - 1
                                        ? displacement
                                        : 0.0;
                                    const double difference =
                                        positions(source, cartesian)
                                        + source_offset
                                        - positions(receiver, cartesian)
                                        - receiver_offset;
                                    distance_squared =
                                        realspace_norm_accumulate(difference,
                                            difference, distance_squared);
                                }
                                const double distance =
                                    realspace_sqrt(distance_squared);
                                total += sample_seeds[receiver_replica]
                                    * constant * transform
                                    * realspace_kernel(
                                        distance, effective_widths(sigma));
                            }
                        }
                    }
                }
            }
            if (include_self_interaction) {
                for (std::size_t output = 0; output < self_count; ++output)
                    if (self_select(output) == multipole)
                        total += self_constants(output)
                            * static_cast<double>(
                                grad_features(source, output));
            }
            grad_multipoles(source, multipole) = static_cast<Scalar>(total);
        });

    Kokkos::parallel_for(
        "long_range realspace feature position adjoint owners",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes * 3),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t owner = flat / 3;
            const std::size_t owner_cartesian = flat % 3;
            double total = 0.0;

            // The owner acts as the source atom. Every replica position owns
            // the same base Cartesian coordinate, so their seeds add here.
            for (std::size_t tile_begin = 0; tile_begin < n_nodes;
                tile_begin += pair_tile) {
                const std::size_t tile_end = tile_begin + pair_tile < n_nodes
                    ? tile_begin + pair_tile
                    : n_nodes;
                for (std::size_t receiver = tile_begin; receiver < tile_end;
                    ++receiver) {
                    if (receiver == owner
                        || node_graph(receiver) != node_graph(owner))
                        continue;
                    for (std::size_t sigma = 0; sigma < n_sigma; ++sigma) {
                        double sample_seeds[4] = {0.0, 0.0, 0.0, 0.0};
                        sample_seeds[0] =
                            static_cast<double>(grad_features(receiver, sigma))
                            * l0_factors(sigma);
                        if (feature_max_l == 1) {
                            for (std::size_t component = 0; component < 3;
                                ++component) {
                                const std::size_t output =
                                    n_sigma + 3 * sigma + component;
                                const double seed = static_cast<double>(
                                                        grad_features(
                                                            receiver, output))
                                    * l1_factors(sigma);
                                const std::size_t receiver_replica =
                                    component == 0
                                    ? 2
                                    : (component == 1 ? 3 : 1);
                                sample_seeds[receiver_replica] += seed;
                                sample_seeds[0] -= seed;
                            }
                        }
                        for (std::size_t source_replica = 0;
                            source_replica < source_replicas;
                            ++source_replica) {
                            const double charge = realspace_replica_charge(
                                multipoles, owner, source_replica,
                                density_max_l, displacement);
                            for (std::size_t receiver_replica = 0;
                                receiver_replica < receiver_replicas;
                                ++receiver_replica) {
                                if (sample_seeds[receiver_replica] == 0.0)
                                    continue;
                                double distance_squared = 0.0;
                                double selected_difference = 0.0;
                                for (std::size_t cartesian = 0; cartesian < 3;
                                    ++cartesian) {
                                    const double source_offset =
                                        source_replica > 0
                                            && cartesian == source_replica - 1
                                        ? displacement
                                        : 0.0;
                                    const double receiver_offset =
                                        receiver_replica > 0
                                            && cartesian
                                                == receiver_replica - 1
                                        ? displacement
                                        : 0.0;
                                    const double difference =
                                        realspace_displaced_difference(
                                            positions(owner, cartesian),
                                            source_offset,
                                            positions(receiver, cartesian),
                                            receiver_offset);
                                    if (cartesian == owner_cartesian)
                                        selected_difference = difference;
                                    distance_squared =
                                        realspace_norm_accumulate(difference,
                                            difference, distance_squared);
                                }
                                const double distance =
                                    realspace_sqrt(distance_squared);
                                total += sample_seeds[receiver_replica]
                                    * constant * charge
                                    * realspace_kernel_gradient_factor(
                                        distance, effective_widths(sigma))
                                    * selected_difference;
                            }
                        }
                    }
                }
            }

            // The owner acts as receiver. Its replica-coordinate derivative
            // is the exact negative of the source displacement derivative.
            for (std::size_t tile_begin = 0; tile_begin < n_nodes;
                tile_begin += pair_tile) {
                const std::size_t tile_end = tile_begin + pair_tile < n_nodes
                    ? tile_begin + pair_tile
                    : n_nodes;
                for (std::size_t source = tile_begin; source < tile_end;
                    ++source) {
                    if (source == owner
                        || node_graph(source) != node_graph(owner))
                        continue;
                    for (std::size_t sigma = 0; sigma < n_sigma; ++sigma) {
                        double sample_seeds[4] = {0.0, 0.0, 0.0, 0.0};
                        sample_seeds[0] =
                            static_cast<double>(grad_features(owner, sigma))
                            * l0_factors(sigma);
                        if (feature_max_l == 1) {
                            for (std::size_t component = 0; component < 3;
                                ++component) {
                                const std::size_t output =
                                    n_sigma + 3 * sigma + component;
                                const double seed = static_cast<double>(
                                                        grad_features(
                                                            owner, output))
                                    * l1_factors(sigma);
                                const std::size_t receiver_replica =
                                    component == 0
                                    ? 2
                                    : (component == 1 ? 3 : 1);
                                sample_seeds[receiver_replica] += seed;
                                sample_seeds[0] -= seed;
                            }
                        }
                        for (std::size_t source_replica = 0;
                            source_replica < source_replicas;
                            ++source_replica) {
                            const double charge = realspace_replica_charge(
                                multipoles, source, source_replica,
                                density_max_l, displacement);
                            for (std::size_t receiver_replica = 0;
                                receiver_replica < receiver_replicas;
                                ++receiver_replica) {
                                if (sample_seeds[receiver_replica] == 0.0)
                                    continue;
                                double distance_squared = 0.0;
                                double selected_difference = 0.0;
                                for (std::size_t cartesian = 0; cartesian < 3;
                                    ++cartesian) {
                                    const double source_offset =
                                        source_replica > 0
                                            && cartesian == source_replica - 1
                                        ? displacement
                                        : 0.0;
                                    const double receiver_offset =
                                        receiver_replica > 0
                                            && cartesian
                                                == receiver_replica - 1
                                        ? displacement
                                        : 0.0;
                                    const double difference =
                                        realspace_displaced_difference(
                                            positions(source, cartesian),
                                            source_offset,
                                            positions(owner, cartesian),
                                            receiver_offset);
                                    if (cartesian == owner_cartesian)
                                        selected_difference = difference;
                                    distance_squared =
                                        realspace_norm_accumulate(difference,
                                            difference, distance_squared);
                                }
                                const double distance =
                                    realspace_sqrt(distance_squared);
                                total -= sample_seeds[receiver_replica]
                                    * constant * charge
                                    * realspace_kernel_gradient_factor(
                                        distance, effective_widths(sigma))
                                    * selected_difference;
                            }
                        }
                    }
                }
            }
            grad_positions(owner, owner_cartesian) = total;
        });
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::RealspaceAdjointDiagnosticResult
PolarLongRangeOpsT<Scalar>::diagnostic_adjoint_realspace_features(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& multipoles,
    const LongRangeConstMatrix<double>& grad_features,
    const bool include_self_interaction,
    const bool use_workspace) const
{
    if (use_workspace)
        prepare_workspace(0, cache.n_nodes, cache.n_graphs);
    auto make_matrix = [this, use_workspace] (const char* label,
                           const std::size_t rows,
                           const std::size_t columns) {
        return use_workspace ? workspace_matrix(label, rows, columns)
                             : LongRangeKernelMatrix<Scalar>(
                                   Kokkos::view_alloc(std::string(label)), rows,
                                   columns);
    };
    auto multipoles_kernel = make_matrix(
        "long_range_realspace_feature_adjoint_multipoles",
        multipoles.extent(0), multipoles.extent(1));
    auto grad_features_kernel = make_matrix(
        "long_range_realspace_feature_adjoint_seed", grad_features.extent(0),
        grad_features.extent(1));
    stage_matrix_into<Scalar>(multipoles, multipoles_kernel);
    stage_matrix_into<Scalar>(grad_features, grad_features_kernel);
    const RealspaceAdjointKernelResult kernel =
        adjoint_realspace_features_kernel(cache, multipoles_kernel,
            grad_features_kernel, include_self_interaction, use_workspace);
    RealspaceAdjointDiagnosticResult result;
    result.grad_multipoles = download_matrix<double>(
        kernel.grad_multipoles,
        "long_range_realspace_feature_grad_multipoles_host");
    result.grad_positions = download_matrix<double>(kernel.grad_positions,
        "long_range_realspace_feature_grad_positions_host");
    result.grad_cell = download_tensor3<double>(
        kernel.grad_cell, "long_range_realspace_feature_grad_cell_host");
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::RealspaceAdjointKernelResult
PolarLongRangeOpsT<Scalar>::adjoint_realspace_energy_kernel(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& multipoles,
    const LongRangeKernelVector<Scalar>& grad_energy,
    const bool include_self_interaction,
    const bool use_workspace) const
{
    if (cache.mode != LongRangeGeometryMode::realspace)
        throw std::invalid_argument(
            "adjoint_realspace_energy requires a real-space geometry cache.");
    if (multipoles.extent(0) != cache.n_nodes
        || multipoles.extent(1) != density_width_)
        throw std::invalid_argument(
            "adjoint_realspace_energy: multipoles must be "
            "[nodes, density_width].");
    if (grad_energy.extent(0) != cache.n_graphs)
        throw std::invalid_argument(
            "adjoint_realspace_energy: energy seeds must carry n_graphs.");
    if (density_width_ != 0
        && cache.n_nodes
            > std::numeric_limits<std::size_t>::max() / density_width_)
        throw std::overflow_error(
            "adjoint_realspace_energy: multipole owner range overflows "
            "size_t.");
    if (cache.n_nodes > std::numeric_limits<std::size_t>::max() / 3)
        throw std::overflow_error(
            "adjoint_realspace_energy: position owner range overflows "
            "size_t.");

    resolve_realspace_reverse_strategy();
    const bool delegate_full_materialized_reverse =
        realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward
        && realspace_full_materialized_reducer_
            != RealspaceFullMaterializedReducer::serial;

    if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::pair_materialized
        && !admit_realspace_materialized(cache.n_nodes)) {
        // Keep the resolved bounded fallback selected by admission.
    }
    if (realspace_reverse_strategy_
        == RealspaceReverseStrategy::pair_materialized)
        ensure_realspace_materialized_tables(cache);
    if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward
        && !delegate_full_materialized_reverse
        && !admit_realspace_full_materialized(cache.n_nodes, true)) {
        // Keep the pair-atomic fallback chosen by admission.
    }
    if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward
        && !delegate_full_materialized_reverse)
        ensure_realspace_full_materialized_tables(cache, true);
    const bool pair_parallel =
        realspace_reverse_strategy_ != RealspaceReverseStrategy::owner;
    const std::size_t atomic_accumulator_bytes = pair_parallel
        ? checked_realspace_atomic_accumulator_bytes(cache.n_nodes)
        : 0;

    RealspaceAdjointKernelResult result;
    result.grad_multipoles = use_workspace
        ? workspace_matrix("long_range_realspace_energy_grad_multipoles",
              cache.n_nodes, density_width_)
        : LongRangeKernelMatrix<Scalar>(
              "long_range_realspace_energy_grad_multipoles", cache.n_nodes,
              density_width_);
    result.grad_positions = use_workspace
        ? workspace_realspace_compute_matrix(
              "long_range_realspace_energy_grad_positions", cache.n_nodes, 3)
        : LongRangeKernelMatrix<double>(
              "long_range_realspace_energy_grad_positions", cache.n_nodes, 3);
    result.grad_cell = use_workspace
        ? workspace_realspace_compute_tensor3(
              "long_range_realspace_energy_grad_cell", cache.n_graphs, 3, 3)
        : LongRangeKernelTensor3<double>(
              "long_range_realspace_energy_grad_cell", cache.n_graphs, 3, 3);
    Kokkos::deep_copy(result.grad_cell, 0.0);

    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t density_width = density_width_;
    const std::size_t replicas = density_max_l_ == 0 ? 1 : 4;
    const int density_max_l = density_max_l_;
    constexpr std::size_t pair_tile = realspace_pair_tile();
    constexpr double displacement = 0.02;
    constexpr double constant = field_constant / (4.0 * pi);
    const double density_smearing_width = density_smearing_width_;
    auto positions = cache.realspace_positions;
    auto node_graph = cache.node_graph_index;
    auto self_constants = si_energy_constants_kernel_;
    auto self_select = si_energy_select_kernel_;
    auto grad_multipoles = result.grad_multipoles;
    auto grad_positions = result.grad_positions;

    ++realspace_reverse_launch_diagnostics_.energy_selected_count;
    if (realspace_reverse_strategy_ == RealspaceReverseStrategy::pair_tile) {
        validate_realspace_pair_tile_backend();
        const std::size_t tile_width = realspace_pair_tile_width_;
        const std::size_t tiles_per_left = n_nodes / tile_width
            + static_cast<std::size_t>(n_nodes % tile_width != 0);
        const std::size_t league_extent =
            checked_realspace_pair_tile_energy_league_extent(n_nodes);
        const std::size_t lane_values = density_width + 3;
        const std::size_t scratch_bytes =
            checked_realspace_pair_tile_scratch_bytes();
        auto atomic_grad_multipoles = use_workspace
            ? workspace_realspace_compute_matrix(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width)
            : LongRangeKernelMatrix<double>(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width);
        Kokkos::deep_copy(atomic_grad_multipoles, 0.0);
        Kokkos::deep_copy(grad_positions, 0.0);
        ++realspace_reverse_launch_diagnostics_
              .energy_pair_tile_selected_count;
        realspace_reverse_launch_diagnostics_.energy_pair_tile_league_extent =
            league_extent;
        realspace_reverse_launch_diagnostics_.tile_width = tile_width;
        realspace_reverse_launch_diagnostics_.partial_accumulator_bytes =
            scratch_bytes;
        realspace_reverse_launch_diagnostics_.atomic_accumulator_bytes =
            atomic_accumulator_bytes;

        using TeamPolicy = Kokkos::TeamPolicy<LongRangeKernelSpace>;
        TeamPolicy policy(league_extent, tile_width);
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
        Kokkos::parallel_for(
            "long_range realspace energy team tiled unordered pairs", policy,
            KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
                const std::size_t league = team.league_rank();
                const std::size_t left = league / tiles_per_left;
                const std::size_t tile = league - left * tiles_per_left;
                const std::size_t lane = team.team_rank();
                const std::size_t right = tile * tile_width + lane;
                double* scratch = static_cast<double*>(
                    team.team_shmem().get_shmem(scratch_bytes));
                double* lane_scratch = scratch + lane * lane_values;
                for (std::size_t value = 0; value < lane_values; ++value)
                    lane_scratch[value] = 0.0;
                if (right < n_nodes && right > left
                    && node_graph(left) == node_graph(right)) {
                    const std::size_t graph =
                        static_cast<std::size_t>(node_graph(left));
                    const double energy_seed =
                        static_cast<double>(grad_energy(graph));
                    double dm_right[4] = {0.0, 0.0, 0.0, 0.0};
                    for (std::size_t left_replica = 0;
                        left_replica < replicas; ++left_replica) {
                        const double left_charge = realspace_replica_charge(
                            multipoles, left, left_replica, density_max_l,
                            displacement);
                        for (std::size_t right_replica = 0;
                            right_replica < replicas; ++right_replica) {
                            const double right_charge =
                                realspace_replica_charge(multipoles, right,
                                    right_replica, density_max_l,
                                    displacement);
                            double difference[3];
                            double distance_squared = 0.0;
                            for (std::size_t cartesian = 0; cartesian < 3;
                                ++cartesian) {
                                const double left_offset = left_replica > 0
                                        && cartesian == left_replica - 1
                                    ? displacement
                                    : 0.0;
                                const double right_offset = right_replica > 0
                                        && cartesian == right_replica - 1
                                    ? displacement
                                    : 0.0;
                                difference[cartesian] =
                                    realspace_displaced_difference(
                                        positions(right, cartesian),
                                        right_offset,
                                        positions(left, cartesian),
                                        left_offset);
                                distance_squared = realspace_norm_accumulate(
                                    difference[cartesian],
                                    difference[cartesian], distance_squared);
                            }
                            const double distance =
                                realspace_sqrt(distance_squared);
                            const double radial = realspace_kernel(
                                distance, density_smearing_width);
                            const double weighted = energy_seed * constant;
                            for (std::size_t multipole = 0;
                                multipole < density_width; ++multipole) {
                                lane_scratch[multipole] += weighted
                                    * realspace_replica_transform_entry(
                                        density_max_l, left_replica,
                                        multipole, displacement)
                                    * right_charge * radial;
                                dm_right[multipole] += weighted
                                    * realspace_replica_transform_entry(
                                        density_max_l, right_replica,
                                        multipole, displacement)
                                    * left_charge * radial;
                            }
                            const double position_factor =
                                weighted * left_charge * right_charge
                                * realspace_kernel_gradient_factor(
                                    distance, density_smearing_width);
                            for (std::size_t cartesian = 0; cartesian < 3;
                                ++cartesian)
                                lane_scratch[density_width + cartesian] -=
                                    position_factor * difference[cartesian];
                        }
                    }
                    for (std::size_t multipole = 0;
                        multipole < density_width; ++multipole)
                        Kokkos::atomic_add(
                            &atomic_grad_multipoles(right, multipole),
                            dm_right[multipole]);
                    for (std::size_t cartesian = 0; cartesian < 3;
                        ++cartesian)
                        Kokkos::atomic_add(
                            &grad_positions(right, cartesian),
                            -lane_scratch[density_width + cartesian]);
                }
                team.team_barrier();
                if (lane == 0) {
                    for (std::size_t other = 0; other < tile_width; ++other) {
                        const double* other_scratch =
                            scratch + other * lane_values;
                        for (std::size_t multipole = 0;
                            multipole < density_width; ++multipole)
                            Kokkos::atomic_add(
                                &atomic_grad_multipoles(left, multipole),
                                other_scratch[multipole]);
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian)
                            Kokkos::atomic_add(
                                &grad_positions(left, cartesian),
                                other_scratch[density_width + cartesian]);
                    }
                }
            });
        Kokkos::parallel_for(
            "long_range realspace energy tile narrow and self",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(
                0, n_nodes * density_width),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t owner = flat / density_width;
                const std::size_t multipole = flat % density_width;
                const std::size_t graph =
                    static_cast<std::size_t>(node_graph(owner));
                const double energy_seed =
                    static_cast<double>(grad_energy(graph));
                double total = atomic_grad_multipoles(owner, multipole);
                if (include_self_interaction) {
                    for (std::size_t channel = 0; channel < density_width;
                        ++channel) {
                        const std::size_t selected = self_select(channel);
                        const double coefficient =
                            0.5 * self_constants(channel);
                        if (channel == multipole)
                            total += energy_seed * coefficient
                                * static_cast<double>(
                                    multipoles(owner, selected));
                        if (selected == multipole)
                            total += energy_seed * coefficient
                                * static_cast<double>(
                                    multipoles(owner, channel));
                    }
                }
                grad_multipoles(owner, multipole) =
                    static_cast<Scalar>(total);
            });
        return result;
    }
    if (realspace_reverse_strategy_
            == RealspaceReverseStrategy::materialized_forward
        && !delegate_full_materialized_reverse) {
        const std::size_t edge_extent = full_energy_sender_.extent(0);
        auto atomic_grad_multipoles = use_workspace
            ? workspace_realspace_compute_matrix(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width)
            : LongRangeKernelMatrix<double>(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width);
        Kokkos::deep_copy(atomic_grad_multipoles, 0.0);
        Kokkos::deep_copy(grad_positions, 0.0);
        ++realspace_reverse_launch_diagnostics_
              .energy_materialized_selected_count;
        realspace_reverse_launch_diagnostics_.energy_pair_league_extent =
            edge_extent;
        realspace_reverse_launch_diagnostics_.atomic_accumulator_bytes =
            atomic_accumulator_bytes;
        auto edge_sender = full_energy_sender_;
        auto edge_receiver = full_energy_receiver_;
        auto edge_valid = full_energy_valid_;
        auto edge_difference = full_energy_difference_;
        auto edge_k = full_energy_k_;
        auto edge_g = full_energy_g_;
        Kokkos::parallel_for(
            "long_range realspace full materialized energy edges",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(0, edge_extent),
            KOKKOS_LAMBDA(const std::size_t edge) {
                if (!edge_valid(edge))
                    return;
                const std::size_t sender_replica_flat = edge_sender(edge);
                const std::size_t receiver_replica_flat = edge_receiver(edge);
                const std::size_t source = sender_replica_flat / replicas;
                const std::size_t source_replica = sender_replica_flat % replicas;
                const std::size_t receiver = receiver_replica_flat / replicas;
                const std::size_t receiver_replica =
                    receiver_replica_flat % replicas;
                const std::size_t graph =
                    static_cast<std::size_t>(node_graph(source));
                const double weighted = 0.5 * constant
                    * static_cast<double>(grad_energy(graph));
                const double source_charge = realspace_replica_charge(
                    multipoles, source, source_replica, density_max_l,
                    displacement);
                const double receiver_charge = realspace_replica_charge(
                    multipoles, receiver, receiver_replica, density_max_l,
                    displacement);
                const double radial = edge_k(edge);
                for (std::size_t multipole = 0;
                    multipole < density_width; ++multipole) {
                    Kokkos::atomic_add(&atomic_grad_multipoles(source, multipole),
                        weighted * realspace_replica_transform_entry(
                            density_max_l, source_replica, multipole,
                            displacement) * receiver_charge * radial);
                    Kokkos::atomic_add(
                        &atomic_grad_multipoles(receiver, multipole),
                        weighted * realspace_replica_transform_entry(
                            density_max_l, receiver_replica, multipole,
                            displacement) * source_charge * radial);
                }
                const double factor = weighted * source_charge
                    * receiver_charge * edge_g(edge);
                for (std::size_t cartesian = 0; cartesian < 3; ++cartesian) {
                    const double dx = factor * edge_difference(edge, cartesian);
                    Kokkos::atomic_add(&grad_positions(source, cartesian), dx);
                    Kokkos::atomic_add(&grad_positions(receiver, cartesian), -dx);
                }
            });
        Kokkos::parallel_for(
            "long_range realspace full materialized energy narrow and self",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(
                0, n_nodes * density_width),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t owner = flat / density_width;
                const std::size_t multipole = flat % density_width;
                const std::size_t graph =
                    static_cast<std::size_t>(node_graph(owner));
                const double energy_seed =
                    static_cast<double>(grad_energy(graph));
                double total = atomic_grad_multipoles(owner, multipole);
                if (include_self_interaction)
                    for (std::size_t channel = 0; channel < density_width;
                        ++channel) {
                        const std::size_t selected = self_select(channel);
                        const double coefficient = 0.5 * self_constants(channel);
                        if (channel == multipole)
                            total += energy_seed * coefficient
                                * static_cast<double>(multipoles(owner, selected));
                        if (selected == multipole)
                            total += energy_seed * coefficient
                                * static_cast<double>(multipoles(owner, channel));
                    }
                grad_multipoles(owner, multipole) = static_cast<Scalar>(total);
            });
        return result;
    }
    if (realspace_reverse_strategy_ == RealspaceReverseStrategy::pair_atomic
        || realspace_reverse_strategy_
            == RealspaceReverseStrategy::pair_materialized
        || delegate_full_materialized_reverse) {
        const bool use_materialized = realspace_reverse_strategy_
            == RealspaceReverseStrategy::pair_materialized;
        if (n_nodes != 0
            && n_nodes > std::numeric_limits<std::size_t>::max() / n_nodes)
            throw std::overflow_error(
                "adjoint_realspace_energy: pair range overflows size_t.");
        const std::size_t pair_extent = n_nodes * n_nodes;
        auto atomic_grad_multipoles = use_workspace
            ? workspace_realspace_compute_matrix(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width)
            : LongRangeKernelMatrix<double>(
                  "long_range_realspace_atomic_grad_multipoles", n_nodes,
                  density_width);
        Kokkos::deep_copy(atomic_grad_multipoles, 0.0);
        Kokkos::deep_copy(grad_positions, 0.0);
        if (use_materialized)
            ++realspace_reverse_launch_diagnostics_
                  .energy_materialized_selected_count;
        else {
            ++realspace_reverse_launch_diagnostics_
                  .energy_pair_atomic_selected_count;
            if (delegate_full_materialized_reverse)
                ++realspace_reverse_launch_diagnostics_
                      .full_materialized_energy_pair_atomic_reverse_count;
        }
        realspace_reverse_launch_diagnostics_.energy_pair_league_extent =
            pair_extent;
        realspace_reverse_launch_diagnostics_.atomic_accumulator_bytes =
            atomic_accumulator_bytes;
        auto materialized_k = materialized_energy_k_;
        auto materialized_g = materialized_energy_g_;

        Kokkos::parallel_for(
            "long_range realspace energy fused unordered pairs",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(0, pair_extent),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t left = flat / n_nodes;
                const std::size_t right = flat - left * n_nodes;
                if (right <= left
                    || node_graph(left) != node_graph(right))
                    return;
                const std::size_t graph =
                    static_cast<std::size_t>(node_graph(left));
                const double energy_seed =
                    static_cast<double>(grad_energy(graph));
                double dm_left[4] = {0.0, 0.0, 0.0, 0.0};
                double dm_right[4] = {0.0, 0.0, 0.0, 0.0};
                double dx[3] = {0.0, 0.0, 0.0};
                for (std::size_t left_replica = 0;
                    left_replica < replicas; ++left_replica) {
                    const double left_charge = realspace_replica_charge(
                        multipoles, left, left_replica, density_max_l,
                        displacement);
                    for (std::size_t right_replica = 0;
                        right_replica < replicas; ++right_replica) {
                        const double right_charge = realspace_replica_charge(
                            multipoles, right, right_replica, density_max_l,
                            displacement);
                        double difference[3];
                        double distance_squared = 0.0;
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian) {
                            const double left_offset = left_replica > 0
                                    && cartesian == left_replica - 1
                                ? displacement
                                : 0.0;
                            const double right_offset = right_replica > 0
                                    && cartesian == right_replica - 1
                                ? displacement
                                : 0.0;
                            difference[cartesian] =
                                realspace_displaced_difference(
                                    positions(right, cartesian), right_offset,
                                    positions(left, cartesian), left_offset);
                            distance_squared = realspace_norm_accumulate(
                                difference[cartesian], difference[cartesian],
                                distance_squared);
                        }
                        const double distance =
                            realspace_sqrt(distance_squared);
                        double radial;
                        double gradient_factor;
                        if (use_materialized) {
                            const std::size_t material_pair = left * n_nodes
                                - left * (left + 1) / 2
                                + (right - left - 1);
                            const std::size_t record =
                                (material_pair * replicas + left_replica)
                                    * replicas
                                + right_replica;
                            radial = materialized_k(record);
                            gradient_factor = materialized_g(record);
                        } else {
                            radial = realspace_kernel(
                                distance, density_smearing_width);
                            gradient_factor =
                                realspace_kernel_gradient_factor(
                                    distance, density_smearing_width);
                        }
                        const double weighted = energy_seed * constant;
                        for (std::size_t multipole = 0;
                            multipole < density_width; ++multipole) {
                            dm_left[multipole] += weighted
                                * realspace_replica_transform_entry(
                                    density_max_l, left_replica, multipole,
                                    displacement)
                                * right_charge * radial;
                            dm_right[multipole] += weighted
                                * realspace_replica_transform_entry(
                                    density_max_l, right_replica, multipole,
                                    displacement)
                                * left_charge * radial;
                        }
                        const double position_factor = weighted * left_charge
                            * right_charge * gradient_factor;
                        for (std::size_t cartesian = 0; cartesian < 3;
                            ++cartesian)
                            dx[cartesian] +=
                                position_factor * difference[cartesian];
                    }
                }
                for (std::size_t multipole = 0;
                    multipole < density_width; ++multipole) {
                    Kokkos::atomic_add(
                        &atomic_grad_multipoles(left, multipole),
                        dm_left[multipole]);
                    Kokkos::atomic_add(
                        &atomic_grad_multipoles(right, multipole),
                        dm_right[multipole]);
                }
                for (std::size_t cartesian = 0; cartesian < 3; ++cartesian) {
                    Kokkos::atomic_add(
                        &grad_positions(left, cartesian), -dx[cartesian]);
                    Kokkos::atomic_add(
                        &grad_positions(right, cartesian), dx[cartesian]);
                }
            });
        Kokkos::parallel_for(
            "long_range realspace energy atomic narrow and self",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(
                0, n_nodes * density_width),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t owner = flat / density_width;
                const std::size_t multipole = flat % density_width;
                const std::size_t graph =
                    static_cast<std::size_t>(node_graph(owner));
                const double energy_seed =
                    static_cast<double>(grad_energy(graph));
                double total = atomic_grad_multipoles(owner, multipole);
                if (include_self_interaction) {
                    for (std::size_t channel = 0; channel < density_width;
                        ++channel) {
                        const std::size_t selected = self_select(channel);
                        const double coefficient =
                            0.5 * self_constants(channel);
                        if (channel == multipole)
                            total += energy_seed * coefficient
                                * static_cast<double>(
                                    multipoles(owner, selected));
                        if (selected == multipole)
                            total += energy_seed * coefficient
                                * static_cast<double>(
                                    multipoles(owner, channel));
                    }
                }
                grad_multipoles(owner, multipole) =
                    static_cast<Scalar>(total);
            });
        return result;
    }

    realspace_reverse_launch_diagnostics_.energy_multipole_league_extent =
        n_nodes * density_width;
    realspace_reverse_launch_diagnostics_.energy_position_league_extent =
        n_nodes * 3;
    ++realspace_reverse_launch_diagnostics_.energy_owner_selected_count;

    Kokkos::parallel_for(
        "long_range realspace energy multipole adjoint owners",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes * density_width),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t owner = flat / density_width;
            const std::size_t multipole = flat % density_width;
            const std::size_t graph =
                static_cast<std::size_t>(node_graph(owner));
            const double energy_seed =
                static_cast<double>(grad_energy(graph));
            double total = 0.0;
            for (std::size_t owner_replica = 0; owner_replica < replicas;
                ++owner_replica) {
                const double transform = realspace_replica_transform_entry(
                    density_max_l, owner_replica, multipole, displacement);
                if (transform == 0.0)
                    continue;
                for (std::size_t tile_begin = 0; tile_begin < n_nodes;
                    tile_begin += pair_tile) {
                    const std::size_t tile_end = tile_begin + pair_tile < n_nodes
                        ? tile_begin + pair_tile
                        : n_nodes;
                    for (std::size_t other = tile_begin; other < tile_end;
                        ++other) {
                        if (other == owner
                            || static_cast<std::size_t>(node_graph(other))
                                != graph)
                            continue;
                        for (std::size_t other_replica = 0;
                            other_replica < replicas; ++other_replica) {
                            const double other_charge =
                                realspace_replica_charge(multipoles, other,
                                    other_replica, density_max_l, displacement);
                            double distance_squared = 0.0;
                            for (std::size_t cartesian = 0; cartesian < 3;
                                ++cartesian) {
                                const double owner_offset = owner_replica > 0
                                        && cartesian == owner_replica - 1
                                    ? displacement
                                    : 0.0;
                                const double other_offset = other_replica > 0
                                        && cartesian == other_replica - 1
                                    ? displacement
                                    : 0.0;
                                const double difference =
                                    realspace_displaced_difference(
                                        positions(other, cartesian), other_offset,
                                        positions(owner, cartesian), owner_offset);
                                distance_squared = realspace_norm_accumulate(
                                    difference, difference, distance_squared);
                            }
                            const double distance =
                                realspace_sqrt(distance_squared);
                            total += energy_seed * constant * transform
                                * other_charge
                                * realspace_kernel(
                                    distance, density_smearing_width);
                        }
                    }
                }
            }
            if (include_self_interaction) {
                for (std::size_t channel = 0; channel < density_width;
                    ++channel) {
                    const std::size_t selected = self_select(channel);
                    const double coefficient = 0.5 * self_constants(channel);
                    if (channel == multipole)
                        total += energy_seed * coefficient
                            * static_cast<double>(multipoles(owner, selected));
                    if (selected == multipole)
                        total += energy_seed * coefficient
                            * static_cast<double>(multipoles(owner, channel));
                }
            }
            grad_multipoles(owner, multipole) = static_cast<Scalar>(total);
        });

    Kokkos::parallel_for(
        "long_range realspace energy position adjoint owners",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes * 3),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t owner = flat / 3;
            const std::size_t owner_cartesian = flat % 3;
            const std::size_t graph =
                static_cast<std::size_t>(node_graph(owner));
            const double energy_seed =
                static_cast<double>(grad_energy(graph));
            double total = 0.0;
            for (std::size_t owner_replica = 0; owner_replica < replicas;
                ++owner_replica) {
                const double owner_charge = realspace_replica_charge(
                    multipoles, owner, owner_replica, density_max_l,
                    displacement);
                for (std::size_t tile_begin = 0; tile_begin < n_nodes;
                    tile_begin += pair_tile) {
                    const std::size_t tile_end = tile_begin + pair_tile < n_nodes
                        ? tile_begin + pair_tile
                        : n_nodes;
                    for (std::size_t other = tile_begin; other < tile_end;
                        ++other) {
                        if (other == owner
                            || static_cast<std::size_t>(node_graph(other))
                                != graph)
                            continue;
                        for (std::size_t other_replica = 0;
                            other_replica < replicas; ++other_replica) {
                            const double other_charge =
                                realspace_replica_charge(multipoles, other,
                                    other_replica, density_max_l, displacement);
                            double distance_squared = 0.0;
                            double selected_difference = 0.0;
                            for (std::size_t cartesian = 0; cartesian < 3;
                                ++cartesian) {
                                const double owner_offset = owner_replica > 0
                                        && cartesian == owner_replica - 1
                                    ? displacement
                                    : 0.0;
                                const double other_offset = other_replica > 0
                                        && cartesian == other_replica - 1
                                    ? displacement
                                    : 0.0;
                                const double difference =
                                    realspace_displaced_difference(
                                        positions(other, cartesian), other_offset,
                                        positions(owner, cartesian), owner_offset);
                                if (cartesian == owner_cartesian)
                                    selected_difference = difference;
                                distance_squared = realspace_norm_accumulate(
                                    difference, difference, distance_squared);
                            }
                            const double distance =
                                realspace_sqrt(distance_squared);
                            total -= energy_seed * constant * owner_charge
                                * other_charge
                                * realspace_kernel_gradient_factor(
                                    distance, density_smearing_width)
                                * selected_difference;
                        }
                    }
                }
            }
            grad_positions(owner, owner_cartesian) = total;
        });
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::RealspaceAdjointDiagnosticResult
PolarLongRangeOpsT<Scalar>::diagnostic_adjoint_realspace_energy(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& multipoles,
    const LongRangeConstVector<double>& grad_energy,
    const bool include_self_interaction,
    const bool use_workspace) const
{
    if (use_workspace)
        prepare_workspace(0, cache.n_nodes, cache.n_graphs);
    auto multipoles_kernel = use_workspace
        ? workspace_matrix("long_range_realspace_energy_adjoint_multipoles",
              multipoles.extent(0), multipoles.extent(1))
        : LongRangeKernelMatrix<Scalar>(
              "long_range_realspace_energy_adjoint_multipoles",
              multipoles.extent(0), multipoles.extent(1));
    auto grad_energy_kernel = use_workspace
        ? workspace_vector("long_range_realspace_energy_adjoint_seed",
              grad_energy.extent(0))
        : LongRangeKernelVector<Scalar>(
              "long_range_realspace_energy_adjoint_seed",
              grad_energy.extent(0));
    stage_matrix_into<Scalar>(multipoles, multipoles_kernel);
    stage_vector_into<Scalar>(grad_energy, grad_energy_kernel);
    const RealspaceAdjointKernelResult kernel =
        adjoint_realspace_energy_kernel(cache, multipoles_kernel,
            grad_energy_kernel, include_self_interaction, use_workspace);
    RealspaceAdjointDiagnosticResult result;
    result.grad_multipoles = download_matrix<double>(kernel.grad_multipoles,
        "long_range_realspace_energy_grad_multipoles_host");
    result.grad_positions = download_matrix<double>(kernel.grad_positions,
        "long_range_realspace_energy_grad_positions_host");
    result.grad_cell = download_tensor3<double>(
        kernel.grad_cell, "long_range_realspace_energy_grad_cell_host");
    return result;
}

template <typename Scalar>
LongRangeAllocationUsage
PolarLongRangeOpsT<Scalar>::realspace_cache_allocation_usage(
    const LongRangeGeometryCacheT<Scalar>& cache) const
{
    return cache.realspace_allocation_usage();
}

template <typename Scalar>
LongRangeAllocationUsage
PolarLongRangeOpsT<Scalar>::reciprocal_cache_allocation_usage(
    const LongRangeGeometryCacheT<Scalar>& cache) const
{
    return cache.reciprocal_allocation_usage();
}

template <typename Scalar>
LongRangeGeometryCacheT<Scalar> PolarLongRangeOpsT<Scalar>::geometry_cache(
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index) const
{
    LongRangeGeometryCacheT<Scalar> cache;
    geometry_cache_into(positions, cell, node_graph_index, cache);
    return cache;
}

template <typename Scalar>
std::size_t PolarLongRangeOpsT<Scalar>::reciprocal_mode_count_up_to(
    const LongRangeConstTensor3<double>& cell,
    const std::size_t maximum_count) const
{
    if (maximum_count == 0)
        return 0;
    const std::size_t n_graphs = cell.extent(0);
    long long n_max[3] = {0, 0, 0};
    for (std::size_t graph = 0; graph < n_graphs; ++graph) {
        double reciprocal[9];
        reciprocal_cell(&cell(graph, 0, 0), reciprocal);
        for (int axis = 0; axis < 3; ++axis) {
            const double norm = std::sqrt(
                cell(graph, axis, 0) * cell(graph, axis, 0)
                + cell(graph, axis, 1) * cell(graph, axis, 1)
                + cell(graph, axis, 2) * cell(graph, axis, 2));
            double projection = 0.0;
            for (int component = 0; component < 3; ++component)
                projection += reciprocal[3 * axis + component]
                    * cell(graph, axis, component) / norm;
            if (!(projection > 0.0))
                throw std::invalid_argument(
                    "reciprocal_mode_count: the reciprocal cell must project "
                    "positively onto the normed lattice rows.");
            const double requested_bound = std::ceil(kspace_cutoff_ / projection);
            if (!std::isfinite(requested_bound)
                || requested_bound
                    > static_cast<double>(std::numeric_limits<int>::max()))
                throw std::length_error(
                    "reciprocal_mode_count: a reciprocal index bound exceeds "
                    "the supported int range.");
            const long long bound = static_cast<long long>(requested_bound);
            n_max[axis] = std::max(n_max[axis], bound);
        }
    }

    const double cutoff_squared = kspace_cutoff_ * kspace_cutoff_;
    std::size_t count = 0;
    for (std::size_t graph = 0; graph < n_graphs; ++graph) {
        double reciprocal[9];
        reciprocal_cell(&cell(graph, 0, 0), reciprocal);
        std::size_t graph_count = 0;
        const auto keep = [&] (const std::array<int, 3>& triple) {
            double wave[3] = {0.0, 0.0, 0.0};
            for (int axis = 0; axis < 3; ++axis) {
                const double coefficient = static_cast<double>(triple[axis]);
                for (int component = 0; component < 3; ++component)
                    wave[component] += coefficient
                        * reciprocal[3 * axis + component];
            }
            const double norm_squared = wave[0] * wave[0]
                + wave[1] * wave[1] + wave[2] * wave[2];
            if (norm_squared <= cutoff_squared) {
                if (count == std::numeric_limits<std::size_t>::max())
                    throw std::overflow_error(
                        "reciprocal_mode_count overflows size_t.");
                ++count;
                ++graph_count;
            }
            return count >= maximum_count;
        };
        if (keep({0, 0, 0}))
            return count;
        for (long long z = 1; z < n_max[2]; ++z)
            if (keep({0, 0, static_cast<int>(z)}))
                return count;
        for (long long y = 1; y < n_max[1]; ++y)
            for (long long z = -n_max[2]; z < n_max[2]; ++z)
                if (keep({0, static_cast<int>(y), static_cast<int>(z)}))
                    return count;
        for (long long x = 1; x < n_max[0]; ++x)
            for (long long y = -n_max[1]; y < n_max[1]; ++y)
                for (long long z = -n_max[2]; z < n_max[2]; ++z)
                    if (keep({static_cast<int>(x), static_cast<int>(y),
                            static_cast<int>(z)}))
                        return count;
        if (graph_count == 0)
            throw std::invalid_argument(
                "reciprocal_mode_count: every graph must keep at least the origin.");
    }
    return count;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::geometry_cache_into(
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index,
    LongRangeGeometryCacheT<Scalar>& cache) const
{
    release_realspace_full_materialized_tables();
    const std::size_t n_nodes = positions.extent(0);
    const std::size_t n_graphs = cell.extent(0);
    if (node_graph_index.extent(0) != n_nodes)
        throw std::invalid_argument(
            "geometry_cache: node_graph_index must match the position rows.");
    for (std::size_t n = 0; n < n_nodes; ++n)
        if (node_graph_index(n) < 0
            || static_cast<std::size_t>(node_graph_index(n)) >= n_graphs)
            throw std::invalid_argument(
                "geometry_cache: node_graph_index must address the cell's "
                "graphs.");

    const bool exact_reuse =
        cache.mode == LongRangeGeometryMode::direct_reciprocal
        && cache.cos_table.data() != nullptr
        && cache.n_nodes == n_nodes && cache.n_graphs == n_graphs
        && cache.positions_host_values.size() == n_nodes * 3
        && cache.cell_host_values.size() == n_graphs * 9
        && cache.node_graph_host_values.size() == n_nodes
        && std::equal(cache.positions_host_values.begin(),
            cache.positions_host_values.end(), positions.data())
        && std::equal(cache.cell_host_values.begin(),
            cache.cell_host_values.end(), cell.data())
        && std::equal(cache.node_graph_host_values.begin(),
            cache.node_graph_host_values.end(), node_graph_index.data());
    if (exact_reuse)
        return;

    // --- half-sphere k-grid (kspace.py:11-127) --------------------------
    // Per direction the bound is ceil(cutoff / (b_i . a_hat_i)) maximized
    // over graphs; the enumeration keeps exactly one of +/-k (origin, then
    // the open half line, half plane, half space with the last factor
    // varying fastest); each graph keeps its |k| <= cutoff subset in
    // enumeration order, and the flattened output is grouped by graph.
    std::vector<std::array<int, 3>> triples;
    long long n_max[3] = {0, 0, 0};
    for (std::size_t g = 0; g < n_graphs; ++g) {
        double rcell[9];
        reciprocal_cell(&cell(g, 0, 0), rcell);
        for (int i = 0; i < 3; ++i) {
            const double norm = std::sqrt(cell(g, i, 0) * cell(g, i, 0)
                + cell(g, i, 1) * cell(g, i, 1) + cell(g, i, 2) * cell(g, i, 2));
            double projection = 0.0;
            for (int j = 0; j < 3; ++j)
                projection += rcell[3 * i + j] * cell(g, i, j) / norm;
            if (!(projection > 0.0))
                throw std::invalid_argument(
                    "geometry_cache: the reciprocal cell must project "
                    "positively onto the normed lattice rows.");
            const long long bound = static_cast<long long>(
                std::ceil(kspace_cutoff_ / projection));
            n_max[i] = std::max(n_max[i], bound);
        }
    }
    triples.push_back({0, 0, 0});
    for (long long z = 1; z < n_max[2]; ++z)
        triples.push_back({0, 0, static_cast<int>(z)});
    for (long long y = 1; y < n_max[1]; ++y)
        for (long long z = -n_max[2]; z < n_max[2]; ++z)
            triples.push_back({0, static_cast<int>(y), static_cast<int>(z)});
    for (long long x = 1; x < n_max[0]; ++x)
        for (long long y = -n_max[1]; y < n_max[1]; ++y)
            for (long long z = -n_max[2]; z < n_max[2]; ++z)
                triples.push_back({static_cast<int>(x), static_cast<int>(y),
                    static_cast<int>(z)});

    const double cutoff_squared = kspace_cutoff_ * kspace_cutoff_;
    std::vector<double> kept_k;
    std::vector<double> kept_norm2;
    std::vector<int> kept_graph;
    for (std::size_t g = 0; g < n_graphs; ++g) {
        double rcell[9];
        reciprocal_cell(&cell(g, 0, 0), rcell);
        std::size_t graph_kept = 0;
        for (const auto& triple : triples) {
            double k[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i) {
                const double coefficient = static_cast<double>(triple[i]);
                for (int j = 0; j < 3; ++j)
                    k[j] += coefficient * rcell[3 * i + j];
            }
            const double norm2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            if (norm2 <= cutoff_squared) {
                kept_k.push_back(k[0]);
                kept_k.push_back(k[1]);
                kept_k.push_back(k[2]);
                kept_norm2.push_back(norm2);
                kept_graph.push_back(static_cast<int>(g));
                ++graph_kept;
            }
        }
        if (graph_kept == 0)
            throw std::invalid_argument(
                "geometry_cache: every graph must keep at least the origin.");
    }

    const std::size_t n_k = kept_graph.size();
    const bool replace = cache.n_k != n_k || cache.n_nodes != n_nodes
        || cache.n_graphs != n_graphs || cache.cos_table.data() == nullptr;
    if (replace)
        cache = LongRangeGeometryCacheT<Scalar>{};
    cache.n_k = n_k;
    cache.n_nodes = n_nodes;
    cache.n_graphs = n_graphs;
    cache.mode = LongRangeGeometryMode::direct_reciprocal;
    if (replace) {
        cache.k_vectors = LongRangeKernelMatrix<Scalar>(
            "long_range_k_vectors", n_k, 3);
        cache.k_norm2 = LongRangeKernelVector<Scalar>("long_range_k_norm2", n_k);
        cache.k_graph_index = LongRangeKernelVector<int>("long_range_k_graph", n_k);
        cache.node_graph_index = LongRangeKernelVector<int>(
            "long_range_node_graph", n_nodes);
        cache.k0_mask = LongRangeKernelVector<Scalar>("long_range_k0_mask", n_k);
        cache.volume_per_k = LongRangeKernelVector<Scalar>(
            "long_range_volume_per_k", n_k);
        cache.k_factor_coulomb = LongRangeKernelVector<Scalar>(
            "long_range_k_factor_coulomb", n_k);
        cache.k_factor_proj = LongRangeKernelVector<Scalar>(
            "long_range_k_factor_proj", n_k);
        cache.cos_table = LongRangeKernelMatrix<Scalar>(
            "long_range_cos", n_k, n_nodes);
        cache.sin_table = LongRangeKernelMatrix<Scalar>(
            "long_range_sin", n_k, n_nodes);
        cache.density_fourier_factors = LongRangeKernelTensor4<Scalar>(
            "long_range_density_fs", n_k, 1, density_width_, 2);
        cache.feature_fourier_factors = LongRangeKernelTensor4<Scalar>(
            "long_range_feature_fs", n_k, feature_smearing_widths_.size(),
            feature_width_, 2);
        cache.kernel_volumes = LongRangeKernelVector<Scalar>(
            "long_range_volumes", n_graphs);
        cache.positions_workspace = LongRangeKernelMatrix<Scalar>(
            "long_range_positions", n_nodes, 3);
        cache.phase_workspace = LongRangeKernelMatrix<Scalar>(
            "long_range_k_dot_r", n_k, n_nodes);
    }

    // The k-grid metadata is enumerated on the host: the serial drivers'
    // mirrors are these host arrays themselves, and the kernel copies are
    // one upload each.
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_node_graph("long_range_node_graph_host", n_nodes);
    bool topology_changed = replace
        || cache.node_graph_index_host.extent(0) != n_nodes;
    for (std::size_t n = 0; n < n_nodes; ++n)
    {
        host_node_graph(n) = node_graph_index(n);
        if (!topology_changed
            && cache.node_graph_index_host(n) != node_graph_index(n))
            topology_changed = true;
    }
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_volumes("long_range_volumes_host", n_graphs);
    for (std::size_t g = 0; g < n_graphs; ++g)
        host_volumes(g) = cell_determinant(&cell(g, 0, 0));
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_k("long_range_k_host", 3 * n_k);
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_norm2("long_range_k_norm2_host", n_k);
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_k_graph("long_range_k_graph_host", n_k);
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_k0("long_range_k0_host", n_k);
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_volume_per_k("long_range_volume_per_k_host", n_k);
    // The first kept mode of every graph is its origin: the k=0 mask is
    // data (exact branchy logic), never a floating scale factor.
    for (std::size_t i = 0; i < n_k; ++i) {
        for (int j = 0; j < 3; ++j)
            host_k(3 * i + j) = kept_k[3 * i + j];
        host_norm2(i) = kept_norm2[i];
        host_k_graph(i) = kept_graph[i];
        host_k0(i) = 0.0;
        host_volume_per_k(i) = host_volumes(
            static_cast<std::size_t>(kept_graph[i]));
    }
    for (std::size_t g = 0; g < n_graphs; ++g) {
        for (std::size_t i = 0; i < n_k; ++i) {
            if (static_cast<std::size_t>(kept_graph[i]) == g) {
                host_k0(i) = 1.0;
                break;
            }
        }
    }
    Kokkos::View<const double**, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_k_matrix(host_k.data(), n_k, 3);
    stage_matrix_into<Scalar>(host_k_matrix, cache.k_vectors);
    record_input_h2d(n_k,3*sizeof(Scalar),"reciprocal k-vector");
    const LongRangeConstVector<double> host_norm2_const(
        host_norm2.data(), host_norm2.extent(0));
    stage_vector_into<Scalar>(host_norm2_const, cache.k_norm2);
    record_input_h2d(n_k,sizeof(Scalar),"reciprocal k-norm");
    Kokkos::deep_copy(cache.k_graph_index, host_k_graph);
    record_input_h2d(n_k,sizeof(int),"reciprocal k graph");
    if (topology_changed) {
        Kokkos::deep_copy(cache.node_graph_index, host_node_graph);
        record_input_h2d(n_nodes,sizeof(int),"reciprocal node graph");
    }
    const LongRangeConstVector<double> host_k0_const(
        host_k0.data(), host_k0.extent(0));
    const LongRangeConstVector<double> host_volume_per_k_const(
        host_volume_per_k.data(), host_volume_per_k.extent(0));
    const LongRangeConstVector<double> host_volumes_const(
        host_volumes.data(), host_volumes.extent(0));
    stage_vector_into<Scalar>(host_k0_const, cache.k0_mask);
    record_input_h2d(n_k,sizeof(Scalar),"reciprocal k0 mask");
    stage_vector_into<Scalar>(host_volume_per_k_const, cache.volume_per_k);
    record_input_h2d(n_k,sizeof(Scalar),"reciprocal volume-per-k");
    stage_vector_into<Scalar>(host_volumes_const, cache.kernel_volumes);
    record_input_h2d(n_graphs,sizeof(Scalar),"reciprocal graph volume");
    // The serial drivers' mirrors keep the host spelling of the same
    // arrays (k row major, one entry per k).
    cache.k_vectors_host = host_k;
    cache.k_norm2_host = host_norm2;
    cache.volume_per_k_host = host_volume_per_k;
    cache.volumes_host = host_volumes;
    cache.k_graph_index_host = host_k_graph;
    cache.node_graph_index_host = host_node_graph;

    auto k_vectors = cache.k_vectors;
    auto k_norm2 = cache.k_norm2;
    auto k0_mask = cache.k0_mask;
    auto k_factor_coulomb = cache.k_factor_coulomb;
    auto k_factor_proj = cache.k_factor_proj;
    Kokkos::parallel_for(
        "long_range per-k factors",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            if (k0_mask(i) > 0.5) {
                k_factor_coulomb(i) = 0.0;
                k_factor_proj(i) = 0.5;
            } else {
                k_factor_coulomb(i) = 1.0 / k_norm2(i);
                k_factor_proj(i) = 1.0;
            }
        });
    // --- the K.R^T phase GEMM and its masked cos/sin tables --------------
    stage_matrix_into<Scalar>(positions, cache.positions_workspace);
    record_input_h2d(n_nodes,3*sizeof(Scalar),"reciprocal position");
    const auto positions_kernel = cache.positions_workspace;
    auto inner_products = cache.phase_workspace;
    KokkosBlas::gemm(
        "N", "T", 1.0, k_vectors, positions_kernel, 0.0, inner_products);
    auto cos_table = cache.cos_table;
    auto sin_table = cache.sin_table;
    auto k_graph_index = cache.k_graph_index;
    auto node_graph_kernel = cache.node_graph_index;
    Kokkos::parallel_for(
        "long_range cos sin tables",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const int graph = k_graph_index(i);
#if defined(_OPENMP)
#pragma omp simd
#endif
            for (std::size_t j = 0; j < n_nodes; ++j) {
                const double masked =
                    graph == node_graph_kernel(j) ? 1.0 : 0.0;
                cos_table(i, j) = Kokkos::cos(inner_products(i, j)) * masked;
                sin_table(i, j) = Kokkos::sin(inner_products(i, j)) * masked;
            }
        });

    // --- analytic GTO Fourier factors ------------------------------------
    evaluate_fourier_basis(cache.k_vectors, cache.k_norm2, cache.k0_mask,
        density_max_l_, density_fourier_parameters_,
        cache.density_fourier_factors);
    evaluate_fourier_basis(cache.k_vectors, cache.k_norm2, cache.k0_mask,
        feature_max_l_, feature_fourier_parameters_,
        cache.feature_fourier_factors);
    cache.positions_host_values.assign(
        positions.data(), positions.data() + n_nodes * 3);
    cache.cell_host_values.assign(
        cell.data(), cell.data() + n_graphs * 9);
    cache.node_graph_host_values.assign(
        node_graph_index.data(), node_graph_index.data() + n_nodes);
}
template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::evaluate_fourier_basis(
    const LongRangeKernelMatrix<Scalar>& k_vectors,
    const LongRangeKernelVector<Scalar>& k_norm2,
    const LongRangeKernelVector<Scalar>& k0_mask,
    const int max_l,
    const LongRangeKernelMatrix<Scalar>& parameters,
    const LongRangeKernelTensor4<Scalar>& out) const
{
    const std::size_t n_k = k_vectors.extent(0);
    const double l1_constant = harmonic_l1();

    const std::size_t n_sigma = parameters.extent(0);
    const bool with_l1 = max_l >= 1;
    const double l0 = harmonic_l0();
    auto parameters_v = parameters;
    Kokkos::parallel_for(
        "long_range fourier factors",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const bool at_origin = k0_mask(i) > 0.5;
            const double modulus = at_origin ? 0.0 : std::sqrt(k_norm2(i));
            // e3nn l <= 1 harmonics of the permuted direction (y, z, x),
            // input-normalized; at the origin the replacement direction
            // multiplies a vanishing l=1 radial factor.
            double direction[3] = {0.0, 0.0, l1_constant};
            if (!at_origin) {
                const double inverse = l1_constant / modulus;
                direction[0] = k_vectors(i, 1) * inverse;
                direction[1] = k_vectors(i, 2) * inverse;
                direction[2] = k_vectors(i, 0) * inverse;
            }
            for (std::size_t i_s = 0; i_s < n_sigma; ++i_s) {
                const double exponential =
                    std::exp(-0.5 * k_norm2(i) * parameters_v(i_s, 2));
                // The (-i)^l phase buffers: l=0 real, l=1 purely imaginary.
                const double f0 = parameters_v(i_s, 0) * exponential
                    * parameters_v(i_s, 3);
                out(i, i_s, 0, 0) = f0 * l0;
                out(i, i_s, 0, 1) = 0.0;
                if (with_l1) {
                    const double f1 = parameters_v(i_s, 1) * modulus
                        * exponential * parameters_v(i_s, 4);
                    for (int m = 0; m < 3; ++m) {
                        out(i, i_s, 1 + m, 0) = 0.0;
                        out(i, i_s, 1 + m, 1) = -f1 * direction[m];
                    }
                }
            }
        });
}

template <typename Scalar>
LongRangeKernelMatrix<Scalar> PolarLongRangeOpsT<Scalar>::assemble_fourier_kernel(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& multipoles,
    const bool use_workspace) const
{
    if (multipoles.extent(1) != density_width_)
        throw std::invalid_argument(
            "assemble_fourier: multipoles must carry the density width.");
    const std::size_t n_k = cache.n_k;
    const std::size_t width = density_width_;
    const double scale = two_pi_cubed();
    auto make_matrix = [this, use_workspace] (const char* label,
        const std::size_t rows, const std::size_t columns) {
        return use_workspace
            ? workspace_matrix(label, rows, columns)
            : LongRangeKernelMatrix<Scalar>(
                  Kokkos::view_alloc(std::string(label)), rows, columns);
    };
    auto density = make_matrix("long_range_rho", n_k, 2);
    auto coeff_cos = make_matrix("long_range_coeff_cos", n_k, density_width_);
    auto coeff_sin = make_matrix("long_range_coeff_sin", n_k, density_width_);
    // The two thin GEMMs [n_k, n] @ [n, m] of the assemble stage.
    KokkosBlas::gemm(
        "N", "N", 1.0, cache.cos_table, multipoles, 0.0, coeff_cos);
    KokkosBlas::gemm(
        "N", "N", 1.0, cache.sin_table, multipoles, 0.0, coeff_sin);
    auto density_out = density;
    auto factors = cache.density_fourier_factors;
    auto volume_per_k = cache.volume_per_k;
    Kokkos::parallel_for(
        "long_range assemble",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            double real = 0.0;
            double imaginary = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : real, imaginary)
#endif
            for (std::size_t m = 0; m < width; ++m) {
                const double factor_real = factors(i, 0, m, 0);
                const double factor_imaginary = factors(i, 0, m, 1);
                const double c_cos = coeff_cos(i, m);
                const double c_sin = coeff_sin(i, m);
                real += factor_real * c_cos + factor_imaginary * c_sin;
                imaginary += factor_imaginary * c_cos - factor_real * c_sin;
            }
            density_out(i, 0) = scale * real / volume_per_k(i);
            density_out(i, 1) = scale * imaginary / volume_per_k(i);
        });
    return density;
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::assemble_fourier(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& multipoles) const
{
    LongRangeKernelMatrix<Scalar> multipoles_kernel(
        "long_range_public_multipoles", multipoles.extent(0), multipoles.extent(1));
    stage_matrix_into<Scalar>(multipoles, multipoles_kernel);
    return download_matrix<double>(
        assemble_fourier_kernel(cache, multipoles_kernel),
        "long_range_rho_host");
}

template <typename Scalar>
LongRangeKernelMatrix<Scalar> PolarLongRangeOpsT<Scalar>::apply_coulomb_kernel(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& density,
    const bool use_workspace) const
{
    const std::size_t n_k = cache.n_k;
    auto potential = use_workspace
        ? workspace_matrix("long_range_potential", n_k, 2)
        : LongRangeKernelMatrix<Scalar>("long_range_potential", n_k, 2);
    auto factor = cache.k_factor_coulomb;
    auto density_v = density;
    Kokkos::parallel_for(
        "long_range coulomb convolve",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            potential(i, 0) =
                density_v(i, 0) * factor(i) * field_constant;
            potential(i, 1) =
                density_v(i, 1) * factor(i) * field_constant;
        });
    return potential;
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::apply_coulomb(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& density) const
{
    LongRangeKernelMatrix<Scalar> density_kernel(
        "long_range_public_density_in", density.extent(0), density.extent(1));
    stage_matrix_into<Scalar>(density, density_kernel);
    return download_matrix<double>(
        apply_coulomb_kernel(cache, density_kernel),
        "long_range_potential_host");
}

template <typename Scalar>
LongRangeKernelMatrix<Scalar> PolarLongRangeOpsT<Scalar>::project_to_features_kernel(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& potential,
    const bool use_workspace) const
{
    const std::size_t n_k = cache.n_k;
    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t feature_dim = feature_dim_;
    auto make_matrix = [this, use_workspace] (const char* label,
        const std::size_t rows, const std::size_t columns) {
        return use_workspace
            ? workspace_matrix(label, rows, columns)
            : LongRangeKernelMatrix<Scalar>(
                  Kokkos::view_alloc(std::string(label)), rows, columns);
    };
    auto amplitude_a = make_matrix("long_range_proj_a", n_k, feature_dim);
    auto amplitude_b = make_matrix("long_range_proj_b", n_k, feature_dim);
    projection_amplitudes(cache, potential, amplitude_a, amplitude_b);
    // The two transposed GEMMs [f_dim, n_k] @ [n_k, n], accumulated.
    auto projections = make_matrix(
        "long_range_proj_total", feature_dim, n_nodes);
    KokkosBlas::gemm(
        "T", "N", 1.0, amplitude_a, cache.cos_table, 0.0, projections);
    KokkosBlas::gemm(
        "T", "N", 1.0, amplitude_b, cache.sin_table, 1.0, projections);
    // The half-sphere factor-2 weight, the (2 pi)^-3 projection factor, and
    // the l-major channel permutation.
    const double scale = 2.0 / two_pi_cubed();
    auto features = make_matrix("long_range_features", n_nodes, feature_dim);
    auto permutation_v = output_permutation_kernel_;
    Kokkos::parallel_for(
        "long_range project permutation",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes),
        KOKKOS_LAMBDA(const std::size_t n) {
            for (std::size_t j = 0; j < feature_dim; ++j)
                features(n, j) = projections(permutation_v(j), n) * scale;
        });
    return features;
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::project_to_features(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& potential) const
{
    LongRangeKernelMatrix<Scalar> potential_kernel(
        "long_range_public_potential_in", potential.extent(0), potential.extent(1));
    stage_matrix_into<Scalar>(potential, potential_kernel);
    return download_matrix<double>(
        project_to_features_kernel(cache, potential_kernel),
        "long_range_features_host");
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::coulomb_features(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& multipoles,
    const bool use_workspace) const
{
    if (use_workspace)
        prepare_workspace(cache.n_k, cache.n_nodes, cache.n_graphs);
    auto multipoles_kernel = use_workspace
        ? workspace_matrix("long_range_composed_multipoles_forward",
            multipoles.extent(0), multipoles.extent(1))
        : LongRangeKernelMatrix<Scalar>("long_range_public_coulomb_multipoles",
            multipoles.extent(0), multipoles.extent(1));
    stage_matrix_into<Scalar>(multipoles, multipoles_kernel);
    const auto density = assemble_fourier_kernel(
        cache, multipoles_kernel, use_workspace);
    const auto potential = apply_coulomb_kernel(
        cache, density, use_workspace);
    return download_matrix<double>(
        project_to_features_kernel(cache, potential, use_workspace),
        "long_range_features_host");
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::adjoint_coulomb_features_composed(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& multipoles,
    const LongRangeConstMatrix<double>& grad_features,
    const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_k_accumulator,
    const LongRangeKernelVector<Scalar>& grad_volume_accumulator) const
{
    auto multipoles_kernel = workspace_matrix("long_range_composed_multipoles",
        multipoles.extent(0), multipoles.extent(1));
    stage_matrix_into<Scalar>(multipoles, multipoles_kernel);
    auto grad_features_kernel = workspace_matrix(
        "long_range_composed_grad_features", grad_features.extent(0),
        grad_features.extent(1));
    stage_matrix_into<Scalar>(grad_features, grad_features_kernel);
    return download_matrix<double>(adjoint_coulomb_features_kernel(cache,
        multipoles_kernel, grad_features_kernel, grad_cos_accumulator,
        grad_sin_accumulator, grad_k_accumulator, grad_volume_accumulator),
        "long_range_composed_grad_multipoles_host");
}

template <typename Scalar>
LongRangeKernelMatrix<Scalar>
PolarLongRangeOpsT<Scalar>::adjoint_coulomb_features_kernel(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& multipoles_kernel,
    const LongRangeKernelMatrix<Scalar>& grad_features_kernel,
    const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_k_accumulator,
    const LongRangeKernelVector<Scalar>& grad_volume_accumulator) const
{
    const std::size_t n_k = cache.n_k;
    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t feature_dim = feature_dim_;
    const std::size_t width = density_width_;
    if (multipoles_kernel.extent(0) != n_nodes
        || multipoles_kernel.extent(1) != width)
        throw std::invalid_argument(
            "composed feature adjoint: multipoles have the wrong shape.");
    if (grad_features_kernel.extent(0) != n_nodes
        || grad_features_kernel.extent(1) != feature_dim)
        throw std::invalid_argument(
            "composed feature adjoint: feature seeds have the wrong shape.");
    if (grad_cos_accumulator.extent(0) != n_k
        || grad_cos_accumulator.extent(1) != n_nodes
        || grad_sin_accumulator.extent(0) != n_k
        || grad_sin_accumulator.extent(1) != n_nodes
        || grad_k_accumulator.extent(0) != n_k
        || grad_k_accumulator.extent(1) != 3
        || grad_volume_accumulator.extent(0) != n_k)
        throw std::invalid_argument(
            "composed feature adjoint: device accumulators do not match the "
            "geometry cache.");

    const auto density = assemble_fourier_kernel(
        cache, multipoles_kernel, /*use_workspace=*/true);
    const auto potential = apply_coulomb_kernel(
        cache, density, /*use_workspace=*/true);

    // Project adjoint: scatter the feature seed, form amplitude seeds, add
    // the two table seeds, and retain dL/dpotential in kernel memory.
    const double projection_scale = 2.0 / two_pi_cubed();
    auto grad_projections = workspace_matrix(
        "long_range_composed_grad_projections", feature_dim, n_nodes);
    auto permutation_v = output_permutation_kernel_;
    Kokkos::parallel_for(
        "long_range composed project adjoint scatter",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes),
        KOKKOS_LAMBDA(const std::size_t n) {
            for (std::size_t j = 0; j < feature_dim; ++j)
                grad_projections(permutation_v(j), n) =
                    projection_scale * grad_features_kernel(n, j);
        });
    auto amplitude_a = workspace_matrix(
        "long_range_composed_proj_a", n_k, feature_dim);
    auto amplitude_b = workspace_matrix(
        "long_range_composed_proj_b", n_k, feature_dim);
    projection_amplitudes(cache, potential, amplitude_a, amplitude_b);
    auto grad_amplitude_a = workspace_matrix(
        "long_range_composed_grad_proj_a", n_k, feature_dim);
    auto grad_amplitude_b = workspace_matrix(
        "long_range_composed_grad_proj_b", n_k, feature_dim);
    KokkosBlas::gemm("N", "T", 1.0, cache.cos_table, grad_projections, 0.0,
        grad_amplitude_a);
    KokkosBlas::gemm("N", "T", 1.0, cache.sin_table, grad_projections, 0.0,
        grad_amplitude_b);
    auto grad_project_cos = workspace_matrix(
        "long_range_composed_project_cos", n_k, n_nodes);
    auto grad_project_sin = workspace_matrix(
        "long_range_composed_project_sin", n_k, n_nodes);
    KokkosBlas::gemm("N", "N", 1.0, amplitude_a, grad_projections, 0.0,
        grad_project_cos);
    KokkosBlas::gemm("N", "N", 1.0, amplitude_b, grad_projections, 0.0,
        grad_project_sin);

    auto grad_potential = workspace_matrix(
        "long_range_composed_grad_potential", n_k, 2);
    const std::size_t feature_width = feature_width_;
    auto feature_factors = cache.feature_fourier_factors;
    auto factor_proj = cache.k_factor_proj;
    Kokkos::parallel_for(
        "long_range composed project adjoint combine",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            double g0 = 0.0;
            double g1 = 0.0;
            for (std::size_t m = 0; m < feature_dim; ++m) {
                const std::size_t sigma = m / feature_width;
                const std::size_t channel = m % feature_width;
                const double fr = feature_factors(i, sigma, channel, 0);
                const double fi = feature_factors(i, sigma, channel, 1);
                g0 += grad_amplitude_a(i, m) * fr
                    + grad_amplitude_b(i, m) * fi;
                g1 += grad_amplitude_a(i, m) * fi
                    - grad_amplitude_b(i, m) * fr;
            }
            grad_potential(i, 0) = factor_proj(i) * g0;
            grad_potential(i, 1) = factor_proj(i) * g1;
        });

    // Project cell seed.  It reuses the amplitude seeds above, so the
    // projection scatter and the two table GEMMs are not recomputed.
    const auto derivatives = feature_fourier_derivatives_;
    auto grad_project_k = workspace_matrix(
        "long_range_composed_project_k", n_k, 3);
    const bool feature_with_l1 = feature_max_l_ >= 1;
    auto k_norm2 = cache.k_norm2;
    auto k_vectors = cache.k_vectors;
    Kokkos::parallel_for(
        "long_range composed project cell k seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            for (int j = 0; j < 3; ++j) {
                double value = 0.0;
                for (std::size_t m = 0; m < feature_dim; ++m) {
                    const std::size_t s = m / feature_width;
                    const std::size_t channel = m % feature_width;
                    const double sigma2 = derivatives(s, 0);
                    const double exponential =
                        Kokkos::exp(-0.5 * k_norm2(i) * sigma2);
                    double dfr = 0.0;
                    double dfi = 0.0;
                    if (channel == 0) {
                        dfr = -sigma2 * k_vectors(i, j)
                            * derivatives(s, 1) * exponential;
                    } else if (feature_with_l1) {
                        const int p = static_cast<int>(channel - 1);
                        const int q = p == 0 ? 1 : (p == 1 ? 2 : 0);
                        dfi = -derivatives(s, 2) * exponential
                            * ((q == j ? 1.0 : 0.0)
                                - sigma2 * k_vectors(i, q)
                                    * k_vectors(i, j));
                    }
                    const double da = factor_proj(i)
                        * (potential(i, 0) * dfr + potential(i, 1) * dfi);
                    const double db = factor_proj(i)
                        * (potential(i, 0) * dfi - potential(i, 1) * dfr);
                    value += da * grad_amplitude_a(i, m)
                        + db * grad_amplitude_b(i, m);
                }
                grad_project_k(i, j) = value;
            }
        });
    accumulate_kernel_matrix(grad_k_accumulator, grad_project_k,
        "long_range composed accumulate project k");

    // Coulomb reverse and its k-vector cell seed.
    const auto grad_density = apply_coulomb_kernel(
        cache, grad_potential, /*use_workspace=*/true);
    auto grad_coulomb_k = workspace_matrix(
        "long_range_composed_coulomb_k", n_k, 3);
    Kokkos::parallel_for(
        "long_range composed coulomb cell k seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double norm2 = k_norm2(i);
            double factor = 0.0;
            if (norm2 != 0.0)
                factor = -2.0 * field_constant
                    * (grad_potential(i, 0) * density(i, 0)
                        + grad_potential(i, 1) * density(i, 1))
                    / (norm2 * norm2);
            for (int j = 0; j < 3; ++j)
                grad_coulomb_k(i, j) = factor * k_vectors(i, j);
        });
    accumulate_kernel_matrix(grad_k_accumulator, grad_coulomb_k,
        "long_range composed accumulate coulomb k");

    // Assemble reverse: retain the density seed, table seeds, and returned
    // node seed on-device until the final compact node-seed materialization.
    const double assemble_scale = two_pi_cubed();
    auto grad_coeff_cos = workspace_matrix(
        "long_range_composed_grad_coeff_cos", n_k, width);
    auto grad_coeff_sin = workspace_matrix(
        "long_range_composed_grad_coeff_sin", n_k, width);
    auto density_factors = cache.density_fourier_factors;
    auto volume_per_k = cache.volume_per_k;
    Kokkos::parallel_for(
        "long_range composed assemble adjoint combine",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double s = assemble_scale / volume_per_k(i);
            const double g0 = grad_density(i, 0);
            const double g1 = grad_density(i, 1);
            for (std::size_t m = 0; m < width; ++m) {
                const double fr = density_factors(i, 0, m, 0);
                const double fi = density_factors(i, 0, m, 1);
                grad_coeff_cos(i, m) = s * (g0 * fr + g1 * fi);
                grad_coeff_sin(i, m) = s * (g0 * fi - g1 * fr);
            }
        });
    auto grad_assemble_cos = workspace_matrix(
        "long_range_composed_assemble_cos", n_k, n_nodes);
    auto grad_assemble_sin = workspace_matrix(
        "long_range_composed_assemble_sin", n_k, n_nodes);
    auto grad_multipoles = workspace_matrix(
        "long_range_composed_grad_multipoles", n_nodes, width);
    KokkosBlas::gemm("N", "T", 1.0, grad_coeff_cos, multipoles_kernel, 0.0,
        grad_assemble_cos);
    KokkosBlas::gemm("N", "T", 1.0, grad_coeff_sin, multipoles_kernel, 0.0,
        grad_assemble_sin);
    KokkosBlas::gemm("T", "N", 1.0, cache.cos_table, grad_coeff_cos, 0.0,
        grad_multipoles);
    KokkosBlas::gemm("T", "N", 1.0, cache.sin_table, grad_coeff_sin, 1.0,
        grad_multipoles);
    const std::size_t table_size = grad_cos_accumulator.size();
    Kokkos::parallel_for("long_range composed accumulate feature tables",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, table_size),
        KOKKOS_LAMBDA(const std::size_t index) {
            grad_cos_accumulator.data()[index] +=
                grad_project_cos.data()[index];
            grad_cos_accumulator.data()[index] +=
                grad_assemble_cos.data()[index];
            grad_sin_accumulator.data()[index] +=
                grad_project_sin.data()[index];
            grad_sin_accumulator.data()[index] +=
                grad_assemble_sin.data()[index];
        });

    // Assemble cell metadata.  The coefficient GEMMs are forward
    // intermediates needed only for analytic factor derivatives.
    auto coeff_cos = workspace_matrix(
        "long_range_composed_coeff_cos", n_k, width);
    auto coeff_sin = workspace_matrix(
        "long_range_composed_coeff_sin", n_k, width);
    KokkosBlas::gemm(
        "N", "N", 1.0, cache.cos_table, multipoles_kernel, 0.0, coeff_cos);
    KokkosBlas::gemm(
        "N", "N", 1.0, cache.sin_table, multipoles_kernel, 0.0, coeff_sin);
    auto grad_assemble_k = workspace_matrix(
        "long_range_composed_assemble_k", n_k, 3);
    auto grad_volume = workspace_vector(
        "long_range_composed_assemble_volume", n_k);
    const double sigma = density_smearing_width_;
    const double sigma2 = sigma * sigma;
    const double pref0 =
        (4.0 * pi) * std::sqrt(pi / 2.0) * sigma * sigma * sigma;
    const double a0 =
        pref0 * cl_sigma(0, sigma, false) * harmonic_l0();
    const double b = pref0 * sigma2 * cl_sigma(1, sigma, false)
        * harmonic_l1();
    const bool density_with_l1 = density_max_l_ >= 1;
    Kokkos::parallel_for(
        "long_range composed assemble cell seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double g0 = grad_density(i, 0);
            const double g1 = grad_density(i, 1);
            grad_volume(i) = -(g0 * density(i, 0) + g1 * density(i, 1))
                / volume_per_k(i);
            const double s = assemble_scale / volume_per_k(i);
            const double exponential = Kokkos::exp(-0.5 * k_norm2(i) * sigma2);
            for (int j = 0; j < 3; ++j) {
                const double df0 = -sigma2 * k_vectors(i, j) * a0 * exponential;
                double dre = df0 * coeff_cos(i, 0);
                double dim = -df0 * coeff_sin(i, 0);
                if (density_with_l1) {
                    for (int p = 0; p < 3; ++p) {
                        const int q = p == 0 ? 1 : (p == 1 ? 2 : 0);
                        const double dfi = -b * exponential
                            * ((q == j ? 1.0 : 0.0)
                                - sigma2 * k_vectors(i, q)
                                    * k_vectors(i, j));
                        dre += dfi * coeff_sin(i, 1 + p);
                        dim += dfi * coeff_cos(i, 1 + p);
                    }
                }
                grad_assemble_k(i, j) = s * (g0 * dre + g1 * dim);
            }
        });
    accumulate_kernel_matrix(grad_k_accumulator, grad_assemble_k,
        "long_range composed accumulate assemble k");
    accumulate_kernel_vector(grad_volume_accumulator, grad_volume,
        "long_range composed accumulate assemble volume");
    return grad_multipoles;
}

template <typename Scalar>
LongRangeVector<double> PolarLongRangeOpsT<Scalar>::coulomb_energy(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeMatrix<double>* energy_self_fields,
    const bool use_workspace) const
{
    auto make_matrix = [this, use_workspace] (const char* label,
        const std::size_t rows, const std::size_t columns) {
        return use_workspace
            ? workspace_matrix(label, rows, columns)
            : LongRangeKernelMatrix<Scalar>(
                  Kokkos::view_alloc(std::string(label)), rows, columns);
    };
    auto charge_density_kernel = make_matrix(
        "long_range_energy_charge_density", charge_density.extent(0),
        charge_density.extent(1));
    stage_matrix_into<Scalar>(charge_density, charge_density_kernel);
    const auto density =
        assemble_fourier_kernel(cache, charge_density_kernel, use_workspace);
    const auto potential = apply_coulomb_kernel(cache, density, use_workspace);
    const std::size_t n_graphs = cache.n_graphs;
    const double scale = 0.5 / (two_pi_cubed() * two_pi_cubed());
    LongRangeKernelMatrix<Scalar> self_fields_kernel;
    const bool with_self_fields = energy_self_fields != nullptr;
    if (with_self_fields)
        self_fields_kernel = make_matrix("long_range_energy_self_fields_in",
            energy_self_fields->extent(0), energy_self_fields->extent(1));
    if (with_self_fields) {
        const LongRangeConstMatrix<double> self_fields_const(
            energy_self_fields->data(), energy_self_fields->extent(0),
            energy_self_fields->extent(1));
        stage_matrix_into<Scalar>(self_fields_const, self_fields_kernel);
    }
    auto energy_kernel = use_workspace
        ? workspace_vector("long_range_energy_kernel", n_graphs)
        : LongRangeKernelVector<Scalar>("long_range_energy_kernel", n_graphs);
    const std::size_t n_k = cache.n_k;
    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t width = density_width_;
    auto k_graph_index = cache.k_graph_index;
    auto node_graph_index = cache.node_graph_index;
    auto volumes = cache.kernel_volumes;
    auto self_fields = self_fields_kernel;
    // One worker per graph preserves the original fp64 k-row and node-row
    // accumulation order exactly while keeping the k-space intermediates in
    // kernel memory.
    Kokkos::parallel_for(
        "long_range coulomb energy",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_graphs),
        KOKKOS_LAMBDA(const std::size_t graph) {
            double value = 0.0;
            for (std::size_t i = 0; i < n_k; ++i) {
                if (static_cast<std::size_t>(k_graph_index(i)) != graph)
                    continue;
                value += 2.0
                    * (density(i, 0) * potential(i, 0)
                        + density(i, 1) * potential(i, 1));
            }
            value *= volumes(graph) * scale;
            if (with_self_fields) {
                for (std::size_t node = 0; node < n_nodes; ++node) {
                    if (static_cast<std::size_t>(node_graph_index(node))
                        != graph)
                        continue;
                    double node_energy = 0.0;
                    for (std::size_t m = 0; m < width; ++m)
                        node_energy += charge_density_kernel(node, m)
                            * self_fields(node, m);
                    value -= 0.5 * node_energy;
                }
            }
            energy_kernel(graph) = value;
        });
    return download_vector<double>(energy_kernel, "long_range_energy");
}
template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::feature_self_terms(
    const LongRangeConstMatrix<double>& multipoles) const
{
    const std::size_t n_nodes = multipoles.extent(0);
    LongRangeMatrix<double> terms(
        "long_range_feature_self_terms", n_nodes, feature_dim_);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, terms, 0.0);
    for (std::size_t n = 0; n < n_nodes; ++n)
        for (std::size_t d = 0; d < si_feature_constants_.extent(0); ++d)
            terms(n, d) = si_feature_constants_(d)
                * multipoles(n, si_feature_select_[d]);
    return terms;
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::energy_self_fields(
    const LongRangeConstMatrix<double>& multipoles) const
{
    const std::size_t n_nodes = multipoles.extent(0);
    LongRangeMatrix<double> fields(
        "long_range_energy_self_fields", n_nodes, density_width_);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, fields, 0.0);
    for (std::size_t n = 0; n < n_nodes; ++n)
        for (std::size_t d = 0; d < si_energy_constants_.extent(0); ++d)
            fields(n, d) = si_energy_constants_(d)
                * multipoles(n, si_energy_select_[d]);
    return fields;
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::external_field_features(
    const LongRangeConstMatrix<double>& external_potential,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeVector<double>& field_feature_norms,
    const bool use_workspace,
    const LongRangeKernelVector<int>* prepared_node_graph) const
{
    const std::size_t n_nodes = positions.extent(0);
    const std::size_t n_graphs = external_potential.extent(0);
    for (std::size_t n = 0; n < n_nodes; ++n)
        if (node_graph_index(n) < 0
            || static_cast<std::size_t>(node_graph_index(n)) >= n_graphs)
            throw std::invalid_argument(
                "external_field_features: node_graph_index must address the "
                "external potential rows.");
    // An unset norm vector means unit norms (the family's no-norm choice); a
    // set one must carry exactly one entry per feature, so a wrong-length
    // vector is a caller error rather than a silent unit-norm fallback.
    if (field_feature_norms.extent(0) != 0
        && field_feature_norms.extent(0) != feature_dim_)
        throw std::invalid_argument(
            "external_field_features: field_feature_norms must be empty "
            "(unit norms) or carry one entry per feature: expected "
            + std::to_string(feature_dim_) + ", got "
            + std::to_string(field_feature_norms.extent(0)) + ".");
    const bool with_norms = field_feature_norms.extent(0) != 0;
    auto device_matrix = [this, use_workspace] (const char* label,
        const std::size_t rows, const std::size_t columns) {
        return use_workspace
            ? workspace_matrix(label, rows, columns)
            : LongRangeKernelMatrix<Scalar>(
                  Kokkos::view_alloc(std::string(label)), rows, columns);
    };
    auto device_vector = [this, use_workspace] (const char* label,
        const std::size_t size) {
        return use_workspace
            ? workspace_vector(label, size)
            : LongRangeKernelVector<Scalar>(
                  Kokkos::view_alloc(std::string(label)), size);
    };
    auto features = device_matrix(
        "long_range_external_field_features", n_nodes, feature_dim_);

    // Per-graph barycenter (fp64 scatter mean) for the gauge-invariant
    // displacement E.(r - barycenter), staged into a kernel view.
    LongRangeKernelMatrix<Scalar> barycenter;
    {
        std::vector<double> sums(3 * n_graphs, 0.0);
        std::vector<std::size_t> counts(n_graphs, 0);
        for (std::size_t n = 0; n < n_nodes; ++n) {
            const std::size_t g =
                static_cast<std::size_t>(node_graph_index(n));
            for (int d = 0; d < 3; ++d)
                sums[3 * g + d] += positions(n, d);
            ++counts[g];
        }
        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>
            host("long_range_barycenter_host", n_graphs, 3);
        for (std::size_t g = 0; g < n_graphs; ++g)
            for (int d = 0; d < 3; ++d)
                host(g, d) = counts[g] == 0
                    ? 0.0
                    : sums[3 * g + d] / static_cast<double>(counts[g]);
        barycenter = device_matrix("long_range_barycenter", n_graphs, 3);
        const LongRangeConstMatrix<double> host_const(
            host.data(), n_graphs, 3);
        stage_matrix_into<Scalar>(host_const, barycenter);
    }
    auto external_potential_kernel = device_matrix(
        "long_range_external_potential", n_graphs, 4);
    stage_matrix_into<Scalar>(external_potential, external_potential_kernel);
    auto positions_kernel = device_matrix(
        "long_range_external_positions", n_nodes, 3);
    stage_matrix_into<Scalar>(positions, positions_kernel);
    LongRangeKernelVector<int> node_graph_kernel;
    if (prepared_node_graph != nullptr) {
        if (prepared_node_graph->extent(0) != n_nodes)
            throw std::invalid_argument(
                "external_field_features: prepared topology has the wrong "
                "node count.");
        node_graph_kernel = *prepared_node_graph;
    } else {
        node_graph_kernel = use_workspace
            ? workspace_int_vector("long_range_external_node_graph", n_nodes)
            : LongRangeKernelVector<int>(
                "long_range_external_node_graph", n_nodes);
        auto node_graph_host = Kokkos::create_mirror_view(node_graph_kernel);
        for (std::size_t n = 0; n < n_nodes; ++n)
            node_graph_host(n) = node_graph_index(n);
        Kokkos::deep_copy(node_graph_kernel, node_graph_host);
    }
    LongRangeKernelVector<Scalar> norms_kernel;
    if (with_norms) {
        norms_kernel = device_vector("long_range_field_norms", feature_dim_);
        const LongRangeConstVector<double> norms_const(
            field_feature_norms.data(), field_feature_norms.extent(0));
        stage_vector_into<Scalar>(norms_const, norms_kernel);
    }

    const double field_scale = 0.5;
    const std::size_t feature_dim = feature_dim_;
    auto matrix = external_field_kernel_;
    auto barycenter_v = barycenter;
    auto norms = norms_kernel;
    Kokkos::parallel_for(
        "long_range external field features",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes),
        KOKKOS_LAMBDA(const std::size_t n) {
            const std::size_t g =
                static_cast<std::size_t>(node_graph_kernel(n));
            double node_fields[4];
            for (int f = 0; f < 4; ++f)
                node_fields[f] = external_potential_kernel(g, f);
            double displacement_dot_field = 0.0;
            for (int d = 0; d < 3; ++d)
                displacement_dot_field
                    += (positions_kernel(n, d) - barycenter_v(g, d))
                    * external_potential_kernel(g, 1 + d);
            node_fields[0] += displacement_dot_field;
            // [V, E_x, E_y, E_z] -> [V, E_z, E_x, E_y], then the constant
            // matrix, one half per spin channel, and the linear norm divide.
            const double permuted[4] = {node_fields[0], node_fields[3],
                node_fields[1], node_fields[2]};
            for (std::size_t p = 0; p < feature_dim; ++p) {
                double accumulator = 0.0;
                for (int f = 0; f < 4; ++f)
                    accumulator += matrix(p, f) * permuted[f];
                accumulator *= field_scale;
                if (with_norms)
                    accumulator /= norms(p);
                features(n, p) = accumulator;
            }
        });
    return download_matrix<double>(
        features, "long_range_external_field_features_host");
}
template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::FukuiResult PolarLongRangeOpsT<Scalar>::fukui_normalize(
    const LongRangeConstMatrix<double>& fukui_sources,
    const LongRangeConstVector<double>& charge_l0_alpha,
    const LongRangeConstVector<double>& charge_l0_beta,
    const LongRangeConstVector<double>& total_charge,
    const LongRangeConstVector<double>& total_spin,
    const LongRangeConstVector<int>& node_graph_index) const
{
    const std::size_t n_nodes = fukui_sources.extent(0);
    const std::size_t n_graphs = total_charge.extent(0);
    for (std::size_t n = 0; n < n_nodes; ++n)
        if (node_graph_index(n) < 0
            || static_cast<std::size_t>(node_graph_index(n)) >= n_graphs)
            throw std::invalid_argument(
                "fukui_normalize: node_graph_index must address the "
                "charge/spin rows.");
    FukuiResult result;
    result.alpha = LongRangeVector<double>("long_range_fukui_alpha", n_nodes);
    result.beta = LongRangeVector<double>("long_range_fukui_beta", n_nodes);

    // fp64 per-graph scatter of the sources (zero-guarded) and of the l=0
    // charges, then the broadcast charge/spin target correction.
    std::vector<double> source_sum(2 * n_graphs, 0.0);
    std::vector<double> charge_sum(2 * n_graphs, 0.0);
    for (std::size_t n = 0; n < n_nodes; ++n) {
        const std::size_t g = static_cast<std::size_t>(node_graph_index(n));
        source_sum[2 * g] += fukui_sources(n, 0);
        source_sum[2 * g + 1] += fukui_sources(n, 1);
        charge_sum[2 * g] += charge_l0_alpha(n);
        charge_sum[2 * g + 1] += charge_l0_beta(n);
    }
    for (std::size_t n = 0; n < n_nodes; ++n) {
        const std::size_t g = static_cast<std::size_t>(node_graph_index(n));
        double norm_alpha = source_sum[2 * g];
        if (norm_alpha == 0.0)
            norm_alpha = 1.0;
        double norm_beta = source_sum[2 * g + 1];
        if (norm_beta == 0.0)
            norm_beta = 1.0;
        const double q_p_s = total_charge(g) + (total_spin(g) - 1.0);
        const double q_m_s = total_charge(g) - (total_spin(g) - 1.0);
        result.alpha(n) = fukui_sources(n, 0) / norm_alpha
            * (q_p_s / 2.0 - charge_sum[2 * g]);
        result.beta(n) = fukui_sources(n, 1) / norm_beta
            * (q_m_s / 2.0 - charge_sum[2 * g + 1]);
    }
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::DipoleResult PolarLongRangeOpsT<Scalar>::dipole_readout(
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstVector<int>& node_graph_index) const
{
    if (charge_density.extent(0) != positions.extent(0)
        || charge_density.extent(0) != node_graph_index.extent(0))
        throw std::invalid_argument(
            "dipole_readout: density, positions, and node_graph_index must "
            "agree on the node count.");
    const std::size_t n_nodes = charge_density.extent(0);
    std::size_t n_graphs = 0;
    for (std::size_t n = 0; n < n_nodes; ++n)
        n_graphs = std::max(n_graphs,
            static_cast<std::size_t>(node_graph_index(n)) + 1);
    DipoleResult result;
    result.total_charge = LongRangeVector<double>(
        "long_range_total_charge", n_graphs);
    result.total_dipole = LongRangeMatrix<double>(
        "long_range_total_dipole", n_graphs, 3);
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, result.total_charge, 0.0);
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, result.total_dipole, 0.0);
    // fp64 accumulators; the l=1 block carries the e3nn (y, z, x) phase
    // order, permuted to cartesian by [2, 0, 1].
    const bool with_dipoles = charge_density.extent(1) > 1;
    for (std::size_t n = 0; n < n_nodes; ++n) {
        const std::size_t g = static_cast<std::size_t>(node_graph_index(n));
        const double q = charge_density(n, 0);
        result.total_charge(g) += q;
        for (int d = 0; d < 3; ++d)
            result.total_dipole(g, d) += q * positions(n, d);
        if (with_dipoles) {
            result.total_dipole(g, 0) += charge_density(n, 3);
            result.total_dipole(g, 1) += charge_density(n, 1);
            result.total_dipole(g, 2) += charge_density(n, 2);
        }
    }
    return result;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::graph_multipoles(
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstVector<int>& node_graph_index,
    const std::size_t n_graphs,
    LongRangeVector<double>& total_charge,
    LongRangeMatrix<double>& total_dipole,
    LongRangeVector<double>& quadrupole) const
{
    total_charge = LongRangeVector<double>("long_range_q", n_graphs);
    total_dipole = LongRangeMatrix<double>("long_range_dip", n_graphs, 3);
    quadrupole = LongRangeVector<double>("long_range_quad", n_graphs);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, total_charge, 0.0);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, total_dipole, 0.0);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, quadrupole, 0.0);
    const bool with_dipoles = charge_density.extent(1) > 1;
    for (std::size_t n = 0; n < charge_density.extent(0); ++n) {
        const std::size_t g = static_cast<std::size_t>(node_graph_index(n));
        if (node_graph_index(n) < 0 || g >= n_graphs)
            throw std::invalid_argument(
                "node_graph_index must address the graph rows.");
        const double q = charge_density(n, 0);
        total_charge(g) += q;
        // The l=1 block in cartesian (x, y, z) = columns [3, 1, 2].
        double p[3] = {0.0, 0.0, 0.0};
        if (with_dipoles) {
            p[0] = charge_density(n, 3);
            p[1] = charge_density(n, 1);
            p[2] = charge_density(n, 2);
        }
        double p_dot_r = 0.0;
        double r_squared = 0.0;
        for (int d = 0; d < 3; ++d) {
            total_dipole(g, d) += q * positions(n, d) + p[d];
            p_dot_r += positions(n, d) * p[d];
            r_squared += positions(n, d) * positions(n, d);
        }
        quadrupole(g) += q * r_squared + 2.0 * p_dot_r;
    }
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::CorrectionEnergyResult
PolarLongRangeOpsT<Scalar>::analytic_correction_energy(
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeConstMatrix<int>& graph_pbc) const
{
    const std::size_t n_graphs = cell.extent(0);
    CorrectionEnergyResult result;
    result.selected = LongRangeVector<double>(
        "long_range_correction_energy", n_graphs);
    result.molecule = LongRangeVector<double>(
        "long_range_molecule_energy", n_graphs);
    result.slab = LongRangeVector<double>(
        "long_range_slab_energy", n_graphs);
    LongRangeVector<double> total_charge;
    LongRangeMatrix<double> total_dipole;
    LongRangeVector<double> quadrupole;
    graph_multipoles(charge_density, positions, node_graph_index, n_graphs,
        total_charge, total_dipole, quadrupole);

    const double constant = field_constant / (4.0 * pi);
    for (std::size_t g = 0; g < n_graphs; ++g) {
        const double volume = cell_determinant(&cell(g, 0, 0));
        // The energy block's literal exponent (four threes).
        const double box_length = std::pow(volume, 0.3333);
        const double charge = total_charge(g);
        const double dipole_norm_squared = total_dipole(g, 0)
                * total_dipole(g, 0)
            + total_dipole(g, 1) * total_dipole(g, 1)
            + total_dipole(g, 2) * total_dipole(g, 2);
        result.molecule(g) = 0.5 * cubic_madelung * constant * charge * charge
                / box_length
            + 2.0 * constant * pi * dipole_norm_squared / (3.0 * volume)
            - 2.0 * constant * pi * charge * quadrupole(g) / (3.0 * volume);
        // The slab z-dipole of the same density (charges plus local dipoles).
        double dipole_z = 0.0;
        for (std::size_t n = 0; n < charge_density.extent(0); ++n) {
            if (static_cast<std::size_t>(node_graph_index(n)) != g)
                continue;
            dipole_z += positions(n, 2) * charge_density(n, 0);
            if (charge_density.extent(1) > 1)
                dipole_z += charge_density(n, 2);
        }
        result.slab(g) = constant * 2.0 * pi * dipole_z * dipole_z / volume;

        // Mode selection from the graph's pbc row: molecule (no periodic
        // axis) takes the box term, slab (x, y periodic, z not) the dipole
        // term, anything else nothing.
        const bool periodic_x = graph_pbc(g, 0) != 0;
        const bool periodic_y = graph_pbc(g, 1) != 0;
        const bool periodic_z = graph_pbc(g, 2) != 0;
        if (!periodic_x && !periodic_y && !periodic_z)
            result.selected(g) = result.molecule(g);
        else if (periodic_x && periodic_y && !periodic_z)
            result.selected(g) = result.slab(g);
        else
            result.selected(g) = 0.0;
    }
    return result;
}

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::project_node_fields(
    const LongRangeMatrix<double>& node_fields,
    LongRangeMatrix<double>& features) const
{
    const std::size_t feature_dim = feature_dim_;
    const auto fields_kernel = stage_matrix<Scalar>(
        node_fields, "long_range_node_fields");
    LongRangeKernelMatrix<Scalar> features_kernel(
        "long_range_projected_node_fields", node_fields.extent(0),
        feature_dim);
    auto matrix = external_field_kernel_;
    Kokkos::parallel_for(
        "long_range node field projection",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, node_fields.extent(0)),
        KOKKOS_LAMBDA(const std::size_t n) {
            // [V, E_x, E_y, E_z] -> [V, E_z, E_x, E_y] then the constant
            // matrix: features[n, p] = sum_f matrix[p, f] * fields[n, f].
            const double permuted[4] = {fields_kernel(n, 0), fields_kernel(n, 3),
                fields_kernel(n, 1), fields_kernel(n, 2)};
            for (std::size_t p = 0; p < feature_dim; ++p) {
                double accumulator = 0.0;
                for (int f = 0; f < 4; ++f)
                    accumulator += matrix(p, f) * permuted[f];
                features_kernel(n, p) = accumulator;
            }
        });
    Kokkos::deep_copy(features,
        download_matrix<double>(features_kernel,
            "long_range_projected_node_fields_host"));
}
template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::correction_node_field_seeds(
    const LongRangeConstMatrix<double>& grad_fields,
    LongRangeMatrix<double>& node_field_seeds) const
{
    const std::size_t feature_dim = feature_dim_;
    const auto seeds_kernel = stage_matrix<Scalar>(
        grad_fields, "long_range_correction_field_seeds");
    LongRangeKernelMatrix<Scalar> fields_kernel(
        "long_range_correction_node_field_seeds", grad_fields.extent(0), 4);
    auto matrix = external_field_kernel_;
    Kokkos::parallel_for(
        "long_range correction node field seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, grad_fields.extent(0)),
        KOKKOS_LAMBDA(const std::size_t n) {
            // The exact adjoint of the forward's permutation and constant
            // matrix: seeds of the features become seeds of the permuted
            // [V, E_z, E_x, E_y] fields, un-permuted back to [V, E_x, E_y,
            // E_z].
            double permuted[4] = {0.0, 0.0, 0.0, 0.0};
            for (std::size_t p = 0; p < feature_dim; ++p) {
                const double seed = seeds_kernel(n, p);
                for (int f = 0; f < 4; ++f)
                    permuted[f] += matrix(p, f) * seed;
            }
            fields_kernel(n, 0) = permuted[0];
            fields_kernel(n, 1) = permuted[2];
            fields_kernel(n, 2) = permuted[3];
            fields_kernel(n, 3) = permuted[1];
        });
    Kokkos::deep_copy(node_field_seeds,
        download_matrix<double>(fields_kernel,
            "long_range_correction_node_field_seeds_host"));
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::CorrectionFeatureResult
PolarLongRangeOpsT<Scalar>::analytic_correction_features(
    const LongRangeConstMatrix<double>& charge_density_alpha,
    const LongRangeConstMatrix<double>& charge_density_beta,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeConstMatrix<int>& graph_pbc) const
{
    const std::size_t n_nodes = positions.extent(0);
    const std::size_t n_graphs = cell.extent(0);
    CorrectionFeatureResult result;
    result.alpha = LongRangeMatrix<double>(
        "long_range_correction_alpha", n_nodes, feature_dim_);
    result.beta = LongRangeMatrix<double>(
        "long_range_correction_beta", n_nodes, feature_dim_);
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, result.alpha, 0.0);
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, result.beta, 0.0);

    LongRangeMatrix<double>* outputs[2] = {&result.alpha, &result.beta};
    const LongRangeConstMatrix<double>* inputs[2] = {&charge_density_alpha,
        &charge_density_beta};
    for (int channel = 0; channel < 2; ++channel) {
        LongRangeVector<double> total_charge;
        LongRangeMatrix<double> total_dipole;
        LongRangeVector<double> quadrupole;
        graph_multipoles(*inputs[channel], positions, node_graph_index,
            n_graphs, total_charge, total_dipole, quadrupole);

        const double constant = field_constant / (4.0 * pi);
        LongRangeMatrix<double> node_fields(
            "long_range_correction_node_fields", n_nodes, 4);
        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, node_fields, 0.0);
        for (std::size_t g = 0; g < n_graphs; ++g) {
            const bool periodic_x = graph_pbc(g, 0) != 0;
            const bool periodic_y = graph_pbc(g, 1) != 0;
            const bool periodic_z = graph_pbc(g, 2) != 0;
            const bool is_molecule =
                !periodic_x && !periodic_y && !periodic_z;
            const bool is_slab = periodic_x && periodic_y && !periodic_z;
            if (!is_molecule && !is_slab)
                continue;
            const double volume = cell_determinant(&cell(g, 0, 0));
            if (is_molecule) {
                // The point-charge corrective potential of the box term; the
                // node-field block's literal exponent (six threes).
                const double box_length = std::pow(volume, 0.333333);
                const double charge = total_charge(g);
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    const double r_squared = positions(n, 0) * positions(n, 0)
                        + positions(n, 1) * positions(n, 1)
                        + positions(n, 2) * positions(n, 2);
                    double dipole_dot_r = 0.0;
                    for (int d = 0; d < 3; ++d)
                        dipole_dot_r += total_dipole(g, d) * positions(n, d);
                    node_fields(n, 0) = cubic_madelung * constant * charge
                            / box_length
                        - constant * 2.0 * pi * charge * r_squared
                            / (3.0 * volume)
                        + constant * 4.0 * pi * dipole_dot_r
                            / (3.0 * volume)
                        - constant * 2.0 * pi * quadrupole(g)
                            / (3.0 * volume);
                    for (int d = 0; d < 3; ++d)
                        node_fields(n, 1 + d) = 4.0 * pi * constant
                            * (total_dipole(g, d) - charge * positions(n, d))
                            / (3.0 * volume);
                }
            } else {
                // Slab z-dipole corrective fields: [field_z * z, 0, 0,
                // field_z] with field_z = FIELD_CONSTANT * mu_z / V.
                double dipole_z = 0.0;
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    dipole_z += positions(n, 2) * (*inputs[channel])(n, 0);
                    if ((*inputs[channel]).extent(1) > 1)
                        dipole_z += (*inputs[channel])(n, 2);
                }
                const double field_z =
                    constant * 4.0 * pi * dipole_z / volume;
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    node_fields(n, 0) = field_z * positions(n, 2);
                    node_fields(n, 3) = field_z;
                }
            }
        }
        project_node_fields(node_fields, *outputs[channel]);
    }
    return result;
}

// ===========================================================================
// Reverse-mode adjoints
// ===========================================================================

template <typename Scalar>
void PolarLongRangeOpsT<Scalar>::projection_amplitudes(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& potential,
    LongRangeKernelMatrix<Scalar>& amplitude_a,
    LongRangeKernelMatrix<Scalar>& amplitude_b) const
{
    const std::size_t feature_dim = feature_dim_;
    const std::size_t feature_width = feature_width_;
    auto factors = cache.feature_fourier_factors;
    auto factor_proj = cache.k_factor_proj;
    Kokkos::parallel_for(
        "long_range project amplitudes",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, cache.n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double scale = factor_proj(i);
#if defined(_OPENMP)
#pragma omp simd
#endif
            for (std::size_t m = 0; m < feature_dim; ++m) {
                const std::size_t sigma = m / feature_width;
                const std::size_t channel = m % feature_width;
                const double factor_real = factors(i, sigma, channel, 0);
                const double factor_imaginary = factors(i, sigma, channel, 1);
                amplitude_a(i, m) = scale
                    * (potential(i, 0) * factor_real
                        + potential(i, 1) * factor_imaginary);
                amplitude_b(i, m) = scale
                    * (potential(i, 0) * factor_imaginary
                        - potential(i, 1) * factor_real);
            }
        });
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::adjoint_geometry(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_cos_table,
    const LongRangeConstMatrix<double>& grad_sin_table) const
{
    const auto grad_cos_kernel = stage_matrix<Scalar>(
        grad_cos_table, "long_range_grad_cos_in");
    const auto grad_sin_kernel = stage_matrix<Scalar>(
        grad_sin_table, "long_range_grad_sin_in");
    const auto device = adjoint_geometry_device(
        cache, grad_cos_kernel, grad_sin_kernel);
    return download_matrix<double>(
        device, "long_range_grad_positions_host");
}

template <typename Scalar>
LongRangeKernelMatrix<Scalar>
PolarLongRangeOpsT<Scalar>::adjoint_geometry_device(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& grad_cos_table,
    const LongRangeKernelMatrix<Scalar>& grad_sin_table) const
{
    const std::size_t n_k = cache.n_k;
    const std::size_t n_nodes = cache.n_nodes;
    if (grad_cos_table.extent(0) != n_k
        || grad_cos_table.extent(1) != n_nodes
        || grad_sin_table.extent(0) != n_k
        || grad_sin_table.extent(1) != n_nodes)
        throw std::invalid_argument(
            "adjoint_geometry: the table seeds must carry the cache's "
            "[n_k, n_nodes] shape.");

    // coefficient(i, j) = gsin(i, j) * cos(i, j) - gcos(i, j) * sin(i, j):
    // d cos(k.r)/dr = -k sin(k.r) and d sin(k.r)/dr = +k cos(k.r).  The
    // masked tables zero the cross-graph entries, so their (formally
    // nonzero) seeds contribute nothing -- the mask information rides in
    // the tables themselves.
    auto coefficient = workspace_matrix(
        "long_range_geometry_adjoint_coeff", n_k, n_nodes);
    auto cos_table = cache.cos_table;
    auto sin_table = cache.sin_table;
    Kokkos::parallel_for(
        "long_range geometry adjoint coefficient",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
#if defined(_OPENMP)
#pragma omp simd
#endif
            for (std::size_t j = 0; j < n_nodes; ++j)
                coefficient(i, j) = grad_sin_table(i, j) * cos_table(i, j)
                    - grad_cos_table(i, j) * sin_table(i, j);
        });

    // dL/dr = coefficient^T @ K: one transposed GEMM over the same
    // [n_k, n] x [n_k, 3] shapes the forward cache fills.
    auto grad_positions = workspace_matrix(
        "long_range_grad_positions", n_nodes, 3);
    KokkosBlas::gemm(
        "T", "N", 1.0, coefficient, cache.k_vectors, 0.0, grad_positions);
    return grad_positions;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::AssembleAdjointResult PolarLongRangeOpsT<Scalar>::assemble_adjoint(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_density,
    const LongRangeConstMatrix<double>& multipoles,
    const LongRangeKernelMatrix<Scalar>* grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>* grad_sin_accumulator) const
{
    if ((grad_cos_accumulator == nullptr) != (grad_sin_accumulator == nullptr))
        throw std::invalid_argument(
            "assemble adjoint: both device table accumulators are required.");
    if (multipoles.extent(1) != density_width_)
        throw std::invalid_argument(
            "assemble adjoint: multipoles must carry the density width.");
    if (grad_density.extent(0) != cache.n_k || grad_density.extent(1) != 2)
        throw std::invalid_argument(
            "assemble adjoint: the density seeds must carry [n_k, 2].");
    const std::size_t n_k = cache.n_k;
    const std::size_t width = density_width_;
    const double scale = two_pi_cubed();
    const auto grad_density_kernel = stage_matrix<Scalar>(
        grad_density, "long_range_grad_density_in");
    const auto multipoles_kernel = stage_matrix<Scalar>(
        multipoles, "long_range_multipoles_in");

    // The broadcast-combine adjoint: with s = (2 pi)^3 / V_k, the seeds of
    // the coefficient GEMM outputs are gcc = s*(g0*fR + g1*fI) and
    // gcs = s*(g0*fI - g1*fR) -- the exact transpose of the forward's
    // per-k combine.
    LongRangeKernelMatrix<Scalar> grad_coeff_cos(
        "long_range_grad_coeff_cos", n_k, density_width_);
    LongRangeKernelMatrix<Scalar> grad_coeff_sin(
        "long_range_grad_coeff_sin", n_k, density_width_);
    auto factors = cache.density_fourier_factors;
    auto volume_per_k = cache.volume_per_k;
    Kokkos::parallel_for(
        "long_range assemble adjoint combine",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double s = scale / volume_per_k(i);
            const double g0 = grad_density_kernel(i, 0);
            const double g1 = grad_density_kernel(i, 1);
#if defined(_OPENMP)
#pragma omp simd
#endif
            for (std::size_t m = 0; m < width; ++m) {
                const double factor_real = factors(i, 0, m, 0);
                const double factor_imaginary = factors(i, 0, m, 1);
                grad_coeff_cos(i, m) =
                    s * (g0 * factor_real + g1 * factor_imaginary);
                grad_coeff_sin(i, m) =
                    s * (g0 * factor_imaginary - g1 * factor_real);
            }
        });

    // The thin-GEMM adjoints: the table seeds are gcc @ M^T and
    // gcs @ M^T, and dL/dM = cos^T @ gcc + sin^T @ gcs (accumulated, one
    // fp64 GEMM each, the same shapes as the forwards).
    LongRangeKernelMatrix<Scalar> grad_cos_table(
        "long_range_grad_cos_table", n_k, cache.n_nodes);
    LongRangeKernelMatrix<Scalar> grad_sin_table(
        "long_range_grad_sin_table", n_k, cache.n_nodes);
    LongRangeKernelMatrix<Scalar> grad_multipoles(
        "long_range_grad_multipoles", cache.n_nodes, density_width_);
    KokkosBlas::gemm("N", "T", 1.0, grad_coeff_cos, multipoles_kernel, 0.0,
        grad_cos_table);
    KokkosBlas::gemm("N", "T", 1.0, grad_coeff_sin, multipoles_kernel, 0.0,
        grad_sin_table);
    KokkosBlas::gemm("T", "N", 1.0, cache.cos_table, grad_coeff_cos, 0.0,
        grad_multipoles);
    KokkosBlas::gemm("T", "N", 1.0, cache.sin_table, grad_coeff_sin, 1.0,
        grad_multipoles);
    AssembleAdjointResult result;
    if (grad_cos_accumulator != nullptr) {
        accumulate_kernel_matrix(*grad_cos_accumulator, grad_cos_table,
            "long_range accumulate assemble cos seed");
        accumulate_kernel_matrix(*grad_sin_accumulator, grad_sin_table,
            "long_range accumulate assemble sin seed");
    } else {
        result.grad_cos_table = download_matrix<double>(
            grad_cos_table, "long_range_grad_cos_host");
        result.grad_sin_table = download_matrix<double>(
            grad_sin_table, "long_range_grad_sin_host");
    }
    result.grad_multipoles = download_matrix<double>(
        grad_multipoles, "long_range_grad_multipoles_host");
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::AssembleAdjointResult PolarLongRangeOpsT<Scalar>::adjoint_assemble(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_density,
    const LongRangeConstMatrix<double>& multipoles) const
{
    return assemble_adjoint(cache, grad_density, multipoles);
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::AssembleAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_assemble_accumulate(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_density,
    const LongRangeConstMatrix<double>& multipoles,
    const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator) const
{
    return assemble_adjoint(cache, grad_density, multipoles,
        &grad_cos_accumulator, &grad_sin_accumulator);
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::adjoint_coulomb(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_potential) const
{
    if (grad_potential.extent(0) != cache.n_k
        || grad_potential.extent(1) != 2)
        throw std::invalid_argument(
            "adjoint_coulomb: the potential seeds must carry [n_k, 2].");
    const auto grad_potential_kernel = stage_matrix<Scalar>(
        grad_potential, "long_range_grad_potential_in");
    LongRangeKernelMatrix<Scalar> grad_density(
        "long_range_grad_density", cache.n_k, 2);
    auto factor = cache.k_factor_coulomb;
    Kokkos::parallel_for(
        "long_range coulomb adjoint",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, cache.n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            // The kernel is diagonal in k with the origin mode zeroed
            // exactly; the adjoint carries the identical factor.
            grad_density(i, 0) =
                grad_potential_kernel(i, 0) * factor(i) * field_constant;
            grad_density(i, 1) =
                grad_potential_kernel(i, 1) * factor(i) * field_constant;
        });
    return download_matrix<double>(
        grad_density, "long_range_grad_density_host");
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::ProjectAdjointResult PolarLongRangeOpsT<Scalar>::project_adjoint(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_features,
    const LongRangeConstMatrix<double>& potential,
    const LongRangeKernelMatrix<Scalar>* grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>* grad_sin_accumulator) const
{
    if ((grad_cos_accumulator == nullptr) != (grad_sin_accumulator == nullptr))
        throw std::invalid_argument(
            "project adjoint: both device table accumulators are required.");
    const std::size_t n_k = cache.n_k;
    const std::size_t n_nodes = cache.n_nodes;
    const std::size_t feature_dim = feature_dim_;
    if (grad_features.extent(0) != n_nodes
        || grad_features.extent(1) != feature_dim)
        throw std::invalid_argument(
            "adjoint_project: the feature seeds must carry "
            "[n_nodes, feature_dim].");
    if (potential.extent(0) != n_k || potential.extent(1) != 2)
        throw std::invalid_argument(
            "adjoint_project: the potential must carry [n_k, 2].");

    // The permutation-and-scale adjoint: features(n, j) = scale *
    // projections(perm(j), n), so the projection seed is the scatter of
    // the scaled feature seed through the same fixed permutation buffer.
    const double scale = 2.0 / two_pi_cubed();
    const auto grad_features_kernel = stage_matrix<Scalar>(
        grad_features, "long_range_grad_features_in");
    LongRangeKernelMatrix<Scalar> grad_projections(
        "long_range_grad_projections", feature_dim, n_nodes);
    auto permutation_v = output_permutation_kernel_;
    Kokkos::parallel_for(
        "long_range project adjoint scatter",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes),
        KOKKOS_LAMBDA(const std::size_t n) {
            for (std::size_t j = 0; j < feature_dim; ++j)
                grad_projections(permutation_v(j), n) =
                    scale * grad_features_kernel(n, j);
        });

    // The transposed-GEMM adjoints, both directions at once: the seed of
    // the amplitudes is the tables times the projection seed
    // (Ga = cos @ Gp^T, Gb = sin @ Gp^T), and the table seeds are the
    // forward amplitudes times the projection seed (Gcos = a @ Gp,
    // Gsin = b @ Gp).
    const auto potential_kernel = stage_matrix<Scalar>(
        potential, "long_range_potential_in");
    LongRangeKernelMatrix<Scalar> amplitude_a(
        "long_range_proj_a", n_k, feature_dim);
    LongRangeKernelMatrix<Scalar> amplitude_b(
        "long_range_proj_b", n_k, feature_dim);
    projection_amplitudes(cache, potential_kernel, amplitude_a, amplitude_b);
    LongRangeKernelMatrix<Scalar> grad_amplitude_a(
        "long_range_grad_proj_a", n_k, feature_dim);
    LongRangeKernelMatrix<Scalar> grad_amplitude_b(
        "long_range_grad_proj_b", n_k, feature_dim);
    KokkosBlas::gemm("N", "T", 1.0, cache.cos_table, grad_projections, 0.0,
        grad_amplitude_a);
    KokkosBlas::gemm("N", "T", 1.0, cache.sin_table, grad_projections, 0.0,
        grad_amplitude_b);
    LongRangeKernelMatrix<Scalar> grad_cos_table(
        "long_range_grad_cos_table", n_k, n_nodes);
    LongRangeKernelMatrix<Scalar> grad_sin_table(
        "long_range_grad_sin_table", n_k, n_nodes);
    KokkosBlas::gemm("N", "N", 1.0, amplitude_a, grad_projections, 0.0,
        grad_cos_table);
    KokkosBlas::gemm("N", "N", 1.0, amplitude_b, grad_projections, 0.0,
        grad_sin_table);

    // The amplitude-combine adjoint: Gpot0 = fp * sum_m (Ga*fR + Gb*fI)
    // and Gpot1 = fp * sum_m (Ga*fI - Gb*fR), fp64 reductions per k row.
    LongRangeKernelMatrix<Scalar> grad_potential(
        "long_range_grad_potential", n_k, 2);
    const std::size_t feature_width = feature_width_;
    auto factors = cache.feature_fourier_factors;
    auto factor_proj = cache.k_factor_proj;
    Kokkos::parallel_for(
        "long_range project adjoint combine",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double s = factor_proj(i);
            double g0 = 0.0;
            double g1 = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : g0, g1)
#endif
            for (std::size_t m = 0; m < feature_dim; ++m) {
                const std::size_t sigma = m / feature_width;
                const std::size_t channel = m % feature_width;
                const double factor_real = factors(i, sigma, channel, 0);
                const double factor_imaginary =
                    factors(i, sigma, channel, 1);
                g0 += grad_amplitude_a(i, m) * factor_real
                    + grad_amplitude_b(i, m) * factor_imaginary;
                g1 += grad_amplitude_a(i, m) * factor_imaginary
                    - grad_amplitude_b(i, m) * factor_real;
            }
            grad_potential(i, 0) = s * g0;
            grad_potential(i, 1) = s * g1;
        });
    ProjectAdjointResult result;
    if (grad_cos_accumulator != nullptr) {
        accumulate_kernel_matrix(*grad_cos_accumulator, grad_cos_table,
            "long_range accumulate project cos seed");
        accumulate_kernel_matrix(*grad_sin_accumulator, grad_sin_table,
            "long_range accumulate project sin seed");
    } else {
        result.grad_cos_table = download_matrix<double>(
            grad_cos_table, "long_range_grad_cos_host");
        result.grad_sin_table = download_matrix<double>(
            grad_sin_table, "long_range_grad_sin_host");
    }
    result.grad_potential = download_matrix<double>(
        grad_potential, "long_range_grad_potential_host");
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::ProjectAdjointResult PolarLongRangeOpsT<Scalar>::adjoint_project(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_features,
    const LongRangeConstMatrix<double>& potential) const
{
    return project_adjoint(cache, grad_features, potential, nullptr, nullptr);
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::ProjectAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_project_accumulate(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_features,
    const LongRangeConstMatrix<double>& potential,
    const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator) const
{
    return project_adjoint(cache, grad_features, potential,
        &grad_cos_accumulator, &grad_sin_accumulator);
}


template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::EnergyAdjointResult
PolarLongRangeOpsT<Scalar>::coulomb_energy_adjoint(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstVector<double>& grad_energy,
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeMatrix<double>* energy_self_fields,
    const LongRangeKernelMatrix<Scalar>* grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>* grad_sin_accumulator) const
{
    if ((grad_cos_accumulator == nullptr) != (grad_sin_accumulator == nullptr))
        throw std::invalid_argument(
            "energy adjoint: both device table accumulators are required.");
    if (grad_energy.extent(0) != cache.n_graphs)
        throw std::invalid_argument(
            "adjoint_coulomb_energy: the energy seeds must carry "
            "n_graphs.");

    // The forward's own intermediates, recomputed exactly as the forward
    // computes them (the shared-cache contract: the adjoint consumes the
    // same cache, never a rebuilt one).
    const LongRangeMatrix<double> density =
        assemble_fourier(cache, charge_density);
    const LongRangeMatrix<double> potential = apply_coulomb(cache, density);

    // The index_add reduction adjoint: E(g) = V_g * 0.5/(2 pi)^6 *
    // sum_{k in g} 2*Re(rho_bar . V), so the per-k product seed is the
    // broadcast graph seed times 2*V_g*0.5/(2 pi)^6.  It lands on both
    // factors; the potential factor flows through the diagonal Coulomb
    // kernel onto the density (the potential is not an independent
    // input), giving the density seed below.
    const std::size_t n_k = cache.n_k;
    const double scale = 0.5 / (two_pi_cubed() * two_pi_cubed());
    const auto density_kernel =
        stage_matrix<Scalar>(density, "long_range_energy_density");
    const auto potential_kernel =
        stage_matrix<Scalar>(potential, "long_range_energy_potential");
    const auto grad_energy_kernel =
        stage_vector<Scalar>(grad_energy, "long_range_energy_grad_seed");
    LongRangeKernelMatrix<Scalar> grad_density_kernel(
        "long_range_energy_grad_density", n_k, 2);
    auto factor = cache.k_factor_coulomb;
    auto k_graph_index = cache.k_graph_index;
    auto volumes = cache.kernel_volumes;
    Kokkos::parallel_for(
        "long_range energy adjoint per-k",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const std::size_t g =
                static_cast<std::size_t>(k_graph_index(i));
            const double c =
                2.0 * scale * volumes(g) * grad_energy_kernel(g);
            grad_density_kernel(i, 0) = (c * potential_kernel(i, 0))
                + (c * density_kernel(i, 0)) * factor(i) * field_constant;
            grad_density_kernel(i, 1) = (c * potential_kernel(i, 1))
                + (c * density_kernel(i, 1)) * factor(i) * field_constant;
        });
    const auto grad_density = download_matrix<double>(
        grad_density_kernel, "long_range_energy_grad_density_host");

    // The shared assemble adjoint -- the one spelling of the step both
    // the feature and energy paths backpropagate through.
    EnergyAdjointResult result;
    AssembleAdjointResult assembled =
        assemble_adjoint(cache, grad_density, charge_density,
            grad_cos_accumulator, grad_sin_accumulator);
    result.grad_cos_table = std::move(assembled.grad_cos_table);
    result.grad_sin_table = std::move(assembled.grad_sin_table);
    result.grad_charge_density = std::move(assembled.grad_multipoles);

    // The self-interaction subtraction is constant-linear in the charge
    // density: E -= 0.5 * sum_n sum_m charge(n, m) * self_fields(n, m),
    // so its adjoint contributes -0.5 * seed * self_fields per node.
    if (energy_self_fields != nullptr) {
        for (std::size_t n = 0; n < cache.n_nodes; ++n) {
            const double seed = -0.5
                * grad_energy(static_cast<std::size_t>(
                    cache.node_graph_index_host(n)));
            for (std::size_t m = 0; m < density_width_; ++m)
                result.grad_charge_density(n, m) +=
                    seed * (*energy_self_fields)(n, m);
        }
    }
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::EnergyAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_coulomb_energy(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstVector<double>& grad_energy,
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeMatrix<double>* energy_self_fields) const
{
    return coulomb_energy_adjoint(cache, grad_energy, charge_density,
        energy_self_fields, nullptr, nullptr);
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::EnergyAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_coulomb_energy_accumulate(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstVector<double>& grad_energy,
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeMatrix<double>* energy_self_fields,
    const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator) const
{
    return coulomb_energy_adjoint(cache, grad_energy, charge_density,
        energy_self_fields, &grad_cos_accumulator, &grad_sin_accumulator);
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::ComposedEnergyAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_coulomb_energy_composed(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstVector<double>& grad_energy,
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeMatrix<double>* energy_self_fields,
    const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator) const
{
    if (grad_energy.extent(0) != cache.n_graphs
        || charge_density.extent(0) != cache.n_nodes
        || charge_density.extent(1) != density_width_)
        throw std::invalid_argument(
            "composed energy adjoint inputs do not match the cache.");
    const std::size_t n_k = cache.n_k;
    const std::size_t n_nodes = cache.n_nodes;
    if (grad_cos_accumulator.extent(0) != n_k
        || grad_cos_accumulator.extent(1) != n_nodes)
        throw std::invalid_argument(
            "long_range composed energy accumulate cos: the device "
            "accumulator shape does not match the table seed.");
    if (grad_sin_accumulator.extent(0) != n_k
        || grad_sin_accumulator.extent(1) != n_nodes)
        throw std::invalid_argument(
            "long_range composed energy accumulate sin: the device "
            "accumulator shape does not match the table seed.");
    const std::size_t width = density_width_;
    const double energy_scale =
        0.5 / (two_pi_cubed() * two_pi_cubed());
    const double assemble_scale = two_pi_cubed();
    auto charge_kernel = workspace_matrix("long_range_composed_energy_charge",
        charge_density.extent(0), charge_density.extent(1));
    stage_matrix_into<Scalar>(charge_density, charge_kernel);
    auto energy_seed = workspace_vector(
        "long_range_composed_energy_seed", grad_energy.extent(0));
    stage_vector_into<Scalar>(grad_energy, energy_seed);
    const auto density = assemble_fourier_kernel(
        cache, charge_kernel, /*use_workspace=*/true);
    const auto potential = apply_coulomb_kernel(
        cache, density, /*use_workspace=*/true);
    auto k_graph_index = cache.k_graph_index;
    auto volumes = cache.kernel_volumes;
    auto coulomb_factor = cache.k_factor_coulomb;

    auto grad_density = workspace_matrix(
        "long_range_composed_energy_grad_density", n_k, 2);
    Kokkos::parallel_for(
        "long_range composed energy adjoint per-k",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const std::size_t graph =
                static_cast<std::size_t>(k_graph_index(i));
            const double c =
                2.0 * energy_scale * volumes(graph) * energy_seed(graph);
            grad_density(i, 0) = c * potential(i, 0)
                + c * density(i, 0) * coulomb_factor(i) * field_constant;
            grad_density(i, 1) = c * potential(i, 1)
                + c * density(i, 1) * coulomb_factor(i) * field_constant;
        });

    // Assemble reverse and table accumulation, with all [n_k, 2]
    // intermediates resident.
    auto grad_coeff_cos = workspace_matrix(
        "long_range_composed_energy_grad_coeff_cos", n_k, width);
    auto grad_coeff_sin = workspace_matrix(
        "long_range_composed_energy_grad_coeff_sin", n_k, width);
    auto factors = cache.density_fourier_factors;
    auto volume_per_k = cache.volume_per_k;
    Kokkos::parallel_for(
        "long_range composed energy assemble adjoint combine",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double s = assemble_scale / volume_per_k(i);
            const double g0 = grad_density(i, 0);
            const double g1 = grad_density(i, 1);
            for (std::size_t m = 0; m < width; ++m) {
                const double fr = factors(i, 0, m, 0);
                const double fi = factors(i, 0, m, 1);
                grad_coeff_cos(i, m) = s * (g0 * fr + g1 * fi);
                grad_coeff_sin(i, m) = s * (g0 * fi - g1 * fr);
            }
        });
    auto grad_cos = workspace_matrix(
        "long_range_composed_energy_cos", n_k, n_nodes);
    auto grad_sin = workspace_matrix(
        "long_range_composed_energy_sin", n_k, n_nodes);
    auto grad_charge = workspace_matrix(
        "long_range_composed_energy_grad_charge", n_nodes, width);
    KokkosBlas::gemm(
        "N", "T", 1.0, grad_coeff_cos, charge_kernel, 0.0, grad_cos);
    KokkosBlas::gemm(
        "N", "T", 1.0, grad_coeff_sin, charge_kernel, 0.0, grad_sin);
    KokkosBlas::gemm("T", "N", 1.0, cache.cos_table, grad_coeff_cos, 0.0,
        grad_charge);
    KokkosBlas::gemm("T", "N", 1.0, cache.sin_table, grad_coeff_sin, 1.0,
        grad_charge);
    const std::size_t table_size = grad_cos_accumulator.size();
    Kokkos::parallel_for("long_range composed energy accumulate tables",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, table_size),
        KOKKOS_LAMBDA(const std::size_t index) {
            grad_cos_accumulator.data()[index] += grad_cos.data()[index];
            grad_sin_accumulator.data()[index] += grad_sin.data()[index];
        });

    // Cell metadata from the assemble factors plus the energy reduction's
    // direct Coulomb-kernel and volume-prefactor terms.
    auto coeff_cos = workspace_matrix(
        "long_range_composed_energy_coeff_cos", n_k, width);
    auto coeff_sin = workspace_matrix(
        "long_range_composed_energy_coeff_sin", n_k, width);
    KokkosBlas::gemm(
        "N", "N", 1.0, cache.cos_table, charge_kernel, 0.0, coeff_cos);
    KokkosBlas::gemm(
        "N", "N", 1.0, cache.sin_table, charge_kernel, 0.0, coeff_sin);
    auto grad_k = workspace_matrix(
        "long_range_composed_energy_grad_k", n_k, 3);
    auto grad_volume = workspace_vector(
        "long_range_composed_energy_grad_volume", n_k);
    const double sigma = density_smearing_width_;
    const double sigma2 = sigma * sigma;
    const double pref0 =
        (4.0 * pi) * std::sqrt(pi / 2.0) * sigma * sigma * sigma;
    const double a0 =
        pref0 * cl_sigma(0, sigma, false) * harmonic_l0();
    const double b = pref0 * sigma2 * cl_sigma(1, sigma, false)
        * harmonic_l1();
    const bool with_l1 = density_max_l_ >= 1;
    auto k_norm2 = cache.k_norm2;
    auto k_vectors = cache.k_vectors;
    Kokkos::parallel_for(
        "long_range composed energy cell seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const std::size_t graph =
                static_cast<std::size_t>(k_graph_index(i));
            const double g0 = grad_density(i, 0);
            const double g1 = grad_density(i, 1);
            grad_volume(i) = -(g0 * density(i, 0) + g1 * density(i, 1))
                    / volume_per_k(i)
                + energy_scale * energy_seed(graph) * 2.0
                    * (density(i, 0) * potential(i, 0)
                        + density(i, 1) * potential(i, 1));
            const double s = assemble_scale / volume_per_k(i);
            const double exponential = Kokkos::exp(-0.5 * k_norm2(i) * sigma2);
            double direct_factor = 0.0;
            if (k_norm2(i) != 0.0) {
                const double c = 2.0 * energy_scale * volumes(graph)
                    * energy_seed(graph);
                const double rho2 = density(i, 0) * density(i, 0)
                    + density(i, 1) * density(i, 1);
                direct_factor = c * -2.0 * field_constant * rho2
                    / (k_norm2(i) * k_norm2(i));
            }
            for (int j = 0; j < 3; ++j) {
                const double df0 = -sigma2 * k_vectors(i, j) * a0 * exponential;
                double dre = df0 * coeff_cos(i, 0);
                double dim = -df0 * coeff_sin(i, 0);
                if (with_l1) {
                    for (int p = 0; p < 3; ++p) {
                        const int q = p == 0 ? 1 : (p == 1 ? 2 : 0);
                        const double dfi = -b * exponential
                            * ((q == j ? 1.0 : 0.0)
                                - sigma2 * k_vectors(i, q)
                                    * k_vectors(i, j));
                        dre += dfi * coeff_sin(i, 1 + p);
                        dim += dfi * coeff_cos(i, 1 + p);
                    }
                }
                grad_k(i, j) = s * (g0 * dre + g1 * dim)
                    + direct_factor * k_vectors(i, j);
            }
        });

    auto grad_strain_kernel = workspace_tensor3(
        "long_range_composed_energy_strain", cache.n_graphs, 3, 3);
    const std::size_t n_graphs = cache.n_graphs;
    Kokkos::parallel_for(
        "long_range composed energy strain reduction",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_graphs * 9),
        KOKKOS_LAMBDA(const std::size_t index) {
            const std::size_t graph = index / 9;
            const int a = static_cast<int>((index % 9) / 3);
            const int component = static_cast<int>(index % 3);
            double value = 0.0;
            for (std::size_t i = 0; i < n_k; ++i) {
                if (static_cast<std::size_t>(k_graph_index(i)) != graph)
                    continue;
                value -= grad_k(i, a) * k_vectors(i, component);
            }
            if (a == component) {
                double volume_seed = 0.0;
                for (std::size_t i = 0; i < n_k; ++i)
                    if (static_cast<std::size_t>(k_graph_index(i)) == graph)
                        volume_seed += grad_volume(i);
                value += volumes(graph) * volume_seed;
            }
            grad_strain_kernel(graph, a, component) = value;
        });

    ComposedEnergyAdjointResult result;
    result.grad_charge_density = download_matrix<double>(
        grad_charge, "long_range_composed_energy_grad_charge_host");
    result.grad_strain = download_tensor3<double>(grad_strain_kernel,
        "long_range_composed_energy_strain_host");
    if (energy_self_fields != nullptr) {
        for (std::size_t n = 0; n < n_nodes; ++n) {
            const std::size_t graph = static_cast<std::size_t>(
                cache.node_graph_index_host(n));
            const double seed = -0.5 * grad_energy(graph);
            for (std::size_t m = 0; m < width; ++m)
                result.grad_charge_density(n, m) +=
                    seed * (*energy_self_fields)(n, m);
        }
    }
    return result;
}

template <typename Scalar>
LongRangeKernelMatrix<Scalar>
PolarLongRangeOpsT<Scalar>::adjoint_coulomb_energy_composed_device(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& charge,
    const bool subtract_self,
    const LongRangeKernelVector<Scalar>& self_constants,
    const LongRangeKernelVector<std::size_t>& self_select,
    const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
    const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator,
    const LongRangeKernelTensor3<double>& grad_strain) const
{
    const std::size_t n_k = cache.n_k;
    const std::size_t n_nodes = cache.n_nodes;
    if (grad_cos_accumulator.extent(0) != n_k
        || grad_cos_accumulator.extent(1) != n_nodes)
        throw std::invalid_argument(
            "long_range resident energy accumulate cos: the device "
            "accumulator shape does not match the table seed.");
    if (grad_sin_accumulator.extent(0) != n_k
        || grad_sin_accumulator.extent(1) != n_nodes)
        throw std::invalid_argument(
            "long_range resident energy accumulate sin: the device "
            "accumulator shape does not match the table seed.");
    const std::size_t width = density_width_;
    const double energy_scale =
        0.5 / (two_pi_cubed() * two_pi_cubed());
    const double assemble_scale = two_pi_cubed();
    const auto density = assemble_fourier_kernel(
        cache, charge, /*use_workspace=*/true);
    const auto potential = apply_coulomb_kernel(
        cache, density, /*use_workspace=*/true);
    auto k_graph_index = cache.k_graph_index;
    auto volumes = cache.kernel_volumes;
    auto coulomb_factor = cache.k_factor_coulomb;
    auto grad_density = workspace_matrix(
        "long_range_resident_energy_grad_density", n_k, 2);
    Kokkos::parallel_for("long_range resident energy adjoint per-k",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const std::size_t graph =
                static_cast<std::size_t>(k_graph_index(i));
            const double c = 2.0 * energy_scale * volumes(graph);
            grad_density(i, 0) = c * potential(i, 0)
                + c * density(i, 0) * coulomb_factor(i) * field_constant;
            grad_density(i, 1) = c * potential(i, 1)
                + c * density(i, 1) * coulomb_factor(i) * field_constant;
        });
    auto grad_coeff_cos = workspace_matrix(
        "long_range_resident_energy_grad_coeff_cos", n_k, width);
    auto grad_coeff_sin = workspace_matrix(
        "long_range_resident_energy_grad_coeff_sin", n_k, width);
    auto factors = cache.density_fourier_factors;
    auto volume_per_k = cache.volume_per_k;
    Kokkos::parallel_for("long_range resident energy assemble adjoint",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double s = assemble_scale / volume_per_k(i);
            const double g0 = grad_density(i, 0);
            const double g1 = grad_density(i, 1);
            for (std::size_t m = 0; m < width; ++m) {
                const double fr = factors(i, 0, m, 0);
                const double fi = factors(i, 0, m, 1);
                grad_coeff_cos(i, m) = s * (g0 * fr + g1 * fi);
                grad_coeff_sin(i, m) = s * (g0 * fi - g1 * fr);
            }
        });
    auto grad_cos = workspace_matrix(
        "long_range_resident_energy_cos", n_k, n_nodes);
    auto grad_sin = workspace_matrix(
        "long_range_resident_energy_sin", n_k, n_nodes);
    auto grad_charge = workspace_matrix(
        "long_range_resident_energy_grad_charge", n_nodes, width);
    KokkosBlas::gemm("N", "T", 1.0, grad_coeff_cos, charge, 0.0, grad_cos);
    KokkosBlas::gemm("N", "T", 1.0, grad_coeff_sin, charge, 0.0, grad_sin);
    KokkosBlas::gemm("T", "N", 1.0, cache.cos_table, grad_coeff_cos, 0.0,
        grad_charge);
    KokkosBlas::gemm("T", "N", 1.0, cache.sin_table, grad_coeff_sin, 1.0,
        grad_charge);
    const std::size_t table_size = grad_cos_accumulator.size();
    Kokkos::parallel_for("long_range resident energy accumulate tables",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, table_size),
        KOKKOS_LAMBDA(const std::size_t index) {
            grad_cos_accumulator.data()[index] += grad_cos.data()[index];
            grad_sin_accumulator.data()[index] += grad_sin.data()[index];
        });

    auto coeff_cos = workspace_matrix(
        "long_range_resident_energy_coeff_cos", n_k, width);
    auto coeff_sin = workspace_matrix(
        "long_range_resident_energy_coeff_sin", n_k, width);
    KokkosBlas::gemm("N", "N", 1.0, cache.cos_table, charge, 0.0, coeff_cos);
    KokkosBlas::gemm("N", "N", 1.0, cache.sin_table, charge, 0.0, coeff_sin);
    auto grad_k = workspace_matrix(
        "long_range_resident_energy_grad_k", n_k, 3);
    auto grad_volume = workspace_vector(
        "long_range_resident_energy_grad_volume", n_k);
    const double sigma = density_smearing_width_;
    const double sigma2 = sigma * sigma;
    const double pref0 =
        (4.0 * pi) * std::sqrt(pi / 2.0) * sigma * sigma * sigma;
    const double a0 = pref0 * cl_sigma(0, sigma, false) * harmonic_l0();
    const double b = pref0 * sigma2 * cl_sigma(1, sigma, false)
        * harmonic_l1();
    const bool with_l1 = density_max_l_ >= 1;
    auto k_norm2 = cache.k_norm2;
    auto k_vectors = cache.k_vectors;
    Kokkos::parallel_for("long_range resident energy cell seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const std::size_t graph =
                static_cast<std::size_t>(k_graph_index(i));
            const double g0 = grad_density(i, 0);
            const double g1 = grad_density(i, 1);
            grad_volume(i) = -(g0 * density(i, 0) + g1 * density(i, 1))
                    / volume_per_k(i)
                + energy_scale * 2.0
                    * (density(i, 0) * potential(i, 0)
                        + density(i, 1) * potential(i, 1));
            const double s = assemble_scale / volume_per_k(i);
            const double exponential = Kokkos::exp(-0.5 * k_norm2(i) * sigma2);
            double direct_factor = 0.0;
            if (k_norm2(i) != 0.0) {
                const double rho2 = density(i, 0) * density(i, 0)
                    + density(i, 1) * density(i, 1);
                direct_factor = 2.0 * energy_scale * volumes(graph)
                    * -2.0 * field_constant * rho2
                    / (k_norm2(i) * k_norm2(i));
            }
            for (int j = 0; j < 3; ++j) {
                const double df0 = -sigma2 * k_vectors(i, j) * a0 * exponential;
                double dre = df0 * coeff_cos(i, 0);
                double dim = -df0 * coeff_sin(i, 0);
                if (with_l1)
                    for (int p = 0; p < 3; ++p) {
                        const int q = p == 0 ? 1 : (p == 1 ? 2 : 0);
                        const double dfi = -b * exponential
                            * ((q == j ? 1.0 : 0.0)
                                - sigma2 * k_vectors(i, q) * k_vectors(i, j));
                        dre += dfi * coeff_sin(i, 1 + p);
                        dim += dfi * coeff_cos(i, 1 + p);
                    }
                grad_k(i, j) = s * (g0 * dre + g1 * dim)
                    + direct_factor * k_vectors(i, j);
            }
        });
    const std::size_t n_graphs = cache.n_graphs;
    constexpr std::size_t strain_component_count = 9;
    if (n_graphs
        > static_cast<std::size_t>(std::numeric_limits<int>::max())
            / strain_component_count)
        throw std::length_error(
            "long-range energy strain team count exceeds INT_MAX.");
    const int strain_team_count =
        static_cast<int>(n_graphs * strain_component_count);
    using EnergyStrainPolicy = Kokkos::TeamPolicy<LongRangeKernelSpace>;
    using EnergyStrainMember = typename EnergyStrainPolicy::member_type;
    Kokkos::parallel_for("long_range resident energy strain reduction",
        EnergyStrainPolicy(strain_team_count, Kokkos::AUTO),
        KOKKOS_LAMBDA(const EnergyStrainMember& team) {
            const std::size_t index = team.league_rank();
            const std::size_t graph = index / 9;
            const int a = static_cast<int>((index % 9) / 3);
            const int component = static_cast<int>(index % 3);
            double value = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, n_k),
                [=](const std::size_t i, double& sum) {
                    if (static_cast<std::size_t>(k_graph_index(i)) != graph)
                        return;
                    sum -= double(grad_k(i, a))
                        * double(k_vectors(i, component));
                },
                value);
            double volume_seed = 0.0;
            if (a == component)
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, n_k),
                    [=](const std::size_t i, double& sum) {
                        if (static_cast<std::size_t>(k_graph_index(i)) == graph)
                            sum += grad_volume(i);
                    },
                    volume_seed);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                grad_strain(graph, a, component) = value
                    + double(volumes(graph)) * volume_seed;
            });
        });
    if (subtract_self)
        Kokkos::parallel_for("long_range resident self energy adjoint",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(0, n_nodes),
            KOKKOS_LAMBDA(const std::size_t node) {
                for (std::size_t term = 0;
                    term < self_constants.extent(0); ++term) {
                    const std::size_t selected = self_select(term);
                    const Scalar c = self_constants(term);
                    grad_charge(node, term) -= Scalar(0.5) * c
                        * charge(node, selected);
                    grad_charge(node, selected) -= Scalar(0.5) * c
                        * charge(node, term);
                }
            });
    return grad_charge;
}


template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::adjoint_feature_self_terms(
    const LongRangeConstMatrix<double>& grad_terms) const
{
    const std::size_t n_nodes = grad_terms.extent(0);
    LongRangeMatrix<double> grad_multipoles(
        "long_range_grad_si_feature_multipoles", n_nodes, density_width_);
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, grad_multipoles, 0.0);
    // The transpose of the gather-and-scale: the scaled seeds scatter
    // onto the gathered columns (the constants are init-time data).
    for (std::size_t n = 0; n < n_nodes; ++n)
        for (std::size_t d = 0; d < si_feature_constants_.extent(0); ++d)
            grad_multipoles(n, si_feature_select_[d]) +=
                si_feature_constants_(d) * grad_terms(n, d);
    return grad_multipoles;
}

template <typename Scalar>
LongRangeMatrix<double> PolarLongRangeOpsT<Scalar>::adjoint_energy_self_fields(
    const LongRangeConstMatrix<double>& grad_fields) const
{
    const std::size_t n_nodes = grad_fields.extent(0);
    LongRangeMatrix<double> grad_multipoles(
        "long_range_grad_si_energy_multipoles", n_nodes, density_width_);
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, grad_multipoles, 0.0);
    for (std::size_t n = 0; n < n_nodes; ++n)
        for (std::size_t d = 0; d < si_energy_constants_.extent(0); ++d)
            grad_multipoles(n, si_energy_select_[d]) +=
                si_energy_constants_(d) * grad_fields(n, d);
    return grad_multipoles;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::ExternalFieldAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_external_field(
    const LongRangeConstMatrix<double>& grad_features,
    const LongRangeConstMatrix<double>& external_potential,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeVector<double>& field_feature_norms,
    const bool use_workspace) const
{
    const std::size_t n_nodes = positions.extent(0);
    const std::size_t n_graphs = external_potential.extent(0);
    for (std::size_t n = 0; n < n_nodes; ++n)
        if (node_graph_index(n) < 0
            || static_cast<std::size_t>(node_graph_index(n)) >= n_graphs)
            throw std::invalid_argument(
                "adjoint_external_field: node_graph_index must address "
                "the external potential rows.");
    if (grad_features.extent(0) != n_nodes
        || grad_features.extent(1) != feature_dim_)
        throw std::invalid_argument(
            "adjoint_external_field: the feature seeds must carry "
            "[n_nodes, feature_dim].");
    // The same norm contract as the forward: unset means unit norms, a set
    // vector must carry one entry per feature (a wrong length is a caller
    // error, not a silent unit-norm fallback).
    if (field_feature_norms.extent(0) != 0
        && field_feature_norms.extent(0) != feature_dim_)
        throw std::invalid_argument(
            "adjoint_external_field: field_feature_norms must be empty "
            "(unit norms) or carry one entry per feature: expected "
            + std::to_string(feature_dim_) + ", got "
            + std::to_string(field_feature_norms.extent(0)) + ".");
    const bool with_norms = field_feature_norms.extent(0) != 0;
    const std::size_t feature_dim = feature_dim_;

    // Column-scaled seeds through the forward's per-spin half and norm
    // divide, then the constant-matrix transpose as one GEMM:
    // h(n, f) = sum_p matrix(p, f) * s_p * Gfeat(n, p).
    const double field_scale = 0.5;
    auto device_matrix = [this, use_workspace] (const char* label,
        const std::size_t rows, const std::size_t columns) {
        return use_workspace
            ? workspace_matrix(label, rows, columns)
            : LongRangeKernelMatrix<Scalar>(
                  Kokkos::view_alloc(std::string(label)), rows, columns);
    };
    auto device_vector = [this, use_workspace] (const char* label,
        const std::size_t size) {
        return use_workspace
            ? workspace_vector(label, size)
            : LongRangeKernelVector<Scalar>(
                  Kokkos::view_alloc(std::string(label)), size);
    };
    auto grad_features_kernel = device_matrix(
        "long_range_field_grad_features_in", n_nodes, feature_dim);
    stage_matrix_into<Scalar>(grad_features, grad_features_kernel);
    LongRangeKernelVector<Scalar> norms_kernel;
    if (with_norms) {
        norms_kernel = device_vector("long_range_field_norms", feature_dim);
        const LongRangeConstVector<double> norms_const(
            field_feature_norms.data(), field_feature_norms.extent(0));
        stage_vector_into<Scalar>(norms_const, norms_kernel);
    }
    auto scaled_seeds = device_matrix(
        "long_range_field_scaled_seeds", n_nodes, feature_dim);
    auto norms = norms_kernel;
    Kokkos::parallel_for(
        "long_range external field adjoint scale",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_nodes),
        KOKKOS_LAMBDA(const std::size_t n) {
#if defined(_OPENMP)
#pragma omp simd
#endif
            for (std::size_t p = 0; p < feature_dim; ++p) {
                double s = field_scale;
                if (with_norms)
                    s /= norms(p);
                scaled_seeds(n, p) = s * grad_features_kernel(n, p);
            }
        });
    auto node_field_seeds_kernel = device_matrix(
        "long_range_field_node_seeds", n_nodes, 4);
    KokkosBlas::gemm("N", "N", 1.0, scaled_seeds, external_field_kernel_,
        0.0, node_field_seeds_kernel);
    const auto node_field_seeds = download_matrix<double>(
        node_field_seeds_kernel, "long_range_field_node_seeds_host");

    // The barycenter of the forward (fp64 scatter mean) and the per-graph
    // fp64 sums of the V-column seed: the gauge E.(r - barycenter) is
    // linear in the positions of every node of the graph at once.
    LongRangeMatrix<double> barycenter(
        "long_range_barycenter", n_graphs, 3);
    std::vector<double> gauge_seed_sums(n_graphs, 0.0);
    std::vector<std::size_t> counts(n_graphs, 0);
    {
        std::vector<double> sums(3 * n_graphs, 0.0);
        for (std::size_t n = 0; n < n_nodes; ++n) {
            const std::size_t g =
                static_cast<std::size_t>(node_graph_index(n));
            for (int d = 0; d < 3; ++d)
                sums[3 * g + d] += positions(n, d);
            gauge_seed_sums[g] += node_field_seeds(n, 0);
            ++counts[g];
        }
        for (std::size_t g = 0; g < n_graphs; ++g)
            for (int d = 0; d < 3; ++d)
                barycenter(g, d) = counts[g] == 0
                    ? 0.0
                    : sums[3 * g + d] / static_cast<double>(counts[g]);
    }

    ExternalFieldAdjointResult result;
    result.grad_positions = LongRangeMatrix<double>(
        "long_range_field_grad_positions", n_nodes, 3);
    result.grad_external_potential = LongRangeMatrix<double>(
        "long_range_field_grad_potential", n_graphs, 4);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        result.grad_external_potential, 0.0);

    // Per-node position seed: grad_positions(n, d) = E_{g, d} * (h0(n) -
    // mean_g h0) -- the mean subtraction is the barycenter scatter's
    // adjoint.  The external-potential scatter: V takes h0, the field
    // components take their un-permuted seeds plus the gauge term's
    // displacement weights ([V, E_x, E_y, E_z] <- [h0, h2, h3, h1]).
    for (std::size_t n = 0; n < n_nodes; ++n) {
        const std::size_t g =
            static_cast<std::size_t>(node_graph_index(n));
        const double mean_gauge = counts[g] == 0
            ? 0.0
            : gauge_seed_sums[g] / static_cast<double>(counts[g]);
        // The [V, E_x, E_y, E_z] column seeds un-permute the forward's
        // [V, E_z, E_x, E_y] order: E_x <- h2, E_y <- h3, E_z <- h1.
        const std::size_t field_seed_column[3] = {2, 3, 1};
        for (int d = 0; d < 3; ++d) {
            result.grad_positions(n, d) = external_potential(g, 1 + d)
                * (node_field_seeds(n, 0) - mean_gauge);
            result.grad_external_potential(g, 1 + d) +=
                node_field_seeds(n, field_seed_column[d]);
            result.grad_external_potential(g, 1 + d) +=
                node_field_seeds(n, 0)
                    * (positions(n, d) - barycenter(g, d));
        }
        result.grad_external_potential(g, 0) += node_field_seeds(n, 0);
    }
    return result;
}


template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::DipoleAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_dipole_readout(
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeConstVector<double>& grad_total_charge,
    const LongRangeConstMatrix<double>& grad_total_dipole) const
{
    if (charge_density.extent(0) != positions.extent(0)
        || charge_density.extent(0) != node_graph_index.extent(0))
        throw std::invalid_argument(
            "adjoint_dipole_readout: density, positions, and "
            "node_graph_index must agree on the node count.");
    const std::size_t n_nodes = charge_density.extent(0);
    std::size_t n_graphs = 0;
    for (std::size_t n = 0; n < n_nodes; ++n)
        n_graphs = std::max(n_graphs,
            static_cast<std::size_t>(node_graph_index(n)) + 1);
    if (grad_total_charge.extent(0) != n_graphs
        || grad_total_dipole.extent(0) != n_graphs
        || grad_total_dipole.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_dipole_readout: the readout seeds must carry "
            "[n_graphs] and [n_graphs, 3].");

    DipoleAdjointResult result;
    result.grad_positions = LongRangeMatrix<double>(
        "long_range_dipole_grad_positions", n_nodes, 3);
    result.grad_charge_density = LongRangeMatrix<double>(
        "long_range_dipole_grad_density", n_nodes, charge_density.extent(1));
    // The readout is linear: Q = sum q carries no position dependence,
    // D += q * r gives grad_positions(n, d) = q * GD(g, d), and the l=1
    // gather keeps the e3nn (y, z, x) phase order (dipole x <- column 3,
    // y <- column 1, z <- column 2).
    const bool with_dipoles = charge_density.extent(1) > 1;
    for (std::size_t n = 0; n < n_nodes; ++n) {
        const std::size_t g =
            static_cast<std::size_t>(node_graph_index(n));
        const double q = charge_density(n, 0);
        double displacement_seed = 0.0;
        for (int d = 0; d < 3; ++d) {
            result.grad_positions(n, d) = q * grad_total_dipole(g, d);
            displacement_seed +=
                grad_total_dipole(g, d) * positions(n, d);
        }
        result.grad_charge_density(n, 0) =
            grad_total_charge(g) + displacement_seed;
        if (with_dipoles) {
            result.grad_charge_density(n, 3) = grad_total_dipole(g, 0);
            result.grad_charge_density(n, 1) = grad_total_dipole(g, 1);
            result.grad_charge_density(n, 2) = grad_total_dipole(g, 2);
        }
    }
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::FukuiAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_fukui_normalize(
    const LongRangeConstMatrix<double>& fukui_sources,
    const LongRangeConstVector<double>& charge_l0_alpha,
    const LongRangeConstVector<double>& charge_l0_beta,
    const LongRangeConstVector<double>& total_charge,
    const LongRangeConstVector<double>& total_spin,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeConstVector<double>& grad_alpha,
    const LongRangeConstVector<double>& grad_beta) const
{
    if (fukui_sources.extent(1) != 2)
        throw std::invalid_argument(
            "adjoint_fukui_normalize: fukui_sources must carry [n_nodes, 2].");
    const std::size_t n_nodes = fukui_sources.extent(0);
    const std::size_t n_graphs = total_charge.extent(0);
    if (total_spin.extent(0) != n_graphs)
        throw std::invalid_argument(
            "adjoint_fukui_normalize: total_charge and total_spin must agree "
            "on the graph count.");
    if (node_graph_index.extent(0) != n_nodes
        || charge_l0_alpha.extent(0) != n_nodes
        || charge_l0_beta.extent(0) != n_nodes
        || grad_alpha.extent(0) != n_nodes
        || grad_beta.extent(0) != n_nodes)
        throw std::invalid_argument(
            "adjoint_fukui_normalize: sources, the l=0 charges, the graph "
            "index, and the channel seeds must agree on the node count.");
    for (std::size_t n = 0; n < n_nodes; ++n)
        if (node_graph_index(n) < 0
            || static_cast<std::size_t>(node_graph_index(n)) >= n_graphs)
            throw std::invalid_argument(
                "adjoint_fukui_normalize: node_graph_index must address the "
                "charge/spin rows.");

    // The forward's per-graph scatters (its zero guard included) and the
    // per-graph broadcast seeds: seed_c(g) = sum_n grad_c(n) * f_c(n) / norm_c.
    std::vector<double> source_sum(2 * n_graphs, 0.0);
    std::vector<double> charge_sum(2 * n_graphs, 0.0);
    std::vector<double> broadcast_seed(2 * n_graphs, 0.0);
    for (std::size_t n = 0; n < n_nodes; ++n) {
        const std::size_t g = static_cast<std::size_t>(node_graph_index(n));
        source_sum[2 * g] += fukui_sources(n, 0);
        source_sum[2 * g + 1] += fukui_sources(n, 1);
        charge_sum[2 * g] += charge_l0_alpha(n);
        charge_sum[2 * g + 1] += charge_l0_beta(n);
        broadcast_seed[2 * g] += grad_alpha(n) * fukui_sources(n, 0);
        broadcast_seed[2 * g + 1] += grad_beta(n) * fukui_sources(n, 1);
    }

    FukuiAdjointResult result;
    result.grad_fukui_sources = LongRangeMatrix<double>(
        "long_range_fukui_grad_sources", n_nodes, 2);
    result.grad_charge_l0_alpha = LongRangeVector<double>(
        "long_range_fukui_grad_charge_alpha", n_nodes);
    result.grad_charge_l0_beta = LongRangeVector<double>(
        "long_range_fukui_grad_charge_beta", n_nodes);
    result.grad_total_charge = LongRangeVector<double>(
        "long_range_fukui_grad_total_charge", n_graphs);
    result.grad_total_spin = LongRangeVector<double>(
        "long_range_fukui_grad_total_spin", n_graphs);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        result.grad_total_charge, 0.0);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        result.grad_total_spin, 0.0);

    for (std::size_t n = 0; n < n_nodes; ++n) {
        const std::size_t g = static_cast<std::size_t>(node_graph_index(n));
        const bool guarded_alpha = source_sum[2 * g] == 0.0;
        const bool guarded_beta = source_sum[2 * g + 1] == 0.0;
        const double norm_alpha = guarded_alpha ? 1.0 : source_sum[2 * g];
        const double norm_beta = guarded_beta ? 1.0 : source_sum[2 * g + 1];
        // The forward's broadcast targets, recomputed exactly as it writes
        // them (the guard touches only the norm, never the target).
        const double target_alpha = (total_charge(g)
                + (total_spin(g) - 1.0)) / 2.0 - charge_sum[2 * g];
        const double target_beta = (total_charge(g)
                - (total_spin(g) - 1.0)) / 2.0 - charge_sum[2 * g + 1];
        const double seed_alpha = broadcast_seed[2 * g] / norm_alpha;
        const double seed_beta = broadcast_seed[2 * g + 1] / norm_beta;
        result.grad_fukui_sources(n, 0) = grad_alpha(n) * target_alpha
                / norm_alpha
            - (guarded_alpha ? 0.0 : seed_alpha * target_alpha / norm_alpha);
        result.grad_fukui_sources(n, 1) = grad_beta(n) * target_beta
                / norm_beta
            - (guarded_beta ? 0.0 : seed_beta * target_beta / norm_beta);
        result.grad_charge_l0_alpha(n) = -seed_alpha;
        result.grad_charge_l0_beta(n) = -seed_beta;
    }
    // The graph-level seeds scatter once per graph, not once per node: the
    // broadcast gradient is the per-graph sum of the normalized seeds.
    for (std::size_t g = 0; g < n_graphs; ++g) {
        const double norm_alpha =
            source_sum[2 * g] == 0.0 ? 1.0 : source_sum[2 * g];
        const double norm_beta =
            source_sum[2 * g + 1] == 0.0 ? 1.0 : source_sum[2 * g + 1];
        const double seed_alpha = broadcast_seed[2 * g] / norm_alpha;
        const double seed_beta = broadcast_seed[2 * g + 1] / norm_beta;
        // Q_p_S/2 = (Q + S - 1)/2 and Q_m_S/2 = (Q - S + 1)/2.
        result.grad_total_charge(g) = 0.5 * (seed_alpha + seed_beta);
        result.grad_total_spin(g) = 0.5 * (seed_alpha - seed_beta);
    }
    return result;
}

// ===========================================================================
// Analytic-correction position adjoints
// ===========================================================================

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::CorrectionEnergyAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_correction_energy(
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeConstMatrix<int>& graph_pbc,
    const LongRangeConstVector<double>& grad_energy) const
{
    const std::size_t n_graphs = cell.extent(0);
    const std::size_t n_nodes = positions.extent(0);
    if (grad_energy.extent(0) != n_graphs)
        throw std::invalid_argument(
            "adjoint_correction_energy: the energy seeds must carry "
            "n_graphs.");
    if (charge_density.extent(0) != n_nodes || positions.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_correction_energy: positions must carry [n_nodes, 3] and "
            "the density must agree on the node count.");
    if (graph_pbc.extent(0) != n_graphs || graph_pbc.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_correction_energy: graph_pbc must carry [n_graphs, 3].");
    LongRangeVector<double> total_charge;
    LongRangeMatrix<double> total_dipole;
    LongRangeVector<double> quadrupole;
    graph_multipoles(charge_density, positions, node_graph_index, n_graphs,
        total_charge, total_dipole, quadrupole);

    const double constant = field_constant / (4.0 * pi);
    const bool with_dipoles = charge_density.extent(1) > 1;
    // The graph-level seeds of the forward's aggregates: the charge, the
    // three dipole components, and the quadrupole.  The slab term's z-dipole
    // is exactly the dipole's z component, so both modes scatter through one
    // aggregate set.
    std::vector<double> seed_charge(n_graphs, 0.0);
    std::vector<double> seed_dipole(3 * n_graphs, 0.0);
    std::vector<double> seed_quadrupole(n_graphs, 0.0);
    for (std::size_t g = 0; g < n_graphs; ++g) {
        const bool periodic_x = graph_pbc(g, 0) != 0;
        const bool periodic_y = graph_pbc(g, 1) != 0;
        const bool periodic_z = graph_pbc(g, 2) != 0;
        const bool is_molecule =
            !periodic_x && !periodic_y && !periodic_z;
        const bool is_slab = periodic_x && periodic_y && !periodic_z;
        if (!is_molecule && !is_slab)
            continue;
        const double volume = cell_determinant(&cell(g, 0, 0));
        if (is_molecule) {
            const double charge = total_charge(g);
            const double box_length = std::pow(volume, 0.3333);
            seed_charge[g] = grad_energy(g)
                * (cubic_madelung * constant * charge / box_length
                    - 2.0 * constant * pi * quadrupole(g)
                        / (3.0 * volume));
            for (int d = 0; d < 3; ++d)
                seed_dipole[3 * g + d] = grad_energy(g)
                    * 4.0 * constant * pi * total_dipole(g, d)
                    / (3.0 * volume);
            seed_quadrupole[g] = grad_energy(g)
                * (-2.0 * constant * pi * charge / (3.0 * volume));
        } else {
            // The slab dipole term const * 2 pi * mu_z^2 / V: the z-dipole is
            // the dipole's z component.
            seed_dipole[3 * g + 2] = grad_energy(g)
                * 4.0 * constant * pi * total_dipole(g, 2) / volume;
        }
    }

    CorrectionEnergyAdjointResult result;
    result.grad_charge_density = LongRangeMatrix<double>(
        "long_range_correction_grad_density", n_nodes,
        charge_density.extent(1));
    result.grad_positions = LongRangeMatrix<double>(
        "long_range_correction_grad_positions", n_nodes, 3);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        result.grad_charge_density, 0.0);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        result.grad_positions, 0.0);
    // The aggregate scatter: dQ/dq(n) = 1, dD_j/dq(n) = r_j(n), dD_j/dp_j(n)
    // = 1, dQuad/dq(n) = |r(n)|^2, dQuad/dp_j(n) = 2 r_j(n), and
    // dD_j/dr_j(n) = q(n), dQuad/dr_j(n) = 2 q(n) r_j(n) + 2 p_j(n).
    for (std::size_t n = 0; n < n_nodes; ++n) {
        const std::size_t g = static_cast<std::size_t>(node_graph_index(n));
        if (node_graph_index(n) < 0 || g >= n_graphs)
            throw std::invalid_argument(
                "adjoint_correction_energy: node_graph_index must address "
                "the graph rows.");
        const double q = charge_density(n, 0);
        const double p[3] = {with_dipoles ? charge_density(n, 3) : 0.0,
            with_dipoles ? charge_density(n, 1) : 0.0,
            with_dipoles ? charge_density(n, 2) : 0.0};
        double r_squared = 0.0;
        for (int d = 0; d < 3; ++d)
            r_squared += positions(n, d) * positions(n, d);
        double displacement_seed = 0.0;
        for (int d = 0; d < 3; ++d)
            displacement_seed += seed_dipole[3 * g + d] * positions(n, d);
        result.grad_charge_density(n, 0) = seed_charge[g] + displacement_seed
            + seed_quadrupole[g] * r_squared;
        for (int d = 0; d < 3; ++d) {
            result.grad_positions(n, d) = seed_dipole[3 * g + d] * q
                + seed_quadrupole[g]
                    * (2.0 * q * positions(n, d) + 2.0 * p[d]);
            if (with_dipoles)
                result.grad_charge_density(n, dipole_column(d)) =
                    seed_dipole[3 * g + d]
                    + seed_quadrupole[g] * 2.0 * positions(n, d);
        }
    }
    return result;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::CorrectionFeatureAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_correction_features(
    const LongRangeConstMatrix<double>& charge_density_alpha,
    const LongRangeConstMatrix<double>& charge_density_beta,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeConstMatrix<int>& graph_pbc,
    const LongRangeConstMatrix<double>& grad_fields_alpha,
    const LongRangeConstMatrix<double>& grad_fields_beta) const
{
    const std::size_t n_graphs = cell.extent(0);
    const std::size_t n_nodes = positions.extent(0);
    if (charge_density_alpha.extent(0) != n_nodes
        || charge_density_beta.extent(0) != n_nodes
        || positions.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_correction_features: positions must carry "
            "[n_nodes, 3] and both channel densities must agree on the node "
            "count.");
    if (grad_fields_alpha.extent(0) != n_nodes
        || grad_fields_alpha.extent(1) != feature_dim_
        || grad_fields_beta.extent(0) != n_nodes
        || grad_fields_beta.extent(1) != feature_dim_)
        throw std::invalid_argument(
            "adjoint_correction_features: the corrective-field seeds must "
            "carry [n_nodes, feature_dim] per channel.");
    if (graph_pbc.extent(0) != n_graphs || graph_pbc.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_correction_features: graph_pbc must carry "
            "[n_graphs, 3].");

    const double constant = field_constant / (4.0 * pi);
    CorrectionFeatureAdjointResult result;
    result.grad_charge_density_alpha = LongRangeMatrix<double>(
        "long_range_correction_grad_density_alpha", n_nodes,
        charge_density_alpha.extent(1));
    result.grad_charge_density_beta = LongRangeMatrix<double>(
        "long_range_correction_grad_density_beta", n_nodes,
        charge_density_beta.extent(1));
    result.grad_positions = LongRangeMatrix<double>(
        "long_range_correction_grad_positions", n_nodes, 3);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        result.grad_charge_density_alpha, 0.0);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        result.grad_charge_density_beta, 0.0);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        result.grad_positions, 0.0);

    const LongRangeConstMatrix<double>* densities[2] = {
        &charge_density_alpha, &charge_density_beta};
    const LongRangeConstMatrix<double>* seeds[2] = {
        &grad_fields_alpha, &grad_fields_beta};
    LongRangeMatrix<double>* density_gradients[2] = {
        &result.grad_charge_density_alpha,
        &result.grad_charge_density_beta};
    for (int channel = 0; channel < 2; ++channel) {
        const bool with_dipoles = densities[channel]->extent(1) > 1;
        LongRangeMatrix<double> node_field_seeds(
            "long_range_correction_node_field_seeds", n_nodes, 4);
        correction_node_field_seeds(*seeds[channel], node_field_seeds);
        LongRangeVector<double> total_charge;
        LongRangeMatrix<double> total_dipole;
        LongRangeVector<double> quadrupole;
        graph_multipoles(*densities[channel], positions, node_graph_index,
            n_graphs, total_charge, total_dipole, quadrupole);
        LongRangeMatrix<double>& density_gradient = *density_gradients[channel];

        for (std::size_t g = 0; g < n_graphs; ++g) {
            const bool periodic_x = graph_pbc(g, 0) != 0;
            const bool periodic_y = graph_pbc(g, 1) != 0;
            const bool periodic_z = graph_pbc(g, 2) != 0;
            const bool is_molecule =
                !periodic_x && !periodic_y && !periodic_z;
            const bool is_slab = periodic_x && periodic_y && !periodic_z;
            if (!is_molecule && !is_slab)
                continue;
            const double volume = cell_determinant(&cell(g, 0, 0));
            if (is_molecule) {
                // The node fields' aggregate weights (the box term): the
                // graph-level seeds of the charge, the dipole, and the
                // quadrupole, plus each node's own direct terms.
                const double charge = total_charge(g);
                const double box_length = std::pow(volume, 0.333333);
                const double inverse_3v = 1.0 / (3.0 * volume);
                double seed_charge = 0.0;
                double seed_dipole[3] = {0.0, 0.0, 0.0};
                double seed_quadrupole = 0.0;
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    const double s0 = node_field_seeds(n, 0);
                    double r_squared = 0.0;
                    for (int d = 0; d < 3; ++d)
                        r_squared += positions(n, d) * positions(n, d);
                    // The explicit-Q derivative only: the quadrupole's own
                    // derivative is the quadrupole seed below.
                    seed_charge += s0
                        * (cubic_madelung * constant / box_length
                            - 2.0 * constant * pi * r_squared * inverse_3v);
                    seed_quadrupole += s0
                        * (-2.0 * constant * pi * inverse_3v);
                    for (int d = 0; d < 3; ++d) {
                        seed_charge += node_field_seeds(n, 1 + d)
                            * (-4.0 * constant * pi * positions(n, d)
                                * inverse_3v);
                        seed_dipole[d] += s0
                                * 4.0 * constant * pi * positions(n, d)
                                * inverse_3v
                            + node_field_seeds(n, 1 + d)
                                * 4.0 * constant * pi * inverse_3v;
                    }
                    // The direct per-node terms of n0 and n_{1+d}.
                    for (int d = 0; d < 3; ++d) {
                        result.grad_positions(n, d) += s0
                                * (-4.0 * constant * pi * charge
                                        * positions(n, d) * inverse_3v
                                    + 4.0 * constant * pi * total_dipole(g, d)
                                        * inverse_3v)
                            + node_field_seeds(n, 1 + d)
                                * (-4.0 * constant * pi * charge * inverse_3v);
                    }
                }
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    const double q = (*densities[channel])(n, 0);
                    const double p[3] = {
                        with_dipoles ? (*densities[channel])(n, 3) : 0.0,
                        with_dipoles ? (*densities[channel])(n, 1) : 0.0,
                        with_dipoles ? (*densities[channel])(n, 2) : 0.0};
                    double r_squared = 0.0;
                    double displacement_seed = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        r_squared += positions(n, d) * positions(n, d);
                        displacement_seed += seed_dipole[d] * positions(n, d);
                    }
                    density_gradient(n, 0) += seed_charge + displacement_seed
                        + seed_quadrupole * r_squared;
                    for (int d = 0; d < 3; ++d) {
                        result.grad_positions(n, d) +=
                            seed_dipole[d] * q
                            + seed_quadrupole
                                * (2.0 * q * positions(n, d) + 2.0 * p[d]);
                        if (with_dipoles)
                            density_gradient(n, dipole_column(d)) +=
                                seed_dipole[d]
                                + seed_quadrupole * 2.0 * positions(n, d);
                    }
                }
            } else {
                // The slab z-dipole fields: field_z = 4 pi const mu_z / V,
                // n0 = field_z * r_z, n3 = field_z, and the z-dipole is the
                // dipole's z component.
                const double field_scale = 4.0 * constant * pi / volume;
                const double dipole_z = total_dipole(g, 2);
                const double field_z = field_scale * dipole_z;
                double seed_field = 0.0;
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    seed_field += node_field_seeds(n, 0) * positions(n, 2)
                        + node_field_seeds(n, 3);
                    result.grad_positions(n, 2) +=
                        node_field_seeds(n, 0) * field_z;
                }
                const double seed_dipole_z = seed_field * field_scale;
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    const double q = (*densities[channel])(n, 0);
                    density_gradient(n, 0) += seed_dipole_z * positions(n, 2);
                    if (with_dipoles)
                        density_gradient(n, dipole_column(2)) += seed_dipole_z;
                    result.grad_positions(n, 2) +=
                        seed_dipole_z * q;
                }
            }
        }
    }
    return result;
}

// ===========================================================================
// Cell (strain) adjoints: the reciprocal-space virial
// ===========================================================================

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::BasisKDerivative PolarLongRangeOpsT<Scalar>::basis_k_derivative(
    const int max_l,
    const double sigma,
    const bool normalize_receiver,
    const double* k,
    const double k_norm2,
    const int j) const
{
    BasisKDerivative out;
    const double pref0 = (4.0 * pi) * std::sqrt(pi / 2.0) * sigma * sigma * sigma;
    const double exponential = std::exp(-0.5 * k_norm2 * sigma * sigma);
    const double sigma_squared = sigma * sigma;
    // The l=0 real factor: pref0 * exp * C_0 * Y_0.
    const double a0 = pref0 * cl_sigma(0, sigma, normalize_receiver) * harmonic_l0();
    out.df0_real = -sigma_squared * k[j] * a0 * exponential;
    if (max_l >= 1) {
        // The l=1 imaginary factors: -B * exp * k_q exactly (the radial
        // |k| of f_1 cancels the direction's division), q the e3nn
        // (y, z, x) permutation of the m index.
        const double b = pref0 * sigma_squared * cl_sigma(1, sigma, normalize_receiver)
            * harmonic_l1();
        static const int permutation[3] = {1, 2, 0};
        for (int p = 0; p < 3; ++p) {
            const int q = permutation[p];
            out.dimag[p] = -b * exponential
                * ((q == j ? 1.0 : 0.0) - sigma_squared * k[q] * k[j]);
        }
    }
    return out;
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::AssembleCellAdjointResult
PolarLongRangeOpsT<Scalar>::assemble_cell_seeds(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_density,
    const LongRangeConstMatrix<double>& multipoles,
    const bool include_table_seeds,
    const LongRangeKernelMatrix<Scalar>* grad_k_accumulator,
    const LongRangeKernelVector<Scalar>* grad_volume_accumulator) const
{
    if ((grad_k_accumulator == nullptr)
        != (grad_volume_accumulator == nullptr))
        throw std::invalid_argument(
            "assemble cell adjoint requires both device metadata "
            "accumulators.");
    const std::size_t n_k = cache.n_k;
    const double scale = two_pi_cubed();

    // The forward's own intermediates, recomputed exactly as the forward
    // computes them: rho(k) for the volume seed and the two coefficient
    // GEMM outputs for the factor derivative.
    const auto multipoles_kernel = stage_matrix<Scalar>(
        multipoles, "long_range_cell_multipoles");
    const auto grad_density_kernel = stage_matrix<Scalar>(
        grad_density, "long_range_cell_grad_density");
    const auto density =
        assemble_fourier_kernel(cache, multipoles_kernel);
    LongRangeKernelMatrix<Scalar> coeff_cos(
        "long_range_coeff_cos", n_k, density_width_);
    LongRangeKernelMatrix<Scalar> coeff_sin(
        "long_range_coeff_sin", n_k, density_width_);
    KokkosBlas::gemm(
        "N", "N", 1.0, cache.cos_table, multipoles_kernel, 0.0, coeff_cos);
    KokkosBlas::gemm(
        "N", "N", 1.0, cache.sin_table, multipoles_kernel, 0.0, coeff_sin);
    AssembleCellAdjointResult result;
    // The position-adjoint products through the one shared spelling.  The
    // k-vector and volume seeds below never read them, so the composed
    // chain opts out of the [n_k, n] table downloads entirely.
    if (include_table_seeds) {
        AssembleAdjointResult base =
            assemble_adjoint(cache, grad_density, multipoles);
        result.grad_cos_table = std::move(base.grad_cos_table);
        result.grad_sin_table = std::move(base.grad_sin_table);
        result.grad_multipoles = std::move(base.grad_multipoles);
    }
    LongRangeKernelMatrix<Scalar> grad_k_kernel(
        "long_range_grad_k", n_k, 3);
    LongRangeKernelVector<Scalar> grad_volume_kernel(
        "long_range_grad_volume", n_k);
    const double sigma = density_smearing_width_;
    const double sigma_squared = sigma * sigma;
    const double pref0 =
        (4.0 * pi) * std::sqrt(pi / 2.0) * sigma * sigma * sigma;
    const double a0 =
        pref0 * cl_sigma(0, sigma, false) * harmonic_l0();
    const double b = pref0 * sigma_squared * cl_sigma(1, sigma, false)
        * harmonic_l1();
    const bool with_l1 = density_max_l_ >= 1;
    auto volume_per_k = cache.volume_per_k;
    auto k_norm2 = cache.k_norm2;
    auto k_vectors = cache.k_vectors;
    Kokkos::parallel_for(
        "long_range assemble cell k seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double g0 = grad_density_kernel(i, 0);
            const double g1 = grad_density_kernel(i, 1);
            grad_volume_kernel(i) =
                -(g0 * density(i, 0) + g1 * density(i, 1))
                / volume_per_k(i);
            const double s = scale / volume_per_k(i);
            const double exponential =
                Kokkos::exp(-0.5 * k_norm2(i) * sigma_squared);
            for (int j = 0; j < 3; ++j) {
                const double df0 = -sigma_squared * k_vectors(i, j)
                    * a0 * exponential;
                double dre = df0 * coeff_cos(i, 0);
                double dim = -df0 * coeff_sin(i, 0);
                if (with_l1) {
                    for (int p = 0; p < 3; ++p) {
                        const int q = p == 0 ? 1 : (p == 1 ? 2 : 0);
                        const double dimag = -b * exponential
                            * ((q == j ? 1.0 : 0.0)
                                - sigma_squared * k_vectors(i, q)
                                    * k_vectors(i, j));
                        dre += dimag * coeff_sin(i, 1 + p);
                        dim += dimag * coeff_cos(i, 1 + p);
                    }
                }
                grad_k_kernel(i, j) = s * (g0 * dre + g1 * dim);
            }
        });
    if (grad_k_accumulator != nullptr) {
        accumulate_kernel_matrix(*grad_k_accumulator, grad_k_kernel,
            "long_range accumulate assemble k seed");
        accumulate_kernel_vector(*grad_volume_accumulator, grad_volume_kernel,
            "long_range accumulate assemble volume seed");
    } else {
        result.grad_k_vectors =
            download_matrix<double>(grad_k_kernel, "long_range_grad_k_host");
        result.grad_volume_per_k = download_vector<double>(
            grad_volume_kernel, "long_range_grad_volume_host");
    }
    return result;
}


template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::AssembleCellAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_assemble_cell(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_density,
    const LongRangeConstMatrix<double>& multipoles,
    const bool include_table_seeds,
    const LongRangeKernelMatrix<Scalar>* grad_k_accumulator,
    const LongRangeKernelVector<Scalar>* grad_volume_accumulator) const
{
    if (multipoles.extent(1) != density_width_)
        throw std::invalid_argument(
            "adjoint_assemble_cell: multipoles must carry the density width.");
    if (grad_density.extent(0) != cache.n_k || grad_density.extent(1) != 2)
        throw std::invalid_argument(
            "adjoint_assemble_cell: the density seeds must carry [n_k, 2].");
    return assemble_cell_seeds(cache, grad_density, multipoles,
        include_table_seeds, grad_k_accumulator, grad_volume_accumulator);
}

template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::CoulombCellAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_coulomb_cell(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_potential,
    const LongRangeConstMatrix<double>& density,
    const LongRangeKernelMatrix<Scalar>* grad_k_accumulator) const
{
    if (grad_potential.extent(0) != cache.n_k
        || grad_potential.extent(1) != 2)
        throw std::invalid_argument(
            "adjoint_coulomb_cell: the potential seeds must carry [n_k, 2].");
    if (density.extent(0) != cache.n_k || density.extent(1) != 2)
        throw std::invalid_argument(
            "adjoint_coulomb_cell: the density must carry [n_k, 2].");
    CoulombCellAdjointResult result;
    const auto grad_potential_kernel = stage_matrix<Scalar>(
        grad_potential, "long_range_cell_grad_potential");
    const auto density_kernel = stage_matrix<Scalar>(
        density, "long_range_cell_density");
    // The position-chain seed is the Coulomb adjoint's own.
    const auto grad_density_kernel =
        apply_coulomb_kernel(cache, grad_potential_kernel);
    result.grad_density = download_matrix<double>(
        grad_density_kernel, "long_range_grad_density_host");
    // The diagonal kernel's cell seed: d(FC/k^2)/dk_j = -2 FC k_j/k^4,
    // exactly zero at the origin mode (k_j = 0 there), times the density
    // value each potential component multiplies.
    LongRangeKernelMatrix<Scalar> grad_k_kernel(
        "long_range_coulomb_grad_k", cache.n_k, 3);
    auto k_norm2 = cache.k_norm2;
    auto k_vectors = cache.k_vectors;
    Kokkos::parallel_for(
        "long_range coulomb cell k seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, cache.n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const double norm2 = k_norm2(i);
            double factor = 0.0;
            if (norm2 != 0.0)
                factor = -2.0 * field_constant
                    * (grad_potential_kernel(i, 0) * density_kernel(i, 0)
                        + grad_potential_kernel(i, 1) * density_kernel(i, 1))
                    / (norm2 * norm2);
            for (int j = 0; j < 3; ++j)
                grad_k_kernel(i, j) = factor * k_vectors(i, j);
        });
    if (grad_k_accumulator != nullptr)
        accumulate_kernel_matrix(*grad_k_accumulator, grad_k_kernel,
            "long_range accumulate coulomb k seed");
    else
        result.grad_k_vectors = download_matrix<double>(
            grad_k_kernel, "long_range_coulomb_grad_k_host");
    return result;
}


template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::ProjectCellAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_project_cell(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_features,
    const LongRangeConstMatrix<double>& potential,
    const bool include_table_seeds,
    const LongRangeKernelMatrix<Scalar>* grad_k_accumulator) const
{
    const std::size_t n_k = cache.n_k;
    const std::size_t feature_dim = feature_dim_;
    if (grad_features.extent(0) != cache.n_nodes
        || grad_features.extent(1) != feature_dim)
        throw std::invalid_argument(
            "adjoint_project_cell: the feature seeds must carry "
            "[n_nodes, feature_dim].");
    if (potential.extent(0) != n_k || potential.extent(1) != 2)
        throw std::invalid_argument(
            "adjoint_project_cell: the potential must carry [n_k, 2].");

    // The permutation-and-scale adjoint of the position surface: the
    // projection seed is the scatter of the scaled feature seed.
    const double scale = 2.0 / two_pi_cubed();
    const auto grad_features_kernel = stage_matrix<Scalar>(
        grad_features, "long_range_cell_grad_features_in");
    LongRangeKernelMatrix<Scalar> grad_projections(
        "long_range_grad_projections", feature_dim, cache.n_nodes);
    auto permutation_v = output_permutation_kernel_;
    Kokkos::parallel_for(
        "long_range project cell adjoint scatter",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, cache.n_nodes),
        KOKKOS_LAMBDA(const std::size_t n) {
            for (std::size_t j = 0; j < feature_dim; ++j)
                grad_projections(permutation_v(j), n) =
                    scale * grad_features_kernel(n, j);
        });

    // The forward amplitudes and the four transposed-GEMM products, the
    // identical shapes and spells of adjoint_project.
    const auto potential_kernel = stage_matrix<Scalar>(
        potential, "long_range_cell_potential_in");
    LongRangeKernelMatrix<Scalar> amplitude_a(
        "long_range_proj_a", n_k, feature_dim);
    LongRangeKernelMatrix<Scalar> amplitude_b(
        "long_range_proj_b", n_k, feature_dim);
    projection_amplitudes(cache, potential_kernel, amplitude_a, amplitude_b);
    LongRangeKernelMatrix<Scalar> grad_amplitude_a(
        "long_range_grad_proj_a", n_k, feature_dim);
    LongRangeKernelMatrix<Scalar> grad_amplitude_b(
        "long_range_grad_proj_b", n_k, feature_dim);
    KokkosBlas::gemm("N", "T", 1.0, cache.cos_table, grad_projections, 0.0,
        grad_amplitude_a);
    KokkosBlas::gemm("N", "T", 1.0, cache.sin_table, grad_projections, 0.0,
        grad_amplitude_b);
    LongRangeKernelMatrix<Scalar> grad_k_vectors(
        "long_range_grad_k", n_k, 3);

    ProjectCellAdjointResult result;
    if (include_table_seeds) {
        LongRangeKernelMatrix<Scalar> grad_cos_table(
            "long_range_grad_cos_table", n_k, cache.n_nodes);
        LongRangeKernelMatrix<Scalar> grad_sin_table(
            "long_range_grad_sin_table", n_k, cache.n_nodes);
        KokkosBlas::gemm(
            "N", "N", 1.0, amplitude_a, grad_projections, 0.0, grad_cos_table);
        KokkosBlas::gemm(
            "N", "N", 1.0, amplitude_b, grad_projections, 0.0, grad_sin_table);
        result.grad_cos_table = download_matrix<double>(
            grad_cos_table, "long_range_grad_cos_host");
        result.grad_sin_table = download_matrix<double>(
            grad_sin_table, "long_range_grad_sin_host");
    }
    // The amplitude-combine adjoint of the position surface, plus the
    // cell seed: the feature basis factors' k-derivative contracted with
    // the amplitude seeds.  dA_p/dk_j = fp (pot0 dfR + pot1 dfI) and
    // dB_p/dk_j = fp (pot0 dfI - pot1 dfR), so
    // dL/dk_j = sum_p (dA_p Gaa_p + dB_p Gab_p).
    const std::size_t feature_width = feature_width_;
    const std::size_t n_sigma = feature_smearing_widths_.size();
    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>
        derivative_parameters_host(
            "long_range_feature_derivative_parameters_host", n_sigma, 3);
    for (std::size_t sigma_index = 0; sigma_index < n_sigma; ++sigma_index) {
        const double sigma = feature_smearing_widths_[sigma_index];
        const double sigma_squared = sigma * sigma;
        const double pref0 =
            (4.0 * pi) * std::sqrt(pi / 2.0) * sigma * sigma * sigma;
        derivative_parameters_host(sigma_index, 0) = sigma_squared;
        derivative_parameters_host(sigma_index, 1) =
            pref0 * cl_sigma(0, sigma, true) * harmonic_l0();
        derivative_parameters_host(sigma_index, 2) = pref0 * sigma_squared
            * cl_sigma(1, sigma, true) * harmonic_l1();
    }
    Kokkos::View<const double**, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        derivative_parameters_unmanaged(
            derivative_parameters_host.data(), n_sigma, 3);
    const auto derivative_parameters = stage_matrix<Scalar>(
        derivative_parameters_unmanaged,
        "long_range_feature_derivative_parameters");
    const bool with_l1 = feature_max_l_ >= 1;
    auto k_norm2 = cache.k_norm2;
    auto k_vectors = cache.k_vectors;
    auto factor_proj = cache.k_factor_proj;
    Kokkos::parallel_for(
        "long_range project cell k seeds",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
            for (int j = 0; j < 3; ++j) {
                double gj = 0.0;
                for (std::size_t m = 0; m < feature_dim; ++m) {
                    const double gaa = grad_amplitude_a(i, m);
                    const double gab = grad_amplitude_b(i, m);
                    if (gaa == 0.0 && gab == 0.0)
                        continue;
                    const std::size_t sigma_index = m / feature_width;
                    const std::size_t channel = m % feature_width;
                    const double sigma_squared =
                        derivative_parameters(sigma_index, 0);
                    const double exponential = Kokkos::exp(
                        -0.5 * k_norm2(i) * sigma_squared);
                    double df_real = 0.0;
                    double df_imag = 0.0;
                    if (channel == 0) {
                        df_real = -sigma_squared * k_vectors(i, j)
                            * derivative_parameters(sigma_index, 1)
                            * exponential;
                    } else if (with_l1) {
                        const int p = static_cast<int>(channel - 1);
                        const int q = p == 0 ? 1 : (p == 1 ? 2 : 0);
                        df_imag = -derivative_parameters(sigma_index, 2)
                            * exponential
                            * ((q == j ? 1.0 : 0.0)
                                - sigma_squared * k_vectors(i, q)
                                    * k_vectors(i, j));
                    }
                    const double fp = factor_proj(i);
                    const double pot0 = potential_kernel(i, 0);
                    const double pot1 = potential_kernel(i, 1);
                    const double da =
                        fp * (pot0 * df_real + pot1 * df_imag);
                    const double db =
                        fp * (pot0 * df_imag - pot1 * df_real);
                    gj += da * gaa + db * gab;
                }
                grad_k_vectors(i, j) = gj;
            }
        });
    if (grad_k_accumulator != nullptr)
        accumulate_kernel_matrix(*grad_k_accumulator, grad_k_vectors,
            "long_range accumulate project k seed");
    else
        result.grad_k_vectors = download_matrix<double>(
            grad_k_vectors, "long_range_grad_k_host");
    // The potential seed: the amplitude-combine adjoint (identical to
    // the position surface's).
    {
        LongRangeKernelMatrix<Scalar> grad_potential(
            "long_range_grad_potential", n_k, 2);
        auto factors = cache.feature_fourier_factors;
        auto factor_proj = cache.k_factor_proj;
        Kokkos::parallel_for(
            "long_range project cell adjoint combine",
            Kokkos::RangePolicy<LongRangeKernelSpace,
                Kokkos::IndexType<std::size_t>>(0, n_k),
            KOKKOS_LAMBDA(const std::size_t i) {
                const double s = factor_proj(i);
                double g0 = 0.0;
                double g1 = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : g0, g1)
#endif
                for (std::size_t m = 0; m < feature_dim; ++m) {
                    const std::size_t sigma = m / feature_width;
                    const std::size_t channel = m % feature_width;
                    const double factor_real =
                        factors(i, sigma, channel, 0);
                    const double factor_imaginary =
                        factors(i, sigma, channel, 1);
                    g0 += grad_amplitude_a(i, m) * factor_real
                        + grad_amplitude_b(i, m) * factor_imaginary;
                    g1 += grad_amplitude_a(i, m) * factor_imaginary
                        - grad_amplitude_b(i, m) * factor_real;
                }
                grad_potential(i, 0) = s * g0;
                grad_potential(i, 1) = s * g1;
            });
        result.grad_potential = download_matrix<double>(
            grad_potential, "long_range_grad_potential_host");
    }
    return result;
}


template <typename Scalar>
LongRangeTensor3<double> PolarLongRangeOpsT<Scalar>::adjoint_geometry_cell(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_k_vectors,
    const LongRangeConstVector<double>& grad_volume_per_k,
    const LongRangeConstMatrix<double>& grad_cos_table,
    const LongRangeConstMatrix<double>& grad_sin_table,
    const LongRangeConstMatrix<double>& positions) const
{
    const auto grad_cos_kernel = stage_matrix<Scalar>(
        grad_cos_table, "long_range_cell_grad_cos_in");
    const auto grad_sin_kernel = stage_matrix<Scalar>(
        grad_sin_table, "long_range_cell_grad_sin_in");
    return adjoint_geometry_cell_device(cache, grad_k_vectors,
        grad_volume_per_k, grad_cos_kernel, grad_sin_kernel, positions);
}

template <typename Scalar>
LongRangeTensor3<double> PolarLongRangeOpsT<Scalar>::adjoint_geometry_cell_device(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_k_vectors,
    const LongRangeConstVector<double>& grad_volume_per_k,
    const LongRangeKernelMatrix<Scalar>& grad_cos_table,
    const LongRangeKernelMatrix<Scalar>& grad_sin_table,
    const LongRangeConstMatrix<double>& positions) const
{
    const auto grad_k_kernel = stage_matrix<Scalar>(
        grad_k_vectors, "long_range_cell_grad_k_in");
    const auto grad_volume_kernel = stage_vector<Scalar>(
        grad_volume_per_k, "long_range_cell_grad_volume_in");
    const auto device = adjoint_geometry_cell_all_device(cache, grad_k_kernel,
        grad_volume_kernel, grad_cos_table, grad_sin_table, positions);
    return download_tensor3<double>(device, "long_range_grad_strain");
}

template <typename Scalar>
LongRangeKernelTensor3<Scalar>
PolarLongRangeOpsT<Scalar>::adjoint_geometry_cell_all_device(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeKernelMatrix<Scalar>& grad_k_vectors,
    const LongRangeKernelVector<Scalar>& grad_volume_per_k,
    const LongRangeKernelMatrix<Scalar>& grad_cos_table,
    const LongRangeKernelMatrix<Scalar>& grad_sin_table,
    const LongRangeConstMatrix<double>& positions,
    const bool use_workspace) const
{
    const std::size_t n_k = cache.n_k;
    const std::size_t n_nodes = cache.n_nodes;
    if (grad_k_vectors.extent(0) != n_k || grad_k_vectors.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_geometry_cell: the k seeds must carry [n_k, 3].");
    if (grad_volume_per_k.extent(0) != n_k)
        throw std::invalid_argument(
            "adjoint_geometry_cell: the volume seeds must carry [n_k].");
    if (grad_cos_table.extent(0) != n_k
        || grad_cos_table.extent(1) != n_nodes
        || grad_sin_table.extent(0) != n_k
        || grad_sin_table.extent(1) != n_nodes)
        throw std::invalid_argument(
            "adjoint_geometry_cell: the table seeds must carry the cache's "
            "[n_k, n_nodes] shape.");
    if (positions.extent(0) != n_nodes || positions.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_geometry_cell: positions must carry [n_nodes, 3].");

    // The phase coefficient is the adjoint_geometry one (dL/dphi =
    // Gs cos - Gc sin), so the phase GEMM output S(n, b) is exactly the
    // tables-only position gradient and d(k.r)/deps_ab = -r_a k_b gives
    // the minus-outer-product contraction below.
    auto device_matrix = [this, use_workspace] (const char* label,
        const std::size_t rows, const std::size_t columns) {
        return use_workspace
            ? workspace_matrix(label, rows, columns)
            : LongRangeKernelMatrix<Scalar>(
                  Kokkos::view_alloc(std::string(label)), rows, columns);
    };
    auto coefficient = device_matrix(
        "long_range_geometry_cell_coeff", n_k, n_nodes);
    auto cos_table = cache.cos_table;
    auto sin_table = cache.sin_table;
    Kokkos::parallel_for(
        "long_range geometry cell coefficient",
        Kokkos::RangePolicy<LongRangeKernelSpace,
            Kokkos::IndexType<std::size_t>>(0, n_k),
        KOKKOS_LAMBDA(const std::size_t i) {
#if defined(_OPENMP)
#pragma omp simd
#endif
            for (std::size_t j = 0; j < n_nodes; ++j)
                coefficient(i, j) = grad_sin_table(i, j) * cos_table(i, j)
                    - grad_cos_table(i, j) * sin_table(i, j);
        });
    auto phase_positions_kernel = device_matrix(
        "long_range_geometry_cell_phase", n_nodes, 3);
    KokkosBlas::gemm(
        "T", "N", 1.0, coefficient, cache.k_vectors, 0.0,
        phase_positions_kernel);
    auto positions_kernel = use_workspace
        ? cache.positions_workspace
        : device_matrix("long_range_geometry_cell_positions_in", n_nodes, 3);
    if (!use_workspace)
        stage_matrix_into<Scalar>(positions, positions_kernel);
    auto grad_strain_kernel = use_workspace
        ? workspace_tensor3(
            "long_range_grad_strain_kernel", cache.n_graphs, 3, 3)
        : LongRangeKernelTensor3<Scalar>(
            "long_range_grad_strain_kernel", cache.n_graphs, 3, 3);
    auto k_vectors = cache.k_vectors;
    auto k_graph_index = cache.k_graph_index;
    auto node_graph_index = cache.node_graph_index;
    auto volumes = cache.kernel_volumes;
    const std::size_t n_graphs = cache.n_graphs;
    constexpr std::size_t strain_component_count = 9;
    if (n_graphs
        > static_cast<std::size_t>(std::numeric_limits<int>::max())
            / strain_component_count)
        throw std::length_error(
            "long-range geometry strain team count exceeds INT_MAX.");
    const int strain_team_count =
        static_cast<int>(n_graphs * strain_component_count);
    // Give every graph/tensor element its own team.  The common one-graph
    // case then launches nine cooperative teams instead of one block with
    // nine workers that each serially scan all reciprocal and node rows.
    // Only the compact [graph, 3, 3] result crosses back to the host.
    using CellReductionPolicy = Kokkos::TeamPolicy<LongRangeKernelSpace>;
    using CellReductionMember = typename CellReductionPolicy::member_type;
    Kokkos::parallel_for(
        "long_range geometry cell reduction",
        CellReductionPolicy(strain_team_count, Kokkos::AUTO),
        KOKKOS_LAMBDA(const CellReductionMember& team) {
            const std::size_t index = team.league_rank();
            const std::size_t graph = index / 9;
            const int a = static_cast<int>((index % 9) / 3);
            const int b = static_cast<int>(index % 3);
            double k_value = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, n_k),
                [=](const std::size_t i, double& sum) {
                    if (static_cast<std::size_t>(k_graph_index(i)) != graph)
                        return;
                    sum -= grad_k_vectors(i, a) * k_vectors(i, b);
                },
                k_value);
            double node_value = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, n_nodes),
                [=](const std::size_t n, double& sum) {
                    if (static_cast<std::size_t>(node_graph_index(n)) == graph)
                        sum -= positions_kernel(n, a)
                            * phase_positions_kernel(n, b);
                },
                node_value);
            double volume_seed = 0.0;
            if (a == b)
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, n_k),
                    [=](const std::size_t i, double& sum) {
                        if (static_cast<std::size_t>(k_graph_index(i)) == graph)
                            sum += grad_volume_per_k(i);
                    },
                    volume_seed);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                grad_strain_kernel(graph, a, b) = k_value + node_value
                    + double(volumes(graph)) * volume_seed;
            });
        });
    return grad_strain_kernel;
}

template <typename Scalar>
LongRangeTensor3<double>
PolarLongRangeOpsT<Scalar>::adjoint_geometry_cell_metadata(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstMatrix<double>& grad_k_vectors,
    const LongRangeConstVector<double>& grad_volume_per_k) const
{
    const std::size_t n_k = cache.n_k;
    if (grad_k_vectors.extent(0) != n_k || grad_k_vectors.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_geometry_cell_metadata: k seeds must carry [n_k, 3].");
    if (grad_volume_per_k.extent(0) != n_k)
        throw std::invalid_argument(
            "adjoint_geometry_cell_metadata: volume seeds must carry [n_k].");
    LongRangeTensor3<double> grad_strain(
        "long_range_metadata_grad_strain", cache.n_graphs, 3, 3);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, grad_strain, 0.0);
    for (std::size_t i = 0; i < n_k; ++i) {
        const std::size_t graph =
            static_cast<std::size_t>(cache.k_graph_index_host(i));
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                grad_strain(graph, a, b) -=
                    grad_k_vectors(i, a) * cache.k_vectors_host(3 * i + b);
    }
    for (std::size_t graph = 0; graph < cache.n_graphs; ++graph) {
        double volume_seed = 0.0;
        for (std::size_t i = 0; i < n_k; ++i)
            if (static_cast<std::size_t>(cache.k_graph_index_host(i)) == graph)
                volume_seed += grad_volume_per_k(i);
        for (int a = 0; a < 3; ++a)
            grad_strain(graph, a, a) +=
                cache.volumes_host(graph) * volume_seed;
    }
    return grad_strain;
}


template <typename Scalar>
typename PolarLongRangeOpsT<Scalar>::EnergyCellAdjointResult
PolarLongRangeOpsT<Scalar>::adjoint_coulomb_energy_cell(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstVector<double>& grad_energy,
    const LongRangeConstMatrix<double>& charge_density,
    const bool include_table_seeds) const
{
    if (grad_energy.extent(0) != cache.n_graphs)
        throw std::invalid_argument(
            "adjoint_coulomb_energy_cell: the energy seeds must carry "
            "n_graphs.");

    // The forward's own intermediates, recomputed exactly as the forward
    // computes them (the shared-cache contract).
    const LongRangeMatrix<double> density =
        assemble_fourier(cache, charge_density);
    const LongRangeMatrix<double> potential = apply_coulomb(cache, density);
    const std::size_t n_k = cache.n_k;
    const double scale = 0.5 / (two_pi_cubed() * two_pi_cubed());

    // The per-k product seed of the index_add reduction: c = 2 V_g
    // 0.5/(2 pi)^6 seed_g lands on both factors, so the density seed
    // below is the adjoint_coulomb_energy one and the potential branch's
    // rho factor feeds the direct Coulomb-kernel k seed.
    LongRangeMatrix<double> grad_density(
        "long_range_energy_cell_grad_density", n_k, 2);
    for (std::size_t i = 0; i < n_k; ++i) {
        const std::size_t g =
            static_cast<std::size_t>(cache.k_graph_index_host(i));
        const double c =
            2.0 * scale * cache.volumes_host(g) * grad_energy(g);
        const double coulomb_factor = cache.k_norm2_host(i) == 0.0
            ? 0.0
            : 1.0 / cache.k_norm2_host(i);
        grad_density(i, 0) = (c * potential(i, 0))
            + (c * density(i, 0)) * coulomb_factor * field_constant;
        grad_density(i, 1) = (c * potential(i, 1))
            + (c * density(i, 1)) * coulomb_factor * field_constant;
    }

    // The shared assemble cell seeds -- the one spelling of the step's
    // 1/V and Fourier-factor dependence.
    EnergyCellAdjointResult result;
    AssembleCellAdjointResult assembled =
        assemble_cell_seeds(
            cache, grad_density, charge_density, include_table_seeds);
    result.grad_cos_table = std::move(assembled.grad_cos_table);
    result.grad_sin_table = std::move(assembled.grad_sin_table);
    result.grad_k_vectors = std::move(assembled.grad_k_vectors);
    result.grad_volume_per_k = std::move(assembled.grad_volume_per_k);

    for (std::size_t i = 0; i < n_k; ++i) {
        const std::size_t g =
            static_cast<std::size_t>(cache.k_graph_index_host(i));
        // The direct Coulomb-kernel k seed: only one factor of the
        // rho_bar . V(k) product carries the kernel's k dependence (the
        // other flows through the assemble chain), zero exactly at the
        // origin mode (the branchy-mask convention).
        if (cache.k_norm2_host(i) != 0.0) {
            const double c =
                2.0 * scale * cache.volumes_host(g) * grad_energy(g);
            const double rho_squared = density(i, 0) * density(i, 0)
                + density(i, 1) * density(i, 1);
            const double factor = -2.0 * field_constant * rho_squared
                / (cache.k_norm2_host(i) * cache.k_norm2_host(i));
            for (int j = 0; j < 3; ++j)
                result.grad_k_vectors(i, j) +=
                    c * factor * cache.k_vectors_host(3 * i + j);
        }
        // The energy's own V prefactor in its natural per-k spelling:
        // dE/dvolume_per_k(i) = scale_g 2 Re(rho_bar . V)_i.
        result.grad_volume_per_k(i) +=
            scale * grad_energy(g) * 2.0
                * (density(i, 0) * potential(i, 0)
                    + density(i, 1) * potential(i, 1));
    }
    return result;
}


template <typename Scalar>
LongRangeTensor3<double>
PolarLongRangeOpsT<Scalar>::adjoint_correction_energy_cell(
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeConstMatrix<int>& graph_pbc,
    const LongRangeConstVector<double>& grad_energy) const
{
    const std::size_t n_graphs = cell.extent(0);
    if (grad_energy.extent(0) != n_graphs)
        throw std::invalid_argument(
            "adjoint_correction_energy_cell: the energy seeds must carry "
            "n_graphs.");
    LongRangeVector<double> total_charge;
    LongRangeMatrix<double> total_dipole;
    LongRangeVector<double> quadrupole;
    graph_multipoles(charge_density, positions, node_graph_index, n_graphs,
        total_charge, total_dipole, quadrupole);

    const double constant = field_constant / (4.0 * pi);
    LongRangeTensor3<double> grad_strain(
        "long_range_correction_grad_strain", n_graphs, 3, 3);
    for (std::size_t g = 0; g < n_graphs; ++g) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                grad_strain(g, a, b) = 0.0;
        const bool periodic_x = graph_pbc(g, 0) != 0;
        const bool periodic_y = graph_pbc(g, 1) != 0;
        const bool periodic_z = graph_pbc(g, 2) != 0;
        const bool is_molecule =
            !periodic_x && !periodic_y && !periodic_z;
        const bool is_slab = periodic_x && periodic_y && !periodic_z;
        if (!is_molecule && !is_slab)
            continue;
        const double volume = cell_determinant(&cell(g, 0, 0));
        // dE/dV through the term's volume factors only, then
        // dV/deps_ab = V delta_ab.
        double d_energy_d_volume = 0.0;
        if (is_molecule) {
            const double charge = total_charge(g);
            const double dipole_norm_squared = total_dipole(g, 0)
                    * total_dipole(g, 0)
                + total_dipole(g, 1) * total_dipole(g, 1)
                + total_dipole(g, 2) * total_dipole(g, 2);
            // The box length's literal exponent (four threes): the
            // derivative of V^-0.3333 is -0.3333 V^(-1.3333).
            d_energy_d_volume += -0.3333 * 0.5 * cubic_madelung * constant
                * charge * charge
                * std::pow(volume, -0.3333 - 1.0);
            d_energy_d_volume += -2.0 * constant * pi * dipole_norm_squared
                / (3.0 * volume * volume);
            d_energy_d_volume += 2.0 * constant * pi * charge * quadrupole(g)
                / (3.0 * volume * volume);
        } else {
            double dipole_z = 0.0;
            for (std::size_t n = 0; n < charge_density.extent(0); ++n) {
                if (static_cast<std::size_t>(node_graph_index(n)) != g)
                    continue;
                dipole_z += positions(n, 2) * charge_density(n, 0);
                if (charge_density.extent(1) > 1)
                    dipole_z += charge_density(n, 2);
            }
            d_energy_d_volume = -constant * 2.0 * pi * dipole_z * dipole_z
                / (volume * volume);
        }
        const double diagonal =
            grad_energy(g) * d_energy_d_volume * volume;
        for (int a = 0; a < 3; ++a)
            grad_strain(g, a, a) = diagonal;
    }
    return grad_strain;
}

template <typename Scalar>
LongRangeTensor3<double>
PolarLongRangeOpsT<Scalar>::adjoint_correction_features_cell(
    const LongRangeConstMatrix<double>& charge_density_alpha,
    const LongRangeConstMatrix<double>& charge_density_beta,
    const LongRangeConstMatrix<double>& positions,
    const LongRangeConstTensor3<double>& cell,
    const LongRangeConstVector<int>& node_graph_index,
    const LongRangeConstMatrix<int>& graph_pbc,
    const LongRangeConstMatrix<double>& grad_fields_alpha,
    const LongRangeConstMatrix<double>& grad_fields_beta) const
{
    const std::size_t n_graphs = cell.extent(0);
    const std::size_t n_nodes = positions.extent(0);
    if (grad_fields_alpha.extent(0) != n_nodes
        || grad_fields_alpha.extent(1) != feature_dim_
        || grad_fields_beta.extent(0) != n_nodes
        || grad_fields_beta.extent(1) != feature_dim_)
        throw std::invalid_argument(
            "adjoint_correction_features_cell: the corrective-field seeds "
            "must carry [n_nodes, feature_dim] per channel.");
    if (charge_density_alpha.extent(0) != n_nodes
        || charge_density_beta.extent(0) != n_nodes
        || positions.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_correction_features_cell: positions must carry "
            "[n_nodes, 3] and both channel densities must agree on the node "
            "count.");
    if (graph_pbc.extent(0) != n_graphs || graph_pbc.extent(1) != 3)
        throw std::invalid_argument(
            "adjoint_correction_features_cell: graph_pbc must carry "
            "[n_graphs, 3].");

    const double constant = field_constant / (4.0 * pi);
    const LongRangeConstMatrix<double>* densities[2] = {
        &charge_density_alpha, &charge_density_beta};
    const LongRangeConstMatrix<double>* seeds[2] = {
        &grad_fields_alpha, &grad_fields_beta};
    LongRangeTensor3<double> grad_strain(
        "long_range_correction_features_grad_strain", n_graphs, 3, 3);
    for (std::size_t g = 0; g < n_graphs; ++g) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                grad_strain(g, a, b) = 0.0;
        const bool periodic_x = graph_pbc(g, 0) != 0;
        const bool periodic_y = graph_pbc(g, 1) != 0;
        const bool periodic_z = graph_pbc(g, 2) != 0;
        const bool is_molecule =
            !periodic_x && !periodic_y && !periodic_z;
        const bool is_slab = periodic_x && periodic_y && !periodic_z;
        if (!is_molecule && !is_slab)
            continue;
        const double volume = cell_determinant(&cell(g, 0, 0));
        // dL/dV through the node fields' volume factors only; positions and
        // density stay fixed cartesian, and dV/deps_ab = V delta_ab.
        double d_loss_d_volume = 0.0;
        for (int channel = 0; channel < 2; ++channel) {
            LongRangeMatrix<double> node_field_seeds(
                "long_range_correction_node_field_seeds", n_nodes, 4);
            correction_node_field_seeds(*seeds[channel], node_field_seeds);
            LongRangeVector<double> total_charge;
            LongRangeMatrix<double> total_dipole;
            LongRangeVector<double> quadrupole;
            graph_multipoles(*densities[channel], positions,
                node_graph_index, n_graphs, total_charge, total_dipole,
                quadrupole);
            if (is_molecule) {
                const double charge = total_charge(g);
                const double inverse_3v = 1.0 / (3.0 * volume);
                const double inverse_3v_squared = inverse_3v / volume;
                // The box length's literal exponent (six threes): the
                // derivative of V^-0.333333 is -0.333333 V^(-1.333333), and
                // the field's 1/L factor carries it once.
                const double length_derivative = -0.333333
                    * std::pow(volume, -0.333333 - 1.0);
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    const double s0 = node_field_seeds(n, 0);
                    double r_squared = 0.0;
                    double dipole_dot_r = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        r_squared += positions(n, d) * positions(n, d);
                        dipole_dot_r += total_dipole(g, d) * positions(n, d);
                    }
                    d_loss_d_volume += s0
                        * (cubic_madelung * constant * charge
                                * length_derivative
                            + 2.0 * constant * pi * charge * r_squared
                                * inverse_3v_squared
                            - 4.0 * constant * pi * dipole_dot_r
                                * inverse_3v_squared
                            + 2.0 * constant * pi * quadrupole(g)
                                * inverse_3v_squared);
                    for (int d = 0; d < 3; ++d)
                        d_loss_d_volume += node_field_seeds(n, 1 + d)
                            * (-4.0 * constant * pi
                                * (total_dipole(g, d)
                                    - charge * positions(n, d))
                                * inverse_3v_squared);
                }
            } else {
                const double field_z = 4.0 * constant * pi
                    * total_dipole(g, 2) / volume;
                double seed_field = 0.0;
                for (std::size_t n = 0; n < n_nodes; ++n) {
                    if (static_cast<std::size_t>(node_graph_index(n)) != g)
                        continue;
                    seed_field += node_field_seeds(n, 0) * positions(n, 2)
                        + node_field_seeds(n, 3);
                }
                d_loss_d_volume += -field_z * seed_field / volume;
            }
        }
        const double diagonal = d_loss_d_volume * volume;
        for (int a = 0; a < 3; ++a)
            grad_strain(g, a, a) = diagonal;
    }
    return grad_strain;
}

template <typename Scalar>
LongRangeTensor3<double> PolarLongRangeOpsT<Scalar>::coulomb_stress(
    const LongRangeGeometryCacheT<Scalar>& cache,
    const LongRangeConstVector<double>& grad_energy,
    const LongRangeMatrix<double>* grad_features,
    const LongRangeConstMatrix<double>& charge_density,
    const LongRangeConstMatrix<double>& positions) const
{
    if (grad_energy.extent(0) != cache.n_graphs)
        throw std::invalid_argument(
            "coulomb_stress: the energy seeds must carry n_graphs.");
    if (positions.extent(0) != cache.n_nodes || positions.extent(1) != 3)
        throw std::invalid_argument(
            "coulomb_stress: positions must carry [n_nodes, 3].");

    // The energy path's cell seeds.
    EnergyCellAdjointResult energy =
        adjoint_coulomb_energy_cell(cache, grad_energy, charge_density);
    LongRangeMatrix<double> grad_cos_table = std::move(energy.grad_cos_table);
    LongRangeMatrix<double> grad_sin_table = std::move(energy.grad_sin_table);
    LongRangeMatrix<double> grad_k_vectors = std::move(energy.grad_k_vectors);
    LongRangeVector<double> grad_volume_per_k =
        std::move(energy.grad_volume_per_k);

    // The features path's cell chain (the same shared steps the composed
    // position gradient walks), added seed by seed.
    if (grad_features != nullptr) {
        if (grad_features->extent(0) != cache.n_nodes
            || grad_features->extent(1) != feature_dim_)
            throw std::invalid_argument(
                "coulomb_stress: the feature seeds must carry "
                "[n_nodes, feature_dim].");
        const LongRangeMatrix<double> density =
            assemble_fourier(cache, charge_density);
        const LongRangeMatrix<double> potential =
            apply_coulomb(cache, density);
        const ProjectCellAdjointResult projected =
            adjoint_project_cell(cache, *grad_features, potential);
        const CoulombCellAdjointResult convolved = adjoint_coulomb_cell(
            cache, projected.grad_potential, density);
        const AssembleCellAdjointResult assembled = assemble_cell_seeds(
            cache, convolved.grad_density, charge_density);
        for (std::size_t i = 0; i < cache.n_k; ++i) {
            for (int j = 0; j < 3; ++j) {
                grad_k_vectors(i, j) += projected.grad_k_vectors(i, j)
                    + convolved.grad_k_vectors(i, j)
                    + assembled.grad_k_vectors(i, j);
            }
            grad_volume_per_k(i) += assembled.grad_volume_per_k(i);
            for (std::size_t n = 0; n < cache.n_nodes; ++n) {
                grad_cos_table(i, n) += projected.grad_cos_table(i, n)
                    + assembled.grad_cos_table(i, n);
                grad_sin_table(i, n) += projected.grad_sin_table(i, n)
                    + assembled.grad_sin_table(i, n);
            }
        }
    }

    // The composed virial: the cell seeds' strain plus the position
    // virial of the same loss, divided by the graph volume -- the
    // affine-deformation stress (the phase terms cancel between the two,
    // as they must: k.r is invariant under the affine deformation).
    const LongRangeTensor3<double> strain = adjoint_geometry_cell(
        cache, grad_k_vectors, grad_volume_per_k, grad_cos_table,
        grad_sin_table, positions);
    const LongRangeMatrix<double> grad_positions =
        adjoint_geometry(cache, grad_cos_table, grad_sin_table);
    LongRangeTensor3<double> stress(
        "long_range_stress", cache.n_graphs, 3, 3);
    for (std::size_t g = 0; g < cache.n_graphs; ++g) {
        double inverse_volume = 1.0 / cache.volumes_host(g);
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double virial = 0.0;
                for (std::size_t n = 0; n < cache.n_nodes; ++n)
                    if (static_cast<std::size_t>(
                            cache.node_graph_index_host(n))
                        == g)
                        virial += positions(n, a) * grad_positions(n, b);
                stress(g, a, b) = (strain(g, a, b) + virial) * inverse_volume;
            }
        }
    }
    return stress;
}
template class PolarLongRangeOpsT<float>;
template class PolarLongRangeOpsT<double>;

} // namespace symmetrix::execution::long_range
