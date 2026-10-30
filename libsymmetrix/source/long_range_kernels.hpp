#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Kokkos_Core.hpp"

namespace symmetrix::mace_polar {
template<typename Precision> class PolarDeltaChainKokkosT;
}

// The portable (T1 host) forward kernels of the long-range operator kind
// group declared in long_range_operator_kinds.hpp.  Everything dynamic is a
// GEMM against precomputed real tables, so each kernel is host Kokkos views
// plus KokkosBlas on the host execution space -- the correctness backbone
// that later stages (adjoints, device tiers) are validated against.
//
// See docs/user/models.md#mace-polar-1 for the model contract. The Coulomb kernel is
// FIELD_CONSTANT/k^2 with the k=0 mode zeroed by an explicit mask; the
// half-sphere factor-2 weights live in the features and energy kernels, not
// in the k-grid; the energy path consumes the same shared cache as the
// features path; radial Fourier factors are the analytic closed forms for
// l <= 1 (there is no spline anywhere); the self-interaction overlaps and
// the [features_dim, 4] field matrices are constants derived in closed form
// from the basis parameters (sigma set, widths) exactly as the oracle's
// gto_utils derives them -- fixed-grid init-time quadrature, never runtime
// quadrature.
//
// Precision: kernel-space storage and GEMMs use Scalar, selected from the
// evaluator precision by the production chain. Numerically sensitive scalar
// reductions remain in double before their result is stored back to Scalar.
// The public PolarLongRangeOps alias remains the fp64 diagnostic surface used
// by the golden operator fixtures and NumPy bindings.
namespace symmetrix::execution::long_range {

// Host-space views with row-major (LayoutRight) layout so the whole-rank
// KokkosBlas GEMMs map directly onto the host BLAS.  These spell the kind
// group's public I/O: callers hand it host buffers (the chain's vectors,
// the bindings' numpy arrays) and read the results on the host.
using LongRangeExecSpace = Kokkos::DefaultHostExecutionSpace;
template <typename T>
using LongRangeVector = Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace>;
template <typename T>
using LongRangeMatrix = Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace>;
template <typename T>
using LongRangeTensor3 = Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace>;
template <typename T>
using LongRangeTensor4 = Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::HostSpace>;

// The kernel space: the default execution space, which is the device when
// one is initialized (CUDA builds) and the host space otherwise.  The kind
// group's kernels, GEMMs, tables, and workspaces live here, so on a device
// build the composition's per-evaluation round-trips shrink to the small
// host-facing operands below instead of the [n_k, n] tables.
using LongRangeKernelSpace = Kokkos::DefaultExecutionSpace;
using LongRangeKernelMemory = LongRangeKernelSpace::memory_space;
template <typename T>
using LongRangeKernelVector =
    Kokkos::View<T*, Kokkos::LayoutRight, LongRangeKernelMemory>;
template <typename T>
using LongRangeKernelMatrix =
    Kokkos::View<T**, Kokkos::LayoutRight, LongRangeKernelMemory>;
template <typename T>
using LongRangeKernelTensor3 =
    Kokkos::View<T***, Kokkos::LayoutRight, LongRangeKernelMemory>;
template <typename T>
using LongRangeKernelTensor4 =
    Kokkos::View<T****, Kokkos::LayoutRight, LongRangeKernelMemory>;

// Unmanaged read-only aliases for caller-owned buffers.
template <typename T>
using LongRangeConstVector = Kokkos::View<const T*, Kokkos::LayoutRight,
    Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using LongRangeConstMatrix = Kokkos::View<const T**, Kokkos::LayoutRight,
    Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using LongRangeConstTensor3 = Kokkos::View<const T***, Kokkos::LayoutRight,
    Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

enum class LongRangeGeometryMode {
    empty,
    realspace,
    direct_reciprocal,
};

// Current live payload owned by cache/workspace views. ``elements`` counts
// typed allocation slots; ``bytes`` applies each slot's actual value type.
struct LongRangeAllocationUsage {
    std::size_t elements = 0;
    std::size_t bytes = 0;

    LongRangeAllocationUsage& operator+=(const LongRangeAllocationUsage& other)
    {
        elements += other.elements;
        bytes += other.bytes;
        return *this;
    }
};

// Lightweight launch metadata for tests and profiling.  It records only
// extents selected by the production energy path; no device data is exposed.
struct RealspaceEnergyLaunchDiagnostics {
    std::size_t selected_count = 0;
    std::size_t receiver_replica_league_extent = 0;
    std::size_t graph_reduction_league_extent = 0;
    std::size_t partial_elements = 0;
    std::size_t full_materialized_serial_selected_count = 0;
    std::size_t full_materialized_parallel_selected_count = 0;
    std::size_t full_materialized_parallel_edge_extent = 0;
};

enum class RealspaceReverseStrategy {
    owner,
    pair_atomic,
    pair_tile,
    pair_materialized,
    materialized_forward
};
enum class RealspaceReverseStrategyRequest {
    automatic,
    owner,
    pair_atomic,
    pair_tile,
    pair_materialized,
    materialized_forward
};
enum class RealspaceFullMaterializedReducer {
    serial,
    atomic_scatter,
    segmented
};

inline std::string_view to_string(const RealspaceReverseStrategy strategy)
{
    switch (strategy) {
    case RealspaceReverseStrategy::owner: return "owner";
    case RealspaceReverseStrategy::pair_atomic: return "pair_atomic";
    case RealspaceReverseStrategy::pair_tile: return "pair_tile";
    case RealspaceReverseStrategy::pair_materialized: return "pair_materialized";
    case RealspaceReverseStrategy::materialized_forward:
        return "materialized_forward";
    }
    return "unknown";
}

inline std::string_view to_string(
    const RealspaceReverseStrategyRequest strategy)
{
    switch (strategy) {
    case RealspaceReverseStrategyRequest::automatic: return "auto";
    case RealspaceReverseStrategyRequest::owner: return "owner";
    case RealspaceReverseStrategyRequest::pair_atomic: return "pair_atomic";
    case RealspaceReverseStrategyRequest::pair_tile: return "pair_tile";
    case RealspaceReverseStrategyRequest::pair_materialized:
        return "pair_materialized";
    case RealspaceReverseStrategyRequest::materialized_forward:
        return "materialized_forward";
    }
    return "unknown";
}

inline std::string_view to_string(
    const RealspaceFullMaterializedReducer reducer)
{
    switch (reducer) {
    case RealspaceFullMaterializedReducer::serial: return "serial";
    case RealspaceFullMaterializedReducer::atomic_scatter:
        return "atomic_scatter";
    case RealspaceFullMaterializedReducer::segmented: return "segmented";
    }
    return "unknown";
}

// Launch extents selected by the real-space reverse stages. Owner extents
// describe the bounded reference path; pair extents and FP64 accumulator
// bytes describe the pair-atomic path when selected.
struct RealspaceReverseLaunchDiagnostics {
    std::size_t feature_selected_count = 0;
    std::size_t feature_multipole_league_extent = 0;
    std::size_t feature_position_league_extent = 0;
    std::size_t energy_selected_count = 0;
    std::size_t energy_multipole_league_extent = 0;
    std::size_t energy_position_league_extent = 0;
    std::size_t feature_owner_selected_count = 0;
    std::size_t energy_owner_selected_count = 0;
    std::size_t feature_pair_atomic_selected_count = 0;
    std::size_t feature_pair_league_extent = 0;
    std::size_t energy_pair_atomic_selected_count = 0;
    std::size_t energy_pair_league_extent = 0;
    std::size_t atomic_accumulator_bytes = 0;
    std::size_t feature_pair_tile_selected_count = 0;
    std::size_t feature_pair_tile_league_extent = 0;
    std::size_t energy_pair_tile_selected_count = 0;
    std::size_t energy_pair_tile_league_extent = 0;
    std::size_t tile_width = 16;
    std::size_t partial_accumulator_bytes = 0;
    std::size_t feature_materialized_selected_count = 0;
    std::size_t energy_materialized_selected_count = 0;
    std::size_t materialized_table_build_count = 0;
    std::size_t materialized_table_reuse_count = 0;
    std::size_t materialized_table_invalidation_count = 0;
    std::size_t materialized_admission_rejection_count = 0;
    std::size_t materialized_radial_entries = 0;
    std::size_t materialized_radial_bytes = 0;
    std::size_t full_materialized_table_build_count = 0;
    std::size_t full_materialized_table_reuse_count = 0;
    std::size_t full_materialized_table_invalidation_count = 0;
    std::size_t full_materialized_admission_rejection_count = 0;
    std::size_t full_materialized_feature_directed_edges = 0;
    std::size_t full_materialized_energy_directed_edges = 0;
    std::size_t full_materialized_admission_bytes = 0;
    std::size_t full_materialized_retained_bytes = 0;
    bool full_materialized_has_gradients = false;
    std::size_t full_materialized_feature_serial_selected_count = 0;
    std::size_t full_materialized_feature_atomic_selected_count = 0;
    std::size_t full_materialized_feature_segmented_selected_count = 0;
    std::size_t full_materialized_feature_parallel_edge_extent = 0;
    std::size_t full_materialized_feature_pair_atomic_reverse_count = 0;
    std::size_t full_materialized_energy_pair_atomic_reverse_count = 0;
};

// The outputs of the kspace_geometry_cache kind: one per evaluation, shared
// by the features and energy kinds, both spin channels, and every recursion
// step (the binding-level sharing the declarative layer asserts).  The
// tables live in the kernel space; the small per-k and per-graph metadata
// the drivers' serial logic reads carries host mirrors filled at build
// time, so no runtime download of them is ever needed.
template <typename Scalar>
struct LongRangeGeometryCacheT {
    LongRangeGeometryMode mode = LongRangeGeometryMode::empty;
    std::size_t n_k = 0;
    std::size_t n_nodes = 0;
    std::size_t n_graphs = 0;
    LongRangeKernelMatrix<Scalar> k_vectors;       // [n_k, 3]
    LongRangeKernelVector<Scalar> k_norm2;         // [n_k]
    LongRangeKernelVector<int> k_graph_index;      // [n_k]
    LongRangeKernelVector<int> node_graph_index;   // [n_nodes]
    LongRangeKernelVector<Scalar> k0_mask;         // [n_k] 1.0 at the origin mode
    LongRangeKernelVector<Scalar> volume_per_k;    // [n_k]
    LongRangeKernelVector<Scalar> k_factor_coulomb;  // [n_k] 1/k^2, zero at k=0
    LongRangeKernelVector<Scalar> k_factor_proj;   // [n_k] 0.5 at k=0, else 1
    LongRangeKernelMatrix<Scalar> cos_table;       // [n_k, n_nodes]
    LongRangeKernelMatrix<Scalar> sin_table;       // [n_k, n_nodes]
    LongRangeKernelTensor4<Scalar> feature_fourier_factors;  // [n_k, n_sigma, m_f, 2]
    LongRangeKernelTensor4<Scalar> density_fourier_factors;  // [n_k, 1, m_d, 2]
    LongRangeKernelVector<Scalar> kernel_volumes;  // [n_graphs]
    // Real-space positions follow the public calculator/native FP64 boundary
    // even when learned multipoles and compact outputs use FP32 storage.
    LongRangeKernelMatrix<double> realspace_positions;  // [n_nodes, 3]
    LongRangeKernelMatrix<Scalar> positions_workspace;  // [n_nodes, 3]
    LongRangeKernelMatrix<Scalar> phase_workspace;      // [n_k, n_nodes]
    // Host mirrors of the metadata the serial drivers read (the k-grid
    // enumeration fills them before anything is uploaded).
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> k_vectors_host;
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> k_norm2_host;
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> volume_per_k_host;
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> volumes_host;
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> k_graph_index_host;
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> node_graph_index_host;
    std::vector<double> positions_host_values;
    std::vector<double> cell_host_values;
    std::vector<int> node_graph_host_values;

    LongRangeAllocationUsage realspace_allocation_usage() const
    {
        LongRangeAllocationUsage usage;
        if (mode != LongRangeGeometryMode::realspace)
            return usage;
        const auto add_view = [&usage] (const auto& view) {
            const std::size_t elements = static_cast<std::size_t>(view.span());
            usage.elements += elements;
            usage.bytes += elements
                * sizeof(typename std::decay_t<decltype(view)>::value_type);
        };
        const auto add_vector = [&usage] (const auto& values) {
            const std::size_t elements = values.capacity();
            usage.elements += elements;
            usage.bytes += elements
                * sizeof(typename std::decay_t<decltype(values)>::value_type);
        };
        add_view(node_graph_index);
        add_view(realspace_positions);
        add_view(node_graph_index_host);
        add_vector(positions_host_values);
        add_vector(node_graph_host_values);
        return usage;
    }

    LongRangeAllocationUsage reciprocal_allocation_usage() const
    {
        LongRangeAllocationUsage usage;
        const auto add_view = [&usage] (const auto& view) {
            const std::size_t elements = static_cast<std::size_t>(view.span());
            usage.elements += elements;
            usage.bytes += elements
                * sizeof(typename std::decay_t<decltype(view)>::value_type);
        };
        const auto add_vector = [&usage] (const auto& values) {
            const std::size_t elements = values.capacity();
            usage.elements += elements;
            usage.bytes += elements
                * sizeof(typename std::decay_t<decltype(values)>::value_type);
        };
        add_view(k_vectors);
        add_view(k_norm2);
        add_view(k_graph_index);
        add_view(k0_mask);
        add_view(volume_per_k);
        add_view(k_factor_coulomb);
        add_view(k_factor_proj);
        add_view(cos_table);
        add_view(sin_table);
        add_view(feature_fourier_factors);
        add_view(density_fourier_factors);
        add_view(kernel_volumes);
        add_view(phase_workspace);
        add_view(k_vectors_host);
        add_view(k_norm2_host);
        add_view(volume_per_k_host);
        add_view(volumes_host);
        add_view(k_graph_index_host);
        add_vector(cell_host_values);
        return usage;
    }
};

// The portable forward kernels of every declared kind.
template <typename Scalar>
class PolarLongRangeOpsT {
public:
    // The basis parameters of the kind group's contract data.  Validation
    // mirrors the declarative layer: the analytic radial Fourier factors
    // hard-reject l > 1.
    PolarLongRangeOpsT(double kspace_cutoff,
        int density_max_l,
        double density_smearing_width,
        int feature_max_l,
        std::vector<double> feature_smearing_widths);

    double kspace_cutoff() const { return kspace_cutoff_; }
    int density_max_l() const { return density_max_l_; }
    double density_smearing_width() const { return density_smearing_width_; }
    int feature_max_l() const { return feature_max_l_; }
    const std::vector<double>& feature_smearing_widths() const
    {
        return feature_smearing_widths_;
    }
    // The e3nn-sorted (l-major) feature spelling of the project stage.
    const std::vector<std::size_t>& output_permutation() const
    {
        return output_permutation_;
    }

    // --- kspace_geometry_cache -------------------------------------------
    // Builds the half-sphere k-grid from the reciprocal cell (one of +/-k,
    // grouped by graph), the K.R^T phase GEMM into masked cos/sin tables,
    // the per-k factors, and the analytic GTO Fourier factors. Standalone
    // calls own their returned/cache storage and do not mutate the evaluator's
    // composed scratch arena.
    LongRangeGeometryCacheT<Scalar> geometry_cache(
        const LongRangeConstMatrix<double>& positions,       // [n_nodes, 3]
        const LongRangeConstTensor3<double>& cell,           // [n_graphs, 3, 3]
        const LongRangeConstVector<int>& node_graph_index) const;  // [n_nodes]
    void geometry_cache_into(
        const LongRangeConstMatrix<double>& positions,
        const LongRangeConstTensor3<double>& cell,
        const LongRangeConstVector<int>& node_graph_index,
        LongRangeGeometryCacheT<Scalar>& cache) const;
    std::size_t reciprocal_mode_count_up_to(
        const LongRangeConstTensor3<double>& cell,
        std::size_t maximum_count) const;
    LongRangeGeometryCacheT<Scalar> realspace_geometry_cache(
        const LongRangeConstMatrix<double>& positions,
        const LongRangeConstTensor3<double>& cell,
        const LongRangeConstVector<int>& node_graph_index) const;
    void realspace_geometry_cache_into(
        const LongRangeConstMatrix<double>& positions,
        const LongRangeConstTensor3<double>& cell,
        const LongRangeConstVector<int>& node_graph_index,
        LongRangeGeometryCacheT<Scalar>& cache) const;
    LongRangeAllocationUsage realspace_cache_allocation_usage(
        const LongRangeGeometryCacheT<Scalar>& cache) const;
    LongRangeAllocationUsage reciprocal_cache_allocation_usage(
        const LongRangeGeometryCacheT<Scalar>& cache) const;
    static constexpr std::size_t realspace_pair_tile() { return 128; }
    std::size_t arena_view_allocation_count() const
    {
        return arena_view_allocation_count_;
    }
    std::size_t input_h2d_bytes() const { return input_h2d_bytes_; }
    LongRangeAllocationUsage workspace_allocation_usage() const;
    const RealspaceEnergyLaunchDiagnostics&
    realspace_energy_launch_diagnostics() const
    {
        return realspace_energy_launch_diagnostics_;
    }
    const RealspaceReverseLaunchDiagnostics&
    realspace_reverse_launch_diagnostics() const
    {
        return realspace_reverse_launch_diagnostics_;
    }
    void set_realspace_reverse_strategy(std::string_view strategy);
    void set_realspace_full_materialized_reducer(std::string_view reducer);
    void set_realspace_pair_tile_width(std::size_t tile_width);
    std::size_t checked_realspace_pair_tile_scratch_bytes() const;
    std::size_t checked_realspace_pair_tile_feature_league_extent(
        std::size_t n_nodes) const;
    std::size_t checked_realspace_pair_tile_energy_league_extent(
        std::size_t n_nodes) const;
    void validate_realspace_pair_tile_backend() const;
    void set_realspace_materialized_budget_bytes(std::size_t budget_bytes);
    void set_realspace_full_materialized_gradients_required(bool required)
    {
        realspace_full_materialized_gradients_required_ = required;
    }
    std::size_t checked_realspace_materialized_table_bytes(
        std::size_t n_nodes) const;
    std::size_t checked_realspace_full_materialized_bytes(
        std::size_t n_nodes, bool include_gradients) const;
    // Public solely because nvcc requires an enclosing member of an extended
    // device lambda to have public access. It remains an internal native API.
    void ensure_realspace_materialized_tables(
        const LongRangeGeometryCacheT<Scalar>& cache) const;
    void ensure_realspace_full_materialized_tables(
        const LongRangeGeometryCacheT<Scalar>& cache,
        bool include_gradients) const;
    std::size_t checked_realspace_atomic_accumulator_bytes(
        std::size_t n_nodes) const;
    RealspaceReverseStrategy realspace_reverse_strategy() const
    {
        return realspace_reverse_strategy_;
    }
    RealspaceFullMaterializedReducer realspace_full_materialized_reducer() const
    {
        return realspace_full_materialized_reducer_;
    }
    RealspaceReverseStrategyRequest realspace_reverse_strategy_request() const
    {
        return realspace_reverse_strategy_request_;
    }
    std::string_view realspace_reverse_strategy_resolved() const
    {
        return realspace_reverse_strategy_resolved_
            ? to_string(realspace_reverse_strategy_)
            : "unresolved";
    }
    std::string_view realspace_reverse_strategy_reason() const
    {
        return realspace_reverse_strategy_reason_;
    }

    // --- kspace_coulomb_features -----------------------------------------
    // rho(k) = (2 pi)^3/V_k * [basis-weighted combination of cos.c and
    // sin.c] -- two thin GEMMs [n_k, n] @ [n, m_d] plus a broadcast combine.
    LongRangeMatrix<double> assemble_fourier(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& multipoles) const;  // [n_nodes, m_d]
    // V(k) = rho(k) * (FIELD_CONSTANT / k^2), k=0 zeroed exactly.
    LongRangeMatrix<double> apply_coulomb(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& density) const;     // [n_k, 2]
    // phi(n) = 2/(2 pi)^3 * sum_k [A.cos + B.sin] -- two transposed GEMMs
    // [m, n_k] @ [n_k, n], then the l-major channel permutation.
    LongRangeMatrix<double> project_to_features(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& potential) const;   // [n_k, 2]
    // Composition surface for the common assemble -> Coulomb -> project
    // sequence.  Only the caller's multipoles are uploaded and the final
    // node features downloaded; both [n_k, 2] intermediates remain in the
    // kernel memory space.
    LongRangeMatrix<double> coulomb_features(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& multipoles,
        bool use_workspace = false) const;

    // Complete within-graph real-space pair stages. The cache and all
    // intermediates stay in kernel memory; these host-returning overloads are
    // diagnostic materialization surfaces only.
    LongRangeMatrix<double> realspace_features(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& multipoles,
        bool include_self_interaction,
        bool use_workspace = false) const;
    LongRangeVector<double> realspace_energy(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& multipoles,
        bool include_self_interaction,
        bool use_workspace = false) const;

    // Host materialization exists only for stage diagnostics/tests. Production
    // composition consumes the kernel-space result types declared below.
    struct RealspaceAdjointDiagnosticResult {
        LongRangeMatrix<double> grad_multipoles;  // [n, m_d]
        LongRangeMatrix<double> grad_positions;   // [n, 3], always fp64
        LongRangeTensor3<double> grad_cell;        // [graphs, 3, 3], exact zero
    };
    RealspaceAdjointDiagnosticResult diagnostic_adjoint_realspace_features(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& multipoles,
        const LongRangeConstMatrix<double>& grad_features,
        bool include_self_interaction,
        bool use_workspace = false) const;
    RealspaceAdjointDiagnosticResult diagnostic_adjoint_realspace_energy(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& multipoles,
        const LongRangeConstVector<double>& grad_energy,
        bool include_self_interaction,
        bool use_workspace = false) const;

    // Native reverse of the composed feature path.  All [n_k, 2]
    // intermediates and their seeds remain in kernel memory; only the node
    // multipole seed is materialized for the still-host-resident chain state.
    // Shared table and cell-metadata seeds accumulate into caller-owned
    // device views.
    LongRangeMatrix<double> adjoint_coulomb_features_composed(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& multipoles,
        const LongRangeConstMatrix<double>& grad_features,
        const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_k_accumulator,
        const LongRangeKernelVector<Scalar>& grad_volume_accumulator) const;

    // --- kspace_coulomb_energy -------------------------------------------
    // E = 0.5 * V * sum_k 2*Re(rho_bar.V)/(2 pi)^6 per graph, fp64
    // accumulators.  When energy_self_fields is non-null (the node's
    // include_self_interaction=0 contract) the self-interaction overlaps
    // 0.5*sum_n(source.self_fields) are subtracted per graph.
    LongRangeVector<double> coulomb_energy(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& charge_density,     // [n_nodes, m_d]
        const LongRangeMatrix<double>* energy_self_fields,       // [n_graphs]
        bool use_workspace = false) const;

    // --- self_interaction_linear -----------------------------------------
    // Constants derived at construction from the basis parameters exactly
    // as the oracle's gto_utils derives them (fixed 10 000-point init-time
    // trapezoid quadrature); the forwards are gathers and broadcast scales.
    const LongRangeVector<double>& si_feature_constants() const
    {
        return si_feature_constants_;
    }
    const LongRangeVector<double>& si_energy_constants() const
    {
        return si_energy_constants_;
    }
    LongRangeMatrix<double> feature_self_terms(
        const LongRangeConstMatrix<double>& multipoles) const;  // [n_nodes, f_dim]
    LongRangeMatrix<double> energy_self_fields(
        const LongRangeConstMatrix<double>& multipoles) const;  // [n_nodes, m_d]

    // --- external_field_projection ---------------------------------------
    // The constant [f_dim, 4] matrix (l=0 and l=1 analytic entries).
    const LongRangeMatrix<double>& external_field_matrix() const
    {
        return external_field_matrix_;
    }
    // The gauge term E.(r - barycenter) through the constant matrix, halved
    // per spin channel; the barycenter is a per-graph fp64 scatter mean.
    // A null/empty norms argument means the unit vector (the norm divide is
    // the family's linear composition, not the kernel's); a non-empty vector
    // must carry exactly one entry per feature, or the call throws.
    LongRangeMatrix<double> external_field_features(
        const LongRangeConstMatrix<double>& external_potential,  // [n_graphs, 4]
        const LongRangeConstMatrix<double>& positions,           // [n_nodes, 3]
        const LongRangeConstVector<int>& node_graph_index,       // [n_nodes]
        const LongRangeVector<double>& field_feature_norms,       // [f_dim]
        bool use_workspace = false,
        const LongRangeKernelVector<int>* prepared_node_graph = nullptr) const;

    // --- fukui_normalize -------------------------------------------------
    struct FukuiResult {
        LongRangeVector<double> alpha;  // [n_nodes]
        LongRangeVector<double> beta;   // [n_nodes]
    };
    // Normalize the per-node sources against their per-graph sums
    // (zero-guarded, fp64 accumulators), then broadcast the charge/spin
    // targets onto the l=0 channel of each spin channel.
    FukuiResult fukui_normalize(
        const LongRangeConstMatrix<double>& fukui_sources,   // [n_nodes, 2]
        const LongRangeConstVector<double>& charge_l0_alpha, // [n_nodes]
        const LongRangeConstVector<double>& charge_l0_beta,  // [n_nodes]
        const LongRangeConstVector<double>& total_charge,    // [n_graphs]
        const LongRangeConstVector<double>& total_spin,      // [n_graphs]
        const LongRangeConstVector<int>& node_graph_index) const;

    // --- dipole_readout ---------------------------------------------------
    struct DipoleResult {
        LongRangeVector<double> total_charge;   // [n_graphs]
        LongRangeMatrix<double> total_dipole;   // [n_graphs, 3]
    };
    // Total charge and dipole of the (spin-summed) density, fp64
    // accumulators; the l=1 block carries the e3nn (y, z, x) phase order.
    DipoleResult dipole_readout(
        const LongRangeConstMatrix<double>& charge_density,  // [n_nodes, m_d]
        const LongRangeConstMatrix<double>& positions,       // [n_nodes, 3]
        const LongRangeConstVector<int>& node_graph_index) const;

    // --- analytic corrections ---------------------------------------------
    struct CorrectionEnergyResult {
        LongRangeVector<double> selected;    // [n_graphs] the mode the pbc row picks
        LongRangeVector<double> molecule;    // [n_graphs] Makov-Payne/Dabo box term
        LongRangeVector<double> slab;        // [n_graphs] 2 pi mu_z^2 dipole term
    };
    // The box term for molecule graphs (pbc all false), the slab term for
    // (T, T, F) graphs, zero otherwise -- selected per graph from its pbc
    // row, never a mode parameter.
    CorrectionEnergyResult analytic_correction_energy(
        const LongRangeConstMatrix<double>& charge_density,  // [n_nodes, m_d]
        const LongRangeConstMatrix<double>& positions,       // [n_nodes, 3]
        const LongRangeConstTensor3<double>& cell,           // [n_graphs, 3, 3]
        const LongRangeConstVector<int>& node_graph_index,
        const LongRangeConstMatrix<int>& graph_pbc) const;   // [n_graphs, 3]

    struct CorrectionFeatureResult {
        LongRangeMatrix<double> alpha;  // [n_nodes, f_dim]
        LongRangeMatrix<double> beta;   // [n_nodes, f_dim]
    };
    // Point-charge corrective potential node fields for molecule graphs,
    // z-dipole corrective fields for slab graphs, projected onto GTO
    // features by the constant matrix; per spin channel.
    CorrectionFeatureResult analytic_correction_features(
        const LongRangeConstMatrix<double>& charge_density_alpha,
        const LongRangeConstMatrix<double>& charge_density_beta,
        const LongRangeConstMatrix<double>& positions,
        const LongRangeConstTensor3<double>& cell,
        const LongRangeConstVector<int>& node_graph_index,
        const LongRangeConstMatrix<int>& graph_pbc) const;


    // =====================================================================
    // Reverse-mode adjoints
    //
    // The derivative surface of the design note's Adjoint set: every stage
    // consumes the seed of its own forward output and returns the seeds of
    // its declared inputs (fp64 accumulation throughout; the GEMM adjoints
    // reuse the same tables and shapes as the forwards).  The stages
    // compose: the feature and energy paths share the assemble step and
    // the cos/sin tables, so a composed position gradient adds their table
    // seeds before the geometry adjoint fires.
    // =====================================================================

    // --- geometry-cache position adjoint ---------------------------------
    // d cos(k.r)/dr = -k sin(k.r) and d sin(k.r)/dr = +k cos(k.r): the
    // seeds of the two tables become dL/dr through the cache's own masked
    // tables (cross-graph entries are zero there, so their seeds are
    // inert) and one [n_k, n] x [n_k, 3] transposed GEMM.
    LongRangeMatrix<double> adjoint_geometry(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_cos_table,  // [n_k, n]
        const LongRangeConstMatrix<double>& grad_sin_table) const;  // [n, 3]

    // --- assemble adjoint --------------------------------------------------
    // Seeds of rho(k) -> the two thin-GEMM adjoints: table seeds (for the
    // geometry adjoint) and dL/d(multipoles).
    struct AssembleAdjointResult {
        LongRangeMatrix<double> grad_cos_table;   // [n_k, n]
        LongRangeMatrix<double> grad_sin_table;   // [n_k, n]
        LongRangeMatrix<double> grad_multipoles;  // [n, m_d]
    };
    AssembleAdjointResult adjoint_assemble(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_density,  // [n_k, 2]
        const LongRangeConstMatrix<double>& multipoles) const;  // [n, m_d]

    // Composition-only variant: add the two large table seeds directly to
    // caller-owned kernel-space accumulators and return only the small
    // host-facing multipole seed.  This keeps a composed reverse pass from
    // downloading [n_k, n] tables after every stage.
    AssembleAdjointResult adjoint_assemble_accumulate(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_density,
        const LongRangeConstMatrix<double>& multipoles,
        const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator) const;

    // --- Coulomb convolve adjoint ------------------------------------------
    // The kernel is diagonal in k with the branchy k=0 zero: the adjoint
    // carries the same factor, exactly zero at the origin mode.
    LongRangeMatrix<double> adjoint_coulomb(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_potential) const;  // [n_k, 2]

    // --- project adjoint -----------------------------------------------------
    // Seeds of the features -> dL/d(potential) (the two transposed-GEMM
    // adjoints through the forward amplitudes, recomputed from the given
    // potential) and the table seeds (for the geometry adjoint).
    struct ProjectAdjointResult {
        LongRangeMatrix<double> grad_potential;   // [n_k, 2]
        LongRangeMatrix<double> grad_cos_table;   // [n_k, n]
        LongRangeMatrix<double> grad_sin_table;   // [n_k, n]
    };
    ProjectAdjointResult adjoint_project(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_features,  // [n, f_dim]
        const LongRangeConstMatrix<double>& potential) const;  // [n_k, 2]

    // Composition-only variant of adjoint_project with device-resident table
    // accumulation.  The returned table views are empty; grad_potential keeps
    // the public host representation consumed by the following stage.
    ProjectAdjointResult adjoint_project_accumulate(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_features,
        const LongRangeConstMatrix<double>& potential,
        const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator) const;

    // Consume table seeds that already live in kernel memory.  These are the
    // terminal operations of a composed reverse pass; only their compact
    // position/strain results cross back to the host.
    LongRangeKernelMatrix<Scalar> adjoint_geometry_device(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& grad_cos_table,
        const LongRangeKernelMatrix<Scalar>& grad_sin_table) const;

    // --- Coulomb energy adjoint ----------------------------------------------
    // Per-graph scalar seeds -> dL/d(charge density) through the shared
    // assemble/convolve chain (the intermediates are recomputed exactly as
    // the forward computes them) plus the constant-linear self-interaction
    // subtraction, and the table seeds (for the geometry adjoint).
    struct EnergyAdjointResult {
        LongRangeMatrix<double> grad_charge_density;  // [n, m_d]
        LongRangeMatrix<double> grad_cos_table;       // [n_k, n]
        LongRangeMatrix<double> grad_sin_table;       // [n_k, n]
    };
    EnergyAdjointResult adjoint_coulomb_energy(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstVector<double>& grad_energy,  // [n_graphs]
        const LongRangeConstMatrix<double>& charge_density,  // [n, m_d]
        const LongRangeMatrix<double>* energy_self_fields) const;

    // Native-composition variant: the energy path contributes its phase-table
    // adjoints directly to the same kernel-space accumulator as the polar
    // chain.  Only the compact charge-density seed is materialized on host.
    EnergyAdjointResult adjoint_coulomb_energy_accumulate(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstVector<double>& grad_energy,
        const LongRangeConstMatrix<double>& charge_density,
        const LongRangeMatrix<double>* energy_self_fields,
        const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator) const;

    struct ComposedEnergyAdjointResult {
        LongRangeMatrix<double> grad_charge_density;  // [n, m_d]
        LongRangeTensor3<double> grad_strain;         // [graphs, 3, 3]
    };
    ComposedEnergyAdjointResult adjoint_coulomb_energy_composed(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstVector<double>& grad_energy,
        const LongRangeConstMatrix<double>& charge_density,
        const LongRangeMatrix<double>* energy_self_fields,
        const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator) const;

    // Resident counterpart used by the production polar composition. The
    // density and all density/table/cell seeds stay in kernel memory.
    LongRangeKernelMatrix<Scalar>
    adjoint_coulomb_energy_composed_device(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& charge_density,
        bool subtract_self_interaction,
        const LongRangeKernelVector<Scalar>& self_constants,
        const LongRangeKernelVector<std::size_t>& self_select,
        const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator,
        const LongRangeKernelTensor3<double>& grad_strain) const;

    // --- self-interaction adjoints --------------------------------------------
    // The forwards are gathers plus broadcast scales of constants; their
    // adjoints are the scatter transposes (constant-linear, no runtime
    // quadrature).
    LongRangeMatrix<double> adjoint_feature_self_terms(
        const LongRangeConstMatrix<double>& grad_terms) const;  // [n, m_d]
    LongRangeMatrix<double> adjoint_energy_self_fields(
        const LongRangeConstMatrix<double>& grad_fields) const;  // [n, m_d]

    // --- external-field adjoint -----------------------------------------------
    // The constant-matrix transpose plus the gauge term's position
    // derivative: E.(r - barycenter) is linear in the positions of every
    // node of the graph at once (the barycenter scatter), so the position
    // adjoint carries the mean-subtracted broadcast.
    struct ExternalFieldAdjointResult {
        LongRangeMatrix<double> grad_positions;           // [n, 3]
        LongRangeMatrix<double> grad_external_potential;  // [n_graphs, 4]
    };
    ExternalFieldAdjointResult adjoint_external_field(
        const LongRangeConstMatrix<double>& grad_features,       // [n, f_dim]
        const LongRangeConstMatrix<double>& external_potential,  // [g, 4]
        const LongRangeConstMatrix<double>& positions,           // [n, 3]
        const LongRangeConstVector<int>& node_graph_index,       // [n]
        // [f_dim]; empty means unit norms, a set vector must match f_dim.
        const LongRangeVector<double>& field_feature_norms,
        bool use_workspace = false) const;

    // --- dipole-readout adjoint ------------------------------------------------
    // The readout is linear in the positions on the l=0 channel (q*r) and
    // constant-linear in the density; the adjoint of the l=1 gather keeps
    // the e3nn (y, z, x) phase order.
    struct DipoleAdjointResult {
        LongRangeMatrix<double> grad_positions;       // [n, 3]
        LongRangeMatrix<double> grad_charge_density;  // [n, m_d]
    };
    DipoleAdjointResult adjoint_dipole_readout(
        const LongRangeConstMatrix<double>& charge_density,      // [n, m_d]
        const LongRangeConstMatrix<double>& positions,           // [n, 3]
        const LongRangeConstVector<int>& node_graph_index,       // [n]
        const LongRangeConstVector<double>& grad_total_charge,   // [g]
        const LongRangeConstMatrix<double>& grad_total_dipole) const;  // [g, 3]

    // --- Fukui-normalize adjoint ------------------------------------------
    // The first global scatter-and-broadcast adjoint in the framework: the
    // forward normalizes the per-node sources by their per-graph scatter and
    // broadcasts the charge/spin target through the per-graph charge
    // scatter, so the adjoint scatters the broadcast gradient back onto
    // every node of the graph and returns the graph-level charge/spin
    // seeds.  The zero guard is differentiated as the forward writes it: a
    // graph whose source sum vanishes holds the norm at one, so the norm
    // carries no derivative there.
    struct FukuiAdjointResult {
        LongRangeMatrix<double> grad_fukui_sources;    // [n, 2]
        LongRangeVector<double> grad_charge_l0_alpha;  // [n]
        LongRangeVector<double> grad_charge_l0_beta;   // [n]
        LongRangeVector<double> grad_total_charge;     // [g]
        LongRangeVector<double> grad_total_spin;       // [g]
    };
    FukuiAdjointResult adjoint_fukui_normalize(
        const LongRangeConstMatrix<double>& fukui_sources,   // [n, 2]
        const LongRangeConstVector<double>& charge_l0_alpha, // [n]
        const LongRangeConstVector<double>& charge_l0_beta,  // [n]
        const LongRangeConstVector<double>& total_charge,    // [g]
        const LongRangeConstVector<double>& total_spin,      // [g]
        const LongRangeConstVector<int>& node_graph_index,   // [n]
        const LongRangeConstVector<double>& grad_alpha,      // [n]
        const LongRangeConstVector<double>& grad_beta) const;  // [n]

    // --- analytic-correction position adjoints -----------------------------
    // The box/slab corrections are graph aggregates of the density
    // multipoles (the charge, the dipole, the quadrupole) plus per-node
    // terms, so their position adjoints scatter the graph-level seeds back
    // onto every node of the graph -- exactly the aggregate structure
    // graph_multipoles carries.  Mode selection stays data (the graph's pbc
    // row); the l=1 block keeps the e3nn (y, z, x) order.
    struct CorrectionEnergyAdjointResult {
        LongRangeMatrix<double> grad_charge_density;  // [n, m_d]
        LongRangeMatrix<double> grad_positions;       // [n, 3]
    };
    CorrectionEnergyAdjointResult adjoint_correction_energy(
        const LongRangeConstMatrix<double>& charge_density,  // [n, m_d]
        const LongRangeConstMatrix<double>& positions,       // [n, 3]
        const LongRangeConstTensor3<double>& cell,           // [g, 3, 3]
        const LongRangeConstVector<int>& node_graph_index,   // [n]
        const LongRangeConstMatrix<int>& graph_pbc,          // [g, 3]
        const LongRangeConstVector<double>& grad_energy) const;  // [g]

    struct CorrectionFeatureAdjointResult {
        LongRangeMatrix<double> grad_charge_density_alpha;  // [n, m_d]
        LongRangeMatrix<double> grad_charge_density_beta;   // [n, m_d]
        LongRangeMatrix<double> grad_positions;             // [n, 3]
    };
    CorrectionFeatureAdjointResult adjoint_correction_features(
        const LongRangeConstMatrix<double>& charge_density_alpha,  // [n, m_d]
        const LongRangeConstMatrix<double>& charge_density_beta,   // [n, m_d]
        const LongRangeConstMatrix<double>& positions,             // [n, 3]
        const LongRangeConstTensor3<double>& cell,                 // [g, 3, 3]
        const LongRangeConstVector<int>& node_graph_index,         // [n]
        const LongRangeConstMatrix<int>& graph_pbc,                // [g, 3]
        const LongRangeConstMatrix<double>& grad_fields_alpha,     // [n, f_dim]
        const LongRangeConstMatrix<double>& grad_fields_beta) const;  // [n, f_dim]

    // =====================================================================
    // Cell (strain) adjoints: the reciprocal-space virial
    //
    // The classic Ewald reciprocal stress surface.  Under the strain
    // spelling cell -> cell (I + eps) (row-vector lattice rows) with
    // positions held fixed in cartesian coordinates, the cell-dependent
    // quantities of the forward chains are exactly the cache's k vectors
    // (the reciprocal cell deforms as 2 pi inv(A)^T, so
    // dk_j/deps_ab = -delta_ja k_b), the per-k volumes
    // (dV/deps_ab = V delta_ab), and the phase tables
    // (d(k.r)/deps_ab = -r_a k_b -- the same coefficient the position
    // adjoint reduces).  The k-grid masking boundary is piecewise
    // constant in strain (the oracle re-masks a fixed integer superset
    // per deformed cell), so the adjoint differentiates at fixed grid
    // membership -- the derivative is through |k| and the prefactors,
    // never through grid membership.  Every stage adjoint consumes the
    // seed of its forward output and returns the seeds of the
    // cell-dependent cache quantities it reaches (plus the position
    // chain's seeds, exactly like its position counterpart); the stages
    // compose by adding seeds before adjoint_geometry_cell fires.  The
    // affine stress (positions scaled with the cell) adds the position
    // virial sum_n (dL/dr_n)_b r_na through the position adjoints --
    // coulomb_stress composes both (the phase terms then cancel, as
    // they must: k.r is exactly invariant under the affine deformation).
    // =====================================================================

    // --- geometry-cache cell adjoint -------------------------------------
    // The seeds of the cache's cell-dependent outputs (k vectors, per-k
    // volumes, tables) become dL/d(strain) per graph at fixed cartesian
    // positions: one [n_k, n] x [n_k, 3] GEMM (the phase coefficient is
    // the adjoint_geometry one) plus fp64 host reductions.
    LongRangeTensor3<double> adjoint_geometry_cell(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_k_vectors,    // [n_k, 3]
        const LongRangeConstVector<double>& grad_volume_per_k, // [n_k]
        const LongRangeConstMatrix<double>& grad_cos_table,    // [n_k, n]
        const LongRangeConstMatrix<double>& grad_sin_table,    // [n_k, n]
        const LongRangeConstMatrix<double>& positions) const;  // [n, 3]

    LongRangeTensor3<double> adjoint_geometry_cell_device(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_k_vectors,
        const LongRangeConstVector<double>& grad_volume_per_k,
        const LongRangeKernelMatrix<Scalar>& grad_cos_table,
        const LongRangeKernelMatrix<Scalar>& grad_sin_table,
        const LongRangeConstMatrix<double>& positions) const;

    LongRangeKernelTensor3<Scalar> adjoint_geometry_cell_all_device(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& grad_k_vectors,
        const LongRangeKernelVector<Scalar>& grad_volume_per_k,
        const LongRangeKernelMatrix<Scalar>& grad_cos_table,
        const LongRangeKernelMatrix<Scalar>& grad_sin_table,
        const LongRangeConstMatrix<double>& positions,
        bool use_workspace = false) const;

    // Cell adjoint for cache metadata only.  The composed polar reverse has
    // already consumed the shared phase-table seeds through its chain, so the
    // energy path contributes only k-vector and per-k volume terms here.
    LongRangeTensor3<double> adjoint_geometry_cell_metadata(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_k_vectors,
        const LongRangeConstVector<double>& grad_volume_per_k) const;

    // --- assemble cell adjoint ---------------------------------------------
    // Seeds of rho(k) -> the position-adjoint products of the assemble
    // stage plus its cell seeds: the 1/V volume factor and the density
    // Fourier factors' |k| and k-direction dependence (the l=1 angular
    // factor's |k| cancels against the direction's division, leaving an
    // exactly linear k_q -- the derivative is the closed form below).
    struct AssembleCellAdjointResult {
        LongRangeMatrix<double> grad_cos_table;    // [n_k, n]
        LongRangeMatrix<double> grad_sin_table;    // [n_k, n]
        LongRangeMatrix<double> grad_multipoles;   // [n, m_d]
        LongRangeMatrix<double> grad_k_vectors;    // [n_k, 3]
        LongRangeVector<double> grad_volume_per_k; // [n_k]
    };
    // ``include_table_seeds = false`` leaves grad_cos_table, grad_sin_table,
    // and grad_multipoles empty and skips their GEMMs and host downloads:
    // the composed position chain consumes only the k-vector and per-k
    // volume seeds, and the discarded [n_k, n] table downloads measured as
    // gigabytes per evaluation at production node counts.
    AssembleCellAdjointResult adjoint_assemble_cell(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_density,  // [n_k, 2]
        const LongRangeConstMatrix<double>& multipoles,  // [n, m_d]
        bool include_table_seeds = true,
        const LongRangeKernelMatrix<Scalar>* grad_k_accumulator = nullptr,
        const LongRangeKernelVector<Scalar>* grad_volume_accumulator = nullptr) const;

    // --- Coulomb convolve cell adjoint --------------------------------------
    // The kernel's diagonal 1/k^2 factor adds dL/dk through the density
    // values (zero at the origin mode, exactly as the forward zeroes it).
    struct CoulombCellAdjointResult {
        LongRangeMatrix<double> grad_density;    // [n_k, 2] (position chain)
        LongRangeMatrix<double> grad_k_vectors;  // [n_k, 3]
    };
    CoulombCellAdjointResult adjoint_coulomb_cell(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_potential,  // [n_k, 2]
        const LongRangeConstMatrix<double>& density,  // [n_k, 2]
        const LongRangeKernelMatrix<Scalar>* grad_k_accumulator = nullptr) const;

    // --- project cell adjoint -------------------------------------------------
    // The projection amplitudes' Fourier factors carry the feature basis'
    // |k| and k-direction dependence; the transposed-GEMM products are the
    // position adjoint's own.
    struct ProjectCellAdjointResult {
        LongRangeMatrix<double> grad_potential;   // [n_k, 2]
        LongRangeMatrix<double> grad_cos_table;   // [n_k, n]
        LongRangeMatrix<double> grad_sin_table;   // [n_k, n]
        LongRangeMatrix<double> grad_k_vectors;   // [n_k, 3]
    };
    // ``include_table_seeds = false`` leaves grad_cos_table and
    // grad_sin_table empty and skips their GEMMs and host downloads; the
    // composed chain reads only grad_potential and grad_k_vectors.
    ProjectCellAdjointResult adjoint_project_cell(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_features,  // [n, f_dim]
        const LongRangeConstMatrix<double>& potential,  // [n_k, 2]
        bool include_table_seeds = true,
        const LongRangeKernelMatrix<Scalar>* grad_k_accumulator = nullptr) const;

    // --- Coulomb energy cell adjoint -------------------------------------------
    // Per-graph energy seeds -> the cell seeds of the whole energy chain
    // (the forward intermediates recomputed exactly as the forward computes
    // them): the assemble factors, the Coulomb kernel, the energy's V
    // prefactor (its natural per-k spelling is dE/dvolume_per_k(i) =
    // scale_g * 2 Re(rho_bar . V)_i), and the phase tables.  The
    // self-interaction subtraction is constant-linear in the density with
    // no cell dependence, so it contributes no cell seeds.
    struct EnergyCellAdjointResult {
        LongRangeMatrix<double> grad_cos_table;    // [n_k, n]
        LongRangeMatrix<double> grad_sin_table;    // [n_k, n]
        LongRangeMatrix<double> grad_k_vectors;    // [n_k, 3]
        LongRangeVector<double> grad_volume_per_k; // [n_k]
    };
    EnergyCellAdjointResult adjoint_coulomb_energy_cell(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstVector<double>& grad_energy,     // [n_graphs]
        const LongRangeConstMatrix<double>& charge_density,  // [n, m_d]
        bool include_table_seeds = true) const;

    // --- analytic correction energy cell adjoint -------------------------------
    // The corrections' volume factors only (the molecule box term's
    // V^0.3333 length, the 1/V dipole/quadrupole and slab terms), selected
    // per graph from its pbc row exactly as the forward selects: the mode
    // is data, positions are fixed cartesian (the position coupling of the
    // multipoles belongs to the position virial, not to the cell term).
    LongRangeTensor3<double> adjoint_correction_energy_cell(
        const LongRangeConstMatrix<double>& charge_density,  // [n, m_d]
        const LongRangeConstMatrix<double>& positions,       // [n, 3]
        const LongRangeConstTensor3<double>& cell,           // [g, 3, 3]
        const LongRangeConstVector<int>& node_graph_index,   // [n]
        const LongRangeConstMatrix<int>& graph_pbc,          // [g, 3]
        const LongRangeConstVector<double>& grad_energy) const;  // [g, 3, 3]

    // --- analytic correction features cell adjoint ------------------------------
    // The corrective node fields' volume factors (the box term's V^0.333333
    // length and 1/V terms, the slab field's 1/V), per spin channel and per
    // graph, at fixed cartesian positions: the mode is data, and the
    // position coupling belongs to the position virial, exactly as the
    // correction-energy cell adjoint spells it.
    LongRangeTensor3<double> adjoint_correction_features_cell(
        const LongRangeConstMatrix<double>& charge_density_alpha,  // [n, m_d]
        const LongRangeConstMatrix<double>& charge_density_beta,   // [n, m_d]
        const LongRangeConstMatrix<double>& positions,             // [n, 3]
        const LongRangeConstTensor3<double>& cell,                 // [g, 3, 3]
        const LongRangeConstVector<int>& node_graph_index,         // [n]
        const LongRangeConstMatrix<int>& graph_pbc,                // [g, 3]
        const LongRangeConstMatrix<double>& grad_fields_alpha,     // [n, f_dim]
        const LongRangeConstMatrix<double>& grad_fields_beta) const;  // [n, f_dim]

    // --- stress assembly --------------------------------------------------------
    // sigma_ab = (1/Omega_g) dL/deps_ab of L = <grad_energy, energy> +
    // <grad_features, features> under the affine deformation (cell ->
    // cell(I + eps), positions scaled with it): the energy path's cell
    // adjoint, the features path's cell chain (when grad_features is
    // non-null), and the position virial of the same loss through the
    // position adjoints -- the composed reciprocal-space virial, fp64
    // accumulation.  The returned tensor is the full derivative; a
    // symmetric strain probe measures its symmetric part.
    //
    // SCOPE: this assembly is the K-SPACE (Coulomb) stress and nothing else.
    // The analytic corrections' volume terms and their position coupling are
    // NOT composed here: they are separate adjoints
    // (adjoint_correction_energy_cell, adjoint_correction_features_cell, and
    // the correction position adjoints above) whose inputs -- the cell rows
    // and the per-graph pbc rows -- this signature does not carry, and whose
    // kinds carry their own cell-adjoint capability.  A caller that reads
    // `stress = true` off a correction kind's descriptor and calls this
    // assembly gets a PARTIAL stress; the family's stress reduction must add
    // the correction terms itself (the end-to-end parity stage owns the
    // composed gate).  The finite-strain battery pins this scope: it
    // differences the k-space loss, so composing the corrections here without
    // extending the oracle would fail it.
    LongRangeTensor3<double> coulomb_stress(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstVector<double>& grad_energy,    // [n_graphs]
        const LongRangeMatrix<double>* grad_features,       // [n, f_dim] or null
        const LongRangeConstMatrix<double>& charge_density, // [n, m_d]
        const LongRangeConstMatrix<double>& positions) const;  // -> [g, 3, 3]

private:

    template<typename Precision>
    friend class ::symmetrix::mace_polar::PolarDeltaChainKokkosT;

    void record_input_h2d(
        std::size_t elements,std::size_t element_size,const char* what) const;

    double kspace_cutoff_;
    int density_max_l_;
    double density_smearing_width_;
    int feature_max_l_;
    std::vector<double> feature_smearing_widths_;
    std::size_t density_width_ = 0;   // (density_max_l + 1)^2
    std::size_t feature_width_ = 0;   // (feature_max_l + 1)^2
    std::size_t feature_dim_ = 0;     // feature_width_ * sigma count
    std::vector<std::size_t> output_permutation_;
    LongRangeKernelVector<std::size_t> output_permutation_kernel_;
    std::vector<std::size_t> si_feature_select_;
    std::vector<std::size_t> si_energy_select_;
    LongRangeVector<double> si_feature_constants_;
    LongRangeVector<double> si_energy_constants_;
    LongRangeMatrix<double> external_field_matrix_;
    LongRangeKernelMatrix<Scalar> external_field_kernel_;
    LongRangeKernelMatrix<Scalar> density_fourier_parameters_;
    LongRangeKernelMatrix<Scalar> feature_fourier_parameters_;
    LongRangeKernelMatrix<Scalar> feature_fourier_derivatives_;
    // FP32 real-space storage is evaluated through an FP64 arithmetic policy;
    // immutable radial/projection/self constants therefore stay FP64 too.
    LongRangeKernelVector<double> realspace_effective_widths_;
    LongRangeKernelVector<double> realspace_l0_factors_;
    LongRangeKernelVector<double> realspace_l1_factors_;
    LongRangeKernelVector<double> si_feature_constants_kernel_;
    LongRangeKernelVector<double> si_energy_constants_kernel_;
    LongRangeKernelVector<std::size_t> si_feature_select_kernel_;
    LongRangeKernelVector<std::size_t> si_energy_select_kernel_;

    // Evaluator-owned scratch for the composed production path. The arena is
    // invalidated as a unit when graph or k-grid extents change; distinct
    // labels preserve non-aliasing live ranges within one composition.
    mutable std::unordered_map<std::string, LongRangeKernelMatrix<Scalar>>
        matrix_workspace_;
    mutable std::unordered_map<std::string, LongRangeKernelVector<Scalar>>
        vector_workspace_;
    mutable std::unordered_map<std::string, LongRangeKernelVector<double>>
        realspace_compute_vector_workspace_;
    mutable std::unordered_map<std::string, LongRangeKernelMatrix<double>>
        realspace_compute_matrix_workspace_;
    mutable std::unordered_map<std::string, LongRangeKernelTensor3<double>>
        realspace_compute_tensor3_workspace_;
    mutable std::unordered_map<std::string, LongRangeKernelVector<int>>
        int_vector_workspace_;
    mutable std::unordered_map<std::string, LongRangeKernelTensor3<Scalar>>
        tensor3_workspace_;
    mutable std::size_t workspace_nodes_ = 0;
    mutable std::size_t workspace_graphs_ = 0;
    mutable std::size_t workspace_n_k_ = 0;
    mutable std::size_t arena_view_allocation_count_ = 0;
    mutable std::size_t input_h2d_bytes_ = 0;
    mutable RealspaceEnergyLaunchDiagnostics
        realspace_energy_launch_diagnostics_;
    mutable RealspaceReverseLaunchDiagnostics
        realspace_reverse_launch_diagnostics_;
    mutable RealspaceReverseStrategy realspace_reverse_strategy_ =
        RealspaceReverseStrategy::owner;
    RealspaceReverseStrategyRequest realspace_reverse_strategy_request_ =
        RealspaceReverseStrategyRequest::automatic;
    mutable bool realspace_reverse_strategy_resolved_ = false;
    mutable std::string_view realspace_reverse_strategy_reason_ =
        "reverse_not_requested";
    std::size_t realspace_pair_tile_width_ = 16;
    std::size_t realspace_materialized_budget_bytes_ = 0;
    bool realspace_full_materialized_gradients_required_ = false;
    RealspaceFullMaterializedReducer realspace_full_materialized_reducer_ =
        RealspaceFullMaterializedReducer::segmented;
    mutable std::size_t realspace_geometry_revision_ = 0;
    mutable std::size_t materialized_table_revision_ =
        std::numeric_limits<std::size_t>::max();
    mutable LongRangeKernelVector<double> materialized_feature_k_;
    mutable LongRangeKernelVector<double> materialized_feature_g_;
    mutable LongRangeKernelVector<double> materialized_energy_k_;
    mutable LongRangeKernelVector<double> materialized_energy_g_;
    mutable std::size_t full_materialized_table_revision_ =
        std::numeric_limits<std::size_t>::max();
    mutable bool full_materialized_has_gradients_ = false;
    mutable LongRangeKernelVector<std::uint64_t> full_feature_sender_;
    mutable LongRangeKernelVector<std::uint64_t> full_feature_receiver_;
    mutable LongRangeKernelVector<std::uint8_t> full_feature_valid_;
    mutable LongRangeKernelMatrix<double> full_feature_difference_;
    mutable LongRangeKernelVector<double> full_feature_norm_;
    mutable LongRangeKernelMatrix<double> full_feature_k_;
    mutable LongRangeKernelMatrix<double> full_feature_g_;
    mutable LongRangeKernelMatrix<double> full_feature_contribution_;
    mutable LongRangeKernelMatrix<double> full_feature_replica_positions_;
    mutable LongRangeKernelVector<double> full_feature_replica_charges_;
    mutable LongRangeKernelVector<std::uint64_t> full_energy_sender_;
    mutable LongRangeKernelVector<std::uint64_t> full_energy_receiver_;
    mutable LongRangeKernelVector<std::uint8_t> full_energy_valid_;
    mutable LongRangeKernelMatrix<double> full_energy_difference_;
    mutable LongRangeKernelVector<double> full_energy_norm_;
    mutable LongRangeKernelVector<double> full_energy_k_;
    mutable LongRangeKernelVector<double> full_energy_g_;
    mutable LongRangeKernelVector<double> full_energy_contribution_;
    mutable LongRangeKernelMatrix<double> full_energy_replica_positions_;
    mutable LongRangeKernelVector<double> full_energy_replica_charges_;

    void resolve_realspace_reverse_strategy() const;
    bool admit_realspace_materialized(std::size_t n_nodes) const;
    bool admit_realspace_full_materialized(
        std::size_t n_nodes, bool include_gradients) const;
    void release_realspace_full_materialized_tables() const;
    void prepare_workspace(
        std::size_t n_k, std::size_t n_nodes, std::size_t n_graphs) const;
    LongRangeKernelMatrix<Scalar> workspace_matrix(
        const std::string& label, std::size_t rows, std::size_t columns) const;
    LongRangeKernelVector<Scalar> workspace_vector(
        const std::string& label, std::size_t size) const;
    LongRangeKernelVector<double> workspace_realspace_compute_vector(
        const std::string& label, std::size_t size) const;
    LongRangeKernelMatrix<double> workspace_realspace_compute_matrix(
        const std::string& label, std::size_t rows, std::size_t columns) const;
    LongRangeKernelTensor3<double> workspace_realspace_compute_tensor3(
        const std::string& label,
        std::size_t first,
        std::size_t second,
        std::size_t third) const;
    LongRangeKernelVector<int> workspace_int_vector(
        const std::string& label, std::size_t size) const;
    LongRangeKernelTensor3<Scalar> workspace_tensor3(const std::string& label,
        std::size_t first, std::size_t second, std::size_t third) const;

    // The k-derivative of one basis' analytic Fourier factors at a k row:
    // the l=0 real factor is A0 exp(-k^2 sigma^2 / 2) (derivative
    // -sigma^2 k_j times itself) and the l=1 imaginary factor is exactly
    // -B exp(...) k_q (the radial |k| cancels the direction's division),
    // so dI_p/dk_j = -B exp(...) (delta_qj - sigma^2 k_q k_j) with q the
    // e3nn (y, z, x) permutation of p.
    struct BasisKDerivative {
        double df0_real = 0.0;
        double dimag[3] = {0.0, 0.0, 0.0};
    };
    BasisKDerivative basis_k_derivative(
        int max_l,
        double sigma,
        bool normalize_receiver,
        const double* k,
        double k_norm2,
        int j) const;

    // Shared helper: the self-interaction overlap constants of one
    // (source sigma, receiver sigmas, receiver normalization) pair.
    void build_si_constants(
        int l_source,
        double sigma_source,
        int l_receive,
        const std::vector<double>& sigmas_receive,
        bool normalize_receiver,
        LongRangeVector<double>& constants,
        std::vector<std::size_t>& select) const;

    // Shared helper: per-graph (charge, dipole, quadrupole) of the density
    // in the l=0/l=1 multipoles convention, fp64 accumulators.
    void graph_multipoles(
        const LongRangeConstMatrix<double>& charge_density,
        const LongRangeConstMatrix<double>& positions,
        const LongRangeConstVector<int>& node_graph_index,
        std::size_t n_graphs,
        LongRangeVector<double>& total_charge,
        LongRangeMatrix<double>& total_dipole,
        LongRangeVector<double>& quadrupole) const;

public:
    // nvcc requires every function that encloses a Kokkos extended
    // ``__device__`` lambda to be public, so the helpers that fire kernels sit
    // here while the parse-time helpers above stay private.  The kernels run
    // in ``LongRangeKernelSpace`` (the device on a CUDA build) over
    // kernel-space tables and workspaces; the staging between those and the
    // host-typed public I/O is the methods' own concern.

    // Shared adjoint helper: the two thin-GEMM adjoints of the assemble
    // stage -- seeds of rho(k) into table seeds (for the geometry
    // adjoint) and dL/d(multipoles).  Both adjoint entry points that
    // reach the tables through assemble (the assemble adjoint itself and
    // the Coulomb energy adjoint) use this one spelling so the shared
    // step has exactly one adjoint.
    AssembleAdjointResult assemble_adjoint(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_density,
        const LongRangeConstMatrix<double>& multipoles,
        const LongRangeKernelMatrix<Scalar>* grad_cos_accumulator = nullptr,
        const LongRangeKernelMatrix<Scalar>* grad_sin_accumulator = nullptr) const;

    // Device-resident forward stages. Standalone calls use independent
    // storage, so a later call cannot overwrite a retained result. Only the
    // evaluator's non-reentrant composed path opts into its private arena.
    LongRangeKernelMatrix<Scalar> assemble_fourier_kernel(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& multipoles,
        bool use_workspace = false) const;
    LongRangeKernelMatrix<Scalar> apply_coulomb_kernel(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& density,
        bool use_workspace = false) const;
    LongRangeKernelMatrix<Scalar> project_to_features_kernel(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& potential,
        bool use_workspace = false) const;
    LongRangeKernelMatrix<Scalar> realspace_features_kernel(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& multipoles,
        bool include_self_interaction,
        bool use_workspace = false) const;
    LongRangeKernelVector<Scalar> realspace_energy_kernel(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& multipoles,
        bool include_self_interaction,
        bool use_workspace = false) const;
    struct RealspaceAdjointKernelResult {
        LongRangeKernelMatrix<Scalar> grad_multipoles;  // compact model storage
        LongRangeKernelMatrix<double> grad_positions;   // fp64 Cartesian storage
        LongRangeKernelTensor3<double> grad_cell;       // exact fixed-R zero
    };
    RealspaceAdjointKernelResult adjoint_realspace_features_kernel(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& multipoles,
        const LongRangeKernelMatrix<Scalar>& grad_features,
        bool include_self_interaction,
        bool use_workspace = false) const;
    RealspaceAdjointKernelResult adjoint_realspace_energy_kernel(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& multipoles,
        const LongRangeKernelVector<Scalar>& grad_energy,
        bool include_self_interaction,
        bool use_workspace = false) const;
    LongRangeKernelMatrix<Scalar> adjoint_coulomb_features_kernel(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& multipoles,
        const LongRangeKernelMatrix<Scalar>& grad_features,
        const LongRangeKernelMatrix<Scalar>& grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_sin_accumulator,
        const LongRangeKernelMatrix<Scalar>& grad_k_accumulator,
        const LongRangeKernelVector<Scalar>& grad_volume_accumulator) const;

    // Shared cell-adjoint helper: the assemble stage's seeds of the
    // cell-dependent cache quantities (the 1/V factor and the density
    // Fourier factors' |k| and direction dependence) together with the
    // position-adjoint products.  This lives in the public kernel-helper
    // section because nvcc requires enclosing functions of extended device
    // lambdas to have public access.
    AssembleCellAdjointResult assemble_cell_seeds(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_density,
        const LongRangeConstMatrix<double>& multipoles,
        bool include_table_seeds = true,
        const LongRangeKernelMatrix<Scalar>* grad_k_accumulator = nullptr,
        const LongRangeKernelVector<Scalar>* grad_volume_accumulator = nullptr) const;

    ProjectAdjointResult project_adjoint(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstMatrix<double>& grad_features,
        const LongRangeConstMatrix<double>& potential,
        const LongRangeKernelMatrix<Scalar>* grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>* grad_sin_accumulator) const;

    EnergyAdjointResult coulomb_energy_adjoint(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeConstVector<double>& grad_energy,
        const LongRangeConstMatrix<double>& charge_density,
        const LongRangeMatrix<double>* energy_self_fields,
        const LongRangeKernelMatrix<Scalar>* grad_cos_accumulator,
        const LongRangeKernelMatrix<Scalar>* grad_sin_accumulator) const;

    // Shared helper: the per-k projection amplitudes of the project stage
    // (the forward and its adjoint recompute the identical buffers).
    void projection_amplitudes(
        const LongRangeGeometryCacheT<Scalar>& cache,
        const LongRangeKernelMatrix<Scalar>& potential,
        LongRangeKernelMatrix<Scalar>& amplitude_a,   // [n_k, f_dim]
        LongRangeKernelMatrix<Scalar>& amplitude_b) const;

    // Shared helper: evaluate the analytic GTO Fourier factors of one basis
    // at the cache's k-grid from evaluator-prepared immutable parameters.
    void evaluate_fourier_basis(
        const LongRangeKernelMatrix<Scalar>& k_vectors,
        const LongRangeKernelVector<Scalar>& k_norm2,
        const LongRangeKernelVector<Scalar>& k0_mask,
        int max_l,
        const LongRangeKernelMatrix<Scalar>& parameters,
        const LongRangeKernelTensor4<Scalar>& out) const;

    // Shared helper: the [V, E_x, E_y, E_z] -> feature projection through
    // the constant matrix (column perm [0, 3, 1, 2]).
    void project_node_fields(
        const LongRangeMatrix<double>& node_fields,  // [n_nodes, 4]
        LongRangeMatrix<double>& features) const;    // [n_nodes, f_dim]

    // Shared helper: the constant-matrix transpose of the corrective-field
    // seeds back into the [V, E_x, E_y, E_z] node-field seeds -- the exact
    // adjoint of project_node_fields, so the correction features' position
    // and cell adjoints share one spelling of the projection step.
    void correction_node_field_seeds(
        const LongRangeConstMatrix<double>& grad_fields,  // [n_nodes, f_dim]
        LongRangeMatrix<double>& node_field_seeds) const;  // [n_nodes, 4]
};

using LongRangeGeometryCache = LongRangeGeometryCacheT<double>;
using PolarLongRangeOps = PolarLongRangeOpsT<double>;
using LongRangeGeometryCacheFloat = LongRangeGeometryCacheT<float>;
using PolarLongRangeOpsFloat = PolarLongRangeOpsT<float>;

} // namespace symmetrix::execution::long_range
