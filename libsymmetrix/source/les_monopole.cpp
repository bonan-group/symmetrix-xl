#include "les_monopole.hpp"
#include "les_particle_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace symmetrix::les {
namespace {
constexpr double coulomb_norm_factor = 90.4756;
constexpr double two_pi = 2.0 * std::numbers::pi;

[[noreturn]] void throw_reciprocal_bound_error(const std::string& detail)
{
    throw std::invalid_argument(
        "LES reciprocal bound "+detail
        +"; increase dl or reduce the periodic cell.");
}

std::size_t checked_reciprocal_add(
    const std::size_t left, const std::size_t right, const char* detail)
{
    if (right > std::numeric_limits<std::size_t>::max()-left)
        throw_reciprocal_bound_error(detail);
    return left+right;
}

std::size_t checked_reciprocal_multiply(
    const std::size_t left, const std::size_t right, const char* detail)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max()/right)
        throw_reciprocal_bound_error(detail);
    return left*right;
}

std::size_t checked_node_triplet(const std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max()/3)
        throw std::invalid_argument(
            "LES active node count exceeds supported allocation limits.");
    return 3*count;
}

std::array<double, 9> inverse(const std::vector<double>& matrix)
{
    const double determinant = matrix[0]*(matrix[4]*matrix[8] - matrix[5]*matrix[7])
        - matrix[1]*(matrix[3]*matrix[8] - matrix[5]*matrix[6])
        + matrix[2]*(matrix[3]*matrix[7] - matrix[4]*matrix[6]);
    if (!std::isfinite(determinant) || determinant == 0.0)
        throw std::invalid_argument("LES periodic cell must be finite and nonsingular.");
    const double scale = 1.0/determinant;
    return {
        (matrix[4]*matrix[8] - matrix[5]*matrix[7])*scale,
        (matrix[2]*matrix[7] - matrix[1]*matrix[8])*scale,
        (matrix[1]*matrix[5] - matrix[2]*matrix[4])*scale,
        (matrix[5]*matrix[6] - matrix[3]*matrix[8])*scale,
        (matrix[0]*matrix[8] - matrix[2]*matrix[6])*scale,
        (matrix[2]*matrix[3] - matrix[0]*matrix[5])*scale,
        (matrix[3]*matrix[7] - matrix[4]*matrix[6])*scale,
        (matrix[1]*matrix[6] - matrix[0]*matrix[7])*scale,
        (matrix[0]*matrix[4] - matrix[1]*matrix[3])*scale,
    };
}

double determinant(const std::vector<double>& matrix)
{
    return matrix[0]*(matrix[4]*matrix[8] - matrix[5]*matrix[7])
        - matrix[1]*(matrix[3]*matrix[8] - matrix[5]*matrix[6])
        + matrix[2]*(matrix[3]*matrix[7] - matrix[4]*matrix[6]);
}

void validate_inputs(
    const std::vector<double>& charges, const std::vector<double>& positions)
{
    if (positions.size() != checked_node_triplet(charges.size()))
        throw std::invalid_argument("LES positions must have shape (N, 3).");
    for (const double value : charges)
        if (!std::isfinite(value))
            throw std::invalid_argument("LES charges must be finite.");
    for (const double value : positions)
        if (!std::isfinite(value))
            throw std::invalid_argument("LES positions must be finite.");
}

struct StructureFactorSum {
    double real;
    double imaginary;
};

struct PeriodicAtomOutputSum {
    double energy;
    double charge_seed;
    double force_x;
    double force_y;
    double force_z;
};

template <typename ChargesView, typename PositionsView, typename VectorsView>
struct StructureFactorAtomReduction {
    using value_type = StructureFactorSum;

    ChargesView charges;
    PositionsView positions;
    VectorsView vectors;
    std::size_t reciprocal_index;

    KOKKOS_INLINE_FUNCTION
    void init(value_type& update) const
    {
        update.real = 0.0;
        update.imaginary = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& destination, const value_type& source) const
    {
        destination.real += source.real;
        destination.imaginary += source.imaginary;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t atom, value_type& update) const
    {
        const double phase = positions(3*atom)*vectors(reciprocal_index,0)
            + positions(3*atom+1)*vectors(reciprocal_index,1)
            + positions(3*atom+2)*vectors(reciprocal_index,2);
        update.real += charges(atom)*Kokkos::cos(phase);
        update.imaginary += charges(atom)*Kokkos::sin(phase);
    }
};

template <
    typename ChargesView, typename PositionsView, typename VectorsView,
    typename KernelsView, typename ActiveIndicesView, typename RealView,
    typename ImaginaryView>
struct PeriodicAtomOutputReduction {
    using value_type = PeriodicAtomOutputSum;

    ChargesView charges;
    PositionsView positions;
    VectorsView vectors;
    KernelsView kernels;
    ActiveIndicesView active_indices;
    RealView real;
    ImaginaryView imaginary;
    std::size_t atom;
    double prefactor;

    KOKKOS_INLINE_FUNCTION
    void init(value_type& update) const
    {
        update.energy = 0.0;
        update.charge_seed = 0.0;
        update.force_x = 0.0;
        update.force_y = 0.0;
        update.force_z = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& destination, const value_type& source) const
    {
        destination.energy += source.energy;
        destination.charge_seed += source.charge_seed;
        destination.force_x += source.force_x;
        destination.force_y += source.force_y;
        destination.force_z += source.force_z;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t active_slot, value_type& update) const
    {
        const std::size_t k_index = active_indices(active_slot);
        const double phase = positions(3*atom)*vectors(k_index,0)
            + positions(3*atom+1)*vectors(k_index,1)
            + positions(3*atom+2)*vectors(k_index,2);
        const double cosine = Kokkos::cos(phase);
        const double sine = Kokkos::sin(phase);
        const double kernel = kernels(k_index);
        const double projection =
            real(k_index)*cosine+imaginary(k_index)*sine;
        const double qi = charges(atom);
        update.energy += prefactor*qi*projection*kernel;
        update.charge_seed += 2.0*prefactor*projection*kernel;
        const double force_scale = 2.0*prefactor*qi
            *(real(k_index)*sine-imaginary(k_index)*cosine)*kernel;
        update.force_x += force_scale*vectors(k_index,0);
        update.force_y += force_scale*vectors(k_index,1);
        update.force_z += force_scale*vectors(k_index,2);
    }
};

template <
    typename VectorsView, typename SquaredView, typename KernelsView,
    typename ActiveIndicesView, typename RealView, typename ImaginaryView>
struct PeriodicStrainReduction {
    using value_type = double;

    VectorsView vectors;
    SquaredView squared;
    KernelsView kernels;
    ActiveIndicesView active_indices;
    RealView real;
    ImaginaryView imaginary;
    int row;
    int column;
    double prefactor;
    double sigma_squared;

    KOKKOS_INLINE_FUNCTION
    void init(value_type& update) const
    {
        update = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& destination, const value_type& source) const
    {
        destination += source;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t active_slot, value_type& update) const
    {
        const std::size_t k_index = active_indices(active_slot);
        const double rho_squared = real(k_index)*real(k_index)
            + imaginary(k_index)*imaginary(k_index);
        const double term = prefactor*kernels(k_index)*rho_squared;
        update += term*((row == column ? -1.0 : 0.0)
            +(sigma_squared+2.0/squared(k_index))
                *vectors(k_index,row)*vectors(k_index,column));
    }
};
} // namespace

template <typename Precision>
LESMonopoleKokkos<Precision>::LESMonopoleKokkos(
    const double sigma, const double dl, const bool remove_self_interaction)
    : sigma_(sigma), dl_(dl), remove_self_interaction_(remove_self_interaction)
{
    if (!std::isfinite(sigma_) || sigma_ <= 0.0 || !std::isfinite(dl_) || dl_ <= 0.0)
        throw std::invalid_argument("LES sigma and dl must be finite and positive.");
}

template <typename Precision>
LESMonopoleKokkos<Precision>::~LESMonopoleKokkos() = default;

template <typename Precision>
void LESMonopoleKokkos<Precision>::configure_particle_mesh_for_testing(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t spline_order)
{
    require_les_particle_mesh_available();
    if (nx <= 0 || ny <= 0 || nz <= 0)
        throw std::invalid_argument("LES particle-mesh extents must be positive.");
    if (spline_order < 2 || spline_order > 6)
        throw std::invalid_argument(
            "LES particle-mesh spline order must be between 2 and 6.");
    particle_mesh_shape_ = {nx, ny, nz};
    particle_mesh_spline_order_ = spline_order;
    periodic_algorithm_ = LESPeriodicAlgorithm::particle_mesh;
    particle_mesh_.reset();
    cell_ready_ = false;
}

template <typename Precision>
void LESMonopoleKokkos<Precision>::use_explicit_periodic_for_testing()
{
    periodic_algorithm_ = LESPeriodicAlgorithm::explicit_sum;
    particle_mesh_.reset();
    cell_ready_ = false;
}

template <typename Precision>
void LESMonopoleKokkos<Precision>::reserve_outputs(
    const std::size_t count, const bool periodic)
{
    const std::size_t coordinate_count = checked_node_triplet(count);
    if (coordinate_count > std::numeric_limits<std::size_t>::max()/sizeof(double)
        || count > std::numeric_limits<std::size_t>::max()/sizeof(double))
        throw std::invalid_argument(
            "LES active node count exceeds supported allocation limits.");
    if (atomic_energies_.extent(0) < count)
        Kokkos::realloc(Kokkos::WithoutInitializing, atomic_energies_, count);
    if (energy_.data() == nullptr)
        energy_ = Kokkos::View<double>("LES total energy");
    if (forces_.extent(0) < coordinate_count)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, forces_, coordinate_count);
    if (charge_adjoints_.extent(0) < count)
        Kokkos::realloc(Kokkos::WithoutInitializing, charge_adjoints_, count);
    if (strain_derivative_.extent(0) != 9)
        Kokkos::realloc(Kokkos::WithoutInitializing, strain_derivative_, 9);
    active_nodes_ = count;
}

template <typename Precision>
MonopoleDeviceResult LESMonopoleKokkos<Precision>::compute_nonperiodic_device(
    const Kokkos::View<double*>& charges,
    const Kokkos::View<double*>& positions,
    const std::size_t count)
{
    return compute_nonperiodic_device(
        execution_space(), charges, positions, count);
}

template <typename Precision>
MonopoleDeviceResult LESMonopoleKokkos<Precision>::compute_nonperiodic_device(
    const execution_space& execution_space,
    const Kokkos::View<double*>& charges,
    const Kokkos::View<double*>& positions,
    const std::size_t count)
{
    const std::size_t coordinate_count = checked_node_triplet(count);
    if (charges.extent(0) < count || positions.extent(0) < coordinate_count)
        throw std::invalid_argument("LES device inputs do not cover the active node count.");
    if (periodic_algorithm_ == LESPeriodicAlgorithm::particle_mesh
        && particle_mesh_) {
        particle_mesh_->prepare_particle_count_change(count);
        particle_mesh_->cancel_particle_count_change();
    }
    reserve_outputs(count, false);
    ++geometry_updates_;
    const auto atomic = atomic_energies_;
    const auto force = forces_;
    const auto adjoint = charge_adjoints_;
    const auto strain = strain_derivative_;
    Kokkos::deep_copy(execution_space, strain, 0.0);
    const double sigma = sigma_;
    const bool remove_self = remove_self_interaction_;
    const double prefactor = coulomb_norm_factor/two_pi;
    const double root_two = std::sqrt(2.0);
    const double root_two_over_pi = std::sqrt(2.0/std::numbers::pi);
    const double self = coulomb_norm_factor/(sigma_*std::pow(two_pi, 1.5));
    Kokkos::parallel_for(
        "LES nonperiodic monopole",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(execution_space, 0, count),
        KOKKOS_LAMBDA(const std::size_t atom) {
            const double qi = charges(atom);
            double energy = remove_self ? 0.0 : self*qi*qi;
            double charge_seed = remove_self ? 0.0 : 2.0*self*qi;
            double fx = 0.0, fy = 0.0, fz = 0.0;
            for (std::size_t other=0; other<count; ++other) {
                if (other == atom)
                    continue;
                const double dx = positions(3*other)-positions(3*atom);
                const double dy = positions(3*other+1)-positions(3*atom+1);
                const double dz = positions(3*other+2)-positions(3*atom+2);
                const double radius = Kokkos::sqrt(dx*dx+dy*dy+dz*dz);
                const double kernel = radius == 0.0
                    ? prefactor*root_two_over_pi/sigma
                    : prefactor*::erf(radius/(root_two*sigma))/radius;
                energy += 0.5*qi*charges(other)*kernel;
                charge_seed += charges(other)*kernel;
                if (radius != 0.0) {
                    const double scaled = radius/(root_two*sigma);
                    const double derivative = prefactor*(
                        root_two_over_pi*Kokkos::exp(-scaled*scaled)/(sigma*radius)
                        - ::erf(scaled)/(radius*radius));
                    const double scale = qi*charges(other)*derivative/radius;
                    fx += scale*dx;
                    fy += scale*dy;
                    fz += scale*dz;
                }
            }
            atomic(atom) = energy;
            adjoint(atom) = charge_seed;
            force(3*atom) = fx;
            force(3*atom+1) = fy;
            force(3*atom+2) = fz;
        });
    Kokkos::parallel_reduce(
        "LES nonperiodic energy",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(execution_space, 0, count),
        KOKKOS_LAMBDA(const std::size_t atom, double& sum) {
            sum += atomic(atom);
        }, energy_);
    return {
        energy_, atomic_energies_, forces_, charge_adjoints_, strain_derivative_};
}

template <typename Precision>
void LESMonopoleKokkos<Precision>::prepare_periodic(
    const std::vector<double>& cell)
{
    prepare_periodic(execution_space(), cell);
}

template <typename Precision>
void LESMonopoleKokkos<Precision>::prepare_periodic(
    const execution_space& execution_space,
    const std::vector<double>& cell)
{
    if (cell.size() != 9)
        throw std::invalid_argument("LES periodic cell must have shape (3, 3).");
    for (const double value : cell)
        if (!std::isfinite(value))
            throw std::invalid_argument("LES periodic cell must be finite.");
    if (cell_ready_ && std::equal(cell.begin(), cell.end(), cached_cell_.begin()))
        return;
    const auto cell_inverse = inverse(cell);
    if (periodic_algorithm_ == LESPeriodicAlgorithm::particle_mesh) {
        if (!particle_mesh_) {
            particle_mesh_ = std::make_unique<LESParticleMesh>(
                execution_space,
                particle_mesh_shape_[0],
                particle_mesh_shape_[1],
                particle_mesh_shape_[2],
                particle_mesh_spline_order_);
        }
        particle_mesh_->prepare_cell(
            cell, cell_inverse, std::abs(determinant(cell)), dl_);
        std::copy(cell.begin(), cell.end(), cached_cell_.begin());
        cell_ready_ = true;
        ++table_rebuilds_;
        return;
    }
    constexpr std::size_t maximum_axis_count =
        static_cast<std::size_t>((std::numeric_limits<int>::max()-1)/2);
    std::size_t counts[3];
    std::size_t widths[3];
    for (int axis=0; axis<3; ++axis) {
        const double norm = std::hypot(
            cell[3*axis], cell[3*axis+1], cell[3*axis+2]);
        const double ratio = norm/dl_;
        if (!std::isfinite(ratio))
            throw_reciprocal_bound_error(
                "is non-finite on axis "+std::to_string(axis));
        const double truncated = std::max(1.0, std::floor(ratio));
        if (truncated > static_cast<double>(maximum_axis_count))
            throw_reciprocal_bound_error(
                "exceeds the supported integer range on axis "
                +std::to_string(axis));
        counts[axis] = static_cast<std::size_t>(truncated);
        widths[axis] = checked_reciprocal_add(
            checked_reciprocal_multiply(
                counts[axis], 2, "width multiplication overflows size_t"),
            1, "width addition overflows size_t");
    }
    const std::size_t plane = checked_reciprocal_multiply(
        widths[0], widths[1], "table plane overflows size_t");
    const std::size_t candidates = checked_reciprocal_multiply(
        plane, widths[2], "table volume overflows size_t");
    const std::size_t vector_elements = checked_reciprocal_multiply(
        candidates, 3, "vector triplet count overflows size_t");
    (void)checked_reciprocal_multiply(
        vector_elements, sizeof(double), "vector allocation bytes overflow size_t");
    (void)checked_reciprocal_multiply(
        candidates, 7*sizeof(double)+sizeof(unsigned char),
        "workspace allocation bytes overflow size_t");
    (void)checked_reciprocal_multiply(
        candidates, sizeof(std::size_t),
        "active-index allocation bytes overflow size_t");
    if (candidates
        > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw_reciprocal_bound_error("launch range exceeds PTRDIFF_MAX");

    if (reciprocal_vectors_.extent(0) < candidates)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, reciprocal_vectors_,
            candidates, 3);
    if (reciprocal_squared_.extent(0) < candidates)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, reciprocal_squared_,
            candidates);
    if (reciprocal_kernels_.extent(0) < candidates)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, reciprocal_kernels_,
            candidates);
    if (reciprocal_active_.extent(0) < candidates)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, reciprocal_active_,
            candidates);
    if (reciprocal_active_indices_.extent(0) < candidates)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, reciprocal_active_indices_,
            candidates);
    if (structure_real_.extent(0) < candidates)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, structure_real_,
            candidates);
    if (structure_imaginary_.extent(0) < candidates)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, structure_imaginary_,
            candidates);
    if (reciprocal_cell_.extent(0) != 10)
        Kokkos::realloc(Kokkos::WithoutInitializing, reciprocal_cell_, 10);
    auto cell_host = Kokkos::create_mirror_view(reciprocal_cell_);
    std::copy(cell_inverse.begin(), cell_inverse.end(), cell_host.data());
    cell_host(9) = std::abs(determinant(cell));
    Kokkos::deep_copy(execution_space, reciprocal_cell_, cell_host);
    std::copy(cell.begin(), cell.end(), cached_cell_.begin());
    cell_ready_ = true;
    reciprocal_candidates_ = candidates;
    ++table_rebuilds_;
    const auto inverse_cell = reciprocal_cell_;
    const auto vectors = reciprocal_vectors_;
    const auto squared_values = reciprocal_squared_;
    const auto kernels = reciprocal_kernels_;
    const auto active = reciprocal_active_;
    const auto active_indices = reciprocal_active_indices_;
    const int c0 = static_cast<int>(counts[0]);
    const int c1 = static_cast<int>(counts[1]);
    const int c2 = static_cast<int>(counts[2]);
    const std::size_t width1 = widths[1];
    const std::size_t width2 = widths[2];
    const double cutoff_squared = (two_pi/dl_)*(two_pi/dl_);
    const double sigma = sigma_;
    Kokkos::parallel_for(
        "LES reciprocal table",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
                execution_space, 0, reciprocal_candidates_),
        KOKKOS_LAMBDA(const std::size_t index) {
            const int n2 = static_cast<int>(index%width2)-c2;
            const std::size_t quotient = index/width2;
            const int n1 = static_cast<int>(quotient%width1)-c1;
            const int n0 = static_cast<int>(quotient/width1)-c0;
            const int n[3] = {n0,n1,n2};
            int first = 0;
            while (first < 3 && n[first] == 0)
                ++first;
            double squared = 0.0;
            for (int component=0; component<3; ++component) {
                double k = 0.0;
                for (int axis=0; axis<3; ++axis)
                    k += n[axis]*two_pi*inverse_cell(3*component+axis);
                vectors(index,component) = k;
                squared += k*k;
            }
            const bool include =
                first < 3 && n[first] > 0 && squared <= cutoff_squared;
            active(index) = include ? 1 : 0;
            squared_values(index) = squared;
            kernels(index) = include
                ? 2.0*Kokkos::exp(-0.5*sigma*sigma*squared)/squared : 0.0;
        });
    std::size_t vector_count = 0;
    Kokkos::parallel_scan(
        "Count LES reciprocal vectors",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
                execution_space, 0, reciprocal_candidates_),
        KOKKOS_LAMBDA(
            const std::size_t index,
            std::size_t& offset,
            const bool final) {
            const bool include = active(index) != 0;
            if (final && include)
                active_indices(offset) = index;
            offset += include;
        },
        vector_count);
    reciprocal_vector_count_ = vector_count;
    ++index_pack_count_;
}

template <typename Precision>
void LESMonopoleKokkos<Precision>::prepare_periodic_geometry(
    const execution_space& execution_space,
    const std::vector<double>& cell,
    const std::span<const double> host_positions,
    const Kokkos::View<double*>& device_positions,
    const std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / 3
        || host_positions.size() != 3 * count
        || device_positions.extent(0) < 3 * count) {
        throw std::invalid_argument(
            "LES periodic geometry does not cover the active node count.");
    }
    prepare_particle_count_change(count);
    prepare_periodic(execution_space, cell);
    if (periodic_algorithm_ != LESPeriodicAlgorithm::particle_mesh) {
        return;
    }
    if (!particle_mesh_) {
        throw std::logic_error("LES particle-mesh evaluator is not prepared.");
    }
    reserve_outputs(count, true);
    particle_mesh_->prepare_positions(
        host_positions, device_positions, count);
}

template <typename Precision>
void LESMonopoleKokkos<Precision>::prepare_particle_count_change(
    const std::size_t count)
{
    if (periodic_algorithm_ == LESPeriodicAlgorithm::particle_mesh
        && particle_mesh_) {
        particle_mesh_->prepare_particle_count_change(count);
    }
}

template <typename Precision>
void LESMonopoleKokkos<Precision>::prepare_particle_count_change(
    const execution_space& execution_space,
    const std::size_t count)
{
    if (periodic_algorithm_ != LESPeriodicAlgorithm::particle_mesh) {
        return;
    }
    if (!particle_mesh_) {
        particle_mesh_ = std::make_unique<LESParticleMesh>(
            execution_space,
            particle_mesh_shape_[0],
            particle_mesh_shape_[1],
            particle_mesh_shape_[2],
            particle_mesh_spline_order_);
    }
    particle_mesh_->prepare_particle_count_change(count);
}

template <typename Precision>
MonopoleDeviceResult LESMonopoleKokkos<Precision>::compute_periodic_device(
    const Kokkos::View<double*>& charges,
    const Kokkos::View<double*>& positions,
    const std::size_t count)
{
    return compute_periodic_device(
        execution_space(), charges, positions, count);
}

template <typename Precision>
MonopoleDeviceResult LESMonopoleKokkos<Precision>::compute_periodic_device(
    const execution_space& execution_space,
    const Kokkos::View<double*>& charges,
    const Kokkos::View<double*>& positions,
    const std::size_t count)
{
    if (!cell_ready_)
        throw std::logic_error("LES periodic cell has not been prepared.");
    const std::size_t coordinate_count = checked_node_triplet(count);
    if (charges.extent(0) < count || positions.extent(0) < coordinate_count)
        throw std::invalid_argument("LES device inputs do not cover the active node count.");
    prepare_particle_count_change(count);
    reserve_outputs(count, true);
    ++geometry_updates_;
    if (periodic_algorithm_ == LESPeriodicAlgorithm::particle_mesh) {
        if (!particle_mesh_)
            throw std::logic_error("LES particle-mesh evaluator is not prepared.");
        const MonopoleDeviceResult result{
            energy_, atomic_energies_, forces_, charge_adjoints_, strain_derivative_};
        particle_mesh_->compute(
            charges, positions, count, sigma_, dl_, remove_self_interaction_, result);
        return result;
    }
    const auto vectors = reciprocal_vectors_;
    const auto squared = reciprocal_squared_;
    const auto kernels = reciprocal_kernels_;
    const auto active_indices = reciprocal_active_indices_;
    const auto real = structure_real_;
    const auto imaginary = structure_imaginary_;
    const std::size_t active_count = reciprocal_vector_count_;
    if (active_count > 0) {
        using team_policy = Kokkos::TeamPolicy<
            typename LESMonopoleKokkos<Precision>::execution_space>;
        using team_member = typename team_policy::member_type;
        Kokkos::parallel_for(
            "LES structure factors",
            team_policy(execution_space, active_count, Kokkos::AUTO, 1),
            KOKKOS_LAMBDA(const team_member& team) {
                const std::size_t active_slot = team.league_rank();
                const std::size_t k_index = active_indices(active_slot);
                StructureFactorSum sum;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, count),
                    StructureFactorAtomReduction{
                        charges, positions, vectors, k_index},
                    sum);
                Kokkos::single(Kokkos::PerTeam(team), [=] () {
                    real(k_index) = sum.real;
                    imaginary(k_index) = sum.imaginary;
                });
            });
    }
    const auto atomic = atomic_energies_;
    const auto force = forces_;
    const auto adjoint = charge_adjoints_;
    const auto cell = reciprocal_cell_;
    const bool remove_self = remove_self_interaction_;
    const double self = coulomb_norm_factor/(sigma_*std::pow(two_pi, 1.5));
    using team_policy = Kokkos::TeamPolicy<
        typename LESMonopoleKokkos<Precision>::execution_space>;
    using team_member = typename team_policy::member_type;
    Kokkos::parallel_for(
        "LES periodic atom outputs",
        team_policy(execution_space, count, Kokkos::AUTO, 1),
        KOKKOS_LAMBDA(const team_member& team) {
            const std::size_t atom = team.league_rank();
            const double prefactor = coulomb_norm_factor/cell(9);
            const double qi = charges(atom);
            const double self_sign = remove_self ? -1.0 : 1.0;
            const PeriodicAtomOutputReduction reduction{
                charges, positions, vectors, kernels, active_indices,
                real, imaginary, atom, prefactor};
            PeriodicAtomOutputSum sum;
            if (active_count == 0)
                reduction.init(sum);
            else
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, active_count),
                    reduction, sum);
            Kokkos::single(Kokkos::PerTeam(team), [=] () {
                atomic(atom) = sum.energy+self_sign*self*qi*qi;
                adjoint(atom) = sum.charge_seed+2.0*self_sign*self*qi;
                force(3*atom) = sum.force_x;
                force(3*atom+1) = sum.force_y;
                force(3*atom+2) = sum.force_z;
            });
        });
    Kokkos::parallel_reduce(
        "LES periodic energy",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(execution_space, 0, count),
        KOKKOS_LAMBDA(const std::size_t atom, double& sum) {
            sum += atomic(atom);
        }, energy_);
    const auto strain = strain_derivative_;
    const double sigma = sigma_;
    if (active_count == 0) {
        Kokkos::deep_copy(execution_space, strain, 0.0);
    } else {
        Kokkos::parallel_for(
            "LES analytic reciprocal strain",
            team_policy(execution_space, 9, Kokkos::AUTO, 1),
            KOKKOS_LAMBDA(const team_member& team) {
                const int component = team.league_rank();
                const int row = component/3;
                const int column = component%3;
                const double prefactor = coulomb_norm_factor/cell(9);
                double derivative;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, active_count),
                    PeriodicStrainReduction{
                        vectors, squared, kernels, active_indices,
                        real, imaginary, row, column, prefactor, sigma*sigma},
                    derivative);
                Kokkos::single(Kokkos::PerTeam(team), [=] () {
                    strain(component) = derivative;
                });
            });
    }
    return {
        energy_, atomic_energies_, forces_, charge_adjoints_, strain_derivative_};
}

template <typename Precision>
MonopoleResult LESMonopoleKokkos<Precision>::copy_to_host(
    const MonopoleDeviceResult& result, const std::size_t count) const
{
    MonopoleResult host;
    const auto energy = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), result.energy);
    host.energy = energy();
    const auto atomic = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Kokkos::subview(
            result.atomic_energies, std::make_pair(std::size_t(0),count)));
    const auto forces = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Kokkos::subview(
            result.forces, std::make_pair(std::size_t(0),3*count)));
    const auto adjoints = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Kokkos::subview(
            result.charge_adjoints, std::make_pair(std::size_t(0),count)));
    host.atomic_energies.assign(atomic.data(), atomic.data()+count);
    host.forces.assign(forces.data(), forces.data()+3*count);
    host.charge_adjoints.assign(adjoints.data(), adjoints.data()+count);
    if (result.strain_derivative.extent(0) == 9) {
        const auto strain = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), result.strain_derivative);
        host.strain_derivative.assign(strain.data(), strain.data()+9);
    }
    return host;
}

template <typename Precision>
MonopoleResult LESMonopoleKokkos<Precision>::compute_nonperiodic(
    const std::vector<double>& charges, const std::vector<double>& positions)
{
    validate_inputs(charges, positions);
    Kokkos::View<double*> charge_view("LES host boundary charges", charges.size());
    Kokkos::View<double*> position_view(
        "LES host boundary positions", positions.size());
    auto charge_host = Kokkos::create_mirror_view(charge_view);
    auto position_host = Kokkos::create_mirror_view(position_view);
    std::copy(charges.begin(), charges.end(), charge_host.data());
    std::copy(positions.begin(), positions.end(), position_host.data());
    Kokkos::deep_copy(charge_view, charge_host);
    Kokkos::deep_copy(position_view, position_host);
    return copy_to_host(
        compute_nonperiodic_device(
            charge_view, position_view, charges.size()),
        charges.size());
}

template <typename Precision>
MonopoleResult LESMonopoleKokkos<Precision>::compute_periodic(
    const std::vector<double>& charges, const std::vector<double>& positions,
    const std::vector<double>& cell)
{
    validate_inputs(charges, positions);
    const execution_space execution_space;
    prepare_periodic(execution_space, cell);
    Kokkos::View<double*> charge_view("LES host boundary charges", charges.size());
    Kokkos::View<double*> position_view(
        "LES host boundary positions", positions.size());
    auto charge_host = Kokkos::create_mirror_view(charge_view);
    auto position_host = Kokkos::create_mirror_view(position_view);
    std::copy(charges.begin(), charges.end(), charge_host.data());
    std::copy(positions.begin(), positions.end(), position_host.data());
    Kokkos::deep_copy(charge_view, charge_host);
    Kokkos::deep_copy(position_view, position_host);
    return copy_to_host(
        compute_periodic_device(
            execution_space, charge_view, position_view, charges.size()),
        charges.size());
}

template <typename Precision>
MonopoleDiagnostics LESMonopoleKokkos<Precision>::diagnostics() const
{
    const auto checked_bytes = [] (
        const std::size_t elements, const std::size_t element_size) {
        return checked_reciprocal_multiply(
            elements, element_size, "diagnostic byte count overflows size_t");
    };
    const auto checked_sum = [] (const std::size_t left, const std::size_t right) {
        return checked_reciprocal_add(
            left, right, "diagnostic byte sum overflows size_t");
    };
    const std::size_t energy_elements =
        energy_.data() == nullptr ? std::size_t(0) : std::size_t(1);
    const std::size_t logical_output_elements = checked_reciprocal_add(
        checked_reciprocal_multiply(
            active_nodes_, 5, "logical output extent overflows size_t"),
        strain_derivative_.extent(0),
        "logical strain extent overflows size_t");
    const std::size_t logical_reciprocal_elements = checked_reciprocal_add(
        cell_ready_ ? std::size_t(10) : std::size_t(0),
        checked_reciprocal_multiply(
            reciprocal_candidates_, 7,
            "logical reciprocal extent overflows size_t"),
        "logical reciprocal-cell extent overflows size_t");
    const std::size_t logical_double_elements = checked_reciprocal_add(
        energy_elements,
        checked_reciprocal_add(
            logical_output_elements, logical_reciprocal_elements,
            "logical double extent overflows size_t"),
        "logical energy extent overflows size_t");
    const std::size_t logical_extent_bytes = checked_sum(
        checked_sum(
            checked_bytes(logical_double_elements, sizeof(double)),
            checked_bytes(reciprocal_candidates_, sizeof(unsigned char))),
        checked_bytes(reciprocal_vector_count_, sizeof(std::size_t)));

    std::size_t capacity_double_elements = energy_elements;
    for (const std::size_t extent : {
             atomic_energies_.extent(0), forces_.extent(0),
             charge_adjoints_.extent(0), strain_derivative_.extent(0),
             checked_reciprocal_multiply(
                 reciprocal_vectors_.extent(0), 3,
                 "vector capacity extent overflows size_t"),
             reciprocal_squared_.extent(0), reciprocal_kernels_.extent(0),
             structure_real_.extent(0), structure_imaginary_.extent(0),
             reciprocal_cell_.extent(0)}) {
        capacity_double_elements = checked_reciprocal_add(
            capacity_double_elements, extent,
            "capacity double extent overflows size_t");
    }
    const std::size_t allocated_capacity_bytes = checked_sum(
        checked_sum(
            checked_bytes(capacity_double_elements, sizeof(double)),
            checked_bytes(reciprocal_active_.extent(0), sizeof(unsigned char))),
        checked_bytes(
            reciprocal_active_indices_.extent(0), sizeof(std::size_t)));

    MonopoleDiagnostics result;
    result.reciprocal_candidate_count = reciprocal_candidates_;
    result.reciprocal_vector_count = reciprocal_vector_count_;
    result.logical_extent_bytes = logical_extent_bytes;
    result.allocated_capacity_bytes = allocated_capacity_bytes;
    result.active_index_capacity_bytes = checked_bytes(
        reciprocal_active_indices_.extent(0), sizeof(std::size_t));
    result.index_pack_count = index_pack_count_;
    result.table_rebuilds = table_rebuilds_;
    result.geometry_updates = geometry_updates_;
    result.active_bytes = logical_extent_bytes;
    result.capacity_bytes = allocated_capacity_bytes;
    result.periodic_algorithm = periodic_algorithm_;
    if (particle_mesh_) {
        const auto particle_mesh = particle_mesh_->diagnostics();
        result.particle_mesh_allocation_generation =
            particle_mesh.allocation_generation;
        result.particle_mesh_reservations = particle_mesh.reservations;
        result.particle_mesh_allocations = particle_mesh.allocations;
        result.particle_mesh_forward_plan_builds = particle_mesh.forward_plan_builds;
        result.particle_mesh_forward_plan_reuses = particle_mesh.forward_plan_reuses;
        result.particle_mesh_inverse_plan_builds = particle_mesh.inverse_plan_builds;
        result.particle_mesh_inverse_plan_reuses = particle_mesh.inverse_plan_reuses;
        result.particle_mesh_reciprocal_preparations =
            particle_mesh.reciprocal_preparations;
        result.particle_mesh_stencil_preparations =
            particle_mesh.stencil_preparations;
        result.particle_mesh_particle_count_fences =
            particle_mesh.particle_count_fences;
        result.particle_mesh_execution_instance_id =
            particle_mesh.execution_instance_id;
    }
    return result;
}

template <typename Precision>
std::vector<std::size_t>
LESMonopoleKokkos<Precision>::active_reciprocal_indices_for_testing() const
{
    const auto active_indices = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Kokkos::subview(
            reciprocal_active_indices_,
            std::make_pair(std::size_t(0), reciprocal_vector_count_)));
    std::vector<std::size_t> indices(reciprocal_vector_count_);
    for (std::size_t index=0; index<reciprocal_vector_count_; ++index)
        indices[index] = active_indices(index);
    return indices;
}

template class LESMonopoleKokkos<float>;
template class LESMonopoleKokkos<double>;

} // namespace symmetrix::les
