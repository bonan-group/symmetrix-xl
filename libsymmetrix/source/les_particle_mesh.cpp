#include "les_particle_mesh.hpp"

#include "les_monopole.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#ifndef SYMMETRIX_PARTICLE_MESH_AVAILABLE
#define SYMMETRIX_PARTICLE_MESH_AVAILABLE 0
#endif
#ifndef SYMMETRIX_PARTICLE_MESH_REASON
#define SYMMETRIX_PARTICLE_MESH_REASON "unavailable"
#endif

#if SYMMETRIX_PARTICLE_MESH_AVAILABLE
#include "particle_mesh_strain.hpp"
#include "particle_mesh_transport.hpp"
#endif

namespace symmetrix::les {
namespace {

constexpr double coulomb_norm_factor = 90.4756;
constexpr double two_pi = 2.0 * std::numbers::pi;
// Adequacy validation is a serial host enumeration backed by an ordered set.
// Above this count, enumeration time and the active-tuple storage can consume
// multiple gigabytes even before an FFT workspace is allocated.
constexpr std::size_t maximum_miller_enumeration_candidates = 100'000'000;

[[noreturn]] void throw_unavailable()
{
    throw std::runtime_error(
        "LES particle-mesh periodic evaluation is unavailable ("
        SYMMETRIX_PARTICLE_MESH_REASON ").");
}

#if SYMMETRIX_PARTICLE_MESH_AVAILABLE

void validate_mesh_adequacy(
    const std::vector<double>& cell,
    const std::array<double, 9>& inverse,
    const std::array<std::int64_t, 3>& mesh,
    const double reciprocal_spacing)
{
    constexpr double maximum_axis_count =
        static_cast<double>((std::numeric_limits<int>::max() - 1) / 2);
    std::array<int, 3> count{};
    std::array<std::size_t, 3> width{};
    for (int axis = 0; axis < 3; ++axis) {
        const double norm = std::hypot(
            cell[3 * axis], cell[3 * axis + 1], cell[3 * axis + 2]);
        const double ratio = norm / reciprocal_spacing;
        if (!std::isfinite(ratio)) {
            throw std::invalid_argument(
                "LES reciprocal search is non-finite on axis "
                + std::to_string(axis) + ".");
        }
        const double truncated = std::max(1.0, std::floor(ratio));
        if (truncated > maximum_axis_count) {
            throw std::invalid_argument(
                "LES reciprocal search exceeds the supported integer range on axis "
                + std::to_string(axis) + ".");
        }
        count[axis] = static_cast<int>(truncated);
        width[axis] = 2 * static_cast<std::size_t>(count[axis]) + 1;
    }
    std::size_t candidate_count = 1;
    for (const std::size_t axis_width : width) {
        if (candidate_count > std::numeric_limits<std::size_t>::max() / axis_width) {
            throw std::invalid_argument(
                "LES reciprocal search candidate count overflows size_t.");
        }
        candidate_count *= axis_width;
    }
    if (candidate_count > maximum_miller_enumeration_candidates) {
        throw std::invalid_argument(
            "LES reciprocal search exceeds the supported enumeration limit of "
            + std::to_string(maximum_miller_enumeration_candidates)
            + " candidates.");
    }
    const double cutoff_squared = (two_pi / reciprocal_spacing)
        * (two_pi / reciprocal_spacing);
    std::set<std::tuple<std::int64_t, std::int64_t, std::int64_t>> representatives;
    bool omitted_active_vector = false;
    for (int n0 = -count[0]; n0 <= count[0]; ++n0) {
        for (int n1 = -count[1]; n1 <= count[1]; ++n1) {
            for (int n2 = -count[2]; n2 <= count[2]; ++n2) {
                const int n[3] = {n0, n1, n2};
                int first = 0;
                while (first < 3 && n[first] == 0) {
                    ++first;
                }
                if (first == 3 || n[first] < 0) {
                    continue;
                }
                double squared = 0.0;
                for (int component = 0; component < 3; ++component) {
                    double wave = 0.0;
                    for (int axis = 0; axis < 3; ++axis) {
                        wave += static_cast<double>(n[axis]) * two_pi
                            * inverse[static_cast<std::size_t>(3 * component + axis)];
                    }
                    squared += wave * wave;
                }
                if (!(squared > 0.0) || squared > cutoff_squared) {
                    continue;
                }
                std::array<std::int64_t, 3> canonical = {n0, n1, n2};
                if (canonical[2] < 0) {
                    for (auto& value : canonical) {
                        value = -value;
                    }
                }
                for (int axis = 0; axis < 3; ++axis) {
                    const std::int64_t magnitude = std::abs(canonical[axis]);
                    const std::int64_t half = mesh[axis] / 2;
                    if (mesh[axis] % 2 == 0 && magnitude == half) {
                        throw std::invalid_argument(
                            "LES particle mesh places an active reciprocal vector on "
                            "an ambiguous Nyquist representative on axis "
                            + std::to_string(axis) + ".");
                    }
                    if (magnitude > half) {
                        omitted_active_vector = true;
                    }
                }
                const auto representative = std::make_tuple(
                    (canonical[0] % mesh[0] + mesh[0]) % mesh[0],
                    (canonical[1] % mesh[1] + mesh[1]) % mesh[1],
                    canonical[2]);
                if (!representatives.insert(representative).second) {
                    throw std::invalid_argument(
                        "LES particle mesh aliases active reciprocal vectors.");
                }
            }
        }
    }
    if (omitted_active_vector) {
        throw std::invalid_argument(
            "LES particle mesh omits an active reciprocal vector.");
    }
}

#endif

}  // namespace

void require_les_particle_mesh_available()
{
#if !SYMMETRIX_PARTICLE_MESH_AVAILABLE
    throw_unavailable();
#endif
}

#if SYMMETRIX_PARTICLE_MESH_AVAILABLE

struct LESParticleMesh::Impl {
    using Workspace = particle_mesh::ParticleMeshWorkspace<double>;
    using memory_space = typename execution_space::memory_space;
    using complex_type = Kokkos::complex<double>;
    static constexpr std::uint64_t auxiliary_allocations_per_reserve =
        3 + 3 + 3;  // stencil, reciprocal-grid metadata, and LES cotangents

    Impl(
        execution_space instance,
        const std::int64_t nx_value,
        const std::int64_t ny_value,
        const std::int64_t nz_value,
        const std::int64_t order_value)
        : execution(std::move(instance)),
          workspace(particle_mesh::MeshLayout::right, 1, execution),
          nx(nx_value),
          ny(ny_value),
          nz(nz_value),
          order(order_value)
    {
        if (nx <= 0 || ny <= 0 || nz <= 0) {
            throw std::invalid_argument("LES particle-mesh extents must be positive.");
        }
        if (order < 2 || order > particle_mesh::bspline::max_order) {
            throw std::invalid_argument(
                "LES particle-mesh spline order must be between 2 and 6.");
        }
        if (nx < order || ny < order || nz < order) {
            throw std::invalid_argument(
                "LES particle-mesh extents must each be at least the spline order.");
        }
    }

    void prepare_cell(
        const std::vector<double>& values,
        const std::array<double, 9>& inverse,
        const double determinant,
        const double reciprocal_spacing)
    {
        validate_mesh_adequacy(
            values, inverse, {nx, ny, nz}, reciprocal_spacing);
        cell_inverse = inverse;
        volume = determinant;
        cell_ready = true;
        reciprocal_ready = false;
        stencil_ready = false;
    }

    void reserve(const std::int64_t particles)
    {
        if (particles == active_particles) {
            prefenced_particle_count = -1;
            return;
        }
        prepare_particle_count_change(particles);
        try {
            workspace.reserve(nx, ny, nz, particles, 1, order);
            workspace.prepare_transforms();
            const auto shape = workspace.storage().shape();
            if (!shape) {
                throw std::logic_error("LES particle-mesh workspace was not reserved.");
            }
            const std::int64_t bins = shape->spectrum_point_count();
            cotangent = Kokkos::View<complex_type*>(
                "LES particle-mesh physical cotangent", bins);
            bin_energy = Kokkos::View<double*>("LES particle-mesh bin energy", bins);
            bin_radial_derivative = Kokkos::View<double*>(
                "LES particle-mesh bin radial derivative", bins);
            ++workspace_reservations;
            ++workspace_allocation_generation;
            workspace_auxiliary_allocations += auxiliary_allocations_per_reserve;
            active_particles = particles;
            reciprocal_ready = false;
            stencil_ready = false;
            prefenced_particle_count = -1;
            owner_work_may_be_in_flight = false;
        } catch (...) {
            // The shared workspace commits its count before FFT-plan creation.
            // Treat a failed replacement as unknown so retry and revert both
            // re-enter reserve instead of trusting the adapter's old count.
            active_particles = -1;
            prefenced_particle_count = -1;
            owner_work_may_be_in_flight = true;
            reciprocal_ready = false;
            stencil_ready = false;
            throw;
        }
    }

    void prepare_particle_count_change(const std::int64_t particles)
    {
        if (!owner_work_may_be_in_flight && particles == active_particles) {
            // New work for the active count invalidates a fence retained from
            // an abandoned preparation for another count.
            prefenced_particle_count = -1;
            return;
        }
        if (!owner_work_may_be_in_flight
            && particles == prefenced_particle_count) {
            return;
        }
        if (active_particles >= 0 || owner_work_may_be_in_flight
            || prefenced_particle_count >= 0) {
            execution.fence("Replace LES particle-mesh particle-count workspace");
            ++particle_count_fences;
            owner_work_may_be_in_flight = false;
        }
        prefenced_particle_count = particles;
    }

    void cancel_particle_count_change()
    {
        if (active_particles < 0 && prefenced_particle_count >= 0) {
            // A caller can enqueue geometry or nonperiodic output work after
            // preflight without ever reaching reserve. With no committed
            // particle count, retain an explicit unknown-work state so the
            // next replacement still fences the owner instance.
            owner_work_may_be_in_flight = true;
        }
        prefenced_particle_count = -1;
    }

    void prepare_positions(
        const std::span<const double> host_positions,
        const Kokkos::View<double*>& device_positions,
        const std::size_t count)
    {
        if (!cell_ready) {
            throw std::logic_error(
                "LES particle-mesh periodic cell has not been prepared.");
        }
        if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("LES particle count exceeds the particle-mesh range.");
        }
        const std::int64_t particles = static_cast<std::int64_t>(count);
        reserve(particles);
        prepare_reciprocal_grid();
        const auto active_positions = Kokkos::subview(
            device_positions, std::make_pair(std::size_t(0), 3 * count));
        const auto stencil_path = workspace.prepare_stencil(
            host_positions, active_positions);
        (void)stencil_path;
        ++stencil_preparations;
        stencil_ready = true;
    }

    void prepare_reciprocal_grid()
    {
        if (reciprocal_ready) {
            return;
        }
        workspace.set_cell(cell_inverse, volume);
        workspace.prepare_reciprocal_grid();
        ++reciprocal_preparations;
        reciprocal_ready = true;
    }

    execution_space execution;
    Workspace workspace;
    std::int64_t nx;
    std::int64_t ny;
    std::int64_t nz;
    std::int64_t order;
    std::int64_t active_particles = -1;
    std::int64_t prefenced_particle_count = -1;
    bool owner_work_may_be_in_flight = false;
    std::array<double, 9> cell_inverse{};
    double volume = 0.0;
    bool cell_ready = false;
    bool reciprocal_ready = false;
    bool stencil_ready = false;
    std::uint64_t reciprocal_preparations = 0;
    std::uint64_t stencil_preparations = 0;
    std::uint64_t workspace_allocation_generation = 0;
    std::uint64_t workspace_reservations = 0;
    std::uint64_t workspace_auxiliary_allocations = 0;
    std::uint64_t particle_count_fences = 0;
    Kokkos::View<complex_type*> cotangent;
    Kokkos::View<double*> bin_energy;
    Kokkos::View<double*> bin_radial_derivative;
};

LESParticleMesh::LESParticleMesh(
    execution_space execution_space,
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t spline_order)
    : impl_(std::make_unique<Impl>(
          std::move(execution_space), nx, ny, nz, spline_order))
{
}

LESParticleMesh::~LESParticleMesh() = default;

void LESParticleMesh::prepare_cell(
    const std::vector<double>& cell,
    const std::array<double, 9>& cell_inverse,
    const double volume,
    const double reciprocal_spacing)
{
    impl_->prepare_cell(cell, cell_inverse, volume, reciprocal_spacing);
}

void LESParticleMesh::prepare_particle_count_change(const std::size_t count)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("LES particle count exceeds the particle-mesh range.");
    }
    impl_->prepare_particle_count_change(static_cast<std::int64_t>(count));
}

void LESParticleMesh::cancel_particle_count_change()
{
    impl_->cancel_particle_count_change();
}

void LESParticleMesh::prepare_positions(
    const std::span<const double> host_positions,
    const Kokkos::View<double*>& device_positions,
    const std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / 3
        || host_positions.size() != 3 * count
        || device_positions.extent(0) < 3 * count) {
        throw std::invalid_argument(
            "LES particle-mesh positions do not cover the active particle count.");
    }
    impl_->prepare_positions(host_positions, device_positions, count);
}

void LESParticleMesh::compute(
    const Kokkos::View<double*>& charges,
    const Kokkos::View<double*>& positions,
    const std::size_t count,
    const double sigma,
    const double reciprocal_spacing,
    const bool remove_self_interaction,
    const MonopoleDeviceResult& output)
{
    if (!impl_->cell_ready) {
        throw std::logic_error("LES particle-mesh periodic cell has not been prepared.");
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("LES particle count exceeds the particle-mesh range.");
    }
    const std::int64_t particles = static_cast<std::int64_t>(count);
    impl_->reserve(particles);
    auto& workspace = impl_->workspace;
    if (!impl_->stencil_ready) {
        impl_->prepare_reciprocal_grid();
        Kokkos::View<double*, Kokkos::HostSpace> host_positions(
            "LES particle-mesh host positions", 3 * count);
        const auto active_positions = Kokkos::subview(
            positions, std::make_pair(std::size_t(0), 3 * count));
        Kokkos::deep_copy(impl_->execution, host_positions, active_positions);
        impl_->execution.fence("LES particle-mesh safe stencil positions");
        const auto stencil_path = workspace.prepare_stencil(
            std::span<const double>(host_positions.data(), host_positions.extent(0)),
            active_positions);
        (void)stencil_path;
        ++impl_->stencil_preparations;
    }
    impl_->stencil_ready = false;

    const typename Impl::Workspace::weights_view weights(
        charges.data(), particles, 1);
    workspace.spread(weights);
    workspace.transform_forward();
    workspace.deconvolve_amplitudes();

    const auto spectrum = workspace.spectrum();
    const auto grid = workspace.reciprocal_grid();
    const auto cotangent = impl_->cotangent;
    const auto bin_energy = impl_->bin_energy;
    const auto bin_radial = impl_->bin_radial_derivative;
    const double prefactor = coulomb_norm_factor / impl_->volume;
    const double cutoff_squared = (two_pi / reciprocal_spacing)
        * (two_pi / reciprocal_spacing);
    const double sigma_squared = sigma * sigma;
    using complex_type = Impl::complex_type;
    Kokkos::parallel_for(
        "LES particle-mesh spectral contraction",
        Kokkos::RangePolicy<execution_space>(impl_->execution, 0, grid.bins),
        KOKKOS_LAMBDA(const std::int64_t bin) {
            const double norm = grid.k_norm(bin);
            const double squared = norm * norm;
            const bool active = squared > 0.0 && squared <= cutoff_squared;
            const double kernel = active
                ? prefactor * Kokkos::exp(-0.5 * sigma_squared * squared) / squared
                : 0.0;
            const complex_type amplitude = spectrum.data()[bin];
            const double energy = static_cast<double>(grid.hermitian_weight(bin))
                * kernel
                * (amplitude.real() * amplitude.real()
                    + amplitude.imag() * amplitude.imag());
            bin_energy(bin) = energy;
            bin_radial(bin) = active
                ? energy * (-sigma_squared * norm - 2.0 / norm)
                : 0.0;
            cotangent(bin) = complex_type(
                2.0 * kernel * amplitude.real(),
                2.0 * kernel * amplitude.imag());
        });
    workspace.load_physical_cotangent(
        typename Impl::Workspace::complex_input_view(cotangent.data(), grid.bins));
    workspace.deconvolve_cotangent();
    workspace.transform_inverse();

    const typename Impl::Workspace::weight_adjoints_view adjoints(
        output.charge_adjoints.data(), particles, 1);
    const typename Impl::Workspace::position_forces_view forces(
        output.forces.data(), particles, 3);
    workspace.gather(
        weights, typename Impl::Workspace::groups_view(), 0, adjoints, forces);

    const auto atomic = output.atomic_energies;
    const auto charge_adjoints = output.charge_adjoints;
    const double self = coulomb_norm_factor / (sigma * std::pow(two_pi, 1.5));
    const double self_sign = remove_self_interaction ? -1.0 : 1.0;
    Kokkos::parallel_for(
        "LES particle-mesh atomic outputs",
        Kokkos::RangePolicy<execution_space>(impl_->execution, 0, particles),
        KOKKOS_LAMBDA(const std::int64_t atom) {
            const double charge = charges(atom);
            const double reciprocal_adjoint = charge_adjoints(atom);
            atomic(atom) = 0.5 * charge * reciprocal_adjoint
                + self_sign * self * charge * charge;
            charge_adjoints(atom) = reciprocal_adjoint
                + 2.0 * self_sign * self * charge;
        });
    Kokkos::parallel_reduce(
        "LES particle-mesh energy",
        Kokkos::RangePolicy<execution_space>(impl_->execution, 0, particles),
        KOKKOS_LAMBDA(const std::int64_t atom, double& sum) {
            sum += atomic(atom);
        },
        output.energy);
    particle_mesh::reduce_radial_strain(
        impl_->execution,
        grid,
        bin_energy,
        bin_radial,
        output.strain_derivative);
}

LESParticleMeshDiagnostics LESParticleMesh::diagnostics() const
{
    const auto storage = impl_->workspace.storage().usage();
    const auto forward = impl_->workspace.plans().usage(
        particle_mesh::FftPlanRole::forward_r2c);
    const auto inverse = impl_->workspace.plans().usage(
        particle_mesh::FftPlanRole::inverse_c2r);
    return {
        impl_->workspace_allocation_generation,
        impl_->workspace_reservations,
        storage.allocations + impl_->workspace_auxiliary_allocations,
        forward.builds,
        forward.reuses,
        inverse.builds,
        inverse.reuses,
        impl_->reciprocal_preparations,
        impl_->stencil_preparations,
        impl_->particle_count_fences,
        impl_->execution.impl_instance_id(),
    };
}

#else

struct LESParticleMesh::Impl {};

LESParticleMesh::LESParticleMesh(
    execution_space,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t)
{
    throw_unavailable();
}

LESParticleMesh::~LESParticleMesh() = default;

void LESParticleMesh::prepare_cell(
    const std::vector<double>&,
    const std::array<double, 9>&,
    double,
    double)
{
    throw_unavailable();
}

void LESParticleMesh::prepare_particle_count_change(std::size_t)
{
    throw_unavailable();
}

void LESParticleMesh::cancel_particle_count_change()
{
    throw_unavailable();
}

void LESParticleMesh::prepare_positions(
    std::span<const double>,
    const Kokkos::View<double*>&,
    std::size_t)
{
    throw_unavailable();
}

void LESParticleMesh::compute(
    const Kokkos::View<double*>&,
    const Kokkos::View<double*>&,
    std::size_t,
    double,
    double,
    bool,
    const MonopoleDeviceResult&)
{
    throw_unavailable();
}

LESParticleMeshDiagnostics LESParticleMesh::diagnostics() const
{
    throw_unavailable();
}

#endif

}  // namespace symmetrix::les
