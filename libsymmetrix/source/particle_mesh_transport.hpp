#ifndef SYMMETRIX_PARTICLE_MESH_TRANSPORT_HPP
#define SYMMETRIX_PARTICLE_MESH_TRANSPORT_HPP

// Physics-neutral particle-to-mesh assignment and forward FFT transport.

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "particle_mesh_geometry.hpp"
#include "particle_mesh_stencil.hpp"
#include "particle_mesh_storage.hpp"
#include "tools_kokkos.hpp"

namespace symmetrix::particle_mesh {

template <typename Float>
class ParticleMeshWorkspace {
public:
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using complex_type = Kokkos::complex<Float>;
    using weights_view = Kokkos::View<
        const Float**, Kokkos::LayoutRight, memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using groups_view = Kokkos::View<
        const std::int32_t*, memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using complex_input_view = Kokkos::View<
        const complex_type*, memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using weight_adjoints_view = Kokkos::View<
        Float**, Kokkos::LayoutRight, memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using position_forces_view = Kokkos::View<
        Float**, Kokkos::LayoutRight, memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    ParticleMeshWorkspace(
        MeshLayout layout, std::int64_t channel_bound, execution_space instance)
        : execution_space_(std::move(instance)),
          storage_(layout, channel_bound),
          plans_(execution_space_)
    {
    }

    ParticleMeshWorkspace(const ParticleMeshWorkspace&) = delete;
    ParticleMeshWorkspace& operator=(const ParticleMeshWorkspace&) = delete;
    ParticleMeshWorkspace(ParticleMeshWorkspace&&) = delete;
    ParticleMeshWorkspace& operator=(ParticleMeshWorkspace&&) = delete;

    struct Reservation {
        Kokkos::View<std::int32_t*> stencil_base;
        Kokkos::View<Float*> stencil_weight;
        Kokkos::View<Float*> stencil_slope;
        Kokkos::View<double*> k_norm;
        Kokkos::View<Float*> deconvolution;
        Kokkos::View<Float*> hermitian;
        typename MeshStorage<Float>::Reservation storage;
        MeshShape shape;
        std::int64_t particles = 0;
        std::int64_t lanes = 0;
        std::int64_t spline_order = 0;
    };

    void reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t particles,
        std::int64_t lanes,
        std::int64_t spline_order);
    // Validate every extent and flattened range without allocating. Composite
    // owners call this before preparing their own potentially large buffers.
    void validate_reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t particles,
        std::int64_t lanes,
        std::int64_t spline_order) const;
    [[nodiscard]] Reservation plan_reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t particles,
        std::int64_t lanes,
        std::int64_t spline_order) const;
    void commit(Reservation&& reservation);

    void set_cell(const std::array<double, 9>& cell_inverse, double volume)
    {
        if (!cell_set_ || cell_inverse_ != cell_inverse || volume_ != volume) {
            reciprocal_grid_prepared_ = false;
        }
        cell_inverse_ = cell_inverse;
        volume_ = volume;
        grid_.cell_inverse = pack_matrix(cell_inverse);
        grid_.volume = volume;
        cell_set_ = true;
    }

    template <typename DevicePositionsView>
    [[nodiscard]] StencilPreparationPath prepare_stencil(
        std::span<const double> host_positions,
        const DevicePositionsView& device_positions)
    {
        static_assert(Kokkos::is_view<DevicePositionsView>::value);
        static_assert(DevicePositionsView::rank == 1);
        static_assert(std::is_same_v<
                      typename DevicePositionsView::non_const_value_type,
                      double>);
        static_assert(Kokkos::SpaceAccessibility<
                      execution_space,
                      typename DevicePositionsView::memory_space>::accessible);
        require_reserved("the assignment stencil");
        if (!cell_set_) {
            throw std::runtime_error(
                "Particle-mesh assignment requires a cell; call set_cell first.");
        }
        const std::int64_t coordinates = detail::checked_product(
            particles_, 3, "particle-mesh position coordinates");
        if (host_positions.size() != static_cast<std::size_t>(coordinates)
            || static_cast<std::int64_t>(device_positions.extent(0)) != coordinates) {
            throw std::invalid_argument(
                "Particle-mesh host and device positions must both have shape "
                "[particles, 3].");
        }
        return stencil_.prepare_safe(
            execution_space_, particles_, spline_order_, cell_inverse_, host_positions,
            device_positions, *storage_.shape());
    }

    template <typename DevicePositionsView>
    void prepare_stencil_grid_coordinates_device(
        const DevicePositionsView& device_grid_coordinates)
    {
        static_assert(Kokkos::is_view<DevicePositionsView>::value);
        static_assert(DevicePositionsView::rank == 1);
        static_assert(std::is_same_v<
                      typename DevicePositionsView::non_const_value_type,
                      double>);
        static_assert(Kokkos::SpaceAccessibility<
                      execution_space,
                      typename DevicePositionsView::memory_space>::accessible);
        require_reserved("the device grid-coordinate assignment stencil");
        if (!cell_set_) {
            throw std::runtime_error(
                "Particle-mesh assignment requires a cell; call set_cell first.");
        }
        const std::int64_t coordinates = detail::checked_product(
            particles_, 3, "particle-mesh device position coordinates");
        if (device_grid_coordinates.extent(0)
                > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
            || static_cast<std::int64_t>(device_grid_coordinates.extent(0))
                != coordinates) {
            throw std::invalid_argument(
                "Particle-mesh device grid coordinates must have shape "
                "[particles, 3].");
        }
        stencil_.prepare_device_grid_coordinates(execution_space_, particles_,
            spline_order_, device_grid_coordinates, *storage_.shape());
    }

    void prepare_host_stencil_baseline(std::span<const double> host_positions)
    {
        require_reserved("the host stencil diagnostic");
        if (!cell_set_) {
            throw std::runtime_error(
                "Particle-mesh assignment requires a cell; call set_cell first.");
        }
        stencil_.prepare_host_diagnostic_baseline(
            execution_space_, particles_, spline_order_, cell_inverse_, host_positions,
            *storage_.shape());
    }

    void prepare_reciprocal_grid()
    {
        require_reserved("the reciprocal grid");
        if (!cell_set_) {
            throw std::runtime_error(
                "Particle-mesh reciprocal geometry requires a cell; call set_cell first.");
        }
        if (reciprocal_grid_prepared_) return;
        particle_mesh::prepare_reciprocal_grid(execution_space_, grid_, spline_order_);
        reciprocal_grid_prepared_ = true;
    }

    void spread(
        const weights_view& weights,
        const groups_view& groups = groups_view(),
        std::int32_t selected_group = 0);

    void transform_forward()
    {
        require_reserved("the forward transform");
        plans_.acquire(storage_, FftPlanRole::forward_r2c);
        plans_.execute(storage_, FftPlanRole::forward_r2c);
    }

    void prepare_transforms()
    {
        require_reserved("transform preparation");
        plans_.acquire(storage_, FftPlanRole::forward_r2c);
        plans_.acquire(storage_, FftPlanRole::inverse_c2r);
    }

    void transform_inverse()
    {
        require_reserved("the inverse transform");
        plans_.acquire(storage_, FftPlanRole::inverse_c2r);
        plans_.execute(storage_, FftPlanRole::inverse_c2r);
    }

    void deconvolve_amplitudes();

    // The input is exactly dE/dA for A = S/B. Hermitian multiplicity belongs
    // to the adapter's energy reduction and must not be folded into this
    // physical cotangent: the real inverse transform supplies the conjugate
    // half of the spectrum itself.
    void load_physical_cotangent(
        const complex_input_view& deconvolved_amplitude_cotangent);
    void deconvolve_cotangent();
    // Empty `groups` defines a complete gather and clears both outputs before
    // writing them. A nonempty `groups` view accumulates only `selected_group`
    // into caller-cleared outputs, allowing adapters to compose grouped passes.
    void gather(
        const weights_view& weights,
        const groups_view& groups,
        std::int32_t selected_group,
        const weight_adjoints_view& weight_adjoints,
        const position_forces_view& position_forces);

    [[nodiscard]] typename MeshStorage<Float>::real_active_view real_mesh() const
    {
        return storage_.real_field();
    }
    [[nodiscard]] typename MeshStorage<Float>::spectrum_active_view spectrum() const
    {
        return storage_.spectrum();
    }
    [[nodiscard]] const ReciprocalGridDeviceView<Float>& reciprocal_grid() const noexcept
    {
        return grid_;
    }

    [[nodiscard]] const execution_space& execution_instance() const noexcept
    {
        return execution_space_;
    }
    [[nodiscard]] MeshStorage<Float>& storage() noexcept { return storage_; }
    [[nodiscard]] const MeshStorage<Float>& storage() const noexcept { return storage_; }
    [[nodiscard]] FftPlanDomain<Float>& plans() noexcept { return plans_; }
    [[nodiscard]] const FftPlanDomain<Float>& plans() const noexcept { return plans_; }
    [[nodiscard]] ParticleStencil<Float>& stencil() noexcept { return stencil_; }
    [[nodiscard]] const ParticleStencil<Float>& stencil() const noexcept { return stencil_; }
    [[nodiscard]] ReciprocalGridDeviceView<Float>& reciprocal_grid() noexcept { return grid_; }

    template <typename FlatView>
    [[nodiscard]] weights_view checked_weights_view(
        const FlatView& flat,
        std::int64_t particles,
        std::int64_t lanes,
        std::string_view what) const
    {
        const std::int64_t required =
            detail::checked_product(particles, lanes, what);
        if (required != static_cast<std::int64_t>(flat.extent(0))) {
            throw std::invalid_argument(
                std::string(what) + " must hold exactly " + std::to_string(required)
                + " elements, got " + std::to_string(flat.extent(0)) + ".");
        }
        return weights_view(flat.data(), particles, lanes);
    }

    template <typename FlatView>
    [[nodiscard]] groups_view checked_groups_view(
        const FlatView& flat, std::int64_t particles, std::string_view what) const
    {
        const std::int64_t checked_particles = detail::checked_extent(particles, what);
        if (checked_particles != static_cast<std::int64_t>(flat.extent(0))) {
            throw std::invalid_argument(
                std::string(what) + " must hold exactly " + std::to_string(particles)
                + " elements, got " + std::to_string(flat.extent(0)) + ".");
        }
        return groups_view(flat.data(), particles);
    }

private:
    void require_reserved(std::string_view what) const
    {
        if (!storage_.shape()) {
            throw std::runtime_error(
                "Particle-mesh " + std::string(what)
                + " requires a reserved mesh; call reserve first.");
        }
    }

    // Declared first so all Kokkos allocations are gone before the live-object
    // guard is released.
    KokkosLiveObjectGuard kokkos_live_object_guard_;
    execution_space execution_space_;
    MeshStorage<Float> storage_;
    FftPlanDomain<Float> plans_;
    ParticleStencil<Float> stencil_;
    ReciprocalGridDeviceView<Float> grid_;
    std::int64_t particles_ = 0;
    std::int64_t lanes_ = 0;
    std::int64_t spline_order_ = 0;
    std::array<double, 9> cell_inverse_{};
    double volume_ = 0.0;
    bool cell_set_ = false;
    bool reciprocal_grid_prepared_ = false;
};

template <typename Float>
void ParticleMeshWorkspace<Float>::reserve(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t particles,
    const std::int64_t lanes,
    const std::int64_t spline_order)
{
    commit(plan_reserve(nx, ny, nz, particles, lanes, spline_order));
}

template <typename Float>
void ParticleMeshWorkspace<Float>::validate_reserve(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t particles,
    const std::int64_t lanes,
    const std::int64_t spline_order) const
{
    if (particles < 0) {
        throw std::invalid_argument(
            "Particle-mesh particle count must not be negative, got "
            + std::to_string(particles) + ".");
    }
    if (spline_order < 2 || spline_order > bspline::max_order) {
        throw std::invalid_argument(
            "Particle-mesh spline order must be between 2 and 6, got "
            + std::to_string(spline_order) + ".");
    }
    const MeshShape requested(nx, ny, nz, lanes, MeshLayout::right);
    storage_.validate_reserve(nx, ny, nz, lanes);
    const std::int64_t offsets =
        detail::checked_product(3, particles, "particle-mesh stencil offsets");
    const std::int64_t stencil_elements = detail::checked_product(
        offsets, spline_order, "particle-mesh stencil values");
    const std::int64_t bins = requested.spectrum_point_count();
    static_cast<void>(detail::checked_flat_index(offsets, "stencil_base"));
    static_cast<void>(detail::checked_flat_index(stencil_elements, "stencil_weight"));
    static_cast<void>(detail::checked_flat_index(bins, "k_norm"));
}

template <typename Float>
auto ParticleMeshWorkspace<Float>::plan_reserve(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t particles,
    const std::int64_t lanes,
    const std::int64_t spline_order) const -> Reservation
{
    validate_reserve(nx, ny, nz, particles, lanes, spline_order);
    const MeshShape requested(nx, ny, nz, lanes, MeshLayout::right);
    const std::int64_t offsets =
        detail::checked_product(3, particles, "particle-mesh stencil offsets");
    const std::int64_t stencil_elements = detail::checked_product(
        offsets, spline_order, "particle-mesh stencil values");
    const std::int64_t bins = requested.spectrum_point_count();

    Reservation reservation;
    reservation.shape = requested;
    reservation.particles = particles;
    reservation.lanes = lanes;
    reservation.spline_order = spline_order;

    // Build every replacement locally. If any allocation fails, the prior
    // workspace, allocation identity, and plans remain untouched.
    detail::maybe_fail_allocation("the particle-mesh stencil offsets");
    reservation.stencil_base = Kokkos::View<std::int32_t*>(
        "symmetrix_particle_mesh_stencil_base", offsets);
    detail::maybe_fail_allocation("the particle-mesh stencil weights");
    reservation.stencil_weight = Kokkos::View<Float*>(
        "symmetrix_particle_mesh_stencil_weight", stencil_elements);
    detail::maybe_fail_allocation("the particle-mesh stencil slopes");
    reservation.stencil_slope = Kokkos::View<Float*>(
        "symmetrix_particle_mesh_stencil_slope", stencil_elements);
    detail::maybe_fail_allocation("the particle-mesh wave-vector norms");
    reservation.k_norm =
        Kokkos::View<double*>("symmetrix_particle_mesh_k_norm", bins);
    detail::maybe_fail_allocation("the particle-mesh deconvolution symbol");
    reservation.deconvolution =
        Kokkos::View<Float*>("symmetrix_particle_mesh_deconvolution", bins);
    detail::maybe_fail_allocation("the particle-mesh Hermitian multiplicities");
    reservation.hermitian =
        Kokkos::View<Float*>("symmetrix_particle_mesh_hermitian_weight", bins);

    reservation.storage = storage_.plan_reserve(nx, ny, nz, lanes);
    return reservation;
}

template <typename Float>
void ParticleMeshWorkspace<Float>::commit(Reservation&& reservation)
{
    storage_.commit(std::move(reservation.storage));
    stencil_ = ParticleStencil<Float>(
        std::move(reservation.stencil_base), std::move(reservation.stencil_weight),
        std::move(reservation.stencil_slope));
    grid_ = ReciprocalGridDeviceView<Float>{
        reservation.shape.extent(0),
        reservation.shape.extent(1),
        reservation.shape.extent(2),
        reservation.shape.spectrum_extent(),
        reservation.shape.point_count(),
        reservation.shape.spectrum_point_count(),
        {},
        0.0,
        std::move(reservation.k_norm),
        std::move(reservation.deconvolution),
        std::move(reservation.hermitian)};
    particles_ = reservation.particles;
    lanes_ = reservation.lanes;
    spline_order_ = reservation.spline_order;
    cell_inverse_ = {};
    volume_ = 0.0;
    cell_set_ = false;
    reciprocal_grid_prepared_ = false;
}

template <typename Float>
void ParticleMeshWorkspace<Float>::spread(
    const weights_view& weights,
    const groups_view& groups,
    const std::int32_t selected_group)
{
    require_reserved("spread");
    if (static_cast<std::int64_t>(weights.extent(0)) != particles_
        || static_cast<std::int64_t>(weights.extent(1)) != lanes_) {
        throw std::invalid_argument(
            "Particle-mesh spread weights must have shape [particles, lanes].");
    }
    if (groups.extent(0) != 0
        && static_cast<std::int64_t>(groups.extent(0)) != particles_) {
        throw std::invalid_argument(
            "Particle-mesh spread groups must be empty or hold one entry per particle.");
    }

    auto field = storage_.real_field();
    Kokkos::deep_copy(execution_space_, field, Float(0));
    const auto base = stencil_.base();
    const auto stencil_weight = stencil_.weights();
    const std::int64_t particles = particles_;
    const std::int64_t lanes = lanes_;
    const std::int64_t nx = grid_.nx;
    const std::int64_t ny = grid_.ny;
    const std::int64_t nz = grid_.nz;
    const int order = static_cast<int>(spline_order_);
    const bool filter_groups = groups.extent(0) != 0;
    const std::int64_t particle_lanes = detail::checked_product(
        particles, lanes, "particle-mesh spread particle lanes");

    Kokkos::parallel_for(
        "symmetrix_particle_mesh_spread",
        Kokkos::RangePolicy<execution_space>(execution_space_, 0, particle_lanes),
        KOKKOS_LAMBDA(const std::int64_t element) {
            const std::int64_t particle = element / lanes;
            const std::int64_t lane = element % lanes;
            if (filter_groups && groups(particle) != selected_group) {
                return;
            }
            const Float particle_weight = weights(particle, lane);
            if (particle_weight == Float(0)) {
                return;
            }
            for (int i = 0; i < order; ++i) {
                const Float w0 = stencil_weight((particle * 3 + 0) * order + i);
                const std::int64_t n0 = wrap_node(base(particle * 3 + 0) + i, nx);
                for (int j = 0; j < order; ++j) {
                    const Float w1 = stencil_weight((particle * 3 + 1) * order + j);
                    const std::int64_t n1 = wrap_node(base(particle * 3 + 1) + j, ny);
                    const Float plane = particle_weight * w0 * w1;
                    for (int k = 0; k < order; ++k) {
                        const Float w2 = stencil_weight((particle * 3 + 2) * order + k);
                        const std::int64_t n2 =
                            wrap_node(base(particle * 3 + 2) + k, nz);
                        Kokkos::atomic_add(&field(lane, n0, n1, n2), plane * w2);
                    }
                }
            }
        });
}

template <typename Float>
void ParticleMeshWorkspace<Float>::deconvolve_amplitudes()
{
    require_reserved("deconvolution");
    const auto spectrum_view = storage_.spectrum();
    const auto divisor = grid_.deconvolution;
    const std::int64_t lanes = lanes_;
    const std::int64_t bins = grid_.bins;
    const std::int64_t elements =
        detail::checked_product(lanes, bins, "particle-mesh spectrum elements");
    Kokkos::parallel_for(
        "symmetrix_particle_mesh_deconvolve",
        Kokkos::RangePolicy<execution_space>(execution_space_, 0, elements),
        KOKKOS_LAMBDA(const std::int64_t element) {
            spectrum_view.data()[element] /= divisor(element % bins);
        });
}

template <typename Float>
void ParticleMeshWorkspace<Float>::load_physical_cotangent(
    const complex_input_view& deconvolved_amplitude_cotangent)
{
    require_reserved("physical cotangent loading");
    const std::int64_t elements = detail::checked_product(
        lanes_, grid_.bins, "particle-mesh physical cotangent elements");
    if (static_cast<std::int64_t>(deconvolved_amplitude_cotangent.extent(0))
        != elements) {
        throw std::invalid_argument(
            "Particle-mesh deconvolved amplitude cotangent must have shape "
            "[lanes, spectrum bins].");
    }
    Kokkos::View<complex_type*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        destination(storage_.spectrum().data(), elements);
    Kokkos::deep_copy(
        execution_space_, destination, deconvolved_amplitude_cotangent);
}

template <typename Float>
void ParticleMeshWorkspace<Float>::deconvolve_cotangent()
{
    require_reserved("cotangent deconvolution");
    const auto spectrum_view = storage_.spectrum();
    const auto divisor = grid_.deconvolution;
    const std::int64_t bins = grid_.bins;
    const std::int64_t elements = detail::checked_product(
        lanes_, bins, "particle-mesh cotangent spectrum elements");
    Kokkos::parallel_for(
        "symmetrix_particle_mesh_deconvolve_cotangent",
        Kokkos::RangePolicy<execution_space>(execution_space_, 0, elements),
        KOKKOS_LAMBDA(const std::int64_t element) {
            spectrum_view.data()[element] /= divisor(element % bins);
        });
}

template <typename Float>
void ParticleMeshWorkspace<Float>::gather(
    const weights_view& weights,
    const groups_view& groups,
    const std::int32_t selected_group,
    const weight_adjoints_view& weight_adjoints,
    const position_forces_view& position_forces)
{
    require_reserved("gather");
    if (static_cast<std::int64_t>(weights.extent(0)) != particles_
        || static_cast<std::int64_t>(weights.extent(1)) != lanes_
        || static_cast<std::int64_t>(weight_adjoints.extent(0)) != particles_
        || static_cast<std::int64_t>(weight_adjoints.extent(1)) != lanes_
        || static_cast<std::int64_t>(position_forces.extent(0)) != particles_
        || static_cast<std::int64_t>(position_forces.extent(1)) != 3) {
        throw std::invalid_argument(
            "Particle-mesh gather weights/adjoints must have shape [particles, "
            "lanes] and position forces shape [particles, 3].");
    }
    if (groups.extent(0) != 0
        && static_cast<std::int64_t>(groups.extent(0)) != particles_) {
        throw std::invalid_argument(
            "Particle-mesh gather groups must be empty or hold one entry per particle.");
    }

    const bool filter_groups = groups.extent(0) != 0;
    // An ungrouped gather is a complete operation. Grouped gathers are the
    // adapter composition form and accumulate in the caller-cleared outputs.
    if (!filter_groups) {
        Kokkos::deep_copy(execution_space_, weight_adjoints, Float(0));
        Kokkos::deep_copy(execution_space_, position_forces, Float(0));
    }

    const auto field = storage_.real_field();
    const auto base_view = stencil_.base();
    const auto stencil_weight = stencil_.weights();
    const auto stencil_slope = stencil_.slopes();
    const std::int64_t particles = particles_;
    const std::int64_t lanes = lanes_;
    const std::int64_t nx = grid_.nx;
    const std::int64_t ny = grid_.ny;
    const std::int64_t nz = grid_.nz;
    const int order = static_cast<int>(spline_order_);
    const PackedMatrix3 inverse = pack_matrix(cell_inverse_);
    const std::int64_t particle_lanes = detail::checked_product(
        particles, lanes, "particle-mesh gather particle lanes");

    Kokkos::parallel_for(
        "symmetrix_particle_mesh_gather",
        Kokkos::RangePolicy<execution_space>(execution_space_, 0, particle_lanes),
        KOKKOS_LAMBDA(const std::int64_t element) {
            const std::int64_t particle = element / lanes;
            const std::int64_t lane = element % lanes;
            if (filter_groups && groups(particle) != selected_group) {
                return;
            }
            Float weight[3][bspline::max_order];
            Float slope[3][bspline::max_order];
            std::int64_t base[3];
            for (std::int64_t axis = 0; axis < 3; ++axis) {
                base[axis] = base_view(particle * 3 + axis);
                for (int offset = 0; offset < order; ++offset) {
                    weight[axis][offset] =
                        stencil_weight((particle * 3 + axis) * order + offset);
                    slope[axis][offset] =
                        stencil_slope((particle * 3 + axis) * order + offset);
                }
            }
            Float accumulated_adjoint = Float(0);
            Float accumulated_force[3] = {Float(0), Float(0), Float(0)};
            const Float particle_weight = weights(particle, lane);
            for (int i = 0; i < order; ++i) {
                const Float w0 = weight[0][i];
                const std::int64_t n0 = wrap_node(base[0] + i, nx);
                for (int j = 0; j < order; ++j) {
                    const Float w1 = weight[1][j];
                    const std::int64_t n1 = wrap_node(base[1] + j, ny);
                    for (int k = 0; k < order; ++k) {
                        const Float w2 = weight[2][k];
                        const std::int64_t n2 = wrap_node(base[2] + k, nz);
                        const Float value = w0 * w1 * w2;
                        const Float fractional[3] = {
                            slope[0][i] * w1 * w2,
                            w0 * slope[1][j] * w2,
                            w0 * w1 * slope[2][k]};
                        Float cartesian[3] = {Float(0), Float(0), Float(0)};
                        for (std::int64_t c = 0; c < 3; ++c) {
                            for (std::int64_t a = 0; a < 3; ++a) {
                                cartesian[c] += fractional[a]
                                    * static_cast<Float>(inverse(c, a));
                            }
                        }
                        const Float potential = field(lane, n0, n1, n2);
                        accumulated_adjoint += value * potential;
                        accumulated_force[0] -=
                            particle_weight * potential * cartesian[0];
                        accumulated_force[1] -=
                            particle_weight * potential * cartesian[1];
                        accumulated_force[2] -=
                            particle_weight * potential * cartesian[2];
                    }
                }
            }
            weight_adjoints(particle, lane) += accumulated_adjoint;
            Kokkos::atomic_add(
                &position_forces(particle, 0), accumulated_force[0]);
            Kokkos::atomic_add(
                &position_forces(particle, 1), accumulated_force[1]);
            Kokkos::atomic_add(
                &position_forces(particle, 2), accumulated_force[2]);
        });
}

extern template class ParticleMeshWorkspace<float>;
extern template class ParticleMeshWorkspace<double>;

}  // namespace symmetrix::particle_mesh

#endif
