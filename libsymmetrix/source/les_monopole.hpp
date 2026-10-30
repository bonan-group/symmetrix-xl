#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace symmetrix::les {

class LESParticleMesh;

enum class LESPeriodicAlgorithm {
    explicit_sum,
    particle_mesh,
};

struct MonopoleResult {
    double energy = 0.0;
    std::vector<double> atomic_energies;
    std::vector<double> forces;
    std::vector<double> charge_adjoints;
    std::vector<double> strain_derivative;
};

struct MonopoleDeviceResult {
    Kokkos::View<double> energy;
    Kokkos::View<double*> atomic_energies;
    Kokkos::View<double*> forces;
    Kokkos::View<double*> charge_adjoints;
    Kokkos::View<double*> strain_derivative;
};

struct MonopoleDiagnostics {
    std::size_t reciprocal_candidate_count = 0;
    std::size_t reciprocal_vector_count = 0;
    std::size_t logical_extent_bytes = 0;
    std::size_t allocated_capacity_bytes = 0;
    std::size_t active_index_capacity_bytes = 0;
    std::size_t index_pack_count = 0;
    std::size_t table_rebuilds = 0;
    std::size_t geometry_updates = 0;
    std::size_t active_bytes = 0;
    std::size_t capacity_bytes = 0;
    LESPeriodicAlgorithm periodic_algorithm = LESPeriodicAlgorithm::explicit_sum;
    std::uint64_t particle_mesh_allocation_generation = 0;
    std::uint64_t particle_mesh_reservations = 0;
    std::uint64_t particle_mesh_allocations = 0;
    std::uint64_t particle_mesh_forward_plan_builds = 0;
    std::uint64_t particle_mesh_forward_plan_reuses = 0;
    std::uint64_t particle_mesh_inverse_plan_builds = 0;
    std::uint64_t particle_mesh_inverse_plan_reuses = 0;
    std::uint64_t particle_mesh_reciprocal_preparations = 0;
    std::uint64_t particle_mesh_stencil_preparations = 0;
    std::uint64_t particle_mesh_particle_count_fences = 0;
    std::uint32_t particle_mesh_execution_instance_id = 0;
};

template <typename Precision>
class LESMonopoleKokkos {
public:
    using execution_space = Kokkos::DefaultExecutionSpace;

    LESMonopoleKokkos(double sigma, double dl, bool remove_self_interaction);
    ~LESMonopoleKokkos();

    MonopoleResult compute_nonperiodic(
        const std::vector<double>& charges,
        const std::vector<double>& positions);
    MonopoleResult compute_periodic(
        const std::vector<double>& charges,
        const std::vector<double>& positions,
        const std::vector<double>& cell);

    MonopoleDeviceResult compute_nonperiodic_device(
        const Kokkos::View<double*>& charges,
        const Kokkos::View<double*>& positions,
        std::size_t count);
    MonopoleDeviceResult compute_nonperiodic_device(
        const execution_space& execution_space,
        const Kokkos::View<double*>& charges,
        const Kokkos::View<double*>& positions,
        std::size_t count);
    void prepare_periodic(const std::vector<double>& cell);
    void prepare_periodic(
        const execution_space& execution_space,
        const std::vector<double>& cell);
    void prepare_particle_count_change(std::size_t count);
    void prepare_particle_count_change(
        const execution_space& execution_space,
        std::size_t count);
    void prepare_periodic_geometry(
        const execution_space& execution_space,
        const std::vector<double>& cell,
        std::span<const double> host_positions,
        const Kokkos::View<double*>& device_positions,
        std::size_t count);
    MonopoleDeviceResult compute_periodic_device(
        const Kokkos::View<double*>& charges,
        const Kokkos::View<double*>& positions,
        std::size_t count);
    MonopoleDeviceResult compute_periodic_device(
        const execution_space& execution_space,
        const Kokkos::View<double*>& charges,
        const Kokkos::View<double*>& positions,
        std::size_t count);
    [[nodiscard]] MonopoleDiagnostics diagnostics() const;
    [[nodiscard]] std::vector<std::size_t>
    active_reciprocal_indices_for_testing() const;
    void configure_particle_mesh_for_testing(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t spline_order);
    void use_explicit_periodic_for_testing();

private:
    void reserve_outputs(std::size_t count, bool periodic);
    MonopoleResult copy_to_host(
        const MonopoleDeviceResult& result, std::size_t count) const;

    double sigma_;
    double dl_;
    bool remove_self_interaction_;
    Kokkos::View<double*> atomic_energies_;
    Kokkos::View<double> energy_;
    Kokkos::View<double*> forces_;
    Kokkos::View<double*> charge_adjoints_;
    Kokkos::View<double*> strain_derivative_;
    Kokkos::View<double*> reciprocal_cell_;
    Kokkos::View<double**, Kokkos::LayoutRight> reciprocal_vectors_;
    Kokkos::View<double*> reciprocal_squared_;
    Kokkos::View<double*> reciprocal_kernels_;
    Kokkos::View<unsigned char*> reciprocal_active_;
    Kokkos::View<std::size_t*> reciprocal_active_indices_;
    Kokkos::View<double*> structure_real_;
    Kokkos::View<double*> structure_imaginary_;
    std::array<double, 9> cached_cell_{};
    bool cell_ready_ = false;
    std::size_t active_nodes_ = 0;
    std::size_t reciprocal_candidates_ = 0;
    std::size_t reciprocal_vector_count_ = 0;
    std::size_t index_pack_count_ = 0;
    std::size_t table_rebuilds_ = 0;
    std::size_t geometry_updates_ = 0;
    LESPeriodicAlgorithm periodic_algorithm_ = LESPeriodicAlgorithm::explicit_sum;
    std::array<std::int64_t, 3> particle_mesh_shape_{};
    std::int64_t particle_mesh_spline_order_ = 0;
    std::unique_ptr<LESParticleMesh> particle_mesh_;
};

} // namespace symmetrix::les
