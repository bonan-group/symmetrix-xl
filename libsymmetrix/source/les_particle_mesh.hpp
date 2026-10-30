#ifndef SYMMETRIX_LES_PARTICLE_MESH_HPP
#define SYMMETRIX_LES_PARTICLE_MESH_HPP

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace symmetrix::les {

struct MonopoleDeviceResult;

struct LESParticleMeshDiagnostics {
    std::uint64_t allocation_generation = 0;
    std::uint64_t reservations = 0;
    std::uint64_t allocations = 0;
    std::uint64_t forward_plan_builds = 0;
    std::uint64_t forward_plan_reuses = 0;
    std::uint64_t inverse_plan_builds = 0;
    std::uint64_t inverse_plan_reuses = 0;
    std::uint64_t reciprocal_preparations = 0;
    std::uint64_t stencil_preparations = 0;
    std::uint64_t particle_count_fences = 0;
    std::uint32_t execution_instance_id = 0;
};

class LESParticleMesh {
public:
    using execution_space = Kokkos::DefaultExecutionSpace;

    LESParticleMesh(
        execution_space execution_space,
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t spline_order);
    ~LESParticleMesh();

    LESParticleMesh(const LESParticleMesh&) = delete;
    LESParticleMesh& operator=(const LESParticleMesh&) = delete;
    LESParticleMesh(LESParticleMesh&&) = delete;
    LESParticleMesh& operator=(LESParticleMesh&&) = delete;

    void prepare_cell(
        const std::vector<double>& cell,
        const std::array<double, 9>& cell_inverse,
        double volume,
        double reciprocal_spacing);
    void prepare_particle_count_change(std::size_t count);
    void cancel_particle_count_change();
    void prepare_positions(
        std::span<const double> host_positions,
        const Kokkos::View<double*>& device_positions,
        std::size_t count);
    void compute(
        const Kokkos::View<double*>& charges,
        const Kokkos::View<double*>& positions,
        std::size_t count,
        double sigma,
        double reciprocal_spacing,
        bool remove_self_interaction,
        const MonopoleDeviceResult& output);
    [[nodiscard]] LESParticleMeshDiagnostics diagnostics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void require_les_particle_mesh_available();

}  // namespace symmetrix::les

#endif
