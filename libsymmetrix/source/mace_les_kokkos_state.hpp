#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "les_monopole.hpp"
#include "mace_kokkos.hpp"
#include "mace_les_model.hpp"

template <typename Precision>
struct MACELESState {
    MACELESState() = default;
    MACELESState(const MACELESState&) = delete;
    MACELESState& operator=(const MACELESState&) = delete;

    std::unique_ptr<MACELESParameters<Precision>> parameters;
    std::unique_ptr<symmetrix::les::LESMonopoleKokkos<Precision>> monopole;
    bool periodic = false;
    Kokkos::View<double*> readout_1_weights;
    MultilayerPerceptronKokkos readout_2;
    Kokkos::View<double*> charge_readout_1;
    Kokkos::View<double*> charge_readout_2;
    Kokkos::View<double**, Kokkos::LayoutRight> charge_readout_2_gradient;
    Kokkos::View<double*> latent_charges;
    Kokkos::View<double*> charge_adjoints;
    Kokkos::View<double*> positions;
    Kokkos::View<double*> cell;
    Kokkos::View<int*> pbc;
    Kokkos::View<double*> atomic_energies;
    Kokkos::View<double> energy_device;
    Kokkos::View<double, Kokkos::HostSpace> energy_host;
    Kokkos::View<double*> explicit_forces;
    Kokkos::View<double*> strain_derivative;
    double energy = 0.0;
    bool energy_copy_pending = false;
    int geometry_active_nodes = 0;
    bool geometry_ready = false;
    bool results_ready = false;
    bool charge_readouts_ready = false;
    int charge_readout_nodes = 0;
    bool geometry_execution_space_selected = false;
    bool geometry_execution_space_factorized = false;
    bool evaluation_execution_space_selected = false;
    bool evaluation_execution_space_factorized = false;
    bool charge_adjoint_accumulation_enabled = true;
    int charge_active_nodes = 0;
    std::size_t charge_workspace_replacements = 0;
    std::size_t charge_workspace_reuses = 0;
    symmetrix::les::LESPeriodicAlgorithm periodic_algorithm =
        symmetrix::les::LESPeriodicAlgorithm::explicit_sum;
    std::array<std::int64_t, 3> particle_mesh_shape{};
    std::int64_t particle_mesh_spline_order = 0;
    bool particle_mesh_owner_valid = false;
    ExecutionMode particle_mesh_owner_mode = ExecutionMode::materialized;
    Kokkos::DefaultExecutionSpace particle_mesh_owner_execution_space;
    std::uint32_t particle_mesh_execution_instance_id = 0;
    std::uint64_t particle_mesh_owner_generation = 0;
    std::uint64_t particle_mesh_owner_fences = 0;
};
