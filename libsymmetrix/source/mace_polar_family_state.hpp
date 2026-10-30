#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>

#include "Kokkos_Core.hpp"
#include "mace_polar_deltas.hpp"
#include "mace_polar_program.hpp"

#include "mace_polar_particle_mesh.hpp"

template<typename Precision>
class MacePolarFamilyState final {
public:
    std::optional<symmetrix::execution::polar::PolarLongRangeContract>
        long_range_contract;
    bool graph_inputs_set=false;
    double total_charge=0.0;
    double total_spin=0.0;
    double fermi_level=0.0;
    std::array<double,3> electric_field{0.0,0.0,0.0};
    std::shared_ptr<symmetrix::mace_polar::PolarDeltaChainKokkosT<Precision>>
        deltas;
    // The family owns its PME workspace. The shared particle-mesh layer below
    // it remains physics-neutral and is never replaced by the LES adapter.
    std::shared_ptr<symmetrix::mace_polar::PolarParticleMeshWorkspaceT<Precision>>
        particle_mesh_workspace;
    Kokkos::View<double*> public_outputs_storage;
    std::size_t public_outputs_storage_replacements=0;
};
