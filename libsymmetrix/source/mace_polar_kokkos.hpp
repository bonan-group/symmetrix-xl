#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>

#include "mace_nonlinear_kokkos.hpp"
#include "mace_polar_family_state.hpp"
#include "mace_polar_schema.hpp"

namespace symmetrix::mace_polar {

inline nlohmann::json load_model_json(
    const std::string& filename,const std::string& requested_head)
{
    std::ifstream stream(filename);
    if(!stream)
        throw std::runtime_error(
            "Could not open MACE Polar Kokkos model file: "+filename);
    return select_prediction_head(
        nlohmann::json::parse(stream),requested_head);
}

// Compact, frontend-neutral request for the complete Polar scientific
// transaction over an already prepared short-range graph.  The spans are
// borrowed only for the duration of evaluate_prepared(); every large
// intermediate remains owned by the evaluator or its Polar chain.
struct PreparedEvaluationRequest {
    std::uint64_t graph_generation=0;
    std::span<const int> node_types;
    std::span<const double> positions;
    std::span<const double> cell;
    std::span<const int> graph_pbc;
    std::span<const int> node_graph_index;
    double total_charge=0.0;
    double total_spin=0.0;
    std::array<double,4> external_potential{0.0,0.0,0.0,0.0};
    bool include_forces=false;
    bool include_stress=false;
    bool include_dipole=false;
    execution::long_range::LongRangeSolverRequest solver_request=
        execution::long_range::LongRangeSolverRequest::automatic;
    std::array<std::int64_t,3> pme_mesh{64,64,64};
    std::int64_t pme_spline_order=6;
};

struct PreparedEvaluationDiagnostics {
    execution::long_range::LongRangeSolverRequest requested_solver=
        execution::long_range::LongRangeSolverRequest::automatic;
    execution::long_range::LongRangeSolver resolved_solver=
        execution::long_range::LongRangeSolver::unresolved;
    std::string solver_reason;
    std::size_t automatic_atom_count=0;
    std::size_t automatic_reciprocal_mode_count=0;
    std::size_t automatic_atom_mode_pair_count=0;
    std::size_t automatic_atom_mode_pair_threshold=
        execution::long_range::automatic_pme_atom_mode_pair_threshold;
    bool automatic_reciprocal_mode_count_is_lower_bound=false;
    std::size_t n_k=0;
    std::size_t reciprocal_allocation_elements=0;
    std::size_t reciprocal_allocation_bytes=0;
    std::size_t reciprocal_table_elements=0;
    std::uint64_t particle_mesh_active_bytes=0;
    std::uint64_t particle_mesh_capacity_bytes=0;
    std::uint64_t particle_mesh_volume=0;
    std::array<std::int64_t,3> particle_mesh_shape{0,0,0};
    std::uint64_t particle_mesh_rebuilds=0;
    std::uint64_t particle_mesh_reuses=0;
    std::uint64_t particle_mesh_forward_plan_builds=0;
    std::uint64_t particle_mesh_forward_plan_reuses=0;
    std::uint64_t particle_mesh_inverse_plan_builds=0;
    std::uint64_t particle_mesh_inverse_plan_reuses=0;
    std::size_t realspace_retained_elements=0;
    std::size_t realspace_retained_bytes=0;
    std::size_t diagnostic_materialization_count=0;
    std::size_t public_output_elements=0;
    std::size_t backbone_fallback_count=0;
    std::size_t polar_fallback_count=0;
    std::size_t fallback_count=0;
    std::string reverse_strategy;
    std::size_t cumulative_transaction_count=0;
    std::size_t cumulative_polar_reverse_count=0;
    std::size_t backbone_evaluation_count=0;
    std::size_t backbone_forward_layer_count=0;
    std::size_t backbone_reverse_layer_count=0;
    std::size_t seeded_backbone_evaluation_count=0;
    std::size_t polar_reverse_count=0;
    bool reverse_executed=false;
    bool product_seed_cleared=false;
};

struct PreparedEvaluationResult {
    double energy=0.0;
    double free_energy=0.0;
    std::vector<double> forces;
    std::vector<double> stress;
    std::vector<double> dipole;
    PreparedEvaluationDiagnostics diagnostics;
};

struct PreparedOutputLayout {
    std::size_t energy_offset=0;
    std::size_t forces_offset=1;
    std::size_t forces_elements=0;
    std::size_t stress_offset=1;
    std::size_t stress_elements=0;
    std::size_t dipole_offset=1;
    std::size_t dipole_elements=0;
    std::size_t elements=1;
};

// Borrowed device result for native consumers such as LAMMPS.  The view is
// evaluator-owned and remains valid until the next evaluation on this owner.
// Copying compact public scalars to a host frontend is optional; forces can be
// accumulated directly into a consumer-owned device view.
struct PreparedDeviceEvaluationResult {
    Kokkos::View<const double*> public_outputs;
    PreparedOutputLayout layout;
    PreparedEvaluationDiagnostics diagnostics;
};

} // namespace symmetrix::mace_polar

// The PolarMACE family's public evaluator.  It is a sibling of the ordinary
// nonlinear evaluator over the same internal backbone engine; neither public
// family evaluator invokes or derives from the other.  Only this type exposes
// the Polar contract, graph inputs, delta chain, and public-output collector.
//
// The graph-level scalar channel is part of the compact request accepted by
// every native consumer. Its mapping is:
//
//   * ``total_charge`` and ``total_spin`` -> the ``fukui_normalize`` node's
//     per-graph targets, the closed-form equilibration targets
//     (Q + S - 1) / 2 and (Q - S + 1) / 2;
//   * ``electric_field`` -> lanes 1..3 of the ``external_field_projection``
//     node's ``external_potential`` buffer;
//   * ``fermi_level`` -> lane 0 of that buffer, which the frozen oracle fills
//     with ``zeros_like(fermi_level)`` (mace/modules/extensions.py:643-645) and
//     never couples.
//
// Those global nodes, and the polar backbone regions that produce their
// remaining inputs (the spin-source heads, the feature mixer, the field-update
// chain, and the local-electron-energy readout), are declared program data by
// mace_polar_program.hpp; this family evaluator owns the chain that runs them.
template<typename Precision>
class MacePolarKokkosT final : public MaceBackboneKokkosT<Precision> {
    using Backbone=MaceBackboneKokkosT<Precision>;

    MacePolarFamilyState<Precision> polar_state_;
    std::size_t whole_model_transaction_count_=0;
    std::size_t whole_model_reverse_count_=0;
    std::size_t whole_model_failure_count_=0;

    symmetrix::execution::long_range::LongRangeSolverCapabilities
    polar_solver_capabilities() const noexcept
    {
        symmetrix::execution::long_range::LongRangeSolverCapabilities result;
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
        result.pme_fft_provider_available=
            polar_state_.particle_mesh_workspace!=nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
        result.pme_backend_qualified=
            polar_state_.particle_mesh_workspace!=nullptr;
#endif
#endif
        return result;
    }

public:
    double polar_cutoff() const { return this->r_cut; }
    std::string polar_streamed_edges_mode() const {
        return this->streamed_edges_mode();
    }
    symmetrix::execution::long_range::LongRangeSolverDecision
    preflight_polar_solver(
        const symmetrix::execution::long_range::LongRangeSolverRequest request,
        const std::array<int,3>& pbc) const
    {
        const auto& long_range=this->polar_deltas().contract().long_range;
        const auto decision=
            symmetrix::execution::long_range::resolve_long_range_solver(
            request,pbc,long_range.density_max_l,long_range.feature_max_l,
            polar_solver_capabilities());
        if(!decision.accepted)throw std::invalid_argument(decision.error);
        return decision;
    }
    void arm_polar_geometry_prepare_failure_for_testing() {
        this->arm_factorized_geometry_prepare_failure_for_testing();
    }
    std::size_t polar_geometry_allocation_count() const {
        return this->execution_geometry_allocation_count();
    }
    std::size_t polar_geometry_workspace_bytes() const {
        return this->execution_geometry_workspace_bytes();
    }
    std::size_t polar_public_output_allocation_count() const {
        return this->polar_public_outputs_storage_replacement_count();
    }
    bool owns_polar_particle_mesh_workspace() const noexcept
    {
        return polar_state_.particle_mesh_workspace != nullptr;
    }
    std::size_t polar_topology_validation_count() const {
        return this->factorized_prepared_topology_validation_count();
    }
    std::size_t polar_topology_validated_edge_count() const {
        return this->factorized_prepared_topology_validated_edge_count();
    }
    std::size_t polar_input_h2d_bytes() const {
        const std::size_t backbone=this->factorized_input_h2d_bytes();
        const std::size_t polar=this->polar_deltas().input_h2d_bytes();
        if(backbone>std::numeric_limits<std::size_t>::max()-polar)
            throw std::overflow_error(
                "MACE Polar cumulative input byte count overflows size_t.");
        return backbone+polar;
    }
    using precision_type=Precision;
    explicit MacePolarKokkosT(
        const std::string& filename, const std::string& requested_head = {})
        :MacePolarKokkosT(
            symmetrix::mace_polar::load_model_json(filename,requested_head)) {}

    void set_polar_long_range_contract(
        const symmetrix::execution::polar::PolarLongRangeContract& contract);
    bool has_polar_long_range_contract() const;
    const std::optional<symmetrix::execution::polar::PolarLongRangeContract>&
        polar_long_range_contract() const;
    symmetrix::execution::polar::PolarSchedule polar_schedule() const;
    bool polar_delta_ready() const;
    const symmetrix::mace_polar::PolarDeltaChainKokkosT<Precision>&
        polar_deltas() const;
    symmetrix::mace_polar::PolarDeltaChainKokkosT<Precision>& polar_deltas();
    void set_polar_graph_inputs(
        double total_charge,double total_spin,double fermi_level,
        const std::vector<double>& electric_field);
    bool polar_graph_inputs_set() const;
    double polar_total_charge() const;
    double polar_total_spin() const;
    double polar_fermi_level() const;
    std::array<double,3> polar_electric_field() const;
    std::vector<double> collect_polar_public_outputs(
        double volume,std::uint64_t graph_generation,
        bool include_forces,bool include_stress,bool include_dipole=false);
    std::size_t polar_public_outputs_storage_replacement_count() const;
    symmetrix::execution::FamilyCapabilities family_capabilities() const;
    symmetrix::execution::Program compile_program() const;
    symmetrix::execution::ImplementationRegistry
        family_implementation_registry() const;
    symmetrix::execution::ImplementationBindingReport
        family_binding_report() const;

    symmetrix::mace_polar::PreparedEvaluationResult evaluate_prepared(
        const symmetrix::mace_polar::PreparedEvaluationRequest& request);
    symmetrix::mace_polar::PreparedDeviceEvaluationResult
    evaluate_prepared_device(
        const symmetrix::mace_polar::PreparedEvaluationRequest& request);
    std::size_t whole_model_transaction_count() const
    {
        return whole_model_transaction_count_;
    }
    std::size_t whole_model_reverse_count() const
    {
        return whole_model_reverse_count_;
    }
    std::size_t whole_model_failure_count() const
    {
        return whole_model_failure_count_;
    }

private:
    explicit MacePolarKokkosT(const nlohmann::json& data);
    Kokkos::View<const double*> pack_polar_public_outputs_device(
        double volume,std::uint64_t graph_generation,
        bool include_forces,bool include_stress,bool include_dipole);
};

namespace symmetrix::mace_polar::detail {
template<class Energies>
struct PublicEnergyReducer {
    Energies energies;
    KOKKOS_FUNCTION void operator()(const std::size_t node,double& value) const
    { value+=energies(node); }
};
template<class Outputs,class Forces,class PositionAdjoint,
    class BackboneStress,class PolarStress>
struct PublicPayloadPacker {
    Outputs outputs;
    Forces forces;
    PositionAdjoint polar_position;
    BackboneStress backbone_stress;
    PolarStress polar_stress;
    std::size_t force_values;
    bool zero_isolated_stress;
    KOKKOS_FUNCTION void operator()(const std::size_t payload) const
    {
        if(payload<force_values) {
            const std::size_t node=payload/3;
            const int component=static_cast<int>(payload%3);
            outputs(1+payload)=forces(payload)
                -double(polar_position(node,component));
            return;
        }
        const std::size_t component=payload-force_values;
        const std::size_t offset=1+force_values;
        if(zero_isolated_stress)outputs(offset+component)=0.0;
        else if(component==0)
            outputs(offset)=backbone_stress(0)+polar_stress(0,0);
        else if(component==1)
            outputs(offset+1)=backbone_stress(4)+polar_stress(0,4);
        else if(component==2)
            outputs(offset+2)=backbone_stress(8)+polar_stress(0,8);
        else if(component==3)
            outputs(offset+3)=0.5*(backbone_stress(5)+backbone_stress(7)
                +polar_stress(0,5)+polar_stress(0,7));
        else if(component==4)
            outputs(offset+4)=0.5*(backbone_stress(2)+backbone_stress(6)
                +polar_stress(0,2)+polar_stress(0,6));
        else
            outputs(offset+5)=0.5*(backbone_stress(1)+backbone_stress(3)
                +polar_stress(0,1)+polar_stress(0,3));
    }
};
template<class Outputs,class PolarOutputs>
struct PublicGraphEnergyAdder {
    Outputs outputs;
    PolarOutputs polar_outputs;
    KOKKOS_FUNCTION void operator()(const int) const
    { outputs(0)+=polar_outputs(0,0); }
};
template<class Outputs,class PolarOutputs>
struct PublicDipolePacker {
    Outputs outputs;
    PolarOutputs polar_outputs;
    std::size_t dipole_offset;
    KOKKOS_FUNCTION void operator()(const int component) const
    { outputs(dipole_offset+component)=polar_outputs(0,5+component); }
};
} // namespace symmetrix::mace_polar::detail

template<typename Precision>
MacePolarKokkosT<Precision>::MacePolarKokkosT(const nlohmann::json& data)
    :Backbone(data,MaceBackboneSchema::polar,false)
{
    try {
        symmetrix::mace_polar::validate_mace_polar_schema(data);
        set_polar_long_range_contract(
            symmetrix::mace_polar::long_range_contract_from_model(data));
    } catch(const std::exception& error) {
        throw std::invalid_argument(std::string("polar schema: ")+error.what());
    }
    if(!data.contains("polar"))return;
    using DeltaChain=symmetrix::mace_polar::PolarDeltaChainKokkosT<Precision>;
    if(!data.at("interactions").is_array())
        throw std::invalid_argument("polar deltas: interactions");
    if(!data.at("products").is_array()||data.at("products").empty())
        throw std::invalid_argument("polar deltas: products");
    if(!data.at("products").at(0).contains("target_irreps"))
        throw std::invalid_argument("polar deltas: target_irreps");
    if(!data.at("num_elements").is_number_integer())
        throw std::invalid_argument("polar deltas: num_elements");
    if(!data.at("polar").is_object())
        throw std::invalid_argument("polar deltas: polar block");
    typename DeltaChain::Contract contract;
    contract.long_range=*polar_long_range_contract();
    contract.interaction_count=static_cast<int>(data.at("interactions").size());
    const Irreps targets(data.at("products").at(0).at("target_irreps")
        .get<std::string>());
    contract.node_feature_dimension=targets.dimension();
    contract.num_elements=data.at("num_elements").get<int>();
    contract.subtract_feature_self_terms=!data.at("polar").value("field_si",false);
    try {
        polar_state_.deltas=std::make_shared<DeltaChain>(data,contract);
        polar_state_.deltas->set_particle_mesh_workspace(
            polar_state_.particle_mesh_workspace);
    } catch(const std::exception& error) {
        throw std::invalid_argument(std::string("polar deltas: ")+error.what());
    }
}

template<typename Precision>
void MacePolarKokkosT<Precision>::set_polar_long_range_contract(
    const symmetrix::execution::polar::PolarLongRangeContract& contract)
{
    const std::string reason=
        symmetrix::execution::polar::validate_polar_long_range_contract(contract);
    if(!reason.empty())
        throw std::invalid_argument(
            "the PolarMACE long-range contract is invalid: "+reason);
    polar_state_.long_range_contract=contract;
    polar_state_.long_range_contract->table_dtype=
        sizeof(Precision)==sizeof(float)?"float32":"float64";
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
    if(contract.density_max_l<=1 && contract.feature_max_l<=1) {
        polar_state_.particle_mesh_workspace=std::make_shared<
            symmetrix::mace_polar::PolarParticleMeshWorkspaceT<Precision>>(
                Kokkos::DefaultExecutionSpace{},contract.density_max_l,
                contract.feature_max_l,contract.spin_channel_count,
                static_cast<std::int64_t>(
                    contract.feature_smearing_widths.size()));
    } else {
        polar_state_.particle_mesh_workspace.reset();
    }
    if(polar_state_.deltas)
        polar_state_.deltas->set_particle_mesh_workspace(
            polar_state_.particle_mesh_workspace);
#endif
}

template<typename Precision>
bool MacePolarKokkosT<Precision>::has_polar_long_range_contract() const
{ return polar_state_.long_range_contract.has_value(); }

template<typename Precision>
const std::optional<symmetrix::execution::polar::PolarLongRangeContract>&
MacePolarKokkosT<Precision>::polar_long_range_contract() const
{ return polar_state_.long_range_contract; }

template<typename Precision>
symmetrix::execution::polar::PolarSchedule
MacePolarKokkosT<Precision>::polar_schedule() const
{
    if(!has_polar_long_range_contract())return {};
    return symmetrix::execution::polar::polar_execution_schedule(
        polar_state_.long_range_contract->recursion_step_count);
}

template<typename Precision>
bool MacePolarKokkosT<Precision>::polar_delta_ready() const
{ return polar_state_.deltas!=nullptr; }

template<typename Precision>
const symmetrix::mace_polar::PolarDeltaChainKokkosT<Precision>&
MacePolarKokkosT<Precision>::polar_deltas() const
{
    if(!polar_state_.deltas)
        throw std::logic_error("this evaluator carries no PolarMACE deltas");
    return *polar_state_.deltas;
}

template<typename Precision>
symmetrix::mace_polar::PolarDeltaChainKokkosT<Precision>&
MacePolarKokkosT<Precision>::polar_deltas()
{
    if(!polar_state_.deltas)
        throw std::logic_error("this evaluator carries no PolarMACE deltas");
    return *polar_state_.deltas;
}

template<typename Precision>
void MacePolarKokkosT<Precision>::set_polar_graph_inputs(
    const double total_charge,const double total_spin,const double fermi_level,
    const std::vector<double>& electric_field)
{
    if(!has_polar_long_range_contract())
        throw std::invalid_argument(
            "the polar graph-input channel requires a configured evaluator");
    for(const auto& [name,value]:
        std::initializer_list<std::pair<const char*,double>>{
            {"total_charge",total_charge},{"total_spin",total_spin},
            {"fermi_level",fermi_level}})
        if(!std::isfinite(value))
            throw std::invalid_argument(
                std::string("the polar graph input ")+name+" must be finite.");
    if(electric_field.size()!=3
        ||!std::all_of(electric_field.begin(),electric_field.end(),
            [](const double value){return std::isfinite(value);}))
        throw std::invalid_argument(
            "the polar graph input electric_field must carry three finite components.");
    polar_state_.total_charge=total_charge;
    polar_state_.total_spin=total_spin;
    polar_state_.fermi_level=fermi_level;
    polar_state_.electric_field={electric_field[0],electric_field[1],electric_field[2]};
    polar_state_.graph_inputs_set=true;
}

template<typename Precision> bool MacePolarKokkosT<Precision>::polar_graph_inputs_set() const {return polar_state_.graph_inputs_set;}
template<typename Precision> double MacePolarKokkosT<Precision>::polar_total_charge() const {return polar_state_.total_charge;}
template<typename Precision> double MacePolarKokkosT<Precision>::polar_total_spin() const {return polar_state_.total_spin;}
template<typename Precision> double MacePolarKokkosT<Precision>::polar_fermi_level() const {return polar_state_.fermi_level;}
template<typename Precision> std::array<double,3> MacePolarKokkosT<Precision>::polar_electric_field() const {return polar_state_.electric_field;}
template<typename Precision> std::size_t MacePolarKokkosT<Precision>::polar_public_outputs_storage_replacement_count() const {return polar_state_.public_outputs_storage_replacements;}

template<typename Precision>
symmetrix::execution::FamilyCapabilities
MacePolarKokkosT<Precision>::family_capabilities() const
{
    auto capabilities=Backbone::family_capabilities();
    capabilities.reference_ready=
        symmetrix::execution::host_execution_space<Kokkos::DefaultExecutionSpace>;
    capabilities.field=true;
    return capabilities;
}

template<typename Precision>
symmetrix::execution::Program MacePolarKokkosT<Precision>::compile_program() const
{
    symmetrix::execution::polar::DeltaContract delta;
    if(polar_state_.deltas) {
        const auto& contract=polar_state_.deltas->contract();
        delta.node_feature_dimension=contract.node_feature_dimension;
        delta.num_elements=contract.num_elements;
        delta.interaction_count=contract.interaction_count;
        delta.subtract_feature_self_terms=contract.subtract_feature_self_terms;
    }
    return symmetrix::execution::polar::compile_polar_program(
        Backbone::compile_program(),*polar_state_.long_range_contract,delta);
}

template<typename Precision>
symmetrix::execution::ImplementationRegistry
MacePolarKokkosT<Precision>::family_implementation_registry() const
{
    symmetrix::execution::polar::DeltaContract delta;
    if(polar_state_.deltas) {
        const auto& contract=polar_state_.deltas->contract();
        delta.node_feature_dimension=contract.node_feature_dimension;
        delta.num_elements=contract.num_elements;
        delta.interaction_count=contract.interaction_count;
        delta.subtract_feature_self_terms=contract.subtract_feature_self_terms;
    }
    const auto capabilities=family_capabilities();
    return symmetrix::execution::polar::polar_implementation_registry(
        Backbone::family_implementation_registry(),capabilities.target,
        *polar_state_.long_range_contract,capabilities.precision,delta);
}

template<typename Precision>
symmetrix::execution::ImplementationBindingReport
MacePolarKokkosT<Precision>::family_binding_report() const
{
    const auto capabilities=family_capabilities();
    return symmetrix::execution::plan_program(
        compile_program(),family_implementation_registry(),capabilities.target,
        capabilities.precision,this->family_artifact_ready(),
        this->family_artifact_readiness().artifact_id);
}

template<typename Precision>
Kokkos::View<const double*>
MacePolarKokkosT<Precision>::pack_polar_public_outputs_device(
    const double volume,const std::uint64_t graph_generation,
    const bool include_forces,const bool include_stress,
    const bool include_dipole)
{
    auto& chain=polar_deltas();
    const bool zero_isolated_stress=include_stress
        &&chain.solver_decision().resolved
            ==symmetrix::execution::long_range::LongRangeSolver::realspace;
    const std::size_t nodes=this->prepared_node_count();
    const std::size_t force_values=include_forces?3*nodes:0;
    const std::size_t stress_values=include_stress?6:0;
    const std::size_t dipole_values=include_dipole?3:0;
    const std::size_t output_values=
        1+force_values+stress_values+dipole_values;
    if(include_forces)this->reduce_prepared_node_forces(graph_generation);
    if(include_stress&&!zero_isolated_stress)
        this->reduce_prepared_stress(volume,graph_generation);
    if(polar_state_.public_outputs_storage.extent(0)<output_values) {
        polar_state_.public_outputs_storage=Kokkos::View<double*>(
            Kokkos::view_alloc(
                "polar public outputs",Kokkos::WithoutInitializing),
            output_values);
        ++polar_state_.public_outputs_storage_replacements;
    }
    const auto outputs=polar_state_.public_outputs_storage;
    const auto polar_outputs=chain.composed_graph_outputs_device();
    const auto polar_position=chain.position_adjoint_device();
    const auto polar_stress=chain.composed_stress_device();
    auto& execution_space=this->backbone_execution_space();
    Kokkos::parallel_reduce(
        "polar reduce public energy",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(execution_space,0,nodes),
        symmetrix::mace_polar::detail::PublicEnergyReducer<
            decltype(this->node_energies)>{this->node_energies},
        Kokkos::subview(outputs,std::size_t(0)));
    Kokkos::parallel_for(
        "polar pack public payload",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space,0,force_values+stress_values),
        symmetrix::mace_polar::detail::PublicPayloadPacker{
            outputs,this->atom_forces,polar_position,this->stress_tensor,
            polar_stress,force_values,zero_isolated_stress});
    Kokkos::parallel_for(
        "polar add public graph energy",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space,0,1),
        symmetrix::mace_polar::detail::PublicGraphEnergyAdder{
            outputs,polar_outputs});
    if(include_dipole) {
        const std::size_t offset=1+force_values+stress_values;
        Kokkos::parallel_for(
            "polar pack public dipole",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space,0,3),
            symmetrix::mace_polar::detail::PublicDipolePacker{
                outputs,polar_outputs,offset});
    }
    return Kokkos::subview(polar_state_.public_outputs_storage,
        std::make_pair(std::size_t(0),output_values));
}

template<typename Precision>
std::vector<double> MacePolarKokkosT<Precision>::collect_polar_public_outputs(
    const double volume,const std::uint64_t graph_generation,
    const bool include_forces,const bool include_stress,
    const bool include_dipole)
{
    const auto outputs=pack_polar_public_outputs_device(
        volume,graph_generation,include_forces,include_stress,include_dipole);
    const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),outputs);
    return std::vector<double>(host.data(),host.data()+host.extent(0));
}

template<typename Precision>
symmetrix::mace_polar::PreparedDeviceEvaluationResult
MacePolarKokkosT<Precision>::evaluate_prepared_device(
    const symmetrix::mace_polar::PreparedEvaluationRequest& request)
{
    using namespace symmetrix::mace_polar;
    this->clear_layer_product_seed();
    const std::size_t nodes=request.node_types.size();
    if(request.graph_generation==0
        ||request.graph_generation!=this->factorized_graph_generation())
        throw std::invalid_argument(
            "polar whole-model evaluation requires the current prepared "
            "graph token.");
    const bool invalid_position_extent=
        nodes>std::numeric_limits<std::size_t>::max()/3
        ||request.positions.size()!=3*nodes;
    if(nodes==0||invalid_position_extent
        ||request.node_graph_index.size()!=nodes||request.cell.size()!=9
        ||request.graph_pbc.size()!=3)
        throw std::invalid_argument(
            "polar whole-model request extents are inconsistent with exactly "
            "one input system.");
    for(const int graph:request.node_graph_index)
        if(graph!=0)
            throw std::invalid_argument(
                "polar whole-model evaluation requires every node graph "
                "index to be zero; mixed or multi-system batches are "
                "unsupported.");
    const auto finite=[](const double value) {return std::isfinite(value);};
    if(!std::all_of(request.positions.begin(),request.positions.end(),finite)
        ||!std::all_of(request.cell.begin(),request.cell.end(),finite)
        ||!std::all_of(request.external_potential.begin(),
            request.external_potential.end(),finite)
        ||!finite(request.total_charge)||!finite(request.total_spin))
        throw std::invalid_argument(
            "polar whole-model request contains a non-finite input.");

    this->validate_factorized_prepared_system(
        request.graph_generation,request.node_types,request.cell,
        request.graph_pbc);
    auto& chain=this->polar_deltas();
    const std::array<int,3> pbc{
        request.graph_pbc[0],request.graph_pbc[1],request.graph_pbc[2]};
    const auto& long_range=chain.contract().long_range;
    const auto route=symmetrix::execution::long_range::resolve_long_range_solver(
        request.solver_request,pbc,long_range.density_max_l,
        long_range.feature_max_l,polar_solver_capabilities());
    if(!route.accepted)
        throw std::invalid_argument(route.error);
    const bool periodic_route=route.resolved
            ==symmetrix::execution::long_range::LongRangeSolver::direct_reciprocal
        ||route.resolved
            ==symmetrix::execution::long_range::LongRangeSolver::pme;
    const double direct_volume=periodic_route
        ?this->factorized_prepared_cell_volume(request.graph_generation)
        :1.0;
    const double reporting_volume=
        request.include_stress?direct_volume:1.0;

    ++whole_model_transaction_count_;
    try {
        const std::size_t forward_before=
            this->factorized_forward_evaluation_count();
        const std::size_t reverse_before=
            this->factorized_reverse_evaluation_count();
        const std::size_t prepared_before=
            this->factorized_prepared_evaluation_count();
        const std::size_t fallback_before=
            this->factorized_fallback_evaluation_count();
        const std::size_t polar_fallback_before=
            chain.e3linear_diagnostics().fallback_count;
        const std::size_t polar_reverse_before=chain.resident_reverse_count();
        this->set_retain_layer_features(true);
        this->compute_prepared_factorized_positions(
            request.graph_generation,request.positions);

        typename symmetrix::mace_polar::PolarDeltaChainKokkosT<Precision>::Inputs
            inputs;
        inputs.solver_request=request.solver_request;
        const auto& retained=this->layer_features();
        if(retained.size()!=static_cast<std::size_t>(
                chain.interaction_count()))
            throw std::invalid_argument(
                "polar whole-model evaluation requires one retained backbone "
                "feature view per interaction layer.");
        inputs.node_feature_views.reserve(retained.size());
        for(const auto& layer:retained)
            inputs.node_feature_views.emplace_back(layer);
        inputs.node_types.assign(
            request.node_types.begin(),request.node_types.end());
        inputs.positions.assign(
            request.positions.begin(),request.positions.end());
        inputs.cell.assign(request.cell.begin(),request.cell.end());
        inputs.graph_pbc.assign(
            request.graph_pbc.begin(),request.graph_pbc.end());
        inputs.node_graph_index.assign(
            request.node_graph_index.begin(),request.node_graph_index.end());
        inputs.total_charge={request.total_charge};
        inputs.total_spin={request.total_spin};
        inputs.external_potential.assign(
            request.external_potential.begin(),
            request.external_potential.end());
        inputs.pme_mesh=request.pme_mesh;
        inputs.pme_spline_order=request.pme_spline_order;

        const auto composed=chain.forward_composed_energy(
            inputs,/*materialize_public_state=*/false,
            /*materialize_outputs=*/false);
        const bool reverse=request.include_forces||request.include_stress;
        if(reverse) {
            const auto reverse_result=chain.reverse_composed_energy(
                inputs,composed.state,
                /*materialize_learned_adjoints=*/false,
                /*materialize_input_adjoints=*/false);
            (void)reverse_result;
            Kokkos::View<const Precision**,Kokkos::LayoutRight> flat_seed=
                chain.node_feature_seed();
            this->set_layer_product_seed_flat(flat_seed);
            ++whole_model_reverse_count_;
            this->compute_prepared_factorized_positions(
                request.graph_generation,request.positions);
        }

        const auto packed=this->pack_polar_public_outputs_device(
            reporting_volume,request.graph_generation,
            request.include_forces,request.include_stress,
            request.include_dipole);
        PreparedDeviceEvaluationResult result;
        result.public_outputs=packed;
        result.layout.forces_elements=request.include_forces?3*nodes:0;
        result.layout.stress_offset=
            result.layout.forces_offset+result.layout.forces_elements;
        result.layout.stress_elements=request.include_stress?6:0;
        result.layout.dipole_offset=
            result.layout.stress_offset+result.layout.stress_elements;
        result.layout.dipole_elements=request.include_dipole?3:0;
        result.layout.elements=
            result.layout.dipole_offset+result.layout.dipole_elements;

        const auto& decision=chain.solver_decision();
        const auto reciprocal=chain.reciprocal_allocation_usage();
        const auto realspace=chain.realspace_retained_allocation_usage();
        result.diagnostics.requested_solver=decision.requested;
        result.diagnostics.resolved_solver=decision.resolved;
        result.diagnostics.solver_reason=decision.reason;
        result.diagnostics.automatic_atom_count=chain.automatic_nodes();
        result.diagnostics.automatic_reciprocal_mode_count=
            chain.automatic_reciprocal_modes();
        result.diagnostics.automatic_reciprocal_mode_count_is_lower_bound=
            chain.automatic_reciprocal_modes_are_lower_bound();
        if(result.diagnostics.automatic_reciprocal_mode_count!=0
            &&result.diagnostics.automatic_atom_count
                >std::numeric_limits<std::size_t>::max()
                    /result.diagnostics.automatic_reciprocal_mode_count)
            throw std::overflow_error(
                "MACE Polar automatic atom-mode diagnostic overflows size_t.");
        result.diagnostics.automatic_atom_mode_pair_count=
            result.diagnostics.automatic_atom_count
            *result.diagnostics.automatic_reciprocal_mode_count;
        result.diagnostics.n_k=chain.workspace_n_k();
        result.diagnostics.reciprocal_allocation_elements=
            reciprocal.elements;
        result.diagnostics.reciprocal_allocation_bytes=reciprocal.bytes;
        if(result.diagnostics.n_k!=0
            &&nodes>std::numeric_limits<std::size_t>::max()
                /result.diagnostics.n_k)
            throw std::overflow_error(
                "MACE Polar reciprocal table element count overflows size_t.");
        result.diagnostics.reciprocal_table_elements=
            result.diagnostics.n_k*nodes;
        if(decision.resolved
            == symmetrix::execution::long_range::LongRangeSolver::pme) {
            const auto mesh=chain.particle_mesh_usage();
            const auto adapter=chain.particle_mesh_adapter_allocation_usage();
            const auto adapter_bytes=static_cast<std::uint64_t>(adapter.bytes);
            if(mesh.active_bytes
                    >std::numeric_limits<std::uint64_t>::max()-adapter_bytes
                ||mesh.capacity_bytes
                    >std::numeric_limits<std::uint64_t>::max()-adapter_bytes)
                throw std::overflow_error(
                    "MACE Polar particle-mesh byte diagnostics overflow uint64_t.");
            result.diagnostics.particle_mesh_active_bytes=
                mesh.active_bytes+adapter_bytes;
            result.diagnostics.particle_mesh_capacity_bytes=
                mesh.capacity_bytes+adapter_bytes;
            std::uint64_t volume=1;
            for(const auto extent:mesh.mesh) {
                const auto value=static_cast<std::uint64_t>(extent);
                if(value!=0 && volume>std::numeric_limits<std::uint64_t>::max()/value)
                    throw std::overflow_error(
                        "MACE Polar particle-mesh volume overflows uint64_t.");
                volume*=value;
            }
            result.diagnostics.particle_mesh_volume=volume;
            result.diagnostics.particle_mesh_shape=chain.effective_pme_mesh();
            result.diagnostics.particle_mesh_rebuilds=mesh.rebuilds;
            result.diagnostics.particle_mesh_reuses=mesh.reuses;
            result.diagnostics.particle_mesh_forward_plan_builds=
                mesh.forward_plan_builds;
            result.diagnostics.particle_mesh_forward_plan_reuses=
                mesh.forward_plan_reuses;
            result.diagnostics.particle_mesh_inverse_plan_builds=
                mesh.inverse_plan_builds;
            result.diagnostics.particle_mesh_inverse_plan_reuses=
                mesh.inverse_plan_reuses;
        }
        result.diagnostics.realspace_retained_elements=realspace.elements;
        result.diagnostics.realspace_retained_bytes=realspace.bytes;
        result.diagnostics.diagnostic_materialization_count=
            chain.diagnostic_state_materialization_count();
        result.diagnostics.public_output_elements=result.layout.elements;
        result.diagnostics.backbone_fallback_count=
            this->factorized_fallback_evaluation_count()-fallback_before;
        result.diagnostics.polar_fallback_count=
            chain.e3linear_diagnostics().fallback_count
            -polar_fallback_before;
        result.diagnostics.fallback_count=
            result.diagnostics.backbone_fallback_count
            +result.diagnostics.polar_fallback_count;
        result.diagnostics.reverse_strategy=std::string(
            chain.realspace_reverse_strategy_resolved());
        result.diagnostics.cumulative_transaction_count=
            whole_model_transaction_count_;
        result.diagnostics.cumulative_polar_reverse_count=
            whole_model_reverse_count_;
        result.diagnostics.backbone_evaluation_count=
            this->factorized_prepared_evaluation_count()-prepared_before;
        result.diagnostics.backbone_forward_layer_count=
            this->factorized_forward_evaluation_count()-forward_before;
        result.diagnostics.backbone_reverse_layer_count=
            this->factorized_reverse_evaluation_count()-reverse_before;
        if(result.diagnostics.backbone_evaluation_count==0)
            throw std::logic_error(
                "polar whole-model evaluation completed without a backbone "
                "transaction.");
        result.diagnostics.seeded_backbone_evaluation_count=
            result.diagnostics.backbone_evaluation_count-1;
        result.diagnostics.polar_reverse_count=
            chain.resident_reverse_count()-polar_reverse_before;
        result.diagnostics.reverse_executed=reverse;
        this->clear_layer_product_seed();
        result.diagnostics.product_seed_cleared=
            !this->has_layer_product_seed();
        return result;
    } catch(...) {
        this->clear_layer_product_seed();
        ++whole_model_failure_count_;
        throw;
    }
}

template<typename Precision>
symmetrix::mace_polar::PreparedEvaluationResult
MacePolarKokkosT<Precision>::evaluate_prepared(
    const symmetrix::mace_polar::PreparedEvaluationRequest& request)
{
    const auto device=evaluate_prepared_device(request);
    const auto host=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),device.public_outputs);
    symmetrix::mace_polar::PreparedEvaluationResult result;
    result.energy=host(device.layout.energy_offset);
    result.free_energy=result.energy;
    if(device.layout.forces_elements!=0)
        result.forces.assign(
            host.data()+device.layout.forces_offset,
            host.data()+device.layout.forces_offset
                +device.layout.forces_elements);
    if(device.layout.stress_elements!=0)
        result.stress.assign(
            host.data()+device.layout.stress_offset,
            host.data()+device.layout.stress_offset
                +device.layout.stress_elements);
    if(device.layout.dipole_elements!=0)
        result.dipole.assign(
            host.data()+device.layout.dipole_offset,
            host.data()+device.layout.dipole_offset
                +device.layout.dipole_elements);
    result.diagnostics=device.diagnostics;
    return result;
}

using MacePolarKokkos = MacePolarKokkosT<double>;
using MacePolarFloatKokkos = MacePolarKokkosT<float>;
