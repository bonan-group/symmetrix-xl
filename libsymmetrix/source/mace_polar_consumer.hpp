#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "long_range_solver_policy.hpp"
#include "mace_polar_kokkos.hpp"
#include "nlohmann/json.hpp"

namespace symmetrix::mace_polar {

struct ConsumerSystemRequest {
    std::span<const int> atomic_numbers;
    std::span<const double> positions;
    std::span<const double> cell;
    std::span<const int> pbc;
    double total_charge=0.0;
    double total_spin=0.0;
    double fermi_level=0.0;
    std::array<double,3> electric_field{0.0,0.0,0.0};
    execution::long_range::LongRangeSolverRequest solver_request=
        execution::long_range::LongRangeSolverRequest::automatic;
    bool include_forces=false;
    bool include_stress=false;
    bool include_dipole=false;
};

inline std::array<double,4> consumer_scientific_external_potential(
    const ConsumerSystemRequest& request)
{
    // The exported model retains Fermi level as graph metadata, but the
    // frozen POLAR program couples only the electric-field lanes.
    return {0.0,request.electric_field[0],request.electric_field[1],
        request.electric_field[2]};
}

struct ConsumerPreparedGraphRequest {
    std::span<const int> num_neigh;
    std::span<const int> sources;
    std::span<const int> source_types;
    std::span<const double> reference_xyz;
};

struct ConsumerDiagnostics {
    PreparedEvaluationDiagnostics evaluation;
    std::size_t transaction_count=0;
    std::size_t reverse_count=0;
    std::size_t graph_generation=0;
    std::size_t graph_rebuild_count=0;
    std::size_t graph_reuse_count=0;
    std::size_t topology_validation_count=0;
    std::size_t topology_validation_edge_count=0;
    std::size_t topology_reuse_scan_count=0;
    std::size_t topology_facade_scan_count=0;
    std::size_t geometry_update_count=0;
    std::size_t geometry_publish_count=0;
    std::size_t geometry_reuse_count=0;
    std::size_t geometry_allocation_count=0;
    std::size_t geometry_allocation_bytes=0;
    std::size_t public_output_allocation_count=0;
    std::size_t active_nodes=0;
    std::size_t active_edges=0;
    std::size_t input_d2h_bytes=0;
    std::size_t input_h2d_bytes=0;
    std::size_t topology_input_d2h_bytes=0;
    std::size_t metadata_d2h_bytes=0;
    std::size_t public_output_d2h_bytes=0;
    std::size_t scalar_output_d2h_bytes=0;
    std::size_t virial_output_d2h_bytes=0;
    std::size_t force_output_d2h_bytes=0;
    std::size_t output_h2d_bytes=0;
    std::size_t retained_output_elements=0;
};

struct ConsumerResult {
    double energy=0.0;
    double free_energy=0.0;
    std::vector<double> forces;
    std::vector<double> stress;
    std::vector<double> dipole;
    ConsumerDiagnostics diagnostics;
};

struct ConsumerDeviceResult {
    Kokkos::View<const double*> public_outputs;
    PreparedOutputLayout layout;
    ConsumerDiagnostics diagnostics;
};

template<class Source,class Destination>
struct PolarPublicOutputCopier {
    Source source;
    Destination destination;
    std::size_t source_offset;

    KOKKOS_FUNCTION
    void operator()(const std::size_t i) const
    {
        destination(i)=source(source_offset+i);
    }
};

template<class Source,class Destination>
void copy_polar_public_output_device(
    const Source& source,const std::size_t source_offset,
    const std::size_t elements,const Destination& destination,
    const char* name)
{
    if(destination.extent(0)<elements)
        throw std::invalid_argument(
            std::string(name)+" output view is too small.");
    Kokkos::parallel_for(
        std::string("copy ")+name,
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0,elements),
        PolarPublicOutputCopier<Source,Destination>{
            source,destination,source_offset});
}

template<class PublicOutputView,class NodeIndexView,class ForceView>
void add_polar_public_forces_device(
    const PublicOutputView& public_outputs,const std::size_t force_offset,
    const NodeIndexView& node_indices,const ForceView& forces,
    const std::size_t nodes)
{
    if(node_indices.extent(0)<nodes||forces.extent(0)<nodes)
        throw std::invalid_argument(
            "MACE Polar force writeback views are too small.");
    Kokkos::parallel_for(
        "MACE Polar consumer force writeback",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0,nodes),
        KOKKOS_LAMBDA(const std::size_t node) {
            const int atom=node_indices(node);
            for(std::size_t component=0;component<3;++component)
                forces(atom,component)+=
                    public_outputs(force_offset+3*node+component);
        });
}

template<class StressView,class VirialView>
void map_polar_stress_to_virial_device(
    const StressView& stress,const double volume,const bool periodic,
    const VirialView& virial)
{
    if(stress.extent(0)<6||virial.extent(0)<6)
        throw std::invalid_argument(
            "MACE Polar stress/virial views require six elements.");
    Kokkos::parallel_for(
        "MACE Polar consumer virial mapping",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,6),
        KOKKOS_LAMBDA(const int component) {
            if(!periodic) {
                virial(component)=0.0;
                return;
            }
            const int source=component<3?component:
                (component==3?5:(component==4?4:3));
            virial(component)=-volume*stress(source);
        });
}

template<typename Precision>
class MacePolarConsumerT {
public:
    using Evaluator=MacePolarKokkosT<Precision>;

    explicit MacePolarConsumerT(
        const std::string& filename,const std::string& requested_head={})
        :evaluator_(filename,requested_head)
    {
        evaluator_.set_streamed_edges("direct_streamed");
        const auto values=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),evaluator_.atomic_numbers);
        model_atomic_numbers_.reserve(values.extent(0));
        for(std::size_t index=0;index<values.extent(0);++index)
            model_atomic_numbers_.push_back(values(index));
        metadata_d2h_bytes_=checked_bytes(
            values.size(),sizeof(int),"model atomic-number metadata");
    }

    double cutoff() const { return evaluator_.polar_cutoff(); }
    std::string streamed_edges_mode() const {
        const auto mode=evaluator_.polar_streamed_edges_mode();
        return mode=="direct"?"direct_streamed":mode;
    }
    void preflight_solver(
        const execution::long_range::LongRangeSolverRequest request,
        const std::array<int,3>& pbc) const
    {
        (void)evaluator_.preflight_polar_solver(request,pbc);
    }
    const ConsumerDiagnostics& diagnostics() const { return diagnostics_; }
    void arm_graph_geometry_failure_for_testing() {
        evaluator_.arm_polar_geometry_prepare_failure_for_testing();
    }

    void set_streamed_edges(const std::string& mode)
    {
        evaluator_.set_streamed_edges(mode);
    }

    bool load_artifact(const std::string& path,const std::string& target)
    {
        install_artifact_contract_identity(path);
        symmetrix::execution::ArtifactRequest request;
        request.path=path;
        request.target=target;
        return evaluator_.load_family_artifact(request).ready;
    }

    bool load_product_artifact(
        const int layer,const std::string& kind,const std::string& path,
        const std::string& metadata_json,const std::string& schedule,
        const int persistent_blocks_per_compute_unit,const std::string& backend)
    {
        evaluator_.load_execution_mh1_product_module(
            layer,kind,path,metadata_json,schedule,
            persistent_blocks_per_compute_unit,backend);
        return evaluator_.execution_mh1_product_module_ready(layer);
    }

    bool product_artifacts_ready() const
    {
        return evaluator_.execution_mh1_product_modules_ready();
    }

    // LAMMPS owns neighbor-list lifetime.  Invalidate the prepared graph when
    // LAMMPS rebuilds that topology, even if the resulting edge identity is
    // numerically unchanged, so the native generation follows its owner.
    void invalidate_graph() { graph_generation_=0; }

    ConsumerResult evaluate(const ConsumerSystemRequest& request)
    {
        const auto device=evaluate_device(request);
        const auto host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),device.public_outputs);
        ConsumerResult result;
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
        result.diagnostics.public_output_d2h_bytes=
            checked_bytes(device.layout.elements,sizeof(double),
                "public output");
        result.diagnostics.force_output_d2h_bytes=
            checked_bytes(device.layout.forces_elements,sizeof(double),
                "public force output");
        result.diagnostics.scalar_output_d2h_bytes=
            checked_bytes(
                device.layout.elements-device.layout.forces_elements,
                sizeof(double),"public scalar output");
        diagnostics_=result.diagnostics;
        return result;
    }

    ConsumerDeviceResult evaluate_device(const ConsumerSystemRequest& request)
    {
        const std::size_t input_h2d_before=evaluator_.polar_input_h2d_bytes();
        validate_request(request);
        // Solver admission precedes species mapping, graph scanning, and every
        // graph/geometry/workspace allocation.
        const std::array<int,3> pbc{
            request.pbc[0],request.pbc[1],request.pbc[2]};
        (void)evaluator_.preflight_polar_solver(request.solver_request,pbc);
        const auto node_types=map_atomic_numbers(request.atomic_numbers);
        const auto graph=build_graph(node_types,request);
        const bool same_graph=graph_generation_!=0
            &&node_types==node_types_
            &&std::equal(request.cell.begin(),request.cell.end(),cell_.begin())
            &&std::equal(request.pbc.begin(),request.pbc.end(),pbc_.begin())
            &&graph.num_neigh==num_neigh_
            &&graph.sources==sources_
            &&graph.source_types==source_types_;
        if(!same_graph) {
            const auto generation=evaluator_.replace_factorized_graph_geometry(
                static_cast<int>(node_types.size()),node_types,graph.num_neigh,
                graph.sources,graph.source_types,request.positions,graph.xyz,
                request.cell,graph.inverse_cell,request.pbc);
            graph_generation_=generation;
            node_types_=node_types;
            cell_.assign(request.cell.begin(),request.cell.end());
            pbc_.assign(request.pbc.begin(),request.pbc.end());
            num_neigh_=graph.num_neigh;
            sources_=graph.sources;
            source_types_=graph.source_types;
            ++graph_rebuild_count_;
            ++geometry_publish_count_;
        } else {
            ++graph_reuse_count_;
            if(std::equal(request.positions.begin(),request.positions.end(),
                    positions_.begin()))
                ++geometry_reuse_count_;
            else
                ++geometry_update_count_;
        }
        positions_.assign(request.positions.begin(),request.positions.end());
        auto result=execute_prepared(request,node_types,graph.sources.size());
        result.diagnostics.input_h2d_bytes=
            evaluator_.polar_input_h2d_bytes()-input_h2d_before;
        diagnostics_=result.diagnostics;
        return result;
    }

    ConsumerDeviceResult evaluate_prepared_device(
        const ConsumerSystemRequest& request,
        const ConsumerPreparedGraphRequest& graph,const bool replace_graph)
    {
        const std::size_t input_h2d_before=evaluator_.polar_input_h2d_bytes();
        validate_request(request);
        const std::array<int,3> pbc{
            request.pbc[0],request.pbc[1],request.pbc[2]};
        (void)evaluator_.preflight_polar_solver(request.solver_request,pbc);
        const auto node_types=map_atomic_numbers(request.atomic_numbers);
        const bool must_replace=replace_graph||graph_generation_==0
            ||node_types!=node_types_
            ||!std::equal(request.cell.begin(),request.cell.end(),cell_.begin())
            ||!std::equal(request.pbc.begin(),request.pbc.end(),pbc_.begin());
        if(must_replace) {
            validate_prepared_graph_extents(node_types.size(),graph);
            const auto inverse=invert_cell(request.cell);
            graph_generation_=evaluator_.replace_factorized_graph_geometry(
                static_cast<int>(node_types.size()),node_types,graph.num_neigh,
                graph.sources,graph.source_types,request.positions,
                graph.reference_xyz,request.cell,inverse,request.pbc);
            node_types_=node_types;
            cell_.assign(request.cell.begin(),request.cell.end());
            pbc_.assign(request.pbc.begin(),request.pbc.end());
            num_neigh_.assign(
                graph.num_neigh.begin(),graph.num_neigh.end());
            sources_.assign(graph.sources.begin(),graph.sources.end());
            source_types_.assign(
                graph.source_types.begin(),graph.source_types.end());
            reference_xyz_.assign(
                graph.reference_xyz.begin(),graph.reference_xyz.end());
            atomic_numbers_.assign(
                request.atomic_numbers.begin(),request.atomic_numbers.end());
            ++graph_rebuild_count_;
            ++geometry_publish_count_;
        } else {
            ++graph_reuse_count_;
            if(positions_.size()==request.positions.size()
                &&std::equal(request.positions.begin(),request.positions.end(),
                    positions_.begin()))
                ++geometry_reuse_count_;
            else
                ++geometry_update_count_;
        }
        positions_.assign(request.positions.begin(),request.positions.end());
        auto result=execute_prepared(request,node_types,sources_.size());
        result.diagnostics.input_h2d_bytes=
            evaluator_.polar_input_h2d_bytes()-input_h2d_before;
        diagnostics_=result.diagnostics;
        return result;
    }

    template<class AtomicNumberView,class PositionView,class CellView,
        class PbcView,class ForceView,class StressView,class DipoleView>
    ConsumerResult evaluate_views(
        const AtomicNumberView& atomic_numbers,const PositionView& positions,
        const CellView& cell,const PbcView& pbc,
        const double total_charge,const double total_spin,
        const double fermi_level,const std::array<double,3>& electric_field,
        const execution::long_range::LongRangeSolverRequest solver_request,
        const bool include_forces,const bool include_stress,
        const bool include_dipole,const ForceView& forces,
        const StressView& stress,const DipoleView& dipole)
    {
        static_assert(AtomicNumberView::rank==1&&PositionView::rank==1
            &&CellView::rank==1&&PbcView::rank==1&&ForceView::rank==1
            &&StressView::rank==1&&DipoleView::rank==1);
        const auto host_atomic=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),atomic_numbers);
        const auto host_positions=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),positions);
        const auto host_cell=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),cell);
        const auto host_pbc=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),pbc);
        std::vector<int> atomic(host_atomic.extent(0));
        std::vector<double> position_values(host_positions.extent(0));
        std::vector<double> cell_values(host_cell.extent(0));
        std::vector<int> pbc_values(host_pbc.extent(0));
        for(std::size_t i=0;i<atomic.size();++i)atomic[i]=host_atomic(i);
        for(std::size_t i=0;i<position_values.size();++i)
            position_values[i]=host_positions(i);
        for(std::size_t i=0;i<cell_values.size();++i)cell_values[i]=host_cell(i);
        for(std::size_t i=0;i<pbc_values.size();++i)pbc_values[i]=host_pbc(i);
        ConsumerSystemRequest request{
            atomic,position_values,cell_values,pbc_values,total_charge,total_spin,
            fermi_level,electric_field,solver_request,include_forces,
            include_stress,include_dipole};
        const auto device=evaluate_device(request);
        ConsumerResult result;
        result.diagnostics=device.diagnostics;
        result.diagnostics.input_d2h_bytes=checked_add_bytes(
            result.diagnostics.input_d2h_bytes,
            checked_add_bytes(
                checked_bytes(atomic.size(),sizeof(int),"atomic-number input"),
                checked_bytes(position_values.size(),sizeof(double),
                    "position input"),
                "compact input"),
            "compact input");
        result.diagnostics.input_d2h_bytes=checked_add_bytes(
            result.diagnostics.input_d2h_bytes,
            checked_add_bytes(
                checked_bytes(cell_values.size(),sizeof(double),"cell input"),
                checked_bytes(pbc_values.size(),sizeof(int),"PBC input"),
                "compact input"),
            "compact input");
        if(include_forces)copy_polar_public_output_device(
            device.public_outputs,device.layout.forces_offset,
            device.layout.forces_elements,forces,"Polar force");
        if(include_stress)copy_polar_public_output_device(
            device.public_outputs,device.layout.stress_offset,
            device.layout.stress_elements,stress,"Polar stress");
        if(include_dipole)copy_polar_public_output_device(
            device.public_outputs,device.layout.dipole_offset,
            device.layout.dipole_elements,dipole,"Polar dipole");
        const auto host_energy=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),Kokkos::subview(
                device.public_outputs,device.layout.energy_offset));
        result.energy=host_energy();
        result.free_energy=result.energy;
        result.diagnostics.public_output_d2h_bytes=sizeof(double);
        result.diagnostics.force_output_d2h_bytes=0;
        result.diagnostics.output_h2d_bytes=0;
        result.diagnostics.scalar_output_d2h_bytes=sizeof(double);
        diagnostics_=result.diagnostics;
        return result;
    }

    template<class AtomicNumberView,class PositionView,class CellView,
        class PbcView,class DegreeView,class SourceView,class SourceTypeView,
        class ReferenceView,class ForceView,class StressView,class DipoleView>
    ConsumerResult evaluate_prepared_views(
        const AtomicNumberView& atomic_numbers,const PositionView& positions,
        const CellView& cell,const PbcView& pbc,const DegreeView& num_neigh,
        const SourceView& sources,const SourceTypeView& source_types,
        const ReferenceView& reference_xyz,const bool replace_graph,
        const double total_charge,const double total_spin,
        const double fermi_level,const std::array<double,3>& electric_field,
        const execution::long_range::LongRangeSolverRequest solver_request,
        const bool include_forces,const bool include_stress,
        const bool include_dipole,const ForceView& forces,
        const StressView& stress,const DipoleView& dipole)
    {
        static_assert(AtomicNumberView::rank==1&&PositionView::rank==1
            &&CellView::rank==1&&PbcView::rank==1&&DegreeView::rank==1
            &&SourceView::rank==1&&SourceTypeView::rank==1
            &&ReferenceView::rank==1&&ForceView::rank==1
            &&StressView::rank==1&&DipoleView::rank==1);
        const auto host_positions=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),positions);
        const auto host_cell=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),cell);
        const auto host_pbc=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),pbc);
        std::vector<double> position_values(
            host_positions.data(),host_positions.data()+host_positions.size());
        std::vector<double> cell_values(
            host_cell.data(),host_cell.data()+host_cell.size());
        std::vector<int> pbc_values(
            host_pbc.data(),host_pbc.data()+host_pbc.size());
        std::vector<int> atomic;
        std::vector<int> degrees;
        std::vector<int> source_values;
        std::vector<int> source_type_values;
        std::vector<double> reference_values;
        std::size_t topology_bytes=0;
        if(replace_graph||graph_generation_==0) {
            const auto host_atomic=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),atomic_numbers);
            const auto host_degrees=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),num_neigh);
            const auto host_sources=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),sources);
            const auto host_types=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),source_types);
            const auto host_reference=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),reference_xyz);
            atomic.assign(host_atomic.data(),host_atomic.data()+host_atomic.size());
            degrees.assign(
                host_degrees.data(),host_degrees.data()+host_degrees.size());
            source_values.assign(
                host_sources.data(),host_sources.data()+host_sources.size());
            source_type_values.assign(
                host_types.data(),host_types.data()+host_types.size());
            // The device consumer supplies stable atomic numbers so a LAMMPS
            // type order never becomes part of the exported model contract.
            // Convert them once, only when LAMMPS publishes new topology.
            source_type_values=map_atomic_numbers(source_type_values);
            reference_values.assign(
                host_reference.data(),host_reference.data()+host_reference.size());
            const auto integer_topology=checked_add_bytes(
                checked_add_bytes(atomic.size(),degrees.size(),"topology"),
                checked_add_bytes(source_values.size(),source_type_values.size(),
                    "topology"),
                "topology");
            topology_bytes=checked_add_bytes(
                checked_bytes(integer_topology,sizeof(int),"topology"),
                checked_bytes(reference_values.size(),sizeof(double),
                    "reference geometry"),
                "topology");
        } else {
            atomic=atomic_numbers_;
        }
        ConsumerSystemRequest request{
            atomic,position_values,cell_values,pbc_values,total_charge,total_spin,
            fermi_level,electric_field,solver_request,include_forces,
            include_stress,include_dipole};
        ConsumerPreparedGraphRequest graph{
            degrees,source_values,source_type_values,reference_values};
        const auto device=evaluate_prepared_device(request,graph,replace_graph);
        ConsumerResult result;
        result.diagnostics=device.diagnostics;
        const auto compact_geometry_bytes=checked_add_bytes(
            checked_bytes(position_values.size(),sizeof(double),
                "position input"),
            checked_add_bytes(
                checked_bytes(cell_values.size(),sizeof(double),"cell input"),
                checked_bytes(pbc_values.size(),sizeof(int),"PBC input"),
                "compact geometry input"),
            "compact geometry input");
        result.diagnostics.input_d2h_bytes=checked_add_bytes(
            result.diagnostics.input_d2h_bytes,
            checked_add_bytes(compact_geometry_bytes,topology_bytes,
                "prepared input"),
            "prepared input");
        result.diagnostics.topology_input_d2h_bytes=checked_add_bytes(
            result.diagnostics.topology_input_d2h_bytes,topology_bytes,
            "topology input");
        if(include_forces)copy_polar_public_output_device(
            device.public_outputs,device.layout.forces_offset,
            device.layout.forces_elements,forces,"Polar force");
        if(include_stress)copy_polar_public_output_device(
            device.public_outputs,device.layout.stress_offset,
            device.layout.stress_elements,stress,"Polar stress");
        if(include_dipole)copy_polar_public_output_device(
            device.public_outputs,device.layout.dipole_offset,
            device.layout.dipole_elements,dipole,"Polar dipole");
        const auto host_energy=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),Kokkos::subview(
                device.public_outputs,device.layout.energy_offset));
        result.energy=host_energy();
        result.free_energy=result.energy;
        result.diagnostics.public_output_d2h_bytes=sizeof(double);
        result.diagnostics.scalar_output_d2h_bytes=sizeof(double);
        result.diagnostics.force_output_d2h_bytes=0;
        diagnostics_=result.diagnostics;
        return result;
    }

private:
    static std::size_t checked_bytes(
        const std::size_t elements,const std::size_t element_size,
        const char* what)
    {
        if(element_size!=0
            &&elements>std::numeric_limits<std::size_t>::max()/element_size)
            throw std::overflow_error(
                std::string("MACE Polar ")+what+" byte count overflows size_t.");
        return elements*element_size;
    }

    static std::size_t checked_add_bytes(
        const std::size_t left,const std::size_t right,const char* what)
    {
        if(left>std::numeric_limits<std::size_t>::max()-right)
            throw std::overflow_error(
                std::string("MACE Polar ")+what+" byte count overflows size_t.");
        return left+right;
    }

    ConsumerDeviceResult execute_prepared(
        const ConsumerSystemRequest& request,const std::vector<int>& node_types,
        const std::size_t active_edges)
    {
        std::vector<int> graph_index(node_types.size(),0);
        PreparedEvaluationRequest prepared;
        prepared.graph_generation=graph_generation_;
        prepared.node_types=node_types;
        prepared.positions=request.positions;
        prepared.cell=request.cell;
        prepared.graph_pbc=request.pbc;
        prepared.node_graph_index=graph_index;
        prepared.total_charge=request.total_charge;
        prepared.total_spin=request.total_spin;
        prepared.external_potential=
            consumer_scientific_external_potential(request);
        prepared.include_forces=request.include_forces;
        prepared.include_stress=request.include_stress;
        prepared.include_dipole=request.include_dipole;
        prepared.solver_request=request.solver_request;
        auto native=evaluator_.evaluate_prepared_device(prepared);
        ConsumerDeviceResult result;
        result.public_outputs=native.public_outputs;
        result.layout=native.layout;
        result.diagnostics.evaluation=std::move(native.diagnostics);
        result.diagnostics.transaction_count=evaluator_.whole_model_transaction_count();
        result.diagnostics.reverse_count=evaluator_.whole_model_reverse_count();
        result.diagnostics.graph_generation=graph_generation_;
        result.diagnostics.graph_rebuild_count=graph_rebuild_count_;
        result.diagnostics.graph_reuse_count=graph_reuse_count_;
        result.diagnostics.topology_validation_count=
            evaluator_.polar_topology_validation_count();
        result.diagnostics.topology_validation_edge_count=
            evaluator_.polar_topology_validated_edge_count();
        result.diagnostics.topology_reuse_scan_count=
            topology_reuse_scan_count_;
        result.diagnostics.topology_facade_scan_count=
            topology_facade_scan_count_;
        result.diagnostics.geometry_update_count=geometry_update_count_;
        result.diagnostics.geometry_publish_count=geometry_publish_count_;
        result.diagnostics.geometry_reuse_count=geometry_reuse_count_;
        result.diagnostics.geometry_allocation_count=
            evaluator_.polar_geometry_allocation_count();
        result.diagnostics.geometry_allocation_bytes=
            evaluator_.polar_geometry_workspace_bytes();
        result.diagnostics.public_output_allocation_count=
            evaluator_.polar_public_output_allocation_count();
        result.diagnostics.active_nodes=node_types.size();
        result.diagnostics.active_edges=active_edges;
        result.diagnostics.metadata_d2h_bytes=metadata_d2h_bytes_;
        result.diagnostics.retained_output_elements=
            result.diagnostics.evaluation.diagnostic_materialization_count;
        diagnostics_=result.diagnostics;
        return result;
    }

    static void validate_prepared_graph_extents(const std::size_t nodes,
        const ConsumerPreparedGraphRequest& graph)
    {
        if(graph.num_neigh.size()!=nodes
            ||graph.sources.size()!=graph.source_types.size()
            ||graph.sources.size()>static_cast<std::size_t>(
                std::numeric_limits<int>::max())
            ||graph.sources.size()>std::numeric_limits<std::size_t>::max()/3
            ||graph.reference_xyz.size()!=3*graph.sources.size())
            throw std::invalid_argument(
                "MACE Polar prepared graph extents are inconsistent.");
    }

    void install_artifact_contract_identity(const std::string& artifact_path)
    {
        const auto manifest_path=
            std::filesystem::path(artifact_path).parent_path()/"manifest.json";
        std::ifstream stream(manifest_path);
        if(!stream)
            throw std::invalid_argument(
                "MACE Polar artifact manifest is not readable: "
                +manifest_path.string());
        nlohmann::json manifest;
        try { stream>>manifest; }
        catch(const nlohmann::json::exception& exception) {
            throw std::invalid_argument(
                std::string("MACE Polar artifact manifest is malformed: ")
                +exception.what());
        }
        try {
            const auto& contract=manifest.at("key_inputs").at("build")
                .at("contract");
            if(contract.at("tag").get<std::string>()
                !="symmetrix.execution.mh1.node-metadata/1")
                throw std::invalid_argument(
                    "MACE Polar artifact node contract tag is unsupported.");
            const auto interactions=contract.at("layers").size();
            if(interactions==0
                ||interactions>std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument(
                    "MACE Polar artifact interaction count is invalid.");
            evaluator_.set_execution_mh1_v4_contract_identity(
                "symmetrix.execution.mh1_uvu/4",
                contract.at("generation_fingerprint").get<std::string>(),
                contract.at("semantic_fingerprint").get<std::string>(),
                contract.at("structure_fingerprint").get<std::string>(),
                contract.at("runtime_layout_fingerprint").get<std::string>(),
                static_cast<std::uint32_t>(interactions));
        } catch(const nlohmann::json::exception& exception) {
            throw std::invalid_argument(
                std::string("MACE Polar artifact node contract is incomplete: ")
                +exception.what());
        }
    }

    struct GraphData {
        std::vector<int> num_neigh;
        std::vector<int> sources;
        std::vector<int> source_types;
        std::vector<double> xyz;
        std::array<double,9> inverse_cell{};
    };

    static void validate_request(const ConsumerSystemRequest& request)
    {
        if(request.atomic_numbers.empty()
            ||request.atomic_numbers.size()>
                static_cast<std::size_t>(std::numeric_limits<int>::max()/3)
            ||request.positions.size()!=3*request.atomic_numbers.size()
            ||request.cell.size()!=9||request.pbc.size()!=3)
            throw std::invalid_argument(
                "MACE Polar consumer requires one consistent system.");
        const auto finite=[](const double value){return std::isfinite(value);};
        if(!std::all_of(request.positions.begin(),request.positions.end(),finite)
            ||!std::all_of(request.cell.begin(),request.cell.end(),finite)
            ||!finite(request.total_charge)||!finite(request.total_spin)
            ||!finite(request.fermi_level)
            ||!std::all_of(request.electric_field.begin(),
                request.electric_field.end(),finite))
            throw std::invalid_argument(
                "MACE Polar consumer inputs must be finite.");
        if(!std::all_of(request.pbc.begin(),request.pbc.end(),
                [] (const int value) {return value==0||value==1;}))
            throw std::invalid_argument(
                "MACE Polar consumer PBC flags must be zero or one.");
    }

    std::vector<int> map_atomic_numbers(std::span<const int> atomic_numbers) const
    {
        std::vector<int> result;
        result.reserve(atomic_numbers.size());
        for(const int atomic_number:atomic_numbers) {
            int type=-1;
            for(std::size_t index=0;index<model_atomic_numbers_.size();++index)
                if(model_atomic_numbers_[index]==atomic_number) {
                    type=static_cast<int>(index);
                    break;
                }
            if(type<0)
                throw std::invalid_argument(
                    "MACE Polar consumer atomic number is absent from the model.");
            result.push_back(type);
        }
        return result;
    }

    static std::array<double,9> invert_cell(std::span<const double> cell)
    {
        const double determinant=
            cell[0]*(cell[4]*cell[8]-cell[5]*cell[7])
            -cell[1]*(cell[3]*cell[8]-cell[5]*cell[6])
            +cell[2]*(cell[3]*cell[7]-cell[4]*cell[6]);
        if(!std::isfinite(determinant)||std::abs(determinant)<1.0e-14)
            return {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};
        const double inverse=1.0/determinant;
        return {
            (cell[4]*cell[8]-cell[5]*cell[7])*inverse,
            (cell[2]*cell[7]-cell[1]*cell[8])*inverse,
            (cell[1]*cell[5]-cell[2]*cell[4])*inverse,
            (cell[5]*cell[6]-cell[3]*cell[8])*inverse,
            (cell[0]*cell[8]-cell[2]*cell[6])*inverse,
            (cell[2]*cell[3]-cell[0]*cell[5])*inverse,
            (cell[3]*cell[7]-cell[4]*cell[6])*inverse,
            (cell[1]*cell[6]-cell[0]*cell[7])*inverse,
            (cell[0]*cell[4]-cell[1]*cell[3])*inverse};
    }

    GraphData build_graph(const std::vector<int>& node_types,
        const ConsumerSystemRequest& request) const
    {
        GraphData graph;
        graph.inverse_cell=invert_cell(request.cell);
        const std::size_t nodes=node_types.size();
        graph.num_neigh.resize(nodes,0);
        const double cutoff_value=this->cutoff();
        std::array<int,3> image_extent{0,0,0};
        std::size_t image_count=1;
        for(int axis=0;axis<3;++axis) {
            if(request.pbc[axis]==0)continue;
            const double reciprocal_norm=std::sqrt(
                graph.inverse_cell[axis]*graph.inverse_cell[axis]
                +graph.inverse_cell[3+axis]*graph.inverse_cell[3+axis]
                +graph.inverse_cell[6+axis]*graph.inverse_cell[6+axis]);
            const double extent=std::ceil(cutoff_value*reciprocal_norm)+1.0;
            if(!std::isfinite(extent)
                ||extent>static_cast<double>(
                    (std::numeric_limits<int>::max()-1)/2))
                throw std::overflow_error(
                    "MACE Polar periodic image extent exceeds native indexing.");
            image_extent[axis]=static_cast<int>(extent);
            const auto axis_images=std::size_t{2}
                *static_cast<std::size_t>(image_extent[axis])+1;
            if(image_count>static_cast<std::size_t>(
                    std::numeric_limits<int>::max())/axis_images)
                throw std::overflow_error(
                    "MACE Polar periodic image count exceeds native indexing.");
            image_count*=axis_images;
        }
        if((nodes!=0&&image_count>
                static_cast<std::size_t>(std::numeric_limits<int>::max())/nodes)
            ||(nodes!=0&&nodes*image_count>
                static_cast<std::size_t>(std::numeric_limits<int>::max())/nodes))
            throw std::overflow_error(
                "MACE Polar periodic pair range exceeds native indexing.");
        for(std::size_t receiver=0;receiver<nodes;++receiver) {
            for(std::size_t source=0;source<nodes;++source)
                for(int a=-image_extent[0];a<=image_extent[0];++a)
                    for(int b=-image_extent[1];b<=image_extent[1];++b)
                        for(int c=-image_extent[2];c<=image_extent[2];++c) {
                            if(receiver==source&&a==0&&b==0&&c==0)continue;
                            const std::array<int,3> shift{a,b,c};
                            std::array<double,3> displacement{};
                            for(int component=0;component<3;++component) {
                                displacement[component]=
                                    request.positions[3*source+component]
                                    -request.positions[3*receiver+component];
                                for(int lattice=0;lattice<3;++lattice)
                                    displacement[component]+=
                                        shift[lattice]*request.cell[3*lattice+component];
                            }
                            const double distance2=
                                displacement[0]*displacement[0]
                                +displacement[1]*displacement[1]
                                +displacement[2]*displacement[2];
                            if(!(distance2<cutoff_value*cutoff_value))continue;
                            if(graph.num_neigh[receiver]
                                    ==std::numeric_limits<int>::max()
                                ||graph.sources.size()
                                    >=static_cast<std::size_t>(
                                        std::numeric_limits<int>::max()))
                                throw std::overflow_error(
                                    "MACE Polar graph exceeds native edge indexing.");
                            ++graph.num_neigh[receiver];
                            graph.sources.push_back(static_cast<int>(source));
                            graph.source_types.push_back(node_types[source]);
                            graph.xyz.insert(
                                graph.xyz.end(),displacement.begin(),displacement.end());
                        }
        }
        return graph;
    }

    Evaluator evaluator_;
    std::uint64_t graph_generation_=0;
    std::size_t graph_rebuild_count_=0;
    std::size_t graph_reuse_count_=0;
    std::size_t topology_reuse_scan_count_=0;
    std::size_t topology_facade_scan_count_=0;
    std::size_t geometry_update_count_=0;
    std::size_t geometry_publish_count_=0;
    std::size_t geometry_reuse_count_=0;
    std::size_t metadata_d2h_bytes_=0;
    ConsumerDiagnostics diagnostics_{};
    std::vector<int> model_atomic_numbers_;
    std::vector<int> atomic_numbers_;
    std::vector<int> node_types_;
    std::vector<double> positions_;
    std::vector<double> cell_;
    std::vector<int> pbc_;
    std::vector<int> num_neigh_;
    std::vector<int> sources_;
    std::vector<int> source_types_;
    std::vector<double> reference_xyz_;
};

using MacePolarConsumer=MacePolarConsumerT<double>;
using MacePolarConsumerFloat=MacePolarConsumerT<float>;

} // namespace symmetrix::mace_polar
