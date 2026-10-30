#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "affine_mlp_kokkos.hpp"
#include "device_backend.hpp"
#include "e3nn_kokkos.hpp"
#include "e3nn_product_kokkos.hpp"
#include "family_interface.hpp"
#include "implementation_registry.hpp"
#include "mace_device_backend.hpp"
#include "mace_nonlinear_schema.hpp"
#include "execution_mode.hpp"
#include "mace_prediction_heads.hpp"
#include "mace_nonlinear_stage.hpp"
#include "radial_function_set_kokkos.hpp"
#include "mace_nonlinear_jit_cuda_plugin.hpp"
#include "mace_nonlinear_jit_host_plugin.hpp"
#include "mace_nonlinear_split_phase_admission.hpp"
#include "mace_jit_m0_host_plugin.hpp"
#include "mace_jit_operator_module.hpp"
#include "tools_kokkos.hpp"
#include "zbl_kokkos.hpp"

template<typename Precision>
class MaceBackboneKokkosT {
    KokkosLiveObjectGuard kokkos_live_object_guard;

protected:
    using precision_type=Precision;
    explicit MaceBackboneKokkosT(
        const std::string& filename, const std::string& requested_head = {});
    // Internal family-schema entry point. Public family evaluators are sibling
    // types and provide their own state object and schema; the backbone itself
    // is not registered as a model family.
    explicit MaceBackboneKokkosT(
        const std::string& filename, const std::string& requested_head,
        MaceBackboneSchema schema, bool require_fast_float_path=true);
    explicit MaceBackboneKokkosT(
        const nlohmann::json& data, MaceBackboneSchema schema,
        bool require_fast_float_path=true);
    ~MaceBackboneKokkosT();

public:
    double r_cut=0.0;
    bool has_field_coupling=false;
    std::string selected_head;
    std::vector<std::string> available_heads;
    bool is_mh1_family() const { return mh1_family.compatible; }
    int mh1_node_channels() const { return mh1_family.node_channels; }
    std::size_t mh1_layer_count() const { return mh1_family.layer_count(); }
    const MaceMH1LayerDescriptor& mh1_layer_descriptor(
        const std::size_t index) const {
        return mh1_family.layer(index);
    }
    int mh1_first_layer_edge_channels() const {
        return mh1_family.first_layer_edge_channels();
    }
    int mh1_radial_size() const { return mh1_family.radial_size; }
    int mh1_l_max() const { return mh1_family.l_max; }
    const std::string& mh1_family_rejection_reason() const {
        return mh1_family.rejection_reason;
    }
    bool mh1_uses_compiled_products() const { return mh1_compiled_products; }
    bool mh1_uses_pair_conditioning() const { return mh1_pair_conditioning; }
    bool mh1_uses_external_uvu_tensors() const { return mh1_external_uvu_tensors; }
    const std::string& mh1_fast_path_rejection_reason() const {
        return mh1_fast_path_rejection;
    }
    bool uses_mh1_fast_path() const { return mh1_fast_path; }
    bool mh1_layer_tensor_product_ready(std::size_t index) const;
    bool mh1_layer_compiled_product_ready(std::size_t index) const;
    bool mh1_layer_pair_conditioning_ready(std::size_t index) const;
    bool mh1_layer_portable_ready(std::size_t index) const;
    std::string mh1_layer_capability_reason(std::size_t index) const;
    bool mh1_portable_final_layer_ready() const {
        return mh1_portable_final_layer_ready_;
    }
    bool mh1_portable_final_layer_enabled() const {
        return mh1_portable_final_layer_enabled_;
    }
    const std::string& mh1_portable_final_layer_kind() const {
        return mh1_portable_final_layer_kind_;
    }
    std::size_t mh1_portable_final_layer_evaluation_count() const {
        return mh1_portable_final_layer_evaluations_;
    }
    void set_mh1_portable_final_layer(bool enabled);
    bool supports_streamed_edges() const { return mh1_fast_path; }
    bool supports_factorized() const;
    bool has_execution_mh1_contract() const { return execution_mh1_has_model_contract; }
    const std::string& execution_mh1_generation_fingerprint() const {
        return execution_mh1_model_generation_fingerprint;
    }
    const std::string& execution_mh1_semantic_fingerprint() const {
        return execution_mh1_model_semantic_fingerprint;
    }
    const std::string& execution_mh1_structure_fingerprint() const {
        return execution_mh1_model_structure_fingerprint;
    }
    void set_execution_mh1_contract_identity(
        std::string tag,std::string generation_fingerprint,
        std::string semantic_fingerprint,std::string structure_fingerprint,
        std::uint32_t interaction_count);
    void set_execution_mh1_v4_contract_identity(
        std::string tag,std::string generation_fingerprint,
        std::string semantic_fingerprint,std::string structure_fingerprint,
        std::string runtime_layout_fingerprint,
        std::uint32_t interaction_count);
    void load_jit_mh1_host_plugin(std::string path);
    void load_jit_mh1_host_plugin_v4(std::string path);
    void load_jit_mh1_host_plugin_v5(std::string path);
    void load_jit_mh1_host_plugin_v6(std::string path);
    void load_jit_mh1_host_plugin_v7(std::string path);
    bool jit_mh1_host_plugin_ready() const;
    bool jit_mh1_host_plugin_v4_ready() const;
    bool jit_mh1_host_plugin_v5_ready() const;
    bool jit_mh1_host_plugin_v6_ready() const;
    bool jit_mh1_host_plugin_v7_ready() const;
    std::string jit_mh1_host_plugin_path() const;
    std::string jit_mh1_host_plugin_artifact_id() const;
    void load_jit_mh1_cuda_plugin(std::string path);
    void load_jit_mh1_cuda_plugin_v4(
        std::string path, std::string launch_plan_json="");
    void load_jit_mh1_cuda_plugin_v5(
        std::string path, std::string launch_plan_json);
    bool jit_mh1_cuda_plugin_ready() const;
    bool jit_mh1_cuda_plugin_v4_ready() const;
    bool jit_mh1_cuda_plugin_v5_ready() const;
    std::string jit_mh1_cuda_plugin_path() const;
    std::string jit_mh1_cuda_plugin_artifact_id() const;
    void load_execution_mh1_product_module(
        int layer,std::string kind,std::string path,std::string metadata_json,
        std::string schedule,int persistent_blocks,std::string backend);
    bool execution_mh1_product_module_ready(int layer) const;
    bool execution_mh1_product_modules_ready() const;
    void clear_execution_mh1_product_modules();
    std::string execution_mh1_product_module_artifact_id(int layer) const;
    std::size_t execution_mh1_product_module_forward_launch_count(int layer) const;
    std::size_t execution_mh1_product_module_reverse_launch_count(int layer) const;
    void test_execution_mh1_product_module_launch_validation(
        int layer,bool reverse,int num_nodes,bool alias_input=true) const;
    std::vector<std::string> execution_mh1_product_module_diagnostics() const;
    bool device_cuda_available() const;
    std::string execution_cuda_device_name() const;
    int execution_cuda_device_ordinal() const;
    int execution_cuda_compute_capability() const;
    int execution_cuda_multiprocessor_count() const;
    int execution_cuda_warp_width() const;
    int execution_cuda_runtime_version() const;
    int execution_cuda_driver_version() const;
    std::string execution_mh1_execution_backend() const {
        if(!uses_prepared_execution(streamed_edges)) return "inactive";
        if(mh1_edge_executor_=="pair_spline_v1") {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
                return jit_mh1_cuda_plugin_v5_ready()
                        &&jit_mh1_cuda_plugin_v4->has_spline_r_program()
                    ?"pair_spline_v1_staged_device_v5"
                    :"pair_spline_v1_kokkos_device";
#endif
            if(jit_mh1_host_plugin_v7_ready())
                return "pair_spline_v1_staged_host_v7_shared_products";
            return "pair_spline_v1_kokkos";
        }
        if(jit_mh1_cuda_plugin_v5_ready()) {
#ifdef KOKKOS_ENABLE_HIP
            if constexpr(std::is_same_v<
                Kokkos::DefaultExecutionSpace,Kokkos::HIP>)
                return "generated_hip_v4";
#endif
            return "generated_cuda_v5";
        }
        if(jit_mh1_host_plugin_v7_ready()) return "generated_host_v7";
        return "unavailable";
    }
    std::string streamed_edges_mode() const;
    void set_streamed_edges(std::string mode);
    const std::string& mh1_edge_executor() const { return mh1_edge_executor_; }
    void set_mh1_edge_executor(std::string executor);
    bool mh1_pair_spline_ready() const { return mh1_pair_spline_ready_; }
    int mh1_pair_spline_nodes() const { return mh1_pair_spline_nodes_; }
    int mh1_pair_spline_type_count() const { return mh1_spline_type_count_; }
    void set_execution_mh1_node_arena_policy(std::string policy);
    const std::string& execution_mh1_node_arena_policy() const {
        return execution_mh1_node_arena_policy_;
    }
    int execution_mh1_node_arena_tile_rows() const;
    void set_execution_mh1_node_arena_tile_rows_for_testing(int rows);
    std::size_t factorized_forward_evaluation_count() const {
        return factorized_forward_evaluations;
    }
    std::size_t factorized_reverse_evaluation_count() const {
        return factorized_reverse_evaluations;
    }
    std::size_t factorized_fallback_evaluation_count() const {
        return factorized_fallback_evaluations;
    }
    std::size_t factorized_schedule_build_count() const {
        return factorized_schedule_builds;
    }
    std::size_t factorized_device_schedule_build_count() const {
        return factorized_device_schedule_builds;
    }
    std::size_t factorized_schedule_topology_d2h_elements() const {
        return factorized_schedule_topology_d2h_elements_;
    }
    std::size_t factorized_prepared_host_topology_elements() const {
        return execution_prepared_num_neigh_host.capacity()
            +execution_prepared_neigh_indices_host.capacity()
            +execution_prepared_neigh_types_host.capacity();
    }
    std::size_t factorized_schedule_host_topology_elements() const {
        return execution_schedule_sources_host.capacity()
            +execution_schedule_targets_host.capacity()
            +factorized_prepared_host_topology_elements();
    }
    std::size_t factorized_schedule_entries() const {
        return execution_source_edges.extent(0);
    }
    std::size_t factorized_schedule_active_sources() const {
        return execution_source_edge_offsets.extent(0)==0
            ?0:execution_source_edge_offsets.extent(0)-1;
    }
    std::size_t factorized_schedule_bytes() const {
        return sizeof(int)*(execution_block_source_segments.extent(0)
            +execution_source_edge_offsets.extent(0)
            +execution_source_edges.extent(0)
            +execution_block_receivers.extent(0)
            +execution_graph_source_edge_offsets.extent(0)
            +execution_graph_source_edges.extent(0)
            +execution_graph_receivers.extent(0));
    }
    // --- Family adapter interface (family_interface.hpp) ------------------
    // The neutral lifecycle, capability, program-compilation, artifact, and
    // diagnostic surface.  Each lifecycle member delegates to the MH-1 entry
    // point that already owns the behaviour, so the interface adds no second
    // execution path.
    symmetrix::execution::FamilyCapabilities family_capabilities() const;
    symmetrix::execution::Program compile_program() const;
    std::uint64_t prepare_graph(
        int num_nodes,std::span<const int> node_types,
        std::span<const int> num_neigh,std::span<const int> neigh_indices,
        std::span<const int> neigh_types);
    void prepare_geometry(
        std::uint64_t graph_generation,
        std::span<const double> reference_positions,
        std::span<const double> reference_xyz,std::span<const double> cell,
        std::span<const double> inverse_cell,std::span<const int> pbc);
    void compute_prepared(
        std::uint64_t graph_generation,
        std::span<const double> xyz,std::span<const double> distances);
    void reduce_forces(std::uint64_t graph_generation);
    void reduce_stress(double volume,std::uint64_t graph_generation);
    std::uint64_t graph_generation() const;
    std::uint64_t geometry_generation() const;
    symmetrix::execution::ArtifactReadiness load_family_artifact(
        const symmetrix::execution::ArtifactRequest& request);
    bool family_artifact_ready() const;
    symmetrix::execution::ArtifactReadiness family_artifact_readiness() const;
    symmetrix::execution::FamilyDiagnostics family_diagnostics() const;
    // --- Tiered implementation registry (implementation_registry.hpp) -----
    symmetrix::execution::ImplementationRegistry
        family_implementation_registry() const;
    symmetrix::execution::ImplementationBindingReport
        family_binding_report() const;
public:
    std::uint64_t prepare_factorized_graph(
        int num_nodes,std::span<const int> node_types,
        std::span<const int> num_neigh,
        std::span<const int> neigh_indices,
        std::span<const int> neigh_types);
    std::pair<std::uint64_t,std::size_t> prepare_periodic_factorized_graph(
        std::span<const int> node_types,
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> inverse_cell,double neighbor_cutoff);
    void prepare_factorized_geometry(
        std::uint64_t graph_generation,
        std::span<const double> reference_positions,
        std::span<const double> reference_xyz,
        std::span<const double> cell,
        std::span<const double> inverse_cell,
        std::span<const int> pbc);
    // Replace an explicit graph and its geometry as one published state.  The
    // predecessor's Kokkos handles remain alive until both candidate stages
    // succeed, so a failed replacement never requires an O(E) host rollback.
    std::uint64_t replace_factorized_graph_geometry(
        int num_nodes,std::span<const int> node_types,
        std::span<const int> num_neigh,
        std::span<const int> neigh_indices,
        std::span<const int> neigh_types,
        std::span<const double> reference_positions,
        std::span<const double> reference_xyz,
        std::span<const double> cell,
        std::span<const double> inverse_cell,
        std::span<const int> pbc);
    void arm_factorized_geometry_prepare_failure_for_testing() {
        factorized_geometry_prepare_failure_armed_=true;
    }
    void compute_prepared_factorized(
        std::uint64_t graph_generation,
        std::span<const double> xyz,std::span<const double> distances);
    void compute_prepared_factorized_positions(
        std::uint64_t graph_generation,
        std::span<const double> positions);
    void reduce_node_forces(
        int num_nodes,
        Kokkos::View<const int*> edge_receivers,
        Kokkos::View<const int*> edge_sources);
    void reduce_prepared_node_forces(std::uint64_t graph_generation);
    void reduce_stress(
        double volume,Kokkos::View<const double*> xyz);
    void reduce_prepared_stress(
        double volume,std::uint64_t graph_generation);
public:
    void prepare_factorized_batch(
        std::uint64_t graph_generation,
        std::span<const std::int64_t> edge_offsets);
    void reduce_prepared_batched_stress(
        std::span<const double> volumes,
        std::uint64_t graph_generation);
    std::uint64_t factorized_graph_generation() const {
        return factorized_prepared_graph_generation;
    }
    void validate_factorized_prepared_system(
        std::uint64_t graph_generation,std::span<const int> node_types,
        std::span<const double> cell,std::span<const int> pbc) const;
    double factorized_prepared_cell_volume(
        std::uint64_t graph_generation) const;
    std::size_t factorized_prepared_graph_count() const {
        return factorized_prepared_graphs;
    }
    std::size_t factorized_prepared_evaluation_count() const {
        return factorized_prepared_evaluations;
    }
    std::size_t factorized_prepared_topology_validation_count() const {
        return factorized_prepared_topology_validations;
    }
    std::size_t factorized_prepared_topology_validated_edge_count() const {
        return factorized_prepared_topology_validated_edges;
    }
    std::size_t factorized_topology_validation_count() const {
        return factorized_topology_validations;
    }
    std::size_t factorized_topology_validation_skip_count() const {
        return factorized_topology_validation_skips;
    }
    std::size_t factorized_preparation_fence_count() const {
        return factorized_preparation_fences;
    }
    std::size_t factorized_evaluation_fence_count() const {
        return factorized_evaluation_fences;
    }
    std::size_t factorized_stage_fence_count() const {
        return factorized_stage_fences;
    }
    std::size_t factorized_zbl_evaluator_stream_launch_count() const {
        return factorized_zbl_evaluator_stream_launches;
    }
    int execution_geometry_capacity_edges() const {
        return execution_geometry_edge_capacity;
    }
    const std::string& execution_geometry_growth_reason() const {
        return execution_geometry_growth_reason_;
    }
    void set_execution_geometry_device_memory_info_for_testing(
        std::size_t free_bytes,std::size_t total_bytes) {
        if(free_bytes>total_bytes)
            throw std::invalid_argument(
                "Execution geometry device free bytes exceed total bytes.");
        execution_geometry_memory_info_override_=total_bytes!=0;
        execution_geometry_device_free_=free_bytes;
        execution_geometry_device_total_=total_bytes;
    }
    std::size_t execution_geometry_workspace_bytes() const;
    std::size_t execution_geometry_allocation_count() const {
        return execution_geometry_allocations;
    }
    std::size_t execution_geometry_copy_count() const {
        return execution_geometry_copies;
    }
    std::size_t factorized_input_h2d_bytes() const {
        return factorized_input_h2d_bytes_;
    }
    std::size_t execution_sphericart_initialization_count() const {
        return execution_sphericart_initializations;
    }
    std::size_t execution_sphericart_launch_count() const {
        return execution_sphericart_launches;
    }
    std::size_t execution_sphericart_async_launch_count() const {
        return execution_sphericart_async_launches;
    }
    void set_execution_mh1_scratch_budget_bytes(std::size_t budget_bytes);
    std::size_t execution_mh1_scratch_budget_bytes() const {
        return execution_mh1_scratch_budget;
    }
    std::size_t execution_mh1_scratch_minimum_bytes() const {
        return execution_mh1_scratch_minimum;
    }
    std::size_t execution_mh1_scratch_planned_bytes() const {
        return execution_mh1_scratch_planned;
    }
    std::size_t execution_mh1_scratch_retained_bytes() const {
        return execution_mh1_scratch_retained;
    }
    std::size_t execution_mh1_scratch_recomputed_bytes() const {
        return execution_mh1_scratch_recomputed;
    }
    std::size_t execution_mh1_scratch_recomputed_layer_count() const {
        return execution_mh1_scratch_recomputed_layers;
    }
    bool execution_mh1_scratch_budget_satisfied() const {
        return execution_mh1_scratch_planned<=execution_mh1_scratch_budget;
    }
    std::size_t execution_mh1_conditioning_recomputation_count() const {
        return execution_mh1_conditioning_recomputations;
    }
    std::array<int,2> execution_mh1_retained_interaction_output_dimensions() const {
        // These dimensions belong to the prepared node runtime, which only the
        // direct route's host and device arena preparation builds.  An
        // unprepared route answers with the field's own default of 0: that is
        // the same "not retained" value the retained-storage release branches
        // test for, and every real dimension is a positive element count, so a
        // caller cannot mistake an unprepared route for a retained width.  The
        // size guard replaces an unchecked index that read past the empty
        // vector and aborted the process on every non-direct route.
        const std::size_t layers=execution_mh1_node_runtime.size();
        return {
            layers>0?execution_mh1_node_runtime[0].retained_interaction_output_dimension:0,
            layers>1?execution_mh1_node_runtime[1].retained_interaction_output_dimension:0,
        };
    }
    std::size_t execution_mh1_retained_interaction_output_bytes() const {
        std::size_t bytes=0;
        for(const auto& state:states)
            bytes+=sizeof(Precision)*state.interaction_output.size();
        return bytes;
    }
    std::size_t execution_mh1_retained_pre_gate_bytes() const {
        std::size_t bytes=0;
        for(const auto& state:states)
            bytes+=sizeof(Precision)*state.pre_gate.size();
        return bytes;
    }
    std::size_t execution_mh1_product_tile_scratch_bytes() const {
        std::size_t bytes=0;
        for(const auto& runtime:execution_mh1_node_runtime)
            bytes+=sizeof(Precision)*(runtime.interaction_output.size()
                +runtime.interaction_adjoint.size()
                +runtime.contraction_output.size()
                +runtime.contraction_adjoint.size());
        return bytes;
    }
    std::size_t execution_mh1_layer_up_adjoint_storage_bytes() const {
        std::size_t bytes=0;
        for(const auto& state:states)
            bytes+=sizeof(Precision)*state.up_adj_storage.size();
        return bytes;
    }
    std::size_t execution_mh1_up_adjoint_scratch_bytes() const {
        return sizeof(Precision)*execution_mh1_up_adjoint_scratch_storage.size();
    }
    std::size_t execution_mh1_up_adjoint_scratch_reuse_count() const {
        return execution_mh1_up_adjoint_scratch_reuses;
    }
    std::size_t execution_mh1_generated_forward_launch_count() const {
        return execution_mh1_generated_forward_launches;
    }
    std::size_t execution_mh1_generated_source_reverse_launch_count() const {
        return execution_mh1_generated_source_reverse_launches;
    }
    std::size_t execution_mh1_generated_edge_reverse_launch_count() const {
        return execution_mh1_generated_edge_reverse_launches;
    }
    std::size_t execution_mh1_generated_conditioning_forward_launch_count() const {
        return execution_mh1_generated_conditioning_forward_launches;
    }
    std::size_t execution_mh1_generated_conditioning_reverse_launch_count() const {
        return execution_mh1_generated_conditioning_reverse_launches;
    }
    bool factorized_source_owned_reverse() const {
        return uses_prepared_execution(streamed_edges)
            &&factorized_source_owned_reverse_used;
    }
    int edge_workspace_rows() const;
    std::size_t edge_workspace_bytes() const;
    std::size_t conditioned_mlp_workspace_bytes() const;
    void fence() const {
        factorized_execution_space.fence("MACE_Nonlinear public fence");
    }
    void set_e3_linear_backend(const std::string& backend);
    std::string e3_linear_backend() const;
    std::string selected_e3_linear_backend(std::size_t samples) const;
    void set_execution_mh1_host_node_backend(const std::string& backend);
    const std::string& execution_mh1_host_node_backend() const {
        return execution_mh1_host_node_backend_request;
    }
    std::string selected_execution_mh1_host_node_backend() const;
    std::string tensor_product_backend() const;
    std::string tensor_product_execution_backend() const;
    int tensor_product_channel_team_size() const;
    int tensor_product_harmonic_team_size() const;
    void set_fused_gate_normalization_reverse(bool enabled);
    bool fused_gate_normalization_reverse_available() const;
    bool uses_fused_gate_normalization_reverse() const;
    void set_direct_node_tensor_reverse(bool enabled);
    bool direct_node_tensor_reverse_available() const;
    bool uses_direct_node_tensor_reverse() const;
    std::size_t linear_workspace_bytes() const;
    std::size_t tensor_workspace_bytes() const;
    std::size_t product_workspace_bytes() const;
    std::size_t node_workspace_bytes() const;
    std::size_t precision_workspace_bytes() const;
    std::vector<int> atomic_numbers_host;
    Kokkos::View<int*> atomic_numbers;
    Kokkos::View<double*> atomic_energies,node_energies,node_forces,atom_forces,
        stress_tensor;
    void compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> num_neigh,Kokkos::View<const int*> neigh_indices,
        Kokkos::View<const int*> neigh_types,Kokkos::View<const double*> xyz,
        Kokkos::View<const double*> distances,
        std::uint64_t execution_graph_generation=0);
    void compute_Y(
        Kokkos::View<const double*> xyz,bool evaluator_stream=false);

private:
    struct SphericalHarmonicsState;

public:
    struct Gate {
        int input_size=0,scalar_size=0,gate_size=0,gated_size=0,output_size=0;
        std::vector<IrrepBlock> scalar_blocks,gated_blocks;
        std::vector<Precision> scalar_constants;
        std::vector<Precision> gate_constants;
        Kokkos::View<Precision*> fused_scalar_constants,fused_gate_constants;
        Kokkos::View<int**,Kokkos::LayoutRight> fused_gate_plan,
            fused_packed_input_plan;
        explicit Gate(const nlohmann::json& data);
        bool supports_fused_normalized_reverse() const;
        void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                      Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
        void evaluate_normalized_to_packed(
            Kokkos::View<const Precision**,Kokkos::LayoutRight> linear_forward,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> residual,
            Kokkos::View<const int*> residual_active,
            Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
            Kokkos::View<Precision**,Kokkos::LayoutRight> input,
            Kokkos::View<Precision**,Kokkos::LayoutRight> packed_output,
            bool linear_is_normalized=false,
            Kokkos::View<const Precision*> normalization_inverse={}) const;
        void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                     Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                     Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const;
        void reverse_normalized(
            Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> linear_forward,
            Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
            Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
            Kokkos::View<Precision**,Kokkos::LayoutRight> linear_adjoint,
            Kokkos::View<Precision*> density_adjoint) const;
        void reverse_normalized_packed(
            Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_output_adjoint,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_linear_forward,
            Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
            Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
            Kokkos::View<Precision**,Kokkos::LayoutRight> packed_linear_adjoint,
            Kokkos::View<Precision*> density_adjoint,
            bool linear_is_normalized=false) const;
    };
private:
    struct Interaction {
        E3LinearKokkosT<Precision> source_embedding,target_embedding,linear_up,skip,linear_res,linear_1,linear_2;
        E3TensorProductKokkosT<Precision> convolution;
        AffineMLPKokkosT<Precision> convolution_weights,density;
        Gate gate;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_source_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_target_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_source_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_target_contributions;
        RadialFunctionSetKokkos<Precision> pair_spline;
        int pair_spline_weight_count=0;
        bool pair_spline_ready=false;
        Precision alpha,beta;
        explicit Interaction(const nlohmann::json& data);
        bool prepare_pair_conditioning(
            const nlohmann::json& data,int radial_size,int model_element_count,
            const std::vector<int>& selected_model_indices);
        bool prepare_pair_spline(
            const nlohmann::json& data,const nlohmann::json& radial_data,
            double cutoff,int radial_size,int model_element_count,
            const std::vector<int>& selected_model_indices,
            const std::vector<int>& model_atomic_numbers,int spline_nodes);
        void set_e3_linear_backend(const std::string& backend);
        void set_e3_linear_workspace(
            const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace);
        void set_conditioned_mlp_workspace(
            const std::shared_ptr<typename AffineMLPKokkosT<Precision>::Workspace>&
                workspace);
        std::size_t linear_workspace_bytes() const;
    };
    void prepare_mh1_pair_spline_for_graph(
        Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> neigh_types,
        std::uint64_t graph_generation);
public:
    struct Readout {
        // ``nonlinear`` is the plain nonlinear readout (one activation between
        // its two linears); ``bias`` is the bias variant, which carries a
        // learned scalar bias through a middle linear with a second activation
        // (``linear_mid``).  Both consume the same hidden width, so the arena
        // dimensions are the hidden width either way.
        bool nonlinear=false,bias=false; E3LinearKokkosT<Precision> linear,linear_1,linear_mid,linear_2; Precision activation_constant=1.0;
        Kokkos::View<Precision**,Kokkos::LayoutRight> hidden,activated,result,seed,
            activated_adj,hidden_adj;
        Kokkos::View<Precision**,Kokkos::LayoutRight> hidden_storage,activated_storage,
            result_storage,seed_storage,activated_adj_storage,hidden_adj_storage;
        explicit Readout(const nlohmann::json& data);
        void set_e3_linear_backend(const std::string& backend);
        void set_e3_linear_workspace(
            const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace);
        std::size_t linear_workspace_bytes() const;
        bool supports_portable_scalar_schedule() const;
        void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision*> output);
        void evaluate_portable_scalar(
            Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
            Kokkos::View<Precision*> output);
        void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Precision scale,
                     Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
        void reverse_portable_scalar(
            Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
            Precision scale,
            Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
    };
private:
    struct LayerState {
        Kokkos::View<Precision**,Kokkos::LayoutRight> input,up,residual,skip,messages,linear_1_output,pre_gate,gated,interaction_output,output,
            contraction_output,contraction_output_adjoint;
        Kokkos::View<Precision**,Kokkos::LayoutRight> source_embeddings,target_embeddings,edge_features,raw_weights,weights,edge_up,edge_messages;
        Kokkos::View<Precision**,Kokkos::LayoutRight> execution_embedding,
            execution_embedding_adj,execution_graph_embedding,
            execution_graph_embedding_adj,execution_graph_fixed_contribution,
            execution_graph_harmonic_adj,execution_input_ir_mul,
            execution_output_ir_mul,execution_input_adjoint_ir_mul;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_contributions,density_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_matrix,layer_adjoint,
            interaction_output_adj,skip_adj,gated_adj,pre_gate_adj,linear_adj,
            message_adj,up_adj,edge_message_adj,edge_up_adj,
            edge_harmonic_adj,weight_adj,raw_weight_adj,edge_feature_adj,
            density_raw_adj,density_feature_adj;
        Kokkos::View<Precision*> density_raw,density_base,densities,density_adj,
            normalization_inverse,execution_cutoff_adj,
            execution_graph_cutoff_adj;
        Kokkos::View<int*> agnostic_elements;

        Kokkos::View<Precision**,Kokkos::LayoutRight> up_storage,residual_storage,
            skip_storage,messages_storage,linear_1_output_storage,pre_gate_storage,
            gated_storage,interaction_output_storage,output_storage,
            contraction_output_storage,contraction_output_adjoint_storage,
            source_embeddings_storage,target_embeddings_storage,
            edge_features_storage,raw_weights_storage,weights_storage,
            edge_up_storage,edge_messages_storage,convolution_contributions_storage,
            execution_embedding_storage,execution_embedding_adj_storage,
            execution_graph_embedding_storage,execution_graph_embedding_adj_storage,
            execution_graph_fixed_contribution_storage,
            execution_graph_harmonic_adj_storage,execution_input_ir_mul_storage,
            execution_output_ir_mul_storage,execution_input_adjoint_ir_mul_storage,
            density_contributions_storage,density_matrix_storage,
            layer_adjoint_storage,interaction_output_adj_storage,skip_adj_storage,
            gated_adj_storage,pre_gate_adj_storage,linear_adj_storage,
            message_adj_storage,up_adj_storage,
            edge_message_adj_storage,edge_up_adj_storage,edge_harmonic_adj_storage,
            weight_adj_storage,raw_weight_adj_storage,edge_feature_adj_storage,
            density_raw_adj_storage,density_feature_adj_storage;
        Kokkos::View<Precision*> density_raw_storage,density_base_storage,
            densities_storage,density_adj_storage,normalization_inverse_storage,
            execution_cutoff_adj_storage,
            execution_graph_cutoff_adj_storage;
        Kokkos::View<int*> agnostic_elements_storage;
    };
    struct ExecutionMH1NodeRuntime {
        static constexpr int capacity_tile_rows=256;
        static constexpr int cuda_throughput_tile_rows=1024;
        Kokkos::View<Precision*> linear_parameters,product_parameters,
            readout_parameters;
        Kokkos::View<Precision**,Kokkos::LayoutRight> arena,residual,
            linear_1_output,pre_gate,gated,interaction_output,
            interaction_adjoint,contraction_output,contraction_adjoint,
            gated_adjoint,pre_gate_adjoint,
            linear_1_adjoint;
        int arena_dimension=0;
        int tile_rows=capacity_tile_rows;
        int retained_pre_gate_dimension=0;
        int retained_interaction_output_dimension=0;
        bool reuse_message_adjoint=false;
        std::size_t bytes() const {
            return sizeof(Precision)*(linear_parameters.size()
                +product_parameters.size()+readout_parameters.size()
                +arena.size()+residual.size()+linear_1_output.size()
                +pre_gate.size()+gated.size()+interaction_output.size()
                +interaction_adjoint.size()+contraction_output.size()
                +contraction_adjoint.size()+gated_adjoint.size()
                +pre_gate_adjoint.size()+linear_1_adjoint.size());
        }
    };
    int l_max=0,num_lm=0,model_num_elements=0,cutoff_power=0,num_bessel=0;
    bool apply_cutoff=false,has_agnesi=false,has_zbl=false,mh1_fast_path=false;
    bool mh1_compiled_products=false,mh1_pair_conditioning=false,
        mh1_external_uvu_tensors=false;
    bool mh1_portable_final_layer_ready_=false;
    bool mh1_portable_final_layer_enabled_=false;
    std::string mh1_portable_final_layer_kind_="generic";
    std::size_t mh1_portable_final_layer_evaluations_=0;
    bool fused_gate_normalization_reverse_enabled=true;
    bool mh1_pair_spline_ready_=false;
    bool mh1_pair_spline_lazy_=false;
    // Format-v3 artifacts before this metadata existed retain the 256-node
    // default; extracted artifacts can select a different resolution.
    int mh1_pair_spline_nodes_=256;
    std::vector<int> mh1_selected_model_indices_host_;
    std::vector<int> mh1_model_atomic_numbers_host_;
    nlohmann::json mh1_pair_spline_interactions_data_;
    nlohmann::json mh1_pair_spline_radial_data_;
    std::vector<int> mh1_spline_global_types_;
    std::vector<int> mh1_spline_global_to_active_;
    Kokkos::View<int*> mh1_spline_node_types_;
    Kokkos::View<int*> mh1_spline_neigh_types_;
    int mh1_spline_type_count_=0;
    std::uint64_t mh1_spline_graph_generation_=0;
    std::string mh1_edge_executor_="mlp_reference";
    std::string execution_mh1_node_arena_policy_="throughput-v1";
    int execution_mh1_node_arena_tile_rows_override_=0;
    int selected_execution_mh1_node_arena_tile_rows() const;
    void apply_execution_mh1_node_arena_tile_rows();
    bool uses_packed_node_linears() const;
    bool direct_node_tensor_reverse_enabled=true;
    std::string execution_mh1_host_node_backend_request="auto";
    MaceMH1FamilyDescriptor mh1_family;
    std::string mh1_fast_path_rejection;
    ExecutionMode streamed_edges=ExecutionMode::materialized;
    Kokkos::DefaultExecutionSpace factorized_execution_space;
    std::size_t factorized_forward_evaluations=0;
    std::size_t factorized_reverse_evaluations=0;
    std::size_t factorized_fallback_evaluations=0;
    std::size_t factorized_schedule_builds=0;
    std::size_t factorized_device_schedule_builds=0;
    std::size_t factorized_schedule_topology_d2h_elements_=0;
    std::uint64_t factorized_graph_generation_counter=0;
    std::uint64_t factorized_prepared_graph_generation=0;
    std::uint64_t factorized_prepared_geometry_graph_generation=0;
    bool factorized_geometry_prepare_failure_armed_=false;
    std::uint64_t factorized_completed_evaluation_graph_generation=0;
    std::size_t factorized_prepared_graphs=0;
    std::size_t factorized_prepared_evaluations=0;
    std::size_t factorized_topology_validations=0;
    std::size_t factorized_topology_validation_skips=0;
    std::size_t factorized_prepared_topology_validations=0;
    std::size_t factorized_prepared_topology_validated_edges=0;
    std::size_t factorized_preparation_fences=0;
    std::size_t factorized_evaluation_fences=0;
    std::size_t factorized_stage_fences=0;
    std::size_t factorized_zbl_evaluator_stream_launches=0;
    int execution_geometry_edge_capacity=0;
    std::string execution_geometry_growth_reason_="geometric growth";
    bool execution_geometry_memory_info_override_=false;
    std::size_t execution_geometry_device_free_=0;
    std::size_t execution_geometry_device_total_=0;
    std::size_t execution_geometry_allocations=0;
    std::size_t execution_geometry_copies=0;
    std::size_t factorized_input_h2d_bytes_=0;
    std::size_t execution_sphericart_initializations=0;
    std::size_t execution_sphericart_launches=0;
    std::size_t execution_sphericart_async_launches=0;
    std::size_t execution_mh1_scratch_budget=
        std::numeric_limits<std::size_t>::max();
    std::size_t execution_mh1_scratch_minimum=0;
    std::size_t execution_mh1_scratch_planned=0;
    std::size_t execution_mh1_scratch_retained=0;
    std::size_t execution_mh1_scratch_recomputed=0;
    std::size_t execution_mh1_scratch_recomputed_layers=0;
    std::size_t execution_mh1_conditioning_recomputations=0;
    std::vector<unsigned char> execution_mh1_retain_graph_embedding;
    std::vector<int> execution_mh1_graph_embedding_scratch_slot;
    std::vector<int> execution_mh1_graph_embedding_scratch_widths;
    std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>>
        execution_mh1_graph_embedding_scratch_storage,
        execution_mh1_graph_embedding_adjoint_scratch_storage;
    Kokkos::View<Precision**,Kokkos::LayoutRight>
        execution_mh1_graph_harmonic_adjoint_scratch_storage;
    Kokkos::View<Precision*> execution_mh1_graph_cutoff_adjoint_scratch_storage,
        execution_mh1_message_adjoint_scratch_storage,
        execution_mh1_up_adjoint_scratch_storage,
        execution_mh1_input_adjoint_scratch_storage,
        execution_mh1_density_adjoint_scratch_storage;
    bool execution_source_schedule_dirty=true;
    std::size_t execution_mh1_generated_forward_launches=0;
    std::size_t execution_mh1_generated_source_reverse_launches=0;
    std::size_t execution_mh1_generated_edge_reverse_launches=0;
    std::size_t execution_mh1_generated_conditioning_forward_launches=0;
    std::size_t execution_mh1_generated_conditioning_reverse_launches=0;
    std::size_t execution_mh1_up_adjoint_scratch_reuses=0;
    bool factorized_source_owned_reverse_used=false;
    bool execution_mh1_has_model_contract=false;
    bool execution_mh1_has_model_contract_v4=false;
    // The interaction count the validated model contract declares.  Both the
    // ABI-v3 projection and the ABI-v4 contract carry the same count, so one
    // member holds the contract field the plugin loaders validate.
    std::uint32_t execution_mh1_contract_interaction_count=0;
    std::string execution_mh1_model_generation_fingerprint;
    std::string execution_mh1_model_semantic_fingerprint;
    std::string execution_mh1_model_structure_fingerprint;
    std::string execution_mh1_model_generation_fingerprint_v4;
    std::string execution_mh1_model_semantic_fingerprint_v4;
    std::string execution_mh1_model_structure_fingerprint_v4;
    std::string execution_mh1_model_runtime_layout_fingerprint_v4;
    std::unique_ptr<symmetrix::execution::MH1HostPlugin> jit_mh1_host_plugin;
    std::unique_ptr<symmetrix::execution::MH1HostPluginV4>
        jit_mh1_host_plugin_v4;
    std::unique_ptr<symmetrix::execution::MH1HostPluginV5>
        jit_mh1_host_plugin_v5;
    std::unique_ptr<symmetrix::execution::MH1HostPluginV6>
        jit_mh1_host_plugin_v6;
    std::unique_ptr<symmetrix::execution::MH1HostPluginV7>
        jit_mh1_host_plugin_v7;
    // The per-interaction tables and owner entry points of whichever MH-1 host
    // plugin generation is loaded. ABI v3/v4/v5 embed the tables at a
    // compiled-in bound; ABI v6 points at tables sized to the contract. The
    // current ABI v7 retains those pointer-backed tables and marks node programs
    // as composed because their contractions live in separate M0/M1 modules.
    // The evaluator reads every admitted layout through this common view.
    struct MH1HostPluginView {
        const SymmetrixJitMH1HostConditioningOwnersV4* conditioning=nullptr;
        const SymmetrixJitMH1HostNodeProgramV4* node_programs=nullptr;
        const SymmetrixJitMH1HostSplineRProgramV5* spline_r_programs=nullptr;
        SymmetrixJitMH1HostForwardOwnerV4 forward_owner=nullptr;
        SymmetrixJitMH1HostSourceReverseOwnerV4 source_reverse_owner=nullptr;
        SymmetrixJitMH1HostEdgeReverseOwnerV4 edge_reverse_owner=nullptr;
        std::uint32_t capabilities=0;
        bool composed_node_programs=false;
        bool node_programs_ready() const {
            return node_programs!=nullptr||composed_node_programs;
        }
        bool spline_r_ready() const { return spline_r_programs!=nullptr; }
    };
    MH1HostPluginView mh1_host_view;
    // The per-layer node features the backbone produced, retained when a
    // consumer asks for them (``set_retain_layer_features``).  They are the
    // product outputs the model's own per-layer readouts consume -- the
    // family's feature-mixer and spin-source inputs -- and the copy is the
    // only thing the retention costs, so it is opt-in: the default path keeps
    // its scratch discipline and the family layer pays for what it reads.
    std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>>
        retained_layer_features;
    bool retain_layer_features=false;
    // The per-layer product seeds a consumer owns
    // (``set_layer_product_seed``): the retention's reverse twin.  A retained
    // feature is the forward tap of one layer's product output, so a
    // consumer's seed of it is a seed of that product; the reverse adds the
    // seed's model-basis image into the layer's product adjoint before the
    // product stage fires, exactly where the model's own per-layer readouts
    // seed the same tensor.  Empty means no consumer owns one.
    std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>>
        layer_product_seeds;
    // Clearing a consumer seed deactivates it but does not release its
    // evaluator-owned storage; the next same-extent consumer request reuses the
    // buffers and merely overwrites their contents.
    std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>>
        layer_product_seed_storage;
    std::size_t layer_product_seed_workspace_replacements=0;
    std::unique_ptr<symmetrix::execution::MH1CudaPlugin> jit_mh1_cuda_plugin;
    std::unique_ptr<symmetrix::execution::MH1CudaPluginV4>
        jit_mh1_cuda_plugin_v4;
    struct ExecutionMH1ProductArtifact {
        std::string kind;
        std::string schedule;
        std::string target;
        std::string artifact_id;
        std::string structure_fingerprint;
        std::unique_ptr<symmetrix::execution::OperatorModule> device_module;
        std::unique_ptr<symmetrix::execution::M0HostPlugin> host_plugin;
        std::size_t forward_launch_count=0;
        std::size_t reverse_launch_count=0;
        int persistent_blocks_per_compute_unit=0;
        int persistent_blocks=0;
        int channel_tile=0;
        int forward_node_tile=0;
        int forward_component_tile=0;
        int reverse_node_tile=0;
        int reverse_component_tile=0;
        int logical_tiles_per_block=0;
        int channels=0;
        int input_components=0;
        int output_components=0;
        std::int64_t forward_logical_tiles=0;
        std::int64_t forward_required_blocks=0;
        std::int64_t forward_realized_blocks=0;
        std::int64_t reverse_logical_tiles=0;
        std::int64_t reverse_required_blocks=0;
        std::int64_t reverse_realized_blocks=0;
        bool ready() const {
            return (device_module&&static_cast<bool>(*device_module))
                ||(host_plugin&&static_cast<bool>(*host_plugin));
        }
    };
    std::vector<ExecutionMH1ProductArtifact> execution_mh1_product_artifacts;
#ifdef KOKKOS_ENABLE_CUDA
    static constexpr int streamed_edge_block_size=16384;
#else
    static constexpr int streamed_edge_block_size=1024;
#endif
    bool streams_layer(int layer) const;
public:
    struct LayerRetainedFeatureLayout {
        int width=0;
        int channels=0;
        int scalar_offset=-1;
        int scalar_multiplicity=0;
        int vector_offset=-1;
        int vector_multiplicity=0;
        int vector_components=0;
        std::vector<int> retained_from_native_components;
        std::vector<int> retained_from_native_signs;
        Kokkos::View<int*> portable_source_indices;
        Kokkos::View<int*> generated_source_indices;
        Kokkos::View<Precision*> retained_signs;
    };
    // Retain (or stop retaining) the per-layer node features.  Retaining
    // copies each layer's product output, so it is the caller's decision; the
    // retained features are the ones the model's own per-layer readouts see,
    // in layer order.
    void set_retain_layer_features(const bool retain);
    bool retains_layer_features() const { return retain_layer_features; }
    // The retained features, one view per interaction layer.  Empty when the
    // retention is off or no forward pass has run since it was enabled.
    const std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>>&
        layer_features() const { return retained_layer_features; }
    // Adopt (or drop) the consumers' per-layer product seeds.  Each seed is
    // given in the retained features' basis -- the model-convention view the
    // consumer read, one entry per interaction layer in layer order, of that
    // layer's own width -- and is converted back to the product basis inside
    // the reverse.  The seed only reaches a forward pass whose retention was
    // on, so the two taps are enabled together by their consumer.
    void set_layer_product_seed(
        const std::vector<Kokkos::View<const Precision**,Kokkos::LayoutRight>>&
            seeds);
    // Adopt a concatenated device-resident seed produced by another native
    // component.  The method splits it using the retained layouts and copies
    // device-to-device into the evaluator-owned reverse taps; no host buffer
    // participates in the handoff.
    void set_layer_product_seed_flat(
        const Kokkos::View<const Precision**,Kokkos::LayoutRight>& seeds);
    void clear_layer_product_seed();
    bool has_layer_product_seed() const { return !layer_product_seeds.empty(); }
    std::size_t layer_product_seed_workspace_replacement_count() const
    {
        return layer_product_seed_workspace_replacements;
    }
    // The model-convention conversion's component count for a layer whose
    // node-feature width is ``width``, or an exception naming the basis this
    // build cannot convert.  The retention and the reverse tap share one parse
    // of the convention, so a family that has to refuse such a basis *before*
    // its first forward (a checkpoint whose layers are not convertible) asks
    // here rather than re-spelling the rule in another language.
    int layer_feature_components(int width) const;
    const LayerRetainedFeatureLayout& mh1_layer_retained_feature_layout(
        std::size_t layer) const;
protected:
    Kokkos::DefaultExecutionSpace& backbone_execution_space() {
        return factorized_execution_space;
    }
    const Kokkos::DefaultExecutionSpace& backbone_execution_space() const {
        return factorized_execution_space;
    }
    std::size_t prepared_node_count() const {
        return execution_prepared_node_types.extent(0);
    }
private:
    // One canonical retained/model-basis layout per interaction.  Its two
    // source-index maps differ only in native storage order (portable mul-ir
    // versus generated ir-mul); forward retention and reverse transpose both
    // consume these exact maps.
    std::vector<LayerRetainedFeatureLayout> layer_retained_feature_layouts;
    LayerRetainedFeatureLayout make_layer_retained_feature_layout(
        std::size_t layer) const;
    void release_layer_edge_workspace(int layer);
public:
    // nvcc requires functions enclosing Kokkos extended lambdas to be public.
    void prepare_execution_source_schedule(
        int num_nodes,int samples,Kokkos::View<const int*> source_indices,
        Kokkos::View<const int*> target_indices);
    void prepare_execution_source_schedule_device(
        int num_nodes,int samples,Kokkos::View<const int*> source_indices,
        Kokkos::View<const int*> target_indices);
    // Add one layer's seed, given in the retained basis, into that layer's
    // model-basis product adjoint.  Public for the same nvcc rule as above:
    // its conversion runs as an extended lambda.
    void add_layer_product_seed(int layer,int num_nodes,
        const Kokkos::View<Precision**,Kokkos::LayoutRight>& adjoint,
        bool generated_layout=false) const;
    // Copy one layer's product output into the retained slot (a no-op when the
    // retention is off).  Public for the same nvcc rule as above.
    void retain_layer_output(int layer, int num_nodes,
        const Kokkos::View<const Precision**,Kokkos::LayoutRight>& output,
        bool generated_layout=false);
private:
    void record_factorized_input_h2d(
        std::size_t elements,std::size_t element_size,const char* what);
    void invalidate_factorized_prepared_graph();
    void reserve_execution_geometry_workspace(int edges);
    bool prepare_execution_source_schedule_host(
        int num_nodes,std::span<const int> source_indices,
        std::span<const int> target_indices);
    double radial_prefactor=0.0,agnesi_a=0.0,agnesi_q=0.0,agnesi_p=0.0,scale=1.0,shift=0.0;
    Kokkos::View<int*> model_indices,model_atomic_numbers;
    Kokkos::View<Precision*> bessel_weights,covalent_radii,cutoffs,Y,Y_grad,xyz_shuffled,Y_grad_shuffled;
    Kokkos::View<Precision**,Kokkos::LayoutRight> attrs,radial,edge_harmonics,features;
    Kokkos::View<int*> targets,product_elements,offsets;
    Kokkos::View<int*> execution_prepared_node_types,
        execution_prepared_num_neigh,execution_prepared_neigh_indices,
        execution_prepared_neigh_types,execution_prepared_offsets,
        execution_prepared_targets;
    Kokkos::View<double*> execution_prepared_xyz,execution_prepared_distances,
        execution_prepared_reference_positions,execution_prepared_reference_xyz,
        execution_prepared_cell,execution_prepared_inverse_cell,
        execution_prepared_positions,execution_prepared_displacements;
    Kokkos::View<int*> execution_prepared_pbc,
        execution_prepared_geometry_invalid;
    Kokkos::View<std::int64_t*> execution_prepared_batch_edge_offsets;
    Kokkos::View<double*> execution_prepared_batch_volumes;
    Kokkos::View<int*> execution_block_source_segments,
        execution_source_edge_offsets,execution_source_edges,
        execution_block_receivers,execution_graph_source_edge_offsets,
        execution_graph_source_edges,execution_graph_receivers,
        execution_mh1_spline_intervals_storage;
    Kokkos::View<Precision*> cutoffs_storage,Y_storage,Y_grad_storage,
        xyz_shuffled_storage,Y_grad_shuffled_storage,readout_contribution,
        readout_contribution_storage,cutoff_adjoints,cutoff_adjoints_storage,
        pair_spline_distance_adjoints,pair_spline_distance_adjoints_storage,
        execution_mh1_spline_coordinates_storage;
    Kokkos::View<double*> node_energies_storage,node_forces_storage,
        atom_forces_storage,stress_tensor_storage,
        zbl_energies,zbl_energies_storage,zbl_forces,zbl_forces_storage;
    Kokkos::View<Precision**,Kokkos::LayoutRight> attrs_storage,radial_storage,
        edge_harmonics_storage,features_storage,radial_adjoints,
        radial_adjoints_storage,harmonic_adjoints,harmonic_adjoints_storage;
    Kokkos::View<int*> targets_storage,product_elements_storage,offsets_storage;
    Kokkos::View<int*> execution_block_source_segments_storage,
        execution_source_edge_offsets_storage,execution_source_edges_storage,
        execution_block_receivers_storage,execution_graph_source_edge_offsets_storage,
        execution_graph_source_edges_storage,execution_graph_receivers_storage;
    int execution_graph_source_owner_count=0;
    int execution_graph_receiver_count=0;
    std::vector<int> execution_block_source_segments_host;
    std::vector<int> execution_block_receiver_counts_host;
    std::vector<int> execution_schedule_sources_host;
    std::vector<int> execution_schedule_targets_host;
    int execution_schedule_num_nodes=0;
    std::vector<int> execution_prepared_node_types_host;
    std::vector<double> execution_prepared_cell_host;
    std::vector<int> execution_prepared_pbc_host;
    std::vector<double> execution_prepared_positions_host;
    std::uint64_t execution_prepared_positions_graph_generation=0;
    std::vector<std::int64_t> execution_prepared_batch_edge_offsets_host;
    std::uint64_t factorized_prepared_batch_graph_generation=0;
    std::vector<int> execution_prepared_num_neigh_host;
    std::vector<int> execution_prepared_neigh_indices_host;
    std::vector<int> execution_prepared_neigh_types_host;
    std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>
        e3_linear_workspace;
    std::shared_ptr<typename AffineMLPKokkosT<Precision>::Workspace>
        conditioned_mlp_workspace;
    E3LinearKokkosT<Precision> node_embedding;
    std::vector<Interaction> interactions;
    std::vector<E3ProductBasisKokkosT<Precision>> products;
    std::vector<Readout> readouts;
    std::vector<LayerState> states;
    std::vector<ExecutionMH1NodeRuntime> execution_mh1_node_runtime;
    // The interaction-layer count and every layer's profiling region names are
    // this description's data, built once from the validated model's layer
    // count; the orchestration reads its regions and phase ordinals here
    // instead of from a compiled-in two-layer table.
    symmetrix::mace_nonlinear::StageDescription stage_description;
    void prepare_execution_mh1_node_runtime_parameters();
    void reserve_execution_mh1_node_arena(int num_nodes);
    symmetrix::execution::MH1HostNodeProgramExpectation
        execution_mh1_host_node_expectation(int layer) const;
    symmetrix::execution::MH1CudaNodeProgramExpectation
        execution_mh1_cuda_node_expectation(int layer) const;
    ZBLKokkos zbl;
    std::unique_ptr<SphericalHarmonicsState> spherical_harmonics_state;
};

// Public family evaluators are sibling types over the same internal backbone.
// The ordinary nonlinear family deliberately exposes only the neutral
// backbone lifecycle.  Family-specific evaluators select their own schema and
// selectively publish their own extension surface.
template<typename Precision>
class MaceNonlinearKokkosT final : public MaceBackboneKokkosT<Precision> {
public:
    using precision_type=Precision;
    explicit MaceNonlinearKokkosT(
        const std::string& filename, const std::string& requested_head = {})
        :MaceBackboneKokkosT<Precision>(
            filename,requested_head,MaceBackboneSchema::mh1) {}
};

using MaceNonlinearKokkos = MaceNonlinearKokkosT<double>;
using MaceNonlinearFloatKokkos = MaceNonlinearKokkosT<float>;
