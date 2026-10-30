#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "e3nn_kokkos.hpp"
#include "execution_plan.hpp"
#include "family_interface.hpp"
#include "implementation_registry.hpp"
#include "plan_candidate_space.hpp"
#include "oam_schema.hpp"
#include "oam_radial.hpp"
#include "oam_memory_planner.hpp"
#include "tools_kokkos.hpp"
#include "oam_jit_host_plugin.hpp"
#include "cuda_r1_identity.hpp"
#include "oam_cuda_plugin.hpp"

namespace symmetrix_oam_detail {

// Grow-only capacity admission for the evaluator-owned buffers a prepared
// evaluation needs for the whole graph.  A retained buffer that already covers
// the requested extent is reused in place: its capacity may exceed the caller's
// current counts, and only the launch bounds follow those counts.  A buffer
// that is too small is replaced once at the requested extent.  Buffers are
// never shrunk, so a smaller geometry update keeps the allocation and merely
// rebinds the active extent.
template <typename T>
inline void ensure_view_capacity(
    Kokkos::View<T*>& view, const char* label, const std::size_t required)
{
    if (view.extent(0) >= required) return;
    // A retained buffer is never allocated with extent zero, so an empty
    // extent means this is the first admission.
    if (view.extent(0) == 0)
        // The legacy Kokkos::View label constructor, which LAMMPS forces on,
        // accepts a std::string but not a borrowed const char*.
        view = Kokkos::View<T*>(std::string(label), required);
    else
        Kokkos::realloc(Kokkos::WithoutInitializing, view, required);
}

}  // namespace symmetrix_oam_detail

template <typename Precision> struct NequIPOAMArenaTestAccess;

// Kokkos owner for the version-2 NequIP-OAM contract.  The validated model
// metadata is owned by NequIPOAMModel; execution uses the immutable Kokkos E3
// schedules below and generated host tensor-product owners.
template <typename Precision>
class NequIPOAMKokkos {
public:
    explicit NequIPOAMKokkos(
        const std::string& filename, const std::string& host_plugin_path = {},
        const std::string& cuda_plugin_path = {},
        const symmetrix::execution::CudaR1ArtifactIdentity&
            cuda_artifact_identity = {},
        std::string host_active_target = {},
        std::string host_artifact_origin = {},
        std::string requested_profile = {},
        std::string requested_edge_variant = {},
        std::string requested_precision = {});
    ~NequIPOAMKokkos();

    void set_streamed_edges(std::string mode);
    void set_execution_plan_request(std::string algorithm, std::string profile);
    void set_execution_edge_variant(std::string variant);
    std::string execution_edge_variant() const;
    std::string execution_edge_variant_reason() const;
    void set_execution_plan_qualification(const std::string& plan_id,
        const std::string& artifact_identity, const std::string& backend,
        const std::string& precision, const std::string& schedule,
        const std::string& capability, const std::string& empirical_status,
        std::optional<std::size_t> estimated_bytes,
        std::size_t available_bytes, std::size_t reserve_bytes);
    std::string streamed_edges_mode() const { return streamed_edges_mode_; }
    bool supports_streamed_edges() const { return true; }
    bool supports_stress() const { return true; }
    std::string execution_backend() const;
    std::string execution_diagnostic() const { return last_execution_diagnostic_; }
    std::size_t execution_workspace_bytes() const;
    std::size_t execution_topology_bytes() const;
    std::size_t execution_linear_workspace_bytes() const;
    std::size_t execution_packed_linear_workspace_bytes() const;
    std::size_t execution_active_edge_count() const { return edge_count(); }
    std::size_t execution_chunk_count() const { return execution_chunk_count_; }
    std::string execution_reduction_mode() const { return execution_reduction_mode_; }
    std::size_t execution_arena_active_bytes() const
    {
        return transient_arena_state_.active_bytes;
    }
    std::size_t execution_arena_retained_capacity_bytes() const
    {
        return transient_arena_state_.retained_capacity_bytes;
    }
    std::size_t execution_arena_peak_transient_bytes() const
    {
        return transient_arena_state_.peak_transient_bytes;
    }
    std::size_t execution_arena_replacements() const
    {
        return transient_arena_state_.replacements;
    }
    const std::string& execution_arena_high_water_reason() const
    {
        return transient_arena_state_.high_water_reason;
    }
    std::size_t execution_arena_runtime_subview_count() const
    {
        return transient_arena_runtime_subview_count_;
    }
    std::size_t execution_arena_canary_initializations() const
    {
        return transient_arena_canary_initializations_;
    }
    bool execution_arena_canaries_intact() const;
    std::size_t execution_bilinear_workspace_bytes() const;
    std::size_t execution_independent_graph_temporary_bytes() const;
    std::uint64_t graph_generation() const { return graph_generation_; }
    std::uint64_t geometry_generation() const { return geometry_generation_; }

    void compute_node_energies_forces(
        int num_nodes,
        std::span<const int> node_types,
        std::span<const int> num_neigh,
        std::span<const int> neigh_indices,
        std::span<const int> neigh_types,
        std::span<const double> xyz,
        std::span<const double> r);

    std::uint64_t prepare_graph(
        int num_nodes,
        std::span<const int> node_types,
        std::span<const int> num_neigh,
        std::span<const int> neigh_indices,
        std::span<const int> neigh_types);
    void prepare_geometry(
        std::uint64_t graph_generation,
        std::span<const double> reference_positions,
        std::span<const double> reference_xyz,
        std::span<const double> cell,
        std::span<const double> inverse_cell,
        std::span<const int> pbc);
    void compute_prepared(
        std::uint64_t graph_generation,
        std::span<const double> xyz,
        std::span<const double> distances);
    void reduce_forces(std::uint64_t graph_generation);
    void reduce_stress(double volume, std::uint64_t graph_generation);
    void compute_stress();

    // --- Family adapter interface (family_interface.hpp) -------------------
    // The lifecycle names above are already the neutral ones; these declare the
    // remaining interface surface.  Each delegates to the loader, readiness, or
    // state this evaluator already owns, so nothing here adds a second
    // execution path.
    symmetrix::execution::FamilyCapabilities family_capabilities() const;
    symmetrix::execution::Program compile_program() const;
    symmetrix::execution::ArtifactReadiness load_family_artifact(
        const symmetrix::execution::ArtifactRequest& request);
    bool family_artifact_ready() const;
    symmetrix::execution::ArtifactReadiness family_artifact_readiness() const;
    // The readiness of one artifact class, so a caller can ask about either
    // target without loading anything.
    symmetrix::execution::ArtifactReadiness artifact_readiness_for(
        const std::string& target) const;
    symmetrix::execution::FamilyDiagnostics family_diagnostics() const;
    // --- Tiered implementation registry ------------------------------------
    symmetrix::execution::ImplementationRegistry
    family_implementation_registry() const;
    symmetrix::execution::ImplementationBindingReport
    family_binding_report() const;
    // The plan this evaluator selected and the tier every operator node of its
    // compiled program binds.  The report is the shape every family publishes,
    // so a consumer reads one surface across families; the per-node bindings
    // are the plan's own, and each binding's `selected_implementation_id` is
    // the implementation the execution path actually dispatches.
    const symmetrix::execution::ExecutionPlanReport& execution_plan_report() const
    {
        return execution_plan_;
    }
    bool execution_plan_memory_available() const
    {
        return streamed_edges_mode_ == "direct" && execution_plan_memory_available_;
    }
    std::string execution_plan_empirical_qualification(
        const std::string& plan_id) const;
    std::string execution_plan_qualification_artifact_identity(
        const std::string& plan_id) const;
    std::string execution_loaded_artifact_identity() const;
    std::string execution_forward_owner_schedule() const;
    std::string execution_reverse_owner_schedule() const;
    std::string execution_tiling_policy() const;
    std::string execution_vector_policy() const;
    std::string execution_shared_memory_policy() const;
    std::string execution_register_budget_policy() const;
    std::string execution_direct_geometry_policy() const;
    int execution_channel_tile() const;
    int execution_scalar_width() const;
    std::string execution_required_isa() const;
    std::string execution_reduction_policy() const;
    int execution_target_compute_capability() const;
    std::string execution_precision_policy() const;
    std::string execution_schedule_qualification() const;
    std::string execution_artifact_origin() const
    { return host_artifact_origin_; }
    std::size_t execution_full_message_launch_count() const
    { return edge_full_message_launches_; }

    const std::vector<int>& atomic_numbers() const { return model_.atomic_numbers(); }
    double r_cut() const { return model_.r_cut(); }
    std::size_t num_layers() const { return model_.num_layers(); }
    const std::string& last_execution_diagnostic() const { return last_execution_diagnostic_; }
    bool jit_oam_host_plugin_ready() const;
    std::string jit_oam_host_plugin_path() const;
    std::string jit_oam_host_plugin_artifact_id() const;
    bool cuda_r1_ready() const { return cuda_r1_ready_; }
    std::string cuda_r1_reason() const { return cuda_r1_reason_; }
    std::string cuda_r1_artifact_id() const { return cuda_r1_artifact_id_; }
    std::string cuda_r1_plugin_path() const { return cuda_r1_plugin_path_; }
    std::string cuda_r1_manifest_path() const { return cuda_r1_manifest_path_; }
    std::size_t cuda_r1_launch_count() const { return cuda_r1_launch_count_; }
    std::size_t cuda_r1_fallback_count() const { return cuda_r1_fallback_count_; }
    std::int32_t cuda_r1_grid_blocks() const { return cuda_r1_grid_blocks_; }

    // The neural readout remains in model precision, but physical energies
    // and directed-force reductions use double accumulators.  In particular,
    // exposing double values here must reflect the arithmetic that produced
    // them rather than a post-reduction promotion in the Python calculator.
    std::vector<double> node_energies;
    std::vector<Precision> node_forces;
    std::vector<double> atom_forces;
    std::vector<double> stress_tensor;

private:
    std::size_t conditioned_self_bytes() const;
    std::size_t conditioned_self_layer_count() const;
    void validate_graph_generation(std::uint64_t graph_generation) const;
    void validate_geometry_generation(std::uint64_t graph_generation) const;
    void validate_completed_generation(std::uint64_t graph_generation) const;
    bool receiver_owned_execution() const;
    // Whether every layer admits the portable, compiler-free edge product on
    // this execution space.  A device build has no compiler-free path, so it
    // stays generated-only.
    bool has_portable_edge_implementation() const;
    // Whether generated edge code actually serves this evaluator's execution.
    // The direct-mode dispatcher runs the receiver-forward/source-reverse ABI
    // v7 host artifact
    // or the generated CUDA plugin; the edge-owned (ABI v1) host artifact
    // serves only the explicit debug_v1 mode.  The tier registry reads this
    // rather than mere artifact readiness, so an artifact that is loaded but
    // never dispatched does not claim the generated tier.
    bool generated_edges_serve() const;
    symmetrix::execution::PlanCandidateSpace plan_candidate_space() const;
    // Recompute the plan report from the current mode, artifact readiness, and
    // layer state.  Called wherever one of those changes, so the report is a
    // statement about the current configuration rather than a cached hope.
    void refresh_execution_plan_report();
    std::string execution_implementation_label() const;
    void refresh_execution_state_diagnostic();
    std::string execution_artifact_id() const;
    std::size_t edge_count() const;
    std::size_t retained_workspace_bytes() const;
    symmetrix::oam::OAMMemoryLayout execution_memory_layout(
        symmetrix::oam::OAMRetentionPolicy policy) const;
    typename E3LinearKokkosT<Precision>::WorkspaceShape
    execution_linear_workspace_shape() const;
    typename E3TensorProductKokkosT<Precision>::BilinearWorkspaceShape
    execution_bilinear_workspace_shape() const;
    std::string execution_memory_layout_identity(
        symmetrix::oam::OAMRetentionPolicy policy) const;
    void ensure_execution_arena();
    void initialize_execution_arena_canaries();
    void bind_execution_arena_retained_views();
    Kokkos::View<Precision**, Kokkos::LayoutRight> arena_matrix(
        const std::string& slot_name, std::size_t relative_values,
        std::size_t rows, std::size_t columns);
    template <typename T>
    Kokkos::View<T*> arena_vector(const std::string& slot_name,
        std::size_t relative_values, std::size_t values)
    {
        const auto& slot = active_memory_layout_.slot(slot_name);
        const std::size_t relative_bytes = symmetrix::oam::checked_oam_multiply(
            relative_values, sizeof(T));
        const std::size_t bytes = symmetrix::oam::checked_oam_multiply(
            values, sizeof(T));
        if (relative_bytes > slot.bytes || bytes > slot.bytes - relative_bytes)
            throw std::logic_error("OAM arena vector exceeds its planned slot");
        auto* pointer = reinterpret_cast<T*>(transient_arena_.data()
            + slot.offset + relative_bytes);
        if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0)
            throw std::logic_error("OAM arena vector is misaligned");
        ++transient_arena_runtime_subview_count_;
        return Kokkos::View<T*>(Kokkos::view_wrap(pointer), values);
    }
    public:
    void run_forward(
        std::span<const double> xyz, std::span<const double> distances,
        const char* operation, bool calculate_forces = true);
    void run_reverse(const char* operation);
    private:
    struct LayerState;
    void load_host_artifact(const std::string& host_plugin_path);
    void bind_host_artifact_owners();
    void initialize_cuda_r1(
        const std::string& plugin_path,
        const symmetrix::execution::CudaR1ArtifactIdentity& identity);
    void prepare_conditioned_self_plan(
        LayerState& state, const nlohmann::json& self_connection);
    void apply_conditioned_self_forward(
        const LayerState& state,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
        Kokkos::View<Precision**, Kokkos::LayoutRight> output);
    void apply_conditioned_self_reverse(
        const LayerState& state,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> input_adjoint);
    std::vector<symmetrix::execution::OAMCudaLayerExpectation>
    cuda_r1_layer_expectations() const;
    bool execution_space_is_host() const;
    bool execution_space_is_cuda() const;
    std::string execution_target() const;
    std::string cuda_r1_state_text() const;
    void update_diagnostic(const char* operation);
    std::string execution_kernel_summary() const;
    struct LayerState {
        E3LinearKokkosT<Precision> pre_linear;
        SymmetrixJitOAMHostForwardOwnerV1 generated_forward_owner = nullptr;
        SymmetrixJitOAMHostReverseOwnerV1 generated_reverse_owner = nullptr;
        SymmetrixJitOAMHostReceiverForwardOwnerV4 receiver_forward_owner = nullptr;
        SymmetrixJitOAMHostSourceReverseOwnerV4 source_reverse_owner = nullptr;
        int generated_channel_tile = 1;
        bool generated_channel_tiled = false;
        // The portable (compiler-free) edge product: the generic Kokkos
        // implementation of this layer's edge tensor product with the per-edge
        // radial weights.  Every layer carries one, and the direct path selects
        // it for a layer no generated owner serves, so a host family stays
        // runnable with no artifact at all.  The generated owners remain the
        // production implementation on every backend that has one.
        std::unique_ptr<E3TensorProductKokkosT<Precision>> edge_tensor_product;
        std::uint32_t generated_layer = 0;
        E3LinearKokkosT<Precision> post_linear;
        std::unique_ptr<E3LinearKokkosT<Precision>> self_linear;
        std::unique_ptr<E3TensorProductKokkosT<Precision>> self_tensor_product;
        struct ConditionedSelfPlan {
            // One row per serialized UVW path. Rows retain block identity and
            // use 64-bit offsets so device products cannot truncate.
            Kokkos::View<std::int64_t**, Kokkos::LayoutRight> path_data;
            // Per-model effective multiplicity matrices. Attribute values are
            // folded here once; angular constants and masks remain separate.
            Kokkos::View<Precision*> coefficients;
            Kokkos::View<Precision*> wigner_diagonal;
            Kokkos::View<Precision*> output_mask;
            std::size_t coefficients_per_model = 0;
            int path_count = 0;
            int output_width = 0;
            std::size_t original_tensor_bytes = 0;
            bool eligible = false;
        };
        ConditionedSelfPlan conditioned_self;
        // Lane kernels address every element through the self tensor product's
        // own irrep blocks, so they are only selected when that decomposition
        // matches the runtime node plans.  A node-major-equivalent (adjacent
        // same-angular) decomposition takes the definition-order fallback
        // instead of silently re-indexing the row.
        bool self_lane_kernels = false;
        int input_width = 0;
        int prelinear_width = 0;
        int edge_width = 0;
        int tp_output_width = 0;
        int pre_gate_width = 0;
        int output_width = 0;
        int weight_count = 0;
        int scalar_dimension = 0;
        int gate_dimension = 0;
        int gated_dimension = 0;
        Precision normalization_factor = Precision(1);
        Kokkos::View<int*> edge_harmonic_indices;
        // Gate/gated index maps stay in their definition (node-major) order;
        // the gate kernels convert them to lane addressing.  gated_block_plan
        // holds one (offset, multiplicity, width) row per gated-irrep block
        // with offsets relative to the gated region; gated_gate_channels holds
        // one gate position per gated channel in definition order (-1 when the
        // layer has no gate irreps).
        Kokkos::View<int**, Kokkos::LayoutRight> gated_block_plan;
        Kokkos::View<int*> gated_gate_channels;
        // Lane-major conversion plan and definition-order mask for the layer's
        // tensor-product output block (the per-node message adjoint epilogue).
        Kokkos::View<int**, Kokkos::LayoutRight> tp_output_block_plan;
        Kokkos::View<Precision*> tp_output_mask;
        // Irrep layout of the node attributes consumed by the self
        // tensor-product fast path (its irreps_in2).
        Kokkos::View<int**, Kokkos::LayoutRight> self_attribute_block_plan;
        // Input features can expand after the initial attribute layer.
        Kokkos::View<int**, Kokkos::LayoutRight> input_block_plan;
        // Irrep layout of the pre-gate tensor (self tensor-product output).
        Kokkos::View<int**, Kokkos::LayoutRight> pre_gate_block_plan;
    };

    struct RuntimeLayerState {
        Kokkos::View<Precision**, Kokkos::LayoutRight> input;
        Kokkos::View<Precision**, Kokkos::LayoutRight> prelinear;
        Kokkos::View<Precision**, Kokkos::LayoutRight> pre_gate;
        Kokkos::View<Precision**, Kokkos::LayoutRight> output;
        Kokkos::View<Precision*> reverse_source_partials;
        Kokkos::View<Precision*> reverse_geometry_partials;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_source_features;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_edge_features;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_edge_weights;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_edge_messages;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_source_adjoint;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_edge_adjoint;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_output_adjoint;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_weights_adjoint;
        std::size_t chunk_edge_capacity = 0;
        // Node-major operands and adjoints for the portable edge product.  The
        // generic tensor product addresses node-major rows while the node
        // features and message buffers above are lane-major, so the portable
        // path keeps one node-major source, one node-major source adjoint, and
        // one node-major message row per receiver chunk.
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_source_node;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_source_adjoint_node;
        Kokkos::View<Precision**, Kokkos::LayoutRight> chunk_messages_node;
        // Widths the retained chunk views were allocated with: a reused state
        // must be re-allocated when a caller asks for different trailing
        // extents, because the subviews and the generated callbacks assume the
        // parent row stride equals the requested width.
        int chunk_prelinear_width = 0;
        int chunk_edge_width = 0;
        int chunk_weight_count = 0;
        int chunk_output_width = 0;
    };

    void ensure_chunk_workspace(
        RuntimeLayerState& runtime, std::size_t edge_capacity,
        int prelinear_width, int edge_width, int weight_count,
        int output_width);
    public:
    // Permutes every node row between the definition (node-major) layout and
    // the lane-major layout described by `plan` (one (offset, multiplicity,
    // width) row per irrep block).  Source and destination must not alias.
    void permute_row_layout(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> source,
        Kokkos::View<Precision**, Kokkos::LayoutRight> destination,
        Kokkos::View<int**, Kokkos::LayoutRight> plan, bool to_lane) const;

    // Writes a lane-major node row for every node from the per-model-type
    // attribute table, permuting the definition-order attributes into `plan`.
    void gather_attribute_rows(
        Kokkos::View<Precision**, Kokkos::LayoutRight> destination,
        Kokkos::View<int**, Kokkos::LayoutRight> plan) const;
    private:

    KokkosLiveObjectGuard kokkos_live_object_guard_;
    NequIPOAMModel model_;
    std::unique_ptr<symmetrix::execution::OAMHostPlugin> oam_host_plugin_;
    std::unique_ptr<symmetrix::execution::OAMHostPluginV2Handle> oam_host_plugin_v2_;
    std::unique_ptr<symmetrix::execution::OAMCudaPlugin> cuda_r1_plugin_;
    symmetrix::execution::CudaR1ArtifactIdentity cuda_r1_identity_;
    std::string cuda_r1_plugin_path_;
    std::string cuda_r1_artifact_id_;
    std::string cuda_r1_manifest_path_;
    std::string cuda_r1_reason_;
    // Identity the host loader validated, retained so the adapter can report
    // the artifact it admitted rather than re-reading its manifest.
    std::string host_artifact_generation_fingerprint_;
    std::string host_artifact_semantic_fingerprint_;
    std::string host_artifact_structure_fingerprint_;
    bool cuda_r1_ready_ = false;
    std::size_t cuda_r1_launch_count_ = 0;
    std::size_t cuda_r1_fallback_count_ = 0;
    std::int32_t cuda_r1_grid_blocks_ = 0;
    bool qualified_sm120_fp32_ = false;
    std::size_t conditioned_self_forward_launch_count_ = 0;
    std::size_t conditioned_self_reverse_launch_count_ = 0;
    std::vector<NequIPOAMRadialSet<Precision>> radial_sets_;
    std::vector<LayerState> layer_states_;
    std::vector<RuntimeLayerState> runtime_layers_;
    Kokkos::View<unsigned char*> transient_arena_;
    symmetrix::oam::OAMTransientArenaState transient_arena_state_;
    symmetrix::oam::OAMMemoryLayout active_memory_layout_;
    std::size_t transient_arena_runtime_subview_count_ = 0;
    std::size_t transient_arena_canary_initializations_ = 0;
    std::string transient_arena_canary_layout_identity_;
    typename E3LinearKokkosT<Precision>::WorkspaceShape linear_workspace_shape_;
    // OAM linears execute serially on one evaluator stream. Sharing their
    // packed-GEMM scratch changes only buffer lifetime, not the contraction
    // schedule, and avoids one graph-sized packing arena per operator.
    std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>
        linear_workspace_ =
            std::make_shared<typename E3LinearKokkosT<Precision>::Workspace>();
    std::shared_ptr<typename E3TensorProductKokkosT<Precision>::BilinearWorkspace>
        bilinear_workspace_ = std::make_shared<
            typename E3TensorProductKokkosT<Precision>::BilinearWorkspace>();
    E3LinearKokkosT<Precision> readout_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> node_attributes_;
    Kokkos::View<int*> type_to_model_;
    std::vector<int> type_to_model_host_;
    std::vector<double> node_attributes_host_;
    std::vector<double> per_type_shift_;
    std::vector<double> per_type_scale_;
    std::vector<int> model_atomic_numbers_;
    std::vector<double> zbl_exponents_;
    std::vector<double> zbl_coefficients_;
    double zbl_cutoff_ = 0.0;
    double zbl_cutoff_p_ = 6.0;
    double zbl_half_prefactor_ = 0.0;
    int node_attribute_width_ = 0;
    int max_edge_l_ = 0;
    int edge_harmonic_count_ = 1;
    std::size_t receiver_chunk_limit_ = 128;

    int num_nodes_ = 0;
    std::vector<int> node_types_host_;
    std::vector<int> num_neigh_host_;
    std::vector<int> neigh_indices_host_;
    std::vector<int> neigh_types_host_;
    std::vector<int> edge_receivers_host_;
    std::vector<int> first_neigh_host_;
    std::vector<int> source_edge_offsets_host_;
    std::vector<int> source_edge_indices_host_;
    std::vector<int> reverse_segment_sources_host_;
    std::vector<int> reverse_segment_begin_host_;
    std::vector<int> reverse_segment_end_host_;
    std::vector<int> reverse_segment_rowptr_host_;
    std::vector<int> reverse_segment_order_host_;
    std::vector<double> reference_positions_host_;
    std::vector<double> reference_xyz_host_;
    std::vector<double> cell_host_;
    std::vector<double> inverse_cell_host_;
    std::vector<int> pbc_host_;
    std::size_t execution_chunk_count_ = 0;
    std::string execution_reduction_mode_ = "serial";
    // Layers whose self tensor product executes on the definition-order
    // fallback instead of the lane kernels (an inapplicable definition, or a
    // decomposition that only matches the runtime node plans up to adjacent
    // same-angular aggregation).
    std::vector<int> self_definition_order_layers_;
    // Measured edge-dispatch counters.  A layer without a selected owner
    // cannot be constructed and a dispatch only runs a loaded generated or
    // receiver-owned callback, so the reported generic counters are the
    // measured difference between the regions and dispatches performed and the
    // ones that ran generated code rather than constants.
    std::size_t edge_regions_ = 0;
    std::size_t edge_generated_regions_ = 0;
    std::size_t edge_dispatches_ = 0;
    std::size_t edge_generated_dispatches_ = 0;
    std::size_t edge_full_message_launches_ = 0;
    std::string host_active_target_;
    std::string host_artifact_origin_;
    Kokkos::View<int*> node_types_;
    Kokkos::View<int*> num_neigh_;
    Kokkos::View<int*> neigh_indices_;
    Kokkos::View<int*> neigh_types_;
    Kokkos::View<int*> edge_receivers_;
    Kokkos::View<int*> first_neigh_;
    Kokkos::View<int*> source_edge_offsets_;
    Kokkos::View<int*> source_edge_indices_;
    Kokkos::View<int*> reverse_segment_sources_;
    Kokkos::View<int*> reverse_segment_begin_;
    Kokkos::View<int*> reverse_segment_end_;
    Kokkos::View<int*> reverse_segment_rowptr_;
    Kokkos::View<int*> reverse_segment_order_;
    Kokkos::View<double*> receiver_owned_distance_adjoint_;
    // Lane-major irrep layout of the first layer's node input; the initial
    // feature gather permutes the definition-order attribute rows into it.
    Kokkos::View<int**, Kokkos::LayoutRight> node_feature_block_plan_;
    Kokkos::View<Precision*> receiver_owned_harmonic_adjoint_;
    Kokkos::View<Precision*> active_harmonics_;
    Kokkos::View<Precision*> active_harmonic_gradients_;
    Kokkos::View<Precision*> active_xyz_;
    Kokkos::View<double*> active_distances_;
    Kokkos::View<Precision*> active_shuffled_xyz_;
    Kokkos::View<Precision*> active_geometry_adjoint_;
    // Six symmetric virial components and the scaled row-major tensor stay
    // resident; stress export transfers only the final nine doubles.
    Kokkos::View<double*> active_stress_components_;
    Kokkos::View<double*> active_stress_tensor_;
    // Reduced atom forces accumulated by reduce_forces; retained so a repeated
    // reduction with an unchanged node count never rebuilds the buffer.
    Kokkos::View<double*> active_reduced_forces_;
    Kokkos::View<double*> per_type_shift_device_;
    Kokkos::View<double*> per_type_scale_device_;
    Kokkos::View<double*> zbl_exponents_device_;
    Kokkos::View<double*> zbl_coefficients_device_;
    Kokkos::View<double*> zbl_pair_zsum_device_;
    Kokkos::View<double*> zbl_pair_prefactor_device_;
    // Physical energy contributions are accumulated in double even for an
    // FP32 neural model.  The receiver sum remains ordered and deterministic.
    Kokkos::View<double*> active_zbl_energies_;
    Kokkos::View<double*> active_node_energies_;
    std::uint64_t generation_counter_ = 0;
    std::uint64_t graph_generation_ = 0;
    std::uint64_t geometry_generation_ = 0;
    std::uint64_t completed_generation_ = 0;
    bool graph_prepared_ = false;
    bool geometry_prepared_ = false;
    std::string streamed_edges_mode_ = "direct";
    std::string execution_edge_variant_request_ = "auto";
    std::string last_execution_diagnostic_;
    // The selected plan and its per-node tier bindings.  Refreshed wherever the
    // mode, the artifact readiness, or the layer state changes.
    symmetrix::execution::ExecutionPlanReport execution_plan_;
    bool execution_plan_memory_available_ = false;
    struct PlanQualification {
        bool admitted = false;
        std::string empirical_status = "required";
        std::string reason = "empirical qualification is required";
        std::string artifact_identity;
        std::optional<std::size_t> estimated_bytes;
        std::string layout_identity;
    };
    PlanQualification execution_plan_speed_qualification_;
    PlanQualification execution_plan_capacity_qualification_;

    friend struct NequIPOAMArenaTestAccess<Precision>;
};

template <typename Precision>
struct NequIPOAMArenaTestAccess {
    static void corrupt_first_guard(NequIPOAMKokkos<Precision>& evaluator)
    {
        if (evaluator.active_memory_layout_.slots.empty())
            throw std::logic_error("OAM arena has no active guard to corrupt");
        const auto offset = evaluator.active_memory_layout_.slots.front().offset
            - symmetrix::oam::oam_arena_canary_bytes;
        Kokkos::deep_copy(Kokkos::subview(evaluator.transient_arena_, offset),
            static_cast<unsigned char>(0));
    }
};

extern template class NequIPOAMKokkos<float>;
extern template class NequIPOAMKokkos<double>;
