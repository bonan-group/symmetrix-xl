#pragma once

#include <Kokkos_Core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "e3nn.hpp"
#include "radial_function_set_kokkos.hpp"

template<typename Precision>
class E3LinearKokkosT {
public:
    enum class Backend { automatic, scalar, packed_gemm };
    struct Workspace {
        Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input,
            packed_output;
        bool fixed_capacity=false;
        std::size_t bytes() const {
            return sizeof(Precision)*(packed_input.size()+packed_output.size());
        }
    };
    struct WorkspaceShape {
        std::size_t input_rows = 0;
        std::size_t input_columns = 0;
        std::size_t output_rows = 0;
        std::size_t output_columns = 0;
    };
    struct Instruction {
        int input_offset, output_offset, input_multiplicity, output_multiplicity, width, weight_offset;
        Precision path_weight;
        bool first_input_instruction=false,first_output_instruction=false,
            last_output_instruction=false,identity=false;
    };
    struct ScalarDensePlan {
        int input_size=0,output_size=0;
        Precision path_weight=Precision(0);
        Kokkos::View<const Precision*> weights,bias,output_mask;
    };
    E3LinearKokkosT() = default;
    explicit E3LinearKokkosT(const nlohmann::json& data);
    int input_dimension() const { return input_dimension_; }
    int output_dimension() const { return output_dimension_; }
    Kokkos::View<const Precision*> runtime_weights() const { return weights; }
    Kokkos::View<const Precision*> runtime_bias() const { return bias; }
    Kokkos::View<const Precision*> runtime_output_mask() const {
        return output_mask;
    }
    Kokkos::View<const int*> runtime_active_output_mask() const {
        return active_output_mask;
    }
    int active_output_prefix_dimension() const {
        return active_output_prefix_dimension_;
    }
    bool supports_packed_reverse() const {
        return all_input_blocks_covered_&&active_output_mask_is_identity_;
    }
    std::vector<Precision> runtime_parameters_ir_mul() const;
    bool has_instruction_width(int width) const;
    bool supports_identity_replacement(int width) const;
    bool supports_scalar_dense_plan() const;
    ScalarDensePlan scalar_dense_plan() const;
    bool try_compose_input_block(
        const E3LinearKokkosT<Precision>& inner,int width);
    void replace_block_with_identity(int width);
    void set_backend(const std::string& backend);
    std::string backend() const;
    std::string selected_backend(std::size_t samples) const;
    std::string specialization_family() const {
        return polar_specialization_ ? "polar_e3linear" : "generic";
    }
    std::string specialization_role() const { return specialization_role_; }
    std::string selected_family(std::size_t) const {
        return backend_==Backend::automatic&&polar_specialization_
            ?"polar_e3linear":"generic";
    }
    std::size_t specialized_forward_count() const {
        return specialized_forward_count_;
    }
    std::size_t specialized_reverse_count() const {
        return specialized_reverse_count_;
    }
    std::size_t specialization_fallback_count() const {
        return specialization_fallback_count_;
    }
    std::size_t workspace_bytes() const;
    WorkspaceShape lane_workspace_shape(std::size_t samples) const;
    void set_profile_name(std::string name) { profile_name_=std::move(name); }
    void set_workspace(std::shared_ptr<Workspace> workspace) {
        workspace_=std::move(workspace);
    }
    void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                  Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void evaluate_generic(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    // Lane-major variant of evaluate(): for a block at offset o with
    // multiplicity m and angular width d, element (channel a, component x)
    // is stored at o + x*m + a (components outer, channels contiguous)
    // instead of the node-major o + a*d + x.  Both the input and the output
    // view use the lane-major layout; the instruction set, weights, bias,
    // output mask, and path weights are shared with evaluate().
    void evaluate_lane(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void evaluate_to_packed(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> packed_output,
        bool initialize_uncovered=true,
        Kokkos::View<const Precision*> row_denominator={},
        Precision denominator_alpha=Precision(0),
        Precision denominator_beta=Precision(0)) const;
    void normalize_packed_rows(
        Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input,
        Kokkos::View<const Precision*> row_denominator,
        Precision denominator_alpha,Precision denominator_beta) const;
    void reverse_normalize_packed_rows(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            packed_input_adjoint,
        Kokkos::View<const Precision*> row_denominator,
        Precision denominator_alpha,Precision denominator_beta,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_ir_mul_adjoint,
        Kokkos::View<Precision*> row_denominator_adjoint,
        bool packed_input_is_normalized=true) const;
    void evaluate_packed_to_packed(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> packed_output) const;
    void evaluate_from_packed(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void evaluate_from_packed_to_block_major(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input,
        Kokkos::View<Precision*> block_major_output,int samples) const;
    void evaluate_ir_mul(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
                 bool initialize_input_adjoint=true) const;
    void reverse_generic(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
        bool initialize_input_adjoint=true) const;
    // Lane-major variant of reverse(): both adjoint views use the lane-major
    // element order (channel a, component x of a block at offset o with
    // multiplicity m sits at o + x*m + a).  input_adjoint is initialized to
    // zero first unless initialize_input_adjoint is false, in which case the
    // contribution is accumulated into the existing adjoint.
    void reverse_lane(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
        bool initialize_input_adjoint=true) const;
    void reverse_to_packed(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input_adjoint) const;
    void reverse_angular_major_to_packed(
        Kokkos::View<const Precision***,Kokkos::LayoutRight>
            angular_major_output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input_adjoint) const;
    void reverse_packed_to_packed(
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            packed_output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            packed_input_adjoint) const;
    void reverse_from_packed(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const;
    void reverse_ir_mul(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const;
    // nvcc requires an enclosing member that contains extended device lambdas
    // to be public, even though these two entry points are implementation
    // details used only by the selected and explicit-reference wrappers.
    void evaluate_impl(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output,
        bool use_scalar_path) const;
    void reverse_impl(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
        bool initialize_input_adjoint,bool use_scalar_path) const;
private:
    struct RuntimeOutputBlock {
        int offset, multiplicity, width;
    };
    bool use_scalar_backend(std::size_t samples) const;
public:
    // Applies (value + bias) * mask to a lane-major output batch, indexing the
    // definition-order bias and mask tables at each element's node-major
    // position.
    void finalize_lane_output(
        Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
private:
    int input_dimension_=0, output_dimension_=0;
    int active_output_prefix_dimension_=-1;
    bool all_input_blocks_covered_=false,all_output_blocks_covered_=false,
        output_mask_is_identity_=false,active_output_mask_is_identity_=false,
        uncovered_output_finalization_is_zero_=false;
    std::string profile_name_="unnamed";
    Backend backend_=Backend::automatic;
    bool specialization_requested_=false,polar_specialization_=false;
    std::string specialization_role_;
    mutable std::size_t specialized_forward_count_=0,
        specialized_reverse_count_=0,specialization_fallback_count_=0;
    std::shared_ptr<Workspace> workspace_=std::make_shared<Workspace>();
    std::vector<Instruction> instructions;
    // Unmerged instruction list for the lane-major kernels.  The float32 host
    // path coalesces adjacent same-angular couplings in *instructions*, which
    // is valid for the node-major schedule but not for lane addressing (each
    // merged row would have to be permuted separately).  evaluate_lane() and
    // reverse_lane() therefore iterate this pre-merge copy.
    std::vector<Instruction> lane_instructions;
    std::vector<RuntimeOutputBlock> runtime_input_blocks;
    std::vector<RuntimeOutputBlock> runtime_output_blocks;
    Kokkos::View<Precision*> weights, bias, output_mask;
    Kokkos::View<int*> active_output_mask;
    Kokkos::View<int**,Kokkos::LayoutRight> runtime_input_block_plan;
    // Per-output-block (offset, multiplicity, width) plan for the lane-major
    // kernels, which need the node-major index of a lane-major element.
    Kokkos::View<int**,Kokkos::LayoutRight> runtime_output_block_plan;
};

template<typename Precision>
class E3TensorProductKokkosT {
public:
    struct BilinearWorkspace {
        Kokkos::View<Precision**,Kokkos::LayoutRight> matrix,result,
            adjoint,k_adjoint;
        bool fixed_capacity=false;
        std::size_t bytes() const {
            return sizeof(Precision)*(matrix.size()+result.size()
                +adjoint.size()+k_adjoint.size());
        }
    };
    struct BilinearWorkspaceShape {
        std::size_t matrix_values=0,result_values=0,
            adjoint_values=0,k_adjoint_values=0;
        std::size_t forward_values() const {
            return matrix_values+result_values;
        }
        std::size_t reverse_values() const {
            return adjoint_values+k_adjoint_values;
        }
    };
    struct Instruction {
        int input_1_offset, input_2_offset, output_offset;
        int multiplicity_1, multiplicity_2, output_multiplicity;
        int width_1, width_2, output_width, weight_offset;
        bool has_weight, uuu, uvw;
        Precision path_weight;
        Kokkos::View<Precision*> wigner;
    };
    struct OutputBlock {
        int offset, multiplicity, width;
    };
    E3TensorProductKokkosT() = default;
    explicit E3TensorProductKokkosT(const nlohmann::json& data);
    int input_1_dimension() const { return input_1_dimension_; }
    int input_2_dimension() const { return input_2_dimension_; }
    int output_dimension() const { return output_dimension_; }
    int weight_size() const { return weight_size_; }
    bool has_internal_weights() const { return internal_weights.extent(0)!=0; }
    // True when the object carries a compiled direct-execution plan.  The plan
    // is family-owned state held behind an opaque handle (see the family
    // layer); an object that never installs one reports false and keeps the
    // generic schedule.  The accessors below report that plan's extents.
    bool uses_direct_execution_plan() const;
    std::string backend() const;
    std::string execution_backend() const;
    int channel_team_size() const;
    int harmonic_team_size() const;
    bool supports_direct_node_reverse() const;
    bool supports_execution_uvu() const;
    bool supports_execution_direct_uvu() const { return supports_execution_uvu(); }
    // Node-local self-connection fast path: every instruction is a full
    // (u,v,w) tensor product with internal weights, so the contraction
    // collapses to one GEMM per input-1 angular component instead of a
    // scalar per-node path loop.
    bool supports_node_bilinear_uvw() const;
    void evaluate_node_bilinear_uvw(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void reverse_node_bilinear_uvw(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint) const;
    // Lane-major variant of the node-local self-connection fast path: blocks
    // are addressed component-outer (offset + component*multiplicity +
    // channel) instead of node-major (offset + channel*width + component).
    // The GEMM structure, tiling, workspace, and wigner/mask indexing are
    // identical; only operand and adjoint addressing changes.
    bool supports_node_bilinear_uvw_lane() const {
        return supports_node_bilinear_uvw();
    }
    BilinearWorkspaceShape bilinear_workspace_shape(std::size_t samples) const;
    void set_bilinear_workspace(std::shared_ptr<BilinearWorkspace> workspace) {
        bilinear_workspace_=std::move(workspace);
    }
    std::size_t bilinear_workspace_bytes() const {
        return bilinear_workspace_->bytes();
    }
    void evaluate_node_bilinear_uvw_lane(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void reverse_node_bilinear_uvw_lane(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint) const;
    int execution_multiplicity() const;
    int execution_input_1_angular_dimension() const;
    int execution_instruction_count() const;
    bool execution_output_mask_is_identity() const {
        return output_mask_is_identity_;
    }
    Kokkos::View<const Precision*> execution_output_mask() const {
        return output_mask;
    }
    Kokkos::View<const Precision*> execution_output_mask_ir_mul() const;
    Kokkos::View<const int*> execution_input_1_mul_to_ir() const;
    Kokkos::View<const int*> execution_output_mul_to_ir() const;
    std::size_t workspace_bytes() const { return bilinear_workspace_->bytes(); }
    void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
                  Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
                  Kokkos::View<const Precision**,Kokkos::LayoutRight> weights,
                  Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
                 Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
                 Kokkos::View<const Precision**,Kokkos::LayoutRight> weights,
                 Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> weights_adjoint) const;
    // Adds source-node adjoints and overwrites both edge-adjoint outputs.
    bool try_reverse_from_nodes(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values,
        Kokkos::View<const int*> source_indices,
        int first_edge,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_weights,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            target_node_output_adjoint,
        Kokkos::View<const int*> target_indices,
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            source_node_input_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> edge_input_2_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> edge_weights_adjoint) const;
    // Fuses the final affine edge-weight projection with the UVU tensor
    // product and accumulates directly into receiver-owned node messages.
    // edge_phi, edge_linear_contribution, edge_input_2, and their adjoints are
    // local to the block that starts at first_edge. An empty
    // edge_linear_contribution means zero; a nonempty view is added to
    // bias+linear_weight*edge_phi before cutoff scaling and has no adjoint.
    // Topology, receiver offsets, and cutoff scales use global edge indices.
    // The destination node_messages is incremented.
    bool try_evaluate_execution_uvu_from_nodes(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values,
        Kokkos::View<const int*> source_indices,
        Kokkos::View<const int*> target_indices,
        Kokkos::View<const int*> active_receivers,
        Kokkos::View<const int*> receiver_offsets,
        int first_edge,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_phi,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> linear_weight,
        Kokkos::View<const Precision*> linear_bias,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            edge_linear_contribution,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
        Kokkos::View<const Precision*> edge_cutoff_scale,
        bool apply_edge_cutoff_scale,
        Kokkos::View<Precision**,Kokkos::LayoutRight> node_messages) const;
    // Reverse of try_evaluate_execution_uvu_from_nodes. source_edge_offsets
    // identify compact source-owned segments for this edge block, and
    // source_edge_indices stores global edge IDs for all blocks contiguously.
    // The source is recovered from each nonempty segment's first edge. The
    // source node adjoint is incremented; all three edge adjoints are
    // overwritten.
    // The fixed linear parameters are inference constants and have no
    // adjoints in this API.
    bool try_reverse_execution_uvu_from_nodes(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values,
        Kokkos::View<const int*> source_indices,
        Kokkos::View<const int*> target_indices,
        int first_edge,
        Kokkos::View<const int*> source_edge_offsets,
        Kokkos::View<const int*> source_edge_indices,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_phi,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> linear_weight,
        Kokkos::View<const Precision*> linear_bias,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            edge_linear_contribution,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
        Kokkos::View<const Precision*> edge_cutoff_scale,
        bool apply_edge_cutoff_scale,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            target_node_output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            source_node_input_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> edge_phi_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> edge_input_2_adjoint,
        Kokkos::View<Precision*> edge_cutoff_scale_adjoint) const;
    bool try_evaluate_execution_spline_uvu_from_nodes(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values,
        Kokkos::View<const int*> source_indices,
        Kokkos::View<const int*> target_indices,
        Kokkos::View<const int*> active_receivers,
        Kokkos::View<const int*> receiver_offsets,
        Kokkos::View<const int*> source_types,
        Kokkos::View<const int*> node_types,
        Kokkos::View<const double*> distances,
        Kokkos::View<const int*> spline_intervals,
        Kokkos::View<const Precision*> spline_coordinates,
        int type_count,
        RadialFunctionSetKokkos<Precision> spline,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
        Kokkos::View<Precision**,Kokkos::LayoutRight> node_messages,
        Kokkos::View<Precision*> node_density) const;
    bool try_reverse_execution_spline_uvu_from_nodes(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values,
        Kokkos::View<const int*> source_indices,
        Kokkos::View<const int*> target_indices,
        Kokkos::View<const int*> source_edge_offsets,
        Kokkos::View<const int*> source_edge_indices,
        Kokkos::View<const int*> source_types,
        Kokkos::View<const int*> node_types,
        Kokkos::View<const double*> distances,
        Kokkos::View<const int*> spline_intervals,
        Kokkos::View<const Precision*> spline_coordinates,
        int type_count,
        RadialFunctionSetKokkos<Precision> spline,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            target_node_output_adjoint,
        Kokkos::View<const Precision*> node_density_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            source_node_input_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> edge_input_2_adjoint,
        Kokkos::View<Precision*> distance_adjoint) const;
private:
    static constexpr int max_direct_node_input_components_=16;
    int node_tile_capacity(int remaining,int k_dimension,int result_width) const;
    void ensure_bilinear_workspace(
        int tile,int k_dimension,int result_width) const;
    static constexpr int max_direct_node_output_components_=16;
    static constexpr int max_direct_harmonic_components_=16;
    static constexpr int direct_channel_tile_=64;
    static constexpr int direct_light_edge_team_size_=64;
    static constexpr int direct_edge_team_size_=128;
    int input_1_dimension_=0,input_2_dimension_=0,output_dimension_=0,weight_size_=0;
    bool output_mask_is_identity_=false;
    std::vector<Instruction> instructions;
    std::vector<OutputBlock> output_blocks;
    Kokkos::View<Precision*> internal_weights, output_mask;
    std::shared_ptr<BilinearWorkspace> bilinear_workspace_=
        std::make_shared<BilinearWorkspace>();
    // Compiled direct-execution plan.  The layout is family-private and the
    // shared class stores the handle only, so a default-constructed object and
    // any object that does not install a plan run the generic schedule.  The
    // two try_* entry points stay public: nvcc rejects an extended
    // __host__ __device__ lambda whose enclosing member function is private or
    // protected, and the family-owned kernels these entry points call are such
    // lambdas.  They return false when no plan is installed or when the
    // caller's operands do not select it.
    struct DirectExecutionPlan;
    void initialize_direct_execution_plan(
        const nlohmann::json& data,const E3TensorProduct& validated);
public:
    bool try_evaluate_direct_execution(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> weights,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    bool try_reverse_direct_execution(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> weights,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> weights_adjoint) const;
private:
    std::shared_ptr<DirectExecutionPlan> direct_execution_plan_;
};

using E3LinearKokkos = E3LinearKokkosT<double>;
using E3LinearFloatKokkos = E3LinearKokkosT<float>;
using E3TensorProductKokkos = E3TensorProductKokkosT<double>;
using E3TensorProductFloatKokkos = E3TensorProductKokkosT<float>;
