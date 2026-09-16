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
        std::size_t bytes() const {
            return sizeof(Precision)*(packed_input.size()+packed_output.size());
        }
    };
    struct Instruction {
        int input_offset, output_offset, input_multiplicity, output_multiplicity, width, weight_offset;
        Precision path_weight;
        bool first_input_instruction=false,first_output_instruction=false,
            last_output_instruction=false,identity=false;
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
    bool try_compose_input_block(
        const E3LinearKokkosT<Precision>& inner,int width);
    void replace_block_with_identity(int width);
    void set_backend(const std::string& backend);
    std::string backend() const;
    std::string selected_backend(std::size_t samples) const;
    std::size_t workspace_bytes() const;
    void set_profile_name(std::string name) { profile_name_=std::move(name); }
    void set_workspace(std::shared_ptr<Workspace> workspace) {
        workspace_=std::move(workspace);
    }
    void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
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
private:
    struct RuntimeOutputBlock {
        int offset, multiplicity, width;
    };
    bool use_scalar_backend(std::size_t samples) const;
    int input_dimension_=0, output_dimension_=0;
    int active_output_prefix_dimension_=-1;
    bool all_input_blocks_covered_=false,all_output_blocks_covered_=false,
        output_mask_is_identity_=false,active_output_mask_is_identity_=false,
        uncovered_output_finalization_is_zero_=false;
    std::string profile_name_="unnamed";
    Backend backend_=Backend::automatic;
    std::shared_ptr<Workspace> workspace_=std::make_shared<Workspace>();
    std::vector<Instruction> instructions;
    std::vector<RuntimeOutputBlock> runtime_input_blocks;
    std::vector<RuntimeOutputBlock> runtime_output_blocks;
    Kokkos::View<Precision*> weights, bias, output_mask;
    Kokkos::View<int*> active_output_mask;
    Kokkos::View<int**,Kokkos::LayoutRight> runtime_input_block_plan;
};

template<typename Precision>
class E3TensorProductKokkosT {
public:
    struct Instruction {
        int input_1_offset, input_2_offset, output_offset;
        int multiplicity_1, multiplicity_2, output_multiplicity;
        int width_1, width_2, output_width, weight_offset;
        bool has_weight, uuu;
        Precision path_weight;
        Kokkos::View<Precision*> wigner;
        Kokkos::View<int**,Kokkos::LayoutRight> sparse_indices;
        Kokkos::View<Precision*> sparse_values;
        Kokkos::View<int*> component_offsets;
        int sparse_count=0;
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
    bool uses_mh1_fast_path() const { return mh1_fast_path; }
    std::string backend() const {
        return mh1_fast_path ? "official_kokkos" : "generic_kokkos";
    }
    std::string execution_backend() const;
    int channel_team_size() const;
    int harmonic_team_size() const;
    bool supports_direct_node_reverse() const;
    bool supports_execution_uvu() const {
        return mh1_fast_path&&mh1_direct_node_layout_;
    }
    bool supports_execution_direct_uvu() const { return supports_execution_uvu(); }
    int execution_multiplicity() const { return mh1_multiplicity_; }
    int execution_input_1_angular_dimension() const {
        return mh1_input_1_angular_dimension_;
    }
    int execution_instruction_count() const { return mh1_instruction_count_; }
    bool execution_output_mask_is_identity() const {
        return output_mask_is_identity_;
    }
    Kokkos::View<const Precision*> execution_output_mask() const {
        return output_mask;
    }
    Kokkos::View<const Precision*> execution_output_mask_ir_mul() const {
        return mh1_output_mask_ir_mul;
    }
    Kokkos::View<const int*> execution_input_1_mul_to_ir() const {
        return mh1_input_1_mul_to_ir;
    }
    Kokkos::View<const int*> execution_output_mul_to_ir() const {
        return mh1_output_mul_to_ir;
    }
    std::size_t workspace_bytes() const { return 0; }
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
    static constexpr int max_direct_node_output_components_=16;
    static constexpr int max_direct_harmonic_components_=16;
    static constexpr int direct_channel_tile_=64;
    static constexpr int direct_light_edge_team_size_=64;
    static constexpr int direct_edge_team_size_=128;
    int input_1_dimension_=0,input_2_dimension_=0,output_dimension_=0,weight_size_=0;
    bool mh1_fast_path=false,mh1_direct_node_layout_=false,
        output_mask_is_identity_=false;
    int mh1_instruction_count_=0,mh1_multiplicity_=0,
        mh1_input_1_angular_dimension_=0,mh1_max_output_width_=0;
    int mh1_cuda_channel_team_size_=0,mh1_cuda_harmonic_team_size_=0;
    std::vector<Instruction> instructions;
    std::vector<OutputBlock> output_blocks;
    Kokkos::View<Precision*> internal_weights, output_mask;
    Kokkos::View<int**,Kokkos::LayoutRight> mh1_instruction_data,
        mh1_sparse_indices,mh1_harmonic_terms,mh1_input_component_data,
        mh1_input_terms,mh1_input_instruction_term_offsets;
    Kokkos::View<int*> mh1_component_offsets,mh1_harmonic_offsets,
        mh1_input_term_offsets,mh1_input_1_mul_to_ir,mh1_output_mul_to_ir;
    Kokkos::View<Precision*> mh1_path_weights,mh1_sparse_values,
        mh1_harmonic_values,mh1_input_values,mh1_output_mask_ir_mul;
};

using E3LinearKokkos = E3LinearKokkosT<double>;
using E3LinearFloatKokkos = E3LinearKokkosT<float>;
using E3TensorProductKokkos = E3TensorProductKokkosT<double>;
using E3TensorProductFloatKokkos = E3TensorProductKokkosT<float>;
