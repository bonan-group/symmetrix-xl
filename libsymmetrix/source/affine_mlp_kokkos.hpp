#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <memory>

#include "nlohmann/json.hpp"

template<typename Precision>
class AffineMLPKokkosT {
public:
    using TapeView = Kokkos::View<
        Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    struct Workspace {
        Kokkos::View<Precision*> value_storage,adjoint_storage;
        int rows=0;
        std::size_t generation=0;

        int workspace_rows() const { return rows; }
        std::size_t bytes() const {
            return sizeof(Precision)
                *(value_storage.size()+adjoint_storage.size());
        }
        void clear();
    };

    AffineMLPKokkosT() = default;
    explicit AffineMLPKokkosT(const nlohmann::json& definition);

    int input_size() const;
    int output_size() const;
    bool supports_conditioned_input(int dynamic_input_size) const;
    bool supports_final_linear_factorization() const;
    bool supports_final_linear_factorization(int dynamic_input_size) const;
    int prefix_output_size() const;
    int prefix_output_size(int dynamic_input_size) const;
    bool factorized_prefix_is_identity(int dynamic_input_size) const;
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        final_linear_weight() const;
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        final_linear_weight(int dynamic_input_size) const;
    void prepare_final_linear_weight_phi_major(int dynamic_input_size) const;
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        final_linear_weight_phi_major(int dynamic_input_size) const;
    Kokkos::View<const Precision*> final_linear_bias() const;
    // Deterministic layer-order parameter pack for generated runtimes:
    // Linear stores row-major [output,input] weight then bias; LayerNorm
    // stores gamma then beta; SiLU stores no parameters.
    Kokkos::View<const Precision*> packed_runtime_parameters() const;
    void evaluate(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
    void evaluate_conditioned(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void evaluate_conditioned_prefix(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void evaluate_conditioned_indexed(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const int*> source_edge_types,
        Kokkos::View<const int*> target_node_indices,
        Kokkos::View<const int*> node_types,
        int first_edge,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            source_contribution_table,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            target_contribution_table,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void evaluate_conditioned_prefix_indexed(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const int*> source_edge_types,
        Kokkos::View<const int*> target_node_indices,
        Kokkos::View<const int*> node_types,
        int first_edge,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            source_contribution_table,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            target_contribution_table,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void reverse_from_tape(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
    void reverse_prefix_from_tape(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> prefix_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
    void prepare_conditioned_weight(int active_input_size) const;
    int workspace_rows() const;
    std::size_t workspace_bytes() const;
    void clear_workspace();
    // A workspace can be reused by sequential MLP invocations. Starting a
    // forward pass on another MLP that shares it invalidates the prior tape.
    void set_workspace(std::shared_ptr<Workspace> workspace);
    void forward_impl(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions);

private:
    enum LayerType : int { Linear = 0, LayerNorm = 1, SiLU = 2 };
    Kokkos::View<int*,Kokkos::SharedSpace> types;
    Kokkos::View<int*,Kokkos::SharedSpace> input_sizes;
    Kokkos::View<int*,Kokkos::SharedSpace> output_sizes;
    Kokkos::View<Precision*,Kokkos::SharedSpace> eps;
    Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> weights;
    Kokkos::View<Kokkos::View<Precision*>*,Kokkos::SharedSpace> biases;
    Kokkos::View<TapeView*,Kokkos::SharedSpace> values;
    Kokkos::View<TapeView*,Kokkos::SharedSpace> adjoints;
    std::shared_ptr<Workspace> workspace_=std::make_shared<Workspace>();
    mutable Kokkos::View<Precision**,Kokkos::LayoutRight> conditioned_weight;
    mutable int conditioned_input_size=-1;
    mutable Kokkos::View<Precision**,Kokkos::LayoutRight>
        final_linear_weight_phi_major_storage;
    mutable int final_linear_weight_phi_major_input_size=-1;
    Kokkos::View<Precision*> packed_runtime_parameter_storage;
    int tape_batch_size=-1,tape_input_size=-1,tape_layer_count=-1;
    std::size_t tape_workspace_generation=0;
    void prepare(int batch_size,int active_input_size,int layer_count);
    void forward(Kokkos::View<const Precision**,Kokkos::LayoutRight> input);
public:
    // nvcc requires functions enclosing Kokkos extended lambdas to be public.
    void forward_impl(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
        int layer_count);
    void forward_indexed_impl(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const int*> source_edge_types,
        Kokkos::View<const int*> target_node_indices,
        Kokkos::View<const int*> node_types,
        int first_edge,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            source_contribution_table,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            target_contribution_table,
        int layer_count);
    void forward_impl(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
        Kokkos::View<const int*> source_edge_types,
        Kokkos::View<const int*> target_node_indices,
        Kokkos::View<const int*> node_types,
        int first_edge,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            source_contribution_table,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            target_contribution_table,
        int layer_count);
    void reverse_from_tape_impl(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
        int layer_count);
};

using AffineMLPKokkos = AffineMLPKokkosT<double>;
using AffineMLPFloatKokkos = AffineMLPKokkosT<float>;
