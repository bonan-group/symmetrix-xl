#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <memory>
#include <utility>

#include "e3nn_kokkos.hpp"

template<typename Precision>
class E3ProductBasisKokkosT {
public:
    enum class InputLayout { native,block_major };
    struct Tensor { Kokkos::View<Precision*> values; std::vector<int> shape; };
    E3ProductBasisKokkosT() = default;
    explicit E3ProductBasisKokkosT(const nlohmann::json& data);
    int input_dimension() const { return input_dimension_; }
    int output_dimension() const { return output_dimension_; }
    bool is_agnostic() const { return agnostic; }
    bool uses_compiled_plan() const { return !compiled_blocks.empty(); }
    bool uses_standard_host_plan() const {
        return compiled_host_standard_term_count!=0;
    }
    int compiled_term_count() const { return compiled_terms; }
    Kokkos::View<const Precision***,Kokkos::LayoutRight>
    runtime_compiled_coefficients() const { return compiled_coefficients; }
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
    packed_input_adjoint() const {
        return compiled_host_packed_input_adjoint;
    }
    Kokkos::View<Precision*> prepare_block_major_input(int batch) {
        prepare(batch);
        const std::size_t size=static_cast<std::size_t>(batch)
            *angular_dimension*num_features;
        if(compiled_host_block_major_input.extent(0)<size)
            Kokkos::realloc(compiled_host_block_major_input,size);
        return Kokkos::subview(
            compiled_host_block_major_input,std::make_pair<std::size_t>(0,size));
    }
    const E3LinearKokkosT<Precision>& output_linear() const { return linear; }
    E3LinearKokkosT<Precision>& mutable_output_linear() { return linear; }
    void set_profile_name(std::string name) {
        linear.set_profile_name(std::move(name));
    }
    std::size_t workspace_bytes() const;
    std::size_t feature_major_workspace_bytes() const;
    void set_e3_linear_workspace(
        std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace> workspace) {
        linear.set_workspace(std::move(workspace));
    }
    void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                  Kokkos::View<const Precision**,Kokkos::LayoutRight> skip,
                  Kokkos::View<const int*> elements,
                  Kokkos::View<Precision**,Kokkos::LayoutRight> output,
                  bool validate_elements=true,
                  InputLayout input_layout=InputLayout::native,
                  int skip_active_dimension=-1);
    void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                 Kokkos::View<const int*> elements,
                 Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> skip_adjoint,
                 bool validate_elements=true,
                 bool emit_native_input_adjoint=true,
                 InputLayout input_layout=InputLayout::native,
                 int skip_active_dimension=-1);
    void to_feature_major(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> source);

private:
    struct Contraction { int correlation; std::vector<Tensor> u; std::vector<Tensor> weights; };
    struct CompiledBlock {
        int angular_offset=0,output_offset=0,width=0,num_elements=0,term_offset=0;
    };
    Irreps input{"1x0e"}, output{"1x0e"};
    E3LinearKokkosT<Precision> linear;
    int input_dimension_=0,output_dimension_=0,num_features=0,angular_dimension=0;
    bool use_sc=false,agnostic=false;
    int compiled_terms=0,compiled_component_count=0;
    std::vector<Contraction> contractions;
    std::vector<CompiledBlock> compiled_blocks;
    Kokkos::View<int**,Kokkos::LayoutRight> compiled_term_data;
    Kokkos::View<int**,Kokkos::LayoutRight> compiled_term_native_layout;
    Kokkos::View<int**,Kokkos::LayoutRight> compiled_component_data;
    Kokkos::View<int**,Kokkos::LayoutRight> compiled_angular_native_layout;
    Kokkos::View<int**,Kokkos::LayoutRight> compiled_angular_packed_layout;
    Kokkos::View<Precision***,Kokkos::LayoutRight> compiled_coefficients;
    int compiled_host_standard_term_count=0;
    Kokkos::View<Precision***,Kokkos::LayoutRight>
        compiled_host_standard_weights;
    Kokkos::View<Precision***,Kokkos::LayoutRight>
        compiled_host_angular_major_input,
        compiled_host_angular_major_adjoint;
    Kokkos::View<Precision*> compiled_host_block_major_input;
    Kokkos::View<Precision**,Kokkos::LayoutRight>
        compiled_host_packed_input_adjoint,
        compiled_host_packed_input_adjoint_storage;
    Kokkos::View<Precision***,Kokkos::LayoutRight> feature_major, feature_major_adjoint;
    Kokkos::View<Precision**,Kokkos::LayoutRight> contracted, contracted_adjoint;
    Kokkos::View<Precision***,Kokkos::LayoutRight> feature_major_storage,
        feature_major_adjoint_storage;
    Kokkos::View<Precision**,Kokkos::LayoutRight> contracted_storage,
        contracted_adjoint_storage;
    void prepare(int batch);
    void prepare_compiled_angular_input(int batch);
    void prepare_compiled_angular_adjoint(int batch);
};

using E3ProductBasisKokkos = E3ProductBasisKokkosT<double>;
using E3ProductBasisFloatKokkos = E3ProductBasisKokkosT<float>;
