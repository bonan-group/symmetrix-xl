#include "affine_mlp_kokkos.hpp"
#include "affine_mlp.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "KokkosBlas.hpp"
#include "tools_kokkos.hpp"

namespace {
template<typename Precision>
std::vector<Precision> tensor_values(const nlohmann::json& value)
{
    return value.at("values").get<std::vector<Precision>>();
}
}

template<typename Precision>
AffineMLPKokkosT<Precision>::AffineMLPKokkosT(const nlohmann::json& definition)
{
    AffineMLP validated(definition);
    const int count = definition.at("layers").size();
    types = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp types",count);
    input_sizes = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp inputs",count);
    output_sizes = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp outputs",count);
    eps = Kokkos::View<Precision*,Kokkos::SharedSpace>("affine mlp eps",count);
    weights = decltype(weights)(Kokkos::view_alloc("affine mlp weights",Kokkos::SequentialHostInit),count);
    biases = decltype(biases)(Kokkos::view_alloc("affine mlp biases",Kokkos::SequentialHostInit),count);
    values = decltype(values)(Kokkos::view_alloc("affine mlp values",Kokkos::SequentialHostInit),count+1);
    adjoints = decltype(adjoints)(Kokkos::view_alloc("affine mlp adjoints",Kokkos::SequentialHostInit),count+1);
    std::vector<Precision> packed_parameters;
    int previous = -1;
    for (int index=0; index<count; ++index) {
        const auto& layer = definition.at("layers").at(index);
        const auto type = layer.at("type").get<std::string>();
        if (type == "linear") {
            types(index) = Linear;
            const auto shape = layer.at("weight").at("shape").get<std::vector<int>>();
            if (shape.size()!=2) throw std::invalid_argument("Kokkos affine linear weight must be rank two.");
            input_sizes(index)=shape[1]; output_sizes(index)=shape[0];
            const auto weight_values=tensor_values<Precision>(layer.at("weight"));
            const auto bias_values=tensor_values<Precision>(layer.at("bias"));
            weights(index)=toKokkosView(
                "affine mlp weight",weight_values,shape[0],shape[1]);
            biases(index)=toKokkosView("affine mlp bias",bias_values);
            packed_parameters.insert(
                packed_parameters.end(),weight_values.begin(),weight_values.end());
            packed_parameters.insert(
                packed_parameters.end(),bias_values.begin(),bias_values.end());
        } else if (type == "layer_norm") {
            types(index)=LayerNorm;
            input_sizes(index)=layer.at("normalized_shape").at(0).get<int>();
            output_sizes(index)=input_sizes(index); eps(index)=layer.at("eps").get<Precision>();
            const auto gamma=tensor_values<Precision>(layer.at("weight"));
            const auto beta=tensor_values<Precision>(layer.at("bias"));
            weights(index)=toKokkosView("affine layer norm gamma",gamma,1,gamma.size());
            biases(index)=toKokkosView("affine layer norm beta",beta);
            packed_parameters.insert(
                packed_parameters.end(),gamma.begin(),gamma.end());
            packed_parameters.insert(
                packed_parameters.end(),beta.begin(),beta.end());
        } else if (type == "silu") {
            if (previous<0) throw std::invalid_argument("Kokkos affine SiLU cannot be the first layer.");
            types(index)=SiLU; input_sizes(index)=previous; output_sizes(index)=previous;
        } else throw std::invalid_argument("Unsupported Kokkos affine MLP layer: "+type);
        if (previous>=0 && input_sizes(index)!=previous)
            throw std::invalid_argument("Kokkos affine MLP layer dimensions are inconsistent.");
        previous=output_sizes(index);
    }
    packed_runtime_parameter_storage=toKokkosView(
        "affine mlp packed runtime parameters",packed_parameters);
}

template<typename Precision>
int AffineMLPKokkosT<Precision>::input_size() const { return input_sizes(0); }
template<typename Precision>
int AffineMLPKokkosT<Precision>::output_size() const { return output_sizes(output_sizes.size()-1); }
template<typename Precision>
bool AffineMLPKokkosT<Precision>::supports_conditioned_input(int dynamic_input_size) const
{
    return types.size()>0&&types(0)==Linear&&dynamic_input_size>0
        &&dynamic_input_size<input_size();
}

template<typename Precision>
bool AffineMLPKokkosT<Precision>::supports_final_linear_factorization() const
{
    return types.extent_int(0)>1&&types(types.extent_int(0)-1)==Linear;
}

template<typename Precision>
bool AffineMLPKokkosT<Precision>::supports_final_linear_factorization(
    int dynamic_input_size) const
{
    return types.extent_int(0)>0
        &&types(types.extent_int(0)-1)==Linear
        &&supports_conditioned_input(dynamic_input_size);
}

template<typename Precision>
int AffineMLPKokkosT<Precision>::prefix_output_size() const
{
    if(!supports_final_linear_factorization())
        throw std::logic_error(
            "Kokkos affine MLP does not end in a factorable linear layer.");
    return input_sizes(types.extent_int(0)-1);
}

template<typename Precision>
int AffineMLPKokkosT<Precision>::prefix_output_size(
    int dynamic_input_size) const
{
    if(!supports_final_linear_factorization(dynamic_input_size))
        throw std::logic_error(
            "Kokkos affine MLP does not support the requested conditioned "
            "final-linear factorization.");
    if(types.extent_int(0)==1) return dynamic_input_size;
    return input_sizes(types.extent_int(0)-1);
}

template<typename Precision>
bool AffineMLPKokkosT<Precision>::factorized_prefix_is_identity(
    int dynamic_input_size) const
{
    if(!supports_final_linear_factorization(dynamic_input_size))
        throw std::logic_error(
            "Kokkos affine MLP does not support the requested conditioned "
            "final-linear factorization.");
    return types.extent_int(0)==1;
}

template<typename Precision>
Kokkos::View<const Precision**,Kokkos::LayoutRight>
AffineMLPKokkosT<Precision>::final_linear_weight() const
{
    if(!supports_final_linear_factorization())
        throw std::logic_error(
            "Kokkos affine MLP does not end in a factorable linear layer.");
    return weights(types.extent_int(0)-1);
}

template<typename Precision>
Kokkos::View<const Precision**,Kokkos::LayoutRight>
AffineMLPKokkosT<Precision>::final_linear_weight(
    int dynamic_input_size) const
{
    if(!supports_final_linear_factorization(dynamic_input_size))
        throw std::logic_error(
            "Kokkos affine MLP does not support the requested conditioned "
            "final-linear factorization.");
    if(types.extent_int(0)>1) return weights(types.extent_int(0)-1);
    prepare_conditioned_weight(dynamic_input_size);
    return conditioned_weight;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::prepare_final_linear_weight_phi_major(
    int dynamic_input_size) const
{
    const auto row_major=final_linear_weight(dynamic_input_size);
    const int weight_size=row_major.extent_int(0);
    const int phi_dimension=row_major.extent_int(1);
    if(final_linear_weight_phi_major_input_size==dynamic_input_size
        &&final_linear_weight_phi_major_storage.extent_int(0)==phi_dimension
        &&final_linear_weight_phi_major_storage.extent_int(1)==weight_size)
        return;

    // A changed conditioned width replaces model-owned storage. Ensure an old
    // generated launch cannot retain the previous allocation while it is freed.
    if(final_linear_weight_phi_major_storage.data()!=nullptr)
        Kokkos::fence("Replace phi-major affine final-linear weight");
    Kokkos::realloc(
        Kokkos::WithoutInitializing,
        final_linear_weight_phi_major_storage,phi_dimension,weight_size);
    auto phi_major=final_linear_weight_phi_major_storage;
    Kokkos::parallel_for(
        "prepare phi-major affine final-linear weight",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{phi_dimension,weight_size}),
        KOKKOS_LAMBDA(int phi,int weight) {
            phi_major(phi,weight)=row_major(weight,phi);
        });
    final_linear_weight_phi_major_input_size=dynamic_input_size;
}

template<typename Precision>
Kokkos::View<const Precision**,Kokkos::LayoutRight>
AffineMLPKokkosT<Precision>::final_linear_weight_phi_major(
    int dynamic_input_size) const
{
    if(!supports_final_linear_factorization(dynamic_input_size))
        throw std::logic_error(
            "Kokkos affine MLP does not support the requested conditioned "
            "final-linear factorization.");
    const int weight_size=output_sizes(types.extent_int(0)-1);
    const int phi_dimension=prefix_output_size(dynamic_input_size);
    if(final_linear_weight_phi_major_input_size!=dynamic_input_size
        ||final_linear_weight_phi_major_storage.extent_int(0)!=phi_dimension
        ||final_linear_weight_phi_major_storage.extent_int(1)!=weight_size)
        throw std::logic_error(
            "Kokkos affine MLP phi-major final-linear weight was not prepared.");
    return final_linear_weight_phi_major_storage;
}

template<typename Precision>
Kokkos::View<const Precision*> AffineMLPKokkosT<Precision>::final_linear_bias() const
{
    if(types.extent_int(0)==0||types(types.extent_int(0)-1)!=Linear)
        throw std::logic_error(
            "Kokkos affine MLP does not end in a factorable linear layer.");
    return biases(types.extent_int(0)-1);
}

template<typename Precision>
Kokkos::View<const Precision*>
AffineMLPKokkosT<Precision>::packed_runtime_parameters() const
{
    return packed_runtime_parameter_storage;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::Workspace::clear()
{
    if(value_storage.data()!=nullptr||adjoint_storage.data()!=nullptr)
        Kokkos::fence("Clear shared affine MLP workspace");
    value_storage={};
    adjoint_storage={};
    rows=0;
    ++generation;
}

template<typename Precision>
int AffineMLPKokkosT<Precision>::workspace_rows() const
{
    return workspace_?workspace_->workspace_rows():0;
}

template<typename Precision>
std::size_t AffineMLPKokkosT<Precision>::workspace_bytes() const
{
    return workspace_?workspace_->bytes():0;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::clear_workspace()
{
    for(int index=0;index<values.extent_int(0);++index) {
        values(index)={};
        adjoints(index)={};
    }
    if(workspace_) workspace_->clear();
    tape_batch_size=-1;
    tape_input_size=-1;
    tape_layer_count=-1;
    tape_workspace_generation=0;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::set_workspace(
    std::shared_ptr<Workspace> workspace)
{
    if(!workspace)
        throw std::invalid_argument(
            "Kokkos affine MLP workspace must not be null.");
    if(workspace_==workspace) return;
    if(workspace_&&(workspace_->value_storage.data()!=nullptr
        ||workspace_->adjoint_storage.data()!=nullptr))
        Kokkos::fence("Replace shared affine MLP workspace");
    for(int index=0;index<values.extent_int(0);++index) {
        values(index)={};
        adjoints(index)={};
    }
    workspace_=std::move(workspace);
    tape_batch_size=-1;
    tape_input_size=-1;
    tape_layer_count=-1;
    tape_workspace_generation=0;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::prepare(
    int batch_size,int active_input_size,int layer_count)
{
    if(layer_count<=0||layer_count>types.extent_int(0))
        throw std::invalid_argument(
            "Kokkos affine MLP active layer count is invalid.");
    if(!workspace_)
        throw std::logic_error("Kokkos affine MLP workspace is not set.");

    std::size_t required=
        static_cast<std::size_t>(batch_size)*active_input_size;
    for(int layer=0;layer<layer_count;++layer)
        required+=static_cast<std::size_t>(batch_size)*output_sizes(layer);
    const bool grow_values=workspace_->value_storage.extent(0)<required;
    const bool grow_adjoints=workspace_->adjoint_storage.extent(0)<required;
    if((grow_values||grow_adjoints)
        &&(workspace_->value_storage.data()!=nullptr
            ||workspace_->adjoint_storage.data()!=nullptr))
        Kokkos::fence("Grow shared affine MLP workspace");
    if(grow_values)
        Kokkos::realloc(
            Kokkos::WithoutInitializing,workspace_->value_storage,required);
    if(grow_adjoints)
        Kokkos::realloc(
            Kokkos::WithoutInitializing,workspace_->adjoint_storage,required);
    workspace_->rows=std::max(workspace_->rows,batch_size);
    ++workspace_->generation;

    std::size_t offset=0;
    auto* value_base=workspace_->value_storage.data();
    auto* adjoint_base=workspace_->adjoint_storage.data();
    values(0)=TapeView(
        value_base==nullptr?nullptr:value_base+offset,
        batch_size,active_input_size);
    adjoints(0)=TapeView(
        adjoint_base==nullptr?nullptr:adjoint_base+offset,
        batch_size,active_input_size);
    offset+=static_cast<std::size_t>(batch_size)*active_input_size;
    for (int layer=0; layer<layer_count; ++layer) {
        values(layer+1)=TapeView(
            value_base==nullptr?nullptr:value_base+offset,
            batch_size,output_sizes(layer));
        adjoints(layer+1)=TapeView(
            adjoint_base==nullptr?nullptr:adjoint_base+offset,
            batch_size,output_sizes(layer));
        offset+=static_cast<std::size_t>(batch_size)*output_sizes(layer);
    }
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::prepare_conditioned_weight(
    int active_input_size) const
{
    if(conditioned_input_size==active_input_size) return;
    if(!supports_conditioned_input(active_input_size))
        throw std::invalid_argument("Kokkos affine MLP conditioned weight dimensions are inconsistent.");
    Kokkos::realloc(
        Kokkos::WithoutInitializing,conditioned_weight,output_sizes(0),active_input_size);
    auto full_weight=weights(0);
    auto compact_weight=conditioned_weight;
    Kokkos::parallel_for(
        "prepare conditioned affine weight",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{output_sizes(0),active_input_size}),
        KOKKOS_LAMBDA(int row,int column) {
            compact_weight(row,column)=full_weight(row,column);
        });
    conditioned_input_size=active_input_size;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::forward(Kokkos::View<const Precision**,Kokkos::LayoutRight> input)
{
    if(input.extent(1)!=static_cast<std::size_t>(input_size()))
        throw std::invalid_argument("Kokkos affine MLP input dimensions are inconsistent.");
    forward_impl(input,{});
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::forward_impl(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions)
{
    forward_impl(input,row_contributions,types.extent_int(0));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::forward_impl(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
    int layer_count)
{
    forward_impl(
        input,row_contributions,{},{},{},0,{},{},layer_count);
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::forward_indexed_impl(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const int*> source_edge_types,
    Kokkos::View<const int*> target_node_indices,
    Kokkos::View<const int*> node_types,
    int first_edge,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        source_contribution_table,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        target_contribution_table,
    int layer_count)
{
    forward_impl(
        input,{},source_edge_types,target_node_indices,node_types,first_edge,
        source_contribution_table,target_contribution_table,layer_count);
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::forward_impl(
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
    int layer_count)
{
    const bool row_conditioned=row_contributions.data()!=nullptr;
    const bool indexed_conditioned=
        source_contribution_table.data()!=nullptr
        ||target_contribution_table.data()!=nullptr;
    const bool conditioned=row_conditioned||indexed_conditioned;
    if(row_conditioned&&indexed_conditioned)
        throw std::invalid_argument(
            "Kokkos affine MLP accepts one conditioned input form at a time.");
    if(row_conditioned&&(!supports_conditioned_input(input.extent(1))
        ||row_contributions.extent(0)!=input.extent(0)
        ||row_contributions.extent(1)!=static_cast<std::size_t>(output_sizes(0))))
        throw std::invalid_argument("Kokkos affine MLP conditioned dimensions are inconsistent.");
    const std::size_t last_edge=first_edge<0
        ?std::size_t(0):static_cast<std::size_t>(first_edge)+input.extent(0);
    if(indexed_conditioned&&(
        !supports_conditioned_input(input.extent(1))
        ||first_edge<0
        ||source_edge_types.extent(0)<last_edge
        ||target_node_indices.extent(0)<last_edge
        ||node_types.extent(0)==0
        ||source_contribution_table.extent(0)==0
        ||target_contribution_table.extent(0)==0
        ||source_contribution_table.extent(1)
            !=static_cast<std::size_t>(output_sizes(0))
        ||target_contribution_table.extent(1)
            !=static_cast<std::size_t>(output_sizes(0))))
        throw std::invalid_argument(
            "Kokkos affine MLP indexed conditioned dimensions are "
            "inconsistent.");
    prepare(input.extent(0),input.extent(1),layer_count);
    if(conditioned) prepare_conditioned_weight(input.extent(1));
    ordered_kokkos_deep_copy(values(0),input);
    for (int layer=0; layer<layer_count;) {
        if(types(layer)==LayerNorm&&layer+1<layer_count&&types(layer+1)==SiLU) {
            auto source=values(layer);
            auto target=values(layer+2);
            auto gamma=weights(layer);
            auto beta=biases(layer);
            const Precision epsilon=eps(layer);
            Kokkos::parallel_for(
                "affine fused layer norm silu",target.extent(0),
                KOKKOS_LAMBDA(int sample) {
                    Precision mean=Precision(0);
                    for(int column=0;column<source.extent(1);++column)
                        mean+=source(sample,column);
                    mean/=source.extent(1);
                    Precision variance=Precision(0);
                    for(int column=0;column<source.extent(1);++column) {
                        const Precision delta=source(sample,column)-mean;
                        variance+=delta*delta;
                    }
                    variance/=source.extent(1);
                    const Precision inverse=Precision(1)/Kokkos::sqrt(variance+epsilon);
                    for(int column=0;column<source.extent(1);++column) {
                        const Precision value=(source(sample,column)-mean)*inverse
                            *gamma(0,column)+beta(column);
                        target(sample,column)=value/(Precision(1)+Kokkos::exp(-value));
                    }
                });
            layer+=2;
            continue;
        }
        auto source=values(layer); auto target=values(layer+1);
        if (types(layer)==Linear) {
            auto weight=weights(layer); auto bias=biases(layer);
            const bool use_gemm=source.extent(0)>4;
            if(use_gemm) {
                if(conditioned&&layer==0)
                    KokkosBlas::gemm(
                        "N","T",Precision(1),source,conditioned_weight,Precision(0),target);
                else
                    KokkosBlas::gemm("N","T",Precision(1),source,weight,Precision(0),target);
                Kokkos::parallel_for(
                    "affine linear epilogue",target.size(),KOKKOS_LAMBDA(int flat) {
                        const int sample=flat/target.extent(1);
                        const int row=flat%target.extent(1);
                        Precision contribution=Precision(0);
                        if(layer==0&&row_conditioned)
                            contribution=row_contributions(sample,row);
                        else if(layer==0&&indexed_conditioned) {
                            const int edge=first_edge+sample;
                            const int target_node=target_node_indices(edge);
                            contribution=source_contribution_table(
                                source_edge_types(edge),row)
                                +target_contribution_table(
                                    node_types(target_node),row);
                        }
                        target(sample,row)+=bias(row)+contribution;
                    });
            } else {
                Kokkos::parallel_for("affine linear",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{target.extent(0),target.extent(1)}),
                    KOKKOS_LAMBDA(int sample,int row) {
                        Precision value=bias(row);
                        for (int column=0; column<source.extent(1); ++column) value+=weight(row,column)*source(sample,column);
                        if(layer==0&&row_conditioned)
                            value+=row_contributions(sample,row);
                        else if(layer==0&&indexed_conditioned) {
                            const int edge=first_edge+sample;
                            const int target_node=target_node_indices(edge);
                            value+=source_contribution_table(
                                source_edge_types(edge),row)
                                +target_contribution_table(
                                    node_types(target_node),row);
                        }
                        target(sample,row)=value;
                    });
            }
        } else if (types(layer)==LayerNorm) {
            auto gamma=weights(layer); auto beta=biases(layer); const Precision epsilon=eps(layer);
            Kokkos::parallel_for("affine layer norm",target.extent(0),KOKKOS_LAMBDA(int sample) {
                Precision mean=Precision(0); for (int column=0; column<source.extent(1); ++column) mean+=source(sample,column); mean/=source.extent(1);
                Precision variance=Precision(0); for (int column=0; column<source.extent(1); ++column) { const Precision delta=source(sample,column)-mean; variance+=delta*delta; } variance/=source.extent(1);
                const Precision inverse=Precision(1)/Kokkos::sqrt(variance+epsilon);
                for (int column=0; column<source.extent(1); ++column) target(sample,column)=(source(sample,column)-mean)*inverse*gamma(0,column)+beta(column);
            });
        } else Kokkos::parallel_for("affine silu",target.size(),KOKKOS_LAMBDA(int flat) {
            const int sample=flat/target.extent(1), column=flat%target.extent(1); const Precision value=source(sample,column); target(sample,column)=value/(Precision(1)+Kokkos::exp(-value));
        });
        ++layer;
    }
    tape_batch_size=input.extent(0);
    tape_input_size=input.extent(1);
    tape_layer_count=layer_count;
    tape_workspace_generation=workspace_->generation;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision**,Kokkos::LayoutRight> output)
{
    forward(input); ordered_kokkos_deep_copy(output,values(types.size()));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::evaluate_conditioned(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output)
{
    forward_impl(input,row_contributions);
    ordered_kokkos_deep_copy(output,values(types.size()));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::evaluate_conditioned_indexed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const int*> source_edge_types,
    Kokkos::View<const int*> target_node_indices,
    Kokkos::View<const int*> node_types,
    int first_edge,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        source_contribution_table,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        target_contribution_table,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output)
{
    forward_indexed_impl(
        input,source_edge_types,target_node_indices,node_types,first_edge,
        source_contribution_table,target_contribution_table,
        types.extent_int(0));
    ordered_kokkos_deep_copy(output,values(types.size()));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::evaluate_conditioned_prefix(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output)
{
    const int dynamic_input_size=input.extent_int(1);
    if(!supports_final_linear_factorization(dynamic_input_size))
        throw std::logic_error(
            "Kokkos affine MLP does not support the requested conditioned "
            "final-linear factorization.");
    const int layer_count=types.extent_int(0)-1;
    if(output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(
            prefix_output_size(dynamic_input_size))
        ||row_contributions.extent(0)!=input.extent(0)
        ||row_contributions.extent(1)
            !=static_cast<std::size_t>(output_sizes(0)))
        throw std::invalid_argument(
            "Kokkos affine MLP prefix output dimensions are inconsistent.");
    if(layer_count==0) {
        prepare_conditioned_weight(dynamic_input_size);
        ordered_kokkos_deep_copy(output,input);
        tape_batch_size=input.extent_int(0);
        tape_input_size=dynamic_input_size;
        tape_layer_count=0;
        tape_workspace_generation=0;
        return;
    }
    forward_impl(input,row_contributions,layer_count);
    ordered_kokkos_deep_copy(output,values(layer_count));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::evaluate_conditioned_prefix_indexed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const int*> source_edge_types,
    Kokkos::View<const int*> target_node_indices,
    Kokkos::View<const int*> node_types,
    int first_edge,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        source_contribution_table,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        target_contribution_table,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output)
{
    const int dynamic_input_size=input.extent_int(1);
    if(!supports_final_linear_factorization(dynamic_input_size))
        throw std::logic_error(
            "Kokkos affine MLP does not support the requested indexed "
            "conditioned final-linear factorization.");
    const int layer_count=types.extent_int(0)-1;
    const std::size_t last_edge=first_edge<0
        ?std::size_t(0):static_cast<std::size_t>(first_edge)+input.extent(0);
    if(output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(
            prefix_output_size(dynamic_input_size))
        ||first_edge<0
        ||source_edge_types.extent(0)<last_edge
        ||target_node_indices.extent(0)<last_edge
        ||node_types.extent(0)==0
        ||source_contribution_table.extent(0)==0
        ||target_contribution_table.extent(0)==0
        ||source_contribution_table.extent(1)
            !=static_cast<std::size_t>(output_sizes(0))
        ||target_contribution_table.extent(1)
            !=static_cast<std::size_t>(output_sizes(0)))
        throw std::invalid_argument(
            "Kokkos affine MLP indexed prefix dimensions are inconsistent.");
    if(layer_count==0) {
        prepare_conditioned_weight(dynamic_input_size);
        ordered_kokkos_deep_copy(output,input);
        tape_batch_size=input.extent_int(0);
        tape_input_size=dynamic_input_size;
        tape_layer_count=0;
        tape_workspace_generation=0;
        return;
    }
    forward_indexed_impl(
        input,source_edge_types,target_node_indices,node_types,first_edge,
        source_contribution_table,target_contribution_table,layer_count);
    ordered_kokkos_deep_copy(output,values(layer_count));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint)
{
    forward(input);
    reverse_from_tape(output_adjoint,input_adjoint);
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::reverse_from_tape(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint)
{
    reverse_from_tape_impl(
        output_adjoint,input_adjoint,types.extent_int(0));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::reverse_prefix_from_tape(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> prefix_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint)
{
    if(tape_layer_count==0) {
        if(tape_batch_size<0
            ||prefix_adjoint.extent(0)
                !=static_cast<std::size_t>(tape_batch_size)
            ||prefix_adjoint.extent(1)
                !=static_cast<std::size_t>(tape_input_size)
            ||input_adjoint.extent(0)!=prefix_adjoint.extent(0)
            ||input_adjoint.extent(1)!=prefix_adjoint.extent(1))
            throw std::invalid_argument(
                "Kokkos affine MLP identity-prefix reverse dimensions are "
                "inconsistent.");
        ordered_kokkos_deep_copy(input_adjoint,prefix_adjoint);
        return;
    }
    if(tape_layer_count<0||!supports_final_linear_factorization())
        throw std::logic_error(
            "Kokkos affine MLP does not end in a factorable linear layer.");
    reverse_from_tape_impl(
        prefix_adjoint,input_adjoint,types.extent_int(0)-1);
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::reverse_from_tape_impl(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
    int layer_count)
{
    if(!workspace_
        ||tape_workspace_generation!=workspace_->generation)
        throw std::logic_error(
            "Kokkos affine MLP reverse tape was invalidated by a shared "
            "workspace user.");
    if(tape_batch_size<0
        ||tape_layer_count!=layer_count
        ||output_adjoint.extent(0)!=static_cast<std::size_t>(tape_batch_size)
        ||output_adjoint.extent(1)
            !=static_cast<std::size_t>(output_sizes(layer_count-1))
        ||input_adjoint.extent(0)!=static_cast<std::size_t>(tape_batch_size)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(tape_input_size))
        throw std::invalid_argument("Kokkos affine MLP reverse tape dimensions are inconsistent.");
    ordered_kokkos_deep_copy(adjoints(layer_count),output_adjoint);
    for (int layer=layer_count-1; layer>=0;) {
        if(types(layer)==SiLU&&layer>0&&types(layer-1)==LayerNorm) {
            auto source=values(layer-1);
            auto source_adj=adjoints(layer-1);
            auto target_adj=adjoints(layer+1);
            auto gamma=weights(layer-1);
            auto beta=biases(layer-1);
            const Precision epsilon=eps(layer-1);
            Kokkos::parallel_for(
                "affine fused reverse layer norm silu",source.extent(0),
                KOKKOS_LAMBDA(int sample) {
                    const int width=source.extent(1);
                    Precision mean=Precision(0);
                    for(int column=0;column<width;++column)
                        mean+=source(sample,column);
                    mean/=width;
                    Precision variance=Precision(0);
                    for(int column=0;column<width;++column) {
                        const Precision delta=source(sample,column)-mean;
                        variance+=delta*delta;
                    }
                    variance/=width;
                    const Precision inverse=Precision(1)/Kokkos::sqrt(variance+epsilon);
                    Precision sum=Precision(0);
                    Precision sum_normalized=Precision(0);
                    for(int column=0;column<width;++column) {
                        const Precision normalized=(source(sample,column)-mean)*inverse;
                        const Precision value=normalized*gamma(0,column)+beta(column);
                        const Precision probability=Precision(1)/(Precision(1)+Kokkos::exp(-value));
                        const Precision scaled=target_adj(sample,column)
                            *(probability+value*probability*(Precision(1)-probability))
                            *gamma(0,column);
                        sum+=scaled;
                        sum_normalized+=scaled*normalized;
                    }
                    for(int column=0;column<width;++column) {
                        const Precision normalized=(source(sample,column)-mean)*inverse;
                        const Precision value=normalized*gamma(0,column)+beta(column);
                        const Precision probability=Precision(1)/(Precision(1)+Kokkos::exp(-value));
                        const Precision scaled=target_adj(sample,column)
                            *(probability+value*probability*(Precision(1)-probability))
                            *gamma(0,column);
                        source_adj(sample,column)=inverse
                            *(width*scaled-sum-normalized*sum_normalized)/width;
                    }
                });
            layer-=2;
            continue;
        }
        auto source=values(layer); auto source_adj=adjoints(layer); auto target_adj=adjoints(layer+1);
        if (types(layer)==Linear) {
            auto weight=weights(layer);
            if(source_adj.extent(0)>4) {
                if(layer==0&&tape_input_size<input_size())
                    KokkosBlas::gemm(
                        "N","N",Precision(1),target_adj,conditioned_weight,Precision(0),source_adj);
                else
                    KokkosBlas::gemm("N","N",Precision(1),target_adj,weight,Precision(0),source_adj);
            } else {
                Kokkos::parallel_for("affine reverse linear",source_adj.size(),KOKKOS_LAMBDA(int flat) {
                    const int sample=flat/source_adj.extent(1), column=flat%source_adj.extent(1); Precision value=Precision(0);
                    for (int row=0; row<target_adj.extent(1); ++row) value+=weight(row,column)*target_adj(sample,row); source_adj(sample,column)=value;
                });
            }
        } else if (types(layer)==LayerNorm) {
            ordered_kokkos_deep_copy(source_adj,Precision(0));
            auto gamma=weights(layer); const Precision epsilon=eps(layer);
            Kokkos::parallel_for("affine reverse layer norm",source.extent(0),KOKKOS_LAMBDA(int sample) {
                const int width=source.extent(1); Precision mean=Precision(0); for (int i=0;i<width;++i) mean+=source(sample,i); mean/=width;
                Precision variance=Precision(0); for (int i=0;i<width;++i) { const Precision delta=source(sample,i)-mean; variance+=delta*delta; } variance/=width;
                const Precision inverse=Precision(1)/Kokkos::sqrt(variance+epsilon); Precision sum=Precision(0),sum_normalized=Precision(0);
                for (int i=0;i<width;++i) { const Precision scaled=target_adj(sample,i)*gamma(0,i); sum+=scaled; sum_normalized+=scaled*(source(sample,i)-mean)*inverse; }
                for (int i=0;i<width;++i) { const Precision scaled=target_adj(sample,i)*gamma(0,i); const Precision normalized=(source(sample,i)-mean)*inverse; source_adj(sample,i)=inverse*(width*scaled-sum-normalized*sum_normalized)/width; }
            });
        } else {
            ordered_kokkos_deep_copy(source_adj,Precision(0));
            Kokkos::parallel_for("affine reverse silu",source_adj.size(),KOKKOS_LAMBDA(int flat) {
                const int sample=flat/source_adj.extent(1), column=flat%source_adj.extent(1); const Precision value=source(sample,column); const Precision probability=Precision(1)/(Precision(1)+Kokkos::exp(-value)); source_adj(sample,column)=target_adj(sample,column)*(probability+value*probability*(Precision(1)-probability));
            });
        }
        --layer;
    }
    ordered_kokkos_deep_copy(input_adjoint,adjoints(0));
}

template class AffineMLPKokkosT<float>;
template class AffineMLPKokkosT<double>;
