#include "e3nn_product_kokkos.hpp"
#include "e3nn_product.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "tools_kokkos.hpp"
#include "standard_m0.hpp"

namespace {
constexpr int product_host_forward_feature_tile=4;
constexpr int product_host_standard_feature_tile=512;

template<typename Precision>
typename E3ProductBasisKokkosT<Precision>::Tensor load_tensor(
    const nlohmann::json& data)
{
    return {
        toKokkosView(
            "product tensor",
            data.at("values").get<std::vector<Precision>>()),
        data.at("shape").get<std::vector<int>>()};
}

template<typename Precision>
struct ProductAngularAccumulator {
    Precision value_0=Precision(0),value_1=Precision(0);
    Precision value_2=Precision(0),value_3=Precision(0);
    Precision value_4=Precision(0),value_5=Precision(0);
    Precision value_6=Precision(0),value_7=Precision(0);
    Precision value_8=Precision(0),value_9=Precision(0);
    Precision value_10=Precision(0),value_11=Precision(0);
    Precision value_12=Precision(0),value_13=Precision(0);
    Precision value_14=Precision(0),value_15=Precision(0);

    KOKKOS_INLINE_FUNCTION
    void add(const int index,const Precision value)
    {
        switch(index) {
        case 0:value_0+=value;break;
        case 1:value_1+=value;break;
        case 2:value_2+=value;break;
        case 3:value_3+=value;break;
        case 4:value_4+=value;break;
        case 5:value_5+=value;break;
        case 6:value_6+=value;break;
        case 7:value_7+=value;break;
        case 8:value_8+=value;break;
        case 9:value_9+=value;break;
        case 10:value_10+=value;break;
        case 11:value_11+=value;break;
        case 12:value_12+=value;break;
        case 13:value_13+=value;break;
        case 14:value_14+=value;break;
        case 15:value_15+=value;break;
        }
    }
};

}

template<typename Precision>
E3ProductBasisKokkosT<Precision>::E3ProductBasisKokkosT(const nlohmann::json& data)
    : input(data.at("symmetric_contractions").at("irreps_in").get<std::string>()),
      output(data.at("symmetric_contractions").at("irreps_out").get<std::string>()),
      linear(data.at("linear"))
{
    E3ProductBasis validated(data);
    input_dimension_=input.dimension(); output_dimension_=output.dimension(); num_features=input.blocks.front().multiplicity;
    for(const auto& block:input.blocks) { if(block.multiplicity!=num_features) throw std::invalid_argument("Kokkos product requires common multiplicity."); angular_dimension+=2*block.l+1; }
    use_sc=data.at("use_sc").get<bool>(); agnostic=data.value("use_agnostic_product",false);
    if(validated.uses_compiled_plan()) {
        const auto& plan=validated.compiled_plan();
        if(plan.empty())
            throw std::invalid_argument("Kokkos compiled product plan has an invalid angular layout.");
        if(plan.size()!=output.blocks.size())
            throw std::invalid_argument(
                "Kokkos compiled product plan does not cover every output block.");
        int num_elements=plan.front().num_elements;
        for(const auto& block:plan) {
            if(block.num_elements!=num_elements)
                throw std::invalid_argument("Kokkos compiled product element dimensions are inconsistent.");
            compiled_terms+=static_cast<int>(block.terms.size());
            compiled_component_count+=block.width;
        }
        if(angular_dimension>16)
            throw std::invalid_argument(
                "Kokkos compiled product supports at most 16 angular components.");
        compiled_term_data=Kokkos::View<int**,Kokkos::LayoutRight>(
            "compiled product terms",compiled_terms,4);
        compiled_term_native_layout=Kokkos::View<int**,Kokkos::LayoutRight>(
            "compiled product term native layout",compiled_terms,6);
        compiled_component_data=Kokkos::View<int**,Kokkos::LayoutRight>(
            "compiled product components",compiled_component_count,6);
        compiled_coefficients=Kokkos::View<Precision***,Kokkos::LayoutRight>(
            "compiled product coefficients",compiled_terms,num_elements,num_features);
        compiled_angular_native_layout=Kokkos::View<int**,Kokkos::LayoutRight>(
            "compiled product native angular layout",angular_dimension,2);
        compiled_angular_packed_layout=Kokkos::View<int**,Kokkos::LayoutRight>(
            "compiled product packed angular layout",angular_dimension,3);
        auto host_terms=Kokkos::create_mirror_view(compiled_term_data);
        auto host_term_native_layout=
            Kokkos::create_mirror_view(compiled_term_native_layout);
        auto host_components=Kokkos::create_mirror_view(compiled_component_data);
        auto host_coefficients=Kokkos::create_mirror_view(compiled_coefficients);
        auto host_angular_layout=
            Kokkos::create_mirror_view(compiled_angular_native_layout);
        auto host_packed_layout=
            Kokkos::create_mirror_view(compiled_angular_packed_layout);
        std::vector<std::pair<int,int>> native_layout(angular_dimension);
        int angular_offset=0;
        for(const auto& block:input.blocks) {
            const int width=2*block.l+1;
            for(int component=0;component<width;++component) {
                native_layout[angular_offset+component]={
                    block.offset+component,width};
                host_angular_layout(angular_offset+component,0)=
                    block.offset+component;
                host_angular_layout(angular_offset+component,1)=width;
                host_packed_layout(angular_offset+component,0)=block.offset;
                host_packed_layout(angular_offset+component,1)=component;
                host_packed_layout(angular_offset+component,2)=width;
            }
            angular_offset+=width;
        }
        int term_offset=0;
        int component_offset=0;
        for(int block_index=0;block_index<static_cast<int>(plan.size());++block_index) {
            const auto& source=plan[block_index];
            const auto& output_block=output.blocks.at(block_index);
            if(source.width!=2*output_block.l+1
                ||output_block.multiplicity!=num_features)
                throw std::invalid_argument(
                    "Kokkos compiled product output layout is inconsistent.");
            CompiledBlock block;
            block.angular_offset=source.angular_offset;
            block.output_offset=output_block.offset;
            block.width=source.width;
            block.num_elements=source.num_elements;
            block.term_offset=term_offset;
            for(int component=0;component<source.width;++component) {
                // The canonical plan sorts monomials by degree, allowing the
                // host reverse kernel to avoid a branch for every term.
                const int local_first=source.component_offsets[component];
                const int local_last=source.component_offsets[component+1];
                int degree_1_last=local_first;
                while(degree_1_last<local_last
                    &&source.terms[degree_1_last].degree==1)
                    ++degree_1_last;
                int degree_2_last=degree_1_last;
                while(degree_2_last<local_last
                    &&source.terms[degree_2_last].degree==2)
                    ++degree_2_last;
                if(degree_2_last<local_last
                    &&source.terms[degree_2_last].degree!=3)
                    throw std::invalid_argument(
                        "Kokkos compiled product term degree is unsupported.");
                host_components(component_offset,0)=
                    block.output_offset+component;
                host_components(component_offset,1)=source.width;
                host_components(component_offset,2)=
                    term_offset+local_first;
                host_components(component_offset,3)=
                    term_offset+local_last;
                host_components(component_offset,4)=
                    term_offset+degree_1_last;
                host_components(component_offset,5)=
                    term_offset+degree_2_last;
                ++component_offset;
            }
            for(const auto& term:source.terms) {
                host_terms(term_offset,0)=term.degree;
                for(int axis=0;axis<3;++axis) {
                    host_terms(term_offset,axis+1)=term.indices[axis];
                    const int index=axis<term.degree?term.indices[axis]:0;
                    host_term_native_layout(term_offset,2*axis)=
                        native_layout[index].first;
                    host_term_native_layout(term_offset,2*axis+1)=
                        native_layout[index].second;
                }
                for(int element=0;element<num_elements;++element)
                    for(int feature=0;feature<num_features;++feature)
                        host_coefficients(term_offset,element,feature)=
                            term.coefficients[element*num_features+feature];
                ++term_offset;
            }
            compiled_blocks.push_back(std::move(block));
        }
        if(term_offset!=compiled_terms
            ||component_offset!=compiled_component_count)
            throw std::logic_error(
                "Kokkos compiled product descriptor construction is incomplete.");
        Kokkos::deep_copy(compiled_term_data,host_terms);
        Kokkos::deep_copy(
            compiled_term_native_layout,host_term_native_layout);
        Kokkos::deep_copy(compiled_component_data,host_components);
        Kokkos::deep_copy(compiled_coefficients,host_coefficients);
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            const int candidate_terms=compiled_component_count==4
                    &&compiled_terms==symmetrix::standard_m0::total_terms
                ?symmetrix::standard_m0::total_terms
                :(compiled_component_count==1
                        &&compiled_terms==symmetrix::standard_m0::term_offsets[1]
                    ?symmetrix::standard_m0::term_offsets[1]:0);
            bool standard_plan=candidate_terms!=0
                &&angular_dimension==symmetrix::standard_m0::input_components;
            for(int component=0;standard_plan
                    &&component<compiled_component_count;++component)
                standard_plan=host_components(component,2)
                        ==symmetrix::standard_m0::term_offsets[component]
                    &&host_components(component,3)
                        ==symmetrix::standard_m0::term_offsets[component+1];
            for(int term=0;standard_plan&&term<candidate_terms;++term) {
                standard_plan=host_terms(term,0)
                    ==symmetrix::standard_m0::term_degrees[term];
                for(int axis=0;standard_plan
                        &&axis<host_terms(term,0);++axis)
                    standard_plan=host_terms(term,axis+1)
                        ==symmetrix::standard_m0::term_components[term][axis];
            }
            if(standard_plan) {
                compiled_host_standard_term_count=candidate_terms;
                compiled_host_standard_weights=
                    Kokkos::View<Precision***,Kokkos::LayoutRight>(
                        "compiled product standard host weights",
                        num_elements,compiled_terms,num_features);
                auto host_standard_weights=Kokkos::create_mirror_view(
                    compiled_host_standard_weights);
                for(int element=0;element<num_elements;++element)
                    for(int term=0;term<compiled_terms;++term)
                        for(int feature=0;feature<num_features;++feature)
                            host_standard_weights(element,term,feature)=
                                host_coefficients(term,element,feature);
                Kokkos::deep_copy(
                    compiled_host_standard_weights,host_standard_weights);
            }
        }
        Kokkos::deep_copy(
            compiled_angular_native_layout,host_angular_layout);
        Kokkos::deep_copy(
            compiled_angular_packed_layout,host_packed_layout);
        return;
    }
    for(const auto& value:data.at("symmetric_contractions").at("contractions")) {
        Contraction contraction; contraction.correlation=value.at("correlation").get<int>();
        for(const auto& tensor:value.at("u_tensors")) contraction.u.push_back(load_tensor<Precision>(tensor));
        for(auto it=value.at("weights").rbegin();it!=value.at("weights").rend();++it) contraction.weights.push_back(load_tensor<Precision>(*it));
        contraction.weights.push_back(load_tensor<Precision>(value.at("weights_max"))); contractions.push_back(std::move(contraction));
    }
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::prepare(int batch)
{
    if(!uses_compiled_plan()) {
        if(feature_major_storage.extent(0)<batch)
            Kokkos::realloc(
                feature_major_storage,batch,num_features,angular_dimension);
        if(feature_major_adjoint_storage.extent(0)<batch)
            Kokkos::realloc(
                feature_major_adjoint_storage,batch,num_features,angular_dimension);
        feature_major=Kokkos::subview(
            feature_major_storage,std::make_pair(0,batch),Kokkos::ALL,Kokkos::ALL);
        feature_major_adjoint=Kokkos::subview(
            feature_major_adjoint_storage,
            std::make_pair(0,batch),Kokkos::ALL,Kokkos::ALL);
    }
    if(contracted_storage.extent(0)<batch)
        Kokkos::realloc(contracted_storage,batch,output_dimension_);
    if(contracted_adjoint_storage.extent(0)<batch)
        Kokkos::realloc(contracted_adjoint_storage,batch,output_dimension_);
    contracted=Kokkos::subview(
        contracted_storage,std::make_pair(0,batch),Kokkos::ALL);
    contracted_adjoint=Kokkos::subview(
        contracted_adjoint_storage,std::make_pair(0,batch),Kokkos::ALL);
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::prepare_compiled_angular_input(
    const int batch)
{
    if(compiled_host_angular_major_input.extent(0)
        <static_cast<std::size_t>(batch))
        Kokkos::realloc(compiled_host_angular_major_input,batch,
            angular_dimension,num_features);
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::prepare_compiled_angular_adjoint(
    const int batch)
{
    if(compiled_host_angular_major_adjoint.extent(0)
        <static_cast<std::size_t>(batch))
        Kokkos::realloc(compiled_host_angular_major_adjoint,batch,
            angular_dimension,num_features);
}

template<typename Precision>
std::size_t E3ProductBasisKokkosT<Precision>::feature_major_workspace_bytes() const
{
    return sizeof(Precision)*(
        feature_major_storage.size()+feature_major_adjoint_storage.size()
        +compiled_host_angular_major_input.size()
        +compiled_host_angular_major_adjoint.size()
        +compiled_host_block_major_input.size()
        +compiled_host_packed_input_adjoint_storage.size());
}

template<typename Precision>
std::size_t E3ProductBasisKokkosT<Precision>::workspace_bytes() const
{
    return feature_major_workspace_bytes()+sizeof(Precision)*(
        contracted_storage.size()+contracted_adjoint_storage.size());
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::to_feature_major(Kokkos::View<const Precision**,Kokkos::LayoutRight> source)
{
    int angular_offset=0; auto destination=feature_major;
    for(const auto block:input.blocks) { const int width=2*block.l+1; const int offset=angular_offset;
        Kokkos::parallel_for("product layout",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),num_features,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { destination(sample,feature,offset+component)=source(sample,block.offset+feature*width+component); }); angular_offset+=width;
    }
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> source,Kokkos::View<const Precision**,Kokkos::LayoutRight> skip,Kokkos::View<const int*> elements,Kokkos::View<Precision**,Kokkos::LayoutRight> result,const bool validate_elements,const InputLayout input_layout,const int skip_active_dimension)
{
    prepare(source.extent(0));
    const bool reuse_compiled_block_major_input=
        input_layout==InputLayout::block_major;
    if(reuse_compiled_block_major_input&&!uses_compiled_plan())
        throw std::invalid_argument(
            "Compiled block-major product input requires a compiled plan.");
    if(reuse_compiled_block_major_input&&!uses_standard_host_plan())
        throw std::invalid_argument(
            "Compiled block-major input requires the standard host plan.");
    auto target=contracted;
    const int feature_count=num_features;
    if(uses_compiled_plan()) {
        if(validate_elements) {
            int invalid_elements=0;
            const int element_count=compiled_blocks.front().num_elements;
            Kokkos::parallel_reduce(
                "validate compiled product elements",source.extent(0),
                KOKKOS_LAMBDA(int sample,int& invalid) {
                    if(elements(sample)<0||elements(sample)>=element_count)
                        ++invalid;
                },invalid_elements);
            if(invalid_elements)
                throw std::out_of_range(
                    "MACE_Nonlinear product element index is out of range.");
        }
        auto terms=compiled_term_data;
        auto term_native_layout=compiled_term_native_layout;
        auto components=compiled_component_data;
        auto coefficients=compiled_coefficients;
        auto native_source=source;
        const int angular_count=angular_dimension;
        const int component_count=compiled_component_count;
        const int element_count=compiled_blocks.front().num_elements;
        const std::size_t sample_count=source.extent(0);
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            if(!reuse_compiled_block_major_input)
                prepare_compiled_angular_input(source.extent(0));
            auto angular_layout=compiled_angular_native_layout;
            const Precision* const native_source_data=native_source.data();
            Precision* const angular_source_data=
                compiled_host_angular_major_input.data();
            const Precision* const block_source_data=
                compiled_host_block_major_input.data();
            Precision* const target_data=target.data();
            const int* const element_data=elements.data();
            const int* const angular_layout_data=angular_layout.data();
            const int* const packed_layout_data=
                compiled_angular_packed_layout.data();
            const int* const component_data=components.data();
            const int* const term_data=terms.data();
            const Precision* const coefficient_data=coefficients.data();
            const std::size_t native_source_stride=native_source.extent(1);
            const std::size_t source_stride=
                static_cast<std::size_t>(angular_count)*feature_count;
            const std::size_t target_stride=target.extent(1);
            const std::size_t coefficient_term_stride=
                static_cast<std::size_t>(element_count)*feature_count;
            const int coefficient_element_stride=feature_count;
            const int feature_tiles=
                (feature_count+product_host_forward_feature_tile-1)
                    /product_host_forward_feature_tile;
            if(!reuse_compiled_block_major_input)
                Kokkos::parallel_for(
                    "compiled product angular input forward",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<std::size_t>>(
                        0,sample_count*angular_count),
                    KOKKOS_LAMBDA(const std::size_t flat) {
                        const int index=flat%angular_count;
                        const std::size_t sample=flat/angular_count;
                        const Precision* const source=native_source_data
                            +sample*native_source_stride
                            +angular_layout_data[2*index];
                        Precision* const destination=angular_source_data
                            +(sample*angular_count+index)*feature_count;
                        const int stride=angular_layout_data[2*index+1];
                        for(int feature=0;feature<feature_count;++feature)
                            destination[feature]=source[feature*stride];
                    });
            if(compiled_host_standard_term_count!=0) {
                auto standard_weights=compiled_host_standard_weights;
                const int standard_terms=compiled_host_standard_term_count;
                Kokkos::parallel_for(
                    "compiled product standard host forward",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<std::size_t>>(0,sample_count),
                    KOKKOS_LAMBDA(const std::size_t sample) {
                    const int element=element_data[sample];
                    for(int feature_first=0;feature_first<feature_count;
                        feature_first+=product_host_standard_feature_tile) {
                        const int active_features=feature_first
                                +product_host_standard_feature_tile<feature_count
                            ?product_host_standard_feature_tile
                            :feature_count-feature_first;
                        Precision input_values[
                            symmetrix::standard_m0::input_components]
                            [product_host_standard_feature_tile];
                        Precision output_values[
                            symmetrix::standard_m0::output_components]
                            [product_host_standard_feature_tile]={};
                        if(reuse_compiled_block_major_input) {
                            for(int component=0;component
                                    <symmetrix::standard_m0::input_components;
                                ++component) {
                                const int block_offset=
                                    packed_layout_data[3*component];
                                const int block_component=
                                    packed_layout_data[3*component+1];
                                const int block_width=
                                    packed_layout_data[3*component+2];
                                const Precision* const component_input=
                                    block_source_data
                                    +sample_count*block_offset
                                    +sample*block_width*feature_count
                                    +block_component*feature_count+feature_first;
#if defined(_OPENMP)
#pragma omp simd
#endif
                                for(int lane=0;lane<active_features;++lane)
                                    input_values[component][lane]=
                                        component_input[lane];
                            }
                        } else
                            for(int component=0;component
                                    <symmetrix::standard_m0::input_components;
                                ++component) {
                                const Precision* const component_input=
                                    angular_source_data+sample*source_stride
                                    +component*feature_count+feature_first;
#if defined(_OPENMP)
#pragma omp simd
#endif
                                for(int lane=0;lane<active_features;++lane)
                                    input_values[component][lane]=
                                        component_input[lane];
                            }
                        if(element>=0&&element<element_count) {
                            if(standard_terms
                                ==symmetrix::standard_m0::total_terms)
                                symmetrix::standard_m0::accumulate_forward_terms<
                                    product_host_standard_feature_tile>(
                                    element,feature_first,active_features,
                                    input_values,standard_weights,output_values,
                                    std::make_index_sequence<
                                        symmetrix::standard_m0::total_terms>{});
                            else
                                symmetrix::standard_m0::accumulate_forward_terms<
                                    product_host_standard_feature_tile>(
                                    element,feature_first,active_features,
                                    input_values,standard_weights,output_values,
                                    std::make_index_sequence<
                                        symmetrix::standard_m0::term_offsets[1]>{});
                        }
                        for(int output_component=0;
                            output_component<component_count;++output_component) {
                            const int* const component=
                                component_data+6*output_component;
                            for(int lane=0;lane<active_features;++lane)
                                target_data[
                                    sample*target_stride+component[0]
                                    +(feature_first+lane)*component[1]]=
                                    output_values[output_component][lane];
                        }
                    }
                });
        } else {
            if(reuse_compiled_block_major_input)
                throw std::invalid_argument(
                    "Compiled block-major product input is available only "
                    "on host execution.");
            Kokkos::parallel_for(
            "compiled product contraction",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    0,sample_count*feature_tiles),
                KOKKOS_LAMBDA(const std::size_t flat) {
                const int feature_tile=flat%feature_tiles;
                const std::size_t sample=flat/feature_tiles;
                const int feature_first=
                    feature_tile*product_host_forward_feature_tile;
                const int element=element_data[sample];
                Precision input_values[16][product_host_forward_feature_tile];
                for(int index=0;index<angular_count;++index)
                    for(int lane=0;lane<product_host_forward_feature_tile;++lane) {
                        const int feature=feature_first+lane;
                        if(feature<feature_count)
                            input_values[index][lane]=angular_source_data[
                                sample*source_stride+index*feature_count+feature];
                    }
                if(element<0||element>=element_count) {
                    for(int output_component=0;
                        output_component<component_count;++output_component)
                        for(int lane=0;lane<product_host_forward_feature_tile;
                            ++lane) {
                            const int feature=feature_first+lane;
                            if(feature<feature_count)
                                target_data[
                                    sample*target_stride
                                    +component_data[6*output_component]
                                    +feature*component_data[
                                        6*output_component+1]]=
                                    Precision(0);
                        }
                    return;
                }
                for(int output_component=0;
                    output_component<component_count;++output_component) {
                    const int* const component=
                        component_data+6*output_component;
                    Precision sum[product_host_forward_feature_tile]={};
                    const int first=component[2];
                    const int last=component[3];
                    const int degree_1_last=component[4];
                    const int degree_2_last=component[5];
                    for(int term=first;term<degree_1_last;++term) {
                        const int i0=term_data[4*term+1];
                        const Precision* const term_coefficients=
                            coefficient_data
                            +static_cast<std::size_t>(term)
                                *coefficient_term_stride
                            +element*coefficient_element_stride;
                        for(int lane=0;lane<product_host_forward_feature_tile;
                            ++lane) {
                            const int feature=feature_first+lane;
                            if(feature>=feature_count)
                                continue;
                            sum[lane]+=term_coefficients[feature]
                                *input_values[i0][lane];
                        }
                    }
                    for(int term=degree_1_last;term<degree_2_last;++term) {
                        const int i0=term_data[4*term+1];
                        const int i1=term_data[4*term+2];
                        const Precision* const term_coefficients=
                            coefficient_data
                            +static_cast<std::size_t>(term)
                                *coefficient_term_stride
                            +element*coefficient_element_stride;
                        for(int lane=0;lane<product_host_forward_feature_tile;
                            ++lane) {
                            const int feature=feature_first+lane;
                            if(feature>=feature_count)
                                continue;
                            Precision monomial=input_values[i0][lane];
                            monomial*=input_values[i1][lane];
                            sum[lane]+=term_coefficients[feature]
                                *monomial;
                        }
                    }
                    for(int term=degree_2_last;term<last;++term) {
                        const int i0=term_data[4*term+1];
                        const int i1=term_data[4*term+2];
                        const int i2=term_data[4*term+3];
                        const Precision* const term_coefficients=
                            coefficient_data
                            +static_cast<std::size_t>(term)
                                *coefficient_term_stride
                            +element*coefficient_element_stride;
                        for(int lane=0;lane<product_host_forward_feature_tile;
                            ++lane) {
                            const int feature=feature_first+lane;
                            if(feature>=feature_count)
                                continue;
                            Precision monomial=input_values[i0][lane];
                            monomial*=input_values[i1][lane];
                            monomial*=input_values[i2][lane];
                            sum[lane]+=term_coefficients[feature]
                                *monomial;
                        }
                    }
                    for(int lane=0;lane<product_host_forward_feature_tile;++lane) {
                        const int feature=feature_first+lane;
                        if(feature<feature_count)
                            target_data[
                                sample*target_stride+component[0]
                                    +feature*component[1]]=
                                sum[lane];
                    }
                }
            });
        }
        } else {
            Kokkos::parallel_for(
                "compiled product contraction",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    0,sample_count*component_count*feature_count),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int feature=flat%feature_count;
                    const std::size_t remainder=flat/feature_count;
                    const int output_component=remainder%component_count;
                    const std::size_t sample=remainder/component_count;
                    const int element=elements(sample);
                    const int output_column=components(output_component,0)
                        +feature*components(output_component,1);
                    if(element<0||element>=element_count) {
                        target(sample,output_column)=Precision(0);
                        return;
                    }
                    Precision sum=Precision(0);
                    const int first=components(output_component,2);
                    const int last=components(output_component,3);
                    for(int term=first;term<last;++term) {
                        const int degree=terms(term,0);
                        Precision monomial=native_source(
                            sample,term_native_layout(term,0)
                                +feature*term_native_layout(term,1));
                        if(degree>1)
                            monomial*=native_source(
                                sample,term_native_layout(term,2)
                                    +feature*term_native_layout(term,3));
                        if(degree>2)
                            monomial*=native_source(
                                sample,term_native_layout(term,4)
                                    +feature*term_native_layout(term,5));
                        sum+=coefficients(term,element,feature)*monomial;
                    }
                    target(sample,output_column)=sum;
                });
        }
        linear.evaluate(contracted,result);
        if(use_sc) {
            const int active=skip_active_dimension<0
                ?static_cast<int>(result.extent(1)):skip_active_dimension;
            Kokkos::parallel_for("product skip",
                static_cast<std::size_t>(result.extent(0))*active,
                KOKKOS_LAMBDA(std::size_t flat) {
                    const std::size_t sample=flat/active;
                    const int column=flat%active;
                    result(sample,column)+=skip(sample,column);
                });
        }
        return;
    }
    ordered_kokkos_deep_copy(contracted,Precision(0));
    to_feature_major(source);
    auto features=feature_major;
    for(int block_index=0;block_index<static_cast<int>(output.blocks.size());++block_index) { const auto block=output.blocks[block_index]; const int width=2*block.l+1; const auto& contraction=contractions[block_index];
        for(int degree=1;degree<=contraction.correlation;++degree) { const auto& u=contraction.u[degree-1]; const auto& weights=contraction.weights[degree-1]; const auto u_values=u.values; const auto weight_values=weights.values; const int output_axes=block.l==0?0:1; const int parameters=u.shape.back(); int tuples=1; for(int axis=0;axis<degree;++axis) tuples*=u.shape[output_axes+axis]; const int tuple_dimension=angular_dimension;
            Kokkos::parallel_for("product contraction",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),feature_count,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { const int element=elements(sample); Precision sum=Precision(0); const int component_offset=(output_axes?component*tuples*parameters:0); const int weight_offset=element*parameters*feature_count;
                for(int tuple=0;tuple<tuples;++tuple) { int remainder=tuple; Precision monomial=Precision(1); for(int axis=degree-1;axis>=0;--axis) { const int index=remainder%tuple_dimension; remainder/=tuple_dimension; monomial*=features(sample,feature,index); } Precision coefficient=Precision(0); for(int parameter=0;parameter<parameters;++parameter) coefficient+=u_values(component_offset+tuple*parameters+parameter)*weight_values(weight_offset+parameter*feature_count+feature); sum+=coefficient*monomial; }
                target(sample,block.offset+feature*width+component)+=sum; });
        }
    }
    linear.evaluate(contracted,result); if(use_sc) {
        const int active=skip_active_dimension<0
            ?static_cast<int>(result.extent(1)):skip_active_dimension;
        Kokkos::parallel_for("product skip",
            static_cast<std::size_t>(result.extent(0))*active,
            KOKKOS_LAMBDA(std::size_t flat) {
                const std::size_t sample=flat/active;
                const int column=flat%active;
                result(sample,column)+=skip(sample,column);
            });
    }
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> source,Kokkos::View<const int*> elements,Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> skip_adjoint,const bool validate_elements,const bool emit_native_input_adjoint,const InputLayout input_layout,const int skip_active_dimension)
{
    prepare(source.extent(0));
    const bool reuse_compiled_block_major_input=
        input_layout==InputLayout::block_major;
    if(reuse_compiled_block_major_input&&!uses_compiled_plan())
        throw std::invalid_argument(
            "Compiled block-major product input requires a compiled plan.");
    if(reuse_compiled_block_major_input&&!uses_standard_host_plan())
        throw std::invalid_argument(
            "Compiled block-major input requires the standard host plan.");
    linear.reverse(output_adjoint,contracted_adjoint);
    auto target_adj=contracted_adjoint;
    const int feature_count=num_features;
    if(uses_compiled_plan()) {
        if(validate_elements) {
            int invalid_elements=0;
            const int element_count=compiled_blocks.front().num_elements;
            Kokkos::parallel_reduce(
                "validate compiled product reverse elements",source.extent(0),
                KOKKOS_LAMBDA(int sample,int& invalid) {
                    if(elements(sample)<0||elements(sample)>=element_count)
                        ++invalid;
                },invalid_elements);
            if(invalid_elements)
                throw std::out_of_range(
                    "MACE_Nonlinear product element index is out of range.");
        }
        auto terms=compiled_term_data;
        auto term_native_layout=compiled_term_native_layout;
        auto components=compiled_component_data;
        auto coefficients=compiled_coefficients;
        auto angular_layout=compiled_angular_native_layout;
        auto packed_layout=compiled_angular_packed_layout;
        auto native_source=source;
        auto native_input_adjoint=input_adjoint;
        const int angular_count=angular_dimension;
        const int component_count=compiled_component_count;
        const int element_count=compiled_blocks.front().num_elements;
        const std::size_t sample_count=source.extent(0);
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            if(!reuse_compiled_block_major_input)
                prepare_compiled_angular_input(source.extent(0));
            if(emit_native_input_adjoint)
                prepare_compiled_angular_adjoint(source.extent(0));
            if(!emit_native_input_adjoint) {
                if(compiled_host_standard_term_count==0)
                    throw std::invalid_argument(
                        "Packed product reverse requires the standard host plan.");
                if(compiled_host_packed_input_adjoint_storage.extent(0)
                    <sample_count)
                    Kokkos::realloc(
                        compiled_host_packed_input_adjoint_storage,sample_count,
                        input_dimension_);
                compiled_host_packed_input_adjoint=Kokkos::subview(
                    compiled_host_packed_input_adjoint_storage,
                    std::make_pair(std::size_t(0),sample_count),Kokkos::ALL);
            }
            const Precision* const native_source_data=native_source.data();
            Precision* const angular_source_data=
                compiled_host_angular_major_input.data();
            const Precision* const block_source_data=
                compiled_host_block_major_input.data();
            const Precision* const target_adjoint_data=target_adj.data();
            Precision* const input_adjoint_data=native_input_adjoint.data();
            const int* const element_data=elements.data();
            const int* const angular_layout_data=angular_layout.data();
            const int* const packed_layout_data=packed_layout.data();
            const int* const component_data=components.data();
            const int* const term_data=terms.data();
            const Precision* const coefficient_data=coefficients.data();
            Precision* const angular_major_adjoint_data=
                compiled_host_angular_major_adjoint.data();
            Precision* const packed_input_adjoint_data=
                compiled_host_packed_input_adjoint.data();
            const std::size_t native_source_stride=native_source.extent(1);
            const std::size_t source_stride=
                static_cast<std::size_t>(angular_count)*feature_count;
            const std::size_t target_adjoint_stride=target_adj.extent(1);
            const std::size_t input_adjoint_stride=
                native_input_adjoint.extent(1);
            const std::size_t angular_major_sample_stride=
                static_cast<std::size_t>(angular_count)*feature_count;
            const std::size_t coefficient_term_stride=
                static_cast<std::size_t>(element_count)*feature_count;
            if(!reuse_compiled_block_major_input)
                Kokkos::parallel_for(
                    "compiled product angular input reverse",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<std::size_t>>(
                        0,sample_count*angular_count),
                    KOKKOS_LAMBDA(const std::size_t flat) {
                        const int index=flat%angular_count;
                        const std::size_t sample=flat/angular_count;
                        const Precision* const source=native_source_data
                            +sample*native_source_stride
                            +angular_layout_data[2*index];
                        Precision* const destination=angular_source_data
                            +(sample*angular_count+index)*feature_count;
                        const int stride=angular_layout_data[2*index+1];
                        for(int feature=0;feature<feature_count;++feature)
                            destination[feature]=source[feature*stride];
                    });
            if(compiled_host_standard_term_count!=0) {
                auto standard_weights=compiled_host_standard_weights;
                const int standard_terms=compiled_host_standard_term_count;
                Kokkos::parallel_for(
                    "compiled product standard host reverse",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<std::size_t>>(0,sample_count),
                    KOKKOS_LAMBDA(const std::size_t sample) {
                    const int element=element_data[sample];
                    Precision* const sample_adjoint=
                        angular_major_adjoint_data
                        +sample*angular_major_sample_stride;
                    const Precision* const sample_input=
                        angular_source_data+sample*source_stride;
                    for(int feature_first=0;feature_first<feature_count;
                        feature_first+=product_host_standard_feature_tile) {
                        const int active_features=feature_first
                                +product_host_standard_feature_tile<feature_count
                            ?product_host_standard_feature_tile
                            :feature_count-feature_first;
                        Precision input_values[
                            symmetrix::standard_m0::input_components]
                            [product_host_standard_feature_tile];
                        Precision output_adjoints[
                            symmetrix::standard_m0::output_components]
                            [product_host_standard_feature_tile]={};
                        Precision input_adjoints[
                            symmetrix::standard_m0::input_components]
                            [product_host_standard_feature_tile]={};
                        if(reuse_compiled_block_major_input) {
                            for(int component=0;component
                                    <symmetrix::standard_m0::input_components;
                                ++component) {
                                const int block_offset=
                                    packed_layout_data[3*component];
                                const int block_component=
                                    packed_layout_data[3*component+1];
                                const int block_width=
                                    packed_layout_data[3*component+2];
                                const Precision* const component_input=
                                    block_source_data
                                    +sample_count*block_offset
                                    +sample*block_width*feature_count
                                    +block_component*feature_count+feature_first;
#if defined(_OPENMP)
#pragma omp simd
#endif
                                for(int lane=0;lane<active_features;++lane)
                                    input_values[component][lane]=
                                        component_input[lane];
                            }
                        } else
                            for(int component=0;component
                                    <symmetrix::standard_m0::input_components;
                                ++component) {
                                const Precision* const component_input=
                                    sample_input+component*feature_count
                                    +feature_first;
#if defined(_OPENMP)
#pragma omp simd
#endif
                                for(int lane=0;lane<active_features;++lane)
                                    input_values[component][lane]=
                                        component_input[lane];
                            }
                        for(int output_component=0;
                            output_component<component_count;++output_component) {
                            const int* const component=
                                component_data+6*output_component;
                            const Precision* const output_values=
                                target_adjoint_data
                                +sample*target_adjoint_stride+component[0];
                            for(int lane=0;lane<active_features;++lane)
                                output_adjoints[output_component][lane]=
                                    output_values[
                                        (feature_first+lane)*component[1]];
                        }
                        if(element>=0&&element<element_count) {
                            if(standard_terms
                                ==symmetrix::standard_m0::total_terms)
                                symmetrix::standard_m0::accumulate_reverse_terms<
                                    product_host_standard_feature_tile>(
                                    element,feature_first,active_features,
                                    input_values,output_adjoints,standard_weights,
                                    input_adjoints,std::make_index_sequence<
                                        symmetrix::standard_m0::total_terms>{});
                            else
                                symmetrix::standard_m0::accumulate_reverse_terms<
                                    product_host_standard_feature_tile>(
                                    element,feature_first,active_features,
                                    input_values,output_adjoints,standard_weights,
                                    input_adjoints,std::make_index_sequence<
                                        symmetrix::standard_m0::term_offsets[1]>{});
                        }
                        for(int component=0;
                            component<symmetrix::standard_m0::input_components;
                            ++component) {
                            if(emit_native_input_adjoint) {
                                for(int lane=0;lane<active_features;++lane)
                                    sample_adjoint[
                                        component*feature_count+feature_first
                                            +lane]=input_adjoints[component][lane];
                            } else {
                                const int block_offset=
                                    packed_layout_data[3*component];
                                const int block_component=
                                    packed_layout_data[3*component+1];
                                const int block_width=
                                    packed_layout_data[3*component+2];
                                for(int lane=0;lane<active_features;++lane) {
                                    const int feature=feature_first+lane;
                                    packed_input_adjoint_data[
                                        sample_count*block_offset
                                        +static_cast<std::size_t>(feature)
                                            *sample_count*block_width
                                        +sample*block_width+block_component]=
                                        input_adjoints[component][lane];
                                }
                            }
                        }
                    }
                });
            } else Kokkos::parallel_for(
                "compiled product reverse",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0,sample_count),
                KOKKOS_LAMBDA(const std::size_t sample) {
                    Precision* const sample_adjoint=
                        angular_major_adjoint_data
                        +sample*angular_major_sample_stride;
                    const Precision* const sample_input=
                        angular_source_data+sample*source_stride;
                    for(std::size_t flat=0;flat<angular_major_sample_stride;++flat)
                        sample_adjoint[flat]=Precision(0);
                    const int element=element_data[sample];
                    if(element>=0&&element<element_count) {
                        for(int output_component=0;
                            output_component<component_count;++output_component) {
                            const int* const component=
                                component_data+6*output_component;
                            const int first=component[2];
                            const int last=component[3];
                            const int degree_1_last=component[4];
                            const int degree_2_last=component[5];
                            const Precision* const output_values=
                                target_adjoint_data
                                +sample*target_adjoint_stride+component[0];
                            const int output_width=component[1];
                            for(int term=first;term<degree_1_last;++term) {
                                const int i0=term_data[4*term+1];
                                const Precision* const term_coefficients=
                                    coefficient_data
                                    +static_cast<std::size_t>(term)
                                        *coefficient_term_stride
                                    +element*feature_count;
                                Precision* const adj0=
                                    sample_adjoint+i0*feature_count;
                                for(int feature=0;feature<feature_count;++feature)
                                    adj0[feature]+=term_coefficients[feature]
                                        *output_values[feature*output_width];
                            }
                            for(int term=degree_1_last;term<degree_2_last;++term) {
                                const int i0=term_data[4*term+1];
                                const int i1=term_data[4*term+2];
                                const Precision* const term_coefficients=
                                    coefficient_data
                                    +static_cast<std::size_t>(term)
                                        *coefficient_term_stride
                                    +element*feature_count;
                                const Precision* const x0=
                                    sample_input+i0*feature_count;
                                const Precision* const x1=
                                    sample_input+i1*feature_count;
                                Precision* const adj0=
                                    sample_adjoint+i0*feature_count;
                                Precision* const adj1=
                                    sample_adjoint+i1*feature_count;
                                for(int feature=0;feature<feature_count;++feature) {
                                    const Precision common=term_coefficients[feature]
                                        *output_values[feature*output_width];
                                    adj0[feature]+=common*x1[feature];
                                    adj1[feature]+=common*x0[feature];
                                }
                            }
                            for(int term=degree_2_last;term<last;++term) {
                                const int i0=term_data[4*term+1];
                                const int i1=term_data[4*term+2];
                                const int i2=term_data[4*term+3];
                                const Precision* const term_coefficients=
                                    coefficient_data
                                    +static_cast<std::size_t>(term)
                                        *coefficient_term_stride
                                    +element*feature_count;
                                const Precision* const x0=
                                    sample_input+i0*feature_count;
                                const Precision* const x1=
                                    sample_input+i1*feature_count;
                                const Precision* const x2=
                                    sample_input+i2*feature_count;
                                Precision* const adj0=
                                    sample_adjoint+i0*feature_count;
                                Precision* const adj1=
                                    sample_adjoint+i1*feature_count;
                                Precision* const adj2=
                                    sample_adjoint+i2*feature_count;
                                for(int feature=0;feature<feature_count;++feature) {
                                    const Precision common=term_coefficients[feature]
                                        *output_values[feature*output_width];
                                    const Precision value0=x0[feature];
                                    const Precision value1=x1[feature];
                                    const Precision value2=x2[feature];
                                    adj0[feature]+=common*value1*value2;
                                    adj1[feature]+=common*value0*value2;
                                    adj2[feature]+=common*value0*value1;
                                }
                            }
                        }
                    }
                });
            if(emit_native_input_adjoint)
                Kokkos::parallel_for(
                    "compiled product reverse native output",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<std::size_t>>(
                        0,sample_count*angular_count*feature_count),
                    KOKKOS_LAMBDA(const std::size_t flat) {
                        const int feature=flat%feature_count;
                        const std::size_t remainder=flat/feature_count;
                        const int index=remainder%angular_count;
                        const std::size_t sample=remainder/angular_count;
                        input_adjoint_data[
                            sample*input_adjoint_stride
                            +angular_layout_data[2*index]
                            +feature*angular_layout_data[2*index+1]]=
                            angular_major_adjoint_data[flat];
                    });
        } else {
            if(reuse_compiled_block_major_input)
                throw std::invalid_argument(
                    "Compiled block-major product input is available only "
                    "on host execution.");
            if(!emit_native_input_adjoint)
                throw std::invalid_argument(
                    "Packed product reverse is available only on host execution.");
            Kokkos::parallel_for(
                "compiled product reverse",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    0,sample_count*feature_count),
                KOKKOS_LAMBDA(const std::size_t flat) {
                const int feature=flat%feature_count;
                const std::size_t sample=flat/feature_count;
                ProductAngularAccumulator<Precision> local;
                const int element=elements(sample);
                if(element>=0&&element<element_count) {
                    for(int output_component=0;
                        output_component<component_count;++output_component) {
                        const Precision output_value=target_adj(
                            sample,components(output_component,0)
                                +feature*components(output_component,1));
                        const int first=components(output_component,2);
                        const int last=components(output_component,3);
                        for(int term=first;term<last;++term) {
                            const Precision common=
                                coefficients(term,element,feature)*output_value;
                            const int degree=terms(term,0);
                            const int i0=terms(term,1);
                            if(degree==1) {
                                local.add(i0,common);
                                continue;
                            }
                            const int i1=terms(term,2);
                            const Precision x0=native_source(
                                sample,term_native_layout(term,0)
                                    +feature*term_native_layout(term,1));
                            const Precision x1=native_source(
                                sample,term_native_layout(term,2)
                                    +feature*term_native_layout(term,3));
                            if(degree==2) {
                                local.add(i0,common*x1);
                                local.add(i1,common*x0);
                                continue;
                            }
                            const int i2=terms(term,3);
                            const Precision x2=native_source(
                                sample,term_native_layout(term,4)
                                    +feature*term_native_layout(term,5));
                            local.add(i0,common*x1*x2);
                            local.add(i1,common*x0*x2);
                            local.add(i2,common*x0*x1);
                        }
                    }
                }
                if(angular_count>0) native_input_adjoint(
                    sample,angular_layout(0,0)+feature*angular_layout(0,1))=
                    local.value_0;
                if(angular_count>1) native_input_adjoint(
                    sample,angular_layout(1,0)+feature*angular_layout(1,1))=
                    local.value_1;
                if(angular_count>2) native_input_adjoint(
                    sample,angular_layout(2,0)+feature*angular_layout(2,1))=
                    local.value_2;
                if(angular_count>3) native_input_adjoint(
                    sample,angular_layout(3,0)+feature*angular_layout(3,1))=
                    local.value_3;
                if(angular_count>4) native_input_adjoint(
                    sample,angular_layout(4,0)+feature*angular_layout(4,1))=
                    local.value_4;
                if(angular_count>5) native_input_adjoint(
                    sample,angular_layout(5,0)+feature*angular_layout(5,1))=
                    local.value_5;
                if(angular_count>6) native_input_adjoint(
                    sample,angular_layout(6,0)+feature*angular_layout(6,1))=
                    local.value_6;
                if(angular_count>7) native_input_adjoint(
                    sample,angular_layout(7,0)+feature*angular_layout(7,1))=
                    local.value_7;
                if(angular_count>8) native_input_adjoint(
                    sample,angular_layout(8,0)+feature*angular_layout(8,1))=
                    local.value_8;
                if(angular_count>9) native_input_adjoint(
                    sample,angular_layout(9,0)+feature*angular_layout(9,1))=
                    local.value_9;
                if(angular_count>10) native_input_adjoint(
                    sample,angular_layout(10,0)+feature*angular_layout(10,1))=
                    local.value_10;
                if(angular_count>11) native_input_adjoint(
                    sample,angular_layout(11,0)+feature*angular_layout(11,1))=
                    local.value_11;
                if(angular_count>12) native_input_adjoint(
                    sample,angular_layout(12,0)+feature*angular_layout(12,1))=
                    local.value_12;
                if(angular_count>13) native_input_adjoint(
                    sample,angular_layout(13,0)+feature*angular_layout(13,1))=
                    local.value_13;
                if(angular_count>14) native_input_adjoint(
                    sample,angular_layout(14,0)+feature*angular_layout(14,1))=
                    local.value_14;
                if(angular_count>15) native_input_adjoint(
                    sample,angular_layout(15,0)+feature*angular_layout(15,1))=
                    local.value_15;
                });
        }
        if(use_sc) {
            const int active=skip_active_dimension<0
                ?static_cast<int>(skip_adjoint.extent(1)):skip_active_dimension;
            Kokkos::parallel_for("product skip reverse",
                static_cast<std::size_t>(skip_adjoint.extent(0))*active,
                KOKKOS_LAMBDA(std::size_t flat) {
                    const std::size_t sample=flat/active;
                    const int column=flat%active;
                    skip_adjoint(sample,column)=output_adjoint(sample,column);
                });
        } else ordered_kokkos_deep_copy(skip_adjoint,Precision(0));
        return;
    }
    to_feature_major(source);
    ordered_kokkos_deep_copy(feature_major_adjoint,Precision(0));
    auto features=feature_major;
    auto features_adj=feature_major_adjoint;
    for(int block_index=0;block_index<static_cast<int>(output.blocks.size());++block_index) { const auto block=output.blocks[block_index]; const int width=2*block.l+1; const auto& contraction=contractions[block_index];
        for(int degree=1;degree<=contraction.correlation;++degree) { const auto& u=contraction.u[degree-1]; const auto& weights=contraction.weights[degree-1]; const auto u_values=u.values; const auto weight_values=weights.values; const int output_axes=block.l==0?0:1; const int parameters=u.shape.back(); int tuples=1; for(int axis=0;axis<degree;++axis) tuples*=u.shape[output_axes+axis]; const int tuple_dimension=angular_dimension;
            Kokkos::parallel_for("product reverse",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),feature_count,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { const int element=elements(sample); const Precision adjoint=target_adj(sample,block.offset+feature*width+component); const int component_offset=(output_axes?component*tuples*parameters:0); const int weight_offset=element*parameters*feature_count;
                for(int tuple=0;tuple<tuples;++tuple) { Precision coefficient=Precision(0); for(int parameter=0;parameter<parameters;++parameter) coefficient+=u_values(component_offset+tuple*parameters+parameter)*weight_values(weight_offset+parameter*feature_count+feature); coefficient*=adjoint; for(int differentiated=0;differentiated<degree;++differentiated) { int remainder=tuple,differentiated_index=0; Precision derivative=coefficient; for(int axis=degree-1;axis>=0;--axis) { const int index=remainder%tuple_dimension; remainder/=tuple_dimension; if(axis==differentiated)differentiated_index=index;else derivative*=features(sample,feature,index); } Kokkos::atomic_add(&features_adj(sample,feature,differentiated_index),derivative); } }
            });
        }
    }
    ordered_kokkos_deep_copy(input_adjoint,Precision(0)); int angular_offset=0;
    for(const auto block:input.blocks) { const int width=2*block.l+1; const int offset=angular_offset; Kokkos::parallel_for("product reverse layout",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),num_features,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { input_adjoint(sample,block.offset+feature*width+component)=features_adj(sample,feature,offset+component); }); angular_offset+=width; }
    if(use_sc) {
        const int active=skip_active_dimension<0
            ?static_cast<int>(skip_adjoint.extent(1)):skip_active_dimension;
        Kokkos::parallel_for("product skip reverse",
            static_cast<std::size_t>(skip_adjoint.extent(0))*active,
            KOKKOS_LAMBDA(std::size_t flat) {
                const std::size_t sample=flat/active;
                const int column=flat%active;
                skip_adjoint(sample,column)=output_adjoint(sample,column);
            });
    } else ordered_kokkos_deep_copy(skip_adjoint,Precision(0));
}

template class E3ProductBasisKokkosT<float>;
template class E3ProductBasisKokkosT<double>;
