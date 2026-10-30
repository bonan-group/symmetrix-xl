#include "e3nn_product_kokkos.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "mace_standard_m0.hpp"

namespace {
// Local copy of the shared tile width.  The family owner keeps its own
// constant so the shared translation unit exports no family helper.
constexpr int product_host_standard_feature_tile=512;
}  // namespace

// Compiled host-plan accelerator for the shared product basis.  The plan
// layout and every kernel that reads it are family-private: the shared class
// stores the handle and the neutral entry points only, so a definition that
// does not match this frozen term plan keeps the generic compiled schedule.
template<typename Precision>
struct E3ProductBasisKokkosT<Precision>::CompiledHostPlan {
    int term_count=0;
    Kokkos::View<Precision***,Kokkos::LayoutRight> weights;
};

template<typename Precision>
bool E3ProductBasisKokkosT<Precision>::uses_standard_host_plan() const
{
    return compiled_host_plan_!=nullptr;
}

// Recognizes the frozen term plan from the parsed compiled-plan mirrors and
// installs its weights.  The shared constructor calls this after the device
// copies; a plan that does not match leaves the handle empty.
template<typename Precision>
void E3ProductBasisKokkosT<Precision>::initialize_compiled_host_plan(
    const Kokkos::View<int**,Kokkos::LayoutRight,Kokkos::HostSpace>& term_data,
    const Kokkos::View<int**,Kokkos::LayoutRight,Kokkos::HostSpace>&
        component_data,
    const Kokkos::View<Precision***,Kokkos::LayoutRight,Kokkos::HostSpace>&
        coefficients)
{
    if constexpr(!std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>)
        return;
    const int num_elements=compiled_blocks.front().num_elements;
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
        standard_plan=component_data(component,2)
                ==symmetrix::standard_m0::term_offsets[component]
            &&component_data(component,3)
                ==symmetrix::standard_m0::term_offsets[component+1];
    for(int term=0;standard_plan&&term<candidate_terms;++term) {
        standard_plan=term_data(term,0)
            ==symmetrix::standard_m0::term_degrees[term];
        for(int axis=0;standard_plan
                &&axis<term_data(term,0);++axis)
            standard_plan=term_data(term,axis+1)
                ==symmetrix::standard_m0::term_components[term][axis];
    }
    if(!standard_plan) return;
    auto plan=std::make_shared<CompiledHostPlan>();
    plan->term_count=candidate_terms;
    plan->weights=Kokkos::View<Precision***,Kokkos::LayoutRight>(
        "compiled product standard host weights",
        num_elements,compiled_terms,num_features);
    auto host_standard_weights=Kokkos::create_mirror_view(
        plan->weights);
    for(int element=0;element<num_elements;++element)
        for(int term=0;term<compiled_terms;++term)
            for(int feature=0;feature<num_features;++feature)
                host_standard_weights(element,term,feature)=
                    coefficients(term,element,feature);
    Kokkos::deep_copy(plan->weights,host_standard_weights);
    compiled_host_plan_=std::move(plan);
}

// Frozen standard-plan forward.  Reads the shared compiled-plan geometry and
// the plan weights through the shared members; the term loop is the family
// module's compile-time unrolled accumulation.
template<typename Precision>
void E3ProductBasisKokkosT<Precision>::evaluate_compiled_host_plan(
    const std::size_t sample_count,Kokkos::View<const int*> elements,
    const bool reuse_compiled_block_major_input)
{
    const int feature_count=num_features;
    const int component_count=compiled_component_count;
    const int element_count=compiled_blocks.front().num_elements;
    const int* const element_data=elements.data();
    const int* const component_data=compiled_component_data.data();
    const int* const packed_layout_data=compiled_angular_packed_layout.data();
    const Precision* const block_source_data=
        compiled_host_block_major_input.data();
    const Precision* const angular_source_data=
        compiled_host_angular_major_input.data();
    const std::size_t source_stride=
        static_cast<std::size_t>(angular_dimension)*feature_count;
    Precision* const target_data=contracted.data();
    const std::size_t target_stride=contracted.extent(1);
    auto standard_weights=compiled_host_plan_->weights;
    const int standard_terms=compiled_host_plan_->term_count;
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
}

// Frozen standard-plan reverse.  The emitted adjoint is either the native
// angular-major input adjoint or the packed block-major adjoint, selected by
// the caller exactly as the shared schedule does.
template<typename Precision>
void E3ProductBasisKokkosT<Precision>::reverse_compiled_host_plan(
    const std::size_t sample_count,Kokkos::View<const int*> elements,
    const bool emit_native_input_adjoint,
    const bool reuse_compiled_block_major_input)
{
    const int feature_count=num_features;
    const int component_count=compiled_component_count;
    const int element_count=compiled_blocks.front().num_elements;
    const int* const element_data=elements.data();
    const int* const component_data=compiled_component_data.data();
    const int* const packed_layout_data=compiled_angular_packed_layout.data();
    const Precision* const block_source_data=
        compiled_host_block_major_input.data();
    const Precision* const angular_source_data=
        compiled_host_angular_major_input.data();
    const std::size_t source_stride=
        static_cast<std::size_t>(angular_dimension)*feature_count;
    const std::size_t angular_major_sample_stride=source_stride;
    Precision* const angular_major_adjoint_data=
        compiled_host_angular_major_adjoint.data();
    const Precision* const target_adjoint_data=contracted_adjoint.data();
    const std::size_t target_adjoint_stride=contracted_adjoint.extent(1);
    Precision* const packed_input_adjoint_data=
        compiled_host_packed_input_adjoint.data();
    auto standard_weights=compiled_host_plan_->weights;
    const int standard_terms=compiled_host_plan_->term_count;
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
}

template class E3ProductBasisKokkosT<float>;
template class E3ProductBasisKokkosT<double>;
