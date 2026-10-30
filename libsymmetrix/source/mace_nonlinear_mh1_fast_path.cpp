#include "mace_nonlinear_mh1_fast_path.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tools_kokkos.hpp"

namespace {
// Local mirror of the shared profiling region in e3nn_kokkos.cpp.  The family
// owner keeps its own copy so the shared translation unit exports no helper.
class ProfileRegion {
public:
    explicit ProfileRegion(const char* name) { Kokkos::Profiling::pushRegion(name); }
    explicit ProfileRegion(const std::string& name) {
        Kokkos::Profiling::pushRegion(name.c_str());
    }
    ~ProfileRegion() { Kokkos::Profiling::popRegion(); }
};

int mh1_cuda_team_size(int multiplicity)
{
    if(multiplicity<=0) return 0;
    constexpr int warp_size=32;
    constexpr int maximum_team_size=128;
    const int capped=std::min(multiplicity,maximum_team_size);
    return std::max(
        warp_size,((capped+warp_size-1)/warp_size)*warp_size);
}
}  // namespace

// Builds the direct-execution plan from the model definition.  The shared
// class calls this at the end of its constructor; a definition that carries an
// internal weight tensor is never admitted, so the generic schedule is kept.
template<typename Precision>
void E3TensorProductKokkosT<Precision>::initialize_direct_execution_plan(
    const nlohmann::json& data,const E3TensorProduct& validated)
{
    if(internal_weights.extent(0)!=0) return;
    Irreps in1(data.at("irreps_in1").get<std::string>()),
        out(data.at("irreps_out").get<std::string>());
    bool official_layout=!validated.instructions.empty();
    int multiplicity=0,max_output_width=0;
    std::vector<int> mh1_instruction_data_host;
    std::vector<Precision> mh1_path_weights_host;
    std::vector<int> mh1_component_offsets_host;
    std::vector<int> mh1_sparse_indices_host;
    std::vector<Precision> mh1_sparse_values_host;
    struct HarmonicTerm { int instruction,a,c; Precision value; };
    struct InputTerm { int instruction,b,c; Precision value; };
    std::vector<std::vector<HarmonicTerm>> mh1_harmonic_terms_host(
        input_2_dimension_);
    std::vector<std::pair<int,InputTerm>> mh1_input_terms_unsorted;
    const auto& definitions=data.at("instructions");
    int instruction_index=0;
    for (const auto& instruction:instructions) {
        const auto path_shape=definitions.at(instruction_index)
            .at("path_shape").get<std::vector<int>>();
        const int instruction_multiplicity=instruction.multiplicity_1;
        official_layout=official_layout&&!instruction.uuu&&!instruction.uvw
            &&instruction.has_weight
            &&instruction_multiplicity>0&&instruction.multiplicity_2==1
            &&instruction.output_multiplicity==instruction_multiplicity
            &&(multiplicity==0||multiplicity==instruction_multiplicity)
            &&instruction.width_1<=3
            &&path_shape==std::vector<int>({instruction_multiplicity,1});
        if(official_layout) {
            multiplicity=instruction_multiplicity;
            max_output_width=std::max(
                max_output_width,instruction.output_width);
            const auto& entries=validated.instructions.at(instruction_index)
                .nonzero_wigner;
            std::vector<int> component_offsets(instruction.output_width+1,0);
            std::vector<int> indices;
            std::vector<Precision> values;
            for(int component=0;component<instruction.output_width;++component) {
                for(const auto& entry:entries)
                    if(entry.c==component) {
                        indices.push_back(entry.a);
                        indices.push_back(entry.b);
                        indices.push_back(entry.c);
                        values.push_back(entry.value);
                    }
                component_offsets[component+1]=values.size();
            }
            const int component_base=mh1_component_offsets_host.size();
            const int sparse_base=mh1_sparse_values_host.size();
            mh1_instruction_data_host.insert(
                mh1_instruction_data_host.end(),
                {instruction.input_1_offset,instruction.input_2_offset,
                    instruction.output_offset,instruction.width_1,
                    instruction.output_width,instruction.weight_offset,
                    component_base});
            mh1_path_weights_host.push_back(instruction.path_weight);
            for(const int component_offset:component_offsets)
                mh1_component_offsets_host.push_back(
                    sparse_base+component_offset);
            for(int entry=0;entry<static_cast<int>(values.size());++entry) {
                const int a=indices[3*entry];
                const int b=indices[3*entry+1];
                const int c=indices[3*entry+2];
                mh1_sparse_indices_host.insert(
                    mh1_sparse_indices_host.end(),{a,b,c});
                mh1_sparse_values_host.push_back(values[entry]);
                mh1_harmonic_terms_host.at(instruction.input_2_offset+b)
                    .push_back({instruction_index,a,c,values[entry]});
                mh1_input_terms_unsorted.push_back({
                    instruction.input_1_offset/instruction_multiplicity+a,
                    {instruction_index,b,c,values[entry]}});
            }
        }
        ++instruction_index;
    }
    if(!official_layout) return;
    direct_execution_plan_=std::make_shared<DirectExecutionPlan>();
    direct_execution_plan_->mh1_multiplicity=multiplicity;
    direct_execution_plan_->mh1_max_output_width=max_output_width;
    const auto output_mask_host=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),output_mask);
    const auto make_mul_to_ir=[](const Irreps& irreps) {
        std::vector<int> mapping(static_cast<std::size_t>(irreps.dimension()));
        for(const auto& block:irreps.blocks) {
            const int width=2*block.l+1;
            for(int channel=0;channel<block.multiplicity;++channel)
                for(int component=0;component<width;++component) {
                    const int mul_ir=block.offset+channel*width+component;
                    const int ir_mul=block.offset
                        +component*block.multiplicity+channel;
                    mapping[static_cast<std::size_t>(mul_ir)]=ir_mul;
                }
        }
        return mapping;
    };
    const auto input_1_mul_to_ir_host=make_mul_to_ir(in1);
    const auto output_mul_to_ir_host=make_mul_to_ir(out);
    direct_execution_plan_->mh1_input_1_mul_to_ir=toKokkosView(
        "e3 mh1 input mul-to-ir",input_1_mul_to_ir_host);
    direct_execution_plan_->mh1_output_mul_to_ir=toKokkosView(
        "e3 mh1 output mul-to-ir",output_mul_to_ir_host);
    std::vector<Precision> output_mask_ir_mul_host(output_mask_host.size());
    for(std::size_t mul_ir=0;mul_ir<output_mask_host.size();++mul_ir)
        output_mask_ir_mul_host[static_cast<std::size_t>(
            output_mul_to_ir_host[mul_ir])]=output_mask_host[mul_ir];
    direct_execution_plan_->mh1_output_mask_ir_mul=toKokkosView(
        "e3 mh1 tensor mask ir-mul",output_mask_ir_mul_host);
    // The compact MH1 plan supplies its weights per edge, so a contract that
    // carries an internal weight tensor is admitted to the generic schedule
    // even when its layout is otherwise official.  NequIP-OAM does not need
    // the relaxation: its only tensor-product consumer is the node-local
    // (u,v,w) self connection, which runs the node-bilinear kernels and never
    // consults this flag.
    std::vector<int> input_component_data;
    direct_execution_plan_->mh1_direct_node_layout=true;
    for(const auto& block:in1.blocks) {
        if(block.multiplicity!=direct_execution_plan_->mh1_multiplicity) {
            direct_execution_plan_->mh1_direct_node_layout=false;
            break;
        }
        const int width=2*block.l+1;
        for(int component=0;component<width;++component) {
            input_component_data.push_back(block.offset+component);
            input_component_data.push_back(width);
        }
    }
    direct_execution_plan_->mh1_input_1_angular_dimension=input_component_data.size()/2;
    direct_execution_plan_->mh1_direct_node_layout=direct_execution_plan_->mh1_direct_node_layout
        &&direct_execution_plan_->mh1_input_1_angular_dimension>0;
    if(direct_execution_plan_->mh1_direct_node_layout)
        set_kokkos_view(
            direct_execution_plan_->mh1_input_component_data,input_component_data,
            direct_execution_plan_->mh1_input_1_angular_dimension,2);
    direct_execution_plan_->mh1_cuda_channel_team_size=mh1_cuda_team_size(direct_execution_plan_->mh1_multiplicity);
    direct_execution_plan_->mh1_cuda_harmonic_team_size=mh1_cuda_team_size(direct_execution_plan_->mh1_multiplicity);
    direct_execution_plan_->mh1_instruction_count=instructions.size();
    set_kokkos_view(
        direct_execution_plan_->mh1_instruction_data,mh1_instruction_data_host,
        direct_execution_plan_->mh1_instruction_count,7);
    direct_execution_plan_->mh1_path_weights=toKokkosView(
        "e3 mh1 path weights",mh1_path_weights_host);
    direct_execution_plan_->mh1_component_offsets=toKokkosView(
        "e3 mh1 component offsets",mh1_component_offsets_host);
    set_kokkos_view(
        direct_execution_plan_->mh1_sparse_indices,mh1_sparse_indices_host,
        mh1_sparse_values_host.size(),3);
    direct_execution_plan_->mh1_sparse_values=toKokkosView(
        "e3 mh1 sparse values",mh1_sparse_values_host);

    std::vector<int> harmonic_offsets(input_2_dimension_+1,0);
    std::vector<int> harmonic_terms;
    std::vector<Precision> harmonic_values;
    for(int component=0;component<input_2_dimension_;++component) {
        for(const auto& term:mh1_harmonic_terms_host[component]) {
            harmonic_terms.insert(
                harmonic_terms.end(),{term.instruction,term.a,term.c});
            harmonic_values.push_back(term.value);
        }
        harmonic_offsets[component+1]=harmonic_values.size();
    }
    direct_execution_plan_->mh1_harmonic_offsets=toKokkosView(
        "e3 mh1 harmonic offsets",harmonic_offsets);
    set_kokkos_view(
        direct_execution_plan_->mh1_harmonic_terms,harmonic_terms,harmonic_values.size(),3);
    direct_execution_plan_->mh1_harmonic_values=toKokkosView(
        "e3 mh1 harmonic values",harmonic_values);

    std::vector<std::vector<InputTerm>> input_terms(
        direct_execution_plan_->mh1_input_1_angular_dimension);
    if(direct_execution_plan_->mh1_direct_node_layout) {
        for(const auto& item:mh1_input_terms_unsorted) {
            if(item.first<0
                ||item.first>=direct_execution_plan_->mh1_input_1_angular_dimension) {
                direct_execution_plan_->mh1_direct_node_layout=false;
                break;
            }
            input_terms[item.first].push_back(item.second);
        }
    }
    if(direct_execution_plan_->mh1_direct_node_layout) {
        std::vector<int> input_term_offsets(
            direct_execution_plan_->mh1_input_1_angular_dimension+1,0);
        std::vector<int> input_instruction_term_offsets(
            static_cast<std::size_t>(direct_execution_plan_->mh1_input_1_angular_dimension)
                *static_cast<std::size_t>(direct_execution_plan_->mh1_instruction_count+1),
            0);
        std::vector<int> packed_input_terms;
        std::vector<Precision> input_values;
        for(int component=0;
            component<direct_execution_plan_->mh1_input_1_angular_dimension;++component) {
            for(int instruction=0;instruction<direct_execution_plan_->mh1_instruction_count;
                ++instruction) {
                input_instruction_term_offsets[
                    static_cast<std::size_t>(component)
                        *static_cast<std::size_t>(direct_execution_plan_->mh1_instruction_count+1)
                    +static_cast<std::size_t>(instruction)]
                    =input_values.size();
                for(const auto& term:input_terms[component]) {
                    if(term.instruction!=instruction) continue;
                    packed_input_terms.insert(packed_input_terms.end(),
                        {term.instruction,term.b,term.c});
                    input_values.push_back(term.value);
                }
            }
            input_instruction_term_offsets[
                static_cast<std::size_t>(component)
                    *static_cast<std::size_t>(direct_execution_plan_->mh1_instruction_count+1)
                +static_cast<std::size_t>(direct_execution_plan_->mh1_instruction_count)]
                =input_values.size();
            input_term_offsets[component+1]=input_values.size();
        }
        direct_execution_plan_->mh1_input_term_offsets=toKokkosView(
            "e3 mh1 input term offsets",input_term_offsets);
        set_kokkos_view(
            direct_execution_plan_->mh1_input_instruction_term_offsets,
            input_instruction_term_offsets,
            direct_execution_plan_->mh1_input_1_angular_dimension,direct_execution_plan_->mh1_instruction_count+1);
        set_kokkos_view(direct_execution_plan_->mh1_input_terms,packed_input_terms,
            input_values.size(),3);
        direct_execution_plan_->mh1_input_values=toKokkosView(
            "e3 mh1 input values",input_values);
    }
}

// The predicates and accessors below report the installed plan.  They are
// declared by the shared class and defined here, in the family that owns the
// plan, so the shared header carries no plan state.
template<typename Precision>
bool E3TensorProductKokkosT<Precision>::uses_direct_execution_plan() const
{
    return direct_execution_plan_!=nullptr;
}

template<typename Precision>
std::string E3TensorProductKokkosT<Precision>::backend() const
{
    return direct_execution_plan_ ? "official_kokkos" : "generic_kokkos";
}

template<typename Precision>
std::string E3TensorProductKokkosT<Precision>::execution_backend() const
{
    if(!direct_execution_plan_) return "generic_kokkos";
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return "official_cuda_team";
#endif
    return "official_kokkos_mdrange";
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::channel_team_size() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return direct_execution_plan_
            ? direct_execution_plan_->mh1_cuda_channel_team_size : 0;
#endif
    return 0;
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::harmonic_team_size() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return direct_execution_plan_
            ? direct_execution_plan_->mh1_cuda_harmonic_team_size : 0;
#endif
    return 0;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::supports_direct_node_reverse() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return direct_execution_plan_
            &&direct_execution_plan_->mh1_direct_node_layout
            &&direct_execution_plan_->mh1_input_1_angular_dimension
                <=max_direct_node_input_components_;
#endif
    return false;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::supports_execution_uvu() const
{
    return direct_execution_plan_
        &&direct_execution_plan_->mh1_direct_node_layout;
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::execution_multiplicity() const
{
    return direct_execution_plan_
        ? direct_execution_plan_->mh1_multiplicity : 0;
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::execution_input_1_angular_dimension() const
{
    return direct_execution_plan_
        ? direct_execution_plan_->mh1_input_1_angular_dimension : 0;
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::execution_instruction_count() const
{
    return direct_execution_plan_
        ? direct_execution_plan_->mh1_instruction_count : 0;
}

template<typename Precision>
Kokkos::View<const Precision*>
E3TensorProductKokkosT<Precision>::execution_output_mask_ir_mul() const
{
    if(!direct_execution_plan_) return Kokkos::View<const Precision*>();
    return direct_execution_plan_->mh1_output_mask_ir_mul;
}

template<typename Precision>
Kokkos::View<const int*>
E3TensorProductKokkosT<Precision>::execution_input_1_mul_to_ir() const
{
    if(!direct_execution_plan_) return Kokkos::View<const int*>();
    return direct_execution_plan_->mh1_input_1_mul_to_ir;
}

template<typename Precision>
Kokkos::View<const int*>
E3TensorProductKokkosT<Precision>::execution_output_mul_to_ir() const
{
    if(!direct_execution_plan_) return Kokkos::View<const int*>();
    return direct_execution_plan_->mh1_output_mul_to_ir;
}

// Direct-execution forward.  Returns false when no plan is installed or when
// the caller passes fixed weights, in which case the shared owner runs the
// generic schedule.
template<typename Precision>
bool E3TensorProductKokkosT<Precision>::try_evaluate_direct_execution(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> dynamic_weights,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(!direct_execution_plan_||dynamic_weights.extent(1)==0) return false;
    ProfileRegion profile("symmetrix/mh1/tensor_product/forward");
    ordered_kokkos_deep_copy(output,Precision(0));
    auto mask=output_mask; auto fixed=internal_weights;
    auto plan=direct_execution_plan_->mh1_instruction_data;
    auto paths=direct_execution_plan_->mh1_path_weights;
    auto indices=direct_execution_plan_->mh1_sparse_indices;
    auto values=direct_execution_plan_->mh1_sparse_values;
    auto component_offsets=direct_execution_plan_->mh1_component_offsets;
    const int instruction_count=direct_execution_plan_->mh1_instruction_count;
    const int multiplicity=direct_execution_plan_->mh1_multiplicity;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
    const int channel_team_size=direct_execution_plan_->mh1_cuda_channel_team_size;
    using channel_policy=Kokkos::TeamPolicy<>;
    Kokkos::parallel_for(
        "e3 tensor product mh1 team channels",
        channel_policy(input_1.extent(0),channel_team_size),
        KOKKOS_LAMBDA(const typename channel_policy::member_type& team) {
            const int sample=team.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team,multiplicity),
                [&](const int channel) {
            for(int instruction=0;instruction<instruction_count;
                ++instruction) {
                const int input_1_offset=plan(instruction,0);
                const int input_2_offset=plan(instruction,1);
                const int output_offset=plan(instruction,2);
                const int input_width=plan(instruction,3);
                const int output_width=plan(instruction,4);
                const int weight_offset=plan(instruction,5);
                const int component_base=plan(instruction,6);
                const Precision scale=paths(instruction)
                    *dynamic_weights(sample,weight_offset+channel);
                for(int component=0;component<output_width;++component) {
                    Precision result=Precision(0);
                    for(int entry=component_offsets(component_base+component);
                        entry<component_offsets(
                            component_base+component+1);++entry)
                        result+=values(entry)
                            *input_1(sample,input_1_offset
                                +channel*input_width+indices(entry,0))
                            *input_2(sample,input_2_offset+indices(entry,1));
                    const int output_index=output_offset
                        +channel*output_width+component;
                    output(sample,output_index)+=
                        scale*result*mask(output_index);
                }
            }
            });
        });
    return true;
    }
#endif
    Kokkos::parallel_for(
        "e3 tensor product mh1 fused",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{static_cast<int>(input_1.extent(0)),multiplicity}),
        KOKKOS_LAMBDA(int sample,int channel) {
            for(int instruction=0;instruction<instruction_count;
                ++instruction) {
                const int input_1_offset=plan(instruction,0);
                const int input_2_offset=plan(instruction,1);
                const int output_offset=plan(instruction,2);
                const int input_width=plan(instruction,3);
                const int output_width=plan(instruction,4);
                const int weight_offset=plan(instruction,5);
                const int component_base=plan(instruction,6);
                const Precision scale=paths(instruction)
                    *dynamic_weights(sample,weight_offset+channel);
                for(int component=0;component<output_width;++component) {
                    Precision result=Precision(0);
                    for(int entry=component_offsets(component_base+component);
                        entry<component_offsets(
                            component_base+component+1);++entry)
                        result+=values(entry)
                            *input_1(sample,input_1_offset
                                +channel*input_width+indices(entry,0))
                            *input_2(sample,input_2_offset+indices(entry,1));
                    const int output_index=output_offset
                        +channel*output_width+component;
                    output(sample,output_index)+=
                        scale*result*mask(output_index);
                }
            }
        });
    return true;
}

// Direct-execution reverse.  Same contract as the forward entry point.
template<typename Precision>
bool E3TensorProductKokkosT<Precision>::try_reverse_direct_execution(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> dynamic_weights,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> weights_adjoint) const
{
    if(!direct_execution_plan_||dynamic_weights.extent(1)==0) return false;
    ProfileRegion profile("symmetrix/mh1/tensor_product/reverse");
    ordered_kokkos_deep_copy(input_1_adjoint,Precision(0));
    ordered_kokkos_deep_copy(input_2_adjoint,Precision(0));
    ordered_kokkos_deep_copy(weights_adjoint,Precision(0));
    auto mask=output_mask; auto fixed=internal_weights;
    auto plan=direct_execution_plan_->mh1_instruction_data;
    auto paths=direct_execution_plan_->mh1_path_weights;
    auto indices=direct_execution_plan_->mh1_sparse_indices;
    auto values=direct_execution_plan_->mh1_sparse_values;
    auto component_offsets=direct_execution_plan_->mh1_component_offsets;
    auto harmonic_offsets=direct_execution_plan_->mh1_harmonic_offsets;
    auto harmonic_terms=direct_execution_plan_->mh1_harmonic_terms;
    auto harmonic_values=direct_execution_plan_->mh1_harmonic_values;
    const int instruction_count=direct_execution_plan_->mh1_instruction_count;
    const int multiplicity=direct_execution_plan_->mh1_multiplicity;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
    const int channel_team_size=direct_execution_plan_->mh1_cuda_channel_team_size;
    const int harmonic_team_size=direct_execution_plan_->mh1_cuda_harmonic_team_size;
    using channel_policy=Kokkos::TeamPolicy<>;
    Kokkos::parallel_for(
        "e3 tensor reverse mh1 team channels",
        channel_policy(input_1.extent(0),channel_team_size),
        KOKKOS_LAMBDA(const typename channel_policy::member_type& team) {
            const int sample=team.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team,multiplicity),
                [&](const int channel) {
            for(int instruction=0;instruction<instruction_count;
                ++instruction) {
                    const int input_1_offset=plan(instruction,0);
                    const int input_2_offset=plan(instruction,1);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_offset=plan(instruction,5);
                    const int component_base=plan(instruction,6);
                    const int first_entry=component_offsets(component_base);
                    const int last_entry=component_offsets(
                        component_base+output_width);
                    const Precision weight=dynamic_weights(
                        sample,weight_offset+channel);
                    Precision weight_value=Precision(0);
                    for(int entry=first_entry;entry<last_entry;++entry) {
                        const int a=indices(entry,0);
                        const int b=indices(entry,1);
                        const int c=indices(entry,2);
                        const int output_index=output_offset
                            +channel*output_width+c;
                        const Precision common=paths(instruction)*values(entry)
                            *mask(output_index)*output_adjoint(sample,output_index);
                        const Precision first=input_1(sample,input_1_offset
                            +channel*input_width+a);
                        const Precision second=input_2(sample,input_2_offset+b);
                        input_1_adjoint(sample,input_1_offset
                            +channel*input_width+a)+=common*weight*second;
                        weight_value+=common*first*second;
                    }
                    weights_adjoint(sample,weight_offset+channel)=weight_value;
            }
            });
        });
    using harmonic_policy=Kokkos::TeamPolicy<>;
    const int harmonic_dimension=input_2_dimension_;
    Kokkos::parallel_for(
        "e3 tensor reverse mh1 team harmonics",
        harmonic_policy(
            input_1.extent(0)*harmonic_dimension,harmonic_team_size),
        KOKKOS_LAMBDA(const typename harmonic_policy::member_type& team) {
            const int sample=team.league_rank()/harmonic_dimension;
            const int component=team.league_rank()%harmonic_dimension;
            Precision result=Precision(0);
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team,multiplicity),
                [&](const int channel,Precision& channel_result) {
                for(int term=harmonic_offsets(component);
                    term<harmonic_offsets(component+1);++term) {
                    const int instruction=harmonic_terms(term,0);
                    const int a=harmonic_terms(term,1);
                    const int c=harmonic_terms(term,2);
                    const int input_1_offset=plan(instruction,0);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_offset=plan(instruction,5);
                    const int output_index=output_offset
                        +channel*output_width+c;
                    const Precision common=paths(instruction)
                        *harmonic_values(term)*mask(output_index)
                        *output_adjoint(sample,output_index);
                    channel_result+=common
                        *dynamic_weights(sample,weight_offset+channel)
                        *input_1(sample,input_1_offset
                            +channel*input_width+a);
                }},result);
            Kokkos::single(Kokkos::PerTeam(team),[&]() {
                input_2_adjoint(sample,component)=result;
            });
        });
    return true;
    }
#endif
    Kokkos::parallel_for(
        "e3 tensor reverse mh1 fused channels",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{static_cast<int>(input_1.extent(0)),multiplicity}),
        KOKKOS_LAMBDA(int sample,int channel) {
            for(int instruction=0;instruction<instruction_count;
                ++instruction) {
                const int input_1_offset=plan(instruction,0);
                const int input_2_offset=plan(instruction,1);
                const int output_offset=plan(instruction,2);
                const int input_width=plan(instruction,3);
                const int output_width=plan(instruction,4);
                const int weight_offset=plan(instruction,5);
                const int component_base=plan(instruction,6);
                const int first_entry=component_offsets(component_base);
                const int last_entry=component_offsets(
                    component_base+output_width);
                const Precision weight=dynamic_weights(
                    sample,weight_offset+channel);
                Precision weight_value=Precision(0);
                for(int entry=first_entry;entry<last_entry;++entry) {
                    const int a=indices(entry,0);
                    const int b=indices(entry,1);
                    const int c=indices(entry,2);
                    const int output_index=output_offset
                        +channel*output_width+c;
                    const Precision common=paths(instruction)*values(entry)
                        *mask(output_index)*output_adjoint(sample,output_index);
                    const Precision first=input_1(sample,input_1_offset
                        +channel*input_width+a);
                    const Precision second=input_2(sample,input_2_offset+b);
                    input_1_adjoint(sample,input_1_offset
                        +channel*input_width+a)+=common*weight*second;
                    weight_value+=common*first*second;
                }
                weights_adjoint(sample,weight_offset+channel)=weight_value;
            }
        });
    Kokkos::parallel_for(
        "e3 tensor reverse mh1 fused harmonics",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{static_cast<int>(input_1.extent(0)),input_2_dimension_}),
        KOKKOS_LAMBDA(int sample,int component) {
            Precision result=Precision(0);
            for(int channel=0;channel<multiplicity;++channel)
                for(int term=harmonic_offsets(component);
                    term<harmonic_offsets(component+1);++term) {
                    const int instruction=harmonic_terms(term,0);
                    const int a=harmonic_terms(term,1);
                    const int c=harmonic_terms(term,2);
                    const int input_1_offset=plan(instruction,0);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_offset=plan(instruction,5);
                    const int output_index=output_offset
                        +channel*output_width+c;
                    const Precision common=paths(instruction)
                        *harmonic_values(term)*mask(output_index)
                        *output_adjoint(sample,output_index);
                    result+=common
                        *dynamic_weights(sample,weight_offset+channel)
                        *input_1(sample,input_1_offset
                            +channel*input_width+a);
                }
            input_2_adjoint(sample,component)=result;
        });
    return true;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::try_reverse_from_nodes(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values,
    Kokkos::View<const int*> source_indices,
    int first_edge,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_weights,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        target_node_output_adjoint,
    Kokkos::View<const int*> target_indices,
    Kokkos::View<Precision**,Kokkos::LayoutRight> source_node_input_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> edge_input_2_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> edge_weights_adjoint) const
{
    if(!supports_direct_node_reverse()) return false;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        const std::size_t samples=edge_input_2.extent(0);
        const std::size_t last_edge=first_edge<0
            ?std::size_t(0):static_cast<std::size_t>(first_edge)+samples;
        if(first_edge<0||source_indices.extent(0)<last_edge
            ||target_indices.extent(0)<last_edge
            ||source_node_values.extent(1)
                !=static_cast<std::size_t>(input_1_dimension_)
            ||target_node_output_adjoint.extent(1)
                !=static_cast<std::size_t>(output_dimension_)
            ||source_node_input_adjoint.extent(0)
                !=source_node_values.extent(0)
            ||source_node_input_adjoint.extent(1)
                !=source_node_values.extent(1)
            ||edge_input_2.extent(1)
                !=static_cast<std::size_t>(input_2_dimension_)
            ||edge_weights.extent(0)!=samples
            ||edge_weights.extent(1)
                !=static_cast<std::size_t>(weight_size_)
            ||edge_input_2_adjoint.extent(0)!=samples
            ||edge_input_2_adjoint.extent(1)!=edge_input_2.extent(1)
            ||edge_weights_adjoint.extent(0)!=samples
            ||edge_weights_adjoint.extent(1)!=edge_weights.extent(1))
            throw std::invalid_argument(
                "Kokkos direct-node tensor-product reverse dimensions are inconsistent.");

        ProfileRegion profile("symmetrix/mh1/tensor_product/reverse_direct_nodes");
        auto plan=direct_execution_plan_->mh1_instruction_data;
        auto paths=direct_execution_plan_->mh1_path_weights;
        auto indices=direct_execution_plan_->mh1_sparse_indices;
        auto values=direct_execution_plan_->mh1_sparse_values;
        auto component_offsets=direct_execution_plan_->mh1_component_offsets;
        auto harmonic_offsets=direct_execution_plan_->mh1_harmonic_offsets;
        auto harmonic_terms=direct_execution_plan_->mh1_harmonic_terms;
        auto harmonic_values=direct_execution_plan_->mh1_harmonic_values;
        auto input_components=direct_execution_plan_->mh1_input_component_data;
        auto mask=output_mask;
        const int instruction_count=direct_execution_plan_->mh1_instruction_count;
        const int multiplicity=direct_execution_plan_->mh1_multiplicity;
        const int angular_dimension=direct_execution_plan_->mh1_input_1_angular_dimension;

        using channel_policy=Kokkos::TeamPolicy<>;
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 direct node channels",
            channel_policy(samples,direct_execution_plan_->mh1_cuda_channel_team_size),
            KOKKOS_LAMBDA(const typename channel_policy::member_type& team) {
                const int sample=team.league_rank();
                const int edge=first_edge+sample;
                const int source=source_indices(edge);
                const int target=target_indices(edge);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team,multiplicity),
                    [&](const int channel) {
                    Precision input_values[max_direct_node_input_components_];
                    for(int component=0;component<angular_dimension;++component)
                        input_values[component]=Precision(0);
                    for(int instruction=0;instruction<instruction_count;
                        ++instruction) {
                        const int input_1_offset=plan(instruction,0);
                        const int input_2_offset=plan(instruction,1);
                        const int output_offset=plan(instruction,2);
                        const int input_width=plan(instruction,3);
                        const int output_width=plan(instruction,4);
                        const int weight_offset=plan(instruction,5);
                        const int component_base=plan(instruction,6);
                        const int first_entry=component_offsets(component_base);
                        const int last_entry=component_offsets(
                            component_base+output_width);
                        const Precision weight=edge_weights(
                            sample,weight_offset+channel);
                        Precision weight_value=Precision(0);
                        for(int entry=first_entry;entry<last_entry;++entry) {
                            const int a=indices(entry,0);
                            const int b=indices(entry,1);
                            const int c=indices(entry,2);
                            const int output_index=output_offset
                                +channel*output_width+c;
                            const Precision common=paths(instruction)*values(entry)
                                *mask(output_index)
                                *target_node_output_adjoint(target,output_index);
                            const Precision first=source_node_values(
                                source,input_1_offset+channel*input_width+a);
                            const Precision second=edge_input_2(
                                sample,input_2_offset+b);
                            input_values[input_1_offset/multiplicity+a]
                                +=common*weight*second;
                            weight_value+=common*first*second;
                        }
                        edge_weights_adjoint(
                            sample,weight_offset+channel)=weight_value;
                    }
                    for(int component=0;component<angular_dimension;++component) {
                        const Precision value=input_values[component];
                        if(value!=Precision(0))
                            Kokkos::atomic_add(
                                &source_node_input_adjoint(
                                    source,input_components(component,0)
                                        +channel*input_components(component,1)),
                                value);
                    }
                });
            });

        using harmonic_policy=Kokkos::TeamPolicy<>;
        const int harmonic_dimension=input_2_dimension_;
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 direct node harmonics",
            harmonic_policy(
                samples*harmonic_dimension,direct_execution_plan_->mh1_cuda_harmonic_team_size),
            KOKKOS_LAMBDA(const typename harmonic_policy::member_type& team) {
                const int sample=team.league_rank()/harmonic_dimension;
                const int component=team.league_rank()%harmonic_dimension;
                const int edge=first_edge+sample;
                const int source=source_indices(edge);
                const int target=target_indices(edge);
                Precision result=Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team,multiplicity),
                    [&](const int channel,Precision& channel_result) {
                    for(int term=harmonic_offsets(component);
                        term<harmonic_offsets(component+1);++term) {
                        const int instruction=harmonic_terms(term,0);
                        const int a=harmonic_terms(term,1);
                        const int c=harmonic_terms(term,2);
                        const int input_1_offset=plan(instruction,0);
                        const int output_offset=plan(instruction,2);
                        const int input_width=plan(instruction,3);
                        const int output_width=plan(instruction,4);
                        const int weight_offset=plan(instruction,5);
                        const int output_index=output_offset
                            +channel*output_width+c;
                        const Precision common=paths(instruction)
                            *harmonic_values(term)*mask(output_index)
                            *target_node_output_adjoint(target,output_index);
                        channel_result+=common
                            *edge_weights(sample,weight_offset+channel)
                            *source_node_values(
                                source,input_1_offset+channel*input_width+a);
                    }},result);
                Kokkos::single(Kokkos::PerTeam(team),[&]() {
                    edge_input_2_adjoint(sample,component)=result;
                });
            });
        return true;
    }
#endif
    return false;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::try_evaluate_execution_uvu_from_nodes(
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
    Kokkos::View<Precision**,Kokkos::LayoutRight> node_messages) const
{
    if(!supports_execution_direct_uvu()) return false;
    const std::size_t samples=edge_phi.extent(0);
    const std::size_t last_edge=first_edge<0
        ?std::size_t(0):static_cast<std::size_t>(first_edge)+samples;
    const bool bias_valid=linear_bias.extent(0)==0
        ||linear_bias.extent(0)==static_cast<std::size_t>(weight_size_);
    const bool contribution_valid=edge_linear_contribution.size()==0
        ||(edge_linear_contribution.extent(0)==samples
            &&edge_linear_contribution.extent(1)
                ==static_cast<std::size_t>(weight_size_));
    if(first_edge<0||source_indices.extent(0)<last_edge
        ||target_indices.extent(0)<last_edge
        ||receiver_offsets.extent(0)!=node_messages.extent(0)+1
        ||source_node_values.extent(1)
            !=static_cast<std::size_t>(input_1_dimension_)
        ||node_messages.extent(1)
            !=static_cast<std::size_t>(output_dimension_)
        ||edge_input_2.extent(0)!=samples
        ||edge_input_2.extent(1)
            !=static_cast<std::size_t>(input_2_dimension_)
        ||linear_weight.extent(0)
            !=static_cast<std::size_t>(weight_size_)
        ||linear_weight.extent(1)!=edge_phi.extent(1)
        ||!bias_valid
        ||!contribution_valid
        ||(apply_edge_cutoff_scale
            &&edge_cutoff_scale.extent(0)<last_edge))
        throw std::invalid_argument(
            "Kokkos Execution direct UVU forward dimensions are inconsistent.");

    ProfileRegion profile("symmetrix/mh1/execution_uvu/forward_direct_nodes");
    auto plan=direct_execution_plan_->mh1_instruction_data;
    auto paths=direct_execution_plan_->mh1_path_weights;
    auto indices=direct_execution_plan_->mh1_sparse_indices;
    auto values=direct_execution_plan_->mh1_sparse_values;
    auto component_offsets=direct_execution_plan_->mh1_component_offsets;
    auto mask=output_mask;
    const int instruction_count=direct_execution_plan_->mh1_instruction_count;
    const int multiplicity=direct_execution_plan_->mh1_multiplicity;
    const int phi_dimension=edge_phi.extent(1);
    const int block_end=first_edge+static_cast<int>(samples);
    const bool has_bias=linear_bias.extent(0)!=0;
    const bool has_edge_contribution=edge_linear_contribution.size()!=0;

    Kokkos::parallel_for(
        "e3 tensor execution uvu receiver-owned forward",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{static_cast<int>(active_receivers.extent(0)),multiplicity}),
        KOKKOS_LAMBDA(int owner,int channel) {
            const int receiver=active_receivers(owner);
            const int receiver_begin=receiver_offsets(receiver);
            const int receiver_end=receiver_offsets(receiver+1);
            const int begin=receiver_begin>first_edge
                ?receiver_begin:first_edge;
            const int end=receiver_end<block_end
                ?receiver_end:block_end;
            if(begin>=end) return;
            for(int instruction=0;instruction<instruction_count;
                ++instruction) {
                const int input_1_offset=plan(instruction,0);
                const int input_2_offset=plan(instruction,1);
                const int output_offset=plan(instruction,2);
                const int input_width=plan(instruction,3);
                const int output_width=plan(instruction,4);
                const int weight_index=plan(instruction,5)+channel;
                const int component_base=plan(instruction,6);
                for(int edge=begin;edge<end;++edge) {
                    if(target_indices(edge)!=receiver) continue;
                    const int local_edge=edge-first_edge;
                    const int source=source_indices(edge);
                    Precision weight=has_bias
                        ?linear_bias(weight_index):Precision(0);
                    if(has_edge_contribution)
                        weight+=edge_linear_contribution(
                            local_edge,weight_index);
                    for(int phi=0;phi<phi_dimension;++phi)
                        weight+=linear_weight(weight_index,phi)
                            *edge_phi(local_edge,phi);
                    if(apply_edge_cutoff_scale)
                        weight*=edge_cutoff_scale(edge);
                    const Precision scale=paths(instruction)*weight;
                    for(int component=0;component<output_width;++component) {
                        const int first_entry=component_offsets(
                            component_base+component);
                        const int last_entry=component_offsets(
                            component_base+component+1);
                        Precision product=Precision(0);
                        for(int entry=first_entry;entry<last_entry;++entry)
                            product+=values(entry)
                                *source_node_values(source,input_1_offset
                                    +channel*input_width+indices(entry,0))
                                *edge_input_2(local_edge,input_2_offset
                                    +indices(entry,1));
                        const int output_index=output_offset
                            +channel*output_width+component;
                        node_messages(receiver,output_index)+=
                            scale*mask(output_index)*product;
                    }
                }
            }
        });
    return true;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::try_reverse_execution_uvu_from_nodes(
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
    Kokkos::View<Precision*> edge_cutoff_scale_adjoint) const
{
    if(!supports_execution_direct_uvu()) return false;
    const std::size_t samples=edge_phi.extent(0);
    const std::size_t last_edge=first_edge<0
        ?std::size_t(0):static_cast<std::size_t>(first_edge)+samples;
    const bool bias_valid=linear_bias.extent(0)==0
        ||linear_bias.extent(0)==static_cast<std::size_t>(weight_size_);
    const bool contribution_valid=edge_linear_contribution.size()==0
        ||(edge_linear_contribution.extent(0)==samples
            &&edge_linear_contribution.extent(1)
                ==static_cast<std::size_t>(weight_size_));
    if(first_edge<0||source_indices.extent(0)<last_edge
        ||target_indices.extent(0)<last_edge
        ||source_edge_offsets.extent(0)==0
        ||source_edge_indices.extent(0)<last_edge
        ||source_node_values.extent(1)
            !=static_cast<std::size_t>(input_1_dimension_)
        ||source_node_input_adjoint.extent(0)
            !=source_node_values.extent(0)
        ||source_node_input_adjoint.extent(1)
            !=source_node_values.extent(1)
        ||target_node_output_adjoint.extent(1)
            !=static_cast<std::size_t>(output_dimension_)
        ||edge_input_2.extent(0)!=samples
        ||edge_input_2.extent(1)
            !=static_cast<std::size_t>(input_2_dimension_)
        ||linear_weight.extent(0)
            !=static_cast<std::size_t>(weight_size_)
        ||linear_weight.extent(1)!=edge_phi.extent(1)
        ||!bias_valid
        ||!contribution_valid
        ||(apply_edge_cutoff_scale
            &&edge_cutoff_scale.extent(0)<last_edge)
        ||edge_phi_adjoint.extent(0)!=samples
        ||edge_phi_adjoint.extent(1)!=edge_phi.extent(1)
        ||edge_input_2_adjoint.extent(0)!=samples
        ||edge_input_2_adjoint.extent(1)!=edge_input_2.extent(1)
        ||edge_cutoff_scale_adjoint.extent(0)!=samples)
        throw std::invalid_argument(
            "Kokkos Execution direct UVU reverse dimensions are inconsistent.");

    ProfileRegion profile("symmetrix/mh1/execution_uvu/reverse_direct_nodes");
    auto plan=direct_execution_plan_->mh1_instruction_data;
    auto paths=direct_execution_plan_->mh1_path_weights;
    auto indices=direct_execution_plan_->mh1_sparse_indices;
    auto values=direct_execution_plan_->mh1_sparse_values;
    auto component_offsets=direct_execution_plan_->mh1_component_offsets;
    auto harmonic_offsets=direct_execution_plan_->mh1_harmonic_offsets;
    auto harmonic_terms=direct_execution_plan_->mh1_harmonic_terms;
    auto harmonic_values=direct_execution_plan_->mh1_harmonic_values;
    auto input_components=direct_execution_plan_->mh1_input_component_data;
    auto input_term_offsets=direct_execution_plan_->mh1_input_term_offsets;
    auto input_terms=direct_execution_plan_->mh1_input_terms;
    auto input_values=direct_execution_plan_->mh1_input_values;
    auto mask=output_mask;
    const int instruction_count=direct_execution_plan_->mh1_instruction_count;
    const int multiplicity=direct_execution_plan_->mh1_multiplicity;
    const int angular_dimension=direct_execution_plan_->mh1_input_1_angular_dimension;
    const int harmonic_dimension=input_2_dimension_;
    const int phi_dimension=edge_phi.extent(1);
    const bool has_bias=linear_bias.extent(0)!=0;
    const bool has_edge_contribution=edge_linear_contribution.size()!=0;

    Kokkos::parallel_for(
        "e3 tensor execution uvu source-owned reverse",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0,0,0},{static_cast<int>(source_edge_offsets.extent(0))-1,
                multiplicity,angular_dimension}),
        KOKKOS_LAMBDA(int owner,int channel,int input_component) {
            const int first_position=source_edge_offsets(owner);
            const int source=source_indices(source_edge_indices(first_position));
            Precision result=Precision(0);
            for(int position=source_edge_offsets(owner);
                position<source_edge_offsets(owner+1);++position) {
                const int edge=source_edge_indices(position);
                const int local_edge=edge-first_edge;
                const int target=target_indices(edge);
                int previous_instruction=-1;
                Precision weight=Precision(0);
                for(int term=input_term_offsets(input_component);
                    term<input_term_offsets(input_component+1);++term) {
                    const int instruction=input_terms(term,0);
                    const int b=input_terms(term,1);
                    const int c=input_terms(term,2);
                    const int input_2_offset=plan(instruction,1);
                    const int output_offset=plan(instruction,2);
                    const int output_width=plan(instruction,4);
                    const int weight_index=plan(instruction,5)+channel;
                    if(instruction!=previous_instruction) {
                        weight=has_bias
                            ?linear_bias(weight_index):Precision(0);
                        if(has_edge_contribution)
                            weight+=edge_linear_contribution(
                                local_edge,weight_index);
                        for(int phi=0;phi<phi_dimension;++phi)
                            weight+=linear_weight(weight_index,phi)
                                *edge_phi(local_edge,phi);
                        if(apply_edge_cutoff_scale)
                            weight*=edge_cutoff_scale(edge);
                        previous_instruction=instruction;
                    }
                    const int output_index=output_offset
                        +channel*output_width+c;
                    result+=paths(instruction)*input_values(term)
                        *mask(output_index)
                        *target_node_output_adjoint(target,output_index)
                        *weight*edge_input_2(local_edge,input_2_offset+b);
                }
            }
            source_node_input_adjoint(
                source,input_components(input_component,0)
                    +channel*input_components(input_component,1))+=result;
        });

    Kokkos::parallel_for(
        "e3 tensor execution uvu compact prefix reverse",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{static_cast<int>(samples),phi_dimension}),
        KOKKOS_LAMBDA(int local_edge,int phi) {
            const int edge=first_edge+local_edge;
            const int source=source_indices(edge);
            const int target=target_indices(edge);
            const Precision cutoff=apply_edge_cutoff_scale
                ?edge_cutoff_scale(edge):Precision(1);
            Precision result=Precision(0);
            for(int instruction=0;instruction<instruction_count;
                ++instruction) {
                const int input_1_offset=plan(instruction,0);
                const int input_2_offset=plan(instruction,1);
                const int output_offset=plan(instruction,2);
                const int input_width=plan(instruction,3);
                const int output_width=plan(instruction,4);
                const int weight_offset=plan(instruction,5);
                const int component_base=plan(instruction,6);
                const int first_entry=component_offsets(component_base);
                const int last_entry=component_offsets(
                    component_base+output_width);
                for(int channel=0;channel<multiplicity;++channel) {
                    Precision weight_adjoint=Precision(0);
                    for(int entry=first_entry;entry<last_entry;++entry) {
                        const int output_index=output_offset
                            +channel*output_width+indices(entry,2);
                        weight_adjoint+=paths(instruction)*values(entry)
                            *mask(output_index)
                            *target_node_output_adjoint(target,output_index)
                            *source_node_values(source,input_1_offset
                                +channel*input_width+indices(entry,0))
                            *edge_input_2(local_edge,input_2_offset
                                +indices(entry,1));
                    }
                    result+=cutoff
                        *linear_weight(weight_offset+channel,phi)
                        *weight_adjoint;
                }
            }
            edge_phi_adjoint(local_edge,phi)=result;
        });

    Kokkos::parallel_for(
        "e3 tensor execution uvu harmonic reverse",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{static_cast<int>(samples),harmonic_dimension}),
        KOKKOS_LAMBDA(int local_edge,int component) {
            const int edge=first_edge+local_edge;
            const int source=source_indices(edge);
            const int target=target_indices(edge);
            const Precision cutoff=apply_edge_cutoff_scale
                ?edge_cutoff_scale(edge):Precision(1);
            Precision result=Precision(0);
            for(int channel=0;channel<multiplicity;++channel) {
                int previous_instruction=-1;
                Precision weight=Precision(0);
                for(int term=harmonic_offsets(component);
                    term<harmonic_offsets(component+1);++term) {
                    const int instruction=harmonic_terms(term,0);
                    const int a=harmonic_terms(term,1);
                    const int c=harmonic_terms(term,2);
                    const int input_1_offset=plan(instruction,0);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_index=plan(instruction,5)+channel;
                    if(instruction!=previous_instruction) {
                        weight=has_bias
                            ?linear_bias(weight_index):Precision(0);
                        if(has_edge_contribution)
                            weight+=edge_linear_contribution(
                                local_edge,weight_index);
                        for(int phi=0;phi<phi_dimension;++phi)
                            weight+=linear_weight(weight_index,phi)
                                *edge_phi(local_edge,phi);
                        previous_instruction=instruction;
                    }
                    const int output_index=output_offset
                        +channel*output_width+c;
                    result+=paths(instruction)*harmonic_values(term)
                        *mask(output_index)
                        *target_node_output_adjoint(target,output_index)
                        *source_node_values(source,input_1_offset
                            +channel*input_width+a)*cutoff*weight;
                }
            }
            edge_input_2_adjoint(local_edge,component)=result;
        });

    Kokkos::parallel_for(
        "e3 tensor execution uvu cutoff reverse",samples,
        KOKKOS_LAMBDA(int local_edge) {
            if(!apply_edge_cutoff_scale) {
                edge_cutoff_scale_adjoint(local_edge)=Precision(0);
                return;
            }
            const int edge=first_edge+local_edge;
            const int source=source_indices(edge);
            const int target=target_indices(edge);
            Precision result=Precision(0);
            for(int instruction=0;instruction<instruction_count;
                ++instruction) {
                const int input_1_offset=plan(instruction,0);
                const int input_2_offset=plan(instruction,1);
                const int output_offset=plan(instruction,2);
                const int input_width=plan(instruction,3);
                const int output_width=plan(instruction,4);
                const int weight_offset=plan(instruction,5);
                const int component_base=plan(instruction,6);
                const int first_entry=component_offsets(component_base);
                const int last_entry=component_offsets(
                    component_base+output_width);
                for(int channel=0;channel<multiplicity;++channel) {
                    const int weight_index=weight_offset+channel;
                    Precision weight=has_bias
                        ?linear_bias(weight_index):Precision(0);
                    if(has_edge_contribution)
                        weight+=edge_linear_contribution(
                            local_edge,weight_index);
                    for(int phi=0;phi<phi_dimension;++phi)
                        weight+=linear_weight(weight_index,phi)
                            *edge_phi(local_edge,phi);
                    Precision weight_adjoint=Precision(0);
                    for(int entry=first_entry;entry<last_entry;++entry) {
                        const int output_index=output_offset
                            +channel*output_width+indices(entry,2);
                        weight_adjoint+=paths(instruction)*values(entry)
                            *mask(output_index)
                            *target_node_output_adjoint(target,output_index)
                            *source_node_values(source,input_1_offset
                                +channel*input_width+indices(entry,0))
                            *edge_input_2(local_edge,input_2_offset
                                +indices(entry,1));
                    }
                    result+=weight*weight_adjoint;
                }
            }
            edge_cutoff_scale_adjoint(local_edge)=result;
        });
    return true;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::
try_evaluate_execution_spline_uvu_from_nodes(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values_ir_mul,
    Kokkos::View<const int*> source_indices,
    Kokkos::View<const int*> target_indices,
    Kokkos::View<const int*> active_receivers,
    Kokkos::View<const int*> receiver_offsets,
    Kokkos::View<const int*> source_types,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const double*> distances,
    Kokkos::View<const int*> spline_intervals,
    Kokkos::View<const Precision*> spline_coordinates,
    const int type_count,
    RadialFunctionSetKokkos<Precision> spline,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
    Kokkos::View<Precision**,Kokkos::LayoutRight> node_messages_ir_mul,
    Kokkos::View<Precision*> node_density) const
{
    if(!supports_execution_direct_uvu()
        ||direct_execution_plan_->mh1_max_output_width>max_direct_node_output_components_) return false;
    const auto edges=source_indices.extent(0);
    if(type_count<=0
        ||target_indices.extent(0)!=edges||source_types.extent(0)!=edges
        ||distances.extent(0)!=edges||spline_intervals.extent(0)!=edges
        ||spline_coordinates.extent(0)!=edges||edge_input_2.extent(0)!=edges
        ||edge_input_2.extent(1)!=static_cast<std::size_t>(input_2_dimension_)
        ||receiver_offsets.extent(0)!=node_messages_ir_mul.extent(0)+1
        ||node_types.extent(0)!=node_messages_ir_mul.extent(0)
        ||node_density.extent(0)!=node_messages_ir_mul.extent(0)
        ||source_node_values_ir_mul.extent(1)
            !=static_cast<std::size_t>(input_1_dimension_)
        ||node_messages_ir_mul.extent(1)
            !=static_cast<std::size_t>(output_dimension_)
        ||spline.edge_type_count()!=type_count*type_count
        ||spline.function_count()!=weight_size_+1)
        throw std::invalid_argument(
            "Kokkos Execution direct spline UVU forward dimensions are inconsistent.");

    ProfileRegion profile("symmetrix/mh1/execution_spline_uvu/forward_direct_nodes");
    auto plan=direct_execution_plan_->mh1_instruction_data;
    auto paths=direct_execution_plan_->mh1_path_weights;
    auto indices=direct_execution_plan_->mh1_sparse_indices;
    auto values=direct_execution_plan_->mh1_sparse_values;
    auto component_offsets=direct_execution_plan_->mh1_component_offsets;
    auto mask=output_mask;
    const int instruction_count=direct_execution_plan_->mh1_instruction_count;
    const int multiplicity=direct_execution_plan_->mh1_multiplicity;
    const int density_function=weight_size_;

    // A 64-channel tile maps to one wave64 or two wave32/CUDA warps while
    // retaining a backend-neutral Kokkos launch policy.
    Kokkos::parallel_for(
        "e3 tensor execution spline uvu receiver-owned forward",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{static_cast<int>(active_receivers.extent(0)),multiplicity},
            {1,direct_channel_tile_}),
        KOKKOS_LAMBDA(int owner,int channel) {
            const int receiver=active_receivers(owner);
            const int begin=receiver_offsets(receiver);
            const int end=receiver_offsets(receiver+1);
            if(channel==0) {
                Precision density=Precision(0);
                for(int edge=begin;edge<end;++edge) {
                    const int pair=source_types(edge)*type_count
                        +node_types(receiver);
                    const auto point=spline.cached_evaluation_point(
                        spline_intervals(edge),spline_coordinates(edge));
                    density+=spline.evaluate_function(
                        pair,point,density_function);
                }
                node_density(receiver)+=density;
            }
            for(int instruction=0;instruction<instruction_count;
                ++instruction) {
                const int input_1_offset=plan(instruction,0);
                const int input_2_offset=plan(instruction,1);
                const int output_offset=plan(instruction,2);
                const int output_width=plan(instruction,4);
                const int weight_index=plan(instruction,5)+channel;
                const int component_base=plan(instruction,6);
                Precision results[max_direct_node_output_components_]={};
                for(int edge=begin;edge<end;++edge) {
                    if(target_indices(edge)!=receiver) continue;
                    const int source=source_indices(edge);
                    const int pair=source_types(edge)*type_count
                        +node_types(receiver);
                    const auto point=spline.cached_evaluation_point(
                        spline_intervals(edge),spline_coordinates(edge));
                    const Precision scale=paths(instruction)
                        *spline.evaluate_function(pair,point,weight_index);
                    for(int component=0;component<output_width;++component) {
                        const int first_entry=component_offsets(
                            component_base+component);
                        const int last_entry=component_offsets(
                            component_base+component+1);
                        Precision product=Precision(0);
                        for(int entry=first_entry;entry<last_entry;++entry)
                            product+=values(entry)
                                *source_node_values_ir_mul(source,input_1_offset
                                    +indices(entry,0)*multiplicity+channel)
                                *edge_input_2(edge,input_2_offset+indices(entry,1));
                        const int output_index=output_offset
                            +component*multiplicity+channel;
                        results[component]+=
                            scale*mask(output_index)*product;
                    }
                }
                for(int component=0;component<output_width;++component)
                    node_messages_ir_mul(receiver,output_offset
                        +component*multiplicity+channel)+=results[component];
            }
        });
    return true;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::
try_reverse_execution_spline_uvu_from_nodes(
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
    const int type_count,
    RadialFunctionSetKokkos<Precision> spline,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        target_node_output_adjoint,
    Kokkos::View<const Precision*> node_density_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> source_node_input_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> edge_input_2_adjoint,
    Kokkos::View<Precision*> distance_adjoint) const
{
    if(!supports_execution_direct_uvu()) return false;
    const auto edges=source_indices.extent(0);
    if(type_count<=0||source_edge_offsets.extent(0)==0
        ||source_edge_indices.extent(0)!=edges
        ||target_indices.extent(0)!=edges||source_types.extent(0)!=edges
        ||distances.extent(0)!=edges||spline_intervals.extent(0)!=edges
        ||spline_coordinates.extent(0)!=edges||edge_input_2.extent(0)!=edges
        ||edge_input_2.extent(1)!=static_cast<std::size_t>(input_2_dimension_)
        ||edge_input_2_adjoint.extent(0)!=edges
        ||edge_input_2_adjoint.extent(1)!=edge_input_2.extent(1)
        ||distance_adjoint.extent(0)!=edges
        ||node_types.extent(0)!=source_node_values.extent(0)
        ||node_density_adjoint.extent(0)!=source_node_values.extent(0)
        ||source_node_input_adjoint.extent(0)!=source_node_values.extent(0)
        ||source_node_input_adjoint.extent(1)!=source_node_values.extent(1)
        ||source_node_values.extent(1)
            !=static_cast<std::size_t>(input_1_dimension_)
        ||target_node_output_adjoint.extent(0)!=source_node_values.extent(0)
        ||target_node_output_adjoint.extent(1)
            !=static_cast<std::size_t>(output_dimension_)
        ||spline.edge_type_count()!=type_count*type_count
        ||spline.function_count()!=weight_size_+1)
        throw std::invalid_argument(
            "Kokkos Execution direct spline UVU reverse dimensions are inconsistent.");

    ProfileRegion profile("symmetrix/mh1/execution_spline_uvu/reverse_direct_nodes");
    auto plan=direct_execution_plan_->mh1_instruction_data;
    auto paths=direct_execution_plan_->mh1_path_weights;
    auto indices=direct_execution_plan_->mh1_sparse_indices;
    auto values=direct_execution_plan_->mh1_sparse_values;
    auto component_offsets=direct_execution_plan_->mh1_component_offsets;
    auto harmonic_offsets=direct_execution_plan_->mh1_harmonic_offsets;
    auto harmonic_terms=direct_execution_plan_->mh1_harmonic_terms;
    auto harmonic_values=direct_execution_plan_->mh1_harmonic_values;
    auto input_components=direct_execution_plan_->mh1_input_component_data;
    auto input_instruction_term_offsets=direct_execution_plan_->mh1_input_instruction_term_offsets;
    auto input_terms=direct_execution_plan_->mh1_input_terms;
    auto input_values=direct_execution_plan_->mh1_input_values;
    auto mask=output_mask;
    const int instruction_count=direct_execution_plan_->mh1_instruction_count;
    const int multiplicity=direct_execution_plan_->mh1_multiplicity;
    const int angular_dimension=direct_execution_plan_->mh1_input_1_angular_dimension;
    const int harmonic_dimension=input_2_dimension_;
    const int density_function=weight_size_;

    Kokkos::parallel_for(
        "e3 tensor execution spline uvu source-owned reverse",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{static_cast<int>(source_edge_offsets.extent(0))-1,multiplicity},
            {1,direct_channel_tile_}),
        KOKKOS_LAMBDA(int owner,int channel) {
            const int first_position=source_edge_offsets(owner);
            const int end_position=source_edge_offsets(owner+1);
            if(first_position==end_position) return;
            const int source=source_indices(source_edge_indices(first_position));
            Precision results[max_direct_node_input_components_]={};
            for(int position=first_position;position<end_position;++position) {
                const int edge=source_edge_indices(position);
                const int target=target_indices(edge);
                const int pair=source_types(edge)*type_count+node_types(target);
                const auto point=spline.cached_evaluation_point(
                    spline_intervals(edge),spline_coordinates(edge));
                for(int instruction=0;instruction<instruction_count;
                    ++instruction) {
                    const int input_2_offset=plan(instruction,1);
                    const int output_offset=plan(instruction,2);
                    const int output_width=plan(instruction,4);
                    const Precision scale=paths(instruction)
                        *spline.evaluate_function(
                            pair,point,plan(instruction,5)+channel);
                    for(int input_component=0;
                        input_component<angular_dimension;++input_component) {
                        Precision result=Precision(0);
                        for(int term=input_instruction_term_offsets(
                                input_component,instruction);
                            term<input_instruction_term_offsets(
                                input_component,instruction+1);++term) {
                            const int b=input_terms(term,1);
                            const int c=input_terms(term,2);
                            const int output_index=output_offset
                                +channel*output_width+c;
                            result+=input_values(term)*mask(output_index)
                                *target_node_output_adjoint(target,output_index)
                                *edge_input_2(edge,input_2_offset+b);
                        }
                        results[input_component]+=scale*result;
                    }
                }
            }
            for(int input_component=0;input_component<angular_dimension;
                ++input_component)
                source_node_input_adjoint(
                    source,input_components(input_component,0)
                        +channel*input_components(input_component,1))
                    +=results[input_component];
        });

    if(harmonic_dimension<=max_direct_harmonic_components_) {
        using team_policy=Kokkos::TeamPolicy<>;
        using member_type=team_policy::member_type;
        // Bound both the thread-local accumulator and team scratch. Models
        // with wider harmonic inputs use the portable fallback below.
        const int preferred_team_size=
            instruction_count<=4?direct_light_edge_team_size_
                                :direct_edge_team_size_;
        const int team_size=std::min(multiplicity,preferred_team_size);
        const int scratch_columns=harmonic_dimension+1;
        const std::size_t scratch_values=static_cast<std::size_t>(team_size)
            *static_cast<std::size_t>(scratch_columns);
        const std::size_t scratch_bytes=scratch_values*sizeof(Precision);
        team_policy policy(static_cast<int>(edges),team_size);
        policy.set_scratch_size(0,Kokkos::PerTeam(scratch_bytes));
        Kokkos::parallel_for(
            "e3 tensor execution spline uvu fused edge reverse",policy,
            KOKKOS_LAMBDA(const member_type& team) {
            const int edge=team.league_rank();
            const int source=source_indices(edge);
            const int target=target_indices(edge);
            const int pair=source_types(edge)*type_count+node_types(target);
            const auto point=spline.cached_evaluation_point(
                spline_intervals(edge),spline_coordinates(edge));
            Precision harmonic_result[max_direct_harmonic_components_];
            for(int component=0;component<harmonic_dimension;++component)
                harmonic_result[component]=Precision(0);
            Precision distance_result=Precision(0);
            for(int channel=team.team_rank();channel<multiplicity;
                channel+=team.team_size()) {
                for(int instruction=0;instruction<instruction_count;
                    ++instruction) {
                    const int input_1_offset=plan(instruction,0);
                    const int input_2_offset=plan(instruction,1);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_index=plan(instruction,5)+channel;
                    const int component_base=plan(instruction,6);
                    const int first_entry=component_offsets(component_base);
                    const int last_entry=component_offsets(
                        component_base+output_width);
                    Precision weight,weight_derivative;
                    spline.evaluate_function(
                        pair,point,weight_index,weight,weight_derivative);
                    Precision weight_adjoint=Precision(0);
                    for(int entry=first_entry;entry<last_entry;++entry) {
                        const int a=indices(entry,0);
                        const int b=indices(entry,1);
                        const int c=indices(entry,2);
                        const int output_index=output_offset
                            +channel*output_width+c;
                        const Precision common=paths(instruction)*values(entry)
                            *mask(output_index)
                            *target_node_output_adjoint(target,output_index)
                            *source_node_values(source,input_1_offset
                                +channel*input_width+a);
                        harmonic_result[input_2_offset+b]+=common*weight;
                        weight_adjoint+=common
                            *edge_input_2(edge,input_2_offset+b);
                    }
                    distance_result+=weight_derivative*weight_adjoint;
                }
            }
            auto* scratch=static_cast<Precision*>(
                team.team_shmem().get_shmem(scratch_bytes));
            const int scratch_offset=team.team_rank()*scratch_columns;
            for(int component=0;component<harmonic_dimension;++component)
                scratch[scratch_offset+component]=harmonic_result[component];
            scratch[scratch_offset+harmonic_dimension]=distance_result;
            team.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team,harmonic_dimension),
                [&](int component) {
                    Precision result=Precision(0);
                    for(int rank=0;rank<team.team_size();++rank)
                        result+=scratch[rank*scratch_columns+component];
                    edge_input_2_adjoint(edge,component)+=result;
                });
            Kokkos::single(Kokkos::PerTeam(team),[&]() {
                Precision result=Precision(0);
                for(int rank=0;rank<team.team_size();++rank)
                    result+=scratch[rank*scratch_columns+harmonic_dimension];
                Precision density_value,density_derivative;
                spline.evaluate_function(
                    pair,point,density_function,density_value,density_derivative);
                distance_adjoint(edge)+=result
                    +node_density_adjoint(target)*density_derivative;
            });
        });
    } else {
        Kokkos::parallel_for(
            "e3 tensor execution spline uvu harmonic reverse",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0,0},{static_cast<int>(edges),harmonic_dimension}),
            KOKKOS_LAMBDA(int edge,int component) {
                const int source=source_indices(edge);
                const int target=target_indices(edge);
                const int pair=source_types(edge)*type_count+node_types(target);
                const auto point=spline.cached_evaluation_point(
                    spline_intervals(edge),spline_coordinates(edge));
                Precision result=Precision(0);
                for(int channel=0;channel<multiplicity;++channel) {
                    int previous_instruction=-1;
                    Precision weight=Precision(0);
                    for(int term=harmonic_offsets(component);
                        term<harmonic_offsets(component+1);++term) {
                        const int instruction=harmonic_terms(term,0);
                        const int a=harmonic_terms(term,1);
                        const int c=harmonic_terms(term,2);
                        const int input_1_offset=plan(instruction,0);
                        const int output_offset=plan(instruction,2);
                        const int input_width=plan(instruction,3);
                        const int output_width=plan(instruction,4);
                        if(instruction!=previous_instruction) {
                            weight=spline.evaluate_function(
                                pair,point,plan(instruction,5)+channel);
                            previous_instruction=instruction;
                        }
                        const int output_index=output_offset
                            +channel*output_width+c;
                        result+=paths(instruction)*harmonic_values(term)
                            *mask(output_index)
                            *target_node_output_adjoint(target,output_index)
                            *source_node_values(source,input_1_offset
                                +channel*input_width+a)*weight;
                    }
                }
                edge_input_2_adjoint(edge,component)+=result;
            });

        Kokkos::parallel_for(
            "e3 tensor execution spline uvu distance reverse",edges,
            KOKKOS_LAMBDA(int edge) {
                const int source=source_indices(edge);
                const int target=target_indices(edge);
                const int pair=source_types(edge)*type_count+node_types(target);
                const auto point=spline.cached_evaluation_point(
                    spline_intervals(edge),spline_coordinates(edge));
                Precision result=Precision(0);
                for(int instruction=0;instruction<instruction_count;++instruction) {
                    const int input_1_offset=plan(instruction,0);
                    const int input_2_offset=plan(instruction,1);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_offset=plan(instruction,5);
                    const int component_base=plan(instruction,6);
                    const int first_entry=component_offsets(component_base);
                    const int last_entry=component_offsets(
                        component_base+output_width);
                    for(int channel=0;channel<multiplicity;++channel) {
                        Precision weight_value,weight_derivative;
                        spline.evaluate_function(
                            pair,point,weight_offset+channel,
                            weight_value,weight_derivative);
                        Precision weight_adjoint=Precision(0);
                        for(int entry=first_entry;entry<last_entry;++entry) {
                            const int output_index=output_offset
                                +channel*output_width+indices(entry,2);
                            weight_adjoint+=paths(instruction)*values(entry)
                                *mask(output_index)
                                *target_node_output_adjoint(target,output_index)
                                *source_node_values(source,input_1_offset
                                    +channel*input_width+indices(entry,0))
                                *edge_input_2(edge,input_2_offset+indices(entry,1));
                        }
                        result+=weight_derivative*weight_adjoint;
                    }
                }
                Precision density_value,density_derivative;
                spline.evaluate_function(
                    pair,point,density_function,density_value,density_derivative);
                distance_adjoint(edge)+=
                    result+node_density_adjoint(target)*density_derivative;
            });
    }
    return true;
}

template class E3TensorProductKokkosT<float>;
template class E3TensorProductKokkosT<double>;
