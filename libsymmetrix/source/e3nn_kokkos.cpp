#include "e3nn_kokkos.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "KokkosBlas.hpp"
#include "cblas.hpp"
#include "tools_kokkos.hpp"

namespace {
constexpr int linear_host_transpose_channel_tile=16;
// Four samples keep the channel-major and sample-major sides of the host
// transpose resident together for the wide MH-1 node features.
constexpr int linear_host_transpose_sample_tile=4;
constexpr int linear_host_angular_output_sample_tile=8;
// Keep each lane-major GEMM comfortably within the 32-bit matrix indexing
// used by vendor BLAS while bounding the shared transpose workspace. The
// common 5,000-node benchmark remains a single GEMM.
constexpr std::size_t linear_lane_gemm_sample_tile=8192;

template<typename Precision>
std::vector<Precision> tensor_values(const nlohmann::json& value) { return value.at("values").get<std::vector<Precision>>(); }
int shape_product(const std::vector<int>& shape) { return std::accumulate(shape.begin(),shape.end(),1,std::multiplies<int>()); }

class ProfileRegion {
public:
    explicit ProfileRegion(const char* name) { Kokkos::Profiling::pushRegion(name); }
    explicit ProfileRegion(const std::string& name) {
        Kokkos::Profiling::pushRegion(name.c_str());
    }
    ~ProfileRegion() { Kokkos::Profiling::popRegion(); }
};

struct PolarLinearSignature {
    const char* role;
    const char* irreps_in;
    const char* irreps_out;
};

bool is_validated_polar_linear_signature(
    const std::string& role,const std::string& irreps_in,
    const std::string& irreps_out)
{
    // This is deliberately an allow-list rather than a dimensional predicate:
    // packed GEMM happens to support other widths, but those shapes have not
    // been inventoried and qualified as production polar roles.
    static constexpr PolarLinearSignature signatures[]={
        {"feature_mixer","512x0e","512x0e"},
        {"feature_mixer","512x0e+512x1o","512x0e+512x1o"},
        {"feature_mixer","512x0e+512x1o+512x2e",
            "512x0e+512x1o+512x2e"},
        {"feature_mixer","512x0e+384x1o","512x0e+384x1o"},
        {"spin_source","512x0e","1x0e+1x1o+1x0e+1x1o"},
        {"spin_source","512x0e+512x1o","1x0e+1x1o+1x0e+1x1o"},
        {"spin_source","512x0e+512x1o+512x2e",
            "1x0e+1x1o+1x0e+1x1o"},
        {"fukui_first","512x0e","16x0e"},
        {"fukui_first","512x0e+512x1o","16x0e"},
        {"fukui_first","512x0e+512x1o+512x2e","16x0e"},
        {"fukui_mid","16x0e","16x0e"},
        {"fukui_final","16x0e","2x0e"},
        {"potential_embedding","2x0e+2x1o+2x0e+2x1o","512x0e"},
        {"potential_embedding","2x0e+2x1o+2x0e+2x1o",
            "512x0e+512x1o"},
        {"potential_embedding","2x0e+2x1o+2x0e+2x1o",
            "512x0e+512x1o+512x2e"},
        {"node_embedding","512x0e","512x0e"},
        {"node_embedding","512x0e+512x1o","512x0e+512x1o"},
        {"node_embedding","512x0e+512x1o+512x2e",
            "512x0e+512x1o+512x2e"},
        {"charge_embedding","1x0e+1x1o+1x0e+1x1o","512x0e"},
        {"charge_embedding","1x0e+1x1o+1x0e+1x1o",
            "512x0e+512x1o"},
        {"charge_embedding","1x0e+1x1o+1x0e+1x1o",
            "512x0e+512x1o+512x2e"},
        {"source_embedding","83x0e","512x0e"},
        {"update_readout_first","512x0e","64x0e+32x1o"},
        {"update_readout_first","512x0e+512x1o","64x0e+32x1o"},
        {"update_readout_first","512x0e+512x1o+512x2e",
            "64x0e+32x1o"},
        {"update_readout_mid","32x0e+32x1o","64x0e+32x1o"},
        {"update_readout_final","512x0e","10x0e"},
        {"update_readout_final","32x0e+32x1o",
            "1x0e+1x1o+1x0e+1x1o+2x0e"},
        {"local_charge_embedding","1x0e+1x1o+1x0e+1x1o","512x0e"},
        {"local_charge_embedding","1x0e+1x1o+1x0e+1x1o",
            "512x0e+512x1o"},
        {"local_charge_embedding","1x0e+1x1o+1x0e+1x1o",
            "512x0e+512x1o+512x2e"},
        {"local_potential_embedding","2x0e+2x1o+2x0e+2x1o","512x0e"},
        {"local_potential_embedding","2x0e+2x1o+2x0e+2x1o",
            "512x0e+512x1o"},
        {"local_potential_embedding","2x0e+2x1o+2x0e+2x1o",
            "512x0e+512x1o+512x2e"},
    };
    return std::any_of(std::begin(signatures),std::end(signatures),
        [&](const PolarLinearSignature& signature) {
            return role==signature.role&&irreps_in==signature.irreps_in
                &&irreps_out==signature.irreps_out;
        });
}

}

template<typename Precision>
E3LinearKokkosT<Precision>::E3LinearKokkosT(const nlohmann::json& data)
{
    E3Linear validated(data);
    Irreps input(data.at("irreps_in").get<std::string>()), output(data.at("irreps_out").get<std::string>());
    input_dimension_=input.dimension(); output_dimension_=output.dimension(); int offset=0;
    for(const auto& block:input.blocks)
        runtime_input_blocks.push_back(
            {block.offset,block.multiplicity,2*block.l+1});
    std::vector<int> input_block_plan;
    input_block_plan.reserve(3*runtime_input_blocks.size());
    for(const auto& block:runtime_input_blocks)
        input_block_plan.insert(input_block_plan.end(),
            {block.offset,block.multiplicity,block.width});
    set_kokkos_view(runtime_input_block_plan,std::move(input_block_plan),
        static_cast<int>(runtime_input_blocks.size()),3);
    for(const auto& block:output.blocks)
        runtime_output_blocks.push_back(
            {block.offset,block.multiplicity,2*block.l+1});
    std::vector<int> output_block_plan;
    output_block_plan.reserve(3*runtime_output_blocks.size());
    for(const auto& block:runtime_output_blocks)
        output_block_plan.insert(output_block_plan.end(),
            {block.offset,block.multiplicity,block.width});
    set_kokkos_view(runtime_output_block_plan,std::move(output_block_plan),
        static_cast<int>(runtime_output_blocks.size()),3);
    for (const auto& value:data.at("instructions")) {
        const auto& in=input.blocks.at(value.at("i_in").get<int>()); const auto& out=output.blocks.at(value.at("i_out").get<int>());
        const auto shape=value.at("path_shape").get<std::vector<int>>();
        instructions.push_back({in.offset,out.offset,in.multiplicity,
            out.multiplicity,2*in.l+1,offset,
            value.at("path_weight").get<Precision>()});
        offset+=shape_product(shape);
    }
    // Lane-major kernels address each instruction's block independently, so
    // they must see the unmerged instruction list.
    lane_instructions=instructions;
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        std::vector<Instruction> coalesced;
        coalesced.reserve(instructions.size());
        for(const auto& instruction:instructions) {
            if(!coalesced.empty()) {
                auto& previous=coalesced.back();
                const bool contiguous_input=instruction.input_offset
                    ==previous.input_offset
                        +previous.input_multiplicity*previous.width;
                const bool contiguous_weights=instruction.weight_offset
                    ==previous.weight_offset+previous.input_multiplicity
                        *previous.output_multiplicity;
                if(contiguous_input&&contiguous_weights
                    &&instruction.output_offset==previous.output_offset
                    &&instruction.output_multiplicity
                        ==previous.output_multiplicity
                    &&instruction.width==previous.width
                    &&instruction.path_weight==previous.path_weight) {
                    previous.input_multiplicity+=instruction.input_multiplicity;
                    continue;
                }
            }
            coalesced.push_back(instruction);
        }
        instructions=std::move(coalesced);
    }
    const auto assign_instruction_flags=[](std::vector<Instruction>& list) {
        for(std::size_t index=0;index<list.size();++index) {
            auto& instruction=list[index];
            instruction.first_input_instruction=std::none_of(
                list.begin(),list.begin()+index,
                [&](const auto& other) {
                    const int first=instruction.input_offset;
                    const int last=first
                        +instruction.input_multiplicity*instruction.width;
                    const int other_first=other.input_offset;
                    const int other_last=other_first
                        +other.input_multiplicity*other.width;
                    return first<other_last&&other_first<last;
                });
            instruction.first_output_instruction=std::none_of(
                list.begin(),list.begin()+index,
                [&](const auto& other) {
                    return other.output_offset==instruction.output_offset;
                });
            instruction.last_output_instruction=std::none_of(
                list.begin()+index+1,list.end(),
                [&](const auto& other) {
                    return other.output_offset==instruction.output_offset;
                });
        }
    };
    assign_instruction_flags(instructions);
    // The lane list is a separate copy: it needs the same flags whether or not
    // the node-major schedule was coalesced.
    assign_instruction_flags(lane_instructions);
    all_output_blocks_covered_=std::all_of(
        runtime_output_blocks.begin(),runtime_output_blocks.end(),
        [&](const auto& block) {
            return std::any_of(
                instructions.begin(),instructions.end(),
                [&](const auto& instruction) {
                    return instruction.output_offset==block.offset;
                });
        });
    all_input_blocks_covered_=std::all_of(
        input.blocks.begin(),input.blocks.end(),[&](const auto& block) {
            return std::any_of(
                instructions.begin(),instructions.end(),
                [&](const auto& instruction) {
                    const int block_width=2*block.l+1;
                    const int block_last=
                        block.offset+block.multiplicity*block_width;
                    const int instruction_last=instruction.input_offset
                        +instruction.input_multiplicity*instruction.width;
                    return instruction.width==block_width
                        &&instruction.input_offset<=block.offset
                        &&instruction_last>=block_last;
                });
        });
    weights=toKokkosView(
        "e3 linear weights",tensor_values<Precision>(data.at("weight")));
    const auto bias_values=tensor_values<Precision>(data.at("bias"));
    bias=toKokkosView("e3 linear bias",bias_values);
    const auto output_mask_values=
        tensor_values<Precision>(data.at("output_mask"));
    output_mask=toKokkosView("e3 linear output mask",output_mask_values);
    output_mask_is_identity_=std::all_of(
        output_mask_values.begin(),output_mask_values.end(),
        [](Precision value) { return value==Precision(1); });
    active_output_mask_is_identity_=std::all_of(
        instructions.begin(),instructions.end(),[&](const auto& instruction) {
            for(int target=0;target<instruction.output_multiplicity;++target)
                for(int component=0;component<instruction.width;++component)
                    if(output_mask_values[instruction.output_offset
                            +target*instruction.width+component]!=Precision(1))
                        return false;
            return true;
        });
    std::vector<int> active_output_values(output_dimension_,0);
    for(const auto& instruction:instructions)
        for(int target=0;target<instruction.output_multiplicity;++target)
            for(int component=0;component<instruction.width;++component)
                active_output_values[instruction.output_offset
                    +target*instruction.width+component]=1;
    active_output_mask=toKokkosView(
        "e3 linear active output mask",active_output_values);
    uncovered_output_finalization_is_zero_=true;
    for(int index=0;index<output_dimension_;++index)
        if(!active_output_values[index]
            &&(bias_values.empty()?Precision(0):bias_values[index])
                *output_mask_values[index]!=Precision(0)) {
            uncovered_output_finalization_is_zero_=false;
            break;
        }
    active_output_prefix_dimension_=0;
    while(active_output_prefix_dimension_<output_dimension_
        &&active_output_values[active_output_prefix_dimension_])
        ++active_output_prefix_dimension_;
    if(std::any_of(
            active_output_values.begin()+active_output_prefix_dimension_,
            active_output_values.end(),[](const int active) { return active!=0; }))
        active_output_prefix_dimension_=-1;
    if (weights.size()!=offset) throw std::invalid_argument("Kokkos e3 linear weight size is invalid.");
    const auto host_weights=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),weights);
    for(auto& instruction:instructions) {
        if(instruction.input_multiplicity!=instruction.output_multiplicity)
            continue;
        instruction.identity=true;
        for(int source=0;source<instruction.input_multiplicity;++source)
            for(int target=0;target<instruction.output_multiplicity;++target) {
                const Precision expected=source==target?Precision(1):Precision(0);
                if(instruction.path_weight*host_weights(
                        instruction.weight_offset
                            +source*instruction.output_multiplicity+target)
                    !=expected) {
                    instruction.identity=false;
                    break;
                }
            }
    }
    if(data.contains("execution_specialization")) {
        const auto& specialization=data.at("execution_specialization");
        specialization_requested_=specialization.value(
            "family",std::string())=="polar_e3linear";
        specialization_role_=specialization.value("role",std::string());
        const std::string irreps_in=data.at("irreps_in").get<std::string>();
        const std::string irreps_out=data.at("irreps_out").get<std::string>();
        if(specialization_requested_)
            polar_specialization_=is_validated_polar_linear_signature(
                specialization_role_,irreps_in,irreps_out);
    }
}

template<typename Precision>
std::vector<Precision> E3LinearKokkosT<Precision>::runtime_parameters_ir_mul() const
{
    const auto host_weights=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),weights);
    const auto host_bias=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),bias);
    const auto host_mask=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),output_mask);
    std::vector<Precision> result;
    result.reserve(weights.size()+bias.size()+output_mask.size());
    result.insert(result.end(),host_weights.data(),
        host_weights.data()+host_weights.size());
    auto append_ir_mul=[&](const auto& values) {
        for(const auto& block:runtime_output_blocks)
            for(int component=0;component<block.width;++component)
                for(int channel=0;channel<block.multiplicity;++channel)
                    result.push_back(values(block.offset
                        +channel*block.width+component));
    };
    if(host_bias.size()!=0) append_ir_mul(host_bias);
    append_ir_mul(host_mask);
    return result;
}

template<typename Precision>
bool E3LinearKokkosT<Precision>::has_instruction_width(const int width) const
{
    return std::any_of(
        instructions.begin(),instructions.end(),[&](const auto& instruction) {
            return instruction.width==width;
        });
}

template<typename Precision>
bool E3LinearKokkosT<Precision>::supports_identity_replacement(
    const int width) const
{
    if(bias.size()!=0)
        return false;
    const Instruction* selected=nullptr;
    for(const auto& instruction:instructions) {
        if(instruction.width!=width)
            continue;
        if(selected||instruction.input_multiplicity
                !=instruction.output_multiplicity)
            return false;
        selected=&instruction;
    }
    if(!selected||selected->path_weight==Precision(0))
        return false;
    const auto host_mask=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),output_mask);
    for(int channel=0;channel<selected->output_multiplicity;++channel)
        for(int component=0;component<width;++component)
            if(host_mask(selected->output_offset+channel*width+component)
                !=Precision(1))
                return false;
    return true;
}

template<typename Precision>
bool E3LinearKokkosT<Precision>::try_compose_input_block(
    const E3LinearKokkosT<Precision>& inner,const int width)
{
    if(bias.size()!=0||inner.bias.size()!=0)
        return false;
    Instruction* outer_instruction=nullptr;
    const Instruction* inner_instruction=nullptr;
    for(auto& instruction:instructions) {
        if(instruction.width!=width)
            continue;
        if(outer_instruction)
            return false;
        outer_instruction=&instruction;
    }
    for(const auto& instruction:inner.instructions) {
        if(instruction.width!=width)
            continue;
        if(inner_instruction)
            return false;
        inner_instruction=&instruction;
    }
    if(!outer_instruction||!inner_instruction
        ||inner.input_dimension_!=input_dimension_
        ||inner_instruction->input_offset!=outer_instruction->input_offset
        ||inner_instruction->output_offset!=outer_instruction->input_offset
        ||inner_instruction->input_multiplicity
            !=outer_instruction->input_multiplicity
        ||inner_instruction->output_multiplicity
            !=outer_instruction->input_multiplicity)
        return false;
    const auto inner_mask=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),inner.output_mask);
    for(int channel=0;channel<inner_instruction->output_multiplicity;++channel)
        for(int component=0;component<width;++component)
            if(inner_mask(inner_instruction->output_offset
                    +channel*width+component)!=Precision(1))
                return false;

    const auto inner_weights=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),inner.weights);
    auto outer_weights=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),weights);
    const int input_count=inner_instruction->input_multiplicity;
    const int middle_count=inner_instruction->output_multiplicity;
    const int output_count=outer_instruction->output_multiplicity;
    std::vector<Precision> composed(
        static_cast<std::size_t>(input_count)*output_count);
    for(int source=0;source<input_count;++source)
        for(int target=0;target<output_count;++target) {
            Precision value=Precision(0);
            for(int middle=0;middle<middle_count;++middle)
                value+=inner_instruction->path_weight*inner_weights(
                    inner_instruction->weight_offset
                        +source*middle_count+middle)
                    *outer_weights(outer_instruction->weight_offset
                        +middle*output_count+target);
            composed[static_cast<std::size_t>(source)*output_count+target]=value;
        }
    for(std::size_t index=0;index<composed.size();++index)
        outer_weights(outer_instruction->weight_offset+index)=composed[index];
    Kokkos::deep_copy(weights,outer_weights);
    return true;
}

template<typename Precision>
void E3LinearKokkosT<Precision>::replace_block_with_identity(const int width)
{
    if(!supports_identity_replacement(width))
        throw std::invalid_argument(
            "Kokkos e3 linear block cannot be replaced with identity.");
    auto instruction=std::find_if(
        instructions.begin(),instructions.end(),[&](const auto& value) {
            return value.width==width;
        });
    auto host_weights=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),weights);
    const Precision diagonal=Precision(1)/instruction->path_weight;
    for(int source=0;source<instruction->input_multiplicity;++source)
        for(int target=0;target<instruction->output_multiplicity;++target)
            host_weights(instruction->weight_offset
                +source*instruction->output_multiplicity+target)=
                source==target?diagonal:Precision(0);
    Kokkos::deep_copy(weights,host_weights);
    instruction->identity=true;
}

template<typename Precision>
bool E3LinearKokkosT<Precision>::supports_scalar_dense_plan() const
{
    if(instructions.size()!=1||runtime_input_blocks.empty()
        ||runtime_output_blocks.size()!=1)
        return false;
    const auto& instruction=instructions.front();
    return instruction.width==1&&instruction.input_offset==0
        &&instruction.output_offset==0&&instruction.weight_offset==0
        &&runtime_input_blocks.front().offset==0
        &&runtime_input_blocks.front().multiplicity
            ==instruction.input_multiplicity
        &&instruction.output_multiplicity==output_dimension_
        &&runtime_input_blocks.front().width==1
        &&runtime_output_blocks.front().width==1;
}

template<typename Precision>
typename E3LinearKokkosT<Precision>::ScalarDensePlan
E3LinearKokkosT<Precision>::scalar_dense_plan() const
{
    if(!supports_scalar_dense_plan())
        throw std::invalid_argument(
            "Kokkos E3 linear does not have a single scalar dense plan.");
    const auto& instruction=instructions.front();
    using const_view=Kokkos::View<const Precision*>;
    return {instruction.input_multiplicity,
        instruction.output_multiplicity,instruction.path_weight,
        const_view(weights),const_view(bias),const_view(output_mask)};
}

template<typename Precision>
void E3LinearKokkosT<Precision>::set_backend(const std::string& backend)
{
    if(backend=="auto") backend_=Backend::automatic;
    else if(backend=="scalar") backend_=Backend::scalar;
    else if(backend=="packed_gemm") backend_=Backend::packed_gemm;
    else throw std::invalid_argument(
        "Kokkos E3 linear backend must be auto, scalar, or packed_gemm.");
}

template<typename Precision>
std::string E3LinearKokkosT<Precision>::backend() const
{
    if(backend_==Backend::scalar) return "scalar";
    if(backend_==Backend::packed_gemm) return "packed_gemm";
    return "auto";
}

template<typename Precision>
bool E3LinearKokkosT<Precision>::use_scalar_backend(std::size_t samples) const
{
    if(backend_==Backend::scalar) return true;
    if(backend_==Backend::packed_gemm) return false;
#ifdef KOKKOS_ENABLE_CUDA
    return true;
#else
    return samples<=4;
#endif
}

template<typename Precision>
std::string E3LinearKokkosT<Precision>::selected_backend(
    std::size_t samples) const
{
    return use_scalar_backend(samples) ? "scalar" : "packed_gemm";
}

template<typename Precision>
std::size_t E3LinearKokkosT<Precision>::workspace_bytes() const
{
    return workspace_->bytes();
}

template<typename Precision>
typename E3LinearKokkosT<Precision>::WorkspaceShape
E3LinearKokkosT<Precision>::lane_workspace_shape(const std::size_t samples) const
{
    WorkspaceShape shape;
    if(use_scalar_backend(samples)) return shape;
    for(const auto& instruction:lane_instructions) {
        const std::size_t columns=std::min(
            samples,linear_lane_gemm_sample_tile)
            *static_cast<std::size_t>(instruction.width);
        shape.input_rows=std::max(shape.input_rows,
            static_cast<std::size_t>(instruction.input_multiplicity));
        shape.output_rows=std::max(shape.output_rows,
            static_cast<std::size_t>(instruction.output_multiplicity));
        shape.input_columns=std::max(shape.input_columns,columns);
        shape.output_columns=std::max(shape.output_columns,columns);
    }
    return shape;
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(input.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument(
            "Kokkos e3 linear batch dimensions are inconsistent.");
    if(backend_==Backend::automatic&&polar_specialization_) {
        ++specialized_forward_count_;
        // CPU keeps its qualified small-batch scalar threshold; the CUDA
        // specialization is the route that replaces the always-scalar CUDA
        // schedule with the persistent packed-GEMM workspace.
#ifdef KOKKOS_ENABLE_CUDA
        evaluate_impl(input,output,false);
#else
        evaluate_impl(input,output,use_scalar_backend(input.extent(0)));
#endif
    } else {
        if(backend_==Backend::automatic&&specialization_requested_)
            ++specialization_fallback_count_;
        evaluate_impl(input,output,use_scalar_backend(input.extent(0)));
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate_generic(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    evaluate_impl(input,output,true);
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate_impl(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision**,Kokkos::LayoutRight> output,const bool use_scalar_path) const
{
    if(input.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument("Kokkos e3 linear batch dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/forward");
    auto all_weights=weights;
    if(use_scalar_path) {
        ordered_kokkos_deep_copy(output,Precision(0));
        for(const auto instruction:instructions)
            Kokkos::parallel_for(
                "e3 linear small batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{static_cast<int>(input.extent(0)),
                        instruction.output_multiplicity,instruction.width}),
                KOKKOS_LAMBDA(int sample,int target,int component) {
                    Precision value=Precision(0);
                    for(int source=0;source<instruction.input_multiplicity;++source)
                        value+=instruction.path_weight
                            *all_weights(instruction.weight_offset
                                +source*instruction.output_multiplicity+target)
                            *input(sample,instruction.input_offset
                                +source*instruction.width+component);
                    output(sample,instruction.output_offset
                        +target*instruction.width+component)+=value;
                });
        auto mask=output_mask; auto all_bias=bias;
        Kokkos::parallel_for("e3 linear mask",output.size(),KOKKOS_LAMBDA(std::size_t flat) {
            const std::size_t sample=flat/output.extent(1);
            const int column=flat%output.extent(1);
            output(sample,column)=(output(sample,column)
                +(all_bias.size()?all_bias(column):Precision(0)))*mask(column);
        });
        return;
    }
    if(!all_output_blocks_covered_)
        ordered_kokkos_deep_copy(output,Precision(0));
    const bool fuse_output_finalization=all_output_blocks_covered_
        ||uncovered_output_finalization_is_zero_;
    for (const auto& instruction:instructions) {
        const int samples=input.extent(0);
        if(instruction.identity) {
            auto mask=output_mask;
            auto all_bias=bias;
            const bool first=instruction.first_output_instruction;
            const bool last=instruction.last_output_instruction;
            const bool fuse_finalization=fuse_output_finalization;
            if constexpr(std::is_same_v<
                typename Kokkos::DefaultExecutionSpace::memory_space,
                Kokkos::HostSpace>) {
                const int block_size=
                    instruction.output_multiplicity*instruction.width;
                const int input_offset=instruction.input_offset;
                const int output_offset=instruction.output_offset;
                Kokkos::parallel_for(
                    "e3 linear identity",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        0,samples),
                    KOKKOS_LAMBDA(const int sample) {
                        for(int local=0;local<block_size;++local) {
                            const int input_index=input_offset+local;
                            const int output_index=output_offset+local;
                            Precision value=input(sample,input_index);
                            if(!fuse_finalization||!first)
                                value+=output(sample,output_index);
                            if(fuse_finalization&&last)
                                value=(value+(all_bias.size()
                                    ?all_bias(output_index):Precision(0)))
                                    *mask(output_index);
                            output(sample,output_index)=value;
                        }
                    });
            } else
                Kokkos::parallel_for(
                    "e3 linear identity",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                        {0,0,0},{instruction.output_multiplicity,samples,
                            instruction.width}),
                    KOKKOS_LAMBDA(int channel,int sample,int component) {
                        const int input_index=instruction.input_offset
                            +channel*instruction.width+component;
                        const int output_index=instruction.output_offset
                            +channel*instruction.width+component;
                        Precision value=input(sample,input_index);
                        if(!fuse_finalization||!first)
                            value+=output(sample,output_index);
                        if(fuse_finalization&&last)
                            value=(value+(all_bias.size()
                                ?all_bias(output_index):Precision(0)))
                                *mask(output_index);
                        output(sample,output_index)=value;
                    });
            continue;
        }
        const int columns=samples*instruction.width;
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(
                workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max<std::size_t>(workspace_->packed_input.extent(1),columns));
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(
                workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_input=Kokkos::subview(
            workspace_->packed_input,
            std::make_pair(0,instruction.input_multiplicity),
            std::make_pair(0,columns));
        auto packed_output=Kokkos::subview(
            workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            const int channel_tiles=(instruction.input_multiplicity
                +linear_host_transpose_channel_tile-1)
                /linear_host_transpose_channel_tile;
            const int sample_tiles=(samples+linear_host_transpose_sample_tile-1)
                /linear_host_transpose_sample_tile;
            const int input_offset=instruction.input_offset;
            const int input_multiplicity=instruction.input_multiplicity;
            const int width=instruction.width;
            Kokkos::parallel_for("e3 linear pack blocked",
                channel_tiles*sample_tiles,KOKKOS_LAMBDA(int tile) {
                    const int channel_first=(tile%channel_tiles)
                        *linear_host_transpose_channel_tile;
                    const int sample_first=(tile/channel_tiles)
                        *linear_host_transpose_sample_tile;
                    const int channel_last=channel_first
                            +linear_host_transpose_channel_tile<input_multiplicity
                        ?channel_first+linear_host_transpose_channel_tile
                        :input_multiplicity;
                    const int sample_last=sample_first
                            +linear_host_transpose_sample_tile<samples
                        ?sample_first+linear_host_transpose_sample_tile:samples;
                    for(int component=0;component<width;++component)
                        for(int channel=channel_first;channel<channel_last;
                            ++channel)
                            for(int sample=sample_first;sample<sample_last;
                                ++sample)
                                packed_input(channel,sample*width+component)=
                                    input(sample,input_offset
                                        +channel*width+component);
                });
        } else
            Kokkos::parallel_for(
                "e3 linear pack",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{instruction.input_multiplicity,samples,
                        instruction.width}),
                KOKKOS_LAMBDA(int channel,int sample,int component) {
                    packed_input(channel,sample*instruction.width+component)=
                        input(sample,instruction.input_offset
                            +channel*instruction.width+component);
                });
        using weight_matrix=Kokkos::View<
            const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        weight_matrix weight(
            all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm("T","N",instruction.path_weight,weight,packed_input,
            Precision(0),packed_output);
        auto mask=output_mask;
        auto all_bias=bias;
        const bool first=instruction.first_output_instruction;
        const bool last=instruction.last_output_instruction;
        const bool fuse_finalization=fuse_output_finalization;
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            const int output_multiplicity=instruction.output_multiplicity;
            const int width=instruction.width;
            const int output_offset=instruction.output_offset;
            const int channel_tiles=(output_multiplicity
                +linear_host_transpose_channel_tile-1)
                /linear_host_transpose_channel_tile;
            const int sample_tiles=(samples+linear_host_transpose_sample_tile-1)
                /linear_host_transpose_sample_tile;
            Kokkos::parallel_for("e3 linear unpack blocked",
                channel_tiles*sample_tiles,KOKKOS_LAMBDA(int tile) {
                    const int channel_first=(tile%channel_tiles)
                        *linear_host_transpose_channel_tile;
                    const int sample_first=(tile/channel_tiles)
                        *linear_host_transpose_sample_tile;
                    const int channel_last=channel_first
                            +linear_host_transpose_channel_tile
                            <output_multiplicity
                        ?channel_first+linear_host_transpose_channel_tile
                        :output_multiplicity;
                    const int sample_last=sample_first
                            +linear_host_transpose_sample_tile<samples
                        ?sample_first+linear_host_transpose_sample_tile:samples;
                    for(int component=0;component<width;++component)
                        for(int channel=channel_first;channel<channel_last;
                            ++channel) {
                            const int output_index=output_offset
                                +channel*width+component;
                            for(int sample=sample_first;sample<sample_last;
                                ++sample) {
                                Precision value=packed_output(
                                    channel,sample*width+component);
                                if(!fuse_finalization||!first)
                                    value+=output(sample,output_index);
                                if(fuse_finalization&&last)
                                    value=(value+(all_bias.size()
                                        ?all_bias(output_index):Precision(0)))
                                        *mask(output_index);
                                output(sample,output_index)=value;
                            }
                        }
                });
        } else Kokkos::parallel_for(
                "e3 linear unpack",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{instruction.output_multiplicity,samples,
                        instruction.width}),
                KOKKOS_LAMBDA(int channel,int sample,int component) {
                    const int output_index=instruction.output_offset
                        +channel*instruction.width+component;
                    Precision value=packed_output(
                        channel,sample*instruction.width+component);
                    if(!fuse_finalization||!first)
                        value+=output(sample,output_index);
                    if(fuse_finalization&&last)
                        value=(value+(all_bias.size()
                            ?all_bias(output_index):Precision(0)))
                            *mask(output_index);
                    output(sample,output_index)=value;
                });
    }
    if(!fuse_output_finalization) {
        auto mask=output_mask;
        auto all_bias=bias;
        Kokkos::parallel_for(
            "e3 linear mask",output.size(),KOKKOS_LAMBDA(std::size_t flat) {
                const std::size_t sample=flat/output.extent(1);
                const int column=flat%output.extent(1);
                output(sample,column)=(output(sample,column)
                    +(all_bias.size()?all_bias(column):Precision(0)))
                    *mask(column);
            });
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate_lane(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(input.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument(
            "Kokkos e3 linear lane batch dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/forward_lane");
    auto all_weights=weights;
    const std::size_t samples=input.extent(0);
    if(use_scalar_backend(input.extent(0))) {
        ordered_kokkos_deep_copy(output,Precision(0));
        for(const auto instruction:lane_instructions)
            Kokkos::parallel_for(
                "e3 linear lane small batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                    Kokkos::IndexType<std::size_t>>(
                    {std::size_t(0),std::size_t(0),std::size_t(0)},
                    {samples,
                        static_cast<std::size_t>(instruction.output_multiplicity),
                        static_cast<std::size_t>(instruction.width)}),
                KOKKOS_LAMBDA(std::size_t sample,std::size_t target,
                    std::size_t component) {
                    Precision value=Precision(0);
                    const int input_base=instruction.input_offset
                        +component*instruction.input_multiplicity;
                    for(int source=0;source<instruction.input_multiplicity;++source)
                        value+=instruction.path_weight
                            *all_weights(instruction.weight_offset
                                +source*instruction.output_multiplicity+target)
                            *input(sample,input_base+source);
                    output(sample,instruction.output_offset
                        +component*instruction.output_multiplicity+target)+=value;
                });
        finalize_lane_output(output);
        return;
    }
    if(!all_output_blocks_covered_)
        ordered_kokkos_deep_copy(output,Precision(0));
    const bool fuse_output_finalization=all_output_blocks_covered_
        ||uncovered_output_finalization_is_zero_;
    auto mask=output_mask; auto all_bias=bias;
    using weight_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:lane_instructions) {
        const std::size_t workspace_samples=std::min(
            samples,linear_lane_gemm_sample_tile);
        const std::size_t workspace_columns=workspace_samples
            *static_cast<std::size_t>(instruction.width);
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<workspace_columns) {
            if(workspace_->fixed_capacity)
                throw std::logic_error("fixed e3 linear input workspace is too small");
            Kokkos::realloc(workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max(workspace_->packed_input.extent(1),workspace_columns));
        }
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<workspace_columns) {
            if(workspace_->fixed_capacity)
                throw std::logic_error("fixed e3 linear output workspace is too small");
            Kokkos::realloc(workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max(workspace_->packed_output.extent(1),workspace_columns));
        }
        weight_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        const bool first=instruction.first_output_instruction;
        const bool last=instruction.last_output_instruction;
        const bool fuse_finalization=fuse_output_finalization;
        for(std::size_t sample_begin=0;sample_begin<samples;
            sample_begin+=linear_lane_gemm_sample_tile) {
            const std::size_t active_samples=std::min(
                samples-sample_begin,linear_lane_gemm_sample_tile);
            const std::size_t columns=active_samples
                *static_cast<std::size_t>(instruction.width);
            auto packed_input=Kokkos::subview(workspace_->packed_input,
                std::make_pair(std::size_t(0),
                    static_cast<std::size_t>(instruction.input_multiplicity)),
                std::make_pair(std::size_t(0),columns));
            auto packed_output=Kokkos::subview(workspace_->packed_output,
                std::make_pair(std::size_t(0),
                    static_cast<std::size_t>(instruction.output_multiplicity)),
                std::make_pair(std::size_t(0),columns));
            Kokkos::parallel_for(
                "e3 linear lane pack",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                    Kokkos::IndexType<std::size_t>>(
                    {std::size_t(0),std::size_t(0),std::size_t(0)},
                    {static_cast<std::size_t>(instruction.input_multiplicity),
                        active_samples,
                        static_cast<std::size_t>(instruction.width)}),
                KOKKOS_LAMBDA(std::size_t channel,std::size_t sample,
                    std::size_t component) {
                    packed_input(channel,sample*instruction.width+component)=
                        input(sample_begin+sample,instruction.input_offset
                            +component*instruction.input_multiplicity+channel);
                });
            KokkosBlas::gemm("T","N",instruction.path_weight,weight,
                packed_input,Precision(0),packed_output);
            Kokkos::parallel_for(
                "e3 linear lane unpack",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                    Kokkos::IndexType<std::size_t>>(
                    {std::size_t(0),std::size_t(0),std::size_t(0)},
                    {static_cast<std::size_t>(instruction.output_multiplicity),
                        active_samples,
                        static_cast<std::size_t>(instruction.width)}),
                KOKKOS_LAMBDA(std::size_t channel,std::size_t sample,
                    std::size_t component) {
                    const std::size_t output_index=instruction.output_offset
                        +component*instruction.output_multiplicity+channel;
                    // bias and output_mask keep their definition (node-major)
                    // order, so they are indexed at the node-major position
                    // of this lane-major element.
                    const std::size_t table_index=instruction.output_offset
                        +channel*instruction.width+component;
                    Precision value=packed_output(
                        channel,sample*instruction.width+component);
                    if(!fuse_finalization||!first)
                        value+=output(sample_begin+sample,output_index);
                    if(fuse_finalization&&last)
                        value=(value+(all_bias.size()
                            ?all_bias(table_index):Precision(0)))
                            *mask(table_index);
                    output(sample_begin+sample,output_index)=value;
                });
        }
    }
    if(!fuse_output_finalization)
        finalize_lane_output(output);
}

template<typename Precision>
void E3LinearKokkosT<Precision>::finalize_lane_output(
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    auto all_bias=bias; auto mask=output_mask;
    auto plan=runtime_output_block_plan;
    const std::size_t samples=output.extent(0);
    const std::size_t blocks=plan.extent(0);
    Kokkos::parallel_for(
        "e3 linear lane finalize",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>,
            Kokkos::IndexType<std::size_t>>({0,0},{samples,blocks}),
        KOKKOS_LAMBDA(const std::size_t sample,const std::size_t block) {
            const int offset=plan(block,0);
            const int multiplicity=plan(block,1);
            const int width=plan(block,2);
            for(int channel=0;channel<multiplicity;++channel)
                for(int component=0;component<width;++component) {
                    const int table_index=offset+channel*width+component;
                    const int lane_index=offset+component*multiplicity
                        +channel;
                    output(sample,lane_index)=(output(sample,lane_index)
                        +(all_bias.size()?all_bias(table_index)
                            :Precision(0)))*mask(table_index);
                }
        });
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate_to_packed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<Precision**,Kokkos::LayoutRight> packed_output,
    const bool initialize_uncovered,
    Kokkos::View<const Precision*> row_denominator,
    const Precision denominator_alpha,
    const Precision denominator_beta) const
{
    if(input.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||packed_output.size()!=input.extent(0)
            *static_cast<std::size_t>(output_dimension_)
        ||(row_denominator.size()
            &&row_denominator.extent(0)!=input.extent(0)))
        throw std::invalid_argument(
            "Kokkos e3 linear packed output dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/forward_to_packed");
    auto all_weights=weights;
    const int samples=input.extent(0);
    if(initialize_uncovered&&!all_output_blocks_covered_)
        ordered_kokkos_deep_copy(packed_output,Precision(0));
    using matrix=Kokkos::View<
        Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    using const_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:instructions) {
        const int columns=samples*instruction.width;
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max<std::size_t>(workspace_->packed_input.extent(1),columns));
        auto packed_input=Kokkos::subview(workspace_->packed_input,
            std::make_pair(0,instruction.input_multiplicity),
            std::make_pair(0,columns));
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            const int channel_tiles=(instruction.input_multiplicity
                +linear_host_transpose_channel_tile-1)
                /linear_host_transpose_channel_tile;
            const int sample_tiles=(samples+linear_host_transpose_sample_tile-1)
                /linear_host_transpose_sample_tile;
            const int input_offset=instruction.input_offset;
            const int input_multiplicity=instruction.input_multiplicity;
            const int width=instruction.width;
            Kokkos::parallel_for(
                "e3 linear pack input blocked",channel_tiles*sample_tiles,
                KOKKOS_LAMBDA(int tile) {
                    const int channel_first=(tile%channel_tiles)
                        *linear_host_transpose_channel_tile;
                    const int sample_first=(tile/channel_tiles)
                        *linear_host_transpose_sample_tile;
                    const int channel_last=channel_first
                            +linear_host_transpose_channel_tile<input_multiplicity
                        ?channel_first+linear_host_transpose_channel_tile
                        :input_multiplicity;
                    const int sample_last=sample_first
                            +linear_host_transpose_sample_tile<samples
                        ?sample_first+linear_host_transpose_sample_tile:samples;
                    for(int component=0;component<width;++component)
                        for(int channel=channel_first;channel<channel_last;
                            ++channel)
                            for(int sample=sample_first;sample<sample_last;
                                ++sample)
                                packed_input(channel,sample*width+component)=
                                    input(sample,input_offset
                                        +channel*width+component)
                                    /(row_denominator.size()
                                        ?denominator_alpha+denominator_beta
                                            *row_denominator(sample)
                                        :Precision(1));
                });
        } else
            Kokkos::parallel_for("e3 linear pack input",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{instruction.input_multiplicity,samples,
                        instruction.width}),
                KOKKOS_LAMBDA(int channel,int sample,int component) {
                    packed_input(channel,sample*instruction.width+component)=
                        input(sample,instruction.input_offset
                            +channel*instruction.width+component)
                        /(row_denominator.size()
                            ?denominator_alpha+denominator_beta
                                *row_denominator(sample)
                            :Precision(1));
                });
        matrix destination(packed_output.data()
                +static_cast<std::size_t>(samples)*instruction.output_offset,
            instruction.output_multiplicity,columns);
        const_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm("T","N",instruction.path_weight,weight,packed_input,
            instruction.first_output_instruction?Precision(0):Precision(1),
            destination);
    }
    auto all_bias=bias;
    auto mask=output_mask;
    if(!all_bias.size()&&(initialize_uncovered
            ?output_mask_is_identity_:active_output_mask_is_identity_))
        return;
    if(initialize_uncovered) for(const auto block:runtime_output_blocks) {
        const int offset=block.offset;
        const int multiplicity=block.multiplicity;
        const int width=block.width;
        matrix destination(packed_output.data()
                +static_cast<std::size_t>(samples)*offset,
            multiplicity,samples*width);
        Kokkos::parallel_for("e3 linear finalize packed output",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{multiplicity,samples,width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                const int index=offset+channel*width+component;
                destination(channel,sample*width+component)=
                    (destination(channel,sample*width+component)
                        +(all_bias.size()?all_bias(index):Precision(0)))*mask(index);
            });
    } else for(const auto instruction:instructions) {
        if(!instruction.last_output_instruction)
            continue;
        const int offset=instruction.output_offset;
        const int multiplicity=instruction.output_multiplicity;
        const int width=instruction.width;
        matrix destination(packed_output.data()
                +static_cast<std::size_t>(samples)*offset,
            multiplicity,samples*width);
        Kokkos::parallel_for("e3 linear finalize active packed output",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{multiplicity,samples,width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                const int index=offset+channel*width+component;
                destination(channel,sample*width+component)=
                    (destination(channel,sample*width+component)
                        +(all_bias.size()?all_bias(index):Precision(0)))*mask(index);
            });
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::normalize_packed_rows(
    Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input,
    Kokkos::View<const Precision*> row_denominator,
    const Precision denominator_alpha,const Precision denominator_beta) const
{
    const std::size_t samples=row_denominator.extent(0);
    if(packed_input.size()!=samples*static_cast<std::size_t>(input_dimension_)
        ||!all_input_blocks_covered_)
        throw std::invalid_argument(
            "Kokkos packed input normalization dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/normalize_packed_rows");
    auto* const packed=packed_input.data();
    auto blocks=runtime_input_block_plan;
    const int block_count=blocks.extent(0);
    Kokkos::parallel_for("e3 linear normalize packed rows",samples,
        KOKKOS_LAMBDA(int sample) {
            const Precision inverse=Precision(1)
                /(denominator_alpha
                    +denominator_beta*row_denominator(sample));
            for(int block=0;block<block_count;++block) {
                const int input_offset=blocks(block,0);
                const int multiplicity=blocks(block,1);
                const int width=blocks(block,2);
                const std::size_t packed_block=samples*input_offset;
                for(int channel=0;channel<multiplicity;++channel)
                    for(int component=0;component<width;++component) {
                        const std::size_t index=packed_block
                            +static_cast<std::size_t>(channel)*samples*width
                            +sample*width+component;
                        packed[index]*=inverse;
                    }
            }
        });
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_normalize_packed_rows(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input_adjoint,
    Kokkos::View<const Precision*> row_denominator,
    const Precision denominator_alpha,const Precision denominator_beta,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_ir_mul_adjoint,
    Kokkos::View<Precision*> row_denominator_adjoint,
    const bool packed_input_is_normalized) const
{
    const std::size_t samples=row_denominator.extent(0);
    if(packed_input.size()!=samples*static_cast<std::size_t>(input_dimension_)
        ||packed_input_adjoint.size()!=packed_input.size()
        ||input_ir_mul_adjoint.extent(0)!=samples
        ||input_ir_mul_adjoint.extent(1)
            !=static_cast<std::size_t>(input_dimension_)
        ||row_denominator_adjoint.extent(0)!=samples
        ||!all_input_blocks_covered_)
        throw std::invalid_argument(
            "Kokkos packed input normalization reverse dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_
            +"/reverse_normalize_packed_rows");
    ordered_kokkos_deep_copy(row_denominator_adjoint,Precision(0));
    const auto* const packed=packed_input.data();
    const auto* const packed_adjoint=packed_input_adjoint.data();
    auto* const ir_mul_adjoint=input_ir_mul_adjoint.data();
    const int input_dimension=input_dimension_;
    for(const auto& block:runtime_input_blocks) {
        const int input_offset=block.offset;
        const int multiplicity=block.multiplicity;
        const int width=block.width;
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            // A small sample tile makes packed feature reads cache-friendly
            // while each SIMD lane writes a distinct native sample row.
            constexpr std::size_t sample_tile=8;
            const std::size_t sample_tiles=
                (samples+sample_tile-1)/sample_tile;
            Kokkos::parallel_for(
                "e3 linear reverse normalize packed rows blocked",sample_tiles,
                KOKKOS_LAMBDA(const std::size_t tile) {
                    const std::size_t sample_first=tile*sample_tile;
                    const std::size_t active_samples=sample_first+sample_tile
                            <samples
                        ?sample_tile:samples-sample_first;
                    Precision inverses[sample_tile];
                    Precision density_scales[sample_tile];
                    Precision denominator_values[sample_tile];
#if defined(_OPENMP)
#pragma omp simd
#endif
                    for(std::size_t lane=0;lane<active_samples;++lane) {
                        const std::size_t sample=sample_first+lane;
                        const Precision denominator=denominator_alpha
                            +denominator_beta*row_denominator(sample);
                        const Precision inverse=Precision(1)/denominator;
                        inverses[lane]=inverse;
                        density_scales[lane]=denominator_beta*inverse
                            *(packed_input_is_normalized
                                ?Precision(1):inverse);
                        denominator_values[lane]=
                            row_denominator_adjoint(sample);
                    }
                    const std::size_t packed_block=samples*input_offset;
                    for(int channel=0;channel<multiplicity;++channel)
                        for(int component=0;component<width;++component) {
                            const std::size_t packed_channel=packed_block
                                +static_cast<std::size_t>(channel)*samples*width
                                +sample_first*width+component;
#if defined(_OPENMP)
#pragma omp simd
#endif
                            for(std::size_t lane=0;lane<active_samples;++lane) {
                                const std::size_t sample=sample_first+lane;
                                const std::size_t packed_index=
                                    packed_channel+lane*width;
                                const Precision adjoint=
                                    packed_adjoint[packed_index];
                                ir_mul_adjoint[sample*input_dimension
                                    +input_offset+component*multiplicity+channel]=
                                    adjoint*inverses[lane];
                                denominator_values[lane]-=
                                    density_scales[lane]*adjoint
                                    *packed[packed_index];
                            }
                        }
#if defined(_OPENMP)
#pragma omp simd
#endif
                    for(std::size_t lane=0;lane<active_samples;++lane)
                        row_denominator_adjoint(sample_first+lane)=
                            denominator_values[lane];
                });
        } else Kokkos::parallel_for(
            "e3 linear reverse normalize packed rows",samples,
            KOKKOS_LAMBDA(int sample) {
                const Precision denominator=denominator_alpha
                    +denominator_beta*row_denominator(sample);
                const Precision inverse=Precision(1)/denominator;
                const Precision density_scale=denominator_beta*inverse
                    *(packed_input_is_normalized?Precision(1):inverse);
                Precision denominator_value=row_denominator_adjoint(sample);
                const std::size_t packed_block=samples*input_offset;
                for(int channel=0;channel<multiplicity;++channel)
                    for(int component=0;component<width;++component) {
                        const std::size_t packed_index=packed_block
                            +static_cast<std::size_t>(channel)*samples*width
                            +sample*width+component;
                        const Precision adjoint=packed_adjoint[packed_index];
                        ir_mul_adjoint[
                            static_cast<std::size_t>(sample)*input_dimension
                                +input_offset+component*multiplicity+channel]=
                            adjoint*inverse;
                        denominator_value-=density_scale*adjoint
                            *packed[packed_index];
                    }
                row_denominator_adjoint(sample)=denominator_value;
            });
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate_packed_to_packed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input,
    Kokkos::View<Precision**,Kokkos::LayoutRight> packed_output) const
{
    const std::size_t samples=packed_output.extent(0);
    if(packed_input.size()!=samples*static_cast<std::size_t>(input_dimension_)
        ||packed_output.size()!=samples*static_cast<std::size_t>(output_dimension_)
        ||!all_input_blocks_covered_||!all_output_blocks_covered_
        ||!output_mask_is_identity_||bias.size()!=0)
        throw std::invalid_argument(
            "Kokkos packed-to-packed linear dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/forward_packed_to_packed");
    auto all_weights=weights;
    using matrix=Kokkos::View<
        Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    using const_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:instructions) {
        const int columns=static_cast<int>(samples)*instruction.width;
        const_matrix source(packed_input.data()
                +samples*instruction.input_offset,
            instruction.input_multiplicity,columns);
        const_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        matrix destination(packed_output.data()
                +samples*instruction.output_offset,
            instruction.output_multiplicity,columns);
        KokkosBlas::gemm("T","N",instruction.path_weight,weight,source,
            instruction.first_output_instruction?Precision(0):Precision(1),
            destination);
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate_from_packed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(packed_input.size()!=output.extent(0)
            *static_cast<std::size_t>(input_dimension_)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||!all_output_blocks_covered_)
        throw std::invalid_argument(
            "Kokkos e3 linear packed input dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/forward_from_packed");
    ordered_kokkos_deep_copy(output,Precision(0));
    auto all_weights=weights;
    const int samples=output.extent(0);
    using matrix=Kokkos::View<
        Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    using const_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:instructions) {
        const int columns=samples*instruction.width;
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_result=Kokkos::subview(workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
        const_matrix source(packed_input.data()
                +static_cast<std::size_t>(samples)*instruction.input_offset,
            instruction.input_multiplicity,columns);
        const_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm("T","N",instruction.path_weight,weight,source,
            Precision(0),packed_result);
        auto all_bias=bias;
        auto mask=output_mask;
        Kokkos::parallel_for("e3 linear unpack output",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{instruction.output_multiplicity,samples,
                    instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                const int index=instruction.output_offset
                    +channel*instruction.width+component;
                Precision value=packed_result(
                    channel,sample*instruction.width+component);
                if(!instruction.first_output_instruction)
                    value+=output(sample,index);
                if(instruction.last_output_instruction)
                    value=(value+(all_bias.size()
                        ?all_bias(index):Precision(0)))*mask(index);
                output(sample,index)=value;
            });
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate_from_packed_to_block_major(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_input,
    Kokkos::View<Precision*> block_major_output,const int samples) const
{
    if(samples<0
        ||packed_input.size()!=static_cast<std::size_t>(samples)*input_dimension_
        ||block_major_output.size()
            !=static_cast<std::size_t>(samples)*output_dimension_
        ||!all_output_blocks_covered_||bias.size()!=0
        ||!output_mask_is_identity_)
        throw std::invalid_argument(
            "Kokkos block-major packed forward dimensions are inconsistent.");
    if constexpr(!std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>)
        throw std::invalid_argument(
            "Kokkos block-major packed forward is host-only.");
    else {
        ProfileRegion profile(
            "symmetrix/e3_linear/"+profile_name_
                +"/forward_from_packed_to_block_major");
        auto all_weights=weights;
        for(const auto& instruction:instructions) {
            int angular_offset=0;
            bool matched=false;
            for(const auto& block:runtime_output_blocks) {
                if(block.offset==instruction.output_offset) {
                    if(block.multiplicity!=instruction.output_multiplicity
                        ||block.width!=instruction.width)
                        throw std::invalid_argument(
                            "Kokkos block-major forward block is inconsistent.");
                    matched=true;
                    break;
                }
                angular_offset+=block.width;
            }
            if(!matched)
                throw std::invalid_argument(
                    "Kokkos block-major forward block is unavailable.");
            const int columns=samples*instruction.width;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,CblasTrans,CblasNoTrans,
                columns,instruction.output_multiplicity,
                instruction.input_multiplicity,instruction.path_weight,
                packed_input.data()+static_cast<std::size_t>(samples)
                    *instruction.input_offset,
                columns,all_weights.data()+instruction.weight_offset,
                instruction.output_multiplicity,
                instruction.first_output_instruction
                    ?Precision(0):Precision(1),
                block_major_output.data()
                    +static_cast<std::size_t>(samples)*angular_offset
                        *instruction.output_multiplicity,
                instruction.output_multiplicity);
        }
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate_ir_mul(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(input.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument(
            "Kokkos ir-mul e3 linear batch dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/forward_ir_mul");
    ordered_kokkos_deep_copy(output,Precision(0));
    auto all_weights=weights;
    for(const auto& instruction:instructions) {
        const int samples=input.extent(0);
        const int columns=samples*instruction.width;
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max<std::size_t>(workspace_->packed_input.extent(1),columns));
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_input=Kokkos::subview(workspace_->packed_input,
            std::make_pair(0,instruction.input_multiplicity),
            std::make_pair(0,columns));
        auto packed_output=Kokkos::subview(workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
        Kokkos::parallel_for("e3 ir-mul linear pack",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
                {instruction.input_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                packed_input(channel,sample*instruction.width+component)=
                    input(sample,instruction.input_offset
                        +component*instruction.input_multiplicity+channel);
            });
        using weight_matrix=Kokkos::View<
            const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        weight_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm("T","N",instruction.path_weight,weight,packed_input,
            Precision(0),packed_output);
        Kokkos::parallel_for("e3 ir-mul linear unpack",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
                {instruction.output_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                output(sample,instruction.output_offset
                    +component*instruction.output_multiplicity+channel)
                    +=packed_output(channel,sample*instruction.width+component);
            });
    }
    auto mask=output_mask;
    auto all_bias=bias;
    for(const auto block:runtime_output_blocks) {
        const int offset=block.offset;
        const int multiplicity=block.multiplicity;
        const int width=block.width;
        Kokkos::parallel_for("e3 ir-mul linear mask",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
                {static_cast<int>(output.extent(0)),width,multiplicity}),
            KOKKOS_LAMBDA(int sample,int component,int channel) {
                const int ir_mul=offset+component*multiplicity+channel;
                const int mul_ir=offset+channel*width+component;
                output(sample,ir_mul)=(output(sample,ir_mul)
                    +(all_bias.size()?all_bias(mul_ir):Precision(0)))
                    *mask(mul_ir);
            });
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
    const bool initialize_input_adjoint) const
{
    if(output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||input_adjoint.extent(0)!=output_adjoint.extent(0)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(input_dimension_))
        throw std::invalid_argument(
            "Kokkos e3 linear reverse batch dimensions are inconsistent.");
    if(backend_==Backend::automatic&&polar_specialization_) {
        ++specialized_reverse_count_;
#ifdef KOKKOS_ENABLE_CUDA
        reverse_impl(output_adjoint,input_adjoint,initialize_input_adjoint,false);
#else
        reverse_impl(output_adjoint,input_adjoint,initialize_input_adjoint,
            use_scalar_backend(output_adjoint.extent(0)));
#endif
    } else {
        if(backend_==Backend::automatic&&specialization_requested_)
            ++specialization_fallback_count_;
        reverse_impl(output_adjoint,input_adjoint,initialize_input_adjoint,
            use_scalar_backend(output_adjoint.extent(0)));
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_generic(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
    const bool initialize_input_adjoint) const
{
    reverse_impl(output_adjoint,input_adjoint,initialize_input_adjoint,true);
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_impl(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
    const bool initialize_input_adjoint,const bool use_scalar_path) const
{
    if(output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||input_adjoint.extent(0)!=output_adjoint.extent(0)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(input_dimension_))
        throw std::invalid_argument("Kokkos e3 linear reverse batch dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/reverse");
    if(initialize_input_adjoint)
        ordered_kokkos_deep_copy(input_adjoint,Precision(0));
    auto all_weights=weights; auto mask=output_mask;
    if(use_scalar_path) {
        for(const auto instruction:instructions)
            Kokkos::parallel_for(
                "e3 reverse linear small batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{static_cast<int>(output_adjoint.extent(0)),
                        instruction.input_multiplicity,instruction.width}),
                KOKKOS_LAMBDA(int sample,int source,int component) {
                    Precision value=Precision(0);
                    for(int target=0;target<instruction.output_multiplicity;++target) {
                        const int output_index=instruction.output_offset
                            +target*instruction.width+component;
                        value+=instruction.path_weight
                            *all_weights(instruction.weight_offset
                                +source*instruction.output_multiplicity+target)
                            *mask(output_index)*output_adjoint(sample,output_index);
                    }
                    input_adjoint(sample,instruction.input_offset
                        +source*instruction.width+component)+=value;
                });
        return;
    }
    for (const auto& instruction:instructions) {
        const int samples=output_adjoint.extent(0);
        if(instruction.identity) {
            if constexpr(std::is_same_v<
                typename Kokkos::DefaultExecutionSpace::memory_space,
                Kokkos::HostSpace>) {
                const int block_size=
                    instruction.input_multiplicity*instruction.width;
                const int input_offset=instruction.input_offset;
                const int output_offset=instruction.output_offset;
                Kokkos::parallel_for(
                    "e3 reverse linear identity",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        0,samples),
                    KOKKOS_LAMBDA(const int sample) {
                        for(int local=0;local<block_size;++local)
                            input_adjoint(sample,input_offset+local)+=
                                mask(output_offset+local)
                                *output_adjoint(sample,output_offset+local);
                    });
            } else
                Kokkos::parallel_for(
                    "e3 reverse linear identity",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                        {0,0,0},{instruction.input_multiplicity,samples,
                            instruction.width}),
                    KOKKOS_LAMBDA(int channel,int sample,int component) {
                        const int input_index=instruction.input_offset
                            +channel*instruction.width+component;
                        const int output_index=instruction.output_offset
                            +channel*instruction.width+component;
                        input_adjoint(sample,input_index)+=mask(output_index)
                            *output_adjoint(sample,output_index);
                    });
            continue;
        }
        const int columns=samples*instruction.width;
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(
                workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max<std::size_t>(workspace_->packed_input.extent(1),columns));
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(
                workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_input=Kokkos::subview(
            workspace_->packed_input,
            std::make_pair(0,instruction.input_multiplicity),
            std::make_pair(0,columns));
        auto packed_output=Kokkos::subview(
            workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            const int channel_tiles=(instruction.output_multiplicity
                +linear_host_transpose_channel_tile-1)
                /linear_host_transpose_channel_tile;
            const int sample_tiles=(samples+linear_host_transpose_sample_tile-1)
                /linear_host_transpose_sample_tile;
            const int output_offset=instruction.output_offset;
            const int output_multiplicity=instruction.output_multiplicity;
            const int width=instruction.width;
            Kokkos::parallel_for("e3 reverse linear pack blocked",
                channel_tiles*sample_tiles,KOKKOS_LAMBDA(int tile) {
                    const int channel_first=(tile%channel_tiles)
                        *linear_host_transpose_channel_tile;
                    const int sample_first=(tile/channel_tiles)
                        *linear_host_transpose_sample_tile;
                    const int channel_last=channel_first
                            +linear_host_transpose_channel_tile<output_multiplicity
                        ?channel_first+linear_host_transpose_channel_tile
                        :output_multiplicity;
                    const int sample_last=sample_first
                            +linear_host_transpose_sample_tile<samples
                        ?sample_first+linear_host_transpose_sample_tile:samples;
                    for(int component=0;component<width;++component)
                        for(int channel=channel_first;channel<channel_last;
                            ++channel) {
                            const int output_index=
                                output_offset+channel*width+component;
                            const Precision scale=mask(output_index);
                            for(int sample=sample_first;sample<sample_last;
                                ++sample)
                                packed_output(channel,sample*width+component)=
                                    scale*output_adjoint(sample,output_index);
                        }
                });
        } else
            Kokkos::parallel_for(
                "e3 reverse linear pack",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{instruction.output_multiplicity,samples,
                        instruction.width}),
                KOKKOS_LAMBDA(int channel,int sample,int component) {
                    const int output_index=instruction.output_offset
                        +channel*instruction.width+component;
                    packed_output(channel,sample*instruction.width+component)=
                        mask(output_index)*output_adjoint(sample,output_index);
                });
        using weight_matrix=Kokkos::View<
            const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        weight_matrix weight(
            all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm(
            "N","N",instruction.path_weight,weight,packed_output,Precision(0),packed_input);
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            const int input_multiplicity=instruction.input_multiplicity;
            const int width=instruction.width;
            const int input_offset=instruction.input_offset;
            const int channel_tiles=(input_multiplicity
                +linear_host_transpose_channel_tile-1)
                /linear_host_transpose_channel_tile;
            const int sample_tiles=(samples+linear_host_transpose_sample_tile-1)
                /linear_host_transpose_sample_tile;
            Kokkos::parallel_for("e3 reverse linear unpack blocked",
                channel_tiles*sample_tiles,KOKKOS_LAMBDA(int tile) {
                    const int channel_first=(tile%channel_tiles)
                        *linear_host_transpose_channel_tile;
                    const int sample_first=(tile/channel_tiles)
                        *linear_host_transpose_sample_tile;
                    const int channel_last=channel_first
                            +linear_host_transpose_channel_tile<input_multiplicity
                        ?channel_first+linear_host_transpose_channel_tile
                        :input_multiplicity;
                    const int sample_last=sample_first
                            +linear_host_transpose_sample_tile<samples
                        ?sample_first+linear_host_transpose_sample_tile:samples;
                    for(int component=0;component<width;++component)
                        for(int channel=channel_first;channel<channel_last;
                            ++channel)
                            for(int sample=sample_first;sample<sample_last;
                                ++sample)
                                input_adjoint(sample,input_offset
                                    +channel*width+component)
                                    +=packed_input(
                                        channel,sample*width+component);
                });
        } else Kokkos::parallel_for(
                "e3 reverse linear unpack",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{instruction.input_multiplicity,samples,
                        instruction.width}),
                KOKKOS_LAMBDA(int channel,int sample,int component) {
                    input_adjoint(sample,instruction.input_offset
                        +channel*instruction.width+component)
                        +=packed_input(
                            channel,sample*instruction.width+component);
                });
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_lane(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
    const bool initialize_input_adjoint) const
{
    if(output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||input_adjoint.extent(0)!=output_adjoint.extent(0)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(input_dimension_))
        throw std::invalid_argument(
            "Kokkos e3 linear lane reverse batch dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/reverse_lane");
    auto all_weights=weights; auto mask=output_mask;
    const std::size_t samples=output_adjoint.extent(0);
    if(initialize_input_adjoint) ordered_kokkos_deep_copy(input_adjoint,Precision(0));
    if(use_scalar_backend(output_adjoint.extent(0))) {
        for(const auto instruction:lane_instructions)
            Kokkos::parallel_for(
                "e3 reverse linear lane small batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                    Kokkos::IndexType<std::size_t>>(
                    {std::size_t(0),std::size_t(0),std::size_t(0)},
                    {samples,
                        static_cast<std::size_t>(instruction.input_multiplicity),
                        static_cast<std::size_t>(instruction.width)}),
                KOKKOS_LAMBDA(std::size_t sample,std::size_t source,
                    std::size_t component) {
                    Precision value=Precision(0);
                    const int input_base=instruction.input_offset
                        +component*instruction.input_multiplicity;
                    for(int target=0;target<instruction.output_multiplicity;
                        ++target) {
                        const int output_index=instruction.output_offset
                            +component*instruction.output_multiplicity+target;
                        // The mask table keeps its definition (node-major)
                        // order.
                        const int table_index=instruction.output_offset
                            +target*instruction.width+component;
                        value+=instruction.path_weight
                            *all_weights(instruction.weight_offset
                                +source*instruction.output_multiplicity+target)
                            *mask(table_index)
                            *output_adjoint(sample,output_index);
                    }
                    input_adjoint(sample,input_base+source)+=value;
                });
        return;
    }
    using weight_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:lane_instructions) {
        const std::size_t workspace_samples=std::min(
            samples,linear_lane_gemm_sample_tile);
        const std::size_t workspace_columns=workspace_samples
            *static_cast<std::size_t>(instruction.width);
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<workspace_columns) {
            if(workspace_->fixed_capacity)
                throw std::logic_error("fixed e3 linear input workspace is too small");
            Kokkos::realloc(workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max(workspace_->packed_input.extent(1),workspace_columns));
        }
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<workspace_columns) {
            if(workspace_->fixed_capacity)
                throw std::logic_error("fixed e3 linear output workspace is too small");
            Kokkos::realloc(workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max(workspace_->packed_output.extent(1),workspace_columns));
        }
        weight_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        for(std::size_t sample_begin=0;sample_begin<samples;
            sample_begin+=linear_lane_gemm_sample_tile) {
            const std::size_t active_samples=std::min(
                samples-sample_begin,linear_lane_gemm_sample_tile);
            const std::size_t columns=active_samples
                *static_cast<std::size_t>(instruction.width);
            auto packed_input=Kokkos::subview(workspace_->packed_input,
                std::make_pair(std::size_t(0),
                    static_cast<std::size_t>(instruction.input_multiplicity)),
                std::make_pair(std::size_t(0),columns));
            auto packed_output=Kokkos::subview(workspace_->packed_output,
                std::make_pair(std::size_t(0),
                    static_cast<std::size_t>(instruction.output_multiplicity)),
                std::make_pair(std::size_t(0),columns));
            Kokkos::parallel_for(
                "e3 reverse linear lane pack",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                    Kokkos::IndexType<std::size_t>>(
                    {std::size_t(0),std::size_t(0),std::size_t(0)},
                    {static_cast<std::size_t>(instruction.output_multiplicity),
                        active_samples,
                        static_cast<std::size_t>(instruction.width)}),
                KOKKOS_LAMBDA(std::size_t channel,std::size_t sample,
                    std::size_t component) {
                    const std::size_t output_index=instruction.output_offset
                        +component*instruction.output_multiplicity+channel;
                    // The mask table keeps its definition (node-major) order.
                    const std::size_t table_index=instruction.output_offset
                        +channel*instruction.width+component;
                    packed_output(channel,sample*instruction.width+component)=
                        mask(table_index)*output_adjoint(
                            sample_begin+sample,output_index);
                });
            KokkosBlas::gemm("N","N",instruction.path_weight,weight,
                packed_output,Precision(0),packed_input);
            Kokkos::parallel_for(
                "e3 reverse linear lane unpack",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                    Kokkos::IndexType<std::size_t>>(
                    {std::size_t(0),std::size_t(0),std::size_t(0)},
                    {static_cast<std::size_t>(instruction.input_multiplicity),
                        active_samples,
                        static_cast<std::size_t>(instruction.width)}),
                KOKKOS_LAMBDA(std::size_t channel,std::size_t sample,
                    std::size_t component) {
                    input_adjoint(sample_begin+sample,instruction.input_offset
                        +component*instruction.input_multiplicity+channel)
                        +=packed_input(
                            channel,sample*instruction.width+component);
                });
        }
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_to_packed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input_adjoint) const
{
    if(output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||packed_input_adjoint.size()!=output_adjoint.extent(0)
            *static_cast<std::size_t>(input_dimension_)
        ||!all_input_blocks_covered_)
        throw std::invalid_argument(
            "Kokkos e3 linear packed reverse output dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/reverse_to_packed");
    auto all_weights=weights;
    auto mask=output_mask;
    const int samples=output_adjoint.extent(0);
    using matrix=Kokkos::View<
        Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    using const_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:instructions) {
        const int columns=samples*instruction.width;
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_output=Kokkos::subview(workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
        Kokkos::parallel_for("e3 reverse linear pack output",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{instruction.output_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                const int index=instruction.output_offset
                    +channel*instruction.width+component;
                packed_output(channel,sample*instruction.width+component)=
                    mask(index)*output_adjoint(sample,index);
            });
        const_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        matrix destination(packed_input_adjoint.data()
                +static_cast<std::size_t>(samples)*instruction.input_offset,
            instruction.input_multiplicity,columns);
        KokkosBlas::gemm("N","N",instruction.path_weight,weight,packed_output,
            instruction.first_input_instruction?Precision(0):Precision(1),
            destination);
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_angular_major_to_packed(
    Kokkos::View<const Precision***,Kokkos::LayoutRight>
        angular_major_output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input_adjoint) const
{
    const int samples=angular_major_output_adjoint.extent(0);
    int angular_dimension=0;
    for(const auto& block:runtime_output_blocks) {
        if(block.multiplicity
            !=static_cast<int>(angular_major_output_adjoint.extent(2)))
            throw std::invalid_argument(
                "Kokkos angular-major reverse multiplicity is inconsistent.");
        angular_dimension+=block.width;
    }
    if(angular_major_output_adjoint.extent(1)
            !=static_cast<std::size_t>(angular_dimension)
        ||packed_input_adjoint.size()
            !=static_cast<std::size_t>(samples)*input_dimension_
        ||!all_input_blocks_covered_)
        throw std::invalid_argument(
            "Kokkos angular-major packed reverse dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_
            +"/reverse_angular_major_to_packed");
    auto all_weights=weights;
    auto mask=output_mask;
    using matrix=Kokkos::View<
        Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    using const_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:instructions) {
        int angular_offset=0;
        bool matched=false;
        for(const auto& block:runtime_output_blocks) {
            if(block.offset==instruction.output_offset) {
                if(block.multiplicity!=instruction.output_multiplicity
                    ||block.width!=instruction.width)
                    throw std::invalid_argument(
                        "Kokkos angular-major reverse block is inconsistent.");
                matched=true;
                break;
            }
            angular_offset+=block.width;
        }
        if(!matched)
            throw std::invalid_argument(
                "Kokkos angular-major reverse block is unavailable.");
        const int columns=samples*instruction.width;
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)
                <static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_output=Kokkos::subview(workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
            const int channel_tiles=(instruction.output_multiplicity
                +linear_host_transpose_channel_tile-1)
                /linear_host_transpose_channel_tile;
            const int sample_tiles=(samples+linear_host_transpose_sample_tile-1)
                /linear_host_transpose_sample_tile;
            const int output_offset=instruction.output_offset;
            const int output_multiplicity=instruction.output_multiplicity;
            const int width=instruction.width;
            Kokkos::parallel_for(
                "e3 reverse angular-major pack output blocked",
                channel_tiles*sample_tiles,KOKKOS_LAMBDA(int tile) {
                    const int channel_first=(tile%channel_tiles)
                        *linear_host_transpose_channel_tile;
                    const int sample_first=(tile/channel_tiles)
                        *linear_host_transpose_sample_tile;
                    const int channel_last=channel_first
                            +linear_host_transpose_channel_tile
                        <output_multiplicity
                        ?channel_first+linear_host_transpose_channel_tile
                        :output_multiplicity;
                    const int sample_last=sample_first
                            +linear_host_transpose_sample_tile<samples
                        ?sample_first+linear_host_transpose_sample_tile:samples;
                    for(int component=0;component<width;++component)
                        for(int channel=channel_first;channel<channel_last;
                            ++channel) {
                            const Precision scale=mask(
                                output_offset+channel*width+component);
                            for(int sample=sample_first;sample<sample_last;
                                ++sample)
                                packed_output(
                                    channel,sample*width+component)=scale
                                    *angular_major_output_adjoint(
                                        sample,angular_offset+component,channel);
                        }
                });
        } else
            Kokkos::parallel_for("e3 reverse angular-major pack output",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{instruction.output_multiplicity,samples,
                        instruction.width}),
                KOKKOS_LAMBDA(int channel,int sample,int component) {
                    const int index=instruction.output_offset
                        +channel*instruction.width+component;
                    packed_output(channel,sample*instruction.width+component)=
                        mask(index)*angular_major_output_adjoint(
                            sample,angular_offset+component,channel);
                });
        const_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        matrix destination(packed_input_adjoint.data()
                +static_cast<std::size_t>(samples)*instruction.input_offset,
            instruction.input_multiplicity,columns);
        KokkosBlas::gemm("N","N",instruction.path_weight,weight,packed_output,
            instruction.first_input_instruction?Precision(0):Precision(1),
            destination);
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_packed_to_packed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input_adjoint) const
{
    const std::size_t samples=packed_input_adjoint.extent(0);
    if(packed_output_adjoint.size()
            !=samples*static_cast<std::size_t>(output_dimension_)
        ||packed_input_adjoint.size()
            !=samples*static_cast<std::size_t>(input_dimension_)
        ||!supports_packed_reverse())
        throw std::invalid_argument(
            "Kokkos e3 linear packed reverse dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/reverse_packed_to_packed");
    auto all_weights=weights;
    using matrix=Kokkos::View<
        Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    using const_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:instructions) {
        const int columns=static_cast<int>(samples)*instruction.width;
        const_matrix source(packed_output_adjoint.data()
                +samples*instruction.output_offset,
            instruction.output_multiplicity,columns);
        const_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        matrix destination(packed_input_adjoint.data()
                +samples*instruction.input_offset,
            instruction.input_multiplicity,columns);
        KokkosBlas::gemm("N","N",instruction.path_weight,weight,source,
            instruction.first_input_instruction?Precision(0):Precision(1),
            destination);
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_from_packed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const
{
    if(packed_output_adjoint.size()!=input_adjoint.extent(0)
            *static_cast<std::size_t>(output_dimension_)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||!all_input_blocks_covered_||!active_output_mask_is_identity_)
        throw std::invalid_argument(
            "Kokkos e3 linear packed reverse input dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/reverse_from_packed");
    ordered_kokkos_deep_copy(input_adjoint,Precision(0));
    auto all_weights=weights;
    auto mask=output_mask;
    const int samples=input_adjoint.extent(0);
    using matrix=Kokkos::View<
        Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    using const_matrix=Kokkos::View<
        const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
    for(const auto& instruction:instructions) {
        const int columns=samples*instruction.width;
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max<std::size_t>(workspace_->packed_input.extent(1),columns));
        auto packed_result=Kokkos::subview(workspace_->packed_input,
            std::make_pair(0,instruction.input_multiplicity),
            std::make_pair(0,columns));
        const_matrix source(packed_output_adjoint.data()
                +static_cast<std::size_t>(samples)*instruction.output_offset,
            instruction.output_multiplicity,columns);
        const_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm("N","N",instruction.path_weight,weight,source,
            Precision(0),packed_result);
        Kokkos::parallel_for("e3 reverse linear unpack input",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{instruction.input_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                input_adjoint(sample,instruction.input_offset
                    +channel*instruction.width+component)+=
                    packed_result(channel,sample*instruction.width+component);
            });
    }
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse_ir_mul(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const
{
    if(output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||input_adjoint.extent(0)!=output_adjoint.extent(0)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(input_dimension_))
        throw std::invalid_argument(
            "Kokkos ir-mul e3 linear reverse batch dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/reverse_ir_mul");
    ordered_kokkos_deep_copy(input_adjoint,Precision(0));
    auto all_weights=weights;
    auto mask=output_mask;
    for(const auto& instruction:instructions) {
        const int samples=output_adjoint.extent(0);
        const int columns=samples*instruction.width;
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max<std::size_t>(workspace_->packed_input.extent(1),columns));
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_input=Kokkos::subview(workspace_->packed_input,
            std::make_pair(0,instruction.input_multiplicity),
            std::make_pair(0,columns));
        auto packed_output=Kokkos::subview(workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
        Kokkos::parallel_for("e3 ir-mul reverse pack",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
                {instruction.output_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                const int ir_mul=instruction.output_offset
                    +component*instruction.output_multiplicity+channel;
                const int mul_ir=instruction.output_offset
                    +channel*instruction.width+component;
                packed_output(channel,sample*instruction.width+component)=
                    mask(mul_ir)*output_adjoint(sample,ir_mul);
            });
        using weight_matrix=Kokkos::View<
            const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        weight_matrix weight(all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm("N","N",instruction.path_weight,weight,packed_output,
            Precision(0),packed_input);
        Kokkos::parallel_for("e3 ir-mul reverse unpack",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
                {instruction.input_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                input_adjoint(sample,instruction.input_offset
                    +component*instruction.input_multiplicity+channel)
                    +=packed_input(channel,sample*instruction.width+component);
            });
    }
}

template<typename Precision>
E3TensorProductKokkosT<Precision>::E3TensorProductKokkosT(const nlohmann::json& data)
{
    E3TensorProduct validated(data);
    Irreps in1(data.at("irreps_in1").get<std::string>()),in2(data.at("irreps_in2").get<std::string>()),out(data.at("irreps_out").get<std::string>());
    input_1_dimension_=in1.dimension(); input_2_dimension_=in2.dimension(); output_dimension_=out.dimension(); int offset=0;
    for (const auto& value:data.at("instructions")) {
        const auto& a=in1.blocks.at(value.at("i_in1").get<int>()); const auto& b=in2.blocks.at(value.at("i_in2").get<int>()); const auto& c=out.blocks.at(value.at("i_out").get<int>());
        const bool has=value.at("has_weight").get<bool>(); const auto mode=value.at("connection_mode").get<std::string>();
        if (mode!="uvu"&&mode!="uuu"&&mode!="uvw") throw std::invalid_argument("Unsupported Kokkos tensor-product mode: "+mode);
        Instruction instruction{a.offset,b.offset,c.offset,a.multiplicity,b.multiplicity,c.multiplicity,2*a.l+1,2*b.l+1,2*c.l+1,offset,has,mode=="uuu",mode=="uvw",value.at("path_weight").get<Precision>(),toKokkosView("e3 wigner",tensor_values<Precision>(value.at("wigner_3j")))};
        const auto path_shape=value.at("path_shape").get<std::vector<int>>();
        instructions.push_back(instruction); if(has) offset+=shape_product(value.at("path_shape").get<std::vector<int>>());
    }
    weight_size_=offset;
    internal_weights=toKokkosView(
        "e3 tensor weights",tensor_values<Precision>(data.at("weight")));
    const auto output_mask_host=
        tensor_values<Precision>(data.at("output_mask"));
    output_mask=toKokkosView("e3 tensor mask",output_mask_host);
    output_mask_is_identity_=std::all_of(
        output_mask_host.begin(),output_mask_host.end(),
        [](const Precision value) { return value==Precision(1); });
    initialize_direct_execution_plan(data,validated);
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,Kokkos::View<const Precision**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(try_evaluate_direct_execution(input_1,input_2,dynamic_weights,output))
        return;
    ProfileRegion profile("symmetrix/generic/tensor_product/forward");
    ordered_kokkos_deep_copy(output,Precision(0)); auto mask=output_mask; auto fixed=internal_weights;
    for (const auto instruction:instructions) { auto wigner=instruction.wigner;
        if (instruction.uvw) {
            Kokkos::parallel_for("e3 tensor product uvw",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(input_1.extent(0)),instruction.output_multiplicity,instruction.output_width}),KOKKOS_LAMBDA(int sample,int w,int c) {
                Precision result=Precision(0);
                for(int u=0;u<instruction.multiplicity_1;++u)
                    for(int v=0;v<instruction.multiplicity_2;++v) {
                        const int wi=instruction.weight_offset
                            +(u*instruction.multiplicity_2+v)
                                *instruction.output_multiplicity+w;
                        const Precision weight=instruction.has_weight
                            ?(dynamic_weights.extent(1)
                                ?dynamic_weights(sample,wi):fixed(wi))
                            :Precision(1);
                        for(int a=0;a<instruction.width_1;++a)
                            for(int b=0;b<instruction.width_2;++b)
                                result+=instruction.path_weight*weight
                                    *wigner((a*instruction.width_2+b)
                                        *instruction.output_width+c)
                                    *input_1(sample,instruction.input_1_offset
                                        +u*instruction.width_1+a)
                                    *input_2(sample,instruction.input_2_offset
                                        +v*instruction.width_2+b);
                    }
                const int index=instruction.output_offset
                    +w*instruction.output_width+c;
                output(sample,index)+=result*mask(index);
            });
            continue;
        }
        Kokkos::parallel_for("e3 tensor product",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(input_1.extent(0)),instruction.multiplicity_1,instruction.output_width}),KOKKOS_LAMBDA(int sample,int u,int c) {
            for(int v=0;v<instruction.multiplicity_2;++v) {
                if(instruction.uuu&&u!=v) continue;
                const int w_begin=instruction.uvw?0:u;
                const int w_end=instruction.uvw?instruction.output_multiplicity:u+1;
                for(int w=w_begin;w<w_end;++w) {
                    const int wi=instruction.weight_offset+(instruction.uvw
                        ?(u*instruction.multiplicity_2+v)*instruction.output_multiplicity+w
                        :(instruction.uuu?u:u*instruction.multiplicity_2+v));
                    const Precision weight=instruction.has_weight
                        ?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi))
                        :Precision(1);
                    Precision result=Precision(0);
                    for(int a=0;a<instruction.width_1;++a)
                        for(int b=0;b<instruction.width_2;++b)
                            result+=instruction.path_weight*weight
                                *wigner((a*instruction.width_2+b)*instruction.output_width+c)
                                *input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)
                                *input_2(sample,instruction.input_2_offset+v*instruction.width_2+b);
                    const int index=instruction.output_offset+w*instruction.output_width+c;
                    output(sample,index)+=result*mask(index);
                }
            }
        });
    }
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,Kokkos::View<const Precision**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> weights_adjoint) const
{
    if(try_reverse_direct_execution(input_1,input_2,dynamic_weights,
            output_adjoint,input_1_adjoint,input_2_adjoint,weights_adjoint))
        return;
    ProfileRegion profile("symmetrix/generic/tensor_product/reverse");
    ordered_kokkos_deep_copy(input_1_adjoint,Precision(0));
    ordered_kokkos_deep_copy(input_2_adjoint,Precision(0));
    ordered_kokkos_deep_copy(weights_adjoint,Precision(0));
    auto mask=output_mask; auto fixed=internal_weights;
    for(const auto instruction:instructions) { auto wigner=instruction.wigner;
        Kokkos::parallel_for("e3 tensor reverse",input_1.extent(0),KOKKOS_LAMBDA(int sample) {
            for(int u=0;u<instruction.multiplicity_1;++u) for(int v=0;v<instruction.multiplicity_2;++v) {
                if(instruction.uuu&&u!=v) continue;
                const int w_begin=instruction.uvw?0:u;
                const int w_end=instruction.uvw?instruction.output_multiplicity:u+1;
                for(int w=w_begin;w<w_end;++w) {
                    const int wi=instruction.weight_offset+(instruction.uvw
                        ?(u*instruction.multiplicity_2+v)*instruction.output_multiplicity+w
                        :(instruction.uuu?u:u*instruction.multiplicity_2+v));
                    const Precision weight=instruction.has_weight
                        ?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi))
                        :Precision(1);
                    for(int a=0;a<instruction.width_1;++a)
                        for(int b=0;b<instruction.width_2;++b)
                            for(int c=0;c<instruction.output_width;++c) {
                                const int oi=instruction.output_offset+w*instruction.output_width+c;
                                const Precision common=instruction.path_weight
                                    *wigner((a*instruction.width_2+b)*instruction.output_width+c)
                                    *mask(oi)*output_adjoint(sample,oi);
                                input_1_adjoint(sample,instruction.input_1_offset+u*instruction.width_1+a)
                                    +=common*weight*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b);
                                input_2_adjoint(sample,instruction.input_2_offset+v*instruction.width_2+b)
                                    +=common*weight*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a);
                                if(instruction.has_weight && dynamic_weights.extent(1))
                                    weights_adjoint(sample,wi)+=common
                                        *input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)
                                        *input_2(sample,instruction.input_2_offset+v*instruction.width_2+b);
                            }
                }
            }
        });
    }
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::node_tile_capacity(
    const int remaining,const int k_dimension,const int result_width) const
{
    constexpr std::size_t budget=std::size_t(4)<<20;
    const std::size_t per_node=(static_cast<std::size_t>(k_dimension)
        +static_cast<std::size_t>(result_width))*sizeof(Precision);
    const std::size_t capacity=per_node==0
        ?1:std::max<std::size_t>(1,budget/per_node);
    return static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(std::max(remaining,1)),capacity));
}

template<typename Precision>
typename E3TensorProductKokkosT<Precision>::BilinearWorkspaceShape
E3TensorProductKokkosT<Precision>::bilinear_workspace_shape(
    const std::size_t samples) const
{
    BilinearWorkspaceShape shape;
    for(const auto& instruction:instructions) {
        const int k_dimension=instruction.multiplicity_1
            *instruction.multiplicity_2;
        const int result_width=instruction.output_multiplicity;
        const int tile=node_tile_capacity(
            static_cast<int>(samples),k_dimension,result_width);
        const std::size_t matrix_values=static_cast<std::size_t>(tile)
            *static_cast<std::size_t>(k_dimension);
        const std::size_t result_values=static_cast<std::size_t>(tile)
            *static_cast<std::size_t>(result_width);
        shape.matrix_values=std::max(shape.matrix_values,matrix_values);
        shape.result_values=std::max(shape.result_values,result_values);
        shape.adjoint_values=std::max(shape.adjoint_values,result_values);
        shape.k_adjoint_values=std::max(
            shape.k_adjoint_values,matrix_values);
    }
    return shape;
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::ensure_bilinear_workspace(
    const int tile,const int k_dimension,const int result_width) const
{
    const auto wanted_matrix=static_cast<std::size_t>(tile)
        *static_cast<std::size_t>(k_dimension);
    const auto wanted_result=static_cast<std::size_t>(tile)
        *static_cast<std::size_t>(result_width);
    auto ensure=[this](auto& view,const std::size_t wanted) {
        if(view.size()>=wanted) return;
        if(bilinear_workspace_->fixed_capacity)
            throw std::logic_error(
                "fixed Kokkos bilinear workspace is smaller than its plan");
        Kokkos::realloc(view,std::size_t(1),wanted);
    };
    ensure(bilinear_workspace_->matrix,wanted_matrix);
    ensure(bilinear_workspace_->result,wanted_result);
    ensure(bilinear_workspace_->adjoint,wanted_result);
    ensure(bilinear_workspace_->k_adjoint,wanted_matrix);
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::supports_node_bilinear_uvw() const
{
    if(instructions.empty()||internal_weights.extent(0)==0)
        return false;
    // Any all-uvw definition with internal weights is eligible: per
    // instruction block offsets and widths are honoured by the tiled GEMM
    // below.  Verified against the general tensor-product path on the
    // multi-instruction self connections with differing k dimensions.
    if(output_mask.extent(0)!=static_cast<std::size_t>(output_dimension_))
        return false;
    // The tiled GEMM below is portable across Kokkos execution spaces, but
    // only the host and CUDA backends are validated; other device backends
    // (HIP, SYCL, OpenMP-target) stay on the general tensor-product path.
    if constexpr(!std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,Kokkos::HostSpace>) {
#ifdef KOKKOS_ENABLE_CUDA
        if constexpr(!std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
            return false;
#else
        return false;
#endif
    }
    for(const auto& instruction:instructions) {
        if(!instruction.uvw||!instruction.has_weight)
            return false;
        if(instruction.multiplicity_1*instruction.multiplicity_2<=0
            ||instruction.output_multiplicity<=0)
            return false;
        if(instruction.width_1*instruction.width_2>49)
            return false;
    }
    return true;
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::evaluate_node_bilinear_uvw(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(!supports_node_bilinear_uvw())
        throw std::invalid_argument(
            "Kokkos node bilinear uvw forward is unavailable.");
    if(input_1.extent(1)!=static_cast<std::size_t>(input_1_dimension_)
        ||input_2.extent(1)!=static_cast<std::size_t>(input_2_dimension_)
        ||output.extent(0)!=input_1.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument(
            "Kokkos node bilinear uvw forward dimensions are inconsistent.");
    ProfileRegion profile("symmetrix/node_bilinear_uvw/forward");
    ordered_kokkos_deep_copy(output,Precision(0));
    const int samples=static_cast<int>(input_1.extent(0));
    auto fixed=internal_weights;
    auto mask=output_mask;
    for(const auto& instruction:instructions) { auto wigner=instruction.wigner;
        const int d1=instruction.width_1,d2=instruction.width_2;
        const int output_width=instruction.output_width;
        const int k_dimension=instruction.multiplicity_1
            *instruction.multiplicity_2;
        const int result_width=instruction.output_multiplicity;
        for(int a=0;a<d1;++a) for(int b=0;b<d2;++b) {
            const int wigner_base=(a*d2+b)*output_width;
            for(int node_begin=0;node_begin<samples;) {
                const int tile=node_tile_capacity(
                    samples-node_begin,k_dimension,result_width);
                ensure_bilinear_workspace(tile,k_dimension,result_width);
                // The workspace is sized for the widest instruction seen so
                // far, so a Kokkos::subview of it would carry the parent row
                // stride while the GEMM below interprets the same buffer as
                // compact rows.  Interpret the buffer as compact rows of this
                // instruction's width instead.
                using node_matrix=Kokkos::View<Precision**,
                    Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
                node_matrix matrix(bilinear_workspace_->matrix.data(),tile,
                    k_dimension);
                node_matrix result(bilinear_workspace_->result.data(),tile,
                    result_width);
                const int first=node_begin;
                const int multiplicity_1=instruction.multiplicity_1;
                const int multiplicity_2=instruction.multiplicity_2;
                const int input_1_offset=instruction.input_1_offset+a;
                const int input_2_offset=instruction.input_2_offset+b;
                // Row-owner build: the (u,v) decomposition is hoisted out of
                // the element loop and the inner v run is contiguous, so the
                // kernel is a plain broadcast-multiply with no per-element
                // division or strided gather of the packed axis.
                Kokkos::parallel_for("e3 node bilinear build matrix",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0,tile),
                    KOKKOS_LAMBDA(const int row) {
                        const int sample=first+row;
                        for(int u=0;u<multiplicity_1;++u) {
                            const Precision value=
                                input_1(sample,input_1_offset+u*d1);
                            for(int v=0;v<multiplicity_2;++v)
                                matrix(row,u*multiplicity_2+v)=value
                                    *input_2(sample,input_2_offset+v*d2);
                        }
                    });
                using const_node_matrix=Kokkos::View<const Precision**,
                    Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
                const_node_matrix weight(
                    fixed.data()+instruction.weight_offset,
                    k_dimension,result_width);
                KokkosBlas::gemm("N","N",instruction.path_weight,
                    matrix,weight,Precision(0),result);
                const int output_offset=instruction.output_offset;
                auto all_output=output;
                Kokkos::parallel_for("e3 node bilinear accumulate output",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0,tile),
                    KOKKOS_LAMBDA(const int row) {
                        const int sample=first+row;
                        for(int w=0;w<result_width;++w) {
                            const Precision value=result(row,w);
                            const int base=output_offset+w*output_width;
                            for(int c=0;c<output_width;++c)
                                all_output(sample,base+c)+=
                                    value*wigner(wigner_base+c)
                                    *mask(base+c);
                        }
                    });
                node_begin+=tile;
            }
        }
    }
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::reverse_node_bilinear_uvw(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint) const
{
    if(!supports_node_bilinear_uvw())
        throw std::invalid_argument(
            "Kokkos node bilinear uvw reverse is unavailable.");
    if(input_1.extent(1)!=static_cast<std::size_t>(input_1_dimension_)
        ||input_2.extent(1)!=static_cast<std::size_t>(input_2_dimension_)
        ||output_adjoint.extent(0)!=input_1.extent(0)
        ||output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||input_1_adjoint.extent(0)!=input_1.extent(0)
        ||input_1_adjoint.extent(1)!=static_cast<std::size_t>(input_1_dimension_)
        ||input_2_adjoint.extent(0)!=input_1.extent(0)
        ||input_2_adjoint.extent(1)!=static_cast<std::size_t>(input_2_dimension_))
        throw std::invalid_argument(
            "Kokkos node bilinear uvw reverse dimensions are inconsistent.");
    ProfileRegion profile("symmetrix/node_bilinear_uvw/reverse");
    ordered_kokkos_deep_copy(input_1_adjoint,Precision(0));
    ordered_kokkos_deep_copy(input_2_adjoint,Precision(0));
    const int samples=static_cast<int>(input_1.extent(0));
    auto fixed=internal_weights;
    auto mask=output_mask;
    for(const auto& instruction:instructions) { auto wigner=instruction.wigner;
        const int d1=instruction.width_1,d2=instruction.width_2;
        const int output_width=instruction.output_width;
        const int k_dimension=instruction.multiplicity_1
            *instruction.multiplicity_2;
        const int result_width=instruction.output_multiplicity;
        for(int a=0;a<d1;++a) for(int b=0;b<d2;++b) {
            const int wigner_base=(a*d2+b)*output_width;
            for(int node_begin=0;node_begin<samples;) {
                const int tile=node_tile_capacity(
                    samples-node_begin,k_dimension,result_width);
                ensure_bilinear_workspace(tile,k_dimension,result_width);
                // Compact unmanaged rows of this instruction's width; the
                // workspace may be wider (see evaluate_node_bilinear_uvw).
                using node_matrix=Kokkos::View<Precision**,
                    Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
                node_matrix adjoint(bilinear_workspace_->adjoint.data(),tile,
                    result_width);
                node_matrix k_adjoint(bilinear_workspace_->k_adjoint.data(),tile,
                    k_dimension);
                const int first=node_begin;
                const int output_offset=instruction.output_offset;
                const int multiplicity_2=instruction.multiplicity_2;
                const int input_1_offset=instruction.input_1_offset+a;
                const int input_2_offset=instruction.input_2_offset+b;
                Kokkos::parallel_for("e3 node bilinear build output adjoint",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0,tile),
                    KOKKOS_LAMBDA(const int row) {
                        const int sample=first+row;
                        for(int w=0;w<result_width;++w) {
                            const int base=output_offset+w*output_width;
                            Precision value=Precision(0);
                            for(int c=0;c<output_width;++c)
                                value+=wigner(wigner_base+c)*mask(base+c)
                                    *output_adjoint(sample,base+c);
                            adjoint(row,w)=instruction.path_weight*value;
                        }
                    });
                using matrix_view=Kokkos::View<
                    Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>;
                using const_matrix_view=Kokkos::View<
                    const Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>;
                const_matrix_view weight(
                    fixed.data()+instruction.weight_offset,
                    k_dimension,result_width);
                KokkosBlas::gemm("N","T",Precision(1),
                    adjoint,weight,Precision(0),k_adjoint);
                auto input_1_adjoint_view=input_1_adjoint;
                auto input_2_adjoint_view=input_2_adjoint;
                const int multiplicity_1=instruction.multiplicity_1;
                // Row-owner scatter: each output element is reduced in a
                // register over the contracted axis, giving one read-modify-
                // write per element instead of one per packed k entry.  The
                // k_adjoint run is unit-stride; the other operand is the
                // strided angular component of its row.
                Kokkos::parallel_for("e3 node bilinear scatter adjoints",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0,tile),
                    KOKKOS_LAMBDA(const int row) {
                        const int sample=first+row;
                        for(int u=0;u<multiplicity_1;++u) {
                            Precision reduced=Precision(0);
                            const std::size_t k_base=
                                static_cast<std::size_t>(row)
                                    * static_cast<std::size_t>(k_dimension)
                                + static_cast<std::size_t>(u)
                                    * static_cast<std::size_t>(multiplicity_2);
                            for(int v=0;v<multiplicity_2;++v)
                                reduced+=k_adjoint.data()[k_base
                                        +static_cast<std::size_t>(v)]
                                    *input_2(sample,input_2_offset+v*d2);
                            input_1_adjoint_view(sample,input_1_offset+u*d1)+=
                                reduced;
                        }
                        for(int v=0;v<multiplicity_2;++v) {
                            Precision reduced=Precision(0);
                            for(int u=0;u<multiplicity_1;++u)
                                reduced+=k_adjoint(row,u*multiplicity_2+v)
                                    *input_1(sample,input_1_offset+u*d1);
                            input_2_adjoint_view(sample,input_2_offset+v*d2)+=
                                reduced;
                        }
                    });
                node_begin+=tile;
            }
        }
    }
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::evaluate_node_bilinear_uvw_lane(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(!supports_node_bilinear_uvw_lane())
        throw std::invalid_argument(
            "Kokkos node bilinear uvw lane forward is unavailable.");
    if(input_1.extent(1)!=static_cast<std::size_t>(input_1_dimension_)
        ||input_2.extent(1)!=static_cast<std::size_t>(input_2_dimension_)
        ||output.extent(0)!=input_1.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument(
            "Kokkos node bilinear uvw lane forward dimensions are "
            "inconsistent.");
    ProfileRegion profile("symmetrix/node_bilinear_uvw_lane/forward");
    ordered_kokkos_deep_copy(output,Precision(0));
    const int samples=static_cast<int>(input_1.extent(0));
    auto fixed=internal_weights;
    auto mask=output_mask;
    for(const auto& instruction:instructions) { auto wigner=instruction.wigner;
        const int d1=instruction.width_1,d2=instruction.width_2;
        const int output_width=instruction.output_width;
        const int k_dimension=instruction.multiplicity_1
            *instruction.multiplicity_2;
        const int result_width=instruction.output_multiplicity;
        for(int a=0;a<d1;++a) for(int b=0;b<d2;++b) {
            const int wigner_base=(a*d2+b)*output_width;
            for(int node_begin=0;node_begin<samples;) {
                const int tile=node_tile_capacity(
                    samples-node_begin,k_dimension,result_width);
                ensure_bilinear_workspace(tile,k_dimension,result_width);
                // Compact unmanaged rows of this instruction's width; the
                // workspace may be wider (see evaluate_node_bilinear_uvw).
                using node_matrix=Kokkos::View<Precision**,
                    Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
                node_matrix matrix(bilinear_workspace_->matrix.data(),tile,
                    k_dimension);
                node_matrix result(bilinear_workspace_->result.data(),tile,
                    result_width);
                const int first=node_begin;
                const int multiplicity_1=instruction.multiplicity_1;
                const int multiplicity_2=instruction.multiplicity_2;
                // Lane-major: block element (channel a, component x) lives at
                // offset + x*multiplicity + a, so the a-th component of this
                // (u,v) slice starts at input_1_offset + a*multiplicity_1.
                const int input_1_lane=instruction.input_1_offset
                    +a*multiplicity_1;
                const int input_2_lane=instruction.input_2_offset
                    +b*multiplicity_2;
                // Row-owner build: the (u,v) decomposition is hoisted out of
                // the element loop and the inner v run is contiguous, so the
                // kernel is a plain broadcast-multiply.
                Kokkos::parallel_for("e3 node bilinear lane build matrix",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0,tile),
                    KOKKOS_LAMBDA(const int row) {
                        const int sample=first+row;
                        for(int u=0;u<multiplicity_1;++u) {
                            const Precision value=
                                input_1(sample,input_1_lane+u);
                            for(int v=0;v<multiplicity_2;++v)
                                matrix(row,u*multiplicity_2+v)=value
                                    *input_2(sample,input_2_lane+v);
                        }
                    });
                using const_node_matrix=Kokkos::View<const Precision**,
                    Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
                const_node_matrix weight(
                    fixed.data()+instruction.weight_offset,
                    k_dimension,result_width);
                KokkosBlas::gemm("N","N",instruction.path_weight,
                    matrix,weight,Precision(0),result);
                const int output_offset=instruction.output_offset;
                auto all_output=output;
                Kokkos::parallel_for("e3 node bilinear lane accumulate output",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0,tile),
                    KOKKOS_LAMBDA(const int row) {
                        const int sample=first+row;
                        for(int w=0;w<result_width;++w) {
                            const Precision value=result(row,w);
                            // The mask is a definition-layout (node-major)
                            // table, so it keeps the node-major index of the
                            // (channel w, component c) element; only the
                            // output view is lane-major.
                            const int base=output_offset+w*output_width;
                            for(int c=0;c<output_width;++c)
                                all_output(sample,
                                    output_offset+w+c*result_width)+=
                                    value*wigner(wigner_base+c)
                                    *mask(base+c);
                        }
                    });
                node_begin+=tile;
            }
        }
    }
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::reverse_node_bilinear_uvw_lane(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint) const
{
    if(!supports_node_bilinear_uvw_lane())
        throw std::invalid_argument(
            "Kokkos node bilinear uvw lane reverse is unavailable.");
    if(input_1.extent(1)!=static_cast<std::size_t>(input_1_dimension_)
        ||input_2.extent(1)!=static_cast<std::size_t>(input_2_dimension_)
        ||output_adjoint.extent(0)!=input_1.extent(0)
        ||output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||input_1_adjoint.extent(0)!=input_1.extent(0)
        ||input_1_adjoint.extent(1)!=static_cast<std::size_t>(input_1_dimension_)
        ||input_2_adjoint.extent(0)!=input_1.extent(0)
        ||input_2_adjoint.extent(1)!=static_cast<std::size_t>(input_2_dimension_))
        throw std::invalid_argument(
            "Kokkos node bilinear uvw lane reverse dimensions are "
            "inconsistent.");
    ProfileRegion profile("symmetrix/node_bilinear_uvw_lane/reverse");
    ordered_kokkos_deep_copy(input_1_adjoint,Precision(0));
    ordered_kokkos_deep_copy(input_2_adjoint,Precision(0));
    const int samples=static_cast<int>(input_1.extent(0));
    auto fixed=internal_weights;
    auto mask=output_mask;
    for(const auto& instruction:instructions) { auto wigner=instruction.wigner;
        const int d1=instruction.width_1,d2=instruction.width_2;
        const int output_width=instruction.output_width;
        const int k_dimension=instruction.multiplicity_1
            *instruction.multiplicity_2;
        const int result_width=instruction.output_multiplicity;
        for(int a=0;a<d1;++a) for(int b=0;b<d2;++b) {
            const int wigner_base=(a*d2+b)*output_width;
            for(int node_begin=0;node_begin<samples;) {
                const int tile=node_tile_capacity(
                    samples-node_begin,k_dimension,result_width);
                ensure_bilinear_workspace(tile,k_dimension,result_width);
                // Compact unmanaged rows of this instruction's width; the
                // workspace may be wider (see evaluate_node_bilinear_uvw).
                using node_matrix=Kokkos::View<Precision**,
                    Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
                node_matrix adjoint(bilinear_workspace_->adjoint.data(),tile,
                    result_width);
                node_matrix k_adjoint(bilinear_workspace_->k_adjoint.data(),tile,
                    k_dimension);
                const int first=node_begin;
                const int output_offset=instruction.output_offset;
                const int multiplicity_1=instruction.multiplicity_1;
                const int multiplicity_2=instruction.multiplicity_2;
                // Lane-major component offsets for this (a,b) pair, matching
                // the forward kernel's operand addressing.
                const int input_1_lane=instruction.input_1_offset
                    +a*multiplicity_1;
                const int input_2_lane=instruction.input_2_offset
                    +b*multiplicity_2;
                Kokkos::parallel_for("e3 node bilinear lane build adjoint",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0,tile),
                    KOKKOS_LAMBDA(const int row) {
                        const int sample=first+row;
                        for(int w=0;w<result_width;++w) {
                            const int base=output_offset+w*output_width;
                            Precision value=Precision(0);
                            for(int c=0;c<output_width;++c)
                                value+=wigner(wigner_base+c)*mask(base+c)
                                    *output_adjoint(sample,
                                        output_offset+w+c*result_width);
                            adjoint(row,w)=instruction.path_weight*value;
                        }
                    });
                using const_matrix_view=Kokkos::View<
                    const Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>;
                const_matrix_view weight(
                    fixed.data()+instruction.weight_offset,
                    k_dimension,result_width);
                KokkosBlas::gemm("N","T",Precision(1),
                    adjoint,weight,Precision(0),k_adjoint);
                auto input_1_adjoint_view=input_1_adjoint;
                auto input_2_adjoint_view=input_2_adjoint;
                Kokkos::parallel_for("e3 node bilinear lane scatter adjoints",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0,tile),
                    KOKKOS_LAMBDA(const int row) {
                        const int sample=first+row;
                        for(int u=0;u<multiplicity_1;++u) {
                            Precision reduced=Precision(0);
                            const std::size_t k_base=
                                static_cast<std::size_t>(row)
                                    * static_cast<std::size_t>(k_dimension)
                                + static_cast<std::size_t>(u)
                                    * static_cast<std::size_t>(multiplicity_2);
                            for(int v=0;v<multiplicity_2;++v)
                                reduced+=k_adjoint.data()[k_base
                                        +static_cast<std::size_t>(v)]
                                    *input_2(sample,input_2_lane+v);
                            input_1_adjoint_view(sample,input_1_lane+u)+=
                                reduced;
                        }
                        for(int v=0;v<multiplicity_2;++v) {
                            Precision reduced=Precision(0);
                            for(int u=0;u<multiplicity_1;++u)
                                reduced+=k_adjoint(row,u*multiplicity_2+v)
                                    *input_1(sample,input_1_lane+u);
                            input_2_adjoint_view(sample,input_2_lane+v)+=
                                reduced;
                        }
                    });
                node_begin+=tile;
            }
        }
    }
}

template class E3LinearKokkosT<float>;
template class E3LinearKokkosT<double>;
template class E3TensorProductKokkosT<float>;
template class E3TensorProductKokkosT<double>;
