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

template<typename Precision>
std::vector<Precision> tensor_values(const nlohmann::json& value) { return value.at("values").get<std::vector<Precision>>(); }
int shape_product(const std::vector<int>& shape) { return std::accumulate(shape.begin(),shape.end(),1,std::multiplies<int>()); }

int mh1_cuda_team_size(int multiplicity)
{
    if(multiplicity<=0) return 0;
    constexpr int warp_size=32;
    constexpr int maximum_team_size=128;
    const int capped=std::min(multiplicity,maximum_team_size);
    return std::max(
        warp_size,((capped+warp_size-1)/warp_size)*warp_size);
}

class ProfileRegion {
public:
    explicit ProfileRegion(const char* name) { Kokkos::Profiling::pushRegion(name); }
    explicit ProfileRegion(const std::string& name) {
        Kokkos::Profiling::pushRegion(name.c_str());
    }
    ~ProfileRegion() { Kokkos::Profiling::popRegion(); }
};

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
    for (const auto& value:data.at("instructions")) {
        const auto& in=input.blocks.at(value.at("i_in").get<int>()); const auto& out=output.blocks.at(value.at("i_out").get<int>());
        const auto shape=value.at("path_shape").get<std::vector<int>>();
        instructions.push_back({in.offset,out.offset,in.multiplicity,
            out.multiplicity,2*in.l+1,offset,
            value.at("path_weight").get<Precision>()});
        offset+=shape_product(shape);
    }
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
    for(std::size_t index=0;index<instructions.size();++index) {
        auto& instruction=instructions[index];
        instruction.first_input_instruction=std::none_of(
            instructions.begin(),instructions.begin()+index,
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
            instructions.begin(),instructions.begin()+index,
            [&](const auto& other) {
                return other.output_offset==instruction.output_offset;
            });
        instruction.last_output_instruction=std::none_of(
            instructions.begin()+index+1,instructions.end(),
            [&](const auto& other) {
                return other.output_offset==instruction.output_offset;
            });
    }
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
void E3LinearKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(input.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument("Kokkos e3 linear batch dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/forward");
    auto all_weights=weights;
    const bool use_scalar_path=use_scalar_backend(input.extent(0));
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
        throw std::invalid_argument("Kokkos e3 linear reverse batch dimensions are inconsistent.");
    ProfileRegion profile(
        "symmetrix/e3_linear/"+profile_name_+"/reverse");
    if(initialize_input_adjoint)
        ordered_kokkos_deep_copy(input_adjoint,Precision(0));
    auto all_weights=weights; auto mask=output_mask;
    const bool use_scalar_path=use_scalar_backend(output_adjoint.extent(0));
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
    bool official_layout=!validated.instructions.empty();
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
    int instruction_index=0;
    for (const auto& value:data.at("instructions")) {
        const auto& a=in1.blocks.at(value.at("i_in1").get<int>()); const auto& b=in2.blocks.at(value.at("i_in2").get<int>()); const auto& c=out.blocks.at(value.at("i_out").get<int>());
        const bool has=value.at("has_weight").get<bool>(); const auto mode=value.at("connection_mode").get<std::string>();
        if (mode!="uvu"&&mode!="uuu") throw std::invalid_argument("Unsupported Kokkos tensor-product mode: "+mode);
        Instruction instruction{a.offset,b.offset,c.offset,a.multiplicity,b.multiplicity,c.multiplicity,2*a.l+1,2*b.l+1,2*c.l+1,offset,has,mode=="uuu",value.at("path_weight").get<Precision>(),toKokkosView("e3 wigner",tensor_values<Precision>(value.at("wigner_3j")))};
        const auto path_shape=value.at("path_shape").get<std::vector<int>>();
        const int instruction_multiplicity=a.multiplicity;
        official_layout=official_layout&&mode=="uvu"&&has
            &&instruction_multiplicity>0&&b.multiplicity==1
            &&c.multiplicity==instruction_multiplicity
            &&(mh1_multiplicity_==0||mh1_multiplicity_==instruction_multiplicity)
            &&instruction.width_1<=3
            &&path_shape==std::vector<int>({instruction_multiplicity,1});
        if(official_layout) {
            mh1_multiplicity_=instruction_multiplicity;
            mh1_max_output_width_=std::max(
                mh1_max_output_width_,instruction.output_width);
            const auto& entries=validated.instructions.at(instruction_index).nonzero_wigner;
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
            set_kokkos_view(
                instruction.sparse_indices,indices,values.size(),3);
            instruction.sparse_values=toKokkosView("e3 sparse wigner",values);
            instruction.component_offsets=toKokkosView(
                "e3 sparse component offsets",component_offsets);
            instruction.sparse_count=values.size();

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
        instructions.push_back(instruction); if(has) offset+=shape_product(value.at("path_shape").get<std::vector<int>>());
        ++instruction_index;
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
    mh1_input_1_mul_to_ir=toKokkosView(
        "e3 mh1 input mul-to-ir",input_1_mul_to_ir_host);
    mh1_output_mul_to_ir=toKokkosView(
        "e3 mh1 output mul-to-ir",output_mul_to_ir_host);
    std::vector<Precision> output_mask_ir_mul_host(output_mask_host.size());
    for(std::size_t mul_ir=0;mul_ir<output_mask_host.size();++mul_ir)
        output_mask_ir_mul_host[static_cast<std::size_t>(
            output_mul_to_ir_host[mul_ir])]=output_mask_host[mul_ir];
    mh1_output_mask_ir_mul=toKokkosView(
        "e3 mh1 tensor mask ir-mul",output_mask_ir_mul_host);
    mh1_fast_path=official_layout&&internal_weights.extent(0)==0;
    if(mh1_fast_path) {
        std::vector<int> input_component_data;
        mh1_direct_node_layout_=true;
        for(const auto& block:in1.blocks) {
            if(block.multiplicity!=mh1_multiplicity_) {
                mh1_direct_node_layout_=false;
                break;
            }
            const int width=2*block.l+1;
            for(int component=0;component<width;++component) {
                input_component_data.push_back(block.offset+component);
                input_component_data.push_back(width);
            }
        }
        mh1_input_1_angular_dimension_=input_component_data.size()/2;
        mh1_direct_node_layout_=mh1_direct_node_layout_
            &&mh1_input_1_angular_dimension_>0;
        if(mh1_direct_node_layout_)
            set_kokkos_view(
                mh1_input_component_data,input_component_data,
                mh1_input_1_angular_dimension_,2);
        mh1_cuda_channel_team_size_=mh1_cuda_team_size(mh1_multiplicity_);
        mh1_cuda_harmonic_team_size_=mh1_cuda_team_size(mh1_multiplicity_);
        mh1_instruction_count_=instructions.size();
        set_kokkos_view(
            mh1_instruction_data,mh1_instruction_data_host,
            mh1_instruction_count_,7);
        mh1_path_weights=toKokkosView(
            "e3 mh1 path weights",mh1_path_weights_host);
        mh1_component_offsets=toKokkosView(
            "e3 mh1 component offsets",mh1_component_offsets_host);
        set_kokkos_view(
            mh1_sparse_indices,mh1_sparse_indices_host,
            mh1_sparse_values_host.size(),3);
        mh1_sparse_values=toKokkosView(
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
        mh1_harmonic_offsets=toKokkosView(
            "e3 mh1 harmonic offsets",harmonic_offsets);
        set_kokkos_view(
            mh1_harmonic_terms,harmonic_terms,harmonic_values.size(),3);
        mh1_harmonic_values=toKokkosView(
            "e3 mh1 harmonic values",harmonic_values);

        std::vector<std::vector<InputTerm>> input_terms(
            mh1_input_1_angular_dimension_);
        if(mh1_direct_node_layout_) {
            for(const auto& item:mh1_input_terms_unsorted) {
                if(item.first<0
                    ||item.first>=mh1_input_1_angular_dimension_) {
                    mh1_direct_node_layout_=false;
                    break;
                }
                input_terms[item.first].push_back(item.second);
            }
        }
        if(mh1_direct_node_layout_) {
            std::vector<int> input_term_offsets(
                mh1_input_1_angular_dimension_+1,0);
            std::vector<int> input_instruction_term_offsets(
                static_cast<std::size_t>(mh1_input_1_angular_dimension_)
                    *static_cast<std::size_t>(mh1_instruction_count_+1),
                0);
            std::vector<int> packed_input_terms;
            std::vector<Precision> input_values;
            for(int component=0;
                component<mh1_input_1_angular_dimension_;++component) {
                for(int instruction=0;instruction<mh1_instruction_count_;
                    ++instruction) {
                    input_instruction_term_offsets[
                        static_cast<std::size_t>(component)
                            *static_cast<std::size_t>(mh1_instruction_count_+1)
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
                        *static_cast<std::size_t>(mh1_instruction_count_+1)
                    +static_cast<std::size_t>(mh1_instruction_count_)]
                    =input_values.size();
                input_term_offsets[component+1]=input_values.size();
            }
            mh1_input_term_offsets=toKokkosView(
                "e3 mh1 input term offsets",input_term_offsets);
            set_kokkos_view(
                mh1_input_instruction_term_offsets,
                input_instruction_term_offsets,
                mh1_input_1_angular_dimension_,mh1_instruction_count_+1);
            set_kokkos_view(mh1_input_terms,packed_input_terms,
                input_values.size(),3);
            mh1_input_values=toKokkosView(
                "e3 mh1 input values",input_values);
        }

    }
}

template<typename Precision>
std::string E3TensorProductKokkosT<Precision>::execution_backend() const
{
    if(!mh1_fast_path) return "generic_kokkos";
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
        return mh1_fast_path ? mh1_cuda_channel_team_size_ : 0;
#endif
    return 0;
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::harmonic_team_size() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return mh1_fast_path ? mh1_cuda_harmonic_team_size_ : 0;
#endif
    return 0;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::supports_direct_node_reverse() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return mh1_fast_path&&mh1_direct_node_layout_
            &&mh1_input_1_angular_dimension_
                <=max_direct_node_input_components_;
#endif
    return false;
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,Kokkos::View<const Precision**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    ProfileRegion profile(mh1_fast_path
        ? "symmetrix/mh1/tensor_product/forward"
        : "symmetrix/generic/tensor_product/forward");
    ordered_kokkos_deep_copy(output,Precision(0)); auto mask=output_mask; auto fixed=internal_weights;
    if(mh1_fast_path) {
        auto plan=mh1_instruction_data;
        auto paths=mh1_path_weights;
        auto indices=mh1_sparse_indices;
        auto values=mh1_sparse_values;
        auto component_offsets=mh1_component_offsets;
        const int instruction_count=mh1_instruction_count_;
        const int multiplicity=mh1_multiplicity_;
#ifdef KOKKOS_ENABLE_CUDA
        if constexpr(std::is_same_v<Precision,float>
            &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        const int channel_team_size=mh1_cuda_channel_team_size_;
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
        return;
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
        return;
    }
    for (const auto instruction:instructions) { auto wigner=instruction.wigner;
        Kokkos::parallel_for("e3 tensor product",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(input_1.extent(0)),instruction.output_multiplicity,instruction.output_width}),KOKKOS_LAMBDA(int sample,int u,int c) {
            Precision result=Precision(0); for(int v=0;v<instruction.multiplicity_2;++v) { if(instruction.uuu&&u!=v) continue; const int wi=instruction.weight_offset+(instruction.uuu?u:u*instruction.multiplicity_2+v); const Precision weight=instruction.has_weight?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi)):Precision(1);
                for(int a=0;a<instruction.width_1;++a) for(int b=0;b<instruction.width_2;++b) result+=instruction.path_weight*weight*wigner((a*instruction.width_2+b)*instruction.output_width+c)*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b);
            } const int index=instruction.output_offset+u*instruction.output_width+c; output(sample,index)+=result*mask(index);
        });
    }
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,Kokkos::View<const Precision**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> weights_adjoint) const
{
    ProfileRegion profile(mh1_fast_path
        ? "symmetrix/mh1/tensor_product/reverse"
        : "symmetrix/generic/tensor_product/reverse");
    ordered_kokkos_deep_copy(input_1_adjoint,Precision(0));
    ordered_kokkos_deep_copy(input_2_adjoint,Precision(0));
    ordered_kokkos_deep_copy(weights_adjoint,Precision(0));
    auto mask=output_mask; auto fixed=internal_weights;
    if(mh1_fast_path) {
        auto plan=mh1_instruction_data;
        auto paths=mh1_path_weights;
        auto indices=mh1_sparse_indices;
        auto values=mh1_sparse_values;
        auto component_offsets=mh1_component_offsets;
        auto harmonic_offsets=mh1_harmonic_offsets;
        auto harmonic_terms=mh1_harmonic_terms;
        auto harmonic_values=mh1_harmonic_values;
        const int instruction_count=mh1_instruction_count_;
        const int multiplicity=mh1_multiplicity_;
#ifdef KOKKOS_ENABLE_CUDA
        if constexpr(std::is_same_v<Precision,float>
            &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        const int channel_team_size=mh1_cuda_channel_team_size_;
        const int harmonic_team_size=mh1_cuda_harmonic_team_size_;
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
        return;
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
        return;
    }
    for(const auto instruction:instructions) { auto wigner=instruction.wigner;
        Kokkos::parallel_for("e3 tensor reverse",input_1.extent(0),KOKKOS_LAMBDA(int sample) {
            for(int u=0;u<instruction.multiplicity_1;++u) for(int v=0;v<instruction.multiplicity_2;++v) { if(instruction.uuu&&u!=v) continue; const int wi=instruction.weight_offset+(instruction.uuu?u:u*instruction.multiplicity_2+v); const Precision weight=instruction.has_weight?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi)):Precision(1);
                for(int a=0;a<instruction.width_1;++a) for(int b=0;b<instruction.width_2;++b) for(int c=0;c<instruction.output_width;++c) { const int oi=instruction.output_offset+u*instruction.output_width+c; const Precision common=instruction.path_weight*wigner((a*instruction.width_2+b)*instruction.output_width+c)*mask(oi)*output_adjoint(sample,oi); input_1_adjoint(sample,instruction.input_1_offset+u*instruction.width_1+a)+=common*weight*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b); input_2_adjoint(sample,instruction.input_2_offset+v*instruction.width_2+b)+=common*weight*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a); if(instruction.has_weight) weights_adjoint(sample,wi)+=common*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b); }
            }
        });
    }
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
        auto plan=mh1_instruction_data;
        auto paths=mh1_path_weights;
        auto indices=mh1_sparse_indices;
        auto values=mh1_sparse_values;
        auto component_offsets=mh1_component_offsets;
        auto harmonic_offsets=mh1_harmonic_offsets;
        auto harmonic_terms=mh1_harmonic_terms;
        auto harmonic_values=mh1_harmonic_values;
        auto input_components=mh1_input_component_data;
        auto mask=output_mask;
        const int instruction_count=mh1_instruction_count_;
        const int multiplicity=mh1_multiplicity_;
        const int angular_dimension=mh1_input_1_angular_dimension_;

        using channel_policy=Kokkos::TeamPolicy<>;
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 direct node channels",
            channel_policy(samples,mh1_cuda_channel_team_size_),
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
                samples*harmonic_dimension,mh1_cuda_harmonic_team_size_),
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
    auto plan=mh1_instruction_data;
    auto paths=mh1_path_weights;
    auto indices=mh1_sparse_indices;
    auto values=mh1_sparse_values;
    auto component_offsets=mh1_component_offsets;
    auto mask=output_mask;
    const int instruction_count=mh1_instruction_count_;
    const int multiplicity=mh1_multiplicity_;
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
    auto plan=mh1_instruction_data;
    auto paths=mh1_path_weights;
    auto indices=mh1_sparse_indices;
    auto values=mh1_sparse_values;
    auto component_offsets=mh1_component_offsets;
    auto harmonic_offsets=mh1_harmonic_offsets;
    auto harmonic_terms=mh1_harmonic_terms;
    auto harmonic_values=mh1_harmonic_values;
    auto input_components=mh1_input_component_data;
    auto input_term_offsets=mh1_input_term_offsets;
    auto input_terms=mh1_input_terms;
    auto input_values=mh1_input_values;
    auto mask=output_mask;
    const int instruction_count=mh1_instruction_count_;
    const int multiplicity=mh1_multiplicity_;
    const int angular_dimension=mh1_input_1_angular_dimension_;
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
        ||mh1_max_output_width_>max_direct_node_output_components_) return false;
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
    auto plan=mh1_instruction_data;
    auto paths=mh1_path_weights;
    auto indices=mh1_sparse_indices;
    auto values=mh1_sparse_values;
    auto component_offsets=mh1_component_offsets;
    auto mask=output_mask;
    const int instruction_count=mh1_instruction_count_;
    const int multiplicity=mh1_multiplicity_;
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
    auto plan=mh1_instruction_data;
    auto paths=mh1_path_weights;
    auto indices=mh1_sparse_indices;
    auto values=mh1_sparse_values;
    auto component_offsets=mh1_component_offsets;
    auto harmonic_offsets=mh1_harmonic_offsets;
    auto harmonic_terms=mh1_harmonic_terms;
    auto harmonic_values=mh1_harmonic_values;
    auto input_components=mh1_input_component_data;
    auto input_instruction_term_offsets=mh1_input_instruction_term_offsets;
    auto input_terms=mh1_input_terms;
    auto input_values=mh1_input_values;
    auto mask=output_mask;
    const int instruction_count=mh1_instruction_count_;
    const int multiplicity=mh1_multiplicity_;
    const int angular_dimension=mh1_input_1_angular_dimension_;
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

template class E3LinearKokkosT<float>;
template class E3LinearKokkosT<double>;
template class E3TensorProductKokkosT<float>;
template class E3TensorProductKokkosT<double>;
