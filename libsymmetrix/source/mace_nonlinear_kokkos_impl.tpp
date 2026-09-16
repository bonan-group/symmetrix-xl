#include "mace_nonlinear_kokkos.hpp"
#include "mace_nonlinear_schema.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include "affine_mlp.hpp"
#include "device_backend.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"
#include "spherical_harmonic_device.hpp"
#include "tools_kokkos.hpp"

namespace {
class ScopedKokkosProfileRegion {
public:
    explicit ScopedKokkosProfileRegion(const char* name)
    {
        Kokkos::Profiling::pushRegion(name);
    }

    ~ScopedKokkosProfileRegion()
    {
        Kokkos::Profiling::popRegion();
    }

    ScopedKokkosProfileRegion(const ScopedKokkosProfileRegion&) = delete;
    ScopedKokkosProfileRegion& operator=(const ScopedKokkosProfileRegion&) = delete;
};

template<typename Value>
std::vector<Value> tensor_values(const nlohmann::json& value){return value.at("values").get<std::vector<Value>>();}
nlohmann::json load_model_json(const std::string& filename){std::ifstream stream(filename);if(!stream)throw std::runtime_error("Could not open MACE_Nonlinear Kokkos model file: "+filename);return nlohmann::json::parse(stream);}
const nlohmann::json& checked_interaction(const nlohmann::json& data){if(data.at("class").get<std::string>()!="RealAgnosticResidualNonLinearInteractionBlock")throw std::invalid_argument("MACE_Nonlinear Kokkos interaction class is unsupported.");return data;}
const nlohmann::json& readout_component(const nlohmann::json& data,int component){const auto readout_class=data.at("class").get<std::string>();if(readout_class=="LinearReadoutBlock")return data.at("linear");if(readout_class=="NonLinearReadoutBlock")return data.at(component==2?"linear_2":"linear_1");throw std::invalid_argument("MACE_Nonlinear Kokkos readout class is unsupported: "+readout_class);}

template<class View>
void ensure_view(View& view,View& storage,std::size_t extent)
{
    if(storage.extent(0)<extent)
        Kokkos::realloc(Kokkos::WithoutInitializing,storage,extent);
    view=Kokkos::subview(
        storage,std::make_pair(std::size_t(0),extent));
}

template<class View>
void ensure_view(View& view,View& storage,int first,int second)
{
    if(storage.extent(0)<static_cast<std::size_t>(first)
        ||storage.extent(1)!=static_cast<std::size_t>(second))
        Kokkos::realloc(Kokkos::WithoutInitializing,storage,first,second);
    view=Kokkos::subview(storage,std::make_pair(0,first),Kokkos::ALL);
}

inline SymmetrixJitMH1CudaNodeForwardArgsV4 device_node_forward_args(
    const SymmetrixJitMH1HostNodeForwardArgsV4& host,
    float* retained_pre_gate,float* retained_interaction_output)
{
    static_assert(sizeof(host)==offsetof(
        SymmetrixJitMH1CudaNodeForwardArgsV4,retained_pre_gate));
    SymmetrixJitMH1CudaNodeForwardArgsV4 result{};
    std::memcpy(&result,&host,sizeof(host));
    result.struct_size=sizeof(result);
    result.retained_pre_gate=retained_pre_gate;
    result.retained_interaction_output=retained_interaction_output;
    return result;
}

inline SymmetrixJitMH1CudaNodeReverseArgsV4 device_node_reverse_args(
    const SymmetrixJitMH1HostNodeReverseArgsV4& host,
    const float* retained_pre_gate,const float* retained_interaction_output)
{
    static_assert(sizeof(host)==offsetof(
        SymmetrixJitMH1CudaNodeReverseArgsV4,retained_pre_gate));
    SymmetrixJitMH1CudaNodeReverseArgsV4 result{};
    std::memcpy(&result,&host,sizeof(host));
    result.struct_size=sizeof(result);
    result.retained_pre_gate=retained_pre_gate;
    result.retained_interaction_output=retained_interaction_output;
    return result;
}

template<typename Value>
inline Value* offset_rows(Value* pointer,const int first,const int width,
    const std::size_t scalar_size=sizeof(Value))
{
    if(pointer==nullptr) return nullptr;
    return reinterpret_cast<Value*>(
        reinterpret_cast<std::uintptr_t>(pointer)
        +static_cast<std::size_t>(first)*width*scalar_size);
}

inline SymmetrixJitMH1CudaNodeForwardArgsV4 device_node_forward_tile_args(
    const SymmetrixJitMH1HostNodeForwardArgsV4& host,const int first,
    const int rows,const int input_width,const int up_width,
    const int message_width,const int output_width,
    const int retained_pre_gate_width,
    const int retained_interaction_output_width,
    float* retained_pre_gate,float* retained_interaction_output,
    const std::size_t scalar_size=sizeof(float))
{
    auto result=device_node_forward_args(
        host,retained_pre_gate,retained_interaction_output);
    result.num_nodes=rows;
    result.element_indices=offset_rows(result.element_indices,first,1);
    result.node_density=offset_rows(result.node_density,first,1,scalar_size);
    result.layer_input=offset_rows(
        result.layer_input,first,input_width,scalar_size);
    result.up=offset_rows(result.up,first,up_width,scalar_size);
    result.messages=offset_rows(
        result.messages,first,message_width,scalar_size);
    result.up_output=offset_rows(
        result.up_output,first,up_width,scalar_size);
    result.layer_output=offset_rows(
        result.layer_output,first,output_width,scalar_size);
    result.readout_contribution=
        offset_rows(result.readout_contribution,first,1,scalar_size);
    result.retained_pre_gate=
        offset_rows(result.retained_pre_gate,first,retained_pre_gate_width,
            scalar_size);
    result.retained_interaction_output=offset_rows(
        result.retained_interaction_output,first,
        retained_interaction_output_width,scalar_size);
    return result;
}

inline SymmetrixJitMH1CudaNodeReverseArgsV4 device_node_reverse_tile_args(
    const SymmetrixJitMH1HostNodeReverseArgsV4& host,const int first,
    const int rows,const int input_width,const int up_width,
    const int message_width,const int output_width,
    const int retained_pre_gate_width,
    const int retained_interaction_output_width,
    const float* retained_pre_gate,const float* retained_interaction_output,
    const std::size_t scalar_size=sizeof(float))
{
    auto result=device_node_reverse_args(
        host,retained_pre_gate,retained_interaction_output);
    result.num_nodes=rows;
    result.element_indices=offset_rows(result.element_indices,first,1);
    result.node_density=offset_rows(result.node_density,first,1,scalar_size);
    result.layer_input=offset_rows(
        result.layer_input,first,input_width,scalar_size);
    result.up=offset_rows(result.up,first,up_width,scalar_size);
    result.messages=offset_rows(
        result.messages,first,message_width,scalar_size);
    result.layer_output=offset_rows(
        result.layer_output,first,output_width,scalar_size);
    result.layer_output_adjoint=
        offset_rows(result.layer_output_adjoint,first,output_width,scalar_size);
    result.message_adjoint=
        offset_rows(result.message_adjoint,first,message_width,scalar_size);
    result.up_adjoint=offset_rows(
        result.up_adjoint,first,up_width,scalar_size);
    result.layer_input_adjoint=
        offset_rows(result.layer_input_adjoint,first,input_width,scalar_size);
    result.node_density_adjoint=
        offset_rows(result.node_density_adjoint,first,1,scalar_size);
    result.retained_pre_gate=
        offset_rows(result.retained_pre_gate,first,retained_pre_gate_width,
            scalar_size);
    result.retained_interaction_output=offset_rows(
        result.retained_interaction_output,first,
        retained_interaction_output_width,scalar_size);
    return result;
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
void check_execution_mh1_cuda_status(
    const std::int32_t status,const char* operation)
{
    DeviceBackendTraits<Kokkos::DefaultExecutionSpace>::check_status(
        status,operation);
}

using ExecutionMH1CudaDeviceGuard=
    typename DeviceBackendTraits<Kokkos::DefaultExecutionSpace>::DeviceGuard;

int execution_mh1_device_ordinal(
    const Kokkos::DefaultExecutionSpace& execution_space)
{
    return DeviceBackendTraits<Kokkos::DefaultExecutionSpace>::device_ordinal(
        execution_space);
}

void* execution_mh1_device_stream(
    const Kokkos::DefaultExecutionSpace& execution_space)
{
    return reinterpret_cast<void*>(
        DeviceBackendTraits<Kokkos::DefaultExecutionSpace>::native_stream(
            execution_space));
}

int execution_mh1_persistent_blocks()
{
#ifdef KOKKOS_ENABLE_HIP
    constexpr int blocks_per_compute_unit=8;
#else
    constexpr int blocks_per_compute_unit=4;
#endif
    return std::max(
        1,blocks_per_compute_unit
            *execution_device_execution_environment().compute_unit_count);
}

int execution_mh1_spline_reverse_persistent_blocks()
{
    constexpr int blocks_per_compute_unit=8;
    return std::max(
        1,blocks_per_compute_unit
            *execution_device_execution_environment().compute_unit_count);
}
#endif
}

template<typename Precision>
struct MaceNonlinearKokkosT<Precision>::SphericalHarmonicsState {
#ifdef SYMMETRIX_SPHERICART_CUDA
    explicit SphericalHarmonicsState(int l_max) : calculator(l_max) {}
    ~SphericalHarmonicsState() noexcept
    {
        try {
            calculator.release_device_resources();
        } catch (...) {
            // Evaluator teardown must not propagate CUDA cleanup errors.
        }
    }
    sphericart::cuda::SphericalHarmonics<Precision> calculator;
#else
    explicit SphericalHarmonicsState(int l_max) : calculator(l_max) {}
    sphericart::SphericalHarmonics<Precision> calculator;
#endif
};

#if SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART == 1

template<typename Precision>
MaceNonlinearKokkosT<Precision>::~MaceNonlinearKokkosT()
{
    Kokkos::fence("MaceNonlinearKokkosT teardown");
    spherical_harmonics_state.reset();
}

template<typename Precision>
MaceNonlinearKokkosT<Precision>::Gate::Gate(const nlohmann::json& data)
{
    if(data.at("scalar_activation").get<std::string>()!="silu"||data.at("gate_activation").get<std::string>()!="sigmoid")throw std::invalid_argument("MACE_Nonlinear Kokkos gate has unsupported activations.");
    Irreps input(data.at("irreps_in").get<std::string>()),scalars(data.at("irreps_scalars").get<std::string>()),gates(data.at("irreps_gates").get<std::string>()),gated(data.at("irreps_gated").get<std::string>()),output(data.at("irreps_out").get<std::string>());
    input_size=input.dimension();scalar_size=scalars.dimension(); gate_size=gates.dimension(); gated_size=gated.dimension(); output_size=output.dimension(); scalar_blocks=scalars.blocks; gated_blocks=gated.blocks;
    scalar_constants=data.at("scalar_activation_constants").get<std::vector<Precision>>(); gate_constants=data.at("gate_activation_constants").get<std::vector<Precision>>();
    if(scalar_constants.size()!=scalar_blocks.size()||gate_constants.size()!=gated_blocks.size())throw std::invalid_argument("MACE_Nonlinear Kokkos gate activation counts are inconsistent.");
    int planned_gates=0,planned_gated=0;
    for(const auto& block:gated_blocks) {
        planned_gates+=block.multiplicity;
        planned_gated+=block.dimension();
    }
    if(input_size!=scalar_size+gate_size+gated_size
        ||output_size!=scalar_size+gated_size
        ||planned_gates!=gate_size||planned_gated!=gated_size)
        throw std::invalid_argument("MACE_Nonlinear Kokkos gate irreps are inconsistent.");

    fused_scalar_constants=Kokkos::View<Precision*>(
        "nonlinear fused gate scalar constants",scalar_size);
    auto scalar_constant_host=Kokkos::create_mirror_view(fused_scalar_constants);
    for(int block_index=0;block_index<static_cast<int>(scalar_blocks.size());++block_index) {
        const auto block=scalar_blocks[block_index];
        for(int index=0;index<block.dimension();++index)
            scalar_constant_host(block.offset+index)=scalar_constants[block_index];
    }
    Kokkos::deep_copy(fused_scalar_constants,scalar_constant_host);

    fused_gate_plan=Kokkos::View<int**,Kokkos::LayoutRight>(
        "nonlinear fused gate reverse plan",gate_size,7);
    fused_gate_constants=Kokkos::View<Precision*>(
        "nonlinear fused gate constants",gate_size);
    auto plan_host=Kokkos::create_mirror_view(fused_gate_plan);
    auto gate_constant_host=Kokkos::create_mirror_view(fused_gate_constants);
    int gate_offset=scalar_size,gated_offset=scalar_size+gate_size;
    int output_offset=scalar_size,gate_feature=0;
    for(int block_index=0;block_index<static_cast<int>(gated_blocks.size());++block_index) {
        const auto block=gated_blocks[block_index];
        const int width=2*block.l+1;
        const int packed_gated_block=gated_offset;
        const int packed_output_block=output_offset;
        for(int feature=0;feature<block.multiplicity;++feature) {
            plan_host(gate_feature,0)=gate_offset+feature;
            plan_host(gate_feature,1)=gated_offset+feature*width;
            plan_host(gate_feature,2)=output_offset+feature*width;
            plan_host(gate_feature,3)=width;
            plan_host(gate_feature,4)=packed_gated_block;
            plan_host(gate_feature,5)=packed_output_block;
            plan_host(gate_feature,6)=feature;
            gate_constant_host(gate_feature)=gate_constants[block_index];
            ++gate_feature;
        }
        gate_offset+=block.multiplicity;
        gated_offset+=block.dimension();
        output_offset+=block.dimension();
    }
    Kokkos::deep_copy(fused_gate_plan,plan_host);
    Kokkos::deep_copy(fused_gate_constants,gate_constant_host);

    fused_packed_input_plan=Kokkos::View<int**,Kokkos::LayoutRight>(
        "nonlinear packed gate input plan",input_size,4);
    auto packed_input_plan_host=Kokkos::create_mirror_view(
        fused_packed_input_plan);
    for(const auto& block:input.blocks) {
        const int width=2*block.l+1;
        for(int channel=0;channel<block.multiplicity;++channel)
            for(int component=0;component<width;++component) {
                const int index=block.offset+channel*width+component;
                packed_input_plan_host(index,0)=block.offset;
                packed_input_plan_host(index,1)=channel;
                packed_input_plan_host(index,2)=width;
                packed_input_plan_host(index,3)=component;
            }
    }
    Kokkos::deep_copy(fused_packed_input_plan,packed_input_plan_host);
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::Gate::supports_fused_normalized_reverse() const
{
    return input_size>0&&input_size==scalar_size+gate_size+gated_size
        &&output_size==scalar_size+gated_size
        &&fused_scalar_constants.extent(0)==static_cast<std::size_t>(scalar_size)
        &&fused_gate_plan.extent(0)==static_cast<std::size_t>(gate_size)
        &&fused_gate_plan.extent(1)>=4
        &&fused_gate_constants.extent(0)==static_cast<std::size_t>(gate_size)
        &&fused_packed_input_plan.extent(0)
            ==static_cast<std::size_t>(input_size)
        &&fused_packed_input_plan.extent(1)==4;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Gate::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    for(int block_index=0;block_index<static_cast<int>(scalar_blocks.size());++block_index){const auto block=scalar_blocks[block_index];const Precision constant=scalar_constants[block_index];Kokkos::parallel_for("nonlinear gate scalars",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{static_cast<int>(input.extent(0)),block.dimension()}),KOKKOS_LAMBDA(int sample,int index){const int offset=block.offset+index;const Precision x=input(sample,offset);output(sample,offset)=constant*x/(Precision(1)+Kokkos::exp(-x));});}
    int gate_offset=scalar_size,gated_offset=scalar_size+gate_size,output_offset=scalar_size;
    for(int block_index=0;block_index<static_cast<int>(gated_blocks.size());++block_index){const auto block=gated_blocks[block_index];const int width=2*block.l+1;const int go=gate_offset,gio=gated_offset,oo=output_offset;const Precision constant=gate_constants[block_index];
        Kokkos::parallel_for("nonlinear gate tensors",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(input.extent(0)),block.multiplicity,width}),KOKKOS_LAMBDA(int sample,int feature,int component){const Precision probability=Precision(1)/(Precision(1)+Kokkos::exp(-input(sample,go+feature)));output(sample,oo+feature*width+component)=constant*probability*input(sample,gio+feature*width+component);});
        gate_offset+=block.multiplicity;gated_offset+=block.dimension();output_offset+=block.dimension();}
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Gate::evaluate_normalized_to_packed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> linear_forward,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> residual,
    Kokkos::View<const int*> residual_active,
    Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<Precision**,Kokkos::LayoutRight> packed_output,
    const bool linear_is_normalized,
    Kokkos::View<const Precision*> normalization_inverse) const
{
    const int samples=input.extent(0);
    if(!supports_fused_normalized_reverse()
        ||linear_forward.size()!=static_cast<std::size_t>(samples)*input_size
        ||residual.extent(0)!=input.extent(0)
        ||residual.extent(1)!=static_cast<std::size_t>(input_size)
        ||residual_active.extent(0)!=static_cast<std::size_t>(input_size)
        ||input.extent(1)!=static_cast<std::size_t>(input_size)
        ||packed_output.size()!=static_cast<std::size_t>(samples)*output_size
        ||(normalization_inverse.extent(0)!=0
            &&normalization_inverse.extent(0)!=static_cast<std::size_t>(samples)))
        throw std::invalid_argument(
            "MACE_Nonlinear packed normalized gate dimensions are inconsistent.");
    auto scalar_constants_device=fused_scalar_constants;
    auto plan=fused_gate_plan;
    auto gate_constants_device=fused_gate_constants;
    const auto linear_data=linear_forward.data();
    const auto residual_data=residual.data();
    const bool has_inverse=normalization_inverse.extent(0)!=0;
    auto input_data=input.data();
    auto packed_output_data=packed_output.data();
    const int scalar_count=scalar_size;
    Kokkos::parallel_for("mh1 packed normalized gate scalars",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{scalar_count,samples}),
        KOKKOS_LAMBDA(int feature,int node) {
            const Precision linear_value=linear_data[
                static_cast<std::size_t>(feature)*samples+node];
            const Precision inverse=linear_is_normalized?Precision(1):
                (has_inverse?normalization_inverse(node):
                    Precision(1)/(alpha+beta*densities(node)));
            const Precision normalized=linear_value*inverse
                +(residual_active(feature)
                    ?residual_data[static_cast<std::size_t>(feature)*samples+node]
                    :Precision(0));
            const std::size_t packed_index=
                static_cast<std::size_t>(feature)*samples+node;
            input_data[packed_index]=normalized;
            packed_output_data[packed_index]=
                scalar_constants_device(feature)
                *normalized/(Precision(1)+Kokkos::exp(-normalized));
        });
    if constexpr(std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>)
        Kokkos::parallel_for("mh1 packed normalized gate tensors host",
            gate_size,KOKKOS_LAMBDA(int feature) {
            const int width=plan(feature,3);
            const int gate_index=plan(feature,0);
            const int gated_index=plan(feature,1);
            const int packed_gated=plan(feature,4);
            const int packed_output_block=plan(feature,5);
            const int channel=plan(feature,6);
            const Precision constant=gate_constants_device(feature);
            const bool gate_has_residual=residual_active(gate_index)!=0;
            unsigned long long gated_residual_mask=0;
            if(width<=64)
                for(int component=0;component<width;++component)
                    gated_residual_mask|=
                        static_cast<unsigned long long>(
                            residual_active(gated_index+component)!=0)
                        << component;
            for(int node=0;node<samples;++node) {
                const Precision inverse=linear_is_normalized?Precision(1):
                    (has_inverse?normalization_inverse(node):
                        Precision(1)/(alpha+beta*densities(node)));
                const std::size_t packed_gate_index=
                    static_cast<std::size_t>(gate_index)*samples+node;
                input_data[packed_gate_index]=
                    linear_data[packed_gate_index]*inverse
                    +(gate_has_residual?residual_data[
                        packed_gate_index]:Precision(0));
                const Precision probability=Precision(1)/(Precision(1)
                    +Kokkos::exp(-input_data[packed_gate_index]));
                // Reverse needs the sigmoid probability, not the pre-activation.
                input_data[packed_gate_index]=probability;
                const Precision gate=constant*probability;
                for(int component=0;component<width;++component) {
                    const int local_gated_index=gated_index+component;
                    const std::size_t packed_gated_index=
                        static_cast<std::size_t>(samples)*packed_gated
                        +static_cast<std::size_t>(channel)*samples*width
                        +node*width+component;
                    const bool component_has_residual=width<=64
                        ?((gated_residual_mask>>component)&1u)!=0
                        :residual_active(local_gated_index)!=0;
                    const Precision value=linear_data[packed_gated_index]*inverse
                        +(component_has_residual
                            ?residual_data[packed_gated_index]:Precision(0));
                    input_data[packed_gated_index]=value;
                    const std::size_t packed_output_index=
                        static_cast<std::size_t>(samples)*packed_output_block
                        +static_cast<std::size_t>(channel)*samples*width
                        +node*width+component;
                    packed_output_data[packed_output_index]=gate*value;
                }
            }
        });
    else Kokkos::parallel_for("mh1 packed normalized gate tensors",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{gate_size,samples}),
        KOKKOS_LAMBDA(int feature,int node) {
            const int width=plan(feature,3);
            const int gate_index=plan(feature,0);
            const int gated_index=plan(feature,1);
            const int packed_gated=plan(feature,4);
            const int packed_output_block=plan(feature,5);
            const int channel=plan(feature,6);
            const Precision inverse=linear_is_normalized?Precision(1):
                (has_inverse?normalization_inverse(node):
                    Precision(1)/(alpha+beta*densities(node)));
            const std::size_t packed_gate_index=
                static_cast<std::size_t>(gate_index)*samples+node;
            input_data[packed_gate_index]=linear_data[packed_gate_index]*inverse
                +(residual_active(gate_index)?residual_data[
                    packed_gate_index]
                    :Precision(0));
            const Precision probability=Precision(1)/(Precision(1)
                +Kokkos::exp(-input_data[packed_gate_index]));
            // Reverse needs the sigmoid probability, not the gate pre-activation.
            input_data[packed_gate_index]=probability;
            const Precision gate=gate_constants_device(feature)*probability;
            for(int component=0;component<width;++component) {
                const int local_gated_index=gated_index+component;
                const std::size_t packed_gated_index=
                    static_cast<std::size_t>(samples)*packed_gated
                    +static_cast<std::size_t>(channel)*samples*width
                    +node*width+component;
                const Precision value=linear_data[packed_gated_index]*inverse
                    +(residual_active(local_gated_index)
                        ?residual_data[packed_gated_index]:Precision(0));
                input_data[packed_gated_index]=value;
                const std::size_t packed_output_index=
                    static_cast<std::size_t>(samples)*packed_output_block
                    +static_cast<std::size_t>(channel)*samples*width
                    +node*width+component;
                packed_output_data[packed_output_index]=gate*value;
            }
        });
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Gate::reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const
{
    ordered_kokkos_deep_copy(input_adjoint,Precision(0));for(int block_index=0;block_index<static_cast<int>(scalar_blocks.size());++block_index){const auto block=scalar_blocks[block_index];const Precision constant=scalar_constants[block_index];Kokkos::parallel_for("reverse gate scalars",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{static_cast<int>(input.extent(0)),block.dimension()}),KOKKOS_LAMBDA(int sample,int index){const int offset=block.offset+index;const Precision x=input(sample,offset),probability=Precision(1)/(Precision(1)+Kokkos::exp(-x));input_adjoint(sample,offset)=output_adjoint(sample,offset)*constant*(probability+x*probability*(Precision(1)-probability));});}
    int gate_offset=scalar_size,gated_offset=scalar_size+gate_size,output_offset=scalar_size;
    for(int block_index=0;block_index<static_cast<int>(gated_blocks.size());++block_index){const auto block=gated_blocks[block_index];const int width=2*block.l+1,go=gate_offset,gio=gated_offset,oo=output_offset;const Precision constant=gate_constants[block_index];
        Kokkos::parallel_for("reverse gate tensors",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{static_cast<int>(input.extent(0)),block.multiplicity}),KOKKOS_LAMBDA(int sample,int feature){const Precision probability=Precision(1)/(Precision(1)+Kokkos::exp(-input(sample,go+feature))),gate=constant*probability;Precision gate_adjoint=Precision(0);for(int component=0;component<width;++component){input_adjoint(sample,gio+feature*width+component)=gate*output_adjoint(sample,oo+feature*width+component);gate_adjoint+=input(sample,gio+feature*width+component)*output_adjoint(sample,oo+feature*width+component);}input_adjoint(sample,go+feature)=gate_adjoint*constant*probability*(Precision(1)-probability);});
        gate_offset+=block.multiplicity;gated_offset+=block.dimension();output_offset+=block.dimension();}
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Gate::reverse_normalized(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> linear_forward,
    Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> linear_adjoint,
    Kokkos::View<Precision*> density_adjoint) const
{
    if(!supports_fused_normalized_reverse()
        ||input.extent(1)!=static_cast<std::size_t>(input_size)
        ||output_adjoint.extent(0)!=input.extent(0)
        ||output_adjoint.extent(1)!=static_cast<std::size_t>(output_size)
        ||linear_forward.extent(0)!=input.extent(0)
        ||linear_forward.extent(1)!=input.extent(1)
        ||densities.extent(0)!=input.extent(0)
        ||input_adjoint.extent(0)!=input.extent(0)
        ||input_adjoint.extent(1)!=input.extent(1)
        ||linear_adjoint.extent(0)!=input.extent(0)
        ||linear_adjoint.extent(1)!=input.extent(1)
        ||density_adjoint.extent(0)!=input.extent(0))
        throw std::invalid_argument(
            "MACE_Nonlinear fused gate-normalization reverse dimensions are inconsistent.");

    auto scalar_activation_constants=fused_scalar_constants;
    auto gate_plan=fused_gate_plan;
    auto gated_activation_constants=fused_gate_constants;
    const int scalar_count=scalar_size;
    const int work_items=scalar_size+gate_size;
    // Execution-space concurrency is device-wide on accelerators, not a
    // valid per-team size. Keep teams within the portable launch budget.
    constexpr int maximum_team_size=128;
    const int team_size=std::max(1,std::min({work_items,maximum_team_size,
        Kokkos::DefaultExecutionSpace().concurrency()}));
    using team_policy=Kokkos::TeamPolicy<>;
    Kokkos::parallel_for(
        "mh1 fused reverse gate normalization",
        team_policy(input.extent(0),team_size),
        KOKKOS_LAMBDA(const typename team_policy::member_type& team) {
            const int node=team.league_rank();
            const Precision normalization=alpha+beta*densities(node);
            const Precision inverse=Precision(1)/normalization;
            const Precision density_scale=-beta*inverse*inverse;
            Precision density_value=Precision(0);
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team,work_items),
                [&](const int item,Precision& update) {
                    if(item<scalar_count) {
                        const Precision x=input(node,item);
                        const Precision probability=Precision(1)
                            /(Precision(1)+Kokkos::exp(-x));
                        const Precision value=output_adjoint(node,item)
                            *scalar_activation_constants(item)
                            *(probability+x*probability*(Precision(1)-probability));
                        input_adjoint(node,item)=value;
                        linear_adjoint(node,item)=value*inverse;
                        update+=density_scale*value*linear_forward(node,item);
                        return;
                    }

                    const int feature=item-scalar_count;
                    const int gate_index=gate_plan(feature,0);
                    const int gated_index=gate_plan(feature,1);
                    const int output_index=gate_plan(feature,2);
                    const int width=gate_plan(feature,3);
                    const Precision x=input(node,gate_index);
                    const Precision probability=Precision(1)
                        /(Precision(1)+Kokkos::exp(-x));
                    const Precision constant=gated_activation_constants(feature);
                    const Precision gate=constant*probability;
                    Precision gate_adjoint=Precision(0);
                    for(int component=0;component<width;++component) {
                        const int local_input=gated_index+component;
                        const Precision output_value=output_adjoint(
                            node,output_index+component);
                        const Precision value=gate*output_value;
                        input_adjoint(node,local_input)=value;
                        linear_adjoint(node,local_input)=value*inverse;
                        update+=density_scale*value*linear_forward(node,local_input);
                        gate_adjoint+=input(node,local_input)*output_value;
                    }
                    const Precision gate_value=gate_adjoint*constant*probability
                        *(Precision(1)-probability);
                    input_adjoint(node,gate_index)=gate_value;
                    linear_adjoint(node,gate_index)=gate_value*inverse;
                    update+=density_scale*gate_value*linear_forward(node,gate_index);
                },density_value);
            Kokkos::single(Kokkos::PerTeam(team),[&]() {
                density_adjoint(node)=density_value;
            });
        });
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Gate::reverse_normalized_packed(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_output_adjoint,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_linear_forward,
    Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> packed_linear_adjoint,
    Kokkos::View<Precision*> density_adjoint,
    const bool linear_is_normalized) const
{
    const int samples=input.extent(0);
    if(!supports_fused_normalized_reverse()
        ||input.extent(1)!=static_cast<std::size_t>(input_size)
        ||packed_output_adjoint.size()
            !=static_cast<std::size_t>(samples)*output_size
        ||packed_linear_forward.size()
            !=static_cast<std::size_t>(samples)*input_size
        ||input_adjoint.extent(0)!=input.extent(0)
        ||input_adjoint.extent(1)!=input.extent(1)
        ||packed_linear_adjoint.size()!=packed_linear_forward.size()
        ||density_adjoint.extent(0)!=input.extent(0))
        throw std::invalid_argument(
            "MACE_Nonlinear packed gate reverse dimensions are inconsistent.");
    auto scalar_constants_device=fused_scalar_constants;
    auto plan=fused_gate_plan;
    auto input_plan=fused_packed_input_plan;
    auto gate_constants_device=fused_gate_constants;
    const auto output_adjoint_data=packed_output_adjoint.data();
    const auto linear_forward_data=packed_linear_forward.data();
    const auto input_data=input.data();
    auto input_adjoint_data=input_adjoint.data();
    auto linear_adjoint_data=packed_linear_adjoint.data();
    const bool shared_normalized_adjoint=linear_is_normalized
        &&input_adjoint_data==linear_adjoint_data;
    const int packed_input_size=input_size;
    Kokkos::parallel_for("mh1 packed reverse gate scalars",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{scalar_size,samples}),
        KOKKOS_LAMBDA(int item,int node) {
            const Precision x=input_data[
                static_cast<std::size_t>(item)*samples+node];
            const Precision probability=Precision(1)
                /(Precision(1)+Kokkos::exp(-x));
            const Precision value=output_adjoint_data[
                static_cast<std::size_t>(item)*samples+node]
                *scalar_constants_device(item)
                *(probability+x*probability*(Precision(1)-probability));
            input_adjoint_data[static_cast<std::size_t>(item)*samples+node]=
                value;
            if(!shared_normalized_adjoint)
                linear_adjoint_data[
                    static_cast<std::size_t>(item)*samples+node]=
                    linear_is_normalized?value:
                    value/(alpha+beta*densities(node));
        });
    if constexpr(std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>)
        Kokkos::parallel_for("mh1 packed reverse gate tensors host",
            gate_size,KOKKOS_LAMBDA(int feature) {
            const int gate_index=plan(feature,0);
            const int width=plan(feature,3);
            const int packed_gated=plan(feature,4);
            const int packed_output=plan(feature,5);
            const int channel=plan(feature,6);
            const Precision constant=gate_constants_device(feature);
            for(int node=0;node<samples;++node) {
                const std::size_t packed_gate_index=
                    static_cast<std::size_t>(gate_index)*samples+node;
                const Precision probability=input_data[packed_gate_index];
                const Precision gate=constant*probability;
                const Precision inverse=linear_is_normalized?Precision(1):
                    Precision(1)/(alpha+beta*densities(node));
                Precision gate_adjoint=Precision(0);
                for(int component=0;component<width;++component) {
                    const std::size_t linear_index=
                        static_cast<std::size_t>(samples)*packed_gated
                        +static_cast<std::size_t>(channel)*samples*width
                        +node*width+component;
                    const std::size_t output_index=
                        static_cast<std::size_t>(samples)*packed_output
                        +static_cast<std::size_t>(channel)*samples*width
                        +node*width+component;
                    const Precision output_value=
                        output_adjoint_data[output_index];
                    const Precision value=gate*output_value;
                    input_adjoint_data[linear_index]=value;
                    if(!shared_normalized_adjoint)
                        linear_adjoint_data[linear_index]=value*inverse;
                    gate_adjoint+=input_data[linear_index]*output_value;
                }
                const Precision gate_value=gate_adjoint*constant*probability
                    *(Precision(1)-probability);
                input_adjoint_data[packed_gate_index]=gate_value;
                if(!shared_normalized_adjoint)
                    linear_adjoint_data[packed_gate_index]=gate_value*inverse;
            }
        });
    else Kokkos::parallel_for("mh1 packed reverse gate tensors",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{gate_size,samples}),
        KOKKOS_LAMBDA(int feature,int node) {
            const int gate_index=plan(feature,0);
            const int width=plan(feature,3);
            const int packed_gated=plan(feature,4);
            const int packed_output=plan(feature,5);
            const int channel=plan(feature,6);
            const std::size_t packed_gate_index=
                static_cast<std::size_t>(gate_index)*samples+node;
            const Precision probability=input_data[packed_gate_index];
            const Precision constant=gate_constants_device(feature);
            const Precision gate=constant*probability;
            const Precision inverse=linear_is_normalized?Precision(1):
                Precision(1)/(alpha+beta*densities(node));
            Precision gate_adjoint=Precision(0);
            for(int component=0;component<width;++component) {
                const std::size_t linear_index=
                    static_cast<std::size_t>(samples)*packed_gated
                    +static_cast<std::size_t>(channel)*samples*width
                    +node*width+component;
                const std::size_t output_index=
                    static_cast<std::size_t>(samples)*packed_output
                    +static_cast<std::size_t>(channel)*samples*width
                    +node*width+component;
                const Precision output_value=output_adjoint_data[output_index];
                const Precision value=gate*output_value;
                input_adjoint_data[linear_index]=value;
                if(!shared_normalized_adjoint)
                    linear_adjoint_data[linear_index]=value*inverse;
                gate_adjoint+=input_data[linear_index]*output_value;
            }
            const Precision gate_value=gate_adjoint*constant*probability
                *(Precision(1)-probability);
            input_adjoint_data[
                static_cast<std::size_t>(gate_index)*samples+node]=gate_value;
            if(!shared_normalized_adjoint)
                linear_adjoint_data[
                    static_cast<std::size_t>(gate_index)*samples+node]=
                    gate_value*inverse;
        });
    if(linear_is_normalized)
        return;
    Kokkos::parallel_for("mh1 packed reverse density",samples,
        KOKKOS_LAMBDA(int node) {
            Precision value=Precision(0);
            const Precision normalization=alpha+beta*densities(node);
            const Precision scale=-beta/(normalization*normalization);
            for(int index=0;index<packed_input_size;++index) {
                const int block_offset=input_plan(index,0);
                const int channel=input_plan(index,1);
                const int width=input_plan(index,2);
                const int component=input_plan(index,3);
                const std::size_t packed_index=
                    static_cast<std::size_t>(samples)*block_offset
                    +static_cast<std::size_t>(channel)*samples*width
                    +node*width+component;
                value+=scale*input_adjoint_data[packed_index]
                    *linear_forward_data[packed_index];
            }
            density_adjoint(node)=value;
        });
}

template<typename Precision>
MaceNonlinearKokkosT<Precision>::Interaction::Interaction(const nlohmann::json& data)
    :source_embedding(checked_interaction(data).at("source_embedding")),target_embedding(data.at("target_embedding")),linear_up(data.at("linear_up")),skip(data.at("skip_tp")),linear_res(data.at("linear_res")),linear_1(data.at("linear_1")),linear_2(data.at("linear_2")),convolution(data.at("conv_tp")),convolution_weights(data.at("conv_tp_weights")),density(data.at("density_fn")),gate(data.at("gate")),alpha(data.at("alpha").get<Precision>()),beta(data.at("beta").get<Precision>()){}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Interaction::set_e3_linear_backend(
    const std::string& backend)
{
    source_embedding.set_backend(backend);
    target_embedding.set_backend(backend);
    linear_up.set_backend(backend);
    skip.set_backend(backend);
    linear_res.set_backend(backend);
    linear_1.set_backend(backend);
    linear_2.set_backend(backend);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Interaction::set_e3_linear_workspace(
    const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace)
{
    source_embedding.set_workspace(workspace);
    target_embedding.set_workspace(workspace);
    linear_up.set_workspace(workspace);
    skip.set_workspace(workspace);
    linear_res.set_workspace(workspace);
    linear_1.set_workspace(workspace);
    linear_2.set_workspace(workspace);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Interaction::set_conditioned_mlp_workspace(
    const std::shared_ptr<typename AffineMLPKokkosT<Precision>::Workspace>&
        workspace)
{
    convolution_weights.set_workspace(workspace);
    density.set_workspace(workspace);
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::Interaction::linear_workspace_bytes() const
{
    return source_embedding.workspace_bytes()+target_embedding.workspace_bytes()
        +linear_up.workspace_bytes()+skip.workspace_bytes()
        +linear_res.workspace_bytes()+linear_1.workspace_bytes()
        +linear_2.workspace_bytes();
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::Interaction::prepare_pair_conditioning(
    const nlohmann::json& data,int radial_size,int model_element_count,
    const std::vector<int>& selected_model_indices)
{
    AffineMLP convolution_mlp(data.at("conv_tp_weights"));
    AffineMLP density_mlp(data.at("density_fn"));
    E3Linear source_linear(data.at("source_embedding"));
    E3Linear target_linear(data.at("target_embedding"));
    const int source_width=source_linear.output_dimension();
    const int target_width=target_linear.output_dimension();
    const int conditioned_input_size=radial_size+source_width+target_width;
    if(source_linear.input_dimension()!=model_element_count
        ||target_linear.input_dimension()!=model_element_count
        ||source_width<=0||target_width<=0
        ||convolution_mlp.input_size()!=conditioned_input_size
        ||density_mlp.input_size()!=conditioned_input_size
        ||!convolution_mlp.supports_conditioned_input(radial_size)
        ||!density_mlp.supports_conditioned_input(radial_size))
        return false;
    const int types=selected_model_indices.size();
    const int convolution_width=data.at("conv_tp_weights").at("layers").at(0)
        .at("weight").at("shape").at(0).get<int>();
    const int density_width=data.at("density_fn").at("layers").at(0)
        .at("weight").at("shape").at(0).get<int>();
    std::vector<Precision> convolution_source(types*convolution_width);
    std::vector<Precision> convolution_target(types*convolution_width);
    std::vector<Precision> density_source(types*density_width);
    std::vector<Precision> density_target(types*density_width);
    const int target_offset=radial_size+source_width;
    for(int type=0;type<types;++type) {
        std::vector<double> attrs(model_element_count,Precision(0));
        attrs.at(selected_model_indices.at(type))=Precision(1);
        const auto source=source_linear.evaluate(attrs);
        const auto target=target_linear.evaluate(attrs);
        const auto convolution_source_row=
            convolution_mlp.first_layer_contribution(radial_size,source);
        const auto convolution_target_row=
            convolution_mlp.first_layer_contribution(target_offset,target);
        const auto density_source_row=
            density_mlp.first_layer_contribution(radial_size,source);
        const auto density_target_row=
            density_mlp.first_layer_contribution(target_offset,target);
        std::copy(convolution_source_row.begin(),convolution_source_row.end(),
            convolution_source.begin()+type*convolution_width);
        std::copy(convolution_target_row.begin(),convolution_target_row.end(),
            convolution_target.begin()+type*convolution_width);
        std::copy(density_source_row.begin(),density_source_row.end(),
            density_source.begin()+type*density_width);
        std::copy(density_target_row.begin(),density_target_row.end(),
            density_target.begin()+type*density_width);
    }
    set_kokkos_view(
        convolution_source_contributions,convolution_source,types,convolution_width);
    set_kokkos_view(
        convolution_target_contributions,convolution_target,types,convolution_width);
    set_kokkos_view(density_source_contributions,density_source,types,density_width);
    set_kokkos_view(density_target_contributions,density_target,types,density_width);
    return true;
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::Interaction::prepare_pair_spline(
    const nlohmann::json& data,const nlohmann::json& radial_data,
    const double cutoff,const int radial_size,const int model_element_count,
    const std::vector<int>& selected_model_indices,
    const std::vector<int>& model_atomic_numbers,const int spline_nodes)
{
    if(spline_nodes<4||!(cutoff>0.0)
        ||selected_model_indices.empty()
        ||static_cast<int>(model_atomic_numbers.size())!=model_element_count)
        return false;
    AffineMLP convolution_mlp(data.at("conv_tp_weights"));
    AffineMLP density_mlp(data.at("density_fn"));
    E3Linear source_linear(data.at("source_embedding"));
    E3Linear target_linear(data.at("target_embedding"));
    const int source_width=source_linear.output_dimension();
    const int target_width=target_linear.output_dimension();
    if(source_linear.input_dimension()!=model_element_count
        ||target_linear.input_dimension()!=model_element_count
        ||convolution_mlp.output_size()!=convolution.weight_size()
        ||density_mlp.output_size()!=1
        ||!convolution_mlp.supports_conditioned_input(radial_size)
        ||!density_mlp.supports_conditioned_input(radial_size))
        return false;

    const auto& basis=radial_data.at("basis");
    const auto bessel=tensor_values<double>(basis.at("weights"));
    if(static_cast<int>(bessel.size())!=radial_size)return false;
    const double prefactor=basis.at("prefactor").get<double>();
    const int cutoff_degree=radial_data.at("cutoff").at("p").get<int>();
    const bool embed_cutoff=radial_data.at("apply_cutoff").get<bool>();
    const auto& transform=radial_data.at("distance_transform");
    const bool agnesi=transform.at("type").get<std::string>()=="agnesi";
    const double agnesi_a=agnesi?transform.at("a").get<double>():0.0;
    const double agnesi_q=agnesi?transform.at("q").get<double>():0.0;
    const double agnesi_p=agnesi?transform.at("p").get<double>():0.0;
    const auto covalent_radii=agnesi
        ?transform.at("covalent_radii").get<std::vector<double>>()
        :std::vector<double>();

    const int types=static_cast<int>(selected_model_indices.size());
    const int target_offset=radial_size+source_width;
    std::vector<std::vector<double>> convolution_source(types),
        convolution_target(types),density_source(types),density_target(types);
    for(int type=0;type<types;++type) {
        std::vector<double> attrs(model_element_count,0.0);
        attrs.at(selected_model_indices.at(type))=1.0;
        const auto source=source_linear.evaluate(attrs);
        const auto target=target_linear.evaluate(attrs);
        convolution_source[type]=
            convolution_mlp.first_layer_contribution(radial_size,source);
        convolution_target[type]=
            convolution_mlp.first_layer_contribution(target_offset,target);
        density_source[type]=
            density_mlp.first_layer_contribution(radial_size,source);
        density_target[type]=
            density_mlp.first_layer_contribution(target_offset,target);
    }

    const int weight_count=convolution_mlp.output_size();
    const int function_count=weight_count+1;
    const std::size_t pair_count=static_cast<std::size_t>(types)*types;
    std::vector<std::vector<std::vector<double>>> values(
        pair_count,std::vector<std::vector<double>>(
            function_count,std::vector<double>(spline_nodes)));
    auto derivatives=values;
    constexpr double x0=1.0e-12;
    const double h=(cutoff-x0)/(spline_nodes-1);

    for(int source_type=0;source_type<types;++source_type) {
        for(int target_type=0;target_type<types;++target_type) {
            const std::size_t pair=static_cast<std::size_t>(source_type)*types
                +target_type;
            const int source_model=selected_model_indices.at(source_type);
            const int target_model=selected_model_indices.at(target_type);
            const int source_z=model_atomic_numbers.at(source_model);
            const int target_z=model_atomic_numbers.at(target_model);
            double r0=1.0;
            if(agnesi) {
                if(source_z<0||target_z<0
                    ||static_cast<std::size_t>(source_z)>=covalent_radii.size()
                    ||static_cast<std::size_t>(target_z)>=covalent_radii.size())
                    return false;
                r0=0.5*(covalent_radii.at(source_z)+covalent_radii.at(target_z));
                if(!(r0>0.0))return false;
            }
            for(int node=0;node<spline_nodes;++node) {
                const double distance=x0+h*node;
                const double x=distance/cutoff;
                double envelope=0.0;
                double envelope_derivative=0.0;
                if(distance<cutoff) {
                    envelope=1.0-0.5*(cutoff_degree+1)*(cutoff_degree+2)
                        *std::pow(x,cutoff_degree)
                        +cutoff_degree*(cutoff_degree+2)
                            *std::pow(x,cutoff_degree+1)
                        -0.5*cutoff_degree*(cutoff_degree+1)
                            *std::pow(x,cutoff_degree+2);
                    envelope_derivative=(
                        -0.5*cutoff_degree*(cutoff_degree+1)*(cutoff_degree+2)
                            *std::pow(x,cutoff_degree-1)
                        +cutoff_degree*(cutoff_degree+1)*(cutoff_degree+2)
                            *std::pow(x,cutoff_degree)
                        -0.5*cutoff_degree*(cutoff_degree+1)*(cutoff_degree+2)
                            *std::pow(x,cutoff_degree+1))/cutoff;
                }

                double transformed=distance;
                double transform_derivative=1.0;
                if(agnesi) {
                    const double ratio=distance/r0;
                    const double ratio_power=std::pow(ratio,agnesi_q-agnesi_p);
                    const double denominator=1.0+ratio_power;
                    const double g=agnesi_a*std::pow(ratio,agnesi_q)/denominator;
                    const double dgdx=agnesi_a*(
                        agnesi_q*std::pow(ratio,agnesi_q-1.0)*denominator
                        -(agnesi_q-agnesi_p)
                            *std::pow(ratio,2.0*agnesi_q-agnesi_p-1.0))
                        /(denominator*denominator);
                    transformed=1.0/(1.0+g);
                    transform_derivative=-dgdx/(r0*(1.0+g)*(1.0+g));
                }
                std::vector<double> radial(radial_size),radial_derivative(radial_size);
                for(int k=0;k<radial_size;++k) {
                    const double base=prefactor*std::sin(bessel[k]*transformed)
                        /transformed;
                    const double base_derivative=prefactor
                        *(bessel[k]*std::cos(bessel[k]*transformed)*transformed
                            -std::sin(bessel[k]*transformed))
                        /(transformed*transformed)*transform_derivative;
                    radial[k]=embed_cutoff?base*envelope:base;
                    radial_derivative[k]=embed_cutoff
                        ?base_derivative*envelope+base*envelope_derivative
                        :base_derivative;
                }

                std::vector<double> weight,weight_derivative;
                convolution_mlp.evaluate_conditioned_with_directional_derivative(
                    radial,radial_derivative,convolution_source[source_type],
                    convolution_target[target_type],weight,weight_derivative);
                const double external_cutoff=embed_cutoff?1.0:envelope;
                const double external_cutoff_derivative=
                    embed_cutoff?0.0:envelope_derivative;
                for(int function=0;function<weight_count;++function) {
                    values[pair][function][node]=
                        weight[function]*external_cutoff;
                    derivatives[pair][function][node]=
                        weight_derivative[function]*external_cutoff
                        +weight[function]*external_cutoff_derivative;
                }

                std::vector<double> density_value,density_derivative;
                density_mlp.evaluate_conditioned_with_directional_derivative(
                    radial,radial_derivative,density_source[source_type],
                    density_target[target_type],density_value,density_derivative);
                const double density_base=std::tanh(
                    density_value[0]*density_value[0]);
                const double density_base_derivative=
                    (1.0-density_base*density_base)*2.0*density_value[0]
                    *density_derivative[0];
                values[pair][weight_count][node]=
                    density_base*external_cutoff;
                derivatives[pair][weight_count][node]=
                    density_base_derivative*external_cutoff
                    +density_base*external_cutoff_derivative;
            }
        }
    }
    pair_spline=RadialFunctionSetKokkos<Precision>(
        h,std::move(values),std::move(derivatives),x0);
    pair_spline_weight_count=weight_count;
    pair_spline_ready=true;
    return true;
}

template<typename Precision>
MaceNonlinearKokkosT<Precision>::Readout::Readout(const nlohmann::json& data)
    :nonlinear(data.at("class").get<std::string>()=="NonLinearReadoutBlock"),linear(readout_component(data,0)),linear_1(readout_component(data,1)),linear_2(readout_component(data,2))
{if(nonlinear){if(data.at("activation").get<std::string>()!="silu")throw std::invalid_argument("MACE_Nonlinear Kokkos readout activation is unsupported.");activation_constant=data.at("activation_constants").at(0).get<Precision>();}}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::prepare_mh1_pair_spline_for_graph(
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> neigh_types)
{
    if(!mh1_pair_spline_lazy_)
        return;

    const auto node_types_host=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),node_types);
    const auto neigh_types_host=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),neigh_types);
    std::vector<int> global_types;
    global_types.reserve(
        node_types_host.extent(0)+neigh_types_host.extent(0));
    const auto append_type=[&](const int type) {
        if(type<0||type>=static_cast<int>(atomic_numbers_host.size()))
            throw std::out_of_range(
                "MH-1 pair-spline graph type is out of range.");
        global_types.push_back(type);
    };
    for(std::size_t index=0;index<node_types_host.extent(0);++index)
        append_type(node_types_host(index));
    for(std::size_t index=0;index<neigh_types_host.extent(0);++index)
        append_type(neigh_types_host(index));
    std::sort(global_types.begin(),global_types.end());
    global_types.erase(
        std::unique(global_types.begin(),global_types.end()),global_types.end());
    if(global_types.empty()) {
        Kokkos::fence("Replace MH-1 active pair-spline type views");
        mh1_spline_global_types_.clear();
        mh1_spline_global_to_active_.clear();
        mh1_spline_node_types_=toKokkosView(
            "MH-1 compact spline node types",std::vector<int>());
        mh1_spline_neigh_types_=toKokkosView(
            "MH-1 compact spline neighbor types",std::vector<int>());
        mh1_spline_type_count_=0;
        return;
    }

    if(global_types!=mh1_spline_global_types_) {
        // Previous evaluations may still reference the old pair tables or
        // compact type views on an asynchronous execution space.
        Kokkos::fence("Replace MH-1 active pair-spline tables");
        std::vector<int> selected_indices;
        selected_indices.reserve(global_types.size());
        for(const int global_type : global_types)
            selected_indices.push_back(
                mh1_selected_model_indices_host_.at(global_type));
        const int spline_nodes=mh1_pair_spline_nodes_;
        bool prepared=true;
        for(int layer=0;layer<static_cast<int>(interactions.size());++layer)
            prepared=interactions[layer].prepare_pair_spline(
                mh1_pair_spline_interactions_data_.at(layer),
                mh1_pair_spline_radial_data_,r_cut,num_bessel,
                model_num_elements,selected_indices,
                mh1_model_atomic_numbers_host_,spline_nodes)&&prepared;
        if(!prepared)
            throw std::runtime_error(
                "Could not prepare the MH-1 pair-spline table for the graph species.");
        mh1_spline_global_types_=std::move(global_types);
        mh1_spline_global_to_active_.assign(atomic_numbers_host.size(),-1);
        for(std::size_t type=0;type<mh1_spline_global_types_.size();++type)
            mh1_spline_global_to_active_.at(
                mh1_spline_global_types_[type])=static_cast<int>(type);
    }

    std::vector<int> spline_node_types(node_types_host.extent(0));
    std::vector<int> spline_neigh_types(neigh_types_host.extent(0));
    for(std::size_t index=0;index<node_types_host.extent(0);++index)
        spline_node_types[index]=mh1_spline_global_to_active_.at(
            node_types_host(index));
    for(std::size_t index=0;index<neigh_types_host.extent(0);++index)
        spline_neigh_types[index]=mh1_spline_global_to_active_.at(
            neigh_types_host(index));
    Kokkos::fence("Replace MH-1 active pair-spline type views");
    mh1_spline_node_types_=toKokkosView(
        "MH-1 compact spline node types",spline_node_types);
    mh1_spline_neigh_types_=toKokkosView(
        "MH-1 compact spline neighbor types",spline_neigh_types);
    mh1_spline_type_count_=static_cast<int>(mh1_spline_global_types_.size());
    for(const auto& interaction : interactions)
        if(!interaction.pair_spline_ready)
            throw std::logic_error(
                "MH-1 pair-spline preparation did not produce all layers.");
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Readout::set_e3_linear_backend(
    const std::string& backend)
{
    linear.set_backend(backend);
    linear_1.set_backend(backend);
    linear_2.set_backend(backend);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Readout::set_e3_linear_workspace(
    const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace)
{
    linear.set_workspace(workspace);
    linear_1.set_workspace(workspace);
    linear_2.set_workspace(workspace);
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::Readout::linear_workspace_bytes() const
{
    return linear.workspace_bytes()+linear_1.workspace_bytes()
        +linear_2.workspace_bytes();
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Readout::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision*> output)
{
    ensure_view(result,result_storage,input.extent(0),1);
    if(!nonlinear){linear.evaluate(input,result);ordered_kokkos_deep_copy(output,Kokkos::subview(result,Kokkos::ALL,0));return;}
    ensure_view(hidden,hidden_storage,input.extent(0),linear_1.output_dimension());
    ensure_view(activated,activated_storage,input.extent(0),linear_1.output_dimension());
    auto local_hidden=hidden;auto local_activated=activated;
    linear_1.evaluate(input,hidden);const Precision constant=activation_constant;Kokkos::parallel_for("readout silu",hidden.size(),KOKKOS_LAMBDA(std::size_t flat){const std::size_t i=flat/local_hidden.extent(1);const int j=flat%local_hidden.extent(1);const Precision x=local_hidden(i,j);local_activated(i,j)=constant*x/(Precision(1)+Kokkos::exp(-x));});linear_2.evaluate(activated,result);ordered_kokkos_deep_copy(output,Kokkos::subview(result,Kokkos::ALL,0));
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Readout::reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Precision scale_value,Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint)
{
    ensure_view(seed,seed_storage,input.extent(0),1);ordered_kokkos_deep_copy(seed,scale_value);
    if(!nonlinear){linear.reverse(seed,input_adjoint);return;}
    ensure_view(hidden,hidden_storage,input.extent(0),linear_1.output_dimension());
    ensure_view(activated_adj,activated_adj_storage,input.extent(0),linear_1.output_dimension());
    ensure_view(hidden_adj,hidden_adj_storage,input.extent(0),linear_1.output_dimension());
    auto local_hidden=hidden;auto local_activated_adj=activated_adj;auto local_hidden_adj=hidden_adj;
    linear_2.reverse(seed,activated_adj);const Precision constant=activation_constant;Kokkos::parallel_for("reverse readout silu",hidden.size(),KOKKOS_LAMBDA(std::size_t flat){const std::size_t i=flat/local_hidden.extent(1);const int j=flat%local_hidden.extent(1);const Precision x=local_hidden(i,j),probability=Precision(1)/(Precision(1)+Kokkos::exp(-x));local_hidden_adj(i,j)=local_activated_adj(i,j)*constant*(probability+x*probability*(Precision(1)-probability));});linear_1.reverse(hidden_adj,input_adjoint);
}

#elif SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART == 2

template<typename Precision>
MaceNonlinearKokkosT<Precision>::MaceNonlinearKokkosT(
    const std::string& filename, const std::string& requested_head)
    :MaceNonlinearKokkosT(select_prediction_head(
        load_model_json(filename), requested_head))
{}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::streamed_edges_mode() const
{
    return mace_streamed_edges_mode_name(streamed_edges);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_mh1_edge_executor(std::string executor)
{
    if(executor!="mlp_reference"&&executor!="pair_spline_v1")
        throw std::invalid_argument(
            "MH-1 edge executor must be 'mlp_reference' or 'pair_spline_v1'.");
    if(executor=="pair_spline_v1") {
        if(!mh1_pair_spline_ready_)
            throw std::invalid_argument(
                "MH-1 pair_spline_v1 is unavailable for this model or backend.");
    }
    if(executor==mh1_edge_executor_)return;
    Kokkos::fence("MACE_Nonlinear MH-1 edge-executor transition");
    invalidate_factorized_prepared_graph();
    execution_source_schedule_dirty=true;
    mh1_edge_executor_=std::move(executor);
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::selected_execution_mh1_node_arena_tile_rows()
    const
{
    if(execution_mh1_node_arena_tile_rows_override_>0)
        return execution_mh1_node_arena_tile_rows_override_;
    if(execution_mh1_node_arena_policy_=="capacity-v1")
        return ExecutionMH1NodeRuntime::capacity_tile_rows;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return ExecutionMH1NodeRuntime::cuda_throughput_tile_rows;
#endif
    return ExecutionMH1NodeRuntime::capacity_tile_rows;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::apply_execution_mh1_node_arena_tile_rows()
{
    const int rows=selected_execution_mh1_node_arena_tile_rows();
    for(auto& runtime:execution_mh1_node_runtime) {
        runtime.tile_rows=rows;
        runtime.arena={};
        runtime.residual={};
        runtime.linear_1_output={};
        runtime.pre_gate={};
        runtime.gated={};
        runtime.interaction_output={};
        runtime.interaction_adjoint={};
        runtime.gated_adjoint={};
        runtime.pre_gate_adjoint={};
        runtime.linear_1_adjoint={};
    }
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_execution_mh1_node_arena_policy(
    std::string policy)
{
    if(policy!="throughput-v1"&&policy!="capacity-v1")
        throw std::invalid_argument(
            "Execution MH-1 node arena policy must be 'throughput-v1' or "
            "'capacity-v1'.");
    factorized_execution_space.fence("Set Execution MH-1 node arena policy");
    execution_mh1_node_arena_policy_=std::move(policy);
    execution_mh1_node_arena_tile_rows_override_=0;
    apply_execution_mh1_node_arena_tile_rows();
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::execution_mh1_node_arena_tile_rows() const
{
    return selected_execution_mh1_node_arena_tile_rows();
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>
    ::set_execution_mh1_node_arena_tile_rows_for_testing(const int rows)
{
    if(rows<=0)
        throw std::invalid_argument(
            "Execution MH-1 node arena tile rows must be positive.");
    factorized_execution_space.fence(
        "Set Execution MH-1 node arena test tile rows");
    execution_mh1_node_arena_tile_rows_override_=rows;
    apply_execution_mh1_node_arena_tile_rows();
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::supports_factorized() const
{
    if(!mh1_fast_path||interactions.empty())return false;
    return std::all_of(
        interactions.begin(),interactions.end(),[&](const auto& interaction) {
            if(!interaction.convolution.supports_execution_uvu()
                ||!interaction.convolution_weights
                    .supports_final_linear_factorization(num_bessel))
                return false;
            const auto weight=
                interaction.convolution_weights.final_linear_weight(num_bessel);
            const auto bias=interaction.convolution_weights.final_linear_bias();
            return weight.extent_int(0)==interaction.convolution.weight_size()
                &&weight.extent_int(1)==interaction.convolution_weights
                    .prefix_output_size(num_bessel)
                &&(bias.extent_int(0)==0
                    ||bias.extent_int(0)==interaction.convolution.weight_size());
        });
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_execution_mh1_contract_identity(
    std::string tag,std::string generation_fingerprint,
    std::string semantic_fingerprint,std::string structure_fingerprint)
{
    if(tag!="symmetrix.execution.mh1_uvu/1"
        &&tag!="symmetrix.execution.mh1_uvu/2"
        &&tag!="symmetrix.execution.mh1_uvu/3")
        throw std::invalid_argument("Execution MH-1 contract tag is unsupported.");
    const auto valid_fingerprint=[](const std::string& value) {
        constexpr std::string_view prefix="sha256:";
        return std::string_view(value).starts_with(prefix)
            &&value.size()==prefix.size()+64
            &&std::all_of(
                value.begin()+static_cast<std::ptrdiff_t>(prefix.size()),
                value.end(),[](const unsigned char character) {
                    return (character>='0'&&character<='9')
                        ||(character>='a'&&character<='f');
                });
    };
    if(!valid_fingerprint(generation_fingerprint)
        ||!valid_fingerprint(semantic_fingerprint)
        ||!valid_fingerprint(structure_fingerprint))
        throw std::invalid_argument(
            "Execution MH-1 contract fingerprints are malformed.");
    if(jit_mh1_host_plugin_ready()||jit_mh1_host_plugin_v4_ready()
        ||jit_mh1_cuda_plugin_ready()||jit_mh1_cuda_plugin_v4_ready())
        throw std::logic_error(
            "Execution MH-1 contract identity cannot change after plugin loading.");
    execution_mh1_has_model_contract=true;
    execution_mh1_model_generation_fingerprint=
        std::move(generation_fingerprint);
    execution_mh1_model_semantic_fingerprint=std::move(semantic_fingerprint);
    execution_mh1_model_structure_fingerprint=std::move(structure_fingerprint);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_execution_mh1_v4_contract_identity(
    std::string tag,std::string generation_fingerprint,
    std::string semantic_fingerprint,std::string structure_fingerprint,
    std::string runtime_layout_fingerprint)
{
    if(tag!="symmetrix.execution.mh1_uvu/4")
        throw std::invalid_argument("Execution MH-1 v4 contract tag is unsupported.");
    const auto valid_fingerprint=[](const std::string& value) {
        constexpr std::string_view prefix="sha256:";
        return std::string_view(value).starts_with(prefix)
            &&value.size()==prefix.size()+64
            &&std::all_of(
                value.begin()+static_cast<std::ptrdiff_t>(prefix.size()),
                value.end(),[](const unsigned char character) {
                    return (character>='0'&&character<='9')
                        ||(character>='a'&&character<='f');
                });
    };
    if(!valid_fingerprint(generation_fingerprint)
        ||!valid_fingerprint(semantic_fingerprint)
        ||!valid_fingerprint(structure_fingerprint)
        ||!valid_fingerprint(runtime_layout_fingerprint))
        throw std::invalid_argument(
            "Execution MH-1 v4 contract fingerprints are malformed.");
    if(jit_mh1_host_plugin_ready()||jit_mh1_host_plugin_v4_ready()
        ||jit_mh1_cuda_plugin_ready()||jit_mh1_cuda_plugin_v4_ready())
        throw std::logic_error(
            "Execution MH-1 v4 contract identity cannot change after plugin loading.");
    execution_mh1_has_model_contract_v4=true;
    execution_mh1_model_generation_fingerprint_v4=
        std::move(generation_fingerprint);
    execution_mh1_model_semantic_fingerprint_v4=std::move(semantic_fingerprint);
    execution_mh1_model_structure_fingerprint_v4=std::move(structure_fingerprint);
    execution_mh1_model_runtime_layout_fingerprint_v4=
        std::move(runtime_layout_fingerprint);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::load_jit_mh1_host_plugin(
    std::string path)
{
    if constexpr(!std::is_same_v<Precision,float>)
        throw std::invalid_argument(
            "Execution MH-1 host plugins require float32 model execution.");
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        throw std::invalid_argument(
            "Execution MH-1 host plugins require a host Kokkos execution space.");
#endif
    if(!execution_mh1_has_model_contract||!supports_factorized())
        throw std::invalid_argument(
            "Execution MH-1 host plugins require a validated two-interaction contract.");
    constexpr std::string_view prefix="sha256:";
    if(!std::string_view(execution_mh1_model_generation_fingerprint)
            .starts_with(prefix)
        ||execution_mh1_model_generation_fingerprint.size()<prefix.size()+16)
        throw std::invalid_argument(
            "Execution MH-1 generation fingerprint is malformed.");
    const std::string artifact_id="jit-mh1-gen"
        +std::to_string(symmetrix::execution::required_jit_generation_version)
        +"-v3-"
        +execution_mh1_model_generation_fingerprint.substr(prefix.size(),16);
    const std::uint32_t capabilities=
        SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V3
        |SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V3
        |SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V3
        |SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V3
        |SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V3
        |SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V3;
    symmetrix::execution::MH1HostPluginExpectation expectation;
    expectation.artifact_id=artifact_id;
    expectation.generation_fingerprint=
        execution_mh1_model_generation_fingerprint;
    expectation.semantic_fingerprint=execution_mh1_model_semantic_fingerprint;
    expectation.structure_fingerprint=execution_mh1_model_structure_fingerprint;
    expectation.required_capabilities=capabilities;
    for(int layer=0;layer<2;++layer) {
        const auto& interaction=interactions.at(layer);
        auto& extent=expectation.interactions.at(layer);
        extent.input_1_dimension=interaction.convolution.input_1_dimension();
        extent.input_2_dimension=interaction.convolution.input_2_dimension();
        extent.output_dimension=interaction.convolution.output_dimension();
        extent.weight_size=interaction.convolution.weight_size();
        extent.phi_dimension=interaction.convolution_weights
            .prefix_output_size(num_bessel);
        extent.multiplicity=interaction.convolution.execution_multiplicity();
        extent.input_1_angular_dimension=interaction.convolution
            .execution_input_1_angular_dimension();
        extent.instruction_count=interaction.convolution
            .execution_instruction_count();
    }
    auto loaded=symmetrix::execution::MH1HostPlugin::load(
        std::move(path),expectation);
    Kokkos::fence("Replace Execution MH-1 host plugin");
    jit_mh1_host_plugin=
        std::make_unique<symmetrix::execution::MH1HostPlugin>(std::move(loaded));
    for(int layer=0;layer<static_cast<int>(interactions.size());++layer)
        release_layer_edge_workspace(layer);
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::jit_mh1_host_plugin_ready() const
{
    return std::is_same_v<Precision,float>
        &&jit_mh1_host_plugin&&static_cast<bool>(*jit_mh1_host_plugin);
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::jit_mh1_host_plugin_path() const
{
    if(jit_mh1_host_plugin_v5_ready())
        return jit_mh1_host_plugin_v5->path();
    if(jit_mh1_host_plugin_v4_ready())
        return jit_mh1_host_plugin_v4->path();
    return jit_mh1_host_plugin_ready()
        ?jit_mh1_host_plugin->path():std::string();
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::jit_mh1_host_plugin_artifact_id() const
{
    if(jit_mh1_host_plugin_v5_ready())
        return std::string(jit_mh1_host_plugin_v5->descriptor().artifact_id);
    if(jit_mh1_host_plugin_v4_ready())
        return std::string(jit_mh1_host_plugin_v4->descriptor().artifact_id);
    return jit_mh1_host_plugin_ready()
        ?std::string(jit_mh1_host_plugin->descriptor().artifact_id)
        :std::string();
}

template<typename Precision>
symmetrix::execution::MH1HostNodeProgramExpectation
MaceNonlinearKokkosT<Precision>::execution_mh1_host_node_expectation(
    int layer) const
{
    if(layer<0||layer>=static_cast<int>(interactions.size())
        ||layer>=static_cast<int>(products.size())
        ||layer>=static_cast<int>(readouts.size())
        ||layer>=static_cast<int>(execution_mh1_node_runtime.size()))
        throw std::out_of_range("Execution MH-1 node layer is invalid.");
    const auto& interaction=interactions.at(layer);
    const auto& product=products.at(layer);
    const auto& readout=readouts.at(layer);
    const auto& runtime=execution_mh1_node_runtime.at(layer);
    const int input_dimension=interaction.linear_up.input_dimension();
    const int up_dimension=interaction.linear_up.output_dimension();
    const int residual_dimension=interaction.linear_res.output_dimension();
    const int gated_dimension=interaction.gate.output_size;
    const int interaction_dimension=interaction.linear_2.output_dimension();
    const int output_dimension=product.output_dimension();
    const int readout_hidden=readout.nonlinear
        ?readout.linear_1.output_dimension():0;
    const int arena_dimension=std::max({
        residual_dimension+gated_dimension,
        gated_dimension+interaction_dimension,
        interaction_dimension+output_dimension,
        2*interaction_dimension+output_dimension,
        layer==0
            ?interaction_dimension+input_dimension+2*output_dimension
            :interaction_dimension+2*output_dimension,
        2*readout_hidden+output_dimension});
    symmetrix::execution::MH1HostNodeProgramExpectation expectation;
    expectation.element_count=node_embedding.input_dimension();
    expectation.input_dimension=input_dimension;
    expectation.up_dimension=up_dimension;
    expectation.residual_dimension=residual_dimension;
    expectation.skip_dimension=interaction.skip.output_dimension();
    expectation.message_dimension=interaction.linear_1.input_dimension();
    expectation.interaction_output_dimension=interaction_dimension;
    expectation.output_dimension=output_dimension;
    expectation.product_term_count=product.compiled_term_count();
    expectation.node_arena_dimension=arena_dimension;
    expectation.requires_tp_source_state_adjoint=layer>0;
    expectation.linear_parameter_count=runtime.linear_parameters.size();
    expectation.product_parameter_count=runtime.product_parameters.size();
    expectation.readout_parameter_count=runtime.readout_parameters.size();
    expectation.forward_owners_per_node={1,1};
    expectation.reverse_owners_per_node={1,layer>0?1:0};
    return expectation;
}

template<typename Precision>
symmetrix::execution::MH1CudaNodeProgramExpectation
MaceNonlinearKokkosT<Precision>::execution_mh1_cuda_node_expectation(
    int layer) const
{
    const auto host=execution_mh1_host_node_expectation(layer);
    symmetrix::execution::MH1CudaNodeProgramExpectation expectation;
    expectation.element_count=host.element_count;
    expectation.input_dimension=host.input_dimension;
    expectation.up_dimension=host.up_dimension;
    expectation.residual_dimension=host.residual_dimension;
    expectation.skip_dimension=host.skip_dimension;
    expectation.message_dimension=host.message_dimension;
    expectation.interaction_output_dimension=
        host.interaction_output_dimension;
    expectation.output_dimension=host.output_dimension;
    expectation.product_term_count=host.product_term_count;
    expectation.node_arena_dimension=host.node_arena_dimension;
    expectation.requires_tp_source_state_adjoint=
        host.requires_tp_source_state_adjoint;
    expectation.linear_parameter_count=host.linear_parameter_count;
    expectation.product_parameter_count=host.product_parameter_count;
    expectation.readout_parameter_count=host.readout_parameter_count;
    expectation.forward_threads_per_block={128,128};
    expectation.reverse_threads_per_block={128,layer>0?128:0};
    return expectation;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::load_jit_mh1_host_plugin_v4(
    std::string path)
{
    if constexpr(!std::is_same_v<Precision,float>)
        throw std::invalid_argument(
            "Execution MH-1 v4 host plugins require float32 model execution.");
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        throw std::invalid_argument(
            "Execution MH-1 v4 host plugins require a host Kokkos execution space.");
#endif
    if(!execution_mh1_has_model_contract_v4||!supports_factorized())
        throw std::invalid_argument(
            "Execution MH-1 v4 host plugins require a validated node contract.");
    prepare_execution_mh1_node_runtime_parameters();
    constexpr std::string_view prefix="sha256:";
    const std::string artifact_id="jit-mh1-gen"
        +std::to_string(symmetrix::execution::required_jit_generation_version)
        +"-v4-"
        +execution_mh1_model_generation_fingerprint_v4.substr(prefix.size(),16);
    symmetrix::execution::MH1HostPluginV4Expectation expectation;
    expectation.artifact_id=artifact_id;
    expectation.generation_fingerprint=
        execution_mh1_model_generation_fingerprint_v4;
    expectation.semantic_fingerprint=execution_mh1_model_semantic_fingerprint_v4;
    expectation.structure_fingerprint=execution_mh1_model_structure_fingerprint_v4;
    expectation.runtime_layout_fingerprint=
        execution_mh1_model_runtime_layout_fingerprint_v4;
    expectation.required_capabilities=
        SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V4
        |SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V4
        |SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V4
        |SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V4
        |SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V4
        |SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V4
        |SYMMETRIX_JIT_MH1_HOST_NODE_PROGRAM_PHASES_V4
        |SYMMETRIX_JIT_MH1_HOST_PERSISTENT_IR_MUL_NODE_STATE_V4
        |SYMMETRIX_JIT_MH1_HOST_FIXED_WEIGHT_COORDINATES_V4
        |SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_V4
        |SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_V4
        |SYMMETRIX_JIT_MH1_HOST_PHI_MAJOR_FORWARD_WEIGHT_V4;
    for(int layer=0;layer<2;++layer) {
        const auto& interaction=interactions.at(layer);
        auto& extent=expectation.interactions.at(layer);
        extent.input_1_dimension=interaction.convolution.input_1_dimension();
        extent.input_2_dimension=interaction.convolution.input_2_dimension();
        extent.output_dimension=interaction.convolution.output_dimension();
        extent.weight_size=interaction.convolution.weight_size();
        extent.phi_dimension=interaction.convolution_weights
            .prefix_output_size(num_bessel);
        extent.multiplicity=interaction.convolution.execution_multiplicity();
        extent.input_1_angular_dimension=interaction.convolution
            .execution_input_1_angular_dimension();
        extent.instruction_count=interaction.convolution
            .execution_instruction_count();
        expectation.node_programs.at(layer)=
            execution_mh1_host_node_expectation(layer);
    }
    auto loaded=symmetrix::execution::MH1HostPluginV4::load(
        std::move(path),expectation);
    for(auto& interaction:interactions)
        interaction.convolution_weights
            .prepare_final_linear_weight_phi_major(num_bessel);
    Kokkos::fence("Replace Execution MH-1 v4 host plugin");
    jit_mh1_host_plugin_v4=
        std::make_unique<symmetrix::execution::MH1HostPluginV4>(
            std::move(loaded));
    for(int layer=0;layer<2;++layer) {
        auto& runtime=execution_mh1_node_runtime.at(layer);
        const auto& node=jit_mh1_host_plugin_v4->descriptor()
            .node_programs[layer];
        runtime.arena_dimension=node.node_arena_dimension;
        runtime.retained_pre_gate_dimension=0;
        runtime.retained_interaction_output_dimension=(node.flags
                &SYMMETRIX_JIT_MH1_HOST_NODE_RETAIN_INTERACTION_OUTPUT_V4)
            ?node.interaction_output_dimension:0;
        runtime.reuse_message_adjoint=false;
        states.at(layer).pre_gate={};
        states.at(layer).pre_gate_storage={};
        states.at(layer).interaction_output={};
        states.at(layer).interaction_output_storage={};
    }
    for(int layer=0;layer<static_cast<int>(interactions.size());++layer)
        release_layer_edge_workspace(layer);
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::jit_mh1_host_plugin_v4_ready() const
{
    return std::is_same_v<Precision,float>
        &&jit_mh1_host_plugin_v4
        &&static_cast<bool>(*jit_mh1_host_plugin_v4);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::load_jit_mh1_host_plugin_v5(
    std::string path)
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        throw std::invalid_argument(
            "Execution MH-1 v5 host plugins require a host Kokkos execution space.");
#endif
    if(!execution_mh1_has_model_contract_v4||!supports_factorized())
        throw std::invalid_argument(
            "Execution MH-1 v5 host plugins require a validated node contract.");
    prepare_execution_mh1_node_runtime_parameters();
    constexpr std::string_view prefix="sha256:";
    const std::string artifact_id="jit-mh1-gen"
        +std::to_string(symmetrix::execution::required_jit_generation_version)
        +"-v4-"
        +execution_mh1_model_generation_fingerprint_v4.substr(prefix.size(),16);
    symmetrix::execution::MH1HostPluginV5Expectation expectation;
    expectation.scalar_size=sizeof(Precision);
    auto& v4=expectation.v4;
    v4.artifact_id=artifact_id;
    v4.generation_fingerprint=execution_mh1_model_generation_fingerprint_v4;
    v4.semantic_fingerprint=execution_mh1_model_semantic_fingerprint_v4;
    v4.structure_fingerprint=execution_mh1_model_structure_fingerprint_v4;
    v4.runtime_layout_fingerprint=
        execution_mh1_model_runtime_layout_fingerprint_v4;
    v4.required_capabilities=
        SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V4
        |SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V4
        |SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V4
        |SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V4
        |SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V4
        |SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V4
        |SYMMETRIX_JIT_MH1_HOST_NODE_PROGRAM_PHASES_V4
        |SYMMETRIX_JIT_MH1_HOST_PERSISTENT_IR_MUL_NODE_STATE_V4
        |SYMMETRIX_JIT_MH1_HOST_FIXED_WEIGHT_COORDINATES_V4
        |SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_V4
        |SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_V4
        |SYMMETRIX_JIT_MH1_HOST_SOURCE_OWNS_EDGE_REVERSE_V4
        |SYMMETRIX_JIT_MH1_HOST_PHI_MAJOR_FORWARD_WEIGHT_V4;
    expectation.required_capabilities=v4.required_capabilities
        |SYMMETRIX_JIT_MH1_HOST_SPLINE_R_FORWARD_V5
        |SYMMETRIX_JIT_MH1_HOST_SPLINE_R_SOURCE_REVERSE_V5;
    for(int layer=0;layer<2;++layer) {
        const auto& interaction=interactions.at(layer);
        auto& extent=v4.interactions.at(layer);
        extent.input_1_dimension=interaction.convolution.input_1_dimension();
        extent.input_2_dimension=interaction.convolution.input_2_dimension();
        extent.output_dimension=interaction.convolution.output_dimension();
        extent.weight_size=interaction.convolution.weight_size();
        extent.phi_dimension=
            interaction.convolution_weights.prefix_output_size(num_bessel);
        extent.multiplicity=interaction.convolution.execution_multiplicity();
        extent.input_1_angular_dimension=
            interaction.convolution.execution_input_1_angular_dimension();
        extent.instruction_count=
            interaction.convolution.execution_instruction_count();
        v4.node_programs.at(layer)=execution_mh1_host_node_expectation(layer);
        auto& spline=expectation.spline_r_programs.at(layer);
        spline.flags=SYMMETRIX_JIT_MH1_HOST_SPLINE_R_PACKED_FORWARD_MESSAGES_V5;
        spline.weight_function_count=interaction.convolution.weight_size();
        spline.density_function_index=interaction.convolution.weight_size();
        spline.coefficient_count=4;
    }
    auto loaded=symmetrix::execution::MH1HostPluginV5::load(path,expectation);
    if constexpr(std::is_same_v<Precision,float>)
        load_jit_mh1_host_plugin_v4(path);
    else
        for(int layer=0;layer<static_cast<int>(interactions.size());++layer)
            release_layer_edge_workspace(layer);
    Kokkos::fence("Replace Execution MH-1 v5 host plugin");
    jit_mh1_host_plugin_v5=
        std::make_unique<symmetrix::execution::MH1HostPluginV5>(
            std::move(loaded));
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::jit_mh1_host_plugin_v5_ready() const
{
    return jit_mh1_host_plugin_v5
        &&static_cast<bool>(*jit_mh1_host_plugin_v5)
        &&jit_mh1_host_plugin_v5->descriptor().scalar_size==sizeof(Precision);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::load_jit_mh1_cuda_plugin(
    std::string path)
{
    if constexpr(!std::is_same_v<Precision,float>)
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugins require float32 model execution.");
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(!std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugins require a CUDA Kokkos execution space.");
    } else {
        if(!execution_mh1_has_model_contract||!supports_factorized())
            throw std::invalid_argument(
                "Execution MH-1 CUDA plugins require a validated "
                "two-interaction contract.");
        constexpr std::string_view prefix="sha256:";
        if(!std::string_view(execution_mh1_model_generation_fingerprint)
                .starts_with(prefix)
            ||execution_mh1_model_generation_fingerprint.size()<prefix.size()+16)
            throw std::invalid_argument(
                "Execution MH-1 generation fingerprint is malformed.");
        Kokkos::DefaultExecutionSpace execution_space;
        ExecutionMH1CudaDeviceGuard device_guard(execution_space.cuda_device());
        const auto& device=execution_space.cuda_device_prop();
        const std::string artifact_id="jit-mh1-gen"
            +std::to_string(symmetrix::execution::required_jit_generation_version)
            +"-v3-"
            +execution_mh1_model_generation_fingerprint.substr(prefix.size(),16);
        const std::uint32_t capabilities=
            SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V3
            |SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V3
            |SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V3
            |SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V3
            |SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V3
            |SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V3
            |SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V3
            |SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V3
            |SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V3;
        symmetrix::execution::MH1CudaPluginExpectation expectation;
        expectation.artifact_id=artifact_id;
        expectation.generation_fingerprint=
            execution_mh1_model_generation_fingerprint;
        expectation.semantic_fingerprint=execution_mh1_model_semantic_fingerprint;
        expectation.structure_fingerprint=execution_mh1_model_structure_fingerprint;
        expectation.target_compute_capability=10*device.major+device.minor;
        expectation.max_threads_per_block=device.maxThreadsPerBlock;
        expectation.required_capabilities=capabilities;
        for(int layer=0;layer<2;++layer) {
            const auto& interaction=interactions.at(layer);
            auto& extent=expectation.interactions.at(layer);
            extent.input_1_dimension=interaction.convolution.input_1_dimension();
            extent.input_2_dimension=interaction.convolution.input_2_dimension();
            extent.output_dimension=interaction.convolution.output_dimension();
            extent.weight_size=interaction.convolution.weight_size();
            extent.phi_dimension=interaction.convolution_weights
                .prefix_output_size(num_bessel);
            extent.multiplicity=interaction.convolution.execution_multiplicity();
            extent.input_1_angular_dimension=interaction.convolution
                .execution_input_1_angular_dimension();
            extent.instruction_count=interaction.convolution
                .execution_instruction_count();
        }
        auto loaded=symmetrix::execution::MH1CudaPlugin::load(
            std::move(path),expectation);
        for(auto& interaction:interactions)
            interaction.convolution_weights
                .prepare_final_linear_weight_phi_major(num_bessel);
        execution_space.fence("Replace Execution MH-1 CUDA plugin");
        jit_mh1_cuda_plugin=
            std::make_unique<symmetrix::execution::MH1CudaPlugin>(
                std::move(loaded));
        for(int layer=0;layer<static_cast<int>(interactions.size());++layer)
            release_layer_edge_workspace(layer);
    }
#else
    (void)path;
    throw std::invalid_argument(
        "Execution MH-1 CUDA plugins require a CUDA-enabled Kokkos build.");
#endif
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::load_jit_mh1_cuda_plugin_v4(
    std::string path, std::string launch_plan_json)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if constexpr(!device_execution_space<Kokkos::DefaultExecutionSpace>) {
        throw std::invalid_argument(
            "Execution MH-1 v4 device modules require a GPU Kokkos execution space.");
    } else {
        if(!execution_mh1_has_model_contract_v4||!supports_factorized())
            throw std::invalid_argument(
                "Execution MH-1 v4 CUDA plugins require a validated node contract.");
        prepare_execution_mh1_node_runtime_parameters();
        constexpr std::string_view prefix="sha256:";
        const std::string artifact_id="jit-mh1-gen"
            +std::to_string(symmetrix::execution::required_jit_generation_version)
            +"-v4-"
            +execution_mh1_model_generation_fingerprint_v4.substr(
                prefix.size(),16);
        Kokkos::DefaultExecutionSpace execution_space;
        using Traits=DeviceBackendTraits<Kokkos::DefaultExecutionSpace>;
        ExecutionMH1CudaDeviceGuard device_guard(
            Traits::device_ordinal(execution_space));
        const auto device=execution_device_execution_environment();
        symmetrix::execution::MH1CudaPluginV4Expectation expectation;
        expectation.artifact_id=artifact_id;
        expectation.generation_fingerprint=
            execution_mh1_model_generation_fingerprint_v4;
        expectation.semantic_fingerprint=
            execution_mh1_model_semantic_fingerprint_v4;
        expectation.structure_fingerprint=
            execution_mh1_model_structure_fingerprint_v4;
        expectation.runtime_layout_fingerprint=
            execution_mh1_model_runtime_layout_fingerprint_v4;
        expectation.target_backend=Traits::backend_name;
        expectation.target_architecture=device.architecture;
        expectation.target_compute_capability=device.compute_capability;
        expectation.max_threads_per_block=device.max_team_size;
        expectation.scalar_size=sizeof(Precision);
        expectation.required_capabilities=
            SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V4
            |SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V4
            |SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V4
            |SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V4
            |SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V4
            |SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V4
            |SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V4
            |SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V4
            |SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V4
            |SYMMETRIX_JIT_MH1_CUDA_NODE_PROGRAM_PHASES_V4
            |SYMMETRIX_JIT_MH1_CUDA_PERSISTENT_IR_MUL_NODE_STATE_V4
            |SYMMETRIX_JIT_MH1_CUDA_FIXED_WEIGHT_COORDINATES_V4;
        for(int layer=0;layer<2;++layer) {
            const auto& interaction=interactions.at(layer);
            auto& extent=expectation.interactions.at(layer);
            extent.input_1_dimension=
                interaction.convolution.input_1_dimension();
            extent.input_2_dimension=
                interaction.convolution.input_2_dimension();
            extent.output_dimension=interaction.convolution.output_dimension();
            extent.weight_size=interaction.convolution.weight_size();
            extent.phi_dimension=interaction.convolution_weights
                .prefix_output_size(num_bessel);
            extent.multiplicity=interaction.convolution.execution_multiplicity();
            extent.input_1_angular_dimension=interaction.convolution
                .execution_input_1_angular_dimension();
            extent.instruction_count=interaction.convolution
                .execution_instruction_count();
            expectation.node_programs.at(layer)=
                execution_mh1_cuda_node_expectation(layer);
        }
        const bool is_module=!launch_plan_json.empty();
        auto loaded=is_module
            ?symmetrix::execution::MH1CudaPluginV4::load_module(
                std::move(path),expectation,std::move(launch_plan_json))
            :symmetrix::execution::MH1CudaPluginV4::load(
                std::move(path),expectation);
        for(auto& interaction:interactions)
            interaction.convolution_weights
                .prepare_final_linear_weight_phi_major(num_bessel);
        execution_space.fence("Replace Execution MH-1 v4 CUDA plugin");
        const auto& descriptor=loaded.descriptor();
        for(int layer=0;layer<2;++layer) {
            const auto& node=descriptor.node_programs[layer];
            auto& runtime=execution_mh1_node_runtime.at(layer);
            runtime.arena_dimension=node.node_arena_dimension;
            runtime.retained_pre_gate_dimension=
                node.retained_pre_gate_dimension;
            runtime.retained_interaction_output_dimension=
                static_cast<int>(node.retained_interaction_output_dimension);
            runtime.reuse_message_adjoint=(node.flags
                &SYMMETRIX_JIT_MH1_CUDA_NODE_REUSE_MESSAGE_ADJOINT_V4)!=0;
            runtime.tile_rows=selected_execution_mh1_node_arena_tile_rows();
            if(runtime.retained_pre_gate_dimension==0) {
                states.at(layer).pre_gate={};
                states.at(layer).pre_gate_storage={};
            }
            if(runtime.retained_interaction_output_dimension==0) {
                states.at(layer).interaction_output={};
                states.at(layer).interaction_output_storage={};
            }
        }
        jit_mh1_cuda_plugin_v4=
            std::make_unique<symmetrix::execution::MH1CudaPluginV4>(
                std::move(loaded));
        for(int layer=0;layer<static_cast<int>(interactions.size());++layer)
            release_layer_edge_workspace(layer);
    }
#else
    (void)path;(void)launch_plan_json;
    throw std::invalid_argument(
        "Execution MH-1 v4 device modules require a GPU-enabled Kokkos build.");
#endif
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::jit_mh1_cuda_plugin_ready() const
{
#ifdef KOKKOS_ENABLE_CUDA
    return std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>
        &&jit_mh1_cuda_plugin&&static_cast<bool>(*jit_mh1_cuda_plugin);
#else
    return false;
#endif
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::jit_mh1_cuda_plugin_v4_ready() const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return device_execution_space<Kokkos::DefaultExecutionSpace>
        &&jit_mh1_cuda_plugin_v4
        &&static_cast<bool>(*jit_mh1_cuda_plugin_v4)
        &&jit_mh1_cuda_plugin_v4->descriptor().scalar_size==sizeof(Precision);
#else
    return false;
#endif
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::jit_mh1_cuda_plugin_path() const
{
    if(jit_mh1_cuda_plugin_v4_ready())
        return jit_mh1_cuda_plugin_v4->path();
    return jit_mh1_cuda_plugin_ready()
        ?jit_mh1_cuda_plugin->path():std::string();
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::jit_mh1_cuda_plugin_artifact_id() const
{
    if(jit_mh1_cuda_plugin_v4_ready())
        return std::string(jit_mh1_cuda_plugin_v4->descriptor().artifact_id);
    return jit_mh1_cuda_plugin_ready()
        ?std::string(jit_mh1_cuda_plugin->descriptor().artifact_id)
        :std::string();
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::device_cuda_available() const
{
#ifdef KOKKOS_ENABLE_CUDA
    return std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>;
#else
    return false;
#endif
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::execution_cuda_device_name() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        Kokkos::DefaultExecutionSpace execution_space;
        return execution_space.cuda_device_prop().name;
    }
#endif
    return {};
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::execution_cuda_device_ordinal() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        Kokkos::DefaultExecutionSpace execution_space;
        return execution_space.cuda_device();
    }
#endif
    return -1;
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::execution_cuda_compute_capability() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        Kokkos::DefaultExecutionSpace execution_space;
        const auto& device=execution_space.cuda_device_prop();
        return 10*device.major+device.minor;
    }
#endif
    return 0;
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::execution_cuda_multiprocessor_count() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        Kokkos::DefaultExecutionSpace execution_space;
        return execution_space.cuda_device_prop().multiProcessorCount;
    }
#endif
    return 0;
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::execution_cuda_warp_width() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        Kokkos::DefaultExecutionSpace execution_space;
        return execution_space.cuda_device_prop().warpSize;
    }
#endif
    return 0;
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::execution_cuda_runtime_version() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        int version=0;
        check_execution_mh1_cuda_status(
            static_cast<std::int32_t>(cudaRuntimeGetVersion(&version)),
            "Querying the CUDA runtime version");
        return version;
    }
#endif
    return 0;
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::execution_cuda_driver_version() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        int version=0;
        check_execution_mh1_cuda_status(
            static_cast<std::int32_t>(cudaDriverGetVersion(&version)),
            "Querying the CUDA driver version");
        return version;
    }
#endif
    return 0;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_streamed_edges(std::string mode)
{
    const auto requested=parse_mace_streamed_edges_mode(mode);
    if(mh1_fast_path&&requested==MACEStreamedEdgesMode::materialized)
        throw std::invalid_argument(
            "MACE-MH-1 Kokkos execution supports only streamed_edges="
            "'generic' or 'direct'; 'materialized' is disabled "
            "because its shared conditioned-MLP "
            "workspace can invalidate the reverse tape.");
    if(requested!=MACEStreamedEdgesMode::materialized&&!supports_streamed_edges())
        throw std::invalid_argument(
            "Streamed edges require a compatible MACE-MH-1 family fast path.");
    if(mace_uses_receiver_factorization(requested))
        throw std::invalid_argument(
            "MACE-MH-1 receiver_factorized RTC is not implemented.");
    if(mace_uses_prepared_execution(requested)&&!supports_factorized())
        throw std::invalid_argument(
            "direct and receiver_factorized require a factorable MACE-MH-1 "
            "edge tensor product.");
    Kokkos::fence("MACE_Nonlinear streamed-mode transition");
    invalidate_factorized_prepared_graph();
    execution_source_schedule_dirty=true;
    streamed_edges=requested;
    factorized_source_owned_reverse_used=false;
    if(streamed_edges==MACEStreamedEdgesMode::generic
        ||mace_uses_prepared_execution(streamed_edges))
        for(int layer=0;layer<static_cast<int>(states.size());++layer)
            release_layer_edge_workspace(layer);
    if(!mace_uses_prepared_execution(streamed_edges))
        set_execution_mh1_scratch_budget_bytes(execution_mh1_scratch_budget);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_execution_mh1_scratch_budget_bytes(
    const std::size_t budget_bytes)
{
    Kokkos::fence("MACE_Nonlinear Execution MH-1 scratch-budget transition");
    execution_mh1_scratch_budget=budget_bytes;
    execution_mh1_scratch_minimum=0;
    execution_mh1_scratch_planned=0;
    execution_mh1_scratch_retained=0;
    execution_mh1_scratch_recomputed=0;
    execution_mh1_scratch_recomputed_layers=0;
    execution_mh1_retain_graph_embedding.clear();
    execution_mh1_graph_embedding_scratch_slot.clear();
    execution_mh1_graph_embedding_scratch_widths.clear();
    execution_mh1_graph_embedding_scratch_storage.clear();
    execution_mh1_graph_embedding_adjoint_scratch_storage.clear();
    execution_mh1_graph_harmonic_adjoint_scratch_storage={};
    execution_mh1_graph_cutoff_adjoint_scratch_storage={};
    execution_mh1_message_adjoint_scratch_storage={};
    execution_mh1_up_adjoint_scratch_storage={};
    execution_mh1_input_adjoint_scratch_storage={};
    execution_mh1_density_adjoint_scratch_storage={};
    for(auto& state:states) {
        state.message_adj={};
        state.up_adj={};
        state.execution_input_adjoint_ir_mul={};
        state.density_adj={};
        state.execution_graph_embedding={};
        state.execution_graph_embedding_storage={};
        state.execution_graph_embedding_adj={};
        state.execution_graph_embedding_adj_storage={};
        state.execution_graph_harmonic_adj={};
        state.execution_graph_harmonic_adj_storage={};
        state.execution_graph_cutoff_adj={};
        state.execution_graph_cutoff_adj_storage={};
    }
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::invalidate_factorized_prepared_graph()
{
    factorized_prepared_graph_generation=0;
    factorized_prepared_geometry_graph_generation=0;
    factorized_prepared_batch_graph_generation=0;
    factorized_completed_evaluation_graph_generation=0;
    execution_prepared_batch_edge_offsets_host.clear();
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::reserve_execution_geometry_workspace(
    const int edges)
{
    if(edges<0)
        throw std::invalid_argument(
            "Execution geometry workspace requires a non-negative edge count.");
    if(edges<=execution_geometry_edge_capacity) {
        execution_geometry_growth_reason_="existing capacity reused";
        return;
    }
    const int doubled=execution_geometry_edge_capacity
            >std::numeric_limits<int>::max()/2
        ?std::numeric_limits<int>::max()
        :2*execution_geometry_edge_capacity;
    int capacity=std::max(edges,std::max(1,doubled));
    const std::size_t harmonics=static_cast<std::size_t>(num_lm);
    const std::size_t geometry_bytes_per_edge=
        4*sizeof(double)+(3+7*harmonics)*sizeof(Precision);
    std::size_t free_bytes=execution_geometry_device_free_;
    std::size_t total_bytes=execution_geometry_device_total_;
    bool memory_info_available=execution_geometry_memory_info_override_;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if(!memory_info_available) {
        ExecutionMH1CudaDeviceGuard device_guard(
            execution_mh1_device_ordinal(factorized_execution_space));
#ifdef KOKKOS_ENABLE_CUDA
        memory_info_available=
            cudaMemGetInfo(&free_bytes,&total_bytes)==cudaSuccess;
#else
        memory_info_available=
            hipMemGetInfo(&free_bytes,&total_bytes)==hipSuccess;
#endif
    }
#endif
    if(memory_info_available) {
        constexpr std::size_t minimum_reserve=std::size_t(512)*1024*1024;
        const std::size_t reserve=std::max(minimum_reserve,total_bytes/20);
        const std::size_t current_bytes=
            static_cast<std::size_t>(execution_geometry_edge_capacity)
                *geometry_bytes_per_edge;
        const std::size_t reclaimable=current_bytes>total_bytes-free_bytes
            ?total_bytes:free_bytes+current_bytes;
        const std::size_t available=reclaimable>reserve
            ?reclaimable-reserve:0;
        const std::size_t exact_bytes=
            static_cast<std::size_t>(edges)*geometry_bytes_per_edge;
        if(exact_bytes>available)
            throw std::runtime_error(
                "MH-1 direct geometry requires "+std::to_string(exact_bytes)+
                " bytes for "+std::to_string(edges)+" directed edges ("+
                std::to_string(geometry_bytes_per_edge)+
                " bytes per edge), but only "+std::to_string(available)+
                " bytes are available after the "+std::to_string(reserve)+
                "-byte device reserve (free="+std::to_string(free_bytes)+
                ", total="+std::to_string(total_bytes)+").");
        const std::size_t geometric_bytes=
            static_cast<std::size_t>(capacity)*geometry_bytes_per_edge;
        if(geometric_bytes>available) {
            capacity=edges;
            execution_geometry_growth_reason_=
                "exact growth: geometric headroom exceeds available device memory";
        } else {
            execution_geometry_growth_reason_=
                "geometric growth fits available device memory";
        }
    } else {
        execution_geometry_growth_reason_=
            "geometric growth: device memory query unavailable";
    }
    const std::size_t edge_capacity=static_cast<std::size_t>(capacity);
    const std::size_t harmonic_capacity=edge_capacity
        *static_cast<std::size_t>(num_lm);

    factorized_execution_space.fence(
        "Replace MH-1 execution geometry workspace capacity");
    xyz_shuffled={};
    Y={};
    Y_grad={};
    Y_grad_shuffled={};
    xyz_shuffled_storage={};
    Y_storage={};
    Y_grad_storage={};
    Y_grad_shuffled_storage={};
    execution_prepared_xyz={};
    execution_prepared_distances={};
    execution_geometry_edge_capacity=0;
    execution_prepared_xyz=decltype(execution_prepared_xyz)(
        Kokkos::view_alloc(
            "MH-1 Execution prepared xyz",Kokkos::WithoutInitializing),
        3*edge_capacity);
    execution_prepared_distances=decltype(execution_prepared_distances)(
        Kokkos::view_alloc(
            "MH-1 Execution prepared distances",Kokkos::WithoutInitializing),
        edge_capacity);
    xyz_shuffled_storage=decltype(xyz_shuffled_storage)(
        Kokkos::view_alloc(
            "MH-1 Execution shuffled xyz storage",Kokkos::WithoutInitializing),
        3*edge_capacity);
    Y_storage=decltype(Y_storage)(
        Kokkos::view_alloc(
            "MH-1 Execution harmonic storage",Kokkos::WithoutInitializing),
        harmonic_capacity);
    Y_grad_storage=decltype(Y_grad_storage)(
        Kokkos::view_alloc(
            "MH-1 Execution harmonic gradient storage",
            Kokkos::WithoutInitializing),
        3*harmonic_capacity);
    Y_grad_shuffled_storage=decltype(Y_grad_shuffled_storage)(
        Kokkos::view_alloc(
            "MH-1 Execution raw harmonic gradient storage",
            Kokkos::WithoutInitializing),
        3*harmonic_capacity);
    execution_geometry_edge_capacity=capacity;
    ++execution_geometry_allocations;
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::execution_geometry_workspace_bytes()
    const
{
    return sizeof(double)*(execution_prepared_xyz.size()
        +execution_prepared_distances.size()
        +execution_prepared_reference_positions.size()
        +execution_prepared_reference_xyz.size()
        +execution_prepared_cell.size()
        +execution_prepared_inverse_cell.size()
        +execution_prepared_positions.size()
        +execution_prepared_displacements.size())
        +sizeof(int)*(execution_prepared_pbc.size()
            +execution_prepared_geometry_invalid.size())
        +sizeof(Precision)*(xyz_shuffled_storage.size()+Y_storage.size()
            +Y_grad_storage.size()+Y_grad_shuffled_storage.size());
}

template<typename Precision>
std::uint64_t MaceNonlinearKokkosT<Precision>::prepare_factorized_graph(
    const int num_nodes,const std::span<const int> node_types,
    const std::span<const int> num_neigh,
    const std::span<const int> neigh_indices,
    const std::span<const int> neigh_types)
{
    if(!mace_uses_prepared_execution(streamed_edges))
        throw std::invalid_argument(
            "Prepared Execution R1 graphs require streamed_edges='factorized'.");
    if(num_nodes<0
        ||node_types.size()!=static_cast<std::size_t>(num_nodes)
        ||num_neigh.size()!=static_cast<std::size_t>(num_nodes)
        ||neigh_indices.size()!=neigh_types.size()
        ||neigh_indices.size()
            >static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "Prepared MH-1 Execution R1 graph extents are inconsistent.");

    const auto prepared_node_types=std::vector<int>(
        node_types.begin(),node_types.end());
    const auto prepared_num_neigh=std::vector<int>(
        num_neigh.begin(),num_neigh.end());
    const auto prepared_neigh_indices=std::vector<int>(
        neigh_indices.begin(),neigh_indices.end());
    const auto prepared_neigh_types=std::vector<int>(
        neigh_types.begin(),neigh_types.end());
    const int local_types=static_cast<int>(atomic_numbers_host.size());
    for(const int type:prepared_node_types)
        if(type<0||type>=local_types)
            throw std::out_of_range(
                "Prepared MH-1 Execution R1 node type is out of range.");
    std::int64_t counted_edges=0;
    for(const int degree:prepared_num_neigh) {
        if(degree<0)
            throw std::invalid_argument(
                "Prepared MH-1 Execution R1 graph has a negative receiver degree.");
        counted_edges+=degree;
    }
    if(counted_edges!=static_cast<std::int64_t>(prepared_neigh_indices.size()))
        throw std::invalid_argument(
            "Prepared MH-1 Execution R1 receiver degrees do not sum to the edge count.");
    for(std::size_t edge=0;edge<prepared_neigh_indices.size();++edge) {
        const int source=prepared_neigh_indices[edge];
        const int type=prepared_neigh_types[edge];
        if(source<0||source>=num_nodes)
            throw std::out_of_range(
                "Prepared MH-1 Execution R1 source index is out of range.");
        if(type<0||type>=local_types
            ||type!=prepared_node_types[static_cast<std::size_t>(source)])
            throw std::invalid_argument(
                "Prepared MH-1 Execution R1 neighbor type is inconsistent.");
    }

    const bool same_identity=factorized_prepared_graph_generation!=0
        &&prepared_node_types==execution_prepared_node_types_host
        &&prepared_num_neigh==execution_prepared_num_neigh_host
        &&prepared_neigh_indices==execution_prepared_neigh_indices_host
        &&prepared_neigh_types==execution_prepared_neigh_types_host
        &&!execution_source_schedule_dirty;
    if(same_identity)return factorized_prepared_graph_generation;

    invalidate_factorized_prepared_graph();
    std::vector<int> receiver_offsets(
        static_cast<std::size_t>(num_nodes)+1,0);
    std::vector<int> edge_targets(prepared_neigh_indices.size(),0);
    int edge=0;
    for(int receiver=0;receiver<num_nodes;++receiver) {
        receiver_offsets[static_cast<std::size_t>(receiver)]=edge;
        for(int offset=0;offset<prepared_num_neigh[receiver];++offset)
            edge_targets[static_cast<std::size_t>(edge++)]=receiver;
    }
    receiver_offsets[static_cast<std::size_t>(num_nodes)]=edge;

    prepare_execution_source_schedule_host(
        num_nodes,prepared_neigh_indices,edge_targets);
    reserve_execution_geometry_workspace(
        static_cast<int>(prepared_neigh_indices.size()));
    execution_prepared_node_types=toKokkosView(
        "MH-1 Execution prepared node types",prepared_node_types);
    execution_prepared_num_neigh=toKokkosView(
        "MH-1 Execution prepared receiver degrees",prepared_num_neigh);
    execution_prepared_neigh_indices=toKokkosView(
        "MH-1 Execution prepared source indices",prepared_neigh_indices);
    execution_prepared_neigh_types=toKokkosView(
        "MH-1 Execution prepared neighbor types",prepared_neigh_types);
    execution_prepared_offsets=toKokkosView(
        "MH-1 Execution prepared receiver offsets",receiver_offsets);
    execution_prepared_targets=toKokkosView(
        "MH-1 Execution prepared edge targets",edge_targets);
    factorized_execution_space.fence("MH-1 Execution explicit graph preparation");
    ++factorized_preparation_fences;

    execution_prepared_node_types_host=prepared_node_types;
    execution_prepared_num_neigh_host=prepared_num_neigh;
    execution_prepared_neigh_indices_host=prepared_neigh_indices;
    execution_prepared_neigh_types_host=prepared_neigh_types;
    ++factorized_graph_generation_counter;
    if(factorized_graph_generation_counter==0)
        ++factorized_graph_generation_counter;
    factorized_prepared_graph_generation=factorized_graph_generation_counter;
    ++factorized_prepared_graphs;
    return factorized_prepared_graph_generation;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::prepare_factorized_geometry(
    const std::uint64_t graph_generation,
    const std::span<const double> reference_positions,
    const std::span<const double> reference_xyz,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const std::span<const int> pbc)
{
    if(graph_generation==0
        ||graph_generation!=factorized_prepared_graph_generation
        ||execution_source_schedule_dirty)
        throw std::invalid_argument(
            "MH-1 Execution geometry requires the current prepared graph token.");
    const std::size_t num_nodes=execution_prepared_node_types.extent(0);
    const std::size_t num_edges=execution_prepared_neigh_indices.extent(0);
    if(reference_positions.size()!=3*num_nodes
        ||reference_xyz.size()!=3*num_edges
        ||cell.size()!=9||inverse_cell.size()!=9||pbc.size()!=3)
        throw std::invalid_argument(
            "MH-1 Execution prepared geometry extents are inconsistent.");
    const auto finite=[](const double value) {return std::isfinite(value);};
    if(!std::all_of(reference_positions.begin(),reference_positions.end(),finite)
        ||!std::all_of(reference_xyz.begin(),reference_xyz.end(),finite)
        ||!std::all_of(cell.begin(),cell.end(),finite)
        ||!std::all_of(inverse_cell.begin(),inverse_cell.end(),finite))
        throw std::invalid_argument(
            "MH-1 Execution prepared geometry contains a non-finite value.");

    const auto copy_double=[&](
        const char* label,const std::span<const double> source,
        Kokkos::View<double*>& destination) {
        const auto host=Kokkos::View<
            const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                source.data(),source.size());
        destination=Kokkos::View<double*>(
            Kokkos::view_alloc(
                std::string(label),Kokkos::WithoutInitializing),source.size());
        Kokkos::deep_copy(factorized_execution_space,destination,host);
    };
    copy_double("MH-1 Execution reference positions",reference_positions,
        execution_prepared_reference_positions);
    copy_double("MH-1 Execution reference xyz",reference_xyz,
        execution_prepared_reference_xyz);
    copy_double("MH-1 Execution cell",cell,execution_prepared_cell);
    copy_double("MH-1 Execution inverse cell",inverse_cell,
        execution_prepared_inverse_cell);
    const auto pbc_host=Kokkos::View<
        const int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            pbc.data(),pbc.size());
    execution_prepared_pbc=Kokkos::View<int*>(
        Kokkos::view_alloc("MH-1 Execution pbc",Kokkos::WithoutInitializing),
        pbc.size());
    Kokkos::deep_copy(
        factorized_execution_space,execution_prepared_pbc,pbc_host);
    execution_prepared_positions=Kokkos::View<double*>(
        Kokkos::view_alloc(
            "MH-1 Execution current positions",Kokkos::WithoutInitializing),
        3*num_nodes);
    execution_prepared_displacements=Kokkos::View<double*>(
        Kokkos::view_alloc(
            "MH-1 Execution atom displacements",Kokkos::WithoutInitializing),
        3*num_nodes);
    execution_prepared_geometry_invalid=Kokkos::View<int*>(
        "MH-1 Execution geometry invalid",1);
    factorized_execution_space.fence(
        "MH-1 Execution prepared geometry initialization");
    factorized_prepared_geometry_graph_generation=graph_generation;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::compute_prepared_factorized(
    const std::uint64_t graph_generation,const std::span<const double> xyz,
    const std::span<const double> distances)
{
    if(!mace_uses_prepared_execution(streamed_edges))
        throw std::invalid_argument(
            "An execution graph token requires streamed_edges='direct' or "
            "'receiver_factorized'.");
    if(graph_generation==0
        ||graph_generation!=factorized_prepared_graph_generation
        ||execution_source_schedule_dirty)
        throw std::invalid_argument(
            "Execution R1 graph token is stale; prepare the graph again.");
    const std::size_t edges=execution_prepared_neigh_indices.extent(0);
    if(distances.size()!=edges||xyz.size()!=3*edges)
        throw std::invalid_argument(
            "Execution R1 prepared coordinates do not match the graph extents.");
    reserve_execution_geometry_workspace(static_cast<int>(edges));

    const auto xyz_device=Kokkos::subview(
        execution_prepared_xyz,std::make_pair(std::size_t(0),xyz.size()));
    const auto distances_device=Kokkos::subview(
        execution_prepared_distances,
        std::make_pair(std::size_t(0),distances.size()));
    const auto xyz_host=Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            xyz.data(),xyz.size());
    const auto distances_host=Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            distances.data(),distances.size());
    try {
        Kokkos::deep_copy(factorized_execution_space,xyz_device,xyz_host);
        Kokkos::deep_copy(
            factorized_execution_space,distances_device,distances_host);
        execution_geometry_copies+=2;
        compute_node_energies_forces(
            static_cast<int>(execution_prepared_node_types.extent(0)),
            Kokkos::View<const int*>(),Kokkos::View<const int*>(),
            Kokkos::View<const int*>(),Kokkos::View<const int*>(),
            xyz_device,distances_device,graph_generation);
    } catch(...) {
        const auto failure=std::current_exception();
        factorized_execution_space.fence(
            "MH-1 Execution failed evaluation host input lifetime");
        ++factorized_evaluation_fences;
        std::rethrow_exception(failure);
    }
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::compute_prepared_factorized_positions(
    const std::uint64_t graph_generation,
    const std::span<const double> positions)
{
    if(!mace_uses_prepared_execution(streamed_edges))
        throw std::invalid_argument(
            "Prepared MH-1 Execution positions require streamed_edges='factorized'.");
    if(graph_generation==0
        ||graph_generation!=factorized_prepared_graph_generation
        ||graph_generation!=factorized_prepared_geometry_graph_generation
        ||execution_source_schedule_dirty)
        throw std::invalid_argument(
            "MH-1 Execution positions require current graph and geometry tokens.");
    const int num_nodes=static_cast<int>(execution_prepared_node_types.extent(0));
    const int num_edges=static_cast<int>(execution_prepared_neigh_indices.extent(0));
    if(positions.size()!=3*static_cast<std::size_t>(num_nodes))
        throw std::invalid_argument(
            "MH-1 Execution prepared positions do not match the graph extent.");
    if(!std::all_of(positions.begin(),positions.end(),
            [](const double value) {return std::isfinite(value);}))
        throw std::invalid_argument(
            "MH-1 Execution prepared positions contain a non-finite value.");

    const auto positions_host=Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            positions.data(),positions.size());
    Kokkos::deep_copy(
        factorized_execution_space,execution_prepared_positions,positions_host);
    ++execution_geometry_copies;
    Kokkos::deep_copy(
        factorized_execution_space,execution_prepared_geometry_invalid,0);

    const auto current_positions=execution_prepared_positions;
    const auto reference_positions=execution_prepared_reference_positions;
    const auto reference_xyz=execution_prepared_reference_xyz;
    const auto cell_view=execution_prepared_cell;
    const auto inverse_cell_view=execution_prepared_inverse_cell;
    const auto pbc_view=execution_prepared_pbc;
    const auto edge_receivers=execution_prepared_targets;
    const auto edge_sources=execution_prepared_neigh_indices;
    auto displacements=execution_prepared_displacements;
    auto xyz=Kokkos::subview(execution_prepared_xyz,
        std::make_pair(std::size_t(0),3*static_cast<std::size_t>(num_edges)));
    auto distances=Kokkos::subview(execution_prepared_distances,
        std::make_pair(std::size_t(0),static_cast<std::size_t>(num_edges)));
    auto invalid=execution_prepared_geometry_invalid;
    const double cutoff=r_cut;

    Kokkos::parallel_for(
        "MaceNonlinearKokkos::prepared_atom_displacements",
        Kokkos::RangePolicy<decltype(factorized_execution_space)>(
            factorized_execution_space,0,num_nodes),
        KOKKOS_LAMBDA(const int atom) {
            double delta[3];
            double fractional[3];
            for(int component=0;component<3;++component)
                delta[component]=current_positions(3*atom+component)
                    -reference_positions(3*atom+component);
            if(pbc_view(0)==0&&pbc_view(1)==0&&pbc_view(2)==0) {
                for(int component=0;component<3;++component)
                    displacements(3*atom+component)=delta[component];
                return;
            }
            for(int lattice=0;lattice<3;++lattice) {
                fractional[lattice]=0.0;
                for(int component=0;component<3;++component)
                    fractional[lattice]+=delta[component]
                        *inverse_cell_view(3*component+lattice);
                if(pbc_view(lattice)!=0)
                    fractional[lattice]-=Kokkos::floor(
                        fractional[lattice]+0.5);
            }
            for(int component=0;component<3;++component) {
                double value=0.0;
                for(int lattice=0;lattice<3;++lattice)
                    value+=fractional[lattice]*cell_view(3*lattice+component);
                displacements(3*atom+component)=value;
            }
        });
    Kokkos::parallel_for(
        "MaceNonlinearKokkos::prepared_edge_geometry",
        Kokkos::RangePolicy<decltype(factorized_execution_space)>(
            factorized_execution_space,0,num_edges),
        KOKKOS_LAMBDA(const int edge) {
            const int receiver=edge_receivers(edge);
            const int source=edge_sources(edge);
            double vector[3];
            double squared_distance=0.0;
            for(int component=0;component<3;++component) {
                vector[component]=reference_xyz(3*edge+component)
                    +displacements(3*source+component)
                    -displacements(3*receiver+component);
                squared_distance+=vector[component]*vector[component];
            }
            if(!Kokkos::isfinite(squared_distance)
                ||!(squared_distance>0.0)) {
                Kokkos::atomic_add(&invalid(0),1);
                xyz(3*edge)=cutoff;
                xyz(3*edge+1)=0.0;
                xyz(3*edge+2)=0.0;
                distances(edge)=cutoff;
                return;
            }
            const double distance=Kokkos::sqrt(squared_distance);
            const double clamp=distance>=cutoff?cutoff/distance:1.0;
            for(int component=0;component<3;++component)
                xyz(3*edge+component)=clamp*vector[component];
            distances(edge)=clamp*distance;
        });

    compute_node_energies_forces(
        num_nodes,Kokkos::View<const int*>(),Kokkos::View<const int*>(),
        Kokkos::View<const int*>(),Kokkos::View<const int*>(),
        xyz,distances,graph_generation);
    const auto invalid_host=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),execution_prepared_geometry_invalid);
    if(invalid_host(0)!=0) {
        factorized_completed_evaluation_graph_generation=0;
        throw std::invalid_argument(
            "MH-1 Execution prepared positions produced invalid edge geometry.");
    }
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::streams_layer(int) const
{
    return streamed_edges==MACEStreamedEdgesMode::generic
        ||mace_uses_prepared_execution(streamed_edges);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::prepare_execution_source_schedule(
    int num_nodes,int samples,Kokkos::View<const int*> source_indices,
    Kokkos::View<const int*> target_indices)
{
    if(num_nodes<0||samples<0
        ||source_indices.extent(0)
            >static_cast<std::size_t>(std::numeric_limits<int>::max())
        ||target_indices.extent(0)
            >static_cast<std::size_t>(std::numeric_limits<int>::max())
        ||source_indices.extent(0)<static_cast<std::size_t>(samples)
        ||target_indices.extent(0)<static_cast<std::size_t>(samples))
        throw std::invalid_argument(
            "MH-1 Execution source schedule dimensions are inconsistent.");
    const auto host_sources=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),source_indices);
    const auto host_targets=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),target_indices);
    prepare_execution_source_schedule_host(
        num_nodes,
        std::span<const int>(host_sources.data(),samples),
        std::span<const int>(host_targets.data(),samples));
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::prepare_execution_source_schedule_host(
    const int num_nodes,const std::span<const int> source_indices,
    const std::span<const int> target_indices)
{
    if(num_nodes<0||source_indices.size()!=target_indices.size()
        ||source_indices.size()
            >static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "MH-1 Execution source schedule dimensions are inconsistent.");
    const int samples=static_cast<int>(source_indices.size());
    const auto schedule_sources=std::vector<int>(
        source_indices.begin(),source_indices.end());
    const auto schedule_targets=std::vector<int>(
        target_indices.begin(),target_indices.end());
    if(!execution_source_schedule_dirty&&num_nodes==execution_schedule_num_nodes
        &&schedule_sources==execution_schedule_sources_host
        &&schedule_targets==execution_schedule_targets_host)
        return false;
    if(factorized_prepared_graph_generation!=0)
        invalidate_factorized_prepared_graph();

    const std::int64_t block_count_64=(static_cast<std::int64_t>(samples)
        +streamed_edge_block_size-1)/streamed_edge_block_size;
    if(block_count_64>std::numeric_limits<int>::max()
        ||static_cast<std::int64_t>(samples)+1
            >std::numeric_limits<int>::max())
        throw std::overflow_error("MH-1 Execution source schedule is too large.");
    const int block_count=static_cast<int>(block_count_64);

    // Build stable block-local and graph-wide source CSRs. Segments are source
    // ordered and edges within a segment retain their original graph order.
    std::vector<int> block_source_segments;
    std::vector<int> source_edge_offsets;
    std::vector<int> source_edges;
    std::vector<int> block_receiver_counts(block_count,0);
    std::vector<int> block_receivers(samples,0);
    std::vector<int> graph_source_edge_offsets;
    std::vector<int> graph_source_edges;
    std::vector<int> graph_receivers;
    block_source_segments.reserve(static_cast<std::size_t>(block_count)+1);
    source_edge_offsets.reserve(static_cast<std::size_t>(samples)+1);
    source_edges.reserve(samples);
    block_source_segments.push_back(0);
    std::vector<std::pair<int,int>> block_edges;
    block_edges.reserve(std::min(samples,streamed_edge_block_size));
    for(int block=0;block<block_count;++block) {
        const int block_begin=block*streamed_edge_block_size;
        const int block_end=block_begin
            +std::min(streamed_edge_block_size,samples-block_begin);
        block_edges.clear();
        for(int edge=block_begin;edge<block_end;++edge) {
            const int source=schedule_sources[static_cast<std::size_t>(edge)];
            const int target=schedule_targets[static_cast<std::size_t>(edge)];
            if(source<0||source>=num_nodes||target<0||target>=num_nodes)
                throw std::invalid_argument(
                    "MH-1 Execution source schedule contains an invalid node.");
            block_edges.emplace_back(source,edge);
            if(edge==block_begin
                ||target!=schedule_targets[static_cast<std::size_t>(edge-1)])
                block_receivers[block_begin
                    +block_receiver_counts[block]++]=target;
        }
        std::sort(block_edges.begin(),block_edges.end());
        int previous_source=-1;
        for(const auto [source,edge]:block_edges) {
            if(source!=previous_source) {
                source_edge_offsets.push_back(
                    static_cast<int>(source_edges.size()));
                previous_source=source;
            }
            source_edges.push_back(edge);
        }
        block_source_segments.push_back(
            static_cast<int>(source_edge_offsets.size()));
    }
    source_edge_offsets.push_back(static_cast<int>(source_edges.size()));

    std::vector<std::pair<int,int>> graph_edges;
    graph_edges.reserve(samples);
    graph_source_edge_offsets.reserve(static_cast<std::size_t>(samples)+1);
    graph_source_edges.reserve(samples);
    graph_receivers.reserve(std::min(num_nodes,samples));
    for(int edge=0;edge<samples;++edge) {
        graph_edges.emplace_back(
            schedule_sources[static_cast<std::size_t>(edge)],edge);
        if(edge==0
            ||schedule_targets[static_cast<std::size_t>(edge)]
                !=schedule_targets[static_cast<std::size_t>(edge-1)])
            graph_receivers.push_back(
                schedule_targets[static_cast<std::size_t>(edge)]);
    }
    std::sort(graph_edges.begin(),graph_edges.end());
    int previous_graph_source=-1;
    for(const auto [source,edge]:graph_edges) {
        if(source!=previous_graph_source) {
            graph_source_edge_offsets.push_back(
                static_cast<int>(graph_source_edges.size()));
            previous_graph_source=source;
        }
        graph_source_edges.push_back(edge);
    }
    graph_source_edge_offsets.push_back(
        static_cast<int>(graph_source_edges.size()));

    ensure_view(execution_block_source_segments,
        execution_block_source_segments_storage,
        static_cast<int>(block_source_segments.size()));
    ensure_view(execution_source_edge_offsets,execution_source_edge_offsets_storage,
        static_cast<int>(source_edge_offsets.size()));
    ensure_view(execution_source_edges,execution_source_edges_storage,
        static_cast<int>(source_edges.size()));
    ensure_view(execution_block_receivers,execution_block_receivers_storage,samples);
    ensure_view(execution_graph_source_edge_offsets,
        execution_graph_source_edge_offsets_storage,
        static_cast<int>(graph_source_edge_offsets.size()));
    ensure_view(execution_graph_source_edges,execution_graph_source_edges_storage,
        static_cast<int>(graph_source_edges.size()));
    ensure_view(execution_graph_receivers,execution_graph_receivers_storage,
        static_cast<int>(graph_receivers.size()));
    const auto copy_to_device=[](auto destination,const std::vector<int>& values) {
        auto host=Kokkos::create_mirror_view(destination);
        for(std::size_t index=0;index<values.size();++index)
            host(index)=values[index];
        Kokkos::deep_copy(destination,host);
    };
    copy_to_device(execution_block_source_segments,block_source_segments);
    copy_to_device(execution_source_edge_offsets,source_edge_offsets);
    copy_to_device(execution_source_edges,source_edges);
    copy_to_device(execution_block_receivers,block_receivers);
    copy_to_device(execution_graph_source_edge_offsets,graph_source_edge_offsets);
    copy_to_device(execution_graph_source_edges,graph_source_edges);
    copy_to_device(execution_graph_receivers,graph_receivers);
    execution_graph_source_owner_count=
        static_cast<int>(graph_source_edge_offsets.size())-1;
    execution_graph_receiver_count=static_cast<int>(graph_receivers.size());
    execution_block_source_segments_host=std::move(block_source_segments);
    execution_block_receiver_counts_host=std::move(block_receiver_counts);
    execution_schedule_sources_host=schedule_sources;
    execution_schedule_targets_host=schedule_targets;
    execution_schedule_num_nodes=num_nodes;
    execution_source_schedule_dirty=false;
    ++factorized_schedule_builds;
    return true;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::release_layer_edge_workspace(int layer)
{
    if(layer<0||layer>=static_cast<int>(states.size()))return;
    auto& state=states[layer];
#define SYMMETRIX_CLEAR_EDGE_VIEW(name) state.name={};state.name##_storage={}
    SYMMETRIX_CLEAR_EDGE_VIEW(edge_features);
    SYMMETRIX_CLEAR_EDGE_VIEW(raw_weights);
    SYMMETRIX_CLEAR_EDGE_VIEW(weights);
    SYMMETRIX_CLEAR_EDGE_VIEW(edge_up);
    SYMMETRIX_CLEAR_EDGE_VIEW(edge_messages);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_embedding);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_embedding_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_graph_embedding);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_graph_embedding_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_graph_fixed_contribution);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_graph_harmonic_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_input_ir_mul);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_output_ir_mul);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_input_adjoint_ir_mul);
    SYMMETRIX_CLEAR_EDGE_VIEW(convolution_contributions);
    SYMMETRIX_CLEAR_EDGE_VIEW(density_contributions);
    SYMMETRIX_CLEAR_EDGE_VIEW(density_matrix);
    SYMMETRIX_CLEAR_EDGE_VIEW(edge_message_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(edge_up_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(edge_harmonic_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(weight_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(raw_weight_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(edge_feature_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(density_raw_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(density_feature_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(density_raw);
    SYMMETRIX_CLEAR_EDGE_VIEW(density_base);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_cutoff_adj);
    SYMMETRIX_CLEAR_EDGE_VIEW(execution_graph_cutoff_adj);
#undef SYMMETRIX_CLEAR_EDGE_VIEW
    interactions[layer].convolution_weights.clear_workspace();
    interactions[layer].density.clear_workspace();
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::edge_workspace_rows() const
{
    int rows=0;
    for(int layer=0;layer<static_cast<int>(states.size());++layer) {
        const auto& state=states[layer];
#define SYMMETRIX_MAX_EDGE_ROWS(name) rows=std::max(rows,state.name##_storage.extent_int(0))
        SYMMETRIX_MAX_EDGE_ROWS(edge_features);
        SYMMETRIX_MAX_EDGE_ROWS(raw_weights);
        SYMMETRIX_MAX_EDGE_ROWS(weights);
        SYMMETRIX_MAX_EDGE_ROWS(edge_up);
        SYMMETRIX_MAX_EDGE_ROWS(edge_messages);
        SYMMETRIX_MAX_EDGE_ROWS(execution_embedding);
        SYMMETRIX_MAX_EDGE_ROWS(execution_embedding_adj);
        SYMMETRIX_MAX_EDGE_ROWS(execution_graph_embedding);
        SYMMETRIX_MAX_EDGE_ROWS(execution_graph_embedding_adj);
        SYMMETRIX_MAX_EDGE_ROWS(execution_graph_fixed_contribution);
        SYMMETRIX_MAX_EDGE_ROWS(execution_graph_harmonic_adj);
        SYMMETRIX_MAX_EDGE_ROWS(execution_input_ir_mul);
        SYMMETRIX_MAX_EDGE_ROWS(execution_output_ir_mul);
        SYMMETRIX_MAX_EDGE_ROWS(execution_input_adjoint_ir_mul);
        SYMMETRIX_MAX_EDGE_ROWS(convolution_contributions);
        SYMMETRIX_MAX_EDGE_ROWS(density_contributions);
        SYMMETRIX_MAX_EDGE_ROWS(density_matrix);
        SYMMETRIX_MAX_EDGE_ROWS(edge_message_adj);
        SYMMETRIX_MAX_EDGE_ROWS(edge_up_adj);
        SYMMETRIX_MAX_EDGE_ROWS(edge_harmonic_adj);
        SYMMETRIX_MAX_EDGE_ROWS(weight_adj);
        SYMMETRIX_MAX_EDGE_ROWS(raw_weight_adj);
        SYMMETRIX_MAX_EDGE_ROWS(edge_feature_adj);
        SYMMETRIX_MAX_EDGE_ROWS(density_raw_adj);
        SYMMETRIX_MAX_EDGE_ROWS(density_feature_adj);
        SYMMETRIX_MAX_EDGE_ROWS(density_raw);
        SYMMETRIX_MAX_EDGE_ROWS(density_base);
        SYMMETRIX_MAX_EDGE_ROWS(execution_cutoff_adj);
        SYMMETRIX_MAX_EDGE_ROWS(execution_graph_cutoff_adj);
#undef SYMMETRIX_MAX_EDGE_ROWS
    }
    if(conditioned_mlp_workspace)
        rows=std::max(rows,conditioned_mlp_workspace->workspace_rows());
    for(const auto& storage:execution_mh1_graph_embedding_scratch_storage)
        rows=std::max(rows,storage.extent_int(0));
    for(const auto& storage:execution_mh1_graph_embedding_adjoint_scratch_storage)
        rows=std::max(rows,storage.extent_int(0));
    rows=std::max(
        rows,execution_mh1_graph_harmonic_adjoint_scratch_storage.extent_int(0));
    rows=std::max(
        rows,execution_mh1_graph_cutoff_adjoint_scratch_storage.extent_int(0));
    rows=std::max(rows,execution_mh1_spline_intervals_storage.extent_int(0));
    rows=std::max(rows,execution_mh1_spline_coordinates_storage.extent_int(0));
    return rows;
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::edge_workspace_bytes() const
{
    std::size_t bytes=0;
    for(int layer=0;layer<static_cast<int>(states.size());++layer) {
        const auto& state=states[layer];
#define SYMMETRIX_ADD_EDGE_BYTES(name) bytes+=sizeof(Precision)*state.name##_storage.size()
        SYMMETRIX_ADD_EDGE_BYTES(edge_features);
        SYMMETRIX_ADD_EDGE_BYTES(raw_weights);
        SYMMETRIX_ADD_EDGE_BYTES(weights);
        SYMMETRIX_ADD_EDGE_BYTES(edge_up);
        SYMMETRIX_ADD_EDGE_BYTES(edge_messages);
        SYMMETRIX_ADD_EDGE_BYTES(execution_embedding);
        SYMMETRIX_ADD_EDGE_BYTES(execution_embedding_adj);
        SYMMETRIX_ADD_EDGE_BYTES(execution_graph_embedding);
        SYMMETRIX_ADD_EDGE_BYTES(execution_graph_embedding_adj);
        SYMMETRIX_ADD_EDGE_BYTES(execution_graph_fixed_contribution);
        SYMMETRIX_ADD_EDGE_BYTES(execution_graph_harmonic_adj);
        SYMMETRIX_ADD_EDGE_BYTES(execution_input_ir_mul);
        SYMMETRIX_ADD_EDGE_BYTES(execution_output_ir_mul);
        SYMMETRIX_ADD_EDGE_BYTES(execution_input_adjoint_ir_mul);
        SYMMETRIX_ADD_EDGE_BYTES(convolution_contributions);
        SYMMETRIX_ADD_EDGE_BYTES(density_contributions);
        SYMMETRIX_ADD_EDGE_BYTES(density_matrix);
        SYMMETRIX_ADD_EDGE_BYTES(edge_message_adj);
        SYMMETRIX_ADD_EDGE_BYTES(edge_up_adj);
        SYMMETRIX_ADD_EDGE_BYTES(edge_harmonic_adj);
        SYMMETRIX_ADD_EDGE_BYTES(weight_adj);
        SYMMETRIX_ADD_EDGE_BYTES(raw_weight_adj);
        SYMMETRIX_ADD_EDGE_BYTES(edge_feature_adj);
        SYMMETRIX_ADD_EDGE_BYTES(density_raw_adj);
        SYMMETRIX_ADD_EDGE_BYTES(density_feature_adj);
        SYMMETRIX_ADD_EDGE_BYTES(density_raw);
        SYMMETRIX_ADD_EDGE_BYTES(density_base);
        SYMMETRIX_ADD_EDGE_BYTES(execution_cutoff_adj);
        SYMMETRIX_ADD_EDGE_BYTES(execution_graph_cutoff_adj);
#undef SYMMETRIX_ADD_EDGE_BYTES
    }
    if(conditioned_mlp_workspace)
        bytes+=conditioned_mlp_workspace->bytes();
    for(const auto& storage:execution_mh1_graph_embedding_scratch_storage)
        bytes+=sizeof(Precision)*storage.size();
    for(const auto& storage:execution_mh1_graph_embedding_adjoint_scratch_storage)
        bytes+=sizeof(Precision)*storage.size();
    bytes+=sizeof(Precision)*(
        execution_mh1_graph_harmonic_adjoint_scratch_storage.size()
        +execution_mh1_graph_cutoff_adjoint_scratch_storage.size()
        +execution_mh1_input_adjoint_scratch_storage.size());
    bytes+=sizeof(int)*(execution_block_source_segments_storage.size()
        +execution_source_edge_offsets_storage.size()
        +execution_source_edges_storage.size()
        +execution_block_receivers_storage.size()
        +execution_graph_source_edge_offsets_storage.size()
        +execution_graph_source_edges_storage.size()
        +execution_graph_receivers_storage.size()
        +execution_mh1_spline_intervals_storage.size());
    bytes+=sizeof(Precision)*execution_mh1_spline_coordinates_storage.size();
    return bytes;
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::conditioned_mlp_workspace_bytes() const
{
    return conditioned_mlp_workspace?conditioned_mlp_workspace->bytes():0;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_e3_linear_backend(
    const std::string& backend)
{
    std::string selected=backend;
#ifdef KOKKOS_ENABLE_CUDA
    if(backend=="auto"&&std::is_same_v<Precision,float>&&mh1_fast_path)
        selected="packed_gemm";
#endif
    node_embedding.set_backend(selected);
    for(auto& interaction:interactions)
        interaction.set_e3_linear_backend(selected);
    for(auto& readout:readouts) readout.set_e3_linear_backend(selected);
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::e3_linear_backend() const
{
    return node_embedding.backend();
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::selected_e3_linear_backend(
    std::size_t samples) const
{
    return node_embedding.selected_backend(samples);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_execution_mh1_host_node_backend(
    const std::string& backend)
{
    if(backend!="auto"&&backend!="generated"&&backend!="kokkos")
        throw std::invalid_argument(
            "Execution MH-1 host node backend must be auto, generated, or kokkos.");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if(backend=="kokkos"
        &&device_execution_space<Kokkos::DefaultExecutionSpace>)
        throw std::invalid_argument(
            "Execution MH-1 Kokkos node fallback is currently host-only.");
#endif
    execution_mh1_host_node_backend_request=backend;
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::
selected_execution_mh1_host_node_backend() const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
        return "inactive";
#endif
    if(execution_mh1_host_node_backend_request=="kokkos") return "kokkos";
    if(execution_mh1_host_node_backend_request=="generated")
        return jit_mh1_host_plugin_v4_ready()?"generated":"unavailable";
    return jit_mh1_host_plugin_v4_ready()?"generated":"kokkos";
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::tensor_product_backend() const
{
    return std::all_of(interactions.begin(),interactions.end(),[](const auto& interaction) {
        return interaction.convolution.uses_mh1_fast_path();
    }) ? "official_kokkos" : "generic_kokkos";
}

template<typename Precision>
std::string MaceNonlinearKokkosT<Precision>::tensor_product_execution_backend() const
{
    if(interactions.empty()) return "none";
    if(mace_uses_prepared_execution(streamed_edges)) return "mixed";
    const auto selected=interactions.front().convolution.execution_backend();
    return std::all_of(
        interactions.begin(),interactions.end(),[&selected](const auto& interaction) {
            return interaction.convolution.execution_backend()==selected;
        }) ? selected : "mixed";
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::tensor_product_channel_team_size() const
{
    if(interactions.empty()) return 0;
    if(mace_uses_prepared_execution(streamed_edges)) return -1;
    const int selected=interactions.front().convolution.channel_team_size();
    return std::all_of(
        interactions.begin(),interactions.end(),[selected](const auto& interaction) {
            return interaction.convolution.channel_team_size()==selected;
        }) ? selected : -1;
}

template<typename Precision>
int MaceNonlinearKokkosT<Precision>::tensor_product_harmonic_team_size() const
{
    if(interactions.empty()) return 0;
    if(mace_uses_prepared_execution(streamed_edges)) return -1;
    const int selected=interactions.front().convolution.harmonic_team_size();
    return std::all_of(
        interactions.begin(),interactions.end(),[selected](const auto& interaction) {
            return interaction.convolution.harmonic_team_size()==selected;
        }) ? selected : -1;
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::fused_gate_normalization_reverse_available() const
{
    if constexpr(std::is_same_v<Precision,float>)
        return mh1_fast_path&&std::all_of(
            interactions.begin(),interactions.end(),[](const auto& interaction) {
                return interaction.gate.supports_fused_normalized_reverse();
            });
    if constexpr(std::is_same_v<Precision,double>
        &&std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>)
        return mh1_fast_path&&std::all_of(
            interactions.begin(),interactions.end(),[](const auto& interaction) {
                return interaction.gate.supports_fused_normalized_reverse();
            });
    return false;
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::uses_packed_node_linears() const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
        return false;
#endif
    return uses_fused_gate_normalization_reverse();
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::uses_fused_gate_normalization_reverse() const
{
    return fused_gate_normalization_reverse_enabled
        &&fused_gate_normalization_reverse_available();
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_fused_gate_normalization_reverse(bool enabled)
{
    if(enabled&&!fused_gate_normalization_reverse_available())
        throw std::invalid_argument(
            "Fused gate-normalization reverse requires a compatible host Float32 or Float64 "
            "MACE-MH-1 evaluator, or a supported Float32 device evaluator.");
    fused_gate_normalization_reverse_enabled=enabled;
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::direct_node_tensor_reverse_available() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return mh1_fast_path&&std::all_of(
            interactions.begin(),interactions.end(),[](const auto& interaction) {
                return interaction.convolution.supports_direct_node_reverse();
            });
#endif
    return false;
}

template<typename Precision>
bool MaceNonlinearKokkosT<Precision>::uses_direct_node_tensor_reverse() const
{
    return direct_node_tensor_reverse_enabled
        &&streamed_edges==MACEStreamedEdgesMode::generic
        &&direct_node_tensor_reverse_available();
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::set_direct_node_tensor_reverse(bool enabled)
{
    if(enabled&&!direct_node_tensor_reverse_available())
        throw std::invalid_argument(
            "Direct-node tensor reverse requires a compatible Float32 CUDA MACE-MH-1 evaluator.");
    if(enabled&&!direct_node_tensor_reverse_enabled
        &&streamed_edges==MACEStreamedEdgesMode::generic) {
        Kokkos::fence("MACE_Nonlinear direct-node reverse transition");
        for(auto& state:states) {
            state.edge_up_adj={};
            state.edge_up_adj_storage={};
        }
    }
    direct_node_tensor_reverse_enabled=enabled;
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::linear_workspace_bytes() const
{
    return e3_linear_workspace ? e3_linear_workspace->bytes() : 0;
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::tensor_workspace_bytes() const
{
    std::size_t bytes=0;
    for(const auto& interaction:interactions)
        bytes+=interaction.convolution.workspace_bytes();
    return bytes;
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::product_workspace_bytes() const
{
    std::size_t bytes=0;
    for(const auto& product:products)
        bytes+=product.workspace_bytes();
    return bytes;
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::node_workspace_bytes() const
{
    std::size_t bytes=sizeof(Precision)*(
        attrs_storage.size()+features_storage.size()
        +readout_contribution_storage.size());
    for(const auto& state:states) {
#define SYMMETRIX_ADD_NODE_BYTES(name) \
        bytes+=sizeof(Precision)*state.name##_storage.size()
        SYMMETRIX_ADD_NODE_BYTES(up);
        SYMMETRIX_ADD_NODE_BYTES(residual);
        SYMMETRIX_ADD_NODE_BYTES(skip);
        SYMMETRIX_ADD_NODE_BYTES(messages);
        SYMMETRIX_ADD_NODE_BYTES(linear_1_output);
        SYMMETRIX_ADD_NODE_BYTES(pre_gate);
        SYMMETRIX_ADD_NODE_BYTES(gated);
        SYMMETRIX_ADD_NODE_BYTES(interaction_output);
        SYMMETRIX_ADD_NODE_BYTES(output);
        SYMMETRIX_ADD_NODE_BYTES(execution_input_ir_mul);
        SYMMETRIX_ADD_NODE_BYTES(execution_output_ir_mul);
        SYMMETRIX_ADD_NODE_BYTES(execution_input_adjoint_ir_mul);
        SYMMETRIX_ADD_NODE_BYTES(layer_adjoint);
        SYMMETRIX_ADD_NODE_BYTES(interaction_output_adj);
        SYMMETRIX_ADD_NODE_BYTES(skip_adj);
        SYMMETRIX_ADD_NODE_BYTES(gated_adj);
        SYMMETRIX_ADD_NODE_BYTES(pre_gate_adj);
        SYMMETRIX_ADD_NODE_BYTES(linear_adj);
        SYMMETRIX_ADD_NODE_BYTES(message_adj);
        SYMMETRIX_ADD_NODE_BYTES(up_adj);
        SYMMETRIX_ADD_NODE_BYTES(densities);
        SYMMETRIX_ADD_NODE_BYTES(density_adj);
        SYMMETRIX_ADD_NODE_BYTES(normalization_inverse);
#undef SYMMETRIX_ADD_NODE_BYTES
    }
    for(const auto& readout:readouts)
        bytes+=sizeof(Precision)*(
            readout.hidden_storage.size()+readout.activated_storage.size()
            +readout.result_storage.size()+readout.seed_storage.size()
            +readout.activated_adj_storage.size()
            +readout.hidden_adj_storage.size());
    for(const auto& runtime:execution_mh1_node_runtime)
        bytes+=runtime.bytes();
    bytes+=sizeof(Precision)*(
        execution_mh1_message_adjoint_scratch_storage.size()
        +execution_mh1_up_adjoint_scratch_storage.size()
        +execution_mh1_input_adjoint_scratch_storage.size()
        +execution_mh1_density_adjoint_scratch_storage.size());
    return bytes;
}

template<typename Precision>
std::size_t MaceNonlinearKokkosT<Precision>::precision_workspace_bytes() const
{
    return edge_workspace_bytes()+linear_workspace_bytes()
        +tensor_workspace_bytes()+product_workspace_bytes();
}

template<typename Precision>
MaceNonlinearKokkosT<Precision>::MaceNonlinearKokkosT(const nlohmann::json& data)
    :node_embedding(data.at("node_embedding"))
{
    validate_mace_nonlinear_schema(data);
    mh1_pair_spline_nodes_=data.at("radial_embedding").value(
        "num_spline_points",256);
    selected_head=data.value("selected_head",std::string());available_heads=data.value("available_heads",std::vector<std::string>());
    atomic_numbers_host=data.at("atomic_numbers").get<std::vector<int>>();
    mh1_selected_model_indices_host_=data.at("model_indices").get<std::vector<int>>();
    mh1_model_atomic_numbers_host_=data.at("model_atomic_numbers").get<std::vector<int>>();
    atomic_numbers=toKokkosView("atomic numbers",atomic_numbers_host);model_indices=toKokkosView("model indices",mh1_selected_model_indices_host_);model_atomic_numbers=toKokkosView("model atomic numbers",mh1_model_atomic_numbers_host_);model_num_elements=model_atomic_numbers.size();r_cut=data.at("r_cut").get<double>();l_max=data.at("l_max").get<int>();num_lm=(l_max+1)*(l_max+1);
    const auto& radial_data=data.at("radial_embedding");apply_cutoff=radial_data.at("apply_cutoff").get<bool>();bessel_weights=toKokkosView("bessel weights",tensor_values<Precision>(radial_data.at("basis").at("weights")));num_bessel=bessel_weights.size();radial_prefactor=radial_data.at("basis").at("prefactor").get<double>();cutoff_power=radial_data.at("cutoff").at("p").get<int>();const auto& transform=radial_data.at("distance_transform");has_agnesi=transform.at("type").get<std::string>()=="agnesi";if(has_agnesi){agnesi_a=transform.at("a").get<double>();agnesi_q=transform.at("q").get<double>();agnesi_p=transform.at("p").get<double>();covalent_radii=toKokkosView("covalent radii",transform.at("covalent_radii").get<std::vector<Precision>>());}
    scale=tensor_values<double>(data.at("scale_shift").at("scale")).at(0);shift=tensor_values<double>(data.at("scale_shift").at("shift")).at(0);const auto all_e0=tensor_values<double>(data.at("atomic_energies"));std::vector<double> e0(atomic_numbers_host.size());auto indices=data.at("model_indices").get<std::vector<int>>();for(int i=0;i<e0.size();++i)e0[i]=all_e0.at(indices[i]);atomic_energies=toKokkosView("atomic energies",e0);
    for(const auto& value:data.at("interactions"))interactions.emplace_back(value);for(const auto& value:data.at("products"))products.emplace_back(value);for(const auto& value:data.at("readouts"))readouts.emplace_back(value);
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        constexpr int vector_width=3;
        if(interactions.size()==2&&products.size()==2&&readouts.size()==2
            &&!readouts[0].nonlinear
            &&products[0].mutable_output_linear()
                .supports_identity_replacement(vector_width)
            &&!interactions[0].skip.has_instruction_width(vector_width)
            &&!interactions[1].skip.has_instruction_width(vector_width)
            &&!readouts[0].linear.has_instruction_width(vector_width)
            &&interactions[1].linear_up.try_compose_input_block(
                products[0].output_linear(),vector_width))
            products[0].mutable_output_linear().replace_block_with_identity(
                vector_width);

    }
    node_embedding.set_profile_name("node_embedding");
    for(int layer=0;layer<static_cast<int>(interactions.size());++layer) {
        auto& interaction=interactions[layer];
        const auto prefix="layer_"+std::to_string(layer)+"/";
        interaction.source_embedding.set_profile_name(prefix+"source_embedding");
        interaction.target_embedding.set_profile_name(prefix+"target_embedding");
        interaction.linear_up.set_profile_name(prefix+"linear_up");
        interaction.skip.set_profile_name(prefix+"skip");
        interaction.linear_res.set_profile_name(prefix+"linear_res");
        interaction.linear_1.set_profile_name(prefix+"linear_1");
        interaction.linear_2.set_profile_name(prefix+"linear_2");
        products[layer].set_profile_name(prefix+"product_output");
        readouts[layer].linear.set_profile_name(prefix+"readout_linear");
        readouts[layer].linear_1.set_profile_name(prefix+"readout_linear_1");
        readouts[layer].linear_2.set_profile_name(prefix+"readout_linear_2");
    }
    e3_linear_workspace=
        std::make_shared<typename E3LinearKokkosT<Precision>::Workspace>();
    conditioned_mlp_workspace=
        std::make_shared<typename AffineMLPKokkosT<Precision>::Workspace>();
    node_embedding.set_workspace(e3_linear_workspace);
    for(auto& interaction:interactions) {
        interaction.set_e3_linear_workspace(e3_linear_workspace);
        interaction.set_conditioned_mlp_workspace(conditioned_mlp_workspace);
    }
    for(auto& product:products)
        product.set_e3_linear_workspace(e3_linear_workspace);
    for(auto& readout:readouts)
        readout.set_e3_linear_workspace(e3_linear_workspace);
    if(interactions.size()!=products.size()||interactions.size()!=readouts.size())
        throw std::invalid_argument("MACE_Nonlinear Kokkos layer counts are inconsistent.");
    mh1_family=analyze_mh1_family_architecture(data);
    // The Python routing layer validates the contract against the complete
    // model before setting this identity. Do not trust embedded hashes here:
    // same-sized tensor-product programs can have different semantics.
    mh1_compiled_products=std::all_of(
        products.begin(),products.end(),[](const auto& product) {
            return product.uses_compiled_plan();
        });
    mh1_external_uvu_tensors=std::all_of(
        interactions.begin(),interactions.end(),[](const auto& interaction) {
            return interaction.convolution.uses_mh1_fast_path();
        });
    const bool primitive_candidate=mh1_family.compatible
        &&mh1_compiled_products&&mh1_external_uvu_tensors;
    mh1_pair_conditioning=primitive_candidate;
    if(primitive_candidate)
        for(int layer=0;layer<static_cast<int>(interactions.size());++layer)
            mh1_pair_conditioning=interactions[layer].prepare_pair_conditioning(
                data.at("interactions").at(layer),num_bessel,model_num_elements,indices)
                &&mh1_pair_conditioning;
    mh1_fast_path=primitive_candidate&&mh1_pair_conditioning;
    if(!mh1_fast_path) {
        if(!mh1_family.compatible)
            mh1_fast_path_rejection=mh1_family.rejection_reason;
        else if(!mh1_compiled_products)
            mh1_fast_path_rejection="Compiled correlation-three product plan is unavailable.";
        else if(!mh1_external_uvu_tensors)
            mh1_fast_path_rejection="External weighted uvu tensor plan is unavailable.";
        else
            mh1_fast_path_rejection="Conditioned edge MLP plan is unavailable.";
    }
    if constexpr(std::is_same_v<Precision,float>)
        if(!mh1_fast_path)
            throw std::invalid_argument(
                "Float32 MACE_Nonlinear requires a compatible MACE-MH-1 family fast path.");
    mh1_pair_spline_lazy_=mh1_fast_path;
    mh1_pair_spline_ready_=mh1_fast_path;
    if(mh1_pair_spline_lazy_) {
        // Retain the full model-domain conditioning data and construct the
        // active pair table only after graph types are known.
        mh1_pair_spline_interactions_data_=data.at("interactions");
        mh1_pair_spline_radial_data_=radial_data;
    }
    if(mh1_pair_spline_ready_)
        mh1_edge_executor_="pair_spline_v1";
    set_e3_linear_backend("auto");
    states.resize(interactions.size());has_zbl=data.at("has_zbl").get<bool>();if(has_zbl){const auto& value=data.at("zbl");zbl=ZBLKokkos(value.at("a_exp").get<double>(),value.at("a_prefactor").get<double>(),tensor_values<double>(value.at("c")),tensor_values<double>(value.at("covalent_radii")),static_cast<int>(tensor_values<double>(value.at("p")).at(0)));}
    if(supports_streamed_edges())streamed_edges=MACEStreamedEdgesMode::direct;
    spherical_harmonics_state=std::make_unique<SphericalHarmonicsState>(l_max);
    execution_sphericart_initializations=1;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::prepare_execution_mh1_node_runtime_parameters()
{
    if(!execution_mh1_node_runtime.empty()) return;
    if(interactions.size()!=2||products.size()!=2||readouts.size()!=2)
        throw std::logic_error(
            "Execution MH-1 node runtime requires exactly two interactions.");
    execution_mh1_node_runtime.resize(2);
    auto append_linear=[](std::vector<Precision>& destination,
                          const E3LinearKokkosT<Precision>& linear) {
        auto values=linear.runtime_parameters_ir_mul();
        destination.insert(destination.end(),values.begin(),values.end());
    };
    for(int layer=0;layer<2;++layer) {
        const auto& interaction=interactions.at(layer);
        const auto& product=products.at(layer);
        const auto& readout=readouts.at(layer);
        std::vector<Precision> linear_parameters;
        if(layer==0) append_linear(linear_parameters,node_embedding);
        append_linear(linear_parameters,interaction.linear_up);
        append_linear(linear_parameters,interaction.linear_res);
        append_linear(linear_parameters,interaction.skip);
        append_linear(linear_parameters,interaction.linear_1);
        append_linear(linear_parameters,interaction.linear_2);
        append_linear(linear_parameters,product.output_linear());
        linear_parameters.push_back(interaction.alpha);
        linear_parameters.push_back(interaction.beta);

        const auto coefficients=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),product.runtime_compiled_coefficients());
        std::vector<Precision> product_parameters(
            coefficients.data(),coefficients.data()+coefficients.size());

        std::vector<Precision> readout_parameters;
        if(readout.nonlinear) {
            append_linear(readout_parameters,readout.linear_1);
            append_linear(readout_parameters,readout.linear_2);
        } else append_linear(readout_parameters,readout.linear);

        auto& runtime=execution_mh1_node_runtime.at(layer);
        runtime.linear_parameters=toKokkosView(
            layer==0?"Execution MH-1 node linear parameters 0"
                :"Execution MH-1 node linear parameters 1",
            linear_parameters);
        runtime.product_parameters=toKokkosView(
            layer==0?"Execution MH-1 node product parameters 0"
                :"Execution MH-1 node product parameters 1",
            product_parameters);
        runtime.readout_parameters=toKokkosView(
            layer==0?"Execution MH-1 node readout parameters 0"
                :"Execution MH-1 node readout parameters 1",
            readout_parameters);
    }
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::reserve_execution_mh1_node_arena(int num_nodes)
{
    if(num_nodes<0)
        throw std::invalid_argument("Execution MH-1 node count must not be negative.");
    for(int layer=0;layer<static_cast<int>(execution_mh1_node_runtime.size());
        ++layer) {
        auto& runtime=execution_mh1_node_runtime.at(layer);
        if(runtime.arena_dimension<=0)
            throw std::logic_error(
                "Execution MH-1 node arena layout has not been admitted.");
        const int rows=std::min(num_nodes,runtime.tile_rows);
        const auto& interaction=interactions.at(layer);
        const int residual=interaction.linear_res.output_dimension();
        const int gated=interaction.gate.output_size;
        const int interaction_output=interaction.linear_2.output_dimension();
        auto reserve=[&](auto& view,const int columns) {
            if(view.extent(0)<static_cast<std::size_t>(rows)
                ||view.extent(1)!=static_cast<std::size_t>(columns))
                Kokkos::realloc(Kokkos::WithoutInitializing,view,rows,columns);
        };
        reserve(runtime.arena,runtime.arena_dimension);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
            continue;
#endif
        reserve(runtime.residual,residual);
        reserve(runtime.linear_1_output,residual);
        reserve(runtime.pre_gate,residual);
        reserve(runtime.gated,gated);
        reserve(runtime.interaction_output,interaction_output);
        reserve(runtime.interaction_adjoint,interaction_output);
        reserve(runtime.gated_adjoint,gated);
        reserve(runtime.pre_gate_adjoint,residual);
        reserve(runtime.linear_1_adjoint,residual);
    }
}

#elif SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART == 3

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::compute_Y(
    Kokkos::View<const double*> xyz,const bool evaluator_stream)
{
    if(xyz.extent(0)%3!=0)
        throw std::invalid_argument(
            "Spherical-harmonic coordinates must have a multiple-of-three extent.");
    if(xyz.extent(0)/3>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(
            "Spherical-harmonic edge count exceeds the 32-bit graph ABI.");
    const int edges=static_cast<int>(xyz.extent(0)/3);
    const std::size_t harmonic_values=static_cast<std::size_t>(edges)*num_lm;
    ensure_view(Y,Y_storage,harmonic_values);
    ensure_view(Y_grad,Y_grad_storage,3*harmonic_values);
    ensure_view(
        xyz_shuffled,xyz_shuffled_storage,
        static_cast<std::size_t>(3)*edges);
    ensure_view(
        Y_grad_shuffled,Y_grad_shuffled_storage,
        3*harmonic_values);
    auto y=Y;
    auto grad=Y_grad;
    auto raw_grad=Y_grad_shuffled;
    auto shuffled=xyz_shuffled;
    const int nlm=num_lm;
    const Precision factor=Precision(2)*Precision(std::sqrt(M_PI));

#if defined(KOKKOS_ENABLE_HIP)
    const auto execution_space=evaluator_stream
        ?factorized_execution_space:Kokkos::DefaultExecutionSpace();
    Kokkos::parallel_for(
        "nonlinear shuffle xyz",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space,0,edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            shuffled(3*edge)=xyz(3*edge+2);
            shuffled(3*edge+1)=xyz(3*edge);
            shuffled(3*edge+2)=xyz(3*edge+1);
        });
    symmetrix::launch_spherical_harmonics_device(
        execution_space,xyz_shuffled,Y,Y_grad_shuffled,edges,l_max);
    if(edges>0) {
        ++execution_sphericart_launches;
        ++execution_sphericart_async_launches;
    }
    Kokkos::parallel_for(
        "nonlinear normalize harmonics",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space,0,edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            for(int lm=0;lm<nlm;++lm) {
                y(edge*nlm+lm)*=factor;
                grad((3*edge+0)*nlm+lm)=factor
                    *raw_grad((3*edge+1)*nlm+lm);
                grad((3*edge+1)*nlm+lm)=factor
                    *raw_grad((3*edge+2)*nlm+lm);
                grad((3*edge+2)*nlm+lm)=factor
                    *raw_grad((3*edge+0)*nlm+lm);
            }
        });
#elif !defined(SYMMETRIX_SPHERICART_CUDA)
    static_cast<void>(evaluator_stream);
    Kokkos::parallel_for(
        "nonlinear shuffle xyz",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0,static_cast<std::size_t>(edges)),
        KOKKOS_LAMBDA(const std::size_t edge) {
            shuffled(3*edge)=xyz(3*edge+2);
            shuffled(3*edge+1)=xyz(3*edge);
            shuffled(3*edge+2)=xyz(3*edge+1);
        });
    const auto host_xyz=Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),xyz_shuffled);
    auto host_y=Kokkos::create_mirror_view(Y);
    auto host_raw_grad=Kokkos::create_mirror_view(Y_grad_shuffled);
    spherical_harmonics_state->calculator.compute_array_with_gradients(
        host_xyz.data(),3*static_cast<std::size_t>(edges),
        host_y.data(),harmonic_values,
        host_raw_grad.data(),3*harmonic_values);
    if(edges>0)++execution_sphericart_launches;
    Kokkos::deep_copy(Y,host_y);
    Kokkos::deep_copy(Y_grad_shuffled,host_raw_grad);
    Kokkos::parallel_for(
        "nonlinear normalize harmonics",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0,static_cast<std::size_t>(edges)),
        KOKKOS_LAMBDA(const std::size_t edge) {
            for(int lm=0;lm<nlm;++lm) {
                y(edge*nlm+lm)*=factor;
                grad((3*edge+0)*nlm+lm)=factor
                    *raw_grad((3*edge+1)*nlm+lm);
                grad((3*edge+1)*nlm+lm)=factor
                    *raw_grad((3*edge+2)*nlm+lm);
                grad((3*edge+2)*nlm+lm)=factor
                    *raw_grad((3*edge+0)*nlm+lm);
            }
        });
#else
    const auto execution_space=evaluator_stream
        ?factorized_execution_space:Kokkos::DefaultExecutionSpace();
    Kokkos::parallel_for(
        "nonlinear shuffle xyz",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space,0,edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            shuffled(3*edge)=xyz(3*edge+2);
            shuffled(3*edge+1)=xyz(3*edge);
            shuffled(3*edge+2)=xyz(3*edge+1);
        });
    if(evaluator_stream) {
        spherical_harmonics_state->calculator.compute_with_gradients_async(
            xyz_shuffled.data(),edges,Y.data(),Y_grad_shuffled.data(),
            reinterpret_cast<void*>(execution_space.cuda_stream()));
        if(edges>0)++execution_sphericart_async_launches;
    } else {
        execution_space.fence(
            "Prepare CUDA MH-1 spherical-harmonic coordinates");
        spherical_harmonics_state->calculator.compute_with_gradients(
            xyz_shuffled.data(),edges,Y.data(),Y_grad_shuffled.data());
    }
    if(edges>0)++execution_sphericart_launches;
    Kokkos::parallel_for(
        "nonlinear normalize harmonics",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space,0,edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            for(int lm=0;lm<nlm;++lm) {
                y(edge*nlm+lm)*=factor;
                grad((3*edge+0)*nlm+lm)=factor
                    *raw_grad((3*edge+1)*nlm+lm);
                grad((3*edge+1)*nlm+lm)=factor
                    *raw_grad((3*edge+2)*nlm+lm);
                grad((3*edge+2)*nlm+lm)=factor
                    *raw_grad((3*edge+0)*nlm+lm);
            }
        });
    if(!evaluator_stream)
        execution_space.fence("Complete CUDA MH-1 spherical harmonics");
#endif
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,Kokkos::View<const int*> num_neigh,Kokkos::View<const int*> neigh_indices,Kokkos::View<const int*> neigh_types,Kokkos::View<const double*> xyz,Kokkos::View<const double*> distances,const std::uint64_t execution_graph_generation)
{
    const bool use_pair_spline=mh1_edge_executor_=="pair_spline_v1";
    if(!use_pair_spline&&mace_uses_direct_execution(streamed_edges)
        &&!jit_mh1_host_plugin_ready()&&!jit_mh1_host_plugin_v4_ready()
        &&!jit_mh1_cuda_plugin_ready()&&!jit_mh1_cuda_plugin_v4_ready())
        throw std::runtime_error(
            "streamed_edges='direct' requires a loaded MH-1 RTC artifact; "
            "direct execution does not fall back to another algorithm.");
    if(execution_graph_generation!=0) {
        factorized_completed_evaluation_graph_generation=0;
        if(!mace_uses_prepared_execution(streamed_edges))
            throw std::invalid_argument(
                "An execution graph token requires streamed_edges='direct' or "
                "'receiver_factorized'.");
        if(execution_graph_generation!=factorized_prepared_graph_generation
            ||execution_source_schedule_dirty)
            throw std::invalid_argument(
                "Execution R1 graph token is stale; prepare the graph again.");
        if(num_nodes!=static_cast<int>(execution_prepared_node_types.extent(0))
            ||distances.extent(0)!=execution_prepared_neigh_indices.extent(0)
            ||xyz.extent(0)!=3*execution_prepared_neigh_indices.extent(0))
            throw std::invalid_argument(
                "Execution R1 prepared coordinates do not match the graph extents.");
        node_types=execution_prepared_node_types;
        num_neigh=execution_prepared_num_neigh;
        neigh_indices=execution_prepared_neigh_indices;
        neigh_types=execution_prepared_neigh_types;
        ++factorized_prepared_evaluations;
        ++factorized_topology_validation_skips;
    } else if(mace_uses_prepared_execution(streamed_edges)) {
        ++factorized_fallback_evaluations;
        ++factorized_topology_validations;
    }
    if(num_nodes==std::numeric_limits<int>::max()
        ||distances.extent(0)
            >static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(
            "MACE_Nonlinear Kokkos graph exceeds int32 indexing limits.");
    if(num_nodes<0||node_types.extent(0)!=static_cast<std::size_t>(num_nodes)||num_neigh.extent(0)!=static_cast<std::size_t>(num_nodes)||neigh_indices.extent(0)!=distances.extent(0)||neigh_types.extent(0)!=distances.extent(0)||xyz.extent(0)!=3*distances.extent(0))throw std::invalid_argument("MACE_Nonlinear Kokkos graph input sizes are inconsistent.");
    const int edges=distances.size();
    if(execution_graph_generation==0) {
        const int local_types=atomic_numbers_host.size();
        int invalid_nodes=0;
        Kokkos::parallel_reduce("validate nonlinear nodes",num_nodes,KOKKOS_LAMBDA(int node,int& invalid){if(node_types(node)<0||node_types(node)>=local_types||num_neigh(node)<0)++invalid;},invalid_nodes);
        if(invalid_nodes)throw std::invalid_argument("MACE_Nonlinear Kokkos graph has an invalid node type or neighbor count.");
        long long edge_total=0;
        Kokkos::parallel_reduce("validate nonlinear neighbor total",num_nodes,KOKKOS_LAMBDA(int node,long long& total){total+=num_neigh(node);},edge_total);
        if(edge_total!=edges)throw std::invalid_argument("MACE_Nonlinear Kokkos neighbor counts do not match the edge arrays.");
        int invalid_edges=0;
        Kokkos::parallel_reduce("validate nonlinear edges",edges,KOKKOS_LAMBDA(int edge,int& invalid){if(neigh_indices(edge)<0||neigh_indices(edge)>=num_nodes||neigh_types(edge)<0||neigh_types(edge)>=local_types||neigh_types(edge)!=node_types(neigh_indices(edge))||!Kokkos::isfinite(distances(edge))||!(distances(edge)>Precision(0))||!Kokkos::isfinite(xyz(3*edge))||!Kokkos::isfinite(xyz(3*edge+1))||!Kokkos::isfinite(xyz(3*edge+2)))++invalid;},invalid_edges);
        if(invalid_edges)throw std::invalid_argument("MACE_Nonlinear Kokkos graph has an invalid edge index, type, distance, or vector.");
    }
    Kokkos::View<const int*> spline_node_types=node_types;
    Kokkos::View<const int*> spline_neigh_types=neigh_types;
    int spline_type_count=static_cast<int>(atomic_numbers_host.size());
    if(use_pair_spline) {
        prepare_mh1_pair_spline_for_graph(node_types,neigh_types);
        if(mh1_pair_spline_lazy_) {
            spline_node_types=mh1_spline_node_types_;
            spline_neigh_types=mh1_spline_neigh_types_;
            spline_type_count=mh1_spline_type_count_;
        }
    }
    compute_Y(xyz,mace_uses_prepared_execution(streamed_edges));

    if(execution_graph_generation!=0) {
        offsets=execution_prepared_offsets;
        targets=execution_prepared_targets;
    } else {
        ensure_view(offsets,offsets_storage,num_nodes+1);
        ensure_view(targets,targets_storage,edges);
        ordered_kokkos_deep_copy(offsets,0);
        auto local_offsets_for_build=offsets;
        Kokkos::parallel_scan("nonlinear edge offsets",num_nodes,
            KOKKOS_LAMBDA(int node,int& update,bool final) {
                const int value=num_neigh(node);
                if(final) {
                    local_offsets_for_build(node)=update;
                    if(node==num_nodes-1)
                        local_offsets_for_build(num_nodes)=update+value;
                }
                update+=value;
            });
        auto local_targets_for_build=targets;
        Kokkos::parallel_for(
            "nonlinear edge targets",num_nodes,KOKKOS_LAMBDA(int node) {
                for(int edge=local_offsets_for_build(node);
                    edge<local_offsets_for_build(node+1);++edge)
                    local_targets_for_build(edge)=node;
            });
        if(mace_uses_prepared_execution(streamed_edges))
            prepare_execution_source_schedule(
                num_nodes,edges,neigh_indices,local_targets_for_build);
    }
    auto local_offsets=offsets;
    auto local_targets=targets;

    bool generated_node_program=false;
    bool generated_device_spline_node_adapter=false;
    bool generated_device_spline_r_program=false;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
            generated_node_program=
                mace_uses_direct_execution(streamed_edges)
                &&jit_mh1_cuda_plugin_v4_ready();
            generated_device_spline_r_program=
                generated_node_program&&use_pair_spline
                &&jit_mh1_cuda_plugin_v4->has_spline_r_program();
            generated_device_spline_node_adapter=
                generated_node_program&&use_pair_spline
                &&!generated_device_spline_r_program;
    } else
#endif
    if constexpr(std::is_same_v<Precision,float>)
        generated_node_program=
            !use_pair_spline&&mace_uses_direct_execution(streamed_edges)
            &&jit_mh1_host_plugin_v4_ready()
            &&execution_mh1_host_node_backend_request!="kokkos";
    Kokkos::View<int*> execution_mh1_spline_intervals;
    Kokkos::View<Precision*> execution_mh1_spline_coordinates;
    if(generated_device_spline_node_adapter) {
        if(execution_mh1_spline_intervals_storage.extent(0)
            <static_cast<std::size_t>(edges))
            Kokkos::realloc(Kokkos::WithoutInitializing,
                execution_mh1_spline_intervals_storage,edges);
        if(execution_mh1_spline_coordinates_storage.extent(0)
            <static_cast<std::size_t>(edges))
            Kokkos::realloc(Kokkos::WithoutInitializing,
                execution_mh1_spline_coordinates_storage,edges);
        execution_mh1_spline_intervals=Kokkos::subview(
            execution_mh1_spline_intervals_storage,std::make_pair(0,edges));
        execution_mh1_spline_coordinates=Kokkos::subview(
            execution_mh1_spline_coordinates_storage,std::make_pair(0,edges));
    }
    ensure_view(product_elements,product_elements_storage,num_nodes);
    auto indices=model_indices;
    auto elements=product_elements;
    if(generated_node_program) {
        attrs={};
        attrs_storage={};
        features={};
        features_storage={};
        Kokkos::parallel_for(
            "nonlinear product elements",num_nodes,KOKKOS_LAMBDA(int node) {
                elements(node)=indices(node_types(node));
            });
    } else {
        ensure_view(attrs,attrs_storage,num_nodes,model_num_elements);
        ordered_kokkos_deep_copy(attrs,Precision(0));
        auto local_attrs=attrs;
        Kokkos::parallel_for(
            "nonlinear attrs",num_nodes,KOKKOS_LAMBDA(int node) {
                const int element=indices(node_types(node));
                local_attrs(node,element)=Precision(1);
                elements(node)=element;
            });
        ensure_view(features,features_storage,num_nodes,
            node_embedding.output_dimension());
        node_embedding.evaluate(attrs,features);
    }
    bool generated_graphwide_program=false;
    bool generated_spline_r_program=false;
    if constexpr(std::is_same_v<Precision,float>) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
            generated_graphwide_program=
                !use_pair_spline&&mace_uses_direct_execution(streamed_edges)&&edges>0
                &&(jit_mh1_cuda_plugin_ready()
                    ||jit_mh1_cuda_plugin_v4_ready());
        } else
#endif
        {
            generated_graphwide_program=
                !use_pair_spline&&mace_uses_direct_execution(streamed_edges)
                &&edges>0&&(jit_mh1_host_plugin_ready()
                    ||jit_mh1_host_plugin_v4_ready());
        }
    }
    generated_spline_r_program=
        use_pair_spline&&mace_uses_direct_execution(streamed_edges)
        &&edges>0&&uses_packed_node_linears()&&jit_mh1_host_plugin_v5_ready()
        &&std::all_of(
            interactions.begin(),interactions.end(),
            [](const auto& interaction) {
                return interaction.convolution.execution_output_mask_is_identity();
            });
    execution_mh1_retain_graph_embedding.assign(interactions.size(),0);
    execution_mh1_graph_embedding_scratch_slot.assign(interactions.size(),-1);
    execution_mh1_scratch_minimum=0;
    execution_mh1_scratch_planned=0;
    execution_mh1_scratch_retained=0;
    execution_mh1_scratch_recomputed=0;
    execution_mh1_scratch_recomputed_layers=0;
    std::vector<std::size_t> graph_embedding_bytes(interactions.size(),0);
    std::vector<int> scratch_widths;
    std::vector<std::vector<int>> scratch_layers;
    if(generated_graphwide_program) {
        for(int layer=0;layer<static_cast<int>(interactions.size());++layer) {
            auto& interaction=interactions[layer];
            const int width=interaction.convolution_weights
                .prefix_output_size(num_bessel);
            auto found=std::find(scratch_widths.begin(),scratch_widths.end(),width);
            int slot=static_cast<int>(found-scratch_widths.begin());
            if(found==scratch_widths.end()) {
                scratch_widths.push_back(width);
                scratch_layers.emplace_back();
            }
            execution_mh1_graph_embedding_scratch_slot[layer]=slot;
            scratch_layers[slot].push_back(layer);
            graph_embedding_bytes[layer]=sizeof(Precision)
                *static_cast<std::size_t>(edges)
                *static_cast<std::size_t>(width);
        }
        for(int slot=0;slot<static_cast<int>(scratch_layers.size());++slot) {
            const auto bytes=graph_embedding_bytes[scratch_layers[slot].front()];
            execution_mh1_scratch_minimum+=bytes;
        }
        if(execution_mh1_scratch_budget==std::numeric_limits<std::size_t>::max()) {
            for(const auto& layers:scratch_layers)
                for(const int layer:layers)
                    execution_mh1_retain_graph_embedding[layer]=1;
        } else {
            std::size_t planned=execution_mh1_scratch_minimum;
            std::vector<int> retained_per_slot(scratch_layers.size(),0);
            for(int layer=static_cast<int>(interactions.size())-1;
                layer>=0;--layer) {
                const int slot=execution_mh1_graph_embedding_scratch_slot[layer];
                if(slot<0)continue;
                const bool replaces_phase_buffer=retained_per_slot[slot]+1
                    ==static_cast<int>(scratch_layers[slot].size());
                const std::size_t increment=replaces_phase_buffer
                    ?0:graph_embedding_bytes[layer];
                if(increment!=0
                    &&(planned>execution_mh1_scratch_budget
                        ||increment>execution_mh1_scratch_budget-planned))
                    continue;
                execution_mh1_retain_graph_embedding[layer]=1;
                planned+=increment;
                ++retained_per_slot[slot];
            }
        }
        for(int layer=0;layer<static_cast<int>(interactions.size());++layer) {
            if(graph_embedding_bytes[layer]==0)continue;
            if(execution_mh1_retain_graph_embedding[layer])
                execution_mh1_scratch_retained+=graph_embedding_bytes[layer];
            else {
                execution_mh1_scratch_recomputed+=graph_embedding_bytes[layer];
                ++execution_mh1_scratch_recomputed_layers;
            }
        }
        execution_mh1_scratch_planned=execution_mh1_scratch_retained;
        for(const auto& layers:scratch_layers)
            if(std::any_of(layers.begin(),layers.end(),[&](const int layer) {
                    return !execution_mh1_retain_graph_embedding[layer];
                }))
                execution_mh1_scratch_planned+=graph_embedding_bytes[layers.front()];
    }
    if(scratch_widths!=execution_mh1_graph_embedding_scratch_widths) {
        execution_mh1_graph_embedding_scratch_storage.clear();
        execution_mh1_graph_embedding_adjoint_scratch_storage.clear();
        execution_mh1_graph_embedding_scratch_widths=scratch_widths;
    }
    execution_mh1_graph_embedding_scratch_storage.resize(scratch_widths.size());
    execution_mh1_graph_embedding_adjoint_scratch_storage.resize(
        scratch_widths.size());
    for(int slot=0;slot<static_cast<int>(scratch_layers.size());++slot) {
        const bool needs_scratch=std::any_of(
            scratch_layers[slot].begin(),scratch_layers[slot].end(),
            [&](const int layer) {
                return !execution_mh1_retain_graph_embedding[layer];
            });
        if(!needs_scratch)
            execution_mh1_graph_embedding_scratch_storage[slot]={};
    }
    for(int layer=0;layer<static_cast<int>(states.size());++layer) {
        auto& state=states[layer];
        state.execution_graph_embedding={};
        if(layer>=static_cast<int>(execution_mh1_retain_graph_embedding.size())
            ||!execution_mh1_retain_graph_embedding[layer])
            state.execution_graph_embedding_storage={};
        state.execution_graph_embedding_adj={};
        state.execution_graph_embedding_adj_storage={};
        state.execution_graph_harmonic_adj={};
        state.execution_graph_harmonic_adj_storage={};
        state.execution_graph_cutoff_adj={};
        state.execution_graph_cutoff_adj_storage={};
    }
    if(generated_node_program) reserve_execution_mh1_node_arena(num_nodes);
    ensure_view(node_energies,node_energies_storage,num_nodes);
    ordered_kokkos_deep_copy(node_energies,Precision(0));
    ensure_view(readout_contribution,readout_contribution_storage,num_nodes);

    if(use_pair_spline) {
        cutoffs={};
        radial={};
    } else {
        ensure_view(cutoffs,cutoffs_storage,edges);
        ensure_view(radial,radial_storage,edges,num_bessel);
    }
    ensure_view(edge_harmonics,edge_harmonics_storage,edges,num_lm);
    auto local_cutoffs=cutoffs;
    auto local_radial=radial;
    auto local_harmonics=edge_harmonics;
    auto bw=bessel_weights;
    auto radii=covalent_radii;
    auto model_z=model_atomic_numbers;
    auto flat_y=Y;
    const Precision rc=r_cut,prefactor=radial_prefactor,a=agnesi_a,q=agnesi_q,p=agnesi_p;
    const int cp=cutoff_power,nb=num_bessel,nlm=num_lm;
    const bool agnesi=has_agnesi,embed_cutoff=apply_cutoff,
        pair_spline_execution=use_pair_spline;
    Kokkos::parallel_for("nonlinear radial",edges,KOKKOS_LAMBDA(int edge) {
        if(!pair_spline_execution) {
            const Precision distance=distances(edge);
            const Precision x=distance/rc;
            const Precision envelope=distance<rc
                ?Precision(1)-Precision(0.5)*(cp+Precision(1))
                        *(cp+Precision(2))*Kokkos::pow(x,cp)
                    +cp*(cp+Precision(2))*Kokkos::pow(x,cp+1)
                    -Precision(0.5)*cp*(cp+Precision(1))
                        *Kokkos::pow(x,cp+2)
                :Precision(0);
            local_cutoffs(edge)=envelope;
            Precision transformed=distance;
            if(agnesi) {
                const int source_z=model_z(indices(neigh_types(edge)));
                const int target_z=model_z(
                    indices(node_types(local_targets(edge))));
                const Precision r0=
                    Precision(0.5)*(radii(source_z)+radii(target_z));
                const Precision ratio=distance/r0;
                transformed=Precision(1)/(Precision(1)+a*Kokkos::pow(ratio,q)
                    /(Precision(1)+Kokkos::pow(ratio,q-p)));
            }
            for(int k=0;k<nb;++k)
                local_radial(edge,k)=prefactor*Kokkos::sin(bw(k)*transformed)
                    /transformed*(embed_cutoff?envelope:Precision(1));
        }
        for(int lm=0;lm<nlm;++lm)
            local_harmonics(edge,lm)=flat_y(edge*nlm+lm);
    });

    const auto launch_generated_conditioning_forward=
        [&](const int layer,Interaction& interaction,LayerState& state) {
        const auto convolution_source=
            interaction.convolution_source_contributions;
        const auto convolution_target=
            interaction.convolution_target_contributions;
        const auto density_source=interaction.density_source_contributions;
        const auto density_target=interaction.density_target_contributions;
        const auto convolution_parameters=interaction.convolution_weights
            .packed_runtime_parameters();
        const auto density_parameters=interaction.density
            .packed_runtime_parameters();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        if constexpr(std::is_same_v<Precision,float>
            &&device_execution_space<Kokkos::DefaultExecutionSpace>) {
            std::uint32_t flags=0;
            if(!apply_cutoff)
                flags|=SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3;
            const SymmetrixJitMH1CudaConditioningForwardArgsV3 args{
                sizeof(SymmetrixJitMH1CudaConditioningForwardArgsV3),
                static_cast<std::uint32_t>(layer),flags,
                static_cast<std::uint32_t>(execution_graph_receiver_count),
                num_nodes,edges,
                reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                reinterpret_cast<const std::int32_t*>(local_targets.data()),
                reinterpret_cast<const std::int32_t*>(node_types.data()),
                reinterpret_cast<const std::int32_t*>(
                    execution_graph_receivers.data()),
                reinterpret_cast<const std::int32_t*>(local_offsets.data()),
                radial.data(),cutoffs.data(),convolution_source.data(),
                convolution_target.data(),density_source.data(),
                density_target.data(),convolution_parameters.data(),
                density_parameters.data(),state.execution_graph_embedding.data(),
                state.densities.data()};
            Kokkos::DefaultExecutionSpace execution_space;
            ExecutionMH1CudaDeviceGuard device_guard(
                execution_mh1_device_ordinal(execution_space));
            const int persistent_blocks=execution_mh1_persistent_blocks();
            check_execution_mh1_cuda_status(
                jit_mh1_cuda_plugin_v4_ready()
                    ?jit_mh1_cuda_plugin_v4->launch_conditioning_forward(
                        &args,execution_mh1_device_stream(execution_space),
                        persistent_blocks)
                    :jit_mh1_cuda_plugin->descriptor()
                        .conditioning_launches[layer].forward_launch(
                            &args,execution_mh1_device_stream(execution_space),
                            persistent_blocks),
                "Launching graph-wide Execution MH-1 CUDA conditioning forward plugin");
            ++execution_mh1_generated_conditioning_forward_launches;
        }
        else
#endif
        if constexpr(std::is_same_v<Precision,float>) {
            std::uint32_t flags=0;
            if(!apply_cutoff)
                flags|=SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3;
            const SymmetrixJitMH1HostConditioningForwardArgsV3 args{
                sizeof(SymmetrixJitMH1HostConditioningForwardArgsV3),
                static_cast<std::uint32_t>(layer),flags,
                static_cast<std::uint32_t>(execution_graph_receiver_count),
                num_nodes,edges,
                reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                reinterpret_cast<const std::int32_t*>(local_targets.data()),
                reinterpret_cast<const std::int32_t*>(node_types.data()),
                reinterpret_cast<const std::int32_t*>(
                    execution_graph_receivers.data()),
                reinterpret_cast<const std::int32_t*>(local_offsets.data()),
                radial.data(),cutoffs.data(),convolution_source.data(),
                convolution_target.data(),density_source.data(),
                density_target.data(),convolution_parameters.data(),
                density_parameters.data(),state.execution_graph_embedding.data(),
                state.densities.data()};
            const auto owner=jit_mh1_host_plugin_v4_ready()
                ?jit_mh1_host_plugin_v4->descriptor()
                    .conditioning[layer].forward_owner
                :jit_mh1_host_plugin->descriptor()
                    .conditioning[layer].forward_owner;
            Kokkos::parallel_for(
                "ExecutionMH1HostPlugin::conditioning_forward",
                execution_graph_receiver_count,
                [=](int owner_index) { owner(&args,owner_index); });
            ++execution_mh1_generated_conditioning_forward_launches;
        }
    };

    const auto recompute_identity_graph_embedding=
        [&](Interaction& interaction,LayerState& state) {
        const int convolution_width=
            interaction.convolution_source_contributions.extent(1);
        const auto convolution_source=
            interaction.convolution_source_contributions;
        const auto convolution_target=
            interaction.convolution_target_contributions;
        for(int first_edge=0;first_edge<edges;
            first_edge+=streamed_edge_block_size) {
            const int samples=std::min(
                streamed_edge_block_size,edges-first_edge);
            ensure_view(state.convolution_contributions,
                state.convolution_contributions_storage,
                samples,convolution_width);
            auto contributions=state.convolution_contributions;
            Kokkos::parallel_for(
                "Execution MH-1 recompute identity-prefix conditioning",samples,
                KOKKOS_LAMBDA(int local_edge) {
                    const int edge=first_edge+local_edge;
                    const int source_type=neigh_types(edge);
                    const int target_type=node_types(local_targets(edge));
                    for(int column=0;column<convolution_width;++column)
                        contributions(local_edge,column)=
                            convolution_source(source_type,column)
                            +convolution_target(target_type,column);
                });
            auto radial_block=Kokkos::subview(
                radial,std::make_pair(first_edge,first_edge+samples),
                Kokkos::ALL);
            state.execution_embedding=Kokkos::subview(
                state.execution_graph_embedding,
                std::make_pair(first_edge,first_edge+samples),Kokkos::ALL);
            interaction.convolution_weights.evaluate_conditioned_prefix(
                radial_block,state.convolution_contributions,
                state.execution_embedding);
        }
    };

    for(int layer=0;layer<static_cast<int>(interactions.size());++layer) {
        auto& interaction=interactions[layer];
        auto& state=states[layer];
        const auto stage=symmetrix::mh1::layer_from_index(layer);
        state.input=features;
        const int input_width=interaction.linear_up.input_dimension();
        const int up_width=interaction.linear_up.output_dimension();
        const int res_width=interaction.linear_res.output_dimension();
        const int skip_width=interaction.skip.output_dimension();
        const int message_width=interaction.convolution.output_dimension();
        const int output_width=interaction.linear_2.output_dimension();
        const int layer_output_width=products[layer].output_dimension();
        const bool spline_layer=use_pair_spline&&interaction.pair_spline_ready;
        bool packed_spline_messages=false;
        {
        ScopedKokkosProfileRegion profile(symmetrix::mh1::region_name(
            symmetrix::mh1::r_forward_regions,stage));
        if(generated_node_program) {
            ensure_view(state.execution_input_ir_mul,
                state.execution_input_ir_mul_storage,num_nodes,up_width);
            if(generated_device_spline_node_adapter)
                ensure_view(state.up,state.up_storage,num_nodes,up_width);
            else state.up=state.execution_input_ir_mul;
            const auto& runtime=execution_mh1_node_runtime.at(layer);
            const SymmetrixJitMH1HostNodeForwardArgsV4 host_args{
                sizeof(SymmetrixJitMH1HostNodeForwardArgsV4),
                static_cast<std::uint32_t>(layer),
                SYMMETRIX_JIT_MH1_HOST_NODE_PRE_FORWARD_V4,0u,num_nodes,
                reinterpret_cast<const std::int32_t*>(product_elements.data()),
                nullptr,
                reinterpret_cast<const float*>(runtime.linear_parameters.data()),
                reinterpret_cast<const float*>(runtime.product_parameters.data()),
                reinterpret_cast<const float*>(runtime.readout_parameters.data()),
                layer==0?nullptr:reinterpret_cast<const float*>(features.data()),
                nullptr,nullptr,
                reinterpret_cast<float*>(
                    generated_device_spline_node_adapter
                        ?state.execution_input_ir_mul.data():state.up.data()),
                nullptr,nullptr,
                reinterpret_cast<float*>(runtime.arena.data())};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
                Kokkos::DefaultExecutionSpace execution_space;
                const int persistent_blocks=execution_mh1_persistent_blocks();
                for(int first=0;first<num_nodes;first+=runtime.tile_rows) {
                    const int rows=std::min(
                        runtime.tile_rows,num_nodes-first);
                    auto args=device_node_forward_tile_args(
                        host_args,first,rows,input_width,up_width,message_width,
                        layer_output_width,0,0,nullptr,nullptr,
                        sizeof(Precision));
                    check_execution_mh1_cuda_status(
                        jit_mh1_cuda_plugin_v4->launch_node_forward(
                            &args,execution_mh1_device_stream(execution_space),
                            persistent_blocks),
                        "Launching Execution MH-1 v4 CUDA node pre-forward");
                }
            } else
#endif
            {
                const auto owner=jit_mh1_host_plugin_v4->descriptor()
                    .node_programs[layer].forward_phases[0].owner;
                for(int first=0;first<num_nodes;first+=runtime.tile_rows) {
                    const int rows=std::min(runtime.tile_rows,num_nodes-first);
                    const auto input_pointer=layer==0?nullptr:
                        reinterpret_cast<const float*>(features.data())
                        +static_cast<std::size_t>(first)*features.extent(1);
                    auto tile_args=host_args;
                    tile_args.num_nodes=rows;
                    tile_args.element_indices+=first;
                    tile_args.layer_input=input_pointer;
                    tile_args.up_output+=static_cast<std::size_t>(first)*up_width;
                    Kokkos::parallel_for(
                        "ExecutionMH1HostPluginV4::node_pre_forward",rows,
                        [=](int node) { owner(&tile_args,node); });
                }
            }
            if(generated_device_spline_node_adapter) {
                const auto input_mul_to_ir=
                    interaction.convolution.execution_input_1_mul_to_ir();
                auto input_ir_mul=state.execution_input_ir_mul;
                auto input_mul_ir=state.up;
                Kokkos::parallel_for(
                    "Execution MH-1 spline device input ir-mul to mul-ir",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                        {0,0},{num_nodes,up_width}),
                    KOKKOS_LAMBDA(int node,int mul_ir) {
                        input_mul_ir(node,mul_ir)=
                            input_ir_mul(node,input_mul_to_ir(mul_ir));
                    });
            }
        } else {
            ensure_view(state.up,state.up_storage,num_nodes,up_width);
            interaction.linear_up.evaluate(features,state.up);
        }

        if(generated_node_program) {
            ensure_view(state.execution_output_ir_mul,
                state.execution_output_ir_mul_storage,num_nodes,message_width);
            if(generated_device_spline_node_adapter)
                ensure_view(
                    state.messages,state.messages_storage,num_nodes,message_width);
            else {
                state.messages={};
                state.messages_storage={};
                state.messages=state.execution_output_ir_mul;
            }
        } else
            ensure_view(state.messages,state.messages_storage,num_nodes,message_width);
        ensure_view(state.densities,state.densities_storage,num_nodes);
        auto messages=state.messages;
        auto densities=state.densities;
        const bool stream_layer=streams_layer(layer);
        if(stream_layer) {
            ordered_kokkos_deep_copy(messages,Precision(0));
            ordered_kokkos_deep_copy(densities,Precision(0));
            const int convolution_width=
                interaction.convolution_source_contributions.extent(1);
            const int density_width=interaction.density_source_contributions.extent(1);
            const int weight_width=interaction.convolution.weight_size();
            auto convolution_source=interaction.convolution_source_contributions;
            auto convolution_target=interaction.convolution_target_contributions;
            auto density_source=interaction.density_source_contributions;
            auto density_target=interaction.density_target_contributions;
            auto up=state.up;
            const bool execution_layer=
                mace_uses_prepared_execution(streamed_edges);
            const bool generated_execution_layer=
                mace_uses_direct_execution(streamed_edges);
            bool generated_spline_forward=false;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
                generated_spline_forward=false;
            else
#endif
            generated_spline_forward=spline_layer&&generated_execution_layer
                &&edges>0&&generated_spline_r_program;
            const bool generated_device_spline_forward=
                spline_layer&&edges>0&&generated_device_spline_r_program;
            bool direct_device_spline_forward=false;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
                direct_device_spline_forward=spline_layer
                    &&generated_execution_layer&&edges>0
                    &&!generated_device_spline_r_program;
#endif
            bool generated_graphwide=false;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(std::is_same_v<Precision,float>
                &&device_execution_space<Kokkos::DefaultExecutionSpace>)
                generated_graphwide=
                    !spline_layer&&generated_execution_layer&&edges>0&&(jit_mh1_cuda_plugin_ready()
                        ||jit_mh1_cuda_plugin_v4_ready());
            else
#endif
            if constexpr(std::is_same_v<Precision,float>)
                generated_graphwide=
                    !spline_layer&&generated_execution_layer&&edges>0&&(jit_mh1_host_plugin_ready()
                        ||jit_mh1_host_plugin_v4_ready());
            const int embedding_width=execution_layer
                ?interaction.convolution_weights.prefix_output_size(num_bessel):0;
            const bool identity_prefix=execution_layer
                &&interaction.convolution_weights
                    .factorized_prefix_is_identity(num_bessel);
            const bool generated_conditioning=
                generated_graphwide&&!identity_prefix;
            if(generated_graphwide) {
                if(!execution_mh1_retain_graph_embedding.at(layer)) {
                    const int slot=
                        execution_mh1_graph_embedding_scratch_slot.at(layer);
                    ensure_view(state.execution_graph_embedding,
                        execution_mh1_graph_embedding_scratch_storage.at(slot),
                        edges,embedding_width);
                } else
                    ensure_view(state.execution_graph_embedding,
                        state.execution_graph_embedding_storage,
                        edges,embedding_width);
                if(identity_prefix)
                    ensure_view(state.execution_graph_fixed_contribution,
                        state.execution_graph_fixed_contribution_storage,
                        edges,convolution_width);
                ensure_view(state.execution_input_ir_mul,
                    state.execution_input_ir_mul_storage,num_nodes,up_width);
                if(!generated_node_program)
                    ensure_view(state.execution_output_ir_mul,
                        state.execution_output_ir_mul_storage,
                        num_nodes,message_width);
                if(!generated_node_program) {
                    const auto input_mul_to_ir=
                        interaction.convolution.execution_input_1_mul_to_ir();
                    auto input_ir_mul=state.execution_input_ir_mul;
                    Kokkos::parallel_for(
                        "Execution MH-1 input mul-ir to ir-mul",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,up_width}),
                        KOKKOS_LAMBDA(int node,int mul_ir) {
                            input_ir_mul(node,input_mul_to_ir(mul_ir))=
                                up(node,mul_ir);
                        });
                }
                // Generated node programs also accumulate graph messages into
                // this buffer; their output is produced only after the graph
                // wide forward, so it must start from zero here as well.
                ordered_kokkos_deep_copy(
                    state.execution_output_ir_mul,Precision(0));
            }
            if(generated_conditioning)
                launch_generated_conditioning_forward(layer,interaction,state);
            if(generated_spline_forward) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
                    throw std::logic_error(
                        "MH-1 spline host R forward selected for a device backend.");
                } else
#endif
                {
                    ensure_view(state.execution_input_ir_mul,
                        state.execution_input_ir_mul_storage,num_nodes,up_width);
                    const auto& spline_program=jit_mh1_host_plugin_v5
                        ->descriptor().spline_r_programs[layer];
                    packed_spline_messages=(spline_program.flags
                        &SYMMETRIX_JIT_MH1_HOST_SPLINE_R_PACKED_FORWARD_MESSAGES_V5)
                        !=0;
                    if(!packed_spline_messages)
                        ensure_view(state.execution_output_ir_mul,
                            state.execution_output_ir_mul_storage,
                            num_nodes,message_width);
                    const auto input_mul_to_ir=
                        interaction.convolution.execution_input_1_mul_to_ir();
                    auto input_ir_mul=state.execution_input_ir_mul;
                    Kokkos::parallel_for(
                        "Execution MH-1 spline input mul-ir to ir-mul",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,up_width}),
                        KOKKOS_LAMBDA(int node,int mul_ir) {
                            input_ir_mul(node,input_mul_to_ir(mul_ir))=
                                up(node,mul_ir);
                        });
                    if(!packed_spline_messages)
                        ordered_kokkos_deep_copy(
                            state.execution_output_ir_mul,Precision(0));
                    const auto pair_spline=interaction.pair_spline;
                    const auto output_mask=
                        interaction.convolution.execution_output_mask_ir_mul();
                    const SymmetrixJitMH1HostSplineRForwardArgsV5 args{
                        sizeof(SymmetrixJitMH1HostSplineRForwardArgsV5),
                        static_cast<std::uint32_t>(layer),0u,0u,
                        num_nodes,edges,
                        static_cast<std::int32_t>(spline_type_count),
                        pair_spline.edge_type_count(),
                        pair_spline.interval_count(),pair_spline.function_count(),
                        pair_spline.spline_h(),pair_spline.spline_x0(),
                        reinterpret_cast<const std::int32_t*>(
                            neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_targets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_offsets.data()),
                            reinterpret_cast<const std::int32_t*>(spline_neigh_types.data()),
                            reinterpret_cast<const std::int32_t*>(spline_node_types.data()),
                        distances.data(),pair_spline.coefficient_data(),
                        edge_harmonics.data(),output_mask.data(),
                        state.execution_input_ir_mul.data(),
                        packed_spline_messages?messages.data()
                            :state.execution_output_ir_mul.data(),
                        densities.data()};
                    const auto owner=spline_program.forward_owner;
                    auto graph_receivers=execution_graph_receivers;
                    Kokkos::parallel_for(
                        "ExecutionMH1HostPluginV5::spline_r_forward",
                        execution_graph_receiver_count,
                        [=](int owner_index) {
                            owner(&args,graph_receivers(owner_index));
                        });
                    if(!packed_spline_messages) {
                        const auto output_mul_to_ir=
                            interaction.convolution.execution_output_mul_to_ir();
                        auto output_ir_mul=state.execution_output_ir_mul;
                        Kokkos::parallel_for(
                            "Execution MH-1 spline output ir-mul to mul-ir",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{num_nodes,message_width}),
                            KOKKOS_LAMBDA(int node,int mul_ir) {
                                messages(node,mul_ir)=output_ir_mul(
                                    node,output_mul_to_ir(mul_ir));
                            });
                    }
                    ++execution_mh1_generated_forward_launches;
                }
            }
            if(generated_device_spline_forward) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
                    ordered_kokkos_deep_copy(
                        state.execution_output_ir_mul,Precision(0));
                    const auto pair_spline=interaction.pair_spline;
                    const auto output_mask=
                        interaction.convolution.execution_output_mask_ir_mul();
                    const SymmetrixJitMH1CudaSplineV5 radial_packet{
                        sizeof(SymmetrixJitMH1CudaSplineV5),
                        static_cast<std::uint32_t>(
                            pair_spline.edge_type_count()),
                        static_cast<std::uint32_t>(
                            pair_spline.interval_count()),
                        static_cast<std::uint32_t>(
                            pair_spline.function_count()),
                        pair_spline.spline_h(),pair_spline.spline_x0(),
                        pair_spline.coefficient_data()};
                    const SymmetrixJitMH1CudaSplineRForwardArgsV5 args{
                        sizeof(SymmetrixJitMH1CudaSplineRForwardArgsV5),
                        static_cast<std::uint32_t>(layer),
                        static_cast<std::uint32_t>(spline_type_count),0u,
                        num_nodes,edges,
                        reinterpret_cast<const std::int32_t*>(spline_node_types.data()),
                        reinterpret_cast<const std::int32_t*>(num_neigh.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_offsets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(spline_neigh_types.data()),
                        distances.data(),radial_packet,edge_harmonics.data(),
                        state.execution_input_ir_mul.data(),output_mask.data(),
                        state.densities.data(),
                        state.execution_output_ir_mul.data()};
                    Kokkos::DefaultExecutionSpace execution_space;
                    check_execution_mh1_cuda_status(
                        jit_mh1_cuda_plugin_v4->launch_spline_r_forward(
                            &args,execution_mh1_device_stream(execution_space),
                            execution_mh1_persistent_blocks()),
                        "Launching generated Execution MH-1 spline R forward");
                    ++execution_mh1_generated_forward_launches;
                }
#endif
            }
            if(direct_device_spline_forward) {
                if(layer==0)
                    interaction.pair_spline.prepare_evaluation_points(
                        distances,execution_mh1_spline_intervals,
                        execution_mh1_spline_coordinates);
                ordered_kokkos_deep_copy(
                    state.execution_output_ir_mul,Precision(0));
                const bool dispatched=interaction.convolution
                    .try_evaluate_execution_spline_uvu_from_nodes(
                        state.execution_input_ir_mul,neigh_indices,local_targets,
                        execution_graph_receivers,local_offsets,
                        spline_neigh_types,spline_node_types,distances,
                        execution_mh1_spline_intervals,
                        execution_mh1_spline_coordinates,
                        spline_type_count,
                        interaction.pair_spline,edge_harmonics,
                        state.execution_output_ir_mul,state.densities);
                if(!dispatched)
                    throw std::logic_error(
                        "Qualified MH-1 direct spline UVU forward did not dispatch.");
            }
            if(!generated_conditioning&&!generated_spline_forward
                &&!generated_device_spline_forward
                &&!direct_device_spline_forward)
            for(int first_edge=0;first_edge<edges;first_edge+=streamed_edge_block_size) {
                const int samples=std::min(streamed_edge_block_size,edges-first_edge);
                if(!spline_layer&&(!execution_layer||identity_prefix))
                    ensure_view(state.convolution_contributions,
                        state.convolution_contributions_storage,
                        samples,convolution_width);
                if(!spline_layer&&!execution_layer)
                    ensure_view(state.density_contributions,
                        state.density_contributions_storage,samples,density_width);
                if(!spline_layer&&!execution_layer) {
                    auto convolution_contributions=
                        state.convolution_contributions;
                    auto density_contributions=state.density_contributions;
                    Kokkos::parallel_for(
                        "stream nonlinear edge conditioning",samples,
                        KOKKOS_LAMBDA(int local_edge) {
                            const int edge=first_edge+local_edge;
                            const int source_type=neigh_types(edge);
                            const int target_type=
                                node_types(local_targets(edge));
                            for(int column=0;column<convolution_width;++column)
                                convolution_contributions(local_edge,column)=
                                    convolution_source(source_type,column)
                                    +convolution_target(target_type,column);
                            for(int column=0;column<density_width;++column)
                                density_contributions(local_edge,column)=
                                    density_source(source_type,column)
                                    +density_target(target_type,column);
                        });
                } else if(!spline_layer&&identity_prefix) {
                    auto convolution_contributions=
                        state.convolution_contributions;
                    Kokkos::parallel_for(
                        "Execution MH-1 identity-prefix conditioning",samples,
                        KOKKOS_LAMBDA(int local_edge) {
                            const int edge=first_edge+local_edge;
                            const int source_type=neigh_types(edge);
                            const int target_type=
                                node_types(local_targets(edge));
                            for(int column=0;column<convolution_width;++column)
                                convolution_contributions(local_edge,column)=
                                    convolution_source(source_type,column)
                                    +convolution_target(target_type,column);
                        });
                }
                auto radial_block=Kokkos::subview(
                    radial,std::make_pair(first_edge,first_edge+samples),Kokkos::ALL);
                auto harmonics_block=Kokkos::subview(
                    edge_harmonics,std::make_pair(first_edge,first_edge+samples),Kokkos::ALL);
                if(execution_layer&&!spline_layer) {
                    const int block=first_edge/streamed_edge_block_size;
                    const int receiver_count=
                        execution_block_receiver_counts_host.at(block);
                    auto active_receivers=Kokkos::subview(
                        execution_block_receivers,
                        std::make_pair(first_edge,first_edge+receiver_count));
                    Kokkos::View<const Precision**,Kokkos::LayoutRight>
                        edge_linear_contribution;
                    if(identity_prefix&&!generated_graphwide)
                        edge_linear_contribution=state.convolution_contributions;
                    if(generated_graphwide) {
                        state.execution_embedding=Kokkos::subview(
                            state.execution_graph_embedding,
                            std::make_pair(first_edge,first_edge+samples),
                            Kokkos::ALL);
                        if(identity_prefix) {
                            auto fixed_block=Kokkos::subview(
                                state.execution_graph_fixed_contribution,
                                std::make_pair(first_edge,first_edge+samples),
                                Kokkos::ALL);
                            auto contributions=state.convolution_contributions;
                            Kokkos::parallel_for(
                                "Execution MH-1 retain graph fixed contribution",
                                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                    {0,0},{samples,convolution_width}),
                                KOKKOS_LAMBDA(int edge,int column) {
                                    fixed_block(edge,column)=
                                        contributions(edge,column);
                                });
                            edge_linear_contribution=fixed_block;
                        }
                    } else {
                        ensure_view(
                            state.execution_embedding,state.execution_embedding_storage,
                            samples,embedding_width);
                    }
                    if(identity_prefix)
                        interaction.convolution_weights
                            .evaluate_conditioned_prefix(
                                radial_block,state.convolution_contributions,
                                state.execution_embedding);
                    else
                        interaction.convolution_weights
                            .evaluate_conditioned_prefix_indexed(
                                radial_block,neigh_types,local_targets,node_types,
                                first_edge,convolution_source,
                                convolution_target,state.execution_embedding);
                    bool dispatched=generated_graphwide;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                    if constexpr(std::is_same_v<Precision,float>
                        &&device_execution_space<
                            Kokkos::DefaultExecutionSpace>) {
                        if(generated_execution_layer&&!generated_graphwide
                            &&jit_mh1_cuda_plugin_ready()) {
                            std::uint32_t flags=0;
                            if(!apply_cutoff)
                                flags|=
                                    SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3;
                            const auto linear_bias=interaction.convolution_weights
                                .final_linear_bias();
                            if(linear_bias.extent(0)!=0)
                                flags|=
                                    SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3;
                            if(edge_linear_contribution.size()!=0)
                                flags|=
                                    SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3;
                            const auto linear_weight=interaction.convolution_weights
                                .final_linear_weight_phi_major(num_bessel);
                            const auto output_mask=
                                interaction.convolution.execution_output_mask();
                            const SymmetrixJitMH1CudaForwardArgsV3 args{
                                sizeof(SymmetrixJitMH1CudaForwardArgsV3),
                                static_cast<std::uint32_t>(layer),flags,
                                static_cast<std::uint32_t>(receiver_count),
                                num_nodes,edges,first_edge,samples,
                                reinterpret_cast<const std::int32_t*>(
                                    neigh_indices.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    active_receivers.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    local_offsets.data()),
                                state.execution_embedding.data(),linear_weight.data(),
                                linear_bias.data(),edge_linear_contribution.data(),
                                harmonics_block.data(),cutoffs.data(),
                                state.up.data(),output_mask.data(),
                                state.messages.data()};
                            // Kokkos::Cuda's default instance and the surrounding
                            // unqualified Kokkos launches share the singleton stream.
                            Kokkos::DefaultExecutionSpace execution_space;
                            ExecutionMH1CudaDeviceGuard device_guard(
                                execution_mh1_device_ordinal(execution_space));
                            const int persistent_blocks=
                                execution_mh1_persistent_blocks();
                            const auto& launches=jit_mh1_cuda_plugin
                                ->descriptor().launches[layer];
                            check_execution_mh1_cuda_status(
                                launches.forward_launch(
                                    &args,execution_mh1_device_stream(
                                        execution_space),
                                    persistent_blocks),
                                "Launching the Execution MH-1 CUDA forward plugin");
                            ++execution_mh1_generated_forward_launches;
                            dispatched=true;
                        }
                    }
                    else
#endif
                    if constexpr(std::is_same_v<Precision,float>) {
                        if(generated_execution_layer&&!generated_graphwide
                            &&jit_mh1_host_plugin_ready()) {
                            std::uint32_t flags=0;
                            if(!apply_cutoff)
                                flags|=SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3;
                            const auto linear_bias=interaction.convolution_weights
                                .final_linear_bias();
                            if(linear_bias.extent(0)!=0)
                                flags|=SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3;
                            if(edge_linear_contribution.size()!=0)
                                flags|=
                                    SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3;
                            const auto linear_weight=interaction.convolution_weights
                                .final_linear_weight(num_bessel);
                            const auto output_mask=
                                interaction.convolution.execution_output_mask();
                            const SymmetrixJitMH1HostForwardArgsV3 args{
                                sizeof(SymmetrixJitMH1HostForwardArgsV3),
                                static_cast<std::uint32_t>(layer),flags,0u,
                                num_nodes,edges,first_edge,samples,
                                reinterpret_cast<const std::int32_t*>(
                                    neigh_indices.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    local_targets.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    local_offsets.data()),
                                state.execution_embedding.data(),linear_weight.data(),
                                linear_bias.data(),edge_linear_contribution.data(),
                                harmonics_block.data(),cutoffs.data(),
                                output_mask.data(),state.up.data(),state.messages.data()};
                            const auto owner=jit_mh1_host_plugin
                                ->descriptor().forward_owner;
                            Kokkos::parallel_for(
                                "ExecutionMH1HostPlugin::forward",receiver_count,
                                [=](int owner_index) {
                                    owner(&args,active_receivers(owner_index));
                                });
                            ++execution_mh1_generated_forward_launches;
                            dispatched=true;
                        }
                    }
                    if(!dispatched)
                        dispatched=interaction.convolution
                        .try_evaluate_execution_uvu_from_nodes(
                            state.up,neigh_indices,local_targets,active_receivers,
                            local_offsets,
                            first_edge,state.execution_embedding,
                            interaction.convolution_weights
                                .final_linear_weight(num_bessel),
                            interaction.convolution_weights.final_linear_bias(),
                            edge_linear_contribution,
                            harmonics_block,cutoffs,!apply_cutoff,state.messages);
                    if(!dispatched)
                        throw std::logic_error(
                            "Qualified MH-1 Execution UVU forward did not dispatch.");
                } else {
                    ensure_view(state.weights,state.weights_storage,
                        samples,weight_width);
                    if(spline_layer) {
                        const auto pair_spline=interaction.pair_spline;
                        const int type_count=spline_type_count;
                        auto weights=state.weights;
                        Kokkos::parallel_for(
                            "MH-1 pair spline R1 weights",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{samples,weight_width}),
                            KOKKOS_LAMBDA(int local_edge,int function) {
                                const int edge=first_edge+local_edge;
                                const int pair=spline_neigh_types(edge)*type_count
                                    +spline_node_types(local_targets(edge));
                                weights(local_edge,function)=
                                    pair_spline.evaluate_function(
                                        pair,distances(edge),function);
                            });
                    } else {
                        ensure_view(state.raw_weights,state.raw_weights_storage,
                            samples,weight_width);
                        interaction.convolution_weights.evaluate_conditioned(
                            radial_block,state.convolution_contributions,
                            state.raw_weights);
                        ordered_kokkos_deep_copy(state.weights,state.raw_weights);
                        if(!apply_cutoff) {
                            auto weights=state.weights;
                            Kokkos::parallel_for(
                                "stream nonlinear weight cutoff",weights.size(),
                                KOKKOS_LAMBDA(std::size_t flat) {
                                    const std::size_t local_edge=
                                        flat/weights.extent(1);
                                    weights(local_edge,flat%weights.extent(1))
                                        *=local_cutoffs(first_edge+local_edge);
                                });
                        }
                    }
                    ensure_view(state.edge_up,state.edge_up_storage,samples,up_width);
                    auto edge_up=state.edge_up;
                    Kokkos::parallel_for(
                        "stream nonlinear gather edge up",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{samples,up_width}),
                        KOKKOS_LAMBDA(int local_edge,int k) {
                            edge_up(local_edge,k)=
                                up(neigh_indices(first_edge+local_edge),k);
                        });
                    ensure_view(state.edge_messages,state.edge_messages_storage,
                        samples,message_width);
                    interaction.convolution.evaluate(
                        state.edge_up,harmonics_block,state.weights,
                        state.edge_messages);
                    auto edge_messages=state.edge_messages;
                    Kokkos::parallel_for(
                        "stream nonlinear gather messages",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,message_width}),
                        KOKKOS_LAMBDA(int node,int k) {
                            const int begin=local_offsets(node)>first_edge
                                ?local_offsets(node):first_edge;
                            const int block_end=first_edge+samples;
                            const int end=local_offsets(node+1)<block_end
                                ?local_offsets(node+1):block_end;
                            Precision value=Precision(0);
                            for(int edge=begin;edge<end;++edge)
                                value+=edge_messages(edge-first_edge,k);
                            messages(node,k)+=value;
                        });
                }
                ensure_view(state.density_matrix,state.density_matrix_storage,samples,1);
                if(spline_layer) {
                    const auto pair_spline=interaction.pair_spline;
                    const int density_function=interaction.pair_spline_weight_count;
                    const int type_count=spline_type_count;
                    auto density_matrix=state.density_matrix;
                    Kokkos::parallel_for(
                        "MH-1 pair spline R1 density",samples,
                        KOKKOS_LAMBDA(int local_edge) {
                            const int edge=first_edge+local_edge;
                            const int pair=spline_neigh_types(edge)*type_count
                                +spline_node_types(local_targets(edge));
                            density_matrix(local_edge,0)=
                                pair_spline.evaluate_function(
                                    pair,distances(edge),density_function);
                        });
                } else if(execution_layer)
                    interaction.density.evaluate_conditioned_indexed(
                        radial_block,neigh_types,local_targets,node_types,
                        first_edge,density_source,density_target,
                        state.density_matrix);
                else
                    interaction.density.evaluate_conditioned(
                        radial_block,state.density_contributions,
                        state.density_matrix);
                auto density_matrix=state.density_matrix;
                if(execution_layer) {
                    const int block=first_edge/streamed_edge_block_size;
                    const int receiver_count=
                        execution_block_receiver_counts_host.at(block);
                    auto active_receivers=Kokkos::subview(
                        execution_block_receivers,
                        std::make_pair(first_edge,first_edge+receiver_count));
                    Kokkos::parallel_for(
                    "stream nonlinear gather density",receiver_count,
                    KOKKOS_LAMBDA(int owner) {
                        const int node=active_receivers(owner);
                        const int begin=local_offsets(node)>first_edge
                            ?local_offsets(node):first_edge;
                        const int block_end=first_edge+samples;
                        const int end=local_offsets(node+1)<block_end
                            ?local_offsets(node+1):block_end;
                        Precision value=Precision(0);
                        for(int edge=begin;edge<end;++edge) {
                            const Precision raw=density_matrix(edge-first_edge,0);
                            value+=spline_layer?raw:Kokkos::tanh(raw*raw)
                                *(embed_cutoff?Precision(1):local_cutoffs(edge));
                        }
                        densities(node)+=value;
                    });
                } else {
                    Kokkos::parallel_for(
                    "stream nonlinear gather density",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        const int begin=local_offsets(node)>first_edge
                            ?local_offsets(node):first_edge;
                        const int block_end=first_edge+samples;
                        const int end=local_offsets(node+1)<block_end
                            ?local_offsets(node+1):block_end;
                        Precision value=Precision(0);
                        for(int edge=begin;edge<end;++edge) {
                            const Precision raw=density_matrix(edge-first_edge,0);
                            value+=spline_layer?raw:Kokkos::tanh(raw*raw)
                                *(embed_cutoff?Precision(1):local_cutoffs(edge));
                        }
                        densities(node)+=value;
                    });
                }
            }
            if(generated_graphwide) {
                Kokkos::View<const Precision**,Kokkos::LayoutRight>
                    graph_fixed_contribution;
                if(identity_prefix)
                    graph_fixed_contribution=
                        state.execution_graph_fixed_contribution;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr(std::is_same_v<Precision,float>
                    &&device_execution_space<
                        Kokkos::DefaultExecutionSpace>) {
                    std::uint32_t flags=0;
                    if(!apply_cutoff)
                        flags|=
                            SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3;
                    const auto linear_bias=interaction.convolution_weights
                        .final_linear_bias();
                    if(linear_bias.extent(0)!=0)
                        flags|=
                            SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3;
                    if(graph_fixed_contribution.size()!=0)
                        flags|=
                            SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3;
                    const auto linear_weight=interaction.convolution_weights
                        .final_linear_weight_phi_major(num_bessel);
                    const auto output_mask=interaction.convolution
                        .execution_output_mask_ir_mul();
                    const SymmetrixJitMH1CudaForwardArgsV3 args{
                        sizeof(SymmetrixJitMH1CudaForwardArgsV3),
                        static_cast<std::uint32_t>(layer),flags,
                        static_cast<std::uint32_t>(execution_graph_receiver_count),
                        num_nodes,edges,0,edges,
                        reinterpret_cast<const std::int32_t*>(
                            neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_receivers.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_offsets.data()),
                        state.execution_graph_embedding.data(),linear_weight.data(),
                        linear_bias.data(),graph_fixed_contribution.data(),
                        edge_harmonics.data(),cutoffs.data(),
                        state.execution_input_ir_mul.data(),output_mask.data(),
                        state.execution_output_ir_mul.data()};
                    Kokkos::DefaultExecutionSpace execution_space;
                    ExecutionMH1CudaDeviceGuard device_guard(
                        execution_mh1_device_ordinal(execution_space));
                    const int persistent_blocks=execution_mh1_persistent_blocks();
                    check_execution_mh1_cuda_status(
                        jit_mh1_cuda_plugin_v4_ready()
                            ?jit_mh1_cuda_plugin_v4->launch_forward(
                                &args,execution_mh1_device_stream(
                                    execution_space),
                                persistent_blocks)
                            :jit_mh1_cuda_plugin->descriptor()
                                .launches[layer].forward_launch(
                                    &args,execution_mh1_device_stream(
                                        execution_space),
                                    persistent_blocks),
                        "Launching graph-wide Execution MH-1 CUDA forward plugin");
                    ++execution_mh1_generated_forward_launches;
                }
                else
#endif
                if constexpr(std::is_same_v<Precision,float>) {
                    std::uint32_t flags=0;
                    if(!apply_cutoff)
                        flags|=SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3;
                    const auto linear_bias=interaction.convolution_weights
                        .final_linear_bias();
                    if(linear_bias.extent(0)!=0)
                        flags|=SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3;
                    if(graph_fixed_contribution.size()!=0)
                        flags|=
                            SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3;
                    const auto linear_weight=jit_mh1_host_plugin_v4_ready()
                        ?interaction.convolution_weights
                            .final_linear_weight_phi_major(num_bessel)
                        :interaction.convolution_weights
                            .final_linear_weight(num_bessel);
                    const auto output_mask=interaction.convolution
                        .execution_output_mask_ir_mul();
                    const SymmetrixJitMH1HostForwardArgsV3 args{
                        sizeof(SymmetrixJitMH1HostForwardArgsV3),
                        static_cast<std::uint32_t>(layer),flags,0u,
                        num_nodes,edges,0,edges,
                        reinterpret_cast<const std::int32_t*>(
                            neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_targets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_offsets.data()),
                        state.execution_graph_embedding.data(),linear_weight.data(),
                        linear_bias.data(),graph_fixed_contribution.data(),
                        edge_harmonics.data(),cutoffs.data(),output_mask.data(),
                        state.execution_input_ir_mul.data(),
                        state.execution_output_ir_mul.data()};
                    const auto owner=jit_mh1_host_plugin_v4_ready()
                        ?jit_mh1_host_plugin_v4->descriptor().forward_owner
                        :jit_mh1_host_plugin->descriptor().forward_owner;
                    auto graph_receivers=execution_graph_receivers;
                    Kokkos::parallel_for(
                        "ExecutionMH1HostPlugin::graph_forward",
                        execution_graph_receiver_count,
                        [=](int owner_index) {
                            owner(&args,graph_receivers(owner_index));
                        });
                    ++execution_mh1_generated_forward_launches;
                }
                if(generated_node_program)
                    state.messages=state.execution_output_ir_mul;
                else {
                    const auto output_mul_to_ir=
                        interaction.convolution.execution_output_mul_to_ir();
                    auto output_ir_mul=state.execution_output_ir_mul;
                    Kokkos::parallel_for(
                        "Execution MH-1 output ir-mul to mul-ir",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,message_width}),
                        KOKKOS_LAMBDA(int node,int mul_ir) {
                            messages(node,mul_ir)=
                                output_ir_mul(node,output_mul_to_ir(mul_ir));
                        });
                }
            }
            if(mace_uses_prepared_execution(streamed_edges))
                ++factorized_forward_evaluations;
        } else {
        if(mh1_fast_path) {
            const int convolution_width=
                interaction.convolution_source_contributions.extent(1);
            const int density_width=interaction.density_source_contributions.extent(1);
            ensure_view(state.convolution_contributions,
                state.convolution_contributions_storage,edges,convolution_width);
            ensure_view(state.density_contributions,
                state.density_contributions_storage,edges,density_width);
            auto convolution_source=interaction.convolution_source_contributions;
            auto convolution_target=interaction.convolution_target_contributions;
            auto density_source=interaction.density_source_contributions;
            auto density_target=interaction.density_target_contributions;
            auto convolution_contributions=state.convolution_contributions;
            auto density_contributions=state.density_contributions;
            Kokkos::parallel_for("nonlinear edge conditioning",edges,KOKKOS_LAMBDA(int edge) {
                const int source_type=neigh_types(edge);
                const int target_type=node_types(local_targets(edge));
                for(int column=0;column<convolution_width;++column)
                    convolution_contributions(edge,column)=
                        convolution_source(source_type,column)+convolution_target(target_type,column);
                for(int column=0;column<density_width;++column)
                    density_contributions(edge,column)=
                        density_source(source_type,column)+density_target(target_type,column);
            });
        } else {
            ensure_view(state.source_embeddings,state.source_embeddings_storage,num_nodes,
                interaction.source_embedding.output_dimension());
            ensure_view(state.target_embeddings,state.target_embeddings_storage,num_nodes,
                interaction.target_embedding.output_dimension());
            interaction.source_embedding.evaluate(attrs,state.source_embeddings);
            interaction.target_embedding.evaluate(attrs,state.target_embeddings);
            const int edge_width=num_bessel+state.source_embeddings.extent(1)
                +state.target_embeddings.extent(1);
            ensure_view(state.edge_features,state.edge_features_storage,edges,edge_width);
            auto source_embed=state.source_embeddings;
            auto target_embed=state.target_embeddings;
            auto edge_features=state.edge_features;
            Kokkos::parallel_for("nonlinear edge features",edges,KOKKOS_LAMBDA(int edge) {
                int column=0;
                for(int k=0;k<nb;++k) edge_features(edge,column++)=local_radial(edge,k);
                const int source=neigh_indices(edge),target=local_targets(edge);
                for(int k=0;k<source_embed.extent(1);++k)
                    edge_features(edge,column++)=source_embed(source,k);
                for(int k=0;k<target_embed.extent(1);++k)
                    edge_features(edge,column++)=target_embed(target,k);
            });
        }

        ensure_view(state.raw_weights,state.raw_weights_storage,edges,
            interaction.convolution.weight_size());
        ensure_view(state.weights,state.weights_storage,edges,
            interaction.convolution.weight_size());
        if(mh1_fast_path)
            interaction.convolution_weights.evaluate_conditioned(
                radial,state.convolution_contributions,state.raw_weights);
        else interaction.convolution_weights.evaluate(state.edge_features,state.raw_weights);
        ordered_kokkos_deep_copy(state.weights,state.raw_weights);
        if(!apply_cutoff) {
            auto weights=state.weights;
            Kokkos::parallel_for("nonlinear weight cutoff",weights.size(),
                KOKKOS_LAMBDA(std::size_t flat) {
                    const std::size_t edge=flat/weights.extent(1);
                    weights(edge,flat%weights.extent(1))*=local_cutoffs(edge);
                });
        }

        ensure_view(state.edge_up,state.edge_up_storage,edges,up_width);
        auto edge_up=state.edge_up;
        auto up=state.up;
        Kokkos::parallel_for("nonlinear gather edge up",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,up_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                edge_up(edge,k)=up(neigh_indices(edge),k);
            });
        ensure_view(state.edge_messages,state.edge_messages_storage,edges,message_width);
        interaction.convolution.evaluate(
            state.edge_up,edge_harmonics,state.weights,state.edge_messages);
        auto edge_messages=state.edge_messages;
        Kokkos::parallel_for("nonlinear gather messages",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,message_width}),
            KOKKOS_LAMBDA(int node,int k) {
                Precision value=Precision(0);
                for(int edge=local_offsets(node);edge<local_offsets(node+1);++edge)
                    value+=edge_messages(edge,k);
                messages(node,k)=value;
            });

        ensure_view(state.density_matrix,state.density_matrix_storage,edges,1);
        ensure_view(state.density_raw,state.density_raw_storage,edges);
        ensure_view(state.density_base,state.density_base_storage,edges);
        if(mh1_fast_path)
            interaction.density.evaluate_conditioned(
                radial,state.density_contributions,state.density_matrix);
        else interaction.density.evaluate(state.edge_features,state.density_matrix);
        auto density_raw=state.density_raw;
        auto density_base=state.density_base;
        auto density_matrix=state.density_matrix;
        Kokkos::parallel_for("nonlinear density",edges,KOKKOS_LAMBDA(int edge) {
            const Precision raw=density_matrix(edge,0);
            const Precision base=Kokkos::tanh(raw*raw);
            density_raw(edge)=raw;
            density_base(edge)=base;
        });
        Kokkos::parallel_for("nonlinear density gather",num_nodes,KOKKOS_LAMBDA(int node) {
            Precision value=Precision(0);
            for(int edge=local_offsets(node);edge<local_offsets(node+1);++edge)
                value+=density_base(edge)*(embed_cutoff?Precision(1):local_cutoffs(edge));
            densities(node)=value;
        });
        }
        }

        if(generated_device_spline_node_adapter&&!use_pair_spline) {
            const auto output_mul_to_ir=
                interaction.convolution.execution_output_mul_to_ir();
            auto output_mul_ir=state.messages;
            auto output_ir_mul=state.execution_output_ir_mul;
            Kokkos::parallel_for(
                "Execution MH-1 spline device output mul-ir to ir-mul",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {0,0},{num_nodes,message_width}),
                KOKKOS_LAMBDA(int node,int mul_ir) {
                    output_ir_mul(node,output_mul_to_ir(mul_ir))=
                        output_mul_ir(node,mul_ir);
                });
        }
        ScopedKokkosProfileRegion profile(symmetrix::mh1::region_name(
            symmetrix::mh1::m_forward_regions,stage));
        auto densities=state.densities;
        if(!generated_node_program) {
            ensure_view(state.residual,state.residual_storage,num_nodes,res_width);
            ensure_view(state.skip,state.skip_storage,num_nodes,skip_width);
            if(uses_packed_node_linears())
                interaction.linear_res.evaluate_to_packed(
                    state.up,state.residual,false);
            else interaction.linear_res.evaluate(state.up,state.residual);
            interaction.skip.evaluate(features,state.skip);
        }
        if(generated_node_program) {
            ensure_view(state.output,state.output_storage,num_nodes,
                products[layer].output_dimension());
            const auto& runtime=execution_mh1_node_runtime.at(layer);
            if(runtime.retained_pre_gate_dimension>0)
                ensure_view(state.pre_gate,state.pre_gate_storage,num_nodes,
                    runtime.retained_pre_gate_dimension);
            else state.pre_gate={};
            if(runtime.retained_interaction_output_dimension>0)
                ensure_view(state.interaction_output,
                    state.interaction_output_storage,num_nodes,
                    runtime.retained_interaction_output_dimension);
            else state.interaction_output={};
            const auto node_up=generated_device_spline_node_adapter
                ?state.execution_input_ir_mul:state.up;
            const auto node_messages=generated_device_spline_node_adapter
                ?state.execution_output_ir_mul:state.messages;
            const SymmetrixJitMH1HostNodeForwardArgsV4 host_args{
                sizeof(SymmetrixJitMH1HostNodeForwardArgsV4),
                static_cast<std::uint32_t>(layer),
                SYMMETRIX_JIT_MH1_HOST_NODE_POST_FORWARD_V4,0u,num_nodes,
                reinterpret_cast<const std::int32_t*>(product_elements.data()),
                reinterpret_cast<const float*>(state.densities.data()),
                reinterpret_cast<const float*>(runtime.linear_parameters.data()),
                reinterpret_cast<const float*>(runtime.product_parameters.data()),
                reinterpret_cast<const float*>(runtime.readout_parameters.data()),
                layer==0?nullptr:reinterpret_cast<const float*>(features.data()),
                reinterpret_cast<const float*>(node_up.data()),
                reinterpret_cast<const float*>(node_messages.data()),nullptr,
                reinterpret_cast<float*>(state.output.data()),
                reinterpret_cast<float*>(readout_contribution.data()),
                reinterpret_cast<float*>(runtime.arena.data())};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
                Kokkos::DefaultExecutionSpace execution_space;
                const int persistent_blocks=execution_mh1_persistent_blocks();
                auto* retained_pre_gate=runtime.retained_pre_gate_dimension>0
                    ?reinterpret_cast<float*>(state.pre_gate.data()):nullptr;
                auto* retained_interaction_output=
                    runtime.retained_interaction_output_dimension>0
                        ?reinterpret_cast<float*>(
                            state.interaction_output.data()):nullptr;
                for(int first=0;first<num_nodes;first+=runtime.tile_rows) {
                    const int rows=std::min(
                        runtime.tile_rows,num_nodes-first);
                    auto args=device_node_forward_tile_args(
                        host_args,first,rows,input_width,up_width,message_width,
                        layer_output_width,runtime.retained_pre_gate_dimension,
                        runtime.retained_interaction_output_dimension,
                        retained_pre_gate,retained_interaction_output,
                        sizeof(Precision));
                    check_execution_mh1_cuda_status(
                        jit_mh1_cuda_plugin_v4->launch_node_forward(
                            &args,execution_mh1_device_stream(execution_space),
                            persistent_blocks),
                        "Launching Execution MH-1 v4 CUDA node post-forward");
                }
            } else
#endif
            {
                const auto owner=jit_mh1_host_plugin_v4->descriptor()
                    .node_programs[layer].forward_phases[1].owner;
                for(int first=0;first<num_nodes;first+=runtime.tile_rows) {
                    const int rows=std::min(runtime.tile_rows,num_nodes-first);
                    auto tile_args=host_args;
                    tile_args.num_nodes=rows;
                    tile_args.element_indices+=first;
                    tile_args.node_density+=first;
                    if(tile_args.layer_input!=nullptr)
                        tile_args.layer_input+=static_cast<std::size_t>(first)
                            *input_width;
                    tile_args.up+=static_cast<std::size_t>(first)*up_width;
                    tile_args.messages+=static_cast<std::size_t>(first)
                        *message_width;
                    tile_args.layer_output+=static_cast<std::size_t>(first)
                        *layer_output_width;
                    tile_args.readout_contribution+=first;
                    Kokkos::parallel_for(
                        "ExecutionMH1HostPluginV4::node_post_forward",rows,
                        [=](int node) { owner(&tile_args,node); });
                }
            }
            auto energies=node_energies;
            auto contribution=readout_contribution;
            Kokkos::parallel_for(
                "sum generated readout",num_nodes,KOKKOS_LAMBDA(int node) {
                    energies(node)+=contribution(node);
                });
            features=state.output;
            continue;
        }

        ensure_view(state.linear_1_output,state.linear_1_output_storage,num_nodes,
            interaction.linear_1.output_dimension());
        ensure_view(state.pre_gate,state.pre_gate_storage,num_nodes,res_width);
        ensure_view(state.gated,state.gated_storage,num_nodes,interaction.gate.output_size);
        ensure_view(state.interaction_output,state.interaction_output_storage,
            num_nodes,output_width);
        bool use_direct_block_major_product_input=false;
        if constexpr(std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>)
            use_direct_block_major_product_input=uses_packed_node_linears()
                &&products[layer].uses_standard_host_plan();
        if(uses_packed_node_linears()) {
            if(packed_spline_messages) {
                interaction.linear_1.evaluate_packed_to_packed(
                    state.messages,state.linear_1_output);
                ensure_view(state.normalization_inverse,
                    state.normalization_inverse_storage,num_nodes);
                auto normalization_inverse=state.normalization_inverse;
                auto densities=state.densities;
                const Precision alpha=interaction.alpha,beta=interaction.beta;
                Kokkos::parallel_for("mh1 packed normalization inverse",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        normalization_inverse(node)=Precision(1)
                            /(alpha+beta*densities(node));
                    });
            } else
                interaction.linear_1.evaluate_to_packed(
                    state.messages,state.linear_1_output,true,state.densities,
                    interaction.alpha,interaction.beta);
            interaction.gate.evaluate_normalized_to_packed(
                state.linear_1_output,state.residual,
                interaction.linear_res.runtime_active_output_mask(),state.densities,
                interaction.alpha,interaction.beta,state.pre_gate,state.gated,
                !packed_spline_messages,packed_spline_messages
                    ?Kokkos::View<const Precision*>(state.normalization_inverse)
                    :Kokkos::View<const Precision*>());
            if(use_direct_block_major_product_input)
                interaction.linear_2.evaluate_from_packed_to_block_major(
                    state.gated,
                    products[layer].prepare_block_major_input(num_nodes),
                    num_nodes);
            else interaction.linear_2.evaluate_from_packed(
                    state.gated,state.interaction_output);
        } else {
            interaction.linear_1.evaluate(state.messages,state.linear_1_output);
            auto linear_value=state.linear_1_output;
            auto residual=state.residual;
            auto pre_gate=state.pre_gate;
            const Precision alpha=interaction.alpha,beta=interaction.beta;
            Kokkos::parallel_for("nonlinear normalization",pre_gate.size(),
                KOKKOS_LAMBDA(std::size_t flat) {
                    const std::size_t node=flat/pre_gate.extent(1);
                    const int k=flat%pre_gate.extent(1);
                    pre_gate(node,k)=linear_value(node,k)
                        /(alpha+beta*densities(node))+residual(node,k);
                });
            interaction.gate.evaluate(state.pre_gate,state.gated);
            interaction.linear_2.evaluate(state.gated,state.interaction_output);
        }
        ensure_view(state.output,state.output_storage,num_nodes,
            products[layer].output_dimension());
        Kokkos::View<const int*> layer_elements=product_elements;
        if(products[layer].is_agnostic()) {
            ensure_view(state.agnostic_elements,state.agnostic_elements_storage,num_nodes);
            ordered_kokkos_deep_copy(state.agnostic_elements,0);
            layer_elements=state.agnostic_elements;
        }
        products[layer].evaluate(
            state.interaction_output,state.skip,layer_elements,state.output,
            execution_graph_generation==0,
            use_direct_block_major_product_input
                ?E3ProductBasisKokkosT<Precision>::InputLayout::block_major
                :E3ProductBasisKokkosT<Precision>::InputLayout::native,
            interaction.skip.active_output_prefix_dimension());
        features=state.output;
    }

    for(int layer=0;!generated_node_program
        &&layer<static_cast<int>(readouts.size());++layer) {
        auto& state=states[layer];
        readouts[layer].evaluate(state.output,readout_contribution);
        auto energies=node_energies;
        auto contribution=readout_contribution;
        Kokkos::parallel_for("sum readout",num_nodes,KOKKOS_LAMBDA(int node) {
            energies(node)+=contribution(node);
        });
        ensure_view(state.layer_adjoint,state.layer_adjoint_storage,num_nodes,
            products[layer].output_dimension());
        readouts[layer].reverse(state.output,scale,state.layer_adjoint);
    }
    if(generated_node_program)
        for(int layer=0;layer<static_cast<int>(states.size());++layer) {
            auto& state=states[layer];
            ensure_view(state.layer_adjoint,state.layer_adjoint_storage,
                num_nodes,products[layer].output_dimension());
            ordered_kokkos_deep_copy(state.layer_adjoint,Precision(0));
        }
    auto energies=node_energies;
    auto e0=atomic_energies;
    const Precision energy_scale=scale,energy_shift=shift;
    Kokkos::parallel_for("scale nonlinear energies",num_nodes,KOKKOS_LAMBDA(int node) {
        energies(node)=e0(node_types(node))+energy_scale*energies(node)+energy_shift;
    });

    ensure_view(harmonic_adjoints,harmonic_adjoints_storage,edges,num_lm);
    ordered_kokkos_deep_copy(harmonic_adjoints,Precision(0));
    if(use_pair_spline) {
        radial_adjoints={};
        cutoff_adjoints={};
        ensure_view(pair_spline_distance_adjoints,
            pair_spline_distance_adjoints_storage,edges);
        ordered_kokkos_deep_copy(
            pair_spline_distance_adjoints,Precision(0));
        if(generated_device_spline_r_program) {
            ensure_view(node_forces,node_forces_storage,xyz.size());
            ordered_kokkos_deep_copy(node_forces,0.0);
        }
    } else {
        ensure_view(radial_adjoints,radial_adjoints_storage,edges,num_bessel);
        ensure_view(cutoff_adjoints,cutoff_adjoints_storage,edges);
        ordered_kokkos_deep_copy(radial_adjoints,Precision(0));
        ordered_kokkos_deep_copy(cutoff_adjoints,Precision(0));
        pair_spline_distance_adjoints={};
    }
    auto radial_adjoint_values=radial_adjoints;
    auto harmonic_adjoint_values=harmonic_adjoints;
    auto cutoff_adjoint_values=cutoff_adjoints;

    int maximum_up_width=0;
    for(const auto& interaction:interactions)
        maximum_up_width=std::max(maximum_up_width,
            interaction.linear_up.output_dimension());
    const auto reserve=[](auto& storage,const std::size_t extent) {
        if(storage.extent(0)<extent)
            Kokkos::realloc(Kokkos::WithoutInitializing,storage,extent);
    };
    if(generated_node_program) {
        int maximum_message_width=0;
        for(const auto& interaction:interactions) {
            maximum_message_width=std::max(maximum_message_width,
                interaction.convolution.output_dimension());
        }
        for(auto& state:states) {
            if(!generated_device_spline_node_adapter) {
                state.message_adj={};
                state.message_adj_storage={};
                state.up_adj={};
                state.up_adj_storage={};
            }
            state.density_adj={};
            state.density_adj_storage={};
        }
        const bool reuse_message_adjoint=std::all_of(
            execution_mh1_node_runtime.begin(),
            execution_mh1_node_runtime.end(),
            [] (const auto& runtime) { return runtime.reuse_message_adjoint; });
        if(reuse_message_adjoint)
            execution_mh1_message_adjoint_scratch_storage={};
        else reserve(execution_mh1_message_adjoint_scratch_storage,
            static_cast<std::size_t>(num_nodes)*maximum_message_width);
        reserve(execution_mh1_up_adjoint_scratch_storage,
            static_cast<std::size_t>(num_nodes)*maximum_up_width);
        reserve(execution_mh1_density_adjoint_scratch_storage,num_nodes);
    } else {
        execution_mh1_message_adjoint_scratch_storage={};
        execution_mh1_up_adjoint_scratch_storage={};
        execution_mh1_density_adjoint_scratch_storage={};
    }
    for(auto& state:states) {
        state.execution_input_adjoint_ir_mul={};
        state.execution_input_adjoint_ir_mul_storage={};
    }
    if(generated_graphwide_program||generated_spline_r_program
        ||generated_device_spline_node_adapter) {
        reserve(execution_mh1_input_adjoint_scratch_storage,
            static_cast<std::size_t>(num_nodes)*maximum_up_width);
    } else execution_mh1_input_adjoint_scratch_storage={};

    for(int layer=static_cast<int>(interactions.size())-1;layer>=0;--layer) {
        auto& interaction=interactions[layer];
        auto& state=states[layer];
        const auto stage=symmetrix::mh1::layer_from_index(layer);
        const int input_width=interaction.linear_up.input_dimension();
        const int up_width=interaction.linear_up.output_dimension();
        const int message_width=interaction.convolution.output_dimension();
        const int interaction_width=interaction.linear_2.output_dimension();
        const int layer_output_width=products[layer].output_dimension();
        const int pre_gate_width=state.pre_gate.extent(1);
        const int skip_width=interaction.skip.output_dimension();
        const bool spline_layer=use_pair_spline&&interaction.pair_spline_ready;
        bool packed_spline_messages=false;
        if(generated_spline_r_program&&spline_layer)
            packed_spline_messages=(jit_mh1_host_plugin_v5->descriptor()
                    .spline_r_programs[layer].flags
                &SYMMETRIX_JIT_MH1_HOST_SPLINE_R_PACKED_FORWARD_MESSAGES_V5)!=0;

        {
        ScopedKokkosProfileRegion profile(symmetrix::mh1::region_name(
            symmetrix::mh1::m_reverse_regions,stage));
        if(generated_node_program) {
            // The ABI requires a packed row stride, so wrap each flat shared
            // owner with the current layer's exact width.
            const auto& runtime=execution_mh1_node_runtime.at(layer);
            auto node_message_adj=runtime.reuse_message_adjoint
                ?decltype(state.message_adj)(
                    state.execution_output_ir_mul.data(),num_nodes,message_width)
                :decltype(state.message_adj)(
                    execution_mh1_message_adjoint_scratch_storage.data(),
                    num_nodes,message_width);
            Kokkos::View<Precision**,Kokkos::LayoutRight> node_up_adj;
            if(generated_device_spline_node_adapter) {
                ensure_view(
                    state.message_adj,state.message_adj_storage,
                    num_nodes,message_width);
                ensure_view(state.up_adj,state.up_adj_storage,num_nodes,up_width);
                state.execution_input_adjoint_ir_mul=
                    decltype(state.execution_input_adjoint_ir_mul)(
                        execution_mh1_input_adjoint_scratch_storage.data(),
                        num_nodes,up_width);
                node_up_adj=state.execution_input_adjoint_ir_mul;
            } else {
                state.message_adj=node_message_adj;
                state.up_adj=decltype(state.up_adj)(
                    execution_mh1_up_adjoint_scratch_storage.data(),
                    num_nodes,up_width);
                node_up_adj=state.up_adj;
            }
            state.density_adj=decltype(state.density_adj)(
                execution_mh1_density_adjoint_scratch_storage.data(),num_nodes);
            auto previous_adjoint=layer==0
                ?Kokkos::View<Precision**,Kokkos::LayoutRight>()
                :states[layer-1].layer_adjoint;
            const auto node_up=generated_device_spline_node_adapter
                ?state.execution_input_ir_mul:state.up;
            const auto node_messages=generated_device_spline_node_adapter
                ?state.execution_output_ir_mul:state.messages;
            const SymmetrixJitMH1HostNodeReverseArgsV4 host_args{
                sizeof(SymmetrixJitMH1HostNodeReverseArgsV4),
                static_cast<std::uint32_t>(layer),
                SYMMETRIX_JIT_MH1_HOST_NODE_POST_REVERSE_V4,0u,
                static_cast<float>(scale),0u,num_nodes,
                reinterpret_cast<const std::int32_t*>(product_elements.data()),
                reinterpret_cast<const float*>(state.densities.data()),
                reinterpret_cast<const float*>(runtime.linear_parameters.data()),
                reinterpret_cast<const float*>(runtime.product_parameters.data()),
                reinterpret_cast<const float*>(runtime.readout_parameters.data()),
                layer==0?nullptr:reinterpret_cast<const float*>(state.input.data()),
                reinterpret_cast<const float*>(node_up.data()),
                reinterpret_cast<const float*>(node_messages.data()),
                reinterpret_cast<const float*>(state.output.data()),
                reinterpret_cast<float*>(runtime.arena.data()),
                reinterpret_cast<float*>(state.layer_adjoint.data()),
                reinterpret_cast<float*>(node_message_adj.data()),
                reinterpret_cast<float*>(node_up_adj.data()),
                layer==0?nullptr:reinterpret_cast<float*>(previous_adjoint.data()),
                reinterpret_cast<float*>(state.density_adj.data())};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
                Kokkos::DefaultExecutionSpace execution_space;
                const int persistent_blocks=execution_mh1_persistent_blocks();
                const auto* retained_pre_gate=
                    runtime.retained_pre_gate_dimension>0
                        ?reinterpret_cast<const float*>(state.pre_gate.data())
                        :nullptr;
                const auto* retained_interaction_output=
                    runtime.retained_interaction_output_dimension>0
                        ?reinterpret_cast<const float*>(
                            state.interaction_output.data()):nullptr;
                for(int first=0;first<num_nodes;first+=runtime.tile_rows) {
                    const int rows=std::min(
                        runtime.tile_rows,num_nodes-first);
                    auto args=device_node_reverse_tile_args(
                        host_args,first,rows,input_width,up_width,message_width,
                        layer_output_width,
                        runtime.retained_pre_gate_dimension,
                        runtime.retained_interaction_output_dimension,
                        retained_pre_gate,retained_interaction_output,
                        sizeof(Precision));
                    check_execution_mh1_cuda_status(
                        jit_mh1_cuda_plugin_v4->launch_node_reverse(
                            &args,execution_mh1_device_stream(execution_space),
                            persistent_blocks),
                        "Launching Execution MH-1 v4 CUDA node post-reverse");
                }
            } else
#endif
            {
                const auto owner=jit_mh1_host_plugin_v4->descriptor()
                    .node_programs[layer].reverse_phases[0].owner;
                for(int first=0;first<num_nodes;first+=runtime.tile_rows) {
                    const int rows=std::min(runtime.tile_rows,num_nodes-first);
                    auto tile_args=host_args;
                    tile_args.num_nodes=rows;
                    tile_args.element_indices+=first;
                    tile_args.node_density+=first;
                    if(tile_args.layer_input!=nullptr)
                        tile_args.layer_input+=static_cast<std::size_t>(first)
                            *input_width;
                    tile_args.up+=static_cast<std::size_t>(first)*up_width;
                    tile_args.messages+=static_cast<std::size_t>(first)
                        *message_width;
                    tile_args.layer_output+=static_cast<std::size_t>(first)
                        *layer_output_width;
                    tile_args.layer_output_adjoint+=
                        static_cast<std::size_t>(first)*layer_output_width;
                    tile_args.message_adjoint+=static_cast<std::size_t>(first)
                        *message_width;
                    tile_args.up_adjoint+=static_cast<std::size_t>(first)
                        *up_width;
                    if(tile_args.layer_input_adjoint!=nullptr)
                        tile_args.layer_input_adjoint+=
                            static_cast<std::size_t>(first)*input_width;
                    tile_args.node_density_adjoint+=first;
                    Kokkos::parallel_for(
                        "ExecutionMH1HostPluginV4::node_post_reverse",rows,
                        [=](int node) { owner(&tile_args,node); });
                }
            }
            if(generated_device_spline_node_adapter) {
                const auto input_mul_to_ir=
                    interaction.convolution.execution_input_1_mul_to_ir();
                const auto output_mul_to_ir=
                    interaction.convolution.execution_output_mul_to_ir();
                auto input_ir_mul_adjoint=state.execution_input_adjoint_ir_mul;
                auto input_mul_ir_adjoint=state.up_adj;
                auto output_ir_mul_adjoint=node_message_adj;
                auto output_mul_ir_adjoint=state.message_adj;
                Kokkos::parallel_for(
                    "Execution MH-1 spline device node adjoints ir-mul to mul-ir",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                        {0,0},{num_nodes,std::max(up_width,message_width)}),
                    KOKKOS_LAMBDA(int node,int column) {
                        if(column<up_width)
                            input_mul_ir_adjoint(node,column)=
                                input_ir_mul_adjoint(
                                    node,input_mul_to_ir(column));
                        if(column<message_width)
                            output_mul_ir_adjoint(node,column)=
                                output_ir_mul_adjoint(
                                    node,output_mul_to_ir(column));
                    });
            }
        } else {
            ensure_view(state.interaction_output_adj,
                state.interaction_output_adj_storage,num_nodes,interaction_width);
            ensure_view(state.skip_adj,state.skip_adj_storage,num_nodes,skip_width);
            Kokkos::View<const int*> layer_elements=products[layer].is_agnostic()
                ?Kokkos::View<const int*>(state.agnostic_elements)
                :Kokkos::View<const int*>(product_elements);
            bool use_packed_product_adjoint=false;
            if constexpr(std::is_same_v<
                typename Kokkos::DefaultExecutionSpace::memory_space,
                Kokkos::HostSpace>)
                use_packed_product_adjoint=uses_packed_node_linears()
                    &&products[layer].uses_standard_host_plan()
                    &&interaction.linear_2.supports_packed_reverse();
            products[layer].reverse(
                state.interaction_output,layer_elements,state.layer_adjoint,
                state.interaction_output_adj,state.skip_adj,
                execution_graph_generation==0,
                !use_packed_product_adjoint,
                use_packed_product_adjoint
                    ?E3ProductBasisKokkosT<Precision>::InputLayout::block_major
                    :E3ProductBasisKokkosT<Precision>::InputLayout::native,
                interaction.skip.active_output_prefix_dimension());

            ensure_view(state.gated_adj,state.gated_adj_storage,num_nodes,
                state.gated.extent(1));
            ensure_view(state.pre_gate_adj,state.pre_gate_adj_storage,num_nodes,
                pre_gate_width);
            if(use_packed_product_adjoint)
                interaction.linear_2.reverse_packed_to_packed(
                    products[layer].packed_input_adjoint(),
                    state.gated_adj);
            else if(uses_packed_node_linears())
                interaction.linear_2.reverse_to_packed(
                    state.interaction_output_adj,state.gated_adj);
            else interaction.linear_2.reverse(
                state.interaction_output_adj,state.gated_adj);

            ensure_view(state.linear_adj,state.linear_adj_storage,
                num_nodes,pre_gate_width);
            ensure_view(state.density_adj,state.density_adj_storage,num_nodes);
            auto density_adj=state.density_adj;
            auto linear_adj=packed_spline_messages
                ?state.pre_gate_adj:state.linear_adj;
            if(uses_packed_node_linears()
                &&interaction.gate.supports_fused_normalized_reverse())
                interaction.gate.reverse_normalized_packed(
                    state.pre_gate,state.gated_adj,state.linear_1_output,
                    state.densities,interaction.alpha,interaction.beta,
                    state.pre_gate_adj,linear_adj,state.density_adj,true);
            else if(uses_fused_gate_normalization_reverse()
                &&interaction.gate.supports_fused_normalized_reverse())
                interaction.gate.reverse_normalized(
                    state.pre_gate,state.gated_adj,state.linear_1_output,
                    state.densities,interaction.alpha,interaction.beta,
                    state.pre_gate_adj,state.linear_adj,state.density_adj);
            else {
                interaction.gate.reverse(
                    state.pre_gate,state.gated_adj,state.pre_gate_adj);
                ordered_kokkos_deep_copy(state.density_adj,Precision(0));
                auto linear_adj=state.linear_adj;
                auto pre_gate_adj=state.pre_gate_adj;
                auto linear_forward=state.linear_1_output;
                auto layer_densities=state.densities;
                const Precision alpha=interaction.alpha,beta=interaction.beta;
                Kokkos::parallel_for("reverse nonlinear normalization",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        const Precision normalization=
                            alpha+beta*layer_densities(node);
                        Precision density_value=Precision(0);
                        for(int k=0;k<pre_gate_width;++k) {
                            linear_adj(node,k)=pre_gate_adj(node,k)/normalization;
                            density_value-=pre_gate_adj(node,k)
                                *linear_forward(node,k)*beta
                                /(normalization*normalization);
                        }
                        density_adj(node)=density_value;
                    });
            }

            ensure_view(state.message_adj,state.message_adj_storage,
                num_nodes,message_width);
            if(packed_spline_messages)
                interaction.linear_1.reverse_packed_to_packed(
                    linear_adj,state.message_adj);
            else if(uses_packed_node_linears())
                interaction.linear_1.reverse_from_packed(
                    state.linear_adj,state.message_adj);
            else interaction.linear_1.reverse(
                state.linear_adj,state.message_adj);
            if(packed_spline_messages) {
                ensure_view(state.execution_output_ir_mul,
                    state.execution_output_ir_mul_storage,
                    num_nodes,message_width);
                interaction.linear_1.reverse_normalize_packed_rows(
                    state.messages,state.message_adj,state.densities,
                    interaction.alpha,interaction.beta,
                    state.execution_output_ir_mul,state.density_adj,false);
            }
            else if(uses_packed_node_linears()) {
                auto messages=state.messages;
                auto message_adj=state.message_adj;
                auto densities=state.densities;
                auto density_adj=state.density_adj;
                const Precision alpha=interaction.alpha,beta=interaction.beta;
                Kokkos::parallel_for("mh1 reverse normalized messages",
                    num_nodes,KOKKOS_LAMBDA(int node) {
                        const Precision inverse=Precision(1)
                            /(alpha+beta*densities(node));
                        Precision density_value=Precision(0);
                        for(std::size_t feature=0;
                            feature<messages.extent(1);++feature) {
                            const Precision value=message_adj(node,feature)*inverse;
                            message_adj(node,feature)=value;
                            density_value-=beta*inverse*value
                                *messages(node,feature);
                        }
                        density_adj(node)=density_value;
                    });
            }
            ensure_view(state.up_adj,state.up_adj_storage,num_nodes,up_width);
            if(layer>0) {
                if(uses_packed_node_linears())
                    interaction.linear_res.reverse_from_packed(
                        state.pre_gate_adj,state.up_adj);
                else interaction.linear_res.reverse(
                    state.pre_gate_adj,state.up_adj);
            } else ordered_kokkos_deep_copy(state.up_adj,Precision(0));
        }
        }
        auto density_adj=state.density_adj;

        ScopedKokkosProfileRegion profile(symmetrix::mh1::region_name(
            symmetrix::mh1::r_reverse_regions,stage));
        const bool stream_layer=streams_layer(layer);
        if(stream_layer) {
            const int convolution_width=
                interaction.convolution_source_contributions.extent(1);
            const int density_width=interaction.density_source_contributions.extent(1);
            const int weight_width=interaction.convolution.weight_size();
            auto convolution_source=interaction.convolution_source_contributions;
            auto convolution_target=interaction.convolution_target_contributions;
            auto density_source=interaction.density_source_contributions;
            auto density_target=interaction.density_target_contributions;
            auto local_message_adj=state.message_adj;
            auto local_up_adj=state.up_adj;
            auto up=state.up;
            const bool execution_layer=
                mace_uses_prepared_execution(streamed_edges);
            const bool generated_execution_layer=
                mace_uses_direct_execution(streamed_edges);
            bool generated_graphwide=false;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(std::is_same_v<Precision,float>
                &&device_execution_space<Kokkos::DefaultExecutionSpace>)
                generated_graphwide=
                    !spline_layer&&generated_execution_layer&&edges>0&&(jit_mh1_cuda_plugin_ready()
                        ||jit_mh1_cuda_plugin_v4_ready());
            else
#endif
            if constexpr(std::is_same_v<Precision,float>)
                generated_graphwide=
                    !spline_layer&&generated_execution_layer&&edges>0&&(jit_mh1_host_plugin_ready()
                        ||jit_mh1_host_plugin_v4_ready());
            const int embedding_width=execution_layer
                ?interaction.convolution_weights.prefix_output_size(num_bessel):0;
            const bool identity_prefix=execution_layer
                &&interaction.convolution_weights
                    .factorized_prefix_is_identity(num_bessel);
            const bool generated_conditioning=
                generated_graphwide&&!identity_prefix;
            if(generated_graphwide) {
                if(!execution_mh1_retain_graph_embedding.at(layer)) {
                    const int slot=
                        execution_mh1_graph_embedding_scratch_slot.at(layer);
                    ensure_view(state.execution_graph_embedding,
                        execution_mh1_graph_embedding_scratch_storage.at(slot),
                        edges,embedding_width);
                    if(generated_conditioning)
                        launch_generated_conditioning_forward(
                            layer,interaction,state);
                    else
                        recompute_identity_graph_embedding(interaction,state);
                    ++execution_mh1_conditioning_recomputations;
                }
                const int scratch_slot=
                    execution_mh1_graph_embedding_scratch_slot.at(layer);
                if(scratch_slot>=0)
                    ensure_view(state.execution_graph_embedding_adj,
                        execution_mh1_graph_embedding_adjoint_scratch_storage.at(
                            scratch_slot),edges,embedding_width);
                else
                    ensure_view(state.execution_graph_embedding_adj,
                        state.execution_graph_embedding_adj_storage,
                        edges,embedding_width);
                ensure_view(state.execution_graph_harmonic_adj,
                    execution_mh1_graph_harmonic_adjoint_scratch_storage,
                    edges,num_lm);
                ensure_view(state.execution_graph_cutoff_adj,
                    execution_mh1_graph_cutoff_adjoint_scratch_storage,edges);
                if(generated_node_program)
                    state.execution_output_ir_mul=state.message_adj;
                else ensure_view(state.execution_output_ir_mul,
                    state.execution_output_ir_mul_storage,num_nodes,message_width);
                state.execution_input_adjoint_ir_mul=
                    decltype(state.execution_input_adjoint_ir_mul)(
                        execution_mh1_input_adjoint_scratch_storage.data(),
                        num_nodes,up_width);
                if(!generated_node_program) {
                    const auto output_mul_to_ir=
                        interaction.convolution.execution_output_mul_to_ir();
                    auto output_ir_mul=state.execution_output_ir_mul;
                    Kokkos::parallel_for(
                        "Execution MH-1 output adjoint mul-ir to ir-mul",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,message_width}),
                        KOKKOS_LAMBDA(int node,int mul_ir) {
                            output_ir_mul(node,output_mul_to_ir(mul_ir))=
                                local_message_adj(node,mul_ir);
                        });
                }
                ordered_kokkos_deep_copy(
                    state.execution_input_adjoint_ir_mul,Precision(0));
                Kokkos::View<const Precision**,Kokkos::LayoutRight>
                    graph_fixed_contribution;
                if(identity_prefix)
                    graph_fixed_contribution=
                        state.execution_graph_fixed_contribution;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr(std::is_same_v<Precision,float>
                    &&device_execution_space<
                        Kokkos::DefaultExecutionSpace>) {
                    std::uint32_t flags=0;
                    if(!apply_cutoff)
                        flags|=
                            SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3;
                    const auto linear_bias=interaction.convolution_weights
                        .final_linear_bias();
                    if(linear_bias.extent(0)!=0)
                        flags|=
                            SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3;
                    if(graph_fixed_contribution.size()!=0)
                        flags|=
                            SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3;
                    const auto linear_weight=interaction.convolution_weights
                        .final_linear_weight_phi_major(num_bessel);
                    const auto output_mask=interaction.convolution
                        .execution_output_mask_ir_mul();
                    const SymmetrixJitMH1CudaReverseArgsV3 args{
                        sizeof(SymmetrixJitMH1CudaReverseArgsV3),
                        static_cast<std::uint32_t>(layer),flags,0u,
                        num_nodes,edges,0,edges,execution_graph_source_owner_count,
                        reinterpret_cast<const std::int32_t*>(
                            neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_targets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_source_edge_offsets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_source_edges.data()),
                        state.execution_graph_embedding.data(),linear_weight.data(),
                        linear_bias.data(),graph_fixed_contribution.data(),
                        edge_harmonics.data(),cutoffs.data(),
                        state.execution_input_ir_mul.data(),output_mask.data(),
                        state.execution_output_ir_mul.data(),
                        state.execution_input_adjoint_ir_mul.data(),
                        state.execution_graph_embedding_adj.data(),
                        state.execution_graph_harmonic_adj.data(),
                        state.execution_graph_cutoff_adj.data()};
                    Kokkos::DefaultExecutionSpace execution_space;
                    ExecutionMH1CudaDeviceGuard device_guard(
                        execution_mh1_device_ordinal(execution_space));
                    const int persistent_blocks=execution_mh1_persistent_blocks();
                    check_execution_mh1_cuda_status(
                        jit_mh1_cuda_plugin_v4_ready()
                            ?jit_mh1_cuda_plugin_v4->launch_reverse(
                                &args,execution_mh1_device_stream(
                                    execution_space),
                                persistent_blocks,persistent_blocks)
                            :jit_mh1_cuda_plugin->descriptor()
                                .launches[layer].reverse_launch(
                                    &args,execution_mh1_device_stream(
                                        execution_space),
                                    persistent_blocks,persistent_blocks),
                        "Launching graph-wide Execution MH-1 CUDA reverse plugin");
                    ++execution_mh1_generated_source_reverse_launches;
                    execution_mh1_generated_edge_reverse_launches+=
                        jit_mh1_cuda_plugin_v4_ready()
                            ?jit_mh1_cuda_plugin_v4->descriptor()
                                .interactions[layer]
                                .edge_reverse_physical_launch_count
                            :jit_mh1_cuda_plugin->descriptor()
                                .interactions[layer]
                                .edge_reverse_physical_launch_count;
                }
                else
#endif
                if constexpr(std::is_same_v<Precision,float>) {
                    std::uint32_t flags=0;
                    if(!apply_cutoff)
                        flags|=SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3;
                    const auto linear_bias=interaction.convolution_weights
                        .final_linear_bias();
                    if(linear_bias.extent(0)!=0)
                        flags|=SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3;
                    if(graph_fixed_contribution.size()!=0)
                        flags|=
                            SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3;
                    const auto linear_weight=interaction.convolution_weights
                        .final_linear_weight(num_bessel);
                    const auto output_mask=interaction.convolution
                        .execution_output_mask_ir_mul();
                    const SymmetrixJitMH1HostReverseArgsV3 args{
                        sizeof(SymmetrixJitMH1HostReverseArgsV3),
                        static_cast<std::uint32_t>(layer),flags,0u,
                        num_nodes,edges,0,edges,execution_graph_source_owner_count,
                        reinterpret_cast<const std::int32_t*>(
                            neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_targets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_source_edge_offsets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_source_edges.data()),
                        state.execution_graph_embedding.data(),linear_weight.data(),
                        linear_bias.data(),graph_fixed_contribution.data(),
                        edge_harmonics.data(),cutoffs.data(),output_mask.data(),
                        state.execution_input_ir_mul.data(),
                        state.execution_output_ir_mul.data(),
                        state.execution_input_adjoint_ir_mul.data(),
                        state.execution_graph_embedding_adj.data(),
                        state.execution_graph_harmonic_adj.data(),
                        state.execution_graph_cutoff_adj.data()};
                    const auto source_owner=jit_mh1_host_plugin_v4_ready()
                        ?jit_mh1_host_plugin_v4->descriptor()
                            .source_reverse_owner
                        :jit_mh1_host_plugin->descriptor()
                            .source_reverse_owner;
                    const auto edge_owner=jit_mh1_host_plugin_v4_ready()
                        ?jit_mh1_host_plugin_v4->descriptor()
                            .edge_reverse_owner
                        :jit_mh1_host_plugin->descriptor()
                            .edge_reverse_owner;
                    Kokkos::parallel_for(
                        "ExecutionMH1HostPlugin::graph_source_reverse",
                        execution_graph_source_owner_count,
                        [=](int owner) { source_owner(&args,owner); });
                    const bool source_owns_edge_reverse=
                        jit_mh1_host_plugin_v4_ready()
                        &&(jit_mh1_host_plugin_v4->descriptor().capabilities
                            &SYMMETRIX_JIT_MH1_HOST_SOURCE_OWNS_EDGE_REVERSE_V4)
                            !=0u;
                    if(!source_owns_edge_reverse) Kokkos::parallel_for(
                        "ExecutionMH1HostPlugin::graph_edge_reverse",edges,
                        [=](int edge) { edge_owner(&args,edge); });
                    ++execution_mh1_generated_source_reverse_launches;
                    ++execution_mh1_generated_edge_reverse_launches;
                }
                auto input_adjoint_ir_mul=state.execution_input_adjoint_ir_mul;
                if(generated_node_program)
                    Kokkos::parallel_for(
                        "Execution MH-1 add input ir-mul adjoint",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,up_width}),
                        KOKKOS_LAMBDA(int node,int ir_mul) {
                            local_up_adj(node,ir_mul)+=
                                input_adjoint_ir_mul(node,ir_mul);
                        });
                else {
                    const auto input_mul_to_ir=
                        interaction.convolution.execution_input_1_mul_to_ir();
                    Kokkos::parallel_for(
                        "Execution MH-1 input adjoint ir-mul to mul-ir",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,up_width}),
                        KOKKOS_LAMBDA(int node,int mul_ir) {
                            local_up_adj(node,mul_ir)+=input_adjoint_ir_mul(
                                node,input_mul_to_ir(mul_ir));
                        });
                }
            }
            if(generated_conditioning) {
                const auto convolution_parameters=interaction.convolution_weights
                    .packed_runtime_parameters();
                const auto density_parameters=interaction.density
                    .packed_runtime_parameters();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr(std::is_same_v<Precision,float>
                    &&device_execution_space<
                        Kokkos::DefaultExecutionSpace>) {
                    std::uint32_t flags=0;
                    if(!apply_cutoff)
                        flags|=
                            SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3;
                    const SymmetrixJitMH1CudaConditioningReverseArgsV3 args{
                        sizeof(
                            SymmetrixJitMH1CudaConditioningReverseArgsV3),
                        static_cast<std::uint32_t>(layer),flags,0u,
                        num_nodes,edges,
                        reinterpret_cast<const std::int32_t*>(
                            neigh_types.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_targets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            node_types.data()),
                        radial.data(),cutoffs.data(),convolution_source.data(),
                        convolution_target.data(),density_source.data(),
                        density_target.data(),convolution_parameters.data(),
                        density_parameters.data(),
                        state.execution_graph_embedding_adj.data(),density_adj.data(),
                        radial_adjoint_values.data(),
                        state.execution_graph_cutoff_adj.data()};
                    Kokkos::DefaultExecutionSpace execution_space;
                    ExecutionMH1CudaDeviceGuard device_guard(
                        execution_mh1_device_ordinal(execution_space));
                    const int persistent_blocks=execution_mh1_persistent_blocks();
                    check_execution_mh1_cuda_status(
                        jit_mh1_cuda_plugin_v4_ready()
                            ?jit_mh1_cuda_plugin_v4->launch_conditioning_reverse(
                                &args,execution_mh1_device_stream(
                                    execution_space),
                                persistent_blocks)
                            :jit_mh1_cuda_plugin->descriptor()
                                .conditioning_launches[layer].reverse_launch(
                                    &args,execution_mh1_device_stream(
                                        execution_space),
                                    persistent_blocks),
                        "Launching graph-wide Execution MH-1 CUDA conditioning "
                        "reverse plugin");
                    ++execution_mh1_generated_conditioning_reverse_launches;
                }
                else
#endif
                if constexpr(std::is_same_v<Precision,float>) {
                    std::uint32_t flags=0;
                    if(!apply_cutoff)
                        flags|=SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3;
                    const SymmetrixJitMH1HostConditioningReverseArgsV3 args{
                        sizeof(
                            SymmetrixJitMH1HostConditioningReverseArgsV3),
                        static_cast<std::uint32_t>(layer),flags,0u,
                        num_nodes,edges,
                        reinterpret_cast<const std::int32_t*>(
                            neigh_types.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_targets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            node_types.data()),
                        radial.data(),cutoffs.data(),convolution_source.data(),
                        convolution_target.data(),density_source.data(),
                        density_target.data(),convolution_parameters.data(),
                        density_parameters.data(),
                        state.execution_graph_embedding_adj.data(),density_adj.data(),
                        radial_adjoint_values.data(),
                        state.execution_graph_cutoff_adj.data()};
                    const auto owner=jit_mh1_host_plugin_v4_ready()
                        ?jit_mh1_host_plugin_v4->descriptor()
                            .conditioning[layer].reverse_owner
                        :jit_mh1_host_plugin->descriptor()
                            .conditioning[layer].reverse_owner;
                    Kokkos::parallel_for(
                        "ExecutionMH1HostPlugin::conditioning_reverse",edges,
                        [=](int edge) { owner(&args,edge); });
                    ++execution_mh1_generated_conditioning_reverse_launches;
                }
                auto graph_harmonic_adj=state.execution_graph_harmonic_adj;
                auto graph_cutoff_adj=state.execution_graph_cutoff_adj;
                const int harmonic_width=num_lm;
                Kokkos::parallel_for(
                    "sum graph-wide Execution MH-1 edge adjoints",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                        {0,0},{edges,harmonic_width+1}),
                    KOKKOS_LAMBDA(int edge,int column) {
                        if(column<harmonic_width)
                            harmonic_adjoint_values(edge,column)+=
                                graph_harmonic_adj(edge,column);
                        else
                            cutoff_adjoint_values(edge)+=graph_cutoff_adj(edge);
                    });
                factorized_source_owned_reverse_used=true;
            }
            const bool generated_spline_reverse=
                generated_spline_r_program&&spline_layer;
            const bool generated_device_spline_reverse=
                generated_device_spline_r_program&&spline_layer&&edges>0;
            bool direct_device_spline_reverse=false;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
                direct_device_spline_reverse=spline_layer
                    &&mace_uses_direct_execution(streamed_edges)&&edges>0
                    &&!generated_device_spline_r_program;
#endif
            if(generated_spline_reverse) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
                    throw std::logic_error(
                        "MH-1 spline host R reverse selected for a device backend.");
                } else
#endif
                {
                    if(!packed_spline_messages) {
                        ensure_view(state.execution_output_ir_mul,
                            state.execution_output_ir_mul_storage,
                            num_nodes,message_width);
                        const auto output_mul_to_ir=
                            interaction.convolution.execution_output_mul_to_ir();
                        auto output_ir_mul=state.execution_output_ir_mul;
                        Kokkos::parallel_for(
                            "Execution MH-1 spline output adjoint mul-ir to ir-mul",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{num_nodes,message_width}),
                            KOKKOS_LAMBDA(int node,int mul_ir) {
                                output_ir_mul(node,output_mul_to_ir(mul_ir))=
                                    local_message_adj(node,mul_ir);
                            });
                    }
                    state.execution_input_adjoint_ir_mul=
                        decltype(state.execution_input_adjoint_ir_mul)(
                            execution_mh1_input_adjoint_scratch_storage.data(),
                            num_nodes,up_width);
                    ordered_kokkos_deep_copy(
                        state.execution_input_adjoint_ir_mul,Precision(0));
                    const auto pair_spline=interaction.pair_spline;
                    const auto output_mask=
                        interaction.convolution.execution_output_mask_ir_mul();
                    const SymmetrixJitMH1HostSplineRReverseArgsV5 args{
                        sizeof(SymmetrixJitMH1HostSplineRReverseArgsV5),
                        static_cast<std::uint32_t>(layer),0u,0u,
                        num_nodes,edges,execution_graph_source_owner_count,
                        static_cast<std::int32_t>(spline_type_count),
                        pair_spline.edge_type_count(),
                        pair_spline.interval_count(),pair_spline.function_count(),0,
                        pair_spline.spline_h(),pair_spline.spline_x0(),
                        reinterpret_cast<const std::int32_t*>(
                            neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_targets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_source_edge_offsets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_source_edges.data()),
                        reinterpret_cast<const std::int32_t*>(spline_neigh_types.data()),
                        reinterpret_cast<const std::int32_t*>(spline_node_types.data()),
                        distances.data(),pair_spline.coefficient_data(),
                        edge_harmonics.data(),output_mask.data(),
                        state.execution_input_ir_mul.data(),
                        state.execution_output_ir_mul.data(),density_adj.data(),
                        state.execution_input_adjoint_ir_mul.data(),
                        harmonic_adjoint_values.data(),
                        pair_spline_distance_adjoints.data()};
                    const auto owner=jit_mh1_host_plugin_v5->descriptor()
                        .spline_r_programs[layer].source_reverse_owner;
                    Kokkos::parallel_for(
                        "ExecutionMH1HostPluginV5::spline_r_source_reverse",
                        execution_graph_source_owner_count,
                        [=](int source_owner) { owner(&args,source_owner); });
                    const auto input_mul_to_ir=
                        interaction.convolution.execution_input_1_mul_to_ir();
                    auto input_adjoint_ir_mul=
                        state.execution_input_adjoint_ir_mul;
                    Kokkos::parallel_for(
                        "Execution MH-1 spline input adjoint ir-mul to mul-ir",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,up_width}),
                        KOKKOS_LAMBDA(int node,int mul_ir) {
                            local_up_adj(node,mul_ir)+=input_adjoint_ir_mul(
                                node,input_mul_to_ir(mul_ir));
                        });
                    ++execution_mh1_generated_source_reverse_launches;
                    ++execution_mh1_generated_edge_reverse_launches;
                    factorized_source_owned_reverse_used=true;
                }
            }
            if(generated_device_spline_reverse) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
                    const auto pair_spline=interaction.pair_spline;
                    const auto output_mask=
                        interaction.convolution.execution_output_mask_ir_mul();
                    const SymmetrixJitMH1CudaSplineV5 radial_packet{
                        sizeof(SymmetrixJitMH1CudaSplineV5),
                        static_cast<std::uint32_t>(
                            pair_spline.edge_type_count()),
                        static_cast<std::uint32_t>(
                            pair_spline.interval_count()),
                        static_cast<std::uint32_t>(
                            pair_spline.function_count()),
                        pair_spline.spline_h(),pair_spline.spline_x0(),
                        pair_spline.coefficient_data()};
                    const SymmetrixJitMH1CudaSplineRSourceArgsV5 source_args{
                        sizeof(SymmetrixJitMH1CudaSplineRSourceArgsV5),
                        static_cast<std::uint32_t>(layer),
                        static_cast<std::uint32_t>(spline_type_count),0u,
                        num_nodes,edges,execution_graph_source_owner_count,
                        reinterpret_cast<const std::int32_t*>(spline_node_types.data()),
                        reinterpret_cast<const std::int32_t*>(
                            neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(spline_neigh_types.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_source_edge_offsets.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_graph_source_edges.data()),
                        reinterpret_cast<const std::int32_t*>(
                            local_targets.data()),
                        distances.data(),radial_packet,local_message_adj.data(),
                        output_mask.data(),density_adj.data(),local_up_adj.data()};
                    const SymmetrixJitMH1CudaSplineREdgeArgsV5 edge_args{
                        sizeof(SymmetrixJitMH1CudaSplineREdgeArgsV5),
                        static_cast<std::uint32_t>(layer),sizeof(double),0u,
                        num_nodes,edges,xyz.data(),distances.data(),radial_packet,
                        edge_harmonics.data(),Y_grad.data(),up.data(),
                        local_message_adj.data(),node_forces.data()};
                    Kokkos::DefaultExecutionSpace execution_space;
                    check_execution_mh1_cuda_status(
                        jit_mh1_cuda_plugin_v4->launch_spline_r_reverse(
                            &source_args,&edge_args,
                            execution_mh1_device_stream(execution_space),
                            execution_mh1_spline_reverse_persistent_blocks()),
                        "Launching generated Execution MH-1 spline R reverse");
                    ++execution_mh1_generated_source_reverse_launches;
                    ++execution_mh1_generated_edge_reverse_launches;
                    factorized_source_owned_reverse_used=true;
                }
#endif
            }
            if(direct_device_spline_reverse) {
                const bool dispatched=interaction.convolution
                    .try_reverse_execution_spline_uvu_from_nodes(
                        state.up,neigh_indices,local_targets,
                        execution_graph_source_edge_offsets,
                        execution_graph_source_edges,spline_neigh_types,
                        spline_node_types,
                        distances,execution_mh1_spline_intervals,
                        execution_mh1_spline_coordinates,
                        spline_type_count,
                        interaction.pair_spline,edge_harmonics,
                        local_message_adj,density_adj,local_up_adj,
                        harmonic_adjoint_values,
                        pair_spline_distance_adjoints);
                if(!dispatched)
                    throw std::logic_error(
                        "Qualified MH-1 direct spline UVU reverse did not dispatch.");
                factorized_source_owned_reverse_used=true;
            }
            if(!generated_conditioning&&!generated_spline_reverse
                &&!generated_device_spline_reverse
                &&!direct_device_spline_reverse)
            for(int first_edge=0;first_edge<edges;first_edge+=streamed_edge_block_size) {
                const int samples=std::min(streamed_edge_block_size,edges-first_edge);
                if(!spline_layer&&(!execution_layer||identity_prefix))
                    ensure_view(state.convolution_contributions,
                        state.convolution_contributions_storage,
                        samples,convolution_width);
                if(!spline_layer&&!execution_layer)
                    ensure_view(state.density_contributions,
                        state.density_contributions_storage,samples,density_width);
                if(!spline_layer&&!execution_layer) {
                    auto convolution_contributions=
                        state.convolution_contributions;
                    auto density_contributions=state.density_contributions;
                    Kokkos::parallel_for(
                        "stream reverse nonlinear edge conditioning",samples,
                        KOKKOS_LAMBDA(int local_edge) {
                            const int edge=first_edge+local_edge;
                            const int source_type=neigh_types(edge);
                            const int target_type=
                                node_types(local_targets(edge));
                            for(int column=0;column<convolution_width;++column)
                                convolution_contributions(local_edge,column)=
                                    convolution_source(source_type,column)
                                    +convolution_target(target_type,column);
                            for(int column=0;column<density_width;++column)
                                density_contributions(local_edge,column)=
                                    density_source(source_type,column)
                                    +density_target(target_type,column);
                        });
                } else if(!spline_layer&&identity_prefix) {
                    auto convolution_contributions=
                        state.convolution_contributions;
                    Kokkos::parallel_for(
                        "Execution MH-1 reverse identity-prefix conditioning",
                        samples,KOKKOS_LAMBDA(int local_edge) {
                            const int edge=first_edge+local_edge;
                            const int source_type=neigh_types(edge);
                            const int target_type=
                                node_types(local_targets(edge));
                            for(int column=0;column<convolution_width;++column)
                                convolution_contributions(local_edge,column)=
                                    convolution_source(source_type,column)
                                    +convolution_target(target_type,column);
                        });
                }
                auto radial_block=Kokkos::subview(
                    radial,std::make_pair(first_edge,first_edge+samples),Kokkos::ALL);
                auto harmonics_block=Kokkos::subview(
                    edge_harmonics,std::make_pair(first_edge,first_edge+samples),Kokkos::ALL);
                if(generated_graphwide)
                    state.edge_harmonic_adj=Kokkos::subview(
                        state.execution_graph_harmonic_adj,
                        std::make_pair(first_edge,first_edge+samples),
                        Kokkos::ALL);
                else
                    ensure_view(state.edge_harmonic_adj,
                        state.edge_harmonic_adj_storage,samples,num_lm);
                ensure_view(state.edge_feature_adj,state.edge_feature_adj_storage,
                    samples,num_bessel);
                if(execution_layer&&!spline_layer) {
                    Kokkos::View<const Precision**,Kokkos::LayoutRight>
                        edge_linear_contribution;
                    if(identity_prefix&&!generated_graphwide)
                        edge_linear_contribution=state.convolution_contributions;
                    if(generated_graphwide) {
                        state.execution_embedding=Kokkos::subview(
                            state.execution_graph_embedding,
                            std::make_pair(first_edge,first_edge+samples),
                            Kokkos::ALL);
                        state.execution_embedding_adj=Kokkos::subview(
                            state.execution_graph_embedding_adj,
                            std::make_pair(first_edge,first_edge+samples),
                            Kokkos::ALL);
                        state.execution_cutoff_adj=Kokkos::subview(
                            state.execution_graph_cutoff_adj,
                            std::make_pair(first_edge,first_edge+samples));
                    } else {
                        ensure_view(
                            state.execution_embedding,state.execution_embedding_storage,
                            samples,embedding_width);
                        ensure_view(
                            state.execution_embedding_adj,
                            state.execution_embedding_adj_storage,
                            samples,embedding_width);
                        ensure_view(
                            state.execution_cutoff_adj,
                            state.execution_cutoff_adj_storage,samples);
                    }
                    if(identity_prefix)
                        interaction.convolution_weights
                            .evaluate_conditioned_prefix(
                                radial_block,state.convolution_contributions,
                                state.execution_embedding);
                    else
                        interaction.convolution_weights
                            .evaluate_conditioned_prefix_indexed(
                                radial_block,neigh_types,local_targets,node_types,
                                first_edge,convolution_source,
                                convolution_target,state.execution_embedding);
                    const int block=first_edge/streamed_edge_block_size;
                    const int source_owner_begin=
                        execution_block_source_segments_host.at(block);
                    const int source_owner_end=
                        execution_block_source_segments_host.at(block+1);
                    const int source_owner_count=
                        source_owner_end-source_owner_begin;
                    auto source_offsets=Kokkos::subview(
                        execution_source_edge_offsets,
                        std::make_pair(source_owner_begin,source_owner_end+1));
                    auto source_edges=execution_source_edges;
                    bool dispatched=generated_graphwide;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                    if constexpr(std::is_same_v<Precision,float>
                        &&device_execution_space<
                            Kokkos::DefaultExecutionSpace>) {
                        if(generated_execution_layer&&!generated_graphwide
                            &&jit_mh1_cuda_plugin_ready()) {
                            std::uint32_t flags=0;
                            if(!apply_cutoff)
                                flags|=
                                    SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3;
                            const auto linear_bias=interaction.convolution_weights
                                .final_linear_bias();
                            if(linear_bias.extent(0)!=0)
                                flags|=
                                    SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3;
                            if(edge_linear_contribution.size()!=0)
                                flags|=
                                    SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3;
                            const auto linear_weight=interaction.convolution_weights
                                .final_linear_weight_phi_major(num_bessel);
                            const auto output_mask=
                                interaction.convolution.execution_output_mask();
                            const SymmetrixJitMH1CudaReverseArgsV3 args{
                                sizeof(SymmetrixJitMH1CudaReverseArgsV3),
                                static_cast<std::uint32_t>(layer),flags,0u,
                                num_nodes,edges,first_edge,samples,
                                source_owner_count,
                                reinterpret_cast<const std::int32_t*>(
                                    neigh_indices.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    local_targets.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    source_offsets.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    source_edges.data()),
                                state.execution_embedding.data(),linear_weight.data(),
                                linear_bias.data(),edge_linear_contribution.data(),
                                harmonics_block.data(),cutoffs.data(),
                                state.up.data(),output_mask.data(),
                                state.message_adj.data(),state.up_adj.data(),
                                state.execution_embedding_adj.data(),
                                state.edge_harmonic_adj.data(),
                                state.execution_cutoff_adj.data()};
                            // Keep the plugin and reverse-prefix work ordered on
                            // Kokkos::Cuda's singleton default stream.
                            Kokkos::DefaultExecutionSpace execution_space;
                            ExecutionMH1CudaDeviceGuard device_guard(
                                execution_mh1_device_ordinal(execution_space));
                            const int persistent_blocks=
                                execution_mh1_persistent_blocks();
                            const auto& launches=jit_mh1_cuda_plugin
                                ->descriptor().launches[layer];
                            check_execution_mh1_cuda_status(
                                launches.reverse_launch(
                                    &args,execution_mh1_device_stream(
                                        execution_space),
                                    persistent_blocks,persistent_blocks),
                                "Launching the Execution MH-1 CUDA reverse plugin");
                            ++execution_mh1_generated_source_reverse_launches;
                            ++execution_mh1_generated_edge_reverse_launches;
                            dispatched=true;
                        }
                    }
                    else
#endif
                    if constexpr(std::is_same_v<Precision,float>) {
                        if(generated_execution_layer&&!generated_graphwide
                            &&jit_mh1_host_plugin_ready()) {
                            std::uint32_t flags=0;
                            if(!apply_cutoff)
                                flags|=SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3;
                            const auto linear_bias=interaction.convolution_weights
                                .final_linear_bias();
                            if(linear_bias.extent(0)!=0)
                                flags|=SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3;
                            if(edge_linear_contribution.size()!=0)
                                flags|=
                                    SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3;
                            const auto linear_weight=interaction.convolution_weights
                                .final_linear_weight(num_bessel);
                            const auto output_mask=
                                interaction.convolution.execution_output_mask();
                            const SymmetrixJitMH1HostReverseArgsV3 args{
                                sizeof(SymmetrixJitMH1HostReverseArgsV3),
                                static_cast<std::uint32_t>(layer),flags,0u,
                                num_nodes,edges,first_edge,samples,
                                source_owner_count,
                                reinterpret_cast<const std::int32_t*>(
                                    neigh_indices.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    local_targets.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    source_offsets.data()),
                                reinterpret_cast<const std::int32_t*>(
                                    source_edges.data()),
                                state.execution_embedding.data(),linear_weight.data(),
                                linear_bias.data(),edge_linear_contribution.data(),
                                harmonics_block.data(),cutoffs.data(),
                                output_mask.data(),state.up.data(),
                                state.message_adj.data(),state.up_adj.data(),
                                state.execution_embedding_adj.data(),
                                state.edge_harmonic_adj.data(),
                                state.execution_cutoff_adj.data()};
                            const auto& descriptor=
                                jit_mh1_host_plugin->descriptor();
                            const auto source_owner=descriptor.source_reverse_owner;
                            const auto edge_owner=descriptor.edge_reverse_owner;
                            Kokkos::parallel_for(
                                "ExecutionMH1HostPlugin::source_reverse",
                                source_owner_count,
                                [=](int owner) { source_owner(&args,owner); });
                            Kokkos::parallel_for(
                                "ExecutionMH1HostPlugin::edge_reverse",samples,
                                [=](int local_edge) {
                                    edge_owner(&args,local_edge);
                                });
                            ++execution_mh1_generated_source_reverse_launches;
                            ++execution_mh1_generated_edge_reverse_launches;
                            dispatched=true;
                        }
                    }
                    if(!dispatched)
                        dispatched=interaction.convolution
                        .try_reverse_execution_uvu_from_nodes(
                            state.up,neigh_indices,local_targets,first_edge,
                            source_offsets,source_edges,
                            state.execution_embedding,
                            interaction.convolution_weights
                                .final_linear_weight(num_bessel),
                            interaction.convolution_weights.final_linear_bias(),
                            edge_linear_contribution,
                            harmonics_block,cutoffs,!apply_cutoff,
                            state.message_adj,state.up_adj,
                            state.execution_embedding_adj,state.edge_harmonic_adj,
                            state.execution_cutoff_adj);
                    if(!dispatched)
                        throw std::logic_error(
                            "Qualified MH-1 Execution UVU reverse did not dispatch.");
                    interaction.convolution_weights.reverse_prefix_from_tape(
                        state.execution_embedding_adj,state.edge_feature_adj);
                    auto execution_cutoff_adj=state.execution_cutoff_adj;
                    Kokkos::parallel_for(
                        "stream sum execution cutoff adjoints",samples,
                        KOKKOS_LAMBDA(int local_edge) {
                            cutoff_adjoint_values(first_edge+local_edge)
                                +=execution_cutoff_adj(local_edge);
                        });
                    factorized_source_owned_reverse_used=true;
                } else {
                    ensure_view(state.weights,state.weights_storage,samples,weight_width);
                    if(spline_layer) {
                        const auto pair_spline=interaction.pair_spline;
                        const int type_count=spline_type_count;
                        auto weights=state.weights;
                        Kokkos::parallel_for(
                            "MH-1 pair spline R1 reverse weights",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{samples,weight_width}),
                            KOKKOS_LAMBDA(int local_edge,int function) {
                                const int edge=first_edge+local_edge;
                                const int pair=spline_neigh_types(edge)*type_count
                                    +spline_node_types(local_targets(edge));
                                weights(local_edge,function)=
                                    pair_spline.evaluate_function(
                                        pair,distances(edge),function);
                            });
                    } else {
                        ensure_view(state.raw_weights,state.raw_weights_storage,
                            samples,weight_width);
                        interaction.convolution_weights.evaluate_conditioned(
                            radial_block,state.convolution_contributions,
                            state.raw_weights);
                        ordered_kokkos_deep_copy(state.weights,state.raw_weights);
                        if(!apply_cutoff) {
                            auto weights=state.weights;
                            Kokkos::parallel_for(
                                "stream reverse nonlinear weight cutoff",weights.size(),
                                KOKKOS_LAMBDA(std::size_t flat) {
                                    const std::size_t local_edge=
                                        flat/weights.extent(1);
                                    weights(local_edge,flat%weights.extent(1))
                                        *=local_cutoffs(first_edge+local_edge);
                                });
                        }
                    }
                    ensure_view(state.weight_adj,state.weight_adj_storage,
                        samples,weight_width);
                    const bool direct_node_reverse=uses_direct_node_tensor_reverse()
                        &&interaction.convolution.supports_direct_node_reverse();
                    if(direct_node_reverse) {
                        const bool dispatched=
                            interaction.convolution.try_reverse_from_nodes(
                                state.up,neigh_indices,first_edge,harmonics_block,
                                state.weights,state.message_adj,local_targets,
                                state.up_adj,state.edge_harmonic_adj,state.weight_adj);
                        if(!dispatched)
                            throw std::logic_error(
                                "Qualified direct-node tensor reverse did not dispatch.");
                    } else {
                        ensure_view(state.edge_up,state.edge_up_storage,samples,up_width);
                        auto edge_up=state.edge_up;
                        Kokkos::parallel_for(
                            "stream reverse nonlinear gather edge up",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{samples,up_width}),
                            KOKKOS_LAMBDA(int local_edge,int k) {
                                edge_up(local_edge,k)=
                                    up(neigh_indices(first_edge+local_edge),k);
                            });
                        // Forward edge messages are dead before the streamed reverse
                        // pass, so the equally shaped adjoint can reuse their capacity.
                        ensure_view(state.edge_message_adj,state.edge_messages_storage,
                            samples,message_width);
                        auto edge_message_adj=state.edge_message_adj;
                        Kokkos::parallel_for(
                            "stream gather edge message adjoint",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{samples,message_width}),
                            KOKKOS_LAMBDA(int local_edge,int k) {
                                edge_message_adj(local_edge,k)=
                                    local_message_adj(
                                        local_targets(first_edge+local_edge),k);
                            });
                        ensure_view(state.edge_up_adj,state.edge_up_adj_storage,
                            samples,up_width);
                        interaction.convolution.reverse(
                            state.edge_up,harmonics_block,state.weights,
                            state.edge_message_adj,state.edge_up_adj,
                            state.edge_harmonic_adj,state.weight_adj);
                        auto edge_up_adj=state.edge_up_adj;
                        Kokkos::parallel_for(
                            "stream scatter edge up adjoint",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{samples,up_width}),
                            KOKKOS_LAMBDA(int local_edge,int k) {
                                Kokkos::atomic_add(
                                    &local_up_adj(
                                        neigh_indices(first_edge+local_edge),k),
                                    edge_up_adj(local_edge,k));
                            });
                    }
                    if(!spline_layer) {
                        ensure_view(state.raw_weight_adj,state.raw_weight_adj_storage,
                            samples,weight_width);
                        auto raw_weight_adj=state.raw_weight_adj;
                        auto weight_adj=state.weight_adj;
                        auto raw_weights=state.raw_weights;
                        if(!apply_cutoff) {
                        if(direct_node_reverse) {
                            using cutoff_policy=Kokkos::TeamPolicy<>;
                            constexpr int warp_size=32;
                            const int cutoff_team_size=std::min(
                                128,std::max(
                                    warp_size,
                                    ((weight_width+warp_size-1)/warp_size)*warp_size));
                            Kokkos::parallel_for(
                                "stream reverse convolution cutoff team",
                                cutoff_policy(samples,cutoff_team_size),
                                KOKKOS_LAMBDA(
                                    const typename cutoff_policy::member_type& team) {
                                    const int local_edge=team.league_rank();
                                    const int edge=first_edge+local_edge;
                                    Precision cutoff_value=Precision(0);
                                    Kokkos::parallel_reduce(
                                        Kokkos::TeamThreadRange(team,weight_width),
                                        [&](const int k,Precision& update) {
                                            update+=weight_adj(local_edge,k)
                                                *raw_weights(local_edge,k);
                                            raw_weight_adj(local_edge,k)=
                                                weight_adj(local_edge,k)
                                                *local_cutoffs(edge);
                                        },cutoff_value);
                                    Kokkos::single(Kokkos::PerTeam(team),[&]() {
                                        cutoff_adjoint_values(edge)+=cutoff_value;
                                    });
                                });
                        } else {
                            Kokkos::parallel_for(
                                "stream reverse convolution cutoff",samples,
                                KOKKOS_LAMBDA(int local_edge) {
                                    const int edge=first_edge+local_edge;
                                    Precision cutoff_value=Precision(0);
                                    for(int k=0;k<weight_width;++k) {
                                        cutoff_value+=weight_adj(local_edge,k)
                                            *raw_weights(local_edge,k);
                                        raw_weight_adj(local_edge,k)=
                                            weight_adj(local_edge,k)
                                            *local_cutoffs(edge);
                                    }
                                    cutoff_adjoint_values(edge)+=cutoff_value;
                                });
                        }
                        } else ordered_kokkos_deep_copy(raw_weight_adj,weight_adj);
                        interaction.convolution_weights.reverse_from_tape(
                            state.raw_weight_adj,state.edge_feature_adj);
                    }
                }
                auto edge_harmonic_adj=state.edge_harmonic_adj;
                Kokkos::parallel_for(
                    "stream sum harmonic adjoints",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                        {0,0},{samples,num_lm}),
                    KOKKOS_LAMBDA(int local_edge,int lm) {
                        harmonic_adjoint_values(first_edge+local_edge,lm)
                            +=edge_harmonic_adj(local_edge,lm);
                    });

                if(spline_layer) {
                    const auto pair_spline=interaction.pair_spline;
                    const int density_function=interaction.pair_spline_weight_count;
                    const int type_count=spline_type_count;
                    auto weight_adj=state.weight_adj;
                    auto distance_adjoint=pair_spline_distance_adjoints;
                    Kokkos::parallel_for(
                        "MH-1 pair spline R1 radial reverse",samples,
                        KOKKOS_LAMBDA(int local_edge) {
                            const int edge=first_edge+local_edge;
                            const int pair=spline_neigh_types(edge)*type_count
                                +spline_node_types(local_targets(edge));
                            const auto point=
                                pair_spline.evaluation_point(distances(edge));
                            Precision radial_value=Precision(0);
                            for(int function=0;function<weight_width;++function) {
                                Precision value,derivative;
                                pair_spline.evaluate_function(
                                    pair,point,function,value,derivative);
                                radial_value+=weight_adj(local_edge,function)
                                    *derivative;
                            }
                            Precision density_value,density_derivative;
                            pair_spline.evaluate_function(
                                pair,point,density_function,
                                density_value,density_derivative);
                            radial_value+=density_adj(local_targets(edge))
                                *density_derivative;
                            distance_adjoint(edge)+=radial_value;
                        });
                } else {
                    ensure_view(state.density_matrix,
                        state.density_matrix_storage,samples,1);
                    if(execution_layer)
                        interaction.density.evaluate_conditioned_indexed(
                            radial_block,neigh_types,local_targets,node_types,
                            first_edge,density_source,density_target,
                            state.density_matrix);
                    else
                        interaction.density.evaluate_conditioned(
                            radial_block,state.density_contributions,
                            state.density_matrix);
                    ensure_view(state.density_raw_adj,
                        state.density_raw_adj_storage,samples,1);
                    auto density_matrix=state.density_matrix;
                    auto density_raw_adj=state.density_raw_adj;
                    Kokkos::parallel_for(
                        "stream reverse density envelope",samples,
                        KOKKOS_LAMBDA(int local_edge) {
                            const int edge=first_edge+local_edge;
                            const Precision raw=density_matrix(local_edge,0);
                            const Precision base=Kokkos::tanh(raw*raw);
                            Precision value=density_adj(local_targets(edge))
                                *(Precision(1)-base*base)*Precision(2)*raw;
                            if(!embed_cutoff) {
                                cutoff_adjoint_values(edge)+=
                                    density_adj(local_targets(edge))*base;
                                value*=local_cutoffs(edge);
                            }
                            density_raw_adj(local_edge,0)=value;
                        });
                    ensure_view(state.density_feature_adj,
                        state.density_feature_adj_storage,samples,num_bessel);
                    interaction.density.reverse_from_tape(
                        state.density_raw_adj,state.density_feature_adj);
                    auto edge_feature_adj=state.edge_feature_adj;
                    auto density_feature_adj=state.density_feature_adj;
                    Kokkos::parallel_for(
                        "stream sum radial adjoints",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{samples,num_bessel}),
                        KOKKOS_LAMBDA(int local_edge,int k) {
                            radial_adjoint_values(first_edge+local_edge,k)
                                +=edge_feature_adj(local_edge,k)
                                +density_feature_adj(local_edge,k);
                        });
                }
            }
            if(mace_uses_prepared_execution(streamed_edges))
                ++factorized_reverse_evaluations;
        } else {
        ensure_view(state.edge_message_adj,state.edge_message_adj_storage,edges,
            message_width);
        auto edge_message_adj=state.edge_message_adj;
        auto local_message_adj=state.message_adj;
        Kokkos::parallel_for("gather edge message adjoint",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,message_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                edge_message_adj(edge,k)=local_message_adj(local_targets(edge),k);
            });
        ensure_view(state.edge_up_adj,state.edge_up_adj_storage,edges,up_width);
        ensure_view(state.edge_harmonic_adj,state.edge_harmonic_adj_storage,edges,num_lm);
        ensure_view(state.weight_adj,state.weight_adj_storage,edges,
            interaction.convolution.weight_size());
        interaction.convolution.reverse(
            state.edge_up,edge_harmonics,state.weights,state.edge_message_adj,
            state.edge_up_adj,state.edge_harmonic_adj,state.weight_adj);
        auto local_up_adj=state.up_adj;
        auto edge_up_adj=state.edge_up_adj;
        auto edge_harmonic_adj=state.edge_harmonic_adj;
        Kokkos::parallel_for("scatter edge up adjoint",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,up_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                Kokkos::atomic_add(&local_up_adj(neigh_indices(edge),k),edge_up_adj(edge,k));
            });
        Kokkos::parallel_for("sum harmonic adjoints",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,num_lm}),
            KOKKOS_LAMBDA(int edge,int lm) {
                harmonic_adjoint_values(edge,lm)+=edge_harmonic_adj(edge,lm);
            });

        ensure_view(state.raw_weight_adj,state.raw_weight_adj_storage,edges,
            interaction.convolution.weight_size());
        auto raw_weight_adj=state.raw_weight_adj;
        auto weight_adj=state.weight_adj;
        auto raw_weights=state.raw_weights;
        if(!apply_cutoff) {
            Kokkos::parallel_for("reverse convolution cutoff",edges,
                KOKKOS_LAMBDA(int edge) {
                    Precision cutoff_value=Precision(0);
                    for(int k=0;k<weight_adj.extent(1);++k) {
                        cutoff_value+=weight_adj(edge,k)*raw_weights(edge,k);
                        raw_weight_adj(edge,k)=weight_adj(edge,k)*local_cutoffs(edge);
                    }
                    cutoff_adjoint_values(edge)+=cutoff_value;
                });
        } else ordered_kokkos_deep_copy(raw_weight_adj,weight_adj);

        const int edge_feature_width=mh1_fast_path?num_bessel:state.edge_features.extent(1);
        ensure_view(state.edge_feature_adj,state.edge_feature_adj_storage,edges,
            edge_feature_width);
        if(mh1_fast_path)
            interaction.convolution_weights.reverse_from_tape(
                state.raw_weight_adj,state.edge_feature_adj);
        else interaction.convolution_weights.reverse(
            state.edge_features,state.raw_weight_adj,state.edge_feature_adj);
        ensure_view(state.density_raw_adj,state.density_raw_adj_storage,edges,1);
        auto density_raw_adj=state.density_raw_adj;
        auto density_raw=state.density_raw;
        auto density_base=state.density_base;
        Kokkos::parallel_for("reverse density envelope",edges,KOKKOS_LAMBDA(int edge) {
            Precision value=density_adj(local_targets(edge))
                *(Precision(1)-density_base(edge)*density_base(edge))*Precision(2)*density_raw(edge);
            if(!embed_cutoff) {
                cutoff_adjoint_values(edge)+=density_adj(local_targets(edge))*density_base(edge);
                value*=local_cutoffs(edge);
            }
            density_raw_adj(edge,0)=value;
        });
        ensure_view(state.density_feature_adj,state.density_feature_adj_storage,
            edges,edge_feature_width);
        if(mh1_fast_path)
            interaction.density.reverse_from_tape(
                state.density_raw_adj,state.density_feature_adj);
        else interaction.density.reverse(
            state.edge_features,state.density_raw_adj,state.density_feature_adj);
        auto edge_feature_adj=state.edge_feature_adj;
        auto density_feature_adj=state.density_feature_adj;
        Kokkos::parallel_for("sum radial adjoints",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,num_bessel}),
            KOKKOS_LAMBDA(int edge,int k) {
                radial_adjoint_values(edge,k)+=edge_feature_adj(edge,k)
                    +density_feature_adj(edge,k);
            });
        }

        if(generated_node_program) {
            if(layer>0) {
                const auto& runtime=execution_mh1_node_runtime.at(layer);
                auto previous=states[layer-1].layer_adjoint;
                if(generated_device_spline_node_adapter) {
                    const auto input_mul_to_ir=
                        interaction.convolution.execution_input_1_mul_to_ir();
                    auto input_mul_ir_adjoint=state.up_adj;
                    auto input_ir_mul_adjoint=
                        state.execution_input_adjoint_ir_mul;
                    Kokkos::parallel_for(
                        "Execution MH-1 spline device input adjoint mul-ir to ir-mul",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,up_width}),
                        KOKKOS_LAMBDA(int node,int mul_ir) {
                            input_ir_mul_adjoint(
                                node,input_mul_to_ir(mul_ir))=
                                input_mul_ir_adjoint(node,mul_ir);
                        });
                }
                const auto node_up=generated_device_spline_node_adapter
                    ?state.execution_input_ir_mul:state.up;
                const auto node_up_adjoint=generated_device_spline_node_adapter
                    ?state.execution_input_adjoint_ir_mul:state.up_adj;
                const SymmetrixJitMH1HostNodeReverseArgsV4 host_args{
                    sizeof(SymmetrixJitMH1HostNodeReverseArgsV4),
                    static_cast<std::uint32_t>(layer),
                    SYMMETRIX_JIT_MH1_HOST_NODE_PRE_REVERSE_V4,0u,
                    static_cast<float>(scale),0u,num_nodes,
                    reinterpret_cast<const std::int32_t*>(
                        product_elements.data()),nullptr,
                    reinterpret_cast<const float*>(
                        runtime.linear_parameters.data()),
                    reinterpret_cast<const float*>(
                        runtime.product_parameters.data()),
                    reinterpret_cast<const float*>(
                        runtime.readout_parameters.data()),
                    reinterpret_cast<const float*>(state.input.data()),
                    reinterpret_cast<const float*>(node_up.data()),nullptr,nullptr,
                    reinterpret_cast<float*>(runtime.arena.data()),nullptr,nullptr,
                    reinterpret_cast<float*>(node_up_adjoint.data()),
                    reinterpret_cast<float*>(previous.data()),nullptr};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr(device_execution_space<
                    Kokkos::DefaultExecutionSpace>) {
                    Kokkos::DefaultExecutionSpace execution_space;
                    const int persistent_blocks=execution_mh1_persistent_blocks();
                    for(int first=0;first<num_nodes;
                        first+=runtime.tile_rows) {
                        const int rows=std::min(
                            runtime.tile_rows,num_nodes-first);
                        auto args=device_node_reverse_tile_args(
                            host_args,first,rows,input_width,up_width,
                            message_width,products[layer].output_dimension(),
                            0,0,nullptr,nullptr,sizeof(Precision));
                        check_execution_mh1_cuda_status(
                            jit_mh1_cuda_plugin_v4->launch_node_reverse(
                                &args,execution_mh1_device_stream(execution_space),
                                persistent_blocks),
                            "Launching Execution MH-1 v4 CUDA node pre-reverse");
                    }
                } else
#endif
                {
                    const auto owner=jit_mh1_host_plugin_v4->descriptor()
                        .node_programs[layer].reverse_phases[1].owner;
                    for(int first=0;first<num_nodes;first+=runtime.tile_rows) {
                        const int rows=std::min(
                            runtime.tile_rows,num_nodes-first);
                        auto tile_args=host_args;
                        tile_args.num_nodes=rows;
                        tile_args.element_indices+=first;
                        tile_args.layer_input+=static_cast<std::size_t>(first)
                            *input_width;
                        tile_args.up+=static_cast<std::size_t>(first)*up_width;
                        tile_args.up_adjoint+=static_cast<std::size_t>(first)
                            *up_width;
                        tile_args.layer_input_adjoint+=
                            static_cast<std::size_t>(first)*input_width;
                        Kokkos::parallel_for(
                            "ExecutionMH1HostPluginV4::node_pre_reverse",rows,
                            [=](int node) { owner(&tile_args,node); });
                    }
                }
            }
            continue;
        }

        if(layer>0) {
            auto previous=states[layer-1].layer_adjoint;
            interaction.linear_up.reverse(state.up_adj,previous,false);
            interaction.skip.reverse(state.skip_adj,previous,false);
        }
    }

    ensure_view(node_forces,node_forces_storage,xyz.size());
    auto forces=node_forces;
    auto flat_grad=Y_grad;
    auto pair_spline_distance_adjoint_values=pair_spline_distance_adjoints;
    if(!generated_device_spline_r_program)
    Kokkos::parallel_for("nonlinear edge forces",edges,KOKKOS_LAMBDA(int edge) {
        const Precision distance=distances(edge);
        const Precision x=distance/rc;
        Precision envelope_derivative=Precision(0);
        if(distance<rc)
            envelope_derivative=(-Precision(0.5)*cp*(cp+Precision(1))*(cp+Precision(2))*Kokkos::pow(x,cp-1)
                +cp*(cp+Precision(1))*(cp+Precision(2))*Kokkos::pow(x,cp)
                -Precision(0.5)*cp*(cp+Precision(1))*(cp+Precision(2))*Kokkos::pow(x,cp+1))/rc;
        Precision transformed=distance,transform_derivative=Precision(1);
        if(agnesi) {
            const int source_z=model_z(indices(neigh_types(edge)));
            const int target_z=model_z(indices(node_types(local_targets(edge))));
            const Precision r0=Precision(0.5)*(radii(source_z)+radii(target_z));
            const Precision ratio=distance/r0;
            const Precision ratio_power=Kokkos::pow(ratio,q-p);
            const Precision denominator=Precision(1)+ratio_power;
            const Precision g=a*Kokkos::pow(ratio,q)/denominator;
            const Precision dgdx=a*(q*Kokkos::pow(ratio,q-Precision(1))*denominator
                -(q-p)*Kokkos::pow(ratio,Precision(2)*q-p-Precision(1)))
                /(denominator*denominator);
            transformed=Precision(1)/(Precision(1)+g);
            transform_derivative=-dgdx/(r0*(Precision(1)+g)*(Precision(1)+g));
        }
        Precision distance_adjoint=pair_spline_execution
            ?pair_spline_distance_adjoint_values(edge)
            :cutoff_adjoint_values(edge)*envelope_derivative;
        if(!pair_spline_execution)for(int k=0;k<nb;++k) {
            const Precision weight=bw(k);
            const Precision base=prefactor*Kokkos::sin(weight*transformed)/transformed;
            const Precision base_derivative=prefactor
                *(weight*Kokkos::cos(weight*transformed)*transformed
                    -Kokkos::sin(weight*transformed))
                /(transformed*transformed)*transform_derivative;
            const Precision derivative=embed_cutoff
                ?base_derivative*local_cutoffs(edge)+base*envelope_derivative
                :base_derivative;
            distance_adjoint+=radial_adjoint_values(edge,k)*derivative;
        }
        for(int component=0;component<3;++component) {
            const std::size_t coordinate=
                static_cast<std::size_t>(3)*edge+component;
            const std::size_t gradient=coordinate*nlm;
            Precision vector_adjoint=distance_adjoint*xyz(coordinate)/distance;
            for(int lm=0;lm<nlm;++lm)
                vector_adjoint+=harmonic_adjoint_values(edge,lm)
                    *flat_grad(gradient+lm);
            forces(coordinate)=-vector_adjoint;
        }
    });

    if(has_zbl) {
        ensure_view(zbl_energies,zbl_energies_storage,num_nodes);
        ensure_view(zbl_forces,zbl_forces_storage,xyz.size());
        ordered_kokkos_deep_copy(zbl_energies,Precision(0));
        ordered_kokkos_deep_copy(zbl_forces,Precision(0));
        if(mace_uses_prepared_execution(streamed_edges)) {
            ++factorized_zbl_evaluator_stream_launches;
            zbl.compute_ZBL(
                factorized_execution_space,num_nodes,node_types,num_neigh,
                neigh_types,atomic_numbers,local_offsets,distances,xyz,
                zbl_energies,zbl_forces);
        } else
            zbl.compute_ZBL(
                num_nodes,node_types,num_neigh,neigh_types,atomic_numbers,
                distances,xyz,zbl_energies,zbl_forces);
        auto zbl_e=zbl_energies;
        auto zbl_f=zbl_forces;
        Kokkos::parallel_for("add zbl energy",num_nodes,KOKKOS_LAMBDA(int node) {
            energies(node)+=energy_scale*zbl_e(node);
        });
        Kokkos::parallel_for("add zbl forces",xyz.size(),KOKKOS_LAMBDA(int index) {
            forces(index)+=energy_scale*zbl_f(index);
        });
    }
    if(mace_uses_prepared_execution(streamed_edges)) {
        factorized_execution_space.fence(
            "MH-1 Execution energy and force evaluation");
        ++factorized_evaluation_fences;
        factorized_completed_evaluation_graph_generation=
            execution_graph_generation;
    }
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::reduce_node_forces(
    const int num_nodes,Kokkos::View<const int*> edge_receivers,
    Kokkos::View<const int*> edge_sources)
{
    if(num_nodes<0||edge_receivers.extent(0)!=edge_sources.extent(0))
        throw std::invalid_argument(
            "MH-1 atom-force reduction extents are inconsistent.");
    const std::size_t num_edges=edge_sources.extent(0);
    if(node_forces.extent(0)<3*num_edges)
        throw std::invalid_argument(
            "MH-1 atom-force reduction exceeds the evaluated edge extent.");
    ensure_view(atom_forces,atom_forces_storage,
        3*static_cast<std::size_t>(num_nodes));
    Kokkos::deep_copy(factorized_execution_space,atom_forces,0.0);
    const auto reduced_forces=atom_forces;
    const auto directed_forces=node_forces;
    Kokkos::parallel_for(
        "MaceNonlinearKokkos::reduce_node_forces",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space,0,num_edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            const int receiver=edge_receivers(edge);
            const int source=edge_sources(edge);
            for(int component=0;component<3;++component) {
                const double force=directed_forces(3*edge+component);
                Kokkos::atomic_add(
                    &reduced_forces(3*source+component),force);
                Kokkos::atomic_add(
                    &reduced_forces(3*receiver+component),-force);
            }
        });
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::reduce_prepared_node_forces(
    const std::uint64_t graph_generation)
{
    if(graph_generation==0
        ||graph_generation!=factorized_prepared_graph_generation
        ||graph_generation!=factorized_completed_evaluation_graph_generation)
        throw std::invalid_argument(
            "MH-1 atom-force reduction requires the current completed graph.");
    reduce_node_forces(
        static_cast<int>(execution_prepared_node_types.extent(0)),
        execution_prepared_targets,execution_prepared_neigh_indices);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::reduce_stress(
    const double volume,Kokkos::View<const double*> xyz)
{
    if(!std::isfinite(volume)||!(volume>0.0)||xyz.extent(0)%3!=0)
        throw std::invalid_argument(
            "MH-1 stress reduction inputs are inconsistent.");
    const std::size_t num_edges=xyz.extent(0)/3;
    if(node_forces.extent(0)<3*num_edges)
        throw std::invalid_argument(
            "MH-1 stress reduction exceeds the evaluated edge extent.");
    ensure_view(stress_tensor,stress_tensor_storage,9);
    const auto reduced_stress=stress_tensor;
    const auto directed_forces=node_forces;
    const double scale=-1.0/volume;
    for(int component=0;component<9;++component) {
        const int force_component=component/3;
        const int vector_component=component%3;
        Kokkos::parallel_reduce(
            "MaceNonlinearKokkos::reduce_stress_component",
            Kokkos::RangePolicy<decltype(factorized_execution_space),
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space,0,num_edges),
            KOKKOS_LAMBDA(const std::size_t edge,double& value) {
                value+=scale*directed_forces(3*edge+force_component)
                    *xyz(3*edge+vector_component);
            },
            Kokkos::subview(reduced_stress,component));
    }
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::reduce_prepared_stress(
    const double volume,const std::uint64_t graph_generation)
{
    if(graph_generation==0
        ||graph_generation!=factorized_prepared_graph_generation
        ||graph_generation!=factorized_completed_evaluation_graph_generation)
        throw std::invalid_argument(
            "MH-1 stress reduction requires the current completed graph.");
    const std::size_t num_edges=execution_prepared_neigh_indices.extent(0);
    reduce_stress(volume,Kokkos::subview(
        execution_prepared_xyz,
        std::make_pair(std::size_t(0),3*num_edges)));
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::prepare_factorized_batch(
    const std::uint64_t graph_generation,
    const std::span<const std::int64_t> edge_offsets)
{
    if(graph_generation==0
        ||graph_generation!=factorized_prepared_graph_generation)
        throw std::invalid_argument(
            "MH-1 batched stress requires the current prepared graph.");
    const std::size_t num_edges=execution_prepared_neigh_indices.extent(0);
    if(edge_offsets.size()<2||edge_offsets.front()!=0
        ||edge_offsets.back()!=static_cast<std::int64_t>(num_edges))
        throw std::invalid_argument(
            "MH-1 batched stress edge offsets do not cover the prepared graph.");
    for(std::size_t graph=0;graph+1<edge_offsets.size();++graph)
        if(edge_offsets[graph]<0||edge_offsets[graph]>edge_offsets[graph+1])
            throw std::invalid_argument(
                "MH-1 batched stress edge offsets must be monotonic.");

    if(graph_generation==factorized_prepared_batch_graph_generation
        &&std::equal(edge_offsets.begin(),edge_offsets.end(),
            execution_prepared_batch_edge_offsets_host.begin(),
            execution_prepared_batch_edge_offsets_host.end()))
        return;
    execution_prepared_batch_edge_offsets_host.assign(
        edge_offsets.begin(),edge_offsets.end());
    const auto host=Kokkos::View<
        const std::int64_t*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            edge_offsets.data(),edge_offsets.size());
    execution_prepared_batch_edge_offsets=Kokkos::View<std::int64_t*>(
        Kokkos::view_alloc(
            "MH-1 prepared batch edge offsets",Kokkos::WithoutInitializing),
        edge_offsets.size());
    Kokkos::deep_copy(factorized_execution_space,
        execution_prepared_batch_edge_offsets,host);
    factorized_prepared_batch_graph_generation=graph_generation;
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::reduce_prepared_batched_stress(
    const std::span<const double> volumes,
    const std::uint64_t graph_generation)
{
    if(graph_generation==0
        ||graph_generation!=factorized_prepared_graph_generation
        ||graph_generation!=factorized_prepared_batch_graph_generation
        ||graph_generation!=factorized_completed_evaluation_graph_generation)
        throw std::invalid_argument(
            "MH-1 batched stress requires the current completed batch graph.");
    const std::size_t graph_count=
        execution_prepared_batch_edge_offsets.extent(0)-1;
    if(volumes.size()!=graph_count
        ||!std::all_of(volumes.begin(),volumes.end(),[](const double volume) {
            return std::isfinite(volume)&&volume>0.0;
        }))
        throw std::invalid_argument(
            "MH-1 batched stress volumes are inconsistent.");

    if(execution_prepared_batch_volumes.extent(0)!=graph_count)
        execution_prepared_batch_volumes=Kokkos::View<double*>(
            Kokkos::view_alloc(
                "MH-1 prepared batch volumes",Kokkos::WithoutInitializing),
            graph_count);
    const auto volumes_host=Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            volumes.data(),volumes.size());
    Kokkos::deep_copy(factorized_execution_space,
        execution_prepared_batch_volumes,volumes_host);
    ensure_view(stress_tensor,stress_tensor_storage,9*graph_count);

    const auto reduced_stress=stress_tensor;
    const auto directed_forces=node_forces;
    const auto xyz=execution_prepared_xyz;
    const auto offsets=execution_prepared_batch_edge_offsets;
    const auto device_volumes=execution_prepared_batch_volumes;
    using team_policy=Kokkos::TeamPolicy<decltype(factorized_execution_space)>;
    Kokkos::parallel_for(
        "MaceNonlinearKokkos::reduce_batched_stress",
        team_policy(factorized_execution_space,9*graph_count,Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename team_policy::member_type& team) {
            const std::size_t owner=team.league_rank();
            const std::size_t graph=owner/9;
            const int component=owner%9;
            const int force_component=component/3;
            const int vector_component=component%3;
            const std::int64_t edge_begin=offsets(graph);
            const std::int64_t edge_end=offsets(graph+1);
            double value=0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team,edge_begin,edge_end),
                [&](const std::int64_t edge,double& update) {
                    const std::size_t index=static_cast<std::size_t>(edge);
                    update+=directed_forces(3*index+force_component)
                        *xyz(3*index+vector_component);
                },value);
            Kokkos::single(Kokkos::PerTeam(team),[&]() {
                reduced_stress(owner)=-value/device_volumes(graph);
            });
        });
}

#else
#error "SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART must select one implementation partition"
#endif

template class MaceNonlinearKokkosT<float>;
template class MaceNonlinearKokkosT<double>;
