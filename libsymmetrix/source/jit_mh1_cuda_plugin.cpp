#include "jit_mh1_cuda_plugin.hpp"

#if __has_include(<Kokkos_Macros.hpp>)
#include <Kokkos_Macros.hpp>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include <cstddef>
#include <fstream>
#include <limits>
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#include <nlohmann/json.hpp>
#endif
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(std::is_standard_layout_v<SymmetrixJitMH1CudaInteractionV3>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1CudaForwardArgsV3>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1CudaReverseArgsV3>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaConditioningForwardArgsV3>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaConditioningReverseArgsV3>);
static_assert(
    std::is_standard_layout_v<SymmetrixJitMH1CudaInteractionLaunchesV3>);
static_assert(
    std::is_standard_layout_v<SymmetrixJitMH1CudaConditioningLaunchesV3>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1CudaPluginV3>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaNodeForwardArgsV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaNodeReverseArgsV4>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1CudaSplineV5>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaSplineRForwardArgsV5>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaSplineRSourceArgsV5>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaSplineREdgeArgsV5>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaInteractionLaunchesV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaConditioningLaunchesV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaNodeForwardPhaseV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1CudaNodeReversePhaseV4>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1CudaNodeProgramV4>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1CudaPluginV4>);

#if defined(__unix__) || defined(__APPLE__)
namespace symmetrix::execution {

struct MH1CudaModuleLaunch {
    void* function = nullptr;
    std::int32_t grid_kind = 0;
    std::int32_t channels = 0;
    std::int32_t components = 0;
    std::int32_t tile_nodes = 0;
    std::int32_t tile_channels = 0;
    std::int32_t threads = 0;
};

struct MH1CudaModuleInteraction {
    void* forward = nullptr;
    void* source = nullptr;
    std::array<void*,2> edge{};
    std::int32_t edge_count = 0;
    void* conditioning_forward = nullptr;
    std::array<void*,2> conditioning_reverse{};
    std::int32_t forward_partitions = 0;
    std::int32_t source_partitions = 0;
    std::int32_t forward_threads = 0;
    std::int32_t source_threads = 0;
    std::int32_t edge_threads = 0;
    std::int32_t edge_phi_edge_batch = 1;
    std::int32_t edge_phi_block_multiplier = 1;
    void* spline_forward = nullptr;
    void* spline_reverse = nullptr;
    std::int32_t spline_forward_threads = 0;
    std::int32_t spline_reverse_threads = 0;
};

struct MH1CudaModuleNode {
    std::vector<MH1CudaModuleLaunch> pre_forward;
    std::vector<MH1CudaModuleLaunch> post_forward;
    std::vector<MH1CudaModuleLaunch> post_reverse;
    std::vector<MH1CudaModuleLaunch> pre_reverse;
};

struct MH1CudaModuleState {
    void* module = nullptr;
    void* context = nullptr;
    int device = -1;
    std::array<MH1CudaModuleInteraction,2> interactions;
    std::array<MH1CudaModuleNode,2> nodes;
    std::string abi_tag;
    std::string artifact_id;
    std::string generation_fingerprint;
    std::string semantic_fingerprint;
    std::string structure_fingerprint;
    std::string runtime_layout_fingerprint;
    SymmetrixJitMH1CudaPluginV4 descriptor{};
};

namespace {

std::runtime_error plugin_error(
    const std::string& path,
    const std::string& message)
{
    return std::runtime_error(
        "Could not load Execution MH-1 CUDA plugin '"+path+"': "+message);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#if defined(KOKKOS_ENABLE_CUDA)
using MH1Module = CUmodule;
using MH1Function = CUfunction;
using MH1Status = CUresult;
constexpr MH1Status mh1_success = CUDA_SUCCESS;
#else
using MH1Module = hipModule_t;
using MH1Function = hipFunction_t;
using MH1Status = hipError_t;
constexpr MH1Status mh1_success = hipSuccess;
#endif

std::runtime_error driver_error(MH1Status status, const std::string& operation)
{
#if defined(KOKKOS_ENABLE_CUDA)
    const char* name=nullptr;
    const char* message=nullptr;
    cuGetErrorName(status,&name);
    cuGetErrorString(status,&message);
    return std::runtime_error(
        operation+" failed: "+(name==nullptr?"CUDA_ERROR_UNKNOWN":name)
        +(message==nullptr?std::string():std::string(" (")+message+")"));
#else
    return std::runtime_error(
        operation+" failed: "+hipGetErrorName(status)
        +" ("+hipGetErrorString(status)+")");
#endif
}

void check_driver(MH1Status status, const std::string& operation)
{
    if(status!=mh1_success) throw driver_error(status,operation);
}

std::vector<char> read_cubin(const std::string& path)
{
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input) throw plugin_error(path,"could not open cubin");
    const auto end=input.tellg();
    if(end<=0) throw plugin_error(path,"cubin is empty");
    if(static_cast<unsigned long long>(end)
        >static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max()))
        throw plugin_error(path,"cubin is too large");
    std::vector<char> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    if(!input.read(bytes.data(),static_cast<std::streamsize>(bytes.size())))
        throw plugin_error(path,"could not read cubin");
    return bytes;
}

MH1Function resolve_function(
    const std::string& path, MH1Module module, const std::string& name,
    std::int32_t threads)
{
#if defined(KOKKOS_ENABLE_CUDA)
    MH1Function function=nullptr;
    const auto status=cuModuleGetFunction(&function,module,name.c_str());
    if(status!=CUDA_SUCCESS)
        throw plugin_error(path,"could not resolve kernel "+name+": "
            +driver_error(status,"cuModuleGetFunction").what());
    int maximum=0;
    check_driver(cuFuncGetAttribute(
        &maximum,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function),
        "cuFuncGetAttribute");
#else
    MH1Function function=nullptr;
    const auto status=hipModuleGetFunction(&function,module,name.c_str());
    if(status!=hipSuccess)
        throw plugin_error(path,"could not resolve kernel "+name+": "
            +driver_error(status,"hipModuleGetFunction").what());
    int maximum=0;
    check_driver(hipFuncGetAttribute(
        &maximum,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function),
        "hipFuncGetAttribute");
#endif
    if(threads<=0||threads>maximum)
        throw plugin_error(path,"kernel "+name
            +" thread-block size exceeds function limit");
    return function;
}

void validate_module_context(const MH1CudaModuleState& state)
{
#if defined(KOKKOS_ENABLE_CUDA)
    CUcontext current=nullptr;
    check_driver(cuCtxGetCurrent(&current),"cuCtxGetCurrent");
    if(current==nullptr)
        throw std::runtime_error("No CUDA context is active for the Execution MH1 module");
    if(reinterpret_cast<void*>(current)!=state.context)
        throw std::runtime_error(
            "The active CUDA context does not own the loaded Execution MH1 module");
#else
    int current=-1;
    check_driver(hipGetDevice(&current),"hipGetDevice");
    if(current!=state.device)
        throw std::runtime_error(
            "The active HIP device does not own the loaded Execution MH1 module");
#endif
}

std::int32_t launch_blocks(
    std::int64_t work_items, std::int32_t threads,
    std::int32_t persistent_blocks)
{
    const auto required=(work_items+threads-1)/threads;
    return static_cast<std::int32_t>(
        required<persistent_blocks?required:persistent_blocks);
}

std::string read_module_text(
    const std::string& path,
    MH1Module module,
    const char* symbol,
    std::size_t maximum_size = 512)
{
#if defined(KOKKOS_ENABLE_CUDA)
    CUdeviceptr address = 0;
    std::size_t size = 0;
    const auto lookup = cuModuleGetGlobal(&address, &size, module, symbol);
    if(lookup!=CUDA_SUCCESS)
        throw plugin_error(path,"could not resolve module identity "
            +std::string(symbol)+": "
            +driver_error(lookup,"cuModuleGetGlobal").what());
    if(size<2||size>maximum_size)
        throw plugin_error(path,"module identity has an invalid size");
    std::string value(size,'\0');
    check_driver(cuMemcpyDtoH(value.data(),address,size),
        "cuMemcpyDtoH(module identity)");
#else
    hipDeviceptr_t address{};
    std::size_t size=0;
    const auto lookup=hipModuleGetGlobal(&address,&size,module,symbol);
    if(lookup!=hipSuccess)
        throw plugin_error(path,"could not resolve module identity "
            +std::string(symbol)+": "
            +driver_error(lookup,"hipModuleGetGlobal").what());
    if(size<2||size>maximum_size)
        throw plugin_error(path,"module identity has an invalid size");
    std::string value(size,'\0');
    check_driver(hipMemcpyDtoH(value.data(),address,size),
        "hipMemcpyDtoH(module identity)");
#endif
    if(value.back()!='\0')
        throw plugin_error(path,"module identity is not null terminated");
    value.pop_back();
    if(value.empty()) throw plugin_error(path,"module identity is empty");
    return value;
}

template<typename Packet>
void launch_packet(
    void* raw_function, const Packet& packet, void* raw_stream,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int threads)
{
    auto argument=packet;
    void* parameters[]={&argument};
#if defined(KOKKOS_ENABLE_CUDA)
    check_driver(cuLaunchKernel(
        reinterpret_cast<CUfunction>(raw_function),grid_x,grid_y,grid_z,
        threads,1,1,0,reinterpret_cast<CUstream>(raw_stream),parameters,nullptr),
        "cuLaunchKernel(Execution MH1 module)");
#else
    check_driver(hipModuleLaunchKernel(
        reinterpret_cast<hipFunction_t>(raw_function),grid_x,grid_y,grid_z,
        threads,1,1,0,reinterpret_cast<hipStream_t>(raw_stream),parameters,nullptr),
        "hipModuleLaunchKernel(Execution MH1 module)");
#endif
}

template<typename FirstPacket,typename SecondPacket>
void launch_packet_pair(
    void* raw_function, const FirstPacket& first, const SecondPacket& second,
    void* raw_stream, unsigned int blocks, unsigned int threads)
{
    auto first_argument=first;
    auto second_argument=second;
    void* parameters[]={&first_argument,&second_argument};
#if defined(KOKKOS_ENABLE_CUDA)
    check_driver(cuLaunchKernel(
        reinterpret_cast<CUfunction>(raw_function),blocks,1,1,threads,1,1,0,
        reinterpret_cast<CUstream>(raw_stream),parameters,nullptr),
        "cuLaunchKernel(Execution MH1 spline R module)");
#else
    check_driver(hipModuleLaunchKernel(
        reinterpret_cast<hipFunction_t>(raw_function),blocks,1,1,threads,1,1,0,
        reinterpret_cast<hipStream_t>(raw_stream),parameters,nullptr),
        "hipModuleLaunchKernel(Execution MH1 spline R module)");
#endif
}

MH1CudaModuleLaunch parse_module_launch(
    const std::string& path, MH1Module module, const nlohmann::json& value,
    const std::int32_t launch_plan_version)
{
    if(!value.is_object()) throw plugin_error(path,"node launch is not an object");
    const auto kernel=value.at("kernel").get<std::string>();
    const auto grid=value.at("grid").get<std::string>();
    const auto threads=value.at("threads").get<std::int32_t>();
    MH1CudaModuleLaunch result;
    result.function=reinterpret_cast<void*>(
        resolve_function(path,module,kernel,threads));
    result.threads=threads;
    if(grid=="persistent_nodes") {
        result.grid_kind=0;
    } else if(grid=="node_channel_tiles") {
        result.grid_kind=1;
        result.channels=value.at("channels").get<std::int32_t>();
        result.components=value.at("components").get<std::int32_t>();
        const bool has_tile_nodes=value.contains("tile_nodes");
        const bool has_tile_channels=value.contains("tile_channels");
        if(launch_plan_version==1) {
            if(has_tile_nodes||has_tile_channels)
                throw plugin_error(
                    path,"version-1 node launch contains tile geometry");
            result.tile_nodes=8;
            result.tile_channels=32;
        } else {
            if(!has_tile_nodes||!has_tile_channels)
                throw plugin_error(
                    path,"version-2 node launch omits tile geometry");
            result.tile_nodes=value.at("tile_nodes").get<std::int32_t>();
            result.tile_channels=value.at("tile_channels").get<std::int32_t>();
        }
        const bool standard_tiles=(result.tile_nodes==8
            ||result.tile_nodes==16||result.tile_nodes==32)
            &&result.tile_channels==32&&threads==256;
        const bool product_tiles=
            (kernel.ends_with("_reverse_product_tiled")
                ||kernel.ends_with("_forward_product_tiled"))
            &&result.tile_nodes==1&&result.tile_channels==128&&threads==128;
        const bool paired_product_reverse_tile=
            kernel.ends_with("_reverse_product_tiled")
            &&(result.tile_nodes==2||result.tile_nodes==4)
            &&result.tile_channels==128&&threads==128;
        const bool wide_grouped_tile=
            (kernel.ends_with("_l1_reverse_message_grouped")
                ||kernel.ends_with("_l1_reverse_linear2_grouped")
                ||kernel.ends_with("_l0_reverse_linear2_grouped")
                ||kernel.ends_with("_l0_forward_linear2_grouped")
                ||kernel.ends_with("_l1_forward_linear2_grouped"))
            &&result.tile_nodes==64&&result.tile_channels==32&&threads==256;
        const bool xwide_grouped_tile=
            (kernel.ends_with("_l0_reverse_linear2_grouped")
                ||kernel.ends_with("_l1_reverse_linear2_grouped")
                ||kernel.ends_with("_l0_forward_linear2_grouped")
                ||kernel.ends_with("_l1_forward_linear2_grouped"))
            &&result.tile_nodes==128&&result.tile_channels==32&&threads==256;
        if(result.channels<=0||result.components<=0
            ||(!standard_tiles&&!product_tiles&&!paired_product_reverse_tile
                &&!wide_grouped_tile&&!xwide_grouped_tile))
            throw plugin_error(path,"node tiled launch extents are invalid");
    } else {
        throw plugin_error(path,"node launch grid kind is invalid");
    }
    return result;
}

template<typename Packet>
void launch_node_sequence(
    const MH1CudaModuleState& state,
    const std::vector<MH1CudaModuleLaunch>& launches,
    const Packet& packet, void* stream, std::int32_t persistent_blocks)
{
    for(const auto& launch:launches) {
        unsigned int grid_x=0,grid_y=1,grid_z=1;
        if(launch.grid_kind==0) {
            grid_x=static_cast<unsigned int>(
                packet.num_nodes<persistent_blocks
                    ?packet.num_nodes:persistent_blocks);
        } else {
            grid_x=static_cast<unsigned int>(
                (launch.channels+launch.tile_channels-1)/launch.tile_channels);
            grid_y=static_cast<unsigned int>(
                (packet.num_nodes+launch.tile_nodes-1)/launch.tile_nodes);
            grid_z=static_cast<unsigned int>(launch.components);
        }
        launch_packet(
            launch.function,packet,stream,grid_x,grid_y,grid_z,
            static_cast<unsigned int>(launch.threads));
    }
}
#endif

std::string require_text(
    const std::string& path,
    const char* value,
    const char* field)
{
    if(value==nullptr||value[0]=='\0')
        throw plugin_error(path,std::string(field)+" is empty");
    return value;
}

void require_match(
    const std::string& path,
    const std::string_view expected,
    const char* actual,
    const char* field)
{
    if(!expected.empty()&&expected!=actual)
        throw plugin_error(path,std::string(field)+" does not match");
}

void require_extent(
    const std::string& path,
    const std::int32_t expected,
    const std::int32_t actual,
    const char* field,
    const std::size_t interaction)
{
    if(expected>=0&&expected!=actual)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)+" "+field
                +" does not match");
}

void require_count(
    const std::string& path,
    const std::int64_t expected,
    const std::int64_t actual,
    const char* field,
    const std::size_t interaction)
{
    if(expected>=0&&expected!=actual)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)+" "+field
                +" does not match");
}

constexpr std::uint32_t known_capabilities=
    SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V3
    |SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V3
    |SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V3
    |SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V3
    |SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V3
    |SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V3
    |SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V3
    |SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V3
    |SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V3;

constexpr std::uint32_t mandatory_node_capabilities_v4=
    SYMMETRIX_JIT_MH1_CUDA_NODE_PROGRAM_PHASES_V4
    |SYMMETRIX_JIT_MH1_CUDA_PERSISTENT_IR_MUL_NODE_STATE_V4
    |SYMMETRIX_JIT_MH1_CUDA_FIXED_WEIGHT_COORDINATES_V4;

constexpr std::uint32_t known_capabilities_v4=
    SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V4
    |SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V4
    |SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V4
    |SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V4
    |SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V4
    |SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V4
    |SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V4
    |SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V4
    |SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V4
    |mandatory_node_capabilities_v4;

void validate_interaction_expectation(
    const MH1CudaInteractionExpectation& interaction)
{
    if(interaction.input_1_dimension<=0
        ||interaction.input_2_dimension<=0
        ||interaction.output_dimension<=0
        ||interaction.weight_size<=0
        ||interaction.phi_dimension<=0
        ||interaction.multiplicity<=0
        ||interaction.input_1_angular_dimension<=0
        ||interaction.instruction_count<=0)
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugin expectation requires exact positive "
            "interaction extents.");
}

void validate_expectation(const MH1CudaPluginExpectation& expectation)
{
    if(expectation.artifact_id.empty())
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugin expectation requires an artifact id.");
    if(expectation.generation_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugin expectation requires a generation "
            "fingerprint.");
    if(expectation.semantic_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugin expectation requires a semantic "
            "fingerprint.");
    if(expectation.structure_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugin expectation requires a structure "
            "fingerprint.");
    for(const auto& interaction:expectation.interactions)
        validate_interaction_expectation(interaction);
    if(expectation.target_compute_capability<=0
        ||expectation.max_threads_per_block<=0)
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugin expectation requires exact device extents.");
    if(expectation.required_capabilities==0
        ||(expectation.required_capabilities&~known_capabilities)!=0)
        throw std::invalid_argument(
            "Execution MH-1 CUDA plugin expectation requires exact known "
            "capabilities.");
}

void validate_node_expectation(
    const MH1CudaNodeProgramExpectation& node,
    const std::int32_t max_threads_per_block)
{
    if(node.element_count<=0||node.input_dimension<=0
        ||node.up_dimension<=0||node.residual_dimension<=0
        ||node.skip_dimension<=0||node.message_dimension<=0
        ||node.interaction_output_dimension<=0||node.output_dimension<=0
        ||node.product_term_count<=0||node.node_arena_dimension<0
        ||node.linear_parameter_count<=0||node.product_parameter_count<=0
        ||node.readout_parameter_count<=0)
        throw std::invalid_argument(
            "Execution MH-1 CUDA ABI-v4 expectation requires exact "
            "node-program extents and runtime parameter counts.");
    for(const auto threads:node.forward_threads_per_block)
        if(threads<=0||threads>max_threads_per_block)
            throw std::invalid_argument(
                "Execution MH-1 CUDA ABI-v4 expectation has an invalid forward "
                "phase thread-block size.");
    for(std::size_t phase=0;
        phase<node.reverse_threads_per_block.size();++phase) {
        const auto threads=node.reverse_threads_per_block[phase];
        const bool disabled=phase==1&&!node.requires_tp_source_state_adjoint;
        if((disabled&&threads!=0)
            ||(!disabled&&(threads<=0||threads>max_threads_per_block)))
            throw std::invalid_argument(
                "Execution MH-1 CUDA ABI-v4 expectation has an invalid reverse "
                "phase thread-block size.");
    }
}

void validate_expectation(const MH1CudaPluginV4Expectation& expectation)
{
    if(expectation.artifact_id.empty())
        throw std::invalid_argument(
            "Execution MH-1 CUDA ABI-v4 expectation requires an artifact id.");
    if(expectation.generation_fingerprint.empty()
        ||expectation.semantic_fingerprint.empty()
        ||expectation.structure_fingerprint.empty()
        ||expectation.runtime_layout_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 CUDA ABI-v4 expectation requires exact model and "
            "runtime layout fingerprints.");
    const bool cuda_target=expectation.target_backend=="cuda"
        &&expectation.target_compute_capability>0;
    const bool hip_target=expectation.target_backend=="hip"
        &&!expectation.target_architecture.empty()
        &&expectation.target_compute_capability==0;
    if((!cuda_target&&!hip_target)||expectation.max_threads_per_block<=0)
        throw std::invalid_argument(
            "Execution MH-1 device ABI-v4 expectation requires exact backend, "
            "architecture, and device extents.");
    for(const auto& interaction:expectation.interactions)
        validate_interaction_expectation(interaction);
    for(std::size_t interaction=0;
        interaction<expectation.node_programs.size();++interaction) {
        const auto& node=expectation.node_programs[interaction];
        validate_node_expectation(node,expectation.max_threads_per_block);
        if(node.requires_tp_source_state_adjoint!=(interaction!=0))
            throw std::invalid_argument(
                "Execution MH-1 CUDA ABI-v4 expectation must omit the layer-0 "
                "TP source-state adjoint and require it for layer 1.");
    }
    if(expectation.required_capabilities!=known_capabilities_v4
        ||(expectation.required_capabilities
            &mandatory_node_capabilities_v4)!=mandatory_node_capabilities_v4)
        throw std::invalid_argument(
            "Execution MH-1 CUDA ABI-v4 expectation requires the complete "
            "generated interaction, node-program, persistent-ir_mul, and "
            "fixed-weight-coordinate capability set.");
}

void validate_interaction(
    const std::string& path,
    const SymmetrixJitMH1CudaInteractionV3& actual,
    const MH1CudaInteractionExpectation& expected,
    const std::int32_t max_threads_per_block,
    const std::size_t interaction)
{
    if(actual.struct_size<sizeof(SymmetrixJitMH1CudaInteractionV3))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" descriptor is smaller than ABI version 3");
    if(actual.reserved!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" reserved field is nonzero");
    if(actual.edge_reverse_physical_launch_count!=1
        &&actual.edge_reverse_physical_launch_count!=2)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" edge reverse launch count is invalid");
    if(actual.input_1_dimension<=0||actual.input_2_dimension<=0
        ||actual.output_dimension<=0||actual.weight_size<=0
        ||actual.phi_dimension<=0||actual.multiplicity<=0
        ||actual.input_1_angular_dimension<=0
        ||actual.instruction_count<=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" extents are invalid");
    if(actual.forward_threads_per_block<=0
        ||actual.source_threads_per_block<=0
        ||actual.edge_threads_per_block<=0
        ||actual.forward_threads_per_block>max_threads_per_block
        ||actual.source_threads_per_block>max_threads_per_block
        ||actual.edge_threads_per_block>max_threads_per_block)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" thread-block size is invalid for this device");

    require_extent(path,expected.input_1_dimension,
        actual.input_1_dimension,"input-1 dimension",interaction);
    require_extent(path,expected.input_2_dimension,
        actual.input_2_dimension,"input-2 dimension",interaction);
    require_extent(path,expected.output_dimension,
        actual.output_dimension,"output dimension",interaction);
    require_extent(path,expected.weight_size,
        actual.weight_size,"weight size",interaction);
    require_extent(path,expected.phi_dimension,
        actual.phi_dimension,"phi dimension",interaction);
    require_extent(path,expected.multiplicity,
        actual.multiplicity,"multiplicity",interaction);
    require_extent(path,expected.input_1_angular_dimension,
        actual.input_1_angular_dimension,
        "input-1 angular dimension",interaction);
    require_extent(path,expected.instruction_count,
        actual.instruction_count,"instruction count",interaction);
}

void validate_launches(
    const std::string& path,
    const SymmetrixJitMH1CudaInteractionLaunchesV3& launches,
    const std::uint32_t capabilities,
    const std::size_t interaction)
{
    if(launches.struct_size
        <sizeof(SymmetrixJitMH1CudaInteractionLaunchesV3))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" launch descriptor is smaller than ABI version 3");
    if(launches.interaction!=interaction)
        throw plugin_error(
            path,"interaction launch descriptor index does not match");
    if(launches.reserved!=0||launches.reserved_2!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" launch descriptor reserved field is nonzero");
    if(((capabilities&SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V3)!=0)
            !=(launches.forward_launch!=nullptr)
        ||((capabilities&SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V3)!=0)
            !=(launches.reverse_launch!=nullptr))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" launcher capability and function pointers disagree");
}

void validate_conditioning_launches(
    const std::string& path,
    const SymmetrixJitMH1CudaConditioningLaunchesV3& launches,
    const std::uint32_t capabilities,
    const std::size_t interaction)
{
    if(launches.struct_size
        <sizeof(SymmetrixJitMH1CudaConditioningLaunchesV3))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning launch descriptor is smaller than ABI "
                "version 3");
    if(launches.interaction!=interaction)
        throw plugin_error(
            path,"conditioning launch descriptor index does not match");
    if(launches.reserved!=0||launches.reserved_2!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning launch descriptor reserved field is nonzero");
    const bool declared=(capabilities
        &SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V3)!=0;
    if(declared!=(launches.forward_launch!=nullptr)
        ||declared!=(launches.reverse_launch!=nullptr))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning capability and function pointers disagree");
}

void validate_descriptor(
    const std::string& path,
    const SymmetrixJitMH1CudaPluginV3& descriptor,
    const MH1CudaPluginExpectation& expectation)
{
    if(descriptor.abi_version!=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_ABI_VERSION)
        throw plugin_error(path,"unsupported ABI version");
    if(descriptor.struct_size<sizeof(SymmetrixJitMH1CudaPluginV3))
        throw plugin_error(path,"descriptor is smaller than ABI version 3");
    if(descriptor.pointer_size!=sizeof(void*))
        throw plugin_error(path,"pointer width does not match this process");
    if(descriptor.byte_order!=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER)
        throw plugin_error(path,"byte order does not match this process");
    if(descriptor.interaction_count
        !=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS)
        throw plugin_error(path,"descriptor must contain exactly two interactions");
    if(descriptor.scalar_size!=sizeof(float))
        throw plugin_error(path,"only float32 MH-1 CUDA plugins are supported");
    if(descriptor.reserved!=0||descriptor.reserved_2!=0)
        throw plugin_error(path,"reserved descriptor field is nonzero");

    const auto abi_tag=require_text(path,descriptor.abi_tag,"ABI tag");
    if(abi_tag!=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_ABI_TAG)
        throw plugin_error(path,"ABI tag does not match");
    const auto artifact_id=
        require_text(path,descriptor.artifact_id,"artifact id");
    const auto generation_fingerprint=require_text(
        path,descriptor.generation_fingerprint,"generation fingerprint");
    const auto semantic_fingerprint=require_text(
        path,descriptor.semantic_fingerprint,"semantic fingerprint");
    const auto structure_fingerprint=require_text(
        path,descriptor.structure_fingerprint,"structure fingerprint");

    if(descriptor.target_compute_capability<=0)
        throw plugin_error(path,"target compute capability is invalid");
    if(descriptor.capabilities!=expectation.required_capabilities)
        throw plugin_error(path,"launcher capabilities do not match");

    require_match(path,expectation.artifact_id,
        artifact_id.c_str(),"artifact id");
    require_match(path,expectation.generation_fingerprint,
        generation_fingerprint.c_str(),"generation fingerprint");
    require_match(path,expectation.semantic_fingerprint,
        semantic_fingerprint.c_str(),"semantic fingerprint");
    require_match(path,expectation.structure_fingerprint,
        structure_fingerprint.c_str(),"structure fingerprint");
    if(descriptor.target_compute_capability
        !=expectation.target_compute_capability)
        throw plugin_error(path,"target compute capability does not match");

    for(std::size_t interaction=0;
        interaction<SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS;
        ++interaction) {
        validate_interaction(
            path,descriptor.interactions[interaction],
            expectation.interactions[interaction],
            expectation.max_threads_per_block,interaction);
        validate_launches(
            path,descriptor.launches[interaction],descriptor.capabilities,
            interaction);
        validate_conditioning_launches(
            path,descriptor.conditioning_launches[interaction],
            descriptor.capabilities,interaction);
    }
}

void validate_launches_v4(
    const std::string& path,
    const SymmetrixJitMH1CudaInteractionLaunchesV4& launches,
    const std::uint32_t capabilities,
    const std::size_t interaction)
{
    if(launches.struct_size
        <sizeof(SymmetrixJitMH1CudaInteractionLaunchesV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" launch descriptor is smaller than ABI version 4");
    if(launches.interaction!=interaction)
        throw plugin_error(
            path,"interaction launch descriptor index does not match");
    if(launches.reserved!=0||launches.reserved_2!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" launch descriptor reserved field is nonzero");
    if(((capabilities&SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V4)!=0)
            !=(launches.forward_launch!=nullptr)
        ||((capabilities&SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V4)!=0)
            !=(launches.reverse_launch!=nullptr))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" launcher capability and function pointers disagree");
}

void validate_conditioning_launches_v4(
    const std::string& path,
    const SymmetrixJitMH1CudaConditioningLaunchesV4& launches,
    const std::uint32_t capabilities,
    const std::size_t interaction)
{
    if(launches.struct_size
        <sizeof(SymmetrixJitMH1CudaConditioningLaunchesV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning launch descriptor is smaller than ABI "
                "version 4");
    if(launches.interaction!=interaction)
        throw plugin_error(
            path,"conditioning launch descriptor index does not match");
    if(launches.reserved!=0||launches.reserved_2!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning launch descriptor reserved field is nonzero");
    const bool declared=(capabilities
        &SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V4)!=0;
    if(declared!=(launches.forward_launch!=nullptr)
        ||declared!=(launches.reverse_launch!=nullptr))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning capability and function pointers disagree");
}

void validate_node_forward_phase(
    const std::string& path,
    const SymmetrixJitMH1CudaNodeForwardPhaseV4& phase,
    const MH1CudaNodeProgramExpectation& expectation,
    const std::int32_t max_threads_per_block,
    const std::size_t interaction,
    const std::size_t index)
{
    if(phase.struct_size<sizeof(SymmetrixJitMH1CudaNodeForwardPhaseV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node forward phase descriptor is smaller than ABI "
                "version 4");
    if(phase.interaction!=interaction
        ||phase.phase!=SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_FORWARD_V4+index)
        throw plugin_error(
            path,"node forward phase descriptor index does not match");
    if(phase.reserved!=0||phase.reserved_2!=0||phase.reserved_3!=0
        ||phase.reserved_4!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node forward phase reserved field is nonzero");
    if(phase.threads_per_block<=0
        ||phase.threads_per_block>max_threads_per_block||phase.launch==nullptr)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node forward phase launch is invalid");
    require_extent(path,expectation.forward_threads_per_block[index],
        phase.threads_per_block,"node forward threads per block",interaction);
}

void validate_node_reverse_phase(
    const std::string& path,
    const SymmetrixJitMH1CudaNodeReversePhaseV4& phase,
    const MH1CudaNodeProgramExpectation& expectation,
    const std::int32_t max_threads_per_block,
    const std::size_t interaction,
    const std::size_t index)
{
    if(phase.struct_size<sizeof(SymmetrixJitMH1CudaNodeReversePhaseV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node reverse phase descriptor is smaller than ABI "
                "version 4");
    if(phase.interaction!=interaction
        ||phase.phase!=SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4+index)
        throw plugin_error(
            path,"node reverse phase descriptor index does not match");
    if(phase.reserved!=0||phase.reserved_2!=0||phase.reserved_3!=0
        ||phase.reserved_4!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node reverse phase reserved field is nonzero");
    const bool disabled=index==1
        &&!expectation.requires_tp_source_state_adjoint;
    if(disabled) {
        if(phase.threads_per_block!=0||phase.launch!=nullptr)
            throw plugin_error(
                path,"interaction "+std::to_string(interaction)
                    +" disabled node reverse phase is not empty");
        return;
    }
    if(phase.threads_per_block<=0
        ||phase.threads_per_block>max_threads_per_block||phase.launch==nullptr)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node reverse phase launch is invalid");
    require_extent(path,expectation.reverse_threads_per_block[index],
        phase.threads_per_block,"node reverse threads per block",interaction);
}

void validate_node_program(
    const std::string& path,
    const SymmetrixJitMH1CudaNodeProgramV4& node,
    const MH1CudaNodeProgramExpectation& expectation,
    const std::uint32_t capabilities,
    const std::int32_t max_threads_per_block,
    const std::size_t interaction)
{
    if((capabilities&mandatory_node_capabilities_v4)
        !=mandatory_node_capabilities_v4)
        throw plugin_error(path,"ABI-v4 node capabilities are incomplete");
    if(node.struct_size<sizeof(SymmetrixJitMH1CudaNodeProgramV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node-program descriptor is smaller than ABI version 4");
    if(node.interaction!=interaction)
        throw plugin_error(path,"node-program descriptor index does not match");
    if(node.forward_phase_count
            !=SYMMETRIX_JIT_MH1_CUDA_NODE_FORWARD_PHASES_V4
        ||node.reverse_phase_count
            !=SYMMETRIX_JIT_MH1_CUDA_NODE_REVERSE_PHASES_V4)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node-program phase count is invalid");
    const std::uint32_t known_flags=
        SYMMETRIX_JIT_MH1_CUDA_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4
        |SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_PRE_GATE_V4
        |SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_INTERACTION_OUTPUT_V4
        |SYMMETRIX_JIT_MH1_CUDA_NODE_REUSE_MESSAGE_ADJOINT_V4;
    const bool requires_tp=(node.flags
        &SYMMETRIX_JIT_MH1_CUDA_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4)!=0;
    if((node.flags&~known_flags)!=0
        ||requires_tp!=expectation.requires_tp_source_state_adjoint)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" TP source-state adjoint requirement does not match");
    const bool retains_pre_gate=(node.flags
        &SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_PRE_GATE_V4)!=0;
    const bool retains_interaction_output=(node.flags
        &SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_INTERACTION_OUTPUT_V4)!=0;
    const bool reuses_message_adjoint=(node.flags
        &SYMMETRIX_JIT_MH1_CUDA_NODE_REUSE_MESSAGE_ADJOINT_V4)!=0;
    if((retains_pre_gate
            &&node.retained_pre_gate_dimension!=node.residual_dimension)
        ||(!retains_pre_gate&&node.retained_pre_gate_dimension!=0)
        ||(retains_interaction_output
            &&node.retained_interaction_output_dimension
                !=node.interaction_output_dimension)
        ||(!retains_interaction_output
            &&node.retained_interaction_output_dimension!=0)
        ||(reuses_message_adjoint&&!retains_interaction_output))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" retained node-state flags and extents disagree");
    if(node.element_count<=0||node.input_dimension<=0
        ||node.up_dimension<=0||node.residual_dimension<=0
        ||node.skip_dimension<=0||node.message_dimension<=0
        ||node.interaction_output_dimension<=0||node.output_dimension<=0
        ||node.product_term_count<=0||node.node_arena_dimension<0
        ||node.linear_parameter_count<=0||node.product_parameter_count<=0
        ||node.readout_parameter_count<=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node-program extents are invalid");

    require_extent(path,expectation.element_count,node.element_count,
        "node element count",interaction);
    require_extent(path,expectation.input_dimension,node.input_dimension,
        "node input dimension",interaction);
    require_extent(path,expectation.up_dimension,node.up_dimension,
        "node up dimension",interaction);
    require_extent(path,expectation.residual_dimension,node.residual_dimension,
        "node residual dimension",interaction);
    require_extent(path,expectation.skip_dimension,node.skip_dimension,
        "node skip dimension",interaction);
    require_extent(path,expectation.message_dimension,node.message_dimension,
        "node message dimension",interaction);
    require_extent(path,expectation.interaction_output_dimension,
        node.interaction_output_dimension,
        "node interaction output dimension",interaction);
    require_extent(path,expectation.output_dimension,node.output_dimension,
        "node output dimension",interaction);
    require_extent(path,expectation.product_term_count,node.product_term_count,
        "node product term count",interaction);
    require_extent(path,expectation.node_arena_dimension,
        node.node_arena_dimension,"node arena dimension",interaction);
    require_count(path,expectation.linear_parameter_count,
        node.linear_parameter_count,"node linear parameter count",interaction);
    require_count(path,expectation.product_parameter_count,
        node.product_parameter_count,"node product parameter count",interaction);
    require_count(path,expectation.readout_parameter_count,
        node.readout_parameter_count,"node readout parameter count",interaction);
    for(std::size_t phase=0;
        phase<SYMMETRIX_JIT_MH1_CUDA_NODE_FORWARD_PHASES_V4;++phase)
        validate_node_forward_phase(path,node.forward_phases[phase],expectation,
            max_threads_per_block,interaction,phase);
    for(std::size_t phase=0;
        phase<SYMMETRIX_JIT_MH1_CUDA_NODE_REVERSE_PHASES_V4;++phase)
        validate_node_reverse_phase(path,node.reverse_phases[phase],expectation,
            max_threads_per_block,interaction,phase);
}

void validate_descriptor(
    const std::string& path,
    const SymmetrixJitMH1CudaPluginV4& descriptor,
    const MH1CudaPluginV4Expectation& expectation)
{
    if(descriptor.abi_version
        !=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_VERSION)
        throw plugin_error(path,"unsupported ABI version");
    if(descriptor.struct_size<sizeof(SymmetrixJitMH1CudaPluginV4))
        throw plugin_error(path,"descriptor is smaller than ABI version 4");
    if(descriptor.pointer_size!=sizeof(void*))
        throw plugin_error(path,"pointer width does not match this process");
    if(descriptor.byte_order!=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER)
        throw plugin_error(path,"byte order does not match this process");
    if(descriptor.interaction_count
        !=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS)
        throw plugin_error(path,"descriptor must contain exactly two interactions");
    if(descriptor.scalar_size!=expectation.scalar_size)
        throw plugin_error(path,"scalar width does not match model execution");
    if(descriptor.scalar_size!=sizeof(float)
        &&descriptor.scalar_size!=sizeof(double))
        throw plugin_error(path,"unsupported MH-1 device scalar width");
    if(descriptor.reserved!=0||descriptor.reserved_2!=0)
        throw plugin_error(path,"reserved descriptor field is nonzero");
    if(descriptor.target_compute_capability<=0)
        throw plugin_error(path,"target compute capability is invalid");

    const auto abi_tag=require_text(path,descriptor.abi_tag,"ABI tag");
    if(abi_tag!=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_TAG)
        throw plugin_error(path,"ABI tag does not match");
    const auto artifact_id=require_text(
        path,descriptor.artifact_id,"artifact id");
    const auto generation_fingerprint=require_text(
        path,descriptor.generation_fingerprint,"generation fingerprint");
    const auto semantic_fingerprint=require_text(
        path,descriptor.semantic_fingerprint,"semantic fingerprint");
    const auto structure_fingerprint=require_text(
        path,descriptor.structure_fingerprint,"structure fingerprint");
    const auto runtime_layout_fingerprint=require_text(
        path,descriptor.runtime_layout_fingerprint,
        "runtime layout fingerprint");
    if(descriptor.capabilities!=expectation.required_capabilities
        ||descriptor.capabilities!=known_capabilities_v4)
        throw plugin_error(path,"ABI-v4 launcher capabilities do not match");

    require_match(path,expectation.artifact_id,
        artifact_id.c_str(),"artifact id");
    require_match(path,expectation.generation_fingerprint,
        generation_fingerprint.c_str(),"generation fingerprint");
    require_match(path,expectation.semantic_fingerprint,
        semantic_fingerprint.c_str(),"semantic fingerprint");
    require_match(path,expectation.structure_fingerprint,
        structure_fingerprint.c_str(),"structure fingerprint");
    require_match(path,expectation.runtime_layout_fingerprint,
        runtime_layout_fingerprint.c_str(),"runtime layout fingerprint");
    if(descriptor.target_compute_capability
        !=expectation.target_compute_capability)
        throw plugin_error(path,"target compute capability does not match");

    for(std::size_t interaction=0;
        interaction<SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS;
        ++interaction) {
        validate_interaction(path,descriptor.interactions[interaction],
            expectation.interactions[interaction],
            expectation.max_threads_per_block,interaction);
        validate_launches_v4(path,descriptor.launches[interaction],
            descriptor.capabilities,interaction);
        validate_conditioning_launches_v4(
            path,descriptor.conditioning_launches[interaction],
            descriptor.capabilities,interaction);
        validate_node_program(path,descriptor.node_programs[interaction],
            expectation.node_programs[interaction],descriptor.capabilities,
            expectation.max_threads_per_block,interaction);
    }
}

void validate_retained_node_state(
    const SymmetrixJitMH1CudaNodeProgramV4& node,
    const float* retained_pre_gate,
    const float* retained_interaction_output)
{
    const bool requires_pre_gate=(node.flags
        &SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_PRE_GATE_V4)!=0;
    const bool requires_interaction_output=(node.flags
        &SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_INTERACTION_OUTPUT_V4)!=0;
    if((retained_pre_gate!=nullptr)!=requires_pre_gate
        ||(retained_interaction_output!=nullptr)
            !=requires_interaction_output)
        throw std::invalid_argument(
            "Execution MH1 retained node-state packet does not match the loaded "
            "node program");
}

}  // namespace

MH1CudaPlugin MH1CudaPlugin::load(
    std::string path,
    const MH1CudaPluginExpectation& expectation)
{
    if(path.empty())
        throw std::invalid_argument("Execution MH-1 CUDA plugin path is empty.");
    validate_expectation(expectation);

    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(handle==nullptr) {
        const char* error=dlerror();
        throw plugin_error(path,error==nullptr?"dlopen failed":error);
    }
    try {
        dlerror();
        void* symbol=dlsym(
            handle,SYMMETRIX_JIT_MH1_CUDA_PLUGIN_QUERY_SYMBOL);
        const char* symbol_error=dlerror();
        if(symbol_error!=nullptr) throw plugin_error(path,symbol_error);
        if(symbol==nullptr) throw plugin_error(path,"query symbol is null");
        const auto query=
            reinterpret_cast<SymmetrixJitMH1CudaPluginQueryV3>(symbol);
        const auto* descriptor=query();
        if(descriptor==nullptr)
            throw plugin_error(path,"query returned a null descriptor");
        validate_descriptor(path,*descriptor,expectation);
        return MH1CudaPlugin(handle,descriptor,std::move(path));
    } catch(...) {
        dlclose(handle);
        throw;
    }
}

MH1CudaPlugin::MH1CudaPlugin(
    void* handle,
    const SymmetrixJitMH1CudaPluginV3* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1CudaPlugin::~MH1CudaPlugin()
{
    reset();
}

MH1CudaPlugin::MH1CudaPlugin(MH1CudaPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

MH1CudaPlugin& MH1CudaPlugin::operator=(MH1CudaPlugin&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1CudaPluginV3& MH1CudaPlugin::descriptor() const
{
    if(descriptor_==nullptr)
        throw std::logic_error("Execution MH-1 CUDA plugin handle is empty.");
    return *descriptor_;
}

void MH1CudaPlugin::reset() noexcept
{
    descriptor_=nullptr;
    if(handle_!=nullptr) {
        dlclose(handle_);
        handle_=nullptr;
    }
    path_.clear();
}

MH1CudaPluginV4 MH1CudaPluginV4::load(
    std::string path,
    const MH1CudaPluginV4Expectation& expectation)
{
    if(path.empty())
        throw std::invalid_argument(
            "Execution MH-1 CUDA ABI-v4 plugin path is empty.");
    validate_expectation(expectation);

    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(handle==nullptr) {
        const char* error=dlerror();
        throw plugin_error(path,error==nullptr?"dlopen failed":error);
    }
    try {
        dlerror();
        void* symbol=dlsym(
            handle,SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_QUERY_SYMBOL);
        const char* symbol_error=dlerror();
        if(symbol_error!=nullptr)
            throw plugin_error(path,symbol_error);
        if(symbol==nullptr)
            throw plugin_error(path,"ABI-v4 query symbol is null");
        const auto query=
            reinterpret_cast<SymmetrixJitMH1CudaPluginQueryV4>(symbol);
        const auto* descriptor=query();
        if(descriptor==nullptr)
            throw plugin_error(path,"ABI-v4 query returned a null descriptor");
        validate_descriptor(path,*descriptor,expectation);
        return MH1CudaPluginV4(handle,descriptor,std::move(path));
    } catch(...) {
        dlclose(handle);
        throw;
    }
}

MH1CudaPluginV4 MH1CudaPluginV4::load_module(
    std::string path,
    const MH1CudaPluginV4Expectation& expectation,
    std::string launch_plan_json)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if(path.empty())
        throw std::invalid_argument("Execution MH-1 device module path is empty.");
    validate_expectation(expectation);
    if(launch_plan_json.empty())
        throw std::invalid_argument("Execution MH-1 device launch plan is empty.");
    const auto bytes=read_cubin(path);
#if defined(KOKKOS_ENABLE_CUDA)
    CUcontext context=nullptr;
    check_driver(cuCtxGetCurrent(&context),"cuCtxGetCurrent");
    if(context==nullptr)
        throw plugin_error(path,"no CUDA context is active");
    MH1Module module=nullptr;
    check_driver(cuModuleLoadData(&module,bytes.data()),"cuModuleLoadData");
#else
    int device=-1;
    check_driver(hipGetDevice(&device),"hipGetDevice");
    MH1Module module=nullptr;
    check_driver(hipModuleLoadData(&module,bytes.data()),"hipModuleLoadData");
#endif
    try {
        const auto plan=nlohmann::json::parse(launch_plan_json);
        const auto launch_plan_schema=plan.at("schema").get<std::string>();
        std::int32_t launch_plan_version=0;
#if defined(KOKKOS_ENABLE_CUDA)
        if(launch_plan_schema=="symmetrix.jit.mh1.cuda-launch-plan/1")
            launch_plan_version=1;
        else if(launch_plan_schema=="symmetrix.jit.mh1.cuda-launch-plan/2")
            launch_plan_version=2;
        else if(launch_plan_schema=="symmetrix.jit.mh1.cuda-launch-plan/3")
            launch_plan_version=3;
        else
            throw plugin_error(path,"launch-plan schema is unsupported");
#else
        if(launch_plan_schema=="symmetrix.jit.mh1.hip-launch-plan/1")
            launch_plan_version=1;
        else if(launch_plan_schema=="symmetrix.jit.mh1.hip-launch-plan/2")
            launch_plan_version=2;
        else if(launch_plan_schema=="symmetrix.jit.mh1.hip-launch-plan/3")
            launch_plan_version=3;
        else
            throw plugin_error(path,"launch-plan schema is unsupported");
#endif
        if(plan.at("version")!=1
            ||plan.at("artifact_id")!=expectation.artifact_id
            ||plan.at("generation_fingerprint")
                !=expectation.generation_fingerprint
            ||plan.at("semantic_fingerprint")
                !=expectation.semantic_fingerprint
            ||plan.at("structure_fingerprint")
                !=expectation.structure_fingerprint
            ||plan.at("runtime_layout_fingerprint")
                !=expectation.runtime_layout_fingerprint
            ||plan.at("scalar_size")!=expectation.scalar_size)
            throw plugin_error(path,"launch-plan identity does not match");
#if defined(KOKKOS_ENABLE_CUDA)
        if(expectation.target_backend!="cuda"
            ||plan.at("target_compute_capability")
                !=expectation.target_compute_capability)
            throw plugin_error(path,"launch-plan CUDA target does not match");
#else
        if(expectation.target_backend!="hip"
            ||plan.at("target").at("backend")!="hip"
            ||plan.at("target").at("architecture")
                !=expectation.target_architecture)
            throw plugin_error(path,"launch-plan HIP target does not match");
#endif
        const auto module_identity=read_module_text(
            path,module,"symmetrix_execution_mh1_module_identity");
        if(plan.at("module_identity")!=module_identity)
            throw plugin_error(path,"module and launch-plan identities do not match");
        const auto& interaction_plans=plan.at("interactions");
        const auto& node_plans=plan.at("nodes");
        if(launch_plan_version==2
            &&(!plan.contains("node_state_policy")
                ||!plan.at("node_state_policy").is_string()))
            throw plugin_error(
                path,"launch-plan v2 requires an explicit node-state policy");
        const auto node_state_policy=launch_plan_version==1
            ?plan.value("node_state_policy",std::string("full-retention-v1"))
            :plan.at("node_state_policy").get<std::string>();
        if(node_state_policy!="full-retention-v1"
            &&node_state_policy!="recompute-v1"
            &&node_state_policy!="reuse-adjoints-v1"
            &&node_state_policy!="retain-interaction-v1")
            throw plugin_error(path,"node-state retention policy is invalid");
        const bool retain_pre_gate=node_state_policy!="recompute-v1"
            &&node_state_policy!="retain-interaction-v1";
        const bool retain_interaction_output=
            node_state_policy!="recompute-v1";
        const bool reuse_message_adjoint=
            node_state_policy=="reuse-adjoints-v1"
            ||node_state_policy=="retain-interaction-v1";
        if(!interaction_plans.is_array()||interaction_plans.size()!=2
            ||!node_plans.is_array()||node_plans.size()!=2)
            throw plugin_error(path,"launch plan must contain two interactions");

        auto state=std::make_unique<MH1CudaModuleState>();
        state->module=reinterpret_cast<void*>(module);
#if defined(KOKKOS_ENABLE_CUDA)
        state->context=reinterpret_cast<void*>(context);
#else
        state->device=device;
#endif
        state->abi_tag=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_TAG;
        state->artifact_id=std::string(expectation.artifact_id);
        state->generation_fingerprint=
            std::string(expectation.generation_fingerprint);
        state->semantic_fingerprint=
            std::string(expectation.semantic_fingerprint);
        state->structure_fingerprint=
            std::string(expectation.structure_fingerprint);
        state->runtime_layout_fingerprint=
            std::string(expectation.runtime_layout_fingerprint);

        for(std::size_t index=0;index<2;++index) {
            const auto& value=interaction_plans.at(index);
            if(value.at("index")!=index)
                throw plugin_error(path,"interaction launch-plan index does not match");
            auto& interaction=state->interactions.at(index);
            interaction.forward_partitions=
                value.at("forward_partitions").get<std::int32_t>();
            interaction.source_partitions=
                value.at("source_partitions").get<std::int32_t>();
            interaction.forward_threads=
                value.at("forward_threads").get<std::int32_t>();
            interaction.source_threads=
                value.at("source_threads").get<std::int32_t>();
            interaction.edge_threads=
                value.at("edge_threads").get<std::int32_t>();
            if(interaction.forward_partitions<=0
                ||interaction.source_partitions<=0)
                throw plugin_error(path,"interaction partition count is invalid");
            const std::string prefix="symmetrix_execution_mh1_";
            interaction.forward=reinterpret_cast<void*>(resolve_function(
                path,module,prefix+"forward_kernel_"+std::to_string(index),
                interaction.forward_threads));
            interaction.source=reinterpret_cast<void*>(resolve_function(
                path,module,prefix+"source_reverse_kernel_"+std::to_string(index),
                interaction.source_threads));
            interaction.conditioning_forward=reinterpret_cast<void*>(
                resolve_function(path,module,prefix+"conditioning_forward_kernel_"
                    +std::to_string(index),interaction.edge_threads));
            interaction.conditioning_reverse[0]=reinterpret_cast<void*>(
                resolve_function(path,module,
                    prefix+"conditioning_reverse_prefix_kernel_"
                        +std::to_string(index),interaction.edge_threads));
            interaction.conditioning_reverse[1]=reinterpret_cast<void*>(
                resolve_function(path,module,
                    prefix+"conditioning_reverse_density_kernel_"
                        +std::to_string(index),interaction.edge_threads));
            const auto strategy=value.at("edge_strategy").get<std::string>();
            const auto phi_schedule=
                value.at("edge_phi_schedule").get<std::string>();
            interaction.edge_phi_block_multiplier=
                value.value("edge_phi_block_multiplier",1);
            interaction.edge_phi_edge_batch=
                strategy=="compact_fused"||phi_schedule=="path_tiled"?4:1;
            if(interaction.edge_phi_block_multiplier<=0)
                throw plugin_error(path,"edge phi block multiplier is invalid");
            if(strategy=="split") {
                interaction.edge_count=2;
                interaction.edge[0]=reinterpret_cast<void*>(resolve_function(
                    path,module,prefix+"edge_reverse_phi_kernel_"
                        +std::to_string(index),interaction.edge_threads));
                interaction.edge[1]=reinterpret_cast<void*>(resolve_function(
                    path,module,prefix+"edge_reverse_harmonic_kernel_"
                        +std::to_string(index),interaction.edge_threads));
            } else if(strategy=="compact_fused") {
                if(interaction.edge_threads!=128
                    ||interaction.edge_phi_block_multiplier!=4)
                    throw plugin_error(
                        path,"compact fused edge schedule is invalid");
                interaction.edge_count=1;
                interaction.edge[0]=reinterpret_cast<void*>(resolve_function(
                    path,module,prefix+"edge_reverse_compact_fused_kernel_"
                        +std::to_string(index),interaction.edge_threads));
            } else if(strategy=="fused") {
                interaction.edge_count=1;
                interaction.edge[0]=reinterpret_cast<void*>(resolve_function(
                    path,module,prefix+"edge_reverse_fused_kernel_"
                        +std::to_string(index),interaction.edge_threads));
            } else {
                throw plugin_error(path,"edge launch strategy is invalid");
            }
            if(launch_plan_version>=3) {
                const auto& spline=value.at("spline_r");
                if(spline.at("schedule")
                    !="shared-r1-receiver-source-fused-v1")
                    throw plugin_error(path,"spline R schedule is unsupported");
                interaction.spline_forward_threads=
                    spline.at("forward_threads").get<std::int32_t>();
                interaction.spline_reverse_threads=
                    spline.at("reverse_threads").get<std::int32_t>();
                interaction.spline_forward=reinterpret_cast<void*>(
                    resolve_function(path,module,
                        spline.at("forward_kernel").get<std::string>(),
                        interaction.spline_forward_threads));
                interaction.spline_reverse=reinterpret_cast<void*>(
                    resolve_function(path,module,
                        spline.at("reverse_kernel").get<std::string>(),
                        interaction.spline_reverse_threads));
            }

            const auto& node_value=node_plans.at(index);
            if(node_value.at("index")!=index)
                throw plugin_error(path,"node launch-plan index does not match");
            auto parse_sequence=[&](const char* field) {
                std::vector<MH1CudaModuleLaunch> result;
                for(const auto& launch:node_value.at(field))
                    result.push_back(parse_module_launch(
                        path,module,launch,launch_plan_version));
                return result;
            };
            auto& node=state->nodes.at(index);
            node.pre_forward=parse_sequence("pre_forward");
            node.post_forward=parse_sequence("post_forward");
            node.post_reverse=parse_sequence("post_reverse");
            node.pre_reverse=parse_sequence("pre_reverse");
            if(node.pre_forward.size()!=1||node.post_forward.empty()
                ||node.post_reverse.empty()
                ||node.pre_reverse.size()!=(index==0?0u:1u))
                throw plugin_error(path,"node phase launch sequence is incomplete");
        }

        auto& descriptor=state->descriptor;
        descriptor.abi_version=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_VERSION;
        descriptor.struct_size=sizeof(descriptor);
        descriptor.pointer_size=sizeof(void*);
        descriptor.byte_order=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER;
        descriptor.capabilities=expectation.required_capabilities;
        descriptor.interaction_count=2;
        descriptor.scalar_size=expectation.scalar_size;
        descriptor.abi_tag=state->abi_tag.c_str();
        descriptor.artifact_id=state->artifact_id.c_str();
        descriptor.generation_fingerprint=state->generation_fingerprint.c_str();
        descriptor.semantic_fingerprint=state->semantic_fingerprint.c_str();
        descriptor.structure_fingerprint=state->structure_fingerprint.c_str();
        descriptor.runtime_layout_fingerprint=
            state->runtime_layout_fingerprint.c_str();
        descriptor.target_compute_capability=
            expectation.target_compute_capability;
        for(std::size_t index=0;index<2;++index) {
            const auto& expected=expectation.interactions[index];
            const auto& launch=state->interactions[index];
            descriptor.interactions[index]={
                sizeof(SymmetrixJitMH1CudaInteractionV4),0u,
                expected.input_1_dimension,expected.input_2_dimension,
                expected.output_dimension,expected.weight_size,
                expected.phi_dimension,expected.multiplicity,
                expected.input_1_angular_dimension,expected.instruction_count,
                launch.forward_threads,launch.source_threads,launch.edge_threads,
                launch.edge_count};
            const auto& expected_node=expectation.node_programs[index];
            auto& node=descriptor.node_programs[index];
            node.struct_size=sizeof(node);
            node.interaction=index;
            node.forward_phase_count=2;
            node.reverse_phase_count=2;
            node.element_count=expected_node.element_count;
            node.input_dimension=expected_node.input_dimension;
            node.up_dimension=expected_node.up_dimension;
            node.residual_dimension=expected_node.residual_dimension;
            node.skip_dimension=expected_node.skip_dimension;
            node.message_dimension=expected_node.message_dimension;
            node.interaction_output_dimension=
                expected_node.interaction_output_dimension;
            node.output_dimension=expected_node.output_dimension;
            node.product_term_count=expected_node.product_term_count;
            node.node_arena_dimension=expected_node.node_arena_dimension;
            node.flags=expected_node.requires_tp_source_state_adjoint
                ?SYMMETRIX_JIT_MH1_CUDA_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4
                :0u;
            if(retain_pre_gate)
                node.flags|=SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_PRE_GATE_V4;
            if(retain_interaction_output)
                node.flags|=
                    SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_INTERACTION_OUTPUT_V4;
            if(reuse_message_adjoint)
                node.flags|=
                    SYMMETRIX_JIT_MH1_CUDA_NODE_REUSE_MESSAGE_ADJOINT_V4;
            node.retained_pre_gate_dimension=retain_pre_gate
                ?expected_node.residual_dimension:0;
            node.retained_interaction_output_dimension=retain_interaction_output
                ?expected_node.interaction_output_dimension:0;
            node.linear_parameter_count=expected_node.linear_parameter_count;
            node.product_parameter_count=expected_node.product_parameter_count;
            node.readout_parameter_count=expected_node.readout_parameter_count;
        }
        MH1CudaPluginV4 result;
        result.module_state_=std::move(state);
        result.path_=std::move(path);
        return result;
    } catch(...) {
#if defined(KOKKOS_ENABLE_CUDA)
        static_cast<void>(cuModuleUnload(module));
#else
        static_cast<void>(hipModuleUnload(module));
#endif
        throw;
    }
#else
    (void)path;(void)expectation;(void)launch_plan_json;
    throw std::runtime_error(
        "Execution MH-1 device modules require a GPU-enabled Kokkos build.");
#endif
}

MH1CudaPluginV4 MH1CudaPluginV4::load_cubin(
    std::string path,
    const MH1CudaPluginV4Expectation& expectation,
    std::string launch_plan_json)
{
    return load_module(
        std::move(path),expectation,std::move(launch_plan_json));
}

MH1CudaPluginV4::MH1CudaPluginV4() = default;

MH1CudaPluginV4::MH1CudaPluginV4(
    void* handle,
    const SymmetrixJitMH1CudaPluginV4* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1CudaPluginV4::~MH1CudaPluginV4()
{
    reset();
}

MH1CudaPluginV4::MH1CudaPluginV4(MH1CudaPluginV4&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      module_state_(std::move(other.module_state_)),
      path_(std::move(other.path_))
{}

MH1CudaPluginV4& MH1CudaPluginV4::operator=(MH1CudaPluginV4&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        module_state_=std::move(other.module_state_);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1CudaPluginV4& MH1CudaPluginV4::descriptor() const
{
    if(module_state_!=nullptr) return module_state_->descriptor;
    if(descriptor_==nullptr)
        throw std::logic_error(
            "Execution MH-1 CUDA ABI-v4 plugin handle is empty.");
    return *descriptor_;
}

std::int32_t MH1CudaPluginV4::launch_forward(
    const SymmetrixJitMH1CudaForwardArgsV4* args, void* stream,
    std::int32_t persistent_blocks) const
{
    if(module_state_==nullptr) {
        if(args==nullptr||args->interaction>=2) return 1;
        return descriptor().launches[args->interaction].forward_launch(
            args,stream,persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    constexpr std::uint32_t known_flags =
        SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4
        |SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V4
        |SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V4;
    if(args==nullptr
        ||args->struct_size<sizeof(SymmetrixJitMH1CudaForwardArgsV4)
        ||args->interaction>=2||(args->flags&~known_flags)!=0u
        ||args->num_nodes<0||args->num_nodes>INT32_MAX
        ||args->num_edges<0||args->num_edges>INT32_MAX
        ||args->first_edge!=0||args->samples<0
        ||args->samples!=args->num_edges
        ||(args->samples>0&&args->num_nodes==0)
        ||args->active_receiver_count>args->num_nodes
        ||args->active_receiver_count>args->samples
        ||(args->samples>0&&args->active_receiver_count==0)
        ||persistent_blocks<=0)
        throw std::invalid_argument("Execution MH1 forward packet is invalid");
    if(args->samples==0) return 0;
    const bool apply_cutoff=(args->flags
        &SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4)!=0;
    const bool has_bias=(args->flags
        &SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V4)!=0;
    const bool has_fixed=(args->flags
        &SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V4)!=0;
    if(args->source_indices==nullptr||args->active_receivers==nullptr
        ||args->receiver_offsets==nullptr||args->edge_phi==nullptr
        ||args->linear_weight==nullptr||(has_bias&&args->linear_bias==nullptr)
        ||(has_fixed&&args->edge_linear_contribution==nullptr)
        ||args->edge_input_2==nullptr
        ||(apply_cutoff&&args->edge_cutoff_scale==nullptr)
        ||args->source_node_values==nullptr||args->output_mask==nullptr
        ||args->node_messages==nullptr)
        throw std::invalid_argument("Execution MH1 forward packet has null storage");
    validate_module_context(*module_state_);
    const auto& plan=module_state_->interactions[args->interaction];
    const auto channels=descriptor().interactions[args->interaction].multiplicity;
    const auto blocks=launch_blocks(
        static_cast<std::int64_t>(args->active_receiver_count)
            *plan.forward_partitions*((channels+31)/32)*32,
        plan.forward_threads,persistent_blocks);
    launch_packet(plan.forward,*args,stream,blocks,1,1,plan.forward_threads);
    return 0;
#else
    (void)args;(void)stream;(void)persistent_blocks;return 1;
#endif
}

std::int32_t MH1CudaPluginV4::launch_reverse(
    const SymmetrixJitMH1CudaReverseArgsV4* args, void* stream,
    std::int32_t source_persistent_blocks,
    std::int32_t edge_persistent_blocks) const
{
    if(module_state_==nullptr) {
        if(args==nullptr||args->interaction>=2) return 1;
        return descriptor().launches[args->interaction].reverse_launch(
            args,stream,source_persistent_blocks,edge_persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    constexpr std::uint32_t known_flags =
        SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4
        |SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V4
        |SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V4;
    if(args==nullptr
        ||args->struct_size<sizeof(SymmetrixJitMH1CudaReverseArgsV4)
        ||args->interaction>=2||(args->flags&~known_flags)!=0u
        ||args->reserved!=0u||args->num_nodes<0||args->num_nodes>INT32_MAX
        ||args->num_edges<0||args->num_edges>INT32_MAX
        ||args->first_edge!=0||args->samples<0
        ||args->samples!=args->num_edges
        ||(args->samples>0&&args->num_nodes==0)
        ||args->source_owner_count<0
        ||args->source_owner_count>args->samples
        ||args->source_owner_count>args->num_nodes
        ||(args->samples>0&&args->source_owner_count==0)
        ||source_persistent_blocks<=0||edge_persistent_blocks<=0)
        throw std::invalid_argument("Execution MH1 reverse packet is invalid");
    if(args->samples==0) return 0;
    const bool apply_cutoff=(args->flags
        &SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4)!=0;
    const bool has_bias=(args->flags
        &SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V4)!=0;
    const bool has_fixed=(args->flags
        &SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V4)!=0;
    if(args->source_indices==nullptr||args->target_indices==nullptr
        ||args->source_edge_offsets==nullptr||args->source_edge_indices==nullptr
        ||args->edge_phi==nullptr||args->linear_weight==nullptr
        ||(has_bias&&args->linear_bias==nullptr)
        ||(has_fixed&&args->edge_linear_contribution==nullptr)
        ||args->edge_input_2==nullptr
        ||(apply_cutoff&&args->edge_cutoff_scale==nullptr)
        ||args->source_node_values==nullptr||args->output_mask==nullptr
        ||args->target_node_output_adjoint==nullptr
        ||args->source_node_input_adjoint==nullptr
        ||args->edge_phi_adjoint==nullptr||args->edge_input_2_adjoint==nullptr
        ||args->edge_cutoff_scale_adjoint==nullptr)
        throw std::invalid_argument("Execution MH1 reverse packet has null storage");
    validate_module_context(*module_state_);
    const auto& plan=module_state_->interactions[args->interaction];
    const auto channels=descriptor().interactions[args->interaction].multiplicity;
    const auto source_blocks=launch_blocks(
        args->source_owner_count*plan.source_partitions
            *((channels+31)/32)*32,
        plan.source_threads,source_persistent_blocks);
    launch_packet(plan.source,*args,stream,source_blocks,1,1,plan.source_threads);
    const auto edge_blocks=launch_blocks(
        args->samples*16,plan.edge_threads,edge_persistent_blocks);
    for(int launch=0;launch<plan.edge_count;++launch) {
        auto blocks=edge_blocks;
        if(launch==0&&plan.edge_phi_edge_batch>1) {
            const auto requested=static_cast<std::int64_t>(
                edge_persistent_blocks)*plan.edge_phi_block_multiplier;
            const auto groups=args->samples/plan.edge_phi_edge_batch
                +(args->samples%plan.edge_phi_edge_batch!=0?1:0);
            blocks=static_cast<unsigned int>(
                groups<requested?groups:requested);
        }
        launch_packet(plan.edge[launch],*args,stream,blocks,1,1,
            plan.edge_threads);
    }
    return 0;
#else
    (void)args;(void)stream;(void)source_persistent_blocks;
    (void)edge_persistent_blocks;return 1;
#endif
}

bool MH1CudaPluginV4::has_spline_r_program() const noexcept
{
    return module_state_!=nullptr
        &&module_state_->interactions[0].spline_forward!=nullptr
        &&module_state_->interactions[0].spline_reverse!=nullptr
        &&module_state_->interactions[1].spline_forward!=nullptr
        &&module_state_->interactions[1].spline_reverse!=nullptr;
}

std::int32_t MH1CudaPluginV4::launch_spline_r_forward(
    const SymmetrixJitMH1CudaSplineRForwardArgsV5* args, void* stream,
    std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if(module_state_==nullptr||args==nullptr||args->interaction>=2
        ||args->struct_size<sizeof(*args)||args->reserved!=0u
        ||args->num_nodes<0||args->num_nodes>INT32_MAX
        ||args->num_edges<0||args->num_edges>INT32_MAX
        ||(args->num_edges>0&&args->num_nodes==0)
        ||args->active_type_count==0||persistent_blocks<=0)
        throw std::invalid_argument(
            "Execution MH1 spline R forward packet is invalid");
    if(args->num_nodes==0) return 0;
    const auto expected_functions=static_cast<std::uint32_t>(
        descriptor().interactions[args->interaction].weight_size+1);
    if(args->radial.struct_size<sizeof(args->radial)
        ||args->radial.edge_types
            !=args->active_type_count*args->active_type_count
        ||args->radial.intervals==0||args->radial.functions!=expected_functions
        ||!(args->radial.h>0.0)||args->radial.coefficients==nullptr
        ||args->node_types==nullptr||args->num_neigh==nullptr
        ||args->first_neigh==nullptr||args->neigh_indices==nullptr
        ||args->neigh_types==nullptr||args->radius==nullptr
        ||args->harmonics_values==nullptr||args->neighbor_features==nullptr
        ||args->output_mask==nullptr||args->node_density==nullptr
        ||args->output==nullptr)
        throw std::invalid_argument(
            "Execution MH1 spline R forward packet has invalid storage");
    validate_module_context(*module_state_);
    const auto& plan=module_state_->interactions[args->interaction];
    if(plan.spline_forward==nullptr||plan.spline_forward_threads<=0)
        throw std::logic_error("Execution MH1 spline R forward is unavailable");
    const auto channels=descriptor().interactions[args->interaction].multiplicity;
    const auto blocks=launch_blocks(
        args->num_nodes*static_cast<std::int64_t>(channels),
        plan.spline_forward_threads,persistent_blocks);
    launch_packet(plan.spline_forward,*args,stream,blocks,1,1,
        plan.spline_forward_threads);
    return 0;
#else
    (void)args;(void)stream;(void)persistent_blocks;return 1;
#endif
}

std::int32_t MH1CudaPluginV4::launch_spline_r_reverse(
    const SymmetrixJitMH1CudaSplineRSourceArgsV5* source_args,
    const SymmetrixJitMH1CudaSplineREdgeArgsV5* edge_args, void* stream,
    std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if(module_state_==nullptr||source_args==nullptr||edge_args==nullptr
        ||source_args->interaction>=2
        ||edge_args->interaction!=source_args->interaction
        ||source_args->struct_size<sizeof(*source_args)
        ||edge_args->struct_size<sizeof(*edge_args)
        ||source_args->reserved!=0u||source_args->num_nodes<0
        ||source_args->num_nodes>INT32_MAX||source_args->num_edges<0
        ||source_args->num_edges>INT32_MAX
        ||edge_args->num_nodes!=source_args->num_nodes
        ||edge_args->num_edges!=source_args->num_edges
        ||source_args->source_owner_count<0
        ||source_args->source_owner_count>source_args->num_nodes
        ||source_args->source_owner_count>source_args->num_edges
        ||(source_args->num_edges>0&&source_args->source_owner_count==0)
        ||source_args->active_type_count==0||persistent_blocks<=0
        ||edge_args->coordinates_are_unit>1u
        ||(edge_args->coordinates_are_unit!=0u
            ?edge_args->coordinate_scalar_size!=sizeof(float)
            :edge_args->coordinate_scalar_size!=sizeof(double)))
        throw std::invalid_argument(
            "Execution MH1 spline R reverse packet is invalid");
    if(source_args->num_edges==0) return 0;
    const auto expected_functions=static_cast<std::uint32_t>(
        descriptor().interactions[source_args->interaction].weight_size+1);
    if(source_args->radial.struct_size<sizeof(source_args->radial)
        ||source_args->radial.edge_types
            !=source_args->active_type_count*source_args->active_type_count
        ||source_args->radial.intervals==0
        ||source_args->radial.functions!=expected_functions
        ||!(source_args->radial.h>0.0)
        ||source_args->radial.coefficients==nullptr
        ||edge_args->radial.struct_size<sizeof(edge_args->radial)
        ||edge_args->radial.edge_types!=source_args->radial.edge_types
        ||edge_args->radial.intervals!=source_args->radial.intervals
        ||edge_args->radial.functions!=source_args->radial.functions
        ||edge_args->radial.h!=source_args->radial.h
        ||edge_args->radial.x0!=source_args->radial.x0
        ||edge_args->radial.coefficients!=source_args->radial.coefficients
        ||source_args->node_types==nullptr||source_args->neigh_indices==nullptr
        ||source_args->neigh_types==nullptr||source_args->source_offsets==nullptr
        ||source_args->source_edges==nullptr
        ||source_args->edge_receivers==nullptr||source_args->radius==nullptr
        ||source_args->output_adjoint==nullptr
        ||source_args->output_mask==nullptr
        ||source_args->node_density_adjoint==nullptr
        ||source_args->source_adjoint==nullptr||edge_args->xyz==nullptr
        ||edge_args->radius==nullptr||edge_args->radial.coefficients==nullptr
        ||edge_args->harmonics_values==nullptr
        ||edge_args->harmonics_gradients==nullptr
        ||edge_args->neighbor_features==nullptr
        ||edge_args->output_adjoint==nullptr
        ||edge_args->directed_forces==nullptr)
        throw std::invalid_argument(
            "Execution MH1 spline R reverse packet has invalid storage");
    validate_module_context(*module_state_);
    const auto& plan=module_state_->interactions[source_args->interaction];
    if(plan.spline_reverse==nullptr||plan.spline_reverse_threads<=0)
        throw std::logic_error("Execution MH1 spline R reverse is unavailable");
    const auto blocks=static_cast<unsigned int>(
        source_args->source_owner_count<persistent_blocks
            ?source_args->source_owner_count:persistent_blocks);
    launch_packet_pair(plan.spline_reverse,*source_args,*edge_args,stream,
        blocks,plan.spline_reverse_threads);
    return 0;
#else
    (void)source_args;(void)edge_args;(void)stream;(void)persistent_blocks;
    return 1;
#endif
}

std::int32_t MH1CudaPluginV4::launch_conditioning_forward(
    const SymmetrixJitMH1CudaConditioningForwardArgsV4* args, void* stream,
    std::int32_t persistent_blocks) const
{
    if(module_state_==nullptr) {
        if(args==nullptr||args->interaction>=2) return 1;
        return descriptor().conditioning_launches[args->interaction]
            .forward_launch(args,stream,persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    constexpr std::uint32_t known_flags =
        SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4;
    if(args==nullptr
        ||args->struct_size
            <sizeof(SymmetrixJitMH1CudaConditioningForwardArgsV4)
        ||args->interaction>=2||(args->flags&~known_flags)!=0u
        ||args->num_nodes<0||args->num_nodes>INT32_MAX
        ||args->num_edges<0||args->num_edges>INT32_MAX
        ||args->active_receiver_count>args->num_nodes
        ||args->active_receiver_count>args->num_edges
        ||(args->num_edges>0&&args->num_nodes==0)
        ||(args->num_edges>0&&args->active_receiver_count==0)
        ||persistent_blocks<=0)
        throw std::invalid_argument("Execution MH1 conditioning-forward packet is invalid");
    if(args->num_edges==0) return 0;
    const bool apply_cutoff=(args->flags
        &SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4)!=0;
    if(args->source_types==nullptr||args->target_indices==nullptr
        ||args->node_types==nullptr||args->active_receivers==nullptr
        ||args->receiver_offsets==nullptr||args->radial==nullptr
        ||(apply_cutoff&&args->edge_cutoff_scale==nullptr)
        ||args->edge_phi==nullptr||args->node_density==nullptr)
        throw std::invalid_argument(
            "Execution MH1 conditioning-forward packet has null storage");
    validate_module_context(*module_state_);
    const auto& plan=module_state_->interactions[args->interaction];
    const auto blocks=static_cast<unsigned int>(
        args->active_receiver_count<persistent_blocks
            ?args->active_receiver_count:persistent_blocks);
    launch_packet(plan.conditioning_forward,*args,stream,blocks,1,1,
        plan.edge_threads);
    return 0;
#else
    (void)args;(void)stream;(void)persistent_blocks;return 1;
#endif
}

std::int32_t MH1CudaPluginV4::launch_conditioning_reverse(
    const SymmetrixJitMH1CudaConditioningReverseArgsV4* args, void* stream,
    std::int32_t persistent_blocks) const
{
    if(module_state_==nullptr) {
        if(args==nullptr||args->interaction>=2) return 1;
        return descriptor().conditioning_launches[args->interaction]
            .reverse_launch(args,stream,persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    constexpr std::uint32_t known_flags =
        SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4;
    if(args==nullptr
        ||args->struct_size
            <sizeof(SymmetrixJitMH1CudaConditioningReverseArgsV4)
        ||args->interaction>=2||(args->flags&~known_flags)!=0u
        ||args->reserved!=0u||args->num_nodes<0||args->num_nodes>INT32_MAX
        ||args->num_edges<0||args->num_edges>INT32_MAX
        ||(args->num_edges>0&&args->num_nodes==0)||persistent_blocks<=0)
        throw std::invalid_argument("Execution MH1 conditioning-reverse packet is invalid");
    if(args->num_edges==0) return 0;
    const bool apply_cutoff=(args->flags
        &SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4)!=0;
    if(args->source_types==nullptr||args->target_indices==nullptr
        ||args->node_types==nullptr||args->radial==nullptr
        ||(apply_cutoff&&args->edge_cutoff_scale==nullptr)
        ||args->edge_phi_adjoint==nullptr||args->node_density_adjoint==nullptr
        ||args->radial_adjoint==nullptr||args->edge_cutoff_scale_adjoint==nullptr)
        throw std::invalid_argument(
            "Execution MH1 conditioning-reverse packet has null storage");
    validate_module_context(*module_state_);
    const auto& plan=module_state_->interactions[args->interaction];
    const auto blocks=launch_blocks(
        args->num_edges,plan.edge_threads,persistent_blocks);
    for(const auto function:plan.conditioning_reverse)
        launch_packet(function,*args,stream,blocks,1,1,plan.edge_threads);
    return 0;
#else
    (void)args;(void)stream;(void)persistent_blocks;return 1;
#endif
}

std::int32_t MH1CudaPluginV4::launch_node_forward(
    const SymmetrixJitMH1CudaNodeForwardArgsV4* args, void* stream,
    std::int32_t persistent_blocks) const
{
    if(module_state_==nullptr) {
        if(args==nullptr||args->struct_size<sizeof(*args)
            ||args->interaction>=2||args->phase>1) return 1;
        if(args->num_nodes>0
            &&args->phase==SYMMETRIX_JIT_MH1_CUDA_NODE_POST_FORWARD_V4)
            validate_retained_node_state(
                descriptor().node_programs[args->interaction],
                args->retained_pre_gate,args->retained_interaction_output);
        return descriptor().node_programs[args->interaction]
            .forward_phases[args->phase].launch(args,stream,persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if(args==nullptr
        ||args->struct_size<sizeof(SymmetrixJitMH1CudaNodeForwardArgsV4)
        ||args->interaction>=2||args->phase>1
        ||args->num_nodes<0||args->num_nodes>INT32_MAX
        ||persistent_blocks<=0)
        throw std::invalid_argument("Execution MH1 node-forward packet is invalid");
    if(args->num_nodes==0) return 0;
    if(args->phase==SYMMETRIX_JIT_MH1_CUDA_NODE_POST_FORWARD_V4)
        validate_retained_node_state(
            descriptor().node_programs[args->interaction],
            args->retained_pre_gate,args->retained_interaction_output);
    const bool first_layer=args->interaction==0;
    if(args->phase==SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_FORWARD_V4) {
        if(args->element_indices==nullptr||args->linear_parameters==nullptr
            ||args->up_output==nullptr||args->node_arena==nullptr
            ||(!first_layer&&args->layer_input==nullptr))
            throw std::invalid_argument(
                "Execution MH1 node pre-forward packet has null storage");
    } else if(args->element_indices==nullptr||args->node_density==nullptr
        ||args->linear_parameters==nullptr||args->product_parameters==nullptr
        ||args->readout_parameters==nullptr||args->up==nullptr
        ||args->messages==nullptr||args->layer_output==nullptr
        ||args->readout_contribution==nullptr||args->node_arena==nullptr
        ||(!first_layer&&args->layer_input==nullptr)) {
        throw std::invalid_argument(
            "Execution MH1 node post-forward packet has null storage");
    }
    validate_module_context(*module_state_);
    const auto& node=module_state_->nodes[args->interaction];
    const auto& launches=args->phase==0?node.pre_forward:node.post_forward;
    launch_node_sequence(*module_state_,launches,*args,stream,persistent_blocks);
    return 0;
#else
    (void)args;(void)stream;(void)persistent_blocks;return 1;
#endif
}

std::int32_t MH1CudaPluginV4::launch_node_reverse(
    const SymmetrixJitMH1CudaNodeReverseArgsV4* args, void* stream,
    std::int32_t persistent_blocks) const
{
    if(module_state_==nullptr) {
        if(args==nullptr||args->struct_size<sizeof(*args)
            ||args->interaction>=2||args->phase<2||args->phase>3)
            return 1;
        if(args->num_nodes>0
            &&args->phase==SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4)
            validate_retained_node_state(
                descriptor().node_programs[args->interaction],
                args->retained_pre_gate,args->retained_interaction_output);
        return descriptor().node_programs[args->interaction]
            .reverse_phases[args->phase-2].launch(
                args,stream,persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if(args==nullptr
        ||args->struct_size<sizeof(SymmetrixJitMH1CudaNodeReverseArgsV4)
        ||args->interaction>=2||args->phase<2||args->phase>3
        ||args->reserved!=0u||args->num_nodes<0||args->num_nodes>INT32_MAX
        ||persistent_blocks<=0)
        throw std::invalid_argument("Execution MH1 node-reverse packet is invalid");
    if(args->num_nodes==0) return 0;
    if(args->phase==SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4)
        validate_retained_node_state(
            descriptor().node_programs[args->interaction],
            args->retained_pre_gate,args->retained_interaction_output);
    const bool first_layer=args->interaction==0;
    if(args->phase==SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4) {
        if(args->element_indices==nullptr||args->node_density==nullptr
            ||args->linear_parameters==nullptr||args->product_parameters==nullptr
            ||args->readout_parameters==nullptr||args->up==nullptr
            ||args->messages==nullptr||args->layer_output==nullptr
            ||args->node_arena==nullptr||args->layer_output_adjoint==nullptr
            ||args->message_adjoint==nullptr||args->up_adjoint==nullptr
            ||args->node_density_adjoint==nullptr
            ||(!first_layer&&(args->layer_input==nullptr
                ||args->layer_input_adjoint==nullptr)))
            throw std::invalid_argument(
                "Execution MH1 node post-reverse packet has null storage");
    } else if(args->linear_parameters==nullptr||args->up_adjoint==nullptr
        ||args->node_arena==nullptr||args->layer_input_adjoint==nullptr) {
        throw std::invalid_argument(
            "Execution MH1 node pre-reverse packet has null storage");
    }
    validate_module_context(*module_state_);
    const auto& node=module_state_->nodes[args->interaction];
    const auto& launches=args->phase==2?node.post_reverse:node.pre_reverse;
    if(launches.empty())
        throw std::invalid_argument("Execution MH1 node reverse phase is unavailable");
    launch_node_sequence(*module_state_,launches,*args,stream,persistent_blocks);
    return 0;
#else
    (void)args;(void)stream;(void)persistent_blocks;return 1;
#endif
}

void MH1CudaPluginV4::reset() noexcept
{
    descriptor_=nullptr;
    if(module_state_!=nullptr) {
#if defined(KOKKOS_ENABLE_CUDA)
        if(module_state_->module!=nullptr)
            static_cast<void>(cuModuleUnload(
                reinterpret_cast<CUmodule>(module_state_->module)));
#elif defined(KOKKOS_ENABLE_HIP)
        if(module_state_->module!=nullptr)
            static_cast<void>(hipModuleUnload(
                reinterpret_cast<hipModule_t>(module_state_->module)));
#endif
        module_state_.reset();
    }
    if(handle_!=nullptr) {
        dlclose(handle_);
        handle_=nullptr;
    }
    path_.clear();
}

}  // namespace symmetrix::execution
#else
namespace symmetrix::execution {

struct MH1CudaModuleState {};

MH1CudaPlugin MH1CudaPlugin::load(
    std::string,
    const MH1CudaPluginExpectation&)
{
    throw std::runtime_error(
        "Execution MH-1 CUDA plugins require POSIX dlopen/dlsym support.");
}

MH1CudaPlugin::MH1CudaPlugin(
    void* handle,
    const SymmetrixJitMH1CudaPluginV3* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1CudaPlugin::~MH1CudaPlugin() { reset(); }

MH1CudaPlugin::MH1CudaPlugin(MH1CudaPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

MH1CudaPlugin& MH1CudaPlugin::operator=(MH1CudaPlugin&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1CudaPluginV3& MH1CudaPlugin::descriptor() const
{
    throw std::logic_error("Execution MH-1 CUDA plugin support is disabled.");
}

void MH1CudaPlugin::reset() noexcept
{
    handle_=nullptr;
    descriptor_=nullptr;
    path_.clear();
}

MH1CudaPluginV4 MH1CudaPluginV4::load(
    std::string,
    const MH1CudaPluginV4Expectation&)
{
    throw std::runtime_error(
        "Execution MH-1 CUDA ABI-v4 plugins require POSIX dlopen/dlsym support.");
}

MH1CudaPluginV4 MH1CudaPluginV4::load_cubin(
    std::string,
    const MH1CudaPluginV4Expectation&,
    std::string)
{
    throw std::runtime_error(
        "Execution MH-1 CUDA modules require POSIX and CUDA support.");
}

MH1CudaPluginV4 MH1CudaPluginV4::load_module(
    std::string,
    const MH1CudaPluginV4Expectation&,
    std::string)
{
    throw std::runtime_error(
        "Execution MH-1 device modules require POSIX and GPU support.");
}

MH1CudaPluginV4::MH1CudaPluginV4() = default;

MH1CudaPluginV4::MH1CudaPluginV4(
    void* handle,
    const SymmetrixJitMH1CudaPluginV4* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1CudaPluginV4::~MH1CudaPluginV4()
{
    reset();
}

MH1CudaPluginV4::MH1CudaPluginV4(MH1CudaPluginV4&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      module_state_(std::move(other.module_state_)),
      path_(std::move(other.path_))
{}

MH1CudaPluginV4& MH1CudaPluginV4::operator=(MH1CudaPluginV4&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        module_state_=std::move(other.module_state_);
        path_=std::move(other.path_);
    }
    return *this;
}

std::int32_t MH1CudaPluginV4::launch_forward(
    const SymmetrixJitMH1CudaForwardArgsV4*,void*,std::int32_t) const
{ return 1; }

std::int32_t MH1CudaPluginV4::launch_reverse(
    const SymmetrixJitMH1CudaReverseArgsV4*,void*,std::int32_t,
    std::int32_t) const
{ return 1; }

bool MH1CudaPluginV4::has_spline_r_program() const noexcept
{ return false; }

std::int32_t MH1CudaPluginV4::launch_spline_r_forward(
    const SymmetrixJitMH1CudaSplineRForwardArgsV5*,void*,std::int32_t) const
{ return 1; }

std::int32_t MH1CudaPluginV4::launch_spline_r_reverse(
    const SymmetrixJitMH1CudaSplineRSourceArgsV5*,
    const SymmetrixJitMH1CudaSplineREdgeArgsV5*,void*,std::int32_t) const
{ return 1; }

std::int32_t MH1CudaPluginV4::launch_conditioning_forward(
    const SymmetrixJitMH1CudaConditioningForwardArgsV4*,void*,
    std::int32_t) const
{ return 1; }

std::int32_t MH1CudaPluginV4::launch_conditioning_reverse(
    const SymmetrixJitMH1CudaConditioningReverseArgsV4*,void*,
    std::int32_t) const
{ return 1; }

std::int32_t MH1CudaPluginV4::launch_node_forward(
    const SymmetrixJitMH1CudaNodeForwardArgsV4*,void*,std::int32_t) const
{ return 1; }

std::int32_t MH1CudaPluginV4::launch_node_reverse(
    const SymmetrixJitMH1CudaNodeReverseArgsV4*,void*,std::int32_t) const
{ return 1; }

const SymmetrixJitMH1CudaPluginV4& MH1CudaPluginV4::descriptor() const
{
    throw std::logic_error(
        "Execution MH-1 CUDA ABI-v4 plugin support is disabled.");
}

void MH1CudaPluginV4::reset() noexcept
{
    handle_=nullptr;
    descriptor_=nullptr;
    module_state_.reset();
    path_.clear();
}

}  // namespace symmetrix::execution
#endif
