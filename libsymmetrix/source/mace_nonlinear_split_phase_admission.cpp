#include "mace_nonlinear_split_phase_admission.hpp"

#include <stdexcept>
#include <string>

namespace symmetrix::execution {
namespace {

void require_match(
    const char* actual, const std::string_view expected,
    const std::string_view field)
{
    if(actual==nullptr||actual[0]=='\0'||actual!=expected)
        throw std::invalid_argument(
            "Execution MH-1 split-phase "+std::string(field)+" does not match.");
}

void require_common(
    const std::uint32_t pointer_size,const std::uint32_t byte_order,
    const std::uint32_t capabilities,const std::uint32_t interaction_count,
    const std::uint32_t scalar_size,const char* artifact_id,
    const char* generation_fingerprint,const char* semantic_fingerprint,
    const char* structure_fingerprint,const char* runtime_layout_fingerprint,
    const MH1SplitPhaseExpectation& expected,const std::uint32_t product_capability)
{
    if(pointer_size!=sizeof(void*))
        throw std::invalid_argument(
            "Execution MH-1 split-phase pointer size does not match.");
    if(byte_order!=SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER)
        throw std::invalid_argument(
            "Execution MH-1 split-phase byte order does not match.");
    if((capabilities&product_capability)==0)
        throw std::invalid_argument(
            "Execution MH-1 split-phase external-product capability is missing.");
    if(interaction_count!=expected.interaction_count)
        throw std::invalid_argument(
            "Execution MH-1 split-phase interaction dimensions do not match.");
    if(scalar_size!=expected.scalar_size)
        throw std::invalid_argument(
            "Execution MH-1 split-phase precision does not match.");
    require_match(artifact_id,expected.artifact_id,"artifact identity");
    require_match(
        generation_fingerprint,expected.generation_fingerprint,
        "generation identity");
    require_match(
        semantic_fingerprint,expected.semantic_fingerprint,"semantic identity");
    require_match(
        structure_fingerprint,expected.structure_fingerprint,"structure identity");
    require_match(
        runtime_layout_fingerprint,expected.runtime_layout_fingerprint,
        "runtime-layout identity");
}

void require_module_common(
    const MH1FullModuleDescriptor& value,
    const MH1SplitPhaseExpectation& expected)
{
    if(value.interaction_count!=expected.interaction_count)
        throw std::invalid_argument(
            "Execution MH-1 full-module interaction dimensions do not match.");
    if(value.scalar_size!=expected.scalar_size)
        throw std::invalid_argument(
            "Execution MH-1 full-module precision does not match.");
    if(value.artifact_id!=expected.artifact_id
        ||value.generation_fingerprint!=expected.generation_fingerprint
        ||value.semantic_fingerprint!=expected.semantic_fingerprint
        ||value.structure_fingerprint!=expected.structure_fingerprint
        ||value.runtime_layout_fingerprint!=expected.runtime_layout_fingerprint)
        throw std::invalid_argument(
            "Execution MH-1 full-module identity does not match.");
    if(value.target_backend!=expected.target_backend)
        throw std::invalid_argument(
            "Execution MH-1 full-module backend does not match.");
    if(value.target_architecture!=expected.target_architecture)
        throw std::invalid_argument(
            "Execution MH-1 full-module target does not match.");
}

}  // namespace

void validate_mh1_split_phase_host_descriptor(
    const SymmetrixJitMH1HostPluginV7& value,
    const MH1SplitPhaseExpectation& expected)
{
    if(value.abi_version!=SYMMETRIX_JIT_MH1_HOST_PLUGIN_V7_ABI_VERSION
        ||value.struct_size!=sizeof(SymmetrixJitMH1HostPluginV7))
        throw std::invalid_argument(
            "Execution MH-1 host split-phase ABI is retired or malformed.");
    require_match(
        value.abi_tag,SYMMETRIX_JIT_MH1_HOST_PLUGIN_V7_ABI_TAG,"host ABI tag");
    require_common(
        value.pointer_size,value.byte_order,value.capabilities,
        value.interaction_count,value.scalar_size,value.artifact_id,
        value.generation_fingerprint,value.semantic_fingerprint,
        value.structure_fingerprint,value.runtime_layout_fingerprint,
        expected,SYMMETRIX_JIT_MH1_HOST_EXTERNAL_PRODUCT_V7);
    if(value.node_programs==nullptr)
        throw std::invalid_argument(
            "Execution MH-1 host split-phase node programs are missing.");
}

void validate_mh1_split_phase_cuda_descriptor(
    const SymmetrixJitMH1CudaPluginV5& value,
    const MH1SplitPhaseExpectation& expected)
{
    if(value.abi_version!=SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V5_ABI_VERSION
        ||value.struct_size!=sizeof(SymmetrixJitMH1CudaPluginV5))
        throw std::invalid_argument(
            "Execution MH-1 CUDA split-phase ABI is retired or malformed.");
    require_match(
        value.abi_tag,SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V5_ABI_TAG,"CUDA ABI tag");
    require_common(
        value.pointer_size,value.byte_order,value.capabilities,
        value.interaction_count,value.scalar_size,value.artifact_id,
        value.generation_fingerprint,value.semantic_fingerprint,
        value.structure_fingerprint,value.runtime_layout_fingerprint,
        expected,SYMMETRIX_JIT_MH1_CUDA_EXTERNAL_PRODUCT_V5);
    const auto required_capabilities=
        SYMMETRIX_JIT_MH1_CUDA_EXTERNAL_PRODUCT_V5
        |SYMMETRIX_JIT_MH1_CUDA_NODE_PROGRAM_METADATA_V5;
    if(value.capabilities!=required_capabilities)
        throw std::invalid_argument(
            "Execution MH-1 CUDA split-phase capabilities are not the exact "
            "product-free V5 set.");
    require_match(value.target_backend,expected.target_backend,"backend");
    require_match(value.target_architecture,expected.target_architecture,"target");
    if(value.node_programs==nullptr)
        throw std::invalid_argument(
            "Execution MH-1 CUDA split-phase node programs are missing.");
    for(std::uint32_t interaction=0;interaction<value.interaction_count;
        ++interaction) {
        const auto& node=value.node_programs[interaction];
        const auto known_flags=
            SYMMETRIX_JIT_MH1_CUDA_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V5
            |SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_PRE_GATE_V5
            |SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_INTERACTION_OUTPUT_V5
            |SYMMETRIX_JIT_MH1_CUDA_NODE_REUSE_MESSAGE_ADJOINT_V5;
        if(node.struct_size!=sizeof(SymmetrixJitMH1CudaNodeProgramV5)
            ||node.interaction!=interaction||node.element_count<=0
            ||node.input_dimension<=0||node.up_dimension<=0
            ||node.contraction_input_dimension<=0
            ||node.contraction_output_dimension<=0
            ||node.output_dimension<=0||node.node_arena_dimension<=0
            ||node.linear_parameter_count<=0||node.readout_parameter_count<=0
            ||(node.flags&~known_flags)!=0)
            throw std::invalid_argument(
                "Execution MH-1 CUDA split-phase node metadata is malformed.");
    }
}

void validate_mh1_full_module_descriptor(
    const MH1FullModuleDescriptor& value,
    const MH1SplitPhaseExpectation& expected)
{
    if(value.abi_version!=5u)
        throw std::invalid_argument(
            "Execution MH-1 full-module ABI is retired or malformed.");
    const std::string_view expected_tag=expected.target_backend=="cuda"
        ?"symmetrix.jit.mh1.cuda-module/5"
        :"symmetrix.jit.mh1.hip-module/5";
    if(value.abi_tag!=expected_tag)
        throw std::invalid_argument(
            "Execution MH-1 full-module ABI tag does not match.");
    require_module_common(value,expected);
}

void validate_mh1_product_module_candidate(
    const MH1ProductModuleCandidate& value,
    const MH1ProductModuleExpectation& expected)
{
    if(expected.product_count<=0||value.layer<0
        ||value.layer>=expected.product_count||value.layer!=expected.layer)
        throw std::invalid_argument(
            "Execution MH-1 product layer does not match.");
    if((value.kind!="M0"&&value.kind!="M1")||value.kind!=expected.kind)
        throw std::invalid_argument(
            "Execution MH-1 product operator kind does not match.");
    if(value.backend!=expected.backend)
        throw std::invalid_argument(
            "Execution MH-1 product backend does not match.");
    if(value.target.empty()||value.target!=expected.target)
        throw std::invalid_argument(
            "Execution MH-1 product target does not match.");
    if(value.artifact_id.empty()||value.artifact_id!=expected.artifact_id
        ||value.structure_fingerprint.empty()
        ||value.structure_fingerprint!=expected.structure_fingerprint)
        throw std::invalid_argument(
            "Execution MH-1 product identity does not match.");
    if(value.scalar_size!=expected.scalar_size)
        throw std::invalid_argument(
            "Execution MH-1 product precision does not match.");
    if(value.channels!=expected.channels
        ||value.input_components!=expected.input_components
        ||value.output_components!=expected.output_components
        ||value.correlation!=expected.correlation
        ||value.term_count!=expected.term_count)
        throw std::invalid_argument(
            "Execution MH-1 product topology dimensions do not match.");
}

void validate_mh1_launch_plan_schema(
    const std::string_view schema,const std::string_view backend)
{
    if(backend!="cuda"&&backend!="hip")
        throw std::invalid_argument(
            "Execution MH-1 launch-plan backend is unsupported.");
    const std::string expected="symmetrix.jit.mh1."+std::string(backend)
        +"-launch-plan/6";
    if(schema!=expected)
        throw std::invalid_argument(
            "Execution MH-1 launch-plan schema is retired or malformed.");
}

}  // namespace symmetrix::execution
