#include "jit_mh1_host_plugin.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_standard_layout_v<SymmetrixJitMH1HostInteractionV3>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1HostForwardArgsV3>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1HostReverseArgsV3>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostConditioningForwardArgsV3>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostConditioningReverseArgsV3>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostConditioningOwnersV3>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1HostPluginV3>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostNodeForwardArgsV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostNodeReverseArgsV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostConditioningOwnersV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostNodeForwardPhaseV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostNodeReversePhaseV4>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1HostNodeProgramV4>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1HostPluginV4>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostSplineRForwardArgsV5>);
static_assert(std::is_standard_layout_v<
    SymmetrixJitMH1HostSplineRReverseArgsV5>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1HostSplineRProgramV5>);
static_assert(std::is_standard_layout_v<SymmetrixJitMH1HostPluginV5>);

#if defined(__unix__) || defined(__APPLE__)
namespace symmetrix::execution {
namespace {

std::runtime_error plugin_error(
    const std::string& path,
    const std::string& message)
{
    return std::runtime_error(
        "Could not load Execution MH-1 host plugin '"+path+"': "+message);
}

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

void validate_capability(
    const std::string& path,
    const std::uint32_t capabilities,
    const std::uint32_t capability,
    const bool has_function,
    const char* name)
{
    const bool declared=(capabilities&capability)!=0;
    if(declared!=has_function)
        throw plugin_error(
            path,std::string(name)+" capability and function pointer disagree");
}

constexpr std::uint32_t known_capabilities=
    SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V3
    |SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V3
    |SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V3
    |SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V3
    |SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V3
    |SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V3;

constexpr std::uint32_t mandatory_node_capabilities_v4=
    SYMMETRIX_JIT_MH1_HOST_NODE_PROGRAM_PHASES_V4
    |SYMMETRIX_JIT_MH1_HOST_PERSISTENT_IR_MUL_NODE_STATE_V4
    |SYMMETRIX_JIT_MH1_HOST_FIXED_WEIGHT_COORDINATES_V4;

constexpr std::uint32_t known_capabilities_v4=
    SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V4
    |SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V4
    |SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V4
    |SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V4
    |SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V4
    |SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V4
    |mandatory_node_capabilities_v4
    |SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_V4
    |SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_V4
    |SYMMETRIX_JIT_MH1_HOST_SOURCE_OWNS_EDGE_REVERSE_V4
    |SYMMETRIX_JIT_MH1_HOST_PHI_MAJOR_FORWARD_WEIGHT_V4;

constexpr std::uint32_t known_capabilities_v5=
    known_capabilities_v4|SYMMETRIX_JIT_MH1_HOST_SPLINE_R_FORWARD_V5
    |SYMMETRIX_JIT_MH1_HOST_SPLINE_R_SOURCE_REVERSE_V5;

void validate_interaction_expectation(
    const MH1HostInteractionExpectation& interaction)
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
            "Execution MH-1 host plugin expectation requires exact positive "
            "interaction extents.");
}

void validate_expectation(const MH1HostPluginExpectation& expectation)
{
    if(expectation.artifact_id.empty())
        throw std::invalid_argument(
            "Execution MH-1 host plugin expectation requires an artifact id.");
    if(expectation.generation_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 host plugin expectation requires a generation "
            "fingerprint.");
    if(expectation.semantic_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 host plugin expectation requires a semantic "
            "fingerprint.");
    if(expectation.structure_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 host plugin expectation requires a structure "
            "fingerprint.");
    for(const auto& interaction:expectation.interactions)
        validate_interaction_expectation(interaction);
    if(expectation.required_capabilities==0
        ||(expectation.required_capabilities&~known_capabilities)!=0)
        throw std::invalid_argument(
            "Execution MH-1 host plugin expectation requires exact known "
            "capabilities.");
}

void validate_node_expectation(
    const MH1HostNodeProgramExpectation& node)
{
    if(node.element_count<=0||node.input_dimension<=0
        ||node.up_dimension<=0||node.residual_dimension<=0
        ||node.skip_dimension<=0||node.message_dimension<=0
        ||node.interaction_output_dimension<=0||node.output_dimension<=0
        ||node.product_term_count<=0||node.node_arena_dimension<0
        ||node.linear_parameter_count<=0||node.product_parameter_count<=0
        ||node.readout_parameter_count<=0)
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v4 expectation requires exact node-program "
            "extents and runtime parameter counts.");
    for(const auto owners:node.forward_owners_per_node)
        if(owners<=0)
            throw std::invalid_argument(
                "Execution MH-1 host ABI-v4 expectation requires positive "
                "forward phase owner counts.");
    for(std::size_t phase=0;phase<node.reverse_owners_per_node.size();++phase) {
        const auto owners=node.reverse_owners_per_node[phase];
        const bool disabled=phase==1&&!node.requires_tp_source_state_adjoint;
        if((disabled&&owners!=0)||(!disabled&&owners<=0))
            throw std::invalid_argument(
                "Execution MH-1 host ABI-v4 expectation has an invalid reverse "
                "phase owner count.");
    }
}

void validate_expectation(const MH1HostPluginV4Expectation& expectation)
{
    if(expectation.artifact_id.empty())
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v4 expectation requires an artifact id.");
    if(expectation.generation_fingerprint.empty()
        ||expectation.semantic_fingerprint.empty()
        ||expectation.structure_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v4 expectation requires exact "
            "fingerprints.");
    for(const auto& interaction:expectation.interactions)
        validate_interaction_expectation(interaction);
    for(std::size_t interaction=0;
        interaction<expectation.node_programs.size();++interaction) {
        const auto& node=expectation.node_programs[interaction];
        validate_node_expectation(node);
        if(node.requires_tp_source_state_adjoint!=(interaction!=0))
            throw std::invalid_argument(
                "Execution MH-1 host ABI-v4 expectation must omit the layer-0 "
                "TP source-state adjoint and require it for layer 1.");
    }
    if(expectation.runtime_layout_fingerprint.empty())
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v4 expectation requires a runtime layout "
            "fingerprint.");
    if((expectation.required_capabilities&~known_capabilities_v4)!=0
        ||(expectation.required_capabilities
            &mandatory_node_capabilities_v4)!=mandatory_node_capabilities_v4)
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v4 expectation requires the complete "
            "generated interaction, node-program, persistent-ir_mul, and "
            "fixed-weight-coordinate capability set.");
}

void validate_interaction(
    const std::string& path,
    const SymmetrixJitMH1HostInteractionV3& actual,
    const MH1HostInteractionExpectation& expected,
    const std::size_t interaction)
{
    if(actual.struct_size<sizeof(SymmetrixJitMH1HostInteractionV3))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" descriptor is smaller than ABI version 3");
    if(actual.reserved!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" reserved field is nonzero");
    if(actual.input_1_dimension<=0||actual.input_2_dimension<=0
        ||actual.output_dimension<=0||actual.weight_size<=0
        ||actual.phi_dimension<=0||actual.multiplicity<=0
        ||actual.input_1_angular_dimension<=0
        ||actual.instruction_count<=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" extents are invalid");

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

void validate_conditioning(
    const std::string& path,
    const SymmetrixJitMH1HostConditioningOwnersV3& owners,
    const std::uint32_t capabilities,
    const std::size_t interaction)
{
    if(owners.struct_size<sizeof(SymmetrixJitMH1HostConditioningOwnersV3))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning owner descriptor is smaller than ABI "
                "version 3");
    if(owners.interaction!=interaction)
        throw plugin_error(
            path,"conditioning owner descriptor index does not match");
    if(owners.reserved!=0||owners.reserved_2!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning owner descriptor reserved field is nonzero");
    const bool declared=(capabilities
        &SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V3)!=0;
    if(declared!=(owners.forward_owner!=nullptr)
        ||declared!=(owners.reverse_owner!=nullptr))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning capability and function pointers disagree");
}

void validate_descriptor(
    const std::string& path,
    const SymmetrixJitMH1HostPluginV3& descriptor,
    const MH1HostPluginExpectation& expectation)
{
    if(descriptor.abi_version!=SYMMETRIX_JIT_MH1_HOST_PLUGIN_ABI_VERSION)
        throw plugin_error(path,"unsupported ABI version");
    if(descriptor.struct_size<sizeof(SymmetrixJitMH1HostPluginV3))
        throw plugin_error(path,"descriptor is smaller than ABI version 3");
    if(descriptor.pointer_size!=sizeof(void*))
        throw plugin_error(path,"pointer width does not match this process");
    if(descriptor.byte_order!=SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER)
        throw plugin_error(path,"byte order does not match this process");
    if(descriptor.interaction_count
        !=SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS)
        throw plugin_error(path,"descriptor must contain exactly two interactions");
    if(descriptor.scalar_size!=sizeof(float))
        throw plugin_error(path,"only float32 MH-1 host plugins are supported");
    if(descriptor.reserved!=0)
        throw plugin_error(path,"reserved descriptor field is nonzero");

    const auto abi_tag=require_text(path,descriptor.abi_tag,"ABI tag");
    if(abi_tag!=SYMMETRIX_JIT_MH1_HOST_PLUGIN_ABI_TAG)
        throw plugin_error(path,"ABI tag does not match");
    const auto artifact_id=
        require_text(path,descriptor.artifact_id,"artifact id");
    const auto generation_fingerprint=require_text(
        path,descriptor.generation_fingerprint,"generation fingerprint");
    const auto semantic_fingerprint=require_text(
        path,descriptor.semantic_fingerprint,"semantic fingerprint");
    const auto structure_fingerprint=require_text(
        path,descriptor.structure_fingerprint,"structure fingerprint");

    if(descriptor.capabilities!=expectation.required_capabilities)
        throw plugin_error(path,"owner function capabilities do not match");
    validate_capability(
        path,descriptor.capabilities,
        SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V3,
        descriptor.forward_owner!=nullptr,"forward owner");
    validate_capability(
        path,descriptor.capabilities,
        SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V3,
        descriptor.source_reverse_owner!=nullptr,"source reverse owner");
    validate_capability(
        path,descriptor.capabilities,
        SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V3,
        descriptor.edge_reverse_owner!=nullptr,"edge reverse owner");

    require_match(path,expectation.artifact_id,
        artifact_id.c_str(),"artifact id");
    require_match(path,expectation.generation_fingerprint,
        generation_fingerprint.c_str(),"generation fingerprint");
    require_match(path,expectation.semantic_fingerprint,
        semantic_fingerprint.c_str(),"semantic fingerprint");
    require_match(path,expectation.structure_fingerprint,
        structure_fingerprint.c_str(),"structure fingerprint");
    for(std::size_t interaction=0;
        interaction<SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS;
        ++interaction) {
        validate_interaction(
            path,descriptor.interactions[interaction],
            expectation.interactions[interaction],interaction);
        validate_conditioning(
            path,descriptor.conditioning[interaction],
            descriptor.capabilities,interaction);
    }
}

void validate_conditioning_v4(
    const std::string& path,
    const SymmetrixJitMH1HostConditioningOwnersV4& owners,
    const std::uint32_t capabilities,
    const std::size_t interaction)
{
    if(owners.struct_size<sizeof(SymmetrixJitMH1HostConditioningOwnersV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning owner descriptor is smaller than ABI "
                "version 4");
    if(owners.interaction!=interaction)
        throw plugin_error(
            path,"conditioning owner descriptor index does not match");
    if(owners.reserved!=0||owners.reserved_2!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning owner descriptor reserved field is nonzero");
    const bool declared=(capabilities
        &SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V4)!=0;
    if(declared!=(owners.forward_owner!=nullptr)
        ||declared!=(owners.reverse_owner!=nullptr))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" conditioning capability and function pointers disagree");
}

void validate_node_forward_phase(
    const std::string& path,
    const SymmetrixJitMH1HostNodeForwardPhaseV4& phase,
    const MH1HostNodeProgramExpectation& expectation,
    const std::size_t interaction,
    const std::size_t index)
{
    if(phase.struct_size<sizeof(SymmetrixJitMH1HostNodeForwardPhaseV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node forward phase descriptor is smaller than ABI "
                "version 4");
    if(phase.interaction!=interaction
        ||phase.phase!=SYMMETRIX_JIT_MH1_HOST_NODE_PRE_FORWARD_V4+index)
        throw plugin_error(
            path,"node forward phase descriptor index does not match");
    if(phase.reserved!=0||phase.reserved_2!=0||phase.reserved_3!=0
        ||phase.reserved_4!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node forward phase reserved field is nonzero");
    if(phase.owners_per_node<=0||phase.owner==nullptr)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node forward phase owner is invalid");
    require_extent(
        path,expectation.forward_owners_per_node[index],
        phase.owners_per_node,"node forward owners per node",interaction);
}

void validate_node_reverse_phase(
    const std::string& path,
    const SymmetrixJitMH1HostNodeReversePhaseV4& phase,
    const MH1HostNodeProgramExpectation& expectation,
    const std::size_t interaction,
    const std::size_t index)
{
    if(phase.struct_size<sizeof(SymmetrixJitMH1HostNodeReversePhaseV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node reverse phase descriptor is smaller than ABI "
                "version 4");
    if(phase.interaction!=interaction
        ||phase.phase!=SYMMETRIX_JIT_MH1_HOST_NODE_POST_REVERSE_V4+index)
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
        if(phase.owners_per_node!=0||phase.owner!=nullptr)
            throw plugin_error(
                path,"interaction "+std::to_string(interaction)
                    +" disabled node reverse phase is not empty");
        return;
    }
    if(phase.owners_per_node<=0||phase.owner==nullptr)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node reverse phase owner is invalid");
    require_extent(
        path,expectation.reverse_owners_per_node[index],
        phase.owners_per_node,"node reverse owners per node",interaction);
}

void validate_node_program(
    const std::string& path,
    const SymmetrixJitMH1HostNodeProgramV4& node,
    const MH1HostNodeProgramExpectation& expectation,
    const std::uint32_t capabilities,
    const std::size_t interaction)
{
    if((capabilities&mandatory_node_capabilities_v4)
        !=mandatory_node_capabilities_v4)
        throw plugin_error(
            path,"ABI-v4 node capabilities are incomplete");
    if(node.struct_size<sizeof(SymmetrixJitMH1HostNodeProgramV4))
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node-program descriptor is smaller than ABI version 4");
    if(node.interaction!=interaction)
        throw plugin_error(path,"node-program descriptor index does not match");
    if(node.forward_phase_count
            !=SYMMETRIX_JIT_MH1_HOST_NODE_FORWARD_PHASES_V4
        ||node.reverse_phase_count
            !=SYMMETRIX_JIT_MH1_HOST_NODE_REVERSE_PHASES_V4)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node-program phase count is invalid");
    const std::uint32_t known_flags=
        SYMMETRIX_JIT_MH1_HOST_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4
        |SYMMETRIX_JIT_MH1_HOST_NODE_RETAIN_INTERACTION_OUTPUT_V4;
    const std::uint32_t expected_flags=
        expectation.requires_tp_source_state_adjoint
            ?SYMMETRIX_JIT_MH1_HOST_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4:0u;
    if((node.flags&~known_flags)!=0
        ||(node.flags
            &SYMMETRIX_JIT_MH1_HOST_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4)
            !=expected_flags)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" TP source-state adjoint requirement does not match");
    if(node.reserved_2!=0||node.reserved_3!=0)
        throw plugin_error(
            path,"interaction "+std::to_string(interaction)
                +" node-program reserved field is nonzero");
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
        phase<SYMMETRIX_JIT_MH1_HOST_NODE_FORWARD_PHASES_V4;++phase)
        validate_node_forward_phase(
            path,node.forward_phases[phase],expectation,interaction,phase);
    for(std::size_t phase=0;
        phase<SYMMETRIX_JIT_MH1_HOST_NODE_REVERSE_PHASES_V4;++phase)
        validate_node_reverse_phase(
            path,node.reverse_phases[phase],expectation,interaction,phase);
}

void validate_descriptor(
    const std::string& path,
    const SymmetrixJitMH1HostPluginV4& descriptor,
    const MH1HostPluginV4Expectation& expectation)
{
    if(descriptor.abi_version
        !=SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_VERSION)
        throw plugin_error(path,"unsupported ABI version");
    if(descriptor.struct_size<sizeof(SymmetrixJitMH1HostPluginV4))
        throw plugin_error(path,"descriptor is smaller than ABI version 4");
    if(descriptor.pointer_size!=sizeof(void*))
        throw plugin_error(path,"pointer width does not match this process");
    if(descriptor.byte_order!=SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER)
        throw plugin_error(path,"byte order does not match this process");
    if(descriptor.interaction_count
        !=SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS)
        throw plugin_error(path,"descriptor must contain exactly two interactions");
    if(descriptor.scalar_size!=sizeof(float))
        throw plugin_error(path,"only float32 MH-1 host plugins are supported");
    if(descriptor.reserved!=0)
        throw plugin_error(path,"reserved descriptor field is nonzero");

    const auto abi_tag=require_text(path,descriptor.abi_tag,"ABI tag");
    if(abi_tag!=SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_TAG)
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
    if((descriptor.capabilities&expectation.required_capabilities)
            !=expectation.required_capabilities
        ||(descriptor.capabilities&~known_capabilities_v4)!=0)
        throw plugin_error(path,"ABI-v4 owner capabilities do not match");
    validate_capability(path,descriptor.capabilities,
        SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V4,
        descriptor.forward_owner!=nullptr,"forward owner");
    validate_capability(path,descriptor.capabilities,
        SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V4,
        descriptor.source_reverse_owner!=nullptr,"source reverse owner");
    validate_capability(path,descriptor.capabilities,
        SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V4,
        descriptor.edge_reverse_owner!=nullptr,"edge reverse owner");

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
    for(std::size_t interaction=0;
        interaction<SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS;
        ++interaction) {
        validate_interaction(path,descriptor.interactions[interaction],
            expectation.interactions[interaction],interaction);
        validate_conditioning_v4(path,descriptor.conditioning[interaction],
            descriptor.capabilities,interaction);
        validate_node_program(path,descriptor.node_programs[interaction],
            expectation.node_programs[interaction],descriptor.capabilities,
            interaction);
    }
}

void validate_expectation(const MH1HostPluginV5Expectation& expectation)
{
    validate_expectation(expectation.v4);
    if(expectation.scalar_size!=sizeof(float)
        &&expectation.scalar_size!=sizeof(double))
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v5 expectation requires float32 or "
            "float64 spline scalars.");
    if(expectation.required_capabilities
        !=(expectation.v4.required_capabilities
            |SYMMETRIX_JIT_MH1_HOST_SPLINE_R_FORWARD_V5
            |SYMMETRIX_JIT_MH1_HOST_SPLINE_R_SOURCE_REVERSE_V5)
        ||(expectation.required_capabilities&~known_capabilities_v5)!=0)
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v5 expectation requires the ABI-v4 "
            "capabilities plus spline R forward and source reverse.");
    for(const auto& spline:expectation.spline_r_programs)
        if((spline.flags
                &~SYMMETRIX_JIT_MH1_HOST_SPLINE_R_PACKED_FORWARD_MESSAGES_V5)!=0
            ||spline.weight_function_count<=0
            ||spline.density_function_index!=spline.weight_function_count
            ||spline.coefficient_count!=4)
            throw std::invalid_argument(
                "Execution MH-1 host ABI-v5 expectation requires complete "
                "cubic spline R extents.");
}

SymmetrixJitMH1HostPluginV4 project_v4_descriptor(
    const SymmetrixJitMH1HostPluginV5& descriptor)
{
    SymmetrixJitMH1HostPluginV4 result{};
    result.abi_version=SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_VERSION;
    result.struct_size=sizeof(SymmetrixJitMH1HostPluginV4);
    result.pointer_size=descriptor.pointer_size;
    result.byte_order=descriptor.byte_order;
    result.capabilities=descriptor.capabilities&known_capabilities_v4;
    result.interaction_count=descriptor.interaction_count;
    // ABI-v4 owners remain float32. ABI-v5 uses descriptor.scalar_size for
    // the independently typed spline packets.
    result.scalar_size=sizeof(float);
    result.reserved=0;
    result.abi_tag=SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_TAG;
    result.artifact_id=descriptor.artifact_id;
    result.generation_fingerprint=descriptor.generation_fingerprint;
    result.semantic_fingerprint=descriptor.semantic_fingerprint;
    result.structure_fingerprint=descriptor.structure_fingerprint;
    result.runtime_layout_fingerprint=descriptor.runtime_layout_fingerprint;
    result.forward_owner=descriptor.forward_owner;
    result.source_reverse_owner=descriptor.source_reverse_owner;
    result.edge_reverse_owner=descriptor.edge_reverse_owner;
    for(std::size_t interaction=0;
        interaction<SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS;++interaction) {
        result.interactions[interaction]=descriptor.interactions[interaction];
        result.conditioning[interaction]=descriptor.conditioning[interaction];
        result.node_programs[interaction]=descriptor.node_programs[interaction];
    }
    return result;
}

void validate_descriptor(
    const std::string& path,
    const SymmetrixJitMH1HostPluginV5& descriptor,
    const MH1HostPluginV5Expectation& expectation)
{
    if(descriptor.abi_version
        !=SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_ABI_VERSION)
        throw plugin_error(path,"unsupported ABI version");
    if(descriptor.struct_size<sizeof(SymmetrixJitMH1HostPluginV5))
        throw plugin_error(path,"descriptor is smaller than ABI version 5");
    if(require_text(path,descriptor.abi_tag,"ABI tag")
        !=SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_ABI_TAG)
        throw plugin_error(path,"ABI tag does not match");
    if(descriptor.capabilities!=expectation.required_capabilities)
        throw plugin_error(path,"ABI-v5 owner capabilities do not match");
    if(descriptor.scalar_size!=expectation.scalar_size)
        throw plugin_error(path,"ABI-v5 spline scalar width does not match");

    const auto v4=project_v4_descriptor(descriptor);
    validate_descriptor(path,v4,expectation.v4);
    for(std::size_t interaction=0;
        interaction<SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS;++interaction) {
        const auto& program=descriptor.spline_r_programs[interaction];
        const auto& expected=expectation.spline_r_programs[interaction];
        if(program.struct_size<sizeof(SymmetrixJitMH1HostSplineRProgramV5))
            throw plugin_error(
                path,"interaction "+std::to_string(interaction)
                    +" spline R descriptor is smaller than ABI version 5");
        if(program.interaction!=interaction||program.flags!=expected.flags
            ||program.reserved!=0||program.reserved_2!=0)
            throw plugin_error(
                path,"interaction "+std::to_string(interaction)
                    +" spline R descriptor metadata is invalid");
        require_extent(path,expected.weight_function_count,
            program.weight_function_count,"spline weight function count",interaction);
        require_extent(path,expected.density_function_index,
            program.density_function_index,"spline density function index",interaction);
        require_extent(path,expected.coefficient_count,
            program.coefficient_count,"spline coefficient count",interaction);
        if(program.forward_owner==nullptr)
            throw plugin_error(
                path,"interaction "+std::to_string(interaction)
                    +" spline R forward owner is null");
        if(program.source_reverse_owner==nullptr)
            throw plugin_error(
                path,"interaction "+std::to_string(interaction)
                    +" spline R source reverse owner is null");
    }
}

}  // namespace

MH1HostPlugin MH1HostPlugin::load(
    std::string path,
    const MH1HostPluginExpectation& expectation)
{
    if(path.empty())
        throw std::invalid_argument("Execution MH-1 host plugin path is empty.");
    validate_expectation(expectation);

    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(handle==nullptr) {
        const char* error=dlerror();
        throw plugin_error(path,error==nullptr?"dlopen failed":error);
    }

    try {
        dlerror();
        void* symbol=dlsym(
            handle,SYMMETRIX_JIT_MH1_HOST_PLUGIN_QUERY_SYMBOL);
        const char* symbol_error=dlerror();
        if(symbol_error!=nullptr)
            throw plugin_error(path,symbol_error);
        if(symbol==nullptr)
            throw plugin_error(path,"query symbol is null");

        const auto query=
            reinterpret_cast<SymmetrixJitMH1HostPluginQueryV3>(symbol);
        const auto* descriptor=query();
        if(descriptor==nullptr)
            throw plugin_error(path,"query returned a null descriptor");
        validate_descriptor(path,*descriptor,expectation);
        return MH1HostPlugin(handle,descriptor,std::move(path));
    } catch(...) {
        dlclose(handle);
        throw;
    }
}

MH1HostPlugin::MH1HostPlugin(
    void* handle,
    const SymmetrixJitMH1HostPluginV3* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1HostPlugin::~MH1HostPlugin()
{
    reset();
}

MH1HostPlugin::MH1HostPlugin(MH1HostPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

MH1HostPlugin& MH1HostPlugin::operator=(MH1HostPlugin&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1HostPluginV3& MH1HostPlugin::descriptor() const
{
    if(descriptor_==nullptr)
        throw std::logic_error("Execution MH-1 host plugin handle is empty.");
    return *descriptor_;
}

void MH1HostPlugin::reset() noexcept
{
    descriptor_=nullptr;
    if(handle_!=nullptr) {
        dlclose(handle_);
        handle_=nullptr;
    }
    path_.clear();
}

MH1HostPluginV4 MH1HostPluginV4::load(
    std::string path,
    const MH1HostPluginV4Expectation& expectation)
{
    if(path.empty())
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v4 plugin path is empty.");
    validate_expectation(expectation);

    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(handle==nullptr) {
        const char* error=dlerror();
        throw plugin_error(path,error==nullptr?"dlopen failed":error);
    }
    try {
        dlerror();
        void* symbol=dlsym(
            handle,SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_QUERY_SYMBOL);
        const char* symbol_error=dlerror();
        if(symbol_error!=nullptr)
            throw plugin_error(path,symbol_error);
        if(symbol==nullptr)
            throw plugin_error(path,"ABI-v4 query symbol is null");
        const auto query=
            reinterpret_cast<SymmetrixJitMH1HostPluginQueryV4>(symbol);
        const auto* descriptor=query();
        if(descriptor==nullptr)
            throw plugin_error(path,"ABI-v4 query returned a null descriptor");
        validate_descriptor(path,*descriptor,expectation);
        return MH1HostPluginV4(handle,descriptor,std::move(path));
    } catch(...) {
        dlclose(handle);
        throw;
    }
}

MH1HostPluginV4::MH1HostPluginV4(
    void* handle,
    const SymmetrixJitMH1HostPluginV4* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1HostPluginV4::~MH1HostPluginV4()
{
    reset();
}

MH1HostPluginV4::MH1HostPluginV4(MH1HostPluginV4&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

MH1HostPluginV4& MH1HostPluginV4::operator=(MH1HostPluginV4&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1HostPluginV4& MH1HostPluginV4::descriptor() const
{
    if(descriptor_==nullptr)
        throw std::logic_error(
            "Execution MH-1 host ABI-v4 plugin handle is empty.");
    return *descriptor_;
}

void MH1HostPluginV4::reset() noexcept
{
    descriptor_=nullptr;
    if(handle_!=nullptr) {
        dlclose(handle_);
        handle_=nullptr;
    }
    path_.clear();
}

MH1HostPluginV5 MH1HostPluginV5::load(
    std::string path,
    const MH1HostPluginV5Expectation& expectation)
{
    if(path.empty())
        throw std::invalid_argument(
            "Execution MH-1 host ABI-v5 plugin path is empty.");
    validate_expectation(expectation);

    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(handle==nullptr) {
        const char* error=dlerror();
        throw plugin_error(path,error==nullptr?"dlopen failed":error);
    }
    try {
        dlerror();
        void* symbol=dlsym(
            handle,SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_QUERY_SYMBOL);
        const char* symbol_error=dlerror();
        if(symbol_error!=nullptr)
            throw plugin_error(path,symbol_error);
        if(symbol==nullptr)
            throw plugin_error(path,"ABI-v5 query symbol is null");
        const auto query=
            reinterpret_cast<SymmetrixJitMH1HostPluginQueryV5>(symbol);
        const auto* descriptor=query();
        if(descriptor==nullptr)
            throw plugin_error(path,"ABI-v5 query returned a null descriptor");
        validate_descriptor(path,*descriptor,expectation);
        return MH1HostPluginV5(handle,descriptor,std::move(path));
    } catch(...) {
        dlclose(handle);
        throw;
    }
}

MH1HostPluginV5::MH1HostPluginV5(
    void* handle,
    const SymmetrixJitMH1HostPluginV5* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1HostPluginV5::~MH1HostPluginV5()
{
    reset();
}

MH1HostPluginV5::MH1HostPluginV5(MH1HostPluginV5&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

MH1HostPluginV5& MH1HostPluginV5::operator=(MH1HostPluginV5&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1HostPluginV5& MH1HostPluginV5::descriptor() const
{
    if(descriptor_==nullptr)
        throw std::logic_error(
            "Execution MH-1 host ABI-v5 plugin handle is empty.");
    return *descriptor_;
}

void MH1HostPluginV5::reset() noexcept
{
    descriptor_=nullptr;
    if(handle_!=nullptr) {
        dlclose(handle_);
        handle_=nullptr;
    }
    path_.clear();
}

}  // namespace symmetrix::execution
#else
namespace symmetrix::execution {

MH1HostPlugin MH1HostPlugin::load(
    std::string,
    const MH1HostPluginExpectation&)
{
    throw std::runtime_error(
        "Execution MH-1 host plugins require POSIX dlopen/dlsym support.");
}

MH1HostPlugin::MH1HostPlugin(
    void* handle,
    const SymmetrixJitMH1HostPluginV3* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1HostPlugin::~MH1HostPlugin()
{
    reset();
}

MH1HostPlugin::MH1HostPlugin(MH1HostPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

MH1HostPlugin& MH1HostPlugin::operator=(MH1HostPlugin&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1HostPluginV3& MH1HostPlugin::descriptor() const
{
    throw std::logic_error("Execution MH-1 host plugin support is disabled.");
}

void MH1HostPlugin::reset() noexcept
{
    handle_=nullptr;
    descriptor_=nullptr;
    path_.clear();
}

MH1HostPluginV4 MH1HostPluginV4::load(
    std::string,
    const MH1HostPluginV4Expectation&)
{
    throw std::runtime_error(
        "Execution MH-1 host ABI-v4 plugins require POSIX dlopen/dlsym support.");
}

MH1HostPluginV4::MH1HostPluginV4(
    void* handle,
    const SymmetrixJitMH1HostPluginV4* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1HostPluginV4::~MH1HostPluginV4()
{
    reset();
}

MH1HostPluginV4::MH1HostPluginV4(MH1HostPluginV4&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

MH1HostPluginV4& MH1HostPluginV4::operator=(MH1HostPluginV4&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1HostPluginV4& MH1HostPluginV4::descriptor() const
{
    throw std::logic_error(
        "Execution MH-1 host ABI-v4 plugin support is disabled.");
}

void MH1HostPluginV4::reset() noexcept
{
    handle_=nullptr;
    descriptor_=nullptr;
    path_.clear();
}

MH1HostPluginV5 MH1HostPluginV5::load(
    std::string,
    const MH1HostPluginV5Expectation&)
{
    throw std::runtime_error(
        "Execution MH-1 host ABI-v5 plugins require POSIX dlopen/dlsym support.");
}

MH1HostPluginV5::MH1HostPluginV5(
    void* handle,
    const SymmetrixJitMH1HostPluginV5* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

MH1HostPluginV5::~MH1HostPluginV5()
{
    reset();
}

MH1HostPluginV5::MH1HostPluginV5(MH1HostPluginV5&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

MH1HostPluginV5& MH1HostPluginV5::operator=(MH1HostPluginV5&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitMH1HostPluginV5& MH1HostPluginV5::descriptor() const
{
    throw std::logic_error(
        "Execution MH-1 host ABI-v5 plugin support is disabled.");
}

void MH1HostPluginV5::reset() noexcept
{
    handle_=nullptr;
    descriptor_=nullptr;
    path_.clear();
}

}  // namespace symmetrix::execution
#endif
