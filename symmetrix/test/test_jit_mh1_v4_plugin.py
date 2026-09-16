import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "libsymmetrix" / "source"
PACKAGE_SOURCE = REPOSITORY / "symmetrix" / "source" / "symmetrix"
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Execution MH-1 plugin loaders are POSIX-only"
)


def _compile(compiler, *arguments):
    subprocess.run(
        [compiler, *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
    )


def _host_plugin_source():
    return r"""
#include "jit_mh1_host_plugin_abi.h"

extern "C" void forward_owner(
    const SymmetrixJitMH1HostForwardArgsV4*, int32_t) {}
extern "C" void source_reverse_owner(
    const SymmetrixJitMH1HostReverseArgsV4*, int32_t) {}
extern "C" void edge_reverse_owner(
    const SymmetrixJitMH1HostReverseArgsV4*, int32_t) {}
extern "C" void conditioning_forward_owner(
    const SymmetrixJitMH1HostConditioningForwardArgsV4*, int32_t) {}
extern "C" void conditioning_reverse_owner(
    const SymmetrixJitMH1HostConditioningReverseArgsV4*, int32_t) {}
extern "C" void node_forward_owner(
    const SymmetrixJitMH1HostNodeForwardArgsV4*, int32_t) {}
extern "C" void node_reverse_owner(
    const SymmetrixJitMH1HostNodeReverseArgsV4*, int32_t) {}

static SymmetrixJitMH1HostNodeProgramV4 make_node(
    uint32_t index, bool requires_source_adjoint)
{
    SymmetrixJitMH1HostNodeProgramV4 node{};
    node.struct_size = sizeof(node);
    node.interaction = index;
    node.forward_phase_count = SYMMETRIX_JIT_MH1_HOST_NODE_FORWARD_PHASES_V4;
    node.reverse_phase_count = SYMMETRIX_JIT_MH1_HOST_NODE_REVERSE_PHASES_V4;
    node.element_count = 5;
    node.input_dimension = index == 0 ? 8 : 11;
    node.up_dimension = index == 0 ? 6 : 8;
    node.residual_dimension = 6;
    node.skip_dimension = index == 0 ? 7 : 4;
    node.message_dimension = index == 0 ? 9 : 10;
    node.interaction_output_dimension = 10;
    node.output_dimension = index == 0 ? 11 : 4;
    node.product_term_count = index == 0 ? 3 : 2;
    node.node_arena_dimension = 12;
    node.flags = requires_source_adjoint
        ? SYMMETRIX_JIT_MH1_HOST_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4
        : 0u;
    node.linear_parameter_count = index == 0 ? 20 : 21;
    node.product_parameter_count = index == 0 ? 9 : 8;
    node.readout_parameter_count = index == 0 ? 4 : 5;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        node.forward_phases[phase] = {
            sizeof(SymmetrixJitMH1HostNodeForwardPhaseV4), index,
            SYMMETRIX_JIT_MH1_HOST_NODE_PRE_FORWARD_V4 + phase,
            0u, 1, 0, 0, 0, &node_forward_owner};
        const bool disabled = !requires_source_adjoint && phase == 1;
        node.reverse_phases[phase] = {
            sizeof(SymmetrixJitMH1HostNodeReversePhaseV4), index,
            SYMMETRIX_JIT_MH1_HOST_NODE_POST_REVERSE_V4 + phase,
            0u, disabled ? 0 : 1, 0, 0, 0,
            disabled ? nullptr : &node_reverse_owner};
    }
#ifdef ENABLE_DEAD_LAYER0_REVERSE
    if (index == 0) {
        node.reverse_phases[1].owners_per_node = 1;
        node.reverse_phases[1].owner = &node_reverse_owner;
    }
#endif
    return node;
}

static const SymmetrixJitMH1HostPluginV4 descriptor = [] {
    SymmetrixJitMH1HostPluginV4 value{};
    value.abi_version = SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_VERSION;
    value.struct_size = sizeof(value);
    value.pointer_size = sizeof(void*);
    value.byte_order = SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER;
    value.capabilities =
        SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V4
        | SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V4
        | SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V4
        | SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V4
        | SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V4
        | SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V4
        | SYMMETRIX_JIT_MH1_HOST_NODE_PROGRAM_PHASES_V4
        | SYMMETRIX_JIT_MH1_HOST_PERSISTENT_IR_MUL_NODE_STATE_V4
        | SYMMETRIX_JIT_MH1_HOST_FIXED_WEIGHT_COORDINATES_V4;
#ifdef ENABLE_OPTIONAL_SOURCE_OWNERSHIP
    value.capabilities |=
        SYMMETRIX_JIT_MH1_HOST_SOURCE_OWNS_EDGE_REVERSE_V4 |
        SYMMETRIX_JIT_MH1_HOST_PHI_MAJOR_FORWARD_WEIGHT_V4;
#endif
    value.interaction_count = 2;
    value.scalar_size = sizeof(float);
    value.abi_tag = SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_TAG;
    value.artifact_id = "fixture-mh1-host-v4";
    value.generation_fingerprint = "sha256:generation";
    value.semantic_fingerprint = "sha256:semantic";
    value.structure_fingerprint = "sha256:structure";
#ifdef STALE_LAYOUT
    value.runtime_layout_fingerprint = "sha256:stale";
#else
    value.runtime_layout_fingerprint = "sha256:runtime-layout";
#endif
    value.interactions[0] = {sizeof(SymmetrixJitMH1HostInteractionV4),
        0u, 8, 4, 6, 2, 3, 2, 4, 1};
    value.interactions[1] = {sizeof(SymmetrixJitMH1HostInteractionV4),
        0u, 12, 9, 10, 4, 5, 2, 6, 2};
    value.forward_owner = &forward_owner;
    value.source_reverse_owner = &source_reverse_owner;
    value.edge_reverse_owner = &edge_reverse_owner;
    for (uint32_t index = 0; index < 2; ++index) {
        value.conditioning[index] = {
            sizeof(SymmetrixJitMH1HostConditioningOwnersV4),
            index, 0u, 0u, &conditioning_forward_owner,
            &conditioning_reverse_owner};
    }
    value.node_programs[0] = make_node(0, false);
    value.node_programs[1] = make_node(1, true);
    return value;
}();

extern "C" SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT
const SymmetrixJitMH1HostPluginV4*
symmetrix_jit_mh1_host_plugin_query_v4()
{
    return &descriptor;
}
"""


def _host_harness_source():
    return r"""
#include "jit_mh1_host_plugin.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3) return 64;
    symmetrix::execution::MH1HostPluginV4Expectation expected;
    expected.artifact_id = "fixture-mh1-host-v4";
    expected.generation_fingerprint = "sha256:generation";
    expected.semantic_fingerprint = "sha256:semantic";
    expected.structure_fingerprint = "sha256:structure";
    expected.runtime_layout_fingerprint = "sha256:runtime-layout";
    expected.required_capabilities = 0x1ffu;
    expected.interactions[0] = {8, 4, 6, 2, 3, 2, 4, 1};
    expected.interactions[1] = {12, 9, 10, 4, 5, 2, 6, 2};
    expected.node_programs[0] = {
        5, 8, 6, 6, 7, 9, 10, 11, 3, 12, false,
        20, 9, 4, {1, 1}, {1, 0}};
    expected.node_programs[1] = {
        5, 11, 8, 6, 4, 10, 10, 4, 2, 12, true,
        21, 8, 5, {1, 1}, {1, 1}};
    try {
        auto plugin = symmetrix::execution::MH1HostPluginV4::load(
            argv[2], expected);
        return std::string(argv[1]) == "accept" && plugin ? 0 : 1;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        const std::string mode = argv[1];
        if (mode == "reject-layout"
            && message.find("runtime layout fingerprint does not match")
                != std::string::npos) return 0;
        if (mode == "reject-dead-reverse"
            && message.find("disabled node reverse phase is not empty")
                != std::string::npos) return 0;
        std::cerr << message << '\n';
        return 2;
    }
}
"""


def test_mh1_host_v4_loader_validates_layout_and_dead_reverse(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the MH-1 v4 loader test")
    plugin_source = tmp_path / "host_plugin.cpp"
    harness_source = tmp_path / "host_harness.cpp"
    plugin_source.write_text(_host_plugin_source(), encoding="ascii")
    harness_source.write_text(_host_harness_source(), encoding="ascii")
    common = ("-std=c++20", "-O2", "-I", SOURCE, "-I", PACKAGE_SOURCE)
    variants = (
        ("accept", (), tmp_path / "host_valid.so"),
        (
            "accept",
            ("-DENABLE_OPTIONAL_SOURCE_OWNERSHIP=1",),
            tmp_path / "host_optional.so",
        ),
        ("reject-layout", ("-DSTALE_LAYOUT=1",), tmp_path / "host_stale.so"),
        (
            "reject-dead-reverse",
            ("-DENABLE_DEAD_LAYER0_REVERSE=1",),
            tmp_path / "host_dead_reverse.so",
        ),
    )
    for _, defines, output in variants:
        _compile(
            compiler,
            *common,
            *defines,
            "-shared",
            "-fPIC",
            plugin_source,
            "-o",
            output,
        )
    harness = tmp_path / "host_harness"
    _compile(
        compiler,
        *common,
        harness_source,
        SOURCE / "jit_mh1_host_plugin.cpp",
        "-ldl",
        "-o",
        harness,
    )
    for mode, _, plugin in variants:
        subprocess.run(
            [harness, mode, plugin], check=True, capture_output=True, text=True
        )


def _cuda_plugin_source():
    return r"""
#include "jit_mh1_cuda_plugin_abi.h"

extern "C" int32_t forward_launch(
    const SymmetrixJitMH1CudaForwardArgsV4*, void*, int32_t) { return 0; }
extern "C" int32_t reverse_launch(
    const SymmetrixJitMH1CudaReverseArgsV4*, void*, int32_t, int32_t)
    { return 0; }
extern "C" int32_t conditioning_forward_launch(
    const SymmetrixJitMH1CudaConditioningForwardArgsV4*, void*, int32_t)
    { return 0; }
extern "C" int32_t conditioning_reverse_launch(
    const SymmetrixJitMH1CudaConditioningReverseArgsV4*, void*, int32_t)
    { return 0; }
extern "C" int32_t node_forward_launch(
    const SymmetrixJitMH1CudaNodeForwardArgsV4*, void*, int32_t)
    { return 0; }
extern "C" int32_t node_reverse_launch(
    const SymmetrixJitMH1CudaNodeReverseArgsV4*, void*, int32_t)
    { return 0; }

static SymmetrixJitMH1CudaNodeProgramV4 make_node(
    uint32_t index, bool requires_source_adjoint)
{
    SymmetrixJitMH1CudaNodeProgramV4 node{};
    node.struct_size = sizeof(node);
    node.interaction = index;
    node.forward_phase_count = 2;
    node.reverse_phase_count = 2;
    node.element_count = 5;
    node.input_dimension = index == 0 ? 8 : 11;
    node.up_dimension = index == 0 ? 6 : 8;
    node.residual_dimension = 6;
    node.skip_dimension = index == 0 ? 7 : 4;
    node.message_dimension = index == 0 ? 9 : 10;
    node.interaction_output_dimension = 10;
    node.output_dimension = index == 0 ? 11 : 4;
    node.product_term_count = index == 0 ? 3 : 2;
    node.node_arena_dimension = 12;
    node.flags = requires_source_adjoint
        ? SYMMETRIX_JIT_MH1_CUDA_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4
        : 0u;
#ifdef RETAIN_NODE_STATE
    node.flags |= SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_PRE_GATE_V4
        | SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_INTERACTION_OUTPUT_V4;
    node.retained_pre_gate_dimension = node.residual_dimension;
    node.retained_interaction_output_dimension =
        node.interaction_output_dimension;
#ifdef INVALID_RETAINED_EXTENT
    ++node.retained_pre_gate_dimension;
#endif
#endif
    node.linear_parameter_count = index == 0 ? 20 : 21;
    node.product_parameter_count = index == 0 ? 9 : 8;
    node.readout_parameter_count = index == 0 ? 4 : 5;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        node.forward_phases[phase] = {
            sizeof(SymmetrixJitMH1CudaNodeForwardPhaseV4), index,
            SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_FORWARD_V4 + phase,
            0u, 128, 0, 0, 0, &node_forward_launch};
        const bool disabled = !requires_source_adjoint && phase == 1;
        node.reverse_phases[phase] = {
            sizeof(SymmetrixJitMH1CudaNodeReversePhaseV4), index,
            SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4 + phase,
            0u, disabled ? 0 : 128, 0, 0, 0,
            disabled ? nullptr : &node_reverse_launch};
    }
    return node;
}

static const SymmetrixJitMH1CudaPluginV4 descriptor = [] {
    SymmetrixJitMH1CudaPluginV4 value{};
    value.abi_version = SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_VERSION;
    value.struct_size = sizeof(value);
    value.pointer_size = sizeof(void*);
    value.byte_order = SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER;
    value.capabilities = 0xfffu;
    value.interaction_count = 2;
    value.scalar_size = sizeof(float);
    value.abi_tag = SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_TAG;
    value.artifact_id = "fixture-mh1-cuda-v4";
    value.generation_fingerprint = "sha256:generation";
    value.semantic_fingerprint = "sha256:semantic";
    value.structure_fingerprint = "sha256:structure";
#ifdef STALE_LAYOUT
    value.runtime_layout_fingerprint = "sha256:stale";
#else
    value.runtime_layout_fingerprint = "sha256:runtime-layout";
#endif
    value.target_compute_capability = 80;
    value.interactions[0] = {sizeof(SymmetrixJitMH1CudaInteractionV4),
        0u, 8, 4, 6, 2, 3, 2, 4, 1, 128, 128, 128, 2};
    value.interactions[1] = {sizeof(SymmetrixJitMH1CudaInteractionV4),
        0u, 12, 9, 10, 4, 5, 2, 6, 2, 128, 128, 128, 2};
    for (uint32_t index = 0; index < 2; ++index) {
        value.launches[index] = {
            sizeof(SymmetrixJitMH1CudaInteractionLaunchesV4),
            index, 0u, 0u, &forward_launch, &reverse_launch};
        value.conditioning_launches[index] = {
            sizeof(SymmetrixJitMH1CudaConditioningLaunchesV4),
            index, 0u, 0u, &conditioning_forward_launch,
            &conditioning_reverse_launch};
    }
    value.node_programs[0] = make_node(0, false);
    value.node_programs[1] = make_node(1, true);
    return value;
}();

extern "C" SYMMETRIX_JIT_MH1_CUDA_PLUGIN_EXPORT
const SymmetrixJitMH1CudaPluginV4*
symmetrix_jit_mh1_cuda_plugin_query_v4()
{
    return &descriptor;
}
"""


def _cuda_harness_source():
    return r"""
#include "jit_mh1_cuda_plugin.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3) return 64;
    symmetrix::execution::MH1CudaPluginV4Expectation expected;
    expected.artifact_id = "fixture-mh1-cuda-v4";
    expected.generation_fingerprint = "sha256:generation";
    expected.semantic_fingerprint = "sha256:semantic";
    expected.structure_fingerprint = "sha256:structure";
    expected.runtime_layout_fingerprint = "sha256:runtime-layout";
    expected.target_compute_capability = 80;
    expected.max_threads_per_block = 256;
    expected.required_capabilities = 0xfffu;
    expected.interactions[0] = {8, 4, 6, 2, 3, 2, 4, 1};
    expected.interactions[1] = {12, 9, 10, 4, 5, 2, 6, 2};
    expected.node_programs[0] = {
        5, 8, 6, 6, 7, 9, 10, 11, 3, 12, false,
        20, 9, 4, {128, 128}, {128, 0}};
    expected.node_programs[1] = {
        5, 11, 8, 6, 4, 10, 10, 4, 2, 12, true,
        21, 8, 5, {128, 128}, {128, 128}};
    try {
        auto plugin = symmetrix::execution::MH1CudaPluginV4::load(
            argv[2], expected);
        const std::string mode = argv[1];
        if (!mode.starts_with("accept-") || !plugin) return 1;
        const bool retention = mode == "accept-retention";
        float retained = 0.0f;
        SymmetrixJitMH1CudaNodeForwardArgsV4 forward{};
        forward.struct_size = sizeof(forward);
        forward.phase = SYMMETRIX_JIT_MH1_CUDA_NODE_POST_FORWARD_V4;
        forward.num_nodes = 1;
        SymmetrixJitMH1CudaNodeReverseArgsV4 reverse{};
        reverse.struct_size = sizeof(reverse);
        reverse.phase = SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4;
        reverse.num_nodes = 1;
        if (retention) {
            bool rejected = false;
            try {
                plugin.launch_node_forward(&forward, nullptr, 1);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected) return 1;
            forward.retained_pre_gate = &retained;
            forward.retained_interaction_output = &retained;
            reverse.retained_pre_gate = &retained;
            reverse.retained_interaction_output = &retained;
        }
        if (plugin.launch_node_forward(&forward, nullptr, 1) != 0
            || plugin.launch_node_reverse(&reverse, nullptr, 1) != 0) return 1;
        forward.struct_size = offsetof(
            SymmetrixJitMH1CudaNodeForwardArgsV4, retained_pre_gate);
        reverse.struct_size = offsetof(
            SymmetrixJitMH1CudaNodeReverseArgsV4, retained_pre_gate);
        return plugin.launch_node_forward(&forward, nullptr, 1) == 1
                && plugin.launch_node_reverse(&reverse, nullptr, 1) == 1
            ? 0 : 1;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (std::string(argv[1]) == "reject-layout"
            && message.find("runtime layout fingerprint does not match")
                != std::string::npos) return 0;
        if (std::string(argv[1]) == "reject-retained-extent"
            && message.find("retained node-state flags and extents disagree")
                != std::string::npos) return 0;
        std::cerr << message << '\n';
        return 2;
    }
}
"""


def test_mh1_cuda_v4_loader_validates_runtime_layout(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the MH-1 v4 loader test")
    plugin_source = tmp_path / "cuda_plugin.cpp"
    harness_source = tmp_path / "cuda_harness.cpp"
    plugin_source.write_text(_cuda_plugin_source(), encoding="ascii")
    harness_source.write_text(_cuda_harness_source(), encoding="ascii")
    common = ("-std=c++20", "-O2", "-I", SOURCE, "-I", PACKAGE_SOURCE)
    variants = (
        ("accept-recompute", (), tmp_path / "cuda_valid.so"),
        (
            "accept-retention",
            ("-DRETAIN_NODE_STATE=1",),
            tmp_path / "cuda_retention.so",
        ),
        (
            "reject-retained-extent",
            ("-DRETAIN_NODE_STATE=1", "-DINVALID_RETAINED_EXTENT=1"),
            tmp_path / "cuda_bad_retention.so",
        ),
        ("reject-layout", ("-DSTALE_LAYOUT=1",), tmp_path / "cuda_stale.so"),
    )
    for _, defines, output in variants:
        _compile(
            compiler,
            *common,
            *defines,
            "-shared",
            "-fPIC",
            plugin_source,
            "-o",
            output,
        )
    harness = tmp_path / "cuda_harness"
    _compile(
        compiler,
        *common,
        harness_source,
        SOURCE / "jit_mh1_cuda_plugin.cpp",
        "-ldl",
        "-o",
        harness,
    )
    for mode, _, plugin in variants:
        subprocess.run(
            [harness, mode, plugin], check=True, capture_output=True, text=True
        )


def test_mh1_device_module_defaults_to_retention_and_v2_requires_policy():
    loader = (SOURCE / "jit_mh1_cuda_plugin.cpp").read_text(encoding="ascii")

    assert (
        'launch_plan_version==2\n            &&(!plan.contains("node_state_policy")'
        in loader
    )
    assert '||!plan.at("node_state_policy").is_string()' in loader
    assert '"launch-plan v2 requires an explicit node-state policy"' in loader
    assert "const auto node_state_policy=launch_plan_version==1" in loader
    assert '?plan.value("node_state_policy",std::string("full-retention-v1"))' in loader
    assert ':plan.at("node_state_policy").get<std::string>()' in loader
    assert '&&node_state_policy!="reuse-adjoints-v1"' in loader
    assert '&&node_state_policy!="retain-interaction-v1"' in loader
    assert "SYMMETRIX_JIT_MH1_CUDA_NODE_REUSE_MESSAGE_ADJOINT_V4" in loader


def test_mh1_device_module_admits_only_named_nonstandard_tiles():
    loader = (SOURCE / "jit_mh1_cuda_plugin.cpp").read_text(encoding="ascii")

    assert 'kernel.ends_with("_reverse_product_tiled")' in loader
    assert 'kernel.ends_with("_forward_product_tiled")' in loader
    assert "result.tile_nodes==1&&result.tile_channels==128&&threads==128" in loader
    assert 'kernel.ends_with("_reverse_product_tiled")' in loader
    assert "(result.tile_nodes==2||result.tile_nodes==4)" in loader
    assert "&&result.tile_channels==128&&threads==128" in loader
    assert 'kernel.ends_with("_l1_reverse_message_grouped")' in loader
    assert 'kernel.ends_with("_l1_reverse_linear2_grouped")' in loader
    assert 'kernel.ends_with("_l0_reverse_linear2_grouped")' in loader
    assert '||kernel.ends_with("_l1_reverse_linear2_grouped")' in loader
    assert '||kernel.ends_with("_l0_forward_linear2_grouped")' in loader
    assert '||kernel.ends_with("_l1_forward_linear2_grouped")' in loader
    assert 'kernel.ends_with("_l0_forward_linear2_grouped")' in loader
    assert 'kernel.ends_with("_l1_forward_linear2_grouped")' in loader
    assert "result.tile_nodes==64&&result.tile_channels==32&&threads==256" in loader
    assert 'kernel.ends_with("_l0_reverse_linear2_grouped")' in loader
    assert "result.tile_nodes==128&&result.tile_channels==32&&threads==256" in loader
    assert "(!standard_tiles&&!product_tiles&&!paired_product_reverse_tile" in loader
    assert "&&!wide_grouped_tile&&!xwide_grouped_tile)" in loader


def test_mh1_compact_edge_loader_rejects_mutated_thread_geometry():
    loader = (SOURCE / "jit_mh1_cuda_plugin.cpp").read_text(encoding="ascii")
    compact_geometry_guard = (
        "if(interaction.edge_threads!=128\n"
        "                    ||interaction.edge_phi_block_multiplier!=4)"
    )

    assert compact_geometry_guard in loader
    assert (
        compact_geometry_guard.replace("edge_threads!=128", "edge_threads!=256")
        not in loader
    )
