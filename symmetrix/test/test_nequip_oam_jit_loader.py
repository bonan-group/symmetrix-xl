import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from symmetrix.oam_receiver_codegen import (
    OAMHostGenerationPolicy,
    oam_receiver_jit_metadata,
    render_oam_receiver_host_plugin,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "libsymmetrix" / "source"
PACKAGE_SOURCE = REPOSITORY / "symmetrix" / "source" / "symmetrix"
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="OAM host plugin loaders are POSIX-only"
)


def _compile(compiler, *arguments):
    subprocess.run(
        [compiler, *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_v7_loader_rejects_exact_host_resource_mismatches(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the OAM host loader test")
    fixture = (
        REPOSITORY / "symmetrix" / "test" / "data" / "nequip" / "oam_s_v2_minimal.json"
    )
    contract = json.loads(fixture.read_text(encoding="utf-8"))
    policy = OAMHostGenerationPolicy(
        owner_schedule="channel_tiled", channel_tile=3, scalar_width=8
    )
    metadata = oam_receiver_jit_metadata(
        contract, "float64", host_target="host-isa-x86-64-v3", generation_policy=policy
    )
    rendered = render_oam_receiver_host_plugin(
        contract,
        "float64",
        host_target="host-isa-x86-64-v3",
        generation_policy=policy,
    )
    plugin_source = tmp_path / "host_v7.cpp"
    plugin_source.write_text(rendered, encoding="ascii")
    plugin = tmp_path / "host_v7.so"
    common = ("-std=c++20", "-O2", "-I", SOURCE, "-I", PACKAGE_SOURCE)
    _compile(compiler, *common, "-shared", "-fPIC", plugin_source, "-o", plugin)
    layers = ",".join(
        "{"
        + ",".join(
            str(row[key])
            for key in (
                "source_dimension",
                "edge_dimension",
                "weight_dimension",
                "output_dimension",
                "instruction_count",
                "channel_tile",
                "target_dimension",
            )
        )
        + "}"
        for row in metadata["layers"]
    )
    malformed_source = tmp_path / "host_v7_small_layer.cpp"
    malformed_source.write_text(
        rendered.replace("sizeof(SymmetrixJitOAMHostLayerV7)", "1u", 1),
        encoding="ascii",
    )
    malformed_plugin = tmp_path / "host_v7_small_layer.so"
    _compile(
        compiler,
        *common,
        "-shared",
        "-fPIC",
        malformed_source,
        "-o",
        malformed_plugin,
    )
    harness_source = tmp_path / "host_v7_loader.cpp"
    harness_source.write_text(
        '#include "oam_jit_host_plugin.hpp"\n#include <iostream>\n#include <string>\n'
        "int main(int argc,char** argv){ symmetrix::execution::OAMHostPluginExpectationV2 e;"
        f'e.artifact_id="{metadata["artifact_id"]}";'
        f'e.generation_fingerprint="{metadata["generation_fingerprint"]}";'
        f'e.semantic_fingerprint="{metadata["semantic_fingerprint"]}";'
        f'e.structure_fingerprint="{metadata["structure_fingerprint"]}";'
        f'e.schedule_identity="{metadata["schedule_identity"]}";'
        f'e.host_target="{metadata["host_target"]}";'
        f'e.scalar_size=8;e.scalar_width=8;e.required_isa="{metadata["required_isa"]}";'
        f'e.full_reverse_reduction_policy="{metadata["full_reverse_reduction_policy"]}";'
        f'e.schedule_qualification="{metadata["schedule_qualification"]}";'
        f'e.forward_owner_schedule="{metadata["forward_owner_schedule"]}";'
        f'e.full_reverse_owner_schedule="{metadata["full_reverse_owner_schedule"]}";'
        f'e.tiling_policy="{metadata["tiling_policy"]}";'
        f'e.vector_policy="{metadata["vector_policy"]}";'
        f'e.register_budget_policy="{metadata["register_budget_policy"]}";'
        f"e.required_capabilities={metadata['required_capabilities']}u;e.layers={{{layers}}};"
        'std::string mode=argv[1];if(mode=="width")e.scalar_width=4;'
        'if(mode=="isa")e.required_isa="host-isa-wrong";'
        'if(mode=="full-schedule")e.full_reverse_owner_schedule="wrong";'
        'if(mode=="full-reduction")e.full_reverse_reduction_policy="wrong";'
        'try{auto p=symmetrix::execution::OAMHostPluginV2Handle::load(argv[2],e);return mode=="accept"&&p?0:1;}catch(const std::exception& x){std::string m=x.what();if((mode=="width"&&m.find("scalar width")!=std::string::npos)||(mode=="isa"&&m.find("required ISA")!=std::string::npos)||(mode=="full-schedule"&&m.find("full reverse owner schedule")!=std::string::npos)||(mode=="full-reduction"&&m.find("full reverse reduction policy")!=std::string::npos)||(mode=="small-layer"&&m.find("smaller than ABI version 7")!=std::string::npos)||(mode=="stale"&&m.find("stale OAM host artifact ABI")!=std::string::npos))return 0;std::cerr<<m;return 2;}}',
        encoding="ascii",
    )
    harness = tmp_path / "host_v7_loader"
    _compile(
        compiler,
        *common,
        harness_source,
        SOURCE / "oam_jit_host_plugin.cpp",
        "-ldl",
        "-o",
        harness,
    )
    for mode in (
        "accept",
        "width",
        "isa",
        "full-schedule",
        "full-reduction",
    ):
        subprocess.run(
            [harness, mode, plugin], check=True, capture_output=True, text=True
        )
    subprocess.run(
        [harness, "small-layer", malformed_plugin],
        check=True,
        capture_output=True,
        text=True,
    )
    stale_source = tmp_path / "host_v6_stale.cpp"
    stale_source.write_text(
        'extern "C" void* symmetrix_jit_oam_host_plugin_query_v6(){return nullptr;}',
        encoding="ascii",
    )
    stale_plugin = tmp_path / "host_v6_stale.so"
    _compile(compiler, "-shared", "-fPIC", stale_source, "-o", stale_plugin)
    subprocess.run(
        [harness, "stale", stale_plugin], check=True, capture_output=True, text=True
    )


def _plugin_source():
    return r"""
#include "oam_jit_host_plugin_abi.h"

extern "C" void forward_owner(
    const SymmetrixJitOAMHostForwardArgsV1*, int32_t) {}
extern "C" void reverse_owner(
    const SymmetrixJitOAMHostReverseArgsV1*, int32_t) {}

static const SymmetrixJitOAMHostLayerV1 layers[] = {
    {sizeof(SymmetrixJitOAMHostLayerV1), 0u, 0u, 0u, 8, 4, 6, 10, 3},
    {sizeof(SymmetrixJitOAMHostLayerV1), 1u, 0u, 0u, 10, 4, 8, 12, 5},
};
static const SymmetrixJitOAMHostForwardOwnerV1 forward_owners[] = {
    &forward_owner, &forward_owner};
static const SymmetrixJitOAMHostReverseOwnerV1 reverse_owners[] = {
    &reverse_owner, &reverse_owner};

static const SymmetrixJitOAMHostPluginV1 descriptor = [] {
    SymmetrixJitOAMHostPluginV1 value{};
    value.abi_version = SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION;
    value.struct_size = sizeof(value);
    value.pointer_size = sizeof(void*);
    value.byte_order = SYMMETRIX_JIT_OAM_HOST_PLUGIN_BYTE_ORDER;
    value.capabilities =
        SYMMETRIX_JIT_OAM_HOST_FORWARD_OWNER_V1
        | SYMMETRIX_JIT_OAM_HOST_REVERSE_OWNER_V1;
    value.layer_count = 2;
#ifdef BAD_SCALAR
    value.scalar_size = sizeof(float);
#else
    value.scalar_size = sizeof(double);
#endif
    value.abi_tag =
#ifdef BAD_TAG
        "wrong.oam.tag";
#else
        SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG;
#endif
    value.artifact_id = "fixture-oam-host-v1";
    value.generation_fingerprint =
#ifdef BAD_GENERATION
        "sha256:wrong-generation";
#else
        "sha256:generation";
#endif
    value.semantic_fingerprint =
#ifdef BAD_SEMANTIC
        "sha256:wrong-semantic";
#else
        "sha256:semantic";
#endif
    value.structure_fingerprint =
#ifdef BAD_STRUCTURE
        "sha256:wrong-structure";
#else
        "sha256:structure";
#endif
    value.layers = layers;
#ifdef BAD_LAYER_COUNT
    value.layer_count = 1;
#endif
#ifdef BAD_DIMENSION
    static SymmetrixJitOAMHostLayerV1 bad_layers[2];
    bad_layers[0] = layers[0];
    bad_layers[1] = layers[1];
    bad_layers[1].output_dimension = 99;
    value.layers = bad_layers;
#endif
#ifdef BAD_INSTRUCTION_COUNT
    static SymmetrixJitOAMHostLayerV1 bad_instruction_layers[2];
    bad_instruction_layers[0] = layers[0];
    bad_instruction_layers[1] = layers[1];
    bad_instruction_layers[1].instruction_count = 99;
    value.layers = bad_instruction_layers;
#endif
    value.forward_owners = forward_owners;
    value.reverse_owners = reverse_owners;
    return value;
}();

extern "C" SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
const SymmetrixJitOAMHostPluginV1*
symmetrix_jit_oam_host_plugin_query_v1()
{
    return &descriptor;
}
"""


def _harness_source():
    return r"""
#include "oam_jit_host_plugin.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3) return 64;
    symmetrix::execution::OAMHostPluginExpectation expected;
    expected.artifact_id = "fixture-oam-host-v1";
    expected.generation_fingerprint = "sha256:generation";
    expected.semantic_fingerprint = "sha256:semantic";
    expected.structure_fingerprint = "sha256:structure";
    expected.scalar_size = sizeof(double);
    expected.required_capabilities =
        SYMMETRIX_JIT_OAM_HOST_FORWARD_OWNER_V1
        | SYMMETRIX_JIT_OAM_HOST_REVERSE_OWNER_V1;
    expected.layers = {{8, 4, 6, 10, 3}, {10, 4, 8, 12, 5}};
    try {
        auto plugin = symmetrix::execution::OAMHostPlugin::load(
            argv[2], expected);
        return std::string(argv[1]) == "accept" && plugin ? 0 : 1;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        const std::string mode = argv[1];
        if ((mode == "reject-tag"
                && message.find("ABI tag does not match") != std::string::npos)
            || (mode == "reject-scalar"
                && message.find("scalar size does not match") != std::string::npos)
            || (mode == "reject-layer-count"
                && message.find("layer count does not match") != std::string::npos)
            || (mode == "reject-dimension"
                && message.find("output dimension does not match") != std::string::npos)
            || (mode == "reject-instruction"
                && message.find("instruction count does not match") != std::string::npos)
            || (mode == "reject-generation"
                && message.find("generation fingerprint does not match") != std::string::npos)
            || (mode == "reject-semantic"
                && message.find("semantic fingerprint does not match") != std::string::npos)
            || (mode == "reject-structure"
                && message.find("structure fingerprint does not match") != std::string::npos))
            return 0;
        std::cerr << message << '\n';
        return 2;
    }
}
"""


def test_oam_host_loader_validates_descriptor_identity_and_layers(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the OAM host loader test")
    plugin_source = tmp_path / "oam_host_plugin.cpp"
    harness_source = tmp_path / "oam_host_harness.cpp"
    plugin_source.write_text(_plugin_source(), encoding="ascii")
    harness_source.write_text(_harness_source(), encoding="ascii")
    common = ("-std=c++20", "-O2", "-I", SOURCE, "-I", PACKAGE_SOURCE)
    variants = (
        ("accept", (), tmp_path / "valid.so"),
        ("reject-tag", ("-DBAD_TAG=1",), tmp_path / "bad_tag.so"),
        ("reject-scalar", ("-DBAD_SCALAR=1",), tmp_path / "bad_scalar.so"),
        (
            "reject-layer-count",
            ("-DBAD_LAYER_COUNT=1",),
            tmp_path / "bad_layer_count.so",
        ),
        (
            "reject-dimension",
            ("-DBAD_DIMENSION=1",),
            tmp_path / "bad_dimension.so",
        ),
        (
            "reject-instruction",
            ("-DBAD_INSTRUCTION_COUNT=1",),
            tmp_path / "bad_instruction.so",
        ),
        (
            "reject-generation",
            ("-DBAD_GENERATION=1",),
            tmp_path / "bad_generation.so",
        ),
        (
            "reject-semantic",
            ("-DBAD_SEMANTIC=1",),
            tmp_path / "bad_semantic.so",
        ),
        (
            "reject-structure",
            ("-DBAD_STRUCTURE=1",),
            tmp_path / "bad_structure.so",
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
    harness = tmp_path / "oam_host_harness"
    _compile(
        compiler,
        *common,
        harness_source,
        SOURCE / "oam_jit_host_plugin.cpp",
        "-ldl",
        "-o",
        harness,
    )
    for mode, _, plugin in variants:
        subprocess.run(
            [harness, mode, plugin], check=True, capture_output=True, text=True
        )


def _v2_plugin_source():
    return r"""
#include "oam_jit_host_plugin_abi.h"

extern "C" void receiver_forward_owner(
    const SymmetrixJitOAMHostReceiverForwardArgsV2*, int32_t) {}
extern "C" void source_reverse_owner(
    const SymmetrixJitOAMHostSourceReverseArgsV2*, int32_t) {}

static const SymmetrixJitOAMHostLayerV2 layers[] = {
    {sizeof(SymmetrixJitOAMHostLayerV2), 0u, 0u, 0u, 8, 4, 6, 10, 3, 2, 5},
    {sizeof(SymmetrixJitOAMHostLayerV2), 1u, 0u, 0u, 10, 4, 8, 12, 5, 4, 7},
};
static const SymmetrixJitOAMHostReceiverForwardOwnerV2
    receiver_forward_owners[] = {&receiver_forward_owner, &receiver_forward_owner};
static const SymmetrixJitOAMHostSourceReverseOwnerV2
    source_reverse_owners[] = {&source_reverse_owner, &source_reverse_owner};

static const SymmetrixJitOAMHostPluginV2 descriptor = [] {
    SymmetrixJitOAMHostPluginV2 value{};
    value.abi_version =
#ifdef BAD_ABI_VERSION
        1u;
#else
        SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V2;
#endif
    value.struct_size = sizeof(value);
    value.pointer_size = sizeof(void*);
    value.byte_order = SYMMETRIX_JIT_OAM_HOST_PLUGIN_BYTE_ORDER;
    value.capabilities =
        SYMMETRIX_JIT_OAM_HOST_RECEIVER_FORWARD_V2
        | SYMMETRIX_JIT_OAM_HOST_SOURCE_REVERSE_V2
        | SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V2;
    value.layer_count = 2;
#ifdef BAD_SCALAR
    value.scalar_size = sizeof(float);
#else
    value.scalar_size = sizeof(double);
#endif
    value.abi_tag =
#ifdef BAD_TAG
        "wrong.oam.tag";
#else
        SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V2;
#endif
    value.artifact_id =
#ifdef BAD_ARTIFACT
        "other-oam-host-v2";
#else
        "fixture-oam-host-v2";
#endif
    value.generation_fingerprint = "sha256:generation";
    value.semantic_fingerprint = "sha256:semantic";
    value.structure_fingerprint = "sha256:structure";
    value.layers = layers;
#ifdef BAD_LAYER_COUNT
    value.layer_count = 1;
#endif
#ifdef BAD_DIMENSION
    static SymmetrixJitOAMHostLayerV2 bad_layers[2];
    bad_layers[0] = layers[0];
    bad_layers[1] = layers[1];
    bad_layers[1].source_dimension = 99;
    value.layers = bad_layers;
#endif
#ifdef BAD_CHANNEL_TILE
    static SymmetrixJitOAMHostLayerV2 bad_tile_layers[2];
    bad_tile_layers[0] = layers[0];
    bad_tile_layers[1] = layers[1];
    bad_tile_layers[1].channel_tile = 0;
    value.layers = bad_tile_layers;
#endif
#ifdef BAD_CAPABILITY_BIT
    value.capabilities |= 1u << 5;
#endif
#ifdef MISSING_CAPABILITY
    value.capabilities = SYMMETRIX_JIT_OAM_HOST_RECEIVER_FORWARD_V2;
#endif
#ifdef MISSING_LANE_MAJOR
    value.capabilities =
        SYMMETRIX_JIT_OAM_HOST_RECEIVER_FORWARD_V2
        | SYMMETRIX_JIT_OAM_HOST_SOURCE_REVERSE_V2;
#endif
    value.receiver_forward_owners = receiver_forward_owners;
    value.source_reverse_owners = source_reverse_owners;
#ifdef BAD_RECEIVER_OWNER
    static const SymmetrixJitOAMHostReceiverForwardOwnerV2
        bad_receiver_forward_owners[] = {
            &receiver_forward_owner, nullptr};
    value.receiver_forward_owners = bad_receiver_forward_owners;
#endif
    return value;
}();

extern "C" SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
const SymmetrixJitOAMHostPluginV2*
symmetrix_jit_oam_host_plugin_query_v2()
{
    return &descriptor;
}
"""


def _v2_harness_source():
    return r"""
#include "oam_jit_host_plugin.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3) return 64;
    symmetrix::execution::OAMHostPluginExpectationV2 expected;
    expected.artifact_id = "fixture-oam-host-v2";
    expected.generation_fingerprint = "sha256:generation";
    expected.semantic_fingerprint = "sha256:semantic";
    expected.structure_fingerprint = "sha256:structure";
    expected.scalar_size = sizeof(double);
    // The production receiver requires the lane-major bit too
    // (oam_kokkos.cpp sets required_capabilities to these three bits),
    // so the harness must demand it for the negative case below to be real.
    expected.required_capabilities =
        SYMMETRIX_JIT_OAM_HOST_FULL_FORWARD_V3
        | SYMMETRIX_JIT_OAM_HOST_FULL_REVERSE_V3
        | SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V3;
    expected.schedule_identity = "sha256:schedule";
    expected.host_target = "host-isa-x86-64-v3";
    expected.full_reverse_owner_schedule = "source-owner-v1";
    expected.full_reverse_reduction_policy = "owner-local-deterministic-v1";
    expected.layers = {
        {8, 4, 6, 10, 3, 2, 5},
        {10, 4, 8, 12, 5, 4, 7}};
    const std::string plugin_path = argv[2];
    const std::size_t separator = plugin_path.find_last_of('/');
    const std::string plugin_directory = separator == std::string::npos
        ? std::string(".") : plugin_path.substr(0, separator);
    try {
        auto plugin = symmetrix::execution::OAMHostPluginV2Handle::load(
            argv[2], expected);
        if (std::string(argv[1]) != "accept") return 1;
        if (!plugin) return 1;
        return plugin->path() == plugin_path
                && plugin->directory() == plugin_directory
                && plugin->descriptor().layer_count == 2
                && plugin->descriptor().capabilities
                    == (SYMMETRIX_JIT_OAM_HOST_RECEIVER_FORWARD_V2
                        | SYMMETRIX_JIT_OAM_HOST_SOURCE_REVERSE_V2
                        | SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V2)
            ? 0 : 1;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        const std::string mode = argv[1];
        if ((mode == "reject-stale"
                && message.find("stale OAM host artifact ABI") != std::string::npos)
            || (mode == "reject-abi"
                && message.find("unsupported ABI version") != std::string::npos)
            || (mode == "reject-tag"
                && message.find("ABI tag does not match") != std::string::npos)
            || (mode == "reject-scalar"
                && message.find("scalar size does not match") != std::string::npos)
            || (mode == "reject-capability-bit"
                && message.find("unknown capability bits are set") != std::string::npos)
            || (mode == "reject-required-capability"
                && message.find("required capabilities are missing") != std::string::npos)
            || (mode == "reject-lane-major"
                && message.find("required capabilities are missing") != std::string::npos)
            || (mode == "reject-receiver-owner"
                && message.find("layer 1 receiver forward owner is null") != std::string::npos)
            || (mode == "reject-layer-count"
                && message.find("layer count does not match") != std::string::npos)
            || (mode == "reject-dimension"
                && message.find("layer 1 source dimension does not match") != std::string::npos)
            || (mode == "reject-channel-tile"
                && message.find("layer 1 channel tile is invalid") != std::string::npos)
            || (mode == "reject-artifact"
                && message.find("artifact id does not match") != std::string::npos))
            return 0;
        std::cerr << message << '\n';
        return 2;
    }
}
"""


def test_oam_host_loader_rejects_stale_receiver_contract_with_regeneration(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the OAM host loader test")
    plugin_source = tmp_path / "oam_host_plugin_v2.cpp"
    harness_source = tmp_path / "oam_host_harness_v2.cpp"
    plugin_source.write_text(_v2_plugin_source(), encoding="ascii")
    harness_source.write_text(_v2_harness_source(), encoding="ascii")
    common = ("-std=c++20", "-O2", "-I", SOURCE, "-I", PACKAGE_SOURCE)
    variants = (("reject-stale", (), tmp_path / "stale_v2.so"),)
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
    harness = tmp_path / "oam_host_harness_v2"
    _compile(
        compiler,
        *common,
        harness_source,
        SOURCE / "oam_jit_host_plugin.cpp",
        "-ldl",
        "-o",
        harness,
    )
    for mode, _, plugin in variants:
        subprocess.run(
            [harness, mode, plugin], check=True, capture_output=True, text=True
        )
