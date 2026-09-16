import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "libsymmetrix" / "source"
PACKAGE_SOURCE = REPOSITORY / "symmetrix" / "source" / "symmetrix"
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the Execution MH-1 CUDA-plugin loader is POSIX-only"
)


def _compile(compiler, *arguments):
    subprocess.run(
        [compiler, *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_mh1_cuda_loader_rejects_pre_active_receiver_plugin(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the CUDA-plugin loader test")

    plugin_source = tmp_path / "fixture_plugin.cpp"
    plugin_source.write_text(
        r"""
#include "jit_mh1_cuda_plugin_abi.h"

#include <cstddef>

static_assert(offsetof(
    SymmetrixJitMH1CudaForwardArgsV3, active_receiver_count)
    == 3 * sizeof(uint32_t));
static_assert(offsetof(SymmetrixJitMH1CudaForwardArgsV3, active_receivers)
    == offsetof(SymmetrixJitMH1CudaForwardArgsV3, source_indices)
        + sizeof(const int32_t*));

extern "C" int32_t fixture_forward(
    const SymmetrixJitMH1CudaForwardArgsV3*, void*, int32_t)
{
    return 0;
}

extern "C" int32_t fixture_reverse(
    const SymmetrixJitMH1CudaReverseArgsV3*, void*, int32_t, int32_t)
{
    return 0;
}

extern "C" int32_t fixture_conditioning_forward(
    const SymmetrixJitMH1CudaConditioningForwardArgsV3*, void*, int32_t)
{
    return 0;
}

extern "C" int32_t fixture_conditioning_reverse(
    const SymmetrixJitMH1CudaConditioningReverseArgsV3*, void*, int32_t)
{
    return 0;
}

static const SymmetrixJitMH1CudaPluginV3 descriptor = {
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_ABI_VERSION,
    sizeof(SymmetrixJitMH1CudaPluginV3),
    sizeof(void*),
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER,
    SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V3
        | SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V3
        | SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V3
        | SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V3
        | SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V3
        | SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V3
        | SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V3
        | SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V3
#ifdef ACTIVE_RECEIVER_CAPABILITY
        | SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V3
#endif
        ,
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS,
    sizeof(float),
    0u,
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_ABI_TAG,
    "fixture-mh1-cuda-v3",
    "sha256:generation",
    "sha256:semantic",
    "sha256:structure",
    80,
    0,
    {
        {sizeof(SymmetrixJitMH1CudaInteractionV3), 0u,
            8, 4, 6, 2, 3, 2, 4, 1, 128, 128, 128, 2},
        {sizeof(SymmetrixJitMH1CudaInteractionV3), 0u,
            12, 9, 10, 4, 5, 2, 6, 2, 128, 128, 128, 2},
    },
    {
        {sizeof(SymmetrixJitMH1CudaInteractionLaunchesV3),
            0u, 0u, 0u, &fixture_forward, &fixture_reverse},
        {sizeof(SymmetrixJitMH1CudaInteractionLaunchesV3),
            1u, 0u, 0u, &fixture_forward, &fixture_reverse},
    },
    {
        {sizeof(SymmetrixJitMH1CudaConditioningLaunchesV3),
            0u, 0u, 0u, &fixture_conditioning_forward,
#ifdef MISSING_CONDITIONING_REVERSE
            nullptr},
#else
            &fixture_conditioning_reverse},
#endif
        {sizeof(SymmetrixJitMH1CudaConditioningLaunchesV3),
            1u, 0u, 0u, &fixture_conditioning_forward,
            &fixture_conditioning_reverse},
    },
};

extern "C" SYMMETRIX_JIT_MH1_CUDA_PLUGIN_EXPORT
const SymmetrixJitMH1CudaPluginV3*
symmetrix_jit_mh1_cuda_plugin_query_v3()
{
    return &descriptor;
}
""",
        encoding="ascii",
    )

    harness_source = tmp_path / "loader_harness.cpp"
    harness_source.write_text(
        r"""
#include "jit_mh1_cuda_plugin.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3)
        return 64;
    const std::string mode = argv[1];
    symmetrix::execution::MH1CudaPluginExpectation expected;
    expected.artifact_id = "fixture-mh1-cuda-v3";
    expected.generation_fingerprint = "sha256:generation";
    expected.semantic_fingerprint = "sha256:semantic";
    expected.structure_fingerprint = "sha256:structure";
    expected.target_compute_capability = 80;
    expected.max_threads_per_block = 256;
    expected.required_capabilities =
        SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V3
        | SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V3
        | SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V3
        | SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V3
        | SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V3
        | SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V3
        | SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V3
        | SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V3
        | SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V3;
    expected.interactions[0] = {8, 4, 6, 2, 3, 2, 4, 1};
    expected.interactions[1] = {12, 9, 10, 4, 5, 2, 6, 2};

    try {
        auto plugin = symmetrix::execution::MH1CudaPlugin::load(argv[2], expected);
        return mode == "accept" && plugin ? 0 : 1;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (mode == "reject"
            && message.find("launcher capabilities do not match")
                != std::string::npos)
            return 0;
        if (mode == "reject-conditioning"
            && message.find(
                "conditioning capability and function pointers disagree")
                != std::string::npos)
            return 0;
        std::cerr << message << '\n';
        return 2;
    }
}
""",
        encoding="ascii",
    )

    common = (
        "-std=c++20",
        "-O2",
        "-I",
        SOURCE,
        "-I",
        PACKAGE_SOURCE,
    )
    legacy_plugin = tmp_path / "legacy_plugin.so"
    current_plugin = tmp_path / "current_plugin.so"
    malformed_plugin = tmp_path / "malformed_plugin.so"
    _compile(
        compiler,
        *common,
        "-shared",
        "-fPIC",
        plugin_source,
        "-o",
        legacy_plugin,
    )
    _compile(
        compiler,
        *common,
        "-DACTIVE_RECEIVER_CAPABILITY=1",
        "-shared",
        "-fPIC",
        plugin_source,
        "-o",
        current_plugin,
    )
    _compile(
        compiler,
        *common,
        "-DACTIVE_RECEIVER_CAPABILITY=1",
        "-DMISSING_CONDITIONING_REVERSE=1",
        "-shared",
        "-fPIC",
        plugin_source,
        "-o",
        malformed_plugin,
    )

    harness = tmp_path / "loader_harness"
    _compile(
        compiler,
        *common,
        harness_source,
        SOURCE / "jit_mh1_cuda_plugin.cpp",
        "-ldl",
        "-o",
        harness,
    )
    subprocess.run(
        [harness, "accept", current_plugin],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [harness, "reject", legacy_plugin],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [harness, "reject-conditioning", malformed_plugin],
        check=True,
        capture_output=True,
        text=True,
    )
