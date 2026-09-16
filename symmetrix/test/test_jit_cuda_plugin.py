import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "libsymmetrix" / "source"
PACKAGE_SOURCE = REPOSITORY / "symmetrix" / "source" / "symmetrix"
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the Execution CUDA-plugin loader is POSIX-only"
)


def _compiler():
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the CUDA-plugin loader test")
    return compiler


def _compile(compiler, *arguments):
    subprocess.run(
        [compiler, *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def cuda_plugin_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("execution-cuda-plugin")
    plugin_source = directory / "fixture_plugin.cpp"
    plugin_source.write_text(
        r"""
#include "jit_cuda_plugin_abi.h"

extern "C" int32_t fixture_forward(
    const SymmetrixJitCudaR1ForwardArgsV1* args,
    void* stream,
    int32_t persistent_blocks)
{
    if (args == nullptr || args->output == nullptr
        || reinterpret_cast<uintptr_t>(stream) != uintptr_t{0x1234}
        || persistent_blocks != 7)
        return -1;
    args->output[0] = 42.0f;
    return 17;
}

extern "C" int32_t fixture_reverse(
    const SymmetrixJitCudaR1SourceArgsV1* source_args,
    const SymmetrixJitCudaR1EdgeArgsV1* edge_args,
    void* stream,
    int32_t persistent_blocks)
{
    if (source_args == nullptr || edge_args == nullptr
        || source_args->struct_size != sizeof(*source_args)
        || edge_args->struct_size != sizeof(*edge_args)
        || reinterpret_cast<uintptr_t>(stream) != uintptr_t{0x5678}
        || persistent_blocks != 9)
        return -1;
    return 19;
}

static const SymmetrixJitCudaPluginV1 descriptor = {
#ifdef INVALID_ABI
    99u,
#else
    SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION,
#endif
    sizeof(SymmetrixJitCudaPluginV1),
    sizeof(void*),
    SYMMETRIX_JIT_CUDA_PLUGIN_BYTE_ORDER,
#ifdef MISSING_CAPABILITY
    SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V1,
#else
    SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V1
        | SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V1,
#endif
    0u,
    SYMMETRIX_JIT_CUDA_PLUGIN_ABI_TAG,
    "fixture-cuda-r1-v1",
    "sha256:contract",
    "sha256:semantic",
    "sha256:structure",
    4,
    3,
    2,
    0,
    80,
    256,
    256,
    256,
    &fixture_forward,
#ifdef MISSING_CALLBACK
    nullptr,
#else
    &fixture_reverse,
#endif
};

#ifndef MISSING_QUERY
extern "C" SYMMETRIX_JIT_CUDA_PLUGIN_EXPORT
const SymmetrixJitCudaPluginV1*
symmetrix_jit_cuda_plugin_query_v1()
{
#ifdef NULL_DESCRIPTOR
    return nullptr;
#else
    return &descriptor;
#endif
}
#endif
""",
        encoding="utf-8",
    )
    harness_source = directory / "loader_harness.cpp"
    harness_source.write_text(
        r"""
#include "jit_cuda_plugin.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

int main(int argc, char** argv)
{
    if (argc != 3)
        return 64;
    const std::string mode = argv[1];
    symmetrix::execution::CudaPluginExpectation expected;
    expected.artifact_id = "fixture-cuda-r1-v1";
    expected.contract_fingerprint = "sha256:contract";
    expected.semantic_fingerprint = "sha256:semantic";
    expected.structure_fingerprint = "sha256:structure";
    expected.channels = mode == "extent-mismatch" ? 5 : 4;
    expected.embedding = 3;
    expected.edge_l_max = 2;
    expected.source_l_max = 0;
    expected.target_compute_capability =
        mode == "compute-capability-mismatch" ? 90 : 80;
    expected.max_threads_per_block =
        mode == "thread-block-limit" ? 128 : 256;
    expected.required_capabilities =
        SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V1
        | SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V1;

    if (mode == "missing-artifact-expectation")
        expected.artifact_id = {};
    else if (mode == "missing-contract-expectation")
        expected.contract_fingerprint = {};
    else if (mode == "missing-extent-expectation")
        expected.target_compute_capability = 0;
    else if (mode == "missing-capability-expectation")
        expected.required_capabilities = 0;

    try {
        if (mode == "good") {
            for (int iteration = 0; iteration < 8; ++iteration) {
                auto repeated =
                    symmetrix::execution::CudaPlugin::load(argv[2], expected);
                if (!repeated || repeated.path() != argv[2])
                    return 8;
            }
        }
        auto plugin = symmetrix::execution::CudaPlugin::load(argv[2], expected);
        if (mode != "good")
            return 1;
        if (!plugin || plugin.path() != argv[2])
            return 2;

        std::array<float, 1> output{};
        SymmetrixJitCudaR1ForwardArgsV1 forward_args{};
        forward_args.struct_size = sizeof(forward_args);
        forward_args.output = output.data();
        const auto& descriptor = plugin.descriptor();
        if (descriptor.r1_forward_launch(
                &forward_args,
                reinterpret_cast<void*>(uintptr_t{0x1234}),
                7) != 17
            || output[0] != 42.0f)
            return 3;

        SymmetrixJitCudaR1SourceArgsV1 source_args{};
        source_args.struct_size = sizeof(source_args);
        SymmetrixJitCudaR1EdgeArgsV1 edge_args{};
        edge_args.struct_size = sizeof(edge_args);
        if (descriptor.r1_coordinate_reverse_launch(
                &source_args,
                &edge_args,
                reinterpret_cast<void*>(uintptr_t{0x5678}),
                9) != 19)
            return 4;

        auto moved = std::move(plugin);
        if (plugin || !moved)
            return 5;
        symmetrix::execution::CudaPlugin assigned;
        assigned = std::move(moved);
        if (moved || !assigned)
            return 6;
        return 0;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (mode == "extent-mismatch"
            && message.find("channel count does not match") != std::string::npos)
            return 0;
        if (mode == "compute-capability-mismatch"
            && message.find("target compute capability does not match")
                != std::string::npos)
            return 0;
        if (mode == "thread-block-limit"
            && message.find("thread-block size is invalid") != std::string::npos)
            return 0;
        if (mode == "invalid-abi"
            && message.find("unsupported ABI version") != std::string::npos)
            return 0;
        if (mode == "missing-query"
            && message.find("symmetrix_jit_cuda_plugin_query_v1")
                != std::string::npos)
            return 0;
        if (mode == "null-descriptor"
            && message.find("null descriptor") != std::string::npos)
            return 0;
        if (mode == "missing-callback"
            && message.find("function pointers disagree") != std::string::npos)
            return 0;
        if (mode == "missing-capability"
            && message.find("launcher capabilities do not match")
                != std::string::npos)
            return 0;
        if (mode == "missing-artifact-expectation"
            && message.find("requires an artifact id") != std::string::npos)
            return 0;
        if (mode == "missing-contract-expectation"
            && message.find("requires a contract fingerprint")
                != std::string::npos)
            return 0;
        if (mode == "missing-extent-expectation"
            && message.find("requires exact model and device extents")
                != std::string::npos)
            return 0;
        if (mode == "missing-capability-expectation"
            && message.find("requires exact known capabilities")
                != std::string::npos)
            return 0;
        std::cerr << message << '\n';
        return 7;
    }
}
""",
        encoding="utf-8",
    )

    compiler = _compiler()
    common = (
        "-std=c++20",
        "-O2",
        "-I",
        SOURCE,
        "-I",
        PACKAGE_SOURCE,
    )
    plugins = {}
    for name, define in (
        ("good", None),
        ("invalid-abi", "INVALID_ABI=1"),
        ("missing-query", "MISSING_QUERY=1"),
        ("null-descriptor", "NULL_DESCRIPTOR=1"),
        ("missing-callback", "MISSING_CALLBACK=1"),
        ("missing-capability", "MISSING_CAPABILITY=1"),
    ):
        output = directory / f"fixture_plugin_{name}.so"
        arguments = [*common, "-shared", "-fPIC"]
        if define is not None:
            arguments.append(f"-D{define}")
        arguments.extend((plugin_source, "-o", output))
        _compile(compiler, *arguments)
        plugins[name] = output

    harness = directory / "loader_harness"
    _compile(
        compiler,
        *common,
        harness_source,
        SOURCE / "jit_cuda_plugin.cpp",
        "-ldl",
        "-o",
        harness,
    )
    return harness, plugins


@pytest.mark.parametrize(
    ("mode", "plugin_name"),
    (
        ("good", "good"),
        ("extent-mismatch", "good"),
        ("compute-capability-mismatch", "good"),
        ("thread-block-limit", "good"),
        ("invalid-abi", "invalid-abi"),
        ("missing-query", "missing-query"),
        ("null-descriptor", "null-descriptor"),
        ("missing-callback", "missing-callback"),
        ("missing-capability", "missing-capability"),
        ("missing-artifact-expectation", "good"),
        ("missing-contract-expectation", "good"),
        ("missing-extent-expectation", "good"),
        ("missing-capability-expectation", "good"),
    ),
)
def test_jit_cuda_plugin_loader(cuda_plugin_fixture, mode, plugin_name):
    harness, plugins = cuda_plugin_fixture
    subprocess.run(
        [harness, mode, plugins[plugin_name]],
        check=True,
        capture_output=True,
        text=True,
    )
