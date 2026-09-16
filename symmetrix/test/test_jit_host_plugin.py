import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "libsymmetrix" / "source"
PACKAGE_SOURCE = REPOSITORY / "symmetrix" / "source" / "symmetrix"
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the Execution host-plugin loader is POSIX-only"
)


def _compiler():
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the host-plugin loader test")
    return compiler


def _compile(compiler, *arguments):
    subprocess.run(
        [compiler, *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def host_plugin_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("execution-host-plugin")
    plugin_source = directory / "fixture_plugin.cpp"
    plugin_source.write_text(
        r"""
#include "jit_host_plugin_abi.h"

extern "C" void fixture_forward(
    const SymmetrixJitHostR1ForwardArgsV1* args,
    int32_t receiver,
    int32_t channel)
{
    args->output[receiver*4+channel] = float(receiver*10+channel);
}

static const SymmetrixJitHostPluginV1 descriptor = {
#ifdef INVALID_ABI
    99u,
#else
    SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION,
#endif
    sizeof(SymmetrixJitHostPluginV1),
    sizeof(void*),
    SYMMETRIX_JIT_HOST_PLUGIN_BYTE_ORDER,
    SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V1,
    0u,
    SYMMETRIX_JIT_HOST_PLUGIN_ABI_TAG,
    "fixture-r1-v1",
    "sha256:contract",
    "sha256:semantic",
    "sha256:structure",
    4,
    3,
    2,
    0,
#ifdef MISSING_FUNCTION
    nullptr,
#else
    fixture_forward,
#endif
    nullptr,
    nullptr,
    nullptr,
};

#ifndef MISSING_QUERY
extern "C" SYMMETRIX_JIT_HOST_PLUGIN_EXPORT
const SymmetrixJitHostPluginV1*
symmetrix_jit_host_plugin_query_v1()
{
#ifdef NULL_DESCRIPTOR
    return nullptr;
#else
    return &descriptor;
#endif
}
#endif
"""
    )
    harness_source = directory / "loader_harness.cpp"
    harness_source.write_text(
        r"""
#include "jit_host_plugin.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

int main(int argc, char** argv)
{
    if (argc != 3)
        return 64;
    const std::string mode = argv[1];
    symmetrix::execution::HostPluginExpectation expected;
    expected.artifact_id = "fixture-r1-v1";
    expected.contract_fingerprint = "sha256:contract";
    expected.semantic_fingerprint = "sha256:semantic";
    expected.structure_fingerprint = "sha256:structure";
    expected.channels = mode == "extent-mismatch" ? 5 : 4;
    expected.embedding = 3;
    expected.edge_l_max = 2;
    expected.source_l_max = 0;
    expected.required_capabilities =
        SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V1;
    if (mode == "missing-expectation")
        expected.contract_fingerprint = {};

    try {
        auto plugin = symmetrix::execution::HostPlugin::load(argv[2], expected);
        if (mode != "good")
            return 1;
        if (!plugin || plugin.path() != argv[2])
            return 2;
        std::array<float, 12> output{};
        SymmetrixJitHostR1ForwardArgsV1 args{};
        args.struct_size = sizeof(args);
        args.output = output.data();
        plugin.descriptor().r1_forward_owner(&args, 2, 3);
        if (output[11] != 23.0f)
            return 3;
        auto moved = std::move(plugin);
        if (plugin || !moved)
            return 4;
        symmetrix::execution::HostPlugin assigned;
        assigned = std::move(moved);
        if (moved || !assigned)
            return 5;
        return 0;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (mode == "extent-mismatch"
            && message.find("channel count does not match") != std::string::npos)
            return 0;
        if (mode == "invalid-abi"
            && message.find("unsupported ABI version") != std::string::npos)
            return 0;
        if (mode == "missing-query"
            && message.find("symmetrix_jit_host_plugin_query_v1")
                != std::string::npos)
            return 0;
        if (mode == "null-descriptor"
            && message.find("null descriptor") != std::string::npos)
            return 0;
        if (mode == "missing-function"
            && message.find("function pointer disagree") != std::string::npos)
            return 0;
        if (mode == "missing-expectation"
            && message.find("requires a contract fingerprint")
                != std::string::npos)
            return 0;
        std::cerr << message << '\n';
        return 6;
    }
}
"""
    )

    compiler = _compiler()
    good_plugin = directory / "fixture_plugin.so"
    invalid_plugin = directory / "fixture_plugin_invalid_abi.so"
    missing_query_plugin = directory / "fixture_plugin_missing_query.so"
    null_descriptor_plugin = directory / "fixture_plugin_null_descriptor.so"
    missing_function_plugin = directory / "fixture_plugin_missing_function.so"
    harness = directory / "loader_harness"
    common = (
        "-std=c++20",
        "-O2",
        "-I",
        SOURCE,
        "-I",
        PACKAGE_SOURCE,
    )
    _compile(
        compiler,
        *common,
        "-shared",
        "-fPIC",
        plugin_source,
        "-o",
        good_plugin,
    )
    _compile(
        compiler,
        *common,
        "-shared",
        "-fPIC",
        "-DINVALID_ABI=1",
        plugin_source,
        "-o",
        invalid_plugin,
    )
    for define, output in (
        ("MISSING_QUERY=1", missing_query_plugin),
        ("NULL_DESCRIPTOR=1", null_descriptor_plugin),
        ("MISSING_FUNCTION=1", missing_function_plugin),
    ):
        _compile(
            compiler,
            *common,
            "-shared",
            "-fPIC",
            f"-D{define}",
            plugin_source,
            "-o",
            output,
        )
    _compile(
        compiler,
        *common,
        harness_source,
        SOURCE / "jit_host_plugin.cpp",
        "-ldl",
        "-o",
        harness,
    )
    return (
        harness,
        good_plugin,
        invalid_plugin,
        missing_query_plugin,
        null_descriptor_plugin,
        missing_function_plugin,
    )


@pytest.mark.parametrize(
    ("mode", "plugin_index"),
    (
        ("good", 1),
        ("extent-mismatch", 1),
        ("invalid-abi", 2),
        ("missing-query", 3),
        ("null-descriptor", 4),
        ("missing-function", 5),
        ("missing-expectation", 1),
    ),
)
def test_jit_host_plugin_loader(host_plugin_fixture, mode, plugin_index):
    harness = host_plugin_fixture[0]
    plugin = host_plugin_fixture[plugin_index]
    subprocess.run(
        [harness, mode, plugin],
        check=True,
        capture_output=True,
        text=True,
    )
