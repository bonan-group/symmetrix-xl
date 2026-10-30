import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "libsymmetrix/source"
INCLUDE = ROOT / "symmetrix/source/symmetrix"


def test_split_phase_native_admission_rejects_retired_and_mismatched_descriptors(
    tmp_path,
):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("a C++ compiler is required for split-phase admission tests")
    probe = tmp_path / "split_phase_admission.cpp"
    probe.write_text(
        r"""
#include "mace_nonlinear_split_phase_admission.hpp"
#include <cstdint>
#include <exception>
#include <string>

using namespace symmetrix::execution;

int main(int argc, char** argv) {
    const std::string mutation = argc > 1 ? argv[1] : "valid";
    MH1SplitPhaseExpectation expected{
        "artifact", "sha256:generation", "sha256:semantic",
        "sha256:structure", "sha256:layout", 2u, 4u, "cuda", "sm_120"};
    SymmetrixJitMH1HostPluginV7 host{};
    host.abi_version = SYMMETRIX_JIT_MH1_HOST_PLUGIN_V7_ABI_VERSION;
    host.struct_size = sizeof(host);
    host.pointer_size = sizeof(void*);
    host.byte_order = SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER;
    host.capabilities = SYMMETRIX_JIT_MH1_HOST_EXTERNAL_PRODUCT_V7;
    host.interaction_count = 2u;
    host.scalar_size = 4u;
    host.abi_tag = SYMMETRIX_JIT_MH1_HOST_PLUGIN_V7_ABI_TAG;
    host.artifact_id = "artifact";
    host.generation_fingerprint = "sha256:generation";
    host.semantic_fingerprint = "sha256:semantic";
    host.structure_fingerprint = "sha256:structure";
    host.runtime_layout_fingerprint = "sha256:layout";
    host.node_programs = reinterpret_cast<const SymmetrixJitMH1HostNodeProgramV7*>(1);

    SymmetrixJitMH1CudaPluginV5 cuda{};
    cuda.abi_version = SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V5_ABI_VERSION;
    cuda.struct_size = sizeof(cuda);
    cuda.pointer_size = sizeof(void*);
    cuda.byte_order = SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER;
    cuda.capabilities = SYMMETRIX_JIT_MH1_CUDA_EXTERNAL_PRODUCT_V5
        | SYMMETRIX_JIT_MH1_CUDA_NODE_PROGRAM_METADATA_V5;
    cuda.interaction_count = 2u;
    cuda.scalar_size = 4u;
    cuda.abi_tag = SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V5_ABI_TAG;
    cuda.artifact_id = "artifact";
    cuda.generation_fingerprint = "sha256:generation";
    cuda.semantic_fingerprint = "sha256:semantic";
    cuda.structure_fingerprint = "sha256:structure";
    cuda.runtime_layout_fingerprint = "sha256:layout";
    cuda.target_backend = "cuda";
    cuda.target_architecture = "sm_120";
    SymmetrixJitMH1CudaNodeProgramV5 cuda_nodes[2]{};
    for (std::uint32_t interaction = 0; interaction < 2; ++interaction) {
        auto& node = cuda_nodes[interaction];
        node.struct_size = sizeof(node);
        node.interaction = interaction;
        node.element_count = 2;
        node.input_dimension = 4;
        node.up_dimension = 8;
        node.contraction_input_dimension = 8;
        node.contraction_output_dimension = 4;
        node.output_dimension = 4;
        node.node_arena_dimension = 24;
        node.linear_parameter_count = 16;
        node.readout_parameter_count = 8;
    }
    cuda.node_programs = cuda_nodes;

    MH1FullModuleDescriptor module{
        5u, "symmetrix.jit.mh1.cuda-module/5", "artifact",
        "sha256:generation", "sha256:semantic", "sha256:structure",
        "sha256:layout", "cuda", "sm_120", 2u, 4u};
    MH1SplitPhaseExpectation hip_expected{
        "artifact", "sha256:generation", "sha256:semantic",
        "sha256:structure", "sha256:layout", 2u, 4u, "hip", "gfx1100"};
    MH1FullModuleDescriptor hip_module{
        5u, "symmetrix.jit.mh1.hip-module/5", "artifact",
        "sha256:generation", "sha256:semantic", "sha256:structure",
        "sha256:layout", "hip", "gfx1100", 2u, 4u};
    MH1ProductModuleExpectation product_expected{
        1, 3, "M0", "cuda", "sm_120", "artifact", "sha256:structure",
        4u, 7, 9, 4, 3, 12};
    MH1ProductModuleCandidate product{
        1, "M0", "cuda", "sm_120", "artifact", "sha256:structure",
        4u, 7, 9, 4, 3, 12};
    try {
        if (mutation.rfind("host-abi-", 0) == 0)
            host.abi_version = static_cast<std::uint32_t>(std::stoi(mutation.substr(9)));
        else if (mutation.rfind("cuda-abi-", 0) == 0)
            cuda.abi_version = static_cast<std::uint32_t>(std::stoi(mutation.substr(9)));
        else if (mutation.rfind("module-abi-", 0) == 0)
            module.abi_version = static_cast<std::uint32_t>(std::stoi(mutation.substr(11)));
        else if (mutation.rfind("hip-module-abi-", 0) == 0)
            hip_module.abi_version = static_cast<std::uint32_t>(std::stoi(mutation.substr(15)));
        else if (mutation == "precision") host.scalar_size = 8u;
        else if (mutation == "dimensions") host.interaction_count = 3u;
        else if (mutation == "identity") cuda.structure_fingerprint = "sha256:wrong";
        else if (mutation == "target") cuda.target_architecture = "sm_90";
        else if (mutation == "backend") cuda.target_backend = "hip";
        else if (mutation == "cuda-tag")
            cuda.abi_tag = SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_TAG;
        else if (mutation == "cuda-capability") cuda.capabilities = 0u;
        else if (mutation == "cuda-legacy-capability")
            cuda.capabilities |= SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V4;
        else if (mutation == "cuda-null-nodes") cuda.node_programs = nullptr;
        else if (mutation == "cuda-node-size") cuda_nodes[0].struct_size = 0u;
        else if (mutation == "module-tag")
            module.abi_tag = "symmetrix.jit.mh1.cuda-module/4";
        else if (mutation == "product-layer") product.layer = 0;
        else if (mutation == "product-kind") product.kind = "M1";
        else if (mutation == "product-topology") product.input_components = 10;
        else if (mutation == "product-precision") product.scalar_size = 8u;
        else if (mutation == "product-target") product.target = "sm_90";
        else if (mutation == "product-identity") product.artifact_id = "wrong";
        else if (mutation == "product-backend") product.backend = "hip";
        validate_mh1_split_phase_host_descriptor(host, expected);
        validate_mh1_split_phase_cuda_descriptor(cuda, expected);
        validate_mh1_full_module_descriptor(module, expected);
        validate_mh1_full_module_descriptor(hip_module, hip_expected);
        validate_mh1_product_module_candidate(product, product_expected);
        const std::string plan_schema = mutation.rfind("plan-", 0) == 0
            ? "symmetrix.jit.mh1.cuda-launch-plan/" + mutation.substr(5)
            : "symmetrix.jit.mh1.cuda-launch-plan/6";
        validate_mh1_launch_plan_schema(plan_schema, "cuda");
        return mutation == "valid" ? 0 : 4;
    } catch (const std::exception&) {
        return mutation == "valid" ? 3 : 0;
    }
}
"""
    )
    executable = tmp_path / "split_phase_admission"
    completed = subprocess.run(
        [
            cxx,
            "-std=c++20",
            str(probe),
            str(SOURCE / "mace_nonlinear_split_phase_admission.cpp"),
            f"-I{SOURCE}",
            f"-I{INCLUDE}",
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert subprocess.run([executable, "valid"], check=False).returncode == 0
    mutations = [
        *(f"host-abi-{version}" for version in range(1, 7)),
        *(f"cuda-abi-{version}" for version in range(1, 5)),
        "module-abi-1",
        "module-abi-2",
        "module-abi-3",
        "module-abi-4",
        "hip-module-abi-1",
        "hip-module-abi-2",
        "hip-module-abi-3",
        "hip-module-abi-4",
        "precision",
        "dimensions",
        "identity",
        "target",
        "backend",
        "cuda-tag",
        "cuda-capability",
        "cuda-legacy-capability",
        "cuda-null-nodes",
        "cuda-node-size",
        "module-tag",
        "product-layer",
        "product-kind",
        "product-topology",
        "product-precision",
        "product-target",
        "product-identity",
        "product-backend",
        "plan-1",
        "plan-2",
        "plan-3",
        "plan-4",
        "plan-5",
    ]
    for mutation in mutations:
        assert subprocess.run([executable, mutation], check=False).returncode == 0
