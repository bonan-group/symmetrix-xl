import json
from pathlib import Path

import pytest

CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "jit_r1_fixture_c4_e3_l0_contract.json.in"
)


def _assert_artifact(result, compiler, build_backend, suffix):
    assert result.status in ("built", "cached"), result.diagnostics
    assert result.artifact_path is not None
    assert result.artifact_path.suffix == suffix
    assert result.manifest is not None
    key_inputs = result.manifest["key_inputs"]
    assert key_inputs["compiler"]["kind"] == compiler
    assert key_inputs["build"]["backend"] == build_backend
    assert result.manifest["artifact"] == result.artifact_path.name


@pytest.mark.cuda
@pytest.mark.device_compile
@pytest.mark.jit
@pytest.mark.parametrize("compiler", ("nvrtc", "nvcc"))
def test_cuda_forced_compiler_produces_expected_artifact(
    accelerator_capabilities, tmp_path, monkeypatch, compiler
):
    from symmetrix import jit
    from symmetrix.jit_codegen import (
        render_jit_r1_cuda_module,
        render_jit_r1_cuda_plugin,
        jit_r1_cuda_plugin_metadata,
    )

    target = accelerator_capabilities
    assert target["backend"] == "cuda"
    compute_capability = target.get("compute_capability_code")
    assert isinstance(compute_capability, int)
    contract = json.loads(CONTRACT_PATH.read_text())
    metadata = jit_r1_cuda_plugin_metadata(contract, compute_capability)

    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", compiler)
    assert jit.execution_cuda_jit_backend_request() == compiler
    if compiler == "nvcc":
        with pytest.warns(FutureWarning, match="runtime nvcc"):
            assert jit.select_execution_cuda_jit_backend() == compiler
    else:
        assert jit.select_execution_cuda_jit_backend() == compiler

    if compiler == "nvrtc":
        source = render_jit_r1_cuda_module(contract, compute_capability)
        result = jit.prepare_nvrtc_jit_artifact(
            source,
            abi={"tag": metadata["abi"], "version": metadata["abi_version"]},
            build={"generator": "qualification.cuda-nvrtc", "contract": metadata},
            compute_capability=target,
            cache_root=tmp_path / "nvrtc-cache",
        )
        _assert_artifact(result, "nvrtc", "nvrtc", ".cubin")
        assert "-O3" not in result.manifest["compiler_options"]
    else:
        source = render_jit_r1_cuda_plugin(contract, compute_capability)
        result = jit.prepare_execution_cuda_jit_artifact(
            source,
            abi={"tag": metadata["abi"], "version": metadata["abi_version"]},
            build={"generator": "qualification.cuda-nvcc", "contract": metadata},
            compute_capability=target,
            cache_root=tmp_path / "nvcc-cache",
        )
        _assert_artifact(result, "nvcc", "cuda", ".so")
        assert "-O3" in result.manifest["compiler_options"]


@pytest.mark.hip
@pytest.mark.device_compile
@pytest.mark.jit
@pytest.mark.parametrize("compiler", ("hiprtc", "hipcc"))
def test_hip_forced_compiler_produces_expected_artifact(
    accelerator_capabilities, tmp_path, monkeypatch, compiler
):
    from symmetrix import jit
    from symmetrix.jit_codegen import (
        render_jit_r1_hip_module,
        render_jit_r1_hip_plugin,
        factorized_hip_plugin_metadata,
    )

    target = accelerator_capabilities
    assert target["backend"] == "hip"
    contract = json.loads(CONTRACT_PATH.read_text())
    metadata = factorized_hip_plugin_metadata(contract, target)

    monkeypatch.setenv("SYMMETRIX_JIT_HIP_JIT_BACKEND", compiler)
    assert jit.execution_hip_jit_backend_request() == compiler
    if compiler == "hipcc":
        with pytest.warns(FutureWarning, match="runtime hipcc"):
            assert jit.select_execution_hip_jit_backend() == compiler
    else:
        assert jit.select_execution_hip_jit_backend() == compiler

    render_arguments = {
        "edge_strategy": metadata["edge_strategy"],
        "edge_threads_per_block": metadata["edge_threads_per_block"],
        "persistent_blocks_per_compute_unit": metadata[
            "persistent_blocks_per_compute_unit"
        ],
    }
    if compiler == "hiprtc":
        source = render_jit_r1_hip_module(contract, target, **render_arguments)
        result = jit.prepare_hiprtc_jit_artifact(
            source,
            abi={"tag": metadata["abi"], "version": metadata["abi_version"]},
            build={"generator": "qualification.hip-hiprtc", "contract": metadata},
            target=target,
            cache_root=tmp_path / "hiprtc-cache",
        )
        _assert_artifact(result, "hiprtc", "hiprtc", ".hsaco")
    else:
        source = render_jit_r1_hip_plugin(contract, target, **render_arguments)
        result = jit.prepare_execution_hip_jit_artifact(
            source,
            abi={"tag": metadata["abi"], "version": metadata["abi_version"]},
            build={"generator": "qualification.hip-hipcc", "contract": metadata},
            target=target,
            cache_root=tmp_path / "hipcc-cache",
        )
        _assert_artifact(result, "hipcc", "hip", ".so")
    assert "-O3" in result.manifest["compiler_options"]
