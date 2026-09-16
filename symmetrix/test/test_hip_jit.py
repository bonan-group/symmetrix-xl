import json
from pathlib import Path

import pytest
from symmetrix import jit as jit


def test_hip_jit_policy_prefers_hiprtc_and_preserves_hipcc(monkeypatch):
    monkeypatch.setattr(
        jit,
        "jit_hiprtc_information",
        lambda: {"available": True, "library": "libhiprtc.so"},
    )
    assert jit.select_execution_hip_jit_backend("automatic") == "hiprtc"
    assert jit.select_execution_hip_jit_backend("hiprtc") == "hiprtc"
    with pytest.warns(FutureWarning, match="runtime hipcc"):
        assert jit.select_execution_hip_jit_backend("hipcc") == "hipcc"

    monkeypatch.setattr(
        jit,
        "jit_hiprtc_information",
        lambda: {"available": False, "reason": "fixture unavailable"},
    )
    with pytest.raises(jit.JitError, match="no AOT compiler fallback"):
        jit.select_execution_hip_jit_backend("automatic")
    with pytest.raises(jit.JitError, match="explicitly requested"):
        jit.select_execution_hip_jit_backend("hiprtc")
    with pytest.raises(ValueError, match="must be automatic"):
        jit.execution_hip_jit_backend_request("clang")
    monkeypatch.setenv(jit.HIP_JIT_BACKEND_ENVIRONMENT_VARIABLE, "hipcc")
    assert jit.execution_hip_jit_backend_request() == "hipcc"


def test_fake_hiprtc_artifact_is_cached_as_code_object(tmp_path):
    calls = []

    def compile_source(source, options):
        calls.append((source, tuple(options)))
        return b"synthetic gfx1151 code object", "synthetic hipRTC log"

    arguments = {
        "source": 'extern "C" __global__ void kernel() {}\n',
        "abi": {"tag": "symmetrix.jit.hip-module/1", "version": 1},
        "build": {"generator": "hiprtc-test-v1", "contract": "contract-a"},
        "target": {
            "raw_agent_target": "gfx1151",
            "runtime_version": "7.1",
            "native_subgroup_width": 32,
        },
        "cache_root": tmp_path / "cache",
        "hiprtc_information": {
            "available": True,
            "library": "libhiprtc.so",
            "major": 9,
            "minor": 0,
        },
        "compile_source": compile_source,
    }
    built = jit.prepare_hiprtc_jit_artifact(**arguments)
    assert built.status == "built"
    assert built.artifact_path.suffix == ".hsaco"
    manifest = json.loads(Path(built.manifest_path).read_text())
    assert manifest["source"] == "module.hip"
    assert manifest["key_inputs"]["compiler"]["kind"] == "hiprtc"
    assert manifest["key_inputs"]["build"]["backend"] == "hiprtc"
    assert manifest["compiler_options"] == [
        "--std=c++20",
        "-O3",
        "--gpu-architecture=gfx1151",
    ]
    assert len(calls) == 1

    cached = jit.prepare_hiprtc_jit_artifact(**arguments)
    assert cached.status == "cached"
    assert cached.cache_key == built.cache_key
    assert len(calls) == 1


def test_hip_r1_edge_strategy_has_independent_rollback(monkeypatch):
    assert jit.execution_hip_r1_edge_strategy("wave") == "wave"
    assert jit.execution_hip_r1_edge_strategy("serial") == "serial"
    assert jit.execution_hip_r1_edge_strategy(channels=64) == "wave"
    assert jit.execution_hip_r1_edge_strategy(channels=128) == "wave"
    monkeypatch.setenv(jit.HIP_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE, "serial")
    assert jit.execution_hip_r1_edge_strategy(channels=64) == "serial"
    with pytest.raises(ValueError, match="must be wave or serial"):
        jit.execution_hip_r1_edge_strategy("automatic")


def test_hip_r1_edge_launch_policy_is_validated_and_tunable(monkeypatch):
    assert jit.execution_hip_r1_edge_launch_policy(channels=128) == {
        "strategy": "wave",
        "edge_threads_per_block": 64,
        "persistent_blocks_per_compute_unit": 8,
    }
    monkeypatch.setenv(jit.HIP_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE, "128")
    monkeypatch.setenv(jit.HIP_R1_EDGE_BLOCKS_PER_CU_ENVIRONMENT_VARIABLE, "4")
    assert jit.execution_hip_r1_edge_launch_policy(channels=128) == {
        "strategy": "wave",
        "edge_threads_per_block": 128,
        "persistent_blocks_per_compute_unit": 4,
    }
    monkeypatch.setenv(jit.HIP_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE, "96")
    with pytest.raises(ValueError, match="power of two"):
        jit.execution_hip_r1_edge_launch_policy(channels=128)


def test_hip_target_identity_preserves_features():
    target = jit.normalize_execution_hip_target_identity(
        {
            "raw_agent_target": "gfx942:xnack-:sramecc+",
            "target_features": "sramecc+:xnack-",
            "compiler_offload_target": "gfx942",
            "runtime_version": "7.1",
            "native_subgroup_width": 64,
        }
    )
    assert target["backend"] == "hip"
    assert target["architecture"] == "gfx942"
    assert target["target_features"] == ["sramecc+", "xnack-"]
    assert target["runtime"]["native_subgroup_width"] == 64


def test_hip_target_rejects_feature_mismatch():
    with pytest.raises(ValueError, match="disagree"):
        jit.normalize_execution_hip_target_identity(
            {"raw_agent_target": "gfx942:xnack+", "target_features": "xnack-"}
        )


def test_cuda_and_hip_cache_keys_do_not_collide():
    common = {
        "source": 'extern "C" int value() { return 1; }',
        "abi": {"tag": "fixture", "version": 1},
        "compiler": {"version": "fixture"},
        "cxx_flags": (),
    }
    cuda = jit.jit_cache_key(
        **common,
        build={"backend": "cuda", "architecture": "sm_90"},
        cpu={"backend": "cuda", "architecture": "sm_90"},
    )
    hip = jit.jit_cache_key(
        **common,
        build={"backend": "hip", "architecture": "gfx942"},
        cpu={"backend": "hip", "architecture": "gfx942"},
    )
    assert cuda != hip


def test_fake_hipcc_artifact_records_backend(tmp_path):
    compiler = tmp_path / "hipcc"
    compiler.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "if '--version' in sys.argv:\n"
        " print('HIP version: 7.1.0')\n"
        " raise SystemExit(0)\n"
        "pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_bytes(b'hip')\n"
    )
    compiler.chmod(0o755)
    result = jit.prepare_execution_hip_jit_artifact(
        'extern "C" int value() { return 1; }',
        abi={"tag": "fixture", "version": 1},
        build={"contract": "same"},
        target="gfx942",
        hipcc=compiler,
        cache_root=tmp_path / "cache",
    )
    assert result.status == "built"
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["key_inputs"]["build"]["backend"] == "hip"
    assert manifest["key_inputs"]["cpu"]["architecture"] == "gfx942"
