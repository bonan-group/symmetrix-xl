from types import SimpleNamespace
import threading
import json
from pathlib import Path

import pytest
from symmetrix import jit as jit_runtime
from symmetrix import installed_operator_artifact as installed_artifact
from symmetrix import mace_jit_operator_artifact as operator_artifact
from symmetrix import mace_jit_operator_codegen as operator_codegen
from symmetrix import symmetrix as native_module


class _Evaluator:
    def __init__(self, *, m0="generic", m1="generic", r0="generic"):
        self.m0_implementation = m0
        self.m1_implementation = m1
        self.r0_implementation = "generic" if r0 == "generic" else "generated"
        self._r0_loader_implementation = r0
        self.m0_device_module_ready = False
        self.m0_host_plugin_ready = False
        self.m1_device_module_ready = False
        self.m1_host_plugin_ready = False
        self.r0_device_module_ready = False
        self.r0_host_plugin_ready = False
        self.m0_device_module_artifact_id = ""
        self.m0_host_plugin_artifact_id = ""
        self.m1_device_module_artifact_id = ""
        self.m1_host_plugin_artifact_id = ""
        self.r0_device_module_artifact_id = ""
        self.r0_host_plugin_artifact_id = ""
        self.m0_module_id = "builtin-m0" if m0 == "builtin" else ""
        self.r0_module_id = "builtin-r0" if r0 == "builtin" else ""
        self.r0_device_uses_edge_weights = False
        self.loads = []
        self.r0_artifact_origin = None

    def _load_m0_device_module(self, path, schedule, blocks):
        self.loads.append(("M0", path, schedule, blocks))
        self.m0_device_module_ready = True
        self.m0_device_module_artifact_id = "m0-artifact"

    def _load_r0_device_module(
        self, path, blocks, apply_edge_weights=False, schedule="receiver-edge16"
    ):
        self.loads.append(("R0", path, schedule, blocks))
        self.r0_device_uses_edge_weights = apply_edge_weights
        self.r0_device_module_ready = True
        self.r0_device_module_artifact_id = "r0-artifact"

    def _load_m1_device_module(self, path, schedule, blocks):
        self.loads.append(("M1", path, schedule, blocks))
        self.m1_device_module_ready = True
        self.m1_device_module_artifact_id = "m1-artifact"

    def _load_m0_host_plugin(self, path):
        self.loads.append(("M0-host", path))
        self.m0_host_plugin_ready = True
        self.m0_host_plugin_artifact_id = "m0-host-artifact"

    def _load_r0_host_plugin(self, path, apply_edge_weights=False, target="host"):
        self.loads.append(("R0-host", path, target))
        self.r0_host_plugin_ready = True
        self.r0_host_plugin_artifact_id = "r0-host-artifact"

    def _load_m1_host_plugin(self, path):
        self.loads.append(("M1-host", path))
        self.m1_host_plugin_ready = True
        self.m1_host_plugin_artifact_id = "m1-host-artifact"

    def _set_r0_artifact_origin(self, origin):
        self.r0_artifact_origin = origin


@pytest.fixture(autouse=True)
def _isolate_installed_operator_metadata(monkeypatch):
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: None,
    )

    def absent(**arguments):
        operator = arguments["operator"]
        return SimpleNamespace(
            status="absent",
            diagnostics=(
                f"origin=prebuilt status=absent operator={operator} "
                "reason=catalog-absent",
            ),
        )

    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        absent,
    )

    validate_entry = jit_runtime._validated_entry

    def validate_real_entries(entry, cache_key, inputs):
        # The compilation tests below use lightweight prepared-result doubles;
        # cache publication and manifest validation have dedicated coverage in
        # test_jit.py.  Keep validation enabled for every real (SHA-256) key.
        if cache_key.startswith("cache-"):
            return None
        return validate_entry(entry, cache_key, inputs)

    monkeypatch.setattr(jit_runtime, "_validated_entry", validate_real_entries)


def _result(tmp_path, name):
    return SimpleNamespace(
        status="built",
        available=True,
        cache_key=f"cache-{name}",
        artifact_path=tmp_path / f"{name}.cubin",
        manifest_path=tmp_path / f"{name}.json",
        manifest={"key_inputs": {"fixture": name}},
        diagnostics=(f"compiled-{name}",),
        reason=None,
    )


def _operator_metadata(artifact_id, schema="symmetrix.jit.operator-module/1"):
    """Envelope-complete stand-in for the operator metadata builder."""

    return {
        "schema": schema,
        "abi_version": 1,
        "artifact_id": artifact_id,
        "structure_fingerprint": "sha256:" + "d" * 64,
        "capabilities": 0x1F,
    }


def test_prepare_operator_modules_compiles_missing_stages(monkeypatch, tmp_path):
    evaluator = _Evaluator()
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "tiled128x4-v1")
    monkeypatch.setenv("SYMMETRIX_M1_RTC_SCHEDULE", "chunk32")
    calls = []
    compile_threads = set()
    compile_lock = threading.Lock()
    two_compilers_active = threading.Event()
    active_compilers = 0
    maximum_active_compilers = 0

    def render_m0(contract, **arguments):
        assert contract == {"fixture": "m0"}
        assert arguments["schedule"] == "tiled128x4-v1"
        return "m0-source", _operator_metadata("m0-generated")

    def render_r0(contract, **arguments):
        assert contract == {"fixture": "r0"}
        assert arguments["apply_edge_weights"] is True
        assert arguments["schedule"] == "receiver-edge16"
        return "r0-source", {
            **_operator_metadata("r0-generated"),
            "compatibility_version": 2,
            "schedule_revision": 3,
            "schedule_capabilities": ["forward-receiver", "persistent-grid"],
            "architecture_bounds": {
                "channels": 128,
                "l_max": 3,
                "harmonic_count": 16,
            },
            "persistent_profile": {
                "blocks_per_compute_unit": 4,
                "calibration_candidates": [2, 4, 8],
            },
        }

    def render_m1(contract, **arguments):
        assert contract == {"fixture": "m1"}
        assert arguments["schedule"] == "chunk32"
        return "m1-source", _operator_metadata("m1-generated")

    def prepare(source, **arguments):
        nonlocal active_compilers, maximum_active_compilers
        calls.append((source, arguments))
        compile_threads.add(threading.get_ident())
        with compile_lock:
            active_compilers += 1
            maximum_active_compilers = max(maximum_active_compilers, active_compilers)
            if active_compilers == 2:
                two_compilers_active.set()
        assert two_compilers_active.wait(timeout=2)
        try:
            return _result(tmp_path, source.split("-")[0])
        finally:
            with compile_lock:
                active_compilers -= 1

    monkeypatch.setattr(operator_codegen, "render_jit_m0_device_module", render_m0)
    monkeypatch.setattr(operator_codegen, "render_jit_m1_device_module", render_m1)
    monkeypatch.setattr(operator_codegen, "render_jit_r0_device_module", render_r0)
    monkeypatch.setattr(jit_runtime, "prepare_nvrtc_jit_artifact", prepare)

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={
            "execution_contracts": {
                "M0": {"fixture": "m0"},
                "M1": {"fixture": "m1"},
                "R0": {"fixture": "r0"},
            }
        },
        precision="float64",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=2,
        apply_edge_weights=True,
        cache_root=tmp_path / "cache",
    )

    assert {call[0] for call in calls} == {"m0-source", "m1-source", "r0-source"}
    assert len(compile_threads) == 2
    assert maximum_active_compilers == 2
    assert all(call[1]["cache_root"] == tmp_path / "cache" for call in calls)
    assert {source: arguments["nvrtc_options"] for source, arguments in calls} == {
        "m0-source": ("--split-compile=2",),
        "m1-source": ("--split-compile=2",),
        "r0-source": ("--split-compile=2", "--use_fast_math"),
    }
    assert evaluator.loads == [
        ("M0", str(tmp_path / "m0.cubin"), "tiled128x4-v1", 8),
        ("M1", str(tmp_path / "m1.cubin"), "chunk32", 8),
        ("R0", str(tmp_path / "r0.cubin"), "receiver-edge16", 4),
    ]
    assert modules["M0"]["artifact_id"] == "m0-artifact"
    assert modules["M1"]["artifact_id"] == "m1-artifact"
    assert modules["R0"]["artifact_id"] == "r0-artifact"
    assert modules["R0"]["compatibility_version"] == 2
    assert evaluator.r0_device_uses_edge_weights is True
    assert modules["M0"]["target"] == "sm_120"
    assert modules["M0"]["persistent_blocks_per_compute_unit"] == 8
    assert modules["R0"]["schedule_revision"] == 3
    assert modules["R0"]["schedule_capabilities"] == [
        "forward-receiver",
        "persistent-grid",
    ]
    assert modules["R0"]["architecture_bounds"]["channels"] == 128
    assert diagnostics == (
        "stage=M0 schedule=tiled128x4-v1: origin=prebuilt status=absent "
        "operator=M0 reason=catalog-absent",
        "stage=M0 schedule=tiled128x4-v1: compiled-m0",
        "stage=M0 schedule=tiled128x4-v1: origin=runtime status=compiled",
        "stage=M1 schedule=chunk32: origin=prebuilt status=absent "
        "operator=M1 reason=catalog-absent",
        "stage=M1 schedule=chunk32: compiled-m1",
        "stage=M1 schedule=chunk32: origin=runtime status=compiled",
        "stage=R0 schedule=receiver-edge16: origin=prebuilt status=absent "
        "operator=R0 reason=catalog-absent",
        "stage=R0 schedule=receiver-edge16: compiled-r0",
        "stage=R0 schedule=receiver-edge16: origin=runtime status=compiled",
    )


def test_r0_persistent_grid_comes_from_generated_schedule_metadata():
    metadata = {
        "persistent_profile": {
            "blocks_per_compute_unit": 3,
            "calibration_candidates": [2, 3, 6],
        }
    }

    assert (
        operator_artifact._persistent_blocks_per_compute_unit(
            "R0",
            metadata,
            precision="float32",
            backend="cuda",
            target="sm_120",
        )
        == 3
    )


def test_production_r0_schedule_resolver_honors_default_and_override(monkeypatch):
    monkeypatch.delenv("SYMMETRIX_R0_RTC_SCHEDULE", raising=False)
    assert operator_artifact.resolve_r0_device_schedule() == "receiver-edge16"

    monkeypatch.setenv("SYMMETRIX_R0_RTC_SCHEDULE", "receiver-edge32")
    assert operator_artifact.resolve_r0_device_schedule() == "receiver-edge32"

    monkeypatch.setenv("SYMMETRIX_R0_RTC_SCHEDULE", "invalid")
    with pytest.raises(ValueError, match="receiver-edge16.*receiver-edge32"):
        operator_artifact.resolve_r0_device_schedule()


def test_wide_l2_m0_uses_calibrated_gfx1151_grid():
    metadata = {
        "schedule": "chunk32",
        "contract": {"channels": 128},
        "term_count": 923,
        "structure_fingerprint": operator_artifact._M0_GFX1151_WIDE_L2_STRUCTURE,
    }

    assert (
        operator_artifact._persistent_blocks_per_compute_unit(
            "M0",
            metadata,
            precision="float32",
            backend="hip",
            target="gfx1151",
        )
        == 2
    )
    assert (
        operator_artifact._persistent_blocks_per_compute_unit(
            "M0",
            metadata,
            precision="float32",
            backend="cuda",
            target="sm_120",
        )
        == 8
    )


@pytest.mark.parametrize(("backend", "target"), (("cuda", {}), ("metal", {})))
def test_prepare_operator_modules_rejects_invalid_target(backend, target):
    with pytest.raises(RuntimeError):
        operator_artifact.prepare_low_memory_operator_modules(
            _Evaluator(),
            model_data={"execution_contracts": {}},
            precision="float32",
            backend=backend,
            target=target,
            jit_generation_version=2,
        )


def test_prepare_operator_modules_keeps_builtin_stages_compiler_free():
    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        _Evaluator(m0="builtin", r0="builtin"),
        model_data={"execution_contracts": {}},
        precision="float32",
        backend="cuda",
        target={"compute_capability_code": 90},
        jit_generation_version=2,
    )

    assert diagnostics == ()
    assert modules == {
        "M0": {
            "status": "builtin",
            "implementation": "builtin",
            "compiler": None,
            "artifact_id": "builtin-m0",
        },
        "R0": {
            "status": "builtin",
            "implementation": "generated",
            "backend": "cuda",
            "artifact_origin": None,
            "schedule": "frozen",
            "profile": "fixed-weight-coordinate",
            "compiler": None,
            "artifact_id": "builtin-r0",
        },
    }


def test_prepare_operator_modules_builds_missing_host_m0(monkeypatch, tmp_path):
    evaluator = _Evaluator(r0="builtin")
    calls = []

    def render_m0(contract, **arguments):
        assert contract == {"fixture": "m0"}
        return "host-m0-source", _operator_metadata(
            "m0-host-generated", schema="symmetrix.jit.m0-host-plugin/1"
        )

    def prepare(source, **arguments):
        calls.append((source, arguments))
        return _result(tmp_path, "host-m0")

    monkeypatch.setattr(operator_codegen, "render_jit_m0_host_plugin", render_m0)
    monkeypatch.setattr(jit_runtime, "prepare_jit_artifact", prepare)

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"M0": {"fixture": "m0"}}},
        precision="float32",
        backend="host",
        target={},
        jit_generation_version=3,
        cache_root=tmp_path / "cache",
    )

    assert evaluator.loads == [("M0-host", str(tmp_path / "host-m0.cubin"))]
    assert calls[0][1]["cxx_flags"] is None
    assert calls[0][1]["abi"] == {
        "tag": "symmetrix.jit.m0-host-plugin/1",
        "version": 1,
    }
    assert modules["M0"]["implementation"] == "host_plugin"
    assert modules["M0"]["artifact_id"] == "m0-host-artifact"
    assert modules["R0"]["status"] == "builtin"
    assert diagnostics == (
        "stage=M0 schedule=chunk32: origin=prebuilt status=absent "
        "operator=M0 reason=catalog-absent",
        "stage=M0 schedule=chunk32: compiled-host-m0",
        "stage=M0 schedule=chunk32: origin=runtime status=compiled",
    )


def test_prepare_operator_modules_bounds_parallel_host_compilers(monkeypatch, tmp_path):
    evaluator = _Evaluator(r0="builtin")
    compile_lock = threading.Lock()
    two_compilers_active = threading.Event()
    active_compilers = 0
    maximum_active_compilers = 0

    def render_m0(_contract, **_arguments):
        return "host-m0-source", _operator_metadata(
            "m0-host-generated", schema="symmetrix.jit.m0-host-plugin/1"
        )

    def render_m1(_contract, **_arguments):
        return "host-m1-source", _operator_metadata(
            "m1-host-generated", schema="symmetrix.jit.m1-host-plugin/1"
        )

    def prepare(source, **_arguments):
        nonlocal active_compilers, maximum_active_compilers
        with compile_lock:
            active_compilers += 1
            maximum_active_compilers = max(maximum_active_compilers, active_compilers)
            if active_compilers == 2:
                two_compilers_active.set()
        assert two_compilers_active.wait(timeout=2)
        try:
            return _result(tmp_path, source.split("-")[1])
        finally:
            with compile_lock:
                active_compilers -= 1

    monkeypatch.setattr(operator_codegen, "render_jit_m0_host_plugin", render_m0)
    monkeypatch.setattr(operator_codegen, "render_jit_m1_host_plugin", render_m1)
    monkeypatch.setattr(jit_runtime, "prepare_jit_artifact", prepare)

    modules, _diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={
            "execution_contracts": {
                "M0": {"fixture": "m0"},
                "M1": {"fixture": "m1"},
            }
        },
        precision="float32",
        backend="host",
        target={},
        jit_generation_version=3,
        cache_root=tmp_path / "cache",
    )

    assert maximum_active_compilers == 2
    assert modules["M0"]["implementation"] == "host_plugin"
    assert modules["M1"]["implementation"] == "host_plugin"
    assert evaluator.loads == [
        ("M0-host", str(tmp_path / "m0.cubin")),
        ("M1-host", str(tmp_path / "m1.cubin")),
    ]


def test_host_m0_fallback_uses_one_native_policy_for_identity_and_manifest(
    monkeypatch, tmp_path
):
    evaluator = _Evaluator(r0="builtin")
    installed_target = "x86-64-v3"
    runtime_target = "host-sha256-" + "7" * 64
    rendered_targets = []
    requested_targets = []

    monkeypatch.delenv("SYMMETRIX_JIT_HOST_TARGET", raising=False)
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda *, mode=None, **_arguments: (installed_target if mode is None else None),
    )

    def render_m0(contract, **arguments):
        target = arguments["target"]
        rendered_targets.append(target)
        return (
            'extern "C" int symmetrix_test_m0_target = 1;\n',
            {
                **_operator_metadata(
                    "m0-host-generated", schema="symmetrix.jit.m0-host-plugin/1"
                ),
                "generated_identity": {
                    "artifact_id": "m0-host-generated",
                    "target": target,
                },
            },
        )

    policy = jit_runtime.HostJitPolicy(
        requested="automatic",
        selected="native",
        effective_target=runtime_target,
        flags=("-ffast-math", "-march=native"),
        diagnostics=("host JIT policy: native",),
    )

    def resolve_policy(_command, *, target=None, flags=None):
        assert flags is None
        requested_targets.append(target)
        return policy

    monkeypatch.setattr(operator_codegen, "render_jit_m0_host_plugin", render_m0)
    monkeypatch.setattr(jit_runtime, "resolve_host_jit_policy", resolve_policy)

    modules, _diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"M0": {"fixture": "m0"}}},
        precision="float32",
        backend="host",
        target={},
        jit_generation_version=3,
        cache_root=tmp_path / "cache",
    )

    assert rendered_targets == [installed_target, runtime_target]
    assert requested_targets == [None]
    assert modules["M0"]["target"] == runtime_target
    manifest = json.loads(
        (Path(modules["M0"]["artifact_path"]).parent / "manifest.json").read_text()
    )
    key_inputs = manifest["key_inputs"]
    assert key_inputs["cxx_flags"] == list(policy.flags)
    assert key_inputs["build"]["host_jit_policy"] == policy.identity
    assert (
        key_inputs["build"]["operator"]["generated_identity"]["target"]
        == runtime_target
    )


def test_prepare_operator_modules_builds_architecture_specialized_host_r0(
    monkeypatch, tmp_path
):
    evaluator = _Evaluator(m0="builtin")
    calls = []
    rendered_targets = []
    monkeypatch.setenv("SYMMETRIX_JIT_HOST_TARGET", "x86-64-v3")

    def render_r0(contract, **arguments):
        assert contract == {"fixture": "r0", "l_max": 2, "channels": 48}
        assert arguments["apply_edge_weights"] is True
        assert arguments["target"] == "host-isa-x86-64-v3"
        rendered_targets.append(arguments["target"])
        return "host-r0-source", {
            **_operator_metadata(
                "r0-host-l2-c48-weighted",
                schema="symmetrix.jit.r0-host-plugin/2",
            ),
            "abi_version": 2,
            "compatibility_version": 2,
        }

    def prepare(source, **arguments):
        calls.append((source, arguments))
        return _result(tmp_path, "host-r0")

    monkeypatch.setattr(operator_codegen, "render_jit_r0_host_plugin", render_r0)
    monkeypatch.setattr(jit_runtime, "prepare_jit_artifact", prepare)

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={
            "execution_contracts": {"R0": {"fixture": "r0", "l_max": 2, "channels": 48}}
        },
        precision="float64",
        backend="host",
        target={"backend": "host", "architecture": "host"},
        jit_generation_version=3,
        apply_edge_weights=True,
        cache_root=tmp_path / "cache",
    )

    assert evaluator.loads == [
        ("R0-host", str(tmp_path / "host-r0.cubin"), rendered_targets[0])
    ]
    assert calls[0][1]["abi"] == {
        "tag": "symmetrix.jit.r0-host-plugin/2",
        "version": 2,
    }
    assert calls[0][1]["host_policy"].effective_target == rendered_targets[0]
    assert calls[0][1]["host_policy"].flags[-1] == "-march=x86-64-v3"
    assert calls[0][1]["cxx"]
    assert modules["M0"]["status"] == "builtin"
    assert modules["R0"]["implementation"] == "generated"
    assert modules["R0"]["backend"] == "host"
    assert modules["R0"]["artifact_origin"] == "runtime"
    assert modules["R0"]["schedule"] == "host-owner"
    assert modules["R0"]["profile"] == "owner-batched"
    assert evaluator.r0_artifact_origin == "runtime"
    assert modules["R0"]["artifact_id"] == "r0-host-artifact"
    assert modules["R0"]["compatibility_version"] == 2
    assert diagnostics == (
        "stage=R0 schedule=host-owner: origin=prebuilt status=absent "
        "operator=R0 reason=catalog-absent",
        "stage=R0 schedule=host-owner: compiled-host-r0",
        "stage=R0 schedule=host-owner: origin=runtime status=compiled",
    )


def test_exact_installed_host_r0_precedes_compiler_and_cache(monkeypatch, tmp_path):
    evaluator = _Evaluator(m0="builtin", r0="builtin")
    artifact = tmp_path / "installed/operator.so"
    artifact.parent.mkdir()
    artifact.write_bytes(b"installed")
    compiler_attempts = []
    cache_attempts = []

    monkeypatch.setattr(
        jit_runtime,
        "probe_compiler_identity",
        lambda: compiler_attempts.append("probe") or pytest.fail("compiler probed"),
    )
    monkeypatch.setattr(
        jit_runtime,
        "prepare_jit_artifact",
        lambda source, **arguments: (
            cache_attempts.append(arguments)
            or pytest.fail("cache/runtime preparation attempted")
        ),
    )
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: "x86-64-v3",
        raising=False,
    )

    def render_r0(contract, **arguments):
        assert arguments["target"] == "x86-64-v3"
        return "unused-source", {
            **_operator_metadata(
                "canonical-installed-r0", schema="symmetrix.jit.r0-host-plugin/2"
            ),
            "abi_version": 2,
            "operator": "R0",
            "generated_identity": {"artifact_id": "canonical-installed-r0"},
            "schedule": "host-owner",
            "compatibility_version": 2,
        }

    monkeypatch.setattr(operator_codegen, "render_jit_r0_host_plugin", render_r0)

    def resolve_installed(**arguments):
        arguments["loader"](str(artifact))
        return SimpleNamespace(
            status="resolved",
            origin="prebuilt",
            artifact_path=artifact,
            artifact_id="canonical-installed-r0",
            diagnostics=("origin=prebuilt status=loaded operator=R0",),
        )

    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        resolve_installed,
        raising=False,
    )

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"R0": {"fixture": "r0"}}},
        precision="float32",
        backend="host",
        target={"backend": "host"},
        jit_generation_version=14,
        cache_root=tmp_path / "empty-cache",
    )

    assert compiler_attempts == []
    assert cache_attempts == []
    assert evaluator.loads == [("R0-host", str(artifact), "x86-64-v3")]
    assert modules["R0"]["implementation"] == "generated"
    assert modules["R0"]["backend"] == "host"
    assert modules["R0"]["artifact_origin"] == "prebuilt"
    assert modules["R0"]["schedule"] == "host-owner"
    assert modules["R0"]["profile"] == "owner-batched"
    assert evaluator.r0_artifact_origin == "prebuilt"
    assert modules["R0"]["artifact_path"] == str(artifact)
    assert diagnostics == (
        "stage=R0 schedule=host-owner: origin=prebuilt status=loaded operator=R0",
    )


def test_exact_installed_device_r0_precedes_builtin(monkeypatch, tmp_path):
    evaluator = _Evaluator(m0="builtin", r0="builtin")
    artifact = tmp_path / "installed/operator.cubin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"installed")

    def render_r0(contract, **arguments):
        assert arguments["target"] == "sm_120"
        return "unused-source", {
            **_operator_metadata("canonical-installed-r0"),
            "operator": "R0",
            "generated_identity": {"artifact_id": "canonical-installed-r0"},
            "schedule": "receiver-edge16",
            "compatibility_version": 2,
            "persistent_profile": {"blocks_per_compute_unit": 3},
        }

    monkeypatch.setattr(operator_codegen, "render_jit_r0_device_module", render_r0)

    def resolve_installed(**arguments):
        arguments["loader"](str(artifact))
        return SimpleNamespace(
            status="resolved",
            origin="prebuilt",
            artifact_path=artifact,
            artifact_id="canonical-installed-r0",
            diagnostics=("origin=prebuilt status=loaded operator=R0",),
        )

    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        resolve_installed,
        raising=False,
    )

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"R0": {"fixture": "r0"}}},
        precision="float32",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=14,
        cache_root=tmp_path / "empty-cache",
    )

    assert evaluator.loads == [
        ("R0", str(artifact), "receiver-edge16", 3),
    ]
    assert modules["R0"]["implementation"] == "generated"
    assert modules["R0"]["backend"] == "cuda"
    assert modules["R0"]["artifact_origin"] == "prebuilt"
    assert modules["R0"]["schedule"] == "receiver-edge16"
    assert modules["R0"]["profile"] == "persistent-3"
    assert evaluator.r0_artifact_origin == "prebuilt"
    assert diagnostics == (
        "stage=R0 schedule=receiver-edge16: origin=prebuilt status=loaded operator=R0",
    )


def test_exact_installed_device_m0_precedes_builtin(monkeypatch, tmp_path):
    evaluator = _Evaluator(m0="builtin", r0="builtin")
    artifact = tmp_path / "installed/m0.cubin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"installed")

    def render_m0(contract, **arguments):
        assert arguments["target"] == "sm_120"
        return "unused-source", {
            **_operator_metadata("canonical-installed-m0"),
            "operator": "M0",
            "generated_identity": {"artifact_id": "canonical-installed-m0"},
            "schedule": "chunk32",
        }

    monkeypatch.setattr(operator_codegen, "render_jit_m0_device_module", render_m0)
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: ("sm_120" if arguments.get("mode") == "full" else None),
    )

    def resolve_installed(**arguments):
        arguments["loader"](str(artifact))
        return SimpleNamespace(
            status="resolved",
            origin="prebuilt",
            artifact_path=artifact,
            artifact_id="canonical-installed-m0",
            diagnostics=("origin=prebuilt status=loaded operator=M0",),
        )

    monkeypatch.setattr(
        operator_artifact, "resolve_installed_operator_artifact", resolve_installed
    )

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"M0": {"fixture": "m0"}}},
        precision="float32",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=14,
        cache_root=tmp_path / "empty-cache",
    )

    assert evaluator.loads == [("M0", str(artifact), "chunk32", 8)]
    assert modules["M0"]["status"] == "loaded"
    assert modules["M0"]["origin"] == "prebuilt"
    assert modules["M0"]["compiler"] is None
    assert diagnostics == (
        "stage=M0 schedule=chunk32: origin=prebuilt status=loaded operator=M0",
    )


@pytest.mark.parametrize("mode", ("off", "r0"))
def test_non_full_device_catalog_short_circuits_builtin_m0_before_rtc(
    monkeypatch, mode
):
    evaluator = _Evaluator(m0="builtin", r0="builtin")
    target = "" if mode == "off" else "sm_120"
    monkeypatch.setattr(
        native_module,
        "_backend_build_info",
        lambda: {
            "prebuilt_operator_artifacts": {
                "mode": mode,
                "backend": "cuda" if mode != "off" else "",
                "target": target,
            }
        },
    )
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        installed_artifact.installed_operator_build_target,
    )
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "invalid-schedule")
    monkeypatch.setenv("SYMMETRIX_JIT_NVCC", "/definitely/missing-nvcc")
    monkeypatch.setattr(
        operator_codegen,
        "render_jit_m0_device_module",
        lambda *args, **kwargs: pytest.fail("builtin M0 rendered RTC source"),
    )
    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        lambda **arguments: pytest.fail("non-full M0 probed installed artifacts"),
    )
    monkeypatch.setattr(
        jit_runtime,
        "prepare_nvrtc_jit_artifact",
        lambda *args, **kwargs: pytest.fail("builtin M0 prepared RTC state"),
    )

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"M0": {"fixture": "m0"}}},
        precision="float32",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=14,
    )

    assert modules["M0"] == {
        "status": "builtin",
        "implementation": "builtin",
        "compiler": None,
        "artifact_id": "builtin-m0",
    }
    assert diagnostics == ()


@pytest.mark.parametrize("mode", ("off", "r0"))
def test_non_full_host_catalog_short_circuits_builtin_m0_before_compiler_probe(
    monkeypatch, mode
):
    evaluator = _Evaluator(m0="builtin", r0="builtin")
    target = "" if mode == "off" else "x86-64-v3"
    monkeypatch.setattr(
        native_module,
        "_backend_build_info",
        lambda: {
            "prebuilt_operator_artifacts": {
                "mode": mode,
                "backend": "host" if mode != "off" else "",
                "target": target,
            }
        },
    )
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        installed_artifact.installed_operator_build_target,
    )
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "invalid-schedule")
    monkeypatch.setenv("CXX", "/definitely/missing-cxx")
    monkeypatch.setattr(
        operator_codegen,
        "render_jit_m0_host_plugin",
        lambda *args, **kwargs: pytest.fail("builtin M0 rendered host source"),
    )
    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        lambda **arguments: pytest.fail("non-full M0 probed installed artifacts"),
    )
    monkeypatch.setattr(
        jit_runtime,
        "probe_compiler_identity",
        lambda: pytest.fail("builtin M0 probed the host compiler"),
    )

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"M0": {"fixture": "m0"}}},
        precision="float32",
        backend="host",
        target={"host_target": "native"},
        jit_generation_version=14,
        prefer_host_m0_plugin=True,
    )

    assert modules["M0"] == {
        "status": "builtin",
        "implementation": "builtin",
        "compiler": None,
        "artifact_id": "builtin-m0",
    }
    assert diagnostics == ()


@pytest.mark.parametrize("reason", ("load-failed", "checksum-mismatch"))
def test_rejected_full_installed_device_m0_recovers_to_builtin_without_rtc(
    monkeypatch, reason
):
    evaluator = _Evaluator(m0="builtin", r0="builtin")
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: ("sm_120" if arguments.get("mode") == "full" else None),
    )
    monkeypatch.setattr(
        operator_codegen,
        "render_jit_m0_device_module",
        lambda contract, **arguments: (
            "unused-source",
            {
                **_operator_metadata("canonical-installed-m0"),
                "operator": "M0",
                "generated_identity": {"artifact_id": "canonical-installed-m0"},
                "schedule": "chunk32",
            },
        ),
    )
    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        lambda **arguments: SimpleNamespace(
            status="rejected",
            diagnostics=(
                f"origin=prebuilt status=rejected operator=M0 reason={reason}",
            ),
        ),
    )
    monkeypatch.setattr(
        jit_runtime,
        "prepare_nvrtc_jit_artifact",
        lambda *args, **kwargs: pytest.fail("builtin recovery prepared RTC state"),
    )

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"M0": {"fixture": "m0"}}},
        precision="float32",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=14,
    )

    assert modules["M0"] == {
        "status": "builtin",
        "origin": "builtin",
        "implementation": "builtin",
        "compiler": None,
        "artifact_id": "builtin-m0",
        "diagnostics": (
            "stage=M0 schedule=chunk32: origin=prebuilt status=rejected "
            f"operator=M0 reason={reason}",
        ),
    }
    assert diagnostics == modules["M0"]["diagnostics"]


def test_rejected_installed_host_r0_recovers_to_builtin_without_compiler(monkeypatch):
    evaluator = _Evaluator(m0="builtin", r0="builtin")
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: "x86-64-v3",
    )
    monkeypatch.setattr(
        jit_runtime,
        "probe_compiler_identity",
        lambda: pytest.fail("builtin recovery probed the compiler"),
    )
    monkeypatch.setattr(
        operator_codegen,
        "render_jit_r0_host_plugin",
        lambda contract, **arguments: (
            "unused-source",
            {
                **_operator_metadata(
                    "canonical-installed-r0",
                    schema="symmetrix.jit.r0-host-plugin/2",
                ),
                "abi_version": 2,
                "operator": "R0",
                "generated_identity": {"artifact_id": "canonical-installed-r0"},
                "schedule": "host-owner",
                "compatibility_version": 2,
            },
        ),
    )
    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        lambda **arguments: SimpleNamespace(
            status="rejected",
            diagnostics=(
                "origin=prebuilt status=rejected operator=R0 reason=checksum-mismatch",
            ),
        ),
    )

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"R0": {"fixture": "r0"}}},
        precision="float32",
        backend="host",
        target={"backend": "host"},
        jit_generation_version=14,
    )

    assert evaluator.loads == []
    assert modules["R0"] == {
        "status": "builtin",
        "implementation": "generated",
        "backend": "host",
        "artifact_origin": None,
        "schedule": "frozen",
        "profile": "fixed-weight-coordinate",
        "compiler": None,
        "artifact_id": "builtin-r0",
        "diagnostics": (
            "stage=R0 schedule=host-owner: origin=prebuilt status=rejected "
            "operator=R0 reason=checksum-mismatch",
        ),
    }
    assert diagnostics == modules["R0"]["diagnostics"]


def test_unsupported_installed_target_keeps_runtime_canonical_identity(
    monkeypatch,
):
    contract_path = (
        Path(__file__).parent / "data/execution_contracts/standard_r0_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    runtime_target = "host-sha256-" + "7" * 64
    policy = SimpleNamespace(
        effective_target=runtime_target,
        flags=("-ffast-math", "-march=native"),
        identity={"schema": "fixture"},
    )
    monkeypatch.setattr(
        jit_runtime, "probe_compiler_identity", lambda: (("c++",), {"id": "GNU"})
    )
    monkeypatch.setattr(jit_runtime, "resolve_host_jit_policy", lambda *a, **k: policy)
    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        lambda **arguments: SimpleNamespace(
            status="absent",
            diagnostics=(
                "origin=prebuilt status=absent operator=R0 "
                "reason=backend-target-mismatch",
            ),
        ),
    )

    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: None,
    )
    baseline = operator_artifact._plan_low_memory_operator_modules(
        _Evaluator(m0="builtin"),
        model_data={"execution_contracts": {"R0": contract}},
        precision="float32",
        backend="host",
        target={"backend": "host"},
        jit_generation_version=14,
    )
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: "unsupported-architecture",
    )
    with_installed_mismatch = operator_artifact._plan_low_memory_operator_modules(
        _Evaluator(m0="builtin"),
        model_data={"execution_contracts": {"R0": contract}},
        precision="float32",
        backend="host",
        target={"backend": "host"},
        jit_generation_version=14,
    )

    assert baseline.jobs[0].module_target == runtime_target
    assert with_installed_mismatch.jobs[0].module_target == runtime_target
    assert (
        with_installed_mismatch.jobs[0].metadata["generated_identity"]
        == baseline.jobs[0].metadata["generated_identity"]
    )


def test_operator_artifact_diagnostics_name_stage_and_schedule(monkeypatch, tmp_path):
    evaluator = _Evaluator(m0="builtin")

    monkeypatch.setattr(
        operator_codegen,
        "render_jit_r0_host_plugin",
        lambda contract, **arguments: (
            "host-r0-source",
            {
                **_operator_metadata(
                    "r0-host-generated", schema="symmetrix.jit.r0-host-plugin/2"
                ),
                "abi_version": 2,
                "compatibility_version": 2,
                "schedule": "host-owner",
            },
        ),
    )
    monkeypatch.setattr(
        jit_runtime,
        "prepare_jit_artifact",
        lambda source, **arguments: _result(tmp_path, "host-r0"),
    )

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"R0": {"fixture": "r0"}}},
        precision="float64",
        backend="host",
        target={},
        jit_generation_version=3,
        cache_root=tmp_path / "cache",
    )

    assert diagnostics == (
        "stage=R0 schedule=host-owner: origin=prebuilt status=absent "
        "operator=R0 reason=catalog-absent",
        "stage=R0 schedule=host-owner: compiled-host-r0",
        "stage=R0 schedule=host-owner: origin=runtime status=compiled",
    )
    assert modules["R0"]["diagnostics"] == diagnostics


def test_operator_codegen_failure_names_stage_and_selected_schedule(monkeypatch):
    evaluator = _Evaluator(m0="builtin")

    def fail_render(contract, **arguments):
        raise ValueError("contract topology is unsupported")

    monkeypatch.setattr(operator_codegen, "render_jit_r0_host_plugin", fail_render)

    with pytest.raises(
        RuntimeError,
        match="stage=R0 schedule=host-owner.*contract topology is unsupported",
    ):
        operator_artifact.prepare_low_memory_operator_modules(
            evaluator,
            model_data={"execution_contracts": {"R0": {"fixture": "r0"}}},
            precision="float64",
            backend="host",
            target={},
            jit_generation_version=3,
        )


def test_operator_preparation_failure_names_stage_and_selected_schedule(
    monkeypatch,
):
    evaluator = _Evaluator(m0="builtin")
    monkeypatch.setattr(
        operator_codegen,
        "render_jit_r0_host_plugin",
        lambda contract, **arguments: (
            "host-r0-source",
            {
                **_operator_metadata(
                    "r0-host-generated", schema="symmetrix.jit.r0-host-plugin/2"
                ),
                "abi_version": 2,
                "compatibility_version": 2,
                "schedule": "host-owner",
            },
        ),
    )
    monkeypatch.setattr(
        jit_runtime,
        "load_validated_jit_artifact",
        lambda prepare, load, artifact_key: SimpleNamespace(
            status="unavailable",
            result=SimpleNamespace(reason="the C++ compiler is unavailable"),
            diagnostics=("compiler lookup failed",),
            quarantines=(),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=("stage=R0 schedule=host-owner.*the C\\+\\+ compiler is unavailable"),
    ):
        operator_artifact.prepare_low_memory_operator_modules(
            evaluator,
            model_data={"execution_contracts": {"R0": {"fixture": "r0"}}},
            precision="float64",
            backend="host",
            target={},
            jit_generation_version=3,
        )


def test_prepare_operator_modules_compiles_generic_host_m0(monkeypatch, tmp_path):
    evaluator = _Evaluator(m0="generic", r0="builtin")

    monkeypatch.setattr(
        operator_codegen,
        "render_jit_m0_host_plugin",
        lambda contract, **_arguments: (
            "host-m0-source",
            _operator_metadata("m0", schema="symmetrix.jit.m0-host-plugin/1"),
        ),
    )
    monkeypatch.setattr(
        jit_runtime,
        "prepare_jit_artifact",
        lambda source, **_arguments: _result(tmp_path, source),
    )

    modules, _ = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={"execution_contracts": {"M0": {"fixture": "m0"}}},
        precision="float32",
        backend="host",
        target={},
        jit_generation_version=3,
        cache_root=tmp_path / "cache",
    )

    assert evaluator.loads == [("M0-host", str(tmp_path / "host-m0-source.cubin"))]
    assert modules["M0"]["implementation"] == "host_plugin"
    assert modules["R0"]["status"] == "builtin"
