from types import SimpleNamespace

import pytest
from symmetrix import jit as jit_runtime
from symmetrix import jit_operator_artifact as operator_artifact
from symmetrix import jit_operator_codegen as operator_codegen


class _Evaluator:
    def __init__(self, *, m0="generic", r0="generic"):
        self.m0_implementation = m0
        self.r0_implementation = r0
        self.m0_device_module_ready = False
        self.m0_host_plugin_ready = False
        self.r0_device_module_ready = False
        self.m0_device_module_artifact_id = ""
        self.m0_host_plugin_artifact_id = ""
        self.r0_device_module_artifact_id = ""
        self.m0_module_id = "builtin-m0" if m0 == "builtin" else ""
        self.r0_module_id = "builtin-r0" if r0 == "builtin" else ""
        self.loads = []

    def _load_m0_device_module(self, path, schedule, blocks):
        self.loads.append(("M0", path, schedule, blocks))
        self.m0_device_module_ready = True
        self.m0_device_module_artifact_id = "m0-artifact"

    def _load_r0_device_module(self, path, blocks):
        self.loads.append(("R0", path, blocks))
        self.r0_device_module_ready = True
        self.r0_device_module_artifact_id = "r0-artifact"

    def _load_m0_host_plugin(self, path):
        self.loads.append(("M0-host", path))
        self.m0_host_plugin_ready = True
        self.m0_host_plugin_artifact_id = "m0-host-artifact"


def _result(tmp_path, name):
    return SimpleNamespace(
        status="built",
        available=True,
        cache_key=f"cache-{name}",
        artifact_path=tmp_path / f"{name}.cubin",
        manifest_path=tmp_path / f"{name}.json",
        diagnostics=(f"compiled-{name}",),
        reason=None,
    )


def test_prepare_operator_modules_compiles_missing_stages(monkeypatch, tmp_path):
    evaluator = _Evaluator()
    calls = []

    def render_m0(contract, **arguments):
        assert contract == {"fixture": "m0"}
        return "m0-source", {
            "artifact_id": "m0-generated",
            "fixture": arguments,
        }

    def render_r0(contract, **arguments):
        assert contract == {"fixture": "r0"}
        return "r0-source", {
            "artifact_id": "r0-generated",
            "fixture": arguments,
        }

    def prepare(source, **arguments):
        calls.append((source, arguments))
        return _result(tmp_path, source.split("-")[0])

    monkeypatch.setattr(operator_codegen, "render_jit_m0_device_module", render_m0)
    monkeypatch.setattr(operator_codegen, "render_jit_r0_device_module", render_r0)
    monkeypatch.setattr(jit_runtime, "prepare_nvrtc_jit_artifact", prepare)

    modules, diagnostics = operator_artifact.prepare_low_memory_operator_modules(
        evaluator,
        model_data={
            "execution_contracts": {
                "M0": {"fixture": "m0"},
                "R0": {"fixture": "r0"},
            }
        },
        precision="float64",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=2,
        cache_root=tmp_path / "cache",
    )

    assert [call[0] for call in calls] == ["m0-source", "r0-source"]
    assert all(call[1]["cache_root"] == tmp_path / "cache" for call in calls)
    assert evaluator.loads == [
        ("M0", str(tmp_path / "m0.cubin"), "chunk32", 8),
        ("R0", str(tmp_path / "r0.cubin"), 8),
    ]
    assert modules["M0"]["artifact_id"] == "m0-artifact"
    assert modules["R0"]["artifact_id"] == "r0-artifact"
    assert modules["M0"]["target"] == "sm_120"
    assert modules["M0"]["persistent_blocks_per_compute_unit"] == 8
    assert diagnostics == ("compiled-m0", "compiled-r0")


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
            "implementation": "builtin",
            "compiler": None,
            "artifact_id": "builtin-r0",
        },
    }


def test_prepare_operator_modules_builds_missing_host_m0(monkeypatch, tmp_path):
    evaluator = _Evaluator(r0="builtin")
    calls = []

    def render_m0(contract, **arguments):
        assert contract == {"fixture": "m0"}
        return "host-m0-source", {
            "artifact_id": "m0-host-generated",
            "fixture": arguments,
        }

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
    assert diagnostics == ("compiled-host-m0",)


def test_prepare_operator_modules_can_replace_builtin_host_m0(monkeypatch, tmp_path):
    evaluator = _Evaluator(m0="builtin", r0="builtin")

    monkeypatch.setattr(
        operator_codegen,
        "render_jit_m0_host_plugin",
        lambda contract, **_arguments: ("host-m0-source", {"artifact_id": "m0"}),
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
        prefer_host_m0_plugin=True,
        cache_root=tmp_path / "cache",
    )

    assert evaluator.loads == [("M0-host", str(tmp_path / "host-m0-source.cubin"))]
    assert modules["M0"]["implementation"] == "host_plugin"
    assert modules["R0"]["status"] == "builtin"
