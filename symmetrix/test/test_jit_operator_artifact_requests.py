from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import symmetrix as symmetrix_package
import symmetrix.jit as jit_runtime
import symmetrix.mace_jit_operator_artifact as operator_artifact
import symmetrix.mace_jit_operator_codegen as operator_codegen


class _LayeredEvaluator:
    m0_implementation = "generic"
    m1_implementation = "generic"
    m0_device_module_ready = False
    m1_device_module_ready = False
    m0_device_module_artifact_id = ""
    m1_device_module_artifact_id = ""

    def __init__(self):
        self.loads = []

    def _load_m0_device_module(self, path, schedule, blocks):
        self.loads.append(("M0", Path(path).name, schedule, blocks))
        self.m0_device_module_ready = True
        self.m0_device_module_artifact_id = Path(path).stem

    def _load_m1_device_module(self, path, schedule, blocks):
        self.loads.append(("M1", Path(path).name, schedule, blocks))
        self.m1_device_module_ready = True
        self.m1_device_module_artifact_id = Path(path).stem


def _metadata(name, operator):
    return {
        "schema": "symmetrix.jit.operator-module/1",
        "abi_version": 2,
        "artifact_id": f"{name}-generated",
        "generated_identity": {"operator": operator, "artifact_id": name},
        "structure_fingerprint": "sha256:" + name[-1] * 64,
        "capabilities": 0x1F,
        "schedule": "chunk32",
        "contract": {"channels": 7},
    }


def test_actual_planner_keeps_layer_requests_distinct_across_prebuilt_and_cache(
    monkeypatch, tmp_path
):
    evaluator = _LayeredEvaluator()
    render_calls = []
    catalog_calls = []

    monkeypatch.setattr(
        symmetrix_package,
        "symmetrix",
        SimpleNamespace(_backend_build_info=lambda: {}),
        raising=False,
    )
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **_arguments: None,
    )

    def render_m0(contract, **_arguments):
        render_calls.append((contract["name"], "M0"))
        return f"source-{contract['name']}", _metadata(contract["name"], "M0")

    def render_m1(contract, **_arguments):
        render_calls.append((contract["name"], "M1"))
        return f"source-{contract['name']}", _metadata(contract["name"], "M1")

    monkeypatch.setattr(operator_codegen, "render_jit_m0_device_module", render_m0)
    monkeypatch.setattr(operator_codegen, "render_jit_m1_device_module", render_m1)

    def resolve_installed(**arguments):
        name = arguments["identity"]["artifact_id"]
        catalog_calls.append((name, arguments["operator"]))
        if name == "layer0":
            path = tmp_path / "layer0-prebuilt.cubin"
            arguments["loader"](path)
            return SimpleNamespace(
                status="resolved",
                artifact_path=path,
                artifact_id="layer0-prebuilt",
                diagnostics=("origin=prebuilt status=loaded",),
            )
        status = "rejected" if name == "layer1" else "absent"
        return SimpleNamespace(
            status=status,
            diagnostics=(f"origin=prebuilt status={status}",),
        )

    monkeypatch.setattr(
        operator_artifact, "resolve_installed_operator_artifact", resolve_installed
    )

    def prepare(source, **_arguments):
        name = source.removeprefix("source-")
        operator = "m1" if name == "layer2" else "m0"
        assert _arguments["artifact_name"] == f"execution_{operator}_cuda_module"
        return SimpleNamespace(
            status="cached" if name == "layer1" else "built",
            cache_key=f"cache-{name}",
            artifact_path=tmp_path / f"{name}.cubin",
            diagnostics=(f"prepared-{name}",),
        )

    monkeypatch.setattr(jit_runtime, "prepare_nvrtc_jit_artifact", prepare)

    def load_validated(prepare_artifact, load, **_arguments):
        result = prepare_artifact()
        load(result.artifact_path)
        return SimpleNamespace(
            status="loaded",
            result=result,
            diagnostics=result.diagnostics,
            quarantines=(),
            error=None,
        )

    monkeypatch.setattr(jit_runtime, "load_validated_jit_artifact", load_validated)

    requests = (
        operator_artifact._OperatorArtifactRequest(
            request_key="product_0", operator="M0", contract={"name": "layer0"}
        ),
        operator_artifact._OperatorArtifactRequest(
            request_key="product_1", operator="M0", contract={"name": "layer1"}
        ),
        operator_artifact._OperatorArtifactRequest(
            request_key="product_2", operator="M1", contract={"name": "layer2"}
        ),
    )
    plan = operator_artifact._plan_low_memory_operator_modules(
        evaluator,
        model_data={},
        precision="float64",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=2,
        operator_requests=requests,
        cache_root=tmp_path / "cache",
    )

    assert render_calls == [("layer0", "M0"), ("layer1", "M0"), ("layer2", "M1")]
    assert catalog_calls == [("layer0", "M0"), ("layer1", "M0"), ("layer2", "M1")]
    assert set(plan.modules) == {"product_0"}
    assert [(job.request_key, job.operator) for job in plan.jobs] == [
        ("product_1", "M0"),
        ("product_2", "M1"),
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = operator_artifact._submit_operator_artifact_jobs(executor, plan.jobs)
    assert set(futures) == {"product_1", "product_2"}

    modules, diagnostics = operator_artifact._load_low_memory_operator_modules(
        evaluator, plan, prepared=futures
    )

    assert {key: value["operator"] for key, value in modules.items()} == {
        "product_0": "M0",
        "product_1": "M0",
        "product_2": "M1",
    }
    assert modules["product_0"]["origin"] == "prebuilt"
    assert modules["product_1"]["origin"] == "cache"
    assert modules["product_2"]["origin"] == "runtime"
    assert any("stage=product_1 operator=M0" in line for line in diagnostics)
    assert any("stage=product_2 operator=M1" in line for line in diagnostics)


def test_planner_and_future_submission_reject_duplicate_request_keys():
    duplicate_requests = (
        operator_artifact._OperatorArtifactRequest("product_0", "M0", {}),
        operator_artifact._OperatorArtifactRequest("product_0", "M1", {}),
    )

    with pytest.raises(ValueError, match="duplicate operator request key: product_0"):
        operator_artifact._plan_low_memory_operator_modules(
            SimpleNamespace(),
            model_data={},
            precision="float64",
            backend="cuda",
            target={"compute_capability_code": 120},
            jit_generation_version=2,
            operator_requests=duplicate_requests,
        )

    submitted = []
    executor = SimpleNamespace(submit=lambda prepare: submitted.append(prepare))
    duplicate_jobs = (
        SimpleNamespace(request_key="product_0", prepare=lambda: None),
        SimpleNamespace(request_key="product_0", prepare=lambda: None),
    )
    with pytest.raises(ValueError, match="duplicate operator request key: product_0"):
        operator_artifact._submit_operator_artifact_jobs(executor, duplicate_jobs)
    assert submitted == []


def test_builtin_request_record_uses_layer_key(monkeypatch):
    monkeypatch.setattr(
        symmetrix_package,
        "symmetrix",
        SimpleNamespace(_backend_build_info=lambda: {}),
        raising=False,
    )
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **_arguments: None,
    )
    evaluator = SimpleNamespace(
        m0_implementation="builtin",
        m0_module_id="builtin-m0",
    )

    plan = operator_artifact._plan_low_memory_operator_modules(
        evaluator,
        model_data={},
        precision="float64",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=2,
        operator_requests=(
            operator_artifact._OperatorArtifactRequest(
                "product_0", "M0", {"name": "layer0"}
            ),
        ),
    )

    assert plan.jobs == ()
    assert plan.modules == {
        "product_0": {
            "status": "builtin",
            "implementation": "builtin",
            "compiler": None,
            "artifact_id": "builtin-m0",
            "operator": "M0",
        }
    }
