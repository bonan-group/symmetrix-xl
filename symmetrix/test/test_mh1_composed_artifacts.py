from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import symmetrix.jit as jit_runtime
import symmetrix.mace_jit_operator_artifact as operator_artifact
from symmetrix.mace_execution_contract import (
    make_generated_m1_contract,
    make_standard_m0_contract,
)
from symmetrix.mace_jit_operator_artifact import (
    _mh1_product_artifact_requests,
    _validate_mh1_product_module_records,
)
from symmetrix.mace_nonlinear_jit_codegen import (
    MH1_CUDA_MODULE_ABI,
    MH1_CUDA_MODULE_ABI_VERSION,
    MH1_CUDA_PLUGIN_V5_ABI,
    MH1_CUDA_PLUGIN_V5_ABI_VERSION,
    MH1_HIP_MODULE_ABI,
    MH1_HIP_MODULE_ABI_VERSION,
    MH1_HOST_PLUGIN_V7_ABI,
    MH1_HOST_PLUGIN_V7_ABI_VERSION,
    MH1_LAUNCH_PLAN_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]


def test_composed_product_generation_retires_generation_18():
    assert jit_runtime.JIT_GENERATION_VERSION == 19


def _contracts():
    m0 = make_standard_m0_contract(
        channels=7,
        type_count=1,
        input_l_max=2,
        output_l_max=1,
        correlation=3,
        monomials={index: [[index], [0, index], [0, 0, index]] for index in range(4)},
    )
    m1 = make_generated_m1_contract(
        channels=7,
        type_count=1,
        input_l_max=2,
        correlation=3,
        monomials=[[0], [0, 0], [0, 0, 0]],
    )
    return (m0, m1)


def _records(requests, *, backend="cuda", precision="float32", target="sm_120"):
    return {
        request.request_key: {
            "status": "loaded",
            "operator": request.operator,
            "layer": request.layer,
            "artifact_id": request.contract["generation_fingerprint"],
            "generation_fingerprint": request.contract["generation_fingerprint"],
            "semantic_fingerprint": request.contract["semantic_fingerprint"],
            "structure_fingerprint": request.contract["structure_fingerprint"],
            "precision": precision,
            "backend": backend,
            "target": target,
            "schedule": "chunk32",
            "channels": request.contract["channels"],
            "type_count": request.contract["type_count"],
            "input_components": request.contract["input_components"],
            "output_components": request.contract["output_components"],
            "correlation": request.contract["correlation"],
            "term_count": sum(
                group["term_count"] for group in request.contract["monomial_groups"]
            ),
            "coefficient_count": request.contract["channels"]
            * sum(group["term_count"] for group in request.contract["monomial_groups"]),
        }
        for request in requests
    }


def test_mh1_product_requests_are_layer_qualified_and_final_is_m1():
    requests = _mh1_product_artifact_requests(_contracts())

    assert [(item.request_key, item.operator, item.layer) for item in requests] == [
        ("product_0", "M0", 0),
        ("product_1", "M1", 1),
    ]
    assert all(item.contract["type_count"] == 1 for item in requests)


def test_mh1_product_requests_allow_keep_last_final_m0():
    m0, _ = _contracts()
    requests = _mh1_product_artifact_requests((m0, m0))

    assert [(item.request_key, item.operator, item.layer) for item in requests] == [
        ("product_0", "M0", 0),
        ("product_1", "M0", 1),
    ]


def test_mh1_product_record_set_is_complete_and_identity_qualified():
    requests = _mh1_product_artifact_requests(_contracts())
    records = _records(requests)

    validated = _validate_mh1_product_module_records(
        records,
        requests,
        backend="cuda",
        precision="float32",
        target="sm_120",
    )

    assert tuple(validated) == ("product_0", "product_1")


def test_mh1_device_product_records_reject_noncanonical_schedule():
    requests = _mh1_product_artifact_requests(_contracts())
    records = _records(requests)
    records["product_0"]["schedule"] = "table"

    with pytest.raises(RuntimeError, match="schedule"):
        _validate_mh1_product_module_records(
            records,
            requests,
            backend="cuda",
            precision="float32",
            target="sm_120",
        )


def test_mh1_product_records_reject_inconsistent_coefficient_count():
    requests = _mh1_product_artifact_requests(_contracts())
    records = _records(requests)
    records["product_0"]["coefficient_count"] -= 1

    with pytest.raises(RuntimeError, match="coefficient count"):
        _validate_mh1_product_module_records(
            records,
            requests,
            backend="cuda",
            precision="float32",
            target="sm_120",
        )


def test_mh1_device_product_planner_rejects_table_schedule(monkeypatch, tmp_path):
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "table")

    with pytest.raises(RuntimeError, match="canonical schedule"):
        operator_artifact._plan_low_memory_operator_modules(
            SimpleNamespace(),
            model_data={},
            precision="float32",
            backend="cuda",
            target={"compute_capability_code": 120},
            jit_generation_version=2,
            operator_requests=_mh1_product_artifact_requests(_contracts()),
            cache_root=tmp_path / "cache",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records.pop("product_1"), "incomplete"),
        (lambda records: records["product_0"].update(layer=1), "layer"),
        (lambda records: records["product_0"].update(operator="M1"), "kind"),
        (lambda records: records["product_0"].update(channels=8), "dimensions"),
        (lambda records: records["product_0"].update(type_count=2), "type count"),
        (lambda records: records["product_0"].update(precision="float64"), "precision"),
        (lambda records: records["product_0"].update(target="sm_90"), "target"),
        (
            lambda records: records["product_0"].update(
                structure_fingerprint="sha256:stale"
            ),
            "identity",
        ),
        (lambda records: records["product_0"].update(status="builtin"), "loaded"),
    ],
)
def test_mh1_product_record_validation_fails_closed(mutation, message):
    requests = _mh1_product_artifact_requests(_contracts())
    records = _records(requests)
    mutation(records)

    with pytest.raises(RuntimeError, match=message):
        _validate_mh1_product_module_records(
            records,
            requests,
            backend="cuda",
            precision="float32",
            target="sm_120",
        )


def test_mh1_split_phase_abi_versions_retire_all_previous_generations():
    assert (MH1_HOST_PLUGIN_V7_ABI, MH1_HOST_PLUGIN_V7_ABI_VERSION) == (
        "symmetrix.jit.mh1.host-plugin/7",
        7,
    )
    assert (MH1_CUDA_PLUGIN_V5_ABI, MH1_CUDA_PLUGIN_V5_ABI_VERSION) == (
        "symmetrix.jit.mh1.cuda-plugin/5",
        5,
    )
    assert (MH1_CUDA_MODULE_ABI, MH1_CUDA_MODULE_ABI_VERSION) == (
        "symmetrix.jit.mh1.cuda-module/5",
        5,
    )
    assert (MH1_HIP_MODULE_ABI, MH1_HIP_MODULE_ABI_VERSION) == (
        "symmetrix.jit.mh1.hip-module/5",
        5,
    )
    assert MH1_LAUNCH_PLAN_VERSION == 6


def test_split_phase_headers_do_not_give_plugins_product_coefficients():
    package = ROOT / "symmetrix/source/symmetrix"
    host = (package / "mace_nonlinear_jit_host_plugin_abi.h").read_text()
    cuda = (package / "mace_nonlinear_jit_cuda_plugin_abi.h").read_text()
    host_v7 = host.split("typedef struct SymmetrixJitMH1HostNodeForwardArgsV7", 1)[
        1
    ].split("} SymmetrixJitMH1HostNodeForwardArgsV7;", 1)[0]
    cuda_v5 = cuda.split("typedef struct SymmetrixJitMH1CudaNodeForwardArgsV5", 1)[
        1
    ].split("} SymmetrixJitMH1CudaNodeForwardArgsV5;", 1)[0]

    assert "product_parameters" not in host_v7
    assert "product_parameters" not in cuda_v5
    assert "contraction_output" in host_v7
    assert "contraction_output" in cuda_v5


def test_mh1_product_validation_rejects_swapped_layer_identities():
    requests = _mh1_product_artifact_requests(_contracts())
    records = _records(requests)
    changed = deepcopy(records)
    changed["product_0"]["generation_fingerprint"] = records["product_1"][
        "generation_fingerprint"
    ]

    with pytest.raises(RuntimeError, match="identity"):
        _validate_mh1_product_module_records(
            changed,
            requests,
            backend="cuda",
            precision="float32",
            target="sm_120",
        )


def test_product_plan_loads_each_layer_through_native_composed_loader(
    monkeypatch, tmp_path
):
    requests = _mh1_product_artifact_requests(_contracts())

    class Evaluator:
        def __init__(self):
            self.loaded = {}

        def _load_execution_mh1_product_module(
            self, layer, kind, path, metadata_json, schedule, blocks, backend
        ):
            self.loaded[layer] = (kind, Path(path).stem, schedule, blocks, backend)

        def _execution_mh1_product_module_ready(self, layer):
            return layer in self.loaded

        def _execution_mh1_product_module_artifact_id(self, layer):
            return f"native-product-{layer}"

    evaluator = Evaluator()
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **_arguments: None,
    )
    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        lambda **_arguments: SimpleNamespace(status="absent", diagnostics=()),
    )

    def prepare(_source, **arguments):
        operator = arguments["build"]["operator"]["operator"].lower()
        layer = len(list(tmp_path.glob("*.cubin")))
        path = tmp_path / f"{operator}-{layer}.cubin"
        path.touch()
        return SimpleNamespace(
            status="built",
            cache_key=f"cache-{layer}",
            artifact_path=path,
            diagnostics=(),
        )

    monkeypatch.setattr(jit_runtime, "prepare_nvrtc_jit_artifact", prepare)

    def load_validated(prepare_artifact, load, **_arguments):
        result = prepare_artifact()
        load(result.artifact_path)
        return SimpleNamespace(
            status="loaded",
            result=result,
            diagnostics=(),
            quarantines=(),
            error=None,
        )

    monkeypatch.setattr(jit_runtime, "load_validated_jit_artifact", load_validated)
    plan = operator_artifact._plan_low_memory_operator_modules(
        evaluator,
        model_data={},
        precision="float32",
        backend="cuda",
        target={"compute_capability_code": 120},
        jit_generation_version=2,
        operator_requests=requests,
        cache_root=tmp_path / "cache",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = operator_artifact._submit_operator_artifact_jobs(executor, plan.jobs)
    records, _ = operator_artifact._load_low_memory_operator_modules(
        evaluator, plan, prepared=futures
    )

    validated = _validate_mh1_product_module_records(
        records,
        requests,
        backend="cuda",
        precision="float32",
        target="sm_120",
    )
    assert set(validated) == {"product_0", "product_1"}
    assert [evaluator.loaded[index][0] for index in range(2)] == ["M0", "M1"]


def test_host_product_plan_keeps_layer_adapter_after_runtime_target_retarget(
    monkeypatch, tmp_path
):
    requests = _mh1_product_artifact_requests(_contracts())

    class Evaluator:
        def __init__(self):
            self.loaded = {}

        def _load_execution_mh1_product_module(
            self, layer, kind, path, metadata_json, schedule, blocks, backend
        ):
            self.loaded[layer] = (
                kind,
                Path(path).name,
                __import__("json").loads(metadata_json),
                schedule,
                blocks,
                backend,
            )

        def _execution_mh1_product_module_ready(self, layer):
            return layer in self.loaded

        def _execution_mh1_product_module_artifact_id(self, layer):
            return self.loaded[layer][2]["artifact_id"]

    evaluator = Evaluator()
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: "x86-64-v2" if arguments.get("mode") is None else None,
    )
    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        lambda **_arguments: SimpleNamespace(status="absent", diagnostics=()),
    )
    monkeypatch.setattr(
        jit_runtime, "probe_compiler_identity", lambda: (("c++",), {"id": "GNU"})
    )
    monkeypatch.setattr(
        jit_runtime,
        "resolve_host_jit_policy",
        lambda *_args, **_kwargs: SimpleNamespace(effective_target="x86-64-v3"),
    )

    def prepare(_source, **arguments):
        layer = len(list(tmp_path.glob("*.so")))
        path = tmp_path / f"product-{layer}.so"
        path.touch()
        assert arguments["build"]["operator"]["target"] == "x86-64-v3"
        return SimpleNamespace(
            status="built",
            cache_key=f"cache-{layer}",
            artifact_path=path,
            diagnostics=(),
        )

    monkeypatch.setattr(jit_runtime, "prepare_jit_artifact", prepare)

    def load_validated(prepare_artifact, load, **_arguments):
        result = prepare_artifact()
        load(result.artifact_path)
        return SimpleNamespace(
            status="loaded",
            result=result,
            diagnostics=(),
            quarantines=(),
            error=None,
        )

    monkeypatch.setattr(jit_runtime, "load_validated_jit_artifact", load_validated)
    plan = operator_artifact._plan_low_memory_operator_modules(
        evaluator,
        model_data={},
        precision="float32",
        backend="host",
        target={"cpu_target": "x86-64-v2"},
        jit_generation_version=2,
        operator_requests=requests,
        cache_root=tmp_path / "cache",
    )
    records, _ = operator_artifact._load_low_memory_operator_modules(evaluator, plan)

    assert tuple(records) == ("product_0", "product_1")
    assert [evaluator.loaded[layer][5] for layer in range(2)] == ["host", "host"]
    assert [evaluator.loaded[layer][2]["target"] for layer in range(2)] == [
        "x86-64-v3",
        "x86-64-v3",
    ]


def test_host_product_plan_compiles_runtime_cache_with_one_argument_jobs(
    monkeypatch, tmp_path
):
    requests = _mh1_product_artifact_requests(_contracts())

    class Evaluator:
        def __init__(self):
            self.loaded = {}

        def _load_execution_mh1_product_module(
            self, layer, kind, path, metadata_json, schedule, blocks, backend
        ):
            import json

            assert Path(path).is_file()
            self.loaded[layer] = {
                "kind": kind,
                "metadata": json.loads(metadata_json),
                "schedule": schedule,
                "blocks": blocks,
                "backend": backend,
            }

        def _execution_mh1_product_module_ready(self, layer):
            return layer in self.loaded

        def _execution_mh1_product_module_artifact_id(self, layer):
            return self.loaded[layer]["metadata"]["artifact_id"]

    evaluator = Evaluator()
    monkeypatch.setattr(
        operator_artifact,
        "installed_operator_build_target",
        lambda **arguments: (
            "host-installed-different" if arguments.get("mode") is None else None
        ),
    )
    monkeypatch.setattr(
        operator_artifact,
        "resolve_installed_operator_artifact",
        lambda **_arguments: SimpleNamespace(status="absent", diagnostics=()),
    )
    plan = operator_artifact._plan_low_memory_operator_modules(
        evaluator,
        model_data={},
        precision="float32",
        backend="host",
        target={"cpu_target": "x86-64-v3"},
        jit_generation_version=2,
        operator_requests=requests,
        cache_root=tmp_path / "cache",
    )
    records, _ = operator_artifact._load_low_memory_operator_modules(evaluator, plan)

    assert tuple(records) == ("product_0", "product_1")
    assert all(record["status"] == "built" for record in records.values())
    assert all(item["backend"] == "host" for item in evaluator.loaded.values())
    assert all(
        item["metadata"]["target"] == "host-isa-x86-64-v3"
        for item in evaluator.loaded.values()
    )


def test_product_loading_drains_all_failures_and_clears_partial_native_state(
    monkeypatch,
):
    consumed = []

    class Evaluator:
        def __init__(self):
            self.retained = {0}
            self.cleared = False

        def _clear_execution_mh1_product_modules(self):
            self.retained.clear()
            self.cleared = True

    evaluator = Evaluator()

    def prepare_failure(layer):
        def prepare():
            consumed.append(layer)
            raise RuntimeError(f"compile-{layer}")

        return prepare

    jobs = tuple(
        operator_artifact._OperatorArtifactJob(
            request_key=f"product_{layer}",
            operator="M1" if layer == 2 else "M0",
            compiler="host-cxx",
            module_target="x86-64-v3",
            metadata={},
            schedule="chunk32",
            persistent_blocks=8,
            ready_attribute="unused",
            artifact_attribute="unused",
            prepare=prepare_failure(layer),
            load=lambda _path: None,
            layer=layer,
        )
        for layer in range(3)
    )
    plan = operator_artifact._LowMemoryOperatorPlan(modules={}, jobs=jobs)

    with pytest.raises(
        RuntimeError, match="product artifact composition failed"
    ) as caught:
        operator_artifact._load_low_memory_operator_modules(evaluator, plan)

    assert consumed == [0, 1, 2]
    assert evaluator.cleared
    assert caught.value.records.keys() == {"product_0", "product_1", "product_2"}
    assert all(record["status"] == "failed" for record in caught.value.records.values())
    assert all(f"compile-{layer}" in str(caught.value) for layer in range(3))
