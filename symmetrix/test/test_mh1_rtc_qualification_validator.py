import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPOSITORY / "benchmarks" / "validate_mh1_rtc_qualification.py"


@pytest.fixture(scope="module")
def validator():
    specification = importlib.util.spec_from_file_location(
        "symmetrix_mh1_rtc_validator_test", VALIDATOR_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _runs(role, value):
    return [
        {
            "process_id": f"{role}-{index}",
            "fresh_process": True,
            "warmup_count": 20,
            "samples_ms": [value] * 20,
            "cold_compile_load_ms": 100.0 + index,
        }
        for index in range(3)
    ]


def _record(role, value):
    model_hash = ("a" if role == "mh0" else "b") * 64
    return {
        "provenance": {
            "device_ordinal": 0,
            "device_name": "Radeon 8060S",
            "architecture": "gfx1151",
            "raw_agent_target": "gfx1151",
            "native_subgroup_width": 32,
            "runtime_version": "7.14",
            "driver_version": "7.14",
            "commit": "abc123",
            "model_sha256": model_hash,
        },
        "correctness": {
            "passed": True,
            "nonfinite_count": 0,
            "validation_contract": {"oracle": "generic", "force_atol": 5e-4},
        },
        "path": {
            "active": True,
            "backend": "hip",
            "compiler": "hiprtc",
            "fallback": False,
            "artifact_id": f"artifact-{role}",
            "execution_backend": "factorized" if role == "mh0" else "generated_hip_v4",
            "schedule_id": f"schedule-{role}",
        },
        "graph": {"atoms": 864, "directed_edges": 78624, "prepared": True},
        "resources": {
            "workspace_bytes": 4096,
            "peak_device_memory_bytes": 1_000_000,
            "artifact_size_bytes": 10_000,
            "launches_per_evaluation": 3,
            "local_memory_bytes": 0,
            "spill_loads": 0,
            "spill_stores": 0,
            "stack_bytes": 0,
        },
        "process_runs": _runs(role, value),
    }


def _document():
    return {
        "schema": "symmetrix.direct.mh1-rtc-qualification",
        "version": 1,
        "backend": "hip",
        "process_order_policy": "balanced",
        "execution_order": [
            "mh0:mh0-0",
            "mh1:mh1-0",
            "mh1:mh1-1",
            "mh0:mh0-1",
            "mh0:mh0-2",
            "mh1:mh1-2",
        ],
        "contract": {
            "structure_id": "wurtzite-AlN-6x6x6",
            "head": "omat_pbe",
            "atoms": 864,
            "directed_edges": 78624,
            "dtype": "float32",
            "timing_scope": "prepared graph, synchronized native energy and forces",
            "mh0_model_sha256": "a" * 64,
            "mh1_model_sha256": "b" * 64,
        },
        "mh0": _record("mh0", 25.0),
        "mh1": _record("mh1", 60.0),
    }


def test_accepts_complete_matched_rtc_evidence(validator):
    summary = validator.validate_document(_document())

    assert summary["backend"] == "hip"
    assert summary["mh1_to_mh0_ratio"] == pytest.approx(2.4)
    assert summary["mh1_to_mh0_ratio_upper_95"] == pytest.approx(2.4)
    assert summary["processes_per_model"] == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["mh1"]["path"].update(compiler="hipcc"), "must be hiprtc"),
        (lambda value: value["mh1"]["path"].update(fallback=True), "fallback=false"),
        (lambda value: value["mh1"]["path"].update(active=False), "not active"),
        (lambda value: value["mh1"]["graph"].update(directed_edges=1), "edge count"),
        (
            lambda value: value["mh1"]["provenance"].update(architecture="gfx1100"),
            "provenance mismatch",
        ),
        (lambda value: value["mh1"].update(process_runs=[]), "at least 3"),
        (
            lambda value: value["mh1"]["process_runs"][0].update(warmup_count=19),
            "at least 20 warmups",
        ),
        (
            lambda value: value["mh1"]["process_runs"][0].update(
                samples_ms=[60.0] * 19
            ),
            "at least 20 samples",
        ),
        (lambda value: value.update(execution_order=[]), "every fresh process"),
        (
            lambda value: value["mh1"].update(process_runs=_runs("mh1", 63.0)),
            "upper 95% ratio",
        ),
    ],
)
def test_rejects_invalid_or_slow_evidence(validator, mutation, message):
    document = _document()
    mutation(document)

    with pytest.raises(validator.QualificationError, match=message):
        validator.validate_document(document)


def test_cuda_requires_nvrtc(validator):
    document = _document()
    document["backend"] = "cuda"
    for role in ("mh0", "mh1"):
        document[role]["path"].update(backend="cuda", compiler="nvrtc")
        document[role]["provenance"].update(
            architecture="sm_120", raw_agent_target="sm_120"
        )

    assert validator.validate_document(document)["backend"] == "cuda"

    document["mh1"]["path"]["compiler"] = "nvcc"
    with pytest.raises(validator.QualificationError, match="must be nvrtc"):
        validator.validate_document(document)


def test_validation_is_deterministic(validator):
    first = validator.validate_document(copy.deepcopy(_document()))
    second = validator.validate_document(copy.deepcopy(_document()))

    assert first == second


def test_cli_writes_summary(validator, tmp_path):
    input_path = tmp_path / "qualification.json"
    output_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_document()), encoding="utf-8")

    assert validator.main([str(input_path), "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "passed"
