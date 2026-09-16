import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPOSITORY / "benchmarks" / "validate_cuda_hip_ab.py"


@pytest.fixture(scope="module")
def validator():
    specification = importlib.util.spec_from_file_location(
        "symmetrix_cuda_hip_ab_validator_test", VALIDATOR_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _runs(prefix, samples, cold, artifact):
    return [
        {
            "process_id": f"{prefix}-{index}",
            "fresh_process": True,
            "warmup_count": 20,
            "samples_us_per_atom": list(samples),
            "cold_compile_load_ms": cold,
            "artifact_size_bytes": artifact,
        }
        for index in range(3)
    ]


def _record(compiler, commit, samples, cold, artifact, *, candidate=False):
    return {
        "provenance": {
            "device_id": "0000:01:00.0",
            "architecture": "gfx1151",
            "driver_version": "7.14",
            "runtime_version": "7.14",
            "compiler": compiler,
            "kokkos_version": "5.0.2",
            "commit": commit,
            "model_sha256": "sha256:" + "a" * 64,
        },
        "correctness": {
            "passed": True,
            "nonfinite_count": 0,
            "validation_contract": {
                "energy_atol": 0.0005,
                "force_atol": 0.0005,
            },
        },
        "path": {
            "active": True,
            "backend": "hip",
            "compiler": compiler,
            "artifact_id": f"artifact-{compiler}",
            "launch_order": ["forward", "source", "edge"],
            "forward_launches_per_evaluation": 1,
            "reverse_launches_per_evaluation": 2,
            "synchronization_count": 0,
            "host_device_transfer_bytes": 0,
        },
        "resources": {
            "workspace_bytes": 4096,
            "peak_device_memory_bytes": 100_500 if candidate else 100_000,
            "registers_per_thread": 64,
            "resident_blocks": 4,
            "local_memory_bytes": 0,
            "spill_loads": 0,
            "spill_stores": 0,
            "stack_bytes": 0,
            "dynamic_shared_memory_bytes": 0,
        },
        "process_runs": _runs(
            "candidate" if candidate else "baseline", samples, cold, artifact
        ),
    }


def _document(backend="hip", workload="ordinary_r1"):
    baseline = _record("hipcc", "base", [40.0] * 20, 100.0, 1000)
    candidate = _record("hiprtc", "candidate", [40.4] * 20, 104.0, 1040, candidate=True)
    for record in (baseline, candidate):
        record["path"]["backend"] = backend
        record["provenance"]["architecture"] = (
            "sm_90" if backend == "cuda" else "gfx1151"
        )
    if backend == "cuda":
        baseline["path"].update(compiler="nvcc", artifact_id="artifact-nvcc")
        candidate["path"].update(compiler="nvrtc", artifact_id="artifact-nvrtc")
        baseline["provenance"]["compiler"] = "nvcc 13.1"
        candidate["provenance"]["compiler"] = "NVRTC 13.1"
    return {
        "schema": "symmetrix.factorized.cuda-hip-ab",
        "version": 1,
        "cases": [
            {
                "name": f"{backend}-r1",
                "backend": backend,
                "workload": workload,
                "process_order_policy": "randomized",
                "execution_order": [
                    "baseline:baseline-0",
                    "candidate:candidate-0",
                    "candidate:candidate-1",
                    "baseline:baseline-1",
                    "baseline:baseline-2",
                    "candidate:candidate-2",
                ],
                "baseline": baseline,
                "candidate": candidate,
            }
        ],
    }


@pytest.mark.parametrize("backend", ("cuda", "hip"), ids=("nvidia", "amd"))
def test_validator_accepts_complete_fresh_process_evidence(validator, backend):
    summary = validator.validate_document(_document(backend))[0]

    assert summary["backend"] == backend
    assert summary["median_slowdown_percent"] == pytest.approx(1.0)
    assert summary["median_slowdown_upper_95_percent"] == pytest.approx(1.0)
    assert summary["device_memory_growth_percent"] == pytest.approx(0.5)
    assert summary["absolute_limit_us_per_atom"] == (
        None if backend == "cuda" else 50.0
    )


def _candidate(document):
    return document["cases"][0]["candidate"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: _candidate(document).update(
                process_runs=_runs("candidate", [41.2] * 20, 104, 1040)
            ),
            "upper 95% bound",
        ),
        (
            lambda document: _candidate(document).update(
                process_runs=_runs("candidate", [42.4] * 20, 104, 1040)
            ),
            "hard 5% gate",
        ),
        (
            lambda document: _candidate(document).update(
                process_runs=_runs("candidate", [40.0] * 17 + [44.8] * 3, 104, 1040)
            ),
            "hard 10% gate",
        ),
        (
            lambda document: _candidate(document).update(
                process_runs=_runs("candidate", [40.4] * 20, 106, 1040)
            ),
            "cold compile/load",
        ),
        (
            lambda document: _candidate(document).update(
                process_runs=_runs("candidate", [40.4] * 20, 104, 1060)
            ),
            "artifact size",
        ),
        (
            lambda document: _candidate(document)["correctness"].update(passed=False),
            "correctness did not pass",
        ),
        (
            lambda document: _candidate(document)["path"].update(
                reverse_launches_per_evaluation=3
            ),
            "path field changed",
        ),
        (
            lambda document: _candidate(document)["path"].update(
                synchronization_count=1
            ),
            "path field changed: synchronization_count",
        ),
        (
            lambda document: _candidate(document)["resources"].update(
                workspace_bytes=4097
            ),
            "explicit workspace changed",
        ),
        (
            lambda document: _candidate(document)["resources"].update(
                peak_device_memory_bytes=101_100
            ),
            "device memory grew",
        ),
        (
            lambda document: _candidate(document)["resources"].update(
                local_memory_bytes=1
            ),
            "resource grew: local_memory_bytes",
        ),
        (
            lambda document: _candidate(document)["resources"].update(
                registers_per_thread=65, resident_blocks=3
            ),
            "register growth reduced resident blocks",
        ),
        (
            lambda document: _candidate(document)["resources"].update(
                dynamic_shared_memory_bytes=1, resident_blocks=3
            ),
            "shared-memory growth reduced resident blocks",
        ),
        (
            lambda document: _candidate(document).update(process_runs=[]),
            "at least 3 runs",
        ),
        (
            lambda document: document["cases"][0].update(execution_order=[]),
            "list every fresh process",
        ),
    ],
    ids=(
        "confidence",
        "median-hard",
        "p90-hard",
        "cold",
        "artifact",
        "correctness",
        "launch-count",
        "synchronization",
        "workspace",
        "memory",
        "local-memory",
        "occupancy",
        "shared-memory-occupancy",
        "fresh-process-count",
        "process-order",
    ),
)
def test_validator_rejects_each_relative_or_resource_gate(validator, mutation, message):
    document = _document()
    mutation(document)

    with pytest.raises(validator.QualificationError, match=message):
        validator.validate_document(document)


@pytest.mark.parametrize(
    ("workload", "latency", "limit"),
    (("ordinary_r1", 50.1, 50), ("qualified_macefield", 45.1, 45)),
)
def test_validator_preserves_hip_absolute_limits(validator, workload, latency, limit):
    document = _document(workload=workload)
    baseline = document["cases"][0]["baseline"]
    candidate = _candidate(document)
    baseline["process_runs"] = _runs("baseline", [latency] * 20, 100, 1000)
    candidate["process_runs"] = _runs("candidate", [latency] * 20, 104, 1040)

    with pytest.raises(validator.QualificationError, match=rf"exceeds {limit} us/atom"):
        validator.validate_document(document)


def test_validator_is_deterministic(validator):
    document = _document()

    first = validator.validate_document(copy.deepcopy(document))
    second = validator.validate_document(copy.deepcopy(document))

    assert first == second


def test_cli_writes_machine_readable_summary(validator, tmp_path):
    input_path = tmp_path / "qualification.json"
    output_path = tmp_path / "validation.json"
    input_path.write_text(json.dumps(_document()))

    assert validator.main([str(input_path), "--output", str(output_path)]) == 0

    summary = json.loads(output_path.read_text())
    assert summary["status"] == "passed"
    assert summary["cases"][0]["backend"] == "hip"
