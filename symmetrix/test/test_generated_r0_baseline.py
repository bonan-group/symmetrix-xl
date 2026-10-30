import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "generated_r0_baseline.py"


@pytest.fixture(scope="module")
def baseline():
    spec = importlib.util.spec_from_file_location(
        "symmetrix_generated_r0_baseline_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _measurement(baseline, case, implementation, latency_ms):
    atoms = 4 if case.graph == "small" else 32
    return baseline.measurement_record(
        case,
        implementation=implementation,
        r0_report={
            "implementation": "generated",
            "backend": "host",
            "artifact_origin": "runtime" if implementation == "generated" else None,
            "schedule": "host-owner" if implementation == "generated" else "frozen",
            "profile": (
                "owner-batched"
                if implementation == "generated"
                else "fixed-weight-coordinate"
            ),
            "artifact_id": (
                "r0_host_test" if implementation == "generated" else "builtin-r0"
            ),
            "qualification_route": (
                "host_plugin" if implementation == "generated" else "builtin"
            ),
        },
        preparation_ms=37.0 if implementation == "generated" else 0.0,
        launch_count=4,
        steady_state_samples_ms=[latency_ms, latency_ms, latency_ms],
        first_call_ms={"energy": 12.0, "forces": 14.0, "stress": 16.0},
        cache={"path": "/task-cache", "empty_before": True},
        workload={
            "model": "compact-mh0",
            "model_sha256": "sha256:" + "a" * 64,
            "atoms": atoms,
            "directed_edges": atoms * 20,
            "cutoff_A": 6.0,
            "skin_A": 0.0,
            "effective_cutoff_A": 6.0,
        },
        provenance={
            "source_commit": "0123456789abcdef",
            "source_dirty": False,
            "source_content_sha256": "sha256:" + "c" * 64,
            "native_build_source_content_sha256": "sha256:" + "d" * 64,
            "extension_sha256": "sha256:" + "b" * 64,
            "hardware": "test-cpu",
            "threads": {"kokkos": 1, "openmp": 1, "blas": 1},
            "benchmark_command": "taskset -c 7 python benchmark.py",
            "generated_r0_artifact": {
                "artifact_id": "r0_host_test",
                "path": "/task-cache/r0_host_test.so",
                "sha256": "sha256:" + "e" * 64,
            },
        },
    )


def test_required_matrix_covers_cpu_precision_weights_graphs_and_paths(baseline):
    cases = baseline.required_cases(("cpu",))

    assert len(cases) == 2 * 2 * 2 * 6
    assert {case.precision for case in cases} == {"float32", "float64"}
    assert {case.edge_weight_mode for case in cases} == {"unity", "weighted"}
    assert {case.graph for case in cases} == {"small", "large"}
    assert {case.path for case in cases} == {
        "energy",
        "forces",
        "stress",
        "density_scaling",
        "direct_harmonic_gradients",
        "prepared_scale_adjoint",
    }


def test_measurement_record_requires_reproducible_cold_and_steady_state_data(
    baseline,
):
    case = baseline.required_cases(("cpu",))[0]
    record = _measurement(baseline, case, "generated", 1.0)

    assert record["case"] == case.as_dict()
    assert record["pipeline"] == {
        "stage": "R0",
        "schedule": "host-owner",
        "implementation": "generated",
        "backend": "host",
        "artifact_origin": "runtime",
        "profile": "owner-batched",
        "artifact_id": "r0_host_test",
        "qualification_route": "host_plugin",
        "qualification_role": "generated",
    }
    assert record["preparation_ms"] == 37.0
    assert record["launch_count"] == 4
    assert record["steady_state"]["median_ms"] == 1.0
    assert record["steady_state"]["primary_metric"] == "us/atom"
    assert record["steady_state"]["median_us_per_atom"] == pytest.approx(250.0)
    assert record["steady_state"]["samples_us_per_atom"] == pytest.approx(
        [250.0, 250.0, 250.0]
    )
    assert record["first_call_ms"] == {
        "energy": 12.0,
        "forces": 14.0,
        "stress": 16.0,
    }
    assert record["cache"]["empty_before"] is True
    assert record["workload"]["atoms"] == 4
    assert record["workload"]["directed_edges"] == 80
    assert record["provenance"]["threads"] == {
        "kokkos": 1,
        "openmp": 1,
        "blas": 1,
    }
    assert record["provenance"]["source_content_sha256"] == "sha256:" + "c" * 64
    assert record["provenance"]["generated_r0_artifact"] == {
        "artifact_id": "r0_host_test",
        "path": "/task-cache/r0_host_test.so",
        "sha256": "sha256:" + "e" * 64,
    }
    assert record["provenance"]["benchmark_command"].startswith("taskset -c 7")

    with pytest.raises(ValueError, match="empty before"):
        baseline.measurement_record(
            case,
            implementation="generated",
            r0_report={
                "implementation": "generated",
                "backend": "host",
                "artifact_origin": "runtime",
                "schedule": "host-owner",
                "profile": "owner-batched",
                "artifact_id": "r0_host_test",
                "qualification_route": "host_plugin",
            },
            preparation_ms=1.0,
            launch_count=1,
            steady_state_samples_ms=[1.0],
            first_call_ms={"energy": 1.0, "forces": 1.0, "stress": 1.0},
            cache={"path": "/reused-cache", "empty_before": False},
            workload={
                "model": "compact-mh0",
                "model_sha256": "sha256:" + "a" * 64,
                "atoms": 32,
                "directed_edges": 640,
                "cutoff_A": 6.0,
                "skin_A": 0.0,
                "effective_cutoff_A": 6.0,
            },
            provenance={
                "source_commit": "0123456789abcdef",
                "source_dirty": False,
                "source_content_sha256": "sha256:" + "c" * 64,
                "native_build_source_content_sha256": "sha256:" + "d" * 64,
                "extension_sha256": "sha256:" + "b" * 64,
                "hardware": "test-cpu",
                "threads": {"kokkos": 1, "openmp": 1, "blas": 1},
                "benchmark_command": "python benchmark.py",
                "generated_r0_artifact": {
                    "artifact_id": "r0_host_test",
                    "path": "/task-cache/r0_host_test.so",
                    "sha256": "sha256:" + "e" * 64,
                },
            },
        )


def test_measure_pipeline_captures_preparation_cold_calls_launches_and_samples(
    baseline,
):
    case = baseline.required_cases(("cpu",))[0]
    events = []
    launches = {"value": 10}
    clock_values = iter(
        (
            0.000,
            0.020,
            0.100,
            0.112,
            0.200,
            0.214,
            0.300,
            0.316,
            0.400,
            0.405,
            0.500,
            0.506,
            0.600,
            0.607,
        )
    )

    def call(name):
        def run():
            events.append(name)
            launches["value"] += 1

        return run

    record = baseline.measure_pipeline(
        case,
        implementation="generated",
        r0_report={
            "implementation": "generated",
            "backend": "host",
            "artifact_origin": "runtime",
            "schedule": "host-owner",
            "profile": "owner-batched",
            "artifact_id": "r0_host_test",
            "qualification_route": "host_plugin",
        },
        prepare=call("prepare"),
        first_calls={name: call(name) for name in ("energy", "forces", "stress")},
        steady_call=call("steady"),
        launch_count=lambda: launches["value"],
        warmups=1,
        samples=3,
        cache={"path": "/empty-task-cache", "empty_before": True},
        workload={
            "model": "compact-mh0",
            "model_sha256": "sha256:" + "a" * 64,
            "atoms": 4,
            "directed_edges": 72,
            "cutoff_A": 6.0,
            "skin_A": 0.0,
            "effective_cutoff_A": 6.0,
        },
        provenance={
            "source_commit": "0123456789abcdef",
            "source_dirty": False,
            "source_content_sha256": "sha256:" + "c" * 64,
            "native_build_source_content_sha256": "sha256:" + "d" * 64,
            "extension_sha256": "sha256:" + "b" * 64,
            "hardware": "test-cpu",
            "threads": {"kokkos": 1, "openmp": 1, "blas": 1},
            "benchmark_command": "python benchmark.py",
            "generated_r0_artifact": {
                "artifact_id": "r0_host_test",
                "path": "/task-cache/r0_host_test.so",
                "sha256": "sha256:" + "e" * 64,
            },
        },
        clock=lambda: next(clock_values),
    )

    assert events == [
        "prepare",
        "energy",
        "forces",
        "stress",
        "steady",
        "steady",
        "steady",
        "steady",
    ]
    assert record["preparation_ms"] == pytest.approx(20.0)
    assert record["first_call_ms"] == pytest.approx(
        {"energy": 12.0, "forces": 14.0, "stress": 16.0}
    )
    assert record["steady_state"]["samples_ms"] == pytest.approx([5.0, 6.0, 7.0])
    assert record["steady_state"]["samples_us_per_atom"] == pytest.approx(
        [1250.0, 1500.0, 1750.0]
    )
    assert record["launch_count"] == 7


def test_accelerator_unavailability_is_an_explicit_skip_with_a_concrete_reason(
    baseline,
):
    case = baseline.required_cases(("cuda",))[0]
    skip = baseline.skipped_case(
        case,
        "no architecture-qualified Kokkos CUDA extension is installed",
    )

    assert skip["status"] == "skipped"
    assert skip["case"]["backend"] == "cuda"
    assert "Kokkos CUDA" in skip["reason"]
    with pytest.raises(ValueError, match="concrete unavailable reason"):
        baseline.skipped_case(case, "unavailable")

    cpu_case = baseline.required_cases(("cpu",))[0]
    with pytest.raises(ValueError, match="accelerator cases"):
        baseline.skipped_case(
            cpu_case,
            "no architecture-qualified Kokkos accelerator extension is installed",
        )


def _complete_cpu_records(baseline, generated_ratio=1.0):
    records = []
    for case in baseline.required_cases(("cpu",)):
        frozen_ms = 1.0 if case.graph == "small" else 8.0
        records.extend(
            (
                _measurement(baseline, case, "frozen", frozen_ms),
                _measurement(
                    baseline,
                    case,
                    "generated",
                    frozen_ms * generated_ratio,
                ),
            )
        )
    return records


def test_retirement_gate_rejects_a_per_case_regression_above_five_percent(baseline):
    case = baseline.required_cases(("cpu",))[0]
    records = _complete_cpu_records(baseline)
    generated = next(
        record
        for record in records
        if record["case"] == case.as_dict()
        and record["pipeline"]["qualification_role"] == "generated"
    )
    generated["steady_state"]["median_us_per_atom"] *= 1.051

    result = baseline.evaluate_retirement_gate(records)

    assert result["status"] == "fail"
    failed = next(item for item in result["cases"] if item["status"] == "fail")
    assert failed["slowdown_percent"] == pytest.approx(5.1)
    assert "R0/host-owner" in failed["diagnostic"]


def test_retirement_gate_rejects_an_incomplete_cpu_matrix(baseline):
    case = baseline.required_cases(("cpu",))[0]
    records = [
        _measurement(baseline, case, "frozen", 1.0),
        _measurement(baseline, case, "generated", 1.0),
    ]

    with pytest.raises(ValueError, match="missing required cases"):
        baseline.evaluate_retirement_gate(records)


def test_retirement_gate_rejects_unmatched_workload_or_hardware(baseline):
    case = baseline.required_cases(("cpu",))[0]
    records = _complete_cpu_records(baseline)
    generated = next(
        record
        for record in records
        if record["case"] == case.as_dict()
        and record["pipeline"]["qualification_role"] == "generated"
    )
    generated["workload"]["atoms"] = 64

    with pytest.raises(ValueError, match="workload mismatch"):
        baseline.evaluate_retirement_gate(records)

    records = _complete_cpu_records(baseline)
    generated = next(
        record
        for record in records
        if record["case"] == case.as_dict()
        and record["pipeline"]["qualification_role"] == "generated"
    )
    generated["provenance"]["hardware"] = "other-cpu"
    with pytest.raises(ValueError, match="hardware mismatch"):
        baseline.evaluate_retirement_gate(records)


def test_retirement_gate_accepts_matched_records_and_reports_aggregate(baseline):
    records = _complete_cpu_records(baseline, generated_ratio=0.99)

    result = baseline.evaluate_retirement_gate(records)

    assert result["status"] == "pass"
    assert result["aggregate"]["primary_metric"] == "us/atom"
    assert result["aggregate"]["generated_over_frozen"] == pytest.approx(0.99)


def test_retirement_gate_aggregates_normalized_cost_not_graph_latency(baseline):
    records = _complete_cpu_records(baseline)
    for record in records:
        # Contradict the normalized values deliberately: the gate must use the
        # declared repository-standard primary metric, not raw milliseconds.
        if record["pipeline"]["qualification_role"] == "generated":
            record["steady_state"]["median_ms"] *= 100.0

    result = baseline.evaluate_retirement_gate(records)

    assert result["status"] == "pass"
    assert result["aggregate"]["generated_over_frozen"] == pytest.approx(1.0)


def test_campaign_accepts_complete_accelerator_skips_and_rejects_missing_cases(
    baseline,
):
    reason = "no architecture-qualified Kokkos accelerator extension is installed"
    records = [
        baseline.skipped_case(case, reason)
        for case in baseline.required_cases(("cuda", "hip"))
    ]

    summary = baseline.validate_campaign(records, backends=("cuda", "hip"))

    assert summary == {
        "schema": baseline.SCHEMA,
        "required_cases": 96,
        "measured_cases": 0,
        "skipped_cases": 96,
        "status": "hardware-unavailable",
    }
    with pytest.raises(ValueError, match="missing required cases"):
        baseline.validate_campaign(records[:-1], backends=("cuda", "hip"))


def test_campaign_revalidates_skip_backend_and_reason(baseline):
    reason = "no architecture-qualified Kokkos accelerator extension is installed"
    records = [
        baseline.skipped_case(case, reason)
        for case in baseline.required_cases(("cuda",))
    ]
    records[0]["reason"] = ""

    with pytest.raises(ValueError, match="concrete unavailable reason"):
        baseline.validate_campaign(records, backends=("cuda",))

    cpu_skips = [
        {
            "schema": baseline.SCHEMA,
            "status": "skipped",
            "case": case.as_dict(),
            "reason": reason,
        }
        for case in baseline.required_cases(("cpu",))
    ]
    with pytest.raises(ValueError, match="accelerator cases"):
        baseline.validate_campaign(cpu_skips, backends=("cpu",))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("implementation", "generic"),
        ("backend", "cuda"),
        ("artifact_origin", None),
        ("schedule", "frozen"),
        ("profile", "fixed-weight-coordinate"),
        ("artifact_id", ""),
        ("artifact_id", "other-generated-artifact"),
        ("qualification_route", "builtin"),
    ),
)
def test_campaign_rejects_pipeline_evidence_mismatches(baseline, field, value):
    records = _complete_cpu_records(baseline)
    generated = next(
        record
        for record in records
        if record["pipeline"]["qualification_role"] == "generated"
    )
    generated["pipeline"][field] = value

    with pytest.raises(ValueError, match="R0 pipeline"):
        baseline.validate_campaign(records, backends=("cpu",))


def test_campaign_rejects_missing_pipeline_evidence_field(baseline):
    records = _complete_cpu_records(baseline)
    generated = next(
        record
        for record in records
        if record["pipeline"]["qualification_role"] == "generated"
    )
    del generated["pipeline"]["schedule"]

    with pytest.raises(ValueError, match="R0 pipeline evidence is incomplete"):
        baseline.validate_campaign(records, backends=("cpu",))
