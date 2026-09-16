#!/usr/bin/env python3
"""Validate one CUDA/HIP fresh-process A/B qualification record."""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

SCHEMA = "symmetrix.factorized.cuda-hip-ab"
SCHEMA_VERSION = 1
MIN_PROCESSES = 3
MIN_WARMUPS = 20
MIN_SAMPLES = 20
BOOTSTRAP_DRAWS = 20_000
HIP_ABSOLUTE_LIMITS = {
    "ordinary_r1": 50.0,
    "qualified_macefield": 45.0,
}


class QualificationError(ValueError):
    pass


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be an object")
    return value


def _require_number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or value < 0:
        qualifier = "positive and finite" if positive else "non-negative and finite"
        raise QualificationError(f"{label} must be {qualifier}")
    return value


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        raise QualificationError("cannot summarize empty samples")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _slowdown(candidate, baseline):
    return 100.0 * (candidate / baseline - 1.0)


def _validate_runs(record, label):
    runs = record.get("process_runs")
    if not isinstance(runs, list) or len(runs) < MIN_PROCESSES:
        raise QualificationError(f"{label} requires at least {MIN_PROCESSES} runs")
    process_ids = set()
    validated = []
    for index, run in enumerate(runs):
        run_label = f"{label}.process_runs[{index}]"
        run = _require_mapping(run, run_label)
        process_id = run.get("process_id")
        if not isinstance(process_id, str) or not process_id:
            raise QualificationError(f"{run_label}.process_id must be non-empty")
        if process_id in process_ids:
            raise QualificationError(f"{label} process IDs must be unique")
        process_ids.add(process_id)
        if run.get("fresh_process") is not True:
            raise QualificationError(f"{run_label} must declare fresh_process=true")
        warmups = run.get("warmup_count")
        if isinstance(warmups, bool) or not isinstance(warmups, int):
            raise QualificationError(f"{run_label}.warmup_count must be an integer")
        if warmups < MIN_WARMUPS:
            raise QualificationError(f"{run_label} requires at least 20 warmups")
        samples = run.get("samples_us_per_atom")
        if not isinstance(samples, list) or len(samples) < MIN_SAMPLES:
            raise QualificationError(f"{run_label} requires at least 20 raw samples")
        samples = [
            _require_number(value, f"{run_label}.samples_us_per_atom", positive=True)
            for value in samples
        ]
        validated.append(
            {
                "process_id": process_id,
                "samples": samples,
                "median": statistics.median(samples),
                "cold": _require_number(
                    run.get("cold_compile_load_ms"),
                    f"{run_label}.cold_compile_load_ms",
                    positive=True,
                ),
                "artifact": _require_number(
                    run.get("artifact_size_bytes"),
                    f"{run_label}.artifact_size_bytes",
                    positive=True,
                ),
            }
        )
    return validated


def _bootstrap_median_slowdown_upper(baseline_runs, candidate_runs):
    baseline_medians = [run["median"] for run in baseline_runs]
    candidate_medians = [run["median"] for run in candidate_runs]
    generator = random.Random(0)
    bootstrapped = []
    for _ in range(BOOTSTRAP_DRAWS):
        baseline = statistics.median(
            generator.choice(baseline_medians) for _ in baseline_medians
        )
        candidate = statistics.median(
            generator.choice(candidate_medians) for _ in candidate_medians
        )
        bootstrapped.append(_slowdown(candidate, baseline))
    return _percentile(bootstrapped, 0.95)


def _validate_provenance(case, baseline, candidate):
    shared_fields = (
        "device_id",
        "architecture",
        "driver_version",
        "runtime_version",
        "kokkos_version",
        "model_sha256",
    )
    baseline_provenance = _require_mapping(
        baseline.get("provenance"), "baseline.provenance"
    )
    candidate_provenance = _require_mapping(
        candidate.get("provenance"), "candidate.provenance"
    )
    for field in (*shared_fields, "compiler", "commit"):
        for label, provenance in (
            ("baseline", baseline_provenance),
            ("candidate", candidate_provenance),
        ):
            if not isinstance(provenance.get(field), str) or not provenance[field]:
                raise QualificationError(f"{label}.provenance.{field} is required")
    for field in shared_fields:
        if baseline_provenance[field] != candidate_provenance[field]:
            raise QualificationError(f"provenance mismatch for {field}")
    backend = case.get("backend")
    if backend not in ("cuda", "hip"):
        raise QualificationError("case.backend must be cuda or hip")
    allowed_compilers = {
        "cuda": {"nvcc", "nvrtc"},
        "hip": {"hipcc", "hiprtc"},
    }[backend]
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        path = _require_mapping(record.get("path"), f"{label}.path")
        if path.get("backend") != backend:
            raise QualificationError(f"{label} path backend is not {backend}")
        if path.get("compiler") not in allowed_compilers:
            raise QualificationError(f"{label} compiler path is not native {backend}")


def _validate_correctness_and_path(baseline, candidate):
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        correctness = _require_mapping(
            record.get("correctness"), f"{label}.correctness"
        )
        if correctness.get("passed") is not True:
            raise QualificationError(f"{label} correctness did not pass")
        if correctness.get("nonfinite_count") != 0:
            raise QualificationError(f"{label} correctness found non-finite values")
        path = _require_mapping(record.get("path"), f"{label}.path")
        if path.get("active") is not True:
            raise QualificationError(f"{label} generated path is not active")
        for field in ("compiler", "artifact_id"):
            if not isinstance(path.get(field), str) or not path[field]:
                raise QualificationError(f"{label}.path.{field} is required")
    baseline_contract = _require_mapping(
        baseline["correctness"].get("validation_contract"),
        "baseline.correctness.validation_contract",
    )
    candidate_contract = _require_mapping(
        candidate["correctness"].get("validation_contract"),
        "candidate.correctness.validation_contract",
    )
    if not baseline_contract or baseline_contract != candidate_contract:
        raise QualificationError("correctness validation contract changed")
    exact_path_fields = (
        "launch_order",
        "forward_launches_per_evaluation",
        "reverse_launches_per_evaluation",
        "synchronization_count",
        "host_device_transfer_bytes",
    )
    expected_path = {
        "launch_order": ["forward", "source", "edge"],
        "forward_launches_per_evaluation": 1,
        "reverse_launches_per_evaluation": 2,
    }
    for field, expected in expected_path.items():
        if baseline["path"].get(field) != expected:
            raise QualificationError(f"baseline path field is invalid: {field}")
    for field in ("synchronization_count", "host_device_transfer_bytes"):
        _require_number(baseline["path"].get(field), f"baseline.path.{field}")
        _require_number(candidate["path"].get(field), f"candidate.path.{field}")
    for field in exact_path_fields:
        if baseline["path"].get(field) != candidate["path"].get(field):
            raise QualificationError(f"path field changed: {field}")


def _validate_resources(baseline, candidate):
    base = _require_mapping(baseline.get("resources"), "baseline.resources")
    new = _require_mapping(candidate.get("resources"), "candidate.resources")
    fields = (
        "workspace_bytes",
        "peak_device_memory_bytes",
        "registers_per_thread",
        "resident_blocks",
        "local_memory_bytes",
        "spill_loads",
        "spill_stores",
        "stack_bytes",
        "dynamic_shared_memory_bytes",
    )
    values = {}
    for field in fields:
        values[field] = (
            _require_number(
                base.get(field),
                f"baseline.resources.{field}",
                positive=field
                in (
                    "peak_device_memory_bytes",
                    "registers_per_thread",
                    "resident_blocks",
                ),
            ),
            _require_number(
                new.get(field),
                f"candidate.resources.{field}",
                positive=field
                in (
                    "peak_device_memory_bytes",
                    "registers_per_thread",
                    "resident_blocks",
                ),
            ),
        )
    if values["workspace_bytes"][0] != values["workspace_bytes"][1]:
        raise QualificationError("explicit workspace changed")
    baseline_memory, candidate_memory = values["peak_device_memory_bytes"]
    memory_growth = _slowdown(candidate_memory, baseline_memory)
    if memory_growth > 1.0 + 1e-12:
        raise QualificationError(f"device memory grew by {memory_growth:.3f}% (>1%)")
    for field in ("local_memory_bytes", "spill_loads", "spill_stores", "stack_bytes"):
        if values[field][1] > values[field][0]:
            raise QualificationError(f"resource grew: {field}")
    if values["registers_per_thread"][1] > values["registers_per_thread"][0] and (
        values["resident_blocks"][1] < values["resident_blocks"][0]
    ):
        raise QualificationError("register growth reduced resident blocks")
    if (
        values["dynamic_shared_memory_bytes"][1]
        > values["dynamic_shared_memory_bytes"][0]
        and values["resident_blocks"][1] < values["resident_blocks"][0]
    ):
        raise QualificationError("shared-memory growth reduced resident blocks")
    return memory_growth


def validate_case(case):
    case = _require_mapping(case, "case")
    baseline = _require_mapping(case.get("baseline"), "case.baseline")
    candidate = _require_mapping(case.get("candidate"), "case.candidate")
    _validate_provenance(case, baseline, candidate)
    _validate_correctness_and_path(baseline, candidate)
    memory_growth = _validate_resources(baseline, candidate)
    baseline_runs = _validate_runs(baseline, "baseline")
    candidate_runs = _validate_runs(candidate, "candidate")
    if len(baseline_runs) != len(candidate_runs):
        raise QualificationError("baseline and candidate process counts differ")
    order_policy = case.get("process_order_policy")
    if order_policy not in ("randomized", "balanced_abba"):
        raise QualificationError(
            "case.process_order_policy must be randomized or balanced_abba"
        )
    execution_order = case.get("execution_order")
    expected_processes = {
        *(f"baseline:{run['process_id']}" for run in baseline_runs),
        *(f"candidate:{run['process_id']}" for run in candidate_runs),
    }
    if (
        not isinstance(execution_order, list)
        or len(execution_order) != len(expected_processes)
        or set(execution_order) != expected_processes
    ):
        raise QualificationError(
            "case.execution_order must list every fresh process exactly once"
        )

    baseline_samples = [value for run in baseline_runs for value in run["samples"]]
    candidate_samples = [value for run in candidate_runs for value in run["samples"]]
    median_slowdown = _slowdown(
        statistics.median(candidate_samples), statistics.median(baseline_samples)
    )
    p90_slowdown = _slowdown(
        _percentile(candidate_samples, 0.90), _percentile(baseline_samples, 0.90)
    )
    confidence_upper = _bootstrap_median_slowdown_upper(baseline_runs, candidate_runs)
    cold_slowdown = _slowdown(
        statistics.median(run["cold"] for run in candidate_runs),
        statistics.median(run["cold"] for run in baseline_runs),
    )
    artifact_slowdown = _slowdown(
        statistics.median(run["artifact"] for run in candidate_runs),
        statistics.median(run["artifact"] for run in baseline_runs),
    )

    failures = []
    if median_slowdown > 5.0 + 1e-12:
        failures.append(f"median slowdown {median_slowdown:.3f}% exceeds hard 5% gate")
    if p90_slowdown > 10.0 + 1e-12:
        failures.append(f"p90 slowdown {p90_slowdown:.3f}% exceeds hard 10% gate")
    if confidence_upper > 2.0 + 1e-12:
        failures.append(
            f"median slowdown upper 95% bound {confidence_upper:.3f}% exceeds 2%"
        )
    if cold_slowdown > 5.0 + 1e-12:
        failures.append(f"cold compile/load grew by {cold_slowdown:.3f}% (>5%)")
    if artifact_slowdown > 5.0 + 1e-12:
        failures.append(f"artifact size grew by {artifact_slowdown:.3f}% (>5%)")

    absolute_limit = None
    if case["backend"] == "hip":
        workload = case.get("workload")
        if workload not in HIP_ABSOLUTE_LIMITS:
            raise QualificationError("HIP workload has no fixed absolute limit")
        absolute_limit = HIP_ABSOLUTE_LIMITS[workload]
        candidate_median = statistics.median(candidate_samples)
        if candidate_median > absolute_limit:
            failures.append(
                f"HIP {workload} median {candidate_median:.3f} us/atom exceeds "
                f"{absolute_limit:.0f} us/atom"
            )
    if failures:
        raise QualificationError("; ".join(failures))
    return {
        "name": case.get("name", "unnamed"),
        "backend": case["backend"],
        "median_slowdown_percent": median_slowdown,
        "p90_slowdown_percent": p90_slowdown,
        "median_slowdown_upper_95_percent": confidence_upper,
        "device_memory_growth_percent": memory_growth,
        "cold_compile_load_growth_percent": cold_slowdown,
        "artifact_size_growth_percent": artifact_slowdown,
        "absolute_limit_us_per_atom": absolute_limit,
    }


def validate_document(document):
    document = _require_mapping(document, "document")
    if document.get("schema") != SCHEMA or document.get("version") != SCHEMA_VERSION:
        raise QualificationError("unsupported qualification schema or version")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise QualificationError("document.cases must be a non-empty array")
    return [validate_case(case) for case in cases]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="fresh-process A/B JSON record")
    parser.add_argument("--output", type=Path, help="write validation summary JSON")
    arguments = parser.parse_args(argv)
    try:
        summaries = validate_document(json.loads(arguments.input.read_text()))
    except (OSError, json.JSONDecodeError, QualificationError) as exc:
        parser.exit(1, f"qualification failed: {exc}\n")
    output = json.dumps({"status": "passed", "cases": summaries}, indent=2) + "\n"
    if arguments.output is None:
        sys.stdout.write(output)
    else:
        arguments.output.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
