#!/usr/bin/env python3
"""Schema and retirement gates for frozen-versus-generated R0 benchmarks.

The benchmark runners stay backend-specific.  This module owns the common case
matrix and validates their records, so CPU, CUDA, and HIP evidence cannot drift
to subtly different property or timing boundaries.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "symmetrix.generated-r0-baseline/2"
STAGE = "R0"
PRECISIONS = ("float32", "float64")
EDGE_WEIGHT_MODES = ("unity", "weighted")
GRAPHS = ("small", "large")
PATHS = (
    "energy",
    "forces",
    "stress",
    "density_scaling",
    "direct_harmonic_gradients",
    "prepared_scale_adjoint",
)
BACKENDS = ("cpu", "cuda", "hip")
IMPLEMENTATIONS = ("frozen", "generated")
MAX_CASE_SLOWDOWN = 1.05


@dataclass(frozen=True, order=True)
class BaselineCase:
    backend: str
    precision: str
    edge_weight_mode: str
    graph: str
    path: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.backend,
            self.precision,
            self.edge_weight_mode,
            self.graph,
            self.path,
        )


def required_cases(backends: Sequence[str] = BACKENDS) -> tuple[BaselineCase, ...]:
    """Return the complete, deterministic qualification matrix."""

    unknown = sorted(set(backends) - set(BACKENDS))
    if unknown:
        raise ValueError("unknown R0 baseline backends: " + ", ".join(unknown))
    return tuple(
        BaselineCase(*values)
        for values in product(
            tuple(backends), PRECISIONS, EDGE_WEIGHT_MODES, GRAPHS, PATHS
        )
    )


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive finite" if positive else "nonnegative finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return result


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be a complete SHA-256 identity")
    return value


def _workload_record(workload: Mapping[str, Any]) -> dict[str, Any]:
    model = workload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("workload model must not be empty")
    integer_fields = {}
    for name in ("atoms", "directed_edges"):
        value = workload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"workload {name} must be a positive integer")
        integer_fields[name] = value
    cutoff = _finite_number(workload.get("cutoff_A"), "workload cutoff", positive=True)
    skin = _finite_number(workload.get("skin_A"), "workload skin")
    effective = _finite_number(
        workload.get("effective_cutoff_A"), "workload effective cutoff", positive=True
    )
    if not math.isclose(cutoff + skin, effective, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("workload effective cutoff must equal cutoff plus skin")
    return {
        "model": model,
        "model_sha256": _sha256(workload.get("model_sha256"), "workload model"),
        **integer_fields,
        "cutoff_A": cutoff,
        "skin_A": skin,
        "effective_cutoff_A": effective,
    }


def _provenance_record(provenance: Mapping[str, Any]) -> dict[str, Any]:
    commit = provenance.get("source_commit")
    hardware = provenance.get("hardware")
    dirty = provenance.get("source_dirty")
    if not isinstance(commit, str) or not commit:
        raise ValueError("source commit must not be empty")
    if not isinstance(hardware, str) or not hardware:
        raise ValueError("hardware identity must not be empty")
    if not isinstance(dirty, bool):
        raise ValueError("source_dirty must be boolean")
    benchmark_command = provenance.get("benchmark_command")
    if not isinstance(benchmark_command, str) or not benchmark_command.strip():
        raise ValueError("benchmark command must not be empty")
    artifact = provenance.get("generated_r0_artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("generated R0 artifact provenance is required")
    artifact_id = artifact.get("artifact_id")
    artifact_path = artifact.get("path")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("generated R0 artifact ID must not be empty")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError("generated R0 artifact path must not be empty")
    threads = provenance.get("threads")
    if not isinstance(threads, Mapping) or set(threads) != {
        "kokkos",
        "openmp",
        "blas",
    }:
        raise ValueError("thread provenance must contain kokkos, openmp, and blas")
    normalized_threads = {}
    for name, value in threads.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} thread count must be a positive integer")
        normalized_threads[str(name)] = value
    return {
        "source_commit": commit,
        "source_dirty": dirty,
        "source_content_sha256": _sha256(
            provenance.get("source_content_sha256"), "source content"
        ),
        "native_build_source_content_sha256": _sha256(
            provenance.get("native_build_source_content_sha256"),
            "native build source content",
        ),
        "extension_sha256": _sha256(
            provenance.get("extension_sha256"), "native extension"
        ),
        "hardware": hardware,
        "threads": normalized_threads,
        "benchmark_command": benchmark_command.strip(),
        "generated_r0_artifact": {
            "artifact_id": artifact_id,
            "path": artifact_path,
            "sha256": _sha256(artifact.get("sha256"), "generated R0 artifact"),
        },
    }


def measurement_record(
    case: BaselineCase,
    *,
    implementation: str,
    r0_report: Mapping[str, Any],
    preparation_ms: float,
    launch_count: int,
    steady_state_samples_ms: Sequence[float],
    first_call_ms: Mapping[str, float],
    cache: Mapping[str, Any],
    workload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one reproducible timing record from a backend-specific runner."""

    if implementation not in IMPLEMENTATIONS:
        raise ValueError("implementation must be 'frozen' or 'generated'")
    pipeline = _validate_r0_pipeline(case, implementation, r0_report)
    preparation = _finite_number(preparation_ms, "preparation_ms")
    if isinstance(launch_count, bool) or not isinstance(launch_count, int):
        raise ValueError("launch_count must be a nonnegative integer")
    if launch_count < 0:
        raise ValueError("launch_count must be a nonnegative integer")
    samples = [
        _finite_number(value, "steady-state sample", positive=True)
        for value in steady_state_samples_ms
    ]
    if not samples:
        raise ValueError("at least one steady-state sample is required")
    if set(first_call_ms) != {"energy", "forces", "stress"}:
        raise ValueError("first_call_ms must contain energy, forces, and stress")
    cold = {
        name: _finite_number(
            first_call_ms[name], f"first {name} latency", positive=True
        )
        for name in ("energy", "forces", "stress")
    }
    cache_path = cache.get("path")
    if not isinstance(cache_path, str) or not cache_path:
        raise ValueError("the task-specific cache path must not be empty")
    if cache.get("empty_before") is not True:
        raise ValueError("the task-specific cache must be empty before measurement")
    normalized_workload = _workload_record(workload)
    normalized_provenance = _provenance_record(provenance)
    if (
        implementation == "generated"
        and pipeline["artifact_id"]
        != normalized_provenance["generated_r0_artifact"]["artifact_id"]
    ):
        raise ValueError("R0 pipeline artifact ID does not match provenance")
    samples_us_per_atom = [
        sample * 1000.0 / normalized_workload["atoms"] for sample in samples
    ]
    return {
        "schema": SCHEMA,
        "status": "measured",
        "case": case.as_dict(),
        "pipeline": {
            "stage": STAGE,
            **pipeline,
            "qualification_role": implementation,
        },
        "preparation_ms": preparation,
        "launch_count": launch_count,
        "steady_state": {
            "primary_metric": "us/atom",
            "median_us_per_atom": statistics.median(samples_us_per_atom),
            "samples_us_per_atom": samples_us_per_atom,
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
        },
        "first_call_ms": cold,
        "cache": {"path": cache_path, "empty_before": True},
        "workload": normalized_workload,
        "provenance": normalized_provenance,
    }


def measure_pipeline(
    case: BaselineCase,
    *,
    implementation: str,
    r0_report: Mapping[str, Any],
    prepare: Any,
    first_calls: Mapping[str, Any],
    steady_call: Any,
    launch_count: Any,
    warmups: int,
    samples: int,
    cache: Mapping[str, Any],
    workload: Mapping[str, Any],
    provenance: Mapping[str, Any],
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Measure a backend runner while keeping every timing boundary explicit."""

    for name, value, minimum in (
        ("warmups", warmups, 0),
        ("samples", samples, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    expected_first_calls = {"energy", "forces", "stress"}
    if set(first_calls) != expected_first_calls:
        raise ValueError("first_calls must contain energy, forces, and stress")
    if not all(callable(callback) for callback in first_calls.values()):
        raise ValueError("every first-call entry must be callable")
    if not callable(prepare) or not callable(steady_call) or not callable(launch_count):
        raise ValueError("prepare, steady_call, and launch_count must be callable")

    started = clock()
    prepare()
    preparation_ms = 1000.0 * (clock() - started)
    launches_before = launch_count()
    first_call_ms = {}
    for name in ("energy", "forces", "stress"):
        started = clock()
        first_calls[name]()
        first_call_ms[name] = 1000.0 * (clock() - started)
    for _ in range(warmups):
        steady_call()
    steady_samples = []
    for _ in range(samples):
        started = clock()
        steady_call()
        steady_samples.append(1000.0 * (clock() - started))
    launches_after = launch_count()
    return measurement_record(
        case,
        implementation=implementation,
        r0_report=r0_report,
        preparation_ms=preparation_ms,
        launch_count=launches_after - launches_before,
        steady_state_samples_ms=steady_samples,
        first_call_ms=first_call_ms,
        cache=cache,
        workload=workload,
        provenance=provenance,
    )


def _validate_skip(case: BaselineCase, reason: Any) -> str:
    if case.backend not in {"cuda", "hip"}:
        raise ValueError("only accelerator cases may be skipped")
    generic = {"unavailable", "not available", "missing", "skip", "skipped"}
    if (
        not isinstance(reason, str)
        or len(reason.strip()) < 16
        or reason.strip().lower() in generic
    ):
        raise ValueError("accelerator skips require a concrete unavailable reason")
    return reason.strip()


def skipped_case(case: BaselineCase, reason: str) -> dict[str, Any]:
    """Record unavailable accelerator hardware as explicit non-evidence."""

    normalized_reason = _validate_skip(case, reason)
    return {
        "schema": SCHEMA,
        "status": "skipped",
        "case": case.as_dict(),
        "reason": normalized_reason,
    }


def _case_from_record(record: Mapping[str, Any]) -> BaselineCase:
    case = record.get("case")
    if not isinstance(case, Mapping):
        raise ValueError("R0 baseline record is missing its case")
    try:
        parsed = BaselineCase(**dict(case))
    except TypeError as exc:
        raise ValueError("R0 baseline record has an invalid case") from exc
    if parsed not in required_cases((parsed.backend,)):
        raise ValueError(f"R0 baseline record has an unsupported case: {parsed.key}")
    return parsed


def _validate_r0_pipeline(
    case: BaselineCase, role: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    if role not in IMPLEMENTATIONS:
        raise ValueError("R0 pipeline qualification role is invalid")
    required = {
        "implementation",
        "backend",
        "artifact_origin",
        "schedule",
        "profile",
        "artifact_id",
        "qualification_route",
    }
    if not isinstance(report, Mapping) or set(report) != required:
        raise ValueError("R0 pipeline evidence is incomplete")
    normalized = dict(report)
    expected_backend = "host" if case.backend == "cpu" else case.backend
    if normalized["implementation"] != "generated":
        raise ValueError("R0 pipeline must report implementation=generated")
    if normalized["backend"] != expected_backend:
        raise ValueError("R0 pipeline backend does not match the campaign case")
    if not isinstance(normalized["artifact_id"], str) or not normalized["artifact_id"]:
        raise ValueError("R0 pipeline artifact ID must not be empty")
    if role == "frozen":
        expected = {
            "artifact_origin": None,
            "schedule": "frozen",
            "profile": "fixed-weight-coordinate",
            "qualification_route": "builtin",
        }
        if any(normalized[name] != value for name, value in expected.items()):
            raise ValueError("R0 pipeline frozen route evidence is inconsistent")
    elif case.backend == "cpu":
        expected = {
            "schedule": "host-owner",
            "profile": "owner-batched",
            "qualification_route": "host_plugin",
        }
        if normalized["artifact_origin"] not in {"prebuilt", "cache", "runtime"}:
            raise ValueError("R0 pipeline generated origin is invalid")
        if any(normalized[name] != value for name, value in expected.items()):
            raise ValueError(
                "R0 pipeline generated host route evidence is inconsistent"
            )
    else:
        if normalized["artifact_origin"] not in {"prebuilt", "cache", "runtime"}:
            raise ValueError("R0 pipeline generated origin is invalid")
        if normalized["qualification_route"] != "device_module":
            raise ValueError("R0 pipeline generated device route is inconsistent")
        if normalized["schedule"] in {None, "", "generic", "frozen", "host-owner"}:
            raise ValueError("R0 pipeline generated device schedule is invalid")
        if not isinstance(normalized["profile"], str) or not re.fullmatch(
            r"persistent-[1-9][0-9]*", normalized["profile"]
        ):
            raise ValueError("R0 pipeline generated device profile is invalid")
    return normalized


def evaluate_retirement_gate(
    records: Iterable[Mapping[str, Any]],
    *,
    maximum_slowdown: float = MAX_CASE_SLOWDOWN,
    backends: Sequence[str] = ("cpu",),
) -> dict[str, Any]:
    """Compare matched measured pairs against the R0 retirement threshold."""

    limit = _finite_number(maximum_slowdown, "maximum_slowdown", positive=True)
    if limit < 1.0:
        raise ValueError("maximum_slowdown must be at least one")
    materialized = list(records)
    campaign = validate_campaign(materialized, backends=backends)
    if campaign["measured_cases"] != campaign["required_cases"]:
        raise ValueError("retirement gates require measured records for every case")
    grouped: dict[BaselineCase, dict[str, Mapping[str, Any]]] = {}
    for record in materialized:
        if record.get("status") != "measured":
            raise ValueError("retirement gates require measured records")
        case = _case_from_record(record)
        pipeline = record.get("pipeline")
        if not isinstance(pipeline, Mapping) or pipeline.get("stage") != STAGE:
            raise ValueError("measurement pipeline must identify stage R0")
        if pipeline.get("implementation") != "generated":
            raise ValueError("measurement must use semantic generated R0 reporting")
        implementation = pipeline.get("qualification_role")
        if implementation not in IMPLEMENTATIONS:
            raise ValueError("measurement has an invalid implementation")
        if implementation in grouped.setdefault(case, {}):
            raise ValueError(f"duplicate {implementation} record for {case.key}")
        grouped[case][str(implementation)] = record
    if not grouped:
        raise ValueError("the retirement gate requires at least one matched case")

    results = []
    frozen_total = 0.0
    generated_total = 0.0
    for case in sorted(grouped):
        pair = grouped[case]
        if set(pair) != set(IMPLEMENTATIONS):
            raise ValueError(f"R0 baseline case is missing a matched pair: {case.key}")
        if pair["frozen"].get("workload") != pair["generated"].get("workload"):
            raise ValueError(f"R0 baseline workload mismatch for {case.key}")
        frozen_provenance = pair["frozen"].get("provenance")
        generated_provenance = pair["generated"].get("provenance")
        if not isinstance(frozen_provenance, Mapping) or not isinstance(
            generated_provenance, Mapping
        ):
            raise ValueError(f"R0 baseline provenance is missing for {case.key}")
        if frozen_provenance.get("hardware") != generated_provenance.get("hardware"):
            raise ValueError(f"R0 baseline hardware mismatch for {case.key}")
        if frozen_provenance != generated_provenance:
            raise ValueError(f"R0 baseline provenance mismatch for {case.key}")
        frozen_us_per_atom = _finite_number(
            pair["frozen"]["steady_state"]["median_us_per_atom"],
            "frozen steady-state median",
            positive=True,
        )
        generated_us_per_atom = _finite_number(
            pair["generated"]["steady_state"]["median_us_per_atom"],
            "generated steady-state median",
            positive=True,
        )
        ratio = generated_us_per_atom / frozen_us_per_atom
        schedule = pair["generated"]["pipeline"]["schedule"]
        passed = ratio <= limit
        results.append(
            {
                "case": case.as_dict(),
                "primary_metric": "us/atom",
                "frozen_median_us_per_atom": frozen_us_per_atom,
                "generated_median_us_per_atom": generated_us_per_atom,
                "generated_over_frozen": ratio,
                "slowdown_percent": 100.0 * (ratio - 1.0),
                "status": "pass" if passed else "fail",
                "diagnostic": (
                    f"{STAGE}/{schedule} {case.precision} "
                    f"{case.edge_weight_mode} {case.graph} {case.path}"
                ),
            }
        )
        frozen_total += frozen_us_per_atom
        generated_total += generated_us_per_atom
    aggregate_ratio = generated_total / frozen_total
    passed = all(item["status"] == "pass" for item in results)
    passed = passed and aggregate_ratio <= 1.0
    return {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "maximum_case_slowdown": limit,
        "cases": results,
        "aggregate": {
            "primary_metric": "us/atom",
            "frozen_sum_median_us_per_atom": frozen_total,
            "generated_sum_median_us_per_atom": generated_total,
            "generated_over_frozen": aggregate_ratio,
            "status": "pass" if aggregate_ratio <= 1.0 else "fail",
        },
    }


def validate_campaign(
    records: Iterable[Mapping[str, Any]],
    *,
    backends: Sequence[str] = BACKENDS,
) -> dict[str, Any]:
    """Require every matrix case to be measured as a pair or explicitly skipped."""

    required = set(required_cases(backends))
    observed: dict[BaselineCase, list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("schema") != SCHEMA:
            raise ValueError("R0 baseline record has the wrong schema")
        case = _case_from_record(record)
        if case not in required:
            raise ValueError(f"R0 baseline record is outside the campaign: {case.key}")
        observed.setdefault(case, []).append(record)
    missing = sorted(required - observed.keys())
    if missing:
        raise ValueError(
            f"R0 baseline campaign is missing required cases ({len(missing)} missing)"
        )

    measured_cases = 0
    skipped_cases = 0
    for case, case_records in observed.items():
        statuses = {record.get("status") for record in case_records}
        if statuses == {"skipped"} and len(case_records) == 1:
            _validate_skip(case, case_records[0].get("reason"))
            skipped_cases += 1
            continue
        if statuses != {"measured"} or len(case_records) != 2:
            raise ValueError(f"R0 baseline case has invalid evidence: {case.key}")
        implementations = {
            record.get("pipeline", {}).get("qualification_role")
            for record in case_records
            if isinstance(record.get("pipeline"), Mapping)
        }
        if implementations != set(IMPLEMENTATIONS):
            raise ValueError(
                f"R0 baseline case is missing its matched pair: {case.key}"
            )
        for record in case_records:
            pipeline = record.get("pipeline")
            required_pipeline_fields = {
                "stage",
                "qualification_role",
                "implementation",
                "backend",
                "artifact_origin",
                "schedule",
                "profile",
                "artifact_id",
                "qualification_route",
            }
            if (
                not isinstance(pipeline, Mapping)
                or set(pipeline) != required_pipeline_fields
                or pipeline.get("stage") != STAGE
            ):
                raise ValueError("R0 pipeline evidence is incomplete")
            role = pipeline.get("qualification_role")
            _validate_r0_pipeline(
                case,
                str(role),
                {
                    name: pipeline.get(name)
                    for name in (
                        "implementation",
                        "backend",
                        "artifact_origin",
                        "schedule",
                        "profile",
                        "artifact_id",
                        "qualification_route",
                    )
                },
            )
            provenance = record.get("provenance")
            artifact = (
                provenance.get("generated_r0_artifact")
                if isinstance(provenance, Mapping)
                else None
            )
            if role == "generated" and (
                not isinstance(artifact, Mapping)
                or pipeline["artifact_id"] != artifact.get("artifact_id")
            ):
                raise ValueError("R0 pipeline artifact ID does not match provenance")
        measured_cases += 1
    status = (
        "hardware-unavailable"
        if skipped_cases == len(required)
        else "measured"
        if measured_cases == len(required)
        else "complete-with-skips"
    )
    return {
        "schema": SCHEMA,
        "required_cases": len(required),
        "measured_cases": measured_cases,
        "skipped_cases": skipped_cases,
        "status": status,
    }


__all__ = [
    "BACKENDS",
    "BaselineCase",
    "EDGE_WEIGHT_MODES",
    "GRAPHS",
    "IMPLEMENTATIONS",
    "MAX_CASE_SLOWDOWN",
    "PATHS",
    "PRECISIONS",
    "SCHEMA",
    "evaluate_retirement_gate",
    "measurement_record",
    "measure_pipeline",
    "required_cases",
    "skipped_case",
    "validate_campaign",
]
