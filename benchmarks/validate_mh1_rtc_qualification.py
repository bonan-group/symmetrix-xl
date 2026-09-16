#!/usr/bin/env python3
"""Validate matched fresh-process MACE-MH-1/MH-0 RTC evidence."""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

SCHEMA = "symmetrix.direct.mh1-rtc-qualification"
SCHEMA_VERSION = 1
MAX_MH1_TO_MH0_RATIO = 2.5
MIN_PROCESSES = 3
MIN_WARMUPS = 20
MIN_SAMPLES = 20
BOOTSTRAP_DRAWS = 20_000


class QualificationError(ValueError):
    pass


def _mapping(value, label):
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be an object")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{label} must be non-empty")
    return value


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise QualificationError(f"{label} must be {qualifier} and finite")
    return result


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_contract(document):
    contract = _mapping(document.get("contract"), "contract")
    required = {
        "structure_id",
        "head",
        "atoms",
        "directed_edges",
        "dtype",
        "timing_scope",
        "mh0_model_sha256",
        "mh1_model_sha256",
    }
    if set(contract) != required:
        raise QualificationError("contract fields are invalid")
    if _text(contract["structure_id"], "contract.structure_id") != "wurtzite-AlN-6x6x6":
        raise QualificationError("contract structure must be 864-atom wurtzite AlN")
    if _text(contract["head"], "contract.head") != "omat_pbe":
        raise QualificationError("contract head must be omat_pbe")
    if contract["atoms"] != 864:
        raise QualificationError("contract atom count must be 864")
    if (
        isinstance(contract["directed_edges"], bool)
        or not isinstance(contract["directed_edges"], int)
        or contract["directed_edges"] <= 0
    ):
        raise QualificationError("contract.directed_edges must be positive")
    if contract["dtype"] != "float32":
        raise QualificationError("contract dtype must be float32")
    _text(contract["timing_scope"], "contract.timing_scope")
    for role in ("mh0", "mh1"):
        digest = _text(
            contract[f"{role}_model_sha256"], f"contract.{role}_model_sha256"
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise QualificationError(
                f"contract.{role}_model_sha256 must be lowercase SHA256"
            )
    if contract["mh0_model_sha256"] == contract["mh1_model_sha256"]:
        raise QualificationError("MH0 and MH1 checkpoints must differ")
    return contract


def _validate_runs(record, role):
    runs = record.get("process_runs")
    if not isinstance(runs, list) or len(runs) < MIN_PROCESSES:
        raise QualificationError(
            f"{role} requires at least {MIN_PROCESSES} fresh processes"
        )
    process_ids = set()
    validated = []
    for index, run in enumerate(runs):
        label = f"{role}.process_runs[{index}]"
        run = _mapping(run, label)
        process_id = _text(run.get("process_id"), f"{label}.process_id")
        if process_id in process_ids:
            raise QualificationError(f"{role} process IDs must be unique")
        process_ids.add(process_id)
        if run.get("fresh_process") is not True:
            raise QualificationError(f"{label} must declare fresh_process=true")
        warmups = run.get("warmup_count")
        if (
            isinstance(warmups, bool)
            or not isinstance(warmups, int)
            or warmups < MIN_WARMUPS
        ):
            raise QualificationError(f"{label} requires at least {MIN_WARMUPS} warmups")
        samples = run.get("samples_ms")
        if not isinstance(samples, list) or len(samples) < MIN_SAMPLES:
            raise QualificationError(f"{label} requires at least {MIN_SAMPLES} samples")
        samples = [
            _number(value, f"{label}.samples_ms", positive=True) for value in samples
        ]
        validated.append(
            {
                "process_id": process_id,
                "samples": samples,
                "median": statistics.median(samples),
                "cold_compile_load_ms": _number(
                    run.get("cold_compile_load_ms"),
                    f"{label}.cold_compile_load_ms",
                    positive=True,
                ),
            }
        )
    return validated


def _validate_record(record, role, backend, contract):
    record = _mapping(record, role)
    provenance = _mapping(record.get("provenance"), f"{role}.provenance")
    required_provenance = (
        "device_ordinal",
        "device_name",
        "architecture",
        "raw_agent_target",
        "native_subgroup_width",
        "runtime_version",
        "driver_version",
        "commit",
        "model_sha256",
    )
    for field in required_provenance:
        if field in ("device_ordinal", "native_subgroup_width"):
            value = provenance.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise QualificationError(
                    f"{role}.provenance.{field} must be non-negative"
                )
        else:
            _text(provenance.get(field), f"{role}.provenance.{field}")
    if provenance["model_sha256"] != contract[f"{role}_model_sha256"]:
        raise QualificationError(f"{role} checkpoint hash does not match the contract")

    correctness = _mapping(record.get("correctness"), f"{role}.correctness")
    if correctness.get("passed") is not True:
        raise QualificationError(f"{role} correctness did not pass")
    if correctness.get("nonfinite_count") != 0:
        raise QualificationError(f"{role} correctness found non-finite values")
    _mapping(
        correctness.get("validation_contract"),
        f"{role}.correctness.validation_contract",
    )

    path = _mapping(record.get("path"), f"{role}.path")
    expected_compiler = "nvrtc" if backend == "cuda" else "hiprtc"
    if path.get("active") is not True:
        raise QualificationError(f"{role} generated path is not active")
    if path.get("backend") != backend:
        raise QualificationError(f"{role} path backend does not match {backend}")
    if path.get("compiler") != expected_compiler:
        raise QualificationError(f"{role} compiler must be {expected_compiler}")
    if path.get("fallback") is not False:
        raise QualificationError(f"{role} path must declare fallback=false")
    for field in ("artifact_id", "execution_backend", "schedule_id"):
        _text(path.get(field), f"{role}.path.{field}")

    graph = _mapping(record.get("graph"), f"{role}.graph")
    if graph.get("atoms") != contract["atoms"]:
        raise QualificationError(f"{role} graph atom count changed")
    if graph.get("directed_edges") != contract["directed_edges"]:
        raise QualificationError(f"{role} graph edge count changed")
    if graph.get("prepared") is not True:
        raise QualificationError(f"{role} graph must be prepared")

    resources = _mapping(record.get("resources"), f"{role}.resources")
    for field in (
        "workspace_bytes",
        "peak_device_memory_bytes",
        "artifact_size_bytes",
        "launches_per_evaluation",
        "local_memory_bytes",
        "spill_loads",
        "spill_stores",
        "stack_bytes",
    ):
        _number(
            resources.get(field),
            f"{role}.resources.{field}",
            positive=field
            in (
                "peak_device_memory_bytes",
                "artifact_size_bytes",
                "launches_per_evaluation",
            ),
        )
    return provenance, _validate_runs(record, role)


def _bootstrap_ratio_upper(mh0_runs, mh1_runs):
    mh0_medians = [run["median"] for run in mh0_runs]
    mh1_medians = [run["median"] for run in mh1_runs]
    generator = random.Random(0)
    ratios = []
    for _ in range(BOOTSTRAP_DRAWS):
        mh0 = statistics.median(generator.choice(mh0_medians) for _ in mh0_medians)
        mh1 = statistics.median(generator.choice(mh1_medians) for _ in mh1_medians)
        ratios.append(mh1 / mh0)
    return _percentile(ratios, 0.95)


def validate_document(document):
    document = _mapping(document, "document")
    if document.get("schema") != SCHEMA or document.get("version") != SCHEMA_VERSION:
        raise QualificationError("unsupported qualification schema")
    backend = document.get("backend")
    if backend not in ("cuda", "hip"):
        raise QualificationError("backend must be cuda or hip")
    contract = _validate_contract(document)
    mh0_provenance, mh0_runs = _validate_record(
        document.get("mh0"), "mh0", backend, contract
    )
    mh1_provenance, mh1_runs = _validate_record(
        document.get("mh1"), "mh1", backend, contract
    )
    shared_provenance = (
        "device_ordinal",
        "device_name",
        "architecture",
        "raw_agent_target",
        "native_subgroup_width",
        "runtime_version",
        "driver_version",
        "commit",
    )
    for field in shared_provenance:
        if mh0_provenance[field] != mh1_provenance[field]:
            raise QualificationError(f"provenance mismatch for {field}")
    if len(mh0_runs) != len(mh1_runs):
        raise QualificationError("MH0 and MH1 process counts differ")
    if document.get("process_order_policy") not in ("randomized", "balanced"):
        raise QualificationError("process_order_policy must be randomized or balanced")
    execution_order = document.get("execution_order")
    expected = {
        *(f"mh0:{run['process_id']}" for run in mh0_runs),
        *(f"mh1:{run['process_id']}" for run in mh1_runs),
    }
    if (
        not isinstance(execution_order, list)
        or len(execution_order) != len(expected)
        or set(execution_order) != expected
    ):
        raise QualificationError(
            "execution_order must list every fresh process exactly once"
        )

    mh0_median = statistics.median(run["median"] for run in mh0_runs)
    mh1_median = statistics.median(run["median"] for run in mh1_runs)
    ratio = mh1_median / mh0_median
    ratio_upper = _bootstrap_ratio_upper(mh0_runs, mh1_runs)
    if ratio_upper > MAX_MH1_TO_MH0_RATIO:
        raise QualificationError(
            f"MH1/MH0 upper 95% ratio {ratio_upper:.6f} exceeds {MAX_MH1_TO_MH0_RATIO:.1f}"
        )
    return {
        "backend": backend,
        "architecture": mh0_provenance["architecture"],
        "processes_per_model": len(mh0_runs),
        "mh0_median_ms": mh0_median,
        "mh1_median_ms": mh1_median,
        "mh1_to_mh0_ratio": ratio,
        "mh1_to_mh0_ratio_upper_95": ratio_upper,
        "limit": MAX_MH1_TO_MH0_RATIO,
        "mh0_p90_ms": _percentile(
            [sample for run in mh0_runs for sample in run["samples"]], 0.90
        ),
        "mh1_p90_ms": _percentile(
            [sample for run in mh1_runs for sample in run["samples"]], 0.90
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_document(json.loads(args.input.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, QualificationError) as error:
        parser.exit(1, f"qualification failed: {error}\n")
    result = {"status": "passed", **summary}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    main()
