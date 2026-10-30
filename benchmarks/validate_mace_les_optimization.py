#!/usr/bin/env python3
"""Validate paired complete-MACELES optimization timing evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

SCHEMA = "symmetrix.mace-les-optimization/2"
TIMING_SCHEMA = "symmetrix.mace-les-timing/4"
MACELES_ROLES = ("baseline", "candidate")
STANDARD_ROLE = "standard_mace"
WORKLOAD_FIELDS = (
    "num_atoms",
    "precision",
    "backend",
    "pbc",
    "streamed_edges",
    "execution_profile",
    "replication",
    "cell_scale",
    "structure_sha256",
    "cutoff",
    "skin",
    "directed_edges",
)
MODEL_IDENTITY_FIELDS = (
    "model_type",
    "model_sha256",
    "mace_les_parameters_sha256",
    "mace_les_evaluator_fingerprint",
    "mace_architecture",
)
SOURCE_IDENTITY_FIELDS = (
    "source_commit",
    "source_dirty",
    "source_content_sha256",
    "native_source_content_sha256",
)
RUN_IDENTITY_FIELDS = (
    "native_extension_sha256",
    "native_build",
    "source_provenance",
    "interpreter_path",
    "python_version",
    "symmetrix_package_path",
    "symmetrix_version",
)
HOT_RECIPROCAL_CATEGORIES = ("structure_factors", "atom_outputs", "strain")
MACE_ARCHITECTURE_FIELDS = (
    "mace_num_interactions",
    "mace_num_channels",
    "mace_contract_fingerprints",
    "atomic_numbers",
    "cutoff",
)
MACE_CONTRACTS = ("R0", "M0", "R1")


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


def _digest(value, label):
    value = _text(value, label)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise QualificationError(f"{label} must be a lowercase SHA-256")
    return value


def _fingerprint(value, label):
    value = _text(value, label)
    if not value.startswith("sha256:"):
        raise QualificationError(f"{label} must be a SHA-256 fingerprint")
    _digest(value.removeprefix("sha256:"), label)
    return value


def _validate_mace_architecture(record, label):
    architecture = _mapping(
        record.get("mace_architecture"), f"{label}.mace_architecture"
    )
    if set(architecture) != set(MACE_ARCHITECTURE_FIELDS):
        raise QualificationError(
            f"{label}.mace_architecture must contain exactly "
            + ", ".join(MACE_ARCHITECTURE_FIELDS)
        )
    for field in ("mace_num_interactions", "mace_num_channels"):
        value = architecture[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise QualificationError(
                f"{label}.mace_architecture.{field} must be a positive integer"
            )
    fingerprints = _mapping(
        architecture["mace_contract_fingerprints"],
        f"{label}.mace_architecture.mace_contract_fingerprints",
    )
    if set(fingerprints) != set(MACE_CONTRACTS):
        raise QualificationError(
            f"{label}.mace_architecture.mace_contract_fingerprints must contain "
            "exactly R0, M0, R1"
        )
    for contract in MACE_CONTRACTS:
        _fingerprint(
            fingerprints[contract],
            f"{label}.mace_architecture.mace_contract_fingerprints.{contract}",
        )
    atomic_numbers = architecture["atomic_numbers"]
    if (
        not isinstance(atomic_numbers, list)
        or not atomic_numbers
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in atomic_numbers
        )
        or len(set(atomic_numbers)) != len(atomic_numbers)
    ):
        raise QualificationError(
            f"{label}.mace_architecture.atomic_numbers must be distinct positive integers"
        )
    cutoff = architecture["cutoff"]
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, (int, float))
        or not math.isfinite(cutoff)
        or cutoff <= 0
    ):
        raise QualificationError(
            f"{label}.mace_architecture.cutoff must be positive and finite"
        )
    if cutoff != record.get("cutoff"):
        raise QualificationError(
            f"{label}.mace_architecture.cutoff does not match the workload cutoff"
        )
    return architecture


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _samples(record, label):
    values = record.get("samples_ns")
    if not isinstance(values, list) or not values:
        raise QualificationError(f"{label}.samples_ns must be a non-empty array")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QualificationError(f"{label}.samples_ns contains a non-number")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise QualificationError(f"{label}.samples_ns must be positive and finite")
        result.append(value / 1.0e6)
    return result


def _revision(record, label):
    native = _mapping(record.get("native_build"), f"{label}.native_build")
    return _text(native.get("source_commit"), f"{label}.native_build.source_commit")


def _validate_source_identity(record, label):
    native = _mapping(record.get("native_build"), f"{label}.native_build")
    source = _mapping(record.get("source_provenance"), f"{label}.source_provenance")
    for field in SOURCE_IDENTITY_FIELDS:
        if field == "source_dirty":
            if not isinstance(source.get(field), bool) or not isinstance(
                native.get(field), bool
            ):
                raise QualificationError(f"{label}.{field} must be boolean")
        elif field.endswith("sha256"):
            _digest(source.get(field), f"{label}.source_provenance.{field}")
            _digest(native.get(field), f"{label}.native_build.{field}")
        else:
            _text(source.get(field), f"{label}.source_provenance.{field}")
            _text(native.get(field), f"{label}.native_build.{field}")
        if source[field] != native[field]:
            raise QualificationError(
                f"{label} source provenance does not match native build for {field}"
            )


def _validate_record(record, label, *, expected_model_type):
    if not isinstance(record, dict):
        raise QualificationError(f"{label} must be an object")
    if record.get("schema") != TIMING_SCHEMA:
        raise QualificationError(f"{label} has an unsupported timing schema")
    if record.get("model_type") != expected_model_type:
        raise QualificationError(f"{label}.model_type must be {expected_model_type}")
    _digest(record.get("model_sha256"), f"{label}.model_sha256")
    _digest(
        record.get("mace_les_evaluator_fingerprint"),
        f"{label}.mace_les_evaluator_fingerprint",
    )
    _text(record.get("model"), f"{label}.model")
    if expected_model_type == "MACELES":
        _digest(
            record.get("mace_les_parameters_sha256"),
            f"{label}.mace_les_parameters_sha256",
        )
    elif record.get("mace_les_parameters_sha256") is not None:
        raise QualificationError(
            f"{label}.mace_les_parameters_sha256 must be null for MACE"
        )
    _validate_mace_architecture(record, label)
    if record.get("fallback_evaluation_count") != 0:
        raise QualificationError(f"{label} used a fallback evaluation")
    if record.get("host_graph_constructions_during_timing") != 0:
        raise QualificationError(f"{label} rebuilt the host graph while timing")
    if record.get("prepared_graph_count") != 1:
        raise QualificationError(f"{label} must use exactly one prepared graph")
    if record.get("native_graph_rebuilds_during_timing") != 0:
        raise QualificationError(f"{label} rebuilt the native graph while timing")
    graph_before = record.get("native_graph_generation_before_timing")
    graph_after = record.get("native_graph_generation_after_timing")
    if (
        isinstance(graph_before, bool)
        or not isinstance(graph_before, int)
        or graph_before <= 0
        or graph_after != graph_before
    ):
        raise QualificationError(f"{label} native graph generations are inconsistent")
    if record.get("warmups", 0) < 20:
        raise QualificationError(f"{label} requires at least 20 warmups")
    if len(record.get("samples_ns", ())) < 30:
        raise QualificationError(f"{label} requires at least 30 synchronized samples")
    backend = record.get("backend")
    if backend not in {"cpu", "hip", "cuda"}:
        raise QualificationError(f"{label}.backend is unsupported")
    synchronization = _mapping(
        record.get("synchronization"), f"{label}.synchronization"
    )
    if backend in {"hip", "cuda"}:
        if record.get("device_synchronization_bracketed_samples") is not True:
            raise QualificationError(f"{label} GPU samples are not synchronized")
        if synchronization.get("backend") != backend:
            raise QualificationError(f"{label} synchronization backend does not match")
        _text(
            synchronization.get("runtime_library"),
            f"{label}.synchronization.runtime_library",
        )
    elif (
        record.get("device_synchronization_bracketed_samples") is not False
        or synchronization.get("backend") != "none"
    ):
        raise QualificationError(f"{label} CPU synchronization metadata is invalid")
    if record.get("streamed_edges") == "direct":
        if record.get("jit_status") not in {"built", "cached"}:
            raise QualificationError(f"{label} direct JIT status is invalid")
        _text(record.get("jit_artifact_path"), f"{label}.jit_artifact_path")
    _digest(record.get("native_extension_sha256"), f"{label}.native_extension_sha256")
    _validate_source_identity(record, label)
    for field in (
        "interpreter_path",
        "python_version",
        "symmetrix_package_path",
        "symmetrix_version",
    ):
        _text(record.get(field), f"{label}.{field}")
    device = _mapping(record.get("device_provenance"), f"{label}.device_provenance")
    if not device.get("architecture") or not device.get("device_name"):
        raise QualificationError(f"{label}.device_provenance is incomplete")
    return statistics.median(_samples(record, label))


def _match(records, roles, fields, label):
    anchor = records[roles[0]]
    for role in roles[1:]:
        mismatches = [
            field for field in fields if records[role].get(field) != anchor.get(field)
        ]
        if mismatches:
            raise QualificationError(
                f"{label} workload/provenance mismatch for {role}: "
                + ", ".join(mismatches)
            )


def _profile_result(document, backend):
    profiles = _mapping(document.get("profiles"), "profiles")
    durations = {}
    for role in MACELES_ROLES:
        profile = _mapping(profiles.get(role), f"profiles.{role}")
        if backend == "hip":
            schema_valid = (
                profile.get("schema") == "symmetrix.rocprof-kernel-summary"
                and profile.get("version") == 2
            )
            expected = "a rocprof LES-stage summary with version 2"
        else:
            schema_valid = (
                profile.get("schema") == "symmetrix.nsys-les-kernel-summary/2"
            )
            expected = "an Nsight Systems LES-stage summary"
        if not schema_valid:
            raise QualificationError(f"profiles.{role} must be {expected}")
        count = profile.get("selected_evaluation_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 10:
            raise QualificationError(f"profiles.{role} requires ten evaluations")
        categories = profile.get("les_stage_categories")
        if not isinstance(categories, list):
            raise QualificationError(
                f"profiles.{role}.les_stage_categories must be an array"
            )
        by_name = {
            item.get("category"): item
            for item in categories
            if isinstance(item, dict) and isinstance(item.get("category"), str)
        }
        durations[role] = {}
        for category in HOT_RECIPROCAL_CATEGORIES:
            value = by_name.get(category, {}).get("duration_ns_per_evaluation")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise QualificationError(
                    f"profiles.{role} lacks a positive {category} duration"
                )
            durations[role][category] = float(value)
    family_speedups = {
        category: durations["baseline"][category] / durations["candidate"][category]
        for category in HOT_RECIPROCAL_CATEGORIES
    }
    speedup = sum(durations["baseline"].values()) / sum(durations["candidate"].values())
    if backend == "hip":
        for category, value in family_speedups.items():
            if value < 2.0:
                raise QualificationError(
                    f"HIP LES-stage profile {category} did not achieve 2x speedup"
                )
    elif speedup <= 1.0:
        raise QualificationError("CUDA LES-stage profile does not improve")
    return speedup, family_speedups


def _bootstrap_ratio_upper(pairs, seed, draws):
    generator = random.Random(seed)
    ratios = []
    count = len(pairs)
    for _ in range(draws):
        sample = [pairs[generator.randrange(count)] for _ in range(count)]
        baseline = statistics.median(item[0] for item in sample)
        candidate = statistics.median(item[1] for item in sample)
        ratios.append(candidate / baseline)
    return _percentile(ratios, 0.95)


def _binary_identity(record):
    source = record["source_provenance"]
    return (
        source["source_content_sha256"],
        source["native_source_content_sha256"],
        record["native_extension_sha256"],
    )


def validate_document(document):
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise QualificationError(f"document schema must be {SCHEMA}")
    seed = document.get("bootstrap_seed")
    draws = document.get("bootstrap_draws")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise QualificationError("bootstrap_seed must be an integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 100:
        raise QualificationError("bootstrap_draws must be an integer of at least 100")
    blocks = document.get("blocks")
    if not isinstance(blocks, list) or len(blocks) < 5:
        raise QualificationError("at least five complete timing blocks are required")

    first_records = None
    block_results = []
    first_block = _mapping(blocks[0], "blocks[0]")
    first_block_records = _mapping(first_block.get("records"), "blocks[0].records")
    includes_standard = STANDARD_ROLE in first_block_records
    expected_roles = set(MACELES_ROLES)
    if includes_standard:
        expected_roles.add(STANDARD_ROLE)
    for index, block in enumerate(blocks):
        label = f"blocks[{index}]"
        if not isinstance(block, dict) or not block.get("block_id"):
            raise QualificationError(f"{label} requires a block_id")
        records = block.get("records")
        if not isinstance(records, dict) or set(records) != expected_roles:
            raise QualificationError(
                f"{label} must contain exactly the complete-model timing roles"
            )
        medians = {
            role: _validate_record(
                records[role], f"{label}.{role}", expected_model_type="MACELES"
            )
            for role in MACELES_ROLES
        }
        _match(
            records,
            MACELES_ROLES,
            (*WORKLOAD_FIELDS, *MODEL_IDENTITY_FIELDS, "device_provenance"),
            label,
        )
        if includes_standard:
            medians[STANDARD_ROLE] = _validate_record(
                records[STANDARD_ROLE],
                f"{label}.{STANDARD_ROLE}",
                expected_model_type="MACE",
            )
            _match(
                records,
                ("candidate", STANDARD_ROLE),
                (*WORKLOAD_FIELDS, "device_provenance", *RUN_IDENTITY_FIELDS),
                label,
            )
            standard = records[STANDARD_ROLE]
            candidate = records["candidate"]
            if (
                standard["model"] == candidate["model"]
                or standard["model_sha256"] == candidate["model_sha256"]
            ):
                raise QualificationError(
                    f"{label} standard MACE must use a separate model record/path"
                )
            if standard["mace_architecture"] != candidate["mace_architecture"]:
                raise QualificationError(
                    f"{label} standard MACE architecture does not match MACELES"
                )
        if first_records is None:
            first_records = records
        else:
            for role in expected_roles:
                for field in (
                    *WORKLOAD_FIELDS,
                    *MODEL_IDENTITY_FIELDS,
                    "device_provenance",
                    *RUN_IDENTITY_FIELDS,
                ):
                    if records[role].get(field) != first_records[role].get(field):
                        raise QualificationError(
                            f"{label} differs from the first block in {role}.{field}"
                        )
        ratio = medians["candidate"] / medians["baseline"]
        result = {
            "block_id": block["block_id"],
            "medians_ms": medians,
            "candidate_to_baseline_ratio": ratio,
            "speedup": 1.0 / ratio,
        }
        if includes_standard:
            result["maceles_over_standard_mace"] = (
                medians["candidate"] / medians[STANDARD_ROLE]
            )
        block_results.append(result)

    if _binary_identity(first_records["baseline"]) == _binary_identity(
        first_records["candidate"]
    ):
        raise QualificationError(
            "baseline and candidate must have distinct source/native or extension identity"
        )
    pairs = [
        (item["medians_ms"]["baseline"], item["medians_ms"]["candidate"])
        for item in block_results
    ]
    ratio_upper = _bootstrap_ratio_upper(pairs, seed, draws)
    if ratio_upper > 1.05:
        raise QualificationError(
            "paired complete-MACELES candidate/baseline upper bound exceeds 1.05"
        )
    backend = first_records["baseline"]["backend"]
    profile_speedup = None
    profile_family_speedups = None
    if backend in {"hip", "cuda"}:
        profile_speedup, profile_family_speedups = _profile_result(document, backend)
    candidate_median = statistics.median(item[1] for item in pairs)
    result = {
        "schema": "symmetrix.mace-les-optimization-validation/2",
        "passed": True,
        "backend": backend,
        "block_count": len(block_results),
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "paired_ratio_upper_95": ratio_upper,
        "median_baseline_ms": statistics.median(item[0] for item in pairs),
        "median_candidate_ms": candidate_median,
        "median_candidate_to_baseline_ratio": statistics.median(
            item["candidate_to_baseline_ratio"] for item in block_results
        ),
        "median_speedup": statistics.median(item["speedup"] for item in block_results),
        "les_stage_profile_speedup": profile_speedup,
        "les_stage_profile_family_speedups": profile_family_speedups,
        "blocks": block_results,
    }
    if includes_standard:
        standard_median = statistics.median(
            item["medians_ms"][STANDARD_ROLE] for item in block_results
        )
        result.update(
            {
                "median_standard_mace_ms": standard_median,
                "median_maceles_over_standard_mace": (
                    candidate_median / standard_median
                ),
            }
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_document(json.loads(args.document.read_text()))
    except (OSError, json.JSONDecodeError, QualificationError) as error:
        parser.error(str(error))
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.json_output == Path("-"):
        sys.stdout.write(rendered)
    elif args.json_output is not None:
        args.json_output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
