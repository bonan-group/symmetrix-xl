"""Compose a strict OAM qualification cell from paired driver records."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import stat
import statistics
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.oam_release_qualification import (
    CELL_SCHEMA,
    CORRECTNESS_TOLERANCES,
    REQUIRED_BINDING_FIELDS,
    SCHEMA_VERSION,
    TIMED_REGION,
    TIMING_EXCLUSIONS,
    QualificationError,
    _json_values_equal,
    _normalized_cuda_target,
    build_driver_command,
    build_driver_environment,
)

DRIVER_SCHEMA = "symmetrix.oam-qualified-driver-record"
DRIVER_VERSION = 3
MAX_NUMERICAL_EVIDENCE_BYTES = 64 * 1024 * 1024
RAW_BASE_FIELDS = {
    "schema",
    "version",
    "report_version",
    "route",
    "execution_profile",
    "qualification_cell",
    "backend",
    "precision",
    "structure",
    "implementation",
    "model",
    "model_sha256",
    "provenance",
    "command",
    "python",
    "package",
    "native_extension",
    "native_sha256",
    "execution_environment",
    "cpu_affinity",
    "thread_environment",
    "observed_runtime",
    "atoms",
    "directed_edges",
    "model_cutoff",
    "skin",
    "effective_cutoff",
    "graph_sha256",
    "graph",
    "repeat",
    "rattle",
    "seed",
    "warmup",
    "sample_count",
    "profile_capture",
    "generation_policy",
    "generation_policy_selection",
    "median_us_per_atom_step",
    "median_us_per_atom",
    "min_us_per_atom_step",
    "samples_us_per_atom_step",
    "energy",
    "max_force",
    "numerical_evidence",
    "artifact_id",
    "artifact_path",
    "artifact_cache_status",
    "artifact",
    "execution",
    "memory",
    "reuse",
    "capacity_boundary",
    "cpu",
    "timing_contract",
    "execution_diagnostic",
    "workspace_bytes",
    "topology_bytes",
    "linear_workspace_bytes",
    "model_preparation_seconds",
}
RAW_OPTIONAL_FIELDS = {
    "conditioned_self_bytes",
    "conditioned_self_layer_count",
    "conditioned_self_original_tensor_bytes",
    "conditioned_self_schedule_bytes",
    "conditioned_self_forward_launch_count",
    "conditioned_self_reverse_launch_count",
    "compiler",
    "threads_per_block",
    "forward_threads_per_block",
    "cuda_launch_count",
    "cuda_fallback_count",
}

HOST_EXECUTION_ENVIRONMENT_FIELDS = {
    "available",
    "backend",
    "execution_space",
    "memory_space",
    "target",
}
CUDA_EXECUTION_ENVIRONMENT_FIELDS = {
    "available",
    "backend",
    "execution_space",
    "memory_space",
    "architecture",
    "device_ordinal",
    "compute_capability_code",
}
HOST_GENERATION_POLICY_FIELDS = {
    "owner_schedule",
    "channel_tile",
    "scalar_width",
    "vector_policy",
    "required_isa",
    "full_reverse_reduction_policy",
    "resource_policy",
}
CUDA_GENERATION_POLICY_FIELDS = {
    "forward_live_output_budget",
    "forward_grouping_mode",
    "forward_group_execution",
    "forward_input_mode",
    "radial_adjoint_mode",
    "path_reuse_mode",
    "reverse_group_execution",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{name} must be an object")
    return value


def _field(record: Mapping[str, Any], name: str, route: str) -> Any:
    if name not in record or record[name] is None or record[name] == "":
        raise QualificationError(f"{route} record is missing {name}")
    return record[name]


def _digest(value: Any, name: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None
    ):
        raise QualificationError(
            f"{name} must be a hexadecimal {length}-character digest"
        )
    return value


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{name} must be a numeric value")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise QualificationError(
            f"{name} must be finite" + (" and positive" if positive else "")
        )
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    allowed: tuple[int, ...] | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise QualificationError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise QualificationError(f"{name} must be at most {maximum}")
    if allowed is not None and value not in allowed:
        raise QualificationError(f"{name} must be one of {allowed}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{name} must be a non-empty string")
    return value


def _canonical_timing(record: Mapping[str, Any], route: str) -> dict[str, Any]:
    samples_value = record.get("samples_us_per_atom_step")
    if not isinstance(samples_value, list) or not samples_value:
        raise QualificationError(
            f"{route}.samples_us_per_atom_step must be a non-empty array"
        )
    samples = [
        _finite_number(
            value, f"{route}.samples_us_per_atom_step[{index}]", positive=True
        )
        for index, value in enumerate(samples_value)
    ]
    minimum = min(samples)
    median = statistics.median(samples)
    for name, expected in (
        ("min_us_per_atom_step", minimum),
        ("median_us_per_atom_step", median),
        ("median_us_per_atom", median),
    ):
        if (
            _finite_number(record.get(name), f"{route}.{name}", positive=True)
            != expected
        ):
            raise QualificationError(f"{route}.{name} does not match canonical samples")
    contract = _mapping(record.get("timing_contract"), f"{route}.timing_contract")
    return {
        "unit": contract.get("unit"),
        "timed_region": contract.get("timed_region"),
        "excluded": list(contract.get("excluded", [])),
        "samples_us_per_atom_step": samples,
        "min_us_per_atom_step": minimum,
        "median_us_per_atom_step": median,
        "median_us_per_atom": median,
    }


def _canonical_cell_timing(
    direct: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    direct_timing = _canonical_timing(direct, "direct")
    reference_timing = _canonical_timing(reference, "reference")
    return {
        "unit": direct_timing["unit"],
        "direct_us_per_atom": direct_timing["median_us_per_atom"],
        "reference_us_per_atom": reference_timing["median_us_per_atom"],
        "timed_region": direct_timing["timed_region"],
        "excluded": list(direct_timing["excluded"]),
        "direct": direct_timing,
        "reference": reference_timing,
    }


def _expected_generation_policy(
    record: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any] | None:
    if record.get("route") != "direct":
        return None
    policy = request.get("generation_policy_request")
    if not isinstance(policy, Mapping):
        raise QualificationError("request.generation_policy_request must be an object")
    return dict(policy)


def _route_independent_command(command: list[str]) -> list[str]:
    normalized = list(command)
    normalized[normalized.index("--routing") + 1] = "<route>"
    normalized[normalized.index("--output") + 1] = "<output>"
    normalized[normalized.index("--qualification-arrays") + 1] = "<numerics>"
    return normalized


def _load_numerical_evidence(
    request: Mapping[str, Any], record: Mapping[str, Any], route: str
) -> dict[str, Any]:
    evidence = _mapping(record.get("numerical_evidence"), f"{route}.numerical_evidence")
    if set(evidence) != {"format", "path", "sha256"}:
        raise QualificationError(f"{route}.numerical_evidence fields mismatch")
    if evidence.get("format") != "npz-v1":
        raise QualificationError(f"{route}.numerical_evidence format is unsupported")
    expected_path = Path(str(request[f"{route}_numerical_path"]))
    declared_path = Path(str(evidence.get("path")))
    if not expected_path.is_absolute() or declared_path != expected_path:
        raise QualificationError(f"{route}.numerical_evidence path mismatch")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(declared_path, flags)
    except OSError as error:
        raise QualificationError(
            f"{route}.numerical_evidence is unavailable: {error}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > MAX_NUMERICAL_EVIDENCE_BYTES
        ):
            raise QualificationError(
                f"{route}.numerical_evidence must be a bounded regular file"
            )
        chunks = []
        remaining = MAX_NUMERICAL_EVIDENCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if not payload or len(payload) > MAX_NUMERICAL_EVIDENCE_BYTES:
            raise QualificationError(f"{route}.numerical_evidence file size is invalid")
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(payload).hexdigest()
    _digest(evidence.get("sha256"), f"{route}.numerical_evidence.sha256")
    if evidence["sha256"] != digest:
        raise QualificationError(f"{route}.numerical_evidence hash mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if (
                sum(member.file_size for member in archive.infolist())
                > MAX_NUMERICAL_EVIDENCE_BYTES
            ):
                raise QualificationError(
                    f"{route}.numerical_evidence expanded size is invalid"
                )
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            if set(arrays.files) != {"energy", "forces", "stress"}:
                raise QualificationError(f"{route}.numerical_evidence arrays mismatch")
            values = {name: np.asarray(arrays[name]) for name in arrays.files}
    except QualificationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise QualificationError(
            f"{route}.numerical_evidence NPZ is invalid: {error}"
        ) from error
    atoms = int(record["atoms"])
    expected_shapes = {"energy": (), "forces": (atoms, 3), "stress": (3, 3)}
    for name, value in values.items():
        if value.shape != expected_shapes[name] or value.dtype.kind != "f":
            raise QualificationError(
                f"{route}.numerical_evidence {name} shape/type mismatch"
            )
        if not np.isfinite(value).all():
            raise QualificationError(
                f"{route}.numerical_evidence {name} must be finite"
            )
    return values


def _correctness_report(
    request: Mapping[str, Any],
    direct: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    direct_values = _load_numerical_evidence(request, direct, "direct")
    reference_values = _load_numerical_evidence(request, reference, "reference")
    atoms = int(direct["atoms"])
    errors = {
        "energy_abs_eV_per_atom": abs(
            float(direct_values["energy"]) - float(reference_values["energy"])
        )
        / atoms,
        "forces_max_abs_eV_per_A": float(
            np.max(np.abs(direct_values["forces"] - reference_values["forces"]))
        ),
        "stress_max_abs_eV_per_A3": float(
            np.max(np.abs(direct_values["stress"] - reference_values["stress"]))
        ),
    }
    tolerances = dict(CORRECTNESS_TOLERANCES[str(request["precision"])])
    for name, error in errors.items():
        if error > tolerances[name]:
            label = name.split("_", 1)[0]
            raise QualificationError(
                f"direct/reference {label} parity exceeds {tolerances[name]}: {error}"
            )
    return {
        "tolerances": tolerances,
        "errors": errors,
        "direct_sha256": direct["numerical_evidence"]["sha256"],
        "reference_sha256": reference["numerical_evidence"]["sha256"],
        "passed": True,
    }


def validate_raw_driver_record(
    request: Mapping[str, Any], record: Mapping[str, Any], route: str
) -> None:
    """Validate one raw driver record without leaking schema exceptions."""

    try:
        if not isinstance(record, Mapping):
            raise QualificationError(f"{route} record must be an object")
        missing_fields = sorted(RAW_BASE_FIELDS - set(record))
        unknown_fields = sorted(set(record) - RAW_BASE_FIELDS - RAW_OPTIONAL_FIELDS)
        if missing_fields or unknown_fields:
            raise QualificationError(
                f"{route} raw fields invalid: missing={missing_fields}, "
                f"unknown={unknown_fields}"
            )
        if (
            record.get("schema") != DRIVER_SCHEMA
            or not isinstance(record.get("version"), int)
            or isinstance(record.get("version"), bool)
            or record.get("version") != DRIVER_VERSION
        ):
            raise QualificationError(f"unsupported {route} driver schema/version")
        if (
            not isinstance(record.get("report_version"), int)
            or isinstance(record.get("report_version"), bool)
            or record.get("report_version") != DRIVER_VERSION
        ):
            raise QualificationError(f"unsupported {route} report_version")
        if record.get("route") != route:
            raise QualificationError(f"{route} record route mismatch")
        if record.get("qualification_cell") != dict(request):
            raise QualificationError(f"{route} qualification_cell mismatch")
        expected_backend = "cpu" if request["backend"] == "host" else "cuda"
        if record.get("backend") != expected_backend:
            raise QualificationError(f"{route} backend mismatch")
        if record.get("precision") != request["precision"]:
            raise QualificationError(f"{route} precision mismatch")
        if record.get("execution_profile") != request["profile"]:
            raise QualificationError(f"{route} execution profile mismatch")

        implementation = _field(record, "implementation", route)
        expected_implementation = (
            "receiver-owned-generated"
            if route == "direct"
            else "portable-oam-kokkos-reference"
        )
        if implementation != expected_implementation:
            raise QualificationError(f"{route} implementation is not truthful")

        _digest(record.get("model_sha256"), f"{route}.model_sha256")
        _digest(record.get("graph_sha256"), f"{route}.graph_sha256")
        provenance = _mapping(record.get("provenance"), f"{route}.provenance")
        if set(provenance) != {
            "source_fingerprint",
            "source_commit",
            "source_dirty",
            "native_sha256",
            "native_fast_math",
            "jit_cache",
            "jit_cache_owner_uid",
            "jit_cache_mode",
        }:
            raise QualificationError(f"{route}.provenance fields mismatch")
        _digest(
            provenance.get("source_fingerprint"),
            f"{route}.provenance.source_fingerprint",
        )
        _digest(
            provenance.get("source_commit"),
            f"{route}.provenance.source_commit",
            40,
        )
        _digest(provenance.get("native_sha256"), f"{route}.provenance.native_sha256")
        if not isinstance(provenance.get("source_dirty"), bool):
            raise QualificationError(f"{route}.provenance.source_dirty must be boolean")
        jit_cache = provenance.get("jit_cache")
        if (
            not isinstance(jit_cache, str)
            or request["request_fingerprint"] not in jit_cache
        ):
            raise QualificationError(
                f"{route}.provenance.jit_cache is not request-bound"
            )
        for name in (
            "source_fingerprint",
            "source_commit",
            "source_dirty",
            "native_sha256",
        ):
            if provenance.get(name) != request.get(name):
                raise QualificationError(
                    f"{route}.provenance.{name} does not match the request"
                )
        native_fast_math = provenance.get("native_fast_math")
        if request["backend"] == "host":
            if not isinstance(native_fast_math, bool):
                raise QualificationError(
                    f"{route}.provenance.native_fast_math must be boolean"
                )
            expected_fast = request["host_floating_point_policy"] == "fast"
            if native_fast_math is not expected_fast:
                raise QualificationError(
                    f"{route}.provenance.native_fast_math policy mismatch"
                )
        elif native_fast_math is not None:
            raise QualificationError(
                f"{route}.provenance.native_fast_math must be null for CUDA"
            )
        if record["model_sha256"] != request["model_sha256"]:
            raise QualificationError(f"{route}.model_sha256 does not match the request")

        command = record.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) for item in command)
        ):
            raise QualificationError(
                f"{route}.command must be a non-empty string array"
            )
        if command.count("--output") != 1:
            raise QualificationError(f"{route}.command must contain one output path")
        expected_command = build_driver_command({}, request, route)
        if command != expected_command:
            raise QualificationError(f"{route}.command is not the canonical invocation")
        command_cell = json.loads(command[command.index("--qualification-cell") + 1])
        if command_cell != dict(request):
            raise QualificationError(f"{route}.command qualification cell mismatch")

        _, expected_cache = build_driver_environment(request)
        if provenance.get("jit_cache") != str(expected_cache):
            raise QualificationError(f"{route}.provenance.jit_cache path mismatch")
        try:
            cache_stat = expected_cache.stat()
        except OSError as error:
            raise QualificationError(
                f"{route}.provenance.jit_cache is unavailable: {error}"
            ) from error
        if cache_stat.st_uid != os.getuid() or cache_stat.st_mode & 0o077:
            raise QualificationError(
                f"{route}.provenance.jit_cache ownership/mode is unsafe"
            )
        if provenance.get("jit_cache_owner_uid") != cache_stat.st_uid:
            raise QualificationError(f"{route}.provenance.jit_cache owner mismatch")
        if provenance.get("jit_cache_mode") != (cache_stat.st_mode & 0o777):
            raise QualificationError(f"{route}.provenance.jit_cache mode mismatch")

        graph = _mapping(record.get("graph"), f"{route}.graph")
        if set(graph) != {
            "atom_count",
            "cutoff",
            "skin",
            "effective_cutoff",
            "directed_edges",
        }:
            raise QualificationError(f"{route}.graph fields mismatch")
        for name in ("atom_count", "directed_edges"):
            value = graph.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise QualificationError(
                    f"{route}.graph.{name} must be a positive integer"
                )
        for name in ("cutoff", "skin", "effective_cutoff"):
            _finite_number(graph.get(name), f"{route}.graph.{name}")
        if record.get("model") != request["model_path"]:
            raise QualificationError(f"{route}.model does not match the request")
        if record.get("python") != request["python_executable"]:
            raise QualificationError(f"{route}.python does not match the command")
        for name in ("package", "native_extension"):
            value = record.get(name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise QualificationError(f"{route}.{name} must be an absolute path")
        if record.get("native_extension") != request["native_path"]:
            raise QualificationError(
                f"{route}.native_extension does not match the request"
            )
        _digest(record.get("native_sha256"), f"{route}.native_sha256")
        if record["native_sha256"] != request["native_sha256"]:
            raise QualificationError(f"{route}.native_sha256 mismatch")
        if record.get("package") != request["package_path"]:
            raise QualificationError(f"{route}.package does not match the request")
        for name in ("atoms", "directed_edges"):
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise QualificationError(f"{route}.{name} must be a positive integer")
        for name in (
            "model_cutoff",
            "effective_cutoff",
            "energy",
            "model_preparation_seconds",
        ):
            _finite_number(record.get(name), f"{route}.{name}")
        max_force = _finite_number(record.get("max_force"), f"{route}.max_force")
        preparation_seconds = _finite_number(
            record.get("model_preparation_seconds"),
            f"{route}.model_preparation_seconds",
        )
        if max_force < 0.0 or preparation_seconds < 0.0:
            raise QualificationError(f"{route} diagnostics must be nonnegative")
        samples = record.get("samples_us_per_atom_step")
        if not isinstance(samples, list) or len(samples) != request["samples"]:
            raise QualificationError(f"{route}.samples_us_per_atom_step is malformed")
        canonical_timing = _canonical_timing(record, route)
        for name in ("workspace_bytes", "topology_bytes", "linear_workspace_bytes"):
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise QualificationError(f"{route}.{name} is invalid")
        if not isinstance(record.get("execution_diagnostic"), str):
            raise QualificationError(f"{route}.execution_diagnostic is malformed")
        if record.get("profile_capture") is not None:
            raise QualificationError(
                f"{route}.profile_capture is not allowed for qualification timing"
            )
        generation_policy = record.get("generation_policy")
        expected_policy = _expected_generation_policy(record, request)
        if not _json_values_equal(generation_policy, expected_policy):
            raise QualificationError(
                f"{route}.generation_policy does not match command"
            )
        if generation_policy is not None:
            fields = (
                HOST_GENERATION_POLICY_FIELDS
                if request["backend"] == "host"
                else CUDA_GENERATION_POLICY_FIELDS
            )
            if (
                not isinstance(generation_policy, Mapping)
                or set(generation_policy) != fields
            ):
                raise QualificationError(f"{route}.generation_policy fields mismatch")
            if request["backend"] == "host":
                _integer(
                    generation_policy["channel_tile"],
                    f"{route}.generation_policy.channel_tile",
                    minimum=1,
                    maximum=2**31 - 1,
                )
                _integer(
                    generation_policy["scalar_width"],
                    f"{route}.generation_policy.scalar_width",
                    allowed=(4, 8),
                )
                for name in HOST_GENERATION_POLICY_FIELDS - {
                    "channel_tile",
                    "scalar_width",
                }:
                    _string(
                        generation_policy[name],
                        f"{route}.generation_policy.{name}",
                    )
                from symmetrix.oam_receiver_codegen import OAMHostGenerationPolicy

                OAMHostGenerationPolicy(**generation_policy)
            else:
                budget = generation_policy["forward_live_output_budget"]
                if budget is not None:
                    _integer(
                        budget,
                        f"{route}.generation_policy.forward_live_output_budget",
                        allowed=(8, 16, 24, 32),
                    )
                for name in CUDA_GENERATION_POLICY_FIELDS - {
                    "forward_live_output_budget",
                }:
                    _string(
                        generation_policy[name],
                        f"{route}.generation_policy.{name}",
                    )
                from symmetrix.oam_cuda_codegen import OAMCudaGenerationPolicy

                OAMCudaGenerationPolicy(**generation_policy)
        if record.get("generation_policy_selection") != "automatic":
            raise QualificationError(
                f"{route}.generation_policy_selection contradicts the canonical command"
            )
        if not math.isclose(
            graph["effective_cutoff"],
            graph["cutoff"] + graph["skin"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise QualificationError(f"{route}.graph effective cutoff mismatch")
        if (
            record["atoms"] != graph["atom_count"]
            or record["directed_edges"] != graph["directed_edges"]
            or record["model_cutoff"] != graph["cutoff"]
            or record["effective_cutoff"] != graph["effective_cutoff"]
        ):
            raise QualificationError(f"{route} top-level graph facts mismatch")
        graph_config = _mapping(request.get("graph_config"), "request.graph_config")
        for name in ("structure", "seed", "rattle", "skin"):
            if record.get(name) != graph_config.get(name):
                raise QualificationError(f"{route}.{name} does not match graph request")
        count_name = "repeat" if graph_config["structure"] == "srtio3" else "atoms"
        if record.get(count_name) != graph_config[count_name]:
            raise QualificationError(
                f"{route}.{count_name} does not match graph request"
            )
        if graph_config["structure"] == "srtio3":
            repeat = graph_config["repeat"]
            factors = (repeat, repeat, repeat) if isinstance(repeat, int) else repeat
            expected_atoms = 5 * factors[0] * factors[1] * factors[2]
            if record["atoms"] != expected_atoms:
                raise QualificationError(
                    f"{route}.graph must describe exactly {expected_atoms} atoms"
                )
        numerical_values = _load_numerical_evidence(request, record, route)
        if record["energy"] != float(numerical_values["energy"]):
            raise QualificationError(f"{route}.energy contradicts numerical evidence")
        if record["max_force"] != float(np.max(np.abs(numerical_values["forces"]))):
            raise QualificationError(
                f"{route}.max_force contradicts numerical evidence"
            )
        for name, request_name in (("sample_count", "samples"), ("warmup", "warmups")):
            value = record.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != request[request_name]
            ):
                raise QualificationError(f"{route}.{name} does not match the request")

        affinity = record.get("cpu_affinity")
        if not isinstance(affinity, list) or not all(
            isinstance(cpu, int) and not isinstance(cpu, bool) for cpu in affinity
        ):
            raise QualificationError(f"{route}.cpu_affinity must be an integer array")
        thread_environment = _mapping(
            record.get("thread_environment"), f"{route}.thread_environment"
        )
        expected_thread_keys = {
            "KOKKOS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        }
        if set(thread_environment) != expected_thread_keys or not all(
            value is None or isinstance(value, str)
            for value in thread_environment.values()
        ):
            raise QualificationError(f"{route}.thread_environment is malformed")
        if request["backend"] == "host":
            requested = str(request["threads"])
            if any(
                thread_environment[name] != requested
                for name in ("KOKKOS_NUM_THREADS", "OMP_NUM_THREADS")
            ) or any(
                thread_environment[name] != "1"
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            ):
                raise QualificationError(f"{route}.thread_environment policy mismatch")
        execution_environment = _mapping(
            record.get("execution_environment"), f"{route}.execution_environment"
        )
        expected_environment_fields = (
            HOST_EXECUTION_ENVIRONMENT_FIELDS
            if request["backend"] == "host"
            else CUDA_EXECUTION_ENVIRONMENT_FIELDS
        )
        if set(execution_environment) != expected_environment_fields:
            raise QualificationError(f"{route}.execution_environment fields mismatch")
        if execution_environment.get("available") is not True:
            raise QualificationError(f"{route}.execution_environment is unavailable")
        expected_environment_backend = (
            "host" if request["backend"] == "host" else "cuda"
        )
        if execution_environment.get("backend") != expected_environment_backend:
            raise QualificationError(f"{route}.execution_environment backend mismatch")
        for name in ("execution_space", "memory_space"):
            if (
                not isinstance(execution_environment.get(name), str)
                or not execution_environment[name]
            ):
                raise QualificationError(
                    f"{route}.execution_environment.{name} must be non-empty"
                )
        if request["backend"] == "host":
            observed_target = _string(
                execution_environment.get("target"),
                f"{route}.execution_environment.target",
            )
        else:
            for name, minimum in (
                ("device_ordinal", 0),
                ("compute_capability_code", 1),
            ):
                _integer(
                    execution_environment.get(name),
                    f"{route}.execution_environment.{name}",
                    minimum=minimum,
                )
            observed_target = _string(
                execution_environment.get("architecture"),
                f"{route}.execution_environment.architecture",
            )
        expected_target = (
            "host" if request["backend"] == "host" else request["cuda_target"]
        )
        targets_match = observed_target == expected_target
        if request["backend"] == "cuda":
            targets_match = _normalized_cuda_target(
                observed_target
            ) == _normalized_cuda_target(expected_target)
        if not targets_match:
            raise QualificationError(f"{route}.execution_environment target mismatch")
        observed_runtime = record.get("observed_runtime")
        if request["backend"] == "host":
            observed_runtime = _mapping(observed_runtime, f"{route}.observed_runtime")
            if set(observed_runtime) != {
                "kokkos_execution_space",
                "kokkos_concurrency",
                "omp_max_threads",
                "worker_actual_team_size",
                "worker_cpu_observation_supported",
            }:
                raise QualificationError(f"{route}.observed_runtime fields mismatch")
            for name in (
                "kokkos_concurrency",
                "omp_max_threads",
                "worker_actual_team_size",
            ):
                if observed_runtime.get(name) != request["threads"]:
                    raise QualificationError(
                        f"{route}.observed_runtime.{name} mismatch"
                    )
            if observed_runtime.get("worker_cpu_observation_supported") is not True:
                raise QualificationError(
                    f"{route}.observed_runtime CPU observation is unavailable"
                )
            if not isinstance(observed_runtime.get("kokkos_execution_space"), str):
                raise QualificationError(
                    f"{route}.observed_runtime execution space is missing"
                )
        elif observed_runtime is not None:
            raise QualificationError(f"{route}.observed_runtime must be null for CUDA")
        timing = _mapping(record.get("timing_contract"), f"{route}.timing_contract")
        if timing != {
            "unit": "us/atom",
            "timed_region": TIMED_REGION,
            "excluded": list(TIMING_EXCLUSIONS),
        }:
            raise QualificationError(
                f"{route} timing contract is not qualification eligible"
            )
        if canonical_timing["unit"] != "us/atom":
            raise QualificationError(f"{route} canonical timing unit mismatch")

        execution = _mapping(record.get("execution"), f"{route}.execution")
        if set(execution) != {
            "plan_id",
            "edge_variant",
            "bindings",
            "schedule",
            "backend",
            "target",
            "fallback_count",
            "binding_report_version",
        }:
            raise QualificationError(f"{route}.execution fields mismatch")
        if execution.get("backend") != request["backend"]:
            raise QualificationError(f"{route}.execution.backend mismatch")
        expected_variant = (
            request["edge_variant"] if route == "direct" else "full_message"
        )
        if execution.get("edge_variant") != expected_variant:
            raise QualificationError(f"{route}.execution.edge_variant mismatch")
        execution_target = _string(execution.get("target"), f"{route}.execution.target")
        execution_target_matches = execution_target == expected_target
        if request["backend"] == "cuda":
            execution_target_matches = _normalized_cuda_target(
                execution_target
            ) == _normalized_cuda_target(expected_target)
        if not execution_target_matches:
            raise QualificationError(f"{route}.execution.target mismatch")
        if execution.get("binding_report_version") != request["binding_report_version"]:
            raise QualificationError(
                f"{route}.execution binding report version mismatch"
            )
        fallback = execution.get("fallback_count")
        if isinstance(fallback, bool) or not isinstance(fallback, int) or fallback != 0:
            raise QualificationError(
                f"{route}.execution.fallback_count must be integer zero"
            )
        expected_schedule = (
            "capacity" if request["profile"] == "capacity" else "throughput"
        )
        if execution.get("schedule") != expected_schedule:
            raise QualificationError(f"{route}.execution.schedule mismatch")
        expected_plan = (
            (
                "oam-direct-capacity"
                if request["profile"] == "capacity"
                else "oam-direct-speed"
            )
            if route == "direct"
            else "portable_oam_kokkos_generic"
        )
        if execution.get("plan_id") != expected_plan:
            raise QualificationError(f"{route}.execution.plan_id mismatch")
        memory = _mapping(record.get("memory"), f"{route}.memory")
        if set(memory) != {
            "active_bytes",
            "retained_bytes",
            "peak_transient_bytes",
            "replacements",
        }:
            raise QualificationError(f"{route}.memory fields mismatch")
        for name in (
            "active_bytes",
            "retained_bytes",
            "peak_transient_bytes",
            "replacements",
        ):
            value = memory.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise QualificationError(
                    f"{route}.memory.{name} must be a nonnegative integer"
                )
        if memory["retained_bytes"] < memory["active_bytes"]:
            raise QualificationError(
                f"{route}.memory.retained_bytes is smaller than active_bytes"
            )
        reuse = _mapping(record.get("reuse"), f"{route}.reuse")
        if set(reuse) != {"replacement_delta", "completed"}:
            raise QualificationError(f"{route}.reuse fields mismatch")
        if reuse.get("replacement_delta") != 0 or reuse.get("completed") is not True:
            raise QualificationError(f"{route}.reuse does not prove zero replacements")
        capacity_boundary = _mapping(
            record.get("capacity_boundary"), f"{route}.capacity_boundary"
        )
        expected_completed = request["kind"] == "capacity-boundary"
        if (
            set(capacity_boundary) != {"completed"}
            or not isinstance(capacity_boundary["completed"], bool)
            or capacity_boundary["completed"] != expected_completed
        ):
            raise QualificationError(f"{route}.capacity_boundary metadata mismatch")
        cpu = _mapping(record.get("cpu"), f"{route}.cpu")
        expected_cpu_fields = {
            "physical_cores",
            "kokkos_threads",
            "blas_threads",
            "logical_cpu_ids",
            "topology",
            "observed_kokkos_threads",
            "observed_openmp_threads",
        }
        if set(cpu) != expected_cpu_fields:
            raise QualificationError(f"{route}.cpu fields mismatch")
        if request["backend"] == "host":
            for name in (
                "physical_cores",
                "kokkos_threads",
                "observed_kokkos_threads",
                "observed_openmp_threads",
            ):
                if cpu[name] != request["threads"]:
                    raise QualificationError(f"{route}.cpu.{name} mismatch")
            if cpu["blas_threads"] != 1 or cpu["logical_cpu_ids"] != affinity:
                raise QualificationError(f"{route}.cpu affinity/thread mismatch")
            topology = cpu["topology"]
            if not isinstance(topology, list) or len(topology) != len(affinity):
                raise QualificationError(f"{route}.cpu.topology is malformed")
            if len(affinity) != request["threads"] or len(set(affinity)) != len(
                affinity
            ):
                raise QualificationError(f"{route}.cpu_affinity is not exact")
            physical = set()
            for index, row in enumerate(topology):
                row = _mapping(row, f"{route}.cpu.topology[{index}]")
                if (
                    set(row) != {"logical_cpu", "package", "core"}
                    or row["logical_cpu"] != affinity[index]
                ):
                    raise QualificationError(f"{route}.cpu.topology row mismatch")
                if any(
                    isinstance(row[name], bool) or not isinstance(row[name], int)
                    for name in ("logical_cpu", "package", "core")
                ):
                    raise QualificationError(f"{route}.cpu.topology types mismatch")
                physical.add((row["package"], row["core"]))
            if len(physical) != request["threads"]:
                raise QualificationError(f"{route}.cpu.topology contains SMT siblings")
        elif any(value is not None for value in cpu.values()):
            raise QualificationError(f"{route}.cpu must be null-valued for CUDA")
        artifact = _mapping(record.get("artifact"), f"{route}.artifact")
        if route == "direct":
            contract = _mapping(
                request.get("artifact_evidence_contract"),
                "request.artifact_evidence_contract",
            )
            direct_contract = _mapping(contract.get("direct"), "artifact contract")
            if request["backend"] == "host":
                expected_fields = {
                    "kind",
                    "expected_status",
                    "actual_status",
                    "expected_identity",
                    "actual_identity",
                    "expected_origin",
                    "actual_origin",
                    "sha256",
                    "expected_target",
                    "native_build_target",
                    "active_host_target",
                    "expected_floating_point_policy",
                    "actual_floating_point_policy",
                }
                if set(artifact) != expected_fields:
                    raise QualificationError("direct.artifact fields mismatch")
                expected_policy = request["host_floating_point_policy"]
                if (
                    artifact.get("kind") != "host-direct"
                    or artifact.get("expected_status") != "installed"
                    or artifact.get("actual_status") != "installed"
                    or artifact.get("expected_origin") != "prebuilt"
                    or artifact.get("actual_origin") != "prebuilt"
                    or artifact.get("expected_target") != request["host_target"]
                    or artifact.get("native_build_target")
                    != f"host-isa-{request['host_target']}"
                    or artifact.get("active_host_target")
                    != f"host-isa-{request['host_target']}"
                    or artifact.get("expected_floating_point_policy") != expected_policy
                    or artifact.get("actual_floating_point_policy") != expected_policy
                    or direct_contract
                    != {
                        "kind": "host-direct",
                        "expected_status": "installed",
                        "expected_origin": "prebuilt",
                        "expected_target": request["host_target"],
                        "expected_floating_point_policy": expected_policy,
                        "identity": "exact-derived",
                    }
                ):
                    raise QualificationError(
                        "direct.artifact host expectation mismatch"
                    )
            else:
                expected_fields = {
                    "kind",
                    "expected_status",
                    "actual_status",
                    "expected_identity",
                    "actual_identity",
                    "expected_origin",
                    "actual_origin",
                    "sha256",
                    "target",
                    "expected_compiler",
                    "actual_compiler",
                    "schedule",
                }
                if set(artifact) != expected_fields:
                    raise QualificationError("direct.artifact fields mismatch")
                try:
                    artifact_target_matches = _normalized_cuda_target(
                        artifact.get("target")
                    ) == _normalized_cuda_target(request["cuda_target"])
                except QualificationError:
                    artifact_target_matches = False
                if (
                    artifact.get("kind") != "cuda-direct"
                    or artifact.get("expected_status") != "installed"
                    or artifact.get("actual_status") != "installed"
                    or artifact.get("expected_origin") != "prebuilt"
                    or artifact.get("actual_origin") != "prebuilt"
                    or not artifact_target_matches
                    or not _json_values_equal(
                        artifact.get("schedule"), request["generation_policy_request"]
                    )
                    or artifact.get("expected_compiler") != "prebuilt"
                    or artifact.get("actual_compiler") != "prebuilt"
                    or direct_contract
                    != {
                        "kind": "cuda-direct",
                        "expected_target": request["cuda_target"],
                        "expected_status": "installed",
                        "expected_origin": "prebuilt",
                        "expected_compiler": "prebuilt",
                        "identity": "exact-derived",
                    }
                ):
                    raise QualificationError(
                        "direct.artifact CUDA expectation mismatch"
                    )
            for name in ("expected_identity", "actual_identity"):
                if not isinstance(artifact.get(name), str) or not artifact[name]:
                    raise QualificationError(f"direct.artifact.{name} is empty")
            if artifact["expected_identity"] != artifact["actual_identity"]:
                raise QualificationError("direct.artifact identity mismatch")
            if record.get("artifact_id") != artifact["actual_identity"]:
                raise QualificationError("direct.artifact_id mismatch")
            _digest(artifact.get("sha256"), "direct.artifact.sha256")
            artifact_path = record.get("artifact_path")
            if (
                not isinstance(artifact_path, str)
                or not Path(artifact_path).is_absolute()
            ):
                raise QualificationError("direct.artifact_path must be absolute")
            if not Path(artifact_path).is_file():
                raise QualificationError("direct artifact is absent")
            actual_artifact_sha256 = hashlib.sha256(
                Path(artifact_path).read_bytes()
            ).hexdigest()
            if actual_artifact_sha256 != artifact["sha256"]:
                raise QualificationError(
                    "direct.artifact.sha256 does not match artifact bytes"
                )
            if not isinstance(record.get("artifact_cache_status"), str):
                raise QualificationError("direct.artifact_cache_status is missing")
            if record["artifact_cache_status"] != "installed":
                raise QualificationError(
                    "direct artifact_cache_status must be installed"
                )
            if (
                request["backend"] == "cuda"
                and record.get("compiler") != artifact["actual_compiler"]
            ):
                raise QualificationError(
                    "direct compiler diagnostic contradicts artifact compiler"
                )
            expected_bindings = request["required_bindings"]
        elif artifact != {"kind": "reference", "origin": "none"}:
            raise QualificationError("reference artifact report is not truthful")
        else:
            contract = _mapping(
                request.get("artifact_evidence_contract"),
                "request.artifact_evidence_contract",
            )
            if contract.get("reference") != artifact:
                raise QualificationError(
                    "reference artifact contract does not match the request"
                )
            if record.get("artifact_id") != "reference":
                raise QualificationError("reference artifact_id is not truthful")
            if (
                record.get("artifact_path") is not None
                or record.get("artifact_cache_status") is not None
            ):
                raise QualificationError(
                    "reference artifact path/status is not truthful"
                )
            expected_bindings = request["required_reference_bindings"]
        bindings = execution.get("bindings")
        if not isinstance(bindings, list) or not bindings:
            raise QualificationError(f"{route}.execution.bindings must be non-empty")
        canonical_bindings = []
        operator_ids = []
        for index, binding in enumerate(bindings):
            binding = _mapping(binding, f"{route}.execution.bindings[{index}]")
            for field in (*REQUIRED_BINDING_FIELDS, "target", "precision"):
                if field not in binding:
                    raise QualificationError(
                        f"{route}.execution.bindings[{index}] missing {field}"
                    )
            if binding["target"] != request["backend"]:
                raise QualificationError(
                    f"{route}.execution.bindings[{index}] target mismatch"
                )
            precision = _mapping(
                binding["precision"], f"{route}.execution.bindings[{index}].precision"
            )
            if precision.get("compute") != request["precision"] or not isinstance(
                precision.get("accumulator"), str
            ):
                raise QualificationError(
                    f"{route}.execution.bindings[{index}] precision mismatch"
                )
            canonical_bindings.append(
                {field: binding[field] for field in REQUIRED_BINDING_FIELDS}
            )
            operator_ids.append(binding["operator_id"])
        if len(set(operator_ids)) != len(operator_ids):
            raise QualificationError(f"{route}.execution.bindings are duplicated")
        if canonical_bindings != expected_bindings:
            raise QualificationError(
                f"{route}.execution.bindings do not exactly match the required binding set"
            )
        optional_fields = set(record) & RAW_OPTIONAL_FIELDS
        cuda_fields = {
            "compiler",
            "threads_per_block",
            "forward_threads_per_block",
            "cuda_launch_count",
            "cuda_fallback_count",
        }
        if request["backend"] == "cuda" and route == "direct":
            if not cuda_fields.issubset(optional_fields):
                raise QualificationError(f"{route} CUDA diagnostic fields are missing")
        elif optional_fields & cuda_fields:
            raise QualificationError(f"{route} has unexpected CUDA diagnostic fields")
        for name in optional_fields - {"compiler"}:
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise QualificationError(f"{route}.{name} is invalid")
        if "compiler" in optional_fields and not isinstance(record["compiler"], str):
            raise QualificationError(f"{route}.compiler is invalid")
    except QualificationError:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise QualificationError(f"malformed {route} driver record: {error}") from error


def _build_cell_record(
    request: Mapping[str, Any],
    direct: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    baseline_us_per_atom: float | None,
) -> dict[str, Any]:
    """Strictly validate, compare, and preserve both raw evidence records."""

    validate_raw_driver_record(request, direct, "direct")
    validate_raw_driver_record(request, reference, "reference")
    if baseline_us_per_atom != request["baseline_us_per_atom"]:
        raise QualificationError("approved baseline does not match the request")
    for field in (
        "model_sha256",
        "graph_sha256",
        "graph",
        "structure",
        "repeat",
        "atoms",
        "rattle",
        "seed",
        "skin",
        "sample_count",
        "warmup",
        "cpu_affinity",
        "thread_environment",
        "execution_environment",
        "observed_runtime",
        "cpu",
        "capacity_boundary",
        "timing_contract",
        "python",
        "package",
        "native_extension",
        "native_sha256",
        "model",
        "atoms",
        "directed_edges",
        "model_cutoff",
        "effective_cutoff",
    ):
        if field not in direct and field not in reference:
            continue
        if direct[field] != reference[field]:
            raise QualificationError(f"direct/reference {field} mismatch")
    if direct["provenance"] != reference["provenance"]:
        raise QualificationError("direct/reference provenance mismatch")
    if _route_independent_command(direct["command"]) != _route_independent_command(
        reference["command"]
    ):
        raise QualificationError("direct/reference command mismatch")
    if (
        direct["execution"]["backend"] != reference["execution"]["backend"]
        or direct["execution"]["target"] != reference["execution"]["target"]
    ):
        raise QualificationError("direct/reference backend target mismatch")

    provenance = dict(direct["provenance"])
    provenance.update(
        model_sha256=direct["model_sha256"],
        graph_sha256=direct["graph_sha256"],
        command=direct["command"],
    )
    timing = _canonical_cell_timing(direct, reference)
    correctness = _correctness_report(request, direct, reference)
    return {
        "schema": CELL_SCHEMA,
        "version": SCHEMA_VERSION,
        "status": "measured",
        **{
            name: request[name]
            for name in (
                "cell_id",
                "kind",
                "model_size",
                "backend",
                "precision",
                "graph_class",
                "profile",
                "threads",
            )
        },
        "request": dict(request),
        "provenance": provenance,
        "artifact": dict(direct["artifact"]),
        "execution": dict(direct["execution"]),
        "graph": dict(direct["graph"]),
        "memory": dict(direct["memory"]),
        "reuse": dict(direct["reuse"]),
        "timing": timing,
        "correctness": correctness,
        "baseline": {
            "identity": request["baseline_id"],
            "sha256": request["baseline_sha256"],
            "direct_us_per_atom": baseline_us_per_atom,
        },
        "capacity_boundary": dict(direct["capacity_boundary"]),
        "cpu": dict(direct.get("cpu", {})),
        "evidence": {
            "direct": copy.deepcopy(direct),
            "reference": copy.deepcopy(reference),
        },
    }


def build_cell_record(
    request: Mapping[str, Any],
    direct: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    baseline_us_per_atom: float | None,
) -> dict[str, Any]:
    """Normalize every malformed paired-record path to QualificationError."""

    try:
        return _build_cell_record(
            request,
            direct,
            reference,
            baseline_us_per_atom=baseline_us_per_atom,
        )
    except QualificationError:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise QualificationError(f"malformed paired driver records: {error}") from error
