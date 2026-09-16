"""Advisory launch-calibration records for device implementations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "symmetrix.launch.tuning"
SCHEMA_VERSION = 2
TUNER_REVISION = "persistent-grid-v1"
DEFAULT_STAGE_SET = (
    "m0_forward",
    "m0_reverse",
    "m0_response_reverse",
    "r0_reverse",
    "r1_forward",
    "r1_reverse",
)
CACHE_ENVIRONMENT_VARIABLE = "SYMMETRIX_LAUNCH_TUNING_CACHE"
_RECORD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "cache_key",
        "identity",
        "decision",
        "measurements",
        "created_utc",
        "publication_id",
        "payload_sha256",
    }
)
_DECISION_FIELDS = frozenset({"stage", "profile_id", "kind", "blocks_per_compute_unit"})


class KernelLaunchTuningError(RuntimeError):
    """Raised when an explicit tuning-store operation cannot be completed."""


@dataclass(frozen=True)
class KernelLaunchTuningLookup:
    status: str
    cache_key: str
    record: dict | None = None
    reason: str | None = None


@dataclass(frozen=True)
class KernelLaunchTuningPublication:
    cache_key: str
    record: dict
    published: bool


def _canonical_json(value) -> bytes:
    def validate_object_keys(candidate) -> None:
        if isinstance(candidate, dict):
            for key, item in candidate.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "Execution launch tuning JSON object keys must be strings"
                    )
                validate_object_keys(item)
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                validate_object_keys(item)

    validate_object_keys(value)
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Execution launch tuning data must be canonical JSON"
        ) from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def kernel_launch_tuning_cache_root(
    cache_root: str | os.PathLike[str] | None = None,
) -> Path:
    if cache_root is not None:
        return Path(cache_root).expanduser()
    override = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser()
    from .jit import jit_cache_root

    return jit_cache_root() / "launch-tuning-v2"


def _ensure_private_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise KernelLaunchTuningError(
            "Execution launch tuning cache root must not be a symbolic link"
        )
    metadata = root.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise KernelLaunchTuningError(
            "Execution launch tuning cache root is not a directory"
        )
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise KernelLaunchTuningError(
            "Execution launch tuning cache root must be owned by the current user"
        )
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise KernelLaunchTuningError(
            "Execution launch tuning cache root must use permissions 0700"
        )


def _count_bucket(count: int) -> dict[str, int]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(
            "Execution tuning workload counts must be non-negative integers"
        )
    if count == 0:
        return {"lower": 0, "upper": 0}
    upper = 1 << (count - 1).bit_length()
    return {"lower": upper // 2 + 1 if upper > 1 else 1, "upper": upper}


def kernel_launch_workload_identity(num_nodes: int, num_edges: int, properties) -> dict:
    selected_properties = sorted({str(value) for value in properties})
    if not selected_properties:
        raise ValueError("Execution tuning property workflow must not be empty")
    return {
        "nodes": _count_bucket(num_nodes),
        "edges": _count_bucket(num_edges),
        "properties": selected_properties,
    }


def kernel_launch_tuning_identity(
    *,
    device_environment: dict,
    implementation_identity: str,
    model_identity: str,
    precision: str,
    workload: dict,
    stage_set=DEFAULT_STAGE_SET,
    tuner_revision: str = TUNER_REVISION,
) -> dict:
    if not isinstance(device_environment, dict):
        raise TypeError("Execution tuning device environment must be a dictionary")
    backend = device_environment.get("backend")
    architecture = device_environment.get("architecture")
    compute_units = device_environment.get("compute_unit_count")
    subgroup_width = device_environment.get("native_subgroup_width")
    if backend not in ("cuda", "hip"):
        raise ValueError("Execution launch tuning requires a CUDA or HIP backend")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("Execution tuning device architecture is unavailable")
    if isinstance(compute_units, bool) or not isinstance(compute_units, int):
        raise ValueError("Execution tuning compute-unit count is unavailable")
    if compute_units <= 0:
        raise ValueError("Execution tuning compute-unit count must be positive")
    if isinstance(subgroup_width, bool) or not isinstance(subgroup_width, int):
        raise ValueError("Execution tuning subgroup width is unavailable")
    if subgroup_width <= 0:
        raise ValueError("Execution tuning subgroup width must be positive")
    if precision not in ("float32", "float64"):
        raise ValueError("Execution tuning precision must be float32 or float64")

    def compatibility_major(value) -> str:
        text = str(value or "")
        return text.partition(".")[0] if text else ""

    identity = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "tuner_revision": str(tuner_revision),
        "device": {
            "backend": backend,
            "architecture": architecture,
            "target_features": str(device_environment.get("target_features", "") or ""),
            "compute_unit_count": compute_units,
            "native_subgroup_width": subgroup_width,
            "max_team_size": int(device_environment.get("max_team_size", 0)),
            "max_shared_memory_per_block": int(
                device_environment.get("max_shared_memory_per_block", 0)
            ),
            "runtime_major": compatibility_major(
                device_environment.get("runtime_version")
            ),
            "driver_major": compatibility_major(
                device_environment.get("driver_version")
            ),
        },
        "implementation_identity": str(implementation_identity),
        "model_identity": str(model_identity),
        "precision": precision,
        "stage_set": sorted({str(stage) for stage in stage_set}),
        "candidate_policy": "module-bounded-v2",
        "workload": workload,
    }
    if not identity["implementation_identity"] or not identity["model_identity"]:
        raise ValueError(
            "Execution tuning implementation and model identities are required"
        )
    if not identity["stage_set"]:
        raise ValueError("Execution tuning stage set must not be empty")
    _canonical_json(identity)
    return identity


def kernel_launch_tuning_cache_key(identity: dict) -> str:
    return _sha256(_canonical_json(identity))


def _record_digest(record: dict) -> str:
    return _sha256(
        _canonical_json(
            {key: value for key, value in record.items() if key != "payload_sha256"}
        )
    )


def validate_kernel_launch_tuning_record(
    record: dict, *, expected_identity: dict | None = None
) -> dict:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise KernelLaunchTuningError("launch tuning record has invalid fields")
    if record["schema"] != SCHEMA or record["version"] != SCHEMA_VERSION:
        raise KernelLaunchTuningError("launch tuning record schema is unsupported")
    identity = record["identity"]
    cache_key = kernel_launch_tuning_cache_key(identity)
    if record["cache_key"] != cache_key:
        raise KernelLaunchTuningError("launch tuning record cache key is invalid")
    if expected_identity is not None and identity != expected_identity:
        raise KernelLaunchTuningError("launch tuning record identity does not match")
    decisions = record["decision"]
    if not isinstance(decisions, list) or not decisions:
        raise KernelLaunchTuningError("launch tuning record has no decisions")
    stages = set()
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != _DECISION_FIELDS:
            raise KernelLaunchTuningError("launch tuning decision has invalid fields")
        if decision["kind"] not in ("module", "plugin"):
            raise KernelLaunchTuningError("launch tuning decision kind is invalid")
        if not isinstance(decision["stage"], str) or not decision["stage"]:
            raise KernelLaunchTuningError("launch tuning decision stage is invalid")
        if decision["stage"] in stages:
            raise KernelLaunchTuningError(
                "launch tuning decision stages must be unique"
            )
        stages.add(decision["stage"])
        if not isinstance(decision["profile_id"], str):
            raise KernelLaunchTuningError("launch tuning profile ID is invalid")
        blocks = decision["blocks_per_compute_unit"]
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
            raise KernelLaunchTuningError("launch tuning block count is invalid")
    if not isinstance(record["measurements"], dict):
        raise KernelLaunchTuningError("launch tuning measurements are invalid")
    if not isinstance(record["publication_id"], str) or not record["publication_id"]:
        raise KernelLaunchTuningError("launch tuning publication ID is invalid")
    if record["payload_sha256"] != _record_digest(record):
        raise KernelLaunchTuningError("launch tuning record digest is invalid")
    return record


def load_kernel_launch_tuning_record(
    identity: dict,
    *,
    cache_root: str | os.PathLike[str] | None = None,
) -> KernelLaunchTuningLookup:
    cache_key = kernel_launch_tuning_cache_key(identity)
    root = kernel_launch_tuning_cache_root(cache_root)
    if root.exists():
        try:
            _ensure_private_root(root)
        except (OSError, KernelLaunchTuningError) as error:
            return KernelLaunchTuningLookup(
                "ignored", cache_key, reason=f"{type(error).__name__}: {error}"
            )
    path = root / "records" / f"{cache_key}.json"
    if not path.is_file():
        return KernelLaunchTuningLookup("missing", cache_key)
    if path.is_symlink():
        return KernelLaunchTuningLookup(
            "ignored", cache_key, reason="launch tuning record must not be a symlink"
        )
    try:
        record = validate_kernel_launch_tuning_record(
            json.loads(path.read_text()), expected_identity=identity
        )
    except (
        OSError,
        TypeError,
        ValueError,
        KernelLaunchTuningError,
    ) as error:
        return KernelLaunchTuningLookup(
            "ignored", cache_key, reason=f"{type(error).__name__}: {error}"
        )
    return KernelLaunchTuningLookup("loaded", cache_key, record=record)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_kernel_launch_tuning_record(
    identity: dict,
    decisions: list[dict],
    measurements: dict,
    *,
    cache_root: str | os.PathLike[str] | None = None,
) -> KernelLaunchTuningPublication:
    cache_key = kernel_launch_tuning_cache_key(identity)
    _canonical_json(decisions)
    _canonical_json(measurements)
    root = kernel_launch_tuning_cache_root(cache_root)
    _ensure_private_root(root)
    records = root / "records"
    staging = root / ".staging"
    records.mkdir(mode=0o700, exist_ok=True)
    staging.mkdir(mode=0o700, exist_ok=True)
    destination = records / f"{cache_key}.json"
    stage = staging / f"{cache_key}-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    temporary = stage / "record.json"
    record = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "cache_key": cache_key,
        "identity": identity,
        "decision": decisions,
        "measurements": measurements,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "publication_id": uuid.uuid4().hex,
        "payload_sha256": "",
    }
    record["payload_sha256"] = _record_digest(record)
    validate_kernel_launch_tuning_record(record, expected_identity=identity)
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical_json(record) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        _fsync_directory(stage)
        try:
            os.link(temporary, destination)
            published = True
            _fsync_directory(records)
        except FileExistsError:
            existing = load_kernel_launch_tuning_record(identity, cache_root=cache_root)
            if existing.status != "loaded":
                raise KernelLaunchTuningError(
                    "a conflicting launch tuning record already exists"
                )
            record = existing.record
    finally:
        try:
            temporary.unlink(missing_ok=True)
            stage.rmdir()
        except OSError:
            pass
    return KernelLaunchTuningPublication(cache_key, record, published)


def remove_kernel_launch_tuning_record(
    identity: dict,
    *,
    cache_root: str | os.PathLike[str] | None = None,
) -> bool:
    cache_key = kernel_launch_tuning_cache_key(identity)
    path = kernel_launch_tuning_cache_root(cache_root) / "records" / f"{cache_key}.json"
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    _fsync_directory(path.parent)
    return True


__all__ = [
    "CACHE_ENVIRONMENT_VARIABLE",
    "DEFAULT_STAGE_SET",
    "SCHEMA",
    "SCHEMA_VERSION",
    "KernelLaunchTuningError",
    "KernelLaunchTuningLookup",
    "KernelLaunchTuningPublication",
    "load_kernel_launch_tuning_record",
    "publish_kernel_launch_tuning_record",
    "remove_kernel_launch_tuning_record",
    "kernel_launch_tuning_cache_key",
    "kernel_launch_tuning_cache_root",
    "kernel_launch_tuning_identity",
    "kernel_launch_workload_identity",
    "validate_kernel_launch_tuning_record",
]
