"""Build-safe identity and source helpers for generated execution operators."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Any

GENERATED_OPERATOR_IDENTITY_SCHEMA = "symmetrix.generated-operator-identity/1"
GENERATED_OPERATOR_CATALOG_SCHEMA = "symmetrix.generated-operator-catalog/1"
R0_DEVICE_ABI_VERSION = 5
R0_HOST_ABI_VERSION = 2
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTITY_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]*\Z")
_R0_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "operator",
        "generation_revision",
        "compatibility_version",
        "contract_fingerprint",
        "structure_fingerprint",
        "precision",
        "backend",
        "target",
        "channels",
        "l_max",
        "harmonic_count",
        "edge_weight_mode",
        "floating_point_policy",
        "schedule",
        "schedule_revision",
        "artifact_id",
    }
)
_M0_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "operator",
        "generation_revision",
        "structure_fingerprint",
        "precision",
        "backend",
        "target",
        "channels",
        "input_components",
        "output_components",
        "correlation",
        "term_count",
        "schedule",
        "schedule_revision",
        "artifact_id",
    }
)
_M1_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "operator",
        "generation_revision",
        "structure_fingerprint",
        "precision",
        "backend",
        "target",
        "channels",
        "input_components",
        "correlation",
        "term_count",
        "schedule",
        "schedule_revision",
        "artifact_id",
    }
)
_R1_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "operator",
        "generation_revision",
        "contract_fingerprint",
        "semantic_fingerprint",
        "structure_fingerprint",
        "precision",
        "backend",
        "target",
        "channels",
        "embedding",
        "edge_l_max",
        "source_l_max",
        "output_components",
        "edge_weight_mode",
        "schedule",
        "schedule_revision",
        "artifact_id",
    }
)
_OAM_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "operator",
        "generation_revision",
        "architecture",
        "architecture_fingerprint",
        "precision",
        "backend",
        "target",
        "generation_fingerprint",
        "semantic_fingerprint",
        "structure_fingerprint",
        "artifact_id",
    }
)
_CATALOG_ARTIFACT_FIELDS = frozenset(
    {
        "operator",
        "identity",
        "binary_format",
        "installed_relative_path",
        "checksum",
        "abi",
        "capabilities",
        "build_provenance",
    }
)
_CATALOG_ARTIFACT_AUXILIARY_FIELDS = _CATALOG_ARTIFACT_FIELDS | frozenset(
    {"auxiliary_files"}
)
_CATALOG_ABI_BY_OPERATOR_BACKEND_AND_FORMAT = {
    ("M0", "host", "shared-library"): ("symmetrix.jit.m0-host-plugin/2", 2),
    ("M0", "cuda", "cubin"): ("symmetrix.jit.operator-module/1", 3),
    ("M0", "hip", "code-object"): ("symmetrix.jit.operator-module/1", 3),
    ("M1", "host", "shared-library"): ("symmetrix.jit.m1-host-plugin/2", 2),
    ("M1", "cuda", "cubin"): ("symmetrix.jit.operator-module/1", 3),
    ("M1", "hip", "code-object"): ("symmetrix.jit.operator-module/1", 3),
    ("R0", "host", "shared-library"): (
        "symmetrix.jit.r0-host-plugin/2",
        R0_HOST_ABI_VERSION,
    ),
    ("R0", "cuda", "cubin"): (
        "symmetrix.jit.operator-module/1",
        R0_DEVICE_ABI_VERSION,
    ),
    ("R0", "hip", "code-object"): (
        "symmetrix.jit.operator-module/1",
        R0_DEVICE_ABI_VERSION,
    ),
    ("R1", "host", "shared-library"): ("symmetrix.jit.host-plugin/4", 4),
    ("R1", "cuda", "cubin"): ("symmetrix.jit.cuda-plugin/4", 4),
    ("R1", "hip", "code-object"): ("symmetrix.jit.hip-plugin/3", 3),
    ("OAM", "host", "shared-library"): (
        "symmetrix.jit.oam.host-plugin/7",
        7,
    ),
    ("OAM", "cuda", "cubin"): ("symmetrix.jit.oam.cuda-plugin/3", 3),
}
_INSTALLED_BUILD_FIELDS = frozenset(
    {
        "mode",
        "backend",
        "target",
        "catalog_relative_path",
        "catalog_sha256",
        "artifact_count",
        "installed_bytes",
        "source_content_sha256",
        "native_source_content_sha256",
        "compiler_id",
        "compiler_version",
        "build_duration_seconds",
    }
)


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a complete SHA-256 fingerprint")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must not be empty")
    return value


def _identity_token(value: Any, name: str) -> str:
    value = _nonempty_text(value, name)
    if _IDENTITY_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _exact_fields(value: Any, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = expected - value.keys()
    if missing:
        raise ValueError(f"{name} is missing {', '.join(sorted(missing))}")
    unexpected = value.keys() - expected
    if unexpected:
        raise ValueError(f"{name} has unexpected {', '.join(sorted(unexpected))}")
    return value


def _generation_revision(generation_revision: Any, generation_version: Any) -> int:
    if generation_revision is None:
        generation_revision = generation_version
    elif generation_version is not None and generation_revision != generation_version:
        raise ValueError("generation_revision and generation_version disagree")
    return _positive_integer(generation_revision, "generation_revision")


def _operator_identity(
    operator: str, generation_revision: int, precision: str, backend: str, values: dict
) -> dict[str, Any]:
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if backend not in ("host", "cuda", "hip"):
        raise ValueError("backend must be 'host', 'cuda', or 'hip'")
    identity = {
        "schema": GENERATED_OPERATOR_IDENTITY_SCHEMA,
        "operator": operator,
        "generation_revision": _positive_integer(
            generation_revision, "generation_revision"
        ),
        "precision": precision,
        "backend": backend,
        **values,
    }
    payload = json.dumps(
        identity, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    suffix = "f32" if precision == "float32" else "f64"
    identity["artifact_id"] = (
        f"generated-{operator.lower()}-gen{generation_revision}-{backend}-{suffix}-{digest}"
    )
    return identity


def generated_m0_identity(
    *,
    generation_revision: int,
    structure_fingerprint: str,
    precision: str,
    backend: str,
    target: str,
    channels: int,
    input_components: int,
    output_components: int,
    correlation: int,
    term_count: int,
    schedule: str,
    schedule_revision: int,
) -> dict[str, Any]:
    """Return the canonical identity for a generated M0 artifact."""

    return _operator_identity(
        "M0",
        generation_revision,
        precision,
        backend,
        {
            "structure_fingerprint": _fingerprint(
                structure_fingerprint, "structure_fingerprint"
            ),
            "target": _identity_token(target, "target"),
            "channels": _positive_integer(channels, "channels"),
            "input_components": _positive_integer(input_components, "input_components"),
            "output_components": _positive_integer(
                output_components, "output_components"
            ),
            "correlation": _positive_integer(correlation, "correlation"),
            "term_count": _positive_integer(term_count, "term_count"),
            "schedule": _identity_token(schedule, "schedule"),
            "schedule_revision": _positive_integer(
                schedule_revision, "schedule_revision"
            ),
        },
    )


def generated_m1_identity(
    *,
    generation_revision: int,
    structure_fingerprint: str,
    precision: str,
    backend: str,
    target: str,
    channels: int,
    input_components: int,
    correlation: int,
    term_count: int,
    schedule: str,
    schedule_revision: int,
) -> dict[str, Any]:
    """Return the canonical identity for a generated M1 artifact."""

    return _operator_identity(
        "M1",
        generation_revision,
        precision,
        backend,
        {
            "structure_fingerprint": _fingerprint(
                structure_fingerprint, "structure_fingerprint"
            ),
            "target": _identity_token(target, "target"),
            "channels": _positive_integer(channels, "channels"),
            "input_components": _positive_integer(input_components, "input_components"),
            "correlation": _positive_integer(correlation, "correlation"),
            "term_count": _positive_integer(term_count, "term_count"),
            "schedule": _identity_token(schedule, "schedule"),
            "schedule_revision": _positive_integer(
                schedule_revision, "schedule_revision"
            ),
        },
    )


def generated_r1_identity(
    *,
    generation_revision: int,
    contract_fingerprint: str,
    semantic_fingerprint: str,
    structure_fingerprint: str,
    precision: str,
    backend: str,
    target: str,
    channels: int,
    embedding: int,
    edge_l_max: int,
    source_l_max: int,
    output_components: int,
    edge_weight_mode: str,
    schedule: str,
    schedule_revision: int,
) -> dict[str, Any]:
    """Return the canonical identity for a generated Execution R1 artifact."""

    if edge_weight_mode not in ("unit", "runtime"):
        raise ValueError("edge_weight_mode must be 'unit' or 'runtime'")
    return _operator_identity(
        "R1",
        generation_revision,
        precision,
        backend,
        {
            "contract_fingerprint": _fingerprint(
                contract_fingerprint, "contract_fingerprint"
            ),
            "semantic_fingerprint": _fingerprint(
                semantic_fingerprint, "semantic_fingerprint"
            ),
            "structure_fingerprint": _fingerprint(
                structure_fingerprint, "structure_fingerprint"
            ),
            "target": _identity_token(target, "target"),
            "channels": _positive_integer(channels, "channels"),
            "embedding": _positive_integer(embedding, "embedding"),
            "edge_l_max": _positive_integer(edge_l_max, "edge_l_max", allow_zero=True),
            "source_l_max": _positive_integer(
                source_l_max, "source_l_max", allow_zero=True
            ),
            "output_components": _positive_integer(
                output_components, "output_components"
            ),
            "edge_weight_mode": edge_weight_mode,
            "schedule": _identity_token(schedule, "schedule"),
            "schedule_revision": _positive_integer(
                schedule_revision, "schedule_revision"
            ),
        },
    )


def generated_r0_identity(
    *,
    contract_fingerprint: str,
    structure_fingerprint: str,
    precision: str,
    backend: str,
    target: str,
    channels: int,
    l_max: int,
    edge_weight_mode: str,
    floating_point_policy: str | None = None,
    schedule: str,
    schedule_revision: int,
    compatibility_version: int,
    generation_revision: int | None = None,
    generation_version: int | None = None,
) -> dict[str, Any]:
    """Return the canonical identity shared by prebuilt and runtime R0 artifacts."""

    generation_revision = _generation_revision(generation_revision, generation_version)
    contract_fingerprint = _fingerprint(contract_fingerprint, "contract_fingerprint")
    structure_fingerprint = _fingerprint(structure_fingerprint, "structure_fingerprint")
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if backend not in ("host", "cuda", "hip"):
        raise ValueError("backend must be 'host', 'cuda', or 'hip'")
    if floating_point_policy is None:
        floating_point_policy = "cuda-fast-math" if backend == "cuda" else "strict"
    allowed_policies = {"strict", "cuda-fast-math"} if backend == "cuda" else {"strict"}
    if floating_point_policy not in allowed_policies:
        raise ValueError(
            f"floating_point_policy {floating_point_policy!r} is unsupported for "
            f"backend {backend!r}"
        )
    target = _identity_token(target, "target")
    channels = _positive_integer(channels, "channels")
    l_max = _positive_integer(l_max, "l_max", allow_zero=True)
    if edge_weight_mode not in ("unity", "weighted"):
        raise ValueError("edge_weight_mode must be 'unity' or 'weighted'")
    schedule = _identity_token(schedule, "schedule")
    schedule_revision = _positive_integer(schedule_revision, "schedule_revision")
    compatibility_version = _positive_integer(
        compatibility_version, "compatibility_version"
    )
    harmonic_count = (l_max + 1) ** 2
    identity = {
        "schema": GENERATED_OPERATOR_IDENTITY_SCHEMA,
        "operator": "R0",
        "generation_revision": generation_revision,
        "compatibility_version": compatibility_version,
        "contract_fingerprint": contract_fingerprint,
        "structure_fingerprint": structure_fingerprint,
        "precision": precision,
        "backend": backend,
        "target": target,
        "channels": channels,
        "l_max": l_max,
        "harmonic_count": harmonic_count,
        "edge_weight_mode": edge_weight_mode,
        "floating_point_policy": floating_point_policy,
        "schedule": schedule,
        "schedule_revision": schedule_revision,
    }
    payload = json.dumps(
        identity, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    precision_suffix = "f32" if precision == "float32" else "f64"
    identity["artifact_id"] = (
        f"generated-r0-gen{generation_revision}-{backend}-{precision_suffix}-{digest}"
    )
    return identity


def validate_generated_r0_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a serialized canonical R0 identity."""

    identity = _exact_fields(identity, _R0_IDENTITY_FIELDS, "identity")
    if identity["schema"] != GENERATED_OPERATOR_IDENTITY_SCHEMA:
        raise ValueError("identity schema is unsupported")
    if identity["operator"] != "R0":
        raise ValueError("identity operator must be 'R0'")
    expected = generated_r0_identity(
        generation_revision=identity["generation_revision"],
        compatibility_version=identity["compatibility_version"],
        contract_fingerprint=identity["contract_fingerprint"],
        structure_fingerprint=identity["structure_fingerprint"],
        precision=identity["precision"],
        backend=identity["backend"],
        target=identity["target"],
        channels=identity["channels"],
        l_max=identity["l_max"],
        edge_weight_mode=identity["edge_weight_mode"],
        floating_point_policy=identity["floating_point_policy"],
        schedule=identity["schedule"],
        schedule_revision=identity["schedule_revision"],
    )
    if identity["harmonic_count"] != expected["harmonic_count"]:
        raise ValueError("identity harmonic_count does not match l_max")
    if identity["artifact_id"] != expected["artifact_id"]:
        raise ValueError("identity artifact_id does not match its dimensions")
    return expected


def validate_generated_m0_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a serialized canonical M0 identity."""

    identity = _exact_fields(identity, _M0_IDENTITY_FIELDS, "identity")
    if identity["schema"] != GENERATED_OPERATOR_IDENTITY_SCHEMA:
        raise ValueError("identity schema is unsupported")
    if identity["operator"] != "M0":
        raise ValueError("identity operator must be 'M0'")
    expected = generated_m0_identity(
        generation_revision=identity["generation_revision"],
        structure_fingerprint=identity["structure_fingerprint"],
        precision=identity["precision"],
        backend=identity["backend"],
        target=identity["target"],
        channels=identity["channels"],
        input_components=identity["input_components"],
        output_components=identity["output_components"],
        correlation=identity["correlation"],
        term_count=identity["term_count"],
        schedule=identity["schedule"],
        schedule_revision=identity["schedule_revision"],
    )
    if identity["artifact_id"] != expected["artifact_id"]:
        raise ValueError("identity artifact_id does not match its dimensions")
    return expected


def validate_generated_m1_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a serialized canonical M1 identity."""

    identity = _exact_fields(identity, _M1_IDENTITY_FIELDS, "identity")
    if identity["schema"] != GENERATED_OPERATOR_IDENTITY_SCHEMA:
        raise ValueError("identity schema is unsupported")
    if identity["operator"] != "M1":
        raise ValueError("identity operator must be 'M1'")
    expected = generated_m1_identity(
        generation_revision=identity["generation_revision"],
        structure_fingerprint=identity["structure_fingerprint"],
        precision=identity["precision"],
        backend=identity["backend"],
        target=identity["target"],
        channels=identity["channels"],
        input_components=identity["input_components"],
        correlation=identity["correlation"],
        term_count=identity["term_count"],
        schedule=identity["schedule"],
        schedule_revision=identity["schedule_revision"],
    )
    if identity["artifact_id"] != expected["artifact_id"]:
        raise ValueError("identity artifact_id does not match its dimensions")
    return expected


def validate_generated_r1_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a serialized canonical R1 identity."""

    identity = _exact_fields(identity, _R1_IDENTITY_FIELDS, "identity")
    if identity["schema"] != GENERATED_OPERATOR_IDENTITY_SCHEMA:
        raise ValueError("identity schema is unsupported")
    if identity["operator"] != "R1":
        raise ValueError("identity operator must be 'R1'")
    expected = generated_r1_identity(
        generation_revision=identity["generation_revision"],
        contract_fingerprint=identity["contract_fingerprint"],
        semantic_fingerprint=identity["semantic_fingerprint"],
        structure_fingerprint=identity["structure_fingerprint"],
        precision=identity["precision"],
        backend=identity["backend"],
        target=identity["target"],
        channels=identity["channels"],
        embedding=identity["embedding"],
        edge_l_max=identity["edge_l_max"],
        source_l_max=identity["source_l_max"],
        output_components=identity["output_components"],
        edge_weight_mode=identity["edge_weight_mode"],
        schedule=identity["schedule"],
        schedule_revision=identity["schedule_revision"],
    )
    if identity["artifact_id"] != expected["artifact_id"]:
        raise ValueError("identity artifact_id does not match its dimensions")
    return expected


def validate_generated_operator_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact canonical M0, M1, R0, or R1 identity."""

    if not isinstance(identity, Mapping):
        raise ValueError("identity must be an object")
    validator = {
        "M0": validate_generated_m0_identity,
        "M1": validate_generated_m1_identity,
        "R0": validate_generated_r0_identity,
        "R1": validate_generated_r1_identity,
        "OAM": validate_generated_oam_identity,
    }.get(identity.get("operator"))
    if validator is None:
        raise ValueError("identity operator is unsupported")
    return validator(identity)


def validate_generated_oam_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an OAM identity selected by its explicit layer architecture."""

    identity_fields = _OAM_IDENTITY_FIELDS
    if identity.get("backend") == "host":
        identity_fields |= frozenset({"floating_point_policy"})
    identity = _exact_fields(identity, identity_fields, "OAM identity")
    if identity["schema"] != GENERATED_OPERATOR_IDENTITY_SCHEMA:
        raise ValueError("OAM identity schema is unsupported")
    if identity["operator"] != "OAM":
        raise ValueError("identity operator must be 'OAM'")
    architecture = _exact_fields(
        identity["architecture"],
        frozenset({"schema", "layer_count", "layers"}),
        "OAM architecture",
    )
    if architecture["schema"] != "symmetrix.oam.generated-architecture/1":
        raise ValueError("OAM architecture schema is unsupported")
    layers = architecture["layers"]
    if not isinstance(layers, list) or not layers:
        raise ValueError("OAM architecture layers must be non-empty")
    if architecture["layer_count"] != len(layers):
        raise ValueError("OAM architecture layer_count does not match layers")
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or layer.get("index") != index:
            raise ValueError("OAM architecture layer indices must be contiguous")
    codegen_architecture = {
        **architecture,
        "layers": [
            {key: value for key, value in layer.items() if key != "post_linear"}
            for layer in layers
        ],
    }
    try:
        encoded = json.dumps(
            codegen_architecture,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("OAM codegen architecture is not canonical JSON") from error
    architecture_fingerprint = _fingerprint(
        identity["architecture_fingerprint"], "architecture_fingerprint"
    )
    expected_fingerprint = "sha256:" + hashlib.sha256(encoded).hexdigest()
    if architecture_fingerprint != expected_fingerprint:
        raise ValueError(
            "OAM architecture fingerprint does not match codegen architecture"
        )
    precision = identity["precision"]
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    backend = identity["backend"]
    if backend not in ("host", "cuda"):
        raise ValueError("OAM backend must be 'host' or 'cuda'")
    normalized = {
        "schema": GENERATED_OPERATOR_IDENTITY_SCHEMA,
        "operator": "OAM",
        "generation_revision": _positive_integer(
            identity["generation_revision"], "generation_revision"
        ),
        "architecture": dict(architecture),
        "architecture_fingerprint": architecture_fingerprint,
        "precision": precision,
        "backend": backend,
        "target": _identity_token(identity["target"], "target"),
        "generation_fingerprint": _fingerprint(
            identity["generation_fingerprint"], "generation_fingerprint"
        ),
        "semantic_fingerprint": _fingerprint(
            identity["semantic_fingerprint"], "semantic_fingerprint"
        ),
        "structure_fingerprint": _fingerprint(
            identity["structure_fingerprint"], "structure_fingerprint"
        ),
        "artifact_id": _identity_token(identity["artifact_id"], "artifact_id"),
    }
    if backend == "host":
        policy = _nonempty_text(
            identity["floating_point_policy"], "floating_point_policy"
        )
        if policy not in ("strict", "fast"):
            raise ValueError("OAM host floating-point policy must be strict or fast")
        normalized["floating_point_policy"] = policy
    return normalized


def _installed_relative_path(value: Any) -> str:
    value = _nonempty_text(value, "installed_relative_path")
    if (
        "\\" in value
        or PureWindowsPath(value).drive
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("installed_relative_path must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0].endswith(":")
        or ".." in path.parts
        or str(path) != value
        or value.endswith("/")
    ):
        raise ValueError("installed_relative_path must be a normalized safe path")
    return value


def _catalog_artifact(value: Any) -> dict[str, Any]:
    fields = (
        _CATALOG_ARTIFACT_AUXILIARY_FIELDS
        if isinstance(value, Mapping) and "auxiliary_files" in value
        else _CATALOG_ARTIFACT_FIELDS
    )
    value = _exact_fields(value, fields, "catalog artifact")
    operator = _nonempty_text(value["operator"], "operator")
    identity = validate_generated_operator_identity(value["identity"])
    if operator != identity["operator"]:
        raise ValueError("operator does not match identity operator")

    binary_format = _nonempty_text(value["binary_format"], "binary_format")
    expected_format = {
        "host": "shared-library",
        "cuda": "cubin",
        "hip": "code-object",
    }[identity["backend"]]
    if binary_format != expected_format:
        raise ValueError(
            f"binary_format for {identity['backend']} must be {expected_format!r}"
        )

    abi = _exact_fields(value["abi"], frozenset({"tag", "version"}), "abi")
    normalized_abi = {
        "tag": _nonempty_text(abi["tag"], "abi tag"),
        "version": _positive_integer(abi["version"], "abi version"),
    }
    expected_abi = _CATALOG_ABI_BY_OPERATOR_BACKEND_AND_FORMAT.get(
        (operator, identity["backend"], binary_format)
    )
    if expected_abi is None:
        raise ValueError(
            f"binary format {binary_format!r} is unsupported for "
            f"{operator}/{identity['backend']}"
        )
    if (normalized_abi["tag"], normalized_abi["version"]) != expected_abi:
        raise ValueError(
            f"abi for {identity['backend']}/{binary_format} must be "
            f"tag {expected_abi[0]!r} at version {expected_abi[1]}"
        )
    provenance_fields = {"source_content_sha256", "compiler_id", "compiler_version"}
    if operator == "R0" or (operator == "OAM" and identity["backend"] == "host"):
        provenance_fields.add("floating_point_policy")
    provenance = _exact_fields(
        value["build_provenance"], frozenset(provenance_fields), "build_provenance"
    )
    normalized_provenance = {
        "source_content_sha256": _fingerprint(
            provenance["source_content_sha256"], "source_content_sha256"
        ),
        "compiler_id": _nonempty_text(provenance["compiler_id"], "compiler_id"),
        "compiler_version": _nonempty_text(
            provenance["compiler_version"], "compiler_version"
        ),
    }
    if operator == "R0" or (operator == "OAM" and identity["backend"] == "host"):
        policy = _nonempty_text(
            provenance["floating_point_policy"], "floating_point_policy"
        )
        if policy != identity["floating_point_policy"]:
            raise ValueError(
                "build provenance floating-point policy does not match identity"
            )
        normalized_provenance["floating_point_policy"] = policy
    capabilities = _positive_integer(value["capabilities"], "capabilities")
    normalized = {
        "operator": operator,
        "identity": identity,
        "binary_format": binary_format,
        "installed_relative_path": _installed_relative_path(
            value["installed_relative_path"]
        ),
        "checksum": _fingerprint(value["checksum"], "checksum"),
        "abi": normalized_abi,
        "capabilities": capabilities,
        "build_provenance": normalized_provenance,
    }
    if operator == "OAM":
        auxiliary = value.get("auxiliary_files")
        if not isinstance(auxiliary, list) or len(auxiliary) != 1:
            raise ValueError("OAM catalog artifact must have one manifest")
        item = _exact_fields(
            auxiliary[0],
            frozenset({"role", "installed_relative_path", "checksum"}),
            "OAM auxiliary file",
        )
        if item["role"] != "manifest":
            raise ValueError("OAM auxiliary file role must be manifest")
        normalized["auxiliary_files"] = [
            {
                "role": "manifest",
                "installed_relative_path": _installed_relative_path(
                    item["installed_relative_path"]
                ),
                "checksum": _fingerprint(item["checksum"], "manifest checksum"),
            }
        ]
    elif "auxiliary_files" in value:
        raise ValueError("only OAM artifacts may declare auxiliary files")
    return normalized


def validate_generated_operator_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a versioned installed generated-operator artifact catalog."""

    catalog = _exact_fields(catalog, frozenset({"schema", "artifacts"}), "catalog")
    if catalog["schema"] != GENERATED_OPERATOR_CATALOG_SCHEMA:
        raise ValueError("catalog schema is unsupported")
    artifacts = catalog["artifacts"]
    if not isinstance(artifacts, list):
        raise ValueError("catalog artifacts must be a list")
    normalized = [_catalog_artifact(artifact) for artifact in artifacts]
    artifact_ids = [artifact["identity"]["artifact_id"] for artifact in normalized]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("catalog artifact identities must be unique")
    paths = [artifact["installed_relative_path"] for artifact in normalized]
    paths.extend(
        auxiliary["installed_relative_path"]
        for artifact in normalized
        for auxiliary in artifact.get("auxiliary_files", ())
    )
    if len(set(paths)) != len(paths):
        raise ValueError("catalog installed_relative_path values must be unique")
    normalized.sort(key=lambda artifact: artifact["identity"]["artifact_id"])
    return {"schema": GENERATED_OPERATOR_CATALOG_SCHEMA, "artifacts": normalized}


def generated_operator_catalog(
    artifacts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic validated generated-operator catalog."""

    if isinstance(artifacts, (str, bytes, Mapping)):
        raise ValueError("catalog artifacts must be an iterable of objects")
    try:
        artifact_list = list(artifacts)
    except TypeError as exc:
        raise ValueError("catalog artifacts must be an iterable of objects") from exc
    return validate_generated_operator_catalog(
        {"schema": GENERATED_OPERATOR_CATALOG_SCHEMA, "artifacts": artifact_list}
    )


def _read_regular_file_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is unreadable: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(chunks)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while it was being read")
    return payload


def _validate_installed_operator_artifacts_at_root(
    package_root: Path, build_info: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    build_info = _exact_fields(
        build_info, _INSTALLED_BUILD_FIELDS, "prebuilt operator build info"
    )
    mode = build_info["mode"]
    if mode not in ("off", "r0", "full", "oam"):
        raise ValueError("prebuilt operator mode is unsupported")
    if mode == "off":
        if any(
            build_info[field]
            for field in (
                "catalog_relative_path",
                "catalog_sha256",
                "artifact_count",
                "installed_bytes",
            )
        ):
            raise ValueError("off prebuilt operator mode must not name package data")
        return (
            {
                "status": "not_applicable",
                "artifact_count": 0,
                "installed_bytes": 0,
                "catalog_sha256": "",
                "catalog_path": "",
            },
            {"schema": GENERATED_OPERATOR_CATALOG_SCHEMA, "artifacts": []},
            package_root,
        )

    backend = _nonempty_text(build_info["backend"], "prebuilt operator backend")
    if backend not in ("host", "cuda", "hip"):
        raise ValueError("prebuilt operator backend is unsupported")
    target = _identity_token(build_info["target"], "prebuilt operator target")
    source_sha256 = _fingerprint(
        "sha256:" + str(build_info["source_content_sha256"]),
        "prebuilt operator source_content_sha256",
    )
    _fingerprint(
        "sha256:" + str(build_info["native_source_content_sha256"]),
        "prebuilt operator native_source_content_sha256",
    )
    compiler_id = _nonempty_text(build_info["compiler_id"], "compiler_id")
    compiler_version = _nonempty_text(
        build_info["compiler_version"], "compiler_version"
    )
    expected_count = _positive_integer(build_info["artifact_count"], "artifact_count")
    expected_bytes = _positive_integer(build_info["installed_bytes"], "installed_bytes")
    expected_catalog_sha256 = _fingerprint(
        "sha256:" + str(build_info["catalog_sha256"]), "catalog_sha256"
    ).removeprefix("sha256:")
    catalog_relative = _installed_relative_path(build_info["catalog_relative_path"])
    package_root = package_root.resolve()
    catalog_parts = PurePosixPath(catalog_relative).parts
    catalog_path = package_root.joinpath(*catalog_parts)
    current = package_root
    for part in catalog_parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError(
                "installed operator catalog must use a contained real path"
            )
    try:
        catalog_path.resolve().relative_to(package_root)
    except ValueError as error:
        raise ValueError(
            "installed operator catalog must use a contained real path"
        ) from error
    data_root = catalog_path.parent
    if data_root.is_symlink():
        raise ValueError("installed operator package data root must not be a symlink")
    if catalog_path.is_symlink() or not catalog_path.is_file():
        raise ValueError(f"installed operator catalog is missing: {catalog_path}")
    catalog_bytes = _read_regular_file_bytes(catalog_path, "installed operator catalog")
    if hashlib.sha256(catalog_bytes).hexdigest() != expected_catalog_sha256:
        raise ValueError(
            "installed operator catalog checksum does not match build info"
        )
    try:
        raw_catalog = json.loads(catalog_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"installed operator catalog is unreadable: {error}"
        ) from error
    catalog = validate_generated_operator_catalog(raw_catalog)
    if len(catalog["artifacts"]) != expected_count:
        raise ValueError(
            "installed operator catalog artifact count does not match build info"
        )

    expected_members = {catalog_path}
    installed_bytes = len(catalog_bytes)
    for artifact in catalog["artifacts"]:
        identity = artifact["identity"]
        provenance = artifact["build_provenance"]
        if identity["backend"] != backend or identity["target"] != target:
            raise ValueError("installed operator catalog backend target mismatch")
        if (
            provenance["source_content_sha256"] != source_sha256
            or provenance["compiler_id"] != compiler_id
            or provenance["compiler_version"] != compiler_version
        ):
            raise ValueError("installed operator catalog provenance mismatch")
        relative = PurePosixPath(artifact["installed_relative_path"])
        binary_path = data_root.joinpath(*relative.parts)
        current = data_root
        for part in relative.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise ValueError(
                    "installed operator binary must use a contained real path"
                )
        try:
            binary_path.resolve().relative_to(data_root)
        except ValueError as error:
            raise ValueError(
                "installed operator binary must use a contained real path"
            ) from error
        expected_members.add(binary_path)
        if binary_path.is_symlink() or not binary_path.is_file():
            raise ValueError(f"installed operator binary is missing: {binary_path}")
        binary_bytes = _read_regular_file_bytes(
            binary_path, f"installed operator binary {binary_path}"
        )
        if "sha256:" + hashlib.sha256(binary_bytes).hexdigest() != artifact["checksum"]:
            raise ValueError(f"installed operator checksum mismatch: {binary_path}")
        installed_bytes += len(binary_bytes)
        for auxiliary in artifact.get("auxiliary_files", ()):
            auxiliary_relative = PurePosixPath(auxiliary["installed_relative_path"])
            auxiliary_path = data_root.joinpath(*auxiliary_relative.parts)
            current = data_root
            for part in auxiliary_relative.parts[:-1]:
                current /= part
                if current.is_symlink():
                    raise ValueError(
                        "installed operator auxiliary file must use a contained "
                        "real path"
                    )
            try:
                auxiliary_path.resolve().relative_to(data_root)
            except ValueError as error:
                raise ValueError(
                    "installed operator auxiliary file must use a contained real path"
                ) from error
            expected_members.add(auxiliary_path)
            if auxiliary_path.is_symlink() or not auxiliary_path.is_file():
                raise ValueError(
                    f"installed operator auxiliary file is missing: {auxiliary_path}"
                )
            auxiliary_bytes = _read_regular_file_bytes(
                auxiliary_path,
                f"installed operator auxiliary file {auxiliary_path}",
            )
            if (
                "sha256:" + hashlib.sha256(auxiliary_bytes).hexdigest()
                != auxiliary["checksum"]
            ):
                raise ValueError(
                    f"installed operator auxiliary checksum mismatch: {auxiliary_path}"
                )
            installed_bytes += len(auxiliary_bytes)

    actual_members = {path for path in data_root.rglob("*") if path.is_file()}
    if any(path.is_symlink() for path in data_root.rglob("*")):
        raise ValueError("installed operator package data must not contain symlinks")
    if actual_members != expected_members:
        raise ValueError("installed operator package data membership mismatch")
    if installed_bytes != expected_bytes:
        raise ValueError("installed operator package data byte count mismatch")
    return (
        {
            "status": "ok",
            "artifact_count": expected_count,
            "installed_bytes": installed_bytes,
            "catalog_sha256": expected_catalog_sha256,
            "catalog_path": str(catalog_path),
        },
        catalog,
        data_root,
    )


def validated_installed_operator_catalog(
    package_root: str | Path, build_info: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Return validated installed provenance, catalog, and immutable data root."""

    package_root = Path(package_root)
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError("installed operator package root must be a real directory")
    return _validate_installed_operator_artifacts_at_root(package_root, build_info)


def validate_installed_operator_artifacts(
    native_extension_path: str | Path, build_info: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate package-relative operator data against compiled provenance."""

    report, _, _ = validated_installed_operator_catalog(
        Path(native_extension_path).parent, build_info
    )
    return report


def generated_operator_origin_record(
    identity: Mapping[str, Any], origin: str
) -> dict[str, str]:
    """Report artifact origin without making origin part of its identity."""

    identity = validate_generated_r0_identity(identity)
    if origin not in ("prebuilt", "cache", "runtime"):
        raise ValueError("origin must be 'prebuilt', 'cache', or 'runtime'")
    return {"artifact_id": identity["artifact_id"], "origin": origin}


def render_direct_harmonic_gradient_helper(
    device_qualifier: str,
    edge_harmonics: int,
    *,
    index_type: str = "std::int32_t",
) -> str:
    """Render the shared real-spherical-harmonic gradient implementation."""

    if edge_harmonics not in (1, 4, 9, 16):
        raise ValueError("direct harmonic gradients require l_max from 0 through 3")
    return f"""{device_qualifier} void direct_harmonic_gradients(
    const void* coordinates,
    unsigned int coordinate_scalar_size,
    unsigned int coordinates_are_unit,
    std::size_t coordinate_offset,
    double radius,
    Scalar* gradients)
{{
    Scalar ux;
    Scalar uy;
    Scalar uz;
    if (coordinates_are_unit != 0) {{
        using CoordinateFloat = decltype(0.0f);
        if (coordinate_scalar_size == sizeof(CoordinateFloat)) {{
            const auto* values = static_cast<const CoordinateFloat*>(coordinates);
            ux = static_cast<Scalar>(values[coordinate_offset]);
            uy = static_cast<Scalar>(values[coordinate_offset + 1]);
            uz = static_cast<Scalar>(values[coordinate_offset + 2]);
        }} else {{
            const auto* values = static_cast<const double*>(coordinates);
            ux = static_cast<Scalar>(values[coordinate_offset]);
            uy = static_cast<Scalar>(values[coordinate_offset + 1]);
            uz = static_cast<Scalar>(values[coordinate_offset + 2]);
        }}
    }} else {{
        const auto* values = static_cast<const double*>(coordinates);
        const Scalar inverse_radius = Scalar(1)/static_cast<Scalar>(radius);
        ux = static_cast<Scalar>(values[coordinate_offset])*inverse_radius;
        uy = static_cast<Scalar>(values[coordinate_offset + 1])*inverse_radius;
        uz = static_cast<Scalar>(values[coordinate_offset + 2])*inverse_radius;
    }}
    const Scalar inverse_norm = Scalar(1)/sqrt(ux*ux + uy*uy + uz*uz);
    const Scalar x = uz*inverse_norm;
    const Scalar y = ux*inverse_norm;
    const Scalar z = uy*inverse_norm;
    const Scalar x2 = x*x;
    const Scalar y2 = y*y;
    const Scalar z2 = z*z;
    Scalar sph[16];
    Scalar dx[16];
    Scalar dy[16];
    Scalar dz[16];
    sph[0] = Scalar(0.282094791773878);
    sph[1] = Scalar(0.48860251190292)*y;
    sph[2] = Scalar(0.48860251190292)*z;
    sph[3] = Scalar(0.48860251190292)*x;
    const Scalar l2_tmp = Scalar(2.23606797749979)*x;
    sph[4] = l2_tmp*sph[1];
    sph[7] = l2_tmp*sph[2];
    sph[5] = Scalar(2.23606797749979)*z*sph[1];
    sph[6] = -Scalar(0.315391565252520)*(x2+y2-Scalar(2)*z2);
    sph[8] = Scalar(0.54627421529604)*(x2-y2);
    sph[9] = -Scalar(0.59004358992664)*y*(y2-Scalar(3)*x2);
    sph[10] = Scalar(2.64575131106459)*z*sph[4];
    const Scalar l3_tmp =
        -Scalar(0.457045799464466)*(x2+y2-Scalar(4)*z2);
    sph[11] = y*l3_tmp;
    sph[13] = x*l3_tmp;
    sph[12] = -Scalar(1.49270533036046)*z
        *(z2-Scalar(2.37799637856361)*sph[6]);
    sph[14] = Scalar(1.44530572132028)*z*(x2-y2);
    sph[15] = Scalar(0.59004358992664)*x*(x2-Scalar(3)*y2);
    dx[0] = dy[0] = dz[0] = Scalar(0);
    dx[1] = dx[2] = Scalar(0);
    dx[3] = Scalar(0.48860251190292);
    dy[1] = Scalar(0.48860251190292);
    dy[2] = dy[3] = Scalar(0);
    dz[1] = Scalar(0);
    dz[2] = Scalar(0.48860251190292);
    dz[3] = Scalar(0);
    dx[4] = Scalar(2.23606797749979)*sph[1];
    dx[5] = Scalar(0);
    dx[6] = -Scalar(1.29099444873581)*sph[3];
    dx[7] = Scalar(2.23606797749979)*sph[2];
    dx[8] = Scalar(2.23606797749979)*sph[3];
    dy[4] = -Scalar(1.73205080756888)*dx[6];
    dy[5] = dx[7];
    dy[6] = -Scalar(0.577350269189626)*dx[4];
    dy[7] = Scalar(0);
    dy[8] = -dx[4];
    dz[4] = dz[8] = Scalar(0);
    dz[5] = dx[4];
    dz[6] = Scalar(1.15470053837925)*dx[7];
    dz[7] = dy[4];
    dx[9] = Scalar(3.24037034920393)*sph[4];
    dx[10] = Scalar(2.64575131106459)*sph[5];
    dx[11] = -Scalar(0.83666002653408)*sph[4];
    dx[12] = -Scalar(2.04939015319192)*sph[7];
    dx[13] = Scalar(0.91409159892893)
        *(y2-z2+Scalar(4.75599275712721)*sph[6]);
    dx[14] = Scalar(2.64575131106459)*sph[7];
    dx[15] = Scalar(3.24037034920393)*sph[8];
    dy[9] = dx[15];
    dy[10] = dx[14];
    dy[11] = -Scalar(0.91409159892893)
        *(y2-z2-Scalar(1.58533091904240)*sph[6]);
    dy[12] = -Scalar(2.04939015319192)*sph[5];
    dy[13] = -Scalar(0.83666002653408)*sph[4];
    dy[14] = -dx[10];
    dy[15] = -dx[9];
    dz[9] = Scalar(0);
    dz[10] = Scalar(2.64575131106459)*sph[4];
    dz[11] = Scalar(3.34664010613630)*sph[5];
    dz[12] = Scalar(3.54964786985977)*sph[6];
    dz[13] = Scalar(3.34664010613630)*sph[7];
    dz[14] = Scalar(2.64575131106459)*sph[8];
    dz[15] = Scalar(0);
    const Scalar scale = Scalar(3.5449077018110318)
        *inverse_norm/static_cast<Scalar>(radius);
    for ({index_type} lm = 0; lm < {edge_harmonics}; ++lm) {{
        const Scalar radial = dx[lm]*x + dy[lm]*y + dz[lm]*z;
        const Scalar gx = (dx[lm]-x*radial)*scale;
        const Scalar gy = (dy[lm]-y*radial)*scale;
        const Scalar gz = (dz[lm]-z*radial)*scale;
        gradients[lm] = gy;
        gradients[{edge_harmonics}+lm] = gz;
        gradients[{2 * edge_harmonics}+lm] = gx;
    }}
}}"""


__all__ = [
    "GENERATED_OPERATOR_CATALOG_SCHEMA",
    "GENERATED_OPERATOR_IDENTITY_SCHEMA",
    "generated_m0_identity",
    "generated_operator_catalog",
    "generated_operator_origin_record",
    "generated_r0_identity",
    "generated_r1_identity",
    "render_direct_harmonic_gradient_helper",
    "validate_generated_m0_identity",
    "validate_generated_operator_catalog",
    "validate_generated_operator_identity",
    "validate_generated_r0_identity",
    "validate_generated_r1_identity",
    "validate_installed_operator_artifacts",
    "validated_installed_operator_catalog",
]
