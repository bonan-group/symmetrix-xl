"""Plan and render verified build-tree generated operator artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import numbers
import re
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _adapter_path = Path(__file__).with_name("generated_operator_codegen.py")
    _adapter_spec = importlib.util.spec_from_file_location(
        "_symmetrix_generated_operator_build_adapter", _adapter_path
    )
    if _adapter_spec is None or _adapter_spec.loader is None:
        raise ImportError(f"cannot load generated operator adapter {_adapter_path}")
    _adapter = importlib.util.module_from_spec(_adapter_spec)
    _adapter_spec.loader.exec_module(_adapter)
    load_build_source_module = _adapter.load_build_source_module
    generated_r1_metadata = _adapter.generated_r1_metadata
    render_generated_m0_operator = _adapter.render_generated_m0_operator
    render_generated_m1_operator = _adapter.render_generated_m1_operator
    render_generated_r0_operator = _adapter.render_generated_r0_operator
    render_generated_r1_operator = _adapter.render_generated_r1_operator
else:
    from .generated_operator_codegen import (
        generated_r1_metadata,
        load_build_source_module,
        render_generated_m0_operator,
        render_generated_m1_operator,
        render_generated_r0_operator,
        render_generated_r1_operator,
    )


_SOURCE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODES = ("off", "r0", "full")
_OAM_MODES = ("off", "known")
_BACKENDS = ("host", "cuda", "hip")
OAM_RELEASE_ARTIFACT_COUNT = 8
OAM_RELEASE_INSTALLED_BYTES_MAX = 128_000_000
OAM_RELEASE_BUILD_SECONDS_MAX = 1_800.0
OAM_RELEASE_WHEEL_BYTES_MAX = 256_000_000
OAM_RELEASE_ARCHITECTURE_FINGERPRINTS = {
    "oam-l-v0.1": "sha256:8e8b6c963d1be1449a43b0c02c6b80bdbd1b83070fb7abff5f6c806a0efa670f",
    "oam-m-v0.1": "sha256:d8b773a86ebec83d1ea4dcbc3babe72242aebac3c88df245d6afbfc8352ced79",
    "oam-s-v0.1": "sha256:676b385abb23c436411e3d9cb6032a7030b7141d4dd42db5b2e2f7104d93a307",
    "oam-xl-v0.1": "sha256:4ad5f8500b618300493547dca76321e8d9c870988239228e071d91bd9cc40c12",
}


def _canonical_oam_host_target(target: str) -> str:
    return load_build_source_module("jit").canonical_host_artifact_target(target)


def effective_prebuilt_host_target(
    *, declared_target: str, host_arch: str, compile_flags: list[str] | tuple[str, ...]
) -> str:
    """Derive host artifact metadata from the flags used for compilation."""

    jit = load_build_source_module("jit")
    supported_arches = ("native", "x86-64", "x86-64-v2", "x86-64-v3", "x86-64-v4")
    if host_arch not in supported_arches:
        raise ValueError(
            "host prebuilt artifacts require SYMMETRIX_HOST_ARCH to be native, "
            "x86-64, x86-64-v2, x86-64-v3, or x86-64-v4"
        )
    declared = jit.canonical_host_artifact_target(declared_target)
    architecture = jit.canonical_host_artifact_target(host_arch)
    if declared != architecture:
        raise ValueError(
            "prebuilt operator target does not match effective host compile target: "
            f"declared {declared_target!r}, SYMMETRIX_HOST_ARCH {host_arch!r}"
        )
    return jit._host_jit_effective_target(
        (*tuple(str(flag) for flag in compile_flags), f"-march={host_arch}")
    )


def _build_modules():
    return (
        load_build_source_module("foundation_operator_manifest"),
        load_build_source_module("generated_operator_version"),
    )


def _catalog_module():
    return load_build_source_module("generated_operator_core")


def _oam_modules():
    return (
        load_build_source_module("oam_architecture"),
        load_build_source_module("oam_receiver_codegen"),
        load_build_source_module("oam_cuda_codegen"),
        load_build_source_module("generated_operator_version"),
    )


def _load_oam_manifest(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "symmetrix.foundation-oam-architectures/1":
        raise ValueError("foundation OAM architecture manifest schema is unsupported")
    architectures = value.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("foundation OAM architecture manifest must not be empty")
    architecture_module, _, _, _ = _oam_modules()
    identifiers = []
    normalized = []
    for entry in architectures:
        if not isinstance(entry, dict):
            raise ValueError("foundation OAM architecture entry must be an object")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("foundation OAM architecture id must not be empty")
        architecture = architecture_module.validate_oam_architecture(
            entry.get("architecture")
        )
        source_fingerprint = architecture_module.oam_architecture_fingerprint(
            architecture
        )
        if entry.get("architecture_fingerprint") != source_fingerprint:
            raise ValueError(
                f"foundation OAM architecture fingerprint mismatch for {identifier}"
            )
        artifact_fingerprint = (
            architecture_module.oam_tensor_product_architecture_fingerprint(
                architecture
            )
        )
        source_sha256 = entry.get("source_contract_sha256")
        _complete_sha256(source_sha256, "source_contract_sha256")
        identifiers.append(identifier)
        normalized.append(
            {
                "id": identifier,
                "source_contract_sha256": source_sha256,
                "architecture_fingerprint": artifact_fingerprint,
                "architecture": architecture,
            }
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("foundation OAM architecture ids must be unique")
    return {"schema": value["schema"], "architectures": normalized}


def _oam_generated_metadata(
    architecture: dict[str, Any],
    backend: str,
    target: str,
    precision: str,
    floating_point_policy: str | None = None,
) -> tuple[dict[str, Any], str]:
    architecture_module, receiver, cuda, _ = _oam_modules()
    contract = architecture_module.oam_codegen_contract(architecture)
    if backend == "host":
        target = _canonical_oam_host_target(target)
        metadata = receiver.oam_receiver_jit_metadata(
            contract,
            precision,
            architecture=architecture,
            host_target=target,
            floating_point_policy=floating_point_policy or "strict",
        )
        source = receiver.render_oam_receiver_host_plugin(
            contract,
            precision,
            architecture=architecture,
            host_target=target,
            floating_point_policy=floating_point_policy or "strict",
        )
    elif backend == "cuda":
        metadata = cuda.oam_cuda_artifact_metadata(
            contract,
            precision=precision,
            target=target,
            architecture=architecture,
        )
        source = cuda.render_oam_cuda_plugin(
            contract,
            target,
            precision=precision,
            architecture=architecture,
        )
    else:
        raise ValueError("known OAM artifacts support host and CUDA backends only")
    return metadata, source


def _oam_identity(
    metadata: dict[str, Any], *, backend: str, target: str, generation_revision: int
) -> dict[str, Any]:
    architecture = metadata["architecture"]
    architecture_module, _, _, _ = _oam_modules()
    identity = {
        "schema": "symmetrix.generated-operator-identity/1",
        "operator": "OAM",
        "generation_revision": generation_revision,
        "architecture": architecture,
        "architecture_fingerprint": (
            architecture_module.oam_tensor_product_architecture_fingerprint(
                architecture
            )
        ),
        "precision": metadata["precision"],
        "backend": backend,
        "target": metadata.get("host_target", target),
        "generation_fingerprint": metadata["generation_fingerprint"],
        "semantic_fingerprint": metadata["semantic_fingerprint"],
        "structure_fingerprint": metadata["structure_fingerprint"],
        "artifact_id": metadata["artifact_id"],
    }
    if backend == "host":
        identity["floating_point_policy"] = metadata["floating_point_policy"]
    return identity


def _r0_schedule_revision(version, backend: str) -> int:
    return (
        version.R0_HOST_SCHEDULE_REVISION
        if backend == "host"
        else version.R0_DEVICE_SCHEDULE_REVISION
    )


def validate_generated_operator_catalog(catalog):
    """Expose canonical validation through the isolated build adapter."""

    return _catalog_module().validate_generated_operator_catalog(catalog)


def sha256_file(path: Path | str) -> str:
    """Return the canonical fingerprint spelling for one exact build output."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _complete_sha256(value: str, name: str) -> str:
    if _SOURCE_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _write_provenance_header(path: Path, provenance: dict[str, Any]) -> None:
    fields = (
        "mode",
        "backend",
        "target",
        "catalog_relative_path",
        "catalog_sha256",
        "source_content_sha256",
        "native_source_content_sha256",
        "compiler_id",
        "compiler_version",
    )
    declarations = [
        f"inline constexpr const char* prebuilt_operator_{field} = "
        f"{_cpp_string(str(provenance[field]))};"
        for field in fields
    ]
    declarations.extend(
        (
            "inline constexpr unsigned long long prebuilt_operator_artifact_count = "
            f"{provenance['artifact_count']}ULL;",
            "inline constexpr unsigned long long prebuilt_operator_installed_bytes = "
            f"{provenance['installed_bytes']}ULL;",
            "inline constexpr double prebuilt_operator_build_duration_seconds = "
            f"{provenance['build_duration_seconds']:.9f};",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#pragma once\n\nnamespace symmetrix::build {\n"
        + "\n".join(f"{declaration}" for declaration in declarations)
        + "\n}\n",
        encoding="utf-8",
    )


def _artifact_build_duration_seconds(
    output_and_start_times: tuple[tuple[int, int], ...],
    *,
    previous_catalog_mtime_ns: int | None,
) -> float:
    """Measure the current build/relink batch, excluding stale artifact starts."""

    selected = [
        (output_mtime_ns, started_ns)
        for output_mtime_ns, started_ns in output_and_start_times
        if previous_catalog_mtime_ns is None
        or output_mtime_ns > previous_catalog_mtime_ns
    ]
    if not selected:
        return 0.0
    intervals = sorted(
        (min(started_ns, output_mtime_ns), output_mtime_ns)
        for output_mtime_ns, started_ns in selected
    )
    current_start, current_finish = intervals[0]
    duration_ns = 0
    for started_ns, finished_ns in intervals[1:]:
        if started_ns <= current_finish:
            current_finish = max(current_finish, finished_ns)
            continue
        duration_ns += current_finish - current_start
        current_start, current_finish = started_ns, finished_ns
    duration_ns += current_finish - current_start
    return duration_ns / 1_000_000_000


def _read_build_start_timestamp(path: Path) -> int:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(
            f"artifact build start timestamp is invalid: {path}"
        ) from error
    if value < 0:
        raise ValueError(f"artifact build start timestamp is invalid: {path}")
    return value


def write_build_start_timestamp(
    output: Path | str, *, preserve_if_newer_than: Path | str | None = None
) -> None:
    """Record a build start, preserving a newer generation-stage timestamp."""

    output = Path(output)
    reference = (
        Path(preserve_if_newer_than) if preserve_if_newer_than is not None else None
    )
    if (
        reference is not None
        and reference.is_file()
        and output.is_file()
        and output.stat().st_mtime_ns > reference.stat().st_mtime_ns
    ):
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{time.time_ns()}\n", encoding="ascii")


def write_installed_operator_catalog(
    *,
    artifacts: list[tuple[Path | str, ...]],
    catalog_path: Path | str,
    header_path: Path | str,
    mode: str,
    backend: str,
    target: str,
    source_content_sha256: str,
    native_source_content_sha256: str,
    compiler_id: str,
    compiler_version: str,
) -> dict[str, Any]:
    """Create a canonical installed catalog from exact compiled outputs."""

    if mode not in ("r0", "full", "oam"):
        raise ValueError("installed artifact mode must be 'r0', 'full', or 'oam'")
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(_BACKENDS)}")
    if not target:
        raise ValueError("target must not be empty")
    if backend == "host":
        target = _canonical_oam_host_target(target)
    source_content_sha256 = _complete_sha256(
        source_content_sha256, "source_content_sha256"
    )
    native_source_content_sha256 = _complete_sha256(
        native_source_content_sha256, "native_source_content_sha256"
    )
    if not compiler_id or not compiler_version:
        raise ValueError("compiler identity must be complete")
    core = _catalog_module()
    entries = []
    binary_bytes = 0
    output_and_start_times = []
    catalog_path = Path(catalog_path)
    previous_catalog_mtime_ns = (
        catalog_path.stat().st_mtime_ns if catalog_path.is_file() else None
    )
    for artifact_values in artifacts:
        if len(artifact_values) not in (4, 6):
            raise ValueError("artifact record must contain 4 or 6 values")
        metadata_value, binary_value, installed_relative_path, started_value = (
            artifact_values[:4]
        )
        auxiliary_value = artifact_values[4] if len(artifact_values) == 6 else None
        auxiliary_installed_path = (
            artifact_values[5] if len(artifact_values) == 6 else None
        )
        # Reject unsafe paths before opening caller-controlled metadata so path
        # diagnostics remain precise and no absolute build path can leak out.
        installed_relative_path = core._installed_relative_path(installed_relative_path)
        metadata_path = Path(metadata_value)
        binary_path = Path(binary_value)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read artifact metadata {metadata_path}: {error}"
            ) from error
        if not binary_path.is_file():
            raise ValueError(f"compiled artifact is unavailable: {binary_path}")
        output_and_start_times.append(
            (
                binary_path.stat().st_mtime_ns,
                _read_build_start_timestamp(Path(started_value)),
            )
        )
        identity = metadata.get("generated_identity")
        if not isinstance(identity, dict):
            raise ValueError(
                f"artifact metadata has no generated identity: {metadata_path}"
            )
        if identity.get("backend") != backend or identity.get("target") != target:
            raise ValueError(
                "artifact metadata backend target does not match the catalog"
            )
        build = metadata.get("build_provenance")
        if not isinstance(build, dict):
            raise ValueError(
                f"artifact metadata has no build provenance: {metadata_path}"
            )
        if (
            build.get("source_content_sha256") != source_content_sha256
            or build.get("compiler_id") != compiler_id
            or build.get("compiler_version") != compiler_version
            or (
                identity.get("operator") in ("R0", "OAM")
                and "floating_point_policy" in identity
                and build.get("floating_point_policy")
                != identity.get("floating_point_policy")
            )
        ):
            raise ValueError(
                "artifact metadata build provenance does not match the catalog"
            )
        entry = {
            "operator": metadata.get("operator"),
            "identity": identity,
            "binary_format": metadata.get("binary_format"),
            "installed_relative_path": installed_relative_path,
            "checksum": sha256_file(binary_path),
            "abi": {
                "tag": metadata.get("schema"),
                "version": metadata.get("abi_version"),
            },
            "capabilities": metadata.get("capabilities"),
            "build_provenance": {
                "source_content_sha256": "sha256:" + source_content_sha256,
                "compiler_id": compiler_id,
                "compiler_version": compiler_version,
                **(
                    {"floating_point_policy": identity["floating_point_policy"]}
                    if "floating_point_policy" in identity
                    else {}
                ),
            },
        }
        if auxiliary_value is not None:
            auxiliary_path = Path(auxiliary_value)
            if not auxiliary_path.is_file():
                raise ValueError(
                    f"artifact auxiliary file is unavailable: {auxiliary_path}"
                )
            entry["auxiliary_files"] = [
                {
                    "role": "manifest",
                    "installed_relative_path": str(auxiliary_installed_path),
                    "checksum": sha256_file(auxiliary_path),
                }
            ]
            binary_bytes += auxiliary_path.stat().st_size
        entries.append(entry)
        binary_bytes += binary_path.stat().st_size

    catalog = core.generated_operator_catalog(entries)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    duration = _artifact_build_duration_seconds(
        tuple(output_and_start_times),
        previous_catalog_mtime_ns=previous_catalog_mtime_ns,
    )
    provenance = {
        "mode": mode,
        "backend": backend,
        "target": target,
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": sha256_file(catalog_path).removeprefix("sha256:"),
        "artifact_count": len(catalog["artifacts"]),
        "installed_bytes": binary_bytes + catalog_path.stat().st_size,
        "source_content_sha256": source_content_sha256,
        "native_source_content_sha256": native_source_content_sha256,
        "compiler_id": compiler_id,
        "compiler_version": compiler_version,
        "build_duration_seconds": duration,
    }
    _write_provenance_header(Path(header_path), provenance)
    return provenance


def plan_foundation_operator_artifacts(
    *,
    manifest_path: Path | str,
    mode: str,
    backend: str,
    target: str,
    source_content_sha256: str,
    floating_point_policy: str | None = None,
    oam_manifest_path: Path | str | None = None,
    oam_mode: str = "off",
    oam_floating_point_policy: str | None = None,
) -> dict[str, Any]:
    """Return the deterministic build plan for one qualified target."""

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {', '.join(_MODES)}")
    if oam_mode not in _OAM_MODES:
        raise ValueError(f"oam_mode must be one of {', '.join(_OAM_MODES)}")
    if mode == "off" and oam_mode == "off":
        return {"mode": "off", "artifacts": []}
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(_BACKENDS)}")
    if not target:
        raise ValueError("target must not be empty")
    if backend == "host":
        target = _canonical_oam_host_target(target)
    if _SOURCE_SHA256.fullmatch(source_content_sha256) is None:
        raise ValueError("source_content_sha256 must be a SHA-256 digest")

    foundation, version = _build_modules()
    manifest = foundation.load_foundation_operator_manifest(manifest_path)
    schedule = "host-owner" if backend == "host" else "receiver-edge16"
    r0_requests = foundation.foundation_r0_artifact_inputs(
        manifest,
        generation_revision=version.JIT_GENERATION_VERSION,
        compatibility_version=2,
        backend=backend,
        target=target,
        precisions=("float32", "float64"),
        schedule=schedule,
        schedule_revision=_r0_schedule_revision(version, backend),
        floating_point_policy=floating_point_policy,
    )
    binary_format = {
        "host": "shared-library",
        "cuda": "cubin",
        "hip": "code-object",
    }[backend]
    extension = {"host": "cpp", "cuda": "cu", "hip": "hip"}[backend]
    requests: list[tuple[str, dict[str, Any]]] = (
        [("R0", request) for request in r0_requests] if mode != "off" else []
    )
    if mode == "full":
        m0_requests = foundation.foundation_m0_artifact_inputs(
            manifest,
            generation_revision=version.JIT_GENERATION_VERSION,
            backend=backend,
            target=target,
            precisions=("float32", "float64"),
        )
        requests.extend(("M0", request) for request in m0_requests)
        m1_requests = foundation.foundation_m1_artifact_inputs(
            manifest,
            generation_revision=version.JIT_GENERATION_VERSION,
            backend=backend,
            target=target,
            precisions=("float32", "float64"),
        )
        requests.extend(("M1", request) for request in m1_requests)
        r1_requests: dict[str, dict[str, Any]] = {}
        for bundle in manifest["bundles"]:
            contract = bundle["operators"]["R1"]["contract"]
            for precision in ("float32", "float64"):
                metadata = generated_r1_metadata(
                    contract,
                    backend=backend,
                    target=target,
                    precision=precision,
                )
                identity = metadata["generated_identity"]
                request = r1_requests.setdefault(
                    identity["artifact_id"],
                    {"identity": identity, "contract": contract, "bundle_ids": set()},
                )
                request["bundle_ids"].add(bundle["id"])
        requests.extend(
            (
                "R1",
                {
                    **r1_requests[key],
                    "bundle_ids": sorted(r1_requests[key]["bundle_ids"]),
                },
            )
            for key in sorted(r1_requests)
        )

    artifacts = []
    for operator, request in requests:
        identity = request["identity"]
        precision = identity["precision"]
        input_ids = request.get("artifact_input_ids", ())
        input_id = input_ids[0] if input_ids else None
        bundle_ids = request["bundle_ids"]
        stem_owner = input_id or bundle_ids[0]
        stem = (
            f"{stem_owner}-{precision}"
            if operator == "R0"
            else f"{operator.lower()}-{stem_owner}-{precision}"
        )
        artifacts.append(
            {
                "artifact_id": identity["artifact_id"],
                "artifact_input_id": input_id,
                "backend": backend,
                "binary_format": binary_format,
                "bundle_ids": bundle_ids,
                "channels": identity.get("channels"),
                "operator": operator,
                "precision": precision,
                "schedule": identity["schedule"],
                "source_extension": extension,
                "stem": stem,
                "target": target,
            }
        )
    if oam_mode == "known":
        if backend == "hip":
            raise ValueError("known OAM artifacts are not available for HIP")
        if oam_manifest_path is None:
            raise ValueError("known OAM artifacts require an architecture manifest")
        oam_manifest = _load_oam_manifest(oam_manifest_path)
        for entry in oam_manifest["architectures"]:
            for precision in ("float32", "float64"):
                metadata, _ = _oam_generated_metadata(
                    entry["architecture"],
                    backend,
                    target,
                    precision,
                    oam_floating_point_policy,
                )
                artifacts.append(
                    {
                        "artifact_id": metadata["artifact_id"],
                        "architecture_id": entry["id"],
                        "architecture_fingerprint": entry["architecture_fingerprint"],
                        "backend": backend,
                        "binary_format": binary_format,
                        "bundle_ids": [entry["id"]],
                        "channels": None,
                        "operator": "OAM",
                        "precision": precision,
                        "capabilities": metadata["required_capabilities"],
                        "variants": ["full-message"],
                        "schedule": "generated-architecture",
                        "source_extension": extension,
                        "stem": f"oam-{entry['id']}-{precision}",
                        "target": target,
                        **(
                            {"floating_point_policy": metadata["floating_point_policy"]}
                            if backend == "host"
                            else {}
                        ),
                    }
                )
    artifacts.sort(key=lambda value: value["stem"])
    for index, artifact in enumerate(artifacts):
        # Two dependency lanes bound artifact compilation without starting a
        # nested worker pool, so the outer build tool retains job ownership.
        artifact["build_lane"] = index % 2
    return {"mode": mode, "oam_mode": oam_mode, "artifacts": artifacts}


def audit_oam_release_plan(
    plan: dict[str, Any], *, backend: str, target: str
) -> dict[str, Any]:
    """Validate the minimal full-message-only OAM release artifact plan."""

    if backend not in ("host", "cuda"):
        raise ValueError("OAM release artifacts support host and CUDA only")
    expected_target = (
        _canonical_oam_host_target(target) if backend == "host" else target
    )
    artifacts = [
        artifact
        for artifact in plan.get("artifacts", ())
        if artifact.get("operator") == "OAM"
    ]
    if len(artifacts) != OAM_RELEASE_ARTIFACT_COUNT:
        raise ValueError("release plan must contain exactly 8 OAM artifacts")
    keys = {
        (artifact.get("architecture_id"), artifact.get("precision"))
        for artifact in artifacts
    }
    expected_keys = {
        (f"oam-{size}-v0.1", precision)
        for size in ("s", "m", "l", "xl")
        for precision in ("float32", "float64")
    }
    if keys != expected_keys:
        raise ValueError(
            "release plan must contain one artifact per OAM architecture/precision"
        )
    fingerprints = {
        (artifact.get("architecture_id"), artifact.get("architecture_fingerprint"))
        for artifact in artifacts
    }
    if fingerprints != set(OAM_RELEASE_ARCHITECTURE_FINGERPRINTS.items()):
        raise ValueError("release plan must contain only qualified OAM architectures")
    for artifact in artifacts:
        if (
            artifact.get("backend") != backend
            or artifact.get("target") != expected_target
        ):
            raise ValueError("release OAM artifact backend or target mismatch")
        if artifact.get("variants") != ["full-message"]:
            raise ValueError("release OAM artifacts must contain only full-message")
        capabilities = artifact.get("capabilities")
        if isinstance(capabilities, bool) or not isinstance(capabilities, int):
            raise ValueError("release OAM artifact capabilities are invalid")
    return {
        "artifact_count": len(artifacts),
        "architectures": sorted({key[0] for key in keys}),
        "precisions": sorted({key[1] for key in keys}),
        "variants": ["full-message"],
    }


def audit_oam_release_budgets(measurement: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when an installed release catalog exceeds packaging budgets."""

    limits = {
        "oam_artifact_count": OAM_RELEASE_ARTIFACT_COUNT,
        "installed_bytes": OAM_RELEASE_INSTALLED_BYTES_MAX,
        "build_duration_seconds": OAM_RELEASE_BUILD_SECONDS_MAX,
        "wheel_bytes": OAM_RELEASE_WHEEL_BYTES_MAX,
    }
    for field in limits:
        value = measurement.get(field)
        valid_number = (
            not isinstance(value, bool)
            and isinstance(value, numbers.Real)
            and math.isfinite(float(value))
            and value >= 0
        )
        integer_field = field in (
            "oam_artifact_count",
            "installed_bytes",
            "wheel_bytes",
        )
        if not valid_number or (integer_field and not isinstance(value, int)):
            raise ValueError(
                f"release measurement {field} must be a finite nonnegative "
                + ("integer" if integer_field else "real number")
            )
    if measurement["oam_artifact_count"] != limits["oam_artifact_count"]:
        raise ValueError("OAM artifact count must be exactly 8")
    if measurement["installed_bytes"] > limits["installed_bytes"]:
        raise ValueError("installed-byte budget exceeded")
    if measurement["build_duration_seconds"] > limits["build_duration_seconds"]:
        raise ValueError("build-time budget exceeded")
    if measurement["wheel_bytes"] > limits["wheel_bytes"]:
        raise ValueError("wheel-size budget exceeded")
    return dict(measurement)


def audit_installed_oam_release_catalog(
    *,
    native_extension_path: Path | str,
    build_info: dict[str, Any],
    backend_descriptor: dict[str, Any],
    native_build_info: dict[str, Any],
    reported_validation: dict[str, Any],
    wheel_bytes: int,
    backend: str,
    target: str,
) -> dict[str, Any]:
    """Audit a repaired wheel's installed OAM catalog and packaging budgets."""

    if backend not in ("host", "cuda"):
        raise ValueError("OAM release catalogs support host and CUDA only")
    extension = Path(native_extension_path)
    if extension.is_symlink() or not extension.is_file():
        raise ValueError("installed native extension must be a regular file")
    package_root = extension.parent
    validation, catalog, _ = _catalog_module().validated_installed_operator_catalog(
        package_root, build_info
    )
    if reported_validation != validation:
        raise ValueError("probe installed-package validation is stale or inconsistent")
    if native_build_info.get("prebuilt_operator_artifacts") != build_info:
        raise ValueError("probe native build info does not bind the audited catalog")
    entries = catalog["artifacts"]
    expected_target = (
        _canonical_oam_host_target(target)
        if backend == "host"
        else load_build_source_module(
            "cuda_codegen_common"
        ).canonical_cuda_target_identity(target)
    )
    build_target = build_info.get("target")
    canonical_build_target = (
        _canonical_oam_host_target(build_target)
        if backend == "host"
        else load_build_source_module(
            "cuda_codegen_common"
        ).canonical_cuda_target_identity(build_target)
    )
    if (
        build_info.get("backend") != backend
        or canonical_build_target != expected_target
    ):
        raise ValueError("installed release backend or target mismatch")
    descriptor_target = backend_descriptor.get("architecture")
    canonical_descriptor_target = (
        _canonical_oam_host_target(descriptor_target)
        if backend == "host"
        else load_build_source_module(
            "cuda_codegen_common"
        ).canonical_cuda_target_identity(descriptor_target)
    )
    if (
        backend_descriptor.get("backend") != backend
        or canonical_descriptor_target != expected_target
    ):
        raise ValueError("probe backend descriptor does not bind the requested target")
    commit = native_build_info.get("source_commit")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or backend_descriptor.get("source_commit") != commit
        or native_build_info.get("source_dirty") is not False
        or backend_descriptor.get("source_dirty") is not False
    ):
        raise ValueError("probe/backend provenance is missing, dirty, or inconsistent")
    for field in ("source_content_sha256", "native_source_content_sha256"):
        value = build_info.get(field)
        if (
            not isinstance(value, str)
            or _SOURCE_SHA256.fullmatch(value) is None
            or native_build_info.get(field) != value
            or backend_descriptor.get(field) != value
        ):
            raise ValueError(f"probe/backend provenance {field} is inconsistent")
    oam_entries = [entry for entry in entries if entry.get("operator") == "OAM"]
    if len(oam_entries) != OAM_RELEASE_ARTIFACT_COUNT:
        raise ValueError("installed release must contain exactly 8 OAM artifacts")
    expected_host_policy = None
    if backend == "host":
        native_fast_math = native_build_info.get("fast_math")
        if not isinstance(native_fast_math, bool):
            raise ValueError(
                "probe native build info must bind host floating-point policy"
            )
        expected_host_policy = "fast" if native_fast_math else "strict"
    architecture_precisions: dict[str, set[str]] = {}
    installed_paths: list[str] = []
    generation_revision = _oam_modules()[3].JIT_GENERATION_VERSION
    for entry in oam_entries:
        identity = entry["identity"]
        if identity.get("generation_revision") != generation_revision:
            raise ValueError("installed OAM generation revision is stale")
        entry_target = identity.get("target")
        canonical_entry_target = (
            _canonical_oam_host_target(entry_target)
            if backend == "host"
            else load_build_source_module(
                "cuda_codegen_common"
            ).canonical_cuda_target_identity(entry_target)
        )
        if (
            identity.get("backend") != backend
            or canonical_entry_target != expected_target
        ):
            raise ValueError("installed OAM artifact backend or target mismatch")
        metadata, _ = _oam_generated_metadata(
            identity["architecture"],
            backend,
            expected_target,
            identity["precision"],
            identity.get("floating_point_policy"),
        )
        expected_identity = _oam_identity(
            metadata,
            backend=backend,
            target=expected_target,
            generation_revision=generation_revision,
        )
        if identity != expected_identity:
            raise ValueError("installed OAM artifact identity is not canonical")
        if entry.get("capabilities") != metadata["required_capabilities"]:
            raise ValueError("installed OAM artifact lacks exact dual capabilities")
        provenance = entry["build_provenance"]
        if (
            provenance.get("source_content_sha256")
            != "sha256:" + build_info["source_content_sha256"]
        ):
            raise ValueError("installed OAM artifact source provenance mismatch")
        if backend == "host" and provenance.get(
            "floating_point_policy"
        ) != identity.get("floating_point_policy"):
            raise ValueError("installed OAM floating-point policy mismatch")
        if (
            backend == "host"
            and identity.get("floating_point_policy") != expected_host_policy
        ):
            raise ValueError(
                "installed OAM floating-point policy does not match native fast_math"
            )
        installed_paths.append(entry["installed_relative_path"])
        installed_paths.extend(
            auxiliary["installed_relative_path"]
            for auxiliary in entry.get("auxiliary_files", ())
        )
        architecture_key = identity.get("architecture_fingerprint")
        architecture_precisions.setdefault(architecture_key, set()).add(
            identity.get("precision")
        )
    if len(installed_paths) != len(set(installed_paths)):
        raise ValueError("installed OAM artifact paths must be unique")
    if set(architecture_precisions) != set(
        OAM_RELEASE_ARCHITECTURE_FINGERPRINTS.values()
    ):
        raise ValueError(
            "installed OAM release must contain only qualified OAM architectures"
        )
    if any(
        precisions != {"float32", "float64"}
        for precisions in architecture_precisions.values()
    ):
        raise ValueError(
            "installed OAM release must contain four architectures in both precisions"
        )
    measurement = audit_oam_release_budgets(
        {
            "oam_artifact_count": len(oam_entries),
            "installed_bytes": build_info.get("installed_bytes"),
            "build_duration_seconds": build_info.get("build_duration_seconds"),
            "wheel_bytes": wheel_bytes,
        }
    )
    return {
        **measurement,
        "backend": backend,
        "target": expected_target,
        "catalog_sha256": validation["catalog_sha256"],
    }


def _foundation_requests(
    *,
    manifest: dict[str, Any],
    backend: str,
    target: str,
    precision: str,
    floating_point_policy: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    if backend == "host":
        target = _canonical_oam_host_target(target)
    foundation, version = _build_modules()
    r0_schedule = "host-owner" if backend == "host" else "receiver-edge16"
    requests: list[tuple[str, dict[str, Any]]] = [
        ("R0", request)
        for request in foundation.foundation_r0_artifact_inputs(
            manifest,
            generation_revision=version.JIT_GENERATION_VERSION,
            compatibility_version=2,
            backend=backend,
            target=target,
            precisions=(precision,),
            schedule=r0_schedule,
            schedule_revision=_r0_schedule_revision(version, backend),
            floating_point_policy=floating_point_policy,
        )
    ]
    requests.extend(
        ("M0", request)
        for request in foundation.foundation_m0_artifact_inputs(
            manifest,
            generation_revision=version.JIT_GENERATION_VERSION,
            backend=backend,
            target=target,
            precisions=(precision,),
        )
    )
    requests.extend(
        ("M1", request)
        for request in foundation.foundation_m1_artifact_inputs(
            manifest,
            generation_revision=version.JIT_GENERATION_VERSION,
            backend=backend,
            target=target,
            precisions=(precision,),
        )
    )
    for bundle in manifest["bundles"]:
        contract = bundle["operators"]["R1"]["contract"]
        metadata = generated_r1_metadata(
            contract, backend=backend, target=target, precision=precision
        )
        requests.append(
            (
                "R1",
                {
                    "identity": metadata["generated_identity"],
                    "contract": contract,
                    "bundle_ids": [bundle["id"]],
                },
            )
        )
    return requests


def render_foundation_operator_artifact(
    *,
    manifest_path: Path | str,
    operator: str,
    artifact_id: str,
    backend: str,
    target: str,
    precision: str,
    source_content_sha256: str,
    compiler_id: str,
    compiler_version: str,
    source_path: Path | str,
    metadata_path: Path | str,
    floating_point_policy: str | None = None,
) -> None:
    """Render one exact verified R1, R0, or M0 artifact."""

    if backend == "host":
        target = _canonical_oam_host_target(target)
    foundation, version = _build_modules()
    manifest = foundation.load_foundation_operator_manifest(manifest_path)
    selected = next(
        (
            request
            for request_operator, request in _foundation_requests(
                manifest=manifest,
                backend=backend,
                target=target,
                precision=precision,
                floating_point_policy=floating_point_policy,
            )
            if request_operator == operator
            and request["identity"]["artifact_id"] == artifact_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            f"the production manifest has no {operator} artifact {artifact_id}"
        )
    if operator == "R0":
        source, metadata = render_generated_r0_operator(
            selected["contract"],
            generation_version=version.JIT_GENERATION_VERSION,
            source_content_sha256=source_content_sha256,
            backend=backend,
            target=target,
            precision=precision,
            schedule=selected["identity"]["schedule"],
            floating_point_policy=selected["identity"]["floating_point_policy"],
        )
    elif operator == "M0":
        source, metadata = render_generated_m0_operator(
            selected["contract"],
            generation_version=version.JIT_GENERATION_VERSION,
            backend=backend,
            target=target,
            precision=precision,
            schedule=selected["identity"]["schedule"],
        )
    elif operator == "M1":
        source, metadata = render_generated_m1_operator(
            selected["contract"],
            generation_version=version.JIT_GENERATION_VERSION,
            backend=backend,
            target=target,
            precision=precision,
            schedule=selected["identity"]["schedule"],
        )
    elif operator == "R1":
        source, metadata = render_generated_r1_operator(
            selected["contract"],
            backend=backend,
            target=target,
            precision=precision,
        )
    else:
        raise ValueError("operator must be R1, R0, M0, or M1")
    if metadata["generated_identity"] != selected["identity"]:
        raise RuntimeError("planned and rendered artifact identities disagree")
    metadata = {
        **metadata,
        "artifact_input_ids": selected.get("artifact_input_ids", []),
        "bundle_ids": selected["bundle_ids"],
        "binary_format": {
            "host": "shared-library",
            "cuda": "cubin",
            "hip": "code-object",
        }[backend],
        "build_provenance": {
            "source_content_sha256": source_content_sha256,
            "compiler_id": compiler_id,
            "compiler_version": compiler_version,
            **(
                {"floating_point_policy": selected["identity"]["floating_point_policy"]}
                if operator == "R0"
                else {}
            ),
        },
    }
    source_path = Path(source_path)
    metadata_path = Path(metadata_path)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_foundation_r0_artifact(
    *,
    manifest_path: Path | str,
    artifact_input_id: str,
    backend: str,
    target: str,
    precision: str,
    source_content_sha256: str,
    compiler_id: str,
    compiler_version: str,
    source_path: Path | str,
    metadata_path: Path | str,
    floating_point_policy: str | None = None,
) -> None:
    """Render one exact verified R0 source and its build metadata."""

    if backend == "host":
        target = _canonical_oam_host_target(target)

    plan = plan_foundation_operator_artifacts(
        manifest_path=manifest_path,
        mode="r0",
        backend=backend,
        target=target,
        source_content_sha256=source_content_sha256,
        floating_point_policy=floating_point_policy,
    )
    selected = next(
        (
            artifact
            for artifact in plan["artifacts"]
            if artifact["artifact_input_id"] == artifact_input_id
            and artifact["precision"] == precision
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            f"the production manifest has no {artifact_input_id}/{precision} R0 input"
        )
    foundation, version = _build_modules()
    manifest = foundation.load_foundation_operator_manifest(manifest_path)
    requests = foundation.foundation_r0_artifact_inputs(
        manifest,
        generation_revision=version.JIT_GENERATION_VERSION,
        compatibility_version=2,
        backend=backend,
        target=target,
        precisions=(precision,),
        schedule=selected["schedule"],
        schedule_revision=_r0_schedule_revision(version, backend),
        floating_point_policy=floating_point_policy,
    )
    request = next(
        value for value in requests if artifact_input_id in value["artifact_input_ids"]
    )
    source, metadata = render_generated_r0_operator(
        request["contract"],
        generation_version=version.JIT_GENERATION_VERSION,
        source_content_sha256=source_content_sha256,
        backend=backend,
        target=target,
        precision=precision,
        schedule=selected["schedule"],
        floating_point_policy=request["identity"]["floating_point_policy"],
    )
    if metadata["artifact_id"] != selected["artifact_id"]:
        raise RuntimeError("planned and rendered artifact identities disagree")
    metadata = {
        **metadata,
        "artifact_input_ids": request["artifact_input_ids"],
        "bundle_ids": request["bundle_ids"],
        "binary_format": selected["binary_format"],
        "build_provenance": {
            "source_content_sha256": source_content_sha256,
            "compiler_id": compiler_id,
            "compiler_version": compiler_version,
            "floating_point_policy": request["identity"]["floating_point_policy"],
        },
    }
    source_path = Path(source_path)
    metadata_path = Path(metadata_path)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_foundation_oam_artifact(
    *,
    oam_manifest_path: Path | str,
    architecture_id: str,
    backend: str,
    target: str,
    precision: str,
    source_content_sha256: str,
    compiler_id: str,
    compiler_version: str,
    source_path: Path | str,
    metadata_path: Path | str,
    companion_manifest_path: Path | str,
    floating_point_policy: str | None = None,
) -> None:
    """Render one model-independent known-OAM artifact and loader manifest."""

    manifest = _load_oam_manifest(oam_manifest_path)
    selected = next(
        (
            entry
            for entry in manifest["architectures"]
            if entry["id"] == architecture_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"unknown OAM architecture {architecture_id!r}")
    metadata, source = _oam_generated_metadata(
        selected["architecture"],
        backend,
        target,
        precision,
        floating_point_policy,
    )
    _, version = _build_modules()
    generated_identity = _oam_identity(
        metadata,
        backend=backend,
        target=target,
        generation_revision=version.JIT_GENERATION_VERSION,
    )
    catalog_metadata = {
        **metadata,
        "operator": "OAM",
        "schema": metadata["abi"],
        "capabilities": metadata["required_capabilities"],
        "generated_identity": generated_identity,
        "architecture_id": architecture_id,
        "source_contract_sha256": selected["source_contract_sha256"],
        "binary_format": {
            "host": "shared-library",
            "cuda": "cubin",
        }[backend],
        "build_provenance": {
            "source_content_sha256": source_content_sha256,
            "compiler_id": compiler_id,
            "compiler_version": compiler_version,
            **(
                {"floating_point_policy": metadata["floating_point_policy"]}
                if backend == "host"
                else {}
            ),
        },
    }
    source_path = Path(source_path)
    metadata_path = Path(metadata_path)
    companion_manifest_path = Path(companion_manifest_path)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    companion_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(catalog_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    companion_manifest_path.write_text(
        json.dumps(
            {"key_inputs": {"build": {"metadata": metadata}}},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    render = commands.add_parser("render")
    for command in (plan, render):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--oam-manifest", type=Path)
        command.add_argument("--backend", choices=_BACKENDS, required=True)
        command.add_argument("--target", required=True)
        command.add_argument("--source-content-sha256", required=True)
        command.add_argument(
            "--floating-point-policy", choices=("strict", "fast", "cuda-fast-math")
        )
    plan.add_argument("--oam-floating-point-policy", choices=("strict", "fast"))
    plan.add_argument("--mode", choices=_MODES, required=True)
    plan.add_argument("--oam-mode", choices=_OAM_MODES, default="off")
    plan.add_argument("--output", type=Path, required=True)
    render.add_argument(
        "--operator", choices=("R1", "R0", "M0", "M1", "OAM"), default="R0"
    )
    render.add_argument("--artifact-id")
    render.add_argument("--artifact-input-id")
    render.add_argument("--precision", choices=("float32", "float64"), required=True)
    render.add_argument("--compiler-id", required=True)
    render.add_argument("--compiler-version", required=True)
    render.add_argument("--source", type=Path, required=True)
    render.add_argument("--metadata", type=Path, required=True)
    render.add_argument("--architecture-id")
    render.add_argument("--companion-manifest", type=Path)
    catalog = commands.add_parser("catalog")
    catalog.add_argument(
        "--artifact",
        nargs="+",
        action="append",
        metavar="ARTIFACT_FIELD",
        required=True,
    )
    catalog.add_argument("--mode", choices=("r0", "full", "oam"), required=True)
    catalog.add_argument("--backend", choices=_BACKENDS, required=True)
    catalog.add_argument("--target", required=True)
    catalog.add_argument("--source-content-sha256", required=True)
    catalog.add_argument("--native-source-content-sha256", required=True)
    catalog.add_argument("--compiler-id", required=True)
    catalog.add_argument("--compiler-version", required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--header", type=Path, required=True)
    timestamp = commands.add_parser("timestamp")
    timestamp.add_argument("--output", type=Path, required=True)
    timestamp.add_argument("--preserve-if-newer-than", type=Path)
    host_target = commands.add_parser("host-target")
    host_target.add_argument("--declared-target", required=True)
    host_target.add_argument("--host-arch", required=True)
    host_target.add_argument("--flag", action="append", default=[])
    host_target.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        value = plan_foundation_operator_artifacts(
            manifest_path=args.manifest,
            mode=args.mode,
            backend=args.backend,
            target=args.target,
            source_content_sha256=args.source_content_sha256,
            floating_point_policy=args.floating_point_policy,
            oam_manifest_path=args.oam_manifest,
            oam_mode=args.oam_mode,
            oam_floating_point_policy=args.oam_floating_point_policy,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif args.command == "render":
        common = {
            "manifest_path": args.manifest,
            "backend": args.backend,
            "target": args.target,
            "precision": args.precision,
            "source_content_sha256": args.source_content_sha256,
            "compiler_id": args.compiler_id,
            "compiler_version": args.compiler_version,
            "source_path": args.source,
            "metadata_path": args.metadata,
            "floating_point_policy": args.floating_point_policy,
        }
        if args.operator == "OAM":
            if (
                args.oam_manifest is None
                or args.architecture_id is None
                or args.companion_manifest is None
            ):
                raise ValueError(
                    "OAM render requires --oam-manifest, --architecture-id, "
                    "and --companion-manifest"
                )
            render_foundation_oam_artifact(
                oam_manifest_path=args.oam_manifest,
                architecture_id=args.architecture_id,
                backend=args.backend,
                target=args.target,
                precision=args.precision,
                source_content_sha256=args.source_content_sha256,
                compiler_id=args.compiler_id,
                compiler_version=args.compiler_version,
                source_path=args.source,
                metadata_path=args.metadata,
                companion_manifest_path=args.companion_manifest,
                floating_point_policy=args.floating_point_policy,
            )
        elif args.artifact_id:
            render_foundation_operator_artifact(
                **common, operator=args.operator, artifact_id=args.artifact_id
            )
        elif args.operator == "R0" and args.artifact_input_id:
            render_foundation_r0_artifact(
                **common, artifact_input_id=args.artifact_input_id
            )
        else:
            raise ValueError("render requires --artifact-id")
    elif args.command == "catalog":
        write_installed_operator_catalog(
            artifacts=[
                tuple(
                    Path(value) if index in (0, 1, 3, 4) else value
                    for index, value in enumerate(record)
                )
                for record in args.artifact
            ],
            catalog_path=args.output,
            header_path=args.header,
            mode=args.mode,
            backend=args.backend,
            target=args.target,
            source_content_sha256=args.source_content_sha256,
            native_source_content_sha256=args.native_source_content_sha256,
            compiler_id=args.compiler_id,
            compiler_version=args.compiler_version,
        )
    elif args.command == "timestamp":
        write_build_start_timestamp(
            args.output, preserve_if_newer_than=args.preserve_if_newer_than
        )
    else:
        target = effective_prebuilt_host_target(
            declared_target=args.declared_target,
            host_arch=args.host_arch,
            compile_flags=args.flag,
        )
        args.output.write_text(target + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OAM_RELEASE_ARCHITECTURE_FINGERPRINTS",
    "audit_installed_oam_release_catalog",
    "audit_oam_release_budgets",
    "audit_oam_release_plan",
    "main",
    "plan_foundation_operator_artifacts",
    "render_foundation_operator_artifact",
    "render_foundation_oam_artifact",
    "render_foundation_r0_artifact",
    "sha256_file",
    "validate_generated_operator_catalog",
    "write_build_start_timestamp",
    "write_installed_operator_catalog",
]
