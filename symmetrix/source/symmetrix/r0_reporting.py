"""Public semantic reporting for the selected R0 implementation."""

from __future__ import annotations

from typing import Any


R0_REPORTING_SCHEMA = {
    "schema": "symmetrix.r0-reporting/1",
    "implementation": ["generic", "generated"],
    "backend": ["host", "cuda", "hip"],
    "artifact_origin": ["prebuilt", "cache", "runtime"],
    "nullable": ["artifact_origin", "artifact_id"],
    "separate_fields": ["schedule", "profile"],
}


def r0_reporting_schema() -> dict[str, Any]:
    """Return an independent copy of the public R0 reporting contract."""

    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in R0_REPORTING_SCHEMA.items()
    }


def r0_diagnostics(evaluator) -> dict[str, Any]:
    """Normalize and validate an evaluator's public R0 selection report."""

    implementation = str(getattr(evaluator, "r0_implementation", ""))
    if implementation not in {"generic", "generated"}:
        raise RuntimeError(f"invalid public R0 implementation {implementation!r}")
    backend = str(getattr(evaluator, "r0_backend", ""))
    if backend not in {"host", "cuda", "hip"}:
        raise RuntimeError(f"invalid R0 backend {backend!r}")
    origin = getattr(evaluator, "r0_artifact_origin", None)
    if origin == "":
        origin = None
    if origin not in {None, "prebuilt", "cache", "runtime"}:
        raise RuntimeError(f"invalid R0 artifact origin {origin!r}")
    schedule = str(getattr(evaluator, "r0_schedule", ""))
    profile = str(getattr(evaluator, "r0_profile", ""))
    if not schedule or not profile:
        raise RuntimeError("R0 schedule and profile must be reported separately")
    artifact_id = getattr(evaluator, "r0_module_id", None)
    if not artifact_id:
        artifact_id = None
    if implementation == "generic":
        if origin is not None:
            raise RuntimeError("generic R0 cannot report an artifact origin")
        if artifact_id is not None:
            raise RuntimeError("generic R0 cannot report a generated artifact ID")
        if schedule != "generic":
            raise RuntimeError("generic R0 cannot report a generated schedule")
        if profile != "generic":
            raise RuntimeError("generic R0 cannot report a generated profile")
    return {
        "implementation": implementation,
        "backend": backend,
        "artifact_origin": origin,
        "schedule": schedule,
        "profile": profile,
        "artifact_id": artifact_id,
    }


__all__ = ["R0_REPORTING_SCHEMA", "r0_diagnostics", "r0_reporting_schema"]
