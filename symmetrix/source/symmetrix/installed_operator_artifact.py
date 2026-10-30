"""Resolve immutable installed generated-operator artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from .generated_operator_core import validated_installed_operator_catalog


@dataclass(frozen=True)
class InstalledOperatorResolution:
    """One installed-origin resolution attempt."""

    status: str
    origin: str | None
    artifact_path: Path | None
    artifact_id: str | None
    diagnostics: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.status == "resolved"


_rejected_catalogs: set[tuple[str, str]] = set()
_rejected_artifacts: set[tuple[str, str, str]] = set()


def _clear_installed_operator_rejections_for_tests() -> None:
    _rejected_catalogs.clear()
    _rejected_artifacts.clear()


def _diagnostic(operator: str, status: str, reason: str, detail: object = "") -> str:
    message = f"origin=prebuilt status={status} operator={operator} reason={reason}"
    if detail:
        message += f" detail={detail}"
    return message


def _result(operator: str, status: str, reason: str, detail: object = ""):
    return InstalledOperatorResolution(
        status=status,
        origin=None,
        artifact_path=None,
        artifact_id=None,
        diagnostics=(_diagnostic(operator, status, reason, detail),),
    )


def _validation_failure_reason(error: Exception) -> str:
    message = str(error).lower()
    if "checksum" in message:
        return "checksum-mismatch"
    if "backend target" in message:
        return "backend-target-mismatch"
    if "abi" in message:
        return "abi-mismatch"
    if "capabilit" in message:
        return "capability-mismatch"
    return "installed-package-invalid"


def _prebuilt_build_info(native_module: Any) -> Mapping[str, Any] | None:
    query = getattr(native_module, "_backend_build_info", None)
    if not callable(query):
        return None
    value = query()
    if not isinstance(value, Mapping):
        return None
    prebuilt = value.get("prebuilt_operator_artifacts")
    return prebuilt if isinstance(prebuilt, Mapping) else None


def installed_operator_build_target(
    *, native_module: Any, backend: str, mode: str | None = None
) -> str | None:
    """Return the package's installed target for ``backend``, if declared."""

    build_info = _prebuilt_build_info(native_module)
    if (
        build_info is None
        or build_info.get("mode") == "off"
        or (mode is not None and build_info.get("mode") != mode)
        or build_info.get("backend") != backend
    ):
        return None
    target = build_info.get("target")
    return target if isinstance(target, str) and target else None


def _identity_without_target(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in identity.items()
        if key not in ("target", "artifact_id")
    }


def _concrete_package_root(
    package_resource: Any, build_info: Mapping[str, Any]
) -> Path:
    """Resolve a filesystem-backed package root through its catalog resource."""

    relative_value = build_info.get("catalog_relative_path")
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError("installed operator catalog path must be nonempty")
    if "\\" in relative_value or any(
        ord(character) < 32 for character in relative_value
    ):
        raise ValueError("installed operator catalog path must use POSIX separators")
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0].endswith(":")
        or ".." in relative.parts
        or str(relative) != relative_value
        or relative_value.endswith("/")
    ):
        raise ValueError("installed operator catalog path must be normalized and safe")
    catalog_resource = package_resource.joinpath(*relative.parts)
    try:
        catalog_path = Path(catalog_resource)
    except TypeError as error:
        raise ValueError(
            "installed operator package data must be filesystem-backed"
        ) from error
    package_root = catalog_path
    for _ in relative.parts:
        package_root = package_root.parent
    return package_root


def resolve_installed_operator_artifact(
    *,
    operator: str,
    identity: Mapping[str, Any],
    abi: Mapping[str, Any],
    required_capabilities: int,
    loader: Callable[[str], Any],
    admitted: Callable[[], bool] | None = None,
    native_module: Any,
    policy: str = "prefer",
) -> InstalledOperatorResolution:
    """Validate and load one exact package artifact without touching JIT state."""

    if policy not in ("prefer", "required", "disabled"):
        raise ValueError(
            "installed artifact policy must be prefer, required, or disabled"
        )
    if policy == "disabled":
        return InstalledOperatorResolution(
            status="disabled",
            origin=None,
            artifact_path=None,
            artifact_id=None,
            diagnostics=(_diagnostic(operator, "skipped", "disabled"),),
        )
    if not isinstance(operator, str) or not operator:
        raise ValueError("operator must not be empty")
    if not isinstance(identity, Mapping):
        raise TypeError("identity must be a mapping")
    artifact_id = identity.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("identity must contain a nonempty artifact_id")
    if not isinstance(abi, Mapping):
        raise TypeError("abi must be a mapping")
    if admitted is not None and not callable(admitted):
        raise TypeError("admitted must be callable")
    if (
        isinstance(required_capabilities, bool)
        or not isinstance(required_capabilities, int)
        or required_capabilities < 0
    ):
        raise ValueError("required_capabilities must be a nonnegative integer")

    build_info = _prebuilt_build_info(native_module)
    if build_info is None or build_info.get("mode") == "off":
        return _result(operator, "absent", "catalog-absent")
    package_name = getattr(native_module, "__package__", None)
    if not isinstance(package_name, str) or not package_name:
        return _result(operator, "rejected", "installed-package-invalid", "package")
    catalog_key = (package_name, str(build_info.get("catalog_sha256", "")))
    if catalog_key in _rejected_catalogs:
        return _result(operator, "rejected", "previously-rejected")
    try:
        package_resource = resources.files(package_name)
        package_root = _concrete_package_root(package_resource, build_info)
        _, catalog, data_root = validated_installed_operator_catalog(
            package_root, build_info
        )
    except Exception as error:
        _rejected_catalogs.add(catalog_key)
        return _result(
            operator,
            "rejected",
            _validation_failure_reason(error),
            f"{type(error).__name__}: {error}",
        )

    candidates = [
        entry for entry in catalog["artifacts"] if entry["operator"] == operator
    ]
    exact = next(
        (entry for entry in candidates if entry["identity"] == dict(identity)), None
    )
    if exact is None:
        request_without_target = _identity_without_target(identity)
        target_mismatch = any(
            _identity_without_target(entry["identity"]) == request_without_target
            for entry in candidates
        )
        return _result(
            operator,
            "absent",
            "backend-target-mismatch" if target_mismatch else "identity-mismatch",
        )
    if exact["abi"] != dict(abi):
        return _result(operator, "rejected", "abi-mismatch")
    if exact["capabilities"] != required_capabilities:
        return _result(operator, "rejected", "capability-mismatch")

    rejection_key = (package_name, catalog_key[1], artifact_id)
    if rejection_key in _rejected_artifacts:
        return _result(operator, "rejected", "previously-rejected")
    relative = PurePosixPath(exact["installed_relative_path"])
    artifact_path = data_root.joinpath(*relative.parts)
    try:
        loader(str(artifact_path))
    except Exception as error:
        _rejected_artifacts.add(rejection_key)
        return _result(
            operator,
            "rejected",
            "load-failed",
            f"{type(error).__name__}: {error}",
        )
    try:
        retained = True if admitted is None else admitted()
    except Exception as error:
        _rejected_artifacts.add(rejection_key)
        return _result(
            operator,
            "rejected",
            "admission-failed",
            f"{type(error).__name__}: {error}",
        )
    if retained is not True:
        _rejected_artifacts.add(rejection_key)
        return _result(operator, "rejected", "admission-failed", "not-retained")
    return InstalledOperatorResolution(
        status="resolved",
        origin="prebuilt",
        artifact_path=artifact_path,
        artifact_id=artifact_id,
        diagnostics=(
            f"origin=prebuilt status=loaded operator={operator} "
            f"artifact_id={artifact_id}",
        ),
    )


__all__ = [
    "InstalledOperatorResolution",
    "installed_operator_build_target",
    "resolve_installed_operator_artifact",
]
