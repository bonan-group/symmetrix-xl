"""Resolve architecture-bound OAM artifacts installed in a backend wheel."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from .cuda_codegen_common import canonical_cuda_target_identity
from .generated_operator_core import validated_installed_operator_catalog
from .installed_operator_artifact import (
    _concrete_package_root,
    _prebuilt_build_info,
    _validation_failure_reason,
)
from .oam_architecture import (
    oam_tensor_product_architecture_fingerprint,
    validate_oam_architecture,
)


@dataclass(frozen=True)
class InstalledOAMResolution:
    """One package-origin OAM artifact resolution attempt."""

    status: str
    artifact_path: Path | None
    manifest_path: Path | None
    diagnostics: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.status == "resolved"


def resolve_installed_oam_artifact(
    *,
    architecture: Mapping[str, Any],
    precision: str,
    backend: str,
    native_module: Any,
    target: str | None = None,
    host_target: str | None = None,
    floating_point_policy: str | None = None,
    artifact_id: str | None = None,
    required_capabilities: int | None = None,
) -> InstalledOAMResolution:
    """Return an exact bundled artifact when the backend declares one.

    Selection compares the generated tensor-product architecture. Runtime-only
    post-linear topology remains in the catalog for contract reconstruction but
    does not split otherwise identical full-message artifacts.
    """

    architecture = validate_oam_architecture(architecture)
    architecture_fingerprint = oam_tensor_product_architecture_fingerprint(architecture)
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be float32 or float64")
    if backend not in ("host", "cuda"):
        raise ValueError("OAM installed backend must be host or cuda")
    if target is not None and (not isinstance(target, str) or not target):
        raise ValueError("installed OAM target must be a nonempty string")
    if required_capabilities is not None and (
        isinstance(required_capabilities, bool)
        or not isinstance(required_capabilities, int)
        or required_capabilities < 0
    ):
        raise ValueError("required_capabilities must be a nonnegative integer")
    build_info = _prebuilt_build_info(native_module)
    absent = InstalledOAMResolution(
        status="absent",
        artifact_path=None,
        manifest_path=None,
        diagnostics=(
            "origin=prebuilt status=absent family=OAM reason=architecture-mismatch",
        ),
    )
    if build_info is None or build_info.get("mode") == "off":
        return InstalledOAMResolution(
            status="absent",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=absent family=OAM reason=catalog-absent",
            ),
        )
    if build_info.get("backend") != backend:
        return absent
    package_name = getattr(native_module, "__package__", None)
    if not isinstance(package_name, str) or not package_name:
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM "
                "reason=installed-package-invalid",
            ),
        )
    try:
        package_resource = resources.files(package_name)
        package_root = _concrete_package_root(package_resource, build_info)
        _, catalog, data_root = validated_installed_operator_catalog(
            package_root, build_info
        )
    except Exception as error:
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM reason={} detail={}: {}".format(
                    _validation_failure_reason(error), type(error).__name__, error
                ),
            ),
        )
    architecture_matches = [
        entry
        for entry in catalog["artifacts"]
        if entry["operator"] == "OAM"
        and entry["identity"]["architecture_fingerprint"] == architecture_fingerprint
        and entry["identity"]["precision"] == precision
        and entry["identity"]["backend"] == backend
    ]
    requested_target = (
        canonical_cuda_target_identity(target)
        if backend == "cuda" and target is not None
        else target
    )
    target_matches = [
        entry
        for entry in architecture_matches
        if (
            (backend != "host" or host_target is None)
            and (
                requested_target is None
                or (
                    backend == "cuda"
                    and canonical_cuda_target_identity(entry["identity"].get("target"))
                    == requested_target
                )
                or (
                    backend != "cuda"
                    and entry["identity"].get("target") == requested_target
                )
            )
        )
        or (
            backend == "host"
            and entry["identity"].get("host_target", entry["identity"].get("target"))
            == host_target
        )
    ]
    if not target_matches and architecture_matches and target is not None:
        available_targets = sorted(
            {
                canonical_cuda_target_identity(entry["identity"].get("target"))
                if backend == "cuda"
                else entry["identity"].get(
                    "host_target", entry["identity"].get("target")
                )
                for entry in architecture_matches
            }
        )
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM "
                f"reason=target-mismatch requested={requested_target} "
                f"available={','.join(available_targets)} "
                "action=install-matching-target",
            ),
        )
    matches = [
        entry
        for entry in target_matches
        if (
            floating_point_policy is None
            or entry["identity"].get("floating_point_policy") == floating_point_policy
        )
        and (artifact_id is None or entry["identity"].get("artifact_id") == artifact_id)
    ]
    if not target_matches and architecture_matches and backend == "host":
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM "
                "reason=host-target-incompatible action=regenerate",
            ),
        )
    if not matches and target_matches and backend == "host":
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM "
                "reason=host-policy-or-artifact-identity-stale action=regenerate",
            ),
        )
    if not matches and target_matches and artifact_id is not None:
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM "
                "reason=abi-or-artifact-identity-stale action=regenerate",
            ),
        )
    if not matches:
        return absent
    capability_matches = [
        entry
        for entry in matches
        if required_capabilities is None
        or entry.get("capabilities") == required_capabilities
    ]
    if matches and not capability_matches:
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM "
                "reason=capability-mismatch action=regenerate",
            ),
        )
    matches = capability_matches
    if len(matches) != 1:
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM "
                "reason=ambiguous-architecture",
            ),
        )
    entry = matches[0]
    auxiliary = entry.get("auxiliary_files", ())
    if len(auxiliary) != 1 or auxiliary[0].get("role") != "manifest":
        return InstalledOAMResolution(
            status="rejected",
            artifact_path=None,
            manifest_path=None,
            diagnostics=(
                "origin=prebuilt status=rejected family=OAM reason=manifest-missing",
            ),
        )
    artifact_path = data_root.joinpath(
        *PurePosixPath(entry["installed_relative_path"]).parts
    )
    manifest_path = data_root.joinpath(
        *PurePosixPath(auxiliary[0]["installed_relative_path"]).parts
    )
    return InstalledOAMResolution(
        status="resolved",
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        diagnostics=(
            "origin=prebuilt status=loaded family=OAM "
            f"artifact_id={entry['identity']['artifact_id']}",
        ),
    )


__all__ = ["InstalledOAMResolution", "resolve_installed_oam_artifact"]
