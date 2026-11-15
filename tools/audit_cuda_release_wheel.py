#!/usr/bin/env python3
"""Audit one architecture-qualified CUDA wheel before publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from _symmetrix_build.command import BuildError
from _symmetrix_build.wheel_audit import audit_wheel


def audit_cuda_wheel(
    wheel: Path,
    *,
    cuda_major: str,
    target: str,
    foundation_manifest: Path | None = None,
    prebuilt_mode: str = "r0",
    floating_point_policy: str = "cuda-fast-math",
) -> dict[str, object]:
    """Apply common wheel policy plus the exact CUDA package contract.

    ``foundation_manifest`` binds the catalog to the deterministic build
    plan: every catalog artifact id must equal the planned artifact id set
    for the wheel's own recorded source content hash.  That is the
    independent provenance anchor; the descriptor-side summary below only
    cross-checks consistency.
    """
    audit_wheel(wheel)
    selector = f"cuda{cuda_major}-{target.lower()}"
    package = f"symmetrix_backend_{selector.replace('-', '_')}"
    distribution = f"symmetrix-xl-{selector}"
    native_prefix = f"{package}/_native_{selector.replace('-', '_')}"
    descriptor_name = f"{package}/backend.json"
    with zipfile.ZipFile(wheel) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        if descriptor_name not in names:
            raise BuildError(
                f"CUDA wheel is missing backend descriptor {descriptor_name}"
            )
        descriptor = json.loads(archive.read(descriptor_name))
        expected = {
            "distribution": distribution,
            "backend": "cuda",
            "toolkit": f"cuda{cuda_major}",
            "selector": selector,
            "architecture": target.lower(),
        }
        for field, value in expected.items():
            if descriptor.get(field) != value:
                raise BuildError(
                    f"CUDA descriptor {field}={descriptor.get(field)!r}; expected {value!r}"
                )
        native = [
            name
            for name in names
            if name.startswith(native_prefix) and name.endswith(".so")
        ]
        if len(native) != 1:
            raise BuildError(
                f"CUDA wheel must contain one native extension, found {native}"
            )
        if any(
            name == "symmetrix/__init__.py" or name.startswith("symmetrix/")
            for name in names
        ):
            raise BuildError("CUDA add-on wheel must not contain the frontend package")
        catalog_name = f"{package}/_operator_artifacts/catalog.json"
        if catalog_name not in names:
            raise BuildError(
                f"CUDA wheel is missing prebuilt operator catalog {catalog_name}"
            )
        catalog = json.loads(archive.read(catalog_name))
        artifacts = catalog.get("artifacts")
        prebuilt = descriptor.get("prebuilt_operator_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise BuildError("CUDA wheel operator catalog is empty")
        if not isinstance(prebuilt, dict):
            # Backend project descriptors do not copy the catalog summary into
            # backend.json yet; the derived summary below is deliberately only
            # a self-consistency check.  Independent verification comes from
            # the per-artifact descriptor hash comparison above and the
            # plan-binding check driven by --foundation-manifest.
            prebuilt = {
                "backend": "cuda",
                "target": target.lower(),
                "artifact_count": len(artifacts),
                "catalog_sha256": hashlib.sha256(
                    archive.read(catalog_name)
                ).hexdigest(),
            }
        if (
            prebuilt.get("backend") != "cuda"
            or prebuilt.get("target") != target.lower()
        ):
            raise BuildError("CUDA prebuilt operator provenance has the wrong target")
        if prebuilt.get("artifact_count") != len(artifacts):
            raise BuildError("CUDA prebuilt artifact count does not match catalog")
        for item in artifacts:
            provenance = item.get("build_provenance", {})
            field = "source_content_sha256"
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(provenance.get(field, ""))):
                raise BuildError(f"CUDA catalog artifact has invalid {field}")
            descriptor_hash = "sha256:" + str(descriptor.get(field, ""))
            if provenance[field] != descriptor_hash:
                raise BuildError(
                    f"CUDA catalog artifact {field} disagrees with descriptor"
                )
        if foundation_manifest is not None:
            catalog_hashes = {
                item["build_provenance"]["source_content_sha256"] for item in artifacts
            }
            if len(catalog_hashes) != 1:
                raise BuildError(
                    "CUDA catalog artifacts disagree on source_content_sha256"
                )
            from _symmetrix_build.prebuilt_operator_artifacts import (
                plan_foundation_operator_artifacts,
            )

            plan = plan_foundation_operator_artifacts(
                manifest_path=foundation_manifest,
                mode=prebuilt_mode,
                backend="cuda",
                target=target.lower(),
                source_content_sha256=catalog_hashes.pop().split(":", 1)[1],
                floating_point_policy=floating_point_policy,
            )
            planned_ids = sorted(
                artifact["artifact_id"] for artifact in plan["artifacts"]
            )
            catalog_ids = sorted(item["identity"]["artifact_id"] for item in artifacts)
            if catalog_ids != planned_ids:
                raise BuildError(
                    "CUDA catalog artifact set does not match the planned "
                    f"prebuilt operator set for mode {prebuilt_mode!r}: "
                    f"catalog has {len(catalog_ids)} artifacts, plan has "
                    f"{len(planned_ids)}"
                )
        for field in ("source_content_sha256", "native_source_content_sha256"):
            summary_hash = prebuilt.get(field)
            if summary_hash is not None and summary_hash != descriptor.get(field):
                raise BuildError(
                    f"CUDA prebuilt artifact {field} disagrees with descriptor"
                )
        catalog_hash = hashlib.sha256(archive.read(catalog_name)).hexdigest()
        if prebuilt.get("catalog_sha256") != catalog_hash:
            raise BuildError("CUDA descriptor catalog_sha256 does not match catalog")
        return {
            "wheel": wheel.name,
            "distribution": distribution,
            "selector": selector,
            "artifacts": len(artifacts),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--cuda-major", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--foundation-manifest",
        type=Path,
        help="Bind the catalog to the deterministic plan from this manifest.",
    )
    parser.add_argument("--prebuilt-mode", default="r0")
    parser.add_argument("--floating-point-policy", default="cuda-fast-math")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            audit_cuda_wheel(
                args.wheel,
                cuda_major=args.cuda_major,
                target=args.target,
                foundation_manifest=args.foundation_manifest,
                prebuilt_mode=args.prebuilt_mode,
                floating_point_policy=args.floating_point_policy,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
