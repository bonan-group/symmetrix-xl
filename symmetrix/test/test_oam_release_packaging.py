from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools._symmetrix_build import prebuilt_operator_artifacts as prebuilt
from symmetrix import generated_operator_core as core


ROOT = Path(__file__).resolve().parents[2]
FOUNDATION_MANIFEST = (
    ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
)
OAM_MANIFEST = ROOT / "symmetrix/source/symmetrix/foundation_oam_architectures.json"
OAM_ARCHITECTURE_FINGERPRINTS = (
    "sha256:8e8b6c963d1be1449a43b0c02c6b80bdbd1b83070fb7abff5f6c806a0efa670f",
    "sha256:d8b773a86ebec83d1ea4dcbc3babe72242aebac3c88df245d6afbfc8352ced79",
    "sha256:676b385abb23c436411e3d9cb6032a7030b7141d4dd42db5b2e2f7104d93a307",
    "sha256:4ad5f8500b618300493547dca76321e8d9c870988239228e071d91bd9cc40c12",
)


@pytest.mark.parametrize(
    ("backend", "target", "floating_point_policy"),
    (("host", "x86-64-v3", "strict"), ("cuda", "sm120", None)),
)
def test_release_plan_contains_only_minimal_full_message_oam_catalog(
    backend, target, floating_point_policy
):
    plan = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=FOUNDATION_MANIFEST,
        mode="r0",
        backend=backend,
        target=target,
        source_content_sha256="a" * 64,
        oam_manifest_path=OAM_MANIFEST,
        oam_mode="known",
        oam_floating_point_policy=floating_point_policy,
    )

    release = prebuilt.audit_oam_release_plan(
        plan,
        backend=backend,
        target=target,
    )

    assert release == {
        "artifact_count": 8,
        "architectures": [
            "oam-l-v0.1",
            "oam-m-v0.1",
            "oam-s-v0.1",
            "oam-xl-v0.1",
        ],
        "precisions": ["float32", "float64"],
        "variants": ["full-message"],
    }
    assert {
        artifact["architecture_fingerprint"]
        for artifact in plan["artifacts"]
        if artifact["operator"] == "OAM"
    } == set(OAM_ARCHITECTURE_FINGERPRINTS)


def test_release_plan_rejects_incidental_tuning_variant():
    plan = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=FOUNDATION_MANIFEST,
        mode="r0",
        backend="cuda",
        target="sm120",
        source_content_sha256="a" * 64,
        oam_manifest_path=OAM_MANIFEST,
        oam_mode="known",
    )
    oam = next(item for item in plan["artifacts"] if item["operator"] == "OAM")
    plan["artifacts"].append(dict(oam, stem=oam["stem"] + "-tuning"))

    with pytest.raises(ValueError, match="exactly 8 OAM artifacts"):
        prebuilt.audit_oam_release_plan(plan, backend="cuda", target="sm120")


def test_release_plan_rejects_an_unknown_oam_architecture():
    plan = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=FOUNDATION_MANIFEST,
        mode="r0",
        backend="cuda",
        target="sm120",
        source_content_sha256="a" * 64,
        oam_manifest_path=OAM_MANIFEST,
        oam_mode="known",
    )
    oam = next(item for item in plan["artifacts"] if item["operator"] == "OAM")
    oam["architecture_fingerprint"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="qualified OAM architectures"):
        prebuilt.audit_oam_release_plan(plan, backend="cuda", target="sm120")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("oam_artifact_count", 9, "OAM artifact count"),
        ("installed_bytes", 128_000_001, "installed-byte budget"),
        ("build_duration_seconds", 1_800.001, "build-time budget"),
        ("wheel_bytes", 256_000_001, "wheel-size budget"),
    ),
)
def test_release_budget_fails_closed(field, value, match):
    measurement = {
        "oam_artifact_count": 8,
        "installed_bytes": 128_000_000,
        "build_duration_seconds": 1_800.0,
        "wheel_bytes": 256_000_000,
    }
    measurement[field] = value

    with pytest.raises(ValueError, match=match):
        prebuilt.audit_oam_release_budgets(measurement)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("oam_artifact_count", 8.0),
        ("oam_artifact_count", True),
        ("installed_bytes", 1.5),
        ("installed_bytes", float("nan")),
        ("installed_bytes", float("inf")),
        ("build_duration_seconds", float("-inf")),
        ("wheel_bytes", 2.5),
        ("wheel_bytes", True),
        ("wheel_bytes", 1 + 2j),
    ),
)
def test_release_budget_rejects_nonfinite_nonreal_or_noninteger_metrics(field, value):
    measurement = {
        "oam_artifact_count": 8,
        "installed_bytes": 1_000,
        "build_duration_seconds": 2.0,
        "wheel_bytes": 2_000,
    }
    measurement[field] = value

    with pytest.raises(ValueError, match="release measurement"):
        prebuilt.audit_oam_release_budgets(measurement)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_installed_package(tmp_path: Path, backend: str = "cuda") -> dict:
    target = "sm120" if backend == "cuda" else "x86-64-v3"
    package_root = tmp_path / f"symmetrix_backend_{backend}"
    data_root = package_root / "_operator_artifacts"
    data_root.mkdir(parents=True)
    source_hash = "a" * 64
    native_hash = "b" * 64
    entries = []
    oam_manifest = prebuilt._load_oam_manifest(OAM_MANIFEST)
    for architecture in oam_manifest["architectures"]:
        for precision in ("float32", "float64"):
            metadata, _ = prebuilt._oam_generated_metadata(
                architecture["architecture"], backend, target, precision, "strict"
            )
            identity = prebuilt._oam_identity(
                metadata,
                backend=backend,
                target=target,
                generation_revision=prebuilt._oam_modules()[3].JIT_GENERATION_VERSION,
            )
            member_root = Path("oam") / architecture["id"] / precision
            binary_name = "plugin.cubin" if backend == "cuda" else "plugin.so"
            binary = data_root / member_root / binary_name
            auxiliary = data_root / member_root / "manifest.json"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(f"{architecture['id']}:{precision}".encode())
            auxiliary.write_text(json.dumps({"identity": identity}, sort_keys=True))
            entries.append(
                {
                    "operator": "OAM",
                    "identity": identity,
                    "binary_format": (
                        "cubin" if backend == "cuda" else "shared-library"
                    ),
                    "installed_relative_path": str(member_root / binary_name),
                    "checksum": _sha256(binary),
                    "abi": {
                        "tag": metadata["abi"],
                        "version": metadata["abi_version"],
                    },
                    "capabilities": metadata["required_capabilities"],
                    "build_provenance": {
                        "source_content_sha256": "sha256:" + source_hash,
                        "compiler_id": "NVIDIA" if backend == "cuda" else "GNU",
                        "compiler_version": "13.3.73" if backend == "cuda" else "13.3",
                        **(
                            {"floating_point_policy": "strict"}
                            if backend == "host"
                            else {}
                        ),
                    },
                    "auxiliary_files": [
                        {
                            "role": "manifest",
                            "installed_relative_path": str(
                                member_root / "manifest.json"
                            ),
                            "checksum": _sha256(auxiliary),
                        }
                    ],
                }
            )
    catalog = core.generated_operator_catalog(entries)
    catalog_path = data_root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    installed_bytes = sum(
        path.stat().st_size for path in data_root.rglob("*") if path.is_file()
    )
    build_info = {
        "mode": "oam",
        "backend": backend,
        "target": identity["target"],
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "artifact_count": 8,
        "installed_bytes": installed_bytes,
        "source_content_sha256": source_hash,
        "native_source_content_sha256": native_hash,
        "compiler_id": "NVIDIA" if backend == "cuda" else "GNU",
        "compiler_version": "13.3.73" if backend == "cuda" else "13.3",
        "build_duration_seconds": 2.0,
    }
    extension = package_root / f"_native_{backend}.so"
    extension.write_bytes(b"native")
    installed_report = core.validate_installed_operator_artifacts(extension, build_info)
    native_build_info = {
        "source_commit": "c" * 40,
        "source_dirty": False,
        "source_content_sha256": source_hash,
        "native_source_content_sha256": native_hash,
        "fast_math": False,
        "prebuilt_operator_artifacts": build_info,
    }
    backend_descriptor = {
        "selector": "cuda13-sm120" if backend == "cuda" else "cpu",
        "backend": backend,
        "architecture": target,
        "source_commit": "c" * 40,
        "source_dirty": False,
        "source_content_sha256": source_hash,
        "native_source_content_sha256": native_hash,
    }
    probe = {
        "probe_status": "ok",
        "probe": {
            "backend": backend_descriptor,
            "extension": str(extension),
            "build_info": native_build_info,
            "installed_operator_artifacts": installed_report,
        },
    }
    wheel = tmp_path / "symmetrix_xl_cuda13_sm120-0.2-cp312-manylinux.whl"
    wheel.write_bytes(b"repaired-wheel")
    return {
        "package_root": package_root,
        "data_root": data_root,
        "catalog": catalog,
        "catalog_path": catalog_path,
        "build_info": build_info,
        "probe": probe,
        "wheel": wheel,
        "backend": backend,
        "target": target,
    }


def _rewrite_catalog(fixture: dict, catalog: dict) -> None:
    catalog_path = fixture["catalog_path"]
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    build_info = fixture["build_info"]
    build_info["catalog_sha256"] = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    build_info["installed_bytes"] = sum(
        path.stat().st_size
        for path in fixture["data_root"].rglob("*")
        if path.is_file()
    )
    reported = fixture["probe"]["probe"]["installed_operator_artifacts"]
    reported["catalog_sha256"] = build_info["catalog_sha256"]
    reported["installed_bytes"] = build_info["installed_bytes"]


def test_installed_oam_lookup_ignores_runtime_post_linear_topology(
    monkeypatch, tmp_path
):
    import copy
    from types import SimpleNamespace

    from symmetrix import oam_installed_artifact as installed

    fixture = _canonical_installed_package(tmp_path, backend="host")
    entry = next(
        artifact
        for artifact in fixture["catalog"]["artifacts"]
        if artifact["operator"] == "OAM"
        and artifact["identity"]["precision"] == "float64"
    )
    requested = copy.deepcopy(entry["identity"]["architecture"])
    requested["layers"][0]["post_linear"]["instructions"][0]["path_weight"] *= 2.0
    native = SimpleNamespace(
        __package__="symmetrix_backend_host",
        _backend_build_info=lambda: {
            "prebuilt_operator_artifacts": fixture["build_info"]
        },
    )
    monkeypatch.setattr(
        installed.resources,
        "files",
        lambda package_name: fixture["package_root"],
    )

    resolved = installed.resolve_installed_oam_artifact(
        architecture=requested,
        precision="float64",
        backend="host",
        host_target=entry["identity"]["target"],
        floating_point_policy=entry["identity"]["floating_point_policy"],
        artifact_id=entry["identity"]["artifact_id"],
        required_capabilities=entry["capabilities"],
        native_module=native,
    )

    assert resolved.available
    assert resolved.artifact_path == (
        fixture["data_root"] / entry["installed_relative_path"]
    )


def _run_package_audit(fixture: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(fixture["probe"]))
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/audit_oam_release_package.py"),
            "--probe-json",
            str(probe_path),
            "--wheel",
            str(fixture["wheel"]),
            "--backend",
            fixture["backend"],
            "--target",
            fixture["target"],
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_release_manifest_declares_pending_current_qualification():
    from symmetrix.oam_cuda_codegen import ABI_VERSION as cuda_abi_version
    from symmetrix.oam_receiver_codegen import ABI_VERSION as host_abi_version

    release = json.loads(
        (ROOT / "benchmarks/oam_release_package_manifest.json").read_text()
    )

    assert release["schema"] == "symmetrix.oam-release-package/2"
    assert release["qualification"] == {
        "status": "pending",
        "artifact_abi_versions": {
            "host": host_abi_version,
            "cuda": cuda_abi_version,
        },
        "execution_variants": ["full-message"],
    }
    assert release["host"]["target"] == "x86-64-v3"
    assert release["host"]["runtime_qualified"] == []
    assert release["host"]["available_not_runtime_qualified"] == [
        "S/float32",
        "S/float64",
        "M/float32",
        "M/float64",
        "L/float32",
        "L/float64",
        "XL/float32",
        "XL/float64",
    ]
    assert release["cuda"]["runtime_qualified_targets"] == []
    assert release["cuda"]["runtime_qualified"] == []
    assert release["cuda"]["capacity_qualified"] == []
    assert "sm120" in release["cuda"]["available_not_runtime_qualified_targets"]
    assert release["catalog"]["oam_artifact_count"] == (
        prebuilt.OAM_RELEASE_ARTIFACT_COUNT
    )
    assert set(release["catalog"]["architecture_fingerprints"].values()) == set(
        prebuilt.OAM_RELEASE_ARCHITECTURE_FINGERPRINTS.values()
    )
    assert release["catalog"]["budgets"] == {
        "installed_bytes_max": prebuilt.OAM_RELEASE_INSTALLED_BYTES_MAX,
        "build_duration_seconds_max": prebuilt.OAM_RELEASE_BUILD_SECONDS_MAX,
        "repaired_wheel_bytes_max": prebuilt.OAM_RELEASE_WHEEL_BYTES_MAX,
    }
    assert release["unsupported"] == [
        "HIP",
        "parameter gradients",
        "LAMMPS/MPI",
        "multi-GPU",
        "oam-fixed-workspace-v1",
    ]


def test_repaired_wheel_audit_cli_reads_exact_installed_probe(tmp_path):
    fixture = _canonical_installed_package(tmp_path)

    completed = _run_package_audit(fixture, tmp_path)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["oam_artifact_count"] == 8
    assert report["target"] == "sm120"
    assert report["wheel"] == str(fixture["wheel"].resolve())
    assert report["catalog_sha256"] == fixture["build_info"]["catalog_sha256"]


def test_repaired_wheel_audit_binds_host_policy_to_native_fast_math(tmp_path):
    fixture = _canonical_installed_package(tmp_path, backend="host")
    completed = _run_package_audit(fixture, tmp_path)
    assert completed.returncode == 0, completed.stderr

    fixture["probe"]["probe"]["build_info"]["fast_math"] = True
    completed = _run_package_audit(fixture, tmp_path)

    assert completed.returncode != 0
    assert "floating-point policy" in completed.stderr


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("schema", "schema"),
        ("abi", "ABI"),
        ("catalog_hash", "checksum"),
        ("file_checksum", "checksum"),
        ("path_escape", "safe path"),
        ("missing_file", "missing"),
        ("provenance", "provenance"),
        ("capability", "capabilities"),
        ("unrelated_file", "membership"),
        ("artifact_id", "artifact identity"),
        ("generation", "generation revision"),
        ("target", "requested target"),
        ("probe_binding", "probe/backend provenance"),
    ),
)
def test_repaired_wheel_audit_cli_rejects_noncanonical_package(
    tmp_path, mutation, match
):
    fixture = _canonical_installed_package(tmp_path)
    catalog = fixture["catalog"]
    entry = catalog["artifacts"][0]
    if mutation == "schema":
        catalog["schema"] = "symmetrix.generated-operator-catalog/0"
        _rewrite_catalog(fixture, catalog)
    elif mutation == "abi":
        entry["abi"]["version"] -= 1
        _rewrite_catalog(fixture, catalog)
    elif mutation == "catalog_hash":
        fixture["build_info"]["catalog_sha256"] = "0" * 64
    elif mutation == "file_checksum":
        binary = fixture["data_root"] / entry["installed_relative_path"]
        binary.write_bytes(b"tampered")
    elif mutation == "path_escape":
        entry["installed_relative_path"] = "../escape.cubin"
        _rewrite_catalog(fixture, catalog)
    elif mutation == "missing_file":
        binary = fixture["data_root"] / entry["installed_relative_path"]
        binary.unlink()
    elif mutation == "provenance":
        entry["build_provenance"]["source_content_sha256"] = "sha256:" + "0" * 64
        _rewrite_catalog(fixture, catalog)
    elif mutation == "capability":
        entry["capabilities"] = ["full-message"]
        _rewrite_catalog(fixture, catalog)
    elif mutation == "unrelated_file":
        (fixture["data_root"] / "unrelated.bin").write_bytes(b"unrelated")
    elif mutation == "artifact_id":
        entry["identity"]["artifact_id"] = "oam-cuda-float32-" + "0" * 32
        _rewrite_catalog(fixture, catalog)
    elif mutation == "generation":
        entry["identity"]["generation_revision"] += 1
        _rewrite_catalog(fixture, catalog)
    elif mutation == "target":
        fixture["probe"]["probe"]["backend"]["architecture"] = "sm90"
    elif mutation == "probe_binding":
        fixture["probe"]["probe"]["backend"]["source_content_sha256"] = "0" * 64
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    completed = _run_package_audit(fixture, tmp_path)

    assert completed.returncode != 0
    assert match.lower() in completed.stderr.lower()


def test_installed_catalog_validation_uses_the_catalog_bytes_already_read(
    tmp_path, monkeypatch
):
    fixture = _canonical_installed_package(tmp_path)
    original_read_text = Path.read_text
    original_open = core.os.open
    catalog_reads = 0

    def reject_catalog_reread(path, *args, **kwargs):
        if path == fixture["catalog_path"]:
            raise AssertionError("catalog bytes were read more than once")
        return original_read_text(path, *args, **kwargs)

    def count_catalog_open(path, *args, **kwargs):
        nonlocal catalog_reads
        if Path(path) == fixture["catalog_path"]:
            catalog_reads += 1
            assert catalog_reads == 1, "catalog was opened more than once"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_catalog_reread)
    monkeypatch.setattr(core.os, "open", count_catalog_open)

    report, catalog, data_root = core.validated_installed_operator_catalog(
        fixture["package_root"], fixture["build_info"]
    )

    assert report["status"] == "ok"
    assert catalog_reads == 1
    assert catalog["schema"] == core.GENERATED_OPERATOR_CATALOG_SCHEMA
    assert data_root == fixture["data_root"]


def test_installed_catalog_validation_rejects_an_intermediate_symlink_escape(
    tmp_path,
):
    fixture = _canonical_installed_package(tmp_path)
    outside = tmp_path / "outside" / "nested"
    outside.parent.mkdir()
    fixture["data_root"].rename(outside)
    (fixture["package_root"] / "bridge").symlink_to(
        outside.parent, target_is_directory=True
    )
    fixture["build_info"]["catalog_relative_path"] = "bridge/nested/catalog.json"

    with pytest.raises(ValueError, match="contained"):
        core.validated_installed_operator_catalog(
            fixture["package_root"], fixture["build_info"]
        )
