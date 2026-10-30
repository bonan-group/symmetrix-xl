from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from symmetrix import generated_operator_core as core
from symmetrix import oam_installed_artifact as installed
from symmetrix.oam_architecture import (
    oam_tensor_product_architecture_fingerprint,
    resolve_oam_architecture,
)
from symmetrix.oam_cuda_codegen import oam_cuda_artifact_metadata
from symmetrix.cuda_codegen_common import canonical_cuda_target_identity
from symmetrix.oam_receiver_codegen import oam_receiver_jit_metadata

FIXTURE = Path(__file__).parent / "data/nequip/oam_s_v2_minimal.json"


@pytest.mark.parametrize(
    "value",
    ("sm120", "sm_120", "compute_120", {"target": "sm_120"}),
)
def test_cuda_target_identity_normalizes_runtime_and_build_spellings(value):
    assert canonical_cuda_target_identity(value) == "sm120"


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_installed_package(tmp_path, *, mode, include_expected):
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    generated = oam_cuda_artifact_metadata(
        contract, precision="float32", target="sm_120"
    )
    catalog_generated = (
        generated
        if include_expected
        else oam_cuda_artifact_metadata(contract, precision="float64", target="sm_120")
    )
    package = tmp_path / f"installed_cuda_{mode}"
    artifact = package / "_operator_artifacts/oam/fixture/plugin.cubin"
    manifest = package / "_operator_artifacts/oam/fixture/manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"installed-oam-cuda")
    manifest.write_text(json.dumps({"artifact_id": catalog_generated["artifact_id"]}))
    entries = [
        {
            "operator": "OAM",
            "identity": {
                "schema": "symmetrix.generated-operator-identity/1",
                "operator": "OAM",
                "generation_revision": 19,
                "architecture": catalog_generated["architecture"],
                "architecture_fingerprint": oam_tensor_product_architecture_fingerprint(
                    catalog_generated["architecture"]
                ),
                "precision": catalog_generated["precision"],
                "backend": "cuda",
                "target": "sm_120",
                "generation_fingerprint": catalog_generated["generation_fingerprint"],
                "semantic_fingerprint": catalog_generated["semantic_fingerprint"],
                "structure_fingerprint": catalog_generated["structure_fingerprint"],
                "artifact_id": catalog_generated["artifact_id"],
            },
            "binary_format": "cubin",
            "installed_relative_path": "oam/fixture/plugin.cubin",
            "checksum": _sha(artifact),
            "abi": {
                "tag": catalog_generated["abi"],
                "version": catalog_generated["abi_version"],
            },
            "capabilities": catalog_generated["required_capabilities"],
            "build_provenance": {
                "source_content_sha256": "sha256:" + "a" * 64,
                "compiler_id": "NVIDIA",
                "compiler_version": "13.0",
            },
            "auxiliary_files": [
                {
                    "role": "manifest",
                    "installed_relative_path": "oam/fixture/manifest.json",
                    "checksum": _sha(manifest),
                }
            ],
        }
    ]
    catalog = core.generated_operator_catalog(entries)
    catalog_path = package / "_operator_artifacts/catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    files = [catalog_path, artifact, manifest]
    build_info = {
        "mode": mode,
        "backend": "cuda",
        "target": "sm_120",
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "artifact_count": len(entries),
        "installed_bytes": sum(path.stat().st_size for path in files),
        "source_content_sha256": "a" * 64,
        "native_source_content_sha256": "b" * 64,
        "compiler_id": "NVIDIA",
        "compiler_version": "13.0",
        "build_duration_seconds": 0.1,
    }
    native = SimpleNamespace(
        __package__=package.name,
        _backend_build_info=lambda: {"prebuilt_operator_artifacts": build_info},
    )
    return generated, package, native, artifact, manifest


@pytest.mark.parametrize("mode", ("r0", "full", "oam"))
def test_cuda_resolver_accepts_standard_catalog_modes_with_exact_oam(
    monkeypatch, tmp_path, mode
):
    generated, package, native, artifact, manifest = _cuda_installed_package(
        tmp_path, mode=mode, include_expected=True
    )
    monkeypatch.setattr(installed.resources, "files", lambda name: package)

    result = installed.resolve_installed_oam_artifact(
        architecture=generated["architecture"],
        precision="float32",
        backend="cuda",
        native_module=native,
        artifact_id=generated["artifact_id"],
    )

    assert result.available
    assert result.artifact_path == artifact
    assert result.manifest_path == manifest


def test_cuda_resolver_rejects_catalog_without_expected_oam(monkeypatch, tmp_path):
    generated, package, native, _, _ = _cuda_installed_package(
        tmp_path, mode="full", include_expected=False
    )
    monkeypatch.setattr(installed.resources, "files", lambda name: package)

    result = installed.resolve_installed_oam_artifact(
        architecture=generated["architecture"],
        precision="float32",
        backend="cuda",
        native_module=native,
        artifact_id=generated["artifact_id"],
    )

    assert result.status == "absent"
    assert result.artifact_path is None


def test_cuda_resolver_requires_exact_target_and_capabilities(monkeypatch, tmp_path):
    generated, package, native, _, _ = _cuda_installed_package(
        tmp_path, mode="r0", include_expected=True
    )
    monkeypatch.setattr(installed.resources, "files", lambda name: package)

    wrong_target = installed.resolve_installed_oam_artifact(
        architecture=generated["architecture"],
        precision="float32",
        backend="cuda",
        target="sm90",
        required_capabilities=generated["required_capabilities"],
        native_module=native,
        artifact_id=generated["artifact_id"],
    )
    wrong_capabilities = installed.resolve_installed_oam_artifact(
        architecture=generated["architecture"],
        precision="float32",
        backend="cuda",
        target="sm120",
        required_capabilities=generated["required_capabilities"] ^ 1,
        native_module=native,
        artifact_id=generated["artifact_id"],
    )

    assert wrong_target.status == "rejected"
    assert "reason=target-mismatch" in wrong_target.diagnostics[0]
    assert "requested=sm90" in wrong_target.diagnostics[0]
    assert "available=sm120" in wrong_target.diagnostics[0]
    assert wrong_capabilities.status == "rejected"
    assert "reason=capability-mismatch" in wrong_capabilities.diagnostics[0]
    assert "action=regenerate" in wrong_capabilities.diagnostics[0]


def test_cuda_resolver_reports_stale_artifact_identity(monkeypatch, tmp_path):
    generated, package, native, _, _ = _cuda_installed_package(
        tmp_path, mode="r0", include_expected=True
    )
    monkeypatch.setattr(installed.resources, "files", lambda name: package)

    result = installed.resolve_installed_oam_artifact(
        architecture=generated["architecture"],
        precision="float32",
        backend="cuda",
        target="sm120",
        artifact_id="oam-stale-abi-identity",
        native_module=native,
    )

    assert result.status == "rejected"
    assert "reason=abi-or-artifact-identity-stale" in result.diagnostics[0]
    assert "action=regenerate" in result.diagnostics[0]


def test_cuda_resolver_compares_maintained_and_codegen_target_spellings(
    monkeypatch, tmp_path
):
    generated, package, native, artifact, _ = _cuda_installed_package(
        tmp_path, mode="r0", include_expected=True
    )
    monkeypatch.setattr(installed.resources, "files", lambda name: package)

    result = installed.resolve_installed_oam_artifact(
        architecture=generated["architecture"],
        precision="float32",
        backend="cuda",
        target="sm120",
        required_capabilities=generated["required_capabilities"],
        native_module=native,
        artifact_id=generated["artifact_id"],
    )

    assert result.available
    assert result.artifact_path == artifact


def test_cuda_resolver_fails_closed_for_ambiguous_exact_oam(monkeypatch, tmp_path):
    generated, package, native, artifact, manifest = _cuda_installed_package(
        tmp_path, mode="full", include_expected=True
    )
    catalog_path = package / "_operator_artifacts/catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["artifacts"].append(dict(catalog["artifacts"][0]))
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    build_info = native._backend_build_info()["prebuilt_operator_artifacts"]
    build_info["catalog_sha256"] = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    build_info["artifact_count"] = 2
    build_info["installed_bytes"] = sum(
        path.stat().st_size for path in (catalog_path, artifact, manifest)
    )
    monkeypatch.setattr(installed.resources, "files", lambda name: package)

    result = installed.resolve_installed_oam_artifact(
        architecture=generated["architecture"],
        precision="float32",
        backend="cuda",
        target="sm120",
        required_capabilities=generated["required_capabilities"],
        native_module=native,
        artifact_id=generated["artifact_id"],
    )

    assert result.status == "rejected"
    assert not result.available


def test_resolver_selects_explicit_architecture_without_model_hash(
    monkeypatch, tmp_path
):
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    architecture = resolve_oam_architecture(contract)
    generated = oam_receiver_jit_metadata(
        contract, "float64", host_target="host-isa-x86-64-v3"
    )
    identity = {
        "schema": "symmetrix.generated-operator-identity/1",
        "operator": "OAM",
        "generation_revision": 19,
        "architecture": architecture,
        "architecture_fingerprint": oam_tensor_product_architecture_fingerprint(
            architecture
        ),
        "precision": "float64",
        "backend": "host",
        "target": "host-isa-x86-64-v3",
        "floating_point_policy": "strict",
        "generation_fingerprint": generated["generation_fingerprint"],
        "semantic_fingerprint": generated["semantic_fingerprint"],
        "structure_fingerprint": generated["structure_fingerprint"],
        "artifact_id": generated["artifact_id"],
    }
    package = tmp_path / "installed_backend"
    artifact = package / "_operator_artifacts/oam/fixture/plugin.so"
    manifest = package / "_operator_artifacts/oam/fixture/manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"installed-oam")
    manifest.write_text(
        json.dumps({"key_inputs": {"build": {"metadata": generated}}}),
        encoding="utf-8",
    )
    catalog = core.generated_operator_catalog(
        [
            {
                "operator": "OAM",
                "identity": identity,
                "binary_format": "shared-library",
                "installed_relative_path": "oam/fixture/plugin.so",
                "checksum": _sha(artifact),
                "abi": {"tag": generated["abi"], "version": generated["abi_version"]},
                "capabilities": generated["required_capabilities"],
                "build_provenance": {
                    "source_content_sha256": "sha256:" + "a" * 64,
                    "compiler_id": "GNU",
                    "compiler_version": "15",
                    "floating_point_policy": "strict",
                },
                "auxiliary_files": [
                    {
                        "role": "manifest",
                        "installed_relative_path": "oam/fixture/manifest.json",
                        "checksum": _sha(manifest),
                    }
                ],
            }
        ]
    )
    catalog_path = package / "_operator_artifacts/catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    build_info = {
        "mode": "r0",
        "backend": "host",
        "target": "host-isa-x86-64-v3",
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "artifact_count": 1,
        "installed_bytes": sum(
            path.stat().st_size for path in (catalog_path, artifact, manifest)
        ),
        "source_content_sha256": "a" * 64,
        "native_source_content_sha256": "b" * 64,
        "compiler_id": "GNU",
        "compiler_version": "15",
        "build_duration_seconds": 0.1,
    }
    native = SimpleNamespace(
        __package__="installed_backend",
        _backend_build_info=lambda: {"prebuilt_operator_artifacts": build_info},
    )
    monkeypatch.setattr(installed.resources, "files", lambda name: package)

    result = installed.resolve_installed_oam_artifact(
        architecture=architecture,
        precision="float64",
        backend="host",
        native_module=native,
        host_target="host-isa-x86-64-v3",
    )

    assert result.available
    assert result.artifact_path == artifact
    assert result.manifest_path == manifest

    wrong_policy = installed.resolve_installed_oam_artifact(
        architecture=architecture,
        precision="float64",
        backend="host",
        native_module=native,
        host_target="host-isa-x86-64-v3",
        floating_point_policy="fast",
    )
    assert wrong_policy.status == "rejected"
    assert "policy-or-artifact-identity-stale" in wrong_policy.diagnostics[0]
    assert "regenerate" in wrong_policy.diagnostics[0]

    relocated = installed.resolve_installed_oam_artifact(
        architecture=architecture,
        precision="float64",
        backend="host",
        native_module=native,
        host_target="host-sha256-incompatible",
    )
    assert not relocated.available
    assert relocated.status == "rejected"
    assert "host-target-incompatible" in relocated.diagnostics[0]
    assert "regenerate" in relocated.diagnostics[0]

    retrained = json.loads(FIXTURE.read_text(encoding="utf-8"))
    retrained["layers"][0]["pre_linear"]["weight"]["values"][0] += 0.25
    assert resolve_oam_architecture(retrained) == architecture


def test_resolver_rejects_a_different_explicit_architecture(monkeypatch, tmp_path):
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    architecture = resolve_oam_architecture(contract)
    generated = oam_receiver_jit_metadata(
        contract, "float64", host_target="host-isa-x86-64-v3"
    )
    package = tmp_path / "installed_backend"
    artifact = package / "_operator_artifacts/oam/fixture/plugin.so"
    manifest = package / "_operator_artifacts/oam/fixture/manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"installed-oam")
    manifest.write_text(
        json.dumps({"key_inputs": {"build": {"metadata": generated}}}),
        encoding="utf-8",
    )
    identity = {
        "schema": "symmetrix.generated-operator-identity/1",
        "operator": "OAM",
        "generation_revision": 19,
        "architecture": architecture,
        "architecture_fingerprint": oam_tensor_product_architecture_fingerprint(
            architecture
        ),
        "precision": "float64",
        "backend": "host",
        "target": "host-isa-x86-64-v3",
        "floating_point_policy": "strict",
        "generation_fingerprint": generated["generation_fingerprint"],
        "semantic_fingerprint": generated["semantic_fingerprint"],
        "structure_fingerprint": generated["structure_fingerprint"],
        "artifact_id": generated["artifact_id"],
    }
    catalog = core.generated_operator_catalog(
        [
            {
                "operator": "OAM",
                "identity": identity,
                "binary_format": "shared-library",
                "installed_relative_path": "oam/fixture/plugin.so",
                "checksum": _sha(artifact),
                "abi": {"tag": generated["abi"], "version": generated["abi_version"]},
                "capabilities": generated["required_capabilities"],
                "build_provenance": {
                    "source_content_sha256": "sha256:" + "a" * 64,
                    "compiler_id": "GNU",
                    "compiler_version": "15",
                    "floating_point_policy": "strict",
                },
                "auxiliary_files": [
                    {
                        "role": "manifest",
                        "installed_relative_path": "oam/fixture/manifest.json",
                        "checksum": _sha(manifest),
                    }
                ],
            }
        ]
    )
    catalog_path = package / "_operator_artifacts/catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    native = SimpleNamespace(
        __package__="installed_backend",
        _backend_build_info=lambda: {
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "host",
                "target": "host-isa-x86-64-v3",
                "catalog_relative_path": "_operator_artifacts/catalog.json",
                "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                "artifact_count": 1,
                "installed_bytes": sum(
                    path.stat().st_size for path in (catalog_path, artifact, manifest)
                ),
                "source_content_sha256": "a" * 64,
                "native_source_content_sha256": "b" * 64,
                "compiler_id": "GNU",
                "compiler_version": "15",
                "build_duration_seconds": 0.1,
            }
        },
    )
    monkeypatch.setattr(installed.resources, "files", lambda name: package)
    changed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    changed["layers"][0]["edge_tensor_product"]["instructions"][0]["path_weight"] *= 2.0

    result = installed.resolve_installed_oam_artifact(
        architecture=resolve_oam_architecture(changed),
        precision="float64",
        backend="host",
        native_module=native,
    )

    assert result.status == "absent"
    assert result.artifact_path is None
    assert "architecture-mismatch" in result.diagnostics[0]
