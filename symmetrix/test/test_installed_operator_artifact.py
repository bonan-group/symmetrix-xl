import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from symmetrix import generated_operator_core as operator_core
from symmetrix import installed_operator_artifact as installed


def _identity(*, target="x86-64-v3", channels=96):
    return operator_core.generated_r0_identity(
        generation_revision=14,
        compatibility_version=2,
        contract_fingerprint="sha256:" + "1" * 64,
        structure_fingerprint="sha256:" + "2" * 64,
        precision="float32",
        backend="host",
        target=target,
        channels=channels,
        l_max=3,
        edge_weight_mode="unity",
        schedule="host-owner",
        schedule_revision=4,
    )


def _installed_package(tmp_path, *, identity=None, capabilities=0x3F):
    package = tmp_path / "installed_backend"
    package.mkdir()
    artifact = package / "_operator_artifacts/r0/operator.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"immutable generated operator")
    identity = _identity() if identity is None else identity
    catalog = operator_core.generated_operator_catalog(
        [
            {
                "operator": "R0",
                "identity": identity,
                "binary_format": "shared-library",
                "installed_relative_path": "r0/operator.so",
                "checksum": "sha256:"
                + hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "abi": {"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
                "capabilities": capabilities,
                "build_provenance": {
                    "source_content_sha256": "sha256:" + "a" * 64,
                    "compiler_id": "GNU",
                    "compiler_version": "15.2.0",
                    "floating_point_policy": "strict",
                },
            }
        ]
    )
    catalog_path = package / "_operator_artifacts/catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_info = {
        "mode": "r0",
        "backend": "host",
        "target": identity["target"],
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "artifact_count": 1,
        "installed_bytes": catalog_path.stat().st_size + artifact.stat().st_size,
        "source_content_sha256": "a" * 64,
        "native_source_content_sha256": "b" * 64,
        "compiler_id": "GNU",
        "compiler_version": "15.2.0",
        "build_duration_seconds": 0.5,
    }
    native = SimpleNamespace(
        __package__="installed_backend",
        _backend_build_info=lambda: {"prebuilt_operator_artifacts": build_info},
    )
    return package, artifact, build_info, native


@pytest.fixture(autouse=True)
def _clear_rejections():
    installed._clear_installed_operator_rejections_for_tests()
    yield
    installed._clear_installed_operator_rejections_for_tests()


def test_exact_installed_artifact_loads_directly_without_cache_writes(
    monkeypatch, tmp_path
):
    package, artifact, _, native = _installed_package(tmp_path)
    cache = tmp_path / "monitored-cache"
    cache.mkdir()
    before = tuple(cache.rglob("*"))
    loaded = []
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)

    result = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=lambda path: loaded.append(Path(path)),
        native_module=native,
    )

    assert result.status == "resolved"
    assert result.origin == "prebuilt"
    assert result.artifact_path == artifact
    assert result.artifact_id == _identity()["artifact_id"]
    assert result.diagnostics == (
        "origin=prebuilt status=loaded operator=R0 "
        f"artifact_id={_identity()['artifact_id']}",
    )
    assert loaded == [artifact]
    assert tuple(cache.rglob("*")) == before


def test_composite_editable_package_resolves_concrete_artifact_subtree(
    monkeypatch, tmp_path
):
    package, artifact, _, native = _installed_package(tmp_path)

    class CompositePackage:
        def joinpath(self, *parts):
            return package.joinpath(*parts)

    monkeypatch.setattr(
        installed.resources, "files", lambda package_name: CompositePackage()
    )
    loaded = []

    result = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=lambda path: loaded.append(Path(path)),
        native_module=native,
    )

    assert result.status == "resolved"
    assert result.artifact_path == artifact
    assert loaded == [artifact]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("absent", "catalog-absent"),
        ("tampered", "checksum-mismatch"),
        ("identity", "identity-mismatch"),
        ("target", "backend-target-mismatch"),
        ("abi", "abi-mismatch"),
        ("capabilities", "capability-mismatch"),
    ),
)
def test_installed_resolution_reports_deterministic_rejections(
    monkeypatch, tmp_path, mutation, reason
):
    package, artifact, build_info, native = _installed_package(tmp_path)
    requested_identity = _identity()
    requested_abi = {"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2}
    required_capabilities = 0x3F
    if mutation == "absent":
        build_info["mode"] = "off"
        build_info.update(
            catalog_relative_path="",
            catalog_sha256="",
            artifact_count=0,
            installed_bytes=0,
            backend="",
            target="",
            source_content_sha256="",
            native_source_content_sha256="",
            compiler_id="",
            compiler_version="",
            build_duration_seconds=0.0,
        )
    elif mutation == "tampered":
        artifact.write_bytes(b"tampered")
    elif mutation == "identity":
        requested_identity = _identity(channels=128)
    elif mutation == "target":
        requested_identity = _identity(target="host-sha256-" + "1" * 64)
    elif mutation == "abi":
        requested_abi = {"tag": "symmetrix.jit.r0-host-plugin/2", "version": 3}
    else:
        required_capabilities = 0x80
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)

    result = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=requested_identity,
        abi=requested_abi,
        required_capabilities=required_capabilities,
        loader=lambda path: None,
        native_module=native,
    )

    assert result.status in {"absent", "rejected"}
    assert result.origin is None
    assert len(result.diagnostics) == 1
    assert f"reason={reason}" in result.diagnostics[0]


def test_installed_resolution_rejects_surplus_alias_capability(monkeypatch, tmp_path):
    package, _, _, native = _installed_package(tmp_path, capabilities=0x1F)
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)

    result = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x1B,
        loader=lambda path: pytest.fail(f"unexpected load: {path}"),
        native_module=native,
    )

    assert result.status == "rejected"
    assert "reason=capability-mismatch" in result.diagnostics[0]


def test_unloadable_installed_artifact_is_skipped_for_process(monkeypatch, tmp_path):
    package, _, _, native = _installed_package(tmp_path)
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)
    attempts = 0

    def reject(_path):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("native ABI rejected the image")

    first = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=reject,
        native_module=native,
    )
    second = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=reject,
        native_module=native,
    )

    assert attempts == 1
    assert first.status == second.status == "rejected"
    assert "reason=load-failed" in first.diagnostics[0]
    assert "reason=previously-rejected" in second.diagnostics[0]


def test_loaded_but_unretained_installed_artifact_is_skipped_for_process(
    monkeypatch, tmp_path
):
    package, _, _, native = _installed_package(tmp_path)
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)
    loads = []

    first = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=lambda path: loads.append(path),
        admitted=lambda: False,
        native_module=native,
    )
    second = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=lambda path: loads.append(path),
        admitted=lambda: True,
        native_module=native,
    )

    assert len(loads) == 1
    assert first.status == second.status == "rejected"
    assert "reason=admission-failed" in first.diagnostics[0]
    assert "reason=previously-rejected" in second.diagnostics[0]


def test_r1_and_m0_use_the_same_absent_resolution_api(monkeypatch, tmp_path):
    package, _, _, native = _installed_package(tmp_path)
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)

    for operator in ("R1", "M0"):
        result = installed.resolve_installed_operator_artifact(
            operator=operator,
            identity={"artifact_id": f"generated-{operator.lower()}"},
            abi={"tag": f"symmetrix.jit.{operator.lower()}/1", "version": 1},
            required_capabilities=0x3F,
            loader=lambda path: pytest.fail(f"unexpected {operator} load: {path}"),
            native_module=native,
        )
        assert result.status == "absent"
        assert f"operator={operator}" in result.diagnostics[0]
        assert "reason=identity-mismatch" in result.diagnostics[0]


def test_full_catalog_resolves_exact_m0_and_r1_entries(monkeypatch, tmp_path):
    package = tmp_path / "installed_full_backend"
    data_root = package / "_operator_artifacts"
    data_root.mkdir(parents=True)
    identities = {
        "M0": operator_core.generated_m0_identity(
            generation_revision=14,
            structure_fingerprint="sha256:" + "3" * 64,
            precision="float32",
            backend="host",
            target="x86-64-v3",
            channels=96,
            input_components=16,
            output_components=9,
            correlation=3,
            term_count=42,
            schedule="chunk32",
            schedule_revision=2,
        ),
        "R1": operator_core.generated_r1_identity(
            generation_revision=14,
            contract_fingerprint="sha256:" + "4" * 64,
            semantic_fingerprint="sha256:" + "5" * 64,
            structure_fingerprint="sha256:" + "6" * 64,
            precision="float32",
            backend="host",
            target="x86-64-v3",
            channels=96,
            embedding=8,
            edge_l_max=3,
            source_l_max=2,
            output_components=16,
            edge_weight_mode="unit",
            schedule="host-owner",
            schedule_revision=1,
        ),
    }
    abis = {
        "M0": {"tag": "symmetrix.jit.m0-host-plugin/2", "version": 2},
        "R1": {"tag": "symmetrix.jit.host-plugin/4", "version": 4},
    }
    capabilities = {"M0": 0x1B, "R1": 0xF}
    entries = []
    paths = {}
    for operator in ("M0", "R1"):
        path = data_root / operator.lower() / "operator.so"
        path.parent.mkdir()
        path.write_bytes(f"compiled-{operator}".encode())
        paths[operator] = path
        entries.append(
            {
                "operator": operator,
                "identity": identities[operator],
                "binary_format": "shared-library",
                "installed_relative_path": f"{operator.lower()}/operator.so",
                "checksum": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                "abi": abis[operator],
                "capabilities": capabilities[operator],
                "build_provenance": {
                    "source_content_sha256": "sha256:" + "a" * 64,
                    "compiler_id": "GNU",
                    "compiler_version": "15.2.0",
                },
            }
        )
    catalog = operator_core.generated_operator_catalog(entries)
    catalog_path = data_root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    build_info = {
        "mode": "full",
        "backend": "host",
        "target": "x86-64-v3",
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "artifact_count": 2,
        "installed_bytes": sum(path.stat().st_size for path in paths.values())
        + catalog_path.stat().st_size,
        "source_content_sha256": "a" * 64,
        "native_source_content_sha256": "b" * 64,
        "compiler_id": "GNU",
        "compiler_version": "15.2.0",
        "build_duration_seconds": 0.5,
    }
    native = SimpleNamespace(
        __package__="installed_full_backend",
        _backend_build_info=lambda: {"prebuilt_operator_artifacts": build_info},
    )
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)
    loaded = []

    for operator in ("M0", "R1"):
        result = installed.resolve_installed_operator_artifact(
            operator=operator,
            identity=identities[operator],
            abi=abis[operator],
            required_capabilities=capabilities[operator],
            loader=lambda path, operator=operator: loaded.append(
                (operator, Path(path))
            ),
            admitted=lambda: True,
            native_module=native,
        )
        assert result.status == "resolved"
        assert result.artifact_path == paths[operator]

    assert loaded == [("M0", paths["M0"]), ("R1", paths["R1"])]


@pytest.mark.parametrize("policy", ("prefer", "required"))
def test_prefer_and_required_policies_resolve_exact_installed_artifact(
    monkeypatch, tmp_path, policy
):
    package, artifact, _, native = _installed_package(tmp_path)
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)

    result = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=lambda path: None,
        native_module=native,
        policy=policy,
    )

    assert result.status == "resolved"
    assert result.artifact_path == artifact


def test_disabled_resolution_does_not_read_package_resources(monkeypatch):
    monkeypatch.setattr(
        installed.resources,
        "files",
        lambda package_name: pytest.fail("disabled resolution read package data"),
    )

    result = installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=_identity(),
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=lambda path: pytest.fail(f"unexpected load: {path}"),
        native_module=SimpleNamespace(__package__="unused"),
        policy="disabled",
    )

    assert result.status == "disabled"
    assert result.diagnostics == (
        "origin=prebuilt status=skipped operator=R0 reason=disabled",
    )

    with pytest.raises(ValueError, match="policy"):
        installed.resolve_installed_operator_artifact(
            operator="R0",
            identity=_identity(),
            abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
            required_capabilities=0x3F,
            loader=lambda path: None,
            native_module=SimpleNamespace(__package__="unused"),
            policy="automatic",
        )


def test_catalog_request_objects_are_not_mutated(monkeypatch, tmp_path):
    package, _, _, native = _installed_package(tmp_path)
    monkeypatch.setattr(installed.resources, "files", lambda package_name: package)
    identity = _identity()
    original = deepcopy(identity)

    installed.resolve_installed_operator_artifact(
        operator="R0",
        identity=identity,
        abi={"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2},
        required_capabilities=0x3F,
        loader=lambda path: None,
        native_module=native,
    )

    assert identity == original
