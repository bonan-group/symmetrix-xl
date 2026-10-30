import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "symmetrix"
sys.path.insert(0, str(PACKAGE.parents[2]))


def _core_module():
    try:
        return importlib.import_module("symmetrix.generated_operator_core")
    except ModuleNotFoundError:
        pytest.fail("the build-safe generated operator core is missing")


def test_generated_operator_core_has_only_standard_library_imports():
    source_path = PACKAGE / "generated_operator_core.py"
    assert source_path.is_file()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_roots.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imported_roots <= {
        "__future__",
        "collections",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "re",
        "stat",
        "typing",
    }


def _r0_identity_arguments():
    return {
        "generation_revision": 14,
        "compatibility_version": 2,
        "contract_fingerprint": "sha256:" + "a" * 64,
        "structure_fingerprint": "sha256:" + "b" * 64,
        "precision": "float32",
        "backend": "cuda",
        "target": "sm_120",
        "channels": 128,
        "l_max": 3,
        "edge_weight_mode": "unity",
        "schedule": "receiver-edge32",
        "schedule_revision": 4,
        "floating_point_policy": "cuda-fast-math",
    }


def _catalog_artifact(core):
    return {
        "operator": "R0",
        "identity": core.generated_r0_identity(**_r0_identity_arguments()),
        "binary_format": "cubin",
        "installed_relative_path": "artifacts/r0/cuda/operator.cubin",
        "checksum": "sha256:" + "c" * 64,
        "abi": {"tag": "symmetrix.jit.operator-module/1", "version": 5},
        "capabilities": 0x7F00,
        "build_provenance": {
            "source_content_sha256": "sha256:" + "d" * 64,
            "compiler_id": "NVRTC",
            "compiler_version": "13.3",
            "floating_point_policy": "cuda-fast-math",
        },
    }


def _full_catalog_artifacts(core):
    provenance = {
        "source_content_sha256": "sha256:" + "d" * 64,
        "compiler_id": "GNU",
        "compiler_version": "15.2.0",
    }
    m0 = core.generated_m0_identity(
        generation_revision=14,
        structure_fingerprint="sha256:" + "1" * 64,
        precision="float32",
        backend="host",
        target="x86-64-v3",
        channels=96,
        input_components=16,
        output_components=9,
        correlation=3,
        term_count=42,
        schedule="chunk32",
        schedule_revision=1,
    )
    revised_m0 = core.generated_m0_identity(
        **{
            key: value
            for key, value in m0.items()
            if key not in ("schema", "operator", "artifact_id", "schedule_revision")
        },
        schedule_revision=2,
    )
    assert revised_m0["artifact_id"] != m0["artifact_id"]
    r1 = core.generated_r1_identity(
        generation_revision=14,
        contract_fingerprint="sha256:" + "2" * 64,
        semantic_fingerprint="sha256:" + "3" * 64,
        structure_fingerprint="sha256:" + "4" * 64,
        precision="float64",
        backend="cuda",
        target="sm_120",
        channels=128,
        embedding=8,
        edge_l_max=3,
        source_l_max=2,
        output_components=16,
        edge_weight_mode="runtime",
        schedule="wave-w32-t32-b8",
        schedule_revision=1,
    )
    m1 = core.generated_m1_identity(
        generation_revision=18,
        structure_fingerprint="sha256:" + "7" * 64,
        precision="float32",
        backend="host",
        target="x86-64-v3",
        channels=96,
        input_components=16,
        correlation=3,
        term_count=94,
        schedule="chunk32",
        schedule_revision=1,
    )
    return [
        {
            "operator": "M0",
            "identity": m0,
            "binary_format": "shared-library",
            "installed_relative_path": "m0/operator.so",
            "checksum": "sha256:" + "5" * 64,
            "abi": {"tag": "symmetrix.jit.m0-host-plugin/2", "version": 2},
            "capabilities": 0x1F,
            "build_provenance": provenance,
        },
        {
            "operator": "M1",
            "identity": m1,
            "binary_format": "shared-library",
            "installed_relative_path": "m1/operator.so",
            "checksum": "sha256:" + "8" * 64,
            "abi": {"tag": "symmetrix.jit.m1-host-plugin/2", "version": 2},
            "capabilities": 0x1F,
            "build_provenance": provenance,
        },
        {
            "operator": "R1",
            "identity": r1,
            "binary_format": "cubin",
            "installed_relative_path": "r1/operator.cubin",
            "checksum": "sha256:" + "6" * 64,
            "abi": {"tag": "symmetrix.jit.cuda-plugin/4", "version": 4},
            "capabilities": 0x3,
            "build_provenance": provenance,
        },
    ]


def _installed_catalog_fixture(core, package: Path):
    artifact_path = package / "_operator_artifacts/r0/operator.cubin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"relocatable operator")
    artifact = _catalog_artifact(core)
    artifact["installed_relative_path"] = "r0/operator.cubin"
    artifact["checksum"] = (
        "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )
    catalog = core.generated_operator_catalog([artifact])
    catalog_path = package / "_operator_artifacts/catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    return {
        "mode": "r0",
        "backend": "cuda",
        "target": "sm_120",
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": catalog_sha256,
        "artifact_count": 1,
        "installed_bytes": catalog_path.stat().st_size + artifact_path.stat().st_size,
        "source_content_sha256": "d" * 64,
        "native_source_content_sha256": "e" * 64,
        "compiler_id": "NVRTC",
        "compiler_version": "13.3",
        "build_duration_seconds": 0.25,
    }


def test_generated_r0_identity_is_complete_and_origin_independent():
    core = _core_module()
    arguments = _r0_identity_arguments()

    identity = core.generated_r0_identity(**arguments)

    assert identity == core.generated_r0_identity(**arguments)
    assert "origin" not in identity
    assert identity["contract_fingerprint"] == arguments["contract_fingerprint"]
    assert identity["structure_fingerprint"] == arguments["structure_fingerprint"]
    assert identity["harmonic_count"] == 16
    assert identity["channels"] == 128
    assert identity["target"] == "sm_120"
    assert identity["edge_weight_mode"] == "unity"
    assert identity["compatibility_version"] == 2
    assert identity["schedule_revision"] == 4
    assert identity["artifact_id"].startswith("generated-r0-gen14-cuda-f32-")


def test_generated_m1_identity_is_distinct_and_schedule_sensitive():
    core = _core_module()
    arguments = {
        "generation_revision": 18,
        "structure_fingerprint": "sha256:" + "7" * 64,
        "precision": "float32",
        "backend": "cuda",
        "target": "sm_120",
        "channels": 128,
        "input_components": 16,
        "correlation": 3,
        "term_count": 94,
        "schedule": "node-channel-owner",
        "schedule_revision": 1,
    }

    identity = core.generated_m1_identity(**arguments)

    assert core.validate_generated_m1_identity(identity) == identity
    assert identity["operator"] == "M1"
    assert identity["artifact_id"].startswith("generated-m1-gen18-cuda-f32-")
    changed = core.generated_m1_identity(**{**arguments, "schedule_revision": 2})
    assert changed["artifact_id"] != identity["artifact_id"]

    wrong_operator = {**identity, "operator": "M0"}
    with pytest.raises(ValueError):
        core.validate_generated_operator_identity(wrong_operator)


def test_full_catalog_validates_exact_m0_m1_and_r1_identities_and_abis():
    core = _core_module()

    catalog = core.generated_operator_catalog(_full_catalog_artifacts(core))

    assert {entry["operator"] for entry in catalog["artifacts"]} == {
        "M0",
        "M1",
        "R1",
    }
    for entry in catalog["artifacts"]:
        assert (
            core.validate_generated_operator_identity(entry["identity"])
            == entry["identity"]
        )


@pytest.mark.parametrize("operator", ("M0", "M1"))
@pytest.mark.parametrize(
    ("backend", "target", "binary_format"),
    (("cuda", "sm_120", "cubin"), ("hip", "gfx1151", "code-object")),
)
def test_device_product_catalog_requires_operator_module_abi_v3(
    operator, backend, target, binary_format
):
    core = _core_module()
    common = {
        "generation_revision": 18,
        "structure_fingerprint": "sha256:" + "9" * 64,
        "precision": "float32",
        "backend": backend,
        "target": target,
        "channels": 128,
        "input_components": 16,
        "correlation": 3,
        "term_count": 42,
        "schedule": "chunk32",
        "schedule_revision": 1,
    }
    identity = (
        core.generated_m0_identity(**common, output_components=9)
        if operator == "M0"
        else core.generated_m1_identity(**common)
    )
    artifact = {
        "operator": operator,
        "identity": identity,
        "binary_format": binary_format,
        "installed_relative_path": f"{operator.lower()}/operator.{binary_format}",
        "checksum": "sha256:" + "a" * 64,
        "abi": {"tag": "symmetrix.jit.operator-module/1", "version": 3},
        "capabilities": 0x1F,
        "build_provenance": {
            "source_content_sha256": "sha256:" + "b" * 64,
            "compiler_id": "fixture-compiler",
            "compiler_version": "1",
        },
    }

    catalog = core.generated_operator_catalog([artifact])
    assert catalog["artifacts"] == [artifact]

    artifact["abi"] = {"tag": "symmetrix.jit.operator-module/1", "version": 2}
    with pytest.raises(ValueError, match="version 3"):
        core.generated_operator_catalog([artifact])


@pytest.mark.parametrize("operator", ("M0", "M1", "R1"))
def test_full_catalog_rejects_tampered_exact_identity(operator):
    core = _core_module()
    artifacts = _full_catalog_artifacts(core)
    artifact = next(item for item in artifacts if item["operator"] == operator)
    artifact["identity"] = {**artifact["identity"], "artifact_id": "tampered"}

    with pytest.raises(ValueError, match="artifact_id"):
        core.generated_operator_catalog(artifacts)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("generation_revision", 5),
        ("compatibility_version", 3),
        ("contract_fingerprint", "sha256:" + "e" * 64),
        ("structure_fingerprint", "sha256:" + "f" * 64),
        ("precision", "float64"),
        ("backend", "hip"),
        ("target", "sm_100"),
        ("channels", 96),
        ("l_max", 2),
        ("edge_weight_mode", "weighted"),
        ("schedule", "receiver-edge16"),
        ("schedule_revision", 3),
        ("floating_point_policy", "strict"),
    ),
)
def test_generated_r0_identity_is_sensitive_to_every_dimension(field, replacement):
    core = _core_module()
    baseline_arguments = _r0_identity_arguments()
    changed_arguments = {**baseline_arguments, field: replacement}
    if field == "backend":
        changed_arguments["floating_point_policy"] = "strict"

    baseline = core.generated_r0_identity(**baseline_arguments)
    changed = core.generated_r0_identity(**changed_arguments)

    assert changed["artifact_id"] != baseline["artifact_id"]


def test_generated_r0_identity_rejects_tampered_derived_fields():
    core = _core_module()
    identity = core.generated_r0_identity(**_r0_identity_arguments())

    for field, replacement in (("harmonic_count", 9), ("artifact_id", "other")):
        tampered = {**identity, field: replacement}
        with pytest.raises(ValueError, match=field):
            core.validate_generated_r0_identity(tampered)


def test_runtime_r0_metadata_publishes_the_canonical_generated_identity():
    core = _core_module()
    codegen = importlib.import_module("symmetrix.mace_jit_operator_codegen")
    contract_path = Path(__file__).parent / "data" / "execution_contracts"
    contract = json.loads((contract_path / "standard_r0_contract.json").read_text())

    host = codegen.jit_r0_host_metadata(
        contract, precision="float32", target="x86-64-v3"
    )
    device = codegen.jit_r0_device_metadata(
        contract, precision="float32", target="sm_120"
    )

    common = {
        "generation_revision": codegen.JIT_GENERATION_VERSION,
        "compatibility_version": 2,
        "contract_fingerprint": contract["generation_fingerprint"],
        "structure_fingerprint": contract["structure_fingerprint"],
        "precision": "float32",
        "channels": contract["channels"],
        "l_max": contract["edge_harmonics"]["l_max"],
        "edge_weight_mode": "unity",
    }
    assert host["generated_identity"] == core.generated_r0_identity(
        **common,
        backend="host",
        target="x86-64-v3",
        schedule="host-owner",
        schedule_revision=4,
    )
    assert device["generated_identity"] == core.generated_r0_identity(
        **common,
        backend="cuda",
        target="sm_120",
        schedule="receiver-edge16",
        schedule_revision=6,
    )
    assert host["artifact_id"] == host["generated_identity"]["artifact_id"]
    assert device["artifact_id"] == device["generated_identity"]["artifact_id"]
    assert host["legacy_artifact_id"].startswith("jit-r0-host-gen")
    assert device["legacy_artifact_id"].startswith("jit-r0-gen")


@pytest.mark.parametrize(
    "backend,target,schedule,weighted",
    (
        ("host", "x86-64-v3", "host-owner", True),
        ("cuda", "sm_120", "receiver-edge16", False),
        ("hip", "gfx1151:sramecc-:xnack-", "receiver-edge32", True),
    ),
)
def test_runtime_and_build_codegen_adapters_are_byte_identical(
    backend, target, schedule, weighted
):
    runtime = importlib.import_module("symmetrix.mace_jit_operator_codegen")
    build = importlib.import_module("tools._symmetrix_build.generated_operator_codegen")
    contract_path = Path(__file__).parent / "data" / "execution_contracts"
    contract = json.loads((contract_path / "standard_r0_contract.json").read_text())
    arguments = {
        "generation_version": 13,
        "source_content_sha256": "a" * 64,
        "backend": backend,
        "target": target,
        "precision": "float64",
        "schedule": schedule,
        "apply_edge_weights": weighted,
    }

    runtime_source, runtime_metadata = runtime.render_generated_r0_operator(
        contract, **arguments
    )
    build_source, build_metadata = build.render_generated_r0_operator(
        contract, **arguments
    )

    assert runtime_source.encode("utf-8") == build_source.encode("utf-8")
    assert json.dumps(
        runtime_metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii") == json.dumps(
        build_metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    assert runtime_metadata["artifact_id"] == build_metadata["artifact_id"]
    assert (
        runtime_metadata["generated_identity"]["artifact_id"]
        == build_metadata["generated_identity"]["artifact_id"]
    )
    assert runtime_metadata["source_content_sha256"] == "a" * 64
    assert runtime_metadata["target"] == target
    assert runtime_metadata["schedule"] == schedule


def test_build_codegen_adapter_import_is_runtime_free():
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    script = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from tools._symmetrix_build import generated_operator_codegen as codegen

contract = json.loads(Path(sys.argv[2]).read_text())
codegen.render_generated_r0_operator(
    contract,
    generation_version=13,
    source_content_sha256="a" * 64,
    backend="host",
    target="x86-64-v3",
    precision="float64",
    schedule="host-owner",
)

forbidden = {
    name
    for name in sys.modules
    if name == "numpy"
    or name.startswith("numpy.")
    or name.startswith("symmetrix.backend_loader")
    or name.startswith("symmetrix.jit")
    or name.startswith("symmetrix._native")
}
if forbidden:
    raise SystemExit("unexpected imports: " + ", ".join(sorted(forbidden)))
if "symmetrix" in sys.modules:
    raise SystemExit("build adapter executed symmetrix.__init__")
"""

    subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            script,
            str(root),
            str(
                Path(__file__).parent
                / "data"
                / "execution_contracts"
                / "standard_r0_contract.json"
            ),
        ),
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("contract_fingerprint", "sha256:short"),
        ("precision", "float16"),
        ("backend", "metal"),
        ("channels", 0),
        ("l_max", -1),
        ("edge_weight_mode", "optional"),
        ("schedule_revision", 0),
        ("compatibility_version", 0),
    ),
)
def test_generated_r0_identity_rejects_incomplete_dimensions(field, value):
    core = _core_module()
    arguments = {
        **_r0_identity_arguments(),
        "precision": "float64",
        "backend": "host",
        "target": "x86-64-v3",
        "channels": 96,
        "edge_weight_mode": "weighted",
        "schedule": "host-tile8",
        "schedule_revision": 1,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        core.generated_r0_identity(**arguments)


def test_generated_operator_catalog_is_versioned_deterministic_and_validated():
    core = _core_module()
    artifact = _catalog_artifact(core)

    catalog = core.generated_operator_catalog([artifact])

    assert catalog == core.generated_operator_catalog([deepcopy(artifact)])
    assert catalog == core.validate_generated_operator_catalog(catalog)
    assert catalog["schema"] == "symmetrix.generated-operator-catalog/1"
    assert catalog["artifacts"][0] == artifact
    assert "origin" not in catalog["artifacts"][0]


def test_generated_r0_catalog_rejects_floating_policy_provenance_mismatch():
    core = _core_module()
    artifact = _catalog_artifact(core)
    artifact["build_provenance"]["floating_point_policy"] = "strict"

    with pytest.raises(ValueError, match="floating-point policy"):
        core.generated_operator_catalog([artifact])


def test_generated_operator_catalog_admits_only_current_host_compatibility():
    core = _core_module()
    artifact = _catalog_artifact(core)
    artifact["identity"] = core.generated_r0_identity(
        **{
            **_r0_identity_arguments(),
            "backend": "host",
            "target": "x86-64-v3",
            "schedule": "host-owner",
            "floating_point_policy": "strict",
        }
    )
    artifact["binary_format"] = "shared-library"
    artifact["installed_relative_path"] = "artifacts/r0/host/operator.so"
    artifact["abi"] = {"tag": "symmetrix.jit.r0-host-plugin/2", "version": 2}
    artifact["build_provenance"]["floating_point_policy"] = "strict"

    assert core.generated_operator_catalog([artifact])["artifacts"] == [artifact]
    artifact["abi"] = {"tag": "symmetrix.jit.r0-host-plugin/1", "version": 1}
    with pytest.raises(ValueError, match="abi"):
        core.generated_operator_catalog([artifact])


@pytest.mark.parametrize("origin", ("prebuilt", "cache", "runtime"))
def test_generated_operator_origin_is_reported_outside_identity(origin):
    core = _core_module()
    identity = core.generated_r0_identity(**_r0_identity_arguments())

    report = core.generated_operator_origin_record(identity, origin)

    assert report == {"artifact_id": identity["artifact_id"], "origin": origin}
    assert "origin" not in identity


def test_generated_operator_origin_rejects_unknown_values():
    core = _core_module()
    identity = core.generated_r0_identity(**_r0_identity_arguments())

    with pytest.raises(ValueError, match="origin"):
        core.generated_operator_origin_record(identity, "installed")


@pytest.mark.parametrize(
    "path",
    (
        "/absolute/operator.cubin",
        "../operator.cubin",
        "artifacts/../../operator.cubin",
        "artifacts/./operator.cubin",
        "artifacts\\operator.cubin",
        "C:/operator.cubin",
        "C:payload/operator.cubin",
        ".",
        "artifacts/\0operator.cubin",
        "",
    ),
)
def test_generated_operator_catalog_rejects_unsafe_installed_paths(path):
    core = _core_module()
    artifact = _catalog_artifact(core)
    artifact["installed_relative_path"] = path

    with pytest.raises(ValueError, match="installed_relative_path"):
        core.generated_operator_catalog([artifact])


@pytest.mark.parametrize(
    "missing_field",
    (
        "operator",
        "identity",
        "binary_format",
        "installed_relative_path",
        "checksum",
        "abi",
        "capabilities",
        "build_provenance",
    ),
)
def test_generated_operator_catalog_rejects_incomplete_artifacts(missing_field):
    core = _core_module()
    artifact = _catalog_artifact(core)
    artifact.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        core.generated_operator_catalog([artifact])


@pytest.mark.parametrize(
    "abi",
    (
        {"tag": "symmetrix.jit.r0-host-plugin/1", "version": 1},
        {"tag": "symmetrix.jit.operator-module/1", "version": 1},
    ),
)
def test_generated_operator_catalog_rejects_backend_incompatible_abi(abi):
    core = _core_module()
    artifact = _catalog_artifact(core)
    artifact["abi"] = abi

    with pytest.raises(ValueError, match="abi"):
        core.generated_operator_catalog([artifact])


def test_generated_operator_catalog_rejects_malformed_catalogs():
    core = _core_module()
    artifact = _catalog_artifact(core)
    malformed = []

    wrong_operator = deepcopy(artifact)
    wrong_operator["operator"] = "M0"
    malformed.append(wrong_operator)
    wrong_format = deepcopy(artifact)
    wrong_format["binary_format"] = "shared-library"
    malformed.append(wrong_format)
    wrong_checksum = deepcopy(artifact)
    wrong_checksum["checksum"] = "sha256:short"
    malformed.append(wrong_checksum)
    incomplete_abi = deepcopy(artifact)
    incomplete_abi["abi"].pop("version")
    malformed.append(incomplete_abi)
    incomplete_provenance = deepcopy(artifact)
    incomplete_provenance["build_provenance"].pop("compiler_version")
    malformed.append(incomplete_provenance)
    tampered_identity = deepcopy(artifact)
    tampered_identity["identity"]["harmonic_count"] = 9
    malformed.append(tampered_identity)

    for malformed_artifact in malformed:
        with pytest.raises(ValueError):
            core.generated_operator_catalog([malformed_artifact])

    with pytest.raises(ValueError, match="schema"):
        core.validate_generated_operator_catalog(
            {"schema": "symmetrix.generated-operator-catalog/2", "artifacts": []}
        )
    with pytest.raises(ValueError, match="iterable"):
        core.generated_operator_catalog(None)


def test_installed_operator_catalog_validation_survives_package_relocation(tmp_path):
    core = _core_module()
    package = tmp_path / "original/symmetrix"
    package.mkdir(parents=True)
    native = package / "_native_cuda13_sm120.so"
    native.write_bytes(b"native")
    build_info = _installed_catalog_fixture(core, package)
    relocated = tmp_path / "relocated/symmetrix"
    relocated.parent.mkdir()
    package.rename(relocated)

    report = core.validate_installed_operator_artifacts(
        relocated / native.name, build_info
    )

    assert report["status"] == "ok"
    assert report["artifact_count"] == 1
    assert report["installed_bytes"] == build_info["installed_bytes"]
    assert report["catalog_sha256"] == build_info["catalog_sha256"]
    assert report["catalog_path"] == str(relocated / "_operator_artifacts/catalog.json")


@pytest.mark.parametrize("tamper", ("binary", "missing", "catalog", "extra"))
def test_installed_operator_catalog_validation_rejects_tampering(tmp_path, tamper):
    core = _core_module()
    package = tmp_path / "symmetrix"
    package.mkdir()
    native = package / "_native_cuda13_sm120.so"
    native.write_bytes(b"native")
    build_info = _installed_catalog_fixture(core, package)
    artifact = package / "_operator_artifacts/r0/operator.cubin"
    if tamper == "binary":
        artifact.write_bytes(b"tampered")
    elif tamper == "missing":
        artifact.unlink()
    elif tamper == "catalog":
        catalog = package / "_operator_artifacts/catalog.json"
        catalog.write_bytes(catalog.read_bytes() + b" ")
    else:
        (package / "_operator_artifacts/unrecorded.json").write_text("{}")

    with pytest.raises(ValueError, match="catalog|checksum|missing|membership|bytes"):
        core.validate_installed_operator_artifacts(native, build_info)


def test_build_provenance_does_not_change_generated_operator_identity():
    core = _core_module()
    artifact = _catalog_artifact(core)
    other = deepcopy(artifact)
    other["build_provenance"] = {
        "source_content_sha256": "sha256:" + "e" * 64,
        "compiler_id": "nvcc",
        "compiler_version": "13.3.1",
        "floating_point_policy": "cuda-fast-math",
    }

    first = core.generated_operator_catalog([artifact])["artifacts"][0]
    second = core.generated_operator_catalog([other])["artifacts"][0]

    assert first["identity"] == second["identity"]
