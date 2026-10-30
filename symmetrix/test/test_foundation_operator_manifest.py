import ast
import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "symmetrix"
CONTRACTS = Path(__file__).resolve().parent / "data" / "execution_contracts"


def _manifest_module():
    try:
        return importlib.import_module("symmetrix.foundation_operator_manifest")
    except ModuleNotFoundError:
        pytest.fail("the production foundation operator manifest is missing")


def test_production_manifest_has_five_complete_bundles_and_separate_aliases():
    module = _manifest_module()

    manifest = module.load_foundation_operator_manifest()

    assert manifest["schema"] == "symmetrix.foundation-operator-manifest/1"
    assert manifest["version"] == 1
    assert [bundle["id"] for bundle in manifest["bundles"]] == [
        "off23-l0",
        "off23-l2",
        "standard-l0",
        "standard-l1",
        "standard-l2",
    ]
    assert [model["id"] for model in manifest["models"]] == [
        "mace-mp-0b2-large",
        "mace-mpa-0-medium",
        "mace-off23-large",
        "mace-off23-small",
        "mace-omat-0-small",
    ]
    assert manifest["aliases"] == [
        {"id": "mace-mp-0b3-medium", "target": "mace-mpa-0-medium"},
        {"id": "mace-omat-0-medium", "target": "mace-mpa-0-medium"},
    ]
    assert all(
        set(bundle["operators"]) == {"M0", "R0", "R1"} for bundle in manifest["bundles"]
    )
    assert all("aliases" not in model for model in manifest["models"])
    assert all(
        "artifact_id" not in json.dumps(bundle) for bundle in manifest["bundles"]
    )


def test_operator_descriptors_match_canonical_contract_evidence():
    module = _manifest_module()
    from symmetrix.mace_execution_contract import normalize_standard_m0_contract

    manifest = module.load_foundation_operator_manifest()
    bundles = {bundle["id"]: bundle for bundle in manifest["bundles"]}
    standard_m0 = normalize_standard_m0_contract(
        json.loads((CONTRACTS / "standard_m0_contract.json").read_text())
    )
    descriptor = bundles["standard-l1"]["operators"]["M0"]

    assert descriptor["structure_fingerprint"] == standard_m0["structure_fingerprint"]
    assert descriptor["term_count"] == sum(
        group["term_count"] for group in standard_m0["monomial_groups"]
    )

    requests = module.foundation_r0_artifact_inputs(
        manifest,
        generation_revision=14,
        compatibility_version=2,
        backend="host",
        target="x86-64-v3",
        precisions=("float64",),
        schedule="host-owner",
        schedule_revision=4,
    )
    c128 = next(
        request for request in requests if request["identity"]["channels"] == 128
    )
    assert c128["contract"] == json.loads(
        (CONTRACTS / "standard_r0_contract.json").read_text()
    )


def test_foundation_r0_requests_deduplicate_groups_by_canonical_identity():
    module = _manifest_module()
    core = importlib.import_module("symmetrix.generated_operator_core")
    manifest = module.load_foundation_operator_manifest()

    requests = module.foundation_r0_artifact_inputs(
        manifest,
        generation_revision=14,
        compatibility_version=2,
        backend="host",
        target="x86-64-v3",
        precisions=("float64", "float32"),
        schedule="host-owner",
        schedule_revision=4,
    )

    assert len(requests) == 6
    assert {
        (request["identity"]["channels"], request["identity"]["precision"])
        for request in requests
    } == {
        (96, "float32"),
        (96, "float64"),
        (128, "float32"),
        (128, "float64"),
        (224, "float32"),
        (224, "float64"),
    }
    assert len({request["identity"]["artifact_id"] for request in requests}) == 6
    for request in requests:
        contract = request["contract"]
        identity = request["identity"]
        assert identity == core.generated_r0_identity(
            generation_revision=14,
            compatibility_version=2,
            contract_fingerprint=contract["generation_fingerprint"],
            structure_fingerprint=contract["structure_fingerprint"],
            precision=identity["precision"],
            backend="host",
            target="x86-64-v3",
            channels=contract["channels"],
            l_max=contract["edge_harmonics"]["l_max"],
            edge_weight_mode="unity",
            schedule="host-owner",
            schedule_revision=4,
        )
    c128 = next(
        request for request in requests if request["identity"]["channels"] == 128
    )
    assert c128["bundle_ids"] == ["standard-l0", "standard-l1", "standard-l2"]


def test_foundation_m1_requests_deduplicate_known_channel_profiles():
    module = _manifest_module()
    manifest = module.load_foundation_operator_manifest()

    requests = module.foundation_m1_artifact_inputs(
        manifest,
        generation_revision=18,
        backend="host",
        target="x86-64-v3",
        precisions=("float32", "float64"),
    )

    assert len(requests) == 6
    assert {
        (request["identity"]["channels"], request["identity"]["precision"])
        for request in requests
    } == {
        (96, "float32"),
        (96, "float64"),
        (128, "float32"),
        (128, "float64"),
        (224, "float32"),
        (224, "float64"),
    }
    assert all(request["contract"]["interaction"] == "M1" for request in requests)
    assert all(request["contract"]["output_components"] == 1 for request in requests)


@pytest.mark.parametrize("backend,target", (("cuda", "sm_120"), ("hip", "gfx942")))
@pytest.mark.parametrize("schedule", ("table", "chunk32", "tiled128x4-v1"))
def test_foundation_device_m1_identity_matches_codegen_metadata(
    backend, target, schedule
):
    module = _manifest_module()
    from symmetrix.mace_jit_operator_codegen import (
        JIT_GENERATION_VERSION,
        jit_m1_device_metadata,
    )

    requests = module.foundation_m1_artifact_inputs(
        module.load_foundation_operator_manifest(),
        generation_revision=JIT_GENERATION_VERSION,
        backend=backend,
        target=target,
        precisions=("float32", "float64"),
        schedule=schedule,
    )

    for request in requests:
        metadata = jit_m1_device_metadata(
            request["contract"],
            precision=request["identity"]["precision"],
            target=target,
            schedule=schedule,
        )
        assert request["identity"] == metadata["generated_identity"]


def test_foundation_host_m1_preserves_chunk32_revision_two_and_rejects_others():
    module = _manifest_module()
    manifest = module.load_foundation_operator_manifest()
    requests = module.foundation_m1_artifact_inputs(
        manifest,
        generation_revision=19,
        backend="host",
        target="x86-64-v3",
        precisions=("float64",),
    )

    assert {request["identity"]["schedule_revision"] for request in requests} == {2}
    for schedule in ("table", "tiled128x4-v1"):
        with pytest.raises(ValueError, match="host M1 artifacts support only"):
            module.foundation_m1_artifact_inputs(
                manifest,
                generation_revision=19,
                backend="host",
                target="x86-64-v3",
                precisions=("float64",),
                schedule=schedule,
            )
    with pytest.raises(ValueError, match="requires schedule revision 2, got 3"):
        module.foundation_m1_artifact_inputs(
            manifest,
            generation_revision=19,
            backend="host",
            target="x86-64-v3",
            precisions=("float64",),
            schedule_revision=3,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value.update(
                schema="symmetrix.foundation-operator-manifest/2"
            ),
            "schema",
        ),
        (
            lambda value: value["aliases"][0].update(target="unverified-model"),
            "unknown model",
        ),
        (
            lambda value: value["bundles"][0]["operators"].pop("M0"),
            "M0",
        ),
        (
            lambda value: value["models"][0].update(bundle="unverified-group"),
            "unknown bundle",
        ),
        (
            lambda value: value["r0_artifact_inputs"][0].pop("channels"),
            "channels",
        ),
    ),
)
def test_foundation_manifest_rejects_unknown_or_incomplete_groups(mutation, message):
    module = _manifest_module()
    manifest = json.loads(module.FOUNDATION_OPERATOR_MANIFEST_PATH.read_text())
    mutation(manifest)

    with pytest.raises(ValueError, match=message):
        module.normalize_foundation_operator_manifest(manifest)


@pytest.mark.parametrize(
    "catalog",
    ("r0_artifact_inputs", "bundles", "models", "aliases"),
)
@pytest.mark.parametrize("mutation", ("empty", "missing", "extra"))
def test_foundation_manifest_requires_exact_verified_catalog_ids(catalog, mutation):
    module = _manifest_module()
    manifest = json.loads(module.FOUNDATION_OPERATOR_MANIFEST_PATH.read_text())
    if mutation == "empty":
        manifest[catalog] = []
    elif mutation == "missing":
        manifest[catalog].pop()
    else:
        extra = deepcopy(manifest[catalog][0])
        extra["id"] = "unverified-entry"
        manifest[catalog].append(extra)

    with pytest.raises(ValueError, match="verified .* IDs"):
        module.normalize_foundation_operator_manifest(manifest)


@pytest.mark.parametrize("bundle_id", ("standard-l0", "standard-l1", "standard-l2"))
@pytest.mark.parametrize("field", ("term_count", "structure_fingerprint"))
def test_foundation_manifest_validates_every_m0_topology_from_canonical_evidence(
    bundle_id, field
):
    module = _manifest_module()
    manifest = json.loads(module.FOUNDATION_OPERATOR_MANIFEST_PATH.read_text())
    bundle = next(value for value in manifest["bundles"] if value["id"] == bundle_id)
    descriptor = bundle["operators"]["M0"]
    if field == "term_count":
        descriptor[field] += 1
    else:
        descriptor[field] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="canonical M0 topology"):
        module.normalize_foundation_operator_manifest(manifest)


@pytest.mark.parametrize("location", ("manifest", "R1", "M0"))
def test_foundation_manifest_rejects_boolean_schema_versions(location):
    module = _manifest_module()
    manifest = json.loads(module.FOUNDATION_OPERATOR_MANIFEST_PATH.read_text())
    if location == "manifest":
        manifest["version"] = True
    else:
        manifest["bundles"][0]["operators"][location]["version"] = True

    with pytest.raises(ValueError, match="version must be an integer"):
        module.normalize_foundation_operator_manifest(manifest)


def test_test_registry_is_a_validated_view_of_production_data():
    module = _manifest_module()
    from symmetrix.mace_execution_contract import normalize_jit_r1_contract

    production = module.load_foundation_operator_manifest()
    registry = json.loads((CONTRACTS / "foundation_model_registry.json").read_text())

    assert registry["schema"] == "symmetrix.foundation-model-registry"
    assert registry["version"] == 2
    assert registry["aliases"] == production["aliases"]
    production_models = {model["id"]: model for model in production["models"]}
    production_bundles = {bundle["id"]: bundle for bundle in production["bundles"]}
    assert set(production_models) == {model["id"] for model in registry["models"]}
    for model in registry["models"]:
        expected = production_models[model["id"]]
        assert {
            key: model[key] for key in ("id", "provenance", "head", "bundle")
        } == expected
        contract = normalize_jit_r1_contract(
            json.loads((CONTRACTS / model["r1_contract"]).read_text())
        )
        production_r1 = production_bundles[model["bundle"]]["operators"]["R1"]
        assert module.r1_operator_descriptor(contract) == {
            key: value for key, value in production_r1.items() if key != "contract"
        }
        assert contract == production_r1["contract"]


def test_foundation_manifest_normalization_is_deterministic():
    module = _manifest_module()
    source = json.loads(module.FOUNDATION_OPERATOR_MANIFEST_PATH.read_text())
    reordered = deepcopy(source)
    for field in ("aliases", "bundles", "models", "r0_artifact_inputs"):
        reordered[field].reverse()

    assert module.normalize_foundation_operator_manifest(
        reordered
    ) == module.normalize_foundation_operator_manifest(source)


def test_foundation_manifest_path_is_standard_library_only_and_runtime_free():
    source_path = PACKAGE / "foundation_operator_manifest.py"
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
        "copy",
        "generated_operator_core",
        "hashlib",
        "json",
        "mace_execution_contract",
        "pathlib",
        "re",
        "typing",
    }

    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    script = """
import importlib
from pathlib import Path
from types import ModuleType
import sys

package = ModuleType("_foundation_manifest_source")
package.__path__ = [sys.argv[1]]
package.__package__ = package.__name__
sys.modules[package.__name__] = package
module = importlib.import_module(package.__name__ + ".foundation_operator_manifest")
manifest = module.load_foundation_operator_manifest()
module.foundation_r0_artifact_inputs(
    manifest,
    generation_revision=14,
    compatibility_version=2,
    backend="host",
    target="x86-64-v3",
    precisions=("float32", "float64"),
    schedule="host-owner",
    schedule_revision=4,
)
forbidden = {
    name for name in sys.modules
    if name == "numpy" or name.startswith("numpy.")
    or name.startswith("symmetrix.backend_loader")
    or name.startswith("symmetrix.jit")
    or name.startswith("symmetrix._native")
}
if forbidden:
    raise SystemExit("unexpected imports: " + ", ".join(sorted(forbidden)))
if "symmetrix" in sys.modules:
    raise SystemExit("manifest executed symmetrix.__init__")
"""
    subprocess.run(
        (sys.executable, "-I", "-c", script, str(PACKAGE)),
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
