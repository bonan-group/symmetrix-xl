import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "foundation_model_support_matrix.py"
MANIFEST = ROOT / "benchmarks" / "foundation_model_support_manifest.json"
MODELS_DOCUMENT = ROOT / "docs" / "user" / "models.md"
PYTHON_API_DOCUMENT = ROOT / "docs" / "reference" / "python_api.md"
EXECUTION_DOCUMENT = ROOT / "docs" / "user" / "execution.md"


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location("foundation_support_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nequip_support_manifest_has_versioned_contract_and_model_identifiers(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    support = manifest["nequip_support"]

    assert manifest["version"] == matrix.MANIFEST_VERSION == 5
    # The gated models are OAM contracts at format version 2.
    assert support["format_version"] == matrix.NEQUIP_FORMAT_VERSION == 2
    assert support["model_type"] == "NequIP"
    assert set(support["models"]) == {
        "nequip-oam-s",
        "nequip-oam-m",
        "nequip-oam-l",
        "nequip-oam-xl",
    }
    assert support["runtime_input"] == "exported_json"
    assert not any(cell["state"] == "qualified" for cell in support["release_matrix"])
    assert (
        "live upstream-reference float64 parity"
        in support["models"]["nequip-oam-xl"]["open_gates"]
    )
    assert {model["release_matrix_size"] for model in support["models"].values()} == {
        "S",
        "M",
        "L",
        "XL",
    }


def test_host_fp64_artifact_availability_does_not_claim_runtime_qualification(
    matrix,
):
    manifest = matrix.load_manifest(MANIFEST)
    rows = {
        (cell["size"], cell["precision"]): cell
        for cell in manifest["nequip_support"]["release_matrix"]
        if cell["backend"] == "host"
    }
    for size in ("S", "L", "XL"):
        row = rows[(size, "float64")]
        assert row["artifact"]["availability"] == "available"
        assert row["runtime_qualification"] == "available_not_runtime_qualified"


def test_host_artifact_kind_names_the_current_oam_abi(matrix):
    from symmetrix.oam_receiver_codegen import ABI_VERSION

    manifest = matrix.load_manifest(MANIFEST)
    kinds = {
        cell["artifact"]["kind"]
        for cell in manifest["nequip_support"]["release_matrix"]
        if cell["backend"] == "host"
    }

    assert kinds == {f"generated_oam_host_abi_v{ABI_VERSION}"}


def test_manifest_rejects_a_stale_host_artifact_abi_kind(matrix, tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    host = next(
        cell
        for cell in manifest["nequip_support"]["release_matrix"]
        if cell["backend"] == "host"
    )
    host["artifact"]["kind"] = "generated_oam_host_abi_v3"
    invalid = tmp_path / "stale-host-artifact-abi.json"
    invalid.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact kind"):
        matrix.load_manifest(invalid)


def test_user_model_document_does_not_label_the_retained_oam_boundary_experimental():
    from symmetrix.oam_receiver_codegen import ABI_VERSION

    document = MODELS_DOCUMENT.read_text()

    assert "## NequIP and OAM (experimental)" not in document
    assert "## NequIP and OAM" in document
    assert f"receiver-owned ABI v{ABI_VERSION}" in document
    assert "All eight host artifacts are shipped" not in document
    assert "Official CPU and CUDA wheels include generated" not in document
    assert "wheel targets remain available" not in document
    assert "Fresh correctness, performance, capacity" in document
    assert "repaired-wheel gates are required before a release matrix" in document


def test_user_execution_document_defers_oam_wheel_inclusion_claims():
    document = EXECUTION_DOCUMENT.read_text()

    assert "packages include the six verified MACE R0 artifacts" in document
    assert "and eight OAM artifacts" not in document
    assert "OAM wheel inclusion is pending post-merge repaired-wheel" in document


def test_python_api_documents_the_pending_oam_release_boundary():
    from symmetrix.oam_receiver_codegen import ABI_VERSION

    document = PYTHON_API_DOCUMENT.read_text()

    assert "NequIPCalculator (experimental OAM v2)" not in document
    assert "experimental stress" not in document
    assert f"host ABI-v{ABI_VERSION}" in document
    assert "CUDA\nABI-v3" in document
    assert "Fresh runtime and repaired-wheel qualification\nis pending" in document


@pytest.mark.parametrize("stress_status", ("qualified", None))
def test_stress_release_gate_rejects_invalid_status(matrix, tmp_path, stress_status):
    manifest = json.loads(MANIFEST.read_text())
    manifest["nequip_support"]["release_matrix"][0]["properties"]["stress"] = (
        stress_status
    )
    invalid = tmp_path / "invalid-stress-gate.json"
    invalid.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        matrix.load_manifest(invalid)


def test_nequip_support_prerequisites_name_export_runtime_and_release_gates(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    prerequisites = manifest["nequip_support"]["prerequisites"]

    assert {
        "export",
        "runtime",
        "parity",
        "capacity",
    } <= set(prerequisites)
    assert "nequip" in prerequisites["export"]
    assert "e3nn" in prerequisites["export"]
    assert "symmetrix_nequip_format_version=2" in prerequisites["runtime"]
    assert "symmetrix_nequip_format_version=1" not in prerequisites["runtime"]
    assert "ASE + upstream NequIP reference" in prerequisites["parity"]
    assert "fresh-process backend execution record" in prerequisites["capacity"]


def test_nequip_support_validation_rejects_missing_prerequisites(matrix, tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    del manifest["nequip_support"]["prerequisites"]["runtime"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="runtime prerequisites"):
        matrix.load_manifest(invalid)


@pytest.mark.parametrize("historical_version", (1, 2, 3))
def test_release_harness_rejects_historical_manifest_versions(
    matrix, tmp_path, historical_version
):
    current = json.loads(MANIFEST.read_text())
    if historical_version == 1:
        current.pop("nequip_support")
    historical = tmp_path / f"historical-v{historical_version}.json"
    historical.write_text(json.dumps({**current, "version": historical_version}))
    with pytest.raises(
        ValueError, match="unsupported Foundation support manifest version"
    ):
        matrix.load_manifest(historical)


def test_release_harness_rejects_future_or_version_tampered_manifest(matrix, tmp_path):
    current = json.loads(MANIFEST.read_text())

    unsupported = tmp_path / "unsupported.json"
    unsupported.write_text(
        json.dumps({**current, "version": matrix.MANIFEST_VERSION + 1})
    )
    with pytest.raises(
        ValueError, match="unsupported Foundation support manifest version"
    ):
        matrix.load_manifest(unsupported)
