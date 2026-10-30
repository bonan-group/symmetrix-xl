import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "benchmarks" / "validate_mace_les_optimization.py"


@pytest.fixture(scope="module")
def validator():
    spec = importlib.util.spec_from_file_location("les_optimization_validator", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ms, *, revision, backend="hip", model_type="MACELES"):
    digest = ("1" if revision == "base" else "2") * 64
    maceles = model_type == "MACELES"
    return {
        "schema": "symmetrix.mace-les-timing/4",
        "samples_ns": [
            int(ms * 1e6 * (0.99 + (index % 3) * 0.01)) for index in range(30)
        ],
        "warmups": 20,
        "num_atoms": 108,
        "precision": "float32",
        "backend": backend,
        "pbc": "periodic",
        "streamed_edges": "direct",
        "execution_profile": "speed",
        "replication": [1, 1, 1],
        "cell_scale": 1.0,
        "model_type": model_type,
        "model": f"/qualified/{'maceles' if maceles else 'mace'}.json",
        "model_sha256": ("a" if maceles else "e") * 64,
        "mace_les_parameters_sha256": "c" * 64 if maceles else None,
        "mace_les_evaluator_fingerprint": ("d" if maceles else "f") * 64,
        "mace_architecture": {
            "mace_num_interactions": 2,
            "mace_num_channels": 128,
            "mace_contract_fingerprints": {
                "R0": "sha256:" + "3" * 64,
                "M0": "sha256:" + "4" * 64,
                "R1": "sha256:" + "5" * 64,
            },
            "atomic_numbers": [1, 8],
            "cutoff": 5.0,
        },
        "structure_sha256": "b" * 64,
        "cutoff": 5.0,
        "skin": 0.0,
        "directed_edges": 4000,
        "fallback_evaluation_count": 0,
        "host_graph_constructions_during_timing": 0,
        "prepared_graph_count": 1,
        "native_graph_generation_before_warmup": 0,
        "native_graph_generation_before_timing": 1,
        "native_graph_generation_after_timing": 1,
        "native_graph_rebuilds_during_warmup": 1,
        "native_graph_rebuilds_during_timing": 0,
        "native_extension_sha256": digest,
        "native_build": {
            "source_commit": revision,
            "source_dirty": False,
            "source_content_sha256": digest,
            "native_source_content_sha256": digest,
        },
        "source_provenance": {
            "source_commit": revision,
            "source_dirty": False,
            "source_content_sha256": digest,
            "native_source_content_sha256": digest,
        },
        "interpreter_path": "/qualified/venv/bin/python",
        "python_version": "3.12.7",
        "symmetrix_package_path": "/qualified/site-packages/symmetrix/__init__.py",
        "symmetrix_version": "1.2.3",
        "device_synchronization_bracketed_samples": backend in {"hip", "cuda"},
        "synchronization": {
            "backend": backend if backend in {"hip", "cuda"} else "none",
            "runtime_library": (
                f"lib{backend}.so" if backend in {"hip", "cuda"} else None
            ),
        },
        "jit_status": "cached",
        "jit_artifact_path": "/qualified/cache/operator.so",
        "device_provenance": {
            "architecture": "gfx1151" if backend == "hip" else "sm120",
            "device_name": "test",
        },
    }


def _profile(*, backend, speedup=2.0):
    def summary(candidate):
        categories = [
            {
                "category": category,
                "duration_ns_per_evaluation": 300.0 / (speedup if candidate else 1),
            }
            for category in ("structure_factors", "atom_outputs", "strain")
        ]
        result = {
            "schema": (
                "symmetrix.rocprof-kernel-summary"
                if backend == "hip"
                else "symmetrix.nsys-les-kernel-summary/2"
            ),
            "selected_evaluation_count": 10,
            "les_stage_categories": categories,
        }
        if backend == "hip":
            result["version"] = 2
        return result

    return {"baseline": summary(False), "candidate": summary(True)}


def _document(candidate_ms=2.9, *, backend="hip", standard=False):
    blocks = []
    for index in range(5):
        records = {
            "baseline": _record(3.0, revision="base", backend=backend),
            "candidate": _record(candidate_ms, revision="candidate", backend=backend),
        }
        if standard:
            records["standard_mace"] = _record(
                2.0, revision="candidate", backend=backend, model_type="MACE"
            )
        blocks.append({"block_id": f"block-{index}", "records": records})
    return {
        "schema": "symmetrix.mace-les-optimization/2",
        "bootstrap_seed": 17,
        "bootstrap_draws": 2000,
        "blocks": blocks,
        "profiles": _profile(backend=backend),
    }


def test_validates_paired_complete_maceles_timings(validator):
    summary = validator.validate_document(_document())
    assert summary["block_count"] == 5
    assert summary["median_speedup"] == pytest.approx(3.0 / 2.9)
    assert summary["median_candidate_to_baseline_ratio"] == pytest.approx(2.9 / 3.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "MACE"),
        ("mace_les_parameters_sha256", None),
        ("mace_les_evaluator_fingerprint", "bad"),
    ],
)
def test_rejects_incomplete_maceles_identity(validator, field, value):
    document = _document()
    document["blocks"][0]["records"]["candidate"][field] = value
    with pytest.raises(validator.QualificationError, match=field):
        validator.validate_document(document)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "empty",
        "wrong_object_type",
        "wrong_interactions_type",
        "wrong_channels_type",
        "wrong_fingerprints_type",
        "incomplete_fingerprints",
        "invalid_fingerprint",
        "wrong_atomic_numbers_type",
        "wrong_atomic_number_type",
        "wrong_cutoff_type",
    ],
)
def test_rejects_incomplete_or_malformed_mace_architecture(validator, mutation):
    document = _document()
    for block in document["blocks"]:
        for candidate in block["records"].values():
            architecture = candidate["mace_architecture"]
            if mutation == "missing":
                candidate.pop("mace_architecture")
            elif mutation == "empty":
                candidate["mace_architecture"] = {}
            elif mutation == "wrong_object_type":
                candidate["mace_architecture"] = []
            elif mutation == "wrong_interactions_type":
                architecture["mace_num_interactions"] = "2"
            elif mutation == "wrong_channels_type":
                architecture["mace_num_channels"] = 128.0
            elif mutation == "wrong_fingerprints_type":
                architecture["mace_contract_fingerprints"] = []
            elif mutation == "incomplete_fingerprints":
                architecture["mace_contract_fingerprints"].pop("R1")
            elif mutation == "invalid_fingerprint":
                architecture["mace_contract_fingerprints"]["M0"] = "sha256:bad"
            elif mutation == "wrong_atomic_numbers_type":
                architecture["atomic_numbers"] = "1,8"
            elif mutation == "wrong_atomic_number_type":
                architecture["atomic_numbers"] = [1, True]
            else:
                architecture["cutoff"] = "5.0"

    with pytest.raises(validator.QualificationError, match="mace_architecture"):
        validator.validate_document(document)


@pytest.mark.parametrize("malformed_block", [None, [], "block", 1])
def test_rejects_non_object_first_block_with_qualification_error(
    validator, malformed_block
):
    document = _document()
    document["blocks"][0] = malformed_block

    with pytest.raises(validator.QualificationError, match=r"blocks\[0\]"):
        validator.validate_document(document)


def test_rejects_obsolete_enabled_disabled_roles(validator):
    document = _document()
    records = document["blocks"][0]["records"]
    records["baseline_enabled"] = records.pop("baseline")
    records["candidate_enabled"] = records.pop("candidate")
    with pytest.raises(validator.QualificationError, match="timing roles"):
        validator.validate_document(document)


@pytest.mark.parametrize("mutation", ["fallback", "graph", "workload", "samples"])
def test_rejects_invalid_complete_model_evidence(validator, mutation):
    document = _document()
    candidate = document["blocks"][0]["records"]["candidate"]
    if mutation == "fallback":
        candidate["fallback_evaluation_count"] = 1
    elif mutation == "graph":
        candidate["host_graph_constructions_during_timing"] = 1
    elif mutation == "workload":
        candidate["directed_edges"] += 2
    else:
        candidate["samples_ns"] = [1_000_000] * 3
    with pytest.raises(validator.QualificationError):
        validator.validate_document(document)


def test_standard_mace_cost_uses_separate_architecture_matched_model(validator):
    summary = validator.validate_document(_document(standard=True))
    assert summary["median_standard_mace_ms"] == pytest.approx(2.0)
    assert summary["median_maceles_over_standard_mace"] == pytest.approx(2.9 / 2.0)

    same_model = _document(standard=True)
    block = same_model["blocks"][0]["records"]
    block["standard_mace"]["model_sha256"] = block["candidate"]["model_sha256"]
    with pytest.raises(validator.QualificationError, match="separate"):
        validator.validate_document(same_model)

    wrong_architecture = _document(standard=True)
    wrong_architecture["blocks"][0]["records"]["standard_mace"]["mace_architecture"][
        "mace_num_channels"
    ] = 64
    with pytest.raises(validator.QualificationError, match="architecture"):
        validator.validate_document(wrong_architecture)


def test_rejects_same_baseline_and_candidate_binary_identity(validator):
    document = _document()
    for block in document["blocks"]:
        baseline = block["records"]["baseline"]
        candidate = block["records"]["candidate"]
        candidate["native_extension_sha256"] = baseline["native_extension_sha256"]
        for owner in ("native_build", "source_provenance"):
            candidate[owner] = dict(baseline[owner])
    with pytest.raises(validator.QualificationError, match="distinct"):
        validator.validate_document(document)


def test_gpu_requires_les_stage_profiles(validator):
    document = _document()
    del document["profiles"]
    with pytest.raises(validator.QualificationError, match="profiles"):
        validator.validate_document(document)

    wrong = _document()
    wrong["profiles"]["candidate"]["les_stage_categories"][0]["category"] = "wrong"
    with pytest.raises(validator.QualificationError, match="structure_factors"):
        validator.validate_document(wrong)
