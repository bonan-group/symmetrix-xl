import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "benchmarks" / "validate_mace_les_correctness.py"
TIMING_PATH = ROOT / "benchmarks" / "mace_les_timing.py"


@pytest.fixture(scope="module")
def validator():
    spec = importlib.util.spec_from_file_location("les_correctness_validator", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_timing_driver():
    spec = importlib.util.spec_from_file_location(
        "mace_les_timing_contract", TIMING_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_timing_cli_identifies_the_complete_model_file():
    parser = _load_timing_driver()._parser()
    arguments = parser.parse_args(["model.json"])

    assert arguments.model == Path("model.json")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(kind, checkpoint, converted, structure, delta=0.0):
    def identity(path):
        return {
            "path": str(path),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    return {
        "schema": "symmetrix.mace-les-correctness/2",
        "model_type": "MACELES",
        "mace_les_parameters_sha256": "c" * 64,
        "mace_les_evaluator_fingerprint": "d" * 64,
        "implementation": kind,
        "precision": "float64",
        "backend": "cpu",
        "mode": "reference" if kind == "symmetrix" else "upstream",
        "checkpoint": identity(checkpoint),
        "converted_model": identity(converted),
        "structure": {
            **identity(structure),
            "frame": 0,
            "atoms": 2,
        },
        "runtime_provenance": {
            "interpreter_path": sys.executable,
            "python_version": "3.12.7",
            "package_path": pytest.__file__,
            "package_version": "1.2.3",
        },
        "energy": 3.0 + delta,
        "forces": [[1.0 + delta, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "latent_charges": [0.2 + delta, -0.2],
        "stress": [0.1 + delta, 0.2, 0.3, 0.0, 0.0, 0.0],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "MACE"),
        ("mace_les_parameters_sha256", None),
        ("mace_les_evaluator_fingerprint", "not-a-digest"),
    ],
)
def test_validator_requires_complete_maceles_identity(
    validator, tmp_path, field, value
):
    checkpoint = tmp_path / "model.pt"
    converted = tmp_path / "model.json"
    structure = tmp_path / "fixture.xyz"
    checkpoint.write_bytes(b"checkpoint")
    converted.write_text("{}")
    structure.write_text("fixture")
    reference = _result("upstream", checkpoint, converted, structure)
    candidate = _result("symmetrix", checkpoint, converted, structure)
    candidate[field] = value

    with pytest.raises(validator.CorrectnessError, match=field):
        validator.validate_documents(reference, candidate)


def test_validator_recomputes_deviations_and_verifies_hashes(validator, tmp_path):
    checkpoint = tmp_path / "model.pt"
    converted = tmp_path / "model.json"
    structure = tmp_path / "fixture.xyz"
    checkpoint.write_bytes(b"checkpoint")
    converted.write_text("{}")
    structure.write_text("fixture")
    summary = validator.validate_documents(
        _result("upstream", checkpoint, converted, structure),
        _result("symmetrix", checkpoint, converted, structure, delta=1e-8),
    )
    assert summary["passed"] is True
    assert summary["deviations"]["force_max_abs_eV_per_A"] == pytest.approx(1e-8)


def test_validator_rejects_threshold_and_provenance_mismatch(validator, tmp_path):
    checkpoint = tmp_path / "model.pt"
    converted = tmp_path / "model.json"
    structure = tmp_path / "fixture.xyz"
    checkpoint.write_bytes(b"checkpoint")
    converted.write_text("{}")
    structure.write_text("fixture")
    with pytest.raises(validator.CorrectnessError, match="tolerance"):
        validator.validate_documents(
            _result("upstream", checkpoint, converted, structure),
            _result("symmetrix", checkpoint, converted, structure, delta=1e-2),
        )
    bad = _result("symmetrix", checkpoint, converted, structure)
    bad["structure"]["sha256"] = "d" * 64
    with pytest.raises(validator.CorrectnessError, match="structure"):
        validator.validate_documents(
            _result("upstream", checkpoint, converted, structure), bad
        )
    bad_size = _result("symmetrix", checkpoint, converted, structure)
    bad_size["converted_model"]["size_bytes"] += 1
    with pytest.raises(validator.CorrectnessError, match="size"):
        validator.validate_documents(
            _result("upstream", checkpoint, converted, structure), bad_size
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("energy", 3.01),
        ("forces", [[1.001, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ("latent_charges", [0.201, -0.2]),
        ("stress", [0.101, 0.2, 0.3, 0.0, 0.0, 0.0]),
    ],
)
def test_validator_enforces_each_property_threshold(validator, tmp_path, field, value):
    checkpoint = tmp_path / "model.pt"
    converted = tmp_path / "model.json"
    structure = tmp_path / "fixture.xyz"
    checkpoint.write_bytes(b"checkpoint")
    converted.write_text("{}")
    structure.write_text("fixture")
    reference = _result("upstream", checkpoint, converted, structure)
    candidate = _result("symmetrix", checkpoint, converted, structure)
    candidate[field] = value
    with pytest.raises(validator.CorrectnessError, match="tolerance"):
        validator.validate_documents(reference, candidate)


@pytest.mark.parametrize(
    "mutation", ["implementation", "mode", "backend", "stress_shape"]
)
def test_validator_rejects_mislabeled_pair_and_shape(validator, tmp_path, mutation):
    checkpoint = tmp_path / "model.pt"
    converted = tmp_path / "model.json"
    structure = tmp_path / "fixture.xyz"
    checkpoint.write_bytes(b"checkpoint")
    converted.write_text("{}")
    structure.write_text("fixture")
    reference = _result("symmetrix", checkpoint, converted, structure)
    reference["mode"] = "reference"
    reference["backend"] = "hip"
    candidate = _result("symmetrix", checkpoint, converted, structure)
    candidate["mode"] = "direct"
    candidate["backend"] = "hip"
    if mutation == "implementation":
        candidate["implementation"] = "upstream"
    elif mutation == "mode":
        candidate["mode"] = "reference"
    elif mutation == "backend":
        candidate["backend"] = "cuda"
    else:
        candidate["stress"] = [[0.0] * 3 for _ in range(3)]
    with pytest.raises(validator.CorrectnessError):
        validator.validate_documents(reference, candidate)
