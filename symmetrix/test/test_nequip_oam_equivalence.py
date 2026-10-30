import importlib.util
import json
from pathlib import Path
import sys

import pytest


BENCHMARK_PATH = Path(__file__).parents[2] / "benchmarks" / "nequip_oam_equivalence.py"
CONTRACT_PATH = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"
XL_STRUCTURE_PATH = (
    Path(__file__).parent / "data" / "nequip" / "oam_xl_qualification_structure.json"
)


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "nequip_oam_equivalence", BENCHMARK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_structure(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "atomic_numbers": [1, 1],
                "positions": [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
                "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                "pbc": [False, False, False],
            }
        ),
        encoding="utf-8",
    )


def test_oam_xl_qualification_structure_has_the_recorded_edge_count():
    benchmark = load_benchmark_module()
    structure = benchmark.load_structure(XL_STRUCTURE_PATH)

    assert len(structure.numbers) == 5
    assert benchmark.directed_edges(structure, 6.5) == 510


def test_dry_run_manifest_contains_contract_and_graph_metadata(tmp_path):
    benchmark = load_benchmark_module()
    structure = tmp_path / "structure.json"
    write_structure(structure)

    manifest = benchmark.build_manifest(
        model_json=CONTRACT_PATH,
        reference_model=None,
        structure=structure,
        precision="float32",
        backend="cpu",
        neighbor_skin=0.5,
        properties=("energy", "forces", "stress"),
        dry_run=True,
    )

    assert manifest["status"] == "dry-run"
    assert len(manifest["model_sha256"]) == 64
    assert manifest["contract_fingerprint"].startswith("sha256:")
    assert manifest["upstream_package_versions"]["python"]
    assert manifest["cutoff"] == pytest.approx(4.5)
    assert manifest["atom_count"] == 2
    assert manifest["directed_edges"] == 2
    assert manifest["tolerances"]["energy_abs_eV_per_atom"] == pytest.approx(1e-5)
    assert manifest["execution_diagnostics"]["backend"] == "cpu"


def test_manifest_validation_rejects_missing_required_metadata():
    benchmark = load_benchmark_module()
    with pytest.raises(ValueError, match="contract_fingerprint"):
        benchmark.validate_manifest({"model_sha256": "0" * 64})


def _valid_generated_qualification() -> dict:
    return {
        "plugin_abi": "symmetrix-nequip-oam-host-v1",
        "plugin_abi_version": 1,
        "artifact_id": "sha256:artifact",
        "codegen_generation": "sha256:generation",
        "compiler_identity": {"compiler": "c++"},
        "scalar_size": 4,
        "generic_dispatch_count": 0,
        "generic_region_count": 0,
    }


def test_manifest_validation_rejects_invalid_generated_metadata():
    benchmark = load_benchmark_module()
    manifest = {
        "model_sha256": "0" * 64,
        "contract_fingerprint": "sha256:contract",
        "upstream_package_versions": {},
        "cutoff": 4.5,
        "atom_count": 2,
        "directed_edges": 2,
        "tolerances": dict(benchmark.DEFAULT_TOLERANCES),
        "execution_diagnostics": {},
        "generated_qualification": _valid_generated_qualification(),
    }
    manifest["generated_qualification"]["plugin_abi_version"] = 0
    with pytest.raises(ValueError, match="plugin_abi_version"):
        benchmark.validate_manifest(manifest)


def test_cuda_manifest_metadata_uses_the_published_oam_identity(tmp_path):
    benchmark = load_benchmark_module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "compiler_command": ["NVRTC-13.0"],
                "key_inputs": {
                    "compiler": {"kind": "nvrtc"},
                    "jit_generation_version": 13,
                    "build": {
                        "request": {
                            "abi": {"tag": "oam-cuda-abi", "version": 1},
                            "oam": {
                                "artifact_id": "oam-cuda-r1-test",
                                "generation_fingerprint": "sha256:generation",
                                "scalar_size": 4,
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = benchmark._artifact_manifest_metadata(manifest)

    assert metadata["artifact_id"] == "oam-cuda-r1-test"
    assert metadata["plugin_abi"] == "oam-cuda-abi"
    assert metadata["compiler_identity"] == {"kind": "nvrtc"}
    assert metadata["jit_generation_version"] == 13


@pytest.mark.parametrize(
    "field",
    ["model_sha256", "contract_fingerprint", "cutoff", "atom_count", "directed_edges"],
)
def test_passed_qualification_manifest_requires_its_workload_identity(field):
    """A qualification claim must identify the workload it measured.

    The shape checks tolerate ``null`` on a skipped or dry-run record, where the
    contract or structure was unavailable.  A ``passed`` record is different: it
    reports measured parity errors, so a null identity field would leave the
    claim unreproducible.
    """

    benchmark = load_benchmark_module()
    manifest = {
        "status": "passed",
        "model_sha256": "0" * 64,
        "contract_fingerprint": "sha256:contract",
        "upstream_package_versions": {},
        "cutoff": 4.5,
        "atom_count": 2,
        "directed_edges": 2,
        "tolerances": dict(benchmark.DEFAULT_TOLERANCES),
        "execution_diagnostics": {},
        "generated_qualification": _valid_generated_qualification(),
    }
    benchmark.validate_manifest(manifest)

    manifest[field] = None
    with pytest.raises(ValueError, match=field):
        benchmark.validate_manifest(manifest)


def test_skipped_manifest_may_still_omit_its_workload_identity():
    """An unavailable contract/structure legitimately leaves those fields null."""

    benchmark = load_benchmark_module()
    benchmark.validate_manifest(
        {
            "status": "skipped",
            "model_sha256": None,
            "contract_fingerprint": None,
            "upstream_package_versions": {},
            "cutoff": None,
            "atom_count": None,
            "directed_edges": None,
            "tolerances": dict(benchmark.DEFAULT_TOLERANCES),
            "execution_diagnostics": {},
            "generated_qualification": _valid_generated_qualification(),
        }
    )


def test_missing_generated_counters_are_a_qualification_failure():
    benchmark = load_benchmark_module()
    manifest = {"status": "pending", "execution_diagnostics": {}}
    benchmark._mark_comparison_failure(
        manifest,
        RuntimeError(
            "generated OAM qualification did not report both generic tensor-product counters"
        ),
    )
    assert manifest["status"] == "failed"
    assert "generic tensor-product counters" in manifest["error"]


def test_generic_dispatch_failure_returns_nonzero(tmp_path, monkeypatch):
    benchmark = load_benchmark_module()
    output = tmp_path / "manifest.json"
    args = benchmark.make_parser().parse_args(
        ["--model-json", str(CONTRACT_PATH), "--output", str(output)]
    )
    manifest = {
        "status": "failed",
        "error": "generated OAM qualification observed generic tensor-product dispatch",
    }
    monkeypatch.setattr(benchmark, "build_manifest", lambda **_: manifest)
    # The CLI's exit policy must not turn a failed qualification into a
    # successful process exit.
    assert benchmark.main(args) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_profile_missing_runtime_writes_skipped_manifest(tmp_path, monkeypatch):
    profile_path = (
        Path(__file__).parents[2] / "benchmarks" / "nequip_oam_kokkos_profile.py"
    )
    spec = importlib.util.spec_from_file_location(
        "nequip_oam_kokkos_profile", profile_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = tmp_path / "profile.json"
    monkeypatch.setattr(
        module,
        "_configure_process",
        lambda _: (_ for _ in ()).throw(ImportError("ASE missing")),
    )
    args = module.make_parser().parse_args(
        ["--model-json", str(CONTRACT_PATH), "--output", str(output)]
    )
    assert module.main(args) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "skipped"
    assert record["skip_reason"]


def test_profile_generic_dispatch_failure_returns_nonzero(tmp_path):
    profile_path = (
        Path(__file__).parents[2] / "benchmarks" / "nequip_oam_kokkos_profile.py"
    )
    spec = importlib.util.spec_from_file_location(
        "nequip_oam_kokkos_profile_failed", profile_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = tmp_path / "profile-failed.json"
    module.profile = lambda *args, **kwargs: {
        "status": "failed",
        "error": "generic dispatch",
    }
    args = module.make_parser().parse_args(
        ["--model-json", str(CONTRACT_PATH), "--output", str(output)]
    )
    assert module.main(args) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_missing_reference_reports_actionable_skip(tmp_path, monkeypatch):
    benchmark = load_benchmark_module()
    structure = tmp_path / "structure.json"
    write_structure(structure)
    monkeypatch.delenv("SYMMETRIX_NEQUIP_OAM_S", raising=False)

    manifest = benchmark.build_manifest(
        model_json=CONTRACT_PATH,
        reference_model=None,
        structure=structure,
        precision="float32",
        backend="cpu",
        neighbor_skin=0.5,
        properties=("energy", "forces"),
        dry_run=False,
    )

    assert manifest["status"] == "skipped"
    assert "SYMMETRIX_NEQUIP_OAM_S" in manifest["skip_reason"]


def test_cli_dry_run_writes_manifest_without_optional_dependencies(tmp_path):
    benchmark = load_benchmark_module()
    structure = tmp_path / "structure.json"
    output = tmp_path / "manifest.json"
    write_structure(structure)

    args = benchmark.make_parser().parse_args(
        [
            "--model-json",
            str(CONTRACT_PATH),
            "--structure",
            str(structure),
            "--output",
            str(output),
            "--dry-run",
        ]
    )
    assert benchmark.main(args) == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["status"] == "dry-run"


@pytest.mark.parametrize("initialized_before", [False, True])
def test_cli_finalizes_only_the_kokkos_runtime_it_initialized(
    tmp_path, monkeypatch, initialized_before
):
    """In-process callers retain ownership of an existing Kokkos runtime."""

    benchmark = load_benchmark_module()
    output = tmp_path / "manifest.json"

    class Native:
        initialized = initialized_before
        finalize_calls = 0

        @classmethod
        def _kokkos_is_initialized(cls):
            return cls.initialized

        @classmethod
        def _finalize_kokkos(cls):
            cls.finalize_calls += 1
            cls.initialized = False

    class SymmetrixModule:
        @staticmethod
        def load_backend():
            return Native

    def build_manifest(**_):
        Native.initialized = True
        return {"status": "dry-run"}

    monkeypatch.setitem(sys.modules, "symmetrix", SymmetrixModule)
    monkeypatch.setattr(benchmark, "build_manifest", build_manifest)
    args = benchmark.make_parser().parse_args(
        ["--model-json", str(CONTRACT_PATH), "--output", str(output), "--dry-run"]
    )

    assert benchmark.main(args) == 0
    assert Native.finalize_calls == (0 if initialized_before else 1)
    assert Native.initialized is initialized_before


def test_default_properties_exclude_stress():
    benchmark = load_benchmark_module()
    args = benchmark.make_parser().parse_args(
        ["--model-json", str(CONTRACT_PATH), "--output", "manifest.json"]
    )
    assert args.properties == "energy,forces"


def test_dependency_free_fallback_rejects_malformed_contract(tmp_path, monkeypatch):
    benchmark = load_benchmark_module()
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps({"model_type": "NequIP", "r_cut": 4.5}), encoding="utf-8"
    )
    monkeypatch.setitem(__import__("sys").modules, "symmetrix", None)

    manifest = benchmark.build_manifest(
        model_json=malformed,
        reference_model=None,
        structure=None,
        precision="float32",
        backend="cpu",
        neighbor_skin=0.5,
        dry_run=True,
    )
    assert manifest["status"] == "failed"
    assert "model contract validation failed" in manifest["skip_reason"]


class _StressUnsupportedEvaluator:
    """An evaluator whose capability query reports no stress reduction."""

    supports_stress = False


class _StressUnsupportedCalculator:
    """A candidate calculator whose evaluator serves no stress reduction."""

    def __init__(self, *args, **kwargs):
        self.evaluator = _StressUnsupportedEvaluator()


def test_unsupported_native_stress_is_skipped_by_runtime_capability(
    tmp_path, monkeypatch
):
    import sys
    import types

    # Only the optional upstream constructor is substituted, plus a candidate
    # evaluator that reports no stress support: the capability decision belongs
    # to the runtime gate, which reads the selected evaluator's own flag.  The
    # shipped OAM host evaluator serves stress (its compiler-free host
    # reduction is qualified), so an evaluator that cannot is the only way to
    # reach the gate at all.
    pytest.importorskip("ase")
    pytest.importorskip("symmetrix")
    benchmark = load_benchmark_module()
    reference_module = types.ModuleType("nequip.integrations.ase")
    reference_module.NequIPCalculator = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "nequip.integrations.ase", reference_module)
    monkeypatch.setattr("symmetrix.NequIPCalculator", _StressUnsupportedCalculator)
    reference = tmp_path / "reference.nequip"
    reference.write_text("unused upstream model")
    structure = tmp_path / "structure.json"
    write_structure(structure)
    manifest = benchmark.build_manifest(
        model_json=CONTRACT_PATH,
        reference_model=reference,
        structure=structure,
        precision="float64",
        backend="cpu",
        neighbor_skin=0.5,
        properties=("energy", "forces", "stress"),
        dry_run=False,
    )
    assert manifest["status"] == "skipped"
    assert manifest["property_status"]["stress"]["status"] == "skipped"
    assert "stress" in manifest["skip_reason"]


def test_configured_missing_oam_reference_is_explicitly_skipped(tmp_path, monkeypatch):
    benchmark = load_benchmark_module()
    missing = tmp_path / "missing-oam-s.nequip"
    monkeypatch.setenv("SYMMETRIX_NEQUIP_OAM_S", str(missing))
    manifest = benchmark.build_manifest(
        model_json=CONTRACT_PATH,
        reference_model=None,
        structure=None,
        precision="float32",
        backend="cpu",
        neighbor_skin=0.5,
        dry_run=True,
    )
    assert manifest["oam_references"]["S"]["status"] == "skipped"
    assert "does not exist" in manifest["oam_references"]["S"]["reason"]
    assert set(manifest["oam_references"]) == {"S", "M", "L", "XL"}


def test_backend_mismatch_is_skipped(tmp_path, monkeypatch):
    benchmark = load_benchmark_module()
    reference = tmp_path / "reference.nequip"
    reference.write_text("placeholder", encoding="utf-8")
    structure = tmp_path / "structure.json"
    write_structure(structure)
    monkeypatch.setattr(
        benchmark,
        "_run_comparison",
        lambda manifest, *args: manifest.update(
            status="skipped",
            skip_reason="selected evaluator backend mismatch: requested cuda, selected cpu",
        ),
    )
    manifest = benchmark.build_manifest(
        model_json=CONTRACT_PATH,
        reference_model=reference,
        structure=structure,
        precision="float32",
        backend="cuda",
        neighbor_skin=0.5,
        properties=("energy", "forces"),
        dry_run=False,
    )
    assert manifest["status"] == "skipped"
    assert "backend mismatch" in manifest["skip_reason"]


def test_evaluation_error_is_failed_not_skipped(tmp_path):
    benchmark = load_benchmark_module()
    manifest = {
        "status": "pending",
        "execution_diagnostics": {},
    }
    benchmark._mark_comparison_failure(
        manifest, RuntimeError("model evaluation failed")
    )
    assert manifest["status"] == "failed"
    assert "model evaluation failed" in manifest["error"]


def test_extended_xyz_preserves_periodic_metadata(tmp_path):
    import numpy as np

    benchmark = load_benchmark_module()
    path = tmp_path / "periodic.extxyz"
    path.write_text(
        '1\nLattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3 pbc="T T T"\nH 0 0 0\n',
        encoding="utf-8",
    )
    structure = benchmark.load_structure(path)
    np.testing.assert_allclose(
        np.asarray(structure.cell, dtype=float), np.diag([2.0, 2.0, 2.0])
    )
    assert structure.pbc == (True, True, True)
    assert structure.numbers == (1,)
    assert np.asarray(structure.positions, dtype=float).shape == (1, 3)
