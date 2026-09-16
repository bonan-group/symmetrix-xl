import argparse
import importlib.util
import json
from pathlib import Path

import pytest

BENCHMARK_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "mace_omat0_equivalence_benchmark.py"
)


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "mace_omat0_benchmark", BENCHMARK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_record(path, backend, device):
    identity = {"implementation": backend}
    if backend == "symmetrix":
        identity = {
            "implementation": "symmetrix-factorized",
            "streamed_edges": "factorized",
            "jit_status": "cached",
        }
    path.write_text(
        json.dumps(
            {
                "backend": backend,
                "device": device,
                "case": {"atoms": 1},
                "md": {"steps": 20},
                "identity": identity,
                "outputs": {
                    "trajectory": [
                        {
                            "potential_energy_eV": 0.0,
                            "forces_eV_per_A": [[0.0, 0.0, 0.0]],
                            "positions_A": [[0.0, 0.0, 0.0]],
                        }
                        for _ in range(20)
                    ],
                },
                "timing": {"median_ms": 1.0, "median_us_per_atom": 1000.0},
            }
        )
    )


def test_run_parser_selects_explicit_torch_backend():
    benchmark = load_benchmark_module()
    args = benchmark.make_parser().parse_args(
        [
            "run",
            "--backend",
            "torch",
            "--device",
            "cuda",
            "--torch-backend",
            "e3nn",
            "--checkpoint",
            "mh1.model",
            "--compact-model",
            "mh1.json",
            "--head",
            "omat_pbe",
            "--repeat",
            "8",
        ]
    )
    assert args.torch_backend == "e3nn"
    assert args.head == "omat_pbe"
    assert args.repeat == 8


def test_git_metadata_is_independent_of_working_directory(monkeypatch, tmp_path):
    benchmark = load_benchmark_module()
    monkeypatch.chdir(tmp_path)
    metadata = benchmark.git_metadata()
    assert metadata["revision"]
    assert metadata["dirty"] is not None


def test_compare_rejects_mixed_devices(tmp_path):
    benchmark = load_benchmark_module()
    symmetrix = tmp_path / "symmetrix.json"
    reference = tmp_path / "reference.json"
    write_record(symmetrix, "symmetrix", "cuda")
    write_record(reference, "torch", "cpu")
    args = argparse.Namespace(
        symmetrix=symmetrix,
        reference=reference,
        energy_atol_per_atom=2.0e-4,
        force_atol=5.0e-3,
        position_atol=2.0e-4,
        output=None,
    )

    with pytest.raises(RuntimeError, match="device differs"):
        benchmark.compare(args)


def test_compare_rejects_swapped_backends(tmp_path):
    benchmark = load_benchmark_module()
    symmetrix = tmp_path / "symmetrix.json"
    reference = tmp_path / "reference.json"
    write_record(symmetrix, "torch", "cpu")
    write_record(reference, "symmetrix", "cpu")
    args = argparse.Namespace(
        symmetrix=symmetrix,
        reference=reference,
        energy_atol_per_atom=2.0e-4,
        force_atol=5.0e-3,
        position_atol=2.0e-4,
        output=None,
    )

    with pytest.raises(RuntimeError, match="not a Symmetrix run"):
        benchmark.compare(args)


def test_compare_rejects_non_jit_symmetrix_record(tmp_path):
    benchmark = load_benchmark_module()
    symmetrix = tmp_path / "symmetrix.json"
    reference = tmp_path / "reference.json"
    write_record(symmetrix, "symmetrix", "cuda")
    write_record(reference, "torch", "cuda")
    record = json.loads(symmetrix.read_text())
    record["identity"]["jit_status"] = "fallback"
    symmetrix.write_text(json.dumps(record))
    args = argparse.Namespace(
        symmetrix=symmetrix,
        reference=reference,
        energy_atol_per_atom=2.0e-4,
        force_atol=5.0e-3,
        position_atol=2.0e-4,
        output=None,
    )

    with pytest.raises(RuntimeError, match="JIT specialization"):
        benchmark.compare(args)


def test_compare_accepts_complete_qualified_trajectory(tmp_path):
    benchmark = load_benchmark_module()
    symmetrix = tmp_path / "symmetrix.json"
    reference = tmp_path / "reference.json"
    output = tmp_path / "comparison.json"
    write_record(symmetrix, "symmetrix", "cuda")
    write_record(reference, "torch", "cuda")
    args = argparse.Namespace(
        symmetrix=symmetrix,
        reference=reference,
        energy_atol_per_atom=2.0e-4,
        force_atol=5.0e-3,
        position_atol=2.0e-4,
        output=output,
    )

    assert benchmark.compare(args) == 0
    assert json.loads(output.read_text())["qualified"] is True


def test_compare_rejects_truncated_trajectory(tmp_path):
    benchmark = load_benchmark_module()
    symmetrix = tmp_path / "symmetrix.json"
    reference = tmp_path / "reference.json"
    write_record(symmetrix, "symmetrix", "cuda")
    write_record(reference, "torch", "cuda")
    record = json.loads(symmetrix.read_text())
    record["outputs"]["trajectory"].pop()
    symmetrix.write_text(json.dumps(record))
    args = argparse.Namespace(
        symmetrix=symmetrix,
        reference=reference,
        energy_atol_per_atom=2.0e-4,
        force_atol=5.0e-3,
        position_atol=2.0e-4,
        output=None,
    )

    with pytest.raises(RuntimeError, match="trajectory length"):
        benchmark.compare(args)
