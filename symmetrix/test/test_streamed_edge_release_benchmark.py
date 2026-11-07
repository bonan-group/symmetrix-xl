import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def benchmark_module():
    path = (
        Path(__file__).parents[2] / "benchmarks" / "streamed_edge_release_benchmark.py"
    )
    specification = importlib.util.spec_from_file_location("release_benchmark", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_build_atoms_and_edge_counts_are_deterministic(benchmark_module):
    first = benchmark_module.build_atoms(2, 11)
    second = benchmark_module.build_atoms(2, 11)
    assert len(first) == 32
    assert np.array_equal(first.positions, second.positions)
    assert benchmark_module.input_state_sha256(
        first
    ) == benchmark_module.input_state_sha256(second)
    assert benchmark_module.directed_edges(
        first, 6.5
    ) >= benchmark_module.directed_edges(first, 6.0)


def test_srtio3_medium_case_builds_deterministic_5000_atom_periodic_cell(
    benchmark_module,
):
    first = benchmark_module.build_atoms(10, 17, structure="srtio3-cubic")
    second = benchmark_module.build_atoms(10, 17, structure="srtio3-cubic")

    assert len(first) == 5000
    assert first.pbc.all()
    assert first.cell.lengths() == pytest.approx([39.05, 39.05, 39.05])
    assert first.get_chemical_symbols().count("Sr") == 1000
    assert first.get_chemical_symbols().count("Ti") == 1000
    assert first.get_chemical_symbols().count("O") == 3000
    assert np.array_equal(first.positions, second.positions)


def test_medium_srtio3_parser_exposes_paper_protocol(benchmark_module, tmp_path):
    args = benchmark_module.parser().parse_args(
        [
            "run",
            "--implementation",
            "pytorch",
            "--pytorch-backend",
            "cueq",
            "--device",
            "cuda",
            "--structure",
            "srtio3-cubic",
            "--repeat",
            "10",
            "--seed",
            "17",
            "--checkpoint",
            str(tmp_path / "medium.model"),
            "--revision",
            "test",
            "--output",
            str(tmp_path / "record.json"),
        ]
    )

    assert args.structure == "srtio3-cubic"
    assert args.repeat == 10
    assert args.pytorch_backend == "cueq"
    assert args.seed == 17
    assert args.warmups == 3
    assert args.samples == 10


def test_medium_srtio3_workload_metadata_records_actual_case(benchmark_module):
    args = SimpleNamespace(
        structure="srtio3-cubic",
        repeat=10,
        seed=17,
        dtype="float32",
        implementation="current",
        skin=0.5,
    )
    atoms = benchmark_module.build_atoms(10, args.seed, structure=args.structure)

    workload = benchmark_module.workload_metadata(atoms, args, 6.0, 6.5)

    assert workload["structure"] == "perturbed-cubic-SrTiO3"
    assert workload["repeat"] == 10
    assert workload["seed"] == 17
    assert workload["atoms"] == 5000
    assert workload["chemical_formula"] == "O3000Sr1000Ti1000"
    assert workload["species"] == ["Sr", "Ti", "O"]
    assert workload["pbc"] == [True, True, True]
    assert len(workload["input_state_sha256"]) == 64
    assert workload["position_noise_std_A"] == 0.01
    assert workload["properties"] == ["energy", "forces", "stress"]
    assert workload["precision"] == "float32"
    assert workload["effective_cutoff_A"] == 6.5
    assert workload["directed_edges_at_model_cutoff"] > 0
    assert (
        workload["directed_edges_at_effective_cutoff"]
        >= workload["directed_edges_at_model_cutoff"]
    )


def test_cuda_synchronize_uses_symmetrix_fence_without_torch(
    benchmark_module, monkeypatch
):
    calls = []
    calculator = SimpleNamespace(
        evaluator=SimpleNamespace(fence=lambda: calls.append("fenced"))
    )
    monkeypatch.setitem(sys.modules, "torch", None)

    benchmark_module.synchronize("cuda", calculator)

    assert calls == ["fenced"]


def test_cuda_synchronize_accepts_synchronous_symmetrix_evaluator(
    benchmark_module, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch", None)

    benchmark_module.synchronize("cuda", SimpleNamespace(evaluator=object()))


def test_isolate_symmetrix_stage_removes_only_symmetrix_finder(
    benchmark_module, monkeypatch, tmp_path
):
    SymmetrixFinder = type(
        "SymmetrixFinder", (), {"__module__": "_editable_skbc_symmetrix"}
    )
    OtherFinder = type("OtherFinder", (), {"__module__": "other_finder"})
    symmetrix_finder = SymmetrixFinder()
    other_finder = OtherFinder()
    monkeypatch.setattr(
        benchmark_module.sys, "meta_path", [symmetrix_finder, other_finder]
    )
    monkeypatch.setenv("SYMMETRIX_BENCHMARK_PACKAGE_ROOT", str(tmp_path))
    benchmark_module.isolate_symmetrix_stage()
    assert benchmark_module.sys.meta_path == [other_finder]
    assert benchmark_module.sys.path[0] == str(tmp_path.resolve())


def test_compare_records_reports_numerical_errors(tmp_path, benchmark_module):
    def record(implementation, energy, force, stress, milliseconds):
        return {
            "identity": {"implementation": implementation, "streamed_edges": "direct"},
            "workload": {"atoms": 2},
            "timing": {
                "median_ms": milliseconds,
                "median_us_per_atom": milliseconds * 500,
            },
            "outputs": {
                "energy_eV": energy,
                "forces_eV_per_A": [[force, 0, 0], [0, 0, 0]],
                "stress_eV_per_A3_voigt": [stress, 0, 0, 0, 0, 0],
            },
        }

    paths = []
    for index, data in enumerate(
        (
            record("pytorch-mace-e3nn", 2.0, 1.0, 0.5, 4.0),
            record("symmetrix-streamed-edge", 2.2, 1.2, 0.7, 2.0),
        )
    ):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(data))
        paths.append(path)
    report = benchmark_module.compare_records(paths)
    candidate = report["rows"][1]
    assert candidate["speedup_vs_pytorch"] == pytest.approx(2.0)
    assert candidate["energy_abs_error_eV_per_atom"] == pytest.approx(0.1)
    assert candidate["force_max_abs_error_eV_per_A"] == pytest.approx(0.2)
    assert candidate["stress_max_abs_error_eV_per_A3"] == pytest.approx(0.2)


def test_compare_records_rejects_different_protocols(tmp_path, benchmark_module):
    def record(implementation, protocol):
        return {
            "protocol": protocol,
            "identity": {"implementation": implementation},
            "workload": {"atoms": 2, "structure": "perturbed-cubic-SrTiO3"},
            "timing": {"median_ms": 1.0, "median_us_per_atom": 500.0},
            "outputs": {
                "energy_eV": 1.0,
                "forces_eV_per_A": [[0.0, 0.0, 0.0]] * 2,
                "stress_eV_per_A3_voigt": [0.0] * 6,
            },
        }

    paths = []
    for index, data in enumerate(
        (record("pytorch-mace-e3nn", "evaluation"), record("symmetrix", "nvt"))
    ):
        path = tmp_path / f"protocol-{index}.json"
        path.write_text(json.dumps(data))
        paths.append(path)

    with pytest.raises(ValueError, match="protocols differ"):
        benchmark_module.compare_records(paths)


def test_compare_records_prefers_cueq_reference(tmp_path, benchmark_module):
    def record(implementation, milliseconds):
        return {
            "protocol": "complete-ase-energy-forces-stress",
            "identity": {"implementation": implementation},
            "model": {"head": "default", "cutoff_A": 6.0},
            "workload": {
                "atoms": 2,
                "structure": "perturbed-cubic-SrTiO3",
                "repeat": 1,
                "seed": 17,
                "properties": ["energy", "forces", "stress"],
                "precision": "float32",
            },
            "timing": {
                "median_ms": milliseconds,
                "median_us_per_atom": milliseconds * 500,
            },
            "outputs": {
                "energy_eV": 1.0,
                "forces_eV_per_A": [[0.0, 0.0, 0.0]] * 2,
                "stress_eV_per_A3_voigt": [0.0] * 6,
            },
        }

    paths = []
    for index, data in enumerate(
        (
            record("pytorch-mace-e3nn", 4.0),
            record("pytorch-mace-cueq", 2.0),
            record("symmetrix-streamed-edge", 1.0),
        )
    ):
        path = tmp_path / f"cueq-reference-{index}.json"
        path.write_text(json.dumps(data))
        paths.append(path)

    rows = benchmark_module.compare_records(paths)["rows"]

    assert rows[0]["speedup_vs_pytorch"] == pytest.approx(0.5)
    assert rows[1]["speedup_vs_pytorch"] == pytest.approx(1.0)
    assert rows[2]["speedup_vs_pytorch"] == pytest.approx(2.0)


def test_parser_defaults_to_e3nn_and_accepts_cueq(benchmark_module, tmp_path):
    common = [
        "run",
        "--implementation",
        "pytorch",
        "--device",
        "cuda",
        "--checkpoint",
        str(tmp_path / "model.pt"),
        "--revision",
        "test",
        "--output",
        str(tmp_path / "result.json"),
    ]
    assert benchmark_module.parser().parse_args(common).pytorch_backend == "e3nn"
    assert (
        benchmark_module.parser()
        .parse_args([*common, "--pytorch-backend", "cueq"])
        .pytorch_backend
        == "cueq"
    )
    assert benchmark_module.parser().parse_args(common).head == "default"


def test_pytorch_head_identity_rejects_silent_fallback(benchmark_module):
    calculator = SimpleNamespace(head="default", available_heads=["default"])
    assert benchmark_module.pytorch_head_identity(calculator, "default") == {
        "requested_head": "default",
        "selected_head": "default",
        "available_heads": ["default"],
    }
    with pytest.raises(RuntimeError, match="selected head 'default'.*'omat_pbe'"):
        benchmark_module.pytorch_head_identity(calculator, "omat_pbe")
