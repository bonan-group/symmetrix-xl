import importlib.util
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
    assert benchmark_module.directed_edges(
        first, 6.5
    ) >= benchmark_module.directed_edges(first, 6.0)


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
    import json

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
