import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/materials_project_correctness.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("mp_correctness_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Atoms:
    def __init__(self, nonpolar, polar, source_index):
        self.info = {
            "nonpolar_mpid": nonpolar,
            "polar_mpid": polar,
            "macefield_source_index": source_index,
        }


def test_endpoint_sampling_is_unique_and_deterministic():
    benchmark = _load_benchmark()
    structures = []
    for path in range(12):
        for image in range(10):
            structures.append(_Atoms(f"mp-n{path}", f"mp-p{path}", 10 * path + image))

    first = benchmark._sample_endpoints(structures, count=10, seed=20260806)
    second = benchmark._sample_endpoints(structures, count=10, seed=20260806)
    assert [row["material_id"] for row in first] == [
        row["material_id"] for row in second
    ]
    assert len({row["material_id"] for row in first}) == 10
    for row in first:
        source_index = row["atoms"].info["macefield_source_index"]
        assert source_index % 10 in (0, 9)


def test_endpoint_sampling_rejects_incomplete_paths():
    benchmark = _load_benchmark()
    structures = [_Atoms("mp-a", "mp-b", image) for image in range(9)]
    with pytest.raises(ValueError, match="has 9 images, expected 10"):
        benchmark._sample_endpoints(structures, count=1, seed=1)


def _result(material_id="mp-1", energy=4.0, force_offset=0.0, stress_offset=0.0):
    return {
        "material_id": material_id,
        "formula": "AlN",
        "atoms": 2,
        "energy_eV": energy,
        "forces_eV_per_A": (
            np.arange(6, dtype=float).reshape(2, 3) + force_offset
        ).tolist(),
        "stress_eV_per_A3": (np.arange(6, dtype=float) + stress_offset).tolist(),
    }


def test_error_metrics_are_normalized_and_componentwise():
    benchmark = _load_benchmark()
    metrics = benchmark._error_metrics(
        _result(),
        _result(energy=4.2, force_offset=0.003, stress_offset=-0.004),
    )
    assert metrics["energy_abs_eV"] == pytest.approx(0.2)
    assert metrics["energy_abs_eV_per_atom"] == pytest.approx(0.1)
    assert metrics["forces_max_abs_eV_per_A"] == pytest.approx(0.003)
    assert metrics["stress_max_abs_eV_per_A3"] == pytest.approx(0.004)


def test_comparison_fails_any_exceeded_tolerance():
    benchmark = _load_benchmark()
    reference = {"profile": "torch_cpu_f64", "results": [_result()]}
    candidate = {
        "profile": "cuda_all_retained_f32",
        "dtype": "float32",
        "results": [_result(force_offset=0.006)],
    }
    comparison = benchmark._compare(reference, candidate)
    assert not comparison["passed"]
    assert not comparison["structures"][0]["passed"]


def test_requested_standard_policy_must_be_selected():
    benchmark = _load_benchmark()
    profile = benchmark.PROFILES["cuda_all_standard_m0_f32"]

    class Evaluator:
        standard_m0_selected_executor = "runtime"
        m1_polynomial_policy = "retained"

    class Calculator:
        streamed_edges = "all_interactions"
        jit_status = "disabled"
        jit_artifact_id = None
        evaluator = Evaluator()

    with pytest.raises(RuntimeError, match="requested standard M0"):
        benchmark._selection_report(Calculator(), profile)


def test_help_lists_all_profiles(monkeypatch, capsys):
    benchmark = _load_benchmark()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "run", "--help"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for profile in benchmark.PROFILES:
        assert profile in output
