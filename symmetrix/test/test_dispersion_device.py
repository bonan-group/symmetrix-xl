from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "dispersion_kokkos_timing.py"


@pytest.fixture(scope="module")
def benchmark_module():
    spec = importlib.util.spec_from_file_location("dispersion_kokkos_timing", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _structure() -> Atoms:
    return Atoms(
        numbers=[6, 8, 7, 1],
        positions=[
            [0.7, 0.8, 0.9],
            [2.4, 1.1, 1.3],
            [1.5, 3.0, 1.8],
            [3.2, 2.7, 3.1],
        ],
        cell=[[7.5, 0.2, 0.1], [0.0, 8.0, 0.3], [0.0, 0.0, 8.5]],
        pbc=True,
    )


def _configuration(method: str):
    from symmetrix import D2Correction, D3Correction

    if method == "d2":
        return D2Correction(functional="pbe", cutoff=5.0, skin=0.5)
    return D3Correction(
        functional="pbe",
        damping="bj",
        cutoff=5.0,
        coordination_cutoff=4.0,
        skin=0.5,
    )


def _pipeline(method: str, precision: str):
    import symmetrix
    from symmetrix.corrections import CorrectionPipeline

    native = symmetrix.load_backend()
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    dtype = np.float32 if precision == "float32" else np.float64
    return CorrectionPipeline([_configuration(method)], native=native, dtype=dtype)


def _worker_payload(method: str, precision: str) -> dict[str, object]:
    import symmetrix

    pipeline = _pipeline(method, precision)
    result = pipeline.calculate(_structure(), ("energy", "forces", "stress"))
    return {
        "backend": symmetrix.selected_backend()["backend"],
        "energy": result["energy"],
        "forces": np.asarray(result["forces"]).tolist(),
        "stress": np.asarray(result["stress"]).tolist(),
        "diagnostics": dict(pipeline.diagnostics()[0]),
    }


def _run_worker(method: str, precision: str, backend: str) -> dict[str, object]:
    environment = os.environ.copy()
    environment["SYMMETRIX_BACKEND"] = backend
    environment["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", method, precision],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def _assert_parity(reference, candidate, precision):
    if precision == "float32":
        energy_atol, force_atol, stress_atol = 5e-5, 8e-5, 8e-6
    else:
        energy_atol, force_atol, stress_atol = 2e-11, 3e-11, 3e-12
    assert candidate["energy"] == pytest.approx(
        reference["energy"], rel=0.0, abs=energy_atol
    )
    np.testing.assert_allclose(
        candidate["forces"], reference["forces"], rtol=0.0, atol=force_atol
    )
    np.testing.assert_allclose(
        candidate["stress"], reference["stress"], rtol=0.0, atol=stress_atol
    )


def _exercise_device_graph_reuse(method, monkeypatch, expected_backend):
    from symmetrix import corrections

    def fail_host_builder(*_args, **_kwargs):
        raise AssertionError("eligible periodic correction used the host graph builder")

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(corrections, "build_host_neighbor_graph", fail_host_builder)
    pipeline = _pipeline(method, "float64")
    atoms = _structure()

    first = pipeline.calculate(atoms, ("energy", "forces", "stress"))
    first_diagnostics = pipeline.diagnostics()[0]
    assert first_diagnostics["graph_backend"] == expected_backend
    assert first_diagnostics["graph_build_count"] == 1
    assert first_diagnostics["graph_reuse_count"] == 0
    assert first_diagnostics["num_edges"] > 0
    assert first_diagnostics["backend_fallback_count"] == 0
    if method == "d3":
        coordination = pipeline._runtimes[0].evaluator._coordination_path_diagnostics()
        assert coordination["execution_path"] == "team"
        assert coordination["team_launch_count"] == 1
        assert coordination["serial_fallback_launch_count"] == 0
    assert np.isfinite(first["energy"])
    assert np.all(np.isfinite(first["forces"]))
    assert np.all(np.isfinite(first["stress"]))

    original = atoms.positions.copy()
    for offset in (0.05, -0.05, 0.08):
        atoms.positions[:] = original
        atoms.positions[0, 0] += offset
        values = pipeline.calculate(atoms, ("energy", "forces", "stress"))
        assert np.isfinite(values["energy"])

    reused = pipeline.diagnostics()[0]
    assert reused["graph_build_count"] == 1
    assert reused["graph_reuse_count"] == 3
    assert reused["backend_fallback_count"] == 0
    if method == "d3":
        coordination = pipeline._runtimes[0].evaluator._coordination_path_diagnostics()
        assert coordination["team_launch_count"] == 4
        assert coordination["serial_fallback_launch_count"] == 0

    atoms.positions[:] = original
    atoms.positions[0, 0] += 0.3
    pipeline.calculate(atoms, ("energy", "forces", "stress"))
    rebuilt = pipeline.diagnostics()[0]
    assert rebuilt["graph_build_count"] == 2
    assert rebuilt["graph_reuse_count"] == 3
    assert rebuilt["backend_fallback_count"] == 0
    if method == "d3":
        coordination = pipeline._runtimes[0].evaluator._coordination_path_diagnostics()
        assert coordination["team_launch_count"] == 5
        assert coordination["serial_fallback_launch_count"] == 0


@pytest.mark.parametrize("method", ("d2", "d3"))
def test_cpu_dispersion_precision_parity(method):
    reference = _run_worker(method, "float64", "cpu")
    candidate = _run_worker(method, "float32", "cpu")

    assert reference["backend"] == candidate["backend"] == "cpu"
    _assert_parity(reference, candidate, "float32")


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("method", ("d2", "d3"))
def test_cuda_dispersion_device_graph_reuse(method, monkeypatch):
    _exercise_device_graph_reuse(method, monkeypatch, "kokkos")


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("method", ("d2", "d3"))
def test_cuda_dispersion_precision_parity(method):
    import symmetrix

    reference = _run_worker(method, "float64", "cpu")
    selector = symmetrix.selected_backend()["selector"]
    for precision in ("float64", "float32"):
        candidate = _run_worker(method, precision, selector)
        assert candidate["backend"] == "cuda"
        _assert_parity(reference, candidate, precision)


@pytest.mark.gpu
@pytest.mark.hip
@pytest.mark.parametrize("method", ("d2", "d3"))
def test_hip_dispersion_device_graph_reuse(method, monkeypatch):
    _exercise_device_graph_reuse(method, monkeypatch, "kokkos")


@pytest.mark.gpu
@pytest.mark.hip
@pytest.mark.parametrize("method", ("d2", "d3"))
def test_hip_dispersion_precision_parity(method):
    import symmetrix

    reference = _run_worker(method, "float64", "cpu")
    selector = symmetrix.selected_backend()["selector"]
    for precision in ("float64", "float32"):
        candidate = _run_worker(method, precision, selector)
        assert candidate["backend"] == "hip"
        _assert_parity(reference, candidate, precision)


def test_forced_kokkos_prepare_failure_does_not_fallback(monkeypatch):
    pipeline = _pipeline("d2", "float64")
    runtime = pipeline._runtimes[0]
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")

    class FailingPeriodicPrepare:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def prepare_periodic_graph(self, *_args):
            raise RuntimeError("device build failed")

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    runtime.evaluator = FailingPeriodicPrepare(runtime.evaluator)

    with pytest.raises(RuntimeError, match="device build failed"):
        pipeline.calculate(_structure(), ("energy",))
    assert pipeline.diagnostics()[0]["backend_fallback_count"] == 0


def test_d3_runtime_diagnostics_report_actual_workspace_extents():
    pipeline = _pipeline("d3", "float64")
    pipeline.calculate(_structure(), ("energy", "forces", "stress"))

    diagnostics = pipeline.diagnostics()[0]

    assert diagnostics["reference_weight_active_atoms"] == 4
    assert diagnostics["reference_weight_capacity_atoms"] >= 4
    assert diagnostics["reference_weight_workspace_bytes"] == (
        2 * diagnostics["reference_weight_capacity_atoms"] * 5 * 8
    )
    assert diagnostics["coordination_cache_active_edges"] == diagnostics["num_edges"]
    assert diagnostics["coordination_cache_capacity_edges"] >= diagnostics["num_edges"]
    assert diagnostics["coordination_cache_workspace_bytes"] == (
        2 * diagnostics["coordination_cache_capacity_edges"] * 8
    )
    assert diagnostics["d3_workspace_bytes"] == (
        diagnostics["reference_weight_workspace_bytes"]
        + diagnostics["coordination_cache_workspace_bytes"]
    )


def _benchmark_driver_command(output: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(BENCHMARK),
        "--backend",
        "cpu",
        "--method",
        "d3",
        "--damping",
        "bj",
        "--functional",
        "pbe",
        "--precision",
        "float64",
        "--atoms",
        "8",
        "--pair-cutoff",
        "4.0",
        "--coordination-cutoff",
        "3.0",
        "--skin",
        "0.5",
        "--warmup",
        "1",
        "--repeat",
        "2",
        "--output",
        str(output),
        *extra,
    ]


def _fourier_benchmark_driver_command(output: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(BENCHMARK),
        "--backend",
        "cpu",
        "--method",
        "fourier-d3",
        "--functional",
        "pbe",
        "--precision",
        "float64",
        "--structure",
        "graphite",
        "--atoms",
        "8",
        "--coordination-cutoff",
        "3.0",
        "--skin",
        "0.5",
        "--mesh-spacing",
        "2.0",
        "--spline-order",
        "4",
        "--chunk-size",
        "512",
        "--warmup",
        "1",
        "--repeat",
        "2",
        "--output",
        str(output),
        *extra,
    ]


def test_benchmark_driver_emits_validated_carbon_record_by_default(tmp_path):
    output = tmp_path / "record.json"
    completed = subprocess.run(
        _benchmark_driver_command(output),
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(completed.stdout)

    assert json.loads(output.read_text()) == record
    assert record["schema"] == "symmetrix.dispersion-kokkos-timing"
    assert record["version"] == 4
    assert record["structure"] == "carbon-grid"
    assert record["structure_repeats"] is None
    assert record["properties"] == ["energy", "forces", "stress"]
    assert record["metric"] == "us/atom"
    assert record["backend"] == "cpu"
    assert record["cpu_count"] == 1
    assert record["method"] == "d3"
    assert record["damping"] == "bj"
    assert record["functional"] == "pbe"
    assert record["precision"] == "float64"
    assert record["atoms"] == 8
    assert record["pair_cutoff"] == 4.0
    assert record["coordination_cutoff"] == 3.0
    assert record["skin"] == 0.5
    assert record["warmup"] == 1
    assert record["repeat"] == 2
    assert record["effective_cutoff"] == 4.5
    assert record["directed_edges"] > 0
    assert record["graph_build_us"] > 0.0
    assert record["reuse_step_us"] > 0.0
    assert record["peak_memory_bytes"] > 0
    assert record["graph_build_count"] == 1
    assert record["graph_reuse_count"] == 3
    assert record["backend_fallback_count"] == 0
    assert record["reference_weight_active_atoms"] == 8
    assert record["reference_weight_capacity_atoms"] >= 8
    assert record["reference_weight_workspace_bytes"] == (
        2 * record["reference_weight_capacity_atoms"] * 5 * 8
    )
    assert record["coordination_cache_active_edges"] == record["directed_edges"]
    assert record["coordination_cache_capacity_edges"] >= record["directed_edges"]
    assert record["coordination_cache_workspace_bytes"] == (
        2 * record["coordination_cache_capacity_edges"] * 8
    )
    assert record["d3_workspace_bytes"] == (
        record["reference_weight_workspace_bytes"]
        + record["coordination_cache_workspace_bytes"]
    )
    assert record["cpu_model"]
    assert len(record["cpu_affinity"]) == 1
    assert set(record["thread_counts"].values()) == {1}


def test_benchmark_driver_selects_requested_physical_cpu_count(tmp_path):
    output = tmp_path / "record.json"
    command = _benchmark_driver_command(output, "--cpu-count", "2")
    environment = os.environ.copy()
    environment.update(OMP_PROC_BIND="close", OMP_PLACES="cores")

    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=environment
    )
    record = json.loads(completed.stdout)

    assert len(record["cpu_affinity"]) == 2
    assert record["cpu_count"] == 2
    assert set(record["thread_counts"].values()) == {2}


def test_benchmark_driver_selects_deterministic_srtio3_structure(tmp_path):
    output = tmp_path / "record.json"
    command = _benchmark_driver_command(output)
    atoms_index = command.index("--atoms") + 1
    command[atoms_index] = "10"
    command.extend(("--structure", "srtio3"))

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    record = json.loads(completed.stdout)

    assert record["structure"] == "srtio3"
    assert record["structure_repeats"] == [1, 1, 2]
    assert record["atoms"] == 10
    assert record["properties"] == ["energy", "forces", "stress"]
    assert record["directed_edges"] > 0
    assert record["graph_build_count"] == 1
    assert record["graph_reuse_count"] == 3
    assert record["backend_fallback_count"] == 0
    assert record["d3_workspace_bytes"] > 0
    assert record["reuse_step_us"] > 0.0


def test_benchmark_driver_rejects_non_unit_cell_srtio3_atom_count(tmp_path):
    output = tmp_path / "record.json"
    command = _benchmark_driver_command(output)
    atoms_index = command.index("--atoms") + 1
    command[atoms_index] = "6"
    command.extend(("--structure", "srtio3"))

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 2
    assert "divisible by five" in completed.stderr
    assert not output.exists()


def test_benchmark_driver_builds_exact_repeated_srtio3_cell(benchmark_module):
    atoms, repeats = benchmark_module._build_atoms(500, "srtio3")

    numbers, counts = np.unique(atoms.numbers, return_counts=True)
    assert dict(zip(numbers.tolist(), counts.tolist(), strict=True)) == {
        8: 300,
        22: 100,
        38: 100,
    }
    assert len(atoms) == 500
    assert repeats == (4, 5, 5)
    assert np.prod(repeats) == 100
    np.testing.assert_allclose(
        np.diag(atoms.cell.array), [15.62, 19.525, 19.525], rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        atoms.cell.array - np.diag(np.diag(atoms.cell.array)),
        0.0,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(atoms.pbc, [True, True, True])


@pytest.mark.parametrize(
    "structure,numbers",
    (
        ("graphite", {6: 1000}),
        ("layered-bn", {5: 500, 7: 500}),
    ),
)
def test_benchmark_driver_builds_exact_layered_cells(
    benchmark_module, structure, numbers
):
    atoms, repeats = benchmark_module._build_atoms(1000, structure)

    unique, counts = np.unique(atoms.numbers, return_counts=True)
    assert dict(zip(unique.tolist(), counts.tolist(), strict=True)) == numbers
    assert len(atoms) == 1000
    assert repeats == (5, 5, 10)
    assert np.prod(repeats) == 250
    assert np.all(np.asarray(atoms.pbc))
    assert atoms.cell.volume > 0.0


def test_fourier_benchmark_record_pins_accuracy_and_reuse_state(tmp_path):
    import symmetrix

    native = symmetrix.load_backend("cpu")
    if not hasattr(native, "FourierD3Kokkos"):
        pytest.skip("the active CPU build has no FourierD3 FFT provider")
    output = tmp_path / "fourier-record.json"
    completed = subprocess.run(
        _fourier_benchmark_driver_command(output),
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(completed.stdout)

    assert json.loads(output.read_text()) == record
    assert record["version"] == 4
    assert record["method"] == "fourier-d3"
    assert record["structure"] == "graphite"
    assert record["pair_cutoff"] is None
    assert record["mesh_spacing_requested"] == 2.0
    assert record["spline_order"] == 4
    assert record["rank_chunk"] > 0
    assert record["resolved_mesh"]
    assert record["capability_provider"] == "fftw3"
    assert record["backend_fallback_count"] == 0
    assert record["timed_allocation_growth"] == 0
    assert record["timed_capacity_growth_bytes"] == 0
    assert record["timed_graph_rebuilds"] == 0
    assert record["timed_plan_builds"] == 0
    assert record["timed_mesh_invalidations"] == 0
    assert record["timed_kernel_invalidations"] == 0
    assert record["active_bytes"] > 0
    assert record["capacity_bytes"] >= record["active_bytes"]
    assert np.isfinite(record["total_energy_ev"])
    assert record["energy_ev_per_atom"] == pytest.approx(
        record["total_energy_ev"] / record["atoms"]
    )
    assert np.isfinite(record["force_rms_ev_per_angstrom"])
    assert np.isfinite(record["out_of_plane_stress_ev_per_angstrom3"])


@pytest.fixture(scope="module")
def benchmark_record_and_args(tmp_path_factory, benchmark_module):
    output = tmp_path_factory.mktemp("dispersion-benchmark") / "record.json"
    command = _benchmark_driver_command(output)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    args = benchmark_module._parser().parse_args(command[2:])
    return json.loads(completed.stdout), args


@pytest.mark.parametrize(
    "corruption",
    (
        "edge_count",
        "graph_build_count",
        "graph_reuse_count",
        "backend",
        "precision",
        "pair_cutoff",
        "coordination_cutoff",
        "skin",
        "reference_workspace",
        "coordination_workspace",
        "total_workspace",
        "graph_build_timing",
        "reuse_timing",
        "energy",
        "timed_growth",
    ),
)
def test_benchmark_driver_validator_rejects_malformed_required_fields(
    benchmark_module, benchmark_record_and_args, corruption
):
    valid, args = benchmark_record_and_args
    record = copy.deepcopy(valid)
    if corruption == "edge_count":
        record["directed_edges"] += 1
        record["coordination_cache_active_edges"] += 1
        record["coordination_cache_capacity_edges"] += 1
        record["coordination_cache_workspace_bytes"] += 2 * 8
        record["d3_workspace_bytes"] += 2 * 8
    elif corruption in {"graph_build_count", "graph_reuse_count"}:
        record[corruption] += 1
    elif corruption == "backend":
        record["backend"] = "hip"
    elif corruption == "precision":
        record["precision"] = "float32"
    elif corruption in {"pair_cutoff", "coordination_cutoff", "skin"}:
        record[corruption] += 0.25
    elif corruption == "reference_workspace":
        record["reference_weight_workspace_bytes"] += 1
    elif corruption == "coordination_workspace":
        record["coordination_cache_workspace_bytes"] += 1
    elif corruption == "total_workspace":
        record["d3_workspace_bytes"] += 1
    elif corruption == "graph_build_timing":
        record["graph_build_us"] = 0.0
    elif corruption == "reuse_timing":
        record["reuse_step_us"] = 0.0
    elif corruption == "energy":
        record["total_energy_ev"] = float("nan")
    elif corruption == "timed_growth":
        record["timed_graph_rebuilds"] = 1
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(f"unknown corruption {corruption}")

    with pytest.raises(RuntimeError):
        benchmark_module._validate_record(record, args)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--worker":
    print(json.dumps(_worker_payload(sys.argv[2], sys.argv[3]), sort_keys=True))
