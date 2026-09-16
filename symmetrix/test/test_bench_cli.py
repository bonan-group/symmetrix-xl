import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from symmetrix.cli import bench
from symmetrix.cli import main as cli


def test_standard_structure_is_deterministic_1080_atom_srtio3():
    first = bench._structure(6, 0.02, 17)
    second = bench._structure(6, 0.02, 17)

    assert len(first) == 1080
    assert first.get_chemical_formula() == "O648Sr216Ti216"
    np.testing.assert_array_equal(first.positions, second.positions)
    assert first.cell.rank == 3
    assert first.pbc.all()


def test_model_path_accepts_mace_mp_name_and_omat_alias(monkeypatch, tmp_path):
    from mace.calculators import foundations_models

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    requested = []

    def download(name):
        requested.append(name)
        return str(checkpoint)

    monkeypatch.setattr(foundations_models, "download_mace_mp_checkpoint", download)

    assert bench._model_path("mace-omat-0-medium") == checkpoint.resolve()
    assert bench._model_path("small") == checkpoint.resolve()
    assert bench._model_path("mace-omat-0-medium") == checkpoint.resolve()
    assert requested == ["medium-omat-0", "small"]


def test_model_path_falls_back_when_name_registry_is_absent(monkeypatch, tmp_path):
    """mace-torch 0.3.10 exposes no mace_mp_names; names must still resolve."""
    from mace.calculators import foundations_models

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        foundations_models, "download_mace_mp_checkpoint", lambda _name: str(checkpoint)
    )
    monkeypatch.delattr(foundations_models, "mace_mp_names", raising=False)

    assert bench._model_path("mace-omat-0-medium") == checkpoint.resolve()


def test_model_path_rejects_unknown_name_when_registry_exists(monkeypatch, tmp_path):
    """Newer mace-torch exposes mace_mp_names; unknown names stay actionable."""
    from mace.calculators import foundations_models

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        foundations_models, "download_mace_mp_checkpoint", lambda _name: "/unused"
    )
    monkeypatch.setattr(
        foundations_models,
        "mace_mp_names",
        [None, "medium-omat-0", "small"],
        raising=False,
    )

    with pytest.raises(
        FileNotFoundError, match="available names: medium-omat-0, small"
    ):
        bench._model_path("not-a-model-name")


def test_benchmark_extraction_cache_reuses_species_specific_json(monkeypatch, tmp_path):
    import symmetrix.extract_mace_data as extraction

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls = []

    def extract(model, **kwargs):
        calls.append((Path(model), kwargs))
        return {"model_type": "MACE", "atomic_numbers": kwargs["species"]}

    monkeypatch.setattr(extraction, "extract_mace_data", extract)
    first, first_hit = bench._prepare_benchmark_model(checkpoint, [38, 8, 22, 8])
    second, second_hit = bench._prepare_benchmark_model(checkpoint, [8, 22, 38])

    assert first == second
    assert not first_hit
    assert second_hit
    assert json.loads(first.read_text())["atomic_numbers"] == [8, 22, 38]
    assert calls == [
        (
            checkpoint,
            {
                "species": [8, 22, 38],
                "num_spline_points": 256,
                "radial_format": "compact",
            },
        )
    ]


def test_directed_edge_count_uses_device_graph_without_host_materialization():
    class Cache:
        device_graph_generation = 7
        num_edges = 1234

    class Calculator:
        _neighbor_cache = Cache()

        def _mace_inputs(self, _atoms):
            raise AssertionError("device graph should not be materialized on host")

    assert bench._directed_edge_count(Calculator(), object()) == 1234


def test_human_cli_reports_primary_metrics(monkeypatch, capsys):
    report = {
        "model": "/models/mace-mpa-0-medium.model",
        "backend": {"selector": "cpu"},
        "dtype": "float32",
        "streamed_edges": "direct",
        "execution_profile": "capacity",
        "execution_plan": {"selected_id": "mh0-direct-speed"},
        "atoms": 1080,
        "directed_edges": 123456,
        "model_cutoff_angstrom": 6.0,
        "neighbor_skin_angstrom": 0.5,
        "effective_cutoff_angstrom": 6.5,
        "median_us_per_atom": 12.345,
        "atoms_per_second": 81004.5,
        "warmups": 3,
        "repeats": 10,
        "kokkos_openmp_threads": 16,
        "blas_threads": 1,
        "model_setup_seconds": 2.0,
        "warmup_seconds": 1.0,
        "measurement_seconds": 3.0,
        "correctness": {
            "status": "passed",
            "errors": {
                "energy_abs_eV_per_atom": 1.0e-6,
                "forces_max_abs_eV_per_A": 2.0e-5,
                "stress_max_abs_eV_per_A3": 3.0e-7,
            },
        },
    }
    monkeypatch.setattr(bench, "run", lambda args: report)

    assert bench.main([]) == 0
    output = capsys.readouterr().out
    assert "1080 atoms, 123456 directed edges" in output
    assert "12.345 us/atom" in output
    assert "81004.5 atoms/s" in output
    assert "6.000 A model + 0.500 A skin = 6.500 A" in output
    assert "warmups=3, samples=10" in output
    assert "Kokkos/OpenMP threads=16, BLAS threads=1" in output
    assert "Setup: 2.00 s; warmup: 1.00 s; measurement: 3.00 s" in output
    assert "mh0-direct-speed" in output
    assert "Correctness: passed against MACE-Torch" in output
    assert "max|dF|=2.000e-05 eV/A" in output


def test_json_cli_preserves_structured_report(monkeypatch, capsys):
    report = {
        "model": "model.json",
        "backend": {"selector": "cuda13-sm120"},
        "dtype": "float32",
        "streamed_edges": "direct",
        "execution_profile": "capacity",
        "execution_plan": {},
        "atoms": 1080,
        "directed_edges": 1,
        "model_cutoff_angstrom": 6.0,
        "neighbor_skin_angstrom": 0.5,
        "effective_cutoff_angstrom": 6.5,
        "median_us_per_atom": 1.0,
        "atoms_per_second": 1_000_000.0,
    }
    monkeypatch.setattr(bench, "run", lambda args: report)

    assert bench.main(["--json"]) == 0
    assert '"selector": "cuda13-sm120"' in capsys.readouterr().out


def test_human_cli_reports_validation_unavailable(monkeypatch, capsys):
    report = {
        "model": "model.model",
        "backend": {"selector": "cuda13-sm120"},
        "dtype": "float32",
        "streamed_edges": "direct",
        "execution_profile": "capacity",
        "execution_plan": {},
        "atoms": 1000,
        "directed_edges": 10000,
        "model_cutoff_angstrom": 6.0,
        "neighbor_skin_angstrom": 0.5,
        "effective_cutoff_angstrom": 6.5,
        "median_us_per_atom": 2.0,
        "atoms_per_second": 500000.0,
        "warmups": 1,
        "repeats": 1,
        "kokkos_openmp_threads": 1,
        "blas_threads": 1,
        "model_setup_seconds": 1.0,
        "warmup_seconds": 1.0,
        "measurement_seconds": 1.0,
        "correctness": {
            "status": "error",
            "reason": "reference validation could not allocate enough memory",
        },
    }
    monkeypatch.setattr(bench, "run", lambda _args: report)

    assert bench.main([]) == 0
    output = capsys.readouterr().out
    assert "Median: 2.000 us/atom" in output
    assert "Correctness: unavailable" in output


def test_run_rejects_invalid_measurement_counts():
    args = Namespace(warmups=-1, repeats=1)
    with pytest.raises(ValueError, match="warmups"):
        bench.run(args)


def test_json_model_correctness_has_explicit_skip(tmp_path):
    model = tmp_path / "model.json"
    model.write_text("{}")

    result = bench._validate_against_mace_torch(
        model, object(), {}, "float32", {"backend": "cpu"}
    )

    assert result == {
        "status": "skipped",
        "reference": "mace-torch",
        "reason": "an extracted JSON model has no MACE-Torch checkpoint",
    }


def test_skip_validation_is_explicit_and_nonallocating(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("reference calculator must not be constructed")

    monkeypatch.setattr(bench, "_validate_against_mace_torch", fail)
    assert bench._run_validation(
        Path("model.model"), object(), {}, "float32", {"backend": "cuda"}, skip=True
    ) == {
        "status": "skipped",
        "reference": "mace-torch",
        "reason": "disabled by --skip-validation",
    }


def test_validation_oom_is_recorded_after_benchmark(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory while constructing reference model")

    monkeypatch.setattr(bench, "_validate_against_mace_torch", fail)
    result = bench._run_validation(
        Path("model.model"), object(), {}, "float32", {"backend": "cuda"}, skip=False
    )
    assert result["status"] == "error"
    assert "not allocate enough memory" in result["reason"]
    assert result["exception_type"] == "RuntimeError"


def test_mace_torch_correctness_reports_normalized_errors(monkeypatch, tmp_path):
    from ase import Atoms

    model = tmp_path / "model.model"
    model.write_bytes(b"checkpoint")
    constructed = {}

    class ReferenceCalculator:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.results = {}

        def calculate(self, atoms, properties, system_changes):
            assert properties == ["energy", "forces", "stress"]
            assert system_changes
            self.results = {
                "energy": 4.0001,
                "forces": np.full((2, 3), 2.0e-5),
                "stress": np.full(6, 3.0e-6),
            }

    mace = types.ModuleType("mace")
    calculators = types.ModuleType("mace.calculators")
    calculators.MACECalculator = ReferenceCalculator
    mace.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    monkeypatch.setattr(bench.importlib.util, "find_spec", lambda _name: object())
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    result = bench._validate_against_mace_torch(
        model,
        atoms,
        {
            "energy": 4.0,
            "forces": np.zeros((2, 3)),
            "stress": np.zeros(6),
        },
        "float32",
        {"backend": "cpu"},
    )

    assert result["status"] == "passed"
    assert result["errors"]["energy_abs_eV_per_atom"] == pytest.approx(5.0e-5)
    assert result["errors"]["forces_max_abs_eV_per_A"] == pytest.approx(2.0e-5)
    assert result["errors"]["stress_max_abs_eV_per_A3"] == pytest.approx(3.0e-6)
    assert constructed == {
        "model_paths": str(model),
        "device": "cpu",
        "default_dtype": "float32",
    }


def test_mace_torch_correctness_fails_above_tolerance(monkeypatch, tmp_path):
    from ase import Atoms

    model = tmp_path / "model.model"
    model.write_bytes(b"checkpoint")

    class ReferenceCalculator:
        def __init__(self, **_kwargs):
            self.results = {}

        def calculate(self, atoms, properties, system_changes):
            self.results = {
                "energy": 0.0,
                "forces": np.ones((1, 3)),
                "stress": np.zeros(6),
            }

    mace = types.ModuleType("mace")
    calculators = types.ModuleType("mace.calculators")
    calculators.MACECalculator = ReferenceCalculator
    mace.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    monkeypatch.setattr(bench.importlib.util, "find_spec", lambda _name: object())

    with pytest.raises(RuntimeError, match="forces_max_abs_eV_per_A"):
        bench._validate_against_mace_torch(
            model,
            Atoms("H"),
            {"energy": 0.0, "forces": np.zeros((1, 3)), "stress": np.zeros(6)},
            "float32",
            {"backend": "cpu"},
        )


@pytest.mark.parametrize(
    ("selector", "initial", "expected"),
    (("cpu", "cuda13-sm120", "cpu"), ("auto", "cpu", None)),
)
def test_backend_argument_is_applied_before_model_loading(
    monkeypatch, selector, initial, expected
):
    args = Namespace(
        warmups=0,
        repeats=1,
        threads=1,
        backend=selector,
        model="mace-mpa-0-medium",
    )
    monkeypatch.setenv("SYMMETRIX_BACKEND", initial)

    def stop_after_checking_backend(_model):
        assert os.environ.get("SYMMETRIX_BACKEND") == expected
        raise FileNotFoundError("stop before model import")

    monkeypatch.setattr(bench, "_model_path", stop_after_checking_backend)
    with pytest.raises(FileNotFoundError, match="stop before model import"):
        bench.run(args)


def test_main_dispatches_bench_arguments(monkeypatch):
    received = []
    monkeypatch.setattr(bench, "main", lambda argv: received.extend(argv) or 0)

    assert cli.main(["bench", "--repeats", "4"]) == 0
    assert received == ["--repeats", "4"]
