import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def benchmark_module():
    path = Path(__file__).parents[2] / "benchmarks" / "polar_torch_baseline.py"
    specification = importlib.util.spec_from_file_location("polar_torch_baseline", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_parser_requires_oracles_preserves_e3nn_default_and_accepts_cueq(
    benchmark_module,
    monkeypatch,
):
    monkeypatch.delenv("SYMMETRIX_POLAR_ORACLE_MACE", raising=False)
    monkeypatch.delenv("SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE", raising=False)
    parser = benchmark_module.make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    oracle_args = [
        "--oracle-mace",
        "/path/to/mace",
        "--oracle-graph-electrostatics",
        "/path/to/graph-electrostatics",
    ]
    assert parser.parse_args(oracle_args).backend == "e3nn"
    assert parser.parse_args([*oracle_args, "--backend", "cueq"]).backend == "cueq"
    assert benchmark_module.calculator_options("e3nn") == {"enable_cueq": False}
    assert benchmark_module.calculator_options("cueq") == {"enable_cueq": True}


def test_parity_metrics_cover_all_requested_properties(benchmark_module):
    reference = {
        "energy_eV": 4.0,
        "forces_eV_per_A": np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
        "stress_eV_per_A3": np.arange(6.0),
    }
    candidate = {
        "energy_eV": 4.2,
        "forces_eV_per_A": reference["forces_eV_per_A"] + 0.03,
        "stress_eV_per_A3": reference["stress_eV_per_A3"] - 0.04,
    }
    metrics = benchmark_module.parity_metrics(reference, candidate, atoms=2)
    assert metrics["energy_abs_error_eV"] == pytest.approx(0.2)
    assert metrics["energy_abs_error_eV_per_atom"] == pytest.approx(0.1)
    assert metrics["force_max_abs_error_eV_per_A"] == pytest.approx(0.03)
    assert metrics["stress_max_abs_error_eV_per_A3"] == pytest.approx(0.04)


def test_backend_status_proves_cueq_conversion(benchmark_module, monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "cueq_ops_runtime",
        lambda: {"available": True, "module": "/test/cueq.so", "error": None},
    )

    class CueqOperation(torch.nn.Module):
        pass

    CueqOperation.__module__ = "cuequivariance_torch.operations"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.operation = CueqOperation()
            self.cueq_config = SimpleNamespace(
                enabled=True,
                layout_str="ir_mul",
                optimize_all=True,
                conv_fusion=True,
            )

    calculator = SimpleNamespace(
        models=[Model()],
        _enable_cueq=True,
        _uses_accelerated_backend=True,
    )
    status = benchmark_module.accelerated_backend_status(calculator, "cueq")
    assert status["requested_backend_verified"] is True
    assert status["accelerated_backend_active"] is True
    assert status["cueq_module_count"] == 1
    assert status["enabled_cueq_config_count"] == 1
    assert status["enabled_cueq_configs"][0]["conv_fusion"] is True


def test_backend_status_rejects_silent_cueq_fallback(benchmark_module):
    calculator = SimpleNamespace(
        models=[torch.nn.Linear(2, 2)],
        _enable_cueq=False,
        _uses_accelerated_backend=False,
    )
    with pytest.raises(RuntimeError, match="did not activate"):
        benchmark_module.accelerated_backend_status(calculator, "cueq")


def test_backend_status_keeps_standard_e3nn_unaccelerated(benchmark_module):
    calculator = SimpleNamespace(
        models=[torch.nn.Linear(2, 2)],
        _enable_cueq=False,
        _uses_accelerated_backend=False,
    )
    status = benchmark_module.accelerated_backend_status(calculator, "e3nn")
    assert status["requested_backend_verified"] is True
    assert status["accelerated_backend_active"] is False


def test_graph_facts_count_real_evaluation_edges_and_padding(benchmark_module):
    calculator = SimpleNamespace(r_max=6.0)
    # Nodes 0..2 are the evaluated structure. Nodes 3..4 are a separate fake
    # padding graph and must not inflate the measured workload.
    batch = {
        "edge_index": torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
        )
    }
    facts = benchmark_module.graph_facts_from_evaluation_batch(
        calculator, batch, atoms=3
    )
    assert facts == {
        "construction_path": "MACECalculator._atoms_to_batch",
        "model_cutoff_A": 6.0,
        "neighbor_skin_A": 0.0,
        "effective_neighbor_cutoff_A": 6.0,
        "directed_edges_model_cutoff": 4,
        "evaluation_batch_directed_edges": 6,
        "padding_directed_edges": 2,
    }


def test_graph_facts_reject_cross_edges_into_padding(benchmark_module):
    calculator = SimpleNamespace(r_max=6.0)
    batch = {"edge_index": torch.tensor([[0, 3], [3, 0]], dtype=torch.long)}
    with pytest.raises(RuntimeError, match="crosses the real/padding boundary"):
        benchmark_module.graph_facts_from_evaluation_batch(calculator, batch, atoms=3)


def test_graph_facts_require_consistent_evaluations(benchmark_module):
    first = {
        "model_cutoff_A": 6.0,
        "directed_edges_model_cutoff": 4,
    }
    assert (
        benchmark_module.require_consistent_graph_facts([first, dict(first)]) == first
    )
    with pytest.raises(RuntimeError, match="changed across evaluations"):
        benchmark_module.require_consistent_graph_facts(
            [first, {**first, "directed_edges_model_cutoff": 6}]
        )
