from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/mh1_standard_mace_comparison.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "mh1_standard_mace_comparison", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_structure_is_the_profiled_5000_atom_state():
    module = _module()
    atoms = module._make_atoms()

    assert len(atoms) == 5000
    assert atoms.pbc.all()
    assert np.allclose(atoms.cell.lengths(), (39.05, 39.05, 39.05))
    assert module._state_sha256(atoms) == (
        "85c737af3fd276a79e255d97b1b15e5a099a5db140875f0714e38ab1abd8675f"
    )


def test_comparison_structure_repeat_controls_the_cubic_size():
    module = _module()
    atoms = module._make_atoms(2)

    assert len(atoms) == 40
    assert np.allclose(atoms.cell.lengths(), (7.81, 7.81, 7.81))


def test_comparison_forwards_the_standard_mace_debug_plan():
    module = _module()

    kwargs = module._calculator_kwargs(
        Namespace(
            profile="capacity",
            debug_execution_plan="mh0-direct-capacity-y-only",
            mh1_node_state_policy=None,
            mh1_node_arena_policy=None,
            mh1_node_arena_tile_rows=None,
        )
    )

    assert kwargs == {
        "dtype": "float32",
        "streamed_edges": "direct",
        "execution_profile": "capacity",
        "neighbor_skin": 0.5,
        "_debug_execution_plan": "mh0-direct-capacity-y-only",
    }


def test_comparison_forwards_independent_mh1_state_and_arena_policies():
    module = _module()

    kwargs = module._calculator_kwargs(
        Namespace(
            profile="speed",
            debug_execution_plan=None,
            mh1_node_state_policy="retain-interaction-v1",
            mh1_node_arena_policy="throughput-v1",
            mh1_node_arena_tile_rows=None,
        )
    )

    assert kwargs == {
        "dtype": "float32",
        "streamed_edges": "direct",
        "execution_profile": "speed",
        "neighbor_skin": 0.5,
        "_debug_execution_plan": None,
        "execution_mh1_node_state_policy": "retain-interaction-v1",
        "execution_mh1_node_arena_policy": "throughput-v1",
    }


def test_comparison_applies_the_mh1_tile_row_override():
    module = _module()
    calls = []
    calculator = SimpleNamespace(
        evaluator=SimpleNamespace(
            _set_execution_mh1_node_arena_tile_rows_for_testing=calls.append
        )
    )

    module._apply_runtime_overrides(calculator, Namespace(mh1_node_arena_tile_rows=512))

    assert calls == [512]


def test_comparison_configures_the_dense_provider(monkeypatch):
    module = _module()

    module._configure_environment(Namespace(mh1_dense_policy="vendor_fp32"))

    assert module.os.environ["SYMMETRIX_MH1_DENSE_POLICY"] == "vendor_fp32"
