"""The native periodic neighbor graph under the stateful execution profiles.

The Kokkos periodic builder prepares only the direct-profile source schedule.
The factorized observer and execution parameter gradients select the stateful
profile, whose reverse reads the stateful schedule. ``SYMMETRIX_NEIGHBOR_BACKEND
=kokkos`` reaches the builder on a host build, as CUDA and HIP do at 512 atoms
or more under ``automatic``.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from compact_r1_model import foundation_r1_test_model
from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix

PROPERTIES = ["energy", "forces", "stress"]


def _require_host_execution_space():
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    if native_symmetrix._kokkos_default_execution_space() not in ("OpenMP", "Serial"):
        pytest.skip("the host JIT plugin requires Kokkos OpenMP or Serial")


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    contracts = Path(__file__).resolve().parent / "data" / "execution_contracts"
    m0_contract = json.loads((contracts / "standard_m0_contract.json").read_text())
    m0_contract["output_l_max"] = 0
    m0_contract["output_components"] = 1
    m0_contract["monomial_groups"] = m0_contract["monomial_groups"][:1]
    m0_contract["structure_fingerprint"] = (
        "sha256:3b4ab2d979488798a05c07b5d54c4c96c1026da3ee1eb93069fd2bacec76ac90"
    )
    m0_contract["semantic_fingerprint"] = (
        "sha256:9fe598ceffb71e61d162c0afbdbe5727a0da7997dbca3a531a46b63603d7af85"
    )
    m0_contract["generation_fingerprint"] = (
        "sha256:f974e4af32dfb328e06fb6655b965c4cc5c0cad50a2013223630f9960f59d398"
    )
    r0_contract = json.loads((contracts / "standard_r0_contract.json").read_text())
    r1_contract = json.loads(
        (contracts / "jit_r1_mace_omat0_small_contract.json").read_text()
    )
    model = foundation_r1_test_model(
        r1_contract,
        atomic_numbers=(7, 13),
        m0_contract=m0_contract,
        r0_contract=r0_contract,
    )
    model["A0_scaled"] = True
    model["compact_radial"]["networks"]["A0"] = {
        "shape": [r1_contract["radial_embedding"], 1],
        "weights": [[0.01] * r1_contract["radial_embedding"]],
        "activation": "silu",
        "activation_scale": 1.0,
        "postprocess": "tanh-square",
    }
    output = tmp_path_factory.mktemp("device-graph-profiles") / "model.json"
    output.write_text(json.dumps(model, separators=(",", ":")))
    return output


def _periodic_frames():
    scaled = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    atoms = Atoms(
        ["Al"] * 4 + ["N"] * 4,
        scaled_positions=scaled,
        cell=[4.4, 4.5, 4.6],
        pbc=True,
    )
    index = np.arange(len(atoms), dtype=float)[:, None]
    atoms.positions += 0.05 * np.sin(1.7 * index + np.arange(3.0))
    moved = atoms.copy()
    moved.positions += 0.01 * np.cos(2.3 * index + np.arange(3.0))
    return atoms, moved


def _calculator(path):
    # The speed profile keeps the Cartesian edge geometry and full MH-0 state
    # that the observer and parameter gradients need.
    return Symmetrix(
        path,
        dtype="float32",
        execution_profile="speed",
        streamed_edges="direct",
        neighbor_skin=0.3,
    )


def _forces(calculator, frames):
    forces = []
    for frame in frames:
        calculator.calculate(frame, properties=PROPERTIES)
        forces.append(np.array(calculator.results["forces"], dtype=float))
    return forces


@pytest.mark.host
@pytest.mark.jit
def test_stateful_profiles_build_host_graphs_under_the_kokkos_builder(
    model_path, monkeypatch
):
    _require_host_execution_space()
    frames = _periodic_frames()

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    reference = _calculator(model_path)
    reference.evaluator._set_factorized_observer(True)
    expected = _forces(reference, frames)
    assert reference.evaluator.factorized_execution_profile == "factorized_state"
    del reference
    gc.collect()

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    direct = _calculator(model_path)
    assert direct.evaluator.factorized_execution_profile == "direct_fixed_weight"
    direct.calculate(frames[0], properties=PROPERTIES)
    assert direct.neighbor_graph_backend == "kokkos"
    del direct
    gc.collect()

    # The observer is on before the first graph: the host builder prepares it.
    observed = _calculator(model_path)
    observed.evaluator._set_factorized_observer(True)
    for forces, reference_forces in zip(_forces(observed, frames), expected):
        assert observed.neighbor_graph_backend == "host"
        np.testing.assert_allclose(forces, reference_forces, rtol=1e-6, atol=1e-6)
    assert observed.evaluator._factorized_observer()["ready"]
    del observed
    gc.collect()

    # Parameter gradients after one direct evaluation invalidate the device
    # graph; the next evaluation rebuilds the topology instead of reusing it.
    late = _calculator(model_path)
    late.calculate(frames[0], properties=PROPERTIES)
    assert late.neighbor_graph_backend == "kokkos"
    late.evaluator.set_execution_parameter_gradients(True)
    late.calculate(frames[1], properties=PROPERTIES)
    assert late.neighbor_graph_backend == "host"
    assert late.evaluator.execution_parameter_gradients()["ready"]
    del late
    gc.collect()
