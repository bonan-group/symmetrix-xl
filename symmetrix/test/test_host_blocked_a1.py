"""Host owner-computes A1 path: fresh-process parity, counters, and gating.

The blocked host A1 kernel is selected from inside the admitted-CBLAS branch,
so it can only run when ``host_worker_cblas_enabled()`` is true (for example a
runtime-compatible OpenMP OpenBLAS).  The policy statics are read once per
process, so every configuration here runs in a fresh subprocess and the test
asserts both the numerics and the launch counters that prove which path ran.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from model_downloads import MODEL_URLS, cached_model_path

_MODEL_NAME = "mace-mp-0b3-medium-1-8.json"
_COUNTERS = (
    "a1_host_blocked_forward_launch_count",
    "a1_host_blocked_reverse_launch_count",
    "a1_host_block_nodes_used",
    "a1_host_blocked_scratch_bytes",
)

_EVALUATION = textwrap.dedent(
    """
    import json
    import sys

    from ase.build import molecule
    from symmetrix import Symmetrix

    atoms = molecule("H2O")
    atoms.cell = [8.0, 8.0, 8.0]
    atoms.pbc = True
    atoms = atoms.repeat((2, 1, 2))
    atoms.cell = [14.0, 14.0, 14.0]
    calculator = Symmetrix(sys.argv[1], use_kokkos=True, dtype="float64")
    atoms.calc = calculator
    report = {
        "energy": float(atoms.get_potential_energy()),
        "forces": atoms.get_forces().tolist(),
        "stress": atoms.get_stress().tolist(),
    }
    evaluator = calculator.evaluator
    for name in (
        "a1_host_blocked_forward_launch_count",
        "a1_host_blocked_reverse_launch_count",
        "a1_host_block_nodes_used",
        "a1_host_blocked_scratch_bytes",
    ):
        report[name] = int(getattr(evaluator, name, -1))
    print(json.dumps(report))
    """
)


@pytest.fixture(scope="module")
def a1_model_path():
    try:
        return cached_model_path(_MODEL_NAME, MODEL_URLS[_MODEL_NAME])
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _run_evaluation(model_path, overrides):
    environment = dict(os.environ)
    environment.update(overrides)
    completed = subprocess.run(
        [sys.executable, "-c", _EVALUATION, str(model_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_host_blocked_a1_matches_per_node_and_records_launches(a1_model_path):
    blocked = _run_evaluation(a1_model_path, {"SYMMETRIX_A1_HOST_BLOCKED": "1"})
    per_node = _run_evaluation(a1_model_path, {"SYMMETRIX_A1_HOST_BLOCKED": "0"})

    assert blocked["a1_host_blocked_forward_launch_count"] >= 1
    assert blocked["a1_host_blocked_reverse_launch_count"] >= 1
    assert blocked["a1_host_block_nodes_used"] > 0
    assert blocked["a1_host_blocked_scratch_bytes"] > 0
    for name in _COUNTERS:
        assert per_node[name] == 0

    assert blocked["energy"] == pytest.approx(per_node["energy"], rel=1e-9, abs=1e-6)
    np.testing.assert_allclose(
        np.asarray(blocked["forces"]),
        np.asarray(per_node["forces"]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(blocked["stress"]),
        np.asarray(per_node["stress"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_host_blocked_a1_never_runs_when_worker_blas_is_refused(a1_model_path):
    # This is the selection a pthread OpenBLAS deployment with several workers
    # makes: the admitted-CBLAS branch is never entered, so the blocked kernel
    # reserves no scratch and no external call leaves a worker.
    refused = _run_evaluation(
        a1_model_path,
        {"SYMMETRIX_HOST_WORKER_BLAS": "off", "SYMMETRIX_A1_HOST_BLOCKED": "1"},
    )
    for name in _COUNTERS:
        assert refused[name] == 0
