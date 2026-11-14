"""Prepare generated MACE Polar artifacts for the linked LAMMPS tests.

The ASE calculator prepares the backbone and one product module per layer and
evaluates the reference values. It runs in a child Python process, so the test
process holds no Kokkos runtime beside the LAMMPS executable it launches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SYMMETRIX_TESTS = Path(__file__).resolve().parents[2] / "symmetrix" / "test"

_PREPARE = r"""
import json
import sys

tests, model, dtype, layers, species, positions = sys.argv[1:7]
layers = int(layers)
try:
    from ase import Atoms
    from symmetrix.calculator import Symmetrix
    from symmetrix.mace_nonlinear_jit_artifact import polar_lammps_jit_arguments

    if layers:
        sys.path.insert(0, tests)
        # mace sets the weights-only switch that importing e3nn needs.
        from test_mace_polar_backbone import (
            _build_polar_shaped_model,
            polar_import_error,
        )

        if polar_import_error is not None:
            raise polar_import_error
        import torch
        from e3nn import o3
        from mace.modules.blocks import LinearReadoutBlock
        from symmetrix.extract_mace_polar import extract_mace_polar_data
except ImportError as error:
    print(f"POLAR_JIT_UNAVAILABLE {error}")
    raise SystemExit(0)

if layers:
    network = _build_polar_shaped_model(
        num_interactions=layers,
        hidden_irreps="4x0e" if layers == 1 else "4x0e+4x1o",
    )
    if layers == 1:
        # The released S model ends in a linear readout over a scalar basis,
        # which makes its only product an M1 module.
        network.readouts[0] = LinearReadoutBlock(o3.Irreps("4x0e")).to(torch.float64)
    with open(model, "w") as stream:
        json.dump(extract_mace_polar_data(network), stream)

calculator = Symmetrix(
    model,
    dtype=dtype,
    streamed_edges="direct",
    polar_electrostatics="direct_reciprocal",
)
atoms = Atoms(
    "".join(json.loads(species)),
    positions=json.loads(positions),
    cell=(12.0, 12.0, 12.0),
    pbc=True,
)
atoms.calc = calculator
prepared = {
    "arguments": polar_lammps_jit_arguments(calculator),
    "energy": atoms.get_potential_energy(),
    "forces": atoms.get_forces().tolist(),
    "stress": atoms.get_stress().tolist(),
}
print("POLAR_JIT_PREPARED " + json.dumps(prepared))
"""


def skip_or_fail(message):
    """Skip, or fail where the caller declared the LAMMPS tests required."""

    if os.environ.get("SYMMETRIX_REQUIRE_LAMMPS_TESTS"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def prepare_polar_jit(model, dtype, cache, *, species, positions, layers=0):
    """Return a model's ``pair_style`` artifact settings and ASE reference.

    ``layers`` above zero first builds the synthetic Polar model with that
    many interactions and writes it to ``model``. The result holds the
    settings (``arguments``) and the ``direct`` route's ``energy``, ``forces``
    and ``stress`` for the atoms in a periodic 12 A box.
    """

    environment = os.environ.copy()
    environment["SYMMETRIX_JIT_CACHE"] = str(cache)
    environment["SYMMETRIX_JIT_POLICY"] = "required"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PREPARE,
            str(SYMMETRIX_TESTS),
            str(model),
            dtype,
            str(layers),
            json.dumps(list(species)),
            json.dumps(positions),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=1800,
    )
    lines = completed.stdout.splitlines()
    for line in lines:
        if line.startswith("POLAR_JIT_UNAVAILABLE "):
            skip_or_fail(
                "the Polar JIT preparation is unavailable: "
                + line.removeprefix("POLAR_JIT_UNAVAILABLE ")
            )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    marker = next(line for line in lines if line.startswith("POLAR_JIT_PREPARED "))
    return json.loads(marker.removeprefix("POLAR_JIT_PREPARED "))
