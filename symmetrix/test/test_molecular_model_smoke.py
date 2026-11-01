"""Small water-system smoke checks for the released molecular model families.

The model paths are supplied by the environment because checkpoints are not
repository fixtures.  Missing models skip cleanly; a supplied model that
cannot evaluate the water cases is a real test failure.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.polar_paper_systems import build_system


MODEL_CASES = (
    pytest.param(
        "mace_off",
        "SYMMETRIX_MACE_OFF_MODEL",
        ("energy", "forces", "stress"),
        (),
        id="mace-off",
    ),
    pytest.param(
        "mace_off_les",
        "SYMMETRIX_MACELES_OFF_MODEL",
        ("energy", "forces", "stress"),
        ("les_energy", "latent_charges"),
        id="mace-off-les",
    ),
    pytest.param(
        "mace_polar",
        "SYMMETRIX_MACE_POLAR_MODEL",
        ("energy", "forces", "stress", "dipole"),
        ("dipole",),
        id="mace-polar",
    ),
)


@pytest.mark.parametrize("system_name", ("molecule_h2o", "redox_water64"))
@pytest.mark.parametrize(
    "family, environment, properties, required_results", MODEL_CASES
)
def test_water_model_family_smoke(
    system_name, family, environment, properties, required_results
):
    model_value = os.environ.get(environment)
    if not model_value:
        pytest.skip(f"set {environment} to run the {family} water smoke test")
    model = Path(model_value).expanduser()
    if not model.is_file():
        pytest.fail(f"{environment} points to a missing model: {model}")

    from symmetrix import Symmetrix

    atoms, _ = build_system(system_name)
    calculator = Symmetrix(model, dtype="float32", streamed_edges="reference")
    atoms.calc = calculator
    try:
        calculator.calculate(atoms, properties=list(properties))

        for property_name in properties:
            assert property_name in calculator.results, (
                f"{family} omitted requested {property_name} on {system_name}"
            )
            assert np.all(np.isfinite(np.asarray(calculator.results[property_name]))), (
                f"{family} returned non-finite {property_name} on {system_name}"
            )
        for result_name in required_results:
            assert result_name in calculator.results, (
                f"{family} omitted family result {result_name} on {system_name}"
            )
            assert np.all(np.isfinite(np.asarray(calculator.results[result_name]))), (
                f"{family} returned non-finite {result_name} on {system_name}"
            )
    finally:
        atoms.calc = None
        del calculator
        gc.collect()
