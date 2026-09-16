import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "radial_spline_convergence.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("radial_spline_convergence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compact_radial(transform):
    return {
        "basis": {
            "type": "bessel",
            "weights": [0.8, 1.7],
            "prefactor": 0.63,
        },
        "cutoff": {"type": "polynomial", "r_max": 5.0, "p": 6},
        "distance_transform": transform,
        "networks": {
            "R0": {
                "shape": [2, 3, 1],
                "weights": [
                    [0.4, -0.2, 0.7, 0.1, -0.3, 0.6],
                    [0.5, -0.8, 0.25],
                ],
                "activation": "silu",
                "activation_scale": 1.3,
            }
        },
    }


@pytest.mark.parametrize(
    "transform",
    [
        {"type": "none"},
        {
            "type": "agnesi",
            "a": 1.2,
            "q": 2.0,
            "p": 1.0,
            "covalent_radii": [0.71, 1.21],
        },
    ],
)
def test_exact_radial_derivative_matches_finite_difference(transform):
    driver = _load_driver()
    compact = _compact_radial(transform)
    radius = np.asarray([0.55, 0.9, 1.87, 3.2, 4.7])
    values, derivatives = driver.exact_network(compact, "R0", 0, 1, radius)
    step = 1.0e-6
    plus, _ = driver.exact_network(compact, "R0", 0, 1, radius + step)
    minus, _ = driver.exact_network(compact, "R0", 0, 1, radius - step)

    np.testing.assert_allclose(derivatives, (plus - minus) / (2.0 * step), atol=2e-9)
    assert np.all(np.isfinite(values))


def test_bad_radius_report_separates_failure_regions():
    driver = _load_driver()
    radius = np.arange(8, dtype=np.float64)
    error = np.asarray([0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0])[:, None]

    report = driver._bad_radius_regions(radius, error, threshold=1.0)

    assert report == {
        "minimum_A": 1.0,
        "maximum_A": 7.0,
        "sample_count": 4,
        "region_count": 3,
        "largest_regions": [
            {"minimum_A": 1.0, "maximum_A": 2.0, "sample_count": 2},
            {"minimum_A": 5.0, "maximum_A": 5.0, "sample_count": 1},
            {"minimum_A": 7.0, "maximum_A": 7.0, "sample_count": 1},
        ],
    }


def test_end_to_end_structure_defaults_to_generated_aln():
    driver = _load_driver()

    atoms, provenance = driver._load_end_to_end_structure(None)

    assert len(atoms) == 32
    assert atoms.get_chemical_formula() == "Al16N16"
    assert provenance == {
        "kind": "generated",
        "description": "deterministically perturbed 2x2x2 wurtzite AlN",
        "lattice_a_A": 3.112,
        "lattice_c_A": 4.982,
        "maximum_perturbation_A": 0.008,
        "formula": "Al16N16",
        "atoms": 32,
        "periodic": [True, True, True],
    }


def test_end_to_end_structure_loads_ase_file(tmp_path):
    from ase import Atoms
    from ase.io import write

    driver = _load_driver()
    structure = tmp_path / "structure.extxyz"
    write(structure, Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], pbc=False))

    atoms, provenance = driver._load_end_to_end_structure(structure)

    assert atoms.get_chemical_formula() == "Si2"
    assert provenance["kind"] == "file"
    assert provenance["path"] == str(structure.resolve())
    assert provenance["sha256"] == driver._sha256(structure)
    assert provenance["atoms"] == 2
    assert provenance["periodic"] == [False, False, False]


def test_end_to_end_structure_rejects_empty_file(tmp_path):
    from ase import Atoms
    from ase.io import write

    driver = _load_driver()
    structure = tmp_path / "empty.extxyz"
    write(structure, Atoms())

    with pytest.raises(ValueError, match="at least one atom"):
        driver._load_end_to_end_structure(structure)


def test_markdown_identifies_configured_structure():
    driver = _load_driver()
    report = {
        "radius_grid": {"cutoff_A": 5.0, "physical_minimum_A": 1.5},
        "recommended_minimum_spline_points": 256,
        "sweeps": [
            {
                "spline_points": 256,
                "grid_spacing_A": 0.02,
                "physical_max_value_error": 1.0e-5,
                "physical_max_derivative_error": 1.0e-3,
                "passed_all_gates": True,
            }
        ],
        "end_to_end": {
            "structure": {"formula": "Si2", "atoms": 2},
            "rows": [
                {
                    "spline_points": 256,
                    "energy_error_eV": 2.0e-6,
                    "force_max_abs_error_eV_per_A": 3.0e-6,
                }
            ],
        },
    }

    markdown = driver._markdown(report)

    assert "## End-to-end PyTorch parity" in markdown
    assert "Structure: `Si2` (2 atoms)." in markdown
    assert "Physical AlN PyTorch parity" not in markdown
