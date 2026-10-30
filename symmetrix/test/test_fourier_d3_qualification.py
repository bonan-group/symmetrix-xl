"""Release-facing correctness matrix for the reciprocal FourierD3 evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import symmetrix
from symmetrix.corrections import CorrectionPipeline, FourierD3Correction


FIXTURE_DIRECTORY = Path(__file__).parent / "reference" / "fixtures"
MATRIX_PATH = FIXTURE_DIRECTORY / "fourier_d3_qualification_matrix.json"
REFERENCE_PATH = FIXTURE_DIRECTORY / "fourier_d3_reference_cases.json"


def _matrix() -> dict[str, object]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _reference() -> dict[str, object]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def _cases() -> list[object]:
    if not MATRIX_PATH.is_file():
        return []
    return _matrix()["oracle_acceptance"]["cases"]


@pytest.fixture(scope="module")
def native():
    module = symmetrix.load_backend()
    if not module._kokkos_is_initialized():
        module._init_kokkos()
    capability = module._fourier_d3_capability()
    if not capability["available"]:
        pytest.skip(
            "the selected backend has no FourierD3 FFT provider: "
            f"{capability['reason'] or 'unreported'}"
        )
    return module


def test_qualification_matrix_declares_every_release_axis():
    """Dropping a required release axis must make the matrix incomplete."""
    matrix = _matrix()

    assert matrix["schema"] == "symmetrix.fourier-d3-correctness-matrix"
    assert matrix["version"] == 1
    assert set(matrix["axes"]["precisions"]) == {"float32", "float64"}
    assert set(matrix["axes"]["cell_shapes"]) == {
        "orthogonal",
        "strongly_triclinic",
    }
    assert set(matrix["axes"]["spline_orders"]) == {2, 3, 4, 5, 6}
    assert set(matrix["axes"]["mesh_policies"]) == {
        "explicit",
        "automatic_spacing",
    }
    assert {1, 2, 3, 5, 512} <= set(matrix["axes"]["rank_chunks"])
    assert set(matrix["axes"]["updates"]) == {
        "atom_reorder",
        "cell_change",
        "species_change",
    }
    assert set(matrix["axes"]["reuse_domains"]) == {
        "graph",
        "decomposition",
        "reciprocal_kernel",
        "mesh",
        "fft_plan",
    }
    assert set(matrix["backends"]) == {"cpu", "cuda", "hip"}


def test_selected_backend_uses_its_declared_fft_provider(native):
    """A backend may qualify only through its own FFT provider, never a fallback."""
    matrix = _matrix()
    info = native._backend_build_info()
    backend = info["backend"]
    declared = matrix["backends"][backend]
    capability = native._fourier_d3_capability()

    assert capability["provider"] == declared["provider"]
    assert capability["execution_backend"] == backend
    assert capability["kokkosfft_version"] == "2.0.0"


@pytest.mark.parametrize(
    "entry",
    _cases(),
    ids=lambda entry: (
        f"{entry['precision']}-{entry['cell_shape']}-{entry['mesh'][0]}"
    ),
)
def test_release_mesh_meets_the_frozen_oracle(native, entry):
    """The selected release mesh must meet every original v1 accuracy target."""
    reference = _reference()
    parameters = reference["parameters"]
    case = next(case for case in reference["cases"] if case["name"] == entry["case"])
    dtype = np.float32 if entry["precision"] == "float32" else np.float64
    configuration = FourierD3Correction(
        functional=parameters["functional"],
        coordination_cutoff=parameters["coordination_cutoff_angstrom"],
        skin=0.0,
        mesh=tuple(entry["mesh"]),
        spline_order=entry["spline_order"],
        chunk_size=entry["rank_chunk"],
    )
    atoms = Atoms(
        numbers=np.asarray(case["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(case["positions_angstrom"], dtype=np.float64),
        cell=np.asarray(case["cell_angstrom"], dtype=np.float64),
        pbc=True,
    )
    pipeline = CorrectionPipeline([configuration], native=native, dtype=dtype)
    result = pipeline.calculate(atoms, ["energy", "energies", "forces", "stress"])

    expected = case["expected"]
    strain = np.asarray(expected["strain_derivative_ev"], dtype=np.float64)
    volume = abs(float(np.linalg.det(np.asarray(case["cell_angstrom"]))))
    expected_stress = (
        np.asarray(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                strain[1, 2],
                strain[0, 2],
                strain[0, 1],
            ]
        )
        / volume
    )
    thresholds = _matrix()["thresholds"][entry["precision"]]
    atom_count = len(case["atomic_numbers"])

    assert (
        abs(result["energy"] - expected["energy_ev"]) / atom_count
        <= thresholds["energy_ev_per_atom"]
    )
    assert (
        np.max(
            np.abs(
                result["forces"]
                - np.asarray(expected["forces_ev_per_angstrom"], dtype=np.float64)
            )
        )
        <= thresholds["force_ev_per_angstrom"]
    )
    assert (
        np.max(np.abs(result["stress"] - expected_stress))
        <= thresholds["stress_ev_per_angstrom3"]
    )
    assert (
        abs(float(np.sum(result["energies"])) - result["energy"])
        <= thresholds["atomic_sum_ev"]
    )
    assert pipeline.diagnostics()[0]["backend_fallback_count"] == 0
