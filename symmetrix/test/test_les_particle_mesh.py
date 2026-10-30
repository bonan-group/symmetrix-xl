from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import symmetrix
from ase import Atoms
from ase.io import write
from reference.mace_les import periodic_particle_mesh_oracle_arrays

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_PATH = (
    REPOSITORY_ROOT / "benchmarks" / "les_particle_mesh_qualification.py"
)
BENCHMARK_PATH = REPOSITORY_ROOT / "benchmarks" / "les_particle_mesh_benchmark.py"


def _load_qualification_module():
    spec = importlib.util.spec_from_file_location(
        "les_particle_mesh_qualification", QUALIFICATION_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load qualification module from {QUALIFICATION_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_module():
    specification = importlib.util.spec_from_file_location(
        "les_particle_mesh_benchmark_contract", BENCHMARK_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_particle_mesh_benchmark_compares_two_complete_maceles_algorithms():
    benchmark = _load_benchmark_module()
    assert benchmark.ALGORITHMS == ("explicit_sum", "particle_mesh")
    explicit = benchmark.example_record_for_testing(
        algorithm="explicit_sum", median_ns=1_600_000
    )
    particle_mesh = benchmark.example_record_for_testing(
        algorithm="particle_mesh", median_ns=1_200_000
    )

    summary = benchmark.compare_records([explicit, particle_mesh])

    assert summary["explicit_sum_us_per_atom"] == pytest.approx(16.0)
    assert summary["particle_mesh_us_per_atom"] == pytest.approx(12.0)
    assert summary["particle_mesh_speedup"] == pytest.approx(4.0 / 3.0)


def test_particle_mesh_benchmark_defaults_to_paper_bulk_water_conditions():
    benchmark = _load_benchmark_module()

    first = benchmark._structure((1, 1, 1))
    second = benchmark._structure((1, 1, 1))
    density = (
        first.get_masses().sum()
        * benchmark.AMU_PER_ANGSTROM3_TO_G_PER_CM3
        / first.get_volume()
    )

    assert len(first) == 192
    assert first.get_chemical_formula() == "H128O64"
    assert np.all(first.pbc)
    assert density == pytest.approx(benchmark.WATER_DENSITY_G_CM3)
    assert np.array_equal(first.numbers, second.numbers)
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.cell.array, second.cell.array)


def test_particle_mesh_benchmark_accepts_periodic_user_structure(tmp_path):
    benchmark = _load_benchmark_module()
    path = tmp_path / "methane.extxyz"
    source = Atoms(
        "CH4",
        positions=[
            [2.0, 2.0, 2.0],
            [2.6, 2.6, 2.6],
            [1.4, 1.4, 2.6],
            [1.4, 2.6, 1.4],
            [2.6, 1.4, 1.4],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    write(path, source)

    atoms = benchmark._structure((2, 1, 1), path)

    assert len(atoms) == 10
    assert atoms.get_chemical_formula() == "C2H8"
    assert np.allclose(atoms.cell.lengths(), [16.0, 8.0, 8.0])


def test_particle_mesh_benchmark_rejects_nonperiodic_user_structure(tmp_path):
    benchmark = _load_benchmark_module()
    path = tmp_path / "isolated.xyz"
    write(path, Atoms("H2O", positions=np.zeros((3, 3))))

    with pytest.raises(ValueError, match="three-dimensional PBC"):
        benchmark._structure((1, 1, 1), path)


def test_particle_mesh_benchmark_rejects_nonfinite_user_positions(tmp_path):
    benchmark = _load_benchmark_module()
    path = tmp_path / "nonfinite.extxyz"
    path.write_text(
        "1\n"
        'Lattice="8 0 0 0 8 0 0 0 8" pbc="T T T" '
        "Properties=species:S:1:pos:R:3\n"
        "H nan 0 0\n"
    )

    with pytest.raises(ValueError, match="positions must be finite"):
        benchmark._structure((1, 1, 1), path)


def test_particle_mesh_benchmark_cli_accepts_structure_and_replication():
    benchmark = _load_benchmark_module()

    defaults = benchmark._parser().parse_args(["model.json"])
    configured = benchmark._parser().parse_args(
        [
            "model.json",
            "--structure",
            "water.extxyz",
            "--replicate",
            "2",
            "3",
            "4",
        ]
    )

    assert defaults.structure is None
    assert configured.structure == Path("water.extxyz")
    assert configured.replicate == [2, 3, 4]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "MACE"),
        ("mace_les_parameters_sha256", None),
        ("mace_les_evaluator_fingerprint", "bad"),
    ],
)
def test_particle_mesh_benchmark_requires_complete_maceles_identity(field, value):
    benchmark = _load_benchmark_module()
    record = benchmark.example_record_for_testing()
    record[field] = value
    with pytest.raises(ValueError, match=field):
        benchmark.validate_record(record)


def _assert_outside_fd_envelope(actual, expected, *, rel, abs):
    assert actual != pytest.approx(expected, rel=rel, abs=abs)


def test_particle_mesh_qualification_rejects_inadequacy_as_interpolation_error():
    qualification = _load_qualification_module()
    record = qualification.example_record_for_testing()
    record["status"] = "inadequate_mesh"
    record["errors"] = {"energy": 0.0}
    record["samples_ns"] = []
    record["sample_count"] = 0
    record["median_us_per_atom"] = None

    with pytest.raises(ValueError, match="inadequate mesh.*error metrics"):
        qualification.validate_record(record)


def test_particle_mesh_qualification_record_requires_reproducible_identity():
    qualification = _load_qualification_module()
    record = qualification.example_record_for_testing()
    del record["source_provenance"]

    with pytest.raises(ValueError, match="source_provenance"):
        qualification.validate_record(record)


def test_particle_mesh_qualification_cli_requires_maceles_model():
    qualification = _load_qualification_module()
    with pytest.raises(SystemExit):
        qualification._parser().parse_args([])
    arguments = qualification._parser().parse_args(["model.json"])
    assert arguments.model == Path("model.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "MACE"),
        ("mace_les_parameters_sha256", None),
        ("mace_les_evaluator_fingerprint", "bad"),
    ],
)
def test_particle_mesh_qualification_requires_complete_maceles_identity(field, value):
    qualification = _load_qualification_module()
    record = qualification.example_record_for_testing()
    record[field] = value
    with pytest.raises(ValueError, match=field):
        qualification.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("case", None, "case"),
        ("algorithm", "base", "algorithm"),
        ("active_cutoff_margin", -0.1, "cutoff margin"),
        ("envelopes", {}, "envelopes"),
        ("median_us_per_atom", 99.0, "median_us_per_atom"),
    ],
)
def test_particle_mesh_qualification_rejects_contradictory_records(
    field, value, message
):
    qualification = _load_qualification_module()
    record = qualification.example_record_for_testing()
    if value is None:
        record.pop(field, None)
    else:
        record[field] = value

    with pytest.raises(ValueError, match=message):
        qualification.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sigma", 0.0, "sigma"),
        ("sigma", np.inf, "sigma"),
        ("dl", 0.0, "dl"),
        ("dl", np.nan, "dl"),
        ("directed_edges", -1, "directed_edges"),
        ("directed_edges", 1.5, "directed_edges"),
        ("cell", [[1.0, 0.0, 0.0]], "cell"),
        (
            "cell",
            [[np.inf, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "cell",
        ),
        ("translation_phases", [], "translation_phases"),
        ("translation_phases", [[0.0, 0.0]], "translation_phases"),
        ("translation_phases", [[0.0, np.inf, 0.0]], "translation_phases"),
        ("sample_count", 2, "sample_count"),
    ],
)
def test_particle_mesh_qualification_rejects_malformed_physical_record(
    field, value, message
):
    qualification = _load_qualification_module()
    record = qualification.example_record_for_testing()
    record["sample_count"] = len(record["samples_ns"])
    record[field] = value

    with pytest.raises(ValueError, match=message):
        qualification.validate_record(record)


@pytest.mark.parametrize("sample_count", [True, 0, -1, 1.5])
def test_particle_mesh_qualification_rejects_invalid_requested_sample_count(
    monkeypatch, sample_count
):
    qualification = _load_qualification_module()
    monkeypatch.setattr(
        symmetrix,
        "load_backend",
        lambda: pytest.fail("backend loaded before validating sample_count"),
    )

    with pytest.raises(ValueError, match="sample_count.*positive integer"):
        qualification.run_qualification(Path("unread.json"), sample_count=sample_count)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "metal", "backend"),
        ("kokkos_execution_space", "HIP", "kokkos_execution_space"),
        ("precision", "float32", "precision"),
        ("model_sha256", "A" * 64, "model_sha256"),
        ("native_extension_sha256", "g" * 64, "native_extension_sha256"),
    ],
)
def test_particle_mesh_qualification_rejects_invalid_schema_identity(
    field, value, message
):
    qualification = _load_qualification_module()
    record = qualification.example_record_for_testing()
    record[field] = value

    with pytest.raises(ValueError, match=message):
        qualification.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("source_commit", None, ValueError),
        ("source_commit", "", ValueError),
        ("source_dirty", None, ValueError),
        ("source_dirty", "false", TypeError),
        ("source_content_sha256", None, ValueError),
        ("source_content_sha256", "g" * 64, ValueError),
        ("native_source_content_sha256", None, ValueError),
        ("native_source_content_sha256", "A" * 64, ValueError),
    ],
)
def test_particle_mesh_qualification_rejects_invalid_source_provenance(
    field, value, error_type
):
    qualification = _load_qualification_module()
    record = qualification.example_record_for_testing()
    provenance = record["source_provenance"]
    if value is None:
        provenance.pop(field)
    else:
        provenance[field] = value

    with pytest.raises(error_type, match=field):
        qualification.validate_record(record)


@pytest.mark.parametrize(
    ("expected", "rel", "absolute", "inside_delta"),
    [
        (0.37, 6.0e-5, 6.0e-8, 2.0e-10),
        (0.004, 6.0e-5, 6.0e-9, 2.0e-12),
    ],
)
def test_disabled_adjoint_must_fall_outside_accepted_fd_envelope(
    expected, rel, absolute, inside_delta
):
    with pytest.raises(AssertionError):
        _assert_outside_fd_envelope(
            expected + inside_delta,
            expected,
            rel=rel,
            abs=absolute,
        )


def test_particle_mesh_qualification_propagates_non_adequacy_value_error():
    qualification = _load_qualification_module()
    with pytest.raises(ValueError, match="synthetic shape mismatch"):
        qualification._mesh_inadequacy_reason(ValueError("synthetic shape mismatch"))


def test_particle_mesh_qualification_labels_only_known_mesh_inadequacy():
    qualification = _load_qualification_module()
    message = "LES particle mesh omits an active reciprocal vector."
    assert qualification._mesh_inadequacy_reason(ValueError(message)) == message


def test_particle_mesh_qualification_validates_complete_endpoint_records():
    qualification = _load_qualification_module()
    records = []
    for case in ("orthogonal", "triclinic"):
        record = qualification.example_record_for_testing()
        record["case"] = case
        records.append(record)
    qualification.validate_endpoint(records)


@pytest.fixture(scope="module")
def native():
    module = symmetrix.load_backend()
    if not module._kokkos_is_initialized():
        module._init_kokkos()
    return module


@pytest.fixture(scope="module")
def particle_mesh_native(native):
    capability = native._particle_mesh_capability()
    if not capability["available"]:
        pytest.skip(
            "this build has no particle-mesh FFT provider "
            f"({capability['reason'] or 'unreported'})"
        )
    return native


_CASES = (
    pytest.param(
        np.zeros(3),
        np.array([[0.2, 0.3, 0.4], [1.1, 0.7, 0.5], [2.0, 1.4, 0.9]]),
        np.diag([5.1, 5.4, 5.8]),
        id="zero-charge-orthogonal",
    ),
    pytest.param(
        np.array([0.73]),
        np.array([[1.13, 0.77, 2.21]]),
        np.diag([5.1, 5.4, 5.8]),
        id="one-atom-orthogonal",
    ),
    pytest.param(
        np.array([0.41, -0.73, 0.28]),
        np.array([[0.2, 0.3, 0.5], [1.3, 0.8, 0.4], [0.7, 1.6, 1.1]]),
        np.array([[5.71, 0.17, 0.09], [0.11, 5.33, 0.21], [0.07, 0.19, 5.49]]),
        id="non-neutral-triclinic",
    ),
)


@pytest.mark.parametrize("remove_self_interaction", [False, True])
@pytest.mark.parametrize(("charges", "positions", "cell"), _CASES)
@pytest.mark.parametrize(
    "class_name", ["_LESMonopoleKokkos", "_LESMonopoleKokkosFloat"]
)
def test_periodic_particle_mesh_matches_same_mesh_oracle(
    particle_mesh_native,
    charges,
    positions,
    cell,
    remove_self_interaction,
    class_name,
):
    sigma = 0.8
    dl = 1.35
    mesh = (24, 24, 24)
    order = 4
    operator = getattr(particle_mesh_native, class_name)(
        sigma, dl, remove_self_interaction
    )
    operator.configure_particle_mesh_for_testing(*mesh, order)
    result = operator.compute_periodic(charges, positions, cell)
    expected = periodic_particle_mesh_oracle_arrays(
        positions=positions,
        cell=cell,
        charges=charges,
        sigma=sigma,
        dl=dl,
        remove_self_interaction=remove_self_interaction,
        mesh=mesh,
        spline_order=order,
    )

    assert operator.diagnostics["periodic_algorithm"] == "particle_mesh"
    for name in (
        "energy",
        "atomic_energies",
        "forces",
        "charge_adjoints",
        "strain_derivative",
    ):
        np.testing.assert_allclose(result[name], expected[name], rtol=3e-11, atol=3e-10)
    assert result["energy"] == pytest.approx(result["atomic_energies"].sum())


def test_explicit_periodic_remains_the_default_and_can_be_restored(
    particle_mesh_native,
):
    operator = particle_mesh_native._LESMonopoleKokkos(0.8, 1.35, True)
    assert operator.diagnostics["periodic_algorithm"] == "explicit_sum"
    operator.configure_particle_mesh_for_testing(24, 24, 24, 4)
    operator.use_explicit_periodic_for_testing()
    assert operator.diagnostics["periodic_algorithm"] == "explicit_sum"


@pytest.mark.parametrize(
    ("mesh", "match"),
    [
        ((9, 7, 5), "omits"),
        ((5, 5, 5), "aliases"),
        ((8, 24, 24), "Nyquist"),
    ],
)
def test_particle_mesh_rejects_inadequate_active_miller_sphere(
    particle_mesh_native, mesh, match
):
    operator = particle_mesh_native._LESMonopoleKokkos(0.8, 1.35, True)
    operator.configure_particle_mesh_for_testing(*mesh, 4)
    with pytest.raises(ValueError, match=match):
        operator.compute_periodic(
            np.array([0.4, -0.1]),
            np.array([[0.2, 0.3, 0.5], [1.3, 0.8, 0.4]]),
            np.diag([5.7, 5.3, 5.5]),
        )


def test_triclinic_complete_active_sphere_accepts_odd_mesh(particle_mesh_native):
    operator = particle_mesh_native._LESMonopoleKokkos(0.8, 1.35, True)
    operator.configure_particle_mesh_for_testing(25, 25, 25, 4)
    result = operator.compute_periodic(
        np.array([0.4, -0.1]),
        np.array([[0.2, 0.3, 0.5], [1.3, 0.8, 0.4]]),
        np.array([[5.7, 0.2, 0.1], [0.1, 5.3, 0.2], [0.0, 0.1, 5.5]]),
    )
    assert np.isfinite(result["energy"])


def test_particle_mesh_device_path_accepts_retained_position_capacity(
    particle_mesh_native,
):
    sigma = 0.8
    dl = 1.35
    cell = np.diag([5.7, 5.3, 5.5])
    charges = np.array([0.41, -0.73, 0.28, -0.16, 0.09])
    positions = np.array(
        [
            [0.2, 0.3, 0.5],
            [1.3, 0.8, 0.4],
            [0.7, 1.6, 1.1],
            [2.1, 1.2, 0.6],
            [0.4, 2.2, 1.7],
        ]
    )
    retained = particle_mesh_native._LESMonopoleKokkos(sigma, dl, True)
    retained.configure_particle_mesh_for_testing(24, 24, 24, 4)
    retained._compute_periodic_device_prefix_for_testing(
        charges, positions, cell, len(charges)
    )
    result = retained._compute_periodic_device_prefix_for_testing(
        charges, positions, cell, 2
    )
    assert retained.diagnostics["particle_count_fences"] == 1

    fresh = particle_mesh_native._LESMonopoleKokkos(sigma, dl, True)
    fresh.configure_particle_mesh_for_testing(24, 24, 24, 4)
    expected = fresh.compute_periodic(charges[:2], positions[:2], cell)
    for name in (
        "energy",
        "atomic_energies",
        "forces",
        "charge_adjoints",
        "strain_derivative",
    ):
        np.testing.assert_allclose(result[name], expected[name], rtol=0.0, atol=0.0)


def test_particle_mesh_rejects_active_sphere_outside_integer_range(
    particle_mesh_native,
):
    operator = particle_mesh_native._LESMonopoleKokkos(0.8, 1.35, True)
    operator.configure_particle_mesh_for_testing(24, 24, 24, 4)
    with pytest.raises(ValueError, match="supported integer range"):
        operator.compute_periodic(
            np.array([0.4]),
            np.array([[0.2, 0.3, 0.5]]),
            np.diag([1.0e20, 5.3, 5.5]),
        )


def test_particle_mesh_rejects_infeasible_combined_search_extent_promptly(
    particle_mesh_native,
):
    operator = particle_mesh_native._LESMonopoleKokkos(0.8, 1.35, True)
    operator.configure_particle_mesh_for_testing(24, 24, 24, 4)
    with pytest.raises(ValueError, match="enumeration limit"):
        operator.compute_periodic(
            np.array([0.4]),
            np.array([[0.2, 0.3, 0.5]]),
            np.diag([318.0, 318.0, 318.0]),
        )


@pytest.mark.parametrize("remove_self_interaction", [False, True])
def test_particle_mesh_derivatives_match_finite_differences(
    particle_mesh_native, remove_self_interaction
):
    sigma = 0.8
    dl = 1.35
    charges = np.array([0.41, -0.73, 0.28])
    positions = np.array([[0.2, 0.3, 0.5], [1.3, 0.8, 0.4], [0.7, 1.6, 1.1]])
    cell = np.array([[5.71, 0.17, 0.09], [0.11, 5.33, 0.21], [0.07, 0.19, 5.49]])
    operator = particle_mesh_native._LESMonopoleKokkos(
        sigma, dl, remove_self_interaction
    )
    operator.configure_particle_mesh_for_testing(24, 24, 24, 4)
    result = operator.compute_periodic(charges, positions, cell)
    step = 2.0e-6

    charge_plus = charges.copy()
    charge_minus = charges.copy()
    charge_plus[1] += step
    charge_minus[1] -= step
    charge_derivative = (
        operator.compute_periodic(charge_plus, positions, cell)["energy"]
        - operator.compute_periodic(charge_minus, positions, cell)["energy"]
    ) / (2.0 * step)
    assert result["charge_adjoints"][1] == pytest.approx(
        charge_derivative, rel=2e-7, abs=2e-8
    )

    position_plus = positions.copy()
    position_minus = positions.copy()
    position_plus[1, 2] += step
    position_minus[1, 2] -= step
    force_derivative = -(
        operator.compute_periodic(charges, position_plus, cell)["energy"]
        - operator.compute_periodic(charges, position_minus, cell)["energy"]
    ) / (2.0 * step)
    assert result["forces"][1, 2] == pytest.approx(force_derivative, rel=3e-6, abs=3e-7)

    deformation = np.eye(3)
    deformation[0, 1] += step
    strain_plus = operator.compute_periodic(
        charges, positions @ deformation, cell @ deformation
    )["energy"]
    deformation[0, 1] -= 2.0 * step
    strain_minus = operator.compute_periodic(
        charges, positions @ deformation, cell @ deformation
    )["energy"]
    strain_derivative = (strain_plus - strain_minus) / (2.0 * step)
    assert result["strain_derivative"][0, 1] == pytest.approx(
        strain_derivative, rel=3e-6, abs=3e-7
    )


def test_provider_off_keeps_explicit_les_and_reports_the_capability_reason(native):
    capability = native._particle_mesh_capability()
    if capability["available"]:
        pytest.skip("requires a build without a particle-mesh FFT provider")
    operator = native._LESMonopoleKokkos(0.8, 1.35, True)
    charges = np.array([0.4, -0.1])
    positions = np.array([[0.2, 0.3, 0.5], [1.3, 0.8, 0.4]])
    cell = np.diag([5.7, 5.3, 5.5])
    assert np.isfinite(operator.compute_nonperiodic(charges, positions)["energy"])
    assert np.isfinite(operator.compute_periodic(charges, positions, cell)["energy"])
    with pytest.raises(RuntimeError, match=capability["reason"]):
        operator.configure_particle_mesh_for_testing(24, 24, 24, 4)


def test_particle_mesh_preserves_non_default_gpu_stream_ordering(
    particle_mesh_native,
):
    probe = (
        particle_mesh_native.MACELESKokkos._les_particle_mesh_stream_probe_for_testing()
    )
    if not probe["available"]:
        pytest.skip(probe["reason"])
    assert probe["passed"], probe["reason"]
    assert probe["execution_instance_id"] != probe["default_instance_id"]
    assert probe["particle_count_fences"] == 2
