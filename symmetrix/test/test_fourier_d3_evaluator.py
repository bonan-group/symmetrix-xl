"""End-to-end coverage for the dtype-specific FourierD3 evaluator bindings (Task 11).

The module drives the private ``FourierD3Kokkos`` (float64) and ``FourierD3KokkosFloat``
(float32) classes that the Python correction runtime resolves, and the runtime itself,
from four independent directions:

* the canonical tables, their per-array content digests, and their provenance fingerprint
  come from the installed native accessor and are compared against the repository's
  generated C++ tables and provenance header, parsed here and never at installed runtime;
* the evaluator protocol (mesh, decomposition, graph preparation, geometry updates,
  compute, result, diagnostics) is driven directly, including the graph paths, the rank
  chunk, padding atoms, and the validation boundary;
* atomicity and lifecycle: a failed preparation or compute leaves the last published
  result readable, and no Kokkos owner survives finalization;
* the pipeline result for orthogonal and strongly triclinic cells is compared against the
  approved Task 10 private engine driven with the frozen fixture inputs, and the ordinary
  D2/D3 evaluators and their factory behavior are shown to be unchanged.

Only the immutable configuration is public API. The classes, accessor, and private
reverse seam stay out of the package exports.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import symmetrix
from symmetrix import corrections
from symmetrix.corrections import (
    CorrectionPipeline,
    FourierD3Correction,
    build_correction_runtime,
)
from symmetrix.fourier_d3_factors import decompose_d3_c6_reference
from symmetrix.neighbor_graph import build_host_neighbor_graph
from reference import fourier_d3


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TABLES_FILE = "libsymmetrix/source/dispersion_parameters.cpp"
PROVENANCE_FILE = "libsymmetrix/source/dispersion_parameter_provenance.hpp"
FIXTURE_DIRECTORY = Path(__file__).parent / "reference" / "fixtures"
REFERENCE_FIXTURE = json.loads(
    (FIXTURE_DIRECTORY / "fourier_d3_reference_cases.json").read_text(encoding="utf-8")
)
CASES = {case["name"]: case for case in REFERENCE_FIXTURE["cases"]}
PARAMETERS = REFERENCE_FIXTURE["parameters"]
CUTOFF = float(PARAMETERS["coordination_cutoff_angstrom"])
BOHR_TO_ANGSTROM = 0.52917725999999998

# The Task 10 private engine is driven with the frozen fixture damping, so the reference
# and the evaluator must agree on the BJ parameters to the last bit.
DAMPING = fourier_d3.DampingParameters.from_bohr(
    s6=PARAMETERS["s6"],
    s8=PARAMETERS["s8"],
    a1=PARAMETERS["a1"],
    a2_bohr=PARAMETERS["a2_bohr"],
)

MESH = (8, 8, 8)
ACCEPTANCE_MESH = (24, 24, 24)
SPLINE_ORDER = 4
PRECISIONS = ("float64", "float32")


def _assert_force_replay_close(actual, expected):
    """Allow only the few-ULP drift of an OpenMP reduction tree."""

    np.testing.assert_array_max_ulp(actual, expected, maxulp=4)


CLASS_NAMES = {"float64": "FourierD3Kokkos", "float32": "FourierD3KokkosFloat"}
# Two independent paths run the same engine on the same mesh, so the comparison is a
# connection check rather than a convergence check: only the reduction order differs.
_BANDS = {"float64": (1e-11, 1e-13), "float32": (5e-5, 1e-6)}
# The same bands for the two comparison helpers, which spell their keywords differently.
TOLERANCE = {
    precision: {"rtol": relative, "atol": absolute}
    for precision, (relative, absolute) in _BANDS.items()
}
APPROX = {
    precision: {"rel": relative, "abs": absolute}
    for precision, (relative, absolute) in _BANDS.items()
}
# |pipeline - mesh-free oracle| at the Task 10 acceptance mesh, order 4 (Task 10 measured
# 5e-6 eV/Angstrom for forces and 2e-6 eV for the strain derivative).
DECLARED_ORACLE_ENERGY_BAND_EV = 1e-6

# The total force sum is the mesh's own translation residual rather than a zero
# identity: the assignment is not exactly translation invariant at a finite mesh.
# Task 10 declared 2e-3 absolutely at the 8^3 mesh for both precisions.
TRANSLATION_RESIDUAL_BAND = {"float64": 2e-3, "float32": 2e-3}

_EVALUATOR_SYMBOLS = ("FourierD3Kokkos", "FourierD3KokkosFloat")
_ACCESSOR_SYMBOL = "_fourier_d3_canonical_tables"


# ---------------------------------------------------------------------------
# inputs and references
# ---------------------------------------------------------------------------


def _native():
    module = symmetrix.load_backend()
    if not module._kokkos_is_initialized():
        module._init_kokkos()
    return module


@pytest.fixture(scope="module")
def native():
    module = _native()
    capability = getattr(module, "_fourier_d3_capability", None)
    if not callable(capability):
        pytest.skip(
            "the installed native extension predates the FourierD3 capability report; "
            "rebuild the backend to exercise the evaluator"
        )
    report = capability()
    if not report["available"]:
        pytest.skip(
            "this build has no FourierD3 FFT provider "
            f"({report['reason'] or 'unreported'}); the evaluator is not compiled and "
            "nothing is emulated in its place"
        )
    for symbol in (*_EVALUATOR_SYMBOLS, _ACCESSOR_SYMBOL):
        assert hasattr(module, symbol), (
            f"the build reports an available FourierD3 FFT provider, so {symbol} must be "
            "present; no host or real-space fallback exists"
        )
    return module


@pytest.fixture(autouse=True)
def _disarm_allocation_fault(native):
    native._fourier_d3_clear_allocation_fault()
    yield
    native._fourier_d3_clear_allocation_fault()


def _source_tables() -> dict[str, np.ndarray]:
    """The repository's generated D3 tables, parsed from the C++ source (tests only)."""
    saved = sys.dont_write_bytecode
    try:
        if str(REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(REPOSITORY_ROOT))
        from tools.generate_fourier_d3_reference import parse_tables

        return parse_tables(REPOSITORY_ROOT / TABLES_FILE)
    finally:
        sys.dont_write_bytecode = saved


def _source_provenance() -> dict[str, object]:
    """The generated provenance header, parsed from the C++ source (tests only)."""
    text = (REPOSITORY_ROOT / PROVENANCE_FILE).read_text(encoding="utf-8")
    hashes = dict(re.findall(r'SourceHash\{"([^"]+)",\s*"([^"]+)"\}', text))
    schema = re.search(r'schema = "([^"]+)"', text).group(1)
    upstream = {
        name: re.search(rf'{name} = "([^"]+)"', text).group(1)
        for name in ("upstream_version", "upstream_tag", "upstream_commit")
    }
    return {"schema": schema, "source_hashes": hashes, **upstream}


def _case_factors(case) -> fourier_d3.LowRankFactors:
    decomposition = case["decomposition"]
    return fourier_d3.LowRankFactors(
        species=np.asarray(decomposition["species"], dtype=np.int64),
        eigs=np.asarray(decomposition["eigs_ev_angstrom6"], dtype=np.float64),
        v_q=np.asarray(decomposition["v_q"], dtype=np.float64),
        cn_ref=np.asarray(decomposition["cn_ref"], dtype=np.float64),
        valid=np.asarray(decomposition["valid"], dtype=bool),
        q_table=np.asarray(case["q_table_angstrom"], dtype=np.float64),
    )


def _atoms(case_name: str) -> Atoms:
    case = CASES[case_name]
    return Atoms(
        numbers=np.asarray(case["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(case["positions_angstrom"], dtype=np.float64),
        cell=np.asarray(case["cell_angstrom"], dtype=np.float64),
        pbc=True,
    )


def _configuration(**overrides) -> FourierD3Correction:
    arguments = {
        "functional": PARAMETERS["functional"],
        "coordination_cutoff": CUTOFF,
        "skin": 0.0,
        "mesh": MESH,
        "spline_order": SPLINE_ORDER,
    }
    arguments.update(overrides)
    return FourierD3Correction(**arguments)


def _evaluator(native, precision="float64", **overrides):
    """Construct the dtype-specific evaluator from the resolved configuration."""

    configuration = _configuration(**overrides)
    tables = native._fourier_d3_canonical_tables()
    evaluator_class = getattr(native, CLASS_NAMES[precision])
    evaluator = evaluator_class(
        configuration.functional,
        float(configuration.coordination_cutoff),
        int(configuration.spline_order),
        int(configuration.chunk_size),
        tables["table_fingerprint"],
        tables["d3_rcov"],
        tables["d3_r2r4"],
    )
    return evaluator, configuration, tables


def _payload(native, tables, case_name, *, tolerance=None, rank_cap=None):
    """The decomposition payload the runtime pushes, from the canonical tables."""

    case = CASES[case_name]
    species = np.unique(np.asarray(case["atomic_numbers"], dtype=np.int64))
    factors = decompose_d3_c6_reference(
        tables["d3_c6"],
        species,
        table_fingerprint=tables["table_fingerprint"],
        tolerance=PARAMETERS["decomposition_tolerance"]
        if tolerance is None
        else tolerance,
        rank_cap=rank_cap,
    )
    return (
        factors.species,
        factors.eigs,
        factors.v_q,
        factors.cn_ref,
        factors.valid,
        factors.max_relative_error,
        factors.tolerance_met,
    )


def _payload_for_species(native, tables, species):
    """Build the canonical payload for an explicit same-layout species set."""

    factors = decompose_d3_c6_reference(
        tables["d3_c6"],
        species,
        table_fingerprint=tables["table_fingerprint"],
        tolerance=PARAMETERS["decomposition_tolerance"],
    )
    return (
        factors.species,
        factors.eigs,
        factors.v_q,
        factors.cn_ref,
        factors.valid,
        factors.max_relative_error,
        factors.tolerance_met,
    )


def _graph_of(atoms, build_cutoff, positions, cell):
    """The receiver-major host graph of ``atoms`` with its edge distances."""

    graph = build_host_neighbor_graph(atoms, build_cutoff)
    offsets = np.empty(len(atoms) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(graph.num_neigh, out=offsets[1:])
    delta = positions[graph.sources] - positions[graph.receivers] + graph.shifts @ cell
    distances = np.linalg.norm(delta, axis=1)
    return graph, offsets, distances


def _host_graph(case_name, build_cutoff):
    """The receiver-major host graph the runtime builds for a case."""

    case = CASES[case_name]
    return _graph_of(
        _atoms(case_name),
        build_cutoff,
        np.asarray(case["positions_angstrom"], dtype=np.float64),
        np.asarray(case["cell_angstrom"], dtype=np.float64),
    )


def _load(evaluator, tables, case_name, *, native=None, mesh=MESH, **payload):
    """Push the mesh, the decomposition payload, and the host graph of a case."""

    case = CASES[case_name]
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    inverse_cell = np.ascontiguousarray(np.linalg.inv(cell))
    build_cutoff = CUTOFF
    graph, offsets, distances = _host_graph(case_name, build_cutoff)
    evaluator.set_mesh(mesh)
    evaluator.set_decomposition(
        *_payload(_native() if native is None else native, tables, case_name, **payload)
    )
    evaluator.prepare_host_graph(
        numbers,
        offsets.astype(np.int32),
        graph.sources,
        graph.shifts,
        positions,
        cell,
        build_cutoff,
    )
    return {
        "numbers": numbers,
        "positions": positions,
        "cell": cell,
        "inverse_cell": inverse_cell,
        "offsets": offsets.astype(np.int32),
        "sources": graph.sources,
        "shifts": graph.shifts,
        "distances": distances,
        "build_cutoff": build_cutoff,
    }


def _reference(
    native,
    case_name,
    *,
    factors=None,
    precision="float64",
    mesh=MESH,
    chunk=None,
    spline_order=SPLINE_ORDER,
    cutoff=CUTOFF,
):
    """Task 10's private reverse engine driven with the frozen fixture inputs.

    The graph is the runtime's own receiver-major host graph, so the two paths see the
    same topology, the same cell, and the same distances; the decomposition, the
    covalent radii, and the damping come from the frozen fixture rather than from the
    accessor, so a disagreement is a real disagreement.
    """

    case = CASES[case_name]
    factors = _case_factors(case) if factors is None else factors
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int32)
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    radii = np.asarray(case["covalent_radii_angstrom"], dtype=np.float64)
    q_table = np.asarray(case["q_table_angstrom"], dtype=np.float64)
    graph, offsets, distances = _host_graph(case_name, cutoff)
    rank = factors.rank
    width = rank if chunk is None else int(chunk)
    total = int(np.ceil(rank / width)) * width

    energy = 0.0
    atomic = np.zeros(numbers.size, dtype=np.float64)
    forces = np.zeros((numbers.size, 3), dtype=np.float64)
    stress = np.zeros(6, dtype=np.float64)
    for begin in range(0, total, width):
        eigenvalues = np.zeros(width, dtype=np.float64)
        available = min(width, rank - begin)
        if available > 0:
            eigenvalues[:available] = factors.eigs[begin : begin + available]
        columns = np.zeros((factors.n_species, factors.n_ref, total), dtype=np.float64)
        available = min(width, rank - begin)
        if available > 0:
            columns[:, :, begin : begin + available] = factors.v_q[
                :, :, begin : begin + available
            ]
        seam = native._FourierD3ReciprocalReverse(precision, width)
        seam.reserve(
            mesh[0],
            mesh[1],
            mesh[2],
            factors.n_species,
            numbers.size,
            int(offsets[-1]),
            factors.n_ref,
            total,
            width,
            spline_order,
        )
        seam.set_cell(np.ascontiguousarray(cell.reshape(-1)))
        seam.set_species(
            np.ascontiguousarray(factors.species, dtype=np.int32),
            np.ascontiguousarray(q_table, dtype=np.float64),
        )
        seam.set_atoms(np.ascontiguousarray(positions.reshape(-1)), numbers)
        seam.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
        seam.set_rank(begin, eigenvalues)
        seam.set_graph(offsets, graph.sources, graph.shifts, distances, numbers, radii)
        seam.set_reference(
            np.ascontiguousarray(factors.cn_ref, dtype=np.float64),
            np.ascontiguousarray(factors.valid, dtype=bool),
            np.ascontiguousarray(factors.species, dtype=np.int32),
            np.ascontiguousarray(columns, dtype=np.float64),
        )
        outcome = seam.run(cutoff)
        energy += float(outcome["energy"])
        atomic += np.asarray(seam.atomic_energies(), dtype=np.float64)
        forces += np.asarray(seam.forces(), dtype=np.float64).reshape(-1, 3)
        stress += np.asarray(seam.stress(), dtype=np.float64)
    return {
        "energy": energy,
        "atomic_energies": atomic,
        "forces": forces,
        "stress": stress,
    }


# ---------------------------------------------------------------------------
# A. canonical tables and provenance
# ---------------------------------------------------------------------------


def test_canonical_tables_match_the_generated_source_arrays(native):
    """The accessor must publish the generated tables unchanged, byte for byte."""

    tables = native._fourier_d3_canonical_tables()
    source = _source_tables()

    assert tables["d3_c6"].shape == (95, 95, 5, 5, 3)
    assert tables["d3_c6"].dtype == np.float64
    assert tables["d3_rcov"].shape == (95,)
    assert tables["d3_r2r4"].shape == (95,)
    assert tables["supported_atomic_number"] == 94
    np.testing.assert_array_equal(tables["d3_c6"], source["d3_c6"])
    np.testing.assert_array_equal(tables["d3_rcov"], source["d3_rcov"])
    np.testing.assert_array_equal(tables["d3_r2r4"], source["d3_r2r4"])


def test_canonical_table_digests_match_the_source_bytes(native):
    """Per-array content digests must be a function of the generated bytes alone."""

    tables = native._fourier_d3_canonical_tables()
    source = _source_tables()
    hashes = tables["table_hashes"]

    for name in ("d3_c6", "d3_rcov", "d3_r2r4"):
        expected = hashlib.sha256(
            np.ascontiguousarray(source[name], dtype=np.float64).tobytes()
        ).hexdigest()
        assert hashes[name] == f"sha256:{expected}", name


def test_canonical_table_fingerprint_is_stable_and_covers_the_provenance(native):
    """The fingerprint identifies the table set and is reproducible from its inputs."""

    tables = native._fourier_d3_canonical_tables()
    provenance = tables["provenance"]
    source = _source_provenance()
    source_tables = _source_tables()
    hashes = tables["table_hashes"]

    assert provenance["schema"] == source["schema"]
    assert provenance["upstream_version"] == source["upstream_version"]
    assert provenance["upstream_tag"] == source["upstream_tag"]
    assert provenance["upstream_commit"] == source["upstream_commit"]
    assert provenance["source_hashes"] == source["source_hashes"]
    assert provenance["license"] == "MIT"

    fingerprint = tables["table_fingerprint"]
    assert fingerprint.startswith("sha256:")
    assert len(fingerprint) == len("sha256:") + 64
    # Stable across calls, and different from any single array digest.
    assert native._fourier_d3_canonical_tables()["table_fingerprint"] == fingerprint
    assert fingerprint not in set(hashes.values())

    # The fingerprint is a function of the provenance identity and the array bytes, so
    # an independent recomputation from the parsed source must reproduce it.
    digest = hashlib.sha256()
    digest.update((source["schema"] + "\n").encode())
    digest.update((source["upstream_version"] + "\n").encode())
    digest.update((source["upstream_tag"] + "\n").encode())
    digest.update((source["upstream_commit"] + "\n").encode())
    for path, sha in source["source_hashes"].items():
        digest.update(f"{path}:{sha}\n".encode())
    for name in ("d3_c6", "d3_rcov", "d3_r2r4"):
        # Each published array contributes its own content digest, taken over the
        # generated bytes rather than over a re-serialization of the numpy view.
        content = hashlib.sha256(
            np.ascontiguousarray(source_tables[name], dtype=np.float64).tobytes()
        ).hexdigest()
        assert hashes[name] == f"sha256:{content}"
        digest.update(f"{name}:{hashes[name]}\n".encode())
    assert fingerprint == f"sha256:{digest.hexdigest()}"


def test_canonical_tables_agree_with_the_frozen_fixture_payloads(native):
    """The accessor's radii and r2r4 tables must be the fixture's per-atom values."""

    tables = native._fourier_d3_canonical_tables()
    for case in CASES.values():
        numbers = np.asarray(case["atomic_numbers"], dtype=np.int64)
        np.testing.assert_allclose(
            tables["d3_rcov"][numbers],
            np.asarray(case["covalent_radii_angstrom"], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        species = np.unique(numbers)
        np.testing.assert_allclose(
            tables["d3_r2r4"][species],
            np.asarray(case["q_table_angstrom"], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )


def test_evaluator_classes_stay_private(native):
    """The evaluator seam is available natively but is not a package export."""

    for symbol in (*_EVALUATOR_SYMBOLS, _ACCESSOR_SYMBOL):
        # The curated export list is what says "private": the name stays reachable
        # through the selected native module, exactly like every other native seam.
        assert symbol not in symmetrix.__all__, symbol
    assert "FourierD3Correction" in symmetrix.__all__


# ---------------------------------------------------------------------------
# B. the evaluator protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_evaluator_reports_the_configured_state_before_any_geometry(native, precision):
    """Construction resolves the BJ damping and reports the frozen configuration."""

    evaluator, configuration, _ = _evaluator(native, precision)
    report = evaluator.diagnostics()

    assert report["precision"] == precision
    assert report["functional"] == configuration.functional
    assert report["spline_order"] == SPLINE_ORDER
    assert report["rank_chunk"] == int(configuration.chunk_size)
    assert report["coordination_cutoff"] == pytest.approx(CUTOFF)
    assert report["damping"]["a1"] == pytest.approx(PARAMETERS["a1"])
    assert report["damping"]["a2_angstrom"] == pytest.approx(
        PARAMETERS["a2_angstrom"], rel=1e-15
    )
    assert report["damping"]["s6"] == pytest.approx(PARAMETERS["s6"])
    assert report["damping"]["s8"] == pytest.approx(PARAMETERS["s8"])
    assert report["computed"] is False
    assert report["active_species"] == []
    assert evaluator.graph_diagnostics()["num_edges"] == 0


@pytest.mark.parametrize("precision", PRECISIONS)
def test_evaluator_computes_every_property_from_one_generation(native, precision):
    """Energy, atomic energies, forces, and stress come from a single compute."""

    evaluator, _, tables = _evaluator(native, precision)
    _load(evaluator, tables, "orthogonal-mixed-species")
    evaluator.compute(True, True)
    result = evaluator.result(("energy", "atomic_energies", "forces", "stress"))
    tolerance = TOLERANCE[precision]

    assert set(result) == {"energy", "atomic_energies", "forces", "stress"}
    assert np.isfinite(result["energy"])
    np.testing.assert_allclose(
        result["atomic_energies"].sum(), result["energy"], **tolerance
    )
    assert result["forces"].shape == (
        len(CASES["orthogonal-mixed-species"]["atomic_numbers"]),
        3,
    )
    assert result["stress"].shape == (6,)
    # The total force sum is the mesh's translation residual, bounded absolutely;
    # the coordination part telescopes exactly and the seam comparison pins the rest.
    residual = float(np.max(np.abs(result["forces"].sum(axis=0))))
    assert residual <= TRANSLATION_RESIDUAL_BAND[precision], (
        f"the force sum must stay inside the mesh translation residual; measured "
        f"{residual:.3e}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_evaluator_selective_requests_return_only_what_was_computed(native, precision):
    """A selective request publishes exactly the requested properties."""

    evaluator, _, tables = _evaluator(native, precision)
    _load(evaluator, tables, "orthogonal-mixed-species")

    evaluator.compute(False, False)
    assert set(evaluator.result(("energy", "atomic_energies"))) == {
        "energy",
        "atomic_energies",
    }
    with pytest.raises(RuntimeError, match="forces"):
        evaluator.result(("forces",))

    evaluator.compute(True, False)
    assert set(evaluator.result(("energy", "forces"))) == {"energy", "forces"}
    with pytest.raises(RuntimeError, match="stress"):
        evaluator.result(("stress",))

    evaluator.compute(True, True)
    assert set(evaluator.result()) == {
        "energy",
        "atomic_energies",
        "forces",
        "stress",
    }


def test_evaluator_rejects_unknown_result_properties(native):
    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, "orthogonal-mixed-species")
    evaluator.compute(True, True)

    with pytest.raises(ValueError, match="virial"):
        evaluator.result(("virial",))


def test_evaluator_result_before_any_compute_is_rejected(native):
    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, "orthogonal-mixed-species")

    with pytest.raises(RuntimeError, match="compute"):
        evaluator.result(("energy",))


@pytest.mark.parametrize("precision", PRECISIONS)
def test_host_and_periodic_graph_paths_agree(native, precision):
    """The two graph sources must produce the same physics on the same topology."""

    case = "orthogonal-mixed-species"
    tolerance = TOLERANCE[precision]
    approx_tolerance = APPROX[precision]
    host, _, tables = _evaluator(native, precision)
    _load(host, tables, case)
    host.compute(True, True)
    host_result = host.result()

    device, _, tables = _evaluator(native, precision)
    device.set_mesh(MESH)
    device.set_decomposition(*_payload(native, tables, case))
    numbers = np.asarray(CASES[case]["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(CASES[case]["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(CASES[case]["cell_angstrom"], dtype=np.float64)
    device.prepare_periodic_graph(
        numbers, positions, cell, np.ascontiguousarray(np.linalg.inv(cell)), CUTOFF
    )
    device.compute(True, True)
    device_result = device.result()

    assert host.graph_diagnostics()["graph_backend"] == "host"
    assert device.graph_diagnostics()["graph_backend"] == "kokkos"
    assert device.graph_diagnostics()["num_edges"] > 0
    assert host.graph_diagnostics()["num_edges"] > 0
    assert device_result["energy"] == pytest.approx(
        host_result["energy"], **approx_tolerance
    )
    np.testing.assert_allclose(
        device_result["atomic_energies"], host_result["atomic_energies"], **tolerance
    )
    np.testing.assert_allclose(
        device_result["forces"], host_result["forces"], **tolerance
    )
    np.testing.assert_allclose(
        device_result["stress"], host_result["stress"], **tolerance
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_evaluator_matches_the_task10_engine(native, precision):
    """Every published property agrees with the approved Task 10 private engine."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native, precision)
    _load(evaluator, tables, case)
    evaluator.compute(True, True)
    result = evaluator.result()
    reference = _reference(native, case, precision=precision)
    tolerance = TOLERANCE[precision]
    approx_tolerance = APPROX[precision]

    assert result["energy"] == pytest.approx(reference["energy"], **approx_tolerance)
    np.testing.assert_allclose(
        result["atomic_energies"], reference["atomic_energies"], **tolerance
    )
    np.testing.assert_allclose(result["forces"], reference["forces"], **tolerance)
    np.testing.assert_allclose(result["stress"], reference["stress"], **tolerance)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_rank_chunk_invariance(native, chunk):
    """The configured rank chunk must not change any published property."""

    case = "orthogonal-mixed-species"
    baseline, _, tables = _evaluator(native, chunk_size=512)
    _load(baseline, tables, case)
    baseline.compute(True, True)
    reference = baseline.result()

    chunked, _, tables = _evaluator(native, chunk_size=chunk)
    _load(chunked, tables, case)
    chunked.compute(True, True)
    result = chunked.result()

    assert (
        chunked.diagnostics()["retained_rank"]
        == baseline.diagnostics()["retained_rank"]
    )
    assert result["energy"] == pytest.approx(reference["energy"], **APPROX["float64"])
    np.testing.assert_allclose(
        result["atomic_energies"], reference["atomic_energies"], **TOLERANCE["float64"]
    )
    np.testing.assert_allclose(
        result["forces"], reference["forces"], **TOLERANCE["float64"]
    )
    np.testing.assert_allclose(
        result["stress"], reference["stress"], **TOLERANCE["float64"]
    )
    # And the same chunking, driven through the Task 10 engine, agrees too.
    seam = _reference(native, case, chunk=chunk)
    np.testing.assert_allclose(result["forces"], seam["forces"], **TOLERANCE["float64"])


def test_padding_atoms_stay_inactive(native):
    """Z = 0 atoms carry no energy or force and do not enter the decomposition."""

    case = "orthogonal-mixed-species"
    clean, _, tables = _evaluator(native)
    _load(clean, tables, case, native=native)
    clean.compute(True, True)
    reference = clean.result()

    numbers = np.asarray(CASES[case]["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(CASES[case]["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(CASES[case]["cell_angstrom"], dtype=np.float64)
    padded_numbers = np.concatenate([numbers, np.zeros(1, dtype=np.int64)])
    padded_positions = np.concatenate([positions, np.full((1, 3), 1.7)])

    padded, _, tables = _evaluator(native)
    padded.set_mesh(MESH)
    padded.set_decomposition(*_payload(native, tables, case))
    # The host graph is the qualified CPU path, and it is built over the padded atoms, so
    # the padding atom really is a receiver and a source in the loaded topology. The
    # shared device builder requires every atomic number in 1..94, so a padded system is
    # rejected loudly there instead of being given an invented channel.
    padded_atoms = Atoms(
        numbers=padded_numbers,
        positions=padded_positions,
        cell=cell,
        pbc=True,
    )
    graph, offsets, distances = _graph_of(padded_atoms, CUTOFF, padded_positions, cell)
    assert graph.num_neigh[-1] > 0
    padded.prepare_host_graph(
        padded_numbers,
        offsets.astype(np.int32),
        graph.sources,
        graph.shifts,
        padded_positions,
        cell,
        CUTOFF,
    )
    padded.compute(True, True)
    result = padded.result()

    assert (
        padded.diagnostics()["active_species"] == clean.diagnostics()["active_species"]
    )
    assert result["energy"] == pytest.approx(reference["energy"], **APPROX["float64"])
    assert result["atomic_energies"][-1] == 0.0
    np.testing.assert_allclose(result["forces"][-1], np.zeros(3), atol=0.0)
    np.testing.assert_allclose(
        result["forces"][:-1], reference["forces"], **TOLERANCE["float64"]
    )


def test_padding_atoms_are_rejected_by_the_shared_device_graph(native):
    """The device builder refuses Z = 0 rather than inventing a channel for it."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native)
    evaluator.set_mesh(MESH)
    evaluator.set_decomposition(*_payload(native, tables, case))
    numbers = np.asarray(CASES[case]["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(CASES[case]["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(CASES[case]["cell_angstrom"], dtype=np.float64)

    with pytest.raises((ValueError, RuntimeError), match="atomic number"):
        evaluator.prepare_periodic_graph(
            np.concatenate([numbers, np.zeros(1, dtype=np.int64)]),
            np.concatenate([positions, np.full((1, 3), 1.7)]),
            cell,
            np.ascontiguousarray(np.linalg.inv(cell)),
            CUTOFF,
        )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_zero_active_species_publishes_exact_zeros(native, precision):
    """An empty and an all-padding system are exact zero-energy paths.

    Neither has an active species, so there is no decomposition to load and no
    reciprocal pass to run: the published generation is exactly zero and the engine is
    never reserved.  This is not a fallback - no host or real-space sum is substituted -
    it is the value the model has for a system with no active atoms.
    """

    case = "orthogonal-mixed-species"
    cell = np.asarray(CASES[case]["cell_angstrom"], dtype=np.float64)
    empty_i32 = np.empty(0, dtype=np.int32)
    empty_f64 = np.empty(0, dtype=np.float64)
    offsets = np.zeros(1, dtype=np.int32)

    def load(evaluator, numbers, positions, receiver_offsets, sources, shifts):
        evaluator.set_mesh(MESH)
        evaluator.set_decomposition(
            empty_i32,
            empty_f64,
            np.empty((0, 0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=bool),
            0.0,
            True,
        )
        evaluator.prepare_host_graph(
            numbers,
            receiver_offsets,
            sources,
            shifts,
            positions,
            np.ascontiguousarray(cell),
            CUTOFF,
        )

    empty, _, _ = _evaluator(native, precision)
    load(
        empty,
        empty_i32,
        np.empty((0, 3), dtype=np.float64),
        offsets,
        empty_i32,
        np.empty((0, 3), dtype=np.int32),
    )
    empty.compute(True, True)
    result = empty.result()
    report = empty.diagnostics()

    assert result["energy"] == 0.0
    assert result["atomic_energies"].shape == (0,)
    assert result["forces"].shape == (0, 3)
    np.testing.assert_array_equal(result["stress"], np.zeros(6))
    assert report["active_species"] == []
    assert report["rank_total"] == 0
    assert report["retained_rank"] == 0
    assert report["engine_loaded"] is False
    assert report["active_bytes"] == 0
    assert empty.graph_diagnostics()["num_edges"] == 0

    # The all-padding case keeps its real graph - the padding atoms are receivers and
    # sources - but every one of them is inactive, so the energy is still exactly zero.
    padded_numbers = np.zeros(2, dtype=np.int32)
    padded_positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]], dtype=np.float64)
    padded_atoms = Atoms(
        numbers=padded_numbers, positions=padded_positions, cell=cell, pbc=True
    )
    graph, padded_offsets, _ = _graph_of(padded_atoms, CUTOFF, padded_positions, cell)
    padded, _, _ = _evaluator(native, precision)
    load(
        padded,
        padded_numbers,
        padded_positions,
        padded_offsets.astype(np.int32),
        graph.sources,
        graph.shifts,
    )
    padded.compute(True, True)
    padded_result = padded.result()

    assert padded.graph_diagnostics()["num_edges"] > 0
    assert padded.diagnostics()["active_species"] == []
    assert padded.diagnostics()["engine_loaded"] is False
    assert padded_result["energy"] == 0.0
    np.testing.assert_array_equal(padded_result["atomic_energies"], np.zeros(2))
    np.testing.assert_array_equal(padded_result["forces"], np.zeros((2, 3)))
    np.testing.assert_array_equal(padded_result["stress"], np.zeros(6))


def test_atom_count_change_rebuilds_the_geometry(native):
    """A different atom count is a different geometry, not a reused one."""

    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, "orthogonal-mixed-species")
    evaluator.compute(True, True)
    first = evaluator.result()

    case = "triclinic-mixed-species"
    numbers = np.asarray(CASES[case]["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(CASES[case]["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(CASES[case]["cell_angstrom"], dtype=np.float64)
    evaluator.set_mesh(MESH)
    evaluator.set_decomposition(*_payload(native, tables, case))
    evaluator.prepare_periodic_graph(
        numbers, positions, cell, np.ascontiguousarray(np.linalg.inv(cell)), CUTOFF
    )
    evaluator.compute(True, True)
    second = evaluator.result()

    assert second["forces"].shape[0] == numbers.size
    assert second["energy"] != pytest.approx(first["energy"])
    np.testing.assert_allclose(
        second["forces"], _reference(native, case)["forces"], **TOLERANCE["float64"]
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_geometry_update_moves_the_result(native, precision):
    """``update_geometry`` refreshes positions, cell, and the edge distances."""

    case = "orthogonal-mixed-species"
    tolerance = TOLERANCE[precision]
    approx_tolerance = APPROX[precision]
    evaluator, _, tables = _evaluator(native, precision)
    state = _load(evaluator, tables, case)
    evaluator.compute(True, True)
    baseline = evaluator.result()

    shifted = state["positions"] + np.array([0.0, 0.0, 0.13])
    evaluator.update_geometry(shifted, state["cell"], state["inverse_cell"])
    evaluator.compute(True, True)
    moved = evaluator.result()

    assert moved["energy"] != pytest.approx(baseline["energy"])
    assert moved["forces"] != pytest.approx(baseline["forces"])

    # The reference sees the same displaced geometry through the same topology.
    reference = _reference_displaced(native, case, shifted, precision=precision)
    assert moved["energy"] == pytest.approx(reference["energy"], **approx_tolerance)
    np.testing.assert_allclose(moved["forces"], reference["forces"], **tolerance)

    # Repeating the identical update is a no-op: same result, no new generation.
    evaluator.update_geometry(shifted, state["cell"], state["inverse_cell"])
    evaluator.compute(True, True)
    repeated = evaluator.result()
    assert repeated["energy"] == moved["energy"]
    _assert_force_replay_close(repeated["forces"], moved["forces"])


def test_cell_change_with_the_same_topology_is_honoured(native):
    """A safe affine cell change updates the geometry without rebuilding the graph."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native)
    state = _load(evaluator, tables, case)
    evaluator.compute(True, True)
    baseline = evaluator.result()

    strain = np.eye(3) + np.array(
        [[0.01, 0.0, 0.0], [0.0, -0.01, 0.0], [0.0, 0.0, 0.005]]
    )
    stretched_cell = state["cell"] @ strain.T
    evaluator.update_geometry(
        state["positions"],
        stretched_cell,
        np.ascontiguousarray(np.linalg.inv(stretched_cell)),
    )
    evaluator.compute(True, True)
    stretched = evaluator.result()

    assert stretched["energy"] != pytest.approx(baseline["energy"])
    reference = _reference_displaced(
        native, case, state["positions"], cell=stretched_cell
    )
    assert stretched["energy"] == pytest.approx(
        reference["energy"], **APPROX["float64"]
    )
    np.testing.assert_allclose(
        stretched["stress"], reference["stress"], **TOLERANCE["float64"]
    )


def test_mesh_change_invalidates_the_reciprocal_result(native):
    """A different mesh is a different discretisation, and diagnostics says so."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, case, mesh=(8, 8, 8))
    evaluator.compute(True, True)
    coarse = evaluator.result()
    assert evaluator.diagnostics()["mesh"] == [8, 8, 8]

    evaluator.set_mesh((12, 12, 12))
    evaluator.compute(True, True)
    fine = evaluator.result()

    assert evaluator.diagnostics()["mesh"] == [12, 12, 12]
    assert fine["energy"] != pytest.approx(coarse["energy"])
    reference = _reference(native, case, mesh=(12, 12, 12))
    assert fine["energy"] == pytest.approx(reference["energy"], **APPROX["float64"])
    np.testing.assert_allclose(
        fine["forces"], reference["forces"], **TOLERANCE["float64"]
    )


def test_decomposition_change_invalidates_the_reciprocal_result(native):
    """A different decomposition payload is not reused."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, case)
    evaluator.compute(True, True)
    full = evaluator.result()
    assert evaluator.diagnostics()["retained_rank"] > 1

    evaluator.set_decomposition(*_payload(native, tables, case, rank_cap=1))
    evaluator.compute(True, True)
    capped = evaluator.result()

    assert evaluator.diagnostics()["retained_rank"] == 1
    assert evaluator.diagnostics()["decomposition_tolerance_met"] is False
    assert capped["energy"] != pytest.approx(full["energy"])
    capped_factors = decompose_d3_c6_reference(
        tables["d3_c6"],
        np.asarray(CASES[case]["decomposition"]["species"], dtype=np.int64),
        table_fingerprint=tables["table_fingerprint"],
        tolerance=PARAMETERS["decomposition_tolerance"],
        rank_cap=1,
    )
    reference = _reference(
        native,
        case,
        factors=fourier_d3.LowRankFactors(
            species=capped_factors.species,
            eigs=capped_factors.eigs,
            v_q=capped_factors.v_q,
            cn_ref=capped_factors.cn_ref,
            valid=capped_factors.valid,
            q_table=np.asarray(CASES[case]["q_table_angstrom"], dtype=np.float64),
        ),
    )
    assert capped["energy"] == pytest.approx(reference["energy"], **APPROX["float64"])


def test_decomposition_scalars_are_part_of_the_payload(native):
    """A payload that differs only in its achieved error is not the same payload.

    The two scalars a decomposition reports are part of what a caller pushed, so a
    payload that changes only them must be installed rather than skipped as a repeat.
    """

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, case)
    evaluator.compute(True, True)
    first = evaluator.result()
    achieved = evaluator.diagnostics()["decomposition_max_relative_error"]

    payload = list(_payload(native, tables, case))
    payload[5] = achieved / 2.0
    payload[6] = False
    evaluator.set_decomposition(*payload)
    evaluator.compute(True, True)

    report = evaluator.diagnostics()
    assert report["decomposition_max_relative_error"] == pytest.approx(achieved / 2.0)
    assert report["decomposition_tolerance_met"] is False
    # Only the reported scalars changed, so the physics is unchanged.
    assert evaluator.result()["energy"] == first["energy"]


def test_species_change_invalidates_the_decomposition(native):
    """A different active-species set is a different decomposition."""

    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, "orthogonal-mixed-species")
    evaluator.compute(True, True)
    mixed = evaluator.result()

    case = "coordination-sensitive"
    numbers = np.asarray(CASES[case]["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(CASES[case]["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(CASES[case]["cell_angstrom"], dtype=np.float64)
    evaluator.set_decomposition(*_payload(native, tables, case))
    evaluator.prepare_periodic_graph(
        numbers, positions, cell, np.ascontiguousarray(np.linalg.inv(cell)), CUTOFF
    )
    evaluator.compute(True, True)
    other = evaluator.result()

    assert evaluator.diagnostics()["active_species"] == sorted(
        int(value) for value in np.unique(numbers)
    )
    assert other["energy"] != pytest.approx(mixed["energy"])
    np.testing.assert_allclose(
        other["forces"], _reference(native, case)["forces"], **TOLERANCE["float64"]
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_same_layout_species_replacement_matches_a_fresh_evaluator(native, precision):
    """A new payload must be pushed even when every reserved extent is unchanged."""

    case = "orthogonal-mixed-species"
    reused, _, tables = _evaluator(native, precision)
    state = _load(reused, tables, case, native=native)
    reused.compute(True, True)
    original_energy = reused.result()["energy"]

    replacement_numbers = np.asarray([6, 8, 14, 8], dtype=np.int32)
    replacement_payload = _payload_for_species(native, tables, [6, 8, 14])
    assert len(replacement_payload[0]) == 3
    assert len(replacement_payload[1]) == reused.diagnostics()["retained_rank"] == 5
    reused.set_decomposition(*replacement_payload)
    reused.prepare_host_graph(
        replacement_numbers,
        state["offsets"],
        state["sources"],
        state["shifts"],
        state["positions"],
        state["cell"],
        state["build_cutoff"],
    )
    reused.compute(True, True)
    actual = reused.result()

    fresh, _, _ = _evaluator(native, precision)
    fresh.set_mesh(MESH)
    fresh.set_decomposition(*replacement_payload)
    fresh.prepare_host_graph(
        replacement_numbers,
        state["offsets"],
        state["sources"],
        state["shifts"],
        state["positions"],
        state["cell"],
        state["build_cutoff"],
    )
    fresh.compute(True, True)
    expected = fresh.result()

    assert actual["energy"] != pytest.approx(original_energy)
    assert actual["energy"] == expected["energy"]
    np.testing.assert_array_equal(
        actual["atomic_energies"], expected["atomic_energies"]
    )
    _assert_force_replay_close(actual["forces"], expected["forces"])
    np.testing.assert_array_equal(actual["stress"], expected["stress"])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_same_layout_factor_replacement_matches_a_fresh_evaluator(native, precision):
    """Changed factor values are applied when their dimensions stay unchanged."""

    case = "orthogonal-mixed-species"
    reused, _, tables = _evaluator(native, precision)
    state = _load(reused, tables, case, native=native)
    reused.compute(True, True)
    original_energy = reused.result()["energy"]

    replacement_payload = list(_payload(native, tables, case))
    replacement_payload[2] = np.ascontiguousarray(replacement_payload[2] * 0.75)
    reused.set_decomposition(*replacement_payload)
    reused.update_geometry(state["positions"], state["cell"], state["inverse_cell"])
    reused.compute(True, True)
    actual = reused.result()

    fresh, _, _ = _evaluator(native, precision)
    fresh.set_mesh(MESH)
    fresh.set_decomposition(*replacement_payload)
    fresh.prepare_host_graph(
        state["numbers"],
        state["offsets"],
        state["sources"],
        state["shifts"],
        state["positions"],
        state["cell"],
        state["build_cutoff"],
    )
    fresh.compute(True, True)
    expected = fresh.result()

    assert actual["energy"] != pytest.approx(original_energy)
    assert actual["energy"] == expected["energy"]
    np.testing.assert_array_equal(
        actual["atomic_energies"], expected["atomic_energies"]
    )
    _assert_force_replay_close(actual["forces"], expected["forces"])
    np.testing.assert_array_equal(actual["stress"], expected["stress"])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_compute_only_same_layout_species_replacement_matches_fresh(native, precision):
    """Compute applies a staged species payload without another preparation."""

    case = "orthogonal-mixed-species"
    reused, _, tables = _evaluator(native, precision)
    state = _load(reused, tables, case, native=native)
    reused.compute(True, True)
    original = reused.result()

    replacement_payload = _payload_for_species(native, tables, [6, 8, 14])
    reused.set_decomposition(*replacement_payload)
    reused.compute(True, True)
    actual = reused.result()

    fresh, _, _ = _evaluator(native, precision)
    fresh.set_mesh(MESH)
    fresh.set_decomposition(*replacement_payload)
    fresh.prepare_host_graph(
        state["numbers"],
        state["offsets"],
        state["sources"],
        state["shifts"],
        state["positions"],
        state["cell"],
        state["build_cutoff"],
    )
    fresh.compute(True, True)
    expected = fresh.result()

    assert actual["energy"] != pytest.approx(original["energy"])
    assert actual["energy"] == expected["energy"]
    np.testing.assert_array_equal(
        actual["atomic_energies"], expected["atomic_energies"]
    )
    _assert_force_replay_close(actual["forces"], expected["forces"])
    np.testing.assert_array_equal(actual["stress"], expected["stress"])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_compute_only_same_layout_factor_replacement_matches_fresh(native, precision):
    """Compute applies staged numeric factors whose reserved shape is unchanged."""

    case = "orthogonal-mixed-species"
    reused, _, tables = _evaluator(native, precision)
    state = _load(reused, tables, case, native=native)
    reused.compute(True, True)
    original = reused.result()

    replacement_payload = list(_payload(native, tables, case))
    replacement_payload[2] = np.ascontiguousarray(replacement_payload[2] * 0.75)
    reused.set_decomposition(*replacement_payload)
    reused.compute(True, True)
    actual = reused.result()

    fresh, _, _ = _evaluator(native, precision)
    fresh.set_mesh(MESH)
    fresh.set_decomposition(*replacement_payload)
    fresh.prepare_host_graph(
        state["numbers"],
        state["offsets"],
        state["sources"],
        state["shifts"],
        state["positions"],
        state["cell"],
        state["build_cutoff"],
    )
    fresh.compute(True, True)
    expected = fresh.result()

    assert actual["energy"] != pytest.approx(original["energy"])
    assert actual["energy"] == expected["energy"]
    np.testing.assert_array_equal(
        actual["atomic_energies"], expected["atomic_energies"]
    )
    _assert_force_replay_close(actual["forces"], expected["forces"])
    np.testing.assert_array_equal(actual["stress"], expected["stress"])


def test_functional_change_selects_the_matching_bj_parameters(native):
    """A different functional is a different damping, never a silent reuse."""

    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, "orthogonal-mixed-species")
    evaluator.compute(True, True)
    pbe = evaluator.result()

    other, _, _ = _evaluator(native, functional="pbe0")
    _load(other, tables, "orthogonal-mixed-species")
    other.compute(True, True)
    pbe0 = other.result()

    assert other.diagnostics()["damping"] != evaluator.diagnostics()["damping"]
    assert pbe0["energy"] != pytest.approx(pbe["energy"])

    with pytest.raises((RuntimeError, ValueError), match="functional|parameter"):
        _evaluator(native, functional="not-a-functional")


def test_evaluator_diagnostics_describe_the_prepared_state(native):
    """Diagnostics expose the state the runtime and Task 12 need."""

    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, "orthogonal-mixed-species")
    evaluator.compute(True, True)
    report = evaluator.diagnostics()
    graph = evaluator.graph_diagnostics()

    assert report["precision"] == "float64"
    assert report["mesh"] == list(MESH)
    assert report["mesh_points"] == int(np.prod(MESH))
    assert report["active_species"] == [1, 6, 8]
    assert report["retained_rank"] >= 1
    assert report["rank_total"] >= report["retained_rank"]
    assert report["n_references"] == 5
    assert (
        report["decomposition_max_relative_error"]
        <= PARAMETERS["decomposition_tolerance"]
    )
    assert report["decomposition_tolerance_met"] is True
    assert report["table_fingerprint"].startswith("sha256:")
    assert report["computed"] is True
    assert report["active_bytes"] > 0
    assert report["capacity_bytes"] >= report["active_bytes"]
    assert report["plan_builds"] >= 1
    assert report["kernel_key_misses"] >= 1
    assert report["stencil_points"] >= 1
    assert graph["graph_backend"] == "host"
    assert graph["num_nodes"] == len(
        CASES["orthogonal-mixed-species"]["atomic_numbers"]
    )
    _, offsets, _ = _host_graph("orthogonal-mixed-species", CUTOFF)
    assert graph["num_edges"] == int(offsets[-1]) > 0


# ---------------------------------------------------------------------------
# C. the validation boundary
# ---------------------------------------------------------------------------


def test_evaluator_rejects_invalid_construction_arguments(native):
    tables = native._fourier_d3_canonical_tables()
    evaluator_class = native.FourierD3Kokkos
    arguments = (
        PARAMETERS["functional"],
        CUTOFF,
        SPLINE_ORDER,
        512,
        tables["table_fingerprint"],
        tables["d3_rcov"],
        tables["d3_r2r4"],
    )

    with pytest.raises(ValueError, match="spline order"):
        evaluator_class(*arguments[:2], 7, *arguments[3:])
    with pytest.raises(ValueError, match="rank chunk|rank channel"):
        evaluator_class(*arguments[:3], 0, *arguments[4:])
    with pytest.raises(ValueError, match="fingerprint"):
        evaluator_class(*arguments[:4], "", *arguments[5:])
    with pytest.raises(ValueError, match="radii"):
        evaluator_class(*arguments[:5], np.zeros(4), *arguments[6:])
    with pytest.raises(ValueError, match="q_table|r2r4"):
        evaluator_class(*arguments[:6], np.zeros(4))
    with pytest.raises(ValueError, match="coordination cutoff"):
        evaluator_class(arguments[0], 0.0, *arguments[2:])


def test_evaluator_accepts_strided_views_and_rejects_wrong_dtype_and_shape(native):
    evaluator, _, tables = _evaluator(native)
    payload = list(_payload(native, tables, "orthogonal-mixed-species"))
    evaluator.set_mesh(MESH)
    evaluator.set_decomposition(*payload)

    case = CASES["orthogonal-mixed-species"]
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    inverse_cell = np.ascontiguousarray(np.linalg.inv(cell))

    # A strided view of the same values is a view, not a different geometry: it is
    # copied into the contiguous buffer the kernels read rather than reinterpreted.
    padded = np.zeros((positions.shape[0], 6))
    padded[:, ::2] = positions
    strided = padded[:, ::2]
    assert not strided.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(strided, positions)
    strided_evaluator, _, _ = _evaluator(native)
    strided_evaluator.set_mesh(MESH)
    strided_evaluator.set_decomposition(*payload)
    strided_evaluator.prepare_host_graph(
        numbers,
        np.asarray(_host_graph("orthogonal-mixed-species", CUTOFF)[1], dtype=np.int64),
        _host_graph("orthogonal-mixed-species", CUTOFF)[0].sources,
        _host_graph("orthogonal-mixed-species", CUTOFF)[0].shifts,
        strided,
        cell,
        CUTOFF,
    )
    strided_evaluator.compute(True, True)
    reference, _, _ = _evaluator(native)
    _load(reference, tables, "orthogonal-mixed-species")
    reference.compute(True, True)
    assert strided_evaluator.result()["energy"] == reference.result()["energy"]

    with pytest.raises((ValueError, RuntimeError), match="positions|shape|rank"):
        evaluator.prepare_periodic_graph(
            numbers, positions.reshape(-1), cell, inverse_cell, CUTOFF
        )
    with pytest.raises((ValueError, RuntimeError), match="positions|shape|atom"):
        evaluator.prepare_periodic_graph(
            numbers, positions[:-1], cell, inverse_cell, CUTOFF
        )
    with pytest.raises((ValueError, RuntimeError), match="dtype|positions"):
        evaluator.prepare_periodic_graph(
            numbers, positions.astype(np.float32), cell, inverse_cell, CUTOFF
        )

    # Wrong-rank / wrong-dtype / out-of-range decomposition payloads.
    with pytest.raises((ValueError, RuntimeError), match="species|rank"):
        evaluator.set_decomposition(
            payload[0], payload[1], payload[2].reshape(-1), *payload[3:]
        )
    with pytest.raises((ValueError, RuntimeError), match="eigenvalue|dtype"):
        evaluator.set_decomposition(
            payload[0], payload[1].astype(np.float32), *payload[2:]
        )
    with pytest.raises((ValueError, RuntimeError), match="eigenvalue|finite"):
        evaluator.set_decomposition(
            payload[0], np.full_like(payload[1], np.nan), *payload[2:]
        )
    with pytest.raises((ValueError, RuntimeError), match="species|range|table"):
        evaluator.set_decomposition(np.asarray([200], dtype=np.int64), *payload[1:])
    with pytest.raises((ValueError, RuntimeError), match="species|range|table"):
        evaluator.set_decomposition(np.asarray([0], dtype=np.int64), *payload[1:])
    with pytest.raises((ValueError, RuntimeError), match="valid|bool"):
        evaluator.set_decomposition(
            payload[0],
            payload[1],
            payload[2],
            payload[3],
            payload[4].astype(np.int32),
            *payload[5:],
        )


def test_evaluator_rejects_out_of_range_graph_and_mesh_inputs(native):
    evaluator, _, tables = _evaluator(native)
    state = _load(evaluator, tables, "orthogonal-mixed-species")

    with pytest.raises((ValueError, RuntimeError), match="mesh"):
        evaluator.set_mesh((8, 8))
    with pytest.raises((ValueError, RuntimeError), match="mesh"):
        evaluator.set_mesh((0, 8, 8))
    with pytest.raises((ValueError, RuntimeError), match="cutoff"):
        evaluator.set_mesh(MESH)
        evaluator.prepare_host_graph(
            state["numbers"],
            state["offsets"],
            state["sources"],
            state["shifts"],
            state["positions"],
            state["cell"],
            -1.0,
        )
    with pytest.raises((ValueError, RuntimeError), match="offsets|receiver"):
        evaluator.prepare_host_graph(
            state["numbers"],
            state["offsets"] + 1,
            state["sources"],
            state["shifts"],
            state["positions"],
            state["cell"],
            CUTOFF,
        )
    with pytest.raises((ValueError, RuntimeError), match="sources|range|edge"):
        evaluator.prepare_host_graph(
            state["numbers"],
            state["offsets"],
            np.full_like(state["sources"], 99),
            state["shifts"],
            state["positions"],
            state["cell"],
            CUTOFF,
        )
    with pytest.raises((ValueError, RuntimeError), match="shifts|edge|shape"):
        evaluator.prepare_host_graph(
            state["numbers"],
            state["offsets"],
            state["sources"],
            state["shifts"][:-1],
            state["positions"],
            state["cell"],
            CUTOFF,
        )
    with pytest.raises((ValueError, RuntimeError), match="numbers|atomic|range"):
        evaluator.prepare_host_graph(
            np.where(state["numbers"] == 8, 95, state["numbers"]),
            state["offsets"],
            state["sources"],
            state["shifts"],
            state["positions"],
            state["cell"],
            CUTOFF,
        )


def test_evaluator_rejects_inconsistent_geometry_updates(native):
    evaluator, _, tables = _evaluator(native)
    state = _load(evaluator, tables, "orthogonal-mixed-species")

    with pytest.raises((ValueError, RuntimeError), match="inverse"):
        evaluator.update_geometry(state["positions"], state["cell"], np.zeros((3, 3)))
    with pytest.raises((ValueError, RuntimeError), match="positions|shape|rank"):
        evaluator.update_geometry(
            state["positions"].reshape(-1), state["cell"], state["inverse_cell"]
        )
    with pytest.raises((ValueError, RuntimeError), match="finite|positions"):
        evaluator.update_geometry(
            np.full_like(state["positions"], np.nan),
            state["cell"],
            state["inverse_cell"],
        )


# ---------------------------------------------------------------------------
# D. atomicity and lifecycle
# ---------------------------------------------------------------------------


def test_failed_compute_leaves_the_last_result_readable(native):
    """A failed compute must not publish a partial or empty generation."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, case)
    evaluator.compute(True, True)
    published = evaluator.result()

    # A different mesh forces the engine to reserve and re-plan on the next compute,
    # and the armed fault makes that first allocation refuse.
    evaluator.set_mesh((10, 10, 10))
    native._fourier_d3_arm_allocation_fault(0)
    with pytest.raises(RuntimeError, match="allocation"):
        evaluator.compute(True, True)
    native._fourier_d3_clear_allocation_fault()

    # The last successful generation is still readable and unchanged, and diagnostics
    # reports that it is no longer current instead of pretending otherwise.
    report = evaluator.diagnostics()
    assert report["computed"] is True
    assert report["result_available"] is False
    assert report["published_generation"] != report["input_generation"]
    after = evaluator.result()
    assert after["energy"] == published["energy"]
    _assert_force_replay_close(after["forces"], published["forces"])
    np.testing.assert_array_equal(
        after["atomic_energies"], published["atomic_energies"]
    )
    np.testing.assert_array_equal(after["stress"], published["stress"])

    # A repeated compute is exact against a fresh evaluator that never failed.
    evaluator.compute(True, True)
    recovered = evaluator.result()
    reference, _, _ = _evaluator(native)
    _load(reference, tables, case)
    reference.set_mesh((10, 10, 10))
    reference.compute(True, True)
    expected = reference.result()
    assert recovered["energy"] == expected["energy"]
    _assert_force_replay_close(recovered["forces"], expected["forces"])
    np.testing.assert_array_equal(
        recovered["atomic_energies"], expected["atomic_energies"]
    )
    np.testing.assert_array_equal(recovered["stress"], expected["stress"])


def test_failed_preparation_leaves_the_previous_geometry(native):
    """A rejected preparation must not install a partial state."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native)
    state = _load(evaluator, tables, case, native=native)
    evaluator.compute(True, True)
    published = evaluator.result()
    edges = evaluator.graph_diagnostics()["num_edges"]

    # A different build cutoff is a different topology, so preparation has to grow and
    # reload the native state; the armed fault makes the first allocation refuse.
    tighter = CUTOFF - 1.0
    graph, offsets, distances = _host_graph(case, tighter)
    native._fourier_d3_arm_allocation_fault(0)
    with pytest.raises(RuntimeError, match="allocation"):
        evaluator.prepare_host_graph(
            state["numbers"],
            offsets.astype(np.int32),
            graph.sources,
            graph.shifts,
            state["positions"],
            state["cell"],
            tighter,
        )
    native._fourier_d3_clear_allocation_fault()

    assert evaluator.graph_diagnostics()["num_edges"] == edges
    assert evaluator.result()["energy"] == published["energy"]
    np.testing.assert_array_equal(evaluator.result()["forces"], published["forces"])
    # And the runtime's own view of the graph is unchanged, so the next successful
    # preparation starts from the published state.
    evaluator.compute(True, True)
    assert evaluator.result()["energy"] == published["energy"]


@pytest.mark.parametrize("precision", PRECISIONS)
def test_refused_preparation_after_the_reservation_changes_nothing(native, precision):
    """A refusal after the engine was reserved must not advance any observable state.

    The candidate is a different topology whose coordinates are finite but whose edge
    distances overflow to infinity, so the reservation succeeds and the engine then
    refuses the geometry.  Everything a caller can read - the graph generation, the
    topology, the committed domains, the published result - must still describe the
    previous preparation, and the next unchanged-geometry compute must return exactly
    what a fresh evaluator returns.
    """

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native, precision)
    state = _load(evaluator, tables, case, native=native)
    evaluator.compute(True, True)
    published = evaluator.result()
    committed_graph = dict(evaluator.graph_diagnostics())
    committed = dict(evaluator.diagnostics())

    tighter = CUTOFF - 1.0
    graph, offsets, _ = _host_graph(case, tighter)
    assert int(offsets[-1]) != committed_graph["num_edges"]
    far = np.array(state["positions"], copy=True)
    far[0] = 1e300
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        evaluator.prepare_host_graph(
            state["numbers"],
            offsets.astype(np.int32),
            graph.sources,
            graph.shifts,
            far,
            state["cell"],
            tighter,
        )

    assert dict(evaluator.graph_diagnostics()) == committed_graph
    after = dict(evaluator.diagnostics())
    for key in (
        "mesh",
        "active_species",
        "retained_rank",
        "rank_total",
        "rank_chunk",
        "n_references",
        "input_generation",
        "published_generation",
        "computed",
        "result_available",
    ):
        assert after[key] == committed[key], key
    result = evaluator.result()
    assert result["energy"] == published["energy"]
    _assert_force_replay_close(result["forces"], published["forces"])
    np.testing.assert_array_equal(
        result["atomic_energies"], published["atomic_energies"]
    )
    np.testing.assert_array_equal(result["stress"], published["stress"])

    # The next unchanged-geometry compute is exact against an evaluator that never
    # failed, so the refused candidate left no partial state behind anywhere.
    evaluator.update_geometry(state["positions"], state["cell"], state["inverse_cell"])
    evaluator.compute(True, True)
    recovered = evaluator.result()
    fresh, _, _ = _evaluator(native, precision)
    _load(fresh, tables, case, native=native)
    fresh.compute(True, True)
    expected = fresh.result()
    assert recovered["energy"] == expected["energy"]
    _assert_force_replay_close(recovered["forces"], expected["forces"])
    np.testing.assert_array_equal(
        recovered["atomic_energies"], expected["atomic_energies"]
    )
    np.testing.assert_array_equal(recovered["stress"], expected["stress"])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_refused_same_layout_payload_is_applied_on_valid_retry(native, precision):
    """A refused candidate neither commits its payload nor loses it for the retry."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native, precision)
    state = _load(evaluator, tables, case, native=native)
    evaluator.compute(True, True)
    committed = dict(evaluator.diagnostics())
    committed_graph = dict(evaluator.graph_diagnostics())
    published = evaluator.result()

    replacement_numbers = np.asarray([6, 8, 14, 8], dtype=np.int32)
    replacement_payload = _payload_for_species(native, tables, [6, 8, 14])
    evaluator.set_decomposition(*replacement_payload)
    far = np.array(state["positions"], copy=True)
    far[0] = 1e300
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        evaluator.prepare_host_graph(
            replacement_numbers,
            state["offsets"],
            state["sources"],
            state["shifts"],
            far,
            state["cell"],
            state["build_cutoff"],
        )

    after = dict(evaluator.diagnostics())
    for key in (
        "active_species",
        "retained_rank",
        "rank_total",
        "rank_chunk",
        "n_references",
        "input_generation",
        "published_generation",
        "computed",
        "result_available",
    ):
        assert after[key] == committed[key], key
    assert dict(evaluator.graph_diagnostics()) == committed_graph
    refused_result = evaluator.result()
    assert refused_result["energy"] == published["energy"]
    np.testing.assert_array_equal(refused_result["forces"], published["forces"])

    evaluator.prepare_host_graph(
        replacement_numbers,
        state["offsets"],
        state["sources"],
        state["shifts"],
        state["positions"],
        state["cell"],
        state["build_cutoff"],
    )
    evaluator.compute(True, True)
    retried = evaluator.result()

    fresh, _, _ = _evaluator(native, precision)
    fresh.set_mesh(MESH)
    fresh.set_decomposition(*replacement_payload)
    fresh.prepare_host_graph(
        replacement_numbers,
        state["offsets"],
        state["sources"],
        state["shifts"],
        state["positions"],
        state["cell"],
        state["build_cutoff"],
    )
    fresh.compute(True, True)
    expected = fresh.result()
    assert retried["energy"] == expected["energy"]
    _assert_force_replay_close(retried["forces"], expected["forces"])
    assert evaluator.diagnostics()["active_species"] == [6, 8, 14]


def _engine_usage(native, case_name, *, precision="float64", mesh=MESH):
    """The Task 10 engine driven exactly like the evaluator, for its byte itemization.

    The same reservation arguments, the same inputs, and the same rank windows as the
    evaluator's own preparation and compute, so the two reports describe the same
    buffers and can be reconciled byte for byte rather than compared with themselves.
    """

    case = CASES[case_name]
    factors = _case_factors(case)
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int32)
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    radii = np.asarray(case["covalent_radii_angstrom"], dtype=np.float64)
    q_table = np.asarray(case["q_table_angstrom"], dtype=np.float64)
    graph, offsets, distances = _host_graph(case_name, CUTOFF)
    rank = factors.rank
    width = min(int(_configuration().chunk_size), rank)
    total = int(np.ceil(rank / width)) * width

    seam = native._FourierD3ReciprocalReverse(
        precision, int(_configuration().chunk_size)
    )
    seam.reserve(
        mesh[0],
        mesh[1],
        mesh[2],
        factors.n_species,
        numbers.size,
        int(offsets[-1]),
        factors.n_ref,
        total,
        width,
        SPLINE_ORDER,
    )
    seam.set_cell(np.ascontiguousarray(cell.reshape(-1)))
    seam.set_species(
        np.ascontiguousarray(factors.species, dtype=np.int32),
        np.ascontiguousarray(q_table, dtype=np.float64),
    )
    seam.set_atoms(np.ascontiguousarray(positions.reshape(-1)), numbers)
    seam.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    seam.set_graph(offsets, graph.sources, graph.shifts, distances, numbers, radii)
    columns = np.zeros((factors.n_species, factors.n_ref, total), dtype=np.float64)
    columns[:, :, :rank] = factors.v_q
    seam.set_reference(
        np.ascontiguousarray(factors.cn_ref, dtype=np.float64),
        np.ascontiguousarray(factors.valid, dtype=bool),
        np.ascontiguousarray(factors.species, dtype=np.int32),
        np.ascontiguousarray(columns, dtype=np.float64),
    )
    for begin in range(0, total, width):
        eigenvalues = np.zeros(width, dtype=np.float64)
        available = min(width, rank - begin)
        if available > 0:
            eigenvalues[:available] = factors.eigs[begin : begin + available]
        seam.set_rank(begin, eigenvalues)
        seam.run(CUTOFF)
    return seam.usage()


@pytest.mark.parametrize("precision", PRECISIONS)
def test_diagnostics_memory_totals_reconcile_with_the_engine_itemization(
    native, precision
):
    """The reported byte totals are the engine's own totals, each buffer counted once.

    The evaluator owns one engine, and that engine's own report is the authority: the
    reverse buffers, the forward path's buffers, the mesh allocation, the kernel tables,
    and the coordination workspace are each counted once, and the retained FFT plan's
    bytes are the mesh allocation it already holds rather than a second copy of it.
    """

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native, precision)
    previous = {"reservations": 0, "allocations": 0}

    for label, mesh in (("initial", MESH), ("grown", (12, 12, 12)), ("shrunk", MESH)):
        _load(evaluator, tables, case, native=native, mesh=mesh)
        evaluator.compute(True, True)
        report = evaluator.diagnostics()
        usage = _engine_usage(native, case, precision=precision, mesh=mesh)

        # The engine's total is exactly its itemization plus the three owners that are
        # not itemized in the shared list, so this is an independent sum, not a copy.
        active = (
            sum(buffer["active_bytes"] for buffer in usage["buffers"])
            + usage["mesh"]["active_bytes"]
            + usage["kernels"]["active_bytes"]
            + usage["coordination"]["active_bytes"]
        )
        capacity = (
            sum(buffer["capacity_bytes"] for buffer in usage["buffers"])
            + usage["mesh"]["capacity_bytes"]
            + usage["kernels"]["capacity_bytes"]
            + usage["coordination"]["capacity_bytes"]
        )
        assert active == usage["active_bytes"], label
        assert capacity == usage["capacity_bytes"], label

        assert report["active_bytes"] == active, label
        assert report["capacity_bytes"] >= report["active_bytes"], label
        if label == "initial":
            # A fresh reservation holds exactly the committed extents.
            assert report["capacity_bytes"] == capacity, label
        else:
            # A grow-only reservation never shrinks, so retention is at least what a
            # fresh reservation of the same shape would hold.
            assert report["capacity_bytes"] >= capacity, label
        # Reservations and allocations are per-owner counters, so one composite
        # reservation is one increment in each of the three owners - and the step's
        # increment is what a single fresh reservation of the same shape reports.
        assert report["reservations"] - previous["reservations"] == (
            usage["reservations"]
            + usage["forward_reservations"]
            + usage["coordination"]["reservations"]
        ), label
        if label == "initial":
            assert report["allocations"] - previous["allocations"] == (
                usage["allocations"]
                + usage["forward_allocations"]
                + usage["coordination"]["allocations"]
            ), label
        else:
            # A grow-only reservation reuses the buffers it already holds, so a later
            # step allocates strictly fewer than a fresh reservation of the same shape.
            assert report["allocations"] > previous["allocations"], label
        previous = {
            "reservations": report["reservations"],
            "allocations": report["allocations"],
        }
        # The plan retains the storage's own allocation token, so its bytes are the
        # mesh capacity that is already counted, not a second copy of it.
        assert usage["plan_retained_bytes"] == usage["mesh"]["capacity_bytes"], label


def test_evaluator_releases_its_kokkos_owners_before_finalization(native):
    """A live evaluator must block finalization, and a dropped one must not."""

    gc.collect()
    baseline = native._kokkos_live_object_count()
    evaluator, _, tables = _evaluator(native)
    _load(evaluator, tables, "orthogonal-mixed-species")
    evaluator.compute(True, True)
    assert native._kokkos_live_object_count() > baseline
    with pytest.raises(RuntimeError, match="finalize"):
        native._finalize_kokkos()
    del evaluator
    gc.collect()
    assert native._kokkos_live_object_count() == baseline


# ---------------------------------------------------------------------------
# E. the Python runtime and the correction pipeline
# ---------------------------------------------------------------------------


def _pipeline(native, dtype=np.float64, **overrides):
    return CorrectionPipeline([_configuration(**overrides)], native=native, dtype=dtype)


@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_pipeline_matches_the_task10_engine(native, monkeypatch, case):
    """The runtime path returns the same four properties as the Task 10 engine."""

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    pipeline = _pipeline(native)
    atoms = _atoms(case)
    result = pipeline.calculate(atoms, ["energy", "energies", "forces", "stress"])
    reference = _reference(native, case)
    tolerance = TOLERANCE["float64"]
    approx_tolerance = APPROX["float64"]

    assert result["energy"] == pytest.approx(reference["energy"], **approx_tolerance)
    np.testing.assert_allclose(
        result["energies"], reference["atomic_energies"], **tolerance
    )
    np.testing.assert_allclose(result["forces"], reference["forces"], **tolerance)
    np.testing.assert_allclose(result["stress"], reference["stress"], **tolerance)
    assert result["forces"].dtype == np.float64
    assert result["energies"].dtype == np.float64


def test_pipeline_matches_the_frozen_oracle_energy(native, monkeypatch):
    """The connected path reproduces the frozen mesh-free oracle energy."""

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    case = "orthogonal-mixed-species"
    pipeline = _pipeline(native, mesh=ACCEPTANCE_MESH)
    result = pipeline.calculate(_atoms(case), ["energy"])

    assert result["energy"] == pytest.approx(
        CASES[case]["expected"]["energy_ev"], abs=DECLARED_ORACLE_ENERGY_BAND_EV
    )


def test_pipeline_reuses_the_graph_and_its_decomposition(native, monkeypatch):
    """An unchanged geometry reuses the graph and the decomposition payload."""

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    pipeline = _pipeline(native, skin=0.5)
    atoms = _atoms("orthogonal-mixed-species")

    first = pipeline.calculate(atoms, ["energy", "forces"])
    second = pipeline.calculate(atoms, ["energy", "forces"])
    report = pipeline.diagnostics()[0]

    assert second["energy"] == first["energy"]
    _assert_force_replay_close(second["forces"], first["forces"])
    assert report["graph_build_count"] == 1
    assert report["graph_reuse_count"] == 1
    assert report["native_allocation_count"] == 1
    assert report["decomposition_invalidation_count"] == 1
    assert report["mesh_invalidation_count"] == 1


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pipeline_same_layout_species_replacement_matches_fresh(
    native, monkeypatch, dtype
):
    """Pipeline reuse applies a changed species payload with unchanged dimensions."""

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    original = _atoms("orthogonal-mixed-species")
    replacement = Atoms(
        numbers=np.asarray([6, 8, 14, 8], dtype=np.int64),
        positions=original.positions,
        cell=original.cell,
        pbc=True,
    )

    reused_pipeline = _pipeline(native, dtype=dtype)
    reused_pipeline.calculate(original, ["energy", "forces"])
    actual = reused_pipeline.calculate(replacement, ["energy", "forces"])
    expected = _pipeline(native, dtype=dtype).calculate(
        replacement, ["energy", "forces"]
    )

    precision = "float32" if dtype is np.float32 else "float64"
    assert actual["energy"] == pytest.approx(expected["energy"], **APPROX[precision])
    np.testing.assert_allclose(
        actual["forces"], expected["forces"], **TOLERANCE[precision]
    )


def test_pipeline_allocation_stays_lazy_after_the_geometry_gate(native):
    """No native evaluator exists before a usable geometry is presented."""

    runtime = build_correction_runtime(
        _configuration(), native=native, dtype=np.float64
    )
    assert runtime.evaluator is None
    assert runtime.diagnostics()["native_allocation_count"] == 0

    with pytest.raises(RuntimeError, match="fully periodic"):
        runtime.calculate(
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]), ["energy"]
        )
    assert runtime.evaluator is None
    assert runtime.diagnostics()["native_allocation_count"] == 0


def _raise_graph_build(*_args, **_kwargs):
    raise RuntimeError("injected host graph failure")


def test_failed_graph_build_keeps_the_committed_domains(native, monkeypatch):
    """A failed graph build leaves the runtime and the evaluator agreeing.

    The mesh and the active-species payload are prospective until the graph they belong
    to is prepared, so a refusal must leave the committed keys, the counters, the native
    domains, the generations, and the published result exactly as they were, and the
    retry must produce what a runtime that never failed produces.
    """

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    runtime = build_correction_runtime(
        _configuration(mesh=None), native=native, dtype=np.float64
    )
    first = _atoms("orthogonal-mixed-species")
    runtime.calculate(first, ["energy", "forces"])
    committed = dict(runtime.diagnostics())
    committed_key = runtime.decomposition_key
    committed_native = dict(runtime.evaluator.diagnostics())
    committed_graph = dict(runtime.evaluator.graph_diagnostics())
    committed_result = runtime.evaluator.result(["energy", "forces"])

    # A different automatic mesh and a different active species, so both prospective
    # domains really are different from the committed ones.
    changed = first.copy()
    changed.set_cell(
        np.asarray(first.cell.array, dtype=np.float64) * 1.5, scale_atoms=True
    )
    changed.numbers = np.array([1, 6, 7, 8], dtype=np.int64)
    prospective = runtime._resolve_domains(
        np.asarray(changed.numbers, dtype=np.int64),
        np.asarray(changed.cell.array, dtype=np.float64),
    )
    assert prospective.mesh != committed["resolved_mesh"]
    assert prospective.decomposition[0] != committed_key[0]

    original = corrections.build_host_neighbor_graph
    monkeypatch.setattr(corrections, "build_host_neighbor_graph", _raise_graph_build)
    with pytest.raises(RuntimeError, match="injected host graph failure"):
        runtime.calculate(changed, ["energy"])
    monkeypatch.setattr(corrections, "build_host_neighbor_graph", original)

    after = dict(runtime.diagnostics())
    for key in (
        "resolved_mesh",
        "mesh_invalidation_count",
        "decomposition_invalidation_count",
        "reciprocal_kernel_invalidation_count",
        "fft_plan_invalidation_count",
        "native_allocation_count",
        "graph_build_count",
        "graph_reuse_count",
        "geometry_update_count",
        "num_edges",
        "graph_backend",
    ):
        assert after[key] == committed[key], key
    assert runtime.decomposition_key == committed_key

    native_after = dict(runtime.evaluator.diagnostics())
    for key in (
        "mesh",
        "active_species",
        "retained_rank",
        "rank_total",
        "rank_chunk",
        "n_references",
        "decomposition_max_relative_error",
        "decomposition_tolerance_met",
        "input_generation",
        "published_generation",
        "computed",
        "result_available",
    ):
        assert native_after[key] == committed_native[key], key
    assert dict(runtime.evaluator.graph_diagnostics()) == committed_graph
    result_after = runtime.evaluator.result(["energy", "forces"])
    assert result_after["energy"] == committed_result["energy"]
    np.testing.assert_array_equal(result_after["forces"], committed_result["forces"])

    # The retry commits the prospective domains and matches a runtime that never failed.
    retried = runtime.calculate(changed, ["energy", "forces"])
    fresh = build_correction_runtime(
        _configuration(mesh=None), native=native, dtype=np.float64
    )
    expected = fresh.calculate(changed, ["energy", "forces"])
    assert retried["energy"] == pytest.approx(expected["energy"], **APPROX["float64"])
    np.testing.assert_allclose(
        retried["forces"], expected["forces"], **TOLERANCE["float64"]
    )
    assert runtime.diagnostics()["resolved_mesh"] == prospective.mesh
    assert runtime.decomposition_key[0] == prospective.decomposition[0]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pipeline_publishes_exact_zeros_without_active_species(
    native, monkeypatch, dtype
):
    """Empty and all-padding periodic systems are exact zero-energy paths."""

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    cell = np.asarray(
        CASES["orthogonal-mixed-species"]["cell_angstrom"], dtype=np.float64
    )
    systems = (
        Atoms(
            numbers=np.empty(0, dtype=np.int64),
            positions=np.empty((0, 3)),
            cell=cell,
            pbc=True,
        ),
        Atoms(
            numbers=np.zeros(2, dtype=np.int64),
            positions=[[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]],
            cell=cell,
            pbc=True,
        ),
    )
    pipeline = _pipeline(native, dtype=dtype)

    for atoms in systems:
        count = len(atoms)
        result = pipeline.calculate(atoms, ["energy", "energies", "forces", "stress"])
        assert result["energy"] == 0.0
        assert result["energies"].shape == (count,)
        assert result["forces"].shape == (count, 3)
        assert result["forces"].dtype == np.float64
        np.testing.assert_array_equal(result["energies"], np.zeros(count))
        np.testing.assert_array_equal(result["forces"], np.zeros((count, 3)))
        np.testing.assert_array_equal(result["stress"], np.zeros(6))
        runtime = pipeline._runtimes[0]
        assert runtime.evaluator.diagnostics()["active_species"] == []
        assert runtime.evaluator.diagnostics()["engine_loaded"] is False


def test_pipeline_float32_runtime_uses_the_float32_evaluator(native, monkeypatch):
    """The dtype-specific class is the one the factory resolves, for both precisions."""

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    case = "orthogonal-mixed-species"
    atoms = _atoms(case)
    properties = ["energy", "energies", "forces", "stress"]
    double_pipeline = _pipeline(native, dtype=np.float64)
    single_pipeline = _pipeline(native, dtype=np.float32)
    double = double_pipeline.calculate(atoms, properties)
    single = single_pipeline.calculate(atoms, properties)

    # The runtime resolves the dtype-specific native class, not a promoted copy of the
    # other precision, and the pipeline still hands the caller float64.
    assert type(double_pipeline._runtimes[0].evaluator).__name__ == "FourierD3Kokkos"
    assert (
        type(single_pipeline._runtimes[0].evaluator).__name__ == "FourierD3KokkosFloat"
    )
    assert single["forces"].dtype == np.float64
    assert single["energies"].dtype == np.float64
    assert single["energy"] == pytest.approx(double["energy"], rel=1e-4)
    np.testing.assert_allclose(single["forces"], double["forces"], rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(
        single["energies"], double["energies"], rtol=5e-3, atol=1e-5
    )
    np.testing.assert_allclose(single["stress"], double["stress"], rtol=5e-3, atol=1e-5)
    assert sum(single["energies"]) == pytest.approx(single["energy"], rel=1e-5)


def test_fourier_d3_runtime_fails_without_the_native_evaluator():
    """A build without the classes produces the no-fallback error, never D3Kokkos."""

    class _WithoutFourierD3:
        D2Kokkos = object
        D3Kokkos = object

    with pytest.raises(RuntimeError) as error:
        build_correction_runtime(
            _configuration(), native=_WithoutFourierD3(), dtype=np.float64
        )
    message = str(error.value)
    assert "FourierD3Kokkos" in message
    assert "not available in this build" in message
    assert "real-space" in message


# ---------------------------------------------------------------------------
# F. the ordinary corrections are untouched
# ---------------------------------------------------------------------------


def test_ordinary_d2_d3_factory_behavior_is_unchanged(native):
    """D2/D3 keep their eager evaluators, their classes, and their graph semantics."""

    d2 = build_correction_runtime(
        corrections.D2Correction(cutoff=4.0), native=native, dtype=np.float64
    )
    d3 = build_correction_runtime(
        corrections.D3Correction(cutoff=4.0, coordination_cutoff=4.0),
        native=native,
        dtype=np.float64,
    )
    d2_single = build_correction_runtime(
        corrections.D2Correction(cutoff=4.0), native=native, dtype=np.float32
    )
    d3_single = build_correction_runtime(
        corrections.D3Correction(cutoff=4.0, coordination_cutoff=4.0),
        native=native,
        dtype=np.float32,
    )

    assert type(d2.evaluator) is native.D2Kokkos
    assert type(d3.evaluator) is native.D3Kokkos
    assert type(d2_single.evaluator) is native.D2KokkosFloat
    assert type(d3_single.evaluator) is native.D3KokkosFloat
    assert d2.graph_requirement.topology != "global"
    assert d3.graph_requirement.topology != "global"
    assert d2.graph_requirement.pair_cutoff == pytest.approx(4.0)
    assert d3.graph_requirement.pair_cutoff == pytest.approx(4.0)
    assert d3.graph_requirement.coordination_cutoff == pytest.approx(4.0)


def test_ordinary_d2_d3_energies_are_unchanged(native, monkeypatch):
    """The shared correction path still produces the ordinary D2/D3 energies."""

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    for configuration, expected in (
        (corrections.D2Correction(cutoff=6.0), None),
        (corrections.D3Correction(cutoff=6.0, coordination_cutoff=6.0), None),
    ):
        pipeline = CorrectionPipeline([configuration], native=native, dtype=np.float64)
        first = pipeline.calculate(atoms, ["energy", "forces"])
        second = pipeline.calculate(atoms, ["energy", "forces"])
        assert np.isfinite(first["energy"])
        assert first["energy"] < 0.0
        assert second["energy"] == first["energy"]
        np.testing.assert_array_equal(second["forces"], first["forces"])
        if expected is not None:  # pragma: no cover - reserved for a future pin
            assert first["energy"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# helpers used above
# ---------------------------------------------------------------------------


def _reference_displaced(
    native, case_name, positions, *, cell=None, precision="float64"
):
    """The Task 10 engine on displaced positions or a strained cell.

    The topology and its shifts stay those of the reference geometry, which is what the
    evaluator's ``update_geometry`` contract preserves.
    """

    case = CASES[case_name]
    factors = _case_factors(case)
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int32)
    reference_cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    current_cell = (
        reference_cell if cell is None else np.asarray(cell, dtype=np.float64)
    )
    radii = np.asarray(case["covalent_radii_angstrom"], dtype=np.float64)
    q_table = np.asarray(case["q_table_angstrom"], dtype=np.float64)
    graph, offsets, _ = _host_graph(case_name, CUTOFF)
    # The topology and its shifts stay those of the reference geometry, but the distances
    # follow the geometry being evaluated: that is exactly the contract
    # ``update_geometry`` preserves.
    displaced = np.asarray(positions, dtype=np.float64)
    delta = (
        displaced[graph.sources]
        - displaced[graph.receivers]
        + graph.shifts @ current_cell
    )
    distances = np.linalg.norm(delta, axis=1)

    seam = native._FourierD3ReciprocalReverse(precision, factors.rank)
    seam.reserve(
        MESH[0],
        MESH[1],
        MESH[2],
        factors.n_species,
        numbers.size,
        int(offsets[-1]),
        factors.n_ref,
        factors.rank,
        factors.rank,
        SPLINE_ORDER,
    )
    seam.set_cell(np.ascontiguousarray(current_cell.reshape(-1)))
    seam.set_species(
        np.ascontiguousarray(factors.species, dtype=np.int32),
        np.ascontiguousarray(q_table, dtype=np.float64),
    )
    seam.set_atoms(np.ascontiguousarray(positions.reshape(-1)), numbers)
    seam.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    seam.set_rank(0, np.ascontiguousarray(factors.eigs, dtype=np.float64))
    seam.set_graph(offsets, graph.sources, graph.shifts, distances, numbers, radii)
    seam.set_reference(
        np.ascontiguousarray(factors.cn_ref, dtype=np.float64),
        np.ascontiguousarray(factors.valid, dtype=bool),
        np.ascontiguousarray(factors.species, dtype=np.int32),
        np.ascontiguousarray(factors.v_q, dtype=np.float64),
    )
    outcome = seam.run(CUTOFF)
    return {
        "energy": float(outcome["energy"]),
        "atomic_energies": np.asarray(seam.atomic_energies(), dtype=np.float64),
        "forces": np.asarray(seam.forces(), dtype=np.float64).reshape(-1, 3),
        "stress": np.asarray(seam.stress(), dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# F. the Task 12 diagnostic contract
#
# One diagnostic record has to answer, without a caller guessing: which capability is
# active, which accuracy controls were realized, why each domain rebuilt or reused, how
# much memory is active versus retained, and whether the published results are current.
# Diagnostics are state, so every read below is required to be a pure read.
# ---------------------------------------------------------------------------

STAGE_NAMES = (
    "preparation",
    "forward_preparation",
    "coordination",
    "assignment_forward_fft",
    "reciprocal_contraction",
    "inverse_fft_gather_adjoint",
    "virial_and_strain",
    "result_reduction",
    "result_export",
)
MEMORY_GROUPS = {"reverse", "forward", "mesh", "kernel", "coordination"}


def _loaded(
    native,
    case="orthogonal-mixed-species",
    *,
    precision="float64",
    mesh=MESH,
    **payload,
):
    """A loaded and computed evaluator, its configuration, and its tables."""

    evaluator, configuration, tables = _evaluator(native, precision)
    _load(evaluator, tables, case, native=native, mesh=mesh, **payload)
    evaluator.compute(True, True)
    return evaluator, configuration, tables


@pytest.mark.parametrize("precision", PRECISIONS)
def test_diagnostics_memory_components_reconcile_and_name_every_owner(
    native, precision
):
    """The itemization is the authority, and it sums to the reported totals exactly."""

    evaluator, _, _ = _loaded(native, precision=precision)
    report = evaluator.diagnostics()

    components = report["memory_components"]
    assert components
    names = [component["name"] for component in components]
    assert len(set(names)) == len(names)
    assert {name.split(".", 1)[0] for name in names} == MEMORY_GROUPS
    assert sum(c["active_bytes"] for c in components) == report["active_bytes"]
    assert sum(c["capacity_bytes"] for c in components) == report["capacity_bytes"]
    assert all(c["capacity_bytes"] >= c["active_bytes"] for c in components)
    # The mesh is itemized as its two retained allocations, which is what a plan holds.
    assert "mesh.real_mesh" in names
    assert "mesh.spectrum" in names
    assert any(name.startswith("kernel.") for name in names)
    assert any(name.startswith("coordination.") for name in names)
    assert report["capacity_bytes"] >= report["active_bytes"] > 0


@pytest.mark.parametrize("precision", PRECISIONS)
def test_diagnostics_plan_retained_bytes_follow_the_held_allocation(native, precision):
    """A plan reports the allocation it owns, counted once per allocation."""

    evaluator, _, _ = _loaded(native, precision=precision)
    report = evaluator.diagnostics()

    plans = {plan["role"]: plan for plan in report["fft_plans"]}
    assert set(plans) == {"forward_r2c", "inverse_c2r"}
    assert all(plan["held"] for plan in plans.values())
    assert all(plan["allocation"]["owns"] for plan in plans.values())

    # Both roles retain the same committed allocation, so the retained bytes are that
    # allocation's bytes counted once - not a second copy of the mesh, and not the sum
    # of two plans over the same buffers.
    generations = {plan["allocation"]["generation"] for plan in plans.values()}
    assert len(generations) == 1
    retained = plans["forward_r2c"]["allocation"]["retained_bytes"]
    assert retained > 0
    assert report["plan_retained_bytes"] == retained
    assert plans["inverse_c2r"]["allocation"]["retained_bytes"] == retained


@pytest.mark.parametrize("precision", PRECISIONS)
def test_diagnostics_count_a_superseded_allocation_retained_by_plans(native, precision):
    """A mesh generation kept alive only by plans remains in the byte totals."""

    evaluator, _, tables = _loaded(native, precision=precision)
    initial = evaluator.diagnostics()
    retained = initial["plan_retained_bytes"]

    _load(
        evaluator,
        tables,
        "orthogonal-mixed-species",
        native=native,
        mesh=(16, 16, 16),
    )
    grown = evaluator.diagnostics()
    plans = {plan["role"]: plan for plan in grown["fft_plans"]}
    plan_generation = plans["forward_r2c"]["allocation"]["generation"]
    assert plans["inverse_c2r"]["allocation"]["generation"] == plan_generation
    assert grown["plan_retained_bytes"] == retained

    components = grown["memory_components"]
    superseded = [
        component
        for component in components
        if component["name"].startswith("mesh.retained_superseded")
    ]
    assert sum(component["capacity_bytes"] for component in superseded) == retained
    assert sum(component["active_bytes"] for component in superseded) == retained
    assert (
        sum(component["capacity_bytes"] for component in components)
        == grown["capacity_bytes"]
    )
    assert (
        sum(component["active_bytes"] for component in components)
        == grown["active_bytes"]
    )


def test_diagnostics_report_the_actual_spacing_of_the_retained_cell(native):
    """The realized mesh keeps its dimensions; its actual spacing follows the cell."""

    evaluator, _, tables = _loaded(native)
    report = evaluator.diagnostics()
    case = CASES["orthogonal-mixed-species"]
    lengths = np.linalg.norm(
        np.asarray(case["cell_angstrom"], dtype=np.float64), axis=1
    )
    assert report["mesh"] == list(MESH)
    assert report["mesh_points"] == int(np.prod(MESH))
    # The spacing is measured on the retained cell, axis by axis, so an anisotropic
    # cell reports three different spacings for one isotropic mesh.
    assert report["mesh_spacing"] == pytest.approx(
        [length / size for length, size in zip(lengths, MESH)], rel=1e-12
    )
    invalidation = report["mesh_invalidation_count"]

    # A cell update that keeps the mesh changes the actual spacing and the reciprocal
    # kernel, but not the mesh domain.
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = np.diag([9.0, 8.0, 8.0])
    evaluator.update_geometry(
        positions, cell, np.ascontiguousarray(np.linalg.inv(cell))
    )
    evaluator.compute(True, True)
    updated = evaluator.diagnostics()

    assert updated["mesh"] == list(MESH)
    assert updated["mesh_spacing"] == pytest.approx(
        [9.0 / MESH[0], 8.0 / MESH[1], 8.0 / MESH[2]], rel=1e-12
    )
    assert updated["mesh_invalidation_count"] == invalidation


def test_diagnostics_report_the_realized_decomposition_state(native):
    """Retained rank, chunking, achieved error, and the tolerance verdict are named."""

    evaluator, configuration, _ = _loaded(native)
    report = evaluator.diagnostics()

    assert report["retained_rank"] >= 1
    assert report["rank_total"] >= report["retained_rank"]
    assert report["rank_chunk"] == min(
        int(configuration.chunk_size), report["retained_rank"]
    )
    assert report["rank_chunk_count"] == int(
        np.ceil(report["rank_total"] / report["rank_chunk"])
    )
    assert report["n_references"] == 5
    assert report["decomposition_tolerance_met"] is True
    assert (
        report["decomposition_max_relative_error"]
        <= PARAMETERS["decomposition_tolerance"]
    )


def test_diagnostics_report_an_unmet_decomposition_tolerance(native):
    """An unmet tolerance is reported with the error it achieved, never relaxed."""

    evaluator, _, tables = _loaded(native, tolerance=1e-12, rank_cap=1)
    report = evaluator.diagnostics()

    assert report["retained_rank"] == 1
    assert report["rank_total"] == 1
    assert report["rank_chunk_count"] == 1
    assert report["decomposition_tolerance_met"] is False
    assert report["decomposition_max_relative_error"] > 1e-12


def test_diagnostics_keep_padding_atoms_out_of_the_active_species(native):
    """Padding atoms are absent from the active species and from the mesh channels."""

    case = CASES["orthogonal-mixed-species"]
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int64)
    padded = np.concatenate([numbers, np.zeros(3, dtype=np.int64)]).astype(np.int64)
    positions = np.concatenate(
        [
            np.asarray(case["positions_angstrom"], dtype=np.float64),
            np.full((3, 3), 1e3),
        ]
    )
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    graph = build_host_neighbor_graph(
        Atoms(numbers=padded, positions=positions, cell=cell, pbc=True), CUTOFF
    )
    offsets = np.empty(len(padded) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(graph.num_neigh, out=offsets[1:])

    evaluator, _, tables = _evaluator(native)
    evaluator.set_mesh(MESH)
    evaluator.set_decomposition(*_payload(native, tables, "orthogonal-mixed-species"))
    evaluator.prepare_host_graph(
        padded,
        offsets.astype(np.int32),
        graph.sources,
        graph.shifts,
        positions,
        cell,
        CUTOFF,
    )
    evaluator.compute(True, True)
    report = evaluator.diagnostics()

    assert report["active_species"] == [1, 6, 8]
    assert 0 not in report["active_species"]
    assert report["mesh_points"] == int(np.prod(MESH))
    # Padding atoms are real nodes in the topology, but they hold no channel.
    assert evaluator.graph_diagnostics()["num_nodes"] == len(padded)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_diagnostics_report_cache_state_and_result_currency(native, precision):
    """Every rebuild/reuse domain and the currency of the published result are named."""

    evaluator, _, _ = _loaded(native, precision=precision)
    first = evaluator.diagnostics()

    assert first["computed"] is True
    assert first["result_available"] is True
    assert first["forces_available"] is True
    assert first["stress_available"] is True
    assert first["engine_loaded"] is True
    assert first["stencil_points"] >= 1
    assert first["input_generation"] == first["published_generation"]
    assert first["kernel_key_misses"] >= 1

    # An unchanged recompute reuses everything: no new plan, no new kernel miss, no new
    # stencil, and the same generations.
    evaluator.compute(True, True)
    second = evaluator.diagnostics()
    assert second["input_generation"] == first["input_generation"]
    assert second["published_generation"] == first["published_generation"]
    assert second["plan_builds"] == first["plan_builds"]
    assert second["kernel_key_misses"] == first["kernel_key_misses"]
    assert second["stencil_points"] == first["stencil_points"]

    # A mesh change rebuilds the plans for both roles and grows the kernel tables, but
    # it does not change the kernel key: the key covers the cell, the damping, the
    # species, and the spline order, and the tables are refilled because their shape
    # changed.  Reporting that refill as a key miss would name the wrong cause.
    evaluator.set_mesh((12, 12, 12))
    evaluator.compute(True, True)
    third = evaluator.diagnostics()
    assert third["plan_builds"] > second["plan_builds"]
    assert third["kernel_key_hits"] > second["kernel_key_hits"]
    assert third["kernel_key_misses"] == second["kernel_key_misses"]
    assert third["capacity_bytes"] > second["capacity_bytes"]
    assert third["input_generation"] > second["input_generation"]

    # A cell change with an unchanged mesh is a different kernel key, so that one
    # misses and is attributed to the key change.
    case = CASES["orthogonal-mixed-species"]
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = np.diag([9.0, 12.0, 12.0])
    evaluator.update_geometry(
        positions, cell, np.ascontiguousarray(np.linalg.inv(cell))
    )
    evaluator.compute(True, True)
    fourth = evaluator.diagnostics()
    assert fourth["kernel_key_misses"] > third["kernel_key_misses"]
    assert fourth["kernel_key_rebuilds"] > third["kernel_key_rebuilds"]


def test_diagnostics_plan_counters_are_reported_per_role(native):
    evaluator, _, _ = _loaded(native)
    report = evaluator.diagnostics()

    plans = {plan["role"]: plan for plan in report["fft_plans"]}
    for role in ("forward_r2c", "inverse_c2r"):
        plan = plans[role]
        assert plan["key"]
        assert plan["builds"] >= 1
        assert plan["reuses"] >= 0
        assert plan["releases"] >= 0
        assert plan["key_rebuilds"] >= 0
        assert plan["identity_rebuilds"] >= 0
        assert plan["allocation_generation"] >= 0
    # The aggregate is exactly the per-role sum.
    assert report["plan_builds"] == sum(plan["builds"] for plan in plans.values())
    assert report["plan_reuses"] == sum(plan["reuses"] for plan in plans.values())
    assert report["plan_releases"] == sum(plan["releases"] for plan in plans.values())
    assert report["plan_key_rebuilds"] == sum(
        plan["key_rebuilds"] for plan in plans.values()
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reading_diagnostics_is_a_pure_read(native, precision):
    """A diagnostic read neither mutates state nor makes a stale result current."""

    evaluator, _, _ = _loaded(native, precision=precision)
    before = evaluator.diagnostics()
    before_result = evaluator.result()
    for _ in range(3):
        assert evaluator.diagnostics() == before
    after_result = evaluator.result()
    assert after_result["energy"] == before_result["energy"]
    np.testing.assert_array_equal(after_result["forces"], before_result["forces"])

    # And a stale result stays stale across reads.  A committed geometry update advances
    # the input generation and publishes nothing, so the last result is no longer
    # current; a merely staged input would not, which is what keeps a refused
    # preparation from moving any reported field.
    case = CASES["orthogonal-mixed-species"]
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64).copy()
    positions[0, 0] += 0.05
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    evaluator.update_geometry(
        positions, cell, np.ascontiguousarray(np.linalg.inv(cell))
    )
    stale = evaluator.diagnostics()
    assert stale["result_available"] is False
    assert evaluator.diagnostics()["result_available"] is False
    assert (
        evaluator.diagnostics()["published_generation"] == stale["published_generation"]
    )


def test_failed_preparation_reports_no_committed_work_in_diagnostics(native):
    """A refusal is visible only through state that stayed coherent, never as work."""

    case = "orthogonal-mixed-species"
    evaluator, _, tables = _evaluator(native)
    state = _load(evaluator, tables, case, native=native)
    evaluator.compute(True, True)
    committed = dict(evaluator.diagnostics())

    # A different build cutoff is a different topology, so the preparation has to
    # grow and reload the native state; the armed fault makes it refuse.
    tighter = CUTOFF - 1.0
    graph, offsets, _ = _host_graph(case, tighter)
    native._fourier_d3_arm_allocation_fault(0)
    with pytest.raises(RuntimeError, match="allocation"):
        evaluator.prepare_host_graph(
            state["numbers"],
            offsets.astype(np.int32),
            graph.sources,
            graph.shifts,
            state["positions"],
            state["cell"],
            tighter,
        )
    native._fourier_d3_clear_allocation_fault()

    assert evaluator.diagnostics() == committed


# ---------------------------------------------------------------------------
# G. opt-in synchronized stage timing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_stage_timing_is_unavailable_until_it_is_enabled(native, precision):
    evaluator, _, _ = _loaded(native, precision=precision)

    assert evaluator.timing() is None
    report = evaluator.diagnostics()
    assert report["timing_enabled"] is False
    assert report["stage_timing"] is None


@pytest.mark.parametrize("precision", PRECISIONS)
def test_stage_timing_reports_seconds_last_cumulative_and_count(native, precision):
    evaluator, _, _ = _loaded(native, precision=precision)
    evaluator.set_timing_enabled(True)
    evaluator.compute(True, True)
    timing = evaluator.timing()

    assert set(timing) == set(STAGE_NAMES)
    for name, stage in timing.items():
        assert set(stage) == {"last_seconds", "cumulative_seconds", "count"}, name
        assert stage["count"] >= 1, name
        assert stage["last_seconds"] >= 0.0, name
        assert stage["cumulative_seconds"] >= stage["last_seconds"] >= 0.0, name
    # The stages that must run exactly once per compute do.
    for name in ("preparation", "result_export"):
        assert timing[name]["count"] == 1, name

    # Accumulation: a second compute advances every count and never decreases a total.
    evaluator.compute(True, True)
    second = evaluator.timing()
    for name in STAGE_NAMES:
        assert second[name]["count"] > timing[name]["count"], name
        assert (
            second[name]["cumulative_seconds"] >= timing[name]["cumulative_seconds"]
        ), name
    assert second["result_export"]["count"] == 2


@pytest.mark.parametrize("precision", PRECISIONS)
def test_stage_timing_does_not_change_the_published_result(native, precision):
    evaluator, _, _ = _loaded(native, precision=precision)
    plain = evaluator.result()

    evaluator.set_timing_enabled(True)
    evaluator.compute(True, True)
    timed = evaluator.result()

    assert timed["energy"] == plain["energy"]
    _assert_force_replay_close(timed["forces"], plain["forces"])
    np.testing.assert_array_equal(timed["atomic_energies"], plain["atomic_energies"])
    np.testing.assert_array_equal(timed["stress"], plain["stress"])

    # Disabling it again makes the record unavailable rather than zero-valued.
    evaluator.set_timing_enabled(False)
    assert evaluator.timing() is None
    assert evaluator.diagnostics()["stage_timing"] is None
    evaluator.compute(True, True)
    assert evaluator.result()["energy"] == plain["energy"]


def test_diagnostics_expose_the_stage_timing_record(native):
    evaluator, _, _ = _loaded(native)
    evaluator.set_timing_enabled(True)
    evaluator.compute(True, True)
    report = evaluator.diagnostics()

    assert report["timing_enabled"] is True
    assert report["stage_timing"] == evaluator.timing()
