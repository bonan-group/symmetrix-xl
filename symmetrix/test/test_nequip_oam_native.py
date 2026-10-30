from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pytest


FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"
#: Exported OAM-S contract to parse.  This used to be the ambient
#: ``/tmp/oam-s-v2.json``, which meant the test could run against any stale
#: export; an explicit variable plus the identity check below keeps it honest.
EXPORTED_ENVIRONMENT = "SYMMETRIX_NEQUIP_OAM_S_CONTRACT"
EXPORTED = os.environ.get(EXPORTED_ENVIRONMENT)


def _exported_contract() -> dict:
    """Load the configured export and check that it is what the test claims."""

    path = Path(EXPORTED)
    if not path.is_file():
        pytest.skip(f"{EXPORTED_ENVIRONMENT} does not exist: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("architecture") != "oam":
        pytest.skip(f"{path} is not an exported OAM contract")
    if contract.get("symmetrix_nequip_format_version") != 2:
        pytest.skip(f"{path} is not an OAM format version 2 contract")
    layers = contract.get("layers", ())
    trainable = [
        layer
        for layer in layers
        if isinstance(layer.get("self_connection"), dict)
        and "irreps_in1" in layer["self_connection"]
    ]
    if not trainable:
        pytest.skip(f"{path} has no tensor-product self connection to parse")
    modes = {
        str(instruction.get("connection_mode"))
        for layer in trainable
        for instruction in layer["self_connection"].get("instructions", ())
    }
    if "uvw" not in modes:
        pytest.skip(f"{path} self connections are not uvw: {sorted(modes)}")
    return contract


def _native():
    try:
        from symmetrix import symmetrix as native
    except Exception as error:
        pytest.skip(f"native Symmetrix package is unavailable: {error}")
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("OAM Kokkos evaluator is unavailable in this extension")
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    return native


def _reference(native, path):
    """The artifact-free reference routing of the retained OAM evaluator.

    ``NequIPOAMKokkos(path)`` with no artifact argument is the portable Kokkos
    implementation in the selected execution space, selected explicitly by
    the native ``reference`` mode.
    """

    evaluator = native.NequIPOAMKokkos(str(path))
    evaluator.set_streamed_edges("reference")
    return evaluator


def _compute(evaluator, *, num_neigh=(0, 0), indices=(), neigh_types=(), xyz=(), r=()):
    evaluator.compute_node_energies_forces(
        2,
        np.asarray((0, 1), dtype=np.int32),
        np.asarray(num_neigh, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(neigh_types, dtype=np.int32),
        np.asarray(xyz, dtype=float),
        np.asarray(r, dtype=float),
    )


def _nonzero_contract(tmp_path):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["node_attributes"]["values"][0] = 1.0
    data["node_attributes"]["values"][1] = 0.7
    data["readout"]["weight"]["values"][0] = 1.0
    data["zbl"]["cutoff"] = 4.5
    data["zbl"]["half_energy_prefactor"] = 1.0e-30
    path = tmp_path / "oam-force.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.oam_runtime
def test_minimal_oam_forward_has_no_edges():
    native = _native()
    evaluator = _reference(native, FIXTURE)
    _compute(evaluator)
    assert np.asarray(evaluator.node_energies) == pytest.approx([0.0, 0.0])
    assert np.asarray(evaluator.node_forces).size == 0
    assert "active_edges=0" in evaluator.last_execution_diagnostic


@pytest.mark.oam_runtime
def test_minimal_oam_forward_accepts_directed_edges():
    native = _native()
    evaluator = _reference(native, FIXTURE)
    _compute(
        evaluator,
        num_neigh=(1, 0),
        indices=(1,),
        neigh_types=(1,),
        xyz=(1.0, 0.0, 0.0),
        r=(1.0,),
    )
    assert np.isfinite(np.asarray(evaluator.node_energies)).all()
    assert np.asarray(evaluator.node_forces).shape == (3,)
    assert "active_edges=1" in evaluator.last_execution_diagnostic


@pytest.mark.oam_runtime
def test_oam_reverse_forces_match_energy_finite_difference(tmp_path):
    native = _native()
    evaluator = _reference(native, _nonzero_contract(tmp_path))
    xyz = np.asarray((0.8, 0.2, 0.1), dtype=float)
    r = np.asarray((np.linalg.norm(xyz),), dtype=float)
    _compute(evaluator, num_neigh=(1, 0), indices=(1,), neigh_types=(1,), xyz=xyz, r=r)
    force = np.asarray(evaluator.node_forces)
    assert np.linalg.norm(force) > 1.0e-10
    finite_difference = []
    for index in range(3):
        plus = xyz.copy()
        minus = xyz.copy()
        plus[index] += 1.0e-6
        minus[index] -= 1.0e-6
        _compute(
            evaluator,
            num_neigh=(1, 0),
            indices=(1,),
            neigh_types=(1,),
            xyz=plus,
            r=np.asarray((np.linalg.norm(plus),), dtype=float),
        )
        energy_plus = float(np.asarray(evaluator.node_energies).sum())
        _compute(
            evaluator,
            num_neigh=(1, 0),
            indices=(1,),
            neigh_types=(1,),
            xyz=minus,
            r=np.asarray((np.linalg.norm(minus),), dtype=float),
        )
        energy_minus = float(np.asarray(evaluator.node_energies).sum())
        finite_difference.append(-(energy_plus - energy_minus) / 2.0e-6)
    np.testing.assert_allclose(force, finite_difference, rtol=2.0e-4, atol=2.0e-6)


def test_exported_oam_s_contract_parses_uvw_self_connection():
    if EXPORTED is None:
        pytest.skip(
            f"set {EXPORTED_ENVIRONMENT} to an exported OAM-S contract to parse it"
        )
    contract = _exported_contract()
    native = _native()
    evaluator = _reference(native, EXPORTED)
    _compute(evaluator)
    assert evaluator.num_layers == len(contract["layers"])
    assert np.isfinite(np.asarray(evaluator.node_energies)).all()
