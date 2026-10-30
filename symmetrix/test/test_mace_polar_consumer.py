from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from symmetrix.calculator import Symmetrix
from symmetrix.extract_mace_polar import extract_mace_polar_data
from test_mace_polar_backbone import _build_polar_shaped_model


POSITIONS = np.asarray(
    ((0.0, 0.0, 0.0), (2.3, 0.0, 0.0), (0.4, 1.3, 0.2)),
    dtype=np.float64,
)
CELL = np.eye(3, dtype=np.float64) * 12.0
SOURCE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def _initialized_native_runtime(accelerator_capabilities):
    del accelerator_capabilities
    yield


@pytest.fixture(scope="module")
def polar_consumer_model(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("polar-consumer") / "model.json"
    path.write_text(
        json.dumps(
            extract_mace_polar_data(_build_polar_shaped_model()),
            separators=(",", ":"),
        )
    )
    return path


def _consumer(native_module, model_path, dtype):
    cls = (
        native_module.MACEPolarConsumer
        if dtype == "float64"
        else native_module.MACEPolarConsumerFloat
    )
    return cls(str(model_path), "")


def _load_consumer_artifacts(consumer, calculator):
    """Transfer the prepared family artifact and its bound product modules."""

    assert calculator.jit_artifact_path is not None
    assert consumer.load_artifact(calculator.jit_artifact_path, "host")
    products = {
        key: record
        for key, record in calculator.jit_operator_modules.items()
        if key.startswith("product_")
    }
    assert products
    for key in sorted(products):
        record = products[key]
        assert consumer.load_product_artifact(
            record["layer"],
            record["operator"],
            record["artifact_path"],
            json.dumps(record["metadata"], sort_keys=True, separators=(",", ":")),
            record["schedule"],
            record["persistent_blocks_per_compute_unit"],
            record["backend"],
        )
    assert consumer.product_artifacts_ready


def test_native_consumer_exposes_only_narrow_family_controls(
    polar_consumer_model,
):
    import symmetrix.symmetrix as native_module

    consumer = _consumer(native_module, polar_consumer_model, "float64")
    assert consumer.cutoff > 0.0
    assert consumer.streamed_edges == "direct_streamed"
    assert consumer.diagnostics["transaction_count"] == 0
    assert consumer.diagnostics["graph_rebuild_count"] == 0


def test_native_consumer_rejects_solver_before_graph_or_geometry(
    polar_consumer_model,
):
    import symmetrix.symmetrix as native_module

    consumer = _consumer(native_module, polar_consumer_model, "float64")
    with pytest.raises(ValueError, match="requires fully periodic input"):
        consumer.evaluate_system(
            atomic_numbers=np.asarray((14, 14, 1), dtype=np.int32),
            positions=POSITIONS,
            cell=CELL,
            pbc=np.asarray((1, 1, 0), dtype=np.int32),
            electric_field=np.zeros(3),
            solver="pme",
        )
    diagnostics = consumer.diagnostics
    assert diagnostics["transaction_count"] == 0
    assert diagnostics["graph_rebuild_count"] == 0
    assert diagnostics["geometry_publish_count"] == 0
    assert diagnostics["geometry_allocation_count"] == 0


def test_native_consumer_rejects_periodic_image_overflow_before_loop(
    polar_consumer_model,
):
    import symmetrix.symmetrix as native_module

    consumer = _consumer(native_module, polar_consumer_model, "float64")
    singular = np.diag((1.0e-300, 1.0e150, 1.0e150)).astype(np.float64)
    with pytest.raises(OverflowError, match="image extent"):
        consumer.evaluate_system(
            atomic_numbers=np.asarray((14,), dtype=np.int32),
            positions=np.zeros((1, 3), dtype=np.float64),
            cell=singular,
            pbc=np.asarray((1, 0, 0), dtype=np.int32),
            electric_field=np.zeros(3),
            solver="direct_reciprocal",
        )
    assert consumer.diagnostics["transaction_count"] == 0


@pytest.mark.parametrize("dtype,atol", (("float64", 1e-11), ("float32", 3e-4)))
def test_native_consumer_fermi_level_is_inert_metadata(
    polar_consumer_model,
    dtype,
    atol,
):
    """Changing only Fermi metadata must not change the scientific request."""
    import symmetrix.symmetrix as native_module

    consumer = _consumer(native_module, polar_consumer_model, dtype)
    request = dict(
        atomic_numbers=np.asarray((14, 14, 1), dtype=np.int32),
        positions=POSITIONS,
        cell=CELL,
        pbc=np.zeros(3, dtype=np.int32),
        total_charge=0.25,
        total_spin=1.0,
        electric_field=np.asarray((0.01, -0.02, 0.03), dtype=np.float64),
        solver="auto",
        include_forces=True,
        include_stress=True,
        include_dipole=True,
    )
    zero = consumer.evaluate_system(**request, fermi_level=0.0)
    nonzero = consumer.evaluate_system(**request, fermi_level=7.5)

    for name in ("energy", "forces", "stress", "dipole"):
        np.testing.assert_allclose(nonzero[name], zero[name], rtol=0.0, atol=atol)
    for name in (
        "requested_solver",
        "resolved_solver",
        "solver_reason",
        "reverse_strategy",
        "reciprocal_allocation_elements",
        "realspace_retained_elements",
    ):
        assert nonzero["diagnostics"][name] == zero["diagnostics"][name]


def test_native_consumer_scientific_potential_keeps_fermi_lane_zero():
    """The public metadata value must not occupy a scientific potential lane."""
    import symmetrix.symmetrix as native_module

    potential = native_module._polar_consumer_scientific_external_potential(
        7.5, np.asarray((0.25, -0.5, 0.75), dtype=np.float64)
    )
    np.testing.assert_array_equal(potential, (0.0, 0.25, -0.5, 0.75))


@pytest.mark.parametrize("dtype,atol", [("float64", 1e-11), ("float32", 3e-4)])
@pytest.mark.parametrize(
    "pbc,solver,total_charge,electric_field",
    [
        ((False, False, False), "auto", 0.5, (0.01, -0.02, 0.03)),
        ((True, True, True), "direct_reciprocal", 0.0, (0.0, 0.0, 0.0)),
    ],
)
def test_native_consumer_matches_python_adapter_for_efs_and_dipole(
    polar_consumer_model,
    dtype,
    atol,
    pbc,
    solver,
    total_charge,
    electric_field,
):
    import symmetrix.symmetrix as native_module
    from ase import Atoms

    atoms = Atoms("Si2H", positions=POSITIONS, cell=CELL, pbc=pbc)
    adapter = Symmetrix(
        str(polar_consumer_model),
        dtype=dtype,
        streamed_edges="direct",
        polar_electrostatics=solver,
        total_charge=total_charge,
        electric_field=electric_field,
    )
    atoms.calc = adapter
    reference = {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces(),
        "stress": atoms.get_stress(),
        "dipole": adapter.get_property("dipole", atoms),
    }

    consumer = _consumer(native_module, polar_consumer_model, dtype)
    _load_consumer_artifacts(consumer, adapter)
    consumer.streamed_edges = "direct"
    result = consumer.evaluate_system(
        atomic_numbers=np.asarray(atoms.numbers, dtype=np.int32),
        positions=np.asarray(atoms.positions, dtype=np.float64),
        cell=np.asarray(atoms.cell.array, dtype=np.float64),
        pbc=np.asarray(atoms.pbc, dtype=np.int32),
        total_charge=total_charge,
        total_spin=0.0,
        fermi_level=0.0,
        electric_field=np.asarray(electric_field, dtype=np.float64),
        solver=solver,
        include_forces=True,
        include_stress=True,
        include_dipole=True,
    )

    np.testing.assert_allclose(result["energy"], reference["energy"], atol=atol)
    np.testing.assert_allclose(result["forces"], reference["forces"], atol=atol)
    np.testing.assert_allclose(result["stress"], reference["stress"], atol=atol)
    np.testing.assert_allclose(result["dipole"], reference["dipole"], atol=atol)
    assert result["diagnostics"]["backbone_fallback_count"] == 0
    assert result["diagnostics"]["polar_fallback_count"] > 0
    assert (
        result["diagnostics"]["fallback_count"]
        == (result["diagnostics"]["polar_fallback_count"])
    )
    assert result["diagnostics"]["transaction_count"] == 1
    assert result["diagnostics"]["graph_rebuild_count"] == 1
    assert result["diagnostics"]["retained_output_elements"] == 0


def test_native_consumer_reuses_graph_and_replaces_changed_cell(
    polar_consumer_model,
):
    import symmetrix.symmetrix as native_module

    consumer = _consumer(native_module, polar_consumer_model, "float64")
    consumer.streamed_edges = "direct_streamed"
    kwargs = dict(
        atomic_numbers=np.asarray((14, 14, 1), dtype=np.int32),
        positions=POSITIONS.copy(),
        cell=CELL.copy(),
        pbc=np.asarray((0, 0, 0), dtype=np.int32),
        total_charge=0.0,
        total_spin=0.0,
        fermi_level=0.0,
        electric_field=np.zeros(3, dtype=np.float64),
        solver="auto",
        include_forces=True,
        include_stress=False,
        include_dipole=False,
    )
    first = consumer.evaluate_system(**kwargs)
    kwargs["positions"][0, 0] += 0.01
    second = consumer.evaluate_system(**kwargs)
    kwargs["cell"] = CELL * 1.01
    third = consumer.evaluate_system(**kwargs)

    assert first["diagnostics"]["graph_rebuild_count"] == 1
    assert second["diagnostics"]["graph_rebuild_count"] == 1
    assert second["diagnostics"]["graph_reuse_count"] == 1
    assert third["diagnostics"]["graph_rebuild_count"] == 2
    assert third["diagnostics"]["transaction_count"] == 3
    assert third["diagnostics"]["reverse_count"] == 3
    assert third["diagnostics"]["geometry_publish_count"] == 2
    assert third["diagnostics"]["geometry_update_count"] == 1
    assert third["diagnostics"]["geometry_reuse_count"] == 0


def test_native_consumer_accounts_public_transfers_and_reuses_allocations(
    polar_consumer_model,
):
    import symmetrix.symmetrix as native_module

    consumer = _consumer(native_module, polar_consumer_model, "float64")
    request = dict(
        atomic_numbers=np.asarray((14, 14, 1), dtype=np.int32),
        positions=POSITIONS.copy(),
        cell=CELL.copy(),
        pbc=np.zeros(3, dtype=np.int32),
        electric_field=np.zeros(3),
        solver="auto",
        include_forces=True,
        include_stress=True,
        include_dipole=True,
    )
    first = consumer.evaluate_system(**request)["diagnostics"]
    second = consumer.evaluate_system(**request)["diagnostics"]

    assert first["metadata_d2h_bytes"] == 2 * np.dtype(np.int32).itemsize
    assert first["public_output_elements"] == 1 + 9 + 6 + 3
    assert first["public_output_d2h_bytes"] == (1 + 9 + 6 + 3) * 8
    assert first["force_output_d2h_bytes"] == 9 * 8
    assert first["scalar_output_d2h_bytes"] == (1 + 6 + 3) * 8
    assert first["input_d2h_bytes"] == 0
    assert first["input_h2d_bytes"] > 0
    assert first["output_h2d_bytes"] == 0
    assert first["reciprocal_table_elements"] == 0
    assert first["diagnostic_materialization_count"] == 0
    assert first["geometry_allocation_count"] == 1
    assert first["public_output_allocation_count"] == 1
    assert second["geometry_allocation_count"] == 1
    assert second["public_output_allocation_count"] == 1
    assert second["graph_rebuild_count"] == 1
    assert second["graph_reuse_count"] == 1
    assert second["geometry_reuse_count"] == 1


def test_prepared_consumer_reuse_skips_topology_scan_and_counts_uploads(
    polar_consumer_model,
):
    """Removing the reuse fast path must reintroduce a visible scan/count."""
    import symmetrix.symmetrix as native_module
    from ase import Atoms

    atoms = Atoms("Si2H", positions=POSITIONS, cell=CELL, pbc=False)
    preparer = Symmetrix(
        str(polar_consumer_model),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="auto",
    )
    atoms.calc = preparer
    atoms.get_potential_energy()
    assert preparer.jit_artifact_path is not None

    consumer = _consumer(native_module, polar_consumer_model, "float64")
    _load_consumer_artifacts(consumer, preparer)
    consumer.streamed_edges = "direct"

    atomic_numbers = np.asarray((14, 14, 1), dtype=np.int32)
    degrees = np.asarray((2, 2, 2), dtype=np.int32)
    sources = np.asarray((1, 2, 0, 2, 0, 1), dtype=np.int32)
    source_types = np.asarray((1, 0, 1, 0, 1, 1), dtype=np.int32)
    reference_xyz = np.asarray(
        (
            (2.3, 0.0, 0.0),
            (0.4, 1.3, 0.2),
            (-2.3, 0.0, 0.0),
            (-1.9, 1.3, 0.2),
            (-0.4, -1.3, -0.2),
            (1.9, -1.3, -0.2),
        ),
        dtype=np.float64,
    )

    common = dict(
        atomic_numbers=atomic_numbers,
        cell=CELL,
        pbc=np.zeros(3, dtype=np.int32),
        electric_field=np.zeros(3, dtype=np.float64),
        solver="auto",
    )
    first = consumer._evaluate_prepared_for_testing(
        positions=POSITIONS,
        num_neigh=degrees,
        sources=sources,
        source_types=source_types,
        reference_xyz=reference_xyz,
        replace_graph=True,
        **common,
    )
    moved = POSITIONS.copy()
    moved[0, 0] += 0.01
    second = consumer._evaluate_prepared_for_testing(
        positions=moved,
        num_neigh=np.empty(0, dtype=np.int32),
        sources=np.empty(0, dtype=np.int32),
        source_types=np.empty(0, dtype=np.int32),
        reference_xyz=np.empty(0, dtype=np.float64),
        replace_graph=False,
        **common,
    )
    third = consumer._evaluate_prepared_for_testing(
        positions=moved,
        num_neigh=np.empty(0, dtype=np.int32),
        sources=np.empty(0, dtype=np.int32),
        source_types=np.empty(0, dtype=np.int32),
        reference_xyz=np.empty(0, dtype=np.float64),
        replace_graph=False,
        **common,
    )

    # Hand-derived uploads for N=3, E=6 and one source-schedule block.  The
    # rebuild includes 680 backbone bytes, then the Polar position/node graph,
    # volume, counts, channels, and types (72+12+8+4+60+12).  A moved geometry
    # uploads both position spellings, the Polar node graph, and volume; an
    # identical transaction still uploads the reporting volume.
    assert first["input_h2d_bytes"] == 848
    assert second["input_h2d_bytes"] == 164
    assert third["input_h2d_bytes"] == 8
    assert first["topology_validation_count"] == 1
    assert second["topology_validation_count"] == 1
    assert third["topology_validation_count"] == 1
    assert first["topology_validation_edge_count"] == 6
    assert second["topology_validation_edge_count"] == 6
    assert third["topology_validation_edge_count"] == 6
    assert first["topology_reuse_scan_count"] == 0
    assert second["topology_reuse_scan_count"] == 0
    assert third["topology_reuse_scan_count"] == 0
    assert first["topology_facade_scan_count"] == 0
    assert second["topology_facade_scan_count"] == 0
    assert third["topology_facade_scan_count"] == 0
    assert first["input_d2h_bytes"] == 0
    assert second["input_d2h_bytes"] == 0
    assert third["input_d2h_bytes"] == 0
    assert first["force_output_d2h_bytes"] == 0


def test_failed_graph_geometry_replacement_preserves_predecessor(
    polar_consumer_model,
):
    import symmetrix.symmetrix as native_module

    consumer = _consumer(native_module, polar_consumer_model, "float64")
    consumer.streamed_edges = "direct_streamed"
    request = dict(
        atomic_numbers=np.asarray((14, 14, 1), dtype=np.int32),
        positions=POSITIONS.copy(),
        cell=CELL.copy(),
        pbc=np.zeros(3, dtype=np.int32),
        electric_field=np.zeros(3),
        solver="auto",
        include_forces=True,
    )
    reference = consumer.evaluate_system(**request)
    consumer._arm_graph_geometry_failure_for_testing()
    changed = dict(request, cell=CELL * 1.01)
    with pytest.raises(RuntimeError, match="injected prepared-geometry failure"):
        consumer.evaluate_system(**changed)
    recovered = consumer.evaluate_system(**request)
    np.testing.assert_allclose(recovered["energy"], reference["energy"], atol=1e-12)
    np.testing.assert_allclose(recovered["forces"], reference["forces"], atol=1e-12)
    assert recovered["diagnostics"]["transaction_count"] == 2
    assert recovered["diagnostics"]["graph_rebuild_count"] == 1
    assert recovered["diagnostics"]["geometry_publish_count"] == 1


def test_graph_geometry_rollback_retains_device_owners_without_download():
    source = (
        SOURCE_ROOT / "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    transaction = source.split(
        "MaceBackboneKokkosT<Precision>::replace_factorized_graph_geometry(", 1
    )[1].split(
        "MaceBackboneKokkosT<Precision>::validate_factorized_prepared_system(",
        1,
    )[0]
    compact = transaction.replace(" ", "")
    assert "SYMMETRIX_RETAIN_VIEW(execution_prepared_neigh_indices)" in compact
    assert "SYMMETRIX_RESTORE_VIEW(execution_prepared_neigh_indices)" in compact
    assert "Restore predecessor after failed MH-1 graph replacement" in transaction
    assert "create_mirror_view_and_copy" not in transaction


def test_kokkos_public_output_helpers_preserve_force_and_map_virial():
    import symmetrix.symmetrix as native_module

    result = native_module._polar_consumer_device_mapping_probe()
    np.testing.assert_array_equal(
        result["forces"],
        np.asarray(((14.0, 25.0, 36.0), (41.0, 52.0, 63.0), (70.0, 80.0, 90.0))),
    )
    np.testing.assert_array_equal(result["virial"], (-10, -20, -30, -60, -50, -40))
    np.testing.assert_array_equal(result["isolated_virial"], np.zeros(6))


def test_native_consumer_has_no_calculator_import_dependency(
    polar_consumer_model,
    tmp_path,
):
    from ase import Atoms

    atoms = Atoms("Si2H", positions=POSITIONS, cell=CELL, pbc=False)
    preparer = Symmetrix(
        str(polar_consumer_model),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="auto",
    )
    atoms.calc = preparer
    atoms.get_potential_energy()
    assert preparer.jit_artifact_path is not None
    script = tmp_path / "native_consumer.py"
    script.write_text(
        """
import builtins
import json
import numpy as np

real_import = builtins.__import__
def rejecting_import(name, *args, **kwargs):
    if name == "symmetrix.calculator" or name.startswith("symmetrix.calculator."):
        raise AssertionError("calculator import is forbidden")
    return real_import(name, *args, **kwargs)
builtins.__import__ = rejecting_import

import symmetrix.symmetrix as native
native._init_kokkos()
consumer = native.MACEPolarConsumer(MODEL, "")
assert consumer.load_artifact(ARTIFACT, "host")
for record in PRODUCTS:
    assert consumer.load_product_artifact(
        record["layer"], record["operator"], record["artifact_path"],
        json.dumps(record["metadata"], sort_keys=True, separators=(",", ":")),
        record["schedule"], record["persistent_blocks_per_compute_unit"],
        record["backend"],
    )
assert consumer.product_artifacts_ready
consumer.streamed_edges = "direct"
result = consumer.evaluate_system(
    atomic_numbers=np.asarray([14, 14, 1], dtype=np.int32),
    positions=np.asarray(POSITIONS, dtype=np.float64),
    cell=np.eye(3, dtype=np.float64) * 12.0,
    pbc=np.zeros(3, dtype=np.int32),
    total_charge=0.0,
    total_spin=0.0,
    fermi_level=0.0,
    electric_field=np.zeros(3, dtype=np.float64),
    solver="auto",
    include_forces=True,
    include_stress=True,
    include_dipole=True,
)
assert result["diagnostics"]["transaction_count"] == 1
assert result["diagnostics"]["backbone_fallback_count"] == 0
""".replace("MODEL", repr(str(polar_consumer_model)))
        .replace("ARTIFACT", repr(preparer.jit_artifact_path))
        .replace(
            "PRODUCTS",
            repr(
                [
                    record
                    for key, record in sorted(preparer.jit_operator_modules.items())
                    if key.startswith("product_")
                ]
            ),
        )
        .replace("POSITIONS", repr(POSITIONS.tolist()))
    )
    environment = os.environ.copy()
    environment["SYMMETRIX_JIT_CACHE"] = str(tmp_path / "jit")
    environment["SYMMETRIX_JIT_POLICY"] = "required"
    result = subprocess.run(
        [sys.executable, str(script)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
