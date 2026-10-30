"""OAM conformance cases: the edge-product contract and its angular variants.

This module is where the OAM crystal, contracts, adapter, instance specs, and
routing case live; the shared suites name no family and drive whatever the
registry collected.  The merged-l1/l2 angular contract exercises the lane
conversions, and the minimal contract exercises the folded self connection.

OAM also declares the routing facts a shared suite cannot infer: its frontend
prepares the receiver-owned artifact without consulting the shared JIT policy,
and its reference routing is a host-execution-space surface.  Its host direct
forces are checked against the gradient of its own energy, which is the
independent oracle the OAM stress plan names now that the serial evaluator's
exact-MLP radial reference is gone.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import symmetrix.oam_jit_artifact as oam_jit_artifact
from ase import Atoms
from symmetrix import NequIPCalculator

import oam_host_oracle
from conformance.contract import (
    Adapter,
    CXX_RUNNER,
    FLOAT32_RUNNER,
    Graph,
    Unqualified,
    _diagnostic,
    execution_space,
    native_symmetrix,
)
from conformance.registry import (
    FamilyConformance,
    RoutingParityCase,
    register_family,
)
from conformance.surface import EmptyPathCase, _assert_interface_surface, _native

#: The minimal OAM contract every OAM instance and readiness case shares.
OAM_MINIMAL = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"

#: The frontend prepares a direct artifact through the entry point the family
#: declares, so a stub replaces that declared entry point rather than a copy of
#: it on the frontend module.
OAM_HOST_ADAPTER = oam_jit_artifact
OAM_HOST_PREPARER = "prepare_nequip_oam_receiver_host_artifact"


class OAMAdapter(Adapter):
    """OAM: the neutral lifecycle spellings are bound directly on the evaluator.

    The consolidated reference is the compiler-free Kokkos edge product driven
    through the whole-evaluator entry point.  Every OAM reference result in this
    module flows through ``oracle``, so the reference side is one data change
    here and nowhere else.
    """

    family = "OAM"
    contract_key = "oam_minimal"

    def structure(self) -> Atoms:
        return _oam_structure()

    def model_path(self) -> Path:
        return self._models[self.contract_key]

    def _class_name(self) -> str:
        if self.precision == "float32":
            if not hasattr(native_symmetrix, "NequIPOAMKokkosFloat"):
                raise Unqualified("OAM float32 execution", FLOAT32_RUNNER)
            return "NequIPOAMKokkosFloat"
        return "NequIPOAMKokkos"

    def _build(self, tier: str):
        evaluator_class = getattr(native_symmetrix, self._class_name())
        if tier == "reference":
            # The consolidated T0 reference: the compiler-free Kokkos edge
            # product reached through the whole-evaluator entry point, with no
            # artifact.  Its radial semantics are the certified spline table,
            # not the removed serial evaluator's exact MLP; the exact-MLP
            # oracle is retained independently by test_nequip_oam_radial.py.
            # The float64 evaluator is the oracle for every instance.
            evaluator = native_symmetrix.NequIPOAMKokkos(str(self.model_path()))
            evaluator.set_streamed_edges("reference")
            return evaluator
        if tier == "portable":
            evaluator = evaluator_class(str(self.model_path()))
            evaluator.set_streamed_edges("reference")
            return evaluator
        if shutil.which("c++") is None:
            raise Unqualified("the OAM receiver-owned host artifact", CXX_RUNNER)
        from symmetrix.oam_jit_artifact import (
            prepare_nequip_oam_receiver_host_artifact,
        )

        artifact = prepare_nequip_oam_receiver_host_artifact(
            self.model_path(), self.precision, cache_dir=self._models["jit_cache"]
        )
        evaluator = evaluator_class(str(self.model_path()), str(artifact.artifact_path))
        evaluator.set_streamed_edges("direct")
        return evaluator

    def bindings_on(self, evaluator) -> dict[str, str]:
        return {
            "prepare_graph": "prepare_graph",
            "prepare_geometry": "prepare_geometry",
            "compute_prepared": "compute_prepared",
            "reduce_forces": "reduce_forces",
            "reduce_stress": "reduce_stress",
        }

    def prepare_graph(self, evaluator, graph: Graph) -> int:
        return evaluator.prepare_graph(*graph.topology())

    def prepare_geometry(self, evaluator, generation: int, graph: Graph) -> None:
        evaluator.prepare_geometry(
            generation,
            graph.reference_positions,
            graph.xyz,
            graph.cell,
            graph.inverse_cell,
            graph.pbc,
        )

    def compute_prepared(self, evaluator, generation: int, graph: Graph) -> None:
        evaluator.compute_prepared(generation, graph.xyz, graph.distances)

    def reduce_forces(self, evaluator, generation: int, graph: Graph) -> np.ndarray:
        evaluator.reduce_forces(generation)
        return np.asarray(evaluator.atom_forces, dtype=np.float64).reshape(
            graph.num_nodes, 3
        )

    def reduce_stress(self, evaluator, generation: int, graph: Graph):
        evaluator.reduce_stress(graph.volume, generation)
        return np.asarray(evaluator.stress_tensor, dtype=np.float64).reshape(3, 3)

    def generated_execution_evidence(self, evaluator) -> dict:
        assert self.artifact_ready(evaluator) is True
        diagnostic = _diagnostic(evaluator)
        fields = dict(
            item.split("=", 1) for item in diagnostic.split(", ") if "=" in item
        )
        return {
            "artifact_id": evaluator.family_artifact_readiness["artifact_id"],
            # The receiver/source-owned ABI v7 program is the generated implementation
            # the binding report steps the edge node to, and the dispatch
            # accounting proves it served every edge region.
            "receiver_owned_abi": fields.get("host_plugin_abi") == "v7",
            "all_edge_regions_generated": int(
                fields.get("edge_generic_dispatches", "-1")
            )
            == 0,
            "edge_dispatches": int(fields.get("edge_dispatches", "0")),
        }

    def compiler_free_marker(self) -> tuple[str, ...]:
        return ("implementation=portable_oam_kokkos_generic",)

    def accepted_generation(self, evaluator) -> int:
        return int(evaluator.graph_generation)

    def geometry_generation_of(self, evaluator):
        return int(evaluator.geometry_generation)


OAM_ORACLE = (
    "the consolidated OAM reference: the compiler-free Kokkos edge product "
    "(`NequIPOAMKokkos` with no artifact) whole-evaluator entry point (T0 tier); "
    "its radial semantics are the certified spline table"
)

OAM_KEYS = frozenset(
    (
        "geometry",
        "radial",
        "harmonics",
        "edge_tensor_product",
        "pre_linear",
        "post_linear",
        "gate",
        "readout",
        "pair_energy",
        "reduce_forces",
        "reduce_stress",
    )
)
# A contract whose self connection is folded into the pre-linear declares no
# separate self-connection node, so the instance carries the wider set.
OAM_MINIMAL_KEYS = OAM_KEYS | {"self_connection"}
OAM_LAYOUTS = frozenset(("edge_major", "lane_major"))
OAM_REDUCTIONS = frozenset(("reduce_forces", "reduce_stress"))
OAM_GENERATED = frozenset(
    ("radial", "edge_tensor_product", "pre_linear", "post_linear", "readout")
)

# Each entry is one admitted contract instance.  ``parameters`` states the
# instance's shape, layout, precision, and derivative requirements so a failure
# names the instance rather than the family.
INSTANCE_SPECS: dict[str, dict] = {
    "oam-minimal-f64": dict(
        adapter=OAMAdapter,
        contract_key="oam_minimal",
        precision="float64",
        parameters="OAM minimal angular contract, edge/lane-major buffers, float64, "
        "forces and stress",
        oracle=OAM_ORACLE,
        lifecycle_tiers=("portable", "generated"),
        expected_kinds=OAM_MINIMAL_KEYS,
        expected_layouts=OAM_LAYOUTS,
        expected_reductions=OAM_REDUCTIONS,
        generated_kinds=OAM_GENERATED,
        rtol=1e-9,
        atol=1e-8,
    ),
    "oam-minimal-f32": dict(
        adapter=OAMAdapter,
        contract_key="oam_minimal",
        precision="float32",
        parameters="OAM minimal angular contract, edge/lane-major buffers, float32, "
        "forces and stress",
        oracle=OAM_ORACLE,
        lifecycle_tiers=("portable", "generated"),
        expected_kinds=OAM_MINIMAL_KEYS,
        expected_layouts=OAM_LAYOUTS,
        expected_reductions=OAM_REDUCTIONS,
        generated_kinds=OAM_GENERATED,
        rtol=1e-3,
        atol=1e-4,
        stress_rtol=2e-3,
        stress_atol=5e-7,
    ),
    "oam-angular-f64": dict(
        adapter=OAMAdapter,
        contract_key="oam_angular",
        precision="float64",
        parameters="OAM merged-l1/l2 angular contract (lane-major to node-major "
        "conversions exercised), float64, forces and stress",
        oracle=OAM_ORACLE,
        lifecycle_tiers=("portable",),
        expected_kinds=OAM_KEYS,
        expected_layouts=OAM_LAYOUTS,
        expected_reductions=OAM_REDUCTIONS,
        generated_kinds=OAM_GENERATED,
        rtol=1e-9,
        atol=1e-8,
    ),
}


def _oam_structure() -> Atoms:
    """The OAM periodic structure, sized so the skin is real."""
    # A skew cell with a neighbor across its boundary separates the virial
    # from a positions-only force contraction.
    return Atoms(
        numbers=[1, 2, 3],
        scaled_positions=[
            [0.03, 0.07, 0.11],
            [0.94, 0.16, 0.18],
            [0.22, 0.31, 0.37],
        ],
        cell=[[7.0, 0.3, -0.2], [0.1, 7.6, 0.4], [-0.3, 0.2, 8.1]],
        pbc=True,
    )


def _oam_angular_contract(root: Path) -> Path:
    """The merged-l1/l2 angular contract that exercises the lane conversions."""
    try:
        from test_nequip_oam_lane_layout import _build_contract
        from symmetrix.nequip_schema import validate_nequip_oam_data
    except ImportError as exc:  # pragma: no cover - depends on test helpers
        pytest.skip(f"the OAM angular contract builder is unavailable: {exc}")

    import json

    contract = _build_contract()
    validate_nequip_oam_data(contract)
    path = root / "oam-angular-merged.json"
    path.write_text(json.dumps(contract))
    return path


def _build_models(root: Path) -> dict:
    """The minimal contract and the merged-l1/l2 angular contract."""
    return {
        "oam_minimal": OAM_MINIMAL,
        "oam_angular": _oam_angular_contract(root),
    }


def _build_routing_model(request, root: Path) -> Path:
    """The minimal contract the routing-parity suite evaluates."""
    return OAM_MINIMAL


def _oam_routing_calculator(model_path, *, dtype, routing):
    """The OAM frontend, guarded by the surface its reference routing needs."""
    if routing == "reference":
        oam_host_oracle.require_host_execution_space(
            execution_space(), "the OAM reference routing"
        )
    return NequIPCalculator(model_path, dtype=dtype, streamed_edges=routing)


def _oam_routing_atoms(*, routing, dtype):
    """The OAM structure carrying the routing-parity calculator of one routing."""
    atoms = _oam_structure()
    atoms.calc = ROUTING_CASE.calculator(OAM_MINIMAL, dtype=dtype, routing=routing)
    return atoms, atoms.calc


def _observe_reference_preparation(monkeypatch, observed):
    """Refuse artifact preparation and record the target it was asked for.

    The frontend resolves this entry point, so a reference request that reached
    it would fail here; the direct request must reach it, which is what makes
    the refusal the reason the direct request failed rather than an accident.
    """

    def _refuse(path, *, precision, target, native_module):
        observed.append(target)
        raise RuntimeError("an artifact was prepared for a reference request")

    monkeypatch.setattr(OAM_HOST_ADAPTER, OAM_HOST_PREPARER, _refuse)


ROUTING_CASE = RoutingParityCase(
    key="OAM",
    structure="oam",
    build_model=_build_routing_model,
    calculator=_oam_routing_calculator,
    preparer_guard=_observe_reference_preparation,
    reference_preparer_calls=(),
    direct_preparer_calls=("host",),
)


def _oam_readiness_evaluator(request):
    """The OAM evaluator the shared empty-path readiness case drives."""
    native = _native()
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("NequIPOAMKokkos is unavailable in this extension")
    return native.NequIPOAMKokkos(str(OAM_MINIMAL))


register_family(
    FamilyConformance(
        name="OAM",
        structures={"oam": _oam_structure},
        build_models=_build_models,
        instances=INSTANCE_SPECS,
        # The receiver-owned host artifact is staged by a host C++20 compiler.
        host_compiler_gated_instances=("oam-minimal-f64", "oam-minimal-f32"),
        readiness_cases=(EmptyPathCase(name="oam", build=_oam_readiness_evaluator),),
        routing_parity=ROUTING_CASE,
    )
)


def _oam_graph():
    node_types = np.asarray((0, 1), dtype=np.int32)
    num_neigh = np.asarray((1, 1), dtype=np.int32)
    indices = np.asarray((1, 0), dtype=np.int32)
    neigh_types = np.asarray((1, 0), dtype=np.int32)
    xyz = np.asarray((0.9, 0.3, -0.2, 1.4, -0.8, 0.5), dtype=np.float64)
    distances = np.linalg.norm(xyz.reshape(-1, 3), axis=1)
    return len(node_types), node_types, num_neigh, indices, neigh_types, xyz, distances


def test_oam_reference_owns_the_portable_family_implementation():
    native = _native()
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("NequIPOAMKokkos is unavailable in this extension")

    # No artifact path at all: only an explicit reference request may bind the
    # compiler-free Kokkos path.
    evaluator = native.NequIPOAMKokkos(str(OAM_MINIMAL))
    evaluator._set_execution_plan_request("reference", "capacity")
    program = _assert_interface_surface(evaluator)

    capabilities = evaluator.family_capabilities
    assert capabilities["target"] in ("host", "cuda")
    assert capabilities["reference"] is True
    assert capabilities["portable"] is True
    assert capabilities["generated"] is False
    assert capabilities["device_artifact"] is (capabilities["target"] == "cuda")
    assert evaluator.family_artifact_ready is False
    assert evaluator.family_artifact_readiness["artifact_id"] == ""

    report = evaluator.family_binding_report
    bindings = {binding["capability_key"]: binding for binding in report["bindings"]}
    edge = bindings["edge_tensor_product"]
    assert edge["implementation_id"] == "portable_oam_kokkos_generic"
    assert edge["tier"] == "portable"
    assert edge["qualified"] is True
    assert edge["portable_covered"] is True
    assert edge["artifact_required"] is False
    # The stress reduction is execution-space generic and the portable path
    # populates the geometry adjoint it consumes, so every program carries a
    # qualified stress implementation and is admitted with full coverage.
    assert capabilities["stress"] is True
    assert bindings["reduce_stress"]["qualified"] is True
    assert bindings["reduce_stress"]["portable_covered"] is True
    assert report["admitted"] is True
    assert report["portable_node_count"] == report["node_count"]
    assert all(binding["qualified"] for binding in report["bindings"])
    assert {operator["capability_key"] for operator in program["operators"]} >= {
        "geometry",
        "radial",
        "harmonics",
        "edge_tensor_product",
        "pre_linear",
        "post_linear",
        "gate",
        "readout",
        "pair_energy",
        "reduce_forces",
        "reduce_stress",
    }

    plan = evaluator.execution_plan_report
    assert plan["selected_id"] == "portable_oam_kokkos_generic"
    assert plan["candidates"] == []
    assert plan["available_bytes"] is None
    assert plan["reserve_bytes"] is None
    assert plan["boundary_attempt"] is False
    assert "outside direct candidate ranking" in plan["selection_reason"]


@pytest.mark.oam_runtime
def test_oam_neutral_lifecycle_matches_the_reference_entry_point_without_artifacts():
    native = _native()
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("NequIPOAMKokkos is unavailable in this extension")

    num_nodes, node_types, num_neigh, indices, neigh_types, xyz, distances = (
        _oam_graph()
    )
    positions = np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float64)

    reference = native.NequIPOAMKokkos(str(OAM_MINIMAL))
    reference._set_execution_plan_request("reference", "capacity")
    reference.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, indices, neigh_types, xyz, distances
    )

    prepared = native.NequIPOAMKokkos(str(OAM_MINIMAL))
    prepared._set_execution_plan_request("reference", "capacity")
    graph_generation = prepared.prepare_graph(
        num_nodes, node_types, num_neigh, indices, neigh_types
    )
    assert graph_generation == prepared.graph_generation
    prepared.prepare_geometry(
        graph_generation,
        positions,
        xyz,
        np.zeros(0),
        np.zeros(0),
        np.zeros(0, dtype=np.int32),
    )
    assert prepared.geometry_generation == graph_generation
    prepared.compute_prepared(graph_generation, xyz, distances)

    np.testing.assert_allclose(
        np.asarray(prepared.node_energies, dtype=float),
        np.asarray(reference.node_energies, dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(prepared.node_forces, dtype=float),
        np.asarray(reference.node_forces, dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    # The lifecycle reduction is the step the whole-evaluator entry point does
    # not expose: it reduces the retained adjoint to atom forces.
    prepared.reduce_forces(graph_generation)
    assert len(prepared.atom_forces) == 3 * num_nodes
    assert (
        "implementation=portable_oam_kokkos_generic"
        in prepared.last_execution_diagnostic
    )


def test_nequip_calculator_rejects_use_kokkos():
    """The OAM frontend refuses the retired keyword the same way."""

    for value in (True, False):
        with pytest.raises(TypeError, match="use_kokkos has been retired"):
            NequIPCalculator(OAM_MINIMAL, dtype="float64", use_kokkos=value)


@pytest.mark.oam_runtime
def test_oam_host_direct_forces_are_the_energy_gradient():
    """OAM ``direct`` forces are checked against a numerical gradient.

    The removed serial evaluator supplied an exact-MLP radial reference for the
    energy; with it gone, the surviving independent check on the direct path's
    forces is the finite-difference derivative of its own energy.  The step is
    small enough to stay inside the neighbor skin, so the topology is fixed
    across every displacement.
    """

    atoms, calculator = _oam_routing_atoms(routing="direct", dtype="float64")
    assert calculator.streamed_edges == "direct"

    step = 1e-5
    analytic = np.asarray(atoms.get_forces(), dtype=np.float64)
    positions = np.array(atoms.get_positions(), dtype=np.float64)
    numeric = np.zeros_like(analytic)
    for atom in range(len(atoms)):
        for axis in range(3):
            displaced = positions.copy()
            displaced[atom, axis] += step
            atoms.set_positions(displaced)
            forward = float(atoms.get_potential_energy())
            displaced[atom, axis] -= 2.0 * step
            atoms.set_positions(displaced)
            backward = float(atoms.get_potential_energy())
            numeric[atom, axis] = -(forward - backward) / (2.0 * step)
    atoms.set_positions(positions)

    scale = max(float(np.max(np.abs(analytic))), 1e-30)
    deviation = float(np.max(np.abs(analytic - numeric)))
    assert deviation <= 1e-6 * scale, (
        f"OAM direct forces deviate from the energy gradient by {deviation} "
        f"(force scale {scale})"
    )


@pytest.mark.parametrize("family", ("oam",))
def test_selected_implementation_id_names_the_dispatched_implementation(family):
    """The per-node implementation id is the one the execution path dispatches.

    ``selected_implementation_id`` is read from the native binding, so this pins
    it against the execution path's own statement of what it runs instead of
    against a second derivation: the OAM dispatch records the implementation
    that served every edge region in its execution diagnostic, and the binding
    states the same id for the edge nodes.  The whole-plan report carries the
    same per-node ids, so a consumer reads one authority surface rather than
    recomputing a tier.
    """
    native = _native()
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("NequIPOAMKokkos is unavailable in this extension")
    evaluator = native.NequIPOAMKokkos(str(OAM_MINIMAL))
    evaluator._set_execution_plan_request("reference", "capacity")

    report = evaluator.family_binding_report
    bindings = {binding["capability_key"]: binding for binding in report["bindings"]}
    for capability_key, binding in bindings.items():
        assert binding["qualified"] is True, capability_key
        assert binding["selected_implementation_id"], capability_key

    # The OAM dispatch records the implementation that served every edge
    # region, and the binding states the same id for the edge nodes.
    diagnostic = dict(
        item.split("=", 1)
        for item in evaluator.last_execution_diagnostic.split(", ")
        if "=" in item
    )
    edge_ids = {
        binding["selected_implementation_id"]
        for binding in report["bindings"]
        if binding["capability_key"] == "edge_tensor_product"
    }
    assert edge_ids == {diagnostic["implementation"]}, edge_ids

    plan = evaluator.execution_plan_report
    if plan["bindings"]:
        assert [
            binding["selected_implementation_id"] for binding in plan["bindings"]
        ] == [binding["selected_implementation_id"] for binding in report["bindings"]]


@pytest.mark.oam_runtime
@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("profile", ("speed", "capacity"))
@pytest.mark.parametrize("variant", ("full_message",))
def test_oam_full_message_direct_matches_reference_with_truthful_reports(
    tmp_path, precision, profile, variant
):
    """Every release host variant preserves values and reports its real path."""

    native = _native()
    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the OAM host direct/reference conformance matrix",
    )
    from symmetrix.oam_jit_artifact import (
        prepare_nequip_oam_receiver_host_artifact,
    )

    tmp_path.chmod(0o700)
    artifact = prepare_nequip_oam_receiver_host_artifact(
        OAM_MINIMAL, precision, cache_dir=tmp_path
    )
    class_name = "NequIPOAMKokkosFloat" if precision == "float32" else "NequIPOAMKokkos"
    evaluator_class = getattr(native, class_name)
    direct = evaluator_class(
        str(OAM_MINIMAL),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
    reference = evaluator_class(str(OAM_MINIMAL))
    direct.set_streamed_edges("direct")
    reference.set_streamed_edges("reference")
    direct._set_execution_plan_request("direct", profile)
    reference._set_execution_plan_request("reference", profile)
    direct._set_execution_edge_variant(variant)

    num_nodes, node_types, num_neigh, indices, neigh_types, _, _ = _oam_graph()
    xyz = np.asarray((0.9, 0.3, -0.2, -0.9, -0.3, 0.2), dtype=np.float64)
    positions = np.asarray((0.0, 0.0, 0.0, 0.9, 0.3, -0.2), dtype=np.float64)
    cell = np.eye(3, dtype=np.float64) * 5.0
    pbc = np.zeros(3, dtype=np.int32)

    admitted = False

    def evaluate(
        evaluator,
        *,
        positions_value=positions,
        xyz_value=xyz,
        cell_value=cell,
        admit=False,
    ):
        nonlocal admitted
        generation = evaluator.prepare_graph(
            num_nodes, node_types, num_neigh, indices, neigh_types
        )
        if admit and not admitted:
            plan_id = (
                "oam-direct-capacity" if profile == "capacity" else "oam-direct-speed"
            )
            candidate = {
                item["id"]: item
                for item in evaluator.execution_plan_report["candidates"]
            }[plan_id]
            estimate = candidate["estimated_bytes"]
            evaluator._set_execution_plan_qualification(
                plan_id,
                artifact.artifact_id,
                "host",
                precision,
                "capacity" if profile == "capacity" else "throughput",
                candidate["required_capability"],
                "qualified",
                estimate,
                estimate,
                0,
            )
            admitted = True
        evaluator.prepare_geometry(
            generation,
            positions_value,
            xyz_value,
            cell_value.reshape(-1),
            np.linalg.inv(cell_value).reshape(-1),
            pbc,
        )
        evaluator.compute_prepared(
            generation,
            xyz_value,
            np.linalg.norm(xyz_value.reshape(-1, 3), axis=1),
        )
        evaluator.reduce_forces(generation)
        evaluator.reduce_stress(float(np.linalg.det(cell_value)), generation)
        result = (
            np.asarray(evaluator.node_energies, dtype=np.float64).copy(),
            np.asarray(evaluator.atom_forces, dtype=np.float64).copy(),
            np.asarray(evaluator.stress_tensor, dtype=np.float64).copy(),
        )
        evaluator.reduce_stress(float(np.linalg.det(cell_value)), generation)
        evaluator.reduce_forces(generation)
        np.testing.assert_allclose(
            np.asarray(evaluator.atom_forces, dtype=np.float64),
            result[1],
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            np.asarray(evaluator.stress_tensor, dtype=np.float64),
            result[2],
            rtol=0,
            atol=0,
        )
        return result

    expected = evaluate(reference)
    actual = evaluate(direct, admit=True)
    rtol, atol = (2e-3, 1e-5) if precision == "float32" else (1e-11, 1e-11)
    for observed, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=rtol, atol=atol)

    plan = direct.execution_plan_report
    assert plan["selected_id"] == (
        "oam-direct-capacity" if profile == "capacity" else "oam-direct-speed"
    )
    selected_candidate = next(
        candidate
        for candidate in plan["candidates"]
        if candidate["id"] == plan["selected_id"]
    )
    assert selected_candidate["empirical_qualification"] == "qualified"
    assert plan["requested_profile"] == profile
    assert plan["edge_variant"] == variant
    assert plan["artifact_origin"] in {"prebuilt", "cache", "runtime"}
    assert plan["loaded_artifact_identity"] == artifact.artifact_id
    assert plan["fallback_count"] == 0
    assert all(binding["qualified"] for binding in plan["bindings"])
    assert {
        binding["implementation_id"]
        for binding in plan["bindings"]
        if binding["capability_key"] == "edge_tensor_product"
    } == {"receiver_owned_full_message_oam_host_v7"}

    # One central coordinate derivative and one affine strain component keep
    # this full 2x2x2 matrix focused while independently checking both
    # reductions for the candidate that was actually admitted above.
    force_step = 2e-3 if precision == "float32" else 1e-5
    displaced = positions.reshape(-1, 3).copy()
    finite_energies = []
    for sign in (1.0, -1.0):
        moved = displaced.copy()
        moved[1, 0] += sign * force_step
        moved_xyz = np.concatenate((moved[1] - moved[0], moved[0] - moved[1]))
        finite_energies.append(
            float(
                evaluate(
                    direct, positions_value=moved.reshape(-1), xyz_value=moved_xyz
                )[0].sum()
            )
        )
    numerical_force = -(finite_energies[0] - finite_energies[1]) / (2 * force_step)
    np.testing.assert_allclose(
        actual[1].reshape(-1, 3)[1, 0],
        numerical_force,
        rtol=2e-3 if precision == "float32" else 1e-6,
        atol=1e-4 if precision == "float32" else 1e-8,
    )

    strain_step = 2e-3 if precision == "float32" else 1e-5
    strain_energies = []
    for sign in (1.0, -1.0):
        transform = np.eye(3)
        transform[0, 0] += sign * strain_step
        strained_positions = displaced @ transform
        strained_xyz = xyz.reshape(-1, 3) @ transform
        strained_cell = cell @ transform
        strain_energies.append(
            float(
                evaluate(
                    direct,
                    positions_value=strained_positions.reshape(-1),
                    xyz_value=strained_xyz.reshape(-1),
                    cell_value=strained_cell,
                )[0].sum()
            )
        )
    numerical_stress = (strain_energies[0] - strain_energies[1]) / (
        2 * strain_step * np.linalg.det(cell)
    )
    np.testing.assert_allclose(
        actual[2].reshape(3, 3)[0, 0],
        numerical_stress,
        rtol=2e-3 if precision == "float32" else 2e-5,
        atol=5e-7 if precision == "float32" else 2e-9,
    )

    retrained_contract = json.loads(OAM_MINIMAL.read_text())
    retrained_contract["per_type_shift"]["values"] = [
        value + 0.125 for value in retrained_contract["per_type_shift"]["values"]
    ]
    retrained_path = tmp_path / "retrained.json"
    retrained_path.write_text(json.dumps(retrained_contract))
    retrained_direct = evaluator_class(
        str(retrained_path),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
    retrained_reference = evaluator_class(str(retrained_path))
    retrained_direct.set_streamed_edges("direct")
    retrained_reference.set_streamed_edges("reference")
    retrained_direct._set_execution_plan_request("direct", profile)
    retrained_reference._set_execution_plan_request("reference", profile)
    retrained_direct._set_execution_edge_variant(variant)
    admitted = False
    retrained_actual = evaluate(retrained_direct, admit=True)
    retrained_expected = evaluate(retrained_reference)
    for observed, wanted in zip(retrained_actual, retrained_expected, strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=rtol, atol=atol)
    assert not np.allclose(retrained_actual[0], actual[0], rtol=0.0, atol=0.0)


@pytest.mark.oam_runtime
@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("profile", ("speed", "capacity"))
@pytest.mark.parametrize("variant", ("full_message",))
def test_oam_direct_rejects_a_structurally_different_contract(
    tmp_path, precision, profile, variant
):
    native = _native()
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact

    tmp_path.chmod(0o700)
    artifact = prepare_nequip_oam_receiver_host_artifact(
        OAM_MINIMAL, precision, cache_dir=tmp_path
    )
    changed = json.loads(OAM_MINIMAL.read_text())
    changed["layers"][0]["edge_tensor_product"]["instructions"][0]["path_weight"] *= 2.0
    changed_path = tmp_path / "structurally-different.json"
    changed_path.write_text(json.dumps(changed))
    class_name = "NequIPOAMKokkosFloat" if precision == "float32" else "NequIPOAMKokkos"
    evaluator_class = getattr(native, class_name)
    from benchmarks import oam_cuda_r1_timing

    admitted_context = oam_cuda_r1_timing.prepare_direct_evaluator(
        lambda context: evaluator_class(
            str(OAM_MINIMAL),
            str(artifact.artifact_path),
            host_active_target=artifact.active_host_target,
            host_artifact_origin=artifact.artifact_origin,
            requested_profile=context["profile"],
            requested_edge_variant=context["variant"],
            requested_precision=context["precision"],
        ),
        profile=profile,
        variant=variant,
        precision=precision,
    )
    expected_plan = (
        "oam-direct-capacity" if profile == "capacity" else "oam-direct-speed"
    )
    num_nodes, node_types, num_neigh, indices, neigh_types, _, _ = _oam_graph()
    admitted_context.prepare_graph(
        num_nodes, node_types, num_neigh, indices, neigh_types
    )
    oam_cuda_r1_timing.admit_direct_candidate(
        admitted_context, profile, "host", precision
    )
    context_report = admitted_context.execution_plan_report
    assert context_report["selected_id"] == expected_plan
    assert context_report["requested_profile"] == profile
    assert context_report["edge_variant"] == variant
    del admitted_context

    with pytest.raises(
        ValueError,
        match=rf"candidate={expected_plan}.*profile={profile}.*variant={variant}.*precision={precision}.*generated for a different architecture",
    ) as error:
        oam_cuda_r1_timing.prepare_direct_evaluator(
            lambda context: getattr(native, class_name)(
                str(changed_path),
                str(artifact.artifact_path),
                host_active_target=artifact.active_host_target,
                host_artifact_origin=artifact.artifact_origin,
                requested_profile=context["profile"],
                requested_edge_variant=context["variant"],
                requested_precision=context["precision"],
            ),
            profile=profile,
            variant=variant,
            precision=precision,
        )
    assert "manifest architecture fingerprint=" in str(error.value)
    assert "model architecture fingerprint=" in str(error.value)
    assert (
        f"OAM native artifact rejection for requested candidate={expected_plan}, "
        f"profile={profile}, variant={variant}, precision={precision}"
    ) in str(error.value)
