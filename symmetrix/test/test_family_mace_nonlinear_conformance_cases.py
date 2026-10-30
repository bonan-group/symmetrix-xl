"""MH-1 conformance cases: the nonlinear evaluator's UVU program.

This module is where the MH-1 crystal, generalized session model, adapter,
instance specs, and routing case live; the shared suites name no family and
drive whatever the registry collected.  MH-1 declares the narrower program shape
(no reduction node) through its own adapter, and its device tier needs a
generated device module rather than merely a device build, which its runner
requirement states.

Routing parity: this family's float64 energies matched the removed serial
evaluator to ``<= 1.6e-12`` relative and its float64 directed forces to
``<= 4.4e-7`` relative (the golden corpus captured before the deletion), which is
the parity ``ROUTING_TOLERANCE`` gates.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from symmetrix import Symmetrix

from conformance.contract import (
    Adapter,
    CXX_RUNNER,
    FLOAT32_RUNNER,
    Graph,
    Unqualified,
    native_symmetrix,
)
from conformance.registry import (
    FamilyConformance,
    RoutingParityCase,
    crystal,
    register_family,
)
from conformance.surface import _assert_interface_surface, _native

# MH-1's device tier needs a device generated module, not merely a device build:
# the staged host ABI has no device counterpart, so the requirement names NVRTC.
MH1_DEVICE_TIER_RUNNER = (
    "a CUDA build for device generated execution (Kokkos Cuda plus NVRTC), or the "
    "equivalent HIP build"
)


class MH1Adapter(Adapter):
    """MH-1: the nonlinear evaluator, staged ABI-v5 host artifact for T3."""

    family = "MH-1"

    def structure(self) -> Atoms:
        return crystal("mh1")

    def model_path(self) -> Path:
        return self._models["mh1"]

    def _class_name(self) -> str:
        if self.precision == "float32":
            if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
                raise Unqualified("MH-1 float32 execution", FLOAT32_RUNNER)
            return "MACENonlinearKokkosFloat"
        return "MACENonlinearKokkos"

    def _build(self, tier: str):
        if tier == "reference":
            # The float64 evaluator is the oracle for every instance, so a
            # float32 instance is measured against the higher precision.
            evaluator = native_symmetrix.MACENonlinearKokkos(str(self.model_path()))
            evaluator.set_streamed_edges("generic")
            return evaluator
        evaluator_class = getattr(native_symmetrix, self._class_name())
        evaluator = evaluator_class(str(self.model_path()))
        if tier == "portable":
            evaluator.set_streamed_edges("direct")
            return evaluator
        if shutil.which("c++") is None:
            raise Unqualified("the MH-1 host artifact", CXX_RUNNER)
        from symmetrix.jit import prepare_jit_artifact
        from symmetrix.mace_nonlinear_execution_contract import (
            make_execution_mh1_v4_contract,
        )
        from symmetrix.mace_nonlinear_jit_codegen import (
            MH1_HOST_PLUGIN_V7_ABI,
            MH1_HOST_PLUGIN_V7_ABI_VERSION,
            execution_mh1_node_program_metadata,
            render_jit_mh1_host_plugin_v7,
        )

        contract = make_execution_mh1_v4_contract(dict(self._models["mh1_data"]))
        metadata = execution_mh1_node_program_metadata(
            contract, backend="host", node_state_policy="recompute-v1"
        )
        metadata = {
            **metadata,
            "abi": MH1_HOST_PLUGIN_V7_ABI,
            "abi_version": MH1_HOST_PLUGIN_V7_ABI_VERSION,
        }
        prepared = prepare_jit_artifact(
            render_jit_mh1_host_plugin_v7(
                contract, precision=self.precision, node_state_policy="recompute-v1"
            ),
            abi={"tag": metadata["abi"], "version": metadata["abi_version"]},
            build={
                "generator": "symmetrix.jit.mh1-host-v7-composed",
                "precision": self.precision,
                "contract": metadata,
            },
            cache_root=self._models["jit_cache"],
            cxx_flags=None,
            artifact_name="jit_mh1_host_plugin",
        )
        if not prepared.available:
            raise Unqualified("the MH-1 staged host artifact", CXX_RUNNER)
        evaluator._set_execution_mh1_v4_contract_identity(
            contract["tag"],
            contract["generation_fingerprint"],
            contract["semantic_fingerprint"],
            contract["structure_fingerprint"],
            contract["runtime_layout_fingerprint"],
            contract["interaction_count"],
        )
        evaluator.set_streamed_edges("direct")
        evaluator._load_jit_mh1_host_plugin_v7(str(prepared.artifact_path))
        return evaluator

    def bindings_on(self, evaluator) -> dict[str, str]:
        return {
            "prepare_graph": "_prepare_factorized_graph",
            "prepare_geometry": "_prepare_factorized_geometry",
            "compute_prepared": "_compute_prepared_factorized",
            "reduce_forces": "_reduce_atom_forces",
            "reduce_stress": "_reduce_stress",
        }

    def prepare_graph(self, evaluator, graph: Graph) -> int:
        return evaluator._prepare_factorized_graph(*graph.topology())

    def prepare_geometry(self, evaluator, generation: int, graph: Graph) -> None:
        evaluator._prepare_factorized_geometry(
            generation,
            graph.reference_positions,
            graph.xyz,
            graph.cell,
            graph.inverse_cell,
            graph.pbc,
        )

    def compute_prepared(self, evaluator, generation: int, graph: Graph) -> None:
        evaluator._compute_prepared_factorized(generation, graph.xyz, graph.distances)

    def reduce_forces(self, evaluator, generation: int, graph: Graph) -> np.ndarray:
        forces = evaluator._reduce_atom_forces(
            graph.num_nodes, graph.receivers, graph.sources, generation
        )
        return np.asarray(forces, dtype=np.float64).reshape(graph.num_nodes, 3)

    def reduce_stress(self, evaluator, generation: int, graph: Graph):
        # MH-1's compiled program declares no lifecycle reduction node, so its
        # stress is served by the evaluator's own reduction rather than by a
        # registry binding.  The suite asserts that difference explicitly and
        # still validates the reduction numerically against the oracle.
        return np.asarray(
            evaluator._reduce_stress(graph.volume, np.zeros(0), generation),
            dtype=np.float64,
        ).reshape(3, 3)

    def generated_execution_evidence(self, evaluator) -> dict:
        assert self.artifact_ready(evaluator) is True
        backend = str(evaluator.execution_mh1_execution_backend)
        return {
            "artifact_id": evaluator.family_artifact_readiness["artifact_id"],
            "staged_host_backend": backend.startswith("pair_spline_v1_staged"),
            "forward_launches": int(
                evaluator.execution_mh1_generated_forward_launch_count
            ),
        }

    def compiler_free_marker(self) -> tuple[str, ...]:
        # The compiler-free pair-spline Kokkos backend the portable instance
        # reports while no generated plugin is admitted.
        return ("pair_spline_v1_kokkos",)

    def accepted_generation(self, evaluator) -> int:
        return int(evaluator.factorized_graph_generation)

    def prepared_counters(self, evaluator) -> tuple[str, str] | None:
        return (
            "factorized_prepared_graph_count",
            "factorized_prepared_evaluation_count",
        )

    def accelerator_status(self) -> dict:
        status = super().accelerator_status()
        for entry in status.values():
            entry["runner"] = MH1_DEVICE_TIER_RUNNER
        return status


MH1_ORACLE = (
    "the consolidated MH-1 reference: the generic streamed implementation "
    "(`MACENonlinearKokkos` in `generic` mode) whole-evaluator entry point (T0 tier)"
)

MH1_KEYS = frozenset(
    (
        "geometry",
        "r0",
        "harmonics",
        "m0",
        "node_program",
        "readout",
        "pair_energy",
    )
)
MH1_LAYOUTS = frozenset(("edge_major",))
# MH-1's compiled program declares no lifecycle reduction node: the narrower
# program shape the interface review recorded.  Its stress reduction is served
# by the evaluator's own reduction rather than by a registry binding, which the
# suite asserts explicitly instead of assuming every family's program is alike.
MH1_REDUCTIONS = frozenset()
MH1_GENERATED = frozenset(("r0", "m0", "node_program"))

# Each entry is one admitted contract instance.  ``parameters`` states the
# instance's shape, layout, precision, and derivative requirements so a failure
# names the instance rather than the family.
INSTANCE_SPECS: dict[str, dict] = {
    "mh1-f64": dict(
        adapter=MH1Adapter,
        precision="float64",
        parameters="MH-1 UVU plan, edge-major buffers, float64, forces and stress "
        "with no declared reduction node",
        oracle=MH1_ORACLE,
        lifecycle_tiers=("portable", "generated"),
        expected_kinds=MH1_KEYS,
        expected_layouts=MH1_LAYOUTS,
        expected_reductions=MH1_REDUCTIONS,
        generated_kinds=MH1_GENERATED,
        rtol=1e-9,
        atol=1e-8,
    ),
    "mh1-f32": dict(
        adapter=MH1Adapter,
        precision="float32",
        parameters="MH-1 UVU plan, edge-major buffers, float32, forces and stress "
        "with no declared reduction node",
        oracle=MH1_ORACLE,
        lifecycle_tiers=("portable", "generated"),
        expected_kinds=MH1_KEYS,
        expected_layouts=MH1_LAYOUTS,
        expected_reductions=MH1_REDUCTIONS,
        generated_kinds=MH1_GENERATED,
        rtol=1e-3,
        atol=1e-4,
        stress_rtol=2e-3,
        stress_atol=1e-9,
    ),
}


def _mh1_structure() -> Atoms:
    """The MH-1 periodic structure, sized so the skin is real."""
    return Atoms(
        numbers=[14, 1, 14, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [1.7, 0.2, 0.1],
            [0.3, 1.8, 0.4],
            [2.0, 2.1, 0.6],
        ],
        cell=[7.0, 7.0, 7.0],
        pbc=True,
    )


def _mh1_model(root: Path) -> tuple[Path, dict]:
    """Build the generalized MH-1 model once per session.

    Both the model data (the v4 contract is projected from it) and the embedded
    v3 identity contract are returned, so the generated instance can build the
    staged host artifact the same way the calculator does.
    """
    try:
        from test_mh1 import _make_generalized_mh1
    except Exception as exc:  # pragma: no cover - depends on mace-torch
        pytest.skip(f"the MH-1 model builder is unavailable: {exc}")

    import json

    _model, data = _make_generalized_mh1((4, 2, 4, 2, []))
    path = root / "mh1-generalized.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path, data


def _build_models(root: Path) -> dict:
    """The generalized MH-1 model and the v4 contract projected from it."""
    path, data = _mh1_model(root)
    return {
        "mh1": path,
        "mh1_data": data,
        "mh1_contract": data["execution_contracts"]["MH1_UVU"],
    }


def _build_routing_model(request, root: Path) -> Path:
    """The same generalized MH-1 model the conformance suite builds."""
    path, _data = _mh1_model(root)
    return path


def _mh1_routing_calculator(model_path, *, dtype, routing):
    """The shared compact frontend for one routing and precision."""
    return Symmetrix(model_path, dtype=dtype, streamed_edges=routing)


ROUTING_CASE = RoutingParityCase(
    key="MACE_Nonlinear",
    structure="mh1",
    build_model=_build_routing_model,
    calculator=_mh1_routing_calculator,
    # The compact frontend is shared with the other compact family.
    compact_frontend=True,
)


register_family(
    FamilyConformance(
        name="MH-1",
        structures={"mh1": _mh1_structure},
        build_models=_build_models,
        instances=INSTANCE_SPECS,
        # The staged ABI-v5 host artifact is compiled by a host C++20 compiler.
        host_compiler_gated_instances=("mh1-f64", "mh1-f32"),
        routing_parity=ROUTING_CASE,
    )
)


def _mh1_model_path(tmp_path):
    try:
        from test_mh1 import _make_generalized_mh1
    except Exception as error:  # pragma: no cover - missing mace-torch or mace-field
        pytest.skip(f"the MH-1 model builder is unavailable: {error}")
    _model, data = _make_generalized_mh1((4, 2, 4, 2, []))
    path = tmp_path / "mh1-interface.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


def test_mh1_evaluator_satisfies_the_family_interface(tmp_path):
    native = _native()
    if not hasattr(native, "MACENonlinearKokkos"):
        pytest.skip("MACENonlinearKokkos is unavailable in this extension")

    path = _mh1_model_path(tmp_path)
    evaluator = native.MACENonlinearKokkos(str(path))
    if not evaluator.is_mh1_family or not evaluator.uses_mh1_fast_path:
        pytest.skip("the MH-1 fast path is not admitted for this build")
    program = _assert_interface_surface(evaluator)

    capabilities = evaluator.family_capabilities
    assert capabilities["portable"] is True
    assert capabilities["stress"] is True
    assert capabilities["generated"] is False
    assert evaluator.family_artifact_ready is False

    report = evaluator.family_binding_report
    assert report["admitted"] is True
    assert report["portable_node_count"] == report["node_count"]
    assert {operator["capability_key"] for operator in program["operators"]} >= {
        "geometry",
        "r0",
        "harmonics",
        "m0",
        "node_program",
        "readout",
        "pair_energy",
    }
