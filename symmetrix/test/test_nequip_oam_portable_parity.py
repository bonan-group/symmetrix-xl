"""Parity of the compiler-free (T1) Kokkos path against generated receivers.

The portable path is the artifact-free reference routing: the generic Kokkos
edge tensor product evaluated in the selected execution space, which constructs
and evaluates with JIT disabled and no artifact at all.  It evaluates the
certified radial spline table, as do the generated direct routings it is
compared against, so parity is asserted within the existing suite tolerances.

The angular contract is the one the lane-layout parity module builds: it mixes
l=1 and l=2 blocks and merges adjacent same-angular blocks in the post-linear,
so the lane-major to node-major conversions the portable path performs are
actually exercised.  A scalar-only contract would make them vacuous.

The generated side in this module uses receiver-owned ABI v2 and edge-owned ABI
v1 host artifacts, so those cases remain host-only.  CUDA-specific generated
parity lives in ``test_oam_cuda_r1``; artifact-free CUDA execution is covered by
the native and family-conformance suites.
"""

from __future__ import annotations

import json
import re
import shutil

import numpy as np
import pytest

import oam_host_oracle
from test_nequip_oam_l2_parity import _asymmetric_cluster, _build_contract

pytestmark = pytest.mark.oam_runtime


def _native():
    try:
        from symmetrix import symmetrix as native_symmetrix
    except Exception as error:  # pragma: no cover - environment without backend
        pytest.skip(f"native Symmetrix package is unavailable: {error}")
    if not hasattr(native_symmetrix, "NequIPOAMKokkos"):
        pytest.skip("NequIPOAMKokkos is unavailable in this extension")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    return native_symmetrix


def _portable(native, path):
    """The artifact-free reference routing: no host plugin and no CUDA plugin."""

    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the artifact-free OAM reference routing",
    )
    evaluator = native.NequIPOAMKokkos(str(path))
    evaluator.set_streamed_edges("reference")
    return evaluator


def _generated(native, path):
    """The generated direct routing, through its receiver-owned artifact."""

    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact

    artifact = prepare_nequip_oam_receiver_host_artifact(
        path, "float64", cache_dir=path.parent / "cache"
    )
    evaluator = native.NequIPOAMKokkos(
        str(path),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
    evaluator.set_streamed_edges("direct")
    return evaluator


@pytest.fixture(
    scope="module",
    params=(
        pytest.param((1.0, 1.0), id="fixture"),
        pytest.param((1.0, 2.0), id="leading-scaled"),
        pytest.param((3.0, 1.0), id="all-scaled"),
    ),
)
def portable_contract(request, tmp_path_factory):
    """The angular contract with the tensor-product output mask scaled.

    The mask weights the receiver message in the forward, so it must be applied
    exactly once in the reverse; the variants differ in whether they can see a
    second application:

    * ``fixture`` leaves the checked-in ``[1.0 + 0.2 * index]`` mask alone.
      Measured against the reintroduced defect, its forces change by 7e-11 --
      the ordinary parity floor -- so it is a parity baseline for the shipped
      magnitude, not a regression net for the mask.
    * ``leading-scaled`` multiplies the leading mask element (the scalar block,
      whose fixture value is exactly 1.0) by two.  It keeps the contract at the
      fixture's magnitude and, measured, turns a second application into a
      2e-3 absolute force error, far above the parity tolerance.
    * ``all-scaled`` multiplies every element by three, with the same measured
      effect.

    The scalar block is the element that carries the sensitivity: an experiment
    that scaled only the last mask element changed nothing at all.
    """
    from symmetrix.nequip_schema import validate_nequip_oam_data

    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the contract fixture")
    mask_scale, leading_scale = request.param
    contract = _build_contract()
    for layer in contract["layers"]:
        mask = layer["edge_tensor_product"]["output_mask"]
        values = [mask_scale * value for value in mask["values"]]
        values[0] = leading_scale * values[0]
        mask["values"] = values
    validate_nequip_oam_data(contract)
    path = (
        tmp_path_factory.mktemp(f"oam-portable-mask{request.param}")
        / "oam-portable-l2.json"
    )
    path.write_text(json.dumps(contract))
    return path


def test_portable_host_path_matches_the_generated_receiver_without_an_artifact(
    portable_contract,
):
    """The artifact-free reference routing matches the generated direct routing.

    The portable side constructs no artifact at all; the generated receiver
    routing runs the same contract through its prepared artifact.  Both
    evaluate the certified spline, so the two routings must agree on the
    energies and forces of the angular cluster.
    """
    native = _native()
    portable = _portable(native, portable_contract)

    assert portable.family_artifact_ready is False
    assert portable.family_artifact_readiness["artifact_id"] == ""
    capability = portable.family_capabilities
    assert capability["portable"] is True
    assert capability["generated"] is False

    generated = _generated(native, portable_contract)

    graph = _asymmetric_cluster()
    portable.compute_node_energies_forces(*graph)
    generated.compute_node_energies_forces(*graph)

    portable_energy = np.asarray(portable.node_energies, dtype=float)
    generated_energy = np.asarray(generated.node_energies, dtype=float)
    portable_forces = np.asarray(portable.node_forces, dtype=float)
    generated_forces = np.asarray(generated.node_forces, dtype=float)
    # The contract must actually produce angular content, or the parity
    # assertions below would be vacuous.
    assert np.abs(portable_energy).max() > 1e-6
    assert np.abs(portable_forces).max() > 1e-6

    assert "implementation=portable_oam_kokkos_generic" in (
        portable.last_execution_diagnostic
    )
    assert "implementation=receiver_owned_full_message_oam_host_v7" in (
        generated.last_execution_diagnostic
    )
    np.testing.assert_allclose(
        portable_energy, generated_energy, rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(portable_forces, generated_forces, rtol=1e-9, atol=1e-11)


def test_reference_mode_uses_the_portable_path_with_a_legacy_edge_owned_artifact(
    portable_contract,
):
    """A reference request is served by the portable implementation.

    The edge-owned ABI v1 artifact is the explicit `debug_v1` implementation.
    A reference request must run the portable path rather than silently
    executing the legacy edge-owned one.
    """
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_host_artifact

    native = _native()
    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the edge-owned OAM host artifact",
    )
    artifact = prepare_nequip_oam_host_artifact(portable_contract, precision="float64")
    evaluator = native.NequIPOAMKokkos(
        str(portable_contract), str(artifact.artifact_path)
    )
    assert evaluator.jit_oam_host_plugin_ready is True
    evaluator.set_streamed_edges("reference")

    num_nodes, node_types, num_neigh, indices, neigh_types, xyz, distances = (
        _asymmetric_cluster()
    )
    evaluator.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, indices, neigh_types, xyz, distances
    )
    assert "implementation=portable_oam_kokkos_generic" in (
        evaluator.last_execution_diagnostic
    )


def _diagnostic_count(diagnostic, key):
    """Return the integer a native execution diagnostic reports under ``key``."""

    match = re.search(rf"(?:^|[,; ]){re.escape(key)}=(\d+)(?:[,; ]|$)", str(diagnostic))
    assert match is not None, (key, diagnostic)
    return int(match.group(1))


def _edge_bindings(evaluator):
    """Return the tier bindings of the tensor-product nodes of the program."""

    report = evaluator.family_binding_report
    edges = [
        binding
        for binding in report["bindings"]
        if binding["capability_key"] == "edge_tensor_product"
    ]
    assert edges, report["bindings"]
    return report, edges


def test_legacy_edge_owned_artifact_reports_the_tier_it_dispatches(portable_contract):
    """A loaded artifact the dispatcher never runs must not claim a tier.

    The edge-owned ABI v1 artifact serves the explicit ``debug_v1`` mode, so a
    reference request runs the portable implementation instead.  The tier report
    and the dispatch are read from the same state, so the tensor-product nodes
    report the portable implementation that actually serves them; a generated
    tier here would be a report the execution contradicts, which is exactly the
    disagreement this assertion pins.
    """
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_host_artifact

    native = _native()
    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the edge-owned OAM host artifact",
    )
    artifact = prepare_nequip_oam_host_artifact(portable_contract, precision="float64")
    evaluator = native.NequIPOAMKokkos(
        str(portable_contract), str(artifact.artifact_path)
    )
    evaluator.set_streamed_edges("reference")

    report, edges = _edge_bindings(evaluator)
    assert {binding["tier"] for binding in edges} == {"portable"}
    assert {binding["implementation_id"] for binding in edges} == {
        "portable_oam_kokkos_generic"
    }
    assert report["portable_node_count"] == report["node_count"]
    assert report["admitted"] is True

    evaluator.compute_node_energies_forces(*_asymmetric_cluster())
    diagnostic = evaluator.last_execution_diagnostic
    assert "host_plugin_abi=v1" in diagnostic
    assert "implementation=portable_oam_kokkos_generic" in diagnostic
    assert _diagnostic_count(diagnostic, "edge_generated_dispatches") == 0
    assert _diagnostic_count(diagnostic, "edge_generic_dispatches") > 0


def test_legacy_edge_owned_artifact_reports_generated_where_it_dispatches(
    portable_contract,
):
    """The other half of the same either/or: where it does run, it reports so.

    The legacy artifact genuinely serves the edges in the mode it exists for,
    and the report says so — the generated tier is not withheld from an
    artifact that dispatches, it is withheld from one that does not.
    """
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_host_artifact

    native = _native()
    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the edge-owned OAM host artifact",
    )
    artifact = prepare_nequip_oam_host_artifact(portable_contract, precision="float64")
    evaluator = native.NequIPOAMKokkos(
        str(portable_contract), str(artifact.artifact_path)
    )
    evaluator.set_streamed_edges("debug_v1")

    report, edges = _edge_bindings(evaluator)
    assert {binding["tier"] for binding in edges} == {"generated"}
    assert {binding["implementation_id"] for binding in edges} == {
        "generated_oam_host_v1"
    }
    assert report["artifact_ready"] is True

    evaluator.compute_node_energies_forces(*_asymmetric_cluster())
    diagnostic = evaluator.last_execution_diagnostic
    assert "host_plugin_abi=v1" in diagnostic
    assert "implementation=generated_oam_host_v1" in diagnostic
    assert _diagnostic_count(diagnostic, "edge_generated_dispatches") > 0
    assert _diagnostic_count(diagnostic, "edge_generic_dispatches") == 0


def test_portable_host_path_reuses_the_prepared_graph_lifecycle(portable_contract):
    native = _native()
    portable = _portable(native, portable_contract)
    generated = _generated(native, portable_contract)

    num_nodes, node_types, num_neigh, indices, neigh_types, xyz, distances = (
        _asymmetric_cluster()
    )
    positions = np.zeros(3 * num_nodes, dtype=np.float64)

    generation = portable.prepare_graph(
        num_nodes, node_types, num_neigh, indices, neigh_types
    )
    portable.prepare_geometry(
        generation,
        positions,
        xyz,
        np.zeros(0),
        np.zeros(0),
        np.zeros(0, dtype=np.int32),
    )
    portable.compute_prepared(generation, xyz, distances)
    portable.reduce_forces(generation)
    assert portable.graph_generation == generation
    assert portable.geometry_generation == generation

    generated.compute_node_energies_forces(*_asymmetric_cluster())
    generated_forces = np.asarray(generated.node_forces, dtype=float)
    np.testing.assert_allclose(
        np.asarray(portable.node_energies, dtype=float),
        np.asarray(generated.node_energies, dtype=float),
        rtol=1e-10,
        atol=1e-11,
    )
    # The reduced atom forces are this evaluator's own reduction of the same
    # per-edge forces the generated direct routing produces: each directed edge
    # adds its force to its source atom and the opposite to its receiver.
    # Reconstructing that reduction from the generated per-edge forces keeps the
    # comparison on the generated routing as the oracle.
    expected = np.zeros(3 * num_nodes, dtype=float)
    edge = 0
    for receiver in range(num_nodes):
        for _ in range(int(num_neigh[receiver])):
            source = int(indices[edge])
            for component in range(3):
                value = generated_forces[3 * edge + component]
                expected[3 * source + component] += value
                expected[3 * receiver + component] -= value
            edge += 1
    assert edge == len(generated_forces) // 3
    np.testing.assert_allclose(
        np.asarray(portable.atom_forces, dtype=float),
        expected,
        rtol=1e-9,
        atol=1e-11,
    )
