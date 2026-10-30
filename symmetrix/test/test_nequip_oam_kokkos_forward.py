from __future__ import annotations

# The explicitly pinned native extension must load before NumPy or Symmetrix.
# ruff: noqa: E402

import ctypes
import json
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
import sys

from explicit_extension import bootstrap_explicit_extension

_bootstrap_explicit_extension = bootstrap_explicit_extension
_bootstrap_explicit_extension()

import numpy as np
import pytest

import oam_host_oracle
from symmetrix.oam_jit_artifact import (
    OAMHostArtifactError,
    prepare_nequip_oam_host_artifact,
    prepare_nequip_oam_receiver_host_artifact,
)
from symmetrix.jit_codegen_common import block_offsets
from symmetrix.oam_jit_codegen import _irrep_blocks
from symmetrix.oam_receiver_codegen import OAMHostGenerationPolicy


FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"
PRODUCTION_JSON = os.environ.get("SYMMETRIX_NEQUIP_OAM_S_JSON")

#: Private per-session cache for the generated host plugins.  Set by the
#: autouse fixture below: compiling into the per-user Execution JIT cache leaks
#: artifacts between sessions and leaves them behind after the run.
_ARTIFACT_CACHE: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _private_artifact_cache(tmp_path_factory):
    global _ARTIFACT_CACHE
    _ARTIFACT_CACHE = tmp_path_factory.mktemp("oam-host-plugins")
    # The Execution JIT cache is a trust boundary: the loader refuses a cache
    # root that is group- or world-accessible.
    _ARTIFACT_CACHE.chmod(0o700)
    return _ARTIFACT_CACHE


def test_explicit_extension_bootstrap_is_idempotent():
    if not os.environ.get("SYMMETRIX_EXTENSION"):
        pytest.skip("explicit-extension qualification environment is not active")
    package = sys.modules["symmetrix"]
    native = sys.modules["symmetrix._native_cpu"]

    _bootstrap_explicit_extension()

    assert sys.modules["symmetrix"] is package
    assert sys.modules["symmetrix._native_cpu"] is native


def test_explicit_extension_bootstrap_accepts_matching_installed_package():
    if not os.environ.get("SYMMETRIX_EXTENSION"):
        pytest.skip("explicit-extension qualification environment is not active")
    package = sys.modules["symmetrix"]
    native = sys.modules["symmetrix._native_cpu"]
    original_package_file = package.__file__
    package.__file__ = str(Path(native.__file__).with_name("__init__.py"))
    try:
        _bootstrap_explicit_extension()
    finally:
        package.__file__ = original_package_file

    assert sys.modules["symmetrix"] is package
    assert sys.modules["symmetrix._native_cpu"] is native


def _native():
    try:
        from symmetrix import symmetrix as native
    except Exception as error:
        pytest.skip(f"native Symmetrix package is unavailable: {error}")
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("OAM Kokkos evaluator is unavailable in this extension")
    if (
        hasattr(native, "_kokkos_is_initialized")
        and not native._kokkos_is_initialized()
    ):
        native._init_kokkos()
    return native


def _require_cxx() -> None:
    if shutil.which(os.environ.get("CXX", "c++")) is None:
        pytest.skip("a C++ compiler is required for the generated OAM host plugins")


@lru_cache(maxsize=None)
def _host_plugin(path: str):
    """Receiver-owned ABI v2 artifact used by the production direct path."""

    _require_cxx()
    try:
        result = prepare_nequip_oam_receiver_host_artifact(
            Path(path), "float64", cache_dir=_ARTIFACT_CACHE
        )
    except OAMHostArtifactError as error:
        # A compiler is present, so this is a codegen/ABI defect rather than an
        # unavailable environment: fail loudly instead of hiding it as a skip.
        raise AssertionError(
            f"generated OAM receiver host plugin failed: {error}"
        ) from error
    return result


@lru_cache(maxsize=None)
def _debug_v1_host_plugin(path: str) -> str:
    """Edge-owned ABI v1 artifact used only by explicit debug_v1 execution."""

    _require_cxx()
    try:
        result = prepare_nequip_oam_host_artifact(
            Path(path), "float64", cache_dir=_ARTIFACT_CACHE
        )
    except OAMHostArtifactError as error:
        raise AssertionError(
            f"generated OAM debug host plugin failed: {error}"
        ) from error
    return str(result.artifact_path)


def _kokkos(native, path=FIXTURE):
    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the receiver-owned OAM host artifact",
    )
    artifact = _host_plugin(str(Path(path).resolve()))
    return native.NequIPOAMKokkos(
        str(path),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )


def _kokkos_debug_v1(native, path=FIXTURE):
    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the edge-owned OAM debug_v1 host artifact",
    )
    evaluator = native.NequIPOAMKokkos(
        str(path), _debug_v1_host_plugin(str(Path(path).resolve()))
    )
    evaluator.set_streamed_edges("debug_v1")
    return evaluator


def _reference(native, path=FIXTURE):
    """The artifact-free reference routing: the portable host implementation.

    ``NequIPOAMKokkos(path)`` with no artifact argument is the portable host
    implementation, selected explicitly by native ``reference`` mode.
    """

    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the artifact-free OAM reference routing",
    )
    evaluator = native.NequIPOAMKokkos(str(path))
    evaluator.set_streamed_edges("reference")
    return evaluator


def _compute(
    evaluator, *, num_neigh=(0, 0), indices=(), neigh_types=(), xyz=(), distances=()
):
    evaluator.compute_node_energies_forces(
        2,
        np.asarray((0, 1), dtype=np.int32),
        np.asarray(num_neigh, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(neigh_types, dtype=np.int32),
        np.asarray(xyz, dtype=np.float64),
        np.asarray(distances, dtype=np.float64),
    )


def test_artifact_free_direct_reports_unavailable_until_reference_is_selected():
    native = _native()
    evaluator = native.NequIPOAMKokkos(str(FIXTURE))
    try:
        assert "state=unavailable" in evaluator.last_execution_diagnostic
        assert "implementation=unavailable_oam_direct" in (
            evaluator.last_execution_diagnostic
        )
        report = evaluator.execution_plan_report
        assert report["state"] == "unqualified"
        assert report["selected_id"] == ""
        with pytest.raises(ValueError, match="requires an admitted generated"):
            evaluator.set_streamed_edges("direct")

        evaluator.set_streamed_edges("reference")
        report = evaluator.execution_plan_report
        assert report["state"] == "active"
        assert report["selected_id"] == "portable_oam_kokkos_generic"
        _compute(evaluator)
        assert "implementation=portable_oam_kokkos_generic" in (
            evaluator.last_execution_diagnostic
        )
    finally:
        del evaluator


def test_oam_kokkos_zero_edge_forward_matches_the_reference_routing():
    native = _native()
    reference = _reference(native)
    kokkos = _kokkos(native)
    _compute(reference)
    _compute(kokkos)
    np.testing.assert_allclose(kokkos.node_energies, reference.node_energies)
    assert np.asarray(kokkos.node_energies).shape == (2,)
    assert np.asarray(kokkos.node_forces).size == 0
    assert np.asarray(kokkos.atom_forces).size == 0
    assert (
        "implementation=receiver_owned_full_message_oam_host_v7"
        in kokkos.last_execution_diagnostic
    )
    assert "artifact_id=" in kokkos.last_execution_diagnostic
    assert "host_plugin_abi=v7" in kokkos.last_execution_diagnostic
    assert "retained_workspace_bytes=" in kokkos.last_execution_diagnostic
    assert "workspace=full-message-control" in kokkos.last_execution_diagnostic
    assert "edge_generic_dispatches=0" in kokkos.last_execution_diagnostic
    assert "generic_kokkos" not in kokkos.last_execution_diagnostic
    assert "radial=spline" in kokkos.last_execution_diagnostic


def test_oam_kokkos_production_like_radial_contract_constructs():
    """Exercise the certified radial table against an exported OAM-S contract.

    A production archive may use a tensor-product mode the streamed-edge
    evaluator does not admit yet; that architecture rejection is tolerated.  A
    radial-certificate, precision, or fingerprint failure is not, and neither is
    any other construction error.  On success the evaluator must actually run
    the certified spline radial table.
    """
    if PRODUCTION_JSON is None:
        pytest.skip("set SYMMETRIX_NEQUIP_OAM_S_JSON to an exported OAM-S contract")
    native = _native()
    try:
        evaluator = _kokkos(native, Path(PRODUCTION_JSON))
    except (RuntimeError, ValueError) as error:
        message = str(error)
        assert not re.search(
            r"fingerprint|radial|certificat", message, re.IGNORECASE
        ), f"certified radial contract rejected by the receiver path: {message}"
        raise AssertionError(
            "exported OAM-S contract did not construct for an unclassified reason: "
            f"{message}"
        ) from error
    _compute(
        evaluator,
        num_neigh=(1, 0),
        indices=(1,),
        neigh_types=(1,),
        xyz=(1.0, 0.0, 0.0),
        distances=(1.0,),
    )
    energies = np.asarray(evaluator.node_energies, dtype=float)
    assert energies.shape == (2,)
    assert np.isfinite(energies).all()
    diagnostic = evaluator.last_execution_diagnostic
    assert "radial=spline" in diagnostic
    assert "implementation=receiver_owned_full_message_oam_host_v7" in diagnostic


@pytest.mark.parametrize("field", ["model_fingerprint", "table_fingerprint"])
def test_oam_kokkos_rejects_a_stale_radial_certificate(tmp_path, field):
    """The receiver path re-derives the certificate instead of trusting it.

    The native loader rebuilds the spline and recomputes both fingerprints (the
    model fingerprint from the contract's exporter fingerprint, the table
    fingerprint from the sampled nodal values), so a certificate that no longer
    matches its radial basis is still schema-valid and must still be refused.
    """

    native = _native()
    from symmetrix.nequip_schema import validate_nequip_oam_data

    contract = json.loads(FIXTURE.read_text())
    contract["layers"][0]["kokkos_radial"][field] = "sha256:" + "0" * 64
    # The mutation keeps the contract schema-valid: only the value is stale.
    validate_nequip_oam_data(contract)
    path = tmp_path / f"stale-{field}.json"
    path.write_text(json.dumps(contract))
    with pytest.raises((RuntimeError, ValueError), match="fingerprint does not match"):
        evaluator = _kokkos(native, path)
        _compute(
            evaluator,
            num_neigh=(1, 0),
            indices=(1,),
            neigh_types=(1,),
            xyz=(1.0, 0.0, 0.0),
            distances=(1.0,),
        )


def test_oam_kokkos_one_edge_forward_matches_the_reference_routing():
    native = _native()
    reference = _reference(native)
    kokkos = _kokkos(native)
    values = dict(
        num_neigh=(1, 0),
        indices=(1,),
        neigh_types=(1,),
        xyz=(1.0, 0.0, 0.0),
        distances=(1.0,),
    )
    _compute(reference, **values)
    _compute(kokkos, **values)
    np.testing.assert_allclose(
        kokkos.node_energies, reference.node_energies, rtol=1e-12, atol=1e-12
    )
    assert np.asarray(kokkos.node_energies).shape == (2,)
    np.testing.assert_allclose(
        kokkos.node_forces, reference.node_forces, rtol=1e-12, atol=1e-12
    )
    assert np.asarray(kokkos.atom_forces).size == 0
    assert "active_edges=1" in kokkos.last_execution_diagnostic
    assert "receiver_chunks=" in kokkos.last_execution_diagnostic
    assert (
        "implementation=receiver_owned_full_message_oam_host_v7"
        in kokkos.last_execution_diagnostic
    )
    assert "generic_kokkos" not in kokkos.last_execution_diagnostic
    assert (
        "tensor_products=receiver_owned_full_message(layer0),receiver_owned_full_message(layer1)"
        in kokkos.last_execution_diagnostic
    )
    assert "host_plugin_abi=v7" in kokkos.last_execution_diagnostic


def test_oam_kokkos_debug_v1_execution_remains_available():
    """The edge-owned ABI v1 path stays selectable for diagnosis only."""
    native = _native()
    reference = _reference(native)
    debug = _kokkos_debug_v1(native)
    values = dict(
        num_neigh=(1, 0),
        indices=(1,),
        neigh_types=(1,),
        xyz=(1.0, 0.0, 0.0),
        distances=(1.0,),
    )
    _compute(reference, **values)
    _compute(debug, **values)
    np.testing.assert_allclose(
        debug.node_energies, reference.node_energies, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        debug.node_forces, reference.node_forces, rtol=1e-12, atol=1e-12
    )
    assert "streamed_edges=debug_v1" in debug.last_execution_diagnostic
    assert "implementation=generated_oam_host_v1" in debug.last_execution_diagnostic
    assert "host_plugin_abi=v1" in debug.last_execution_diagnostic
    assert debug.streamed_edges_mode == "debug_v1"
    assert "auto" not in debug.streamed_edges_mode


def test_oam_generated_reverse_callback_matches_the_generic_adjoint():
    """The edge-owned ABI v1 owner must preserve all three adjoints."""
    native = _native()
    contract = json.loads(FIXTURE.read_text())
    layer = contract["layers"][0]
    metadata = _debug_v1_host_plugin(str(FIXTURE))

    class ReverseArgs(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("layer", ctypes.c_uint32),
            ("scalar_size", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32),
            ("num_edges", ctypes.c_int64),
            ("first_edge", ctypes.c_int64),
            ("samples", ctypes.c_int64),
            ("source_indices", ctypes.POINTER(ctypes.c_int32)),
            ("source_features", ctypes.c_void_p),
            ("edge_features", ctypes.c_void_p),
            ("edge_weights", ctypes.c_void_p),
            ("edge_output_adjoint", ctypes.c_void_p),
            ("source_adjoint", ctypes.c_void_p),
            ("edge_feature_adjoint", ctypes.c_void_p),
            ("edge_weight_adjoint", ctypes.c_void_p),
        ]

    ReverseOwner = ctypes.CFUNCTYPE(None, ctypes.POINTER(ReverseArgs), ctypes.c_int32)

    class Layer(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("layer", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32),
            ("reserved_2", ctypes.c_uint32),
            ("source_dimension", ctypes.c_int32),
            ("edge_dimension", ctypes.c_int32),
            ("weight_dimension", ctypes.c_int32),
            ("output_dimension", ctypes.c_int32),
            ("instruction_count", ctypes.c_int32),
        ]

    class Plugin(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_uint32),
            ("struct_size", ctypes.c_uint32),
            ("pointer_size", ctypes.c_uint32),
            ("byte_order", ctypes.c_uint32),
            ("capabilities", ctypes.c_uint32),
            ("layer_count", ctypes.c_uint32),
            ("scalar_size", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32),
            ("abi_tag", ctypes.c_char_p),
            ("artifact_id", ctypes.c_char_p),
            ("generation_fingerprint", ctypes.c_char_p),
            ("semantic_fingerprint", ctypes.c_char_p),
            ("structure_fingerprint", ctypes.c_char_p),
            ("layers", ctypes.POINTER(Layer)),
            ("forward_owners", ctypes.c_void_p),
            ("reverse_owners", ctypes.POINTER(ReverseOwner)),
        ]

    library = ctypes.CDLL(metadata)
    library.symmetrix_jit_oam_host_plugin_query_v1.restype = ctypes.POINTER(Plugin)
    descriptor = library.symmetrix_jit_oam_host_plugin_query_v1().contents
    reverse_owner = descriptor.reverse_owners[0]

    product = native.E3TensorProduct(json.dumps(layer["edge_tensor_product"]))
    source_dim = product.input_1_dimension
    edge_dim = product.input_2_dimension
    output_dim = product.output_dimension
    weight_dim = layer["edge_tensor_product"]["weight_numel"]
    source_features = np.asarray(
        np.linspace(-0.4, 0.7, 2 * source_dim), dtype=np.float64
    )
    source_indices = np.asarray((1,), dtype=np.int32)
    edge_features = np.asarray(np.linspace(0.2, 0.9, edge_dim), dtype=np.float64)
    edge_weights = np.asarray(np.linspace(0.5, 0.8, weight_dim), dtype=np.float64)
    output_adjoint = np.asarray(np.linspace(-0.3, 0.4, output_dim), dtype=np.float64)
    source_adjoint = np.empty(source_dim, dtype=np.float64)
    edge_adjoint = np.empty(edge_dim, dtype=np.float64)
    weight_adjoint = np.empty(weight_dim, dtype=np.float64)
    args = ReverseArgs(
        ctypes.sizeof(ReverseArgs),
        0,
        ctypes.sizeof(ctypes.c_double),
        0,
        1,
        0,
        1,
        source_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        source_features.ctypes.data,
        edge_features.ctypes.data,
        edge_weights.ctypes.data,
        output_adjoint.ctypes.data,
        source_adjoint.ctypes.data,
        edge_adjoint.ctypes.data,
        weight_adjoint.ctypes.data,
    )
    reverse_owner(ctypes.byref(args), 0)

    expected = product.reverse(
        source_features[source_dim:].tolist(),
        edge_features.tolist(),
        edge_weights.tolist(),
        output_adjoint.tolist(),
    )
    np.testing.assert_allclose(source_adjoint, expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(edge_adjoint, expected[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weight_adjoint, expected[2], rtol=1e-12, atol=1e-12)


def test_oam_kokkos_reports_receiver_owned_workspace():
    """Direct execution must not allocate per-edge chunk workspaces."""
    native = _native()
    evaluator = _kokkos(native)
    _compute(
        evaluator,
        num_neigh=(1, 0),
        indices=(1,),
        neigh_types=(1,),
        xyz=(1.0, 0.0, 0.0),
        distances=(1.0,),
    )
    diagnostic = evaluator.last_execution_diagnostic
    assert "workspace=full-message-control" in diagnostic
    assert "edge-chunk-reused" not in diagnostic
    bytes_match = re.search(r"retained_workspace_bytes=(\d+)", diagnostic)
    assert bytes_match is not None
    assert int(bytes_match.group(1)) < 65536
    assert evaluator.execution_workspace_bytes >= int(bytes_match.group(1))
    assert evaluator.execution_topology_bytes > 0
    assert evaluator.execution_linear_workspace_bytes >= 0
    assert f"topology_bytes={evaluator.execution_topology_bytes}" in diagnostic
    assert (
        "linear_workspace_bytes="
        f"{evaluator.execution_linear_workspace_bytes}" in diagnostic
    )
    del evaluator


def test_oam_full_message_is_the_only_direct_variant():
    native = _native()
    full = _kokkos(native)
    full._set_execution_edge_variant("full_message")
    with pytest.raises(ValueError, match="must be auto or full_message"):
        full._set_execution_edge_variant("projected")
    values = dict(
        num_neigh=(1, 0),
        indices=(1,),
        neigh_types=(1,),
        xyz=(1.0, 0.0, 0.0),
        distances=(1.0,),
    )
    _compute(full, **values)
    assert full.execution_edge_variant == "full_message"
    assert "edge_variant=full_message" in full.last_execution_diagnostic
    assert "receiver_owned_full_message(layer0)" in full.last_execution_diagnostic
    assert "portable_oam" not in full.last_execution_diagnostic
    assert "edge_full_message_launches=4" in full.last_execution_diagnostic
    full_report = full.execution_plan_report
    assert full_report["edge_variant"] == "full_message"
    assert full_report["reverse_owner_schedule"] == "source-owner-v1"
    assert full_report["reduction_policy"] == "owner-local-deterministic-v1"
    assert full_report["launch_count"] == 4
    assert full_report["full_message_launch_count"] == 4


def test_full_message_shared_source_ownership_matches_reference_with_two_workers():
    """Exercise source ownership when multiple receiver rows share source 0."""
    native = _native()
    runtime = native._openmp_runtime_info()
    if runtime.get("kokkos_execution_space") != "OpenMP":
        pytest.skip("the shared-source atomic runtime test requires OpenMP")
    if int(runtime.get("kokkos_concurrency") or 0) < 2:
        pytest.skip(
            "run in a fresh process with OMP_NUM_THREADS=2 and KOKKOS_NUM_THREADS=2"
        )

    node_types = np.asarray((0, 1, 0), dtype=np.int32)
    counts = np.asarray((1, 1, 1), dtype=np.int32)
    sources = np.asarray((1, 0, 0), dtype=np.int32)
    source_types = node_types[sources]
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.2, -0.1), (-0.7, 0.3, -0.4)),
        dtype=np.float64,
    )
    receivers = np.repeat(np.arange(3), counts)
    xyz = np.ascontiguousarray(positions[sources] - positions[receivers])
    distances = np.linalg.norm(xyz, axis=1)

    def evaluate(evaluator, variant=None):
        if variant is not None:
            evaluator._set_execution_edge_variant(variant)
        generation = evaluator.prepare_graph(
            3, node_types, counts, sources, source_types
        )
        evaluator.prepare_geometry(
            generation,
            positions.ravel(),
            xyz.ravel(),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=np.int32),
        )
        evaluator.compute_prepared(generation, xyz.ravel(), distances)
        directed = np.asarray(evaluator.node_forces).copy()
        evaluator.reduce_forces(generation)
        return (
            np.asarray(evaluator.node_energies).copy(),
            directed,
            np.asarray(evaluator.atom_forces).copy(),
        )

    full = evaluate(_kokkos(native), "full_message")
    reference = evaluate(_reference(native))
    for value, expected in zip(full, reference):
        np.testing.assert_allclose(value, expected, rtol=1e-11, atol=1e-11)


def test_host_auto_variant_routes_both_profiles_to_full_message():
    native = _native()
    speed = _kokkos(native)
    capacity = _kokkos(native)
    try:
        speed._set_execution_plan_request("direct", "speed")
        capacity._set_execution_plan_request("direct", "capacity")

        assert speed.execution_edge_variant == "full_message"
        assert capacity.execution_edge_variant == "full_message"
        assert (
            "only supported OAM direct variant" in speed.execution_edge_variant_reason
        )
        assert (
            "only supported OAM direct variant"
            in capacity.execution_edge_variant_reason
        )
    finally:
        del speed, capacity


def test_channel_tiled_host_artifact_routes_full_message_without_fallback():
    native = _native()
    reference = _kokkos(native)
    policy = OAMHostGenerationPolicy(
        owner_schedule="channel_tiled", channel_tile=3, scalar_width=8
    )
    artifact = prepare_nequip_oam_receiver_host_artifact(
        FIXTURE,
        "float64",
        cache_dir=_ARTIFACT_CACHE,
        generation_policy=policy,
    )
    candidate = native.NequIPOAMKokkos(
        str(FIXTURE),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
    candidate._set_execution_edge_variant("full_message")
    reference._set_execution_edge_variant("full_message")
    values = dict(
        num_neigh=(1, 0),
        indices=(1,),
        neigh_types=(1,),
        xyz=(1.0, 0.2, -0.1),
        distances=(np.sqrt(1.05),),
    )
    _compute(candidate, **values)
    _compute(reference, **values)

    np.testing.assert_allclose(
        candidate.node_energies, reference.node_energies, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        candidate.node_forces, reference.node_forces, rtol=1e-11, atol=1e-11
    )
    report = candidate.execution_plan_report
    assert report["forward_owner_schedule"] == "receiver-channel-tiled-owner-v1"
    assert report["reverse_owner_schedule"] == "source-channel-tiled-owner-v1"
    assert report["tiling_policy"] == "channel-tiled-owner-v1"
    assert report["channel_tile"] == 3
    assert report["scalar_width"] == 8
    assert report["required_isa"] == artifact.active_host_target
    assert report["reduction_policy"] == "owner-local-deterministic-v1"
    assert "projection_mode" not in report
    assert report["schedule_qualification"] == "candidate-unqualified-explicit"
    assert report["fallback_count"] == 0


def test_oam_projected_request_is_rejected():
    native = _native()
    evaluator = native.NequIPOAMKokkos(str(FIXTURE))
    with pytest.raises(ValueError, match="must be auto or full_message"):
        evaluator._set_execution_edge_variant("projected")


def test_full_message_arena_includes_post_linear_packed_scratch(tmp_path):
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    layer = contract["layers"][-1]
    post = layer["post_linear"]
    post["irreps_out"] = "8x0e"
    post["instructions"][0]["path_shape"] = [1, 8]
    post["weight"] = {"shape": [8], "values": [1.0] * 8}
    post["bias"] = {"shape": [8], "values": [0.0] * 8}
    post["output_mask"] = {"shape": [8], "values": [1.0] * 8}
    layer["node_irreps_pre_gate"] = "8x0e"
    layer["gate"]["scalar_irreps"] = "8x0e"
    layer["node_irreps_out"] = "8x0e"
    layer["self_connection"] = None
    readout = contract["readout"]
    readout["irreps_in"] = "8x0e"
    readout["instructions"][0]["path_shape"] = [8, 1]
    readout["weight"] = {"shape": [8], "values": [1.0] * 8}
    path = tmp_path / "post-linear-scratch-maximum.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    native = _native()
    full = _kokkos(native, path)
    full._set_execution_edge_variant("full_message")
    arguments = (
        8,
        np.zeros(8, dtype=np.int32),
        np.zeros(8, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    full.compute_node_energies_forces(*arguments)

    assert full.execution_memory_report["arena_packed_linear_workspace_bytes"] > 0
    del full


def test_oam_kokkos_uvw_tensor_product_matches_the_reference_routing(tmp_path):
    """The streamed evaluator must preserve the reference full-path UVW semantics."""
    native = _native()
    contract = json.loads(FIXTURE.read_text())
    instruction = contract["layers"][0]["edge_tensor_product"]["instructions"][0]
    instruction["connection_mode"] = "uvw"
    instruction["path_shape"] = [1, 1, 1]
    path = tmp_path / "oam-uvw.json"
    path.write_text(json.dumps(contract))
    reference = _reference(native, path)
    kokkos = _kokkos(native, path)
    values = dict(
        num_neigh=(1, 0),
        indices=(1,),
        neigh_types=(1,),
        xyz=(1.0, 0.0, 0.0),
        distances=(1.0,),
    )
    _compute(reference, **values)
    _compute(kokkos, **values)
    np.testing.assert_allclose(kokkos.node_energies, reference.node_energies)
    np.testing.assert_allclose(kokkos.node_forces, reference.node_forces)


@pytest.mark.parametrize("mode", ["auto", "generic"])
def test_oam_kokkos_rejects_generic_streamed_edge_modes(mode):
    native = _native()
    evaluator = _kokkos(native)
    with pytest.raises(
        (ValueError, RuntimeError), match="only direct, reference, or debug_v1"
    ):
        evaluator.set_streamed_edges(mode)


def test_oam_kokkos_prepared_reverse_reduction_preserves_atom_force_balance():
    native = _native()
    evaluator = _kokkos(native)
    node_types = np.asarray((0, 1), dtype=np.int32)
    num_neigh = np.asarray((1, 1), dtype=np.int32)
    neigh_indices = np.asarray((1, 0), dtype=np.int32)
    neigh_types = np.asarray((1, 0), dtype=np.int32)
    generation = evaluator.prepare_graph(
        2, node_types, num_neigh, neigh_indices, neigh_types
    )
    evaluator.prepare_geometry(
        generation,
        np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
        np.asarray((1.0, 0.0, 0.0, -1.0, 0.0, 0.0)),
        np.asarray((), dtype=np.float64),
        np.asarray((), dtype=np.float64),
        np.asarray((), dtype=np.int32),
    )
    evaluator.compute_prepared(
        generation,
        np.asarray((1.0, 0.0, 0.0, -1.0, 0.0, 0.0)),
        np.asarray((1.0, 1.0)),
    )
    directed = np.asarray(evaluator.node_forces, dtype=float).reshape(2, 3)
    evaluator.reduce_forces(generation)
    atom_forces = np.asarray(evaluator.atom_forces, dtype=float).reshape(2, 3)
    expected = np.zeros((2, 3), dtype=float)
    expected[1] += directed[0]
    expected[0] -= directed[0]
    expected[0] += directed[1]
    expected[1] -= directed[1]
    np.testing.assert_allclose(atom_forces, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atom_forces.sum(axis=0), 0.0, atol=1e-12)
    assert "operation=reduce_forces" in evaluator.last_execution_diagnostic


def test_oam_kokkos_prepared_lifecycle_rejects_stale_and_incomplete_generations():
    native = _native()
    evaluator = _kokkos(native)
    node_types = np.asarray((0, 1), dtype=np.int32)
    num_neigh = np.asarray((0, 0), dtype=np.int32)
    empty_int = np.asarray((), dtype=np.int32)
    generation = evaluator.prepare_graph(2, node_types, num_neigh, empty_int, empty_int)
    evaluator.prepare_geometry(
        generation,
        np.zeros(6),
        np.asarray((), dtype=np.float64),
        np.asarray((), dtype=np.float64),
        np.asarray((), dtype=np.float64),
        np.asarray((), dtype=np.int32),
    )
    with pytest.raises(
        (ValueError, RuntimeError), match="no completed prepared evaluation"
    ):
        evaluator.reduce_forces(generation)
    stale = evaluator.prepare_graph(2, node_types, num_neigh, empty_int, empty_int)
    assert stale != generation
    with pytest.raises((ValueError, RuntimeError), match="does not match"):
        evaluator.compute_prepared(
            generation,
            np.asarray((), dtype=np.float64),
            np.asarray((), dtype=np.float64),
        )


def _irrep_layout_blocks(irreps):
    """Per-block (offset, multiplicity, width) for an irrep signature."""
    blocks = _irrep_blocks(irreps)
    return [
        (offset, int(block["multiplicity"]), 2 * int(block["l"]) + 1)
        for block, offset in zip(blocks, block_offsets(blocks))
    ]


def _to_lane_major(rows, irreps):
    """Permute node-major rows to lane-major per irrep block.

    Node-major stores element (channel a, component x) of a block at
    ``offset + a*width + x``; lane-major stores it at
    ``offset + x*multiplicity + a``.
    """
    values = np.asarray(rows, dtype=np.float64).copy()
    for offset, multiplicity, width in _irrep_layout_blocks(irreps):
        span = (slice(None), slice(offset, offset + multiplicity * width))
        block = values[span].reshape(values.shape[0], multiplicity, width)
        values[span] = np.swapaxes(block, 1, 2).reshape(values.shape[0], -1)
    return values


def _to_node_major(rows, irreps):
    """Inverse of ``_to_lane_major``: lane-major rows back to node-major."""
    values = np.asarray(rows, dtype=np.float64).copy()
    for offset, multiplicity, width in _irrep_layout_blocks(irreps):
        span = (slice(None), slice(offset, offset + multiplicity * width))
        block = values[span].reshape(values.shape[0], width, multiplicity)
        values[span] = np.swapaxes(block, 1, 2).reshape(values.shape[0], -1)
    return values


def _self_connection_definition():
    """A uvw self connection with internal weights, as exported OAM models use."""
    return json.dumps(
        {
            "irreps_in1": "3x0e+2x1o",
            "irreps_in2": "2x0e",
            "irreps_out": "4x0e+2x1o",
            "instructions": [
                {
                    "i_in1": 0,
                    "i_in2": 0,
                    "i_out": 0,
                    "connection_mode": "uvw",
                    "has_weight": True,
                    "path_weight": 0.5,
                    "path_shape": [3, 2, 4],
                    "wigner_3j": {"shape": [1, 1, 1], "values": [1.0]},
                },
                {
                    "i_in1": 1,
                    "i_in2": 0,
                    "i_out": 1,
                    "connection_mode": "uvw",
                    "has_weight": True,
                    "path_weight": 0.25,
                    "path_shape": [2, 2, 2],
                    "wigner_3j": {
                        "shape": [3, 1, 3],
                        "values": [0.0, 0.3, -0.2, 0.7, 0.0, 0.1, -0.4, 0.5, 0.0],
                    },
                },
            ],
            "weight": {
                "shape": [32],
                "values": [
                    0.01 * (index + 1) * (-1 if index % 3 else 1) for index in range(32)
                ],
            },
            "output_mask": {"shape": [10], "values": [1.0] * 10},
        }
    )


@pytest.mark.oam_runtime
def test_self_connection_bilinear_path_matches_generic_tensor_product():
    native = _native()
    product = native.E3TensorProductKokkos(_self_connection_definition())
    assert product.supports_node_bilinear_uvw
    assert product.uses_mh1_fast_path is False
    generator = np.random.default_rng(20261010)
    input_1 = generator.normal(size=product.input_1_dimension)
    input_2 = generator.normal(size=product.input_2_dimension)
    reference = np.asarray(product.evaluate(input_1, input_2, []))
    optimized = np.asarray(product.evaluate_node_bilinear_uvw(input_1, input_2))
    np.testing.assert_allclose(optimized, reference, rtol=1e-12, atol=1e-12)

    output_adjoint = generator.normal(size=product.output_dimension)
    reference_1, reference_2, _ = product.reverse(input_1, input_2, [], output_adjoint)
    optimized_1, optimized_2 = product.reverse_node_bilinear_uvw(
        input_1, input_2, output_adjoint
    )
    np.testing.assert_allclose(
        optimized_1, np.asarray(reference_1), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        optimized_2, np.asarray(reference_2), rtol=1e-11, atol=1e-12
    )


@pytest.mark.oam_runtime
def test_self_connection_bilinear_multi_sample_tiles_match_generic_tensor_product():
    """Multi-row GEMM tiles must honour per-instruction widths.

    The node-bilinear workspace is sized for the widest instruction of the
    definition.  OAM-M/L self connections mix instructions with different
    path shapes, and narrower instructions must not inherit the wider row
    stride when the workspace is reused.  A single sample cannot expose that
    (every tile is one row), so this regression needs enough samples to fill
    multi-row tiles for both instructions.
    """
    native = _native()
    product = native.E3TensorProductKokkos(_self_connection_definition())
    assert product.supports_node_bilinear_uvw
    generator = np.random.default_rng(20261011)
    dimension_1 = product.input_1_dimension
    dimension_2 = product.input_2_dimension
    dimension_out = product.output_dimension
    samples = 97
    input_1 = generator.normal(size=samples * dimension_1)
    input_2 = generator.normal(size=samples * dimension_2)
    output_adjoint = generator.normal(size=samples * dimension_out)
    optimized = np.asarray(
        product.evaluate_node_bilinear_uvw(input_1, input_2, samples)
    ).reshape(samples, dimension_out)
    optimized_1, optimized_2 = product.reverse_node_bilinear_uvw(
        input_1, input_2, output_adjoint, samples
    )
    optimized_1 = np.asarray(optimized_1).reshape(samples, dimension_1)
    optimized_2 = np.asarray(optimized_2).reshape(samples, dimension_2)
    # Lane-major addressing: permuting both operands (and the output adjoint)
    # into the component-outer layout must produce the same mathematical
    # result once converted back to node-major.
    assert product.supports_node_bilinear_uvw_lane
    definition = json.loads(_self_connection_definition())
    lane_input_1 = _to_lane_major(
        input_1.reshape(samples, dimension_1), definition["irreps_in1"]
    ).ravel()
    lane_input_2 = _to_lane_major(
        input_2.reshape(samples, dimension_2), definition["irreps_in2"]
    ).ravel()
    lane_adjoint = _to_lane_major(
        output_adjoint.reshape(samples, dimension_out), definition["irreps_out"]
    ).ravel()
    lane_optimized = _to_node_major(
        np.asarray(
            product.evaluate_node_bilinear_uvw_lane(lane_input_1, lane_input_2, samples)
        ).reshape(samples, dimension_out),
        definition["irreps_out"],
    )
    lane_1_raw, lane_2_raw = product.reverse_node_bilinear_uvw_lane(
        lane_input_1, lane_input_2, lane_adjoint, samples
    )
    lane_1 = _to_node_major(
        np.asarray(lane_1_raw).reshape(samples, dimension_1),
        definition["irreps_in1"],
    )
    lane_2 = _to_node_major(
        np.asarray(lane_2_raw).reshape(samples, dimension_2),
        definition["irreps_in2"],
    )
    for sample in range(samples):
        one_1 = input_1[sample * dimension_1 : (sample + 1) * dimension_1]
        one_2 = input_2[sample * dimension_2 : (sample + 1) * dimension_2]
        one_adjoint = output_adjoint[
            sample * dimension_out : (sample + 1) * dimension_out
        ]
        reference = np.asarray(product.evaluate(one_1, one_2, []))
        np.testing.assert_allclose(optimized[sample], reference, rtol=1e-12, atol=1e-12)
        reference_1, reference_2, _ = product.reverse(one_1, one_2, [], one_adjoint)
        np.testing.assert_allclose(
            optimized_1[sample], np.asarray(reference_1), rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(
            optimized_2[sample], np.asarray(reference_2), rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(
            lane_optimized[sample], reference, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            lane_1[sample], np.asarray(reference_1), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            lane_2[sample], np.asarray(reference_2), rtol=1e-12, atol=1e-12
        )


def _cube_site_graph(rattle=0.0):
    """A central atom with eight corner neighbors: a perfect O_h site.

    Equivariance forces the central force to vanish exactly, which is the
    cheapest sharp instrument for adjoint defects that single-site parity
    misses.
    """
    directions = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    )
    if rattle:
        directions = directions + np.random.default_rng(5).normal(
            0.0, rattle, directions.shape
        )
    distances = np.linalg.norm(directions, axis=1)
    unit = directions / distances[:, None]
    return dict(
        atom_types=np.array([0] + [1] * 8, dtype=np.int32),
        num_neigh=np.array([8] + [0] * 8, dtype=np.int32),
        indices=np.arange(1, 9, dtype=np.int32),
        neigh_types=np.array([1] * 8, dtype=np.int32),
        xyz=unit.ravel(),
        distances=distances,
    )


def _compute_site(evaluator, graph):
    evaluator.compute_node_energies_forces(
        len(graph["atom_types"]),
        graph["atom_types"],
        graph["num_neigh"],
        graph["indices"],
        graph["neigh_types"],
        np.asarray(graph["xyz"], dtype=np.float64),
        np.asarray(graph["distances"], dtype=np.float64),
    )


def test_oam_kokkos_symmetric_site_produces_zero_central_force():
    """The Kokkos path must preserve the exact O_h symmetry of the site."""
    native = _native()
    reference = _reference(native)
    kokkos = _kokkos(native)
    graph = _cube_site_graph()
    _compute_site(reference, graph)
    _compute_site(kokkos, graph)
    reference_central = np.asarray(reference.node_forces).reshape(-1, 3)[:8].sum(axis=0)
    kokkos_central = np.asarray(kokkos.node_forces).reshape(-1, 3)[:8].sum(axis=0)
    assert np.max(np.abs(reference_central)) < 1e-10
    assert np.max(np.abs(kokkos_central)) < 1e-10


def test_oam_kokkos_rattled_site_forces_match_the_reference_routing():
    """A symmetry-broken site must keep Kokkos forces at reference accuracy."""
    native = _native()
    reference = _reference(native)
    kokkos = _kokkos(native)
    graph = _cube_site_graph(rattle=0.08)
    _compute_site(reference, graph)
    _compute_site(kokkos, graph)
    np.testing.assert_allclose(
        kokkos.node_forces, reference.node_forces, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        kokkos.node_energies, reference.node_energies, rtol=1e-12, atol=1e-12
    )


def test_oam_float32_evaluates_a_certified_float64_export(tmp_path, monkeypatch):
    """A float32 evaluator must load a certified float64 export.

    The radial spline is built and certified in double precision and cast on
    load, so one export serves both evaluator precisions.  This regression
    covers the loader precision gate and the float32 ABI buffer contract
    (the distance adjoint is double-precision in both builds).
    """
    ase = pytest.importorskip("ase")
    from symmetrix import NequIPCalculator

    oam_host_oracle.require_host_execution_space(
        _native()._kokkos_default_execution_space(),
        "the artifact-free OAM reference routing",
    )
    cache = tmp_path / "jit-cache"
    cache.mkdir(mode=0o700)
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(cache))
    atoms = ase.Atoms(
        numbers=[1, 8], positions=[[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]], cell=[8.0] * 3
    )
    atoms.positions = atoms.positions + np.random.default_rng(11).normal(
        0.0, 0.03, atoms.positions.shape
    )
    atoms.calc = NequIPCalculator(
        str(FIXTURE), dtype="float64", streamed_edges="reference"
    )
    atoms.calc.calculate(atoms, properties=("energy", "forces"))
    reference = np.array(atoms.calc.results["forces"]).copy()
    atoms.calc = NequIPCalculator(
        str(FIXTURE), dtype="float32", streamed_edges="direct"
    )
    atoms.calc.calculate(atoms, properties=("energy", "forces"))
    evaluated = np.array(atoms.calc.results["forces"])
    np.testing.assert_allclose(evaluated, reference, rtol=2e-4, atol=2e-4)
