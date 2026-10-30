import json
import gc
import shutil
from pathlib import Path

import numpy as np
import pytest
from compact_r1_model import foundation_r1_test_model
from symmetrix import symmetrix as native_symmetrix
from symmetrix.mace_jit_device_artifact import prepare_jit_device_artifact
from symmetrix.mace_jit_host_artifact import prepare_jit_host_artifact


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"
CONTRACT_ROOT = Path(__file__).resolve().parent / "data" / "execution_contracts"


@pytest.fixture(scope="module")
def compact_model_path(tmp_path_factory):
    r1_contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_mace_omat0_small_contract.json").read_text()
    )
    m0_contract = json.loads((CONTRACT_ROOT / "standard_m0_contract.json").read_text())
    m0_contract["output_l_max"] = 0
    m0_contract["output_components"] = 1
    m0_contract["monomial_groups"] = m0_contract["monomial_groups"][:1]
    m0_contract["structure_fingerprint"] = (
        "sha256:3b4ab2d979488798a05c07b5d54c4c96c1026da3ee1eb93069fd2bacec76ac90"
    )
    m0_contract["semantic_fingerprint"] = (
        "sha256:9fe598ceffb71e61d162c0afbdbe5727a0da7997dbca3a531a46b63603d7af85"
    )
    m0_contract["generation_fingerprint"] = (
        "sha256:f974e4af32dfb328e06fb6655b965c4cc5c0cad50a2013223630f9960f59d398"
    )
    r0_contract = json.loads((CONTRACT_ROOT / "standard_r0_contract.json").read_text())
    model = foundation_r1_test_model(
        r1_contract,
        atomic_numbers=(7, 13),
        m0_contract=m0_contract,
        r0_contract=r0_contract,
    )
    model["A0_scaled"] = True
    model["compact_radial"]["networks"]["A0"] = {
        "shape": [r1_contract["radial_embedding"], 1],
        "weights": [[0.01] * r1_contract["radial_embedding"]],
        "activation": "silu",
        "activation_scale": 1.0,
        "postprocess": "tanh-square",
    }
    output = tmp_path_factory.mktemp("graph-cardinality") / "compact.json"
    output.write_text(json.dumps(model, separators=(",", ":")))
    return output


@pytest.mark.parametrize("evaluator_name", ["MACEKokkosFloat", "MACEKokkos"])
def test_native_graph_cardinality_guard_covers_counts_and_model_strides(
    compact_model_path, evaluator_name
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = getattr(native_symmetrix, evaluator_name)(str(compact_model_path))
    int_max = 2**31 - 1
    channels = json.loads(compact_model_path.read_text())["num_channels"]

    evaluator._validate_graph_cardinality(1_000_188, 1_000_188, 109_020_492)
    maximum_channel_receivers = int_max // channels
    evaluator._validate_graph_cardinality(
        maximum_channel_receivers, maximum_channel_receivers, int_max
    )
    # Flattened edge-channel and harmonic extents may exceed INT32_MAX; only
    # their address arithmetic and byte extent need to remain representable.
    evaluator._validate_graph_cardinality(1, 1, 45_266_828)

    with pytest.raises(ValueError, match="atom count.*32-bit CSR limit"):
        evaluator._validate_graph_cardinality(int_max, int_max, 0)
    with pytest.raises(ValueError, match="directed-edge count.*32-bit topology"):
        evaluator._validate_graph_cardinality(1, 1, int_max + 1)
    with pytest.raises(ValueError, match="receiver-channel launch extent"):
        receiver_count = maximum_channel_receivers + 1
        evaluator._validate_graph_cardinality(receiver_count, receiver_count, 0)
    with pytest.raises(ValueError, match="feature-node-channel launch extent"):
        evaluator._validate_graph_cardinality(1, maximum_channel_receivers + 1, 0)
    with pytest.raises(ValueError, match="smaller than its receiver count"):
        evaluator._validate_graph_cardinality(2, 1, 0)


def test_exact_target_generated_r0_host_plugin_is_admitted(
    compact_model_path, tmp_path, monkeypatch
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    if native_symmetrix._kokkos_default_execution_space() not in ("OpenMP", "Serial"):
        pytest.skip("generated R0 host plugins require a host Kokkos build")
    if shutil.which("c++") is None:
        pytest.skip("generated R0 host plugins require a host C++ compiler")

    from symmetrix.jit import (
        HOST_TARGET_ENVIRONMENT_VARIABLE,
        artifact_cache_inputs,
        prepare_jit_artifact,
        probe_compiler_identity,
        resolve_host_jit_policy,
    )
    from symmetrix.mace_jit_operator_codegen import (
        OPERATOR_ARTIFACT_ENVELOPE,
        render_jit_r0_host_plugin,
    )

    environment = native_symmetrix._execution_device_execution_environment()
    assert environment["backend"] == "host"
    monkeypatch.setenv(HOST_TARGET_ENVIRONMENT_VARIABLE, "x86-64-v3")
    compiler_command, _ = probe_compiler_identity()
    host_policy = resolve_host_jit_policy(compiler_command)
    model_data = json.loads(compact_model_path.read_text())
    source, metadata = render_jit_r0_host_plugin(
        model_data["execution_contracts"]["R0"],
        precision="float32",
        target=host_policy.effective_target,
    )
    artifact = prepare_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"operator": metadata},
        cxx=compiler_command,
        cxx_flags=None,
        host_policy=host_policy,
        cache_root=tmp_path / "r0-cache",
        artifact_name="r0_host_x86_64_v3",
    )
    assert artifact.available, (artifact.reason, artifact.diagnostics)

    evaluator = native_symmetrix.MACEKokkosFloat(str(compact_model_path))
    evaluator._load_r0_host_plugin(
        str(artifact.artifact_path), False, host_policy.effective_target
    )

    assert metadata["generated_identity"]["target"] == host_policy.effective_target
    assert host_policy.effective_target == "host-isa-x86-64-v3"
    assert evaluator.r0_host_plugin_ready
    assert evaluator.r0_host_plugin_artifact_id == metadata["artifact_id"]


def test_calculator_uses_exact_m0_resolution_for_package_mode(
    compact_model_path, tmp_path, monkeypatch
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    if native_symmetrix._kokkos_default_execution_space() not in ("OpenMP", "Serial"):
        pytest.skip("generated R0 host plugins require a host Kokkos build")
    if shutil.which("c++") is None:
        pytest.skip("generated R0 host plugins require a host C++ compiler")

    from symmetrix import Symmetrix
    from symmetrix.jit import (
        CACHE_ENVIRONMENT_VARIABLE,
        HOST_TARGET_ENVIRONMENT_VARIABLE,
    )

    environment = native_symmetrix._execution_device_execution_environment()
    assert environment["backend"] == "host"
    assert environment["architecture"] == "host"
    monkeypatch.setenv(CACHE_ENVIRONMENT_VARIABLE, str(tmp_path / "calculator-cache"))
    monkeypatch.delenv(HOST_TARGET_ENVIRONMENT_VARIABLE, raising=False)

    calculator = Symmetrix(
        compact_model_path,
        dtype="float32",
        streamed_edges="direct",
        low_memory=True,
    )

    try:
        m0 = calculator.jit_operator_modules["M0"]
        if m0["implementation"] == "host_plugin":
            from symmetrix.installed_operator_artifact import (
                installed_operator_build_target,
            )

            assert m0["target"] == installed_operator_build_target(
                native_module=native_symmetrix, backend="host"
            )
            assert m0["target"] != "host"
            assert calculator.evaluator.m0_host_plugin_ready
            assert calculator.evaluator.m0_host_plugin_artifact_id == m0["artifact_id"]
        else:
            # Standard M0 stays compiler-free for off/r0 packages. Full packages
            # may instead take the exact installed host plugin above.
            assert m0["status"] == "builtin"
            assert m0["implementation"] == "builtin"
            assert m0["compiler"] is None
            assert calculator.evaluator.m0_implementation == "builtin"
    finally:
        del calculator
        gc.collect()


def _load_direct_artifact(evaluator, model_path, execution_space, cache_root):
    """Load the generated R1 artifact the build's execution space serves.

    The direct routing is the contract-bound generated plan, and the family
    prepares one artifact per execution space: the host R1 library on a host
    build and the device R1 module on a CUDA/HIP build, each loaded through its
    own entry point.  A device build has no host plugin path at all, so the
    backend decides which adapter is exercised rather than which is preferred.
    """
    if execution_space in ("OpenMP", "Serial"):
        artifact = prepare_jit_host_artifact(
            model_path, precision="float32", cache_root=cache_root
        )
        evaluator._load_jit_host_plugin(str(artifact.artifact_path))
        return
    backend = execution_space.lower()
    artifact = prepare_jit_device_artifact(
        model_path, precision="float32", backend=backend, cache_root=cache_root
    )
    loader = getattr(evaluator, f"_load_jit_{backend}_plugin")
    if backend == "cuda":
        loader(
            str(artifact.artifact_path),
            artifact.edge_policy["persistent_blocks_per_compute_unit"],
        )
    else:
        loader(str(artifact.artifact_path))


def test_prepared_factorized_graph_passes_cardinality_admission(
    compact_model_path, tmp_path
):
    """A valid graph is admitted on the prepared routing and scheduled.

    The prepared (direct) routing is the plan against a contract-bound generated
    R1 artifact, so the graph is admitted only once that artifact is loaded:
    without it the entry point fails closed instead of scheduling a graph.  The
    test therefore drives the routing the way the family conformance cases do,
    preparing the artifact the build's own execution space serves, and
    initializes Kokkos itself because the evaluator aborts without an
    initialized runtime.
    """
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space in ("OpenMP", "Serial") and shutil.which("c++") is None:
        pytest.skip("the compact model's R1 host artifact requires a host C++ compiler")
    evaluator = native_symmetrix.MACEKokkosFloat(str(compact_model_path))
    _load_direct_artifact(
        evaluator, compact_model_path, execution_space, tmp_path / "jit-cache"
    )
    evaluator.set_streamed_edges("direct")
    token = evaluator._prepare_factorized_graph(
        2,
        np.array([0, 0], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
    )

    assert token > 0
    assert evaluator.factorized_graph_generation == token
    assert evaluator.factorized_schedule_entries == 2


def _function_source(source, start, end):
    return source[source.index(start) : source.index(end)]


def test_graph_entry_points_validate_before_allocation_or_narrowing():
    lifecycle = (SOURCE_ROOT / "mace_kokkos_factorized_lifecycle.cpp").read_text()
    generic_prepare = _function_source(
        lifecycle,
        "std::uint64_t MACEKokkos<Precision>::prepare_all_interactions_graph(",
        "void MACEKokkos<Precision>::prepare_all_interactions_geometry(",
    )
    direct_prepare = _function_source(
        lifecycle,
        "std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph(\n"
        "    const int num_receivers,",
        "std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph_device(",
    )
    schedule = _function_source(
        lifecycle,
        "bool MACEKokkos<Precision>::prepare_factorized_schedule_host(\n"
        "    const int num_receivers,",
        "void MACEKokkos<Precision>::prepare_factorized_model(",
    )

    assert generic_prepare.index("validate_graph_cardinality(") < generic_prepare.index(
        "std::vector<int>("
    )
    assert direct_prepare.index("validate_graph_cardinality(") < direct_prepare.index(
        "std::vector<int>("
    )
    assert schedule.index("validate_graph_cardinality(") < schedule.index(
        "static_cast<int>(schedule_neigh_indices.size())"
    )

    evaluate = (SOURCE_ROOT / "mace_kokkos_evaluate.cpp").read_text()
    ordinary = _function_source(
        evaluate,
        "void MACEKokkos<Precision>::compute_node_energies_forces(",
        "void MACEKokkos<Precision>::compute_node_energies_forces_field(",
    )
    field = _function_source(
        evaluate,
        "void MACEKokkos<Precision>::compute_node_energies_forces_field(",
        "void MACEKokkos<Precision>::begin_factorized_distributed_evaluation(",
    )
    distributed = _function_source(
        evaluate,
        "void MACEKokkos<Precision>::begin_factorized_distributed_evaluation(",
        "void MACEKokkos<Precision>::continue_factorized_distributed_evaluation(",
    )
    for entry_point in (ordinary, field, distributed):
        guard = entry_point.index("validate_graph_cardinality(")
        allocation_markers = (
            "Kokkos::realloc(",
            "ensure_execution_result_capacity(",
        )
        allocation = min(
            entry_point.index(marker)
            for marker in allocation_markers
            if marker in entry_point
        )
        assert guard < allocation
        assert guard < entry_point.index("begin_factorized_production_evaluation()")


def test_dual_layer_distributed_tiled_lifecycle_uses_feature_mapping():
    lifecycle = (SOURCE_ROOT / "mace_kokkos_factorized_lifecycle.cpp").read_text()
    evaluate = (SOURCE_ROOT / "mace_kokkos_evaluate.cpp").read_text()
    runtime = (SOURCE_ROOT / "mace_kokkos_runtime.cpp").read_text()

    assert "num_feature_nodes < num_receivers" in runtime
    assert "std::size_t(2), features, outputs, channels" in runtime
    assert "num_feature_nodes < num_receivers" in lifecycle
    assert "receiver feature indices must be unique" in lifecycle
    assert "num_feature_nodes, num_LM, num_channels" in lifecycle

    tiled = _function_source(
        evaluate,
        "void MACEKokkos<Precision>::compute_dual_layer_tiled_phase(\n",
        "void MACEKokkos<Precision>::accumulate_single_layer_tiled_outputs(\n",
    )
    assert "identity_receiver_features = receiver_features.extent(0) == 0" in tiled
    assert tiled.count("receiver_features(global_receiver)") == 4
    assert "compute_dual_layer_tiled_phase1(num_receivers);" in evaluate
    assert "compute_dual_layer_tiled_phase2(num_receivers);" in evaluate
    assert "compute_dual_layer_tiled_phase3(num_receivers);" in evaluate
    assert evaluate.index("compute_dual_layer_tiled_phase1(num_receivers);") < (
        evaluate.index("FactorizedDistributedPhase::prefix_complete")
    )
    assert evaluate.index("compute_dual_layer_tiled_phase2(num_receivers);") < (
        evaluate.index("FactorizedDistributedPhase::middle_complete")
    )


def test_fp64_tiled_geometry_remains_workspace_bounded():
    runtime = (SOURCE_ROOT / "mace_kokkos_runtime.cpp").read_text()
    lifecycle = (SOURCE_ROOT / "mace_kokkos_factorized_lifecycle.cpp").read_text()
    evaluate = (SOURCE_ROOT / "mace_kokkos_evaluate.cpp").read_text()

    reserve = _function_source(
        runtime,
        "void MACEKokkos<Precision>::reserve_execution_geometry_workspace(",
        "void MACEKokkos<Precision>::ensure_execution_result_capacity(",
    )
    assert "3*geometry_edge_capacity" in reserve
    assert "? execution_prepared_unit_direction.extent(0)" in reserve
    assert ": execution_prepared_xyz.extent(0)" in reserve

    geometry = _function_source(
        lifecycle,
        "void MACEKokkos<Precision>::prepare_single_layer_tiled_geometry(",
        "std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph_device(",
    )
    assert "const bool compact_geometry = use_compact_edge_geometry();" in geometry
    assert "explicit_xyz(3*local_edge+component) =" in geometry
    assert "scale*vector[component]" in geometry

    assert "fixed-workspace MPI requires compact edge geometry" not in evaluate
    assert (
        evaluate.count("Kokkos::subview(\n                execution_prepared_xyz") >= 2
    )
    assert "const auto coordinates = single_layer_workspace_xyz;" in evaluate
