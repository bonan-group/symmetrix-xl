import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.stress import full_3x3_to_voigt_6_stress

from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix
from symmetrix.jit import jit_nvrtc_information


try:
    import torch
    from mace.calculators import MACECalculator
    from mace.tools.scripts_utils import remove_pt_head
    from symmetrix.extract_mace_data import extract_mace_data
except ImportError as exc:
    torch = None
    MACECalculator = None
    remove_pt_head = None
    extract_mace_data = None
    mace_import_error = exc
else:
    mace_import_error = None


MH1_HEADS = (
    "matpes_r2scan",
    "mp_pbe_refit_add",
    "spice_wB97M",
    "oc20_usemppbe",
    "omol",
    "omat_pbe",
)

MH1_GENERATED_FLOAT32_ATOL = 1e-3


def _require_cuda_nvrtc(monkeypatch):
    try:
        information = jit_nvrtc_information()
    except Exception as exc:
        pytest.skip(f"generated CUDA Execution requires NVRTC: {exc}")
    if not information.get("available", False):
        pytest.skip(
            "generated CUDA Execution requires NVRTC: "
            f"{information.get('reason', 'runtime compiler unavailable')}"
        )
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")


def _mh1_streamed_edge_block_size(use_kokkos):
    if use_kokkos and native_symmetrix._kokkos_default_execution_space() == "Cuda":
        return 16384
    return 1024


def _mh1_cross_backend_atol():
    """Full-evaluator serial/Kokkos float64 agreement bound.

    The Kokkos MH-1 fast path evaluates compiled products through its own
    packed pipeline, so host Kokkos backends agree with the serial evaluator
    only at the fast-path product precision instead of float64 rounding (the
    deviation is amplified to ~1e-4 in forces on the repulsive wall); the
    CUDA backend keeps the tighter historical bound.
    """
    if native_symmetrix._kokkos_default_execution_space() == "Cuda":
        return 2e-11
    return 2e-4


def _mh1_sparse_multiblock_graph(block_size, num_nodes, source_shift=0):
    """Build a valid sparse graph whose inactive nodes stress schedule scaling."""
    edge_count = 2 * block_size + 37
    target_count = min(512, num_nodes)
    quotient, remainder = divmod(edge_count, target_count)
    num_neigh = np.zeros(num_nodes, dtype=np.int32)
    num_neigh[:target_count] = quotient
    num_neigh[:remainder] += 1

    edge = np.arange(edge_count, dtype=np.int64)
    active_sources = 17 + 7 * np.arange(31, dtype=np.int32)
    source_indices = active_sources[
        (7 * edge + edge // block_size + source_shift) % len(active_sources)
    ]
    phase = 2.0 * np.pi * (edge % 29) / 29.0
    xyz = np.column_stack(
        (
            1.8 + 0.03 * (edge % 5),
            0.08 * np.cos(phase),
            0.08 * np.sin(phase),
        )
    )
    distances = np.linalg.norm(xyz, axis=1)
    return (
        num_nodes,
        np.zeros(num_nodes, dtype=np.int32),
        num_neigh,
        source_indices,
        np.zeros(edge_count, dtype=np.int32),
        xyz.ravel(),
        distances,
    )


def _mh1_expected_active_sources(source_indices, block_size):
    return sum(
        len(np.unique(source_indices[first_edge : first_edge + block_size]))
        for first_edge in range(0, len(source_indices), block_size)
    )


def _mh1_model_path():
    if mace_import_error is not None:
        pytest.skip(f"mace-torch is not available: {mace_import_error}")
    path = os.environ.get("SYMMETRIX_MH1_MODEL")
    if not path:
        pytest.skip("set SYMMETRIX_MH1_MODEL to run the MACE-MH-1 integration tests")
    path = Path(path)
    if not path.is_file():
        pytest.skip(f"MACE-MH-1 checkpoint does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def mh1_si_artifact(tmp_path_factory):
    data = extract_mace_data(
        _mh1_model_path(),
        head="matpes_r2scan",
    )
    path = tmp_path_factory.mktemp("mh1") / "mh1-si.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return data, path


@pytest.fixture(scope="module")
def mh1_h_si_artifact(tmp_path_factory):
    data = extract_mace_data(
        _mh1_model_path(),
        head="matpes_r2scan",
    )
    path = tmp_path_factory.mktemp("mh1-h-si") / "mh1-h-si.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return data, path


def _make_generalized_mh1(parameters):
    if mace_import_error is not None:
        pytest.skip(f"mace-torch is not available: {mace_import_error}")
    from e3nn import o3
    from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock
    from mace.modules.models import ScaleShiftMACE
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    node_channels, edge_channels, num_bessel, l_max, radial_mlp = parameters
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1701 + node_channels + edge_channels + num_bessel + l_max)
        model = ScaleShiftMACE(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=num_bessel,
            num_polynomial_cutoff=5,
            max_ell=l_max,
            interaction_cls=RealAgnosticResidualNonLinearInteractionBlock,
            interaction_cls_first=RealAgnosticResidualNonLinearInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps(f"{node_channels}x0e+{node_channels}x1o"),
            MLP_irreps=o3.Irreps(f"{max(4, node_channels // 2)}x0e"),
            atomic_energies=np.zeros((1, 2)),
            avg_num_neighbors=4.0,
            atomic_numbers=[1, 14],
            correlation=3,
            gate=torch.nn.functional.silu,
            use_agnostic_product=True,
            edge_irreps=o3.Irreps(f"{edge_channels}x0e+{edge_channels}x1o"),
            use_edge_irreps_first=True,
            radial_MLP=radial_mlp,
            heads=["test"],
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    return model, extract_mace_nonlinear_data(model)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((4, 2, 4, 2, []), id="c4-e2-b4-l2-linear"),
        pytest.param((8, 4, 8, 2, [7, 9]), id="c8-e4-b8-l2"),
        pytest.param((12, 6, 16, 3, [9, 11, 7]), id="c12-e6-b16-l3"),
    ],
)
def generalized_mh1_artifact(request, tmp_path_factory):
    model, data = _make_generalized_mh1(request.param)
    node_channels, edge_channels, _, l_max, radial_mlp = request.param
    if not radial_mlp:
        assert all(
            [layer["type"] for layer in interaction["conv_tp_weights"]["layers"]]
            == ["linear"]
            for interaction in data["interactions"]
        )
    path = (
        tmp_path_factory.mktemp(
            f"mh1-general-c{node_channels}-e{edge_channels}-l{l_max}"
        )
        / "model.json"
    )
    path.write_text(json.dumps(data, separators=(",", ":")))
    return model, data, path


@pytest.fixture(scope="module")
def mh1_256_64_artifact(tmp_path_factory):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("The 256/64 launch-policy qualification requires Kokkos CUDA")
    model, data = _make_generalized_mh1((256, 64, 8, 2, [7, 9]))
    path = tmp_path_factory.mktemp("mh1-general-c256-e64-l2") / "model.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return model, data, path


def _generalized_mh1_atoms():
    return Atoms(
        numbers=[14, 1, 14],
        positions=[[0.0, 0.0, 0.0], [1.7, 0.2, 0.1], [0.3, 1.8, 0.4]],
        cell=[7.0, 7.0, 7.0],
        pbc=True,
    )


def _effective_decimal_digits(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = actual - reference
    reference_norm = np.linalg.norm(reference.ravel())
    relative_l2 = np.linalg.norm(difference.ravel()) / max(
        reference_norm, np.finfo(np.float64).tiny
    )
    scale = np.max(np.abs(reference), initial=0.0)
    significant = np.abs(reference) >= max(scale * 1e-4, 1e-12)
    component_relative = np.abs(difference[significant]) / np.abs(
        reference[significant]
    )
    return {
        "max_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2_error": float(relative_l2),
        "l2_digits": float(-np.log10(max(relative_l2, np.finfo(np.float64).tiny))),
        "minimum_component_digits": float(
            -np.log10(
                max(
                    np.max(component_relative, initial=0.0),
                    np.finfo(np.float64).tiny,
                )
            )
        ),
    }


def _mh1_partial_node_tile_atoms(num_nodes, atomic_numbers=(1, 14)):
    index = np.arange(num_nodes)
    positions = np.column_stack(
        (
            1.15 * (index % 4) + 0.03 * (index % 3),
            1.05 * ((index // 4) % 3) + 0.02 * (index % 5),
            0.95 * (index // 12) + 0.04 * (index % 7),
        )
    )
    return Atoms(
        numbers=np.where(index % 3 == 0, atomic_numbers[0], atomic_numbers[-1]),
        positions=positions,
        cell=[12.0, 11.0, 10.0],
        pbc=False,
    )


def _generated_conditioning_interaction_count(data):
    return sum(
        not interaction["conditioned_mlp"]["factorization"]["prefix_is_identity"]
        for interaction in data["execution_contracts"]["MH1_UVU"]["interactions"]
    )


def _generated_edge_reverse_launch_count(calculator):
    return sum(
        interaction["physical_launch_count"]
        for interaction in calculator.jit_edge_policy["interactions"]
    )


def test_generalized_mh1_native_evaluator_requires_validated_v3_contract(
    generalized_mh1_artifact,
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    from symmetrix.execution_mh1_contract import (
        validate_execution_mh1_contract_for_model,
    )

    _, data, model_path = generalized_mh1_artifact
    evaluator = native_symmetrix.MACENonlinearKokkosFloat(str(model_path))
    assert not evaluator.has_execution_mh1_contract
    contract = validate_execution_mh1_contract_for_model(
        data, data["execution_contracts"]["MH1_UVU"]
    )
    evaluator._set_execution_mh1_contract_identity(
        contract["tag"],
        contract["generation_fingerprint"],
        contract["semantic_fingerprint"],
        contract["structure_fingerprint"],
    )
    assert evaluator.has_execution_mh1_contract


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_generalized_mh1_matches_upstream_and_streamed_modes(
    generalized_mh1_artifact, use_kokkos
):
    if use_kokkos and not hasattr(native_symmetrix, "MACENonlinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    model, data, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]
    upstream = MACECalculator(
        models=model,
        device="cpu",
        default_dtype="float64",
    )
    upstream.calculate(atoms.copy(), properties=properties)
    mode_results = {}
    modes = ("generic",)
    for mode in modes:
        calculator = Symmetrix(
            model_path,
            use_kokkos=use_kokkos,
            dtype="float64",
            streamed_edges=mode,
        )
        evaluator = calculator.evaluator
        assert evaluator.is_mh1_family
        assert evaluator.uses_mh1_fast_path
        assert evaluator.supports_streamed_edges
        assert evaluator.mh1_uses_compiled_products
        assert evaluator.mh1_uses_pair_conditioning
        if use_kokkos:
            assert evaluator.mh1_uses_external_uvu_tensors
            assert evaluator.supports_factorized
        if hasattr(evaluator, "set_mh1_edge_executor"):
            evaluator.set_mh1_edge_executor("mlp_reference")
        assert evaluator.mh1_fast_path_rejection_reason == ""
        assert evaluator.mh1_node_channels == int(
            data["interactions"][0]["node_feats_irreps"].split("x", 1)[0]
        )
        assert evaluator.mh1_edge_channels == int(
            data["interactions"][0]["edge_irreps"].split("x", 1)[0]
        )
        assert evaluator.mh1_radial_size == len(
            data["radial_embedding"]["basis"]["weights"]["values"]
        )
        assert evaluator.mh1_l_max == data["l_max"]
        assert evaluator.streamed_edges_mode == mode
        calculator.calculate(atoms.copy(), properties=properties)
        mode_results[mode] = {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }
    for name in properties:
        np.testing.assert_allclose(
            mode_results["generic"][name],
            upstream.results[name],
            rtol=0.0,
            atol=2e-8,
        )


def test_generalized_mh1_low_memory_retained_interaction_matches_upstream(
    generalized_mh1_artifact,
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    model, _, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]
    upstream = MACECalculator(
        models=model,
        device="cpu",
        default_dtype="float64",
    )
    actual = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        low_memory=True,
    )

    upstream.calculate(atoms.copy(), properties=properties)
    actual.calculate(atoms.copy(), properties=properties)

    evaluator = actual.evaluator
    assert actual.execution_mh1_node_state_policy == "retain-interaction-v1"
    assert actual.jit_node_state_policy == "retain-interaction-v1"
    assert evaluator.mh1_edge_executor == "pair_spline_v1"
    assert all(
        dimension > 0
        for dimension in evaluator.execution_mh1_retained_interaction_output_dimensions
    )
    metrics = {
        name: _effective_decimal_digits(actual.results[name], upstream.results[name])
        for name in properties
    }
    assert metrics["energy"]["max_absolute_error"] / len(atoms) < 1e-4
    assert metrics["energies"]["max_absolute_error"] < 1e-4
    assert metrics["forces"]["max_absolute_error"] < 1e-4
    assert metrics["forces"]["l2_digits"] > 3.0
    assert metrics["forces"]["minimum_component_digits"] > 3.0
    assert metrics["stress"]["max_absolute_error"] < 1e-4


def test_generalized_mh1_kokkos_float32_factorized_matches_float64(
    generalized_mh1_artifact,
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    _, _, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]
    reference = Symmetrix(
        model_path, use_kokkos=True, dtype="float64", streamed_edges="generic"
    )
    actual = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    reference.calculate(atoms.copy(), properties=properties)
    actual.calculate(atoms.copy(), properties=properties)
    assert actual.evaluator.is_mh1_family
    assert actual.evaluator.uses_mh1_fast_path
    assert actual.evaluator.supports_factorized
    assert actual.evaluator.factorized_forward_evaluation_count == 2
    assert actual.evaluator.factorized_reverse_evaluation_count == 2
    assert actual.evaluator.factorized_fallback_evaluation_count == 0
    assert actual.evaluator.factorized_source_owned_reverse
    for name in properties:
        np.testing.assert_allclose(
            actual.results[name],
            reference.results[name],
            rtol=0.0,
            atol=MH1_GENERATED_FLOAT32_ATOL,
        )


def test_generalized_mh1_generated_host_execution_matches_generic(
    generalized_mh1_artifact, monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() not in (
        "Serial",
        "OpenMP",
    ):
        pytest.skip("generated host Execution requires a host Kokkos build")
    if shutil.which("c++") is None:
        pytest.skip("generated host Execution requires a C++20 compiler")

    _, data, model_path = generalized_mh1_artifact
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]

    generic = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    assert generic.evaluator.execution_mh1_execution_backend == "inactive"
    generic.calculate(atoms.copy(), properties=properties)
    expected = {name: np.array(generic.results[name], copy=True) for name in properties}

    generated = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    generated.evaluator.set_mh1_edge_executor("mlp_reference")
    assert generated.jit_status == "built"
    assert generated.evaluator.has_execution_mh1_contract
    assert generated.evaluator.jit_mh1_host_plugin_v4_ready
    assert generated.evaluator.execution_mh1_execution_backend == "generated_host_v4"
    assert generated.evaluator.jit_mh1_host_plugin_artifact_id.startswith(
        "jit-mh1-gen11-v4-"
    )
    generated.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            generated.results[name],
            expected[name],
            rtol=2e-5,
            atol=MH1_GENERATED_FLOAT32_ATOL,
        )
    assert generated.evaluator.execution_mh1_generated_forward_launch_count == 2
    assert generated.evaluator.execution_mh1_generated_source_reverse_launch_count == 2
    assert generated.evaluator.execution_mh1_generated_edge_reverse_launch_count == 2
    expected_conditioning = _generated_conditioning_interaction_count(data)
    assert (
        generated.evaluator.execution_mh1_generated_conditioning_forward_launch_count
        == expected_conditioning
    )
    assert (
        generated.evaluator.execution_mh1_generated_conditioning_reverse_launch_count
        == expected_conditioning
    )
    if expected_conditioning == len(data["interactions"]):
        assert generated.evaluator.conditioned_mlp_workspace_bytes == 0

    retained_results = {
        name: np.array(generated.results[name], copy=True) for name in properties
    }
    retained_edge_workspace = generated.evaluator.edge_workspace_bytes
    assert generated.evaluator.execution_mh1_scratch_recomputed_bytes == 0
    assert generated.evaluator.execution_mh1_scratch_recomputed_layer_count == 0
    assert generated.evaluator.execution_mh1_conditioning_recomputation_count == 0

    bounded = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_mh1_scratch_budget_bytes=0,
    )
    bounded.evaluator.set_mh1_edge_executor("mlp_reference")
    assert bounded.jit_artifact_id == generated.jit_artifact_id
    bounded.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            bounded.results[name], retained_results[name], rtol=2e-5, atol=5e-4
        )
    planner = bounded.evaluator
    assert planner.execution_mh1_scratch_budget_bytes == 0
    assert (
        planner.execution_mh1_scratch_planned_bytes
        >= planner.execution_mh1_scratch_minimum_bytes
    )
    assert len(data["interactions"]) == 2
    assert planner.execution_mh1_scratch_minimum_bytes > 0
    assert (
        planner.execution_mh1_scratch_planned_bytes
        == planner.execution_mh1_scratch_minimum_bytes
    )
    assert planner.execution_mh1_scratch_retained_bytes == 0
    assert planner.execution_mh1_scratch_recomputed_bytes == (
        2 * planner.execution_mh1_scratch_minimum_bytes
    )
    assert planner.execution_mh1_scratch_recomputed_layer_count == 2
    assert not planner.execution_mh1_scratch_budget_satisfied
    assert planner.execution_mh1_conditioning_recomputation_count == 2
    assert planner.execution_mh1_generated_conditioning_forward_launch_count == (
        4 if expected_conditioning else 0
    )
    assert planner.edge_workspace_bytes < retained_edge_workspace
    minimum = planner.execution_mh1_scratch_minimum_bytes
    planner._set_execution_mh1_scratch_budget_bytes(minimum)
    bounded.calculate(atoms.copy(), properties=properties)
    assert planner.execution_mh1_scratch_budget_satisfied
    assert planner.execution_mh1_scratch_planned_bytes == minimum
    assert planner.execution_mh1_scratch_retained_bytes == 0
    assert planner.execution_mh1_scratch_recomputed_layer_count == 2
    assert planner.execution_mh1_conditioning_recomputation_count == 4
    for name in properties:
        np.testing.assert_allclose(
            bounded.results[name],
            retained_results[name],
            rtol=2e-5,
            atol=5e-4,
        )

    planner._set_execution_mh1_scratch_budget_bytes(2 * minimum)
    bounded.calculate(atoms.copy(), properties=properties)
    assert planner.execution_mh1_scratch_budget_satisfied
    assert planner.execution_mh1_scratch_planned_bytes == 2 * minimum
    assert planner.execution_mh1_scratch_retained_bytes == 2 * minimum
    assert planner.execution_mh1_scratch_recomputed_bytes == 0
    assert planner.execution_mh1_scratch_recomputed_layer_count == 0
    assert planner.execution_mh1_conditioning_recomputation_count == 4
    for name in properties:
        np.testing.assert_allclose(
            bounded.results[name],
            retained_results[name],
            rtol=2e-5,
            atol=5e-4,
        )
    with pytest.raises(ValueError, match="non-negative"):
        planner._set_execution_mh1_scratch_budget_bytes(-1)
    full_edge_workspace = planner.edge_workspace_bytes
    planner.set_streamed_edges("generic")
    assert planner.execution_mh1_scratch_budget_bytes == 2 * minimum
    assert planner.execution_mh1_scratch_minimum_bytes == 0
    assert planner.execution_mh1_scratch_planned_bytes == 0
    assert planner.execution_mh1_scratch_retained_bytes == 0
    assert planner.execution_mh1_scratch_recomputed_bytes == 0
    assert planner.edge_workspace_bytes < full_edge_workspace

    planner.set_streamed_edges("direct")
    num_nodes = 4
    empty_indices = np.empty(0, dtype=np.int32)
    planner.compute_node_energies_forces(
        num_nodes,
        np.zeros(num_nodes, dtype=np.int32),
        np.zeros(num_nodes, dtype=np.int32),
        empty_indices,
        empty_indices,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    assert planner.execution_mh1_scratch_minimum_bytes == 0
    assert planner.execution_mh1_scratch_planned_bytes == 0
    assert planner.execution_mh1_scratch_recomputed_layer_count == 0
    assert planner.execution_mh1_scratch_budget_satisfied


def test_generalized_mh1_generated_edges_with_kokkos_nodes_matches_generated(
    generalized_mh1_artifact, monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() not in (
        "Serial",
        "OpenMP",
    ):
        pytest.skip("generated host Execution requires a host Kokkos build")
    if shutil.which("c++") is None:
        pytest.skip("generated host Execution requires a C++20 compiler")

    _, _, model_path = generalized_mh1_artifact
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]
    results = {}
    evaluators = {}
    for node_backend in ("generated", "kokkos"):
        calculator = Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="direct",
            jit="required",
        )
        calculator.evaluator.set_mh1_edge_executor("mlp_reference")
        calculator.evaluator.set_execution_mh1_host_node_backend(node_backend)
        calculator.evaluator.set_e3_linear_backend("packed_gemm")
        calculator.calculate(atoms.copy(), properties=properties)
        results[node_backend] = {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }
        evaluators[node_backend] = calculator.evaluator

    for name in properties:
        np.testing.assert_allclose(
            results["kokkos"][name],
            results["generated"][name],
            rtol=2e-5,
            atol=MH1_GENERATED_FLOAT32_ATOL,
        )
    hybrid = evaluators["kokkos"]
    assert hybrid.execution_mh1_execution_backend == "generated_host_v4_kokkos_nodes"
    assert hybrid.selected_execution_mh1_host_node_backend == "kokkos"
    assert hybrid.execution_mh1_generated_forward_launch_count == 2
    assert hybrid.execution_mh1_generated_source_reverse_launch_count == 2
    assert hybrid.execution_mh1_generated_edge_reverse_launch_count == 2


def test_generalized_mh1_legacy_contract_upgrades_before_host_plugin_load(
    generalized_mh1_artifact, monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() not in (
        "Serial",
        "OpenMP",
    ):
        pytest.skip("generated host Execution requires a host Kokkos build")
    if shutil.which("c++") is None:
        pytest.skip("generated host Execution requires a C++20 compiler")

    from symmetrix.execution_mh1_contract import _legacy_contract_projection

    _, data, _ = generalized_mh1_artifact
    legacy_data = json.loads(json.dumps(data))
    legacy_data["execution_contracts"]["MH1_UVU"] = _legacy_contract_projection(
        data["execution_contracts"]["MH1_UVU"]
    )
    model_path = tmp_path / "legacy-contract-model.json"
    model_path.write_text(json.dumps(legacy_data, separators=(",", ":")))
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))

    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    calculator.evaluator.set_mh1_edge_executor("mlp_reference")
    assert calculator.jit_status == "built"
    assert calculator.evaluator.jit_mh1_host_plugin_v4_ready
    assert calculator.evaluator.execution_mh1_execution_backend == "generated_host_v4"
    assert calculator.evaluator.jit_mh1_host_plugin_artifact_id.startswith(
        "jit-mh1-gen11-v4-"
    )


def test_generalized_mh1_generated_cuda_execution_matches_generic(
    generalized_mh1_artifact, monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("generated CUDA Execution requires a Kokkos CUDA build")
    _require_cuda_nvrtc(monkeypatch)

    _, data, model_path = generalized_mh1_artifact
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]

    generic = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    assert generic.streamed_edges == "generic"
    assert generic.evaluator.streamed_edges_mode == "generic"
    assert generic.evaluator.execution_mh1_execution_backend == "inactive"
    generic.calculate(atoms.copy(), properties=properties)
    expected = {name: np.array(generic.results[name], copy=True) for name in properties}

    generated = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    generated.evaluator.set_mh1_edge_executor("mlp_reference")
    assert generated.jit_status == "built"
    assert generated.jit_compiler_backend == "nvrtc"
    assert generated.jit_artifact_path.endswith(".cubin")
    assert generated.evaluator.jit_mh1_cuda_plugin_v4_ready
    assert generated.evaluator.execution_mh1_execution_backend == "generated_cuda_v4"
    generated.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            generated.results[name], expected[name], rtol=2e-5, atol=5e-4
        )
    assert generated.evaluator.execution_mh1_generated_forward_launch_count == 2
    assert generated.evaluator.execution_mh1_generated_source_reverse_launch_count == 2
    assert (
        generated.evaluator.execution_mh1_generated_edge_reverse_launch_count
        == _generated_edge_reverse_launch_count(generated)
    )
    expected_conditioning = _generated_conditioning_interaction_count(data)
    assert (
        generated.evaluator.execution_mh1_generated_conditioning_forward_launch_count
        == expected_conditioning
    )
    assert (
        generated.evaluator.execution_mh1_generated_conditioning_reverse_launch_count
        == expected_conditioning
    )
    if expected_conditioning == len(data["interactions"]):
        assert generated.evaluator.conditioned_mlp_workspace_bytes == 0

    retained_edge_workspace = generated.evaluator.edge_workspace_bytes
    generated.evaluator._set_execution_mh1_scratch_budget_bytes(0)
    generated.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            generated.results[name], expected[name], rtol=2e-5, atol=5e-4
        )
    planner = generated.evaluator
    assert len(data["interactions"]) == 2
    assert (
        planner.execution_mh1_scratch_planned_bytes
        == planner.execution_mh1_scratch_minimum_bytes
    )
    assert planner.execution_mh1_scratch_retained_bytes == 0
    assert planner.execution_mh1_scratch_recomputed_layer_count == 2
    assert planner.execution_mh1_conditioning_recomputation_count == 2
    assert planner.execution_mh1_generated_conditioning_forward_launch_count == (
        6 if expected_conditioning else 0
    )
    assert planner.edge_workspace_bytes < retained_edge_workspace


@pytest.mark.gpu
@pytest.mark.hip
def test_mh1_generated_hip_dual_node_message_tiles_match_8x32(
    accelerator_capabilities, monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    assert accelerator_capabilities["execution_space"] == "HIP"
    if not accelerator_capabilities["jit_available"]:
        pytest.skip("generated HIP direct execution requires hipRTC")

    extracted_model = os.environ.get("SYMMETRIX_MH1_EXTRACTED_MODEL")
    if extracted_model:
        model_path = Path(extracted_model)
        if not model_path.is_file():
            pytest.skip(f"extracted MACE-MH-1 model does not exist: {model_path}")
        data = json.loads(model_path.read_text())
    else:
        _, data = _make_generalized_mh1((8, 4, 8, 2, [7, 9]))
        model_path = tmp_path / "mh1-hip-partial-node-tiles.json"
        model_path.write_text(json.dumps(data, separators=(",", ":")))
    from symmetrix import mh1_jit_codegen as mh1_codegen

    monkeypatch.setenv("SYMMETRIX_JIT_HIP_JIT_BACKEND", "hiprtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    properties = ["energy", "energies", "forces", "stress"]
    forward_schedule = mh1_codegen._MH1_NODE_FORWARD_SCHEDULE
    reverse_schedule = mh1_codegen._MH1_NODE_REVERSE_SCHEDULE
    reference = None
    candidate = None
    try:
        monkeypatch.setattr(
            mh1_codegen,
            "_CUDA_NODE_MESSAGE_TILE_NODES",
            mh1_codegen._CUDA_NODE_TILE_NODES,
        )
        monkeypatch.setattr(
            mh1_codegen,
            "_MH1_NODE_FORWARD_SCHEDULE",
            forward_schedule + "-test-message-8x32",
        )
        monkeypatch.setattr(
            mh1_codegen,
            "_MH1_NODE_REVERSE_SCHEDULE",
            reverse_schedule + "-test-message-8x32",
        )
        reference = Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="direct",
        )
        reference.evaluator.set_mh1_edge_executor("mlp_reference")
        expected = {}
        for num_nodes in (9, 17):
            atoms = _mh1_partial_node_tile_atoms(num_nodes, data["atomic_numbers"])
            reference.calculate(atoms.copy(), properties=properties)
            expected[num_nodes] = {
                name: np.array(reference.results[name], copy=True)
                for name in properties
            }

        monkeypatch.setattr(
            mh1_codegen,
            "_CUDA_NODE_MESSAGE_TILE_NODES",
            2 * mh1_codegen._CUDA_NODE_TILE_NODES,
        )
        monkeypatch.setattr(mh1_codegen, "_MH1_NODE_FORWARD_SCHEDULE", forward_schedule)
        monkeypatch.setattr(mh1_codegen, "_MH1_NODE_REVERSE_SCHEDULE", reverse_schedule)
        candidate = Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="direct",
        )
        candidate.evaluator.set_mh1_edge_executor("mlp_reference")
        assert candidate.jit_compiler_backend == "hiprtc"
        assert candidate.jit_artifact_path.endswith(".hsaco")
        assert candidate.evaluator.execution_mh1_execution_backend == "generated_hip_v4"
        assert candidate.jit_cache_key != reference.jit_cache_key

        # Asymmetric species and positions give paired logical nodes distinct densities.
        for num_nodes in (9, 17):
            atoms = _mh1_partial_node_tile_atoms(num_nodes, data["atomic_numbers"])
            candidate.calculate(atoms.copy(), properties=properties)
            for name in properties:
                np.testing.assert_array_equal(
                    candidate.results[name], expected[num_nodes][name]
                )
    finally:
        candidate = None
        reference = None
        gc.collect()


def test_generalized_mh1_generated_cuda_multiblock_matches_generic(
    generalized_mh1_artifact, monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("generated CUDA Execution requires a Kokkos CUDA build")
    _require_cuda_nvrtc(monkeypatch)

    _, data, model_path = generalized_mh1_artifact
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    block_size = _mh1_streamed_edge_block_size(True)
    graph = _mh1_sparse_multiblock_graph(block_size, 2048)
    block_count = (len(graph[3]) + block_size - 1) // block_size
    assert block_count == 3

    generic = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    assert generic.streamed_edges == "generic"
    assert generic.evaluator.streamed_edges_mode == "generic"
    assert generic.evaluator.execution_mh1_execution_backend == "inactive"
    generic.evaluator.compute_node_energies_forces(*graph)
    expected_energies = np.array(generic.evaluator.node_energies, copy=True)
    expected_forces = np.array(generic.evaluator.node_forces, copy=True)

    generated = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    generated.evaluator.set_mh1_edge_executor("mlp_reference")
    assert generated.jit_status == "built"
    assert generated.jit_compiler_backend == "nvrtc"
    assert generated.jit_artifact_path.endswith(".cubin")
    assert generated.evaluator.jit_mh1_cuda_plugin_v4_ready
    assert generated.evaluator.execution_mh1_execution_backend == "generated_cuda_v4"
    generated.evaluator.compute_node_energies_forces(*graph)
    actual_energies = np.array(generated.evaluator.node_energies, copy=True)
    actual_forces = np.array(generated.evaluator.node_forces, copy=True)

    np.testing.assert_allclose(actual_energies, expected_energies, rtol=2e-5, atol=5e-4)
    np.testing.assert_allclose(actual_forces, expected_forces, rtol=2e-5, atol=5e-4)
    assert generated.evaluator.execution_mh1_generated_forward_launch_count == 2
    assert generated.evaluator.execution_mh1_generated_source_reverse_launch_count == 2
    assert (
        generated.evaluator.execution_mh1_generated_edge_reverse_launch_count
        == _generated_edge_reverse_launch_count(generated)
    )
    expected_conditioning = _generated_conditioning_interaction_count(data)
    assert (
        generated.evaluator.execution_mh1_generated_conditioning_forward_launch_count
        == expected_conditioning
    )
    assert (
        generated.evaluator.execution_mh1_generated_conditioning_reverse_launch_count
        == expected_conditioning
    )
    if expected_conditioning == len(data["interactions"]):
        assert generated.evaluator.conditioned_mlp_workspace_bytes == 0


def test_generalized_mh1_generated_cuda_is_deterministic_and_accepts_zero_edges(
    generalized_mh1_artifact, monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("generated CUDA Execution requires a Kokkos CUDA build")
    _require_cuda_nvrtc(monkeypatch)

    _, data, model_path = generalized_mh1_artifact
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    calculator.evaluator.set_mh1_edge_executor("mlp_reference")
    evaluator = calculator.evaluator
    assert calculator.jit_compiler_backend == "nvrtc"
    assert calculator.jit_artifact_path.endswith(".cubin")
    assert evaluator.execution_mh1_execution_backend == "generated_cuda_v4"
    block_size = _mh1_streamed_edge_block_size(True)
    graph = _mh1_sparse_multiblock_graph(block_size, 2048)
    different_topology = _mh1_sparse_multiblock_graph(block_size, 2048, source_shift=1)

    evaluator.compute_node_energies_forces(*graph)
    first_energies = np.array(evaluator.node_energies, copy=True)
    first_forces = np.array(evaluator.node_forces, copy=True)
    first_build_count = evaluator.factorized_schedule_build_count
    evaluator.compute_node_energies_forces(*different_topology)
    evaluator.compute_node_energies_forces(*graph)
    second_energies = np.array(evaluator.node_energies, copy=True)
    second_forces = np.array(evaluator.node_forces, copy=True)

    assert evaluator.factorized_schedule_build_count == first_build_count + 2
    np.testing.assert_array_equal(second_energies, first_energies)
    np.testing.assert_array_equal(second_forces, first_forces)
    launch_counts = (
        evaluator.execution_mh1_generated_forward_launch_count,
        evaluator.execution_mh1_generated_source_reverse_launch_count,
        evaluator.execution_mh1_generated_edge_reverse_launch_count,
        evaluator.execution_mh1_generated_conditioning_forward_launch_count,
        evaluator.execution_mh1_generated_conditioning_reverse_launch_count,
    )
    expected_conditioning = 3 * _generated_conditioning_interaction_count(data)
    assert launch_counts[3:] == (
        expected_conditioning,
        expected_conditioning,
    )

    num_nodes = 4
    empty_indices = np.empty(0, dtype=np.int32)
    evaluator.compute_node_energies_forces(
        num_nodes,
        np.zeros(num_nodes, dtype=np.int32),
        np.zeros(num_nodes, dtype=np.int32),
        empty_indices,
        empty_indices,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    assert evaluator.factorized_schedule_entries == 0
    assert evaluator.factorized_schedule_active_sources == 0
    # Empty block-local and graph-wide source CSRs each retain one terminal
    # offset, in addition to the block-segment terminal.
    assert evaluator.factorized_schedule_bytes == 3 * np.dtype(np.int32).itemsize
    assert (
        evaluator.execution_mh1_generated_forward_launch_count,
        evaluator.execution_mh1_generated_source_reverse_launch_count,
        evaluator.execution_mh1_generated_edge_reverse_launch_count,
        evaluator.execution_mh1_generated_conditioning_forward_launch_count,
        evaluator.execution_mh1_generated_conditioning_reverse_launch_count,
    ) == launch_counts
    assert evaluator.execution_mh1_scratch_minimum_bytes == 0
    assert evaluator.execution_mh1_scratch_planned_bytes == 0
    assert evaluator.execution_mh1_scratch_recomputed_layer_count == 0
    assert evaluator.execution_mh1_scratch_budget_satisfied
    assert np.isfinite(np.array(evaluator.node_energies, copy=True)).all()
    assert np.isfinite(np.array(evaluator.node_forces, copy=True)).all()


def test_mh1_generated_cuda_handles_nondivisible_second_subgroup_iteration(
    monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("generated CUDA Execution requires a Kokkos CUDA build")
    _require_cuda_nvrtc(monkeypatch)

    _, data = _make_generalized_mh1((8, 19, 8, 2, [64]))
    contract = data["execution_contracts"]["MH1_UVU"]
    assert [
        interaction["dimensions"]["channels"]
        for interaction in contract["interactions"]
    ] == [19, 19]
    assert [
        interaction["dimensions"]["prefix"] for interaction in contract["interactions"]
    ] == [64, 64]
    model_path = tmp_path / "mh1-c19-phi64.json"
    model_path.write_text(json.dumps(data, separators=(",", ":")))
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]

    generic = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    assert generic.streamed_edges == "generic"
    assert generic.evaluator.streamed_edges_mode == "generic"
    assert generic.evaluator.execution_mh1_execution_backend == "inactive"
    generic.calculate(atoms.copy(), properties=properties)
    expected = {name: np.array(generic.results[name], copy=True) for name in properties}
    generated = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    generated.evaluator.set_mh1_edge_executor("mlp_reference")
    assert generated.jit_compiler_backend == "nvrtc"
    assert generated.jit_artifact_path.endswith(".cubin")
    assert generated.evaluator.execution_mh1_execution_backend == "generated_cuda_v4"
    generated.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            generated.results[name], expected[name], rtol=2e-5, atol=5e-4
        )


def test_mh1_generated_host_multiblock_matches_generic(
    mh1_si_artifact, monkeypatch, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() not in (
        "Serial",
        "OpenMP",
    ):
        pytest.skip("generated host Execution requires a host Kokkos build")
    if shutil.which("c++") is None:
        pytest.skip("generated host Execution requires a C++20 compiler")

    data, model_path = mh1_si_artifact
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    atoms = bulk("Si", "diamond", a=5.43).repeat((4, 4, 4))
    properties = ["energy", "energies", "forces", "stress"]

    generic = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    assert generic.streamed_edges == "generic"
    assert generic.evaluator.streamed_edges_mode == "generic"
    assert generic.evaluator.execution_mh1_execution_backend == "inactive"
    edge_count = len(generic._mace_inputs(atoms)[6])
    assert edge_count > _mh1_streamed_edge_block_size(True)
    generic.calculate(atoms.copy(), properties=properties)
    expected = {name: np.array(generic.results[name], copy=True) for name in properties}

    generated = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    generated.evaluator.set_mh1_edge_executor("mlp_reference")
    assert generated.evaluator.execution_mh1_execution_backend == "generated_host_v4"
    generated.evaluator._set_execution_mh1_node_arena_tile_rows_for_testing(17)
    assert generated.evaluator.execution_mh1_node_arena_tile_rows == 17
    generated.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            generated.results[name], expected[name], rtol=2e-5, atol=5e-4
        )
    assert generated.evaluator.execution_mh1_generated_forward_launch_count == 2
    assert generated.evaluator.execution_mh1_generated_source_reverse_launch_count == 2
    assert generated.evaluator.execution_mh1_generated_edge_reverse_launch_count == 2
    expected_conditioning = _generated_conditioning_interaction_count(data)
    assert (
        generated.evaluator.execution_mh1_generated_conditioning_forward_launch_count
        == expected_conditioning
    )
    assert (
        generated.evaluator.execution_mh1_generated_conditioning_reverse_launch_count
        == expected_conditioning
    )
    assert generated.evaluator.conditioned_mlp_workspace_bytes == 0
    assert generated.evaluator.factorized_schedule_build_count == 1


def test_mh1_generated_host_loader_rejects_fingerprint_mismatch(
    mh1_si_artifact, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() not in (
        "Serial",
        "OpenMP",
    ):
        pytest.skip("generated host Execution requires a host Kokkos build")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("generated host Execution requires a C++20 compiler")

    from symmetrix.jit import prepare_jit_artifact
    from symmetrix.mh1_jit_codegen import (
        render_jit_mh1_host_plugin,
        jit_mh1_host_plugin_metadata,
    )

    data, model_path = mh1_si_artifact
    contract = data["execution_contracts"]["MH1_UVU"]
    metadata = jit_mh1_host_plugin_metadata(contract)
    assert metadata["artifact_id"].startswith("jit-mh1-gen11-v3-")
    source = render_jit_mh1_host_plugin(contract)
    invalid_fingerprint = "sha256:" + "0" * 64
    assert metadata["semantic_fingerprint"] in source
    source = source.replace(metadata["semantic_fingerprint"], invalid_fingerprint, 1)
    result = prepare_jit_artifact(
        source,
        abi={"tag": metadata["abi"], "version": metadata["abi_version"]},
        build={"generator": "mh1-host-negative-test", "contract": metadata},
        cache_root=tmp_path / "jit-cache",
        cxx=compiler,
        cxx_flags=("-ffast-math", "-march=native"),
        cpu="mh1-host-negative-test",
        artifact_name="jit_mh1_host_plugin",
    )
    assert result.available and result.artifact_path is not None
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACENonlinearKokkosFloat(str(model_path))
    evaluator._set_execution_mh1_contract_identity(
        contract["tag"],
        metadata["generation_fingerprint"],
        metadata["semantic_fingerprint"],
        metadata["structure_fingerprint"],
    )
    with pytest.raises(RuntimeError, match="semantic fingerprint does not match"):
        evaluator._load_jit_mh1_host_plugin(str(result.artifact_path))
    assert not evaluator.jit_mh1_host_plugin_ready


def test_mh1_256_node_64_edge_float32_cuda_end_to_end(mh1_256_64_artifact):
    model, _, model_path = mh1_256_64_artifact
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]
    upstream = MACECalculator(
        models=model,
        device="cpu",
        default_dtype="float64",
    )
    actual = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    upstream.calculate(atoms.copy(), properties=properties)
    actual.calculate(atoms.copy(), properties=properties)
    assert actual.evaluator.is_mh1_family
    assert actual.evaluator.mh1_node_channels == 256
    assert actual.evaluator.mh1_edge_channels == 64
    assert actual.evaluator.tensor_product_channel_team_size == 64
    assert actual.evaluator.tensor_product_harmonic_team_size == 64
    for name in properties:
        np.testing.assert_allclose(
            actual.results[name], upstream.results[name], rtol=0.0, atol=5e-4
        )


@pytest.mark.parametrize(
    ("use_kokkos", "dtype"),
    [(False, "float64"), (True, "float64"), (True, "float32")],
)
def test_generalized_mh1_force_matches_finite_difference(
    generalized_mh1_artifact, use_kokkos, dtype
):
    if use_kokkos and not hasattr(native_symmetrix, "MACENonlinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    _, _, model_path = generalized_mh1_artifact
    atoms = Atoms(
        numbers=[14, 1],
        positions=[[0.0, 0.0, 0.0], [1.8, 0.3, 0.2]],
        cell=[7.0, 7.0, 7.0],
        pbc=False,
    )
    atoms.calc = Symmetrix(
        model_path,
        use_kokkos=use_kokkos,
        dtype=dtype,
        streamed_edges="generic",
    )
    force = atoms.get_forces()[1, 0]
    step = 1e-4 if dtype == "float64" else 2e-3
    displaced = atoms.copy()
    displaced.calc = atoms.calc
    displaced.positions[1, 0] += step
    energy_plus = displaced.get_potential_energy()
    displaced.positions[1, 0] -= 2 * step
    energy_minus = displaced.get_potential_energy()
    tolerance = 2e-5 if dtype == "float64" else 3e-3
    assert force == pytest.approx(
        -(energy_plus - energy_minus) / (2 * step), abs=tolerance
    )


@pytest.mark.parametrize(
    ("mutation", "reason", "construction_error"),
    [
        ("correlation", "correlation-three", "invalid correlation data"),
        ("tensor_mode", "external weighted uvu", "uuu path"),
        ("l_max", "l_max=2 or l_max=3", None),
    ],
)
def test_generalized_mh1_family_rejects_unsupported_semantics(
    generalized_mh1_artifact, tmp_path, mutation, reason, construction_error
):
    _, data, _ = generalized_mh1_artifact
    changed = json.loads(json.dumps(data))
    if mutation == "correlation":
        changed["products"][0]["symmetric_contractions"]["contractions"][0][
            "correlation"
        ] = 2
    elif mutation == "tensor_mode":
        changed["interactions"][0]["conv_tp"]["instructions"][0]["connection_mode"] = (
            "uuu"
        )
    else:
        changed["l_max"] = 4
    path = tmp_path / f"unsupported-{mutation}.json"
    path.write_text(json.dumps(changed, separators=(",", ":")))
    if construction_error is not None:
        with pytest.raises(ValueError, match=construction_error):
            native_symmetrix.MACENonlinear(str(path))
        return
    evaluator = native_symmetrix.MACENonlinear(str(path))
    assert not evaluator.is_mh1_family
    assert not evaluator.uses_mh1_fast_path
    assert not evaluator.supports_streamed_edges
    assert reason in evaluator.mh1_family_rejection_reason
    assert reason in evaluator.mh1_fast_path_rejection_reason
    with pytest.raises(ValueError, match="compatible MACE-MH-1 family"):
        evaluator.set_streamed_edges("generic")
    if hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        with pytest.raises(ValueError, match="Float32.*compatible MACE-MH-1 family"):
            native_symmetrix.MACENonlinearKokkosFloat(str(path))


def test_mh1_rejects_malformed_nonlinear_json(tmp_path):
    path = tmp_path / "malformed-nonlinear.json"
    path.write_text(
        json.dumps(
            {
                "symmetrix_format_version": 3,
                "model_type": "MACE_Nonlinear",
            }
        )
    )
    with pytest.raises((RuntimeError, ValueError), match="node_embedding"):
        native_symmetrix.MACENonlinear(str(path))
    if hasattr(native_symmetrix, "MACENonlinearKokkos"):
        with pytest.raises((RuntimeError, ValueError), match="node_embedding"):
            native_symmetrix.MACENonlinearKokkos(str(path))


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_rejects_invalid_species_mapping_before_evaluation(tmp_path, use_kokkos):
    linear = {
        "irreps_in": "1x0e",
        "irreps_out": "1x0e",
        "instructions": [
            {
                "i_in": 0,
                "i_out": 0,
                "path_weight": 1.0,
                "path_shape": [1, 1],
            }
        ],
        "weight": {"shape": [1], "values": [1.0]},
        "bias": {"shape": [0], "values": []},
        "output_mask": {"shape": [1], "values": [1.0]},
    }
    path = tmp_path / "invalid-mapping.json"
    path.write_text(
        json.dumps(
            {
                "symmetrix_format_version": 3,
                "model_type": "MACE_Nonlinear",
                "node_embedding": linear,
                "atomic_numbers": [1],
                "model_atomic_numbers": [1],
                "model_indices": [2],
                "num_elements": 1,
            }
        )
    )
    if use_kokkos:
        if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        evaluator = native_symmetrix.MACENonlinearKokkos
    else:
        evaluator = native_symmetrix.MACENonlinear
    with pytest.raises(ValueError, match="model_indices"):
        evaluator(str(path))


def test_mh1_full_schema_preserves_dynamic_architecture(mh1_si_artifact):
    data, _ = mh1_si_artifact
    assert data["symmetrix_format_version"] == 3
    assert data["model_type"] == "MACE_Nonlinear"
    assert data["head"] == "matpes_r2scan"
    assert data["atomic_numbers"] == data["model_atomic_numbers"]
    assert data["model_indices"] == list(range(len(data["model_atomic_numbers"])))
    assert len(data["model_atomic_numbers"]) == 89
    assert len(data["interactions"]) == data["num_interactions"]
    assert len(data["products"]) == data["num_interactions"]
    assert len(data["readouts"]) == data["num_interactions"]
    assert all(
        interaction["class"] == "RealAgnosticResidualNonLinearInteractionBlock"
        for interaction in data["interactions"]
    )
    assert all(
        interaction["conv_tp_weights"]["layers"] for interaction in data["interactions"]
    )
    assert all(
        interaction["density_fn"]["layers"] for interaction in data["interactions"]
    )
    hidden_irreps = data["readouts"][-1]["linear_1"]["irreps_out"]
    assert hidden_irreps == data["readouts"][-1]["linear_2"]["irreps_in"]
    assert hidden_irreps.endswith("x0e")
    assert (
        max(
            contraction["correlation"]
            for product in data["products"]
            for contraction in product["symmetric_contractions"]["contractions"]
        )
        >= 3
    )


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_generic_mode_is_default_for_float64(mh1_si_artifact, use_kokkos):
    _, model_path = mh1_si_artifact
    if use_kokkos and not hasattr(native_symmetrix, "MACENonlinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    properties = ["energy", "energies", "forces", "stress"]
    workspace_rows = {}
    edge_count = None
    modes = ("generic",)
    for mode in modes:
        calculator = Symmetrix(
            model_path,
            use_kokkos=use_kokkos,
            dtype="float64",
            streamed_edges=mode,
        )
        assert calculator.evaluator.uses_mh1_fast_path
        assert calculator.evaluator.supports_streamed_edges
        assert calculator.evaluator.streamed_edges_mode == mode
        calculator.calculate(atoms.copy(), properties=properties)
        if edge_count is None:
            edge_count = len(calculator._mace_inputs(atoms)[6])
        workspace_rows[mode] = calculator.evaluator.edge_workspace_rows
    assert workspace_rows["generic"] <= min(
        edge_count, _mh1_streamed_edge_block_size(use_kokkos)
    )
    automatic = Symmetrix(model_path, use_kokkos=use_kokkos, dtype="float64")
    assert automatic.streamed_edges == ("generic" if not use_kokkos else "direct")


@pytest.mark.parametrize(
    "use_kokkos,dtype",
    [(False, "float64"), (True, "float64"), (True, "float32")],
)
def test_mh1_rejects_materialized_edge_mode(
    generalized_mh1_artifact, use_kokkos, dtype
):
    evaluator_name = (
        "MACENonlinear"
        if not use_kokkos
        else (
            "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
        )
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without the requested evaluator")
    _, _, model_path = generalized_mh1_artifact
    message = (
        "supports only.*generic.*direct" if use_kokkos else "supports only.*generic"
    )
    default_mode = "direct" if use_kokkos else "generic"

    with pytest.raises(ValueError, match=message):
        Symmetrix(
            model_path,
            use_kokkos=use_kokkos,
            dtype=dtype,
            streamed_edges="materialized",
        )

    evaluator = getattr(native_symmetrix, evaluator_name)(str(model_path))
    assert evaluator.streamed_edges_mode == default_mode
    with pytest.raises(ValueError, match=message):
        evaluator.set_streamed_edges("materialized")
    assert evaluator.streamed_edges_mode == default_mode


@pytest.mark.parametrize(
    "use_kokkos,dtype",
    [(False, "float64"), (True, "float64"), (True, "float32")],
)
def test_mh1_native_evaluator_rejects_removed_second_interaction(
    generalized_mh1_artifact, use_kokkos, dtype
):
    evaluator_name = (
        "MACENonlinear"
        if not use_kokkos
        else (
            "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
        )
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without the requested evaluator")
    _, _, model_path = generalized_mh1_artifact
    evaluator = getattr(native_symmetrix, evaluator_name)(str(model_path))

    with pytest.raises(ValueError, match="materialized.*generic.*direct"):
        evaluator.set_streamed_edges("second_interaction")
    assert evaluator.streamed_edges_mode == ("direct" if use_kokkos else "generic")


def test_mh1_serial_rejects_direct(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    evaluator = native_symmetrix.MACENonlinear(str(model_path))
    with pytest.raises(
        ValueError,
        match="direct execution currently requires the Kokkos evaluator",
    ):
        evaluator.set_streamed_edges("direct")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_pair_spline_executor_matches_mlp_reference(
    generalized_mh1_artifact, dtype
):
    _, _, model_path = generalized_mh1_artifact
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    evaluator = calculator.evaluator
    assert evaluator.mh1_pair_spline_ready
    assert evaluator.mh1_edge_executor == "pair_spline_v1"

    with pytest.raises(ValueError, match="mlp_reference.*pair_spline_v1"):
        evaluator.set_mh1_edge_executor("unknown")
    assert evaluator.mh1_edge_executor == "pair_spline_v1"

    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]
    evaluator.set_mh1_edge_executor("mlp_reference")
    calculator.calculate(atoms.copy(), properties=properties)
    reference = {
        name: np.array(calculator.results[name], copy=True) for name in properties
    }

    evaluator.set_mh1_edge_executor("pair_spline_v1")
    assert evaluator.mh1_edge_executor == "pair_spline_v1"
    assert calculator.streamed_edges == "generic"
    calculator.calculate(atoms.copy(), properties=properties)
    for name in properties:
        tolerance = 5e-5 if name in ("energy", "energies") else 5e-4
        np.testing.assert_allclose(
            calculator.results[name], reference[name], rtol=0.0, atol=tolerance
        )

    evaluator.set_mh1_edge_executor("mlp_reference")
    assert evaluator.mh1_edge_executor == "mlp_reference"


def test_mh1_device_spline_direct_matches_generated(generalized_mh1_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() not in ("Cuda", "HIP"):
        pytest.skip("Accelerator-only MH-1 staged spline regression")
    _, _, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    atoms.positions[1] += [0.013, -0.021, 0.008]
    properties = ["energy", "energies", "forces", "stress"]
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    evaluator = calculator.evaluator
    assert evaluator.jit_mh1_cuda_plugin_v4_ready
    calculator.calculate(atoms.copy(), properties=properties)
    reference = {
        name: np.array(calculator.results[name], copy=True) for name in properties
    }
    edge_count = len(calculator._mace_inputs(atoms)[6])

    evaluator.set_mh1_edge_executor("pair_spline_v1")
    assert evaluator.execution_mh1_execution_backend == (
        "pair_spline_v1_staged_device_v5"
    )
    edge_workspace_before = evaluator.edge_workspace_bytes
    conditioning_before = (
        evaluator.execution_mh1_generated_conditioning_forward_launch_count,
        evaluator.execution_mh1_generated_conditioning_reverse_launch_count,
    )
    fallback_before = evaluator.factorized_fallback_evaluation_count
    calculator.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            calculator.results[name], reference[name], rtol=0.0, atol=5e-4
        )
    assert (
        evaluator.execution_mh1_generated_conditioning_forward_launch_count,
        evaluator.execution_mh1_generated_conditioning_reverse_launch_count,
    ) == conditioning_before
    assert evaluator.factorized_fallback_evaluation_count == fallback_before
    spline_metadata_bytes = edge_count * (
        np.dtype(np.intc).itemsize + np.dtype(np.float32).itemsize
    )
    assert evaluator.edge_workspace_bytes <= (
        edge_workspace_before + spline_metadata_bytes
    )


@pytest.mark.parametrize("dtype", ["float32"])
def test_mh1_kokkos_direct_matches_generic_and_reports_execution(
    mh1_si_artifact, dtype
):
    evaluator_name = (
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without the requested Kokkos precision")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    properties = ["energy", "energies", "forces", "stress"]
    results = {}
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
    )
    evaluator = calculator.evaluator
    assert evaluator.supports_factorized
    assert evaluator.factorized_forward_evaluation_count == 0
    assert evaluator.factorized_reverse_evaluation_count == 0
    assert evaluator.factorized_fallback_evaluation_count == 0
    assert evaluator.factorized_schedule_build_count == 0
    assert not evaluator.factorized_source_owned_reverse

    for mode in ("generic", "direct"):
        evaluator.set_streamed_edges(mode)
        assert evaluator.streamed_edges_mode == mode
        counters_before = (
            evaluator.factorized_forward_evaluation_count,
            evaluator.factorized_reverse_evaluation_count,
            evaluator.factorized_fallback_evaluation_count,
            evaluator.factorized_schedule_build_count,
        )
        calculator.calculate(atoms.copy(), properties=properties)
        results[mode] = {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }
        if mode == "direct":
            assert evaluator.factorized_forward_evaluation_count == (
                counters_before[0] + 2
            )
            assert evaluator.factorized_reverse_evaluation_count == (
                counters_before[1] + 2
            )
            assert evaluator.factorized_fallback_evaluation_count == 0
            assert evaluator.factorized_schedule_build_count > counters_before[3]
            assert evaluator.factorized_source_owned_reverse
        else:
            assert (
                evaluator.factorized_forward_evaluation_count,
                evaluator.factorized_reverse_evaluation_count,
                evaluator.factorized_fallback_evaluation_count,
                evaluator.factorized_schedule_build_count,
            ) == counters_before
            assert not evaluator.factorized_source_owned_reverse

    tolerance = 2e-12 if dtype == "float64" else 2e-5
    for mode in ("direct",):
        for name in properties:
            np.testing.assert_allclose(
                results[mode][name],
                results["generic"][name],
                rtol=0.0,
                atol=tolerance,
            )
    evaluator.set_streamed_edges("generic")
    assert not evaluator.factorized_source_owned_reverse


def test_mh1_execution_source_schedule_scales_with_edges_and_active_sources(
    generalized_mh1_artifact,
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    _, _, model_path = generalized_mh1_artifact
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    evaluator = calculator.evaluator
    block_size = _mh1_streamed_edge_block_size(True)
    schedule_metrics = []

    for num_nodes in (512, 4096):
        graph = _mh1_sparse_multiblock_graph(block_size, num_nodes)
        evaluator.compute_node_energies_forces(*graph)
        edge_count = len(graph[3])
        block_count = (edge_count + block_size - 1) // block_size
        active_sources = _mh1_expected_active_sources(graph[3], block_size)
        graph_active_sources = len(np.unique(graph[3]))
        graph_receivers = np.count_nonzero(graph[2])
        index_bytes = np.dtype(np.int32).itemsize
        expected_bytes = index_bytes * (
            block_count
            + active_sources
            + graph_active_sources
            + graph_receivers
            + 3 * edge_count
            + 3
        )

        assert evaluator.factorized_schedule_entries == edge_count
        assert evaluator.factorized_schedule_active_sources == active_sources
        assert evaluator.factorized_schedule_bytes == expected_bytes
        # Three edge-sized index arrays plus compact ownership metadata remain
        # much smaller than one blocks-by-nodes schedule.
        assert expected_bytes - 3 * index_bytes * edge_count < (
            index_bytes * block_count * num_nodes
        )
        schedule_metrics.append((active_sources, expected_bytes))

    # Adding thousands of unused nodes must not grow a source schedule whose
    # edge topology and per-block active-source set are unchanged.
    assert schedule_metrics[1] == schedule_metrics[0]


def test_mh1_execution_source_schedule_rebuild_is_bitwise_deterministic(
    generalized_mh1_artifact,
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    _, _, model_path = generalized_mh1_artifact
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    evaluator = calculator.evaluator
    block_size = _mh1_streamed_edge_block_size(True)
    graph = _mh1_sparse_multiblock_graph(block_size, 2048)
    different_topology = _mh1_sparse_multiblock_graph(block_size, 2048, source_shift=1)

    evaluator.compute_node_energies_forces(*graph)
    first_energies = np.array(evaluator.node_energies, copy=True)
    first_forces = np.array(evaluator.node_forces, copy=True)
    first_build_count = evaluator.factorized_schedule_build_count

    evaluator.compute_node_energies_forces(*different_topology)
    evaluator.compute_node_energies_forces(*graph)
    assert evaluator.factorized_schedule_build_count == first_build_count + 2
    second_energies = np.array(evaluator.node_energies, copy=True)
    second_forces = np.array(evaluator.node_forces, copy=True)

    assert np.isfinite(second_energies).all()
    assert np.isfinite(second_forces).all()
    np.testing.assert_array_equal(second_energies, first_energies)
    np.testing.assert_array_equal(second_forces, first_forces)


def test_mh1_execution_source_schedule_accepts_zero_edges(
    generalized_mh1_artifact,
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    _, _, model_path = generalized_mh1_artifact
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    evaluator = calculator.evaluator
    num_nodes = 4
    empty_indices = np.empty(0, dtype=np.int32)
    evaluator.compute_node_energies_forces(
        num_nodes,
        np.zeros(num_nodes, dtype=np.int32),
        np.zeros(num_nodes, dtype=np.int32),
        empty_indices,
        empty_indices,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )

    assert evaluator.factorized_schedule_entries == 0
    assert evaluator.factorized_schedule_active_sources == 0
    assert evaluator.factorized_schedule_bytes == 3 * np.dtype(np.int32).itemsize
    assert evaluator.execution_mh1_generated_forward_launch_count == 0
    assert evaluator.execution_mh1_generated_source_reverse_launch_count == 0
    assert evaluator.execution_mh1_generated_edge_reverse_launch_count == 0
    assert evaluator.execution_mh1_generated_conditioning_forward_launch_count == 0
    assert evaluator.execution_mh1_generated_conditioning_reverse_launch_count == 0
    assert np.isfinite(np.array(evaluator.node_energies, copy=True)).all()
    assert np.isfinite(np.array(evaluator.node_forces, copy=True)).all()


def test_mh1_execution_prepared_graph_lifecycle_and_calculator_routing(
    generalized_mh1_artifact,
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    _, _, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]
    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)

    reference.calculate(atoms.copy(), properties=properties)
    calculator.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            calculator.results[name],
            reference.results[name],
            rtol=0.0,
            atol=MH1_GENERATED_FLOAT32_ATOL,
        )
    token = evaluator.factorized_graph_generation
    assert token > 0
    assert evaluator.factorized_prepared_graph_count == 1
    assert evaluator.factorized_prepared_evaluation_count == 1
    assert evaluator.factorized_fallback_evaluation_count == 0
    assert evaluator.factorized_schedule_build_count == 1
    assert evaluator.factorized_topology_validation_count == 0
    assert evaluator.factorized_topology_validation_skip_count == 1
    assert evaluator.factorized_preparation_fence_count == 1
    assert evaluator.factorized_evaluation_fence_count == 1
    assert evaluator.factorized_stage_fence_count == 0
    assert evaluator.execution_geometry_allocation_count == 1
    assert evaluator.execution_geometry_capacity_edges >= len(inputs[6])
    assert evaluator.execution_geometry_workspace_bytes > 0
    assert evaluator.node_workspace_bytes > 0
    assert evaluator.precision_workspace_bytes == (
        evaluator.edge_workspace_bytes
        + evaluator.linear_workspace_bytes
        + evaluator.tensor_workspace_bytes
        + evaluator.product_workspace_bytes
    )
    assert evaluator.execution_geometry_copy_count == 1
    assert evaluator.execution_sphericart_initialization_count == 1
    assert evaluator.execution_sphericart_launch_count == 1
    if native_symmetrix._kokkos_default_execution_space() == "Cuda":
        assert evaluator.execution_sphericart_async_launch_count == 1
    else:
        assert evaluator.execution_sphericart_async_launch_count == 0

    perturbed_atoms = atoms.copy()
    perturbed_atoms.positions[0] += [1.0e-4, -2.0e-4, 3.0e-4]
    perturbed_inputs = calculator._mace_inputs(perturbed_atoms)
    for original, perturbed in zip(inputs[1:5], perturbed_inputs[1:5]):
        np.testing.assert_array_equal(original, perturbed)
    calculator.calculate(perturbed_atoms, properties=properties)
    assert evaluator.factorized_graph_generation == token
    assert evaluator.factorized_prepared_graph_count == 1
    assert evaluator.factorized_prepared_evaluation_count == 2
    assert evaluator.factorized_schedule_build_count == 1
    assert evaluator.factorized_topology_validation_skip_count == 2
    assert evaluator.factorized_evaluation_fence_count == 2
    assert evaluator.execution_geometry_allocation_count == 1
    assert evaluator.execution_geometry_copy_count == 2
    assert evaluator.execution_sphericart_launch_count == 2
    assert evaluator._prepare_factorized_graph(*inputs[:5]) == token
    with pytest.raises(ValueError, match="coordinates do not match"):
        evaluator._compute_prepared_factorized(
            token, np.asarray(inputs[5]).reshape(-1)[:-1], inputs[6]
        )

    offsets = np.concatenate(([0], np.cumsum(inputs[2])))
    permutation = np.concatenate(
        [np.arange(offsets[i], offsets[i + 1])[::-1] for i in range(len(atoms))]
    )
    reordered = (
        inputs[0],
        inputs[1],
        inputs[2],
        np.asarray(inputs[3])[permutation],
        np.asarray(inputs[4])[permutation],
        np.asarray(inputs[5])[permutation].reshape(-1),
        np.asarray(inputs[6])[permutation],
    )
    evaluator.compute_node_energies_forces(*reordered)
    assert evaluator.factorized_fallback_evaluation_count == 1
    assert evaluator.factorized_topology_validation_count == 1
    with pytest.raises(ValueError, match="graph token is stale"):
        evaluator._compute_prepared_factorized(
            token, np.asarray(inputs[5]).reshape(-1), inputs[6]
        )

    refreshed_token = evaluator._prepare_factorized_graph(*inputs[:5])
    assert refreshed_token > token
    initial_capacity = evaluator.execution_geometry_capacity_edges
    initial_allocations = evaluator.execution_geometry_allocation_count
    insertion = int(inputs[2][0])
    repeated_num_neigh = np.array(inputs[2], copy=True)
    repeated_num_neigh[0] += 1
    repeated_topology = (
        inputs[0],
        inputs[1],
        repeated_num_neigh,
        np.insert(np.asarray(inputs[3]), insertion, inputs[3][0]),
        np.insert(np.asarray(inputs[4]), insertion, inputs[4][0]),
    )
    device_bytes = 32 * 1024**3
    reserve = device_bytes // 20
    num_lm = (json.loads(model_path.read_text())["l_max"] + 1) ** 2
    geometry_bytes_per_edge = (
        4 * np.dtype(np.float64).itemsize
        + (3 + 7 * num_lm) * np.dtype(np.float32).itemsize
    )
    exact_bytes = len(repeated_topology[3]) * geometry_bytes_per_edge
    current_bytes = initial_capacity * geometry_bytes_per_edge
    evaluator._set_execution_geometry_device_memory_info_for_testing(
        reserve + exact_bytes - 1 - current_bytes,
        device_bytes,
    )
    with pytest.raises(RuntimeError, match=r"MH-1 direct geometry requires"):
        evaluator._prepare_factorized_graph(*repeated_topology)
    assert evaluator.execution_geometry_capacity_edges == initial_capacity

    constrained_free = reserve + exact_bytes + 1 - current_bytes
    evaluator._set_execution_geometry_device_memory_info_for_testing(
        constrained_free, device_bytes
    )
    larger_token = evaluator._prepare_factorized_graph(*repeated_topology)
    assert larger_token > refreshed_token
    assert evaluator.execution_geometry_capacity_edges == len(repeated_topology[3])
    assert evaluator.execution_geometry_growth_reason.startswith("exact growth:")
    assert evaluator.execution_geometry_allocation_count == initial_allocations + 1
    grown_capacity = evaluator.execution_geometry_capacity_edges
    evaluator._set_execution_geometry_device_memory_info_for_testing(0, 0)

    empty = np.empty(0, dtype=np.int32)
    empty_token = evaluator._prepare_factorized_graph(
        inputs[0],
        inputs[1],
        np.zeros(inputs[0], dtype=np.int32),
        empty,
        empty,
    )
    evaluator._compute_prepared_factorized(
        empty_token,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    assert evaluator.execution_geometry_capacity_edges == grown_capacity
    assert evaluator.execution_geometry_growth_reason == "existing capacity reused"
    assert evaluator.factorized_schedule_entries == 0
    assert evaluator.factorized_schedule_active_sources == 0

    evaluator.set_streamed_edges("generic")
    assert evaluator.factorized_graph_generation == 0
    with pytest.raises(ValueError, match="requires streamed_edges"):
        evaluator._compute_prepared_factorized(
            empty_token,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )


def test_mh1_execution_native_geometry_and_device_force_reduction(
    generalized_mh1_artifact,
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    _, _, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]

    host_geometry = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        jit="off",
    )
    host_inputs = host_geometry._mace_inputs(atoms)
    host_token = host_geometry._compute_mace(host_inputs, atoms=atoms)
    host_results = host_geometry._collect_mace_results(
        atoms, host_inputs, properties, host_token
    )
    assert host_geometry.evaluator.execution_geometry_copy_count == 2

    native_geometry = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        jit="off",
    )
    evaluator = native_geometry.evaluator
    assert callable(evaluator._prepare_factorized_geometry)
    assert callable(evaluator._compute_prepared_factorized_positions)
    assert callable(evaluator._reduce_atom_forces)
    assert callable(evaluator._reduce_stress)
    native_geometry.calculate(atoms.copy(), properties=properties)

    for name in properties:
        np.testing.assert_allclose(
            native_geometry.results[name], host_results[name], rtol=0.0, atol=5e-4
        )
    assert native_geometry.neighbor_cache_build_count == 1
    assert evaluator.execution_geometry_copy_count == 1
    assert evaluator.factorized_prepared_graph_count == 1
    assert evaluator.factorized_prepared_evaluation_count == 1

    token = evaluator.factorized_graph_generation
    inputs = native_geometry._mace_inputs(atoms, native_geometry=True)
    reduced = np.asarray(
        evaluator._reduce_atom_forces(inputs[0], inputs[7], inputs[3], token)
    ).reshape((-1, 3))
    np.testing.assert_allclose(
        reduced, native_geometry.results["forces"], rtol=0.0, atol=5e-4
    )

    empty = np.empty(0, dtype=np.int32)
    replacement = evaluator._prepare_factorized_graph(
        inputs[0], inputs[1], np.zeros(inputs[0], dtype=np.int32), empty, empty
    )
    assert replacement > token
    with pytest.raises(ValueError, match="current completed graph"):
        evaluator._reduce_atom_forces(inputs[0], inputs[7], inputs[3], token)
    with pytest.raises(ValueError, match="current completed graph"):
        evaluator._reduce_stress(atoms.get_volume(), np.empty(0), token)


@pytest.mark.parametrize(
    ("dtype", "mode"),
    [("float64", "generic"), ("float32", "generic"), ("float32", "direct")],
)
def test_mh1_kokkos_device_stress_matches_host_reduction(
    generalized_mh1_artifact, dtype, mode
):
    evaluator_name = (
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without the requested Kokkos precision")
    _, _, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges=mode,
    )
    inputs = calculator._mace_inputs(atoms)
    token = calculator._compute_mace(inputs, atoms=atoms)
    pair_forces = np.asarray(calculator.evaluator.node_forces).reshape((-1, 3))
    expected = full_3x3_to_voigt_6_stress(
        (-pair_forces[: len(inputs[7])].T @ inputs[5]) / atoms.get_volume()
    )
    actual = calculator._collect_mace_results(atoms, inputs, ["stress"], token)[
        "stress"
    ]
    tolerance = 2e-12 if dtype == "float64" else 2e-5
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_disconnected_batch_stress_reduces_on_device(
    generalized_mh1_artifact, dtype
):
    evaluator_name = (
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without the requested Kokkos precision")
    if dtype == "float64" and native_symmetrix._kokkos_default_execution_space() in (
        "Cuda",
        "HIP",
    ):
        pytest.skip("generated accelerator MH-1 direct execution requires FP32")
    _, _, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
        execution_profile="speed",
    )
    evaluator = calculator.evaluator
    evaluator.set_mh1_edge_executor("pair_spline_v1")
    assert callable(evaluator._prepare_factorized_batch)
    assert callable(evaluator._reduce_batched_stress)

    inputs = calculator._mace_inputs(atoms)
    num_nodes, node_types, num_neigh, sources, neigh_types, xyz, distances, _ = inputs
    edge_count = len(sources)
    batch_token = evaluator._prepare_factorized_graph(
        2 * num_nodes,
        np.tile(node_types, 2),
        np.tile(num_neigh, 2),
        np.concatenate((sources, np.asarray(sources) + num_nodes)),
        np.tile(neigh_types, 2),
    )
    edge_offsets = np.asarray([0, edge_count, 2 * edge_count], dtype=np.int64)
    evaluator._prepare_factorized_batch(batch_token, edge_offsets)
    evaluator._compute_prepared_factorized(
        batch_token,
        np.tile(np.asarray(xyz).reshape(-1), 2),
        np.tile(distances, 2),
    )
    volumes = np.asarray([atoms.get_volume(), 2.0 * atoms.get_volume()])
    stresses = np.asarray(
        evaluator._reduce_batched_stress(volumes, batch_token), dtype=float
    ).reshape(2, 3, 3)

    pair_forces = np.asarray(evaluator.node_forces, dtype=float).reshape(-1, 3)
    first_virial = -(pair_forces[:edge_count].T @ xyz)
    second_virial = -(pair_forces[edge_count : 2 * edge_count].T @ xyz)
    expected = np.stack((first_virial / volumes[0], second_virial / volumes[1]))
    tolerance = 2e-12 if dtype == "float64" else 2e-5
    np.testing.assert_allclose(stresses, expected, rtol=0.0, atol=tolerance)

    with pytest.raises(ValueError, match="edge offsets"):
        evaluator._prepare_factorized_batch(
            batch_token, np.asarray([1, edge_count, 2 * edge_count])
        )
    with pytest.raises(ValueError, match="volumes"):
        evaluator._reduce_batched_stress(np.asarray([volumes[0]]), batch_token)
    evaluator._prepare_factorized_graph(
        num_nodes, node_types, num_neigh, sources, neigh_types
    )
    with pytest.raises(ValueError, match="completed batch graph"):
        evaluator._reduce_batched_stress(volumes, batch_token)


@pytest.mark.parametrize(
    ("dtype", "mode"),
    [("float64", "generic"), ("float32", "generic"), ("float32", "direct")],
)
def test_mh1_kokkos_device_stress_matches_cell_finite_difference(
    generalized_mh1_artifact, dtype, mode
):
    evaluator_name = (
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without the requested Kokkos precision")
    _, _, model_path = generalized_mh1_artifact
    atoms = _generalized_mh1_atoms()
    atoms.calc = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges=mode,
    )
    stress = atoms.get_stress()[0]
    step = 1e-5 if dtype == "float64" else 2e-3
    energies = []
    for strain in (step, -step):
        strained = atoms.copy()
        strained.calc = atoms.calc
        transform = np.eye(3)
        transform[0, 0] += strain
        strained.set_cell(atoms.cell.array @ transform, scale_atoms=True)
        energies.append(strained.get_potential_energy())
    derivative = (energies[0] - energies[1]) / (2 * step * atoms.get_volume())
    tolerance = 2e-5 if dtype == "float64" else 3e-3
    assert stress == pytest.approx(derivative, abs=tolerance)


def test_mh1_execution_prepared_zbl_stays_on_evaluator_stream(
    generalized_mh1_artifact, tmp_path
):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    _, data, _ = generalized_mh1_artifact
    zbl_data = json.loads(json.dumps(data))
    zbl_data["has_zbl"] = True
    zbl_data["zbl"] = {
        "a_exp": 0.3,
        "a_prefactor": 0.4543,
        "c": {
            "shape": [4],
            "values": [0.1818, 0.5099, 0.2802, 0.02817],
        },
        "covalent_radii": {
            "shape": [119],
            "values": [1.0] * 119,
        },
        "p": {"shape": [], "values": [5]},
    }
    model_path = tmp_path / "mh1-zbl.json"
    model_path.write_text(json.dumps(zbl_data, separators=(",", ":")))
    atoms = _generalized_mh1_atoms()
    properties = ["energy", "energies", "forces", "stress"]
    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    prepared = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    reference.calculate(atoms.copy(), properties=properties)
    prepared.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            prepared.results[name],
            reference.results[name],
            rtol=0.0,
            atol=MH1_GENERATED_FLOAT32_ATOL,
        )
    assert prepared.evaluator.factorized_prepared_evaluation_count == 1
    assert prepared.evaluator.factorized_stage_fence_count == 0
    assert prepared.evaluator.factorized_evaluation_fence_count == 1
    assert prepared.evaluator.factorized_zbl_evaluator_stream_launch_count == 1


def test_mh1_kokkos_mode_switch_releases_edge_workspaces(mh1_si_artifact):
    dtype = "float32"
    evaluator_name = (
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without Kokkos support")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43).repeat((3, 3, 3))
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    properties = ["energy", "forces"]
    calculator.calculate(atoms, properties=properties)
    streamed_results = {
        name: np.array(calculator.results[name], copy=True) for name in properties
    }
    edge_count = len(calculator._mace_inputs(atoms)[6])
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        assert edge_count > _mh1_streamed_edge_block_size(True)
    streamed_bytes = calculator.evaluator.edge_workspace_bytes
    assert streamed_bytes > 0
    assert calculator.evaluator.edge_workspace_rows <= min(
        edge_count, _mh1_streamed_edge_block_size(True)
    )

    calculator.evaluator.set_streamed_edges("direct")
    calculator.calculate(atoms, properties=properties)
    assert calculator.evaluator.edge_workspace_rows <= min(
        edge_count, _mh1_streamed_edge_block_size(True)
    )
    calculator.evaluator.set_streamed_edges("generic")
    calculator.calculate(atoms, properties=properties)
    # Executor-internal storages can retain capacity across a mode cycle, so
    # the byte footprint need not return exactly to the generic-mode value;
    # retaining BOTH modes' workspaces at once would, however, double it.
    assert calculator.evaluator.edge_workspace_bytes < 2 * streamed_bytes
    tolerance = 2e-12 if dtype == "float64" else 2e-5
    for name in properties:
        np.testing.assert_allclose(
            calculator.results[name],
            streamed_results[name],
            rtol=0.0,
            atol=tolerance,
        )


def test_mh1_kokkos_mode_switch_fences_pending_evaluation(mh1_si_artifact):
    dtype = "float32"
    evaluator_name = (
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without the requested Kokkos precision")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    inputs = calculator._mace_inputs(atoms)
    (
        num_nodes,
        node_types,
        num_neigh,
        neighbors,
        neigh_types,
        xyz,
        distances,
        _,
    ) = inputs

    calculator.evaluator.compute_node_energies_forces(
        num_nodes,
        node_types,
        num_neigh,
        neighbors,
        neigh_types,
        xyz.flatten(),
        distances,
    )
    calculator.evaluator.set_streamed_edges("direct")
    streamed_energies = np.asarray(calculator.evaluator.node_energies)
    streamed_forces = np.asarray(calculator.evaluator.node_forces)

    calculator.evaluator.compute_node_energies_forces(
        num_nodes,
        node_types,
        num_neigh,
        neighbors,
        neigh_types,
        xyz.flatten(),
        distances,
    )
    calculator.evaluator.set_streamed_edges("generic")
    factorized_energies = np.asarray(calculator.evaluator.node_energies)
    factorized_forces = np.asarray(calculator.evaluator.node_forces)
    tolerance = 2e-12 if dtype == "float64" else 2e-5
    np.testing.assert_allclose(
        factorized_energies, streamed_energies, rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(
        factorized_forces, streamed_forces, rtol=0.0, atol=tolerance
    )


def test_mh1_cuda_streamed_fast_path_matches_native(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("CUDA-only nonlinear fallback regression")
    _, model_path = mh1_si_artifact
    atoms = Atoms(
        "Si3",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0], [0.4, 2.1, 0.3]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    properties = ["energy", "energies", "forces", "stress"]
    reference = Symmetrix(
        model_path,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="generic",
    )
    cuda = Symmetrix(model_path, use_kokkos=True, dtype="float64")
    assert cuda.evaluator.uses_mh1_fast_path
    assert cuda.evaluator.supports_streamed_edges
    assert cuda.streamed_edges == "generic"
    reference.calculate(atoms, properties=properties)
    cuda.calculate(atoms, properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            cuda.results[name], reference.results[name], rtol=0.0, atol=2e-11
        )
    del cuda, reference
    gc.collect()


@pytest.mark.parametrize("head", MH1_HEADS)
def test_mh1_extracts_and_evaluates_each_head(head, tmp_path):
    data = extract_mace_data(_mh1_model_path(), head=head)
    assert data["model_type"] == "MACE_Nonlinear"
    assert data["head"] == head
    assert data["atomic_numbers"] == data["model_atomic_numbers"]
    assert data["model_indices"] == list(range(len(data["model_atomic_numbers"])))
    assert data["scale_shift"]["scale"]["shape"] == []
    assert data["atomic_energies"]["shape"] == [1, 89]
    path = tmp_path / f"mh1-{head}.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    expected = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head=head,
    )
    expected.calculate(atoms.copy(), properties=["energy", "forces"])
    native_results = {}
    for use_kokkos in (False, True):
        actual = Symmetrix(path, use_kokkos=use_kokkos, dtype="float64")
        assert actual.evaluator.uses_mh1_fast_path
        actual.calculate(atoms.copy(), properties=["energy", "energies", "forces"])
        assert actual.results["energy"] == pytest.approx(
            expected.results["energy"], abs=2e-5
        )
        assert np.allclose(
            actual.results["forces"], expected.results["forces"], atol=2e-5
        )
        native_results[use_kokkos] = {
            name: np.array(actual.results[name], copy=True)
            for name in ("energy", "energies", "forces")
        }
    for name in native_results[False]:
        assert np.allclose(
            native_results[False][name],
            native_results[True][name],
            atol=_mh1_cross_backend_atol(),
        )


def test_mh1_extraction_rejects_species_subset():
    with pytest.raises(ValueError, match="species subsets are unsupported"):
        extract_mace_data(
            _mh1_model_path(),
            species=[14],
            head="matpes_r2scan",
        )


def _mh1_selected_universal_pairs(data):
    model_atomic_numbers = data["model_atomic_numbers"]
    selected_indices = (
        0,
        len(model_atomic_numbers) // 2,
        len(model_atomic_numbers) - 1,
    )
    selected_numbers = [model_atomic_numbers[index] for index in selected_indices]
    return list(zip(selected_numbers, selected_numbers[1:] + selected_numbers[:1]))


def test_mh1_universal_extraction_preserves_all_species(tmp_path):
    data = extract_mace_data(_mh1_model_path(), head="matpes_r2scan")
    assert data["atomic_numbers"] == data["model_atomic_numbers"]
    assert data["model_indices"] == list(range(len(data["model_atomic_numbers"])))
    assert len(data["atomic_numbers"]) == 89
    path = tmp_path / "mh1-universal.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    serial = Symmetrix(path, use_kokkos=False, dtype="float64")
    assert serial.evaluator.uses_mh1_fast_path
    upstream = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head="matpes_r2scan",
    )
    properties = ["energy", "energies", "forces", "stress"]
    for first, second in _mh1_selected_universal_pairs(data):
        atoms = Atoms(
            numbers=[first, second],
            positions=[[0.0, 0.0, 0.0], [1.8, 0.1, 0.0]],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        )
        upstream.calculate(atoms.copy(), properties=properties)
        serial.calculate(atoms.copy(), properties=properties)
        for name in properties:
            assert np.allclose(serial.results[name], upstream.results[name], atol=2e-5)


def test_mh1_universal_kokkos_matches_serial(tmp_path):
    data = extract_mace_data(_mh1_model_path(), head="matpes_r2scan")
    path = tmp_path / "mh1-universal.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    pairs = _mh1_selected_universal_pairs(data)
    universal_species = data["atomic_numbers"][:4]
    fifth_species = data["atomic_numbers"][4]
    del data
    gc.collect()
    serial = Symmetrix(path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(path, use_kokkos=True, dtype="float64", streamed_edges="generic")
    assert serial.evaluator.uses_mh1_fast_path
    assert kokkos.evaluator.uses_mh1_fast_path
    assert kokkos.evaluator.mh1_pair_spline_ready
    assert kokkos.evaluator.mh1_edge_executor == "pair_spline_v1"
    assert kokkos.evaluator.mh1_pair_spline_type_count == 0
    properties = ["energy", "energies", "forces", "stress"]
    four_species = Atoms(
        numbers=universal_species,
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.1, 0.0],
            [0.1, 1.9, 0.0],
            [0.0, 0.1, 1.7],
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )
    serial.calculate(four_species.copy(), properties=properties)
    kokkos.calculate(four_species.copy(), properties=properties)
    assert kokkos.evaluator.mh1_pair_spline_type_count == 4
    for name in properties:
        np.testing.assert_allclose(
            kokkos.results[name], serial.results[name], atol=_mh1_cross_backend_atol()
        )
    direct = Symmetrix(path, use_kokkos=True, dtype="float64", streamed_edges="direct")
    assert direct.evaluator.mh1_edge_executor == "pair_spline_v1"
    direct.calculate(four_species.copy(), properties=properties)
    assert direct.evaluator.mh1_pair_spline_type_count == 4
    for name in properties:
        np.testing.assert_allclose(
            direct.results[name], serial.results[name], atol=_mh1_cross_backend_atol()
        )
    five_species = Atoms(
        numbers=universal_species + [fifth_species],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.1, 0.0],
            [0.1, 1.9, 0.0],
            [0.0, 0.1, 1.7],
            [1.2, 1.1, 1.4],
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )
    mlp = Symmetrix(path, use_kokkos=True, dtype="float64", streamed_edges="generic")
    mlp.evaluator.set_mh1_edge_executor("mlp_reference")
    serial.calculate(five_species.copy(), properties=properties)
    mlp.calculate(five_species.copy(), properties=properties)
    kokkos.calculate(five_species.copy(), properties=properties)
    assert kokkos.evaluator.mh1_pair_spline_type_count == 5
    direct.calculate(five_species.copy(), properties=properties)
    assert direct.evaluator.mh1_pair_spline_type_count == 5
    for name in properties:
        np.testing.assert_allclose(
            kokkos.results[name], mlp.results[name], atol=_mh1_cross_backend_atol()
        )
        np.testing.assert_allclose(
            direct.results[name], mlp.results[name], atol=_mh1_cross_backend_atol()
        )
        np.testing.assert_allclose(
            mlp.results[name], serial.results[name], atol=_mh1_cross_backend_atol()
        )
    kokkos.evaluator.set_mh1_edge_executor("pair_spline_v1")
    kokkos.calculate(five_species.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            kokkos.results[name], mlp.results[name], atol=_mh1_cross_backend_atol()
        )
    del mlp
    gc.collect()
    for first, second in pairs:
        atoms = Atoms(
            numbers=[first, second],
            positions=[[0.0, 0.0, 0.0], [1.8, 0.1, 0.0]],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        )
        serial.calculate(atoms.copy(), properties=properties)
        kokkos.calculate(atoms.copy(), properties=properties)
        for name in properties:
            assert np.allclose(
                kokkos.results[name],
                serial.results[name],
                atol=_mh1_cross_backend_atol(),
            )


def test_mh1_rejects_legacy_pair_spline_extraction():
    with pytest.raises(ValueError, match="format-version-3 compact schema"):
        extract_mace_data(
            _mh1_model_path(),
            head="matpes_r2scan",
            radial_format="pair-splines",
        )


def test_mh1_raw_checkpoint_dispatches_to_native_kokkos():
    calculator = Symmetrix(
        _mh1_model_path(),
        head="matpes_r2scan",
        dtype="float64",
    )
    assert type(calculator.evaluator).__name__ == "MACENonlinearKokkos"


def test_mh1_raw_checkpoint_builds_generated_host_execution(monkeypatch, tmp_path):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Kokkos Float32 support")
    if native_symmetrix._kokkos_default_execution_space() not in (
        "Serial",
        "OpenMP",
    ):
        pytest.skip("generated host Execution requires a host Kokkos build")
    if shutil.which("c++") is None:
        pytest.skip("generated host Execution requires a C++20 compiler")

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    calculator = Symmetrix(
        _mh1_model_path(),
        head="matpes_r2scan",
        dtype="float32",
        use_kokkos=True,
        streamed_edges="direct",
    )
    calculator.evaluator.set_mh1_edge_executor("mlp_reference")
    assert calculator.jit_status == "built"
    assert calculator.evaluator.has_execution_mh1_contract
    assert calculator.evaluator.jit_mh1_host_plugin_v4_ready
    assert calculator.evaluator.execution_mh1_execution_backend == "generated_host_v4"
    atoms = Atoms(
        "Si3",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0], [0.2, 2.3, 0.1]],
        cell=[9.0, 9.0, 9.0],
        pbc=True,
    )
    calculator.calculate(atoms, properties=["energy", "energies", "forces", "stress"])
    assert calculator.evaluator.execution_mh1_generated_forward_launch_count == 2
    assert calculator.evaluator.execution_mh1_generated_source_reverse_launch_count == 2
    assert calculator.evaluator.execution_mh1_generated_edge_reverse_launch_count == 2
    assert (
        calculator.evaluator.execution_mh1_generated_conditioning_forward_launch_count
        == 2
    )
    assert (
        calculator.evaluator.execution_mh1_generated_conditioning_reverse_launch_count
        == 2
    )
    assert calculator.evaluator.conditioned_mlp_workspace_bytes == 0


def test_mh1_json_dispatch_does_not_depend_on_filename_suffix(
    mh1_si_artifact, tmp_path
):
    _, model_path = mh1_si_artifact
    suffixless_path = tmp_path / "mh1-model-data"
    suffixless_path.symlink_to(model_path)
    calculator = Symmetrix(suffixless_path, use_kokkos=False, dtype="float64")
    assert type(calculator.evaluator).__name__ == "MACENonlinear"


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_fast_path_requires_compatible_family_architecture(
    mh1_si_artifact, tmp_path, use_kokkos
):
    data, model_path = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    changed["interactions"][0]["hidden_irreps"] = "512x0e"
    path = tmp_path / "near-mh1.json"
    path.write_text(json.dumps(changed))
    if use_kokkos and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator_type = (
        native_symmetrix.MACENonlinearKokkos
        if use_kokkos
        else native_symmetrix.MACENonlinear
    )
    evaluator = evaluator_type(str(path))
    assert not evaluator.uses_mh1_fast_path
    assert not evaluator.supports_streamed_edges
    if use_kokkos:
        assert not evaluator.supports_factorized
    assert evaluator.streamed_edges_mode == "materialized"
    assert not evaluator.is_mh1_family
    assert evaluator.mh1_family_rejection_reason
    with pytest.raises(ValueError, match="compatible MACE-MH-1 family"):
        evaluator.set_streamed_edges("generic")
    if use_kokkos:
        with pytest.raises(ValueError, match="compatible MACE-MH-1 family"):
            evaluator.set_streamed_edges("direct")
    if use_kokkos and hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        with pytest.raises(ValueError, match="Float32.*compatible MACE-MH-1 family"):
            native_symmetrix.MACENonlinearKokkosFloat(str(path))
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    fast = Symmetrix(model_path, use_kokkos=use_kokkos, dtype="float64")
    generic = Symmetrix(path, use_kokkos=use_kokkos, dtype="float64")
    fast.calculate(atoms.copy(), properties=["energy", "energies", "forces", "stress"])
    generic.calculate(
        atoms.copy(), properties=["energy", "energies", "forces", "stress"]
    )
    assert fast.evaluator.uses_mh1_fast_path
    assert not generic.evaluator.uses_mh1_fast_path
    for property_name in ("energy", "energies", "forces", "stress"):
        np.testing.assert_allclose(
            generic.results[property_name],
            fast.results[property_name],
            rtol=0.0,
            atol=2e-12 if not use_kokkos else _mh1_cross_backend_atol(),
        )


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_fast_path_accepts_equivalent_irrep_formatting(
    mh1_si_artifact, tmp_path, use_kokkos
):
    data, _ = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    changed["interactions"][1]["node_feats_irreps"] = " 512x0e + 512x1o "
    changed["interactions"][0]["gate"]["irreps_out"] = (
        " 512x0e + 512x1o + 512x2e + 512x3o "
    )
    path = tmp_path / "mh1-equivalent-irreps.json"
    path.write_text(json.dumps(changed, separators=(",", ":")))
    if use_kokkos and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator_type = (
        native_symmetrix.MACENonlinearKokkos
        if use_kokkos
        else native_symmetrix.MACENonlinear
    )
    evaluator = evaluator_type(str(path))
    assert evaluator.uses_mh1_fast_path


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_fast_path_requires_conditionable_edge_mlp(
    mh1_si_artifact, tmp_path, use_kokkos
):
    data, _ = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    width = changed["interactions"][0]["conv_tp_weights"]["layers"][0]["weight"][
        "shape"
    ][1]
    changed["interactions"][0]["conv_tp_weights"]["layers"].insert(
        0,
        {
            "type": "layer_norm",
            "normalized_shape": [width],
            "eps": 1e-5,
            "weight": {"shape": [width], "values": [1.0] * width},
            "bias": {"shape": [width], "values": [0.0] * width},
        },
    )
    path = tmp_path / "near-mh1-unconditionable.json"
    path.write_text(json.dumps(changed, separators=(",", ":")))
    if use_kokkos and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator_type = (
        native_symmetrix.MACENonlinearKokkos
        if use_kokkos
        else native_symmetrix.MACENonlinear
    )
    evaluator = evaluator_type(str(path))
    assert not evaluator.uses_mh1_fast_path
    assert evaluator.is_mh1_family
    assert evaluator.mh1_uses_compiled_products
    assert not evaluator.mh1_uses_pair_conditioning
    assert "Conditioned edge MLP" in evaluator.mh1_fast_path_rejection_reason
    if use_kokkos and hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        with pytest.raises(ValueError, match="Float32.*compatible MACE-MH-1 family"):
            native_symmetrix.MACENonlinearKokkosFloat(str(path))


@pytest.mark.parametrize("use_kokkos", [False, True])
@pytest.mark.parametrize("module_name", ["conv_tp_weights", "density_fn"])
def test_mh1_fast_path_requires_exact_conditioned_mlp_input_width(
    mh1_si_artifact, tmp_path, use_kokkos, module_name
):
    data, _ = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    first_layer = changed["interactions"][0][module_name]["layers"][0]
    output_width, input_width = first_layer["weight"]["shape"]
    weights = np.asarray(first_layer["weight"]["values"]).reshape(
        output_width, input_width
    )
    first_layer["weight"]["shape"][1] = input_width + 1
    first_layer["weight"]["values"] = np.pad(weights, ((0, 0), (0, 1))).ravel().tolist()
    path = tmp_path / f"near-mh1-{module_name}-extra-conditioned-column.json"
    path.write_text(json.dumps(changed, separators=(",", ":")))
    if use_kokkos:
        if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
    evaluator_type = (
        native_symmetrix.MACENonlinearKokkos
        if use_kokkos
        else native_symmetrix.MACENonlinear
    )
    evaluator = evaluator_type(str(path))
    assert evaluator.is_mh1_family
    assert evaluator.mh1_uses_compiled_products
    assert not evaluator.mh1_uses_pair_conditioning
    assert not evaluator.uses_mh1_fast_path
    assert "Conditioned edge MLP" in evaluator.mh1_fast_path_rejection_reason
    if use_kokkos and hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        with pytest.raises(ValueError, match="Float32.*compatible MACE-MH-1 family"):
            native_symmetrix.MACENonlinearKokkosFloat(str(path))


def test_mh1_extraction_rejects_unsupported_architecture_features():
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")

    model.use_last_readout_only = True
    with pytest.raises(RuntimeError, match="use_last_readout_only"):
        extract_mace_nonlinear_data(model)
    model.use_last_readout_only = False

    model.joint_embedding = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="joint_embedding"):
        extract_mace_nonlinear_data(model)
    del model.joint_embedding

    original_readout = model.readouts[-1]
    model.readouts[-1] = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="only supports LinearReadoutBlock"):
        extract_mace_nonlinear_data(model)
    model.readouts[-1] = original_readout


def test_mh1_single_head_wigner_tensors_preserve_float64():
    from e3nn import o3
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        data = extract_mace_nonlinear_data(model)
    finally:
        torch.set_default_dtype(previous_dtype)
    serialized = data["interactions"][0]["conv_tp"]
    module = model.interactions[0].conv_tp
    for native_instruction, instruction in zip(
        serialized["instructions"], module.instructions, strict=True
    ):
        expected = o3.wigner_3j(
            module.irreps_in1[instruction.i_in1].ir.l,
            module.irreps_in2[instruction.i_in2].ir.l,
            module.irreps_out[instruction.i_out].ir.l,
        ).to(torch.float64)
        actual = np.asarray(native_instruction["wigner_3j"]["values"])
        assert np.allclose(actual, expected.numpy().reshape(-1), atol=1e-14)


def test_mh1_native_serial_and_kokkos_match_upstream(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.1, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    expected = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head="matpes_r2scan",
    )
    expected.calculate(
        atoms.copy(), properties=["energy", "energies", "forces", "stress"]
    )

    native_results = {}
    for use_kokkos, evaluator_name in (
        (False, "MACENonlinear"),
        (True, "MACENonlinearKokkos"),
    ):
        actual = Symmetrix(model_path, use_kokkos=use_kokkos, dtype="float64")
        assert actual.evaluator.uses_mh1_fast_path
        actual.calculate(
            atoms.copy(), properties=["energy", "energies", "forces", "stress"]
        )
        assert type(actual.evaluator).__name__ == evaluator_name
        assert actual.cutoff == pytest.approx(expected.r_max)
        assert actual.results["energy"] == pytest.approx(
            expected.results["energy"], abs=2e-5
        )
        assert np.allclose(
            actual.results["energies"], expected.results["energies"], atol=2e-5
        )
        assert np.allclose(
            actual.results["forces"], expected.results["forces"], atol=2e-5
        )
        assert np.allclose(
            actual.results["stress"], expected.results["stress"], atol=2e-5
        )
        native_results[use_kokkos] = {
            name: np.array(actual.results[name], copy=True)
            for name in ("energy", "energies", "forces", "stress")
        }

    for name in native_results[False]:
        assert np.allclose(
            native_results[False][name],
            native_results[True][name],
            atol=_mh1_cross_backend_atol(),
        )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_repeated_calculations(mh1_si_artifact, dtype):
    _, model_path = mh1_si_artifact
    serial = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(model_path, use_kokkos=True, dtype=dtype)
    tolerance = _mh1_cross_backend_atol() if dtype == "float64" else 2e-5
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    first_energy = None
    for displacement in (0.0, 0.04, -0.03):
        moved = atoms.copy()
        moved.positions[1] += [displacement, -0.5 * displacement, 0.25 * displacement]
        serial.calculate(moved, properties=["energy", "forces", "stress"])
        kokkos.calculate(moved, properties=["energy", "forces", "stress"])
        assert kokkos.results["energy"] == pytest.approx(
            serial.results["energy"], abs=tolerance
        )
        assert np.allclose(
            kokkos.results["forces"], serial.results["forces"], atol=tolerance
        )
        assert np.allclose(
            kokkos.results["stress"], serial.results["stress"], atol=tolerance
        )
        if first_energy is None:
            first_energy = kokkos.results["energy"]
        elif displacement != 0.0:
            assert kokkos.results["energy"] != pytest.approx(first_energy, abs=1e-8)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_reuses_workspaces_across_graph_sizes(mh1_si_artifact, dtype):
    _, model_path = mh1_si_artifact
    serial = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(model_path, use_kokkos=True, dtype=dtype)
    primitive = bulk("Si", "diamond", a=5.43)
    systems = (primitive, primitive.repeat((2, 2, 2)), primitive.repeat((3, 3, 3)))

    for atoms in (*systems, *reversed(systems), *systems):
        serial.calculate(atoms, properties=["energy", "energies", "forces", "stress"])
        kokkos.calculate(atoms, properties=["energy", "energies", "forces", "stress"])
        for property_name in ("energy", "energies", "forces", "stress"):
            tolerance = _mh1_cross_backend_atol()
            if dtype == "float32":
                tolerance = 2e-4 * len(atoms) if property_name == "energy" else 2e-4
            assert np.allclose(
                kokkos.results[property_name],
                serial.results[property_name],
                atol=tolerance,
            )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_handles_changes_in_supported_species(mh1_h_si_artifact, dtype):
    _, model_path = mh1_h_si_artifact
    serial = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(model_path, use_kokkos=True, dtype=dtype)
    kokkos_tolerance = _mh1_cross_backend_atol() if dtype == "float64" else 2e-4
    upstream = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head="matpes_r2scan",
    )
    for symbols, distance in (
        ("Si2", 2.2),
        ("H2", 0.8),
        ("SiH", 1.5),
        ("HSi", 1.5),
        ("Si2", 2.4),
    ):
        atoms = Atoms(
            symbols,
            positions=[[0.0, 0.0, 0.0], [distance, 0.1, 0.0]],
            cell=[14.0, 14.0, 14.0],
            pbc=True,
        )
        upstream.calculate(atoms.copy(), properties=["energy", "forces", "stress"])
        serial.calculate(atoms, properties=["energy", "forces", "stress"])
        kokkos.calculate(atoms, properties=["energy", "forces", "stress"])
        assert serial.results["energy"] == pytest.approx(
            upstream.results["energy"], abs=2e-5
        )
        assert np.allclose(
            serial.results["forces"], upstream.results["forces"], atol=2e-5
        )
        assert np.allclose(
            serial.results["stress"], upstream.results["stress"], atol=2e-5
        )
        assert kokkos.results["energy"] == pytest.approx(
            serial.results["energy"], abs=kokkos_tolerance
        )
        assert np.allclose(
            kokkos.results["forces"], serial.results["forces"], atol=kokkos_tolerance
        )
        assert np.allclose(
            kokkos.results["stress"], serial.results["stress"], atol=kokkos_tolerance
        )


def test_mh1_serial_matches_upstream_for_isolated_atom(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0])
    actual = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    expected = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head="matpes_r2scan",
    )
    actual.calculate(atoms.copy(), properties=["energy", "energies", "forces"])
    expected.calculate(atoms.copy(), properties=["energy", "energies", "forces"])
    assert actual.evaluator.uses_mh1_fast_path
    assert len(actual.evaluator.node_forces) == 0
    assert actual.results["energy"] == pytest.approx(
        expected.results["energy"], abs=2e-9
    )
    assert np.allclose(
        actual.results["energies"], expected.results["energies"], atol=2e-9
    )
    assert np.allclose(actual.results["forces"], expected.results["forces"], atol=2e-9)


@pytest.mark.parametrize(
    ("use_kokkos", "dtype"),
    [(False, "float64"), (True, "float64"), (True, "float32")],
)
def test_mh1_native_force_matches_finite_difference(mh1_si_artifact, use_kokkos, dtype):
    _, model_path = mh1_si_artifact
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    atoms.calc = Symmetrix(model_path, use_kokkos=use_kokkos, dtype=dtype)
    force = atoms.get_forces()[1, 0]
    step = 1e-4 if dtype == "float64" else 2e-3
    displaced = atoms.copy()
    displaced.calc = atoms.calc
    displaced.positions[1, 0] += step
    energy_plus = displaced.get_potential_energy()
    displaced.positions[1, 0] -= 2 * step
    energy_minus = displaced.get_potential_energy()
    tolerance = 2e-4 if dtype == "float64" else 3e-3
    assert force == pytest.approx(
        -(energy_plus - energy_minus) / (2 * step), abs=tolerance
    )


@pytest.mark.parametrize("module_name", ["conv_tp_weights", "density_fn"])
def test_mh1_affine_mlp_forward_and_reverse_match_autograd(
    mh1_si_artifact, module_name
):
    data, _ = mh1_si_artifact
    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    torch_module = getattr(model.interactions[0], module_name)
    native_module = native_symmetrix.AffineMLP(
        json.dumps(data["interactions"][0][module_name])
    )
    rng = np.random.default_rng(123)
    values = rng.normal(size=native_module.input_size)
    seed = rng.normal(size=native_module.output_size)
    torch_values = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    expected = torch_module(torch_values[None, :])[0]
    (expected * torch.tensor(seed, dtype=torch.float64)).sum().backward()
    assert np.allclose(
        native_module.evaluate(values), expected.detach().numpy(), atol=2e-11
    )
    assert np.allclose(
        native_module.reverse(values, seed), torch_values.grad.numpy(), atol=2e-10
    )


def test_mh1_e3_linear_batch_matches_repeated_scalar_calls():
    definition = {
        "irreps_in": "2x0e+2x0e+2x1o",
        "irreps_out": "2x0e+2x1o",
        "instructions": [
            {
                "i_in": 0,
                "i_out": 0,
                "path_weight": 0.75,
                "path_shape": [2, 2],
            },
            {
                "i_in": 1,
                "i_out": 0,
                "path_weight": 1.25,
                "path_shape": [2, 2],
            },
            {
                "i_in": 2,
                "i_out": 1,
                "path_weight": -0.4,
                "path_shape": [2, 2],
            },
        ],
        "weight": {
            "shape": [12],
            "values": [
                0.2,
                -0.1,
                0.5,
                0.3,
                -0.2,
                0.4,
                0.8,
                -0.5,
                -0.4,
                0.7,
                0.1,
                0.6,
            ],
        },
        "bias": {"shape": [8], "values": [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        "output_mask": {
            "shape": [8],
            "values": [1.0, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        },
    }
    module = native_symmetrix.E3Linear(json.dumps(definition))
    rng = np.random.default_rng(1024)
    samples = 7
    values = rng.normal(size=(samples, module.input_dimension))
    seeds = rng.normal(size=(samples, module.output_dimension))
    expected = np.concatenate([module.evaluate(row) for row in values])
    expected_adjoint = np.concatenate([module.reverse(row) for row in seeds])
    assert np.allclose(
        module.evaluate_batch(values.ravel(), samples), expected, atol=2e-14
    )
    assert np.allclose(
        module.reverse_batch(seeds.ravel(), samples), expected_adjoint, atol=2e-14
    )
    if hasattr(native_symmetrix, "E3LinearKokkos"):
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        kokkos = native_symmetrix.E3LinearKokkos(json.dumps(definition))
        assert np.allclose(
            kokkos.evaluate_batch(values.ravel(), samples), expected, atol=2e-14
        )
        assert np.allclose(
            kokkos.reverse_batch(seeds.ravel(), samples), expected_adjoint, atol=2e-14
        )
    if hasattr(native_symmetrix, "E3LinearKokkosFloat"):
        float_module = native_symmetrix.E3LinearKokkosFloat(json.dumps(definition))
        float_module.set_backend("scalar")
        scalar_values = np.asarray(
            float_module.evaluate_batch(values.astype(np.float32).ravel(), samples)
        )
        scalar_adjoints = np.asarray(
            float_module.reverse_batch(seeds.astype(np.float32).ravel(), samples)
        )
        float_module.set_backend("packed_gemm")
        packed_values = np.asarray(
            float_module.evaluate_batch(values.astype(np.float32).ravel(), samples)
        )
        packed_adjoints = np.asarray(
            float_module.reverse_batch(seeds.astype(np.float32).ravel(), samples)
        )
        input_mul_to_ir = np.array([0, 1, 2, 3, 4, 6, 8, 5, 7, 9])
        output_mul_to_ir = np.array([0, 1, 2, 4, 6, 3, 5, 7])
        values_ir_mul = np.empty_like(values, dtype=np.float32)
        values_ir_mul[:, input_mul_to_ir] = values.astype(np.float32)
        seeds_ir_mul = np.empty_like(seeds, dtype=np.float32)
        seeds_ir_mul[:, output_mul_to_ir] = seeds.astype(np.float32)
        packed_ir_mul_values = np.asarray(
            float_module.evaluate_ir_mul_batch(values_ir_mul.ravel(), samples)
        ).reshape(samples, -1)
        packed_ir_mul_adjoints = np.asarray(
            float_module.reverse_ir_mul_batch(seeds_ir_mul.ravel(), samples)
        ).reshape(samples, -1)
        assert float_module.backend == "packed_gemm"
        assert np.allclose(packed_values, scalar_values, atol=2e-6)
        assert np.allclose(packed_adjoints, scalar_adjoints, atol=2e-6)
        assert np.allclose(
            packed_ir_mul_values[:, output_mul_to_ir],
            packed_values.reshape(samples, -1),
            atol=2e-6,
        )
        assert np.allclose(
            packed_ir_mul_adjoints[:, input_mul_to_ir],
            packed_adjoints.reshape(samples, -1),
            atol=2e-6,
        )


@pytest.mark.parametrize("layer", [0, 1])
def test_mh1_float32_linear_backends_match_on_official_shapes(mh1_si_artifact, layer):
    if not hasattr(native_symmetrix, "E3LinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    data, _ = mh1_si_artifact
    module = native_symmetrix.E3LinearKokkosFloat(
        json.dumps(data["interactions"][layer]["linear_up"])
    )
    rng = np.random.default_rng(2048 + layer)
    samples = 17
    values = rng.normal(scale=0.1, size=(samples, module.input_dimension)).astype(
        np.float32
    )
    seeds = rng.normal(scale=0.1, size=(samples, module.output_dimension)).astype(
        np.float32
    )
    module.set_backend("scalar")
    scalar_values = np.asarray(module.evaluate_batch(values.ravel(), samples))
    scalar_adjoints = np.asarray(module.reverse_batch(seeds.ravel(), samples))
    module.set_backend("packed_gemm")
    packed_values = np.asarray(module.evaluate_batch(values.ravel(), samples))
    packed_adjoints = np.asarray(module.reverse_batch(seeds.ravel(), samples))
    assert module.selected_backend(samples) == "packed_gemm"
    assert module.workspace_bytes > 0
    assert np.allclose(packed_values, scalar_values, atol=3e-6)
    assert np.allclose(packed_adjoints, scalar_adjoints, atol=3e-6)


@pytest.mark.parametrize("layer", [0, 1])
def test_mh1_tensor_product_forward_and_reverse_match_autograd(mh1_si_artifact, layer):
    data, _ = mh1_si_artifact
    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    torch_module = model.interactions[layer].conv_tp
    native_module = native_symmetrix.E3TensorProduct(
        json.dumps(data["interactions"][layer]["conv_tp"])
    )
    rng = np.random.default_rng(321)
    x = rng.normal(scale=0.2, size=native_module.input_1_dimension)
    y = rng.normal(scale=0.2, size=native_module.input_2_dimension)
    w = rng.normal(scale=0.2, size=torch_module.weight_numel)
    seed = rng.normal(scale=0.2, size=native_module.output_dimension)
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    tw = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    expected = torch_module(tx[None, :], ty[None, :], tw[None, :])[0]
    (expected * torch.tensor(seed, dtype=torch.float64)).sum().backward()
    actual = native_module.evaluate(x, y, w)
    x_adj, y_adj, w_adj = native_module.reverse(x, y, w, seed)
    assert np.allclose(actual, expected.detach().numpy(), atol=2e-6)
    assert np.allclose(x_adj, tx.grad.numpy(), atol=2e-6)
    assert np.allclose(y_adj, ty.grad.numpy(), atol=2e-6)
    assert np.allclose(w_adj, tw.grad.numpy(), atol=2e-6)
    if hasattr(native_symmetrix, "E3TensorProductKokkos"):
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        kokkos = native_symmetrix.E3TensorProductKokkos(
            json.dumps(data["interactions"][layer]["conv_tp"])
        )
        assert kokkos.uses_mh1_fast_path
        kokkos_actual = kokkos.evaluate(x, y, w)
        kokkos_x_adj, kokkos_y_adj, kokkos_w_adj = kokkos.reverse(x, y, w, seed)
        assert np.allclose(kokkos_actual, actual, atol=2e-12)
        assert np.allclose(kokkos_x_adj, x_adj, atol=2e-12)
        assert np.allclose(kokkos_y_adj, y_adj, atol=2e-12)
        assert np.allclose(kokkos_w_adj, w_adj, atol=2e-12)
    if hasattr(native_symmetrix, "E3TensorProductKokkosFloat"):
        float_module = native_symmetrix.E3TensorProductKokkosFloat(
            json.dumps(data["interactions"][layer]["conv_tp"])
        )
        factors = np.array([1.0, 0.7, -0.4], dtype=np.float32)
        first = np.asarray([factor * x for factor in factors], dtype=np.float32)
        second = np.asarray(
            [(1.0 - 0.2 * factor) * y for factor in factors],
            dtype=np.float32,
        )
        batch_weights = np.asarray(
            [(1.0 + 0.1 * factor) * w for factor in factors],
            dtype=np.float32,
        )
        seeds = np.asarray(
            [(1.0 - 0.1 * factor) * seed for factor in factors],
            dtype=np.float32,
        )
        expected_values = np.concatenate(
            [
                native_module.evaluate(node_x, node_y, node_w)
                for node_x, node_y, node_w in zip(first, second, batch_weights)
            ]
        )
        expected_adjoints = [
            native_module.reverse(node_x, node_y, node_w, node_seed)
            for node_x, node_y, node_w, node_seed in zip(
                first, second, batch_weights, seeds
            )
        ]
        actual_values = float_module.evaluate_batch(
            first.ravel(), second.ravel(), batch_weights.ravel(), len(factors)
        )
        actual_adjoints = float_module.reverse_batch(
            first.ravel(),
            second.ravel(),
            batch_weights.ravel(),
            seeds.ravel(),
            len(factors),
        )
        assert float_module.uses_mh1_fast_path
        assert float_module.backend == "official_kokkos"
        assert np.allclose(actual_values, expected_values, atol=3e-6)
        for actual_component, expected_component in zip(
            actual_adjoints, zip(*expected_adjoints)
        ):
            assert np.allclose(
                actual_component, np.concatenate(expected_component), atol=3e-6
            )


def test_mh1_cuda_tensor_reverse_direct_nodes_matches_edge_scatter(
    mh1_si_artifact,
):
    if not hasattr(native_symmetrix, "E3TensorProductKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    data, _ = mh1_si_artifact
    module = native_symmetrix.E3TensorProductKokkosFloat(
        json.dumps(data["interactions"][1]["conv_tp"])
    )
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        assert not module.supports_direct_node_reverse
        pytest.skip("Direct-node tensor-product reverse requires Kokkos CUDA")
    assert module.supports_direct_node_reverse

    rng = np.random.default_rng(8675309)
    source_nodes = 3
    target_nodes = 2
    samples = 6
    first_edge = 2
    source_indices = np.array([1, 2, 0, 1, 0, 2, 0, 0], dtype=np.int32)
    target_indices = np.array([0, 1, 1, 0, 1, 1, 0, 1], dtype=np.int32)
    source_values = rng.normal(
        scale=0.2, size=(source_nodes, module.input_1_dimension)
    ).astype(np.float32)
    edge_input_2 = rng.normal(
        scale=0.2, size=(samples, module.input_2_dimension)
    ).astype(np.float32)
    edge_weights = rng.normal(scale=0.2, size=(samples, module.weight_size)).astype(
        np.float32
    )
    target_adjoint = rng.normal(
        scale=0.2, size=(target_nodes, module.output_dimension)
    ).astype(np.float32)
    initial_source_adjoint = rng.normal(scale=0.1, size=source_values.shape).astype(
        np.float32
    )

    block_sources = source_indices[first_edge : first_edge + samples]
    block_targets = target_indices[first_edge : first_edge + samples]
    edge_source_adjoint, expected_input_2_adjoint, expected_weight_adjoint = (
        module.reverse_batch(
            source_values[block_sources].ravel(),
            edge_input_2.ravel(),
            edge_weights.ravel(),
            target_adjoint[block_targets].ravel(),
            samples,
        )
    )
    expected_source_adjoint = initial_source_adjoint.copy()
    np.add.at(
        expected_source_adjoint,
        block_sources,
        np.asarray(edge_source_adjoint).reshape(source_values[block_sources].shape),
    )

    actual_source_adjoint, actual_input_2_adjoint, actual_weight_adjoint = (
        module.reverse_from_nodes(
            source_values.ravel(),
            source_indices,
            first_edge,
            edge_input_2.ravel(),
            edge_weights.ravel(),
            target_adjoint.ravel(),
            target_indices,
            initial_source_adjoint.ravel(),
            samples,
        )
    )
    np.testing.assert_allclose(
        np.asarray(actual_source_adjoint).reshape(source_values.shape),
        expected_source_adjoint,
        rtol=0.0,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        actual_input_2_adjoint,
        expected_input_2_adjoint,
        rtol=0.0,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        actual_weight_adjoint,
        expected_weight_adjoint,
        rtol=0.0,
        atol=3e-5,
    )

    with pytest.raises(ValueError, match="direct-node.*dimensions"):
        module.reverse_from_nodes(
            source_values.ravel(),
            source_indices,
            first_edge,
            edge_input_2.ravel(),
            edge_weights.ravel(),
            target_adjoint.ravel(),
            target_indices,
            initial_source_adjoint.ravel()[:-1],
            samples,
        )

    invalid_source_negative = source_indices.copy()
    invalid_source_negative[first_edge] = -1
    invalid_source_large = source_indices.copy()
    invalid_source_large[first_edge] = source_nodes
    invalid_target_negative = target_indices.copy()
    invalid_target_negative[first_edge] = -1
    invalid_target_large = target_indices.copy()
    invalid_target_large[first_edge] = target_nodes
    for invalid_sources, invalid_targets in (
        (invalid_source_negative, target_indices),
        (invalid_source_large, target_indices),
        (source_indices, invalid_target_negative),
        (source_indices, invalid_target_large),
        (source_indices, target_indices),
    ):
        invalid_source_values = (
            np.empty(0, dtype=np.float32)
            if invalid_sources is source_indices and invalid_targets is target_indices
            else source_values.ravel()
        )
        invalid_initial_adjoint = np.zeros_like(invalid_source_values)
        with pytest.raises(ValueError, match="direct-node.*indices.*out of bounds"):
            module.reverse_from_nodes(
                invalid_source_values,
                invalid_sources,
                first_edge,
                edge_input_2.ravel(),
                edge_weights.ravel(),
                target_adjoint.ravel(),
                invalid_targets,
                invalid_initial_adjoint,
                samples,
            )


@pytest.mark.parametrize(
    ("multiplicity", "expected_team_size"),
    [(1, 32), (32, 32), (33, 64), (64, 64), (96, 96), (128, 128), (256, 128)],
)
def test_mh1_cuda_tensor_team_size_tracks_edge_multiplicity(
    multiplicity, expected_team_size
):
    if not hasattr(native_symmetrix, "E3TensorProductKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    definition = {
        "irreps_in1": f"{multiplicity}x0e",
        "irreps_in2": "1x0e",
        "irreps_out": f"{multiplicity}x0e",
        "instructions": [
            {
                "i_in1": 0,
                "i_in2": 0,
                "i_out": 0,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
                "path_shape": [multiplicity, 1],
                "wigner_3j": {"shape": [1, 1, 1], "values": [1.0]},
            }
        ],
        "weight": {"shape": [0], "values": []},
        "output_mask": {
            "shape": [multiplicity],
            "values": [1.0] * multiplicity,
        },
    }
    module = native_symmetrix.E3TensorProductKokkosFloat(json.dumps(definition))
    assert module.uses_mh1_fast_path
    if native_symmetrix._kokkos_default_execution_space() == "Cuda":
        assert module.channel_team_size == expected_team_size
        assert module.harmonic_team_size == expected_team_size
    else:
        assert module.channel_team_size == 0
        assert module.harmonic_team_size == 0
    values = np.linspace(0.2, 1.2, multiplicity, dtype=np.float32)
    harmonic = np.array([0.7], dtype=np.float32)
    weights = np.linspace(0.3, 0.8, multiplicity, dtype=np.float32)
    seed = np.linspace(-0.4, 0.6, multiplicity, dtype=np.float32)
    actual = np.asarray(module.evaluate_batch(values, harmonic, weights, 1))
    values_adj, harmonic_adj, weights_adj = module.reverse_batch(
        values, harmonic, weights, seed, 1
    )
    np.testing.assert_allclose(
        actual, values * harmonic[0] * weights, rtol=0.0, atol=2e-7
    )
    np.testing.assert_allclose(
        values_adj, seed * harmonic[0] * weights, rtol=0.0, atol=2e-7
    )
    np.testing.assert_allclose(
        harmonic_adj,
        [np.sum(seed * values * weights)],
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        weights_adj, seed * values * harmonic[0], rtol=0.0, atol=2e-7
    )


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_uuu_tensor_product_uses_diagonal_weights(use_kokkos):
    definition = {
        "irreps_in1": "3x0e",
        "irreps_in2": "3x0e",
        "irreps_out": "3x0e",
        "instructions": [
            {
                "i_in1": 0,
                "i_in2": 0,
                "i_out": 0,
                "connection_mode": "uuu",
                "has_weight": True,
                "path_weight": 1.0,
                "path_shape": [3],
                "wigner_3j": {"shape": [1, 1, 1], "values": [1.0]},
            }
        ],
        "weight": {"shape": [0], "values": []},
        "output_mask": {"shape": [3], "values": [1.0, 1.0, 1.0]},
    }
    if use_kokkos:
        if not hasattr(native_symmetrix, "E3TensorProductKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        module = native_symmetrix.E3TensorProductKokkos(json.dumps(definition))
    else:
        module = native_symmetrix.E3TensorProduct(json.dumps(definition))
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    weights = np.array([0.1, 0.2, 0.3])
    seed = np.array([0.7, -0.4, 0.2])
    assert np.allclose(module.evaluate(x, y, weights), weights * x * y)
    x_adj, y_adj, weight_adj = module.reverse(x, y, weights, seed)
    assert np.allclose(x_adj, seed * weights * y)
    assert np.allclose(y_adj, seed * weights * x)
    assert np.allclose(weight_adj, seed * x * y)


def test_mh1_kokkos_tensor_product_validates_binding_dimensions():
    if not hasattr(native_symmetrix, "E3TensorProductKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    definition = {
        "irreps_in1": "1x0e",
        "irreps_in2": "1x0e",
        "irreps_out": "1x0e",
        "instructions": [
            {
                "i_in1": 0,
                "i_in2": 0,
                "i_out": 0,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
                "path_shape": [1, 1],
                "wigner_3j": {"shape": [1, 1, 1], "values": [1.0]},
            }
        ],
        "weight": {"shape": [1], "values": [2.0]},
        "output_mask": {"shape": [1], "values": [1.0]},
    }
    module = native_symmetrix.E3TensorProductKokkos(json.dumps(definition))
    assert module.has_internal_weights
    assert module.evaluate([3.0], [5.0], []) == pytest.approx([30.0])
    with pytest.raises(ValueError, match="input dimensions"):
        module.evaluate([], [5.0], [])
    with pytest.raises(ValueError, match="weight dimensions"):
        module.evaluate([3.0], [5.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="reverse dimensions"):
        module.reverse([3.0], [5.0], [], [])


def test_mh1_affine_mlp_rejects_inconsistent_layer_dimensions():
    definition = {
        "layers": [
            {
                "type": "linear",
                "weight": {"shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]},
                "bias": {"shape": [2], "values": [0.0, 0.0]},
            },
            {
                "type": "layer_norm",
                "normalized_shape": [3],
                "eps": 1e-5,
                "weight": {"shape": [3], "values": [1.0, 1.0, 1.0]},
                "bias": {"shape": [3], "values": [0.0, 0.0, 0.0]},
            },
        ]
    }
    with pytest.raises(ValueError, match="dimensions are inconsistent"):
        native_symmetrix.AffineMLP(json.dumps(definition))


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_product_basis_rejects_malformed_tensor_layout(use_kokkos):
    linear = {
        "irreps_in": "1x0e",
        "irreps_out": "1x0e",
        "instructions": [
            {
                "i_in": 0,
                "i_out": 0,
                "path_weight": 1.0,
                "path_shape": [1, 1],
            }
        ],
        "weight": {"shape": [1], "values": [1.0]},
        "bias": {"shape": [0], "values": []},
        "output_mask": {"shape": [1], "values": [1.0]},
    }
    definition = {
        "node_feats_irreps": "1x0e",
        "target_irreps": "1x0e",
        "use_sc": False,
        "use_agnostic_product": False,
        "linear": linear,
        "symmetric_contractions": {
            "irreps_in": "1x0e",
            "irreps_out": "1x0e",
            "contractions": [
                {
                    "correlation": 1,
                    "weights": [],
                    "weights_max": {"shape": [1], "values": [1.0]},
                    "u_tensors": [{"shape": [1, 1], "values": [1.0]}],
                }
            ],
        },
    }
    if use_kokkos:
        if not hasattr(native_symmetrix, "E3ProductBasisKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        product = native_symmetrix.E3ProductBasisKokkos
    else:
        product = native_symmetrix.E3ProductBasis
    with pytest.raises(ValueError, match="tensor layout"):
        product(json.dumps(definition))


def test_mh1_generic_product_batch_supports_varying_elements_without_skip():
    linear = {
        "irreps_in": "2x0e",
        "irreps_out": "2x0e",
        "instructions": [
            {
                "i_in": 0,
                "i_out": 0,
                "path_weight": 1.0,
                "path_shape": [2, 2],
            }
        ],
        "weight": {"shape": [4], "values": [1.0, 0.0, 0.0, 1.0]},
        "bias": {"shape": [0], "values": []},
        "output_mask": {"shape": [2], "values": [1.0, 1.0]},
    }
    definition = {
        "node_feats_irreps": "2x0e",
        "target_irreps": "2x0e",
        "use_sc": False,
        "use_agnostic_product": False,
        "linear": linear,
        "symmetric_contractions": {
            "irreps_in": "2x0e",
            "irreps_out": "2x0e",
            "contractions": [
                {
                    "correlation": 1,
                    "weights": [],
                    "weights_max": {
                        "shape": [2, 1, 2],
                        "values": [2.0, 3.0, 5.0, 7.0],
                    },
                    "u_tensors": [{"shape": [1, 1], "values": [1.0]}],
                }
            ],
        },
    }
    product = native_symmetrix.E3ProductBasis(json.dumps(definition))
    features = np.array([[0.2, -0.4], [1.1, 0.3], [-0.7, 0.9]])
    elements = [0, 1, 0]
    seeds = np.array([[0.6, -0.1], [-0.2, 0.8], [0.4, 0.5]])
    expected_values = np.concatenate(
        [product.evaluate(row, [], element) for row, element in zip(features, elements)]
    )
    expected_adjoints = np.concatenate(
        [
            product.reverse(row, element, seed)[0]
            for row, element, seed in zip(features, elements, seeds)
        ]
    )
    actual_values = product.evaluate_batch(
        features.ravel(), [], elements, len(elements)
    )
    actual_adjoints, skip_adjoints = product.reverse_batch(
        features.ravel(), elements, seeds.ravel(), len(elements)
    )
    assert np.allclose(actual_values, expected_values, atol=2e-14)
    assert np.allclose(actual_adjoints, expected_adjoints, atol=2e-14)
    assert np.allclose(skip_adjoints, 0.0)


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_native_rejects_inconsistent_graph_metadata(mh1_si_artifact, use_kokkos):
    _, model_path = mh1_si_artifact
    if use_kokkos:
        if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        evaluator = native_symmetrix.MACENonlinearKokkos(str(model_path))
    else:
        evaluator = native_symmetrix.MACENonlinear(str(model_path))
    with pytest.raises(ValueError, match="neighbor counts do not match"):
        evaluator.compute_node_energies_forces(1, [0], [1], [], [], [], [])


@pytest.mark.parametrize("product_index", [0, 1])
def test_mh1_product_basis_forward_and_reverse_match_autograd(
    mh1_si_artifact, product_index
):
    data, _ = mh1_si_artifact
    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    torch_module = model.products[product_index]
    native_module = native_symmetrix.E3ProductBasis(
        json.dumps(data["products"][product_index])
    )
    assert native_module.uses_compiled_plan
    assert native_module.compiled_term_count > 0
    rng = np.random.default_rng(456)
    features = rng.normal(scale=0.05, size=native_module.input_dimension)
    skip = rng.normal(scale=0.05, size=native_module.output_dimension)
    seed = rng.normal(scale=0.05, size=native_module.output_dimension)
    torch_features = torch.tensor(features, dtype=torch.float64, requires_grad=True)
    torch_skip = torch.tensor(skip, dtype=torch.float64, requires_grad=True)
    attrs = torch.zeros((1, 89), dtype=torch.float64)
    attrs[0, 13] = 1.0
    pieces = []
    offset = 0
    for multiplicity, irrep in torch_module.symmetric_contractions.irreps_in:
        width = irrep.dim
        size = multiplicity * width
        pieces.append(
            torch_features[offset : offset + size].reshape(multiplicity, width)
        )
        offset += size
    feature_major = torch.cat(pieces, dim=1)[None, :, :]
    expected = torch_module(feature_major, torch_skip[None, :], attrs)[0]
    (expected * torch.tensor(seed, dtype=torch.float64)).sum().backward()
    actual = native_module.evaluate(features, skip, 0)
    feature_adj, skip_adj = native_module.reverse(features, 0, seed)
    assert np.allclose(actual, expected.detach().numpy(), atol=2e-6)
    assert np.allclose(feature_adj, torch_features.grad.numpy(), atol=2e-6)
    assert np.allclose(skip_adj, torch_skip.grad.numpy(), atol=2e-6)

    batch_features = np.stack([features, 0.7 * features, -0.4 * features])
    batch_skip = np.stack([skip, -0.2 * skip, 0.5 * skip])
    batch_seed = np.stack([seed, 0.3 * seed, -0.6 * seed])
    expected_batch = np.concatenate(
        [
            native_module.evaluate(node, node_skip, 0)
            for node, node_skip in zip(batch_features, batch_skip)
        ]
    )
    expected_feature_adjoints = []
    expected_skip_adjoints = []
    for node, node_seed in zip(batch_features, batch_seed):
        node_adjoint, node_skip_adjoint = native_module.reverse(node, 0, node_seed)
        expected_feature_adjoints.extend(node_adjoint)
        expected_skip_adjoints.extend(node_skip_adjoint)
    actual_batch = native_module.evaluate_batch(
        batch_features.ravel(), batch_skip.ravel(), [0, 0, 0], 3
    )
    actual_feature_adjoints, actual_skip_adjoints = native_module.reverse_batch(
        batch_features.ravel(), [0, 0, 0], batch_seed.ravel(), 3
    )
    assert np.allclose(actual_batch, expected_batch, atol=2e-12)
    assert np.allclose(actual_feature_adjoints, expected_feature_adjoints, atol=2e-12)
    assert np.allclose(actual_skip_adjoints, expected_skip_adjoints, atol=2e-12)


@pytest.mark.parametrize("product_index", [0, 1])
def test_mh1_kokkos_compiled_product_matches_serial_batch(
    mh1_si_artifact, product_index
):
    if not hasattr(native_symmetrix, "E3ProductBasisKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    data, _ = mh1_si_artifact
    definition = json.dumps(data["products"][product_index])
    serial = native_symmetrix.E3ProductBasis(definition)
    kokkos = native_symmetrix.E3ProductBasisKokkos(definition)
    assert kokkos.uses_compiled_plan
    assert kokkos.compiled_term_count == serial.compiled_term_count

    rng = np.random.default_rng(4096 + product_index)
    samples = 6
    features = rng.normal(scale=0.05, size=(samples, serial.input_dimension))
    skips = rng.normal(scale=0.05, size=(samples, serial.output_dimension))
    seeds = rng.normal(scale=0.05, size=(samples, serial.output_dimension))
    elements = [0] * samples
    expected = serial.evaluate_batch(features.ravel(), skips.ravel(), elements, samples)
    expected_feature_adjoints, expected_skip_adjoints = serial.reverse_batch(
        features.ravel(), elements, seeds.ravel(), samples
    )
    actual = kokkos.evaluate_batch(features.ravel(), skips.ravel(), elements, samples)
    feature_adjoints, skip_adjoints = kokkos.reverse_batch(
        features.ravel(), elements, seeds.ravel(), samples
    )
    assert np.allclose(actual, expected, atol=2e-12)
    assert np.allclose(feature_adjoints, expected_feature_adjoints, atol=2e-12)
    assert np.allclose(skip_adjoints, expected_skip_adjoints, atol=2e-12)
    if native_symmetrix._kokkos_default_execution_space() in {
        "OpenMP",
        "Serial",
        "Threads",
    }:
        assert kokkos.feature_major_workspace_bytes > 0
    else:
        assert kokkos.feature_major_workspace_bytes == 0
    assert kokkos.workspace_bytes == (
        kokkos.feature_major_workspace_bytes
        + 2 * samples * serial.output_dimension * np.dtype(np.float64).itemsize
    )


@pytest.mark.parametrize("layer", [0, 1])
def test_mh1_kokkos_packed_product_linear_reverse_matches_native(
    mh1_si_artifact, layer
):
    from e3nn import o3

    if not hasattr(native_symmetrix, "E3ProductBasisKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    if native_symmetrix._kokkos_default_execution_space() not in {
        "OpenMP",
        "Serial",
        "Threads",
    }:
        pytest.skip("packed product-linear reverse is host-only")

    def unpack(packed, irreps, samples):
        native = np.empty((samples, packed.size // samples), dtype=packed.dtype)
        offset = 0
        for multiplicity, irrep in o3.Irreps(irreps):
            width = irrep.dim
            size = multiplicity * width
            block = packed[samples * offset : samples * (offset + size)].reshape(
                multiplicity, samples, width
            )
            native[:, offset : offset + size] = block.transpose(1, 0, 2).reshape(
                samples, size
            )
            offset += size
        return native

    data, _ = mh1_si_artifact
    product_data = data["products"][layer]
    linear_data = data["interactions"][layer]["linear_2"]
    product = native_symmetrix.E3ProductBasisKokkos(json.dumps(product_data))
    linear = native_symmetrix.E3LinearKokkos(json.dumps(linear_data))
    assert product.uses_standard_host_plan
    assert linear.output_dimension == product.input_dimension

    rng = np.random.default_rng(7168 + layer)
    samples = 5
    product_input = rng.normal(scale=0.05, size=(samples, product.input_dimension))
    product_seed = rng.normal(scale=0.05, size=(samples, product.output_dimension))
    elements = [0] * samples
    native_product_adjoint, _ = product.reverse_batch(
        product_input.ravel(), elements, product_seed.ravel(), samples
    )
    packed_product_adjoint = np.asarray(
        product.reverse_packed_batch(
            product_input.ravel(), elements, product_seed.ravel(), samples
        )
    )
    unpacked_product_adjoint = unpack(
        packed_product_adjoint,
        product_data["symmetric_contractions"]["irreps_in"],
        samples,
    )
    np.testing.assert_allclose(
        unpacked_product_adjoint,
        np.asarray(native_product_adjoint).reshape(samples, -1),
        rtol=0.0,
        atol=2e-12,
    )

    native_linear_adjoint = np.asarray(
        linear.reverse_batch(native_product_adjoint, samples)
    ).reshape(samples, -1)
    packed_linear_adjoint = np.asarray(
        linear.reverse_packed_to_packed_batch(packed_product_adjoint, samples)
    )
    unpacked_linear_adjoint = unpack(
        packed_linear_adjoint, linear_data["irreps_in"], samples
    )
    np.testing.assert_allclose(
        unpacked_linear_adjoint,
        native_linear_adjoint,
        rtol=0.0,
        atol=2e-12,
    )


@pytest.mark.parametrize("product_index", [0, 1])
def test_generalized_mh1_compiled_product_uses_bounded_host_workspace(
    generalized_mh1_artifact, product_index
):
    if not hasattr(native_symmetrix, "E3ProductBasisKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    _, data, _ = generalized_mh1_artifact
    definition = json.dumps(data["products"][product_index])
    serial = native_symmetrix.E3ProductBasis(definition)
    kokkos = native_symmetrix.E3ProductBasisKokkos(definition)
    assert kokkos.uses_compiled_plan

    rng = np.random.default_rng(6120 + product_index)
    maximum_samples = 0
    for samples in (4, 1, 7, 2):
        maximum_samples = max(maximum_samples, samples)
        features = rng.normal(scale=0.05, size=(samples, serial.input_dimension))
        skips = rng.normal(scale=0.05, size=(samples, serial.output_dimension))
        seeds = rng.normal(scale=0.05, size=(samples, serial.output_dimension))
        elements = [0] * samples
        expected = serial.evaluate_batch(
            features.ravel(), skips.ravel(), elements, samples
        )
        expected_feature_adjoints, expected_skip_adjoints = serial.reverse_batch(
            features.ravel(), elements, seeds.ravel(), samples
        )
        actual = kokkos.evaluate_batch(
            features.ravel(), skips.ravel(), elements, samples
        )
        feature_adjoints, skip_adjoints = kokkos.reverse_batch(
            features.ravel(), elements, seeds.ravel(), samples
        )
        repeated_feature_adjoints, repeated_skip_adjoints = kokkos.reverse_batch(
            features.ravel(), elements, seeds.ravel(), samples
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(
            feature_adjoints, expected_feature_adjoints, rtol=0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            skip_adjoints, expected_skip_adjoints, rtol=0.0, atol=2e-12
        )
        np.testing.assert_array_equal(repeated_feature_adjoints, feature_adjoints)
        np.testing.assert_array_equal(repeated_skip_adjoints, skip_adjoints)
        if native_symmetrix._kokkos_default_execution_space() in {
            "OpenMP",
            "Serial",
            "Threads",
        }:
            assert kokkos.feature_major_workspace_bytes > 0
        else:
            assert kokkos.feature_major_workspace_bytes == 0
        assert kokkos.workspace_bytes == (
            kokkos.feature_major_workspace_bytes
            + 2
            * maximum_samples
            * serial.output_dimension
            * np.dtype(np.float64).itemsize
        )


def test_mh1_conditioned_affine_mlp_matches_full_input(mh1_si_artifact):
    data, _ = mh1_si_artifact
    definition = data["interactions"][0]["conv_tp_weights"]
    full = native_symmetrix.AffineMLP(json.dumps(definition))
    rng = np.random.default_rng(918)
    dynamic_size = len(data["radial_embedding"]["basis"]["weights"]["values"])
    dynamic = rng.normal(scale=0.1, size=dynamic_size)
    source = rng.normal(scale=0.1, size=512)
    target = rng.normal(scale=0.1, size=full.input_size - dynamic_size - len(source))
    seed = rng.normal(scale=0.1, size=full.output_size)
    assert full.supports_conditioned_input(dynamic_size)
    source_contribution = full.first_layer_contribution(dynamic_size, source)
    target_contribution = full.first_layer_contribution(
        dynamic_size + len(source), target
    )
    complete = np.concatenate([dynamic, source, target])
    assert np.allclose(
        full.evaluate_conditioned(dynamic, source_contribution, target_contribution),
        full.evaluate(complete),
        atol=2e-12,
    )
    assert np.allclose(
        full.reverse_conditioned(
            dynamic, source_contribution, target_contribution, seed
        ),
        full.reverse(complete, seed)[:dynamic_size],
        atol=2e-12,
    )


def test_conditioned_affine_mlp_directional_derivative_matches_finite_difference():
    definition = {
        "layers": [
            {
                "type": "linear",
                "weight": {
                    "shape": [4, 5],
                    "values": [
                        0.3,
                        -0.7,
                        0.2,
                        0.5,
                        -0.1,
                        -0.4,
                        0.8,
                        0.1,
                        -0.2,
                        0.6,
                        0.9,
                        0.2,
                        -0.5,
                        0.4,
                        0.7,
                        -0.6,
                        0.1,
                        0.8,
                        -0.3,
                        0.2,
                    ],
                },
                "bias": {"shape": [4], "values": [0.1, -0.2, 0.05, 0.3]},
            },
            {
                "type": "layer_norm",
                "normalized_shape": [4],
                "eps": 1e-5,
                "weight": {"shape": [4], "values": [1.1, 0.8, 1.3, 0.9]},
                "bias": {"shape": [4], "values": [0.2, -0.1, 0.3, 0.0]},
            },
            {"type": "silu"},
            {
                "type": "linear",
                "weight": {
                    "shape": [2, 4],
                    "values": [0.4, -0.2, 0.7, 0.1, -0.5, 0.3, 0.2, 0.8],
                },
                "bias": {"shape": [2], "values": [0.15, -0.25]},
            },
        ]
    }
    module = native_symmetrix.AffineMLP(json.dumps(definition))
    dynamic = np.array([0.35, -0.18])
    direction = np.array([-0.4, 0.65])
    first = module.first_layer_contribution(2, [0.7, -0.25])
    second = module.first_layer_contribution(4, [0.45])
    output, derivative = module.evaluate_conditioned_directional(
        dynamic, direction, first, second
    )
    step = 1e-6
    plus = module.evaluate_conditioned(dynamic + step * direction, first, second)
    minus = module.evaluate_conditioned(dynamic - step * direction, first, second)
    finite_difference = (np.asarray(plus) - np.asarray(minus)) / (2.0 * step)
    np.testing.assert_allclose(
        output,
        module.evaluate_conditioned(dynamic, first, second),
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("module_name", ["conv_tp_weights", "density_fn"])
def test_mh1_conditioned_affine_mlp_batch_matches_scalar_calls(
    mh1_si_artifact, layer, module_name
):
    data, _ = mh1_si_artifact
    module = native_symmetrix.AffineMLP(
        json.dumps(data["interactions"][layer][module_name])
    )
    rng = np.random.default_rng(2048 + layer)
    samples = 5
    dynamic_size = len(data["radial_embedding"]["basis"]["weights"]["values"])
    dynamic = rng.normal(scale=0.1, size=(samples, dynamic_size))
    seeds = rng.normal(scale=0.1, size=(samples, module.output_size))
    first_contributions = []
    second_contributions = []
    for _ in range(samples):
        source = rng.normal(scale=0.1, size=512)
        target = rng.normal(scale=0.1, size=512)
        first_contributions.append(
            module.first_layer_contribution(dynamic_size, source)
        )
        second_contributions.append(
            module.first_layer_contribution(dynamic_size + 512, target)
        )
    row_contributions = np.asarray(first_contributions) + np.asarray(
        second_contributions
    )
    expected_outputs = []
    expected_adjoints = []
    for row, first, second, seed in zip(
        dynamic, first_contributions, second_contributions, seeds
    ):
        expected_outputs.extend(module.evaluate_conditioned(row, first, second))
        expected_adjoints.extend(module.reverse_conditioned(row, first, second, seed))
    outputs, adjoints = module.conditioned_batch(
        dynamic.ravel(),
        samples,
        dynamic_size,
        row_contributions.ravel(),
        seeds.ravel(),
    )
    assert np.allclose(outputs, expected_outputs, atol=2e-12)
    assert np.allclose(adjoints, expected_adjoints, atol=2e-12)
    if hasattr(native_symmetrix, "AffineMLPKokkos"):
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        kokkos = native_symmetrix.AffineMLPKokkos(
            json.dumps(data["interactions"][layer][module_name])
        )
        assert kokkos.supports_conditioned_input(dynamic_size)
        kokkos_outputs, kokkos_adjoints = kokkos.conditioned_batch(
            dynamic.ravel(),
            samples,
            dynamic_size,
            row_contributions.ravel(),
            seeds.ravel(),
        )
        assert np.allclose(kokkos_outputs, expected_outputs, atol=2e-12)
        assert np.allclose(kokkos_adjoints, expected_adjoints, atol=2e-12)


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_native_rejects_neighbor_source_type_mismatch(
    mh1_h_si_artifact, use_kokkos
):
    _, model_path = mh1_h_si_artifact
    if use_kokkos:
        if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        evaluator = native_symmetrix.MACENonlinearKokkos(str(model_path))
    else:
        evaluator = native_symmetrix.MACENonlinear(str(model_path))
    with pytest.raises(ValueError, match="invalid edge index, type"):
        evaluator.compute_node_energies_forces(
            2,
            [0, 1],
            [1, 0],
            [1],
            [0],
            [1.5, 0.0, 0.0],
            [1.5],
        )


def test_mh1_native_serial_float32_is_rejected(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    with pytest.raises(ValueError, match="Native serial.*require dtype 'float64'"):
        Symmetrix(model_path, use_kokkos=False, dtype="float32")


def test_mh1_kokkos_float32_supported_modes_match_float64(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos nonlinear support")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    properties = ["energy", "energies", "forces", "stress"]

    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float64",
        streamed_edges="generic",
    )
    reference.calculate(atoms.copy(), properties=properties)
    reference_results = {
        name: np.array(reference.results[name], copy=True) for name in properties
    }
    reference_workspace_bytes = reference.evaluator.edge_workspace_bytes
    float_results = {}
    float_workspace_bytes = {}
    for mode in ("generic", "direct"):
        calculator = Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges=mode,
        )
        assert type(calculator.evaluator).__name__ == "MACENonlinearKokkosFloat"
        assert calculator.evaluator.scalar_size_bytes == 4
        assert calculator.evaluator.uses_mh1_fast_path
        assert calculator.evaluator.supports_streamed_edges
        assert calculator.evaluator.streamed_edges_mode == mode
        calculator.calculate(atoms.copy(), properties=properties)
        float_results[mode] = {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }
        float_workspace_bytes[mode] = calculator.evaluator.edge_workspace_bytes

    for mode in ("generic", "direct"):
        for name in properties:
            # Float32 accumulation error scales with the property magnitude
            # (energies of a 16-atom cell reach ~1e2), so keep a small
            # relative headroom on top of the absolute floor.
            np.testing.assert_allclose(
                float_results[mode][name],
                reference_results[name],
                rtol=3e-7,
                atol=2e-5,
            )
    for name in properties:
        np.testing.assert_allclose(
            float_results["generic"][name],
            float_results["direct"][name],
            rtol=0.0,
            atol=2e-5,
        )
    assert float_workspace_bytes["generic"] <= reference_workspace_bytes * 0.51
    assert float_workspace_bytes["direct"] <= float_workspace_bytes["generic"]

    automatic = Symmetrix(model_path, use_kokkos=True, dtype="float32")
    assert automatic.streamed_edges == "direct"


def test_mh1_kokkos_exposes_synchronized_linear_controls(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos nonlinear support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    _, model_path = mh1_si_artifact
    evaluator = native_symmetrix.MACENonlinearKokkosFloat(str(model_path))
    execution_space = native_symmetrix._kokkos_default_execution_space()
    evaluator.set_streamed_edges("generic")
    expected_backend = "packed_gemm" if execution_space == "Cuda" else "auto"
    assert evaluator.e3_linear_backend == expected_backend
    assert evaluator.selected_e3_linear_backend(17) == "packed_gemm"
    assert evaluator.execution_mh1_host_node_backend == "auto"
    expected_host_node_backend = (
        "inactive" if execution_space in ("Cuda", "HIP") else "kokkos"
    )
    assert (
        evaluator.selected_execution_mh1_host_node_backend == expected_host_node_backend
    )
    if execution_space not in ("Cuda", "HIP"):
        evaluator.set_execution_mh1_host_node_backend("kokkos")
        assert evaluator.execution_mh1_host_node_backend == "kokkos"
        assert evaluator.selected_execution_mh1_host_node_backend == "kokkos"
        evaluator.set_execution_mh1_host_node_backend("auto")
    with pytest.raises(ValueError, match="auto, generated, or kokkos"):
        evaluator.set_execution_mh1_host_node_backend("invalid")
    assert evaluator.tensor_product_backend == "official_kokkos"
    expected_tensor_execution = (
        "official_cuda_team" if execution_space == "Cuda" else "official_kokkos_mdrange"
    )
    assert evaluator.tensor_product_execution_backend == expected_tensor_execution
    expected_team_size = 128 if execution_space == "Cuda" else 0
    assert evaluator.tensor_product_channel_team_size == expected_team_size
    assert evaluator.tensor_product_harmonic_team_size == expected_team_size
    assert evaluator.fused_gate_normalization_reverse_available
    assert evaluator.uses_fused_gate_normalization_reverse
    expected_direct_node_reverse = execution_space == "Cuda"
    assert (
        evaluator.direct_node_tensor_reverse_available is expected_direct_node_reverse
    )
    assert evaluator.uses_direct_node_tensor_reverse is expected_direct_node_reverse
    evaluator.set_streamed_edges("generic")
    assert evaluator.tensor_product_execution_backend == expected_tensor_execution
    assert evaluator.tensor_product_channel_team_size == expected_team_size
    assert evaluator.tensor_product_harmonic_team_size == expected_team_size
    assert evaluator.uses_fused_gate_normalization_reverse
    assert evaluator.uses_direct_node_tensor_reverse is expected_direct_node_reverse
    evaluator.set_streamed_edges("direct")
    assert evaluator.tensor_product_execution_backend == "mixed"
    assert evaluator.tensor_product_channel_team_size == -1
    assert evaluator.tensor_product_harmonic_team_size == -1
    assert evaluator.uses_fused_gate_normalization_reverse
    assert not evaluator.uses_direct_node_tensor_reverse
    evaluator.set_streamed_edges("generic")
    evaluator.set_fused_gate_normalization_reverse(False)
    evaluator.set_direct_node_tensor_reverse(False)
    assert not evaluator.uses_fused_gate_normalization_reverse
    assert not evaluator.uses_direct_node_tensor_reverse
    evaluator.set_fused_gate_normalization_reverse(True)
    assert evaluator.uses_fused_gate_normalization_reverse
    if expected_direct_node_reverse:
        evaluator.set_direct_node_tensor_reverse(True)
        assert evaluator.uses_direct_node_tensor_reverse
    else:
        with pytest.raises(ValueError, match="Float32 CUDA"):
            evaluator.set_direct_node_tensor_reverse(True)
    assert evaluator.linear_workspace_bytes == 0
    assert evaluator.tensor_workspace_bytes == 0
    assert evaluator.precision_workspace_bytes == evaluator.edge_workspace_bytes
    evaluator.set_e3_linear_backend("scalar")
    assert evaluator.e3_linear_backend == "scalar"
    evaluator.set_e3_linear_backend("packed_gemm")
    assert evaluator.e3_linear_backend == "packed_gemm"
    with pytest.raises(ValueError, match="auto, scalar, or packed_gemm"):
        evaluator.set_e3_linear_backend("invalid")

    double_evaluator = native_symmetrix.MACENonlinearKokkos(str(model_path))
    double_evaluator.set_streamed_edges("generic")
    assert (
        double_evaluator.tensor_product_execution_backend == "official_kokkos_mdrange"
    )
    expected_double_fused_gate = execution_space not in ("Cuda", "HIP")
    assert (
        double_evaluator.fused_gate_normalization_reverse_available
        is expected_double_fused_gate
    )
    assert (
        double_evaluator.uses_fused_gate_normalization_reverse
        is expected_double_fused_gate
    )
    assert not double_evaluator.direct_node_tensor_reverse_available
    assert not double_evaluator.uses_direct_node_tensor_reverse
    expected_double_linear = "scalar" if execution_space == "Cuda" else "packed_gemm"
    assert double_evaluator.selected_e3_linear_backend(17) == expected_double_linear
    evaluator.fence()


def test_mh1_cuda_packed_linear_matches_scalar(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos nonlinear support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("packed E3-linear regression requires Kokkos CUDA")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    properties = ["energy", "energies", "forces", "stress"]
    results = {}
    for backend in ("scalar", "packed_gemm"):
        calculator = Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="generic",
        )
        calculator.evaluator.set_e3_linear_backend(backend)
        calculator.calculate(atoms.copy(), properties=properties)
        results[backend] = {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }
    for name in properties:
        np.testing.assert_allclose(
            results["packed_gemm"][name],
            results["scalar"][name],
            rtol=0.0,
            atol=2e-5,
        )


def test_mh1_cuda_fused_reverse_matches_control(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos nonlinear support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("Fused MH-1 reverse requires Kokkos CUDA")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    properties = ["energy", "energies", "forces", "stress"]
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    results = {}
    fallback_workspace_bytes = None
    for fused_gate, direct_tensor in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        calculator.evaluator.set_fused_gate_normalization_reverse(fused_gate)
        calculator.evaluator.set_direct_node_tensor_reverse(direct_tensor)
        assert calculator.evaluator.uses_fused_gate_normalization_reverse is fused_gate
        assert calculator.evaluator.uses_direct_node_tensor_reverse is direct_tensor
        if direct_tensor and fallback_workspace_bytes is not None:
            assert calculator.evaluator.edge_workspace_bytes < fallback_workspace_bytes
        calculator.calculate(atoms.copy(), properties=properties)
        results[(fused_gate, direct_tensor)] = {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }
        if not direct_tensor and fallback_workspace_bytes is None:
            fallback_workspace_bytes = calculator.evaluator.edge_workspace_bytes
        elif direct_tensor:
            assert calculator.evaluator.edge_workspace_bytes < fallback_workspace_bytes
    control = results[(False, False)]
    for configuration, actual in results.items():
        for name in properties:
            np.testing.assert_allclose(
                actual[name],
                control[name],
                rtol=0.0,
                atol=2e-5,
                err_msg=f"configuration={configuration}, property={name}",
            )


def test_mh1_host_cached_gate_probabilities_match_control(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Fused MH-1 reverse requires Float32 Kokkos support")
    if native_symmetrix._kokkos_default_execution_space() in ("Cuda", "HIP"):
        pytest.skip("Cached packed gate probabilities are a host optimization")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    properties = ["energy", "energies", "forces", "stress"]
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="all_interactions",
    )
    results = {}
    for fused_gate in (False, True):
        calculator.evaluator.set_fused_gate_normalization_reverse(fused_gate)
        calculator.calculate(atoms.copy(), properties=properties)
        results[fused_gate] = {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }
    for name in properties:
        np.testing.assert_allclose(
            results[True][name],
            results[False][name],
            rtol=0.0,
            atol=5e-5,
            err_msg=f"property={name}",
        )


def test_legacy_evaluator_rejects_nonlinear_schema(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    with pytest.raises((RuntimeError, ValueError), match="MACE_Nonlinear"):
        native_symmetrix.MACE(str(model_path))
