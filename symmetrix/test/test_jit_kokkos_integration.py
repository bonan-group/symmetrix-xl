import copy
import gc
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from compact_r1_model import foundation_r1_test_model
from debug_jit import debug_factorized_symmetrix
from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix
from symmetrix.execution_contract import make_factorized_contract
from symmetrix.extract_mace_data import extract_mace_data
from symmetrix.jit import (
    JIT_GENERATION_VERSION,
    jit_nvrtc_information,
    prepare_jit_artifact,
)
from symmetrix.jit_codegen import (
    jit_r1_host_plugin_metadata,
    render_jit_r1_host_plugin,
)
from test_mace_streamed_edges import (
    _small_structure,
    streamed_model_paths,  # noqa: F401 - registers the shared pytest fixture
)


_HOST_EXECUTION_SPACES = frozenset({"OpenMP", "Serial"})
_PARITY_ATOL = 5e-5
_REPOSITORY = Path(__file__).resolve().parents[2]


def test_native_and_python_jit_generation_versions_match():
    assert native_symmetrix._required_jit_generation_version() == (
        JIT_GENERATION_VERSION
    )


def _require_host_plugin_evaluator():
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("the imported Symmetrix extension was built without Kokkos")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space not in _HOST_EXECUTION_SPACES:
        pytest.skip("Execution JIT host plugins require Kokkos OpenMP or Serial")
    if not hasattr(native_symmetrix.MACEKokkos, "_load_jit_host_plugin"):
        pytest.skip("the imported Kokkos extension lacks Execution host-plugin support")
    return execution_space


def _compiler():
    for name in ("c++", "g++", "clang++"):
        path = shutil.which(name)
        if path is not None:
            return path
    pytest.skip("a C++ compiler is required for Execution JIT integration coverage")


def _require_cuda_plugin_evaluator():
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("the imported Symmetrix extension was built without Kokkos")
    if not native_symmetrix._kokkos_is_initialized():
        try:
            native_symmetrix._init_kokkos()
        except RuntimeError as exc:
            pytest.skip(f"Kokkos Cuda could not be initialized: {exc}")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("Execution CUDA JIT integration requires Kokkos Cuda")
    required_methods = ("_load_jit_cuda_plugin",)
    if any(
        not hasattr(native_symmetrix.MACEKokkos, method) for method in required_methods
    ):
        pytest.skip("the imported Kokkos extension lacks Execution CUDA-plugin support")


def _require_nvrtc():
    try:
        information = jit_nvrtc_information()
    except Exception as exc:
        pytest.skip(f"NVRTC is required for CUDA JIT coverage: {exc}")
    if not information.get("available", False):
        pytest.skip(
            f"NVRTC is required for CUDA JIT coverage: {information.get('reason')}"
        )


def _make_registry_miss_model(model):
    model = copy.deepcopy(model)
    original = model["execution_contracts"]["R1"]
    original_terms = original["sparse_coupling"]["terms"]
    coefficients = [term["coefficient"] for term in original_terms]
    coefficients[0] *= 0.875
    paths = original["paths"]
    contract = make_factorized_contract(
        channels=original["channels"],
        radial_embedding=original["radial_embedding"],
        edge_l_max=original["edge_harmonics"]["l_max"],
        source_l_max=original["source_harmonics"]["l_max"],
        source_irreps=original["irreps"]["source"],
        edge_irreps=original["irreps"]["edge"],
        output_irreps=original["irreps"]["output"],
        output_l=[path["output_l"] for path in paths],
        edge_l=[path["edge_l"] for path in paths],
        source_l=[path["source_l"] for path in paths],
        instructions=[
            {
                "connection_mode": path["connection_mode"],
                "has_weight": path["has_weight"],
                "path_shape": path["path_shape"],
            }
            for path in paths
        ],
        sparse_lme=[term["lme"] for term in original_terms],
        sparse_coefficients=coefficients,
        sparse_rows=[term["row"] for term in original_terms],
        sparse_lm1=[term["lm1"] for term in original_terms],
        sparse_lm2=[term["lm2"] for term in original_terms],
    )
    assert contract["generation_fingerprint"] != (original["generation_fingerprint"])
    terms = contract["sparse_coupling"]["terms"]
    contract.pop("structure_fingerprint")
    model["execution_contracts"]["R1"] = contract
    model["Phi1_lme"] = [term["lme"] for term in terms]
    model["Phi1_lelm1lm2"] = [term["row"] for term in terms]
    model["Phi1_clebsch_gordan"] = [term["coefficient"] for term in terms]
    return model, contract


def _make_different_size_compact_model(model):
    channels = model["num_channels"]
    assert channels == 96
    assert model["L_max"] == 0
    assert model["Phi1_l"] == [0, 1, 2, 3]
    assert not model["A0_scaled"]
    assert not model["A1_scaled"]

    embedding = 3
    bessel_weights = [
        (index + 1) * math.pi / model["r_cut"] for index in range(embedding)
    ]

    def network(output_width, scale):
        weights = [
            scale * (((output + 1) * (basis + 2)) % 11 - 5) / 50.0
            for output in range(output_width)
            for basis in range(embedding)
        ]
        return {
            "shape": [embedding, output_width],
            "weights": [weights],
            "activation": "silu",
            "activation_scale": 1.0,
        }

    for key in tuple(model):
        if key.startswith("radial_spline_"):
            model.pop(key)
    model.update(
        {
            "symmetrix_format_version": 2,
            "radial_representation": "compact",
            "model_type": "MACE",
            "has_field_coupling": False,
            "field_couplings": [],
        }
    )
    model["compact_radial"] = {
        "spline_grid_min": 1e-12,
        "num_spline_points": 16,
        "basis": {
            "type": "bessel",
            "weights": bessel_weights,
            "prefactor": math.sqrt(2.0 / model["r_cut"]),
        },
        "cutoff": {
            "type": "polynomial",
            "r_max": model["r_cut"],
            "p": 6,
        },
        "distance_transform": {"type": "none"},
        "networks": {
            "R0": network((model["l_max"] + 1) * channels, 1.0),
            "R1": network(len(model["Phi1_l"]) * channels, 0.8),
        },
    }

    path_offsets = [0]
    for edge_l, source_l in zip(model["Phi1_l1"], model["Phi1_l2"]):
        path_offsets.append(path_offsets[-1] + (2 * edge_l + 1) * (2 * source_l + 1))

    def sparse_coordinates(row):
        for path, (begin, end) in enumerate(zip(path_offsets[:-1], path_offsets[1:])):
            if begin <= row < end:
                source_width = 2 * model["Phi1_l2"][path] + 1
                local_row = row - begin
                return (
                    model["Phi1_l1"][path] ** 2 + local_row // source_width,
                    model["Phi1_l2"][path] ** 2 + local_row % source_width,
                )
        raise AssertionError(f"Phi1 sparse row {row} is outside the path layout")

    sparse_coordinates_by_term = [
        sparse_coordinates(row) for row in model["Phi1_lelm1lm2"]
    ]
    contract = make_factorized_contract(
        channels=channels,
        radial_embedding=embedding,
        edge_l_max=model["l_max"],
        source_l_max=model["L_max"],
        source_irreps=f"{channels}x0e",
        edge_irreps="+".join(
            f"1x{l_value}{'e' if l_value % 2 == 0 else 'o'}"
            for l_value in range(model["l_max"] + 1)
        ),
        output_irreps="+".join(
            f"{channels}x{l_value}{'e' if l_value % 2 == 0 else 'o'}"
            for l_value in model["Phi1_l"]
        ),
        output_l=model["Phi1_l"],
        edge_l=model["Phi1_l1"],
        source_l=model["Phi1_l2"],
        instructions=[
            {
                "connection_mode": "uvu",
                "has_weight": True,
                "path_shape": [channels, 1],
            }
            for _ in model["Phi1_l"]
        ],
        sparse_lme=model["Phi1_lme"],
        sparse_coefficients=model["Phi1_clebsch_gordan"],
        sparse_rows=model["Phi1_lelm1lm2"],
        sparse_lm1=[coordinates[0] for coordinates in sparse_coordinates_by_term],
        sparse_lm2=[coordinates[1] for coordinates in sparse_coordinates_by_term],
    )
    model["execution_contracts"] = {"R1": contract}
    return model, contract


def _evaluate(calculator, atoms):
    inputs = calculator._mace_inputs(atoms)
    calculator.evaluator.compute_node_energies_forces(
        *inputs[:5], np.asarray(inputs[5]).reshape(-1), inputs[6]
    )
    result = calculator._collect_mace_results(atoms, inputs)
    return {
        "energy": result["energy"],
        "energies": np.array(result["energies"], copy=True),
        "forces": np.array(result["forces"], copy=True),
        "stress": np.array(result["stress"], copy=True),
    }


def _assert_results_close(actual, expected):
    assert actual["energy"] == pytest.approx(
        expected["energy"], rel=0.0, abs=_PARITY_ATOL
    )
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=0.0, atol=_PARITY_ATOL
        )


def _evaluate_field(calculator, atoms, electric_field):
    inputs = calculator._mace_inputs(atoms)
    calculator.evaluator.compute_node_energies_forces_field(
        *inputs[:5],
        np.asarray(inputs[5]).reshape(-1),
        inputs[6],
        electric_field,
    )
    return {
        "node_energies": np.array(calculator.evaluator.node_energies, copy=True),
        "node_forces": np.array(calculator.evaluator.node_forces, copy=True),
        "field_adjoint": np.array(calculator.evaluator.electric_field_adj, copy=True),
    }


def _assert_field_results_close(actual, expected, atol):
    for name in ("node_energies", "node_forces", "field_adjoint"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=0.0, atol=atol)


def test_cuda_registry_miss_jit_builds_reuses_and_survives_calculator_deletion(
    request,
    tmp_path,
    monkeypatch,
):
    _require_cuda_plugin_evaluator()
    _require_nvrtc()
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    streamed_model_paths = request.getfixturevalue("streamed_model_paths")
    standard_path, _ = streamed_model_paths
    model, contract = _make_registry_miss_model(json.loads(standard_path.read_text()))
    model_path = tmp_path / "compact-standard-r1-cuda-registry-miss.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    baseline_live_objects = native_symmetrix._kokkos_live_object_count()
    atoms = _small_structure()
    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    expected = _evaluate(reference, atoms)
    generic = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    generic_result = _evaluate(generic, atoms)
    _assert_results_close(generic_result, expected)
    assert not generic.evaluator.factorized_jit_ready
    assert not generic.evaluator.jit_cuda_plugin_ready
    assert generic.evaluator.factorized_selected_direct_forward_executor == "runtime"
    assert generic.evaluator.factorized_selected_direct_reverse_executor == "runtime"

    cold = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    cold_evaluator = cold.evaluator
    assert cold.jit_status == "built"
    assert cold.jit_compiler_backend == "nvrtc"
    assert cold.jit_artifact_path.endswith(".cubin")
    assert cold.jit_cache_key is not None
    assert cold.jit_artifact_path is not None
    assert cold.jit_artifact_id == cold_evaluator.jit_cuda_plugin_artifact_id
    assert cold_evaluator.jit_cuda_plugin_ready
    assert cold_evaluator.jit_cuda_plugin_path == cold.jit_artifact_path
    assert cold_evaluator.factorized_model_payload_verified
    assert cold_evaluator.factorized_jit_artifact_id.startswith("jit-r1-gen11-")
    assert cold_evaluator.factorized_jit_contract_fingerprint.startswith("sha256:")
    assert cold_evaluator.factorized_source_strategy == "jit_plugin"

    cold_launches = (
        cold_evaluator.factorized_jit_forward_launch_count,
        cold_evaluator.factorized_jit_reverse_launch_count,
    )
    cold_result = _evaluate(cold, atoms)
    _assert_results_close(cold_result, expected)
    _assert_results_close(cold_result, generic_result)
    assert cold_evaluator.factorized_jit_forward_launch_count - cold_launches[0] == 1
    assert cold_evaluator.factorized_jit_reverse_launch_count - cold_launches[1] == 1
    assert cold_evaluator.factorized_selected_direct_forward_executor == "jit_all"
    assert cold_evaluator.factorized_selected_direct_reverse_executor == "jit"

    warm = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    warm_evaluator = warm.evaluator
    assert warm.jit_status == "cached"
    assert warm.jit_cache_key == cold.jit_cache_key
    assert warm.jit_artifact_path == cold.jit_artifact_path
    assert warm.jit_artifact_id == cold.jit_artifact_id
    assert warm_evaluator.jit_cuda_plugin_ready
    assert warm_evaluator.jit_cuda_plugin_path == warm.jit_artifact_path
    warm_result = _evaluate(warm, atoms)
    _assert_results_close(warm_result, expected)
    _assert_results_close(warm_result, cold_result)

    del cold_evaluator, cold, generic, reference
    gc.collect()
    warm_launches = (
        warm_evaluator.factorized_jit_forward_launch_count,
        warm_evaluator.factorized_jit_reverse_launch_count,
    )
    surviving_result = _evaluate(warm, atoms)
    _assert_results_close(surviving_result, expected)
    assert warm_evaluator.factorized_jit_forward_launch_count - warm_launches[0] == 1
    assert warm_evaluator.factorized_jit_reverse_launch_count - warm_launches[1] == 1

    del warm_evaluator, warm
    gc.collect()
    assert native_symmetrix._kokkos_live_object_count() == baseline_live_objects


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_cuda_macefield_jit_matches_runtime_and_prepared_execution(
    request,
    tmp_path,
    monkeypatch,
    dtype,
):
    _require_cuda_plugin_evaluator()
    _require_nvrtc()
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    checkpoint = request.getfixturevalue("macefield_model_path")
    extracted = extract_mace_data(
        checkpoint,
        species=[7, 13],
        head="mp-dielectric",
    )
    model, _ = _make_registry_miss_model(extracted)
    model_path = tmp_path / "compact-macefield-r1-cuda-registry-miss.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    atoms = Atoms(
        ["Al", "N", "Al", "N"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.1, 0.2],
            [0.3, 1.9, 0.1],
            [1.9, 1.8, 0.4],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    expected = _evaluate_field(reference, atoms, electric_field)
    generated = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
    )
    evaluator = generated.evaluator
    assert generated.jit_status == "built"
    assert generated.jit_compiler_backend == "nvrtc"
    assert generated.jit_artifact_path.endswith(".cubin")
    assert evaluator.jit_cuda_plugin_ready
    assert f"-f{dtype[-2:]}-" in generated.jit_artifact_id
    launches = (
        evaluator.factorized_jit_forward_launch_count,
        evaluator.factorized_jit_reverse_launch_count,
    )
    actual = _evaluate_field(generated, atoms, electric_field)
    tolerance = 8e-5 if dtype == "float32" else 8e-11
    _assert_field_results_close(actual, expected, tolerance)
    assert evaluator.factorized_jit_forward_launch_count - launches[0] == 1
    assert evaluator.factorized_jit_reverse_launch_count - launches[1] == 1

    inputs = generated._mace_inputs(atoms)
    token = evaluator._prepare_factorized_graph(*inputs[:5])
    evaluator._compute_prepared_factorized_field(
        token,
        np.asarray(inputs[5]).reshape(-1),
        inputs[6],
        electric_field,
    )
    prepared = {
        "node_energies": np.array(evaluator.node_energies, copy=True),
        "node_forces": np.array(evaluator.node_forces, copy=True),
        "field_adjoint": np.array(evaluator.electric_field_adj, copy=True),
    }
    _assert_field_results_close(prepared, expected, tolerance)
    assert evaluator.factorized_schedule_build_count == 1
    assert evaluator.factorized_prepared_evaluation_count == 1


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_unregistered_r1_plugin_matches_runtime_and_all_and_survives_warm_load(
    request,
    tmp_path,
    monkeypatch,
    dtype,
):
    _require_host_plugin_evaluator()
    streamed_model_paths = request.getfixturevalue("streamed_model_paths")
    standard_path, _ = streamed_model_paths
    model, contract = _make_registry_miss_model(json.loads(standard_path.read_text()))
    model_path = tmp_path / "compact-standard-r1-registry-miss.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    source = render_jit_r1_host_plugin(contract, precision=dtype)
    metadata = jit_r1_host_plugin_metadata(contract, precision=dtype)
    compile_arguments = {
        "abi": {
            "host_plugin": metadata["abi"],
            "version": metadata["abi_version"],
        },
        "build": {
            "backend": "host",
            "precision": dtype,
            "generator": "r1-host-v2",
            "contract": metadata,
        },
        "cache_root": tmp_path / "jit-cache",
        "cxx": _compiler(),
        "cxx_flags": ("-ffast-math",),
        "artifact_name": "factorized_integration",
    }
    cold = prepare_jit_artifact(source, **compile_arguments)
    assert cold.available, (cold.reason, cold.diagnostics)
    assert cold.status == "built"

    baseline_live_objects = native_symmetrix._kokkos_live_object_count()
    atoms = _small_structure()
    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    expected = _evaluate(reference, atoms)

    candidate = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    evaluator = candidate.evaluator
    runtime = _evaluate(candidate, atoms)
    _assert_results_close(runtime, expected)
    assert evaluator.factorized_has_model_contract
    assert evaluator.factorized_model_payload_verified
    assert not evaluator.jit_host_plugin_ready
    assert not evaluator.factorized_jit_ready
    assert evaluator.factorized_jit_artifact_id == ""
    assert evaluator.factorized_jit_contract_fingerprint == ""

    evaluator._load_jit_host_plugin(str(cold.artifact_path))
    evaluator.set_streamed_edges("direct")
    assert evaluator.jit_host_plugin_ready
    assert evaluator.jit_host_plugin_path == str(cold.artifact_path)
    assert evaluator.jit_host_plugin_artifact_id == metadata["artifact_id"]
    assert evaluator.factorized_jit_ready
    assert evaluator.factorized_jit_artifact_id == metadata["artifact_id"]
    assert (
        evaluator.factorized_jit_contract_fingerprint
        == (metadata["contract_fingerprint"])
    )
    assert evaluator.factorized_source_strategy == "jit_plugin"

    launches = (
        evaluator.factorized_jit_forward_launch_count,
        evaluator.factorized_jit_reverse_launch_count,
    )
    generated = _evaluate(candidate, atoms)
    _assert_results_close(generated, expected)
    _assert_results_close(generated, runtime)
    assert evaluator.factorized_jit_forward_launch_count - launches[0] == 1
    assert evaluator.factorized_jit_reverse_launch_count - launches[1] == 2
    assert evaluator.factorized_selected_direct_forward_executor == "jit_all"
    assert evaluator.factorized_selected_direct_reverse_executor == "jit"

    warm = prepare_jit_artifact(source, **compile_arguments)
    assert warm.available
    assert warm.status == "cached"
    assert warm.cache_key == cold.cache_key
    assert warm.artifact_path == cold.artifact_path

    warm_candidate = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    warm_candidate.evaluator._load_jit_host_plugin(str(warm.artifact_path))
    warm_candidate.evaluator.set_streamed_edges("direct")
    assert warm_candidate.evaluator.jit_host_plugin_ready

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "automatic-jit-cache"))
    automatic = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
    )
    assert automatic.jit_status == "built"
    assert automatic.jit_cache_key is not None
    assert automatic.jit_artifact_path is not None
    assert automatic.evaluator.jit_host_plugin_ready
    _assert_results_close(_evaluate(automatic, atoms), expected)

    automatic_warm = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
    )
    assert automatic_warm.jit_status == "cached"
    assert automatic_warm.jit_cache_key == automatic.jit_cache_key
    assert automatic_warm.jit_artifact_path == automatic.jit_artifact_path
    assert automatic_warm.evaluator.jit_host_plugin_ready
    _assert_results_close(_evaluate(automatic_warm, atoms), expected)

    del evaluator, candidate
    gc.collect()
    warm_result = _evaluate(warm_candidate, atoms)
    _assert_results_close(warm_result, expected)

    del automatic_warm, automatic, warm_candidate, reference
    gc.collect()
    assert native_symmetrix._kokkos_live_object_count() == baseline_live_objects


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_macefield_host_plugin_matches_runtime_and_prepared_execution(
    request,
    tmp_path,
    dtype,
):
    _require_host_plugin_evaluator()
    checkpoint = request.getfixturevalue("macefield_model_path")
    extracted = extract_mace_data(
        checkpoint,
        species=[7, 13],
        head="mp-dielectric",
    )
    model, contract = _make_registry_miss_model(extracted)
    model_path = tmp_path / "compact-macefield-r1-registry-miss.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    source = render_jit_r1_host_plugin(contract, precision=dtype)
    metadata = jit_r1_host_plugin_metadata(contract, precision=dtype)
    artifact = prepare_jit_artifact(
        source,
        abi={
            "host_plugin": metadata["abi"],
            "version": metadata["abi_version"],
        },
        build={
            "backend": "host",
            "precision": dtype,
            "generator": "r1-host-v2",
            "contract": metadata,
        },
        cache_root=tmp_path / "jit-cache",
        cxx=_compiler(),
        cxx_flags=("-ffast-math",),
        artifact_name="factorized_macefield_integration",
    )
    assert artifact.available, (artifact.reason, artifact.diagnostics)

    atoms = Atoms(
        ["Al", "N", "Al", "N"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.1, 0.2],
            [0.3, 1.9, 0.1],
            [1.9, 1.8, 0.4],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    expected = _evaluate_field(reference, atoms, electric_field)
    candidate = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="generic",
    )
    runtime = _evaluate_field(candidate, atoms, electric_field)
    tolerance = 5e-5 if dtype == "float32" else 5e-11
    _assert_field_results_close(runtime, expected, tolerance)

    evaluator = candidate.evaluator
    evaluator._load_jit_host_plugin(str(artifact.artifact_path))
    evaluator.set_streamed_edges("direct")
    assert evaluator.jit_host_plugin_ready
    assert evaluator.jit_host_plugin_artifact_id == metadata["artifact_id"]
    launches = (
        evaluator.factorized_jit_forward_launch_count,
        evaluator.factorized_jit_reverse_launch_count,
    )
    generated = _evaluate_field(candidate, atoms, electric_field)
    _assert_field_results_close(generated, expected, tolerance)
    assert evaluator.factorized_jit_forward_launch_count - launches[0] == 1
    assert evaluator.factorized_jit_reverse_launch_count - launches[1] == 2

    inputs = candidate._mace_inputs(atoms)
    token = evaluator._prepare_factorized_graph(*inputs[:5])
    evaluator._compute_prepared_factorized_field(
        token,
        np.asarray(inputs[5]).reshape(-1),
        inputs[6],
        electric_field,
    )
    prepared = {
        "node_energies": np.array(evaluator.node_energies, copy=True),
        "node_forces": np.array(evaluator.node_forces, copy=True),
        "field_adjoint": np.array(evaluator.electric_field_adj, copy=True),
    }
    _assert_field_results_close(prepared, expected, tolerance)
    assert evaluator.factorized_schedule_build_count == 1
    assert evaluator.factorized_prepared_evaluation_count == 1


def test_automatic_jit_handles_different_channel_and_embedding_sizes(
    tmp_path,
    monkeypatch,
):
    _require_host_plugin_evaluator()
    _compiler()
    contract_path = (
        _REPOSITORY
        / "symmetrix/test/data/execution_contracts"
        / "jit_r1_mace_off23_small_contract.json"
    )
    base_model = foundation_r1_test_model(
        json.loads(contract_path.read_text()), atomic_numbers=(1, 8)
    )
    model, contract = _make_different_size_compact_model(base_model)
    assert contract["channels"] == 96
    assert contract["radial_embedding"] == 3
    model_path = tmp_path / "off23-small-c96-e3-compact.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    atoms = Atoms(
        ["H", "O", "H", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.4, 0.2, 0.1],
            [0.3, 1.6, 0.2],
            [1.5, 1.5, 0.4],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    expected = _evaluate(reference, atoms)

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    cold = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="speed",
    )
    assert cold.jit_status == "built"
    assert cold.evaluator.jit_host_plugin_ready
    assert cold.evaluator.factorized_model_payload_verified
    assert cold.evaluator.factorized_jit_artifact_id.startswith("jit-r1-gen11-")
    assert cold.evaluator.factorized_jit_contract_fingerprint.startswith("sha256:")
    cold_result = _evaluate(cold, atoms)
    _assert_results_close(cold_result, expected)
    assert cold.evaluator.factorized_jit_forward_launch_count == 1
    assert cold.evaluator.factorized_jit_reverse_launch_count == 2

    warm = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="speed",
    )
    assert warm.jit_status == "cached"
    assert warm.jit_cache_key == cold.jit_cache_key
    assert warm.jit_artifact_path == cold.jit_artifact_path
    assert warm.evaluator.jit_host_plugin_ready
    warm_result = _evaluate(warm, atoms)
    _assert_results_close(warm_result, expected)
    _assert_results_close(warm_result, cold_result)
