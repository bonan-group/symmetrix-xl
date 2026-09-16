import gc
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from compact_r1_model import foundation_r1_test_model
from debug_jit import debug_factorized_symmetrix

pytestmark = [pytest.mark.gpu, pytest.mark.hip]


def test_hip_execution_environment(accelerator_capabilities):
    environment = accelerator_capabilities
    assert environment["available"]
    assert environment["backend"] == "hip"
    assert environment["execution_space"] == "HIP"
    assert environment["device_memory"]
    assert environment["architecture"].startswith("gfx")
    assert environment["native_subgroup_width"] in (32, 64)
    assert environment["compute_unit_count"] > 0


def test_hip_rocblas_link_reports_batched_blas(accelerator_capabilities):
    linked_libraries = Path("/proc/self/maps").read_text()
    if "librocblas.so" not in linked_libraries:
        pytest.skip("requires a rocBLAS-enabled HIP build")

    assert accelerator_capabilities["batched_blas_available"]


def test_hip_device_sentinel(accelerator_capabilities):
    import symmetrix

    assert symmetrix._kokkos_device_sentinel()


def _evaluate(calculator, atoms):
    calculator.calculate(atoms, properties=["energy", "forces"])
    return (
        float(calculator.results["energy"]),
        np.array(calculator.results["forces"], copy=True),
    )


def test_hip_r1_jit_wave_and_serial_match_generic(
    accelerator_capabilities,
    tmp_path,
    monkeypatch,
):
    from symmetrix import Symmetrix

    assert accelerator_capabilities["jit_available"]
    contract_path = (
        Path(__file__).resolve().parent
        / "data/execution_contracts/jit_r1_mace_off23_small_contract.json"
    )
    model_path = tmp_path / "compact-r1.json"
    model_path.write_text(
        json.dumps(foundation_r1_test_model(json.loads(contract_path.read_text())))
    )
    atoms = Atoms(
        "H6",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.2],
            [0.2, 1.0, 0.1],
            [1.1, 1.0, 0.3],
            [0.4, 0.3, 1.1],
            [1.2, 0.5, 1.2],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_HIP_JIT_BACKEND", "hiprtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))

    generic = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    expected_energy, expected_forces = _evaluate(generic, atoms)

    monkeypatch.setenv("SYMMETRIX_JIT_HIP_R1_EDGE_STRATEGY", "wave")
    wave = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    assert wave.jit_status in ("built", "cached")
    assert wave.evaluator.jit_hip_plugin_ready
    wave_launches = (
        wave.evaluator.factorized_jit_forward_launch_count,
        wave.evaluator.factorized_jit_reverse_launch_count,
    )
    wave_energy, wave_forces = _evaluate(wave, atoms)
    assert wave.evaluator.factorized_jit_forward_launch_count == (wave_launches[0] + 1)
    assert wave.evaluator.factorized_jit_reverse_launch_count == (wave_launches[1] + 2)

    monkeypatch.setenv("SYMMETRIX_JIT_HIP_R1_EDGE_STRATEGY", "serial")
    serial = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    assert serial.jit_status in ("built", "cached")
    assert serial.evaluator.jit_hip_plugin_ready
    assert serial.jit_cache_key != wave.jit_cache_key
    serial_energy, serial_forces = _evaluate(serial, atoms)

    for actual_energy, actual_forces in (
        (wave_energy, wave_forces),
        (serial_energy, serial_forces),
    ):
        assert actual_energy == pytest.approx(expected_energy, abs=5e-4)
        np.testing.assert_allclose(actual_forces, expected_forces, rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(wave_forces, serial_forces, rtol=0.0, atol=5e-4)

    del serial, wave, generic
    gc.collect()


@pytest.mark.parametrize(
    "dtype,tolerance",
    (("float32", 5e-4), ("float64", 1e-10)),
    ids=("float32", "float64"),
)
def test_hip_r1_hiprtc_module_matches_generic(
    accelerator_capabilities,
    tmp_path,
    monkeypatch,
    dtype,
    tolerance,
):
    from symmetrix import Symmetrix

    contract_path = (
        Path(__file__).resolve().parent
        / "data/execution_contracts/jit_r1_mace_off23_small_contract.json"
    )
    model_path = tmp_path / "compact-r1-hiprtc.json"
    model_path.write_text(
        json.dumps(foundation_r1_test_model(json.loads(contract_path.read_text())))
    )
    atoms = Atoms(
        "H4",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.2],
            [0.2, 1.0, 0.1],
            [1.1, 1.0, 0.3],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_HIP_JIT_BACKEND", "hiprtc")
    monkeypatch.setenv("SYMMETRIX_JIT_HIP_R1_EDGE_STRATEGY", "serial")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))

    generic = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="factorized",
    )
    expected_energy, expected_forces = _evaluate(generic, atoms)
    module = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="factorized",
    )
    assert module.jit_compiler_backend == "hiprtc"
    assert module.jit_status in ("built", "cached")
    assert module.jit_artifact_path.endswith(".hsaco")
    assert module.evaluator.jit_hip_plugin_ready
    actual_energy, actual_forces = _evaluate(module, atoms)
    assert actual_energy == pytest.approx(expected_energy, abs=tolerance)
    np.testing.assert_allclose(actual_forces, expected_forces, rtol=0.0, atol=tolerance)

    cached_module = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="factorized",
    )
    assert cached_module.jit_status == "cached"
    assert cached_module.jit_cache_key == module.jit_cache_key
    cached_energy, cached_forces = _evaluate(cached_module, atoms)
    assert cached_energy == pytest.approx(actual_energy, abs=tolerance)
    np.testing.assert_allclose(cached_forces, actual_forces, rtol=0.0, atol=tolerance)

    del cached_module, module, generic
    gc.collect()


def _assert_hip_standard_m0_matches_runtime(tmp_path, dtype, tolerance):
    contracts = Path(__file__).resolve().parent / "data/execution_contracts"
    r1_contract = json.loads(
        (contracts / "jit_r1_mace_mpa0_medium_contract.json").read_text()
    )
    m0_contract = json.loads((contracts / "standard_m0_contract.json").read_text())
    model_path = tmp_path / "compact-m0.json"
    model_path.write_text(
        json.dumps(
            foundation_r1_test_model(
                r1_contract,
                atomic_numbers=(7, 13),
                m0_contract=m0_contract,
            )
        )
    )
    atoms = Atoms(
        numbers=[7, 13, 7, 13],
        positions=[
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.2],
            [0.2, 1.0, 0.1],
            [1.1, 1.0, 0.3],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    calculator = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    assert evaluator.standard_m0_module_ready
    assert evaluator.standard_m0_module_fallback_reason == ""
    assert evaluator.standard_m0_module_id == ("standard-m0-module-module-v1")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    def evaluate(executor):
        evaluator._set_standard_m0_executor(executor)
        launches = (
            evaluator.standard_m0_module_forward_launch_count,
            evaluator.standard_m0_module_reverse_launch_count,
        )
        evaluator.compute_node_energies_forces(*native_args)
        result = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": float(result["energy"]),
            "forces": np.array(result["forces"], copy=True),
            "M0": np.array(evaluator.M0, copy=True),
            "A0_adj": np.array(evaluator.A0_adj, copy=True),
            "launches": (
                evaluator.standard_m0_module_forward_launch_count - launches[0],
                evaluator.standard_m0_module_reverse_launch_count - launches[1],
            ),
        }

    runtime = evaluate("runtime")
    standard = evaluate("standard")
    assert runtime["launches"] == (0, 0)
    assert standard["launches"] == (1, 1)
    assert evaluator.standard_m0_poly_values_active_bytes == 0
    assert evaluator.standard_m0_poly_adjoints_active_bytes == 0
    assert standard["energy"] == pytest.approx(runtime["energy"], abs=tolerance)
    for name in ("forces", "M0", "A0_adj"):
        np.testing.assert_allclose(
            standard[name], runtime[name], rtol=0.0, atol=tolerance
        )

    evaluator = None
    calculator = None
    gc.collect()


def test_hip_standard_m0_matches_runtime(accelerator_capabilities, tmp_path):
    _assert_hip_standard_m0_matches_runtime(tmp_path, "float32", 3e-5)


def test_hip_standard_m0_float64_matches_runtime(accelerator_capabilities, tmp_path):
    _assert_hip_standard_m0_matches_runtime(tmp_path, "float64", 1e-10)


def test_hip_standard_r0_v2_matches_runtime(accelerator_capabilities, tmp_path):
    contracts = Path(__file__).resolve().parent / "data/execution_contracts"
    r0_contract = json.loads((contracts / "standard_r0_contract.json").read_text())
    r1_contract = json.loads(
        (contracts / "jit_r1_mace_mpa0_medium_contract.json").read_text()
    )
    model_path = tmp_path / "compact-r0.json"
    model_path.write_text(
        json.dumps(
            foundation_r1_test_model(
                r1_contract,
                atomic_numbers=(7, 13),
                r0_contract=r0_contract,
            )
        )
    )
    atoms = Atoms(
        numbers=[7, 13, 7, 13],
        positions=[
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.2],
            [0.2, 1.0, 0.1],
            [1.1, 1.0, 0.3],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    calculator = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    assert evaluator.standard_r0_module_ready
    assert evaluator.standard_r0_module_fallback_reason == ""
    assert evaluator.standard_r0_module_id == "standard-r0-module-module-v2"
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_reverse_executor("runtime")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    def evaluate(executor):
        evaluator._set_standard_r0_executor(executor)
        launches = evaluator.standard_r0_module_launch_count
        evaluator.compute_node_energies_forces(*native_args)
        result = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": float(result["energy"]),
            "forces": np.array(result["forces"], copy=True),
            "A0": np.array(evaluator.A0, copy=True),
            "launches": evaluator.standard_r0_module_launch_count - launches,
        }

    runtime = evaluate("v1")
    assert runtime["launches"] == 0
    for executor in ("v2_receiver", "v2_edge16", "v2_edge32"):
        candidate = evaluate(executor)
        assert evaluator.standard_r0_selected_executor == executor
        assert candidate["launches"] == 2
        assert candidate["energy"] == pytest.approx(runtime["energy"], abs=3e-5)
        for name in ("forces", "A0"):
            np.testing.assert_allclose(
                candidate[name], runtime[name], rtol=0.0, atol=3e-5
            )

    evaluator = None
    calculator = None
    gc.collect()
