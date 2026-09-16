import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from compact_r1_model import foundation_r1_test_model
from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix
from symmetrix.execution_contract import (
    make_standard_m0_contract,
    make_standard_r0_contract,
)
from symmetrix.extract_mace_data import extract_mace_data
from symmetrix.jit import jit_nvrtc_information
from symmetrix.jit_device_artifact import prepare_jit_device_artifact


REPOSITORY = Path(__file__).resolve().parents[2]


def _require_cuda_nvrtc():
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("the imported Symmetrix extension was built without Kokkos")
    if not native_symmetrix._kokkos_is_initialized():
        try:
            native_symmetrix._init_kokkos()
        except RuntimeError as error:
            pytest.skip(f"Kokkos Cuda could not be initialized: {error}")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("operator RTC execution requires Kokkos Cuda")
    information = jit_nvrtc_information()
    if not information.get("available", False):
        pytest.skip(f"NVRTC is unavailable: {information.get('reason')}")


def _custom_scalar_operator_model():
    contract_path = (
        REPOSITORY
        / "symmetrix/test/data/execution_contracts"
        / "jit_r1_fixture_c4_e3_l0_contract.json"
    )
    r1_contract = json.loads(contract_path.read_text())
    m0_contract = make_standard_m0_contract(
        channels=4,
        type_count=1,
        input_l_max=0,
        output_l_max=0,
        correlation=2,
        monomials={0: [[0], [0, 0]]},
    )
    r0_contract = make_standard_r0_contract(
        channels=4,
        radial_embedding=3,
        edge_l_max=0,
        source_irreps=r1_contract["irreps"]["source"],
        edge_irreps=r1_contract["irreps"]["edge"],
        output_irreps=r1_contract["irreps"]["output"],
        output_l=[0],
        edge_l=[0],
        source_l=[0],
        instructions=[
            {
                "connection_mode": "uvu",
                "has_weight": True,
                "path_shape": [4, 1],
            }
        ],
    )
    model = foundation_r1_test_model(
        r1_contract,
        atomic_numbers=(1,),
        m0_contract=m0_contract,
        r0_contract=r0_contract,
    )
    model["A0_scaled"] = True
    model["compact_radial"]["networks"]["A0"] = {
        "shape": [3, 1],
        "weights": [[0.01, -0.02, 0.03]],
        "activation": "silu",
        "activation_scale": 1.0,
        "postprocess": "tanh-square",
    }
    return model


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


@pytest.fixture(scope="module")
def _runtime_specialized_macefield_path(tmp_path_factory, macefield_model_path):
    output_dir = tmp_path_factory.mktemp("rtc-macefield")
    model = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=16,
    )
    zero_component = 0 if 0 in model["M0_monomials"] else "0"
    existing = {tuple(term) for term in model["M0_monomials"][zero_component]}
    input_components = (model["l_max"] + 1) ** 2
    additional = next(
        list(term)
        for degree in range(1, 4)
        for term in combinations_with_replacement(range(input_components), degree)
        if term not in existing
    )
    model["M0_monomials"][zero_component].append(additional)
    for type_weights in model["M0_weights"].values():
        for channel, weights in type_weights[zero_component].items():
            weights.append(1e-4 * (1 + int(channel) % 7))
    monomials = {
        int(component): terms for component, terms in model["M0_monomials"].items()
    }
    model["execution_contracts"]["M0"] = make_standard_m0_contract(
        channels=model["num_channels"],
        type_count=len(model["atomic_numbers"]),
        input_l_max=model["l_max"],
        output_l_max=model["L_max"],
        correlation=max(len(term) for terms in monomials.values() for term in terms),
        monomials=monomials,
    )
    model_path = output_dir / "runtime-specialized-macefield.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    return model_path


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.jit
@pytest.mark.parametrize("schedule", ("chunk32", "table"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_cuda_low_memory_runtime_specializes_custom_m0_r0(
    tmp_path, monkeypatch, dtype, schedule
):
    _require_cuda_nvrtc()
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", schedule)
    model_path = tmp_path / "custom-scalar-operators.json"
    model_path.write_text(
        json.dumps(_custom_scalar_operator_model(), separators=(",", ":"))
    )
    atoms = Atoms(
        "H4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.1],
            [0.3, 1.2, 0.2],
            [1.3, 1.1, 0.4],
        ],
        cell=[7.0, 7.0, 7.0],
        pbc=False,
    )

    retained = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=False,
    )
    assert retained.evaluator.m0_implementation == "generic"
    assert retained.evaluator.r0_implementation == "generic"
    expected = _evaluate(retained, atoms)

    compact = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=True,
    )
    evaluator = compact.evaluator
    assert evaluator.m0_implementation == "device_module"
    assert evaluator.r0_implementation == "device_module"
    assert evaluator.m0_supports_low_memory
    assert evaluator.r0_supports_low_memory
    target = "sm_" + str(
        evaluator.execution_device_execution_environment["compute_capability_code"]
    )
    assert evaluator.m0_module_id.endswith(f"-{target}")
    assert evaluator.r0_module_id.endswith(f"-edge-batch1-{target}")
    assert compact.jit_operator_modules["M0"]["compiler"] == "nvrtc"
    assert compact.jit_operator_modules["R0"]["compiler"] == "nvrtc"
    assert compact.jit_operator_modules["M0"]["schedule"] == schedule
    assert evaluator.standard_m0_poly_values_capacity_bytes == 0
    assert evaluator.standard_m0_poly_adjoints_capacity_bytes == 0

    actual = _evaluate(compact, atoms)
    tolerance = 8e-5 if dtype == "float32" else 8e-11
    assert actual["energy"] == pytest.approx(expected["energy"], rel=0.0, abs=tolerance)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=0.0, atol=tolerance
        )
    assert evaluator.m0_module_forward_launch_count > 0
    assert evaluator.m0_module_reverse_launch_count > 0
    assert evaluator.m0_input_scale_adjoint_launch_count > 0
    assert evaluator.r0_module_launch_count > 0

    m0_path = compact.jit_operator_modules["M0"]["artifact_path"]
    r0_path = compact.jit_operator_modules["R0"]["artifact_path"]
    wrong_schedule = "table" if schedule == "chunk32" else "chunk32"
    with pytest.raises(RuntimeError, match="identity does not match"):
        evaluator._load_m0_device_module(m0_path, wrong_schedule, 8)
    with pytest.raises(RuntimeError, match="identity does not match"):
        evaluator._load_r0_device_module(m0_path, 8)
    damaged_path = tmp_path / "damaged-operator.cubin"
    damaged_path.write_bytes(Path(r0_path).read_bytes()[:128])
    with pytest.raises(RuntimeError, match="ELF"):
        evaluator._load_r0_device_module(str(damaged_path), 8)

    after_rejected_load = _evaluate(compact, atoms)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(
            after_rejected_load[name], actual[name], rtol=0.0, atol=tolerance
        )

    prepared = prepare_jit_device_artifact(
        model_path,
        precision=dtype,
        backend="cuda",
        cache_root=tmp_path / "jit-cache",
    )
    assert prepared.status == "cached"
    assert prepared.operator_modules["M0"]["status"] == "cached"
    assert prepared.operator_modules["R0"]["status"] == "cached"


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.jit
@pytest.mark.parametrize(
    "dtype,tolerance",
    (("float32", 1e-3), ("float64", 2e-8)),
)
def test_cuda_runtime_m0_supports_macefield_analytical_response(
    _runtime_specialized_macefield_path,
    tmp_path,
    monkeypatch,
    dtype,
    tolerance,
):
    _require_cuda_nvrtc()
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "chunk32")

    retained = Symmetrix(
        _runtime_specialized_macefield_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=False,
    )
    compact = Symmetrix(
        _runtime_specialized_macefield_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=True,
    )
    assert retained.evaluator.m0_implementation == "generic"
    assert retained.evaluator.r0_implementation == "builtin"
    assert compact.evaluator.m0_implementation == "device_module"
    assert compact.evaluator.r0_implementation == "builtin"
    assert compact.jit_operator_modules["M0"]["compiler"] == "nvrtc"
    assert compact.jit_operator_modules["R0"]["status"] == "builtin"

    properties = (
        "energy",
        "node_energy",
        "forces",
        "stress",
        "polarization",
        "polarizability",
        "becs",
    )
    results = {}
    for name, calculator in (("retained", retained), ("compact", compact)):
        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
        atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])
        calculator.calculate(atoms, properties=list(properties))
        results[name] = {
            property_name: np.array(calculator.results[property_name], copy=True)
            for property_name in properties
        }

    assert compact.evaluator.low_memory_policy == "speed"
    assert compact.evaluator.macefield_response_primal_reconstruction_count == 0
    assert compact.evaluator.m0_module_reverse_launch_count > 2
    for property_name in properties:
        np.testing.assert_allclose(
            results["compact"][property_name],
            results["retained"][property_name],
            rtol=0.0,
            atol=tolerance,
        )
