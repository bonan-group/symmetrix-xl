import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from compact_r1_model import foundation_r1_test_model
from debug_jit import debug_factorized_symmetrix

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _evaluate(calculator, atoms):
    calculator.calculate(atoms, properties=["energy", "forces"])
    return (
        float(calculator.results["energy"]),
        np.array(calculator.results["forces"], copy=True),
    )


def test_cuda_r1_nvrtc_module_matches_generic(
    accelerator_capabilities,
    tmp_path,
    monkeypatch,
):
    from symmetrix import Symmetrix
    from symmetrix.jit import jit_nvrtc_information

    assert accelerator_capabilities["jit_available"]
    nvrtc = jit_nvrtc_information()
    assert nvrtc["available"], nvrtc.get("reason")
    contract_path = (
        Path(__file__).resolve().parent
        / "data/execution_contracts/jit_r1_mace_off23_small_contract.json"
    )
    model_path = tmp_path / "compact-r1-nvrtc.json"
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
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))

    generic = debug_factorized_symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    expected_energy, expected_forces = _evaluate(generic, atoms)

    module = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    assert module.jit_compiler_backend == "nvrtc"
    assert module.jit_status in ("built", "cached")
    assert module.jit_artifact_path.endswith(".cubin")
    assert module.evaluator.jit_cuda_plugin_ready
    launches = (
        module.evaluator.factorized_jit_forward_launch_count,
        module.evaluator.factorized_jit_reverse_launch_count,
    )
    actual_energy, actual_forces = _evaluate(module, atoms)
    assert module.evaluator.factorized_jit_forward_launch_count == launches[0] + 1
    assert module.evaluator.factorized_jit_reverse_launch_count == launches[1] + 1
    assert actual_energy == pytest.approx(expected_energy, abs=5e-4)
    np.testing.assert_allclose(
        actual_forces,
        expected_forces,
        rtol=0.0,
        atol=5e-4,
    )
