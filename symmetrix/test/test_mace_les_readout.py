import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from compact_r1_model import add_standard_base_contracts, foundation_r1_test_model
from symmetrix import symmetrix as native_symmetrix
from symmetrix.extract_mace_data import _extract_mace_les_parameters

CONTRACT_ROOT = Path(__file__).resolve().parent / "data" / "execution_contracts"


@pytest.fixture(scope="module")
def les_readout_model(tmp_path_factory):
    contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_fixture_c4_e3_l0_contract.json").read_text()
    )
    model = add_standard_base_contracts(foundation_r1_test_model(contract))
    model["model_type"] = "MACELES"
    model["les"] = {
        "schema": "symmetrix.model.mace_les",
        "version": 1,
        "kind": "monopole",
        "periodicity": "geometry",
        "ewald": {
            "sigma": 1.0,
            "dl": 2.0,
            "remove_self_interaction": True,
        },
        "output_scale": 0.25,
        "mace_num_interactions": 2,
        "mace_num_channels": 4,
        "mace_contract_fingerprints": {
            "R0": model["execution_contracts"]["R0"]["fingerprint"],
            "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
            "R1": model["execution_contracts"]["R1"]["fingerprint"],
        },
        "readout_1_weights": [0.3, -0.2, 0.5, 0.1],
        "readout_2_hidden_size": 3,
        "readout_2_weights_1": [
            0.1,
            -0.2,
            0.3,
            0.4,
            -0.3,
            0.2,
            -0.1,
            0.5,
            0.25,
            0.15,
            -0.35,
            0.45,
        ],
        "readout_2_weights_2": [0.7, -0.4, 0.2],
        "readout_2_scale_factor": 1.5,
    }
    path = tmp_path_factory.mktemp("mace-les-readout") / "model.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    return path


@pytest.fixture(scope="module")
def les_readout_l1_model(tmp_path_factory):
    contract = json.loads((CONTRACT_ROOT / "jit_r1_contract.json").read_text())
    model = add_standard_base_contracts(foundation_r1_test_model(contract))
    channels = model["num_channels"]
    readout_1 = np.zeros(channels)
    readout_1[:4] = [0.3, -0.2, 0.5, 0.1]
    readout_2 = np.zeros(channels)
    readout_2[:4] = [0.1, -0.2, 0.3, 0.4]
    model["model_type"] = "MACELES"
    model["les"] = {
        "schema": "symmetrix.model.mace_les",
        "version": 1,
        "kind": "monopole",
        "periodicity": "geometry",
        "ewald": {
            "sigma": 1.0,
            "dl": 2.0,
            "remove_self_interaction": True,
        },
        "output_scale": 0.25,
        "mace_num_interactions": 2,
        "mace_num_channels": channels,
        "mace_contract_fingerprints": {
            "R0": model["execution_contracts"]["R0"]["fingerprint"],
            "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
            "R1": model["execution_contracts"]["R1"]["fingerprint"],
        },
        "readout_1_weights": readout_1.tolist(),
        "readout_2_hidden_size": 1,
        "readout_2_weights_1": readout_2.tolist(),
        "readout_2_weights_2": [0.7],
        "readout_2_scale_factor": 1.5,
    }
    path = tmp_path_factory.mktemp("mace-les-readout-l1") / "model.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    return path


def _expected_readouts(h1, h2):
    weights_1 = np.array([0.3, -0.2, 0.5, 0.1])
    hidden_weights = np.array(
        [
            [0.1, -0.2, 0.3, 0.4],
            [-0.3, 0.2, -0.1, 0.5],
            [0.25, 0.15, -0.35, 0.45],
        ]
    )
    output_weights = np.array([0.7, -0.4, 0.2])
    q1 = h1 @ weights_1
    hidden_linear = h2 @ hidden_weights.T
    hidden = 1.5 * hidden_linear / (1.0 + np.exp(-hidden_linear))
    q2 = hidden @ output_weights
    return q1, q2, 0.25 * (q1 + q2)


def _official_maceles_readout_case(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    o3 = pytest.importorskip("e3nn.o3")
    blocks = pytest.importorskip("mace.modules.blocks")
    extensions = pytest.importorskip("mace.modules.extensions")

    model = extensions.MACELES.__new__(extensions.MACELES)
    torch.nn.Module.__init__(model)
    linear = blocks.LinearReadoutBlock(o3.Irreps("4x0e")).double()
    nonlinear = blocks.NonLinearReadoutBlock(
        o3.Irreps("4x0e"),
        o3.Irreps("3x0e"),
        torch.nn.functional.silu,
        num_heads=1,
    ).double()
    linear_weights = torch.tensor([0.6, -0.4, 1.0, 0.2], dtype=torch.float64)
    nonlinear_weights_1 = torch.tensor(
        [
            0.2,
            -0.4,
            0.6,
            0.8,
            -0.6,
            0.4,
            -0.2,
            1.0,
            0.5,
            0.3,
            -0.7,
            0.9,
        ],
        dtype=torch.float64,
    )
    nonlinear_weights_2 = torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64)
    with torch.no_grad():
        linear.linear.weight.copy_(linear_weights)
        nonlinear.linear_1.weight.copy_(nonlinear_weights_1)
        nonlinear.linear_2.weight.copy_(nonlinear_weights_2)

    fuse = torch.tensor(
        [
            [1.1, 0.1, 0.0, 0.0],
            [0.0, 0.9, 0.2, 0.0],
            [0.0, 0.0, 1.2, -0.1],
            [0.1, 0.0, 0.0, 0.8],
        ],
        dtype=torch.float64,
    )
    model.interactions = [
        SimpleNamespace(),
        SimpleNamespace(linear_up=SimpleNamespace(weight=2.0 * fuse.flatten())),
    ]
    model.les_readouts = torch.nn.ModuleList([linear, nonlinear])
    model.les = SimpleNamespace(
        sigma=1.0,
        dl=2.0,
        remove_self_interaction=True,
        use_atomwise=False,
        use_fixed_atomic_charges=False,
        use_atomic_alpha=False,
        use_epsilon_r_scaling=False,
    )
    model.les_output_scale = 0.37
    model.use_dipoles = False
    model.use_quads = False
    model.use_induced_charges = False
    model.use_induced_dipoles = False
    model.use_anisotropic_polarizability = False
    model.compute_bec = False

    upstream_h1 = torch.tensor(
        [
            [0.2, -0.4, 0.7, 0.1],
            [-0.3, 0.6, 0.25, -0.8],
            [0.9, 0.05, -0.2, 0.4],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    upstream_h2 = torch.tensor(
        [
            [0.5, -0.1, 0.3, 0.8],
            [-0.6, 0.2, 0.9, -0.4],
            [0.15, 0.75, -0.5, 0.35],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    upstream_q1 = linear(upstream_h1).squeeze(-1)
    upstream_q2 = nonlinear(upstream_h2).squeeze(-1)
    upstream_q = model.les_output_scale * (upstream_q1 + upstream_q2)
    charge_adjoints = torch.tensor([0.8, -0.35, 1.1], dtype=torch.float64)
    torch.dot(upstream_q, charge_adjoints).backward()

    # Symmetrix retains the fused first-interaction features. Convert the
    # official MACELES adjoint through that same invertible fusion map.
    native_h1 = upstream_h1.detach() @ fuse
    native_h1_adjoint = upstream_h1.grad @ torch.linalg.inv(fuse).T

    parameters = _extract_mace_les_parameters(
        model,
        num_channels=4,
        l_max=0,
        base_contracts={
            "R0": {"fingerprint": "sha256:" + "1" * 64},
            "M0": {"semantic_fingerprint": "sha256:" + "2" * 64},
            "R1": {"fingerprint": "sha256:" + "3" * 64},
        },
    )
    return SimpleNamespace(
        payload=parameters,
        h1=native_h1.numpy(force=True),
        h2=upstream_h2.detach().numpy(force=True),
        q1=upstream_q1.detach().numpy(force=True),
        q2=upstream_q2.detach().numpy(force=True),
        q=upstream_q.detach().numpy(force=True),
        charge_adjoints=charge_adjoints.numpy(force=True),
        h1_adjoint=native_h1_adjoint.detach().numpy(force=True),
        h2_adjoint=upstream_h2.grad.detach().numpy(force=True),
        upstream_module=type(model).__module__,
    )


@pytest.mark.parametrize(
    ("class_name", "h1_dtype", "rtol", "atol"),
    [
        ("MACELESKokkosFloat", np.float32, 3.0e-6, 3.0e-7),
        ("MACELESKokkos", np.float64, 3.0e-13, 3.0e-14),
    ],
)
def test_native_les_charge_readouts_match_official_maceles(
    tmp_path, monkeypatch, class_name, h1_dtype, rtol, atol
):
    case = _official_maceles_readout_case(monkeypatch)
    assert case.upstream_module == "mace.modules.extensions"
    contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_fixture_c4_e3_l0_contract.json").read_text()
    )
    model = add_standard_base_contracts(foundation_r1_test_model(contract))
    case.payload["mace_contract_fingerprints"] = {
        "R0": model["execution_contracts"]["R0"]["fingerprint"],
        "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
        "R1": model["execution_contracts"]["R1"]["fingerprint"],
    }
    model["model_type"] = "MACELES"
    model["les"] = case.payload
    model_path = tmp_path / "official-maceles-readouts.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = getattr(native_symmetrix, class_name)(str(model_path))

    evaluator._evaluate_les_charge_readouts_for_testing(
        case.h1.astype(h1_dtype), case.h2
    )

    np.testing.assert_allclose(
        evaluator.les_charge_readout_1, case.q1, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        evaluator.les_charge_readout_2, case.q2, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(evaluator.latent_charges, case.q, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("class_name", "h1_dtype", "rtol", "atol"),
    [
        ("MACELESKokkosFloat", np.float32, 5.0e-6, 5.0e-7),
        ("MACELESKokkos", np.float64, 3.0e-13, 3.0e-14),
    ],
)
def test_les_charge_readout_reverse_matches_official_maceles_autograd(
    tmp_path, monkeypatch, class_name, h1_dtype, rtol, atol
):
    case = _official_maceles_readout_case(monkeypatch)
    contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_fixture_c4_e3_l0_contract.json").read_text()
    )
    model = add_standard_base_contracts(foundation_r1_test_model(contract))
    case.payload["mace_contract_fingerprints"] = {
        "R0": model["execution_contracts"]["R0"]["fingerprint"],
        "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
        "R1": model["execution_contracts"]["R1"]["fingerprint"],
    }
    model["model_type"] = "MACELES"
    model["les"] = case.payload
    model_path = tmp_path / "official-maceles-reverse.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = getattr(native_symmetrix, class_name)(str(model_path))

    h1_adjoint, h2_adjoint = evaluator._reverse_les_charge_readouts_for_testing(
        case.h1.astype(h1_dtype), case.h2, case.charge_adjoints
    )

    np.testing.assert_allclose(
        h1_adjoint[:, 0, :], case.h1_adjoint, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(h2_adjoint, case.h2_adjoint, rtol=rtol, atol=atol)


def test_les_charge_readout_reverse_accumulates_energy_seeds_once(
    tmp_path, monkeypatch
):
    case = _official_maceles_readout_case(monkeypatch)
    contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_fixture_c4_e3_l0_contract.json").read_text()
    )
    model = add_standard_base_contracts(foundation_r1_test_model(contract))
    case.payload["mace_contract_fingerprints"] = {
        "R0": model["execution_contracts"]["R0"]["fingerprint"],
        "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
        "R1": model["execution_contracts"]["R1"]["fingerprint"],
    }
    model["model_type"] = "MACELES"
    model["les"] = case.payload
    model_path = tmp_path / "official-maceles-seeded-reverse.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACELESKokkos(str(model_path))
    h1_seed = np.arange(12, dtype=np.float64).reshape(3, 1, 4) / 17.0
    h2_seed = np.arange(12, dtype=np.float64).reshape(3, 4) / 13.0
    accumulated_h1, accumulated_h2 = evaluator._reverse_les_charge_readouts_for_testing(
        case.h1, case.h2, case.charge_adjoints, h1_seed, h2_seed
    )

    np.testing.assert_allclose(
        accumulated_h1[:, 0, :], h1_seed[:, 0, :] + case.h1_adjoint, atol=2.0e-13
    )
    np.testing.assert_allclose(accumulated_h2, h2_seed + case.h2_adjoint, atol=2.0e-13)


def test_les_charge_readout_reverse_preserves_non_scalar_h1_seeds(
    les_readout_l1_model,
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACELESKokkos(str(les_readout_l1_model))
    channels = 128
    h1 = np.zeros((2, channels))
    h2 = np.zeros((2, channels))
    charge_adjoints = np.array([0.8, -0.35])
    h1_seed = np.arange(2 * 4 * channels, dtype=np.float64).reshape(2, 4, channels)
    h2_seed = np.zeros((2, channels))

    accumulated_h1, _ = evaluator._reverse_les_charge_readouts_for_testing(
        h1, h2, charge_adjoints, h1_seed, h2_seed
    )

    np.testing.assert_array_equal(accumulated_h1[:, 1:, :], h1_seed[:, 1:, :])


def test_les_charge_readout_reverse_reuse_exports_only_active_nodes(
    les_readout_model,
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACELESKokkos(str(les_readout_model))
    evaluator._reverse_les_charge_readouts_for_testing(
        np.zeros((3, 4)), np.zeros((3, 4)), np.zeros(3)
    )

    h1_seed = np.arange(8, dtype=np.float64).reshape(2, 1, 4) / 17.0
    h2_seed = np.arange(8, dtype=np.float64).reshape(2, 4) / 13.0
    charge_adjoints = np.array([0.8, -0.35])
    h1_adjoint, h2_adjoint = evaluator._reverse_les_charge_readouts_for_testing(
        np.zeros((2, 4)),
        np.zeros((2, 4)),
        charge_adjoints,
        h1_seed,
        h2_seed,
    )

    expected_h1 = h1_seed.copy()
    expected_h1[:, 0, :] += (
        0.25 * charge_adjoints[:, None] * np.array([0.3, -0.2, 0.5, 0.1])[None, :]
    )
    assert h1_adjoint.shape == (2, 1, 4)
    assert h2_adjoint.shape == (2, 4)
    np.testing.assert_allclose(h1_adjoint, expected_h1, atol=2.0e-13)


@pytest.mark.parametrize(
    ("class_name", "dtype", "rtol", "atol"),
    [
        ("MACELESKokkosFloat", np.float32, 2.0e-6, 2.0e-7),
        ("MACELESKokkos", np.float64, 2.0e-13, 2.0e-14),
    ],
)
def test_native_les_charge_readouts_match_independent_reference(
    les_readout_model, class_name, dtype, rtol, atol
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = getattr(native_symmetrix, class_name)(str(les_readout_model))
    h1 = np.array(
        [
            [0.2, -0.4, 0.7, 0.1],
            [-0.3, 0.6, 0.25, -0.8],
            [0.9, 0.05, -0.2, 0.4],
        ],
        dtype=dtype,
    )
    h2 = np.array(
        [
            [0.5, -0.1, 0.3, 0.8],
            [-0.6, 0.2, 0.9, -0.4],
            [0.15, 0.75, -0.5, 0.35],
        ],
        dtype=np.float64,
    )

    evaluator._evaluate_les_charge_readouts_for_testing(h1, h2)
    expected_q1, expected_q2, expected = _expected_readouts(h1, h2)

    np.testing.assert_allclose(
        evaluator.les_charge_readout_1, expected_q1, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        evaluator.les_charge_readout_2, expected_q2, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(evaluator.latent_charges, expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        evaluator.latent_charges,
        0.25 * (evaluator.les_charge_readout_1 + evaluator.les_charge_readout_2),
        rtol=rtol,
        atol=atol,
    )


def test_les_charge_workspace_grows_and_reuses_capacity(les_readout_model):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACELESKokkos(str(les_readout_model))

    evaluator._evaluate_les_charge_readouts_for_testing(
        np.zeros((3, 4)), np.zeros((3, 4))
    )
    after_growth = evaluator.les_charge_workspace_diagnostics
    evaluator._evaluate_les_charge_readouts_for_testing(
        np.zeros((2, 4)), np.zeros((2, 4))
    )
    after_reuse = evaluator.les_charge_workspace_diagnostics
    evaluator._evaluate_les_charge_readouts_for_testing(
        np.zeros((5, 4)), np.zeros((5, 4))
    )
    after_second_growth = evaluator.les_charge_workspace_diagnostics

    assert after_growth["active_nodes"] == 3
    assert after_growth["capacity_nodes"] >= 3
    assert after_growth["replacement_count"] == 1
    assert after_reuse["active_nodes"] == 2
    assert after_reuse["capacity_nodes"] == after_growth["capacity_nodes"]
    assert after_reuse["replacement_count"] == 1
    assert after_reuse["reuse_count"] == 1
    assert after_reuse["active_bytes"] < after_reuse["capacity_bytes"]
    assert after_second_growth["active_nodes"] == 5
    assert after_second_growth["capacity_nodes"] >= 5
    assert after_second_growth["capacity_nodes"] > after_reuse["capacity_nodes"]
    assert after_second_growth["replacement_count"] == 2
    assert after_second_growth["reuse_count"] == 1
