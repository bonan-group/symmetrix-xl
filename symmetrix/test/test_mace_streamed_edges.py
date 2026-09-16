import copy
import gc
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from compact_r1_model import foundation_r1_test_model
from debug_jit import debug_factorized_symmetrix
from scipy.interpolate import CubicSpline
from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix
from symmetrix.execution_contract import make_standard_m0_contract

try:
    from symmetrix.extract_mace_data import extract_mace_data
except ImportError as exc:
    pytest.skip(
        f"Compact MACE extraction dependencies are not available: {exc}",
        allow_module_level=True,
    )


_EXECUTION_FIXED_WEIGHT_EXECUTION_SPACES = frozenset(
    {"Cuda", "HIP", "OpenMP", "Serial"}
)


def _kokkos_execution_space():
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    return native_symmetrix._kokkos_default_execution_space()


def _execution_fixed_weight_backend_supported():
    return _kokkos_execution_space() in _EXECUTION_FIXED_WEIGHT_EXECUTION_SPACES


def _require_execution_fixed_weight_backend():
    execution_space = _kokkos_execution_space()
    if execution_space not in _EXECUTION_FIXED_WEIGHT_EXECUTION_SPACES:
        pytest.skip(
            "Execution fixed-weight paths require Kokkos CUDA, HIP, or a "
            "HostSpace backend (OpenMP/Serial)"
        )
    return execution_space


def _require_receiver_factorized_host_backend():
    execution_space = _kokkos_execution_space()
    if execution_space not in ("OpenMP", "Serial"):
        pytest.skip("receiver-factorized RTC requires Kokkos OpenMP or Serial")
    return execution_space


@pytest.fixture(scope="module")
def streamed_model_paths(tmp_path_factory, macefield_model_path):
    output_dir = tmp_path_factory.mktemp("mace-streamed-edges")
    field_data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=16,
    )
    field_path = output_dir / "compact-field.json"
    field_path.write_text(json.dumps(field_data, separators=(",", ":")))

    standard_data = copy.deepcopy(field_data)
    standard_data["model_type"] = "MACE"
    standard_data["has_field_coupling"] = False
    standard_data.pop("field_couplings", None)
    standard_path = output_dir / "compact-standard.json"
    standard_path.write_text(json.dumps(standard_data, separators=(",", ":")))
    return standard_path, field_path


@pytest.fixture(scope="module")
def scalar_standard_model_path(tmp_path_factory):
    contracts = Path(__file__).resolve().parent / "data" / "execution_contracts"
    m0_contract = json.loads((contracts / "standard_m0_contract.json").read_text())
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
    r0_contract = json.loads((contracts / "standard_r0_contract.json").read_text())
    r1_contract = json.loads(
        (contracts / "jit_r1_mace_omat0_small_contract.json").read_text()
    )
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
    output = tmp_path_factory.mktemp("scalar-standard-m0") / "model.json"
    output.write_text(json.dumps(model, separators=(",", ":")))
    return output


@pytest.fixture(scope="module")
def legacy_standard_model_path(tmp_path_factory, macefield_model_path):
    legacy_data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=16,
        radial_format="pair-splines",
    )
    legacy_data["model_type"] = "MACE"
    legacy_data["has_field_coupling"] = False
    legacy_data.pop("field_couplings", None)
    legacy_path = tmp_path_factory.mktemp("mace-legacy-edges") / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_data))
    return legacy_path


def _small_structure():
    return Atoms(
        ["Al", "N", "Al", "N"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [1.9, 1.8, 0.7],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )


def _ragged_structure():
    atoms = _small_structure()
    atoms.positions[-1] = [12.0, 11.0, 10.0]
    return atoms


def _torch_compact_radial_weight_gradients(
    model,
    network_name,
    pair_types,
    coefficient_adjoints,
    factorized,
    final_projection_adjoints,
):
    """Differentiate the extracted compact network through an independent graph."""
    compact = model["compact_radial"]
    network = compact["networks"][network_name]
    node_count = compact["num_spline_points"]
    grid_min = compact["spline_grid_min"]
    h = (model["r_cut"] - grid_min) / (node_count - 1)
    radii = grid_min + np.arange(node_count, dtype=np.float64) * h

    transform = compact["distance_transform"]
    cutoff = compact["cutoff"]
    scaled_cutoff_radii = radii / cutoff["r_max"]
    p = cutoff["p"]
    cutoff_values = np.where(
        radii < cutoff["r_max"],
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * scaled_cutoff_radii**p
        + p * (p + 2.0) * scaled_cutoff_radii ** (p + 1)
        - (p * (p + 1.0) / 2.0) * scaled_cutoff_radii ** (p + 2),
        0.0,
    )
    pair_features = []
    for type_i, type_j in pair_types:
        transformed_radii = radii
        if transform["type"] == "agnesi":
            r0 = 0.5 * (
                transform["covalent_radii"][type_i]
                + transform["covalent_radii"][type_j]
            )
            scaled_radii = radii / r0
            transformed_radii = 1.0 / (
                1.0
                + transform["a"]
                * scaled_radii ** transform["q"]
                / (1.0 + scaled_radii ** (transform["q"] - transform["p"]))
            )
        pair_features.append(
            compact["basis"]["prefactor"]
            * np.sin(
                transformed_radii[:, None]
                * np.asarray(compact["basis"]["weights"])[None, :]
            )
            / transformed_radii[:, None]
            * cutoff_values[:, None]
        )
    values = torch.as_tensor(np.asarray(pair_features), dtype=torch.float64)
    weights = [
        torch.tensor(
            serialized,
            dtype=torch.float64,
        )
        .reshape(output_width, input_width)
        .requires_grad_()
        for serialized, input_width, output_width in zip(
            network["weights"], network["shape"][:-1], network["shape"][1:]
        )
    ]
    executed_layer_count = len(weights) - int(factorized)
    for layer in range(executed_layer_count):
        values = values @ weights[layer].T
        if layer + 1 < len(weights):
            values = network["activation_scale"] * torch.nn.functional.silu(values)
    if not factorized and network.get("postprocess") == "tanh-square":
        values = torch.tanh(values * values)

    grid = grid_min + np.arange(node_count, dtype=np.float64) * h
    derivative_operator = CubicSpline(
        grid,
        np.eye(node_count),
        axis=0,
        bc_type=("not-a-knot", "clamped"),
    )(grid, 1)
    derivatives = torch.einsum(
        "ij,pjf->pif",
        torch.as_tensor(derivative_operator, dtype=torch.float64),
        values,
    )
    left_values = values[:, :-1, :]
    right_values = values[:, 1:, :]
    left_derivatives = derivatives[:, :-1, :]
    right_derivatives = derivatives[:, 1:, :]
    coefficients = torch.stack(
        (
            left_values,
            left_derivatives,
            3.0 * (right_values - left_values) / h**2
            - (2.0 * left_derivatives + right_derivatives) / h,
            2.0 * (left_values - right_values) / h**3
            + (left_derivatives + right_derivatives) / h**2,
        ),
        dim=2,
    )
    objective = torch.dot(
        coefficients.flatten(),
        torch.as_tensor(coefficient_adjoints, dtype=torch.float64),
    )
    if factorized:
        objective = objective + torch.dot(
            weights[-1].flatten(),
            torch.as_tensor(final_projection_adjoints, dtype=torch.float64),
        )
    objective.backward()
    return [weight.grad.detach().numpy().flatten() for weight in weights]


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_compact_v2_defaults_to_fully_streamed_execution(
    streamed_model_paths,
    use_kokkos,
):
    standard_path, _ = streamed_model_paths
    calculator = Symmetrix(standard_path, use_kokkos=use_kokkos)

    assert calculator.streamed_edges == ("direct" if use_kokkos else "generic")
    assert calculator.evaluator.supports_streamed_edges
    assert calculator.evaluator.streamed_edges_mode == calculator.streamed_edges


@pytest.mark.parametrize(
    "use_kokkos,dtype,atol",
    [
        (False, "float64", 2e-11),
        (False, "float32", 2e-5),
        (True, "float64", 2e-11),
        (True, "float32", 2e-5),
    ],
)
def test_all_interactions_matches_materialized_and_releases_radial_storage(
    streamed_model_paths,
    use_kokkos,
    dtype,
    atol,
):
    standard_path, _ = streamed_model_paths
    outputs = {}
    storage = {}
    for mode in ("materialized", "all_interactions"):
        atoms = _small_structure()
        calculator = Symmetrix(
            standard_path,
            use_kokkos=use_kokkos,
            dtype=dtype,
            streamed_edges=mode,
        )
        assert calculator.evaluator.supports_streamed_edges
        assert calculator.evaluator.streamed_edges_mode == (
            "generic" if mode == "all_interactions" else mode
        )
        atoms.calc = calculator
        outputs[mode] = (
            atoms.get_potential_energy(),
            atoms.get_forces(),
            atoms.get_stress(),
        )
        storage[mode] = (
            calculator.evaluator.R0_storage_size,
            calculator.evaluator.R1_storage_size,
        )
        if mode == "materialized":
            materialized_evaluator = calculator.evaluator

    reference_energy, reference_forces, reference_stress = outputs["materialized"]
    energy, forces, stress = outputs["all_interactions"]
    assert energy == pytest.approx(reference_energy, rel=0.0, abs=atol)
    np.testing.assert_allclose(forces, reference_forces, rtol=0.0, atol=atol)
    np.testing.assert_allclose(stress, reference_stress, rtol=0.0, atol=atol)

    assert storage["materialized"][0] > 0
    assert storage["materialized"][1] > 0
    assert storage["all_interactions"] == (0, 0)

    materialized_evaluator.set_streamed_edges("all_interactions")
    assert materialized_evaluator.R0_storage_size == 0
    assert materialized_evaluator.R1_storage_size == 0


def test_field_coupled_models_expose_backend_appropriate_streamed_modes(
    streamed_model_paths,
):
    _, field_path = streamed_model_paths
    serial = native_symmetrix.MACE(str(field_path))
    assert serial.supports_streamed_edges
    for mode in ("all_interactions", "materialized"):
        serial.set_streamed_edges(mode)
        assert serial.streamed_edges_mode == (
            "generic" if mode == "all_interactions" else mode
        )
    with pytest.raises(
        ValueError, match="direct and receiver_factorized require the Kokkos evaluator"
    ):
        serial.set_streamed_edges("factorized")

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    kokkos = native_symmetrix.MACEKokkos(str(field_path))
    assert kokkos.supports_streamed_edges
    for mode in (
        "all_interactions",
        "factorized",
        "materialized",
    ):
        kokkos.set_streamed_edges(mode)
        assert kokkos.streamed_edges_mode == {
            "all_interactions": "generic",
            "factorized": "direct",
        }.get(mode, mode)


@pytest.mark.parametrize(
    "use_kokkos,dtype,first_order_atol,response_atol",
    [
        (False, "float64", 2e-9, 2e-6),
        (False, "float32", 5e-5, 5e-4),
        (True, "float64", 2e-9, 2e-6),
        (True, "float32", 5e-5, 5e-4),
    ],
    ids=[
        "native-float64",
        "native-float32",
        "kokkos-float64",
        "kokkos-float32",
    ],
)
def test_field_streamed_modes_match_materialized(
    streamed_model_paths,
    use_kokkos,
    dtype,
    first_order_atol,
    response_atol,
):
    _, field_path = streamed_model_paths
    electric_field = np.array([0.01, -0.02, 0.03])
    outputs = {}
    storage = {}
    for mode in ("materialized", "all_interactions"):
        atoms = _small_structure()
        calculator = Symmetrix(
            field_path,
            use_kokkos=use_kokkos,
            dtype=dtype,
            streamed_edges=mode,
            electric_field=electric_field,
        )
        assert calculator.evaluator.supports_streamed_edges
        atoms.calc = calculator
        calculator.calculate(
            atoms,
            properties=[
                "energy",
                "forces",
                "stress",
                "polarization",
                "polarizability",
                "becs",
            ],
        )
        results = calculator.results
        field_adjoint = np.asarray(calculator.evaluator.electric_field_adj).copy()
        field_hessian = np.asarray(calculator.evaluator.electric_field_hessian).copy()
        field_force_derivative = np.asarray(
            calculator.evaluator.electric_field_force_derivative
        ).copy()
        outputs[mode] = (
            results["energy"],
            results["forces"],
            results["stress"],
            results["polarization"],
            results["polarizability"],
            results["becs"],
            field_adjoint,
            field_hessian,
            field_force_derivative,
        )
        storage[mode] = (
            calculator.evaluator.R0_storage_size,
            calculator.evaluator.R1_storage_size,
        )

    (
        reference_energy,
        reference_forces,
        reference_stress,
        reference_polarization,
        reference_polarizability,
        reference_becs,
        reference_field_adj,
        reference_field_hessian,
        reference_field_force_derivative,
    ) = outputs["materialized"]
    (
        energy,
        forces,
        stress,
        polarization,
        polarizability,
        becs,
        field_adj,
        field_hessian,
        field_force_derivative,
    ) = outputs["all_interactions"]
    assert energy == pytest.approx(reference_energy, rel=0.0, abs=first_order_atol)
    np.testing.assert_allclose(
        forces, reference_forces, rtol=0.0, atol=first_order_atol
    )
    np.testing.assert_allclose(
        stress, reference_stress, rtol=0.0, atol=first_order_atol
    )
    np.testing.assert_allclose(
        polarization, reference_polarization, rtol=0.0, atol=first_order_atol
    )
    np.testing.assert_allclose(
        polarizability, reference_polarizability, rtol=0.0, atol=response_atol
    )
    np.testing.assert_allclose(becs, reference_becs, rtol=0.0, atol=response_atol)
    np.testing.assert_allclose(
        field_adj, reference_field_adj, rtol=0.0, atol=first_order_atol
    )
    np.testing.assert_allclose(
        field_hessian, reference_field_hessian, rtol=0.0, atol=response_atol
    )
    np.testing.assert_allclose(
        field_force_derivative,
        reference_field_force_derivative,
        rtol=0.0,
        atol=response_atol,
    )

    assert storage["materialized"][0] > 0
    assert storage["materialized"][1] > 0
    assert storage["all_interactions"] == (0, 0)


def test_native_float32_matches_native_float64(streamed_model_paths):
    standard_path, field_path = streamed_model_paths
    atoms = _small_structure()

    def evaluate(model_path, dtype, electric_field=None):
        calculator = Symmetrix(
            model_path,
            use_kokkos=False,
            dtype=dtype,
            streamed_edges="all_interactions",
            electric_field=electric_field,
        )
        assert isinstance(
            calculator.evaluator,
            native_symmetrix.MACEFloat if dtype == "float32" else native_symmetrix.MACE,
        )
        assert calculator.evaluator.scalar_size_bytes == (
            4 if dtype == "float32" else 8
        )
        calculator.calculate(
            atoms,
            properties=(
                ["energy", "forces"]
                if electric_field is None
                else ["energy", "forces", "polarization", "polarizability", "becs"]
            ),
        )
        return calculator.results

    standard64 = evaluate(standard_path, "float64")
    standard32 = evaluate(standard_path, "float32")
    assert standard32["energy"] == pytest.approx(
        standard64["energy"], rel=0.0, abs=2e-4
    )
    np.testing.assert_allclose(
        standard32["forces"], standard64["forces"], rtol=0.0, atol=2e-4
    )

    electric_field = np.array([0.01, -0.02, 0.03])
    field64 = evaluate(field_path, "float64", electric_field)
    field32 = evaluate(field_path, "float32", electric_field)
    assert field32["energy"] == pytest.approx(field64["energy"], rel=0.0, abs=2e-4)
    for property_name in ("forces", "polarization"):
        np.testing.assert_allclose(
            field32[property_name], field64[property_name], rtol=0.0, atol=2e-4
        )
    for property_name in ("polarizability", "becs"):
        np.testing.assert_allclose(
            field32[property_name], field64[property_name], rtol=0.0, atol=2e-3
        )


def test_kokkos_rejects_out_of_range_phi1_hidden_degree(
    streamed_model_paths,
    tmp_path,
):
    _, field_path = streamed_model_paths
    model = json.loads(field_path.read_text())
    model["Phi1_l2"][0] = model["L_max"] + 1
    invalid_path = tmp_path / "invalid-phi1-l2.json"
    invalid_path.write_text(json.dumps(model))

    with pytest.raises(RuntimeError, match="out-of-range hidden degree"):
        native_symmetrix.MACEKokkos(str(invalid_path))


def test_native_float32_rejects_out_of_range_model_values(
    streamed_model_paths,
    tmp_path,
):
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    model["H0_weights"][0] = 1e100
    invalid_path = tmp_path / "float32-overflow.json"
    invalid_path.write_text(json.dumps(model))

    with pytest.raises(ValueError, match="outside the requested precision range"):
        native_symmetrix.MACEFloat(str(invalid_path))


@pytest.mark.parametrize(
    "use_kokkos,dtype",
    [
        (False, "float64"),
        (False, "float32"),
        (True, "float64"),
        (True, "float32"),
    ],
)
def test_legacy_pair_spline_models_warn_and_default_to_legacy(
    legacy_standard_model_path,
    use_kokkos,
    dtype,
):
    with pytest.warns(UserWarning, match="format-v1.*streamed_edges='materialized'"):
        calculator = Symmetrix(
            legacy_standard_model_path,
            use_kokkos=use_kokkos,
            dtype=dtype,
        )

    evaluator = calculator.evaluator
    assert calculator.streamed_edges == "materialized"
    assert not evaluator.supports_streamed_edges
    assert evaluator.streamed_edges_mode == "materialized"
    with pytest.raises(ValueError, match="format-v2 compact MACE or MACEField"):
        evaluator.set_streamed_edges("all_interactions")


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_legacy_pair_spline_float32_matches_float64(
    legacy_standard_model_path,
    use_kokkos,
):
    outputs = {}
    for dtype in ("float64", "float32"):
        atoms = _small_structure()
        with pytest.warns(
            UserWarning, match="format-v1.*streamed_edges='materialized'"
        ):
            atoms.calc = Symmetrix(
                legacy_standard_model_path,
                use_kokkos=use_kokkos,
                dtype=dtype,
            )
        assert atoms.calc.evaluator.scalar_size_bytes == (
            8 if dtype == "float64" else 4
        )
        outputs[dtype] = (atoms.get_potential_energy(), atoms.get_forces())

    assert outputs["float32"][0] == pytest.approx(
        outputs["float64"][0], rel=0.0, abs=2e-4
    )
    np.testing.assert_allclose(
        outputs["float32"][1], outputs["float64"][1], rtol=0.0, atol=2e-4
    )


def test_kokkos_streamed_path_offsets_match_model_layout(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    calculator = Symmetrix(
        standard_path, use_kokkos=True, streamed_edges="all_interactions"
    )
    model = json.loads(standard_path.read_text())

    expected = [0]
    for l1, l2 in zip(model["Phi1_l1"], model["Phi1_l2"]):
        expected.append(expected[-1] + (2 * l1 + 1) * (2 * l2 + 1))

    assert calculator.evaluator.Phi1_path_row_offsets == expected
    assert native_symmetrix._kokkos_default_execution_space()


@pytest.mark.parametrize("mode", ["all_interactions", "factorized"])
def test_kokkos_standalone_readout_returns_total_energy(
    streamed_model_paths,
    mode,
):
    standard_path, _ = streamed_model_paths
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges=mode,
    )
    inputs = calculator._mace_inputs(_small_structure())
    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, _ = inputs
    evaluator = calculator.evaluator
    evaluator.compute_node_energies_forces(
        num_nodes,
        node_types,
        num_neigh,
        j_list,
        neigh_types,
        np.asarray(xyz).reshape(-1),
        r,
    )
    evaluator.node_energies = np.zeros(num_nodes, dtype=np.float64)

    total_energy = evaluator.compute_readouts(num_nodes, node_types)
    node_energy_sum = np.sum(np.asarray(evaluator.node_energies))

    assert total_energy == pytest.approx(node_energy_sum, rel=0.0, abs=1e-5)


def test_kokkos_finalize_rejects_live_mace_evaluator(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    gc.collect()
    baseline = native_symmetrix._kokkos_live_object_count()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="all_interactions",
    )
    evaluator = calculator.evaluator
    assert native_symmetrix._kokkos_live_object_count() == baseline + 1

    with pytest.raises(RuntimeError, match="Cannot finalize Kokkos.*alive"):
        native_symmetrix._finalize_kokkos()
    assert native_symmetrix._kokkos_is_initialized()

    del evaluator, calculator
    gc.collect()
    assert native_symmetrix._kokkos_live_object_count() == baseline


@pytest.mark.parametrize("network", ["R0", "R1"])
@pytest.mark.parametrize("type_i,type_j", [(0, 0), (0, 1), (1, 1)])
def test_compact_radial_factorization_reconstructs_projected_tables(
    streamed_model_paths,
    network,
    type_i,
    type_j,
):
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACEKokkos(str(standard_path))
    evaluator.set_streamed_edges("factorized")
    report = evaluator._compact_radial_factorization_report(network, type_i, type_j)
    shape = model["compact_radial"]["networks"][network]["shape"]

    assert report["embedding_width"] == shape[-2]
    assert report["output_width"] == shape[-1]
    assert report["projection_layout"] == "row-major-output-by-embedding"
    assert report["max_value_error"] < 2e-13
    assert report["max_derivative_error"] < 2e-12


@pytest.mark.parametrize(
    "network,factorized",
    [("R0", False), ("R1", True), ("A0", False), ("A1", False)],
)
def test_compact_radial_backpropagation_matches_finite_difference(
    streamed_model_paths,
    tmp_path,
    network,
    factorized,
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    network_definition = model["compact_radial"]["networks"][network]
    pair_types = np.asarray([[0, 0], [0, 1]], dtype=np.int32)
    function_count = network_definition["shape"][-2 if factorized else -1]
    coefficient_count = (
        len(pair_types)
        * (model["compact_radial"]["num_spline_points"] - 1)
        * 4
        * function_count
    )
    rng = np.random.default_rng(1307 + sum(map(ord, network)))
    coefficient_adjoints = rng.normal(size=coefficient_count)
    coefficient_adjoints /= np.linalg.norm(coefficient_adjoints)
    final_projection_adjoints = (
        rng.normal(size=len(network_definition["weights"][-1]))
        if factorized
        else np.empty(0, dtype=np.float64)
    )
    if factorized:
        final_projection_adjoints /= np.linalg.norm(final_projection_adjoints)

    def report_and_objective(candidate):
        candidate_path = tmp_path / f"{network}-finite-difference.json"
        candidate_path.write_text(json.dumps(candidate, separators=(",", ":")))
        evaluator = native_symmetrix.MACEKokkos(str(candidate_path))
        report = evaluator._compact_radial_backpropagation_report(
            network,
            pair_types,
            coefficient_adjoints,
            factorized,
            final_projection_adjoints,
        )
        objective = np.dot(report["coefficient_values"], coefficient_adjoints)
        if factorized:
            objective += np.dot(
                candidate["compact_radial"]["networks"][network]["weights"][-1],
                final_projection_adjoints,
            )
        return report, objective

    report, _ = report_and_objective(model)
    assert report["coefficient_layout"] == "pair-interval-order-function"
    assert report["nodal_layout"] == "pair-node-function"
    assert report["function_count"] == function_count
    assert np.dot(report["coefficient_values"], coefficient_adjoints) == pytest.approx(
        np.dot(report["nodal_values"], report["nodal_adjoints"]),
        rel=2e-11,
        abs=2e-11,
    )

    torch_gradients = _torch_compact_radial_weight_gradients(
        model,
        network,
        pair_types,
        coefficient_adjoints,
        factorized,
        final_projection_adjoints,
    )
    for layer, (analytical_layer, torch_layer) in enumerate(
        zip(report["weight_gradients"], torch_gradients)
    ):
        np.testing.assert_allclose(
            analytical_layer,
            torch_layer,
            rtol=2e-9,
            atol=2e-9,
            err_msg=f"{network} layer {layer} Torch autograd oracle",
        )

    for layer, analytical_layer in enumerate(report["weight_gradients"]):
        analytical_layer = np.asarray(analytical_layer)
        index = int(np.argmax(np.abs(analytical_layer)))
        analytical = analytical_layer[index]
        weight = network_definition["weights"][layer][index]
        epsilon = 2e-6 * max(1.0, abs(weight))
        objectives = []
        for direction in (-1.0, 1.0):
            candidate = copy.deepcopy(model)
            candidate["compact_radial"]["networks"][network]["weights"][layer][
                index
            ] += direction * epsilon
            _, objective = report_and_objective(candidate)
            objectives.append(objective)
        finite_difference = (objectives[1] - objectives[0]) / (2.0 * epsilon)
        assert finite_difference == pytest.approx(
            analytical,
            rel=2e-5,
            abs=2e-7,
        ), f"{network} layer {layer} weight {index}"


def test_execution_parameter_gradient_boundary_is_opt_in_and_model_sized(
    streamed_model_paths,
):
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACEKokkos(str(standard_path))
    evaluator.set_streamed_edges("receiver_factorized")

    assert not evaluator.execution_parameter_gradients_enabled
    assert evaluator.factorized_derivative_signature == "factorized_coordinate"
    assert not evaluator.execution_parameter_gradients_ready
    assert evaluator.execution_parameter_gradients_result_bytes == 0
    assert evaluator.execution_parameter_gradients_workspace_bytes == 0
    assert evaluator.execution_parameter_gradients()["groups"] == []
    with pytest.raises(ValueError, match="configured limit"):
        evaluator.set_execution_parameter_gradients(True, max_bytes=1)
    assert not evaluator.execution_parameter_gradients_enabled
    assert evaluator.execution_parameter_gradients_result_bytes == 0

    evaluator.set_execution_parameter_gradients(True)
    assert evaluator.factorized_derivative_signature == "parameter_coordinate"
    result = evaluator.execution_parameter_gradients()
    assert result["schema"] == "symmetrix.execution.parameter-gradients"
    assert result["version"] == 1
    assert result["enabled"]
    assert not result["ready"]
    assert result["owner"] == "parameter-element,receiver-order"
    assert result["dtype"] == "float64"
    assert result["parameterization"] == "extracted-compact-v2"
    assert result["coverage"] == "two-interaction-execution-replay"
    assert result["source_checkpoint_mapping"] == (
        "compact-radial-exact;H0-A0-runtime-fused"
    )
    assert result["phase_times_ms"] == {"r1": 0.0, "r0": 0.0, "density": 0.0}
    assert result["worker_counts"] == {"r1": 0, "r0": 0}

    expected_shapes = {}
    networks = model["compact_radial"]["networks"]
    for network in ("R0", "R1", "A0", "A1"):
        for layer, (input_width, output_width) in enumerate(
            zip(networks[network]["shape"][:-1], networks[network]["shape"][1:])
        ):
            expected_shapes[f"compact_radial.{network}.weights.{layer}"] = (
                output_width,
                input_width,
            )
    expected_shapes["H0_weights"] = (model["num_elements"], model["num_channels"])
    for l, matrix in enumerate(model["A0_weights"][0]):
        expected_shapes[f"A0_weights.l{l}"] = (
            model["num_elements"],
            len(matrix) // model["num_channels"],
            model["num_channels"],
        )
    for l, matrix in enumerate(model["A1_weights"]):
        expected_shapes[f"A1_weights.l{l}"] = (
            len(matrix) // model["num_channels"],
            model["num_channels"],
        )

    groups = {group["name"]: group for group in result["groups"]}
    assert set(groups) == set(expected_shapes)
    payload_bytes = 0
    for name, expected_shape in expected_shapes.items():
        group = groups[name]
        assert tuple(group["shape"]) == expected_shape
        assert group["values"].shape == (np.prod(expected_shape),)
        assert np.all(group["values"] == 0.0)
        payload_bytes += group["values"].nbytes
    assert result["result_bytes"] == payload_bytes
    assert result["workspace_bytes"] == payload_bytes
    assert result["result_bytes"] < result["max_bytes"]

    evaluator.set_execution_parameter_gradients(False)
    assert not evaluator.execution_parameter_gradients_enabled
    assert evaluator.execution_parameter_gradients_result_bytes == 0
    assert evaluator.execution_parameter_gradients_workspace_bytes == 0
    assert evaluator.execution_parameter_gradients()["groups"] == []


def test_execution_interaction_parameter_replay_populates_equation_36_groups(
    streamed_model_paths,
):
    _require_receiver_factorized_host_backend()
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="receiver_factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])
    evaluator.set_execution_parameter_gradients(True)
    evaluator.compute_node_energies_forces(*native_args)

    result = evaluator.execution_parameter_gradients()
    assert result["enabled"]
    assert result["ready"]
    assert result["workspace_bytes"] >= result["result_bytes"]
    groups = {group["name"]: np.asarray(group["values"]) for group in result["groups"]}
    for name, values in groups.items():
        assert np.all(np.isfinite(values)), name
        assert np.any(values != 0.0), name

    reference = {name: values.copy() for name, values in groups.items()}
    evaluator.compute_node_energies_forces(*native_args)
    repeated = {
        group["name"]: np.asarray(group["values"])
        for group in evaluator.execution_parameter_gradients()["groups"]
    }
    for name in reference:
        np.testing.assert_array_equal(repeated[name], reference[name])

    result_bytes = evaluator.execution_parameter_gradients_result_bytes
    evaluator.set_execution_parameter_gradients(True, max_bytes=result_bytes)
    with pytest.raises(ValueError, match="host bytes.*exceeding"):
        evaluator.compute_node_energies_forces(*native_args)
    assert not evaluator.execution_parameter_gradients_ready


def test_execution_field_parameter_replay_completes_interaction_groups(
    streamed_model_paths,
):
    _require_receiver_factorized_host_backend()
    _, field_path = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        field_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="receiver_factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])
    evaluator.set_execution_parameter_gradients(True)
    evaluator.compute_node_energies_forces_field(
        *native_args,
        np.asarray([0.01, -0.02, 0.03], dtype=np.float64),
    )

    result = evaluator.execution_parameter_gradients()
    assert result["ready"]
    assert result["workspace_bytes"] >= result["result_bytes"]
    for group in result["groups"]:
        values = np.asarray(group["values"])
        assert np.all(np.isfinite(values)), group["name"]
        assert np.any(values != 0.0), group["name"]
    evaluator.compute_electric_field_hessian(
        *native_args,
        np.asarray([0.01, -0.02, 0.03], dtype=np.float64),
    )
    assert np.all(np.isfinite(evaluator.electric_field_hessian))
    assert evaluator.execution_parameter_gradients_ready


@pytest.mark.parametrize("edge_count", [0, 1])
def test_execution_parameter_replay_handles_zero_and_one_directed_edge(
    streamed_model_paths,
    edge_count,
):
    standard_path, _ = streamed_model_paths
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACEKokkos(str(standard_path))
    evaluator.set_streamed_edges("receiver_factorized")
    evaluator.set_execution_parameter_gradients(True)
    if edge_count == 0:
        native_args = (
            1,
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    else:
        native_args = (
            2,
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([1, 0], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            np.asarray([1.7, 0.2, -0.1], dtype=np.float64),
            np.asarray([np.sqrt(1.7**2 + 0.2**2 + 0.1**2)], dtype=np.float64),
        )
    evaluator.compute_node_energies_forces(*native_args)
    result = evaluator.execution_parameter_gradients()
    assert result["ready"]
    total_magnitude = sum(np.sum(np.abs(group["values"])) for group in result["groups"])
    if edge_count == 0:
        assert total_magnitude == 0.0
    else:
        assert total_magnitude > 0.0


def test_execution_parameter_replay_supports_prepared_graph_tokens(
    streamed_model_paths,
):
    _require_receiver_factorized_host_backend()
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="receiver_factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    evaluator.set_execution_parameter_gradients(True)
    token = evaluator._prepare_factorized_graph(*inputs[:5])
    evaluator._compute_prepared_factorized(
        token, np.asarray(inputs[5]).flatten(), inputs[6]
    )
    assert evaluator.execution_parameter_gradients_ready
    first_result = evaluator.execution_parameter_gradients()
    assert 1 <= first_result["worker_counts"]["r1"] <= len(atoms)
    assert 1 <= first_result["worker_counts"]["r0"] <= len(atoms)
    assert set(first_result["phase_times_ms"]) == {"r1", "r0", "density"}
    reference = {
        group["name"]: np.asarray(group["values"]).copy()
        for group in first_result["groups"]
    }
    evaluator._compute_prepared_factorized(
        token, np.asarray(inputs[5]).flatten(), inputs[6]
    )
    repeated = {
        group["name"]: np.asarray(group["values"])
        for group in evaluator.execution_parameter_gradients()["groups"]
    }
    for name in reference:
        np.testing.assert_array_equal(repeated[name], reference[name])

    constrained_max_bytes = first_result["workspace_bytes"] - 1
    evaluator.set_execution_parameter_gradients(True, max_bytes=constrained_max_bytes)
    evaluator._compute_prepared_factorized(
        token, np.asarray(inputs[5]).flatten(), inputs[6]
    )
    constrained = evaluator.execution_parameter_gradients()
    assert constrained["workspace_bytes"] <= constrained_max_bytes
    assert all(
        1 <= constrained["worker_counts"][phase] <= len(atoms) for phase in ("r1", "r0")
    )
    constrained_groups = {
        group["name"]: np.asarray(group["values"]) for group in constrained["groups"]
    }
    for name in reference:
        np.testing.assert_allclose(
            constrained_groups[name], reference[name], rtol=2e-5, atol=2e-7
        )


def test_execution_parameter_replay_ignores_skin_only_edges(
    streamed_model_paths,
):
    _require_receiver_factorized_host_backend()
    standard_path, _ = streamed_model_paths
    candidate = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="receiver_factorized",
        neighbor_skin=0.5,
    )
    exact = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="receiver_factorized",
        neighbor_skin=0.0,
    )
    atoms = Atoms(
        ["Al", "N", "Al"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.2, 0.1],
            [candidate.cutoff + 0.1, 0.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    candidate_edges = len(candidate._mace_inputs(atoms)[3])
    exact_edges = len(exact._mace_inputs(atoms)[3])
    assert candidate_edges == exact_edges + 2

    gradients = {}
    for name, calculator in (("candidate", candidate), ("exact", exact)):
        calculator.evaluator.set_execution_parameter_gradients(True)
        calculator.calculate(atoms, properties=["energy", "forces", "stress"])
        result = calculator.evaluator.execution_parameter_gradients()
        assert result["ready"]
        gradients[name] = {
            group["name"]: np.asarray(group["values"]).copy()
            for group in result["groups"]
        }

    assert gradients["candidate"].keys() == gradients["exact"].keys()
    for name, reference in gradients["exact"].items():
        np.testing.assert_allclose(
            gradients["candidate"][name],
            reference,
            rtol=5e-4,
            atol=5e-5,
            err_msg=name,
        )


def test_execution_native_parameter_replay_matches_tiled_float32(
    streamed_model_paths,
):
    _require_receiver_factorized_host_backend()
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="receiver_factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    evaluator.prepare_active_types(inputs[1])
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])
    evaluator.set_execution_parameter_gradients(True)
    gradients = {}
    for strategy in ("tiled_coupling", "jit_plugin"):
        evaluator._set_factorized_source_strategy(strategy)
        evaluator.compute_node_energies_forces(*native_args)
        assert evaluator.execution_parameter_gradients_ready
        gradients[strategy] = {
            group["name"]: np.asarray(group["values"]).copy()
            for group in evaluator.execution_parameter_gradients()["groups"]
        }
    for name, reference in gradients["tiled_coupling"].items():
        np.testing.assert_allclose(
            gradients["jit_plugin"][name],
            reference,
            rtol=5e-4,
            atol=5e-5,
            err_msg=name,
        )


def test_execution_interaction_parameter_replay_matches_energy_finite_differences(
    streamed_model_paths,
    tmp_path,
):
    _require_receiver_factorized_host_backend()
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    atoms = _small_structure()

    def evaluate(candidate, gradients=False):
        path = tmp_path / "r1-parameter-finite-difference.json"
        path.write_text(json.dumps(candidate, separators=(",", ":")))
        calculator = Symmetrix(
            path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="receiver_factorized",
        )
        inputs = calculator._mace_inputs(atoms)
        native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])
        if gradients:
            calculator.evaluator.set_execution_parameter_gradients(True)
        calculator.evaluator.compute_node_energies_forces(*native_args)
        energy = float(np.sum(calculator.evaluator.node_energies))
        result = (
            calculator.evaluator.execution_parameter_gradients() if gradients else None
        )
        return energy, result

    _, result = evaluate(model, gradients=True)
    groups = {group["name"]: np.asarray(group["values"]) for group in result["groups"]}
    last_layer = len(model["compact_radial"]["networks"]["R1"]["weights"]) - 1
    candidates = [
        ("H0_weights", ("H0_weights",)),
        ("A0_weights.l0", ("A0_weights", 0)),
        ("compact_radial.R0.weights.0", ("compact_radial", "R0", 0)),
        ("compact_radial.A0.weights.0", ("compact_radial", "A0", 0)),
        ("A1_weights.l0", ("A1_weights", 0)),
        ("compact_radial.R1.weights.0", ("compact_radial", "R1", 0)),
        ("compact_radial.A1.weights.0", ("compact_radial", "A1", 0)),
        (
            f"compact_radial.R1.weights.{last_layer}",
            ("compact_radial", "R1", last_layer),
        ),
    ]
    for group_name, location in candidates:
        analytical_values = groups[group_name]
        index = int(np.argmax(np.abs(analytical_values)))
        analytical = analytical_values[index]
        if location[0] == "H0_weights":
            weights = model["H0_weights"]
            weight_index = index
        elif location[0] == "A0_weights":
            matrix_size = len(model["A0_weights"][0][location[1]])
            receiver_type, weight_index = divmod(index, matrix_size)
            weights = model["A0_weights"][receiver_type][location[1]]
        elif location[0] == "A1_weights":
            weights = model["A1_weights"][location[1]]
            weight_index = index
        else:
            weights = model["compact_radial"]["networks"][location[1]]["weights"][
                location[2]
            ]
            weight_index = index
        epsilon = 2e-3 * max(1.0, abs(weights[weight_index]))
        energies = []
        for direction in (-1.0, 1.0):
            candidate = copy.deepcopy(model)
            if location[0] == "H0_weights":
                candidate_weights = candidate["H0_weights"]
            elif location[0] == "A0_weights":
                candidate_weights = candidate["A0_weights"][receiver_type][location[1]]
            elif location[0] == "A1_weights":
                candidate_weights = candidate["A1_weights"][location[1]]
            else:
                candidate_weights = candidate["compact_radial"]["networks"][
                    location[1]
                ]["weights"][location[2]]
            candidate_weights[weight_index] += direction * epsilon
            energy, _ = evaluate(candidate)
            energies.append(energy)
        finite_difference = (energies[1] - energies[0]) / (2.0 * epsilon)
        assert finite_difference == pytest.approx(
            analytical,
            rel=5e-3,
            abs=2e-3,
        ), f"{group_name} weight {index}"


def test_calculator_rejects_unknown_streamed_mode(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    with pytest.raises(
        ValueError,
        match="materialized.*all_interactions.*factorized",
    ):
        Symmetrix(standard_path, streamed_edges="tiles")


@pytest.mark.parametrize(
    "mode", ["legacy", "r1", "all", "second_interaction", "receiver_factorized"]
)
def test_calculator_rejects_removed_streamed_modes(streamed_model_paths, mode):
    standard_path, _ = streamed_model_paths
    with pytest.raises(ValueError, match="streamed_edges must be one of"):
        Symmetrix(standard_path, streamed_edges=mode)


def test_native_evaluators_reject_removed_second_interaction(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    message = "materialized.*all_interactions.*factorized"

    serial = native_symmetrix.MACE(str(standard_path))
    with pytest.raises(ValueError, match=message):
        serial.set_streamed_edges("second_interaction")

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    kokkos = native_symmetrix.MACEKokkos(str(standard_path))
    with pytest.raises(ValueError, match=message):
        kokkos.set_streamed_edges("second_interaction")


def test_serial_evaluator_rejects_factorized(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    with pytest.raises(
        ValueError, match="direct and receiver_factorized require the Kokkos evaluator"
    ):
        Symmetrix(standard_path, use_kokkos=False, streamed_edges="factorized")


@pytest.mark.parametrize("dtype,atol", [("float64", 3e-11), ("float32", 3e-5)])
def test_factorized_matches_materialized_with_bounded_workspace(
    streamed_model_paths,
    dtype,
    atol,
):
    standard_path, _ = streamed_model_paths
    outputs = {}
    evaluators = {}
    for mode in ("materialized", "factorized"):
        atoms = _small_structure()
        calculator = Symmetrix(
            standard_path,
            use_kokkos=True,
            dtype=dtype,
            streamed_edges=mode,
        )
        if mode == "factorized":
            calculator.evaluator._set_factorized_source_strategy("team_cached")
        atoms.calc = calculator
        outputs[mode] = (atoms.get_potential_energy(), atoms.get_forces())
        evaluators[mode] = calculator.evaluator

    reference_energy, reference_forces = outputs["materialized"]
    energy, forces = outputs["factorized"]
    assert energy == pytest.approx(reference_energy, rel=0.0, abs=atol)
    np.testing.assert_allclose(forces, reference_forces, rtol=0.0, atol=atol)

    evaluator = evaluators["factorized"]
    assert evaluator.supports_factorized
    assert evaluator.factorized_ready
    assert evaluator.R1_storage_size == 0
    assert 0 < evaluator.factorized_workspace_bytes <= 64 * 1024 * 1024
    assert evaluator.factorized_workspace_bytes <= (
        evaluator.factorized_workspace_capacity_bytes
    )
    assert evaluator.factorized_source_owned_reverse
    assert evaluator.factorized_schedule_entries > 0
    assert evaluator.factorized_source_strategy == "team_cached"
    assert evaluator.factorized_radial_workspace_bytes > 0
    assert evaluator.factorized_chunk_size == (24 if dtype == "float32" else 16)


@pytest.mark.parametrize("dtype,atol", [("float64", 3e-11), ("float32", 3e-5)])
def test_factorized_compact_mace_reduces_stress_on_device(
    streamed_model_paths,
    dtype,
    atol,
):
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    atoms.set_pbc(True)
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("team_cached")
    inputs = calculator._mace_inputs(atoms)
    token = evaluator._prepare_factorized_graph(*inputs[:5])
    xyz = np.asarray(inputs[5], dtype=float)
    evaluator._compute_prepared_factorized(token, xyz.reshape(-1), inputs[6])

    directed_forces = np.asarray(evaluator.node_forces, dtype=float).reshape(-1, 3)
    expected = full_3x3_to_voigt_6_stress(
        (-directed_forces[: len(xyz)].T @ xyz) / atoms.get_volume()
    )
    reduced = full_3x3_to_voigt_6_stress(
        np.asarray(
            evaluator._reduce_stress(atoms.get_volume(), np.empty(0), token),
            dtype=float,
        ).reshape(3, 3)
    )

    np.testing.assert_allclose(reduced, expected, rtol=0.0, atol=atol)


def test_factorized_compact_edge_geometry_reduces_stress_on_device(
    streamed_model_paths,
):
    _require_execution_fixed_weight_backend()
    standard_path, _ = streamed_model_paths

    def make_calculator(edge_geometry_policy):
        calculator = Symmetrix(
            standard_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="factorized",
            m1_polynomial_policy="recompute",
        )
        evaluator = calculator.evaluator
        evaluator._set_factorized_source_strategy("jit_plugin")
        evaluator._set_factorized_direct_forward_executor("jit_all")
        evaluator._set_factorized_direct_reverse_executor("jit")
        evaluator._set_standard_r0_executor("v2_edge16")
        evaluator._set_standard_m0_executor("standard")
        evaluator._set_mh0_state_policy("reuse-adjoints-v1")
        evaluator._set_edge_geometry_policy(edge_geometry_policy)
        return calculator

    cartesian = make_calculator("cartesian-f64-v1")
    compact = make_calculator("unit-f32-radius-f64-v1")
    direct_stress = {}
    ase_stress = {}
    for name, calculator in (("cartesian", cartesian), ("compact", compact)):
        atoms = _small_structure()
        atoms.set_pbc(True)
        inputs = calculator._mace_inputs(atoms)
        evaluator = calculator.evaluator
        token = evaluator._prepare_factorized_graph(*inputs[:5])
        xyz = np.asarray(inputs[5], dtype=float)
        evaluator._compute_prepared_factorized(token, xyz.reshape(-1), inputs[6])
        direct_stress[name] = full_3x3_to_voigt_6_stress(
            np.asarray(
                evaluator._reduce_stress(
                    atoms.get_volume(), np.empty(0, dtype=float), token
                ),
                dtype=float,
            ).reshape(3, 3)
        )
        directed_forces = np.asarray(evaluator.node_forces, dtype=float).reshape(-1, 3)
        expected = full_3x3_to_voigt_6_stress(
            (-directed_forces[: len(xyz)].T @ xyz) / atoms.get_volume()
        )
        np.testing.assert_allclose(direct_stress[name], expected, rtol=0.0, atol=3e-5)
        atoms.calc = calculator
        ase_stress[name] = atoms.get_stress()

    assert compact.evaluator.edge_geometry_policy == "unit-f32-radius-f64-v1"
    assert compact.evaluator.compact_edge_geometry_bytes > 0
    np.testing.assert_allclose(
        direct_stress["compact"], direct_stress["cartesian"], rtol=0.0, atol=3e-5
    )
    np.testing.assert_allclose(
        ase_stress["compact"], ase_stress["cartesian"], rtol=0.0, atol=3e-5
    )


@pytest.mark.parametrize("dtype,atol", [("float64", 3e-11), ("float32", 3e-5)])
def test_factorized_source_strategies_match_serial_reference(
    streamed_model_paths,
    dtype,
    atol,
):
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    strategies = (
        "serial_reference",
        "team_cached",
        "tiled_coupling",
        "jit_plugin",
    )
    results = {}
    for strategy in strategies:
        evaluator._set_factorized_source_strategy(strategy)
        evaluator.compute_node_energies_forces(*native_args)
        results[strategy] = (
            calculator._collect_mace_results(atoms, inputs),
            np.asarray(evaluator.H1_adj),
        )

    reference, reference_h1_adj = results["serial_reference"]
    for strategy in strategies[1:]:
        candidate, candidate_h1_adj = results[strategy]
        assert candidate["energy"] == pytest.approx(
            reference["energy"], rel=0.0, abs=atol
        )
        np.testing.assert_allclose(
            candidate["forces"], reference["forces"], rtol=0.0, atol=atol
        )
        np.testing.assert_allclose(
            candidate_h1_adj, reference_h1_adj, rtol=0.0, atol=atol
        )
    assert evaluator.factorized_source_strategy == "jit_plugin"
    jit_forward_ready = (
        evaluator.factorized_selected_direct_forward_executor == "jit_all"
    )
    jit_reverse_ready = evaluator.factorized_selected_direct_reverse_executor == "jit"
    assert evaluator.factorized_execution_strategy == (
        "direct_jit_reverse" if jit_reverse_ready else "tiled_coupling"
    )
    assert evaluator.factorized_tiled_ready
    assert evaluator.factorized_coupling_capacity_bytes > 0
    assert evaluator.factorized_execution_profile == (
        "direct_fixed_weight" if jit_reverse_ready else "factorized_state"
    )
    assert evaluator.factorized_derivative_signature == (
        "fixed_weight_coordinate" if jit_reverse_ready else "factorized_coordinate"
    )
    if jit_reverse_ready:
        assert evaluator.factorized_selected_reverse_cache_policy == "not_applicable"
        assert evaluator.factorized_retained_reverse_groups == []
        assert evaluator.factorized_recomputed_reverse_groups == []
        assert evaluator.factorized_planned_coupling_workspace_bytes == 0
    assert evaluator.factorized_coupling_workspace_bytes == (
        0
        if evaluator.factorized_execution_profile == "direct_fixed_weight"
        else evaluator.factorized_coupling_capacity_bytes
    )
    if evaluator.factorized_execution_profile == "direct_fixed_weight":
        assert evaluator.factorized_compact_workspace_bytes == 0
        assert evaluator.factorized_arena_workspace_bytes == 0
        assert evaluator.factorized_radial_workspace_bytes == 0
    else:
        assert evaluator.factorized_compact_workspace_bytes > 0
    assert evaluator.factorized_jit_ready
    assert evaluator.factorized_jit_artifact_id.startswith("jit-r1-gen11-")
    assert evaluator.factorized_jit_contract_fingerprint.startswith("sha256:")
    assert evaluator.factorized_forward_coupling_workspace_bytes == (
        0 if jit_forward_ready else evaluator.factorized_coupling_workspace_bytes
    )
    assert evaluator.factorized_reverse_coupling_workspace_bytes == (
        0
        if evaluator.factorized_execution_profile == "direct_fixed_weight"
        else evaluator.factorized_coupling_workspace_bytes
    )
    assert evaluator.factorized_has_model_contract
    assert evaluator.factorized_model_contract_fingerprint.startswith("sha256:")
    assert evaluator.factorized_model_semantic_fingerprint.startswith("sha256:")
    if jit_reverse_ready:
        assert evaluator.factorized_selected_direct_reverse_executor == "jit"
    if jit_forward_ready:
        assert evaluator.factorized_selected_direct_forward_executor == ("jit_all")

    with pytest.raises(
        ValueError,
        match="serial_reference.*team_cached.*tiled_coupling.*jit_plugin",
    ):
        evaluator._set_factorized_source_strategy("unknown")


def test_jit_reverse_and_standard_r0_v2_match_forced_fallbacks(
    streamed_model_paths,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_forward_executor("jit_all")
    evaluator._set_factorized_direct_reverse_executor("jit")
    evaluator._set_standard_r0_executor("v2_receiver")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    def evaluate():
        evaluator.compute_node_energies_forces(*native_args)
        results = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": results["energy"],
            "forces": np.array(results["forces"], copy=True),
            "Phi1": np.array(evaluator.Phi1, copy=True),
            "A1": np.array(evaluator.A1, copy=True),
            "H1_adj": np.array(evaluator.H1_adj, copy=True),
            "directed_forces": np.array(evaluator.node_forces, copy=True),
        }

    generated_launches = (
        evaluator.factorized_jit_forward_launch_count,
        evaluator.factorized_jit_reverse_launch_count,
        evaluator.standard_r0_module_launch_count,
    )
    generated = evaluate()
    assert evaluator.factorized_selected_direct_forward_executor == "jit_all"
    assert evaluator.factorized_selected_direct_reverse_executor == "jit"
    assert evaluator.standard_r0_selected_executor == "v2_receiver"
    assert evaluator.factorized_jit_forward_launch_count - generated_launches[0] == 1
    expected_reverse_launches = 1 if _kokkos_execution_space() == "Cuda" else 2
    assert (
        evaluator.factorized_jit_reverse_launch_count - generated_launches[1]
        == expected_reverse_launches
    )
    assert evaluator.standard_r0_module_launch_count - generated_launches[2] == 4
    assert evaluator.execution_phi1r_active_bytes == 0
    assert evaluator.execution_phi1r_capacity_bytes == 0
    assert evaluator.execution_dphi1r_active_bytes == 0

    evaluator._set_factorized_direct_forward_executor("runtime")
    runtime = evaluate()
    assert evaluator.factorized_selected_direct_forward_executor == "runtime"
    assert evaluator.execution_phi1r_active_bytes > 0
    assert evaluator.execution_phi1r_capacity_bytes > 0
    assert runtime["energy"] == pytest.approx(generated["energy"], rel=0.0, abs=3e-5)
    for name in ("Phi1", "A1", "forces", "H1_adj", "directed_forces"):
        np.testing.assert_allclose(runtime[name], generated[name], rtol=0.0, atol=3e-5)

    evaluator._set_factorized_direct_forward_executor("jit_all")
    regenerated = evaluate()
    assert evaluator.execution_phi1r_active_bytes == 0
    assert evaluator.execution_phi1r_capacity_bytes == 0
    assert regenerated["energy"] == generated["energy"]
    for name in ("Phi1", "A1", "forces", "H1_adj", "directed_forces"):
        np.testing.assert_allclose(
            regenerated[name], generated[name], rtol=0.0, atol=3e-5
        )

    for executor in ("v2_edge16", "v2_edge32"):
        evaluator._set_standard_r0_executor(executor)
        candidate_launches = evaluator.standard_r0_module_launch_count
        candidate = evaluate()
        assert evaluator.standard_r0_selected_executor == executor
        assert evaluator.standard_r0_module_launch_count - candidate_launches == 4
        assert evaluator.execution_dphi1r_active_bytes == 0
        assert candidate["energy"] == generated["energy"]
        for name in ("forces", "H1_adj", "directed_forces"):
            np.testing.assert_allclose(
                candidate[name], generated[name], rtol=0.0, atol=3e-5
            )
        repeated_candidate = evaluate()
        assert repeated_candidate["energy"] == candidate["energy"]
        for name in ("forces", "H1_adj", "directed_forces"):
            np.testing.assert_allclose(
                repeated_candidate[name], candidate[name], rtol=0.0, atol=3e-5
            )

    legacy_r0_launch_count = 0
    generated_reverse_launch_count = 1 if _kokkos_execution_space() == "Cuda" else 2
    controls = (
        (
            "jit",
            "v1",
            generated_reverse_launch_count,
            legacy_r0_launch_count,
            False,
        ),
        ("runtime", "v2_receiver", 0, 4, True),
        ("runtime", "v1", 0, legacy_r0_launch_count, True),
    )
    for direct_executor, r0_executor, r1_count, r0_count, uses_dphi1r in controls:
        evaluator._set_factorized_direct_reverse_executor(direct_executor)
        evaluator._set_standard_r0_executor(r0_executor)
        fallback_launches = (
            evaluator.factorized_jit_forward_launch_count,
            evaluator.factorized_jit_reverse_launch_count,
            evaluator.standard_r0_module_launch_count,
        )
        fallback = evaluate()
        assert evaluator.factorized_selected_direct_reverse_executor == direct_executor
        assert evaluator.standard_r0_selected_executor == r0_executor
        assert evaluator.factorized_jit_forward_launch_count - fallback_launches[0] == 1
        assert (
            evaluator.factorized_jit_reverse_launch_count - fallback_launches[1]
            == r1_count
        )
        assert (
            evaluator.standard_r0_module_launch_count - fallback_launches[2] == r0_count
        )
        assert bool(evaluator.execution_dphi1r_active_bytes) is uses_dphi1r
        if uses_dphi1r:
            assert evaluator.execution_dphi1r_capacity_bytes > 0

        assert generated["energy"] == pytest.approx(
            fallback["energy"], rel=0.0, abs=3e-5
        )
        for name in ("forces", "H1_adj", "directed_forces"):
            np.testing.assert_allclose(
                generated[name], fallback[name], rtol=0.0, atol=3e-5
            )

    evaluator._set_factorized_direct_reverse_executor("jit")
    evaluator._set_standard_r0_executor("v2_receiver")
    repeated = evaluate()
    assert evaluator.execution_dphi1r_active_bytes == 0
    assert repeated["energy"] == generated["energy"]
    for name in ("forces", "H1_adj", "directed_forces"):
        np.testing.assert_allclose(repeated[name], generated[name], rtol=0.0, atol=3e-5)

    with pytest.raises(ValueError, match="automatic.*runtime.*jit"):
        evaluator._set_factorized_direct_reverse_executor("unknown")
    with pytest.raises(
        ValueError,
        match="automatic.*runtime.*jit_all",
    ):
        evaluator._set_factorized_direct_forward_executor("unknown")
    with pytest.raises(
        ValueError,
        match="automatic.*v1.*v2_receiver.*v2_edge16.*v2_edge32",
    ):
        evaluator._set_standard_r0_executor("unknown")


@pytest.mark.parametrize(
    "dtype,streamed_edges",
    (
        ("float32", "factorized"),
        ("float32", "all_interactions"),
        ("float64", "all_interactions"),
    ),
)
def test_execution_standard_m0_matches_runtime_forward_reverse_oracle(
    streamed_model_paths,
    dtype,
    streamed_edges,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges=streamed_edges,
    )
    evaluator = calculator.evaluator
    if streamed_edges == "all_interactions":
        assert evaluator.standard_m0_selected_executor == "standard"
    if streamed_edges == "factorized":
        evaluator._set_factorized_source_strategy("jit_plugin")
        evaluator._set_factorized_direct_forward_executor("jit_all")
        evaluator._set_factorized_direct_reverse_executor("jit")
        evaluator._set_standard_r0_executor("v2_edge16")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    def evaluate(executor):
        evaluator._set_standard_m0_executor(executor)
        launch_counts = (
            evaluator.standard_m0_module_forward_launch_count,
            evaluator.standard_m0_module_reverse_launch_count,
        )
        evaluator.compute_node_energies_forces(*native_args)
        results = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": results["energy"],
            "energies": np.array(results["energies"], copy=True),
            "forces": np.array(results["forces"], copy=True),
            "stress": np.array(results["stress"], copy=True),
            "M0": np.array(evaluator.M0, copy=True),
            "A0_adj": np.array(evaluator.A0_adj, copy=True),
            "selected_executor": evaluator.standard_m0_selected_executor,
            "forward_launch_delta": (
                evaluator.standard_m0_module_forward_launch_count - launch_counts[0]
            ),
            "reverse_launch_delta": (
                evaluator.standard_m0_module_reverse_launch_count - launch_counts[1]
            ),
            "poly_values_active_bytes": (
                evaluator.standard_m0_poly_values_active_bytes
            ),
            "poly_values_capacity_bytes": (
                evaluator.standard_m0_poly_values_capacity_bytes
            ),
            "poly_adjoints_active_bytes": (
                evaluator.standard_m0_poly_adjoints_active_bytes
            ),
            "poly_adjoints_capacity_bytes": (
                evaluator.standard_m0_poly_adjoints_capacity_bytes
            ),
        }

    runtime = evaluate("runtime")
    assert evaluator.standard_m0_module_ready
    assert evaluator.standard_m0_module_fallback_reason == ""
    assert evaluator.standard_m0_module_id == ("standard-m0-module-module-v1")
    assert evaluator.standard_m0_model_structure_fingerprint
    assert runtime["selected_executor"] == "runtime"
    assert runtime["forward_launch_delta"] == 0
    assert runtime["reverse_launch_delta"] == 0
    assert (
        0
        < runtime["poly_values_active_bytes"]
        <= (runtime["poly_values_capacity_bytes"])
    )
    assert (
        0
        < runtime["poly_adjoints_active_bytes"]
        <= (runtime["poly_adjoints_capacity_bytes"])
    )

    standard = evaluate("standard")
    assert standard["selected_executor"] == "standard"
    assert standard["forward_launch_delta"] == 1
    assert standard["reverse_launch_delta"] == 1
    assert standard["poly_values_active_bytes"] == 0
    assert standard["poly_adjoints_active_bytes"] == 0
    assert standard["poly_values_capacity_bytes"] == 0
    assert standard["poly_adjoints_capacity_bytes"] == 0
    tolerance = 3e-5 if dtype == "float32" else 1e-10
    assert standard["energy"] == pytest.approx(
        runtime["energy"], rel=0.0, abs=tolerance
    )
    for name in ("energies", "forces", "stress", "M0", "A0_adj"):
        np.testing.assert_allclose(
            standard[name], runtime[name], rtol=0.0, atol=tolerance
        )

    repeated = evaluate("standard")
    assert repeated["forward_launch_delta"] == 1
    assert repeated["reverse_launch_delta"] == 1
    assert repeated["energy"] == standard["energy"]
    for name in ("energies", "forces", "stress", "M0", "A0_adj"):
        np.testing.assert_allclose(
            repeated[name], standard[name], rtol=0.0, atol=tolerance
        )

    restored = evaluate("runtime")
    assert restored["energy"] == runtime["energy"]
    assert restored["poly_values_capacity_bytes"] > 0
    assert restored["poly_adjoints_capacity_bytes"] > 0
    for name in ("energies", "forces", "stress", "M0", "A0_adj"):
        np.testing.assert_allclose(
            restored[name], runtime[name], rtol=0.0, atol=tolerance
        )

    with pytest.raises(ValueError, match="automatic.*runtime.*standard"):
        evaluator._set_standard_m0_executor("unknown")


@pytest.mark.parametrize("dtype,tolerance", (("float32", 3e-5), ("float64", 1e-10)))
def test_scalar_standard_m0_matches_runtime_forward_reverse_oracle(
    scalar_standard_model_path,
    dtype,
    tolerance,
):
    _require_execution_fixed_weight_backend()

    calculator = Symmetrix(
        scalar_standard_model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="all_interactions",
    )
    evaluator = calculator.evaluator
    atoms = _small_structure()
    atoms.set_pbc(True)
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    def evaluate(executor):
        evaluator._set_standard_m0_executor(executor)
        launch_counts = (
            evaluator.standard_m0_module_forward_launch_count,
            evaluator.standard_m0_module_reverse_launch_count,
        )
        evaluator.compute_node_energies_forces(*native_args)
        results = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": results["energy"],
            "forces": np.array(results["forces"], copy=True),
            "stress": np.array(results["stress"], copy=True),
            "M0": np.array(evaluator.M0, copy=True),
            "A0_adj": np.array(evaluator.A0_adj, copy=True),
            "forward_launches": (
                evaluator.standard_m0_module_forward_launch_count - launch_counts[0]
            ),
            "reverse_launches": (
                evaluator.standard_m0_module_reverse_launch_count - launch_counts[1]
            ),
        }

    runtime = evaluate("runtime")
    standard = evaluate("standard")

    assert evaluator.standard_m0_module_ready
    assert evaluator.standard_m0_module_fallback_reason == ""
    assert evaluator.standard_m0_module_id == "standard-m0-lmax0-module-v1"
    assert evaluator.standard_m0_module_revision == 1
    assert evaluator.standard_m0_model_structure_fingerprint == (
        "sha256:3b4ab2d979488798a05c07b5d54c4c96c1026da3ee1eb93069fd2bacec76ac90"
    )
    assert evaluator.standard_m0_selected_executor == "standard"
    assert runtime["forward_launches"] == 0
    assert runtime["reverse_launches"] == 0
    assert standard["forward_launches"] == 1
    assert standard["reverse_launches"] == 1
    assert evaluator.standard_m0_poly_values_active_bytes == 0
    assert evaluator.standard_m0_poly_adjoints_active_bytes == 0
    assert evaluator.standard_m0_poly_values_capacity_bytes == 0
    assert evaluator.standard_m0_poly_adjoints_capacity_bytes == 0
    assert standard["energy"] == pytest.approx(
        runtime["energy"], rel=0.0, abs=tolerance
    )
    for name in ("forces", "stress", "M0", "A0_adj"):
        np.testing.assert_allclose(
            standard[name], runtime[name], rtol=0.0, atol=tolerance
        )


@pytest.mark.parametrize(
    "variant",
    ("monomial_order", "degree", "component", "correlation", "term_count"),
)
def test_scalar_standard_m0_rejects_perturbed_structure(
    scalar_standard_model_path,
    tmp_path,
    variant,
):
    _require_execution_fixed_weight_backend()

    model = json.loads(scalar_standard_model_path.read_text())
    monomials = model["M0_monomials"]["0"]
    if variant == "monomial_order":
        asymmetric = next(
            index
            for index, components in enumerate(monomials)
            if len(components) == 3 and components != list(reversed(components))
        )
        monomials[asymmetric] = list(reversed(monomials[asymmetric]))
    elif variant == "degree":
        monomials[0] = [0, 0]
    elif variant == "component":
        monomials[0] = [1]
    elif variant == "correlation":
        model["M0_monomials"]["0"] = [
            components[:2] if len(components) == 3 else components
            for components in monomials
        ]
    else:
        monomials.pop()
        for type_weights in model["M0_weights"].values():
            for weights in type_weights["0"].values():
                weights.pop()
    path = tmp_path / f"scalar-m0-{variant}.json"
    path.write_text(json.dumps(model, separators=(",", ":")))

    calculator = Symmetrix(
        path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="all_interactions",
    )
    evaluator = calculator.evaluator
    assert not evaluator.standard_m0_module_ready
    assert evaluator.standard_m0_module_id == ""
    assert evaluator.standard_m0_selected_executor == "runtime"
    with pytest.raises(ValueError, match="Standard M0 is unavailable"):
        evaluator._set_standard_m0_executor("standard")


@pytest.mark.parametrize(
    "field,value,error",
    (
        ("L_max", 1, "missing monomials for output component 1"),
        ("l_max", 2, "outside the A0 component range"),
    ),
)
def test_scalar_standard_m0_rejects_incompatible_angular_topology(
    scalar_standard_model_path,
    tmp_path,
    field,
    value,
    error,
):
    model = json.loads(scalar_standard_model_path.read_text())
    model[field] = value
    path = tmp_path / f"scalar-m0-{field}.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    with pytest.raises(ValueError, match=error):
        Symmetrix(
            path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="all_interactions",
        )


@pytest.mark.parametrize("dtype,tolerance", (("float32", 4e-5), ("float64", 3e-10)))
def test_scalar_standard_m0_low_memory_matches_retained_direct(
    scalar_standard_model_path,
    monkeypatch,
    tmp_path,
    dtype,
    tolerance,
):
    _require_execution_fixed_weight_backend()
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))

    calculators = {
        "retained": Symmetrix(
            scalar_standard_model_path,
            use_kokkos=True,
            dtype=dtype,
            streamed_edges="factorized",
            low_memory=False,
        ),
        "low_memory": Symmetrix(
            scalar_standard_model_path,
            use_kokkos=True,
            dtype=dtype,
            streamed_edges="factorized",
            low_memory=True,
        ),
    }
    low_memory_evaluator = calculators["low_memory"].evaluator
    assert all(
        calculator.allow_fixed_workspace is False
        and calculator.evaluator.allow_fixed_workspace is False
        for calculator in calculators.values()
    )
    if native_symmetrix._kokkos_default_execution_space() in ("Cuda", "HIP"):
        low_memory_evaluator._set_low_memory_device_memory_info_for_testing(
            512 * 1024**2, 32 * 1024**3
        )
    outputs = {}
    for name, calculator in calculators.items():
        atoms = _small_structure()
        atoms.set_pbc(True)
        calculator.calculate(atoms, properties=["energy", "forces", "stress"])
        outputs[name] = {
            property_name: np.array(calculator.results[property_name], copy=True)
            for property_name in ("energy", "forces", "stress")
        }

    evaluator = low_memory_evaluator
    assert evaluator.low_memory
    assert evaluator.standard_m0_module_id == "standard-m0-lmax0-module-v1"
    if native_symmetrix._kokkos_default_execution_space() in ("Cuda", "HIP"):
        assert evaluator.standard_m0_selected_executor == "standard"
        assert evaluator.m0_implementation == "builtin"
    else:
        assert evaluator.standard_m0_selected_executor == "runtime"
        assert evaluator.m0_implementation == "host_plugin"
    assert evaluator.standard_r0_selected_executor != "v1"
    assert evaluator.factorized_jit_artifact_id
    assert evaluator.factorized_fallback_evaluation_count == 0
    assert evaluator.mh0_state_policy == "reuse-adjoints-v1"
    assert evaluator.standard_m0_input_scale_adjoint_launch_count > 0
    assert evaluator.standard_m0_poly_values_capacity_bytes == 0
    assert evaluator.standard_m0_poly_adjoints_capacity_bytes == 0
    for property_name in outputs["retained"]:
        np.testing.assert_allclose(
            outputs["low_memory"][property_name],
            outputs["retained"][property_name],
            rtol=0.0,
            atol=tolerance,
        )


@pytest.mark.parametrize("streamed_edges", ("factorized", "all_interactions"))
@pytest.mark.parametrize("dtype,tolerance", (("float32", 3e-5), ("float64", 1e-10)))
def test_m1_recompute_matches_retained_and_releases_polynomial_storage(
    streamed_model_paths,
    streamed_edges,
    dtype,
    tolerance,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges=streamed_edges,
        m1_polynomial_policy="retained",
    )
    evaluator = calculator.evaluator
    assert evaluator.m1_polynomial_policy == "retained"
    assert evaluator.m1_recompute_tile_channels in (8, 16, 32)
    if streamed_edges == "factorized":
        evaluator._set_factorized_source_strategy("jit_plugin")
        evaluator._set_factorized_direct_forward_executor("jit_all")
        evaluator._set_factorized_direct_reverse_executor("jit")
        evaluator._set_standard_r0_executor("v2_edge16")
    evaluator._set_standard_m0_executor("standard")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    def evaluate(policy, tile_channels):
        evaluator._set_m1_polynomial_policy("retained")
        evaluator._set_m1_recompute_tile_channels(tile_channels)
        evaluator._set_m1_polynomial_policy(policy)
        launch_counts = (
            evaluator.m1_recompute_forward_launch_count,
            evaluator.m1_recompute_reverse_launch_count,
        )
        evaluator.compute_node_energies_forces(*native_args)
        results = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": results["energy"],
            "energies": np.array(results["energies"], copy=True),
            "forces": np.array(results["forces"], copy=True),
            "stress": np.array(results["stress"], copy=True),
            "M1": np.array(evaluator.M1, copy=True),
            "A1_adj": np.array(evaluator.A1_adj, copy=True),
            "values_active_bytes": evaluator.m1_poly_values_active_bytes,
            "values_capacity_bytes": evaluator.m1_poly_values_capacity_bytes,
            "adjoints_active_bytes": evaluator.m1_poly_adjoints_active_bytes,
            "adjoints_capacity_bytes": evaluator.m1_poly_adjoints_capacity_bytes,
            "scratch_bytes": evaluator.m1_recompute_scratch_bytes,
            "forward_launch_delta": (
                evaluator.m1_recompute_forward_launch_count - launch_counts[0]
            ),
            "reverse_launch_delta": (
                evaluator.m1_recompute_reverse_launch_count - launch_counts[1]
            ),
        }

    retained = evaluate("retained", 8)
    assert 0 < retained["values_active_bytes"] <= (retained["values_capacity_bytes"])
    assert (
        0 < retained["adjoints_active_bytes"] <= (retained["adjoints_capacity_bytes"])
    )
    assert retained["scratch_bytes"] == 0
    assert retained["forward_launch_delta"] == 0
    assert retained["reverse_launch_delta"] == 0

    recomputed = {}
    scratch_limit = evaluator.m1_recompute_scratch_limit_bytes
    for requested_tile in (8, 16, 32):
        result = evaluate("recompute", requested_tile)
        selected_tile = evaluator.m1_recompute_tile_channels
        recomputed[selected_tile] = result
        assert selected_tile in (8, 16, 32)
        assert selected_tile <= requested_tile
        assert result["values_active_bytes"] == 0
        assert result["values_capacity_bytes"] == 0
        assert result["adjoints_active_bytes"] == 0
        assert result["adjoints_capacity_bytes"] == 0
        if (
            evaluator.standard_m1_module_ready
            and evaluator.m1_recompute_backend == "host"
        ):
            assert result["scratch_bytes"] == 0
        else:
            assert result["scratch_bytes"] > 0
        assert result["forward_launch_delta"] == 1
        assert result["reverse_launch_delta"] == 1
        assert result["energy"] == pytest.approx(
            retained["energy"], rel=0.0, abs=tolerance
        )
        for name in ("energies", "forces", "stress", "M1", "A1_adj"):
            np.testing.assert_allclose(
                result[name], retained[name], rtol=0.0, atol=tolerance
            )
        assert result["scratch_bytes"] <= scratch_limit

    selected_tile = max(recomputed)
    repeated = evaluate("recompute", selected_tile)
    assert repeated["energy"] == recomputed[selected_tile]["energy"]
    for name in ("energies", "forces", "stress", "M1", "A1_adj"):
        np.testing.assert_allclose(
            repeated[name],
            recomputed[selected_tile][name],
            rtol=0.0,
            atol=tolerance,
        )

    restored = evaluate("retained", 32)
    assert restored["values_capacity_bytes"] > 0
    assert restored["adjoints_capacity_bytes"] > 0
    assert restored["energy"] == retained["energy"]
    for name in ("energies", "forces", "stress", "M1", "A1_adj"):
        np.testing.assert_allclose(
            restored[name], retained[name], rtol=0.0, atol=tolerance
        )

    with pytest.raises(ValueError, match="retained.*recompute"):
        evaluator._set_m1_polynomial_policy("unknown")
    with pytest.raises(ValueError, match="8, 16, or 32"):
        evaluator._set_m1_recompute_tile_channels(4)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_m1_recompute_automatic_admission_and_precise_fallback(
    streamed_model_paths,
    dtype,
):
    _require_execution_fixed_weight_backend()
    standard_path, _ = streamed_model_paths
    calculator = debug_factorized_symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="all_interactions",
        m1_polynomial_policy="retained",
    )
    evaluator = calculator.evaluator
    evaluator._set_m1_recompute_tile_channels(8)
    evaluator._set_m1_polynomial_policy("recompute")
    minimum_scratch = evaluator.m1_recompute_scratch_bytes
    if evaluator.standard_m1_module_ready and evaluator.m1_recompute_backend == "host":
        assert minimum_scratch == 0
        evaluator._set_m1_polynomial_policy("retained")
        evaluator._set_m1_recompute_scratch_limit_for_testing(1)
        evaluator._set_m1_polynomial_policy("automatic")
        assert evaluator.m1_polynomial_policy == "recompute"
        assert evaluator.m1_recompute_fallback_reason == ""
        evaluator._set_m1_recompute_scratch_limit_for_testing(0)
        return
    assert minimum_scratch > 1

    evaluator._set_m1_polynomial_policy("retained")
    evaluator._set_m1_recompute_scratch_limit_for_testing(minimum_scratch - 1)
    evaluator._set_m1_polynomial_policy("automatic")
    assert evaluator.m1_polynomial_policy_request == "automatic"
    assert evaluator.m1_polynomial_policy == "retained"
    assert evaluator.m1_recompute_backend in ("cuda", "hip", "host")
    assert f"requires at least {minimum_scratch}" in (
        evaluator.m1_recompute_fallback_reason
    )
    assert f"available limit is {minimum_scratch - 1}" in (
        evaluator.m1_recompute_fallback_reason
    )

    with pytest.raises(ValueError, match="requires at least.*available limit"):
        evaluator._set_m1_polynomial_policy("recompute")

    evaluator._set_m1_recompute_scratch_limit_for_testing(minimum_scratch)
    evaluator._set_m1_polynomial_policy("automatic")
    assert evaluator.m1_polynomial_policy == "recompute"
    assert evaluator.m1_recompute_tile_channels == 8
    assert evaluator.m1_recompute_scratch_bytes == minimum_scratch
    assert evaluator.m1_recompute_fallback_reason == ""
    evaluator._set_m1_recompute_scratch_limit_for_testing(0)


def test_mh0_adjoint_reuse_matches_full_retention_and_rolls_back(
    streamed_model_paths,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
        m1_polynomial_policy="recompute",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_forward_executor("jit_all")
    evaluator._set_factorized_direct_reverse_executor("jit")
    evaluator._set_standard_r0_executor("v2_edge16")
    evaluator._set_standard_m0_executor("standard")
    direct_scratch = evaluator.m1_recompute_scratch_bytes
    if evaluator.m1_recompute_backend in ("cuda", "hip"):
        evaluator._set_m1_recompute_scratch_limit_for_testing(direct_scratch)
        evaluator._set_mh0_state_policy("reuse-adjoints-v1")
        assert evaluator.m1_recompute_scratch_bytes == direct_scratch
        evaluator._set_mh0_state_policy("full-retention-v1")
        evaluator._set_m1_recompute_scratch_limit_for_testing(0)
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    def evaluate(policy):
        evaluator._set_mh0_state_policy(policy)
        evaluator.compute_node_energies_forces(*native_args)
        return calculator._collect_mace_results(atoms, inputs)

    retained = evaluate("full-retention-v1")
    assert evaluator.mh0_reused_state_bytes == 0
    standard_reverse_before = evaluator.standard_m1_module_reverse_launch_count
    reused = evaluate("reuse-adjoints-v1")
    if evaluator.standard_m1_module_ready and evaluator.m1_recompute_backend in (
        "cuda",
        "hip",
    ):
        assert evaluator.standard_m1_module_reverse_launch_count == (
            standard_reverse_before + 1
        )
    expected_reused_bytes = np.dtype(np.float32).itemsize * sum(
        len(getattr(evaluator, name)) for name in ("M1", "A1", "M0", "A0")
    ) + np.dtype(np.float64).itemsize * len(evaluator.H2)
    if _kokkos_execution_space() in ("OpenMP", "Serial"):
        expected_reused_bytes += np.dtype(np.float32).itemsize * len(evaluator.H1)
        assert evaluator.h1_m0_adjoint_ping_pong_active
        assert evaluator.h1_m0_adjoint_ping_pong_bytes == (
            np.dtype(np.float32).itemsize * len(evaluator.H1)
        )
    else:
        assert not evaluator.h1_m0_adjoint_ping_pong_active
        assert evaluator.h1_m0_adjoint_ping_pong_bytes == 0
    assert evaluator.mh0_state_policy == "reuse-adjoints-v1"
    assert evaluator.mh0_state_policy_fallback_reason == ""
    assert evaluator.mh0_reused_state_bytes == expected_reused_bytes
    assert evaluator.mh0_auxiliary_state_bytes == 16 * len(atoms)
    assert reused["energy"] == pytest.approx(retained["energy"], rel=0.0, abs=3e-5)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(reused[name], retained[name], rtol=0.0, atol=3e-5)

    evaluator._set_standard_m0_executor("runtime")
    evaluator.compute_node_energies_forces(*native_args)
    fallback = calculator._collect_mace_results(atoms, inputs)
    assert evaluator.mh0_state_policy_request == "reuse-adjoints-v1"
    assert evaluator.mh0_state_policy == "full-retention-v1"
    assert evaluator.mh0_state_policy_fallback_reason == (
        "a state-free M0 implementation is not active"
    )
    assert evaluator.mh0_reused_state_bytes == 0
    assert evaluator.mh0_auxiliary_state_bytes == 0
    assert not evaluator.h1_m0_adjoint_ping_pong_active
    assert evaluator.h1_m0_adjoint_ping_pong_bytes == 0
    assert fallback["energy"] == pytest.approx(retained["energy"], rel=0.0, abs=3e-5)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(fallback[name], retained[name], rtol=0.0, atol=3e-5)

    evaluator._set_standard_m0_executor("standard")
    restored = evaluate("full-retention-v1")
    assert evaluator.mh0_state_policy == "full-retention-v1"
    assert evaluator.mh0_reused_state_bytes == 0
    assert evaluator.mh0_auxiliary_state_bytes == 0
    assert not evaluator.h1_m0_adjoint_ping_pong_active
    assert evaluator.h1_m0_adjoint_ping_pong_bytes == 0
    assert restored["energy"] == retained["energy"]
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(restored[name], retained[name], rtol=0.0, atol=3e-5)

    evaluator._set_m1_polynomial_policy("retained")
    with pytest.raises(ValueError, match="requires M1 polynomial recomputation"):
        evaluator._set_mh0_state_policy("reuse-adjoints-v1")


@pytest.mark.parametrize(
    ("dtype", "geometry_policy", "atol"),
    (
        ("float32", "unit-f32-radius-f64-v1", 3e-5),
        ("float64", "cartesian-f64-v1", 3e-11),
    ),
)
def test_factorized_low_memory_bundle_matches_full_retention_and_is_transactional(
    streamed_model_paths,
    monkeypatch,
    tmp_path,
    dtype,
    geometry_policy,
    atol,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))

    retained = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="factorized",
        m1_polynomial_policy="retained",
    )
    low_memory = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="factorized",
        low_memory=True,
    )
    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space in ("Cuda", "HIP"):
        device_bytes = 32 * 1024**3
        selection_atoms = _small_structure()
        selection_atoms.set_pbc(True)
        selection_inputs = low_memory._mace_inputs(selection_atoms)
        low_memory.evaluator._set_low_memory_device_memory_info_for_testing(
            device_bytes, device_bytes
        )
        low_memory.evaluator._prepare_factorized_graph(*selection_inputs[:5])
        capacity_estimate = low_memory.evaluator.low_memory_capacity_estimated_bytes
        speed_estimate = low_memory.evaluator.low_memory_speed_estimated_bytes
        reserve = device_bytes // 20
        assert capacity_estimate < speed_estimate
        low_memory.evaluator._set_low_memory(True)
        capacity_free_bytes = reserve + capacity_estimate + 1
        low_memory.evaluator._set_low_memory_device_memory_info_for_testing(
            capacity_free_bytes, device_bytes
        )
    outputs = {}
    for name, calculator in (("retained", retained), ("low_memory", low_memory)):
        atoms = _small_structure()
        atoms.set_pbc(True)
        calculator.calculate(atoms, properties=["energy", "forces", "stress"])
        outputs[name] = {
            prop: np.array(calculator.results[prop], copy=True)
            for prop in ("energy", "forces", "stress")
        }

    evaluator = low_memory.evaluator
    num_nodes = len(_small_structure())
    assert retained.evaluator.readout_workspace_bytes == 2328 * num_nodes
    assert evaluator.readout_workspace_bytes == 0
    assert retained.evaluator.readout_policy == "retained"
    assert evaluator.readout_policy == "recompute"
    assert retained.evaluator.phi1_policy == "retained"
    assert evaluator.phi1_policy == "retained"
    assert evaluator.phi1_workspace_bytes > 0
    with pytest.raises(ValueError, match="Phi1 policy must be"):
        evaluator._set_phi1_policy("unknown")
    assert low_memory.low_memory_request is True
    assert low_memory.low_memory is True
    assert low_memory.low_memory_policy in (
        "capacity-retained",
        "capacity-y-only",
    )
    assert evaluator.low_memory_requested is True
    assert evaluator.low_memory is True
    assert evaluator.low_memory_policy in ("capacity-retained", "capacity-y-only")
    assert evaluator.m1_polynomial_policy == "recompute"
    assert evaluator.mh0_state_policy == "reuse-adjoints-v1"
    assert evaluator.edge_geometry_policy == geometry_policy
    if execution_space in ("Cuda", "HIP"):
        assert evaluator.harmonic_storage_policy_request == "automatic"
        assert evaluator.low_memory_policy == "capacity-y-only"
        assert evaluator.harmonic_storage_policy == "y-only-direct-v1"
        assert evaluator.harmonic_storage_fallback_reason == ""
        assert evaluator.harmonic_value_bytes > 0
        assert evaluator.harmonic_gradient_bytes == 0
        assert evaluator.shuffled_coordinate_bytes == 0
        assert evaluator.low_memory_device_free_bytes == capacity_free_bytes
        assert evaluator.low_memory_device_total_bytes == device_bytes
        assert evaluator.low_memory_reserve_bytes == device_bytes // 20
        assert 0 < evaluator.low_memory_capacity_y_only_estimated_bytes
        assert (
            evaluator.low_memory_capacity_retained_estimated_bytes
            > evaluator.low_memory_capacity_y_only_estimated_bytes
        )
    else:
        assert evaluator.harmonic_storage_policy_request == "automatic"
        assert evaluator.harmonic_storage_policy == "y-only-direct-v1"
        assert evaluator.harmonic_storage_fallback_reason == ""
        assert evaluator.harmonic_value_bytes > 0
        assert evaluator.harmonic_gradient_bytes == 0
        assert evaluator.shuffled_coordinate_bytes == 0
    assert evaluator.standard_r0_module_ready
    assert evaluator.standard_r0_module_fallback_reason == ""
    assert evaluator.mh0_reused_state_bytes > 0
    if dtype == "float32":
        assert evaluator.compact_edge_geometry_bytes > 0
    else:
        assert evaluator.compact_edge_geometry_bytes == 0
    for prop in outputs["retained"]:
        np.testing.assert_allclose(
            outputs["low_memory"][prop],
            outputs["retained"][prop],
            rtol=0.0,
            atol=atol,
        )

    evaluator._set_low_memory(False)
    assert evaluator.low_memory_requested is False
    assert evaluator.low_memory_policy == "disabled"
    assert evaluator.low_memory is False
    assert evaluator.readout_policy == "retained"
    assert evaluator.m1_polynomial_policy_request == "automatic"
    assert evaluator.m1_polynomial_policy == "recompute"
    assert evaluator.mh0_state_policy == "full-retention-v1"
    assert evaluator.edge_geometry_policy == "cartesian-f64-v1"
    assert evaluator.phi1_policy == "retained"
    assert evaluator.harmonic_storage_policy == "retained"
    assert evaluator.harmonic_storage_policy_request == "retained"
    assert evaluator.harmonic_storage_fallback_reason == ""

    evaluator._set_m1_recompute_tile_channels(8)
    evaluator._set_m1_polynomial_policy("recompute")
    minimum_scratch = evaluator.m1_recompute_scratch_bytes
    evaluator._set_m1_polynomial_policy("retained")
    if minimum_scratch == 0:
        assert evaluator.standard_m1_module_ready
        evaluator._set_low_memory(True)
        assert evaluator.low_memory is True
        evaluator._set_low_memory(False)
    else:
        evaluator._set_m1_recompute_scratch_limit_for_testing(minimum_scratch - 1)
        with pytest.raises(
            ValueError, match=r"requires (?:at least )?.*scratch bytes.*tile 8"
        ) as error:
            evaluator._set_low_memory(True)
        if evaluator.m1_recompute_backend in ("cuda", "hip"):
            assert f"requires at least {minimum_scratch}" in str(error.value)
        assert evaluator.low_memory is False
        assert evaluator.m1_polynomial_policy_request == "retained"
        assert evaluator.m1_polynomial_policy == "retained"
        assert evaluator.mh0_state_policy == "full-retention-v1"
        assert evaluator.edge_geometry_policy == "cartesian-f64-v1"
    evaluator._set_m1_recompute_scratch_limit_for_testing(0)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_low_memory_selects_speed_or_capacity_from_graph_memory_estimate(
    streamed_model_paths,
    monkeypatch,
    tmp_path,
    dtype,
):
    _require_execution_fixed_weight_backend()
    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space not in ("Cuda", "HIP"):
        pytest.skip("automatic harmonic memory selection requires CUDA or HIP")

    standard_path, _ = streamed_model_paths
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=True,
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(_small_structure())
    device_bytes = 32 * 1024**3

    evaluator._set_low_memory_device_memory_info_for_testing(device_bytes, device_bytes)
    retained_token = evaluator._prepare_factorized_graph(*inputs[:5])
    assert evaluator.low_memory_requested is True
    assert evaluator.low_memory_policy == "speed"
    assert evaluator.low_memory is True
    assert evaluator.low_memory_selection_reason.startswith("speed selection:")
    assert evaluator.harmonic_storage_policy_request == "automatic"
    assert evaluator.harmonic_storage_policy == "retained"
    assert evaluator.mh0_state_policy == "reuse-adjoints-v1"
    assert evaluator.readout_policy == "recompute"
    assert evaluator.edge_geometry_policy == (
        "unit-f32-radius-f64-v1" if dtype == "float32" else "cartesian-f64-v1"
    )
    assert evaluator.low_memory_speed_estimated_bytes > 0
    assert evaluator.low_memory_selected_estimated_bytes == (
        evaluator.low_memory_speed_estimated_bytes
    )
    calculator._sync_low_memory_policy_state()
    assert calculator.low_memory is True
    assert calculator.low_memory_policy == "speed"
    retained_estimate = evaluator.low_memory_capacity_retained_estimated_bytes
    y_only_estimate = evaluator.low_memory_capacity_y_only_estimated_bytes
    assert evaluator.low_memory_speed_estimated_bytes == retained_estimate
    expected_delta = len(inputs[3]) * (3 + 3 * 16) * np.dtype(dtype).itemsize
    assert retained_estimate - y_only_estimate == expected_delta
    evaluator._compute_prepared_factorized(
        retained_token, np.asarray(inputs[5]).reshape(-1), inputs[6]
    )
    retained = calculator._collect_mace_results(_small_structure(), inputs)

    capacity_free_bytes = device_bytes // 20 + y_only_estimate + 1
    assert (
        capacity_free_bytes - device_bytes // 20
        < evaluator.low_memory_speed_estimated_bytes
    )
    evaluator._set_low_memory(True)
    evaluator._set_low_memory_device_memory_info_for_testing(
        device_bytes // 20 + y_only_estimate - 1,
        device_bytes,
    )
    constrained_token = evaluator._prepare_factorized_graph(*inputs[:5])
    assert constrained_token > retained_token
    assert evaluator.low_memory_policy == "capacity-y-only"
    assert evaluator.harmonic_storage_policy == "y-only-direct-v1"
    assert evaluator.low_memory_selected_estimated_bytes == y_only_estimate
    assert evaluator.execution_planned_capacity_bytes == y_only_estimate
    assert evaluator.execution_planned_edge_capacity == len(inputs[3])
    assert "minimum estimate exceeds advisory" in (
        evaluator.low_memory_selection_reason
    )
    assert evaluator.execution_capacity_selection_reason.startswith(
        "admitted 0.25% repeated-run high-water within post-reserve bytes;"
    )
    evaluator._compute_prepared_factorized(
        constrained_token, np.asarray(inputs[5]).reshape(-1), inputs[6]
    )
    y_only = calculator._collect_mace_results(_small_structure(), inputs)
    atol = 3e-5 if dtype == "float32" else 3e-11
    for property_name in ("energy", "energies", "forces", "stress"):
        np.testing.assert_allclose(
            y_only[property_name], retained[property_name], rtol=0.0, atol=atol
        )

    evaluator._set_low_memory_device_memory_info_for_testing(
        capacity_free_bytes, device_bytes
    )
    reused_token = evaluator._prepare_factorized_graph(*inputs[:5])
    assert reused_token > constrained_token
    assert evaluator.low_memory_policy == "speed"
    assert evaluator.low_memory is True
    assert evaluator.low_memory_selection_reason.startswith("speed selection:")
    assert evaluator.harmonic_storage_policy == "retained"
    assert evaluator.mh0_state_policy == "reuse-adjoints-v1"
    assert evaluator.readout_policy == "recompute"
    assert evaluator.edge_geometry_policy == (
        "unit-f32-radius-f64-v1" if dtype == "float32" else "cartesian-f64-v1"
    )
    assert evaluator.low_memory_reserve_bytes == device_bytes // 20
    assert evaluator.low_memory_capacity_retained_estimated_bytes == retained_estimate
    assert evaluator.low_memory_capacity_y_only_estimated_bytes == y_only_estimate
    assert evaluator.low_memory_capacity_estimated_bytes in (
        retained_estimate,
        y_only_estimate,
    )
    assert evaluator.low_memory_selected_estimated_bytes >= y_only_estimate
    calculator._sync_low_memory_policy_state()
    assert calculator.low_memory_policy == "speed"

    evaluator._set_low_memory_device_memory_info_for_testing(0, 0)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_speed_uses_backend_throughput_policy_and_debug_override_is_exact(
    scalar_standard_model_path,
    monkeypatch,
    tmp_path,
    dtype,
):
    execution_space = _require_execution_fixed_weight_backend()

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    atoms = _small_structure()
    calculators = []
    for profile, debug_plan in (
        ("speed", "mh0-direct-speed"),
        ("capacity", "mh0-direct-capacity-retained"),
    ):
        calculator = Symmetrix(
            scalar_standard_model_path,
            use_kokkos=True,
            dtype=dtype,
            streamed_edges="direct",
            execution_profile=profile,
            _debug_execution_plan=debug_plan,
        )
        inputs = calculator._mace_inputs(atoms)
        calculator.evaluator._prepare_factorized_graph(*inputs[:5])
        calculators.append(calculator)

    speed = calculators[0].evaluator
    calculators[0]._sync_low_memory_policy_state()
    assert calculators[0].low_memory_policy == "speed"
    assert calculators[0].low_memory is False
    if execution_space in ("OpenMP", "Serial"):
        assert speed.execution_plan_report["candidates"][0]["reason"] == (
            "memory query unavailable"
        )
    assert speed.execution_plan_report["selected_id"] == "mh0-direct-speed"
    assert speed.low_memory_policy == "speed"
    expected_harmonics = "retained"
    assert speed.harmonic_storage_policy == expected_harmonics
    assert speed.mh0_state_policy == "reuse-adjoints-v1"
    assert speed.readout_policy == "recompute"
    expected_geometry = (
        "unit-f32-radius-f64-v1" if dtype == "float32" else "cartesian-f64-v1"
    )
    assert speed.edge_geometry_policy == expected_geometry
    expected_speed_estimate = speed.low_memory_capacity_retained_estimated_bytes
    assert speed.low_memory_speed_estimated_bytes == expected_speed_estimate

    edge_repetitions = 1 + 40960 // len(inputs[3])
    large_inputs = (
        inputs[0],
        inputs[1],
        np.asarray(inputs[2]) * edge_repetitions,
        np.tile(inputs[3], edge_repetitions),
        np.tile(inputs[4], edge_repetitions),
    )
    speed._prepare_factorized_graph(*large_inputs)
    large_uses_y_only = execution_space in ("OpenMP", "Serial") or (
        execution_space == "Cuda" and dtype == "float32"
    )
    assert speed.harmonic_storage_policy == (
        "y-only-direct-v1" if large_uses_y_only else "retained"
    )
    assert speed.low_memory_speed_estimated_bytes == (
        speed.low_memory_capacity_y_only_estimated_bytes
        if large_uses_y_only
        else speed.low_memory_capacity_retained_estimated_bytes
    )

    speed._set_factorized_observer(True)
    assert speed.mh0_state_policy == "full-retention-v1"
    assert speed.harmonic_storage_policy == "retained"
    assert speed.edge_geometry_policy == "cartesian-f64-v1"
    speed._set_factorized_observer(False)
    speed._prepare_factorized_graph(*calculators[0]._mace_inputs(atoms)[:5])
    assert speed.harmonic_storage_policy == expected_harmonics

    speed.set_execution_parameter_gradients(True)
    assert speed.mh0_state_policy == "full-retention-v1"
    assert speed.harmonic_storage_policy == "retained"
    assert speed.edge_geometry_policy == "cartesian-f64-v1"
    speed.set_execution_parameter_gradients(False)

    retained = calculators[1].evaluator
    assert retained.execution_plan_report["selected_id"] == (
        "mh0-direct-capacity-retained"
    )
    assert retained.low_memory_policy == "capacity-retained"
    assert retained.harmonic_storage_policy == "retained"
    assert retained.mh0_state_policy == "reuse-adjoints-v1"
    assert retained.readout_policy == "recompute"
    assert retained.edge_geometry_policy == expected_geometry
    assert "debug override selected" in retained.low_memory_selection_reason


def test_low_memory_high_water_reuses_graph_and_mh0_state_capacity(
    streamed_model_paths,
    monkeypatch,
    tmp_path,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        low_memory=True,
    )
    evaluator = calculator.evaluator
    small_inputs = calculator._mace_inputs(_small_structure())
    base_nodes = 128
    inputs = (
        base_nodes,
        np.pad(
            np.asarray(small_inputs[1]),
            (0, base_nodes - small_inputs[0]),
            constant_values=small_inputs[1][0],
        ),
        np.pad(
            np.asarray(small_inputs[2]),
            (0, base_nodes - small_inputs[0]),
            constant_values=0,
        ),
        small_inputs[3],
        small_inputs[4],
        small_inputs[5],
        small_inputs[6],
    )
    device_bytes = 32 * 1024**3
    evaluator._set_low_memory_device_memory_info_for_testing(device_bytes, device_bytes)
    evaluator._prepare_factorized_graph(*inputs[:5])
    capacity_estimate = evaluator.low_memory_capacity_estimated_bytes
    speed_estimate = evaluator.low_memory_speed_estimated_bytes
    reserve = device_bytes // 20
    admitted_headroom = speed_estimate - capacity_estimate - 1
    assert admitted_headroom > 0
    # Changing the synthetic device limit invalidates the prepared graph, so
    # no previous selected allocation is reclaimable by the next admission.
    constrained_free = reserve + capacity_estimate + admitted_headroom
    assert 0 < constrained_free < device_bytes

    # Use a fresh evaluator for admission. The estimator above deliberately
    # selected the speed policy and established an exact active-size plan;
    # retained plans are reused unchanged until active size exceeds them.
    del evaluator
    del calculator
    gc.collect()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        low_memory=True,
    )
    evaluator = calculator.evaluator

    offsets = np.concatenate(([0], np.cumsum(inputs[2])))
    permutation = np.concatenate(
        [np.arange(offsets[i], offsets[i + 1])[::-1] for i in range(inputs[0])]
    )
    reordered = (
        inputs[0],
        inputs[1],
        inputs[2],
        np.asarray(inputs[3])[permutation],
        np.asarray(inputs[4])[permutation],
        np.asarray(inputs[5])[permutation],
        np.asarray(inputs[6])[permutation],
    )
    evaluator._set_low_memory_device_memory_info_for_testing(
        constrained_free, device_bytes
    )
    token = evaluator._prepare_factorized_graph(*reordered[:5])
    evaluator._compute_prepared_factorized(
        token, np.asarray(reordered[5]).reshape(-1), reordered[6]
    )
    assert evaluator.low_memory_policy.startswith("capacity-")
    assert evaluator.execution_planned_receiver_capacity > inputs[0], {
        "capacity_estimate": capacity_estimate,
        "speed_estimate": speed_estimate,
        "available": evaluator.low_memory_available_bytes,
        "selected": evaluator.low_memory_selected_estimated_bytes,
        "planned_bytes": evaluator.execution_planned_capacity_bytes,
        "planned_receivers": evaluator.execution_planned_receiver_capacity,
        "planned_features": evaluator.execution_planned_feature_node_capacity,
        "planned_edges": evaluator.execution_planned_edge_capacity,
        "capacity_reason": evaluator.execution_capacity_selection_reason,
        "geometry_reason": evaluator.execution_geometry_growth_reason,
    }
    assert evaluator.execution_planned_feature_node_capacity > inputs[0]
    assert evaluator.execution_planned_edge_capacity > len(reordered[3])
    assert evaluator.mh0_m0_adjoint_alias_active
    assert evaluator.mh0_m0_adjoint_allocation_count == 0

    # cudaMemGetInfo() would report less free memory after this bundle was
    # allocated. Keep the synthetic provider consistent so reclaimable bytes
    # reconstruct, rather than double-count, the original admission budget.
    simulated_free_after_allocation = (
        constrained_free - evaluator.execution_planned_capacity_bytes
    )
    assert simulated_free_after_allocation > 0
    evaluator._set_low_memory_device_memory_info_for_testing(
        simulated_free_after_allocation, device_bytes
    )

    graph_replacements = evaluator.factorized_graph_device_replacement_count
    graph_updates = evaluator.factorized_graph_device_update_count
    planned_before_growth = (
        evaluator.execution_planned_receiver_capacity,
        evaluator.execution_planned_feature_node_capacity,
        evaluator.execution_planned_edge_capacity,
        evaluator.execution_planned_capacity_bytes,
        evaluator.factorized_graph_device_capacity_bytes,
    )
    geometry_allocations = evaluator.execution_geometry_allocation_count
    result_allocations = evaluator.execution_result_allocation_count
    geometry_capacity = evaluator.execution_geometry_capacity_edges
    selected_policy = evaluator.low_memory_policy
    m0_replacements = evaluator.mh0_m0_forward_replacement_count
    m0_detaches = evaluator.mh0_m0_alias_detach_count
    grown_nodes = inputs[0] + 1
    grown = (
        grown_nodes,
        np.append(np.asarray(inputs[1]), inputs[1][0]),
        np.append(np.asarray(inputs[2]), 0),
        reordered[3],
        reordered[4],
    )
    grown_token = evaluator._prepare_factorized_graph(*grown)
    assert evaluator.execution_geometry_allocation_count == geometry_allocations, {
        "before_capacity": geometry_capacity,
        "after_capacity": evaluator.execution_geometry_capacity_edges,
        "before_policy": selected_policy,
        "after_policy": evaluator.low_memory_policy,
        "growth_reason": evaluator.execution_geometry_growth_reason,
    }
    evaluator._compute_prepared_factorized(
        grown_token, np.asarray(reordered[5]).reshape(-1), reordered[6]
    )
    assert evaluator.factorized_graph_device_replacement_count == graph_replacements, {
        "planned_before": planned_before_growth,
        "planned_after": (
            evaluator.execution_planned_receiver_capacity,
            evaluator.execution_planned_feature_node_capacity,
            evaluator.execution_planned_edge_capacity,
            evaluator.execution_planned_capacity_bytes,
            evaluator.factorized_graph_device_capacity_bytes,
        ),
        "capacity_reason": evaluator.execution_capacity_selection_reason,
    }
    assert evaluator.factorized_graph_device_update_count == graph_updates + 1
    assert evaluator.execution_geometry_allocation_count == geometry_allocations
    assert evaluator.execution_result_allocation_count == result_allocations
    assert evaluator.mh0_m0_forward_replacement_count == m0_replacements
    assert evaluator.mh0_m0_alias_detach_count == m0_detaches
    assert evaluator.mh0_m0_adjoint_alias_active

    edge_insert = int(np.asarray(grown[2])[0])
    edge_degrees = np.asarray(grown[2]).copy()
    edge_degrees[0] += 1
    edge_grown = (
        grown_nodes,
        grown[1],
        edge_degrees,
        np.insert(np.asarray(reordered[3]), edge_insert, reordered[3][0]),
        np.insert(np.asarray(reordered[4]), edge_insert, reordered[4][0]),
    )
    edge_xyz = np.insert(
        np.asarray(reordered[5]).reshape(-1, 3),
        edge_insert,
        np.asarray(reordered[5]).reshape(-1, 3)[0],
        axis=0,
    ).reshape(-1)
    edge_r = np.insert(
        np.asarray(reordered[6]), edge_insert, np.asarray(reordered[6])[0]
    )
    edge_token = evaluator._prepare_factorized_graph(*edge_grown)
    evaluator._compute_prepared_factorized(edge_token, edge_xyz, edge_r)
    assert evaluator.factorized_graph_device_replacement_count == graph_replacements
    assert evaluator.factorized_graph_device_update_count == graph_updates + 2
    assert evaluator.execution_geometry_allocation_count == geometry_allocations
    assert evaluator.execution_result_allocation_count == result_allocations
    assert evaluator.mh0_m0_forward_replacement_count == m0_replacements
    assert evaluator.mh0_m0_alias_detach_count == m0_detaches
    assert evaluator.mh0_m0_adjoint_alias_active

    exceptional_nodes = evaluator.execution_planned_receiver_capacity + 1
    exceptional = (
        exceptional_nodes,
        np.pad(
            np.asarray(inputs[1]),
            (0, exceptional_nodes - inputs[0]),
            constant_values=inputs[1][0],
        ),
        np.pad(
            np.asarray(inputs[2]),
            (0, exceptional_nodes - inputs[0]),
            constant_values=0,
        ),
        reordered[3],
        reordered[4],
    )
    exceptional_token = evaluator._prepare_factorized_graph(*exceptional)
    assert evaluator.factorized_graph_device_replacement_count == (
        graph_replacements + 1
    )
    evaluator._compute_prepared_factorized(
        exceptional_token, np.asarray(reordered[5]).reshape(-1), reordered[6]
    )
    assert evaluator.mh0_m0_forward_replacement_count == m0_replacements + 1
    assert evaluator.mh0_m0_alias_detach_count == m0_detaches + 1
    assert evaluator.mh0_m0_adjoint_alias_active
    assert evaluator.mh0_m0_adjoint_allocation_count == 0
    assert evaluator.execution_result_allocation_count == result_allocations + 1


def test_low_memory_high_water_reuses_prepared_geometry_state_capacity(
    streamed_model_paths,
    monkeypatch,
    tmp_path,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        low_memory=True,
    )
    evaluator = calculator.evaluator
    small_inputs = calculator._mace_inputs(_small_structure())
    base_nodes = 128
    node_types = np.pad(
        np.asarray(small_inputs[1]),
        (0, base_nodes - small_inputs[0]),
        constant_values=small_inputs[1][0],
    )
    num_neigh = np.pad(
        np.asarray(small_inputs[2]),
        (0, base_nodes - small_inputs[0]),
        constant_values=0,
    )
    device_bytes = 32 * 1024**3
    evaluator._set_low_memory_device_memory_info_for_testing(device_bytes, device_bytes)
    evaluator._prepare_factorized_graph(
        base_nodes, node_types, num_neigh, small_inputs[3], small_inputs[4]
    )
    capacity_estimate = evaluator.low_memory_capacity_estimated_bytes
    speed_estimate = evaluator.low_memory_speed_estimated_bytes
    constrained_free = (
        device_bytes // 20 + capacity_estimate + speed_estimate - capacity_estimate - 1
    )

    del evaluator
    del calculator
    gc.collect()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        low_memory=True,
    )
    evaluator = calculator.evaluator
    evaluator._set_low_memory_device_memory_info_for_testing(
        constrained_free, device_bytes
    )
    token = evaluator._prepare_factorized_graph(
        base_nodes, node_types, num_neigh, small_inputs[3], small_inputs[4]
    )
    positions = np.zeros((base_nodes, 3), dtype=float)
    positions[: small_inputs[0]] = _small_structure().positions
    reference_xyz = np.asarray(small_inputs[5]).reshape(-1)
    cell = np.eye(3, dtype=float).reshape(-1)
    evaluator._prepare_factorized_geometry(
        token,
        positions.reshape(-1),
        reference_xyz,
        cell,
        cell,
        np.zeros(3, dtype=np.int32),
    )
    geometry_allocations = evaluator.factorized_geometry_state_allocation_count
    assert evaluator.execution_planned_receiver_capacity > base_nodes
    assert evaluator.execution_planned_edge_capacity > len(small_inputs[3])

    simulated_free_after_allocation = (
        constrained_free - evaluator.execution_planned_capacity_bytes
    )
    evaluator._set_low_memory_device_memory_info_for_testing(
        simulated_free_after_allocation, device_bytes
    )
    grown_nodes = base_nodes + 1
    grown_node_types = np.append(node_types, node_types[0])
    grown_num_neigh = np.append(num_neigh, 0)
    grown_token = evaluator._prepare_factorized_graph(
        grown_nodes,
        grown_node_types,
        grown_num_neigh,
        small_inputs[3],
        small_inputs[4],
    )
    grown_positions = np.vstack((positions, positions[0]))
    evaluator._prepare_factorized_geometry(
        grown_token,
        grown_positions.reshape(-1),
        reference_xyz,
        cell,
        cell,
        np.zeros(3, dtype=np.int32),
    )
    assert evaluator.factorized_geometry_state_allocation_count == geometry_allocations


def test_speed_policy_narrows_geometry_growth_before_device_boundary(
    streamed_model_paths,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    device_bytes = 32 * 1024**3
    atoms = _small_structure()

    def prepare_inputs(calculator):
        inputs = calculator._mace_inputs(atoms)
        insertion = int(inputs[2][0])
        repeated_num_neigh = np.array(inputs[2], copy=True)
        repeated_num_neigh[0] += 1
        repeated = (
            inputs[0],
            inputs[1],
            repeated_num_neigh,
            np.insert(np.asarray(inputs[3]), insertion, inputs[3][0]),
            np.insert(np.asarray(inputs[4]), insertion, inputs[4][0]),
        )
        return inputs, repeated

    probe_calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        low_memory=False,
    )
    probe = probe_calculator.evaluator
    probe._set_low_memory_device_memory_info_for_testing(device_bytes, device_bytes)
    probe_inputs, probe_repeated = prepare_inputs(probe_calculator)
    probe._prepare_factorized_graph(*probe_inputs[:5])
    probe_initial_capacity = probe.execution_geometry_capacity_edges
    probe._prepare_factorized_graph(*probe_repeated)
    exact_repeated_estimate = probe.low_memory_speed_estimated_bytes
    assert probe.execution_geometry_capacity_edges == 2 * probe_initial_capacity
    assert probe.low_memory_selected_estimated_bytes > exact_repeated_estimate

    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        low_memory=False,
    )
    evaluator = calculator.evaluator
    evaluator._set_low_memory_device_memory_info_for_testing(device_bytes, device_bytes)
    inputs, repeated = prepare_inputs(calculator)
    evaluator._prepare_factorized_graph(*inputs[:5])
    previous_estimate = evaluator.low_memory_selected_estimated_bytes
    reserve = device_bytes // 20
    desired_available = exact_repeated_estimate + 1024
    constrained_free = desired_available + reserve - previous_estimate
    assert 0 < constrained_free < device_bytes
    evaluator._set_low_memory_device_memory_info_for_testing(
        constrained_free, device_bytes
    )

    evaluator._prepare_factorized_graph(*repeated)
    assert evaluator.low_memory_policy == "disabled"
    assert evaluator.execution_geometry_capacity_edges == len(repeated[3])
    assert evaluator.low_memory_selected_estimated_bytes == (
        evaluator.low_memory_speed_estimated_bytes
    )
    assert evaluator.execution_geometry_growth_reason.startswith("exact growth:")

    exact_capacity = evaluator.execution_geometry_capacity_edges
    evaluator._prepare_factorized_graph(*inputs[:5])
    assert evaluator.execution_geometry_capacity_edges == exact_capacity
    assert evaluator.execution_geometry_growth_reason == "existing capacity reused"

    insertion = int(repeated[2][0])
    too_large_degrees = np.array(repeated[2], copy=True)
    too_large_degrees[0] += 1
    too_large = (
        repeated[0],
        repeated[1],
        too_large_degrees,
        np.insert(np.asarray(repeated[3]), insertion, repeated[3][0]),
        np.insert(np.asarray(repeated[4]), insertion, repeated[4][0]),
    )
    evaluator._set_low_memory_device_memory_info_for_testing(1, device_bytes)
    evaluator._prepare_factorized_graph(*too_large)
    assert evaluator.low_memory_policy == "disabled"
    assert evaluator.execution_geometry_capacity_edges == len(too_large[3])
    assert evaluator.low_memory_selected_estimated_bytes == (
        evaluator.low_memory_speed_estimated_bytes
    )
    assert "allocation determines feasibility" in (
        evaluator.low_memory_selection_reason
    )
    assert evaluator.execution_geometry_growth_reason.startswith("exact growth:")


def test_readout_recompute_matches_macefield_directional_response(
    streamed_model_paths,
    monkeypatch,
    tmp_path,
):
    _require_execution_fixed_weight_backend()

    _, field_path = streamed_model_paths
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    electric_field = np.array([0.01, -0.02, 0.03])
    properties = (
        "energy",
        "forces",
        "stress",
        "polarization",
        "polarizability",
        "becs",
    )
    outputs = {}
    workspaces = {}
    for policy in ("retained", "recompute"):
        calculator = Symmetrix(
            field_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="factorized",
            electric_field=electric_field,
        )
        calculator.evaluator._set_readout_policy(policy)
        atoms = _small_structure()
        calculator.calculate(atoms, properties=properties)
        outputs[policy] = {
            name: np.asarray(calculator.results[name]).copy() for name in properties
        }
        workspaces[policy] = calculator.evaluator.readout_workspace_bytes

    assert workspaces["retained"] > 0
    assert workspaces["recompute"] == 0
    for name in ("energy", "forces", "stress", "polarization"):
        np.testing.assert_allclose(
            outputs["recompute"][name],
            outputs["retained"][name],
            rtol=0.0,
            atol=5e-5,
        )
    for name in ("polarizability", "becs"):
        np.testing.assert_allclose(
            outputs["recompute"][name],
            outputs["retained"][name],
            rtol=0.0,
            atol=5e-4,
        )


@pytest.mark.parametrize(
    "variant,has_contract",
    [
        ("missing", False),
        ("partial", False),
        ("stale_structure", True),
    ],
)
def test_standard_m0_module_admits_validated_payload_without_contract_metadata(
    streamed_model_paths,
    tmp_path,
    variant,
    has_contract,
):
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    if variant == "missing":
        model["execution_contracts"].pop("M0")
    elif variant == "partial":
        model["execution_contracts"]["M0"].pop("generation_fingerprint")
    else:
        model["execution_contracts"]["M0"]["structure_fingerprint"] = (
            "sha256:legacy-m0-structure-profile"
        )
    model_path = tmp_path / f"m0-{variant}.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    assert evaluator.standard_m0_has_model_contract is has_contract
    if _execution_fixed_weight_backend_supported():
        assert evaluator.standard_m0_module_ready
        assert evaluator.standard_m0_module_fallback_reason == ""
        assert evaluator.standard_m0_module_id == ("standard-m0-module-module-v1")


@pytest.mark.parametrize(
    "variant,error",
    [
        ("short_weights", "weight row length"),
        ("degree_five", "degree between one and four"),
        ("bad_component", "outside the A0 component range"),
    ],
)
def test_m0_payload_dimensions_are_validated(
    streamed_model_paths,
    tmp_path,
    variant,
    error,
):
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    if variant == "short_weights":
        model["M0_weights"]["0"]["0"]["0"].pop()
    elif variant == "degree_five":
        model["M0_monomials"]["0"][0] = [0, 0, 0, 0, 0]
    else:
        model["M0_monomials"]["0"][0][0] = (model["l_max"] + 1) ** 2
    model_path = tmp_path / f"invalid-m0-{variant}.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    with pytest.raises(ValueError, match=error):
        Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="all_interactions",
        )


def test_m0_payload_accepts_degree_four(
    streamed_model_paths,
    tmp_path,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    monomials = model["M0_monomials"]["0"]
    quartic = [0, 0, 0, 0]
    assert quartic not in monomials
    monomials.append(quartic)
    for type_weights in model["M0_weights"].values():
        for weights in type_weights["0"].values():
            weights.append(0.0)
    model["execution_contracts"]["M0"] = make_standard_m0_contract(
        channels=model["num_channels"],
        type_count=len(model["atomic_numbers"]),
        input_l_max=model["l_max"],
        output_l_max=model["L_max"],
        correlation=4,
        monomials=model["M0_monomials"],
    )
    model_path = tmp_path / "quartic-m0.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="all_interactions",
    )
    assert calculator.evaluator.standard_m0_has_model_contract
    assert not calculator.evaluator.standard_m0_module_ready


def test_execution_standard_m0_repacks_shuffled_payload_rows(
    streamed_model_paths,
    tmp_path,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    for output, monomials in model["M0_monomials"].items():
        order = list(reversed(range(len(monomials))))
        model["M0_monomials"][output] = [monomials[row] for row in order]
        for type_weights in model["M0_weights"].values():
            for channel, weights in type_weights[output].items():
                type_weights[output][channel] = [weights[row] for row in order]
    model_path = tmp_path / "shuffled-m0-rows.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    atoms = _small_structure()
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    evaluator._set_standard_m0_executor("runtime")
    evaluator.compute_node_energies_forces(*native_args)
    runtime = calculator._collect_mace_results(atoms, inputs)
    evaluator._set_standard_m0_executor("standard")
    evaluator.compute_node_energies_forces(*native_args)
    standard = calculator._collect_mace_results(atoms, inputs)

    assert evaluator.standard_m0_module_id == ("standard-m0-module-module-v1")
    assert standard["energy"] == pytest.approx(runtime["energy"], rel=0.0, abs=3e-5)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(standard[name], runtime[name], rtol=0.0, atol=3e-5)


def test_execution_structural_m0_generalizes_species_count(
    macefield_model_path,
    tmp_path,
):
    _require_execution_fixed_weight_backend()

    model = extract_mace_data(
        macefield_model_path,
        species=[7, 13, 14],
        head="mp-dielectric",
        num_spline_points=16,
    )
    model["model_type"] = "MACE"
    model["has_field_coupling"] = False
    model.pop("field_couplings", None)
    model_path = tmp_path / "compact-standard-three-species.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))

    atoms = _small_structure()
    al_index = int(np.flatnonzero(atoms.numbers == 13)[0])
    atoms.numbers[al_index] = 14
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_forward_executor("jit_all")
    evaluator._set_factorized_direct_reverse_executor("jit")
    evaluator._set_standard_r0_executor("v2_edge16")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    assert model["execution_contracts"]["M0"]["type_count"] == 3
    assert evaluator.standard_m0_has_model_contract
    assert evaluator.standard_m0_module_ready
    assert evaluator.standard_m0_module_id == ("standard-m0-module-module-v1")
    assert evaluator.standard_m0_model_structure_fingerprint

    def evaluate(executor):
        evaluator._set_standard_m0_executor(executor)
        evaluator.compute_node_energies_forces(*native_args)
        result = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": result["energy"],
            "energies": np.array(result["energies"], copy=True),
            "forces": np.array(result["forces"], copy=True),
            "stress": np.array(result["stress"], copy=True),
            "M0": np.array(evaluator.M0, copy=True),
            "A0_adj": np.array(evaluator.A0_adj, copy=True),
        }

    runtime = evaluate("runtime")
    standard = evaluate("standard")
    assert standard["energy"] == pytest.approx(runtime["energy"], rel=0.0, abs=3e-5)
    for name in ("energies", "forces", "stress", "M0", "A0_adj"):
        np.testing.assert_allclose(standard[name], runtime[name], rtol=0.0, atol=3e-5)


@pytest.mark.parametrize("streamed_edges", ("factorized", "all_interactions"))
def test_execution_standard_m0_rejects_mismatched_contract_and_falls_back(
    streamed_model_paths,
    tmp_path,
    streamed_edges,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    mismatched = json.loads(standard_path.read_text())
    first_component = mismatched["M0_monomials"]["0"][0][0]
    mismatched["M0_monomials"]["0"][0][0] = 1 if first_component != 1 else 0
    mismatched_path = tmp_path / "mismatched-execution-m0-contract.json"
    mismatched_path.write_text(json.dumps(mismatched, separators=(",", ":")))

    atoms = _small_structure()
    calculator = Symmetrix(
        mismatched_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges=streamed_edges,
    )
    evaluator = calculator.evaluator
    if streamed_edges == "factorized":
        evaluator._set_factorized_source_strategy("jit_plugin")
        evaluator._set_factorized_direct_forward_executor("jit_all")
        evaluator._set_factorized_direct_reverse_executor("jit")
        evaluator._set_standard_r0_executor("v2_edge16")
    evaluator._set_standard_m0_executor("automatic")
    with pytest.raises(ValueError, match="Standard M0 is unavailable"):
        evaluator._set_standard_m0_executor("standard")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])
    launch_counts = (
        evaluator.standard_m0_module_forward_launch_count,
        evaluator.standard_m0_module_reverse_launch_count,
    )

    evaluator.compute_node_energies_forces(*native_args)
    results = calculator._collect_mace_results(atoms, inputs)
    assert np.isfinite(results["energy"])
    assert not evaluator.standard_m0_module_ready
    assert evaluator.standard_m0_module_fallback_reason
    assert evaluator.standard_m0_selected_executor == "runtime"
    assert evaluator.standard_m0_module_forward_launch_count == launch_counts[0]
    assert evaluator.standard_m0_module_reverse_launch_count == launch_counts[1]
    assert (
        0
        < evaluator.standard_m0_poly_values_active_bytes
        <= (evaluator.standard_m0_poly_values_capacity_bytes)
    )
    assert (
        0
        < evaluator.standard_m0_poly_adjoints_active_bytes
        <= (evaluator.standard_m0_poly_adjoints_capacity_bytes)
    )


def test_execution_standard_m0_prepared_steady_state_counters(
    streamed_model_paths,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_forward_executor("jit_all")
    evaluator._set_factorized_direct_reverse_executor("jit")
    evaluator._set_standard_r0_executor("v2_edge16")
    evaluator._set_standard_m0_executor("standard")
    inputs = calculator._mace_inputs(atoms)
    token = evaluator._prepare_factorized_graph(*inputs[:5])
    xyz = np.asarray(inputs[5]).flatten()
    radii = inputs[6]

    def evaluate():
        evaluator._compute_prepared_factorized(token, xyz, radii)
        results = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": results["energy"],
            "energies": np.array(results["energies"], copy=True),
            "forces": np.array(results["forces"], copy=True),
            "stress": np.array(results["stress"], copy=True),
        }

    evaluate()
    counter_names = (
        "standard_m0_module_forward_launch_count",
        "standard_m0_module_reverse_launch_count",
        "factorized_jit_launch_count",
        "factorized_jit_forward_launch_count",
        "factorized_jit_reverse_launch_count",
        "standard_r0_module_launch_count",
        "factorized_prepared_evaluation_count",
        "factorized_fallback_evaluation_count",
        "factorized_topology_validation_count",
        "factorized_topology_validation_skip_count",
        "factorized_preparation_fence_count",
        "factorized_evaluation_fence_count",
    )
    before = {name: int(getattr(evaluator, name)) for name in counter_names}
    schedule_build_count = evaluator.factorized_schedule_build_count
    arena_allocation_count = evaluator.factorized_arena_allocation_count

    repeated = [evaluate() for _ in range(3)]
    after = {name: int(getattr(evaluator, name)) for name in counter_names}
    reverse_launches_per_evaluation = 1 if _kokkos_execution_space() == "Cuda" else 2
    expected_deltas = {
        "standard_m0_module_forward_launch_count": 3,
        "standard_m0_module_reverse_launch_count": 3,
        "factorized_jit_launch_count": (3 + 3 * reverse_launches_per_evaluation),
        "factorized_jit_forward_launch_count": 3,
        "factorized_jit_reverse_launch_count": (3 * reverse_launches_per_evaluation),
        "standard_r0_module_launch_count": 12,
        "factorized_prepared_evaluation_count": 3,
        "factorized_fallback_evaluation_count": 0,
        "factorized_topology_validation_count": 0,
        "factorized_topology_validation_skip_count": 3,
        "factorized_preparation_fence_count": 0,
        "factorized_evaluation_fence_count": 3,
    }
    assert {
        name: after[name] - before[name] for name in counter_names
    } == expected_deltas
    assert evaluator.factorized_schedule_build_count == schedule_build_count
    assert evaluator.factorized_arena_allocation_count == arena_allocation_count
    assert evaluator.standard_m0_module_ready
    assert evaluator.standard_m0_module_fallback_reason == ""
    assert evaluator.standard_m0_module_id
    assert evaluator.standard_m0_model_structure_fingerprint
    assert evaluator.standard_m0_selected_executor == "standard"
    assert evaluator.standard_m0_poly_values_active_bytes == 0
    assert evaluator.standard_m0_poly_adjoints_active_bytes == 0
    assert evaluator.standard_m0_poly_values_active_bytes <= (
        evaluator.standard_m0_poly_values_capacity_bytes
    )
    assert evaluator.standard_m0_poly_adjoints_active_bytes <= (
        evaluator.standard_m0_poly_adjoints_capacity_bytes
    )
    for result in repeated[1:]:
        assert result["energy"] == repeated[0]["energy"]
        for name in ("energies", "forces", "stress"):
            np.testing.assert_allclose(
                result[name], repeated[0][name], rtol=0.0, atol=3e-5
            )


def test_execution_prepared_runtime_path_stays_on_evaluator_stream(
    streamed_model_paths,
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("asynchronous Execution lifecycle counters require Kokkos CUDA")

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    reference_calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    reference_inputs = reference_calculator._mace_inputs(atoms)
    reference_calculator.evaluator.compute_node_energies_forces(
        *reference_inputs[:5],
        np.asarray(reference_inputs[5]).flatten(),
        reference_inputs[6],
    )
    reference = reference_calculator._collect_mace_results(atoms, reference_inputs)

    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("team_cached")
    evaluator._set_factorized_direct_forward_executor("runtime")
    evaluator._set_factorized_direct_reverse_executor("runtime")
    inputs = calculator._mace_inputs(atoms)
    token = evaluator._prepare_factorized_graph(*inputs[:5])
    xyz = np.asarray(inputs[5]).flatten()
    radii = inputs[6]

    def evaluate():
        evaluator._compute_prepared_factorized(token, xyz, radii)
        return calculator._collect_mace_results(atoms, inputs)

    evaluate()
    assert evaluator.factorized_selected_direct_forward_executor == "runtime"
    assert evaluator.factorized_selected_direct_reverse_executor == "runtime"
    assert evaluator.factorized_execution_profile == "factorized_state"

    before = {
        "async_sphericart": evaluator.execution_sphericart_async_launch_count,
        "sphericart": evaluator.execution_sphericart_launch_count,
        "stage_fences": evaluator.factorized_stage_fence_count,
        "evaluation_fences": evaluator.factorized_evaluation_fence_count,
        "preparation_fences": evaluator.factorized_preparation_fence_count,
    }
    schedule_build_count = evaluator.factorized_schedule_build_count
    arena_allocation_count = evaluator.factorized_arena_allocation_count
    geometry_allocation_count = evaluator.execution_geometry_allocation_count

    fallback = evaluate()

    assert (
        evaluator.execution_sphericart_async_launch_count - before["async_sphericart"]
    ) == 1
    assert evaluator.execution_sphericart_launch_count - before["sphericart"] == 1
    assert evaluator.factorized_stage_fence_count - before["stage_fences"] == 0
    assert (
        evaluator.factorized_evaluation_fence_count - before["evaluation_fences"]
    ) == 1
    assert (
        evaluator.factorized_preparation_fence_count - before["preparation_fences"]
    ) == 0
    assert evaluator.factorized_schedule_build_count == schedule_build_count
    assert evaluator.factorized_arena_allocation_count == arena_allocation_count
    assert evaluator.execution_geometry_allocation_count == geometry_allocation_count

    assert fallback["energy"] == pytest.approx(reference["energy"], rel=0.0, abs=3e-5)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(fallback[name], reference[name], rtol=0.0, atol=3e-5)


def test_generic_fp32_source_adjoint_workspace_is_reused(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    evaluator = calculator.evaluator

    calculator.calculate(atoms, properties=["energy", "forces", "stress"])
    expected_bytes = np.dtype(np.float64).itemsize * len(evaluator.H1)
    assert evaluator.generic_phi1_source_adjoint_workspace_bytes == expected_bytes
    assert evaluator.generic_phi1_source_adjoint_allocation_count == 1
    first = {
        name: np.array(calculator.results[name], copy=True)
        for name in ("energy", "forces", "stress")
    }

    calculator.calculate(atoms, properties=["energy", "forces", "stress"])
    assert evaluator.generic_phi1_source_adjoint_workspace_bytes == expected_bytes
    assert evaluator.generic_phi1_source_adjoint_allocation_count == 1
    for name in first:
        np.testing.assert_array_equal(calculator.results[name], first[name])


def test_execution_standard_r0_v2_state_clears_on_mode_switch(
    streamed_model_paths,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_reverse_executor("jit")
    evaluator._set_standard_r0_executor("v2_edge16")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    def evaluate():
        evaluator.compute_node_energies_forces(*native_args)
        results = calculator._collect_mace_results(atoms, inputs)
        return results["energy"], np.array(results["forces"], copy=True)

    evaluate()
    generated_launches = evaluator.standard_r0_module_launch_count
    assert evaluator.standard_r0_selected_executor == "v2_edge16"

    for mode in ("materialized",):
        fresh_calculator = Symmetrix(
            standard_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges=mode,
        )
        fresh_evaluator = fresh_calculator.evaluator
        fresh_inputs = fresh_calculator._mace_inputs(atoms)
        fresh_evaluator.compute_node_energies_forces(
            *fresh_inputs[:5], np.asarray(fresh_inputs[5]).flatten(), fresh_inputs[6]
        )
        fresh_results = fresh_calculator._collect_mace_results(atoms, fresh_inputs)

        evaluator.set_streamed_edges(mode)
        fallback_energy, fallback_forces = evaluate()
        assert not evaluator.standard_r0_density_scale_fused
        assert evaluator.standard_r0_module_launch_count == generated_launches
        assert fallback_energy == pytest.approx(
            fresh_results["energy"], rel=0.0, abs=3e-5
        )
        np.testing.assert_allclose(
            fallback_forces, fresh_results["forces"], rtol=0.0, atol=3e-5
        )


def test_factorized_contract_gate_and_legacy_compatibility(
    streamed_model_paths,
    tmp_path,
):
    standard_path, _ = streamed_model_paths
    standard = json.loads(standard_path.read_text())
    legacy = copy.deepcopy(standard)
    legacy.pop("execution_contracts")
    legacy_path = tmp_path / "legacy-without-execution-contract.json"
    legacy_path.write_text(json.dumps(legacy, separators=(",", ":")))

    mismatched = copy.deepcopy(standard)
    mismatched["execution_contracts"]["R1"]["generation_fingerprint"] = "sha256:0"
    mismatched_path = tmp_path / "mismatched-execution-contract.json"
    mismatched_path.write_text(json.dumps(mismatched, separators=(",", ":")))

    r0_mismatched = copy.deepcopy(standard)
    r0_mismatched["execution_contracts"]["R0"]["generation_fingerprint"] = "sha256:0"
    r0_mismatched_path = tmp_path / "mismatched-execution-r0-contract.json"
    r0_mismatched_path.write_text(json.dumps(r0_mismatched, separators=(",", ":")))

    with pytest.raises(RuntimeError, match="does not contain a Execution R1 contract"):
        Symmetrix(
            legacy_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="direct",
        )
    legacy_calculator = Symmetrix(
        legacy_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="generic",
    )
    assert legacy_calculator.evaluator.m0_implementation == "builtin"
    assert legacy_calculator.evaluator.m0_supports_low_memory
    atoms = _small_structure()
    atoms.calc = legacy_calculator
    atoms.get_potential_energy()
    assert not legacy_calculator.evaluator.factorized_has_model_contract
    assert legacy_calculator.evaluator.factorized_model_contract_fingerprint == ""
    assert not legacy_calculator.evaluator.standard_r0_has_model_contract
    assert not legacy_calculator.evaluator.standard_r0_module_ready
    assert "does not contain a Execution R0 contract" in (
        legacy_calculator.evaluator.standard_r0_module_fallback_reason
    )

    with pytest.raises(RuntimeError, match="generation fingerprint"):
        Symmetrix(
            mismatched_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="direct",
        )

    atoms = _small_structure()
    r0_mismatched_calculator = Symmetrix(
        r0_mismatched_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
    )
    r0_mismatched_calculator.evaluator._set_factorized_source_strategy("jit_plugin")
    atoms.calc = r0_mismatched_calculator
    assert np.isfinite(atoms.get_potential_energy())
    assert r0_mismatched_calculator.evaluator.standard_r0_has_model_contract
    if _execution_fixed_weight_backend_supported():
        assert r0_mismatched_calculator.evaluator.standard_r0_module_ready
        assert r0_mismatched_calculator.evaluator.standard_r0_module_launch_count == 4
        assert r0_mismatched_calculator.evaluator.factorized_jit_ready
    else:
        assert r0_mismatched_calculator.evaluator.standard_r0_module_launch_count == 0


@pytest.mark.parametrize(
    ("payload_case", "reason_fragment"),
    [
        ("stale_contract", "validated compact-v2 R1 contract"),
        ("malformed_contract", "sparse_coupling.terms does not match term_count"),
    ],
)
def test_factorized_exact_artifact_is_bound_to_loaded_phi_payload(
    streamed_model_paths,
    tmp_path,
    payload_case,
    reason_fragment,
):
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    changed_coefficient = model["Phi1_clebsch_gordan"][0] * 0.5
    if payload_case == "malformed_contract":
        model["execution_contracts"]["R1"]["sparse_coupling"].pop("terms")
    else:
        model["Phi1_clebsch_gordan"][0] = changed_coefficient
        if payload_case == "forged_contract":
            model["execution_contracts"]["R1"]["sparse_coupling"]["terms"][0][
                "coefficient"
            ] = changed_coefficient

    model_path = tmp_path / f"execution-r1-{payload_case}.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    atoms = _small_structure()

    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="all_interactions",
    )
    reference_inputs = reference._mace_inputs(atoms)
    reference.evaluator.compute_node_energies_forces(
        *reference_inputs[:5],
        np.asarray(reference_inputs[5]).flatten(),
        reference_inputs[6],
    )
    reference_result = reference._collect_mace_results(atoms, reference_inputs)

    assert np.isfinite(reference_result["energy"])
    with pytest.raises(RuntimeError, match=reason_fragment):
        Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="direct",
        )


def test_factorized_observer_matches_reference_across_execution_strategies(
    streamed_model_paths,
):
    _require_receiver_factorized_host_backend()
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="receiver_factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])
    evaluator._set_factorized_source_strategy("serial_reference")
    evaluator.compute_node_energies_forces(*native_args)
    core_workspace_bytes = (
        evaluator.factorized_workspace_bytes
        - evaluator.factorized_coupling_workspace_bytes
    )
    evaluator._set_factorized_observer(True)

    def observe(strategy):
        evaluator._set_factorized_source_strategy(strategy)
        evaluator.compute_node_energies_forces(*native_args)
        report = evaluator._factorized_observer()
        assert report["enabled"]
        assert report["ready"]
        assert report["num_nodes"] == len(atoms)
        assert report["num_edges"] == len(inputs[3])
        assert 0 < report["bytes"] <= report["max_bytes"]
        assert (
            evaluator.factorized_workspace_bytes
            - evaluator.factorized_coupling_workspace_bytes
            == core_workspace_bytes
        )
        assert evaluator.factorized_workspace_bytes <= (
            evaluator.factorized_workspace_capacity_bytes
        )
        groups = []
        for group in report["groups"]:
            shape = tuple(group["shape"])
            groups.append(
                (
                    np.asarray(group["state"]).reshape(shape),
                    np.asarray(group["state_adjoint"]).reshape(shape),
                )
            )
        observation = {
            "groups": groups,
            "output": np.asarray(report["output"]).reshape(
                tuple(report["output_shape"])
            ),
            "output_adjoint": np.asarray(report["output_adjoint"]).reshape(
                tuple(report["output_shape"])
            ),
            "r0_output": np.asarray(report["r0_output"]).reshape(
                tuple(report["r0_output_shape"])
            ),
            "r0_output_adjoint": np.asarray(report["r0_output_adjoint"]).reshape(
                tuple(report["r0_output_shape"])
            ),
            "source_h1_delta": np.asarray(report["source_h1_delta"]).reshape(
                tuple(report["source_h1_delta_shape"])
            ),
            "directed_edge_force_delta": np.asarray(
                report["directed_edge_force_delta"]
            ).reshape(tuple(report["directed_edge_force_delta_shape"])),
            "r0_directed_edge_force_delta": np.asarray(
                report["r0_directed_edge_force_delta"]
            ).reshape(tuple(report["r0_directed_edge_force_delta_shape"])),
        }
        output_inner_product = np.dot(
            observation["output"].astype(np.float64, copy=False).ravel(),
            observation["output_adjoint"].astype(np.float64, copy=False).ravel(),
        )
        state_inner_product = sum(
            np.dot(
                state.astype(np.float64, copy=False).ravel(),
                state_adjoint.astype(np.float64, copy=False).ravel(),
            )
            for state, state_adjoint in groups
        )
        assert state_inner_product == pytest.approx(
            output_inner_product, rel=2e-5, abs=2e-4
        )
        return observation

    reference = observe("serial_reference")
    for strategy in ("tiled_coupling", "jit_plugin"):
        candidate = observe(strategy)
        assert len(candidate["groups"]) == len(reference["groups"])
        for (state, state_adjoint), (ref_state, ref_state_adjoint) in zip(
            candidate["groups"], reference["groups"]
        ):
            np.testing.assert_allclose(state, ref_state, rtol=0.0, atol=3e-5)
            np.testing.assert_allclose(
                state_adjoint, ref_state_adjoint, rtol=0.0, atol=3e-5
            )
        for name in (
            "output",
            "output_adjoint",
            "r0_output",
            "r0_output_adjoint",
            "source_h1_delta",
            "directed_edge_force_delta",
            "r0_directed_edge_force_delta",
        ):
            np.testing.assert_allclose(
                candidate[name], reference[name], rtol=0.0, atol=3e-5
            )

    assert np.any(reference["source_h1_delta"] != 0)
    assert np.any(reference["directed_edge_force_delta"] != 0)
    assert np.any(reference["r0_directed_edge_force_delta"] != 0)
    evaluator._set_factorized_observer(False)
    assert not evaluator.factorized_observer_enabled
    assert not evaluator.factorized_observer_ready
    assert evaluator.factorized_observer_bytes == 0
    disabled = evaluator._factorized_observer()
    assert not disabled["enabled"]
    assert not disabled["ready"]
    assert all(np.asarray(group["state"]).size == 0 for group in disabled["groups"])
    evaluator.compute_node_energies_forces(*native_args)
    assert not evaluator.factorized_observer_ready
    assert evaluator.factorized_observer_bytes == 0


def test_factorized_observer_rejects_capture_above_host_byte_limit(
    streamed_model_paths,
):
    _require_receiver_factorized_host_backend()
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="receiver_factorized",
    )
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])
    evaluator = calculator.evaluator
    evaluator.set_streamed_edges("receiver_factorized")
    evaluator._set_factorized_observer(True, max_bytes=1)

    with pytest.raises(ValueError, match="observer requires.*exceeding"):
        evaluator.compute_node_energies_forces(*native_args)

    assert evaluator.factorized_observer_bytes == 0
    assert not evaluator.factorized_observer_ready
    evaluator._set_factorized_observer(False)


def test_factorized_source_schedule_reuses_and_rebuilds_with_topology(
    streamed_model_paths,
):
    standard_path, _ = streamed_model_paths
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float64",
        streamed_edges="factorized",
    )

    def evaluate(atoms):
        inputs = calculator._mace_inputs(atoms)
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, _ = inputs
        calculator.evaluator.compute_node_energies_forces(
            num_nodes,
            node_types,
            num_neigh,
            j_list,
            neigh_types,
            xyz.flatten(),
            r,
        )
        return len(j_list)

    atoms = _small_structure()
    initial_edges = evaluate(atoms)
    evaluator = calculator.evaluator
    assert evaluator.factorized_schedule_build_count == 1
    assert evaluator.factorized_schedule_entries == initial_edges
    initial_schedule_bytes = evaluator.factorized_schedule_bytes
    initial_workspace_bytes = evaluator.factorized_workspace_bytes
    initial_arena_allocations = evaluator.factorized_arena_allocation_count

    evaluate(atoms)
    assert evaluator.factorized_schedule_build_count == 1
    assert evaluator.factorized_schedule_bytes == initial_schedule_bytes
    assert evaluator.factorized_workspace_bytes == initial_workspace_bytes
    assert evaluator.factorized_arena_allocation_count == initial_arena_allocations

    perturbed_atoms = atoms.copy()
    perturbed_atoms.positions[0] += [1.0e-4, -2.0e-4, 3.0e-4]
    assert evaluate(perturbed_atoms) == initial_edges
    assert evaluator.factorized_schedule_build_count == 1

    ragged_atoms = _ragged_structure()
    ragged_edges = evaluate(ragged_atoms)
    assert ragged_edges < initial_edges
    assert evaluator.factorized_schedule_build_count == 2
    assert evaluator.factorized_schedule_entries == ragged_edges
    assert evaluator.factorized_arena_allocation_count == initial_arena_allocations

    evaluate(ragged_atoms)
    assert evaluator.factorized_schedule_build_count == 2

    assert evaluate(atoms) == initial_edges
    assert evaluator.factorized_schedule_build_count == 3
    assert evaluator.factorized_schedule_entries == initial_edges
    assert evaluator.factorized_arena_allocation_count == initial_arena_allocations


def test_factorized_prepared_graph_token_lifecycle(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_reverse_executor("jit")
    evaluator._set_standard_r0_executor("v2_edge16")
    inputs = calculator._mace_inputs(atoms)

    token = evaluator._prepare_factorized_graph(*inputs[:5])
    assert token > 0
    assert evaluator.factorized_graph_generation == token
    assert evaluator.factorized_prepared_graph_count == 1
    assert evaluator.factorized_schedule_build_count == 1
    assert evaluator.factorized_topology_validation_count == 0
    assert evaluator.factorized_preparation_fence_count == 1
    initial_geometry_allocations = evaluator.execution_geometry_allocation_count
    initial_geometry_capacity = evaluator.execution_geometry_capacity_edges
    assert initial_geometry_allocations == 1
    assert initial_geometry_capacity >= len(inputs[6])
    assert evaluator.execution_geometry_workspace_bytes > 0
    assert evaluator.execution_geometry_copy_count == 0
    assert evaluator.execution_sphericart_initialization_count == 1
    assert evaluator.execution_sphericart_launch_count == 0

    evaluator._compute_prepared_factorized(
        token, np.asarray(inputs[5]).flatten(), inputs[6]
    )
    reference = calculator._collect_mace_results(atoms, inputs)
    assert evaluator.factorized_prepared_evaluation_count == 1
    assert evaluator.factorized_fallback_evaluation_count == 0
    assert evaluator.factorized_topology_validation_count == 0
    assert evaluator.factorized_topology_validation_skip_count == 1
    assert evaluator.factorized_evaluation_fence_count == 1
    assert evaluator.execution_geometry_allocation_count == initial_geometry_allocations
    assert evaluator.execution_geometry_copy_count == 2
    assert evaluator.execution_sphericart_initialization_count == 1
    assert evaluator.execution_sphericart_launch_count == 1
    assert evaluator.R0_storage_size == 0
    assert evaluator.A0_scale_storage_size == 0

    perturbed_atoms = atoms.copy()
    perturbed_atoms.positions[0] += [1.0e-4, -2.0e-4, 3.0e-4]
    perturbed_inputs = calculator._mace_inputs(perturbed_atoms)
    for original, perturbed in zip(inputs[1:5], perturbed_inputs[1:5]):
        np.testing.assert_array_equal(original, perturbed)
    evaluator._compute_prepared_factorized(
        token,
        np.asarray(perturbed_inputs[5]).flatten(),
        perturbed_inputs[6],
    )
    perturbed = calculator._collect_mace_results(perturbed_atoms, perturbed_inputs)
    assert np.isfinite(reference["energy"])
    assert np.isfinite(perturbed["energy"])
    assert evaluator.factorized_prepared_evaluation_count == 2
    assert evaluator.factorized_topology_validation_skip_count == 2
    assert evaluator.factorized_schedule_build_count == 1
    assert evaluator.factorized_evaluation_fence_count == 2
    assert evaluator.execution_geometry_allocation_count == initial_geometry_allocations
    assert evaluator.execution_geometry_capacity_edges == initial_geometry_capacity
    assert evaluator.execution_geometry_copy_count == 4
    assert evaluator.execution_sphericart_initialization_count == 1
    assert evaluator.execution_sphericart_launch_count == 2
    if native_symmetrix._kokkos_default_execution_space() == "Cuda":
        assert evaluator.execution_sphericart_async_launch_count == 2

    same_token = evaluator._prepare_factorized_graph(*inputs[:5])
    assert same_token == token
    assert evaluator.factorized_prepared_graph_count == 1
    assert evaluator.factorized_schedule_build_count == 1

    if _execution_fixed_weight_backend_supported():
        assert evaluator.standard_r0_module_ready
        assert evaluator.standard_r0_module_id == "standard-r0-module-module-v2"
        assert evaluator.standard_r0_selected_executor == "v2_edge16"
        assert evaluator.factorized_jit_ready
        assert evaluator.factorized_jit_artifact_id.startswith("jit-r1-gen11-")
        assert evaluator.factorized_jit_contract_fingerprint.startswith("sha256:")
        assert evaluator.factorized_selected_direct_forward_executor == ("jit_all")
        assert evaluator.factorized_selected_direct_reverse_executor == "jit"
        assert evaluator.standard_r0_has_model_contract
        assert evaluator.standard_r0_module_launch_count == 8
        assert evaluator.standard_r0_density_scale_fused
        assert evaluator.standard_r0_workspace_bytes == 8 * len(atoms)
        assert (
            evaluator.execution_unified_workspace_bytes
            == evaluator.factorized_workspace_bytes
            + evaluator.standard_r0_workspace_bytes
        )
        assert evaluator.factorized_execution_profile == "direct_fixed_weight"
        assert evaluator.factorized_execution_strategy == "direct_jit_reverse"
        reverse_launches_per_evaluation = (
            1 if _kokkos_execution_space() == "Cuda" else 2
        )
        assert evaluator.factorized_jit_launch_count == (
            2 + 2 * reverse_launches_per_evaluation
        )
        assert evaluator.factorized_jit_forward_launch_count == 2
        assert evaluator.factorized_jit_reverse_launch_count == (
            2 * reverse_launches_per_evaluation
        )
        assert evaluator.execution_phi1r_active_bytes == 0
        assert evaluator.execution_phi1r_capacity_bytes == 0
        assert evaluator.execution_dphi1r_active_bytes == 0
        assert evaluator.execution_dphi1r_capacity_bytes == 0
        assert evaluator.factorized_custom_blas_launch_count == 0
        assert evaluator.factorized_blas_stream_bind_count == 0

    evaluator._set_factorized_planner_budget_bytes(evaluator.factorized_workspace_bytes)
    assert evaluator.factorized_graph_generation == 0
    with pytest.raises(ValueError, match="graph token is stale"):
        evaluator._compute_prepared_factorized(
            token, np.asarray(inputs[5]).flatten(), inputs[6]
        )
    refreshed_token = evaluator._prepare_factorized_graph(*inputs[:5])
    assert refreshed_token > token
    assert evaluator.factorized_schedule_build_count == 2

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
        np.asarray(inputs[5])[permutation],
        np.asarray(inputs[6])[permutation],
        np.asarray(inputs[7])[permutation],
    )
    reordered_token = evaluator._prepare_factorized_graph(*reordered[:5])
    assert reordered_token > refreshed_token
    assert evaluator.factorized_schedule_build_count == 3
    with pytest.raises(ValueError, match="graph token is stale"):
        evaluator._compute_prepared_factorized(
            refreshed_token,
            np.asarray(reordered[5]).flatten(),
            reordered[6],
        )

    insertion = int(inputs[2][0])
    repeated_num_neigh = np.array(inputs[2], copy=True)
    repeated_num_neigh[0] += 1
    repeated = (
        inputs[0],
        inputs[1],
        repeated_num_neigh,
        np.insert(np.asarray(inputs[3]), insertion, inputs[3][0]),
        np.insert(np.asarray(inputs[4]), insertion, inputs[4][0]),
        np.insert(np.asarray(inputs[5]), insertion, inputs[5][0], axis=0),
        np.insert(np.asarray(inputs[6]), insertion, inputs[6][0]),
        np.insert(np.asarray(inputs[7]), insertion, inputs[7][0]),
    )
    repeated_token = evaluator._prepare_factorized_graph(*repeated[:5])
    assert repeated_token > reordered_token
    assert evaluator.factorized_schedule_build_count == 4
    assert (
        evaluator.execution_geometry_allocation_count
        == initial_geometry_allocations + 1
    )
    assert evaluator.execution_geometry_capacity_edges >= len(repeated[6])
    evaluator._compute_prepared_factorized(
        repeated_token, np.asarray(repeated[5]).flatten(), repeated[6]
    )
    assert evaluator.factorized_prepared_evaluation_count == 3
    assert evaluator.factorized_topology_validation_count == 0
    assert evaluator.execution_sphericart_initialization_count == 1

    evaluator.set_streamed_edges("receiver_factorized")
    evaluator._set_factorized_observer(True, max_bytes=1)
    observer_token = evaluator._prepare_factorized_graph(*repeated[:5])
    assert observer_token > repeated_token
    fence_count = evaluator.factorized_evaluation_fence_count
    with pytest.raises(ValueError, match="host bytes"):
        evaluator._compute_prepared_factorized(
            observer_token, np.asarray(repeated[5]).flatten(), repeated[6]
        )
    assert evaluator.factorized_evaluation_fence_count == fence_count + 1
    evaluator._set_factorized_observer(False)
    evaluator.set_streamed_edges("direct")
    disabled_observer_token = evaluator._prepare_factorized_graph(*repeated[:5])
    assert disabled_observer_token > observer_token
    evaluator._compute_prepared_factorized(
        disabled_observer_token, np.asarray(repeated[5]).flatten(), repeated[6]
    )
    assert evaluator.factorized_evaluation_fence_count == fence_count + 2


def test_factorized_jit_position_geometry_matches_exact_graph(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc(True)
    native = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
        neighbor_skin=0.5,
    )
    exact = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
        neighbor_skin=0.0,
    )

    atoms.calc = native
    native_forces = atoms.get_forces()
    native_energy = atoms.get_potential_energy()
    assert native.evaluator.execution_prepared_geometry_update_count == 1
    assert native.evaluator.execution_geometry_copy_count == 1
    assert native.evaluator.factorized_fractional_geometry_preparation_count == 1
    assert native.evaluator.factorized_cell_update_count == 0
    assert native.neighbor_cache_host_geometry_materialization_count == 0
    initial_geometry_state_allocations = (
        native.evaluator.factorized_geometry_state_allocation_count
    )

    reference_atoms = atoms.copy()
    reference_atoms.calc = exact
    np.testing.assert_allclose(
        native_forces, reference_atoms.get_forces(), rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(
        native_energy, reference_atoms.get_potential_energy(), rtol=2.0e-5, atol=2.0e-5
    )

    atoms.positions[0] += [20.0, 0.0, 0.0]
    wrapped_forces = atoms.get_forces()
    wrapped_energy = atoms.get_potential_energy()
    wrapped_reference = atoms.copy()
    wrapped_reference.calc = exact
    np.testing.assert_allclose(
        wrapped_forces,
        wrapped_reference.get_forces(),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        wrapped_energy,
        wrapped_reference.get_potential_energy(),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert native.neighbor_cache_build_count == 1
    assert native.neighbor_cache_reuse_count == 1
    assert native.evaluator.execution_prepared_geometry_update_count == 2
    assert native.evaluator.execution_geometry_copy_count == 2
    assert native.evaluator.factorized_fractional_geometry_preparation_count == 1
    assert native.evaluator.factorized_cell_update_count == 0

    atoms.set_cell([19.9, 20.0, 20.0], scale_atoms=True)
    deformed_forces = atoms.get_forces()
    deformed_energy = atoms.get_potential_energy()
    deformed_reference = atoms.copy()
    deformed_reference.calc = exact
    np.testing.assert_allclose(
        deformed_forces,
        deformed_reference.get_forces(),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        deformed_energy,
        deformed_reference.get_potential_energy(),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert native.neighbor_cache_build_count == 1
    assert native.evaluator.execution_prepared_geometry_update_count == 3
    assert native.evaluator.execution_geometry_copy_count == 3
    assert native.evaluator.factorized_fractional_geometry_preparation_count == 1
    assert native.evaluator.factorized_cell_update_count == 1
    assert (
        native.evaluator.factorized_geometry_state_allocation_count
        == initial_geometry_state_allocations
    )
    assert native.neighbor_cache_host_geometry_materialization_count == 0


@pytest.mark.parametrize("dtype, atol", (("float32", 2.0e-5), ("float64", 3.0e-11)))
@pytest.mark.parametrize("low_memory", (False, True))
def test_direct_skin_mask_matches_exact_graph_across_cutoff(
    streamed_model_paths,
    monkeypatch,
    tmp_path,
    dtype,
    atol,
    low_memory,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    native = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=low_memory,
        neighbor_skin=0.5,
    )
    exact = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=low_memory,
        neighbor_skin=0.0,
    )
    if low_memory and native_symmetrix._kokkos_default_execution_space() in (
        "Cuda",
        "HIP",
    ):
        device_bytes = 32 * 1024**3
        native.evaluator._set_low_memory_device_memory_info_for_testing(
            512 * 1024**2, device_bytes
        )
        exact.evaluator._set_low_memory_device_memory_info_for_testing(
            512 * 1024**2, device_bytes
        )
    atoms = Atoms(
        ["Al", "N", "Al"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.2, 0.1],
            [native.cutoff - 0.05, 0.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )

    def compare_with_exact():
        native.calculate(atoms, properties=["energy", "forces", "stress"])
        reference_atoms = atoms.copy()
        exact.calculate(reference_atoms, properties=["energy", "forces", "stress"])
        for property_name in ("energy", "forces", "stress"):
            np.testing.assert_allclose(
                native.results[property_name],
                exact.results[property_name],
                rtol=atol,
                atol=atol,
            )

    compare_with_exact()
    candidate_schedule_bytes = native.evaluator.factorized_schedule_bytes
    candidate_schedule_builds = native.evaluator.factorized_schedule_build_count

    atoms.positions[2, 0] += 0.1
    compare_with_exact()

    assert native.evaluator.factorized_schedule_bytes == candidate_schedule_bytes
    assert native.evaluator.factorized_schedule_build_count == candidate_schedule_builds
    assert native.neighbor_cache_build_count == 1
    assert native.neighbor_cache_reuse_count == 1
    assert native.evaluator.factorized_prepared_evaluation_count == 2
    assert native.evaluator.factorized_selected_direct_forward_executor == "jit_all"
    assert native.evaluator.factorized_selected_direct_reverse_executor == "jit"
    if low_memory and native_symmetrix._kokkos_default_execution_space() in (
        "Cuda",
        "HIP",
    ):
        assert native.evaluator.harmonic_storage_policy == "y-only-direct-v1"
        assert native.evaluator.harmonic_gradient_bytes == 0


@pytest.mark.parametrize(
    "dtype, first_order_atol, response_atol",
    (("float32", 5.0e-5, 5.0e-4), ("float64", 2.0e-9, 2.0e-6)),
)
@pytest.mark.parametrize("low_memory", (False, True))
def test_macefield_direct_skin_mask_matches_exact_graph_across_cutoff(
    streamed_model_paths,
    monkeypatch,
    tmp_path,
    dtype,
    first_order_atol,
    response_atol,
    low_memory,
):
    _require_execution_fixed_weight_backend()

    _, field_path = streamed_model_paths
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    electric_field = np.array([0.01, -0.02, 0.03])

    def make_calculator(neighbor_skin):
        calculator = Symmetrix(
            field_path,
            use_kokkos=True,
            dtype=dtype,
            streamed_edges="direct",
            low_memory=low_memory,
            neighbor_skin=neighbor_skin,
            electric_field=electric_field,
        )
        if low_memory and native_symmetrix._kokkos_default_execution_space() in (
            "Cuda",
            "HIP",
        ):
            calculator.evaluator._set_low_memory_device_memory_info_for_testing(
                512 * 1024**2, 32 * 1024**3
            )
        return calculator

    candidate = make_calculator(0.5)
    exact = make_calculator(0.0)
    atoms = Atoms(
        ["Al", "N", "Al"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.2, 0.1],
            [candidate.cutoff - 0.05, 0.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    properties = (
        "energy",
        "forces",
        "stress",
        "polarization",
        "polarizability",
        "becs",
    )

    def compare_with_exact():
        candidate.calculate(atoms, properties=properties)
        exact.calculate(atoms.copy(), properties=properties)
        for property_name in ("energy", "forces", "stress", "polarization"):
            np.testing.assert_allclose(
                candidate.results[property_name],
                exact.results[property_name],
                rtol=0.0,
                atol=first_order_atol,
            )
        for property_name in ("polarizability", "becs"):
            np.testing.assert_allclose(
                candidate.results[property_name],
                exact.results[property_name],
                rtol=0.0,
                atol=response_atol,
            )

    compare_with_exact()
    schedule_bytes = candidate.evaluator.factorized_schedule_bytes
    schedule_builds = candidate.evaluator.factorized_schedule_build_count

    atoms.positions[2, 0] += 0.1
    compare_with_exact()

    assert candidate.evaluator.factorized_schedule_bytes == schedule_bytes
    assert candidate.evaluator.factorized_schedule_build_count == schedule_builds
    assert candidate.neighbor_cache_build_count == 1
    assert candidate.neighbor_cache_reuse_count == 1
    assert candidate.evaluator.factorized_fallback_evaluation_count == 0
    assert candidate.evaluator.factorized_selected_direct_forward_executor == "jit_all"
    assert candidate.evaluator.factorized_selected_direct_reverse_executor == "jit"
    if low_memory and native_symmetrix._kokkos_default_execution_space() in (
        "Cuda",
        "HIP",
    ):
        assert candidate.evaluator.harmonic_storage_policy == "y-only-direct-v1"
        assert candidate.evaluator.harmonic_gradient_bytes == 0


def test_all_interactions_native_positions_compact_exact_cutoff_edges(
    streamed_model_paths,
):
    standard_path, _ = streamed_model_paths
    native = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="all_interactions",
        neighbor_skin=0.5,
    )
    exact = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="all_interactions",
        neighbor_skin=0.0,
    )
    atoms = Atoms(
        ["Al", "N", "Al"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.2, 0.1],
            [native.cutoff - 0.05, 0.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )

    def compare_with_exact():
        atoms.calc = native
        native_forces = atoms.get_forces()
        native_energy = atoms.get_potential_energy()
        reference_atoms = atoms.copy()
        reference_atoms.calc = exact
        reference_forces = reference_atoms.get_forces()
        reference_energy = reference_atoms.get_potential_energy()
        np.testing.assert_allclose(
            native_forces, reference_forces, rtol=2.0e-5, atol=2.0e-5
        )
        np.testing.assert_allclose(
            native_energy, reference_energy, rtol=2.0e-5, atol=2.0e-5
        )
        return len(exact._mace_inputs(reference_atoms)[3])

    initial_edges = compare_with_exact()
    assert native.evaluator.all_interactions_active_edge_count == initial_edges

    atoms.positions[2, 0] += 0.1
    crossed_edges = compare_with_exact()
    assert crossed_edges == initial_edges - 2
    assert native.evaluator.all_interactions_active_edge_count == crossed_edges
    assert native.neighbor_cache_build_count == 1
    assert native.neighbor_cache_reuse_count == 1
    assert native.evaluator.all_interactions_prepared_graph_count == 1
    assert native.evaluator.all_interactions_prepared_evaluation_count == 2
    assert native.evaluator.all_interactions_geometry_update_count == 2
    assert native.evaluator.all_interactions_schedule_build_count == 2


def test_factorized_operator_session_tracks_successful_evaluation_epoch(
    streamed_model_paths,
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("generated Execution R1 operator sessions require Kokkos CUDA")

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_forward_executor("jit_all")
    evaluator._set_factorized_direct_reverse_executor("jit")

    inputs = calculator._mace_inputs(atoms)
    graph_token = evaluator._prepare_factorized_graph(*inputs[:5])
    xyz = np.ascontiguousarray(inputs[5], dtype=np.float64).reshape(-1)
    radii = np.ascontiguousarray(inputs[6], dtype=np.float64)

    with pytest.raises(
        RuntimeError, match="successfully completed prepared evaluation"
    ):
        evaluator._prepare_factorized_operator_benchmark(graph_token, xyz, radii)

    evaluator._compute_prepared_factorized(graph_token, xyz, radii)
    session = evaluator._prepare_factorized_operator_benchmark(graph_token, xyz, radii)
    evaluator._run_factorized_operator_benchmark(session, "forward", True)

    perturbed_atoms = atoms.copy()
    perturbed_atoms.positions[0] += [1.0e-4, -2.0e-4, 3.0e-4]
    perturbed_inputs = calculator._mace_inputs(perturbed_atoms)
    for original, perturbed in zip(inputs[1:5], perturbed_inputs[1:5]):
        np.testing.assert_array_equal(original, perturbed)
    perturbed_xyz = np.ascontiguousarray(perturbed_inputs[5], dtype=np.float64).reshape(
        -1
    )
    perturbed_radii = np.ascontiguousarray(perturbed_inputs[6], dtype=np.float64)
    evaluator._compute_prepared_factorized(graph_token, perturbed_xyz, perturbed_radii)
    with pytest.raises(ValueError, match="benchmark token is stale"):
        evaluator._run_factorized_operator_benchmark(session, "forward", True)

    replacement_session = evaluator._prepare_factorized_operator_benchmark(
        graph_token, perturbed_xyz, perturbed_radii
    )
    with pytest.raises(ValueError, match="coordinates do not match"):
        evaluator._compute_prepared_factorized(
            graph_token, perturbed_xyz[:-3], perturbed_radii
        )
    with pytest.raises(ValueError, match="benchmark token is stale"):
        evaluator._run_factorized_operator_benchmark(
            replacement_session, "forward", True
        )
    with pytest.raises(
        RuntimeError, match="successfully completed prepared evaluation"
    ):
        evaluator._prepare_factorized_operator_benchmark(
            graph_token, perturbed_xyz, perturbed_radii
        )


def test_factorized_matches_materialized_on_ragged_graph_with_isolated_atom(
    streamed_model_paths,
):
    standard_path, _ = streamed_model_paths
    outputs = {}
    for mode in ("materialized", "factorized"):
        atoms = _ragged_structure()
        calculator = Symmetrix(
            standard_path,
            use_kokkos=True,
            dtype="float64",
            streamed_edges=mode,
        )
        if mode == "factorized":
            calculator.evaluator._set_factorized_source_strategy("tiled_coupling")
        atoms.calc = calculator
        outputs[mode] = (atoms.get_potential_energy(), atoms.get_forces())

    energy, forces = outputs["factorized"]
    reference_energy, reference_forces = outputs["materialized"]
    assert energy == pytest.approx(reference_energy, rel=0.0, abs=3e-11)
    np.testing.assert_allclose(forces, reference_forces, rtol=0.0, atol=3e-11)


def test_execution_native_receiver_matches_tiled_on_ragged_graph(
    streamed_model_paths,
):
    standard_path, _ = streamed_model_paths
    atoms = _ragged_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    evaluator._set_factorized_source_strategy("tiled_coupling")
    evaluator.compute_node_energies_forces(*native_args)
    tiled = calculator._collect_mace_results(atoms, inputs)
    tiled_h1_adjoint = np.asarray(evaluator.H1_adj).copy()
    assert evaluator.factorized_forward_coupling_workspace_bytes > 0
    assert evaluator.factorized_reverse_coupling_workspace_bytes > 0
    assert evaluator.factorized_coupling_workspace_bytes > 0

    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_reverse_cache_policy("recompute")
    evaluator.compute_node_energies_forces(*native_args)
    if not evaluator.factorized_jit_ready:
        pytest.skip("a Execution R1 JIT plugin is required for receiver parity")
    native = calculator._collect_mace_results(atoms, inputs)
    native_h1_adjoint = np.asarray(evaluator.H1_adj)

    assert evaluator.factorized_forward_coupling_workspace_bytes == 0
    assert evaluator.factorized_reverse_coupling_workspace_bytes == 0
    assert evaluator.factorized_coupling_workspace_bytes == 0
    assert evaluator.factorized_coupling_capacity_bytes > 0
    assert native["energy"] == pytest.approx(tiled["energy"], rel=0.0, abs=3e-5)
    np.testing.assert_allclose(native["forces"], tiled["forces"], rtol=0.0, atol=3e-5)
    np.testing.assert_allclose(native_h1_adjoint, tiled_h1_adjoint, rtol=0.0, atol=3e-5)


def test_execution_direct_float32_repeated_and_reordered_edges_are_stable(
    streamed_model_paths,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    inputs = calculator._mace_inputs(atoms)

    def evaluate(native_inputs):
        evaluator.compute_node_energies_forces(
            *native_inputs[:5],
            np.asarray(native_inputs[5]).flatten(),
            native_inputs[6],
        )
        assert evaluator.factorized_execution_profile == "direct_fixed_weight"
        assert evaluator.factorized_selected_direct_reverse_executor == "jit"
        expected_r0_executor = (
            "v2_edge16"
            if native_symmetrix._kokkos_default_execution_space() in ("Cuda", "HIP")
            else "v2_receiver"
        )
        assert evaluator.standard_r0_selected_executor == expected_r0_executor
        edge_count = len(native_inputs[3])
        results = calculator._collect_mace_results(atoms, native_inputs)
        return {
            "energy": results["energy"],
            "forces": np.array(results["forces"], copy=True),
            "H1_adj": np.array(evaluator.H1_adj, copy=True),
            "directed_forces": np.array(evaluator.node_forces, copy=True).reshape(
                -1, 3
            )[:edge_count],
        }

    reference = evaluate(inputs)
    repeated_reference = evaluate(inputs)
    assert repeated_reference["energy"] == reference["energy"]
    for name in ("forces", "H1_adj", "directed_forces"):
        np.testing.assert_allclose(
            repeated_reference[name], reference[name], rtol=0.0, atol=3e-5
        )

    offsets = np.concatenate(([0], np.cumsum(inputs[2])))
    permutation = np.concatenate(
        [np.arange(offsets[i], offsets[i + 1])[::-1] for i in range(len(atoms))]
    )
    reordered_inputs = (
        inputs[0],
        inputs[1],
        inputs[2],
        np.asarray(inputs[3])[permutation],
        np.asarray(inputs[4])[permutation],
        np.asarray(inputs[5])[permutation],
        np.asarray(inputs[6])[permutation],
        np.asarray(inputs[7])[permutation],
    )
    reordered = evaluate(reordered_inputs)
    repeated_reordered = evaluate(reordered_inputs)
    assert repeated_reordered["energy"] == reordered["energy"]
    for name in ("forces", "H1_adj", "directed_forces"):
        np.testing.assert_allclose(
            repeated_reordered[name], reordered[name], rtol=0.0, atol=3e-5
        )

    assert reordered["energy"] == pytest.approx(reference["energy"], rel=0.0, abs=3e-5)
    np.testing.assert_allclose(
        reordered["forces"], reference["forces"], rtol=0.0, atol=3e-5
    )
    np.testing.assert_allclose(
        reordered["H1_adj"], reference["H1_adj"], rtol=0.0, atol=3e-5
    )
    np.testing.assert_allclose(
        reordered["directed_forces"],
        reference["directed_forces"][permutation],
        rtol=0.0,
        atol=3e-5,
    )


def test_execution_direct_float32_force_matches_finite_displacement(
    streamed_model_paths,
):
    _require_execution_fixed_weight_backend()

    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_reverse_executor("jit")
    evaluator._set_standard_r0_executor("v2_edge16")
    reference_inputs = calculator._mace_inputs(atoms)
    token = evaluator._prepare_factorized_graph(*reference_inputs[:5])

    def evaluate(candidate_atoms):
        candidate_inputs = calculator._mace_inputs(candidate_atoms)
        for reference, candidate in zip(reference_inputs[1:5], candidate_inputs[1:5]):
            np.testing.assert_array_equal(candidate, reference)
        evaluator._compute_prepared_factorized(
            token,
            np.asarray(candidate_inputs[5]).flatten(),
            candidate_inputs[6],
        )
        assert evaluator.factorized_execution_profile == "direct_fixed_weight"
        assert evaluator.factorized_selected_direct_reverse_executor == "jit"
        assert evaluator.standard_r0_selected_executor == "v2_edge16"
        return calculator._collect_mace_results(candidate_atoms, candidate_inputs)

    reference = evaluate(atoms)
    atom_index = 0
    component = 0
    displacement = 5.0e-3
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[atom_index, component] += displacement
    minus.positions[atom_index, component] -= displacement
    plus_energy = evaluate(plus)["energy"]
    minus_energy = evaluate(minus)["energy"]
    finite_displacement_force = -(plus_energy - minus_energy) / (2.0 * displacement)

    assert finite_displacement_force == pytest.approx(
        reference["forces"][atom_index, component],
        rel=2e-3,
        abs=2e-3,
    )


def test_execution_direct_reenters_after_tight_budget_factorized_workspace(
    streamed_model_paths,
):
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_reverse_cache_policy("recompute")
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], np.asarray(inputs[5]).flatten(), inputs[6])

    evaluator.compute_node_energies_forces(*native_args)
    if not evaluator.factorized_jit_ready:
        pytest.skip(
            "a Execution R1 JIT plugin is required for budget re-entry coverage"
        )
    reference = calculator._collect_mace_results(atoms, inputs)
    reference_h1_adjoint = np.asarray(evaluator.H1_adj).copy()
    assert evaluator.factorized_execution_profile == "direct_fixed_weight"
    edge_count = len(inputs[3])
    expected_direct_schedule_bytes = np.dtype(np.int32).itemsize * (
        2 * edge_count + 2 * len(atoms) + 1
    )
    assert evaluator.factorized_schedule_entries == edge_count
    assert evaluator.factorized_schedule_bytes == expected_direct_schedule_bytes

    evaluator.set_streamed_edges("receiver_factorized")
    evaluator._set_factorized_observer(True)
    evaluator._set_factorized_planner_budget_bytes(1)
    evaluator.compute_node_energies_forces(*native_args)
    tight_budget = evaluator.factorized_schedule_bytes
    assert tight_budget > 1
    assert evaluator.factorized_tile_selection_reason == (
        "minimum footprint exceeds planner budget"
    )

    evaluator._set_factorized_planner_budget_bytes(tight_budget)
    evaluator.compute_node_energies_forces(*native_args)
    assert evaluator.factorized_execution_profile == "factorized_state"
    assert evaluator.factorized_jit_ready
    assert evaluator.factorized_selected_direct_forward_executor == "runtime"
    assert evaluator.factorized_selected_direct_reverse_executor == "runtime"
    assert evaluator.factorized_arena_workspace_bytes > 0
    assert evaluator.factorized_radial_workspace_bytes > 0
    assert evaluator.factorized_workspace_bytes > tight_budget
    assert evaluator.factorized_tiled_ready

    evaluator._set_factorized_observer(False)
    evaluator.set_streamed_edges("direct")
    assert evaluator.factorized_jit_ready
    assert evaluator.factorized_selected_direct_forward_executor == "jit_all"
    assert evaluator.factorized_selected_direct_reverse_executor == "jit"
    assert evaluator.factorized_execution_profile == "direct_fixed_weight"
    assert evaluator.factorized_selected_reverse_cache_policy == "not_applicable"
    evaluator.compute_node_energies_forces(*native_args)
    assert (
        evaluator.factorized_workspace_bytes
        <= evaluator.factorized_workspace_capacity_bytes
    )
    assert evaluator.factorized_arena_workspace_bytes == 0
    assert evaluator.factorized_radial_workspace_bytes == 0

    assert evaluator.factorized_schedule_bytes == expected_direct_schedule_bytes
    direct = calculator._collect_mace_results(atoms, inputs)
    direct_h1_adjoint = np.asarray(evaluator.H1_adj)
    assert direct["energy"] == pytest.approx(reference["energy"], rel=0.0, abs=3e-5)
    np.testing.assert_allclose(
        direct["forces"], reference["forces"], rtol=0.0, atol=3e-5
    )
    np.testing.assert_allclose(
        direct_h1_adjoint, reference_h1_adjoint, rtol=0.0, atol=3e-5
    )


def test_factorized_schedule_handles_reordered_and_repeated_edges(
    streamed_model_paths,
):
    standard_path, _ = streamed_model_paths
    atoms = _small_structure()
    calculator = Symmetrix(
        standard_path,
        use_kokkos=True,
        dtype="float64",
        streamed_edges="factorized",
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)

    def evaluate(native_inputs):
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, _ = native_inputs
        evaluator.compute_node_energies_forces(
            num_nodes,
            node_types,
            num_neigh,
            j_list,
            neigh_types,
            xyz.flatten(),
            r,
        )
        return calculator._collect_mace_results(atoms, native_inputs)

    reference = evaluate(inputs)
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
        np.asarray(inputs[5])[permutation],
        np.asarray(inputs[6])[permutation],
        np.asarray(inputs[7])[permutation],
    )
    reordered_result = evaluate(reordered)
    assert evaluator.factorized_schedule_build_count == 2
    assert reordered_result["energy"] == pytest.approx(
        reference["energy"], rel=0.0, abs=3e-11
    )
    np.testing.assert_allclose(
        reordered_result["forces"], reference["forces"], rtol=0.0, atol=3e-11
    )

    insertion = int(inputs[2][0])
    repeated_num_neigh = np.array(inputs[2], copy=True)
    repeated_num_neigh[0] += 1
    repeated = (
        inputs[0],
        inputs[1],
        repeated_num_neigh,
        np.insert(np.asarray(inputs[3]), insertion, inputs[3][0]),
        np.insert(np.asarray(inputs[4]), insertion, inputs[4][0]),
        np.insert(np.asarray(inputs[5]), insertion, inputs[5][0], axis=0),
        np.insert(np.asarray(inputs[6]), insertion, inputs[6][0]),
        np.insert(np.asarray(inputs[7]), insertion, inputs[7][0]),
    )
    repeated_result = evaluate(repeated)
    assert evaluator.factorized_schedule_build_count == 3
    assert evaluator.factorized_schedule_entries == len(repeated[3])
    assert np.isfinite(repeated_result["energy"])
    assert np.all(np.isfinite(repeated_result["forces"]))

    invalid_sources = np.array(repeated[3], copy=True)
    invalid_sources[0] = inputs[0]
    with pytest.raises(IndexError, match="source index is out of range"):
        evaluate((*repeated[:3], invalid_sources, *repeated[4:]))
    assert evaluator.factorized_schedule_build_count == 3
