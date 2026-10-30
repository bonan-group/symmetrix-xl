import numpy as np
import pytest

from symmetrix.mace_alchemical import (
    AlchemicalConfig,
    AlchemicalPair,
    ElementMixture,
    expand_alchemical_topology,
    normalize_alchemical,
    validate_alchemical_model,
)


def test_expand_dual_topology_preserves_upstream_slot_edges():
    config = AlchemicalConfig(
        groups=(
            (AlchemicalPair(0, 8), AlchemicalPair(2, 6)),
            (AlchemicalPair(0, 6), AlchemicalPair(2, 8)),
        ),
        weights=np.array([0.3, 0.7]),
    )
    topology = expand_alchemical_topology(
        atomic_numbers=np.array([8, 1, 6]),
        receivers=np.array([0, 0, 1, 1, 2, 2, 0]),
        sources=np.array([1, 2, 0, 2, 0, 1, 2]),
        shifts=np.zeros((7, 3), dtype=np.int32),
        config=config,
    )

    np.testing.assert_array_equal(topology.node_to_original_atom, [0, 0, 1, 2, 2])
    np.testing.assert_array_equal(topology.node_slots, [0, 1, -1, 0, 1])
    np.testing.assert_array_equal(topology.node_atomic_numbers, [8, 6, 1, 6, 8])
    node_weights, edge_weights = topology.resolve_weights(config.weights)
    np.testing.assert_allclose(node_weights, [0.3, 0.7, 1.0, 0.3, 0.7])

    np.testing.assert_array_equal(topology.num_neigh, [3, 3, 4, 2, 2])
    np.testing.assert_array_equal(
        topology.edge_sources, [2, 3, 3, 2, 4, 4, 0, 1, 3, 4, 0, 2, 1, 2]
    )
    np.testing.assert_array_equal(
        topology.edge_to_original_edge, [0, 1, 6, 0, 1, 6, 2, 2, 3, 3, 4, 5, 4, 5]
    )
    np.testing.assert_allclose(
        edge_weights, [1, 1, 1, 1, 1, 1, 0.3, 0.7, 0.3, 0.7, 1, 1, 1, 1]
    )
    config.set_weights([0.8, 0.2])
    updated_node_weights, updated_edge_weights = topology.resolve_weights(
        config.weights
    )
    np.testing.assert_allclose(updated_node_weights, [0.8, 0.2, 1.0, 0.8, 0.2])
    np.testing.assert_allclose(
        updated_edge_weights,
        [1, 1, 1, 1, 1, 1, 0.8, 0.2, 0.8, 0.2, 1, 1, 1, 1],
    )


def test_alchemical_config_rejects_invalid_updates():
    config = AlchemicalConfig(
        groups=((AlchemicalPair(0, 8),),),
        weights=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        config.set_weights([np.nan])
    with pytest.raises(ValueError, match="length"):
        config.set_weights([])


def test_alchemical_config_version_changes_only_after_valid_change():
    config = AlchemicalConfig(
        groups=((AlchemicalPair(0, 8),),),
        weights=np.array([1.0]),
    )
    assert config.version == 0
    config.set_weights([1.0])
    assert config.version == 0
    config.set_weights([0.5])
    assert config.version == 1


def test_virtual_edges_match_global_slots_not_species_or_local_order():
    config = AlchemicalConfig(
        groups=[[(0, 8)], [(0, 6), (1, 8)], [(1, 6)]], weights=[0.2, 0.5, 0.3]
    )
    topology = expand_alchemical_topology(
        atomic_numbers=[8, 6],
        receivers=[0, 1],
        sources=[1, 0],
        shifts=np.array([[1, 0, 0], [-1, 0, 0]]),
        config=config,
    )
    np.testing.assert_array_equal(topology.edge_receivers, [1, 2])
    np.testing.assert_array_equal(topology.edge_sources, [2, 1])
    np.testing.assert_array_equal(topology.edge_shifts, [[1, 0, 0], [-1, 0, 0]])
    _, edge_weights = topology.resolve_weights(config.weights)
    np.testing.assert_array_equal(edge_weights, [1, 1])


def test_weight_updates_are_atomic_and_owned():
    values = np.array([0.3, 0.7])
    config = AlchemicalConfig(groups=[[(0, 8)], [(0, 6)]], weights=values)
    values[0] = 100.0
    np.testing.assert_array_equal(config.weights, [0.3, 0.7])
    with pytest.raises(ValueError):
        config.set_weights([0.4, -1.0])
    np.testing.assert_array_equal(config.weights, [0.3, 0.7])
    assert config.version == 0
    with pytest.raises(ValueError):
        config.weights[0] = 10.0


def test_element_mapping_normalizes_symbols_and_owns_fractions():
    request = {"Sr": {"Sr": 0.25, "Ba": 0.75}}

    mixture = normalize_alchemical(request)
    request["Sr"]["Sr"] = 1.0

    assert isinstance(mixture, ElementMixture)
    assert mixture.reference == "Sr"
    assert dict(mixture.fractions) == {"Sr": 0.25, "Ba": 0.75}
    assert mixture.version == 0
    mixture.set_fractions({"Ba": 0.6, "Sr": 0.4})
    assert dict(mixture.fractions) == {"Sr": 0.4, "Ba": 0.6}
    assert mixture.version == 1
    numbers = np.array([38, 8, 22, 38], dtype=np.int32)
    resolved = mixture.resolve(numbers, model_species=[8, 22, 38, 56])
    assert resolved.groups == (
        (AlchemicalPair(0, 38), AlchemicalPair(3, 38)),
        (AlchemicalPair(0, 56), AlchemicalPair(3, 56)),
    )
    np.testing.assert_array_equal(numbers, [38, 8, 22, 38])


@pytest.mark.parametrize(
    "specification,match",
    [
        ({}, "exactly one reference element"),
        ({"Sr": {"Sr": 1.0}, "Ti": {"Ti": 1.0}}, "exactly one reference element"),
        ({"Sr": {"Sr": 0.2, "Ba": 0.7}}, "sum to one"),
        ({"Sr": {"Sr": 1.1, "Ba": -0.1}}, "non-negative"),
    ],
)
def test_element_mapping_rejects_ambiguous_or_invalid_mixtures(specification, match):
    with pytest.raises(ValueError, match=match):
        normalize_alchemical(specification)


@pytest.mark.parametrize(
    "specification",
    [
        ({"X": {"Sr": 0.5, "Ba": 0.5}}),
        ({"Sr": {"X": 0.5, "Ba": 0.5}}),
    ],
)
def test_element_mapping_rejects_non_element_sentinel(specification):
    with pytest.raises(ValueError, match="chemical symbol or atomic number"):
        normalize_alchemical(specification)


def test_element_mixture_rejects_non_scalar_fractions_atomically():
    with pytest.raises(ValueError, match="scalar values"):
        ElementMixture("Sr", {"Sr": [[0.2, 0.3]], "Ba": [[0.3, 0.2]]})

    mixture = ElementMixture("Sr", {"Sr": 0.5, "Ba": 0.5})
    original_fractions = dict(mixture.fractions)
    original_version = mixture.version
    with pytest.raises(ValueError, match="scalar values"):
        mixture.set_fractions({"Sr": [[0.2, 0.3]], "Ba": [[0.3, 0.2]]})
    assert dict(mixture.fractions) == original_fractions
    assert mixture.version == original_version


def test_element_mixture_normalizes_numeric_keys_and_validates_resolution():
    mixture = normalize_alchemical({38: {38: 0.5, 56: 0.5}})
    assert isinstance(mixture, ElementMixture)
    assert mixture.reference == "Sr"
    assert dict(mixture.fractions) == {"Sr": 0.5, "Ba": 0.5}

    mixture.set_fractions({"Sr": 0.5, 56: 0.5})
    assert mixture.version == 0
    with pytest.raises(ValueError, match="contains no reference element"):
        mixture.resolve([8, 22], model_species=[8, 22, 38, 56])
    with pytest.raises(
        ValueError, match="does not support alchemical atomic number 56"
    ):
        mixture.resolve([38, 8, 22], model_species=[8, 22, 38])


def test_element_mixture_rejects_duplicate_symbol_and_number_aliases():
    with pytest.raises(ValueError, match="specified more than once"):
        ElementMixture("Sr", {"Sr": 0.25, 38: 0.25, "Ba": 0.5})


@pytest.fixture(scope="module")
def alchemical_model_path(tmp_path_factory):
    import json
    from pathlib import Path
    from compact_r1_model import foundation_r1_test_model
    from symmetrix.mace_execution_contract import make_standard_m0_contract

    contracts = Path(__file__).parent / "data" / "execution_contracts"
    r1 = json.loads((contracts / "jit_r1_mace_omat0_small_contract.json").read_text())
    model = foundation_r1_test_model(r1, atomic_numbers=(6, 8))
    model["execution_contracts"]["M0"] = make_standard_m0_contract(
        channels=model["num_channels"],
        type_count=2,
        input_l_max=model["l_max"],
        output_l_max=model["L_max"],
        correlation=1,
        monomials=model["M0_monomials"],
    )
    model["execution_contracts"]["R0"] = json.loads(
        (contracts / "standard_r0_contract.json").read_text()
    )
    model["atomic_energies"] = [-0.6, 1.2]
    model["readout_1_weights"] = [0.02] * model["num_channels"]
    path = tmp_path_factory.mktemp("alchemical-model") / "model.json"
    path.write_text(json.dumps(model))
    return path


@pytest.fixture(scope="module")
def upstream_alchemical_model_path(tmp_path_factory):
    """Rebuild the deterministic foundation used for the frozen upstream oracle."""

    import json
    import torch
    import torch.nn.functional as F
    from e3nn import o3
    from mace import modules
    from mace.modules import ScaleShiftMACE
    from symmetrix.extract_mace_data import _extract_single_head_data

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(123)
        model = ScaleShiftMACE(
            r_max=5.0,
            num_bessel=3,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            num_interactions=2,
            num_elements=3,
            hidden_irreps=o3.Irreps("2x0e + 2x1o"),
            MLP_irreps=o3.Irreps("2x0e"),
            gate=F.silu,
            atomic_energies=np.array([-13.6, -1000.0, -2040.0]),
            avg_num_neighbors=8.0,
            atomic_numbers=[1, 6, 8],
            correlation=3,
            radial_type="bessel",
            atomic_inter_shift=0.0,
            atomic_inter_scale=1.0,
        )
        model.eval()
        data = _extract_single_head_data(
            model, radial_format="compact", num_spline_points=64
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    path = tmp_path_factory.mktemp("upstream-alchemical-model") / "model.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture(scope="module")
def omat0_alchemical_model_path(tmp_path_factory):
    """Rebuild the deterministic OMAT-style model used by the frozen oracle."""

    import hashlib
    import json
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from e3nn import o3
    from mace import modules
    from mace.modules import ScaleShiftMACE
    from symmetrix.extract_mace_data import _extract_single_head_data

    oracle_path = (
        Path(__file__).parent / "data" / "alchemical_omat0_oracle.json.fixture"
    )
    oracle = json.loads(oracle_path.read_text())
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(oracle["seed"])
        model = ScaleShiftMACE(
            r_max=5.0,
            num_bessel=3,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticDensityResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticDensityInteractionBlock"
            ],
            num_interactions=2,
            num_elements=4,
            hidden_irreps=o3.Irreps("4x0e + 4x1o"),
            MLP_irreps=o3.Irreps("4x0e"),
            gate=F.silu,
            atomic_energies=np.array([-2040.0, -1580.0, -800.0, -750.0]),
            avg_num_neighbors=8.0,
            atomic_numbers=oracle["atomic_numbers"],
            correlation=3,
            radial_type="bessel",
            distance_transform="Agnesi",
            pair_repulsion=True,
            atomic_inter_shift=0.0,
            atomic_inter_scale=1.0,
        )
        model.eval()
        data = _extract_single_head_data(
            model, radial_format="compact", num_spline_points=128
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    model_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    if model_sha256 != oracle["model_sha256"]:
        pytest.skip(
            "the frozen OMAT-0 oracle requires upstream MACE commit "
            f"{oracle['upstream_commit']}; generated model SHA-256 is {model_sha256}"
        )
    path = tmp_path_factory.mktemp("alchemical-omat0-model") / "model.json"
    path.write_text(canonical)
    return path


@pytest.mark.parametrize(
    "dtype,atol",
    [
        (
            "float64",
            {"energy": 6e-8, "energies": 3e-8, "forces": 5e-6, "stress": 3e-8},
        ),
        (
            "float32",
            {"energy": 5e-3, "energies": 3e-3, "forces": 5e-4, "stress": 5e-6},
        ),
    ],
)
def test_omat0_features_match_frozen_dual_topology_oracle(
    omat0_alchemical_model_path, dtype, atol, monkeypatch, tmp_path
):
    """Match the extended ce2b077 upstream reference for density, Agnesi, and ZBL."""

    import json
    from pathlib import Path
    from ase import Atoms
    from symmetrix import Symmetrix

    oracle = json.loads(
        (
            Path(__file__).parent / "data" / "alchemical_omat0_oracle.json.fixture"
        ).read_text()
    )
    atoms = Atoms(
        numbers=oracle["structure"]["numbers"],
        positions=oracle["structure"]["positions_A"],
        cell=oracle["structure"]["cell_A"],
        pbc=oracle["structure"]["pbc"],
    )
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    calculator = Symmetrix(
        omat0_alchemical_model_path,
        dtype=dtype,
        alchemical=AlchemicalConfig(
            groups=[[(0, 38)], [(0, 56)]], weights=oracle["weights"]
        ),
        kernel_launch_policy="static",
    )
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(
        oracle["energy_eV"][0], abs=atol["energy"]
    )
    np.testing.assert_allclose(
        atoms.get_potential_energies(),
        oracle["energies_eV"],
        rtol=0,
        atol=atol["energies"],
    )
    np.testing.assert_allclose(
        atoms.get_forces(), oracle["forces_eV_per_A"], rtol=0, atol=atol["forces"]
    )
    np.testing.assert_allclose(
        atoms.get_stress(voigt=False),
        oracle["stress_eV_per_A3"][0],
        rtol=0,
        atol=atol["stress"],
    )
    assert oracle["has_nonzero_periodic_shift"]
    assert np.any(calculator._neighbor_cache.shifts != 0)
    assert calculator.evaluator.factorized_jit_ready
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0


def test_alchemical_model_admits_omat0_features(omat0_alchemical_model_path):
    import json

    model = json.loads(omat0_alchemical_model_path.read_text())
    assert model["has_zbl"]
    assert model["A0_scaled"]
    assert model["A1_scaled"]
    assert model["compact_radial"]["distance_transform"]["type"] == "agnesi"
    validate_alchemical_model(model)


@pytest.mark.parametrize("unsupported", ["field", "multihead", "soft_transform"])
def test_alchemical_model_retains_unsupported_feature_guards(
    omat0_alchemical_model_path, unsupported
):
    import copy
    import json

    model = json.loads(omat0_alchemical_model_path.read_text())
    if unsupported == "field":
        model["has_field_coupling"] = True
    elif unsupported == "soft_transform":
        model["compact_radial"]["distance_transform"]["type"] = "soft"
    else:
        fields = [
            "atomic_energies",
            "readout_1_weights",
            "readout_2_weights_1",
            "readout_2_weights_2",
            "zbl_c",
        ]
        payload = {field: copy.deepcopy(model[field]) for field in fields}
        model["prediction_heads"] = {
            "schema_version": 1,
            "names": ["first", "second"],
            "default": "first",
            "parameters": {
                "first": copy.deepcopy(payload),
                "second": copy.deepcopy(payload),
            },
        }
    with pytest.raises(ValueError, match="does not support"):
        validate_alchemical_model(model)


def _omat0_variant(model_path, target, *, a0, a1, zbl):
    import json

    model = json.loads(model_path.read_text())
    model["A0_scaled"] = a0
    model["A1_scaled"] = a1
    model["has_zbl"] = zbl
    if not a0:
        model["compact_radial"]["networks"].pop("A0", None)
    if not a1:
        model["compact_radial"]["networks"].pop("A1", None)
    target.write_text(json.dumps(model))
    return target


def _omat0_atoms():
    import json
    from pathlib import Path
    from ase import Atoms

    oracle = json.loads(
        (
            Path(__file__).parent / "data" / "alchemical_omat0_oracle.json.fixture"
        ).read_text()
    )
    return Atoms(
        numbers=oracle["structure"]["numbers"],
        positions=oracle["structure"]["positions_A"],
        cell=oracle["structure"]["cell_A"],
        pbc=oracle["structure"]["pbc"],
    )


def _assert_observables_match(left, right, *, atol):
    assert left.get_potential_energy() == pytest.approx(
        right.get_potential_energy(), abs=atol
    )
    np.testing.assert_allclose(
        left.get_potential_energies(),
        right.get_potential_energies(),
        rtol=0,
        atol=atol,
    )
    np.testing.assert_allclose(left.get_forces(), right.get_forces(), rtol=0, atol=atol)
    np.testing.assert_allclose(left.get_stress(), right.get_stress(), rtol=0, atol=atol)


def test_a0_density_counts_weighted_physical_sources(
    omat0_alchemical_model_path, monkeypatch, tmp_path
):
    from symmetrix import Symmetrix

    model_path = _omat0_variant(
        omat0_alchemical_model_path,
        tmp_path / "a0-only.json",
        a0=True,
        a1=False,
        zbl=False,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    config = AlchemicalConfig(groups=[[(0, 38)], [(0, 56)]], weights=[1.0, 0.0])
    mixed = _omat0_atoms()
    mixed.calc = Symmetrix(
        model_path,
        dtype="float64",
        alchemical=config,
        kernel_launch_policy="static",
    )
    for weights, atomic_number in [([1.0, 0.0], 38), ([0.0, 1.0], 56)]:
        config.set_weights(weights)
        pure = _omat0_atoms()
        pure.numbers[0] = atomic_number
        pure.calc = Symmetrix(
            model_path, dtype="float64", kernel_launch_policy="static"
        )
        _assert_observables_match(mixed, pure, atol=2e-8)


def _assert_mixed_coordinate_and_strain_derivatives(atoms, *, force_atol, stress_atol):
    forces = atoms.get_forces().copy()
    stress = atoms.get_stress().copy()
    step = 2e-5
    for atom, axis in [(0, 0), (1, 1), (2, 2)]:
        original = atoms.positions[atom, axis]
        atoms.positions[atom, axis] = original + step
        plus = atoms.get_potential_energy()
        atoms.positions[atom, axis] = original - step
        minus = atoms.get_potential_energy()
        atoms.positions[atom, axis] = original
        assert forces[atom, axis] == pytest.approx(
            -(plus - minus) / (2 * step), abs=force_atol
        )
    cell = atoms.cell.copy()
    volume = atoms.get_volume()
    for axis in range(3):
        plus_cell, minus_cell = cell.copy(), cell.copy()
        plus_cell[:, axis] *= 1 + step
        minus_cell[:, axis] *= 1 - step
        atoms.set_cell(plus_cell, scale_atoms=True)
        plus = atoms.get_potential_energy()
        atoms.set_cell(minus_cell, scale_atoms=True)
        minus = atoms.get_potential_energy()
        atoms.set_cell(cell, scale_atoms=True)
        assert stress[axis] == pytest.approx(
            (plus - minus) / (2 * step * volume), abs=stress_atol
        )


def test_a0_density_mixed_force_and_stress_match_finite_difference(
    omat0_alchemical_model_path, monkeypatch, tmp_path
):
    from symmetrix import Symmetrix

    model_path = _omat0_variant(
        omat0_alchemical_model_path,
        tmp_path / "a0-only.json",
        a0=True,
        a1=False,
        zbl=False,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    atoms = _omat0_atoms()
    atoms.calc = Symmetrix(
        model_path,
        dtype="float64",
        alchemical=AlchemicalConfig(
            groups=[[(0, 38)], [(0, 56)]], weights=[0.35, 0.65]
        ),
        kernel_launch_policy="static",
    )
    _assert_mixed_coordinate_and_strain_derivatives(
        atoms, force_atol=2e-7, stress_atol=2e-9
    )


def test_a1_density_counts_weighted_physical_sources(
    omat0_alchemical_model_path, monkeypatch, tmp_path
):
    from symmetrix import Symmetrix

    model_path = _omat0_variant(
        omat0_alchemical_model_path,
        tmp_path / "a1-only.json",
        a0=False,
        a1=True,
        zbl=False,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    config = AlchemicalConfig(groups=[[(0, 38)], [(0, 56)]], weights=[1.0, 0.0])
    mixed = _omat0_atoms()
    mixed.calc = Symmetrix(
        model_path,
        dtype="float64",
        alchemical=config,
        kernel_launch_policy="static",
    )
    for weights, atomic_number in [([1.0, 0.0], 38), ([0.0, 1.0], 56)]:
        config.set_weights(weights)
        pure = _omat0_atoms()
        pure.numbers[0] = atomic_number
        pure.calc = Symmetrix(
            model_path, dtype="float64", kernel_launch_policy="static"
        )
        _assert_observables_match(mixed, pure, atol=2e-8)


def test_a1_density_mixed_force_and_stress_match_finite_difference(
    omat0_alchemical_model_path, monkeypatch, tmp_path
):
    from symmetrix import Symmetrix

    model_path = _omat0_variant(
        omat0_alchemical_model_path,
        tmp_path / "a1-only.json",
        a0=False,
        a1=True,
        zbl=False,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    atoms = _omat0_atoms()
    atoms.calc = Symmetrix(
        model_path,
        dtype="float64",
        alchemical=AlchemicalConfig(
            groups=[[(0, 38)], [(0, 56)]], weights=[0.35, 0.65]
        ),
        kernel_launch_policy="static",
    )
    _assert_mixed_coordinate_and_strain_derivatives(
        atoms, force_atol=2e-7, stress_atol=2e-9
    )


def test_zbl_counts_weighted_concrete_edges(
    omat0_alchemical_model_path, monkeypatch, tmp_path
):
    from symmetrix import Symmetrix

    model_path = _omat0_variant(
        omat0_alchemical_model_path,
        tmp_path / "zbl-only.json",
        a0=False,
        a1=False,
        zbl=True,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    config = AlchemicalConfig(groups=[[(0, 38)], [(0, 56)]], weights=[1.0, 0.0])
    mixed = _omat0_atoms()
    mixed.calc = Symmetrix(
        model_path,
        dtype="float64",
        alchemical=config,
        kernel_launch_policy="static",
    )
    prepared_before = mixed.calc.evaluator.factorized_prepared_evaluation_count
    for weights, atomic_number in [([1.0, 0.0], 38), ([0.0, 1.0], 56)]:
        config.set_weights(weights)
        pure = _omat0_atoms()
        pure.numbers[0] = atomic_number
        pure.calc = Symmetrix(
            model_path, dtype="float64", kernel_launch_policy="static"
        )
        _assert_observables_match(mixed, pure, atol=2e-8)
    assert (
        mixed.calc.evaluator.factorized_prepared_evaluation_count - prepared_before == 4
    )
    assert mixed.calc.evaluator.factorized_fallback_evaluation_count == 0


def test_zbl_mixed_force_and_stress_match_finite_difference(
    omat0_alchemical_model_path, monkeypatch, tmp_path
):
    from symmetrix import Symmetrix

    model_path = _omat0_variant(
        omat0_alchemical_model_path,
        tmp_path / "zbl-only.json",
        a0=False,
        a1=False,
        zbl=True,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    atoms = _omat0_atoms()
    atoms.calc = Symmetrix(
        model_path,
        dtype="float64",
        alchemical=AlchemicalConfig(
            groups=[[(0, 38)], [(0, 56)]], weights=[0.35, 0.65]
        ),
        kernel_launch_policy="static",
    )
    _assert_mixed_coordinate_and_strain_derivatives(
        atoms, force_atol=2e-7, stress_atol=2e-9
    )
    assert atoms.calc.evaluator.factorized_fallback_evaluation_count == 0


def _atoms():
    from ase import Atoms

    return Atoms(
        numbers=[8, 6, 8],
        positions=[[0, 0, 0], [1.3, 0.2, 0.1], [0.3, 1.6, 0.4]],
        cell=[9, 9, 9],
        pbc=False,
    )


def _config(weights):
    return AlchemicalConfig(
        groups=[[(0, 8), (2, 8)], [(0, 6), (2, 6)]], weights=weights
    )


def test_plain_element_mapping_selects_every_reference_atom_without_mutation(
    alchemical_model_path, monkeypatch, tmp_path
):
    from symmetrix import ElementMixture, Symmetrix

    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    requested = {"O": {"O": 0.35, "C": 0.65}}
    mixed = _atoms()
    original_numbers = mixed.numbers.copy()
    calculator = Symmetrix(
        alchemical_model_path,
        dtype="float64",
        alchemical=requested,
        kernel_launch_policy="static",
    )
    mixed.calc = calculator
    reference = _atoms()
    reference.calc = Symmetrix(
        alchemical_model_path,
        dtype="float64",
        alchemical=_config([0.35, 0.65]),
        kernel_launch_policy="static",
    )

    _assert_observables_match(mixed, reference, atol=2e-10)
    np.testing.assert_array_equal(mixed.numbers, original_numbers)
    assert isinstance(calculator.alchemical, ElementMixture)
    assert tuple(
        len(nodes) for nodes in calculator._alchemical_topology.atom_node_indices
    ) == (
        2,
        1,
        2,
    )

    graph_count = calculator.evaluator.factorized_prepared_graph_count
    previous = mixed.get_potential_energy()
    calculator.alchemical.set_fractions({"O": 0.8, "C": 0.2})
    assert abs(mixed.get_potential_energy() - previous) > 0.01
    assert calculator.evaluator.factorized_prepared_graph_count == graph_count
    np.testing.assert_array_equal(mixed.numbers, original_numbers)


@pytest.mark.parametrize(
    "dtype,atol",
    [
        (
            "float64",
            {"energy": 3e-9, "energies": 2e-9, "forces": 2e-7, "stress": 4e-11},
        ),
        ("float32", {"energy": 3e-4, "energies": 3e-4, "forces": 3e-6, "stress": 3e-9}),
    ],
)
def test_interior_mixture_matches_frozen_upstream_periodic_oracle(
    upstream_alchemical_model_path, dtype, atol, monkeypatch, tmp_path
):
    """Match bonan-group/mace alchemical-mace at ce2b07786d5e."""

    from ase import Atoms
    from symmetrix import Symmetrix

    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.1, 0.1, 0.1], [11.7, 0.2, 0.1], [0.2, 11.6, 0.3]],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )
    expected_energy = -1391.2006296357592
    expected_energies = [-1364.0004885388748, -13.600205288441305, -13.599935808443341]
    expected_forces = [
        [-0.0011598192645768872, -0.0014684206694073366, 0.0007403686254953927],
        [0.002054285653345295, -0.001011639385555999, 0.00020971283040828456],
        [-0.0008944663887684079, 0.002480060054963335, -0.0009500814559036774],
    ]
    expected_stress = [
        5.272921876243977e-07,
        7.761539155308266e-07,
        1.0996313147033305e-07,
        -2.8704398784297817e-07,
        1.0352620240375099e-07,
        -3.77697777615008e-07,
    ]

    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    calculator = Symmetrix(
        upstream_alchemical_model_path,
        dtype=dtype,
        alchemical=AlchemicalConfig(groups=[[(0, 8)], [(0, 6)]], weights=[0.35, 0.65]),
        kernel_launch_policy="static",
    )
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(
        expected_energy, abs=atol["energy"]
    )
    np.testing.assert_allclose(
        atoms.get_potential_energies(), expected_energies, rtol=0, atol=atol["energies"]
    )
    np.testing.assert_allclose(
        atoms.get_forces(), expected_forces, rtol=0, atol=atol["forces"]
    )
    np.testing.assert_allclose(
        atoms.get_stress(), expected_stress, rtol=0, atol=atol["stress"]
    )
    assert np.any(calculator._neighbor_cache.shifts != 0)
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0


@pytest.mark.parametrize("dtype,atol", [("float64", 2e-10), ("float32", 3e-5)])
@pytest.mark.parametrize("profile", ["speed", "capacity"])
def test_endpoints_and_weight_updates_preserve_direct_graph(
    alchemical_model_path, dtype, atol, profile, monkeypatch, tmp_path
):
    from symmetrix import Symmetrix

    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    config = _config([1, 0])
    mixed = _atoms()
    calculator = Symmetrix(
        alchemical_model_path,
        dtype=dtype,
        execution_profile=profile,
        alchemical=config,
        kernel_launch_policy="static",
    )
    mixed.calc = calculator
    for weights, number in [([1, 0], 8), ([0, 1], 6)]:
        config.set_weights(weights)
        pure = _atoms()
        pure.numbers[[0, 2]] = number
        pure.calc = Symmetrix(
            alchemical_model_path,
            dtype=dtype,
            execution_profile=profile,
            kernel_launch_policy="static",
        )
        assert mixed.get_potential_energy() == pytest.approx(
            pure.get_potential_energy(), abs=atol
        )
        np.testing.assert_allclose(
            mixed.get_forces(), pure.get_forces(), rtol=0, atol=atol
        )
        np.testing.assert_allclose(
            mixed.get_stress(), pure.get_stress(), rtol=0, atol=atol
        )
        np.testing.assert_allclose(
            mixed.get_potential_energies(),
            pure.get_potential_energies(),
            rtol=0,
            atol=atol,
        )
    previous = mixed.get_potential_energy()
    graph_count = calculator.evaluator.factorized_prepared_graph_count
    config.set_weights([0.35, 0.65])
    assert abs(mixed.get_potential_energy() - previous) > 0.01
    assert calculator.evaluator.factorized_prepared_graph_count == graph_count
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0
    assert calculator.evaluator.factorized_jit_ready
    assert calculator.evaluator.alchemical_weights_active


def test_mixed_coordinate_and_strain_derivatives(
    alchemical_model_path, monkeypatch, tmp_path
):
    from symmetrix import Symmetrix

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    atoms = _atoms()
    atoms.calc = Symmetrix(
        alchemical_model_path,
        dtype="float64",
        alchemical=_config([0.35, 0.65]),
        kernel_launch_policy="static",
    )
    force = atoms.get_forces().copy()
    stress = atoms.get_stress().copy()
    h = 2e-5
    for atom, axis in [(0, 0), (1, 1), (2, 2)]:
        original = atoms.positions[atom, axis]
        atoms.positions[atom, axis] = original + h
        plus = atoms.get_potential_energy()
        atoms.positions[atom, axis] = original - h
        minus = atoms.get_potential_energy()
        atoms.positions[atom, axis] = original
        assert force[atom, axis] == pytest.approx(-(plus - minus) / (2 * h), abs=2e-8)
    cell = atoms.cell.copy()
    for axis in range(3):
        plus_cell, minus_cell = cell.copy(), cell.copy()
        plus_cell[:, axis] *= 1 + h
        minus_cell[:, axis] *= 1 - h
        atoms.set_cell(plus_cell, scale_atoms=True)
        plus = atoms.get_potential_energy()
        atoms.set_cell(minus_cell, scale_atoms=True)
        minus = atoms.get_potential_energy()
        atoms.set_cell(cell, scale_atoms=True)
        assert stress[axis] == pytest.approx(
            (plus - minus) / (2 * h * atoms.get_volume()), abs=2e-10
        )


@pytest.mark.parametrize(
    "options",
    [
        {"streamed_edges": "reference"},
        {"streamed_edges": "non-compiled"},
        {"allow_fixed_workspace": True},
        {"metal": True},
    ],
)
def test_alchemical_execution_rejects_unsupported_paths(alchemical_model_path, options):
    from symmetrix import Symmetrix

    with pytest.raises(ValueError, match="Alchemical execution"):
        Symmetrix(alchemical_model_path, alchemical=_config([0.3, 0.7]), **options)


def test_alchemical_execution_rejects_native_corrections(alchemical_model_path):
    from symmetrix import D3Correction, Symmetrix

    with pytest.raises(ValueError, match="additive corrections"):
        Symmetrix(
            alchemical_model_path,
            alchemical=_config([0.3, 0.7]),
            corrections=[D3Correction()],
        )


@pytest.mark.parametrize("field,value", [("has_field_coupling", True)])
def test_alchemical_execution_rejects_nonlinear_features(
    alchemical_model_path, tmp_path, field, value
):
    import json
    from symmetrix import Symmetrix

    model = json.loads(alchemical_model_path.read_text())
    model[field] = value
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="Alchemical execution does not support"):
        Symmetrix(path, alchemical=_config([0.3, 0.7]))


def test_isolated_mixed_atoms_have_weighted_atomic_energies(
    alchemical_model_path, tmp_path, monkeypatch
):
    from symmetrix import Symmetrix

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    atoms = _atoms()
    atoms.positions[:] = [[0, 0, 0], [20, 0, 0], [0, 20, 0]]
    atoms.calc = Symmetrix(
        alchemical_model_path,
        dtype="float64",
        alchemical=_config([0.35, 0.65]),
        kernel_launch_policy="static",
    )
    np.testing.assert_allclose(
        atoms.get_potential_energies(), [0.03, -0.6, 0.03], atol=2e-14
    )
    np.testing.assert_array_equal(atoms.get_forces(), np.zeros((3, 3)))
    np.testing.assert_array_equal(atoms.get_stress(), np.zeros(6))


def test_invalid_native_coefficients_preserve_weighted_evaluation(
    alchemical_model_path, tmp_path, monkeypatch
):
    from symmetrix import Symmetrix

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit"))
    atoms = _atoms()
    calculator = Symmetrix(
        alchemical_model_path,
        dtype="float64",
        alchemical=_config([0.35, 0.65]),
        kernel_launch_policy="static",
    )
    atoms.calc = calculator
    atoms.get_potential_energy()
    evaluator = calculator.evaluator
    token = evaluator.factorized_graph_generation
    expected = np.array(evaluator.node_energies, copy=True)
    inputs = calculator._mace_inputs(atoms)
    node_count, _, _, sources, _, xyz, distances, _ = inputs
    nodes = np.ones(node_count)
    edges = np.ones(len(sources))
    invalid_nodes = nodes.copy()
    invalid_nodes[0] = np.nan
    invalid_edges = edges.copy()
    invalid_edges[0] = np.inf
    for graph, node_weights, edge_weights, message in [
        (token + 1, nodes, edges, "stale"),
        (token, nodes[:-1], edges, "extents"),
        (token, invalid_nodes, edges, "finite"),
        (token, nodes, invalid_edges, "finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            evaluator._set_alchemical_weights(graph, node_weights, edge_weights)
        evaluator._compute_prepared_factorized(token, xyz.reshape(-1), distances)
        np.testing.assert_allclose(
            evaluator.node_energies, expected, rtol=0, atol=2e-14
        )
