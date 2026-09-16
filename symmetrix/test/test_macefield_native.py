import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    from ase.build import bulk
    from mace.calculators import MACECalculator
    from mace.modules.extensions import MACEField
    from symmetrix import (
        Symmetrix,
        prepare_jit_device_artifact,
        prepare_jit_host_artifact,
    )
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data
    from test_macefield_field_transform import (
        _compact_field_coupling,
        _frozen_inputs,
        _standalone_field_transform,
    )
except ImportError as exc:
    pytest.skip(
        f"MACEField native test dependencies are not available: {exc}",
        allow_module_level=True,
    )


try:
    from matscipy.neighbours import neighbour_list as neighbor_list
except ImportError:
    from ase.neighborlist import neighbor_list


@pytest.fixture(scope="module")
def macefield_json_path(tmp_path_factory, macefield_model_path):
    output_path = tmp_path_factory.mktemp("macefield-json") / "macefield.json"
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=8,
    )
    output_path.write_text(json.dumps(data))
    return output_path


@pytest.fixture(scope="module")
def macefield_full_json_path(tmp_path_factory, macefield_model_path):
    output_path = tmp_path_factory.mktemp("macefield-full-json") / "macefield.json"
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
    )
    output_path.write_text(json.dumps(data))
    return output_path


@pytest.fixture(scope="module")
def small_channel_macefield_json_path(tmp_path_factory):
    from e3nn import o3
    from mace.modules.blocks import (
        RealAgnosticInteractionBlock,
        RealAgnosticResidualInteractionBlock,
    )

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(20260804)
        model = MACEField(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=4,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("4x0e+4x1o"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros((1, 2)),
            avg_num_neighbors=4.0,
            atomic_numbers=[7, 13],
            correlation=3,
            gate=torch.nn.functional.silu,
            heads=["test"],
        )
        artifact_dir = tmp_path_factory.mktemp("small-channel-macefield")
        checkpoint_path = artifact_dir / "macefield.model"
        torch.save(model, checkpoint_path)
        data = extract_mace_data(
            checkpoint_path,
            species=[7, 13],
            head="test",
            num_spline_points=8,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    assert data["num_channels"] == 4
    output_path = artifact_dir / "macefield.json"
    output_path.write_text(json.dumps(data))
    return output_path


@pytest.fixture(scope="module")
def residual_first_macefield_artifacts(tmp_path_factory):
    from e3nn import o3
    from mace.modules.blocks import RealAgnosticResidualInteractionBlock

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(20260806)
        model = MACEField(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=4,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticResidualInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("4x0e+4x1o"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros((1, 2)),
            avg_num_neighbors=4.0,
            atomic_numbers=[7, 13],
            correlation=3,
            gate=torch.nn.functional.silu,
            heads=["test"],
        )
        model.eval()
        artifact_dir = tmp_path_factory.mktemp("residual-first-macefield")
        checkpoint_path = artifact_dir / "macefield.model"
        torch.save(model, checkpoint_path)
        data = extract_mace_data(
            checkpoint_path,
            species=[7, 13],
            head="test",
            num_spline_points=256,
        )
        legacy_data = extract_mace_data(
            checkpoint_path,
            species=[7, 13],
            head="test",
            num_spline_points=256,
            radial_format="pair-splines",
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    json_path = artifact_dir / "macefield.json"
    json_path.write_text(json.dumps(data))
    legacy_json_path = artifact_dir / "macefield-legacy.json"
    legacy_json_path.write_text(json.dumps(legacy_data))
    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "data": data,
        "json_path": json_path,
        "legacy_json_path": legacy_json_path,
    }


@pytest.fixture(scope="module")
def rank2_macefield_artifacts(tmp_path_factory):
    from e3nn import o3
    from mace.modules.blocks import (
        RealAgnosticInteractionBlock,
        RealAgnosticResidualInteractionBlock,
    )

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(20260805)
        model = MACEField(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=4,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("4x0e+4x1o+4x2e"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros((1, 2)),
            avg_num_neighbors=4.0,
            atomic_numbers=[7, 13],
            correlation=3,
            gate=torch.nn.functional.silu,
            heads=["test"],
        )
        artifact_dir = tmp_path_factory.mktemp("rank2-macefield")
        checkpoint_path = artifact_dir / "macefield.model"
        torch.save(model, checkpoint_path)
        data = extract_mace_data(
            checkpoint_path,
            species=[7, 13],
            head="test",
            num_spline_points=128,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    output_path = artifact_dir / "macefield.json"
    output_path.write_text(json.dumps(data))
    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "data": data,
        "json_path": output_path,
    }


@pytest.mark.parametrize("evaluator_name", ["MACE", "MACEKokkos"])
def test_macefield_loads_non_128_channel_model(
    small_channel_macefield_json_path,
    evaluator_name,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos" and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    evaluator = getattr(native_symmetrix, evaluator_name)(
        str(small_channel_macefield_json_path)
    )

    assert evaluator.has_field_coupling is True


def _compact_to_native_h1(compact, num_channels=128, L_max=1):
    compact = np.asarray(compact)
    native = np.zeros(
        (compact.shape[0], (L_max + 1) ** 2, num_channels), dtype=np.float64
    )
    compact_offset = 0
    for l in range(L_max + 1):
        components = 2 * l + 1
        block_size = components * num_channels
        native[:, l * l : (l + 1) ** 2, :] = (-1) ** l * compact[
            :, compact_offset : compact_offset + block_size
        ].reshape(compact.shape[0], num_channels, components).transpose(0, 2, 1)
        compact_offset += block_size
    return native.reshape(-1)


def _native_to_compact_h1(native, num_nodes, num_channels=128, L_max=1):
    native = np.asarray(native, dtype=np.float64).reshape(
        num_nodes, (L_max + 1) ** 2, num_channels
    )
    compact = np.zeros((num_nodes, (L_max + 1) ** 2 * num_channels), dtype=np.float64)
    compact_offset = 0
    for l in range(L_max + 1):
        components = 2 * l + 1
        block_size = components * num_channels
        compact[:, compact_offset : compact_offset + block_size] = (-1) ** l * native[
            :, l * l : (l + 1) ** 2, :
        ].transpose(0, 2, 1).reshape(num_nodes, block_size)
        compact_offset += block_size
    return compact


def test_residual_first_schema_matches_upstream_skip(
    residual_first_macefield_artifacts,
):
    model = residual_first_macefield_artifacts["model"]
    data = residual_first_macefield_artifacts["data"]
    node_attrs = torch.eye(2, dtype=torch.float64)
    with torch.no_grad():
        node_feats = model.node_embedding(node_attrs)
        expected = model.interactions[0].skip_tp(node_feats, node_attrs)
    expected_native = _compact_to_native_h1(expected.numpy(force=True), 4, 1)

    assert data["first_interaction_residual"] is True
    assert data["readout_2_hidden_size"] == 16
    assert len(data["H1_first_residual_weights"]) == 2 * 4 * 4
    assert np.allclose(
        data["H1_first_residual_weights"],
        expected_native,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("evaluator_name", ["MACE", "MACEKokkos"])
def test_loader_accepts_nonstandard_readout_hidden_size(
    residual_first_macefield_artifacts,
    tmp_path,
    evaluator_name,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos" and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    data = dict(residual_first_macefield_artifacts["data"])
    hidden_size = 7
    data["readout_2_hidden_size"] = hidden_size
    data["readout_2_weights_1"] = [0.0] * (data["num_channels"] * hidden_size)
    data["readout_2_weights_2"] = [0.0] * hidden_size
    model_path = tmp_path / f"readout-hidden-{evaluator_name}.json"
    model_path.write_text(json.dumps(data))

    getattr(native_symmetrix, evaluator_name)(str(model_path))


@pytest.mark.parametrize("evaluator_name", ["MACE", "MACEKokkos"])
def test_residual_first_loader_rejects_missing_weights(
    residual_first_macefield_artifacts,
    tmp_path,
    evaluator_name,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos" and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    data = dict(residual_first_macefield_artifacts["data"])
    data.pop("H1_first_residual_weights")
    path = tmp_path / "missing-first-residual.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="invalid H1 residual extent"):
        getattr(native_symmetrix, evaluator_name)(str(path))


def _rank2_upstream_transform(model, h1_pre, electric_field):
    return h1_pre - model.field_linear[0](model.field_feats[0](h1_pre, electric_field))


def test_rank2_field_schema_contains_all_angular_paths(rank2_macefield_artifacts):
    data = rank2_macefield_artifacts["data"]
    coupling = data["field_couplings"][0]

    assert data["L_max"] == 2
    assert coupling["schema_version"] == 1
    assert coupling["field_feats_irreps_in1"] == "4x0e+4x1o+4x2e"
    assert len(coupling["field_feats_instructions"]) == 4
    assert {
        (instruction["i_in1"], instruction["i_out"])
        for instruction in coupling["field_feats_instructions"]
    } == {(0, 1), (1, 0), (1, 2), (2, 1)}
    for instruction in coupling["field_feats_instructions"]:
        l_in = instruction["i_in1"]
        l_out = instruction["i_out"]
        assert instruction["wigner_3j_shape"] == [2 * l_in + 1, 3, 2 * l_out + 1]
        assert len(instruction["wigner_3j"]) == (2 * l_in + 1) * 3 * (2 * l_out + 1)


@pytest.mark.parametrize("evaluator_name", ["MACE", "MACEKokkos"])
def test_field_loader_rejects_L_max_above_two(
    rank2_macefield_artifacts,
    tmp_path,
    evaluator_name,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos" and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    data = dict(rank2_macefield_artifacts["data"])
    data["L_max"] = 3
    json_path = tmp_path / "unsupported-rank3-macefield.json"
    json_path.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match=r"1 <= L_max <= 2"):
        getattr(native_symmetrix, evaluator_name)(str(json_path))


@pytest.mark.parametrize("evaluator_name", ["MACE", "MACEKokkos"])
def test_rank2_field_h1_matches_upstream(
    rank2_macefield_artifacts,
    evaluator_name,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos" and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    model = rank2_macefield_artifacts["model"]
    evaluator = getattr(native_symmetrix, evaluator_name)(
        str(rank2_macefield_artifacts["json_path"])
    )
    generator = torch.Generator(device="cpu").manual_seed(20260806)
    h1_pre = torch.randn((3, 36), dtype=torch.float64, generator=generator)
    electric_field = torch.randn((3, 3), dtype=torch.float64, generator=generator)
    expected = _rank2_upstream_transform(model, h1_pre, electric_field)

    evaluator.H1 = _compact_to_native_h1(h1_pre, 4, 2)
    evaluator.compute_field_H1(3, electric_field.numpy(force=True).reshape(-1))
    actual = _native_to_compact_h1(evaluator.H1, 3, 4, 2)

    assert np.allclose(actual, expected.numpy(force=True), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("evaluator_name", ["MACE", "MACEKokkos"])
def test_rank2_reverse_field_h1_matches_autograd(
    rank2_macefield_artifacts,
    evaluator_name,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos" and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    model = rank2_macefield_artifacts["model"]
    evaluator = getattr(native_symmetrix, evaluator_name)(
        str(rank2_macefield_artifacts["json_path"])
    )
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    h1_pre = torch.randn(
        (2, 36), dtype=torch.float64, generator=generator, requires_grad=True
    )
    electric_field = torch.randn(
        (2, 3), dtype=torch.float64, generator=generator, requires_grad=True
    )
    h1_post_adj = torch.randn((2, 36), dtype=torch.float64, generator=generator)
    h1_post = _rank2_upstream_transform(model, h1_pre, electric_field)
    torch.sum(h1_post * h1_post_adj).backward()

    evaluator.H1 = _compact_to_native_h1(h1_pre.detach(), 4, 2)
    evaluator.compute_field_H1(2, electric_field.detach().numpy(force=True).reshape(-1))
    evaluator.H1_adj = _compact_to_native_h1(h1_post_adj, 4, 2)
    evaluator.reverse_field_H1(2, electric_field.detach().numpy(force=True).reshape(-1))

    actual_h1_adj = _native_to_compact_h1(evaluator.H1_adj, 2, 4, 2)
    actual_field_adj = np.asarray(evaluator.electric_field_adj).reshape(2, 3)
    assert np.allclose(
        actual_h1_adj, h1_pre.grad.numpy(force=True), atol=1e-12, rtol=1e-12
    )
    assert np.allclose(
        actual_field_adj,
        electric_field.grad.numpy(force=True),
        atol=1e-12,
        rtol=1e-12,
    )


def test_native_compute_field_h1_matches_standalone_transform(macefield_json_path):
    evaluator = native_symmetrix.MACE(str(macefield_json_path))

    assert evaluator.has_field_coupling is True

    h1_pre, electric_field = _frozen_inputs()
    expected = _standalone_field_transform(
        _compact_field_coupling_from_json(macefield_json_path),
        h1_pre,
        electric_field,
    ).numpy(force=True)

    evaluator.H1 = _compact_to_native_h1(h1_pre.numpy(force=True)).tolist()
    evaluator.compute_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )

    actual = _native_to_compact_h1(evaluator.H1, h1_pre.shape[0])
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_legacy_unversioned_rank1_field_json_matches_versioned(
    macefield_json_path,
    tmp_path,
):
    data = json.loads(macefield_json_path.read_text())
    coupling = data["field_couplings"][0]
    coupling.pop("schema_version")
    for instruction in coupling["field_feats_instructions"]:
        instruction.pop("wigner_3j_shape")
        instruction.pop("wigner_3j")
    legacy_path = tmp_path / "legacy-macefield.json"
    legacy_path.write_text(json.dumps(data))

    h1_pre, electric_field = _frozen_inputs()
    versioned = native_symmetrix.MACE(str(macefield_json_path))
    legacy = native_symmetrix.MACE(str(legacy_path))
    native_h1 = _compact_to_native_h1(h1_pre.numpy(force=True)).tolist()
    versioned.H1 = native_h1
    legacy.H1 = native_h1
    field = electric_field.numpy(force=True).reshape(-1)
    versioned.compute_field_H1(h1_pre.shape[0], field)
    legacy.compute_field_H1(h1_pre.shape[0], field)

    assert np.allclose(legacy.H1, versioned.H1, atol=1e-12, rtol=1e-12)


def test_native_reverse_field_h1_matches_torch_autograd(macefield_json_path):
    evaluator = native_symmetrix.MACE(str(macefield_json_path))
    coupling = _compact_field_coupling_from_json(macefield_json_path)

    h1_pre, electric_field = _frozen_inputs()
    h1_pre = h1_pre.detach().clone().requires_grad_(True)
    electric_field = electric_field.detach().clone().requires_grad_(True)
    generator = torch.Generator(device="cpu").manual_seed(20260711)
    h1_post_adj = torch.randn(h1_pre.shape, dtype=torch.float64, generator=generator)

    h1_post = _standalone_field_transform(coupling, h1_pre, electric_field)
    torch.sum(h1_post * h1_post_adj).backward()

    evaluator.H1 = _compact_to_native_h1(h1_pre.detach().numpy(force=True)).tolist()
    evaluator.compute_field_H1(
        h1_pre.shape[0], electric_field.detach().numpy(force=True).reshape(-1)
    )
    evaluator.H1_adj = _compact_to_native_h1(h1_post_adj.numpy(force=True)).tolist()
    evaluator.reverse_field_H1(
        h1_pre.shape[0], electric_field.detach().numpy(force=True).reshape(-1)
    )

    actual_h1_adj = _native_to_compact_h1(evaluator.H1_adj, h1_pre.shape[0])
    actual_field_adj = np.asarray(
        evaluator.electric_field_adj, dtype=np.float64
    ).reshape(h1_pre.shape[0], 3)

    assert np.allclose(
        actual_h1_adj, h1_pre.grad.numpy(force=True), atol=1e-12, rtol=1e-12
    )
    assert np.allclose(
        actual_field_adj, electric_field.grad.numpy(force=True), atol=1e-12, rtol=1e-12
    )


def test_kokkos_compute_field_h1_matches_native_transform(macefield_json_path):
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    evaluator = native_symmetrix.MACEKokkos(str(macefield_json_path))
    assert evaluator.has_field_coupling is True

    h1_pre, electric_field = _frozen_inputs()
    expected = _standalone_field_transform(
        _compact_field_coupling_from_json(macefield_json_path),
        h1_pre,
        electric_field,
    ).numpy(force=True)

    evaluator.H1 = _compact_to_native_h1(h1_pre.numpy(force=True))
    evaluator.compute_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )

    actual = _native_to_compact_h1(evaluator.H1, h1_pre.shape[0])
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_kokkos_reverse_field_h1_matches_native_reverse(macefield_json_path):
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    h1_pre, electric_field = _frozen_inputs()
    generator = torch.Generator(device="cpu").manual_seed(20260711)
    h1_post_adj = torch.randn(h1_pre.shape, dtype=torch.float64, generator=generator)

    native = native_symmetrix.MACE(str(macefield_json_path))
    native.H1 = _compact_to_native_h1(h1_pre.numpy(force=True)).tolist()
    native.compute_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )
    native.H1_adj = _compact_to_native_h1(h1_post_adj.numpy(force=True)).tolist()
    native.reverse_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )

    kokkos = native_symmetrix.MACEKokkos(str(macefield_json_path))
    kokkos.H1 = _compact_to_native_h1(h1_pre.numpy(force=True))
    kokkos.compute_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )
    kokkos.H1_adj = _compact_to_native_h1(h1_post_adj.numpy(force=True))
    kokkos.reverse_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )

    assert np.allclose(kokkos.H1_adj, native.H1_adj, atol=1e-12, rtol=1e-12)
    assert np.allclose(
        kokkos.electric_field_adj, native.electric_field_adj, atol=1e-12, rtol=1e-12
    )


def test_native_field_energy_forces_match_ase_macefield(
    macefield_full_json_path, macefield_model_path
):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    atoms.info["electric_field"] = electric_field

    calc_torch = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )
    atoms.calc = calc_torch
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()

    evaluator = native_symmetrix.MACE(str(macefield_full_json_path))
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = [
        mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)
    ]
    num_neigh = np.bincount(j_list, minlength=num_nodes)
    neigh_types = [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list]
    per_atom_field = np.tile(electric_field, (num_nodes, 1))

    evaluator.compute_node_energies_forces_field(
        num_nodes,
        np.asarray(node_types, dtype=np.int32),
        np.asarray(num_neigh, dtype=np.int32),
        np.asarray(j_list, dtype=np.int32),
        np.asarray(neigh_types, dtype=np.int32),
        xyz.reshape(-1),
        r,
        per_atom_field.reshape(-1),
    )

    native_energy = np.sum(evaluator.node_energies)
    pair_forces = np.asarray(evaluator.node_forces).reshape((-1, 3))[: len(i_list), :]
    native_forces = np.zeros((num_nodes, 3))
    for component in range(3):
        native_forces[:, component] = np.bincount(
            j_list, weights=pair_forces[:, component], minlength=num_nodes
        ) - np.bincount(i_list, weights=pair_forces[:, component], minlength=num_nodes)

    assert np.allclose(native_energy, expected_energy, atol=1e-3)
    assert np.allclose(native_forces, expected_forces, atol=2e-3)


def _skip_without_kokkos():
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()


def _load_direct_artifact(evaluator, model_path, dtype):
    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space in ("OpenMP", "Serial"):
        artifact = prepare_jit_host_artifact(model_path, precision=dtype)
        evaluator._load_jit_host_plugin(str(artifact.artifact_path))
    elif execution_space in ("Cuda", "HIP"):
        backend = execution_space.lower()
        artifact = prepare_jit_device_artifact(
            model_path, precision=dtype, backend=backend
        )
        loader = getattr(evaluator, f"_load_jit_{backend}_plugin")
        if backend == "cuda":
            loader(
                str(artifact.artifact_path),
                artifact.edge_policy["persistent_blocks_per_compute_unit"],
            )
        else:
            loader(str(artifact.artifact_path))


def _field_backend_inputs(evaluator):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)],
        dtype=np.int32,
    )
    num_neigh = np.asarray(np.bincount(j_list, minlength=num_nodes), dtype=np.int32)
    neigh_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list], dtype=np.int32
    )
    neigh_indices = np.asarray(j_list, dtype=np.int32)
    return num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, i_list


@pytest.mark.parametrize(
    "evaluator_name,mode",
    [
        ("MACE", "materialized"),
        ("MACE", "generic"),
        ("MACEKokkos", "materialized"),
        ("MACEKokkos", "generic"),
    ],
)
def test_residual_first_field_energy_forces_match_pytorch(
    residual_first_macefield_artifacts,
    evaluator_name,
    mode,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos":
        _skip_without_kokkos()

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.013, -0.021, 0.008], dtype=np.float64)
    atoms.info["electric_field"] = electric_field
    atoms.calc = MACECalculator(
        model_paths=[str(residual_first_macefield_artifacts["checkpoint_path"])],
        model_type="MACEField",
        head="test",
        device="cpu",
        default_dtype="float64",
    )
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()

    evaluator = getattr(native_symmetrix, evaluator_name)(
        str(residual_first_macefield_artifacts["json_path"])
    )
    evaluator.set_streamed_edges(mode)
    assert evaluator.first_interaction_residual is True
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, i_list = (
        _field_backend_inputs(evaluator)
    )
    evaluator.compute_node_energies_forces_field(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    actual_energy = np.sum(evaluator.node_energies)
    pair_forces = np.asarray(evaluator.node_forces).reshape((-1, 3))[: len(i_list)]
    actual_forces = np.zeros((num_nodes, 3), dtype=np.float64)
    for component in range(3):
        actual_forces[:, component] = np.bincount(
            neigh_indices,
            weights=pair_forces[:, component],
            minlength=num_nodes,
        ) - np.bincount(
            i_list,
            weights=pair_forces[:, component],
            minlength=num_nodes,
        )

    assert actual_energy == pytest.approx(expected_energy, abs=2e-3)
    assert np.allclose(actual_forces, expected_forces, atol=3e-3, rtol=3e-3)


def test_residual_first_parameter_gradients_reject(
    residual_first_macefield_artifacts,
):
    _skip_without_kokkos()
    evaluator = native_symmetrix.MACEKokkos(
        str(residual_first_macefield_artifacts["json_path"])
    )
    evaluator.set_streamed_edges("receiver_factorized")
    with pytest.raises(ValueError, match="do not yet support residual-first"):
        evaluator.set_execution_parameter_gradients(True)


@pytest.mark.parametrize("evaluator_name", ["MACE", "MACEKokkos"])
def test_residual_first_legacy_pair_splines_match_compact(
    residual_first_macefield_artifacts,
    evaluator_name,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos":
        _skip_without_kokkos()

    compact = getattr(native_symmetrix, evaluator_name)(
        str(residual_first_macefield_artifacts["json_path"])
    )
    legacy = getattr(native_symmetrix, evaluator_name)(
        str(residual_first_macefield_artifacts["legacy_json_path"])
    )
    compact.set_streamed_edges("materialized")
    legacy.set_streamed_edges("materialized")
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, _ = (
        _field_backend_inputs(compact)
    )
    args = (
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        np.array([0.013, -0.021, 0.008], dtype=np.float64),
    )
    compact.compute_node_energies_forces_field(*args)
    legacy.compute_node_energies_forces_field(*args)

    assert np.allclose(legacy.node_energies, compact.node_energies, atol=2e-6)
    assert np.allclose(legacy.node_forces, compact.node_forces, atol=2e-6)
    assert np.allclose(
        legacy.electric_field_adj,
        compact.electric_field_adj,
        atol=2e-6,
    )


def test_density_residual_first_field_matches_pytorch(tmp_path):
    from e3nn import o3
    from mace.modules.blocks import RealAgnosticDensityResidualInteractionBlock

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(20260807)
        model = MACEField(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=4,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=RealAgnosticDensityResidualInteractionBlock,
            interaction_cls_first=RealAgnosticDensityResidualInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("4x0e+4x1o"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros((1, 2)),
            avg_num_neighbors=4.0,
            atomic_numbers=[7, 13],
            correlation=3,
            gate=torch.nn.functional.silu,
            heads=["test"],
        )
        model.eval()
        checkpoint_path = tmp_path / "density-residual-first.model"
        torch.save(model, checkpoint_path)
        data = extract_mace_data(
            checkpoint_path,
            species=[7, 13],
            head="test",
            num_spline_points=256,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    model_path = tmp_path / "density-residual-first.json"
    model_path.write_text(json.dumps(data))
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.013, -0.021, 0.008], dtype=np.float64)
    atoms.info["electric_field"] = electric_field
    atoms.calc = MACECalculator(
        models=[model],
        model_type="MACEField",
        head="test",
        device="cpu",
        default_dtype="float64",
    )
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()

    evaluator = native_symmetrix.MACE(str(model_path))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, i_list = (
        _field_backend_inputs(evaluator)
    )
    evaluator.compute_node_energies_forces_field(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    pair_forces = np.asarray(evaluator.node_forces).reshape((-1, 3))[: len(i_list)]
    actual_forces = np.zeros((num_nodes, 3), dtype=np.float64)
    for component in range(3):
        actual_forces[:, component] = np.bincount(
            neigh_indices,
            weights=pair_forces[:, component],
            minlength=num_nodes,
        ) - np.bincount(
            i_list,
            weights=pair_forces[:, component],
            minlength=num_nodes,
        )

    assert data["first_interaction_residual"] is True
    assert data["A0_scaled"] is True
    assert np.sum(evaluator.node_energies) == pytest.approx(expected_energy, abs=2e-3)
    assert np.allclose(actual_forces, expected_forces, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize(
    "evaluator_name,mode",
    [
        ("MACE", "materialized"),
        ("MACEKokkos", "generic"),
    ],
)
def test_residual_first_ordinary_fused_path_matches_zero_field(
    residual_first_macefield_artifacts,
    tmp_path,
    evaluator_name,
    mode,
):
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip(f"Symmetrix was built without {evaluator_name}.")
    if evaluator_name == "MACEKokkos":
        _skip_without_kokkos()

    data = json.loads(json.dumps(residual_first_macefield_artifacts["data"]))
    data["model_type"] = "MACE"
    data["has_field_coupling"] = False
    data["field_couplings"] = []
    model_path = tmp_path / "residual-first-ordinary.json"
    model_path.write_text(json.dumps(data))

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.zeros(3, dtype=np.float64)
    atoms.calc = MACECalculator(
        model_paths=[str(residual_first_macefield_artifacts["checkpoint_path"])],
        model_type="MACEField",
        head="test",
        device="cpu",
        default_dtype="float64",
    )
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()

    evaluator = getattr(native_symmetrix, evaluator_name)(str(model_path))
    evaluator.set_streamed_edges(mode)
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, i_list = (
        _field_backend_inputs(evaluator)
    )
    evaluator.compute_node_energies_forces(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
    )
    actual_energy = np.sum(evaluator.node_energies)
    pair_forces = np.asarray(evaluator.node_forces).reshape((-1, 3))[: len(i_list)]
    actual_forces = np.zeros((num_nodes, 3), dtype=np.float64)
    for component in range(3):
        actual_forces[:, component] = np.bincount(
            neigh_indices,
            weights=pair_forces[:, component],
            minlength=num_nodes,
        ) - np.bincount(
            i_list,
            weights=pair_forces[:, component],
            minlength=num_nodes,
        )

    assert actual_energy == pytest.approx(expected_energy, abs=2e-3)
    assert np.allclose(actual_forces, expected_forces, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize(
    "artifact_fixture",
    ["rank2_macefield_artifacts", "residual_first_macefield_artifacts"],
)
def test_analytic_field_responses(request, artifact_fixture):
    _skip_without_kokkos()
    artifacts = request.getfixturevalue(artifact_fixture)
    model_path = artifacts["json_path"]
    native = native_symmetrix.MACE(str(model_path))
    kokkos = native_symmetrix.MACEKokkos(str(model_path))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, _ = (
        _field_backend_inputs(native)
    )
    electric_field = np.array([0.013, -0.021, 0.008], dtype=np.float64)
    args = (
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
    )

    native.compute_electric_field_hessian(*args, electric_field)
    native_hessian = np.asarray(native.electric_field_hessian, dtype=np.float64)
    native.compute_electric_field_force_derivative(*args, electric_field)
    native_force_derivative = np.asarray(
        native.electric_field_force_derivative, dtype=np.float64
    )
    kokkos.compute_electric_field_hessian(*args, electric_field)
    kokkos.compute_electric_field_force_derivative(*args, electric_field)

    assert np.allclose(
        kokkos.electric_field_hessian,
        native_hessian,
        atol=2e-8,
        rtol=2e-8,
    )
    assert np.allclose(
        kokkos.electric_field_force_derivative,
        native_force_derivative,
        atol=2e-8,
        rtol=2e-8,
    )

    step = 1e-4
    finite_difference = np.zeros((3, 3), dtype=np.float64)
    for seed in range(3):
        field_plus = electric_field.copy()
        field_minus = electric_field.copy()
        field_plus[seed] += step
        field_minus[seed] -= step
        native.compute_node_energies_forces_field(*args, field_plus)
        adj_plus = np.asarray(native.electric_field_adj, dtype=np.float64)
        native.compute_node_energies_forces_field(*args, field_minus)
        adj_minus = np.asarray(native.electric_field_adj, dtype=np.float64)
        finite_difference[:, seed] = (adj_plus - adj_minus) / (2.0 * step)

    assert np.allclose(
        native_hessian.reshape(3, 3),
        finite_difference,
        atol=2e-7,
        rtol=2e-5,
    )


def test_rank2_field_energy_forces_match_ase_macefield(
    rank2_macefield_artifacts,
):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.013, -0.021, 0.008], dtype=np.float64)
    atoms.info["electric_field"] = electric_field
    atoms.calc = MACECalculator(
        model_paths=[str(rank2_macefield_artifacts["checkpoint_path"])],
        model_type="MACEField",
        head="test",
        device="cpu",
        default_dtype="float64",
    )
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()

    native = native_symmetrix.MACE(str(rank2_macefield_artifacts["json_path"]))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, i_list = (
        _field_backend_inputs(native)
    )
    native.compute_node_energies_forces_field(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    actual_energy = np.sum(native.node_energies)
    pair_forces = np.asarray(native.node_forces).reshape((-1, 3))[: len(i_list)]
    actual_forces = np.zeros((num_nodes, 3), dtype=np.float64)
    for component in range(3):
        actual_forces[:, component] = np.bincount(
            neigh_indices,
            weights=pair_forces[:, component],
            minlength=num_nodes,
        ) - np.bincount(
            i_list,
            weights=pair_forces[:, component],
            minlength=num_nodes,
        )

    assert actual_energy == pytest.approx(expected_energy, abs=2e-3)
    assert np.allclose(actual_forces, expected_forces, atol=3e-3, rtol=3e-3)


def test_kokkos_field_energy_forces_match_native(macefield_full_json_path):
    _skip_without_kokkos()

    electric_field = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    native = native_symmetrix.MACE(str(macefield_full_json_path))
    kokkos = native_symmetrix.MACEKokkos(str(macefield_full_json_path))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, _ = (
        _field_backend_inputs(native)
    )

    native.compute_node_energies_forces_field(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    kokkos.compute_node_energies_forces_field(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )

    assert np.sum(kokkos.node_energies) == pytest.approx(
        np.sum(native.node_energies), abs=1e-8
    )
    assert np.allclose(kokkos.node_forces, native.node_forces, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("mode", ["generic"])
def test_kokkos_streamed_field_outputs_match_legacy(
    macefield_full_json_path,
    mode,
):
    _skip_without_kokkos()

    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    legacy = native_symmetrix.MACEKokkos(str(macefield_full_json_path))
    streamed = native_symmetrix.MACEKokkos(str(macefield_full_json_path))
    streamed.set_streamed_edges(mode)
    inputs = _field_backend_inputs(legacy)[:7]

    legacy.compute_node_energies_forces_field(*inputs, electric_field)
    streamed.compute_node_energies_forces_field(*inputs, electric_field)

    np.testing.assert_allclose(
        streamed.node_energies, legacy.node_energies, rtol=0.0, atol=3e-11
    )
    np.testing.assert_allclose(
        streamed.node_forces, legacy.node_forces, rtol=0.0, atol=3e-11
    )
    np.testing.assert_allclose(
        streamed.electric_field_adj,
        legacy.electric_field_adj,
        rtol=0.0,
        atol=3e-11,
    )
    assert streamed.R0_storage_size == 0
    assert streamed.R1_storage_size == 0


@pytest.mark.parametrize(
    ("kokkos_class", "atol"),
    (("MACEKokkosFloat", 5e-4), ("MACEKokkos", 5e-10)),
)
def test_kokkos_execution_field_standard_m0_r0_v2_lifecycle(
    macefield_full_json_path,
    kokkos_class,
    atol,
):
    _skip_without_kokkos()

    evaluator = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    dtype = "float32" if kokkos_class == "MACEKokkosFloat" else "float64"
    _load_direct_artifact(evaluator, macefield_full_json_path, dtype)
    evaluator.set_streamed_edges("direct")
    evaluator._set_factorized_source_strategy("jit_plugin")
    evaluator._set_factorized_direct_forward_executor("jit_all")
    evaluator._set_factorized_direct_reverse_executor("jit")
    inputs = _field_backend_inputs(evaluator)[:7]
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    assert evaluator.standard_m0_module_ready
    assert evaluator.standard_m0_module_fallback_reason == ""
    assert evaluator.standard_r0_module_ready
    assert evaluator.standard_r0_module_fallback_reason == ""

    def evaluate(m0_executor, r0_executor):
        evaluator._set_standard_m0_executor(m0_executor)
        evaluator._set_standard_r0_executor(r0_executor)
        launches = (
            evaluator.standard_m0_module_forward_launch_count,
            evaluator.standard_m0_module_reverse_launch_count,
            evaluator.standard_r0_module_launch_count,
        )
        evaluator.compute_node_energies_forces_field(*inputs, electric_field)
        return {
            "energy": float(np.sum(evaluator.node_energies)),
            "energies": np.array(evaluator.node_energies, copy=True),
            "forces": np.array(evaluator.node_forces, copy=True),
            "field_adj": np.array(evaluator.electric_field_adj, copy=True),
            "M0": np.array(evaluator.M0, copy=True),
            "A0_adj": np.array(evaluator.A0_adj, copy=True),
            "m0": evaluator.standard_m0_selected_executor,
            "r0": evaluator.standard_r0_selected_executor,
            "m0_forward": (
                evaluator.standard_m0_module_forward_launch_count - launches[0]
            ),
            "m0_reverse": (
                evaluator.standard_m0_module_reverse_launch_count - launches[1]
            ),
            "r0_launches": evaluator.standard_r0_module_launch_count - launches[2],
            "m0_values": evaluator.standard_m0_poly_values_active_bytes,
            "m0_adjoints": evaluator.standard_m0_poly_adjoints_active_bytes,
        }

    control = evaluate("runtime", "v1")
    m0_only = evaluate("standard", "v1")
    r0_only = evaluate("runtime", "v2_edge16")
    combined = evaluate("standard", "v2_edge16")

    assert (control["m0"], control["r0"]) == ("runtime", "v1")
    assert (control["m0_forward"], control["m0_reverse"], control["r0_launches"]) == (
        0,
        0,
        0,
    )
    assert control["m0_values"] > 0
    assert control["m0_adjoints"] > 0
    assert (m0_only["m0"], m0_only["r0"]) == ("standard", "v1")
    assert (m0_only["m0_forward"], m0_only["m0_reverse"], m0_only["r0_launches"]) == (
        1,
        1,
        0,
    )
    assert m0_only["m0_values"] == 0
    assert m0_only["m0_adjoints"] == 0
    assert (r0_only["m0"], r0_only["r0"]) == ("runtime", "v2_edge16")
    assert (r0_only["m0_forward"], r0_only["m0_reverse"], r0_only["r0_launches"]) == (
        0,
        0,
        4,
    )
    assert (combined["m0"], combined["r0"]) == ("standard", "v2_edge16")
    assert (
        combined["m0_forward"],
        combined["m0_reverse"],
        combined["r0_launches"],
    ) == (1, 1, 4)

    for candidate in (m0_only, r0_only, combined):
        assert candidate["energy"] == pytest.approx(
            control["energy"], rel=0.0, abs=atol
        )
        for name in ("energies", "forces", "field_adj", "M0", "A0_adj"):
            np.testing.assert_allclose(
                candidate[name], control[name], rtol=0.0, atol=atol
            )

    repeated = evaluate("standard", "v2_edge16")
    assert repeated["energy"] == combined["energy"]
    for name in ("energies", "forces", "field_adj", "M0", "A0_adj"):
        np.testing.assert_array_equal(repeated[name], combined[name])

    for executor in ("v2_receiver", "v2_edge32"):
        variant = evaluate("standard", executor)
        assert variant["r0"] == executor
        assert variant["r0_launches"] == 4
        for name in ("energies", "forces", "field_adj", "M0", "A0_adj"):
            np.testing.assert_allclose(
                variant[name], combined[name], rtol=0.0, atol=atol
            )

    rollback = evaluate("runtime", "v1")
    assert rollback["energy"] == control["energy"]
    for name in ("energies", "forces", "field_adj", "M0", "A0_adj"):
        np.testing.assert_array_equal(rollback[name], control[name])

    automatic = evaluate("automatic", "automatic")
    automatic_r0 = (
        "v2_receiver"
        if native_symmetrix._kokkos_default_execution_space() in ("OpenMP", "Serial")
        else "v2_edge16"
    )
    assert (automatic["m0"], automatic["r0"]) == (
        "standard",
        automatic_r0,
    )
    for name in ("energies", "forces", "field_adj", "M0", "A0_adj"):
        np.testing.assert_allclose(automatic[name], control[name], rtol=0.0, atol=atol)

    evaluator._set_standard_m0_executor("runtime")
    evaluator._set_standard_r0_executor("v1")
    evaluator.compute_electric_field_force_derivative(*inputs, electric_field)
    response_control = (
        np.array(evaluator.electric_field_hessian, copy=True),
        np.array(evaluator.electric_field_force_derivative, copy=True),
    )
    evaluator._set_standard_m0_executor("automatic")
    evaluator._set_standard_r0_executor("automatic")
    evaluator.compute_electric_field_force_derivative(*inputs, electric_field)
    np.testing.assert_allclose(
        evaluator.electric_field_hessian,
        response_control[0],
        rtol=0.0,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        evaluator.electric_field_force_derivative,
        response_control[1],
        rtol=0.0,
        atol=5e-4,
    )


@pytest.mark.parametrize("mode", ["generic"])
@pytest.mark.parametrize(
    "kokkos_class,m1_atol",
    [("MACEKokkosFloat", 5e-4), ("MACEKokkos", 3e-11)],
)
def test_kokkos_field_standard_m0_m1_lifecycle(
    macefield_full_json_path,
    mode,
    kokkos_class,
    m1_atol,
):
    _skip_without_kokkos()

    evaluator = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    evaluator._set_m1_polynomial_policy("retained")
    evaluator.set_streamed_edges(mode)
    inputs = _field_backend_inputs(evaluator)[:7]
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    def evaluate(executor):
        evaluator._set_standard_m0_executor(executor)
        launches = (
            evaluator.standard_m0_module_forward_launch_count,
            evaluator.standard_m0_module_reverse_launch_count,
        )
        evaluator.compute_node_energies_forces_field(*inputs, electric_field)
        return {
            "energy": float(np.sum(evaluator.node_energies)),
            "energies": np.array(evaluator.node_energies, copy=True),
            "forces": np.array(evaluator.node_forces, copy=True),
            "field_adj": np.array(evaluator.electric_field_adj, copy=True),
            "M0": np.array(evaluator.M0, copy=True),
            "A0_adj": np.array(evaluator.A0_adj, copy=True),
            "selected": evaluator.standard_m0_selected_executor,
            "forward": (
                evaluator.standard_m0_module_forward_launch_count - launches[0]
            ),
            "reverse": (
                evaluator.standard_m0_module_reverse_launch_count - launches[1]
            ),
            "value_capacity": evaluator.standard_m0_poly_values_capacity_bytes,
            "adjoint_capacity": evaluator.standard_m0_poly_adjoints_capacity_bytes,
        }

    control = evaluate("runtime")
    standard = evaluate("standard")
    assert standard["selected"] == "standard"
    assert (standard["forward"], standard["reverse"]) == (1, 1)
    assert standard["value_capacity"] == 0
    assert standard["adjoint_capacity"] == 0
    assert standard["energy"] == pytest.approx(control["energy"], abs=5e-4)
    for name in ("energies", "forces", "field_adj", "M0", "A0_adj"):
        np.testing.assert_allclose(standard[name], control[name], rtol=0.0, atol=5e-4)

    repeated = evaluate("standard")
    assert repeated["energy"] == standard["energy"]
    for name in ("energies", "forces", "field_adj", "M0", "A0_adj"):
        np.testing.assert_allclose(repeated[name], standard[name], rtol=0.0, atol=5e-4)

    evaluator._set_standard_m0_executor("runtime")
    evaluator.compute_electric_field_hessian(*inputs, electric_field)
    response_hessian = np.array(evaluator.electric_field_hessian, copy=True)
    assert np.asarray(evaluator.electric_field_force_derivative).size == 0
    evaluator.compute_electric_field_force_derivative(*inputs, electric_field)
    response_force_derivative = np.array(
        evaluator.electric_field_force_derivative, copy=True
    )
    evaluator._set_standard_m0_executor("standard")
    response_launches = evaluator.standard_m0_module_reverse_launch_count
    evaluator.compute_electric_field_hessian(*inputs, electric_field)
    assert evaluator.standard_m0_module_reverse_launch_count - response_launches == 1
    assert np.asarray(evaluator.electric_field_force_derivative).size == 0
    assert evaluator.standard_m0_poly_values_capacity_bytes == 0
    assert evaluator.standard_m0_poly_adjoints_capacity_bytes == 0
    np.testing.assert_allclose(
        evaluator.electric_field_hessian,
        response_hessian,
        rtol=0.0,
        atol=2e-4,
    )
    response_launches = evaluator.standard_m0_module_reverse_launch_count
    evaluator.compute_electric_field_force_derivative(*inputs, electric_field)
    assert evaluator.standard_m0_module_reverse_launch_count - response_launches == 4
    np.testing.assert_allclose(
        evaluator.electric_field_force_derivative,
        response_force_derivative,
        rtol=0.0,
        atol=5e-4,
    )

    restored = evaluate("runtime")
    assert restored["energy"] == control["energy"]
    assert restored["value_capacity"] > 0
    assert restored["adjoint_capacity"] > 0
    for name in ("energies", "forces", "field_adj", "M0", "A0_adj"):
        np.testing.assert_allclose(restored[name], control[name], rtol=0.0, atol=5e-4)

    retained_value_capacity = evaluator.m1_poly_values_capacity_bytes
    retained_adjoint_capacity = evaluator.m1_poly_adjoints_capacity_bytes
    assert retained_value_capacity > 0
    assert retained_adjoint_capacity > 0

    recomputed = {}
    for requested_tile in (8, 16, 32):
        evaluator._set_m1_polynomial_policy("retained")
        evaluator._set_m1_recompute_tile_channels(requested_tile)
        evaluator._set_m1_polynomial_policy("recompute")
        selected_tile = evaluator.m1_recompute_tile_channels
        assert selected_tile in (8, 16, 32)
        assert selected_tile <= requested_tile
        assert evaluator.macefield_response_m1_recompute_tile_channels == selected_tile
        launches = (
            evaluator.macefield_response_m1_recompute_forward_launch_count,
            evaluator.macefield_response_m1_recompute_reverse_launch_count,
        )
        evaluator.compute_electric_field_hessian(*inputs, electric_field)
        result = {
            "hessian": np.array(evaluator.electric_field_hessian, copy=True),
            "forward": (
                evaluator.macefield_response_m1_recompute_forward_launch_count
                - launches[0]
            ),
            "reverse": (
                evaluator.macefield_response_m1_recompute_reverse_launch_count
                - launches[1]
            ),
        }
        assert result["forward"] == 3
        assert result["reverse"] == 3
        assert evaluator.m1_poly_values_active_bytes == 0
        assert evaluator.m1_poly_values_capacity_bytes == 0
        assert evaluator.m1_poly_adjoints_active_bytes == 0
        assert evaluator.m1_poly_adjoints_capacity_bytes == 0
        assert evaluator.macefield_response_m1_recompute_scratch_bytes > 0
        assert (
            evaluator.macefield_response_m1_recompute_scratch_bytes
            <= evaluator.macefield_response_m1_recompute_scratch_limit_bytes
        )
        np.testing.assert_allclose(
            result["hessian"], response_hessian, rtol=0.0, atol=m1_atol
        )

        launches = (
            evaluator.macefield_response_m1_recompute_forward_launch_count,
            evaluator.macefield_response_m1_recompute_reverse_launch_count,
        )
        evaluator.compute_electric_field_force_derivative(*inputs, electric_field)
        result["force_hessian"] = np.array(evaluator.electric_field_hessian, copy=True)
        result["force_derivative"] = np.array(
            evaluator.electric_field_force_derivative, copy=True
        )
        assert (
            evaluator.macefield_response_m1_recompute_forward_launch_count - launches[0]
        ) == 3
        assert (
            evaluator.macefield_response_m1_recompute_reverse_launch_count - launches[1]
        ) == 3
        np.testing.assert_allclose(
            result["force_derivative"],
            response_force_derivative,
            rtol=0.0,
            atol=m1_atol,
        )
        np.testing.assert_allclose(
            result["force_hessian"],
            result["hessian"],
            rtol=0.0,
            atol=m1_atol,
        )
        recomputed[selected_tile] = result

    selected_tile = max(recomputed)
    evaluator._set_m1_polynomial_policy("retained")
    evaluator._set_m1_recompute_tile_channels(selected_tile)
    evaluator._set_m1_polynomial_policy("recompute")
    evaluator.compute_electric_field_force_derivative(*inputs, electric_field)
    np.testing.assert_allclose(
        evaluator.electric_field_hessian,
        recomputed[selected_tile]["force_hessian"],
        rtol=0.0,
        atol=m1_atol,
    )
    np.testing.assert_allclose(
        evaluator.electric_field_force_derivative,
        recomputed[selected_tile]["force_derivative"],
        rtol=0.0,
        atol=m1_atol,
    )

    evaluator._set_m1_polynomial_policy("retained")
    evaluator.compute_electric_field_force_derivative(*inputs, electric_field)
    assert evaluator.m1_poly_values_capacity_bytes == retained_value_capacity
    assert evaluator.m1_poly_adjoints_capacity_bytes == retained_adjoint_capacity
    np.testing.assert_allclose(
        evaluator.electric_field_hessian,
        response_hessian,
        rtol=0.0,
        atol=m1_atol,
    )
    np.testing.assert_allclose(
        evaluator.electric_field_force_derivative,
        response_force_derivative,
        rtol=0.0,
        atol=m1_atol,
    )


@pytest.mark.parametrize("kokkos_class", ["MACEKokkosFloat", "MACEKokkos"])
def test_kokkos_field_m1_recompute_rejects_oversized_scratch(
    macefield_full_json_path,
    kokkos_class,
):
    _skip_without_kokkos()

    evaluator = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    evaluator._set_m1_polynomial_policy("retained")
    evaluator._set_m1_recompute_tile_channels(8)
    evaluator._set_m1_polynomial_policy("recompute")
    minimum_scratch = evaluator.macefield_response_m1_recompute_scratch_bytes
    assert minimum_scratch > 1

    evaluator._set_m1_polynomial_policy("retained")
    evaluator._set_m1_recompute_scratch_limit_for_testing(minimum_scratch - 1)
    with pytest.raises(ValueError, match="requires at least.*available limit"):
        evaluator._set_m1_polynomial_policy("recompute")


@pytest.mark.parametrize(
    "kokkos_class,atol",
    [("MACEKokkos", 3e-11), ("MACEKokkosFloat", 5e-5)],
)
def test_kokkos_prepared_execution_field_reuses_topology_and_field_storage(
    macefield_full_json_path,
    kokkos_class,
    atol,
):
    _skip_without_kokkos()

    direct = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    prepared = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    dtype = "float32" if kokkos_class == "MACEKokkosFloat" else "float64"
    _load_direct_artifact(direct, macefield_full_json_path, dtype)
    _load_direct_artifact(prepared, macefield_full_json_path, dtype)
    direct.set_streamed_edges("direct")
    prepared.set_streamed_edges("direct")
    inputs = _field_backend_inputs(direct)[:7]
    topology = inputs[:5]
    xyz = np.ascontiguousarray(inputs[5], dtype=np.float64).reshape(-1)
    distances = np.ascontiguousarray(inputs[6], dtype=np.float64)
    token = prepared._prepare_factorized_graph(*topology)
    field_workspace_bytes = None

    for evaluation, electric_field in enumerate(
        (
            np.array([0.01, -0.02, 0.03], dtype=np.float64),
            np.array([-0.04, 0.01, 0.02], dtype=np.float64),
        ),
        start=1,
    ):
        direct.compute_node_energies_forces_field(
            *topology, xyz, distances, electric_field
        )
        prepared._compute_prepared_factorized_field(
            token, xyz, distances, electric_field
        )
        np.testing.assert_allclose(
            prepared.node_energies, direct.node_energies, rtol=0.0, atol=atol
        )
        np.testing.assert_allclose(
            prepared.node_forces, direct.node_forces, rtol=0.0, atol=atol
        )
        np.testing.assert_allclose(
            prepared.electric_field_adj,
            direct.electric_field_adj,
            rtol=0.0,
            atol=atol,
        )
        assert prepared.factorized_prepared_evaluation_count == evaluation
        assert prepared.factorized_schedule_build_count == 1
        assert prepared.factorized_topology_validation_count == 0
        assert prepared.factorized_topology_validation_skip_count == evaluation
        if field_workspace_bytes is None:
            field_workspace_bytes = prepared.execution_geometry_workspace_bytes
        else:
            assert prepared.execution_geometry_workspace_bytes == field_workspace_bytes


def _configured_low_memory_field_evaluator(
    model_path, geometry_policy, *, low_memory=False, dtype="float32"
):
    evaluator_class = (
        native_symmetrix.MACEKokkosFloat
        if dtype == "float32"
        else native_symmetrix.MACEKokkos
    )
    evaluator = evaluator_class(str(model_path))
    try:
        _load_direct_artifact(evaluator, model_path, dtype)
        evaluator.set_streamed_edges("direct")
        evaluator._set_factorized_source_strategy("jit_plugin")
        evaluator._set_factorized_direct_forward_executor("jit_all")
        evaluator._set_factorized_direct_reverse_executor("jit")
        evaluator._set_standard_r0_executor("v2_edge16")
        evaluator._set_standard_m0_executor("standard")
        evaluator._set_m1_polynomial_policy("recompute")
        if low_memory:
            evaluator._set_low_memory(True)
            if native_symmetrix._kokkos_default_execution_space() in (
                "Cuda",
                "HIP",
            ):
                device_bytes = 32 * 1024**3
                evaluator._set_low_memory_device_memory_info_for_testing(
                    device_bytes // 20 + 512 * 1024, device_bytes
                )
        else:
            evaluator._set_edge_geometry_policy(geometry_policy)
    except Exception:
        evaluator = None
        raise
    return evaluator


@pytest.mark.parametrize(
    ("dtype", "geometry_policy", "parity_atol", "stress_atol"),
    (
        ("float32", "unit-f32-radius-f64-v1", 5e-4, 5e-5),
        ("float64", "cartesian-f64-v1", 2e-9, 2e-10),
    ),
)
def test_kokkos_low_memory_prepared_field_primal_and_responses_match_retained(
    macefield_full_json_path,
    dtype,
    geometry_policy,
    parity_atol,
    stress_atol,
):
    _skip_without_kokkos()

    retained = _configured_low_memory_field_evaluator(
        macefield_full_json_path, "cartesian-f64-v1", dtype=dtype
    )
    low_memory = _configured_low_memory_field_evaluator(
        macefield_full_json_path,
        geometry_policy,
        low_memory=True,
        dtype=dtype,
    )
    execution_space = native_symmetrix._kokkos_default_execution_space()
    device_bytes = 32 * 1024**3
    inputs = _field_backend_inputs(retained)[:7]
    topology = inputs[:5]
    xyz = np.ascontiguousarray(inputs[5], dtype=np.float64).reshape(-1)
    distances = np.ascontiguousarray(inputs[6], dtype=np.float64)
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    volume = bulk("AlN", "wurtzite", a=3.112, c=4.982).get_volume()

    results = {}
    for name, evaluator in (("retained", retained), ("low_memory", low_memory)):
        token = evaluator._prepare_factorized_graph(*topology)
        evaluator._compute_prepared_factorized_field(
            token, xyz, distances, electric_field
        )
        if name == "low_memory":
            assert evaluator.mh0_m0_adjoint_alias_active
            assert evaluator.mh0_m0_adjoint_allocation_count == 0
        else:
            assert not evaluator.mh0_m0_adjoint_alias_active
            assert evaluator.mh0_m0_adjoint_allocation_count > 0
        primal_evaluations = evaluator.factorized_prepared_evaluation_count
        reconstruction_count = evaluator.macefield_response_primal_reconstruction_count
        assert reconstruction_count == 0
        evaluator._compute_prepared_factorized_field_response(
            token, electric_field, False
        )
        hessian = np.array(evaluator.electric_field_hessian, copy=True)
        assert np.asarray(evaluator.electric_field_force_derivative).size == 0
        evaluator._compute_prepared_factorized_field_response(
            token, electric_field, True
        )
        expected_reconstructions = 2 if evaluator.low_memory else 0
        assert evaluator.macefield_response_primal_reconstruction_count == (
            reconstruction_count + expected_reconstructions
        )
        assert evaluator.factorized_prepared_evaluation_count == primal_evaluations
        stress = np.asarray(
            evaluator._reduce_stress(volume, np.empty(0, dtype=np.float64), token),
            dtype=np.float64,
        ).reshape(3, 3)
        directed_forces = np.asarray(evaluator.node_forces, dtype=np.float64).reshape(
            -1, 3
        )
        expected_stress = -(directed_forces.T @ xyz.reshape(-1, 3)) / volume
        np.testing.assert_allclose(stress, expected_stress, rtol=0.0, atol=stress_atol)
        results[name] = {
            "energies": np.array(evaluator.node_energies, copy=True),
            "forces": np.array(evaluator.node_forces, copy=True),
            "field_adj": np.array(evaluator.electric_field_adj, copy=True),
            "hessian": hessian,
            "stress": stress,
            "force_derivative": np.array(
                evaluator.electric_field_force_derivative, copy=True
            ),
        }

        with pytest.raises(RuntimeError, match="current graph token"):
            evaluator._compute_prepared_factorized_field_response(
                token + 1, electric_field, False
            )

    assert low_memory.low_memory is True
    assert low_memory.mh0_state_policy == "reuse-adjoints-v1"
    assert low_memory.edge_geometry_policy == geometry_policy
    if execution_space in ("Cuda", "HIP"):
        assert low_memory.harmonic_storage_policy_request == "automatic"
        assert low_memory.low_memory_policy == "capacity-y-only"
        assert low_memory.harmonic_storage_policy == "y-only-direct-v1"
        assert low_memory.harmonic_storage_fallback_reason == ""
        assert low_memory.harmonic_value_bytes > 0
        assert low_memory.harmonic_gradient_bytes == 0
        assert low_memory.shuffled_coordinate_bytes == 0
        assert low_memory.low_memory_device_free_bytes == (
            device_bytes // 20 + 512 * 1024
        )
        assert low_memory.low_memory_device_total_bytes == device_bytes
    else:
        assert low_memory.harmonic_storage_policy_request == "automatic"
        assert low_memory.harmonic_storage_policy == "retained"
        assert low_memory.harmonic_storage_fallback_reason == (
            "HIP or CUDA execution is not active and host receiver "
            "recomputation is unavailable"
        )
    assert low_memory.standard_r0_module_ready
    assert low_memory.standard_r0_module_fallback_reason == ""
    assert low_memory.mh0_m0_adjoint_alias_active
    assert low_memory.mh0_m0_adjoint_allocation_count == 0
    assert low_memory.mh0_reused_state_bytes > 0
    if dtype == "float32" and native_symmetrix._kokkos_default_execution_space() in (
        "OpenMP",
        "Serial",
    ):
        assert low_memory.standard_m1_module_ready
        assert low_memory.standard_m1_module_forward_launch_count > 0
        assert low_memory.standard_m1_module_reverse_launch_count > 0
        assert low_memory.m1_recompute_scratch_bytes == 0
    if dtype == "float32":
        assert low_memory.compact_edge_geometry_bytes > 0
        assert low_memory.execution_geometry_workspace_bytes < (
            retained.execution_geometry_workspace_bytes
        )
    else:
        assert low_memory.compact_edge_geometry_bytes == 0
        assert low_memory.execution_geometry_workspace_bytes < (
            retained.execution_geometry_workspace_bytes
        )
    for name in (
        "energies",
        "forces",
        "field_adj",
        "hessian",
        "force_derivative",
        "stress",
    ):
        np.testing.assert_allclose(
            results["low_memory"][name],
            results["retained"][name],
            rtol=0.0,
            atol=parity_atol,
        )


@pytest.mark.parametrize(
    ("dtype", "geometry_policy", "parity_atol"),
    (
        ("float32", "unit-f32-radius-f64-v1", 5e-4),
        ("float64", "cartesian-f64-v1", 2e-9),
    ),
)
def test_kokkos_low_memory_field_reuses_admitted_graph_capacity(
    macefield_full_json_path,
    dtype,
    geometry_policy,
    parity_atol,
):
    _skip_without_kokkos()

    probe = _configured_low_memory_field_evaluator(
        macefield_full_json_path,
        geometry_policy,
        low_memory=True,
        dtype=dtype,
    )
    original = _field_backend_inputs(probe)[:7]
    base_nodes = 64
    topology = (
        base_nodes,
        np.pad(
            np.asarray(original[1]),
            (0, base_nodes - original[0]),
            constant_values=original[1][0],
        ),
        np.pad(
            np.asarray(original[2]),
            (0, base_nodes - original[0]),
            constant_values=0,
        ),
        original[3],
        original[4],
    )
    xyz = np.ascontiguousarray(original[5], dtype=np.float64).reshape(-1)
    distances = np.ascontiguousarray(original[6], dtype=np.float64)
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    volume = bulk("AlN", "wurtzite", a=3.112, c=4.982).get_volume()
    device_bytes = 32 * 1024**3
    probe._set_low_memory_device_memory_info_for_testing(device_bytes, device_bytes)
    probe._prepare_factorized_graph(*topology)
    capacity_estimate = probe.low_memory_capacity_estimated_bytes
    speed_estimate = probe.low_memory_speed_estimated_bytes
    constrained_free = device_bytes // 20 + speed_estimate - 1
    assert capacity_estimate < speed_estimate
    assert constrained_free < device_bytes

    retained = _configured_low_memory_field_evaluator(
        macefield_full_json_path,
        "cartesian-f64-v1",
        dtype=dtype,
    )
    low_memory = _configured_low_memory_field_evaluator(
        macefield_full_json_path,
        geometry_policy,
        low_memory=True,
        dtype=dtype,
    )
    low_memory._set_low_memory_device_memory_info_for_testing(
        constrained_free, device_bytes
    )

    def evaluate(evaluator, graph):
        token = evaluator._prepare_factorized_graph(*graph)
        evaluator._compute_prepared_factorized_field(
            token, xyz, distances, electric_field
        )
        evaluator._compute_prepared_factorized_field_response(
            token, electric_field, True
        )
        stress = np.asarray(
            evaluator._reduce_stress(volume, np.empty(0, dtype=np.float64), token),
            dtype=np.float64,
        ).reshape(3, 3)
        return token, {
            "energies": np.array(evaluator.node_energies, copy=True),
            "forces": np.array(evaluator.node_forces, copy=True),
            "field_adj": np.array(evaluator.electric_field_adj, copy=True),
            "hessian": np.array(evaluator.electric_field_hessian, copy=True),
            "force_derivative": np.array(
                evaluator.electric_field_force_derivative, copy=True
            ),
            "stress": stress,
        }

    def assert_matches_retained(graph):
        _, expected = evaluate(retained, graph)
        token, actual = evaluate(low_memory, graph)
        for name in expected:
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=0.0, atol=parity_atol
            )
        return token

    token = assert_matches_retained(topology)
    assert low_memory.low_memory_policy.startswith("capacity-")
    assert low_memory.execution_planned_receiver_capacity > base_nodes
    assert low_memory.execution_planned_feature_node_capacity > base_nodes
    assert low_memory.mh0_m0_adjoint_alias_active
    assert low_memory.mh0_m0_adjoint_allocation_count == 0

    simulated_free = constrained_free - low_memory.execution_planned_capacity_bytes
    assert simulated_free > 0
    low_memory._set_low_memory_device_memory_info_for_testing(
        simulated_free, device_bytes
    )
    with pytest.raises(RuntimeError, match="current graph token"):
        low_memory._compute_prepared_factorized_field_response(
            token, electric_field, True
        )

    graph_replacements = low_memory.factorized_graph_device_replacement_count
    graph_updates = low_memory.factorized_graph_device_update_count
    geometry_allocations = low_memory.execution_geometry_allocation_count
    result_allocations = low_memory.execution_result_allocation_count
    m0_replacements = low_memory.mh0_m0_forward_replacement_count
    m0_detaches = low_memory.mh0_m0_alias_detach_count
    grown_nodes = base_nodes + 1
    grown = (
        grown_nodes,
        np.append(np.asarray(topology[1]), topology[1][0]),
        np.append(np.asarray(topology[2]), 0),
        topology[3],
        topology[4],
    )
    assert_matches_retained(grown)
    assert low_memory.factorized_graph_device_replacement_count == graph_replacements
    assert low_memory.factorized_graph_device_update_count == graph_updates + 1
    assert low_memory.execution_geometry_allocation_count == geometry_allocations
    assert low_memory.execution_result_allocation_count == result_allocations
    assert low_memory.mh0_m0_forward_replacement_count == m0_replacements
    assert low_memory.mh0_m0_alias_detach_count == m0_detaches
    assert low_memory.mh0_m0_adjoint_alias_active

    assert_matches_retained(topology)
    assert low_memory.factorized_graph_device_replacement_count == graph_replacements
    assert low_memory.factorized_graph_device_update_count == graph_updates + 2
    assert low_memory.execution_result_allocation_count == result_allocations
    assert low_memory.mh0_m0_forward_replacement_count == m0_replacements

    exceptional_nodes = low_memory.execution_planned_receiver_capacity + 1
    exceptional = (
        exceptional_nodes,
        np.pad(
            np.asarray(topology[1]),
            (0, exceptional_nodes - base_nodes),
            constant_values=topology[1][0],
        ),
        np.pad(
            np.asarray(topology[2]),
            (0, exceptional_nodes - base_nodes),
            constant_values=0,
        ),
        topology[3],
        topology[4],
    )
    # Provide bounded replacement headroom beyond the admitted high-water.
    # The live selected bundle remains reclaimable, and the 5% reserve is
    # still unavailable to the exceptional allocation.
    low_memory._set_low_memory_device_memory_info_for_testing(
        simulated_free + 1024**2, device_bytes
    )
    assert_matches_retained(exceptional)
    assert low_memory.factorized_graph_device_replacement_count == (
        graph_replacements + 1
    )
    assert low_memory.execution_result_allocation_count == result_allocations + 1
    assert low_memory.mh0_m0_forward_replacement_count == m0_replacements + 1
    assert low_memory.mh0_m0_alias_detach_count == m0_detaches + 1
    assert low_memory.mh0_m0_adjoint_alias_active
    assert low_memory.mh0_m0_adjoint_allocation_count == 0


@pytest.mark.parametrize(
    ("dtype", "geometry_policy"),
    (
        ("float32", "unit-f32-radius-f64-v1"),
        ("float64", "cartesian-f64-v1"),
    ),
)
def test_kokkos_low_memory_field_stress_matches_cell_finite_difference(
    macefield_full_json_path,
    dtype,
    geometry_policy,
):
    _skip_without_kokkos()

    evaluator = _configured_low_memory_field_evaluator(
        macefield_full_json_path,
        geometry_policy,
        low_memory=True,
        dtype=dtype,
    )
    reference_atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    atomic_numbers = reference_atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers

    def inputs_for(atoms):
        i_list, j_list, distances, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
        node_types = np.asarray(
            [mace_atomic_numbers.index(number) for number in atomic_numbers],
            dtype=np.int32,
        )
        num_neigh = np.asarray(
            np.bincount(j_list, minlength=len(atoms)), dtype=np.int32
        )
        neigh_types = np.asarray(
            [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list],
            dtype=np.int32,
        )
        topology = (
            len(atoms),
            node_types,
            num_neigh,
            np.asarray(j_list, dtype=np.int32),
            neigh_types,
        )
        return topology, np.ascontiguousarray(xyz).reshape(-1), distances

    topology, xyz, distances = inputs_for(reference_atoms)
    token = evaluator._prepare_factorized_graph(*topology)
    evaluator._compute_prepared_factorized_field(token, xyz, distances, electric_field)
    stress_xx = np.asarray(
        evaluator._reduce_stress(
            reference_atoms.get_volume(), np.empty(0, dtype=np.float64), token
        ),
        dtype=np.float64,
    ).reshape(3, 3)[0, 0]

    energies = []
    strain_step = 1e-3
    for strain in (-strain_step, strain_step):
        atoms = reference_atoms.copy()
        deformation = np.eye(3)
        deformation[0, 0] += strain
        atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)
        strained_topology, xyz, distances = inputs_for(atoms)
        for actual, expected in zip(strained_topology, topology):
            np.testing.assert_array_equal(actual, expected)
        evaluator._compute_prepared_factorized_field(
            token, xyz, distances, electric_field
        )
        energies.append(float(np.sum(evaluator.node_energies)))

    finite_difference = (energies[1] - energies[0]) / (
        2.0 * strain_step * reference_atoms.get_volume()
    )
    assert stress_xx == pytest.approx(finite_difference, abs=5e-3)


@pytest.mark.parametrize(
    ("dtype", "geometry_policy", "atol"),
    (
        ("float32", "unit-f32-radius-f64-v1", 1e-3),
        ("float64", "cartesian-f64-v1", 2e-8),
    ),
)
def test_kokkos_low_memory_field_calculator_matches_full_retention(
    macefield_full_json_path,
    monkeypatch,
    tmp_path,
    dtype,
    geometry_policy,
    atol,
):
    _skip_without_kokkos()
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))

    def calculator(low_memory):
        return Symmetrix(
            macefield_full_json_path,
            use_kokkos=True,
            dtype=dtype,
            streamed_edges="direct",
            m1_polynomial_policy="retained",
            low_memory=low_memory,
        )

    retained = calculator(False)
    low_memory = calculator(True)
    if native_symmetrix._kokkos_default_execution_space() in ("Cuda", "HIP"):
        low_memory.evaluator._set_low_memory_device_memory_info_for_testing(
            512 * 1024**2, 32 * 1024**3
        )
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
    for name, active_calculator in (
        ("retained", retained),
        ("low_memory", low_memory),
    ):
        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
        atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])
        atoms.calc = active_calculator
        active_calculator.calculate(
            atoms,
            properties=["energy", "forces", "stress", "polarization"],
        )
        assert (
            active_calculator.evaluator.macefield_response_primal_reconstruction_count
            == 0
        )
        active_calculator.calculate(atoms, properties=list(properties))
        results[name] = {
            prop: np.array(active_calculator.results[prop], copy=True)
            for prop in properties
        }

    assert low_memory.low_memory is True
    assert low_memory.evaluator.low_memory is True
    assert low_memory.evaluator.mh0_state_policy == "reuse-adjoints-v1"
    assert low_memory.edge_geometry_policy == geometry_policy
    assert low_memory.evaluator.standard_r0_module_ready
    if dtype == "float32":
        assert low_memory.evaluator.compact_edge_geometry_bytes > 0
    else:
        assert low_memory.evaluator.compact_edge_geometry_bytes == 0
    assert low_memory.evaluator.execution_prepared_geometry_update_count > 0
    assert low_memory.neighbor_cache_host_geometry_materialization_count == 1
    assert low_memory.evaluator.macefield_response_primal_reconstruction_count == 1
    assert retained.evaluator.macefield_response_primal_reconstruction_count == 0
    for prop in properties:
        np.testing.assert_allclose(
            results["low_memory"][prop],
            results["retained"][prop],
            rtol=0.0,
            atol=atol,
        )


def test_kokkos_current_execution_response_reuses_completed_primal(
    macefield_full_json_path,
):
    _skip_without_kokkos()

    reference = native_symmetrix.MACEKokkos(str(macefield_full_json_path))
    current = native_symmetrix.MACEKokkos(str(macefield_full_json_path))
    _load_direct_artifact(reference, macefield_full_json_path, "float64")
    _load_direct_artifact(current, macefield_full_json_path, "float64")
    reference.set_streamed_edges("direct")
    current.set_streamed_edges("direct")
    inputs = _field_backend_inputs(reference)[:7]
    topology = inputs[:5]
    xyz = np.ascontiguousarray(inputs[5], dtype=np.float64).reshape(-1)
    distances = np.ascontiguousarray(inputs[6], dtype=np.float64)
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    token = current._prepare_factorized_graph(*topology)
    current._compute_prepared_factorized_field(token, xyz, distances, electric_field)
    completed_evaluations = current.factorized_prepared_evaluation_count
    schedule_bytes = current.factorized_schedule_bytes
    response_calls = current.macefield_response_call_count
    topology_selections = current.macefield_response_factorized_topology_count

    reference.compute_electric_field_hessian(*topology, xyz, distances, electric_field)
    current._compute_current_electric_field_hessian(
        *topology, xyz, distances, electric_field, token
    )
    assert current.factorized_prepared_evaluation_count == completed_evaluations
    assert current.factorized_schedule_bytes == schedule_bytes
    assert current.macefield_response_call_count == response_calls + 1
    assert (
        current.macefield_response_factorized_topology_count == topology_selections + 1
    )
    assert current.macefield_response_receiver_ownership == "factorized_edge"
    assert current.macefield_response_source_ownership == "edge_atomic"
    assert np.asarray(current.electric_field_force_derivative).size == 0
    assert np.asarray(current.R0).size == 0
    assert np.asarray(current.R1).size == 0
    np.testing.assert_allclose(
        current.electric_field_hessian,
        reference.electric_field_hessian,
        rtol=0.0,
        atol=2e-6,
    )

    with pytest.raises(RuntimeError, match="current graph token"):
        current._compute_current_electric_field_hessian(
            *topology, xyz, distances, electric_field, token + 1
        )
    assert current.macefield_response_call_count == response_calls + 1

    reference.compute_electric_field_force_derivative(
        *topology, xyz, distances, electric_field
    )
    current._compute_current_electric_field_force_derivative(
        *topology, xyz, distances, electric_field, token
    )
    assert current.factorized_prepared_evaluation_count == completed_evaluations
    assert current.factorized_schedule_bytes == schedule_bytes
    assert current.macefield_response_call_count == response_calls + 2
    assert (
        current.macefield_response_factorized_topology_count == topology_selections + 2
    )
    assert np.asarray(current.R0).size == 0
    assert np.asarray(current.R1).size == 0
    np.testing.assert_allclose(
        current.electric_field_hessian,
        reference.electric_field_hessian,
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        current.electric_field_force_derivative,
        reference.electric_field_force_derivative,
        rtol=0.0,
        atol=2e-6,
    )
    if native_symmetrix._kokkos_default_execution_space() == "Cuda":
        assert current.macefield_response_phi1_fused_launch_count == 6
        assert current.macefield_response_phi1_generic_launch_count == 0
        assert current.macefield_response_a0_fused_launch_count == 3
        assert current.macefield_response_a0_generic_launch_count == 0
    else:
        assert current.macefield_response_phi1_fused_launch_count == 0
        assert current.macefield_response_phi1_generic_launch_count == 6
        assert current.macefield_response_a0_fused_launch_count == 0
        assert current.macefield_response_a0_generic_launch_count == 3


@pytest.mark.parametrize("mode", ["generic"])
def test_kokkos_streamed_field_responses_match_legacy(
    macefield_full_json_path,
    mode,
):
    _skip_without_kokkos()

    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    legacy = native_symmetrix.MACEKokkos(str(macefield_full_json_path))
    streamed = native_symmetrix.MACEKokkos(str(macefield_full_json_path))
    streamed.set_streamed_edges(mode)
    inputs = _field_backend_inputs(legacy)[:7]

    legacy.compute_electric_field_hessian(*inputs, electric_field)
    streamed.compute_electric_field_hessian(*inputs, electric_field)

    np.testing.assert_allclose(
        streamed.electric_field_hessian,
        legacy.electric_field_hessian,
        rtol=0.0,
        atol=2e-6,
    )
    assert np.asarray(legacy.electric_field_force_derivative).size == 0
    assert np.asarray(streamed.electric_field_force_derivative).size == 0

    legacy.compute_electric_field_force_derivative(*inputs, electric_field)
    streamed.compute_electric_field_force_derivative(*inputs, electric_field)
    np.testing.assert_allclose(
        streamed.electric_field_force_derivative,
        legacy.electric_field_force_derivative,
        rtol=0.0,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "kokkos_class,atol,rtol",
    [
        ("MACEKokkos", 2e-6, 2e-6),
        ("MACEKokkosFloat", 2e-4, 2e-4),
    ],
)
def test_kokkos_electric_field_hessian_matches_native(
    macefield_full_json_path,
    kokkos_class,
    atol,
    rtol,
):
    _skip_without_kokkos()

    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    native = native_symmetrix.MACE(str(macefield_full_json_path))
    kokkos = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, _ = (
        _field_backend_inputs(native)
    )

    native.compute_electric_field_hessian(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    kokkos.compute_electric_field_hessian(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )

    assert np.allclose(
        kokkos.electric_field_hessian,
        native.electric_field_hessian,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "kokkos_class,atol,rtol",
    [
        ("MACEKokkos", 2e-6, 2e-6),
        ("MACEKokkosFloat", 5e-4, 5e-4),
    ],
)
def test_kokkos_electric_field_force_derivative_matches_native(
    macefield_full_json_path,
    kokkos_class,
    atol,
    rtol,
):
    _skip_without_kokkos()

    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    native = native_symmetrix.MACE(str(macefield_full_json_path))
    kokkos = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, i_list = (
        _field_backend_inputs(native)
    )

    native.compute_electric_field_force_derivative(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    kokkos.compute_electric_field_force_derivative(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )

    native_deriv = np.asarray(
        native.electric_field_force_derivative, dtype=np.float64
    ).reshape(3, -1, 3)
    kokkos_deriv = np.asarray(
        kokkos.electric_field_force_derivative, dtype=np.float64
    ).reshape(3, -1, 3)
    assert np.allclose(
        kokkos_deriv[:, : len(i_list)],
        native_deriv[:, : len(i_list)],
        atol=atol,
        rtol=rtol,
    )


def test_native_exposes_atomic_energies_for_node_energy(macefield_full_json_path):
    evaluator = native_symmetrix.MACE(str(macefield_full_json_path))

    atomic_energies = np.asarray(evaluator.atomic_energies, dtype=np.float64)

    assert atomic_energies.shape == (len(evaluator.atomic_numbers),)
    assert np.all(np.isfinite(atomic_energies))


def test_native_electric_field_hessian_matches_field_adjoint_finite_difference(
    macefield_full_json_path,
):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    evaluator = native_symmetrix.MACE(str(macefield_full_json_path))
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)],
        dtype=np.int32,
    )
    num_neigh = np.asarray(np.bincount(j_list, minlength=num_nodes), dtype=np.int32)
    neigh_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list], dtype=np.int32
    )
    neigh_indices = np.asarray(j_list, dtype=np.int32)

    evaluator.compute_electric_field_hessian(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    actual = np.asarray(evaluator.electric_field_hessian, dtype=np.float64).reshape(
        3, 3
    )

    step = 1e-4
    expected = np.zeros((3, 3))
    for component in range(3):
        field_plus = electric_field.copy()
        field_minus = electric_field.copy()
        field_plus[component] += step
        field_minus[component] -= step
        evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz.reshape(-1),
            r,
            field_plus,
        )
        adj_plus = np.asarray(evaluator.electric_field_adj, dtype=np.float64)
        evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz.reshape(-1),
            r,
            field_minus,
        )
        adj_minus = np.asarray(evaluator.electric_field_adj, dtype=np.float64)
        expected[:, component] = (adj_plus - adj_minus) / (2.0 * step)

    assert np.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_native_electric_field_force_derivative_matches_force_finite_difference(
    macefield_full_json_path,
):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    evaluator = native_symmetrix.MACE(str(macefield_full_json_path))
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)],
        dtype=np.int32,
    )
    num_neigh = np.asarray(np.bincount(j_list, minlength=num_nodes), dtype=np.int32)
    neigh_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list], dtype=np.int32
    )
    neigh_indices = np.asarray(j_list, dtype=np.int32)

    evaluator.compute_electric_field_force_derivative(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    actual = np.asarray(
        evaluator.electric_field_force_derivative, dtype=np.float64
    ).reshape(3, -1, 3)

    step = 1e-4
    expected = np.zeros_like(actual)
    for component in range(3):
        field_plus = electric_field.copy()
        field_minus = electric_field.copy()
        field_plus[component] += step
        field_minus[component] -= step
        evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz.reshape(-1),
            r,
            field_plus,
        )
        forces_plus = np.asarray(evaluator.node_forces, dtype=np.float64).reshape(
            (-1, 3)
        )[: len(i_list)]
        evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz.reshape(-1),
            r,
            field_minus,
        )
        forces_minus = np.asarray(evaluator.node_forces, dtype=np.float64).reshape(
            (-1, 3)
        )[: len(i_list)]
        expected[component] = (forces_plus - forces_minus) / (2.0 * step)

    assert np.allclose(
        actual[:, : len(i_list)], expected[:, : len(i_list)], atol=2e-6, rtol=2e-5
    )


def _compact_field_coupling_from_json(path):
    data = json.loads(Path(path).read_text())
    coupling = data["field_couplings"][0]

    class Module:
        pass

    class Instruction:
        pass

    field_feats = Module()
    field_feats.irreps_in1 = coupling["field_feats_irreps_in1"]
    field_feats.irreps_in2 = coupling["field_feats_irreps_in2"]
    field_feats.irreps_out = coupling["field_feats_irreps_out"]
    field_feats.output_mask = torch.tensor(
        coupling["field_feats_output_mask"], dtype=torch.float64
    )
    field_feats.weight = torch.tensor(
        coupling["field_feats_weight"], dtype=torch.float64
    )
    field_feats.instructions = []
    for item in coupling["field_feats_instructions"]:
        instruction = Instruction()
        instruction.i_in1 = item["i_in1"]
        instruction.i_in2 = item["i_in2"]
        instruction.i_out = item["i_out"]
        instruction.connection_mode = item["connection_mode"]
        instruction.path_shape = tuple(item["path_shape"])
        instruction.path_weight = item["path_weight"]
        field_feats.instructions.append(instruction)

    field_linear = Module()
    field_linear.irreps_in = coupling["field_linear_irreps_in"]
    field_linear.irreps_out = coupling["field_linear_irreps_out"]
    field_linear.output_mask = torch.tensor(
        coupling["field_linear_output_mask"], dtype=torch.float64
    )
    field_linear.bias = torch.tensor(coupling["field_linear_bias"], dtype=torch.float64)
    field_linear.weight = torch.tensor(
        coupling["field_linear_weight"], dtype=torch.float64
    )
    field_linear.instructions = []
    for item in coupling["field_linear_instructions"]:
        instruction = Instruction()
        instruction.i_in = item["i_in"]
        instruction.i_out = item["i_out"]
        instruction.path_shape = tuple(item["path_shape"])
        instruction.path_weight = item["path_weight"]
        field_linear.instructions.append(instruction)

    return _compact_field_coupling(field_feats, field_linear)
