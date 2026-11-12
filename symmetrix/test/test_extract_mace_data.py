import hashlib
import importlib
import json
import logging
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    extract_mace_data_module = importlib.import_module("symmetrix.extract_mace_data")
    from symmetrix.extract_mace_data import (
        _load_mace_checkpoint,
        _remove_prediction_head,
        extract_mace_data,
    )
    from symmetrix.mace_execution_contract import (
        normalize_generated_m1_contract,
        normalize_jit_r1_contract,
        normalize_standard_m0_contract,
        normalize_standard_r0_contract,
    )
except ImportError as exc:
    extract_mace_data = None
    extract_mace_data_import_error = exc
else:
    extract_mace_data_import_error = None

pytestmark = pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)


def _reference_evaluator(path, head=None):
    """Construct the retained MACE evaluator on its artifact-free reference routing.

    The serial MACE evaluator is retired, so ``MACEKokkos`` is the only MACE
    evaluator left.  It defaults to the prepared ``direct`` routing for a compact
    model, which needs a loaded RTC artifact, so the artifact-free reference
    routing is requested explicitly.
    """

    from symmetrix import symmetrix as native_symmetrix

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = (
        native_symmetrix.MACEKokkos(str(path))
        if head is None
        else native_symmetrix.MACEKokkos(str(path), head)
    )
    evaluator.set_streamed_edges("generic")
    return evaluator


def test_ordinary_checkpoint_does_not_import_macefield(monkeypatch):
    checkpoint = object()
    load_calls = []

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return checkpoint

    def unexpected_import(name):
        raise AssertionError(f"ordinary checkpoint imported {name}")

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)
    monkeypatch.setattr(
        extract_mace_data_module, "_import_mace_module", unexpected_import
    )

    assert _load_mace_checkpoint("ordinary.model", "cpu") is checkpoint
    assert load_calls == [
        (
            ("ordinary.model",),
            {"map_location": "cpu", "weights_only": False},
        )
    ]


def _scalar_maceles_model(*, use_dipoles=False):
    class MACELES:
        pass

    model = MACELES()
    model.interactions = [
        object(),
        SimpleNamespace(linear_up=SimpleNamespace(weight=torch.tensor([2.0]))),
    ]
    model.les_readouts = [
        SimpleNamespace(linear=SimpleNamespace(weight=torch.tensor([4.0]))),
        SimpleNamespace(
            linear_1=SimpleNamespace(weight=torch.tensor([6.0])),
            linear_2=SimpleNamespace(weight=torch.tensor([8.0])),
            non_linearity=SimpleNamespace(acts=[SimpleNamespace(cst=1.5)]),
        ),
    ]
    model.les = SimpleNamespace(
        sigma=1.0,
        dl=2.0,
        remove_self_interaction=True,
        use_atomwise=False,
        use_fixed_atomic_charges=False,
        use_atomic_alpha=False,
        use_epsilon_r_scaling=False,
    )
    model.les_output_scale = 0.25
    model.use_dipoles = use_dipoles
    model.use_quads = False
    model.use_induced_charges = False
    model.use_induced_dipoles = False
    model.use_anisotropic_polarizability = False
    model.compute_bec = False
    return model


def _base_contracts():
    return {
        "R0": {"fingerprint": "sha256:" + "1" * 64},
        "M0": {"semantic_fingerprint": "sha256:" + "2" * 64},
        "R1": {"fingerprint": "sha256:" + "3" * 64},
    }


@pytest.fixture(scope="module")
def upstream_maceles_checkpoint_paths(tmp_path_factory):
    pytest.importorskip("les")
    o3 = pytest.importorskip("e3nn.o3")
    modules = pytest.importorskip("mace.modules")
    extensions = pytest.importorskip("mace.modules.extensions")

    def build(heads):
        num_heads = 1 if heads is None else len(heads)
        atomic_energies = np.array(
            [[-1.2 - 0.2 * head, -2.3 - 0.2 * head] for head in range(num_heads)]
        )
        if heads is None:
            atomic_energies = atomic_energies[0]
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            with torch.random.fork_rng():
                torch.manual_seed(123)
                return extensions.MACELES(
                    les_arguments={
                        "use_atomwise": False,
                        "use_fixed_atomic_charges": False,
                        "use_atomic_alpha": False,
                        "use_epsilon_r_scaling": False,
                        "sigma": 1.0,
                        "dl": 2.0,
                        "remove_self_interaction": True,
                        "output_scale": 0.25,
                    },
                    r_max=3.5,
                    num_bessel=4,
                    num_polynomial_cutoff=3,
                    max_ell=3,
                    interaction_cls=modules.interaction_classes[
                        "RealAgnosticResidualInteractionBlock"
                    ],
                    interaction_cls_first=modules.interaction_classes[
                        "RealAgnosticInteractionBlock"
                    ],
                    num_interactions=2,
                    num_elements=2,
                    hidden_irreps=o3.Irreps("4x0e + 4x1o"),
                    MLP_irreps=o3.Irreps("3x0e"),
                    atomic_energies=atomic_energies,
                    avg_num_neighbors=3.0,
                    atomic_numbers=[7, 13],
                    correlation=3,
                    gate=modules.gate_dict["silu"],
                    pair_repulsion=False,
                    distance_transform="None",
                    radial_type="bessel",
                    use_reduced_cg=False,
                    atomic_inter_scale=(
                        1.3 if heads is None else np.linspace(1.3, 1.4, num_heads)
                    ),
                    atomic_inter_shift=(
                        0.0 if heads is None else np.linspace(0.0, 0.1, num_heads)
                    ),
                    heads=heads,
                ).to(torch.float64)
        finally:
            torch.set_default_dtype(previous_dtype)

    root = tmp_path_factory.mktemp("extract-maceles")
    paths = {}
    for name, heads in (("single", None), ("multi", ["first", "second"])):
        model = build(heads)
        model.eval()
        path = root / f"{name}.model"
        torch.save(model, path)
        paths[name] = path
    return paths


def test_extract_single_head_maceles_publishes_intrinsic_model_identity(
    upstream_maceles_checkpoint_paths,
):
    data = extract_mace_data(
        upstream_maceles_checkpoint_paths["single"],
        species=[7, 13],
        num_spline_points=4,
    )

    assert data["model_type"] == "MACELES"
    assert data["les"]["schema"] == "symmetrix.model.mace_les"
    assert data["les"]["version"] == 1


def test_extract_multihead_maceles_publishes_les_parameters_for_every_head(
    upstream_maceles_checkpoint_paths,
):
    data = extract_mace_data(
        upstream_maceles_checkpoint_paths["multi"],
        species=[7, 13],
        num_spline_points=4,
    )

    assert data["model_type"] == "MACELES"
    assert data["prediction_heads"]["names"] == ["first", "second"]
    assert set(data["prediction_heads"]["parameters"]) == {"first", "second"}
    for payload in data["prediction_heads"]["parameters"].values():
        assert payload["les"]["schema"] == "symmetrix.model.mace_les"
        assert payload["les"]["version"] == 1


def test_extract_maceles_serializes_charge_readouts_without_energy_scaling():
    data = extract_mace_data_module._extract_mace_les_parameters(
        _scalar_maceles_model(),
        num_channels=1,
        l_max=0,
        base_contracts=_base_contracts(),
    )

    assert data["kind"] == "monopole"
    assert data["output_scale"] == pytest.approx(0.25)
    assert data["readout_1_weights"] == pytest.approx([2.0])
    assert data["readout_2_weights_1"] == pytest.approx([6.0])
    assert data["readout_2_weights_2"] == pytest.approx([8.0])
    assert data["mace_num_interactions"] == 2
    assert data["mace_num_channels"] == 1
    assert data["mace_contract_fingerprints"] == {
        "R0": "sha256:" + "1" * 64,
        "M0": "sha256:" + "2" * 64,
        "R1": "sha256:" + "3" * 64,
    }


def test_extract_maceles_pre_linear_readout_stays_on_retained_basis():
    data = extract_mace_data_module._extract_mace_les_parameters(
        _scalar_maceles_model(),
        num_channels=1,
        l_max=0,
        base_contracts=_base_contracts(),
        h1_affine_mode="pre_linear_up",
    )

    assert data["readout_1_weights"] == pytest.approx([4.0])


def test_prediction_head_payload_keeps_les_parameters_per_maceles_head():
    data = {
        "model_type": "MACELES",
        "atomic_energies": [1.0],
        "readout_1_weights": [2.0],
        "readout_2_weights_1": [3.0],
        "readout_2_weights_2": [4.0],
        "les": {"schema": "symmetrix.model.mace_les", "version": 1},
        "shared_metadata": "excluded",
    }

    payload = extract_mace_data_module._prediction_head_payload(data)

    assert payload == {
        "atomic_energies": [1.0],
        "readout_1_weights": [2.0],
        "readout_2_weights_1": [3.0],
        "readout_2_weights_2": [4.0],
        "les": {"schema": "symmetrix.model.mace_les", "version": 1},
    }


def test_extract_legacy_maceles_uses_nested_ewald_flag_and_default_scale():
    model = _scalar_maceles_model()
    model.les.ewald = SimpleNamespace(
        remove_self_interaction=False,
        norm_factor=90.0474,
    )
    del model.les.remove_self_interaction
    del model.les_output_scale

    data = extract_mace_data_module._extract_mace_les_parameters(
        model,
        num_channels=1,
        l_max=0,
        base_contracts=_base_contracts(),
    )

    assert data["ewald"]["remove_self_interaction"] is False
    assert data["ewald"]["norm_factor"] == pytest.approx(90.0474)
    assert data["output_scale"] == pytest.approx(1.0)


def test_extract_legacy_maceles_defaults_missing_self_interaction_flag():
    model = _scalar_maceles_model()
    del model.les.remove_self_interaction

    data = extract_mace_data_module._extract_mace_les_parameters(
        model,
        num_channels=1,
        l_max=0,
        base_contracts=_base_contracts(),
    )

    assert data["ewald"]["remove_self_interaction"] is True
    assert data["ewald"]["norm_factor"] == pytest.approx(90.4756)


def test_extract_maceles_prefers_nested_self_interaction_flag():
    model = _scalar_maceles_model()
    model.les.ewald = SimpleNamespace(remove_self_interaction=False)

    data = extract_mace_data_module._extract_mace_les_parameters(
        model,
        num_channels=1,
        l_max=0,
        base_contracts=_base_contracts(),
    )

    assert data["ewald"]["remove_self_interaction"] is False


def test_extract_maceles_rejects_vector_features():
    with pytest.raises(RuntimeError, match="use_dipoles"):
        extract_mace_data_module._extract_mace_les_parameters(
            _scalar_maceles_model(use_dipoles=True),
            num_channels=1,
            l_max=0,
            base_contracts=_base_contracts(),
        )


def test_metatrain_mace_checkpoint_uses_metatrain_loader(monkeypatch):
    checkpoint = {"core": {"architecture_name": "experimental.mace"}}
    restored = object()

    monkeypatch.setattr(
        extract_mace_data_module.torch,
        "load",
        lambda *args, **kwargs: checkpoint,
    )
    monkeypatch.setattr(
        extract_mace_data_module,
        "_load_metatrain_mace_checkpoint",
        lambda loaded, device: restored,
        raising=False,
    )

    assert _load_mace_checkpoint("metatrain.ckpt", "cpu") is restored


def test_metatomic_export_uses_metatrain_export_loader(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    with zipfile.ZipFile(model_path, "w") as archive:
        archive.writestr("model/extra/metatomic-version", "0.1.19")

    restored = object()
    monkeypatch.setattr(
        extract_mace_data_module,
        "_load_metatrain_mace_export",
        lambda path, device: restored,
        raising=False,
    )
    monkeypatch.setattr(
        extract_mace_data_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("metatomic export used torch.load"),
    )

    assert _load_mace_checkpoint(model_path, "cpu") is restored


def _assert_model_data_close(expected, actual, path="model"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_model_data_close(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_model_data_close(expected_item, actual_item, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7), path
    else:
        assert actual == expected, path


def test_metatrain_native_checkpoint_and_export_round_trip(tmp_path, monkeypatch):
    o3 = pytest.importorskip("e3nn.o3")
    modules = pytest.importorskip("mace.modules")
    loader = pytest.importorskip("metatrain.experimental.mace.utils._load_model_file")

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(123)
            native_model = modules.ScaleShiftMACE(
                r_max=3.5,
                num_bessel=4,
                num_polynomial_cutoff=3,
                max_ell=3,
                interaction_cls=modules.interaction_classes[
                    "RealAgnosticResidualInteractionBlock"
                ],
                interaction_cls_first=modules.interaction_classes[
                    "RealAgnosticInteractionBlock"
                ],
                num_interactions=2,
                num_elements=2,
                hidden_irreps=o3.Irreps("4x0e + 4x1o"),
                MLP_irreps=o3.Irreps("3x0e"),
                atomic_energies=np.array([-1.2, -2.3]),
                avg_num_neighbors=3.0,
                atomic_numbers=[1, 8],
                correlation=3,
                gate=modules.gate_dict["silu"],
                pair_repulsion=False,
                distance_transform="None",
                radial_type="bessel",
                atomic_inter_scale=1.3,
                atomic_inter_shift=0.0,
            ).to(torch.float64)
    finally:
        torch.set_default_dtype(previous_dtype)

    native_model.eval()
    native_path = tmp_path / "native.model"
    checkpoint_path = tmp_path / "metatrain.ckpt"
    export_path = tmp_path / "metatrain.pt"
    torch.save(native_model, native_path)

    monkeypatch.chdir(tmp_path)
    metatrain_model = loader.load_mace_model_file(
        native_path, "energy", torch.device("cpu")
    )
    torch.save(metatrain_model.get_checkpoint(), checkpoint_path)
    metatrain_model.export().save(str(export_path))

    native_data = extract_mace_data(native_path, species=[1, 8], num_spline_points=4)
    checkpoint_data = extract_mace_data(
        checkpoint_path, species=[1, 8], num_spline_points=4
    )
    export_data = extract_mace_data(export_path, species=[1, 8], num_spline_points=4)

    _assert_model_data_close(native_data, checkpoint_data, "checkpoint")
    _assert_model_data_close(native_data, export_data, "export")


def test_scripted_readout_activation_uses_module_children():
    activation = SimpleNamespace(code="return torch.silu(x)")
    readout = SimpleNamespace(
        non_linearity=SimpleNamespace(
            acts=SimpleNamespace(children=lambda: iter([activation]))
        )
    )

    assert extract_mace_data_module._scripted_readout_uses_silu(readout)


@pytest.mark.parametrize(
    ("basis_name", "attribute", "values", "expected"),
    [
        ("BesselBasis", "bessel_weights", torch.zeros(5), 5),
        ("GaussianBasis", "gaussian_weights", torch.zeros(7), 7),
        ("ChebychevBasis", "n", torch.zeros(1, 9), 9),
    ],
)
def test_scripted_radial_basis_size(basis_name, attribute, values, expected):
    radial_basis = SimpleNamespace(original_name=basis_name)
    setattr(radial_basis, attribute, values)

    assert (
        extract_mace_data_module._scripted_radial_basis_size(radial_basis) == expected
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("return (torch.mul(radial, cutoff), None)", True),
        ("return (radial, cutoff)", False),
    ],
)
def test_scripted_radial_cutoff_from_frozen_code(code, expected):
    radial_embedding = SimpleNamespace(code=code)

    assert (
        extract_mace_data_module._scripted_radial_applies_cutoff(radial_embedding)
        is expected
    )


def test_scripted_multihead_readout_recovers_per_head_width():
    readout = SimpleNamespace(hidden_irreps=[(32, "0e")])

    assert extract_mace_data_module._scripted_per_head_mlp_irreps(readout, 2) == [
        (16, "0e")
    ]


def test_scripted_single_interaction_uses_placeholder_mlp_irreps():
    readout = SimpleNamespace(original_name="LinearReadoutBlock")

    assert extract_mace_data_module._scripted_per_head_mlp_irreps(readout, 1) == [
        (1, (0, 1))
    ]


def _metatrain_dataset_info(
    *, target="energy", quantity="", unit="", length_unit="Angstrom"
):
    return SimpleNamespace(
        length_unit=length_unit,
        targets={
            target: SimpleNamespace(
                is_scalar=True,
                sample_kind="system",
                quantity=quantity,
                unit=unit,
            )
        },
    )


@pytest.mark.parametrize(
    ("dataset_info", "message"),
    [
        (_metatrain_dataset_info(target="charge"), "requires energy"),
        (_metatrain_dataset_info(unit="kcal/mol"), "requires eV"),
        (_metatrain_dataset_info(length_unit="bohr"), "requires Angstrom"),
    ],
)
def test_metatrain_metadata_rejects_incompatible_semantics(dataset_info, message):
    with pytest.raises(RuntimeError, match=message):
        extract_mace_data_module._validate_metatrain_energy_metadata(
            next(iter(dataset_info.targets)), dataset_info=dataset_info
        )


def test_restore_metatrain_multihead_normalizes_fake_scale_shift(monkeypatch):
    class AtomicEnergies(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("atomic_energies", torch.tensor(values))

    class MaceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.atomic_energies_fn = AtomicEnergies([[0.0, 0.0], [0.0, 0.0]])
            self.atomic_numbers = torch.tensor([1, 8])
            self.heads = ["pretraining", "target"]
            self.scale_shift = SimpleNamespace(
                scale=torch.tensor(1.0), shift=torch.tensor(0.0)
            )

    selected = MaceModel()
    selected.atomic_energies_fn = AtomicEnergies([0.0, 0.0])
    selected.heads = ["target"]

    def fake_remove_prediction_head(model, head, device):
        assert head == "target"
        assert model.scale_shift.scale.tolist() == [1.0, 1.0]
        assert model.scale_shift.shift.tolist() == [0.0, 0.0]
        return selected.to(device)

    monkeypatch.setattr(
        extract_mace_data_module,
        "_remove_prediction_head",
        fake_remove_prediction_head,
    )
    metatrain_model = SimpleNamespace(
        core=SimpleNamespace(
            mace_model=MaceModel(),
            mace_head_index=1,
            hypers={"mace_head_target": "energy"},
        ),
        scaler=None,
        additive_models=[],
    )

    restored = extract_mace_data_module._restore_metatrain_mace_model(
        metatrain_model,
        torch.device("cpu"),
        dataset_info=_metatrain_dataset_info(),
    )

    assert restored.heads == ["target"]
    assert restored.scale_shift.scale.item() == pytest.approx(1.0)


def test_restore_metatrain_mace_scale_and_atomic_baseline():
    class AtomicEnergies(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("atomic_energies", torch.zeros(2))

    class MaceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atomic_energies_fn = AtomicEnergies()
            self.atomic_numbers = torch.tensor([1, 8])
            self.heads = ["default"]

    class TensorMap:
        def __init__(self, values, samples=None):
            self._block = SimpleNamespace(
                values=torch.tensor(values, dtype=torch.float64),
                samples=SimpleNamespace(
                    values=torch.tensor(samples or [[-1]], dtype=torch.int32)
                ),
            )

        def block(self, index):
            assert index == 0
            return self._block

    mace_model = MaceModel()
    composition = SimpleNamespace(
        model=SimpleNamespace(
            weights={"energy": TensorMap([[3.0], [4.0]], samples=[[1], [8]])}
        ),
        supported_outputs=lambda: {"energy": object()},
    )
    metatrain_model = SimpleNamespace(
        core=SimpleNamespace(
            mace_model=mace_model,
            mace_head_index=0,
            hypers={"mace_head_target": "energy"},
        ),
        scaler=SimpleNamespace(
            model=SimpleNamespace(scales={"energy": TensorMap([[2.5]])})
        ),
        additive_models=[composition],
    )

    restored = extract_mace_data_module._restore_metatrain_mace_model(
        metatrain_model,
        torch.device("cpu"),
        dataset_info=_metatrain_dataset_info(),
    )

    assert restored is not mace_model
    assert restored.atomic_energies_fn.atomic_energies.tolist() == [3.0, 4.0]
    assert restored.scale_shift.scale.item() == pytest.approx(2.5)
    assert restored.scale_shift.shift.item() == pytest.approx(0.0)
    assert mace_model.atomic_energies_fn.atomic_energies.tolist() == [0.0, 0.0]


def test_restore_metatrain_accepts_direct_legacy_model():
    class AtomicEnergies(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("atomic_energies", torch.zeros(2))

    class MaceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atomic_energies_fn = AtomicEnergies()
            self.atomic_numbers = torch.tensor([1, 8])
            self.heads = ["default"]

    metatrain_model = SimpleNamespace(
        mace_model=MaceModel(),
        mace_head_index=0,
        hypers={"mace_head_target": "energy"},
        dataset_info=_metatrain_dataset_info(),
        scaler=None,
        additive_models=[],
    )

    restored = extract_mace_data_module._restore_metatrain_mace_model(
        metatrain_model, torch.device("cpu")
    )

    assert restored.atomic_numbers.tolist() == [1, 8]
    assert restored.scale_shift.scale.item() == pytest.approx(1.0)


def test_remove_prediction_head_restores_default_dtype(monkeypatch):
    model = extract_mace_data_module.torch.nn.Linear(1, 1).double()
    previous_dtype = extract_mace_data_module.torch.get_default_dtype()

    def fake_remove(candidate, head):
        assert candidate is model
        assert head == "selected"
        assert extract_mace_data_module.torch.get_default_dtype() is torch.float64
        return candidate

    monkeypatch.setattr(extract_mace_data_module, "remove_pt_head", fake_remove)
    selected = _remove_prediction_head(model, "selected", torch.device("cpu"))

    assert selected is model
    assert extract_mace_data_module.torch.get_default_dtype() is previous_dtype


def test_unrelated_checkpoint_error_is_preserved(monkeypatch):
    load_error = ValueError("checkpoint is corrupt")

    def fake_load(*args, **kwargs):
        raise load_error

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)

    with pytest.raises(ValueError) as error:
        _load_mace_checkpoint("broken.model", "cpu")

    assert error.value is load_error


def test_maceles_checkpoint_missing_les_module_has_conversion_dependency_error(
    monkeypatch,
):
    missing = ModuleNotFoundError("No module named 'les'", name="les")

    def fake_load(*args, **kwargs):
        raise missing

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)

    with pytest.raises(
        RuntimeError, match=r"MACELES.*conversion.*install.*les"
    ) as error:
        _load_mace_checkpoint("maceles.model", "cpu")

    assert error.value.__cause__ is missing


def test_unrelated_missing_module_checkpoint_error_is_preserved(monkeypatch):
    missing = ModuleNotFoundError(
        "No module named 'ordinary_dependency'", name="ordinary_dependency"
    )

    def fake_load(*args, **kwargs):
        raise missing

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)

    with pytest.raises(ModuleNotFoundError) as error:
        _load_mace_checkpoint("ordinary.model", "cpu")

    assert error.value is missing


def test_legacy_macefield_checkpoint_uses_temporary_alias(monkeypatch):
    checkpoint = object()
    mace_models = SimpleNamespace()
    mace_field = type("MACEField", (), {})
    mace_extensions = SimpleNamespace(MACEField=mace_field)
    load_count = 0

    def fake_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            raise AttributeError(
                "Can't get attribute 'ScaleShiftFieldMACE' on "
                "<module 'mace.modules.models'>"
            )
        assert mace_models.ScaleShiftFieldMACE is mace_field
        return checkpoint

    def fake_import(name):
        return {
            "mace.modules.models": mace_models,
            "mace.modules.extensions": mace_extensions,
        }[name]

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)
    monkeypatch.setattr(extract_mace_data_module, "_import_mace_module", fake_import)

    assert _load_mace_checkpoint("field.model", "cpu") is checkpoint
    assert load_count == 2
    assert not hasattr(mace_models, "ScaleShiftFieldMACE")


def test_legacy_macefield_checkpoint_preserves_existing_alias(monkeypatch):
    checkpoint = object()
    existing_alias = object()
    mace_models = SimpleNamespace(ScaleShiftFieldMACE=existing_alias)
    mace_extensions = SimpleNamespace(MACEField=object())
    load_count = 0

    def fake_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            raise AttributeError(
                "Can't get attribute 'ScaleShiftFieldMACE' on "
                "<module 'mace.modules.models'>"
            )
        assert mace_models.ScaleShiftFieldMACE is existing_alias
        return checkpoint

    def fake_import(name):
        return {
            "mace.modules.models": mace_models,
            "mace.modules.extensions": mace_extensions,
        }[name]

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)
    monkeypatch.setattr(extract_mace_data_module, "_import_mace_module", fake_import)

    assert _load_mace_checkpoint("field.model", "cpu") is checkpoint
    assert mace_models.ScaleShiftFieldMACE is existing_alias


def test_legacy_macefield_checkpoint_cleans_alias_after_retry_error(monkeypatch):
    mace_models = SimpleNamespace()
    mace_field = type("MACEField", (), {})
    mace_extensions = SimpleNamespace(MACEField=mace_field)
    retry_error = RuntimeError("checkpoint payload is corrupt")
    load_count = 0

    def fake_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            raise AttributeError(
                "Can't get attribute 'ScaleShiftFieldMACE' on "
                "<module 'mace.modules.models'>"
            )
        assert mace_models.ScaleShiftFieldMACE is mace_field
        raise retry_error

    def fake_import(name):
        return {
            "mace.modules.models": mace_models,
            "mace.modules.extensions": mace_extensions,
        }[name]

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)
    monkeypatch.setattr(extract_mace_data_module, "_import_mace_module", fake_import)

    with pytest.raises(RuntimeError) as error:
        _load_mace_checkpoint("field.model", "cpu")

    assert error.value is retry_error
    assert not hasattr(mace_models, "ScaleShiftFieldMACE")


def test_legacy_macefield_checkpoint_reports_missing_optional_module(monkeypatch):
    mace_models = SimpleNamespace()

    def fake_load(*args, **kwargs):
        raise AttributeError(
            "Can't get attribute 'ScaleShiftFieldMACE' on "
            "<module 'mace.modules.models'>"
        )

    def fake_import(name):
        if name == "mace.modules.models":
            return mace_models
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)
    monkeypatch.setattr(extract_mace_data_module, "_import_mace_module", fake_import)

    with pytest.raises(RuntimeError, match="separate mace-field package"):
        _load_mace_checkpoint("field.model", "cpu")
    assert not hasattr(mace_models, "ScaleShiftFieldMACE")


def test_direct_macefield_checkpoint_reports_missing_optional_module(monkeypatch):
    def fake_load(*args, **kwargs):
        raise ModuleNotFoundError(name="mace.modules.extensions")

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)

    with pytest.raises(RuntimeError, match="separate mace-field package"):
        _load_mace_checkpoint("field.model", "cpu")


def test_macefield_import_failure_inside_installed_extension_is_preserved(
    monkeypatch,
):
    mace_models = SimpleNamespace()
    import_error = ModuleNotFoundError(name="field_extension_dependency")

    def fake_load(*args, **kwargs):
        raise AttributeError(
            "Can't get attribute 'ScaleShiftFieldMACE' on "
            "<module 'mace.modules.models'>"
        )

    def fake_import(name):
        if name == "mace.modules.models":
            return mace_models
        raise import_error

    monkeypatch.setattr(extract_mace_data_module.torch, "load", fake_load)
    monkeypatch.setattr(extract_mace_data_module, "_import_mace_module", fake_import)

    with pytest.raises(ModuleNotFoundError) as error:
        _load_mace_checkpoint("field.model", "cpu")

    assert error.value is import_error


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_macefield_extractor_includes_field_schema(macefield_model_path, tmp_path):
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=8,
    )

    assert data["model_type"] == "MACEField"
    assert data["has_field_coupling"] is True
    assert data["head"] == "mp-dielectric"
    assert data["prediction_heads"]["default"] == "mp-dielectric"
    assert data["prediction_heads"]["names"] == [
        "pt_head",
        "mp-dielectric",
        "mp-ferroelectric",
    ]
    assert set(data["prediction_heads"]["parameters"]) == set(
        data["prediction_heads"]["names"]
    )

    model_path = tmp_path / "multihead.json"
    model_path.write_text(json.dumps(data, separators=(",", ":")))
    default_model = _reference_evaluator(model_path)
    alternate_model = _reference_evaluator(model_path, "mp-ferroelectric")
    assert default_model.selected_head == "mp-dielectric"
    assert alternate_model.selected_head == "mp-ferroelectric"
    assert default_model.available_heads == data["prediction_heads"]["names"]
    assert alternate_model.atomic_energies == pytest.approx(
        data["prediction_heads"]["parameters"]["mp-ferroelectric"]["atomic_energies"]
    )

    legacy_data = dict(data)
    legacy_data.pop("prediction_heads")
    legacy_path = tmp_path / "legacy-single-head.json"
    legacy_path.write_text(json.dumps(legacy_data, separators=(",", ":")))
    legacy_model = _reference_evaluator(legacy_path)
    assert legacy_model.selected_head == "mp-dielectric"
    assert legacy_model.available_heads == ["mp-dielectric"]
    assert len(data["field_couplings"]) == 1

    coupling = data["field_couplings"][0]
    assert coupling["schema_version"] == 1
    assert coupling["field_feats_irreps_in1"] == "128x0e+128x1o"
    assert coupling["field_feats_irreps_in2"] == "1x1o"
    assert coupling["field_feats_irreps_out"] == "128x0e+128x1o"
    assert coupling["field_linear_irreps_in"] == "128x0e+128x1o"
    assert coupling["field_linear_irreps_out"] == "128x0e+128x1o"

    assert len(coupling["field_feats_weight"]) == 32768
    assert len(coupling["field_feats_output_mask"]) == 512
    assert len(coupling["field_linear_weight"]) == 32768
    assert coupling["field_linear_bias"] == []
    assert len(coupling["field_linear_output_mask"]) == 512
    assert [
        instruction["wigner_3j_shape"]
        for instruction in coupling["field_feats_instructions"]
    ] == [[1, 3, 3], [3, 3, 1]]
    assert all(
        len(instruction["wigner_3j"]) == np.prod(instruction["wigner_3j_shape"])
        for instruction in coupling["field_feats_instructions"]
    )

    assert len(data["H1_product_weights"]) == 32768
    assert len(data["H1_linear_up_weights"]) == 32768

    assert data["symmetrix_format_version"] == 4
    assert data["radial_representation"] == "compact"
    assert "radial_spline_values_0" not in data
    assert "radial_spline_values_1" not in data
    assert "A0_spline_values" not in data
    assert "A1_spline_values" not in data

    radial = data["compact_radial"]
    assert radial["num_spline_points"] == 8
    assert radial["basis"]["type"] == "bessel"
    assert radial["cutoff"]["type"] == "polynomial"
    assert radial["distance_transform"]["type"] == "agnesi"
    assert len(radial["distance_transform"]["covalent_radii"]) == 2
    assert radial["networks"]["R0"]["shape"] == [10, 64, 64, 64, 512]
    assert radial["networks"]["R1"]["shape"] == [10, 64, 64, 64, 1280]
    assert radial["networks"]["A0"]["postprocess"] == "tanh-square"
    assert radial["networks"]["A1"]["postprocess"] == "tanh-square"

    m0_contract = data["execution_contracts"]["M0"]
    assert normalize_standard_m0_contract(m0_contract) == m0_contract
    assert m0_contract["interaction"] == "M0"
    assert m0_contract["channels"] == 128
    assert m0_contract["type_count"] == 2
    assert m0_contract["input_l_max"] == 3
    assert m0_contract["output_l_max"] == 1
    assert m0_contract["correlation"] == 3
    assert [group["term_count"] for group in m0_contract["monomial_groups"]] == [
        94,
        118,
        92,
        118,
    ]
    assert m0_contract["structure_fingerprint"].startswith("sha256:")

    m1_contract = data["execution_contracts"]["M1"]
    assert normalize_generated_m1_contract(m1_contract) == m1_contract
    assert m1_contract["interaction"] == "M1"
    assert m1_contract["channels"] == 128
    assert m1_contract["type_count"] == 2
    assert m1_contract["input_l_max"] == 3
    assert m1_contract["output_l_max"] == 0
    assert m1_contract["correlation"] == 3
    assert m1_contract["monomial_groups"][0]["term_count"] == len(data["M1_monomials"])

    r0_contract = data["execution_contracts"]["R0"]
    assert normalize_standard_r0_contract(r0_contract) == r0_contract
    assert r0_contract["interaction"] == "R0"
    assert r0_contract["channels"] == 128
    assert r0_contract["radial_embedding"] == 64
    assert [len(group["path_indices"]) for group in r0_contract["groups"]] == [
        1,
        1,
        1,
        1,
    ]
    assert {path["connection_mode"] for path in r0_contract["paths"]} == {"uvu"}
    assert r0_contract["sparse_coupling"]["term_count"] == 16
    assert r0_contract["derivatives"] == ["forward", "coordinate"]

    contract = data["execution_contracts"]["R1"]
    assert normalize_jit_r1_contract(contract) == contract
    assert contract["schema"] == "symmetrix.execution.tensor_product"
    assert contract["version"] == 1
    assert contract["channels"] == 128
    assert contract["radial_embedding"] == 64
    assert [len(group["path_indices"]) for group in contract["groups"]] == [
        2,
        3,
        3,
        2,
    ]
    assert {path["connection_mode"] for path in contract["paths"]} == {"uvu"}
    assert contract["sparse_coupling"]["term_count"] == 86
    assert contract["generation_fingerprint"] == (
        "sha256:29f6298ae84d2005c56e536f8a54d7202eef90c9fc30fafaf8d2e8270d365fde"
    )


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_macefield_extractor_emits_v4_pair_splines(
    macefield_model_path,
    caplog,
):
    caplog.set_level(logging.WARNING)
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=8,
        radial_format="pair-splines",
    )

    assert data["symmetrix_format_version"] == 4
    assert data["h1_affine_mode"] == "pre_linear_up"
    assert data["radial_representation"] == "pair-splines"
    assert "compact_radial" not in data
    assert "execution_contracts" not in data
    assert data["radial_spline_min"] == pytest.approx(1e-12)
    assert data["A0_spline_min"] == pytest.approx(1e-12)
    assert data["A1_spline_min"] == pytest.approx(1e-12)
    assert len(data["radial_spline_values_0"]) == 3
    assert len(data["radial_spline_values_1"]) == 3
    assert len(data["A0_spline_values"]) == 3
    assert len(data["A1_spline_values"]) == 3
    assert "Generating format-v4 pair-spline radial tables" in caplog.text


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_v4_pair_splines_retain_low_node_count_support(macefield_model_path):
    for num_spline_points in (2, 3):
        data = extract_mace_data(
            macefield_model_path,
            species=[7],
            head="mp-dielectric",
            num_spline_points=num_spline_points,
            radial_format="pair-splines",
        )

        assert len(data["radial_spline_values_0"][0][0]) == num_spline_points
        assert len(data["radial_spline_derivs_0"][0][0]) == num_spline_points
        assert len(data["A0_spline_values"][0]) == num_spline_points
        assert len(data["A1_spline_values"][0]) == num_spline_points


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_macefield_extractor_defaults_to_all_checkpoint_elements(macefield_model_path):
    data = extract_mace_data(
        macefield_model_path,
        head="mp-dielectric",
        num_spline_points=8,
    )

    assert len(data["atomic_numbers"]) == 79
    assert data["num_elements"] == 79
    assert data["atomic_numbers"] == sorted(data["atomic_numbers"])
    assert len(data["compact_radial"]["distance_transform"]["covalent_radii"]) == 79
    assert "radial_spline_values_0" not in data
    assert "radial_spline_values_1" not in data


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_compact_radial_values_match_pair_splines(
    macefield_model_path,
    tmp_path,
):
    from symmetrix import symmetrix as native_symmetrix

    compact = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=32,
    )
    legacy = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=32,
        radial_format="pair-splines",
    )
    compact_path = tmp_path / "compact.json"
    legacy_path = tmp_path / "legacy.json"
    compact_path.write_text(json.dumps(compact, separators=(",", ":")))
    legacy_path.write_text(json.dumps(legacy))

    compact_model = _reference_evaluator(compact_path)
    # Format-v4 pair-spline data remains on materialized execution because
    # streamed edges require the compact radial representation.
    with pytest.warns(UserWarning, match="pair-spline radial representation"):
        legacy_model = native_symmetrix.MACEKokkos(str(legacy_path))
    compact_model.prepare_active_types(
        np.asarray([0, 1, 0, 1], dtype=np.int32)[::2],
    )
    assert compact_model.active_atomic_numbers == [7]
    grid_min = compact["compact_radial"]["spline_grid_min"]
    h = legacy["radial_spline_h"]
    rng = np.random.default_rng(17)
    radii = np.concatenate(
        (
            grid_min + h * np.arange(32),
            rng.uniform(grid_min, compact_model.r_cut - h, size=17),
        )
    )
    node_types = np.asarray([0, 1], dtype=np.int32)
    num_neigh = np.asarray([len(radii), len(radii)], dtype=np.int32)
    neigh_types = np.concatenate(
        (
            np.full(len(radii), 1, dtype=np.int32),
            np.full(len(radii), 0, dtype=np.int32),
        )
    )
    pair_radii = np.tile(radii, 2)

    # The retired serial evaluator's R0_deriv/R1_deriv buffers have no retained
    # equivalent, so only the values are compared: both models materialize the
    # same spline nodes, and the compact runtime derives its derivatives from
    # those nodes.
    for method_name, values_name in (
        ("compute_R0", "R0"),
        ("compute_R1", "R1"),
    ):
        for model in (compact_model, legacy_model):
            getattr(model, method_name)(
                2,
                node_types,
                num_neigh,
                neigh_types,
                pair_radii,
            )
        assert np.allclose(
            getattr(compact_model, values_name),
            getattr(legacy_model, values_name),
            rtol=0.0,
            atol=1e-11,
        )
        assert np.all(np.isfinite(getattr(compact_model, values_name)))


def _two_layer_correlation3_mace_model():
    from mace import modules

    with torch.serialization.safe_globals([slice]):
        from e3nn import o3

    torch.manual_seed(123)
    model = modules.ScaleShiftMACE(
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
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([-13.6, -1000.0, -2040.0]),
        avg_num_neighbors=8.0,
        atomic_numbers=[1, 6, 8],
        correlation=3,
        radial_type="bessel",
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
    )
    return model.eval()


def _flip_u_column(contraction, correlation, column):
    # Negating one U column and its weights leaves the MACE function unchanged,
    # but a recomputed U no longer matches the fitted weights.
    if correlation == contraction.correlation:
        weights = contraction.weights_max
    else:
        weights = contraction.weights[contraction.correlation - correlation - 1]
    with torch.no_grad():
        getattr(contraction, f"U_matrix_{correlation}")[..., column] *= -1
        weights[:, column, :] *= -1


def _symmetrix_energy_forces(data, path):
    from ase import Atoms
    from symmetrix import Symmetrix

    path.write_text(json.dumps(data, separators=(",", ":")))
    atoms = Atoms(
        "COHHO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.21, 0.05, -0.03],
            [-0.62, 0.91, 0.12],
            [-0.58, -0.94, 0.21],
            [2.3, 1.1, 0.6],
        ],
        cell=[14.0, 14.0, 14.0],
        pbc=True,
    )
    atoms.calc = Symmetrix(str(path), dtype="float64", streamed_edges="non-compiled")
    return atoms.get_potential_energy(), atoms.get_forces().copy()


@pytest.mark.parametrize(
    ("product", "contraction", "correlation", "column"),
    [(0, 1, 1, 0), (0, 1, 3, 2), (1, 0, 1, 0), (1, 0, 3, 3)],
    ids=["M0-l1-nu1", "M0-l1-nu3", "M1-nu1", "M1-nu3"],
)
def test_export_takes_u_column_signs_from_the_checkpoint(
    tmp_path, product, contraction, correlation, column
):
    from symmetrix.extract_mace_data import _extract_single_head_data

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = _two_layer_correlation3_mace_model()
        reference = _extract_single_head_data(
            model, radial_format="compact", num_spline_points=64
        )
        target = model.products[product].symmetric_contractions.contractions[
            contraction
        ]
        _flip_u_column(target, correlation, column)
        flipped = _extract_single_head_data(
            model, radial_format="compact", num_spline_points=64
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    reference_energy, reference_forces = _symmetrix_energy_forces(
        reference, tmp_path / "reference.json"
    )
    flipped_energy, flipped_forces = _symmetrix_energy_forces(
        flipped, tmp_path / "flipped.json"
    )
    assert flipped_energy == pytest.approx(reference_energy, abs=1e-12)
    np.testing.assert_allclose(flipped_forces, reference_forces, rtol=0, atol=1e-12)
    # Byte-identical export; compare digests so a failure prints no weights.
    digests = [
        hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("flipped.json", "reference.json")
    ]
    assert digests[0] == digests[1]


def test_export_refuses_a_u_buffer_in_another_basis():
    from symmetrix.extract_mace_data import _extract_single_head_data

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = _two_layer_correlation3_mace_model()
        target = model.products[0].symmetric_contractions.contractions[1]
        with torch.no_grad():
            target.U_matrix_3[..., [0, 1]] = target.U_matrix_3[..., [1, 0]].clone()
        with pytest.raises(
            RuntimeError,
            match=(
                r"products\[0\] contraction 1 \(irrep 1o\), correlation 3 does "
                r"not match the checkpoint's U_matrix_3 buffer in column\(s\) \[0, 1\]"
            ),
        ):
            _extract_single_head_data(
                model, radial_format="compact", num_spline_points=64
            )
    finally:
        torch.set_default_dtype(previous_dtype)


def _small_mace_model(**options):
    """A small two-interaction ScaleShiftMACE with ``options`` applied."""
    import inspect

    from mace import modules

    with torch.serialization.safe_globals([slice]):
        from e3nn import o3

    plain = options.pop("plain", False)
    accepted = inspect.signature(modules.ScaleShiftMACE.__init__).parameters
    accepted = set(accepted) | set(inspect.signature(modules.MACE.__init__).parameters)
    missing = sorted(set(options) - accepted)
    if missing:
        pytest.skip(f"the installed mace does not provide {missing}")
    kwargs = dict(
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
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([-1.1, -2.2, -3.3]),
        avg_num_neighbors=8.0,
        atomic_numbers=[1, 6, 8],
        correlation=3,
        radial_type="bessel",
        use_reduced_cg=False,
    )
    if not plain:
        kwargs.update(atomic_inter_shift=0.1, atomic_inter_scale=1.3)
    kwargs.update(options)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(7)
        model = (modules.MACE if plain else modules.ScaleShiftMACE)(**kwargs)
    finally:
        torch.set_default_dtype(previous_dtype)
    return model.eval()


def _mace_energy_forces_stress(model, atoms):
    from mace.calculators import MACECalculator

    atoms = atoms.copy()
    atoms.calc = MACECalculator(models=[model], device="cpu", default_dtype="float64")
    return (
        atoms.get_potential_energy(),
        atoms.get_forces().copy(),
        atoms.get_stress().copy(),
    )


def _bias_readout_cls():
    from mace.modules import blocks

    if not hasattr(blocks, "NonLinearBiasReadoutBlock"):
        pytest.skip("the installed mace does not provide NonLinearBiasReadoutBlock")
    return blocks.NonLinearBiasReadoutBlock


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (lambda: {"gate": torch.tanh}, "non-linear readout gate tanh"),
        (lambda: {"gate": torch.abs}, "non-linear readout gate abs"),
        (lambda: {"gate": None}, "non-linear readout gate None"),
        (
            lambda: {"readout_cls": _bias_readout_cls()},
            "readout_cls NonLinearBiasReadoutBlock",
        ),
        (
            lambda: {
                "embedding_specs": {
                    "total_charge": {
                        "type": "categorical",
                        "per": "graph",
                        "num_classes": 5,
                        "emb_dim": 2,
                        "offset": 2,
                    }
                }
            },
            "embedding_specs",
        ),
        (lambda: {"use_agnostic_product": True}, "use_agnostic_product"),
        (lambda: {"use_last_readout_only": True}, "use_last_readout_only"),
        (lambda: {"plain": True}, "no scale_shift"),
    ],
    ids=[
        "tanh-gate",
        "abs-gate",
        "no-gate",
        "bias-readout",
        "joint-embedding",
        "agnostic-product",
        "last-readout-only",
        "plain-mace",
    ],
)
def test_export_refuses_mace_options_the_json_cannot_represent(options, message):
    from symmetrix.extract_mace_data import _extract_single_head_data

    model = _small_mace_model(**options())
    with pytest.raises(
        ValueError, match=f"cannot export MACE models with (the )?{message}"
    ):
        _extract_single_head_data(model, radial_format="compact", num_spline_points=64)


def test_silu_readout_export_matches_mace(tmp_path):
    from ase import Atoms
    from symmetrix import Symmetrix
    from symmetrix.extract_mace_data import _extract_single_head_data

    model = _small_mace_model()
    data = _extract_single_head_data(model, radial_format="compact")
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    atoms = Atoms(
        "COHHO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.21, 0.05, -0.03],
            [-0.62, 0.91, 0.12],
            [-0.58, -0.94, 0.21],
            [2.3, 1.1, 0.6],
        ],
        cell=[6.0, 6.5, 7.0],
        pbc=True,
    )
    reference_energy, reference_forces, reference_stress = _mace_energy_forces_stress(
        model, atoms
    )
    atoms.calc = Symmetrix(str(path), dtype="float64", streamed_edges="non-compiled")
    assert atoms.get_potential_energy() == pytest.approx(
        reference_energy, rel=0, abs=1e-8 * len(atoms)
    )
    np.testing.assert_allclose(atoms.get_forces(), reference_forces, rtol=0, atol=1e-6)
    np.testing.assert_allclose(atoms.get_stress(), reference_stress, rtol=0, atol=1e-6)
