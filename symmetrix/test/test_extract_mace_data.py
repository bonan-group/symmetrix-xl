import importlib
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    extract_mace_data_module = importlib.import_module("symmetrix.extract_mace_data")
    from symmetrix.execution_contract import (
        normalize_jit_r1_contract,
        normalize_standard_m0_contract,
        normalize_standard_r0_contract,
    )
    from symmetrix.extract_mace_data import (
        _load_mace_checkpoint,
        _remove_prediction_head,
        extract_mace_data,
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

    from symmetrix import symmetrix as native_symmetrix

    model_path = tmp_path / "multihead.json"
    model_path.write_text(json.dumps(data, separators=(",", ":")))
    default_model = native_symmetrix.MACE(str(model_path))
    alternate_model = native_symmetrix.MACE(str(model_path), "mp-ferroelectric")
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
    legacy_model = native_symmetrix.MACE(str(legacy_path))
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

    assert data["symmetrix_format_version"] == 2
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
def test_macefield_extractor_retains_legacy_pair_splines(
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

    assert "symmetrix_format_version" not in data
    assert "compact_radial" not in data
    assert "execution_contracts" not in data
    assert data["radial_spline_min"] == pytest.approx(1e-12)
    assert data["A0_spline_min"] == pytest.approx(1e-12)
    assert data["A1_spline_min"] == pytest.approx(1e-12)
    assert len(data["radial_spline_values_0"]) == 3
    assert len(data["radial_spline_values_1"]) == 3
    assert len(data["A0_spline_values"]) == 3
    assert len(data["A1_spline_values"]) == 3
    assert "Generating legacy Symmetrix format-v1" in caplog.text


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_legacy_pair_splines_retain_low_node_count_support(macefield_model_path):
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
def test_compact_radial_values_and_derivatives_match_pair_splines(
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

    compact_model = native_symmetrix.MACE(str(compact_path))
    legacy_model = native_symmetrix.MACE(str(legacy_path))
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

    for method_name, values_name, derivatives_name in (
        ("compute_R0", "R0", "R0_deriv"),
        ("compute_R1", "R1", "R1_deriv"),
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
        assert np.allclose(
            getattr(compact_model, derivatives_name),
            getattr(legacy_model, derivatives_name),
            rtol=0.0,
            atol=1e-9,
        )
        assert np.all(np.isfinite(getattr(compact_model, values_name)))
        assert np.all(np.isfinite(getattr(compact_model, derivatives_name)))
