"""Exporter honesty: unsupported configurations must not silently change a model.

The exporter used to drop unsupported weightless radial activations, substitute
zero tensors for missing modules, and hardcode ZBL screening constants, all while
still validating and compiling.  It also published an all-zero interpolation
certificate that read like a passed gate, ignored the declared table precision in
the contract fingerprint, and had no way to record the certified precision.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from symmetrix.cli.extract_nequip import _parser
from symmetrix.extract_nequip_data import (
    _ExportPolicy,
    _oam_radial_evaluator,
    _oam_zbl,
    _serialize_linear,
    _serialize_oam_radial,
    extract_nequip_oam_data,
)
from symmetrix.jit_codegen_common import spline_certification
from symmetrix.nequip_probe import NequIPProbeError
from symmetrix.nequip_oam_graph import normalize_nequip_oam_model
from symmetrix.nequip_schema import (
    load_nequip_oam_data,
    nequip_contract_fingerprint,
    validate_nequip_oam_data,
)


OAM_FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"


def _radial(layers):
    class EdgeMLP:
        pass

    EdgeMLP.layers = layers
    return EdgeMLP()


class _LinearLayer:
    weight = [[1.0]]
    bias = [0.0]


class _ShiftedSoftPlus:
    pass


class _Silu:
    pass


class _ChildContainer:
    def __init__(self, **children):
        self._children = children
        for name, value in children.items():
            setattr(self, name, value)

    def named_children(self):
        return tuple(self._children.items())


class _ConvNetLayer(_ChildContainer):
    is_convnet_layer = True


class _InteractionBlock(_ChildContainer):
    is_interaction = True


def _synthetic_oam_model(radial_layers):
    class _NodeTypeEmbed:
        type_names = ("H",)
        embed_module = SimpleNamespace(weight=[[1.0]])

    class _EdgeNorm:
        r_max = 4.5

    class _Bessel:
        num_bessels = 1
        bessel_weights = [1.0]
        cutoff = SimpleNamespace(p=6)

    class _Factor:
        factor = 1.0

    interaction = _InteractionBlock(edge_mlp=_radial(radial_layers))
    return _ChildContainer(
        type_embed=_NodeTypeEmbed(),
        edge_norm=_EdgeNorm(),
        bessel_encode=_Bessel(),
        factor=_Factor(),
        layer0=_ConvNetLayer(interaction=interaction),
    )


def test_non_silu_weightless_radial_activation_is_rejected():
    model = _synthetic_oam_model([_LinearLayer(), _ShiftedSoftPlus()])

    with pytest.raises(NequIPProbeError, match=r"ShiftedSoftPlus.*only SiLU"):
        extract_nequip_oam_data(model)


def test_silu_weightless_radial_activation_is_accepted():
    model = _synthetic_oam_model([_LinearLayer(), _Silu()])
    graph = normalize_nequip_oam_model(model)

    radial = _serialize_oam_radial(
        graph,
        model.layer0.interaction,
        policy=_ExportPolicy(strict=True),
    )

    assert [layer["type"] for layer in radial["mlp"]["layers"]] == ["linear", "silu"]


def test_missing_internal_weight_is_rejected_in_strict_mode():
    with pytest.raises(NequIPProbeError, match=r"no internal path weight"):
        _serialize_linear(
            SimpleNamespace(), "1x0e", "1x0e", policy=_ExportPolicy(strict=True)
        )


def test_missing_weight_is_recorded_rather_than_hidden():
    policy = _ExportPolicy(strict=False)

    _serialize_linear(SimpleNamespace(), "1x0e", "1x0e", policy=policy)

    assert policy.defaults
    assert any("weight defaulted to zeros" in item for item in policy.defaults)
    assert any("bias defaulted to zeros" in item for item in policy.defaults)


def test_missing_zbl_module_is_rejected():
    model = SimpleNamespace()

    with pytest.raises(NequIPProbeError, match=r"no ZBL/pair-repulsion module"):
        _oam_zbl(model, SimpleNamespace())


def test_substituted_zbl_constants_are_recorded():
    graph = SimpleNamespace(
        type_names=("H",),
        r_cut=4.5,
        edge_embedding=SimpleNamespace(cutoff_p=6.0),
    )
    model = SimpleNamespace(zbl=SimpleNamespace(cutoff=4.5))
    policy = _ExportPolicy(strict=True)

    zbl = _oam_zbl(model, graph, policy=policy)

    assert zbl["screening_exponents"] == [-0.20162, -0.40290, -0.94229, -3.19980]
    assert zbl["cutoff_p"] == pytest.approx(6.0)
    # cutoff_p, half-energy prefactor, screening exponents, screening coefficients.
    assert len(policy.defaults) == 4
    assert any("library constant" in item for item in policy.defaults)


def test_packaged_oam_xl_pair_potential_is_exported_as_zbl():
    graph = SimpleNamespace(
        type_names=("H",),
        r_cut=6.0,
        edge_embedding=SimpleNamespace(cutoff_p=6.0),
    )
    model = SimpleNamespace(pair_potential=SimpleNamespace(cutoff=6.0))

    zbl = _oam_zbl(model, graph)

    assert zbl["type"] == "zbl"
    assert zbl["cutoff"] == pytest.approx(6.0)


def test_explicit_zbl_constants_are_used_without_defaults():
    graph = SimpleNamespace(
        type_names=("H",),
        r_cut=4.5,
        edge_embedding=SimpleNamespace(cutoff_p=6.0),
    )
    source = SimpleNamespace(
        cutoff=4.5,
        cutoff_p=6,
        half_energy_prefactor=7.2,
        screening_exponents=[-1.0],
        screening_coefficients=[0.5],
    )
    policy = _ExportPolicy(strict=True)

    zbl = _oam_zbl(SimpleNamespace(zbl=source), graph, policy=policy)

    assert zbl["screening_exponents"] == [-1.0]
    assert zbl["half_energy_prefactor"] == pytest.approx(7.2)
    assert policy.defaults == []


def _certification(num_points):
    contract = json.loads(OAM_FIXTURE.read_text(encoding="utf-8"))
    cutoff = float(contract["r_cut"])
    evaluate, _ = _oam_radial_evaluator(
        contract["layers"][0]["radial_basis"], cutoff=cutoff
    )
    return spline_certification(
        evaluate,
        num_points=num_points,
        grid_min=0.0,
        cutoff=cutoff,
    )


def test_interpolation_certificate_reports_measured_errors():
    """A zero certificate would mean the qualification never ran."""

    certificate = _certification(256)

    assert certificate["grid_points"] == 256
    assert certificate["clamped_points"] == 0
    assert 0.0 < certificate["max_abs_value_error"] < 1.0e-2
    assert 0.0 < certificate["max_abs_derivative_error"] < 1.0e-1


def test_measured_error_shrinks_with_a_finer_grid():
    coarse = _certification(256)
    fine = _certification(1024)

    assert fine["max_abs_value_error"] < coarse["max_abs_value_error"]
    assert fine["max_abs_derivative_error"] < coarse["max_abs_derivative_error"]


def test_declared_table_precision_participates_in_the_fingerprint():
    data = load_nequip_oam_data(OAM_FIXTURE)
    baseline = nequip_contract_fingerprint(data)

    changed = copy.deepcopy(data)
    changed["layers"][0]["kokkos_radial"]["precision"] = "float32"
    validate_nequip_oam_data(changed)

    assert nequip_contract_fingerprint(changed) != baseline


def test_tensor_placement_tags_do_not_participate_in_the_fingerprint():
    data = load_nequip_oam_data(OAM_FIXTURE)
    baseline = nequip_contract_fingerprint(data)

    changed = copy.deepcopy(data)
    changed["layers"][0]["pre_linear"]["weight"]["dtype"] = "float32"
    changed["layers"][0]["pre_linear"]["weight"]["device"] = "cuda:0"

    assert nequip_contract_fingerprint(changed) == baseline


def test_archive_location_does_not_participate_in_the_fingerprint():
    """The same packaged model must match wheel artifacts after relocation."""

    data = load_nequip_oam_data(OAM_FIXTURE)
    data["model_metadata"]["archive"] = "/build-host/models/NequIP-OAM-S.nequip.zip"
    baseline = nequip_contract_fingerprint(data)

    relocated = copy.deepcopy(data)
    relocated["model_metadata"]["archive"] = (
        "/home/user/.cache/models/NequIP-OAM-S.nequip.zip"
    )

    assert nequip_contract_fingerprint(relocated) == baseline


def test_archive_location_exclusion_does_not_hide_learned_parameter_changes():
    data = load_nequip_oam_data(OAM_FIXTURE)
    data["model_metadata"]["archive"] = "/build-host/model.nequip.zip"
    baseline = nequip_contract_fingerprint(data)

    changed = copy.deepcopy(data)
    changed["model_metadata"]["archive"] = "/another-host/model.nequip.zip"
    values = changed["layers"][0]["edge_tensor_product"]["weight"]["values"]
    values[0] = float(values[0]) + 0.125

    assert nequip_contract_fingerprint(changed) != baseline


def test_cli_records_the_certified_precision():
    parser = _parser()

    assert parser.parse_args(["--model", "m", "--output", "o"]).precision == "float64"
    assert (
        parser.parse_args(
            ["--model", "m", "--output", "o", "--precision", "float32"]
        ).precision
        == "float32"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "m", "--output", "o", "--precision", "float16"])
