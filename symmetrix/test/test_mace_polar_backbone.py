"""PolarMACE backbone-delta extraction against the upstream oracle classes.

Builds a small PolarMACE-shaped model in process — a ``ScaleShiftMACE``
backbone with ``keep_last_layer_irreps=True`` plus the polar modules attached
directly from ``mace.modules.field_blocks`` — and runs the in-tree extractor
and contracts over it. ``PolarMACE.__init__`` itself requires the
out-of-tree ``graph_longrange`` package, so the polar modules are attached to
the parent backbone instead; the extractor is attribute-driven and cannot
tell the difference. The pinned oracle checkout named by
``SYMMETRIX_POLAR_ORACLE_MACE`` is preferred for the import; see
``docs/user/models.md#mace-polar-1`` for the model contract. A pip-installed
``mace-torch`` of the same vintage also satisfies the import.

Runs only where torch, e3nn, and mace are importable; otherwise every test
skips with the missing prerequisite named.
"""

from __future__ import annotations

import json
import sys

import pytest
from polar_oracle_checkout import optional_checkout


def test_shared_polar_extraction_preserves_requested_spline_resolution(tmp_path):
    if polar_import_error is not None:
        pytest.skip(str(polar_import_error))
    from symmetrix.extract_mace_data import extract_mace_data

    checkpoint = tmp_path / "polar.model"
    torch.save(_build_polar_shaped_model(), checkpoint)
    data = extract_mace_data(checkpoint, num_spline_points=8)
    assert data["model_type"] == "MACE_Polar"
    assert data["radial_embedding"]["num_spline_points"] == 8


_ORACLE_MACE_CHECKOUT = optional_checkout("SYMMETRIX_POLAR_ORACLE_MACE")

if (
    _ORACLE_MACE_CHECKOUT is not None
    and _ORACLE_MACE_CHECKOUT.is_dir()
    and str(_ORACLE_MACE_CHECKOUT) not in sys.path
):
    sys.path.insert(0, str(_ORACLE_MACE_CHECKOUT))

try:
    import mace  # noqa: F401  (also sets the weights-only env e3nn requires)
    import numpy
    import torch
    from e3nn import o3
    from mace.modules.blocks import (
        NonLinearBiasReadoutBlock,
        NonLinearReadoutBlock,
        RealAgnosticResidualNonLinearInteractionBlock,
    )
    from mace.modules.field_blocks import (
        AgnosticEmbeddedOneBodyVariableUpdate,
        EnvironmentDependentSpinSourceBlock,
        MultiLayerFeatureMixer,
        OneBodyMLPFieldReadout,
    )
    from mace.modules.models import ScaleShiftMACE

    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data
    from symmetrix.extract_mace_polar import extract_mace_polar_data
    from symmetrix.mace_nonlinear_execution_contract import (
        MH1_CONTRACT_KEY,
        make_execution_mh1_v4_contract,
        normalize_execution_mh1_contract,
        normalize_execution_mh1_v4_contract,
        validate_execution_mh1_contract_for_model,
    )
    from symmetrix.mace_polar_execution_contract import (
        POLAR_CONTRACT_KEY,
        make_execution_polar_mh1_v4_contract,
        normalize_execution_polar_backbone_contract,
        polar_execution_schedule,
        validate_execution_polar_backbone_contract_for_model,
    )

    polar_import_error = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    polar_import_error = exc
    mace = None
    numpy = None
    torch = None
    o3 = None
    ScaleShiftMACE = None
    NonLinearBiasReadoutBlock = None
    NonLinearReadoutBlock = None
    RealAgnosticResidualNonLinearInteractionBlock = None
    EnvironmentDependentSpinSourceBlock = None
    MultiLayerFeatureMixer = None
    AgnosticEmbeddedOneBodyVariableUpdate = None
    OneBodyMLPFieldReadout = None
    extract_mace_nonlinear_data = None
    extract_mace_polar_data = None
    MH1_CONTRACT_KEY = None
    POLAR_CONTRACT_KEY = None
    make_execution_polar_mh1_v4_contract = None
    make_execution_mh1_v4_contract = None
    normalize_execution_mh1_contract = None
    normalize_execution_mh1_v4_contract = None
    normalize_execution_polar_backbone_contract = None
    validate_execution_mh1_contract_for_model = None
    validate_execution_polar_backbone_contract_for_model = None


HIDDEN_IRREPS = "4x0e+4x1o"
CHARGES_IRREPS = "1x0e+1x1o+1x0e+1x1o"
FIELD_IRREPS = "1x0e"
POTENTIAL_IRREPS = "1x0e+1x0e"


def _require_polar_stack():
    if polar_import_error is not None:
        pytest.skip(
            "PolarMACE synthetic extraction requires torch/e3nn/mace: "
            f"{polar_import_error}"
        )


def _build_polar_shaped_model(
    *,
    num_interactions=2,
    steps=2,
    multipoles_max_l=1,
    readout_cls=None,
    hidden_irreps=HIDDEN_IRREPS,
):
    _require_polar_stack()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(20261014)
        hidden = o3.Irreps(hidden_irreps)
        if readout_cls is None:
            readout_cls = NonLinearReadoutBlock
        model = ScaleShiftMACE(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=2,
            interaction_cls=RealAgnosticResidualNonLinearInteractionBlock,
            interaction_cls_first=RealAgnosticResidualNonLinearInteractionBlock,
            num_interactions=num_interactions,
            num_elements=2,
            hidden_irreps=hidden,
            MLP_irreps=o3.Irreps("2x0e"),
            atomic_energies=numpy.zeros((1, 2)),
            avg_num_neighbors=4.0,
            atomic_numbers=[1, 14],
            correlation=3,
            gate=torch.nn.functional.silu,
            use_agnostic_product=True,
            edge_irreps=o3.Irreps("2x0e+2x1o"),
            use_edge_irreps_first=True,
            radial_MLP=[7, 9],
            heads=["test"],
            readout_cls=readout_cls,
            keep_last_layer_irreps=True,
        )
        charges_irreps = 2 * o3.Irreps.spherical_harmonics(multipoles_max_l)
        field_irreps = o3.Irreps("1x0e")
        update_sh_irreps = o3.Irreps.spherical_harmonics(2)
        field_interaction_irreps = (
            (update_sh_irreps * hidden.count(o3.Irrep(0, 1))).sort()[0].simplify()
        )
        model.lr_source_maps = torch.nn.ModuleList(
            EnvironmentDependentSpinSourceBlock(hidden, max_l=multipoles_max_l)
            for _ in range(num_interactions)
        )
        model.layer_feature_mixer = MultiLayerFeatureMixer(hidden, num_interactions)
        model.field_dependent_charges_maps = torch.nn.ModuleList(
            AgnosticEmbeddedOneBodyVariableUpdate(
                node_attrs_irreps=o3.Irreps("2x0e"),
                node_feats_irreps=hidden,
                edge_attrs_irreps=update_sh_irreps,
                edge_feats_irreps=o3.Irreps("8x0e"),
                target_irreps=field_interaction_irreps,
                hidden_irreps=hidden,
                avg_num_neighbors=4.0,
                potential_irreps=field_irreps * 2,
                charges_irreps=charges_irreps,
                field_norm_factor=1.0,
            )
            for _ in range(steps)
        )
        # The chain's two ends: the pre-loop Fukui source map and the
        # post-loop local-electron readout.  ``PolarMACE.__init__`` builds both
        # unconditionally, so a model without them is not a PolarMACE model and
        # the extractor refuses it by name.
        model.fukui_source_map = NonLinearBiasReadoutBlock(
            hidden,
            o3.Irreps("16x0e"),
            torch.nn.functional.silu,
            o3.Irreps("2x0e"),
        )
        model.local_electron_energy = OneBodyMLPFieldReadout(
            node_attrs_irreps=o3.Irreps("2x0e"),
            node_feats_irreps=hidden,
            edge_attrs_irreps=update_sh_irreps,
            edge_feats_irreps=o3.Irreps("8x0e"),
            target_irreps=field_interaction_irreps,
            hidden_irreps=hidden,
            avg_num_neighbors=4.0,
            potential_irreps=field_irreps * 2,
            charges_irreps=charges_irreps,
        )
        model.add_local_electron_energy = True
        model.include_electrostatic_self_interaction = False
        model.quadrupole_feature_corrections = False
        model.field_si = False
        model.num_recursion_steps = steps
        model.keep_last_layer_irreps = True
        model.atomic_multipoles_max_l = multipoles_max_l
        model.atomic_multipoles_smearing_width = 1.5
        model.field_feature_max_l = 0
        model.field_feature_widths = [1.5]
        model.field_feature_norms = torch.ones(1, dtype=torch.float64)
        model.field_norm_factor = 1.0
        model.kspace_cutoff = torch.tensor(4.2, dtype=torch.float64)
        model.charges_irreps = charges_irreps
        model.field_irreps = field_irreps
        model.potential_irreps = field_irreps * 2
        return model
    finally:
        torch.set_default_dtype(previous_dtype)


def test_polar_shaped_model_extracts_backbone_deltas():
    model = _build_polar_shaped_model()
    data = extract_mace_polar_data(model)
    polar = data["polar"]

    assert polar["keep_last_layer_irreps"] is True
    assert polar["num_recursion_steps"] == 2
    assert polar["atomic_multipoles_max_l"] == 1
    assert polar["atomic_multipoles_smearing_width"] == 1.5
    assert polar["field_feature_max_l"] == 0
    assert polar["field_feature_widths"] == [1.5]
    assert polar["field_feature_norms"]["shape"] == [1]
    assert polar["kspace_cutoff"] == 4.2
    assert polar["charges_irreps"] == CHARGES_IRREPS
    assert polar["field_irreps"] == FIELD_IRREPS
    assert polar["potential_irreps"] == POTENTIAL_IRREPS

    # keep_last_layer_irreps: the last readout consumes the full hidden irreps.
    assert data["readouts"][1]["class"] == "NonLinearReadoutBlock"
    assert data["readouts"][1]["linear_1"]["irreps_in"] == HIDDEN_IRREPS
    assert data["products"][1]["target_irreps"] == HIDDEN_IRREPS

    heads = polar["spin_source_heads"]
    assert [head["class"] for head in heads] == [
        "EnvironmentDependentSpinSourceBlock"
    ] * 2
    assert all(head["zero_charges"] is False for head in heads)
    assert [head["linear"]["irreps_in"] for head in heads] == [HIDDEN_IRREPS] * 2
    assert [head["linear"]["irreps_out"] for head in heads] == [CHARGES_IRREPS] * 2

    mixer = polar["feature_mixer"]
    assert mixer["class"] == "MultiLayerFeatureMixer"
    assert [linear["irreps_in"] for linear in mixer["linears"]] == [HIDDEN_IRREPS] * 2
    assert [linear["irreps_out"] for linear in mixer["linears"]] == [HIDDEN_IRREPS] * 2

    update = polar["field_update_blocks"][0]
    assert update["class"] == "AgnosticEmbeddedOneBodyVariableUpdate"
    embedding = update["potential_embedding"]
    assert embedding["class"] == "AgnosticChargeBiasedLinearPotentialEmbedding"
    # Q/S input-channel embedding: charges -> node features.
    assert embedding["charge_embedding"]["irreps_in"] == CHARGES_IRREPS
    assert embedding["charge_embedding"]["irreps_out"] == HIDDEN_IRREPS
    assert embedding["potential_linear"]["irreps_in"] == POTENTIAL_IRREPS
    assert embedding["potential_linear"]["irreps_out"] == HIDDEN_IRREPS
    assert embedding["node_feats_linear"]["irreps_in"] == HIDDEN_IRREPS
    assert embedding["node_feats_linear"]["irreps_out"] == HIDDEN_IRREPS

    assert update["source_embedding"]["irreps_in"] == "2x0e"
    assert update["source_embedding"]["irreps_out"] == "4x0e"
    assert update["dot_products"]["irreps_in1"] == HIDDEN_IRREPS
    assert update["dot_products"]["irreps_in2"] == HIDDEN_IRREPS
    assert update["dot_products"]["irreps_out"] == "4x0e"
    assert update["tp_out"]["irreps_in1"] == HIDDEN_IRREPS
    assert update["tp_out"]["irreps_in2"] == "4x0e"
    assert update["tp_out"]["irreps_out"] == HIDDEN_IRREPS
    assert [layer["type"] for layer in update["nonlinearity"]["layers"]] == [
        "linear",
        "layer_norm",
        "silu",
    ] * 3 + ["linear"]

    readout = update["readout"]
    assert readout["class"] == "GeneralNonLinearBiasReadoutBlock"
    assert readout["linear_1"]["irreps_in"] == HIDDEN_IRREPS
    assert readout["linear_2"]["irreps_out"] == CHARGES_IRREPS + "+2x0e"
    assert readout["gate"]["scalar_activation"] == "silu"
    assert readout["gate"]["gate_activation"] == "sigmoid"
    assert update["field_norm_factor"] == 1.0


def test_polar_extraction_embeds_admissible_contracts():
    model = _build_polar_shaped_model()
    data = extract_mace_polar_data(model)

    assert set(data["execution_contracts"]) == {MH1_CONTRACT_KEY, POLAR_CONTRACT_KEY}
    mh1 = data["execution_contracts"][MH1_CONTRACT_KEY]
    polar = data["execution_contracts"][POLAR_CONTRACT_KEY]
    assert normalize_execution_mh1_contract(mh1) == mh1
    assert normalize_execution_polar_backbone_contract(polar) == polar
    assert validate_execution_mh1_contract_for_model(data, mh1) == mh1
    assert validate_execution_polar_backbone_contract_for_model(data, polar) == polar
    v4 = make_execution_polar_mh1_v4_contract(data)
    assert v4["node_program"]["layers"][1]["product"]["irreps"]["output"] == (
        HIDDEN_IRREPS
    )
    assert normalize_execution_mh1_v4_contract(v4) == v4

    round_trip = json.loads(json.dumps(data))
    assert (
        normalize_execution_mh1_contract(
            round_trip["execution_contracts"][MH1_CONTRACT_KEY]
        )
        == mh1
    )
    assert (
        normalize_execution_polar_backbone_contract(
            round_trip["execution_contracts"][POLAR_CONTRACT_KEY]
        )
        == polar
    )


def test_polar_extraction_is_deterministic_across_weight_draws():
    first = extract_mace_polar_data(_build_polar_shaped_model())
    second = extract_mace_polar_data(_build_polar_shaped_model())
    assert (
        first["execution_contracts"][POLAR_CONTRACT_KEY]
        == second["execution_contracts"][POLAR_CONTRACT_KEY]
    )
    assert (
        first["execution_contracts"][MH1_CONTRACT_KEY]
        == second["execution_contracts"][MH1_CONTRACT_KEY]
    )


def test_polar_three_interaction_model_embeds_the_admitted_contracts():
    """The lifted admission embeds MH-1 and polar contracts for three layers.

    The legacy admission refused exactly-two topologies, so a three-interaction
    checkpoint extracted with the polar backbone but no MH-1 execution
    contract; now the extractor's optional builder admits it, and the embedded
    MH-1 identity validates against the model definition with the count it
    declares.  The polar contract carries the unrolled chain for the model's
    own recursion depth.
    """

    model = _build_polar_shaped_model(num_interactions=3)
    data = extract_mace_polar_data(model)

    polar = data["polar"]
    assert len(polar["spin_source_heads"]) == 3
    assert len(polar["feature_mixer"]["linears"]) == 3
    assert len(data["interactions"]) == 3

    backbone = data["execution_contracts"][POLAR_CONTRACT_KEY]
    assert backbone["interaction_count"] == 3
    assert backbone["execution"] == polar_execution_schedule(
        backbone["recursion_step_count"]
    )
    assert normalize_execution_polar_backbone_contract(backbone) == backbone

    mh1 = data["execution_contracts"][MH1_CONTRACT_KEY]
    assert mh1["interaction_count"] == 3
    assert [item["index"] for item in mh1["interactions"]] == [0, 1, 2]
    assert normalize_execution_mh1_contract(mh1) == mh1
    assert validate_execution_mh1_contract_for_model(data, mh1) == mh1


def test_polar_extraction_rejects_missing_keep_last_flag():
    model = _build_polar_shaped_model()
    model.keep_last_layer_irreps = False
    with pytest.raises(RuntimeError, match="keep_last_layer_irreps"):
        extract_mace_polar_data(model)


def test_polar_extraction_rejects_foreign_update_blocks():
    model = _build_polar_shaped_model()
    model.field_dependent_charges_maps[0] = (
        MultiLayerFeatureMixer(  # wrong class on purpose
            o3.Irreps(HIDDEN_IRREPS), 1
        )
    )
    with pytest.raises(RuntimeError, match="AgnosticEmbeddedOneBodyVariableUpdate"):
        extract_mace_polar_data(model)


def test_polar_extraction_rejects_mismatched_head_counts():
    model = _build_polar_shaped_model()
    model.lr_source_maps = torch.nn.ModuleList([model.lr_source_maps[0]])
    with pytest.raises(RuntimeError, match="spin-source head counts"):
        extract_mace_polar_data(model)


def test_plain_backbone_extracts_without_polar_data():
    model = _build_polar_shaped_model()
    del model.lr_source_maps
    del model.layer_feature_mixer
    del model.field_dependent_charges_maps
    data = extract_mace_nonlinear_data(model)
    assert "polar" not in data
    assert set(data["execution_contracts"]) == {MH1_CONTRACT_KEY}
    with pytest.raises(ValueError, match="MH-1 support matrix"):
        make_execution_mh1_v4_contract(data)
