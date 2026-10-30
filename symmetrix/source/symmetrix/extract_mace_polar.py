"""Checkpoint extraction for the PolarMACE family.

The family's models are MH-1 backbones carrying the PolarMACE deltas.  This
module reuses the neutral nonlinear backbone serialization, then owns the
``polar`` delta payload and ``POLAR_BACKBONE`` execution contract in addition
to the model-type stamp the registry detects and the frontend schema validation
(:mod:`symmetrix.mace_polar_schema`) that runs before the native parse.

A torch checkpoint reaches this module either directly (the family's declared
extractor) or through :func:`symmetrix.extract_mace_data.extract_mace_data`,
which routes a polar model here by the same structural probe used to decide
whether the polar deltas exist at all -- the extractor is attribute-driven and
does not need ``PolarMACE``'s own class, whose constructor requires the
out-of-tree ``graph_longrange`` package.
"""

from __future__ import annotations

from typing import Any

from .extract_mace_nonlinear import (
    _affine_mlp,
    _gate,
    _linear,
    _readout,
    _tensor,
    _tensor_product,
    extract_mace_nonlinear_data,
)
from .mace_polar_execution_contract import (
    POLAR_CONTRACT_KEY,
    make_execution_polar_backbone_contract,
)
from .mace_polar_schema import POLAR_MODEL_TYPE, validate_mace_polar_model_data

__all__ = [
    "POLAR_MODEL_TYPE",
    "is_polar_mace_model",
    "extract_mace_polar_data",
]

#: The attributes that make a model a PolarMACE model.  They are the polar
#: modules ``PolarMACE.__init__`` attaches -- the per-layer spin-source
#: multipole heads, the multi-layer feature mixer, and the field-dependent
#: charge-update chain -- so the probe is structural and independent of
#: ``PolarMACE`` itself.
_POLAR_BACKBONE_ATTRIBUTES = (
    "lr_source_maps",
    "layer_feature_mixer",
    "field_dependent_charges_maps",
)


def _spin_source_head(module) -> dict[str, Any]:
    return {
        "class": type(module).__name__,
        "zero_charges": bool(module.zero_charges),
        "linear": _linear(module.linear),
    }


def _feature_mixer(module) -> dict[str, Any]:
    return {
        "class": type(module).__name__,
        "linears": [_linear(linear) for linear in module.linears],
    }


def _potential_embedding(module) -> dict[str, Any]:
    return {
        "class": type(module).__name__,
        "potential_linear": _linear(module.potential_linear),
        "node_feats_linear": _linear(module.node_feats_linear),
        "charge_embedding": _linear(module.charge_embedding),
    }


def _gated_readout(module) -> dict[str, Any]:
    return {
        "class": type(module).__name__,
        "linear_1": _linear(module.linear_1),
        "linear_mid": _linear(module.linear_mid),
        "linear_2": _linear(module.linear_2),
        "gate": _gate(module.equivariant_nonlin),
    }


def _field_update_block(module) -> dict[str, Any]:
    if type(module).__name__ != "AgnosticEmbeddedOneBodyVariableUpdate":
        raise RuntimeError(
            "PolarMACE extraction only supports "
            "AgnosticEmbeddedOneBodyVariableUpdate field-update blocks; found "
            f"{type(module).__name__}."
        )
    embedding = module.potential_embedding
    if type(embedding).__name__ != "AgnosticChargeBiasedLinearPotentialEmbedding":
        raise RuntimeError(
            "PolarMACE extraction only supports "
            "AgnosticChargeBiasedLinearPotentialEmbedding potential embeddings; "
            f"found {type(embedding).__name__}."
        )
    readout = module.readout
    if type(readout).__name__ != "GeneralNonLinearBiasReadoutBlock":
        raise RuntimeError(
            "PolarMACE extraction only supports "
            "GeneralNonLinearBiasReadoutBlock field-update readouts; found "
            f"{type(readout).__name__}."
        )
    if type(module.nonlinearity).__name__ != "RadialMLP":
        raise RuntimeError(
            "PolarMACE extraction only supports RadialMLP field-update "
            f"nonlinearities; found {type(module.nonlinearity).__name__}."
        )
    for name in ("dot_products", "tp_out"):
        if type(getattr(module, name)).__name__ != "SparseUvuTensorProduct":
            raise RuntimeError(
                "PolarMACE extraction only supports SparseUvuTensorProduct "
                f"field-update tensor products; found "
                f"{type(getattr(module, name)).__name__} for {name}."
            )
    return {
        "class": type(module).__name__,
        "potential_embedding": _potential_embedding(embedding),
        "source_embedding": _linear(module.source_embedding),
        "dot_products": _tensor_product(module.dot_products),
        "nonlinearity": _affine_mlp(module.nonlinearity),
        "tp_out": _tensor_product(module.tp_out),
        "readout": _gated_readout(readout),
        "field_norm_factor": float(module.field_norm_factor),
    }


def _local_electron_readout(module) -> dict[str, Any]:
    """Serialize the post-loop local-electron-energy readout.

    The block is the oracle's ``OneBodyMLPFieldReadout``: two biased dense
    embeddings of the charge density and of the potential features (~l up),
    two node-local sparse ``uvu`` tensor products against the backbone's node
    features, and one affine MLP over the concatenated invariants.  The block
    accepts edge arguments it never reads (``edge_set: "empty"``), so the
    serialization carries no edge tensor.
    """

    if type(module).__name__ != "OneBodyMLPFieldReadout":
        raise RuntimeError(
            "PolarMACE extraction only supports OneBodyMLPFieldReadout "
            f"local-electron readouts; found {type(module).__name__}."
        )
    for name in ("dot_products_q", "dot_products_v"):
        if type(getattr(module, name)).__name__ != "SparseUvuTensorProduct":
            raise RuntimeError(
                "PolarMACE extraction only supports SparseUvuTensorProduct "
                f"local-electron tensor products; found "
                f"{type(getattr(module, name)).__name__} for {name}."
            )
    return {
        "class": type(module).__name__,
        "linear_up_q": _linear(module.linear_up_q),
        "linear_up_v": _linear(module.linear_up_v),
        "dot_products_q": _tensor_product(module.dot_products_q),
        "dot_products_v": _tensor_product(module.dot_products_v),
        "mlp": _affine_mlp(module.mlp),
    }


def _extract_polar_backbone_data(model) -> dict[str, Any] | None:
    """Serialize the PolarMACE backbone deltas when the model carries them."""

    if not all(
        hasattr(model, name)
        for name in (
            "lr_source_maps",
            "layer_feature_mixer",
            "field_dependent_charges_maps",
        )
    ):
        return None
    if not bool(getattr(model, "keep_last_layer_irreps", False)):
        raise RuntimeError(
            "PolarMACE extraction requires keep_last_layer_irreps backbone readouts."
        )
    heads = list(model.lr_source_maps)
    if len(heads) != len(model.interactions):
        raise RuntimeError(
            "PolarMACE model has mismatched interaction and spin-source head counts."
        )
    for head in heads:
        if type(head).__name__ != "EnvironmentDependentSpinSourceBlock":
            raise RuntimeError(
                "PolarMACE extraction only supports "
                "EnvironmentDependentSpinSourceBlock spin-source heads; found "
                f"{type(head).__name__}."
            )
    mixer = model.layer_feature_mixer
    if type(mixer).__name__ != "MultiLayerFeatureMixer":
        raise RuntimeError(
            "PolarMACE extraction only supports MultiLayerFeatureMixer feature "
            f"mixers; found {type(mixer).__name__}."
        )
    if len(mixer.linears) != len(model.interactions):
        raise RuntimeError(
            "PolarMACE model has mismatched interaction and feature-mixer counts."
        )
    steps = int(model.num_recursion_steps)
    updates = list(model.field_dependent_charges_maps)
    if len(updates) != steps:
        raise RuntimeError(
            "PolarMACE model has mismatched recursion and field-update counts."
        )
    # The pre-loop Fukui source map and the post-loop local-electron readout
    # are the chain's two ends: the source map turns the backbone's last-layer
    # features into the per-graph-normalized Fukui sources the equilibration
    # broadcasts, and the readout turns the equilibrated density and the last
    # iteration's potential features into the per-node local-electron energy.
    # The energy flag is what decides whether that term enters the total.
    return {
        "class": type(model).__name__,
        "keep_last_layer_irreps": True,
        "num_recursion_steps": steps,
        "add_local_electron_energy": bool(
            getattr(model, "add_local_electron_energy", False)
        ),
        "include_electrostatic_self_interaction": bool(
            getattr(model, "include_electrostatic_self_interaction", False)
        ),
        "quadrupole_feature_corrections": bool(
            getattr(model, "quadrupole_feature_corrections", False)
        ),
        "field_si": bool(getattr(model, "field_si", False)),
        "fukui_source_map": _readout(model.fukui_source_map),
        "local_electron_energy": _local_electron_readout(model.local_electron_energy),
        "atomic_multipoles_max_l": int(model.atomic_multipoles_max_l),
        "atomic_multipoles_smearing_width": float(
            model.atomic_multipoles_smearing_width
        ),
        "field_feature_max_l": int(model.field_feature_max_l),
        "field_feature_widths": [float(width) for width in model.field_feature_widths],
        "field_feature_norms": _tensor(model.field_feature_norms),
        "field_norm_factor": float(model.field_norm_factor),
        "kspace_cutoff": float(model.kspace_cutoff),
        "charges_irreps": str(model.charges_irreps),
        "field_irreps": str(model.field_irreps),
        "potential_irreps": str(model.potential_irreps),
        "spin_source_heads": [_spin_source_head(head) for head in heads],
        "feature_mixer": _feature_mixer(mixer),
        "field_update_blocks": [_field_update_block(update) for update in updates],
    }


def is_polar_mace_model(model: Any) -> bool:
    """Report whether a torch model carries the PolarMACE backbone deltas."""

    return all(hasattr(model, name) for name in _POLAR_BACKBONE_ATTRIBUTES)


def extract_mace_polar_data(
    model: Any,
    atomic_numbers: list[int] | None = None,
    *,
    num_spline_points: int | None = None,
) -> dict[str, Any]:
    """Return the versioned polar schema for a PolarMACE checkpoint.

    The backbone serialization is shared; this entry point adds the family
    payload and contract, stamps the family's model type, and validates the
    result so a PolarMACE checkpoint and an MH-1 checkpoint cannot be confused.
    """

    if not is_polar_mace_model(model):
        raise RuntimeError(
            "PolarMACE extraction requires the polar backbone deltas "
            f"({', '.join(_POLAR_BACKBONE_ATTRIBUTES)}); this model does not "
            "carry them."
        )
    data = extract_mace_nonlinear_data(
        model, atomic_numbers, num_spline_points=num_spline_points
    )
    data["polar"] = _extract_polar_backbone_data(model)
    try:
        data.setdefault("execution_contracts", {})[POLAR_CONTRACT_KEY] = (
            make_execution_polar_backbone_contract(data)
        )
    except ValueError as error:
        raise RuntimeError(
            "PolarMACE backbone extraction produced an invalid program "
            f"identity: {error}"
        ) from error
    data["model_type"] = POLAR_MODEL_TYPE
    validate_mace_polar_model_data(data)
    return data
