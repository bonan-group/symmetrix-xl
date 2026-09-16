import importlib
import itertools
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline

with torch.serialization.safe_globals([slice]):
    from e3nn.o3 import Irreps, Linear, wigner_3j
from ase.data import chemical_symbols
from mace.modules.radial import ZBLBasis
from mace.tools.cg import U_matrix_real
from mace.tools.scripts_utils import remove_pt_head

from .execution_contract import (
    make_factorized_contract,
    make_standard_m0_contract,
    make_standard_r0_contract,
)
from .prediction_heads import PREDICTION_HEADS_SCHEMA_VERSION


_MACE_FIELD_MODULE = "mace.modules.extensions"
_MACE_MODELS_MODULE = "mace.modules.models"
_OMAT24_AVG_NUM_NEIGHBORS = 61.964672446250916


def _is_single_layer_training_checkpoint(value):
    if not isinstance(value, dict) or not isinstance(value.get("model"), dict):
        return False
    state = value["model"]
    return (
        int(state.get("num_interactions", -1)) == 1
        and "readout_products.0.symmetric_contractions.weight" in state
        and "products.0.symmetric_contractions.weight" in state
    )


def _unpack_cueq_symmetric_contraction(state, target_state, prefix):
    packed_key = f"{prefix}.symmetric_contractions.weight"
    packed = state[packed_key]
    offset = 0
    contraction = 0
    while True:
        base = f"{prefix}.symmetric_contractions.contractions.{contraction}"
        maximum_key = f"{base}.weights_max"
        if maximum_key not in target_state:
            break
        keys = [maximum_key]
        weight = 0
        while f"{base}.weights.{weight}" in target_state:
            keys.append(f"{base}.weights.{weight}")
            weight += 1
        for key in keys:
            width = target_state[key].shape[1]
            target_state[key] = packed[:, offset : offset + width, :].reshape(
                target_state[key].shape
            )
            offset += width
        contraction += 1
    if contraction == 0 or offset != packed.shape[1]:
        raise RuntimeError(
            f"Cannot unpack CuEq contraction {prefix}: consumed {offset} of "
            f"{packed.shape[1]} packed coefficients."
        )


def _reconstruct_single_layer_training_checkpoint(checkpoint, model_path, device):
    state = checkpoint["model"]
    if "without-self" not in Path(model_path).name:
        raise RuntimeError(
            "This single-layer training checkpoint does not record whether its first "
            "interaction has a self connection; reconstruction currently requires a "
            "checkpoint name containing 'without-self'."
        )

    from e3nn import o3
    from mace import modules

    channels = (
        state["node_embedding.linear.weight"].numel() // state["atomic_numbers"].numel()
    )
    hidden_irreps = o3.Irreps(f"{channels}x0e + {channels}x1o + {channels}x2e")
    readout_hidden = state["readouts.0.linear_2.weight"].numel()
    radial_layers = []
    layer = 1
    while f"interactions.0.conv_tp_weights.layer{layer}.weight" in state:
        radial_layers.append(
            state[f"interactions.0.conv_tp_weights.layer{layer}.weight"].shape[0]
        )
        layer += 1
    if not radial_layers:
        raise RuntimeError("The single-layer checkpoint has no radial MLP layers.")

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(state["node_embedding.linear.weight"].dtype)
    try:
        model = modules.ScaleShiftMACE(
            r_max=float(state["r_max"]),
            num_bessel=state["radial_embedding.bessel_fn.bessel_weights"].numel(),
            num_polynomial_cutoff=int(state["radial_embedding.cutoff_fn.p"]),
            max_ell=3,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticDensityInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticDensityInteractionBlock"
            ],
            num_interactions=1,
            num_elements=state["atomic_numbers"].numel(),
            hidden_irreps=hidden_irreps,
            MLP_irreps=o3.Irreps(f"{readout_hidden}x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=state["atomic_energies_fn.atomic_energies"].cpu().numpy(),
            avg_num_neighbors=_OMAT24_AVG_NUM_NEIGHBORS,
            atomic_numbers=state["atomic_numbers"].cpu().numpy(),
            correlation=3,
            readout_correlation=2,
            use_last_readout_only=True,
            atomic_inter_scale=state["scale_shift.scale"].cpu().numpy(),
            atomic_inter_shift=state["scale_shift.shift"].cpu().numpy(),
            radial_MLP=radial_layers,
            pair_repulsion="pair_repulsion_fn.c" in state,
            distance_transform=(
                "Agnesi" if "radial_embedding.distance_transform.q" in state else "None"
            ),
            use_reduced_cg=False,
        )
    finally:
        torch.set_default_dtype(old_dtype)

    target_state = model.state_dict()
    _unpack_cueq_symmetric_contraction(state, target_state, "products.0")
    _unpack_cueq_symmetric_contraction(state, target_state, "readout_products.0")
    for key, target in tuple(target_state.items()):
        if "symmetric_contractions" in key or key not in state:
            continue
        source = state[key]
        if source.numel() != target.numel():
            raise RuntimeError(
                f"Checkpoint tensor {key!r} has {source.numel()} values; "
                f"the reconstructed model expects {target.numel()}."
            )
        target_state[key] = source.reshape(target.shape)
    model.load_state_dict(target_state)
    return model.to(device)


def export_mace_model(checkpoint_path, output_path, device="cpu"):
    """Export a loadable e3nn MACE model from a supported training checkpoint."""
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(device), weights_only=False
    )
    if not _is_single_layer_training_checkpoint(checkpoint):
        raise RuntimeError(
            "The input is not a supported single-layer training checkpoint."
        )
    model = _reconstruct_single_layer_training_checkpoint(
        checkpoint, checkpoint_path, torch.device(device)
    )
    torch.save(model, output_path)
    return model


def _requires_legacy_mace_field_alias(exc):
    if not isinstance(exc, AttributeError):
        return False
    message = str(exc)
    return "ScaleShiftFieldMACE" in message and _MACE_MODELS_MODULE in message


def _missing_mace_field_extension(exc):
    if isinstance(exc, ModuleNotFoundError):
        return exc.name == _MACE_FIELD_MODULE
    if isinstance(exc, AttributeError):
        message = str(exc)
        return "MACEField" in message and _MACE_FIELD_MODULE in message
    return False


def _mace_field_dependency_error():
    return RuntimeError(
        "This checkpoint is field-aware and requires "
        "mace.modules.extensions.MACEField. Install the separate mace-field "
        "package/source until its field-aware modules are available in upstream "
        "mace-torch."
    )


def _import_mace_module(name):
    return importlib.import_module(name)


def _load_legacy_mace_field_checkpoint(model_path, device):
    mace_models = _import_mace_module(_MACE_MODELS_MODULE)
    try:
        mace_field_module = _import_mace_module(_MACE_FIELD_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == _MACE_FIELD_MODULE:
            raise _mace_field_dependency_error() from exc
        raise
    try:
        mace_field = mace_field_module.MACEField
    except AttributeError as exc:
        raise _mace_field_dependency_error() from exc

    added_legacy_alias = False
    if not hasattr(mace_models, "ScaleShiftFieldMACE"):
        mace_models.ScaleShiftFieldMACE = mace_field
        added_legacy_alias = True
    try:
        return torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    finally:
        if added_legacy_alias:
            del mace_models.ScaleShiftFieldMACE


def _load_mace_checkpoint(model_path, device):
    """Load an ordinary checkpoint, with a lazy legacy MACEField fallback."""
    try:
        loaded = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
        if _is_single_layer_training_checkpoint(loaded):
            return _reconstruct_single_layer_training_checkpoint(
                loaded, model_path, device
            )
        return loaded
    except Exception as exc:
        if _requires_legacy_mace_field_alias(exc):
            return _load_legacy_mace_field_checkpoint(model_path, device)
        if _missing_mace_field_extension(exc):
            raise _mace_field_dependency_error() from exc
        raise


def _compact_irreps_to_native(values, num_channels, L_max):
    values = np.asarray(values)
    expected = num_channels * (L_max + 1) ** 2
    if values.ndim != 2 or values.shape[1] != expected:
        raise RuntimeError(
            "First-interaction residual has an incompatible H1 dimension: "
            f"expected (*, {expected}), got {values.shape}."
        )
    native = np.zeros(
        (values.shape[0], (L_max + 1) ** 2, num_channels),
        dtype=values.dtype,
    )
    offset = 0
    for l in range(L_max + 1):
        components = 2 * l + 1
        block_size = components * num_channels
        native[:, l * l : (l + 1) ** 2, :] = (-1) ** l * values[
            :, offset : offset + block_size
        ].reshape(values.shape[0], num_channels, components).transpose(0, 2, 1)
        offset += block_size
    return native


def _remove_prediction_head(model, head, device):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(next(model.parameters()).dtype)
        return remove_pt_head(model, head).to(device=device, dtype=torch.float64)
    finally:
        torch.set_default_dtype(previous_dtype)


def _extract_single_head_data(
    model,
    species=None,
    head=None,
    num_spline_points=256,
    radial_format="compact",
):
    """Extract data from pytorch model file into structure that can be
    written as symmetrix JSON data file

    Parameters
    ----------
    model: str / Path
        path to pytorch model file
    species: list(int / str), default None
        list of atomic numbers or chemical symbols to extract for legacy
        standard-MACE models. MACE_Nonlinear models reject nonempty subsets;
        if omitted, retain every element supported by the checkpoint.
    head: str, default None
        head to keep, if multihead model, default same as mace.tools.scripts_utils.remove_pt_head
    num_spline_points: int, default 256
        number of spline points to approximate various functions
    radial_format: str, default "compact"
        compact stores the shared radial model and materializes splines for
        active compositions at runtime. pair-splines stores legacy pair tables.

    Returns
    -------
    output_data: dict with symmetrix model data
    """
    device = torch.device("cpu")
    if not isinstance(model, torch.nn.Module):
        model = _load_mace_checkpoint(model, device)
    model = model.to(device=device, dtype=torch.float64)
    model.eval()

    if radial_format not in ("compact", "pair-splines"):
        raise ValueError(
            f"Unsupported radial_format '{radial_format}'. "
            "Expected 'compact' or 'pair-splines'."
        )
    if radial_format == "compact" and num_spline_points < 4:
        raise ValueError("Compact radial output requires at least 4 spline points.")
    if radial_format == "pair-splines":
        logging.warning(
            "Generating legacy Symmetrix format-v1 pair-spline data. "
            "Use radial_format='compact' for the optimized format-v2 runtime."
        )

    # ensure that splines goes smoothly to 0 at outer cutoff
    spline_bc_type = ("not-a-knot", "clamped")

    ### ----- EXTRACT SINGLE HEAD -----

    if hasattr(model, "heads") and len(model.heads) != 1:
        model = _remove_prediction_head(model, head, device)
        model.eval()
    model_dtype = next(model.parameters()).dtype

    ### ----- CHECK FOR COMPATIBILITY -----

    is_macefield = type(model).__name__ == "MACEField" or (
        hasattr(model, "field_feats") and hasattr(model, "field_linear")
    )

    try:
        from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock
    except ImportError:
        RealAgnosticResidualNonLinearInteractionBlock = ()
    if (
        not is_macefield
        and RealAgnosticResidualNonLinearInteractionBlock
        and len(model.interactions) > 0
        and all(
            isinstance(interaction, RealAgnosticResidualNonLinearInteractionBlock)
            for interaction in model.interactions
        )
    ):
        if species is not None and len(species) > 0:
            raise ValueError(
                "MACE_Nonlinear extraction always serializes the complete "
                "checkpoint species domain; species subsets are unsupported."
            )
        if radial_format != "compact":
            raise ValueError(
                "MACE_Nonlinear models require the format-version-3 compact schema; "
                "pair-splines is only available for legacy MACE interactions."
            )
        from .extract_mace_nonlinear import extract_mace_nonlinear_data

        return extract_mace_nonlinear_data(model)

    if species is None:
        species = []

    # Legacy standard-MACE extraction retains its existing species-selection
    # behavior. The nonlinear branch above has already enforced its full domain.
    atomic_numbers = []
    for sp in species:
        try:
            Z = int(sp)
        except ValueError:
            try:
                Z = chemical_symbols.index(sp)
            except ValueError as exc:
                raise ValueError(
                    "Failed to parse {sp} as atomic number or chemical species"
                ) from exc
        atomic_numbers.append(Z)
    if len(set(atomic_numbers)) != len(atomic_numbers):
        raise ValueError("Duplicate species are not supported during MACE extraction.")

    num_interactions = len(model.interactions)
    single_layer_invariant_readout = (
        num_interactions == 1
        and len(getattr(model, "readout_reshapes", ())) == 1
        and len(getattr(model, "readout_products", ())) == 1
        and len(model.readouts) == 1
    )
    if num_interactions not in (1, 2):
        raise RuntimeError(
            "Currently, symmetrix supports one- and two-interaction MACE models."
        )
    if num_interactions == 1 and not single_layer_invariant_readout:
        raise RuntimeError(
            "One-interaction MACE models require one invariant readout product and "
            "one nonlinear readout."
        )
    if is_macefield and single_layer_invariant_readout:
        raise RuntimeError(
            "Single-layer invariant readouts are currently supported only for MACE."
        )

    if is_macefield:
        if not hasattr(model, "field_feats") or not hasattr(model, "field_linear"):
            raise RuntimeError(
                "MACEField models must have field_feats and field_linear modules."
            )
        if len(model.field_feats) != 1 or len(model.field_linear) != 1:
            raise RuntimeError(
                "Currently, symmetrix only supports MACEField models with one field coupling."
            )

    from mace.modules.blocks import (
        RealAgnosticDensityInteractionBlock,
        RealAgnosticDensityResidualInteractionBlock,
        RealAgnosticInteractionBlock,
        RealAgnosticResidualInteractionBlock,
    )

    first_interaction_residual = isinstance(
        model.interactions[0],
        (
            RealAgnosticResidualInteractionBlock,
            RealAgnosticDensityResidualInteractionBlock,
        ),
    )
    if not isinstance(
        model.interactions[0],
        (
            RealAgnosticInteractionBlock,
            RealAgnosticDensityInteractionBlock,
            RealAgnosticResidualInteractionBlock,
            RealAgnosticDensityResidualInteractionBlock,
        ),
    ):
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models whose first interaction is "
            "a real-agnostic interaction or residual interaction block."
        )

    if (
        num_interactions == 2
        and not isinstance(model.interactions[1], RealAgnosticResidualInteractionBlock)
        and not isinstance(
            model.interactions[1], RealAgnosticDensityResidualInteractionBlock
        )
    ):
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models whose second interaction is "
            "RealAgnosticResidualInteractionBlock or RealAgnosticDensityResidualInteractionBlock."
        )
    if bool(model.products[0].use_sc) != first_interaction_residual:
        raise RuntimeError(
            "First interaction and product self-connection are inconsistent: "
            f"residual interaction={first_interaction_residual}, "
            f"products[0].use_sc={bool(model.products[0].use_sc)}."
        )

    if single_layer_invariant_readout:
        readout_product = model.readout_products[0]
        product_irreps = Irreps(model.products[0].linear.irreps_out)
        expected_product_irreps = Irreps(
            [
                (model.node_embedding.linear.irreps_out.count("0e"), (l, (-1) ** l))
                for l in range(product_irreps.lmax + 1)
            ]
        )
        if product_irreps != expected_product_irreps:
            raise RuntimeError(
                "The one-interaction product must use the canonical equal-channel "
                f"hidden layout '{expected_product_irreps}', got '{product_irreps}'."
            )
        if Irreps(readout_product.symmetric_contractions.irreps_in) != product_irreps:
            raise RuntimeError(
                "The one-interaction invariant readout input must match the product "
                "output layout."
            )
        if bool(readout_product.use_sc):
            raise RuntimeError(
                "The one-interaction invariant readout product must not use a self connection."
            )
        expected_readout_channels = model.node_embedding.linear.irreps_out.count("0e")
        if readout_product.linear.irreps_out != Irreps(
            f"{expected_readout_channels}x0e"
        ):
            raise RuntimeError(
                "The one-interaction invariant readout product must produce one scalar "
                "per model channel."
            )

    if model.spherical_harmonics._lmax != 3:
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models with l_max=3."
        )

    ### ----- HELPER FUNCTION -----

    def linear_simplify(linear):
        simplified = Linear(
            Irreps(linear.irreps_in).simplify(), Irreps(linear.irreps_out).simplify()
        )
        simplified.weight = linear.weight
        simplified.bias = linear.bias
        return simplified

    def serialize_instruction(instruction):
        serialized = {}
        for name in (
            "i_in",
            "i_in1",
            "i_in2",
            "i_out",
            "connection_mode",
            "has_weight",
            "path_weight",
            "path_shape",
        ):
            if hasattr(instruction, name):
                value = getattr(instruction, name)
                if isinstance(value, tuple):
                    value = list(value)
                serialized[name] = value
        return serialized

    def serialize_field_coupling(field_feats, field_linear):
        field_in1 = list(field_feats.irreps_in1)
        field_in2 = list(field_feats.irreps_in2)
        field_out = list(field_feats.irreps_out)

        field_instructions = []
        for instruction in field_feats.instructions:
            serialized = serialize_instruction(instruction)
            l_in = field_in1[instruction.i_in1][1].l
            l_field = field_in2[instruction.i_in2][1].l
            l_out = field_out[instruction.i_out][1].l
            wigner = wigner_3j(l_in, l_field, l_out).to(dtype=torch.float64)
            serialized["wigner_3j_shape"] = list(wigner.shape)
            serialized["wigner_3j"] = wigner.flatten().tolist()
            field_instructions.append(serialized)

        return {
            "schema_version": 1,
            "field_feats_irreps_in1": str(field_feats.irreps_in1),
            "field_feats_irreps_in2": str(field_feats.irreps_in2),
            "field_feats_irreps_out": str(field_feats.irreps_out),
            "field_feats_instructions": field_instructions,
            "field_feats_weight": field_feats.weight.numpy(force=True).tolist(),
            "field_feats_output_mask": field_feats.output_mask.numpy(
                force=True
            ).tolist(),
            "field_linear_irreps_in": str(field_linear.irreps_in),
            "field_linear_irreps_out": str(field_linear.irreps_out),
            "field_linear_instructions": [
                serialize_instruction(instruction)
                for instruction in field_linear.instructions
            ],
            "field_linear_weight": field_linear.weight.numpy(force=True).tolist(),
            "field_linear_bias": field_linear.bias.numpy(force=True).tolist(),
            "field_linear_output_mask": field_linear.output_mask.numpy(
                force=True
            ).tolist(),
        }

    def serialize_e3nn_fully_connected_net(network, name):
        if (
            type(network).__module__ != "e3nn.nn._fc"
            or type(network).__name__ != "FullyConnectedNet"
        ):
            raise RuntimeError(
                f"Compact radial extraction does not support {name} network "
                f"{type(network).__module__}.{type(network).__name__}. "
                "Use radial_format='pair-splines' with an explicit species list."
            )

        layers = list(network._modules.values())
        if len(layers) != len(network.hs) - 1:
            raise RuntimeError(
                f"Compact radial extraction found an invalid {name} network layout."
            )

        weights = []
        activation_scale = None
        for layer_index, layer in enumerate(layers):
            if tuple(layer.weight.shape) != (layer.h_in, layer.h_out):
                raise RuntimeError(
                    f"Compact radial extraction found invalid {name} layer dimensions."
                )
            if getattr(layer, "bias", None) is not None:
                raise RuntimeError(
                    f"Compact radial extraction does not support biases in {name}. "
                    "Use radial_format='pair-splines' with an explicit species list."
                )
            if layer.var_in != 1 or layer.var_out != 1:
                raise RuntimeError(
                    f"Compact radial extraction only supports unit e3nn variances in {name}. "
                    "Use radial_format='pair-splines' with an explicit species list."
                )

            is_final = layer_index == len(layers) - 1
            if is_final:
                if layer.act is not None:
                    raise RuntimeError(
                        f"Compact radial extraction does not support an output activation in {name}."
                    )
            else:
                act = layer.act
                if (
                    act is None
                    or getattr(act, "f", None) is not torch.nn.functional.silu
                ):
                    raise RuntimeError(
                        f"Compact radial extraction only supports normalized SiLU in {name}. "
                        "Use radial_format='pair-splines' with an explicit species list."
                    )
                if activation_scale is None:
                    activation_scale = float(act.cst)
                elif not np.isclose(
                    activation_scale, float(act.cst), rtol=0.0, atol=1e-15
                ):
                    raise RuntimeError(
                        f"Compact radial extraction requires one activation scale across {name}."
                    )

            effective_weight = (
                layer.weight.detach()
                / np.sqrt(
                    float(layer.h_in) * float(layer.var_in) / float(layer.var_out)
                )
            ).T
            weights.append(effective_weight.numpy(force=True).flatten().tolist())

        return {
            "shape": [int(value) for value in network.hs],
            "weights": weights,
            "activation": "silu",
            "activation_scale": 1.0 if activation_scale is None else activation_scale,
        }

    def serialize_compact_radial(model, atomic_numbers):
        from mace.modules.blocks import RadialEmbeddingBlock
        from mace.modules.radial import AgnesiTransform, BesselBasis, PolynomialCutoff

        radial = model.radial_embedding
        if not isinstance(radial, RadialEmbeddingBlock):
            raise RuntimeError(
                "Compact radial extraction requires mace.modules.blocks.RadialEmbeddingBlock. "
                "Use radial_format='pair-splines' with an explicit species list."
            )
        if not isinstance(radial.bessel_fn, BesselBasis):
            raise RuntimeError(
                f"Compact radial extraction does not support basis {type(radial.bessel_fn).__name__}. "
                "Use radial_format='pair-splines' with an explicit species list."
            )
        if not isinstance(radial.cutoff_fn, PolynomialCutoff) or not getattr(
            radial, "apply_cutoff", True
        ):
            raise RuntimeError(
                "Compact radial extraction requires an applied PolynomialCutoff. "
                "Use radial_format='pair-splines' with an explicit species list."
            )

        transform = getattr(radial, "distance_transform", None)
        if transform is None:
            distance_transform = {"type": "none"}
        else:
            if not isinstance(transform, AgnesiTransform):
                raise RuntimeError(
                    f"Compact radial extraction does not support distance transform "
                    f"{type(transform).__name__}. Use radial_format='pair-splines' "
                    "with an explicit species list."
                )
            distance_transform = {
                "type": "agnesi",
                "a": float(transform.a),
                "q": float(transform.q),
                "p": float(transform.p),
                "covalent_radii": [
                    float(transform.covalent_radii[atomic_number])
                    for atomic_number in atomic_numbers
                ],
            }

        networks = {
            "R0": serialize_e3nn_fully_connected_net(
                model.interactions[0].conv_tp_weights, "R0"
            ),
        }
        if num_interactions == 2:
            networks["R1"] = serialize_e3nn_fully_connected_net(
                model.interactions[1].conv_tp_weights, "R1"
            )
        if "Density" in type(model.interactions[0]).__name__:
            networks["A0"] = serialize_e3nn_fully_connected_net(
                model.interactions[0].density_fn, "A0"
            )
            networks["A0"]["postprocess"] = "tanh-square"
        if num_interactions == 2 and "Density" in type(model.interactions[1]).__name__:
            networks["A1"] = serialize_e3nn_fully_connected_net(
                model.interactions[1].density_fn, "A1"
            )
            networks["A1"]["postprocess"] = "tanh-square"

        return {
            "spline_grid_min": 1e-12,
            "num_spline_points": int(num_spline_points),
            "basis": {
                "type": "bessel",
                "weights": radial.bessel_fn.bessel_weights.numpy(force=True).tolist(),
                "prefactor": float(radial.bessel_fn.prefactor),
            },
            "cutoff": {
                "type": "polynomial",
                "r_max": float(radial.cutoff_fn.r_max),
                "p": int(radial.cutoff_fn.p),
            },
            "distance_transform": distance_transform,
            "networks": networks,
        }

    ### ----- BASIC MODEL INFO -----

    num_channels = model.node_embedding.linear.irreps_out.count("0e")
    r_cut = model.r_max.item()
    l_max = model.spherical_harmonics._lmax
    L_max = model.products[0].linear.irreps_out.lmax
    output = {}
    output["num_channels"] = num_channels
    output["r_cut"] = r_cut
    output["l_max"] = l_max
    output["L_max"] = L_max
    output["num_interactions"] = num_interactions
    output["model_type"] = "MACEField" if is_macefield else "MACE"
    output["has_field_coupling"] = is_macefield
    output["first_interaction_residual"] = first_interaction_residual
    output["field_couplings"] = []

    if is_macefield:
        if not 1 <= L_max <= 2:
            raise RuntimeError(
                "MACEField field coupling currently supports only 1 <= L_max <= 2."
            )
        field_feats = model.field_feats[0]
        field_linear = model.field_linear[0]
        expected_hidden_dim = ((L_max + 1) ** 2) * num_channels
        expected_hidden_irreps = Irreps(
            [(num_channels, (l, 1 if l % 2 == 0 else -1)) for l in range(L_max + 1)]
        )
        if field_feats.irreps_in1 != expected_hidden_irreps:
            raise RuntimeError(
                "MACEField field_feats.0 input irreps must be the canonical "
                f"equal-channel layout '{expected_hidden_irreps}'."
            )
        if str(field_feats.irreps_in2) != "1x1o":
            raise RuntimeError(
                "Currently, symmetrix only supports MACEField electric-field irreps '1x1o'."
            )
        if field_feats.irreps_out != expected_hidden_irreps:
            raise RuntimeError(
                "MACEField field_feats.0 output irreps must match the canonical H1 layout."
            )
        if field_linear.irreps_in != expected_hidden_irreps:
            raise RuntimeError(
                "MACEField field_linear.0 input irreps must match the canonical H1 layout."
            )
        if field_linear.irreps_out != expected_hidden_irreps:
            raise RuntimeError(
                "MACEField field_linear.0 output irreps must match the canonical H1 layout."
            )
        if field_feats.irreps_in1.dim != expected_hidden_dim:
            raise RuntimeError(
                "MACEField canonical H1 layout has an unexpected dimension."
            )
        if not torch.all(field_feats.output_mask == 1):
            raise RuntimeError(
                "Currently, symmetrix only supports MACEField field_feats.0 output masks of all ones."
            )
        if not torch.all(field_linear.output_mask == 1):
            raise RuntimeError(
                "Currently, symmetrix only supports MACEField field_linear.0 output masks of all ones."
            )
        if field_linear.bias.numel() != 0:
            raise RuntimeError(
                "Currently, symmetrix only supports bias-free MACEField field_linear.0."
            )
        output["field_couplings"] = [
            serialize_field_coupling(field_feats, field_linear)
        ]

    ### ----- ATOMIC NUMBERS AND ENERGIES -----

    atomic_numbers = sorted(atomic_numbers)
    if len(atomic_numbers) == 0:
        atomic_numbers = sorted(model.atomic_numbers.tolist())
        logging.warning(f"No atomic_numbers, including all: {atomic_numbers}")
    atomic_energies = [
        torch.atleast_1d(model.atomic_energies_fn.atomic_energies.squeeze())[
            model.atomic_numbers.tolist().index(a)
        ].item()
        + model.scale_shift.shift.item()
        for a in atomic_numbers
    ]
    output["atomic_numbers"] = atomic_numbers
    output["num_elements"] = len(atomic_numbers)
    output["atomic_energies"] = atomic_energies

    if radial_format == "compact":
        output["symmetrix_format_version"] = 2
        output["radial_representation"] = "compact"
        output["compact_radial"] = serialize_compact_radial(model, atomic_numbers)

    ### --- ZBL ---

    if hasattr(model, "pair_repulsion") and model.pair_repulsion:
        if not isinstance(model.pair_repulsion_fn, ZBLBasis):
            raise Exception("Only ZBL pair_repulsion is supported.")
        output["has_zbl"] = True
        zbl = model.pair_repulsion_fn
        output["zbl_a_exp"] = zbl.a_exp.item()
        output["zbl_a_prefactor"] = zbl.a_prefactor.item()
        output["zbl_c"] = (
            model.scale_shift.scale.item() * zbl.c.numpy(force=True)
        ).tolist()
        output["zbl_covalent_radii"] = zbl.covalent_radii.numpy(force=True).tolist()
        output["zbl_p"] = zbl.p.item()
    else:
        output["has_zbl"] = False

    ### ----- RADIAL SPLINES -----

    if radial_format == "pair-splines":
        logging.info("R0+R1")
        r, h = np.linspace(1e-12, r_cut, num_spline_points, retstep=True)
        spline_values_0 = []
        spline_derivatives_0 = []
        spline_values_1 = []
        spline_derivatives_1 = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels = model.radial_embedding(
                    torch.tensor(r, dtype=model_dtype).unsqueeze(-1),
                    torch.eye(len(model.atomic_numbers), dtype=model_dtype),
                    torch.tensor([[model_i], [model_j]], dtype=torch.int64),
                    model.atomic_numbers,
                )
                if isinstance(bessels, tuple):
                    bessels = bessels[0]  # newer versions return (bessels, cutoffs)
                # radial basis for interaction 0
                R = model.interactions[0].conv_tp_weights(bessels).numpy(force=True)
                spl_0 = [
                    CubicSpline(r, R[:, k], bc_type=spline_bc_type)
                    for k in range(R.shape[1])
                ]
                spline_values_0.append([spl(r).tolist() for spl in spl_0])
                spline_derivatives_0.append(
                    [spl.derivative()(r).tolist() for spl in spl_0]
                )
                if num_interactions == 2:
                    R = model.interactions[1].conv_tp_weights(bessels).numpy(force=True)
                    spl_1 = [
                        CubicSpline(r, R[:, k], bc_type=spline_bc_type)
                        for k in range(R.shape[1])
                    ]
                    spline_values_1.append([spl(r).tolist() for spl in spl_1])
                    spline_derivatives_1.append(
                        [spl.derivative()(r).tolist() for spl in spl_1]
                    )

        output["radial_spline_h"] = float(h)
        output["radial_spline_min"] = float(r[0])
        output["radial_spline_values_0"] = spline_values_0
        output["radial_spline_derivs_0"] = spline_derivatives_0
        if num_interactions == 2:
            output["radial_spline_values_1"] = spline_values_1
            output["radial_spline_derivs_1"] = spline_derivatives_1

    ### ----- H0 -----

    logging.info("H0")
    H0_weights = (
        np.reshape(
            model.node_embedding.linear.weight.numpy(force=True),
            [len(model.atomic_numbers), num_channels],
        )
        / np.sqrt(len(model.atomic_numbers))
        @ np.reshape(
            model.interactions[0].linear_up.weight.numpy(force=True),
            [num_channels, num_channels],
        )
        / np.sqrt(num_channels)
    )
    indices = [model.atomic_numbers.tolist().index(a) for a in atomic_numbers]
    H0_weights = H0_weights[indices, :]
    output["H0_weights"] = H0_weights.flatten().tolist()

    ### ----- Phi0 -----

    logging.info("Phi0")

    ### ----- A0 -----

    logging.info("A0")
    A0_scaled = True if ("Density" in type(model.interactions[0]).__name__) else False
    output["A0_scaled"] = A0_scaled
    if A0_scaled and radial_format == "pair-splines":
        r, h = np.linspace(1e-12, r_cut, num_spline_points, retstep=True)
        A0_spline_values = []
        A0_spline_derivs = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels = model.radial_embedding(
                    torch.tensor(r, dtype=model_dtype).unsqueeze(-1),
                    torch.eye(len(model.atomic_numbers), dtype=model_dtype),
                    torch.tensor([[model_i], [model_j]], dtype=torch.int64),
                    model.atomic_numbers,
                )
                if isinstance(bessels, tuple):
                    bessels = bessels[0]  # newer versions return (bessels, cutoffs)
                R = torch.tanh(model.interactions[0].density_fn(bessels) ** 2).numpy(
                    force=True
                )
                spl = CubicSpline(r, R[:, 0], bc_type=spline_bc_type)
                A0_spline_values.append(spl(r).tolist())
                A0_spline_derivs.append(spl.derivative()(r).tolist())
        output["A0_spline_h"] = float(h)
        output["A0_spline_min"] = float(r[0])
        output["A0_spline_values"] = A0_spline_values
        output["A0_spline_derivs"] = A0_spline_derivs
    A0_weights = []
    for i, a in enumerate(atomic_numbers):
        A0_weights.append([])
        if first_interaction_residual:
            for _, _, weight in model.interactions[0].linear.weight_views(
                yield_instruction=True
            ):
                message_weight = weight.numpy(force=True) / np.sqrt(num_channels)
                if not A0_scaled:
                    message_weight /= model.interactions[0].avg_num_neighbors
                A0_weights[i].append(message_weight.flatten().tolist())
        else:
            model_i = model.atomic_numbers.tolist().index(a)
            for l, _, w in model.interactions[0].skip_tp.weight_views(
                yield_instruction=True
            ):
                w_linear = model.interactions[0].linear.weight_view_for_instruction(
                    l
                ).numpy(force=True) / np.sqrt(num_channels)
                if not A0_scaled:
                    w_linear /= model.interactions[0].avg_num_neighbors
                fused = (
                    w_linear
                    @ w[:, model_i, :].numpy(force=True)
                    / np.sqrt(len(model.atomic_numbers) * num_channels)
                )
                A0_weights[i].append(fused.flatten().tolist())
    output["A0_weights"] = A0_weights

    #### ----- M0 -----

    logging.info("M0")
    correlation = model.products[0].symmetric_contractions.contractions[0].correlation

    ### Computes U_{lm\eta, l1m1 l2m2 ...}
    #   * `irrep_out` is essentially l_out
    #   * `irreps_in` essentially provides l_in_max
    #   * `corr_in_max` is the max correlation order
    def U_sparse(irrep_out, irreps_in, corr_in_max, input_l_max=l_max):
        U = [[]]  # list of lists because U[0] should be empty
        use_reduced_cg = bool(getattr(model, "use_reduced_cg", False))
        for corr in range(1, corr_in_max + 1):
            # get U matrix for this correlation order
            try:
                U_matrix = U_matrix_real(
                    Irreps(irreps_in),
                    [irrep_out],
                    corr,
                    use_cueq_cg=use_reduced_cg,
                )[1]
            except TypeError:
                U_matrix = U_matrix_real(irreps_in, [irrep_out], corr)[1]
            if irrep_out.l == 0:  # makes U_matrix.shape consistent with l>0 cases
                U_matrix = U_matrix.unsqueeze(0)
            num_eta = U_matrix.shape[-1]
            U_matrix = U_matrix.flatten()
            # extract sparse U for this correlation order
            U_sparse_corr = [
                [{} for _ in range(num_eta)] for _ in range(2 * irrep_out.l + 1)
            ]
            j = 0
            for m in range(2 * irrep_out.l + 1):
                for lm_list in itertools.product(
                    range((input_l_max + 1) ** 2), repeat=corr
                ):
                    for eta in range(num_eta):
                        if abs(U_matrix[j]) > 1e-12:
                            lm_tuple_sorted = tuple(sorted(lm_list))
                            if lm_tuple_sorted not in U_sparse_corr[m][eta].keys():
                                U_sparse_corr[m][eta][lm_tuple_sorted] = 0.0
                            U_sparse_corr[m][eta][lm_tuple_sorted] += U_matrix[j].item()
                        j += 1
            U.append(U_sparse_corr)
        return U

    irreps_in = [ir[1] for ir in model.products[0].symmetric_contractions.irreps_in]
    irreps_out = [ir[1] for ir in model.products[0].symmetric_contractions.irreps_out]
    # The Clebsch-Gordan sparsification depends only on the contraction
    # topology, not on the active atomic species.  Cache it across the
    # per-species coefficient extraction below; this is especially important
    # for foundation models with many elements.
    u_sparse_cache = {}

    def cached_u_sparse(irrep_out):
        key = (
            str(irrep_out),
            tuple(str(value) for value in irreps_in),
            correlation,
            l_max,
        )
        if key not in u_sparse_cache:
            u_sparse_cache[key] = U_sparse(irrep_out, irreps_in, correlation)
        return u_sparse_cache[key]

    C = {}
    M = {}
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        C[i] = {}
        for l, irrep_out in enumerate(irreps_out):
            # extract U in sparse format
            U = cached_u_sparse(irrep_out)
            # extract weights from model
            # warning: slightly odd order of the contractions weights due to reverse countdown
            W = [[]]  # list of lists because W[0] should be empty
            W.append(
                model.products[0]
                .symmetric_contractions.contractions[l]
                .weights[1]
                .numpy(force=True)
            )
            W.append(
                model.products[0]
                .symmetric_contractions.contractions[l]
                .weights[0]
                .numpy(force=True)
            )
            W.append(
                model.products[0]
                .symmetric_contractions.contractions[l]
                .weights_max.numpy(force=True)
            )
            # combine U and W into polynomial-like terms for recursive evaluator
            for m in range(-l, l + 1):
                lm = l * (l + 1) + m
                C[i][lm] = {}
                for k in range(num_channels):
                    P_lmk = {}
                    for corr in range(1, correlation + 1):
                        for eta in range(len(U[corr][l + m])):
                            for key, value in U[corr][l + m][eta].items():
                                if key not in P_lmk.keys():
                                    P_lmk[key] = 0.0
                                P_lmk[key] += float(W[corr][model_i, eta, k]) * value
                    C[i][lm][k] = list(P_lmk.values())
                    M[lm] = [list(key) for key in P_lmk.keys()]
    output["M0_weights"] = C
    output["M0_monomials"] = M

    def extract_scalar_contraction(product, input_l_max):
        contraction = product.symmetric_contractions.contractions[0]
        contraction_correlation = contraction.correlation
        irreps_in = Irreps([ir[1] for ir in product.symmetric_contractions.irreps_in])
        U = U_sparse(
            Irreps("0e")[0].ir,
            irreps_in,
            contraction_correlation,
            input_l_max,
        )

        weights = [[]]
        for corr in range(1, contraction_correlation):
            weights.append(
                contraction.weights[contraction_correlation - corr - 1].numpy(
                    force=True
                )
            )
        weights.append(contraction.weights_max.numpy(force=True))

        coefficients = {}
        monomials = None
        for output_type, atomic_number in enumerate(atomic_numbers):
            model_type = model.atomic_numbers.tolist().index(atomic_number)
            coefficients[output_type] = {}
            for channel in range(num_channels):
                terms = {}
                for corr in range(1, contraction_correlation + 1):
                    for eta, sparse_terms in enumerate(U[corr][0]):
                        for key, value in sparse_terms.items():
                            if any(index >= (input_l_max + 1) ** 2 for index in key):
                                raise RuntimeError(
                                    "Invariant readout contraction references an "
                                    "out-of-range hidden harmonic."
                                )
                            terms[key] = (
                                terms.get(key, 0.0)
                                + float(weights[corr][model_type, eta, channel]) * value
                            )
                coefficients[output_type][channel] = list(terms.values())
                channel_monomials = [list(key) for key in terms]
                if monomials is None:
                    monomials = channel_monomials
                elif monomials != channel_monomials:
                    raise RuntimeError(
                        "Invariant readout contraction channels have inconsistent monomials."
                    )
        return coefficients, monomials or []

    ### ----- H1 -----

    logging.info("H1")
    H1_weights = np.zeros([L_max + 1, num_channels, num_channels])
    weights_0 = np.reshape(
        model.products[0].linear.weight.numpy(force=True),
        [L_max + 1, num_channels, num_channels],
    ) / np.sqrt(num_channels)
    if num_interactions == 2:
        weights_1 = np.reshape(
            model.interactions[1].linear_up.weight.numpy(force=True),
            [L_max + 1, num_channels, num_channels],
        ) / np.sqrt(num_channels)
    else:
        weights_1 = np.repeat(
            np.eye(num_channels, dtype=weights_0.dtype)[None, :, :],
            L_max + 1,
            axis=0,
        )
    for l in range(L_max + 1):
        H1_weights[l, :, :] = weights_0[l, :, :] @ weights_1[l, :, :]
    output["H1_weights"] = H1_weights.flatten().tolist()
    if is_macefield or first_interaction_residual:
        output["H1_product_weights"] = weights_0.flatten().tolist()
        output["H1_linear_up_weights"] = weights_1.flatten().tolist()
    if first_interaction_residual:
        expected_hidden_irreps = Irreps(
            [(num_channels, (l, 1 if l % 2 == 0 else -1)) for l in range(L_max + 1)]
        )
        skip_irreps = Irreps(model.interactions[0].skip_tp.irreps_out)
        if skip_irreps != expected_hidden_irreps:
            raise RuntimeError(
                "First-interaction residual output must use the canonical H1 layout "
                f"'{expected_hidden_irreps}', got '{skip_irreps}'."
            )
        node_attrs = torch.zeros(
            (len(atomic_numbers), len(model.atomic_numbers)),
            dtype=model_dtype,
            device=device,
        )
        for row, atomic_number in enumerate(atomic_numbers):
            node_attrs[row, model.atomic_numbers.tolist().index(atomic_number)] = 1.0
        with torch.no_grad():
            node_feats = model.node_embedding(node_attrs)
            residual = model.interactions[0].skip_tp(node_feats, node_attrs)
        output["H1_first_residual_weights"] = (
            _compact_irreps_to_native(residual.numpy(force=True), num_channels, L_max)
            .flatten()
            .tolist()
        )

    if single_layer_invariant_readout:
        logging.info("single-layer invariant readout")
        readout_product = model.readout_products[0]
        readout_correlation = readout_product.symmetric_contractions.contractions[
            0
        ].correlation
        if readout_correlation != 2:
            raise RuntimeError(
                "Currently, symmetrix supports correlation-2 one-interaction readouts."
            )
        output["single_layer_readout"] = True
        output["readout_correlation"] = readout_correlation
        output["M1_weights"], output["M1_monomials"] = extract_scalar_contraction(
            readout_product, L_max
        )
        output["H2_weights_for_H1"] = [
            [0.0] * (num_channels * num_channels) for _ in atomic_numbers
        ]
        output["H2_weights_for_M1"] = (
            readout_product.linear.weight.numpy(force=True).flatten()
            / np.sqrt(num_channels)
        ).tolist()
        output["readout_1_weights"] = [0.0] * num_channels
        output["Phi1_l"] = []
        output["Phi1_l1"] = []
        output["Phi1_l2"] = []
        output["Phi1_lme"] = []
        output["Phi1_clebsch_gordan"] = []
        output["Phi1_lelm1lm2"] = []
        output["A1_weights"] = [[] for _ in range(l_max + 1)]
        output["A1_scaled"] = False

        readout = model.readouts[0]
        readout_weights_1 = readout.linear_1.weight
        readout_weights_2 = readout.linear_2.weight
        if readout_weights_1.numel() % num_channels != 0:
            raise ValueError(
                "The nonlinear readout input weights are incompatible with the "
                f"{num_channels}-channel model"
            )
        readout_hidden_size = readout_weights_1.numel() // num_channels
        if readout_weights_2.numel() != readout_hidden_size:
            raise ValueError(
                "The nonlinear readout layers have inconsistent hidden dimensions: "
                f"{readout_hidden_size} and {readout_weights_2.numel()}"
            )
        output["readout_2_hidden_size"] = readout_hidden_size
        output["readout_2_weights_1"] = (
            torch.reshape(readout_weights_1, (num_channels, readout_hidden_size))
            .T.numpy(force=True)
            .flatten()
            / np.sqrt(num_channels)
        ).tolist()
        output["readout_2_weights_2"] = (
            readout_weights_2.numpy(force=True).flatten()
            / np.sqrt(readout_hidden_size)
            * model.scale_shift.scale.item()
        ).tolist()
        output["readout_2_scale_factor"] = readout.non_linearity.acts[0].cst

        if radial_format == "compact":
            r0_tensor_product = model.interactions[0].conv_tp
            r0_instructions = [
                serialize_instruction(instruction)
                for instruction in r0_tensor_product.instructions
            ]
            output["execution_contracts"] = {
                "M0": make_standard_m0_contract(
                    channels=num_channels,
                    type_count=len(atomic_numbers),
                    input_l_max=l_max,
                    output_l_max=L_max,
                    correlation=correlation,
                    monomials=M,
                ),
                "R0": make_standard_r0_contract(
                    channels=num_channels,
                    radial_embedding=output["compact_radial"]["networks"]["R0"][
                        "shape"
                    ][-2],
                    edge_l_max=l_max,
                    source_irreps=str(r0_tensor_product.irreps_in1),
                    edge_irreps=str(r0_tensor_product.irreps_in2),
                    output_irreps=str(r0_tensor_product.irreps_out),
                    output_l=[
                        r0_tensor_product.irreps_out[instruction.i_out].ir.l
                        for instruction in r0_tensor_product.instructions
                    ],
                    edge_l=[
                        r0_tensor_product.irreps_in2[instruction.i_in2].ir.l
                        for instruction in r0_tensor_product.instructions
                    ],
                    source_l=[
                        r0_tensor_product.irreps_in1[instruction.i_in1].ir.l
                        for instruction in r0_tensor_product.instructions
                    ],
                    instructions=r0_instructions,
                ),
            }
        return output

    ### ----- Phi1 -----

    logging.info("Phi1")
    Phi1_l = [ir[1].l for ir in model.interactions[1].conv_tp.irreps_out]
    Phi1_l1 = [ins.i_in2 for ins in model.interactions[1].conv_tp.instructions]
    Phi1_l2 = [ins.i_in1 for ins in model.interactions[1].conv_tp.instructions]
    Phi1_clebsch_gordan = []
    Phi1_lme = []
    Phi1_lelm1lm2 = []
    Phi1_sparse_lm1 = []
    Phi1_sparse_lm2 = []
    num_lm1 = (l_max + 1) ** 2
    num_lm2 = (L_max + 1) ** 2

    def compute_lem(le, l, m):
        lem = 0
        for j in range(le):
            lem += 2 * Phi1_l[j] + 1
        return lem + l + m

    def compute_lme(le, l, m):
        e = le - int(sum(np.array(Phi1_l) < l))
        num_e = [int(sum(np.array(Phi1_l) == ll)) for ll in range(l + 1)]
        lme = 0
        for ll in range(l):
            lme += (2 * ll + 1) * num_e[ll]
        return lme + (l + m) * num_e[l] + e

    def compute_lelm1lm2(le, l1, m1, l2, m2):
        lelm1lm2 = 0
        for j in range(le):
            l1, l2 = (Phi1_l1[j], Phi1_l2[j])
            lelm1lm2 += (2 * l1 + 1) * (2 * l2 + 1)
        l1, l2 = (Phi1_l1[le], Phi1_l2[le])
        return lelm1lm2 + (l1 + m1) * (2 * l2 + 1) + l2 + m2

    tp = model.interactions[1].conv_tp
    for l1 in range(l_max + 1):
        for m1 in range(-l1, l1 + 1):
            lm1 = l1 * l1 + l1 + m1
            for l2 in range(L_max + 1):
                for m2 in range(-l2, l2 + 1):
                    R = torch.ones(
                        [1, len(tp.instructions) * num_channels], dtype=torch.double
                    )
                    Y = torch.zeros([1, num_lm1], dtype=torch.double)
                    Y[0, lm1] = 1.0
                    H = torch.zeros([1, num_lm2 * num_channels], dtype=torch.double)
                    H[
                        0, sum([2 * p + 1 for p in range(l2)]) * num_channels + l2 + m2
                    ] = 1.0
                    Phi = tp(H, Y, R)
                    # extract Phi values for k=0
                    Phi_0 = []
                    for le in range(len(tp.instructions)):
                        Phi_0_start = (
                            sum([2 * Phi1_l[p] + 1 for p in range(le)]) * num_channels
                        )
                        for p in range(2 * Phi1_l[le] + 1):
                            Phi_0.append(Phi[0, Phi_0_start + p].item())
                    for le in range(len(tp.instructions)):
                        l = Phi1_l[le]
                        for m in range(-l, l + 1):
                            lem = compute_lem(le, l, m)
                            if np.abs(Phi_0[lem]) > 1e-12:
                                Phi1_lme.append(compute_lme(le, l, m))
                                Phi1_clebsch_gordan.append(Phi_0[lem])
                                Phi1_lelm1lm2.append(
                                    compute_lelm1lm2(le, l1, m1, l2, m2)
                                )
                                Phi1_sparse_lm1.append(lm1)
                                Phi1_sparse_lm2.append(l2 * l2 + l2 + m2)
    output["Phi1_l"] = Phi1_l
    output["Phi1_l1"] = Phi1_l1
    output["Phi1_l2"] = Phi1_l2
    output["Phi1_lme"] = Phi1_lme
    output["Phi1_clebsch_gordan"] = Phi1_clebsch_gordan
    output["Phi1_lelm1lm2"] = Phi1_lelm1lm2

    ### ----- A1 -----

    logging.info("A1")
    A1_scaled = True if ("Density" in type(model.interactions[1]).__name__) else False
    output["A1_scaled"] = A1_scaled
    if A1_scaled and radial_format == "pair-splines":
        r, h = np.linspace(1e-12, r_cut, num_spline_points, retstep=True)
        A1_spline_values = []
        A1_spline_derivs = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels = model.radial_embedding(
                    torch.tensor(r, dtype=model_dtype).unsqueeze(-1),
                    torch.eye(len(model.atomic_numbers), dtype=model_dtype),
                    torch.tensor([[model_i], [model_j]], dtype=torch.int64),
                    model.atomic_numbers,
                )
                if isinstance(bessels, tuple):
                    bessels = bessels[0]  # newer versions return (bessels, cutoffs)
                R = torch.tanh(model.interactions[1].density_fn(bessels) ** 2).numpy(
                    force=True
                )
                spl = CubicSpline(r, R[:, 0], bc_type=spline_bc_type)
                A1_spline_values.append(spl(r).tolist())
                A1_spline_derivs.append(spl.derivative()(r).tolist())
        output["A1_spline_h"] = float(h)
        output["A1_spline_min"] = float(r[0])
        output["A1_spline_values"] = A1_spline_values
        output["A1_spline_derivs"] = A1_spline_derivs
    A1_weights = []
    num_eta = [sum([l == ll for ll in Phi1_l]) for l in range(l_max + 1)]
    A1_linear = linear_simplify(model.interactions[1].linear)
    for l in range(l_max + 1):
        w_linear = A1_linear.weight_view_for_instruction(l).numpy(force=True) / np.sqrt(
            num_eta[l] * num_channels
        )
        if not A1_scaled:
            w_linear /= model.interactions[1].avg_num_neighbors
        w_linear = np.reshape(w_linear, (num_eta[l], num_channels, num_channels))
        A1_weights.append(w_linear.flatten().tolist())
    output["A1_weights"] = A1_weights

    if radial_format == "compact":
        r0_tensor_product = model.interactions[0].conv_tp
        r0_instructions = [
            serialize_instruction(instruction)
            for instruction in r0_tensor_product.instructions
        ]
        r1_instructions = [
            serialize_instruction(instruction)
            for instruction in model.interactions[1].conv_tp.instructions
        ]
        output["execution_contracts"] = {
            "M0": make_standard_m0_contract(
                channels=num_channels,
                type_count=len(atomic_numbers),
                input_l_max=l_max,
                output_l_max=L_max,
                correlation=correlation,
                monomials=M,
            ),
            "R0": make_standard_r0_contract(
                channels=num_channels,
                radial_embedding=output["compact_radial"]["networks"]["R0"]["shape"][
                    -2
                ],
                edge_l_max=l_max,
                source_irreps=str(r0_tensor_product.irreps_in1),
                edge_irreps=str(r0_tensor_product.irreps_in2),
                output_irreps=str(r0_tensor_product.irreps_out),
                output_l=[
                    r0_tensor_product.irreps_out[instruction.i_out].ir.l
                    for instruction in r0_tensor_product.instructions
                ],
                edge_l=[
                    r0_tensor_product.irreps_in2[instruction.i_in2].ir.l
                    for instruction in r0_tensor_product.instructions
                ],
                source_l=[
                    r0_tensor_product.irreps_in1[instruction.i_in1].ir.l
                    for instruction in r0_tensor_product.instructions
                ],
                instructions=r0_instructions,
            ),
            "R1": make_factorized_contract(
                channels=num_channels,
                radial_embedding=output["compact_radial"]["networks"]["R1"]["shape"][
                    -2
                ],
                edge_l_max=l_max,
                source_l_max=L_max,
                source_irreps=str(model.interactions[1].conv_tp.irreps_in1),
                edge_irreps=str(model.interactions[1].conv_tp.irreps_in2),
                output_irreps=str(model.interactions[1].conv_tp.irreps_out),
                output_l=Phi1_l,
                edge_l=Phi1_l1,
                source_l=Phi1_l2,
                instructions=r1_instructions,
                sparse_lme=Phi1_lme,
                sparse_coefficients=Phi1_clebsch_gordan,
                sparse_rows=Phi1_lelm1lm2,
                sparse_lm1=Phi1_sparse_lm1,
                sparse_lm2=Phi1_sparse_lm2,
            ),
        }

    ### ----- M1 -----

    logging.info("M1")
    # TODO: generalize
    correlation = model.products[1].symmetric_contractions.contractions[0].correlation
    irreps_in = Irreps("0e + 1o + 2e + 3o")
    irreps_out = Irreps("0e")
    # extract U in sparse format
    U = [[]]
    use_reduced_cg = bool(getattr(model, "use_reduced_cg", False))
    for corr in range(1, 4):
        # get U matrix for this correlation order
        try:
            U_matrix = U_matrix_real(
                irreps_in,
                irreps_out,
                corr,
                use_cueq_cg=use_reduced_cg,
            )[1]
        except TypeError:
            U_matrix = U_matrix_real(irreps_in, irreps_out, corr)[1]
        num_nu = U_matrix.shape[-1]
        U_matrix = U_matrix.flatten()
        # extract sparse U for this correlation order
        u_sparse_corr = [{} for _ in range(num_nu)]
        j = 0
        for lm_list in itertools.product(range((l_max + 1) ** 2), repeat=corr):
            for nu in range(num_nu):
                if abs(U_matrix[j]) > 1e-12:
                    lm_tuple_sorted = tuple(sorted(lm_list))
                    if lm_tuple_sorted not in u_sparse_corr[nu].keys():
                        u_sparse_corr[nu][lm_tuple_sorted] = 0.0
                    u_sparse_corr[nu][lm_tuple_sorted] += U_matrix[j].item()
                j += 1
        U.append(u_sparse_corr)
    # extract weights from model
    # warning: slightly odd order of the contractions weights due to reverse countdown
    W = [[]]
    W.append(
        model.products[1]
        .symmetric_contractions.contractions[0]
        .weights[1]
        .numpy(force=True)
    )
    W.append(
        model.products[1]
        .symmetric_contractions.contractions[0]
        .weights[0]
        .numpy(force=True)
    )
    W.append(
        model.products[1]
        .symmetric_contractions.contractions[0]
        .weights_max.numpy(force=True)
    )
    # combine U and W into polynomial-like terms for recursive evaluator
    C = {}
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        C[i] = {}
        for k in range(num_channels):
            P_ik = {}
            for corr in range(1, 4):
                for nu in range(len(U[corr])):
                    for key, value in U[corr][nu].items():
                        if key not in P_ik.keys():
                            P_ik[key] = 0.0
                        P_ik[key] += float(W[corr][model_i, nu, k]) * value
            C[i][k] = list(P_ik.values())
            M = [list(key) for key in P_ik.keys()]
    output["M1_weights"] = C
    output["M1_monomials"] = M

    ### ----- H2 -----

    logging.info("H2")
    weights_to_fuse = model.interactions[1].linear_up.weight_view_for_instruction(
        0
    ).numpy(force=True) / np.sqrt(num_channels)
    weights_to_fuse_rank = np.linalg.matrix_rank(weights_to_fuse)
    if weights_to_fuse_rank < num_channels:
        raise RuntimeError(
            "ERROR: fusing weights have too low rank {weights_to_fuse_rank} < {num_channels}"
        )
    # H2 weights for H1
    H2_weights_for_H1 = []
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        w = (
            model.interactions[1]
            .skip_tp.weight_view_for_instruction(0)[:, model_i, :]
            .numpy(force=True)
        )
        H2_weights_for_H1.append(w / np.sqrt(len(model.atomic_numbers) * num_channels))
        H2_weights_for_H1[i] = np.linalg.inv(weights_to_fuse) @ H2_weights_for_H1[i]
        H2_weights_for_H1[i] = H2_weights_for_H1[i].flatten().tolist()
    output["H2_weights_for_H1"] = H2_weights_for_H1
    # H2 weights for M1
    output["H2_weights_for_M1"] = (
        model.products[1].linear.weight.numpy(force=True) / np.sqrt(num_channels)
    ).tolist()

    ### ----- READOUTS -----

    # linear readout
    weights_to_fuse = np.reshape(
        model.interactions[1].linear_up.weight.numpy(force=True)
        / np.sqrt(num_channels),
        [L_max + 1, num_channels, num_channels],
    )
    readout_1_weights = model.readouts[0].linear.weight.numpy(force=True) / np.sqrt(
        num_channels
    )
    readout_1_weights = np.linalg.inv(weights_to_fuse[0, :, :]) @ readout_1_weights
    output["readout_1_weights"] = (
        readout_1_weights * model.scale_shift.scale.item()
    ).tolist()

    # nonlinear readout
    readout_2_weights_1 = model.readouts[1].linear_1.weight
    readout_2_weights_2 = model.readouts[1].linear_2.weight
    if readout_2_weights_1.numel() % num_channels != 0:
        raise ValueError(
            "The nonlinear readout input weights are incompatible with the "
            f"{num_channels}-channel model"
        )
    readout_2_hidden_size = readout_2_weights_1.numel() // num_channels
    if readout_2_weights_2.numel() != readout_2_hidden_size:
        raise ValueError(
            "The nonlinear readout layers have inconsistent hidden dimensions: "
            f"{readout_2_hidden_size} and {readout_2_weights_2.numel()}"
        )
    output["readout_2_hidden_size"] = readout_2_hidden_size
    output["readout_2_weights_1"] = (
        torch.reshape(readout_2_weights_1, (num_channels, readout_2_hidden_size))
        .T.numpy(force=True)
        .flatten()
        / np.sqrt(num_channels)
    ).tolist()
    output["readout_2_weights_2"] = (
        readout_2_weights_2.numpy(force=True).flatten()
        / np.sqrt(readout_2_hidden_size)
        * model.scale_shift.scale.item()
    ).tolist()
    output["readout_2_scale_factor"] = model.readouts[1].non_linearity.acts[0].cst

    return output


def _prediction_head_payload(data):
    if data.get("model_type") == "MACE_Nonlinear":
        fields = ("atomic_energies", "scale_shift", "readouts")
    else:
        fields = (
            "atomic_energies",
            "readout_1_weights",
            "readout_2_weights_1",
            "readout_2_weights_2",
        )
        if data.get("has_zbl", False):
            fields += ("zbl_c",)
    return {name: data[name] for name in fields}


def _extract_mh0_head_payload(model, atomic_numbers):
    num_channels = model.node_embedding.linear.irreps_out.count("0e")
    L_max = model.products[0].linear.irreps_out.lmax
    atomic_energies = [
        torch.atleast_1d(model.atomic_energies_fn.atomic_energies.squeeze())[
            model.atomic_numbers.tolist().index(number)
        ].item()
        + model.scale_shift.shift.item()
        for number in atomic_numbers
    ]
    weights_to_fuse = np.reshape(
        model.interactions[1].linear_up.weight.numpy(force=True)
        / np.sqrt(num_channels),
        [L_max + 1, num_channels, num_channels],
    )
    readout_1 = model.readouts[0].linear.weight.numpy(force=True) / np.sqrt(
        num_channels
    )
    readout_1 = np.linalg.inv(weights_to_fuse[0]) @ readout_1
    readout_2_weights_1 = model.readouts[1].linear_1.weight
    hidden_size = readout_2_weights_1.numel() // num_channels
    payload = {
        "atomic_energies": atomic_energies,
        "readout_1_weights": (readout_1 * model.scale_shift.scale.item()).tolist(),
        "readout_2_weights_1": (
            torch.reshape(readout_2_weights_1, (num_channels, hidden_size))
            .T.numpy(force=True)
            .flatten()
            / np.sqrt(num_channels)
        ).tolist(),
        "readout_2_weights_2": (
            model.readouts[1].linear_2.weight.numpy(force=True).flatten()
            / np.sqrt(hidden_size)
            * model.scale_shift.scale.item()
        ).tolist(),
    }
    if hasattr(model, "pair_repulsion") and model.pair_repulsion:
        payload["zbl_c"] = (
            model.scale_shift.scale.item() * model.pair_repulsion_fn.c.numpy(force=True)
        ).tolist()
    return payload


def _prediction_head_payload_layout(payload, nonlinear):
    if not nonlinear:
        return {name: len(values) for name, values in payload.items()}

    def layout(value):
        if isinstance(value, dict):
            if set(value) == {"shape", "values"}:
                return {"shape": value["shape"]}
            return {name: layout(item) for name, item in value.items()}
        if isinstance(value, list):
            return [layout(item) for item in value]
        return value

    return layout(payload)


def _default_prediction_head(heads, requested):
    if requested is not None:
        if requested not in heads:
            raise ValueError(
                f"Head {requested!r} not found; available heads are {heads!r}"
            )
        return requested
    return next((name for name in heads if name != "pt_head"), heads[0])


def extract_mace_data(
    model,
    species=None,
    head=None,
    num_spline_points=256,
    radial_format="compact",
):
    """Extract a checkpoint while retaining every compatible prediction head.

    A single-head checkpoint keeps the legacy flat JSON layout. For a
    multi-head checkpoint, ``head`` chooses the declared default; all heads are
    stored in ``prediction_heads`` and the default payload is also retained at
    top level for compatibility with older Symmetrix loaders.
    """

    device = torch.device("cpu")
    loaded = _load_mace_checkpoint(model, device).to(device=device, dtype=torch.float64)
    loaded.eval()
    heads = list(getattr(loaded, "heads", []))
    if len(heads) <= 1:
        return _extract_single_head_data(
            loaded,
            species=species,
            head=head,
            num_spline_points=num_spline_points,
            radial_format=radial_format,
        )

    default_head = _default_prediction_head(heads, head)
    parameters = {}
    default_model = _remove_prediction_head(loaded, default_head, device)
    default_model.eval()
    default_data = _extract_single_head_data(
        default_model,
        species=species,
        head=default_head,
        num_spline_points=num_spline_points,
        radial_format=radial_format,
    )
    nonlinear = default_data.get("model_type") == "MACE_Nonlinear"
    default_payload = _prediction_head_payload(default_data)
    expected_layout = _prediction_head_payload_layout(default_payload, nonlinear)
    parameters[default_head] = default_payload

    atomic_numbers = default_data["atomic_numbers"]
    for name in heads:
        if name == default_head:
            continue
        single_model = _remove_prediction_head(loaded, name, device)
        single_model.eval()
        if nonlinear:
            from .extract_mace_nonlinear import (
                extract_mace_nonlinear_head_payload,
            )

            payload = extract_mace_nonlinear_head_payload(single_model)
        else:
            payload = _extract_mh0_head_payload(single_model, atomic_numbers)
        if _prediction_head_payload_layout(payload, nonlinear) != expected_layout:
            raise ValueError(
                f"Prediction head {name!r} changes the shared model topology; "
                "only parameter-compatible heads can be stored together"
            )
        parameters[name] = payload
    default_data["prediction_heads"] = {
        "schema_version": PREDICTION_HEADS_SCHEMA_VERSION,
        "names": heads,
        "default": default_head,
        "parameters": parameters,
    }
    default_data["head"] = default_head
    return default_data
