"""Build the compact, production-shaped 512/128 polar backbone fixture."""

from __future__ import annotations

from functools import lru_cache
import sys

from polar_oracle_checkout import optional_checkout

RELEASED_M_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)

_ORACLE_MACE_CHECKOUT = optional_checkout("SYMMETRIX_POLAR_ORACLE_MACE")
if (
    _ORACLE_MACE_CHECKOUT is not None
    and _ORACLE_MACE_CHECKOUT.is_dir()
    and str(_ORACLE_MACE_CHECKOUT) not in sys.path
):
    sys.path.insert(0, str(_ORACLE_MACE_CHECKOUT))


def _build_scalar_final_edge_model(
    num_interactions, *, edge_channels=128, use_edge_irreps_first=False
):
    """Build a contract-coherent MH-1 model with selected layer edge widths."""

    import mace  # noqa: F401  (sets the trusted e3nn weights-only policy)
    import numpy
    import torch
    from e3nn import o3
    from mace.modules.blocks import (
        NonLinearBiasReadoutBlock,
        RealAgnosticResidualNonLinearInteractionBlock,
    )
    from mace.modules.models import ScaleShiftMACE
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(20261018 + num_interactions)
        model = ScaleShiftMACE(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=RealAgnosticResidualNonLinearInteractionBlock,
            interaction_cls_first=RealAgnosticResidualNonLinearInteractionBlock,
            num_interactions=num_interactions,
            num_elements=1,
            hidden_irreps=o3.Irreps("512x0e+512x1o"),
            MLP_irreps=o3.Irreps("2x0e"),
            atomic_energies=numpy.zeros((1, 1)),
            avg_num_neighbors=30.0,
            atomic_numbers=[1],
            correlation=3,
            gate=torch.nn.functional.silu,
            use_agnostic_product=True,
            edge_irreps=o3.Irreps(f"{edge_channels}x0e+{edge_channels}x1o"),
            use_edge_irreps_first=use_edge_irreps_first,
            radial_MLP=[4],
            heads=["test"],
            readout_cls=NonLinearBiasReadoutBlock,
            keep_last_layer_irreps=False,
        )
        return extract_mace_nonlinear_data(model)
    finally:
        torch.set_default_dtype(previous_dtype)


@lru_cache(maxsize=1)
def build_scalar_final_mixed_edge_mh1_definition():
    return _build_scalar_final_edge_model(2)


@lru_cache(maxsize=1)
def build_scalar_final_uniform_512_edge_mh1_definition():
    return _build_scalar_final_edge_model(2, edge_channels=512)


@lru_cache(maxsize=1)
def build_scalar_final_uniform_128_edge_mh1_definition():
    return _build_scalar_final_edge_model(
        2, edge_channels=128, use_edge_irreps_first=True
    )


@lru_cache(maxsize=1)
def build_one_layer_scalar_final_mh1_definition():
    return _build_scalar_final_edge_model(1)


@lru_cache(maxsize=1)
def build_polar_mh1_backbone_definition():
    """Return a real extracted model definition with 512/128 UVU layers.

    The model uses the same MACE and polar module classes as the released
    checkpoint.  It is built in memory so the repository stores a compact
    recipe rather than hundreds of megabytes of serialized parameter values.
    The extractor supplies the production nesting, complete tensor metadata,
    parameters, and both execution contracts.
    """

    import mace  # noqa: F401  (sets the trusted e3nn weights-only policy)
    import numpy
    import torch
    from e3nn import o3
    from mace.modules.blocks import (
        NonLinearBiasReadoutBlock,
        RealAgnosticResidualNonLinearInteractionBlock,
    )
    from mace.modules.field_blocks import (
        AgnosticEmbeddedOneBodyVariableUpdate,
        EnvironmentDependentSpinSourceBlock,
        MultiLayerFeatureMixer,
        OneBodyMLPFieldReadout,
    )
    from mace.modules.models import ScaleShiftMACE
    from symmetrix.extract_mace_polar import extract_mace_polar_data

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(20261017)
        hidden = o3.Irreps("512x0e+512x1o")
        model = ScaleShiftMACE(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=RealAgnosticResidualNonLinearInteractionBlock,
            interaction_cls_first=RealAgnosticResidualNonLinearInteractionBlock,
            num_interactions=2,
            num_elements=1,
            hidden_irreps=hidden,
            MLP_irreps=o3.Irreps("2x0e"),
            atomic_energies=numpy.zeros((1, 1)),
            avg_num_neighbors=30.0,
            atomic_numbers=[1],
            correlation=3,
            gate=torch.nn.functional.silu,
            use_agnostic_product=True,
            edge_irreps=o3.Irreps("128x0e+128x1o"),
            use_edge_irreps_first=False,
            radial_MLP=[4],
            heads=["test"],
            readout_cls=NonLinearBiasReadoutBlock,
            keep_last_layer_irreps=True,
        )

        charges_irreps = 2 * o3.Irreps.spherical_harmonics(1)
        field_irreps = o3.Irreps("1x0e")
        update_sh_irreps = o3.Irreps.spherical_harmonics(2)
        field_interaction_irreps = (
            (update_sh_irreps * hidden.count(o3.Irrep(0, 1))).sort()[0].simplify()
        )
        model.lr_source_maps = torch.nn.ModuleList(
            EnvironmentDependentSpinSourceBlock(hidden, max_l=1) for _ in range(2)
        )
        model.layer_feature_mixer = MultiLayerFeatureMixer(hidden, 2)
        model.field_dependent_charges_maps = torch.nn.ModuleList(
            [
                AgnosticEmbeddedOneBodyVariableUpdate(
                    node_attrs_irreps=o3.Irreps("1x0e"),
                    node_feats_irreps=hidden,
                    edge_attrs_irreps=update_sh_irreps,
                    edge_feats_irreps=o3.Irreps("8x0e"),
                    target_irreps=field_interaction_irreps,
                    hidden_irreps=hidden,
                    avg_num_neighbors=30.0,
                    potential_irreps=field_irreps * 2,
                    charges_irreps=charges_irreps,
                    field_norm_factor=1.0,
                )
            ]
        )
        model.fukui_source_map = NonLinearBiasReadoutBlock(
            hidden,
            o3.Irreps("2x0e"),
            torch.nn.functional.silu,
            o3.Irreps("2x0e"),
        )
        model.local_electron_energy = OneBodyMLPFieldReadout(
            node_attrs_irreps=o3.Irreps("1x0e"),
            node_feats_irreps=hidden,
            edge_attrs_irreps=update_sh_irreps,
            edge_feats_irreps=o3.Irreps("8x0e"),
            target_irreps=field_interaction_irreps,
            hidden_irreps=hidden,
            avg_num_neighbors=30.0,
            potential_irreps=field_irreps * 2,
            charges_irreps=charges_irreps,
        )
        model.add_local_electron_energy = True
        model.include_electrostatic_self_interaction = False
        model.quadrupole_feature_corrections = False
        model.field_si = False
        model.num_recursion_steps = 1
        model.keep_last_layer_irreps = True
        model.atomic_multipoles_max_l = 1
        model.atomic_multipoles_smearing_width = 1.5
        model.field_feature_max_l = 0
        model.field_feature_widths = [1.5]
        model.field_feature_norms = torch.ones(1, dtype=torch.float64)
        model.field_norm_factor = 1.0
        model.kspace_cutoff = torch.tensor(4.2, dtype=torch.float64)
        model.charges_irreps = charges_irreps
        model.field_irreps = field_irreps
        model.potential_irreps = field_irreps * 2
        return extract_mace_polar_data(model)
    finally:
        torch.set_default_dtype(previous_dtype)
