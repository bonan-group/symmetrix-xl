from __future__ import annotations

from types import SimpleNamespace

import pytest

from symmetrix.nequip_oam_graph import normalize_nequip_oam_model


class _ChildContainer:
    def __init__(self, **children):
        self._children = children
        for name, value in children.items():
            setattr(self, name, value)

    def named_children(self):
        return tuple(self._children.items())


class ConvNetLayer(_ChildContainer):
    def __init__(self, interaction):
        super().__init__(interaction=interaction)
        self.is_convnet_layer = True


class InteractionBlock:
    is_interaction = True

    def __init__(self, first=False):
        self.is_first_layer = first
        self.irreps_in = "128x0e+64x1e"
        self.irreps_out = "128x0e+64x1e" if first else "128x0e"
        self.linear_1 = SimpleNamespace(
            irreps_in=self.irreps_in, irreps_out=self.irreps_in
        )
        self.avg_num_neighbors_norm = SimpleNamespace(norm_const=[0.5])
        self.edge_mlp = SimpleNamespace(
            input_dim=8,
            output_dim=32,
            layers=[SimpleNamespace(weight=[[1.0] * 8] * 3, bias=[0.0] * 3)],
        )
        self.tp_scatter = SimpleNamespace(
            tp=SimpleNamespace(
                irreps_in1=self.irreps_in,
                irreps_in2="1x0e+1x1e",
                irreps_out=self.irreps_out,
                instructions=[(0, 0, 0, "uvu", True)],
                weight_numel=32,
            )
        )
        self.linear_2 = SimpleNamespace(
            irreps_in=self.irreps_out, irreps_out=self.irreps_out
        )
        self.sc = (
            SimpleNamespace(
                irreps_in1=self.irreps_in,
                irreps_in2="89x0e",
                irreps_out=self.irreps_out,
            )
            if not first
            else None
        )


class NodeTypeEmbed:
    def __init__(self):
        self.type_names = ["H", "Pu"]
        self.num_types = 2
        self.irreps_out = "32x0e"
        self.embed_module = SimpleNamespace(weight=[[1.0] * 32, [2.0] * 32])


class EdgeLengthNormalizer:
    r_max = 4.5


class BesselEdgeLengthEncoding:
    num_bessels = 8
    bessel_weights = [1.0] * 8
    cutoff = SimpleNamespace(p=5)


class ApplyFactor:
    factor = 1.234


class ForceStressOutput:
    do_derivatives = True


class OAMModel(_ChildContainer):
    def __init__(self):
        self.metadata = {"nequip_version": "0.19.1"}
        self.output_keys = ()
        self._children = {
            "type_embed": NodeTypeEmbed(),
            "edge_norm": EdgeLengthNormalizer(),
            "bessel_encode": BesselEdgeLengthEncoding(),
            "factor": ApplyFactor(),
            "layer0_convnet": ConvNetLayer(InteractionBlock(first=True)),
            "layer1_convnet": ConvNetLayer(InteractionBlock()),
            "per_atom_energy_readout": SimpleNamespace(
                irreps_in="128x0e", output_dim=1
            ),
            "force_stress": ForceStressOutput(),
        }
        for name, value in self._children.items():
            setattr(self, name, value)

    def named_children(self):
        return tuple(self._children.items())


class GraphModel:
    is_graph_model = True

    def __init__(self, model):
        self.type_names = ["H", "Pu"]
        self.r_cut = 4.5
        self.model = model


class WrappedForceStressOutput:
    do_derivatives = True

    def __init__(self, func):
        self.func = func


def test_normalize_selects_outer_layers_and_preserves_oam_order():
    graph = normalize_nequip_oam_model(OAMModel())

    assert [layer.name for layer in graph.layers] == [
        "layer0_convnet",
        "layer1_convnet",
    ]
    assert [layer.interaction_path for layer in graph.layers] == [
        "layer0_convnet.interaction",
        "layer1_convnet.interaction",
    ]
    assert graph.layers[0].operation_order == (
        "linear_1",
        "avg_num_neighbors_norm",
        "edge_mlp",
        "tp_scatter",
        "linear_2",
        "gate",
    )
    assert graph.layers[1].self_connection is not None
    assert graph.layers[0].self_connection is None


def test_normalize_captures_edge_embedding_and_type_embedding():
    graph = normalize_nequip_oam_model(OAMModel())

    assert graph.r_cut == pytest.approx(4.5)
    assert graph.type_names == ("H", "Pu")
    assert graph.edge_embedding.num_basis == 8
    assert graph.edge_embedding.bessel_weights == [1.0] * 8
    assert graph.edge_embedding.cutoff_p == 5
    assert graph.edge_embedding.factor == pytest.approx(1.234)
    assert graph.node_embedding.irreps_out == "32x0e"
    assert graph.node_embedding.weight_shape == (2, 32)


def test_normalize_detects_wrapped_stress_output_without_root_keys():
    graph = normalize_nequip_oam_model(OAMModel())
    assert graph.has_stress is True


def test_normalize_unwraps_graph_and_force_stress_wrappers():
    core = OAMModel()
    core.type_embed.embed_module.weight = [[0.0] * 32 for _ in range(89)]
    core.layer0_convnet.equivariant_nonlin = SimpleNamespace(
        irreps_out={"node_features": "128x0e"}
    )
    core.layer0_convnet.interaction.irreps_out = {"node_features": "192x0e+64x1e"}
    core._children["layer0_convnet"].interaction.irreps_in = {
        "node_features": "128x0e+64x1e",
        "edge_attrs": "1x0e+1x1e",
    }
    wrapped = GraphModel(WrappedForceStressOutput(core))
    wrapped.type_names = [f"T{index}" for index in range(89)]
    graph = normalize_nequip_oam_model(wrapped)

    assert graph.r_cut == pytest.approx(4.5)
    assert graph.node_embedding.weight_shape == (89, 32)
    assert graph.node_embedding.irreps_out == "32x0e"
    assert graph.layers[0].node_irreps_in == "128x0e+64x1e"
    assert graph.layers[0].node_irreps_pre_gate == "192x0e+64x1e"
    assert graph.layers[0].node_irreps_out == "128x0e"
    assert graph.layers[0].gate is core.layer0_convnet.equivariant_nonlin
    assert graph.layers[0].edge_irreps == "1x0e+1x1e"


def test_normalize_rejects_missing_outer_convnet_layers():
    model = OAMModel()
    del model.layer0_convnet
    del model.layer1_convnet
    model._children.pop("layer0_convnet")
    model._children.pop("layer1_convnet")
    with pytest.raises(ValueError, match="outer ConvNetLayer"):
        normalize_nequip_oam_model(model)
