"""Normalize the packaged NequIP OAM graph into a package-independent view.

The adapter intentionally uses module names and public attributes rather than
NequIP class identity.  Deployed ``.nequip.zip`` models can contain classes
loaded from an archive-specific module namespace, while their graph structure
and tensor operands remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Iterable, Mapping


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            candidate = value[name]
        else:
            candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _class_name(value: Any) -> str:
    cls = value if inspect.isclass(value) else type(value)
    return cls.__qualname__.split(".")[-1]


def _contract_children(
    name: str, child: Any, *, trusted: bool
) -> Iterable[tuple[str, Any]]:
    """Expand one child entry, flattening list- and mapping-valued containers.

    ``torch`` registers ``ModuleList``/``ModuleDict`` values through
    ``named_children`` as a *single* entry whose value is a list or a mapping;
    the plain-attribute fallback sees the same containers.  Both branches must
    flatten them identically, otherwise OAM layer discovery depends on which
    traversal branch runs.
    """

    def accepted(item: Any) -> bool:
        return (
            trusted or _looks_like_module(item) or _class_name(item) == "ConvNetLayer"
        )

    if isinstance(child, (list, tuple)):
        for index, item in enumerate(child):
            if accepted(item):
                yield f"{name}.{index}", item
        return
    if isinstance(child, Mapping):
        for key, item in child.items():
            if isinstance(key, str) and accepted(item):
                yield f"{name}.{key}", item
        return
    if accepted(child):
        yield name, child


def _children(value: Any) -> Iterable[tuple[str, Any]]:
    named = getattr(value, "named_children", None)
    if callable(named):
        try:
            declared = list(named())
        except (AttributeError, TypeError):
            declared = None
        if declared is not None:
            for name, child in declared:
                yield from _contract_children(str(name), child, trusted=True)
            return
    children = getattr(value, "_modules", None)
    if isinstance(children, Mapping):
        for name, child in children.items():
            yield from _contract_children(str(name), child, trusted=True)
        return
    for name, child in vars(value).items() if hasattr(value, "__dict__") else ():
        if name.startswith("_") or name in {"metadata", "training"}:
            continue
        yield from _contract_children(name, child, trusted=False)


def _looks_like_module(value: Any) -> bool:
    return (
        value is not None
        and not inspect.isclass(value)
        and (
            callable(getattr(value, "named_children", None))
            or callable(getattr(value, "children", None))
            or hasattr(value, "is_convnet_layer")
            or "Layer" in _class_name(value)
            or "Block" in _class_name(value)
        )
    )


def _walk(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    yield path, value
    for name, child in _children(value):
        child_path = f"{path}.{name}" if path else str(name)
        yield from _walk(child, child_path)


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        tail = _tensor_shape(value[0]) or ()
        return (len(value), *tail)
    return None


def _to_list(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except (TypeError, ValueError):
            pass
    return value


def _irreps(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _field_irreps(value: Any, *fields: str) -> str | None:
    """Read an irreps mapping without depending on AtomicDataDict constants."""
    for field in fields:
        candidate = _get(value, field, default=None)
        if candidate is not None:
            return _irreps(candidate)
    if isinstance(value, Mapping):
        for field in fields:
            if field in value:
                return _irreps(value[field])
    return _irreps(value)


@dataclass(frozen=True)
class OAMNodeEmbedding:
    type_names: tuple[str, ...]
    irreps_out: str | None
    weight: Any
    weight_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class OAMEdgeEmbedding:
    num_basis: int | None
    bessel_weights: Any
    cutoff_p: float | None
    factor: float | None
    r_cut: float | None


@dataclass(frozen=True)
class OAMLayer:
    name: str
    path: str
    interaction_path: str
    node_irreps_in: str | None
    node_irreps_pre_gate: str | None
    node_irreps_out: str | None
    edge_irreps: str | None
    operation_order: tuple[str, ...]
    interaction: Any
    self_connection: Any
    gate: Any
    tp_weight_numel: int | None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class NequIPOAMGraph:
    layers: tuple[OAMLayer, ...]
    node_embedding: OAMNodeEmbedding
    edge_embedding: OAMEdgeEmbedding
    type_names: tuple[str, ...]
    r_cut: float | None
    readout: Any
    has_stress: bool
    output_keys: tuple[str, ...]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layers": [layer.__dict__.copy() for layer in self.layers],
            "node_embedding": self.node_embedding.__dict__.copy(),
            "edge_embedding": self.edge_embedding.__dict__.copy(),
            "type_names": list(self.type_names),
            "r_cut": self.r_cut,
            "readout": self.readout,
            "has_stress": self.has_stress,
            "output_keys": list(self.output_keys),
        }


def _outer_layers(model: Any) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []

    def visit(value: Any, path: str, inside_outer: bool = False) -> None:
        name = _class_name(value)
        is_outer = name == "ConvNetLayer" or bool(
            _get(value, "is_convnet_layer", default=False)
        )
        if is_outer:
            found.append((path, value))
            return
        for child_name, child in _children(value):
            child_path = f"{path}.{child_name}" if path else str(child_name)
            visit(child, child_path, inside_outer)

    visit(model, "")
    return found


def _unwrap_graph_model(model: Any) -> Any:
    """Unwrap GraphModel/ForceStressOutput containers around the energy graph."""
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        name = _class_name(current).lower()
        child = (
            _get(current, "func", default=None) if "forcestressoutput" in name else None
        )
        if child is None:
            child = (
                _get(current, "model", default=None)
                if "graphmodel" in name
                or _get(current, "is_graph_model", default=False)
                else None
            )
        if child is None:
            return current if _outer_layers(current) else current
        current = child
    return current


def _find_interaction(outer: Any, outer_path: str) -> tuple[str, Any]:
    candidates = []
    for path, value in _walk(outer, outer_path):
        name = _class_name(value).lower()
        if name == "interactionblock" or bool(
            _get(value, "is_interaction", default=False)
        ):
            candidates.append((path, value))
    if not candidates:
        raise ValueError(f"outer ConvNetLayer {outer_path!r} has no InteractionBlock")
    return candidates[0]


def _node_embedding(model: Any, fallback: Any = None) -> OAMNodeEmbedding:
    module = _get(
        model,
        "type_embed",
        "node_embedding",
        "initial_embedding",
        "embedding",
        default=None,
    )
    module = module or object()
    names = _get(
        module,
        "type_names",
        "chemical_symbols",
        default=_get(
            model, "type_names", default=_get(fallback, "type_names", default=())
        ),
    )
    if isinstance(names, Mapping):
        names = tuple(
            str(k) for k, _ in sorted(names.items(), key=lambda item: item[1])
        )
    else:
        names = tuple(str(item) for item in (names or ()))
    embedding = _get(module, "embed_module", "embedding", default=module)
    weight = _get(embedding, "weight", default=None)
    return OAMNodeEmbedding(
        names,
        _field_irreps(
            _get(module, "irreps_out", default=None),
            "node_features",
            "node_attrs",
            "features",
        ),
        _to_list(weight),
        _tensor_shape(weight),
    )


def _edge_embedding(model: Any, fallback: Any = None) -> OAMEdgeEmbedding:
    normalizer = _get(model, "edge_norm", "edge_length_normalizer", default=None)
    bessel = _get(
        model, "bessel_encode", "bessel_encoding", "edge_embedding", default=None
    )
    factor = _get(model, "factor", "edge_factor", default=None)
    cutoff = _get(bessel, "cutoff", default=None)
    r_cut = _get(
        normalizer,
        "r_max",
        "r_cut",
        "cutoff",
        default=_get(
            model,
            "r_cut",
            "r_max",
            default=_get(fallback, "r_cut", "r_max", default=None),
        ),
    )
    try:
        r_cut = float(r_cut) if r_cut is not None else None
    except (TypeError, ValueError):
        r_cut = None
    return OAMEdgeEmbedding(
        _get(bessel, "num_bessels", "num_basis", "number_of_basis", default=None),
        _to_list(_get(bessel, "bessel_weights", "frequencies", "weight", default=None)),
        _get(
            cutoff,
            "p",
            "polynomial_cutoff_p",
            default=_get(bessel, "polynomial_cutoff_p", default=None),
        ),
        _get(factor, "factor", default=None),
        r_cut,
    )


def normalize_nequip_oam_model(model: Any) -> NequIPOAMGraph:
    """Return a stable graph view for an instantiated packaged OAM model."""
    if model is None:
        raise ValueError("NequIP OAM model is None")
    graph_model = _unwrap_graph_model(model)
    outer = _outer_layers(graph_model)
    if not outer:
        raise ValueError("no outer ConvNetLayer modules found in NequIP OAM model")
    layers: list[OAMLayer] = []
    for path, module in outer:
        interaction_path, interaction = _find_interaction(module, path)
        order = [
            "linear_1",
            "avg_num_neighbors_norm",
            "edge_mlp",
            "tp_scatter",
            "linear_2",
        ]
        if _get(interaction, "sc", "self_connection", default=None) is not None:
            order.append("self_connection")
        order.append("gate")
        layers.append(
            OAMLayer(
                _class_name(module) if path == "" else path.rsplit(".", 1)[-1],
                path,
                interaction_path,
                _field_irreps(
                    _get(interaction, "irreps_in", default=None),
                    "node_features",
                    "node_attrs",
                    "features",
                )
                or _irreps(_get(interaction, "node_irreps_in", default=None)),
                _field_irreps(
                    _get(interaction, "irreps_out", default=None),
                    "node_features",
                    "features",
                )
                or _irreps(_get(interaction, "node_irreps_out", default=None)),
                _field_irreps(
                    _get(
                        _get(
                            module,
                            "equivariant_nonlin",
                            "gate",
                            "nonlinearity",
                            default=None,
                        ),
                        "irreps_out",
                        default=None,
                    ),
                    "node_features",
                    "features",
                )
                or _field_irreps(
                    _get(module, "irreps_out", default=None),
                    "node_features",
                    "features",
                )
                or _field_irreps(
                    _get(interaction, "irreps_out", default=None),
                    "node_features",
                    "features",
                )
                or _irreps(_get(interaction, "node_irreps_out", default=None)),
                _irreps(
                    _get(
                        interaction,
                        "edge_irreps",
                        default=_get(
                            _get(
                                _get(interaction, "tp_scatter", default=None),
                                "tp",
                                default=None,
                            ),
                            "irreps_in2",
                            default=None,
                        ),
                    )
                ),
                tuple(order),
                interaction,
                _get(interaction, "sc", "self_connection", default=None),
                _get(
                    module,
                    "equivariant_nonlin",
                    "gate",
                    "nonlinearity",
                    default=_get(interaction, "gate", default=None),
                ),
                _get(
                    _get(
                        _get(interaction, "tp_scatter", default=None),
                        "tp",
                        default=None,
                    ),
                    "weight_numel",
                    default=None,
                ),
            )
        )
    output_keys = _get(model, "output_keys", "outputs", default=())
    has_stress = any(
        str(key).lower() in {"stress", "virial", "stress_tensor"} for key in output_keys
    )
    if not has_stress:
        has_stress = any(
            "forcestressoutput" in _class_name(value).lower()
            or bool(_get(value, "do_derivatives", default=False))
            for _, value in _walk(model)
        )
    return NequIPOAMGraph(
        tuple(layers),
        _node_embedding(graph_model, model),
        _edge_embedding(graph_model, model),
        _node_embedding(graph_model, model).type_names,
        _edge_embedding(graph_model, model).r_cut,
        _get(
            graph_model,
            "per_atom_energy_readout",
            "readout",
            "output_readout",
            default=None,
        ),
        has_stress,
        tuple(str(key) for key in output_keys),
    )


def _oam_metadata_marker(metadata: Any) -> bool:
    if isinstance(metadata, Mapping):
        # nequip-oam records its package and architecture identity in the
        # package metadata; every value is scanned because the exact key changes
        # between nequip-oam releases.
        return any("oam" in str(value).lower() for value in metadata.values())
    return "oam" in str(metadata).lower()


def _oam_identity_hints(model: Any) -> list[str]:
    """Collect the *explicit* OAM identity hints declared by a model."""

    hints = [type(model).__name__, type(model).__module__]
    metadata = _get(model, "metadata", "config", "model_config", default=None)
    if metadata is not None:
        if isinstance(metadata, Mapping):
            hints.extend(str(value) for value in metadata.values())
        else:
            hints.append(str(metadata))
    architecture = _get(model, "architecture", "model_arch", default=None)
    if architecture is not None:
        hints.append(str(architecture))
    return hints


def looks_like_nequip_oam_model(model: Any) -> bool:
    """Report whether a model declares the NequIP-OAM architecture explicitly.

    Only *explicit* identity counts: an ``oam`` marker in the model's own class,
    module, architecture, or package metadata.  ``ConvNetLayer`` and
    ``InteractionBlock`` are shared with ordinary NequIP models (v1 NequIP uses
    them as its interaction family), so they must not be treated as OAM
    markers.  Callers that route on this predicate treat ``True`` as a
    declaration that must be honoured; a model that merely *might* be OAM is
    reported by :func:`has_nequip_oam_hint` instead and may fall back to the
    legacy NequIP exporter.
    """

    if model is None:
        return False
    if any("oam" in hint.lower() for hint in _oam_identity_hints(model)):
        return True
    return _oam_metadata_marker(_get(model, "metadata", "config", default=None))


def has_nequip_oam_hint(model: Any) -> bool:
    """Report whether any reachable module carries an ``oam`` name hint."""

    if model is None:
        return False
    for _, value in _walk(model):
        if "oam" in _class_name(value).lower():
            return True
        if "oam" in str(getattr(type(value), "__module__", "")).lower():
            return True
    return False


__all__ = [
    "NequIPOAMGraph",
    "OAMLayer",
    "OAMEdgeEmbedding",
    "OAMNodeEmbedding",
    "has_nequip_oam_hint",
    "looks_like_nequip_oam_model",
    "normalize_nequip_oam_model",
]
