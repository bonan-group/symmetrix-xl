"""Compatibility probing for optional NequIP models.

The probe deliberately does not import torch, e3nn, or NequIP at module import
time.  This keeps the normal Symmetrix runtime independent from those optional
packages while allowing the exporter to inspect an already-created module.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import re
from typing import Any, Iterable


class NequIPProbeError(ValueError):
    """Raised when an input cannot be loaded or is outside probe support."""


class NequIPDependencyError(NequIPProbeError, ImportError):
    """Raised when probing a path needs an optional package that is absent."""


_INTERACTION_NAMES = {
    "ConvNetLayer",
    "Convolution",
    "EdgeConvolution",
    "InteractionBlock",
    "NequIPInteractionBlock",
    "SimpleInteractionBlock",
}
_SUPPORTED_FAMILIES = {
    "interaction": "ConvNetLayer/InteractionBlock",
    "gate": "Gate/ScalarMLPFunction",
    "normalization": "Identity/PerTypeScaleShift",
}
_FAMILY_MARKERS = {
    "interaction": ("is_interaction", "is_convolution"),
    "gate": ("is_gate", "is_nonlinearity"),
    "normalization": ("is_normalization", "is_normalizer"),
    "long-range": ("is_long_range", "is_longrange"),
    "plugin": ("is_plugin", "is_custom_op"),
}
_SKIP_ATTRIBUTES = {
    "training",
    "_parameters",
    "_buffers",
    "_modules",
    "metadata",
}
_SUPPORTED_NORMALIZATION_NAMES = {
    "Identity",
    "PerTypeScaleShift",
    "TinyPerTypeScaleShift",
}
_SUPPORTED_GATE_NAMES = {
    "Gate",
    "TinyGate",
    "ScalarMLPFunction",
    "Silu",
    "SiLU",
    "ShiftedSoftPlus",
}
# Renormalizing feature layers outside the supported family are rejected by
# module identity: a class name ending in ``Norm``/``Normalization`` (with an
# optional dimension suffix), or a ``torch.nn`` normalization/batch-norm module.
# A substring test on the lowercased name rejected supported neighbours such as
# the OAM ``EdgeLengthNormalizer`` and any ``torch.nn.LayerNorm``; a class that
# merely carries a normalization *suffix* is genuinely a feature normalization,
# while ``...Normalizer`` helpers (edge-length normalizers) are not.  A custom
# normalizer that follows neither naming convention must set the
# ``is_normalization``/``is_normalizer`` marker to be rejected.
_NORMALIZATION_SUFFIX = re.compile(r"(?:norm|normalization)[0-9]*$", re.IGNORECASE)
_UNSUPPORTED_NORMALIZATION_MODULE_PREFIXES = (
    "torch.nn.modules.normalization",
    "torch.nn.modules.batchnorm",
    "torch.nn.modules.instancenorm",
)


def _unsupported_normalization(module: Any) -> bool:
    cls = module if inspect.isclass(module) else type(module)
    name = cls.__qualname__.split(".")[-1]
    if name in _SUPPORTED_NORMALIZATION_NAMES:
        return False
    if any(
        str(getattr(cls, "__module__", "")).startswith(prefix)
        for prefix in _UNSUPPORTED_NORMALIZATION_MODULE_PREFIXES
    ):
        return True
    return _NORMALIZATION_SUFFIX.search(name) is not None


def _qualified_name(value: Any) -> str:
    cls = value if inspect.isclass(value) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _class_name(value: Any) -> str:
    cls = value if inspect.isclass(value) else type(value)
    return cls.__qualname__.split(".")[-1]


def _iter_children(module: Any) -> Iterable[tuple[str, Any]]:
    """Yield child modules through torch's API or simple fixture conventions."""

    named_children = getattr(module, "named_children", None)
    if callable(named_children):
        try:
            yielded_ids: set[int] = set()
            for child_name, child in named_children():
                if isinstance(child, (list, tuple)):
                    for index, item in enumerate(child):
                        yielded_ids.add(id(item))
                        yield f"{child_name}.{index}", item
                elif isinstance(child, dict):
                    for key, item in child.items():
                        yielded_ids.add(id(item))
                        yield f"{child_name}.{key}", item
                else:
                    yielded_ids.add(id(child))
                    yield child_name, child
            for name, value in vars(module).items():
                if name.startswith("_") or name in _SKIP_ATTRIBUTES:
                    continue
                for child_name, child in _marker_children(value, name):
                    if id(child) not in yielded_ids:
                        yielded_ids.add(id(child))
                        yield child_name, child
            return
        except (AttributeError, TypeError):
            pass
    children = getattr(module, "children", None)
    if callable(children):
        try:
            for index, child in enumerate(children()):
                yield str(index), child
            return
        except (AttributeError, TypeError):
            pass

    seen: set[int] = set()

    def walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
        if id(value) in seen:
            return
        seen.add(id(value))
        if _looks_like_module(value):
            yield prefix or _class_name(value), value
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from walk(child, f"{prefix}.{index}" if prefix else str(index))
        elif isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str):
                    yield from walk(child, f"{prefix}.{key}" if prefix else key)

    for name, value in vars(module).items():
        if name.startswith("_") or name in _SKIP_ATTRIBUTES:
            continue
        if _looks_like_module(value) or isinstance(value, (list, tuple, dict)):
            yield from walk(value, name)


def _marker_children(value: Any, prefix: str) -> Iterable[tuple[str, Any]]:
    """Yield only marker-bearing attributes for named-child fallback traversal."""

    lowered_name = _class_name(value).lower()
    if any(
        token in lowered_name
        for token in (
            "zbl",
            "longrange",
            "electrostatic",
            "coulomb",
            "ewald",
            "dispersion",
        )
    ):
        yield prefix, value
        return
    if any(
        hasattr(value, marker)
        for marker_names in _FAMILY_MARKERS.values()
        for marker in marker_names
    ):
        yield prefix, value
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _marker_children(child, f"{prefix}.{index}")
    elif isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str):
                yield from _marker_children(child, f"{prefix}.{key}")


def _looks_like_module(value: Any) -> bool:
    if value is None or inspect.isclass(value) or inspect.ismodule(value):
        return False
    if callable(getattr(value, "named_children", None)):
        return True
    if callable(getattr(value, "children", None)):
        return True
    if any(
        hasattr(value, marker)
        for marker_names in _FAMILY_MARKERS.values()
        for marker in marker_names
    ):
        return True
    name = _class_name(value)
    return bool(
        name.endswith(("Block", "Layer", "Module", "Network", "Basis"))
        or any(
            token in name for token in ("Interaction", "Convolution", "Linear", "ZBL")
        )
    )


def _walk(module: Any, path: str = "") -> dict[str, Any]:
    node: dict[str, Any] = {
        "path": path,
        "class": _qualified_name(module),
        "name": _class_name(module),
    }
    children = []
    for child_name, child in _iter_children(module):
        child_path = f"{path}.{child_name}" if path else child_name
        if child is module:
            continue
        children.append(_walk(child, child_path))
    if children:
        node["children"] = children
    return node


def _get(source: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(source, dict) and name in source:
            return source[name]
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value
    return default


def _nested_get(source: Any, paths: Iterable[tuple[str, ...]]) -> Any:
    for path in paths:
        value = source
        for key in path:
            value = _get(value, key, default=None)
            if value is None:
                break
        else:
            return value
    return None


def _as_symbols(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        # Preserve explicit type order when a symbol-to-index mapping is used.
        if all(isinstance(index, int) for index in value.values()):
            return [
                str(symbol)
                for symbol, _ in sorted(value.items(), key=lambda pair: pair[1])
            ]
        return [str(symbol) for symbol in value]
    if isinstance(value, str):
        return [value]
    try:
        return [str(symbol) for symbol in value]
    except TypeError:
        return [str(value)]


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, dict):
        value = _get(value, "r_max", "r_cut", "cutoff", default=None)
    elif not isinstance(value, (str, int, float)):
        value = _get(value, "r_max", "r_cut", "cutoff", default=value)
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise NequIPProbeError(
            f"NequIP model does not define a numeric {name}"
        ) from error
    if value <= 0:
        raise NequIPProbeError(f"NequIP model {name} must be positive, got {value}")
    return value


def _tensor_descriptor(value: Any) -> dict[str, Any] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        shape = [int(size) for size in shape]
    except (TypeError, ValueError):
        return None
    descriptor: dict[str, Any] = {"shape": shape}
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if dtype is not None:
        descriptor["dtype"] = str(dtype)
    if device is not None:
        descriptor["device"] = str(device)
    return descriptor


def _descriptor(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    tensor = _tensor_descriptor(value)
    if tensor is not None:
        return tensor
    if isinstance(value, dict):
        return {
            str(key): _descriptor(item)
            for key, item in value.items()
            if isinstance(key, (str, int))
        }
    if isinstance(value, (list, tuple)):
        return [_descriptor(item) for item in value]
    return str(value)


def _radial_descriptor(module: Any) -> dict[str, Any] | None:
    radial = _get(
        module, "radial_network", "radial_mlp", "radial_basis", "basis", default=None
    )
    if radial is None:
        return None
    result: dict[str, Any] = {
        "class": _qualified_name(radial),
        "name": _class_name(radial),
    }
    for key in (
        "basis_type",
        "num_basis",
        "number_of_basis",
        "cutoff_type",
        "p",
        "r_max",
        "out_features",
        "hidden_features",
        "normalization",
        "radial_normalization",
        "normalize",
        "prefactor",
        "basis_normalization",
    ):
        value = _get(radial, key, default=None)
        if value is not None:
            result[key] = _descriptor(value)
    return result


def _layer_descriptor(path: str, module: Any) -> dict[str, Any]:
    layer: dict[str, Any] = {
        "path": path,
        "class": _qualified_name(module),
        "name": _class_name(module),
    }
    aliases = {
        "irreps_in": ("irreps_in", "node_irreps_in", "input_irreps"),
        "irreps_out": ("irreps_out", "node_irreps_out", "output_irreps"),
        "edge_irreps": ("edge_irreps", "irreps_edge"),
        "output_keys": ("output_keys",),
    }
    for target, names in aliases.items():
        value = _get(module, *names, default=None)
        if value is not None:
            layer[target] = _descriptor(value)
    radial = _radial_descriptor(module)
    if radial is not None:
        layer["radial_network"] = radial
        layer["radial"] = radial
    for target, names in {
        "edge_tensor_product": ("edge_tensor_product", "tensor_product", "convolution"),
        "message_linear": ("message_linear", "linear", "linear_update"),
        "gate": ("gate", "nonlinearity"),
        "update_nonlinearity": ("update_nonlinearity", "activation"),
    }.items():
        value = _get(module, *names, default=None)
        if value is not None:
            layer[target] = _descriptor(value)
    return layer


def _family_rejection(module: Any) -> str | None:
    name = _class_name(module)
    lowered = name.lower()
    markers = {
        family
        for family, marker_names in _FAMILY_MARKERS.items()
        if any(bool(_get(module, marker, default=False)) for marker in marker_names)
    }
    if "long-range" in markers:
        return f"unsupported long-range module {_qualified_name(module)}; supported family is local ConvNetLayer/InteractionBlock"
    if "plugin" in markers:
        return f"unsupported plugin module {_qualified_name(module)}; supported family is built-in NequIP modules"
    if "interaction" in markers and name not in _INTERACTION_NAMES:
        return f"unsupported interaction module {_qualified_name(module)}; nearest supported family is {_SUPPORTED_FAMILIES['interaction']}"
    if "gate" in markers and name not in _SUPPORTED_GATE_NAMES:
        return f"unsupported gate module {_qualified_name(module)}; nearest supported family is {_SUPPORTED_FAMILIES['gate']}"
    if "normalization" in markers and name not in _SUPPORTED_NORMALIZATION_NAMES:
        return f"unsupported normalization module {_qualified_name(module)}; nearest supported family is {_SUPPORTED_FAMILIES['normalization']}"
    if any(
        token in lowered
        for token in (
            "longrange",
            "electrostatic",
            "coulomb",
            "ewald",
            "dispersion",
            "zbl",
        )
    ):
        return f"unsupported long-range module {_qualified_name(module)}; supported family is local ConvNetLayer/InteractionBlock"
    if "plugin" in lowered or "customop" in lowered:
        return f"unsupported plugin module {_qualified_name(module)}; supported family is built-in NequIP modules"
    if "interaction" in lowered or "convolution" in lowered:
        if name not in _INTERACTION_NAMES:
            return f"unsupported interaction module {_qualified_name(module)}; nearest supported family is {_SUPPORTED_FAMILIES['interaction']}"
    if (
        any(token in lowered for token in ("gate", "nonlinearity"))
        and name not in _SUPPORTED_GATE_NAMES
    ):
        return f"unsupported gate module {_qualified_name(module)}; nearest supported family is {_SUPPORTED_FAMILIES['gate']}"
    if _unsupported_normalization(module):
        return f"unsupported normalization module {_qualified_name(module)}; nearest supported family is {_SUPPORTED_FAMILIES['normalization']}"
    return None


def _find_layers(model: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(module: Any, path: str) -> None:
        if id(module) in seen:
            return
        seen.add(id(module))
        if _class_name(module) in _INTERACTION_NAMES:
            found.append(_layer_descriptor(path, module))
        for child_name, child in _iter_children(module):
            if child is not module:
                visit(child, f"{path}.{child_name}" if path else child_name)

    visit(model, "")
    return found


def _validate_modules(model: Any) -> None:
    seen: set[int] = set()

    def visit(module: Any) -> None:
        if id(module) in seen:
            return
        seen.add(id(module))
        rejection = _family_rejection(module)
        if rejection:
            raise NequIPProbeError(rejection)
        for _, child in _iter_children(module):
            visit(child)

    visit(model)


def _declared_layers(model: Any) -> list[Any] | None:
    declared = _get(model, "layers", default=None)
    if declared is None or isinstance(declared, (str, bytes, dict)):
        return None
    try:
        return list(declared)
    except TypeError:
        return None


def _validate_declared_layers(model: Any) -> list[Any] | None:
    declared = _declared_layers(model)
    if declared is None:
        return None
    for index, layer in enumerate(declared):
        if _class_name(layer) not in _INTERACTION_NAMES:
            raise NequIPProbeError(
                f"unsupported declared interaction module {_qualified_name(layer)} at layers[{index}]; "
                f"nearest supported family is {_SUPPORTED_FAMILIES['interaction']}"
            )
    return declared


def _instantiate_checkpoint(loaded: Any) -> Any:
    """Turn a checkpoint payload into a module using optional NequIP factories."""

    if not isinstance(loaded, dict):
        return loaded
    for key in ("model", "module", "ema"):
        candidate = loaded.get(key)
        if candidate is not None and not isinstance(candidate, dict):
            return candidate
    state = loaded.get("state_dict", loaded.get("model_state_dict"))
    if state is None:
        return loaded
    config = loaded.get("config", loaded.get("model_config"))
    if config is None and isinstance(loaded.get("metadata"), dict):
        config = loaded["metadata"].get(
            "config", loaded["metadata"].get("model_config")
        )
    if config is None:
        raise NequIPProbeError(
            "checkpoint contains a state dictionary but no model config; "
            "provide a NequIP checkpoint with config/model_config"
        )
    try:
        nequip_model = importlib.import_module("nequip.model")
    except ImportError as error:
        raise NequIPDependencyError(
            "cannot instantiate a NequIP checkpoint without optional dependency 'nequip'"
        ) from error
    factories = (
        getattr(nequip_model, "from_config", None),
        getattr(nequip_model, "build_model", None),
        getattr(nequip_model, "load_model", None),
    )
    factory_errors: list[str] = []
    model = None
    for factory in factories:
        if not callable(factory):
            continue
        try:
            model = factory(config)
            break
        except ImportError as error:
            raise NequIPDependencyError(
                f"NequIP checkpoint factory requires an optional dependency: {error}"
            ) from error
        except Exception as error:  # pragma: no cover - version-specific factory APIs
            factory_errors.append(f"{_qualified_name(factory)}: {error}")
    if model is None:
        detail = "; ".join(factory_errors) or "no supported NequIP model factory"
        raise NequIPProbeError(f"cannot instantiate NequIP checkpoint: {detail}")
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        raise NequIPProbeError(
            f"NequIP checkpoint factory returned {_qualified_name(model)} without load_state_dict"
        )
    try:
        load_state_dict(state)
    except ImportError as error:
        raise NequIPDependencyError(
            f"NequIP checkpoint state loading requires an optional dependency: {error}"
        ) from error
    except Exception as error:
        raise NequIPProbeError(
            f"NequIP checkpoint state_dict could not be loaded into {_qualified_name(model)}: {error}"
        ) from error
    return model


def _load_from_path(path: str | Path) -> Any:
    source = str(path)
    if source.endswith(".nequip.zip"):
        try:
            module = importlib.import_module("nequip.model")
        except ImportError as error:
            raise NequIPDependencyError(
                "cannot load NequIP package; install the optional 'nequip' dependency"
            ) from error
        function = getattr(module, "ModelFromPackage", None)
        if not callable(function):
            raise NequIPDependencyError(
                "cannot load NequIP package; installed 'nequip' has no ModelFromPackage"
            )
        try:
            loaded = function(source, compile_mode="eager")
        except ImportError as error:
            raise NequIPDependencyError(
                f"NequIP package loading requires an optional dependency: {error}"
            ) from error
        except Exception as error:  # pragma: no cover - package-specific errors
            raise NequIPProbeError(
                f"NequIP package loader failed; nequip.model.ModelFromPackage: "
                f"{type(error).__name__}: {error}"
            ) from error
        keys = getattr(loaded, "keys", None)
        if callable(keys):
            try:
                if "sole_model" in keys():
                    return loaded["sole_model"]
            except (KeyError, TypeError):
                pass
        return loaded
    if source.endswith(".nequip"):
        loader_errors: list[str] = []
        dependency_errors: list[str] = []
        for module_name, function_name in (
            ("nequip", "load_model"),
            ("nequip.model", "load_model"),
            ("nequip.scripts.deploy", "load_deployed_model"),
            ("nequip.io", "load_file"),
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError as error:
                dependency_errors.append(f"{module_name}: {error}")
                continue
            function = getattr(module, function_name, None)
            if callable(function):
                try:
                    return function(source)
                except NequIPDependencyError:
                    raise
                except NequIPProbeError:
                    raise
                except ImportError as error:
                    dependency_errors.append(f"{module_name}.{function_name}: {error}")
                except Exception as error:  # pragma: no cover - package-specific errors
                    loader_errors.append(
                        f"{module_name}.{function_name}: {type(error).__name__}: {error}"
                    )
        if loader_errors:
            raise NequIPProbeError(
                "NequIP package loader failed; " + "; ".join(loader_errors)
            )
        raise NequIPDependencyError(
            "cannot load NequIP package; install the optional 'nequip' dependency "
            f"(attempts: {'; '.join(dependency_errors) or 'no supported loader'})"
        )
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise NequIPDependencyError(
            "cannot load a NequIP checkpoint without optional dependency 'torch'"
        ) from error
    try:
        loaded = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # Older torch versions lack weights_only.
        loaded = torch.load(source, map_location="cpu")
    return _instantiate_checkpoint(loaded)


def load_nequip_model(model_or_path: Any) -> Any:
    """Load a path once, or return an already-instantiated NequIP module."""

    model = (
        _load_from_path(model_or_path)
        if isinstance(model_or_path, (str, Path))
        else model_or_path
    )
    if model is None:
        raise NequIPProbeError("NequIP model is None")
    if isinstance(model, dict):
        raise NequIPProbeError(
            "NequIP checkpoint did not produce an instantiated model"
        )
    return model


__all__ = [
    "NequIPDependencyError",
    "NequIPProbeError",
    "load_nequip_model",
]
