"""Export supported NequIP modules to the dependency-free Symmetrix contract.

Optional NequIP, torch, and e3nn imports are deliberately kept inside the
path-loading and tensor/Wigner helpers.  The normal Symmetrix runtime can
therefore import this module without installing the NequIP stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

from .jit_codegen_common import (
    spline_certification,
    spline_nodal_values,
    spline_table_fingerprint_from_nodal,
)
from .nequip_probe import NequIPProbeError, load_nequip_model
from .nequip_schema import validate_nequip_oam_data
from .nequip_oam_graph import (
    has_nequip_oam_hint,
    looks_like_nequip_oam_model,
    normalize_nequip_oam_model,
)


_ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
}


def _value(value: Any) -> Any:
    """Convert tensors and scalar array values to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        try:
            value = numpy()
        except TypeError:
            value = numpy(force=True)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _tensor(
    value: Any, shape: list[int] | None = None, *, fill: float = 0.0
) -> dict[str, Any]:
    converted = _value(value)
    if converted is not None and isinstance(converted, (list, tuple, int, float)):
        if (
            shape is not None
            and isinstance(converted, (list, tuple))
            and not _flatten(converted)
        ):
            size = 1
            for extent in shape:
                size *= extent
            return {"shape": shape, "values": [fill] * size}
        if shape is None:
            if isinstance(converted, (int, float)):
                converted = [converted]
            shape = _shape(converted)
        return {"shape": shape, "values": _flatten(converted)}
    shape = shape or [1]
    size = 1
    for extent in shape:
        size *= extent
    return {"shape": shape, "values": [fill] * size}


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [float(value)]


def _shape(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    if not value:
        return [0]
    return [len(value), *_shape(value[0])]


def _get(source: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(source, dict) and name in source:
            return source[name]
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value
    return default


def _find_module(model: Any, path: str, fallback: Any = None) -> Any:
    current = model
    for part in path.split(".") if path else ():
        if isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return fallback
        elif isinstance(current, dict):
            current = current.get(part, fallback)
        else:
            current = getattr(current, part, fallback)
        if current is fallback:
            return fallback
    return current


def _module_name(module: Any) -> str:
    """Return a module-qualified class name for diagnostics."""

    if module is None:
        return "None"
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


class _ExportPolicy:
    """Track semantic substitutions instead of making them silently.

    The legacy NequIP exporter also accepts *metadata-only* descriptors, where
    no weight-bearing module exists at all and a zero tensor is the documented
    representation of the missing block.  The packaged OAM exporter always
    works from an instantiated graph, so a missing internal weight there means
    the model cannot be represented and must be rejected rather than replaced by
    zeros.  Either way every substitution is recorded in
    ``model_metadata.exporter_defaults`` so an exported contract declares what
    it had to invent.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self.defaults: list[str] = []

    def replace_weight(self, module: Any, shape: list[int]) -> dict[str, Any]:
        if self.strict and module is not None:
            raise NequIPProbeError(
                f"{_module_name(module)} exposes no internal path weight; "
                "externally supplied (shared/internal_weights=False) tensor-product "
                "weights cannot be represented in the fixed contract"
            )
        self.defaults.append(
            f"{_module_name(module)}: weight defaulted to zeros (shape {shape})"
        )
        return _tensor(None, shape)

    def replace_default(
        self, module: Any, description: str, shape: list[int], fill: float
    ) -> dict[str, Any]:
        self.defaults.append(
            f"{_module_name(module)}: {description} defaulted to "
            f"{'ones' if fill else 'zeros'} (shape {shape})"
        )
        return _tensor(None, shape, fill=fill)


def _looks_like_silu(layer: Any) -> bool:
    return (
        "silu" in type(layer).__name__.lower()
        or "silu" in str(_get(layer, "type", default="")).lower()
    )


def _irreps_dimension(text: str) -> int:
    total = 0
    for token in str(text).replace(" ", "").split("+"):
        multiplicity, rest = token.split("x", 1)
        total += int(multiplicity) * (2 * int(rest[:-1]) + 1)
    return total


def _blocks(text: str) -> list[tuple[int, int]]:
    result = []
    for token in str(text).replace(" ", "").split("+"):
        multiplicity, rest = token.split("x", 1)
        result.append((int(multiplicity), int(rest[:-1])))
    return result


def _instruction_value(instruction: Any, name: str, default: Any = None) -> Any:
    return _get(instruction, name, default=default)


def _wigner(l1: int, l2: int, l3: int) -> dict[str, Any]:
    shape = [2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1]
    try:
        import torch

        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            from e3nn.o3 import wigner_3j
        else:
            # e3nn 0.4.x stores trusted package constants containing slices.
            # Newer PyTorch releases require this explicit weights-only allowlist.
            with safe_globals([slice]):
                from e3nn.o3 import wigner_3j

        values = _value(wigner_3j(l1, l2, l3))
    except (ImportError, RuntimeError, TypeError) as error:
        if (l1, l2, l3) == (0, 0, 0):
            values = [1.0]
        else:
            raise NequIPProbeError(
                "exact Wigner 3j coefficients require e3nn or preserved coefficients; "
                f"cannot serialize ({l1}, {l2}, {l3})"
            ) from error
    return _tensor(values, shape)


def _preserved_wigner(value: Any, l1: int, l2: int, l3: int) -> dict[str, Any] | None:
    if value is None:
        return None
    shape = [2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1]
    if isinstance(value, dict) and "values" in value:
        candidate = _tensor(value["values"], list(value.get("shape", shape)))
    else:
        candidate = _tensor(value, shape)
    if candidate["shape"] != shape or not any(
        abs(float(item)) > 0.0 for item in candidate["values"]
    ):
        raise NequIPProbeError(
            "preserved Wigner 3j coefficients have the wrong shape or are all zero"
        )
    return candidate


def _serialize_linear(
    module: Any,
    irreps_in: str,
    irreps_out: str,
    *,
    readout: bool = False,
    policy: _ExportPolicy | None = None,
) -> dict[str, Any]:
    policy = _ExportPolicy() if policy is None else policy
    in_blocks = _blocks(irreps_in)
    out_blocks = _blocks(irreps_out) if irreps_out else []
    instructions = _get(module, "instructions", default=None)
    serialized = []
    if instructions is None:
        for i_in, (_, l_in) in enumerate(in_blocks):
            for i_out, (_, l_out) in enumerate(out_blocks):
                if l_in == l_out:
                    m_in, m_out = in_blocks[i_in][0], out_blocks[i_out][0]
                    serialized.append(
                        {
                            "i_in": i_in,
                            "i_out": i_out,
                            "path_shape": [m_in, m_out],
                            "path_weight": 1.0,
                        }
                    )
    else:
        for instruction in instructions:
            i_in = int(_instruction_value(instruction, "i_in", 0))
            i_out = int(_instruction_value(instruction, "i_out", 0))
            serialized.append(
                {
                    "i_in": i_in,
                    "i_out": i_out,
                    "path_shape": list(
                        _instruction_value(
                            instruction,
                            "path_shape",
                            [in_blocks[i_in][0], out_blocks[i_out][0]],
                        )
                    ),
                    "path_weight": float(
                        _instruction_value(instruction, "path_weight", 1.0)
                    ),
                }
            )
    weight = _get(module, "weight", default=None)
    # NequIP's scalar readout is a ScalarMLP wrapper.  Its only affine
    # layer lives below ``mlp_module.mlp`` and stores weights as [in, out],
    # unlike e3nn Linear's flattened path weights.
    scalar_layer = None
    if readout and weight is None:
        mlp_module = _get(module, "mlp_module", default=None)
        mlp = _get(mlp_module, "mlp", default=None)
        children = (
            list(mlp.children()) if callable(getattr(mlp, "children", None)) else []
        )
        for child in children:
            if _get(child, "weight", default=None) is not None:
                scalar_layer = child
                weight = _get(child, "weight")
                break
        if weight is not None:
            raw = _value(weight)
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                raw = [list(column) for column in zip(*raw)]
            alpha = _get(scalar_layer, "alpha", default=1.0)
            alpha = float(_value(alpha))
            raw = [alpha * float(value) for value in _flatten(raw)]
            weight = _tensor(raw, [_irreps_dimension(irreps_in)])
    if weight is None:
        count = sum(
            int(item["path_shape"][0]) * int(item["path_shape"][1])
            for item in serialized
        )
        weight = policy.replace_weight(
            module,
            [1, _irreps_dimension(irreps_in)] if readout else [count],
        )
    elif not isinstance(weight, dict):
        weight = _tensor(weight)
    out_dim = 1 if readout else _irreps_dimension(irreps_out)
    bias_source = _get(module, "bias", default=None)
    bias = (
        policy.replace_default(module, "bias", [out_dim], 0.0)
        if bias_source is None
        else _tensor(bias_source, [out_dim])
    )
    mask_source = _get(module, "output_mask", default=None)
    mask = (
        policy.replace_default(module, "output_mask", [out_dim], 1.0)
        if mask_source is None
        else _tensor(mask_source, [out_dim], fill=1.0)
    )
    result = {
        "irreps_in": irreps_in,
        "instructions": serialized,
        "weight": weight,
        "bias": bias,
        "output_mask": mask,
    }
    if irreps_out:
        result["irreps_out"] = irreps_out
    return result


def _serialize_product(
    module: Any,
    node_in: str,
    edge: str,
    node_out: str,
    *,
    policy: _ExportPolicy | None = None,
) -> dict[str, Any]:
    policy = _ExportPolicy() if policy is None else policy
    in1, in2, out = _blocks(node_in), _blocks(edge), _blocks(node_out)
    instructions = _get(module, "instructions", default=None)
    serialized = []
    if instructions is None and module is not None:
        raise NequIPProbeError(
            f"{_module_name(module)} exposes no tensor-product instructions; "
            "re-deriving the path set from irreps would silently change the model"
        )
    if instructions is None:
        for i, (m1, l1) in enumerate(in1):
            for j, (m2, l2) in enumerate(in2):
                for k, (m3, l3) in enumerate(out):
                    if m3 == m1 and abs(l1 - l2) <= l3 <= l1 + l2:
                        # Metadata-only fixtures have no tensor-product module
                        # from which coefficients can be preserved. Keep their
                        # exact scalar path; real modules remain strict below.
                        if module is None and (l1, l2, l3) != (0, 0, 0):
                            continue
                        mode = "uvu" if m2 > 1 else "uuu"
                        serialized.append(
                            {
                                "i_in1": i,
                                "i_in2": j,
                                "i_out": k,
                                "connection_mode": mode,
                                "has_weight": True,
                                "path_weight": 1.0,
                                "path_shape": [m1, m2] if mode == "uvu" else [m1],
                                "wigner_3j": _wigner(l1, l2, l3),
                            }
                        )
    else:
        for instruction in instructions:
            i, j, k = (
                int(_instruction_value(instruction, name, 0))
                for name in ("i_in1", "i_in2", "i_out")
            )
            l1, l2, l3 = in1[i][1], in2[j][1], out[k][1]
            mode = str(_instruction_value(instruction, "connection_mode", "uvu"))
            shape = list(
                _instruction_value(
                    instruction,
                    "path_shape",
                    [in1[i][0], in2[j][0]] if mode == "uvu" else [in1[i][0]],
                )
            )
            wigner = _preserved_wigner(
                _instruction_value(instruction, "wigner_3j"), l1, l2, l3
            ) or _wigner(l1, l2, l3)
            serialized.append(
                {
                    "i_in1": i,
                    "i_in2": j,
                    "i_out": k,
                    "connection_mode": mode,
                    "has_weight": bool(
                        _instruction_value(instruction, "has_weight", True)
                    ),
                    "path_weight": float(
                        _instruction_value(instruction, "path_weight", 1.0)
                    ),
                    "path_shape": shape,
                    "wigner_3j": wigner,
                }
            )
    count = sum(
        (item["path_shape"][0] * item["path_shape"][1])
        if item["has_weight"] and item["connection_mode"] == "uvu"
        else (item["path_shape"][0] if item["has_weight"] else 0)
        for item in serialized
    )
    weight_source = _get(module, "weight", default=None)
    weight = (
        policy.replace_weight(module, [count])
        if weight_source is None
        else _tensor(weight_source, [count])
    )
    mask_source = _get(module, "output_mask", default=None)
    mask = (
        policy.replace_default(
            module, "output_mask", [_irreps_dimension(node_out)], 1.0
        )
        if mask_source is None
        else _tensor(mask_source, [_irreps_dimension(node_out)], fill=1.0)
    )
    return {
        "irreps_in1": node_in,
        "irreps_in2": edge,
        "irreps_out": node_out,
        "instructions": serialized,
        "weight": weight,
        "output_mask": mask,
    }


def _radial(
    module: Any, dynamic_count: int, *, policy: _ExportPolicy | None = None
) -> dict[str, Any]:
    policy = _ExportPolicy() if policy is None else policy
    module = module or object()
    num_basis = int(_get(module, "num_basis", "number_of_basis", default=1))
    layers = _get(module, "layers", default=None)
    serialized = []
    if layers is not None:
        for layer in layers:
            weight = _get(layer, "weight", default=None)
            bias = _get(layer, "bias", default=None)
            if weight is not None:
                raw = _value(weight)
                shape = _shape(raw)
                serialized.append(
                    {"weight": _tensor(raw, shape), "bias": _tensor(bias, [shape[0]])}
                )
    if not serialized:
        policy.defaults.append(
            f"{_module_name(module)}: radial network layers defaulted to a single "
            f"zero weight (shape [{dynamic_count}, {num_basis}])"
        )
        serialized = [
            {
                "weight": _tensor(None, [dynamic_count, num_basis]),
                "bias": _tensor(None, [dynamic_count]),
            }
        ]
    result = {
        "type": str(_get(module, "basis_type", "type", default="bessel")).lower(),
        "num_basis": num_basis,
        "cutoff": {
            "type": str(_get(module, "cutoff_type", default="polynomial")).lower(),
            "p": float(_get(module, "p", default=6.0)),
        },
        "mlp": {"layers": serialized},
    }
    for name in (
        "normalization",
        "radial_normalization",
        "normalize",
        "prefactor",
        "basis_normalization",
    ):
        value = _get(module, name, default=None)
        if value is not None:
            raise NequIPProbeError(
                f"radial basis {name} metadata is unsupported by the fixed radial basis"
            )
    return result


def _activation(
    module: Any, descriptor: Any, default: str = "identity"
) -> dict[str, Any]:
    source = module if module is not None else descriptor
    if isinstance(source, dict):
        result = {str(key): _value(value) for key, value in source.items()}
        result.setdefault("type", str(result.get("name", default)).lower())
        return result
    if source is None:
        return {"type": default}
    result = {"type": type(source).__name__.lower()}
    for target, names in {
        "scalar_irreps": ("scalar_irreps", "irreps_scalars"),
        "gate_irreps": ("gate_irreps", "irreps_gates"),
        "gated_irreps": ("gated_irreps", "irreps_gated"),
    }.items():
        value = _get(source, *names, default=None)
        if value is not None:
            result[target] = _value(value)
    return result


def _layer(
    model: Any, descriptor: dict[str, Any], *, policy: _ExportPolicy | None = None
) -> dict[str, Any]:
    policy = _ExportPolicy() if policy is None else policy
    node_in = str(descriptor.get("irreps_in", "1x0e"))
    edge = str(descriptor.get("edge_irreps", "1x0e"))
    node_out = str(descriptor.get("irreps_out", node_in))
    module = _find_module(model, str(descriptor.get("path", "")), None)
    product_module = _get(
        module, "edge_tensor_product", "tensor_product", "convolution", default=None
    )
    radial_module = _get(
        module, "radial_network", "radial_mlp", "radial_basis", "basis", default=None
    )
    message_module = _get(
        module, "message_linear", "linear", "linear_update", default=None
    )
    self_module = _get(module, "self_connection", "skip", default=None)
    gate_module = _get(
        module, "gate", "nonlinearity", default=_get(model, "gate", default=None)
    )
    update_module = _get(module, "update_nonlinearity", "activation", default=None)
    gate = _activation(gate_module, descriptor.get("gate"))
    try:
        node_pre_gate = str(descriptor["node_irreps_pre_gate"])
    except KeyError:
        node_pre_gate = "+".join(
            value
            for value in (
                gate.get("scalar_irreps"),
                gate.get("gate_irreps"),
                gate.get("gated_irreps"),
            )
            if value
        )
    if not node_pre_gate:
        raise NequIPProbeError(
            f"layer {descriptor.get('name', descriptor.get('path', 'interaction'))} "
            "does not expose explicit scalar/gate/gated irreps"
        )
    product = _serialize_product(
        product_module, node_in, edge, node_pre_gate, policy=policy
    )
    dynamic_count = len(product["weight"]["values"])
    return {
        "name": str(descriptor.get("name", descriptor.get("path", "interaction"))),
        "node_irreps_in": node_in,
        "edge_irreps": edge,
        "node_irreps_out": node_out,
        "node_irreps_pre_gate": node_pre_gate,
        "radial_basis": _radial(
            radial_module or descriptor.get("radial_network"),
            dynamic_count,
            policy=policy,
        ),
        "edge_tensor_product": product,
        "message_linear": _serialize_linear(
            message_module, node_pre_gate, node_pre_gate, policy=policy
        ),
        "self_connection": _serialize_linear(
            self_module, node_in, node_pre_gate, policy=policy
        ),
        "gate": gate,
        "update_nonlinearity": _activation(
            update_module, descriptor.get("update_nonlinearity")
        ),
    }


def _oam_instruction(value: Any, index: int) -> dict[str, Any]:
    """Normalize e3nn tuple or object tensor-product instructions."""
    if isinstance(value, (tuple, list)):
        # e3nn exposes ``Instruction`` as a named tuple in packaged models;
        # retain its path weight and shape instead of treating it as the
        # five-field shorthand used by older fixtures.
        names = getattr(
            value,
            "_fields",
            ("i_in1", "i_in2", "i_out", "connection_mode", "has_weight"),
        )
        result = {name: item for name, item in zip(names, value)}
        result.setdefault("path_weight", 1.0)
        return result
    return {
        "i_in1": int(_instruction_value(value, "i_in1", index)),
        "i_in2": int(_instruction_value(value, "i_in2", 0)),
        "i_out": int(_instruction_value(value, "i_out", 0)),
        "connection_mode": str(_instruction_value(value, "connection_mode", "uvu")),
        "has_weight": bool(_instruction_value(value, "has_weight", True)),
        "path_weight": float(_instruction_value(value, "path_weight", 1.0)),
    }


def _serialize_oam_product(
    module: Any,
    node_in: str,
    edge: str,
    node_out: str,
    *,
    policy: _ExportPolicy | None = None,
) -> dict[str, Any]:
    """Serialize OAM's nested dynamic tensor product, retaining UVU weights."""
    policy = _ExportPolicy() if policy is None else policy
    tp = _get(module, "tp", "tensor_product", default=module) or object()
    instructions = _get(tp, "instructions", default=None) or []
    in1, in2, out = _blocks(node_in), _blocks(edge), _blocks(node_out)
    serialized = []
    for index, raw in enumerate(instructions):
        item = _oam_instruction(raw, index)
        i, j, k = item["i_in1"], item["i_in2"], item["i_out"]
        mode = item["connection_mode"]
        shape = _instruction_value(raw, "path_shape", None)
        if shape is None:
            shape = [in1[i][0], in2[j][0]] if mode == "uvu" else [in1[i][0]]
        wigner = _preserved_wigner(
            _instruction_value(raw, "wigner_3j", None), in1[i][1], in2[j][1], out[k][1]
        ) or _wigner(in1[i][1], in2[j][1], out[k][1])
        serialized.append({**item, "path_shape": list(shape), "wigner_3j": wigner})
    weight_numel = _get(tp, "weight_numel", "weight_numel_total", default=None)
    if weight_numel is None:
        weight_numel = sum(
            (x["path_shape"][0] * x["path_shape"][1])
            if x["has_weight"] and x["connection_mode"] == "uvu"
            else (x["path_shape"][0] if x["has_weight"] else 0)
            for x in serialized
        )
    weight_source = _get(tp, "weight", default=None)
    mask_source = _get(tp, "output_mask", default=None)
    return {
        "irreps_in1": node_in,
        "irreps_in2": edge,
        "irreps_out": node_out,
        "instructions": serialized,
        "weight_numel": int(weight_numel),
        "weight": (
            policy.replace_weight(tp, [int(weight_numel)])
            if weight_source is None
            else _tensor(weight_source, [int(weight_numel)])
        ),
        "output_mask": (
            policy.replace_default(
                tp, "output_mask", [_irreps_dimension(node_out)], 1.0
            )
            if mask_source is None
            else _tensor(mask_source, [_irreps_dimension(node_out)], fill=1.0)
        ),
    }


def _serialize_oam_radial(
    graph: Any, interaction: Any, *, policy: _ExportPolicy | None = None
) -> dict[str, Any]:
    policy = _ExportPolicy() if policy is None else policy
    radial = (
        _get(interaction, "edge_mlp", "radial_network", "radial_mlp", default=None)
        or object()
    )
    layers = []
    raw_layers = _get(radial, "layers", default=None)
    if raw_layers is None:
        mlp = _get(radial, "mlp", default=None)
        named_children = getattr(mlp, "named_children", None)
        raw_layers = (
            list(mlp.children())
            if callable(getattr(mlp, "children", None))
            else (
                list(dict(named_children()).values())
                if callable(named_children)
                else ()
            )
        )
    for layer in raw_layers or ():
        weight = _get(layer, "weight", default=None)
        if weight is None:
            if _looks_like_silu(layer):
                layers.append({"type": "silu"})
                continue
            raise NequIPProbeError(
                f"unsupported radial activation {_module_name(layer)} in "
                f"{_module_name(radial)}: only SiLU is representable in the "
                "fixed radial basis; dropping the layer would silently change "
                "the radial network"
            )
        raw = _value(weight)
        alpha = float(_value(_get(layer, "alpha", default=1.0)))
        if (
            "scalarlinearlayer" in type(layer).__name__.lower()
            and isinstance(raw, list)
            and raw
            and isinstance(raw[0], list)
        ):
            raw = [list(column) for column in zip(*raw)]
        if alpha != 1.0:
            raw = [[alpha * float(value) for value in row] for row in raw]
        shape = _shape(raw)
        bias_source = _get(layer, "bias", default=None)
        layers.append(
            {
                "type": "linear",
                "weight": _tensor(raw, shape),
                "bias": (
                    policy.replace_default(layer, "radial bias", [shape[0]], 0.0)
                    if bias_source is None
                    else _tensor(bias_source, [shape[0]])
                ),
            }
        )
    if not layers:
        raise NequIPProbeError(
            f"{_module_name(radial)} exposes no radial network layers; a zero "
            "radial network cannot be substituted"
        )
    frequencies = graph.edge_embedding.bessel_weights
    num_basis = int(
        graph.edge_embedding.num_basis
        or (_shape(_value(frequencies))[0] if frequencies is not None else 1)
    )
    return {
        "type": "bessel",
        "num_basis": num_basis,
        "bessel_weights": _tensor(frequencies, [num_basis]),
        "cutoff": {
            "type": "polynomial",
            "p": float(graph.edge_embedding.cutoff_p or 6.0),
        },
        "input_factor": float(
            graph.edge_embedding.factor
            if graph.edge_embedding.factor is not None
            else 1.0
        ),
        "mlp": {"layers": layers},
    }


def _oam_gate(module: Any, pre_gate: str, out: str) -> dict[str, Any]:
    gate = module or object()
    scalar = _get(gate, "scalar_irreps", "irreps_scalars", default="1x0e")
    gates = _get(gate, "gate_irreps", "irreps_gates", default="0x0e")
    gated = _get(gate, "gated_irreps", "irreps_gated", default=out)
    if scalar in (None, ""):
        scalar = "0x0e"
    if gates in (None, ""):
        gates = "0x0e"
    if gated in (None, ""):
        gated = "0x0e"
    return {
        "type": "scalar_gate",
        "scalar_irreps": str(scalar),
        "gate_irreps": str(gates),
        "gated_irreps": str(gated),
        "irreps_in": pre_gate,
        "irreps_out": out,
    }


def _oam_zbl(
    model: Any, graph: Any, *, policy: _ExportPolicy | None = None
) -> dict[str, Any]:
    policy = _ExportPolicy() if policy is None else policy
    source = _get(
        model,
        "zbl",
        "pair_potential",
        "pair_repulsion_fn",
        "pair_repulsion",
        default=None,
    )
    if source is None:
        raise NequIPProbeError(
            f"{_module_name(model)} exposes no ZBL/pair-repulsion module, but the "
            "explicit OAM contract always carries a ZBL term; substituting the "
            "library screening constants would add a potential the model lacks"
        )
    numbers = _atomic_numbers(list(graph.type_names))[0]
    cutoff = _get(source, "cutoff", "r_cut", default=None)
    if cutoff is None:
        cutoff = graph.r_cut
        policy.defaults.append(
            f"{_module_name(source)}: ZBL cutoff defaulted to r_cut ({cutoff})"
        )
    cutoff_p = _get(source, "cutoff_p", "polynomial_cutoff_p", "p", default=None)
    if cutoff_p is None:
        cutoff_p = graph.edge_embedding.cutoff_p or 6.0
        policy.defaults.append(
            f"{_module_name(source)}: ZBL cutoff exponent defaulted to the edge "
            f"cutoff order ({cutoff_p})"
        )
    prefactor = _get(source, "half_energy_prefactor", "prefactor", default=None)
    if prefactor is None:
        prefactor = 14.399645 / 2.0
        policy.defaults.append(
            f"{_module_name(source)}: ZBL half-energy prefactor defaulted to the "
            f"library constant ({prefactor})"
        )
    exponents = _get(source, "screening_exponents", "exponents", default=None)
    if exponents is None:
        exponents = [-0.20162, -0.40290, -0.94229, -3.19980]
        policy.defaults.append(
            f"{_module_name(source)}: ZBL screening exponents defaulted to the "
            "library constants"
        )
    coefficients = _get(source, "screening_coefficients", "coefficients", default=None)
    if coefficients is None:
        coefficients = [0.02817, 0.28022, 0.50986, 0.18175]
        policy.defaults.append(
            f"{_module_name(source)}: ZBL screening coefficients defaulted to the "
            "library constants"
        )
    return {
        "type": "zbl",
        "cutoff": float(cutoff),
        "cutoff_p": float(cutoff_p),
        "half_energy_prefactor": float(prefactor),
        "atomic_numbers": numbers,
        "screening_exponents": _value(exponents),
        "screening_coefficients": _value(coefficients),
    }


def _oam_radial_evaluator(radial: dict[str, Any], *, cutoff: float):
    """Return ``(evaluate, function_count)`` for the exported radial MLP.

    The evaluation order matches the native ``radial_features``/
    ``build_from_network`` pair term for term.
    """

    frequencies = [float(value) for value in radial["bessel_weights"]["values"]]
    cutoff_power = float(radial["cutoff"]["p"])
    input_factor = float(radial.get("input_factor", 1.0))
    layers = radial["mlp"]["layers"]
    function_count = len(layers[-1]["bias"]["values"])

    def evaluate(radius: float) -> list[float]:
        x = radius / cutoff
        basis = [0.0] * len(frequencies)
        if x < 1.0:
            if x <= 0.0:
                smooth = 1.0
            else:
                xp = x**cutoff_power
                smooth = (
                    1.0
                    - ((cutoff_power + 1.0) * (cutoff_power + 2.0) / 2.0) * xp
                    + cutoff_power * (cutoff_power + 2.0) * x ** (cutoff_power + 1.0)
                    - (cutoff_power * (cutoff_power + 1.0) / 2.0)
                    * x ** (cutoff_power + 2.0)
                )
            for index, frequency in enumerate(frequencies):
                argument = x * frequency
                bessel = (
                    frequency
                    if abs(argument) < 1.0e-14
                    else (
                        math.sin(math.pi * argument) * frequency / (math.pi * argument)
                    )
                )
                basis[index] = bessel * smooth * input_factor
        values = basis
        for index, layer in enumerate(layers):
            if layer.get("type") == "silu":
                values = [value / (1.0 + math.exp(-value)) for value in values]
                continue
            weights = layer["weight"]["values"]
            shape = layer["weight"]["shape"]
            bias = layer["bias"]["values"]
            values = [
                bias[row]
                + sum(
                    float(weights[row * shape[1] + column]) * values[column]
                    for column in range(shape[1])
                )
                for row in range(shape[0])
            ]
            if index + 1 < len(layers) and "type" not in layers[index + 1]:
                values = [value / (1.0 + math.exp(-value)) for value in values]
        return values

    return evaluate, function_count


def extract_nequip_oam_data(
    model: Any, *, output_precision: str = "float64"
) -> dict[str, Any]:
    """Extract a packaged NequIP-OAM graph into the explicit v2 contract."""
    graph = normalize_nequip_oam_model(model)
    if not graph.layers or not graph.r_cut:
        raise NequIPProbeError("NequIP OAM model does not expose layers and cutoff")
    policy = _ExportPolicy(strict=True)
    universal, model_numbers, mapping = _atomic_numbers(list(graph.type_names))
    layers = []
    for layer in graph.layers:
        interaction = layer.interaction
        node_in = str(layer.node_irreps_in or "1x0e")
        node_out = str(layer.node_irreps_out or node_in)
        gate_source = layer.gate or _get(
            interaction, "gate", "nonlinearity", default=None
        )
        gate_scalar = _get(gate_source, "scalar_irreps", "irreps_scalars", default=None)
        gate_gates = _get(gate_source, "gate_irreps", "irreps_gates", default=None)
        gate_gated = _get(gate_source, "gated_irreps", "irreps_gated", default=None)
        inferred_pre = "+".join(
            str(item) for item in (gate_scalar, gate_gates, gate_gated) if item
        )
        linear_1 = _get(interaction, "linear_1", "pre_linear", default=None)
        linear_2 = _get(interaction, "linear_2", "post_linear", default=None)
        linear_1_out = str(_get(linear_1, "irreps_out", default=node_in))
        linear_2_in = str(_get(linear_2, "irreps_in", default=node_out))
        pre_gate = str(
            getattr(layer, "node_irreps_pre_gate", None)
            or _get(
                interaction,
                "node_irreps_pre_gate",
                default=linear_2_in or inferred_pre or node_out,
            )
        )
        product = _get(interaction, "tp_scatter", "tensor_product", default=None)
        tp_module = _get(product, "tp", "tensor_product", default=product)
        tp_output = str(_get(tp_module, "irreps_out", default=linear_2_in))
        edge = str(layer.edge_irreps or "1x0e")
        radial = _serialize_oam_radial(graph, interaction, policy=policy)
        # NequIP's packaged InteractionBlock stores the fixed average-neighbor
        # normalization as ``scatter_norm_factor``.  Older graph adapters may
        # expose the equivalent nested normalization object instead.
        norm = _get(
            interaction,
            "scatter_norm_factor",
            "avg_num_neighbors_norm",
            "normalization",
            default=None,
        )
        norm_factor = (
            norm
            if isinstance(norm, (int, float))
            else _get(norm, "norm_const", "factor", default=1.0)
        )
        if isinstance(norm_factor, (list, tuple)):
            norm_factor = norm_factor[0]
        layers.append(
            {
                "node_irreps_in": node_in,
                "edge_irreps": edge,
                "tp_output_irreps": tp_output,
                "node_irreps_pre_gate": pre_gate,
                "node_irreps_out": node_out,
                "pre_linear": _serialize_linear(
                    linear_1, node_in, linear_1_out, policy=policy
                ),
                "radial_basis": radial,
                "edge_tensor_product": _serialize_oam_product(
                    product, linear_1_out, edge, tp_output, policy=policy
                ),
                "post_linear": _serialize_linear(
                    linear_2, linear_2_in, pre_gate, policy=policy
                ),
                "self_connection": (
                    _serialize_oam_product(
                        layer.self_connection,
                        str(_get(layer.self_connection, "irreps_in1", default=node_in)),
                        str(_get(layer.self_connection, "irreps_in2", default="1x0e")),
                        str(
                            _get(layer.self_connection, "irreps_out", default=pre_gate)
                        ),
                        policy=policy,
                    )
                    if layer.self_connection is not None
                    and _get(layer.self_connection, "irreps_in1", default=None)
                    is not None
                    else (
                        _serialize_linear(
                            layer.self_connection, node_in, pre_gate, policy=policy
                        )
                        if layer.self_connection is not None
                        else None
                    )
                ),
                "normalization": {
                    "type": "global_scalar",
                    "factor": float(norm_factor),
                },
                "gate": _oam_gate(gate_source, pre_gate, node_out),
            }
        )
    metadata = _get(model, "metadata", default={})
    fingerprint = _get(
        metadata, "exporter_fingerprint", "fingerprint", default=None
    ) or ("sha256:" + hashlib.sha256(repr(graph.as_dict()).encode()).hexdigest())
    model_fingerprint = (
        "sha256:" + hashlib.sha256(str(fingerprint).encode("utf-8")).hexdigest()
    )
    for layer in layers:
        radial = layer["radial_basis"]
        function_count = int(layer["edge_tensor_product"]["weight_numel"])
        num_points = 256
        grid_min = 0.0
        cutoff_value = float(graph.r_cut)
        spacing = cutoff_value / float(num_points - 1)
        evaluate, radial_function_count = _oam_radial_evaluator(
            radial, cutoff=cutoff_value
        )
        nodal_values, _ = spline_nodal_values(
            evaluate,
            radial_function_count,
            num_points=num_points,
            grid_min=grid_min,
            cutoff=cutoff_value,
        )
        table_fingerprint = spline_table_fingerprint_from_nodal(
            grid_min=grid_min,
            cutoff=cutoff_value,
            num_points=num_points,
            function_count=len(nodal_values),
            nodal_values=nodal_values,
        )
        certification = spline_certification(
            evaluate,
            num_points=num_points,
            grid_min=grid_min,
            cutoff=cutoff_value,
            nodal_values=nodal_values,
        )
        layer["kokkos_radial"] = {
            "grid_min": grid_min,
            "cutoff": cutoff_value,
            "num_spline_points": num_points,
            "spacing": spacing,
            "function_count": function_count,
            "edge_type_count": 1,
            "model_fingerprint": model_fingerprint,
            "table_fingerprint": table_fingerprint,
            # Records what this export is certified for.  It selects nothing at
            # runtime: the native builder always evaluates the spline in double
            # and casts on load, so a float64 export is valid for a float32
            # evaluator while a float32 export is refused by a float64 one.  No
            # coefficient is quantized here.
            "precision": output_precision,
            "exporter_version": "symmetrix-nequip-exporter/2",
            "certification": certification,
        }
    graph_func = _get(_get(model, "model", default=None), "func", default=None)
    readout_module = graph.readout
    final_irreps = layers[-1]["node_irreps_out"]
    scale_shift = _get(
        model,
        "per_type_energy_scale_shift",
        "scale_shift",
        "per_type_scale_shift",
        default=_get(
            graph_func, "per_type_energy_scale_shift", "scale_shift", default=None
        ),
    )
    extracted_shift = _get(
        scale_shift,
        "shift",
        "shifts",
        default=_get(model, "per_type_shift", "shift", default=None),
    )
    extracted_scale = _get(
        scale_shift,
        "scale",
        "scales",
        default=_get(model, "per_type_scale", "scale", default=None),
    )
    data = {
        "model_type": "NequIP",
        "architecture": "oam",
        "symmetrix_nequip_format_version": 2,
        "model_metadata": {
            "source_package": "nequip-oam",
            "source_version": str(_get(metadata, "nequip_version", default="unknown")),
            "exporter_version": "symmetrix-nequip-exporter/2",
            "exporter_fingerprint": str(fingerprint),
            "output_precision": output_precision,
            "exporter_defaults": list(policy.defaults),
        },
        "atomic_numbers": universal,
        "model_atomic_numbers": model_numbers,
        "atomic_number_to_model_type": mapping,
        "r_cut": float(graph.r_cut),
        "node_attributes": _tensor(graph.node_embedding.weight),
        "edge_embedding": {
            "bessel_weights": _tensor(
                graph.edge_embedding.bessel_weights,
                [int(graph.edge_embedding.num_basis or 1)],
            ),
            "cutoff": {
                "type": "polynomial",
                "p": float(graph.edge_embedding.cutoff_p or 6.0),
            },
            "factor": float(
                graph.edge_embedding.factor
                if graph.edge_embedding.factor is not None
                else 1.0
            ),
        },
        "num_layers": len(layers),
        "layers": layers,
        "readout": _serialize_linear(
            readout_module, final_irreps, "1x0e", readout=True, policy=policy
        ),
        "per_type_shift": _tensor(extracted_shift, [len(model_numbers)]),
        "per_type_scale": _tensor(extracted_scale, [len(model_numbers)], fill=1.0),
        "zbl": _oam_zbl(graph_func or model, graph, policy=policy),
        "output_keys": list(graph.output_keys),
        "has_stress": graph.has_stress,
    }
    validate_nequip_oam_data(data)
    return data


def _atomic_numbers(symbols: list[str]) -> tuple[list[int], list[int], dict[str, int]]:
    numbers = []
    for symbol in symbols:
        try:
            numbers.append(_ATOMIC_NUMBERS[symbol])
        except KeyError as error:
            raise ValueError(f"unknown NequIP element symbol {symbol!r}") from error
    universal = sorted(set(numbers))
    mapping = {str(number): numbers.index(number) for number in universal}
    return universal, numbers, mapping


def _source_declares_nequip_oam(path: str | Path) -> bool:
    """Recognize the explicit OAM identity retained by packaged archives.

    NequIP 0.15 packages do not copy their model name into the instantiated
    ``GraphModel`` metadata.  The official archives do retain it in both the
    public filename and the package's root directory, so inspect those names
    without treating the shared ConvNetLayer/InteractionBlock graph as an OAM
    marker.
    """

    def declares(name: str) -> bool:
        return re.search(r"(?:^|[^a-z0-9])oam(?:[^a-z0-9]|$)", name.lower()) is not None

    source = Path(path)
    if declares(source.name):
        return True
    if not zipfile.is_zipfile(source):
        return False
    with zipfile.ZipFile(source) as archive:
        return any(
            declares(name.partition("/")[0]) for name in archive.namelist() if name
        )


def extract_nequip_data(
    model: Any, *, output_precision: str = "float64"
) -> dict[str, Any]:
    """Extract a supported NequIP OAM model into the v2 contract."""
    source_declares_oam = isinstance(model, (str, Path)) and (
        _source_declares_nequip_oam(model)
    )
    if isinstance(model, (str, Path)):
        model = load_nequip_model(model)
    if source_declares_oam or looks_like_nequip_oam_model(model):
        return extract_nequip_oam_data(model, output_precision=output_precision)
    if has_nequip_oam_hint(model):
        try:
            return extract_nequip_oam_data(model, output_precision=output_precision)
        except (NequIPProbeError, ValueError):
            pass
    raise ValueError("ordinary NequIP v1 contracts are no longer supported")


def export_nequip_model(
    model_path: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
    precision: str = "float64",
) -> Path:
    """Load and export a NequIP package/checkpoint to JSON.

    ``precision`` records the precision the exported contract is *certified
    for* (``kokkos_radial.precision`` and ``model_metadata.output_precision``).
    It does not quantize any coefficient: the runtime always builds the radial
    spline in double and casts on load, and a float64 export is accepted by a
    float32 evaluator.
    """
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")
    destination = Path(output_path)
    if destination.exists() and not force:
        raise FileExistsError(
            f"output already exists: {destination}; pass --force to overwrite"
        )
    data = extract_nequip_data(model_path, output_precision=precision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


__all__ = ["extract_nequip_data", "extract_nequip_oam_data", "export_nequip_model"]
