"""Validation and selection for multi-head Symmetrix model JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PREDICTION_HEADS_SCHEMA_VERSION = 1


def _prediction_head_fields(model_data: Mapping[str, Any]) -> set[str]:
    if model_data.get("model_type") == "MACE_Nonlinear":
        return {"atomic_energies", "scale_shift", "readouts"}
    fields = {
        "atomic_energies",
        "readout_1_weights",
        "readout_2_weights_1",
        "readout_2_weights_2",
    }
    if model_data.get("has_zbl", False):
        fields.add("zbl_c")
    return fields


def available_prediction_heads(model_data: Mapping[str, Any]) -> tuple[str, ...]:
    section = model_data.get("prediction_heads")
    if section is None:
        legacy_head = model_data.get("head")
        return (legacy_head,) if isinstance(legacy_head, str) and legacy_head else ()
    if not isinstance(section, Mapping):
        raise ValueError("prediction_heads must be an object")
    if section.get("schema_version") != PREDICTION_HEADS_SCHEMA_VERSION:
        raise ValueError("prediction_heads has an unsupported schema version")
    names = section.get("names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("prediction_heads.names must contain unique non-empty strings")
    parameters = section.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != set(names):
        raise ValueError("prediction_heads.parameters must define every named head")
    expected_fields = _prediction_head_fields(model_data)
    for name in names:
        payload = parameters[name]
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            raise ValueError(
                "every prediction-head payload must contain exactly the supported fields"
            )
    default = section.get("default")
    if default not in names:
        raise ValueError("prediction_heads.default must name a stored head")
    return tuple(names)


def select_prediction_head(
    model_data: Mapping[str, Any], head: str | None = None
) -> dict[str, Any]:
    """Return a shallow single-head projection suitable for native loading."""

    selected = dict(model_data)
    names = available_prediction_heads(model_data)
    section = model_data.get("prediction_heads")
    if section is None:
        legacy_head = names[0] if names else None
        if head is not None and head != legacy_head:
            if legacy_head is None:
                raise ValueError("this legacy JSON does not identify a prediction head")
            raise ValueError(
                f"prediction head {head!r} is unavailable; the JSON contains {legacy_head!r}"
            )
        selected["selected_head"] = legacy_head
        selected["available_heads"] = list(names)
        return selected

    default = section["default"]
    selected_name = default if head is None else head
    if selected_name not in names:
        raise ValueError(
            f"prediction head {selected_name!r} is unavailable; available heads are {list(names)!r}"
        )
    selected.update(section["parameters"][selected_name])
    selected["head"] = selected_name
    selected["selected_head"] = selected_name
    selected["available_heads"] = list(names)
    return selected
