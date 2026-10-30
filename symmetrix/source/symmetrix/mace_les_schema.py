"""Validation for intrinsic MACELES model parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

MACE_LES_SCHEMA = "symmetrix.model.mace_les"
MACE_LES_SCHEMA_VERSION = 1

_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "version",
        "kind",
        "periodicity",
        "ewald",
        "output_scale",
        "mace_num_interactions",
        "mace_num_channels",
        "mace_contract_fingerprints",
        "readout_1_weights",
        "readout_2_hidden_size",
        "readout_2_weights_1",
        "readout_2_weights_2",
        "readout_2_scale_factor",
    }
)
_EWALD_FIELDS = frozenset({"sigma", "dl", "remove_self_interaction"})
_BASE_CONTRACT_FIELDS = frozenset({"R0", "M0", "R1"})


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or (positive and normalized <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return normalized


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _weights(value: object, name: str, count: int) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} must contain exactly {count} values")
    return [_finite_number(item, name) for item in value]


def validate_mace_les_parameters(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized, detached scalar-monopole MACELES parameters.

    This is intentionally a JSON-only contract: conversion may import MACE and
    LES, while loading and evaluating an extracted model must not.
    """
    if not isinstance(data, Mapping) or set(data) != _REQUIRED_FIELDS:
        raise ValueError("les has an unsupported field set")
    if data["schema"] != MACE_LES_SCHEMA:
        raise ValueError(f"les.schema must be {MACE_LES_SCHEMA!r}")
    if data["version"] != MACE_LES_SCHEMA_VERSION:
        raise ValueError(f"les.version must be {MACE_LES_SCHEMA_VERSION}")
    if data["kind"] != "monopole":
        raise ValueError("les.kind must be 'monopole'")
    if data["periodicity"] != "geometry":
        raise ValueError("les.periodicity must be 'geometry'")

    ewald = data["ewald"]
    if not isinstance(ewald, Mapping) or set(ewald) != _EWALD_FIELDS:
        raise ValueError("les.ewald has an unsupported field set")
    if not isinstance(ewald["remove_self_interaction"], bool):
        raise TypeError("les.ewald.remove_self_interaction must be boolean")

    channels = _positive_int(data["mace_num_channels"], "mace_num_channels")
    hidden = _positive_int(data["readout_2_hidden_size"], "readout_2_hidden_size")
    if data["mace_num_interactions"] != 2:
        raise ValueError("mace_num_interactions must be 2 for monopole LES")
    fingerprints = data["mace_contract_fingerprints"]
    if (
        not isinstance(fingerprints, Mapping)
        or set(fingerprints) != _BASE_CONTRACT_FIELDS
    ):
        raise ValueError("mace_contract_fingerprints must define R0, M0, and R1")
    normalized_fingerprints = {}
    for name in ("R0", "M0", "R1"):
        value = fingerprints[name]
        if (
            not isinstance(value, str)
            or len(value) != 71
            or not value.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[7:])
        ):
            raise ValueError(
                f"mace_contract_fingerprints.{name} must be a sha256 fingerprint"
            )
        normalized_fingerprints[name] = value

    return {
        "schema": MACE_LES_SCHEMA,
        "version": MACE_LES_SCHEMA_VERSION,
        "kind": "monopole",
        "periodicity": "geometry",
        "ewald": {
            "sigma": _finite_number(ewald["sigma"], "ewald.sigma", positive=True),
            "dl": _finite_number(ewald["dl"], "ewald.dl", positive=True),
            "remove_self_interaction": ewald["remove_self_interaction"],
        },
        "output_scale": _finite_number(data["output_scale"], "output_scale"),
        "mace_num_interactions": 2,
        "mace_num_channels": channels,
        "mace_contract_fingerprints": normalized_fingerprints,
        "readout_1_weights": _weights(
            data["readout_1_weights"], "readout_1_weights", channels
        ),
        "readout_2_hidden_size": hidden,
        "readout_2_weights_1": _weights(
            data["readout_2_weights_1"],
            "readout_2_weights_1",
            channels * hidden,
        ),
        "readout_2_weights_2": _weights(
            data["readout_2_weights_2"], "readout_2_weights_2", hidden
        ),
        "readout_2_scale_factor": _finite_number(
            data["readout_2_scale_factor"], "readout_2_scale_factor"
        ),
    }


__all__ = [
    "MACE_LES_SCHEMA",
    "MACE_LES_SCHEMA_VERSION",
    "validate_mace_les_parameters",
]
