"""Frontend schema for the PolarMACE family's backbone JSON.

The polar family's models are MH-1 backbones carrying the PolarMACE deltas, and
they are serialized by the family extractor
(:mod:`symmetrix.extract_mace_polar`) under the family's own model type --
:data:`POLAR_MODEL_TYPE`, the detection key the family registry admits the
family by.  This module owns that boundary and the *difference* between the two
families' schemas:

* the model type and the format version the polar family stamps,
* the three shape spellings the released checkpoints use and the MH-1 schema
  does not admit -- ``scale``/``shift`` serialized as a leading size-1 tensor
  (``[1]``) rather than a scalar, and the atomic-energy vector serialized as
  ``[num_elements]`` rather than ``[1, num_elements]``; the polar schema admits
  *both* spellings, so an MH-1-shaped backbone the family is asked to carry
  (the repository's own synthetic PolarMACE-shaped backbone) loads unchanged,
* the ``polar`` backbone-delta block and the embedded ``POLAR_BACKBONE``
  identity contract the extractor writes.

The remaining backbone structure -- atomic-number metadata, the radial basis
and its cutoff, the layer counts, the irreps -- is validated once, by the
native schema in ``libsymmetrix/source/mace_polar_schema.cpp``.  This module
deliberately does not restate it: two authorities over one file would drift,
and the native parse is the one that actually consumes the data.  What is
checked here is the family boundary plus the shape deltas, so a frontend caller
gets the polar-specific rejection in Python before the native parse is reached.

The native side mirrors this module's constants; ``test_mace_polar_schema.py``
pins the model type and the contract tag across the language boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .mace_polar_execution_contract import (
    POLAR_CONTRACT_KEY,
    POLAR_CONTRACT_TAG,
    POLAR_SCHEMA,
    POLAR_SCHEMA_VERSION,
)

__all__ = [
    "POLAR_CONTRACT_KEY",
    "POLAR_CONTRACT_TAG",
    "POLAR_FORMAT_VERSION",
    "POLAR_MODEL_TYPE",
    "POLAR_SCHEMA",
    "POLAR_SCHEMA_VERSION",
    "polar_scale_shift_shapes",
    "validate_mace_polar_model_data",
]

#: The model type a PolarMACE model file declares.  It is the family registry's
#: detection key for the family, so an MH-1 file can never be admitted to (or
#: captured by) the polar family and vice versa.
POLAR_MODEL_TYPE = "MACE_Polar"

#: The nonlinear backbone schema version the polar family serializes.  The
#: polar deltas ride the shared version-3 nonlinear schema; the family is
#: distinguished by its model type and its embedded contract, not by a second
#: version number.
POLAR_FORMAT_VERSION = 3

#: The shapes the polar family admits for ``scale_shift``: a scalar tensor (the
#: MH-1 spelling and the one a synthetic backbone carries) and the released
#: checkpoints' leading size-1 tensor.
_POLAR_SCALE_SHIFT_SHAPES = ((), (1,))


def polar_scale_shift_shapes() -> tuple[tuple[int, ...], ...]:
    """The ``scale``/``shift`` shapes the polar schema admits."""

    return _POLAR_SCALE_SHIFT_SHAPES


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _admits_shape(shape: tuple[int, ...], allowed: tuple[tuple[int, ...], ...]) -> bool:
    """Report whether ``shape`` is one of ``allowed`` or its leading size-1 form.

    A released checkpoint serializes a scalar as a leading size-1 tensor, so the
    polar schema admits both spellings of every shape it admits; a synthetic
    backbone the family is asked to carry keeps the scalar form.
    """

    for candidate in allowed:
        if shape == candidate:
            return True
        if (
            len(shape) == len(candidate) + 1
            and shape[0] == 1
            and shape[1:] == candidate
        ):
            return True
    return False


def _tensor_shape(
    parent: Mapping[str, Any], name: str, allowed: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    tensor = _require_mapping(parent.get(name), f"scale_shift.{name}")
    shape = tensor.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(extent, int) and not isinstance(extent, bool) for extent in shape
    ):
        raise ValueError(f"scale_shift.{name} tensor shape is invalid")
    resolved = tuple(shape)
    if not _admits_shape(resolved, allowed):
        admitted = ", ".join(str(list(candidate)) for candidate in allowed)
        raise ValueError(
            f"PolarMACE scale_shift.{name} shape {list(resolved)} is outside the "
            f"polar family's admitted spellings ({admitted}, each also admitted "
            "with a leading size-1 extent)"
        )
    return resolved


def _atomic_energy_extent(model_data: Mapping[str, Any]) -> int:
    """The element count the atomic-energy vector must carry."""

    model_numbers = model_data.get("model_atomic_numbers")
    if isinstance(model_numbers, list) and model_numbers:
        return len(model_numbers)
    return _require_int(model_data.get("num_elements"), "num_elements", 1)


def validate_mace_polar_model_data(model_data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate one polar model file's family boundary and shape spellings.

    Returns the mapping unchanged so a caller can validate and pass the same
    object on.  Raises :class:`ValueError` naming the first violation, the way
    the native polar schema reports it.
    """

    data = _require_mapping(model_data, "the polar model data")
    model_type = data.get("model_type")
    if model_type != POLAR_MODEL_TYPE:
        raise ValueError(
            f"the PolarMACE evaluator requires model_type {POLAR_MODEL_TYPE!r}; "
            f"got {model_type!r}"
        )
    version = data.get("symmetrix_format_version")
    if version != POLAR_FORMAT_VERSION:
        raise ValueError(
            "the PolarMACE evaluator requires "
            f"format-version-{POLAR_FORMAT_VERSION} nonlinear JSON"
        )

    scale_shift = _require_mapping(data.get("scale_shift"), "scale_shift")
    for name in ("scale", "shift"):
        _tensor_shape(scale_shift, name, _POLAR_SCALE_SHIFT_SHAPES)
        values = _require_mapping(scale_shift.get(name), f"scale_shift.{name}").get(
            "values"
        )
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(
                f"PolarMACE scale_shift.{name} must carry exactly one value"
            )
        _finite_number(values[0], f"scale_shift.{name}")

    energies = _require_mapping(data.get("atomic_energies"), "atomic_energies")
    shape = energies.get("shape")
    elements = _atomic_energy_extent(data)
    if shape not in ([elements], [1, elements]):
        raise ValueError(
            "PolarMACE atomic energies must be shaped "
            f"{[elements]} or {[1, elements]}; got {shape}"
        )

    polar = _require_mapping(data.get("polar"), "polar")
    if polar.get("keep_last_layer_irreps") is not True:
        raise ValueError(
            "PolarMACE extraction requires keep_last_layer_irreps backbone readouts"
        )
    interactions = data.get("interactions")
    if not isinstance(interactions, list) or not interactions:
        raise ValueError("the polar backbone requires at least one interaction")
    for section in ("charges_irreps", "field_irreps", "potential_irreps"):
        if not isinstance(polar.get(section), str) or not polar[section]:
            raise ValueError(f"polar.{section} must be a non-empty irreps string")
    if _finite_number(polar.get("kspace_cutoff"), "polar.kspace_cutoff") <= 0.0:
        raise ValueError("polar.kspace_cutoff must be positive")
    recursion_steps = _require_int(
        polar.get("num_recursion_steps"), "polar.num_recursion_steps", 1
    )
    heads = polar.get("spin_source_heads")
    if not isinstance(heads, list) or len(heads) != len(interactions):
        raise ValueError("polar.spin_source_heads must match the interaction count")
    mixer = _require_mapping(polar.get("feature_mixer"), "polar.feature_mixer")
    mixer_linears = mixer.get("linears")
    if not isinstance(mixer_linears, list) or len(mixer_linears) != len(interactions):
        raise ValueError("polar.feature_mixer.linears must match the interaction count")
    updates = polar.get("field_update_blocks")
    if not isinstance(updates, list) or len(updates) != recursion_steps:
        raise ValueError(
            "polar.field_update_blocks must match polar.num_recursion_steps"
        )

    contracts = data.get("execution_contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError(
            "the PolarMACE model file must carry its execution contracts; "
            f"{POLAR_CONTRACT_KEY} is missing"
        )
    polar_contract = contracts.get(POLAR_CONTRACT_KEY)
    if not isinstance(polar_contract, Mapping):
        raise ValueError(
            f"the PolarMACE model file must carry the {POLAR_CONTRACT_KEY} contract"
        )
    if polar_contract.get("tag") != POLAR_CONTRACT_TAG:
        raise ValueError(
            f"the {POLAR_CONTRACT_KEY} contract must be tagged "
            f"{POLAR_CONTRACT_TAG!r}; got {polar_contract.get('tag')!r}"
        )
    return data
