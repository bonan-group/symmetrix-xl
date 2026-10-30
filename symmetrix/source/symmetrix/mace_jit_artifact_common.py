"""Cross-language artifact-preparation helpers shared by the R1 adapters.

Every Execution R1 artifact adapter -- the compact host adapter, its device
sibling, and the nonlinear MH-1 host adapter -- prepares its plugin from the
same compact model JSON and builds the same native Kokkos evaluator through
the same extension entry points.  Those pieces are the Python side of the
Python/native contract, so they live here once instead of once per adapter.

What differs between adapters stays with the adapter and is passed in: the
adapter's own error type (each adapter publishes a distinct exception so its
frontend can report its own target) and the streamed-edge policy the adapter's
contract requires (``"direct"`` for the compact host adapter, ``"factorized"``
for the device sibling).  Error text and control flow are otherwise identical
to what each adapter raised before this module existed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import family_registry, symmetrix
from .jit import JIT_GENERATION_VERSION
from .mace_les_schema import validate_mace_les_parameters


def mace_les_evaluator_identity(model_data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the model-driven MACELES evaluator identity.

    MACELES parameters belong to the evaluator rather than the generated
    R0/M0/R1 operators. Keeping their digest outside operator code generation
    lets structurally identical MACE and MACELES backbones share those artifacts
    while preventing prepared evaluator state from aliasing distinct models.
    """

    if not isinstance(model_data, Mapping):
        raise TypeError("model_data must be a mapping")
    model_type = model_data.get("model_type", "MACE")
    if not isinstance(model_type, str) or not model_type:
        raise TypeError("model_type must be a non-empty string")
    parameters_sha256 = None
    if model_type == "MACELES":
        parameters = model_data.get("les")
        if not isinstance(parameters, Mapping):
            raise TypeError("MACELES model les parameters must be a mapping")
        normalized = validate_mace_les_parameters(parameters)
        encoded_parameters = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        parameters_sha256 = hashlib.sha256(encoded_parameters).hexdigest()
    state = {
        "model_type": model_type,
        "mace_les_parameters_sha256": parameters_sha256,
    }
    encoded_state = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **state,
        "mace_les_evaluator_fingerprint": hashlib.sha256(encoded_state).hexdigest(),
    }


def read_model_json(
    model_file,
    *,
    error_type: Callable[..., Exception],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve ``model_file`` and read it as a Symmetrix model JSON object.

    ``label`` names the adapter's model family in the raised messages and
    ``error_type`` is the adapter's own exception class.
    """

    model_path = Path(model_file).expanduser().resolve()
    try:
        with model_path.open(encoding="utf-8") as stream:
            model_data = json.load(stream)
    except (OSError, TypeError, ValueError) as error:
        raise error_type(
            f"could not read {label} Symmetrix model JSON {model_path}: {error}"
        ) from error
    if not isinstance(model_data, dict):
        raise error_type(f"{label} Symmetrix model JSON must contain an object")
    return model_path, model_data


def native_evaluator(
    model_path: Path,
    precision: str,
    *,
    model_type: str,
    error_type: Callable[..., Exception],
    streamed_edges: str,
):
    """Construct the native Kokkos evaluator an R1 artifact is compiled against.

    ``streamed_edges`` is the streamed-edge policy that evaluator is put into,
    which the compiled artifact's contract must match.
    """

    if not isinstance(model_type, str) or not model_type:
        raise error_type("compact Symmetrix model_type must be a non-empty string")
    _, model_data = read_model_json(model_path, error_type=error_type, label="compact")
    document_model_type = model_data.get("model_type", "MACE")
    if not isinstance(document_model_type, str) or not document_model_type:
        raise error_type("compact Symmetrix model_type must be a non-empty string")
    if document_model_type != model_type:
        raise error_type(
            f"declared model_type {model_type!r} does not match model document "
            f"type {document_model_type!r}"
        )
    if family_registry.declared_registry().detect(model_type=model_type) is None:
        raise error_type(f"unsupported compact model_type {model_type!r}")
    try:
        evaluator_class = family_registry.resolve_model_evaluator_class(
            model_type=model_type,
            dtype=precision,
            native=symmetrix,
        )
    except ValueError as error:
        raise error_type(str(error)) from error
    except AttributeError as error:
        raise error_type(
            "this Symmetrix build does not provide the requested Kokkos evaluator"
        ) from error
    try:
        evaluator = evaluator_class(str(model_path))
        evaluator.set_streamed_edges(streamed_edges)
    except Exception as error:
        raise error_type(
            f"could not construct the native Execution R1 evaluator: {error}"
        ) from error
    return evaluator


def require_matching_generation_version(
    *,
    error_type: Callable[..., Exception],
) -> None:
    """Refuse an adapter whose native extension disagrees on the generation version."""

    query = getattr(symmetrix, "_required_jit_generation_version", None)
    if not callable(query):
        raise error_type(
            "the native Symmetrix extension predates JIT generation versioning"
        )
    required = query()
    if required != JIT_GENERATION_VERSION:
        raise error_type(
            "JIT generation version mismatch: native standard modules require "
            f"{required}, but Python generators provide {JIT_GENERATION_VERSION}"
        )
