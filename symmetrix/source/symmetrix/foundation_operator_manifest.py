"""Verified foundation-model operator bundles for package-time generation.

This module is deliberately build-safe: it and the canonical contract/identity
modules it imports use only the Python standard library.  Backend target,
precision, generation revision, and schedule remain explicit build-policy
inputs and are therefore not recorded in the source manifest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .generated_operator_core import (
    generated_m0_identity,
    generated_m1_identity,
    generated_r0_identity,
)
from .mace_execution_contract import (
    make_generated_m1_contract,
    make_standard_m0_contract,
    make_standard_r0_contract,
    normalize_jit_r1_contract,
)

FOUNDATION_OPERATOR_MANIFEST_SCHEMA = "symmetrix.foundation-operator-manifest/1"
FOUNDATION_OPERATOR_MANIFEST_PATH = Path(__file__).with_name(
    "foundation_operator_manifest.json"
)
FOUNDATION_M0_TOPOLOGY_EVIDENCE_PATH = Path(__file__).with_name(
    "foundation_m0_topologies.json"
)
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TOP_FIELDS = frozenset(
    {"schema", "version", "r0_artifact_inputs", "bundles", "models", "aliases"}
)
_R0_INPUT_FIELDS = frozenset(
    {"id", "channels", "radial_embedding", "edge_l_max", "edge_weight_mode"}
)
_R1_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "version",
        "interaction",
        "channels",
        "radial_embedding",
        "edge_l_max",
        "source_l_max",
        "connection_mode",
        "path_count",
        "term_count",
        "fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
        "contract",
    }
)
_M0_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "version",
        "interaction",
        "channels",
        "input_l_max",
        "output_l_max",
        "correlation",
        "channel_mode",
        "type_count_mode",
        "execution_profile",
        "derivative_signature",
        "term_count",
        "structure_fingerprint",
    }
)
_M0_TOPOLOGY_FIELDS = frozenset(
    {"id", "input_l_max", "output_l_max", "correlation", "monomial_groups"}
)
_VERIFIED_R0_INPUTS = {
    "r0-c96-e3-unity": (96, 64, 3, "unity"),
    "r0-c128-e3-unity": (128, 64, 3, "unity"),
    "r0-c224-e3-unity": (224, 64, 3, "unity"),
}
_VERIFIED_BUNDLES = {
    "off23-l0": ("r0-c96-e3-unity", 0),
    "off23-l2": ("r0-c224-e3-unity", 2),
    "standard-l0": ("r0-c128-e3-unity", 0),
    "standard-l1": ("r0-c128-e3-unity", 1),
    "standard-l2": ("r0-c128-e3-unity", 2),
}
_VERIFIED_MODELS = {
    "mace-mp-0b2-large": "standard-l2",
    "mace-mpa-0-medium": "standard-l1",
    "mace-off23-large": "off23-l2",
    "mace-off23-small": "off23-l0",
    "mace-omat-0-small": "standard-l0",
}
_VERIFIED_ALIASES = {
    "mace-mp-0b3-medium": "mace-mpa-0-medium",
    "mace-omat-0-medium": "mace-mpa-0-medium",
}
_VERIFIED_M0_TOPOLOGY_IDS = frozenset({"m0-e3-l0-c3", "m0-e3-l1-c3", "m0-e3-l2-c3"})


def _exact_fields(value: Any, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")  # noqa: TRY004
    missing = expected - value.keys()
    if missing:
        raise ValueError(f"{name} is missing {', '.join(sorted(missing))}")
    unexpected = value.keys() - expected
    if unexpected:
        raise ValueError(f"{name} has unexpected {', '.join(sorted(unexpected))}")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")  # noqa: TRY004
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty normalized text")
    return value


def _identifier(value: Any, name: str) -> str:
    value = _text(value, name)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} is not a normalized identifier")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a complete SHA-256 fingerprint")
    return value


def _unique_ids(rows: list[dict[str, Any]], name: str) -> None:
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} IDs must be unique")


def _require_verified_ids(
    rows: list[dict[str, Any]], expected: set[str] | frozenset[str], name: str
) -> None:
    actual = {row["id"] for row in rows}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{name} must contain the verified {name} IDs; "
            f"missing={missing}, extra={extra}"
        )


def _normalize_r0_input(value: Any) -> dict[str, Any]:
    value = _exact_fields(value, _R0_INPUT_FIELDS, "R0 artifact input")
    edge_weight_mode = value["edge_weight_mode"]
    if edge_weight_mode != "unity":
        raise ValueError("foundation R0 edge_weight_mode must be 'unity'")
    return {
        "id": _identifier(value["id"], "R0 artifact input id"),
        "channels": _integer(value["channels"], "R0 channels", minimum=1),
        "radial_embedding": _integer(
            value["radial_embedding"], "R0 radial_embedding", minimum=1
        ),
        "edge_l_max": _integer(value["edge_l_max"], "R0 edge_l_max"),
        "edge_weight_mode": edge_weight_mode,
    }


def _r0_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    channels = value["channels"]
    l_max = value["edge_l_max"]
    angular_momenta = list(range(l_max + 1))
    edge_irreps = "+".join(
        f"1x{l_value}{'e' if l_value % 2 == 0 else 'o'}" for l_value in angular_momenta
    )
    output_irreps = "+".join(
        f"{channels}x{l_value}{'e' if l_value % 2 == 0 else 'o'}"
        for l_value in angular_momenta
    )
    return make_standard_r0_contract(
        channels=channels,
        radial_embedding=value["radial_embedding"],
        edge_l_max=l_max,
        source_irreps=f"{channels}x0e",
        edge_irreps=edge_irreps,
        output_irreps=output_irreps,
        output_l=angular_momenta,
        edge_l=angular_momenta,
        source_l=[0] * len(angular_momenta),
        instructions=[
            {
                "connection_mode": "uvu",
                "has_weight": True,
                "path_shape": [channels, 1],
            }
            for _ in angular_momenta
        ],
    )


def _normalize_r1_descriptor(value: Any) -> dict[str, Any]:
    value = _exact_fields(value, _R1_DESCRIPTOR_FIELDS, "R1 descriptor")
    contract = normalize_jit_r1_contract(value["contract"])
    if value["schema"] != "symmetrix.execution.tensor_product":
        raise ValueError("R1 descriptor schema is unsupported")
    version = _integer(value["version"], "R1 descriptor version", minimum=1)
    if version != 1 or value["interaction"] != "R1":
        raise ValueError("R1 descriptor version or interaction is unsupported")
    if value["connection_mode"] != "uvu":
        raise ValueError("foundation R1 connection_mode must be 'uvu'")
    descriptor = {
        "schema": value["schema"],
        "version": version,
        "interaction": value["interaction"],
        "channels": _integer(value["channels"], "R1 channels", minimum=1),
        "radial_embedding": _integer(
            value["radial_embedding"], "R1 radial_embedding", minimum=1
        ),
        "edge_l_max": _integer(value["edge_l_max"], "R1 edge_l_max"),
        "source_l_max": _integer(value["source_l_max"], "R1 source_l_max"),
        "connection_mode": value["connection_mode"],
        "path_count": _integer(value["path_count"], "R1 path_count", minimum=1),
        "term_count": _integer(value["term_count"], "R1 term_count", minimum=1),
        "fingerprint": _fingerprint(value["fingerprint"], "R1 fingerprint"),
        "generation_fingerprint": _fingerprint(
            value["generation_fingerprint"], "R1 generation_fingerprint"
        ),
        "structure_fingerprint": _fingerprint(
            value["structure_fingerprint"], "R1 structure_fingerprint"
        ),
    }
    if descriptor != r1_operator_descriptor(contract):
        raise ValueError("R1 descriptor does not match its canonical contract")
    return {**descriptor, "contract": contract}


def r1_operator_descriptor(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized bundle descriptor for one complete R1 contract."""

    contract = normalize_jit_r1_contract(contract)
    modes = {path["connection_mode"] for path in contract["paths"]}
    if len(modes) != 1:
        raise ValueError("R1 descriptor requires one connection mode")
    return {
        "schema": contract["schema"],
        "version": contract["version"],
        "interaction": contract["interaction"],
        "channels": contract["channels"],
        "radial_embedding": contract["radial_embedding"],
        "edge_l_max": contract["edge_harmonics"]["l_max"],
        "source_l_max": contract["source_harmonics"]["l_max"],
        "connection_mode": modes.pop(),
        "path_count": len(contract["paths"]),
        "term_count": contract["sparse_coupling"]["term_count"],
        "fingerprint": contract["fingerprint"],
        "generation_fingerprint": contract["generation_fingerprint"],
        "structure_fingerprint": contract["structure_fingerprint"],
    }


def _normalize_m0_descriptor(value: Any) -> dict[str, Any]:
    value = _exact_fields(value, _M0_DESCRIPTOR_FIELDS, "M0 descriptor")
    if value["schema"] != "symmetrix.execution.symmetric_contraction":
        raise ValueError("M0 descriptor schema is unsupported")
    version = _integer(value["version"], "M0 descriptor version", minimum=1)
    if version != 1 or value["interaction"] != "M0":
        raise ValueError("M0 descriptor version or interaction is unsupported")
    expected_text = {
        "channel_mode": "uvu",
        "type_count_mode": "runtime",
        "execution_profile": "fixed_weight_coordinate",
        "derivative_signature": "fixed_weight_coordinate",
    }
    for field, expected in expected_text.items():
        if value[field] != expected:
            raise ValueError(f"M0 descriptor {field} must be {expected!r}")
    return {
        "schema": value["schema"],
        "version": version,
        "interaction": value["interaction"],
        "channels": _integer(value["channels"], "M0 channels", minimum=1),
        "input_l_max": _integer(value["input_l_max"], "M0 input_l_max"),
        "output_l_max": _integer(value["output_l_max"], "M0 output_l_max"),
        "correlation": _integer(value["correlation"], "M0 correlation", minimum=1),
        "term_count": _integer(value["term_count"], "M0 term_count", minimum=1),
        "structure_fingerprint": _fingerprint(
            value["structure_fingerprint"], "M0 structure_fingerprint"
        ),
        **expected_text,
    }


def _normalize_bundle(value: Any) -> dict[str, Any]:
    value = _exact_fields(value, frozenset({"id", "operators"}), "bundle")
    operators = _exact_fields(
        value["operators"], frozenset({"R0", "R1", "M0"}), "bundle operators"
    )
    r0 = _exact_fields(
        operators["R0"], frozenset({"artifact_input"}), "bundle R0 descriptor"
    )
    return {
        "id": _identifier(value["id"], "bundle id"),
        "operators": {
            "R0": {
                "artifact_input": _identifier(
                    r0["artifact_input"], "bundle R0 artifact_input"
                )
            },
            "R1": _normalize_r1_descriptor(operators["R1"]),
            "M0": _normalize_m0_descriptor(operators["M0"]),
        },
    }


def _normalize_model(value: Any) -> dict[str, str]:
    value = _exact_fields(
        value, frozenset({"id", "provenance", "head", "bundle"}), "model"
    )
    return {
        "id": _identifier(value["id"], "model id"),
        "provenance": _text(value["provenance"], "model provenance"),
        "head": _identifier(value["head"], "model head"),
        "bundle": _identifier(value["bundle"], "model bundle"),
    }


def _normalize_alias(value: Any) -> dict[str, str]:
    value = _exact_fields(value, frozenset({"id", "target"}), "alias")
    return {
        "id": _identifier(value["id"], "alias id"),
        "target": _identifier(value["target"], "alias target"),
    }


def _load_canonical_m0_topologies() -> dict[tuple[int, int, int], dict[str, Any]]:
    try:
        payload = json.loads(
            FOUNDATION_M0_TOPOLOGY_EVIDENCE_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load canonical M0 topology evidence: {exc}") from exc
    rows = _list(payload, "canonical M0 topology evidence")
    normalized_rows = []
    contracts = {}
    for raw in rows:
        raw = _exact_fields(raw, _M0_TOPOLOGY_FIELDS, "canonical M0 topology")
        topology_id = _identifier(raw["id"], "canonical M0 topology id")
        input_l_max = _integer(raw["input_l_max"], "canonical M0 input_l_max")
        output_l_max = _integer(raw["output_l_max"], "canonical M0 output_l_max")
        correlation = _integer(
            raw["correlation"], "canonical M0 correlation", minimum=1
        )
        groups = _list(raw["monomial_groups"], "canonical M0 monomial_groups")
        expected_groups = (output_l_max + 1) ** 2
        if len(groups) != expected_groups:
            raise ValueError(
                "canonical M0 topology evidence has the wrong output component count"
            )
        monomials = {}
        for output_component, group in enumerate(groups):
            terms = _list(group, "canonical M0 monomial group")
            if not terms:
                raise ValueError("canonical M0 monomial groups must be nonempty")
            monomials[output_component] = [
                [
                    _integer(component, "canonical M0 monomial component")
                    for component in _list(term, "canonical M0 monomial")
                ]
                for term in terms
            ]
        contract = make_standard_m0_contract(
            channels=1,
            type_count=1,
            input_l_max=input_l_max,
            output_l_max=output_l_max,
            correlation=correlation,
            monomials=monomials,
        )
        key = (input_l_max, output_l_max, correlation)
        if key in contracts:
            raise ValueError("canonical M0 topology evidence must be unique")
        contracts[key] = {
            "id": topology_id,
            "term_count": sum(
                group["term_count"] for group in contract["monomial_groups"]
            ),
            "structure_fingerprint": contract["structure_fingerprint"],
            "contract": contract,
        }
        normalized_rows.append({"id": topology_id})
    _unique_ids(normalized_rows, "canonical M0 topology")
    _require_verified_ids(
        normalized_rows,
        _VERIFIED_M0_TOPOLOGY_IDS,
        "canonical M0 topology",
    )
    return contracts


def normalize_foundation_operator_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and deterministically normalize the production source data."""

    manifest = _exact_fields(manifest, _TOP_FIELDS, "foundation manifest")
    if manifest["schema"] != FOUNDATION_OPERATOR_MANIFEST_SCHEMA:
        raise ValueError("foundation manifest schema is unsupported")
    version = _integer(manifest["version"], "foundation manifest version", minimum=1)
    if version != 1:
        raise ValueError("foundation manifest version is unsupported")

    r0_inputs = [
        _normalize_r0_input(value)
        for value in _list(manifest["r0_artifact_inputs"], "r0_artifact_inputs")
    ]
    bundles = [
        _normalize_bundle(value) for value in _list(manifest["bundles"], "bundles")
    ]
    models = [_normalize_model(value) for value in _list(manifest["models"], "models")]
    aliases = [
        _normalize_alias(value) for value in _list(manifest["aliases"], "aliases")
    ]
    for rows, name in (
        (r0_inputs, "R0 artifact input"),
        (bundles, "bundle"),
        (models, "model"),
        (aliases, "alias"),
    ):
        _unique_ids(rows, name)
    for rows, expected, name in (
        (r0_inputs, set(_VERIFIED_R0_INPUTS), "R0 artifact input"),
        (bundles, set(_VERIFIED_BUNDLES), "bundle"),
        (models, set(_VERIFIED_MODELS), "model"),
        (aliases, set(_VERIFIED_ALIASES), "alias"),
    ):
        _require_verified_ids(rows, expected, name)

    for r0 in r0_inputs:
        expected = _VERIFIED_R0_INPUTS[r0["id"]]
        actual = (
            r0["channels"],
            r0["radial_embedding"],
            r0["edge_l_max"],
            r0["edge_weight_mode"],
        )
        if actual != expected:
            raise ValueError(f"R0 artifact input {r0['id']!r} is not verified")

    model_ids = {model["id"] for model in models}
    alias_ids = {alias["id"] for alias in aliases}
    if model_ids & alias_ids:
        raise ValueError("model and alias IDs must be disjoint")
    for alias in aliases:
        if alias["target"] not in model_ids:
            raise ValueError(f"alias targets unknown model {alias['target']!r}")
        if alias["target"] != _VERIFIED_ALIASES[alias["id"]]:
            raise ValueError(f"alias {alias['id']!r} does not match verified target")

    bundle_by_id = {bundle["id"]: bundle for bundle in bundles}
    for model in models:
        if model["bundle"] not in bundle_by_id:
            raise ValueError(f"model references unknown bundle {model['bundle']!r}")
        if model["bundle"] != _VERIFIED_MODELS[model["id"]]:
            raise ValueError(f"model {model['id']!r} does not match verified bundle")
    referenced_bundles = {model["bundle"] for model in models}
    if referenced_bundles != set(bundle_by_id):
        unknown = sorted(set(bundle_by_id) - referenced_bundles)
        raise ValueError(f"foundation manifest has unverified bundles: {unknown}")

    r0_by_id = {value["id"]: value for value in r0_inputs}
    canonical_m0 = _load_canonical_m0_topologies()
    referenced_r0 = set()
    for bundle in bundles:
        operators = bundle["operators"]
        r0_id = operators["R0"]["artifact_input"]
        if r0_id not in r0_by_id:
            raise ValueError(f"bundle references unknown R0 artifact input {r0_id!r}")
        referenced_r0.add(r0_id)
        r0 = r0_by_id[r0_id]
        r1 = operators["R1"]
        m0 = operators["M0"]
        verified_r0, verified_output_l_max = _VERIFIED_BUNDLES[bundle["id"]]
        if r0_id != verified_r0 or r1["source_l_max"] != verified_output_l_max:
            raise ValueError(f"bundle {bundle['id']!r} is not a verified architecture")
        if r1["channels"] != r0["channels"] or m0["channels"] != r0["channels"]:
            raise ValueError(f"bundle {bundle['id']!r} operator channels disagree")
        if r1["radial_embedding"] != r0["radial_embedding"]:
            raise ValueError(f"bundle {bundle['id']!r} radial embeddings disagree")
        if r1["edge_l_max"] != r0["edge_l_max"]:
            raise ValueError(f"bundle {bundle['id']!r} edge angular cutoffs disagree")
        if m0["input_l_max"] != r0["edge_l_max"]:
            raise ValueError(f"bundle {bundle['id']!r} M0 input cutoff disagrees")
        if m0["output_l_max"] != r1["source_l_max"]:
            raise ValueError(f"bundle {bundle['id']!r} M0/R1 angular cutoffs disagree")
        m0_evidence = canonical_m0.get(
            (m0["input_l_max"], m0["output_l_max"], m0["correlation"])
        )
        if m0_evidence is None or any(
            m0[field] != m0_evidence[field]
            for field in ("term_count", "structure_fingerprint")
        ):
            raise ValueError(
                f"bundle {bundle['id']!r} does not match canonical M0 topology"
            )
    if referenced_r0 != set(r0_by_id):
        unknown = sorted(set(r0_by_id) - referenced_r0)
        raise ValueError(f"foundation manifest has unverified R0 inputs: {unknown}")

    return {
        "schema": FOUNDATION_OPERATOR_MANIFEST_SCHEMA,
        "version": 1,
        "r0_artifact_inputs": sorted(r0_inputs, key=lambda value: value["id"]),
        "bundles": sorted(bundles, key=lambda value: value["id"]),
        "models": sorted(models, key=lambda value: value["id"]),
        "aliases": sorted(aliases, key=lambda value: value["id"]),
    }


def load_foundation_operator_manifest(
    path: Path | str = FOUNDATION_OPERATOR_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load the checked-in production manifest or an explicit candidate."""

    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load foundation operator manifest {path}: {exc}"
        ) from exc
    return normalize_foundation_operator_manifest(value)


def foundation_r0_artifact_inputs(
    manifest: Mapping[str, Any],
    *,
    generation_revision: int,
    compatibility_version: int,
    backend: str,
    target: str,
    precisions: Iterable[str],
    schedule: str,
    schedule_revision: int,
    floating_point_policy: str | None = None,
) -> list[dict[str, Any]]:
    """Return canonical, identity-deduplicated R0 inputs for one build target."""

    manifest = normalize_foundation_operator_manifest(manifest)
    if isinstance(precisions, (str, bytes)):
        raise ValueError(  # noqa: TRY004
            "precisions must be an iterable of precision names"
        )
    try:
        precision_values = sorted(set(precisions))
    except TypeError as exc:
        raise ValueError("precisions must be an iterable of precision names") from exc
    if not precision_values:
        raise ValueError("precisions must not be empty")

    r0_by_id = {value["id"]: value for value in manifest["r0_artifact_inputs"]}
    requests: dict[str, dict[str, Any]] = {}
    for bundle in manifest["bundles"]:
        input_id = bundle["operators"]["R0"]["artifact_input"]
        descriptor = r0_by_id[input_id]
        contract = _r0_contract(descriptor)
        for precision in precision_values:
            identity = generated_r0_identity(
                generation_revision=generation_revision,
                compatibility_version=compatibility_version,
                contract_fingerprint=contract["generation_fingerprint"],
                structure_fingerprint=contract["structure_fingerprint"],
                precision=precision,
                backend=backend,
                target=target,
                channels=contract["channels"],
                l_max=contract["edge_harmonics"]["l_max"],
                edge_weight_mode=descriptor["edge_weight_mode"],
                floating_point_policy=floating_point_policy,
                schedule=schedule,
                schedule_revision=schedule_revision,
            )
            artifact_id = identity["artifact_id"]
            request = requests.setdefault(
                artifact_id,
                {
                    "identity": identity,
                    "contract": contract,
                    "artifact_input_ids": set(),
                    "bundle_ids": set(),
                },
            )
            request["artifact_input_ids"].add(input_id)
            request["bundle_ids"].add(bundle["id"])

    result = []
    for artifact_id in sorted(requests):
        request = requests[artifact_id]
        result.append(
            {
                "identity": request["identity"],
                "contract": request["contract"],
                "artifact_input_ids": sorted(request["artifact_input_ids"]),
                "bundle_ids": sorted(request["bundle_ids"]),
            }
        )
    return result


def foundation_m0_artifact_inputs(
    manifest: Mapping[str, Any],
    *,
    generation_revision: int,
    backend: str,
    target: str,
    precisions: Iterable[str],
    schedule: str = "chunk32",
) -> list[dict[str, Any]]:
    """Return canonical, identity-deduplicated M0 inputs for one build target."""

    manifest = normalize_foundation_operator_manifest(manifest)
    if isinstance(precisions, (str, bytes)):
        raise ValueError("precisions must be an iterable of precision names")  # noqa: TRY004
    precision_values = sorted(set(precisions))
    if not precision_values:
        raise ValueError("precisions must not be empty")
    topologies = _load_canonical_m0_topologies()
    requests: dict[str, dict[str, Any]] = {}
    for bundle in manifest["bundles"]:
        descriptor = bundle["operators"]["M0"]
        topology = topologies[
            (
                descriptor["input_l_max"],
                descriptor["output_l_max"],
                descriptor["correlation"],
            )
        ]["contract"]
        contract = make_standard_m0_contract(
            channels=descriptor["channels"],
            type_count=1,
            input_l_max=descriptor["input_l_max"],
            output_l_max=descriptor["output_l_max"],
            correlation=descriptor["correlation"],
            monomials={
                group["output_component"]: group["terms"]
                for group in topology["monomial_groups"]
            },
        )
        for precision in precision_values:
            identity = generated_m0_identity(
                generation_revision=generation_revision,
                structure_fingerprint=contract["structure_fingerprint"],
                precision=precision,
                backend=backend,
                target=target,
                channels=contract["channels"],
                input_components=contract["input_components"],
                output_components=contract["output_components"],
                correlation=contract["correlation"],
                term_count=sum(
                    group["term_count"] for group in contract["monomial_groups"]
                ),
                schedule=schedule,
                schedule_revision=(
                    3
                    if schedule == "tiled128x4-v1"
                    else 2
                    if schedule == "chunk32"
                    else 1
                ),
            )
            request = requests.setdefault(
                identity["artifact_id"],
                {"identity": identity, "contract": contract, "bundle_ids": set()},
            )
            request["bundle_ids"].add(bundle["id"])
    return [
        {**requests[key], "bundle_ids": sorted(requests[key]["bundle_ids"])}
        for key in sorted(requests)
    ]


def foundation_m1_artifact_inputs(
    manifest: Mapping[str, Any],
    *,
    generation_revision: int,
    backend: str,
    target: str,
    precisions: Iterable[str],
    schedule: str = "chunk32",
    schedule_revision: int | None = None,
) -> list[dict[str, Any]]:
    """Return deduplicated scalar M1 inputs for known foundation models."""

    manifest = normalize_foundation_operator_manifest(manifest)
    if backend == "host":
        if schedule != "chunk32":
            raise ValueError("host M1 artifacts support only the chunk32 schedule")
        expected_schedule_revision = 2
    elif backend in ("cuda", "hip"):
        try:
            expected_schedule_revision = {
                "table": 2,
                "chunk32": 3,
                "tiled128x4-v1": 4,
            }[schedule]
        except KeyError as error:
            raise ValueError(f"unsupported device M1 schedule: {schedule!r}") from error
    else:
        raise ValueError(f"unsupported M1 artifact backend: {backend!r}")
    if schedule_revision is None:
        schedule_revision = expected_schedule_revision
    elif schedule_revision != expected_schedule_revision:
        raise ValueError(
            f"M1 {backend} {schedule} requires schedule revision "
            f"{expected_schedule_revision}, got {schedule_revision}"
        )
    if isinstance(precisions, (str, bytes)):
        raise ValueError("precisions must be an iterable of precision names")  # noqa: TRY004
    precision_values = sorted(set(precisions))
    if not precision_values:
        raise ValueError("precisions must not be empty")
    topologies = _load_canonical_m0_topologies()
    requests: dict[str, dict[str, Any]] = {}
    for bundle in manifest["bundles"]:
        descriptor = bundle["operators"]["M0"]
        topology = topologies[
            (descriptor["input_l_max"], 0, descriptor["correlation"])
        ]["contract"]
        scalar_terms = topology["monomial_groups"][0]["terms"]
        contract = make_generated_m1_contract(
            channels=descriptor["channels"],
            type_count=1,
            input_l_max=descriptor["input_l_max"],
            correlation=descriptor["correlation"],
            monomials=scalar_terms,
        )
        for precision in precision_values:
            identity = generated_m1_identity(
                generation_revision=generation_revision,
                structure_fingerprint=contract["structure_fingerprint"],
                precision=precision,
                backend=backend,
                target=target,
                channels=contract["channels"],
                input_components=contract["input_components"],
                correlation=contract["correlation"],
                term_count=contract["monomial_groups"][0]["term_count"],
                schedule=schedule,
                schedule_revision=schedule_revision,
            )
            request = requests.setdefault(
                identity["artifact_id"],
                {"identity": identity, "contract": contract, "bundle_ids": set()},
            )
            request["bundle_ids"].add(bundle["id"])
    return [
        {**requests[key], "bundle_ids": sorted(requests[key]["bundle_ids"])}
        for key in sorted(requests)
    ]


__all__ = [
    "FOUNDATION_M0_TOPOLOGY_EVIDENCE_PATH",
    "FOUNDATION_OPERATOR_MANIFEST_PATH",
    "FOUNDATION_OPERATOR_MANIFEST_SCHEMA",
    "foundation_m0_artifact_inputs",
    "foundation_m1_artifact_inputs",
    "foundation_r0_artifact_inputs",
    "load_foundation_operator_manifest",
    "normalize_foundation_operator_manifest",
    "r1_operator_descriptor",
]
