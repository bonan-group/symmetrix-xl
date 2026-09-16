"""Canonical declarative contracts for Execution tensor products."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Iterable, Mapping


SCHEMA = "symmetrix.execution.tensor_product"
SCHEMA_VERSION = 1
R0_INTERACTION = "R0"
R1_INTERACTION = "R1"
INTERACTION = R1_INTERACTION
M0_SCHEMA = "symmetrix.execution.symmetric_contraction"
M0_SCHEMA_VERSION = 1
M0_INTERACTION = "M0"
M0_EXECUTION_PROFILE = "fixed_weight_coordinate"
M0_DERIVATIVE_SIGNATURE = "fixed_weight_coordinate"


def canonical_json(value: Any) -> str:
    """Serialize JSON data in the one representation used for fingerprints."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _fingerprint(value: Any) -> str:
    payload = canonical_json(value).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sequence_fingerprint(values: Iterable[Any]) -> str:
    return _fingerprint(list(values))


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _generation_profile(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": contract["schema"],
        "version": contract["version"],
        "interaction": contract["interaction"],
        "channels": contract["channels"],
        "radial_embedding": contract["radial_embedding"],
        "edge_l_max": contract["edge_harmonics"]["l_max"],
        "source_l_max": contract["source_harmonics"]["l_max"],
        "irreps": contract["irreps"],
        "groups": contract["groups"],
        "paths": contract["paths"],
        "sparse_coupling": contract["sparse_coupling"],
        "layouts": contract["layouts"],
        "derivatives": contract["derivatives"],
        "accumulator": contract["accumulator"],
    }


def _tensor_product_structure_profile(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the executable program topology, excluding runtime extents."""

    paths = []
    for path in contract["paths"]:
        paths.append(
            {
                "index": path["index"],
                "output_l": path["output_l"],
                "edge_l": path["edge_l"],
                "source_l": path["source_l"],
                "connection_mode": path["connection_mode"],
                "has_weight": path["has_weight"],
                "path_rank": len(path["path_shape"]),
            }
        )
    return {
        "schema": contract["schema"],
        "version": contract["version"],
        "interaction": contract["interaction"],
        "edge_l_max": contract["edge_harmonics"]["l_max"],
        "source_l_max": contract["source_harmonics"]["l_max"],
        "groups": contract["groups"],
        "paths": paths,
        "sparse_coupling": contract["sparse_coupling"],
        "layouts": contract["layouts"],
        "derivatives": contract["derivatives"],
        "accumulator": contract["accumulator"],
    }


def _normalize_execution_contract(
    contract: Mapping[str, Any], expected_interaction: str
) -> dict[str, Any]:
    """Validate and return a canonical, fingerprinted tensor-product contract."""

    normalized = deepcopy(dict(contract))
    supplied_generation = normalized.pop("generation_fingerprint", None)
    supplied_semantic = normalized.pop("fingerprint", None)
    # Structural fingerprints are derived dispatch metadata. Recompute them so
    # contracts cached under an earlier structural profile migrate in place.
    normalized.pop("structure_fingerprint", None)

    if normalized.get("schema") != SCHEMA:
        raise ValueError(f"Execution contract schema must be {SCHEMA!r}")
    if normalized.get("version") != SCHEMA_VERSION:
        raise ValueError(f"Execution contract version must be {SCHEMA_VERSION}")
    if normalized.get("interaction") != expected_interaction:
        raise ValueError(
            f"Execution contract interaction must be {expected_interaction!r}"
        )

    channels = _require_int(normalized.get("channels"), "channels", 1)
    radial_embedding = _require_int(
        normalized.get("radial_embedding"), "radial_embedding", 1
    )
    normalized["channels"] = channels
    normalized["radial_embedding"] = radial_embedding

    for name in ("edge_harmonics", "source_harmonics"):
        harmonics = normalized.get(name)
        if not isinstance(harmonics, dict):
            raise ValueError(f"{name} must be an object")
        _require_int(harmonics.get("l_max"), f"{name}.l_max")
        if harmonics.get("layout") != "lm=l*l+l+m":
            raise ValueError(f"{name}.layout is not supported")

    irreps = normalized.get("irreps")
    if not isinstance(irreps, dict) or set(irreps) != {"source", "edge", "output"}:
        raise ValueError("irreps must define source, edge, and output signatures")
    if any(not isinstance(value, str) or not value for value in irreps.values()):
        raise ValueError("irrep signatures must be non-empty strings")

    paths = normalized.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list")
    for index, path in enumerate(paths):
        if not isinstance(path, dict):
            raise ValueError(f"paths[{index}] must be an object")
        if path.get("index") != index:
            raise ValueError("Execution paths must have dense extraction-order indices")
        for name in ("output_l", "edge_l", "source_l"):
            _require_int(path.get(name), f"paths[{index}].{name}")
        if path.get("connection_mode") not in ("uvu", "uvw"):
            raise ValueError(f"paths[{index}].connection_mode is not supported")
        if path.get("has_weight") is not True:
            raise ValueError(f"paths[{index}] must be weighted")
        shape = path.get("path_shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"paths[{index}].path_shape must be a non-empty list")
        for dimension in shape:
            _require_int(dimension, f"paths[{index}].path_shape", 1)
        if path["edge_l"] > normalized["edge_harmonics"]["l_max"]:
            raise ValueError(f"paths[{index}].edge_l exceeds edge l_max")
        if path["source_l"] > normalized["source_harmonics"]["l_max"]:
            raise ValueError(f"paths[{index}].source_l exceeds source l_max")

    groups = normalized.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a non-empty list")
    covered_paths: list[int] = []
    previous_l = -1
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{index}] must be an object")
        l_value = _require_int(group.get("l"), f"groups[{index}].l")
        if l_value <= previous_l:
            raise ValueError("Execution groups must be ordered by increasing l")
        previous_l = l_value
        if group.get("components") != 2 * l_value + 1:
            raise ValueError(f"groups[{index}] has the wrong component count")
        path_indices = group.get("path_indices")
        if not isinstance(path_indices, list) or not path_indices:
            raise ValueError(f"groups[{index}].path_indices must be non-empty")
        expected = [path["index"] for path in paths if path["output_l"] == l_value]
        if path_indices != expected:
            raise ValueError(f"groups[{index}] does not match path output irreps")
        covered_paths.extend(path_indices)
    if sorted(covered_paths) != list(range(len(paths))):
        raise ValueError("Execution groups must cover every path exactly once")
    if [group["l"] for group in groups] != list(range(groups[-1]["l"] + 1)):
        raise ValueError("Execution groups must cover every l from zero through l_max")

    sparse = normalized.get("sparse_coupling")
    if not isinstance(sparse, dict):
        raise ValueError("sparse_coupling must be an object")
    _require_int(sparse.get("term_count"), "sparse_coupling.term_count", 1)
    for name in ("lme", "coefficients", "rows"):
        value = sparse.get(name + "_fingerprint")
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise ValueError(f"sparse_coupling.{name}_fingerprint is invalid")
    if sparse.get("ordering") != "edge_lm,source_lm,path,output_m":
        raise ValueError("sparse_coupling.ordering is not supported")
    terms = sparse.get("terms")
    if not isinstance(terms, list) or len(terms) != sparse["term_count"]:
        raise ValueError("sparse_coupling.terms does not match term_count")
    maximum_lme = sum(
        group["components"] * len(group["path_indices"]) for group in groups
    )
    maximum_lm1 = (normalized["edge_harmonics"]["l_max"] + 1) ** 2
    maximum_lm2 = (normalized["source_harmonics"]["l_max"] + 1) ** 2
    path_offsets = [0]
    for path in paths:
        path_offsets.append(
            path_offsets[-1] + (2 * path["edge_l"] + 1) * (2 * path["source_l"] + 1)
        )
    lme_paths: list[int] = []
    for group in groups:
        for _component in range(group["components"]):
            lme_paths.extend(group["path_indices"])
    for index, term in enumerate(terms):
        if not isinstance(term, dict) or set(term) != {
            "lme",
            "lm1",
            "lm2",
            "row",
            "coefficient",
        }:
            raise ValueError(f"sparse_coupling.terms[{index}] is invalid")
        lme = _require_int(term["lme"], f"sparse_coupling.terms[{index}].lme")
        lm1 = _require_int(term["lm1"], f"sparse_coupling.terms[{index}].lm1")
        lm2 = _require_int(term["lm2"], f"sparse_coupling.terms[{index}].lm2")
        row = _require_int(term["row"], f"sparse_coupling.terms[{index}].row")
        if lme >= maximum_lme or lm1 >= maximum_lm1 or lm2 >= maximum_lm2:
            raise ValueError(f"sparse_coupling.terms[{index}] exceeds its layout")
        if isinstance(term["coefficient"], bool) or not isinstance(
            term["coefficient"], (int, float)
        ):
            raise ValueError(
                f"sparse_coupling.terms[{index}].coefficient must be numeric"
            )
        try:
            coefficient_is_finite = math.isfinite(float(term["coefficient"]))
        except (OverflowError, ValueError):
            coefficient_is_finite = False
        if not coefficient_is_finite:
            raise ValueError(
                f"sparse_coupling.terms[{index}].coefficient must be finite"
            )

        row_path = next(
            (
                path_index
                for path_index in range(len(paths))
                if path_offsets[path_index] <= row < path_offsets[path_index + 1]
            ),
            None,
        )
        lme_path = lme_paths[lme]
        if row_path != lme_path:
            raise ValueError(
                f"sparse_coupling.terms[{index}].row does not match its lme path"
            )
        path = paths[lme_path]
        edge_component = lm1 - path["edge_l"] ** 2
        source_component = lm2 - path["source_l"] ** 2
        if not 0 <= edge_component < 2 * path["edge_l"] + 1:
            raise ValueError(
                f"sparse_coupling.terms[{index}].lm1 does not match its path"
            )
        if not 0 <= source_component < 2 * path["source_l"] + 1:
            raise ValueError(
                f"sparse_coupling.terms[{index}].lm2 does not match its path"
            )
        expected_row = (
            path_offsets[lme_path]
            + edge_component * (2 * path["source_l"] + 1)
            + source_component
        )
        if row != expected_row:
            raise ValueError(
                f"sparse_coupling.terms[{index}].row does not match lm1/lm2"
            )
    for field in ("lme", "coefficients", "rows"):
        values = (
            [term["coefficient"] for term in terms]
            if field == "coefficients"
            else [term["row" if field == "rows" else field] for term in terms]
        )
        if sparse[field + "_fingerprint"] != _sequence_fingerprint(values):
            raise ValueError(
                f"sparse_coupling.{field}_fingerprint does not match terms"
            )

    layouts = normalized.get("layouts")
    expected_layouts = (
        {
            "aggregate": "receiver,lm,channel",
            "output": "receiver,lm,channel",
        }
        if expected_interaction == R0_INTERACTION
        else {
            "aggregate": "receiver,radial_embedding,path_component_channel",
            "message": "receiver_component,path_channel",
            "output": "receiver,lm,channel",
            "source_adjoint": "source,lm,channel",
        }
    )
    if layouts != expected_layouts:
        raise ValueError(
            f"Execution {expected_interaction} layouts do not match the execution ABI"
        )
    expected_derivatives = (
        ["forward", "coordinate"]
        if expected_interaction == R0_INTERACTION
        else ["forward", "state_adjoint", "source_adjoint", "coordinate"]
    )
    if normalized.get("derivatives") != expected_derivatives:
        raise ValueError(
            f"Execution {expected_interaction} derivative capabilities are not supported"
        )
    if normalized.get("accumulator") != {
        "mode": "runtime_precision",
        "native_generated": ["float32"],
        "fallback": ["float32", "float64"],
    }:
        raise ValueError(
            f"Execution {expected_interaction} accumulator capabilities are not supported"
        )

    generation_fingerprint = _fingerprint(_generation_profile(normalized))
    normalized["generation_fingerprint"] = generation_fingerprint
    semantic_payload = deepcopy(normalized)
    semantic_payload.pop("generation_fingerprint")
    normalized["fingerprint"] = _fingerprint(semantic_payload)
    normalized["structure_fingerprint"] = _fingerprint(
        _tensor_product_structure_profile(normalized)
    )

    if (
        supplied_generation is not None
        and supplied_generation != generation_fingerprint
    ):
        raise ValueError(
            f"Execution {expected_interaction} generation fingerprint does not match its contract"
        )
    if supplied_semantic is not None and supplied_semantic != normalized["fingerprint"]:
        raise ValueError(
            f"Execution {expected_interaction} semantic fingerprint does not match its contract"
        )
    return normalized


def normalize_standard_r0_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a canonical, fingerprinted R0 contract."""

    return _normalize_execution_contract(contract, R0_INTERACTION)


def normalize_jit_r1_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a canonical, fingerprinted R1 contract."""

    return _normalize_execution_contract(contract, R1_INTERACTION)


def _m0_semantic_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": contract["schema"],
        "version": contract["version"],
        "interaction": contract["interaction"],
        "channels": contract["channels"],
        "type_count": contract["type_count"],
        "input_l_max": contract["input_l_max"],
        "output_l_max": contract["output_l_max"],
        "input_components": contract["input_components"],
        "output_components": contract["output_components"],
        "correlation": contract["correlation"],
        "channel_mode": contract["channel_mode"],
        "layouts": contract["layouts"],
        "monomial_groups": contract["monomial_groups"],
    }


def _m0_generation_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_m0_semantic_payload(contract),
        "execution_profile": contract["execution_profile"],
        "derivative_signature": contract["derivative_signature"],
    }


def _m0_structure_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return M0 polynomial topology without channel or species extents."""

    semantic = _m0_semantic_payload(contract)
    semantic.pop("channels")
    semantic.pop("type_count")
    return {
        **semantic,
        "execution_profile": contract["execution_profile"],
        "derivative_signature": contract["derivative_signature"],
    }


def standard_m0_fingerprints(
    contract: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Return semantic, generation, and runtime-shape-free M0 fingerprints."""

    return (
        _fingerprint(_m0_semantic_payload(contract)),
        _fingerprint(_m0_generation_payload(contract)),
        _fingerprint(_m0_structure_payload(contract)),
    )


def normalize_standard_m0_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canonical fixed-weight standard-MACE M0 contract."""

    normalized = deepcopy(dict(contract))
    supplied_semantic = normalized.pop("semantic_fingerprint", None)
    supplied_generation = normalized.pop("generation_fingerprint", None)
    # Unlike semantic fingerprints, structural fingerprints are versioned
    # dispatch metadata and may be migrated from cached extraction payloads.
    normalized.pop("structure_fingerprint", None)

    if normalized.get("schema") != M0_SCHEMA:
        raise ValueError(f"M0 contract schema must be {M0_SCHEMA!r}")
    if normalized.get("version") != M0_SCHEMA_VERSION:
        raise ValueError(f"M0 contract version must be {M0_SCHEMA_VERSION}")
    if normalized.get("interaction") != M0_INTERACTION:
        raise ValueError(f"M0 contract interaction must be {M0_INTERACTION!r}")
    if normalized.get("execution_profile") != M0_EXECUTION_PROFILE:
        raise ValueError(f"M0 execution profile must be {M0_EXECUTION_PROFILE!r}")
    if normalized.get("derivative_signature") != M0_DERIVATIVE_SIGNATURE:
        raise ValueError(f"M0 derivative signature must be {M0_DERIVATIVE_SIGNATURE!r}")
    if normalized.get("channel_mode") != "uvu":
        raise ValueError("M0 standard execution requires tied-channel uvu")

    for name in ("channels", "type_count", "input_components", "output_components"):
        normalized[name] = _require_int(normalized.get(name), f"M0 {name}", 1)
    for name in ("input_l_max", "output_l_max"):
        normalized[name] = _require_int(normalized.get(name), f"M0 {name}")
    normalized["correlation"] = _require_int(
        normalized.get("correlation"), "M0 correlation", 1
    )
    if normalized["input_components"] != (normalized["input_l_max"] + 1) ** 2:
        raise ValueError("M0 input component count is inconsistent with l_max")
    if normalized["output_components"] != (normalized["output_l_max"] + 1) ** 2:
        raise ValueError("M0 output component count is inconsistent with L_max")

    expected_layouts = {
        "input": "node_lm_channel_layout_right",
        "weights": "type_term_channel_layout_right",
        "output": "node_LM_channel_layout_right",
        "output_adjoint": "node_LM_channel_layout_right",
        "input_adjoint": "node_lm_channel_layout_right",
    }
    if normalized.get("layouts") != expected_layouts:
        raise ValueError("M0 contract layouts do not match the standard module ABI")

    groups = normalized.get("monomial_groups")
    if not isinstance(groups, list) or len(groups) != normalized["output_components"]:
        raise ValueError("M0 contract must contain one group per output component")
    for output_component, group in enumerate(groups):
        if (
            not isinstance(group, dict)
            or group.get("output_component") != output_component
        ):
            raise ValueError("M0 monomial groups must use canonical output order")
        terms = group.get("terms")
        if not isinstance(terms, list) or group.get("term_count") != len(terms):
            raise ValueError("M0 group term count does not match its term list")
        expected_terms = sorted(terms, key=lambda term: (len(term), tuple(term)))
        if terms != expected_terms:
            raise ValueError("M0 terms must use degree-then-lexicographic order")
        if len({tuple(term) for term in terms}) != len(terms):
            raise ValueError("M0 groups must not contain duplicate monomials")
        degree_counts = [0] * normalized["correlation"]
        for term in terms:
            if (
                not isinstance(term, list)
                or not 1 <= len(term) <= normalized["correlation"]
            ):
                raise ValueError("M0 monomial degree is outside the contract")
            if term != sorted(term):
                raise ValueError("M0 monomial component indices must be sorted")
            if any(
                isinstance(component, bool)
                or not isinstance(component, int)
                or component < 0
                or component >= normalized["input_components"]
                for component in term
            ):
                raise ValueError("M0 monomial component index is invalid")
            degree_counts[len(term) - 1] += 1
        if group.get("degree_counts") != degree_counts:
            raise ValueError("M0 group degree counts do not match its terms")

    (
        semantic_fingerprint,
        generation_fingerprint,
        structure_fingerprint,
    ) = standard_m0_fingerprints(normalized)
    normalized["semantic_fingerprint"] = semantic_fingerprint
    normalized["generation_fingerprint"] = generation_fingerprint
    normalized["structure_fingerprint"] = structure_fingerprint
    for supplied, expected, name in (
        (supplied_semantic, semantic_fingerprint, "semantic"),
        (supplied_generation, generation_fingerprint, "generation"),
    ):
        if supplied is not None and supplied != expected:
            raise ValueError(f"M0 {name} fingerprint is stale")
    return normalized


def make_standard_m0_contract(
    *,
    channels: int,
    type_count: int,
    input_l_max: int,
    output_l_max: int,
    correlation: int,
    monomials: Mapping[int | str, Iterable[Iterable[int]]],
) -> dict[str, Any]:
    """Build an M0 contract from the canonical polynomial model payload."""

    output_components = (int(output_l_max) + 1) ** 2
    groups = []
    for output_component in range(output_components):
        key = (
            output_component if output_component in monomials else str(output_component)
        )
        if key not in monomials:
            raise ValueError(
                f"M0 monomials are missing output component {output_component}"
            )
        terms = sorted(
            ([int(component) for component in term] for term in monomials[key]),
            key=lambda term: (len(term), tuple(term)),
        )
        degree_counts = [0] * int(correlation)
        for term in terms:
            if 1 <= len(term) <= len(degree_counts):
                degree_counts[len(term) - 1] += 1
        groups.append(
            {
                "output_component": output_component,
                "term_count": len(terms),
                "degree_counts": degree_counts,
                "terms": terms,
            }
        )
    return normalize_standard_m0_contract(
        {
            "schema": M0_SCHEMA,
            "version": M0_SCHEMA_VERSION,
            "interaction": M0_INTERACTION,
            "channels": int(channels),
            "type_count": int(type_count),
            "input_l_max": int(input_l_max),
            "output_l_max": int(output_l_max),
            "input_components": (int(input_l_max) + 1) ** 2,
            "output_components": output_components,
            "correlation": int(correlation),
            "channel_mode": "uvu",
            "execution_profile": M0_EXECUTION_PROFILE,
            "derivative_signature": M0_DERIVATIVE_SIGNATURE,
            "layouts": {
                "input": "node_lm_channel_layout_right",
                "weights": "type_term_channel_layout_right",
                "output": "node_LM_channel_layout_right",
                "output_adjoint": "node_LM_channel_layout_right",
                "input_adjoint": "node_lm_channel_layout_right",
            },
            "monomial_groups": groups,
        }
    )


def make_factorized_contract(
    *,
    channels: int,
    radial_embedding: int,
    edge_l_max: int,
    source_l_max: int,
    source_irreps: str,
    edge_irreps: str,
    output_irreps: str,
    output_l: Iterable[int],
    edge_l: Iterable[int],
    source_l: Iterable[int],
    instructions: Iterable[Mapping[str, Any]],
    sparse_lme: Iterable[int],
    sparse_coefficients: Iterable[float],
    sparse_rows: Iterable[int],
    sparse_lm1: Iterable[int],
    sparse_lm2: Iterable[int],
) -> dict[str, Any]:
    """Build the normalized R1 contract directly from extraction semantics."""

    output_l = [int(value) for value in output_l]
    edge_l = [int(value) for value in edge_l]
    source_l = [int(value) for value in source_l]
    instructions = [dict(instruction) for instruction in instructions]
    if not (len(output_l) == len(edge_l) == len(source_l) == len(instructions)):
        raise ValueError("R1 path and instruction counts must agree")

    paths = []
    for index, instruction in enumerate(instructions):
        paths.append(
            {
                "index": index,
                "output_l": output_l[index],
                "edge_l": edge_l[index],
                "source_l": source_l[index],
                "connection_mode": instruction.get("connection_mode"),
                "has_weight": instruction.get("has_weight"),
                "path_shape": list(instruction.get("path_shape", ())),
            }
        )
    groups = [
        {
            "l": l_value,
            "components": 2 * l_value + 1,
            "path_indices": [
                path["index"] for path in paths if path["output_l"] == l_value
            ],
        }
        for l_value in sorted(set(output_l))
    ]
    sparse_lme = list(sparse_lme)
    sparse_coefficients = list(sparse_coefficients)
    sparse_rows = list(sparse_rows)
    sparse_lm1 = list(sparse_lm1)
    sparse_lm2 = list(sparse_lm2)
    if not (
        len(sparse_lme)
        == len(sparse_coefficients)
        == len(sparse_rows)
        == len(sparse_lm1)
        == len(sparse_lm2)
    ):
        raise ValueError("R1 sparse coupling arrays must have equal lengths")
    finite_coefficients = []
    for index, coefficient in enumerate(sparse_coefficients):
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise ValueError(f"R1 sparse coupling coefficient[{index}] must be numeric")
        try:
            coefficient = float(coefficient)
            coefficient_is_finite = math.isfinite(coefficient)
        except (OverflowError, ValueError):
            coefficient_is_finite = False
        if not coefficient_is_finite:
            raise ValueError(f"R1 sparse coupling coefficient[{index}] must be finite")
        finite_coefficients.append(coefficient)
    sparse_coefficients = finite_coefficients

    return normalize_jit_r1_contract(
        {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "interaction": INTERACTION,
            "channels": int(channels),
            "radial_embedding": int(radial_embedding),
            "edge_harmonics": {
                "l_max": int(edge_l_max),
                "layout": "lm=l*l+l+m",
            },
            "source_harmonics": {
                "l_max": int(source_l_max),
                "layout": "lm=l*l+l+m",
            },
            "irreps": {
                "source": str(source_irreps),
                "edge": str(edge_irreps),
                "output": str(output_irreps),
            },
            "groups": groups,
            "paths": paths,
            "sparse_coupling": {
                "term_count": len(sparse_lme),
                "lme_fingerprint": _sequence_fingerprint(sparse_lme),
                "coefficients_fingerprint": _sequence_fingerprint(sparse_coefficients),
                "rows_fingerprint": _sequence_fingerprint(sparse_rows),
                "ordering": "edge_lm,source_lm,path,output_m",
                "terms": [
                    {
                        "lme": int(lme),
                        "lm1": int(lm1),
                        "lm2": int(lm2),
                        "row": int(row),
                        "coefficient": coefficient,
                    }
                    for lme, lm1, lm2, row, coefficient in zip(
                        sparse_lme,
                        sparse_lm1,
                        sparse_lm2,
                        sparse_rows,
                        sparse_coefficients,
                    )
                ],
            },
            "layouts": {
                "aggregate": "receiver,radial_embedding,path_component_channel",
                "message": "receiver_component,path_channel",
                "output": "receiver,lm,channel",
                "source_adjoint": "source,lm,channel",
            },
            "derivatives": [
                "forward",
                "state_adjoint",
                "source_adjoint",
                "coordinate",
            ],
            "accumulator": {
                "mode": "runtime_precision",
                "native_generated": ["float32"],
                "fallback": ["float32", "float64"],
            },
        }
    )


def make_standard_r0_contract(
    *,
    channels: int,
    radial_embedding: int,
    edge_l_max: int,
    source_irreps: str,
    edge_irreps: str,
    output_irreps: str,
    output_l: Iterable[int],
    edge_l: Iterable[int],
    source_l: Iterable[int],
    instructions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the scalar-source R0 contract from extraction semantics."""

    output_l = [int(value) for value in output_l]
    edge_l = [int(value) for value in edge_l]
    source_l = [int(value) for value in source_l]
    group_offsets: dict[int, int] = {}
    offset = 0
    for l_value in sorted(set(output_l)):
        group_offsets[l_value] = offset
        offset += (2 * l_value + 1) * output_l.count(l_value)

    sparse_lme = []
    sparse_lm1 = []
    sparse_lm2 = []
    for path_index, l_value in enumerate(output_l):
        eta = sum(previous_l == l_value for previous_l in output_l[:path_index])
        eta_count = output_l.count(l_value)
        for component in range(2 * l_value + 1):
            sparse_lme.append(group_offsets[l_value] + component * eta_count + eta)
            sparse_lm1.append(l_value * l_value + component)
            sparse_lm2.append(0)

    contract = make_factorized_contract(
        channels=channels,
        radial_embedding=radial_embedding,
        edge_l_max=edge_l_max,
        source_l_max=0,
        source_irreps=source_irreps,
        edge_irreps=edge_irreps,
        output_irreps=output_irreps,
        output_l=output_l,
        edge_l=edge_l,
        source_l=source_l,
        instructions=instructions,
        sparse_lme=sparse_lme,
        sparse_coefficients=[1.0] * len(sparse_lme),
        sparse_rows=list(range(len(sparse_lme))),
        sparse_lm1=sparse_lm1,
        sparse_lm2=sparse_lm2,
    )
    contract.pop("generation_fingerprint")
    contract.pop("fingerprint")
    contract["interaction"] = R0_INTERACTION
    contract["layouts"] = {
        "aggregate": "receiver,lm,channel",
        "output": "receiver,lm,channel",
    }
    contract["derivatives"] = ["forward", "coordinate"]
    return normalize_standard_r0_contract(contract)
