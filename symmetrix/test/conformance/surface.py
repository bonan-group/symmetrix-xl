"""The family-neutral half of the evaluator interface, asserted on any evaluator.

The compile-time half of the contract is the `static_assert` on
``symmetrix::execution::FamilyEvaluator`` and ``TieredFamilyEvaluator`` that each
family's own translation unit carries, so a missing or mistyped member fails the
library build.  This module covers the runtime half: the neutral surface must be
observable and internally consistent per evaluator, and the whole-plan
execution-plan report must carry the additive tier-binding fields while every
pre-existing field keeps its spelling.

The checks here take an evaluator and name no family.  ``EmptyPathCase`` is the
declaration a family module makes so the shared readiness contract can be
asserted on that family's evaluator without the shared suite knowing how to
build it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from symmetrix import family_registry

CAPABILITY_KEYS = family_registry.CAPABILITY_KEYS

PROGRAM_KEYS = ("family", "precision", "version", "fingerprint", "operators")

BINDING_REPORT_KEYS = (
    "version",
    "target",
    "precision",
    "artifact_ready",
    "admitted",
    "node_count",
    "portable_node_count",
    "bindings",
)

BINDING_KEYS = (
    "operator_id",
    "kind",
    "capability_key",
    "implementation_id",
    "selected_implementation_id",
    "tier",
    "target",
    "precision",
    "identity_fingerprint",
    "qualified",
    "portable_covered",
    "artifact_required",
    "reason",
)

READINESS_KEYS = (
    "ready",
    "target",
    "artifact_id",
    "generation_fingerprint",
    "semantic_fingerprint",
    "structure_fingerprint",
    "reason",
)

OPERATOR_KEYS = (
    "kind",
    "id",
    "capability_key",
    "parameters",
    "inputs",
    "outputs",
)

PLAN_REPORT_KEYS = (
    "requested_algorithm",
    "requested_profile",
    "selection_source",
    "state",
    "selected_id",
    "selection_reason",
    "available_bytes",
    "reserve_bytes",
    "boundary_attempt",
    "candidates",
)


def _native():
    try:
        from symmetrix import symmetrix as native_symmetrix
    except Exception as error:  # pragma: no cover - environment without backend
        pytest.skip(f"native Symmetrix package is unavailable: {error}")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    return native_symmetrix


def _assert_capabilities(capabilities):
    assert set(capabilities) == set(CAPABILITY_KEYS)
    assert capabilities["target"] in ("host", "cuda", "hip")
    assert capabilities["precision"] in ("float32", "float64")
    for key in CAPABILITY_KEYS:
        if key not in ("target", "precision"):
            assert isinstance(capabilities[key], bool), key


def _assert_program(program):
    assert set(program) == set(PROGRAM_KEYS)
    assert program["fingerprint"].startswith("sha256:")
    assert len(program["fingerprint"]) == len("sha256:") + 64
    assert program["operators"], "a compiled program must declare its operators"
    for operator in program["operators"]:
        assert set(operator) == set(OPERATOR_KEYS)
        assert operator["id"]
        assert operator["capability_key"]
        for contract in (*operator["inputs"], *operator["outputs"]):
            assert set(contract) == {"name", "dtype", "shape", "layout", "irreps"}
            assert contract["layout"] in (
                "node_major",
                "lane_major",
                "edge_major",
                "block_major",
            )


def _assert_binding_report(report, program):
    assert set(report) == set(BINDING_REPORT_KEYS)
    assert report["version"] == 1
    assert report["target"] in ("host", "cuda", "hip")
    assert report["node_count"] == len(program["operators"])
    assert len(report["bindings"]) == report["node_count"]
    assert [binding["operator_id"] for binding in report["bindings"]] == [
        operator["id"] for operator in program["operators"]
    ]
    for binding in report["bindings"]:
        assert set(binding) == set(BINDING_KEYS)
        assert binding["tier"] in ("reference", "portable", "frozen", "generated")
        assert binding["target"] == report["target"]
        assert set(binding["precision"]) == {"compute", "accumulator"}
        assert binding["reason"]
        # A generated implementation is the only one that needs an artifact, and
        # it never counts as the coverage a host backend requires.
        assert binding["artifact_required"] == (
            binding["qualified"] and not binding["portable_covered"]
        )
        if not binding["qualified"]:
            assert not binding["implementation_id"]
            assert not binding["selected_implementation_id"]
            assert not binding["artifact_required"]
        else:
            assert binding["implementation_id"]
            # The executed implementation is always stated for a bound node:
            # the report is the authority on what ran, so an empty field would
            # leave a consumer to derive the implementation from the tier
            # again.
            assert binding["selected_implementation_id"]
            assert isinstance(binding["selected_implementation_id"], str)
    assert report["portable_node_count"] == sum(
        1 for binding in report["bindings"] if binding["portable_covered"]
    )
    if report["admitted"]:
        assert all(binding["qualified"] for binding in report["bindings"])
        assert report["portable_node_count"] <= report["node_count"]
    if report["portable_node_count"] == report["node_count"]:
        # Full coverage below the generated tier means every node has a
        # qualified implementation, so the plan is complete as well.
        assert report["admitted"] is True


def _assert_artifact_and_diagnostics(evaluator):
    readiness = evaluator.family_artifact_readiness
    assert set(readiness) == set(READINESS_KEYS)
    assert isinstance(readiness["ready"], bool)
    assert readiness["ready"] == evaluator.family_artifact_ready
    assert readiness["reason"]
    diagnostics = evaluator.family_diagnostics
    assert set(diagnostics) == {"fallback_count", "diagnostic"}
    assert isinstance(diagnostics["fallback_count"], int)


def _assert_interface_surface(evaluator):
    _assert_capabilities(evaluator.family_capabilities)
    program = evaluator.family_program
    _assert_program(program)
    _assert_binding_report(evaluator.family_binding_report, program)
    _assert_artifact_and_diagnostics(evaluator)
    return program


DECOY_MANIFEST = {
    "key_inputs": {
        "build": {
            "metadata": {
                "artifact_id": "decoy",
                "generation_fingerprint": "decoy",
                "semantic_fingerprint": "decoy",
                "structure_fingerprint": "decoy",
            }
        }
    }
}


@dataclass(frozen=True)
class EmptyPathCase:
    """One family's evaluator factory for the shared empty-path readiness case.

    ``name`` is the parametrization id, so the shared suite keeps reporting the
    assertion under the family's own name, and ``build`` returns the evaluator
    that family needs -- resolving whatever session fixture its model requires,
    or skipping when this build cannot serve it.
    """

    name: str
    build: Callable[..., object]
