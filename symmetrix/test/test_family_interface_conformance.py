"""The family-neutral half of the evaluator interface conformance.

The compile-time half of the contract is the `static_assert` on
``symmetrix::execution::FamilyEvaluator`` and ``TieredFamilyEvaluator`` that
each family's own translation unit carries, so a missing or mistyped member
fails the library build.  This module covers the runtime half's shared piece.

The surface assertions every evaluator is checked with live in
``conformance.surface``, and each family's own
``test_family_<family>_conformance_cases`` module drives them for its evaluators.
What remains here is the one contract that is stated once for every family: an
empty artifact path must stay a readiness query, so no loader is reached and the
working directory is never read.  Which families it is asserted on, and how each
one's evaluator is built, is declared by the families themselves.
"""

import json

import pytest

from conformance.registry import READINESS_CASES
from conformance.surface import DECOY_MANIFEST, READINESS_KEYS


@pytest.mark.parametrize(
    "case", READINESS_CASES, ids=[case.name for case in READINESS_CASES]
)
def test_load_family_artifact_with_an_empty_path_is_a_readiness_query(
    case, tmp_path, monkeypatch, request
):
    """An empty path must not reach a loader, and must not read the working
    directory.

    The host loaders resolve an identity manifest beside the plugin path, so an
    unguarded empty path would read `manifest.json` from whatever directory the
    process happens to be in.  The decoy manifest below would fail a real load
    loudly, so a passing call proves no load happened.
    """
    evaluator = case.build(request)

    decoy = tmp_path / "decoy-cwd"
    decoy.mkdir()
    (decoy / "manifest.json").write_text(json.dumps(DECOY_MANIFEST))
    monkeypatch.chdir(decoy)

    readiness = evaluator.family_artifact_readiness
    for target in ("", "host", "cuda"):
        report = evaluator.load_family_artifact("", target)
        assert set(report) == set(READINESS_KEYS)
        assert report["ready"] is False
        assert report["artifact_id"] == ""
        assert report["target"] == (
            target if target else evaluator.family_capabilities["target"]
        )
    # The readiness query reports the same state as the family accessor.
    assert (
        evaluator.family_artifact_readiness["artifact_id"] == readiness["artifact_id"]
    )
    assert evaluator.family_artifact_ready is False
