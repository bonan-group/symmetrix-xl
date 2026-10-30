from types import SimpleNamespace

import pytest

from symmetrix.calculator import Symmetrix
from symmetrix.r0_reporting import R0_REPORTING_SCHEMA, r0_diagnostics


@pytest.mark.parametrize(
    ("implementation", "origin"),
    (
        ("generic", None),
        ("generated", "prebuilt"),
        ("generated", "cache"),
        ("generated", "runtime"),
    ),
)
def test_r0_diagnostics_accepts_only_semantic_implementation_and_artifact_origin(
    implementation, origin
):
    evaluator = SimpleNamespace(
        r0_implementation=implementation,
        r0_backend="cuda",
        r0_artifact_origin=origin,
        r0_schedule="receiver-edge16" if implementation == "generated" else "generic",
        r0_profile="persistent-4" if implementation == "generated" else "generic",
        r0_module_id="artifact" if implementation == "generated" else "",
    )

    report = r0_diagnostics(evaluator)

    assert report == {
        "implementation": implementation,
        "backend": "cuda",
        "artifact_origin": origin,
        "schedule": evaluator.r0_schedule,
        "profile": evaluator.r0_profile,
        "artifact_id": "artifact" if implementation == "generated" else None,
    }
    assert "loader" not in report


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("r0_artifact_origin", "cache", "artifact origin"),
        ("r0_module_id", "artifact", "artifact ID"),
        ("r0_schedule", "frozen", "schedule"),
        ("r0_profile", "fixed-weight-coordinate", "profile"),
    ),
)
def test_generic_r0_diagnostics_rejects_generated_metadata(field, value, message):
    evaluator = SimpleNamespace(
        r0_implementation="generic",
        r0_backend="host",
        r0_artifact_origin=None,
        r0_schedule="generic",
        r0_profile="generic",
        r0_module_id="",
    )
    setattr(evaluator, field, value)

    with pytest.raises(RuntimeError, match=f"generic R0 cannot report.*{message}"):
        r0_diagnostics(evaluator)


def test_calculator_exposes_normalized_r0_diagnostics():
    calculator = object.__new__(Symmetrix)
    calculator.evaluator = SimpleNamespace(
        r0_implementation="generated",
        r0_backend="host",
        r0_artifact_origin="runtime",
        r0_schedule="host-owner",
        r0_profile="owner-batched",
        r0_module_id="artifact",
        _r0_loader_implementation="host_plugin",
    )

    assert calculator.r0_diagnostics == {
        "implementation": "generated",
        "backend": "host",
        "artifact_origin": "runtime",
        "schedule": "host-owner",
        "profile": "owner-batched",
        "artifact_id": "artifact",
    }


def test_r0_reporting_schema_names_orthogonal_public_fields():
    assert R0_REPORTING_SCHEMA == {
        "schema": "symmetrix.r0-reporting/1",
        "implementation": ["generic", "generated"],
        "backend": ["host", "cuda", "hip"],
        "artifact_origin": ["prebuilt", "cache", "runtime"],
        "nullable": ["artifact_origin", "artifact_id"],
        "separate_fields": ["schedule", "profile"],
    }
