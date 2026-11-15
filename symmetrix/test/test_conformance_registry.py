"""Conformance discovery must not depend on pytest's first imported family."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "first",
    (
        "conformance.registry",
        "test_family_mace_conformance_cases",
        "test_family_mace_nonlinear_conformance_cases",
        "test_family_mace_polar_conformance_cases",
    ),
)
def test_family_discovery_keeps_field_instances_when_a_family_imports_first(first):
    tests = Path(__file__).parent
    program = """
import importlib
import json
import sys
sys.path.insert(0, sys.argv[1])
importlib.import_module(sys.argv[2])
from conformance.registry import ADAPTER_SPECS, FIELD_RESPONSE_INSTANCES, STRUCTURES
print(json.dumps({"instances": sorted(ADAPTER_SPECS),
                  "field_response": sorted(FIELD_RESPONSE_INSTANCES),
                  "structures": sorted(STRUCTURES)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tests), first],
        env={**os.environ, "SYMMETRIX_BACKEND": "cpu"},
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert "macefield-f64" in result["instances"]
    assert result["field_response"] == ["macefield-f64"]
    assert "mh0" in result["structures"]
