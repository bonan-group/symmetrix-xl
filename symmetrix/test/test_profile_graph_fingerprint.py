import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
HELPER_PATH = REPOSITORY / "benchmarks" / "profile_graph_fingerprint.py"


@pytest.fixture(scope="module")
def fingerprint_module():
    specification = importlib.util.spec_from_file_location(
        "symmetrix_profile_graph_fingerprint_test", HELPER_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _fingerprint(module, order=(0, 1)):
    return module.canonical_graph_fingerprint(
        np.array([13, 7]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]]),
        np.diag([3.0, 4.0, 5.0]),
        np.array([0, 1])[list(order)],
        np.array([1, 0])[list(order)],
        np.array([[0, 0, 0], [1, 0, 0]])[list(order)],
    )


def test_fingerprint_is_edge_order_independent(fingerprint_module):
    forward = _fingerprint(fingerprint_module)
    reversed_order = _fingerprint(fingerprint_module, order=(1, 0))

    assert forward["sha256"] == reversed_order["sha256"]
    assert (
        forward["sorted_edge_records_sha256"]
        == reversed_order["sorted_edge_records_sha256"]
    )
    assert (
        forward["input_order_edge_records_sha256"]
        != reversed_order["input_order_edge_records_sha256"]
    )


def test_required_fingerprint_rejects_reordered_input(fingerprint_module):
    expected = _fingerprint(fingerprint_module)
    reordered = _fingerprint(fingerprint_module, order=(1, 0))

    with pytest.raises(RuntimeError, match="input_order_edge_records_sha256"):
        fingerprint_module.require_graph_fingerprint(reordered, expected)


def test_required_fingerprint_accepts_both_matching_identities(fingerprint_module):
    expected = _fingerprint(fingerprint_module)

    assert fingerprint_module.require_graph_fingerprint(expected, expected) is None


def test_fingerprint_uses_quantized_geometry(fingerprint_module):
    reference = _fingerprint(fingerprint_module)
    below_quantum = fingerprint_module.canonical_graph_fingerprint(
        [13, 7],
        [[0.0, 0.0, 0.0], [1.00001, 0.5, 0.25]],
        np.diag([3.0, 4.0, 5.0]),
        [0, 1],
        [1, 0],
        [[0, 0, 0], [1, 0, 0]],
    )
    above_quantum = fingerprint_module.canonical_graph_fingerprint(
        [13, 7],
        [[0.0, 0.0, 0.0], [1.0001, 0.5, 0.25]],
        np.diag([3.0, 4.0, 5.0]),
        [0, 1],
        [1, 0],
        [[0, 0, 0], [1, 0, 0]],
    )

    assert reference["sha256"] == below_quantum["sha256"]
    assert reference["sha256"] != above_quantum["sha256"]


def test_fingerprint_rejects_nonintegral_shifts(fingerprint_module):
    with pytest.raises(ValueError, match="integral"):
        fingerprint_module.canonical_graph_fingerprint(
            [13, 7],
            [[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]],
            np.diag([3.0, 4.0, 5.0]),
            [0],
            [1],
            [[0.5, 0, 0]],
        )


@pytest.mark.parametrize("quantum", [0.0, -1.0e-4, float("nan")])
def test_fingerprint_rejects_invalid_coordinate_quantum(fingerprint_module, quantum):
    with pytest.raises(ValueError, match="positive reciprocal integer"):
        fingerprint_module.canonical_graph_fingerprint(
            [13, 7],
            [[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]],
            np.diag([3.0, 4.0, 5.0]),
            [0],
            [1],
            [[0, 0, 0]],
            coordinate_quantum_A=quantum,
        )
