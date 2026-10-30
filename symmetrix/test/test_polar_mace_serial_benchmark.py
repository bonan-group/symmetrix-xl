import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "benchmarks/polar_mace_serial_benchmark.py"


@pytest.fixture(scope="module")
def benchmark_driver():
    spec = importlib.util.spec_from_file_location(
        "polar_mace_serial_benchmark_test", DRIVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Calculator:
    def __init__(self, route):
        self.streamed_edges = route


def _counters(forward, source_reverse, edge_reverse, fallback):
    return {
        "execution_mh1_generated_forward_launch_count": forward,
        "execution_mh1_generated_source_reverse_launch_count": source_reverse,
        "execution_mh1_generated_edge_reverse_launch_count": edge_reverse,
        "factorized_fallback_evaluation_count": fallback,
    }


def test_direct_record_uses_measured_window_deltas(benchmark_driver):
    evidence = benchmark_driver._measured_execution_evidence(
        _Calculator("direct"),
        _counters(100, 80, 40, 0),
        _counters(112, 88, 44, 0),
    )
    assert evidence["classification"] == "direct"
    assert evidence["counter_delta"] == _counters(12, 8, 4, 0)


def test_direct_record_rejects_initialization_only_evidence(benchmark_driver):
    with pytest.raises(RuntimeError, match="measured workload"):
        benchmark_driver._measured_execution_evidence(
            _Calculator("direct"),
            _counters(100, 80, 40, 0),
            _counters(100, 80, 40, 0),
        )


def test_fallback_record_classifies_measured_fallback_delta(benchmark_driver):
    evidence = benchmark_driver._measured_execution_evidence(
        _Calculator("reference"),
        _counters(0, 0, 0, 3),
        _counters(0, 0, 0, 9),
    )
    assert evidence["classification"] == "fallback"
    assert evidence["counter_delta"]["factorized_fallback_evaluation_count"] == 6


def test_record_aggregates_only_explicit_measured_windows(benchmark_driver):
    calculator = _Calculator("direct")
    windows = [
        benchmark_driver._measured_execution_evidence(
            calculator, _counters(20, 20, 20, 0), _counters(24, 24, 24, 0)
        ),
        benchmark_driver._measured_execution_evidence(
            calculator, _counters(70, 60, 50, 0), _counters(78, 68, 58, 0)
        ),
    ]
    evidence = benchmark_driver._aggregate_measured_execution_evidence(
        calculator, windows
    )
    assert evidence["counter_delta"] == _counters(12, 12, 12, 0)
    assert evidence["windows"] == windows
    assert "counter_start" not in evidence
    assert "counter_end" not in evidence


def test_disabled_comparator_record_is_explicit_and_omits_equivalence(
    benchmark_driver,
):
    fields = benchmark_driver._comparator_record_fields(False)
    assert fields == {
        "comparator_verification": {
            "enabled": False,
            "status": "disabled_by_cli",
        }
    }
    assert "comparator_equivalence" not in fields
