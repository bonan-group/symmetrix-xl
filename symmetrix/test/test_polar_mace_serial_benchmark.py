import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
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
    def __init__(
        self,
        route,
        dense_policy=None,
        *,
        profile="capacity",
        node_state="retain-interaction-v1",
        node_arena="capacity-v1",
        low_memory_policy="capacity-retained",
    ):
        self.streamed_edges = route
        self.jit_dense_policy = dense_policy
        self.execution_profile = profile
        self.execution_mh1_node_state_policy = node_state
        self.execution_mh1_node_arena_policy = node_arena
        self.low_memory_policy = low_memory_policy


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


def test_direct_dense_policy_evidence_requires_requested_provider(
    benchmark_driver, monkeypatch
):
    monkeypatch.setenv("SYMMETRIX_MH1_DENSE_POLICY", "vendor_fp32")
    evidence = benchmark_driver._dense_policy_evidence(
        _Calculator("direct", "vendor_fp32"), "vendor_fp32"
    )
    assert evidence == {
        "requested": "vendor_fp32",
        "selected": "vendor_fp32",
        "environment": "vendor_fp32",
    }


def test_direct_dense_policy_evidence_rejects_provider_mismatch(benchmark_driver):
    with pytest.raises(RuntimeError, match="unexpected MH-1 dense policy"):
        benchmark_driver._dense_policy_evidence(
            _Calculator("direct", "generated"), "vendor_fp32"
        )


def test_reference_dense_policy_evidence_allows_no_loaded_provider(benchmark_driver):
    evidence = benchmark_driver._dense_policy_evidence(
        _Calculator("reference", None), "generated"
    )
    assert evidence["requested"] == "generated"
    assert evidence["selected"] is None


def test_speed_node_policy_evidence_requires_requested_state(benchmark_driver):
    calculator = _Calculator(
        "direct",
        profile="speed",
        node_state="full-retention-v1",
        node_arena="throughput-v1",
        low_memory_policy="speed",
    )
    evidence = benchmark_driver._node_policy_evidence(
        calculator, "speed", "full-retention-v1", "throughput-v1"
    )
    assert evidence["node_state_selected"] == "full-retention-v1"
    assert evidence["low_memory_policy"] == "speed"


def test_speed_node_policy_evidence_rejects_selected_state_mismatch(
    benchmark_driver,
):
    calculator = _Calculator("direct", profile="speed", node_state="recompute-v1")
    with pytest.raises(RuntimeError, match="unexpected MH-1 node-state policy"):
        benchmark_driver._node_policy_evidence(
            calculator, "speed", "full-retention-v1", "throughput-v1"
        )


def test_capacity_node_policy_evidence_records_intentional_override(benchmark_driver):
    calculator = _Calculator("direct")
    evidence = benchmark_driver._node_policy_evidence(
        calculator, "capacity", "full-retention-v1", "throughput-v1"
    )
    assert evidence["node_state_requested"] == "full-retention-v1"
    assert evidence["node_state_selected"] == "retain-interaction-v1"
    assert evidence["node_arena_selected"] == "capacity-v1"


def test_cuda_tier_probe_uses_current_renderer_api():
    from symmetrix.mace_nonlinear_jit_codegen import render_jit_mh1_cuda_plugin

    assert callable(render_jit_mh1_cuda_plugin)


def test_capacity_fields_accept_native_integrated_long_range_diagnostics(
    benchmark_driver,
):
    calculator = SimpleNamespace(
        _polar_contract={
            "table_dtype": "float32",
            "spin_channel_count": 2,
            "recursion_step_count": 3,
            "kspace_cutoff": 4.5,
        },
        _polar_shapes={
            "interaction_count": 2,
            "node_feature_dimension": 8,
            "charges_dimension": 4,
            "potential_dimension": 6,
        },
    )
    composition = {
        "n_k": 17,
        "positions": np.zeros((5, 3)),
        "pbc": np.ones((1, 3), dtype=np.int32),
    }

    capacity = benchmark_driver._capacity_fields(
        calculator, composition, {"graph_class": "bulk"}, None
    )

    assert capacity["n_k"] == 17
    assert capacity["table_bytes"] == 2 * 17 * 5 * 4
    assert capacity["retention_bytes"] == 5 * 2 * 8 * 4
