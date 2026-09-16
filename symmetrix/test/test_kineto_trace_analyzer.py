import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
ANALYZER_PATH = REPOSITORY / "benchmarks" / "analyze_kineto_trace.py"


@pytest.fixture(scope="module")
def analyzer():
    specification = importlib.util.spec_from_file_location(
        "symmetrix_kineto_trace_analyzer_test", ANALYZER_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _event(category, name, timestamp, duration, *, pid=10, tid=1, external_id=None):
    event = {
        "ph": "X",
        "cat": category,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": timestamp,
        "dur": duration,
        "args": {},
    }
    if external_id is not None:
        event["args"]["External id"] = external_id
    return event


def _trace_events(analyzer):
    events = [
        _event("user_annotation", analyzer.FORWARD_RANGE, 0, 100),
        _event("user_annotation", "mace.module:products.0", 10, 30),
        _event("user_annotation", "mace.module:interactions.0", 50, 20),
        _event("cpu_op", "aten::copy_", 12, 2, external_id=1),
        _event("cpu_op", "aten::add", 20, 2, external_id=2),
        _event("cpu_op", "aten::bmm", 52, 2, external_id=3),
        _event("cpu_op", "aten::sum", 80, 2, external_id=4),
        _event("cpu_op", "BmmBackward0", 30, 2, tid=2, external_id=5),
    ]
    for external_id, duration in enumerate((5, 7, 11, 13, 17), start=1):
        events.append(
            _event(
                "kernel",
                f"kernel_{external_id}",
                200 + external_id,
                duration,
                pid=0,
                tid=9,
                external_id=external_id,
            )
        )
    return events


def _write_trace(path, events):
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


def test_summarizes_unique_launches_and_module_containment(analyzer, tmp_path):
    path = tmp_path / "trace.json"
    _write_trace(path, _trace_events(analyzer))

    summary = analyzer.summarize(path)

    assert summary["totals"] == {"launches": 5, "device_time_us": 53.0}
    assert summary["pathways"] == {
        "forward": {"launches": 4, "device_time_us": 36.0},
        "autograd_reverse": {"launches": 1, "device_time_us": 17.0},
    }
    assert summary["forward_modules"] == {
        "products": {"launches": 2, "device_time_us": 12.0},
        "interactions": {"launches": 1, "device_time_us": 11.0},
        "other": {"launches": 1, "device_time_us": 13.0},
    }
    assert summary["product_operators"]["aten::copy_"] == {
        "launches": 1,
        "device_time_us": 5.0,
    }
    assert summary["threads"]["autograd_thread_ids"] == [2]


def test_rejects_ambiguous_external_id_mapping(analyzer, tmp_path):
    events = _trace_events(analyzer)
    events.append(_event("cpu_op", "aten::mul", 21, 1, external_id=2))
    path = tmp_path / "ambiguous.json"
    _write_trace(path, events)

    with pytest.raises(analyzer.AnalysisError, match="maps to 2 CPU ops"):
        analyzer.summarize(path)


def test_rejects_kernel_without_cpu_op(analyzer, tmp_path):
    events = _trace_events(analyzer)
    events[-1]["args"]["External id"] = 99
    path = tmp_path / "missing.json"
    _write_trace(path, events)

    with pytest.raises(analyzer.AnalysisError, match="maps to 0 CPU ops"):
        analyzer.summarize(path)


def test_rejects_main_thread_launch_outside_forward_range(analyzer, tmp_path):
    events = _trace_events(analyzer)
    cpu_op = next(
        event
        for event in events
        if event.get("cat") == "cpu_op" and event["args"].get("External id") == 4
    )
    cpu_op["ts"] = 101
    path = tmp_path / "outside.json"
    _write_trace(path, events)

    with pytest.raises(analyzer.AnalysisError, match="outside the measured"):
        analyzer.summarize(path)


def test_cli_writes_json_and_stable_table(analyzer, tmp_path, capsys):
    path = tmp_path / "trace.json"
    output = tmp_path / "summary.json"
    _write_trace(path, _trace_events(analyzer))

    assert analyzer.main([str(path), "--json-output", str(output)]) == 0

    rendered = capsys.readouterr().out
    assert "autograd_reverse" in rendered
    assert "products/aten::copy_" in rendered
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == "symmetrix.kineto-mace-summary"
    assert document["invariants"]["device_kernels_assigned_once"] is True
