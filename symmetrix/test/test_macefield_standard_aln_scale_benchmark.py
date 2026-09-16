import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "macefield_standard_aln_scale.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("macefield_aln_scale", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_matrix_does_not_request_removed_streamed_mode():
    assert "second_interaction" not in SCRIPT.read_text(encoding="ascii")


@pytest.mark.parametrize(
    ("response", "routing", "expected"),
    [
        ("none", "public-native", "native-evaluator"),
        ("polarization", "public-native", "calculator-results"),
        ("polarization", "cached-host", "cached-host-results"),
    ],
)
def test_timing_scope_distinguishes_native_public_and_cached_host(
    response, routing, expected
):
    benchmark = _load_benchmark()
    assert benchmark._timing_scope(response, routing) == expected


def test_explicit_r1_forward_accepts_canonical_direct_mode():
    benchmark = _load_benchmark()
    benchmark._validate_r1_forward_selection("jit_all", "direct")
    benchmark._validate_r1_forward_selection("runtime", "direct")
    with pytest.raises(ValueError, match="streamed_edges='direct'"):
        benchmark._validate_r1_forward_selection("jit_all", "generic")


def _args(tmp_path, response):
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="ascii")
    return SimpleNamespace(
        output=tmp_path / "results.json",
        field_model=model,
        standard_model=model,
        sizes=["small"],
        policies=["optimized_recompute"],
        trials=1,
        response=response,
        response_routing="public-native",
        backend="cuda",
        warmups=1,
        samples=2,
        electric_field=(0.01, -0.02, 0.03),
        m1_tile_channels=16,
        edge_geometry_policy="unit-f32-radius-f64-v1",
        low_memory=True,
        neighbor_skin=0.0,
        timeout=10.0,
    )


def test_run_forwards_and_validates_response(monkeypatch, tmp_path):
    benchmark = _load_benchmark()
    args = _args(tmp_path, "polarizability")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        record = {
            "response": "polarizability",
            "atoms": 32,
            "timing": {"median_ms": 1.0},
            "result": {"polarizability_shape": [9], "becs_shape": None},
        }
        return SimpleNamespace(
            returncode=0,
            stdout=benchmark.PREFIX + json.dumps(record) + "\n",
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    benchmark._run(args)

    assert commands[0][commands[0].index("--response") + 1] == "polarizability"
    assert "--low-memory" in commands[0]
    assert commands[0][commands[0].index("--neighbor-skin") + 1] == "0.0"
    payload = json.loads(args.output.read_text())
    assert len(payload["records"]) == 1
    assert payload["failures"] == []


def test_run_rejects_mismatched_response_record(monkeypatch, tmp_path):
    benchmark = _load_benchmark()
    args = _args(tmp_path, "both")

    def fake_run(command, **kwargs):
        record = {
            "response": "none",
            "atoms": 32,
            "timing": {"median_ms": 1.0},
            "result": {"polarizability_shape": None, "becs_shape": None},
        }
        return SimpleNamespace(
            returncode=0,
            stdout=benchmark.PREFIX + json.dumps(record) + "\n",
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    benchmark._run(args)

    payload = json.loads(args.output.read_text())
    assert payload["records"] == []
    assert payload["failures"][0]["requested_response"] == "both"
    assert payload["failures"][0]["recorded_response"] == "none"


def test_run_records_nonzero_worker_with_empty_output(monkeypatch, tmp_path):
    benchmark = _load_benchmark()
    args = _args(tmp_path, "both")

    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="worker failed before producing a record",
        ),
    )
    benchmark._run(args)

    payload = json.loads(args.output.read_text())
    assert payload["records"] == []
    assert payload["failures"] == [
        {
            "backend": "cuda",
            "size": "small",
            "policy": "optimized_recompute",
            "trial": 1,
            "kind": "field",
            "returncode": 2,
            "requested_response": "both",
            "recorded_response": None,
            "stderr_tail": "worker failed before producing a record",
        }
    ]


def test_run_alternates_policy_order_between_trials(monkeypatch, tmp_path):
    benchmark = _load_benchmark()
    args = _args(tmp_path, "polarizability")
    args.policies = ["optimized", "optimized_recompute"]
    args.trials = 2
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        record = {
            "response": "polarizability",
            "atoms": 32,
            "timing": {"median_ms": 1.0},
            "result": {"polarizability_shape": [9], "becs_shape": None},
        }
        return SimpleNamespace(
            returncode=0,
            stdout=benchmark.PREFIX + json.dumps(record) + "\n",
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    benchmark._run(args)

    policies = [command[command.index("--policy") + 1] for command in commands]
    assert policies == [
        "optimized",
        "optimized_recompute",
        "optimized_recompute",
        "optimized",
    ]


def test_run_polarization_compares_field_with_standard(monkeypatch, tmp_path):
    benchmark = _load_benchmark()
    args = _args(tmp_path, "polarization")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        kind = command[command.index("--kind") + 1]
        record = {
            "kind": kind,
            "response": "polarization",
            "atoms": 32,
            "timing": {"median_ms": 1.0},
            "result": {
                "polarization_shape": [3] if kind == "field" else None,
                "polarizability_shape": None,
                "becs_shape": None,
            },
        }
        return SimpleNamespace(
            returncode=0,
            stdout=benchmark.PREFIX + json.dumps(record) + "\n",
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    benchmark._run(args)

    assert [command[command.index("--kind") + 1] for command in commands] == [
        "field",
        "standard",
    ]
    payload = json.loads(args.output.read_text())
    assert len(payload["records"]) == 2
    assert payload["failures"] == []
