import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
SCRIPT = BENCHMARKS / "run_generated_r0_cpu_baseline.py"


@pytest.fixture(scope="module")
def runner():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec = importlib.util.spec_from_file_location(
            "symmetrix_generated_r0_cpu_runner_test", SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(BENCHMARKS))


def test_runner_has_a_repository_cpu_campaign_entrypoint(runner):
    assert callable(runner.run_cpu_campaign)
    completed = subprocess.run(
        (sys.executable, str(SCRIPT), "--help"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--model" in completed.stdout
    assert "--cache-root" in completed.stdout
    assert "--output" in completed.stdout


def test_runner_compiles_r0_with_the_production_host_policy():
    source = SCRIPT.read_text()

    assert "probe_compiler_identity" in source
    assert "resolve_host_jit_policy" in source
    assert "target=host_policy.effective_target" in source
    assert "cxx_flags=None" in source
    assert "host_policy=host_policy" in source
    assert "source_content_sha256=source_content_sha256" in source
    assert "host_policy.effective_target,\n                        )" in source


def test_runner_requires_an_empty_task_specific_cache(runner, monkeypatch, tmp_path):
    for name in runner._THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {7})
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "stale-artifact").write_text("stale")

    with pytest.raises(ValueError, match="task-specific cache must be empty"):
        runner._require_reproducible_cpu_environment(cache)


def test_reproducible_command_captures_affinity_threads_and_arguments(
    runner, monkeypatch
):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {7})
    for name in runner._THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OMP_PROC_BIND", "false")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", "/task-cache")
    for name in runner._HOST_POLICY_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    command = runner._reproducible_benchmark_command(
        ["--model", "/model with spaces.json"]
    )

    assert command.startswith("taskset -c 7 env ")
    assert "OMP_NUM_THREADS=1" in command
    assert "SYMMETRIX_JIT_CACHE=/task-cache" in command
    for name in runner._HOST_POLICY_ENVIRONMENT:
        assert f"-u {name}" in command
    assert "'/model with spaces.json'" in command


def test_reproducible_command_and_provenance_capture_explicit_host_policy(
    runner, monkeypatch
):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {7})
    for name in runner._THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    policy_environment = {
        "CXX": "/opt/compiler with spaces/c++",
        "SYMMETRIX_JIT_HOST_TARGET": "x86-64-v3",
        "SYMMETRIX_JIT_HOST_FLAGS": "-ffast-math -march=x86-64-v3",
        "SYMMETRIX_JIT_HOST_FP_MODE": "ieee",
    }
    for name, value in policy_environment.items():
        monkeypatch.setenv(name, value)
    policy = SimpleNamespace(
        identity={
            "schema": "symmetrix.jit.host-policy/2",
            "requested": "x86-64-v3",
            "selected": "explicit-flags",
            "effective_target": "host-sha256-" + "a" * 64,
            "flags": ["-ffast-math", "-march=x86-64-v3"],
        }
    )
    compiler_identity = {
        "command": ["/opt/compiler with spaces/c++"],
        "resolved_executable": "/opt/compiler with spaces/c++",
        "version": "fixture compiler 1.0",
        "size": 123,
        "mtime_ns": 456,
    }

    command = runner._reproducible_benchmark_command([])
    provenance = runner._host_policy_provenance(
        ("/opt/compiler with spaces/c++",), compiler_identity, policy
    )

    assert "'CXX=/opt/compiler with spaces/c++'" in command
    assert "SYMMETRIX_JIT_HOST_TARGET=x86-64-v3" in command
    assert "'SYMMETRIX_JIT_HOST_FLAGS=-ffast-math -march=x86-64-v3'" in command
    assert "SYMMETRIX_JIT_HOST_FP_MODE=ieee" in command
    assert provenance == {
        "compiler_command": ["/opt/compiler with spaces/c++"],
        "compiler_identity": compiler_identity,
        "effective_target": "host-sha256-" + "a" * 64,
        "flags": ["-ffast-math", "-march=x86-64-v3"],
        "floating_point_mode": None,
        "requested_floating_point_mode": "ieee",
        "policy": policy.identity,
        "environment": policy_environment,
    }


def test_host_policy_provenance_reports_effective_named_fp_mode(runner, monkeypatch):
    monkeypatch.delenv("SYMMETRIX_JIT_HOST_FLAGS", raising=False)
    monkeypatch.setenv("SYMMETRIX_JIT_HOST_FP_MODE", "ieee")
    policy = SimpleNamespace(
        identity={
            "schema": "symmetrix.jit.host-policy/2",
            "requested": "automatic",
            "selected": "portable",
            "effective_target": "x86-64-v3",
            "flags": ["-fno-fast-math", "-ffp-contract=off"],
        }
    )

    provenance = runner._host_policy_provenance(
        ("c++",), {"resolved_executable": "/usr/bin/c++"}, policy
    )

    assert provenance["requested_floating_point_mode"] == "ieee"
    assert provenance["floating_point_mode"] == "ieee"


def test_host_policy_provenance_normalizes_effective_fp_mode(runner, monkeypatch):
    monkeypatch.delenv("SYMMETRIX_JIT_HOST_FLAGS", raising=False)
    monkeypatch.setenv("SYMMETRIX_JIT_HOST_FP_MODE", "IEEE")
    policy = SimpleNamespace(
        identity={
            "schema": "symmetrix.jit.host-policy/2",
            "requested": "automatic",
            "selected": "native",
            "effective_target": "host-sha256-" + "b" * 64,
            "flags": ["-fno-fast-math", "-ffp-contract=off", "-march=native"],
        }
    )

    provenance = runner._host_policy_provenance(
        ("c++",), {"resolved_executable": "/usr/bin/c++"}, policy
    )

    assert provenance["requested_floating_point_mode"] == "IEEE"
    assert provenance["floating_point_mode"] == "ieee"


class _RouteEvaluator:
    def __init__(
        self,
        implementation,
        loader_implementation,
        artifact_id=None,
        artifact_origin=None,
    ):
        self.r0_implementation = implementation
        self._r0_loader_implementation = loader_implementation
        self.r0_backend = "host"
        self.r0_artifact_origin = artifact_origin
        self.r0_schedule = (
            "host-owner" if loader_implementation == "host_plugin" else "frozen"
        )
        self.r0_profile = (
            "owner-batched"
            if loader_implementation == "host_plugin"
            else "fixed-weight-coordinate"
        )
        self.r0_module_id = artifact_id or "builtin-r0"
        self.r0_host_plugin_ready = loader_implementation == "host_plugin"
        self.r0_host_plugin_artifact_id = artifact_id


def test_runner_rejects_generic_or_wrong_generated_r0_routes(runner):
    with pytest.raises(RuntimeError, match="expected builtin route.*observed generic"):
        runner._assert_r0_route(
            _RouteEvaluator("generated", "generic"),
            implementation="frozen",
            expected_artifact_id=None,
        )

    with pytest.raises(RuntimeError, match="artifact ID mismatch"):
        runner._assert_r0_route(
            _RouteEvaluator("generated", "host_plugin", "wrong", "runtime"),
            implementation="generated",
            expected_artifact_id="expected",
        )

    runner._assert_r0_route(
        _RouteEvaluator("generated", "builtin"),
        implementation="frozen",
        expected_artifact_id="builtin-r0",
    )
    runner._assert_r0_route(
        _RouteEvaluator("generated", "host_plugin", "expected", "runtime"),
        implementation="generated",
        expected_artifact_id="expected",
    )


def test_each_path_gets_distinct_first_and_steady_callbacks(runner):
    requested_paths = []

    class Evaluator:
        standard_r0_module_launch_count = 0
        standard_r0_density_scale_fused = True
        harmonic_gradient_bytes = 0
        execution_direct_harmonic_launch_count = 1
        standard_m0_input_scale_adjoint_launch_count = 1
        r0_implementation = "generated"
        _r0_loader_implementation = "builtin"
        r0_backend = "host"
        r0_artifact_origin = None
        r0_schedule = "frozen"
        r0_profile = "fixed-weight-coordinate"
        r0_module_id = "builtin-r0"

    def make_evaluator(path):
        requested_paths.append(path)
        evaluator = Evaluator()

        def call():
            evaluator.standard_r0_module_launch_count += 4
            return path

        return evaluator, call

    baseline_case = runner.BaselineCase(
        "cpu", "float64", "unity", "small", "density_scaling"
    )
    record, output = runner._measure_case(
        make_evaluator=make_evaluator,
        implementation="frozen",
        case=baseline_case,
        artifact_preparation_ms=0.0,
        cache_root=Path("/task-cache"),
        workload={
            "model": "model.json",
            "model_sha256": "sha256:" + "a" * 64,
            "atoms": 4,
            "directed_edges": 80,
            "cutoff_A": 6.0,
            "skin_A": 0.0,
            "effective_cutoff_A": 6.0,
        },
        provenance={
            "source_commit": "0123456789abcdef",
            "source_dirty": True,
            "source_content_sha256": "sha256:" + "b" * 64,
            "native_build_source_content_sha256": "sha256:" + "c" * 64,
            "extension_sha256": "sha256:" + "d" * 64,
            "hardware": "test-cpu",
            "threads": {"kokkos": 1, "openmp": 1, "blas": 1},
            "benchmark_command": "python benchmark.py",
            "generated_r0_artifact": {
                "artifact_id": "r0_host_test",
                "path": "/task-cache/r0_host_test.so",
                "sha256": "sha256:" + "e" * 64,
            },
        },
        warmups=0,
        samples=2,
    )

    assert requested_paths == [
        "energy",
        "forces",
        "stress",
        "density_scaling",
    ]
    assert output == "density_scaling"
    assert record["case"]["path"] == "density_scaling"
    assert record["launch_count"] == 20
    assert len(record["steady_state"]["samples_ms"]) == 2
