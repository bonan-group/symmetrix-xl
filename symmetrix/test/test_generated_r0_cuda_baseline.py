import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
SCRIPT = BENCHMARKS / "run_generated_r0_cuda_baseline.py"


@pytest.fixture(scope="module")
def runner():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec = importlib.util.spec_from_file_location(
            "symmetrix_generated_r0_cuda_runner_test", SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(BENCHMARKS))


def test_runner_has_a_repository_cuda_campaign_entrypoint(runner):
    assert callable(runner.run_cuda_campaign)
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


class _RouteEvaluator:
    def __init__(self, route, artifact_id, origin=None, schedule="receiver-edge16"):
        self.r0_implementation = "generated"
        self._r0_loader_implementation = route
        self.r0_backend = "cuda"
        self.r0_artifact_origin = origin
        self.r0_schedule = "frozen" if route == "builtin" else schedule
        self.r0_profile = (
            "fixed-weight-coordinate" if route == "builtin" else "persistent-3"
        )
        self.r0_module_id = artifact_id
        self.r0_device_module_ready = route == "device_module"
        self.r0_device_module_artifact_id = artifact_id


def test_runner_rejects_wrong_cuda_route_and_accepts_matching_routes(runner):
    runner._assert_r0_route(
        _RouteEvaluator("builtin", "builtin-r0"),
        implementation="frozen",
        expected_artifact_id="builtin-r0",
        schedule="receiver-edge16",
        persistent_blocks=3,
    )
    runner._assert_r0_route(
        _RouteEvaluator("device_module", "generated-r0", "runtime", "receiver-edge32"),
        implementation="generated",
        expected_artifact_id="generated-r0",
        schedule="receiver-edge32",
        persistent_blocks=3,
    )
    with pytest.raises(RuntimeError, match="expected device_module route"):
        runner._assert_r0_route(
            _RouteEvaluator("builtin", "builtin-r0"),
            implementation="generated",
            expected_artifact_id="generated-r0",
            schedule="receiver-edge16",
            persistent_blocks=3,
        )


def test_runner_uses_production_cuda_schedule_selection(runner, monkeypatch):
    monkeypatch.delenv("SYMMETRIX_R0_RTC_SCHEDULE", raising=False)
    assert runner.resolve_r0_device_schedule() == "receiver-edge16"
    monkeypatch.setenv("SYMMETRIX_R0_RTC_SCHEDULE", "receiver-edge32")
    assert runner.resolve_r0_device_schedule() == "receiver-edge32"

    source = SCRIPT.read_text()

    assert "from symmetrix.mace_jit_operator_artifact import (" in source
    assert "resolve_r0_device_schedule" in source
    assert '("float32", "MACEKokkosFloat")' in source
    assert '("float64", "MACEKokkos")' in source
    assert '("float64", "MACEKokkos", "receiver-edge32")' not in source
    assert "prepare_nvrtc_jit_artifact" in source
    assert "source_content_sha256=source_content_sha256" in source
    assert 'backends=("cuda",)' in source


def test_runner_uses_the_floating_point_policy_declared_by_generated_r0(runner):
    assert runner._r0_nvrtc_options({"floating_point_policy": "cuda-fast-math"}) == (
        "--split-compile=2",
        "--use_fast_math",
    )
    assert runner._r0_nvrtc_options({"floating_point_policy": "strict"}) == (
        "--split-compile=2",
    )

    with pytest.raises(ValueError, match="unsupported R0 floating-point policy"):
        runner._r0_nvrtc_options({"floating_point_policy": "unknown"})


def test_reproducible_command_captures_unset_and_explicit_cuda_schedule(
    runner, monkeypatch
):
    for name in runner._THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SYMMETRIX_BACKEND", "cuda13-sm120")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", "/task-cache")
    monkeypatch.delenv("SYMMETRIX_R0_RTC_SCHEDULE", raising=False)

    command = runner._reproducible_benchmark_command([])

    assert "-u SYMMETRIX_R0_RTC_SCHEDULE" in command

    monkeypatch.setenv("SYMMETRIX_R0_RTC_SCHEDULE", "receiver-edge32")
    command = runner._reproducible_benchmark_command([])

    assert "SYMMETRIX_R0_RTC_SCHEDULE=receiver-edge32" in command
    assert "-u SYMMETRIX_R0_RTC_SCHEDULE" not in command
