import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/low_memory_capacity.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("low_memory_capacity_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark():
    return _load_benchmark()


def test_selected_gpu_selector_maps_cuda_visible_devices(benchmark, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first, 7, MIG-third")

    assert benchmark._selected_gpu_selector(0) == "GPU-first"
    assert benchmark._selected_gpu_selector(1) == "7"
    assert benchmark._selected_gpu_selector(2) == "MIG-third"
    assert benchmark._selected_gpu_selector(2, "GPU-explicit") == "GPU-explicit"
    with pytest.raises(RuntimeError, match="absent from CUDA_VISIBLE_DEVICES"):
        benchmark._selected_gpu_selector(3)


def test_gpu_device_memory_reads_total_device_usage(benchmark, monkeypatch):
    def run(command, **kwargs):
        assert command[:3] == ["nvidia-smi", "-i", "GPU-test"]
        assert "--query-gpu=index,uuid,name,memory.total,memory.used" in command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="2, GPU-uuid, NVIDIA A100-SXM4-80GB, 81920, 81047\n",
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)

    assert benchmark._gpu_device_memory("GPU-test") == {
        "selector": "GPU-test",
        "physical_index": 2,
        "uuid": "GPU-uuid",
        "name": "NVIDIA A100-SXM4-80GB",
        "total_mib": 81920,
        "used_mib": 81047,
    }


@pytest.mark.parametrize(
    "stderr,expected",
    (
        (
            "first: CUDA error cudaErrorIllegalAddress\n"
            "finalize: CUDA memory space failed to allocate",
            "cuda_illegal_address",
        ),
        (
            "first: CUDA memory space failed to allocate 2 GiB\n"
            "finalize: illegal memory access",
            "cuda_oom",
        ),
    ),
)
def test_failure_classification_preserves_first_cuda_exception(
    benchmark, stderr, expected
):
    assert benchmark._failure_classification(1, stderr) == expected


def test_only_cuda_oom_is_a_capacity_boundary(benchmark):
    benchmark._require_cuda_oom_boundary("n64", "cuda_oom")
    with pytest.raises(RuntimeError, match="only cuda_oom is a valid"):
        benchmark._require_cuda_oom_boundary("n48", "cuda_illegal_address")


def test_gpu_memory_recovery_waits_for_baseline(benchmark, monkeypatch):
    samples = iter(
        (
            {"used_mib": 1024},
            {"used_mib": 120},
        )
    )
    monkeypatch.setattr(benchmark, "_gpu_device_memory", lambda selector: next(samples))
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: None)

    recovered = benchmark._wait_for_gpu_memory_recovery("0", 100, 1.0, 32)

    assert recovered == {"used_mib": 120}


def test_gpu_memory_sampler_requires_an_initial_device_sample(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark, "_gpu_device_memory", lambda selector: None)
    sampler = benchmark._GpuMemorySampler("GPU-missing")

    with pytest.raises(RuntimeError, match="could not query CUDA device GPU-missing"):
        sampler.start()

    assert sampler._thread is None


def test_worker_parser_exposes_explicit_mh1_node_state_policy(benchmark):
    args = benchmark._parser().parse_args(
        [
            "worker",
            "--model-kind",
            "mh1",
            "--model",
            "mh1.json",
            "--mh1-node-state-policy",
            "recompute-v1",
            "--mh1-edge-executor",
            "pair_spline_v1",
            "--repeat",
            "6",
            "--output",
            "result.json",
        ]
    )

    assert args.low_memory is None
    assert args.mh1_node_state_policy == "recompute-v1"
    assert args.mh1_edge_executor == "pair_spline_v1"


def test_worker_parser_exposes_reused_mh1_adjoint_policy(benchmark):
    args = benchmark._parser().parse_args(
        [
            "worker",
            "--model-kind",
            "mh1",
            "--model",
            "mh1.json",
            "--mh1-node-state-policy",
            "reuse-adjoints-v1",
            "--repeat",
            "6",
            "--output",
            "result.json",
        ]
    )

    assert args.mh1_node_state_policy == "reuse-adjoints-v1"


def test_campaign_model_kinds_remain_omat0_and_macefield(benchmark):
    assert benchmark.CAMPAIGN_MODEL_KINDS == ("omat0", "macefield")


def test_mh1_worker_policy_is_not_encoded_as_low_memory(benchmark):
    args = benchmark._parser().parse_args(
        [
            "worker",
            "--model-kind",
            "mh1",
            "--model",
            "mh1.json",
            "--mh1-node-state-policy",
            "full-retention-v1",
            "--repeat",
            "6",
            "--output",
            "result.json",
        ]
    )

    assert args.low_memory is None
