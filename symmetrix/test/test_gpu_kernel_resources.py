import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "benchmarks" / "gpu_kernel_resources.py"


@pytest.fixture(scope="module")
def resources():
    specification = importlib.util.spec_from_file_location(
        "symmetrix_gpu_kernel_resources_test", SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_hip_metadata_parses_and_applies_scoped_limits(resources):
    text = """
amdhsa.kernels:
  - .name: conditioner_reverse
    .private_segment_fixed_size: 976
    .group_segment_fixed_size: 0
    .sgpr_count: 96
    .sgpr_spill_count: 383
    .vgpr_count: 192
    .vgpr_spill_count: 80
  - .name: edge_reverse
    .private_segment_fixed_size: 0
    .group_segment_fixed_size: 13584
    .sgpr_count: 102
    .sgpr_spill_count: 0
    .vgpr_count: 95
    .vgpr_spill_count: 0
"""
    report = resources.summarize(
        "hip",
        text,
        patterns=("conditioner",),
        limits={"private_bytes": 0, "vgpr_spills": 0},
    )

    assert report["selected_kernel_count"] == 1
    assert report["kernels"][0]["registers"] == 192
    assert {item["field"] for item in report["violations"]} == {
        "private_bytes",
        "vgpr_spills",
    }


def test_cuda_resource_usage_normalizes_stack_and_local(resources):
    text = """
Resource usage:
 Common:
  GLOBAL:0 CONSTANT[3]:72
 Function conditioner_reverse:
  REG:255 STACK:2144 SHARED:1024 LOCAL:16 CONSTANT[0]:1064
 Function edge_reverse:
  REG:72 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1064
"""
    report = resources.summarize("cuda", text, patterns=("conditioner",))

    kernel = report["kernels"][0]
    assert kernel["registers"] == 255
    assert kernel["stack_bytes"] == 2144
    assert kernel["private_bytes"] == 2160


def test_missing_match_and_incomplete_metadata_are_errors(resources):
    complete = """
  - .name: only_kernel
    .private_segment_fixed_size: 0
    .group_segment_fixed_size: 0
    .sgpr_count: 1
    .sgpr_spill_count: 0
    .vgpr_count: 1
    .vgpr_spill_count: 0
"""
    with pytest.raises(resources.ResourceReportError, match="no kernel matched"):
        resources.summarize("hip", complete, patterns=("absent",))
    with pytest.raises(resources.ResourceReportError, match="missing HIP"):
        resources.parse_hip_metadata("  - .name: incomplete\n    .vgpr_count: 4")


def test_cli_returns_one_for_limit_violation(resources, tmp_path):
    report = tmp_path / "cuda.txt"
    report.write_text(
        " Function hot_kernel:\n  REG:96 STACK:768 SHARED:0 LOCAL:0\n",
        encoding="utf-8",
    )

    assert (
        resources.main(
            ["cuda", str(report), "--match", "hot", "--max-stack-bytes", "0"]
        )
        == 1
    )
