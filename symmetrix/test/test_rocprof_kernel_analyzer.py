import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
ANALYZER_PATH = REPOSITORY / "benchmarks" / "analyze_rocprof_kernels.py"


@pytest.fixture(scope="module")
def analyzer():
    specification = importlib.util.spec_from_file_location(
        "symmetrix_rocprof_kernel_analyzer_test", ANALYZER_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def test_stats_csv_parses_quoted_template_names_and_aggregates(analyzer, tmp_path):
    path = tmp_path / "kernel_stats.csv"
    _write_csv(
        path,
        ["Name", "Calls", "TotalDurationNs", "AverageNs"],
        [
            {
                "Name": "symmetrix_execution_mh1_source_reverse_kernel_1",
                "Calls": 2,
                "TotalDurationNs": 3000,
                "AverageNs": 1500,
            },
            {
                "Name": "void at::native::vectorized_elementwise_kernel<int, float>()",
                "Calls": 5,
                "TotalDurationNs": 1000,
                "AverageNs": 200,
            },
        ],
    )

    summary = analyzer.summarize(path)

    assert summary["input_kind"] == "kernel_stats"
    assert summary["totals"] == {"calls": 7, "duration_ns": 4000}
    source = next(
        item for item in summary["categories"] if item["category"] == "source_reverse"
    )
    assert source["duration_ns"] == 3000
    assert source["duration_percent"] == pytest.approx(75.0)
    assert source["display_name"] == "R1 reverse: source"
    unknown = next(
        item for item in summary["kernels"] if item["category"] == "unclassified"
    )
    assert (
        unknown["name"]
        == "void at::native::vectorized_elementwise_kernel<int, float>()"
    )
    assert unknown["rule_id"] is None


def test_trace_csv_computes_duration_and_combines_dispatches(analyzer, tmp_path):
    path = tmp_path / "kernel_trace.csv"
    name = "symmetrix_execution_mh1_edge_reverse_phi_kernel_0"
    _write_csv(
        path,
        ["Kind", "Kernel_Name", "Start_Timestamp", "End_Timestamp"],
        [
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": name,
                "Start_Timestamp": 100,
                "End_Timestamp": 250,
            },
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": name,
                "Start_Timestamp": 300,
                "End_Timestamp": 500,
            },
        ],
    )

    summary = analyzer.summarize(path)

    assert summary["input_kind"] == "kernel_trace"
    assert summary["totals"] == {"calls": 2, "duration_ns": 350}
    assert summary["kernels"][0]["category"] == "edge_reverse_phi"
    assert summary["kernels"][0]["rule_id"] == "mh1.edge_reverse_phi"


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("symmetrix_execution_mh1_forward_kernel_0", "interaction_forward"),
        (
            "MaceNonlinearKokkos<float>::compute_Y(View<double const*>, bool)",
            "common_geometry_harmonics",
        ),
        ("mace.modules.ProductBasisBlock.forward", "node_forward"),
        ("e3nn spherical_harmonics kernel", "common_geometry_harmonics"),
        ("at::native::reduce_kernel<float>", "unclassified"),
        ("at::native::index_add_kernel<float>", "unclassified"),
        ("hipblasLtMatmul", "blas"),
        ("e3nn tensor_product_backward", "unclassified"),
        ("__amd_rocclr_fillBufferAligned", "common_runtime_copy_fill"),
        ("ZBLKokkos::compute_ZBL", "common_zbl"),
    ],
)
def test_classifier_uses_only_semantic_evidence(analyzer, name, category):
    assert analyzer.classify_kernel(name)[0] == category


@pytest.mark.parametrize(
    ("name", "category"),
    [
        (
            "symmetrix_execution_mh1_conditioning_forward_kernel_0",
            "conditioning_forward",
        ),
        (
            "symmetrix_execution_mh1_conditioning_reverse_kernel_1",
            "conditioning_reverse",
        ),
        (
            "symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_0",
            "edge_reverse_compact_fused",
        ),
        (
            "symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_1_wave64",
            "edge_reverse_compact_fused",
        ),
        ("symmetrix_execution_mh1_edge_reverse_phi_kernel_1", "edge_reverse_phi"),
        (
            "symmetrix_execution_mh1_edge_reverse_harmonic_kernel_0",
            "edge_reverse_harmonic",
        ),
        ("symmetrix_execution_mh1_node_pre_forward_kernel_0", "node_forward"),
        ("symmetrix_execution_mh1_node_tiled_l1_reverse_gate", "node_reverse"),
        (
            "symmetrix_execution_mh1_node_tiled_l0_reverse_readout",
            "readout_reduction",
        ),
    ],
)
def test_mh1_generated_phases_are_distinct(analyzer, name, category):
    assert analyzer.classify_kernel(name)[0] == category


@pytest.mark.parametrize(
    ("name", "stage"),
    [
        ("symmetrix/mh1/R0/forward", "r0_forward"),
        ("symmetrix/mh1/M0/forward", "m0_forward"),
        ("symmetrix/mh1/R1/reverse", "r1_reverse"),
        ("symmetrix/mh1/M1/reverse", "m1_reverse"),
        ("symmetrix_execution_mh1_conditioning_forward_kernel_0", "r0_forward"),
        (
            "symmetrix_execution_mh1_conditioning_reverse_prefix_kernel_0",
            "r0_reverse",
        ),
        (
            "symmetrix_execution_mh1_conditioning_reverse_density_kernel_1",
            "r1_reverse",
        ),
        ("symmetrix_execution_mh1_forward_kernel_1", "r1_forward"),
        ("symmetrix_execution_mh1_spline_r_forward_kernel_0", "r0_forward"),
        ("symmetrix_execution_mh1_spline_r_reverse_kernel_1", "r1_reverse"),
        ("symmetrix_execution_mh1_source_reverse_kernel_0", "r0_reverse"),
        (
            "symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_1_wave64",
            "r1_reverse",
        ),
        ("symmetrix_execution_mh1_node_pre_forward_kernel_0", "m0_forward"),
        ("symmetrix_execution_mh1_node_tiled_l1_forward_gate", "m1_forward"),
        ("symmetrix_execution_mh1_node_pre_reverse_kernel_0", "m0_reverse"),
        ("symmetrix_execution_mh1_node_tiled_l1_reverse_gate", "m1_reverse"),
        ("ZBLKokkos::compute_ZBL", None),
    ],
)
def test_mh1_kernels_map_to_cpu_stage_vocabulary(analyzer, name, stage):
    assert analyzer.classify_mh1_stage(name) == stage


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("symmetrix_execution_mh1_spline_r_forward_kernel_0", "r0_forward"),
        (
            "symmetrix_execution_mh1_spline_r_forward_kernel_1",
            "interaction_forward",
        ),
        ("symmetrix_execution_mh1_spline_r_reverse_kernel_0", "r0_reverse"),
        ("symmetrix_execution_mh1_spline_r_reverse_kernel_1", "r1_reverse"),
    ],
)
def test_mh1_spline_r_kernels_use_standard_stage_categories(analyzer, name, category):
    assert analyzer.classify_kernel(name)[0] == category


def test_summary_reports_canonical_mh1_stages(analyzer, tmp_path):
    path = tmp_path / "mh1_stage_trace.csv"
    _write_csv(
        path,
        ["Kind", "Kernel_Name", "Start_Timestamp", "End_Timestamp"],
        [
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": "symmetrix_execution_mh1_forward_kernel_0",
                "Start_Timestamp": 100,
                "End_Timestamp": 300,
            },
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": "symmetrix_execution_mh1_node_tiled_l0_forward_gate",
                "Start_Timestamp": 400,
                "End_Timestamp": 500,
            },
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": "ZBLKokkos::compute_ZBL",
                "Start_Timestamp": 600,
                "End_Timestamp": 700,
            },
        ],
    )

    summary = analyzer.summarize(path)

    assert summary["mh1_stages"]["available"] is True
    assert summary["mh1_stages"]["assigned_duration_ns"] == 300
    stages = {record["stage"]: record for record in summary["mh1_stages"]["stages"]}
    assert stages["r0_forward"]["duration_ns"] == 200
    assert stages["m0_forward"]["duration_ns"] == 100
    assert summary["mh1_stages"]["unassigned"]["duration_ns"] == 100


def test_marker_trace_resolves_nested_spline_kernels(analyzer, tmp_path):
    kernel_path = tmp_path / "kernel_trace.csv"
    marker_path = tmp_path / "marker_api_trace.csv"
    _write_csv(
        kernel_path,
        [
            "Kind",
            "Kernel_Name",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        ],
        [
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": "e3 tensor execution spline uvu receiver-owned forward",
                "Correlation_Id": 12,
                "Start_Timestamp": 1000,
                "End_Timestamp": 1300,
            }
        ],
    )
    _write_csv(
        marker_path,
        [
            "Domain",
            "Function",
            "Process_Id",
            "Thread_Id",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        ],
        [
            {
                "Domain": "MARKER_CORE_RANGE_API",
                "Function": "symmetrix/mh1/R1/forward",
                "Process_Id": 7,
                "Thread_Id": 8,
                "Correlation_Id": 10,
                "Start_Timestamp": 100,
                "End_Timestamp": 900,
            },
            {
                "Domain": "MARKER_CORE_RANGE_API",
                "Function": "symmetrix/mh1/execution_spline_uvu/forward_direct_nodes",
                "Process_Id": 7,
                "Thread_Id": 8,
                "Correlation_Id": 11,
                "Start_Timestamp": 200,
                "End_Timestamp": 800,
            },
            {
                "Domain": "MARKER_CORE_RANGE_API",
                "Function": "e3 tensor execution spline uvu receiver-owned forward",
                "Process_Id": 7,
                "Thread_Id": 8,
                "Correlation_Id": 12,
                "Start_Timestamp": 300,
                "End_Timestamp": 700,
            },
        ],
    )

    summary = analyzer.summarize(kernel_path, marker_path)

    stages = {record["stage"]: record for record in summary["mh1_stages"]["stages"]}
    assert stages["r1_forward"]["duration_ns"] == 300
    assert summary["mh1_stages"]["unassigned"]["duration_ns"] == 0
    assert summary["kernels"][0]["mh1_stage"] == "r1_forward"


def test_display_names_follow_standard_mace_stage_vocabulary(analyzer):
    assert analyzer.CATEGORY_DISPLAY_NAMES["interaction_forward"] == "R1 forward"
    assert (
        analyzer.CATEGORY_DISPLAY_NAMES["edge_reverse_compact_fused"]
        == "R1 reverse: edge (compact fused)"
    )
    assert analyzer.CATEGORY_DISPLAY_NAMES["node_forward"] == "M1/node forward"
    assert analyzer.CATEGORY_DISPLAY_NAMES["node_reverse"] == "M1/node reverse"


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("symmetrix_factorized_forward_v1", "interaction_forward"),
        ("symmetrix_factorized_reverse_fused_v1", "r1_reverse"),
        ("standard_r0::launch_forward<Kokkos::HIP>", "r0_forward"),
        ("standard_r0::launch_coordinate_reverse_edge_owned<16, 16>", "r0_reverse"),
        ("standard_m0::launch_forward<Kokkos::HIP>", "m0_forward"),
        ("standard_m0::launch_reverse<Kokkos::HIP>", "m0_reverse"),
        ("MACEKokkos<float>::compute_H1(int)", "h1_forward"),
        ("MACEKokkos<float>::compute_A1(int, bool)", "a1_forward"),
        ("MACEKokkos<float>::compute_M1(int, View<int>)", "m1_forward"),
        ("MACEKokkos<float>::compute_H2(int, View<int>)", "h2_forward"),
        ("MACEKokkos<float>::reverse_H2(int, View<int>, bool)", "h2_reverse"),
        ("standard_m1::launch_reverse_device_direct<Kokkos::HIP>", "m1_reverse"),
        ("MACEKokkos<float>::reverse_A1_from(int, View<float>)", "a1_reverse"),
        ("MACEKokkos<float>::reverse_H1(int)", "h1_reverse"),
        ("MultilayerPerceptronKokkos::evaluate_gradient", "readout_reduction"),
        (
            "Cijk_Ailk_Bljk_SB_MT32x32x8_ISA1151_WG16_16_1",
            "blas",
        ),
    ],
)
def test_standard_mace_stages_remain_distinct(analyzer, name, category):
    assert analyzer.classify_kernel(name)[0] == category


def test_compact_fused_edge_reverse_aggregates_in_display_order(analyzer, tmp_path):
    path = tmp_path / "compact_fused_stats.csv"
    first = "symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_0"
    second = "symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_1"
    _write_csv(
        path,
        ["Name", "Calls", "TotalDurationNs"],
        [
            {"Name": first, "Calls": 2, "TotalDurationNs": 300},
            {"Name": first, "Calls": 1, "TotalDurationNs": 50},
            {"Name": second, "Calls": 4, "TotalDurationNs": 400},
        ],
    )

    summary = analyzer.summarize(path)

    compact = next(
        item
        for item in summary["categories"]
        if item["category"] == "edge_reverse_compact_fused"
    )
    assert compact == {
        "category": "edge_reverse_compact_fused",
        "display_name": "R1 reverse: edge (compact fused)",
        "calls": 7,
        "duration_ns": 750,
        "kernel_count": 2,
        "duration_percent": 100.0,
    }
    assert all(
        item["rule_id"] == "mh1.edge_reverse_compact_fused"
        for item in summary["kernels"]
    )
    categories = [item["category"] for item in summary["categories"]]
    assert (
        categories.index("source_reverse")
        < categories.index("edge_reverse_compact_fused")
        < categories.index("edge_reverse_phi")
    )


def test_rejects_malformed_trace(analyzer, tmp_path):
    path = tmp_path / "bad.csv"
    _write_csv(
        path,
        ["Kernel_Name", "Start_Timestamp", "End_Timestamp"],
        [{"Kernel_Name": "kernel", "Start_Timestamp": 20, "End_Timestamp": 10}],
    )

    with pytest.raises(analyzer.AnalysisError, match="precedes"):
        analyzer.summarize(path)


def test_cli_writes_json_and_prints_table(analyzer, tmp_path, capsys):
    path = tmp_path / "stats.csv"
    output = tmp_path / "summary.json"
    _write_csv(
        path,
        ["Name", "Calls", "TotalDurationNs"],
        [{"Name": "unknown, templated<one, two>", "Calls": 1, "TotalDurationNs": 12}],
    )

    assert analyzer.main([str(path), "--json-output", str(output)]) == 0

    assert "Unclassified" in capsys.readouterr().out
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == "symmetrix.rocprof-kernel-summary"
    assert document["kernels"][0]["name"] == "unknown, templated<one, two>"
