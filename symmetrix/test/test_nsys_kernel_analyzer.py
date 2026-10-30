import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "benchmarks" / "analyze_nsys_kernels.py"


@pytest.fixture(scope="module")
def analyzer():
    spec = importlib.util.spec_from_file_location("nsys_les_analyzer", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["Name", "Instances", "Total Time (ns)"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_nsys_summary_reports_semantic_les_rates(analyzer, tmp_path):
    path = tmp_path / "cuda_gpu_kern_sum.csv"
    _write(
        path,
        [
            {"Name": "LES reciprocal table", "Instances": 2, "Total Time (ns)": 200},
            {"Name": "LES structure factors", "Instances": 4, "Total Time (ns)": 800},
            {
                "Name": "LES periodic atom outputs",
                "Instances": 4,
                "Total Time (ns)": 1200,
            },
            {
                "Name": "LES analytic reciprocal strain",
                "Instances": 4,
                "Total Time (ns)": 400,
            },
            {
                "Name": "LES charge readout evaluate",
                "Instances": 8,
                "Total Time (ns)": 800,
            },
            {"Name": "LES compose forces", "Instances": 4, "Total Time (ns)": 200},
        ],
    )
    summary = analyzer.summarize(path, evaluation_count=4)
    assert summary["selected_evaluation_count"] == 4
    categories = {item["category"]: item for item in summary["les_stage_categories"]}
    assert categories["structure_factors"]["calls_per_evaluation"] == 1
    assert categories["atom_outputs"]["duration_ns_per_evaluation"] == 300
    assert "assemble_maceles" in categories
    assert categories["assemble_maceles"]["display_name"] == "assemble MACELES"
    assert categories["structure_factors"]["display_name"].startswith("LES stage:")


def test_nsys_rejects_dispersion_in_les_only_region(analyzer, tmp_path):
    path = tmp_path / "cuda_gpu_kern_sum.csv"
    _write(path, [{"Name": "D3 pair kernel", "Instances": 1, "Total Time (ns)": 20}])
    with pytest.raises(analyzer.AnalysisError, match="dispersion"):
        analyzer.summarize(path, evaluation_count=1)


@pytest.mark.parametrize(
    "rows",
    [
        [{"Name": "unrelated kernel", "Instances": 1, "Total Time (ns)": 20}],
        [
            {"Name": "LES structure factors", "Instances": 1, "Total Time (ns)": 20},
            {
                "Name": "LES periodic atom outputs",
                "Instances": 1,
                "Total Time (ns)": 20,
            },
        ],
    ],
)
def test_nsys_rejects_missing_steady_state_les_categories(analyzer, tmp_path, rows):
    path = tmp_path / "incomplete.csv"
    _write(path, rows)
    with pytest.raises(analyzer.AnalysisError, match="missing.*LES categories"):
        analyzer.summarize(path, evaluation_count=1)
