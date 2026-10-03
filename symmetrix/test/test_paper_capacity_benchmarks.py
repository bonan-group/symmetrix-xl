import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CAPACITY_SCRIPT = ROOT / "benchmarks/srtio3_capacity_probe.py"
PLAN_SCRIPT = ROOT / "benchmarks/a100_srtio3_plan_benchmark.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def capacity_driver():
    return _load(CAPACITY_SCRIPT, "srtio3_capacity_probe_test")


@pytest.fixture(scope="module")
def plan_driver():
    return _load(PLAN_SCRIPT, "a100_srtio3_plan_benchmark_test")


@pytest.mark.parametrize(
    "driver_name", ("srtio3_capacity_probe.py", "a100_srtio3_plan_benchmark.py")
)
def test_paper_capacity_driver_help(driver_name, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / driver_name), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("parser_name", ("parse_axes", "_parse_axes"))
def test_paper_capacity_axis_parser(parser_name, capacity_driver, plan_driver):
    driver = capacity_driver if parser_name == "parse_axes" else plan_driver
    parser = getattr(driver, parser_name)
    assert parser("6,6,6") == (6, 6, 6)
    with pytest.raises(argparse.ArgumentTypeError):
        parser("0,2,3")


def test_capacity_structure_is_cubic_srtio3(capacity_driver):
    atoms = capacity_driver.build_structure((1, 1, 1))
    assert len(atoms) == 5
    assert atoms.get_chemical_formula() == "O3SrTi"
    assert list(atoms.cell.lengths()) == pytest.approx([3.905] * 3)
    assert all(atoms.pbc)


def test_plan_structure_matches_capacity_workload(plan_driver):
    atoms = plan_driver._structure((1, 1, 1))
    assert len(atoms) == 5
    assert atoms.get_chemical_formula() == "O3SrTi"
    assert list(atoms.cell.lengths()) == pytest.approx([3.905] * 3)
    assert all(atoms.pbc)
