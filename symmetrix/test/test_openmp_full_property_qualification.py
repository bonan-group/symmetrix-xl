import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "openmp_full_property_qualification.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("openmp_qualification", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_property_comparison_reports_each_property():
    module = _module()
    reference = {name: [1.0, -2.0] for name in module.PROPERTIES}
    candidate = {name: [1.0, -2.0 + 1e-12] for name in module.PROPERTIES}

    passed, differences = module._compare_results(
        reference, candidate, atol=1e-10, rtol=0.0
    )

    assert passed
    assert set(differences) == set(module.PROPERTIES)


def test_full_property_comparison_rejects_nonfinite_and_extent_changes():
    module = _module()
    reference = {name: [1.0] for name in module.PROPERTIES}
    candidate = {name: [1.0] for name in module.PROPERTIES}
    candidate["forces"] = [float("nan")]
    candidate["becs"] = [[1.0]]

    passed, differences = module._compare_results(
        reference, candidate, atol=1e-10, rtol=0.0
    )

    assert not passed
    assert not differences["forces"]["finite"]
    assert differences["becs"]["reason"] == "shape differs"
    assert differences["becs"]["reference_shape"] == (1,)
    assert differences["becs"]["candidate_shape"] == (1, 1)


def test_cpu_set_parser_supports_ranges_and_rejects_descending_ranges():
    module = _module()

    assert module._parse_cpu_set("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert module._parse_cpu_set(None) is None

    try:
        module._parse_cpu_set("4-2")
    except ValueError as exc:
        assert "invalid CPU set" in str(exc)
    else:
        raise AssertionError("descending CPU range was accepted")


def test_speedup_gate_rejects_effective_single_thread_execution():
    module = _module()

    failed = module._speedup_summary(
        100.0,
        95.0,
        reference_threads=1,
        candidate_threads=8,
        minimum=1.25,
    )
    passed = module._speedup_summary(
        100.0,
        20.0,
        reference_threads=1,
        candidate_threads=8,
        minimum=1.25,
    )

    assert failed["speedup"] < failed["minimum"]
    assert not failed["passed"]
    assert passed["speedup"] == 5.0
    assert passed["passed"]


def test_qualification_defaults_cover_modes_threads_and_m0_tiles():
    module = _module()

    args = module._parser().parse_args(
        ["--model", "model.json", "--structure", "structure.extxyz"]
    )

    assert tuple(args.threads) == (1, 2, 4, 8)
    assert tuple(args.modes) == ("generic", "direct")
    assert tuple(args.m0_tiles) == (1, 4, 8, 16)
    assert args.min_speedup == 1.25
