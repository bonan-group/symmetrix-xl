from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_routine_symmetrix_job_is_fixture_only_and_no_torch():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    routine = workflow.split("  symmetrix:\n", 1)[1].split(
        "  metatrain-mace:\n", 1
    )[0]
    for forbidden in (
        "torch==",
        "mace-torch",
        "download.pytorch.org",
        "curl",
        "wget",
        "MACE-MH-1",
    ):
        assert forbidden not in routine
    for required in (
        "test_polar_oracle_fixtures.py",
        "test_polar_end_to_end_fixtures.py",
        "test_mace_les_reference.py",
        "importlib.util.find_spec",
        "symmetrix-cpu-build",
        "SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256",
        '("torch", "mace", "graph_longrange")',
        "pytest -q -m 'not extended' symmetrix/test",
        "--ignore=symmetrix/test/test_extract_mace_data.py",
        "--ignore=symmetrix/test/test_mace_streamed_edges.py",
        "--ignore=symmetrix/test/test_polar_torch_baseline.py",
        "--ignore=symmetrix/test/test_single_layer_mace.py",
        "--ignore=symmetrix/test/test_jit_kokkos_integration.py",
    ):
        assert required in routine
    assert "pytest -q \\\n          symmetrix/test/test_polar_oracle_fixtures.py" in routine


def test_exporter_and_compatibility_lanes_remain_separate():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    assert "  metatrain-mace:\n" in workflow
    assert "  extract-mace-model:\n" in workflow
    assert "  pair_symmetrix:\n" in workflow
    fixture = (ROOT / ".github/workflows/polar-fixture-only.yaml").read_text()
    assert "pull_request:" in fixture and "paths:" in fixture


def test_routine_lane_keeps_native_modules_and_fails_closed_on_downloads():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    routine = workflow.split("  symmetrix:\n", 1)[1].split(
        "  metatrain-mace:\n", 1
    )[0]
    for module in (
        "test_symmetrix_calc.py",
        "test_family_registry.py",
        "test_family_mace_conformance_cases.py",
        "test_macefield_field_transform.py",
        "test_macefield_native.py",
    ):
        assert f"--ignore=symmetrix/test/{module}" not in routine
    source = (ROOT / "symmetrix/test/model_downloads.py").read_text()
    assert 'SYMMETRIX_TEST_MODEL_DOWNLOADS"' in source
    assert "ModelDownloadDisabledError" in source
