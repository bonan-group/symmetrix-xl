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
    ):
        assert required in routine


def test_exporter_and_compatibility_lanes_remain_separate():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    assert "  metatrain-mace:\n" in workflow
    assert "  extract-mace-model:\n" in workflow
    assert "  pair_symmetrix:\n" in workflow
    fixture = (ROOT / ".github/workflows/polar-fixture-only.yaml").read_text()
    assert "pull_request:" in fixture and "paths:" in fixture
