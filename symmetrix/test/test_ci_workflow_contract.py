from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_routine_symmetrix_job_is_fixture_only_and_no_torch():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    routine = workflow.split("  symmetrix:\n", 1)[1].split("  metatrain-mace:\n", 1)[0]
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
        "--ignore=symmetrix/test/test_jit_device_artifact.py",
        "--ignore=symmetrix/test/test_mace_les_readout.py",
    ):
        assert required in routine
    assert (
        "pytest -q \\\n          symmetrix/test/test_polar_oracle_fixtures.py"
        in routine
    )


def test_exporter_and_compatibility_lanes_remain_separate():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    assert "  metatrain-mace:\n" in workflow
    assert "  extract-mace-model:\n" in workflow
    assert "  pair_symmetrix:\n" in workflow
    fixture = (ROOT / ".github/workflows/polar-fixture-only.yaml").read_text()
    assert "pull_request:" in fixture and "paths:" in fixture


def test_routine_lane_keeps_native_modules_and_fails_closed_on_downloads():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    routine = workflow.split("  symmetrix:\n", 1)[1].split("  metatrain-mace:\n", 1)[0]
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


def test_extraction_uses_verified_pinned_checkpoint_without_weakening_parity():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    extraction = workflow.split("  extract-mace-model:\n", 1)[1]
    checksum = "165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f"
    for required in (
        f"key: mace-off23-small-{checksum}",
        f"checksum={checksum}",
        "91a78c5a9c300d1104700d9352c8bfe449227737/mace_off23/",
        "--retry 5 --retry-all-errors",
        "--retry-max-time 600",
        'echo "$checksum  $model.part" | sha256sum --check --strict',
        'mv "$model.part" "$model"',
        'echo "$checksum  $model" | sha256sum --check --strict',
        "symmetrix_extract_mace --model MACE-OFF23_small.model",
        "--chemical-symbols H O -o extract_test.json",
        '("float64", 1e-10, 1e-9)',
        '("float32", 1e-4, 1e-4)',
        'for routing in ("direct", "reference"):',
        "universal_atoms.get_potential_energy()",
        "universal_atoms.get_forces()",
    ):
        assert required in extraction
    assert "refs/heads/main/mace_off23" not in extraction
    download = extraction.split(
        "    - name: Download and verify pinned MACE-OFF23 checkpoint\n", 1
    )[1].split("    - name: Test model extraction\n", 1)[0]
    assert "continue-on-error" not in download
    assert "|| true" not in download


def test_torch_dependent_modules_run_in_macefield_lane():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
    macefield = workflow.split(
        "    - name: Test compact MACEField conversion and ASE integration\n", 1
    )[1].split("    - name: Cache MACE-MH-1 checkpoint\n", 1)[0]
    assert "if: matrix.mace_field_compatible" in macefield
    assert "test_jit_device_artifact.py" in macefield
    assert "test_mace_les_readout.py" in macefield
