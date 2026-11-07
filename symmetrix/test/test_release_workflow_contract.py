from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_cuda_release_audits_before_upload_and_publication():
    workflow = (ROOT / ".github/workflows/release-cuda-wheels.yaml").read_text()
    assert "tools/audit_cuda_release_wheel.py" in workflow
    assert "auditwheel show" in workflow
    assert "manylinux_2_28" in workflow
    assert "pip install --no-deps" in workflow
    assert "entry_points" in workflow
    assert "_operator_artifacts/catalog.json" in workflow
    assert "sha256sum *.whl" in workflow
    assert "sha256sum -c" in workflow
    assert workflow.index("Audit repaired CUDA wheels") < workflow.index(
        "Upload wheel and audit manifest artifacts"
    )
    assert "self-hosted" not in workflow


def test_cuda_release_keeps_declared_matrix_and_metadata_gate():
    workflow = (ROOT / ".github/workflows/release-cuda-wheels.yaml").read_text()
    for target in ("sm70", "sm80", "sm86", "sm89", "sm90", "sm100", "sm120"):
        assert target in workflow
    assert "python tools/check_release_metadata.py" in workflow
    assert "python tools/check_release_metadata.py --require-tag" in workflow
    assert "publish_pypi" in workflow


def test_cpu_release_compatibility_lane_is_pinned_and_downloads_fail_closed():
    workflow = (ROOT / ".github/workflows/release-cpu-wheels.yaml").read_text()
    assert "'torch==2.5.*'" in workflow
    assert "'mace-torch==0.3.16'" in workflow
    assert 'SYMMETRIX_TEST_MODEL_DOWNLOADS: "0"' in workflow
    assert "qualify-full" in workflow
