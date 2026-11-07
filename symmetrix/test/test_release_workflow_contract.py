from pathlib import Path
import hashlib
import json
import zipfile

import pytest

from tools import audit_cuda_release_wheel


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
    assert "name: cuda${{ matrix.cuda_major }}-${{ matrix.target }}-audit" in workflow
    assert "path: wheelhouse/release/*.whl" in workflow
    assert "path: audit" in workflow
    assert "manifest=audit/SHA256SUMS" in workflow
    assert "packages-dir: wheelhouse" in workflow
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


def _cuda_wheel(tmp_path, *, descriptor=None, native=True, catalog=True, oam=True):
    package = "symmetrix_backend_cuda13_sm120"
    descriptor = descriptor or {
        "distribution": "symmetrix-xl-cuda13-sm120",
        "backend": "cuda",
        "toolkit": "cuda13",
        "selector": "cuda13-sm120",
        "architecture": "sm120",
        "source_content_sha256": "a" * 64,
        "native_source_content_sha256": "b" * 64,
    }
    artifacts = []
    if oam:
        artifacts.append(
            {
                "operator": "OAM",
                "build_provenance": {
                    "source_content_sha256": "sha256:" + "a" * 64,
                    "native_source_content_sha256": "sha256:" + "b" * 64,
                },
            }
        )
    payload = {
        "artifacts": artifacts,
    }
    descriptor["prebuilt_operator_artifacts"] = {
        "backend": "cuda",
        "target": "sm120",
        "artifact_count": len(artifacts),
        "catalog_sha256": hashlib.sha256(
            json.dumps(payload).encode()
        ).hexdigest(),
    }
    wheel = tmp_path / "wheel.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{package}/backend.json", json.dumps(descriptor))
        if native:
            archive.writestr(f"{package}/_native_cuda13_sm120.so", b"native")
        if catalog:
            archive.writestr(
                f"{package}/_operator_artifacts/catalog.json",
                json.dumps(payload),
            )
    return wheel


@pytest.fixture(autouse=True)
def _skip_common_wheel_audit(monkeypatch):
    monkeypatch.setattr(audit_cuda_release_wheel, "audit_wheel", lambda wheel: None)


def test_cuda_audit_accepts_valid_synthetic_wheel(tmp_path):
    result = audit_cuda_release_wheel.audit_cuda_wheel(
        _cuda_wheel(tmp_path), cuda_major="13", target="sm120"
    )
    assert result["selector"] == "cuda13-sm120"


@pytest.mark.parametrize(
    "mutator",
    (
        lambda descriptor: descriptor.update(selector="cuda13-sm90"),
        lambda descriptor: descriptor.update(source_content_sha256="c" * 64),
    ),
)
def test_cuda_audit_rejects_descriptor_or_provenance_mismatch(tmp_path, mutator):
    descriptor = {
        "distribution": "symmetrix-xl-cuda13-sm120",
        "backend": "cuda",
        "toolkit": "cuda13",
        "selector": "cuda13-sm120",
        "architecture": "sm120",
        "source_content_sha256": "a" * 64,
        "native_source_content_sha256": "b" * 64,
    }
    mutator(descriptor)
    with pytest.raises(Exception):
        audit_cuda_release_wheel.audit_cuda_wheel(
            _cuda_wheel(tmp_path, descriptor=descriptor),
            cuda_major="13",
            target="sm120",
        )


@pytest.mark.parametrize("kwargs", ({"native": False}, {"catalog": False}, {"oam": False}))
def test_cuda_audit_rejects_missing_release_payload(tmp_path, kwargs):
    with pytest.raises(Exception):
        audit_cuda_release_wheel.audit_cuda_wheel(
            _cuda_wheel(tmp_path, **kwargs), cuda_major="13", target="sm120"
        )
