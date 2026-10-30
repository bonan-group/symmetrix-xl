"""Release audit must reject both copied and indirectly linked vendor runtimes."""

import importlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
audit = importlib.import_module("_symmetrix_build.wheel_audit")
BuildError = importlib.import_module("_symmetrix_build.command").BuildError


@pytest.mark.parametrize(
    ("toolkit", "linkage", "requirements", "error"),
    [
        ("cuda12", "runtime", ["nvidia-cufft-cu12>=11.4,<12"], None),
        ("cuda13", "runtime", ["nvidia-cufft>=12,<13"], None),
        ("cuda13", "direct", ["nvidia-cufft>=12,<13"], "runtime cuFFT"),
        ("cuda13", "runtime", [], "external nvidia-cufft"),
        ("cuda13", "runtime", ["nvidia-cufft-cu12>=11.4,<12"], "wrong CUDA"),
        (
            "cuda13",
            "runtime",
            ["nvidia-cufft>=12,<13; extra == 'fft'"],
            "unconditional",
        ),
        (
            "cuda13",
            "runtime",
            ["nvidia-cufft>=12,<13", "nvidia-cufft-cu12"],
            "wrong CUDA",
        ),
        ("cuda14", "runtime", ["nvidia-cufft>=12,<13"], "unsupported CUDA"),
    ],
)
def test_cuda_wheel_requires_one_matching_external_runtime(
    tmp_path, toolkit, linkage, requirements, error
):
    descriptor = {
        "distribution": "symmetrix-xl-cuda13-sm120",
        "frontend_version": "0.2.0",
        "source_commit": "a" * 40,
        "source_content_sha256": "b" * 64,
        "native_source_content_sha256": "c" * 64,
        "source_dirty": False,
        "backend": "cuda",
        "toolkit": toolkit,
        "cufft_linkage": linkage,
    }
    wheel = tmp_path / "metadata.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "symmetrix_backend_cuda13_sm120/backend.json", json.dumps(descriptor)
        )
        archive.writestr(
            "backend.dist-info/METADATA",
            "Name: symmetrix-xl-cuda13-sm120\nVersion: 0.2.0\n"
            "Requires-Dist: symmetrix-xl==0.2.0\n"
            + "".join(f"Requires-Dist: {item}\n" for item in requirements),
        )
    with zipfile.ZipFile(wheel) as archive:
        if error:
            with pytest.raises(BuildError, match=error):
                audit._audit_release_metadata(archive)
        else:
            audit._audit_release_metadata(archive)


@pytest.mark.parametrize(
    "kind", ["renamed", "direct_import", "soname", "filename", "archive"]
)
def test_wheel_rejects_cufft_payload_and_direct_linkage(tmp_path, kind):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("requires a C compiler")
    source = tmp_path / "vendor.c"
    source.write_text(
        "extern int cufftGetVersion(int*); int invoke(void) { int v; return cufftGetVersion(&v); }"
        if kind == "direct_import"
        else "int cufftGetVersion(int *version) { *version = 120300; return 0; }"
    )
    library = tmp_path / "runtime.so"
    command = [compiler, "-shared", "-fPIC", str(source), "-o", str(library)]
    if kind == "soname":
        command.append("-Wl,-soname,libcufft-deadbeef.so.12")
    subprocess.run(command, check=True, capture_output=True)
    member = "some.libs/renamed.so"
    content = library.read_bytes()
    if kind == "filename":
        member = "some.libs/libcufft-deadbeef.so.12"
        content = b"copied vendor payload"
    elif kind == "archive":
        member = "some.libs/renamed.a"
        content = b"!<arch>\n" + content
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, content)
        archive.writestr(
            "symmetrix/_backend_cpu.json",
            json.dumps(
                {
                    "distribution": "symmetrix-xl",
                    "frontend_version": "0.2.0",
                    "source_commit": "a" * 40,
                    "source_content_sha256": "b" * 64,
                    "native_source_content_sha256": "c" * 64,
                    "source_dirty": False,
                }
            ),
        )
        archive.writestr(
            "symmetrix_xl-0.2.0.dist-info/METADATA",
            "Name: symmetrix-xl\nVersion: 0.2.0\n",
        )
    with pytest.raises(BuildError, match="cuFFT|static archive"):
        audit.audit_wheel(wheel)
