"""Exercise the native loader without a GPU or a CUDA driver."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def loader_probe(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cufft-loader")
    compiler = shutil.which("g++")
    candidates = [
        Path(value)
        for value in (
            os.environ.get("CUDA_HOME"),
            os.environ.get("CUDAToolkit_ROOT"),
            "/usr/local/cuda",
            "/usr/local/cuda-13.3",
            "/usr/local/cuda-12.9",
        )
        if value
    ]
    toolkit = next(
        (path for path in candidates if (path / "include/cufft.h").is_file()), None
    )
    if not compiler or toolkit is None:
        pytest.skip("requires a host C++ compiler and CUDA headers")
    source = tmp_path / "probe.cpp"
    source.write_text(
        '#include "cufft_runtime.hpp"\n'
        "#include <iostream>\n"
        "int main() {\n"
        "  auto before = symmetrix::cufft_runtime::information(false);\n"
        '  std::cout << before.state << "\\n";\n'
        "  auto after = symmetrix::cufft_runtime::information(true);\n"
        '  std::cout << after.state << "\\n" << after.version << "\\n" << after.reason << "\\n";\n'
        "  return 0;\n"
        "}\n"
    )
    executable = tmp_path / "probe"
    build = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(ROOT / "libsymmetrix/source"),
            "-I",
            str(toolkit / "include"),
            "-DSYMMETRIX_CUFFT_ABI_MAJOR=12",
            str(source),
            str(ROOT / "libsymmetrix/source/cufft_runtime.cpp"),
            "-ldl",
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    return executable


def test_native_cufft_loader_is_lazy_and_reports_an_explicit_missing_library(
    tmp_path, loader_probe
):
    result = subprocess.run(
        [str(loader_probe)],
        capture_output=True,
        text=True,
        env={**os.environ, "SYMMETRIX_CUFFT_LIBRARY": str(tmp_path / "missing.so")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("unloaded\nprobe-failed\n0\n")
    assert "missing.so" in result.stdout
    assert "nvidia-cufft" in result.stdout


@pytest.mark.parametrize(
    ("version", "missing", "expected"),
    [
        (12300, False, "loaded\n12300\n"),
        (11401, False, "probe-failed\n0\n"),
        (12300, True, "probe-failed\n0\n"),
    ],
)
def test_native_cufft_loader_validates_provider_before_publishing(
    tmp_path, loader_probe, version, missing, expected
):
    names = (
        "cufftPlan1d",
        "cufftPlan2d",
        "cufftPlan3d",
        "cufftPlanMany",
        "cufftCreate",
        "cufftDestroy",
        "cufftSetStream",
        "cufftSetAutoAllocation",
        "cufftSetWorkArea",
        "cufftMakePlan1d",
        "cufftMakePlan2d",
        "cufftMakePlan3d",
        "cufftMakePlanMany",
        "cufftExecR2C",
        "cufftExecC2R",
        "cufftExecD2Z",
        "cufftExecZ2D",
        "cufftExecC2C",
        "cufftExecZ2Z",
    )
    source = tmp_path / "fake.c"
    # Only the correctly typed version query is called. Other symbols prove
    # complete table validation and are never used as FFT implementations.
    source.write_text(
        f"int cufftGetVersion(int *v) {{ *v = {version}; return 0; }}\n"
        + "\n".join(
            f"int {name}(void) {{ return 0; }}"
            for name in names
            if not (missing and name == "cufftExecZ2D")
        )
    )
    library = tmp_path / "fake.so"
    subprocess.run(
        [shutil.which("cc"), "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    result = subprocess.run(
        [str(loader_probe)],
        capture_output=True,
        text=True,
        env={**os.environ, "SYMMETRIX_CUFFT_LIBRARY": str(library)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("unloaded\n" + expected)
    if missing:
        assert "cufftExecZ2D" in result.stdout
    elif version != 12300:
        assert "expected ABI 12" in result.stdout
