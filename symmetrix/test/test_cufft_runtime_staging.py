"""The CUDA provider patch must work without a Git checkout or a CUDA build."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cufft_patch_stages_refreshes_and_preserves_upstream(tmp_path):
    cmake = shutil.which("cmake")
    git = shutil.which("git")
    upstream = ROOT / "libsymmetrix/external/kokkos-fft"
    if not cmake or not git or not (upstream / "fft/src/CMakeLists.txt").is_file():
        pytest.skip("requires CMake, Git, and the pinned KokkosFFT source")
    # Exercise sdist-like sources with no .git metadata and a real patched tree.
    source = tmp_path / "upstream"
    shutil.copytree(upstream, source, ignore=shutil.ignore_patterns(".git"))
    project = tmp_path / "project"
    patches = project / "cmake/patches"
    patches.mkdir(parents=True)
    patch_name = "kokkosfft-2.0.0-cufft-runtime.patch"
    shutil.copyfile(
        ROOT / "libsymmetrix/cmake/patches" / patch_name, patches / patch_name
    )
    helper = ROOT / "libsymmetrix/cmake/cufft_runtime.cmake"
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(staging NONE)\n"
        "find_package(Git REQUIRED)\n"
        f'set(SYMMETRIX_KOKKOSFFT_SOURCE_DIR "{source}")\n'
        f'include("{helper}")\nsymmetrix_prepare_cufft_runtime()\n'
    )
    build = tmp_path / "build"

    def configure():
        result = subprocess.run(
            [cmake, "-S", str(project), "-B", str(build)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    configure()
    staged = build / "external/kokkosfft-runtime-source"
    header = "fft/src/KokkosFFT_Cuda_types.hpp"
    assert (
        "symmetrix::cufft_runtime::api().cufftPlanMany" in (staged / header).read_text()
    )
    assert "symmetrix::cufft_runtime" not in (source / header).read_text()
    stamp = (staged / header).stat().st_mtime_ns
    configure()
    assert (staged / header).stat().st_mtime_ns == stamp

    # A non-patched input changing must invalidate the staged snapshot too.
    changed = source / "fft/src/KokkosFFT.hpp"
    changed.write_text(changed.read_text() + "\n// Test snapshot refresh.\n")
    configure()
    assert (
        (staged / "fft/src/KokkosFFT.hpp")
        .read_text()
        .endswith("// Test snapshot refresh.\n")
    )
