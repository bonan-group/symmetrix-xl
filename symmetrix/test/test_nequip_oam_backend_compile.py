"""Compile the backend predicate without any MACE-family headers."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_host_execution_space_is_available_without_mace_headers(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    kokkos_dir = Path(
        os.environ.get(
            "Kokkos_DIR", repository / "symmetrix/build/cmake_packages/Kokkos"
        )
    )
    if not (kokkos_dir / "KokkosConfig.cmake").is_file():
        pytest.skip("a configured Kokkos package is required; set Kokkos_DIR")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for the neutral backend compile check")
    source = tmp_path / "backend.cpp"
    source.write_text(
        '#include "device_backend.hpp"\n'
        "struct HostExecution { using memory_space = Kokkos::HostSpace; };\n"
        "struct OtherMemory {};\n"
        "struct OtherExecution { using memory_space = OtherMemory; };\n"
        "static_assert(symmetrix::execution::host_execution_space<HostExecution>);\n"
        "static_assert(!symmetrix::execution::host_execution_space<OtherExecution>);\n"
        "static_assert(symmetrix::execution::host_execution_space<"
        "Kokkos::DefaultHostExecutionSpace>);\n",
        encoding="ascii",
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(neutral_backend_probe LANGUAGES CXX)\n"
        "find_package(Kokkos CONFIG REQUIRED)\n"
        "add_library(backend_probe OBJECT backend.cpp)\n"
        "target_compile_features(backend_probe PRIVATE cxx_std_20)\n"
        "target_link_libraries(backend_probe PRIVATE Kokkos::kokkos)\n"
        f'target_include_directories(backend_probe PRIVATE "{repository / "libsymmetrix/source"}"\n'
        f'    "{repository / "symmetrix/source/symmetrix"}")\n',
        encoding="utf-8",
    )
    build = tmp_path / "build"
    for command in (
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DKokkos_DIR={kokkos_dir.resolve()}",
        ],
        ["cmake", "--build", str(build), "--target", "backend_probe"],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
