"""Host arithmetic for the optimized row kernels."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from compact_r1_model import foundation_r1_test_model


def test_mace_row_arithmetic(tmp_path):
    root = Path(__file__).resolve().parents[2]
    native = root / "libsymmetrix" / "source"
    cpu_build = Path(
        os.environ.get("SYMMETRIX_CPU_TEST_BUILD", root / "symmetrix/build")
    )
    kokkos = cpu_build / "cmake_packages/Kokkos"
    if not (kokkos / "KokkosConfig.cmake").is_file():
        pytest.skip("an OpenMP development build is required for the native row test")

    (native_extension,) = cpu_build.glob("_native_cpu.*.so")
    cases = Path(__file__).with_name("cpp")
    contract_path = (
        cases.parent / "data/execution_contracts/jit_r1_fixture_c4_e3_l0_contract.json"
    )
    model = foundation_r1_test_model(
        json.loads(contract_path.read_text()), atomic_numbers=(1, 6, 8)
    )
    channels = model["num_channels"]
    model["H0_weights"] = [
        (1 + species) * model["H0_weights"][channel]
        for species in range(3)
        for channel in range(channels)
    ]
    model_path = tmp_path / "distinct-embeddings.json"
    model_path.write_text(json.dumps(model))
    (tmp_path / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(mace_row_arithmetic LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Kokkos CONFIG REQUIRED PATHS "{kokkos}" NO_DEFAULT_PATH)
find_package(Threads REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed)
add_executable(mace_row_arithmetic "{cases / 'mace_row_arithmetic.cpp'}")
target_include_directories(mace_row_arithmetic PRIVATE "{native}")
target_link_libraries(mace_row_arithmetic PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(mace_row_arithmetic PRIVATE -fno-fast-math -ffp-contract=off)
add_executable(mace_canonical_r1 "{cases / 'mace_canonical_r1.cpp'}")
target_include_directories(mace_canonical_r1 PRIVATE "{native}"
    "{root / 'libsymmetrix/external/json/single_include'}"
    "{root / 'symmetrix/source/symmetrix'}")
add_library(cpu_extension SHARED IMPORTED)
set_property(TARGET cpu_extension PROPERTY IMPORTED_LOCATION "{native_extension}")
target_link_libraries(mace_canonical_r1 PRIVATE cpu_extension
    Kokkos::kokkos Python::Python)
"""
    )
    build = tmp_path / "build"
    for argv in (
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
        ],
        ["cmake", "--build", str(build), "-j2"],
        [str(build / "mace_row_arithmetic")],
        [str(build / "mace_canonical_r1"), str(model_path)],
    ):
        run = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false"},
        )
        assert run.returncode == 0, run.stdout + run.stderr
        print(run.stdout, end="")
