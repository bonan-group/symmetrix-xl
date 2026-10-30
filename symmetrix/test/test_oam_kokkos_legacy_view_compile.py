"""Compile affected Kokkos units in the legacy View mode LAMMPS requires.

LAMMPS enables ``KOKKOS_ENABLE_IMPL_VIEW_LEGACY`` for its bundled Kokkos and,
on current releases, forces it on because the legacy implementation computes
view offsets in 64-bit arithmetic.  The legacy ``Kokkos::View`` label
constructor only accepts a ``std::string`` or a character-array literal, so a
helper that forwards its own ``const char*`` parameter as the label fails to
compile there even though the same helper builds against a default Kokkos.

These translation units are compiled into ``pair_symmetrix`` builds, so the
LAMMPS mode has to keep compiling.  The probe uses the repository's configured
Kokkos package and adds only the legacy define, which reproduces the LAMMPS
configuration for the View constructor question.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

LEGACY_VIEW_UNITS = (
    "libsymmetrix/source/oam_kokkos.cpp",
    "libsymmetrix/source/oam_kokkos_forward.cpp",
    "libsymmetrix/source/oam_kokkos_reverse.cpp",
    "libsymmetrix/source/mace_polar_particle_mesh.cpp",
)


def test_kokkos_units_compile_with_the_legacy_view_implementation(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    kokkos_dir = Path(
        os.environ.get(
            "Kokkos_DIR", repository / "symmetrix/build/cmake_packages/Kokkos"
        )
    )
    if not (kokkos_dir / "KokkosConfig.cmake").is_file():
        pytest.skip("a configured Kokkos package is required; set Kokkos_DIR")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for the legacy View compile check")

    sources = "\n".join(f'  "{repository / unit}"' for unit in LEGACY_VIEW_UNITS)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(kokkos_legacy_view_probe LANGUAGES CXX)\n"
        "find_package(Kokkos CONFIG REQUIRED)\n"
        "add_library(kokkos_legacy_view_probe OBJECT\n"
        f"{sources})\n"
        "target_compile_features(kokkos_legacy_view_probe PRIVATE cxx_std_20)\n"
        "target_link_libraries(kokkos_legacy_view_probe PRIVATE Kokkos::kokkos)\n"
        # The probe only has to instantiate templates, so keep it cheap.
        "target_compile_options(kokkos_legacy_view_probe PRIVATE"
        " $<$<CXX_COMPILER_ID:GNU,Clang>:-O0>)\n"
        "target_compile_definitions(kokkos_legacy_view_probe PRIVATE"
        " KOKKOS_ENABLE_IMPL_VIEW_LEGACY)\n"
        f'target_include_directories(kokkos_legacy_view_probe PRIVATE "{repository / "libsymmetrix/source"}"\n'
        f'    "{repository / "symmetrix/source/symmetrix"}"\n'
        f'    "{repository / "libsymmetrix/external/json/single_include"}"\n'
        f'    "{repository / "libsymmetrix/external/sphericart/sphericart/include"}")\n',
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
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "kokkos_legacy_view_probe",
            "--parallel",
            "4",
        ],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
