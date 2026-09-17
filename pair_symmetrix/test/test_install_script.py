import subprocess
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
PAIR_SOURCES = (
    "compute_symmetrix_timing.h",
    "compute_symmetrix_timing.cpp",
    "pair_symmetrix_mace.h",
    "pair_symmetrix_mace.cpp",
    "pair_symmetrix_mace_kokkos.h",
    "pair_symmetrix_mace_kokkos.cpp",
)


def _lammps_tree(root, version="10 Dec 2025"):
    (root / "src" / "KOKKOS").mkdir(parents=True)
    (root / "cmake").mkdir()
    (root / "cmake" / "CMakeLists.txt").write_text("add_library(lammps)\n")
    (root / "src" / "version.h").write_text(f'#define LAMMPS_VERSION "{version}"\n')


def test_installer_is_location_independent_and_idempotent(tmp_path):
    lammps = tmp_path / "LAMMPS source with spaces"
    _lammps_tree(lammps)
    with (lammps / "cmake" / "CMakeLists.txt").open("a") as cmake:
        cmake.write(
            "\nfunction(symmetrix_add_lammps_library)\n"
            "  add_subdirectory(/old/libsymmetrix libsymmetrix)\n"
            "endfunction()\n"
            "symmetrix_add_lammps_library()\n"
            "target_include_directories(lammps PRIVATE /old/include)\n"
            "target_link_libraries(lammps PRIVATE symmetrix)\n"
        )
    working_directory = tmp_path / "unrelated working directory"
    working_directory.mkdir()

    for _ in range(2):
        subprocess.run(
            [INSTALLER, lammps],
            cwd=working_directory,
            check=True,
            text=True,
            capture_output=True,
        )

    for source in PAIR_SOURCES:
        destination = (
            lammps / "src" / "KOKKOS" / source
            if "kokkos" in source
            else lammps / "src" / source
        )
        assert destination.is_symlink()
        assert destination.resolve() == (INSTALLER.parent / source).resolve()

    cmake = (lammps / "cmake" / "CMakeLists.txt").read_text()
    assert cmake.count("# BEGIN SYMMETRIX PAIR STYLE") == 1
    assert cmake.count("function(symmetrix_add_lammps_library)") == 1
    assert "add_library(lammps)" in cmake


def test_installer_rejects_old_lammps(tmp_path):
    lammps = tmp_path / "lammps"
    _lammps_tree(lammps, version="22 Jul 2025")

    result = subprocess.run(
        [INSTALLER, lammps],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "requires LAMMPS 10 Dec 2025 or newer" in result.stderr
