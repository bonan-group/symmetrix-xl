import subprocess
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
PAIR_SOURCES = (
    "compute_symmetrix_timing.h",
    "compute_symmetrix_timing.cpp",
    "pair_symmetrix_mace_kokkos.h",
    "pair_symmetrix_mace_kokkos.cpp",
    "pair_symmetrix_mace_polar_kokkos.h",
    "pair_symmetrix_mace_polar_kokkos.cpp",
    "pair_symmetrix_mace_polar_admission.hpp",
)
RETIRED_PAIR_SOURCES = (
    "pair_symmetrix_mace.h",
    "pair_symmetrix_mace.cpp",
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
            if "kokkos" in source or source.endswith("_admission.hpp")
            else lammps / "src" / source
        )
        assert destination.is_symlink()
        assert destination.resolve() == (INSTALLER.parent / source).resolve()

    # The base-header anchor makes the Kokkos style discoverable to LAMMPS.
    anchor = lammps / "src" / "pair_symmetrix_mace.h"
    assert anchor.is_symlink()
    assert (
        anchor.resolve()
        == (INSTALLER.parent / "pair_symmetrix_mace_anchor.h").resolve()
    )
    polar_anchor = lammps / "src" / "pair_symmetrix_mace_polar.h"
    assert polar_anchor.is_symlink()
    assert (
        polar_anchor.resolve()
        == (INSTALLER.parent / "pair_symmetrix_mace_polar_anchor.h").resolve()
    )

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


def test_installer_removes_retired_serial_pair_style_links(tmp_path):
    lammps = tmp_path / "lammps"
    _lammps_tree(lammps)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("unrelated\n")
    for retired in RETIRED_PAIR_SOURCES:
        (lammps / "src" / retired).symlink_to(unrelated)

    subprocess.run([INSTALLER, lammps], check=True, text=True, capture_output=True)

    # The retired translation unit must not survive, and the retired header link
    # must not survive either: it is replaced by the style-discovery anchor.
    retired_unit = lammps / "src" / "pair_symmetrix_mace.cpp"
    assert not retired_unit.is_symlink()
    assert not retired_unit.exists()

    anchor = lammps / "src" / "pair_symmetrix_mace.h"
    assert anchor.resolve() != unrelated.resolve()
    assert (
        anchor.resolve()
        == (INSTALLER.parent / "pair_symmetrix_mace_anchor.h").resolve()
    )


def test_installer_leaves_a_regular_retired_pair_style_file_alone(tmp_path):
    lammps = tmp_path / "lammps"
    _lammps_tree(lammps)
    contents = "// locally patched serial pair style\n"
    (lammps / "src" / "pair_symmetrix_mace.h").write_text(contents)

    subprocess.run([INSTALLER, lammps], check=True, text=True, capture_output=True)

    installed = lammps / "src" / "pair_symmetrix_mace.h"
    assert not installed.is_symlink()
    assert installed.read_text() == contents


def test_installer_preserves_existing_styles_when_adding_polar(tmp_path):
    lammps = tmp_path / "lammps"
    _lammps_tree(lammps)

    subprocess.run([INSTALLER, lammps], check=True, text=True, capture_output=True)

    standard = lammps / "src" / "KOKKOS" / "pair_symmetrix_mace_kokkos.h"
    polar = lammps / "src" / "KOKKOS" / "pair_symmetrix_mace_polar_kokkos.h"
    assert (
        standard.resolve()
        == (INSTALLER.parent / "pair_symmetrix_mace_kokkos.h").resolve()
    )
    assert (
        polar.resolve()
        == (INSTALLER.parent / "pair_symmetrix_mace_polar_kokkos.h").resolve()
    )
