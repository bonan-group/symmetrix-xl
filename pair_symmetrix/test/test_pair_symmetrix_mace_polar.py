from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_polar_rank_zero_view_is_allocated_by_pointer_state(tmp_path):
    """A default scalar view can have extent one while carrying no allocation."""

    source = tmp_path / "scalar_view.cpp"
    executable = tmp_path / "scalar_view"
    source.write_text(
        r"""
#include <cassert>
#include <cstddef>
#include <string>
#include "pair_symmetrix_mace_polar_admission.hpp"

struct ScalarView {
  ScalarView() = default;
  explicit ScalarView(const std::string&) : pointer(&storage), allocations(1) {}
  std::size_t extent(int) const { return 1; }
  int* data() const { return pointer; }
  int storage = 0;
  int* pointer = nullptr;
  int allocations = 0;
};

int main() {
  ScalarView view;
  assert(view.extent(0) == 1);
  assert(view.data() == nullptr);
  symmetrix::lammps::ensure_polar_scalar_view_allocated(view, "mapping");
  assert(view.data() != nullptr);
  assert(view.allocations == 1);
  symmetrix::lammps::ensure_polar_scalar_view_allocated(view, "mapping");
  assert(view.allocations == 1);
}
"""
    )
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{SOURCE_ROOT}",
            str(source),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    subprocess.run([executable], check=True)


def test_polar_lammps_admission_rejects_before_model_factory(tmp_path):
    source = tmp_path / "admission.cpp"
    executable = tmp_path / "admission"
    source.write_text(
        """
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include "pair_symmetrix_mace_polar_admission.hpp"

int main() {
  const auto safe_atoms =
    static_cast<std::int64_t>(std::numeric_limits<int>::max()) / 3;
  assert(symmetrix::lammps::polar_atom_count_fits_native_index(safe_atoms));
  assert(!symmetrix::lammps::polar_atom_count_fits_native_index(safe_atoms + 1));

  std::size_t model_factory_calls = 0;
  const auto factory = [&]() { ++model_factory_calls; };
  std::vector<symmetrix::lammps::PolarAdmissionRequest> rejected;
  auto add_rejected = [&](auto member) {
    symmetrix::lammps::PolarAdmissionRequest request{1, 4, 4, false};
    member(request);
    rejected.push_back(request);
  };
  add_rejected([](auto& r) { r.rank_count = 2; });
  add_rejected([](auto& r) { r.local_atoms = 3; });
  add_rejected([](auto& r) { r.per_atom_energy_requested = true; });
  add_rejected([](auto& r) { r.per_atom_virial_requested = true; });
  add_rejected([](auto& r) { r.domain_decomposition_requested = true; });
  add_rejected([](auto& r) { r.hybrid_substyle = true; });
  add_rejected([](auto& r) { r.complete_type_mapping = false; });
  add_rejected([](auto& r) { r.model_is_polar = false; });
  add_rejected([](auto& r) { r.atom_count_fits_native_index = false; });

  for (const auto& request : rejected) {
    bool rejected = false;
    try {
      symmetrix::lammps::admit_polar_pair_style(request);
      factory();
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
    assert(model_factory_calls == 0);
  }

  symmetrix::lammps::admit_polar_pair_style({1, 4, 4, false});
  factory();
  assert(model_factory_calls == 1);
}
"""
    )
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{SOURCE_ROOT}",
            str(source),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    subprocess.run([executable], check=True)


def test_polar_lammps_rejects_ambiguous_jit_artifact_settings(tmp_path):
    """A pair style invocation selects at most one prepared backbone artifact."""

    source = tmp_path / "artifact.cpp"
    executable = tmp_path / "artifact"
    source.write_text(
        r"""
#include <cassert>
#include <stdexcept>
#include <string>
#include "pair_symmetrix_mace_polar_admission.hpp"

int main() {
  // No artifact selects the consumer's portable pair_spline_v1 backbone.
  const symmetrix::lammps::PolarJitArtifactSelection none;
  assert(none.target.empty());
  assert(none.path.empty());

  symmetrix::lammps::PolarJitArtifactSelection selection;
  symmetrix::lammps::select_polar_jit_artifact(
    selection, "jit_host_artifact", "/tmp/host.so");
  assert(selection.target == "host");
  assert(selection.path == "/tmp/host.so");

  bool rejected = false;
  try {
    symmetrix::lammps::select_polar_jit_artifact(
      selection, "jit_device_artifact", "/tmp/device.cubin");
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("at most one") != std::string::npos;
  }
  assert(rejected);

  symmetrix::lammps::PolarJitArtifactSelection repeated_host;
  symmetrix::lammps::select_polar_jit_artifact(
    repeated_host, "jit_host_artifact", "/tmp/host.so");
  rejected = false;
  try {
    symmetrix::lammps::select_polar_jit_artifact(
      repeated_host, "jit_host_artifact", "/tmp/host-2.so");
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("at most one") != std::string::npos;
  }
  assert(rejected);

  symmetrix::lammps::PolarJitArtifactSelection repeated_device;
  symmetrix::lammps::select_polar_jit_artifact(
    repeated_device, "jit_device_artifact", "/tmp/device.cubin");
  assert(repeated_device.target == "device");
  rejected = false;
  try {
    symmetrix::lammps::select_polar_jit_artifact(
      repeated_device, "jit_device_artifact", "/tmp/device-2.cubin");
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("at most one") != std::string::npos;
  }
  assert(rejected);

  symmetrix::lammps::PolarJitArtifactSelection empty;
  rejected = false;
  try {
    symmetrix::lammps::select_polar_jit_artifact(
      empty, "jit_host_artifact", "");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}
"""
    )
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{SOURCE_ROOT}",
            str(source),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    subprocess.run([executable], check=True)
    pair_source = (SOURCE_ROOT / "pair_symmetrix_mace_polar_kokkos.cpp").read_text()
    settings_body = pair_source.split(
        "void PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::settings", 1
    )[1].split("void PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::coeff", 1)[0]
    assert "select_polar_jit_artifact" in settings_body


def test_polar_lammps_force_and_virial_mapping_is_load_bearing(tmp_path):
    source = tmp_path / "mapping.cpp"
    executable = tmp_path / "mapping"
    source.write_text(
        r"""
#include <array>
#include <cassert>
#include <vector>
#include "pair_symmetrix_mace_polar_admission.hpp"

int main() {
  std::vector<double> forces{10, 20, 30, 40, 50, 60, 70, 80, 90};
  const std::vector<double> model{1, 2, 3, 4, 5, 6};
  const std::vector<int> permuted_nodes{1, 0};
  symmetrix::lammps::add_polar_forces_to_owned_atoms(
    model, permuted_nodes, 2, forces);
  assert((forces == std::vector<double>{14, 25, 36, 41, 52, 63, 70, 80, 90}));

  const std::array<double, 6> stress{1, 2, 3, 4, 5, 6};
  const auto periodic = symmetrix::lammps::polar_stress_to_lammps_virial(
    stress, 10.0, true);
  assert((periodic == std::array<double, 6>{-10, -20, -30, -60, -50, -40}));
  const auto isolated = symmetrix::lammps::polar_stress_to_lammps_virial(
    stress, 10.0, false);
  assert((isolated == std::array<double, 6>{0, 0, 0, 0, 0, 0}));

  bool duplicate_rejected = false;
  try {
    std::vector<double> target(6);
    symmetrix::lammps::add_polar_forces_to_owned_atoms(
      model, std::vector<int>{0, 0}, 2, target);
  } catch (const std::invalid_argument&) {
    duplicate_rejected = true;
  }
  assert(duplicate_rejected);
}
"""
    )
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{SOURCE_ROOT}",
            str(source),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    subprocess.run([executable], check=True)


def test_polar_lammps_settings_validate_finite_graph_scalars(tmp_path):
    source = tmp_path / "settings.cpp"
    executable = tmp_path / "settings"
    source.write_text(
        """
#include <cassert>
#include <cmath>
#include <limits>
#include "pair_symmetrix_mace_polar_admission.hpp"

int main() {
  symmetrix::lammps::PolarGraphSettings settings;
  settings.total_charge = 1.25;
  settings.total_spin = -0.5;
  settings.fermi_level = 0.0;
  settings.electric_field = {0.1, -0.2, 0.3};
  settings.electrostatics = "auto";
  symmetrix::lammps::validate_polar_graph_settings(settings);

  settings.electrostatics = "pme";
  symmetrix::lammps::validate_polar_graph_settings(settings);

  bool trailing_rejected = false;
  try {
    (void)symmetrix::lammps::parse_finite_polar_setting("1.0junk", "charge");
  } catch (const std::invalid_argument&) {
    trailing_rejected = true;
  }
  assert(trailing_rejected);

  settings.total_spin = std::numeric_limits<double>::infinity();
  bool rejected = false;
  try {
    symmetrix::lammps::validate_polar_graph_settings(settings);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}
"""
    )
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{SOURCE_ROOT}",
            str(source),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    subprocess.run([executable], check=True)


def test_polar_lammps_input_upload_accounting_is_transaction_exact(tmp_path):
    source = tmp_path / "input_bytes.cpp"
    executable = tmp_path / "input_bytes"
    source.write_text(
        """
#include <cassert>
#include <cstddef>
#include "pair_symmetrix_mace_polar_admission.hpp"

int main() {
  constexpr std::size_t evaluator_rebuild = 848;
  constexpr std::size_t evaluator_position_update = 164;
  constexpr std::size_t evaluator_identical_reuse = 8;
  constexpr std::size_t pair_cell_pbc = 9 * sizeof(double) + 3 * sizeof(int);
  constexpr std::size_t type_metadata = 2 * sizeof(int);
  assert(symmetrix::lammps::polar_pair_input_h2d_bytes(
    evaluator_rebuild, type_metadata) ==
    evaluator_rebuild + pair_cell_pbc + type_metadata);
  assert(symmetrix::lammps::polar_pair_input_h2d_bytes(
    evaluator_position_update, 0) ==
    evaluator_position_update + pair_cell_pbc);
  assert(symmetrix::lammps::polar_pair_input_h2d_bytes(
    evaluator_identical_reuse, 0) ==
    evaluator_identical_reuse + pair_cell_pbc);
}
"""
    )
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{SOURCE_ROOT}",
            str(source),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    subprocess.run([executable], check=True)


def test_polar_pair_style_uses_full_kokkos_list_and_device_only_outputs():
    header = (SOURCE_ROOT / "pair_symmetrix_mace_polar_kokkos.h").read_text()
    source = (SOURCE_ROOT / "pair_symmetrix_mace_polar_kokkos.cpp").read_text()

    assert "NeighConst::REQ_FULL" in source
    assert "NeighListKokkos<DeviceType>" in source
    assert "d_ilist" in source
    assert "d_numneigh" in source
    assert "d_neighbors" in source
    assert "k_tag" in source
    assert "map_kokkos" in source
    assert "neighbor->ago" in source
    assert "node_indices(node)=d_ilist(node)" in source.replace(" ", "")
    assert "force_output_d2h_bytes_" in header
    assert "using PairSymmetrixMACEPolarKokkosDeviceDouble" in header
    assert "using PairSymmetrixMACEPolarKokkosDeviceFloat" in header
    assert "symmetrix_polar_force_output_d2h_bytes" in source
    assert "add_polar_public_forces_device" in source
    assert "map_polar_stress_to_virial_device" in source
    assert "evaluate_prepared_views" in source
    assert "evaluate_views(" not in source
    assert "create_mirror_view_and_copy(Kokkos::HostSpace(),forces_" not in source


def test_polar_pair_style_rejects_hybrid_before_model_metadata_and_factory():
    source = (SOURCE_ROOT / "pair_symmetrix_mace_polar_kokkos.cpp").read_text()

    coeff = source.split(
        "void PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::coeff", 1
    )[1].split("double PairSymmetrixMACEPolarKokkos", 1)[0]
    hybrid_check = coeff.index("hybrid_substyle")
    metadata_read = coeff.index("symmetrix_model_type")
    factory = coeff.index("std::make_unique<Consumer>")
    assert hybrid_check < metadata_read < factory


def test_polar_lammps_nonzero_fermi_is_inert_with_qualified_fixture(tmp_path):
    import os

    executable = os.environ.get("SYMMETRIX_LAMMPS_POLAR_EXECUTABLE")
    model = os.environ.get("SYMMETRIX_LAMMPS_POLAR_MODEL")
    artifact = os.environ.get("SYMMETRIX_LAMMPS_POLAR_ARTIFACT")
    missing = [
        name
        for name, value in (
            ("SYMMETRIX_LAMMPS_POLAR_EXECUTABLE", executable),
            ("SYMMETRIX_LAMMPS_POLAR_MODEL", model),
            ("SYMMETRIX_LAMMPS_POLAR_ARTIFACT", artifact),
        )
        if not value
    ]
    if missing:
        pytest.skip("missing qualified external fixture: " + ", ".join(missing))

    # This optional external gate drives the real linked pair style twice; a
    # source-only parser check cannot catch the consumer request-mapping bug.
    result = subprocess.run(
        [executable, "-help"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "symmetrix/mace/polar/kk" in result.stdout

    def evaluate(fermi_level: float) -> tuple[float, ...]:
        input_path = tmp_path / f"fermi-{fermi_level}.in"
        screen_path = tmp_path / f"fermi-{fermi_level}.screen"
        input_path.write_text(
            f"""
clear
units metal
atom_style atomic/kk
atom_modify map yes
boundary f f f
region task9a block 0 12 0 12 0 12 units box
create_box 2 task9a
create_atoms 1 single 3.0 3.0 3.0 units box
create_atoms 2 single 3.96 3.0 3.0 units box
create_atoms 2 single 2.76 3.93 3.0 units box
mass 1 15.999
mass 2 1.008
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
pair_style symmetrix/mace/polar/kk fermi_level {fermi_level} polar_electrostatics realspace jit_host_artifact {artifact}
pair_coeff * * {model} O H
run_style verlet/kk
timestep 0.0
run 1
variable result_e equal pe
variable result_fx1 equal fx[1]
variable result_fy1 equal fy[1]
variable result_fz1 equal fz[1]
variable result_fx2 equal fx[2]
variable result_fy2 equal fy[2]
variable result_fz2 equal fz[2]
variable result_fx3 equal fx[3]
variable result_fy3 equal fy[3]
variable result_fz3 equal fz[3]
variable result_pxx equal pxx
variable result_pyy equal pyy
variable result_pzz equal pzz
variable result_pyz equal pyz
variable result_pxz equal pxz
variable result_pxy equal pxy
print "POLAR_FERMI_RESULT ${{result_e}} ${{result_fx1}} ${{result_fy1}} ${{result_fz1}} ${{result_fx2}} ${{result_fy2}} ${{result_fz2}} ${{result_fx3}} ${{result_fy3}} ${{result_fz3}} ${{result_pxx}} ${{result_pyy}} ${{result_pzz}} ${{result_pyz}} ${{result_pxz}} ${{result_pxy}}"
"""
        )
        completed = subprocess.run(
            [
                executable,
                "-k",
                "on",
                "t",
                "1",
                "-screen",
                str(screen_path),
                "-log",
                "none",
                "-in",
                str(input_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        marker = next(
            line
            for line in screen_path.read_text().splitlines()
            if line.startswith("POLAR_FERMI_RESULT ")
        )
        return tuple(float(value) for value in marker.split()[1:])

    assert evaluate(7.5) == evaluate(0.0)
