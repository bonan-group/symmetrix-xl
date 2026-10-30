#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace symmetrix::lammps {

struct PolarAdmissionRequest {
  int rank_count = 0;
  std::int64_t local_atoms = 0;
  std::int64_t global_atoms = 0;
  bool per_atom_energy_requested = false;
  bool per_atom_virial_requested = false;
  bool domain_decomposition_requested = false;
  bool hybrid_substyle = false;
  bool complete_type_mapping = true;
  bool model_is_polar = true;
  bool atom_count_fits_native_index = true;
};

inline bool polar_atom_count_fits_native_index(const std::int64_t atoms)
{
  // Three Cartesian force lanes are addressed through the native signed-index
  // path.  Bound the atom count before flattening so 3*N cannot overflow it.
  return atoms >= 0 && atoms <= std::numeric_limits<int>::max() / 3;
}

template <typename View>
inline void ensure_polar_scalar_view_allocated(View &view, const char *label)
{
  if (view.data() == nullptr) view = View(std::string(label));
}

inline void admit_polar_pair_style(const PolarAdmissionRequest &request)
{
  if (request.rank_count != 1)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk requires exactly one MPI rank");
  if (request.local_atoms < 0 || request.global_atoms < 0 ||
      request.local_atoms != request.global_atoms)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk requires the complete system on its one rank");
  if (request.per_atom_energy_requested)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk does not support per-atom energy");
  if (request.per_atom_virial_requested)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk does not support per-atom virial");
  if (request.domain_decomposition_requested)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk does not support domain decomposition");
  if (request.hybrid_substyle)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk cannot be used as a hybrid substyle");
  if (!request.complete_type_mapping)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk requires a complete type mapping");
  if (!request.model_is_polar)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk requires a MACE_Polar model");
  if (!request.atom_count_fits_native_index)
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk atom count exceeds native index capacity");
}

inline std::array<double, 6> polar_stress_to_lammps_virial(
  const std::array<double, 6> &stress, const double volume, const bool periodic)
{
  if (!periodic) return {};
  return {-volume * stress[0], -volume * stress[1], -volume * stress[2],
          -volume * stress[5], -volume * stress[4], -volume * stress[3]};
}

inline std::size_t polar_pair_input_h2d_bytes(
  const std::size_t evaluator_bytes, const std::size_t type_metadata_bytes)
{
  constexpr std::size_t cell_pbc_bytes =
    9 * sizeof(double) + 3 * sizeof(int);
  if (evaluator_bytes > std::numeric_limits<std::size_t>::max() - cell_pbc_bytes ||
      evaluator_bytes + cell_pbc_bytes >
        std::numeric_limits<std::size_t>::max() - type_metadata_bytes)
    throw std::overflow_error(
      "MACE Polar pair input host-to-device byte count overflows size_t");
  return evaluator_bytes + cell_pbc_bytes + type_metadata_bytes;
}

inline void add_polar_forces_to_owned_atoms(
  const std::vector<double> &model_forces, const std::vector<int> &node_indices,
  const int owned_atoms, std::vector<double> &lammps_forces)
{
  if (owned_atoms < 0 ||
      node_indices.size() > std::numeric_limits<std::size_t>::max() / 3 ||
      model_forces.size() != std::size_t{3} * node_indices.size() ||
      lammps_forces.size() < std::size_t{3} * static_cast<std::size_t>(owned_atoms))
    throw std::invalid_argument("MACE Polar force mapping extents are inconsistent");
  std::vector<bool> visited(static_cast<std::size_t>(owned_atoms), false);
  for (std::size_t node = 0; node < node_indices.size(); ++node) {
    const int atom = node_indices[node];
    if (atom < 0 || atom >= owned_atoms || visited[static_cast<std::size_t>(atom)])
      throw std::invalid_argument(
        "MACE Polar node mapping must cover each owned atom exactly once");
    visited[static_cast<std::size_t>(atom)] = true;
    for (int component = 0; component < 3; ++component)
      lammps_forces[3 * atom + component] += model_forces[3 * node + component];
  }
  for (const bool mapped : visited)
    if (!mapped)
      throw std::invalid_argument(
        "MACE Polar node mapping must cover each owned atom exactly once");
}

struct PolarGraphSettings {
  double total_charge = 0.0;
  double total_spin = 0.0;
  double fermi_level = 0.0;
  std::array<double, 3> electric_field{0.0, 0.0, 0.0};
  std::string electrostatics = "auto";
};

struct PolarJitArtifactSelection {
  std::string target;
  std::string path;
};

inline void select_polar_jit_artifact(
  PolarJitArtifactSelection &selection, const std::string &setting,
  const std::string &path)
{
  if (setting != "jit_host_artifact" && setting != "jit_device_artifact")
    throw std::invalid_argument("Unknown MACE Polar JIT artifact setting");
  if (path.empty())
    throw std::invalid_argument("MACE Polar JIT artifact path must not be empty");
  if (!selection.path.empty())
    throw std::invalid_argument(
      "pair_style symmetrix/mace/polar/kk accepts at most one JIT artifact setting");
  selection.target = setting == "jit_host_artifact" ? "host" : "device";
  selection.path = path;
}

inline double parse_finite_polar_setting(
  const std::string &token, const std::string &name)
{
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(token, &consumed);
  } catch (const std::exception &) {
    throw std::invalid_argument("MACE Polar " + name + " must be numeric");
  }
  if (consumed != token.size() || !std::isfinite(value))
    throw std::invalid_argument("MACE Polar " + name + " must be finite numeric text");
  return value;
}

inline void validate_polar_graph_settings(const PolarGraphSettings &settings)
{
  if (!std::isfinite(settings.total_charge) || !std::isfinite(settings.total_spin) ||
      !std::isfinite(settings.fermi_level) ||
      !std::isfinite(settings.electric_field[0]) ||
      !std::isfinite(settings.electric_field[1]) ||
      !std::isfinite(settings.electric_field[2]))
    throw std::invalid_argument("MACE Polar graph settings must be finite");
  if (settings.electrostatics != "auto" && settings.electrostatics != "realspace" &&
      settings.electrostatics != "periodic" &&
      settings.electrostatics != "direct_reciprocal" &&
      settings.electrostatics != "pme")
    throw std::invalid_argument(
      "MACE Polar electrostatics must be auto, realspace, periodic, "
      "direct_reciprocal, or pme");
}

}    // namespace symmetrix::lammps
