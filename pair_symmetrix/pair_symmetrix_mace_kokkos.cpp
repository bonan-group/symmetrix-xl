/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing author: Chuck Witt

#include "pair_symmetrix_mace_kokkos.h"
#include "model_metadata.hpp"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "kokkos_base.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "platform.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
PairSymmetrixMACEKokkos<DeviceType, Precision>::PairSymmetrixMACEKokkos(LAMMPS *lmp)
  : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  comm_forward = 0;  // possibly changed below
  comm_reverse = 0;  // possibly changed below
  electric_field_set = false;
  streamed_edges = "auto";
  jit_host_artifact.clear();
  jit_device_artifact.clear();
  jit_m0_device_artifact.clear();
  jit_r0_device_artifact.clear();
  jit_m0_device_schedule = "chunk32";
  jit_m0_device_schedule_set = false;
  jit_device_blocks_per_compute_unit = 8;

  kokkosable = 1;
  reverse_comm_device = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | TYPE_MASK | TAG_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  //host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
PairSymmetrixMACEKokkos<DeviceType, Precision>::~PairSymmetrixMACEKokkos()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
    memoryKK->destroy_kokkos(k_eatom,eatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::compute(int eflag, int vflag)
{
  const bool time_pair = execution_timing_enabled != 0.0;
  if (time_pair) Kokkos::fence("Symmetrix timing pair begin");
  const double pair_start = time_pair ? platform::walltime() : 0.0;
  execution_pair_evaluation_count += 1.0;
  if (mode == "no_domain_decomposition") {
    compute_no_domain_decomposition(eflag, vflag);
  } else if (mode == "mpi_message_passing") {
    compute_mpi_message_passing(eflag, vflag);
  } else if (mode == "no_mpi_message_passing") {
    compute_no_mpi_message_passing(eflag, vflag);
  }
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
  if (time_pair) {
    Kokkos::fence("Symmetrix timing pair end");
    execution_pair_seconds += platform::walltime() - pair_start;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void *PairSymmetrixMACEKokkos<DeviceType, Precision>::extract(const char *name, int &dim)
{
  dim = 0;
  if (std::strcmp(name,"symmetrix_pair_evaluation_count") == 0)
    return static_cast<void *>(&execution_pair_evaluation_count);
  if (std::strcmp(name,"symmetrix_pair_seconds") == 0)
    return static_cast<void *>(&execution_pair_seconds);
  if (std::strcmp(name,"symmetrix_timing_enabled") == 0)
    return static_cast<void *>(&execution_timing_enabled);
  if (std::strcmp(name,"symmetrix_mpi_hidden_state_forward_seconds") == 0)
    return static_cast<void *>(&execution_mpi_hidden_state_forward_seconds);
  if (std::strcmp(name,"symmetrix_mpi_hidden_state_reverse_seconds") == 0)
    return static_cast<void *>(&execution_mpi_hidden_state_reverse_seconds);
  if (std::strcmp(name,"symmetrix_mpi_hidden_state_seconds") == 0)
    return static_cast<void *>(&execution_mpi_hidden_state_seconds);
  if (std::strcmp(name,"symmetrix_mpi_hidden_state_forward_calls") == 0)
    return static_cast<void *>(&execution_mpi_hidden_state_forward_calls);
  if (std::strcmp(name,"symmetrix_mpi_hidden_state_reverse_calls") == 0)
    return static_cast<void *>(&execution_mpi_hidden_state_reverse_calls);
  if (std::strcmp(name,"symmetrix_graph_rebuild_count") == 0)
    return static_cast<void *>(&execution_graph_rebuild_count);
  if (std::strcmp(name,"symmetrix_geometry_refresh_count") == 0)
    return static_cast<void *>(&execution_geometry_refresh_count);
  if (std::strcmp(name,"symmetrix_geometry_only_update_count") == 0)
    return static_cast<void *>(&execution_geometry_only_update_count);
  if (std::strcmp(name,"symmetrix_h1_allocation_count") == 0)
    return static_cast<void *>(&execution_h1_allocation_count);
  if (std::strcmp(name,"symmetrix_mpi_staged_packet_d2h_bytes") == 0)
    return static_cast<void *>(&execution_mpi_staged_packet_d2h_bytes);
  if (std::strcmp(name,"symmetrix_mpi_staged_packet_h2d_bytes") == 0)
    return static_cast<void *>(&execution_mpi_staged_packet_h2d_bytes);
  if (std::strcmp(name,"symmetrix_mpi_staged_index_h2d_bytes") == 0)
    return static_cast<void *>(&execution_mpi_staged_index_h2d_bytes);
  return nullptr;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; ++i)
    for (int j=i; j<atom->ntypes+1; ++j)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
  if (ghostneigh)
    memory->create(cutghost, atom->ntypes+1, atom->ntypes+1, "pair:cutghost");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::settings(int narg, char **arg)
{
  mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  electric_field_set = false;
  streamed_edges = "auto";
  execution_profile = "capacity";
  execution_profile_set = false;
  low_memory_alias_set = false;
  low_memory_alias_value = false;
  allow_fixed_workspace = false;
  debug_execution_plan.clear();
  debug_single_layer_workspace_receivers = 0;
  debug_dual_layer_workspace_receivers = 0;
  prediction_head.clear();
  jit_host_artifact.clear();
  jit_device_artifact.clear();
  jit_m0_device_artifact.clear();
  jit_r0_device_artifact.clear();
  jit_m0_device_schedule = "chunk32";
  jit_m0_device_schedule_set = false;
  jit_device_blocks_per_compute_unit = 8;

  for (int i=0; i<narg; ++i) {
    const std::string token(arg[i]);
    if (token == "no_domain_decomposition" || token == "mpi_message_passing" || token == "no_mpi_message_passing") {
      mode = token;
    } else if (token == "electric_field") {
      if (i+3 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk electric_field requires three components");
      std::array<double,3> field_values;
      try {
        field_values[0] = std::stod(arg[i+1]);
        field_values[1] = std::stod(arg[i+2]);
        field_values[2] = std::stod(arg[i+3]);
      } catch (const std::exception&) {
        error->all(FLERR, "pair_style symmetrix/mace/kk electric_field components must be numeric");
      }
      electric_field = Kokkos::View<double*>("symmetrix_mace_electric_field", 3);
      auto h_electric_field = Kokkos::create_mirror_view(electric_field);
      h_electric_field(0) = field_values[0];
      h_electric_field(1) = field_values[1];
      h_electric_field(2) = field_values[2];
      Kokkos::deep_copy(electric_field, h_electric_field);
      electric_field_set = true;
      i += 3;
    } else if (token == "streamed_edges") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk streamed_edges requires auto, materialized, non-compiled, or direct");
      streamed_edges = arg[++i];
      if (streamed_edges != "auto" && streamed_edges != "materialized"
          && streamed_edges != "generic" && streamed_edges != "non-compiled" && streamed_edges != "direct"
          && streamed_edges != "all_interactions"
          && streamed_edges != "factorized"
          && streamed_edges != "direct_streamed")
        error->all(FLERR, "pair_style symmetrix/mace/kk streamed_edges requires auto, materialized, non-compiled, or direct");
    } else if (token == "profile") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk profile requires capacity or speed");
      execution_profile = arg[++i];
      execution_profile_set = true;
      if (execution_profile != "capacity" && execution_profile != "speed")
        error->all(FLERR, "pair_style symmetrix/mace/kk profile requires capacity or speed");
    } else if (token == "low_memory") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk low_memory requires yes or no");
      const std::string value(arg[++i]);
      low_memory_alias_set = true;
      if (value == "yes")
        low_memory_alias_value = true;
      else if (value == "no")
        low_memory_alias_value = false;
      else
        error->all(FLERR, "pair_style symmetrix/mace/kk low_memory requires yes or no");
    } else if (token == "allow_fixed_workspace") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk allow_fixed_workspace requires yes or no");
      const std::string value(arg[++i]);
      if (value == "yes")
        allow_fixed_workspace = true;
      else if (value == "no")
        allow_fixed_workspace = false;
      else
        error->all(FLERR, "pair_style symmetrix/mace/kk allow_fixed_workspace requires yes or no");
    } else if (token == "_debug_execution_plan") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk _debug_execution_plan requires a plan ID");
      debug_execution_plan = arg[++i];
    } else if (token == "_debug_single_layer_workspace_receivers") {
      if (i+1 >= narg)
        error->all(
          FLERR,
          "pair_style symmetrix/mace/kk _debug_single_layer_workspace_receivers requires a positive integer");
      const std::string value(arg[++i]);
      bool invalid_value = false;
      try {
        std::size_t consumed = 0;
        debug_single_layer_workspace_receivers = std::stoi(value, &consumed);
        invalid_value = consumed != value.size();
      } catch (const std::exception&) {
        invalid_value = true;
      }
      if (invalid_value || debug_single_layer_workspace_receivers <= 0)
        error->all(
          FLERR,
          "pair_style symmetrix/mace/kk _debug_single_layer_workspace_receivers requires a positive integer");
    } else if (token == "_debug_dual_layer_workspace_receivers") {
      if (i+1 >= narg)
        error->all(
          FLERR,
          "pair_style symmetrix/mace/kk _debug_dual_layer_workspace_receivers requires a positive integer");
      const std::string value(arg[++i]);
      bool invalid_value = false;
      try {
        std::size_t consumed = 0;
        debug_dual_layer_workspace_receivers = std::stoi(value, &consumed);
        invalid_value = consumed != value.size();
      } catch (const std::exception&) {
        invalid_value = true;
      }
      if (invalid_value || debug_dual_layer_workspace_receivers <= 0)
        error->all(
          FLERR,
          "pair_style symmetrix/mace/kk _debug_dual_layer_workspace_receivers requires a positive integer");
    } else if (token == "head") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk head requires a name");
      prediction_head = arg[++i];
    } else if (token == "jit_host_artifact") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk jit_host_artifact requires a path");
      jit_host_artifact = arg[++i];
    } else if (token == "jit_device_artifact") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk jit_device_artifact requires a path");
      jit_device_artifact = arg[++i];
    } else if (token == "jit_m0_device_artifact") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk jit_m0_device_artifact requires a path");
      jit_m0_device_artifact = arg[++i];
    } else if (token == "jit_r0_device_artifact") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk jit_r0_device_artifact requires a path");
      jit_r0_device_artifact = arg[++i];
    } else if (token == "jit_m0_device_schedule") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk jit_m0_device_schedule requires chunk32 or table");
      jit_m0_device_schedule = arg[++i];
      jit_m0_device_schedule_set = true;
      if (jit_m0_device_schedule != "chunk32" && jit_m0_device_schedule != "table")
        error->all(FLERR, "pair_style symmetrix/mace/kk jit_m0_device_schedule requires chunk32 or table");
    } else if (token == "jit_device_blocks_per_compute_unit") {
      if (i+1 >= narg)
        error->all(FLERR, "pair_style symmetrix/mace/kk jit_device_blocks_per_compute_unit requires an integer in [1, 32]");
      const std::string value(arg[++i]);
      bool invalid_value = false;
      try {
        std::size_t consumed = 0;
        jit_device_blocks_per_compute_unit = std::stoi(value, &consumed);
        invalid_value = consumed != value.size();
      } catch (const std::exception&) {
        invalid_value = true;
      }
      if (invalid_value || jit_device_blocks_per_compute_unit < 1
          || jit_device_blocks_per_compute_unit > 32)
        error->all(FLERR, "pair_style symmetrix/mace/kk jit_device_blocks_per_compute_unit requires an integer in [1, 32]");
    } else {
      error->all(FLERR, "The command \'pair_style symmetrix/mace/kk {}\' is invalid", token);
    }
  }

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");

  ghostneigh = (mode == "no_mpi_message_passing");
  if (low_memory_alias_set) {
    const std::string alias_profile =
      low_memory_alias_value ? "capacity" : "speed";
    if (execution_profile_set && execution_profile != alias_profile)
      error->all(
        FLERR,
        "pair_style symmetrix/mace/kk profile conflicts with deprecated low_memory setting");
    execution_profile = alias_profile;
    if (comm->me == 0)
      error->warning(
        FLERR,
        "pair_style symmetrix/mace/kk low_memory is deprecated; use profile capacity or profile speed");
  }
  if (!jit_host_artifact.empty() && !jit_device_artifact.empty())
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk accepts exactly one of jit_host_artifact "
      "or jit_device_artifact");
  const bool has_operator_artifact =
    !jit_m0_device_artifact.empty() || !jit_r0_device_artifact.empty();
  if (jit_m0_device_schedule_set && jit_m0_device_artifact.empty())
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk jit_m0_device_schedule requires "
      "jit_m0_device_artifact");
  if (has_operator_artifact && jit_device_artifact.empty())
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk M0/R0 device artifacts require "
      "jit_device_artifact");
  if (has_operator_artifact && execution_profile != "capacity")
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk M0/R0 device artifacts require profile capacity");
  if (allow_fixed_workspace && execution_profile != "capacity")
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk allow_fixed_workspace yes requires profile capacity");
  if (debug_single_layer_workspace_receivers != 0
      && debug_execution_plan != "mh0-single-layer-tiled-v1")
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk _debug_single_layer_workspace_receivers requires "
      "_debug_execution_plan mh0-single-layer-tiled-v1");
  if (debug_dual_layer_workspace_receivers != 0
      && debug_execution_plan != "mh0-dual-layer-tiled-v1")
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk _debug_dual_layer_workspace_receivers requires "
      "_debug_execution_plan mh0-dual-layer-tiled-v1");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != atom->ntypes + 3)
    error->all(FLERR, "Incorrect args for pair coefficients");

  if (symmetrix_is_nonlinear_mace_model(arg[2]))
    error->all(FLERR, "MACE_Nonlinear models are not supported by pair_style symmetrix/mace/kk in this release");

  utils::logmesg(lmp, "Loading MACEKokkos model from \'{}\' ... ", arg[2]);
  mace = std::make_unique<MACEKokkos<Precision>>(arg[2], prediction_head);
  mace->set_allow_fixed_workspace(allow_fixed_workspace);
  if (debug_single_layer_workspace_receivers != 0)
    mace->set_single_layer_workspace_receiver_limit_for_testing(
      debug_single_layer_workspace_receivers);
  if (debug_dual_layer_workspace_receivers != 0)
    mace->set_dual_layer_workspace_receiver_limit_for_testing(
      debug_dual_layer_workspace_receivers);
  utils::logmesg(lmp, "success\n");
  const std::string requested_streamed_edges = streamed_edges;
  if (streamed_edges == "auto") {
    if (!jit_host_artifact.empty() || !jit_device_artifact.empty())
      streamed_edges = "direct";
    else if (mace->supports_streamed_edges())
      streamed_edges = "generic";
  }
  if (streamed_edges != "auto") {
    try {
      mace->set_streamed_edges(streamed_edges);
    } catch (const std::exception& exception) {
      error->all(FLERR, "pair_style symmetrix/mace/kk streamed_edges is incompatible with this model: {}", exception.what());
    }
  }
  if (!mace->supports_streamed_edges() && comm->me == 0)
    error->warning(
      FLERR,
      "Loaded the original Symmetrix pair-spline format (named v1 here); using streamed_edges='materialized'. "
      "Re-export with radial_format='compact' to enable streamed_edges='generic' execution.");
  if (comm->me == 0)
    utils::logmesg(
      lmp,
      "Symmetrix streamed_edges requested='{}' resolved='{}'\n",
      requested_streamed_edges, mace->streamed_edges_mode());
  if (allow_fixed_workspace && !mace_uses_direct_execution(mace->streamed_edges))
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk allow_fixed_workspace yes requires streamed_edges direct");
  if (mace_uses_prepared_execution(mace->streamed_edges)) {
    if (mode == "no_mpi_message_passing")
      error->all(
        FLERR,
        "pair_style symmetrix/mace/kk prepared execution requires "
        "no_domain_decomposition or mpi_message_passing");
  }
  if (mace_uses_direct_execution(mace->streamed_edges)) {
    if (!jit_host_artifact.empty()) {
      try {
        mace->load_jit_host_plugin(jit_host_artifact);
      } catch (const std::exception& exception) {
        error->all(
          FLERR,
          "Failed to load Execution host artifact '{}': {}",
          jit_host_artifact, exception.what());
      }
      if (!mace->jit_host_plugin_ready())
        error->all(FLERR, "Execution host artifact loaded without activating a JIT executor");
      utils::logmesg(
        lmp,
        "Activated Execution JIT host artifact '{}'\n",
        mace->jit_host_plugin_artifact_id());
    } else if (!jit_device_artifact.empty()) {
      try {
        mace->load_jit_device_plugin(
          jit_device_artifact, jit_device_blocks_per_compute_unit);
      } catch (const std::exception& exception) {
        error->all(
          FLERR,
          "Failed to load Execution device artifact '{}': {}",
          jit_device_artifact, exception.what());
      }
      if (!mace->jit_device_plugin_ready())
        error->all(FLERR, "Execution device artifact loaded without activating a JIT executor");
      utils::logmesg(
        lmp,
        "Activated Execution JIT device artifact '{}'\n",
        mace->jit_device_plugin_artifact_id());
    } else if (!mace->single_layer_readout) {
      error->all(
        FLERR,
        "pair_style symmetrix/mace/kk streamed_edges direct requires "
        "jit_host_artifact or jit_device_artifact; direct execution does not "
        "fall back to another algorithm");
    }
    if (!jit_m0_device_artifact.empty()) {
      try {
        mace->load_m0_device_module(
          jit_m0_device_artifact, jit_m0_device_schedule,
          jit_device_blocks_per_compute_unit);
      } catch (const std::exception& exception) {
        error->all(
          FLERR,
          "Failed to load Execution M0 device artifact '{}': {}",
          jit_m0_device_artifact, exception.what());
      }
      if (!mace->m0_device_module_ready())
        error->all(FLERR, "Execution M0 device artifact loaded without activating an M0 executor");
      utils::logmesg(
        lmp,
        "Activated Execution M0 device artifact '{}' with schedule '{}'\n",
        mace->m0_device_module_artifact_id(),
        mace->m0_device_module_schedule_name());
    }
    if (!jit_r0_device_artifact.empty()) {
      try {
        mace->load_r0_device_module(
          jit_r0_device_artifact, jit_device_blocks_per_compute_unit);
      } catch (const std::exception& exception) {
        error->all(
          FLERR,
          "Failed to load Execution R0 device artifact '{}': {}",
          jit_r0_device_artifact, exception.what());
      }
      if (!mace->r0_device_module_ready())
        error->all(FLERR, "Execution R0 device artifact loaded without activating an R0 executor");
      utils::logmesg(
        lmp,
        "Activated Execution R0 device artifact '{}'\n",
        mace->r0_device_module_artifact_id());
    }
  } else if (!jit_host_artifact.empty() || !jit_device_artifact.empty()) {
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk JIT artifacts require streamed_edges direct");
  }
  try {
    mace->set_execution_plan_request(
      mace->streamed_edges_mode(), execution_profile, debug_execution_plan);
    mace->resolve_execution_plan();
  } catch (const std::exception& exception) {
    error->all(
      FLERR,
      "pair_style symmetrix/mace/kk could not resolve profile '{}': {}",
      execution_profile, exception.what());
  }
  if (comm->me == 0) {
    utils::logmesg(
      lmp, "Symmetrix execution profile='{}'\n", execution_profile);
    if (allow_fixed_workspace)
      utils::logmesg(lmp, "Allowed fixed-workspace execution\n");
  }
  execution_graph_generation = 0;
  execution_num_nodes = 0;
  execution_num_feature_nodes = 0;
  execution_num_edges = 0;
  execution_topology_fingerprint = 0;
  execution_pair_evaluation_count = 0.0;
  execution_pair_seconds = 0.0;
  execution_mpi_hidden_state_forward_seconds = 0.0;
  execution_mpi_hidden_state_reverse_seconds = 0.0;
  execution_mpi_hidden_state_seconds = 0.0;
  execution_mpi_hidden_state_forward_calls = 0.0;
  execution_mpi_hidden_state_reverse_calls = 0.0;
  execution_graph_rebuild_count = 0.0;
  execution_geometry_refresh_count = 0.0;
  execution_geometry_only_update_count = 0.0;
  execution_h1_allocation_count = 0.0;
  execution_mpi_staged_packet_d2h_bytes = 0.0;
  execution_mpi_staged_packet_h2d_bytes = 0.0;
  execution_mpi_staged_index_h2d_bytes = 0.0;
  if (mace->has_field_coupling && !electric_field_set)
    error->all(FLERR, "MACEField models require pair_style symmetrix/mace/kk electric_field Ex Ey Ez");

  // extract atomic numbers from pair_coeff
  mace_types = Kokkos::View<int*>("mace_types", atom->ntypes);
  auto h_mace_types = Kokkos::create_mirror_view(mace_types);
  auto h_mace_atomic_numbers = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mace->atomic_numbers);
  auto active_mace_types = std::vector<int>();
  active_mace_types.reserve(atom->ntypes);
  for (int i=3; i<narg; ++i) {
    // find atomic number for element in arg[i]
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    // find mace index corresponding to this element
    int mace_index = -1;
    for (int j=0; j<mace->atomic_numbers.size(); ++j)
        if (h_mace_atomic_numbers(j) == atomic_number)
            mace_index = j;
    if (mace_index < 0)
      error->all(FLERR, "Problem matching LAMMPS types to MACEKokkos types.");
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to MACEKokkos type {}\n",
                   i-2, arg[i], mace_index);
    h_mace_types(i-3) = mace_index;
    active_mace_types.push_back(mace_index);
  }
  mace->prepare_active_types(active_mace_types);
  Kokkos::deep_copy(mace_types, h_mace_types);

  // set message size
  if (mode == "mpi_message_passing") {
    if (mace->single_layer_readout) {
      comm_forward = 0;
      comm_reverse = 0;
    } else {
      comm_forward = checked_comm_value_count(1, "per-atom");
      comm_reverse = comm_forward;
    }
  } else {
    comm_forward = 0;
    comm_reverse = 0;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
double PairSymmetrixMACEKokkos<DeviceType, Precision>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  if (ghostneigh) cutghost[i][j] = cutghost[j][i] = mace->r_cut;
  return mace->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::init_style()
{
  // Compute initialization follows pair initialization, so rebuild the number
  // of active symmetrix/timing computes for every full LAMMPS initialization.
  execution_timing_enabled = 0.0;
  if (atom->map_user == atom->MAP_NONE) error->all(FLERR, "symmetrix/mace/kk requires \'atom_modify map [yes|array|hash]\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/mace/kk requires newton pair on");

  if (mode == "no_domain_decomposition" or mode == "mpi_message_passing") {
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
    request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                             !std::is_same_v<DeviceType,LMPDeviceType>);
    request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  } else {
    // enforce the communication cutoff is more than twice the model cutoff
    const double comm_cutoff = comm->get_comm_cutoff();
    if (comm->get_comm_cutoff() < (2*mace->r_cut + neighbor->skin)) {
      std::string cutoff_val = std::to_string((2.0 * mace->r_cut) + neighbor->skin);
      char *args[2];
      args[0] = (char *)"cutoff";
      args[1] = const_cast<char *>(cutoff_val.c_str());
      comm->modify_params(2, args);
      if (comm->me == 0) error->warning(FLERR, "symmetrix/mace/kk is setting the communication cutoff to {}", cutoff_val);
    }
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                             !std::is_same_v<DeviceType,LMPDeviceType>);
    request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::ensure_legacy_comm_capacity(
    std::size_t num_indices,
    std::size_t num_values)
{
  if (legacy_comm_indices.extent(0) < num_indices)
    Kokkos::realloc(legacy_comm_indices, num_indices);
  if (legacy_comm_packet.extent(0) < num_values)
    Kokkos::realloc(legacy_comm_packet, num_values);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
int PairSymmetrixMACEKokkos<DeviceType, Precision>::checked_comm_value_count(
    std::size_t num_atoms,
    const char *operation)
{
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  const auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (num_LM == 0 || num_channels == 0 || num_LM > int_max/num_channels) {
    error->one(FLERR,
      "pair_style symmetrix/mace/kk {} communication width exceeds the LAMMPS int-count ABI",
      operation);
    return 0;
  }
  const auto values_per_atom = num_LM*num_channels;
  if (num_atoms > int_max/values_per_atom) {
    error->one(FLERR,
      "pair_style symmetrix/mace/kk {} communication packet exceeds the LAMMPS int-count ABI",
      operation);
    return 0;
  }
  return static_cast<int>(num_atoms*values_per_atom);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const auto num_indices = static_cast<std::size_t>(n);
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  const auto value_count = checked_comm_value_count(num_indices, "forward-pack");
  const auto num_values = static_cast<std::size_t>(value_count);
  ensure_legacy_comm_capacity(num_indices, num_values);

  using unmanaged_host_indices = Kokkos::View<
      const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using unmanaged_host_packet = Kokkos::View<
      double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const unmanaged_host_indices host_indices(list, num_indices);
  const unmanaged_host_packet host_packet(buf, num_values);
  const auto device_indices = Kokkos::subview(
      legacy_comm_indices, std::pair<std::size_t,std::size_t>{0, num_indices});
  const auto device_packet = Kokkos::subview(
      legacy_comm_packet, std::pair<std::size_t,std::size_t>{0, num_values});
  Kokkos::deep_copy(device_indices, host_indices);

  const auto H1 = this->H1;
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::pack_forward_comm_host_staged",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {0,0,0}, {n,mace->num_LM,mace->num_channels}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const auto offset =
        (static_cast<std::size_t>(ii)*num_LM + static_cast<std::size_t>(LM))*num_channels
        + static_cast<std::size_t>(k);
      device_packet(offset) = H1(device_indices(ii),LM,k);
    });
  Kokkos::deep_copy(host_packet, device_packet);
  execution_mpi_staged_index_h2d_bytes += num_indices * sizeof(int);
  execution_mpi_staged_packet_d2h_bytes += num_values * sizeof(double);
  return value_count;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  auto d_buf = buf.view<DeviceType>();
  const auto H1 = this->H1;
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  const auto value_count = checked_comm_value_count(
    static_cast<std::size_t>(n), "forward-pack");
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::pack_forward_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {0,0,0}, {n,mace->num_LM,mace->num_channels}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const int i = d_sendlist(ii);
      const auto offset =
        (static_cast<std::size_t>(ii)*num_LM + static_cast<std::size_t>(LM))*num_channels
        + static_cast<std::size_t>(k);
      d_buf(offset) = H1(i,LM,k);
    });
  return value_count;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::unpack_forward_comm(int n, int first, double *buf)
{
  const auto value_count = checked_comm_value_count(
    static_cast<std::size_t>(n), "forward-unpack");
  const auto num_values = static_cast<std::size_t>(value_count);
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  ensure_legacy_comm_capacity(0, num_values);

  using unmanaged_host_packet = Kokkos::View<
      const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const unmanaged_host_packet host_packet(buf, num_values);
  const auto device_packet = Kokkos::subview(
      legacy_comm_packet, std::pair<std::size_t,std::size_t>{0, num_values});
  Kokkos::deep_copy(device_packet, host_packet);

  auto H1 = this->H1;
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::unpack_forward_comm_host_staged",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {0,0,0}, {n,mace->num_LM,mace->num_channels}),
    KOKKOS_LAMBDA (const int i, const int LM, const int k) {
      const auto offset =
        (static_cast<std::size_t>(i)*num_LM + static_cast<std::size_t>(LM))*num_channels
        + static_cast<std::size_t>(k);
      H1(first+i,LM,k) = device_packet(offset);
    });
  execution_mpi_staged_packet_h2d_bytes += num_values * sizeof(double);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::unpack_forward_comm_kokkos(int n, int first, DAT::tdual_double_1d &buf)
{
  auto H1 = this->H1;
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  checked_comm_value_count(static_cast<std::size_t>(n), "forward-unpack");
  //typename ArrayTypes<DeviceType>::t_xfloat_1d_um v_buf = buf.view<DeviceType>();
  const auto d_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::unpack_forward_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {0,0,0}, {n,mace->num_LM,mace->num_channels}),
    KOKKOS_LAMBDA (const int i, const int LM, const int k) {
      const auto offset =
        (static_cast<std::size_t>(i)*num_LM + static_cast<std::size_t>(LM))*num_channels
        + static_cast<std::size_t>(k);
      H1((first+i),LM,k) = d_buf(offset);
    });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_reverse_comm(int n, int first, double *buf)
{
  const auto value_count = checked_comm_value_count(
    static_cast<std::size_t>(n), "reverse-pack");
  const auto num_values = static_cast<std::size_t>(value_count);
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  ensure_legacy_comm_capacity(0, num_values);

  using unmanaged_host_packet = Kokkos::View<
      double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const unmanaged_host_packet host_packet(buf, num_values);
  const auto device_packet = Kokkos::subview(
      legacy_comm_packet, std::pair<std::size_t,std::size_t>{0, num_values});
  const auto H1_adj = this->H1_adj;
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::pack_reverse_comm_host_staged",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {0,0,0}, {n,mace->num_LM,mace->num_channels}),
    KOKKOS_LAMBDA (const int i, const int LM, const int k) {
      const auto offset =
        (static_cast<std::size_t>(i)*num_LM + static_cast<std::size_t>(LM))*num_channels
        + static_cast<std::size_t>(k);
      device_packet(offset) = H1_adj(first+i,LM,k);
    });
  Kokkos::deep_copy(host_packet, device_packet);
  execution_mpi_staged_packet_d2h_bytes += num_values * sizeof(double);
  return value_count;
}

/* ---------------------------------------------------------------------- */
template<class DeviceType, typename Precision>
int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &buf)
{
  auto d_buf = buf.view<DeviceType>();
  const auto H1_adj = this->H1_adj;
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  const auto value_count = checked_comm_value_count(
    static_cast<std::size_t>(n), "reverse-pack");
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::pack_reverse_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {0,0,0}, {n,mace->num_LM,mace->num_channels}),
    KOKKOS_LAMBDA (const int i, const int LM, const int k) {
      const auto offset =
        (static_cast<std::size_t>(i)*num_LM + static_cast<std::size_t>(LM))*num_channels
        + static_cast<std::size_t>(k);
      d_buf(offset) = H1_adj((first+i),LM,k);
    });
  return value_count;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::unpack_reverse_comm(int n, int *list, double *buf)
{
  const auto num_indices = static_cast<std::size_t>(n);
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  const auto value_count = checked_comm_value_count(num_indices, "reverse-unpack");
  const auto num_values = static_cast<std::size_t>(value_count);
  ensure_legacy_comm_capacity(num_indices, num_values);

  using unmanaged_host_indices = Kokkos::View<
      const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using unmanaged_host_packet = Kokkos::View<
      const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const unmanaged_host_indices host_indices(list, num_indices);
  const unmanaged_host_packet host_packet(buf, num_values);
  const auto device_indices = Kokkos::subview(
      legacy_comm_indices, std::pair<std::size_t,std::size_t>{0, num_indices});
  const auto device_packet = Kokkos::subview(
      legacy_comm_packet, std::pair<std::size_t,std::size_t>{0, num_values});
  Kokkos::deep_copy(device_indices, host_indices);
  Kokkos::deep_copy(device_packet, host_packet);

  auto H1_adj = this->H1_adj;
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::unpack_reverse_comm_host_staged",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {0,0,0}, {n,mace->num_LM,mace->num_channels}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const auto offset =
        (static_cast<std::size_t>(ii)*num_LM + static_cast<std::size_t>(LM))*num_channels
        + static_cast<std::size_t>(k);
      Kokkos::atomic_add(&H1_adj(device_indices(ii),LM,k), device_packet(offset));
    });
  execution_mpi_staged_index_h2d_bytes += num_indices * sizeof(int);
  execution_mpi_staged_packet_h2d_bytes += num_values * sizeof(double);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d& buf)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  const auto d_buf = buf.view<DeviceType>();
  auto H1_adj = this->H1_adj;
  const auto num_LM = static_cast<std::size_t>(mace->num_LM);
  const auto num_channels = static_cast<std::size_t>(mace->num_channels);
  checked_comm_value_count(static_cast<std::size_t>(n), "reverse-unpack");
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::unpack_reverse_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {0,0,0}, {n,mace->num_LM,mace->num_channels}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const int i = d_sendlist(ii);
      Kokkos::atomic_add(
        &H1_adj(i,LM,k),
        d_buf(
          (static_cast<std::size_t>(ii)*num_LM + static_cast<std::size_t>(LM))*num_channels
          + static_cast<std::size_t>(k)));
    });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::compute_no_domain_decomposition(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.view<DeviceType>().extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  }
  if (eflag_atom)
    Kokkos::deep_copy(k_eatom.template view<DeviceType>(), 0.0);

  const double r_cut = mace->r_cut;
  const double r_cut_squared = r_cut*r_cut;
  const bool stable_execution =
    mace_uses_prepared_execution(mace->streamed_edges);

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();

  auto map_style = atom->map_style;
  auto k_map_array = atomKK->k_map_array;
  auto k_map_hash = atomKK->k_map_hash;

  // node_indices, node_types, and num_neigh
  const int num_nodes = k_list->inum;
  bool rebuild_topology = !stable_execution || execution_graph_generation == 0
    || neighbor->ago == 0 || num_nodes != execution_num_nodes;
  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_nodes));
  auto mace_types = this->mace_types;
  int num_edges = execution_num_edges;
  if (rebuild_topology) {
    k_map_array.template sync<DeviceType>();
    Kokkos::deep_copy(num_neigh, 0);
    Kokkos::parallel_for("PairSymmetrixMACEKokkos::set_node_based_views",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = d_ilist(ii);
        node_indices(ii) = i;
        node_types(ii) = mace_types(type(i)-1);
        const double x_i = x(i,0);
        const double y_i = x(i,1);
        const double z_i = x(i,2);
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
          [&] (const int jj, int& num_neigh_ii) {
            const int j = (d_neighbors(i,jj) & NEIGHMASK);
            const double dx = x(j,0) - x_i;
            const double dy = x(j,1) - y_i;
            const double dz = x(j,2) - z_i;
            const double r_squared = dx*dx + dy*dy + dz*dz;
            if (stable_execution || r_squared < r_cut_squared) {
              num_neigh_ii += 1;
            }
          }, num_neigh(ii));
      });

    Kokkos::parallel_reduce("PairSymmetrixMACEKokkos::count_edges",
      num_nodes,
      KOKKOS_LAMBDA (const int ii, int& edge_count) {
        edge_count += num_neigh(ii);
      }, num_edges);
  }

  // first neighbor
  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_nodes));
  if (rebuild_topology)
    Kokkos::parallel_scan("PairSymmetrixMACEKokkos::populate_first_neigh",
      num_nodes,
      KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
          if (final) first_neigh(ii) = first_neigh_ii;
          first_neigh_ii += num_neigh(ii);
      });

  // neigh_indices, neigh_types, xyz, and r
  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  if (xyz.size() < 3*num_edges) Kokkos::realloc(xyz, 3*num_edges);
  if (r.size() < num_edges) Kokkos::realloc(r, num_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,num_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,num_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*num_edges));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,num_edges));
  const char* edge_kernel_label = rebuild_topology
    ? "PairSymmetrixMACEKokkos::set_edge_topology_geometry"
    : "PairSymmetrixMACEKokkos::update_edge_geometry";
  Kokkos::parallel_for(edge_kernel_label,
    num_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      int ij = first_neigh(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        const int j = (d_neighbors(i,jj) & NEIGHMASK);
        const double dx = x(j,0) - x_i;
        const double dy = x(j,1) - y_i;
        const double dz = x(j,2) - z_i;
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (stable_execution || r_squared < r_cut_squared) {
          const double distance = std::sqrt(r_squared);
          const double scale =
            stable_execution && distance >= r_cut
            ? r_cut/distance : 1.0;
          if (rebuild_topology) {
            const int j_local = AtomKokkos::map_kokkos<DeviceType>(
              tag(j),map_style,k_map_array,k_map_hash);
            neigh_indices(ij) = j_local;
            neigh_types(ij) = mace_types(type(j)-1);
          }
          xyz(3*ij) = scale*dx;
          xyz(3*ij+1) = scale*dy;
          xyz(3*ij+2) = scale*dz;
          r(ij) = scale*distance;
          ij += 1;
        }
      }
  });
  if (stable_execution)
    execution_geometry_refresh_count += 1.0;
  if (stable_execution && !rebuild_topology)
    execution_geometry_only_update_count += 1.0;

  if (stable_execution) {
    if (rebuild_topology) {
      execution_graph_generation = mace->prepare_factorized_graph_device(
        num_nodes, node_types, num_neigh, neigh_indices, neigh_types);
      execution_num_nodes = num_nodes;
      execution_num_edges = num_edges;
      execution_graph_rebuild_count += 1.0;
    }
    if (mace->has_field_coupling)
      mace->compute_node_energies_forces_field(
        num_nodes,
        Kokkos::View<const int*>(), Kokkos::View<const int*>(),
        Kokkos::View<const int*>(), Kokkos::View<const int*>(),
        xyz, r, electric_field, execution_graph_generation);
    else
      mace->compute_node_energies_forces(
        num_nodes,
        Kokkos::View<const int*>(), Kokkos::View<const int*>(),
        Kokkos::View<const int*>(), Kokkos::View<const int*>(),
        xyz, r, execution_graph_generation);
  } else if (mace->has_field_coupling) {
    mace->compute_node_energies_forces_field(
      num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, electric_field);
  } else {
    mace->compute_node_energies_forces(
      num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
  }

  if (eflag_global) {
    auto node_energies = mace->node_energies;
    double energy;
    Kokkos::parallel_reduce("PairSymmetrixMACEKokkos::energy_reduction",
      num_nodes,
      KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("PairSymmetrixMACEKokkos::extract_atomic_energies", num_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(node_indices(ii)) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
    k_eatom.sync_host();
  }

  auto mace_node_forces = mace->node_forces;
  Kokkos::parallel_for("PairSymmetrixMACEKokkos::force_reduction",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      double f_x, f_y, f_z;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
        [&] (const int jj, double& f_x, double& f_y, double& f_z) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          f_x += mace_node_forces(3*ij);
          f_y += mace_node_forces(3*ij+1);
          f_z += mace_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), mace_node_forces(3*ij));
          Kokkos::atomic_add(&f(j,1), mace_node_forces(3*ij+1));
          Kokkos::atomic_add(&f(j,2), mace_node_forces(3*ij+2));
        }, f_x, f_y, f_z);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&f(i,0), -f_x);
          Kokkos::atomic_add(&f(i,1), -f_y);
          Kokkos::atomic_add(&f(i,2), -f_z);
        });
    });

  if (vflag_global) {
    Kokkos::View<double*,Kokkos::LayoutRight> v("v", 6);
    Kokkos::deep_copy(v, 0.0);
    Kokkos::parallel_for("PairSymmetrixMACEKokkos::virial_reduction",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = node_indices(ii);
        double v_0, v_1, v_2, v_3, v_4, v_5;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj, double& v_0, double& v_1, double& v_2,
                             double& v_3, double& v_4, double& v_5) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            v_0 += x*f_x;
            v_1 += y*f_y;
            v_2 += z*f_z;
            v_3 += 0.5*(x*f_y + y*f_x);
            v_4 += 0.5*(x*f_z + z*f_x);
            v_5 += 0.5*(y*f_z + z*f_y);
          }, v_0, v_1, v_2, v_3, v_4, v_5);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&v(0), v_0);
          Kokkos::atomic_add(&v(1), v_1);
          Kokkos::atomic_add(&v(2), v_2);
          Kokkos::atomic_add(&v(3), v_3);
          Kokkos::atomic_add(&v(4), v_4);
          Kokkos::atomic_add(&v(5), v_5);
        });
      });
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    virial[0] += h_v(0);
    virial[1] += h_v(1);
    virial[2] += h_v(2);
    virial[3] += h_v(3);
    virial[4] += h_v(4);
    virial[5] += h_v(5);
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::compute_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.view<DeviceType>().extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  }
  if (eflag_atom)
    Kokkos::deep_copy(k_eatom.template view<DeviceType>(), 0.0);

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();

  const bool stable_execution =
    mace_uses_prepared_execution(mace->streamed_edges);

  // node_indices, node_types, and num_neigh
  const double r_cut = mace->r_cut;
  const double r_cut_squared = r_cut*r_cut;
  const int num_nodes = k_list->inum;
  const auto num_feature_nodes_size = static_cast<std::size_t>(atom->nlocal)
    + static_cast<std::size_t>(atom->nghost);
  if (num_feature_nodes_size
      > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    error->one(FLERR,
      "pair_style symmetrix/mace/kk local feature-node count exceeds INT_MAX");
  const int num_feature_nodes = static_cast<int>(num_feature_nodes_size);
  if (stable_execution && num_nodes != atom->nlocal)
    error->all(FLERR,
      "Factorized MPI execution requires one receiver for every owned atom");
  bool rebuild_topology = !stable_execution
    || execution_graph_generation == 0 || neighbor->ago == 0
    || num_nodes != execution_num_nodes
    || num_feature_nodes != execution_num_feature_nodes;
  std::uint64_t feature_fingerprint = 0;
  std::uint64_t receiver_fingerprint = 0;
  if (stable_execution) {
    Kokkos::parallel_reduce(
      "PairSymmetrixMACEKokkos::fingerprint_feature_nodes",
      num_feature_nodes,
      KOKKOS_LAMBDA (const int i, std::uint64_t &value) {
        std::uint64_t item = static_cast<std::uint64_t>(tag(i));
        item ^= static_cast<std::uint64_t>(type(i))*0x9e3779b97f4a7c15ULL;
        item ^= static_cast<std::uint64_t>(i)*0xbf58476d1ce4e5b9ULL;
        value += item*0x94d049bb133111ebULL;
      }, feature_fingerprint);
    Kokkos::parallel_reduce(
      "PairSymmetrixMACEKokkos::fingerprint_receivers",
      num_nodes,
      KOKKOS_LAMBDA (const int ii, std::uint64_t &value) {
        const int i = d_ilist(ii);
        std::uint64_t item = static_cast<std::uint64_t>(tag(i));
        item ^= static_cast<std::uint64_t>(type(i))*0x9e3779b97f4a7c15ULL;
        item ^= static_cast<std::uint64_t>(i)*0xbf58476d1ce4e5b9ULL;
        item ^= static_cast<std::uint64_t>(ii)*0x94d049bb133111ebULL;
        value += item*0xd6e8feb86659fd93ULL;
      }, receiver_fingerprint);
    feature_fingerprint ^= receiver_fingerprint
      +0x9e3779b97f4a7c15ULL+(feature_fingerprint<<6)+(feature_fingerprint>>2);
    rebuild_topology = rebuild_topology
      || feature_fingerprint != execution_topology_fingerprint;
  }
  if (!mace->single_layer_readout
      && (execution_graph_generation == 0 || neighbor->ago == 0)) {
    int max_ghost_atoms = 0;
    const int local_ghost_atoms = atom->nghost;
    MPI_Allreduce(
      &local_ghost_atoms, &max_ghost_atoms, 1, MPI_INT, MPI_MAX, world);
    // A swap's receive count is bounded by the destination's total ghosts;
    // its matching send count has the same bound on another rank.
    checked_comm_value_count(
      static_cast<std::size_t>(max_ghost_atoms),
      "per-message ghost upper bound");
  }
  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (feature_types.size() < num_feature_nodes)
    Kokkos::realloc(feature_types, num_feature_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_nodes));
  auto feature_types = Kokkos::subview(
    this->feature_types, Kokkos::make_pair(0,num_feature_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_nodes));
  auto mace_types = this->mace_types;
  std::uint64_t num_edges_wide = static_cast<std::uint64_t>(execution_num_edges);
  if (rebuild_topology) {
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::set_mpi_feature_types",
      num_feature_nodes,
      KOKKOS_LAMBDA (const int i) {
        feature_types(i) = mace_types(type(i)-1);
      });
    Kokkos::deep_copy(num_neigh, 0);
    Kokkos::parallel_for("PairSymmetrixMACEKokkos::set_mpi_node_views",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = d_ilist(ii);
        node_indices(ii) = i;
        node_types(ii) = mace_types(type(i)-1);
        const double x_i = x(i,0);
        const double y_i = x(i,1);
        const double z_i = x(i,2);
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
          [&] (const int jj, int& num_neigh_ii) {
            const int j = (d_neighbors(i,jj) & NEIGHMASK);
            const double dx = x(j,0) - x_i;
            const double dy = x(j,1) - y_i;
            const double dz = x(j,2) - z_i;
            const double r_squared = dx*dx + dy*dy + dz*dz;
            if (stable_execution || r_squared < r_cut_squared)
              num_neigh_ii += 1;
          }, num_neigh(ii));
      });

    Kokkos::parallel_reduce("PairSymmetrixMACEKokkos::count_mpi_edges",
      num_nodes,
      KOKKOS_LAMBDA (const int ii, std::uint64_t& edge_count) {
        edge_count += static_cast<std::uint64_t>(num_neigh(ii));
      }, num_edges_wide);
    if (num_edges_wide
        > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      error->one(
        FLERR,
        "pair_style symmetrix/mace/kk rank-local edge count exceeds INT_MAX");
  }
  const int num_edges = static_cast<int>(num_edges_wide);

  // first neighbor
  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_nodes));
  if (rebuild_topology)
    Kokkos::parallel_scan("PairSymmetrixMACEKokkos::set_mpi_first_neighbor",
      num_nodes,
      KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
          if (final) first_neigh(ii) = first_neigh_ii;
          first_neigh_ii += num_neigh(ii);
      });

  // Prepare topology before choosing whether geometry is retained or tiled.
  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,num_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,num_edges));
  if (rebuild_topology)
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::set_mpi_edge_topology",
      num_nodes,
      KOKKOS_LAMBDA (const int ii) {
        const int i = d_ilist(ii);
        int ij = first_neigh(ii);
        for (int jj=0; jj<d_numneigh(i); ++jj) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          bool include_edge = stable_execution;
          if (!stable_execution) {
            const double dx = x(j,0) - x(i,0);
            const double dy = x(j,1) - x(i,1);
            const double dz = x(j,2) - x(i,2);
            include_edge = dx*dx + dy*dy + dz*dz < r_cut_squared;
          }
          if (include_edge) {
            neigh_indices(ij) = j;
            neigh_types(ij) = mace_types(type(j)-1);
            ij += 1;
          }
        }
      });

  if (stable_execution) {
    if (rebuild_topology) {
      execution_graph_generation = mace->prepare_factorized_graph_device(
        num_nodes, num_feature_nodes,
        node_indices, node_types, feature_types,
        num_neigh, neigh_indices, neigh_types);
      execution_num_nodes = num_nodes;
      execution_num_feature_nodes = num_feature_nodes;
      execution_num_edges = num_edges;
      execution_topology_fingerprint = feature_fingerprint;
      execution_graph_rebuild_count += 1.0;
    }
  }

  const bool single_layer_tiled = stable_execution
    && mace->single_layer_readout
    && mace->execution_plan_report().selected_id == "mh0-single-layer-tiled-v1";
  const bool dual_layer_tiled = stable_execution
    && !mace->single_layer_readout
    && mace->execution_plan_report().selected_id == "mh0-dual-layer-tiled-v1";
  const bool fixed_workspace_tiled = single_layer_tiled || dual_layer_tiled;
  Kokkos::View<double*> xyz;
  Kokkos::View<double*> r;
  if (fixed_workspace_tiled) {
    if (this->xyz.extent(0) != 0 || this->r.extent(0) != 0) {
      Kokkos::fence("Release retained MPI edge geometry");
      this->xyz = {};
      this->r = {};
    }
    const auto position_extent = std::size_t(3)
      * static_cast<std::size_t>(num_feature_nodes);
    if (feature_positions.extent(0) < position_extent)
      Kokkos::realloc(feature_positions, position_extent);
    auto active_feature_positions = Kokkos::subview(
      feature_positions, Kokkos::make_pair(std::size_t(0), position_extent));
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::set_mpi_feature_positions",
      num_feature_nodes,
      KOKKOS_LAMBDA (const int i) {
        active_feature_positions(3*i) = x(i,0);
        active_feature_positions(3*i+1) = x(i,1);
        active_feature_positions(3*i+2) = x(i,2);
      });
    if (single_layer_tiled)
      mace->compute_factorized_single_layer_distributed_positions_evaluation(
        num_nodes, num_feature_nodes, active_feature_positions,
        execution_graph_generation);
    else
      mace->begin_factorized_distributed_positions_evaluation(
        num_nodes, num_feature_nodes, active_feature_positions,
        execution_graph_generation);
  } else {
    if (feature_positions.extent(0) != 0) {
      Kokkos::fence("Release MPI feature positions");
      feature_positions = {};
    }
    const auto xyz_extent = std::size_t(3)
      * static_cast<std::size_t>(num_edges);
    if (this->xyz.extent(0) < xyz_extent)
      Kokkos::realloc(this->xyz, xyz_extent);
    if (this->r.extent(0) < static_cast<std::size_t>(num_edges))
      Kokkos::realloc(this->r, static_cast<std::size_t>(num_edges));
    xyz = Kokkos::subview(
      this->xyz, Kokkos::make_pair(std::size_t(0), xyz_extent));
    r = Kokkos::subview(
      this->r,
      Kokkos::make_pair(
        std::size_t(0), static_cast<std::size_t>(num_edges)));
    Kokkos::parallel_for(
      rebuild_topology
        ? "PairSymmetrixMACEKokkos::set_mpi_edge_geometry"
        : "PairSymmetrixMACEKokkos::update_mpi_edge_geometry",
      num_nodes,
      KOKKOS_LAMBDA (const int ii) {
        const int i = d_ilist(ii);
        const double x_i = x(i,0);
        const double y_i = x(i,1);
        const double z_i = x(i,2);
        int ij = first_neigh(ii);
        for (int jj=0; jj<d_numneigh(i); ++jj) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (stable_execution || r_squared < r_cut_squared) {
            const double distance = std::sqrt(r_squared);
            const double scale = stable_execution && distance >= r_cut
              ? r_cut/distance : 1.0;
            xyz(3*ij) = scale*dx;
            xyz(3*ij+1) = scale*dy;
            xyz(3*ij+2) = scale*dz;
            r(ij) = scale*distance;
            ij += 1;
          }
        }
      });

    if (stable_execution) {
      const bool single_layer = mace->single_layer_readout;
      if (single_layer)
        mace->compute_factorized_single_layer_distributed_evaluation(
          num_nodes, num_feature_nodes, xyz, r, execution_graph_generation);
      else
        mace->begin_factorized_distributed_evaluation(
          num_nodes, num_feature_nodes, xyz, r, execution_graph_generation);
    }
  }

  if (stable_execution) {
    execution_geometry_refresh_count += 1.0;
    if (!rebuild_topology)
      execution_geometry_only_update_count += 1.0;

    const bool single_layer = mace->single_layer_readout;
    if (!single_layer) {
      if (dual_layer_tiled) {
        H1 = mace->H1;
      } else {
        const int h1_capacity = std::max(
          num_feature_nodes, mace->execution_planned_feature_node_capacity());
        if (H1.extent(0) < static_cast<std::size_t>(h1_capacity)
            || H1.extent(1) != static_cast<std::size_t>(mace->num_LM)
            || H1.extent(2) != static_cast<std::size_t>(mace->num_channels)) {
          Kokkos::fence("Replace Pair Symmetrix communicated H1 capacity");
          if (H1_adj.data() == H1.data()) H1_adj = {};
          H1 = {};
          H1 = decltype(H1)(
            Kokkos::view_alloc(
              "Pair Symmetrix communicated H1", Kokkos::WithoutInitializing),
            h1_capacity, mace->num_LM, mace->num_channels);
          execution_h1_allocation_count += 1.0;
        }
      }
    }
    if (rebuild_topology) {
      std::ostringstream capacity_message;
      capacity_message
        << "Symmetrix rank " << comm->me
        << " graph capacity: active receivers="
        << mace->execution_active_receiver_count()
        << ", features=" << mace->execution_active_feature_node_count()
        << ", edges=" << mace->execution_active_edge_count()
        << "; planned receivers=" << mace->execution_planned_receiver_capacity()
        << ", features=" << mace->execution_planned_feature_node_capacity()
        << ", edges=" << mace->execution_planned_edge_capacity()
        << "; planned bytes=" << mace->execution_planned_capacity_bytes()
        << "; policy=" << mace->low_memory_policy_name()
        << "; profile=" << execution_profile
        << "; plan=" << mace->execution_plan_report().selected_id
        << "; fixed workspace=" << (allow_fixed_workspace ? "allowed" : "disabled")
        << "; workspace receivers="
        << (dual_layer_tiled ? mace->dual_layer_workspace_active_receivers()
                             : mace->single_layer_workspace_active_receivers())
        << ", edges="
        << (dual_layer_tiled ? mace->dual_layer_workspace_active_edges()
                             : mace->single_layer_workspace_active_edges())
        << ", bytes="
        << (dual_layer_tiled ? mace->dual_layer_workspace_bytes()
                             : mace->single_layer_workspace_bytes())
        << ", batches="
        << (dual_layer_tiled ? mace->dual_layer_workspace_batch_count()
                             : mace->single_layer_workspace_batch_count())
        << "; pair edge geometry bytes="
        << sizeof(double)*(this->xyz.extent(0)+this->r.extent(0))
        << ", feature position bytes="
        << sizeof(double)*feature_positions.extent(0)
        << "; graph replacements=" << mace->factorized_graph_device_replacement_count
        << ", graph updates=" << mace->factorized_graph_device_update_count
        << ", geometry allocations=" << mace->execution_geometry_allocation_count
        << ", result allocations=" << mace->execution_result_allocation_count_value()
        << ", M0 replacements=" << mace->mh0_m0_forward_replacement_count_value()
        << ", M0 adjoint allocations="
        << mace->mh0_m0_adjoint_allocation_count_value()
        << ", M0 alias detaches=" << mace->mh0_m0_alias_detach_count_value()
        << ", M0 alias active="
        << (mace->mh0_m0_adjoint_alias_active() ? 1 : 0)
        << ", communicated H1 allocations=" << execution_h1_allocation_count
        << '\n';
      constexpr int capacity_message_bytes = 2048;
      std::array<char,capacity_message_bytes> local_capacity_message{};
      const std::string local_capacity_text = capacity_message.str();
      if (local_capacity_text.size() >= local_capacity_message.size())
        error->all(FLERR, "Symmetrix rank-local capacity diagnostic is too large");
      std::copy(
        local_capacity_text.begin(), local_capacity_text.end(),
        local_capacity_message.begin());
      std::vector<char> capacity_messages;
      if (comm->me == 0)
        capacity_messages.resize(
          static_cast<std::size_t>(comm->nprocs)*capacity_message_bytes);
      MPI_Gather(
        local_capacity_message.data(), capacity_message_bytes, MPI_CHAR,
        comm->me == 0 ? capacity_messages.data() : nullptr,
        capacity_message_bytes, MPI_CHAR, 0, world);
      if (comm->me == 0) {
        for (int rank = 0; rank < comm->nprocs; ++rank)
          utils::logmesg(
            lmp, std::string(
              capacity_messages.data()
              +static_cast<std::size_t>(rank)*capacity_message_bytes));
      }
    }
    if (!single_layer) {
      if (!dual_layer_tiled) {
        Kokkos::deep_copy(H1, Precision(0));
        const auto num_LM = mace->num_LM;
        const auto num_channels = mace->num_channels;
        const auto mace_H1 = mace->H1;
        const auto communicated_H1 = H1;
        Kokkos::parallel_for(
          "PairSymmetrixMACEKokkos::scatter_factorized_H1",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0,0,0}, {num_nodes,num_LM,num_channels}),
          KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
            communicated_H1(node_indices(ii),LM,k) = mace_H1(ii,LM,k);
          });
      }
      Kokkos::fence();
      const double forward_start = platform::walltime();
      comm->forward_comm(this);
      Kokkos::fence();
      const double forward_seconds = platform::walltime() - forward_start;
      execution_mpi_hidden_state_forward_seconds += forward_seconds;
      execution_mpi_hidden_state_seconds += forward_seconds;
      execution_mpi_hidden_state_forward_calls += 1.0;
      mace->continue_factorized_distributed_evaluation(
        H1,
        mace->has_field_coupling
          ? Kokkos::View<const double*>(electric_field)
          : Kokkos::View<const double*>());
      H1_adj = mace->H1_adj;
      Kokkos::fence();
      const double reverse_start = platform::walltime();
      comm->reverse_comm(this);
      Kokkos::fence();
      const double reverse_seconds = platform::walltime() - reverse_start;
      execution_mpi_hidden_state_reverse_seconds += reverse_seconds;
      execution_mpi_hidden_state_seconds += reverse_seconds;
      execution_mpi_hidden_state_reverse_calls += 1.0;
      H1_adj = {};
      mace->finish_factorized_distributed_evaluation();
    }
  } else {
  if (mace->node_energies.size() < num_nodes) Kokkos::realloc(mace->node_energies, num_nodes);
  if (mace->node_forces.size() < 3*num_edges) Kokkos::realloc(mace->node_forces, 3*num_edges);
  Kokkos::deep_copy(mace->node_energies, 0.0);
  Kokkos::deep_copy(mace->node_forces, 0.0);

  if (mace->has_zbl)
    mace->zbl.compute_ZBL(
      num_nodes, node_types, num_neigh, neigh_types,
      mace->atomic_numbers, r, xyz, mace->node_energies, mace->node_forces);

  if (mace->streamed_edges != MACEStreamedEdgesMode::materialized)
    mace->prepare_streamed_edge_schedule(num_nodes, num_edges, num_neigh);
  mace->compute_Y(xyz);

  if (mace->streamed_edges == MACEStreamedEdgesMode::generic)
    mace->compute_A0_streamed(num_nodes, node_types, num_neigh, neigh_types, r);
  else {
    mace->compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_A0(num_nodes, node_types, num_neigh, neigh_types);
  }
  mace->compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_nodes, node_types);
  if (mace->has_field_coupling)
    mace->compute_H1_product(num_nodes);
  else
    mace->compute_H1(num_nodes);

  // sort H1 contributions by i (rather than ii)
  const int num_h1_nodes = k_list->inum + atom->nghost;
  if (H1.extent(0) < num_h1_nodes)
    Kokkos::realloc(H1, num_h1_nodes, mace->num_LM, mace->num_channels);
  auto num_LM = mace->num_LM;
  auto num_channels = mace->num_channels;
  auto mace_H1 = mace->H1;
  auto H1 = this->H1;
  Kokkos::parallel_for("Sort H1",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {num_nodes,num_LM,num_channels}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const int i = d_ilist(ii);
      H1(i,LM,k) = mace_H1(ii,LM,k);
    });
  Kokkos::fence();
  const double forward_start = platform::walltime();
  comm->forward_comm(this);
  Kokkos::fence();
  const double forward_seconds = platform::walltime() - forward_start;
  execution_mpi_hidden_state_forward_seconds += forward_seconds;
  execution_mpi_hidden_state_seconds += forward_seconds;
  execution_mpi_hidden_state_forward_calls += 1.0;
  mace->H1 = H1;
  if (mace->has_field_coupling) {
    mace->compute_field_H1(num_h1_nodes, electric_field);
    mace->compute_H1_linear_up(num_h1_nodes);
  }

  if (mace->streamed_edges == MACEStreamedEdgesMode::materialized) {
    mace->compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_Phi1(num_nodes, num_neigh, neigh_indices);
  } else {
    mace->compute_Phi1_streamed(
      num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
  }
  mace->compute_A1(num_nodes);
  mace->compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_nodes, node_types);
  mace->compute_H2(num_nodes, node_types);

  mace->compute_readouts(num_nodes, node_types);

  mace->reverse_H2(num_nodes, node_types, false);
  mace->reverse_M1(num_nodes, node_types);
  mace->reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A1(num_nodes);
  if (mace->streamed_edges == MACEStreamedEdgesMode::materialized)
    mace->reverse_Phi1(
      num_nodes, num_neigh, neigh_indices, xyz, r, false, false);
  else
    mace->reverse_Phi1_streamed(
      num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
      xyz, r, false, false);

  if (mace->has_field_coupling) {
    mace->reverse_H1_linear_up(num_h1_nodes);
    mace->reverse_field_H1(num_h1_nodes, electric_field);
  }
  H1_adj = mace->H1_adj;
  Kokkos::fence();
  const double reverse_start = platform::walltime();
  comm->reverse_comm(this);
  Kokkos::fence();
  const double reverse_seconds = platform::walltime() - reverse_start;
  execution_mpi_hidden_state_reverse_seconds += reverse_seconds;
  execution_mpi_hidden_state_seconds += reverse_seconds;
  execution_mpi_hidden_state_reverse_calls += 1.0;
  H1_adj = {};

  if (mace->has_field_coupling)
    mace->reverse_H1_product(num_nodes);
  else
    mace->reverse_H1(num_nodes);
  mace->reverse_M0(num_nodes, node_types);
  mace->reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  if (mace->streamed_edges == MACEStreamedEdgesMode::generic)
    mace->reverse_A0_streamed(
      num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  else
    mace->reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  }

  if (eflag_global) {
    auto node_energies = mace->node_energies;
    double energy;
    Kokkos::parallel_reduce("Energy Reduction",
      num_nodes,
      KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("Extract Atomic Energies", num_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(node_indices(ii)) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
    k_eatom.sync_host();
  }

  auto mace_node_forces = mace->node_forces;
  if (fixed_workspace_tiled) {
    const auto reduced_forces = mace->atom_forces;
    Kokkos::parallel_for(
      "Extract Fixed-Workspace Reduced Forces", num_feature_nodes,
      KOKKOS_LAMBDA (const int i) {
        Kokkos::atomic_add(&f(i,0), reduced_forces(3*i));
        Kokkos::atomic_add(&f(i,1), reduced_forces(3*i+1));
        Kokkos::atomic_add(&f(i,2), reduced_forces(3*i+2));
      });
  } else {
    Kokkos::parallel_for("Force Reduction",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = node_indices(ii);
        double f_x, f_y, f_z;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj, double& f_x, double& f_y, double& f_z) {
            const int ij = first_neigh(ii) + jj;
            const int j = neigh_indices(ij);
            f_x += mace_node_forces(3*ij);
            f_y += mace_node_forces(3*ij+1);
            f_z += mace_node_forces(3*ij+2);
            Kokkos::atomic_add(&f(j,0), mace_node_forces(3*ij));
            Kokkos::atomic_add(&f(j,1), mace_node_forces(3*ij+1));
            Kokkos::atomic_add(&f(j,2), mace_node_forces(3*ij+2));
          }, f_x, f_y, f_z);
          Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
            Kokkos::atomic_add(&f(i,0), -f_x);
            Kokkos::atomic_add(&f(i,1), -f_y);
            Kokkos::atomic_add(&f(i,2), -f_z);
          });
      });
  }

  if (vflag_global) {
    if (fixed_workspace_tiled) {
      mace->reduce_prepared_stress(1.0, execution_graph_generation);
      const auto h_stress = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mace->stress_tensor);
      virial[0] -= h_stress(0);
      virial[1] -= h_stress(4);
      virial[2] -= h_stress(8);
      virial[3] -= 0.5*(h_stress(1)+h_stress(3));
      virial[4] -= 0.5*(h_stress(2)+h_stress(6));
      virial[5] -= 0.5*(h_stress(5)+h_stress(7));
    } else {
    Kokkos::View<double*,Kokkos::LayoutRight> v("v", 6);
    Kokkos::deep_copy(v, 0.0);
    Kokkos::parallel_for("Virial Reduction",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = node_indices(ii);
        double v_0, v_1, v_2, v_3, v_4, v_5;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj, double& v_0, double& v_1, double& v_2,
                             double& v_3, double& v_4, double& v_5) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            v_0 += x*f_x;
            v_1 += y*f_y;
            v_2 += z*f_z;
            v_3 += 0.5*(x*f_y + y*f_x);
            v_4 += 0.5*(x*f_z + z*f_x);
            v_5 += 0.5*(y*f_z + z*f_y);
          }, v_0, v_1, v_2, v_3, v_4, v_5);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&v(0), v_0);
          Kokkos::atomic_add(&v(1), v_1);
          Kokkos::atomic_add(&v(2), v_2);
          Kokkos::atomic_add(&v(3), v_3);
          Kokkos::atomic_add(&v(4), v_4);
          Kokkos::atomic_add(&v(5), v_5);
        });
      });
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    virial[0] += h_v(0);
    virial[1] += h_v(1);
    virial[2] += h_v(2);
    virial[3] += h_v(3);
    virial[4] += h_v(4);
    virial[5] += h_v(5);
    }
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}


/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixMACEKokkos<DeviceType, Precision>::compute_no_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.view<DeviceType>().extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  }
  if (eflag_atom)
    Kokkos::deep_copy(k_eatom.template view<DeviceType>(), 0.0);

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();

  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // locate ghosts within r_cut of locals
  auto is_local = Kokkos::Bitset(atom->nlocal+atom->nghost);
  Kokkos::parallel_for("fill is_local",
    list->inum,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      is_local.set(i);
    });
  Kokkos::fence();
  auto is_ghost = Kokkos::Bitset(atom->nlocal+atom->nghost);
  Kokkos::parallel_for("fill is_ghost",
    Kokkos::TeamPolicy<>(list->inum, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared<r_cut_squared and not is_local.test(j))
            is_ghost.set(j);
        });
    });
  Kokkos::fence();

  // set num_local_nodes and num_ghost_nodes
  const int num_local_nodes = list->inum;
  const int num_ghost_nodes = is_ghost.count();

  // collect indices of ghosts within r_cut of locals
  auto ghost_indices = Kokkos::View<int*>("ghost_indices", num_ghost_nodes);
  Kokkos::parallel_scan("populate ghost_indices",
    is_ghost.size(),
    KOKKOS_LAMBDA(int i, int& update, const bool final) {
    if (final && is_ghost.test(i))
      ghost_indices(update) = i;
    update += is_ghost.test(i);
  });
  Kokkos::fence();

  // populate node_indices, node_types, and num_neigh
  if (node_indices.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(node_indices, num_local_nodes+num_ghost_nodes);
  if (node_types.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(node_types, num_local_nodes+num_ghost_nodes);
  if (num_neigh.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(num_neigh, num_local_nodes+num_ghost_nodes);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  Kokkos::deep_copy(num_neigh, 0);
  auto mace_types = this->mace_types;
  Kokkos::parallel_for("populate node-based views",
    Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = (ii<num_local_nodes) ? d_ilist(ii) : ghost_indices(ii-num_local_nodes);
      node_indices(ii) = i;
      node_types(ii) = mace_types(type(i)-1);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj, int& num_neigh_ii) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared < r_cut_squared)
            num_neigh_ii += 1;
        }, num_neigh(ii));
    });
  Kokkos::fence();

  // count edges
  int num_local_edges;
  Kokkos::parallel_reduce("count local edges",
    num_local_nodes,
    KOKKOS_LAMBDA (const int ii, int& num_local_edges) {
      num_local_edges += num_neigh(ii);
    }, num_local_edges);
  int num_ghost_edges;
  Kokkos::parallel_reduce("count ghost edges",
    Kokkos::RangePolicy<>(num_local_nodes, num_local_nodes+num_ghost_nodes),
    KOKKOS_LAMBDA (const int ii, int& num_ghost_edges) {
      num_ghost_edges += num_neigh(ii);
    }, num_ghost_edges);
  Kokkos::fence();

  // first neighbor
  if (first_neigh.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(first_neigh, num_local_nodes+num_ghost_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  Kokkos::parallel_scan("populate first neighbor",
    num_local_nodes+num_ghost_nodes,
    KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
      if (final) first_neigh(ii) = first_neigh_ii;
      first_neigh_ii += num_neigh(ii);
    });
  Kokkos::fence();

  // populate neigh_indices, neigh_types, xyz, and r
  if (neigh_indices.size() < num_local_edges+num_ghost_edges)
    Kokkos::realloc(neigh_indices, num_local_edges+num_ghost_edges);
  if (neigh_types.size() < num_local_edges+num_ghost_edges)
    Kokkos::realloc(neigh_types, num_local_edges+num_ghost_edges);
  if (xyz.size() < 3*(num_local_edges+num_ghost_edges))
    Kokkos::realloc(xyz, 3*(num_local_edges+num_ghost_edges));
  if (r.size() < num_local_edges+num_ghost_edges)
    Kokkos::realloc(r, num_local_edges+num_ghost_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,num_local_edges+num_ghost_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,num_local_edges+num_ghost_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*(num_local_edges+num_ghost_edges)));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,num_local_edges+num_ghost_edges));
  Kokkos::parallel_for("populate edge-based views",
    num_local_nodes+num_ghost_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = node_indices(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      int ij = first_neigh(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        const int j = (d_neighbors(i,jj) & NEIGHMASK);
        const double dx = x(j,0) - x_i;
        const double dy = x(j,1) - y_i;
        const double dz = x(j,2) - z_i;
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (r_squared < r_cut_squared) {
          neigh_indices(ij) = j;
          neigh_types(ij) = mace_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
  });

  // populate neigh_ii_indices
  if (neigh_ii_indices.size() < num_local_edges)
    Kokkos::realloc(neigh_ii_indices, num_local_edges);
  auto neigh_ii_indices = Kokkos::subview(this->neigh_ii_indices, Kokkos::make_pair(0,num_local_edges));
  Kokkos::parallel_for("populate neigh_ii_indices",
    num_local_edges,
    KOKKOS_LAMBDA (const int ij) {
      const int j = neigh_indices(ij);
      for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
        if (node_indices(ii) == j) {
          neigh_ii_indices(ij) = ii;
          break;
        }
      }
    });
  Kokkos::fence();

  // ----- begin mace evaluation -----

  if (mace->node_energies.size() < num_local_nodes)
    Kokkos::realloc(mace->node_energies, num_local_nodes);
  Kokkos::deep_copy(mace->node_energies, 0.0);
  if (mace->node_forces.size() < 3*(num_local_edges+num_ghost_edges))
    Kokkos::realloc(mace->node_forces, 3*(num_local_edges+num_ghost_edges));
  Kokkos::deep_copy(mace->node_forces, 0.0);

  if (mace->has_zbl)
    mace->zbl.compute_ZBL(
      num_local_nodes, node_types, num_neigh, neigh_types,
      mace->atomic_numbers, r, xyz, mace->node_energies, mace->node_forces);

  if (mace->streamed_edges == MACEStreamedEdgesMode::generic)
    mace->prepare_streamed_edge_schedule(
      num_local_nodes+num_ghost_nodes,
      num_local_edges+num_ghost_edges,
      num_neigh);
  mace->compute_Y(xyz);

  if (mace->streamed_edges == MACEStreamedEdgesMode::generic)
    mace->compute_A0_streamed(
      num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
  else {
    mace->compute_R0(
      num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_A0(
      num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types);
  }
  mace->compute_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_local_nodes+num_ghost_nodes, node_types);
  if (mace->has_field_coupling) {
    mace->compute_H1_product(num_local_nodes+num_ghost_nodes);
    mace->compute_field_H1(num_local_nodes+num_ghost_nodes, electric_field);
    mace->compute_H1_linear_up(num_local_nodes+num_ghost_nodes);
  } else {
    mace->compute_H1(num_local_nodes+num_ghost_nodes);
  }

  if (mace->streamed_edges == MACEStreamedEdgesMode::materialized) {
    mace->compute_R1(num_local_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_Phi1(num_local_nodes, num_neigh, neigh_ii_indices);
  } else {
    mace->compute_Phi1_streamed(
      num_local_nodes, node_types, num_neigh, neigh_ii_indices, neigh_types, r);
  }
  mace->compute_A1(num_local_nodes);
  mace->compute_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_local_nodes, node_types);
  mace->compute_H2(num_local_nodes, node_types);

  mace->compute_readouts(num_local_nodes, node_types);

  mace->reverse_H2(num_local_nodes, node_types, false);
  mace->reverse_M1(num_local_nodes, node_types);
  mace->reverse_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A1(num_local_nodes);
  if (mace->streamed_edges == MACEStreamedEdgesMode::materialized)
    mace->reverse_Phi1(
      num_local_nodes, num_neigh, neigh_ii_indices, xyz, r, false, false);
  else
    mace->reverse_Phi1_streamed(
      num_local_nodes, node_types, num_neigh, neigh_ii_indices, neigh_types,
      xyz, r, false, false);

  if (mace->has_field_coupling) {
    mace->reverse_H1_linear_up(num_local_nodes+num_ghost_nodes);
    mace->reverse_field_H1(num_local_nodes+num_ghost_nodes, electric_field);
    mace->reverse_H1_product(num_local_nodes+num_ghost_nodes);
  } else {
    mace->reverse_H1(num_local_nodes+num_ghost_nodes);
  }
  mace->reverse_M0(num_local_nodes+num_ghost_nodes, node_types);
  mace->reverse_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, xyz, r);
  if (mace->streamed_edges == MACEStreamedEdgesMode::generic)
    mace->reverse_A0_streamed(
      num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types,
      xyz, r);
  else
    mace->reverse_A0(
      num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types,
      xyz, r);

  // ----- end mace evaluation -----

  if (eflag_global) {
    auto node_energies = mace->node_energies;
    double energy;
    Kokkos::parallel_reduce("energy reduction", num_local_nodes, KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("extract atomic energies", num_local_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(node_indices(ii)) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
    k_eatom.sync_host();
  }

  auto mace_node_forces = mace->node_forces;
  Kokkos::parallel_for("force reduction",
    Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      double f_x, f_y, f_z;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
        [&] (const int jj, double& f_x, double& f_y, double& f_z) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          f_x += mace_node_forces(3*ij);
          f_y += mace_node_forces(3*ij+1);
          f_z += mace_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), mace_node_forces(3*ij));
          Kokkos::atomic_add(&f(j,1), mace_node_forces(3*ij+1));
          Kokkos::atomic_add(&f(j,2), mace_node_forces(3*ij+2));
        }, f_x, f_y, f_z);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&f(i,0), -f_x);
          Kokkos::atomic_add(&f(i,1), -f_y);
          Kokkos::atomic_add(&f(i,2), -f_z);
        });
    });

  if (vflag_global) {
    Kokkos::View<double*,Kokkos::LayoutRight> v("v", 6);
    Kokkos::deep_copy(v, 0.0);
    Kokkos::parallel_for("virial reduction",
      Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = node_indices(ii);
        double v_0, v_1, v_2, v_3, v_4, v_5;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj, double& v_0, double& v_1, double& v_2,
                             double& v_3, double& v_4, double& v_5) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            v_0 += x*f_x;
            v_1 += y*f_y;
            v_2 += z*f_z;
            v_3 += 0.5*(x*f_y + y*f_x);
            v_4 += 0.5*(x*f_z + z*f_x);
            v_5 += 0.5*(y*f_z + z*f_y);
          }, v_0, v_1, v_2, v_3, v_4, v_5);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&v(0), v_0);
          Kokkos::atomic_add(&v(1), v_1);
          Kokkos::atomic_add(&v(2), v_2);
          Kokkos::atomic_add(&v(3), v_3);
          Kokkos::atomic_add(&v(4), v_4);
          Kokkos::atomic_add(&v(5), v_5);
        });
      });
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    virial[0] += h_v(0);
    virial[1] += h_v(1);
    virial[2] += h_v(2);
    virial[3] += h_v(3);
    virial[4] += h_v(4);
    virial[5] += h_v(5);
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairSymmetrixMACEKokkos<LMPDeviceType,double>;
template class PairSymmetrixMACEKokkos<LMPDeviceType,float>;
#ifdef LMP_KOKKOS_GPU
template class PairSymmetrixMACEKokkos<LMPHostType,double>;
template class PairSymmetrixMACEKokkos<LMPHostType,float>;
#endif
}
