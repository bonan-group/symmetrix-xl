/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
#define PairSymmetrixMACEKokkosDeviceDouble PairSymmetrixMACEKokkos<LMPDeviceType,double>
#define PairSymmetrixMACEKokkosHostDouble PairSymmetrixMACEKokkos<LMPHostType,double>
#define PairSymmetrixMACEKokkosDeviceFloat PairSymmetrixMACEKokkos<LMPDeviceType,float>
#define PairSymmetrixMACEKokkosHostFloat PairSymmetrixMACEKokkos<LMPHostType,float>

PairStyle(symmetrix/mace/kk,PairSymmetrixMACEKokkosDeviceDouble);
PairStyle(symmetrix/mace/kk/device,PairSymmetrixMACEKokkosDeviceDouble);
PairStyle(symmetrix/mace/kk/host,PairSymmetrixMACEKokkosHostDouble);
PairStyle(symmetrix/mace/float32/kk,PairSymmetrixMACEKokkosDeviceFloat);
PairStyle(symmetrix/mace/float32/kk/device,PairSymmetrixMACEKokkosDeviceFloat);
PairStyle(symmetrix/mace/float32/kk/host,PairSymmetrixMACEKokkosHostFloat);

#undef PairSymmetrixMACEKokkosDeviceDouble
#undef PairSymmetrixMACEKokkosHostDouble
#undef PairSymmetrixMACEKokkosDeviceFloat
#undef PairSymmetrixMACEKokkosHostFloat
// clang-format on
#else

#ifndef LMP_PAIR_SYMMETRIX_MACE_KOKKOS_H
#define LMP_PAIR_SYMMETRIX_MACE_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

#include "mace_kokkos.hpp"

namespace LAMMPS_NS {

template<class DeviceType, typename Precision = double>
class PairSymmetrixMACEKokkos : public Pair, public KokkosBase {

 public:
  PairSymmetrixMACEKokkos(class LAMMPS *);
  ~PairSymmetrixMACEKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void *extract(const char *, int &) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&, int, int*) override;
  void unpack_forward_comm(int, int, double *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  int pack_reverse_comm(int, int, double *) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&) override;
  void compute_no_domain_decomposition(int, int);
  void compute_mpi_message_passing(int, int);
 void compute_no_mpi_message_passing(int, int);

 protected:
  std::string mode;
  std::string prediction_head;
  std::string streamed_edges;
  std::string execution_profile = "capacity";
  bool execution_profile_set = false;
  bool low_memory_alias_set = false;
  bool low_memory_alias_value = false;
  bool allow_fixed_workspace = false;
  std::string debug_execution_plan;
  int debug_single_layer_workspace_receivers = 0;
  int debug_dual_layer_workspace_receivers = 0;
  std::string jit_host_artifact;
  std::string jit_device_artifact;
  std::string jit_m0_device_artifact;
  std::string jit_r0_device_artifact;
  std::string jit_m0_device_schedule = "chunk32";
  bool jit_m0_device_schedule_set = false;
  int jit_device_blocks_per_compute_unit = 8;
  std::uint64_t execution_graph_generation = 0;
  int execution_num_nodes = 0;
  int execution_num_feature_nodes = 0;
  int execution_num_edges = 0;
  std::uint64_t execution_topology_fingerprint = 0;
  double execution_pair_evaluation_count = 0.0;
  double execution_pair_seconds = 0.0;
  double execution_timing_enabled = 0.0;
  double execution_mpi_hidden_state_forward_seconds = 0.0;
  double execution_mpi_hidden_state_reverse_seconds = 0.0;
  double execution_mpi_hidden_state_seconds = 0.0;
  double execution_mpi_hidden_state_forward_calls = 0.0;
  double execution_mpi_hidden_state_reverse_calls = 0.0;
  double execution_graph_rebuild_count = 0.0;
  double execution_geometry_refresh_count = 0.0;
  double execution_geometry_only_update_count = 0.0;
  double execution_h1_allocation_count = 0.0;
  double execution_mpi_staged_packet_d2h_bytes = 0.0;
  double execution_mpi_staged_packet_h2d_bytes = 0.0;
  double execution_mpi_staged_index_h2d_bytes = 0.0;
  bool electric_field_set;
  std::unique_ptr<MACEKokkos<Precision>> mace;
  Kokkos::View<int*> mace_types;
  Kokkos::View<double*> electric_field;
  Kokkos::View<Precision***,Kokkos::LayoutRight> H1, H1_adj;

  // neighbor list variables
  Kokkos::View<int*> node_indices;
  Kokkos::View<int*> node_types;
  Kokkos::View<int*> feature_types;
  Kokkos::View<int*> num_neigh;
  Kokkos::View<int*> first_neigh;
  Kokkos::View<int*> neigh_types;
  Kokkos::View<int*> neigh_indices;
  Kokkos::View<int*> neigh_ii_indices;
  Kokkos::View<double*> xyz;
  Kokkos::View<double*> r;
  Kokkos::View<double*> feature_positions;
  Kokkos::View<int*> legacy_comm_indices;
  Kokkos::View<double*> legacy_comm_packet;

  const std::array<std::string,118> periodic_table =
    { "H", "He",
     "Li", "Be",                                                              "B",  "C",  "N",  "O",  "F", "Ne",
     "Na", "Mg",                                                             "Al", "Si",  "P",  "S", "Cl", "Ar",
     "K",  "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
     "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe",
     "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                       "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
     "Fr", "Ra", "Ac", "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                       "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

  virtual void allocate();
  void ensure_legacy_comm_capacity(std::size_t, std::size_t);
  int checked_comm_value_count(std::size_t, const char *);

 private:
  DAT::ttransform_kkacc_1d k_eatom;

};

using PairSymmetrixMACEKokkosDeviceDouble =
    PairSymmetrixMACEKokkos<LMPDeviceType, double>;
using PairSymmetrixMACEKokkosHostDouble =
    PairSymmetrixMACEKokkos<LMPHostType, double>;
using PairSymmetrixMACEKokkosDeviceFloat =
    PairSymmetrixMACEKokkos<LMPDeviceType, float>;
using PairSymmetrixMACEKokkosHostFloat =
    PairSymmetrixMACEKokkos<LMPHostType, float>;

}    // namespace LAMMPS_NS

#endif
#endif
