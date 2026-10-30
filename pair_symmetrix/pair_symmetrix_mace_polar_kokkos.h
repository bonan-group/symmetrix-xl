/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
#define PairSymmetrixMACEPolarKokkosDeviceDouble PairSymmetrixMACEPolarKokkos<LMPDeviceType,double>
#define PairSymmetrixMACEPolarKokkosDeviceFloat PairSymmetrixMACEPolarKokkos<LMPDeviceType,float>

PairStyle(symmetrix/mace/polar/kk,PairSymmetrixMACEPolarKokkosDeviceDouble);
PairStyle(symmetrix/mace/polar/kk/device,PairSymmetrixMACEPolarKokkosDeviceDouble);
PairStyle(symmetrix/mace/polar/float32/kk,PairSymmetrixMACEPolarKokkosDeviceFloat);
PairStyle(symmetrix/mace/polar/float32/kk/device,PairSymmetrixMACEPolarKokkosDeviceFloat);

#undef PairSymmetrixMACEPolarKokkosDeviceDouble
#undef PairSymmetrixMACEPolarKokkosDeviceFloat
// clang-format on
#else

#ifndef LMP_PAIR_SYMMETRIX_MACE_POLAR_KOKKOS_H
#define LMP_PAIR_SYMMETRIX_MACE_POLAR_KOKKOS_H

#include "kokkos_base.h"
#include "neigh_list_kokkos.h"
#include "pair_kokkos.h"

#include "mace_polar_consumer.hpp"
#include "pair_symmetrix_mace_polar_admission.hpp"

#include <array>
#include <memory>
#include <string>

namespace LAMMPS_NS {

template<class DeviceType,typename Precision=double>
class PairSymmetrixMACEPolarKokkos : public Pair, public KokkosBase {
 public:
  explicit PairSymmetrixMACEPolarKokkos(class LAMMPS *);
  ~PairSymmetrixMACEPolarKokkos() override;

  void compute(int,int) override;
  void settings(int,char **) override;
  void coeff(int,char **) override;
  double init_one(int,int) override;
  void init_style() override;
  void *extract(const char *,int &) override;

 protected:
  void allocate();

 private:
  using Consumer=symmetrix::mace_polar::MacePolarConsumerT<Precision>;

  symmetrix::lammps::PolarGraphSettings graph_settings_;
  std::string prediction_head_;
  std::string jit_artifact_;
  std::string jit_target_="host";
  std::unique_ptr<Consumer> consumer_;
  Kokkos::View<int*,typename DeviceType::memory_space> atomic_numbers_;
  Kokkos::View<int*,typename DeviceType::memory_space> node_indices_;
  Kokkos::View<int*,typename DeviceType::memory_space> atom_to_node_;
  Kokkos::View<int*,typename DeviceType::memory_space> num_neigh_;
  Kokkos::View<int*,typename DeviceType::memory_space> first_neigh_;
  Kokkos::View<int*,typename DeviceType::memory_space> sources_;
  Kokkos::View<int*,typename DeviceType::memory_space> source_types_;
  Kokkos::View<int*,typename DeviceType::memory_space> pbc_;
  Kokkos::View<double*,typename DeviceType::memory_space> positions_;
  Kokkos::View<double*,typename DeviceType::memory_space> cell_;
  Kokkos::View<double*,typename DeviceType::memory_space> reference_xyz_;
  Kokkos::View<Precision*,typename DeviceType::memory_space> forces_;
  Kokkos::View<Precision*,typename DeviceType::memory_space> stress_;
  Kokkos::View<Precision*,typename DeviceType::memory_space> dipole_;
  Kokkos::View<double*,typename DeviceType::memory_space> virial_device_;
  Kokkos::View<int*,typename DeviceType::memory_space> type_atomic_numbers_;
  Kokkos::View<int,typename DeviceType::memory_space> invalid_mapping_;
  int prepared_nodes_=0;
  int prepared_edges_=0;
  bool prepared_graph_valid_=false;
  std::array<double,9> prepared_cell_{};
  std::array<int,3> prepared_pbc_{};
  double transaction_count_=0.0;
  double reverse_count_=0.0;
  double graph_rebuild_count_=0.0;
  double graph_reuse_count_=0.0;
  double fallback_count_=0.0;
  double backbone_fallback_count_=0.0;
  double polar_fallback_count_=0.0;
  double input_d2h_bytes_=0.0;
  double input_h2d_bytes_=0.0;
  double type_metadata_h2d_bytes_=0.0;
  std::size_t type_metadata_h2d_bytes_count_=0;
  bool type_metadata_upload_pending_=false;
  double public_output_d2h_bytes_=0.0;
  double force_output_d2h_bytes_=0.0;
  double output_h2d_bytes_=0.0;
  double topology_input_d2h_bytes_=0.0;
  double metadata_d2h_bytes_=0.0;
  double scalar_output_d2h_bytes_=0.0;
  double virial_output_d2h_bytes_=0.0;
  double geometry_publish_count_=0.0;
  double geometry_update_count_=0.0;
  double geometry_reuse_count_=0.0;
  double geometry_allocation_count_=0.0;
  double geometry_allocation_bytes_=0.0;
  double public_output_allocation_count_=0.0;
  double reciprocal_table_elements_=0.0;
  double active_nodes_=0.0;
  double active_edges_=0.0;
};

using PairSymmetrixMACEPolarKokkosDeviceDouble =
    PairSymmetrixMACEPolarKokkos<LMPDeviceType, double>;
using PairSymmetrixMACEPolarKokkosDeviceFloat =
    PairSymmetrixMACEPolarKokkos<LMPDeviceType, float>;

}    // namespace LAMMPS_NS

#endif
#endif
