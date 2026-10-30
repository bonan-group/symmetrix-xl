/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
------------------------------------------------------------------------- */

#include "pair_symmetrix_mace_polar_kokkos.h"

#include "atom_kokkos.h"
#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "model_metadata.hpp"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

using namespace LAMMPS_NS;

namespace {

const std::array<std::string,118> periodic_table={
  "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
  "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
  "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
  "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
  "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"};

bool polar_hybrid_substyle(const LAMMPS *lmp)
{
  return lmp->force->pair_style
      &&utils::strmatch(lmp->force->pair_style,"^hybrid");
}

}    // namespace

template<class DeviceType,typename Precision>
PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::
PairSymmetrixMACEPolarKokkos(LAMMPS *lmp) : Pair(lmp)
{
  single_enable=0;
  restartinfo=0;
  one_coeff=1;
  manybody_flag=1;
  no_virial_fdotr_compute=1;
  kokkosable=1;
  atomKK=static_cast<AtomKokkos *>(atom);
  execution_space=ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read=X_MASK|TYPE_MASK;
  datamask_modify=F_MASK|ENERGY_MASK|VIRIAL_MASK;
}

template<class DeviceType,typename Precision>
PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::
~PairSymmetrixMACEPolarKokkos()
{
  consumer_.reset();
  if(allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

template<class DeviceType,typename Precision>
void PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::allocate()
{
  allocated=1;
  memory->create(setflag,atom->ntypes+1,atom->ntypes+1,"pair:setflag");
  memory->create(cutsq,atom->ntypes+1,atom->ntypes+1,"pair:cutsq");
  for(int i=1;i<=atom->ntypes;++i)
    for(int j=i;j<=atom->ntypes;++j)setflag[i][j]=0;
}

template<class DeviceType,typename Precision>
void PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::settings(
  int narg,char **arg)
{
  try {
    symmetrix::lammps::PolarAdmissionRequest admission{
      comm->nprocs,atom->nlocal,static_cast<std::int64_t>(atom->natoms)};
    admission.hybrid_substyle=polar_hybrid_substyle(lmp);
    admission.atom_count_fits_native_index=
      symmetrix::lammps::polar_atom_count_fits_native_index(
        static_cast<std::int64_t>(atom->natoms));
    symmetrix::lammps::admit_polar_pair_style(admission);
  } catch(const std::exception& exception) {
    error->all(FLERR,"{}",exception.what());
  }
  graph_settings_={};
  prediction_head_.clear();
  jit_artifact_.clear();
  jit_target_="host";
  symmetrix::lammps::PolarJitArtifactSelection artifact_selection;
  for(int i=0;i<narg;++i) {
    const std::string token(arg[i]);
    const auto scalar=[&](double& value,const char *name) {
      if(i+1>=narg)error->all(FLERR,"pair_style symmetrix/mace/polar/kk {} requires a value",name);
      try {value=symmetrix::lammps::parse_finite_polar_setting(arg[++i],name);}
      catch(const std::exception&) {
        error->all(FLERR,"pair_style symmetrix/mace/polar/kk {} must be numeric",name);
      }
    };
    if(token=="total_charge")scalar(graph_settings_.total_charge,"total_charge");
    else if(token=="total_spin")scalar(graph_settings_.total_spin,"total_spin");
    else if(token=="fermi_level")scalar(graph_settings_.fermi_level,"fermi_level");
    else if(token=="electric_field") {
      if(i+3>=narg)error->all(FLERR,"pair_style symmetrix/mace/polar/kk electric_field requires three values");
      try {
        for(int component=0;component<3;++component)
          graph_settings_.electric_field[component]=
            symmetrix::lammps::parse_finite_polar_setting(
              arg[++i],"electric_field");
      } catch(const std::exception&) {
        error->all(FLERR,"pair_style symmetrix/mace/polar/kk electric_field must be numeric");
      }
    } else if(token=="polar_electrostatics") {
      if(i+1>=narg)error->all(FLERR,"pair_style symmetrix/mace/polar/kk polar_electrostatics requires a value");
      graph_settings_.electrostatics=arg[++i];
    } else if(token=="head") {
      if(i+1>=narg)error->all(FLERR,"pair_style symmetrix/mace/polar/kk head requires a value");
      prediction_head_=arg[++i];
    } else if(token=="jit_host_artifact"||token=="jit_device_artifact") {
      if(i+1>=narg)error->all(FLERR,"pair_style symmetrix/mace/polar/kk {} requires a value",token);
      try {
        symmetrix::lammps::select_polar_jit_artifact(
          artifact_selection,token,arg[++i]);
      } catch(const std::exception& exception) {
        error->all(FLERR,"{}",exception.what());
      }
      jit_target_=artifact_selection.target;
      jit_artifact_=artifact_selection.path;
    } else {
      error->all(FLERR,"Invalid pair_style symmetrix/mace/polar/kk setting: {}",token);
    }
  }
  try {symmetrix::lammps::validate_polar_graph_settings(graph_settings_);}
  catch(const std::exception& exception) {error->all(FLERR,"{}",exception.what());}
}

template<class DeviceType,typename Precision>
void PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::coeff(
  int narg,char **arg)
{
  symmetrix::lammps::PolarAdmissionRequest preflight{
    comm->nprocs,atom->nlocal,static_cast<std::int64_t>(atom->natoms)};
  preflight.hybrid_substyle=polar_hybrid_substyle(lmp);
  preflight.atom_count_fits_native_index=
    symmetrix::lammps::polar_atom_count_fits_native_index(
      static_cast<std::int64_t>(atom->natoms));
  try {symmetrix::lammps::admit_polar_pair_style(preflight);}
  catch(const std::exception& exception) {error->all(FLERR,"{}",exception.what());}
  if(narg!=atom->ntypes+3)error->all(FLERR,"Incorrect args for pair coefficients");
  auto atomic_numbers=std::vector<int>(atom->ntypes);
  for(int type=0;type<atom->ntypes;++type) {
    const auto found=std::find(periodic_table.begin(),periodic_table.end(),arg[type+3]);
    if(found==periodic_table.end())error->all(FLERR,"Unknown element {}",arg[type+3]);
    atomic_numbers[type]=static_cast<int>(found-periodic_table.begin())+1;
  }
  const bool model_is_polar=symmetrix_model_type(arg[2])=="MACE_Polar";
  symmetrix::lammps::PolarAdmissionRequest admission{
    comm->nprocs,atom->nlocal,static_cast<std::int64_t>(atom->natoms)};
  admission.hybrid_substyle=polar_hybrid_substyle(lmp);
  admission.atom_count_fits_native_index=
    symmetrix::lammps::polar_atom_count_fits_native_index(
      static_cast<std::int64_t>(atom->natoms));
  admission.model_is_polar=model_is_polar;
  admission.atom_count_fits_native_index=
    symmetrix::lammps::polar_atom_count_fits_native_index(
      static_cast<std::int64_t>(atom->natoms));
  try {symmetrix::lammps::admit_polar_pair_style(admission);}
  catch(const std::exception& exception) {error->all(FLERR,"{}",exception.what());}
  if(!allocated)allocate();
  try {
    consumer_=std::make_unique<Consumer>(arg[2],prediction_head_);
    if(jit_artifact_.empty()) {
      consumer_->set_streamed_edges("direct_streamed");
    } else {
      if(!consumer_->load_artifact(jit_artifact_,jit_target_))
        throw std::invalid_argument("MACE Polar JIT artifact was not admitted");
      consumer_->set_streamed_edges("direct");
    }
  } catch(const std::exception& exception) {
    error->all(FLERR,"Failed to load MACE Polar model: {}",exception.what());
  }
  type_atomic_numbers_=decltype(type_atomic_numbers_)(
    "polar LAMMPS atomic numbers",atom->ntypes);
  const Kokkos::View<const int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>
    host(atomic_numbers.data(),atomic_numbers.size());
  Kokkos::deep_copy(type_atomic_numbers_,host);
  type_metadata_h2d_bytes_count_=atomic_numbers.size()*sizeof(int);
  type_metadata_h2d_bytes_=static_cast<double>(type_metadata_h2d_bytes_count_);
  type_metadata_upload_pending_=true;
  for(int i=1;i<=atom->ntypes;++i)
    for(int j=i;j<=atom->ntypes;++j)setflag[i][j]=1;
}

template<class DeviceType,typename Precision>
double PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::init_one(int,int)
{
  return consumer_?consumer_->cutoff():0.0;
}

template<class DeviceType,typename Precision>
void PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::init_style()
{
  symmetrix::lammps::PolarAdmissionRequest admission{
    comm->nprocs,atom->nlocal,static_cast<std::int64_t>(atom->natoms)};
  admission.hybrid_substyle=polar_hybrid_substyle(lmp);
  try {symmetrix::lammps::admit_polar_pair_style(admission);}
  catch(const std::exception& exception) {error->all(FLERR,"{}",exception.what());}
  if(!consumer_)error->all(FLERR,"Pair coefficients are not set for symmetrix/mace/polar/kk");
  if(atom->map_style==Atom::MAP_NONE) {
    atom->map_user=Atom::MAP_YES;
    atom->map_style=Atom::MAP_YES;
    atom->map_init();
    atom->map_set();
  }
  auto request=neighbor->add_request(this,NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType>
    &&!std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
}

template<class DeviceType,typename Precision>
void PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::compute(
  int eflag,int vflag)
{
  ev_init(eflag,vflag,0);
  try {
    symmetrix::lammps::PolarAdmissionRequest admission{
      comm->nprocs,atom->nlocal,static_cast<std::int64_t>(atom->natoms)};
    admission.per_atom_energy_requested=eflag_atom;
    admission.per_atom_virial_requested=vflag_atom;
    admission.hybrid_substyle=polar_hybrid_substyle(lmp);
    admission.atom_count_fits_native_index=
      symmetrix::lammps::polar_atom_count_fits_native_index(
        static_cast<std::int64_t>(atom->natoms));
    symmetrix::lammps::admit_polar_pair_style(admission);
  } catch(const std::exception& exception) {error->all(FLERR,"{}",exception.what());}
  if(!consumer_)error->all(FLERR,"Pair coefficients are not set for symmetrix/mace/polar/kk");

  auto *k_list=static_cast<NeighListKokkos<DeviceType> *>(list);
  if(!k_list||list->inum!=atom->nlocal)
    error->all(FLERR,"MACE Polar requires one full neighbor-list entry per owned atom");
  const int nodes=atom->nlocal;
  if(nodes<0||static_cast<std::size_t>(nodes)
      >static_cast<std::size_t>(std::numeric_limits<int>::max()/3))
    error->all(FLERR,"MACE Polar atom count exceeds native index capacity");
  const auto force_values=std::size_t{3}*static_cast<std::size_t>(nodes);
  const std::array<double,9> host_cell{
    domain->xprd,0.0,0.0,domain->xy,domain->yprd,0.0,
    domain->xz,domain->yz,domain->zprd};
  const std::array<int,3> host_pbc{
    domain->xperiodic,domain->yperiodic,domain->zperiodic};
  const auto solver=symmetrix::execution::long_range::
    parse_long_range_solver_request(graph_settings_.electrostatics);
  // Reject an inadmissible boundary condition before scanning the neighbor
  // list or replacing any device allocation owned by this pair style.
  try {consumer_->preflight_solver(solver,host_pbc);}
  catch(const std::exception& exception) {
    error->all(FLERR,"MACE Polar solver request rejected: {}",exception.what());
  }
  const bool cell_changed=!prepared_graph_valid_
    ||host_cell!=prepared_cell_||host_pbc!=prepared_pbc_;
  const bool rebuild_topology=neighbor->ago==0||!prepared_graph_valid_
    ||nodes!=prepared_nodes_||cell_changed;

  if(atomic_numbers_.extent(0)!=static_cast<std::size_t>(nodes))
    Kokkos::realloc(atomic_numbers_,nodes);
  if(node_indices_.extent(0)!=static_cast<std::size_t>(nodes))
    Kokkos::realloc(node_indices_,nodes);
  if(atom_to_node_.extent(0)!=static_cast<std::size_t>(nodes))
    Kokkos::realloc(atom_to_node_,nodes);
  if(num_neigh_.extent(0)!=static_cast<std::size_t>(nodes))
    Kokkos::realloc(num_neigh_,nodes);
  if(first_neigh_.extent(0)!=static_cast<std::size_t>(nodes))
    Kokkos::realloc(first_neigh_,nodes);
  if(positions_.extent(0)!=force_values)
    Kokkos::realloc(positions_,force_values);
  if(forces_.extent(0)!=force_values)
    Kokkos::realloc(forces_,force_values);
  if(cell_.extent(0)!=9)Kokkos::realloc(cell_,9);
  if(pbc_.extent(0)!=3)Kokkos::realloc(pbc_,3);
  if(stress_.extent(0)!=6)Kokkos::realloc(stress_,6);
  if(dipole_.extent(0)!=3)Kokkos::realloc(dipole_,3);
  if(virial_device_.extent(0)!=6)Kokkos::realloc(virial_device_,6);
  symmetrix::lammps::ensure_polar_scalar_view_allocated(
    invalid_mapping_,"MACE Polar invalid mapping");

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);
  const auto x=atomKK->k_x.template view<DeviceType>();
  const auto type=atomKK->k_type.template view<DeviceType>();
  const auto tag=atomKK->k_tag.template view<DeviceType>();
  const auto f=atomKK->k_f.template view<DeviceType>();
  const auto d_numneigh=k_list->d_numneigh;
  const auto d_neighbors=k_list->d_neighbors;
  const auto mapping=type_atomic_numbers_;
  const auto d_ilist=k_list->d_ilist;
  const auto atomic_numbers=atomic_numbers_;
  const auto node_indices=node_indices_;
  const auto atom_to_node=atom_to_node_;
  const auto positions=positions_;
  Kokkos::deep_copy(atom_to_node,-1);
  Kokkos::parallel_for("MACE Polar LAMMPS compact inputs",nodes,
    KOKKOS_LAMBDA(const int node) {
      const int atom_index=d_ilist(node);
      node_indices(node)=d_ilist(node);
      atom_to_node(atom_index)=node;
      for(int component=0;component<3;++component)
        positions(std::size_t{3}*static_cast<std::size_t>(node)
          +static_cast<std::size_t>(component))=x(atom_index,component);
      atomic_numbers(node)=mapping(type(atom_index)-1);
    });
  Kokkos::deep_copy(cell_,Kokkos::View<const double*,Kokkos::HostSpace,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>(host_cell.data(),9));
  Kokkos::deep_copy(pbc_,Kokkos::View<const int*,Kokkos::HostSpace,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>(host_pbc.data(),3));

  if(rebuild_topology) {
    atomKK->k_map_array.template sync<DeviceType>();
    const auto map_style=atom->map_style;
    const auto k_map_array=atomKK->k_map_array;
    const auto k_map_hash=atomKK->k_map_hash;
    const auto degrees=num_neigh_;
    const auto invalid=invalid_mapping_;
    Kokkos::deep_copy(invalid_mapping_,0);
    Kokkos::parallel_for("MACE Polar LAMMPS receiver degrees",nodes,
      KOKKOS_LAMBDA(const int node) {
        const int receiver=d_ilist(node);
        const int degree=d_numneigh(receiver);
        if(degree<0) {
          Kokkos::atomic_exchange(&invalid(),1);
          degrees(node)=0;
        } else degrees(node)=degree;
      });
    std::int64_t edge_count=0;
    Kokkos::parallel_reduce("MACE Polar LAMMPS edge count",nodes,
      KOKKOS_LAMBDA(const int node,std::int64_t& total) {
        total+=static_cast<std::int64_t>(degrees(node));
      },edge_count);
    if(edge_count<0||edge_count>std::numeric_limits<int>::max()
        ||static_cast<std::uint64_t>(edge_count)
          >std::numeric_limits<std::size_t>::max()/3)
      error->all(FLERR,"MACE Polar neighbor graph exceeds native index capacity");
    const int edges=static_cast<int>(edge_count);
    if(sources_.extent(0)!=static_cast<std::size_t>(edges))
      Kokkos::realloc(sources_,edges);
    if(source_types_.extent(0)!=static_cast<std::size_t>(edges))
      Kokkos::realloc(source_types_,edges);
    if(reference_xyz_.extent(0)!=std::size_t{3}*static_cast<std::size_t>(edges))
      Kokkos::realloc(reference_xyz_,std::size_t{3}*static_cast<std::size_t>(edges));
    const auto offsets=first_neigh_;
    Kokkos::parallel_scan("MACE Polar LAMMPS receiver offsets",nodes,
      KOKKOS_LAMBDA(const int node,int& offset,const bool final) {
        if(final)offsets(node)=offset;
        offset+=degrees(node);
      });
    const auto sources=sources_;
    const auto source_types=source_types_;
    const auto reference_xyz=reference_xyz_;
    const double cutoff=consumer_->cutoff();
    Kokkos::parallel_for("MACE Polar LAMMPS full neighbor graph",nodes,
      KOKKOS_LAMBDA(const int node) {
        const int receiver=d_ilist(node);
        int edge=offsets(node);
        for(int jj=0;jj<d_numneigh(receiver);++jj,++edge) {
          const int neighbor=d_neighbors(receiver,jj)&NEIGHMASK;
          const int owned=AtomKokkos::map_kokkos<DeviceType>(
            tag(neighbor),map_style,k_map_array,k_map_hash);
          if(owned<0||owned>=nodes) {
            Kokkos::atomic_exchange(&invalid(),1);
            sources(edge)=0;
            source_types(edge)=0;
          } else {
            const int source=atom_to_node(owned);
            if(source<0||source>=nodes) {
              Kokkos::atomic_exchange(&invalid(),1);
              sources(edge)=0;
              source_types(edge)=0;
            } else {
              sources(edge)=source;
              source_types(edge)=mapping(type(owned)-1);
            }
          }
          const double dx=x(neighbor,0)-x(receiver,0);
          const double dy=x(neighbor,1)-x(receiver,1);
          const double dz=x(neighbor,2)-x(receiver,2);
          const double distance=Kokkos::sqrt(dx*dx+dy*dy+dz*dz);
          const double scale=distance>=cutoff&&distance>0.0
            ?cutoff/distance:1.0;
          reference_xyz(3*edge)=scale*dx;
          reference_xyz(3*edge+1)=scale*dy;
          reference_xyz(3*edge+2)=scale*dz;
        }
      });
    const auto invalid_host=Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(),invalid_mapping_);
    if(invalid_host()!=0)
      error->all(FLERR,
        "MACE Polar full neighbor list contains an unmapped periodic ghost tag");
    prepared_edges_=edges;
  }

  const bool need_forces=true;
  const bool need_stress=vflag_global;
  symmetrix::mace_polar::ConsumerResult result;
  try {
    result=consumer_->evaluate_prepared_views(
      atomic_numbers_,positions_,cell_,pbc_,num_neigh_,sources_,source_types_,
      reference_xyz_,rebuild_topology,graph_settings_.total_charge,
      graph_settings_.total_spin,graph_settings_.fermi_level,
      graph_settings_.electric_field,solver,need_forces,need_stress,false,
      forces_,stress_,dipole_);
  } catch(const std::exception& exception) {
    error->all(FLERR,"MACE Polar evaluation failed: {}",exception.what());
  }
  if(eflag_global)eng_vdwl+=result.energy;
  symmetrix::mace_polar::add_polar_public_forces_device(
    forces_,0,node_indices_,f,static_cast<std::size_t>(nodes));
  atomKK->modified(execution_space,F_MASK);
  if(vflag_global) {
    const double volume=domain->xprd*domain->yprd*domain->zprd;
    const bool periodic=host_pbc[0]||host_pbc[1]||host_pbc[2];
    symmetrix::mace_polar::map_polar_stress_to_virial_device(
      stress_,volume,periodic,virial_device_);
    const auto mapped=Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(),virial_device_);
    for(int component=0;component<6;++component)virial[component]+=mapped(component);
  }
  prepared_graph_valid_=true;
  prepared_nodes_=nodes;
  prepared_cell_=host_cell;
  prepared_pbc_=host_pbc;
  transaction_count_=static_cast<double>(result.diagnostics.transaction_count);
  reverse_count_=static_cast<double>(result.diagnostics.reverse_count);
  graph_rebuild_count_=static_cast<double>(result.diagnostics.graph_rebuild_count);
  graph_reuse_count_=static_cast<double>(result.diagnostics.graph_reuse_count);
  fallback_count_=static_cast<double>(result.diagnostics.evaluation.fallback_count);
  backbone_fallback_count_=static_cast<double>(
    result.diagnostics.evaluation.backbone_fallback_count);
  polar_fallback_count_=static_cast<double>(
    result.diagnostics.evaluation.polar_fallback_count);
  input_d2h_bytes_=static_cast<double>(result.diagnostics.input_d2h_bytes);
  input_h2d_bytes_=static_cast<double>(
    symmetrix::lammps::polar_pair_input_h2d_bytes(
      result.diagnostics.input_h2d_bytes,
      type_metadata_upload_pending_?type_metadata_h2d_bytes_count_:0));
  type_metadata_upload_pending_=false;
  public_output_d2h_bytes_=static_cast<double>(
    result.diagnostics.public_output_d2h_bytes);
  force_output_d2h_bytes_=static_cast<double>(
    result.diagnostics.force_output_d2h_bytes);
  output_h2d_bytes_=static_cast<double>(result.diagnostics.output_h2d_bytes);
  topology_input_d2h_bytes_=static_cast<double>(
    result.diagnostics.topology_input_d2h_bytes);
  metadata_d2h_bytes_=static_cast<double>(result.diagnostics.metadata_d2h_bytes);
  scalar_output_d2h_bytes_=static_cast<double>(
    result.diagnostics.scalar_output_d2h_bytes);
  virial_output_d2h_bytes_=vflag_global?6.0*sizeof(double):0.0;
  public_output_d2h_bytes_+=virial_output_d2h_bytes_;
  geometry_publish_count_=static_cast<double>(
    result.diagnostics.geometry_publish_count);
  geometry_update_count_=static_cast<double>(
    result.diagnostics.geometry_update_count);
  geometry_reuse_count_=static_cast<double>(
    result.diagnostics.geometry_reuse_count);
  geometry_allocation_count_=static_cast<double>(
    result.diagnostics.geometry_allocation_count);
  geometry_allocation_bytes_=static_cast<double>(
    result.diagnostics.geometry_allocation_bytes);
  public_output_allocation_count_=static_cast<double>(
    result.diagnostics.public_output_allocation_count);
  reciprocal_table_elements_=static_cast<double>(
    result.diagnostics.evaluation.reciprocal_table_elements);
  active_nodes_=static_cast<double>(result.diagnostics.active_nodes);
  active_edges_=static_cast<double>(result.diagnostics.active_edges);
}

template<class DeviceType,typename Precision>
void *PairSymmetrixMACEPolarKokkos<DeviceType,Precision>::extract(
  const char *name,int &dim)
{
  dim=0;
  if(std::strcmp(name,"symmetrix_polar_transaction_count")==0)return &transaction_count_;
  if(std::strcmp(name,"symmetrix_polar_reverse_count")==0)return &reverse_count_;
  if(std::strcmp(name,"symmetrix_polar_graph_rebuild_count")==0)return &graph_rebuild_count_;
  if(std::strcmp(name,"symmetrix_polar_graph_reuse_count")==0)return &graph_reuse_count_;
  if(std::strcmp(name,"symmetrix_polar_fallback_count")==0)return &fallback_count_;
  if(std::strcmp(name,"symmetrix_polar_backbone_fallback_count")==0)
    return &backbone_fallback_count_;
  if(std::strcmp(name,"symmetrix_polar_portable_fallback_count")==0)
    return &polar_fallback_count_;
  if(std::strcmp(name,"symmetrix_polar_input_d2h_bytes")==0)return &input_d2h_bytes_;
  if(std::strcmp(name,"symmetrix_polar_input_h2d_bytes")==0)return &input_h2d_bytes_;
  if(std::strcmp(name,"symmetrix_polar_type_metadata_h2d_bytes")==0)
    return &type_metadata_h2d_bytes_;
  if(std::strcmp(name,"symmetrix_polar_public_output_d2h_bytes")==0)
    return &public_output_d2h_bytes_;
  if(std::strcmp(name,"symmetrix_polar_force_output_d2h_bytes")==0)
    return &force_output_d2h_bytes_;
  if(std::strcmp(name,"symmetrix_polar_output_h2d_bytes")==0)return &output_h2d_bytes_;
  if(std::strcmp(name,"symmetrix_polar_topology_input_d2h_bytes")==0)
    return &topology_input_d2h_bytes_;
  if(std::strcmp(name,"symmetrix_polar_metadata_d2h_bytes")==0)
    return &metadata_d2h_bytes_;
  if(std::strcmp(name,"symmetrix_polar_scalar_output_d2h_bytes")==0)
    return &scalar_output_d2h_bytes_;
  if(std::strcmp(name,"symmetrix_polar_virial_output_d2h_bytes")==0)
    return &virial_output_d2h_bytes_;
  if(std::strcmp(name,"symmetrix_polar_geometry_publish_count")==0)
    return &geometry_publish_count_;
  if(std::strcmp(name,"symmetrix_polar_geometry_update_count")==0)
    return &geometry_update_count_;
  if(std::strcmp(name,"symmetrix_polar_geometry_reuse_count")==0)
    return &geometry_reuse_count_;
  if(std::strcmp(name,"symmetrix_polar_geometry_allocation_count")==0)
    return &geometry_allocation_count_;
  if(std::strcmp(name,"symmetrix_polar_geometry_allocation_bytes")==0)
    return &geometry_allocation_bytes_;
  if(std::strcmp(name,"symmetrix_polar_public_output_allocation_count")==0)
    return &public_output_allocation_count_;
  if(std::strcmp(name,"symmetrix_polar_reciprocal_table_elements")==0)
    return &reciprocal_table_elements_;
  if(std::strcmp(name,"symmetrix_polar_active_nodes")==0)return &active_nodes_;
  if(std::strcmp(name,"symmetrix_polar_active_edges")==0)return &active_edges_;
  return nullptr;
}

template class PairSymmetrixMACEPolarKokkos<LMPDeviceType,double>;
template class PairSymmetrixMACEPolarKokkos<LMPDeviceType,float>;
