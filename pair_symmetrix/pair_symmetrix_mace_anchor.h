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

// Base-header anchor for LAMMPS style discovery.
//
// LAMMPS finds a package-accelerated style by pairing the base header
// src/pair_<name>.h with its package counterpart src/<PKG>/pair_<name>_<ext>.h
// (FindStyleHeadersExt in cmake/Modules/StyleHeaderUtils.cmake): only a base
// header in the PAIR_CLASS list makes the package header and its .cpp part of
// the build.  Symmetrix-XL ships Kokkos pair styles only, so this anchor exists
// solely to make src/KOKKOS/pair_symmetrix_mace_kokkos.{h,cpp} discoverable.
// install.sh links it to src/pair_symmetrix_mace.h.
//
// It declares no styles.  The generated style_pair.h does include this header,
// so it must stay self-contained and must never register a style.

#ifdef PAIR_CLASS
// Kokkos pair styles are declared in src/KOKKOS/pair_symmetrix_mace_kokkos.h.
#endif
