/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(symmetrix/timing,ComputeSymmetrixTiming);
// clang-format on
#else

#ifndef LMP_COMPUTE_SYMMETRIX_TIMING_H
#define LMP_COMPUTE_SYMMETRIX_TIMING_H

#include "compute.h"

#include <array>
#include <string>

namespace LAMMPS_NS {

class ComputeSymmetrixTiming : public Compute {
 public:
  ComputeSymmetrixTiming(class LAMMPS *, int, char **);
  ~ComputeSymmetrixTiming() override;

  void init() override;
  void setup() override;
  double compute_scalar() override;
  void compute_vector() override;
  std::string get_thermo_colname(int) override;

 private:
  static constexpr int NUM_COUNTERS = 7;
  static constexpr int NUM_OUTPUTS = 14;

  enum CounterIndex {
    PAIR_SECONDS,
    PAIR_EVALUATIONS,
    FORWARD_SECONDS,
    REVERSE_SECONDS,
    HIDDEN_STATE_SECONDS,
    FORWARD_CALLS,
    REVERSE_CALLS
  };

  std::array<double *, NUM_COUNTERS> counters{};
  double *timing_enabled = nullptr;
  std::array<double, NUM_COUNTERS> baseline{};
  bigint last_reduced_step = -1;

  void resolve_counters();
  void snapshot();
  void reduce();
};

}    // namespace LAMMPS_NS

#endif
#endif
