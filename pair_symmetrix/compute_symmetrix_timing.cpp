/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_symmetrix_timing.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "pair.h"
#include "update.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

using namespace LAMMPS_NS;

namespace {

constexpr std::array<const char *, 7> counter_names = {
  "symmetrix_pair_seconds",
  "symmetrix_pair_evaluation_count",
  "symmetrix_mpi_hidden_state_forward_seconds",
  "symmetrix_mpi_hidden_state_reverse_seconds",
  "symmetrix_mpi_hidden_state_seconds",
  "symmetrix_mpi_hidden_state_forward_calls",
  "symmetrix_mpi_hidden_state_reverse_calls"
};

constexpr std::array<const char *, 14> column_names = {
  "SxPair", "SxNonComm", "SxH1Comm", "SxH1CommPct", "SxH1Fwd", "SxH1Rev",
  "SxH1PctMin", "SxH1PctAvg", "SxH1PctMax", "SxH1Imbal", "SxFwdCallMin",
  "SxFwdCallMax", "SxRevCallMin", "SxRevCallMax"
};

double counter_delta(double current, double initial)
{
  return current >= initial ? current - initial : current;
}

}    // namespace

/* ---------------------------------------------------------------------- */

ComputeSymmetrixTiming::ComputeSymmetrixTiming(
  LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR, "Illegal compute symmetrix/timing command");
  if (igroup) error->all(FLERR, 1, "Compute symmetrix/timing must use group all");

  scalar_flag = vector_flag = 1;
  size_vector = NUM_OUTPUTS;
  extscalar = extvector = 0;
  thermo_modify_colname = 1;
  vector = new double[NUM_OUTPUTS]();
}

/* ---------------------------------------------------------------------- */

ComputeSymmetrixTiming::~ComputeSymmetrixTiming()
{
  // Force is null during LAMMPS shutdown because pair styles are destroyed
  // before computes.  For an explicit uncompute, release this compute's opt-in
  // only when the same pair style is still live.
  if (force && timing_enabled) {
    Pair *pair = force->pair_match("^symmetrix/mace", 0);
    if (pair) {
      int dim = -1;
      auto *live_timing_enabled =
        static_cast<double *>(pair->extract("symmetrix_timing_enabled", dim));
      if (live_timing_enabled == timing_enabled && dim == 0 && *timing_enabled > 0.0)
        *timing_enabled -= 1.0;
    }
  }
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeSymmetrixTiming::resolve_counters()
{
  Pair *pair = force->pair_match("^symmetrix/mace", 0);
  if (!pair)
    error->all(
      FLERR,
      "Compute symmetrix/timing requires exactly one compatible Symmetrix pair style");

  for (int i = 0; i < NUM_COUNTERS; ++i) {
    int dim = -1;
    counters[i] = static_cast<double *>(pair->extract(counter_names[i], dim));
    if (!counters[i] || dim != 0)
      error->all(
        FLERR,
        "Compute symmetrix/timing could not find a compatible Symmetrix pair style");
  }

  int dim = -1;
  auto *resolved_timing_enabled =
    static_cast<double *>(pair->extract("symmetrix_timing_enabled", dim));
  if (!resolved_timing_enabled || dim != 0)
    error->all(FLERR, "Symmetrix pair style does not support synchronized timing");
  timing_enabled = resolved_timing_enabled;
  *timing_enabled += 1.0;
}

/* ---------------------------------------------------------------------- */

void ComputeSymmetrixTiming::init()
{
  resolve_counters();
}

/* ---------------------------------------------------------------------- */

void ComputeSymmetrixTiming::snapshot()
{
  for (int i = 0; i < NUM_COUNTERS; ++i) baseline[i] = *counters[i];
  scalar = 0.0;
  std::fill(vector, vector + NUM_OUTPUTS, 0.0);
  last_reduced_step = -1;
}

/* ---------------------------------------------------------------------- */

void ComputeSymmetrixTiming::setup()
{
  // Verlet calls compute setup after its unmeasured setup force evaluation.
  snapshot();
}

/* ---------------------------------------------------------------------- */

void ComputeSymmetrixTiming::reduce()
{
  if (last_reduced_step == update->ntimestep) return;
  if (*timing_enabled == 0.0)
    error->all(FLERR, "Symmetrix synchronized pair timing was disabled during the run");

  std::array<double, NUM_COUNTERS> local{};
  for (int i = 0; i < NUM_COUNTERS; ++i)
    local[i] = counter_delta(*counters[i], baseline[i]);

  std::vector<double> ranks(
    static_cast<std::size_t>(comm->nprocs) * NUM_COUNTERS);
  MPI_Allgather(
    local.data(), NUM_COUNTERS, MPI_DOUBLE,
    ranks.data(), NUM_COUNTERS, MPI_DOUBLE, world);

  int critical_rank = 0;
  for (int rank = 1; rank < comm->nprocs; ++rank) {
    if (ranks[static_cast<std::size_t>(rank) * NUM_COUNTERS + PAIR_SECONDS]
        > ranks[static_cast<std::size_t>(critical_rank) * NUM_COUNTERS + PAIR_SECONDS])
      critical_rank = rank;
  }

  const double *critical = ranks.data()
    + static_cast<std::size_t>(critical_rank) * NUM_COUNTERS;
  const double pair_seconds = critical[PAIR_SECONDS];
  const double comm_seconds = critical[HIDDEN_STATE_SECONDS];
  const double evaluations = critical[PAIR_EVALUATIONS];
  const double atom_evaluations = static_cast<double>(atom->natoms) * evaluations;
  const double time_scale = atom_evaluations > 0.0 ? 1.0e6 / atom_evaluations : 0.0;

  vector[0] = pair_seconds * time_scale;
  vector[1] = std::max(0.0, pair_seconds - comm_seconds) * time_scale;
  vector[2] = comm_seconds * time_scale;
  vector[3] = pair_seconds > 0.0
    ? 100.0 * comm_seconds / pair_seconds
    : std::numeric_limits<double>::quiet_NaN();
  vector[4] = critical[FORWARD_SECONDS] * time_scale;
  vector[5] = critical[REVERSE_SECONDS] * time_scale;

  double percent_min = std::numeric_limits<double>::max();
  double percent_sum = 0.0;
  double percent_max = 0.0;
  double comm_sum = 0.0;
  double comm_max = 0.0;
  double forward_calls_min = std::numeric_limits<double>::max();
  double forward_calls_max = 0.0;
  double reverse_calls_min = std::numeric_limits<double>::max();
  double reverse_calls_max = 0.0;

  for (int rank = 0; rank < comm->nprocs; ++rank) {
    const double *values = ranks.data() + static_cast<std::size_t>(rank) * NUM_COUNTERS;
    const double rank_pair = values[PAIR_SECONDS];
    const double rank_comm = values[HIDDEN_STATE_SECONDS];
    const double rank_evaluations = values[PAIR_EVALUATIONS];
    const double rank_percent = rank_pair > 0.0 ? 100.0 * rank_comm / rank_pair : 0.0;
    const double forward_calls_per_evaluation = rank_evaluations > 0.0
      ? values[FORWARD_CALLS] / rank_evaluations : 0.0;
    const double reverse_calls_per_evaluation = rank_evaluations > 0.0
      ? values[REVERSE_CALLS] / rank_evaluations : 0.0;

    percent_min = std::min(percent_min, rank_percent);
    percent_sum += rank_percent;
    percent_max = std::max(percent_max, rank_percent);
    comm_sum += rank_comm;
    comm_max = std::max(comm_max, rank_comm);
    forward_calls_min = std::min(forward_calls_min, forward_calls_per_evaluation);
    forward_calls_max = std::max(forward_calls_max, forward_calls_per_evaluation);
    reverse_calls_min = std::min(reverse_calls_min, reverse_calls_per_evaluation);
    reverse_calls_max = std::max(reverse_calls_max, reverse_calls_per_evaluation);
  }

  const double rank_count = static_cast<double>(comm->nprocs);
  const double comm_mean = comm_sum / rank_count;
  vector[6] = percent_min;
  vector[7] = percent_sum / rank_count;
  vector[8] = percent_max;
  vector[9] = comm_mean > 0.0 ? comm_max / comm_mean : 0.0;
  vector[10] = forward_calls_min;
  vector[11] = forward_calls_max;
  vector[12] = reverse_calls_min;
  vector[13] = reverse_calls_max;
  scalar = vector[3];
  last_reduced_step = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

double ComputeSymmetrixTiming::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  reduce();
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeSymmetrixTiming::compute_vector()
{
  invoked_vector = update->ntimestep;
  reduce();
}

/* ---------------------------------------------------------------------- */

std::string ComputeSymmetrixTiming::get_thermo_colname(int index)
{
  if (index < 0) return "SxH1CommPct";
  if (index < NUM_OUTPUTS) return column_names[index];
  return {};
}
