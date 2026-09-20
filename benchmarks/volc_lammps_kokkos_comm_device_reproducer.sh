#!/usr/bin/env bash
set -euo pipefail

RANKS=${1:?usage: $0 RANKS TRIAL}
TRIAL=${2:-1}
case "${RANKS}" in
  2) PX=1; PY=1; PZ=2 ;;
  4) PX=1; PY=2; PZ=2 ;;
  8) PX=2; PY=2; PZ=2 ;;
  *) echo "RANKS must be 2, 4, or 8" >&2; exit 64 ;;
esac

TASK_ROOT=${TASK_ROOT:-/vepfs/symmetrix-bench/lammps-kokkos-comm-device-r8-debug-20260920}
HPCX_ROOT=${HPCX_ROOT:-/vepfs/symmetrix-bench/a100-srtio3-lammps-release72f1f33-20260917/nvhpc-25.1/extract/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/hpcx/hpcx-2.20}
LMP=${LMP:-/vepfs/symmetrix-bench/a100-srtio3-lammps-ac73441-lammps-c8bd2ae5-hpcx-20260920/install/bin/lmp}
INPUT=${INPUT:-${TASK_ROOT}/inputs/lammps_kokkos_comm_device_migration.in}
COMM_MODE=${KOKKOS_COMM_MODE:-device}
COMM_EXCHANGE=${KOKKOS_COMM_EXCHANGE:-device}
EPSILON=${LJ_EPSILON:-0.1}
TIMESTEP=${LAMMPS_TIMESTEP:-0.001}
COMM_CUTOFF=${LAMMPS_COMM_CUTOFF:-6.5}
RUN_DIR=${TASK_ROOT}/standard-lj/r${RANKS}-t${TRIAL}
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/runner.stdout") 2> >(tee "${RUN_DIR}/runner.stderr" >&2)

source "${HPCX_ROOT}/hpcx-init.sh"
hpcx_load
export PATH=/usr/local/cuda-13.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_opal_cuda_support=true
export UCX_MEMTYPE_CACHE=n
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-warn}
export UCX_LOG_FILE=${RUN_DIR}/ucx-%h-%p.log
unset UCX_TLS
unset UCX_NET_DEVICES

MPI_EXTRA_ENV=()
if [[ -n ${SYMMETRIX_LAMMPS_VALIDATE_MPI_COMM:-} ]]; then
  MPI_EXTRA_ENV+=(-x SYMMETRIX_LAMMPS_VALIDATE_MPI_COMM)
fi
if [[ -n ${SYMMETRIX_LAMMPS_FENCE_MPI_COMM:-} ]]; then
  MPI_EXTRA_ENV+=(-x SYMMETRIX_LAMMPS_FENCE_MPI_COMM)
fi

test -x "${LMP}"
test -f "${INPUT}"
{
  echo "hostname=$(hostname)"
  echo "date=$(date --iso-8601=seconds)"
  echo "ranks=${RANKS}"
  echo "rank_grid=${PX}x${PY}x${PZ}"
  echo "expected_atoms=1008000"
  echo "lammps_executable=${LMP}"
  echo "kokkos_gpu_aware=on"
  echo "kokkos_comm_mode=${COMM_MODE}"
  echo "kokkos_comm_exchange=${COMM_EXCHANGE}"
  echo "lj_epsilon=${EPSILON}"
  echo "timestep=${TIMESTEP}"
  echo "comm_cutoff=${COMM_CUTOFF}"
  echo "ucx_tls=automatic"
  echo "ucx_net_devices=automatic"
  echo "ucx_memtype_cache=${UCX_MEMTYPE_CACHE}"
  echo "hpcx_root=${HPCX_ROOT}"
} > "${RUN_DIR}/configuration.txt"
sha256sum "${LMP}" "${INPUT}" > "${RUN_DIR}/runtime-sha256.txt"
ldd "${LMP}" > "${RUN_DIR}/lammps-ldd.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version,compute_cap \
  --format=csv,noheader > "${RUN_DIR}/gpu-inventory.csv"
nvidia-smi topo -m > "${RUN_DIR}/gpu-topology.txt"
"${HPCX_ROOT}/ompi/bin/ompi_info" --parsable --all > "${RUN_DIR}/ompi-info.txt"
grep -q 'mpi_built_with_cuda_support:value:true' "${RUN_DIR}/ompi-info.txt"
grep -q 'opal_built_with_cuda_support:value:true' "${RUN_DIR}/ompi-info.txt"

MPI_RUN=(
  "${HPCX_ROOT}/ompi/bin/mpirun" --allow-run-as-root -np "${RANKS}"
  --bind-to core --map-by "ppr:${RANKS}:node"
  --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0 --mca pml ucx
  -x PATH -x LD_LIBRARY_PATH -x OPAL_PREFIX
  -x UCX_MEMTYPE_CACHE -x UCX_LOG_LEVEL -x UCX_LOG_FILE
  -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS
  -x BLIS_NUM_THREADS -x NUMEXPR_NUM_THREADS
  -x OMPI_MCA_opal_cuda_support -x OMPI_MCA_coll_hcoll_enable
  -x OMPI_MCA_coll_ucc_enable
  "${MPI_EXTRA_ENV[@]}"
)
"${MPI_RUN[@]}" bash -lc 'printf "rank=%s local_rank=%s host=%s visible_gpus=%s\n" \
  "${OMPI_COMM_WORLD_RANK}" "${OMPI_COMM_WORLD_LOCAL_RANK}" "$(hostname)" \
  "$(nvidia-smi -L | wc -l)"' | sort > "${RUN_DIR}/rank-map.txt"

MONITOR_STOP=${RUN_DIR}/monitor.stop
(
  while ! test -e "${MONITOR_STOP}"; do
    printf '%s,' "$(date +%s.%N)"
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | tr '\n' ';'
    printf '\n'
    sleep 0.2
  done
) > "${RUN_DIR}/gpu-memory.csv" &
MONITOR_PID=$!
cleanup() {
  touch "${MONITOR_STOP}"
  wait "${MONITOR_PID}" || true
}
trap cleanup EXIT

set +e
"${MPI_RUN[@]}" "${LMP}" -k on g "${RANKS}" t 1 -sf kk \
  -pk kokkos gpu/aware on comm "${COMM_MODE}" comm/exchange "${COMM_EXCHANGE}" \
  -var nx 56 -var ny 60 -var nz 60 \
  -var px "${PX}" -var py "${PY}" -var pz "${PZ}" \
  -var epsilon "${EPSILON}" -var timestep "${TIMESTEP}" \
  -var comm_cutoff "${COMM_CUTOFF}" \
  -log "${RUN_DIR}/lammps.log" -in "${INPUT}" \
  2>&1 | tee "${RUN_DIR}/launcher.log"
RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "${RC}" > "${RUN_DIR}/exit-status.txt"
cleanup
trap - EXIT
test "${RC}" -eq 0

grep -q 'KOKKOS_COMM_WARMUP_COMPLETE step=5 atoms=1008000' "${RUN_DIR}/lammps.log"
grep -q 'KOKKOS_COMM_MEASUREMENT_COMPLETE step=25 atoms=1008000' "${RUN_DIR}/lammps.log"
grep -Eq 'Neighbor list builds = [1-9][0-9]*' "${RUN_DIR}/lammps.log"
grep -q 'Dangerous builds = 0' "${RUN_DIR}/lammps.log"
awk -v owned="$((1008000 / RANKS))" '
  $1 == "Nlocal:" && ($4 != owned || $6 != owned) { migrated = 1 }
  END { exit !migrated }
' "${RUN_DIR}/lammps.log"
! grep -Eq 'Lost atoms|CUDA error|Atom outside of neighbor bin range|Turning off GPU-aware MPI' \
  "${RUN_DIR}/lammps.log" "${RUN_DIR}/launcher.log"
echo "KOKKOS_COMM_DEVICE_REPRODUCER_COMPLETE=${RUN_DIR}"
