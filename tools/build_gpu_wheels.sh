#!/usr/bin/env bash
# Build the Symmetrix GPU backend wheels locally with the release wheel matrix.
#
# Usage:
#   tools/build_gpu_wheels.sh                 # all toolkits found, default archs
#   CUDA12_ROOT=/opt/cuda-12.9 tools/build_gpu_wheels.sh
#   CUDA_ARCHS="sm90 sm120" JOBS=16 tools/build_gpu_wheels.sh
#   ROCM_ROOT=/opt/rocm tools/build_gpu_wheels.sh
#
# Environment overrides:
#   PYTHON       interpreter with the build tool prerequisites (default: python3)
#   WHEEL_DIR    output root (default: dist)
#   JOBS         parallel compiler processes per target
#   CUDA13_ROOT  CUDA 13 toolkit root (default: /usr/local/cuda-13.3)
#   CUDA12_ROOT  CUDA 12.8+ toolkit root (default: /usr/local/cuda-12.9)
#   FORCE_CUDA12 build cuda12 targets even if CUDA12_ROOT has no nvcc
#   CUDA_ARCHS   override both CUDA generations with the same target list
#   CUDA12_ARCHS default "sm70 sm80 sm86 sm89 sm90 sm100 sm120"
#   CUDA13_ARCHS default "sm80 sm86 sm89 sm90 sm100 sm120"
#   ROCM_ROOT    ROCm root; set to enable HIP targets
#   ROCM_MAJOR   ROCm major version (default: detected from hipcc)
#   HIP_ARCHS    default "gfx1151"
#
# Each toolkit generation builds into its own subdirectory so the per-run
# release-index.json is preserved. These local linux_x86_64 wheels are smoke-test
# artifacts, not release artifacts. Build publishable CUDA 13 wheels with
# tools/build_cuda_manylinux_wheels.sh.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
WHEEL_DIR="${WHEEL_DIR:-dist}"
JOBS="${JOBS:-}"
CUDA13_ROOT="${CUDA13_ROOT:-/usr/local/cuda-13.3}"
CUDA12_ROOT="${CUDA12_ROOT:-/usr/local/cuda-12.9}"
FORCE_CUDA12="${FORCE_CUDA12:-0}"
CUDA_ARCHS="${CUDA_ARCHS:-}"
CUDA12_ARCHS="${CUDA12_ARCHS:-${CUDA_ARCHS:-sm70 sm80 sm86 sm89 sm90 sm100 sm120}}"
CUDA13_ARCHS="${CUDA13_ARCHS:-${CUDA_ARCHS:-sm80 sm86 sm89 sm90 sm100 sm120}}"
ROCM_ROOT="${ROCM_ROOT:-}"
ROCM_MAJOR="${ROCM_MAJOR:-}"
HIP_ARCHS="${HIP_ARCHS:-gfx1151}"
ACCELERATOR_BLAS_ROOT="${ACCELERATOR_BLAS_ROOT:-}"

cd "$(dirname "$0")/.."

extra=()
if [ -n "$JOBS" ]; then
  extra+=(--jobs "$JOBS")
fi
if [ -n "$ACCELERATOR_BLAS_ROOT" ]; then
  extra+=(--accelerator-blas-root "$ACCELERATOR_BLAS_ROOT")
fi

targets_for() {
  local generation="$1"
  shift
  local archs=("$@")
  local targets=()
  for arch in "${archs[@]}"; do
    targets+=(--target "cuda:${generation}:${arch}")
  done
  printf '%s\n' "${targets[*]}"
}

build_cuda() {
  local generation="$1"
  local root="$2"
  local archs
  if [ ! -x "${root}/bin/nvcc" ]; then
    echo "skip cuda${generation}: no nvcc at ${root}/bin/nvcc" >&2
    return 0
  fi
  if [ "$generation" = 12 ]; then
    archs="$CUDA12_ARCHS"
  else
    archs="$CUDA13_ARCHS"
  fi
  local targets
  # shellcheck disable=SC2086
  targets=$(targets_for "$generation" $archs)
  echo "==> cuda${generation} (${root}): ${archs}"
  # shellcheck disable=SC2086
  "$PYTHON" tools/symmetrix_build.py wheel-matrix \
    --cuda-root "$root" \
    --wheel-dir "${WHEEL_DIR}/cuda${generation}" \
    ${targets} "${extra[@]+"${extra[@]}"}"
}

build_cuda 13 "$CUDA13_ROOT"
if [ "$FORCE_CUDA12" = "1" ] || [ -x "${CUDA12_ROOT}/bin/nvcc" ]; then
  build_cuda 12 "$CUDA12_ROOT"
else
  echo "skip cuda12: set CUDA12_ROOT (toolkit 12.8+) or FORCE_CUDA12=1" >&2
fi

if [ -n "$ROCM_ROOT" ]; then
  if [ ! -x "${ROCM_ROOT}/bin/hipcc" ]; then
    echo "error: ROCM_ROOT=${ROCM_ROOT} has no bin/hipcc" >&2
    exit 1
  fi
  if [ -z "$ROCM_MAJOR" ]; then
    ROCM_MAJOR="$(
      "${ROCM_ROOT}/bin/hipcc" --version |
        sed -n 's/^HIP version: *\([0-9][0-9]*\)\..*/\1/p' | head -n 1
    )"
  fi
  if [ -z "$ROCM_MAJOR" ]; then
    echo "error: cannot detect the ROCm major version from ${ROCM_ROOT}/bin/hipcc; set ROCM_MAJOR" >&2
    exit 1
  fi
  hip_targets=()
  for arch in $HIP_ARCHS; do
    hip_targets+=(--target "hip:${ROCM_MAJOR}:${arch}")
  done
  echo "==> hip (${ROCM_ROOT}): ${HIP_ARCHS}"
  "$PYTHON" tools/symmetrix_build.py wheel-matrix \
    --rocm-root "$ROCM_ROOT" \
    --wheel-dir "${WHEEL_DIR}/rocm" \
    "${hip_targets[@]}" "${extra[@]+"${extra[@]}"}"
fi

echo
echo "Built wheels:"
find "$WHEEL_DIR" -name '*.whl' -print 2>/dev/null | sort || true
for index in "$WHEEL_DIR"/*/release-index.json; do
  [ -e "$index" ] || continue
  echo "release index: $index"
done
echo "Local wheels are not publishable. Use tools/build_cuda_manylinux_wheels.sh."
