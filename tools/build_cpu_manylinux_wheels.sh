#!/usr/bin/env bash
# Build publishable CPU wheels in the pinned manylinux 2.28 container.

set -euo pipefail

CONTAINER_ENGINE="${CONTAINER_ENGINE:-docker}"
BUILDER_IMAGE="${BUILDER_IMAGE:-symmetrix-cpu-manylinux_2_28}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
PYTHON_ABIS="${PYTHON_ABIS:-cp312-cp312}"
CPU_TARGET="${CPU_TARGET:-x86-64-v3}"
WHEEL_DIR="${WHEEL_DIR:-wheelhouse/cpu}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$WHEEL_DIR" = /* ]] || [[ "$WHEEL_DIR" == *".."* ]]; then
  echo "error: WHEEL_DIR must be a relative path without '..': $WHEEL_DIR" >&2
  exit 1
fi
mkdir -p "$repo_root/$WHEEL_DIR"

container_workdir=/io
git_mount=()
git_common_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)"
if [[ "$git_common_dir" != "$repo_root/"* ]]; then
  container_workdir=/usr/src/symmetrix-xl
  container_repo_alias="/usr/src/$(basename "$repo_root")"
  container_git_dir="/usr/src/$(basename "$(dirname "$git_common_dir")")/.git"
  git_mount=(
    --volume "$repo_root:$container_workdir"
    --volume "$git_common_dir:$git_common_dir:ro"
    --volume "$git_common_dir:$container_git_dir:ro"
  )
  if [[ "$container_repo_alias" != "$container_workdir" ]]; then
    git_mount+=(--volume "$repo_root:$container_repo_alias")
  fi
fi

if [[ "$BUILD_IMAGE" == 1 ]]; then
  "$CONTAINER_ENGINE" build --pull \
    --file "$repo_root/tools/wheel_images/cpu-manylinux_2_28.Dockerfile" \
    --tag "$BUILDER_IMAGE" \
    "$repo_root/tools/wheel_images"
elif [[ "$BUILD_IMAGE" != 0 ]]; then
  echo "error: BUILD_IMAGE must be 0 or 1, not $BUILD_IMAGE" >&2
  exit 1
fi

for python_abi in $PYTHON_ABIS; do
  output="$repo_root/$WHEEL_DIR/$python_abi"
  mkdir -p "$output"
  "$CONTAINER_ENGINE" run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "$repo_root:/io" \
    "${git_mount[@]}" \
    --workdir "$container_workdir" \
    --env "HOME=/tmp" \
    --env "PYTHON_ABI=$python_abi" \
    --env "CPU_TARGET=$CPU_TARGET" \
    --env "OUTPUT_DIR=/io/$WHEEL_DIR/$python_abi" \
    --env "JOBS=${JOBS:-4}" \
    "$BUILDER_IMAGE" \
    /io/tools/build_cpu_manylinux_in_container.sh
done
