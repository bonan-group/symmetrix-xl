#!/usr/bin/env bash
# Build publishable CUDA wheels in a pinned manylinux 2.28 container.

set -euo pipefail

CONTAINER_ENGINE="${CONTAINER_ENGINE:-docker}"
CUDA_MAJOR="${CUDA_MAJOR:-13}"
PYTHON_ABIS="${PYTHON_ABIS:-cp312-cp312}"
CUDA_TARGET="${CUDA_TARGET:-sm120}"

case "$CUDA_MAJOR" in
  12)
    default_cuda_image="nvidia/cuda:12.9.1-cudnn-devel-rockylinux8@sha256:fd69e8383a7c3101fe97190fc1c6c14b27c07aeab53eac9d1d794b03a89c739c"
    default_cuda_toolkit_path=/usr/local/cuda-12.9
    ;;
  13)
    default_cuda_image="nvidia/cuda:13.3.1-cudnn-devel-rockylinux8@sha256:b0c3200d0bf08f931bc51ba01953eec596d8cec5bbf3748236b23facaec489ce"
    default_cuda_toolkit_path=/usr/local/cuda-13.3
    ;;
  *)
    echo "error: CUDA_MAJOR must be 12 or 13, not $CUDA_MAJOR" >&2
    exit 1
    ;;
esac

CUDA_IMAGE="${CUDA_IMAGE:-$default_cuda_image}"
CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-$default_cuda_toolkit_path}"
BUILDER_IMAGE="${BUILDER_IMAGE:-symmetrix-cuda${CUDA_MAJOR}-manylinux_2_28}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
WHEEL_DIR="${WHEEL_DIR:-wheelhouse/cuda${CUDA_MAJOR}}"

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
    --build-arg "CUDA_IMAGE=$CUDA_IMAGE" \
    --build-arg "CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_PATH" \
    --file "$repo_root/tools/wheel_images/cuda-manylinux_2_28.Dockerfile" \
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
    --env "CUDA_MAJOR=$CUDA_MAJOR" \
    --env "CUDA_TARGET=$CUDA_TARGET" \
    --env "OUTPUT_DIR=/io/$WHEEL_DIR/$python_abi" \
    --env "JOBS=${JOBS:-4}" \
    "$BUILDER_IMAGE" \
    /io/tools/build_cuda_manylinux_in_container.sh
done
