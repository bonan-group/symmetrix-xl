#!/usr/bin/env bash
# Run only inside tools/wheel_images/cuda-manylinux_2_28.Dockerfile.

set -euo pipefail

PYTHON_ABI="${PYTHON_ABI:?set PYTHON_ABI, for example cp312-cp312}"
CUDA_MAJOR="${CUDA_MAJOR:?set CUDA_MAJOR to 12 or 13}"
CUDA_TARGET="${CUDA_TARGET:-sm120}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to an absolute mounted path}"
PYTHON_ROOT="/opt/python/${PYTHON_ABI}"
CUDA_HOST_TOOLSET=/opt/rh/gcc-toolset-13/root/usr/bin

case "$CUDA_MAJOR" in
  12)
    cuda_runtime_soname=libcudart.so.12
    cuda_blas_soname=libcublas.so.12
    cuda_blas_lt_soname=libcublasLt.so.12
    cuda_nvjitlink_soname=libnvJitLink.so.12
    ;;
  13)
    cuda_runtime_soname=libcudart.so.13
    cuda_blas_soname=libcublas.so.13
    cuda_blas_lt_soname=libcublasLt.so.13
    cuda_nvjitlink_soname=libnvJitLink.so.13
    ;;
  *)
    echo "error: CUDA_MAJOR must be 12 or 13, not $CUDA_MAJOR" >&2
    exit 1
    ;;
esac

if [ ! -x "${PYTHON_ROOT}/bin/python" ]; then
  echo "error: ${PYTHON_ROOT}/bin/python does not exist" >&2
  exit 1
fi
if [ ! -x "${CUDA_HOST_TOOLSET}/g++" ]; then
  echo "error: GCC 13 CUDA host compiler is not installed" >&2
  exit 1
fi
if [[ "$OUTPUT_DIR" != /io/* ]]; then
  echo "error: OUTPUT_DIR must be below the mounted /io checkout" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
if find "$OUTPUT_DIR" -maxdepth 1 -name '*.whl' -print -quit | grep -q .; then
  echo "error: OUTPUT_DIR already contains wheels: $OUTPUT_DIR" >&2
  exit 1
fi

export PATH="${PYTHON_ROOT}/bin:/usr/local/cuda/bin:${CUDA_HOST_TOOLSET}:/usr/local/bin:${PATH}"
export CMAKE_PREFIX_PATH=/opt/symmetrix-openblas
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export NVCC_WRAPPER_DEFAULT_COMPILER="${CUDA_HOST_TOOLSET}/g++"
export PIP_DISABLE_PIP_VERSION_CHECK=1

work_root="$(mktemp -d "/tmp/symmetrix-cuda${CUDA_MAJOR}-wheel.XXXXXX")"
trap 'rm -rf "$work_root"' EXIT
"${PYTHON_ROOT}/bin/python" -m venv "$work_root/venv"
build_python="$work_root/venv/bin/python"
raw_dir="$work_root/raw"
repaired_dir="$work_root/repaired"
mkdir -p "$raw_dir" "$repaired_dir"

"$build_python" -m pip install auditwheel twine
"$build_python" tools/symmetrix_build.py wheel-matrix \
  --target "cuda:${CUDA_MAJOR}:${CUDA_TARGET}" \
  --cuda-root /usr/local/cuda \
  --accelerator-blas-root /opt/symmetrix-openblas \
  --build-root /tmp/build \
  --wheel-dir "$raw_dir" \
  --jobs "${JOBS:-4}"

raw_wheels=("$raw_dir"/*.whl)
if [ "${#raw_wheels[@]}" -ne 1 ] || [ ! -f "${raw_wheels[0]}" ]; then
  echo "error: expected exactly one raw wheel" >&2
  exit 1
fi

CUDA_RUNTIME_SONAME="$cuda_runtime_soname" \
CUDA_BLAS_SONAME="$cuda_blas_soname" \
"$build_python" - "${raw_wheels[0]}" "$CUDA_MAJOR" <<'PY'
import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile

wheel = pathlib.Path(sys.argv[1])
cuda_major = sys.argv[2]
with tempfile.TemporaryDirectory() as directory:
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(directory)
    modules = list(pathlib.Path(directory).glob(f"**/_native_cuda{cuda_major}_*.so"))
    assert len(modules) == 1, modules
    module = modules[0]
    dynamic = subprocess.run(
        ["readelf", "-d", module], check=True, capture_output=True, text=True
    ).stdout
    assert os.environ["CUDA_RUNTIME_SONAME"] in dynamic, dynamic
    assert os.environ["CUDA_BLAS_SONAME"] in dynamic, dynamic
    assert "libopenblas" not in dynamic, dynamic
    assert "libgomp" not in dynamic, dynamic
    assert "libcusparse" not in dynamic, dynamic
    assert "libcusolver" not in dynamic, dynamic
    if cuda_major == "13":
        expected_rpaths = ("$ORIGIN/../nvidia/cu13/lib",)
    else:
        expected_rpaths = (
            "$ORIGIN/../nvidia/cuda_runtime/lib",
            "$ORIGIN/../nvidia/cublas/lib",
        )
    assert all(path in dynamic for path in expected_rpaths), dynamic
    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", module],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert " cblas_" not in symbols, "static OpenBLAS symbols are exported"
print("raw CUDA ELF gates passed:", wheel.name)
PY

"$build_python" -m auditwheel show "${raw_wheels[0]}"
"$build_python" -m auditwheel repair "${raw_wheels[0]}" \
  --wheel-dir "$repaired_dir" \
  --only-plat \
  --plat manylinux_2_28_x86_64 \
  --exclude libcuda.so.1 \
  --exclude "$cuda_runtime_soname" \
  --exclude "$cuda_blas_soname" \
  --exclude "$cuda_blas_lt_soname" \
  --exclude "$cuda_nvjitlink_soname"

repaired_wheels=("$repaired_dir"/*.whl)
if [ "${#repaired_wheels[@]}" -ne 1 ] || [ ! -f "${repaired_wheels[0]}" ]; then
  echo "error: expected exactly one repaired wheel" >&2
  exit 1
fi
case "${repaired_wheels[0]}" in
  *-manylinux_2_28_x86_64.whl) ;;
  *) echo "error: repaired wheel has the wrong platform tag" >&2; exit 1 ;;
esac

"$build_python" -m twine check --strict "${repaired_wheels[0]}"
"$build_python" tools/check_wheel_contents.py "${repaired_wheels[0]}"
"$build_python" -m auditwheel show "${repaired_wheels[0]}"
cp "${repaired_wheels[0]}" "$OUTPUT_DIR/"
(
  cd "$OUTPUT_DIR"
  sha256sum ./*.whl > SHA256SUMS
)
cat "$OUTPUT_DIR/SHA256SUMS"
