#!/usr/bin/env bash
# Run only inside tools/wheel_images/cpu-manylinux_2_28.Dockerfile.

set -euo pipefail

PYTHON_ABI="${PYTHON_ABI:?set PYTHON_ABI, for example cp312-cp312}"
CPU_TARGET="${CPU_TARGET:-x86-64-v3}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to an absolute mounted path}"
PYTHON_ROOT="/opt/python/${PYTHON_ABI}"

if [ ! -x "${PYTHON_ROOT}/bin/python" ]; then
  echo "error: ${PYTHON_ROOT}/bin/python does not exist" >&2
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

export PATH="${PYTHON_ROOT}/bin:/usr/local/bin:${PATH}"
export CMAKE_PREFIX_PATH=/opt/symmetrix-openblas
export LIBRARY_PATH=/opt/symmetrix-openblas/lib
export LD_LIBRARY_PATH=/opt/symmetrix-openblas/lib
export PIP_DISABLE_PIP_VERSION_CHECK=1

work_root="$(mktemp -d /tmp/symmetrix-cpu-wheel.XXXXXX)"
trap 'rm -rf "$work_root"' EXIT
"${PYTHON_ROOT}/bin/python" -m venv "$work_root/venv"
build_python="$work_root/venv/bin/python"
raw_dir="$work_root/raw"
repaired_dir="$work_root/repaired"
mkdir -p "$raw_dir" "$repaired_dir"

"$build_python" -m pip install auditwheel twine
"$build_python" tools/symmetrix_build.py wheel \
  --backend cpu \
  --cpu-target "$CPU_TARGET" \
  --build-root /tmp/build \
  --wheel-dir "$raw_dir" \
  --jobs "${JOBS:-4}"

raw_wheels=("$raw_dir"/*.whl)
if [ "${#raw_wheels[@]}" -ne 1 ] || [ ! -f "${raw_wheels[0]}" ]; then
  echo "error: expected exactly one raw wheel" >&2
  exit 1
fi

"$build_python" -m auditwheel show "${raw_wheels[0]}"
"$build_python" -m auditwheel repair "${raw_wheels[0]}" \
  --wheel-dir "$repaired_dir" \
  --only-plat \
  --plat manylinux_2_28_x86_64

repaired_wheels=("$repaired_dir"/*.whl)
if [ "${#repaired_wheels[@]}" -ne 1 ] || [ ! -f "${repaired_wheels[0]}" ]; then
  echo "error: expected exactly one repaired wheel" >&2
  exit 1
fi
case "${repaired_wheels[0]}" in
  *-manylinux_2_28_x86_64.whl) ;;
  *) echo "error: repaired wheel has the wrong platform tag" >&2; exit 1 ;;
esac

"$build_python" - "${repaired_wheels[0]}" "$CPU_TARGET" <<'PY'
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile

wheel = pathlib.Path(sys.argv[1])
cpu_target = sys.argv[2]
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        modules = [name for name in names if pathlib.PurePosixPath(name).name.startswith("_native_cpu") and name.endswith(".so")]
        assert len(modules) == 1, modules
        descriptors = [name for name in names if name == "symmetrix/_backend_cpu.json"]
        assert len(descriptors) == 1, descriptors
        descriptor = json.loads(archive.read(descriptors[0]))
        assert descriptor["architecture"] == cpu_target, descriptor
        archive.extract(modules[0], root)
    module = root / modules[0]
    disassembly = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", module],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert not re.search(r"\bzmm\d+\b", disassembly), "AVX-512 instruction found"
print("CPU wheel layout and x86-64-v3 gates passed:", wheel.name)
PY

"$build_python" -m twine check --strict "${repaired_wheels[0]}"
"$build_python" tools/check_wheel_contents.py "${repaired_wheels[0]}"
"$build_python" -m auditwheel show "${repaired_wheels[0]}"
cp "${repaired_wheels[0]}" "$OUTPUT_DIR/"
(
  cd "$OUTPUT_DIR"
  sha256sum ./*.whl > SHA256SUMS
)
cat "$OUTPUT_DIR/SHA256SUMS"
