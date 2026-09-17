#!/usr/bin/env bash

# Adapted from https://github.com/mir-group/flare/blob/master/lammps_plugins/install.sh

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 path/to/lammps" >&2
    exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if ! lammps_dir=$(cd -- "$1" 2>/dev/null && pwd -P); then
    echo "Error: LAMMPS directory does not exist: $1" >&2
    exit 1
fi

for required_path in src src/KOKKOS cmake/CMakeLists.txt src/version.h; do
    if [[ ! -e "$lammps_dir/$required_path" ]]; then
        echo "Error: '$lammps_dir' is not a supported LAMMPS source tree; missing $required_path" >&2
        exit 1
    fi
done

lammps_version=$(sed -n 's/^#define[[:space:]]\+LAMMPS_VERSION[[:space:]]\+"\([^"]*\)".*/\1/p' \
    "$lammps_dir/src/version.h" | head -n 1)
if [[ ! $lammps_version =~ ^([0-9]{1,2})[[:space:]]+([A-Za-z]{3})[[:space:]]+([0-9]{4}) ]]; then
    echo "Error: could not parse LAMMPS_VERSION from $lammps_dir/src/version.h" >&2
    exit 1
fi

day=${BASH_REMATCH[1]}
month_name=${BASH_REMATCH[2]}
year=${BASH_REMATCH[3]}
case ${month_name,,} in
    jan) month=1 ;; feb) month=2 ;; mar) month=3 ;; apr) month=4 ;;
    may) month=5 ;; jun) month=6 ;; jul) month=7 ;; aug) month=8 ;;
    sep) month=9 ;; oct) month=10 ;; nov) month=11 ;; dec) month=12 ;;
    *) echo "Error: unknown month in LAMMPS_VERSION: $lammps_version" >&2; exit 1 ;;
esac
version_number=$((10#$year * 10000 + month * 100 + 10#$day))
if (( version_number < 20251210 )); then
    echo "Error: pair_symmetrix requires LAMMPS 10 Dec 2025 or newer; found $lammps_version" >&2
    exit 1
fi

case $script_dir in
    *'"'*|*'$'*|*';'*|*'\'*|*$'\n'*)
        echo "Error: the Symmetrix source path contains characters that CMake cannot quote safely" >&2
        exit 1
        ;;
esac

ln -sfn -- "$script_dir/pair_symmetrix_mace.h" \
    "$lammps_dir/src/pair_symmetrix_mace.h"
ln -sfn -- "$script_dir/pair_symmetrix_mace.cpp" \
    "$lammps_dir/src/pair_symmetrix_mace.cpp"
ln -sfn -- "$script_dir/compute_symmetrix_timing.h" \
    "$lammps_dir/src/compute_symmetrix_timing.h"
ln -sfn -- "$script_dir/compute_symmetrix_timing.cpp" \
    "$lammps_dir/src/compute_symmetrix_timing.cpp"
ln -sfn -- "$script_dir/pair_symmetrix_mace_kokkos.h" \
    "$lammps_dir/src/KOKKOS/pair_symmetrix_mace_kokkos.h"
ln -sfn -- "$script_dir/pair_symmetrix_mace_kokkos.cpp" \
    "$lammps_dir/src/KOKKOS/pair_symmetrix_mace_kokkos.cpp"

cmake_file="$lammps_dir/cmake/CMakeLists.txt"
temporary_file=$(mktemp "$lammps_dir/cmake/CMakeLists.symmetrix.XXXXXX")
trap 'rm -f -- "$temporary_file"' EXIT

# Remove a block written by this installer, including the unmarked legacy form.
awk '
    /^# BEGIN SYMMETRIX PAIR STYLE$/ { marked = 1; next }
    /^# END SYMMETRIX PAIR STYLE$/ { marked = 0; next }
    marked { next }
    /^function\(symmetrix_add_lammps_library\)$/ { legacy = 1; next }
    legacy && /^target_link_libraries\(lammps PRIVATE symmetrix\)$/ {
        legacy = 0
        next
    }
    legacy { next }
    { print }
' "$cmake_file" > "$temporary_file"

cat >> "$temporary_file" <<EOF

# BEGIN SYMMETRIX PAIR STYLE
function(symmetrix_add_lammps_library)
  # LAMMPS builds bundled Kokkos statically even when liblammps is shared.
  # Keep Symmetrix and KokkosKernels static so only one Kokkos runtime is linked.
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory("$script_dir/../libsymmetrix" "\${CMAKE_BINARY_DIR}/libsymmetrix")
endfunction()
symmetrix_add_lammps_library()
target_include_directories(lammps BEFORE PRIVATE "$script_dir/../libsymmetrix/external/json/single_include")
target_include_directories(lammps PRIVATE "$script_dir/../libsymmetrix/source")
target_link_libraries(lammps PRIVATE symmetrix)
# END SYMMETRIX PAIR STYLE
EOF

chmod --reference="$cmake_file" "$temporary_file"
mv -- "$temporary_file" "$cmake_file"
trap - EXIT

echo "Installed pair_symmetrix into $lammps_dir"
