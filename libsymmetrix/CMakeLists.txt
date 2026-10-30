cmake_minimum_required(VERSION 3.27)
project(LibSymmetrix)

message(STATUS "Symmetrix: Beginning to process CMakeLists.txt.")

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SYMMETRIX_HOST_ARCH "native" CACHE STRING
    "Host -march target; use x86-64-v3 or x86-64-v4 for a portable x86-64 build or none to disable.")
option(SYMMETRIX_FAST_MATH
    "Enable the historical -ffast-math policy for host and CUDA C++ code." ON)
set(_symmetrix_host_arch_flag "")
if(NOT SYMMETRIX_HOST_ARCH STREQUAL "" AND NOT SYMMETRIX_HOST_ARCH STREQUAL "none")
    set(_symmetrix_host_arch_flag "-march=${SYMMETRIX_HOST_ARCH}")
endif()
if(NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP)
    if(NOT DEFINED Kokkos_ARCH_NATIVE)
        if(SYMMETRIX_HOST_ARCH STREQUAL "native")
            set(Kokkos_ARCH_NATIVE ON CACHE BOOL "Set Kokkos_ARCH_NATIVE.")
        else()
            set(Kokkos_ARCH_NATIVE OFF CACHE BOOL "Set Kokkos_ARCH_NATIVE.")
        endif()
    endif()
    if(NOT SYMMETRIX_HOST_ARCH STREQUAL "native" AND Kokkos_ARCH_NATIVE)
        message(FATAL_ERROR
            "SYMMETRIX_HOST_ARCH=${SYMMETRIX_HOST_ARCH} requires "
            "Kokkos_ARCH_NATIVE=OFF for a consistent CPU target.")
    endif()
    if(_symmetrix_host_arch_flag
            AND NOT SYMMETRIX_HOST_ARCH STREQUAL "native"
            AND NOT _symmetrix_host_arch_applied_globally)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:${_symmetrix_host_arch_flag}>)
        set(_symmetrix_host_arch_applied_globally TRUE)
    endif()
endif()

# ----- dependencies -----

set(SYMMETRIX_BLAS_LIBRARY "" CACHE FILEPATH
    "Optional exact BLAS library used by release builds.")
set(SYMMETRIX_BLAS_INCLUDE_DIR "" CACHE PATH
    "Optional CBLAS include directory paired with SYMMETRIX_BLAS_LIBRARY.")
option(SYMMETRIX_HIDE_STATIC_BLAS_SYMBOLS
    "Hide symbols from a statically linked BLAS archive." OFF)
if(SYMMETRIX_BLAS_LIBRARY)
    if(NOT IS_ABSOLUTE "${SYMMETRIX_BLAS_LIBRARY}"
            OR NOT EXISTS "${SYMMETRIX_BLAS_LIBRARY}")
        message(FATAL_ERROR
            "SYMMETRIX_BLAS_LIBRARY must name an existing absolute path: "
            "${SYMMETRIX_BLAS_LIBRARY}")
    endif()
    if(NOT SYMMETRIX_BLAS_INCLUDE_DIR
            OR NOT EXISTS "${SYMMETRIX_BLAS_INCLUDE_DIR}/cblas.h")
        message(FATAL_ERROR
            "SYMMETRIX_BLAS_INCLUDE_DIR must contain cblas.h when an exact "
            "BLAS library is selected: ${SYMMETRIX_BLAS_INCLUDE_DIR}")
    endif()
    set(BLAS_LIBRARIES "${SYMMETRIX_BLAS_LIBRARY}")
    message(STATUS "Symmetrix: exact BLAS library ${BLAS_LIBRARIES}")
else()
    find_package(BLAS REQUIRED)
endif()
# Keep the exact archive selected for this target even though KokkosKernels
# needs the public BLAS cache normalized to a library name and search path.
# Otherwise the final accelerator extension can degrade the absolute archive
# to `-lopenblas` and depend on an ambient compiler search directory.
set(_symmetrix_blas_link_libraries "${BLAS_LIBRARIES}")
# KokkosKernels caches BLAS_LIBRARIES as a user-facing string. On a later
# configure that cache entry can hide FindBLAS' normal variable, and its
# finder cannot resolve absolute library paths when they are treated as
# library names. Normalize every absolute result to a stable name while
# preserving companion libraries (for example `m` and `dl`) and all search
# directories. This keeps repeated package installs on the same build tree
# working for both single- and multi-entry FindBLAS results.
set(_symmetrix_blas_normalized_libraries "")
set(_symmetrix_blas_normalized_directories "${BLAS_LIBRARY_DIRS}")
set(_symmetrix_blas_can_normalize TRUE)
foreach(_symmetrix_blas_library IN LISTS BLAS_LIBRARIES)
    if(IS_ABSOLUTE "${_symmetrix_blas_library}")
        if(NOT EXISTS "${_symmetrix_blas_library}")
            set(_symmetrix_blas_can_normalize FALSE)
            break()
        endif()
        get_filename_component(
            _symmetrix_blas_library_dir "${_symmetrix_blas_library}" DIRECTORY)
        get_filename_component(
            _symmetrix_blas_library_name "${_symmetrix_blas_library}" NAME)
        string(REGEX MATCH
            "^lib(.+)[.](a|so|dylib)([.].*)?$"
            _symmetrix_blas_library_match "${_symmetrix_blas_library_name}")
        string(REGEX REPLACE
            "^lib(.+)[.](a|so|dylib)([.].*)?$" "\\1"
            _symmetrix_blas_library_name "${_symmetrix_blas_library_name}")
        if(NOT _symmetrix_blas_library_match)
            set(_symmetrix_blas_can_normalize FALSE)
            break()
        endif()
        list(APPEND _symmetrix_blas_normalized_directories
            "${_symmetrix_blas_library_dir}")
        list(APPEND _symmetrix_blas_normalized_libraries
            "${_symmetrix_blas_library_name}")
    elseif(_symmetrix_blas_library MATCHES "^-l(.+)$")
        # FindBLAS' MKL result carries link flags such as `-lm` and `-ldl`.
        # KokkosKernels resolves every entry only inside the recorded
        # directories, so a flag has to bring its own directory with it.
        set(_symmetrix_blas_flag_name "${CMAKE_MATCH_1}")
        unset(_symmetrix_blas_flag_library CACHE)
        unset(_symmetrix_blas_flag_library)
        find_library(_symmetrix_blas_flag_library
            NAMES "${_symmetrix_blas_flag_name}")
        if(NOT _symmetrix_blas_flag_library)
            set(_symmetrix_blas_can_normalize FALSE)
            break()
        endif()
        get_filename_component(_symmetrix_blas_flag_dir
            "${_symmetrix_blas_flag_library}" DIRECTORY)
        list(APPEND _symmetrix_blas_normalized_directories
            "${_symmetrix_blas_flag_dir}")
        list(APPEND _symmetrix_blas_normalized_libraries
            "${_symmetrix_blas_flag_name}")
    else()
        list(APPEND _symmetrix_blas_normalized_libraries
            "${_symmetrix_blas_library}")
    endif()
endforeach()
if(_symmetrix_blas_can_normalize AND _symmetrix_blas_normalized_libraries)
    list(REMOVE_DUPLICATES _symmetrix_blas_normalized_directories)
    set(BLAS_LIBRARIES "${_symmetrix_blas_normalized_libraries}")
    set(BLAS_LIBRARIES "${_symmetrix_blas_normalized_libraries}" CACHE STRING
        "Resolved BLAS library names for repeatable configuration." FORCE)
    if(_symmetrix_blas_normalized_directories)
        set(BLAS_LIBRARY_DIRS "${_symmetrix_blas_normalized_directories}")
        set(BLAS_LIBRARY_DIRS "${_symmetrix_blas_normalized_directories}" CACHE STRING
            "Resolved BLAS library directories for repeatable configuration." FORCE)
    endif()
endif()
unset(_symmetrix_blas_can_normalize)
unset(_symmetrix_blas_normalized_libraries)
unset(_symmetrix_blas_normalized_directories)
unset(_symmetrix_blas_library)
unset(_symmetrix_blas_library_dir)
unset(_symmetrix_blas_library_name)
unset(_symmetrix_blas_library_match)
unset(_symmetrix_blas_flag_name)
unset(_symmetrix_blas_flag_library CACHE)
unset(_symmetrix_blas_flag_library)
unset(_symmetrix_blas_flag_dir)
# MKL bundles LAPACK inside its BLAS libraries and ships no `mkl_lapack`, which
# is the only name CMake's FindLAPACK Intel path accepts, so that search always
# fails for a modern oneAPI installation. Seed KokkosKernels' LAPACK TPL from
# the libraries already resolved above instead of running it. The cache entries
# matter: KokkosKernels' TPL import clears a normal variable of the same name.
if(NOT LAPACK_LIBRARIES AND BLAS_LIBRARIES AND BLA_VENDOR MATCHES "Intel")
    set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}")
    set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING
        "LAPACK libraries resolved from the MKL BLAS result." FORCE)
    if(BLAS_LIBRARY_DIRS)
        set(LAPACK_LIBRARY_DIRS "${BLAS_LIBRARY_DIRS}")
        set(LAPACK_LIBRARY_DIRS "${BLAS_LIBRARY_DIRS}" CACHE STRING
            "LAPACK library directories resolved from the MKL BLAS result."
            FORCE)
    endif()
endif()
find_package(Git REQUIRED)

set(_symmetrix_sphericart_cuda_default OFF)
if(Kokkos_ENABLE_CUDA)
    set(_symmetrix_sphericart_cuda_default ON)
endif()
option(SYMMETRIX_SPHERICART_CUDA "Enable SpheriCart-CUDA."
    ${_symmetrix_sphericart_cuda_default})
unset(_symmetrix_sphericart_cuda_default)
option(SYMMETRIX_SPHERICART_SYCL "Enable SpheriCart-SYCL." OFF)
set(SPHERICART_ARCH_NATIVE OFF CACHE BOOL
    "Try to use -march=native when compiling SpheriCart." FORCE)
if(Kokkos_ENABLE_HIP)
    set(SYMMETRIX_SPHERICART_CUDA OFF CACHE BOOL "Enable SpheriCart-CUDA." FORCE)
    set(SPHERICART_ENABLE_CUDA OFF CACHE BOOL "Build SpheriCart CUDA backend." FORCE)
    set(SPHERICART_OPENMP OFF CACHE BOOL "Build SpheriCart with OpenMP." FORCE)
    set(SPHERICART_ARCH_NATIVE OFF CACHE BOOL
        "Try to use -march=native when compiling SpheriCart." FORCE)
endif()
if(Kokkos_ENABLE_CUDA AND NOT SYMMETRIX_SPHERICART_CUDA)
    message(WARNING
        "Kokkos CUDA is enabled while SYMMETRIX_SPHERICART_CUDA is OFF; "
        "MACE spherical harmonics will copy edge geometry to the host on "
        "every evaluation. Use this configuration only as a diagnostic.")
endif()

# Sphericart 1.0.3 synchronizes every CUDA launch and intentionally leaves its
# lazy device allocation alive. Apply the narrowly scoped lifecycle extension
# to a build-tree copy so the pinned upstream submodule remains reproducible
# and clean in source checkouts.
set(SYMMETRIX_SPHERICART_SOURCE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/external/sphericart-patched-source")
set(SYMMETRIX_SPHERICART_PATCH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/sphericart-1.0.3-async-lifecycle.patch")
set(SYMMETRIX_SPHERICART_BASE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/external/sphericart")
set(SYMMETRIX_SPHERICART_PATCH_STAMP
    "${SYMMETRIX_SPHERICART_SOURCE_DIR}/.symmetrix-patch-fingerprint")
file(GLOB_RECURSE SYMMETRIX_SPHERICART_INPUTS
    LIST_DIRECTORIES FALSE "${SYMMETRIX_SPHERICART_BASE_DIR}/*")
list(SORT SYMMETRIX_SPHERICART_INPUTS)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${SYMMETRIX_SPHERICART_INPUTS} "${SYMMETRIX_SPHERICART_PATCH}")
set(SYMMETRIX_SPHERICART_FINGERPRINT_INPUT "")
foreach(SYMMETRIX_SPHERICART_INPUT IN LISTS SYMMETRIX_SPHERICART_INPUTS)
    file(RELATIVE_PATH SYMMETRIX_SPHERICART_INPUT_RELATIVE
        "${SYMMETRIX_SPHERICART_BASE_DIR}" "${SYMMETRIX_SPHERICART_INPUT}")
    if(NOT SYMMETRIX_SPHERICART_INPUT_RELATIVE STREQUAL ".git"
            AND NOT IS_DIRECTORY "${SYMMETRIX_SPHERICART_INPUT}")
        file(SHA256 "${SYMMETRIX_SPHERICART_INPUT}"
            SYMMETRIX_SPHERICART_INPUT_HASH)
        string(APPEND SYMMETRIX_SPHERICART_FINGERPRINT_INPUT
            "${SYMMETRIX_SPHERICART_INPUT_RELATIVE}:${SYMMETRIX_SPHERICART_INPUT_HASH}\n")
    endif()
endforeach()
file(SHA256 "${SYMMETRIX_SPHERICART_PATCH}" SYMMETRIX_SPHERICART_PATCH_HASH)
string(APPEND SYMMETRIX_SPHERICART_FINGERPRINT_INPUT
    "patch:${SYMMETRIX_SPHERICART_PATCH_HASH}\n")
string(SHA256 SYMMETRIX_SPHERICART_EXPECTED_FINGERPRINT
    "${SYMMETRIX_SPHERICART_FINGERPRINT_INPUT}")

set(SYMMETRIX_SPHERICART_REFRESH TRUE)
if(EXISTS "${SYMMETRIX_SPHERICART_PATCH_STAMP}")
    file(READ "${SYMMETRIX_SPHERICART_PATCH_STAMP}"
        SYMMETRIX_SPHERICART_CURRENT_FINGERPRINT)
    if("${SYMMETRIX_SPHERICART_CURRENT_FINGERPRINT}" STREQUAL
            "${SYMMETRIX_SPHERICART_EXPECTED_FINGERPRINT}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env
                    "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/external"
                    "${GIT_EXECUTABLE}" apply --reverse --check
                    "${SYMMETRIX_SPHERICART_PATCH}"
            WORKING_DIRECTORY "${SYMMETRIX_SPHERICART_SOURCE_DIR}"
            RESULT_VARIABLE SYMMETRIX_SPHERICART_REUSE_CHECK
            ERROR_QUIET)
        if(SYMMETRIX_SPHERICART_REUSE_CHECK EQUAL 0)
            set(SYMMETRIX_SPHERICART_REFRESH FALSE)
        endif()
    endif()
endif()

if(SYMMETRIX_SPHERICART_REFRESH)
    file(REMOVE_RECURSE "${SYMMETRIX_SPHERICART_SOURCE_DIR}")
    file(MAKE_DIRECTORY "${SYMMETRIX_SPHERICART_SOURCE_DIR}")
    file(COPY "${SYMMETRIX_SPHERICART_BASE_DIR}/"
        DESTINATION "${SYMMETRIX_SPHERICART_SOURCE_DIR}"
        PATTERN ".git" EXCLUDE)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/external"
                "${GIT_EXECUTABLE}" apply --check "${SYMMETRIX_SPHERICART_PATCH}"
        WORKING_DIRECTORY "${SYMMETRIX_SPHERICART_SOURCE_DIR}"
        RESULT_VARIABLE SYMMETRIX_SPHERICART_PATCH_CHECK)
    if(NOT SYMMETRIX_SPHERICART_PATCH_CHECK EQUAL 0)
        message(FATAL_ERROR
            "Symmetrix's Sphericart lifecycle patch does not apply to the pinned source.")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/external"
                "${GIT_EXECUTABLE}" apply --whitespace=nowarn
                "${SYMMETRIX_SPHERICART_PATCH}"
        WORKING_DIRECTORY "${SYMMETRIX_SPHERICART_SOURCE_DIR}"
        RESULT_VARIABLE SYMMETRIX_SPHERICART_PATCH_RESULT)
    if(NOT SYMMETRIX_SPHERICART_PATCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to apply Symmetrix's Sphericart lifecycle patch.")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/external"
                "${GIT_EXECUTABLE}" apply --reverse --check
                "${SYMMETRIX_SPHERICART_PATCH}"
        WORKING_DIRECTORY "${SYMMETRIX_SPHERICART_SOURCE_DIR}"
        RESULT_VARIABLE SYMMETRIX_SPHERICART_PATCH_VERIFY)
    if(NOT SYMMETRIX_SPHERICART_PATCH_VERIFY EQUAL 0)
        message(FATAL_ERROR
            "Symmetrix's Sphericart lifecycle patch did not modify the build-tree copy.")
    endif()
    file(WRITE "${SYMMETRIX_SPHERICART_PATCH_STAMP}"
        "${SYMMETRIX_SPHERICART_EXPECTED_FINGERPRINT}")
endif()
add_subdirectory(
    "${SYMMETRIX_SPHERICART_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/external/sphericart-patched-build")
if(_symmetrix_host_arch_flag
        AND NOT _symmetrix_host_arch_applied_globally
        AND NOT Kokkos_ENABLE_CUDA
        AND NOT Kokkos_ENABLE_HIP)
    target_compile_options(sphericart PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${_symmetrix_host_arch_flag}>)
endif()
option(SYMMETRIX_KOKKOS "Enable Kokkos mode in Symmetrix." ON)
set(SYMMETRIX_HIP_BLAS "AUTO" CACHE STRING
    "HIP batched BLAS backend: AUTO, ROCBLAS, or KOKKOS.")
set_property(CACHE SYMMETRIX_HIP_BLAS PROPERTY STRINGS AUTO ROCBLAS KOKKOS)
string(TOUPPER "${SYMMETRIX_HIP_BLAS}" SYMMETRIX_HIP_BLAS)
if(NOT SYMMETRIX_HIP_BLAS MATCHES "^(AUTO|ROCBLAS|KOKKOS)$")
    message(FATAL_ERROR
        "Invalid SYMMETRIX_HIP_BLAS='${SYMMETRIX_HIP_BLAS}'; expected AUTO, ROCBLAS, or KOKKOS.")
endif()
if(Kokkos_ENABLE_HIP AND NOT SYMMETRIX_HIP_BLAS STREQUAL "KOKKOS")
    find_package(rocblas CONFIG QUIET)
    if(rocblas_FOUND OR TARGET roc::rocblas)
        set(KokkosKernels_ENABLE_TPL_ROCBLAS ON CACHE BOOL
            "Enable KokkosKernels rocBLAS TPL." FORCE)
    elseif(SYMMETRIX_HIP_BLAS STREQUAL "ROCBLAS")
        message(FATAL_ERROR
            "SYMMETRIX_HIP_BLAS=ROCBLAS requires a discoverable rocBLAS package.")
    endif()
elseif(Kokkos_ENABLE_HIP)
    set(KokkosKernels_ENABLE_TPL_ROCBLAS OFF CACHE BOOL
        "Enable KokkosKernels rocBLAS TPL." FORCE)
endif()
if(SYMMETRIX_KOKKOS
        AND NOT Kokkos_ENABLE_CUDA
        AND NOT Kokkos_ENABLE_HIP
        AND NOT DEFINED KokkosKernels_ENABLE_TPL_BLAS)
    set(KokkosKernels_ENABLE_TPL_BLAS ON CACHE BOOL
        "Use the discovered host BLAS for KokkosKernels dense operations.")
endif()
if(SYMMETRIX_KOKKOS)
    if(TARGET Kokkos::kokkos)
        message(STATUS "Symmetrix: Found existing Kokkos target.")
    else()
        add_subdirectory(external/kokkos)
    endif()
    # KokkosKernels hands a nonempty BLAS_LIBRARIES to find_library, which
    # cannot resolve Apple framework paths. Its empty cache entry hides the
    # normal variable only on the first configure, so hide it explicitly to
    # make every configure take KokkosKernels' own FindBLAS path.
    if(APPLE AND NOT SYMMETRIX_BLAS_LIBRARY)
        set(_symmetrix_blas_libraries "${BLAS_LIBRARIES}")
        unset(BLAS_LIBRARIES)
    endif()
    add_subdirectory(external/kokkos-kernels)
    if(APPLE AND NOT SYMMETRIX_BLAS_LIBRARY)
        set(BLAS_LIBRARIES "${_symmetrix_blas_libraries}")
        unset(_symmetrix_blas_libraries)
    endif()
endif()
message(STATUS "Symmetrix HIP BLAS policy: ${SYMMETRIX_HIP_BLAS}")

# ----- Particle-mesh FFT provider (KokkosFFT) -----

# Particle-mesh algorithms transform their fields through KokkosFFT, so the
# implementation exists only where an FFT provider was discovered at configure
# time: FFTW3 for the threaded host, cuFFT for CUDA, hipFFT for HIP.
# Nothing here fetches a dependency.  The pinned KokkosFFT source must already
# be in the tree, and an absent provider becomes an explicit unsupported state
# instead of a silently emulated or approximate feature.
set(_symmetrix_legacy_fourier_d3_defined FALSE)
if(DEFINED SYMMETRIX_FOURIER_D3)
    set(_symmetrix_legacy_fourier_d3_defined TRUE)
    string(TOUPPER "${SYMMETRIX_FOURIER_D3}" _symmetrix_legacy_fourier_d3)
    if(NOT _symmetrix_legacy_fourier_d3 MATCHES "^(AUTO|ON|OFF)$")
        message(FATAL_ERROR
            "Invalid deprecated SYMMETRIX_FOURIER_D3='${SYMMETRIX_FOURIER_D3}'; "
            "expected AUTO, ON, or OFF.")
    endif()
    message(DEPRECATION
        "SYMMETRIX_FOURIER_D3 is deprecated; use SYMMETRIX_PARTICLE_MESH.")
endif()

if(_symmetrix_legacy_fourier_d3_defined AND NOT DEFINED SYMMETRIX_PARTICLE_MESH)
    set(SYMMETRIX_PARTICLE_MESH "${_symmetrix_legacy_fourier_d3}" CACHE STRING
        "Particle mesh: AUTO discovers a KokkosFFT FFT provider, ON requires one, OFF disables particle-mesh algorithms.")
else()
    set(SYMMETRIX_PARTICLE_MESH "AUTO" CACHE STRING
        "Particle mesh: AUTO discovers a KokkosFFT FFT provider, ON requires one, OFF disables particle-mesh algorithms.")
endif()
set_property(CACHE SYMMETRIX_PARTICLE_MESH PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${SYMMETRIX_PARTICLE_MESH}" SYMMETRIX_PARTICLE_MESH)
if(NOT SYMMETRIX_PARTICLE_MESH MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
        "Invalid SYMMETRIX_PARTICLE_MESH='${SYMMETRIX_PARTICLE_MESH}'; "
        "expected AUTO, ON, or OFF.")
endif()
if(_symmetrix_legacy_fourier_d3_defined
        AND NOT _symmetrix_legacy_fourier_d3 STREQUAL SYMMETRIX_PARTICLE_MESH)
    message(FATAL_ERROR
        "Conflicting particle-mesh policies: SYMMETRIX_PARTICLE_MESH="
        "${SYMMETRIX_PARTICLE_MESH} but deprecated SYMMETRIX_FOURIER_D3="
        "${_symmetrix_legacy_fourier_d3}.")
endif()
unset(_symmetrix_legacy_fourier_d3)
unset(_symmetrix_legacy_fourier_d3_defined)

set(SYMMETRIX_KOKKOSFFT_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos-fft")
set(SYMMETRIX_PARTICLE_MESH_AVAILABLE 0)
set(SYMMETRIX_PARTICLE_MESH_PROVIDER "")
set(SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT "")
set(SYMMETRIX_PARTICLE_MESH_KOKKOSFFT_VERSION "")
set(SYMMETRIX_PARTICLE_MESH_REASON "")
set(SYMMETRIX_PARTICLE_MESH_PROVIDER_LIBRARIES "")
set(SYMMETRIX_PARTICLE_MESH_PROVIDER_INCLUDE_DIRS "")

if(EXISTS "${SYMMETRIX_KOKKOSFFT_SOURCE_DIR}/CMakeLists.txt")
    # The pinned version is a property of the source tree, so it is reported
    # even when the feature itself is disabled or its provider is missing.
    file(STRINGS "${SYMMETRIX_KOKKOSFFT_SOURCE_DIR}/CMakeLists.txt"
        _symmetrix_kokkosfft_project
        REGEX "^project\\(KokkosFFT VERSION [0-9]+[.][0-9]+[.][0-9]+ LANGUAGES CXX\\)")
    list(LENGTH _symmetrix_kokkosfft_project _symmetrix_kokkosfft_project_count)
    if(NOT _symmetrix_kokkosfft_project_count EQUAL 1)
        message(FATAL_ERROR
            "Cannot read the KokkosFFT version from "
            "${SYMMETRIX_KOKKOSFFT_SOURCE_DIR}/CMakeLists.txt.")
    endif()
    string(REGEX REPLACE
        "^project\\(KokkosFFT VERSION ([0-9]+[.][0-9]+[.][0-9]+) LANGUAGES CXX\\)$"
        "\\1" SYMMETRIX_PARTICLE_MESH_KOKKOSFFT_VERSION
        "${_symmetrix_kokkosfft_project}")
    unset(_symmetrix_kokkosfft_project)
    unset(_symmetrix_kokkosfft_project_count)
endif()

if(SYMMETRIX_PARTICLE_MESH STREQUAL "OFF")
    set(SYMMETRIX_PARTICLE_MESH_REASON "build-option-disabled")
elseif(NOT SYMMETRIX_KOKKOS)
    set(SYMMETRIX_PARTICLE_MESH_REASON "kokkos-disabled")
elseif(NOT EXISTS "${SYMMETRIX_KOKKOSFFT_SOURCE_DIR}/CMakeLists.txt")
    set(SYMMETRIX_PARTICLE_MESH_REASON "kokkosfft-source-missing")
endif()

# KokkosFFT's own FindFFTW module performs the final search, so this mirror only
# has to agree with it: the same library names, the same exclusivity under
# FFTW_ROOT, and the same fftw3.h requirement.  Every probe is refreshed on
# each configure so a reconfigured build root observes the current
# environment.  A disagreement can only be a false negative, which leaves the
# feature unsupported rather than claiming a provider that is not there.
set(_symmetrix_fftw_library_search "")
set(_symmetrix_fftw_include_search "")
if(FFTW_ROOT)
    set(_symmetrix_fftw_library_search
        PATHS "${FFTW_ROOT}"
        PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
        NO_DEFAULT_PATH)
    set(_symmetrix_fftw_include_search
        PATHS "${FFTW_ROOT}" PATH_SUFFIXES "include" NO_DEFAULT_PATH)
endif()
foreach(_symmetrix_fftw_probe IN ITEMS
        SYMMETRIX_FFTW_DOUBLE_LIBRARY
        SYMMETRIX_FFTW_FLOAT_LIBRARY
        SYMMETRIX_FFTW_DOUBLE_OMP_LIBRARY
        SYMMETRIX_FFTW_FLOAT_OMP_LIBRARY
        SYMMETRIX_FFTW_DOUBLE_THREADS_LIBRARY
        SYMMETRIX_FFTW_FLOAT_THREADS_LIBRARY
        SYMMETRIX_FFTW_INCLUDE_DIR)
    unset(${_symmetrix_fftw_probe} CACHE)
endforeach()
unset(_symmetrix_fftw_probe)

if(NOT SYMMETRIX_PARTICLE_MESH_REASON)
    if(Kokkos_ENABLE_CUDA)
        find_package(CUDAToolkit QUIET COMPONENTS cufft)
        if(TARGET CUDA::cufft)
            set(SYMMETRIX_PARTICLE_MESH_AVAILABLE 1)
            set(SYMMETRIX_PARTICLE_MESH_PROVIDER "cufft")
            set(SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT "CUFFT")
            set(SYMMETRIX_PARTICLE_MESH_PROVIDER_LIBRARIES "")
            if(CUDA_cufft_LIBRARY)
                set(SYMMETRIX_PARTICLE_MESH_PROVIDER_LIBRARIES
                    "${CUDA_cufft_LIBRARY}")
            endif()
        else()
            set(SYMMETRIX_PARTICLE_MESH_REASON "cufft-provider-missing")
        endif()
    elseif(Kokkos_ENABLE_HIP)
        find_package(hipfft CONFIG QUIET)
        if(TARGET hip::hipfft)
            set(SYMMETRIX_PARTICLE_MESH_AVAILABLE 1)
            set(SYMMETRIX_PARTICLE_MESH_PROVIDER "hipfft")
            set(SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT "HIPFFT")
            set(SYMMETRIX_PARTICLE_MESH_PROVIDER_LIBRARIES "")
            # The imported target may carry only a configuration-specific
            # location, and an unset property reads back as `-NOTFOUND`.
            foreach(_symmetrix_hipfft_property IN ITEMS
                    IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE)
                get_target_property(_symmetrix_hipfft_library hip::hipfft
                    "${_symmetrix_hipfft_property}")
                if(_symmetrix_hipfft_library
                        AND NOT "${_symmetrix_hipfft_library}" MATCHES "NOTFOUND")
                    set(SYMMETRIX_PARTICLE_MESH_PROVIDER_LIBRARIES
                        "${_symmetrix_hipfft_library}")
                    break()
                endif()
            endforeach()
            unset(_symmetrix_hipfft_library)
            unset(_symmetrix_hipfft_property)
        else()
            set(SYMMETRIX_PARTICLE_MESH_REASON "hipfft-provider-missing")
        endif()
    else()
        find_library(SYMMETRIX_FFTW_DOUBLE_LIBRARY
            NAMES fftw3 libfftw3-3 ${_symmetrix_fftw_library_search})
        find_library(SYMMETRIX_FFTW_FLOAT_LIBRARY
            NAMES fftw3f libfftw3f-3 ${_symmetrix_fftw_library_search})
        find_path(SYMMETRIX_FFTW_INCLUDE_DIR NAMES fftw3.h
            ${_symmetrix_fftw_include_search})
        if(NOT SYMMETRIX_FFTW_DOUBLE_LIBRARY
                OR NOT SYMMETRIX_FFTW_FLOAT_LIBRARY
                OR NOT SYMMETRIX_FFTW_INCLUDE_DIR)
            set(SYMMETRIX_PARTICLE_MESH_REASON "fftw3-provider-missing")
        else()
            set(_symmetrix_fftw_variant_libraries
                "${SYMMETRIX_FFTW_DOUBLE_LIBRARY};${SYMMETRIX_FFTW_FLOAT_LIBRARY}")
            set(_symmetrix_fftw_variant_include "${SYMMETRIX_FFTW_INCLUDE_DIR}")
            if(Kokkos_ENABLE_OPENMP)
                set(SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT "FFTW_OPENMP")
                find_library(SYMMETRIX_FFTW_DOUBLE_OMP_LIBRARY NAMES fftw3_omp
                    ${_symmetrix_fftw_library_search})
                find_library(SYMMETRIX_FFTW_FLOAT_OMP_LIBRARY NAMES fftw3f_omp
                    ${_symmetrix_fftw_library_search})
                if(SYMMETRIX_FFTW_DOUBLE_OMP_LIBRARY
                        AND SYMMETRIX_FFTW_FLOAT_OMP_LIBRARY)
                    list(APPEND _symmetrix_fftw_variant_libraries
                        "${SYMMETRIX_FFTW_DOUBLE_OMP_LIBRARY}"
                        "${SYMMETRIX_FFTW_FLOAT_OMP_LIBRARY}")
                else()
                    set(SYMMETRIX_PARTICLE_MESH_REASON
                        "fftw3-openmp-provider-missing")
                endif()
            elseif(Kokkos_ENABLE_THREADS)
                set(SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT "FFTW_THREADS")
                find_library(SYMMETRIX_FFTW_DOUBLE_THREADS_LIBRARY
                    NAMES fftw3_threads ${_symmetrix_fftw_library_search})
                find_library(SYMMETRIX_FFTW_FLOAT_THREADS_LIBRARY
                    NAMES fftw3f_threads ${_symmetrix_fftw_library_search})
                if(SYMMETRIX_FFTW_DOUBLE_THREADS_LIBRARY
                        AND SYMMETRIX_FFTW_FLOAT_THREADS_LIBRARY)
                    list(APPEND _symmetrix_fftw_variant_libraries
                        "${SYMMETRIX_FFTW_DOUBLE_THREADS_LIBRARY}"
                        "${SYMMETRIX_FFTW_FLOAT_THREADS_LIBRARY}")
                else()
                    set(SYMMETRIX_PARTICLE_MESH_REASON
                        "fftw3-threads-provider-missing")
                endif()
            elseif(Kokkos_ENABLE_SERIAL)
                set(SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT "FFTW_SERIAL")
            else()
                set(SYMMETRIX_PARTICLE_MESH_REASON "unsupported-execution-space")
            endif()
            if(NOT SYMMETRIX_PARTICLE_MESH_REASON)
                set(SYMMETRIX_PARTICLE_MESH_AVAILABLE 1)
                set(SYMMETRIX_PARTICLE_MESH_PROVIDER "fftw3")
                set(SYMMETRIX_PARTICLE_MESH_PROVIDER_LIBRARIES
                    "${_symmetrix_fftw_variant_libraries}")
                set(SYMMETRIX_PARTICLE_MESH_PROVIDER_INCLUDE_DIRS
                    "${_symmetrix_fftw_variant_include}")
            endif()
            unset(_symmetrix_fftw_variant_libraries)
            unset(_symmetrix_fftw_variant_include)
        endif()
    endif()
endif()
unset(_symmetrix_fftw_library_search)
unset(_symmetrix_fftw_include_search)

if(NOT SYMMETRIX_PARTICLE_MESH_AVAILABLE AND SYMMETRIX_PARTICLE_MESH STREQUAL "ON")
    message(FATAL_ERROR
        "SYMMETRIX_PARTICLE_MESH=ON requires a KokkosFFT FFT provider, but this "
        "configure found none (${SYMMETRIX_PARTICLE_MESH_REASON}). Configure "
        "builds never fetch dependencies: provide an FFTW3 development "
        "installation (fftw3.h, libfftw3, libfftw3f and the matching "
        "threading library, or FFTW_ROOT), or a CUDA/ROCm toolkit with cuFFT "
        "or hipFFT. Use SYMMETRIX_PARTICLE_MESH=AUTO to build without the "
        "feature.")
endif()

# Preserve the FourierD3 capability as an exact view of the shared
# particle-mesh provider.
set(SYMMETRIX_FOURIER_D3_AVAILABLE "${SYMMETRIX_PARTICLE_MESH_AVAILABLE}")
set(SYMMETRIX_FOURIER_D3_PROVIDER "${SYMMETRIX_PARTICLE_MESH_PROVIDER}")
set(SYMMETRIX_FOURIER_D3_PROVIDER_VARIANT
    "${SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT}")
set(SYMMETRIX_FOURIER_D3_KOKKOSFFT_VERSION
    "${SYMMETRIX_PARTICLE_MESH_KOKKOSFFT_VERSION}")
set(SYMMETRIX_FOURIER_D3_REASON "${SYMMETRIX_PARTICLE_MESH_REASON}")
set(SYMMETRIX_FOURIER_D3_PROVIDER_LIBRARIES
    "${SYMMETRIX_PARTICLE_MESH_PROVIDER_LIBRARIES}")
set(SYMMETRIX_FOURIER_D3_PROVIDER_INCLUDE_DIRS
    "${SYMMETRIX_PARTICLE_MESH_PROVIDER_INCLUDE_DIRS}")

if(SYMMETRIX_PARTICLE_MESH_AVAILABLE)
    message(STATUS
        "Symmetrix: particle-mesh provider ${SYMMETRIX_PARTICLE_MESH_PROVIDER} "
        "(${SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT}) from KokkosFFT "
        "${SYMMETRIX_PARTICLE_MESH_KOKKOSFFT_VERSION}.")
    # Upstream owns its TPL selection; these normal variables take precedence
    # over its option() defaults and select exactly the provider resolved
    # above. EXCLUDE_FROM_ALL keeps KokkosFFT's install rules (headers and
    # CMake exports) out of the parent install, so no build payload can leak
    # into a wheel.
    set(KokkosFFT_ENABLE_INTERNAL_KOKKOS OFF)
    set(KokkosFFT_ENABLE_TESTS OFF)
    set(KokkosFFT_ENABLE_EXAMPLES OFF)
    set(KokkosFFT_ENABLE_BENCHMARK OFF)
    set(KokkosFFT_ENABLE_DOCS OFF)
    set(KokkosFFT_ENABLE_DISTRIBUTED OFF)
    set(KokkosFFT_ENABLE_FFTW OFF)
    set(KokkosFFT_ENABLE_CUFFT OFF)
    set(KokkosFFT_ENABLE_ROCFFT OFF)
    set(KokkosFFT_ENABLE_HIPFFT OFF)
    set(KokkosFFT_ENABLE_ONEMKL OFF)
    if(SYMMETRIX_PARTICLE_MESH_PROVIDER STREQUAL "fftw3")
        set(KokkosFFT_ENABLE_FFTW ON)
    elseif(SYMMETRIX_PARTICLE_MESH_PROVIDER STREQUAL "cufft")
        set(KokkosFFT_ENABLE_CUFFT ON)
    elseif(SYMMETRIX_PARTICLE_MESH_PROVIDER STREQUAL "hipfft")
        set(KokkosFFT_ENABLE_HIPFFT ON)
    endif()
    add_subdirectory("${SYMMETRIX_KOKKOSFFT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/external/kokkos-fft-build"
        EXCLUDE_FROM_ALL)
else()
    message(STATUS
        "Symmetrix: particle-mesh algorithms are unavailable "
        "(${SYMMETRIX_PARTICLE_MESH_REASON}); dependent algorithms will report "
        "an unsupported build.")
endif()

if(SYMMETRIX_PARTICLE_MESH_AVAILABLE)
    set(SYMMETRIX_PARTICLE_MESH_AVAILABLE_JSON "true")
else()
    set(SYMMETRIX_PARTICLE_MESH_AVAILABLE_JSON "false")
endif()

if(SYMMETRIX_FOURIER_D3_AVAILABLE)
    set(SYMMETRIX_FOURIER_D3_AVAILABLE_JSON "true")
else()
    set(SYMMETRIX_FOURIER_D3_AVAILABLE_JSON "false")
endif()
set(_symmetrix_fourier_d3_libraries_json "")
foreach(_symmetrix_fourier_d3_library
        IN LISTS SYMMETRIX_FOURIER_D3_PROVIDER_LIBRARIES)
    if(_symmetrix_fourier_d3_libraries_json)
        string(APPEND _symmetrix_fourier_d3_libraries_json ", ")
    endif()
    string(APPEND _symmetrix_fourier_d3_libraries_json
        "\"${_symmetrix_fourier_d3_library}\"")
endforeach()
set(SYMMETRIX_FOURIER_D3_PROVIDER_LIBRARIES_JSON
    "${_symmetrix_fourier_d3_libraries_json}")
set(_symmetrix_fourier_d3_includes_json "")
foreach(_symmetrix_fourier_d3_include
        IN LISTS SYMMETRIX_FOURIER_D3_PROVIDER_INCLUDE_DIRS)
    if(_symmetrix_fourier_d3_includes_json)
        string(APPEND _symmetrix_fourier_d3_includes_json ", ")
    endif()
    string(APPEND _symmetrix_fourier_d3_includes_json
        "\"${_symmetrix_fourier_d3_include}\"")
endforeach()
set(SYMMETRIX_FOURIER_D3_PROVIDER_INCLUDE_DIRS_JSON
    "${_symmetrix_fourier_d3_includes_json}")
set(SYMMETRIX_PARTICLE_MESH_PROVIDER_LIBRARIES_JSON
    "${SYMMETRIX_FOURIER_D3_PROVIDER_LIBRARIES_JSON}")
set(SYMMETRIX_PARTICLE_MESH_PROVIDER_INCLUDE_DIRS_JSON
    "${SYMMETRIX_FOURIER_D3_PROVIDER_INCLUDE_DIRS_JSON}")
unset(_symmetrix_fourier_d3_libraries_json)
unset(_symmetrix_fourier_d3_library)
unset(_symmetrix_fourier_d3_includes_json)
unset(_symmetrix_fourier_d3_include)
configure_file(
    cmake/fourier_d3_capability.json.in
    "${CMAKE_CURRENT_BINARY_DIR}/fourier-d3-capability.json" @ONLY)
configure_file(
    cmake/particle_mesh_capability.json.in
    "${CMAKE_CURRENT_BINARY_DIR}/particle-mesh-capability.json" @ONLY)

# The parent project records the same decision in the native module metadata.
set(SYMMETRIX_PARTICLE_MESH_AVAILABLE
    "${SYMMETRIX_PARTICLE_MESH_AVAILABLE}" PARENT_SCOPE)
set(SYMMETRIX_PARTICLE_MESH_PROVIDER
    "${SYMMETRIX_PARTICLE_MESH_PROVIDER}" PARENT_SCOPE)
set(SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT
    "${SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT}" PARENT_SCOPE)
set(SYMMETRIX_PARTICLE_MESH_KOKKOSFFT_VERSION
    "${SYMMETRIX_PARTICLE_MESH_KOKKOSFFT_VERSION}" PARENT_SCOPE)
set(SYMMETRIX_PARTICLE_MESH_REASON
    "${SYMMETRIX_PARTICLE_MESH_REASON}" PARENT_SCOPE)
set(SYMMETRIX_FOURIER_D3_AVAILABLE
    "${SYMMETRIX_FOURIER_D3_AVAILABLE}" PARENT_SCOPE)
set(SYMMETRIX_FOURIER_D3_PROVIDER
    "${SYMMETRIX_FOURIER_D3_PROVIDER}" PARENT_SCOPE)
set(SYMMETRIX_FOURIER_D3_PROVIDER_VARIANT
    "${SYMMETRIX_FOURIER_D3_PROVIDER_VARIANT}" PARENT_SCOPE)
set(SYMMETRIX_FOURIER_D3_KOKKOSFFT_VERSION
    "${SYMMETRIX_FOURIER_D3_KOKKOSFFT_VERSION}" PARENT_SCOPE)
set(SYMMETRIX_FOURIER_D3_REASON
    "${SYMMETRIX_FOURIER_D3_REASON}" PARENT_SCOPE)

# ----- libsymmetrix -----

set(SYMMETRIX_SOURCE_FILES
    source/affine_mlp.cpp
    source/e3nn.cpp
    source/e3nn_product.cpp
    source/model_metadata.cpp
    source/compact_radial.cpp
    source/cubic_spline.cpp
    source/cubic_spline_set.cpp
    source/dispersion_parameters.cpp
    source/mace_nonlinear_schema.cpp
    source/mace_polar_schema.cpp
    source/oam_schema.cpp
    source/multilayer_perceptron.cpp
    source/multivariate_polynomial.cpp
    source/spherical_harmonic.cpp
    source/tools.cpp
    source/zbl.cpp)

list(APPEND SYMMETRIX_SOURCE_FILES
    source/mace_jit_cuda_plugin.cpp
    source/mace_nonlinear_jit_cuda_plugin.cpp
    source/mace_nonlinear_jit_host_plugin.cpp
    source/mace_nonlinear_split_phase_admission.cpp
    source/oam_cuda_plugin.cpp
    source/mace_jit_hip_plugin.cpp
    source/jit_hiprtc.cpp
    source/jit_nvrtc.cpp
    source/mace_jit_host_plugin.cpp
    source/mace_jit_m0_host_plugin.cpp
    source/mace_jit_r0_host_plugin.cpp)
list(APPEND SYMMETRIX_SOURCE_FILES
    source/oam_jit_host_plugin.cpp)
if(NOT UNIX)
    message(STATUS "Symmetrix: Execution runtime plugins require POSIX dynamic loading; building disabled stubs.")
endif()

if(SYMMETRIX_KOKKOS)
    # Keep the evaluator partitions as distinct objects: their CUDA template
    # instantiations dominate build time and are intended to compile in parallel.
    list(APPEND SYMMETRIX_SOURCE_FILES
        source/affine_mlp_kokkos.cpp
        source/e3nn_kokkos.cpp
        source/e3nn_product_kokkos.cpp
        source/cubic_spline_kokkos.cpp
        source/cubic_spline_set_kokkos.cpp
        source/d2_kokkos.cpp
        source/d3_kokkos.cpp
        source/device_backend.cpp
        source/dispersion_graph_kokkos.cpp
        source/mace_model_document.cpp
        source/mace_device_backend.cpp
        source/mace_kokkos_first_interaction.cpp
        source/mace_kokkos_response.cpp
        source/mace_kokkos_factorized_execution.cpp
        source/mace_kokkos_h1_phi1.cpp
        source/mace_les_model.cpp
        source/mace_les_kokkos.cpp
        source/mace_les_kokkos_evaluate.cpp
        source/mace_les_kokkos_lifecycle.cpp
        source/mace_les_kokkos_model.cpp
        source/les_monopole.cpp
        source/les_particle_mesh.cpp
        source/mace_kokkos_model.cpp
        source/neighbor_graph_kokkos.cpp
        source/mace_kokkos_factorized_lifecycle.cpp
        source/mace_kokkos_second_interaction.cpp
        source/mace_kokkos_factorized_analysis.cpp
        source/mace_kokkos_runtime.cpp
        source/mace_kokkos_evaluate.cpp
        source/mace_jit_operator_module.cpp
        source/mace_nonlinear_mh1_fast_path.cpp
        source/mace_nonlinear_product_host_plan.cpp
        source/mace_nonlinear_kokkos_components.cpp
        source/mace_nonlinear_kokkos_runtime.cpp
        source/mace_nonlinear_kokkos_evaluate.cpp
        source/oam_kokkos.cpp
        source/oam_kokkos_forward.cpp
        source/oam_kokkos_reverse.cpp
        source/oam_radial.cpp
        source/multilayer_perceptron_kokkos.cpp
        source/multivariate_polynomial_kokkos.cpp
        source/radial_function_set_kokkos.cpp
        source/tools_kokkos.cpp
        source/zbl_kokkos.cpp
        source/long_range_kernels.cpp
        source/mace_polar_deltas.cpp)
endif()

if(SYMMETRIX_KOKKOS AND SYMMETRIX_PARTICLE_MESH_AVAILABLE)
    # Native reciprocal-space storage, cache keys, and FFT-plan ownership.
    # Compiled only where the Task 6 capability resolved an FFT provider;
    # nothing stands in for it when the provider is absent.
    list(APPEND SYMMETRIX_SOURCE_FILES source/particle_mesh_storage.cpp)
    list(APPEND SYMMETRIX_SOURCE_FILES source/particle_mesh_transport.cpp)
    list(APPEND SYMMETRIX_SOURCE_FILES source/mace_polar_particle_mesh.cpp)
    list(APPEND SYMMETRIX_SOURCE_FILES source/fourier_d3_storage.cpp)
    # Modified coordination-number and separable-coefficient kernels.  They
    # share the same capability guard: an unsupported build exposes no stub.
    list(APPEND SYMMETRIX_SOURCE_FILES source/fourier_d3_coordination.cpp)
    # Particle-mesh forward path: B-spline assignment, unnormalized transform
    # pair, reciprocal contraction, cotangent, and the self term.
    list(APPEND SYMMETRIX_SOURCE_FILES source/fourier_d3_reciprocal.cpp)
    # Analytic reverse path: coefficient sensitivities, analytic gather forces,
    # the receiver-only coordination adjoint, the affine virial, the
    # tensile-positive Voigt stress, and the atomic-energy partition.
    list(APPEND SYMMETRIX_SOURCE_FILES source/fourier_d3_reverse.cpp)
    # The evaluator: the dtype-specific composition of the reverse path behind the
    # graph/update/compute/result protocol the Python runtime consumes, with the
    # canonical table payload, the rank windowing, and the resolved mesh.
    list(APPEND SYMMETRIX_SOURCE_FILES source/fourier_d3_evaluator.cpp)
endif()

add_library(symmetrix ${SYMMETRIX_SOURCE_FILES})

include(cmake/prebuilt_operator_artifacts.cmake)
symmetrix_add_prebuilt_operator_artifacts()
# The catalog helper is a function, so its PARENT_SCOPE is this subdirectory.
# Forward the generated header to the Python parent as well; otherwise a reused
# configure replaces it with the disabled placeholder while the catalog target
# correctly remains up to date.
if(SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER)
    set(SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER
        "${SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER}" PARENT_SCOPE)
endif()

# HIP compilation warns that global fast-math invalidates NaN/Inf checks used
# by model validation. Keep the historical host/CUDA policy while HIP is
# qualified without those assumptions.
if(SYMMETRIX_FAST_MATH AND NOT Kokkos_ENABLE_HIP)
    target_compile_options(symmetrix
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_FAST_MATH_ENABLED=1)
endif()
if(_symmetrix_host_arch_flag
        AND NOT _symmetrix_host_arch_applied_globally
        AND NOT Kokkos_ENABLE_CUDA
        AND NOT Kokkos_ENABLE_HIP)
    target_compile_options(symmetrix PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${_symmetrix_host_arch_flag}>)
endif()

# The standard M0 term folds expand to several hundred operands, beyond the
# default 256-level nesting limit of older Apple Clang releases (Xcode 16).
if(APPLE)
    target_compile_options(symmetrix PUBLIC
        $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fbracket-depth=1024>)
endif()

set(_symmetrix_python_source_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/../symmetrix/source/symmetrix")
if(NOT IS_DIRECTORY "${_symmetrix_python_source_dir}")
    # In a self-contained sdist, libsymmetrix is relocated inside the Python
    # project root instead of remaining beside the symmetrix project directory.
    set(_symmetrix_python_source_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/../source/symmetrix")
endif()
if(NOT IS_DIRECTORY "${_symmetrix_python_source_dir}")
    message(FATAL_ERROR
        "Symmetrix: cannot locate the Python source directory containing the "
        "JIT plugin ABI headers")
endif()

target_include_directories(symmetrix
    PUBLIC  source
            "${_symmetrix_python_source_dir}"
            external/json/single_include
    PRIVATE "${SYMMETRIX_SPHERICART_SOURCE_DIR}/sphericart/include")
if(SYMMETRIX_BLAS_INCLUDE_DIR)
    target_include_directories(symmetrix PRIVATE "${SYMMETRIX_BLAS_INCLUDE_DIR}")
endif()

if(BLAS_LIBRARY_DIRS)
    # The normalized names above are searched only in the default linker
    # directories otherwise, so a pinned BLAS prefix would be resolved from
    # whatever the system happens to expose and a non-default prefix would fail
    # to link at all. `symmetrix` is a static archive, so the directories have to
    # reach the target that performs the link.
    target_link_directories(symmetrix PUBLIC ${BLAS_LIBRARY_DIRS})
endif()
target_link_libraries(symmetrix
    PRIVATE ${_symmetrix_blas_link_libraries}
            sphericart)
if(UNIX)
    target_link_libraries(symmetrix PRIVATE ${CMAKE_DL_LIBS})
endif()
if(SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix
        PRIVATE Kokkos::kokkos
                Kokkos::kokkoskernels)
endif()
target_compile_definitions(symmetrix
    PRIVATE
        SYMMETRIX_PARTICLE_MESH_AVAILABLE=${SYMMETRIX_PARTICLE_MESH_AVAILABLE}
        SYMMETRIX_PARTICLE_MESH_REASON="${SYMMETRIX_PARTICLE_MESH_REASON}")
if(SYMMETRIX_PARTICLE_MESH_AVAILABLE)
    # KokkosFFT is header-only; the provider libraries arrive through its
    # interface target. Native reciprocal-space storage and kernels consume
    # this boundary behind the internal particle-mesh capability.
    target_link_libraries(symmetrix PRIVATE KokkosFFT::fft)
    target_compile_definitions(symmetrix
        PRIVATE
            SYMMETRIX_PARTICLE_MESH_ENABLED=1
            SYMMETRIX_FOURIER_D3_ENABLED=1)
endif()

option(SYMMETRIX_METAL "Build the Apple Metal runtime layer." OFF)
if(SYMMETRIX_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "SYMMETRIX_METAL requires an Apple platform.")
    endif()
    set(SYMMETRIX_METAL_SELFTEST OFF CACHE BOOL
        "Build the Metal runtime self-test executable.")
    add_subdirectory(source/metal)
    target_link_libraries(symmetrix PRIVATE symmetrix_metal)
endif()


# ----- options -----

if (SYMMETRIX_SPHERICART_CUDA)
    message(STATUS "Symmetrix: Will use CUDA version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_CUDA)
endif()

if (SYMMETRIX_SPHERICART_SYCL)
    message(STATUS "Symmetrix: Will use SYCL version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_SYCL)
endif()

# The source tree gates its accelerator batched-BLAS routes on the KokkosKernels
# TPL macros, and ``device_backend.cpp`` reports the HIP batched-BLAS capability
# from the same macro.  KokkosKernels states the TPLs through its own generated
# configuration rather than through a header these translation units include, so
# the options this build resolved are published here: without them every guard
# reads false and a route the TPL actually enables is compiled out.
if(KokkosKernels_ENABLE_TPL_ROCBLAS)
    message(STATUS "Symmetrix: KokkosKernels rocBLAS TPL is available.")
    target_compile_definitions(symmetrix PRIVATE KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
endif()

if(KokkosKernels_ENABLE_TPL_CUBLAS)
    message(STATUS "Symmetrix: KokkosKernels cuBLAS TPL is available.")
    target_compile_definitions(symmetrix PRIVATE KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
endif()
