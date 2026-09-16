cmake_minimum_required(VERSION 3.27)
project(Symmetrix)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "Disable IPO.")

option(SYMMETRIX_REDACT_BUILD_PATHS
  "Replace checkout and build roots embedded in compiled wheel binaries." OFF)
if(SYMMETRIX_REDACT_BUILD_PATHS)
  get_filename_component(_symmetrix_repository_root
    "${CMAKE_CURRENT_SOURCE_DIR}/.." REALPATH)
  foreach(_symmetrix_path_map IN ITEMS
      "${_symmetrix_repository_root}=/usr/src/symmetrix-xl"
      "${CMAKE_BINARY_DIR}=/usr/src/symmetrix-xl-build")
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${_symmetrix_path_map}>"
      "$<$<COMPILE_LANGUAGE:CXX>:-fdebug-prefix-map=${_symmetrix_path_map}>"
      "$<$<COMPILE_LANGUAGE:CXX>:-fmacro-prefix-map=${_symmetrix_path_map}>")
  endforeach()
endif()

set(SYMMETRIX_KOKKOS ON CACHE BOOL "Enable Symmetrix-Kokkos.")
set(SYMMETRIX_HOST_ARCH "native" CACHE STRING
    "Host -march target; use x86-64-v3 for a portable x86-64 build or none to disable.")
set(SYMMETRIX_DEVICE_BACKEND "NONE" CACHE STRING
    "Device backend: AUTO, NONE, CUDA, or HIP.")
set(SYMMETRIX_PYTHON_MODULE_NAME "" CACHE STRING
    "Native Python module token; empty selects the release default.")
set(SYMMETRIX_PYTHON_INSTALL_PACKAGE "" CACHE STRING
    "Python package receiving the native extension.")
set(SYMMETRIX_DISTRIBUTION_NAME "symmetrix-xl" CACHE STRING
    "Owning Python distribution recorded in backend metadata.")
set(SYMMETRIX_FRONTEND_VERSION "0.1.0" CACHE STRING
    "Compatible Symmetrix Python frontend version.")
set(SYMMETRIX_BACKEND_SELECTOR "" CACHE STRING
    "Runtime backend selector recorded in backend metadata.")
set(SYMMETRIX_BACKEND_ARCHITECTURE "" CACHE STRING
    "Normalized architecture recorded in backend metadata.")
set_property(CACHE SYMMETRIX_DEVICE_BACKEND PROPERTY STRINGS AUTO NONE CUDA HIP)
string(TOUPPER "${SYMMETRIX_DEVICE_BACKEND}" SYMMETRIX_DEVICE_BACKEND)
set(_symmetrix_valid_device_backends AUTO NONE CUDA HIP)
if(NOT SYMMETRIX_DEVICE_BACKEND IN_LIST _symmetrix_valid_device_backends)
  message(FATAL_ERROR
    "Invalid SYMMETRIX_DEVICE_BACKEND='${SYMMETRIX_DEVICE_BACKEND}'; "
    "expected AUTO, NONE, CUDA, or HIP.")
endif()

set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "Set Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION.")
set(KokkosKernels_INST_DOUBLE OFF CACHE BOOL "Disable explicit instantiation of double kernels.")
set(KokkosKernels_INST_LAYOUTLEFT OFF CACHE BOOL "Disable explicit instantiation of layoutleft kernels.")

# Preserve explicit backend selections. When no backend is selected, prefer
# CUDA if it is available and otherwise use OpenMP for the threaded CPU build.
set(_symmetrix_kokkos_host_backend_enabled FALSE)
if((DEFINED Kokkos_ENABLE_OPENMP AND Kokkos_ENABLE_OPENMP)
    OR (DEFINED Kokkos_ENABLE_SERIAL AND Kokkos_ENABLE_SERIAL))
  set(_symmetrix_kokkos_host_backend_enabled TRUE)
endif()

include(CheckLanguage)
set(_cuda_found FALSE)
if(SYMMETRIX_DEVICE_BACKEND STREQUAL "AUTO"
    OR SYMMETRIX_DEVICE_BACKEND STREQUAL "CUDA")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(_cuda_found TRUE)
  endif()
endif()

if(Kokkos_ENABLE_CUDA AND Kokkos_ENABLE_HIP)
  message(FATAL_ERROR
    "Kokkos CUDA and HIP cannot be enabled in the same Symmetrix build.")
endif()
if(SYMMETRIX_DEVICE_BACKEND STREQUAL "CUDA" AND Kokkos_ENABLE_HIP)
  message(FATAL_ERROR "SYMMETRIX_DEVICE_BACKEND=CUDA contradicts Kokkos_ENABLE_HIP=ON.")
endif()
if(SYMMETRIX_DEVICE_BACKEND STREQUAL "HIP" AND Kokkos_ENABLE_CUDA)
  message(FATAL_ERROR "SYMMETRIX_DEVICE_BACKEND=HIP contradicts Kokkos_ENABLE_CUDA=ON.")
endif()
if(SYMMETRIX_DEVICE_BACKEND STREQUAL "NONE"
    AND (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP))
  message(FATAL_ERROR
    "SYMMETRIX_DEVICE_BACKEND=NONE contradicts an enabled Kokkos GPU backend.")
endif()

set(_symmetrix_resolved_device_backend NONE)
if(SYMMETRIX_KOKKOS)
  if(SYMMETRIX_DEVICE_BACKEND STREQUAL "HIP" OR Kokkos_ENABLE_HIP)
    set(_symmetrix_resolved_device_backend HIP)
  elseif(SYMMETRIX_DEVICE_BACKEND STREQUAL "CUDA" OR Kokkos_ENABLE_CUDA)
    set(_symmetrix_resolved_device_backend CUDA)
  elseif(SYMMETRIX_DEVICE_BACKEND STREQUAL "AUTO"
      AND _cuda_found AND NOT _symmetrix_kokkos_host_backend_enabled)
    set(_symmetrix_resolved_device_backend CUDA)
  endif()
endif()

# Ignore retained Kokkos GPU cache entries in a native-only build without
# erasing them if the user re-enables SYMMETRIX_KOKKOS later.
if(NOT SYMMETRIX_KOKKOS)
  set(Kokkos_ENABLE_CUDA OFF)
  set(Kokkos_ENABLE_HIP OFF)
  set(SYMMETRIX_SPHERICART_CUDA OFF)
endif()

if(_symmetrix_resolved_device_backend STREQUAL "CUDA")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend." FORCE)
  set(Kokkos_ENABLE_HIP OFF CACHE BOOL "Enable Kokkos HIP backend." FORCE)
if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_CUDA
    AND NOT DEFINED SYMMETRIX_SPHERICART_CUDA)
  set(SYMMETRIX_SPHERICART_CUDA ON CACHE BOOL "Enable SpheriCart-CUDA.")
endif()
elseif(_symmetrix_resolved_device_backend STREQUAL "HIP")
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable Kokkos HIP backend." FORCE)
  set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Enable Kokkos CUDA backend." FORCE)
  set(Kokkos_ARCH_NATIVE OFF CACHE BOOL "Set Kokkos_ARCH_NATIVE." FORCE)
  set(SYMMETRIX_SPHERICART_CUDA OFF CACHE BOOL "Enable SpheriCart-CUDA." FORCE)
  set(SPHERICART_ENABLE_CUDA OFF CACHE BOOL "Build SpheriCart CUDA backend." FORCE)
  set(SPHERICART_OPENMP OFF CACHE BOOL "Build SpheriCart with OpenMP." FORCE)
  set(SPHERICART_ARCH_NATIVE OFF CACHE BOOL
    "Try to use -march=native when compiling SpheriCart." FORCE)

  if(Kokkos_ENABLE_OPENMP)
    message(FATAL_ERROR
      "HIP currently supports Kokkos Serial as its host execution space; "
      "Kokkos_ENABLE_OPENMP=ON is not qualified.")
  endif()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL
    "Enable the supported HIP host execution space." FORCE)

  get_filename_component(_symmetrix_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" --version
    OUTPUT_VARIABLE _symmetrix_hipcc_version_output
    ERROR_VARIABLE _symmetrix_hipcc_version_error
    RESULT_VARIABLE _symmetrix_hipcc_version_result)
  string(CONCAT _symmetrix_hipcc_version_text
    "${_symmetrix_hipcc_version_output}" "${_symmetrix_hipcc_version_error}")
  if(NOT _symmetrix_cxx_name MATCHES "^hipcc([.]bin)?$"
      AND NOT _symmetrix_hipcc_version_text MATCHES "HIP version")
    message(FATAL_ERROR
      "SYMMETRIX_DEVICE_BACKEND=HIP requires hipcc as CMAKE_CXX_COMPILER.")
  endif()
  string(REGEX MATCH "HIP version: *([0-9]+)[.]([0-9]+)"
    _symmetrix_hip_version_match "${_symmetrix_hipcc_version_text}")
  if(NOT _symmetrix_hip_version_match)
    message(FATAL_ERROR "Could not determine the HIP version from hipcc --version.")
  endif()
  if(CMAKE_MATCH_1 LESS 6 OR (CMAKE_MATCH_1 EQUAL 6 AND CMAKE_MATCH_2 LESS 2))
    message(FATAL_ERROR
      "Pinned Kokkos 5.0.2 requires hipcc 6.2 or newer; found ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.")
  endif()

  get_cmake_property(_symmetrix_cache_variables CACHE_VARIABLES)
  set(_symmetrix_amd_target_count 0)
  set(_symmetrix_amd_targets "")
  foreach(_symmetrix_variable IN LISTS _symmetrix_cache_variables)
    if(_symmetrix_variable MATCHES "^Kokkos_ARCH_AMD_" AND ${_symmetrix_variable})
      math(EXPR _symmetrix_amd_target_count "${_symmetrix_amd_target_count}+1")
      list(APPEND _symmetrix_amd_targets "${_symmetrix_variable}")
    endif()
  endforeach()
  if(_symmetrix_amd_target_count EQUAL 0)
    message(FATAL_ERROR
      "HIP requires an explicit Kokkos AMD architecture trait such as "
      "Kokkos_ARCH_AMD_GFX1100=ON and an offload target matching the device.")
  endif()
  set(_symmetrix_kokkos_host_backend_enabled TRUE)
else()
  set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Enable Kokkos CUDA backend." FORCE)
  set(Kokkos_ENABLE_HIP OFF CACHE BOOL "Enable Kokkos HIP backend." FORCE)
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
endif()

# Kokkos has a native architecture trait, but no generic x86-64-v3 trait.
# Apply explicit non-native host targets before adding bundled dependencies so
# the core, bindings, SpheriCart, Kokkos, and KokkosKernels share one baseline.
set(_symmetrix_host_arch_applied_globally FALSE)
if(_symmetrix_resolved_device_backend STREQUAL "NONE"
    AND NOT SYMMETRIX_HOST_ARCH STREQUAL ""
    AND NOT SYMMETRIX_HOST_ARCH STREQUAL "none"
    AND NOT SYMMETRIX_HOST_ARCH STREQUAL "native")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-march=${SYMMETRIX_HOST_ARCH}>)
  set(_symmetrix_host_arch_applied_globally TRUE)
endif()

if(_symmetrix_resolved_device_backend STREQUAL "CUDA" AND NOT _cuda_found)
  message(FATAL_ERROR
    "The CUDA backend requires a discoverable CUDA compiler. "
    "Set CMAKE_CUDA_COMPILER or put nvcc on PATH.")
endif()

if(SYMMETRIX_KOKKOS
    AND _symmetrix_resolved_device_backend STREQUAL "NONE"
    AND NOT DEFINED Kokkos_ENABLE_OPENMP
    AND (NOT DEFINED Kokkos_ENABLE_SERIAL OR NOT Kokkos_ENABLE_SERIAL))
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend.")
endif()

message(STATUS "Symmetrix configuration:")
message(STATUS "  device backend: ${_symmetrix_resolved_device_backend}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Kokkos Serial: ${Kokkos_ENABLE_SERIAL}")
message(STATUS "  Kokkos OpenMP: ${Kokkos_ENABLE_OPENMP}")
message(STATUS "  Kokkos CUDA: ${Kokkos_ENABLE_CUDA}")
message(STATUS "  Kokkos HIP: ${Kokkos_ENABLE_HIP}")
message(STATUS "  Kokkos native architecture: ${Kokkos_ARCH_NATIVE}")
message(STATUS "  host compiler architecture: ${SYMMETRIX_HOST_ARCH}")
if(_symmetrix_resolved_device_backend STREQUAL "HIP")
  message(STATUS "  AMD targets: ${_symmetrix_amd_targets}")
endif()

if(NOT SYMMETRIX_PYTHON_MODULE_NAME)
  if(_symmetrix_resolved_device_backend STREQUAL "NONE")
    set(SYMMETRIX_PYTHON_MODULE_NAME "_native_cpu")
  else()
    # Compatibility for explicit, low-level monolithic GPU builds. Release
    # backend packages always pass an architecture-qualified module name.
    set(SYMMETRIX_PYTHON_MODULE_NAME "symmetrix")
  endif()
endif()
if(NOT SYMMETRIX_PYTHON_MODULE_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR
    "Invalid SYMMETRIX_PYTHON_MODULE_NAME=${SYMMETRIX_PYTHON_MODULE_NAME}")
endif()
if(NOT SYMMETRIX_PYTHON_INSTALL_PACKAGE)
  set(SYMMETRIX_PYTHON_INSTALL_PACKAGE "symmetrix")
endif()
if(NOT SYMMETRIX_PYTHON_INSTALL_PACKAGE MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR
    "Invalid SYMMETRIX_PYTHON_INSTALL_PACKAGE=${SYMMETRIX_PYTHON_INSTALL_PACKAGE}")
endif()
if(NOT SYMMETRIX_BACKEND_SELECTOR)
  if(_symmetrix_resolved_device_backend STREQUAL "NONE")
    set(SYMMETRIX_BACKEND_SELECTOR "cpu")
  else()
    string(TOLOWER "${_symmetrix_resolved_device_backend}"
      SYMMETRIX_BACKEND_SELECTOR)
  endif()
endif()
if(NOT SYMMETRIX_BACKEND_ARCHITECTURE)
  if(_symmetrix_resolved_device_backend STREQUAL "NONE")
    set(SYMMETRIX_BACKEND_ARCHITECTURE "${SYMMETRIX_HOST_ARCH}")
  else()
    set(SYMMETRIX_BACKEND_ARCHITECTURE "unspecified")
  endif()
endif()
message(STATUS "  Python native module: ${SYMMETRIX_PYTHON_MODULE_NAME}")
message(STATUS "  Python install package: ${SYMMETRIX_PYTHON_INSTALL_PACKAGE}")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_OPENMP)
  find_package(OpenMP 3.0 COMPONENTS CXX REQUIRED)
endif()
if(SYMMETRIX_KOKKOS AND _symmetrix_resolved_device_backend STREQUAL "NONE")
  enable_language(Fortran)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../libsymmetrix/CMakeLists.txt")
  set(SYMMETRIX_LIBSYM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libsymmetrix")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libsymmetrix/CMakeLists.txt")
  set(SYMMETRIX_LIBSYM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libsymmetrix")
else()
  message(FATAL_ERROR
    "libsymmetrix sources were not found beside or inside the Python package. "
    "Build from the published sdist or a recursive source checkout, or run "
    "'git submodule update --init --recursive' first.")
endif()
add_subdirectory("${SYMMETRIX_LIBSYM_SOURCE_DIR}"
  ${CMAKE_CURRENT_BINARY_DIR}/libsymmetrix)

add_subdirectory(external/pybind11)

set(SYMMETRIX_BUILD_OPENMP_ENABLED 0)
set(SYMMETRIX_BUILD_OPENMP_VERSION "")
set(SYMMETRIX_BUILD_OPENMP_RUNTIME_PATH "")
set(SYMMETRIX_BUILD_OPENMP_RUNTIME_SHA256 "")
file(REAL_PATH "${CMAKE_CXX_COMPILER}" SYMMETRIX_BUILD_CXX_COMPILER_PATH)
set(SYMMETRIX_BUILD_DEVICE_COMPILER_ID "")
set(SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION "")
if(_symmetrix_resolved_device_backend STREQUAL "CUDA"
    AND CMAKE_CUDA_COMPILER)
  set(SYMMETRIX_BUILD_DEVICE_COMPILER_ID "nvcc")
  execute_process(
    COMMAND "${CMAKE_CUDA_COMPILER}" --version
    OUTPUT_VARIABLE _symmetrix_device_compiler_version
    ERROR_QUIET)
  string(REGEX MATCH "release ([0-9]+\\.[0-9]+(\\.[0-9]+)?)"
    _symmetrix_cuda_release "${_symmetrix_device_compiler_version}")
  if(_symmetrix_cuda_release)
    string(REGEX REPLACE "release " ""
      SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION "${_symmetrix_cuda_release}")
  endif()
elseif(_symmetrix_resolved_device_backend STREQUAL "HIP")
  set(SYMMETRIX_BUILD_DEVICE_COMPILER_ID "hipcc")
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" --version
    OUTPUT_VARIABLE _symmetrix_device_compiler_version
    ERROR_QUIET)
  string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?"
    SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION
    "${_symmetrix_device_compiler_version}")
endif()
set(SYMMETRIX_BUILD_SOURCE_COMMIT "unknown")
set(SYMMETRIX_BUILD_SOURCE_DIRTY 0)
find_package(Git QUIET)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    OUTPUT_VARIABLE SYMMETRIX_BUILD_SOURCE_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _symmetrix_git_result)
  if(NOT _symmetrix_git_result EQUAL 0)
    set(SYMMETRIX_BUILD_SOURCE_COMMIT "unknown")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    OUTPUT_VARIABLE _symmetrix_git_status
    ERROR_QUIET)
  if(_symmetrix_git_status)
    set(SYMMETRIX_BUILD_SOURCE_DIRTY 1)
  endif()
endif()
if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_OPENMP)
  set(SYMMETRIX_BUILD_OPENMP_ENABLED 1)
  if(DEFINED OpenMP_CXX_VERSION)
    set(SYMMETRIX_BUILD_OPENMP_VERSION "${OpenMP_CXX_VERSION}")
  endif()
  foreach(_symmetrix_openmp_library_name IN LISTS OpenMP_CXX_LIB_NAMES)
    if(_symmetrix_openmp_library_name MATCHES "^(gomp|omp|iomp5)$")
      set(_symmetrix_openmp_library_variable
        "OpenMP_${_symmetrix_openmp_library_name}_LIBRARY")
      if(DEFINED ${_symmetrix_openmp_library_variable}
          AND EXISTS "${${_symmetrix_openmp_library_variable}}")
        file(REAL_PATH "${${_symmetrix_openmp_library_variable}}"
          SYMMETRIX_BUILD_OPENMP_RUNTIME_PATH)
        file(READ "${SYMMETRIX_BUILD_OPENMP_RUNTIME_PATH}"
          _symmetrix_openmp_runtime_magic LIMIT 4 HEX)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
            AND _symmetrix_openmp_library_name STREQUAL "gomp"
            AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
            AND NOT _symmetrix_openmp_runtime_magic STREQUAL "7f454c46")
          # GCC installations may expose libgomp.so as a linker script. Record
          # the soname DSO that the loader will actually map, not the script.
          execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libgomp.so.1
            OUTPUT_VARIABLE _symmetrix_gomp_soname_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
          if(EXISTS "${_symmetrix_gomp_soname_path}")
            file(REAL_PATH "${_symmetrix_gomp_soname_path}"
              SYMMETRIX_BUILD_OPENMP_RUNTIME_PATH)
          endif()
          file(READ "${SYMMETRIX_BUILD_OPENMP_RUNTIME_PATH}"
            _symmetrix_openmp_runtime_magic LIMIT 4 HEX)
          if(NOT _symmetrix_openmp_runtime_magic STREQUAL "7f454c46")
            message(FATAL_ERROR
              "GCC's linked libgomp DSO could not be resolved from "
              "${${_symmetrix_openmp_library_variable}}")
          endif()
        endif()
        file(SHA256 "${SYMMETRIX_BUILD_OPENMP_RUNTIME_PATH}"
          SYMMETRIX_BUILD_OPENMP_RUNTIME_SHA256)
        break()
      endif()
    endif()
  endforeach()
  if(NOT SYMMETRIX_BUILD_OPENMP_RUNTIME_PATH)
    message(FATAL_ERROR
      "Kokkos OpenMP is enabled, but its runtime library could not be resolved "
      "for deployment metadata.")
  endif()
endif()

configure_file(
  source/cpp/openmp_build_metadata.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/openmp_build_metadata.hpp
  @ONLY)

set(SYMMETRIX_BINDINGS
    source/cpp/cubic_spline.cpp
    source/cpp/cubic_spline_set.cpp
    source/cpp/e3nn.cpp
    source/cpp/mace.cpp
    source/cpp/mace_nonlinear.cpp
    source/cpp/multilayer_perceptron.cpp
    source/cpp/multivariate_polynomial.cpp
    source/cpp/spherical_harmonic.cpp
    source/cpp/jit_nvrtc.cpp
    source/cpp/tools.cpp
    source/cpp/zbl.cpp)

set(SYMMETRIX_BINDINGS_KOKKOS
    source/cpp/cubic_spline_kokkos.cpp
    source/cpp/cubic_spline_set_kokkos.cpp
    source/cpp/mace_kokkos.cpp
    source/cpp/mace_nonlinear_kokkos.cpp
    source/cpp/multilayer_perceptron_kokkos.cpp
    source/cpp/multivariate_polynomial_kokkos.cpp
    # source/spherical_harmonic_kokkos.cpp
    source/cpp/tools_kokkos.cpp
    source/cpp/zbl_kokkos.cpp)

if(SYMMETRIX_KOKKOS)
    message(STATUS "Symmetrix will have Kokkos support.")
    list(APPEND SYMMETRIX_BINDINGS ${SYMMETRIX_BINDINGS_KOKKOS})
endif()

option(SYMMETRIX_BUILD_BENCHMARKS "Build standalone Symmetrix benchmarks." OFF)
if(SYMMETRIX_KOKKOS AND SYMMETRIX_BUILD_BENCHMARKS)
    add_executable(
        factorized_matrix_feasibility
        ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/factorized_matrix_feasibility.cpp)
    target_link_libraries(
        factorized_matrix_feasibility
        PRIVATE Kokkos::kokkos Kokkos::kokkoskernels)
endif()

pybind11_add_module(symmetrix_bindings MODULE ${SYMMETRIX_BINDINGS} source/cpp/symmetrix.cpp)
set_target_properties(symmetrix_bindings PROPERTIES
  OUTPUT_NAME "${SYMMETRIX_PYTHON_MODULE_NAME}")
target_include_directories(symmetrix_bindings PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_definitions(symmetrix_bindings PRIVATE
  SYMMETRIX_PYTHON_MODULE=${SYMMETRIX_PYTHON_MODULE_NAME}
  SYMMETRIX_NATIVE_ABI=1
  SYMMETRIX_BUILD_BACKEND="${SYMMETRIX_BACKEND_SELECTOR}"
  SYMMETRIX_BUILD_ARCHITECTURE="${SYMMETRIX_BACKEND_ARCHITECTURE}"
  SYMMETRIX_BUILD_DISTRIBUTION="${SYMMETRIX_DISTRIBUTION_NAME}"
  SYMMETRIX_BUILD_SOURCE_COMMIT="${SYMMETRIX_BUILD_SOURCE_COMMIT}"
  SYMMETRIX_BUILD_SOURCE_DIRTY=${SYMMETRIX_BUILD_SOURCE_DIRTY}
  SYMMETRIX_BUILD_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  SYMMETRIX_BUILD_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  SYMMETRIX_BUILD_DEVICE_COMPILER_ID="${SYMMETRIX_BUILD_DEVICE_COMPILER_ID}"
  SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION="${SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION}")
target_link_libraries(symmetrix_bindings PRIVATE symmetrix)
if(SYMMETRIX_HIDE_STATIC_BLAS_SYMBOLS)
  if(UNIX AND NOT APPLE)
    target_link_options(symmetrix_bindings PRIVATE
      "LINKER:--exclude-libs,libopenblas.a")
  else()
    message(FATAL_ERROR
      "SYMMETRIX_HIDE_STATIC_BLAS_SYMBOLS is supported only on ELF platforms.")
  endif()
endif()
if(SYMMETRIX_KOKKOS)
    target_compile_definitions(symmetrix_bindings PRIVATE SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix_bindings PRIVATE Kokkos::kokkos ${CMAKE_DL_LIBS})
endif()

install(TARGETS symmetrix_bindings
  DESTINATION "${SYMMETRIX_PYTHON_INSTALL_PACKAGE}")
if(_symmetrix_resolved_device_backend STREQUAL "NONE"
    AND SYMMETRIX_PYTHON_INSTALL_PACKAGE STREQUAL "symmetrix")
  configure_file(
    source/cpp/backend_cpu.json.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/backend_cpu.json
    @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/backend_cpu.json
    DESTINATION symmetrix RENAME _backend_cpu.json)
endif()
