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
    "Optional exact BLAS archive used by accelerator release builds.")
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
    add_subdirectory(external/kokkos-kernels)
endif()
message(STATUS "Symmetrix HIP BLAS policy: ${SYMMETRIX_HIP_BLAS}")

# ----- libsymmetrix -----

set(SYMMETRIX_SOURCE_FILES
    source/affine_mlp.cpp
    source/e3nn.cpp
    source/e3nn_product.cpp
    source/model_metadata.cpp
    source/compact_radial.cpp
    source/cubic_spline.cpp
    source/cubic_spline_set.cpp
    source/mace.cpp
    source/mace_nonlinear.cpp
    source/mace_nonlinear_schema.cpp
    source/multilayer_perceptron.cpp
    source/multivariate_polynomial.cpp
    source/spherical_harmonic.cpp
    source/tools.cpp
    source/zbl.cpp)

list(APPEND SYMMETRIX_SOURCE_FILES
    source/jit_mh1_cuda_plugin.cpp
    source/jit_mh1_host_plugin.cpp
    source/jit_cuda_plugin.cpp
    source/jit_hip_plugin.cpp
    source/jit_hiprtc.cpp
    source/jit_nvrtc.cpp
    source/jit_host_plugin.cpp
    source/jit_m0_host_plugin.cpp)
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
        source/device_backend.cpp
        source/mace_kokkos_first_interaction.cpp
        source/mace_kokkos_response.cpp
        source/mace_kokkos_factorized_execution.cpp
        source/mace_kokkos_h1_phi1.cpp
        source/mace_kokkos_model.cpp
        source/neighbor_graph_kokkos.cpp
        source/mace_kokkos_factorized_lifecycle.cpp
        source/mace_kokkos_second_interaction.cpp
        source/mace_kokkos_factorized_analysis.cpp
        source/mace_kokkos_runtime.cpp
        source/mace_kokkos_evaluate.cpp
        source/jit_operator_module.cpp
        source/mace_nonlinear_kokkos_components.cpp
        source/mace_nonlinear_kokkos_runtime.cpp
        source/mace_nonlinear_kokkos_evaluate.cpp
        source/multilayer_perceptron_kokkos.cpp
        source/multivariate_polynomial_kokkos.cpp
        source/radial_function_set_kokkos.cpp
        source/tools_kokkos.cpp
        source/zbl_kokkos.cpp)
endif()

add_library(symmetrix ${SYMMETRIX_SOURCE_FILES})

# HIP compilation warns that global fast-math invalidates NaN/Inf checks used
# by model validation. Keep the historical host/CUDA policy while HIP is
# qualified without those assumptions.
if(SYMMETRIX_FAST_MATH AND NOT Kokkos_ENABLE_HIP)
    target_compile_options(symmetrix
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
endif()
if(_symmetrix_host_arch_flag
        AND NOT _symmetrix_host_arch_applied_globally
        AND NOT Kokkos_ENABLE_CUDA
        AND NOT Kokkos_ENABLE_HIP)
    target_compile_options(symmetrix PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${_symmetrix_host_arch_flag}>)
endif()

target_include_directories(symmetrix
    PUBLIC  source
            "${CMAKE_CURRENT_SOURCE_DIR}/../symmetrix/source/symmetrix"
            external/json/single_include
    PRIVATE "${SYMMETRIX_SPHERICART_SOURCE_DIR}/sphericart/include")
if(SYMMETRIX_BLAS_INCLUDE_DIR)
    target_include_directories(symmetrix PRIVATE "${SYMMETRIX_BLAS_INCLUDE_DIR}")
endif()

target_link_libraries(symmetrix
    PRIVATE ${BLAS_LIBRARIES}
            sphericart)
if(UNIX)
    target_link_libraries(symmetrix PRIVATE ${CMAKE_DL_LIBS})
endif()
if(SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix
        PRIVATE Kokkos::kokkos
                Kokkos::kokkoskernels)
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
