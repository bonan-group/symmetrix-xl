cmake_minimum_required(VERSION 3.27)

# The Metal runtime layer is configurable on its own so that it can be built
# and qualified without Kokkos, BLAS, or the Python extension.
if(PROJECT_IS_TOP_LEVEL OR NOT DEFINED PROJECT_NAME)
    project(SymmetrixMetal LANGUAGES CXX OBJCXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    enable_language(OBJCXX)
endif()

if(NOT APPLE)
    message(FATAL_ERROR "The Symmetrix Metal runtime requires an Apple platform.")
endif()

option(SYMMETRIX_METAL_SELFTEST "Build the Metal runtime self-test executable." ON)

# Metal kernels compile SpheriCart's hardcoded spherical-harmonic macros at
# runtime; embed the macro block so they share the host polynomial source.
if(DEFINED SYMMETRIX_SPHERICART_SOURCE_DIR)
    set(_symmetrix_metal_sphericart "${SYMMETRIX_SPHERICART_SOURCE_DIR}")
else()
    set(_symmetrix_metal_sphericart
        "${CMAKE_CURRENT_SOURCE_DIR}/../../external/sphericart")
endif()
set(_symmetrix_metal_sphericart_macros
    "${_symmetrix_metal_sphericart}/sphericart/include/macros.hpp")
file(READ "${_symmetrix_metal_sphericart_macros}" _symmetrix_metal_macros_text)
string(FIND "${_symmetrix_metal_macros_text}" "#define COMPUTE_SPH_L0"
    _symmetrix_metal_macros_begin)
string(FIND "${_symmetrix_metal_macros_text}" "#endif"
    _symmetrix_metal_macros_end REVERSE)
if(_symmetrix_metal_macros_begin EQUAL -1
        OR _symmetrix_metal_macros_end LESS _symmetrix_metal_macros_begin)
    message(FATAL_ERROR
        "Unrecognized SpheriCart macro layout in ${_symmetrix_metal_sphericart_macros}")
endif()
math(EXPR _symmetrix_metal_macros_length
    "${_symmetrix_metal_macros_end}-${_symmetrix_metal_macros_begin}")
string(SUBSTRING "${_symmetrix_metal_macros_text}" ${_symmetrix_metal_macros_begin}
    ${_symmetrix_metal_macros_length} SYMMETRIX_METAL_SPHERICART_MACROS)
configure_file(metal_sphericart_macros.hpp.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_sphericart_macros.hpp" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_symmetrix_metal_sphericart_macros}")

add_library(symmetrix_metal STATIC
    metal_runtime.mm metal_m0_module.cpp metal_r0_module.cpp metal_r1_module.cpp)
target_include_directories(symmetrix_metal
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
set_target_properties(symmetrix_metal PROPERTIES
    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(symmetrix_metal PRIVATE -fobjc-arc)
target_include_directories(symmetrix_metal
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
           "${CMAKE_CURRENT_SOURCE_DIR}/../../../symmetrix/source/symmetrix")
target_link_libraries(symmetrix_metal PUBLIC
    "-framework Metal"
    "-framework Foundation")
target_compile_definitions(symmetrix_metal PUBLIC SYMMETRIX_ENABLE_METAL)

if(SYMMETRIX_METAL_SELFTEST)
    add_executable(symmetrix_metal_selftest metal_runtime_selftest.cpp)
    target_link_libraries(symmetrix_metal_selftest PRIVATE symmetrix_metal)

    add_executable(symmetrix_metal_cpu_microbenchmark
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../benchmarks/metal_cpu_microbenchmark.cpp")
    target_compile_options(symmetrix_metal_cpu_microbenchmark PRIVATE
        -O3 -mcpu=native)
    target_compile_definitions(symmetrix_metal_cpu_microbenchmark PRIVATE
        ACCELERATE_NEW_LAPACK)
    target_link_libraries(symmetrix_metal_cpu_microbenchmark PRIVATE
        symmetrix_metal "-framework Accelerate")
endif()
