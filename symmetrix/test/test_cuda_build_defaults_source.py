from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cpu_defaults_native_and_portable_override_disables_kokkos_native():
    package_source = (ROOT / "symmetrix/CMakeLists.txt").read_text()
    core_source = (ROOT / "libsymmetrix/CMakeLists.txt").read_text()

    default = 'set(SYMMETRIX_HOST_ARCH "native" CACHE STRING'
    assert default in package_source
    assert default in core_source
    assert 'if(SYMMETRIX_HOST_ARCH STREQUAL "native")' in package_source
    assert 'set(Kokkos_ARCH_NATIVE ON CACHE BOOL "Set Kokkos_ARCH_NATIVE.")' in (
        package_source
    )
    assert (
        'if(NOT SYMMETRIX_HOST_ARCH STREQUAL "native" AND Kokkos_ARCH_NATIVE)'
        in package_source
    )
    assert "Kokkos_ARCH_NATIVE=OFF for a consistent CPU target." in package_source
    assert "add_compile_options(" in package_source
    assert "$<$<COMPILE_LANGUAGE:CXX>:-march=${SYMMETRIX_HOST_ARCH}>" in (
        package_source
    )
    assert 'set(_symmetrix_host_arch_flag "-march=${SYMMETRIX_HOST_ARCH}")' in (
        core_source
    )
    assert "AND NOT _symmetrix_host_arch_applied_globally" in core_source
    assert 'set(SYMMETRIX_DEVICE_BACKEND "NONE" CACHE STRING' in package_source
    assert 'set(SYMMETRIX_PYTHON_MODULE_NAME "" CACHE STRING' in package_source
    assert 'set(SYMMETRIX_PYTHON_MODULE_NAME "_native_cpu")' in package_source

    pyproject = (ROOT / "symmetrix/pyproject.toml").read_text()
    assert "[tool.scikit-build.cmake.define]" in pyproject
    assert 'SYMMETRIX_DEVICE_BACKEND = "NONE"' in pyproject
    assert 'Kokkos_ENABLE_CUDA = "OFF"' in pyproject
    assert 'Kokkos_ENABLE_OPENMP = "ON"' in pyproject


def test_explicit_kokkos_cuda_defaults_sphericart_cuda_on():
    source = (ROOT / "symmetrix/CMakeLists.txt").read_text()
    expected = """if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_CUDA
    AND NOT DEFINED SYMMETRIX_SPHERICART_CUDA)
  set(SYMMETRIX_SPHERICART_CUDA ON CACHE BOOL \"Enable SpheriCart-CUDA.\")
endif()"""

    assert expected in source
    assert (
        """if(NOT SYMMETRIX_KOKKOS)
  set(Kokkos_ENABLE_CUDA OFF)
  set(Kokkos_ENABLE_HIP OFF)
  set(SYMMETRIX_SPHERICART_CUDA OFF)"""
        in source
    )


def test_embedded_kokkos_cuda_defaults_sphericart_cuda_on():
    source = (ROOT / "libsymmetrix/CMakeLists.txt").read_text()

    assert (
        """set(_symmetrix_sphericart_cuda_default OFF)
if(Kokkos_ENABLE_CUDA)
    set(_symmetrix_sphericart_cuda_default ON)
endif()
option(SYMMETRIX_SPHERICART_CUDA "Enable SpheriCart-CUDA."
    ${_symmetrix_sphericart_cuda_default})"""
        in source
    )
    assert (
        """if(Kokkos_ENABLE_CUDA AND NOT SYMMETRIX_SPHERICART_CUDA)
    message(WARNING"""
        in source
    )


def test_gpu_backends_never_apply_native_host_architecture():
    package_source = (ROOT / "symmetrix/CMakeLists.txt").read_text()
    core_source = (ROOT / "libsymmetrix/CMakeLists.txt").read_text()

    assert (
        """if(_symmetrix_resolved_device_backend STREQUAL "NONE"
    AND NOT SYMMETRIX_HOST_ARCH STREQUAL ""
    AND NOT SYMMETRIX_HOST_ARCH STREQUAL "none"
    AND NOT SYMMETRIX_HOST_ARCH STREQUAL "native")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-march=${SYMMETRIX_HOST_ARCH}>)"""
        in package_source
    )

    assert (
        """if(NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP)
    if(NOT DEFINED Kokkos_ARCH_NATIVE)"""
        in core_source
    )
    assert (
        """if(_symmetrix_host_arch_flag
            AND NOT SYMMETRIX_HOST_ARCH STREQUAL "native"
            AND NOT _symmetrix_host_arch_applied_globally)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:${_symmetrix_host_arch_flag}>)"""
        in core_source
    )

    guarded_target_flags = """if(_symmetrix_host_arch_flag
        AND NOT _symmetrix_host_arch_applied_globally
        AND NOT Kokkos_ENABLE_CUDA
        AND NOT Kokkos_ENABLE_HIP)"""
    assert core_source.count(guarded_target_flags) == 2

    assert (
        'set(Kokkos_ARCH_NATIVE OFF CACHE BOOL "Set Kokkos_ARCH_NATIVE." FORCE)'
        in package_source
    )
    assert (
        """set(SPHERICART_ARCH_NATIVE OFF CACHE BOOL
    "Try to use -march=native when compiling SpheriCart." FORCE)"""
        in core_source
    )
