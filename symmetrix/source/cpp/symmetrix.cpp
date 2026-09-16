#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "model_metadata.hpp"
#include "jit_generation_version.hpp"

namespace py = pybind11;

void bind_cubic_spline(py::module_ &m);
void bind_cubic_spline_set(py::module_ &m);
void bind_e3nn(py::module_ &m);
void bind_mace(py::module_ &m);
void bind_mace_nonlinear(py::module_ &m);
void bind_multilayer_perceptron(py::module_ &m);
void bind_multivariate_polynomial(py::module_ &m);
void bind_spherical_harmonic(py::module_ &m);
void bind_jit_nvrtc(py::module_ &m);
void bind_tools(py::module_ &m);
void bind_zbl(py::module_ &m);

#ifdef SYMMETRIX_KOKKOS
void bind_cubic_spline_kokkos(py::module_ &m);
void bind_cubic_spline_set_kokkos(py::module_ &m);
void bind_mace_kokkos(py::module_ &m);
void bind_mace_nonlinear_kokkos(py::module_ &m);
void bind_multilayer_perceptron_kokkos(py::module_ &m);
void bind_multivariate_polynomial_kokkos(py::module_ &m);
//void bind_spherical_harmonic(py::module_ &m);
void bind_tools_kokkos(py::module_ &m);
void bind_zbl_kokkos(py::module_ &m);
#endif

#ifndef SYMMETRIX_PYTHON_MODULE
#define SYMMETRIX_PYTHON_MODULE symmetrix
#endif
#ifndef SYMMETRIX_NATIVE_ABI
#define SYMMETRIX_NATIVE_ABI 1
#endif
#ifndef SYMMETRIX_BUILD_BACKEND
#define SYMMETRIX_BUILD_BACKEND "unknown"
#endif
#ifndef SYMMETRIX_BUILD_ARCHITECTURE
#define SYMMETRIX_BUILD_ARCHITECTURE "unknown"
#endif
#ifndef SYMMETRIX_BUILD_DISTRIBUTION
#define SYMMETRIX_BUILD_DISTRIBUTION "symmetrix"
#endif
#ifndef SYMMETRIX_BUILD_SOURCE_COMMIT
#define SYMMETRIX_BUILD_SOURCE_COMMIT "unknown"
#endif
#ifndef SYMMETRIX_BUILD_COMPILER_ID
#define SYMMETRIX_BUILD_COMPILER_ID "unknown"
#endif
#ifndef SYMMETRIX_BUILD_COMPILER_VERSION
#define SYMMETRIX_BUILD_COMPILER_VERSION "unknown"
#endif
#ifndef SYMMETRIX_BUILD_SOURCE_DIRTY
#define SYMMETRIX_BUILD_SOURCE_DIRTY 0
#endif
#ifndef SYMMETRIX_BUILD_DEVICE_COMPILER_ID
#define SYMMETRIX_BUILD_DEVICE_COMPILER_ID ""
#endif
#ifndef SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION
#define SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION ""
#endif

PYBIND11_MODULE(SYMMETRIX_PYTHON_MODULE, m)
{
    m.doc() = "symmetrix";
    m.def("_model_metadata", &symmetrix_model_metadata);
    m.def("_required_jit_generation_version", [] {
        return symmetrix::execution::required_jit_generation_version;
    });
    m.def("_backend_build_info", [] {
        py::dict result;
        result["native_abi"] = SYMMETRIX_NATIVE_ABI;
        result["backend"] = SYMMETRIX_BUILD_BACKEND;
        result["architecture"] = SYMMETRIX_BUILD_ARCHITECTURE;
        result["distribution"] = SYMMETRIX_BUILD_DISTRIBUTION;
        result["source_commit"] = SYMMETRIX_BUILD_SOURCE_COMMIT;
        result["source_dirty"] = bool(SYMMETRIX_BUILD_SOURCE_DIRTY);
        result["compiler_id"] = SYMMETRIX_BUILD_COMPILER_ID;
        result["compiler_version"] = SYMMETRIX_BUILD_COMPILER_VERSION;
        result["device_compiler_id"] = SYMMETRIX_BUILD_DEVICE_COMPILER_ID;
        result["device_compiler_version"] = SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION;
        return result;
    });

    bind_cubic_spline(m);
    bind_cubic_spline_set(m);
    bind_e3nn(m);
    bind_mace(m);
    bind_mace_nonlinear(m);
    bind_multilayer_perceptron(m);
    bind_multivariate_polynomial(m);
    bind_spherical_harmonic(m);
    bind_jit_nvrtc(m);
    bind_tools(m);
    bind_zbl(m);

#ifdef SYMMETRIX_KOKKOS
    bind_cubic_spline_kokkos(m);
    bind_cubic_spline_set_kokkos(m);
    bind_mace_kokkos(m);
    bind_mace_nonlinear_kokkos(m);
    bind_multilayer_perceptron_kokkos(m);
    bind_multivariate_polynomial_kokkos(m);
    //bind_spherical_harmonic_kokkos(m);
    bind_tools_kokkos(m);
    bind_zbl_kokkos(m);
#endif
}
