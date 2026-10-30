#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "build_metadata.hpp"
#include "model_metadata.hpp"
#include "jit_generation_version.hpp"
#include "prebuilt_operator_build_info.hpp"

namespace py = pybind11;

void bind_cubic_spline(py::module_ &m);
void bind_cubic_spline_set(py::module_ &m);
void bind_e3nn(py::module_ &m);
void bind_nequip_schema(py::module_ &m);
void bind_multilayer_perceptron(py::module_ &m);
void bind_multivariate_polynomial(py::module_ &m);
void bind_spherical_harmonic(py::module_ &m);
void bind_jit_nvrtc(py::module_ &m);
void bind_tools(py::module_ &m);
void bind_zbl(py::module_ &m);

#ifdef SYMMETRIX_KOKKOS
// Defined in tools_kokkos.cpp, which is the translation unit that actually
// compiles against the vendored Kokkos headers.
std::string _kokkos_build_version();
void bind_cubic_spline_kokkos(py::module_ &m);
void bind_cubic_spline_set_kokkos(py::module_ &m);
void bind_long_range_kokkos(py::module_ &m);
void bind_mace_polar_program(py::module_ &m);
void bind_dispersion_kokkos(py::module_ &m);
void bind_les_monopole(py::module_ &m);
void bind_mace_kokkos(py::module_ &m);
void bind_mace_les_kokkos(py::module_& m);
void bind_mace_nonlinear_kokkos(py::module_ &m);
void bind_nequip_oam_kokkos(py::module_& m);
void bind_multilayer_perceptron_kokkos(py::module_ &m);
void bind_multivariate_polynomial_kokkos(py::module_ &m);
//void bind_spherical_harmonic(py::module_ &m);
void bind_tools_kokkos(py::module_ &m);
void bind_zbl_kokkos(py::module_ &m);

#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
void bind_particle_mesh(py::module_& m);
#endif
#if defined(SYMMETRIX_FOURIER_D3_AVAILABLE) && SYMMETRIX_FOURIER_D3_AVAILABLE
// Private storage, cache-key, and plan-owner seam for the reciprocal-space
// FourierD3 correction.  It evaluates nothing and publishes no evaluator.
void bind_fourier_d3_storage(py::module_ &m);
// Private modified-coordination and separable-coefficient seam.  It evaluates no
// energy or force and publishes no evaluator.
void bind_fourier_d3_coordination(py::module_ &m);
// Private particle-mesh forward-path seam.  It evaluates no energy or force on
// its own and publishes no evaluator.
void bind_fourier_d3_reciprocal(py::module_ &m);
// Private analytic reverse-path seam.  It owns the analytic reverse chain -
// the coefficient sensitivities, the analytic gather forces, the coordination
// adjoint, the affine virial and Voigt stress, and the atomic-energy partition -
// and publishes no evaluator.
void bind_fourier_d3_reverse(py::module_ &m);

// The dtype-specific evaluator classes the Python correction runtime resolves
// (``FourierD3Kokkos``/``FourierD3KokkosFloat``) and the canonical-table accessor
// they are built from. These stay private behind the public Python configuration.
void bind_fourier_d3_evaluator(py::module_ &m);
#endif
#endif

namespace
{
// Particle-mesh algorithms share one configure-time KokkosFFT provider.
py::dict particle_mesh_capability()
{
    py::dict capability;
    capability["available"] = bool(SYMMETRIX_PARTICLE_MESH_AVAILABLE);
    capability["provider"] = SYMMETRIX_PARTICLE_MESH_PROVIDER;
    capability["provider_variant"] = SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT;
    capability["kokkosfft_version"] = SYMMETRIX_PARTICLE_MESH_KOKKOSFFT_VERSION;
    capability["reason"] = SYMMETRIX_PARTICLE_MESH_REASON;
    return capability;
}

// The reciprocal-space FourierD3 correction needs a KokkosFFT FFT provider.
// Whether this build discovered one is a configure-time decision, so it is
// reported verbatim and never inferred or emulated at runtime.  The Kokkos
// version and the execution backend are reported the same way: the version comes
// from the headers this extension compiled against, and the backend from the
// build's own selector, so neither can name a runtime the binary does not use.
py::dict fourier_d3_capability()
{
    py::dict capability;
    capability["available"] = bool(SYMMETRIX_FOURIER_D3_AVAILABLE);
    capability["provider"] = SYMMETRIX_FOURIER_D3_PROVIDER;
    capability["provider_variant"] = SYMMETRIX_FOURIER_D3_PROVIDER_VARIANT;
    capability["kokkosfft_version"] = SYMMETRIX_FOURIER_D3_KOKKOSFFT_VERSION;
#ifdef SYMMETRIX_KOKKOS
    capability["kokkos_version"] = _kokkos_build_version();
#else
    capability["kokkos_version"] = "";
#endif
    capability["execution_backend"] = SYMMETRIX_BUILD_BACKEND;
    capability["reason"] = SYMMETRIX_FOURIER_D3_REASON;
    return capability;
}
}  // namespace

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
        result["source_content_sha256"] = SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256;
        result["native_source_content_sha256"] =
            SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256;
        result["compiler_id"] = SYMMETRIX_BUILD_COMPILER_ID;
        result["compiler_version"] = SYMMETRIX_BUILD_COMPILER_VERSION;
        result["device_compiler_id"] = SYMMETRIX_BUILD_DEVICE_COMPILER_ID;
        result["device_compiler_version"] = SYMMETRIX_BUILD_DEVICE_COMPILER_VERSION;
        result["particle_mesh"] = particle_mesh_capability();
        result["fast_math"] = bool(SYMMETRIX_BUILD_FAST_MATH);
        py::dict prebuilt;
        prebuilt["mode"] = symmetrix::build::prebuilt_operator_mode;
        prebuilt["backend"] = symmetrix::build::prebuilt_operator_backend;
        prebuilt["target"] = symmetrix::build::prebuilt_operator_target;
        prebuilt["catalog_relative_path"] =
            symmetrix::build::prebuilt_operator_catalog_relative_path;
        prebuilt["catalog_sha256"] =
            symmetrix::build::prebuilt_operator_catalog_sha256;
        prebuilt["artifact_count"] =
            symmetrix::build::prebuilt_operator_artifact_count;
        prebuilt["installed_bytes"] =
            symmetrix::build::prebuilt_operator_installed_bytes;
        prebuilt["source_content_sha256"] =
            symmetrix::build::prebuilt_operator_source_content_sha256;
        prebuilt["native_source_content_sha256"] =
            symmetrix::build::prebuilt_operator_native_source_content_sha256;
        prebuilt["compiler_id"] =
            symmetrix::build::prebuilt_operator_compiler_id;
        prebuilt["compiler_version"] =
            symmetrix::build::prebuilt_operator_compiler_version;
        prebuilt["build_duration_seconds"] =
            symmetrix::build::prebuilt_operator_build_duration_seconds;
        result["prebuilt_operator_artifacts"] = prebuilt;
        result["fourier_d3"] = fourier_d3_capability();
        return result;
    });
    m.def("_particle_mesh_capability", &particle_mesh_capability);
    m.def("_fourier_d3_capability", &fourier_d3_capability);

    bind_cubic_spline(m);
    bind_cubic_spline_set(m);
    bind_e3nn(m);
    bind_nequip_schema(m);
    bind_multilayer_perceptron(m);
    bind_multivariate_polynomial(m);
    bind_spherical_harmonic(m);
    bind_jit_nvrtc(m);
    bind_tools(m);
    bind_zbl(m);

#ifdef SYMMETRIX_KOKKOS
    bind_cubic_spline_kokkos(m);
    bind_cubic_spline_set_kokkos(m);
    bind_long_range_kokkos(m);
    bind_mace_polar_program(m);
    bind_dispersion_kokkos(m);
    bind_les_monopole(m);
    bind_mace_kokkos(m);
    bind_mace_les_kokkos(m);
    bind_mace_nonlinear_kokkos(m);
    bind_nequip_oam_kokkos(m);
    bind_multilayer_perceptron_kokkos(m);
    bind_multivariate_polynomial_kokkos(m);
    //bind_spherical_harmonic_kokkos(m);
    bind_tools_kokkos(m);
    bind_zbl_kokkos(m);
#if SYMMETRIX_PARTICLE_MESH_AVAILABLE
    bind_particle_mesh(m);
#endif
#if SYMMETRIX_FOURIER_D3_AVAILABLE
    bind_fourier_d3_storage(m);
    bind_fourier_d3_coordination(m);
    bind_fourier_d3_reciprocal(m);
    bind_fourier_d3_reverse(m);
    bind_fourier_d3_evaluator(m);
#endif
#endif
}
