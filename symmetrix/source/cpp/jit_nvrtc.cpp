#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "jit_nvrtc.hpp"
#include "jit_hiprtc.hpp"

namespace py = pybind11;

void bind_jit_nvrtc(py::module_& module)
{
    module.def("_jit_nvrtc_information", [] {
        const auto information = symmetrix::execution::nvrtc_information();
        py::dict result;
        result["available"] = information.available;
        result["library"] = information.library;
        result["reason"] = information.reason;
        result["major"] = information.major;
        result["minor"] = information.minor;
        result["supported_architectures"] = information.supported_architectures;
        return result;
    });
    module.def(
        "_execution_compile_cuda_with_nvrtc",
        [] (const std::string& source, const std::vector<std::string>& options) {
            const auto compilation =
                symmetrix::execution::compile_cuda_with_nvrtc(source, options);
            py::dict result;
            result["cubin"] = py::bytes(
                reinterpret_cast<const char*>(compilation.cubin.data()),
                compilation.cubin.size());
            result["log"] = compilation.log;
            result["major"] = compilation.major;
            result["minor"] = compilation.minor;
            return result;
        },
        py::arg("source"), py::arg("options"));
    module.def("_jit_hiprtc_information", [] {
        const auto information = symmetrix::execution::hiprtc_information();
        py::dict result;
        result["available"] = information.available;
        result["library"] = information.library;
        result["reason"] = information.reason;
        result["major"] = information.major;
        result["minor"] = information.minor;
        return result;
    });
    module.def(
        "_execution_compile_hip_with_hiprtc",
        [] (const std::string& source, const std::vector<std::string>& options) {
            const auto compilation =
                symmetrix::execution::compile_hip_with_hiprtc(source, options);
            py::dict result;
            result["code"] = py::bytes(
                reinterpret_cast<const char*>(compilation.code.data()),
                compilation.code.size());
            result["log"] = compilation.log;
            result["major"] = compilation.major;
            result["minor"] = compilation.minor;
            return result;
        },
        py::arg("source"), py::arg("options"));
}
