#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cubic_spline_set.hpp"

namespace py = pybind11;

void bind_cubic_spline_set(py::module_ &m)
{
    py::class_<CubicSplineSet>(m, "CubicSplineSet")
        .def(
            py::init<
                double,
                std::vector<std::vector<double>>,
                std::vector<std::vector<double>>,
                double>(),
            py::arg("h"),
            py::arg("nodal_values"),
            py::arg("nodal_derivs"),
            py::arg("x0") = 0.0)
        .def("evaluate",
            [](CubicSplineSet& self, double r, py::array_t<double> values_numpy) {
                auto values = std::span<double>(values_numpy.mutable_data(), values_numpy.size());
                self.evaluate(r, values);
            })
        .def("evaluate_derivs",
            [](CubicSplineSet& self, double r, py::array_t<double> values_numpy, py::array_t<double> derivs_numpy) {
               auto values = std::span<double>(values_numpy.mutable_data(), values_numpy.size());
               auto derivs = std::span<double>(derivs_numpy.mutable_data(), derivs_numpy.size());
               self.evaluate_derivs(r, values, derivs);
            })
        .def("evaluate_function",
            [](const CubicSplineSet& self, double r, int function) {
                return self.evaluate_function(self.evaluation_point(r), function);
            })
        .def("evaluate_function_derivs",
            [](const CubicSplineSet& self, double r, int function) {
                double value = 0.0;
                double derivative = 0.0;
                self.evaluate_function_derivs(
                    self.evaluation_point(r), function, value, derivative);
                return py::make_tuple(value, derivative);
            });
}
