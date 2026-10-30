#pragma once

#include <pybind11/numpy.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace symmetrix::python::detail {

namespace py = pybind11;

template <typename T>
static py::array_t<T> shaped_array(
    const std::vector<T>& values,
    const std::initializer_list<py::ssize_t> shape)
{
    std::size_t expected_size = 1;
    for (const py::ssize_t extent : shape) {
        if (extent < 0)
            throw std::invalid_argument("A NumPy export extent is negative.");
        expected_size *= static_cast<std::size_t>(extent);
    }
    if (expected_size != values.size())
        throw std::logic_error("A NumPy export shape is inconsistent.");
    py::array_t<T> result(shape);
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

template <typename ViewType>
static auto view_prefix_1d(const ViewType& view, const std::size_t extent)
{
    using T = typename ViewType::non_const_value_type;
    if (view.extent(0) < extent)
        throw std::logic_error("A benchmark export view is too small.");
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view);
    auto result = std::vector<T>(extent);
    for (std::size_t i=0; i<extent; ++i)
        result[i] = host(i);
    return result;
}

template <typename ViewType>
static auto view_prefix_3d(
    const ViewType& view,
    const std::size_t extent_0,
    const std::size_t extent_1,
    const std::size_t extent_2)
{
    using T = typename ViewType::non_const_value_type;
    if (view.extent(0) < extent_0
        || view.extent(1) != extent_1
        || view.extent(2) != extent_2)
        throw std::logic_error("A benchmark export tensor extent is invalid.");
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    auto result = std::vector<T>(extent_0*extent_1*extent_2);
    for (std::size_t i=0; i<extent_0; ++i)
        for (std::size_t j=0; j<extent_1; ++j)
            for (std::size_t k=0; k<extent_2; ++k)
                result[(i*extent_1+j)*extent_2+k] = host(i,j,k);
    return result;
}

}  // namespace symmetrix::python::detail
