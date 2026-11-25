#pragma once

#include <pybind11/numpy.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
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

// The first `extent` values of a contiguous 1-D view as a new NumPy array of
// `shape`, filled by one deep copy from the view's memory space: no host
// mirror, no element loop and no intermediate vector.  Every call returns a
// new array, so results a caller already holds never change.
template <typename ViewType>
static auto view_prefix_array(
    const ViewType& view,
    const std::size_t extent,
    const std::initializer_list<py::ssize_t> shape)
{
    static_assert(ViewType::rank == 1, "A NumPy export reads a 1-D view.");
    using T = typename ViewType::non_const_value_type;
    if (view.extent(0) < extent)
        throw std::logic_error("A NumPy export view is too small.");
    std::size_t expected_size = 1;
    for (const py::ssize_t shape_extent : shape) {
        if (shape_extent < 0)
            throw std::invalid_argument("A NumPy export extent is negative.");
        expected_size *= static_cast<std::size_t>(shape_extent);
    }
    if (expected_size != extent)
        throw std::logic_error("A NumPy export shape is inconsistent.");
    py::array_t<T> result(shape);
    const Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host(
        result.mutable_data(), extent);
    Kokkos::deep_copy(
        host, Kokkos::subview(view, std::make_pair(std::size_t(0), extent)));
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
