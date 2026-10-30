// The array contract every private FourierD3 binding seam shares.
//
// These helpers exist so that a caller mistake fails with an error that names the
// argument and the expected shape, dtype, or value, instead of being reinterpreted
// by pybind11's implicit conversions.  The contract is:
//
// * geometry and coefficient buffers hold IEEE float64 (the Python correction
//   runtime always passes float64, so a narrower buffer is refused rather than
//   silently widened - a silent widening would hide a caller that thinks it is
//   working in another precision);
// * index buffers hold integers of any width, and every value is checked against
//   the int32 range before narrowing, because 2**32 would wrap to a *valid* slot;
// * masks hold booleans;
// * every buffer is C-contiguous, and its rank is checked by the caller that
//   knows the expected shape.
//
// Nothing here is a public capability; the header is included by the private
// binding translation units only.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string>
#include <string_view>
#include <type_traits>

namespace py = pybind11;

namespace symmetrix::fourier_d3::binding {

using index_array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using float_array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using mask_array = py::array_t<bool, py::array::c_style | py::array::forcecast>;

[[noreturn]] inline void reject_buffer(
    const std::string_view name, const py::dtype& dtype, const char* expected)
{
    throw std::invalid_argument(
        "FourierD3 " + std::string(name) + " must hold " + expected + "; got dtype "
        + std::string(py::str(dtype.attr("name"))) + ".");
}

// The shift buffer is an index buffer, but a Python list of small integers arrives
// as int64 and a numpy uint64 array would wrap if it were converted straight to
// int32: 2**32 becomes slot 0, which is a *valid* image.  Every value is therefore
// checked against the int32 range before narrowing.
template <typename Wide>
[[nodiscard]] inline index_array narrow_index_buffer(
    const py::array& array, const std::string_view name)
{
    auto wide = py::array_t<Wide, py::array::c_style | py::array::forcecast>::ensure(array);
    if (!wide) {
        reject_buffer(name, array.dtype(), "integer values representable as int32");
    }
    const auto count = wide.size();
    index_array narrowed(static_cast<py::ssize_t>(count));
    const auto* source = wide.data();
    auto* target = narrowed.mutable_data();
    for (py::ssize_t index = 0; index < count; ++index) {
        const Wide value = source[index];
        bool outside = value > static_cast<Wide>(std::numeric_limits<std::int32_t>::max());
        if constexpr (std::is_signed_v<Wide>) {
            outside = outside
                || value < static_cast<Wide>(std::numeric_limits<std::int32_t>::min());
        }
        if (outside) {
            throw std::invalid_argument(
                "FourierD3 " + std::string(name) + " holds " + std::to_string(value)
                + " at index " + std::to_string(index)
                + ", which is outside the int32 range the kernels address with.");
        }
        target[index] = static_cast<std::int32_t>(value);
    }
    return narrowed;
}

[[nodiscard]] inline index_array index_buffer(
    const py::array& array, const std::string_view name)
{
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'i' && dtype.kind() != 'u') {
        reject_buffer(name, dtype, "integers");
    }
    if (dtype.kind() == 'u') {
        return narrow_index_buffer<std::uint64_t>(array, name);
    }
    return narrow_index_buffer<std::int64_t>(array, name);
}

[[nodiscard]] inline float_array float_buffer(
    const py::array& array, const std::string_view name)
{
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'f' || dtype.itemsize() != 8) {
        reject_buffer(name, dtype, "IEEE float64 values");
    }
    auto converted = float_array::ensure(array);
    if (!converted) {
        reject_buffer(name, dtype, "IEEE float64 values");
    }
    return converted;
}

[[nodiscard]] inline mask_array mask_buffer(
    const py::array& array, const std::string_view name)
{
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'b') {
        reject_buffer(name, dtype, "boolean values");
    }
    auto converted = mask_array::ensure(array);
    if (!converted) {
        reject_buffer(name, dtype, "boolean values");
    }
    return converted;
}

[[nodiscard]] inline std::string shape_text(const py::array& array)
{
    std::string text = "(";
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
        if (axis > 0) {
            text += ", ";
        }
        text += std::to_string(array.shape(axis));
    }
    text += ")";
    return text;
}

// A rank check with a named error, used by the seams whose arguments are vectors
// rather than matrices.
inline void require_rank(const py::array& array, const std::string_view name, const int rank)
{
    if (array.ndim() != rank) {
        throw std::invalid_argument(
            "FourierD3 " + std::string(name) + " must have rank " + std::to_string(rank)
            + ", got rank " + std::to_string(array.ndim()) + " with shape "
            + shape_text(array) + ".");
    }
}

// The row count of a two-dimensional buffer, with a named error when the second
// axis does not hold the expected number of columns.
inline void require_columns(
    const py::array& array, const std::string_view name, const py::ssize_t columns)
{
    require_rank(array, name, 2);
    if (array.shape(1) != columns) {
        throw std::invalid_argument(
            "FourierD3 " + std::string(name) + " must hold " + std::to_string(columns)
            + " value(s) per row; got shape "
            + shape_text(array) + ".");
    }
}

} // namespace symmetrix::fourier_d3::binding
