// Private native seam over the FourierD3 modified coordination-number and
// separable-coefficient kernels.
//
// These bindings are internal infrastructure: they exist so the model contract
// (the toolkit steepness law, the exact cutoff mask, receiver-major edge
// accumulation, masked and exponent-shifted reference weights with a stable
// fallback, rank-chunk-bounded coefficient storage, and fail-loud validation) is
// testable in the qualified build.  They are not package exports, they evaluate
// no energy and no force, and they publish no evaluator type. The public
// FP32/FP64 evaluators and diagnostic surface are bound separately.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fourier_d3_coordination.hpp"

namespace py = pybind11;

namespace {

using symmetrix::fourier_d3::CoordinationGeometry;
using symmetrix::fourier_d3::CoordinationUsage;
using symmetrix::fourier_d3::CoordinationWorkspace;

constexpr std::string_view k_float32 = "float32";
constexpr std::string_view k_float64 = "float64";

void require_known_precision(const std::string& precision)
{
    if (precision != k_float32 && precision != k_float64) {
        throw std::invalid_argument(
            "FourierD3 precision must be 'float32' or 'float64', got '" + precision + "'.");
    }
}

[[nodiscard]] py::dict geometry_report(const CoordinationGeometry& geometry)
{
    py::dict report;
    report["active_nodes"] = geometry.active_nodes;
    report["active_edges"] = geometry.active_edges;
    report["n_species"] = geometry.n_species;
    report["n_ref"] = geometry.n_ref;
    report["rank_total"] = geometry.rank_total;
    report["active_rank_chunk"] = geometry.active_rank_chunk;
    report["rank_chunk_bound"] = geometry.rank_chunk_bound;
    return report;
}

[[nodiscard]] py::dict usage_report(const CoordinationUsage& usage)
{
    py::dict report;
    py::list buffers;
    for (const symmetrix::fourier_d3::BufferUsage& buffer : usage.buffers) {
        py::dict entry;
        entry["name"] = buffer.name;
        entry["active_elements"] = buffer.active_elements;
        entry["capacity_elements"] = buffer.capacity_elements;
        entry["active_bytes"] = buffer.active_bytes;
        entry["capacity_bytes"] = buffer.capacity_bytes;
        entry["allocation_id"] = buffer.allocation_id;
        buffers.append(entry);
    }
    report["buffers"] = buffers;
    report["active_bytes"] = usage.active_bytes;
    report["capacity_bytes"] = usage.capacity_bytes;
    report["reservations"] = usage.reservations;
    report["allocations"] = usage.allocations;
    report["geometry"] = geometry_report(usage.geometry);
    return report;
}

// Input buffers.  The element kind is checked here and the layout is not: an
// index buffer must hold integers and a geometry buffer must hold IEEE float64,
// so a caller mistake (a coordinate array where an index array is expected, a
// float32 table where float64 is required) fails with a named error instead of
// being reinterpreted.  A buffer that already has the required dtype and
// contiguity is passed through without a copy.
using index_array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using float_array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using mask_array = py::array_t<bool, py::array::c_style | py::array::forcecast>;

[[noreturn]] void reject_buffer(const std::string_view name, const py::dtype& dtype, const char* expected)
{
    throw std::invalid_argument(
        "FourierD3 " + std::string(name) + " must hold " + expected + "; got dtype "
        + std::string(py::str(dtype.attr("name"))) + ".");
}

// Widens an integer buffer to `Wide`, checks every value against the int32 range
// the kernels address with, and narrows only the values that fit.
//
// The signed and unsigned widenings are separate on purpose: a uint64 at or above
// 2**63 becomes a *negative* int64, so a uint64 buffer converted straight to
// int64 would carry values the int32 check accepts and only the workspace's own
// range checks would reject, with a message about the wrong thing.
template <typename Wide>
[[nodiscard]] index_array narrow_index_buffer(
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

[[nodiscard]] index_array index_buffer(const py::array& array, const std::string_view name)
{
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'i' && dtype.kind() != 'u') {
        reject_buffer(name, dtype, "integers");
    }
    if (dtype.is(py::dtype::of<std::int32_t>())) {
        auto exact = index_array::ensure(array);
        if (exact) {
            return exact;
        }
    }
    // An integer buffer of any other width is widened to its own signedness first
    // and every value is checked against the int32 range.  Converting straight to
    // int32 would wrap: 2**32 becomes slot 0, which is a *valid* index, so the
    // graph would be accepted with a wrong neighbour instead of being rejected.
    if (dtype.kind() == 'u') {
        return narrow_index_buffer<std::uint64_t>(array, name);
    }
    return narrow_index_buffer<std::int64_t>(array, name);
}

[[nodiscard]] float_array float_buffer(const py::array& array, const std::string_view name)
{
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'f' || dtype.itemsize() != static_cast<py::ssize_t>(sizeof(double))) {
        reject_buffer(name, dtype, "IEEE float64 values");
    }
    auto converted = float_array::ensure(array);
    if (!converted) {
        reject_buffer(name, dtype, "IEEE float64 values");
    }
    return converted;
}

[[nodiscard]] mask_array mask_buffer(const py::array& array, const std::string_view name)
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

template <typename Precision>
[[nodiscard]] py::array_t<Precision> wrap(
    std::vector<Precision> values, const std::vector<py::ssize_t>& shape)
{
    py::ssize_t elements = 1;
    for (const py::ssize_t extent : shape) {
        elements *= extent;
    }
    if (elements != static_cast<py::ssize_t>(values.size())) {
        throw std::logic_error(
            "FourierD3 result shape does not match its element count; this is a "
            "binding defect.");
    }
    py::array_t<Precision> result(shape);
    if (!values.empty()) {
        std::memcpy(
            result.mutable_data(), values.data(), values.size() * sizeof(Precision));
    }
    return result;
}

// Owns one precision-specialized coordination workspace behind a stable Python
// name.  Arrays are accepted with their exact expected dtypes and are converted
// to host spans, so a mismatch fails loudly rather than being reinterpreted.
class CoordinationSeam {
public:
    CoordinationSeam(std::string precision, std::int64_t rank_chunk_bound)
        : precision_(std::move(precision))
    {
        require_known_precision(precision_);
        if (precision_ == k_float32) {
            assign<float>(rank_chunk_bound);
        } else {
            assign<double>(rank_chunk_bound);
        }
    }

    [[nodiscard]] const std::string& precision() const noexcept { return precision_; }

    py::dict reserve(
        std::int64_t nodes,
        std::int64_t edges,
        std::int64_t n_species,
        std::int64_t n_ref,
        std::int64_t rank_total,
        std::int64_t rank_chunk)
    {
        return dispatch_dict([&](auto& workspace) {
            const auto outcome =
                workspace.reserve(nodes, edges, n_species, n_ref, rank_total, rank_chunk);
            py::dict report;
            report["grown"] = outcome.grown;
            report["geometry"] = geometry_report(outcome.geometry);
            return report;
        });
    }

    void set_graph(
        const py::array& offsets,
        const py::array& sources,
        const py::array& distances,
        const py::array& numbers,
        const py::array& covalent_radii)
    {
        const auto offsets_buffer = index_buffer(offsets, "offsets");
        const auto sources_buffer = index_buffer(sources, "sources");
        const auto distances_buffer = float_buffer(distances, "distances");
        const auto numbers_buffer = index_buffer(numbers, "numbers");
        const auto radii_buffer = float_buffer(covalent_radii, "covalent_radii");
        dispatch_void([&](auto& workspace) {
            workspace.set_graph(
                std::span<const std::int32_t>(offsets_buffer.data(), offsets_buffer.size()),
                std::span<const std::int32_t>(sources_buffer.data(), sources_buffer.size()),
                std::span<const double>(distances_buffer.data(), distances_buffer.size()),
                std::span<const std::int32_t>(numbers_buffer.data(), numbers_buffer.size()),
                std::span<const double>(radii_buffer.data(), radii_buffer.size()));
        });
    }

    void set_reference(
        const py::array& cn_ref,
        const py::array& valid,
        const py::array& species,
        const py::array& v_q)
    {
        const auto cn_ref_buffer = float_buffer(cn_ref, "cn_ref");
        const auto valid_buffer = mask_buffer(valid, "valid");
        const auto species_buffer = index_buffer(species, "species");
        const auto v_q_buffer = float_buffer(v_q, "v_q");
        dispatch_void([&](auto& workspace) {
            workspace.set_reference(
                std::span<const double>(cn_ref_buffer.data(), cn_ref_buffer.size()),
                std::span<const bool>(valid_buffer.data(), valid_buffer.size()),
                std::span<const std::int32_t>(species_buffer.data(), species_buffer.size()),
                std::span<const double>(v_q_buffer.data(), v_q_buffer.size()));
        });
    }

    void set_coordination(const py::array& values)
    {
        const auto values_buffer = float_buffer(values, "coordination");
        dispatch_void([&](auto& workspace) {
            workspace.set_coordination(
                std::span<const double>(values_buffer.data(), values_buffer.size()));
        });
    }

    void run_coordination(double cutoff)
    {
        dispatch_void([&](auto& workspace) { workspace.run_coordination(cutoff); });
    }

    void run_coefficients(std::int64_t rank_begin)
    {
        dispatch_void([&](auto& workspace) { workspace.run_coefficients(rank_begin); });
    }

    [[nodiscard]] py::dict usage() const
    {
        return dispatch_const(
            [&](const auto& workspace) { return usage_report(workspace.usage()); });
    }

    [[nodiscard]] py::dict geometry() const
    {
        return dispatch_const(
            [&](const auto& workspace) { return geometry_report(workspace.geometry()); });
    }

    [[nodiscard]] py::object coordination() const
    {
        return read([&](const auto& workspace) { return workspace.coordination(); });
    }

    [[nodiscard]] py::object edge_values() const
    {
        return read([&](const auto& workspace) { return workspace.edge_values(); });
    }

    [[nodiscard]] py::object edge_radial_derivatives() const
    {
        return read(
            [&](const auto& workspace) { return workspace.edge_radial_derivatives(); });
    }

    [[nodiscard]] py::object coefficients() const
    {
        return read_matrix([&](const auto& workspace) { return workspace.coefficients(); });
    }

    [[nodiscard]] py::object coefficient_derivatives() const
    {
        return read_matrix(
            [&](const auto& workspace) { return workspace.coefficient_derivatives(); });
    }

    [[nodiscard]] py::dict debug_reference_row(
        const py::array& cn_ref, const py::array& valid, const double coordination) const
    {
        const auto cn_ref_buffer = float_buffer(cn_ref, "cn_ref");
        const auto valid_buffer = mask_buffer(valid, "valid");
        const auto report = [&](const auto& workspace) {
            const auto probe = workspace.debug_reference_row(
                std::span<const double>(cn_ref_buffer.data(), cn_ref_buffer.size()),
                std::span<const bool>(valid_buffer.data(), valid_buffer.size()),
                coordination);
            using value_type =
                typename std::remove_cvref_t<decltype(probe.weight)>::value_type;
            const std::vector<py::ssize_t> shape{
                static_cast<py::ssize_t>(probe.weight.size())};
            py::dict row;
            row["delta"] = wrap<value_type>(probe.delta, shape);
            row["weight"] = wrap<value_type>(probe.weight, shape);
            row["weight_derivative"] = wrap<value_type>(probe.weight_derivative, shape);
            row["denominator"] = probe.denominator;
            row["denominator_derivative"] = probe.denominator_derivative;
            row["nearest_slot"] = probe.nearest_slot;
            row["fallback_slot"] = probe.fallback_slot;
            return row;
        };
        if (precision_ == k_float32) {
            return report(typed_const<float>());
        }
        return report(typed_const<double>());
    }

private:
    template <typename Precision>
    void assign(std::int64_t rank_chunk_bound)
    {
        workspace_ = std::static_pointer_cast<void>(
            std::make_shared<CoordinationWorkspace<Precision>>(rank_chunk_bound));
    }

    template <typename Precision>
    [[nodiscard]] CoordinationWorkspace<Precision>& typed()
    {
        if (precision_ != CoordinationWorkspace<Precision>::precision_token()) {
            throw std::logic_error(
                "FourierD3 coordination seam precision mismatch; this is a binding defect.");
        }
        return *static_cast<CoordinationWorkspace<Precision>*>(workspace_.get());
    }

    template <typename Precision>
    [[nodiscard]] const CoordinationWorkspace<Precision>& typed_const() const
    {
        if (precision_ != CoordinationWorkspace<Precision>::precision_token()) {
            throw std::logic_error(
                "FourierD3 coordination seam precision mismatch; this is a binding defect.");
        }
        return *static_cast<const CoordinationWorkspace<Precision>*>(workspace_.get());
    }

    template <typename Function>
    void dispatch_void(Function&& function)
    {
        if (precision_ == k_float32) {
            function(typed<float>());
            return;
        }
        function(typed<double>());
    }

    template <typename Function>
    [[nodiscard]] py::dict dispatch_dict(Function&& function)
    {
        if (precision_ == k_float32) {
            return function(typed<float>());
        }
        return function(typed<double>());
    }

    // A distinct name keeps the read-only entry point's workspace constness
    // explicit.
    template <typename Function>
    [[nodiscard]] py::dict dispatch_const(Function&& function) const
    {
        if (precision_ == k_float32) {
            return function(typed_const<float>());
        }
        return function(typed_const<double>());
    }

    // Per-atom and per-edge vectors keep their native length, so the returned
    // buffer's own size is the shape.
    template <typename Function>
    [[nodiscard]] py::object read(Function&& function) const
    {
        if (precision_ == k_float32) {
            auto values = function(typed_const<float>());
            const std::vector<py::ssize_t> shape{
                static_cast<py::ssize_t>(values.size())};
            return wrap<float>(std::move(values), shape);
        }
        auto values = function(typed_const<double>());
        const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(values.size())};
        return wrap<double>(std::move(values), shape);
    }

    // The rank-chunk matrices are returned shaped (nodes, chunk), so a consumer
    // reads one coefficient per atom and rank slot without reshaping and only
    // the active chunk, never the full rank, is materialised.
    template <typename Function>
    [[nodiscard]] py::object read_matrix(Function&& function) const
    {
        const std::vector<py::ssize_t> shape = matrix_shape();
        if (precision_ == k_float32) {
            return wrap<float>(function(typed_const<float>()), shape);
        }
        return wrap<double>(function(typed_const<double>()), shape);
    }

    [[nodiscard]] std::vector<py::ssize_t> matrix_shape() const
    {
        const CoordinationGeometry geometry = precision_ == k_float32
            ? typed_const<float>().geometry()
            : typed_const<double>().geometry();
        return {
            static_cast<py::ssize_t>(geometry.active_nodes),
            static_cast<py::ssize_t>(geometry.active_rank_chunk)};
    }

    std::string precision_;
    std::shared_ptr<void> workspace_;
};

}  // namespace

void bind_fourier_d3_coordination(py::module_& m)
{
    py::class_<CoordinationSeam>(m, "_FourierD3CoordinationWorkspace")
        .def(
            py::init<std::string, std::int64_t>(), py::arg("precision"),
            py::arg("rank_chunk_bound"),
            "Grow-only modified-coordination and separable-coefficient workspace, "
            "bounded by the configured rank chunk.")
        .def_property_readonly("precision", &CoordinationSeam::precision)
        .def(
            "reserve", &CoordinationSeam::reserve, py::arg("num_nodes"),
            py::arg("num_edges"), py::arg("n_species"), py::arg("n_ref"),
            py::arg("rank_total"), py::arg("rank_chunk"),
            "Validates every shape, bound, and 64-bit product, then grows the "
            "retained buffers all-or-nothing.")
        .def(
            "set_graph", &CoordinationSeam::set_graph, py::arg("offsets"),
            py::arg("sources"), py::arg("distances"), py::arg("numbers"),
            py::arg("covalent_radii"),
            "Loads the receiver-major directed graph; both pair orientations are a "
            "caller precondition.")
        .def(
            "set_reference", &CoordinationSeam::set_reference, py::arg("cn_ref"),
            py::arg("valid"), py::arg("species"), py::arg("v_q"),
            "Loads the covered species, their reference coordination numbers, the "
            "validity mask, and the signed low-rank factors.")
        .def(
            "set_coordination", &CoordinationSeam::set_coordination, py::arg("values"),
            "Replaces the coordination buffer; the coefficient stage's private "
            "injection seam for fallback diagnostics.")
        .def("run_coordination", &CoordinationSeam::run_coordination, py::arg("cutoff"))
        .def(
            "run_coefficients", &CoordinationSeam::run_coefficients,
            py::arg("rank_begin"))
        .def("usage", &CoordinationSeam::usage)
        .def("geometry", &CoordinationSeam::geometry)
        .def("coordination", &CoordinationSeam::coordination)
        .def("edge_values", &CoordinationSeam::edge_values)
        .def("edge_radial_derivatives", &CoordinationSeam::edge_radial_derivatives)
        .def("coefficients", &CoordinationSeam::coefficients)
        .def("coefficient_derivatives", &CoordinationSeam::coefficient_derivatives)
        .def(
            "debug_reference_row", &CoordinationSeam::debug_reference_row,
            py::arg("cn_ref"), py::arg("valid"), py::arg("coordination"),
            "Evaluates one reference weight row through the contraction's own "
            "kernel and reports it verbatim; a diagnostic seam, not a forward path.");
}
