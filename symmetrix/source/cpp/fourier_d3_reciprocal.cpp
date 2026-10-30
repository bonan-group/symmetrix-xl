// Private native seam over the FourierD3 particle-mesh forward path.
//
// These bindings are internal infrastructure: they exist so the forward path's
// contract (the fractional wrap and threshold-free B-spline assignment, the
// single continuous-symbol deconvolution, the unnormalized transform pair, the
// reciprocal contraction with the k = 0 term and the Hermitian multiplicity, the
// self term, and the cotangent whose inverse transform is dE/dm for the reverse chain) is
// testable against the frozen references in the qualified build.  They are not
// package exports, they publish no evaluator type, and they expose no
// energy-only state without its inputs.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "fourier_d3_diagnostics_report.hpp"
#include "fourier_d3_reciprocal.hpp"

namespace py = pybind11;

namespace binding = symmetrix::fourier_d3::binding;

namespace {

using symmetrix::fourier_d3::ReciprocalForward;
using symmetrix::fourier_d3::ReciprocalGeometry;
using symmetrix::fourier_d3::ReciprocalKernelProbe;
using symmetrix::fourier_d3::ReciprocalOutcome;
using symmetrix::fourier_d3::ReciprocalUsage;

constexpr std::string_view k_float32 = "float32";
constexpr std::string_view k_float64 = "float64";

void require_known_precision(const std::string& precision)
{
    if (precision != k_float32 && precision != k_float64) {
        throw std::invalid_argument(
            "FourierD3 precision must be 'float32' or 'float64', got '" + precision + "'.");
    }
}

// Input buffers.  A geometry buffer must hold IEEE floats and an index buffer
// must hold integers, so a caller mistake fails with a named error instead of
// being reinterpreted.  `forcecast` lets an integer or a wider float buffer be
// converted once here rather than at every kernel boundary.
using index_array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using float_array = py::array_t<double, py::array::c_style | py::array::forcecast>;

[[noreturn]] void reject_buffer(
    const std::string_view name, const py::dtype& dtype, const char* expected)
{
    throw std::invalid_argument(
        "FourierD3 " + std::string(name) + " must hold " + expected + "; got dtype "
        + std::string(py::str(dtype.attr("name"))) + ".");
}

[[nodiscard]] index_array index_buffer(const py::array& array, const std::string_view name)
{
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'i' && dtype.kind() != 'u') {
        reject_buffer(name, dtype, "integers");
    }
    auto converted = index_array::ensure(array);
    if (!converted) {
        reject_buffer(name, dtype, "integers representable as int32");
    }
    return converted;
}

[[nodiscard]] float_array float_buffer(const py::array& array, const std::string_view name)
{
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'f') {
        reject_buffer(name, dtype, "IEEE floating-point values");
    }
    auto converted = float_array::ensure(array);
    if (!converted) {
        reject_buffer(name, dtype, "IEEE floating-point values");
    }
    return converted;
}

// The coefficient and field buffers carry the storage precision, so a float64
// request against a float32 workspace is converted exactly once here.
template <typename Precision>
[[nodiscard]] py::array_t<Precision, py::array::c_style | py::array::forcecast>
precision_buffer(const py::array& array, const std::string_view name)
{
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'f') {
        reject_buffer(name, dtype, "IEEE floating-point values");
    }
    auto converted =
        py::array_t<Precision, py::array::c_style | py::array::forcecast>::ensure(array);
    if (!converted) {
        reject_buffer(name, dtype, "IEEE floating-point values");
    }
    return converted;
}

template <typename Precision>
[[nodiscard]] py::array_t<Precision> wrap(
    const std::vector<Precision>& values, const std::vector<py::ssize_t>& shape)
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
        std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(Precision));
    }
    return result;
}

template <typename Precision>
[[nodiscard]] py::array_t<std::complex<Precision>> wrap_complex(
    const std::vector<std::complex<Precision>>& values, const std::vector<py::ssize_t>& shape)
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
    py::array_t<std::complex<Precision>> result(shape);
    if (!values.empty()) {
        std::memcpy(
            result.mutable_data(), values.data(),
            values.size() * sizeof(std::complex<Precision>));
    }
    return result;
}

[[nodiscard]] py::dict geometry_report(const ReciprocalGeometry& geometry)
{
    py::dict report;
    report["deconvolution"] = geometry.deconvolution;
    report["forward_normalization"] = geometry.forward_normalization;
    report["inverse_normalization"] = geometry.inverse_normalization;
    report["hermitian_axis"] = geometry.hermitian_axis;
    report["mesh"] = geometry.mesh;
    report["nx"] = geometry.nx;
    report["ny"] = geometry.ny;
    report["nz"] = geometry.nz;
    report["spectrum_extent"] = geometry.spectrum_extent;
    report["mesh_points"] = geometry.mesh_points;
    report["spectrum_bins"] = geometry.spectrum_bins;
    report["species_pairs"] = geometry.species_pairs;
    report["stencil_points"] = geometry.stencil_points;
    report["active_rank_chunk"] = geometry.active_rank_chunk;
    report["rank_begin"] = geometry.rank_begin;
    report["rank_total"] = geometry.rank_total;
    report["num_nodes"] = geometry.num_nodes;
    report["num_species"] = geometry.num_species;
    report["spline_order"] = geometry.spline_order;
    return report;
}

[[nodiscard]] py::dict mesh_usage_report(const symmetrix::fourier_d3::MeshStorageUsage& usage)
{
    py::dict report;
    report["buffers"] = binding::buffer_reports(usage.buffers);
    report["active_bytes"] = usage.active_bytes;
    report["capacity_bytes"] = usage.capacity_bytes;
    report["reservations"] = usage.reservations;
    report["allocations"] = usage.allocations;
    report["allocation_generation"] = usage.allocation_generation;
    report["channel_bound"] = usage.channel_bound;
    report["active_channels"] = usage.active_channels;
    report["peak_channels"] = usage.peak_channels;
    return report;
}

[[nodiscard]] py::dict kernel_usage_report(const symmetrix::fourier_d3::KernelDomainUsage& usage)
{
    py::dict report;
    report["buffers"] = binding::buffer_reports(usage.buffers);
    report["active_bytes"] = usage.active_bytes;
    report["capacity_bytes"] = usage.capacity_bytes;
    report["reservations"] = usage.reservations;
    report["allocations"] = usage.allocations;
    report["key_hits"] = usage.key_hits;
    report["key_misses"] = usage.key_misses;
    report["key_rebuilds"] = usage.key_rebuilds;
    report["has_key"] = usage.has_key;
    report["key"] = usage.key;
    return report;
}

[[nodiscard]] py::dict usage_report(const ReciprocalUsage& usage)
{
    py::dict report;
    report["buffers"] = binding::buffer_reports(usage.buffers);
    report["active_bytes"] = usage.active_bytes;
    report["capacity_bytes"] = usage.capacity_bytes;
    report["plan_retained_bytes"] = usage.plan_retained_bytes;
    report["reservations"] = usage.reservations;
    report["allocations"] = usage.allocations;
    report["mesh"] = mesh_usage_report(usage.mesh);
    report["kernels"] = kernel_usage_report(usage.kernels);
    report["plans"] = binding::plan_usage_reports(usage.plans);
    return report;
}

[[nodiscard]] py::dict outcome_report(const ReciprocalOutcome& outcome)
{
    py::dict report;
    report["reciprocal_energy"] = outcome.reciprocal_energy;
    report["self_energy"] = outcome.self_energy;
    report["energy"] = outcome.energy;
    return report;
}

// Owns one precision-specialized forward path behind a stable Python name.
class ReciprocalSeam {
public:
    ReciprocalSeam(std::string precision, std::int64_t channel_bound)
        : precision_(std::move(precision))
    {
        require_known_precision(precision_);
        if (precision_ == k_float32) {
            initialize_engine<float>(channel_bound);
        } else {
            initialize_engine<double>(channel_bound);
        }
    }

    [[nodiscard]] const std::string& precision() const noexcept { return precision_; }

    py::dict reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t num_species,
        std::int64_t num_nodes,
        std::int64_t rank_total,
        std::int64_t rank_chunk,
        std::int64_t spline_order)
    {
        dispatch_void([&](auto& engine) {
            engine.reserve(
                nx, ny, nz, num_species, num_nodes, rank_total, rank_chunk, spline_order);
        });
        const ReciprocalGeometry report = geometry_of();
        py::dict outcome;
        outcome["mesh_points"] = report.mesh_points;
        outcome["spectrum_bins"] = report.spectrum_bins;
        outcome["spline_order"] = report.spline_order;
        outcome["active_rank_chunk"] = report.active_rank_chunk;
        return outcome;
    }

    void set_cell(const py::array& cell)
    {
        const auto buffer = float_buffer(cell, "cell");
        dispatch_void([&](auto& engine) {
            engine.set_cell(std::span<const double>(buffer.data(), buffer.size()));
        });
    }

    void set_species(const py::array& species, const py::array& q_table)
    {
        const auto species_buffer = index_buffer(species, "species");
        const auto q_buffer = float_buffer(q_table, "q_table");
        dispatch_void([&](auto& engine) {
            engine.set_species(
                std::span<const std::int32_t>(species_buffer.data(), species_buffer.size()),
                std::span<const double>(q_buffer.data(), q_buffer.size()));
        });
    }

    void set_atoms(const py::array& positions, const py::array& numbers)
    {
        const auto positions_buffer = float_buffer(positions, "positions");
        const auto numbers_buffer = index_buffer(numbers, "numbers");
        dispatch_void([&](auto& engine) {
            engine.set_atoms(
                std::span<const double>(positions_buffer.data(), positions_buffer.size()),
                std::span<const std::int32_t>(numbers_buffer.data(), numbers_buffer.size()));
        });
    }

    void set_damping(double a1, double a2, double s6, double s8)
    {
        dispatch_void([&](auto& engine) { engine.set_damping(a1, a2, s6, s8); });
    }

    void set_rank(std::int64_t rank_begin, const py::array& eigenvalues)
    {
        const auto buffer = float_buffer(eigenvalues, "eigenvalues");
        dispatch_void([&](auto& engine) {
            engine.set_rank(rank_begin, std::span<const double>(buffer.data(), buffer.size()));
        });
    }

    void set_coefficients(const py::array& coefficients)
    {
        dispatch_void([&](auto& engine) {
            using value_type = typename std::remove_cvref_t<decltype(engine)>::value_type;
            const auto buffer = precision_buffer<value_type>(coefficients, "coefficients");
            engine.set_coefficients(std::span<const value_type>(buffer.data(), buffer.size()));
        });
    }

    void assign(std::int64_t species_slot)
    {
        dispatch_void([&](auto& engine) { engine.assign(species_slot); });
    }

    void set_field(const py::array& values)
    {
        dispatch_void([&](auto& engine) {
            using value_type = typename std::remove_cvref_t<decltype(engine)>::value_type;
            const auto buffer = precision_buffer<value_type>(values, "field");
            engine.set_field(std::span<const value_type>(buffer.data(), buffer.size()));
        });
    }

    void transform_forward()
    {
        dispatch_void([&](auto& engine) { engine.transform_forward(); });
    }

    void transform_inverse()
    {
        dispatch_void([&](auto& engine) { engine.transform_inverse(); });
    }

    void deconvolve()
    {
        dispatch_void([&](auto& engine) { engine.deconvolve(); });
    }

    void store_spectrum(std::int64_t species_slot)
    {
        dispatch_void([&](auto& engine) { engine.store_spectrum(species_slot); });
    }

    void contract()
    {
        dispatch_void([&](auto& engine) { engine.contract(); });
    }

    // The self term is evaluated by `run` and by `run_self_energy`; the
    // `self_energy` readback below only reports the value that stage produced.
    void run_self_energy()
    {
        dispatch_void([&](auto& engine) { engine.self_energy(); });
    }

    void invert(std::int64_t species_slot)
    {
        dispatch_void([&](auto& engine) { engine.invert(species_slot); });
    }

    [[nodiscard]] py::dict run()
    {
        return dispatch_dict([&](auto& engine) { return outcome_report(engine.run()); });
    }

    [[nodiscard]] double debug_evaluate_field(const py::array& values)
    {
        // The diagnostic evaluation drives the pipeline, so it needs a mutable
        // engine and its own dispatch.
        if (precision_ == k_float32) {
            using value_type = float;
            const auto buffer = precision_buffer<value_type>(values, "field");
            return static_cast<double>(typed<float>().debug_evaluate_field(
                std::span<const value_type>(buffer.data(), buffer.size())));
        }
        using value_type = double;
        const auto buffer = precision_buffer<value_type>(values, "field");
        return static_cast<double>(typed<double>().debug_evaluate_field(
            std::span<const value_type>(buffer.data(), buffer.size())));
    }

    // Readbacks.  Every per-slot vector is returned with the slot axis spelled
    // out, so a caller never has to reconstruct the mesh shape by hand.
    [[nodiscard]] py::object field() const
    {
        const ReciprocalGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.active_rank_chunk),
             static_cast<py::ssize_t>(report.mesh_points)},
            [&](const auto& engine) { return engine.field(); });
    }

    [[nodiscard]] py::object potential(std::int64_t species_slot) const
    {
        const ReciprocalGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.active_rank_chunk),
             static_cast<py::ssize_t>(report.mesh_points)},
            [&](const auto& engine) { return engine.potential(species_slot); });
    }

    [[nodiscard]] py::object spectrum() const
    {
        const ReciprocalGeometry report = geometry_of();
        return read_complex(
            {static_cast<py::ssize_t>(report.active_rank_chunk),
             static_cast<py::ssize_t>(report.nx), static_cast<py::ssize_t>(report.ny),
             static_cast<py::ssize_t>(report.spectrum_extent)},
            [&](const auto& engine) { return engine.spectrum(); });
    }

    [[nodiscard]] py::object structure_factors() const
    {
        const ReciprocalGeometry report = geometry_of();
        return read_complex(
            {static_cast<py::ssize_t>(report.num_species),
             static_cast<py::ssize_t>(report.active_rank_chunk),
             static_cast<py::ssize_t>(report.spectrum_bins)},
            [&](const auto& engine) { return engine.structure_factors(); });
    }

    [[nodiscard]] py::object pair_kernel() const
    {
        const ReciprocalGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.species_pairs),
             static_cast<py::ssize_t>(report.spectrum_bins)},
            [&](const auto& engine) { return engine.pair_kernel(); });
    }

    [[nodiscard]] py::object pair_kernel_derivative() const
    {
        const ReciprocalGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.species_pairs),
             static_cast<py::ssize_t>(report.spectrum_bins)},
            [&](const auto& engine) { return engine.pair_kernel_derivative(); });
    }

    [[nodiscard]] py::object bin_energies() const
    {
        const ReciprocalGeometry report = geometry_of();
        return wrap<double>(
            call_double([&](const auto& engine) { return engine.bin_energies(); }),
            {static_cast<py::ssize_t>(report.spectrum_bins)});
    }

    [[nodiscard]] py::object self_coefficient_derivative() const
    {
        const ReciprocalGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.num_nodes),
             static_cast<py::ssize_t>(report.active_rank_chunk)},
            [&](const auto& engine) { return engine.self_coefficient_derivative(); });
    }

    [[nodiscard]] py::dict stencil(bool host_baseline)
    {
        const ReciprocalGeometry report = geometry_of();
        const std::vector<py::ssize_t> base_shape{
            static_cast<py::ssize_t>(report.num_nodes), 3};
        const std::vector<py::ssize_t> value_shape{
            static_cast<py::ssize_t>(report.num_nodes), 3,
            static_cast<py::ssize_t>(report.spline_order)};
        py::dict result;
        if (precision_ == k_float32) {
            auto& engine = typed<float>();
            const auto diagnostic = engine.stencil_diagnostic(host_baseline);
            result["base"] = wrap<std::int32_t>(diagnostic.base, base_shape);
            result["weights"] = wrap<float>(diagnostic.weights, value_shape);
            result["slopes"] = wrap<float>(diagnostic.slopes, value_shape);
            result["host_fallback"] = diagnostic.host_fallback;
            result["preparation"] = std::string(
                symmetrix::particle_mesh::stencil_preparation_path_token(
                    diagnostic.preparation));
        } else {
            auto& engine = typed<double>();
            const auto diagnostic = engine.stencil_diagnostic(host_baseline);
            result["base"] = wrap<std::int32_t>(diagnostic.base, base_shape);
            result["weights"] = wrap<double>(diagnostic.weights, value_shape);
            result["slopes"] = wrap<double>(diagnostic.slopes, value_shape);
            result["host_fallback"] = diagnostic.host_fallback;
            result["preparation"] = std::string(
                symmetrix::particle_mesh::stencil_preparation_path_token(
                    diagnostic.preparation));
        }
        return result;
    }

    [[nodiscard]] double reciprocal_energy() const
    {
        return dispatch_value(
            [&](const auto& engine) { return engine.reciprocal_energy(); });
    }

    [[nodiscard]] double self_energy() const
    {
        return dispatch_value(
            [&](const auto& engine) { return engine.self_energy_value(); });
    }

    [[nodiscard]] double energy() const
    {
        return dispatch_value([&](const auto& engine) { return engine.energy_value(); });
    }

    [[nodiscard]] py::dict geometry() const
    {
        return geometry_report(geometry_of());
    }

    [[nodiscard]] py::dict usage() const
    {
        return dispatch_const(
            [&](const auto& engine) { return usage_report(engine.usage()); });
    }

private:
    template <typename Precision>
    void initialize_engine(std::int64_t channel_bound)
    {
        engine_ = std::static_pointer_cast<void>(
            std::make_shared<ReciprocalForward<Precision>>(
                symmetrix::fourier_d3::MeshLayout::right, channel_bound));
    }

    template <typename Precision>
    [[nodiscard]] ReciprocalForward<Precision>& typed()
    {
        if (precision_ != ReciprocalForward<Precision>::precision_token()) {
            throw std::logic_error(
                "FourierD3 reciprocal seam precision mismatch; this is a binding defect.");
        }
        return *static_cast<ReciprocalForward<Precision>*>(engine_.get());
    }

    template <typename Precision>
    [[nodiscard]] const ReciprocalForward<Precision>& typed_const() const
    {
        if (precision_ != ReciprocalForward<Precision>::precision_token()) {
            throw std::logic_error(
                "FourierD3 reciprocal seam precision mismatch; this is a binding defect.");
        }
        return *static_cast<const ReciprocalForward<Precision>*>(engine_.get());
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

    template <typename Function>
    [[nodiscard]] py::dict dispatch_const(Function&& function) const
    {
        if (precision_ == k_float32) {
            return function(typed_const<float>());
        }
        return function(typed_const<double>());
    }

    template <typename Function>
    [[nodiscard]] double dispatch_value(Function&& function) const
    {
        if (precision_ == k_float32) {
            return static_cast<double>(function(typed_const<float>()));
        }
        return static_cast<double>(function(typed_const<double>()));
    }

    [[nodiscard]] ReciprocalGeometry geometry_of() const
    {
        if (precision_ == k_float32) {
            return typed_const<float>().geometry();
        }
        return typed_const<double>().geometry();
    }

    template <typename Function>
    [[nodiscard]] std::vector<float> call_float(Function&& function) const
    {
        return function(typed_const<float>());
    }

    template <typename Function>
    [[nodiscard]] std::vector<double> call_double(Function&& function) const
    {
        return function(typed_const<double>());
    }

    template <typename Function>
    [[nodiscard]] py::object read_vector(
        const std::vector<py::ssize_t>& shape, Function&& function) const
    {
        if (precision_ == k_float32) {
            return wrap<float>(call_float(function), shape);
        }
        return wrap<double>(call_double(function), shape);
    }

    template <typename Function>
    [[nodiscard]] py::object read_complex(
        const std::vector<py::ssize_t>& shape, Function&& function) const
    {
        if (precision_ == k_float32) {
            return wrap_complex<float>(
                function(typed_const<float>()), shape);
        }
        return wrap_complex<double>(function(typed_const<double>()), shape);
    }

    std::string precision_;
    std::shared_ptr<void> engine_;
};

}  // namespace

void bind_fourier_d3_reciprocal(py::module_& m)
{
    py::class_<ReciprocalSeam>(m, "_FourierD3ReciprocalForward")
        .def(
            py::init<std::string, std::int64_t>(), py::arg("precision"),
            py::arg("rank_chunk_bound"),
            "Private particle-mesh forward path and reciprocal contraction, bounded "
            "by the configured rank chunk.")
        .def_property_readonly("precision", &ReciprocalSeam::precision)
        .def(
            "reserve", &ReciprocalSeam::reserve, py::arg("nx"), py::arg("ny"),
            py::arg("nz"), py::arg("num_species"), py::arg("num_nodes"),
            py::arg("rank_total"), py::arg("rank_chunk"), py::arg("spline_order"),
            "Validates every shape, bound, and 64-bit product, then replaces the "
            "retained buffers all-or-nothing.")
        .def("set_cell", &ReciprocalSeam::set_cell, py::arg("cell"))
        .def(
            "set_species", &ReciprocalSeam::set_species, py::arg("species"),
            py::arg("q_table"))
        .def(
            "set_atoms", &ReciprocalSeam::set_atoms, py::arg("positions"),
            py::arg("numbers"))
        .def(
            "set_damping", &ReciprocalSeam::set_damping, py::arg("a1"), py::arg("a2"),
            py::arg("s6"), py::arg("s8"))
        .def(
            "set_rank", &ReciprocalSeam::set_rank, py::arg("rank_begin"),
            py::arg("eigenvalues"))
        .def("set_coefficients", &ReciprocalSeam::set_coefficients)
        .def("assign", &ReciprocalSeam::assign, py::arg("species_slot"))
        .def("set_field", &ReciprocalSeam::set_field, py::arg("values"))
        .def("transform_forward", &ReciprocalSeam::transform_forward)
        .def("transform_inverse", &ReciprocalSeam::transform_inverse)
        .def("deconvolve", &ReciprocalSeam::deconvolve)
        .def("store_spectrum", &ReciprocalSeam::store_spectrum, py::arg("species_slot"))
        .def("contract", &ReciprocalSeam::contract)
        .def("run_self_energy", &ReciprocalSeam::run_self_energy)
        .def("invert", &ReciprocalSeam::invert, py::arg("species_slot"))
        .def("run", &ReciprocalSeam::run)
        .def(
            "debug_evaluate_field", &ReciprocalSeam::debug_evaluate_field,
            py::arg("values"),
            "Contracts a supplied real field without spreading, so the cotangent "
            "can be checked against a central difference of the same value "
            "function; a diagnostic seam, not a forward path.")
        .def("field", &ReciprocalSeam::field)
        .def("spectrum", &ReciprocalSeam::spectrum)
        .def("structure_factors", &ReciprocalSeam::structure_factors)
        .def("potential", &ReciprocalSeam::potential, py::arg("species_slot"))
        .def("pair_kernel", &ReciprocalSeam::pair_kernel)
        .def("pair_kernel_derivative", &ReciprocalSeam::pair_kernel_derivative)
        .def("bin_energies", &ReciprocalSeam::bin_energies)
        .def("reciprocal_energy", &ReciprocalSeam::reciprocal_energy)
        .def("self_energy", &ReciprocalSeam::self_energy)
        .def("energy", &ReciprocalSeam::energy)
        .def("self_coefficient_derivative", &ReciprocalSeam::self_coefficient_derivative)
        .def(
            "stencil", &ReciprocalSeam::stencil, py::arg("host_baseline") = false,
            "Returns the prepared base indices, weights, and fractional slopes for "
            "host/device parity diagnostics.")
        .def("geometry", &ReciprocalSeam::geometry)
        .def("usage", &ReciprocalSeam::usage);

    m.def(
        "_fourier_d3_reciprocal_kernel",
        [](const std::string& precision,
           const py::array& k_norms,
           double r0,
           double q_product,
           double s6,
           double s8) {
            require_known_precision(precision);
            const auto norms = float_buffer(k_norms, "k_norms");
            const std::span<const double> span(norms.data(), norms.size());
            const auto report = [&](const auto& probes, const std::string& name) {
                py::dict rendered;
                std::vector<double> values;
                std::vector<double> derivatives;
                values.reserve(probes.size());
                derivatives.reserve(probes.size());
                for (const ReciprocalKernelProbe& probe : probes) {
                    values.push_back(probe.value);
                    derivatives.push_back(probe.derivative);
                }
                rendered["value"] = wrap<double>(
                    values, {static_cast<py::ssize_t>(values.size())});
                rendered["derivative"] = wrap<double>(
                    derivatives, {static_cast<py::ssize_t>(derivatives.size())});
                rendered["precision"] = name;
                return rendered;
            };
            if (precision == k_float32) {
                return report(
                    symmetrix::fourier_d3::probe_reciprocal_kernel<float>(
                        span, r0, q_product, s6, s8),
                    "float32");
            }
            return report(
                symmetrix::fourier_d3::probe_reciprocal_kernel<double>(
                    span, r0, q_product, s6, s8),
                "float64");
        },
        py::arg("precision"), py::arg("k_norms"), py::arg("r0"), py::arg("q_product"),
        py::arg("s6"), py::arg("s8"),
        "Evaluates the Becke-Johnson reciprocal kernel and its radial derivative "
        "through the device kernel itself, including the k = 0 limit and the "
        "series crossover.");
}
