// Private native seam over the FourierD3 analytic reverse path.
//
// These bindings are internal infrastructure: they exist so the reverse chain's
// contract (the coefficient sensitivity including the self derivative, the
// analytic B-spline mesh gather, the receiver-only coordination adjoint, the
// reciprocal and coordination affine virial, the tensile-positive Voigt stress,
// and the normative atomic-energy partition) is testable against the frozen
// references in the qualified build.  They are not package exports, they publish
// no evaluator type, and they expose no energy-only state without its inputs.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "fourier_d3_buffers.hpp"
#include "fourier_d3_diagnostics_report.hpp"
#include "fourier_d3_reverse.hpp"

namespace py = pybind11;

namespace binding = symmetrix::fourier_d3::binding;

namespace {

using symmetrix::fourier_d3::binding::float_array;
using symmetrix::fourier_d3::binding::float_buffer;
using symmetrix::fourier_d3::binding::index_array;
using symmetrix::fourier_d3::binding::index_buffer;
using symmetrix::fourier_d3::binding::mask_array;
using symmetrix::fourier_d3::binding::mask_buffer;
using symmetrix::fourier_d3::binding::reject_buffer;

using symmetrix::fourier_d3::BufferUsage;
using symmetrix::fourier_d3::ReciprocalGeometry;
using symmetrix::fourier_d3::ReciprocalOutcome;
using symmetrix::fourier_d3::ReciprocalReverse;
using symmetrix::fourier_d3::ReverseGeometry;
using symmetrix::fourier_d3::ReverseUsage;

constexpr std::string_view k_float32 = "float32";
constexpr std::string_view k_float64 = "float64";

void require_known_precision(const std::string& precision)
{
    if (precision != k_float32 && precision != k_float64) {
        throw std::invalid_argument(
            "FourierD3 precision must be 'float32' or 'float64', got '" + precision + "'.");
    }
}

// The coefficient, force, and sensitivity buffers carry the storage precision,
// so a float64 request against a float32 workspace is converted exactly once.
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

[[nodiscard]] py::dict forward_geometry_report(const ReciprocalGeometry& geometry)
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

[[nodiscard]] py::dict geometry_report(const ReverseGeometry& geometry)
{
    py::dict report = forward_geometry_report(geometry.forward);
    report["strain_convention"] = geometry.strain_convention;
    report["stress_convention"] = geometry.stress_convention;
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

[[nodiscard]] py::dict usage_report(const ReverseUsage& usage)
{
    py::dict report;
    report["buffers"] = binding::buffer_reports(usage.buffers);
    report["active_bytes"] = usage.active_bytes;
    report["capacity_bytes"] = usage.capacity_bytes;
    report["reservations"] = usage.reservations;
    report["allocations"] = usage.allocations;
    // The composed forward path's own reports, so one call describes the whole
    // object: its mesh allocation, its kernel tables, and its held plans.
    report["mesh"] = mesh_usage_report(usage.forward.mesh);
    report["kernels"] = kernel_usage_report(usage.forward.kernels);
    report["plans"] = binding::plan_usage_reports(usage.forward.plans);
    report["forward_buffer_begin"] = usage.forward_buffer_begin;
    report["plan_retained_bytes"] = usage.forward.plan_retained_bytes;
    report["forward_reservations"] = usage.forward.reservations;
    report["forward_allocations"] = usage.forward.allocations;
    // The composed coordination workspace reports its own counters, geometry,
    // and itemization; several of its buffer names are also forward-path names,
    // so its buffers stay in a nested list rather than being merged.
    py::dict coordination;
    coordination["buffers"] = binding::buffer_reports(usage.coordination.buffers);
    coordination["active_bytes"] = usage.coordination.active_bytes;
    coordination["capacity_bytes"] = usage.coordination.capacity_bytes;
    coordination["reservations"] = usage.coordination.reservations;
    coordination["allocations"] = usage.coordination.allocations;
    report["coordination"] = coordination;
    return report;
}

// Owns one precision-specialized reverse path behind a stable Python name.
class ReverseSeam {
public:
    ReverseSeam(std::string precision, std::int64_t channel_bound)
        : precision_(std::move(precision))
    {
        require_known_precision(precision_);
        if (precision_ == k_float32) {
            assign<float>(channel_bound);
        } else {
            assign<double>(channel_bound);
        }
    }

    [[nodiscard]] const std::string& precision() const noexcept { return precision_; }

    py::dict reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t num_species,
        std::int64_t num_nodes,
        std::int64_t num_edges,
        std::int64_t n_ref,
        std::int64_t rank_total,
        std::int64_t rank_chunk,
        std::int64_t spline_order)
    {
        dispatch_void([&](auto& engine) {
            engine.reserve(
                nx, ny, nz, num_species, num_nodes, num_edges, n_ref, rank_total,
                rank_chunk, spline_order);
        });
        const ReverseGeometry report = geometry_of();
        py::dict outcome;
        outcome["mesh_points"] = report.forward.mesh_points;
        outcome["spectrum_bins"] = report.forward.spectrum_bins;
        outcome["spline_order"] = report.forward.spline_order;
        outcome["active_rank_chunk"] = report.forward.active_rank_chunk;
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

    void set_graph(
        const py::array& offsets,
        const py::array& sources,
        const py::array& shifts,
        const py::array& distances,
        const py::array& numbers,
        const py::array& covalent_radii)
    {
        const auto offsets_buffer = index_buffer(offsets, "offsets");
        const auto sources_buffer = index_buffer(sources, "sources");
        const auto shifts_buffer = index_buffer(shifts, "shifts");
        const auto distances_buffer = float_buffer(distances, "distances");
        const auto numbers_buffer = index_buffer(numbers, "numbers");
        const auto radii_buffer = float_buffer(covalent_radii, "covalent_radii");
        dispatch_void([&](auto& engine) {
            engine.set_graph(
                std::span<const std::int32_t>(offsets_buffer.data(), offsets_buffer.size()),
                std::span<const std::int32_t>(sources_buffer.data(), sources_buffer.size()),
                std::span<const std::int32_t>(shifts_buffer.data(), shifts_buffer.size()),
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
        dispatch_void([&](auto& engine) {
            engine.set_reference(
                std::span<const double>(cn_ref_buffer.data(), cn_ref_buffer.size()),
                std::span<const bool>(valid_buffer.data(), valid_buffer.size()),
                std::span<const std::int32_t>(species_buffer.data(), species_buffer.size()),
                std::span<const double>(v_q_buffer.data(), v_q_buffer.size()));
        });
    }

    [[nodiscard]] py::dict run(double coordination_cutoff)
    {
        return dispatch_dict(
            [&](auto& engine) { return outcome_report(engine.run(coordination_cutoff)); });
    }

    void prepare_host_stencil_baseline()
    {
        dispatch_void([&](auto& engine) { engine.prepare_host_stencil_baseline(); });
    }

    // Readbacks.
    [[nodiscard]] py::object forces() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes), 3},
            [&](const auto& engine) { return engine.forces(); });
    }

    [[nodiscard]] py::object mesh_forces() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes), 3},
            [&](const auto& engine) { return engine.mesh_forces(); });
    }

    [[nodiscard]] py::object coordination_forces() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes), 3},
            [&](const auto& engine) { return engine.coordination_forces(); });
    }

    [[nodiscard]] py::object atomic_energies() const
    {
        const ReverseGeometry report = geometry_of();
        return read_double(
            {static_cast<py::ssize_t>(report.forward.num_nodes)}, [&](const auto& engine) { return engine.atomic_energies(); });
    }

    [[nodiscard]] py::object coefficient_sensitivity() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes),
             static_cast<py::ssize_t>(report.forward.active_rank_chunk)},
            [&](const auto& engine) { return engine.coefficient_sensitivity(); });
    }

    [[nodiscard]] py::object coordination_sensitivity() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes)},
            [&](const auto& engine) { return engine.coordination_sensitivity(); });
    }

    [[nodiscard]] py::object strain_derivative() const
    {
        return read_double(
            {3, 3}, [&](const auto& engine) { return engine.strain_derivative(); });
    }

    [[nodiscard]] py::object stress() const
    {
        return read_double(
            {6}, [&](const auto& engine) { return engine.stress(); });
    }

    [[nodiscard]] py::object virial() const
    {
        return read_double(
            {3, 3}, [&](const auto& engine) { return engine.virial(); });
    }

    [[nodiscard]] py::object coordination() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes)},
            [&](const auto& engine) { return engine.coordination(); });
    }

    [[nodiscard]] py::object edge_values() const
    {
        return read_vector(
            {static_cast<py::ssize_t>(edge_count())},
            [&](const auto& engine) { return engine.edge_values(); });
    }

    [[nodiscard]] py::object edge_radial_derivatives() const
    {
        return read_vector(
            {static_cast<py::ssize_t>(edge_count())},
            [&](const auto& engine) { return engine.edge_radial_derivatives(); });
    }

    [[nodiscard]] py::object coefficients() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes),
             static_cast<py::ssize_t>(report.forward.active_rank_chunk)},
            [&](const auto& engine) { return engine.coefficients(); });
    }

    [[nodiscard]] py::object coefficient_derivatives() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes),
             static_cast<py::ssize_t>(report.forward.active_rank_chunk)},
            [&](const auto& engine) { return engine.coefficient_derivatives(); });
    }

    [[nodiscard]] py::object self_coefficient_derivative() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.num_nodes),
             static_cast<py::ssize_t>(report.forward.active_rank_chunk)},
            [&](const auto& engine) { return engine.self_coefficient_derivative(); });
    }

    [[nodiscard]] py::object structure_factors() const
    {
        const ReverseGeometry report = geometry_of();
        return read_complex(
            {static_cast<py::ssize_t>(report.forward.num_species),
             static_cast<py::ssize_t>(report.forward.active_rank_chunk),
             static_cast<py::ssize_t>(report.forward.spectrum_bins)},
            [&](const auto& engine) { return engine.structure_factors(); });
    }

    [[nodiscard]] py::object pair_kernel() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.species_pairs),
             static_cast<py::ssize_t>(report.forward.spectrum_bins)},
            [&](const auto& engine) { return engine.pair_kernel(); });
    }

    [[nodiscard]] py::object pair_kernel_derivative() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.species_pairs),
             static_cast<py::ssize_t>(report.forward.spectrum_bins)},
            [&](const auto& engine) { return engine.pair_kernel_derivative(); });
    }

    [[nodiscard]] py::object potential(std::int64_t species_slot)
    {
        const ReverseGeometry report = geometry_of();
        const std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(report.forward.active_rank_chunk),
            static_cast<py::ssize_t>(report.forward.mesh_points)};
        // The potential is re-derived from the retained cotangent when the
        // requested slot is not the one the gather left behind, so this readback
        // needs a mutable engine.
        if (precision_ == k_float32) {
            return wrap<float>(typed<float>().potential(species_slot), shape);
        }
        return wrap<double>(typed<double>().potential(species_slot), shape);
    }

    [[nodiscard]] py::object field() const
    {
        const ReverseGeometry report = geometry_of();
        return read_vector(
            {static_cast<py::ssize_t>(report.forward.active_rank_chunk),
             static_cast<py::ssize_t>(report.forward.mesh_points)},
            [&](const auto& engine) { return engine.field(); });
    }

    [[nodiscard]] double reciprocal_energy() const
    {
        return dispatch_value(
            [&](const auto& engine) { return engine.reciprocal_energy(); });
    }

    [[nodiscard]] double self_energy() const
    {
        return dispatch_value([&](const auto& engine) { return engine.self_energy(); });
    }

    [[nodiscard]] double energy() const
    {
        return dispatch_value([&](const auto& engine) { return engine.energy(); });
    }

    [[nodiscard]] py::dict geometry() const { return geometry_report(geometry_of()); }

    [[nodiscard]] py::dict usage() const
    {
        return dispatch_const(
            [&](const auto& engine) { return usage_report(engine.usage()); });
    }

private:
    template <typename Precision>
    void assign(std::int64_t channel_bound)
    {
        engine_ = std::static_pointer_cast<void>(
            std::make_shared<ReciprocalReverse<Precision>>(
                symmetrix::fourier_d3::MeshLayout::right, channel_bound));
    }

    template <typename Precision>
    [[nodiscard]] ReciprocalReverse<Precision>& typed()
    {
        if (precision_ != ReciprocalReverse<Precision>::precision_token()) {
            throw std::logic_error(
                "FourierD3 reverse seam precision mismatch; this is a binding defect.");
        }
        return *static_cast<ReciprocalReverse<Precision>*>(engine_.get());
    }

    template <typename Precision>
    [[nodiscard]] const ReciprocalReverse<Precision>& typed_const() const
    {
        if (precision_ != ReciprocalReverse<Precision>::precision_token()) {
            throw std::logic_error(
                "FourierD3 reverse seam precision mismatch; this is a binding defect.");
        }
        return *static_cast<const ReciprocalReverse<Precision>*>(engine_.get());
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

    [[nodiscard]] ReverseGeometry geometry_of() const
    {
        if (precision_ == k_float32) {
            return typed_const<float>().geometry();
        }
        return typed_const<double>().geometry();
    }

    [[nodiscard]] std::int64_t edge_count() const
    {
        if (precision_ == k_float32) {
            return static_cast<std::int64_t>(typed_const<float>().edge_values().size());
        }
        return static_cast<std::int64_t>(typed_const<double>().edge_values().size());
    }

    // The per-slot vectors carry the storage precision, so the engine the lambda
    // is applied to follows the configured precision, not the result type.
    template <typename Function>
    [[nodiscard]] py::object read_vector(
        const std::vector<py::ssize_t>& shape, Function&& function) const
    {
        if (precision_ == k_float32) {
            return wrap<float>(function(typed_const<float>()), shape);
        }
        return wrap<double>(function(typed_const<double>()), shape);
    }

    // The tensors and the atomic partition are double in both precisions.
    template <typename Function>
    [[nodiscard]] py::object read_double(
        const std::vector<py::ssize_t>& shape, Function&& function) const
    {
        if (precision_ == k_float32) {
            return wrap<double>(function(typed_const<float>()), shape);
        }
        return wrap<double>(function(typed_const<double>()), shape);
    }

    template <typename Function>
    [[nodiscard]] py::object read_complex(
        const std::vector<py::ssize_t>& shape, Function&& function) const
    {
        if (precision_ == k_float32) {
            return wrap_complex<float>(function(typed_const<float>()), shape);
        }
        return wrap_complex<double>(function(typed_const<double>()), shape);
    }

    std::string precision_;
    std::shared_ptr<void> engine_;
};

}  // namespace

void bind_fourier_d3_reverse(py::module_& m)
{
    py::class_<ReverseSeam>(m, "_FourierD3ReciprocalReverse")
        .def(
            py::init<std::string, std::int64_t>(), py::arg("precision"),
            py::arg("rank_chunk_bound"),
            "Private analytic reverse path: coefficient sensitivities, analytic "
            "B-spline gather forces, the receiver-only coordination adjoint, the "
            "affine virial and Voigt stress, and the atomic-energy partition, over "
            "the composed forward path and coordination workspace.")
        .def_property_readonly("precision", &ReverseSeam::precision)
        .def(
            "reserve", &ReverseSeam::reserve, py::arg("nx"), py::arg("ny"),
            py::arg("nz"), py::arg("num_species"), py::arg("num_nodes"),
            py::arg("num_edges"), py::arg("n_ref"), py::arg("rank_total"),
            py::arg("rank_chunk"), py::arg("spline_order"),
            "Validates every shape, bound, and 64-bit product for the forward path, "
            "the coordination workspace, and the reverse buffers, then grows all "
            "three all-or-nothing.")
        .def("set_cell", &ReverseSeam::set_cell, py::arg("cell"))
        .def(
            "set_species", &ReverseSeam::set_species, py::arg("species"),
            py::arg("q_table"))
        .def(
            "set_atoms", &ReverseSeam::set_atoms, py::arg("positions"),
            py::arg("numbers"))
        .def(
            "set_damping", &ReverseSeam::set_damping, py::arg("a1"), py::arg("a2"),
            py::arg("s6"), py::arg("s8"))
        .def(
            "set_rank", &ReverseSeam::set_rank, py::arg("rank_begin"),
            py::arg("eigenvalues"))
        .def(
            "set_graph", &ReverseSeam::set_graph, py::arg("offsets"), py::arg("sources"),
            py::arg("shifts"), py::arg("distances"), py::arg("numbers"),
            py::arg("covalent_radii"),
            "Loads the receiver-major directed graph with its periodic-image shifts "
            "and checks that the shifts and the distances describe the same geometry.")
        .def(
            "set_reference", &ReverseSeam::set_reference, py::arg("cn_ref"),
            py::arg("valid"), py::arg("species"), py::arg("v_q"),
            "Loads the covered species, their reference coordination numbers, the "
            "validity mask, and the signed low-rank factors; the species list must be "
            "the mesh's own channel order.")
        .def(
            "prepare_host_stencil_baseline",
            &ReverseSeam::prepare_host_stencil_baseline,
            "Prepares and retains the host stencil for private parity diagnostics.")
        .def("run", &ReverseSeam::run, py::arg("cutoff"))
        .def("forces", &ReverseSeam::forces)
        .def("mesh_forces", &ReverseSeam::mesh_forces)
        .def("coordination_forces", &ReverseSeam::coordination_forces)
        .def("atomic_energies", &ReverseSeam::atomic_energies)
        .def("coefficient_sensitivity", &ReverseSeam::coefficient_sensitivity)
        .def("coordination_sensitivity", &ReverseSeam::coordination_sensitivity)
        .def("strain_derivative", &ReverseSeam::strain_derivative)
        .def("stress", &ReverseSeam::stress)
        .def("virial", &ReverseSeam::virial)
        .def("coordination", &ReverseSeam::coordination)
        .def("edge_values", &ReverseSeam::edge_values)
        .def("edge_radial_derivatives", &ReverseSeam::edge_radial_derivatives)
        .def("coefficients", &ReverseSeam::coefficients)
        .def("coefficient_derivatives", &ReverseSeam::coefficient_derivatives)
        .def("self_coefficient_derivative", &ReverseSeam::self_coefficient_derivative)
        .def("structure_factors", &ReverseSeam::structure_factors)
        .def("pair_kernel", &ReverseSeam::pair_kernel)
        .def("pair_kernel_derivative", &ReverseSeam::pair_kernel_derivative)
        .def("potential", &ReverseSeam::potential, py::arg("species_slot"))
        .def("field", &ReverseSeam::field)
        .def("reciprocal_energy", &ReverseSeam::reciprocal_energy)
        .def("self_energy", &ReverseSeam::self_energy)
        .def("energy", &ReverseSeam::energy)
        .def("geometry", &ReverseSeam::geometry)
        .def("usage", &ReverseSeam::usage);
}
