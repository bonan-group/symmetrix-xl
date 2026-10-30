// Private native bindings for the FourierD3 evaluator and its canonical tables.
//
// These bindings are internal infrastructure.  They exist so the Python
// correction runtime can drive the dtype-specific evaluator
// (`FourierD3Kokkos` for float64, `FourierD3KokkosFloat` for float32) through the
// protocol it already consumes, and so the decomposition utility can read the
// canonical D3 tables and their provenance fingerprint from the *installed*
// package instead of parsing a source file at runtime.
//
// Nothing here is a package export: the public surface is the immutable Python
// configuration, while these classes and the accessor are resolved through the
// selected native module. The translation unit is compiled only where the FFT
// build capability found a qualified provider.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dispersion_parameter_provenance.hpp"
#include "dispersion_parameters.hpp"
#include "fourier_d3_buffers.hpp"
#include "fourier_d3_diagnostics_report.hpp"
#include "fourier_d3_evaluator.hpp"
#include "sha256.hpp"

namespace py = pybind11;

namespace binding = symmetrix::fourier_d3::binding;

namespace {

using symmetrix::dispersion::provenance::conversion_constants;
using symmetrix::dispersion::provenance::source_hashes;
using symmetrix::fourier_d3::FourierD3Diagnostics;
using symmetrix::fourier_d3::FourierD3Damping;
using symmetrix::fourier_d3::FourierD3GraphDiagnostics;
using symmetrix::fourier_d3::FourierD3Kokkos;
using symmetrix::fourier_d3::binding::float_array;
using symmetrix::fourier_d3::binding::float_buffer;
using symmetrix::fourier_d3::binding::index_array;
using symmetrix::fourier_d3::binding::index_buffer;
using symmetrix::fourier_d3::binding::mask_array;
using symmetrix::fourier_d3::binding::mask_buffer;
using symmetrix::fourier_d3::binding::reject_buffer;
using symmetrix::fourier_d3::binding::require_columns;
using symmetrix::fourier_d3::binding::require_rank;

constexpr py::ssize_t kMaxAtomicNumber = symmetrix::dispersion::d3_max_atomic_number;
constexpr py::ssize_t kTableExtent = kMaxAtomicNumber + 1;
constexpr py::ssize_t kReferenceSlots =
    symmetrix::dispersion::d3_max_coordination_references;
constexpr py::ssize_t kC6Fields = 3;

// ---------------------------------------------------------------------------
// canonical tables and their provenance
// ---------------------------------------------------------------------------

[[nodiscard]] std::string digest_of(const std::span<const double> values)
{
    sha256::Digest digest;
    digest.update(std::string(
        reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double)));
    return digest.finish();
}

[[nodiscard]] std::string table_fingerprint(
    const std::string& c6_digest, const std::string& radii_digest,
    const std::string& q_table_digest)
{
    // The fingerprint identifies the *table set*: the generated provenance
    // identity (schema, upstream revision, and the hashes of every upstream file
    // the generator read) followed by the content digest of each published array.
    // Either half changing therefore changes the fingerprint, which is what the
    // active-species decomposition caches on.
    using namespace symmetrix::dispersion::provenance;
    sha256::Digest digest;
    digest.update(std::string(schema) + "\n");
    digest.update(std::string(upstream_version) + "\n");
    digest.update(std::string(upstream_tag) + "\n");
    digest.update(std::string(upstream_commit) + "\n");
    for (const SourceHash& source : source_hashes) {
        digest.update(std::string(source.path) + ":" + std::string(source.sha256) + "\n");
    }
    digest.update("d3_c6:" + c6_digest + "\n");
    digest.update("d3_rcov:" + radii_digest + "\n");
    digest.update("d3_r2r4:" + q_table_digest + "\n");
    return digest.finish();
}

template <typename Element>
[[nodiscard]] py::array_t<Element> copy_into(
    const std::span<const Element> values, const std::vector<py::ssize_t>& shape)
{
    py::ssize_t elements = 1;
    for (const py::ssize_t extent : shape) {
        elements *= extent;
    }
    if (elements != static_cast<py::ssize_t>(values.size())) {
        throw std::logic_error(
            "FourierD3 canonical table shape does not match its element count; this is "
            "a binding defect.");
    }
    py::array_t<Element> result(shape);
    if (!values.empty()) {
        std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(Element));
    }
    return result;
}

[[nodiscard]] py::dict canonical_tables()
{
    using namespace symmetrix::dispersion;
    const std::span<const double> c6 = d3_c6_ev_angstrom6();
    const std::span<const double> radii = d3_rcov_angstrom();
    const std::span<const double> q_table = d3_r2r4_angstrom();
    const auto c6_elements = static_cast<std::size_t>(
        kTableExtent * kTableExtent * kReferenceSlots * kReferenceSlots * kC6Fields);
    if (c6.size() != c6_elements || radii.size() != static_cast<std::size_t>(kTableExtent)
        || q_table.size() != static_cast<std::size_t>(kTableExtent)) {
        throw std::logic_error(
            "FourierD3 canonical table extents do not match the generated parameter "
            "set; this is a build defect.");
    }

    const std::string c6_digest = digest_of(c6);
    const std::string radii_digest = digest_of(radii);
    const std::string q_digest = digest_of(q_table);

    py::dict report;
    report["d3_c6"] = copy_into<double>(
        c6, {kTableExtent, kTableExtent, kReferenceSlots, kReferenceSlots, kC6Fields});
    report["d3_rcov"] = copy_into<double>(radii, {kTableExtent});
    report["d3_r2r4"] = copy_into<double>(q_table, {kTableExtent});
    report["supported_atomic_number"] = kMaxAtomicNumber;
    report["table_fingerprint"] = table_fingerprint(c6_digest, radii_digest, q_digest);

    py::dict hashes;
    hashes["d3_c6"] = c6_digest;
    hashes["d3_rcov"] = radii_digest;
    hashes["d3_r2r4"] = q_digest;
    report["table_hashes"] = hashes;

    using namespace symmetrix::dispersion::provenance;
    py::dict provenance;
    provenance["schema"] = std::string(schema);
    provenance["generator_name"] = std::string(generator_name);
    provenance["generator_schema"] = std::string(generator_schema);
    provenance["generator_version"] = std::string(generator_version);
    provenance["upstream_version"] = std::string(upstream_version);
    provenance["upstream_tag"] = std::string(upstream_tag);
    provenance["upstream_commit"] = std::string(upstream_commit);
    provenance["license"] = std::string(license);
    provenance["attribution"] = std::string(attribution);
    py::dict sources;
    for (const SourceHash& source : source_hashes) {
        sources[py::str(source.path)] = std::string(source.sha256);
    }
    provenance["source_hashes"] = sources;
    py::dict constants;
    for (const ConversionConstant& constant : conversion_constants) {
        py::dict entry;
        entry["value"] = constant.value;
        entry["units"] = std::string(constant.units);
        constants[py::str(constant.name)] = entry;
    }
    provenance["conversion_constants"] = constants;
    report["provenance"] = provenance;
    return report;
}

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

[[nodiscard]] py::dict damping_report(const FourierD3Damping& damping)
{
    py::dict report;
    report["a1"] = damping.a1;
    report["a2_angstrom"] = damping.a2_angstrom;
    report["s6"] = damping.s6;
    report["s8"] = damping.s8;
    return report;
}

[[nodiscard]] py::dict graph_report(const FourierD3GraphDiagnostics& diagnostics)
{
    py::dict report;
    report["graph_backend"] = diagnostics.graph_backend;
    report["generation"] = diagnostics.generation;
    report["geometry_update_count"] = diagnostics.geometry_update_count;
    report["num_nodes"] = diagnostics.num_nodes;
    report["num_edges"] = diagnostics.num_edges;
    report["build_cutoff"] = diagnostics.build_cutoff;
    return report;
}

[[nodiscard]] py::dict diagnostics_report(const FourierD3Diagnostics& diagnostics)
{
    py::dict report;
    report["precision"] = diagnostics.precision;
    report["functional"] = diagnostics.functional;
    report["table_fingerprint"] = diagnostics.table_fingerprint;
    report["damping"] = damping_report(diagnostics.damping);
    report["coordination_cutoff"] = diagnostics.coordination_cutoff;
    report["spline_order"] = diagnostics.spline_order;
    report["active_species"] = diagnostics.active_species;
    report["retained_rank"] = diagnostics.retained_rank;
    report["rank_total"] = diagnostics.rank_total;
    report["rank_chunk"] = diagnostics.rank_chunk;
    report["rank_chunk_count"] = diagnostics.rank_chunk_count;
    report["n_references"] = diagnostics.n_references;
    report["decomposition_max_relative_error"] =
        diagnostics.decomposition_max_relative_error;
    report["decomposition_tolerance_met"] = diagnostics.decomposition_tolerance_met;
    report["mesh"] = diagnostics.mesh;
    report["mesh_points"] = diagnostics.mesh_points;
    report["mesh_spacing"] = diagnostics.mesh_spacing;
    report["mesh_invalidation_count"] = diagnostics.mesh_invalidation_count;
    report["active_bytes"] = diagnostics.active_bytes;
    report["capacity_bytes"] = diagnostics.capacity_bytes;
    // The itemization reconciles with the two totals above exactly; the plan
    // record and the retained bytes say which held plan owns which allocation.
    report["memory_components"] = binding::buffer_reports(diagnostics.memory_components);
    // Named for the plan domain, not for the list: the aggregate `plan_*` counters
    // below are the sum over these roles.
    report["fft_plans"] = binding::plan_usage_reports(diagnostics.plans);
    report["plan_retained_bytes"] = diagnostics.plan_retained_bytes;
    report["reservations"] = diagnostics.reservations;
    report["allocations"] = diagnostics.allocations;
    report["plan_builds"] = diagnostics.plan_builds;
    report["plan_reuses"] = diagnostics.plan_reuses;
    report["plan_releases"] = diagnostics.plan_releases;
    report["plan_key_rebuilds"] = diagnostics.plan_key_rebuilds;
    report["kernel_key_hits"] = diagnostics.kernel_key_hits;
    report["kernel_key_misses"] = diagnostics.kernel_key_misses;
    report["kernel_key_rebuilds"] = diagnostics.kernel_key_rebuilds;
    report["stencil_points"] = diagnostics.stencil_points;
    report["input_generation"] = diagnostics.input_generation;
    report["published_generation"] = diagnostics.published_generation;
    report["computed"] = diagnostics.computed;
    report["result_available"] = diagnostics.result_available;
    report["forces_available"] = diagnostics.forces_available;
    report["stress_available"] = diagnostics.stress_available;
    report["engine_loaded"] = diagnostics.engine_loaded;
    // The timing record is present only when timing was opted into: a disabled
    // evaluator reports no record rather than a plausible zero.
    report["timing_enabled"] = diagnostics.timing_enabled;
    report["stage_timing"] = diagnostics.timing_enabled
        ? py::object(binding::stage_timings_report(diagnostics.timings))
        : py::object(py::none());
    return report;
}

// ---------------------------------------------------------------------------
// result export
// ---------------------------------------------------------------------------

template <typename Precision>
[[nodiscard]] py::array_t<Precision> wrap_vector(
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
[[nodiscard]] py::dict result_report(
    const FourierD3Kokkos<Precision>& evaluator, const py::object& properties)
{
    constexpr std::array<std::string_view, 4> known{
        "energy", "atomic_energies", "forces", "stress"};
    std::vector<std::string> requested;
    if (!properties.is_none()) {
        for (const py::handle item : properties) {
            requested.push_back(py::cast<std::string>(item));
        }
        for (const std::string& name : requested) {
            if (std::find(known.begin(), known.end(), name) == known.end()) {
                throw std::invalid_argument(
                    "FourierD3 result does not know the property '" + name
                    + "'; it publishes energy, atomic_energies, forces, and stress.");
            }
        }
    }

    py::dict result;
    const bool all = properties.is_none();
    const auto wanted = [&requested, all](const std::string_view name) {
        return all || std::find(requested.begin(), requested.end(), name) != requested.end();
    };
    if (wanted("energy")) {
        result["energy"] = evaluator.energy();
    }
    if (wanted("atomic_energies")) {
        const std::vector<double> atomic = evaluator.atomic_energies();
        result["atomic_energies"] =
            wrap_vector<double>(atomic, {static_cast<py::ssize_t>(atomic.size())});
    }
    if (wanted("forces")) {
        const std::vector<Precision> forces = evaluator.forces();
        result["forces"] = wrap_vector<Precision>(
            forces, {static_cast<py::ssize_t>(forces.size() / 3), 3});
    }
    if (wanted("stress")) {
        const std::vector<double> stress = evaluator.stress();
        result["stress"] =
            wrap_vector<double>(stress, {static_cast<py::ssize_t>(stress.size())});
    }
    return result;
}

// ---------------------------------------------------------------------------
// the evaluator classes
// ---------------------------------------------------------------------------

[[nodiscard]] std::vector<std::int64_t> mesh_extents(
    const py::array& mesh, const std::string_view name)
{
    require_rank(mesh, name, 1);
    const py::dtype dtype = mesh.dtype();
    if (dtype.kind() != 'i' && dtype.kind() != 'u') {
        reject_buffer(name, dtype, "integers");
    }
    auto values =
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(mesh);
    if (!values) {
        reject_buffer(name, dtype, "integers");
    }
    if (values.size() != 3) {
        throw std::invalid_argument(
            "FourierD3 " + std::string(name) + " must hold three extents, got "
            + std::to_string(values.size()) + ".");
    }
    return std::vector<std::int64_t>(values.data(), values.data() + 3);
}

template <typename Precision>
void bind_evaluator_class(py::module_& module, const char* name)
{
    py::class_<FourierD3Kokkos<Precision>>(module, name)
        .def(
            py::init([](const std::string& functional,
                        const double coordination_cutoff,
                        const std::int64_t spline_order,
                        const std::int64_t rank_chunk,
                        const std::string& table_fingerprint,
                        const py::array& radii,
                        const py::array& q_table) {
                require_rank(radii, "radii", 1);
                require_rank(q_table, "q_table", 1);
                const float_array radius_values = float_buffer(radii, "radii");
                const float_array q_values = float_buffer(q_table, "q_table");
                // The evaluator owns a Kokkos live-object guard, which is
                // deliberately neither copyable nor movable, so it is built in
                // place through a holder rather than returned by value.
                return std::make_unique<FourierD3Kokkos<Precision>>(
                    functional,
                    coordination_cutoff,
                    spline_order,
                    rank_chunk,
                    table_fingerprint,
                    std::span<const double>(radius_values.data(), radius_values.size()),
                    std::span<const double>(q_values.data(), q_values.size()));
            }),
            py::arg("functional"),
            py::arg("coordination_cutoff"),
            py::arg("spline_order"),
            py::arg("rank_chunk"),
            py::arg("table_fingerprint"),
            py::arg("radii"),
            py::arg("q_table"))
        .def(
            "set_mesh",
            [](FourierD3Kokkos<Precision>& self, const py::sequence& mesh) {
                // A three-element sequence of integers: the runtime holds the resolved
                // mesh as a plain tuple, and a float extent must not be truncated.
                if (py::len(mesh) != 3) {
                    throw std::invalid_argument(
                        "set_mesh expects three extents, got "
                        + std::to_string(py::len(mesh)) + ".");
                }
                std::array<std::int64_t, 3> extents{};
                for (std::size_t axis = 0; axis < extents.size(); ++axis) {
                    const py::handle value = mesh[axis];
                    if (!PyIndex_Check(value.ptr())) {
                        throw std::invalid_argument(
                            "set_mesh extents must be integers.");
                    }
                    extents[axis] = py::cast<std::int64_t>(value);
                }
                self.set_mesh(extents);
            },
            py::arg("mesh"))
        .def(
            "set_decomposition",
            [](FourierD3Kokkos<Precision>& self,
               const py::array& species,
               const py::array& eigenvalues,
               const py::array& v_q,
               const py::array& cn_ref,
               const py::array& valid,
               const double max_relative_error,
               const bool tolerance_met) {
                require_rank(species, "species", 1);
                require_rank(eigenvalues, "eigenvalues", 1);
                require_rank(v_q, "v_q", 3);
                require_rank(cn_ref, "cn_ref", 2);
                require_rank(valid, "valid", 2);
                const index_array species_values = index_buffer(species, "species");
                const float_array eigenvalue_values =
                    float_buffer(eigenvalues, "eigenvalues");
                const float_array factor_values = float_buffer(v_q, "v_q");
                const float_array reference_values = float_buffer(cn_ref, "cn_ref");
                const mask_array valid_values = mask_buffer(valid, "valid");
                self.set_decomposition(
                    std::span<const std::int32_t>(
                        species_values.data(), species_values.size()),
                    std::span<const double>(
                        eigenvalue_values.data(), eigenvalue_values.size()),
                    std::span<const double>(factor_values.data(), factor_values.size()),
                    std::span<const double>(
                        reference_values.data(), reference_values.size()),
                    std::span<const bool>(valid_values.data(), valid_values.size()),
                    max_relative_error,
                    tolerance_met);
            },
            py::arg("species"),
            py::arg("eigenvalues"),
            py::arg("v_q"),
            py::arg("cn_ref"),
            py::arg("valid"),
            py::arg("max_relative_error"),
            py::arg("tolerance_met"))
        .def(
            "prepare_host_graph",
            [](FourierD3Kokkos<Precision>& self,
               const py::array& numbers,
               const py::array& receiver_offsets,
               const py::array& sources,
               const py::array& shifts,
               const py::array& positions,
               const py::array& cell,
               const double build_cutoff) {
                require_rank(numbers, "numbers", 1);
                require_rank(receiver_offsets, "receiver_offsets", 1);
                require_rank(sources, "sources", 1);
                require_columns(shifts, "shifts", 3);
                require_columns(positions, "positions", 3);
                require_columns(cell, "cell", 3);
                const index_array number_values = index_buffer(numbers, "numbers");
                const index_array offset_values =
                    index_buffer(receiver_offsets, "receiver_offsets");
                const index_array source_values = index_buffer(sources, "sources");
                const index_array shift_values = index_buffer(shifts, "shifts");
                const float_array position_values = float_buffer(positions, "positions");
                const float_array cell_values = float_buffer(cell, "cell");
                self.prepare_host_graph(
                    std::span<const std::int32_t>(number_values.data(), number_values.size()),
                    std::span<const std::int32_t>(offset_values.data(), offset_values.size()),
                    std::span<const std::int32_t>(source_values.data(), source_values.size()),
                    std::span<const std::int32_t>(shift_values.data(), shift_values.size()),
                    std::span<const double>(position_values.data(), position_values.size()),
                    std::span<const double>(cell_values.data(), cell_values.size()),
                    build_cutoff);
            },
            py::arg("numbers"),
            py::arg("receiver_offsets"),
            py::arg("sources"),
            py::arg("shifts"),
            py::arg("positions"),
            py::arg("cell"),
            py::arg("build_cutoff"))
        .def(
            "prepare_periodic_graph",
            [](FourierD3Kokkos<Precision>& self,
               const py::array& numbers,
               const py::array& positions,
               const py::array& cell,
               const py::array& inverse_cell,
               const double build_cutoff) {
                require_rank(numbers, "numbers", 1);
                require_columns(positions, "positions", 3);
                require_columns(cell, "cell", 3);
                require_columns(inverse_cell, "inverse_cell", 3);
                const index_array number_values = index_buffer(numbers, "numbers");
                const float_array position_values = float_buffer(positions, "positions");
                const float_array cell_values = float_buffer(cell, "cell");
                const float_array inverse_values = float_buffer(inverse_cell, "inverse_cell");
                self.prepare_periodic_graph(
                    std::span<const std::int32_t>(number_values.data(), number_values.size()),
                    std::span<const double>(position_values.data(), position_values.size()),
                    std::span<const double>(cell_values.data(), cell_values.size()),
                    std::span<const double>(inverse_values.data(), inverse_values.size()),
                    build_cutoff);
            },
            py::arg("numbers"),
            py::arg("positions"),
            py::arg("cell"),
            py::arg("inverse_cell"),
            py::arg("build_cutoff"))
        .def(
            "update_geometry",
            [](FourierD3Kokkos<Precision>& self,
               const py::array& positions,
               const py::array& cell,
               const py::array& inverse_cell) {
                require_columns(positions, "positions", 3);
                require_columns(cell, "cell", 3);
                require_columns(inverse_cell, "inverse_cell", 3);
                const float_array position_values = float_buffer(positions, "positions");
                const float_array cell_values = float_buffer(cell, "cell");
                const float_array inverse_values = float_buffer(inverse_cell, "inverse_cell");
                self.update_geometry(
                    std::span<const double>(position_values.data(), position_values.size()),
                    std::span<const double>(cell_values.data(), cell_values.size()),
                    std::span<const double>(inverse_values.data(), inverse_values.size()));
            },
            py::arg("positions"),
            py::arg("cell"),
            py::arg("inverse_cell"))
        .def(
            "compute",
            [](FourierD3Kokkos<Precision>& self,
               const bool compute_forces,
               const bool compute_stress) {
                self.compute(compute_forces, compute_stress);
            },
            py::arg("compute_forces"),
            py::arg("compute_stress"))
        .def(
            "result",
            [](const FourierD3Kokkos<Precision>& self, const py::object& properties) {
                return result_report<Precision>(self, properties);
            },
            py::arg("properties") = py::none())
        .def("graph_diagnostics", [](const FourierD3Kokkos<Precision>& self) {
            return graph_report(self.graph_diagnostics());
        })
        .def("diagnostics", [](const FourierD3Kokkos<Precision>& self) {
            return diagnostics_report(self.diagnostics());
        })
        .def(
            "set_timing_enabled",
            [](FourierD3Kokkos<Precision>& self, const bool enabled) {
                self.set_timing_enabled(enabled);
            },
            py::arg("enabled"))
        .def("timing_enabled", [](const FourierD3Kokkos<Precision>& self) {
            return self.timing_enabled();
        })
        // None when timing was never enabled, so "not collected" is not reported
        // as a zero-valued measurement.
        .def("timing", [](const FourierD3Kokkos<Precision>& self) -> py::object {
            if (!self.timing_enabled()) {
                return py::none();
            }
            return binding::stage_timings_report(self.timings());
        });
}

} // namespace

void bind_fourier_d3_evaluator(py::module_& module)
{
    module.def(
        "_fourier_d3_canonical_tables",
        &canonical_tables,
        "Canonical D3 tables, their content digests, and their provenance fingerprint.");
    bind_evaluator_class<double>(module, "FourierD3Kokkos");
    bind_evaluator_class<float>(module, "FourierD3KokkosFloat");
}
