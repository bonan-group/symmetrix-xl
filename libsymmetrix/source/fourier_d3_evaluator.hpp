// Private dtype-specific native evaluator of the reciprocal-space FourierD3
// correction and its graph lifecycle.
//
// This is the connection layer.  It composes the reverse orchestrator -
// which already owns the particle-mesh forward path and the
// coordination workspace - behind the graph/update/compute/result protocol the
// Python correction runtime consumes, and it owns the pieces the reverse path
// deliberately does not: the short coordination graph's topology, the canonical
// D3 table payload, the active-species decomposition, the rank windowing, and
// the resolved mesh.
//
// Nothing here is a public capability.  The seam is bound under private names
// and this translation unit exists only where the FFT build capability found an FFT
// provider.
//
// Two invariants shape the implementation:
//
// * One native state.  A successful compute publishes the energy, the atomic
//   energies, and - when asked - the forces and the stress from one generation
//   of the same inputs.  Every accepted input change advances a generation, and
//   a result that belongs to an earlier generation is never served.  A failed
//   preparation or compute publishes nothing, so the previous generation stays
//   readable and unchanged.
// * One geometry authority.  Positions, cell, and edge distances are re-derived
//   from the retained shifts on every update and re-checked by the engine, so a
//   stale topology can never be accepted silently.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dispersion_parameters.hpp"
#include "fourier_d3_reciprocal.hpp"
#include "fourier_d3_reverse.hpp"
#include "neighbor_graph_kokkos.hpp"
#include "tools_kokkos.hpp"

namespace symmetrix::fourier_d3 {

// The narrowest B-spline assignment the mesh path supports.  The engine enforces
// the same bound; it is repeated here so the binding refuses an unsupported order
// before any native state is built.
inline constexpr std::int64_t kMinimumSplineOrder = 2;

// Becke-Johnson damping of one functional, in the units the forward path takes:
// `a1` in inverse Angstrom, `a2` in Angstrom.
struct FourierD3Damping {
    double a1 = 0.0;
    double a2_angstrom = 0.0;
    double s6 = 0.0;
    double s8 = 0.0;

    friend bool operator==(const FourierD3Damping&, const FourierD3Damping&) = default;
};

// The graph half of the runtime diagnostics.  The Python runtime reads the
// backend, node/edge counts, generation, and build cutoff from this one record.
struct FourierD3GraphDiagnostics {
    std::string graph_backend;
    std::uint64_t generation = 0;
    std::uint64_t geometry_update_count = 0;
    std::int64_t num_nodes = 0;
    std::int64_t num_edges = 0;
    double build_cutoff = 0.0;
};

// The evaluator half of the runtime diagnostics: the resolved configuration, the
// resolved mesh, the active decomposition, the cache and generation state, the
// itemized memory that reconciles with the byte totals, and - when it was opted
// into - the stage timing.  Every field names the state it reports; nothing here
// claims a backend this build did not qualify, and nothing is synthesized from
// the platform.
struct FourierD3Diagnostics {
    std::string precision;
    std::string functional;
    std::string table_fingerprint;
    FourierD3Damping damping;
    double coordination_cutoff = 0.0;
    std::int64_t spline_order = 0;
    std::vector<std::int32_t> active_species;
    std::int64_t retained_rank = 0;
    std::int64_t rank_total = 0;
    std::int64_t rank_chunk = 0;
    // Number of rank windows one evaluation runs: ceil(rank_total / rank_chunk).
    std::int64_t rank_chunk_count = 0;
    std::int64_t n_references = 0;
    double decomposition_max_relative_error = 0.0;
    bool decomposition_tolerance_met = false;
    std::array<std::int64_t, 3> mesh{0, 0, 0};
    std::int64_t mesh_points = 0;
    // Per-axis spacing of the resolved mesh measured on the retained cell
    // (Angstrom).  It follows the cell, so a cell update that keeps an explicit
    // mesh changes this without invalidating the mesh domain.
    std::array<double, 3> mesh_spacing{0.0, 0.0, 0.0};
    // Committed mesh-domain invalidations: the number of times the resolved
    // dimensions changed.  It counts the first resolution, and a cell update that
    // keeps the mesh leaves it alone.
    std::uint64_t mesh_invalidation_count = 0;
    std::uint64_t active_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    // The itemization that reconciles exactly with the two totals above.  Each
    // entry is named `<owner>.<buffer>` with owner one of `reverse`, `forward`,
    // `mesh`, `kernel`, or `coordination`, so every byte is attributable to the
    // owner that holds it and none is counted twice.
    std::vector<BufferUsage> memory_components;
    // Bytes the held FFT plans retain through the allocation each plan owns.  A
    // plan that still holds an allocation the storage has already replaced
    // reports that older allocation's bytes, not the current storage capacity.
    std::uint64_t plan_retained_bytes = 0;
    std::array<FftPlanUsage, plan_role_count> plans;
    std::uint64_t reservations = 0;
    std::uint64_t allocations = 0;
    std::uint64_t plan_builds = 0;
    std::uint64_t plan_reuses = 0;
    std::uint64_t plan_releases = 0;
    std::uint64_t plan_key_rebuilds = 0;
    std::uint64_t kernel_key_hits = 0;
    std::uint64_t kernel_key_misses = 0;
    std::uint64_t kernel_key_rebuilds = 0;
    std::int64_t stencil_points = 0;
    std::uint64_t input_generation = 0;
    std::uint64_t published_generation = 0;
    bool computed = false;
    bool result_available = false;
    bool forces_available = false;
    bool stress_available = false;
    bool engine_loaded = false;
    // Opt-in stage timing.  `timing_enabled` false means no interval was ever
    // measured, so the record is absent rather than zero-valued.
    bool timing_enabled = false;
    FourierD3StageTimings timings;
};

// The shape signature of the engine's reservation.  Every entry is a function of
// the mesh, the atom and edge counts, the species and reference layout, and the
// rank windowing, so comparing it is exactly "would `reserve` allocate again".
struct FourierD3EngineShape {
    std::int64_t nx = 0;
    std::int64_t ny = 0;
    std::int64_t nz = 0;
    std::int64_t nodes = 0;
    std::int64_t edges = 0;
    std::int64_t n_species = 0;
    std::int64_t n_ref = 0;
    std::int64_t rank_total = 0;
    std::int64_t rank_chunk = 0;
    std::int64_t spline_order = 0;

    friend bool operator==(
        const FourierD3EngineShape&, const FourierD3EngineShape&) = default;
};

template <typename Float>
class FourierD3Kokkos {
public:
    using value_type = Float;

    FourierD3Kokkos(
        std::string functional,
        double coordination_cutoff,
        std::int64_t spline_order,
        std::int64_t rank_chunk,
        std::string table_fingerprint,
        std::span<const double> radii,
        std::span<const double> q_table);

    // The resolved mesh.  Owned by the Python runtime's policy; the evaluator
    // never derives one of its own.
    void set_mesh(std::span<const std::int64_t> mesh);

    // The active-species decomposition of the canonical table, as produced by the
    // decomposition contract.  `v_q` is (n_species, n_ref, rank), `cn_ref` and `valid` are
    // (n_species, n_ref), and `valid` is a boolean mask.
    void set_decomposition(
        std::span<const std::int32_t> species,
        std::span<const double> eigenvalues,
        std::span<const double> v_q,
        std::span<const double> cn_ref,
        std::span<const bool> valid,
        double max_relative_error,
        bool tolerance_met);

    // Receiver-major host graph, built by the shared Python builder at
    // `coordination_cutoff + skin`.
    void prepare_host_graph(
        std::span<const std::int32_t> numbers,
        std::span<const std::int32_t> receiver_offsets,
        std::span<const std::int32_t> sources,
        std::span<const std::int32_t> shifts,
        std::span<const double> positions,
        std::span<const double> cell,
        double build_cutoff);

    // Periodic graph, built natively at `coordination_cutoff + skin`.
    void prepare_periodic_graph(
        std::span<const std::int32_t> numbers,
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> inverse_cell,
        double build_cutoff);

    // Refresh the geometry of the loaded topology.  The edge distances follow
    // from the retained shifts and the new positions and cell.
    void update_geometry(
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> inverse_cell);

    void compute(bool compute_forces, bool compute_stress);

    [[nodiscard]] bool result_available() const noexcept
    {
        return published_generation_ != 0 && published_generation_ == input_generation_;
    }
    [[nodiscard]] bool forces_available() const noexcept
    {
        return published_generation_ != 0 && forces_available_;
    }
    [[nodiscard]] bool stress_available() const noexcept
    {
        return published_generation_ != 0 && stress_available_;
    }

    [[nodiscard]] double energy() const;
    [[nodiscard]] std::vector<double> atomic_energies() const;
    [[nodiscard]] std::vector<Float> forces() const;
    [[nodiscard]] std::vector<double> stress() const;

    [[nodiscard]] FourierD3GraphDiagnostics graph_diagnostics() const;
    [[nodiscard]] FourierD3Diagnostics diagnostics() const;

    // -- opt-in stage timing ----------------------------------------------

    // Disabled by default, and propagated to the composed engine so one opt-in
    // covers every stage.  A disabled evaluator records nothing, so the record is
    // absent rather than a plausible zero.
    void set_timing_enabled(const bool enabled) noexcept
    {
        timing_enabled_ = enabled;
        reverse_.set_timing_enabled(enabled);
    }
    [[nodiscard]] bool timing_enabled() const noexcept { return timing_enabled_; }
    // Every stage's last interval, cumulative interval, and completed count, in
    // seconds.  A stage that never ran reports count zero.
    [[nodiscard]] FourierD3StageTimings timings() const;

private:
    void require_payload(std::string_view what) const;
    void require_topology(std::string_view what) const;
    void require_mesh(std::string_view what) const;

    // The flat, exactly reconcilable itemization the diagnostic record carries:
    // every byte of the engine's totals appears once, labelled by its owner.
    [[nodiscard]] static std::vector<BufferUsage> memory_components_of(
        const ReverseUsage& usage);

    // Reserve the composed engine if the shape changed, then apply every input
    // the engine has not seen.  A reservation replaces every view the engine
    // owns - including the damping coefficients - so all of them are re-applied.
    void refresh_engine();
    [[nodiscard]] std::vector<double> eigenvalue_window(std::int64_t begin) const;
    [[nodiscard]] std::vector<double> edge_distances(
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const std::int32_t> receiver_offsets,
        std::span<const std::int32_t> sources,
        std::span<const std::int32_t> shifts) const;
    [[nodiscard]] std::vector<double> species_radii(
        std::span<const std::int32_t> numbers) const;
    [[nodiscard]] std::vector<double> species_q_table() const;
    [[nodiscard]] static std::uint64_t payload_signature(
        std::span<const std::int32_t> species,
        std::span<const double> eigenvalues,
        std::span<const double> v_q,
        std::span<const double> cn_ref,
        std::span<const bool> valid,
        double max_relative_error,
        bool tolerance_met);
    // The atom and directed-edge counts a validated candidate topology would have.
    struct TopologyExtents {
        std::int64_t nodes = 0;
        std::int64_t edges = 0;
    };
    // The topology values an engine push reads.  A preparation pushes a candidate it has
    // validated but not yet installed; every other path pushes the retained topology, so
    // there is one push order and one place where the engine can refuse an input.
    struct TopologyView {
        std::span<const std::int32_t> numbers;
        std::span<const std::int32_t> receiver_offsets;
        std::span<const std::int32_t> sources;
        std::span<const std::int32_t> shifts;
        std::span<const double> positions;
        std::span<const double> cell;
        std::span<const double> distances;
    };
    // Pure validation of a candidate topology: it touches no member, so a rejected
    // candidate leaves the previous topology exactly as it was.
    [[nodiscard]] TopologyExtents validate_topology(
        std::span<const std::int32_t> numbers,
        std::span<const std::int32_t> receiver_offsets,
        std::span<const std::int32_t> sources,
        std::span<const std::int32_t> shifts,
        std::span<const double> positions,
        std::span<const double> cell) const;
    // Install an already validated and already pushed topology.  Every value is
    // assigned here, so this step does not allocate, does not throw, and does not touch
    // the engine: a refusal cannot leave it half done.
    void install_topology(
        std::span<const std::int32_t> numbers,
        std::span<const std::int32_t> receiver_offsets,
        std::span<const std::int32_t> sources,
        std::span<const std::int32_t> shifts,
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> distances,
        double build_cutoff,
        std::string_view backend);
    // The engine reservation for a candidate topology, which is one of the two steps of
    // a preparation that can be refused.  It runs before the candidate is pushed, so a
    // refused reservation installs nothing.
    void reserve_engine_for(TopologyExtents candidate);
    // Push a topology to the engine.  This is the other step a preparation can be
    // refused by, and it runs before the candidate is installed.
    void push_topology(const TopologyView& topology);
    [[nodiscard]] TopologyView retained_topology() const;
    [[nodiscard]] FourierD3EngineShape engine_shape_of(TopologyExtents candidate) const;
    void ensure_engine_for_retained_topology();
    // Commit the prospective domains, which is a move and cannot throw, and report
    // whether a payload was committed.  A preparation has already pushed that payload;
    // compute uses the report to apply a payload that was committed without one.
    bool commit_domains();
    [[nodiscard]] bool mesh_available() const noexcept
    {
        return prospective_domains_.mesh_set || domains_.mesh_set;
    }
    [[nodiscard]] bool payload_available() const noexcept
    {
        return prospective_domains_.payload_set || domains_.payload_set;
    }
    [[nodiscard]] const std::array<std::int64_t, 3>& effective_mesh() const noexcept
    {
        return prospective_domains_.mesh_set ? prospective_domains_.mesh : domains_.mesh;
    }
    [[nodiscard]] const std::vector<std::int32_t>& effective_species() const noexcept
    {
        return prospective_domains_.payload_set ? prospective_domains_.species
                                                : domains_.species;
    }
    [[nodiscard]] const std::vector<double>& effective_cn_ref() const noexcept
    {
        return prospective_domains_.payload_set ? prospective_domains_.cn_ref
                                                : domains_.cn_ref;
    }
    [[nodiscard]] const std::vector<double>& effective_v_q() const noexcept
    {
        return prospective_domains_.payload_set ? prospective_domains_.padded_v_q
                                                : domains_.padded_v_q;
    }
    [[nodiscard]] std::span<const bool> effective_valid() const noexcept
    {
        return prospective_domains_.payload_set
            ? std::span<const bool>(
                  prospective_domains_.valid.get(), prospective_domains_.valid_size)
            : std::span<const bool>(domains_.valid.get(), domains_.valid_size);
    }
    [[nodiscard]] std::int64_t effective_rank_total() const noexcept
    {
        return prospective_domains_.payload_set ? prospective_domains_.rank_total
                                                : domains_.rank_total;
    }
    [[nodiscard]] std::int64_t effective_rank_chunk() const noexcept
    {
        return prospective_domains_.payload_set ? prospective_domains_.rank_chunk
                                                : domains_.rank_chunk;
    }
    [[nodiscard]] std::int64_t effective_n_references() const noexcept
    {
        return prospective_domains_.payload_set ? prospective_domains_.n_references
                                                : domains_.n_references;
    }
    [[nodiscard]] bool has_active_species() const noexcept
    {
        return (prospective_domains_.payload_set ? prospective_domains_ : domains_)
            .has_active_species();
    }
    void validate_positions(std::span<const double> positions, std::int64_t nodes) const;
    void require_cell(std::span<const double> cell) const;
    void require_inverse_cell(
        std::span<const double> cell, std::span<const double> inverse_cell) const;
    static void require_build_cutoff(double build_cutoff);
    static void require_numbers(std::span<const std::int32_t> numbers, bool allow_padding);

    // The library is compiled with the historical -ffast-math policy, under which
    // ``std::isfinite`` is optimized to a constant and cannot detect a NaN or an
    // infinity.  Finiteness is therefore read from the IEEE-754 exponent bits, exactly
    // as the shared cache-key check does.
    [[nodiscard]] static bool is_finite(double value);

    // Declared first so that the composed engine - which owns every Kokkos view
    // reachable from this object - is destroyed before this object stops
    // counting as a live Kokkos owner.
    KokkosLiveObjectGuard kokkos_live_object_guard_;

    // The configured constants.  They are constructor inputs and never change.
    std::string functional_;
    double coordination_cutoff_ = 0.0;
    std::int64_t spline_order_ = 0;
    std::int64_t configured_chunk_ = 0;
    std::string table_fingerprint_;
    FourierD3Damping damping_{};
    std::vector<double> radii_;
    std::vector<double> q_table_;

    ReciprocalReverse<Float> reverse_;

    // One resolved mesh and one active-species payload.  The payload is copied, so a
    // caller's array lifetime is irrelevant after the call, and the zero-padded factor
    // table is what the engine indexes when the rank is chunked.  An empty payload is a
    // system with no active species: a mesh with no decomposition, whose dispersion
    // energy is exactly zero.
    struct DomainState {
        bool mesh_set = false;
        std::array<std::int64_t, 3> mesh{0, 0, 0};
        bool payload_set = false;
        std::vector<std::int32_t> species;
        std::vector<double> eigenvalues;
        std::vector<double> padded_v_q;
        std::vector<double> cn_ref;
        std::unique_ptr<bool[]> valid;
        std::size_t valid_size = 0;
        std::int64_t rank = 0;
        std::int64_t rank_total = 0;
        std::int64_t rank_chunk = 0;
        std::int64_t n_references = 0;
        double max_relative_error = 0.0;
        bool tolerance_met = false;
        std::uint64_t signature = 0;

        [[nodiscard]] bool has_active_species() const noexcept
        {
            return payload_set && !species.empty();
        }
    };

    // The committed domains are what every readback reports and what every engine push
    // uses, and the prospective ones are what a caller has pushed but no preparation has
    // committed yet.  A preparation is the only commit point, so a refused graph - before
    // or after the reservation - leaves the caller's view and this object's view of what
    // is loaded in agreement, and there is no rollback to get wrong.
    DomainState domains_;
    DomainState prospective_domains_;

    // The topology and the geometry it measures.  The host graph is retained
    // because `update_geometry` re-derives the edge distances from it.
    bool topology_set_ = false;
    std::string graph_backend_;
    std::vector<std::int32_t> numbers_;
    std::vector<std::int32_t> receiver_offsets_;
    std::vector<std::int32_t> sources_;
    std::vector<std::int32_t> shifts_;
    std::vector<double> positions_;
    std::vector<double> cell_;
    std::vector<double> distances_;
    double build_cutoff_ = 0.0;
    std::uint64_t graph_generation_ = 0;
    std::uint64_t geometry_update_count_ = 0;

    // Bookkeeping.
    std::uint64_t input_generation_ = 0;
    std::uint64_t published_generation_ = 0;
    bool payload_dirty_ = false;
    bool geometry_dirty_ = false;
    bool graph_dirty_ = false;
    bool damping_dirty_ = true;
    bool engine_loaded_ = false;
    std::uint64_t mesh_invalidation_count_ = 0;
    FourierD3EngineShape engine_shape_{};
    // Opt-in stage timing.  These three stages are the evaluator's own; the
    // composed engine's six live in the engine, so no stage is timed twice.
    bool timing_enabled_ = false;
    StageTimer preparation_timer_;
    StageTimer result_reduction_timer_;
    StageTimer result_export_timer_;
    bool forces_available_ = false;
    bool stress_available_ = false;
    double energy_published_ = 0.0;
    std::vector<double> atomic_published_;
    std::vector<Float> forces_published_;
    std::vector<double> stress_published_;
    // Scratch accumulators, so a failed window can never be read as a result.
    std::vector<double> atomic_scratch_;
    std::vector<double> forces_scratch_;
    std::vector<double> stress_scratch_;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

template <typename Float>
FourierD3Kokkos<Float>::FourierD3Kokkos(
    std::string functional,
    const double coordination_cutoff,
    const std::int64_t spline_order,
    const std::int64_t rank_chunk,
    std::string table_fingerprint,
    const std::span<const double> radii,
    const std::span<const double> q_table)
    : functional_(std::move(functional)),
      coordination_cutoff_(coordination_cutoff),
      spline_order_(spline_order),
      configured_chunk_(rank_chunk),
      table_fingerprint_(std::move(table_fingerprint)),
      radii_(radii.begin(), radii.end()),
      q_table_(q_table.begin(), q_table.end()),
      reverse_(MeshLayout::right, rank_chunk)
{
    // Until a decomposition is loaded there is nothing to narrow the chunk to, so the
    // effective chunk is the configured bound.
    domains_.rank_chunk = rank_chunk;
    if (table_fingerprint_.empty()) {
        throw std::invalid_argument(
            "FourierD3 needs a non-empty canonical-table fingerprint; the "
            "decomposition it was built from must be identifiable.");
    }
    if (!(coordination_cutoff_ > 0.0) || !is_finite(coordination_cutoff_)) {
        throw std::invalid_argument(
            "FourierD3 coordination cutoff must be finite and positive, got "
            + detail::format_scalar(coordination_cutoff_) + ".");
    }
    if (spline_order_ < kMinimumSplineOrder || spline_order_ > bspline::max_order) {
        throw std::invalid_argument(
            "FourierD3 spline order must be between "
            + std::to_string(kMinimumSplineOrder) + " and "
            + std::to_string(bspline::max_order) + ", got "
            + std::to_string(spline_order_) + ".");
    }
    // A non-positive rank chunk is refused by the reverse engine's channel bound,
    // which is the same extent and is checked when that engine is constructed.
    const auto expected = static_cast<std::size_t>(dispersion::d3_max_atomic_number + 1);
    if (radii_.size() != expected) {
        throw std::invalid_argument(
            "FourierD3 covalent radii hold one value per atomic number up to "
            + std::to_string(dispersion::d3_max_atomic_number) + "; "
            + std::to_string(radii_.size()) + " were supplied, expected "
            + std::to_string(expected) + ".");
    }
    if (q_table_.size() != expected) {
        throw std::invalid_argument(
            "FourierD3 q_table holds one r2r4 value per atomic number up to "
            + std::to_string(dispersion::d3_max_atomic_number) + "; "
            + std::to_string(q_table_.size()) + " were supplied, expected "
            + std::to_string(expected) + ".");
    }
    for (std::size_t index = 0; index < expected; ++index) {
        if (!is_finite(radii_[index]) || radii_[index] < 0.0) {
            throw std::invalid_argument(
                "FourierD3 covalent radii must be finite and non-negative; entry "
                + std::to_string(index) + " is "
                + detail::format_scalar(radii_[index]) + ".");
        }
        if (!is_finite(q_table_[index]) || q_table_[index] < 0.0) {
            throw std::invalid_argument(
                "FourierD3 q_table entries must be finite and non-negative; entry "
                + std::to_string(index) + " is "
                + detail::format_scalar(q_table_[index]) + ".");
        }
    }

    // FourierD3 is Becke-Johnson only, and the parameters come from the same
    // generated table real-space D3 uses, so the two models cannot disagree
    // about a functional.
    const dispersion::D3ParameterSet* parameters = nullptr;
    try {
        parameters = &dispersion::lookup_d3_parameters("bj", functional_);
    } catch (const std::exception& error) {
        throw std::invalid_argument(
            "FourierD3 has no Becke-Johnson parameters for functional '"
            + functional_ + "': " + error.what());
    }
    damping_ = FourierD3Damping{
        parameters->parameters.rs6,
        parameters->parameters.rs18 * dispersion::bohr_to_angstrom,
        parameters->parameters.s6,
        parameters->parameters.s18,
    };
    if (!(damping_.a2_angstrom > 0.0)) {
        throw std::invalid_argument(
            "FourierD3 BJ damping radius a2 of functional '" + functional_
            + "' is not positive after unit conversion; the reciprocal kernel is "
            "undefined there.");
    }
}

template <typename Float>
void FourierD3Kokkos<Float>::set_mesh(const std::span<const std::int64_t> mesh)
{
    if (mesh.size() != 3) {
        throw std::invalid_argument(
            "FourierD3 mesh must hold three extents, got "
            + std::to_string(mesh.size()) + ".");
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (mesh[axis] < spline_order_) {
            throw std::invalid_argument(
                "FourierD3 mesh axis " + std::to_string(axis) + " holds "
                + std::to_string(mesh[axis]) + " nodes, fewer than the spline order "
                + std::to_string(spline_order_) + ".");
        }
        if (mesh[axis] < 1) {
            throw std::invalid_argument(
                "FourierD3 mesh extents must be positive; axis "
                + std::to_string(axis) + " is " + std::to_string(mesh[axis]) + ".");
        }
    }
    const std::array<std::int64_t, 3> requested{mesh[0], mesh[1], mesh[2]};
    if (requested == effective_mesh() && mesh_available()) {
        return;
    }
    // The mesh is prospective until the preparation it belongs to has succeeded, so a
    // caller that pushes a mesh for a graph that is then refused has not changed what
    // this object reports or what the next preparation starts from.  Nothing in the
    // diagnostic record moves until the commit, which is what makes a refused
    // preparation indistinguishable from one that was never attempted.
    prospective_domains_.mesh = requested;
    prospective_domains_.mesh_set = true;
}

template <typename Float>
void FourierD3Kokkos<Float>::set_decomposition(
    const std::span<const std::int32_t> species,
    const std::span<const double> eigenvalues,
    const std::span<const double> v_q,
    const std::span<const double> cn_ref,
    const std::span<const bool> valid,
    const double max_relative_error,
    const bool tolerance_met)
{
    // A system with no active species has no decomposition to retain: every atom is
    // either absent or padding, so the dispersion energy is exactly zero.  The payload is
    // then consistently empty rather than a partially filled one, and the engine is never
    // reserved for it.
    if (species.empty()) {
        if (!eigenvalues.empty() || !v_q.empty() || !cn_ref.empty() || !valid.empty()) {
            throw std::invalid_argument(
                "FourierD3 decomposition with no active species must be empty "
                "throughout; a payload with an empty species list but "
                + std::to_string(eigenvalues.size()) + " eigenvalue(s), "
                + std::to_string(v_q.size()) + " factor value(s), "
                + std::to_string(cn_ref.size()) + " reference value(s), and "
                + std::to_string(valid.size()) + " validity flag(s) describes a "
                "decomposition that has no owner.");
        }
        if (!is_finite(max_relative_error) || max_relative_error < 0.0) {
            throw std::invalid_argument(
                "FourierD3 empty decomposition error must be finite and non-negative, "
                "got " + detail::format_scalar(max_relative_error) + ".");
        }
        const std::uint64_t empty_signature =
            payload_signature(species, eigenvalues, v_q, cn_ref, valid, max_relative_error, tolerance_met);
        if (payload_available()
            && (prospective_domains_.payload_set ? prospective_domains_.signature
                                                 : domains_.signature)
                == empty_signature) {
            return;
        }
        prospective_domains_.species.clear();
        prospective_domains_.eigenvalues.clear();
        prospective_domains_.padded_v_q.clear();
        prospective_domains_.cn_ref.clear();
        prospective_domains_.valid.reset();
        prospective_domains_.valid_size = 0;
        prospective_domains_.rank = 0;
        prospective_domains_.rank_total = 0;
        prospective_domains_.rank_chunk = 0;
        prospective_domains_.n_references = 0;
        prospective_domains_.max_relative_error = max_relative_error;
        prospective_domains_.tolerance_met = tolerance_met;
        prospective_domains_.signature = empty_signature;
        prospective_domains_.payload_set = true;
        return;
    }
    for (std::size_t index = 0; index < species.size(); ++index) {
        if (species[index] < 1
            || species[index] > dispersion::d3_max_atomic_number) {
            throw std::invalid_argument(
                "FourierD3 decomposition species must be real atomic numbers up to "
                + std::to_string(dispersion::d3_max_atomic_number) + "; channel "
                + std::to_string(index) + " is "
                + std::to_string(species[index]) + ".");
        }
        if (index > 0 && species[index] <= species[index - 1]) {
            throw std::invalid_argument(
                "FourierD3 decomposition species must be unique and ascending; "
                "channel " + std::to_string(index) + " repeats or reorders "
                + std::to_string(species[index]) + ".");
        }
    }
    if (eigenvalues.empty()) {
        throw std::invalid_argument(
            "FourierD3 decomposition needs at least one retained rank slot.");
    }
    for (const double value : eigenvalues) {
        if (!is_finite(value)) {
            throw std::invalid_argument(
                "FourierD3 decomposition eigenvalues must be finite.");
        }
    }
    if (cn_ref.size() % species.size() != 0) {
        throw std::invalid_argument(
            "FourierD3 reference coordination table must hold one row per species; "
            + std::to_string(cn_ref.size()) + " values for "
            + std::to_string(species.size()) + " species.");
    }
    const auto n_ref = static_cast<std::int64_t>(cn_ref.size() / species.size());
    if (n_ref < 1 || n_ref > dispersion::d3_max_coordination_references) {
        throw std::invalid_argument(
            "FourierD3 species hold between 1 and "
            + std::to_string(dispersion::d3_max_coordination_references)
            + " reference slots; the supplied table has " + std::to_string(n_ref)
            + " per species.");
    }
    const auto slots = static_cast<std::size_t>(cn_ref.size());
    if (valid.size() != slots) {
        throw std::invalid_argument(
            "FourierD3 validity mask holds one flag per reference slot; "
            + std::to_string(valid.size()) + " were supplied for "
            + std::to_string(slots) + " slots.");
    }
    for (const double value : cn_ref) {
        if (!is_finite(value)) {
            throw std::invalid_argument(
                "FourierD3 reference coordination numbers must be finite.");
        }
    }
    const auto rank = static_cast<std::int64_t>(eigenvalues.size());
    const auto expected_v_q = slots * static_cast<std::size_t>(rank);
    if (v_q.size() != expected_v_q) {
        throw std::invalid_argument(
            "FourierD3 low-rank factors must have shape (n_species, n_ref, rank); "
            + std::to_string(v_q.size()) + " values were supplied for "
            + std::to_string(expected_v_q) + " (species " + std::to_string(species.size())
            + ", reference slots " + std::to_string(n_ref) + ", rank "
            + std::to_string(rank) + ").");
    }
    for (const double value : v_q) {
        if (!is_finite(value)) {
            throw std::invalid_argument(
                "FourierD3 low-rank factors must be finite.");
        }
    }
    if (!is_finite(max_relative_error) || max_relative_error < 0.0) {
        throw std::invalid_argument(
            "FourierD3 decomposition error must be finite and non-negative, got "
            + detail::format_scalar(max_relative_error) + ".");
    }

    const std::uint64_t signature =
        payload_signature(species, eigenvalues, v_q, cn_ref, valid, max_relative_error, tolerance_met);
    if (payload_available()
        && (prospective_domains_.payload_set ? prospective_domains_.signature
                                             : domains_.signature)
            == signature) {
        return;
    }

    // Build everything that can fail before anything is staged, so a rejected payload
    // leaves the previous one exactly as it was.  The payload is prospective like the
    // mesh, so a refused preparation leaves the record exactly as it was and the
    // candidate survives for the retry.
    const std::int64_t chunk = std::min(configured_chunk_, rank);
    const std::int64_t total = ((rank + chunk - 1) / chunk) * chunk;
    std::vector<double> padded(static_cast<std::size_t>(slots) * static_cast<std::size_t>(total), 0.0);
    for (std::size_t row = 0; row < slots; ++row) {
        for (std::int64_t slot = 0; slot < rank; ++slot) {
            padded[row * static_cast<std::size_t>(total) + static_cast<std::size_t>(slot)] =
                v_q[row * static_cast<std::size_t>(rank) + static_cast<std::size_t>(slot)];
        }
    }
    std::unique_ptr<bool[]> valid_copy = std::make_unique<bool[]>(valid.size());
    std::copy(valid.begin(), valid.end(), valid_copy.get());

    prospective_domains_.species.assign(species.begin(), species.end());
    prospective_domains_.eigenvalues.assign(eigenvalues.begin(), eigenvalues.end());
    prospective_domains_.padded_v_q = std::move(padded);
    prospective_domains_.cn_ref.assign(cn_ref.begin(), cn_ref.end());
    prospective_domains_.valid = std::move(valid_copy);
    prospective_domains_.valid_size = valid.size();
    prospective_domains_.rank = rank;
    prospective_domains_.rank_total = total;
    prospective_domains_.rank_chunk = chunk;
    prospective_domains_.n_references = n_ref;
    prospective_domains_.max_relative_error = max_relative_error;
    prospective_domains_.tolerance_met = tolerance_met;
    prospective_domains_.signature = signature;
    prospective_domains_.payload_set = true;
}

template <typename Float>
bool FourierD3Kokkos<Float>::commit_domains()
{
    // A move, so the commit cannot fail and cannot leave a half-committed domain set.
    if (!prospective_domains_.mesh_set && !prospective_domains_.payload_set) {
        return false;
    }
    const bool payload_committed = prospective_domains_.payload_set;
    if (prospective_domains_.mesh_set) {
        // The mesh domain is invalidated exactly when the resolved dimensions change.
        // A cell update that keeps the mesh reaches this with identical values and
        // leaves the counter alone, which is what makes "the mesh domain survived"
        // observable rather than inferred from a rebuild count.
        if (!domains_.mesh_set || domains_.mesh != prospective_domains_.mesh) {
            ++mesh_invalidation_count_;
        }
        domains_.mesh = prospective_domains_.mesh;
        domains_.mesh_set = true;
    }
    if (prospective_domains_.payload_set) {
        domains_.species = std::move(prospective_domains_.species);
        domains_.eigenvalues = std::move(prospective_domains_.eigenvalues);
        domains_.padded_v_q = std::move(prospective_domains_.padded_v_q);
        domains_.cn_ref = std::move(prospective_domains_.cn_ref);
        domains_.valid = std::move(prospective_domains_.valid);
        domains_.valid_size = prospective_domains_.valid_size;
        domains_.rank = prospective_domains_.rank;
        domains_.rank_total = prospective_domains_.rank_total;
        domains_.rank_chunk = prospective_domains_.rank_chunk;
        domains_.n_references = prospective_domains_.n_references;
        domains_.max_relative_error = prospective_domains_.max_relative_error;
        domains_.tolerance_met = prospective_domains_.tolerance_met;
        domains_.signature = prospective_domains_.signature;
        domains_.payload_set = true;
    }
    prospective_domains_ = DomainState{};
    ++input_generation_;
    return payload_committed;
}

template <typename Float>
void FourierD3Kokkos<Float>::require_payload(const std::string_view what) const
{
    if (!payload_available()) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires the active-species decomposition; call set_decomposition "
            "first.");
    }
}

template <typename Float>
void FourierD3Kokkos<Float>::require_mesh(const std::string_view what) const
{
    if (!mesh_available()) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires the resolved mesh; call set_mesh first.");
    }
}

template <typename Float>
void FourierD3Kokkos<Float>::require_topology(const std::string_view what) const
{
    if (!topology_set_) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires a prepared coordination graph; call prepare_host_graph or "
            "prepare_periodic_graph first.");
    }
}

template <typename Float>
void FourierD3Kokkos<Float>::require_build_cutoff(const double build_cutoff)
{
    if (!is_finite(build_cutoff) || !(build_cutoff > 0.0)) {
        throw std::invalid_argument(
            "FourierD3 graph build cutoff must be finite and positive, got "
            + detail::format_scalar(build_cutoff) + ".");
    }
}

template <typename Float>
bool FourierD3Kokkos<Float>::is_finite(const double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template <typename Float>
void FourierD3Kokkos<Float>::require_numbers(
    const std::span<const std::int32_t> numbers, const bool allow_padding)
{
    // An empty system is a real topology: no atoms, no edges, and no active species.
    // Every number that is present still has to be a real one, or padding on the path
    // that carries it.
    const std::int32_t lowest = allow_padding ? 0 : 1;
    for (std::size_t index = 0; index < numbers.size(); ++index) {
        if (numbers[index] < lowest
            || numbers[index] > dispersion::d3_max_atomic_number) {
            throw std::invalid_argument(
                "FourierD3 atomic numbers must be in "
                + std::string(allow_padding ? "0.." : "1..")
                + std::to_string(dispersion::d3_max_atomic_number) + "; atom "
                + std::to_string(index) + " is "
                + std::to_string(numbers[index]) + ".");
        }
    }
}

template <typename Float>
void FourierD3Kokkos<Float>::require_cell(const std::span<const double> cell) const
{
    if (cell.size() != 9) {
        throw std::invalid_argument(
            "FourierD3 cell must hold nine row-major values, got "
            + std::to_string(cell.size()) + ".");
    }
    for (const double value : cell) {
        if (!is_finite(value)) {
            throw std::invalid_argument("FourierD3 cell must be finite.");
        }
    }
}

template <typename Float>
void FourierD3Kokkos<Float>::require_inverse_cell(
    const std::span<const double> cell, const std::span<const double> inverse_cell) const
{
    require_cell(cell);
    if (inverse_cell.size() != 9) {
        throw std::invalid_argument(
            "FourierD3 inverse cell must hold nine row-major values, got "
            + std::to_string(inverse_cell.size()) + ".");
    }
    for (const double value : inverse_cell) {
        if (!is_finite(value)) {
            throw std::invalid_argument("FourierD3 inverse cell must be finite.");
        }
    }
    // The engine derives its own inverse from the cell, so the supplied one is a
    // cross-check rather than an input: a caller that passed a stale or transposed
    // inverse is refused instead of being silently ignored.  The band is the one
    // the shared dispersion graph builder applies to the same product, so the two
    // components cannot disagree about which cells they accept.
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double product = 0.0;
            for (std::size_t inner = 0; inner < 3; ++inner) {
                product += cell[row * 3 + inner] * inverse_cell[inner * 3 + column];
            }
            const double expected = row == column ? 1.0 : 0.0;
            if (std::abs(product - expected) > 1e-9) {
                throw std::invalid_argument(
                    "FourierD3 inverse cell is inconsistent with the cell: the "
                    "product deviates from the identity by "
                    + detail::format_scalar(std::abs(product - expected))
                    + ", so the two describe different geometries.");
            }
        }
    }
}

template <typename Float>
void FourierD3Kokkos<Float>::validate_positions(
    const std::span<const double> positions, const std::int64_t nodes) const
{
    if (positions.size() != static_cast<std::size_t>(3 * nodes)) {
        throw std::invalid_argument(
            "FourierD3 positions must hold three coordinates per atom; "
            + std::to_string(positions.size()) + " were supplied for "
            + std::to_string(nodes) + " atom(s).");
    }
    for (const double value : positions) {
        if (!is_finite(value)) {
            throw std::invalid_argument("FourierD3 positions must be finite.");
        }
    }
}

template <typename Float>
auto FourierD3Kokkos<Float>::validate_topology(
    const std::span<const std::int32_t> numbers,
    const std::span<const std::int32_t> receiver_offsets,
    const std::span<const std::int32_t> sources,
    const std::span<const std::int32_t> shifts,
    const std::span<const double> positions,
    const std::span<const double> cell) const -> TopologyExtents
{
    const auto nodes = static_cast<std::int64_t>(numbers.size());
    validate_positions(positions, nodes);
    if (receiver_offsets.size() != static_cast<std::size_t>(nodes) + 1) {
        throw std::invalid_argument(
            "FourierD3 receiver offsets hold one entry per receiver plus one; "
            + std::to_string(receiver_offsets.size()) + " were supplied for "
            + std::to_string(nodes) + " receivers.");
    }
    if (receiver_offsets.front() != 0) {
        throw std::invalid_argument("FourierD3 receiver offsets must start at zero.");
    }
    for (std::size_t index = 0; index < receiver_offsets.size(); ++index) {
        if (receiver_offsets[index] < 0) {
            throw std::invalid_argument(
                "FourierD3 receiver offsets must not be negative.");
        }
        if (index > 0 && receiver_offsets[index] < receiver_offsets[index - 1]) {
            throw std::invalid_argument(
                "FourierD3 receiver offsets must be non-decreasing in receiver "
                "order.");
        }
    }
    const auto edges = static_cast<std::int64_t>(receiver_offsets.back());
    if (sources.size() != static_cast<std::size_t>(edges)) {
        throw std::invalid_argument(
            "FourierD3 sources hold one entry per directed edge; "
            + std::to_string(sources.size()) + " were supplied for "
            + std::to_string(edges) + " edges.");
    }
    for (std::size_t index = 0; index < sources.size(); ++index) {
        if (sources[index] < 0 || sources[index] >= nodes) {
            throw std::invalid_argument(
                "FourierD3 sources must index a receiver; edge "
                + std::to_string(index) + " names atom "
                + std::to_string(sources[index]) + " of " + std::to_string(nodes)
                + ".");
        }
    }
    if (shifts.size() != static_cast<std::size_t>(3 * edges)) {
        throw std::invalid_argument(
            "FourierD3 shifts hold three periodic-image integers per directed edge; "
            + std::to_string(shifts.size()) + " were supplied for "
            + std::to_string(edges) + " edges.");
    }
    return TopologyExtents{nodes, edges};
}

// Installs a topology that :meth:`validate_topology` has already accepted and that the
// engine has already accepted through :meth:`push_topology`.  Every value below is an
// assignment of something already validated, so this step does not touch the engine and
// cannot be refused: it is the last step of a preparation, and nothing can fail after it.
template <typename Float>
void FourierD3Kokkos<Float>::install_topology(
    const std::span<const std::int32_t> numbers,
    const std::span<const std::int32_t> receiver_offsets,
    const std::span<const std::int32_t> sources,
    const std::span<const std::int32_t> shifts,
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> distances,
    const double build_cutoff,
    const std::string_view backend)
{
    // Every value installed here is a pure function of arguments the caller has
    // already been told are valid, so the object moves from one consistent topology
    // to the next without an intermediate state.
    numbers_.assign(numbers.begin(), numbers.end());
    receiver_offsets_.assign(receiver_offsets.begin(), receiver_offsets.end());
    sources_.assign(sources.begin(), sources.end());
    shifts_.assign(shifts.begin(), shifts.end());
    positions_.assign(positions.begin(), positions.end());
    cell_.assign(cell.begin(), cell.end());
    distances_.assign(distances.begin(), distances.end());
    build_cutoff_ = build_cutoff;
    graph_backend_.assign(backend);
    topology_set_ = true;
    ++graph_generation_;
    ++geometry_update_count_;
    ++input_generation_;
}

template <typename Float>
std::vector<double> FourierD3Kokkos<Float>::edge_distances(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const std::int32_t> receiver_offsets,
    const std::span<const std::int32_t> sources,
    const std::span<const std::int32_t> shifts) const
{
    // The same expression the engine's own consistency check uses, in the same
    // order, so the derived length and the checked length cannot disagree.
    const auto edges = static_cast<std::int64_t>(sources.size());
    std::vector<double> distances(static_cast<std::size_t>(edges), 0.0);
    for (std::int64_t receiver = 0;
         receiver < static_cast<std::int64_t>(receiver_offsets.size()) - 1;
         ++receiver) {
        for (std::int64_t edge = receiver_offsets[static_cast<std::size_t>(receiver)];
             edge < receiver_offsets[static_cast<std::size_t>(receiver) + 1]; ++edge) {
            const std::int64_t source = sources[static_cast<std::size_t>(edge)];
            double length = 0.0;
            for (std::int64_t a = 0; a < 3; ++a) {
                double displacement =
                    positions[static_cast<std::size_t>(3 * source + a)]
                    - positions[static_cast<std::size_t>(3 * receiver + a)];
                for (std::int64_t b = 0; b < 3; ++b) {
                    displacement += static_cast<double>(
                                        shifts[static_cast<std::size_t>(3 * edge + b)])
                        * cell[static_cast<std::size_t>(3 * b + a)];
                }
                length += displacement * displacement;
            }
            distances[static_cast<std::size_t>(edge)] = std::sqrt(length);
        }
    }
    return distances;
}

template <typename Float>
std::vector<double> FourierD3Kokkos<Float>::species_radii(
    const std::span<const std::int32_t> numbers) const
{
    std::vector<double> radii(numbers.size(), 0.0);
    for (std::size_t index = 0; index < numbers.size(); ++index) {
        radii[index] = radii_[static_cast<std::size_t>(numbers[index])];
    }
    return radii;
}

template <typename Float>
std::vector<double> FourierD3Kokkos<Float>::species_q_table() const
{
    const std::vector<std::int32_t>& species = effective_species();
    std::vector<double> values(species.size(), 0.0);
    for (std::size_t index = 0; index < species.size(); ++index) {
        values[index] = q_table_[static_cast<std::size_t>(species[index])];
    }
    return values;
}

template <typename Float>
void FourierD3Kokkos<Float>::prepare_host_graph(
    const std::span<const std::int32_t> numbers,
    const std::span<const std::int32_t> receiver_offsets,
    const std::span<const std::int32_t> sources,
    const std::span<const std::int32_t> shifts,
    const std::span<const double> positions,
    const std::span<const double> cell,
    const double build_cutoff)
{
    // The host path is the one that carries padding atoms: a Z = 0 atom is a real
    // receiver with real edges, and every edge that touches it is inert because
    // the coordination kernel refuses a non-positive atomic number at either end.
    require_payload("preparing the host graph");
    require_mesh("preparing the host graph");
    require_numbers(numbers, true);
    require_build_cutoff(build_cutoff);
    require_cell(cell);
    const TopologyExtents candidate =
        validate_topology(numbers, receiver_offsets, sources, shifts, positions, cell);
    const std::vector<double> distances =
        edge_distances(positions, cell, receiver_offsets, sources, shifts);
    const TopologyView view{
        numbers, receiver_offsets, sources, shifts, positions, cell, distances};
    // Both fallible steps of a preparation run before the candidate is installed: the
    // reservation, which allocates, and the engine's own acceptance of the candidate's
    // cell, atoms, distances, and graph.  A refusal therefore leaves the retained graph,
    // its generations, the committed domains, and the published result exactly as they
    // were, and the candidate is never observable.
    reserve_engine_for(candidate);
    // A same-layout payload does not make reserve_engine_for rebuild the engine, but it
    // still replaces species/reference values.  Mark it only after reservation succeeds:
    // a refused reservation changes no state, while a refused push keeps the flag set so
    // the complete candidate payload is retried.
    if (prospective_domains_.payload_set) {
        payload_dirty_ = true;
    }
    geometry_dirty_ = true;
    graph_dirty_ = true;
    push_topology(view);
    install_topology(
        numbers,
        receiver_offsets,
        sources,
        shifts,
        positions,
        cell,
        distances,
        build_cutoff,
        "host");
    commit_domains();
}

template <typename Float>
void FourierD3Kokkos<Float>::prepare_periodic_graph(
    const std::span<const std::int32_t> numbers,
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const double build_cutoff)
{
    // The shared periodic builder requires every atomic number to be a real one,
    // so padding is refused here rather than given an invented channel; the host
    // path is the one that carries padding atoms.
    require_payload("preparing the periodic graph");
    require_mesh("preparing the periodic graph");
    require_numbers(numbers, false);
    require_build_cutoff(build_cutoff);
    require_inverse_cell(cell, inverse_cell);
    validate_positions(positions, static_cast<std::int64_t>(numbers.size()));

    // The shared builder enumerates the directed periodic graph; the evaluator
    // retains it, so the topology and the geometry it measures have one owner.
    const execution::KokkosNeighborGraph native =
        execution::build_periodic_neighbor_graph_kokkos(
            positions, cell, inverse_cell, build_cutoff, true);
    if (native.num_nodes != static_cast<int>(numbers.size())) {
        throw std::runtime_error(
            "FourierD3 periodic graph returned "
            + std::to_string(native.num_nodes) + " nodes for "
            + std::to_string(numbers.size()) + " atoms.");
    }
    const auto offsets = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), native.receiver_offsets);
    const auto sources = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), native.sources);
    const auto shifts = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), native.shifts);
    std::vector<std::int32_t> host_offsets(offsets.data(), offsets.data() + offsets.extent(0));
    std::vector<std::int32_t> host_sources(sources.data(), sources.data() + sources.extent(0));
    std::vector<std::int32_t> host_shifts(
        shifts.data(), shifts.data() + 3 * native.num_edges);
    // Same order as the host path: the candidate is validated and the engine is
    // reserved before anything is installed, so a refused preparation installs
    // nothing and the previous graph stays readable.
    const TopologyExtents candidate = validate_topology(
        numbers, host_offsets, host_sources, host_shifts, positions, cell);
    const std::vector<double> distances = edge_distances(
        positions, cell, host_offsets, host_sources, host_shifts);
    const TopologyView view{numbers,
                            host_offsets,
                            host_sources,
                            host_shifts,
                            positions,
                            cell,
                            distances};
    // Same order as the host path: the reservation and the engine's acceptance of the
    // candidate both run before anything is installed, so a refused preparation installs
    // nothing and the previous graph stays readable.
    reserve_engine_for(candidate);
    if (prospective_domains_.payload_set) {
        payload_dirty_ = true;
    }
    geometry_dirty_ = true;
    graph_dirty_ = true;
    push_topology(view);
    install_topology(
        numbers,
        host_offsets,
        host_sources,
        host_shifts,
        positions,
        cell,
        distances,
        build_cutoff,
        "kokkos");
    commit_domains();
}

template <typename Float>
void FourierD3Kokkos<Float>::update_geometry(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell)
{
    require_topology("updating the geometry");
    require_inverse_cell(cell, inverse_cell);
    // The atom count is fixed by the retained topology, so ``validate_positions`` already
    // refuses anything that is not three coordinates per retained atom.
    validate_positions(positions, static_cast<std::int64_t>(numbers_.size()));
    if (std::equal(positions.begin(), positions.end(), positions_.begin())
        && std::equal(cell.begin(), cell.end(), cell_.begin())
        && !prospective_domains_.mesh_set && !prospective_domains_.payload_set) {
        return;
    }
    // The distances are re-derived from the candidate positions, the candidate cell, and
    // the retained shifts, and the engine re-checks every one of them against its own
    // derivation before the object reports itself updated.  That push runs before the
    // candidate replaces the retained geometry, so a refused update changes nothing.
    const std::vector<double> distances = edge_distances(
        positions, cell, receiver_offsets_, sources_, shifts_);
    const TopologyView view{
        numbers_, receiver_offsets_, sources_, shifts_, positions, cell, distances};
    ensure_engine_for_retained_topology();
    if (prospective_domains_.payload_set) {
        payload_dirty_ = true;
    }
    geometry_dirty_ = true;
    push_topology(view);
    positions_.assign(positions.begin(), positions.end());
    cell_.assign(cell.begin(), cell.end());
    distances_ = distances;
    ++geometry_update_count_;
    ++input_generation_;
    commit_domains();
}

template <typename Float>
void FourierD3Kokkos<Float>::reserve_engine_for(const TopologyExtents candidate)
{
    // A system with no active species has no reciprocal sum to evaluate: the engine is
    // never reserved, so no mesh, decomposition, or FFT work is done for it at all.
    if (!has_active_species()) {
        return;
    }
    const FourierD3EngineShape shape = engine_shape_of(candidate);
    if (engine_loaded_ && shape == engine_shape_) {
        return;
    }
    reverse_.reserve(
        effective_mesh()[0],
        effective_mesh()[1],
        effective_mesh()[2],
        static_cast<std::int64_t>(effective_species().size()),
        candidate.nodes,
        candidate.edges,
        effective_n_references(),
        effective_rank_total(),
        effective_rank_chunk(),
        spline_order_);
    // A reservation replaces every view the engine owns, including the damping
    // coefficients, which the forward path clears on the way through.  Every
    // input therefore has to be applied again.
    payload_dirty_ = true;
    geometry_dirty_ = true;
    graph_dirty_ = true;
    damping_dirty_ = true;
    engine_loaded_ = true;
    engine_shape_ = shape;
}

template <typename Float>
auto FourierD3Kokkos<Float>::engine_shape_of(const TopologyExtents candidate) const
    -> FourierD3EngineShape
{
    // The shape a preparation is about to install: the prospective domains, or the
    // committed ones when nothing is staged.
    return FourierD3EngineShape{
        effective_mesh()[0],
        effective_mesh()[1],
        effective_mesh()[2],
        candidate.nodes,
        candidate.edges,
        static_cast<std::int64_t>(effective_species().size()),
        effective_n_references(),
        effective_rank_total(),
        effective_rank_chunk(),
        spline_order_,
    };
}

template <typename Float>
auto FourierD3Kokkos<Float>::retained_topology() const -> TopologyView
{
    return TopologyView{
        numbers_, receiver_offsets_, sources_, shifts_, positions_, cell_, distances_};
}

template <typename Float>
void FourierD3Kokkos<Float>::ensure_engine_for_retained_topology()
{
    if (!has_active_species()) {
        return;
    }
    const TopologyExtents retained{
        static_cast<std::int64_t>(numbers_.size()),
        static_cast<std::int64_t>(sources_.size())};
    // The shape the domains describe, so a caller that pushed a mesh or a payload and
    // then computed is reserved for exactly what it pushed.
    if (!engine_loaded_ || !(engine_shape_of(retained) == engine_shape_)) {
        reserve_engine_for(retained);
    }
}

template <typename Float>
void FourierD3Kokkos<Float>::refresh_engine()
{
    require_payload("preparing the engine");
    require_topology("preparing the engine");
    require_mesh("preparing the engine");
    ensure_engine_for_retained_topology();
    push_topology(retained_topology());
}

template <typename Float>
void FourierD3Kokkos<Float>::push_topology(const TopologyView& topology)
{
    // A system with no active species has nothing to push: every atom is padding or
    // absent, and the published generation is exactly zero.
    if (!has_active_species()) {
        geometry_dirty_ = false;
        graph_dirty_ = false;
        payload_dirty_ = false;
        damping_dirty_ = false;
        return;
    }
    if (!payload_dirty_ && !geometry_dirty_ && !graph_dirty_ && !damping_dirty_) {
        return;
    }
    if (payload_dirty_) {
        // The species channels index the mesh assignment, the coordination chain,
        // and the reference table, so they are installed before anything that
        // reads a channel.
        reverse_.set_species(effective_species(), species_q_table());
        if (!graph_dirty_ && !geometry_dirty_) {
            // Atom channels are derived from the species list.  A payload-only compute
            // keeps the topology but must rebuild those cached channel indices.
            reverse_.set_atoms(topology.positions, topology.numbers);
        }
        if (!graph_dirty_) {
            // set_species invalidates the engine's reference table, while a
            // geometry-only refresh requires that table to remain complete.
            reverse_.set_reference(
                effective_cn_ref(), effective_valid(), effective_species(), effective_v_q());
        }
    }
    if (graph_dirty_) {
        reverse_.set_cell(topology.cell);
        reverse_.set_atoms(topology.positions, topology.numbers);
    } else if (geometry_dirty_) {
        // A geometry-only refresh keeps the loaded topology and replaces the
        // positions, the cell, and the edge distances the topology measures.
        reverse_.update_geometry(topology.positions, topology.cell, topology.distances);
    }
    if (damping_dirty_) {
        reverse_.set_damping(
            damping_.a1, damping_.a2_angstrom, damping_.s6, damping_.s8);
    }
    if (graph_dirty_) {
        const std::vector<double> radii = species_radii(topology.numbers);
        reverse_.set_graph(
            topology.receiver_offsets,
            topology.sources,
            topology.shifts,
            topology.distances,
            topology.numbers,
            radii);
    }
    if (payload_dirty_ && graph_dirty_) {
        reverse_.set_reference(
            effective_cn_ref(), effective_valid(), effective_species(), effective_v_q());
    }
    geometry_dirty_ = false;
    graph_dirty_ = false;
    payload_dirty_ = false;
    damping_dirty_ = false;
}

template <typename Float>
std::vector<double> FourierD3Kokkos<Float>::eigenvalue_window(
    const std::int64_t begin) const
{
    const std::int64_t chunk = domains_.rank_chunk;
    std::vector<double> window(static_cast<std::size_t>(chunk), 0.0);
    for (std::int64_t slot = 0; slot < chunk; ++slot) {
        const std::int64_t index = begin + slot;
        if (index < domains_.rank) {
            window[static_cast<std::size_t>(slot)] =
                domains_.eigenvalues[static_cast<std::size_t>(index)];
        }
    }
    return window;
}

template <typename Float>
void FourierD3Kokkos<Float>::compute(
    const bool compute_forces, const bool compute_stress)
{
    // The domains the caller pushed are the ones this compute evaluates, so they are
    // committed here, before the engine is reserved for them.  A preparation is the other
    // commit point, and a refused preparation commits nothing.
    const bool payload_committed = commit_domains();
    const auto nodes = static_cast<std::size_t>(numbers_.size());
    if (!domains_.has_active_species()) {
        // No active species means no reciprocal sum at all: the dispersion energy of a
        // system whose atoms are absent or padding is exactly zero, so the generation is
        // published as zeros without reserving, planning, or running anything.
        time_stage(result_export_timer_, timing_enabled_, [&] {
            energy_published_ = 0.0;
            atomic_published_.assign(nodes, 0.0);
            stress_published_.assign(6, 0.0);
            forces_published_.assign(3 * nodes, Float(0));
            forces_available_ = compute_forces;
            stress_available_ = compute_stress;
            published_generation_ = input_generation_;
        });
        return;
    }

    // Unlike prepare/update, compute has not already pushed a prospective payload before
    // committing it.  Mark only this path dirty so refresh_engine applies same-layout
    // species/reference changes without making the successful preparation path replay
    // an identical payload on its next compute.
    if (payload_committed) {
        payload_dirty_ = true;
    }
    time_stage(preparation_timer_, timing_enabled_, [&] { refresh_engine(); });

    // One generation is accumulated in scratch buffers and published only after
    // every rank window has run, so a failure part way through cannot be read as
    // a result.
    atomic_scratch_.assign(nodes, 0.0);
    forces_scratch_.assign(3 * nodes, 0.0);
    stress_scratch_.assign(6, 0.0);
    double energy = 0.0;
    for (std::int64_t begin = 0; begin < domains_.rank_total;
         begin += domains_.rank_chunk) {
        reverse_.set_rank(begin, eigenvalue_window(begin));
        const ReciprocalOutcome outcome = reverse_.run(coordination_cutoff_);
        energy += outcome.energy;
        time_stage(result_reduction_timer_, timing_enabled_, [&] {
            const std::vector<double> atomic = reverse_.atomic_energies();
            for (std::size_t index = 0; index < nodes; ++index) {
                atomic_scratch_[index] += atomic[index];
            }
            if (compute_forces) {
                const std::vector<Float> forces = reverse_.forces();
                for (std::size_t index = 0; index < 3 * nodes; ++index) {
                    forces_scratch_[index] += static_cast<double>(forces[index]);
                }
            }
            if (compute_stress) {
                const std::vector<double> stress = reverse_.stress();
                for (std::size_t index = 0; index < 6; ++index) {
                    stress_scratch_[index] += stress[index];
                }
            }
        });
    }

    time_stage(result_export_timer_, timing_enabled_, [&] {
        energy_published_ = energy;
        atomic_published_ = atomic_scratch_;
        stress_published_ = stress_scratch_;
        forces_published_.assign(3 * nodes, Float(0));
        if (compute_forces) {
            for (std::size_t index = 0; index < 3 * nodes; ++index) {
                forces_published_[index] = static_cast<Float>(forces_scratch_[index]);
            }
        }
        forces_available_ = compute_forces;
        stress_available_ = compute_stress;
        published_generation_ = input_generation_;
    });
}

template <typename Float>
FourierD3StageTimings FourierD3Kokkos<Float>::timings() const
{
    FourierD3StageTimings report = reverse_.timings();
    report.preparation = preparation_timer_.timing();
    report.result_reduction = result_reduction_timer_.timing();
    report.result_export = result_export_timer_.timing();
    return report;
}

template <typename Float>
double FourierD3Kokkos<Float>::energy() const
{
    if (published_generation_ == 0) {
        throw std::runtime_error(
            "FourierD3 has no energy; run compute first. The last published result is "
            "never replaced by a partial one, and diagnostics reports whether it is "
            "current through result_available.");
    }
    return energy_published_;
}

template <typename Float>
std::vector<double> FourierD3Kokkos<Float>::atomic_energies() const
{
    if (published_generation_ == 0) {
        throw std::runtime_error(
            "FourierD3 has no atomic energies; run compute first. The last published "
            "result is never replaced by a partial one, and diagnostics reports whether "
            "it is current through result_available.");
    }
    return atomic_published_;
}

template <typename Float>
std::vector<Float> FourierD3Kokkos<Float>::forces() const
{
    if (!forces_available()) {
        throw std::runtime_error(
            "FourierD3 forces are not available: they must be requested from "
            "compute(compute_forces=True) before they can be read.");
    }
    return forces_published_;
}

template <typename Float>
std::vector<double> FourierD3Kokkos<Float>::stress() const
{
    if (!stress_available()) {
        throw std::runtime_error(
            "FourierD3 stress is not available: it must be requested from "
            "compute(compute_stress=True) before it can be read.");
    }
    return stress_published_;
}

template <typename Float>
FourierD3GraphDiagnostics FourierD3Kokkos<Float>::graph_diagnostics() const
{
    FourierD3GraphDiagnostics report;
    report.graph_backend = graph_backend_;
    report.generation = graph_generation_;
    report.geometry_update_count = geometry_update_count_;
    report.num_nodes = static_cast<std::int64_t>(numbers_.size());
    report.num_edges = static_cast<std::int64_t>(sources_.size());
    report.build_cutoff = build_cutoff_;
    return report;
}

template <typename Float>
FourierD3Diagnostics FourierD3Kokkos<Float>::diagnostics() const
{
    FourierD3Diagnostics report;
    report.precision = std::is_same_v<Float, float> ? "float32" : "float64";
    report.functional = functional_;
    report.table_fingerprint = table_fingerprint_;
    report.damping = damping_;
    report.coordination_cutoff = coordination_cutoff_;
    report.spline_order = spline_order_;
    report.active_species = domains_.species;
    report.retained_rank = domains_.rank;
    report.rank_total = domains_.rank_total;
    report.rank_chunk = domains_.rank_chunk;
    report.rank_chunk_count = domains_.rank_chunk > 0
        ? (domains_.rank_total + domains_.rank_chunk - 1) / domains_.rank_chunk
        : 0;
    report.n_references = domains_.n_references;
    report.decomposition_max_relative_error = domains_.max_relative_error;
    report.decomposition_tolerance_met = domains_.tolerance_met;
    report.mesh = domains_.mesh;
    report.mesh_points =
        domains_.mesh[0] * domains_.mesh[1] * domains_.mesh[2];
    report.mesh_invalidation_count = mesh_invalidation_count_;
    // The actual spacing is measured on the retained cell, so an explicit mesh that
    // survives a cell update reports the spacing that update produced.
    if (topology_set_ && domains_.mesh_set) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const std::size_t base = 3 * axis;
            const double length = std::sqrt(
                cell_[base] * cell_[base] + cell_[base + 1] * cell_[base + 1]
                + cell_[base + 2] * cell_[base + 2]);
            report.mesh_spacing[axis] = domains_.mesh[axis] > 0
                ? length / static_cast<double>(domains_.mesh[axis])
                : 0.0;
        }
    }
    report.input_generation = input_generation_;
    report.published_generation = published_generation_;
    report.computed = published_generation_ != 0;
    report.result_available = result_available();
    report.forces_available = forces_available();
    report.stress_available = stress_available();
    report.engine_loaded = engine_loaded_;
    report.timing_enabled = timing_enabled_;
    if (timing_enabled_) {
        report.timings = timings();
    }
    if (engine_loaded_) {
        const ReverseUsage usage = reverse_.usage();
        report.reservations = usage.reservations + usage.forward.reservations
            + usage.coordination.reservations;
        report.allocations =
            usage.allocations + usage.forward.allocations + usage.coordination.allocations;
        report.plans = usage.forward.plans;
        report.plan_retained_bytes = usage.forward.plan_retained_bytes;
        for (const FftPlanUsage& plan : usage.forward.plans) {
            report.plan_builds += plan.builds;
            report.plan_reuses += plan.reuses;
            report.plan_releases += plan.releases;
            report.plan_key_rebuilds += plan.key_rebuilds;
        }
        report.kernel_key_hits = usage.forward.kernels.key_hits;
        report.kernel_key_misses = usage.forward.kernels.key_misses;
        report.kernel_key_rebuilds = usage.forward.kernels.key_rebuilds;
        report.stencil_points = reverse_.geometry().forward.stencil_points;
        report.memory_components = memory_components_of(usage);
        // The flattened itemization is authoritative.  It includes the current mesh
        // allocation and any superseded allocation a held plan still keeps alive.
        detail::finalize_usage(
            report.memory_components, report.active_bytes, report.capacity_bytes);
    }
    return report;
}

template <typename Float>
std::vector<BufferUsage> FourierD3Kokkos<Float>::memory_components_of(
    const ReverseUsage& usage)
{
    // One flat itemization in which every byte of the two totals appears exactly
    // once, labelled by the owner that holds it.  The engine's list already carries
    // the reverse buffers followed by the forward path's own, so the split index
    // names the owner of each entry without a second copy of the list; the mesh,
    // kernel, and coordination owners keep their own itemizations, which are
    // appended here under their own labels.
    std::vector<BufferUsage> components;
    components.reserve(
        usage.buffers.size() + usage.forward.mesh.buffers.size()
        + usage.forward.kernels.buffers.size() + usage.coordination.buffers.size());
    const auto append = [&components](
                            const std::string_view owner, const BufferUsage& buffer) {
        BufferUsage labelled = buffer;
        labelled.name = std::string(owner) + "." + buffer.name;
        components.push_back(std::move(labelled));
    };
    const std::size_t split = std::min(usage.forward_buffer_begin, usage.buffers.size());
    for (std::size_t index = 0; index < usage.buffers.size(); ++index) {
        append(index < split ? "reverse" : "forward", usage.buffers[index]);
    }
    for (const BufferUsage& buffer : usage.forward.mesh.buffers) {
        append("mesh", buffer);
    }
    for (const BufferUsage& buffer : usage.forward.kernels.buffers) {
        append("kernel", buffer);
    }
    for (const BufferUsage& buffer : usage.coordination.buffers) {
        append("coordination", buffer);
    }
    std::vector<std::uint64_t> retained_generations;
    for (const FftPlanUsage& plan : usage.forward.plans) {
        if (!plan.held || !plan.allocation.owns
            || plan.allocation.generation == usage.forward.mesh.allocation_generation
            || std::find(
                   retained_generations.begin(), retained_generations.end(),
                   plan.allocation.generation)
                != retained_generations.end()) {
            continue;
        }
        retained_generations.push_back(plan.allocation.generation);
        const std::string generation = std::to_string(plan.allocation.generation);
        append(
            "mesh",
            BufferUsage{
                "retained_superseded_" + generation + "_real_mesh",
                plan.allocation.real_elements,
                plan.allocation.real_elements,
                detail::element_bytes(plan.allocation.real_elements, sizeof(Float)),
                detail::element_bytes(plan.allocation.real_elements, sizeof(Float)),
                plan.allocation.generation});
        append(
            "mesh",
            BufferUsage{
                "retained_superseded_" + generation + "_spectrum",
                plan.allocation.spectrum_elements,
                plan.allocation.spectrum_elements,
                detail::element_bytes(
                    plan.allocation.spectrum_elements, sizeof(Kokkos::complex<Float>)),
                detail::element_bytes(
                    plan.allocation.spectrum_elements, sizeof(Kokkos::complex<Float>)),
                plan.allocation.generation});
    }
    return components;
}

template <typename Float>
std::uint64_t FourierD3Kokkos<Float>::payload_signature(
    const std::span<const std::int32_t> species,
    const std::span<const double> eigenvalues,
    const std::span<const double> v_q,
    const std::span<const double> cn_ref,
    const std::span<const bool> valid,
    const double max_relative_error,
    const bool tolerance_met)
{
    // A content hash rather than an address, so a caller that rebuilds an
    // identical payload - which is what every geometry that keeps its species
    // does - reuses the loaded one instead of reloading it.
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](const std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    for (const std::int32_t value : species) {
        mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(value)));
    }
    for (const double value : eigenvalues) {
        mix(std::bit_cast<std::uint64_t>(value));
    }
    for (const double value : v_q) {
        mix(std::bit_cast<std::uint64_t>(value));
    }
    for (const double value : cn_ref) {
        mix(std::bit_cast<std::uint64_t>(value));
    }
    for (const bool value : valid) {
        mix(value ? 1ull : 0ull);
    }
    // The achieved error and the tolerance verdict are part of what a caller pushed, so a
    // payload that differs only in them is a different payload rather than a skipped one.
    mix(std::bit_cast<std::uint64_t>(max_relative_error));
    mix(tolerance_met ? 1ull : 0ull);
    return hash;
}

} // namespace symmetrix::fourier_d3

// The one definition lives in `fourier_d3_evaluator.cpp`; the binding
// translation unit names the class through these declarations instead of
// instantiating the whole composition a second time.
extern template class symmetrix::fourier_d3::FourierD3Kokkos<float>;
extern template class symmetrix::fourier_d3::FourierD3Kokkos<double>;
