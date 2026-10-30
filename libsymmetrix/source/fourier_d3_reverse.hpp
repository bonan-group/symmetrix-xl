// Private analytic reverse path of the reciprocal-space FourierD3 correction
// (forces, stress, and atomic energies).
//
// This header owns the whole reverse chain of specification sections 6.1-6.4:
// the coefficient sensitivity, the analytic B-spline mesh gather, the
// receiver-only coordination adjoint, the reciprocal and coordination affine
// virial, the tensile-positive Voigt stress, and the normative atomic-energy
// partition.  It composes the particle-mesh forward path and the
// coordination workspace and adds only the buffers the reverse stages need, so
// the energy, the forces, the stress, and the atomic energies a caller reads
// back all come from one native state and one set of conventions.
//
// Nothing here is a public capability: the seam is bound under a private name,
// it publishes no evaluator, and it has no host or real-space fallback.  A build
// without a FourierD3 FFT provider does not compile it at all.

#ifndef SYMMETRIX_FOURIER_D3_REVERSE_HPP
#define SYMMETRIX_FOURIER_D3_REVERSE_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fourier_d3_coordination.hpp"
#include "fourier_d3_reciprocal.hpp"
#include "particle_mesh_strain.hpp"

namespace symmetrix::fourier_d3 {

// The reverse path reports the forward geometry plus the two conventions it
// adds, so a caller never has to infer how the strain derivative was taken or
// how the Voigt order was chosen.
struct ReverseGeometry {
    ReciprocalGeometry forward;
    std::string strain_convention;
    std::string stress_convention;
};

struct ReverseUsage {
    // The reverse-path buffers followed by the composed forward path's own
    // buffers.  Names are unique across the two sets.
    std::vector<BufferUsage> buffers;
    // Index of the first entry in `buffers` that belongs to the composed forward
    // path.  The reverse buffers come first, so the split lets a caller label each
    // itemization entry by owner without a second copy of the list.
    std::size_t forward_buffer_begin = 0;
    std::uint64_t active_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    // Reverse-path reservations and allocations.  The composed forward path
    // reports its own counters in `forward`, and the coordination workspace -
    // which owns the retained graph, reference table, and coefficient buffers -
    // reports its counters and its own itemization in `coordination`.  Merging
    // the coordination buffers into `buffers` would collide on `numbers`,
    // `species`, and `coefficients`, which are also forward-path names.
    std::uint64_t reservations = 0;
    std::uint64_t allocations = 0;
    ReciprocalUsage forward;
    CoordinationUsage coordination;
};

// A symmetric 3x3 accumulation with the `operator+=` that `Kokkos::parallel_reduce`
// requires of a custom value type.  The virial sums are reductions over mesh bins
// and over directed edges; a per-bin buffer plus a host sum would be the same
// arithmetic with an allocation the size of the mesh.
struct TensorAccumulator {
    double value[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    KOKKOS_INLINE_FUNCTION TensorAccumulator& operator+=(
        const TensorAccumulator& other)
    {
        for (int index = 0; index < 9; ++index) {
            value[index] += other.value[index];
        }
        return *this;
    }
};

// ---------------------------------------------------------------------------
// The orchestrator
// ---------------------------------------------------------------------------

template <typename Float>
class ReciprocalReverse {
public:
    using value_type = Float;

    ReciprocalReverse(MeshLayout layout, std::int64_t channel_bound);
    ~ReciprocalReverse() = default;

    ReciprocalReverse(const ReciprocalReverse&) = delete;
    ReciprocalReverse& operator=(const ReciprocalReverse&) = delete;
    ReciprocalReverse(ReciprocalReverse&&) = delete;
    ReciprocalReverse& operator=(ReciprocalReverse&&) = delete;

    [[nodiscard]] static constexpr std::string_view precision_token() noexcept
    {
        return ReciprocalForward<Float>::precision_token();
    }

    // Validates every shape, bound, and 64-bit product for the forward path, the
    // coordination workspace, and the reverse buffers, grows all three
    // all-or-nothing, and only then commits.  A failure leaves the committed
    // geometry, buffers, counters, and readbacks untouched.
    void reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t num_species,
        std::int64_t num_nodes,
        std::int64_t num_edges,
        std::int64_t num_references,
        std::int64_t rank_total,
        std::int64_t rank_chunk,
        std::int64_t spline_order);

    void set_cell(std::span<const double> cell);
    void set_species(
        std::span<const std::int32_t> species, std::span<const double> q_table);
    void set_atoms(
        std::span<const double> positions, std::span<const std::int32_t> numbers);
    void set_damping(double a1, double a2, double s6, double s8);
    void set_rank(std::int64_t rank_begin, std::span<const double> eigenvalues);
    void prepare_host_stencil_baseline()
    {
        forward_.prepare_host_stencil_baseline();
    }

    // Receiver-major directed graph with the periodic-image shift of every edge.
    // The shifts are consumed in the row convention of specification section 4.1
    // - the image displacement is `r_source + T C - r_receiver` - and are checked
    // against the supplied distances, because the coordination term and the
    // reciprocal term would otherwise describe two different geometries.
    void set_graph(
        std::span<const std::int32_t> offsets,
        std::span<const std::int32_t> sources,
        std::span<const std::int32_t> shifts,
        std::span<const double> distances,
        std::span<const std::int32_t> numbers,
        std::span<const double> covalent_radii);

    void set_reference(
        std::span<const double> cn_ref,
        std::span<const bool> valid,
        std::span<const std::int32_t> species,
        std::span<const double> v_q);

    // Runs the coordination chain, the forward path, and every reverse stage,
    // in that order, from the current inputs.  Nothing is committed before the
    // last stage, so a failed evaluation leaves the previous readbacks intact.
    [[nodiscard]] ReciprocalOutcome run(double coordination_cutoff);

    // Refresh the geometry of an already loaded topology: the positions and the
    // cell are replaced and the edge distances are re-derived by the caller from
    // the retained shifts.  The topology itself is not accepted again, because a
    // different offsets/sources/shifts set is a different graph and must go
    // through `set_graph`; this is what keeps the coordination chain and the mesh
    // assignment describing the same atoms.  Every supplied distance is checked
    // against the refreshed positions, cell, and retained shifts before the
    // object reports itself as updated.
    void update_geometry(
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> distances);

    // Readbacks.
    [[nodiscard]] std::vector<Float> forces() const;
    [[nodiscard]] std::vector<Float> mesh_forces() const;
    [[nodiscard]] std::vector<Float> coordination_forces() const;
    [[nodiscard]] std::vector<double> atomic_energies() const;
    [[nodiscard]] std::vector<Float> coefficient_sensitivity() const;
    [[nodiscard]] std::vector<Float> coordination_sensitivity() const;
    [[nodiscard]] std::vector<double> strain_derivative() const;
    [[nodiscard]] std::vector<double> stress() const;
    [[nodiscard]] std::vector<double> virial() const;

    [[nodiscard]] std::vector<Float> coordination() const
    {
        return coordination_.coordination();
    }
    [[nodiscard]] std::vector<Float> edge_values() const
    {
        return coordination_.edge_values();
    }
    [[nodiscard]] std::vector<Float> edge_radial_derivatives() const
    {
        return coordination_.edge_radial_derivatives();
    }
    [[nodiscard]] std::vector<Float> coefficients() const
    {
        return coordination_.coefficients();
    }
    [[nodiscard]] std::vector<Float> coefficient_derivatives() const
    {
        return coordination_.coefficient_derivatives();
    }
    [[nodiscard]] std::vector<Float> self_coefficient_derivative() const
    {
        return forward_.self_coefficient_derivative();
    }
    [[nodiscard]] std::vector<std::complex<Float>> structure_factors() const
    {
        return forward_.structure_factors();
    }
    [[nodiscard]] std::vector<Float> pair_kernel() const
    {
        return forward_.pair_kernel();
    }
    [[nodiscard]] std::vector<Float> pair_kernel_derivative() const
    {
        return forward_.pair_kernel_derivative();
    }
    // The real-space potential of any species slot.  The forward path retains
    // the cotangent and the stored spectra, so a slot's inverse transform is a
    // deterministic function of the retained state; re-deriving it here means a
    // caller can read back every channel's potential instead of only the last
    // one the gather visited.  Nothing the reverse path reports is affected: the
    // energy, forces, strain, and atomic energies are already committed to their
    // own buffers.
    [[nodiscard]] std::vector<Float> potential(std::int64_t species_slot)
    {
        require_reserved("the potential readback");
        if (forward_.potential_slot_ != species_slot) {
            forward_.invert(species_slot);
        }
        return forward_.potential(species_slot);
    }
    [[nodiscard]] std::vector<Float> field() const { return forward_.field(); }
    [[nodiscard]] double reciprocal_energy() const
    {
        return forward_.reciprocal_energy();
    }
    [[nodiscard]] double self_energy() const { return forward_.self_energy_value(); }
    [[nodiscard]] double energy() const { return forward_.energy_value(); }

    // -- opt-in stage timing ----------------------------------------------

    // Propagates to the composed forward path, so one opt-in covers every stage
    // of an evaluation.  Disabled by default.
    void set_timing_enabled(const bool enabled) noexcept
    {
        timing_enabled_ = enabled;
        forward_.set_timing_enabled(enabled);
    }
    [[nodiscard]] bool timing_enabled() const noexcept { return timing_enabled_; }
    [[nodiscard]] FourierD3StageTimings timings() const;

    [[nodiscard]] ReverseGeometry geometry() const;
    [[nodiscard]] ReverseUsage usage() const;

private:
    template <typename View>
    [[nodiscard]] static View grow_view(
        const View& current,
        std::int64_t required,
        const char* label,
        std::string_view what)
    {
        if (current.data() != nullptr
            && static_cast<std::int64_t>(current.extent(0)) >= required) {
            return current;
        }
        return View(
            Kokkos::view_alloc(std::string(label)),
            static_cast<std::size_t>(detail::checked_extent(required, what)));
    }

    void require_reserved(std::string_view what) const;
    void require_forward(std::string_view what) const;
    void require_graph(std::string_view what) const;

    // Everything a successful `reserve` would install across all three owners,
    // allocated and validated but not yet installed.  `plan_reserve` allocates
    // and throws; `commit` neither allocates nor throws.  This is what makes a
    // composite reservation atomic: the forward path, the coordination
    // workspace, and the reverse buffers are each prepared independently, and a
    // refusal anywhere - a deterministic bound in the last owner, or an
    // allocation failure injected anywhere in the sequence - happens before the
    // first owner is installed.
    struct Reservation {
        Kokkos::View<Float*> mesh_force;
        Kokkos::View<Float*> mesh_sensitivity;
        Kokkos::View<Float*> sensitivity;
        Kokkos::View<Float*> chi;
        Kokkos::View<Float*> coordination_force;
        Kokkos::View<double*> atomic_energy;
        Kokkos::View<std::int32_t*> shifts;
        Kokkos::View<double*> shift_check;
        Kokkos::View<double*> reciprocal_strain;
        ReciprocalForward<Float>::Reservation forward;
        CoordinationWorkspace<Float>::Reservation coordination;
        bool grow_nodes = false;
        bool grow_edges = false;
        bool install_shift_check = false;
        bool install_reciprocal_strain = false;
        std::int64_t num_edges = 0;
        std::int64_t num_references = 0;
        std::int64_t nodes = 0;
        std::int64_t edges = 0;
    };

    [[nodiscard]] Reservation plan_reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t num_species,
        std::int64_t num_nodes,
        std::int64_t num_edges,
        std::int64_t num_references,
        std::int64_t rank_total,
        std::int64_t rank_chunk,
        std::int64_t spline_order) const;
    void commit(Reservation&& reservation);

    // Recomputes every edge displacement from the positions, the cell, and the
    // shifts and compares its length with the supplied distance.  A mismatch is
    // a caller defect that would silently split the coordination term from the
    // reciprocal term, so it is refused rather than absorbed.
public:
    // NVCC requires an enclosing member function that contains an extended
    // host/device lambda to have public access.  These are still native-only
    // launch stages and are not exposed through the Python seam.
    void check_edge_geometry() const;

    // Composes the reverse buffers from the retained stencil, the retained
    // potential, and the retained coefficients.  Both meshes are visited once
    // per species channel; the mesh force reads the fractional gradient the
    // spread used and maps it through the cell inverse.
    void gather_species(std::int64_t species_slot);
    void fold_self_derivative();
    void run_coordination_adjoint();
    void run_reciprocal_virial();
    void finish_strain();

private:
    // Declared first so that every Kokkos allocation below is destroyed before
    // this object stops counting as a live Kokkos owner.
    KokkosLiveObjectGuard kokkos_live_object_guard_;
    ReciprocalForward<Float> forward_;
    CoordinationWorkspace<Float> coordination_;
    MeshLayout layout_ = MeshLayout::right;

    std::int64_t num_edges_ = 0;
    std::int64_t num_references_ = 0;
    // The atomic numbers of the loaded graph, retained so `update_geometry` can
    // re-load the positions without re-accepting the topology.
    std::vector<std::int32_t> graph_numbers_;
    std::int64_t reverse_capacity_nodes_ = 0;
    std::int64_t reverse_capacity_edges_ = 0;
    bool reverse_ready_ = false;
    bool graph_loaded_ = false;
    bool reference_loaded_ = false;
    bool evaluated_ = false;

    // Opt-in stage timing.  The three stages here are the reverse orchestrator's
    // own; the forward path's three live in the forward object and the evaluator's
    // in the evaluator's, so no stage is timed twice.
    bool timing_enabled_ = false;
    StageTimer coordination_timer_;
    StageTimer inverse_fft_gather_adjoint_timer_;
    StageTimer virial_and_strain_timer_;

    std::uint64_t reservations_ = 0;
    std::uint64_t allocations_ = 0;
    std::uint64_t allocation_serial_ = 0;
    // The identity of each retained reverse allocation, advanced only when that
    // allocation is replaced.  A caller can therefore tell a reuse from a
    // rebuild, which a positional index cannot express.  The host tensors below
    // hold no allocation and keep identity 0.
    std::uint64_t nodes_allocation_id_ = 0;
    std::uint64_t shifts_allocation_id_ = 0;
    std::uint64_t shift_check_allocation_id_ = 0;
    std::uint64_t reciprocal_strain_allocation_id_ = 0;

    // Reverse buffers.  Every one of them is grown in place and reused, so a
    // repeated evaluation at the same geometry allocates nothing.
    Kokkos::View<Float*> mesh_force_;               // 3 * nodes
    Kokkos::View<Float*> mesh_sensitivity_;         // nodes * chunk
    Kokkos::View<Float*> sensitivity_;              // nodes * chunk
    Kokkos::View<Float*> chi_;                      // nodes
    Kokkos::View<Float*> coordination_force_;       // 3 * nodes
    Kokkos::View<double*> atomic_energy_;           // nodes
    Kokkos::View<std::int32_t*> shifts_;            // 3 * edges
    Kokkos::View<double*> shift_check_;             // 1
    Kokkos::View<double*> reciprocal_strain_;       // 9

    // Host scalars the reductions write.  They are tensors of a fixed shape, so
    // a Kokkos buffer would add an allocation without adding accounting.
    std::array<double, 9> coordination_virial_ = {};
    std::array<double, 9> strain_ = {};
    std::array<double, 9> virial_ = {};
    std::array<double, 6> stress_ = {};

    // The species channels of `set_species`, kept so `set_reference` can refuse a
    // reference table that would renumber them.
    std::vector<std::int32_t> species_numbers_;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

template <typename Float>
ReciprocalReverse<Float>::ReciprocalReverse(
    const MeshLayout layout, const std::int64_t channel_bound)
    : forward_(layout, channel_bound), coordination_(channel_bound), layout_(layout)
{
    if (layout_ != MeshLayout::right) {
        throw std::invalid_argument(
            "FourierD3 reverse path is compiled for the layout_right transform axis; "
            "a layout_left plan would need an internal transpose, and none is inserted.");
    }
}

template <typename Float>
void ReciprocalReverse<Float>::require_reserved(const std::string_view what) const
{
    if (!reverse_ready_) {
        throw std::runtime_error(
            "FourierD3 reverse path holds no reservation; reserve the geometry before "
            + std::string(what) + ".");
    }
}

template <typename Float>
void ReciprocalReverse<Float>::require_forward(const std::string_view what) const
{
    require_reserved(what);
    // The composed forward path owns the validation of its own configuration.
    // The coefficients are deliberately excluded here: the reverse path adopts
    // them from the coordination workspace a few lines later, so requiring them
    // before the adoption would reject every first evaluation.
    static_cast<void>(forward_.require_mesh(what));
    forward_.require_cell(what);
    forward_.require_species(what);
    forward_.require_atoms(what);
    forward_.require_rank(what);
    if (!forward_.damping_set_) {
        throw std::runtime_error(
            "FourierD3 reverse path requires BJ damping coefficients; call set_damping "
            "first.");
    }
    if (coordination_.geometry_.active_nodes != forward_.num_nodes_) {
        throw std::runtime_error(
            "FourierD3 reverse path node counts disagree between the forward path and "
            "the coordination workspace; this is a reserve defect.");
    }
    if (coordination_.geometry_.active_rank_chunk != forward_.rank_chunk_) {
        throw std::runtime_error(
            "FourierD3 reverse path rank chunks disagree between the forward path and "
            "the coordination workspace; this is a reserve defect.");
    }
    if (coordination_.geometry_.n_species != forward_.num_species_) {
        throw std::runtime_error(
            "FourierD3 reverse path species counts disagree between the forward path and "
            "the coordination workspace; this is a reserve defect.");
    }
}

template <typename Float>
void ReciprocalReverse<Float>::require_graph(const std::string_view what) const
{
    if (!graph_loaded_) {
        throw std::runtime_error(
            "FourierD3 reverse path holds no directed graph; call set_graph before "
            + std::string(what) + ".");
    }
    if (!reference_loaded_) {
        throw std::runtime_error(
            "FourierD3 reverse path holds no low-rank reference; call set_reference "
            "before " + std::string(what) + ".");
    }
}

template <typename Float>
void ReciprocalReverse<Float>::reserve(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t num_species,
    const std::int64_t num_nodes,
    const std::int64_t num_edges,
    const std::int64_t num_references,
    const std::int64_t rank_total,
    const std::int64_t rank_chunk,
    const std::int64_t spline_order)
{
    commit(plan_reserve(
        nx, ny, nz, num_species, num_nodes, num_edges, num_references, rank_total, rank_chunk,
        spline_order));
}

template <typename Float>
auto ReciprocalReverse<Float>::plan_reserve(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t num_species,
    const std::int64_t num_nodes,
    const std::int64_t num_edges,
    const std::int64_t num_references,
    const std::int64_t rank_total,
    const std::int64_t rank_chunk,
    const std::int64_t spline_order) const -> Reservation
{
    // Validation first: every bound and every prospective 64-bit product, for
    // all three owners, before anything is grown.  A rejected request therefore
    // cannot leave the object partially resized.
    const std::int64_t edges = detail::checked_extent(num_edges, "FourierD3 reverse edges");
    const std::int64_t references =
        detail::checked_extent(num_references, "FourierD3 reverse reference slots");
    if (references < 1) {
        throw std::invalid_argument(
            "FourierD3 reverse path needs at least one reference slot; got "
            + std::to_string(num_references) + ".");
    }
    const std::int64_t nodes =
        detail::checked_extent(num_nodes, "FourierD3 reverse nodes");
    const std::int64_t chunk =
        detail::checked_extent(rank_chunk, "FourierD3 reverse rank chunk");
    const std::int64_t mesh_force_elements =
        detail::checked_product(3, nodes, "FourierD3 reverse mesh-force elements");
    const std::int64_t sensitivity_elements = detail::checked_product(
        nodes, chunk, "FourierD3 reverse sensitivity elements");
    const std::int64_t shift_elements =
        detail::checked_product(3, edges, "FourierD3 reverse shift elements");
    // Every reverse buffer count is a bound, not an allocation size.  These are
    // checked before the first candidate view is built, so a request whose own
    // buffer cannot be addressed is refused without asking the allocator for the
    // tens of gigabytes the count implies - which on a machine that has the
    // memory is an OOM kill rather than an exception.  The coordination workspace
    // applies the same limit to the same node and edge counts, but it is the last
    // owner planned, so without these the composite would pay for the reverse and
    // forward buffers first.
    static_cast<void>(detail::checked_flat_index(mesh_force_elements, "mesh_force"));
    static_cast<void>(detail::checked_flat_index(sensitivity_elements, "mesh_sensitivity"));
    static_cast<void>(detail::checked_flat_index(nodes, "coordination_sensitivity"));
    static_cast<void>(detail::checked_flat_index(shift_elements, "edge_shifts"));
    if (nx < 1 || ny < 1 || nz < 1) {
        throw std::invalid_argument(
            "FourierD3 reverse mesh axes must be positive; got ("
            + std::to_string(nx) + ", " + std::to_string(ny) + ", " + std::to_string(nz)
            + ").");
    }
    if (num_species < 1) {
        throw std::invalid_argument(
            "FourierD3 reverse path needs at least one species; got "
            + std::to_string(num_species) + ".");
    }

    Reservation reservation;
    reservation.nodes = nodes;
    reservation.edges = edges;
    reservation.num_edges = edges;
    reservation.num_references = references;
    reservation.grow_nodes = nodes > reverse_capacity_nodes_;
    reservation.grow_edges = edges > reverse_capacity_edges_;

    // Grow the reverse buffers into the reservation.  The counter of a
    // prospective allocation is advanced only after the allocation succeeded,
    // and nothing is installed here.
    if (reservation.grow_nodes) {
        detail::maybe_fail_allocation("the reverse mesh force");
        reservation.mesh_force = Kokkos::View<Float*>(
            "symmetrix_fourier_d3_reverse_mesh_force", mesh_force_elements);
        detail::maybe_fail_allocation("the reverse mesh sensitivity");
        reservation.mesh_sensitivity = Kokkos::View<Float*>(
            "symmetrix_fourier_d3_reverse_mesh_sensitivity", sensitivity_elements);
        detail::maybe_fail_allocation("the reverse coefficient sensitivity");
        reservation.sensitivity = Kokkos::View<Float*>(
            "symmetrix_fourier_d3_reverse_coefficient_sensitivity", sensitivity_elements);
        detail::maybe_fail_allocation("the reverse coordination sensitivity");
        reservation.chi =
            Kokkos::View<Float*>("symmetrix_fourier_d3_reverse_coordination_sensitivity", nodes);
        detail::maybe_fail_allocation("the reverse coordination force");
        reservation.coordination_force = Kokkos::View<Float*>(
            "symmetrix_fourier_d3_reverse_coordination_force", mesh_force_elements);
        detail::maybe_fail_allocation("the reverse atomic energy");
        reservation.atomic_energy =
            Kokkos::View<double*>("symmetrix_fourier_d3_reverse_atomic_energy", nodes);
    }
    if (reservation.grow_edges) {
        detail::maybe_fail_allocation("the reverse edge shifts");
        reservation.shifts = Kokkos::View<std::int32_t*>(
            "symmetrix_fourier_d3_reverse_shifts", shift_elements);
    }
    reservation.install_shift_check = shift_check_.data() == nullptr;
    if (reservation.install_shift_check) {
        detail::maybe_fail_allocation("the reverse shift check");
        reservation.shift_check =
            Kokkos::View<double*>("symmetrix_fourier_d3_reverse_shift_check", 1);
    }
    reservation.install_reciprocal_strain = reciprocal_strain_.data() == nullptr;
    if (reservation.install_reciprocal_strain) {
        detail::maybe_fail_allocation("the reverse reciprocal strain");
        reservation.reciprocal_strain = Kokkos::View<double*>(
            "symmetrix_fourier_d3_reverse_reciprocal_strain", 9);
    }

    // The composed owners are prepared the same way: each allocates into its own
    // reservation and installs nothing.  A refusal in either - including the
    // deterministic bounds and the flat-index checks that the coordination
    // workspace applies after the forward path has already been prepared - is
    // therefore still ahead of every install.
    reservation.forward = forward_.plan_reserve(
        nx, ny, nz, num_species, num_nodes, rank_total, rank_chunk, spline_order);
    reservation.coordination = coordination_.plan_reserve(
        num_nodes, num_edges, num_species, num_references, rank_total, rank_chunk);
    return reservation;
}

template <typename Float>
void ReciprocalReverse<Float>::commit(Reservation&& reservation)
{
    // Install the composed owners first and the reverse buffers last.  None of
    // the three installs allocates or throws, so the object moves from one
    // consistent state to the next.
    forward_.commit(std::move(reservation.forward));
    coordination_.commit(std::move(reservation.coordination));
    if (reservation.grow_nodes) {
        mesh_force_ = std::move(reservation.mesh_force);
        mesh_sensitivity_ = std::move(reservation.mesh_sensitivity);
        sensitivity_ = std::move(reservation.sensitivity);
        chi_ = std::move(reservation.chi);
        coordination_force_ = std::move(reservation.coordination_force);
        atomic_energy_ = std::move(reservation.atomic_energy);
        reverse_capacity_nodes_ = reservation.nodes;
        nodes_allocation_id_ = ++allocation_serial_;
        allocations_ += 6;
    }
    if (reservation.grow_edges) {
        shifts_ = std::move(reservation.shifts);
        reverse_capacity_edges_ = reservation.edges;
        shifts_allocation_id_ = ++allocation_serial_;
        allocations_ += 1;
    }
    if (reservation.install_shift_check) {
        shift_check_ = std::move(reservation.shift_check);
        shift_check_allocation_id_ = ++allocation_serial_;
        allocations_ += 1;
    }
    if (reservation.install_reciprocal_strain) {
        reciprocal_strain_ = std::move(reservation.reciprocal_strain);
        reciprocal_strain_allocation_id_ = ++allocation_serial_;
        allocations_ += 1;
    }
    num_edges_ = reservation.num_edges;
    num_references_ = reservation.num_references;
    reverse_ready_ = true;
    graph_loaded_ = false;
    reference_loaded_ = false;
    evaluated_ = false;
    species_numbers_.clear();
    ++reservations_;
}

template <typename Float>
void ReciprocalReverse<Float>::set_cell(const std::span<const double> cell)
{
    forward_.set_cell(cell);
    // The cell is an input of the edge displacement, so a geometry that was
    // consistent under the previous cell is re-checked rather than trusted.
    evaluated_ = false;
}

template <typename Float>
void ReciprocalReverse<Float>::set_species(
    const std::span<const std::int32_t> species, const std::span<const double> q_table)
{
    forward_.set_species(species, q_table);
    species_numbers_.assign(species.begin(), species.end());
    reference_loaded_ = false;
    evaluated_ = false;
}

template <typename Float>
void ReciprocalReverse<Float>::set_atoms(
    const std::span<const double> positions, const std::span<const std::int32_t> numbers)
{
    forward_.set_atoms(positions, numbers);
    evaluated_ = false;
}

template <typename Float>
void ReciprocalReverse<Float>::set_damping(
    const double a1, const double a2, const double s6, const double s8)
{
    forward_.set_damping(a1, a2, s6, s8);
    evaluated_ = false;
}

template <typename Float>
void ReciprocalReverse<Float>::set_rank(
    const std::int64_t rank_begin, const std::span<const double> eigenvalues)
{
    forward_.set_rank(rank_begin, eigenvalues);
    evaluated_ = false;
}

template <typename Float>
void ReciprocalReverse<Float>::set_graph(
    const std::span<const std::int32_t> offsets,
    const std::span<const std::int32_t> sources,
    const std::span<const std::int32_t> shifts,
    const std::span<const double> distances,
    const std::span<const std::int32_t> numbers,
    const std::span<const double> covalent_radii)
{
    require_reserved("loading the directed graph");
    forward_.require_cell("loading the directed graph");
    forward_.require_atoms("loading the directed graph");
    forward_.require_species("loading the directed graph");
    if (static_cast<std::int64_t>(shifts.size()) != detail::checked_product(
            3, static_cast<std::int64_t>(sources.size()),
            "FourierD3 reverse shift elements")) {
        throw std::invalid_argument(
            "FourierD3 reverse path needs three periodic-image shifts per directed edge; "
            "got " + std::to_string(shifts.size()) + " values for "
            + std::to_string(sources.size()) + " edges.");
    }
    if (static_cast<std::int64_t>(numbers.size()) != forward_.num_nodes_) {
        throw std::invalid_argument(
            "FourierD3 reverse path needs one atomic number per mesh atom; got "
            + std::to_string(numbers.size()) + " values for "
            + std::to_string(forward_.num_nodes_) + " atoms.");
    }
    // The workspace's numbers drive the coordination chain and the forward
    // path's numbers drive the mesh channels.  Two different numberings would
    // put one atom's coordination on another atom's channel, so they must agree.
    {
        const auto mesh_numbers = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), forward_.numbers_view_);
        for (std::int64_t node = 0; node < forward_.num_nodes_; ++node) {
            if (mesh_numbers(node) != numbers[static_cast<std::size_t>(node)]) {
                throw std::invalid_argument(
                    "FourierD3 reverse path graph numbers disagree with the mesh atomic "
                    "numbers at atom " + std::to_string(node) + "; the coordination "
                    "chain and the mesh assignment must describe the same atoms.");
            }
        }
    }

    coordination_.set_graph(offsets, sources, distances, numbers, covalent_radii);
    Kokkos::View<std::int32_t*, Kokkos::HostSpace> host_shifts(
        "symmetrix_fourier_d3_reverse_shifts_host", shifts.size());
    for (std::size_t index = 0; index < shifts.size(); ++index) {
        host_shifts(index) = shifts[index];
    }
    // The retained shift buffer is a capacity bound, not the current edge count: a
    // graph smaller than the largest one loaded so far keeps its allocation, so the
    // copy targets the active subview exactly as the coordination workspace does.
    if (!shifts.empty()) {
        Kokkos::deep_copy(
            Kokkos::subview(shifts_, std::make_pair(std::size_t(0), shifts.size())),
            host_shifts);
    }
    graph_numbers_.assign(numbers.begin(), numbers.end());
    graph_loaded_ = true;
    evaluated_ = false;
    check_edge_geometry();
}

template <typename Float>
void ReciprocalReverse<Float>::update_geometry(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> distances)
{
    require_reserved("updating the geometry");
    require_graph("updating the geometry");
    forward_.set_cell(cell);
    forward_.set_atoms(positions, graph_numbers_);
    coordination_.set_distances(distances);
    evaluated_ = false;
    // The refreshed positions, cell, and distances are re-checked against the
    // retained shifts, so a stale distance can never reach the coordination term.
    check_edge_geometry();
}

template <typename Float>
void ReciprocalReverse<Float>::set_reference(
    const std::span<const double> cn_ref,
    const std::span<const bool> valid,
    const std::span<const std::int32_t> species,
    const std::span<const double> v_q)
{
    require_reserved("loading the low-rank reference");
    forward_.require_species("loading the low-rank reference");
    if (species.size() != species_numbers_.size()
        || !std::equal(species.begin(), species.end(), species_numbers_.begin())) {
        throw std::invalid_argument(
            "FourierD3 reverse path reference species must be the species channels the "
            "mesh was configured with, in the same order; the coordination coefficients "
            "and the mesh assignment index the same channel axis.");
    }
    coordination_.set_reference(cn_ref, valid, species, v_q);
    reference_loaded_ = true;
    evaluated_ = false;
}

template <typename Float>
void ReciprocalReverse<Float>::check_edge_geometry() const
{
    if (!graph_loaded_ || forward_.positions_view_.data() == nullptr
        || forward_.cell_view_.data() == nullptr) {
        return;
    }
    const auto positions = forward_.positions_view_;
    const auto shifts = shifts_;
    const auto distances = coordination_.distances_;
    const auto offsets = coordination_.offsets_;
    const auto sources = coordination_.sources_;
    const particle_mesh::PackedMatrix3 cell =
        particle_mesh::pack_matrix(forward_.cell_);
    auto worst = shift_check_;
    // A float32 distance buffer cannot hold the derived length exactly, so the
    // tolerance is scaled by the storage precision rather than fixed.
    const double relative = std::is_same_v<Float, float> ? 1e-5 : 1e-10;
    Kokkos::deep_copy(worst, 0.0);
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_reverse_shift_check",
        Kokkos::RangePolicy<>(0, forward_.num_nodes_),
        KOKKOS_LAMBDA(const std::int64_t node) {
            const std::int64_t begin = offsets(node);
            const std::int64_t end = offsets(node + 1);
            for (std::int64_t edge = begin; edge < end; ++edge) {
                const std::int64_t source = sources(edge);
                double length = 0.0;
                for (std::int64_t a = 0; a < 3; ++a) {
                    double displacement = positions(3 * source + a) - positions(3 * node + a);
                    for (std::int64_t b = 0; b < 3; ++b) {
                        displacement +=
                            static_cast<double>(shifts(3 * edge + b)) * cell(b, a);
                    }
                    length += displacement * displacement;
                }
                const double derived = Kokkos::sqrt(length);
                const double supplied = static_cast<double>(distances(edge));
                const double excess =
                    Kokkos::abs(derived - supplied) - relative * (1.0 + supplied);
                if (excess > 0.0) {
                    Kokkos::atomic_max(&worst(0), excess);
                }
            }
        });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), worst);
    if (host(0) > 0.0) {
        throw std::invalid_argument(
            "FourierD3 reverse path periodic-image shifts disagree with the supplied "
            "edge distances by up to " + detail::format_scalar(host(0))
            + " Angstrom; the shifts, the distances, and the positions must describe "
            "the same geometry in the row convention r_source + T C - r_receiver.");
    }
}

template <typename Float>
ReciprocalOutcome ReciprocalReverse<Float>::run(const double coordination_cutoff)
{
    if (!(coordination_cutoff > 0.0) || !std::isfinite(coordination_cutoff)) {
        throw std::invalid_argument(
            "FourierD3 reverse path needs a positive finite coordination cutoff; got "
            + detail::format_scalar(coordination_cutoff) + ".");
    }
    require_forward("the reverse evaluation");
    require_graph("the reverse evaluation");
    check_edge_geometry();

    // Transactional order.  `prepare()` is the only part of this call that
    // allocates, and it writes no output - only keyed caches: the retained
    // stencil, the mesh grid, the reciprocal kernel tables, and both FFT plans.
    // The stages after it validate their preconditions and allocate nothing, so
    // every allocation failure a caller can inject, and every configuration
    // refusal, throws before the first observable byte changes.  The
    // coordination chain then runs two kernels that validate before they write,
    // and the reverse stages below allocate nothing either.  The last successful
    // energy, forces, stress, atomic energies, coefficient sensitivities, and
    // coefficient readbacks therefore stay readable and unchanged together.
    // Without this order the coefficients could be refreshed by a run that then
    // failed, which would leave a caller holding new coefficients beside old
    // forces.
    forward_.prepare();

    // The coordination chain owns the coefficients: the reverse path consumes
    // them rather than computing a second set, so a caller cannot evaluate the
    // reciprocal term and the coordination term from different coefficients.
    time_stage(coordination_timer_, timing_enabled_, [&] {
        coordination_.run_coordination(coordination_cutoff);
        coordination_.run_coefficients(forward_.rank_begin_);
        const std::int64_t nodes = forward_.num_nodes_;
        const std::int64_t chunk = forward_.rank_chunk_;
        const std::int64_t elements = detail::checked_product(
            nodes, chunk, "FourierD3 reverse coefficient elements");
        Kokkos::deep_copy(
            Kokkos::subview(
                forward_.coefficients_view_,
                std::make_pair(std::size_t(0), static_cast<std::size_t>(elements))),
            Kokkos::subview(
                coordination_.coefficients_,
                std::make_pair(std::size_t(0), static_cast<std::size_t>(elements))));
        forward_.coefficients_set_ = true;
        forward_.invalidate_stages();
    });

    const ReciprocalOutcome outcome = forward_.execute();

    // Reverse stages.  The mesh force and the coefficient sensitivity are built
    // once per species channel from the retained stencil, the retained potential,
    // and the retained coefficients; every other stage is a reduction over the
    // retained state.
    time_stage(inverse_fft_gather_adjoint_timer_, timing_enabled_, [&] {
        Kokkos::deep_copy(forward_.execution_space_, mesh_force_, Float(0));
        Kokkos::deep_copy(
            forward_.execution_space_, mesh_sensitivity_, Float(0));
        for (std::int64_t slot = 0; slot < forward_.num_species_; ++slot) {
            forward_.invert(slot);
            gather_species(slot);
        }
        fold_self_derivative();
        run_coordination_adjoint();
    });
    time_stage(virial_and_strain_timer_, timing_enabled_, [&] {
        run_reciprocal_virial();
        finish_strain();
    });
    evaluated_ = true;
    return outcome;
}

template <typename Float>
FourierD3StageTimings ReciprocalReverse<Float>::timings() const
{
    FourierD3StageTimings report = forward_.timings();
    report.coordination = coordination_timer_.timing();
    report.inverse_fft_gather_adjoint = inverse_fft_gather_adjoint_timer_.timing();
    report.virial_and_strain = virial_and_strain_timer_.timing();
    return report;
}

template <typename Float>
void ReciprocalReverse<Float>::gather_species(const std::int64_t species_slot)
{
    const std::int64_t nodes = forward_.num_nodes_;
    const std::int64_t chunk = forward_.rank_chunk_;
    const std::int32_t slot = static_cast<std::int32_t>(species_slot);
    auto& workspace = forward_.workspace_;
    const auto coefficients = workspace.checked_weights_view(
        forward_.coefficients_view_, nodes, chunk,
        "FourierD3 reverse coefficient weights");
    const auto channels = workspace.checked_groups_view(
        forward_.channel_view_, nodes, "FourierD3 reverse species groups");
    using Workspace = particle_mesh::ParticleMeshWorkspace<Float>;
    const typename Workspace::weight_adjoints_view sensitivities(
        mesh_sensitivity_.data(), nodes, chunk);
    const typename Workspace::position_forces_view forces(
        mesh_force_.data(), nodes, 3);
    workspace.gather(coefficients, channels, slot, sensitivities, forces);

}

template <typename Float>
void ReciprocalReverse<Float>::fold_self_derivative()
{
    const std::int64_t nodes = forward_.num_nodes_;
    const std::int64_t chunk = forward_.rank_chunk_;
    auto mesh_sensitivity = mesh_sensitivity_;
    auto self_derivative = forward_.self_derivative_view_;
    auto sensitivity = sensitivity_;
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_reverse_fold", Kokkos::RangePolicy<>(0, nodes * chunk),
        KOKKOS_LAMBDA(const std::int64_t element) {
            sensitivity(element) = mesh_sensitivity(element) + self_derivative(element);
        });
}

template <typename Float>
void ReciprocalReverse<Float>::run_coordination_adjoint()
{
    const std::int64_t nodes = forward_.num_nodes_;
    const std::int64_t chunk = forward_.rank_chunk_;
    const auto positions = forward_.positions_view_;
    const auto shifts = shifts_;
    const particle_mesh::PackedMatrix3 cell =
        particle_mesh::pack_matrix(forward_.cell_);
    const auto offsets = coordination_.offsets_;
    const auto sources = coordination_.sources_;
    const auto slopes = coordination_.edge_radial_derivatives_;
    auto sensitivity = sensitivity_;
    auto derivatives = coordination_.coefficient_derivatives_;
    auto chi = chi_;
    auto force = coordination_force_;

    // chi_i = sum_l (dE/dc_il) * (dc_il/dtheta_i): the total derivative of the
    // energy with respect to the coordination number, which is the only channel
    // through which a coordination number enters.
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_reverse_chi", Kokkos::RangePolicy<>(0, nodes),
        KOKKOS_LAMBDA(const std::int64_t node) {
            double total = 0.0;
            for (std::int64_t l = 0; l < chunk; ++l) {
                total += static_cast<double>(sensitivity(node * chunk + l))
                    * static_cast<double>(derivatives(node * chunk + l));
            }
            chi(node) = static_cast<Float>(total);
        });

    // Receiver-only accumulation over the directed edge list.  Each directed
    // edge contributes its whole force to its receiver; the reverse orientation
    // supplies the opposite force from its own entry, so nothing is halved and
    // nothing is added twice.  A non-positive distance is masked exactly as the
    // coordination pair term masks it, so a coincident pair cannot divide by zero.
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_reverse_coordination_force", Kokkos::RangePolicy<>(0, nodes),
        KOKKOS_LAMBDA(const std::int64_t node) {
            Float accumulated[3] = {Float(0), Float(0), Float(0)};
            const std::int64_t begin = offsets(node);
            const std::int64_t end = offsets(node + 1);
            for (std::int64_t edge = begin; edge < end; ++edge) {
                const std::int64_t source = sources(edge);
                double delta[3];
                double length = 0.0;
                for (std::int64_t a = 0; a < 3; ++a) {
                    double displacement = positions(3 * source + a) - positions(3 * node + a);
                    for (std::int64_t b = 0; b < 3; ++b) {
                        displacement +=
                            static_cast<double>(shifts(3 * edge + b)) * cell(b, a);
                    }
                    delta[a] = displacement;
                    length += displacement * displacement;
                }
                const double radius = Kokkos::sqrt(length);
                if (!(radius > 0.0)) {
                    continue;
                }
                const double slope = static_cast<double>(slopes(edge));
                if (slope == 0.0) {
                    continue;
                }
                const double factor =
                    (static_cast<double>(chi(node)) + static_cast<double>(chi(source)))
                    * slope / radius;
                accumulated[0] += static_cast<Float>(factor * delta[0]);
                accumulated[1] += static_cast<Float>(factor * delta[1]);
                accumulated[2] += static_cast<Float>(factor * delta[2]);
            }
            force(3 * node + 0) = accumulated[0];
            force(3 * node + 1) = accumulated[1];
            force(3 * node + 2) = accumulated[2];
        });

    // The coordination virial is one half of the receiver-major directed-edge
    // sum, because each physical pair appears twice in the directed list.  The
    // per-edge force is recomputed here rather than read from the accumulated
    // buffer: the accumulated force on a receiver sums over all of its edges, so
    // pairing it with one edge's displacement would be a different, and
    // non-symmetric, tensor.
    TensorAccumulator accumulator;
    Kokkos::parallel_reduce(
        "symmetrix_fourier_d3_reverse_coordination_virial", Kokkos::RangePolicy<>(0, nodes),
        KOKKOS_LAMBDA(const std::int64_t node, TensorAccumulator& local) {
            const std::int64_t begin = offsets(node);
            const std::int64_t end = offsets(node + 1);
            for (std::int64_t edge = begin; edge < end; ++edge) {
                const std::int64_t source = sources(edge);
                double delta[3];
                double length = 0.0;
                for (std::int64_t a = 0; a < 3; ++a) {
                    double displacement = positions(3 * source + a) - positions(3 * node + a);
                    for (std::int64_t b = 0; b < 3; ++b) {
                        displacement +=
                            static_cast<double>(shifts(3 * edge + b)) * cell(b, a);
                    }
                    delta[a] = displacement;
                    length += displacement * displacement;
                }
                const double radius = Kokkos::sqrt(length);
                if (!(radius > 0.0)) {
                    continue;
                }
                const double slope = static_cast<double>(slopes(edge));
                if (slope == 0.0) {
                    continue;
                }
                const double factor =
                    0.5
                    * (static_cast<double>(chi(node)) + static_cast<double>(chi(source)))
                    * slope / radius;
                for (std::int64_t a = 0; a < 3; ++a) {
                    const double component = factor * delta[a];
                    for (std::int64_t b = 0; b < 3; ++b) {
                        local.value[static_cast<std::size_t>(3 * a + b)] +=
                            component * delta[b];
                    }
                }
            }
        },
        accumulator);
    std::copy(
        std::begin(accumulator.value), std::end(accumulator.value),
        coordination_virial_.begin());
}

template <typename Float>
void ReciprocalReverse<Float>::run_reciprocal_virial()
{
    const auto spectra = forward_.species_spectra_view_;
    const auto kernels = forward_.kernels_.pair_kernel_derivative();
    const auto eigenvalues = forward_.eigenvalues_view_;
    const auto hermitian = forward_.grid_.hermitian_weight;
    const auto norms = forward_.grid_.k_norm;
    auto radial_derivative = forward_.bin_radial_derivative_view_;
    const std::int64_t nspec = forward_.num_species_;
    const std::int64_t chunk = forward_.rank_chunk_;
    const std::int64_t bins = forward_.grid_.bins;

    const double inverse_volume = forward_.inverse_volume_;

    // d phihat/d|k| is a function of the wave-vector magnitude, so the strain
    // derivative of the reciprocal energy is
    //
    //     (1 / (2 Omega)) sum_m w_m sum_l lambda_l
    //         sum_{X,Y} mu_XY phihat' (k_a k_b / |k|) Re[A_X conj(A_Y)]
    //
    // with the |k| = 0 bin contributing nothing anisotropic, because its wave
    // vector vanishes. The D3 adapter computes de_b/d|k|; the shared reducer
    // owns the geometry-only affine contraction and the isotropic volume term.
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_bin_radial_derivative",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            forward_.execution_space_, 0, bins),
        KOKKOS_LAMBDA(const std::int64_t bin) {
            const double weight = static_cast<double>(hermitian(bin));
            double total = 0.0;
            for (std::int64_t l = 0; l < chunk; ++l) {
                const double eigenvalue = eigenvalues(l);
                for (std::int64_t x = 0; x < nspec; ++x) {
                    const auto amplitude = spectra((x * chunk + l) * bins + bin);
                    double ordered = 0.0;
                    for (std::int64_t y = 0; y < nspec; ++y) {
                        const std::int64_t pair = unordered_pair_index(x, y, nspec);
                        ordered += static_cast<double>(kernels(pair * bins + bin))
                            * (amplitude * Kokkos::conj(spectra((y * chunk + l) * bins + bin)))
                                  .real();
                    }
                    total += eigenvalue * ordered;
                }
            }
            const double norm = static_cast<double>(norms(bin));
            const double small_norm_scale = norm > bj_k_norm_guard ? 1.0 : norm;
            radial_derivative(bin) =
                -0.5 * weight * total * inverse_volume * small_norm_scale;
        });
    particle_mesh::reduce_radial_strain(
        forward_.execution_space_, forward_.grid_, forward_.bin_energy_view_,
        radial_derivative, reciprocal_strain_);
}

template <typename Float>
void ReciprocalReverse<Float>::finish_strain()
{
    const double inverse_volume = forward_.inverse_volume_;
    Kokkos::View<double*, Kokkos::HostSpace> reciprocal(
        "symmetrix_fourier_d3_reverse_reciprocal_strain_host", 9);
    Kokkos::deep_copy(forward_.execution_space_, reciprocal, reciprocal_strain_);
    forward_.execution_space_.fence("FourierD3 reciprocal strain readback");
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const std::size_t index = static_cast<std::size_t>(3 * a + b);
            // The shared reducer supplies the complete reciprocal contribution.
            // The D3 coordination virial enters with the same sign as the
            // lattice sum because it is one half of the same derivative.
            strain_[index] = reciprocal(index) + coordination_virial_[index];
            virial_[index] = -strain_[index];
        }
    }
    // Voigt (xx, yy, zz, yz, xz, xy), tensile positive, shear components not
    // doubled: this is dE/du divided by the cell volume.
    stress_[0] = strain_[0] * inverse_volume;
    stress_[1] = strain_[4] * inverse_volume;
    stress_[2] = strain_[8] * inverse_volume;
    stress_[3] = strain_[5] * inverse_volume;
    stress_[4] = strain_[2] * inverse_volume;
    stress_[5] = strain_[1] * inverse_volume;

    // Normative atomic partition of specification section 6.4: one half of
    // c_il times the mesh sensitivity the atom gathers, plus the atom's own
    // self-term share that the forward path already accumulated per atom.
    const std::int64_t nodes = forward_.num_nodes_;
    const std::int64_t chunk = forward_.rank_chunk_;
    auto mesh_sensitivity = mesh_sensitivity_;
    auto coefficients = forward_.coefficients_view_;
    auto node_energy = forward_.node_energy_view_;
    auto atomic = atomic_energy_;
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_reverse_atomic_energy", Kokkos::RangePolicy<>(0, nodes),
        KOKKOS_LAMBDA(const std::int64_t node) {
            double share = 0.0;
            for (std::int64_t l = 0; l < chunk; ++l) {
                share += 0.5 * static_cast<double>(mesh_sensitivity(node * chunk + l))
                    * static_cast<double>(coefficients(node * chunk + l));
            }
            atomic(node) = share + node_energy(node);
        });
}

// ---------------------------------------------------------------------------
// Readbacks
// ---------------------------------------------------------------------------

template <typename Float>
std::vector<Float> ReciprocalReverse<Float>::forces() const
{
    require_reserved("the reverse force readback");
    std::vector<Float> values = mesh_forces();
    const std::vector<Float> other = coordination_forces();
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] += other[index];
    }
    return values;
}

template <typename Float>
std::vector<Float> ReciprocalReverse<Float>::mesh_forces() const
{
    require_reserved("the reverse mesh-force readback");
    const std::int64_t elements = detail::checked_product(
        3, forward_.num_nodes_, "FourierD3 reverse mesh-force elements");
    const auto mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh_force_);
    std::vector<Float> values(static_cast<std::size_t>(elements));
    for (std::int64_t index = 0; index < elements; ++index) {
        values[static_cast<std::size_t>(index)] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<Float> ReciprocalReverse<Float>::coordination_forces() const
{
    require_reserved("the reverse coordination-force readback");
    const std::int64_t elements = detail::checked_product(
        3, forward_.num_nodes_, "FourierD3 reverse coordination-force elements");
    const auto mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coordination_force_);
    std::vector<Float> values(static_cast<std::size_t>(elements));
    for (std::int64_t index = 0; index < elements; ++index) {
        values[static_cast<std::size_t>(index)] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<double> ReciprocalReverse<Float>::atomic_energies() const
{
    require_reserved("the reverse atomic-energy readback");
    const auto mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), atomic_energy_);
    std::vector<double> values(static_cast<std::size_t>(forward_.num_nodes_));
    for (std::int64_t index = 0; index < forward_.num_nodes_; ++index) {
        values[static_cast<std::size_t>(index)] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<Float> ReciprocalReverse<Float>::coefficient_sensitivity() const
{
    require_reserved("the coefficient-sensitivity readback");
    const std::int64_t elements = detail::checked_product(
        forward_.num_nodes_, forward_.rank_chunk_,
        "FourierD3 reverse sensitivity elements");
    const auto mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sensitivity_);
    std::vector<Float> values(static_cast<std::size_t>(elements));
    for (std::int64_t index = 0; index < elements; ++index) {
        values[static_cast<std::size_t>(index)] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<Float> ReciprocalReverse<Float>::coordination_sensitivity() const
{
    require_reserved("the coordination-sensitivity readback");
    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), chi_);
    std::vector<Float> values(static_cast<std::size_t>(forward_.num_nodes_));
    for (std::int64_t index = 0; index < forward_.num_nodes_; ++index) {
        values[static_cast<std::size_t>(index)] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<double> ReciprocalReverse<Float>::strain_derivative() const
{
    require_reserved("the strain-derivative readback");
    return std::vector<double>(strain_.begin(), strain_.end());
}

template <typename Float>
std::vector<double> ReciprocalReverse<Float>::stress() const
{
    require_reserved("the stress readback");
    return std::vector<double>(stress_.begin(), stress_.end());
}

template <typename Float>
std::vector<double> ReciprocalReverse<Float>::virial() const
{
    require_reserved("the virial readback");
    return std::vector<double>(virial_.begin(), virial_.end());
}

template <typename Float>
ReverseGeometry ReciprocalReverse<Float>::geometry() const
{
    ReverseGeometry report;
    report.forward = forward_.geometry();
    report.strain_convention = "affine_positions_and_cell";
    report.stress_convention = "voigt_xx_yy_zz_yz_xz_xy_tensile_positive";
    return report;
}

template <typename Float>
ReverseUsage ReciprocalReverse<Float>::usage() const
{
    ReverseUsage usage;
    usage.forward = forward_.usage();
    usage.coordination = coordination_.usage();
    usage.reservations = reservations_;
    usage.allocations = allocations_;

    const std::int64_t nodes = forward_.num_nodes_;
    const std::int64_t chunk = forward_.rank_chunk_;
    const std::int64_t force_elements =
        detail::checked_product(3, nodes, "FourierD3 reverse mesh-force elements");
    const std::int64_t sensitivity_elements =
        detail::checked_product(nodes, chunk, "FourierD3 reverse sensitivity elements");

    // Reverse buffers.  The active extent is the committed geometry and the
    // capacity is the retained extent the grow-only reserve never shrinks, so
    // the itemization distinguishes active from held.  The identity of each
    // buffer is the serial of the allocation that produced it, so a caller can
    // tell a reuse from a rebuild; a buffer that holds no allocation yet reports
    // identity zero.
    const auto record = [&usage](
                            const char* name,
                            std::int64_t active,
                            std::int64_t capacity,
                            std::uint64_t size,
                            std::uint64_t allocation_id) {
        BufferUsage buffer;
        buffer.name = name;
        buffer.active_elements = active;
        buffer.capacity_elements = capacity;
        buffer.active_bytes = detail::element_bytes(active, size);
        buffer.capacity_bytes = detail::element_bytes(capacity, size);
        buffer.allocation_id = allocation_id;
        usage.buffers.push_back(buffer);
    };
    record(
        "mesh_force", reverse_ready_ ? force_elements : 0,
        static_cast<std::int64_t>(mesh_force_.extent(0)), sizeof(Float),
        nodes_allocation_id_);
    record(
        "mesh_sensitivity", reverse_ready_ ? sensitivity_elements : 0,
        static_cast<std::int64_t>(mesh_sensitivity_.extent(0)), sizeof(Float),
        nodes_allocation_id_);
    record(
        "coefficient_sensitivity", reverse_ready_ ? sensitivity_elements : 0,
        static_cast<std::int64_t>(sensitivity_.extent(0)), sizeof(Float),
        nodes_allocation_id_);
    record(
        "coordination_sensitivity", reverse_ready_ ? nodes : 0,
        static_cast<std::int64_t>(chi_.extent(0)), sizeof(Float), nodes_allocation_id_);
    record(
        "coordination_force", reverse_ready_ ? force_elements : 0,
        static_cast<std::int64_t>(coordination_force_.extent(0)), sizeof(Float),
        nodes_allocation_id_);
    record(
        "atomic_energy", reverse_ready_ ? nodes : 0,
        static_cast<std::int64_t>(atomic_energy_.extent(0)), sizeof(double),
        nodes_allocation_id_);
    record(
        "edge_shifts", reverse_ready_ ? 3 * num_edges_ : 0,
        static_cast<std::int64_t>(shifts_.extent(0)), sizeof(std::int32_t),
        shifts_allocation_id_);
    // A never-reserved object holds no shift check, so its active extent is zero
    // rather than the one element a reserved object always holds.
    const std::int64_t shift_check_capacity =
        static_cast<std::int64_t>(shift_check_.extent(0));
    record(
        "shift_check", shift_check_capacity > 0 ? 1 : 0, shift_check_capacity,
        sizeof(double), shift_check_allocation_id_);
    const std::int64_t reciprocal_strain_capacity =
        static_cast<std::int64_t>(reciprocal_strain_.extent(0));
    record(
        "reciprocal_strain", reciprocal_strain_capacity > 0 ? 9 : 0,
        reciprocal_strain_capacity, sizeof(double),
        reciprocal_strain_allocation_id_);
    // Fixed-shape host tensors: they hold no Kokkos allocation and keep identity
    // zero, but they are part of the reverse path's retained state and are
    // itemized as such.  They are four distinct arrays, so itemizing all four
    // counts each byte once.
    record("coordination_virial", 9, 9, sizeof(double), 0);
    record("strain_derivative", 9, 9, sizeof(double), 0);
    record("virial", 9, 9, sizeof(double), 0);
    record("stress", 6, 6, sizeof(double), 0);

    // The composed forward path's own buffers are reported alongside the
    // reverse buffers, so one list describes every named buffer of the object.
    usage.forward_buffer_begin = usage.buffers.size();
    usage.buffers.insert(
        usage.buffers.end(), usage.forward.buffers.begin(), usage.forward.buffers.end());
    // finalize_usage sums the itemization and rejects a duplicated name, which
    // is what makes "every buffer exactly once" checkable rather than asserted.
    detail::finalize_usage(usage.buffers, usage.active_bytes, usage.capacity_bytes);
    // The mesh allocation and the kernel tables are not itemized in `buffers`,
    // and the coordination workspace keeps its own itemization, so all three
    // contributions are added here exactly once.
    usage.active_bytes += usage.forward.mesh.active_bytes
        + usage.forward.kernels.active_bytes + usage.coordination.active_bytes;
    usage.capacity_bytes += usage.forward.mesh.capacity_bytes
        + usage.forward.kernels.capacity_bytes + usage.coordination.capacity_bytes;
    return usage;
}

}  // namespace symmetrix::fourier_d3

// The one definition lives in `fourier_d3_reverse.cpp`; the binding translation
// unit names the class through these declarations instead of instantiating the
// whole reverse chain a second time.
extern template class symmetrix::fourier_d3::ReciprocalReverse<float>;
extern template class symmetrix::fourier_d3::ReciprocalReverse<double>;

#endif  // SYMMETRIX_FOURIER_D3_REVERSE_HPP
