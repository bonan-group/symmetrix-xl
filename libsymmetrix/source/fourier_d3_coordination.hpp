#ifndef SYMMETRIX_FOURIER_D3_COORDINATION_HPP
#define SYMMETRIX_FOURIER_D3_COORDINATION_HPP

// Private modified coordination-number and separable-coefficient kernels for
// the reciprocal-space FourierD3 correction and its coordination state.
//
// This layer owns exactly two numerical stages of the forward path:
//
//   * the modified counting function of specification section 3.2, evaluated on
//     a caller-supplied receiver-major directed graph, producing one switch
//     value and one scalar radial derivative ``d theta_ij / d r`` per directed
//     edge and one coordination number ``theta_i`` per receiver;
//   * the separable coefficient contraction of specification section 3.3.5,
//     producing ``c[i,l]`` and ``dc[i,l] / d theta_i`` for a bounded rank chunk
//     from caller-supplied signed low-rank factors.
//
// The steepness law is the toolkit v0.5.0 form, which reaches exact zero with a
// continuous derivative at the cutoff; ordinary D3's counting law and the
// paper's literal Eq. (18) are both deliberately not implemented here.
// Differentiation is receiver-only: this layer produces ``d theta_ij / d r``
// and ``dc_i,l / d theta_i`` and accumulates no pair force; the coordination
// adjoint and every force/virial sign belong to the reverse chain.
//
// The whole translation unit is compiled only when the FFT build capability
// `SYMMETRIX_FOURIER_D3_ENABLED` is set; without a discovered FFT provider
// nothing here exists and no stub claims numerical support.

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "dispersion_parameters.hpp"
#include "fourier_d3_storage.hpp"
#include "tools_kokkos.hpp"

namespace symmetrix::fourier_d3 {

// ---------------------------------------------------------------------------
// Model constants (specification sections 3.2 and 3.3.1)
// ---------------------------------------------------------------------------

// `R_mid = 0.5 * (0.75 * C + R_cut)` uses the bare covalent sum, while the
// counting function keeps the tabulated sum with Grimme's 4/3 factor folded in.
inline constexpr double coordination_unscale = 0.75;
inline constexpr double coordination_steepness = 16.0;
inline constexpr double coordination_epsilon = 1e-6;
// `k3` of the Gaussian reference weight `exp(-k3 * delta^2)`.
inline constexpr double reference_weight_k3 = 4.0;

// The canonical D3 reference tables use five coordination slots per species, and
// the kernel keeps the weights in registers, so the extent is a hard bound.
inline constexpr std::int64_t max_reference_slots =
    static_cast<std::int64_t>(dispersion::d3_max_coordination_references);

namespace kernel {

// Finiteness without a library predicate: the library is compiled with the
// historical -ffast-math policy, under which `std::isfinite` is folded to a
// constant and cannot observe a NaN or an infinity.  The IEEE-754 exponent is
// therefore read directly, which behaves identically on host and device.
template <typename Float>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool is_finite(const Float value) noexcept
{
    if constexpr (std::is_same_v<Float, float>) {
        constexpr std::uint32_t exponent_mask = 0x7f800000u;
        return (Kokkos::bit_cast<std::uint32_t>(value) & exponent_mask) != exponent_mask;
    } else {
        static_assert(std::is_same_v<Float, double>, "unsupported FourierD3 precision");
        constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
        return (Kokkos::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
    }
}

// One directed edge's switch value and its radial derivative `d theta / d r`.
template <typename Float>
struct PairTerm {
    Float value = Float(0);
    Float radial_derivative = Float(0);
};

// Modified counting function of specification section 3.2 on `0 < r < R_cut`.
//
// `covalent_sum` is the tabulated sum of the two radii, which already carries
// Grimme's 4/3 factor; the transition radius rescales it by 3/4.  Activity is a
// caller decision: padding atoms are resolved before this call, so a
// non-positive or out-of-range distance is simply masked here.
template <typename Float>
[[nodiscard]] KOKKOS_INLINE_FUNCTION PairTerm<Float> modified_pair_term(
    const Float distance, const Float covalent_sum, const Float cutoff) noexcept
{
    PairTerm<Float> term;
    if (!(distance > Float(0)) || !(distance < cutoff)) {
        return term;
    }
    const Float transition =
        Float(0.5) * (Float(coordination_unscale) * covalent_sum + cutoff);
    const bool beyond = distance > transition;
    const Float stabilised = beyond ? distance : transition;
    const Float gap = cutoff - stabilised;
    const Float denominator = gap * gap + Float(coordination_epsilon);
    const Float offset = stabilised - transition;
    const Float steepness = Float(coordination_steepness) + offset * offset / denominator;
    const Float ratio = covalent_sum / distance - Float(1);
    const Float argument = steepness * ratio;
    // Branchwise logistic: the argument reaches +-infinity at the cutoff, where a
    // plain 1/(1+exp(-x)) overflows.  Each branch keeps the exponent
    // non-positive.
    const Float exponential = Kokkos::exp(-Kokkos::abs(argument));
    term.value = argument >= Float(0)
        ? Float(1) / (Float(1) + exponential)
        : exponential / (Float(1) + exponential);
    const Float ratio_derivative = -covalent_sum / (distance * distance);
    Float steepness_derivative = Float(0);
    if (beyond) {
        steepness_derivative = Float(2) * offset / denominator
            + Float(2) * offset * offset * gap / (denominator * denominator);
    }
    term.radial_derivative = term.value * (Float(1) - term.value)
        * (steepness_derivative * ratio + steepness * ratio_derivative);
    return term;
}

// One atom's masked Gaussian reference weights and their coordination
// derivative, in the stable form of specification section 3.3.5.
//
// Invalid reference slots are masked exactly: they contribute nothing to the
// numerator or the denominator.  The exponent is shifted by the largest valid
// log weight, so the largest weight is exactly one and the denominator cannot
// underflow.  When an argument is not representable - a delta beyond the safe
// square, or a non-finite delta or log weight - the row falls back to a one-hot
// nearest valid reference with zero derivative.  Selection is by
// ``|theta - theta_ref|`` over valid slots only, ties keeping the smallest slot
// index, which is what makes a non-finite coordination number deterministic.
template <typename Float>
struct ReferenceWeightRow {
    Float delta[max_reference_slots]{};
    Float weight[max_reference_slots]{};
    Float weight_derivative[max_reference_slots]{};
    Float denominator = Float(0);
    Float denominator_derivative = Float(0);
    std::int64_t fallback_slot = -1;
    std::int64_t nearest_slot = -1;
};

// The one fallback row: all weights and slopes zero, plus the slot the caller
// must copy instead of contracting a denominator that is not usable.  The row
// reports the slot as both the nearest and the fallback slot, so a diagnostic
// reading either field sees the slot that was actually selected; the caller keys
// on `fallback_slot`, and `nearest_slot` is -1 only when no valid slot exists.
//
// The selected slot arrives by value.  A fallback that cleared the row in place
// would have to read the slot it intends to keep *before* the clear, and the two
// ways of writing that are easy to swap; passing the slot into the single row
// factory makes the ordering unrepresentable.
template <typename Float>
[[nodiscard]] KOKKOS_INLINE_FUNCTION ReferenceWeightRow<Float> reference_fallback_row(
    const std::int64_t nearest_slot) noexcept
{
    ReferenceWeightRow<Float> row;
    row.nearest_slot = nearest_slot;
    row.fallback_slot = nearest_slot;
    return row;
}

template <typename Float, class ReferenceView, class MaskView>
[[nodiscard]] KOKKOS_INLINE_FUNCTION ReferenceWeightRow<Float> reference_weight_row(
    const std::int64_t slots, const std::int64_t row, const Float coordination,
    const ReferenceView& cn_ref, const MaskView& valid) noexcept
{
    ReferenceWeightRow<Float> result;
    const Float safe_delta =
        Kokkos::sqrt(std::numeric_limits<Float>::max() / Float(2));
    bool requires_fallback = false;
    bool have_weight = false;
    Float maximum_log_weight = Float(0);
    Float nearest_distance = Float(0);
    for (std::int64_t slot = 0; slot < slots; ++slot) {
        if (valid(row + slot) == 0) {
            continue;
        }
        const Float delta = coordination - cn_ref(row + slot);
        const Float distance = delta * delta;
        if (!(Kokkos::abs(delta) <= safe_delta) || !is_finite(distance)) {
            requires_fallback = true;
        }
        const Float log_weight = -Float(reference_weight_k3) * distance;
        if (!is_finite(log_weight)) {
            requires_fallback = true;
        }
        if (!have_weight || log_weight > maximum_log_weight) {
            maximum_log_weight = log_weight;
            have_weight = true;
        }
        if (result.nearest_slot < 0 || distance < nearest_distance) {
            result.nearest_slot = slot;
            nearest_distance = distance;
        }
        result.delta[slot] = delta;
    }
    if (result.nearest_slot < 0) {
        // A species with no valid slot has no reference row at all; the caller
        // validates that this cannot happen for a covered species, and here it
        // simply yields zeros.
        return result;
    }
    if (requires_fallback || !have_weight || !is_finite(maximum_log_weight)) {
        return reference_fallback_row<Float>(result.nearest_slot);
    }
    for (std::int64_t slot = 0; slot < slots; ++slot) {
        if (valid(row + slot) == 0) {
            continue;
        }
        const Float delta = result.delta[slot];
        const Float weight = Kokkos::exp(
            -Float(reference_weight_k3) * delta * delta - maximum_log_weight);
        result.weight[slot] = weight;
        result.weight_derivative[slot] =
            -Float(2 * reference_weight_k3) * delta * weight;
        result.denominator += weight;
        result.denominator_derivative += result.weight_derivative[slot];
    }
    if (!(result.denominator > Float(0)) || !is_finite(result.denominator)) {
        // Unreachable with the shifted weights above: the shift makes the
        // largest valid weight exactly one, so the denominator is in
        // [1, slots] and finite.  It stays as a guard, and it must still report
        // the slot the scan selected.
        return reference_fallback_row<Float>(result.nearest_slot);
    }
    return result;
}

}  // namespace kernel

// Host tables arrive as float64 whatever the workspace precision, and they are
// stored in the workspace precision.  Validation therefore inspects the value
// the workspace will actually hold: a value that is finite as a float64 but
// overflows float32 would be stored as an infinity and would silently change a
// kernel branch - an infinity distance would be masked out of the cutoff, an
// infinity radius would poison every covalent sum.  Such a value is rejected up
// front, whether or not its slot is masked.
template <typename Float>
[[nodiscard]] inline bool is_representable_host(const double value) noexcept
{
    return kernel::is_finite<Float>(static_cast<Float>(value));
}

// ---------------------------------------------------------------------------
// Workspace geometry and accounting
// ---------------------------------------------------------------------------

struct CoordinationGeometry {
    std::int64_t active_nodes = 0;
    std::int64_t active_edges = 0;
    std::int64_t n_species = 0;
    std::int64_t n_ref = 0;
    std::int64_t rank_total = 0;
    std::int64_t active_rank_chunk = 0;
    std::int64_t rank_chunk_bound = 0;

    friend bool operator==(
        const CoordinationGeometry&, const CoordinationGeometry&) = default;
};

struct CoordinationReserveOutcome {
    bool grown = false;
    CoordinationGeometry geometry;
};

struct CoordinationUsage {
    std::vector<BufferUsage> buffers;
    std::uint64_t active_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t reservations = 0;
    std::uint64_t allocations = 0;
    CoordinationGeometry geometry;
};

// One reference weight row, reported verbatim by the diagnostic probe below.
template <typename Float>
struct ReferenceRowProbe {
    std::vector<Float> delta;
    std::vector<Float> weight;
    std::vector<Float> weight_derivative;
    Float denominator = Float(0);
    Float denominator_derivative = Float(0);
    std::int64_t nearest_slot = -1;
    std::int64_t fallback_slot = -1;
};

// ---------------------------------------------------------------------------
// Grow-only coordination workspace
// ---------------------------------------------------------------------------

// Grow-only storage for the modified-coordination and coefficient stages.
//
// Every retained buffer only ever grows: the graph, reference, and coefficient
// buffers are sized once for the configured rank chunk bound and reused, so a
// repeated step allocates nothing.  The coefficient buffers scale with
// `num_nodes * requested_chunk` and never with `num_nodes * rank_total`.
//
// `reserve` is all-or-nothing.  Every replacement buffer is allocated into a
// local candidate first and the committed state (views, ids, counters, active
// geometry) is replaced only after all of them succeed, so a failed reserve
// leaves the workspace exactly as it was.
template <typename Float>
class CoordinationWorkspace {
public:
    using int_view = Kokkos::View<std::int32_t*>;
    using float_view = Kokkos::View<Float*>;
    using mask_view = Kokkos::View<std::uint8_t*>;

    explicit CoordinationWorkspace(std::int64_t rank_chunk_bound);
    ~CoordinationWorkspace() = default;

    CoordinationWorkspace(const CoordinationWorkspace&) = delete;
    CoordinationWorkspace& operator=(const CoordinationWorkspace&) = delete;
    CoordinationWorkspace(CoordinationWorkspace&&) = delete;
    CoordinationWorkspace& operator=(CoordinationWorkspace&&) = delete;

    [[nodiscard]] static constexpr std::string_view precision_token() noexcept
    {
        if constexpr (std::is_same_v<Float, float>) {
            return "float32";
        } else {
            static_assert(std::is_same_v<Float, double>, "unsupported FourierD3 precision");
            return "float64";
        }
    }

    CoordinationReserveOutcome reserve(
        std::int64_t nodes,
        std::int64_t edges,
        std::int64_t n_species,
        std::int64_t n_ref,
        std::int64_t rank_total,
        std::int64_t rank_chunk);

    // Everything a successful `reserve` would install, allocated and validated
    // but not yet installed.  `plan_reserve` allocates and throws; `commit`
    // neither allocates nor throws.  The split exists so the reverse path
    // can prepare this workspace and the composed forward path before it installs
    // either, and so cannot leave one of them grown and the other untouched when
    // the second one refuses.
    struct Reservation {
        int_view offsets;
        int_view sources;
        float_view distances;
        int_view numbers;
        float_view covalent_radii;
        float_view cn_ref;
        mask_view valid;
        int_view species;
        float_view v_q;
        float_view coordination;
        float_view edge_values;
        float_view edge_radial_derivatives;
        float_view coefficients;
        float_view coefficient_derivatives;
        // One flag per buffer above, in the same order.
        std::array<bool, 14> grown = {};
        CoordinationGeometry geometry;
        std::uint64_t serial = 0;
        std::uint64_t allocations = 0;
        bool dimensions_changed = false;
    };

    [[nodiscard]] Reservation plan_reserve(
        std::int64_t nodes,
        std::int64_t edges,
        std::int64_t n_species,
        std::int64_t n_ref,
        std::int64_t rank_total,
        std::int64_t rank_chunk) const;
    void commit(Reservation&& reservation);

    // Receiver-major directed graph.  Both orientations of every pair are a
    // caller precondition: they are neither manufactured nor divided by two.
    void set_graph(
        std::span<const std::int32_t> offsets,
        std::span<const std::int32_t> sources,
        std::span<const double> distances,
        std::span<const std::int32_t> numbers,
        std::span<const double> covalent_radii);

    // Refreshes the stored edge distances of an already loaded graph.  The
    // topology is not accepted again; this is what lets a geometry update keep
    // the receiver-major edge list while replacing the lengths it measures.
    void set_distances(std::span<const double> distances);

    // Covered species (ascending, one channel per entry) with their reference
    // coordination numbers, validity mask, and signed low-rank factors.
    void set_reference(
        std::span<const double> cn_ref,
        std::span<const bool> valid,
        std::span<const std::int32_t> species,
        std::span<const double> v_q);

    // Replaces the coordination buffer with caller-supplied values.  This is the
    // private injection seam for the coefficient stage: the forward path fills
    // the same buffer through `run_coordination`, and a diagnostic or test may
    // supply an arbitrary (including non-finite) coordination number to exercise
    // the fallback.
    void set_coordination(std::span<const double> values);

    void run_coordination(double cutoff);
    void run_coefficients(std::int64_t rank_begin);

    [[nodiscard]] CoordinationGeometry geometry() const noexcept { return geometry_; }
    [[nodiscard]] CoordinationUsage usage() const;

    [[nodiscard]] std::vector<Float> coordination() const;
    [[nodiscard]] std::vector<Float> edge_values() const;
    [[nodiscard]] std::vector<Float> edge_radial_derivatives() const;
    [[nodiscard]] std::vector<Float> coefficients() const;
    [[nodiscard]] std::vector<Float> coefficient_derivatives() const;

    // Diagnostic seam: evaluates one reference weight row on the device through
    // the same kernel the contraction uses and reports it verbatim.  The row's
    // internal stability - the exponent shift, the denominator, and the slot a
    // fallback selected - is invisible in the contracted coefficients, because
    // the shift cancels algebraically whenever the fallback is correct; this
    // probe is not part of the forward path.
    [[nodiscard]] ReferenceRowProbe<Float> debug_reference_row(
        std::span<const double> cn_ref,
        std::span<const bool> valid,
        double coordination) const;

private:
    // The reverse orchestrator reads the retained coordination state -
    // the receiver-major edge list, the stored distances, the coefficient
    // derivatives, and the accumulated coefficients - and runs the receiver-only
    // directed-edge adjoint over it.  The adjoint is defined beside the rest of
    // the reverse path so the whole reverse chain lives in one file; the coupling
    // to this workspace is stated once, here.
    template <typename>
    friend class ReciprocalReverse;

    template <typename View>
    [[nodiscard]] static View grow_candidate(
        const View& current, std::int64_t required, const char* label, std::string_view what);

    template <typename View, typename Source>
    static void copy_to_view(View view, const Source& source, std::size_t count);

    template <typename View>
    [[nodiscard]] static std::vector<Float> read_back(View view, std::int64_t count);

    void require_reserved(std::string_view what) const;
    void require_graph(std::string_view what) const;
    void require_reference(std::string_view what) const;
    void require_coefficients(std::string_view what) const;

    // Declared first so that Kokkos allocations are destroyed before this object
    // stops counting as a live Kokkos owner.
    KokkosLiveObjectGuard kokkos_live_object_guard_;
    CoordinationGeometry geometry_;
    bool graph_loaded_ = false;
    bool reference_loaded_ = false;
    bool coordination_loaded_ = false;
    // Cleared by every input change, so a retained coefficient buffer can never
    // be read as if it belonged to the current inputs.
    bool coefficients_loaded_ = false;
    std::uint64_t reservations_ = 0;
    std::uint64_t allocations_ = 0;
    std::uint64_t allocation_serial_ = 0;

    int_view offsets_;
    int_view sources_;
    float_view distances_;
    int_view numbers_;
    float_view covalent_radii_;
    float_view cn_ref_;
    mask_view valid_;
    int_view species_;
    float_view v_q_;
    float_view coordination_;
    float_view edge_values_;
    float_view edge_radial_derivatives_;
    float_view coefficients_;
    float_view coefficient_derivatives_;

    std::uint64_t offsets_id_ = 0;
    std::uint64_t sources_id_ = 0;
    std::uint64_t distances_id_ = 0;
    std::uint64_t numbers_id_ = 0;
    std::uint64_t covalent_radii_id_ = 0;
    std::uint64_t cn_ref_id_ = 0;
    std::uint64_t valid_id_ = 0;
    std::uint64_t species_id_ = 0;
    std::uint64_t v_q_id_ = 0;
    std::uint64_t coordination_id_ = 0;
    std::uint64_t edge_values_id_ = 0;
    std::uint64_t edge_radial_derivatives_id_ = 0;
    std::uint64_t coefficients_id_ = 0;
    std::uint64_t coefficient_derivatives_id_ = 0;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

template <typename Float>
CoordinationWorkspace<Float>::CoordinationWorkspace(const std::int64_t rank_chunk_bound)
{
    geometry_.rank_chunk_bound = static_cast<std::int64_t>(
        detail::checked_extent(rank_chunk_bound, "FourierD3 rank chunk bound"));
}

template <typename Float>
void CoordinationWorkspace<Float>::require_reserved(const std::string_view what) const
{
    if (geometry_.active_nodes < 1) {
        throw std::runtime_error(
            "FourierD3 coordination workspace holds no reservation; reserve the "
            "geometry before " + std::string(what) + ".");
    }
}

template <typename Float>
void CoordinationWorkspace<Float>::require_graph(const std::string_view what) const
{
    require_reserved(what);
    if (!graph_loaded_) {
        throw std::runtime_error(
            "FourierD3 coordination workspace holds no graph; load the "
            "receiver-major graph before " + std::string(what) + ".");
    }
}

template <typename Float>
void CoordinationWorkspace<Float>::require_reference(const std::string_view what) const
{
    require_reserved(what);
    if (!reference_loaded_) {
        throw std::runtime_error(
            "FourierD3 coordination workspace holds no reference tables; load "
            "them before " + std::string(what) + ".");
    }
    if (!coordination_loaded_) {
        throw std::runtime_error(
            "FourierD3 coefficient contraction needs coordination numbers; run "
            "the modified coordination kernel first, or supply them explicitly, "
            "before " + std::string(what) + ".");
    }
}

template <typename Float>
void CoordinationWorkspace<Float>::require_coefficients(const std::string_view what) const
{
    require_reference(what);
    if (!coefficients_loaded_) {
        throw std::runtime_error(
            "FourierD3 coordination workspace holds no separable coefficients for "
            "the current inputs; run the coefficient contraction before "
            + std::string(what) + ".");
    }
}

template <typename Float>
template <typename View>
View CoordinationWorkspace<Float>::grow_candidate(
    const View& current, const std::int64_t required, const char* label,
    const std::string_view what)
{
    if (required <= static_cast<std::int64_t>(current.extent(0))) {
        return current;
    }
    detail::maybe_fail_allocation(what);
    return View(
        Kokkos::view_alloc(std::string(label)),
        static_cast<std::size_t>(required));
}

template <typename Float>
template <typename View, typename Source>
void CoordinationWorkspace<Float>::copy_to_view(
    View view, const Source& source, const std::size_t count)
{
    if (count == 0) {
        return;
    }
    auto active = Kokkos::subview(view, std::make_pair(std::size_t(0), count));
    auto host = Kokkos::create_mirror_view(active);
    for (std::size_t index = 0; index < count; ++index) {
        host(index) = static_cast<typename View::value_type>(source[index]);
    }
    Kokkos::deep_copy(active, host);
}

template <typename Float>
template <typename View>
std::vector<Float> CoordinationWorkspace<Float>::read_back(
    View view, const std::int64_t count)
{
    if (count <= 0) {
        return {};
    }
    auto active =
        Kokkos::subview(view, std::make_pair(std::size_t(0), static_cast<std::size_t>(count)));
    auto host = Kokkos::create_mirror_view(active);
    Kokkos::deep_copy(host, active);
    return std::vector<Float>(host.data(), host.data() + static_cast<std::size_t>(count));
}

template <typename Float>
CoordinationReserveOutcome CoordinationWorkspace<Float>::reserve(
    const std::int64_t nodes,
    const std::int64_t edges,
    const std::int64_t n_species,
    const std::int64_t n_ref,
    const std::int64_t rank_total,
    const std::int64_t rank_chunk)
{
    Reservation reservation = plan_reserve(
        nodes, edges, n_species, n_ref, rank_total, rank_chunk);
    const bool grown = std::any_of(
        reservation.grown.begin(), reservation.grown.end(), [](const bool flag) { return flag; });
    const CoordinationReserveOutcome outcome{grown, reservation.geometry};
    commit(std::move(reservation));
    return outcome;
}

template <typename Float>
auto CoordinationWorkspace<Float>::plan_reserve(
    const std::int64_t nodes,
    const std::int64_t edges,
    const std::int64_t n_species,
    const std::int64_t n_ref,
    const std::int64_t rank_total,
    const std::int64_t rank_chunk) const -> Reservation
{
    // Every shape, bound, and 64-bit product is validated before the first
    // allocation, so a rejected request leaves the workspace untouched.
    static_cast<void>(detail::checked_extent(nodes, "FourierD3 coordination node count"));
    if (edges < 0) {
        throw std::invalid_argument(
            "FourierD3 coordination edge count must not be negative.");
    }
    static_cast<void>(detail::checked_extent(n_species, "FourierD3 species count"));
    static_cast<void>(detail::checked_extent(n_ref, "FourierD3 reference slot count"));
    if (n_ref > max_reference_slots) {
        throw std::invalid_argument(
            "FourierD3 species hold at most " + std::to_string(max_reference_slots)
            + " reference slots; " + std::to_string(n_ref) + " were requested.");
    }
    static_cast<void>(detail::checked_extent(rank_total, "FourierD3 rank total"));
    static_cast<void>(detail::checked_extent(rank_chunk, "FourierD3 rank chunk"));
    if (rank_chunk > geometry_.rank_chunk_bound) {
        throw std::invalid_argument(
            "FourierD3 rank chunk bound is "
            + std::to_string(geometry_.rank_chunk_bound) + " rank channel(s); "
            + std::to_string(rank_chunk)
            + " were requested, and the request is rejected rather than clipped.");
    }
    if (rank_chunk > rank_total) {
        throw std::invalid_argument(
            "FourierD3 rank chunk of " + std::to_string(rank_chunk)
            + " rank channel(s) is wider than the supplied rank of "
            + std::to_string(rank_total)
            + " slots; a rank chunk cannot span more channels than the rank holds.");
    }

    // Buffer counts.  The flat Kokkos index a kernel can address is narrower than
    // the 64-bit count arithmetic above, so every count is checked against it
    // before the allocation is attempted.
    const auto flat = [](const std::int64_t elements, const std::string_view name) {
        return detail::checked_flat_index(elements, name);
    };
    const std::int64_t flat_nodes = flat(nodes, "numbers");
    const std::int64_t flat_offsets = flat(flat_nodes + 1, "offsets");
    const std::int64_t flat_sources = flat(edges, "sources");
    const std::int64_t flat_species = flat(n_species, "species");
    const std::int64_t flat_reference =
        flat(detail::checked_product(n_species, n_ref, "FourierD3 reference slots"), "cn_ref");
    const std::int64_t flat_factors = flat(
        detail::checked_product(flat_reference, rank_total, "FourierD3 low-rank factor table"),
        "v_q");
    const std::int64_t flat_coefficients = flat(
        detail::checked_product(flat_nodes, rank_chunk, "FourierD3 coefficient chunk"),
        "coefficients");

    // Allocate every replacement into a local candidate first: a failure of any
    // one of them - including the last - commits nothing.
    Reservation reservation;
    reservation.offsets = grow_candidate(
        offsets_, flat_offsets, "symmetrix_fourier_d3_receiver_offsets", "the receiver offsets");
    reservation.sources =
        grow_candidate(sources_, flat_sources, "symmetrix_fourier_d3_sources", "the edge sources");
    reservation.distances = grow_candidate(
        distances_, flat_sources, "symmetrix_fourier_d3_edge_distances", "the edge distances");
    reservation.numbers =
        grow_candidate(numbers_, flat_nodes, "symmetrix_fourier_d3_numbers", "the atomic numbers");
    reservation.covalent_radii = grow_candidate(
        covalent_radii_, flat_nodes, "symmetrix_fourier_d3_covalent_radii",
        "the covalent radii");
    reservation.cn_ref = grow_candidate(
        cn_ref_, flat_reference, "symmetrix_fourier_d3_reference_coordination",
        "the reference coordination numbers");
    reservation.valid = grow_candidate(
        valid_, flat_reference, "symmetrix_fourier_d3_reference_validity",
        "the reference validity mask");
    reservation.species =
        grow_candidate(species_, flat_species, "symmetrix_fourier_d3_species", "the species map");
    reservation.v_q =
        grow_candidate(v_q_, flat_factors, "symmetrix_fourier_d3_low_rank_factors", "the factors");
    reservation.coordination = grow_candidate(
        coordination_, flat_nodes, "symmetrix_fourier_d3_coordination",
        "the coordination numbers");
    reservation.edge_values = grow_candidate(
        edge_values_, flat_sources, "symmetrix_fourier_d3_edge_values", "the edge values");
    reservation.edge_radial_derivatives = grow_candidate(
        edge_radial_derivatives_, flat_sources, "symmetrix_fourier_d3_edge_derivatives",
        "the edge radial derivatives");
    reservation.coefficients = grow_candidate(
        coefficients_, flat_coefficients, "symmetrix_fourier_d3_coefficients",
        "the separable coefficients");
    reservation.coefficient_derivatives = grow_candidate(
        coefficient_derivatives_, flat_coefficients,
        "symmetrix_fourier_d3_coefficient_derivatives", "the coefficient derivatives");

    // One flag per buffer, in the member order above.
    reservation.grown = {
        flat_offsets > static_cast<std::int64_t>(offsets_.extent(0)),
        flat_sources > static_cast<std::int64_t>(sources_.extent(0)),
        flat_sources > static_cast<std::int64_t>(distances_.extent(0)),
        flat_nodes > static_cast<std::int64_t>(numbers_.extent(0)),
        flat_nodes > static_cast<std::int64_t>(covalent_radii_.extent(0)),
        flat_reference > static_cast<std::int64_t>(cn_ref_.extent(0)),
        flat_reference > static_cast<std::int64_t>(valid_.extent(0)),
        flat_species > static_cast<std::int64_t>(species_.extent(0)),
        flat_factors > static_cast<std::int64_t>(v_q_.extent(0)),
        flat_nodes > static_cast<std::int64_t>(coordination_.extent(0)),
        flat_sources > static_cast<std::int64_t>(edge_values_.extent(0)),
        flat_sources > static_cast<std::int64_t>(edge_radial_derivatives_.extent(0)),
        flat_coefficients > static_cast<std::int64_t>(coefficients_.extent(0)),
        flat_coefficients > static_cast<std::int64_t>(coefficient_derivatives_.extent(0)),
    };

    reservation.dimensions_changed = geometry_.active_nodes != nodes
        || geometry_.active_edges != edges || geometry_.n_species != n_species
        || geometry_.n_ref != n_ref || geometry_.rank_total != rank_total;
    reservation.serial = allocation_serial_;
    reservation.allocations = allocations_;
    for (const bool flag : reservation.grown) {
        if (flag) {
            ++reservation.serial;
            ++reservation.allocations;
        }
    }
    reservation.geometry = geometry_;
    reservation.geometry.active_nodes = nodes;
    reservation.geometry.active_edges = edges;
    reservation.geometry.n_species = n_species;
    reservation.geometry.n_ref = n_ref;
    reservation.geometry.rank_total = rank_total;
    reservation.geometry.active_rank_chunk = rank_chunk;
    return reservation;
}

template <typename Float>
void CoordinationWorkspace<Float>::commit(Reservation&& reservation)
{
    // Nothing below allocates or throws, so the workspace moves from one
    // consistent state to the next.
    std::uint64_t serial = allocation_serial_;
    std::uint64_t committed = allocations_;
    const auto install = [&](auto& target, auto candidate, const bool was_grown,
                             std::uint64_t& id) {
        if (was_grown) {
            target = candidate;
            id = ++serial;
            ++committed;
        }
    };
    install(
        offsets_, reservation.offsets, reservation.grown[0], offsets_id_);
    install(
        sources_, reservation.sources, reservation.grown[1], sources_id_);
    install(
        distances_, reservation.distances, reservation.grown[2], distances_id_);
    install(
        numbers_, reservation.numbers, reservation.grown[3], numbers_id_);
    install(
        covalent_radii_, reservation.covalent_radii, reservation.grown[4], covalent_radii_id_);
    install(
        cn_ref_, reservation.cn_ref, reservation.grown[5], cn_ref_id_);
    install(
        valid_, reservation.valid, reservation.grown[6], valid_id_);
    install(
        species_, reservation.species, reservation.grown[7], species_id_);
    install(
        v_q_, reservation.v_q, reservation.grown[8], v_q_id_);
    install(
        coordination_, reservation.coordination, reservation.grown[9], coordination_id_);
    install(
        edge_values_, reservation.edge_values, reservation.grown[10], edge_values_id_);
    install(
        edge_radial_derivatives_, reservation.edge_radial_derivatives,
        reservation.grown[11], edge_radial_derivatives_id_);
    install(
        coefficients_, reservation.coefficients, reservation.grown[12], coefficients_id_);
    install(
        coefficient_derivatives_, reservation.coefficient_derivatives,
        reservation.grown[13], coefficient_derivatives_id_);
    allocation_serial_ = serial;
    allocations_ = committed;
    ++reservations_;
    if (reservation.dimensions_changed) {
        // The chunk width does not change what a loaded graph or reference table
        // means, but any node, edge, species, reference, or rank dimension does.
        graph_loaded_ = false;
        reference_loaded_ = false;
        coordination_loaded_ = false;
    }
    // Every successful reserve re-declares the coefficient geometry, node count
    // and chunk width together, so a retained coefficient buffer is not the
    // current output any more.  A chunk-only change is enough: growing reallocates
    // that buffer, and shrinking keeps the storage but changes the node stride, so
    // a stale read would return unwritten or mis-indexed values.  This sits in the
    // commit path, after every allocation has succeeded, so a failed reserve
    // leaves previously valid outputs readable.
    coefficients_loaded_ = false;
    geometry_ = reservation.geometry;
}

template <typename Float>
void CoordinationWorkspace<Float>::set_graph(
    const std::span<const std::int32_t> offsets,
    const std::span<const std::int32_t> sources,
    const std::span<const double> distances,
    const std::span<const std::int32_t> numbers,
    const std::span<const double> covalent_radii)
{
    require_reserved("loading the graph");
    const auto nodes = geometry_.active_nodes;
    const auto edges = geometry_.active_edges;
    const auto size = [](const auto value) { return static_cast<std::size_t>(value); };
    if (offsets.size() != size(nodes) + 1) {
        throw std::invalid_argument(
            "FourierD3 offsets hold one entry per receiver plus one; "
            + std::to_string(offsets.size()) + " were supplied for "
            + std::to_string(nodes) + " receivers.");
    }
    if (offsets.front() != 0) {
        throw std::invalid_argument("FourierD3 offsets must start at zero.");
    }
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        if (offsets[index] < 0) {
            throw std::invalid_argument("FourierD3 offsets must not be negative.");
        }
        if (index > 0 && offsets[index] < offsets[index - 1]) {
            throw std::invalid_argument(
                "FourierD3 offsets must be non-decreasing in receiver order.");
        }
    }
    if (offsets.back() != edges) {
        throw std::invalid_argument(
            "FourierD3 offsets must end at the edge count; they end at "
            + std::to_string(offsets.back()) + " for " + std::to_string(edges)
            + " edges.");
    }
    if (sources.size() != size(edges)) {
        throw std::invalid_argument(
            "FourierD3 sources hold one entry per directed edge; "
            + std::to_string(sources.size()) + " were supplied for "
            + std::to_string(edges) + " edges.");
    }
    for (const std::int32_t source : sources) {
        if (source < 0 || source >= nodes) {
            throw std::invalid_argument(
                "FourierD3 sources must index a receiver; got " + std::to_string(source)
                + " for " + std::to_string(nodes) + " receivers.");
        }
    }
    if (distances.size() != size(edges)) {
        throw std::invalid_argument(
            "FourierD3 distances hold one entry per directed edge; "
            + std::to_string(distances.size()) + " were supplied for "
            + std::to_string(edges) + " edges.");
    }
    for (const double distance : distances) {
        if (!is_representable_host<Float>(distance)) {
            throw std::invalid_argument("FourierD3 distances must be finite.");
        }
        if (distance < 0.0) {
            throw std::invalid_argument("FourierD3 distances must not be negative.");
        }
    }
    if (numbers.size() != size(nodes)) {
        throw std::invalid_argument(
            "FourierD3 atomic numbers hold one entry per receiver; "
            + std::to_string(numbers.size()) + " were supplied for "
            + std::to_string(nodes) + " receivers.");
    }
    for (const std::int32_t number : numbers) {
        if (number < 0) {
            throw std::invalid_argument("FourierD3 atomic numbers must not be negative.");
        }
    }
    if (covalent_radii.size() != size(nodes)) {
        throw std::invalid_argument(
            "FourierD3 covalent radii hold one entry per receiver; "
            + std::to_string(covalent_radii.size()) + " were supplied for "
            + std::to_string(nodes) + " receivers.");
    }
    for (const double radius : covalent_radii) {
        if (!is_representable_host<Float>(radius) || radius < 0.0) {
            throw std::invalid_argument(
                "FourierD3 covalent radii must be finite and non-negative.");
        }
    }

    copy_to_view(offsets_, offsets, offsets.size());
    copy_to_view(sources_, sources, sources.size());
    copy_to_view(distances_, distances, distances.size());
    copy_to_view(numbers_, numbers, numbers.size());
    copy_to_view(covalent_radii_, covalent_radii, covalent_radii.size());
    graph_loaded_ = true;
    coordination_loaded_ = false;
    coefficients_loaded_ = false;
}

template <typename Float>
void CoordinationWorkspace<Float>::set_distances(
    const std::span<const double> distances)
{
    require_graph("refreshing the edge distances");
    const auto edges = geometry_.active_edges;
    if (distances.size() != static_cast<std::size_t>(edges)) {
        throw std::invalid_argument(
            "FourierD3 distances hold one entry per directed edge; "
            + std::to_string(distances.size()) + " were supplied for "
            + std::to_string(edges) + " edges.");
    }
    for (const double distance : distances) {
        if (!is_representable_host<Float>(distance)) {
            throw std::invalid_argument("FourierD3 distances must be finite.");
        }
        if (distance < 0.0) {
            throw std::invalid_argument("FourierD3 distances must not be negative.");
        }
    }
    copy_to_view(distances_, distances, distances.size());
    // The coordination number, the separable coefficients, and every stage that
    // reads them follow the distances, so a refreshed geometry invalidates them.
    coordination_loaded_ = false;
    coefficients_loaded_ = false;
}

template <typename Float>
void CoordinationWorkspace<Float>::set_reference(
    const std::span<const double> cn_ref,
    const std::span<const bool> valid,
    const std::span<const std::int32_t> species,
    const std::span<const double> v_q)
{
    require_reserved("loading the reference tables");
    const auto n_species = geometry_.n_species;
    const auto n_ref = geometry_.n_ref;
    const auto rank_total = geometry_.rank_total;
    const auto size = [](const auto value) { return static_cast<std::size_t>(value); };
    if (species.size() != size(n_species)) {
        throw std::invalid_argument(
            "FourierD3 species table holds one entry per covered species; "
            + std::to_string(species.size()) + " were supplied for "
            + std::to_string(n_species) + " species.");
    }
    for (std::size_t channel = 0; channel < species.size(); ++channel) {
        if (species[channel] <= 0) {
            throw std::invalid_argument(
                "FourierD3 species must be positive atomic numbers; got "
                + std::to_string(species[channel]) + ".");
        }
        if (channel > 0 && species[channel] <= species[channel - 1]) {
            throw std::invalid_argument(
                "FourierD3 species must be strictly ascending, so that a channel "
                "index is unique and independent of the caller's order.");
        }
    }
    if (cn_ref.size() != size(n_species) * size(n_ref)) {
        throw std::invalid_argument(
            "FourierD3 cn_ref must hold n_species * n_ref entries; "
            + std::to_string(cn_ref.size()) + " were supplied for "
            + std::to_string(n_species) + " species and " + std::to_string(n_ref)
            + " reference slots.");
    }
    if (valid.size() != cn_ref.size()) {
        throw std::invalid_argument(
            "FourierD3 valid must hold one mask entry per reference slot; "
            + std::to_string(valid.size()) + " were supplied for "
            + std::to_string(cn_ref.size()) + " slots.");
    }
    if (v_q.size() != cn_ref.size() * size(rank_total)) {
        throw std::invalid_argument(
            "FourierD3 v_q must hold n_species * n_ref * rank entries; "
            + std::to_string(v_q.size()) + " were supplied for "
            + std::to_string(cn_ref.size()) + " reference slots and "
            + std::to_string(rank_total) + " rank slots.");
    }
    for (const double reference : cn_ref) {
        if (!is_representable_host<Float>(reference)) {
            throw std::invalid_argument(
                "FourierD3 cn_ref must be finite in every slot, including the "
                "sentinel slots it masks.");
        }
    }
    for (const double factor : v_q) {
        if (!is_representable_host<Float>(factor)) {
            throw std::invalid_argument("FourierD3 v_q factors must be finite.");
        }
    }
    for (std::int64_t channel = 0; channel < n_species; ++channel) {
        bool covered = false;
        for (std::int64_t slot = 0; slot < n_ref; ++slot) {
            covered = covered || valid[size(channel) * size(n_ref) + size(slot)];
        }
        if (!covered) {
            throw std::invalid_argument(
                "FourierD3 species at channel " + std::to_string(channel)
                + " holds no valid reference; a covered species needs at least "
                "one valid reference slot.");
        }
    }

    copy_to_view(cn_ref_, cn_ref, cn_ref.size());
    copy_to_view(valid_, valid, valid.size());
    copy_to_view(species_, species, species.size());
    copy_to_view(v_q_, v_q, v_q.size());
    reference_loaded_ = true;
    coefficients_loaded_ = false;
}

template <typename Float>
void CoordinationWorkspace<Float>::set_coordination(const std::span<const double> values)
{
    require_graph("supplying coordination numbers");
    if (values.size() != static_cast<std::size_t>(geometry_.active_nodes)) {
        throw std::invalid_argument(
            "FourierD3 coordination numbers hold one entry per receiver; "
            + std::to_string(values.size()) + " were supplied for "
            + std::to_string(geometry_.active_nodes) + " receivers.");
    }
    copy_to_view(coordination_, values, values.size());
    coordination_loaded_ = true;
    coefficients_loaded_ = false;
}

template <typename Float>
void CoordinationWorkspace<Float>::run_coordination(const double cutoff)
{
    require_graph("running the coordination kernel");
    if (!is_representable_host<Float>(cutoff) || !(cutoff > 0.0)) {
        throw std::invalid_argument(
            "FourierD3 coordination cutoff must be finite and positive.");
    }
    const Float cutoff_value = static_cast<Float>(cutoff);
    const auto offsets = offsets_;
    const auto sources = sources_;
    const auto distances = distances_;
    const auto numbers = numbers_;
    const auto radii = covalent_radii_;
    const auto edge_values = edge_values_;
    const auto edge_derivatives = edge_radial_derivatives_;
    const auto coordination = coordination_;
    const std::int64_t nodes = geometry_.active_nodes;
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_modified_coordination",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(std::int64_t(0), nodes),
        KOKKOS_LAMBDA(const std::int64_t node) {
            const std::int32_t receiver_number = numbers(static_cast<int>(node));
            const Float receiver_radius = radii(static_cast<int>(node));
            const std::int64_t begin = static_cast<std::int64_t>(offsets(static_cast<int>(node)));
            const std::int64_t end =
                static_cast<std::int64_t>(offsets(static_cast<int>(node) + 1));
            Float total = Float(0);
            for (std::int64_t edge = begin; edge < end; ++edge) {
                const int index = static_cast<int>(edge);
                const std::int32_t source_number = numbers(sources(index));
                kernel::PairTerm<Float> term;
                if (receiver_number > 0 && source_number > 0) {
                    term = kernel::modified_pair_term(
                        distances(index), receiver_radius + radii(sources(index)),
                        cutoff_value);
                }
                edge_values(index) = term.value;
                edge_derivatives(index) = term.radial_derivative;
                total += term.value;
            }
            coordination(static_cast<int>(node)) = total;
        });
    coordination_loaded_ = true;
    coefficients_loaded_ = false;
}

template <typename Float>
void CoordinationWorkspace<Float>::run_coefficients(const std::int64_t rank_begin)
{
    const std::int64_t chunk = geometry_.active_rank_chunk;
    // Written without an addition.  `rank_begin + chunk` overflows for a
    // rank_begin near the integer limit, and a wrapped comparison would accept a
    // window that reaches far outside the rank; `chunk <= rank_total` holds from
    // `reserve`, which keeps the subtraction in range.
    if (rank_begin < 0 || rank_begin > geometry_.rank_total - chunk) {
        throw std::invalid_argument(
            "FourierD3 rank chunk at " + std::to_string(rank_begin) + " of "
            + std::to_string(chunk) + " rank channel(s) leaves the supplied rank of "
            + std::to_string(geometry_.rank_total) + " slots.");
    }
    require_reference("running the coefficient contraction");
    const auto numbers = numbers_;
    const auto species = species_;
    const auto cn_ref = cn_ref_;
    const auto valid = valid_;
    const auto v_q = v_q_;
    const auto coordination = coordination_;
    const auto coefficients = coefficients_;
    const auto coefficient_derivatives = coefficient_derivatives_;
    const std::int64_t nodes = geometry_.active_nodes;
    const std::int64_t n_species = geometry_.n_species;
    const std::int64_t n_ref = geometry_.n_ref;
    const std::int64_t rank_total = geometry_.rank_total;
    const std::int64_t begin = rank_begin;
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_separable_coefficients",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(std::int64_t(0), nodes),
        KOKKOS_LAMBDA(const std::int64_t node) {
            const std::int64_t base = node * chunk;
            for (std::int64_t slot = 0; slot < chunk; ++slot) {
                coefficients(base + slot) = Float(0);
                coefficient_derivatives(base + slot) = Float(0);
            }
            const std::int32_t number = numbers(static_cast<int>(node));
            std::int64_t channel = -1;
            for (std::int64_t candidate = 0; candidate < n_species; ++candidate) {
                if (species(static_cast<int>(candidate)) == number) {
                    channel = candidate;
                    break;
                }
            }
            if (channel < 0) {
                // Padding (Z = 0) and uncovered species drop out of the
                // dispersion sum while still counting towards their neighbours'
                // coordination numbers.
                return;
            }
            const std::int64_t row = channel * n_ref;
            const auto weights = kernel::reference_weight_row<Float>(
                n_ref, row, coordination(static_cast<int>(node)), cn_ref, valid);
            if (weights.fallback_slot >= 0) {
                for (std::int64_t slot = 0; slot < chunk; ++slot) {
                    coefficients(base + slot) =
                        v_q((row + weights.fallback_slot) * rank_total + begin + slot);
                }
                return;
            }
            const Float inverse = Float(1) / weights.denominator;
            for (std::int64_t rank = 0; rank < chunk; ++rank) {
                Float numerator = Float(0);
                Float numerator_derivative = Float(0);
                for (std::int64_t slot = 0; slot < n_ref; ++slot) {
                    if (valid(row + slot) == 0) {
                        continue;
                    }
                    const Float factor = v_q((row + slot) * rank_total + begin + rank);
                    numerator += weights.weight[slot] * factor;
                    numerator_derivative += weights.weight_derivative[slot] * factor;
                }
                const Float coefficient = numerator * inverse;
                coefficients(base + rank) = coefficient;
                coefficient_derivatives(base + rank) =
                    (numerator_derivative - coefficient * weights.denominator_derivative)
                    * inverse;
            }
        });
    coefficients_loaded_ = true;
}

template <typename Float>
ReferenceRowProbe<Float> CoordinationWorkspace<Float>::debug_reference_row(
    const std::span<const double> cn_ref,
    const std::span<const bool> valid,
    const double coordination) const
{
    require_reserved("probing a reference weight row");
    const std::int64_t slots = geometry_.n_ref;
    const auto expected = static_cast<std::size_t>(slots);
    if (cn_ref.size() != expected || valid.size() != expected) {
        throw std::invalid_argument(
            "FourierD3 reference row probe needs one reference value and one "
            "validity flag per reserved slot; "
            + std::to_string(slots) + " slots are reserved, "
            + std::to_string(cn_ref.size()) + " reference values and "
            + std::to_string(valid.size()) + " flags were supplied.");
    }
    // The probe reports a stored reference table's row, so it holds the same
    // representability rule as `set_reference`; the coordination number is a
    // measured/injected value and may be arbitrary, exactly as the graph's own
    // coordination number may be.
    for (std::size_t slot = 0; slot < cn_ref.size(); ++slot) {
        if (!is_representable_host<Float>(cn_ref[slot])) {
            throw std::invalid_argument(
                "FourierD3 reference coordination numbers must be finite in the "
                "workspace precision; slot " + std::to_string(slot) + " holds "
                + std::to_string(cn_ref[slot]) + ".");
        }
    }

    const auto row_cn = float_view("symmetrix_fourier_d3_probe_cn_ref", expected);
    const auto row_valid = mask_view("symmetrix_fourier_d3_probe_valid", expected);
    copy_to_view(row_cn, cn_ref, cn_ref.size());
    {
        auto host = Kokkos::create_mirror_view(row_valid);
        for (std::int64_t slot = 0; slot < slots; ++slot) {
            host(slot) = valid[static_cast<std::size_t>(slot)] ? std::uint8_t(1)
                                                              : std::uint8_t(0);
        }
        Kokkos::deep_copy(row_valid, host);
    }

    const auto delta = float_view("symmetrix_fourier_d3_probe_delta", expected);
    const auto weight = float_view("symmetrix_fourier_d3_probe_weight", expected);
    const auto slope = float_view("symmetrix_fourier_d3_probe_slope", expected);
    const auto scalars = float_view("symmetrix_fourier_d3_probe_scalars", 2);
    const auto selected = Kokkos::View<std::int64_t*>("symmetrix_fourier_d3_probe_slots", 2);
    const Float target = static_cast<Float>(coordination);
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_probe_reference_row",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(std::int64_t(0), std::int64_t(1)),
        KOKKOS_LAMBDA(const std::int64_t) {
            const auto row =
                kernel::reference_weight_row<Float>(slots, 0, target, row_cn, row_valid);
            for (std::int64_t slot = 0; slot < slots; ++slot) {
                delta(slot) = row.delta[slot];
                weight(slot) = row.weight[slot];
                slope(slot) = row.weight_derivative[slot];
            }
            scalars(0) = row.denominator;
            scalars(1) = row.denominator_derivative;
            selected(0) = row.nearest_slot;
            selected(1) = row.fallback_slot;
        });

    ReferenceRowProbe<Float> probe;
    probe.delta = read_back(delta, slots);
    probe.weight = read_back(weight, slots);
    probe.weight_derivative = read_back(slope, slots);
    const auto scalars_host = read_back(scalars, 2);
    probe.denominator = scalars_host[0];
    probe.denominator_derivative = scalars_host[1];
    auto selected_host = Kokkos::create_mirror_view(selected);
    Kokkos::deep_copy(selected_host, selected);
    probe.nearest_slot = selected_host(0);
    probe.fallback_slot = selected_host(1);
    return probe;
}

template <typename Float>
CoordinationUsage CoordinationWorkspace<Float>::usage() const
{
    CoordinationUsage result;
    result.reservations = reservations_;
    result.allocations = allocations_;
    result.geometry = geometry_;
    const std::int64_t nodes = geometry_.active_nodes;
    const std::int64_t edges = geometry_.active_edges;
    const std::int64_t references = geometry_.n_species * geometry_.n_ref;
    const std::int64_t factors = references * geometry_.rank_total;
    const std::int64_t coefficients = nodes * geometry_.active_rank_chunk;
    const auto record = [&result](
                            const std::string_view name, const std::int64_t active,
                            const std::int64_t capacity, const std::uint64_t element_size,
                            const std::uint64_t id) {
        result.buffers.push_back(BufferUsage{
            std::string(name),
            active,
            capacity,
            detail::element_bytes(active, element_size),
            detail::element_bytes(capacity, element_size),
            id});
    };
    // An unreserved workspace has no active receiver row at all, so its reported
    // extents start at zero rather than at the receiver count plus one.
    record(
        "offsets", nodes > 0 ? nodes + 1 : 0,
        static_cast<std::int64_t>(offsets_.extent(0)), sizeof(std::int32_t),
        offsets_id_);
    record(
        "sources", edges, static_cast<std::int64_t>(sources_.extent(0)),
        sizeof(std::int32_t), sources_id_);
    record(
        "distances", edges, static_cast<std::int64_t>(distances_.extent(0)), sizeof(Float),
        distances_id_);
    record(
        "numbers", nodes, static_cast<std::int64_t>(numbers_.extent(0)),
        sizeof(std::int32_t), numbers_id_);
    record(
        "covalent_radii", nodes,
        static_cast<std::int64_t>(covalent_radii_.extent(0)), sizeof(Float),
        covalent_radii_id_);
    record(
        "cn_ref", references, static_cast<std::int64_t>(cn_ref_.extent(0)), sizeof(Float),
        cn_ref_id_);
    record(
        "valid", references, static_cast<std::int64_t>(valid_.extent(0)),
        sizeof(std::uint8_t), valid_id_);
    record(
        "species", geometry_.n_species,
        static_cast<std::int64_t>(species_.extent(0)), sizeof(std::int32_t), species_id_);
    record(
        "v_q", factors, static_cast<std::int64_t>(v_q_.extent(0)), sizeof(Float), v_q_id_);
    record(
        "coordination", nodes, static_cast<std::int64_t>(coordination_.extent(0)),
        sizeof(Float), coordination_id_);
    record(
        "edge_values", edges, static_cast<std::int64_t>(edge_values_.extent(0)),
        sizeof(Float), edge_values_id_);
    record(
        "edge_radial_derivatives", edges,
        static_cast<std::int64_t>(edge_radial_derivatives_.extent(0)), sizeof(Float),
        edge_radial_derivatives_id_);
    record(
        "coefficients", coefficients,
        static_cast<std::int64_t>(coefficients_.extent(0)), sizeof(Float),
        coefficients_id_);
    record(
        "coefficient_derivatives", coefficients,
        static_cast<std::int64_t>(coefficient_derivatives_.extent(0)), sizeof(Float),
        coefficient_derivatives_id_);
    detail::finalize_usage(result.buffers, result.active_bytes, result.capacity_bytes);
    return result;
}

template <typename Float>
std::vector<Float> CoordinationWorkspace<Float>::coordination() const
{
    require_graph("reading the coordination numbers");
    return read_back(coordination_, geometry_.active_nodes);
}

template <typename Float>
std::vector<Float> CoordinationWorkspace<Float>::edge_values() const
{
    require_graph("reading the edge switch values");
    return read_back(edge_values_, geometry_.active_edges);
}

template <typename Float>
std::vector<Float> CoordinationWorkspace<Float>::edge_radial_derivatives() const
{
    require_graph("reading the edge radial derivatives");
    return read_back(edge_radial_derivatives_, geometry_.active_edges);
}

template <typename Float>
std::vector<Float> CoordinationWorkspace<Float>::coefficients() const
{
    require_coefficients("reading the separable coefficients");
    return read_back(coefficients_, geometry_.active_nodes * geometry_.active_rank_chunk);
}

template <typename Float>
std::vector<Float> CoordinationWorkspace<Float>::coefficient_derivatives() const
{
    require_coefficients("reading the coefficient derivatives");
    return read_back(
        coefficient_derivatives_,
        geometry_.active_nodes * geometry_.active_rank_chunk);
}

// Both precisions are instantiated once, in this library; consumers only see the
// declarations above.
extern template class CoordinationWorkspace<float>;
extern template class CoordinationWorkspace<double>;

}  // namespace symmetrix::fourier_d3

#endif  // SYMMETRIX_FOURIER_D3_COORDINATION_HPP
