#ifndef SYMMETRIX_PARTICLE_MESH_STORAGE_HPP
#define SYMMETRIX_PARTICLE_MESH_STORAGE_HPP

// Physics-neutral particle-mesh storage, cache keys, and FFT-plan ownership.
//
// This layer owns no physics.  It provides:
//
//   * `MeshShape`      checked 64-bit mesh geometry: extents, the real and
//                      real-to-complex flattened counts, strides, and offsets;
//   * `MeshStorage`    grow-only mesh allocations with active/capacity byte
//                      accounting, bounded by the configured transform batch;
//   * `FftPlanDomain`  the private FFT-plan owner, keyed by backend,
//                      precision, provider, mesh, layout, and batch;
//
// The translation unit is compiled only when an FFT provider is available and
// links `KokkosFFT::fft`; without a discovered provider nothing here exists and
// no stub claims numerical support. Physics-specific spreading, reciprocal
// contraction, and reverse propagation are owned by the consumers.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "tools_kokkos.hpp"

namespace symmetrix::particle_mesh {

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

// Memory order of the two innermost mesh axes.  v1 compiles the right layout
// (transform axis z fastest-varying, spectrum keeping its halved extent),
// which is the layout the FFT backends consume without an internal transpose.
enum class MeshLayout { left, right };

[[nodiscard]] std::string_view layout_token(MeshLayout layout);
[[nodiscard]] MeshLayout layout_from_token(std::string_view token);

// ---------------------------------------------------------------------------
// Itemized byte accounting
// ---------------------------------------------------------------------------

// One retained Kokkos allocation.  `allocation_id` changes whenever the
// allocation is replaced, so callers can tell a reuse from a rebuild.
struct BufferUsage {
    std::string name;
    std::int64_t active_elements = 0;
    std::int64_t capacity_elements = 0;
    std::uint64_t active_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t allocation_id = 0;
};

namespace detail {

// Deterministic allocation fault injection for the private test seam.
//
// `arm_allocation_fault(n)` fails the n-th allocation this layer requests after
// the call (0 = the next one) and disarms itself, so a test can place an
// allocation failure exactly between the two mesh buffers or the two kernel
// tables and prove the all-or-nothing commit.  The hook is observed
// immediately before every Kokkos allocation this layer makes, and it is
// single-threaded test infrastructure: production callers never arm it.  The
// injected failure is a `std::runtime_error` whose message names the injected
// fault, so a test never confuses it with a genuine allocation failure.
void arm_allocation_fault(std::int64_t fail_at);
void clear_allocation_fault();
[[nodiscard]] bool allocation_fault_armed();
void maybe_fail_allocation(std::string_view what);

[[nodiscard]] std::int64_t checked_extent(std::int64_t value, std::string_view what);
// Reject a retained element count that no flat Kokkos range can address.  This
// check happens before allocation so an invalid request cannot first consume a
// very large amount of memory.
[[nodiscard]] std::int64_t checked_flat_index(
    std::int64_t elements, std::string_view what);
// A mesh axis must hold at least two nodes: the assignment stencil is p >= 2
// wide (spec 5.6), and a one-node axis would fold two stencil points onto the
// same mesh node.
[[nodiscard]] std::int64_t checked_axis_extent(std::int64_t value, std::string_view what);
[[nodiscard]] std::int64_t checked_product(
    std::int64_t lhs, std::int64_t rhs, std::string_view what);
[[nodiscard]] std::uint64_t element_bytes(std::int64_t elements, std::uint64_t element_size);
[[nodiscard]] std::string format_scalar(double value);
void require_finite(double value, std::string_view what);
// Sums the itemized buffers once and checks the itemization invariants:
// buffer names are unique, capacity never trails active, and the totals are
// exactly the itemized sum, so an aliased view can never be counted twice.
void finalize_usage(
    std::vector<BufferUsage>& buffers,
    std::uint64_t& active_bytes,
    std::uint64_t& capacity_bytes);

}  // namespace detail

// One committed mesh allocation, shared by the storage that owns it and by
// every plan built from it.  The token carries the two *managed* Kokkos
// allocations, so a plan that retains the token keeps its captured buffers
// allocated for as long as the plan exists - even after the storage is
// destroyed - and the backend plan can never outlive the memory it captured.
// `generation` and the allocation ids move together with the views, so an
// identity comparison can never pair one generation with another generation's
// buffers.
template <typename Float>
struct MeshAllocation {
    std::uint64_t generation = 0;
    std::uint64_t real_id = 0;
    std::uint64_t spectrum_id = 0;
    Kokkos::View<Float*> real;
    Kokkos::View<Kokkos::complex<Float>*> spectrum;

    [[nodiscard]] std::int64_t real_elements() const noexcept
    {
        return static_cast<std::int64_t>(real.extent(0));
    }
    [[nodiscard]] std::int64_t spectrum_elements() const noexcept
    {
        return static_cast<std::int64_t>(spectrum.extent(0));
    }
    // Bytes both retained allocations hold, which is what a plan keeps alive.
    [[nodiscard]] std::uint64_t retained_bytes() const
    {
        return detail::element_bytes(real_elements(), sizeof(Float))
            + detail::element_bytes(spectrum_elements(), sizeof(Kokkos::complex<Float>));
    }
};

// One particle mesh plus the number of meshes a single plan transforms
// together.  Every flattened count, stride product, and offset is computed in
// signed 64-bit arithmetic and rejects overflow instead of clamping, wrapping,
// or truncating to a 32-bit index.
class MeshShape {
public:
    static constexpr int rank = 3;

    MeshShape() = default;
    MeshShape(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t batch,
        MeshLayout layout);

    [[nodiscard]] bool valid() const noexcept { return point_count_ > 0; }
    [[nodiscard]] std::int64_t extent(int axis) const;
    [[nodiscard]] std::int64_t batch() const noexcept { return batch_; }
    [[nodiscard]] MeshLayout layout() const noexcept { return layout_; }
    [[nodiscard]] std::int64_t point_count() const noexcept { return point_count_; }
    [[nodiscard]] std::int64_t real_element_count() const noexcept
    {
        return real_element_count_;
    }
    // Real-to-complex transform: the transform axis keeps nz/2 + 1 bins.
    [[nodiscard]] std::int64_t spectrum_extent() const noexcept { return spectrum_extent_; }
    [[nodiscard]] std::int64_t spectrum_point_count() const noexcept
    {
        return spectrum_point_count_;
    }
    [[nodiscard]] std::int64_t spectrum_element_count() const noexcept
    {
        return spectrum_element_count_;
    }
    [[nodiscard]] std::int64_t stride(int axis) const;
    [[nodiscard]] std::int64_t spectrum_stride(int axis) const;
    [[nodiscard]] std::int64_t offset(
        std::int64_t i, std::int64_t j, std::int64_t k) const;
    [[nodiscard]] std::int64_t spectrum_offset(
        std::int64_t i, std::int64_t j, std::int64_t k) const;
    [[nodiscard]] std::string description() const;

    friend bool operator==(const MeshShape& lhs, const MeshShape& rhs) noexcept
    {
        return lhs.extents_ == rhs.extents_ && lhs.batch_ == rhs.batch_
            && lhs.layout_ == rhs.layout_;
    }
    friend bool operator!=(const MeshShape& lhs, const MeshShape& rhs) noexcept
    {
        return !(lhs == rhs);
    }

private:
    std::array<std::int64_t, rank> extents_{0, 0, 0};
    std::int64_t batch_ = 0;
    MeshLayout layout_ = MeshLayout::right;
    std::int64_t point_count_ = 0;
    std::int64_t real_element_count_ = 0;
    std::int64_t spectrum_extent_ = 0;
    std::int64_t spectrum_point_count_ = 0;
    std::int64_t spectrum_element_count_ = 0;
};

// ---------------------------------------------------------------------------
// Mesh workspace accounting
// ---------------------------------------------------------------------------

struct MeshStorageUsage {
    // Active bytes describe the current problem, capacity bytes the retained
    // allocation.  Each retained buffer appears exactly once: active views are
    // aliases of these allocations and are never counted again.
    std::vector<BufferUsage> buffers;
    std::uint64_t active_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t reservations = 0;
    std::uint64_t allocations = 0;
    std::uint64_t allocation_generation = 0;
    std::int64_t channel_bound = 0;
    std::int64_t active_channels = 0;
    std::int64_t peak_channels = 0;
};

struct MeshReserveOutcome {
    bool grown = false;
    std::uint64_t allocation_generation = 0;
    std::int64_t active_channels = 0;
};

// ---------------------------------------------------------------------------
// Grow-only mesh workspace
// ---------------------------------------------------------------------------

// Grow-only mesh storage for one configured transform batch.
//
// Retained allocations only ever grow, and a smaller batch or a smaller
// mesh reuses them through smaller active views; capacity therefore reaches the
// largest batch actually requested. A batch wider than `channel_bound` is
// rejected rather than clipped. Allocated
// memory is not zeroed: a reused buffer still holds the previous step's values,
// so the forward path owns clearing semantics.
//
// `reserve` is all-or-nothing.  Every replacement allocation is made into a
// local first, and the committed state (allocation, generation, ids,
// capacities, counters, geometry) is replaced only after all of them succeed,
// so a failed reserve leaves the storage bit-for-bit as it was: the retained
// allocation is not freed and its identity does not change, which is what keeps
// a plan built against it valid.  The storage holds the committed allocation
// through a `MeshAllocation` token that plans also retain.
template <typename Float>
class MeshStorage {
public:
    using real_active_view =
        Kokkos::View<Float****, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged>;
    using spectrum_active_view = Kokkos::View<
        Kokkos::complex<Float>****, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged>;

    [[nodiscard]] static constexpr std::string_view precision_token() noexcept
    {
        if constexpr (std::is_same_v<Float, float>) {
            return "float32";
        } else {
            static_assert(std::is_same_v<Float, double>, "unsupported particle-mesh precision");
            return "float64";
        }
    }

    // `channel_bound` is the maximum transform batch width; it is a hard bound,
    // not a hint.
    MeshStorage(MeshLayout layout, std::int64_t channel_bound);
    ~MeshStorage() = default;

    MeshStorage(const MeshStorage&) = delete;
    MeshStorage& operator=(const MeshStorage&) = delete;
    MeshStorage(MeshStorage&&) = delete;
    MeshStorage& operator=(MeshStorage&&) = delete;

    // A prepared storage update.  Composite particle-mesh owners can prepare
    // every child first and install all of them only after every allocation and
    // validation has succeeded.
    struct Reservation {
        std::shared_ptr<const MeshAllocation<Float>> allocation;
        MeshShape shape;
        std::uint64_t serial = 0;
        std::uint64_t allocations = 0;
        std::int64_t peak_channels = 0;
        bool grown = false;
    };

    // Grows the retained allocation to fit the request and makes it active.
    MeshReserveOutcome reserve(
        std::int64_t nx, std::int64_t ny, std::int64_t nz, std::int64_t channels);
    // Validate geometry, the channel bound, and flattened allocation ranges
    // without allocating. Composite owners use this before preparing buffers.
    void validate_reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t channels) const;
    [[nodiscard]] Reservation plan_reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t channels) const;
    void commit(Reservation&& reservation);

    [[nodiscard]] std::optional<MeshShape> shape() const { return shape_; }
    // The committed allocation, or null before the first successful reserve.
    // Callers may hold it: the buffers stay allocated as long as one owner
    // lives, which is how an FFT plan survives its storage.
    [[nodiscard]] const std::shared_ptr<const MeshAllocation<Float>>& allocation() const noexcept
    {
        return allocation_;
    }
    [[nodiscard]] MeshStorageUsage usage() const;

    // Active views alias the retained allocations and are contiguous in the
    // active geometry, which is what the FFT backends require.  Consumer code
    // never sees retained capacity.
    [[nodiscard]] real_active_view real_field() const;
    [[nodiscard]] spectrum_active_view spectrum() const;

private:
    // Declared first so that Kokkos allocations are destroyed before this
    // object stops counting as a live Kokkos owner.
    KokkosLiveObjectGuard kokkos_live_object_guard_;
    MeshLayout layout_ = MeshLayout::right;
    std::int64_t channel_bound_ = 0;
    std::optional<MeshShape> shape_;
    std::int64_t peak_channels_ = 0;
    std::uint64_t reservations_ = 0;
    std::uint64_t allocations_ = 0;
    std::uint64_t allocation_serial_ = 0;
    // Declared last so the retained allocation is released before the guard
    // above stops counting this object as a live Kokkos owner.
    std::shared_ptr<const MeshAllocation<Float>> allocation_;
};

template <typename Float>
MeshStorage<Float>::MeshStorage(MeshLayout layout, std::int64_t channel_bound)
    : layout_(layout),
      channel_bound_(detail::checked_extent(channel_bound, "transform channel bound"))
{
    if (layout_ != MeshLayout::right) {
        throw std::invalid_argument(
            "Particle-mesh storage is compiled for the layout_right transform axis; "
            "a layout_left plan would need an internal transpose, and none is inserted.");
    }
}

template <typename Float>
MeshReserveOutcome MeshStorage<Float>::reserve(
    std::int64_t nx, std::int64_t ny, std::int64_t nz, std::int64_t channels)
{
    Reservation reservation = plan_reserve(nx, ny, nz, channels);
    const MeshReserveOutcome outcome{
        reservation.grown,
        reservation.allocation ? reservation.allocation->generation : 0,
        reservation.shape.batch()};
    commit(std::move(reservation));
    return outcome;
}

template <typename Float>
void MeshStorage<Float>::validate_reserve(
    std::int64_t nx,
    std::int64_t ny,
    std::int64_t nz,
    std::int64_t channels) const
{
    const MeshShape requested(nx, ny, nz, channels, layout_);
    if (requested.batch() > channel_bound_) {
        throw std::invalid_argument(
            "Particle-mesh storage channel bound is "
            + std::to_string(channel_bound_) + " transform channel(s); "
            + std::to_string(requested.batch())
            + " were requested, and the request is rejected rather than clipped.");
    }

    static_cast<void>(
        detail::checked_flat_index(requested.real_element_count(), "real_mesh"));
    static_cast<void>(
        detail::checked_flat_index(requested.spectrum_element_count(), "spectrum"));
}

template <typename Float>
auto MeshStorage<Float>::plan_reserve(
    std::int64_t nx,
    std::int64_t ny,
    std::int64_t nz,
    std::int64_t channels) const -> Reservation
{
    validate_reserve(nx, ny, nz, channels);
    const MeshShape requested(nx, ny, nz, channels, layout_);

    // All-or-nothing: allocate every required replacement locally first.  A
    // failure here - including a failure of the second allocation - leaves the
    // committed allocation, geometry, counters, and identity untouched, so a
    // plan built against the retained allocation stays valid.
    const std::int64_t committed_real = allocation_ ? allocation_->real_elements() : 0;
    const std::int64_t committed_spectrum = allocation_ ? allocation_->spectrum_elements() : 0;
    const bool real_grown = requested.real_element_count() > committed_real;
    const bool spectrum_grown = requested.spectrum_element_count() > committed_spectrum;

    Kokkos::View<Float*> real_replacement;
    if (real_grown) {
        detail::maybe_fail_allocation("the real mesh");
        real_replacement =
            Kokkos::View<Float*>("symmetrix_particle_mesh_real", requested.real_element_count());
    }
    Kokkos::View<Kokkos::complex<Float>*> spectrum_replacement;
    if (spectrum_grown) {
        detail::maybe_fail_allocation("the spectrum mesh");
        spectrum_replacement = Kokkos::View<Kokkos::complex<Float>*>(
            "symmetrix_particle_mesh_spectrum", requested.spectrum_element_count());
    }

    Reservation reservation;
    reservation.shape = requested;
    reservation.grown = real_grown || spectrum_grown;
    reservation.peak_channels = std::max(peak_channels_, requested.batch());
    reservation.serial = allocation_serial_;
    reservation.allocations = allocations_;
    reservation.allocation = allocation_;
    if (reservation.grown) {
        // The token is built before the commit, so an allocation failure of the
        // token itself also leaves the storage unchanged.
        auto candidate = std::make_shared<MeshAllocation<Float>>();
        candidate->generation = (allocation_ ? allocation_->generation : 0) + 1;
        candidate->real = real_grown ? real_replacement : allocation_->real;
        candidate->spectrum = spectrum_grown ? spectrum_replacement : allocation_->spectrum;
        candidate->real_id = real_grown ? ++reservation.serial
                                       : (allocation_ ? allocation_->real_id : 0);
        candidate->spectrum_id = spectrum_grown ? ++reservation.serial
                                               : (allocation_ ? allocation_->spectrum_id : 0);
        reservation.allocations += static_cast<std::uint64_t>(real_grown)
            + static_cast<std::uint64_t>(spectrum_grown);
        reservation.allocation = std::move(candidate);
    }
    return reservation;
}

template <typename Float>
void MeshStorage<Float>::commit(Reservation&& reservation)
{
    // Nothing below allocates or throws, so the storage moves from one
    // consistent state to the next.
    ++reservations_;
    allocation_serial_ = reservation.serial;
    allocations_ = reservation.allocations;
    peak_channels_ = reservation.peak_channels;
    allocation_ = std::move(reservation.allocation);
    shape_ = reservation.shape;
}

template <typename Float>
MeshStorageUsage MeshStorage<Float>::usage() const
{
    const std::int64_t active_real = shape_ ? shape_->real_element_count() : 0;
    const std::int64_t active_spectrum = shape_ ? shape_->spectrum_element_count() : 0;
    const std::int64_t capacity_real = allocation_ ? allocation_->real_elements() : 0;
    const std::int64_t capacity_spectrum = allocation_ ? allocation_->spectrum_elements() : 0;

    MeshStorageUsage usage;
    usage.reservations = reservations_;
    usage.allocations = allocations_;
    usage.allocation_generation = allocation_ ? allocation_->generation : 0;
    usage.channel_bound = channel_bound_;
    usage.active_channels = shape_ ? shape_->batch() : 0;
    usage.peak_channels = peak_channels_;
    usage.buffers.push_back(BufferUsage{
        "real_mesh",
        active_real,
        capacity_real,
        detail::element_bytes(active_real, sizeof(Float)),
        detail::element_bytes(capacity_real, sizeof(Float)),
        allocation_ ? allocation_->real_id : 0});
    usage.buffers.push_back(BufferUsage{
        "spectrum",
        active_spectrum,
        capacity_spectrum,
        detail::element_bytes(active_spectrum, sizeof(Kokkos::complex<Float>)),
        detail::element_bytes(capacity_spectrum, sizeof(Kokkos::complex<Float>)),
        allocation_ ? allocation_->spectrum_id : 0});
    detail::finalize_usage(usage.buffers, usage.active_bytes, usage.capacity_bytes);
    return usage;
}

template <typename Float>
typename MeshStorage<Float>::real_active_view MeshStorage<Float>::real_field() const
{
    if (!shape_ || !allocation_) {
        throw std::runtime_error(
            "Particle-mesh storage holds no reserved mesh; reserve a geometry first.");
    }
    return real_active_view(
        allocation_->real.data(), shape_->batch(), shape_->extent(0), shape_->extent(1),
        shape_->extent(2));
}

template <typename Float>
typename MeshStorage<Float>::spectrum_active_view MeshStorage<Float>::spectrum() const
{
    if (!shape_ || !allocation_) {
        throw std::runtime_error(
            "Particle-mesh storage holds no reserved mesh; reserve a geometry first.");
    }
    return spectrum_active_view(
        allocation_->spectrum.data(), shape_->batch(), shape_->extent(0), shape_->extent(1),
        shape_->spectrum_extent());
}

// ---------------------------------------------------------------------------
// FFT-plan domain
// ---------------------------------------------------------------------------

// Identity of the compiled FFT boundary.  A plan is only ever built for the
// provider, precision, and execution space the library was compiled against;
// there is no runtime substitution.
struct PlanIdentity {
    std::string backend;
    std::string precision;
    std::string provider;

    friend bool operator==(const PlanIdentity&, const PlanIdentity&) = default;
};

template <typename Float>
[[nodiscard]] PlanIdentity compiled_plan_identity();

// The two transforms the forward path needs.  The roles are explicit so that no
// transform can be built or executed without naming its direction, and so that
// the two plans of one mesh are cached, reported, and released separately.
enum class FftPlanRole { forward_r2c, inverse_c2r };

inline constexpr int plan_role_count = 2;

[[nodiscard]] std::string_view plan_role_token(FftPlanRole role);
[[nodiscard]] FftPlanRole plan_role_from_token(std::string_view token);

// Cache key of one FFT-plan domain: the compiled identity plus the mesh, its
// layout, the transform batch, and the direction the plan transforms in.
class FftPlanKey {
public:
    FftPlanKey(
        MeshShape mesh,
        PlanIdentity identity,
        FftPlanRole role = FftPlanRole::forward_r2c);

    [[nodiscard]] const MeshShape& mesh() const noexcept { return mesh_; }
    [[nodiscard]] const PlanIdentity& identity() const noexcept { return identity_; }
    [[nodiscard]] FftPlanRole role() const noexcept { return role_; }
    [[nodiscard]] std::string description() const;

    friend bool operator==(const FftPlanKey&, const FftPlanKey&) = default;

private:
    MeshShape mesh_;
    PlanIdentity identity_;
    FftPlanRole role_ = FftPlanRole::forward_r2c;
};

// What an FFT plan retains: the allocation it captured, and how strongly.
// `owns` is true exactly while a plan holds an allocation.  `allocation_owners`
// counts the owners of the retained token - the storage that created it plus
// every plan built from it - so a caller can observe that a plan keeps the
// buffers alive on its own once the storage is gone.  The view use counts are
// the Kokkos reference counts of the retained managed views: exactly one
// instance of each view exists inside the token, which is why the buffers are
// live rather than merely named.
struct RetainedAllocationReport {
    bool owns = false;
    std::int64_t allocation_owners = 0;
    std::uint64_t generation = 0;
    std::int64_t real_elements = 0;
    std::int64_t spectrum_elements = 0;
    std::uint64_t retained_bytes = 0;
    std::int64_t real_view_use_count = 0;
    std::int64_t spectrum_view_use_count = 0;
};

// Owns one KokkosFFT-created plan and the mesh allocation it was built from.
// The backend plan captures raw buffer addresses while the active views are
// unmanaged, so the plan must own the allocation: it retains the storage's
// `MeshAllocation` token for its whole lifetime, which keeps the buffers
// allocated even if the storage is destroyed first.  The concrete plan type is
// provider specific and stays inside the implementation, so neither this header
// nor the binding translation unit needs an FFT provider include path.
//
// The handle is precision specific because executing it needs the storage's
// typed active views: a plan is only ever run on the mesh it was built for, and
// that check happens in `FftPlanDomain::execute` before the plan sees any
// buffer.
template <typename Float>
class FftPlanHandle {
public:
    using execution_space = Kokkos::DefaultExecutionSpace;
    using real_view_type = typename MeshStorage<Float>::real_active_view;
    using spectrum_view_type = typename MeshStorage<Float>::spectrum_active_view;

    FftPlanHandle() = default;
    virtual ~FftPlanHandle();
    FftPlanHandle(const FftPlanHandle&) = delete;
    FftPlanHandle& operator=(const FftPlanHandle&) = delete;
    FftPlanHandle(FftPlanHandle&&) = delete;
    FftPlanHandle& operator=(FftPlanHandle&&) = delete;

    [[nodiscard]] virtual const FftPlanKey& key() const noexcept = 0;
    [[nodiscard]] virtual std::uint32_t execution_instance_id() const noexcept = 0;
    // Identity of the retained allocation.  A plan is reusable only when the
    // key *and* this token match the requesting storage, which ties every reuse
    // to exactly one allocation generation.
    [[nodiscard]] virtual const void* retained_allocation_token() const noexcept = 0;
    [[nodiscard]] virtual RetainedAllocationReport retained_allocation() const = 0;
    // Runs this plan on the storage's active views, in this plan's direction and
    // with no normalization on either side: the forward transform is the
    // unnormalized real-to-complex transform and the inverse is the unnormalized
    // complex-to-real transform, so a round trip scales by the point count and
    // never by one or by its square root.
    virtual void execute(const real_view_type&, const spectrum_view_type&) const = 0;
};

// Builds the plan for one role and one committed allocation, whose token the
// plan retains.  Defined where KokkosFFT is available; there is no substitute
// implementation.
template <typename Float>
[[nodiscard]] std::unique_ptr<FftPlanHandle<Float>> build_fft_plan(
    const Kokkos::DefaultExecutionSpace& execution_space,
    const FftPlanKey& key,
    std::shared_ptr<const MeshAllocation<Float>> allocation);

struct FftPlanUsage {
    FftPlanRole role = FftPlanRole::forward_r2c;
    bool held = false;
    std::uint64_t builds = 0;
    std::uint64_t reuses = 0;
    std::uint64_t releases = 0;
    std::uint64_t key_rebuilds = 0;
    std::uint64_t identity_rebuilds = 0;
    std::int64_t allocation_generation = -1;
    std::uint32_t execution_instance_id = 0;
    std::string key;
    PlanIdentity identity;
    // Ownership of the captured buffers; `owns` is false when no plan is held.
    RetainedAllocationReport allocation;
};

// One plan slot per role: the forward and inverse transforms of one mesh are
// cached independently, so materializing a potential never tears down the
// forward plan and vice versa.
template <typename Float>
class FftPlanDomain {
public:
    using execution_space = Kokkos::DefaultExecutionSpace;

    struct Outcome {
        bool built = false;
        bool reused = false;
        std::int64_t allocation_generation = -1;
    };

    explicit FftPlanDomain(execution_space instance)
        : execution_space_(std::move(instance))
    {
    }
    ~FftPlanDomain() = default;

    FftPlanDomain(const FftPlanDomain&) = delete;
    FftPlanDomain& operator=(const FftPlanDomain&) = delete;
    FftPlanDomain(FftPlanDomain&&) = delete;
    FftPlanDomain& operator=(FftPlanDomain&&) = delete;

    Outcome acquire(
        const MeshStorage<Float>& storage, FftPlanRole role = FftPlanRole::forward_r2c);
    // Releases every role that holds a plan; the return value reports whether
    // anything was released.
    bool release();
    bool release(FftPlanRole role);
    // Runs the held plan of one role on the storage's active views.  A transform
    // never bypasses this owner: the direction, the normalization, and the mesh
    // the plan captured are all decided here.
    void execute(const MeshStorage<Float>& storage, FftPlanRole role) const;
    [[nodiscard]] FftPlanUsage usage(FftPlanRole role = FftPlanRole::forward_r2c) const;
    [[nodiscard]] std::array<FftPlanUsage, plan_role_count> plan_usage() const;
    [[nodiscard]] std::uint32_t execution_instance_id() const noexcept
    {
        return execution_space_.impl_instance_id();
    }

private:
    struct Slot {
        std::unique_ptr<FftPlanHandle<Float>> plan;
        std::uint64_t builds = 0;
        std::uint64_t reuses = 0;
        std::uint64_t releases = 0;
        std::uint64_t key_rebuilds = 0;
        std::uint64_t identity_rebuilds = 0;
        std::int64_t generation = -1;
    };

    static constexpr std::size_t index_of(FftPlanRole role) noexcept
    {
        return role == FftPlanRole::forward_r2c ? 0U : 1U;
    }

    execution_space execution_space_;
    std::array<Slot, plan_role_count> slots_;
};

template <typename Float>
auto FftPlanDomain<Float>::acquire(const MeshStorage<Float>& storage, FftPlanRole role)
    -> Outcome
{
    const PlanIdentity identity = compiled_plan_identity<Float>();
    if (identity.precision != MeshStorage<Float>::precision_token()) {
        throw std::invalid_argument(
            "Particle-mesh FFT plan domain precision '" + identity.precision
            + "' does not match the storage precision '"
            + std::string(MeshStorage<Float>::precision_token()) + "'.");
    }
    const std::optional<MeshShape> shape = storage.shape();
    if (!shape) {
        throw std::runtime_error(
            "Particle-mesh FFT plans require a reserved mesh; the storage holds none.");
    }
    const std::shared_ptr<const MeshAllocation<Float>> allocation = storage.allocation();
    if (!allocation) {
        throw std::runtime_error(
            "Particle-mesh FFT plans require a committed mesh allocation; the storage "
            "holds none.");
    }
    const FftPlanKey key(*shape, identity, role);
    Slot& slot = slots_[index_of(role)];

    bool same_key = false;
    if (slot.plan) {
        same_key = slot.plan->key() == key;
        // Reuse needs the key *and* the exact retained allocation: the plan
        // captured buffer addresses, so a replacement allocation must never
        // satisfy this comparison.
        if (same_key && slot.plan->retained_allocation_token() == allocation.get()) {
            ++slot.reuses;
            return Outcome{false, true, slot.generation};
        }
    }

    // Build into a local plan and commit only on success: a failed build keeps
    // the retained plan, its buffers, and every counter exactly as they were.
    // The replaced plan is destroyed after the replacement exists, so no window
    // exists in which the mesh buffers have no owner.
    const bool held_before = static_cast<bool>(slot.plan);
    auto replacement = build_fft_plan<Float>(execution_space_, key, allocation);
    slot.plan = std::move(replacement);
    ++slot.builds;
    if (same_key) {
        ++slot.identity_rebuilds;
    } else if (held_before) {
        // A cold build - the first plan of this role, or one after a release -
        // is not attributed to a key change, however many generations of
        // allocations the domain has already seen.
        ++slot.key_rebuilds;
    }
    slot.generation = static_cast<std::int64_t>(allocation->generation);
    return Outcome{true, false, slot.generation};
}

template <typename Float>
bool FftPlanDomain<Float>::release()
{
    bool released = false;
    for (FftPlanRole role : {FftPlanRole::forward_r2c, FftPlanRole::inverse_c2r}) {
        released = release(role) || released;
    }
    return released;
}

template <typename Float>
bool FftPlanDomain<Float>::release(FftPlanRole role)
{
    Slot& slot = slots_[index_of(role)];
    if (!slot.plan) {
        return false;
    }
    slot.plan.reset();
    ++slot.releases;
    return true;
}

template <typename Float>
void FftPlanDomain<Float>::execute(const MeshStorage<Float>& storage, FftPlanRole role) const
{
    const Slot& slot = slots_[index_of(role)];
    if (!slot.plan) {
        throw std::runtime_error(
            "Particle-mesh FFT plan for the " + std::string(plan_role_token(role))
            + " role is not held; acquire it before executing it.");
    }
    const std::optional<MeshShape> shape = storage.shape();
    const std::shared_ptr<const MeshAllocation<Float>> allocation = storage.allocation();
    if (!shape || !allocation) {
        throw std::runtime_error(
            "Particle-mesh FFT execution requires a reserved mesh and a committed "
            "allocation; the storage holds none.");
    }
    if (slot.plan->key().mesh() != *shape) {
        throw std::runtime_error(
            "Particle-mesh FFT plan was built for mesh [" + slot.plan->key().mesh().description()
            + "] but the storage now holds [" + shape->description()
            + "]; acquire the plan for the active mesh first.");
    }
    if (slot.plan->retained_allocation_token() != allocation.get()) {
        throw std::runtime_error(
            "Particle-mesh FFT plan captured a replaced mesh allocation; acquire the "
            "plan for the active allocation first.");
    }
    slot.plan->execute(storage.real_field(), storage.spectrum());
}

template <typename Float>
FftPlanUsage FftPlanDomain<Float>::usage(FftPlanRole role) const
{
    const Slot& slot = slots_[index_of(role)];
    FftPlanUsage usage;
    usage.role = role;
    usage.held = static_cast<bool>(slot.plan);
    usage.builds = slot.builds;
    usage.reuses = slot.reuses;
    usage.releases = slot.releases;
    usage.key_rebuilds = slot.key_rebuilds;
    usage.identity_rebuilds = slot.identity_rebuilds;
    usage.allocation_generation = slot.generation;
    usage.execution_instance_id =
        slot.plan ? slot.plan->execution_instance_id() : execution_space_.impl_instance_id();
    usage.identity = compiled_plan_identity<Float>();
    if (slot.plan) {
        usage.key = slot.plan->key().description();
        usage.allocation = slot.plan->retained_allocation();
    }
    return usage;
}

template <typename Float>
std::array<FftPlanUsage, plan_role_count> FftPlanDomain<Float>::plan_usage() const
{
    return {usage(FftPlanRole::forward_r2c), usage(FftPlanRole::inverse_c2r)};
}

// Both precisions are instantiated once in the particle-mesh library.
extern template class MeshStorage<float>;
extern template class MeshStorage<double>;
extern template class FftPlanDomain<float>;
extern template class FftPlanDomain<double>;

}  // namespace symmetrix::particle_mesh

#endif  // SYMMETRIX_PARTICLE_MESH_STORAGE_HPP
