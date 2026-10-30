"""Native FourierD3 storage, cache keys, and bounded rank-chunk workspaces.

Task 7 of the reciprocal-space plan fixes the private native infrastructure that
the forward and reverse kernels consume: checked 64-bit mesh geometry, grow-only
mesh and kernel workspaces with active/capacity byte accounting, an FFT
plan-owner seam keyed by backend, precision, provider, mesh, layout, and batch,
and a reciprocal-kernel domain keyed by cell, damping, species, and spline
order.

Nothing here evaluates a correction, and no physical kernel value is produced:
the BJ kernel tables belong to Task 9.  These tests drive the private storage
seam of the installed native extension and skip when that extension was built
without a KokkosFFT FFT provider.
"""

from __future__ import annotations

import gc

import pytest

import symmetrix

# The storage seam is private native infrastructure: it is reached through the
# loaded backend module, never through a package export.
_SEAM_SYMBOLS = (
    "_fourier_d3_plan_identity",
    "_fourier_d3_arm_allocation_fault",
    "_fourier_d3_clear_allocation_fault",
    "_fourier_d3_allocation_fault_armed",
    "_FourierD3MeshShape",
    "_FourierD3MeshStorage",
    "_FourierD3FftPlanKey",
    "_FourierD3FftPlanDomain",
    "_FourierD3KernelKey",
    "_FourierD3KernelDomain",
)

_FLOAT64_BYTES = 8
_COMPLEX64_BYTES = 16


def _native():
    module = symmetrix.load_backend()
    if not module._kokkos_is_initialized():
        module._init_kokkos()
    return module


@pytest.fixture(scope="module")
def native():
    module = _native()
    capability = getattr(module, "_fourier_d3_capability", None)
    if not callable(capability):
        pytest.skip(
            "the installed native extension predates the FourierD3 capability "
            "report; rebuild the backend to exercise the storage contract"
        )
    report = capability()
    if not report["available"]:
        pytest.skip(
            "this build has no FourierD3 FFT provider "
            f"({report['reason'] or 'unreported'}); the storage seam is not "
            "compiled and nothing is emulated in its place"
        )
    # A build that reports an available provider must carry the storage seam;
    # nothing stands in for it, so a missing seam is a build defect rather than
    # a skip.
    assert hasattr(module, "_FourierD3MeshStorage"), (
        "the build reports an available FourierD3 FFT provider, so the native "
        "storage seam must be present; no host fallback exists"
    )
    return module


@pytest.fixture(autouse=True)
def _disarm_allocation_fault(native):
    # The fault seam is one-shot state inside the extension: never let a test
    # leave it armed for the next one.
    native._fourier_d3_clear_allocation_fault()
    yield
    native._fourier_d3_clear_allocation_fault()


def _shape(native, nx=8, ny=8, nz=8, batch=2, layout="layout_right"):
    return native._FourierD3MeshShape(nx, ny, nz, batch, layout)


def _storage(native, precision="float64", channel_bound=8, layout="layout_right"):
    return native._FourierD3MeshStorage(precision, layout, channel_bound)


def _usage_buffers(usage):
    return {buffer["name"]: buffer for buffer in usage["buffers"]}


# --------------------------------------------------------------------------
# checked 64-bit mesh geometry
# --------------------------------------------------------------------------


def test_mesh_shape_reports_checked_geometry_and_strides(native):
    shape = _shape(native, 8, 8, 8, 2)

    assert tuple(shape.extents) == (8, 8, 8)
    assert shape.batch == 2
    assert shape.layout == "layout_right"
    assert shape.point_count == 512
    assert shape.real_element_count == 1024
    assert shape.spectrum_extent == 5
    assert shape.spectrum_point_count == 320
    assert shape.spectrum_element_count == 640
    assert tuple(shape.strides) == (64, 8, 1)
    assert tuple(shape.spectrum_strides) == (40, 5, 1)
    assert shape.offset(0, 0, 0) == 0
    assert shape.offset(7, 7, 7) == 511
    assert shape.spectrum_offset(7, 7, 4) == 319


def test_mesh_shape_rejects_out_of_range_indices(native):
    shape = _shape(native)

    with pytest.raises(IndexError):
        shape.offset(8, 0, 0)
    with pytest.raises(IndexError):
        shape.offset(0, -1, 0)
    with pytest.raises(IndexError):
        shape.spectrum_offset(0, 0, 5)


def test_mesh_shape_keeps_indices_64_bit_safe(native):
    # 70000 x 70000 x 2 exceeds 2**31 points, so a 32-bit stride product or
    # index would wrap silently.
    shape = _shape(native, 70000, 70000, 2, 1, "layout_right")

    assert shape.point_count == 9_800_000_000
    assert tuple(shape.strides) == (140_000, 2, 1)
    assert shape.offset(69999, 69999, 1) == 9_799_999_999


@pytest.mark.parametrize(
    ("extents", "batch"),
    [
        ((1 << 32, 1 << 32, 2), 1),
        ((1 << 21, 1 << 21, 1 << 21), 1),
        ((1 << 21, 1 << 21, 2), 1 << 21),
    ],
)
def test_mesh_shape_rejects_overflow_instead_of_wrapping(native, extents, batch):
    with pytest.raises(OverflowError):
        _shape(native, *extents, batch)


@pytest.mark.parametrize(
    ("extents", "batch", "layout"),
    [
        ((1, 8, 8), 1, "layout_right"),
        ((8, 0, 8), 1, "layout_right"),
        ((8, 8, 8), 0, "layout_right"),
        ((8, 8, 8), -1, "layout_right"),
        ((8, 8, 8), 1, "layout_diagonal"),
    ],
)
def test_mesh_shape_rejects_impossible_geometry(native, extents, batch, layout):
    with pytest.raises(ValueError):
        _shape(native, *extents, batch, layout)


def test_mesh_shape_equality_covers_layout_batch_and_extents(native):
    reference = _shape(native, 8, 8, 8, 2)

    assert reference == _shape(native, 8, 8, 8, 2)
    assert reference != _shape(native, 8, 8, 8, 3)
    assert reference != _shape(native, 8, 8, 6, 2)
    assert reference != _shape(native, 8, 8, 8, 2, "layout_left")
    assert "8x8x8" in reference.description


# --------------------------------------------------------------------------
# grow-only mesh storage and its byte accounting
# --------------------------------------------------------------------------


def test_mesh_storage_is_grow_only_across_smaller_geometries(native):
    storage = _storage(native, "float64", channel_bound=8)

    first = storage.reserve(8, 8, 8, 4)
    assert first["grown"] is True
    grown = storage.usage()
    assert grown["allocations"] == 2
    assert grown["reservations"] == 1
    assert grown["active_channels"] == 4
    assert grown["peak_channels"] == 4

    # A smaller rank chunk reuses the retained allocation.
    second = storage.reserve(8, 8, 8, 2)
    assert second["grown"] is False
    reused = storage.usage()
    assert reused["allocations"] == 2
    assert reused["reservations"] == 2
    assert reused["active_channels"] == 2
    assert reused["peak_channels"] == 4
    assert reused["capacity_bytes"] == grown["capacity_bytes"]
    assert reused["active_bytes"] < grown["active_bytes"]
    assert reused["allocation_generation"] == grown["allocation_generation"]

    # A larger geometry grows the retained allocation once.
    third = storage.reserve(16, 16, 16, 2)
    assert third["grown"] is True
    larger = storage.usage()
    assert larger["allocations"] == 4
    assert larger["capacity_bytes"] > grown["capacity_bytes"]
    assert larger["allocation_generation"] == grown["allocation_generation"] + 1


def test_mesh_storage_reports_active_and_capacity_bytes_without_aliases(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 4)

    usage = storage.usage()
    buffers = _usage_buffers(usage)

    assert sorted(buffers) == ["real_mesh", "spectrum"]
    real = buffers["real_mesh"]
    spectrum = buffers["spectrum"]
    assert real["active_elements"] == 8 * 8 * 8 * 4
    assert real["capacity_elements"] == real["active_elements"]
    assert spectrum["active_elements"] == 8 * 8 * 5 * 4
    assert spectrum["capacity_elements"] == spectrum["active_elements"]
    assert real["active_bytes"] == real["active_elements"] * _FLOAT64_BYTES
    assert spectrum["active_bytes"] == spectrum["active_elements"] * _COMPLEX64_BYTES
    assert real["allocation_id"] != spectrum["allocation_id"]
    assert usage["active_bytes"] == real["active_bytes"] + spectrum["active_bytes"]
    assert usage["capacity_bytes"] == (
        real["capacity_bytes"] + spectrum["capacity_bytes"]
    )
    assert usage["capacity_bytes"] == usage["active_bytes"]

    # A rank chunk uses a prefix of the retained allocation; the aliased active
    # view is never counted as a second buffer.
    storage.reserve(8, 8, 8, 1)
    smaller = storage.usage()
    assert len(smaller["buffers"]) == 2
    assert smaller["capacity_bytes"] == usage["capacity_bytes"]
    assert smaller["active_bytes"] == usage["active_bytes"] // 4


def test_mesh_storage_bounds_rank_chunks_by_the_configured_chunk(native):
    chunk = 4
    storage = _storage(native, "float64", channel_bound=chunk)

    for _ in range(3):
        storage.reserve(8, 8, 8, chunk)

    usage = storage.usage()
    assert usage["active_channels"] == chunk
    assert usage["peak_channels"] == chunk
    assert usage["allocations"] == 2
    assert usage["capacity_bytes"] == (
        8 * 8 * 8 * chunk * _FLOAT64_BYTES + 8 * 8 * 5 * chunk * _COMPLEX64_BYTES
    )

    with pytest.raises(ValueError, match="channel bound"):
        storage.reserve(8, 8, 8, chunk + 1)

    rejected = storage.usage()
    assert rejected["capacity_bytes"] == usage["capacity_bytes"]
    assert rejected["allocations"] == usage["allocations"]


def test_mesh_storage_rejects_overflow_before_allocating(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    baseline = storage.usage()

    with pytest.raises(OverflowError):
        storage.reserve(1 << 32, 1 << 32, 2, 1)

    rejected = storage.usage()
    assert rejected["allocations"] == baseline["allocations"]
    assert rejected["capacity_bytes"] == baseline["capacity_bytes"]


def test_mesh_storage_refuses_a_layout_it_was_not_compiled_for(native):
    with pytest.raises(ValueError, match="layout_right"):
        _storage(native, "float64", 4, layout="layout_left")


def test_mesh_storage_requires_a_known_precision(native):
    with pytest.raises(ValueError, match="precision"):
        _storage(native, "float16", 4)


def test_mesh_storage_exposes_the_active_geometry(native):
    storage = _storage(native, "float64", channel_bound=8)
    assert storage.shape is None

    storage.reserve(8, 8, 8, 2)
    assert storage.shape == _shape(native, 8, 8, 8, 2)

    storage.reserve(4, 6, 8, 1)
    assert storage.shape == _shape(native, 4, 6, 8, 1)


# --------------------------------------------------------------------------
# FFT-plan domain keys and reuse
# --------------------------------------------------------------------------


def test_fft_plan_domain_reuses_a_plan_for_an_unchanged_mesh(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")

    first = domain.acquire(storage)
    assert first["built"] is True
    assert first["reused"] is False

    second = domain.acquire(storage)
    assert second["built"] is False
    assert second["reused"] is True

    usage = domain.usage()
    assert usage["held"] is True
    assert usage["builds"] == 1
    assert usage["reuses"] == 1
    assert usage["releases"] == 0
    assert usage["identity_rebuilds"] == 0


def test_fft_plan_domain_captures_its_execution_instance_for_both_roles(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")

    domain.acquire(storage, "forward_r2c")
    domain.acquire(storage, "inverse_c2r")

    usage = domain.usage()
    assert usage["execution_instance_id"] > 0
    assert {plan["role"] for plan in usage["plans"]} == {
        "forward_r2c",
        "inverse_c2r",
    }
    for plan in usage["plans"]:
        assert plan["execution_instance_id"] == usage["execution_instance_id"]


def test_fft_plan_domain_rebuilds_on_a_mesh_batch_or_layout_change(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")
    domain.acquire(storage)

    storage.reserve(8, 8, 8, 1)
    assert domain.acquire(storage)["built"] is True

    storage.reserve(16, 16, 16, 1)
    assert domain.acquire(storage)["built"] is True

    usage = domain.usage()
    assert usage["builds"] == 3
    assert usage["reuses"] == 0
    assert usage["key_rebuilds"] == 2
    assert usage["identity_rebuilds"] == 0


def test_fft_plan_domain_rebuilds_when_the_retained_buffers_move(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 4)
    domain = native._FourierD3FftPlanDomain("float64")
    first = domain.acquire(storage)
    assert first["built"] is True

    # Grow the retained allocation, then return to the original geometry: the
    # key repeats, but the backend plan captures the buffer addresses, so a
    # plan built against the old allocation must not be reused.
    storage.reserve(32, 32, 32, 4)
    storage.reserve(8, 8, 8, 4)
    again = domain.acquire(storage)

    assert again["built"] is True
    assert again["reused"] is False
    usage = domain.usage()
    assert usage["builds"] == 2
    # The mesh key repeats, so the rebuild is attributed to the moved buffers
    # rather than to a changed key.
    assert usage["key_rebuilds"] == 0
    assert usage["identity_rebuilds"] == 1
    assert usage["allocation_generation"] > first["allocation_generation"]


def test_fft_plan_domain_release_tears_the_plan_down(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")
    domain.acquire(storage)

    assert domain.release()["released"] is True
    assert domain.usage()["held"] is False
    assert domain.release()["released"] is False
    # ``releases`` counts explicit releases only: destroying a held plan through
    # the domain's own teardown is not a release a caller asked for.
    assert domain.usage()["releases"] == 1

    domain.acquire(storage)
    usage = domain.usage()
    assert usage["builds"] == 2
    # A plan rebuilt after an explicit release is a cold build, not a key change,
    # however many allocation generations the domain has already seen.
    assert usage["key_rebuilds"] == 0
    assert usage["identity_rebuilds"] == 0


def test_fft_plan_domain_attributes_a_cold_rebuild_after_release_to_no_key(native):
    """A release followed by a different key is still a cold build, not a rebuild."""

    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")
    domain.acquire(storage)
    domain.release()
    assert domain.usage()["releases"] == 1

    # The mesh key differs from the released plan's key, but no plan was held when
    # the replacement was built, so the rebuild reason is "nothing was held".
    storage.reserve(16, 16, 16, 2)
    assert domain.acquire(storage)["built"] is True

    usage = domain.usage()
    assert usage["builds"] == 2
    assert usage["key_rebuilds"] == 0
    assert usage["identity_rebuilds"] == 0

    # A held plan replaced by a different key is attributed to the key.
    storage.reserve(32, 32, 32, 2)
    assert domain.acquire(storage)["built"] is True
    assert domain.usage()["key_rebuilds"] == 1


def test_fft_plan_domain_caches_one_plan_per_role(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 1)
    domain = native._FourierD3FftPlanDomain("float64")

    forward = domain.acquire(storage, "forward_r2c")
    assert (forward["built"], forward["role"]) == (True, "forward_r2c")
    inverse = domain.acquire(storage, "inverse_c2r")
    assert (inverse["built"], inverse["role"]) == (True, "inverse_c2r")

    plans = {entry["role"]: entry for entry in domain.usage()["plans"]}
    assert set(plans) == {"forward_r2c", "inverse_c2r"}
    assert plans["forward_r2c"]["builds"] == 1
    assert plans["inverse_c2r"]["builds"] == 1
    # The two roles are not interchangeable: a plan transforms one way only.
    assert plans["forward_r2c"]["key"] != plans["inverse_c2r"]["key"]

    # Re-acquiring one role reuses that role alone.
    assert domain.acquire(storage, "inverse_c2r")["reused"] is True
    plans = {entry["role"]: entry for entry in domain.usage()["plans"]}
    assert plans["forward_r2c"]["reuses"] == 0
    assert plans["inverse_c2r"]["reuses"] == 1

    # A per-role release leaves the other role - and its buffers - held.
    assert domain.release("inverse_c2r")["released"] is True
    plans = {entry["role"]: entry for entry in domain.usage()["plans"]}
    assert plans["inverse_c2r"]["held"] is False
    assert plans["inverse_c2r"]["releases"] == 1
    assert plans["forward_r2c"]["held"] is True
    assert plans["forward_r2c"]["releases"] == 0


def test_fft_plan_domain_executes_only_a_held_role(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 1)
    domain = native._FourierD3FftPlanDomain("float64")

    with pytest.raises(RuntimeError, match="not held"):
        domain.execute(storage, "forward_r2c")

    domain.acquire(storage, "forward_r2c")
    with pytest.raises(RuntimeError, match="not held"):
        domain.execute(storage, "inverse_c2r")

    executed = domain.execute(storage, "forward_r2c")
    assert executed["role"] == "forward_r2c"
    assert executed["held"] is True
    # Executing the forward role never materializes the inverse role; the
    # unnormalized round trip itself is checked in the forward-path suite.
    plans = {entry["role"]: entry for entry in domain.usage()["plans"]}
    assert plans["inverse_c2r"]["builds"] == 0
    assert plans["inverse_c2r"]["held"] is False


def test_fft_plan_key_separates_the_two_roles(native):
    identity = native._fourier_d3_plan_identity("float64")
    shape = _shape(native, 8, 8, 8, 2)
    forward = native._FourierD3FftPlanKey(
        shape, identity["backend"], identity["precision"], identity["provider"]
    )
    inverse = native._FourierD3FftPlanKey(
        shape,
        identity["backend"],
        identity["precision"],
        identity["provider"],
        "inverse_c2r",
    )

    assert forward != inverse
    assert "forward_r2c" in forward.description
    assert "inverse_c2r" in inverse.description
    assert forward == native._FourierD3FftPlanKey(
        _shape(native, 8, 8, 8, 2),
        identity["backend"],
        identity["precision"],
        identity["provider"],
        "forward_r2c",
    )
    with pytest.raises(ValueError, match="role"):
        native._FourierD3FftPlanKey(
            shape,
            identity["backend"],
            identity["precision"],
            identity["provider"],
            "sideways",
        )


def test_fft_plan_domain_cold_rebuild_after_release_is_not_a_key_change(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")
    domain.acquire(storage)

    storage.reserve(16, 16, 16, 2)
    domain.acquire(storage)
    assert domain.usage()["key_rebuilds"] == 1

    # The storage grows only, so returning to the smaller geometry keeps the
    # retained allocation and is a real key change.
    storage.reserve(8, 8, 8, 2)
    domain.acquire(storage)
    usage = domain.usage()
    assert usage["key_rebuilds"] == 2
    assert usage["builds"] == 3

    # Releasing the plan and acquiring it again is a cold build of a key this
    # domain has already seen, not a key change.
    domain.release()
    assert domain.acquire(storage)["built"] is True
    after = domain.usage()
    assert after["builds"] == 4
    assert after["key_rebuilds"] == 2


def test_fft_plan_domain_rejects_a_storage_without_a_mesh(native):
    storage = _storage(native, "float64", channel_bound=8)
    domain = native._FourierD3FftPlanDomain("float64")

    with pytest.raises(RuntimeError, match="reserved mesh"):
        domain.acquire(storage)


def test_fft_plan_domain_rejects_a_mismatched_precision(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)

    with pytest.raises(ValueError, match="precision"):
        native._FourierD3FftPlanDomain("float32").acquire(storage)


def test_fft_plan_key_covers_backend_precision_provider_and_mesh(native):
    identity = native._fourier_d3_plan_identity("float64")
    shape = _shape(native, 8, 8, 8, 2)
    reference = native._FourierD3FftPlanKey(
        shape, identity["backend"], identity["precision"], identity["provider"]
    )

    assert reference == native._FourierD3FftPlanKey(
        _shape(native, 8, 8, 8, 2),
        identity["backend"],
        identity["precision"],
        identity["provider"],
    )
    assert reference != native._FourierD3FftPlanKey(
        shape, "SomeFutureBackend", identity["precision"], identity["provider"]
    )
    assert reference != native._FourierD3FftPlanKey(
        shape, identity["backend"], "float32", identity["provider"]
    )
    assert reference != native._FourierD3FftPlanKey(
        shape, identity["backend"], identity["precision"], "some-future-fft"
    )
    assert reference != native._FourierD3FftPlanKey(
        _shape(native, 8, 8, 8, 3),
        identity["backend"],
        identity["precision"],
        identity["provider"],
    )
    assert reference != native._FourierD3FftPlanKey(
        _shape(native, 8, 8, 6, 2),
        identity["backend"],
        identity["precision"],
        identity["provider"],
    )
    assert reference != native._FourierD3FftPlanKey(
        _shape(native, 8, 8, 8, 2, "layout_left"),
        identity["backend"],
        identity["precision"],
        identity["provider"],
    )


def test_fft_plan_domain_keys_the_compiled_identity_and_active_mesh(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")
    domain.acquire(storage)

    identity = native._fourier_d3_plan_identity("float64")
    expected = native._FourierD3FftPlanKey(
        storage.shape,
        identity["backend"],
        identity["precision"],
        identity["provider"],
    )

    usage = domain.usage()
    assert usage["key"] == expected.description
    assert usage["identity"]["provider"] == identity["provider"]
    assert usage["identity"]["precision"] == "float64"
    assert usage["identity"]["backend"] == identity["backend"]


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_plan_identity_matches_the_build_capability(native, precision):
    identity = native._fourier_d3_plan_identity(precision)
    capability = native._fourier_d3_capability()

    assert identity["provider"] == capability["provider"]
    assert identity["precision"] == precision
    assert identity["backend"] == native._kokkos_default_execution_space()


# --------------------------------------------------------------------------
# reciprocal-kernel domain keys and workspaces
# --------------------------------------------------------------------------

_CELL = (4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0)
_DAMPING = (0.4289, 4.4407, 1.0, 0.7875)
_SPECIES = (1, 6, 8)


def _kernel_domain(native, precision="float64"):
    return native._FourierD3KernelDomain(precision)


def test_kernel_domain_covers_cell_damping_species_and_spline_order(native):
    identity = native._fourier_d3_plan_identity("float64")
    reference = native._FourierD3KernelKey(
        _CELL, _DAMPING, _SPECIES, 4, identity["backend"], "float64"
    )

    assert reference == native._FourierD3KernelKey(
        tuple(_CELL),
        tuple(_DAMPING),
        list(_SPECIES),
        4,
        identity["backend"],
        "float64",
    )
    strained = list(_CELL)
    strained[0] = 4.000001
    assert reference != native._FourierD3KernelKey(
        tuple(strained), _DAMPING, _SPECIES, 4, identity["backend"], "float64"
    )
    for index in range(4):
        perturbed = list(_DAMPING)
        perturbed[index] = perturbed[index] * 1.000001
        assert reference != native._FourierD3KernelKey(
            _CELL, tuple(perturbed), _SPECIES, 4, identity["backend"], "float64"
        )
    assert reference != native._FourierD3KernelKey(
        _CELL, _DAMPING, (1, 6), 4, identity["backend"], "float64"
    )
    assert reference != native._FourierD3KernelKey(
        _CELL, _DAMPING, _SPECIES, 5, identity["backend"], "float64"
    )
    assert reference != native._FourierD3KernelKey(
        _CELL, _DAMPING, _SPECIES, 4, "SomeFutureBackend", "float64"
    )
    assert reference != native._FourierD3KernelKey(
        _CELL, _DAMPING, _SPECIES, 4, identity["backend"], "float32"
    )


def test_kernel_domain_reuses_an_unchanged_key_and_rebuilds_on_change(native):
    shape = _shape(native, 8, 8, 8, 2)
    domain = _kernel_domain(native)

    domain.reserve(_CELL, _DAMPING, _SPECIES, 4, shape)
    assert domain.usage()["key_hits"] == 0
    assert domain.usage()["key_misses"] == 1

    domain.reserve(tuple(_CELL), tuple(_DAMPING), list(_SPECIES), 4, shape)
    usage = domain.usage()
    assert usage["key_hits"] == 1
    assert usage["key_misses"] == 1
    assert usage["allocations"] == 2
    assert (
        usage["key"]
        == native._FourierD3KernelKey(
            _CELL, _DAMPING, _SPECIES, 4, usage["identity"]["backend"], "float64"
        ).description
    )

    strained = list(_CELL)
    strained[1] = 0.25
    domain.reserve(tuple(strained), _DAMPING, _SPECIES, 4, shape)
    usage = domain.usage()
    assert usage["key_hits"] == 1
    assert usage["key_misses"] == 2
    assert usage["key_rebuilds"] == 1

    # A different spline order invalidates the kernel domain without touching
    # the retained tables.
    domain.reserve(tuple(strained), _DAMPING, _SPECIES, 6, shape)
    changed = domain.usage()
    assert changed["key_misses"] == 3
    assert changed["capacity_bytes"] == usage["capacity_bytes"]
    assert changed["allocations"] == usage["allocations"]


def test_kernel_domain_grows_only_across_smaller_meshes_and_species(native):
    domain = _kernel_domain(native)

    domain.reserve(_CELL, _DAMPING, (1, 6, 8), 4, _shape(native, 16, 16, 16, 2))
    large = domain.usage()
    buffers = _usage_buffers(large)
    assert sorted(buffers) == ["pair_kernel", "pair_kernel_derivative"]
    # 3 species -> 6 unordered pairs, 16x16x9 spectrum bins.
    assert buffers["pair_kernel"]["active_elements"] == 6 * 16 * 16 * 9
    assert buffers["pair_kernel"]["active_bytes"] == (
        buffers["pair_kernel"]["active_elements"] * _FLOAT64_BYTES
    )
    assert (
        buffers["pair_kernel"]["allocation_id"]
        != (buffers["pair_kernel_derivative"]["allocation_id"])
    )
    assert large["active_bytes"] == large["capacity_bytes"]

    domain.reserve(_CELL, _DAMPING, (1, 6), 4, _shape(native, 8, 8, 8, 2))
    smaller = domain.usage()
    small_buffers = _usage_buffers(smaller)
    assert small_buffers["pair_kernel"]["active_elements"] == 3 * 8 * 8 * 5
    assert smaller["capacity_bytes"] == large["capacity_bytes"]
    assert smaller["active_bytes"] < large["active_bytes"]
    assert smaller["reservations"] == 2


@pytest.mark.parametrize(
    ("cell", "damping", "species", "order"),
    [
        ((float("nan"), 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0), _DAMPING, (1,), 4),
        (_CELL, (float("inf"), 4.4407, 1.0, 0.7875), (1,), 4),
        (_CELL, _DAMPING, (6, 1), 4),
        (_CELL, _DAMPING, (1, 1), 4),
        (_CELL, _DAMPING, (0, 1), 4),
        (_CELL, _DAMPING, (-1, 1), 4),
        (_CELL, _DAMPING, (1,), 1),
        (_CELL, _DAMPING, (1,), 7),
    ],
)
def test_kernel_domain_rejects_invalid_inputs(native, cell, damping, species, order):
    domain = _kernel_domain(native)

    with pytest.raises(ValueError):
        domain.reserve(cell, damping, species, order, _shape(native, 8, 8, 8, 1))


def test_kernel_domain_rejects_overflow_in_its_tables(native):
    domain = _kernel_domain(native)

    # 6 unordered species pairs over a 2x2x(2**60) mesh: the table product
    # exceeds a signed 64-bit element count.
    with pytest.raises(OverflowError):
        domain.reserve(_CELL, _DAMPING, (1, 6, 8), 4, _shape(native, 2, 2, 1 << 60, 1))

    assert domain.usage()["allocations"] == 0


def test_kernel_domain_keys_no_graph_or_decomposition_state(native):
    domain = _kernel_domain(native)
    domain.reserve(_CELL, _DAMPING, _SPECIES, 4, _shape(native, 8, 8, 8, 1))

    key = domain.usage()["key"]
    fields = {field.split("=", 1)[0] for field in key.split(" ")}

    assert fields == {
        "backend",
        "precision",
        "cell",
        "damping",
        "species",
        "spline_order",
    }


def test_plan_and_kernel_domains_invalidate_independently(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    plans = native._FourierD3FftPlanDomain("float64")
    kernels = _kernel_domain(native)
    plans.acquire(storage)
    kernels.reserve(_CELL, _DAMPING, _SPECIES, 4, storage.shape)

    plan_key = plans.usage()["key"]
    kernel_key = kernels.usage()["key"]
    plan_builds = plans.usage()["builds"]

    # A cell change is the kernel domain's business only.
    strained = list(_CELL)
    strained[2] = 0.125
    kernels.reserve(tuple(strained), _DAMPING, _SPECIES, 4, storage.shape)
    assert kernels.usage()["key"] != kernel_key
    assert plans.usage()["key"] == plan_key
    assert plans.usage()["builds"] == plan_builds

    # A mesh change is the FFT-plan domain's business only.
    kernel_before = kernels.usage()
    storage.reserve(16, 16, 16, 2)
    assert plans.acquire(storage)["built"] is True
    assert plans.usage()["key"] != plan_key
    kernels.reserve(tuple(strained), _DAMPING, _SPECIES, 4, storage.shape)
    assert kernels.usage()["key"] == kernel_before["key"]
    assert kernels.usage()["key_hits"] == kernel_before["key_hits"] + 1
    assert kernels.usage()["key_misses"] == kernel_before["key_misses"]


# --------------------------------------------------------------------------
# ownership, destruction ordering, and privacy
# --------------------------------------------------------------------------


def test_storage_and_plan_objects_are_released_before_kokkos_finalization(native):
    # The baseline must be read after collecting garbage: an earlier failing
    # test can leave its objects reachable until the next collection.
    gc.collect()
    baseline = native._kokkos_live_object_count()

    storage = _storage(native, "float64", channel_bound=8)
    assert native._kokkos_live_object_count() == baseline + 1

    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")
    domain.acquire(storage)
    assert native._kokkos_live_object_count() == baseline + 2

    # Kokkos must not be finalized while a plan or its buffers are alive.
    with pytest.raises(RuntimeError, match="finalize"):
        native._finalize_kokkos()

    del domain
    gc.collect()
    assert native._kokkos_live_object_count() == baseline + 1

    del storage
    gc.collect()
    assert native._kokkos_live_object_count() == baseline


def test_plan_domain_survives_the_storage_it_planned_against(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")
    domain.acquire(storage)
    retained_bytes = domain.usage()["allocation"]["retained_bytes"]

    del storage
    gc.collect()

    # Survival means ownership, not just a held flag: the plan keeps the
    # allocation it captured while its storage is gone.
    assert domain.usage()["held"] is True
    assert domain.usage()["allocation"]["retained_bytes"] == retained_bytes
    assert domain.release()["released"] is True
    assert domain.usage()["allocation"] is None

    # A fresh storage of the same geometry is a different allocation, so the
    # released domain rebuilds instead of reusing the dead plan.
    replacement = _storage(native, "float64", channel_bound=8)
    replacement.reserve(8, 8, 8, 2)
    assert domain.acquire(replacement)["built"] is True


@pytest.mark.parametrize("fail_at", [0, 1])
def test_mesh_storage_failed_reserve_commits_nothing(native, fail_at):
    storage = _storage(native, "float64", channel_bound=8)

    # fail_at=0 fails the real mesh, fail_at=1 lets it succeed and fails the
    # spectrum mesh: neither ordering may commit anything, so the storage must
    # report a virgin state.
    native._fourier_d3_arm_allocation_fault(fail_at)
    with pytest.raises(RuntimeError, match="injected allocation fault"):
        storage.reserve(8, 8, 8, 2)
    assert native._fourier_d3_allocation_fault_armed() is False

    assert storage.shape is None
    usage = storage.usage()
    assert usage["reservations"] == 0
    assert usage["allocations"] == 0
    assert usage["allocation_generation"] == 0
    assert usage["active_channels"] == 0
    assert usage["peak_channels"] == 0
    assert usage["active_bytes"] == 0
    assert usage["capacity_bytes"] == 0
    for buffer in usage["buffers"]:
        assert buffer["active_elements"] == 0
        assert buffer["capacity_elements"] == 0
        assert buffer["allocation_id"] == 0

    # A later reserve is unaffected by the failed attempt.
    outcome = storage.reserve(8, 8, 8, 2)
    assert outcome["grown"] is True
    assert outcome["allocation_generation"] == 1
    assert storage.usage()["allocations"] == 2


def test_mesh_storage_failed_growth_keeps_the_retained_allocation_and_plan(native):
    storage = _storage(native, "float64", channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain("float64")
    assert domain.acquire(storage)["built"] is True

    committed = storage.usage()
    committed_ids = _usage_buffers(committed)
    plan_usage = domain.usage()

    # The growth replaces both buffers; the second allocation fails.  A
    # partially committed reserve would free the real mesh while the retained
    # plan still captures its address.
    native._fourier_d3_arm_allocation_fault(1)
    with pytest.raises(RuntimeError, match="injected allocation fault"):
        storage.reserve(32, 32, 32, 4)

    after = storage.usage()
    assert after["allocation_generation"] == committed["allocation_generation"]
    assert after["reservations"] == committed["reservations"]
    assert after["allocations"] == committed["allocations"]
    assert after["active_bytes"] == committed["active_bytes"]
    assert after["capacity_bytes"] == committed["capacity_bytes"]
    assert after["active_channels"] == committed["active_channels"]
    # The description names the layout and the batch, but its exact spelling is not
    # part of the contract: the geometry itself is compared structurally.
    description = storage.shape.description
    assert "layout_right" in description
    assert "batch" in description
    assert storage.shape == _shape(native, 8, 8, 8, 2)
    for buffer in after["buffers"]:
        assert buffer == committed_ids[buffer["name"]]

    # The plan still aliases live buffers, so it must be reused, not rebuilt.
    again = domain.acquire(storage)
    assert again["built"] is False
    assert again["reused"] is True
    assert again["allocation_generation"] == committed["allocation_generation"]
    assert domain.usage()["reuses"] == plan_usage["reuses"] + 1
    assert domain.usage()["identity_rebuilds"] == plan_usage["identity_rebuilds"]


@pytest.mark.parametrize("fail_at", [0, 1])
def test_kernel_domain_failed_allocation_leaves_its_state_unchanged(native, fail_at):
    domain = _kernel_domain(native)
    domain.reserve(_CELL, _DAMPING, _SPECIES, 4, _shape(native, 8, 8, 8, 1))
    committed = domain.usage()
    committed_ids = _usage_buffers(committed)

    # A larger mesh needs both tables replaced; fail_at=0 fails the pair-kernel
    # table and fail_at=1 lets it succeed and fails the derivative table.
    native._fourier_d3_arm_allocation_fault(fail_at)
    with pytest.raises(RuntimeError, match="injected allocation fault"):
        domain.reserve(_CELL, _DAMPING, _SPECIES, 5, _shape(native, 32, 32, 32, 1))

    after = domain.usage()
    assert after["key"] == committed["key"]
    assert after["has_key"] is True
    assert after["reservations"] == committed["reservations"]
    assert after["allocations"] == committed["allocations"]
    assert after["key_hits"] == committed["key_hits"]
    assert after["key_misses"] == committed["key_misses"]
    assert after["key_rebuilds"] == committed["key_rebuilds"]
    assert after["active_bytes"] == committed["active_bytes"]
    assert after["capacity_bytes"] == committed["capacity_bytes"]
    for buffer in after["buffers"]:
        assert buffer == committed_ids[buffer["name"]]
    # The two tables are still a matched pair, never one size each.
    assert (
        after["buffers"][0]["capacity_elements"]
        == after["buffers"][1]["capacity_elements"]
    )
    assert after["buffers"][0]["allocation_id"] != after["buffers"][1]["allocation_id"]

    # The retained tables are still the committed ones and still usable.
    outcome = domain.reserve(
        _CELL, _DAMPING, _SPECIES, 5, _shape(native, 32, 32, 32, 1)
    )
    assert outcome == {"key_reused": False, "buffers_grown": True}
    grown = domain.usage()
    assert grown["capacity_bytes"] > committed["capacity_bytes"]
    assert grown["allocations"] == committed["allocations"] + 2
    assert grown["reservations"] == committed["reservations"] + 1


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_fft_plan_retains_the_storage_allocation_it_planned_against(native, precision):
    gc.collect()
    baseline = native._kokkos_live_object_count()

    storage = _storage(native, precision, channel_bound=8)
    storage.reserve(8, 8, 8, 2)
    domain = native._FourierD3FftPlanDomain(precision)
    domain.acquire(storage)

    held = domain.usage()["allocation"]
    assert held["generation"] == 1
    assert held["real_elements"] == 1024
    assert held["spectrum_elements"] == 640
    assert held["retained_bytes"] > 0
    # Two owners of the token (the storage and the plan), which share exactly
    # one instance of each managed view.
    assert held["allocation_owners"] == 2
    assert held["real_view_use_count"] == 1
    assert held["spectrum_view_use_count"] == 1

    del storage
    gc.collect()

    # The plan outlives the storage: the buffers must still be allocated, with
    # the plan as their only owner.
    retained = domain.usage()["allocation"]
    assert retained["allocation_owners"] == 1
    assert retained["generation"] == held["generation"]
    assert retained["real_elements"] == held["real_elements"]
    assert retained["spectrum_elements"] == held["spectrum_elements"]
    assert retained["retained_bytes"] == held["retained_bytes"]
    # The plan alone now holds the tracked allocation, so the buffers are still
    # allocated rather than dangling.
    assert retained["real_view_use_count"] == 1
    assert retained["spectrum_view_use_count"] == 1
    assert native._kokkos_live_object_count() == baseline + 1
    with pytest.raises(RuntimeError, match="finalize"):
        native._finalize_kokkos()

    assert domain.release()["released"] is True
    assert domain.usage()["allocation"] is None
    gc.collect()
    assert native._kokkos_live_object_count() == baseline


def test_storage_seam_is_private_while_configuration_is_public(native):
    for symbol in _SEAM_SYMBOLS:
        assert symbol not in symmetrix.__all__

    assert "FourierD3Correction" in symmetrix.__all__
    # Task 11 adds the native evaluator class, and the package's fallback forwards any
    # name the native module provides, so reachability no longer says "private".
    # Publication does: the curated export list and __all__ are unchanged.
    assert "FourierD3Kokkos" not in symmetrix.__all__
    assert "FourierD3Kokkos" not in symmetrix._FRONTEND_EXPORTS
    assert symmetrix.FourierD3Correction.__name__ == "FourierD3Correction"
