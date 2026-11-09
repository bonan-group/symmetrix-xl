import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SPHERICART_SOURCE = _ROOT / "libsymmetrix/external/sphericart"
_GPULITE_COMMIT = "3fcc6b888f9d8f691d18af01801732ef4222a161"


def test_pinned_sphericart_uses_wide_edge_offsets():
    """The pinned SpheriCart must carry the wide-offset kernel arithmetic.

    Historically this was validated by applying the 1.0.3 lifecycle patch in a
    build-tree copy; the pin now contains the changes directly, so the guards
    check the submodule source itself.
    """
    cuda_source = (_SPHERICART_SOURCE / "sphericart/src/sphericart_impl.cu").read_text()
    unsafe_edge_offsets = re.findall(
        r"(?:sph|dsph|ddsph|sph_grad|xyz|xyz_grad)\[edge_idx\s*\*",
        cuda_source,
    )
    assert unsafe_edge_offsets == []
    assert cuda_source.count("static_cast<unsigned long long>(edge_idx)") == 4
    assert cuda_source.count("const unsigned long long edge_idx =") == 2

    # Unlike the historical 1.0.3 patch (which widened the kernel signatures),
    # the pinned v2-based source keeps the host-visible signatures at `int`
    # and confines the 64-bit edge arithmetic to the kernel bodies.
    cuda_header = (
        _SPHERICART_SOURCE / "sphericart/include/sphericart_impl.cuh"
    ).read_text()
    assert "edge_idx" not in cuda_header
    cuda_base = (_SPHERICART_SOURCE / "sphericart/src/cuda_base.cpp").read_text()
    assert "x / bdim + (x % bdim != 0)" in cuda_base
    assert "x + bdim - 1" not in cuda_base

    hip_base = (_SPHERICART_SOURCE / "sphericart/src/hip_base.cpp").read_text()
    assert "x / bdim + (x % bdim != 0)" in hip_base
    assert "x + bdim - 1" not in hip_base


def test_pinned_sphericart_carries_async_lifecycle_and_hip():
    cuda_header = (
        _SPHERICART_SOURCE / "sphericart/include/sphericart_cuda.hpp"
    ).read_text()
    assert "compute_with_gradients_async" in cuda_header
    assert "release_device_resources" in cuda_header

    hip_header = (
        _SPHERICART_SOURCE / "sphericart/include/sphericart_hip.hpp"
    ).read_text()
    assert "compute_with_gradients_async" in hip_header
    assert "release_device_resources" in hip_header

    cmake = (_SPHERICART_SOURCE / "sphericart/CMakeLists.txt").read_text()
    assert "SPHERICART_ENABLE_HIP" in cmake

    gradient_stride = 3 * (3 + 1) ** 2
    n46_edges = 42_438_496
    n47_edges = 45_266_828
    int32_max = 2**31 - 1
    assert n46_edges * gradient_stride - 1 <= int32_max
    assert n47_edges * gradient_stride - 1 > int32_max


def test_pinned_sphericart_uses_cuda_header_compatible_gpulite():
    cmake = (_SPHERICART_SOURCE / "sphericart/CMakeLists.txt").read_text()
    assert "https://github.com/bonan-group/gpu-lite.git" in cmake
    assert f"GIT_TAG {_GPULITE_COMMIT}" in cmake
