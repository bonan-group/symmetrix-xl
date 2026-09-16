import re
import shutil
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SPHERICART_SOURCE = _ROOT / "libsymmetrix/external/sphericart"
_SPHERICART_PATCH = (
    _ROOT / "libsymmetrix/cmake/patches/sphericart-1.0.3-async-lifecycle.patch"
)


def test_sphericart_cuda_patch_uses_wide_edge_offsets(tmp_path):
    patched_source = tmp_path / "sphericart"
    shutil.copytree(
        _SPHERICART_SOURCE,
        patched_source,
        ignore=shutil.ignore_patterns(".git"),
    )
    subprocess.run(
        ["git", "apply", "--check", str(_SPHERICART_PATCH)],
        cwd=patched_source,
        check=True,
    )
    subprocess.run(
        ["git", "apply", str(_SPHERICART_PATCH)],
        cwd=patched_source,
        check=True,
    )

    cuda_source = (patched_source / "sphericart/src/sphericart_impl.cu").read_text()
    unsafe_edge_offsets = re.findall(
        r"(?:sph|dsph|ddsph|sph_grad|xyz|xyz_grad)\[edge_idx\s*\*",
        cuda_source,
    )
    assert unsafe_edge_offsets == []
    assert cuda_source.count("static_cast<unsigned long long>(edge_idx)") == 4
    assert cuda_source.count("const unsigned long long edge_idx =") == 2

    cuda_header = (
        patched_source / "sphericart/include/sphericart_impl.cuh"
    ).read_text()
    assert "unsigned long long edge_idx" in cuda_header
    assert "int edge_idx" not in cuda_header

    cuda_base = (patched_source / "sphericart/src/cuda_base.cpp").read_text()
    assert "x / bdim + (x % bdim != 0)" in cuda_base
    assert "x + bdim - 1" not in cuda_base

    gradient_stride = 3 * (3 + 1) ** 2
    n46_edges = 42_438_496
    n47_edges = 45_266_828
    int32_max = 2**31 - 1
    assert n46_edges * gradient_stride - 1 <= int32_max
    assert n47_edges * gradient_stride - 1 > int32_max
