"""Installed-package contracts for native dispersion support."""

from importlib.metadata import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_FORBIDDEN_RUNTIME_DEPENDENCIES = {
    "torch",
    "torch-dftd",
}

_ROOT = Path(__file__).resolve().parents[2]


def _requirements_for_extra(extra):
    requirements = metadata("symmetrix-xl").get_all("Requires-Dist") or []
    selected = []
    for requirement_text in requirements:
        requirement = Requirement(requirement_text)
        if requirement.marker is None or requirement.marker.evaluate({"extra": extra}):
            selected.append(canonicalize_name(requirement.name))
    return set(selected)


def test_dispersion_compatibility_extra_has_no_torch_runtime_dependency():
    distribution_metadata = metadata("symmetrix-xl")
    assert "dispersion" in (distribution_metadata.get_all("Provides-Extra") or [])
    assert not (_requirements_for_extra("dispersion") & _FORBIDDEN_RUNTIME_DEPENDENCIES)


def test_standard_install_has_no_torch_runtime_dependency():
    assert not (_requirements_for_extra("") & _FORBIDDEN_RUNTIME_DEPENDENCIES)


def test_fourier_d3_user_guide_states_the_model_and_geometry_boundaries():
    guide = (_ROOT / "docs/user/dispersion_corrections.md").read_text()

    required = (
        "FourierD3Correction",
        "fully periodic",
        "modified model",
        "not a drop-in numerical replacement",
        "mesh_spacing",
        "spline_order",
        "decomposition_tolerance",
        "no real-space D3 fallback",
        "self-image",
        "KokkosFFT",
        "FFTW",
        "cuFFT",
        "hipFFT",
        "gfx1151",
        "does not qualify other HIP architectures",
        "host builder",
    )
    for phrase in required:
        assert phrase in guide

    assert "always requires the native Kokkos graph path" not in guide


def test_support_matrix_declares_architecture_specific_fourier_d3_hip_support():
    matrix = (_ROOT / "docs/reference/execution_support_matrix.md").read_text()

    assert "FourierD3Correction" in matrix
    assert "CPU/FFTW" in matrix
    assert "CUDA/cuFFT" in matrix
    assert "HIP/hipFFT" in matrix
    assert "runtime-qualified" in matrix
    assert "gfx1151" in matrix
    assert "does not qualify other HIP architectures" in matrix
