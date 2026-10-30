"""Build-reported capability of the shared particle-mesh implementation."""

from __future__ import annotations

from pathlib import Path

import pytest
import symmetrix

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PROVIDER_KEYS = (
    "available",
    "provider",
    "provider_variant",
    "kokkosfft_version",
    "reason",
)


@pytest.fixture
def native():
    return symmetrix.load_backend()


def test_particle_mesh_capability_matches_fourier_d3_provider(native):
    mesh = native._particle_mesh_capability()
    d3 = native._fourier_d3_capability()

    assert {key: mesh[key] for key in _PROVIDER_KEYS} == {
        key: d3[key] for key in _PROVIDER_KEYS
    }


def test_backend_build_info_reports_particle_mesh_capability(native):
    mesh = native._particle_mesh_capability()

    assert native._backend_build_info()["particle_mesh"] == mesh


def test_particle_mesh_capability_uses_shared_build_policy():
    core = (REPOSITORY_ROOT / "libsymmetrix/CMakeLists.txt").read_text()
    template = (
        REPOSITORY_ROOT / "libsymmetrix/cmake/particle_mesh_capability.json.in"
    ).read_text()

    assert 'set(SYMMETRIX_PARTICLE_MESH "AUTO" CACHE STRING' in core
    assert "SYMMETRIX_FOURIER_D3 is deprecated" in core
    assert "Conflicting particle-mesh policies:" in core
    assert '"policy": "@SYMMETRIX_PARTICLE_MESH@"' in template
    assert "set(SYMMETRIX_PARTICLE_MESH_AVAILABLE 0)" in core
    assert "set(SYMMETRIX_PARTICLE_MESH_AVAILABLE 0 CACHE" not in core
    assert (
        "if(NOT SYMMETRIX_PARTICLE_MESH_AVAILABLE AND "
        'SYMMETRIX_PARTICLE_MESH STREQUAL "ON")' in core
    )


def test_particle_mesh_capability_is_forwarded_to_the_native_module():
    package = (REPOSITORY_ROOT / "symmetrix/CMakeLists.txt").read_text()
    binding = (REPOSITORY_ROOT / "symmetrix/source/cpp/symmetrix.cpp").read_text()
    metadata = (
        REPOSITORY_ROOT / "symmetrix/source/cpp/build_metadata.hpp.in"
    ).read_text()

    for macro in (
        "SYMMETRIX_PARTICLE_MESH_AVAILABLE",
        "SYMMETRIX_PARTICLE_MESH_PROVIDER",
        "SYMMETRIX_PARTICLE_MESH_PROVIDER_VARIANT",
        "SYMMETRIX_PARTICLE_MESH_KOKKOSFFT_VERSION",
        "SYMMETRIX_PARTICLE_MESH_REASON",
    ):
        assert macro in metadata
        assert macro in binding

    assert "SYMMETRIX_PARTICLE_MESH_AVAILABLE" in package

    assert '"particle_mesh"' in binding
    assert "_particle_mesh_capability" in binding
