"""Build-reported FourierD3 capability and its KokkosFFT provider boundary.

The reciprocal-space correction cannot run without a KokkosFFT FFT provider, so
the build decides availability at configure time and the native extension
reports that decision.  These tests pin the report contract, the explicit
unsupported state, and the CLI surfaces that must agree with it.  No evaluator
is exercised here; that is Task 11 work.
"""

from __future__ import annotations

import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import symmetrix
from symmetrix import backend_loader
from symmetrix.cli import main as cli
from symmetrix.fourier_d3_capability import (
    FourierD3Capability,
    fourier_d3_capability,
    require_fourier_d3_capability,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

build_wheel_audit = importlib.import_module("_symmetrix_build.wheel_audit")
BuildError = importlib.import_module("_symmetrix_build.command").BuildError
audit_wheel = build_wheel_audit.audit_wheel

_AVAILABLE_PAYLOAD = {
    "available": True,
    "provider": "fftw3",
    "provider_variant": "FFTW_OPENMP",
    "kokkosfft_version": "2.0.0",
    "reason": "",
}
_UNAVAILABLE_PAYLOAD = {
    "available": False,
    "provider": "",
    "provider_variant": "",
    "kokkosfft_version": "2.0.0",
    "reason": "fftw3-provider-missing",
}


def _native(payload, *, build_info=None, capability_symbol=True):
    module = SimpleNamespace(__file__="/native.so")
    if capability_symbol:
        module._fourier_d3_capability = lambda: dict(payload)
    if build_info is not None:
        module._backend_build_info = lambda: dict(build_info)
    return module


# --------------------------------------------------------------------------
# capability contract
# --------------------------------------------------------------------------


def test_capability_reports_a_discovered_provider():
    capability = fourier_d3_capability(_native(_AVAILABLE_PAYLOAD))

    assert capability == FourierD3Capability(
        available=True,
        provider="fftw3",
        provider_variant="FFTW_OPENMP",
        kokkosfft_version="2.0.0",
        reason="",
    )
    assert capability.reason_text == ""
    assert "available" in capability.describe()
    assert "fftw3" in capability.describe()
    assert "2.0.0" in capability.describe()


def test_capability_reports_an_absent_provider():
    capability = fourier_d3_capability(_native(_UNAVAILABLE_PAYLOAD))

    assert capability.available is False
    assert capability.provider == ""
    assert capability.provider_variant == ""
    # The pinned tree version is still truthful when the provider is missing.
    assert capability.kokkosfft_version == "2.0.0"
    assert capability.reason == "fftw3-provider-missing"
    assert "FFTW3" in capability.reason_text
    assert "unavailable" in capability.describe()


@pytest.mark.parametrize(
    ("reason", "expected"),
    (
        ("build-option-disabled", "SYMMETRIX_PARTICLE_MESH"),
        ("kokkos-disabled", "Kokkos"),
        ("kokkosfft-source-missing", "KokkosFFT"),
        ("cufft-provider-missing", "cuFFT"),
        ("hipfft-provider-missing", "hipFFT"),
    ),
)
def test_every_build_reason_token_has_explicit_text(reason, expected):
    capability = fourier_d3_capability(
        _native({**_UNAVAILABLE_PAYLOAD, "reason": reason})
    )

    assert expected in capability.reason_text
    assert reason not in capability.reason_text


def test_unknown_reason_tokens_are_reported_verbatim():
    capability = fourier_d3_capability(
        _native({**_UNAVAILABLE_PAYLOAD, "reason": "some-future-reason"})
    )

    assert capability.reason_text == "some-future-reason"


def test_capability_rejects_a_falsely_available_report():
    with pytest.raises(ValueError, match="without an FFT provider"):
        fourier_d3_capability(_native({**_AVAILABLE_PAYLOAD, "provider": ""}))


def test_capability_rejects_a_non_mapping_report():
    native = SimpleNamespace(__file__="/native.so", _fourier_d3_capability=lambda: 7)

    with pytest.raises(ValueError, match="capability"):
        fourier_d3_capability(native)


def test_capability_falls_back_to_the_build_info_report():
    native = _native(
        _AVAILABLE_PAYLOAD,
        build_info={"fourier_d3": dict(_AVAILABLE_PAYLOAD)},
        capability_symbol=False,
    )

    assert fourier_d3_capability(native).available is True


def test_capability_reports_an_unreported_extension():
    capability = fourier_d3_capability(_native(None, capability_symbol=False))

    assert capability.available is False
    assert capability.reason == "unreported"
    assert "does not report" in capability.reason_text


def test_require_accepts_a_discovered_provider():
    assert require_fourier_d3_capability(_native(_AVAILABLE_PAYLOAD)).available


def test_require_rejects_an_absent_provider():
    with pytest.raises(RuntimeError) as error:
        require_fourier_d3_capability(_native(_UNAVAILABLE_PAYLOAD))

    message = str(error.value)
    assert "FourierD3" in message
    assert "FFTW3" in message
    assert "2.0.0" in message
    assert "no KISS, NumPy, host, or real-space" in message


def test_installed_backend_reports_its_fourier_d3_capability():
    build_info = getattr(symmetrix, "_backend_build_info", None)
    native = symmetrix.load_backend()
    build_info = getattr(native, "_backend_build_info", None)
    if not callable(build_info):
        pytest.skip("the installed native extension reports no build information")
    report = build_info().get("fourier_d3")
    if not isinstance(report, dict):
        pytest.skip(
            "the installed native extension predates the FourierD3 capability "
            "report; rebuild the backend to exercise this contract"
        )

    capability = fourier_d3_capability(native)

    assert capability.provider == str(report.get("provider") or "")
    assert capability.provider_variant == str(report.get("provider_variant") or "")
    assert capability.kokkosfft_version == str(report.get("kokkosfft_version") or "")
    assert capability.reason == str(report.get("reason") or "")
    assert capability.available == bool(report.get("available"))
    if capability.available:
        assert capability.provider
        assert capability.kokkosfft_version
        assert capability.reason == ""
    else:
        assert capability.provider == ""
        assert capability.reason


# --------------------------------------------------------------------------
# CLI surfaces
# --------------------------------------------------------------------------


def test_doctor_json_reports_fourier_d3_capability(monkeypatch, capsys):
    native = _native(_AVAILABLE_PAYLOAD)
    monkeypatch.setattr(symmetrix, "load_backend", lambda *_args: native)
    monkeypatch.setattr(symmetrix, "selected_backend", lambda: {"backend": "cpu"})
    monkeypatch.setattr(
        cli,
        "openmp_runtime_diagnostics",
        lambda strict: {"status": "ok", "ok": True, "strict": strict, "checks": []},
    )

    assert cli.main(["doctor", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["fourier_d3"]["available"] is True
    assert report["fourier_d3"]["provider"] == "fftw3"
    assert report["native"]["build_info"] is not None


def test_doctor_reports_an_unsupported_provider(monkeypatch, capsys):
    native = _native(_UNAVAILABLE_PAYLOAD)
    monkeypatch.setattr(symmetrix, "load_backend", lambda *_args: native)
    monkeypatch.setattr(symmetrix, "selected_backend", lambda: {"backend": "cpu"})
    monkeypatch.setattr(
        cli,
        "openmp_runtime_diagnostics",
        lambda strict: {"status": "ok", "ok": True, "strict": strict, "checks": []},
    )

    assert cli.main(["doctor", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["fourier_d3"]["available"] is False
    assert report["fourier_d3"]["reason"] == "fftw3-provider-missing"


def test_version_probe_reports_fourier_d3_capability(monkeypatch, capsys):
    native = _native(_AVAILABLE_PAYLOAD, build_info={"fourier_d3": _AVAILABLE_PAYLOAD})
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda _: "0.1.1")
    monkeypatch.setattr(
        backend_loader,
        "available_backends",
        lambda _: [{"selector": "cpu", "backend": "cpu", "architecture": "native"}],
    )
    monkeypatch.setattr(symmetrix, "load_backend", lambda: native)
    monkeypatch.setattr(
        symmetrix, "selected_backend", lambda: {"selector": "cpu", "backend": "cpu"}
    )

    assert cli.main(["version", "--probe", "--json"]) == 0

    probe = json.loads(capsys.readouterr().out)["probe"]
    assert probe["build_info"]["fourier_d3"]["provider"] == "fftw3"
    assert probe["build_info"]["fourier_d3"]["kokkosfft_version"] == "2.0.0"


def test_backend_show_probe_reports_fourier_d3_capability(monkeypatch, capsys):
    native = _native(_UNAVAILABLE_PAYLOAD)
    native._openmp_runtime_info = lambda: {"kokkos_execution_space": "OpenMP"}
    monkeypatch.setattr(symmetrix, "load_backend", lambda *_args: native)
    monkeypatch.setattr(
        backend_loader,
        "select_backend",
        lambda *_args: SimpleNamespace(
            to_dict=lambda: {
                "selector": "cpu",
                "backend": "cpu",
                "architecture": "native",
                "distribution": "symmetrix-xl",
            }
        ),
    )

    assert cli.main(["backend", "show", "cpu", "--probe", "--json"]) == 0

    value = json.loads(capsys.readouterr().out)
    assert value["fourier_d3"]["available"] is False
    assert value["fourier_d3"]["reason"] == "fftw3-provider-missing"


# --------------------------------------------------------------------------
# build integration contract
# --------------------------------------------------------------------------


def test_libsymmetrix_cmake_never_downloads_dependencies():
    core = (REPOSITORY_ROOT / "libsymmetrix/CMakeLists.txt").read_text()

    for forbidden in (
        "FetchContent",
        "ExternalProject",
        "file(DOWNLOAD",
        "GIT_REPOSITORY",
    ):
        assert forbidden not in core


def test_libsymmetrix_cmake_gates_fourier_d3_on_a_kokkosfft_provider():
    core = (REPOSITORY_ROOT / "libsymmetrix/CMakeLists.txt").read_text()

    assert 'set(SYMMETRIX_PARTICLE_MESH "AUTO" CACHE STRING' in core
    assert (
        "set_property(CACHE SYMMETRIX_PARTICLE_MESH PROPERTY STRINGS AUTO ON OFF)"
        in core
    )
    assert (
        "set(SYMMETRIX_KOKKOSFFT_SOURCE_DIR\n"
        '    "${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos-fft")' in core
    )
    # Upstream's install rules must not leak headers or CMake exports into wheels.
    assert "EXCLUDE_FROM_ALL" in core
    assert "KokkosFFT_ENABLE_FFTW" in core
    assert "KokkosFFT_ENABLE_CUFFT" in core
    assert "KokkosFFT_ENABLE_HIPFFT" in core
    assert "KokkosFFT::fft" in core
    for token in (
        "fftw3-provider-missing",
        "fftw3-openmp-provider-missing",
        "cufft-provider-missing",
        "hipfft-provider-missing",
        "kokkosfft-source-missing",
        "build-option-disabled",
    ):
        assert token in core


def test_cuda_and_hip_use_their_matching_vendor_fft_provider():
    core = (REPOSITORY_ROOT / "libsymmetrix/CMakeLists.txt").read_text()

    assert "find_package(CUDAToolkit QUIET COMPONENTS cufft)" in core
    assert "TARGET CUDA::cufft" in core
    assert "set(KokkosFFT_ENABLE_CUFFT ON)" in core
    assert "find_package(hipfft CONFIG QUIET)" in core
    assert "TARGET hip::hipfft" in core
    assert "set(KokkosFFT_ENABLE_HIPFFT ON)" in core
    # rocFFT is never substituted for hipFFT, and oneMKL never for either.
    assert "set(KokkosFFT_ENABLE_ROCFFT OFF)" in core
    assert "set(KokkosFFT_ENABLE_ONEMKL OFF)" in core


def test_forcing_particle_mesh_without_a_provider_fails_instead_of_fetching():
    core = (REPOSITORY_ROOT / "libsymmetrix/CMakeLists.txt").read_text()

    assert (
        "if(NOT SYMMETRIX_PARTICLE_MESH_AVAILABLE AND "
        'SYMMETRIX_PARTICLE_MESH STREQUAL "ON")' in core
    )
    assert "never fetch dependencies" in core


def test_capability_build_metadata_is_published_to_the_binding():
    package = (REPOSITORY_ROOT / "symmetrix/CMakeLists.txt").read_text()
    binding = (REPOSITORY_ROOT / "symmetrix/source/cpp/symmetrix.cpp").read_text()
    metadata = (
        REPOSITORY_ROOT / "symmetrix/source/cpp/build_metadata.hpp.in"
    ).read_text()

    assert "source/cpp/build_metadata.hpp.in" in package
    assert '"${CMAKE_CURRENT_BINARY_DIR}/generated/build_metadata.hpp"' in package

    for macro in (
        "SYMMETRIX_FOURIER_D3_AVAILABLE",
        "SYMMETRIX_FOURIER_D3_PROVIDER",
        "SYMMETRIX_FOURIER_D3_PROVIDER_VARIANT",
        "SYMMETRIX_FOURIER_D3_KOKKOSFFT_VERSION",
        "SYMMETRIX_FOURIER_D3_REASON",
    ):
        assert macro in metadata
        assert macro in binding

    assert '"fourier_d3"' in binding
    assert "_fourier_d3_capability" in binding


def test_configure_records_the_resolved_provider_outside_the_wheel():
    core = (REPOSITORY_ROOT / "libsymmetrix/CMakeLists.txt").read_text()
    template = (
        REPOSITORY_ROOT / "libsymmetrix/cmake/fourier_d3_capability.json.in"
    ).read_text()

    assert "cmake/fourier_d3_capability.json.in" in core
    assert '"policy": "@SYMMETRIX_PARTICLE_MESH@"' in template
    assert (
        "cmake/fourier_d3_capability.json.in\n"
        '    "${CMAKE_CURRENT_BINARY_DIR}/fourier-d3-capability.json" @ONLY)' in core
    )
    for key in (
        "available",
        "provider",
        "provider_variant",
        "kokkosfft_version",
        "reason",
        "provider_libraries",
    ):
        assert f'"{key}"' in template


def _wheel_with(tmp_path, members):
    wheel = tmp_path / "symmetrix_xl-0.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "symmetrix/_backend_cpu.json",
            json.dumps(
                {
                    "distribution": "symmetrix-xl",
                    "frontend_version": "0.2.0",
                    "source_commit": "a" * 40,
                    "source_content_sha256": "b" * 64,
                    "native_source_content_sha256": "c" * 64,
                    "source_dirty": False,
                }
            ),
        )
        archive.writestr(
            "symmetrix_xl-0.2.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: symmetrix-xl\nVersion: 0.2.0\n",
        )
        for name in members:
            archive.writestr(name, b"payload\n")
    return wheel


@pytest.mark.parametrize(
    "member",
    (
        "symmetrix/lib/libfftw3.so.3",
        "symmetrix/include/KokkosFFT.hpp",
        "symmetrix/lib/cmake/KokkosFFT/KokkosFFTConfig.cmake",
    ),
)
def test_wheel_audit_rejects_a_bundled_fft_provider(tmp_path, member):
    # A release wheel links the discovered provider dynamically; shipping a
    # provider library or the vendored KokkosFFT headers would add build
    # payloads that the audit must reject.
    with pytest.raises(BuildError, match="forbidden archive component"):
        audit_wheel(_wheel_with(tmp_path, [member]))


def test_wheel_audit_accepts_a_release_payload(tmp_path):
    wheel = _wheel_with(
        tmp_path,
        [
            "symmetrix/__init__.py",
            "symmetrix/_native_cpu.cpython-312-x86_64-linux-gnu.so",
        ],
    )

    audit_wheel(wheel)


# --------------------------------------------------------------------------
# the Kokkos version and the execution backend the build reports


# --------------------------------------------------------------------------
# the Kokkos version and the execution backend the build reports
# --------------------------------------------------------------------------


def test_capability_reports_the_kokkos_version_the_build_compiled():
    capability = fourier_d3_capability(
        _native({**_AVAILABLE_PAYLOAD, "kokkos_version": "5.0.2"})
    )

    assert capability.kokkos_version == "5.0.2"
    assert "5.0.2" in capability.describe()


def test_capability_reports_the_execution_backend_the_build_compiled():
    capability = fourier_d3_capability(
        _native({**_AVAILABLE_PAYLOAD, "execution_backend": "cpu"})
    )

    assert capability.execution_backend == "cpu"


def test_capability_leaves_an_unreported_kokkos_version_empty():
    capability = fourier_d3_capability(_native(dict(_AVAILABLE_PAYLOAD)))

    # A build that does not report its Kokkos version says so; nothing is invented.
    assert capability.kokkos_version == ""
    assert capability.execution_backend == ""


def test_installed_backend_reports_its_kokkos_version_and_backend():
    native = symmetrix.load_backend()
    report = native._fourier_d3_capability()
    capability = fourier_d3_capability(native)

    assert capability.kokkos_version == str(report.get("kokkos_version") or "")
    assert capability.execution_backend == str(report.get("execution_backend") or "")
    if capability.available:
        assert capability.kokkos_version
        # The execution backend is the same compiled backend the build info reports.
        assert capability.execution_backend == str(
            native._backend_build_info().get("backend") or ""
        )
