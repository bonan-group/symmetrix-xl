"""Build-reported capability of the reciprocal-space FourierD3 correction.

The correction transforms a particle mesh through KokkosFFT, so it exists only
where the build discovered an FFT provider: FFTW3 for the threaded host, cuFFT
for CUDA, hipFFT for HIP.  The configure step owns that decision; this module
reads it back from the loaded native extension and turns an absent provider into
an explicit unsupported state instead of an approximation.

Nothing here evaluates a correction.  This is the capability seam that the
native storage, kernels, and public API will consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

# Reasons the configure step reports when it cannot enable the feature.  Every
# token names the actual cause; an unrecognized token is reported verbatim
# rather than paraphrased.
_REASON_TEXT = {
    "build-option-disabled": (
        "particle-mesh algorithms were disabled by the build option "
        "SYMMETRIX_PARTICLE_MESH=OFF"
    ),
    "kokkos-disabled": "this build has no Kokkos backend",
    "kokkosfft-source-missing": (
        "the pinned KokkosFFT source is missing from the build tree"
    ),
    "unsupported-execution-space": (
        "no Kokkos host execution space can drive an FFT provider"
    ),
    "fftw3-provider-missing": (
        "no FFTW3 development provider (fftw3.h, libfftw3, libfftw3f) was found"
    ),
    "fftw3-openmp-provider-missing": (
        "no FFTW3 OpenMP provider (libfftw3_omp, libfftw3f_omp) was found"
    ),
    "fftw3-threads-provider-missing": (
        "no FFTW3 threads provider (libfftw3_threads, libfftw3f_threads) was found"
    ),
    "cufft-provider-missing": "no cuFFT provider was found in the CUDA toolkit",
    "hipfft-provider-missing": "no hipFFT provider was found in the ROCm toolkit",
    "unreported": "the native extension does not report FourierD3 capability",
}

NO_FALLBACK_STATEMENT = "there is no KISS, NumPy, host, or real-space D3 fallback"


@dataclass(frozen=True)
class FourierD3Capability:
    """What the loaded native extension was built with."""

    available: bool
    provider: str = ""
    provider_variant: str = ""
    kokkosfft_version: str = ""
    # The Kokkos release the extension was compiled against, empty when the build
    # does not report one, and the execution backend the build selected.
    kokkos_version: str = ""
    execution_backend: str = ""
    reason: str = ""

    @property
    def reason_text(self) -> str:
        if not self.reason:
            return ""
        return _REASON_TEXT.get(self.reason, self.reason)

    def describe(self) -> str:
        kokkos = f" (Kokkos {self.kokkos_version})" if self.kokkos_version else ""
        if self.available:
            provider = self.provider or "an unknown provider"
            if self.provider_variant:
                provider = f"{provider} ({self.provider_variant})"
            version = (
                f"KokkosFFT {self.kokkosfft_version}"
                if self.kokkosfft_version
                else "KokkosFFT (unknown version)"
            )
            return f"FourierD3 is available: {version} via {provider}{kokkos}"
        detail = self.reason_text or "no reason was reported"
        version = (
            f" (KokkosFFT {self.kokkosfft_version})" if self.kokkosfft_version else ""
        )
        return f"FourierD3 is unavailable in this build{version}: {detail}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _payload_from(native_module: Any) -> Mapping[str, Any]:
    query = getattr(native_module, "_fourier_d3_capability", None)
    if callable(query):
        payload: Any = query()
    else:
        build_info = getattr(native_module, "_backend_build_info", None)
        payload = build_info().get("fourier_d3") if callable(build_info) else None
    if payload is None:
        return {"available": False, "reason": "unreported"}
    if not isinstance(payload, Mapping):
        raise ValueError(
            "the native extension reported an invalid FourierD3 capability: "
            f"expected a mapping, got {type(payload).__name__}"
        )
    return payload


def fourier_d3_capability(native_module: Any = None) -> FourierD3Capability:
    """Return the FourierD3 capability of the loaded native extension."""

    if native_module is None:
        from . import load_backend

        native_module = load_backend()
    payload = _payload_from(native_module)
    available = bool(payload.get("available", False))
    provider = str(payload.get("provider") or "")
    if available and not provider:
        raise ValueError(
            "the native extension reports FourierD3 as available without an "
            "FFT provider"
        )
    if not available:
        provider = ""
    return FourierD3Capability(
        available=available,
        provider=provider,
        provider_variant=(
            str(payload.get("provider_variant") or "") if available else ""
        ),
        kokkosfft_version=str(payload.get("kokkosfft_version") or ""),
        # The Kokkos version and the execution backend describe the build rather
        # than the provider, so an unavailable build still reports them.
        kokkos_version=str(payload.get("kokkos_version") or ""),
        execution_backend=str(payload.get("execution_backend") or ""),
        reason=str(payload.get("reason") or ""),
    )


def require_fourier_d3_capability(native_module: Any = None) -> FourierD3Capability:
    """Return the capability or fail with the reason this build lacks it."""

    capability = fourier_d3_capability(native_module)
    if not capability.available:
        raise RuntimeError(
            "FourierD3 requires a KokkosFFT FFT provider. "
            f"{capability.describe()}. Rebuild with a discovered FFTW3, cuFFT, "
            f"or hipFFT provider; {NO_FALLBACK_STATEMENT}."
        )
    return capability
