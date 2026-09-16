"""Backend target normalization and pinned Kokkos architecture mappings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .command import BuildError

CUDA_TARGETS = {
    "sm50": "MAXWELL50",
    "sm52": "MAXWELL52",
    "sm53": "MAXWELL53",
    "sm60": "PASCAL60",
    "sm61": "PASCAL61",
    "sm70": "VOLTA70",
    "sm72": "VOLTA72",
    "sm75": "TURING75",
    "sm80": "AMPERE80",
    "sm86": "AMPERE86",
    "sm87": "AMPERE87",
    "sm89": "ADA89",
    "sm90": "HOPPER90",
    "sm100": "BLACKWELL100",
    "sm120": "BLACKWELL120",
}

# The exact compiler target is retained independently from the Kokkos trait.
# gfx1151 is the qualified compatibility mapping used by the pinned Kokkos.
HIP_TARGETS = {
    "gfx906": "AMD_GFX906",
    "gfx908": "AMD_GFX908",
    "gfx90a": "AMD_GFX90A",
    "gfx940": "AMD_GFX940",
    "gfx942": "AMD_GFX942",
    "gfx1030": "AMD_GFX1030",
    "gfx1100": "AMD_GFX1100",
    "gfx1103": "AMD_GFX1103",
    "gfx1151": "AMD_GFX1100",
    "gfx1201": "AMD_GFX1201",
}


@dataclass(frozen=True)
class Architecture:
    requested: str
    device_target: str
    kokkos_trait: str
    compiler_target: str


CPU_TARGETS = ("native", "x86-64-v3", "x86-64-v4", "none")
BASE_CPU_TARGETS = ("native", "x86-64-v3", "none")
CPU_ADDON_LABELS = {"x86-64-v4": "avx512"}


def normalize_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend not in {"auto", "cpu", "cuda", "hip"}:
        raise BuildError(f"invalid backend {value!r}; expected auto, cpu, cuda, or hip")
    return backend


def canonical_cuda_target(value: str) -> str:
    target = value.strip().lower().replace("_", "").replace(".", "")
    if target.isdigit():
        target = "sm" + target
    if not re.fullmatch(r"sm[0-9]+", target):
        raise BuildError(f"CUDA architecture must look like sm120, not {value!r}")
    return target


def normalize_cuda_target(value: str) -> str:
    target = canonical_cuda_target(value)
    if target not in CUDA_TARGETS:
        supported = ", ".join(CUDA_TARGETS)
        raise BuildError(
            f"CUDA architecture {target} is not supported by pinned Kokkos; "
            f"supported targets: {supported}"
        )
    return target


def validate_cuda_target_for_toolkit(toolkit_version: str, target: str) -> None:
    normalized = normalize_cuda_target(target)
    try:
        toolkit_major = int(toolkit_version.split(".", 1)[0])
    except ValueError as error:
        raise BuildError(f"invalid CUDA toolkit version {toolkit_version!r}") from error
    compute_capability = int(normalized[2:])
    if toolkit_major >= 13 and compute_capability < 75:
        raise BuildError(
            f"CUDA {toolkit_version} does not support offline compilation for "
            f"{normalized}; use a CUDA 12 toolkit for Volta targets"
        )


def canonical_hip_target(value: str) -> str:
    target = value.strip().lower().split(":", 1)[0]
    if not re.fullmatch(r"gfx[0-9a-f]+", target):
        raise BuildError(f"HIP architecture must look like gfx1151, not {value!r}")
    return target


def normalize_hip_target(value: str) -> str:
    target = canonical_hip_target(value)
    if target not in HIP_TARGETS:
        supported = ", ".join(HIP_TARGETS)
        raise BuildError(
            f"HIP architecture {target} is not supported by pinned Kokkos; "
            f"supported targets: {supported}"
        )
    return target


def cuda_architecture(requested: str, detected: str) -> Architecture:
    target = normalize_cuda_target(detected if requested == "auto" else requested)
    return Architecture(requested, target, CUDA_TARGETS[target], f"sm_{target[2:]}")


def hip_architecture(requested: str, detected: str) -> Architecture:
    target = normalize_hip_target(detected if requested == "auto" else requested)
    return Architecture(requested, target, HIP_TARGETS[target], target)


def cpu_architecture(requested: str) -> Architecture:
    target = requested.strip().lower()
    if target not in CPU_TARGETS:
        supported = ", ".join(CPU_TARGETS)
        raise BuildError(
            f"invalid CPU target {requested!r}; expected one of: {supported}"
        )
    return Architecture(target, target, "NATIVE" if target == "native" else "", target)
