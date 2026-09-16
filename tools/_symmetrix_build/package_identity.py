"""Authoritative names for frontend and architecture-qualified packages."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .command import BuildError
from .manifest import TargetManifest
from .targets import BASE_CPU_TARGETS, CPU_ADDON_LABELS


@dataclass(frozen=True)
class PackageIdentity:
    distribution: str
    package: str
    module: str
    selector: str
    backend: str
    architecture: str
    toolkit: str
    base: bool


def _toolkit_major(manifest: TargetManifest) -> str:
    match = re.match(r"(\d+)", manifest.toolchain.toolkit_version)
    if match is None:
        raise BuildError(
            f"cannot derive toolkit generation from {manifest.toolchain.toolkit_version!r}"
        )
    return match.group(1)


def package_identity(manifest: TargetManifest) -> PackageIdentity:
    if manifest.backend == "cpu":
        architecture = manifest.host_target
        if architecture in BASE_CPU_TARGETS:
            return PackageIdentity(
                distribution="symmetrix-xl",
                package="symmetrix",
                module="_native_cpu",
                selector="cpu",
                backend="cpu",
                architecture=architecture,
                toolkit="",
                base=True,
            )
        selector = f"cpu-{CPU_ADDON_LABELS.get(architecture, architecture)}"
        normalized = selector.replace("-", "_")
        return PackageIdentity(
            distribution=f"symmetrix-xl-{selector}",
            package=f"symmetrix_backend_{normalized}",
            module=f"_native_{normalized}",
            selector=selector,
            backend="cpu",
            architecture=architecture,
            toolkit="",
            base=False,
        )
    if manifest.backend == "cuda":
        toolkit = f"cuda{_toolkit_major(manifest)}"
    elif manifest.backend == "hip":
        toolkit = f"rocm{_toolkit_major(manifest)}"
    else:
        raise BuildError(f"unsupported manifest backend: {manifest.backend}")
    device_target = manifest.architecture.device_target.lower().replace("_", "")
    selector = f"{toolkit}-{device_target}"
    normalized = selector.replace("-", "_")
    return PackageIdentity(
        distribution=f"symmetrix-xl-{selector}",
        package=f"symmetrix_backend_{normalized}",
        module=f"_native_{normalized}",
        selector=selector,
        backend=manifest.backend,
        architecture=device_target,
        toolkit=toolkit,
        base=False,
    )
