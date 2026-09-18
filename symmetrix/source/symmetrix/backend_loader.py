"""Discover and load exactly one Symmetrix native backend."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BACKEND_DESCRIPTOR_SCHEMA = 1
NATIVE_ABI = 1
ENTRY_POINT_GROUP = "symmetrix.backends"
BACKEND_ENVIRONMENT = "SYMMETRIX_BACKEND"


class BackendError(ImportError):
    """Raised when no requested native backend can be loaded safely."""


@dataclass(frozen=True)
class BackendDescriptor:
    selector: str
    backend: str
    architecture: str
    distribution: str
    frontend_version: str
    native_abi: int
    package: str
    module: str
    toolkit: str = ""
    compiler: str = ""
    source_commit: str = ""
    source_dirty: bool | None = None
    target_manifest_fingerprint: str = ""
    schema_version: int = BACKEND_DESCRIPTOR_SCHEMA
    descriptor_path: str = ""

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, descriptor_path: str = ""
    ) -> BackendDescriptor:
        if value.get("schema_version") != BACKEND_DESCRIPTOR_SCHEMA:
            raise BackendError(
                "unsupported backend descriptor schema "
                f"{value.get('schema_version')!r}; expected {BACKEND_DESCRIPTOR_SCHEMA}"
            )
        try:
            descriptor = cls(
                selector=str(value["selector"]),
                backend=str(value["backend"]),
                architecture=str(value["architecture"]),
                distribution=str(value["distribution"]),
                frontend_version=str(value["frontend_version"]),
                native_abi=int(value["native_abi"]),
                package=str(value["package"]),
                module=str(value["module"]),
                toolkit=str(value.get("toolkit", "")),
                compiler=str(value.get("compiler", "")),
                source_commit=str(value.get("source_commit", "")),
                source_dirty=(
                    None
                    if "source_dirty" not in value
                    else bool(value.get("source_dirty"))
                ),
                target_manifest_fingerprint=str(
                    value.get("target_manifest_fingerprint", "")
                ),
                schema_version=int(value["schema_version"]),
                descriptor_path=descriptor_path,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BackendError(f"invalid backend descriptor: {error}") from error
        if descriptor.backend not in {"cpu", "cuda", "hip"}:
            raise BackendError(
                f"unsupported backend kind {descriptor.backend!r} in {descriptor_path}"
            )
        for name in (descriptor.package, descriptor.module):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise BackendError(
                    f"invalid native import component {name!r} in {descriptor_path}"
                )
        return descriptor

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_selected: BackendDescriptor | None = None
_native_module = None


def _cpu_descriptor(frontend_version: str) -> BackendDescriptor:
    installed = Path(__file__).with_name("_backend_cpu.json")
    if installed.is_file():
        return _read_descriptor(installed)
    return BackendDescriptor(
        selector="cpu",
        backend="cpu",
        architecture="native-or-wheel-baseline",
        distribution="symmetrix-xl",
        frontend_version=frontend_version,
        native_abi=NATIVE_ABI,
        package="symmetrix",
        module="_native_cpu",
        descriptor_path="bundled",
    )


def _read_descriptor(path: Path) -> BackendDescriptor:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BackendError(f"cannot read backend descriptor {path}: {error}") from error
    if not isinstance(value, dict):
        raise BackendError(f"backend descriptor {path} must contain a JSON object")
    return BackendDescriptor.from_dict(value, descriptor_path=str(path))


def _entry_points():
    points = importlib.metadata.entry_points()
    if hasattr(points, "select"):
        return points.select(group=ENTRY_POINT_GROUP)
    return points.get(ENTRY_POINT_GROUP, ())


def discover_backends(frontend_version: str) -> tuple[BackendDescriptor, ...]:
    """Return descriptors without importing any native backend package."""

    found = [_cpu_descriptor(frontend_version)]
    for point in _entry_points():
        package = point.value.partition(":")[0].strip()
        if not package or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", package):
            raise BackendError(
                f"invalid {ENTRY_POINT_GROUP} entry point {point.name}={point.value}"
            )
        relative = Path(*package.split(".")) / "backend.json"
        path = Path(point.dist.locate_file(relative))
        descriptor = _read_descriptor(path)
        if descriptor.package != package:
            raise BackendError(
                f"descriptor {path} declares package {descriptor.package!r}, "
                f"but entry point declares {package!r}"
            )
        found.append(descriptor)

    extra_paths = os.environ.get("SYMMETRIX_BACKEND_DESCRIPTOR_PATH", "")
    for raw_path in filter(None, extra_paths.split(os.pathsep)):
        found.append(_read_descriptor(Path(raw_path)))

    selectors: dict[str, BackendDescriptor] = {}
    for descriptor in found:
        previous = selectors.get(descriptor.selector)
        if previous is not None:
            raise BackendError(
                f"duplicate backend selector {descriptor.selector!r}: "
                f"{previous.distribution} and {descriptor.distribution}"
            )
        selectors[descriptor.selector] = descriptor
    return tuple(selectors.values())


def _visible_targets() -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {"cuda": set(), "hip": set()}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0:
        for value in result.stdout.splitlines():
            match = re.search(r"(\d+)\.(\d+)", value)
            if match:
                targets["cuda"].add(f"sm{match.group(1)}{match.group(2)}")

    try:
        result = subprocess.run(
            ["rocminfo"], check=False, capture_output=True, text=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0:
        targets["hip"].update(
            value.lower() for value in re.findall(r"\bgfx[0-9a-f]+\b", result.stdout)
        )
    return targets


_V3_CPU_FLAGS = frozenset(
    {
        "abm",
        "avx",
        "avx2",
        "bmi1",
        "bmi2",
        "cx16",
        "f16c",
        "fma",
        "lahf_lm",
        "movbe",
        "pni",
        "popcnt",
        "sse4_1",
        "sse4_2",
        "ssse3",
        "xsave",
    }
)
_V4_CPU_FLAGS = frozenset({"avx512bw", "avx512cd", "avx512dq", "avx512f", "avx512vl"})


def _cpu_feature_requirements(architecture: str) -> frozenset[str] | None:
    if architecture == "x86-64-v3":
        return _V3_CPU_FLAGS
    if architecture == "x86-64-v4":
        return _V3_CPU_FLAGS | _V4_CPU_FLAGS
    return None


def _host_cpu_flags() -> set[str]:
    cpuinfo = Path("/proc/cpuinfo").read_text().lower()
    matches = re.findall(r"^flags\s*:\s*(.*)$", cpuinfo, re.MULTILINE)
    return set(matches[0].split()) if matches else set()


def _host_capability_level() -> int:
    """Highest labeled x86-64 microarchitecture level this host satisfies.

    Local ``native`` builds are compiled on the host they run on, so their
    effective level tracks what the host supports instead of a fixed label.
    """

    try:
        flags = _host_cpu_flags()
    except OSError:
        return 1
    if (_V3_CPU_FLAGS | _V4_CPU_FLAGS) <= flags:
        return 4
    if _V3_CPU_FLAGS <= flags:
        return 3
    return 1


def _cpu_backend_level(architecture: str, host_level: int) -> int:
    if architecture == "native":
        return host_level
    return {
        "none": 0,
        "native-or-wheel-baseline": 1,
        "x86-64-v3": 3,
        "x86-64-v4": 4,
    }.get(architecture, 1)


def _cpu_backend_priority(
    descriptor: BackendDescriptor, host_level: int
) -> tuple[bool, int]:
    """Prefer a host-native build, then the highest compatible ISA level."""

    return (
        descriptor.architecture == "native",
        _cpu_backend_level(descriptor.architecture, host_level),
    )


def _compatible(descriptor: BackendDescriptor, frontend_version: str) -> None:
    if descriptor.frontend_version != frontend_version:
        raise BackendError(
            f"backend {descriptor.selector!r} requires symmetrix "
            f"{descriptor.frontend_version}, but {frontend_version} is installed"
        )
    if descriptor.native_abi != NATIVE_ABI:
        raise BackendError(
            f"backend {descriptor.selector!r} uses native ABI "
            f"{descriptor.native_abi}, expected {NATIVE_ABI}"
        )
    required = (
        _cpu_feature_requirements(descriptor.architecture)
        if descriptor.backend == "cpu"
        else None
    )
    if required is not None:
        try:
            flags = _host_cpu_flags()
        except OSError as error:
            raise BackendError(
                "cannot validate the "
                f"{descriptor.architecture} CPU requirement before native import"
            ) from error
        missing = sorted(required - flags)
        if missing:
            raise BackendError(
                f"backend {descriptor.selector!r} requires "
                f"{descriptor.architecture}; missing CPU features: "
                + ", ".join(missing)
                + ". Build symmetrix from source for this host."
            )


def _preferred_cpu_backend(
    descriptors: tuple[BackendDescriptor, ...], frontend_version: str
) -> BackendDescriptor:
    compatible: list[BackendDescriptor] = []
    base_error: BackendError | None = None
    for descriptor in descriptors:
        if descriptor.backend != "cpu":
            continue
        try:
            _compatible(descriptor, frontend_version)
        except BackendError as error:
            if descriptor.selector == "cpu" and base_error is None:
                base_error = error
            continue
        compatible.append(descriptor)
    if not compatible:
        if base_error is not None:
            raise base_error
        raise BackendError("no compatible CPU backend is installed")
    host_level = _host_capability_level()

    def priority(item: BackendDescriptor) -> tuple[bool, int]:
        return _cpu_backend_priority(item, host_level)

    highest = max(priority(item) for item in compatible)
    top = [item for item in compatible if priority(item) == highest]
    base = next((item for item in top if item.selector == "cpu"), None)
    if base is not None:
        return base
    if len(top) > 1:
        names = ", ".join(sorted(item.selector for item in top))
        raise BackendError(
            f"multiple CPU backends match this host: {names}; set "
            f"{BACKEND_ENVIRONMENT} explicitly"
        )
    return top[0]


def select_backend(
    frontend_version: str, request: str | None = None
) -> BackendDescriptor:
    descriptors = discover_backends(frontend_version)
    by_selector = {item.selector: item for item in descriptors}
    requested = (request or os.environ.get(BACKEND_ENVIRONMENT, "auto")).strip()
    if requested and requested != "auto":
        try:
            selected = by_selector[requested]
        except KeyError as error:
            installed = ", ".join(sorted(by_selector))
            package_hint = (
                "symmetrix-xl" if requested == "cpu" else f"symmetrix-xl-{requested}"
            )
            raise BackendError(
                f"requested backend {requested!r} is not installed; "
                f"available backends: {installed}. Install it with "
                f"'pip install {package_hint}'."
            ) from error
        _compatible(selected, frontend_version)
        if selected.backend in {"cuda", "hip"}:
            visible = _visible_targets()[selected.backend]
            target = selected.architecture.lower().replace("_", "")
            if target not in visible:
                detected = ", ".join(sorted(visible)) or "none"
                raise BackendError(
                    f"backend {selected.selector!r} requires {target}, but visible "
                    f"{selected.backend} targets are: {detected}"
                )
        return selected

    visible = _visible_targets()
    candidates = [
        item
        for item in descriptors
        if item.backend in {"cuda", "hip"}
        and item.architecture.lower().replace("_", "") in visible[item.backend]
    ]
    compatible = []
    for item in candidates:
        try:
            _compatible(item, frontend_version)
        except BackendError:
            continue
        compatible.append(item)
    if len(compatible) > 1:
        names = ", ".join(sorted(item.selector for item in compatible))
        raise BackendError(
            f"multiple exact GPU backends match visible devices: {names}; "
            f"set {BACKEND_ENVIRONMENT} explicitly"
        )
    if compatible:
        return compatible[0]
    return _preferred_cpu_backend(descriptors, frontend_version)


def load_backend(frontend_version: str, request: str | None = None):
    """Select, import, validate, and permanently bind one native module."""

    global _native_module, _selected
    if _native_module is not None:
        if request not in (None, "auto", _selected.selector):
            raise BackendError(
                f"backend {_selected.selector!r} is already loaded; native backends "
                "cannot be switched in one process"
            )
        return _native_module

    descriptor = select_backend(frontend_version, request)
    module_name = f"{descriptor.package}.{descriptor.module}"
    try:
        native = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if descriptor.backend != "cpu" or error.name != module_name:
            raise
        # One-release migration fallback for an existing monolithic install.
        descriptor = BackendDescriptor(
            **{
                **descriptor.to_dict(),
                "module": "symmetrix",
                "selector": "cpu-legacy",
            }
        )
        native = importlib.import_module("symmetrix.symmetrix")
        if getattr(native, "__symmetrix_proxy__", False):
            raise BackendError(
                "the bundled CPU extension is missing; reinstall symmetrix or "
                "build the CPU frontend from source"
            ) from error

    build_info = getattr(native, "_backend_build_info", lambda: None)()
    if build_info is not None:
        if int(build_info.get("native_abi", -1)) != descriptor.native_abi:
            raise BackendError(
                f"loaded module {module_name} reports incompatible native ABI"
            )
        if build_info.get("backend") != descriptor.backend:
            raise BackendError(
                f"loaded module {module_name} reports backend "
                f"{build_info.get('backend')!r}, expected {descriptor.backend!r}"
            )
        if build_info.get("distribution") != descriptor.distribution:
            raise BackendError(
                f"loaded module {module_name} reports distribution "
                f"{build_info.get('distribution')!r}, expected "
                f"{descriptor.distribution!r}"
            )
        built_architecture = (
            str(build_info.get("architecture", "")).lower().replace("_", "")
        )
        expected_architecture = descriptor.architecture.lower().replace("_", "")
        if (
            descriptor.backend in {"cuda", "hip"}
            or descriptor.descriptor_path != "bundled"
        ) and built_architecture != expected_architecture:
            raise BackendError(
                f"loaded module {module_name} reports architecture "
                f"{built_architecture!r}, expected {expected_architecture!r}"
            )
    _selected = descriptor
    _native_module = native
    sys.modules["symmetrix.symmetrix"] = native
    return native


def selected_backend() -> dict[str, Any] | None:
    return None if _selected is None else _selected.to_dict()


def backend_inventory(frontend_version: str) -> list[dict[str, Any]]:
    """Describe installed backends and whether this host can select each one.

    This reads descriptors and probes device names only. It intentionally never
    imports a native extension, so it is safe for command-line inspection.
    """
    visible = _visible_targets()
    descriptors = discover_backends(frontend_version)
    values: list[dict[str, Any]] = []
    usable_accelerators = 0
    for descriptor in descriptors:
        value = descriptor.to_dict()
        try:
            _compatible(descriptor, frontend_version)
        except BackendError as error:
            availability = {"status": "incompatible", "reason": str(error)}
        else:
            if descriptor.backend in {"cuda", "hip"}:
                target = descriptor.architecture.lower().replace("_", "")
                detected = sorted(visible[descriptor.backend])
                if target in visible[descriptor.backend]:
                    usable_accelerators += 1
                    availability = {
                        "status": "usable",
                        "reason": f"exact {descriptor.backend} target {target} is visible",
                    }
                else:
                    availability = {
                        "status": "incompatible",
                        "reason": (
                            f"requires {target}; visible {descriptor.backend} targets: "
                            + (", ".join(detected) or "none")
                        ),
                    }
            else:
                availability = {"status": "usable", "reason": "CPU backend"}
        value["availability"] = availability
        values.append(value)

    if usable_accelerators:
        for value in values:
            if (
                value["backend"] == "cpu"
                and value["availability"]["status"] == "usable"
            ):
                value["availability"] = {
                    "status": "fallback",
                    "reason": "usable when explicitly selected or no GPU backend is selected",
                }
        return values
    try:
        preferred = _preferred_cpu_backend(descriptors, frontend_version)
    except BackendError:
        return values
    for value in values:
        if value["backend"] != "cpu" or value["availability"]["status"] != "usable":
            continue
        if value["selector"] == preferred.selector:
            value["availability"] = {
                "status": "usable",
                "reason": "preferred CPU backend for this host",
            }
        else:
            value["availability"] = {
                "status": "fallback",
                "reason": "usable when explicitly selected",
            }
    return values


def available_backends(frontend_version: str) -> list[dict[str, Any]]:
    return [item.to_dict() for item in discover_backends(frontend_version)]
