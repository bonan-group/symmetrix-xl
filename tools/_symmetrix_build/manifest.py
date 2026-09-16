"""Versioned build target manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .command import BuildError
from .targets import Architecture

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Toolchain:
    cxx: str
    cxx_version: str
    host_cxx: str
    cmake: str
    cmake_version: str
    generator: str
    fortran: str = ""
    toolkit_root: str = ""
    toolkit_version: str = ""
    runtime_library_dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetManifest:
    backend: str
    architecture: Architecture
    toolchain: Toolchain
    python_executable: str
    python_abi: str
    host_target: str
    blas_policy: str
    blas_root: str
    device_probe: str
    visible_devices: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TargetManifest:
        if value.get("schema_version") != SCHEMA_VERSION:
            raise BuildError(
                "unsupported target manifest schema_version "
                f"{value.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        try:
            architecture = Architecture(**value["architecture"])
            toolchain_value = dict(value["toolchain"])
            toolchain_value["runtime_library_dirs"] = tuple(
                toolchain_value.get("runtime_library_dirs", ())
            )
            toolchain = Toolchain(**toolchain_value)
            visible = tuple(value.get("visible_devices", ()))
            return cls(
                backend=value["backend"],
                architecture=architecture,
                toolchain=toolchain,
                python_executable=value["python_executable"],
                python_abi=value["python_abi"],
                host_target=value["host_target"],
                blas_policy=value["blas_policy"],
                blas_root=value.get("blas_root", ""),
                device_probe=value.get("device_probe", "manifest"),
                visible_devices=visible,
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError) as error:
            raise BuildError(f"invalid target manifest: {error}") from error

    @classmethod
    def read(cls, path: str | Path) -> TargetManifest:
        try:
            value = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise BuildError(f"cannot read target manifest {path}: {error}") from error
        if not isinstance(value, dict):
            raise BuildError("target manifest root must be a JSON object")
        return cls.from_dict(value)
