"""Release wheel matrix planning, child builds, and release index emission."""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .command import BuildError, CommandRunner, sha256_file
from .manifest import TargetManifest
from .package_identity import package_identity
from .targets import (
    BASE_CPU_TARGETS,
    CPU_ADDON_LABELS,
    CPU_TARGETS,
    normalize_cuda_target,
    normalize_hip_target,
    validate_cuda_target_for_toolkit,
)

RELEASE_INDEX_SCHEMA = 1
_CPU_LABEL_TARGETS = {label: target for target, label in CPU_ADDON_LABELS.items()}


@dataclass(frozen=True)
class MatrixSpec:
    backend: str
    toolkit_major: str
    target: str

    @property
    def normalized(self) -> str:
        if self.backend == "cpu":
            label = CPU_ADDON_LABELS.get(self.target, self.target)
            return f"cpu:{label}"
        return f"{self.backend}:{self.toolkit_major}:{self.target}"


def parse_matrix_spec(value: str) -> MatrixSpec:
    parts = [part.strip() for part in value.split(":") if part.strip()]
    if not parts:
        raise BuildError(f"empty release target spec: {value!r}")
    backend = parts[0].lower()
    if backend == "cpu":
        if len(parts) != 2:
            raise BuildError(
                f"CPU target spec must be cpu:<host-target>, not {value!r}"
            )
        target = _CPU_LABEL_TARGETS.get(parts[1].lower(), parts[1].lower())
        if target not in CPU_TARGETS:
            supported = ", ".join(sorted(set(CPU_TARGETS) | set(_CPU_LABEL_TARGETS)))
            raise BuildError(
                f"invalid CPU target {parts[1]!r} in {value!r}; expected one of: "
                + supported
            )
        if target in BASE_CPU_TARGETS:
            raise BuildError(
                f"release target {value!r} belongs to the base symmetrix-xl "
                "distribution; the wheel matrix only builds architecture "
                "add-on targets"
            )
        return MatrixSpec("cpu", "", target)
    if backend in {"cuda", "hip"}:
        if len(parts) < 3 or (backend == "cuda" and len(parts) != 3):
            raise BuildError(
                f"{backend.upper()} target spec must be "
                f"{backend}:<toolkit-major>:<device>, not {value!r}"
            )
        if not parts[1].isdigit():
            raise BuildError(
                f"{backend.upper()} toolkit segment must be a major version "
                f"number, not {parts[1]!r} in {value!r}"
            )
        target = (
            normalize_cuda_target(parts[2])
            if backend == "cuda"
            else normalize_hip_target(":".join(parts[2:]))
        )
        if backend == "cuda":
            validate_cuda_target_for_toolkit(parts[1], target)
        return MatrixSpec(backend, parts[1], target)
    raise BuildError(
        f"unsupported release backend {parts[0]!r}; expected cpu, cuda, or hip"
    )


def reject_duplicate_targets(specs: list[MatrixSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        normalized = spec.normalized
        if normalized in seen:
            raise BuildError(
                f"duplicate release target {normalized}; each normalized target "
                "may appear once per matrix run"
            )
        seen.add(normalized)


def _child_arguments(
    spec: MatrixSpec,
    cuda_root: Path | None,
    rocm_root: Path | None,
    build_root: Path | None,
    jobs: int | None,
    accelerator_blas_root: Path | None,
) -> list[str]:
    arguments = ["--backend", spec.backend]
    if spec.backend == "cpu":
        arguments.extend(("--cpu-target", spec.target))
    else:
        arguments.extend(("--arch", spec.target))
        if spec.backend == "cuda" and cuda_root is not None:
            arguments.extend(("--cuda-root", os.fspath(cuda_root)))
        if spec.backend == "hip" and rocm_root is not None:
            arguments.extend(("--rocm-root", os.fspath(rocm_root)))
    if build_root is not None:
        arguments.extend(("--build-root", os.fspath(build_root)))
    if jobs is not None:
        arguments.extend(("--jobs", str(jobs)))
    if spec.backend in {"cuda", "hip"} and accelerator_blas_root is not None:
        arguments.extend(("--accelerator-blas-root", os.fspath(accelerator_blas_root)))
    return arguments


def child_commands(
    spec: MatrixSpec,
    *,
    script: Path,
    wheel_directory: Path,
    cuda_root: Path | None = None,
    rocm_root: Path | None = None,
    build_root: Path | None = None,
    jobs: int | None = None,
    accelerator_blas_root: Path | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    detect_arguments = _child_arguments(
        spec, cuda_root, rocm_root, None, None, accelerator_blas_root
    )
    wheel_arguments = _child_arguments(
        spec, cuda_root, rocm_root, build_root, jobs, accelerator_blas_root
    )
    detect_command = (
        sys.executable,
        os.fspath(script),
        "detect",
        "--json",
        *detect_arguments,
    )
    wheel_command = (
        sys.executable,
        os.fspath(script),
        "wheel",
        *wheel_arguments,
        "--wheel-dir",
        os.fspath(wheel_directory),
    )
    return detect_command, wheel_command


def verify_matrix_target(spec: MatrixSpec, manifest: TargetManifest) -> None:
    identity = package_identity(manifest)
    resolved = manifest.host_target if spec.backend == "cpu" else identity.architecture
    if resolved != spec.target:
        raise BuildError(
            f"target {spec.normalized} resolved {resolved!r} instead of {spec.target!r}"
        )
    if spec.backend == "cpu":
        return
    recorded_major = manifest.toolchain.toolkit_version.split(".")[0]
    if recorded_major != spec.toolkit_major:
        raise BuildError(
            f"target {spec.normalized} requires toolkit major "
            f"{spec.toolkit_major}, but detection resolved "
            f"{manifest.toolchain.toolkit_version} at "
            f"{manifest.toolchain.toolkit_root}; point --cuda-root or "
            "--rocm-root at the requested toolkit generation"
        )


def verify_wheel_descriptor(
    wheel_path: Path, manifest: TargetManifest
) -> dict[str, Any]:
    identity = package_identity(manifest)
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        if identity.base:
            expected_name = "symmetrix/_backend_cpu.json"
        else:
            expected_name = f"{identity.package}/backend.json"
        matches = [name for name in names if name == expected_name]
        if len(matches) != 1:
            raise BuildError(
                f"wheel {wheel_path.name} does not ship exactly one backend "
                f"descriptor at {expected_name}"
            )
        descriptor = json.loads(archive.read(matches[0]))
    expected_fields = (
        ("distribution", identity.distribution),
        ("selector", identity.selector),
        ("backend", identity.backend),
        ("architecture", identity.architecture),
        ("toolkit", identity.toolkit),
        ("package", identity.package),
        ("module", identity.module),
    )
    for field, expected in expected_fields:
        actual = descriptor.get(field)
        if str(actual) != expected:
            raise BuildError(
                f"wheel {wheel_path.name} descriptor field {field!r} is "
                f"{actual!r}, expected {expected!r}"
            )
    return descriptor


def _reject_artifact_collisions(records: list[dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        for key in ("distribution", "selector", "filename"):
            value = record["wheel"]["filename"] if key == "filename" else record[key]
            previous = seen.get((key, value))
            if previous is not None:
                raise BuildError(
                    f"duplicate release {key} {value!r} claimed by targets "
                    f"{previous} and {record['target']}"
                )
            seen[(key, value)] = record["target"]


def run_wheel_matrix(
    target_values: list[str],
    *,
    repo_root: Path,
    runner: CommandRunner,
    wheel_directory: Path,
    build_root: Path | None = None,
    cuda_root: Path | None = None,
    rocm_root: Path | None = None,
    jobs: int | None = None,
    accelerator_blas_root: Path | None = None,
    dry_run: bool = False,
) -> int:
    specs = [parse_matrix_spec(value) for value in target_values]
    reject_duplicate_targets(specs)
    script = repo_root / "tools" / "symmetrix_build.py"
    wheel_dir = wheel_directory.expanduser().resolve()

    if dry_run:
        plan = []
        for index, spec in enumerate(specs):
            staging = wheel_dir / f".matrix-staging-{index:02d}-{spec.backend}"
            detect_command, wheel_command = child_commands(
                spec,
                script=script,
                wheel_directory=staging,
                cuda_root=cuda_root,
                rocm_root=rocm_root,
                build_root=build_root,
                jobs=jobs,
                accelerator_blas_root=accelerator_blas_root,
            )
            plan.append(
                {
                    "target": spec.normalized,
                    "detect": list(detect_command),
                    "wheel": list(wheel_command),
                }
            )
        print(json.dumps({"targets": plan}, indent=2))
        return 0

    wheel_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    staging_directories: list[Path] = []
    for index, spec in enumerate(specs):
        staging = wheel_dir / f".matrix-staging-{index:02d}-{spec.backend}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        staging_directories.append(staging)
        detect_command, wheel_command = child_commands(
            spec,
            script=script,
            wheel_directory=staging,
            cuda_root=cuda_root,
            rocm_root=rocm_root,
            build_root=build_root,
            jobs=jobs,
            accelerator_blas_root=accelerator_blas_root,
        )
        result = runner.run(detect_command)
        if result.returncode != 0:
            raise BuildError(
                f"detect failed for target {spec.normalized} "
                f"({result.returncode}):\n{result.output.strip()}"
            )
        try:
            manifest = TargetManifest.from_dict(json.loads(result.stdout))
        except (json.JSONDecodeError, BuildError) as error:
            raise BuildError(
                f"detect for target {spec.normalized} did not return a valid "
                f"manifest: {error}"
            ) from error
        verify_matrix_target(spec, manifest)
        identity = package_identity(manifest)
        log_path = staging / "wheel-build.log"
        print(f"[{spec.normalized}] {' '.join(wheel_command)}", flush=True)
        result = runner.run_logged(wheel_command, log_path)
        if result.returncode != 0:
            raise BuildError(
                f"wheel build failed for target {spec.normalized} "
                f"({result.returncode}); see {log_path}"
            )
        wheels = sorted(staging.glob("*.whl"))
        if len(wheels) != 1:
            raise BuildError(
                f"target {spec.normalized} produced {len(wheels)} wheels in "
                f"{staging}; expected exactly one"
            )
        wheel_path = wheels[0]
        verify_wheel_descriptor(wheel_path, manifest)
        destination = wheel_dir / wheel_path.name
        if destination.exists():
            raise BuildError(
                f"release wheel {destination.name} already exists in {wheel_dir}"
            )
        records.append(
            {
                "target": spec.normalized,
                "distribution": identity.distribution,
                "selector": identity.selector,
                "backend": identity.backend,
                "architecture": identity.architecture,
                "toolkit": identity.toolkit,
                "kokkos_trait": manifest.architecture.kokkos_trait,
                "python_abi": manifest.python_abi,
                "wheel": {
                    "filename": wheel_path.name,
                    "sha256": sha256_file(wheel_path),
                    "size": wheel_path.stat().st_size,
                },
            }
        )
        shutil.move(os.fspath(wheel_path), os.fspath(destination))
    _reject_artifact_collisions(records)
    index = {"schema_version": RELEASE_INDEX_SCHEMA, "targets": records}
    index_path = wheel_dir / "release-index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(index, indent=2, sort_keys=True))
    for staging in staging_directories:
        shutil.rmtree(staging, ignore_errors=True)
    return 0
