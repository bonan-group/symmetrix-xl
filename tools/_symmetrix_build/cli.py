"""Command-line interface for the standalone Symmetrix build frontend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import lammps_build as build_lammps
from .command import BuildError, CommandRunner
from .detect import (
    DetectionRequest,
    DeviceProbe,
    detect_target,
    validate_cpu_prerequisites,
    validate_source_checkout,
)
from .lammps_build import (
    lammps_build_invocation,
    run_lammps_build,
    verify_lammps,
)
from .manifest import TargetManifest
from .matrix import run_wheel_matrix
from .package_identity import package_identity
from .python_build import (
    python_build_invocation,
    run_python_build,
    verify_frontend_installed,
    verify_installed_backend,
    verify_python_cmake_provenance,
)
from .sdist_build import run_sdist_command


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "hip"))
    parser.add_argument("--arch")
    parser.add_argument(
        "--cpu-target", choices=("native", "x86-64-v3", "x86-64-v4", "none")
    )
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--rocm-root", type=Path)
    parser.add_argument("--cxx", type=Path)
    parser.add_argument("--host-cxx", type=Path)
    parser.add_argument("--generator")
    parser.add_argument("--blas", choices=("auto", "openblas", "mkl", "mkl-sequential"))
    parser.add_argument("--mkl-root", type=Path)
    parser.add_argument(
        "--accelerator-blas-root",
        type=Path,
        help="Static serial OpenBLAS prefix for a self-contained accelerator wheel.",
    )
    parser.add_argument("--target-manifest", type=Path)


def _base_request(
    args: argparse.Namespace,
) -> tuple[DetectionRequest, DeviceProbe | None]:
    if args.target_manifest is None:
        return (
            DetectionRequest(
                backend=args.backend or "auto",
                arch=args.arch or "auto",
                cpu_target=args.cpu_target or "native",
                cuda_root=os.fspath(args.cuda_root) if args.cuda_root else "",
                rocm_root=os.fspath(args.rocm_root) if args.rocm_root else "",
                cxx=os.fspath(args.cxx) if args.cxx else "",
                host_cxx=os.fspath(args.host_cxx) if args.host_cxx else "",
                generator=args.generator or "",
                blas=args.blas or "auto",
                mkl_root=os.fspath(args.mkl_root) if args.mkl_root else "",
                accelerator_blas_root=(
                    os.fspath(getattr(args, "accelerator_blas_root", None))
                    if getattr(args, "accelerator_blas_root", None)
                    else ""
                ),
            ),
            None,
        )

    base = TargetManifest.read(args.target_manifest)
    backend = args.backend or base.backend
    same_backend = backend == base.backend
    arch = args.arch or (base.architecture.requested if same_backend else "auto")
    cuda_root = args.cuda_root
    rocm_root = args.rocm_root
    if not cuda_root and same_backend and backend == "cuda":
        cuda_root = Path(base.toolchain.toolkit_root)
    if not rocm_root and same_backend and backend == "hip":
        rocm_root = Path(base.toolchain.toolkit_root)
    request = DetectionRequest(
        backend=backend,
        arch=arch,
        cpu_target=args.cpu_target or (base.host_target if same_backend else "native"),
        cuda_root=os.fspath(cuda_root) if cuda_root else "",
        rocm_root=os.fspath(rocm_root) if rocm_root else "",
        cxx=(
            os.fspath(args.cxx)
            if args.cxx
            else (base.toolchain.cxx if same_backend else "")
        ),
        host_cxx=(
            os.fspath(args.host_cxx)
            if args.host_cxx
            else (base.toolchain.host_cxx if same_backend else "")
        ),
        generator=args.generator or base.toolchain.generator,
        blas=args.blas
        or (
            "mkl-sequential"
            if base.blas_policy == "mkl_sequential"
            else "mkl"
            if base.blas_policy == "mkl_gnu_thread"
            else "openblas"
            if base.blas_policy == "openblas"
            else "auto"
        ),
        mkl_root=(
            os.fspath(args.mkl_root)
            if args.mkl_root
            else (base.blas_root if same_backend else "")
        ),
        accelerator_blas_root=(
            os.fspath(getattr(args, "accelerator_blas_root", None))
            if getattr(args, "accelerator_blas_root", None)
            else (
                base.blas_root
                if same_backend and base.blas_policy == "openblas_static"
                else ""
            )
        ),
    )
    if not same_backend:
        devices = None
    elif backend == "cuda":
        devices = DeviceProbe(
            (base.architecture.device_target,), (), base.device_probe, ""
        )
    elif backend == "hip":
        devices = DeviceProbe(
            (), (base.architecture.device_target,), "", base.device_probe
        )
    else:
        devices = DeviceProbe((), ())
    return request, devices


def _resolve_manifest(
    args: argparse.Namespace, root: Path, runner: CommandRunner
) -> TargetManifest:
    request, devices = _base_request(args)
    return detect_target(request, repo_root=root, runner=runner, devices=devices)


def _preflight(manifest: TargetManifest, root: Path, runner: CommandRunner) -> None:
    validate_source_checkout(root, runner)
    if manifest.backend == "cpu":
        validate_cpu_prerequisites(runner)


def _human_manifest(manifest: TargetManifest) -> str:
    lines = [
        f"Symmetrix target: {manifest.backend}/{manifest.architecture.device_target}",
        f"Configuration fingerprint: {manifest.fingerprint}",
        f"C++ compiler: {manifest.toolchain.cxx}",
        f"C++ identity: {manifest.toolchain.cxx_version}",
        f"CMake: {manifest.toolchain.cmake} ({manifest.toolchain.cmake_version})",
        f"Generator: {manifest.toolchain.generator}",
    ]
    if manifest.toolchain.toolkit_root:
        lines.append(
            "Toolkit: "
            f"{manifest.toolchain.toolkit_root} ({manifest.toolchain.toolkit_version})"
        )
        lines.append(
            "Runtime libraries: "
            + (", ".join(manifest.toolchain.runtime_library_dirs) or "system paths")
        )
    if manifest.backend in {"cuda", "hip"}:
        lines.extend(
            (
                f"Kokkos trait: {manifest.architecture.kokkos_trait}",
                f"Compiler target: {manifest.architecture.compiler_target}",
                "Visible targets: "
                + (", ".join(manifest.visible_devices) or "none (explicit target)"),
                f"Host BLAS policy: {manifest.blas_policy}",
            )
        )
        if manifest.blas_root:
            lines.append(f"Host BLAS root: {manifest.blas_root}")
    else:
        lines.append(f"Host target: {manifest.host_target}")
        lines.append(f"BLAS policy: {manifest.blas_policy}")
    return "\n".join(lines)


def _print_invocation(command: tuple[str, ...], environment: dict[str, str]) -> None:
    selected_environment = {
        name: environment[name]
        for name in (
            "CXX",
            "CMAKE_BUILD_PARALLEL_LEVEL",
            "CMAKE_GENERATOR",
            "CMAKE_PREFIX_PATH",
            "NVCC_WRAPPER_DEFAULT_COMPILER",
        )
        if name in environment
    }
    print(
        json.dumps({"environment": selected_environment, "command": command}, indent=2)
    )


def _print_lammps_invocation(invocation: build_lammps.LammpsInvocation) -> None:
    selected_environment = {
        name: invocation.environment[name]
        for name in (
            "CXX",
            "CMAKE_BUILD_PARALLEL_LEVEL",
            "CMAKE_GENERATOR",
            "CMAKE_PREFIX_PATH",
            "NVCC_WRAPPER_DEFAULT_COMPILER",
        )
        if name in invocation.environment
    }
    print(
        json.dumps(
            {
                "build_directory": str(invocation.build_directory),
                "build_fingerprint": invocation.build_fingerprint,
                "commands": invocation.commands,
                "environment": selected_environment,
                "provenance": invocation.provenance,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_qualification(build_directory: Path, value: dict[str, object]) -> None:
    (build_directory / "qualification.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def _doctor_environment(manifest: TargetManifest) -> dict[str, str]:
    environment = dict(os.environ)
    environment["SYMMETRIX_BACKEND"] = package_identity(manifest).selector
    if manifest.backend == "cpu":
        # An unbound team can migrate between the sentinel's per-worker CPU
        # samples and falsely look like an undersubscribed OpenMP runtime.
        environment.setdefault("OMP_PROC_BIND", "spread")
        environment.setdefault("OMP_PLACES", "threads")
    return environment


def _runtime_qualification_reason(manifest: TargetManifest) -> str | None:
    """Explain why this host cannot qualify an accelerator target locally."""
    if manifest.backend not in {"cuda", "hip"}:
        return None
    target = manifest.architecture.device_target
    if target in manifest.visible_devices:
        return None
    visible = ", ".join(manifest.visible_devices) or "none"
    return (
        f"built {manifest.backend} target {target}, but visible "
        f"{manifest.backend} targets are {visible}; runtime qualification must "
        "run on a matching host"
    )


def _default_python_build_jobs(backend: str) -> int:
    if backend in {"cuda", "hip"}:
        return min(16, max(1, (os.cpu_count() or 1) // 2))
    return os.cpu_count() or 1


def _add_python_build_arguments(parser: argparse.ArgumentParser) -> None:
    _add_target_arguments(parser)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument(
        "--jobs",
        type=int,
        help="Parallel compiler processes (default: CPU count for CPU; half, capped at 16, for GPU).",
    )
    parser.add_argument("--dry-run", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symmetrix_build.py",
        description="Detect and build a backend-specific Symmetrix target.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    detect = commands.add_parser("detect", help="Resolve and print a target manifest.")
    _add_target_arguments(detect)
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--output", type=Path)

    preflight = commands.add_parser(
        "preflight", help="Validate a target without starting compilation."
    )
    _add_target_arguments(preflight)
    preflight.add_argument("--json", action="store_true")
    preflight.add_argument("--output", type=Path)

    install = commands.add_parser(
        "install", help="Build and install into the active Python environment."
    )
    _add_python_build_arguments(install)

    wheel = commands.add_parser("wheel", help="Build one backend-specific wheel.")
    _add_python_build_arguments(wheel)
    wheel.add_argument("--wheel-dir", type=Path, required=True)

    matrix = commands.add_parser(
        "wheel-matrix",
        help="Build several release wheels in isolated child processes.",
    )
    matrix.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="SPEC",
        help=(
            "Release target such as cpu:x86-64-v3, cpu:x86-64-v4, "
            "cuda:13:sm120, or hip:7:gfx1151; repeat for each target."
        ),
    )
    matrix.add_argument("--wheel-dir", type=Path, required=True)
    matrix.add_argument("--build-root", type=Path)
    matrix.add_argument("--cuda-root", type=Path)
    matrix.add_argument("--rocm-root", type=Path)
    matrix.add_argument("--accelerator-blas-root", type=Path)
    matrix.add_argument(
        "--jobs",
        type=int,
        help="Parallel compiler processes forwarded to every child build.",
    )
    matrix.add_argument("--dry-run", action="store_true")

    sdist = commands.add_parser(
        "sdist",
        help="Build the self-contained source distribution at pinned submodule commits.",
    )
    sdist.add_argument("--output-dir", type=Path, required=True)
    sdist.add_argument("--staging-root", type=Path)
    sdist.add_argument("--allow-dirty", action="store_true")
    sdist.add_argument("--dry-run", action="store_true")

    lammps = commands.add_parser(
        "lammps", help="Integrate and build pair_symmetrix with LAMMPS."
    )
    _add_python_build_arguments(lammps)
    lammps.add_argument("--source", type=Path, required=True)
    lammps.add_argument("--prefix", type=Path, required=True)
    lammps.add_argument("--mpi", choices=("auto", "on", "off"), default="auto")
    lammps.add_argument("--mpi-cxx", type=Path)
    lammps.add_argument(
        "--lammps-package",
        action="append",
        default=[],
        metavar="NAME",
        help="Enable a LAMMPS package such as KSPACE or EXTRA-PAIR; repeatable.",
    )
    lammps.add_argument(
        "--lammps-cmake-define",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Add an unmanaged LAMMPS CMake definition; repeatable.",
    )
    lammps.add_argument(
        "--require-gpu-aware-mpi",
        action="store_true",
        help=(
            "Require the selected MPI to pass the CUDA/ROCm capability query; "
            "implies --mpi on."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = repository_root()
    runner = CommandRunner()
    try:
        if args.command == "wheel-matrix":
            return run_wheel_matrix(
                [value for value in args.target],
                repo_root=root,
                runner=runner,
                wheel_directory=args.wheel_dir,
                build_root=args.build_root,
                cuda_root=args.cuda_root,
                rocm_root=args.rocm_root,
                accelerator_blas_root=args.accelerator_blas_root,
                jobs=args.jobs,
                dry_run=args.dry_run,
            )
        if args.command == "sdist":
            return run_sdist_command(
                repo_root=root,
                runner=runner,
                output_directory=args.output_dir,
                staging_root=args.staging_root,
                dry_run=args.dry_run,
                allow_dirty=args.allow_dirty,
            )
        manifest = _resolve_manifest(args, root, runner)
        _preflight(manifest, root, runner)
        if args.command in {"detect", "preflight"}:
            if args.output:
                manifest.write(args.output)
            print(
                manifest.to_json() if args.json else _human_manifest(manifest), end=""
            )
            if not args.json:
                print()
            return 0

        if args.command in {"install", "wheel"}:
            if args.command == "install" and not package_identity(manifest).base:
                verify_frontend_installed(manifest, root / "symmetrix", runner)
            jobs = (
                args.jobs
                if args.jobs is not None
                else _default_python_build_jobs(manifest.backend)
            )
            invocation = python_build_invocation(
                manifest,
                repo_root=root,
                operation=args.command,
                build_root=args.build_root,
                wheel_directory=getattr(args, "wheel_dir", None),
                jobs=jobs,
            )
            if args.dry_run:
                _print_invocation(invocation.command, invocation.environment)
                return 0
            run_python_build(invocation, runner)
            build_toolchain = verify_python_cmake_provenance(
                invocation, manifest, runner
            )
            if args.command == "install":
                reason = _runtime_qualification_reason(manifest)
                if reason is None:
                    record = verify_installed_backend(manifest, runner)
                    record["runtime_qualification"] = {"status": "ok"}
                    doctor_environment = _doctor_environment(manifest)
                    doctor = runner.run(
                        (
                            manifest.python_executable,
                            "-m",
                            "symmetrix.cli.main",
                            "doctor",
                            "--json",
                        ),
                        check=True,
                        env=doctor_environment,
                    )
                    record["doctor"] = json.loads(doctor.stdout)
                    record["doctor_environment"] = {
                        name: doctor_environment[name]
                        for name in ("OMP_PROC_BIND", "OMP_PLACES")
                        if name in doctor_environment
                    }
                else:
                    identity = package_identity(manifest)
                    record = {
                        "selected_backend": {
                            "selector": identity.selector,
                            "distribution": identity.distribution,
                        },
                        "runtime_qualification": {
                            "status": "not_run",
                            "reason": reason,
                            "backend": manifest.backend,
                            "expected_architecture": manifest.architecture.device_target,
                            "visible_architectures": list(manifest.visible_devices),
                        },
                    }
                record["build_toolchain"] = build_toolchain
                _write_qualification(invocation.build_directory, record)
                print(json.dumps(record, indent=2, sort_keys=True))
            else:
                print(f"Wheel written to {args.wheel_dir.resolve()}")
            return 0

        if args.command == "lammps":
            invocation_arguments = {
                "repo_root": root,
                "source": args.source,
                "prefix": args.prefix,
                "mpi": args.mpi,
                "mpi_cxx": os.fspath(args.mpi_cxx) if args.mpi_cxx else "",
                "require_gpu_aware_mpi": args.require_gpu_aware_mpi,
                "lammps_packages": tuple(args.lammps_package),
                "extra_cmake_definitions": tuple(args.lammps_cmake_define),
                "jobs": args.jobs if args.jobs is not None else (os.cpu_count() or 1),
                "runner": runner,
                "build_root": args.build_root,
            }
            invocation = lammps_build_invocation(
                manifest, **invocation_arguments, materialize=not args.dry_run
            )
            if args.dry_run:
                _print_lammps_invocation(invocation)
                return 0
            record = build_lammps.reuse_lammps_qualification(invocation, runner)
            if record is not None:
                print(json.dumps(record, indent=2, sort_keys=True))
                return 0
            if invocation.reusable_qualification:
                invocation = lammps_build_invocation(
                    manifest, **invocation_arguments, force_new=True
                )
            run_lammps_build(invocation, runner)
            record = verify_lammps(invocation, runner)
            _write_qualification(invocation.build_directory, record)
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command}")
