import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser


_BACKEND_SELECTOR = re.compile(
    r"(?:cpu(?:-[a-z0-9]+)?|cuda\d+-sm\d+|rocm\d+-gfx[0-9a-f]+)\Z"
)


class _NoAutomaticBackend(RuntimeError):
    """Automatic installation found no accelerator that needs a wheel."""


def openmp_runtime_diagnostics(*args, **kwargs):
    """Import native runtime diagnostics only for the doctor command."""

    from ..runtime_diagnostics import openmp_runtime_diagnostics as diagnose

    return diagnose(*args, **kwargs)


def accelerator_runtime_diagnostics(*args, **kwargs):
    """Import accelerator runtime diagnostics only for the doctor command."""

    from ..runtime_diagnostics import accelerator_runtime_diagnostics as diagnose

    return diagnose(*args, **kwargs)


def _human_report(report):
    runtime = report.get("runtime") or {}
    backend = report.get("backend") or {}
    if backend:
        print(
            "Selected backend: "
            f"{backend.get('selector')} ({backend.get('backend')} "
            f"{backend.get('architecture')})"
        )
    accelerator = report.get("schema") == "symmetrix.accelerator-runtime-diagnostics"
    title = "accelerator" if accelerator else "OpenMP runtime"
    print(f"Symmetrix {title} diagnostic: {report['status']}")
    print(f"Python: {report['python_executable']}")
    native = report.get("native") or {}
    if native.get("extension"):
        print(f"Native extension: {native['extension']}")
    build = native.get("build_info") or {}
    if build:
        print(
            "Native build: "
            f"{build.get('compiler_id')} {build.get('compiler_version')}, "
            f"device compiler {build.get('device_compiler_id') or 'none'} "
            f"{build.get('device_compiler_version') or ''}".rstrip()
        )
    if runtime:
        if accelerator:
            print(f"Kokkos execution space: {runtime.get('execution_space')}")
            environment = runtime.get("device_environment") or {}
            print(
                "Device: "
                f"{environment.get('backend', 'unknown')} "
                f"{environment.get('architecture', 'unknown')}"
            )
        else:
            print(
                "Build: "
                f"{runtime.get('compiler_id')} {runtime.get('compiler_version')}, "
                f"OpenMP {runtime.get('compiled_openmp')}, "
                f"Kokkos {runtime.get('kokkos_execution_space')}"
            )
            print(
                f"Expected runtime: {runtime.get('expected_runtime_path') or 'unavailable'}"
            )
            print(
                f"Loaded runtime: {runtime.get('loaded_runtime_path') or 'unavailable'}"
            )
            print(f"OMP max threads: {runtime.get('omp_max_threads')}")
    for check in report["checks"]:
        print(f"[{check['status'].upper()}] {check['name']}: {check['message']}")
    for message in report.get("remediation", ()):
        print(f"Remediation: {message}")


def _toolkit_resolution(kind):
    override = os.environ.get(
        "SYMMETRIX_CUDA_MAJOR" if kind == "cuda" else "SYMMETRIX_ROCM_MAJOR"
    )
    if override and override.isdigit():
        return override, "environment override"

    if kind == "cuda":
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None:
            match = re.search(r"CUDA Version:\s*(\d+)", result.stdout)
            if match:
                return match.group(1), "nvidia-smi CUDA Version"

    commands = (("nvcc", "--version"),) if kind == "cuda" else (("hipcc", "--version"),)
    if kind == "hip":
        commands += (("/opt/rocm/bin/hipcc", "--version"),)
    for command in commands:
        executable = (
            command[0] if os.path.isabs(command[0]) else shutil.which(command[0])
        )
        if executable is None:
            continue
        try:
            result = subprocess.run(
                [executable, *command[1:]],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = f"{result.stdout}\n{result.stderr}"
        match = re.search(
            r"(?:release|HIP version\s*:|ROCm\s*:?)\s*(\d+)",
            output,
            re.IGNORECASE,
        )
        if match:
            return match.group(1), f"{executable} --version"

    return None, "unavailable"


def _automatic_backend_resolution():
    from .. import backend_loader

    visible = backend_loader._visible_targets()
    cuda = sorted(visible["cuda"])
    hip = sorted(visible["hip"])
    if cuda and hip:
        raise RuntimeError(
            "both NVIDIA and AMD GPUs are visible; pass --arch explicitly"
        )
    if cuda:
        if len(cuda) != 1:
            raise RuntimeError(
                "multiple CUDA architectures are visible; pass --arch explicitly"
            )
        major, source = _toolkit_resolution("cuda")
        if major is None:
            raise RuntimeError(
                "cannot determine the CUDA major version; pass --arch "
                f"cuda12-{cuda[0]} or cuda13-{cuda[0]}"
            )
        selector = f"cuda{major}-{cuda[0]}"
        return selector, {
            "visible": f"CUDA architectures: {', '.join(cuda)}",
            "toolkit": f"CUDA major: {major} (source: {source})",
        }
    if hip:
        if len(hip) != 1:
            raise RuntimeError(
                "multiple HIP architectures are visible; pass --arch explicitly"
            )
        major, source = _toolkit_resolution("hip")
        if major is None:
            raise RuntimeError(
                "cannot determine the ROCm major version; pass --arch explicitly"
            )
        selector = f"rocm{major}-{hip[0]}"
        return selector, {
            "visible": f"HIP architectures: {', '.join(hip)}",
            "toolkit": f"ROCm major: {major} (source: {source})",
        }
    raise _NoAutomaticBackend(
        "no visible CUDA or HIP device; the bundled CPU backend is already "
        "available and needs no download"
    )


def _automatic_backend_selector():
    return _automatic_backend_resolution()[0]


def _package_manager():
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable], "uv"
    return [sys.executable, "-m", "pip", "install"], f"{sys.executable} -m pip"


def _published_package(command, distribution):
    try:
        result = subprocess.run(
            [*command, "--dry-run", distribution],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"cannot run the package manager: {error}") from error
    return result.returncode == 0


def _cuda_auto_candidates(selector):
    match = re.fullmatch(r"cuda(\d+)-(sm\d+)", selector)
    if match is None:
        return (selector,)
    detected_major = int(match.group(1))
    target = match.group(2)
    majors = range(min(detected_major, 13), 11, -1)
    candidates = tuple(f"cuda{major}-{target}" for major in majors)
    return candidates or (selector,)


def _source_build_hint(selector):
    if selector.startswith("cuda"):
        _, target = selector.split("-", 1)
        return (
            "python tools/symmetrix_build.py install --backend cuda "
            f"--arch {target} --cuda-root /path/to/cuda"
        )
    if selector.startswith("rocm"):
        _, target = selector.split("-", 1)
        return (
            "python tools/symmetrix_build.py install --backend hip "
            f"--arch {target} --rocm-root /opt/rocm"
        )
    return "python tools/symmetrix_build.py install --backend cpu --cpu-target native"


def _unavailable_package_message(candidates):
    tried = ", ".join(candidates)
    return (
        "no pre-compiled backend wheel was resolved from the configured package "
        f"index (tried: {tried}). The CPU backend remains available. Build the "
        f"backend from source, for example: {_source_build_hint(candidates[0])}"
    )


def _version_report(*, probe=False):
    """Describe the imported frontend and installed backend identities.

    This intentionally reads package metadata and backend descriptors only.  A
    version check must remain useful when the native extension cannot be loaded
    (for example, on a host with an incompatible CUDA runtime).
    """

    from .. import __version__
    from .. import backend_loader

    try:
        distribution_version = importlib.metadata.version("symmetrix-xl")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    package = importlib.import_module("symmetrix")
    report = {
        "version": __version__,
        "distribution": "symmetrix-xl",
        "distribution_version": distribution_version,
        "package": getattr(package, "__file__", ""),
        "python_executable": sys.executable,
        "backends": backend_loader.available_backends(__version__),
    }
    if probe:
        native = package.load_backend()
        selected = package.selected_backend()
        build_info_query = getattr(native, "_backend_build_info", None)
        runtime_query = getattr(native, "_openmp_runtime_info", None)
        device_query = getattr(native, "_execution_device_execution_environment", None)
        initialized_here = False
        is_initialized = getattr(native, "_kokkos_is_initialized", None)
        initialize = getattr(native, "_init_kokkos", None)
        finalize = getattr(native, "_finalize_kokkos", None)
        try:
            if (
                selected
                and selected.get("backend") in {"cuda", "hip"}
                and callable(initialize)
                and callable(is_initialized)
                and not is_initialized()
            ):
                initialize()
                initialized_here = True
            runtime = runtime_query() if callable(runtime_query) else None
            linked_libraries = []
            if isinstance(runtime, dict):
                linked_libraries.extend(runtime.get("loaded_openmp_libraries") or ())
                linked_libraries.append(runtime.get("loaded_runtime_path"))
                host_blas = runtime.get("host_blas") or {}
                linked_libraries.extend(
                    host_blas.get(key)
                    for key in ("library_path", "threading_library_path")
                )
            report["probe"] = {
                "backend": selected,
                "extension": getattr(native, "__file__", ""),
                "build_info": build_info_query() if callable(build_info_query) else {},
                "runtime": runtime,
                "device_environment": (
                    device_query() if callable(device_query) else None
                ),
                "linked_libraries": sorted(
                    {str(path) for path in linked_libraries if path}
                ),
            }
        finally:
            if initialized_here and callable(finalize):
                finalize()
    return report


def _human_version_report(report):
    distribution_version = report.get("distribution_version") or "unavailable"
    print(f"Symmetrix frontend: {report['version']}")
    print(f"Distribution: {report['distribution']} {distribution_version}")
    print(f"Python: {report['python_executable']}")
    print(f"Package: {report['package'] or 'unavailable'}")
    if report.get("probe_error"):
        print(
            "Native probe unavailable; showing metadata-only inspection: "
            f"{report['probe_error']}"
        )
    print("Installed backends:")
    for backend in report["backends"]:
        print(
            f"  {backend['selector']}: {backend['backend']} "
            f"{backend['architecture']} ({backend['distribution']})"
        )
        print(f"    descriptor: {backend.get('descriptor_path') or 'bundled'}")
        print(f"    source commit: {backend.get('source_commit') or 'unknown'}")
        print(
            "    source content sha256: "
            f"{backend.get('source_content_sha256') or 'unknown'}"
        )
        print(
            "    native source content sha256: "
            f"{backend.get('native_source_content_sha256') or 'unknown'}"
        )
        dirty = backend.get("source_dirty")
        print(f"    source dirty: {'unknown' if dirty is None else dirty}")
    probe = report.get("probe") or {}
    if probe:
        backend = probe.get("backend") or {}
        print(
            "Probed backend: "
            f"{backend.get('selector', 'unknown')} "
            f"({backend.get('backend', 'unknown')} "
            f"{backend.get('architecture', 'unknown')})"
        )
        print(f"Native extension: {probe.get('extension') or 'unavailable'}")
        build = probe.get("build_info") or {}
        if build:
            print(
                "Native build: "
                f"{build.get('compiler_id', 'unknown')} "
                f"{build.get('compiler_version', 'unknown')}"
            )
            print(f"  source commit: {build.get('source_commit') or 'unknown'}")
            dirty = build.get("source_dirty")
            print(f"  source dirty: {'unknown' if dirty is None else dirty}")
            print(
                "  source content sha256: "
                f"{build.get('source_content_sha256') or 'unknown'}"
            )
            print(
                "  native source content sha256: "
                f"{build.get('native_source_content_sha256') or 'unknown'}"
            )
        print("Linked libraries:")
        for library in probe.get("linked_libraries") or ("unavailable",):
            print(f"  {library}")
        runtime = probe.get("runtime") or {}
        if runtime:
            loaded_runtime = runtime.get("loaded_runtime_path")
            if loaded_runtime:
                print(f"Loaded OpenMP runtime: {loaded_runtime}")
            print(
                "Kokkos execution space: "
                f"{runtime.get('kokkos_execution_space') or 'unknown'}"
            )
        device = probe.get("device_environment") or {}
        if device:
            print(
                "Device: "
                f"{device.get('backend', 'unknown')} "
                f"{device.get('architecture', 'unknown')}"
            )
            for key in ("device_name", "runtime_version", "driver_version"):
                value = device.get(key)
                if value not in (None, "", "unknown"):
                    print(f"  {key.replace('_', ' ').title()}: {value}")


def _install_backend(request):
    requested = request.strip().lower()
    resolution = {}
    if requested == "auto":
        try:
            selector, resolution = _automatic_backend_resolution()
        except _NoAutomaticBackend as error:
            print(f"Backend resolution: {error}")
            print("No backend download is required for this host.")
            print("Use `symmetrix backend list` to inspect the bundled CPU backend.")
            print(
                "For a headless GPU host, rerun with `--arch`, for example "
                "`--arch cuda13-sm120`."
            )
            return 0
    else:
        selector = requested
    if not _BACKEND_SELECTOR.fullmatch(selector):
        raise RuntimeError(
            f"invalid backend selector {selector!r}; expected cuda13-sm120, "
            "rocm6-gfx1151, cpu, or auto"
        )
    command, manager = _package_manager()
    candidates = _cuda_auto_candidates(selector) if requested == "auto" else (selector,)
    selected_candidate = None
    for candidate in candidates:
        candidate_distribution = (
            "symmetrix-xl" if candidate == "cpu" else f"symmetrix-xl-{candidate}"
        )
        if _published_package(command, candidate_distribution):
            selected_candidate = candidate
            break
    if selected_candidate is None:
        raise RuntimeError(_unavailable_package_message(candidates))
    if selected_candidate != selector:
        resolution["fallback"] = (
            f"CUDA {selector.split('-', 1)[0][4:]} package unavailable; "
            f"using published fallback {selected_candidate}"
        )
    selector = selected_candidate
    distribution = "symmetrix-xl" if selector == "cpu" else f"symmetrix-xl-{selector}"
    command = [*command, distribution]
    resolution["published"] = f"published candidate: {distribution}"
    print("Backend resolution:")
    print(f"  requested: {requested}")
    for detail in resolution.values():
        print(f"  {detail}")
    print(f"  selected selector: {selector}")
    print(f"  distribution: {distribution}")
    print(f"  Python: {sys.executable}")
    print(f"  package manager: {manager}")
    print(f"Installing {distribution} with {manager}:")
    print("  " + " ".join(command))
    subprocess.run(command, check=True)
    print(
        f"Installed {distribution}; verify it with `symmetrix backend show "
        f"{selector} --probe`."
    )
    return 0


def main(argv=None):
    command_argv = sys.argv[1:] if argv is None else list(argv)
    if command_argv and command_argv[0] == "bench":
        from .bench import main as bench_main

        return bench_main(command_argv[1:])
    parser = ArgumentParser(prog="symmetrix")
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser(
        "version",
        help="Print the installed frontend and backend source identities.",
    )
    version.add_argument("--json", action="store_true", help="Emit structured JSON.")
    version.add_argument(
        "--probe",
        dest="probe",
        action="store_true",
        default=True,
        help="Load the selected backend and include native runtime details (default).",
    )
    version.add_argument(
        "--metadata-only",
        dest="probe",
        action="store_false",
        help="Inspect descriptors without loading a native backend.",
    )
    doctor = commands.add_parser(
        "doctor",
        help="Validate the selected CPU/OpenMP or CUDA/HIP native backend.",
    )
    doctor.add_argument("--json", action="store_true", help="Emit structured JSON.")
    doctor.add_argument(
        "--advisory",
        action="store_true",
        help="Warn instead of failing only when the runtime hash differs.",
    )
    backend = commands.add_parser(
        "backend", help="Inspect installed native backend packages."
    )
    backend_commands = backend.add_subparsers(dest="backend_command", required=True)
    backend_list = backend_commands.add_parser(
        "list", help="List descriptors without loading native extensions."
    )
    backend_list.add_argument("--json", action="store_true")
    backend_show = backend_commands.add_parser(
        "show", help="Show the backend that automatic or explicit selection resolves."
    )
    backend_show.add_argument(
        "selector",
        nargs="?",
        help="Optional explicit selector, for example cuda13-sm80.",
    )
    backend_show.add_argument("--json", action="store_true")
    backend_show.add_argument(
        "--probe",
        action="store_true",
        help="Load the selected backend and report runtime details.",
    )
    backend_install = backend_commands.add_parser(
        "install", help="Download and install a matching published backend wheel."
    )
    backend_install.add_argument(
        "--arch",
        default="auto",
        metavar="SELECTOR",
        help="Backend selector, such as cuda13-sm120; default: auto.",
    )
    bench = commands.add_parser(
        "bench", help="Run the standard SrTiO3 MACE-OMAT-0 benchmark."
    )
    bench.set_defaults(_handler="bench")
    args = parser.parse_args(argv)

    if args.command == "version":
        from ..backend_loader import BackendError

        try:
            report = _version_report(probe=args.probe)
            report["probe_status"] = "ok" if args.probe else "skipped"
        except (BackendError, ImportError, OSError, RuntimeError, ValueError) as error:
            if not args.probe:
                print(f"symmetrix version: {error}", file=sys.stderr)
                return 1
            try:
                report = _version_report(probe=False)
            except (BackendError, ImportError, OSError, RuntimeError, ValueError):
                print(f"symmetrix version: {error}", file=sys.stderr)
                return 1
            report["probe_error"] = str(error)
            report["probe_status"] = "unavailable"
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _human_version_report(report)
        return 0

    if args.command == "backend":
        from .. import __version__, load_backend
        from ..backend_loader import backend_inventory, select_backend

        if args.backend_command == "install":
            try:
                return _install_backend(args.arch)
            except (RuntimeError, subprocess.CalledProcessError) as error:
                print(f"symmetrix backend install: {error}", file=sys.stderr)
                return 1

        if args.backend_command == "list":
            values = backend_inventory(__version__)
            if args.json:
                print(json.dumps(values, indent=2, sort_keys=True))
            else:
                for value in values:
                    availability = value["availability"]
                    marker = (
                        "*" if availability["status"] in {"usable", "fallback"} else " "
                    )
                    print(
                        f"{marker} {value['selector']}: {value['backend']} "
                        f"{value['architecture']} ({value['distribution']}) "
                        f"[{availability['status']}: {availability['reason']}]"
                    )
            return 0
        value = select_backend(__version__, args.selector).to_dict()
        if args.probe:
            native = load_backend(args.selector)
            query = getattr(native, "_openmp_runtime_info", None)
            value["runtime"] = query() if callable(query) else None
        if args.json:
            print(json.dumps(value, indent=2, sort_keys=True))
        else:
            print(
                f"{value['selector']}: {value['backend']} "
                f"{value['architecture']} ({value['distribution']})"
            )
            if args.probe:
                runtime = value.get("runtime") or {}
                host_blas = runtime.get("host_blas") or {}
                print(
                    "OpenBLAS: " f"{host_blas.get('openblas_config') or 'unavailable'}"
                )
                print(
                    "OpenBLAS OpenMP enabled: "
                    f"{host_blas.get('openblas_openmp_enabled', 'unknown')}"
                )
                print(
                    "OpenBLAS OpenMP runtime compatible: "
                    f"{host_blas.get('openblas_openmp_runtime_compatible', 'unknown')}"
                )
                print(
                    "OpenMP maximum active levels: "
                    f"{runtime.get('omp_max_active_levels', 'unknown')}"
                )
                print(
                    "OpenBLAS threads: "
                    f"{host_blas.get('openblas_threads', 'unknown')}"
                )
        return 0

    from .. import load_backend, selected_backend

    native = load_backend()
    selected = selected_backend()
    if selected and selected.get("backend") in {"cuda", "hip"}:
        report = accelerator_runtime_diagnostics(native_module=native, backend=selected)
    else:
        report = openmp_runtime_diagnostics(strict=not args.advisory)
        report.setdefault("backend", selected)
    report["native"] = {
        "extension": getattr(native, "__file__", ""),
        "build_info": getattr(native, "_backend_build_info", dict)(),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _human_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
