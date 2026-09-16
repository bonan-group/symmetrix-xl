import json
import sys
from argparse import ArgumentParser


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


def main(argv=None):
    command_argv = sys.argv[1:] if argv is None else list(argv)
    if command_argv and command_argv[0] == "bench":
        from .bench import main as bench_main

        return bench_main(command_argv[1:])
    parser = ArgumentParser(prog="symmetrix")
    commands = parser.add_subparsers(dest="command", required=True)
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
    bench = commands.add_parser(
        "bench", help="Run the standard SrTiO3 MACE-MPA-0 benchmark."
    )
    bench.set_defaults(_handler="bench")
    args = parser.parse_args(argv)

    if args.command == "backend":
        from .. import __version__, load_backend
        from ..backend_loader import backend_inventory, select_backend

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
