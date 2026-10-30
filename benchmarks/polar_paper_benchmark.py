#!/usr/bin/env python3
"""Compare MACE-Torch and Symmetrix on paper-representative POLAR systems.

Each implementation is run in a separate process.  Both workers regenerate the
same deterministic structures and record a SHA-256 over species, FP64
coordinates, cell, PBC flags, and electronic inputs.  CUDA timings are fenced
before and after every sample and exclude calculator construction, graph/JIT
warmup, and the correctness snapshot.

Examples::

    python benchmarks/polar_paper_benchmark.py run --implementation mace-torch \
        --device cuda --dtype float32 --torch-backend cueq \
        --model MACE-POLAR-1-M.model \
        --output torch.json
    python benchmarks/polar_paper_benchmark.py run --implementation symmetrix \
        --device cuda --dtype float32 --model MACE-POLAR-1-M.model \
        --realspace-strategy pair_atomic --output symmetrix.json
    python benchmarks/polar_paper_benchmark.py compare \
        --torch-record torch.json --symmetrix-record symmetrix.json \
        --output comparison.json --enforce
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from polar_paper_systems import (
    build_system,
    default_system_names,
    system_names,
    system_sha256,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path("~/.cache/symmetrix/test-models/MACE-POLAR-1-M.model")
PROPERTY_SETS = {
    "EFS": ["energy", "forces", "stress"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_oracle_path(environment: str) -> Path:
    value = os.environ.get(environment)
    if not value:
        raise RuntimeError(f"set {environment} to the pinned oracle checkout")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"oracle checkout not found: {path}")
    return path


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_cuda_profiler():
    candidates = (
        os.environ.get("SYMMETRIX_CUDART_LIBRARY"),
        ctypes.util.find_library("cudart"),
        "libcudart.so",
    )
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        library.cudaProfilerStart.argtypes = ()
        library.cudaProfilerStart.restype = ctypes.c_int
        library.cudaProfilerStop.argtypes = ()
        library.cudaProfilerStop.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("could not load CUDA profiler API: " + "; ".join(errors))


def _nvidia_smi() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "gpu": result.stdout.strip() if result.returncode == 0 else None,
        "competing_processes": (
            processes.stdout.strip() if processes.returncode == 0 else None
        ),
    }


def _torch_calculator_options(backend: str) -> dict[str, bool]:
    return {"enable_cueq": backend == "cueq"}


def _set_calculator_pbc_handling(calculator, value: str) -> list[dict[str, str]]:
    status = []
    for model in calculator.models:
        model_status = {}
        for name in ("electric_potential_descriptor", "coulomb_energy"):
            module = getattr(model, name)
            module.set_pbc_handling(value)
            resolved = str(module.pbc_handling)
            if resolved != value:
                raise RuntimeError(
                    f"converted MACE model retained {name}.pbc_handling={resolved!r}; "
                    f"expected {value!r}"
                )
            model_status[name] = resolved
        status.append(model_status)
    return status


def _torch_calculator(model_path: Path, device: str, dtype: str, backend: str):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import polar_end_to_end_oracle_runner as oracle

    oracle_args = SimpleNamespace(
        oracle_graph_electrostatics=_required_oracle_path(
            "SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE"
        ),
        oracle_mace=_required_oracle_path("SYMMETRIX_POLAR_ORACLE_MACE"),
    )
    torch, gto_module = oracle.load_oracle(oracle_args)
    from polar_torch_baseline import accelerated_backend_status
    from mace.calculators import MACECalculator

    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    oracle.prepare_checkpoint(model, gto_module)
    model.electric_potential_descriptor.set_pbc_handling("auto")
    model.coulomb_energy.set_pbc_handling("auto")
    model = model.to(device=device, dtype=getattr(torch, dtype))
    calculator = MACECalculator(
        models=[model],
        model_type="PolarMACE",
        device=device,
        default_dtype=dtype,
        **_torch_calculator_options(backend),
        info_keys={
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
            "fermi_level": "fermi_level",
        },
    )
    pbc_status = _set_calculator_pbc_handling(calculator, "auto")
    backend_status = accelerated_backend_status(calculator, backend)

    def fence() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    identity = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mace_torch": _version("mace-torch"),
        "graph_electrostatics": _version("graph-electrostatics"),
        "backend": backend,
        "enable_cueq": backend == "cueq",
        "accelerated_backend": backend_status,
        "converted_model_pbc_handling": pbc_status,
        "pbc_handling": "auto",
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    return calculator, fence, identity, lambda: None


def _symmetrix_calculator(
    model_path: Path,
    device: str,
    dtype: str,
    realspace_strategy: str,
    polar_electrostatics: str,
    polar_pme_mesh: tuple[int, int, int],
    polar_pme_order: int,
):
    graph_longrange = _required_oracle_path("SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE")
    sys.path.insert(0, str(graph_longrange.resolve()))

    import symmetrix
    from symmetrix import symmetrix as native

    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    calculator = symmetrix.Symmetrix(
        str(model_path),
        dtype=dtype,
        streamed_edges="direct",
        polar_electrostatics=polar_electrostatics,
        polar_pme_mesh=polar_pme_mesh,
        polar_pme_order=polar_pme_order,
    )
    calculator.evaluator._set_polar_realspace_reverse_strategy(realspace_strategy)
    extension = Path(native.__file__).resolve()
    build_info = dict(native._backend_build_info())
    if build_info["backend"] != device:
        raise RuntimeError(
            f"requested device {device!r}, but the Symmetrix extension uses "
            f"backend {build_info['backend']!r}"
        )
    identity = {
        "symmetrix_xl": _version("symmetrix-xl"),
        "extension": str(extension),
        "extension_sha256": _sha256(extension),
        "backend_build_info": build_info,
        "kokkos_execution_space": symmetrix._kokkos_default_execution_space(),
        "streamed_edges": calculator.streamed_edges,
        "execution_algorithm": calculator.execution_algorithm,
        "realspace_strategy_requested": realspace_strategy,
        "polar_electrostatics_requested": polar_electrostatics,
        "polar_pme_mesh_requested": list(polar_pme_mesh),
        "polar_pme_order_requested": polar_pme_order,
    }

    def close() -> None:
        nonlocal calculator
        calculator = None
        gc.collect()

    return calculator, calculator.evaluator.fence, identity, close


def _snapshot(calculator, atoms, fence: Callable[[], None]) -> dict[str, Any]:
    from ase.stress import full_3x3_to_voigt_6_stress

    calculator.reset()
    properties = list(PROPERTY_SETS["EFS"])
    if "dipole" in calculator.implemented_properties:
        properties.append("dipole")
    calculator.calculate(atoms, properties=properties)
    fence()
    results = calculator.results
    stress = np.asarray(results["stress"], dtype=np.float64)
    if stress.shape == (3, 3):
        stress = full_3x3_to_voigt_6_stress(stress)
    if stress.size != 6:
        raise RuntimeError(f"unexpected stress shape: {stress.shape}")
    snapshot = {
        "energy_eV": float(results["energy"]),
        "forces_eV_per_A": np.asarray(results["forces"], dtype=np.float64).tolist(),
        "stress_eV_per_A3_voigt": stress.reshape(-1).tolist(),
    }
    for key in ("interaction_energy", "electrostatic_energy", "electron_energy"):
        if key in results:
            snapshot[f"{key}_eV"] = float(np.asarray(results[key]).reshape(-1)[0])
    for key in ("dipole", "density_coefficients", "spin_charge_density"):
        if key in results:
            snapshot[key] = np.asarray(results[key], dtype=np.float64).tolist()
    return snapshot


def _timing(
    calculator,
    atoms,
    properties: list[str],
    fence: Callable[[], None],
    warmups: int,
    repeats: int,
    cuda_profiler=None,
) -> dict[str, Any]:
    def evaluate() -> None:
        calculator.reset()
        calculator.calculate(atoms, properties=properties)

    for _ in range(warmups):
        evaluate()
        fence()
    samples = []
    if cuda_profiler is not None and cuda_profiler.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    try:
        for _ in range(repeats):
            fence()
            start = time.perf_counter()
            evaluate()
            fence()
            samples.append((time.perf_counter() - start) * 1.0e3)
    finally:
        if cuda_profiler is not None and cuda_profiler.cudaProfilerStop() != 0:
            raise RuntimeError("cudaProfilerStop failed")
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "us_per_atom": median * 1000.0 / len(atoms),
    }


def _solver_diagnostics(implementation: str, calculator, atoms) -> dict[str, Any]:
    if implementation == "mace-torch":
        return {
            "requested_solver": "auto",
            "resolved_solver": (
                "direct_reciprocal" if bool(np.any(atoms.pbc)) else "realspace"
            ),
        }
    diagnostics = dict(calculator.polar_electrostatics_diagnostics)
    workspace = calculator.evaluator.polar_delta_workspace()
    diagnostics.update(
        {
            key: workspace.get(key)
            for key in (
                "realspace_reverse_strategy_requested",
                "realspace_reverse_strategy_resolved",
                "realspace_reverse_strategy_reason",
                "realspace_retained_allocation_bytes",
                "realspace_full_materialized_retained_bytes",
                "reciprocal_allocation_bytes",
                "n_k",
            )
            if key in workspace
        }
    )
    return diagnostics


def run(args: argparse.Namespace) -> int:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    requested = tuple(item for item in args.systems.split(",") if item)
    unknown = sorted(set(requested) - set(system_names()))
    if unknown:
        raise SystemExit(f"unknown systems: {', '.join(unknown)}")

    gpu_preflight = _nvidia_smi() if args.device == "cuda" else None
    if (
        gpu_preflight is not None
        and gpu_preflight["competing_processes"]
        and not args.allow_busy_gpu
    ):
        raise SystemExit(
            "GPU has active compute processes; rerun when idle or pass "
            "--allow-busy-gpu for a deliberately contaminated diagnostic run:\n"
            + gpu_preflight["competing_processes"]
        )

    if args.implementation == "mace-torch":
        if args.torch_backend == "cueq" and args.device != "cuda":
            raise SystemExit("--torch-backend cueq requires --device cuda")
        calculator, fence, identity, close = _torch_calculator(
            model_path, args.device, args.dtype, args.torch_backend
        )
    else:
        calculator, fence, identity, close = _symmetrix_calculator(
            model_path,
            args.device,
            args.dtype,
            args.realspace_strategy,
            args.polar_electrostatics,
            tuple(args.polar_pme_mesh),
            args.polar_pme_order,
        )
    cuda_profiler = None
    cuda_profiler_library = None
    if args.cuda_profiler_range:
        if args.device != "cuda":
            raise SystemExit("--cuda-profiler-range requires --device cuda")
        cuda_profiler, cuda_profiler_library = _load_cuda_profiler()

    record = {
        "schema": "symmetrix.polar-paper-reconstruction",
        "version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "implementation": args.implementation,
        "protocol": {
            "device": args.device,
            "dtype": args.dtype,
            "polar_electrostatics": (
                args.polar_electrostatics
                if args.implementation == "symmetrix"
                else None
            ),
            "polar_pme_mesh": (
                args.polar_pme_mesh if args.implementation == "symmetrix" else None
            ),
            "polar_pme_order": (
                args.polar_pme_order if args.implementation == "symmetrix" else None
            ),
            "torch_backend": (
                args.torch_backend if args.implementation == "mace-torch" else None
            ),
            "enable_cueq": (
                args.torch_backend == "cueq"
                if args.implementation == "mace-torch"
                else None
            ),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "synchronized": args.device == "cuda",
            "cuda_profiler_range": args.cuda_profiler_range,
            "cuda_profiler_library": cuda_profiler_library,
            "reset_before_sample": True,
            "property_execution_note": (
                "Only the simulation-relevant energy/forces/stress request is "
                "timed for both implementations."
            ),
            "excluded": [
                "calculator construction",
                "graph/JIT compilation",
                "warmup",
                "correctness snapshot",
            ],
        },
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "bytes": model_path.stat().st_size,
        },
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "implementation": identity,
            "nvidia_smi_preflight": gpu_preflight,
        },
        "systems": [],
    }
    try:
        for name in requested:
            atoms, provenance = build_system(name)
            atoms.calc = calculator
            timings = {
                label: _timing(
                    calculator,
                    atoms,
                    properties,
                    fence,
                    args.warmups,
                    args.repeats,
                    cuda_profiler,
                )
                for label, properties in PROPERTY_SETS.items()
            }
            snapshot = _snapshot(calculator, atoms, fence)
            entry = {
                "name": name,
                **provenance,
                "system_sha256": system_sha256(atoms),
                "atoms": len(atoms),
                "cell_A": np.asarray(atoms.cell, dtype=np.float64).tolist(),
                "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
                "timings": timings,
                "snapshot": snapshot,
                "solver": _solver_diagnostics(args.implementation, calculator, atoms),
            }
            record["systems"].append(entry)
            print(
                f"{name}: N={len(atoms)} "
                + " ".join(
                    f"{label}={timings[label]['us_per_atom']:.3f} us/atom"
                    for label in PROPERTY_SETS
                ),
                flush=True,
            )
            atoms.calc = None
    finally:
        close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


def _indexed(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in record["systems"]}


def compare(args: argparse.Namespace) -> int:
    torch_record = json.loads(args.torch_record.read_text())
    symmetrix_record = json.loads(args.symmetrix_record.read_text())
    if torch_record["implementation"] != "mace-torch":
        raise SystemExit("--torch-record is not a MACE-Torch record")
    if symmetrix_record["implementation"] != "symmetrix":
        raise SystemExit("--symmetrix-record is not a Symmetrix record")
    for field in ("sha256",):
        if torch_record["model"][field] != symmetrix_record["model"][field]:
            raise SystemExit(f"model {field} differs between records")
    for field in ("device", "dtype", "warmups", "repeats", "synchronized"):
        if torch_record["protocol"].get(field) != symmetrix_record["protocol"].get(
            field
        ):
            raise SystemExit(f"protocol {field} differs between records")

    left, right = _indexed(torch_record), _indexed(symmetrix_record)
    if set(left) != set(right):
        raise SystemExit("system sets differ between records")
    output = {
        "schema": "symmetrix.polar-paper-comparison",
        "version": 1,
        "torch_record": str(args.torch_record.resolve()),
        "symmetrix_record": str(args.symmetrix_record.resolve()),
        "dtype": torch_record["protocol"]["dtype"],
        "device": torch_record["protocol"]["device"],
        "timing_comparability": {
            "EFS": "like-for-like public property request",
        },
        "tolerances": {
            "energy_abs_eV_per_atom": args.energy_atol_per_atom,
            "force_max_abs_eV_per_A": args.force_atol,
            "stress_max_abs_eV_per_A3": args.stress_atol,
        },
        "systems": [],
    }
    all_pass = True
    for name in left:
        torch_system, symmetrix_system = left[name], right[name]
        if torch_system["system_sha256"] != symmetrix_system["system_sha256"]:
            raise SystemExit(f"fixture hash differs for {name}")
        if torch_system["pbc"] != symmetrix_system["pbc"]:
            raise SystemExit(f"PBC flags differ for {name}")
        periodic = bool(np.any(torch_system["pbc"]))
        atoms = int(torch_system["atoms"])
        ts, ss = torch_system["snapshot"], symmetrix_system["snapshot"]
        energy_error = abs(ss["energy_eV"] - ts["energy_eV"])
        force_delta = np.asarray(ss["forces_eV_per_A"]) - np.asarray(
            ts["forces_eV_per_A"]
        )
        if periodic:
            stress_delta = np.asarray(ss["stress_eV_per_A3_voigt"]) - np.asarray(
                ts["stress_eV_per_A3_voigt"]
            )
            stress_reference = "mace_torch"
        else:
            stress_delta = np.asarray(ss["stress_eV_per_A3_voigt"])
            stress_reference = "isolated_exact_zero"
        metrics = {
            "energy_abs_eV": energy_error,
            "energy_abs_eV_per_atom": energy_error / atoms,
            "force_max_abs_eV_per_A": float(np.max(np.abs(force_delta))),
            "force_rms_eV_per_A": float(np.sqrt(np.mean(force_delta**2))),
            "stress_max_abs_eV_per_A3": float(np.max(np.abs(stress_delta))),
            "stress_reference": stress_reference,
        }
        for key in (
            "interaction_energy_eV",
            "electrostatic_energy_eV",
            "electron_energy_eV",
        ):
            if key in ts and key in ss:
                metrics[f"{key}_abs"] = abs(ss[key] - ts[key])
        for key in ("dipole", "density_coefficients", "spin_charge_density"):
            if key in ts and key in ss:
                delta = np.asarray(ss[key]) - np.asarray(ts[key])
                metrics[f"{key}_max_abs"] = float(np.max(np.abs(delta)))
        passed = (
            metrics["energy_abs_eV_per_atom"] <= args.energy_atol_per_atom
            and metrics["force_max_abs_eV_per_A"] <= args.force_atol
            and metrics["stress_max_abs_eV_per_A3"] <= args.stress_atol
        )
        all_pass = all_pass and passed
        ratios = {
            label: (
                symmetrix_system["timings"][label]["median_ms"]
                / torch_system["timings"][label]["median_ms"]
            )
            for label in PROPERTY_SETS
        }
        output["systems"].append(
            {
                "name": name,
                "atoms": atoms,
                "system_sha256": torch_system["system_sha256"],
                "correctness": metrics,
                "passed": passed,
                "symmetrix_over_torch_time_ratio": ratios,
                "torch_timings": torch_system["timings"],
                "symmetrix_timings": symmetrix_system["timings"],
                "torch_solver": torch_system["solver"],
                "symmetrix_solver": symmetrix_system["solver"],
            }
        )
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}; "
            f"dE/N={metrics['energy_abs_eV_per_atom']:.3e} eV, "
            f"max|dF|={metrics['force_max_abs_eV_per_A']:.3e} eV/A, "
            f"max|dS|={metrics['stress_max_abs_eV_per_A3']:.3e} eV/A^3; "
            f"EFS ratio={ratios['EFS']:.3f}",
            flush=True,
        )
    output["passed"] = all_pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.enforce and not all_pass:
        return 1
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("run")
    worker.add_argument(
        "--implementation", required=True, choices=("mace-torch", "symmetrix")
    )
    worker.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    worker.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    worker.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    worker.add_argument(
        "--torch-backend",
        choices=("e3nn", "cueq"),
        default="cueq",
        help="MACE-Torch tensor-product backend; ignored for Symmetrix",
    )
    worker.add_argument("--systems", default=",".join(default_system_names()))
    worker.add_argument("--warmups", type=int, default=3)
    worker.add_argument("--repeats", type=int, default=7)
    worker.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="allow a diagnostic CUDA run despite active compute processes",
    )
    worker.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="capture only measured repetitions via the CUDA profiler API",
    )
    worker.add_argument(
        "--realspace-strategy",
        choices=("auto", "owner", "pair_atomic", "materialized_forward"),
        default="auto",
    )
    worker.add_argument(
        "--polar-electrostatics",
        choices=("auto", "direct_reciprocal", "realspace", "pme"),
        default="auto",
        help="runtime POLAR long-range solver request; ignored for MACE-Torch",
    )
    worker.add_argument(
        "--polar-pme-mesh",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=(64, 64, 64),
    )
    worker.add_argument("--polar-pme-order", type=int, default=6)
    worker.add_argument("--output", type=Path, required=True)
    worker.set_defaults(function=run)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--torch-record", type=Path, required=True)
    comparison.add_argument("--symmetrix-record", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--energy-atol-per-atom", type=float, default=2.0e-4)
    comparison.add_argument("--force-atol", type=float, default=5.0e-3)
    comparison.add_argument("--stress-atol", type=float, default=5.0e-4)
    comparison.add_argument("--enforce", action="store_true")
    comparison.set_defaults(function=compare)
    return result


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "warmups", 1) < 0 or getattr(args, "repeats", 1) <= 0:
        raise SystemExit("warmups must be non-negative and repeats must be positive")
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
