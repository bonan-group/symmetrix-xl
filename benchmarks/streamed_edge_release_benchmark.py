"""Milestone benchmark for current, original-develop, and PyTorch MACE.

Run one implementation per fresh process. Use ``compare`` to combine records;
timing runs are intentionally separate from profiler captures.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.neighborlist import neighbor_list


PROPERTIES = ("energy", "forces", "stress")


def isolate_symmetrix_stage() -> None:
    stage = os.environ.get("SYMMETRIX_BENCHMARK_PACKAGE_ROOT")
    if not stage:
        return
    resolved = str(Path(stage).resolve())
    sys.path.insert(0, resolved)
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if not finder.__class__.__module__.startswith("_editable_skbc_symmetrix")
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_atoms(repeat: int, seed: int):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    return atoms


def directed_edges(atoms, cutoff: float) -> int:
    return int(len(neighbor_list("i", atoms, cutoff)))


def synchronize(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_source_provenance(module_path: Path) -> dict[str, object]:
    source = module_path.resolve()
    root_result = subprocess.run(
        ["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if root_result.returncode != 0:
        return {
            "module": str(source),
            "git_root": None,
            "git_revision": None,
            "git_dirty": None,
        }
    root = Path(root_result.stdout.strip()).resolve()
    revision_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "module": str(source),
        "git_root": str(root),
        "git_revision": revision_result.stdout.strip(),
        "git_dirty": bool(status_result.stdout.strip()),
    }


def pytorch_head_identity(calculator, requested_head: str) -> dict[str, object]:
    selected_head = str(calculator.head)
    available_heads = [str(head) for head in calculator.available_heads]
    if selected_head != requested_head:
        raise RuntimeError(
            f"MACE selected head {selected_head!r}, requested {requested_head!r}; "
            f"available heads are {available_heads!r}"
        )
    return {
        "requested_head": requested_head,
        "selected_head": selected_head,
        "available_heads": available_heads,
    }


def gpu_process_memory_mib() -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    process_id = str(os.getpid())
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == process_id:
            return int(fields[1])
    return None


def finalize_symmetrix_runtime() -> None:
    gc.collect()
    native = sys.modules.get("symmetrix.symmetrix")
    is_initialized = getattr(native, "_kokkos_is_initialized", None)
    finalize = getattr(native, "_finalize_kokkos", None)
    if callable(is_initialized) and callable(finalize) and is_initialized():
        finalize()


def make_calculator(args):
    if args.implementation == "pytorch":
        import mace
        import torch
        from mace.calculators.mace import MACECalculator

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
        enable_cueq = args.pytorch_backend == "cueq"
        if enable_cueq and args.device != "cuda":
            raise ValueError("PyTorch cuEquivariance requires --device cuda")
        calculator = MACECalculator(
            model_paths=args.checkpoint,
            device=args.device,
            default_dtype=args.dtype,
            enable_cueq=enable_cueq,
            head=args.head,
        )
        identity = {
            "implementation": f"pytorch-mace-{args.pytorch_backend}",
            "pytorch_backend": args.pytorch_backend,
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            **pytorch_head_identity(calculator, args.head),
            "mace_source": git_source_provenance(Path(mace.__file__)),
            "torch_runtime": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
        }
        return calculator, identity

    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native

    expected_space = "Cuda" if args.device == "cuda" else "OpenMP"
    execution_space_query = getattr(native, "_kokkos_default_execution_space", None)
    execution_space = (
        execution_space_query() if callable(execution_space_query) else expected_space
    )
    if execution_space != expected_space:
        raise RuntimeError(
            f"Kokkos execution space is {execution_space}, expected {expected_space}"
        )
    if args.implementation == "develop":
        calculator = Symmetrix(
            args.checkpoint,
            use_kokkos=True,
            dtype=args.dtype,
            species=("Al", "N"),
            head=args.head,
        )
        identity = {
            "implementation": "symmetrix-origin-develop-materialized",
            "execution_space": execution_space,
            "execution_space_query_available": callable(execution_space_query),
            "streamed_edges": "materialized",
            "low_memory": False,
        }
    else:
        calculator = Symmetrix(
            args.compact_model,
            use_kokkos=True,
            dtype=args.dtype,
            streamed_edges=args.mode,
            low_memory=args.low_memory,
            m1_polynomial_policy=args.m1_polynomial_policy,
            neighbor_skin=args.skin,
        )
        identity = {
            "implementation": "symmetrix-streamed-edge",
            "execution_space": execution_space,
            "streamed_edges_requested": args.mode,
            "streamed_edges": calculator.streamed_edges,
            "low_memory": bool(calculator.low_memory),
            "m1_polynomial_policy_requested": args.m1_polynomial_policy,
            "m1_polynomial_policy": calculator.m1_polynomial_policy,
            "jit_status": calculator.jit_status,
            "jit_compiler_backend": calculator.jit_compiler_backend,
            "jit_artifact_id": calculator.jit_artifact_id,
            "jit_variant_id": calculator.jit_variant_id,
        }
    native_path = Path(native.__file__).resolve()
    identity["native_extension"] = str(native_path)
    identity["native_extension_sha256"] = sha256_file(native_path)
    return calculator, identity


def evaluate(calculator, atoms):
    calculator.calculate(
        atoms,
        properties=list(PROPERTIES),
        system_changes=all_changes,
    )
    return {
        "energy_eV": float(calculator.results["energy"]),
        "forces_eV_per_A": np.asarray(
            calculator.results["forces"], dtype=float
        ).tolist(),
        "stress_eV_per_A3_voigt": np.asarray(
            calculator.results["stress"], dtype=float
        ).tolist(),
    }


def run(args) -> int:
    if args.device == "cpu" or args.implementation == "pytorch":
        for variable in (
            "OMP_NUM_THREADS",
            "KOKKOS_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            os.environ[variable] = str(args.threads)
    atoms = build_atoms(args.repeat, args.seed)
    model_cutoff = float(args.cutoff)
    effective_cutoff = model_cutoff + (
        args.skin if args.implementation == "current" else 0.0
    )

    setup_start = time.perf_counter()
    calculator, identity = make_calculator(args)
    setup_ms = 1000.0 * (time.perf_counter() - setup_start)
    outputs = None
    for _ in range(args.warmups):
        outputs = evaluate(calculator, atoms)
        synchronize(args.device)

    samples_ms = []
    for _ in range(args.samples):
        synchronize(args.device)
        start = time.perf_counter()
        outputs = evaluate(calculator, atoms)
        synchronize(args.device)
        samples_ms.append(1000.0 * (time.perf_counter() - start))

    median_ms = statistics.median(samples_ms)
    module = sys.modules.get("symmetrix")
    report = {
        "schema_version": 1,
        "revision": args.revision,
        "identity": identity,
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "compact_model": (
                str(args.compact_model.resolve())
                if args.implementation == "current"
                else None
            ),
            "compact_model_sha256": (
                sha256_file(args.compact_model)
                if args.implementation == "current"
                else None
            ),
            "head": args.head,
            "cutoff_A": model_cutoff,
        },
        "workload": {
            "structure": "perturbed-wurtzite-AlN",
            "repeat": args.repeat,
            "seed": args.seed,
            "atoms": len(atoms),
            "properties": list(PROPERTIES),
            "precision": args.dtype,
            "skin_A": args.skin if args.implementation == "current" else 0.0,
            "effective_cutoff_A": effective_cutoff,
            "directed_edges_at_model_cutoff": directed_edges(atoms, model_cutoff),
            "directed_edges_at_effective_cutoff": directed_edges(
                atoms, effective_cutoff
            ),
        },
        "timing": {
            "boundary": "ASE calculator energy+forces+stress evaluation",
            "warmups": args.warmups,
            "samples": args.samples,
            "samples_ms": samples_ms,
            "median_ms": median_ms,
            "median_us_per_atom": 1000.0 * median_ms / len(atoms),
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "setup_ms": setup_ms,
        },
        "memory": {
            "process_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "sampled_cuda_process_mib": gpu_process_memory_mib()
            if args.device == "cuda"
            else None,
        },
        "outputs": outputs,
        "environment": {
            "benchmark_driver": str(Path(__file__).resolve()),
            "benchmark_driver_sha256": sha256_file(Path(__file__).resolve()),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "symmetrix_source": str(Path(module.__file__).resolve())
            if module
            else None,
            "packages": {
                name: package_version(name)
                for name in (
                    "symmetrix",
                    "mace-torch",
                    "torch",
                    "cuequivariance",
                    "cuequivariance-torch",
                    "cuequivariance-ops-torch-cu12",
                    "ase",
                    "numpy",
                )
            },
            "device": args.device,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "KOKKOS_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                )
            },
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n")
    print(rendered)
    if args.implementation in ("current", "develop"):
        del calculator
        finalize_symmetrix_runtime()
    return 0


def compare_records(records):
    loaded = [json.loads(path.read_text()) for path in records]
    reference = next(
        record
        for record in loaded
        if record["identity"]["implementation"] == "pytorch-mace-e3nn"
    )
    reference_outputs = reference["outputs"]
    rows = []
    for record in loaded:
        outputs = record["outputs"]
        rows.append(
            {
                "implementation": record["identity"]["implementation"],
                "mode": record["identity"].get("streamed_edges"),
                "low_memory": record["identity"].get("low_memory"),
                "median_us_per_atom": record["timing"]["median_us_per_atom"],
                "speedup_vs_pytorch": reference["timing"]["median_ms"]
                / record["timing"]["median_ms"],
                "energy_abs_error_eV_per_atom": abs(
                    outputs["energy_eV"] - reference_outputs["energy_eV"]
                )
                / record["workload"]["atoms"],
                "force_max_abs_error_eV_per_A": float(
                    np.max(
                        np.abs(
                            np.asarray(outputs["forces_eV_per_A"])
                            - np.asarray(reference_outputs["forces_eV_per_A"])
                        )
                    )
                ),
                "stress_max_abs_error_eV_per_A3": float(
                    np.max(
                        np.abs(
                            np.asarray(outputs["stress_eV_per_A3_voigt"])
                            - np.asarray(reference_outputs["stress_eV_per_A3_voigt"])
                        )
                    )
                ),
            }
        )
    return {
        "schema_version": 1,
        "records": [str(path) for path in records],
        "rows": rows,
    }


def compare(args) -> int:
    report = compare_records(args.records)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--implementation", choices=("current", "develop", "pytorch"), required=True
    )
    run_parser.add_argument(
        "--pytorch-backend", choices=("e3nn", "cueq"), default="e3nn"
    )
    run_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--compact-model", type=Path)
    run_parser.add_argument("--head", default="default")
    run_parser.add_argument(
        "--mode", choices=("materialized", "generic", "direct"), default="direct"
    )
    run_parser.add_argument(
        "--low-memory", action=argparse.BooleanOptionalAction, default=True
    )
    run_parser.add_argument(
        "--m1-polynomial-policy",
        choices=("automatic", "recompute", "retained"),
        default="automatic",
    )
    run_parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    run_parser.add_argument("--repeat", type=int, default=6)
    run_parser.add_argument("--seed", type=int, default=20260822)
    run_parser.add_argument("--cutoff", type=float, default=6.0)
    run_parser.add_argument("--skin", type=float, default=0.5)
    run_parser.add_argument("--warmups", type=int, default=3)
    run_parser.add_argument("--samples", type=int, default=10)
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--revision", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(handler=run)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("records", type=Path, nargs="+")
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(handler=compare)
    return root


def main(argv=None) -> int:
    isolate_symmetrix_stage()
    args = parser().parse_args(argv)
    if (
        getattr(args, "implementation", None) == "current"
        and args.compact_model is None
    ):
        raise SystemExit("--compact-model is required for the current implementation")
    for name in ("repeat", "warmups", "samples", "threads"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
