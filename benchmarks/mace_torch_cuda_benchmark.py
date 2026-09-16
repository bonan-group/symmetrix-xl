"""Benchmark upstream MACE with its e3nn or cuEquivariance CUDA backend."""

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import time

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.mace import MACECalculator


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata():
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _gpu_process_memory_mib():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    used_mib = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            try:
                used_mib += int(fields[1])
            except ValueError:
                return None
    return used_mib


def _mib(value):
    return value / (1024 * 1024)


def _summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _per_atom_summary(samples):
    return {
        "median_ms_per_atom": statistics.median(samples),
        "min_ms_per_atom": min(samples),
        "max_ms_per_atom": max(samples),
        "samples_ms_per_atom": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--backend", choices=("e3nn", "cueq"), required=True)
    parser.add_argument("--head", help="Optional multi-head model head")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is not available in this PyTorch environment")

    model_path = args.model.resolve()
    memory_before_mib = _gpu_process_memory_mib()
    calculator_options = dict(
        model_paths=model_path,
        device="cuda",
        default_dtype="float32",
        enable_cueq=args.backend == "cueq",
    )
    if args.head is not None:
        calculator_options["head"] = args.head
    calculator = MACECalculator(**calculator_options)
    model = calculator.models[0]
    model.eval()

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(
        (args.repeat, args.repeat, args.repeat)
    )
    batch_base = calculator._atoms_to_batch(atoms)
    model_dtype = next(model.parameters()).dtype
    for key in batch_base.keys:
        value = batch_base[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch_base[key] = value.to(dtype=model_dtype)

    torch.cuda.empty_cache()
    memory_after_setup_mib = _gpu_process_memory_mib()
    torch.cuda.reset_peak_memory_stats()

    def evaluate():
        batch = calculator._clone_batch(batch_base)
        output = model(
            batch.to_dict(),
            compute_stress=False,
            training=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )
        torch.cuda.synchronize()
        return output

    output = None
    for _ in range(args.warmups):
        output = evaluate()

    samples_ms = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = evaluate()
        samples_ms.append(1000.0 * (time.perf_counter() - start))

    forces = output["forces"].detach().cpu().numpy()
    per_atom_ms = [sample / len(atoms) for sample in samples_ms]
    report = {
        "symmetrix_git": _git_metadata(),
        "model": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
        },
        "backend": args.backend,
        "head": calculator.head,
        "dtype": str(model_dtype),
        "supercell_repeat": args.repeat,
        "atoms": len(atoms),
        "directed_edges": int(batch_base["edge_index"].shape[1]),
        "timing": _summary(samples_ms),
        "timing_per_atom": _per_atom_summary(per_atom_ms),
        "gpu_process_memory_mib": {
            "before": memory_before_mib,
            "after_setup": memory_after_setup_mib,
            "after_evaluation": _gpu_process_memory_mib(),
        },
        "torch_cuda_memory_mib": {
            "allocated_after_evaluation": _mib(torch.cuda.memory_allocated()),
            "reserved_after_evaluation": _mib(torch.cuda.memory_reserved()),
            "peak_allocated": _mib(torch.cuda.max_memory_allocated()),
            "peak_reserved": _mib(torch.cuda.max_memory_reserved()),
        },
        "energy_eV": float(output["energy"].detach().cpu().reshape(-1)[0]),
        "force_max_abs_eV_per_A": float(np.max(np.abs(forces))),
        "force_l2_eV_per_A": float(np.linalg.norm(forces)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
