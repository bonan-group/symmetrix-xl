"""Time a compiled upstream NequIP model on a fixed periodic graph.

Neighbor-list construction, model loading, and warmup are excluded.  The
timed region includes the model's energy/force/stress execution and CUDA
synchronization, matching the prepared-graph boundary used by the native OAM
timing driver.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("compiled_model", type=Path)
    parser.add_argument(
        "--structure",
        choices=("srtio3", "diamond-si"),
        default="srtio3",
        help="periodic benchmark structure (diamond-si matches the paper's size series)",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--rattle", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.warmup < 1 or args.samples < 1 or args.skin < 0:
        parser.error(
            "repeat, warmup, samples must be positive; skin must be nonnegative"
        )

    import openequivariance  # noqa: F401 - registers TorchScript custom classes
    import torch
    from ase.build import bulk
    from nequip.ase import NequIPCalculator
    from nequip.data import AtomicDataDict, from_ase
    from nequip.data.transforms import NeighborListTransform

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nequip_oam_kokkos_profile import build_srtio3

    if not torch.cuda.is_available():
        raise RuntimeError("the upstream CUDA timing requires a visible CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.compiled_model.suffix == ".pt2":
        # PyTorch 2.10's AOT package loader references this lazily exposed
        # module before importing it in some installation layouts.
        import torch._inductor.codecache  # noqa: F401

    load_start = time.perf_counter()
    calculator = NequIPCalculator.from_compiled_model(
        str(args.compiled_model), device="cuda"
    )
    model_load_seconds = time.perf_counter() - load_start
    model = calculator.model
    model.eval()

    if args.structure == "srtio3":
        atoms = build_srtio3(repeat=args.repeat, rattle=args.rattle, seed=args.seed)
    else:
        atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((args.repeat,) * 3)
        if args.rattle:
            atoms.rattle(stdev=args.rattle, seed=args.seed)
    model_cutoff = float(calculator.transforms[1].r_max)
    data = from_ase(atoms)
    # Reuse the compiled model's species mapping, but deliberately build the
    # same cutoff-plus-skin graph as the prepared Symmetrix benchmark.
    data = calculator.transforms[0](data)
    data = NeighborListTransform(r_max=model_cutoff + args.skin)(data)
    data = AtomicDataDict.to_(data, "cuda")
    directed_edges = int(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1])
    base_data = dict(data)

    def evaluate():
        # GraphModel forwards mutate their dictionary, so retain a stable set
        # of prepared tensors and provide a shallow dictionary per call.
        result = model(dict(base_data))
        torch.cuda.synchronize()
        return result

    for _ in range(args.warmup):
        result = evaluate()
        del result
        gc.collect()

    torch.cuda.reset_peak_memory_stats()
    samples = []
    energy_value = None
    max_force = None
    energy_dtype = None
    forces_dtype = None
    output_properties = None
    for _ in range(args.samples):
        start = time.perf_counter()
        result = evaluate()
        samples.append(time.perf_counter() - start)
        energy = result[AtomicDataDict.TOTAL_ENERGY_KEY].detach()
        forces = result[AtomicDataDict.FORCE_KEY].detach()
        if not bool(torch.isfinite(energy).all() and torch.isfinite(forces).all()):
            raise RuntimeError("upstream NequIP produced nonfinite energy or forces")
        energy_value = float(energy.cpu().reshape(()))
        max_force = float(forces.abs().max().cpu())
        energy_dtype = str(energy.dtype)
        forces_dtype = str(forces.dtype)
        output_properties = sorted(
            key
            for key in (
                AtomicDataDict.TOTAL_ENERGY_KEY,
                AtomicDataDict.FORCE_KEY,
                AtomicDataDict.STRESS_KEY,
            )
            if key in result
        )
        del result, energy, forces
        gc.collect()
    assert energy_value is not None and max_force is not None
    assert energy_dtype is not None and forces_dtype is not None
    assert output_properties is not None
    versions = {}
    for package in ("nequip", "openequivariance", "torch", "e3nn", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    median = statistics.median(samples)
    record = {
        "implementation": (
            "upstream-nequip-openequivariance-aotinductor"
            if args.compiled_model.suffix == ".pt2"
            else "upstream-nequip-openequivariance-torchscript"
        ),
        "precision": "float32",
        "compiled_model": str(args.compiled_model.resolve()),
        "compiled_model_sha256": _sha256(args.compiled_model),
        "python": sys.executable,
        "versions": versions,
        "torch_cuda_version": torch.version.cuda,
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
        "input_dtypes": {
            key: str(value.dtype)
            for key, value in base_data.items()
            if isinstance(value, torch.Tensor)
        },
        "energy_dtype": energy_dtype,
        "forces_dtype": forces_dtype,
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "structure": args.structure,
        "atoms": len(atoms),
        "directed_edges": directed_edges,
        "model_cutoff": model_cutoff,
        "skin": args.skin,
        "effective_cutoff": model_cutoff + args.skin,
        "repeat": args.repeat,
        "rattle": args.rattle,
        "seed": args.seed,
        "warmup": args.warmup,
        "sample_count": args.samples,
        "output_properties": output_properties,
        "median_us_per_atom_step": median * 1e6 / len(atoms),
        "min_us_per_atom_step": min(samples) * 1e6 / len(atoms),
        "samples_us_per_atom_step": [value * 1e6 / len(atoms) for value in samples],
        "median_milliseconds_per_step": median * 1e3,
        "energy": energy_value,
        "max_force": max_force,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "model_load_seconds": model_load_seconds,
    }
    text = json.dumps(record, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
