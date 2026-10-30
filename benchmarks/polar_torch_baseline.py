"""Time the frozen torch oracle stack (mace ``PolarMACE`` + ``graph_longrange``)
on the campaign's fixture systems, mirroring the symmetrix records' protocol.

The default invocation is the one the CPU record used: one physical core, all
thread variables 1, fp64, median of 5 with 2 warmups.  ``--device cuda`` runs
the same fixtures on the accelerator (the model and the calculator on the
device, every timed region bracketed by ``torch.cuda.synchronize``) so the
Symmetrix records have a same-fixture torch ratio on both backends.

The oracle environment provides torch/ase/e3nn; the mace and
graph_electrostatics checkouts are prepended to ``sys.path`` exactly as
``polar_end_to_end_oracle_runner.load_oracle`` does, and this driver reuses that
module's fixture geometries and checkpoint preparation so the timed systems are
the records' own fixtures.  ASE caches calculator results for unchanged atoms,
so every timed call is preceded by ``calc.reset()`` -- without it the second
call returns cached numbers.

Run (CPU, the record's own protocol):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 0 .venv/bin/python benchmarks/polar_torch_baseline.py

Run (CUDA):

    .venv/bin/python benchmarks/polar_torch_baseline.py --device cuda \
        --oracle-mace /path/to/mace \
        --oracle-graph-electrostatics /path/to/graph_electrostatics \
        --cases M:1,M:2,M:3,M:4,M:5 --output benchmarks/.artifacts/polar-torch-cuda/record.json

The CUDA default is the preserved standard PyTorch/e3nn baseline.  Pass
``--backend cueq`` to produce the separate MACE-Torch+cuEquivariance record;
that mode checks energy, forces, and stress against a fresh standard calculator
after the timed region and refuses an unavailable/fallback ops runtime.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import polar_end_to_end_oracle_runner as R  # noqa: E402
from ase import Atoms  # noqa: E402

torch.set_num_threads(1)

DEFAULT_CASES = "M:1,M:2,M:3,S:1,S:2,L:1"
BACKEND_LABELS = {
    "e3nn": "standard PyTorch/e3nn",
    "cueq": "MACE-Torch+cuEquivariance",
}


def parse_cases(spec):
    cases = []
    for item in spec.split(","):
        size, _, repeat = item.partition(":")
        repeat = int(repeat or 1)
        cases.append((size, (repeat, repeat, repeat)))
    return cases


def calculator_options(backend):
    return {"enable_cueq": backend == "cueq"}


def build(size, device, dtype, backend):
    from mace.calculators import MACECalculator

    model = torch.load(
        str(R._model_path(args, size)),
        map_location="cpu",
        weights_only=False,
    )
    R.prepare_checkpoint(model, gto)
    model = model.to(device=device, dtype=getattr(torch, dtype))
    return MACECalculator(
        models=[model],
        model_type="PolarMACE",
        device=device,
        default_dtype=dtype,
        **calculator_options(backend),
        info_keys={
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
            "fermi_level": "fermi_level",
        },
    )


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cuda_driver_version():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
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
    return result.stdout.splitlines()[0].strip() if result.stdout else None


def cueq_ops_runtime():
    try:
        import cuequivariance_ops_torch
    except (ImportError, OSError) as error:
        return {"available": False, "module": None, "error": str(error)}
    return {
        "available": True,
        "module": str(pathlib.Path(cuequivariance_ops_torch.__file__).resolve()),
        "error": None,
    }


def accelerated_backend_status(calculator, requested_backend):
    model = calculator.models[0]
    modules = list(model.modules())
    cueq_module_types = sorted(
        {
            f"{type(module).__module__}.{type(module).__qualname__}"
            for module in modules
            if type(module).__module__.startswith(
                ("cuequivariance", "cuequivariance_torch")
            )
        }
    )
    enabled_configs = []
    for name, module in model.named_modules():
        config = getattr(module, "cueq_config", None)
        if config is not None and bool(getattr(config, "enabled", False)):
            enabled_configs.append(
                {
                    "module": name or "<root>",
                    "layout": getattr(config, "layout_str", None),
                    "optimize_all": bool(getattr(config, "optimize_all", False)),
                    "conv_fusion": bool(getattr(config, "conv_fusion", False)),
                }
            )
    status = {
        "requested_backend": requested_backend,
        "calculator_enable_cueq": bool(getattr(calculator, "_enable_cueq", False)),
        "calculator_uses_accelerated_backend": bool(
            getattr(calculator, "_uses_accelerated_backend", False)
        ),
        "cueq_module_count": sum(
            type(module).__module__.startswith(
                ("cuequivariance", "cuequivariance_torch")
            )
            for module in modules
        ),
        "cueq_module_types": cueq_module_types,
        "enabled_cueq_config_count": len(enabled_configs),
        "enabled_cueq_configs": enabled_configs,
        "ops_runtime": cueq_ops_runtime(),
    }
    accelerated_active = (
        status["calculator_enable_cueq"]
        and status["calculator_uses_accelerated_backend"]
        and (status["cueq_module_count"] > 0 or status["enabled_cueq_config_count"] > 0)
        and status["ops_runtime"]["available"]
    )
    if requested_backend == "cueq":
        verified = accelerated_active
    else:
        verified = (
            not status["calculator_enable_cueq"]
            and not status["calculator_uses_accelerated_backend"]
        )
    status["accelerated_backend_active"] = accelerated_active
    status["requested_backend_verified"] = verified
    if not verified:
        raise RuntimeError(
            f"MACECalculator did not activate the requested {requested_backend} "
            f"backend: {status}"
        )
    return status


geoms = {g["name"]: g for g in R.fixture_geometries()}


def make_system(rep):
    g = geoms["bulk_w4"]
    a = Atoms(g["species"], positions=g["positions"], cell=g["cell"], pbc=g["pbc"])
    if rep != (1, 1, 1):
        a = a.repeat(rep)
    a.info.update(
        {
            "charge": 0.0,
            "spin": 0.0,
            "external_field": np.zeros(3),
            "fermi_level": 0.0,
        }
    )
    return a


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(atoms, calls, device):
    atoms.calc.reset()
    sync(device)
    t0 = time.perf_counter()
    calls()
    sync(device)
    return time.perf_counter() - t0


def property_snapshot(atoms, device):
    atoms.calc.reset()
    sync(device)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    stress = np.asarray(atoms.get_stress(), dtype=np.float64)
    sync(device)
    return {"energy_eV": energy, "forces_eV_per_A": forces, "stress_eV_per_A3": stress}


def graph_facts_from_evaluation_batch(calculator, batch, atoms):
    """Describe the real graph passed through MACE's calculator evaluation path."""
    edge_index = batch["edge_index"]
    source_real = edge_index[0] < atoms
    target_real = edge_index[1] < atoms
    if bool(torch.any(source_real != target_real).item()):
        raise RuntimeError("a MACE graph edge crosses the real/padding boundary")
    real_edges = int(torch.count_nonzero(source_real & target_real).item())
    batch_edges = int(edge_index.shape[1])
    cutoff = float(calculator.r_max)
    return {
        "construction_path": "MACECalculator._atoms_to_batch",
        "model_cutoff_A": cutoff,
        "neighbor_skin_A": 0.0,
        "effective_neighbor_cutoff_A": cutoff,
        "directed_edges_model_cutoff": real_edges,
        "evaluation_batch_directed_edges": batch_edges,
        "padding_directed_edges": batch_edges - real_edges,
    }


def require_consistent_graph_facts(captures):
    if not captures:
        raise RuntimeError("MACE evaluation produced no captured graph")
    first = captures[0]
    if any(item != first for item in captures[1:]):
        raise RuntimeError(f"MACE graph changed across evaluations: {captures}")
    return first


def property_snapshot_with_graph(atoms, device):
    """Capture graph facts from the actual MACE evaluation that makes the snapshot."""
    calculator = atoms.calc
    original_atoms_to_batch = calculator._atoms_to_batch
    captures = []

    def recording_atoms_to_batch(evaluated_atoms):
        batch = original_atoms_to_batch(evaluated_atoms)
        captures.append(
            graph_facts_from_evaluation_batch(
                calculator, batch, atoms=len(evaluated_atoms)
            )
        )
        return batch

    calculator._atoms_to_batch = recording_atoms_to_batch
    try:
        snapshot = property_snapshot(atoms, device)
    finally:
        calculator._atoms_to_batch = original_atoms_to_batch
    return snapshot, require_consistent_graph_facts(captures)


def parity_metrics(reference, candidate, atoms):
    if reference["forces_eV_per_A"].shape != candidate["forces_eV_per_A"].shape:
        raise RuntimeError("cuEquivariance forces have a different shape from e3nn")
    if reference["stress_eV_per_A3"].shape != candidate["stress_eV_per_A3"].shape:
        raise RuntimeError("cuEquivariance stress has a different shape from e3nn")
    return {
        "energy_abs_error_eV": abs(candidate["energy_eV"] - reference["energy_eV"]),
        "energy_abs_error_eV_per_atom": abs(
            candidate["energy_eV"] - reference["energy_eV"]
        )
        / atoms,
        "force_max_abs_error_eV_per_A": float(
            np.max(np.abs(candidate["forces_eV_per_A"] - reference["forces_eV_per_A"]))
        ),
        "stress_max_abs_error_eV_per_A3": float(
            np.max(
                np.abs(candidate["stress_eV_per_A3"] - reference["stress_eV_per_A3"])
            )
        ),
    }


def serializable_snapshot(snapshot):
    return {
        "energy_eV": snapshot["energy_eV"],
        "forces_eV_per_A": snapshot["forces_eV_per_A"].tolist(),
        "stress_eV_per_A3_voigt": snapshot["stress_eV_per_A3"].tolist(),
    }


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument(
        "--backend",
        default="e3nn",
        choices=("e3nn", "cueq"),
        help="MACE tensor-product backend; the default preserves the existing baseline",
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument(
        "--oracle-mace",
        type=pathlib.Path,
        default=os.environ.get("SYMMETRIX_POLAR_ORACLE_MACE"),
        required="SYMMETRIX_POLAR_ORACLE_MACE" not in os.environ,
    )
    parser.add_argument(
        "--oracle-graph-electrostatics",
        type=pathlib.Path,
        default=os.environ.get("SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE"),
        required="SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE" not in os.environ,
    )
    parser.add_argument("--parity-energy-atol-per-atom", type=float, default=2.0e-4)
    parser.add_argument("--parity-force-atol", type=float, default=5.0e-3)
    parser.add_argument("--parity-stress-atol", type=float, default=5.0e-4)
    return parser


def main(parsed=None):
    parser = make_parser()
    if parsed is None:
        parsed = parser.parse_args()

    if parsed.backend == "cueq" and parsed.device != "cuda":
        parser.error("--backend cueq requires --device cuda")

    cases = parse_cases(parsed.cases)
    record = {
        "schema": "symmetrix.polar.torch-baseline-record",
        "version": 3,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "implementation": BACKEND_LABELS[parsed.backend],
            "backend": parsed.backend,
            "enable_cueq": parsed.backend == "cueq",
            "device": parsed.device,
            "dtype": parsed.dtype,
            "warmups": parsed.warmups,
            "repeats": parsed.repeats,
            "synchronize": parsed.device == "cuda",
            "fixture": "bulk_w4 (MACE-POLAR-1-M/slab/bulk fixture cell), supercell repeat",
            "reset_before_every_call": True,
            "requested_properties": ["energy", "forces", "stress"],
            "parity_reference": (
                "standard PyTorch/e3nn" if parsed.backend == "cueq" else None
            ),
        },
        "environment": {
            "mace": package_version("mace-torch"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_driver": cuda_driver_version(),
            "cuequivariance": package_version("cuequivariance"),
            "cuequivariance_torch": package_version("cuequivariance-torch"),
            "cuequivariance_ops": package_version("cuequivariance-ops-cu13"),
            "cuequivariance_ops_torch": package_version(
                "cuequivariance-ops-torch-cu13"
            ),
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "device_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None
            ),
            "cueq_ops_runtime": cueq_ops_runtime(),
            "python": sys.version.split()[0],
        },
        "identity": {
            "implementation": BACKEND_LABELS[parsed.backend],
            "backend": parsed.backend,
        },
        "systems": [],
    }
    print(
        f"{BACKEND_LABELS[parsed.backend]} | torch {torch.__version__} | "
        f"device={parsed.device} | {parsed.dtype} | "
        f"median of {parsed.repeats}, {parsed.warmups} warmups"
    )
    for size, rep in cases:
        calc = build(size, parsed.device, parsed.dtype, parsed.backend)
        backend_status = accelerated_backend_status(calc, parsed.backend)
        a = make_system(rep)
        a.calc = calc
        name = "bulk_w4" + ("" if rep == (1, 1, 1) else f"_x{rep[0] ** 3}")

        for _ in range(parsed.warmups):
            a.calc.reset()
            a.get_potential_energy()
            sync(parsed.device)
        ts_e = [
            timed(a, a.get_potential_energy, parsed.device)
            for _ in range(parsed.repeats)
        ]

        def full():
            a.get_potential_energy()
            a.get_forces()
            a.get_stress()

        for _ in range(parsed.warmups):
            a.calc.reset()
            a.get_forces()
            sync(parsed.device)
        ts_ef = [timed(a, full, parsed.device) for _ in range(parsed.repeats)]

        # Correctness snapshots follow timing so they do not become an
        # undeclared extra warmup.  The cuEquivariance calculator is released
        # before constructing the standard reference, keeping the two timing
        # baselines independently resident.
        snapshot, graph = property_snapshot_with_graph(a, parsed.device)
        parity = None
        if parsed.backend == "cueq":
            a.calc = None
            del calc
            gc.collect()
            if parsed.device == "cuda":
                torch.cuda.empty_cache()
                sync(parsed.device)
            reference_calc = build(size, parsed.device, parsed.dtype, "e3nn")
            reference_atoms = make_system(rep)
            reference_atoms.calc = reference_calc
            reference, reference_graph = property_snapshot_with_graph(
                reference_atoms, parsed.device
            )
            if reference_graph != graph:
                raise RuntimeError(
                    f"cuEquivariance/e3nn graph mismatch for {name}: "
                    f"{graph} != {reference_graph}"
                )
            parity = parity_metrics(reference, snapshot, len(a))
            limits = {
                "energy_abs_error_eV_per_atom": parsed.parity_energy_atol_per_atom,
                "force_max_abs_error_eV_per_A": parsed.parity_force_atol,
                "stress_max_abs_error_eV_per_A3": parsed.parity_stress_atol,
            }
            failures = {
                key: (parity[key], limit)
                for key, limit in limits.items()
                if parity[key] > limit
            }
            parity["tolerances"] = limits
            parity["qualified"] = not failures
            del reference_atoms, reference_calc
            if failures:
                raise RuntimeError(
                    f"cuEquivariance parity failed for {name}: {failures}"
                )

        med_e = statistics.median(ts_e)
        med_ef = statistics.median(ts_ef)
        entry = {
            "size": size,
            "system": name,
            "atoms": len(a),
            "checkpoint": {
                "path": str(R._model_path(args, size)),
                "sha256": file_sha256(R._model_path(args, size)),
            },
            "graph": graph,
            "accelerated_backend": backend_status,
            "outputs": serializable_snapshot(snapshot),
            "parity_against_standard_e3nn": parity,
            "energy_ms": med_e * 1e3,
            "energy_us_per_atom": med_e * 1e6 / len(a),
            "energy_forces_stress_ms": med_ef * 1e3,
            "energy_forces_stress_us_per_atom": med_ef * 1e6 / len(a),
            "energy_samples_ms": [value * 1e3 for value in ts_e],
            "energy_forces_stress_samples_ms": [value * 1e3 for value in ts_ef],
        }
        record["systems"].append(entry)
        print(
            f"{size} {name:14s} atoms={len(a):4d} "
            f"E={entry['energy_us_per_atom']:10.1f} us/atom  "
            f"E+F+sigma={entry['energy_forces_stress_us_per_atom']:10.1f} us/atom  "
            f"(total {entry['energy_ms']:9.1f} / "
            f"{entry['energy_forces_stress_ms']:9.1f} ms)"
        )
    if parsed.output is not None:
        destination = parsed.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=1) + "\n")
        print(f"record written to {destination}")


args = None  # set below: the oracle loader takes a namespace of its own

if __name__ == "__main__":
    from types import SimpleNamespace

    parsed = make_parser().parse_args()
    args = SimpleNamespace(
        oracle_graph_electrostatics=parsed.oracle_graph_electrostatics,
        oracle_mace=parsed.oracle_mace,
        model_dir=pathlib.Path(R.DEFAULT_MODEL_DIR),
    )
    torch_mod, gto = R.load_oracle(args)
    main(parsed)
