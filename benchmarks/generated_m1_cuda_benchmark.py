"""Benchmark generated and frozen standard M1 CUDA execution."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

for variable in (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "symmetrix/test"))

from compact_r1_model import foundation_r1_test_model  # noqa: E402
from symmetrix.mace_execution_contract import make_generated_m1_contract  # noqa: E402

from symmetrix import Symmetrix  # noqa: E402
from symmetrix import symmetrix as native_symmetrix  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standard_model(implementation, model_path=None):
    if model_path is not None:
        model = json.loads(model_path.read_text())
        if implementation == "frozen":
            model["execution_contracts"].pop("M1", None)
        else:
            model["execution_contracts"]["M1"] = make_generated_m1_contract(
                channels=model["num_channels"],
                type_count=model["num_elements"],
                input_l_max=model["l_max"],
                correlation=max(len(term) for term in model["M1_monomials"]),
                monomials=model["M1_monomials"],
            )
        return model

    contracts = REPOSITORY / "symmetrix/test/data/execution_contracts"
    r1_contract = json.loads(
        (contracts / "jit_r1_mace_omat0_small_contract.json").read_text()
    )
    m0_contract = json.loads((contracts / "standard_m0_contract.json").read_text())
    m0_contract["output_l_max"] = 0
    m0_contract["output_components"] = 1
    m0_contract["monomial_groups"] = m0_contract["monomial_groups"][:1]
    m0_contract["structure_fingerprint"] = (
        "sha256:3b4ab2d979488798a05c07b5d54c4c96c1026da3ee1eb93069fd2bacec76ac90"
    )
    m0_contract["semantic_fingerprint"] = (
        "sha256:9fe598ceffb71e61d162c0afbdbe5727a0da7997dbca3a531a46b63603d7af85"
    )
    m0_contract["generation_fingerprint"] = (
        "sha256:f974e4af32dfb328e06fb6655b965c4cc5c0cad50a2013223630f9960f59d398"
    )
    r0_contract = json.loads((contracts / "standard_r0_contract.json").read_text())
    model = foundation_r1_test_model(
        r1_contract,
        atomic_numbers=(7, 13),
        m0_contract=m0_contract,
        r0_contract=r0_contract,
    )
    model["M1_monomials"] = m0_contract["monomial_groups"][0]["terms"]
    model["M1_weights"] = {
        str(element): {
            str(channel): [
                0.01 * (1 + element) * (1 + (term + channel) % 7)
                for term in range(len(model["M1_monomials"]))
            ]
            for channel in range(model["num_channels"])
        }
        for element in range(model["num_elements"])
    }
    if implementation == "generated":
        model["execution_contracts"]["M1"] = make_generated_m1_contract(
            channels=model["num_channels"],
            type_count=model["num_elements"],
            input_l_max=model["l_max"],
            correlation=max(len(term) for term in model["M1_monomials"]),
            monomials=model["M1_monomials"],
        )
    return model


def _counter_snapshot(evaluator):
    names = (
        "factorized_prepared_evaluation_count",
        "factorized_fallback_evaluation_count",
        "m1_device_terms_launch_count",
        "m1_recompute_forward_launch_count",
        "m1_recompute_reverse_launch_count",
        "generated_m1_module_forward_launch_count",
        "generated_m1_module_reverse_launch_count",
        "standard_m1_module_forward_launch_count",
        "standard_m1_module_reverse_launch_count",
    )
    return {name: int(getattr(evaluator, name, 0)) for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", choices=("generated", "frozen"), required=True
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--structure", choices=("aln", "srtio3"), default="aln")
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--rattle", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--neighbor-skin", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("size and repeats must be positive and warmups non-negative")

    model_path = args.model.resolve() if args.model is not None else None
    model = _standard_model(args.implementation, model_path)
    if args.structure == "srtio3":
        from ase.spacegroup import crystal

        atoms = crystal(
            symbols=["Sr", "Ti", "O"],
            basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
            spacegroup=221,
            cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
        ).repeat((args.size, args.size, args.size))
        if args.rattle:
            atoms.positions += np.random.default_rng(args.seed).normal(
                scale=args.rattle, size=atoms.positions.shape
            )
    else:
        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(
            (args.size, args.size, args.size)
        )
    temporary_root = os.environ.get("SYMMETRIX_BENCHMARK_TMPDIR")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=temporary_root
    ) as model_file:
        json.dump(model, model_file)
        model_file.flush()
        calculator = Symmetrix(
            model_file.name,
            dtype=args.dtype,
            streamed_edges="direct",
            m1_polynomial_policy="recompute",
            neighbor_skin=args.neighbor_skin,
        )
    evaluator = calculator.evaluator
    evaluator._set_m1_polynomial_policy("recompute")
    inputs = calculator._mace_inputs(atoms)
    graph = evaluator._prepare_factorized_graph(*inputs[:5])
    evaluation_args = (graph, np.asarray(inputs[5]).reshape(-1), inputs[6])

    for _ in range(args.warmups):
        evaluator._compute_prepared_factorized(*evaluation_args)
    before = _counter_snapshot(evaluator)
    samples_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        evaluator._compute_prepared_factorized(*evaluation_args)
        samples_ms.append(1000.0 * (time.perf_counter() - start))
    after = _counter_snapshot(evaluator)
    result = calculator._collect_mace_results(
        atoms, inputs, ("energy", "forces", "stress"), graph
    )
    deltas = {name: after[name] - before[name] for name in before}
    expected_generated = args.implementation == "generated"
    if deltas["factorized_prepared_evaluation_count"] != args.repeats:
        raise RuntimeError("prepared evaluation counter does not match repeats")
    if deltas["factorized_fallback_evaluation_count"] != 0:
        raise RuntimeError("prepared execution fell back")
    for direction in ("forward", "reverse"):
        generated = deltas[f"generated_m1_module_{direction}_launch_count"]
        frozen = deltas[f"standard_m1_module_{direction}_launch_count"]
        if expected_generated and (generated != args.repeats or frozen != 0):
            raise RuntimeError(
                f"generated M1 {direction} routing was not exclusive: "
                f"generated={generated}, frozen={frozen}"
            )
        if not expected_generated:
            generic_forward = deltas["m1_device_terms_launch_count"]
            routed = frozen + (generic_forward if direction == "forward" else 0)
            if routed != args.repeats or generated != 0:
                raise RuntimeError(
                    f"legacy M1 {direction} routing was not exclusive: "
                    f"generated={generated}, frozen={frozen}, "
                    f"generic_forward={generic_forward}, "
                    f"standard_ready={evaluator.standard_m1_module_ready}"
                )

    extension = Path(native_symmetrix.__file__).resolve()
    median_ms = statistics.median(samples_ms)
    report = {
        "implementation": args.implementation,
        "dtype": args.dtype,
        "architecture": {
            "channels": model["num_channels"],
            "type_count": model["num_elements"],
            "l_max": model["l_max"],
            "max_ell": model["L_max"],
            "correlation": max(len(term) for term in model["M1_monomials"]),
            "m1_terms": len(model["M1_monomials"]),
        },
        "workload": {
            "structure": args.structure,
            "repeat": args.size,
            "rattle_A": args.rattle if args.structure == "srtio3" else 0.0,
            "seed": args.seed if args.structure == "srtio3" else None,
            "properties": ["energy", "forces", "stress"],
            "atoms": len(atoms),
            "directed_edges": len(inputs[6]),
            "model_cutoff_A": model["r_cut"],
            "neighbor_skin_A": args.neighbor_skin,
            "effective_cutoff_A": model["r_cut"] + args.neighbor_skin,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "timing": {
            "median_ms": median_ms,
            "minimum_ms": min(samples_ms),
            "maximum_ms": max(samples_ms),
            "median_us_per_atom": 1000.0 * median_ms / len(atoms),
            "samples_ms": samples_ms,
        },
        "routing": {
            "jit_status": calculator.jit_status,
            "generated_m1_device_module_ready": bool(
                getattr(evaluator, "m1_device_module_ready", False)
            ),
            "standard_m1_module_ready": bool(evaluator.standard_m1_module_ready),
            "measured_counter_deltas": deltas,
        },
        "result_digest": {
            "energy_eV": float(result["energy"]),
            "force_linf_eV_per_A": float(np.max(np.abs(result["forces"]))),
            "stress_linf_eV_per_A3": float(np.max(np.abs(result["stress"]))),
        },
        "runtime": {
            "input_model": str(model_path) if model_path is not None else "synthetic",
            "input_model_sha256": _sha256(model_path)
            if model_path is not None
            else None,
            "python": sys.executable,
            "extension": str(extension),
            "extension_sha256": _sha256(extension),
            "kokkos_execution_space": native_symmetrix._kokkos_default_execution_space(),
            "backend_build_info": native_symmetrix._backend_build_info(),
        },
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.write_text(serialized + "\n")
    del result, evaluation_args, inputs, evaluator, calculator
    gc.collect()
    if native_symmetrix._kokkos_is_initialized():
        native_symmetrix._finalize_kokkos()


if __name__ == "__main__":
    main()
