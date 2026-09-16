#!/usr/bin/env python3
"""Cross-check MACE-PyTorch and Symmetrix on Materials Project structures."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


RESULT_PREFIX = "SYMMETRIX_MP_CORRECTNESS_RESULT="
DEFAULT_SEED = 20260806
DATASET_RELEASE = {
    "release": "MACEField 1.0.2",
    "url": "https://github.com/mdi-group/mace-field/releases/tag/1.0.2",
    "source_asset": "MP-dielectric-ferroelectric-replay-dataset.xyz",
    "source_sha256": "4d7c3d3e8642b5691ccfb93bf3a41b7b89425d9018eeedef8cb73efabcfd350a",
}

PROFILES = {
    "torch_cpu_f64": {
        "kind": "torch",
        "dtype": "float64",
        "execution_space": "CPU",
    },
    "serial_f64": {
        "kind": "symmetrix",
        "dtype": "float64",
        "extension": "cuda",
        "use_kokkos": False,
        "streamed_edges": "materialized",
    },
    "openmp_all_f64": {
        "kind": "symmetrix",
        "dtype": "float64",
        "extension": "openmp",
        "use_kokkos": True,
        "streamed_edges": "all_interactions",
        "m0": "runtime",
        "m1": "retained",
    },
    "cuda_all_retained_f32": {
        "kind": "symmetrix",
        "dtype": "float32",
        "extension": "cuda",
        "use_kokkos": True,
        "streamed_edges": "all_interactions",
        "m0": "runtime",
        "m1": "retained",
    },
    "cuda_all_standard_m0_f32": {
        "kind": "symmetrix",
        "dtype": "float32",
        "extension": "cuda",
        "use_kokkos": True,
        "streamed_edges": "all_interactions",
        "m0": "standard",
        "m1": "retained",
    },
    "cuda_all_standard_m0_m1_f32": {
        "kind": "symmetrix",
        "dtype": "float32",
        "extension": "cuda",
        "use_kokkos": True,
        "streamed_edges": "all_interactions",
        "m0": "standard",
        "m1": "recompute",
    },
    "cuda_factorized_f32": {
        "kind": "symmetrix",
        "dtype": "float32",
        "extension": "cuda",
        "use_kokkos": True,
        "streamed_edges": "factorized",
        "jit": "required",
        "m0": "automatic",
        "m1": "retained",
    },
}

TOLERANCES = {
    "float64": {
        "energy_abs_eV_per_atom": 2.0e-4,
        "forces_max_abs_eV_per_A": 3.0e-3,
        "stress_max_abs_eV_per_A3": 4.0e-3,
    },
    "float32": {
        "energy_abs_eV_per_atom": 5.0e-4,
        "forces_max_abs_eV_per_A": 5.0e-3,
        "stress_max_abs_eV_per_A3": 8.0e-3,
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _value(obj, name, default=None):
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def _bootstrap(source_root, extension):
    root = Path(source_root).resolve()
    extension = Path(extension).resolve()
    package_dir = root / "symmetrix/source/symmetrix"
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if type(finder).__name__ != "ScikitBuildRedirectingFinder"
    ]
    package_spec = importlib.util.spec_from_file_location(
        "symmetrix",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    native_spec = importlib.util.spec_from_file_location(
        "symmetrix.symmetrix", extension
    )
    package = importlib.util.module_from_spec(package_spec)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix"] = package
    sys.modules["symmetrix.symmetrix"] = native
    native_spec.loader.exec_module(native)
    package_spec.loader.exec_module(package)


def _endpoint_candidates(structures):
    groups = {}
    for dataset_index, atoms in enumerate(structures):
        pair = (atoms.info["nonpolar_mpid"], atoms.info["polar_mpid"])
        groups.setdefault(pair, []).append((dataset_index, atoms))

    by_material_id = {}
    for pair, entries in groups.items():
        entries.sort(key=lambda item: int(item[1].info["macefield_source_index"]))
        if len(entries) != 10:
            raise ValueError(f"MP path {pair} has {len(entries)} images, expected 10")
        for endpoint, material_id, entry in (
            ("nonpolar", pair[0], entries[0]),
            ("polar", pair[1], entries[-1]),
        ):
            by_material_id.setdefault(material_id, []).append(
                {
                    "material_id": material_id,
                    "endpoint": endpoint,
                    "path": pair,
                    "dataset_index": entry[0],
                    "atoms": entry[1],
                }
            )
    return by_material_id


def _sample_endpoints(structures, count, seed):
    candidates = _endpoint_candidates(structures)
    if count > len(candidates):
        raise ValueError(
            f"requested {count} structures but only {len(candidates)} unique MP IDs exist"
        )
    rng = random.Random(seed)
    material_ids = rng.sample(sorted(candidates), count)
    return [rng.choice(candidates[material_id]) for material_id in material_ids]


def _snapshot_sample(dataset, output, count, seed):
    from ase.io import read, write

    structures = read(dataset, index=":")
    selected = _sample_endpoints(structures, count, seed)
    snapshot = []
    manifest = []
    for selection_index, item in enumerate(selected):
        atoms = item["atoms"].copy()
        atoms.info["benchmark_material_id"] = item["material_id"]
        atoms.info["benchmark_endpoint"] = item["endpoint"]
        atoms.info["benchmark_selection_index"] = selection_index
        snapshot.append(atoms)
        manifest.append(
            {
                "selection_index": selection_index,
                "material_id": item["material_id"],
                "endpoint": item["endpoint"],
                "path": list(item["path"]),
                "dataset_index": item["dataset_index"],
                "source_index": int(atoms.info["macefield_source_index"]),
                "formula": atoms.get_chemical_formula(),
                "atoms": len(atoms),
                "atomic_numbers": sorted(set(int(z) for z in atoms.numbers)),
            }
        )
    write(output, snapshot, format="extxyz")
    return manifest


def _configure_evaluator(evaluator, profile):
    if "m0" in profile:
        evaluator._set_standard_m0_executor(profile["m0"])
    if "m1" in profile:
        if profile["m1"] == "recompute":
            evaluator._set_m1_recompute_tile_channels(32)
        evaluator._set_m1_polynomial_policy(profile["m1"])


def _selection_report(calculator, profile):
    if profile["kind"] != "symmetrix":
        return {}
    evaluator = calculator.evaluator
    report = {
        "streamed_edges": calculator.streamed_edges,
        "m0": _value(evaluator, "standard_m0_selected_executor"),
        "m1": _value(evaluator, "m1_polynomial_policy"),
        "jit_status": calculator.jit_status,
        "jit_artifact_id": calculator.jit_artifact_id,
        "factorized_execution_profile": _value(
            evaluator, "factorized_execution_profile"
        ),
    }
    expected_m0 = profile.get("m0")
    if expected_m0 == "standard" and report["m0"] != "standard":
        raise RuntimeError(f"requested standard M0, selected {report['m0']}")
    if expected_m0 == "runtime" and report["m0"] != "runtime":
        raise RuntimeError(f"requested runtime M0, selected {report['m0']}")
    if profile.get("m1") and report["m1"] != profile["m1"]:
        raise RuntimeError(f"requested M1 {profile['m1']}, selected {report['m1']}")
    if profile.get("jit") == "required" and report["jit_status"] not in {
        "built",
        "cached",
    }:
        raise RuntimeError(
            "required JIT specialization was not selected: " f"{report['jit_status']}"
        )
    return report


def _evaluate(calculator, atoms, warmups, samples, synchronize):
    import numpy as np
    from ase.calculators.calculator import all_changes

    properties = ["energy", "forces", "stress"]
    for _ in range(warmups):
        calculator.calculate(atoms, properties=properties, system_changes=all_changes)
        synchronize()
    elapsed = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        calculator.calculate(atoms, properties=properties, system_changes=all_changes)
        synchronize()
        elapsed.append((time.perf_counter_ns() - started) / 1.0e6)
    return {
        "energy_eV": float(calculator.results["energy"]),
        "forces_eV_per_A": np.asarray(
            calculator.results["forces"], dtype=np.float64
        ).tolist(),
        "stress_eV_per_A3": np.asarray(calculator.results["stress"], dtype=np.float64)
        .reshape(-1)
        .tolist(),
        "timing_ms": {
            "median": statistics.median(elapsed),
            "minimum": min(elapsed),
            "maximum": max(elapsed),
            "samples": elapsed,
        },
    }


def _worker(args):
    from ase.io import read

    profile = PROFILES[args.profile]
    structures = read(args.structures, index=":")
    native = None
    if profile["kind"] == "torch":
        import torch
        from mace.calculators.mace import MACECalculator

        calculator = MACECalculator(
            model_paths=[str(args.checkpoint.resolve())],
            device="cpu",
            default_dtype="float64",
            head=args.head,
        )

        def synchronize():
            return None

        runtime = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
    else:
        _bootstrap(args.source_root, args.extension)
        from symmetrix import Symmetrix
        from symmetrix import symmetrix as native

        if profile["use_kokkos"] and not native._kokkos_is_initialized():
            native._init_kokkos()
        calculator = Symmetrix(
            args.model_json.resolve(),
            use_kokkos=profile["use_kokkos"],
            dtype=profile["dtype"],
            streamed_edges=profile["streamed_edges"],
            jit=profile.get("jit", "off"),
        )
        _configure_evaluator(calculator.evaluator, profile)

        def synchronize():
            return None

        runtime = {
            "extension": _provenance(args.extension),
            "execution_space": (
                native._kokkos_default_execution_space()
                if profile["use_kokkos"]
                else "Serial evaluator"
            ),
        }

    results = []
    for atoms in structures:
        result = _evaluate(calculator, atoms, args.warmups, args.samples, synchronize)
        result.update(
            {
                "material_id": atoms.info["benchmark_material_id"],
                "endpoint": atoms.info["benchmark_endpoint"],
                "formula": atoms.get_chemical_formula(),
                "atoms": len(atoms),
            }
        )
        results.append(result)
    report = {
        "profile": args.profile,
        "head": args.head,
        "dtype": profile["dtype"],
        "runtime": runtime,
        "selection": _selection_report(calculator, profile),
        "warmups": args.warmups,
        "samples": args.samples,
        "results": results,
    }
    del calculator
    gc.collect()
    if native is not None and native._kokkos_is_initialized():
        native._finalize_kokkos()
    print(RESULT_PREFIX + json.dumps(report, separators=(",", ":")))


def _error_metrics(reference, candidate):
    import numpy as np

    if reference["material_id"] != candidate["material_id"]:
        raise ValueError("reference and candidate material IDs differ")
    natoms = int(reference["atoms"])
    reference_forces = np.asarray(reference["forces_eV_per_A"], dtype=float)
    candidate_forces = np.asarray(candidate["forces_eV_per_A"], dtype=float)
    reference_stress = np.asarray(reference["stress_eV_per_A3"], dtype=float)
    candidate_stress = np.asarray(candidate["stress_eV_per_A3"], dtype=float)
    if reference_forces.shape != candidate_forces.shape:
        raise ValueError("force shapes differ")
    if reference_stress.shape != candidate_stress.shape:
        raise ValueError("stress shapes differ")
    return {
        "energy_abs_eV": abs(candidate["energy_eV"] - reference["energy_eV"]),
        "energy_abs_eV_per_atom": abs(candidate["energy_eV"] - reference["energy_eV"])
        / natoms,
        "forces_max_abs_eV_per_A": float(
            np.max(np.abs(candidate_forces - reference_forces))
        ),
        "stress_max_abs_eV_per_A3": float(
            np.max(np.abs(candidate_stress - reference_stress))
        ),
    }


def _compare(reference_report, candidate_report):
    references = {row["material_id"]: row for row in reference_report["results"]}
    tolerance = TOLERANCES[candidate_report["dtype"]]
    rows = []
    for candidate in candidate_report["results"]:
        metrics = _error_metrics(references[candidate["material_id"]], candidate)
        passed = all(metrics[name] <= limit for name, limit in tolerance.items())
        rows.append(
            {
                "material_id": candidate["material_id"],
                "formula": candidate["formula"],
                "passed": passed,
                **metrics,
            }
        )
    maxima = {name: max(row[name] for row in rows) for name in tolerance}
    return {
        "profile": candidate_report["profile"],
        "tolerances": tolerance,
        "maxima": maxima,
        "passed": len(rows) == len(references) and all(row["passed"] for row in rows),
        "structures": rows,
    }


def _markdown(report):
    lines = [
        "# Materials Project MACE correctness benchmark",
        "",
        f"Overall result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        f"Sample seed: `{report['sample']['seed']}`; structures: {report['sample']['count']}",
        "",
        "| Profile | Result | max dE (eV/atom) | max dF (eV/A) | max dStress (eV/A^3) | median ms/structure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    profile_reports = {row["profile"]: row for row in report["profile_reports"]}
    for comparison in report["comparisons"]:
        timings = [
            row["timing_ms"]["median"]
            for row in profile_reports[comparison["profile"]]["results"]
        ]
        maxima = comparison["maxima"]
        lines.append(
            f"| {comparison['profile']} | {'PASS' if comparison['passed'] else 'FAIL'} "
            f"| {maxima['energy_abs_eV_per_atom']:.3e} "
            f"| {maxima['forces_max_abs_eV_per_A']:.3e} "
            f"| {maxima['stress_max_abs_eV_per_A3']:.3e} "
            f"| {statistics.median(timings):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _run(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.parent / f"materials-project-sample-seed-{args.seed}.extxyz"
    sample_manifest = _snapshot_sample(args.dataset, snapshot, args.count, args.seed)
    extensions = {
        "cuda": args.cuda_extension.resolve(),
        "openmp": args.openmp_extension.resolve(),
    }
    profile_reports = []
    failures = []
    logs = output.parent / f"{output.stem}-logs"
    logs.mkdir(parents=True, exist_ok=True)
    for profile_name in args.profiles:
        profile = PROFILES[profile_name]
        extension = extensions.get(profile.get("extension", "cuda"))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--profile",
            profile_name,
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--model-json",
            str(args.model_json.resolve()),
            "--structures",
            str(snapshot),
            "--head",
            args.head,
            "--source-root",
            str(args.source_root.resolve()),
            "--extension",
            str(extension),
            "--warmups",
            str(args.warmups),
            "--samples",
            str(args.samples),
        ]
        completed = subprocess.run(
            command,
            cwd=args.source_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=os.environ.copy(),
        )
        (logs / f"{profile_name}.stdout.log").write_text(completed.stdout)
        (logs / f"{profile_name}.stderr.log").write_text(completed.stderr)
        parsed = [
            json.loads(line.removeprefix(RESULT_PREFIX))
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        if completed.returncode != 0 or len(parsed) != 1:
            failures.append(
                {
                    "profile": profile_name,
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-4000:],
                }
            )
            print(f"{profile_name}: FAILED", flush=True)
        else:
            profile_reports.append(parsed[0])
            print(f"{profile_name}: complete", flush=True)

    by_profile = {row["profile"]: row for row in profile_reports}
    oracle = by_profile.get("torch_cpu_f64")
    comparisons = []
    if oracle is not None:
        comparisons = [
            _compare(oracle, report)
            for report in profile_reports
            if report["profile"] != "torch_cpu_f64"
        ]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "benchmark": "Materials Project MACE cross-backend correctness",
        "benchmark_driver": _provenance(Path(__file__)),
        "source_commit": source_commit,
        "passed": (
            not failures
            and oracle is not None
            and len(profile_reports) == len(args.profiles)
            and all(row["passed"] for row in comparisons)
        ),
        "sample": {
            "seed": args.seed,
            "count": args.count,
            "public_provenance": DATASET_RELEASE,
            "source": _provenance(args.dataset),
            "snapshot": _provenance(snapshot),
            "structures": sample_manifest,
        },
        "checkpoint": _provenance(args.checkpoint),
        "model_json": _provenance(args.model_json),
        "head": args.head,
        "extensions": {name: _provenance(path) for name, path in extensions.items()},
        "requested_profiles": args.profiles,
        "profile_reports": profile_reports,
        "comparisons": comparisons,
        "failures": failures,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".md").write_text(_markdown(report))
    if not report["passed"]:
        raise SystemExit(1)


def _extract(args):
    _bootstrap(args.source_root, args.extension)
    from symmetrix.extract_mace_data import extract_mace_data

    data = extract_mace_data(
        args.checkpoint.resolve(), head=args.head, radial_format="compact"
    )
    args.output.resolve().write_text(json.dumps(data, separators=(",", ":")) + "\n")


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--source-root", type=Path, required=True)
    extract.add_argument("--extension", type=Path, required=True)
    extract.add_argument("--head")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--profile", choices=tuple(PROFILES), required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--model-json", type=Path, required=True)
    worker.add_argument("--structures", type=Path, required=True)
    worker.add_argument("--source-root", type=Path, required=True)
    worker.add_argument("--extension", type=Path, required=True)
    worker.add_argument("--head", required=True)
    worker.add_argument("--warmups", type=int, default=1)
    worker.add_argument("--samples", type=int, default=3)

    run = subparsers.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--model-json", type=Path, required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--cuda-extension", type=Path, required=True)
    run.add_argument("--openmp-extension", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--head", required=True)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--count", type=int, default=10)
    run.add_argument(
        "--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES)
    )
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--samples", type=int, default=3)
    run.add_argument("--timeout", type=float, default=1800.0)
    return parser


def main():
    args = _parser().parse_args()
    if hasattr(args, "warmups") and (args.warmups < 0 or args.samples < 1):
        raise SystemExit("warmups must be nonnegative and samples must be positive")
    if args.command == "extract":
        _extract(args)
    elif args.command == "worker":
        _worker(args)
    else:
        if args.count < 1:
            raise SystemExit("count must be positive")
        _run(args)


if __name__ == "__main__":
    main()
