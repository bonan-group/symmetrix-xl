#!/usr/bin/env python3
"""Fresh-process qualification for standard M0 and M1 recomputation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


PROFILES = {
    "retained": ("runtime", "retained"),
    "standard_m0": ("standard", "retained"),
    "standard_m0_m1": ("standard", "recompute"),
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(path):
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _record(report, kind, profile, trial, order):
    system = report["systems"][0]
    mode = system["modes"]["all_interactions"]
    m0 = mode["factorized"]["m0_module"]
    m1 = mode["m1_polynomial"]
    requested_m0, requested_m1 = PROFILES[profile]
    expected_m0 = "runtime" if requested_m0 == "runtime" else "standard"
    if m0["selected_executor"] != expected_m0:
        raise RuntimeError(f"{kind}/{profile} selected M0 {m0['selected_executor']}")
    if m1["policy"] != requested_m1:
        raise RuntimeError(f"{kind}/{profile} selected M1 {m1['policy']}")
    if expected_m0 == "standard":
        for tensor in ("poly_values", "poly_adjoints"):
            if m0[tensor]["active_bytes"] or m0[tensor]["capacity_bytes"]:
                raise RuntimeError(f"{kind}/{profile} retained M0 {tensor}")
    if requested_m1 == "recompute":
        for tensor in ("poly_values", "poly_adjoints"):
            if m1[tensor]["active_bytes"] or m1[tensor]["capacity_bytes"]:
                raise RuntimeError(f"{kind}/{profile} retained M1 {tensor}")
    return {
        "kind": kind,
        "profile": profile,
        "trial": trial,
        "order": order,
        "repeat": system["supercell_repeat"],
        "atoms": system["atoms"],
        "directed_edges": system["directed_edges"],
        "timing_ms": mode["timing"],
        "gpu_process_memory_mib": mode["gpu_process_memory_mib"],
        "energy_eV": mode["energy_eV"],
        "m0": m0,
        "m1": m1,
        "phi_capacity_bytes": mode["factorized"]["tensor_capacity_bytes"],
    }


def _aggregate(records):
    result = []
    keys = sorted({(row["kind"], row["profile"], row["atoms"]) for row in records})
    for kind, profile, atoms in keys:
        selected = [
            row
            for row in records
            if (row["kind"], row["profile"], row["atoms"]) == (kind, profile, atoms)
        ]
        medians = [row["timing_ms"]["median_ms"] for row in selected]
        memories = [row["gpu_process_memory_mib"]["peak_sampled"] for row in selected]
        result.append(
            {
                "kind": kind,
                "profile": profile,
                "atoms": atoms,
                "process_medians_ms": medians,
                "median_ms": statistics.median(medians),
                "process_peak_gpu_memory_mib": memories,
                "median_peak_gpu_memory_mib": statistics.median(memories),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-model", type=Path, required=True)
    parser.add_argument("--field-model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", nargs="+", type=int, default=[2, 6, 11, 12, 20])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.trials < 1 or args.warmups < 0 or args.samples < 1:
        parser.error("trials/samples must be positive and warmups nonnegative")

    root = args.source_root.resolve()
    extension = args.extension.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = output.parent / f"{output.stem}-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "standard": args.standard_model.resolve(),
        "field": args.field_model.resolve(),
    }
    env = os.environ.copy()
    env.update(
        {
            "SYMMETRIX_SOURCE_ROOT": str(root),
            "SYMMETRIX_EXTENSION": str(extension),
        }
    )
    benchmark = root / "benchmarks/standard_mace_streamed_benchmark.py"
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    records = []
    failures = []
    rng = random.Random(args.seed)
    for repeat in args.repeats:
        for trial in range(1, args.trials + 1):
            cases = [
                ("standard", "retained"),
                ("standard", "standard_m0"),
                ("standard", "standard_m0_m1"),
                ("field", "retained"),
                ("field", "standard_m0"),
            ]
            rng.shuffle(cases)
            for order, (kind, profile) in enumerate(cases, start=1):
                m0, m1 = PROFILES[profile]
                raw = (
                    raw_dir
                    / f"r{repeat}-t{trial:02d}-{order:02d}-{kind}-{profile}.json"
                )
                command = [
                    sys.executable,
                    str(benchmark),
                    str(models[kind]),
                    "--backend",
                    "kokkos",
                    "--dtype",
                    "float32",
                    "--modes",
                    "all_interactions",
                    "--sizes",
                    str(repeat),
                    "--warmups",
                    str(args.warmups),
                    "--repeats",
                    str(args.samples),
                    "--factorized-m0-executor",
                    m0,
                    "--m1-polynomial-policy",
                    m1,
                    "--m1-recompute-tile-channels",
                    "32",
                    "--source-commit",
                    source_commit,
                    "--output",
                    str(raw),
                ]
                if raw.is_file():
                    report = json.loads(raw.read_text())
                    record = _record(report, kind, profile, trial, order)
                    records.append(record)
                    print(
                        f"cached r{repeat} t{trial} {kind} {profile}: "
                        f"{record['timing_ms']['median_ms']:.6f} ms",
                        flush=True,
                    )
                    continue
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                if completed.returncode != 0:
                    failure = {
                        "kind": kind,
                        "profile": profile,
                        "repeat": repeat,
                        "atoms": 4 * repeat**3,
                        "trial": trial,
                        "order": order,
                        "returncode": completed.returncode,
                        "stderr_tail": completed.stderr[-4000:],
                    }
                    failures.append(failure)
                    (raw_dir / f"{raw.stem}.stderr.log").write_text(completed.stderr)
                    print(
                        f"FAILED r{repeat} t{trial} {kind} {profile}: "
                        f"exit {completed.returncode}",
                        flush=True,
                    )
                    continue
                report = json.loads(raw.read_text())
                record = _record(report, kind, profile, trial, order)
                records.append(record)
                print(
                    f"r{repeat} t{trial} {kind} {profile}: "
                    f"{record['timing_ms']['median_ms']:.6f} ms",
                    flush=True,
                )

    report = {
        "benchmark": "JIT-free all-interactions standard M0 and M1 recomputation",
        "source_commit": source_commit,
        "source_root": str(root),
        "extension": _provenance(extension),
        "models": {kind: _provenance(path) for kind, path in models.items()},
        "execution_space": "Cuda",
        "mode": "all_interactions",
        "dtype": "float32",
        "repeats": args.repeats,
        "trials": args.trials,
        "warmups": args.warmups,
        "samples": args.samples,
        "random_seed": args.seed,
        "profiles": PROFILES,
        "aggregates": _aggregate(records),
        "records": records,
        "failures": failures,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "aggregates": report["aggregates"]}))


if __name__ == "__main__":
    main()
