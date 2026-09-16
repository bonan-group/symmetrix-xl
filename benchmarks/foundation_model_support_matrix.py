#!/usr/bin/env python3
"""Qualify released MACE checkpoints across Symmetrix direct backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import traceback
from pathlib import Path

SCHEMA = "symmetrix.foundation-model-support-result"
VERSION = 1
DEFAULT_MANIFEST = Path(__file__).with_name("foundation_model_support_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema") != "symmetrix.foundation-model-support-matrix":
        raise ValueError("invalid Foundation support manifest schema")
    if value.get("version") != 1:
        raise ValueError("unsupported Foundation support manifest version")
    ids = [row.get("id") for row in value.get("models", ())]
    if not ids or len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("manifest model IDs must be nonempty and unique")
    structures = value.get("structures", {})
    for row in value["models"]:
        classified = set(row.get("chemistry", {}))
        if not classified or not classified.issubset(structures):
            raise ValueError(
                f"{row['id']} must classify at least one known validation structure"
            )
        checksum = row.get("sha256")
        if checksum is not None and (
            len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum)
        ):
            raise ValueError(f"{row['id']} has an invalid SHA-256")
    for probe in value.get("capacity_probes", ()):
        if probe.get("model") not in set(ids):
            raise ValueError(f"capacity probe has unknown model {probe.get('model')!r}")
        if probe.get("structure") not in structures:
            raise ValueError(
                f"capacity probe has unknown structure {probe.get('structure')!r}"
            )
        if probe.get("dtype") not in {"float32", "float64"}:
            raise ValueError("capacity probe has invalid dtype")
        if probe.get("backend") not in {"cpu", "cuda", "hip"}:
            raise ValueError("capacity probe has invalid backend")
        if probe.get("expected_m1_tile_channels") not in {None, 1, 4, 8, 16, 32}:
            raise ValueError("capacity probe has invalid expected M1 tile width")
    cross_limits = value.get("cross_precision_tolerances", {})
    if set(cross_limits) != {
        "energy_abs_eV_per_atom",
        "forces_max_abs_eV_per_A",
        "stress_max_abs_eV_per_A3",
    }:
        raise ValueError("cross-precision tolerances are incomplete")
    return value


def chemistry_status(model: dict, structure: str) -> tuple[bool, str | None]:
    value = model["chemistry"][structure]
    if isinstance(value, bool):
        return value, None if value else "not supported"
    return bool(value.get("applicable")), value.get("reason")


def case_key(
    model: str, structure: str, dtype: str, backend: str, profile: str = "speed"
) -> str:
    suffix = "" if profile == "speed" else f"/{profile}"
    return f"{model}/{structure}/{dtype}/{backend}{suffix}"


def matrix_cases(manifest: dict, models, structures, dtypes, backends):
    selected = set(models or (row["id"] for row in manifest["models"]))
    known = {row["id"] for row in manifest["models"]}
    unknown = selected - known
    if unknown:
        raise ValueError("unknown models: " + ", ".join(sorted(unknown)))
    for model in manifest["models"]:
        if model["id"] not in selected:
            continue
        for structure in structures:
            if structure not in model["chemistry"]:
                continue
            for dtype in dtypes:
                for backend in backends:
                    yield model, structure, dtype, backend, "speed"
    for probe in manifest.get("capacity_probes", ()):
        if (
            probe["model"] in selected
            and probe["structure"] in structures
            and probe["dtype"] in dtypes
            and probe["backend"] in backends
        ):
            model = next(
                row for row in manifest["models"] if row["id"] == probe["model"]
            )
            yield (
                model,
                probe["structure"],
                probe["dtype"],
                probe["backend"],
                "capacity",
            )


def completed_keys(path: Path, retry_failures: bool = False) -> set[str]:
    if not path.is_file():
        return set()
    result = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
        if not retry_failures or row.get("status") in {"passed", "not_applicable"}:
            result.add(row["case_key"])
    return result


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_structure(name: str, specification: dict):
    import numpy as np
    from ase import Atoms
    from ase.build import bulk

    if name == "C64":
        atoms = bulk("C", "diamond", a=3.567, cubic=True).repeat((2, 2, 2))
    elif name == "STO40":
        a = 3.905
        atoms = Atoms(
            "SrTiO3",
            positions=(
                (0, 0, 0),
                (a / 2, a / 2, a / 2),
                (a / 2, a / 2, 0),
                (a / 2, 0, a / 2),
                (0, a / 2, a / 2),
            ),
            cell=np.eye(3) * a,
            pbc=True,
        ).repeat((2, 2, 2))
    elif name == "H2O":
        from ase.build import molecule

        atoms = molecule("H2O")
        atoms.center(vacuum=5.0)
    else:
        raise ValueError(f"unknown structure {name!r}")
    rng = np.random.default_rng(int(specification["seed"]))
    atoms.positions += rng.normal(
        scale=float(specification["displacement_sigma_A"]), size=atoms.positions.shape
    )
    if len(atoms) != int(specification["atoms"]):
        raise RuntimeError(f"{name} produced {len(atoms)} atoms")
    return atoms


def numerical_errors(reference: dict, candidate: dict, atoms: int) -> dict:
    import numpy as np

    return {
        "energy_abs_eV_per_atom": abs(candidate["energy"] - reference["energy"])
        / atoms,
        "forces_max_abs_eV_per_A": float(
            np.max(np.abs(candidate["forces"] - reference["forces"]))
        ),
        "stress_max_abs_eV_per_A3": float(
            np.max(np.abs(candidate["stress"] - reference["stress"]))
        ),
    }


def cross_precision_assessment(
    reference: dict, candidate: dict, atoms: int, tolerances: dict
) -> tuple[dict, list[str]]:
    errors = numerical_errors(reference, candidate, atoms)
    failures = [
        f"direct FP32 versus direct FP64 {name} exceeds tolerance"
        for name, value in errors.items()
        if value > tolerances[name]
    ]
    return {
        "status": "checked",
        "errors": errors,
        "tolerances": tolerances,
    }, failures


def direct_identity(calculator, backend: str, profile: str = "speed") -> dict:
    evaluator = calculator.evaluator
    fallback = int(getattr(evaluator, "factorized_fallback_evaluation_count", -1))
    ordinary = hasattr(evaluator, "factorized_jit_forward_launch_count")
    if ordinary:
        launches = {
            "forward": int(evaluator.factorized_jit_forward_launch_count),
            "reverse": int(evaluator.factorized_jit_reverse_launch_count),
        }
        ready = bool(getattr(evaluator, "factorized_jit_ready", False))
        executors = {
            "forward": getattr(
                evaluator, "factorized_selected_direct_forward_executor", None
            ),
            "reverse": getattr(
                evaluator, "factorized_selected_direct_reverse_executor", None
            ),
            "r0_implementation": getattr(evaluator, "r0_implementation", None),
            "r0_module_id": getattr(evaluator, "r0_module_id", None),
            "m0_implementation": getattr(evaluator, "m0_implementation", None),
            "m0_module_id": getattr(evaluator, "m0_module_id", None),
            "m1_recompute_tile_channels": int(
                getattr(evaluator, "m1_recompute_tile_channels", 0)
            ),
        }
        generated = launches["forward"] > 0 and launches["reverse"] > 0
        generated = generated and executors["forward"] == "jit_all"
        generated = generated and executors["reverse"] == "jit"
    else:
        launches = {
            "forward": int(
                getattr(evaluator, "execution_mh1_generated_forward_launch_count", 0)
            ),
            "source_reverse": int(
                getattr(
                    evaluator, "execution_mh1_generated_source_reverse_launch_count", 0
                )
            ),
            "edge_reverse": int(
                getattr(
                    evaluator, "execution_mh1_generated_edge_reverse_launch_count", 0
                )
            ),
        }
        ready_names = (
            ("jit_mh1_cuda_plugin_ready", "jit_mh1_cuda_plugin_v4_ready")
            if backend in {"cuda", "hip"}
            else (
                "jit_mh1_host_plugin_ready",
                "jit_mh1_host_plugin_v4_ready",
                "jit_mh1_host_plugin_v5_ready",
            )
        )
        ready = any(bool(getattr(evaluator, name, False)) for name in ready_names)
        generated = all(value > 0 for value in launches.values())
        executors = {
            "backend": getattr(evaluator, "execution_mh1_execution_backend", None),
            "edge": getattr(evaluator, "mh1_edge_executor", None),
        }
    result = {
        "execution_profile": profile,
        "streamed_edges": getattr(evaluator, "streamed_edges_mode", None),
        "jit_status": calculator.jit_status,
        "jit_artifact_id": calculator.jit_artifact_id,
        "ready": ready,
        "fallback_evaluations": fallback,
        "generated_launches": launches,
        "executors": executors,
    }
    failures = []
    if result["streamed_edges"] != "direct":
        failures.append("direct streamed-edge execution was not selected")
    if result["jit_status"] not in {"built", "cached"} or not ready:
        failures.append("required generated artifact is not ready")
    if fallback != 0:
        failures.append(f"fallback evaluation count is {fallback}")
    if not generated:
        failures.append("generated forward/reverse execution was not observed")
    if ordinary and profile == "capacity":
        for stage in ("r0", "m0"):
            implementation = executors[f"{stage}_implementation"]
            module_id = executors[f"{stage}_module_id"]
            specialized = (
                {"builtin", "device_module"}
                if stage == "r0"
                else {"builtin", "device_module", "host_plugin"}
            )
            if implementation not in specialized:
                failures.append(
                    f"{stage.upper()} selected non-specialized implementation "
                    f"{implementation!r}"
                )
            if not module_id:
                failures.append(f"{stage.upper()} selected no specialization module ID")
    if not ordinary:
        family = executors["backend"] or ""
        expected_prefixes = {
            "cuda": ("generated_cuda", "pair_spline_v1_staged_device"),
            "hip": ("generated_hip", "pair_spline_v1_staged_device"),
        }.get(
            backend,
            (
                "generated_host",
                "pair_spline_v1_staged_host",
                "pair_spline_v1_staged_r_host",
            ),
        )
        if not family.startswith(expected_prefixes):
            failures.append(
                f"MH-1 selected backend family {family!r}, expected generated {backend}"
            )
        pair_backend = family.startswith("pair_spline_v1_")
        if pair_backend != (executors["edge"] == "pair_spline_v1"):
            failures.append(
                "MH-1 backend family and selected edge executor are inconsistent"
            )
    result["gate_failures"] = failures
    return result


def evaluate(calculator, atoms) -> dict:
    import numpy as np

    calculator.calculate(atoms, properties=("energy", "forces", "stress"))
    return {
        "energy": float(calculator.results["energy"]),
        "forces": np.asarray(calculator.results["forces"], dtype=np.float64),
        "stress": np.asarray(calculator.results["stress"], dtype=np.float64),
    }


def run_worker(args) -> dict:
    os.environ["SYMMETRIX_JIT_POLICY"] = "required"
    os.environ["SYMMETRIX_JIT_CACHE"] = str(args.jit_cache.resolve())
    os.environ["SYMMETRIX_BACKEND"] = args.backend_selector
    manifest = load_manifest(args.manifest)
    model = next(row for row in manifest["models"] if row["id"] == args.model_id)
    checkpoint_hash = sha256_file(args.checkpoint)
    expected = model.get("sha256")
    if expected and checkpoint_hash != expected:
        raise RuntimeError(
            f"checkpoint SHA-256 {checkpoint_hash} != expected {expected}"
        )

    import numpy as np

    import symmetrix
    from symmetrix import Symmetrix

    native = symmetrix.load_backend(args.backend_selector)
    build_info = getattr(native, "_backend_build_info", dict)()
    expected_backend = args.backend if args.backend in {"cuda", "hip"} else "cpu"
    if build_info and build_info.get("backend") != expected_backend:
        raise RuntimeError(
            f"loaded {build_info.get('backend')}, expected {expected_backend}"
        )
    atoms = build_structure(args.structure, manifest["structures"][args.structure])
    neighbor_skin = 0.5
    options = {
        "dtype": args.dtype,
        "use_kokkos": True,
        "execution_profile": args.profile,
        "neighbor_skin": neighbor_skin,
    }
    if args.profile == "capacity":
        options["_debug_execution_plan"] = "mh0-direct-capacity-retained"
    if model.get("head"):
        options["head"] = model["head"]
    direct = Symmetrix(args.compact_model, streamed_edges="direct", **options)
    direct_inputs = direct._mace_inputs(atoms)
    directed_edges = len(direct_inputs[6])
    direct_values = evaluate(direct, atoms.copy())
    identity = direct_identity(direct, args.backend, args.profile)
    generic_options = dict(options)
    generic_options.pop("_debug_execution_plan", None)
    generic = Symmetrix(args.compact_model, streamed_edges="generic", **generic_options)
    generic_values = evaluate(generic, atoms.copy())
    errors = numerical_errors(generic_values, direct_values, len(atoms))
    limits = manifest["tolerances"][args.dtype]
    parity_failures = [name for name, value in errors.items() if value > limits[name]]
    failures = [
        *identity["gate_failures"],
        *[f"{name} exceeds tolerance" for name in parity_failures],
    ]
    cross_precision = None
    if args.reference_output is not None:
        reference_record = json.loads(args.reference_output.read_text())
        reference = {
            key: np.asarray(value, dtype=np.float64)
            if key != "energy"
            else float(value)
            for key, value in reference_record["_direct_values"].items()
        }
        cross_limits = manifest["cross_precision_tolerances"]
        cross_precision, cross_failures = cross_precision_assessment(
            reference, direct_values, len(atoms), cross_limits
        )
        cross_precision["reference_case_key"] = reference_record["case_key"]
        failures.extend(cross_failures)
    elif args.dtype == "float32":
        cross_precision = {
            "status": "not_available",
            "reason": "matching direct FP64 worker was not selected or did not pass",
        }
    if args.profile == "capacity":
        probe = next(
            (
                item
                for item in manifest.get("capacity_probes", ())
                if item["model"] == args.model_id
                and item["structure"] == args.structure
                and item["dtype"] == args.dtype
                and item["backend"] == args.backend
            ),
            None,
        )
        expected_tile = (
            None if probe is None else probe.get("expected_m1_tile_channels")
        )
        selected_tile = identity["executors"].get("m1_recompute_tile_channels")
        if expected_tile is not None and selected_tile != expected_tile:
            failures.append(
                f"capacity probe selected M1 tile {selected_tile}, expected {expected_tile}"
            )
    model_cutoff = float(json.loads(args.compact_model.read_text())["r_cut"])
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "case_key": case_key(
            args.model_id, args.structure, args.dtype, args.backend, args.profile
        ),
        "status": "passed" if not failures else "failed",
        "model": args.model_id,
        "structure": args.structure,
        "atoms": len(atoms),
        "directed_edges": directed_edges,
        "model_cutoff_A": model_cutoff,
        "neighbor_skin_A": neighbor_skin,
        "effective_neighbor_cutoff_A": model_cutoff + neighbor_skin,
        "dtype": args.dtype,
        "backend": args.backend,
        "execution_profile": args.profile,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_hash,
        },
        "compact_model": {
            "path": str(args.compact_model.resolve()),
            "sha256": sha256_file(args.compact_model),
        },
        "native": {
            "path": str(Path(native.__file__).resolve()),
            "sha256": sha256_file(Path(native.__file__)),
            "build_info": build_info,
        },
        "direct": identity,
        "parity": {"errors": errors, "tolerances": limits},
        "cross_precision": cross_precision,
        "failures": failures,
        "outputs": {
            "energy_eV": direct_values["energy"],
            "force_norm_eV_per_A": float(np.linalg.norm(direct_values["forces"])),
            "stress_norm_eV_per_A3": float(np.linalg.norm(direct_values["stress"])),
        },
        "_direct_values": {
            "energy": direct_values["energy"],
            "forces": direct_values["forces"].tolist(),
            "stress": direct_values["stress"].tolist(),
        },
    }


def extraction_command(
    python: Path,
    script: Path,
    checkpoint: Path,
    output: Path,
    model: dict,
    structure: dict,
) -> list[str]:
    command = [
        str(python),
        str(script),
        "extract",
        "--model",
        str(checkpoint),
        "--chemical-symbols",
        *structure["symbols"],
        "--output",
        str(output),
    ]
    if model.get("head"):
        command.extend(("--head", model["head"]))
    return command


def worker_command(
    python: Path,
    script: Path,
    args,
    model: dict,
    structure: str,
    dtype: str,
    backend: str,
    checkpoint: Path,
    compact: Path,
    output: Path,
    profile: str = "speed",
    reference_output: Path | None = None,
) -> list[str]:
    selector = {
        "cpu": args.cpu_selector,
        "cuda": args.cuda_selector,
        "hip": args.hip_selector,
    }[backend]
    command = [
        str(python),
        str(script),
        "worker",
        "--manifest",
        str(args.manifest),
        "--model-id",
        model["id"],
        "--structure",
        structure,
        "--dtype",
        dtype,
        "--backend",
        backend,
        "--backend-selector",
        selector,
        "--profile",
        profile,
        "--checkpoint",
        str(checkpoint),
        "--compact-model",
        str(compact),
        "--jit-cache",
        str(args.jit_cache / backend / dtype / model["id"]),
        "--output",
        str(output),
    ]
    if reference_output is not None:
        command.extend(("--reference-output", str(reference_output)))
    return command


def failure_record(
    model,
    structure,
    dtype,
    backend,
    message,
    checkpoint: dict | None = None,
    profile: str = "speed",
) -> dict:
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "case_key": case_key(model, structure, dtype, backend, profile),
        "status": "failed",
        "model": model,
        "structure": structure,
        "dtype": dtype,
        "backend": backend,
        "execution_profile": profile,
        "failures": [message],
    }
    if checkpoint is not None:
        result["checkpoint"] = checkpoint
    return result


def collect_worker_record(
    output: Path,
    result,
    model: str,
    structure: str,
    dtype: str,
    backend: str,
    profile: str = "speed",
) -> dict:
    if output.is_file():
        record = json.loads(output.read_text())
        if result.returncode != 0:
            record["status"] = "failed"
            record.setdefault("failures", []).append(
                f"worker exited {result.returncode} after writing output: "
                f"{result.stderr[-2000:]}"
            )
        return record
    return failure_record(
        model,
        structure,
        dtype,
        backend,
        f"worker exited {result.returncode}: {result.stderr[-2000:]}",
        profile=profile,
    )


def orchestrate(args) -> int:
    manifest = load_manifest(args.manifest)
    for backend in args.backends:
        python = getattr(args, f"{backend}_python")
        selector = getattr(args, f"{backend}_selector")
        if python is None or selector is None:
            raise ValueError(
                f"backend {backend!r} requires --{backend}-python and "
                f"--{backend}-selector"
            )
    done = completed_keys(args.output_jsonl, args.retry_failures)
    script = Path(__file__).resolve()
    grouped = {}
    for case in matrix_cases(
        manifest, args.models, args.structures, args.dtypes, args.backends
    ):
        grouped.setdefault((case[0]["id"], case[1]), []).append(case)
    failures = 0
    for (_, structure), cases in grouped.items():
        pending = [
            case
            for case in cases
            if case_key(case[0]["id"], structure, case[2], case[3], case[4]) not in done
        ]
        if not pending:
            continue
        model = pending[0][0]
        applicable, reason = chemistry_status(model, structure)
        if not applicable:
            for _, _, dtype, backend, profile in pending:
                append_jsonl(
                    args.output_jsonl,
                    {
                        "schema": SCHEMA,
                        "version": VERSION,
                        "case_key": case_key(
                            model["id"], structure, dtype, backend, profile
                        ),
                        "status": "not_applicable",
                        "model": model["id"],
                        "structure": structure,
                        "dtype": dtype,
                        "backend": backend,
                        "execution_profile": profile,
                        "reason": reason,
                    },
                )
            continue
        checkpoint = args.checkpoint_dir / model["filename"]
        compact = args.compact_dir / f"{model['id']}-{structure}.json"
        if not checkpoint.is_file():
            message = f"missing checkpoint {checkpoint}; source URL: {model['url']}"
            for _, _, dtype, backend, profile in pending:
                append_jsonl(
                    args.output_jsonl,
                    failure_record(
                        model["id"],
                        structure,
                        dtype,
                        backend,
                        message,
                        profile=profile,
                    ),
                )
                failures += 1
            continue
        checkpoint_record = {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "expected_sha256": model.get("sha256"),
        }
        if (
            checkpoint_record["expected_sha256"]
            and checkpoint_record["sha256"] != checkpoint_record["expected_sha256"]
        ):
            message = (
                f"checkpoint SHA-256 {checkpoint_record['sha256']} != expected "
                f"{checkpoint_record['expected_sha256']}"
            )
            for _, _, dtype, backend, profile in pending:
                append_jsonl(
                    args.output_jsonl,
                    failure_record(
                        model["id"],
                        structure,
                        dtype,
                        backend,
                        message,
                        checkpoint_record,
                        profile,
                    ),
                )
                failures += 1
            continue
        if not compact.is_file() or not args.reuse_compact:
            compact.parent.mkdir(parents=True, exist_ok=True)
            command = extraction_command(
                args.converter_python,
                script,
                checkpoint,
                compact,
                model,
                manifest["structures"][structure],
            )
            result = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            if result.returncode != 0:
                message = (
                    f"extraction failed ({result.returncode}): {result.stderr[-2000:]}"
                )
                for _, _, dtype, backend, profile in pending:
                    append_jsonl(
                        args.output_jsonl,
                        failure_record(
                            model["id"],
                            structure,
                            dtype,
                            backend,
                            message,
                            checkpoint_record,
                            profile,
                        ),
                    )
                    failures += 1
                continue
        # FP64 runs first and stay on disk long enough to serve as independent
        # direct references for the matching fresh-process FP32 workers.
        pending.sort(key=lambda row: (row[3], row[2] != "float64", row[4]))
        with tempfile.TemporaryDirectory(
            prefix="symmetrix-foundation-worker-"
        ) as temporary:
            reference_outputs = {}
            for _, _, dtype, backend, profile in pending:
                python = {
                    "cpu": args.cpu_python,
                    "cuda": args.cuda_python,
                    "hip": args.hip_python,
                }[backend]
                output = Path(temporary) / f"{backend}-{dtype}-{profile}.json"
                reference_output = (
                    reference_outputs.get((backend, profile))
                    if dtype == "float32"
                    else None
                )
                command = worker_command(
                    python,
                    script,
                    args,
                    model,
                    structure,
                    dtype,
                    backend,
                    checkpoint,
                    compact,
                    output,
                    profile,
                    reference_output,
                )
                result = subprocess.run(
                    command, text=True, capture_output=True, check=False
                )
                record = collect_worker_record(
                    output,
                    result,
                    model["id"],
                    structure,
                    dtype,
                    backend,
                    profile,
                )
                if dtype == "float64" and record["status"] == "passed":
                    reference_outputs[(backend, profile)] = output
                public_record = dict(record)
                public_record.pop("_direct_values", None)
                append_jsonl(args.output_jsonl, public_record)
                failures += record["status"] == "failed"
    return 1 if failures else 0


def extract(args) -> int:
    from symmetrix.cli.extract_mace import main

    command = [
        "--model",
        str(args.model),
        "--chemical-symbols",
        *args.chemical_symbols,
        "--output",
        str(args.output),
    ]
    if args.head:
        command.extend(("--head", args.head))
    return int(main(command) or 0)


def worker(args) -> int:
    try:
        record = run_worker(args)
    except Exception as error:  # noqa: BLE001 - persist worker failures as records
        record = failure_record(
            args.model_id,
            args.structure,
            args.dtype,
            args.backend,
            f"{type(error).__name__}: {error}",
            profile=args.profile,
        )
        record["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0 if record["status"] == "passed" else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--checkpoint-dir", type=Path, required=True)
    run.add_argument("--compact-dir", type=Path, required=True)
    run.add_argument("--output-jsonl", type=Path, required=True)
    run.add_argument("--jit-cache", type=Path, required=True)
    run.add_argument("--converter-python", type=Path, required=True)
    run.add_argument("--cpu-python", type=Path, required=True)
    run.add_argument("--cuda-python", type=Path)
    run.add_argument("--hip-python", type=Path)
    run.add_argument("--cpu-selector", default="cpu")
    run.add_argument("--cuda-selector")
    run.add_argument("--hip-selector")
    run.add_argument("--models", nargs="*")
    run.add_argument("--structures", nargs="+", default=("C64", "STO40", "H2O"))
    run.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float64"),
        default=("float32", "float64"),
    )
    run.add_argument(
        "--backends",
        nargs="+",
        choices=("cpu", "cuda", "hip"),
        default=("cpu", "cuda"),
    )
    run.add_argument("--reuse-compact", action="store_true")
    run.add_argument("--retry-failures", action="store_true")
    run.set_defaults(handler=orchestrate)
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--model-id", required=True)
    worker_parser.add_argument(
        "--structure", choices=("C64", "STO40", "H2O"), required=True
    )
    worker_parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    worker_parser.add_argument(
        "--backend", choices=("cpu", "cuda", "hip"), required=True
    )
    worker_parser.add_argument("--backend-selector", required=True)
    worker_parser.add_argument(
        "--profile", choices=("speed", "capacity"), default="speed"
    )
    worker_parser.add_argument("--checkpoint", type=Path, required=True)
    worker_parser.add_argument("--compact-model", type=Path, required=True)
    worker_parser.add_argument("--jit-cache", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.add_argument("--reference-output", type=Path)
    worker_parser.set_defaults(handler=worker)
    extract_parser = commands.add_parser("extract")
    extract_parser.add_argument("--model", type=Path, required=True)
    extract_parser.add_argument("--chemical-symbols", nargs="+", required=True)
    extract_parser.add_argument("--head")
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.set_defaults(handler=extract)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
