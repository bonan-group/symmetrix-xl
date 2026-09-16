#!/usr/bin/env python3

import gc
import json
import sys
from argparse import ArgumentParser

from .. import symmetrix as native_symmetrix
from ..jit_device_artifact import (
    JitDeviceArtifactError,
    prepare_jit_device_artifact,
)


def _finalize_owned_kokkos(initialized_before):
    if initialized_before or not native_symmetrix._kokkos_is_initialized():
        return
    gc.collect()
    live_objects = native_symmetrix._kokkos_live_object_count()
    if live_objects:
        raise RuntimeError(
            f"cannot finalize CLI-owned Kokkos with {live_objects} live evaluator(s)"
        )
    native_symmetrix._finalize_kokkos()


def _lammps_token(value):
    text = str(value)
    if not text or any(character.isspace() or character in "'\"" for character in text):
        return json.dumps(text)
    return text


def _format_lammps_arguments(result):
    edge_policy = getattr(result, "edge_policy", None)
    blocks_per_compute_unit = 8
    if isinstance(edge_policy, dict):
        configured_blocks = edge_policy.get("persistent_blocks_per_compute_unit")
        if isinstance(configured_blocks, int):
            blocks_per_compute_unit = configured_blocks
    arguments = [
        "jit_device_artifact",
        _lammps_token(result.artifact_path),
        "jit_device_blocks_per_compute_unit",
        str(blocks_per_compute_unit),
    ]
    operator_modules = getattr(result, "operator_modules", {})
    if not isinstance(operator_modules, dict):
        raise JitDeviceArtifactError(
            "the prepared result contains invalid M0/R0 module metadata"
        )
    m0_module = operator_modules.get("M0", {})
    if m0_module.get("implementation") == "device_module":
        schedule = m0_module.get("schedule")
        if schedule not in ("chunk32", "table"):
            raise JitDeviceArtifactError(
                "the prepared M0 device module has an invalid schedule"
            )
        arguments.extend(
            (
                "jit_m0_device_artifact",
                _lammps_token(m0_module["artifact_path"]),
                "jit_m0_device_schedule",
                schedule,
            )
        )
    r0_module = operator_modules.get("R0", {})
    if r0_module.get("implementation") == "device_module":
        arguments.extend(
            (
                "jit_r0_device_artifact",
                _lammps_token(r0_module["artifact_path"]),
            )
        )
    return " ".join(arguments)


def main(argv=None):
    parser = ArgumentParser(
        description=(
            "Prepare and validate cached JIT CUDA/HIP R1 and required "
            "low-memory M0/R0 artifacts for ASE or pair_symmetrix."
        )
    )
    parser.add_argument("--model", "-m", required=True, help="Compact model JSON.")
    parser.add_argument(
        "--precision",
        required=True,
        action="append",
        choices=("float32", "float64"),
        help=(
            "Must match the consumer: float64 for symmetrix/mace/kk or float32 "
            "for symmetrix/mace/float32/kk. Repeat to prepare both."
        ),
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "hip"), default="auto")
    parser.add_argument("--cache-root")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--path-only",
        action="store_true",
        help="Print only the validated R1 artifact path.",
    )
    output_group.add_argument(
        "--lammps-arguments",
        action="store_true",
        help=(
            "Print the complete validated R1/M0/R0 pair_style argument fragment; "
            "requires exactly one precision."
        ),
    )
    args = parser.parse_args(argv)

    precisions = tuple(dict.fromkeys(args.precision))
    if args.lammps_arguments and len(precisions) != 1:
        parser.error("--lammps-arguments requires exactly one unique precision")

    initialized_before = native_symmetrix._kokkos_is_initialized()
    results = []
    error = None
    try:
        for precision in precisions:
            results.append(
                prepare_jit_device_artifact(
                    args.model,
                    precision=precision,
                    backend=args.backend,
                    cache_root=args.cache_root,
                    include_low_memory_operators=not args.path_only,
                )
            )
    except JitDeviceArtifactError as caught:
        error = caught
    try:
        _finalize_owned_kokkos(initialized_before)
    except Exception as cleanup_error:
        if error is None:
            error = JitDeviceArtifactError(
                f"could not finalize the CLI-owned Kokkos runtime: {cleanup_error}"
            )

    if error is not None:
        payload = {
            "status": "error",
            "available": False,
            "cache_key": error.cache_key,
            "reason": str(error),
            "diagnostics": list(error.diagnostics),
        }
        print(json.dumps(payload, sort_keys=True, indent=2), file=sys.stderr)
        return 1

    assert results
    if args.path_only:
        for result in results:
            print(result.artifact_path)
    elif args.lammps_arguments:
        try:
            print(_format_lammps_arguments(results[0]))
        except (JitDeviceArtifactError, KeyError) as format_error:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "available": False,
                        "cache_key": getattr(results[0], "cache_key", None),
                        "reason": str(format_error),
                        "diagnostics": [],
                    },
                    sort_keys=True,
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1
    elif len(results) == 1:
        print(json.dumps(results[0].as_dict(), sort_keys=True, indent=2))
    else:
        print(
            json.dumps(
                [result.as_dict() for result in results], sort_keys=True, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
