#!/usr/bin/env python3

import gc
import json
import sys
from argparse import ArgumentParser

from .. import symmetrix as native_symmetrix
from ..jit_host_artifact import JitHostArtifactError, prepare_jit_host_artifact


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


def main(argv=None):
    parser = ArgumentParser(
        description=(
            "Prepare and validate a cached JIT host R1 artifact for ASE or "
            "pair_symmetrix."
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
    parser.add_argument("--cache-root")
    parser.add_argument(
        "--host-target",
        choices=("automatic", "native", "portable"),
        help="Override SYMMETRIX_JIT_HOST_TARGET for this artifact.",
    )
    parser.add_argument(
        "--path-only",
        action="store_true",
        help="Print only the validated artifact path.",
    )
    args = parser.parse_args(argv)

    initialized_before = native_symmetrix._kokkos_is_initialized()
    results = []
    error = None
    try:
        for precision in dict.fromkeys(args.precision):
            results.append(
                prepare_jit_host_artifact(
                    args.model,
                    precision=precision,
                    cache_root=args.cache_root,
                    host_target=args.host_target,
                )
            )
    except JitHostArtifactError as caught:
        error = caught
    try:
        _finalize_owned_kokkos(initialized_before)
    # Native runtime queries and finalization cross the pybind boundary.
    except Exception as cleanup_error:  # noqa: BLE001
        if error is None:
            error = JitHostArtifactError(
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
