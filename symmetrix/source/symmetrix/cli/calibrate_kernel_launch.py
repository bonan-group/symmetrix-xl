#!/usr/bin/env python3

import json
from argparse import ArgumentParser

from ase.io import read

from ..kernel_launch_calibrate import (
    calibrate_kernel_launches,
    calibration_result_dict,
)

_STREAMED_EDGE_CHOICES = ("generic", "direct", "all_interactions", "factorized")


def main():
    parser = ArgumentParser(
        description="Explicitly calibrate bounded Execution GPU launch profiles."
    )
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--structure", "-s", required=True)
    parser.add_argument("--index", default="-1")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--streamed-edges",
        choices=_STREAMED_EDGE_CHOICES,
        default="direct",
    )
    parser.add_argument(
        "--jit",
        choices=("auto", "fallback", "jit_only", "off", "required"),
        default=None,
        help="deprecated and ignored; use SYMMETRIX_JIT_POLICY",
    )
    parser.add_argument(
        "--properties", nargs="+", default=("energy", "forces", "stress")
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--confirmation-samples", type=int, default=2)
    parser.add_argument("--time-budget-seconds", type=float, default=120.0)
    parser.add_argument("--minimum-improvement", type=float, default=0.03)
    parser.add_argument("--candidates", type=int, nargs="+")
    parser.add_argument("--cache-root")
    args = parser.parse_args()

    atoms = read(args.structure, index=args.index)
    result = calibrate_kernel_launches(
        args.model,
        atoms,
        precision=args.precision,
        streamed_edges=args.streamed_edges,
        jit=args.jit,
        properties=args.properties,
        warmups=args.warmups,
        samples=args.samples,
        confirmation_samples=args.confirmation_samples,
        time_budget_seconds=args.time_budget_seconds,
        minimum_improvement=args.minimum_improvement,
        candidate_restriction=args.candidates,
        cache_root=args.cache_root,
    )
    print(json.dumps(calibration_result_dict(result), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
