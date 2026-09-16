#!/usr/bin/env python3

from argparse import ArgumentParser

from ..extract_mace_data import export_mace_model, extract_mace_data
from .prepare_jit_device_artifact import main as prepare_jit_device_artifact


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", required=True, help="Torch model file.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--atomic-numbers",
        "-Z",
        "-z",
        nargs="+",
        help="Atomic numbers to extract.",
        default=[],
    )
    group.add_argument(
        "--chemical-symbols",
        "-s",
        nargs="+",
        help="Chemical symbols to extract.",
        default=[],
    )
    parser.add_argument(
        "--head",
        "-H",
        help="Default head for a multi-head JSON. All compatible heads are retained. "
        "Defaults to the first non-PT head.",
    )
    parser.add_argument(
        "--radial-format",
        choices=("compact", "pair-splines"),
        default="compact",
        help="Radial representation. Compact supports universal artifacts; pair-splines is legacy.",
    )
    parser.add_argument(
        "--num-spline-points",
        type=int,
        default=256,
        help="Number of radial spline grid points.",
    )
    parser.add_argument("--output", "-o", help="Output filename.")
    parser.add_argument(
        "--export-model",
        help="Also export a loadable e3nn MACE .pt model from a training checkpoint.",
    )
    parser.add_argument(
        "--prepare-jit-device-artifact",
        action="store_true",
        help=(
            "Prepare and validate both float32 and float64 JIT CUDA/HIP R1 "
            "artifacts after conversion."
        ),
    )
    parser.add_argument(
        "--jit-device-backend",
        choices=("auto", "cuda", "hip"),
        default="auto",
        help="Device backend for conversion-time JIT artifact preparation.",
    )
    parser.add_argument(
        "--jit-device-cache-root",
        help="Optional cache root for the prepared JIT device artifact.",
    )
    args = parser.parse_args(argv)

    if args.export_model is not None:
        export_mace_model(args.model, args.export_model)

    if args.prepare_jit_device_artifact and args.radial_format != "compact":
        parser.error("--prepare-jit-device-artifact requires --radial-format compact")

    import json
    from pathlib import Path

    if len(Path(args.model).suffix) == 0:
        model_name = Path(args.model).name
    else:
        model_name = Path(args.model).stem

    species = (
        args.atomic_numbers if args.chemical_symbols == [] else args.chemical_symbols
    )
    output = extract_mace_data(
        args.model,
        species=species,
        head=args.head,
        num_spline_points=args.num_spline_points,
        radial_format=args.radial_format,
    )

    ### ----- WRITE JSON -----

    if args.output is None:
        suffix = (
            "universal" if not species else "-".join(str(a) for a in sorted(species))
        )
        args.output = model_name + "-" + suffix + ".json"
    print("WRITING JSON TO", args.output)
    with open(args.output, "w") as f:
        if args.radial_format == "compact":
            json.dump(output, f, separators=(",", ":"))
        else:
            json.dump(output, f, indent=4)
        f.write("\n")

    if not args.prepare_jit_device_artifact:
        return 0

    prepare_arguments = [
        "--model",
        args.output,
        "--precision",
        "float32",
        "--precision",
        "float64",
        "--backend",
        args.jit_device_backend,
    ]
    if args.jit_device_cache_root is not None:
        prepare_arguments.extend(("--cache-root", args.jit_device_cache_root))
    return prepare_jit_device_artifact(prepare_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
