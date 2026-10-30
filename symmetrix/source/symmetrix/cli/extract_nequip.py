"""Command line exporter for supported NequIP packages."""

from __future__ import annotations

import argparse
import sys

from ..extract_nequip_data import extract_nequip_data, export_nequip_model
from ..nequip_probe import NequIPProbeError, load_nequip_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="symmetrix_extract_nequip")
    parser.add_argument("--model", required=True, help="NequIP package or checkpoint")
    parser.add_argument(
        "--output", required=True, help="output versioned NequIP JSON contract"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing output"
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
        help=(
            "precision the export is certified for; recorded as "
            "kokkos_radial.precision and model_metadata.output_precision. It does "
            "not quantize coefficients: the radial spline is always built in "
            "double and cast on load, and a float64 export is accepted by a "
            "float32 evaluator."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        model = load_nequip_model(args.model)
        data = extract_nequip_data(model, output_precision=args.precision)
        metadata = data.get("model_metadata", {})
        print(f"source model: {type(model).__module__}.{type(model).__name__}")
        print(f"NequIP version: {metadata.get('source_version', 'unknown')}")
        print(f"layers: {data['num_layers']}")
        print(f"elements: {len(data['model_atomic_numbers'])}")
        print(f"cutoff: {data['r_cut']:g}")
        metadata = data.get("model_metadata", {})
        rejected = metadata.get("rejected_modules", [])
        print(f"rejected modules: {len(rejected)}")
        defaults = metadata.get("exporter_defaults", [])
        print(f"defaulted tensors: {len(defaults)}")
        for item in defaults:
            print(f"  {item}")
        print(f"certified precision: {args.precision}")
        output = export_nequip_model(
            model, args.output, force=args.force, precision=args.precision
        )
        print(f"wrote: {output}")
    except (NequIPProbeError, OSError, ValueError, TypeError) as error:
        print(f"symmetrix_extract_nequip: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
