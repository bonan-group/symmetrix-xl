#!/usr/bin/env python3
"""JSON-only migration of a legacy (format v2) MACE export to the v4 m0_composite basis.

Writes a NEW file; never overwrites an existing path.  No checkpoint is read.
Refuses multi-head, field-coupled, first-interaction-residual, single-layer,
nonlinear and pair-spline (v1) payloads.

  python -m symmetrix.cli.recompose_mace_json --input old.json --output new.json \
      [--h1-affine-mode m0_composite] [--receipt new.recompose.json]
  python -m symmetrix.cli.recompose_mace_json --validate-only file.json [file.json ...]
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from ..h1_affine import (
    H1_AFFINE_MODE_M0_COMPOSITE,
    H1AffineError,
    recompose_legacy_payload,
    sha256_bytes,
    validate_payload,
)


def _write_new(path: Path, text: str) -> None:
    """Create ``path`` exclusively (O_EXCL); a partial write never leaves ``path``."""
    tmp = path.with_name(f".{path.name}.partial.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)  # fails with FileExistsError if path appeared meanwhile
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _identity_verdict(checks: dict) -> tuple[bool, float, float]:
    """fp64 identity gate: composite vs legacy within 1e-12 of the summed-term
    magnitude (pure reassociation round-off, ~K eps, independent of conditioning).
    Returns (ok, worst E1 rel-to-result, worst H2 rel-to-result) for reporting."""
    worst_terms = max(
        [checks["E1_max_rel_terms"], checks["q_vs_Cr_max_rel_terms"]]
        + [item["H2_max_rel_terms"] for item in checks["per_element"]]
        + [item["T_vs_CS_max_rel_terms"] for item in checks["per_element"]]
    )
    worst_e = checks["E1_max_abs_diff"] / max(checks["E1_scale"], 1e-300)
    worst_h2 = max(
        item["H2_max_abs_diff"] / max(item["H2_scale"], 1e-300)
        for item in checks["per_element"]
    )
    checks["worst_rel_terms"] = worst_terms
    return worst_terms <= 1e-12, worst_e, worst_h2


def main(argv=None) -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", "-i")
    parser.add_argument("--output", "-o")
    parser.add_argument(
        "--h1-affine-mode",
        choices=(H1_AFFINE_MODE_M0_COMPOSITE,),
        default=H1_AFFINE_MODE_M0_COMPOSITE,
    )
    parser.add_argument("--receipt", help="Optional NEW receipt JSON path.")
    parser.add_argument("--validate-only", nargs="+", metavar="JSON")
    args = parser.parse_args(argv)

    if args.validate_only:
        status = 0
        for name in args.validate_only:
            raw = Path(name).read_bytes()
            try:
                summary = validate_payload(json.loads(raw))
                print(json.dumps({"path": name, "sha256": sha256_bytes(raw),
                                  "bytes": len(raw), "valid": True, **summary}))
            except (H1AffineError, KeyError, TypeError, ValueError) as exc:
                status = 1
                print(json.dumps({"path": name, "sha256": sha256_bytes(raw),
                                  "valid": False, "error": str(exc)}))
        return status

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --validate-only is used")
    src = Path(args.input)
    dst = Path(args.output)
    if dst.exists() or (args.receipt and Path(args.receipt).exists()):
        print(f"REFUSED: output exists ({dst} / {args.receipt}); never overwritten",
              file=sys.stderr)
        return 2
    if dst.resolve() == src.resolve():
        print("REFUSED: output must differ from input", file=sys.stderr)
        return 2
    raw = src.read_bytes()
    source_sha = sha256_bytes(raw)
    try:
        data = json.loads(raw)
        new, checks = recompose_legacy_payload(
            data, source_sha256=source_sha, source_bytes=len(raw),
            source_path=str(src.resolve()))
    except (H1AffineError, KeyError, TypeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    ok, worst_e, worst_h2 = _identity_verdict(checks)
    if not ok:
        print(f"REFUSED: fp64 identity check failed (rel-to-terms {checks['worst_rel_terms']:.3e}; "
              f"E1 rel {worst_e:.3e}, H2 rel {worst_h2:.3e})", file=sys.stderr)
        return 4
    text = json.dumps(new, separators=(",", ":"), allow_nan=False) + "\n"
    # Round-trip check of the serialized payload before it is published.
    reparsed = json.loads(text)
    validate_payload(reparsed)
    if reparsed["readout_1_weights_for_M0"] != new["readout_1_weights_for_M0"]:
        print("REFUSED: serialization round-trip changed readout_1_weights_for_M0",
              file=sys.stderr)
        return 5
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_new(dst, text)
    out_sha = sha256_bytes(text.encode())
    receipt = {
        "schema": "symmetrix.recompose_mace_json.v1",
        "input": str(src.resolve()), "input_sha256": source_sha, "input_bytes": len(raw),
        "output": str(dst.resolve()), "output_sha256": out_sha,
        "output_bytes": len(text.encode()),
        "h1_affine_mode": H1_AFFINE_MODE_M0_COMPOSITE,
        "identity_fp64": {"E1_max_rel": worst_e, "H2_max_rel": worst_h2, "detail": checks},
        "conditioning": new["h1_affine_provenance"]["conditioning"],
    }
    if args.receipt:
        _write_new(Path(args.receipt), json.dumps(receipt, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("output", "output_sha256", "h1_affine_mode")}
                     | {"E1_max_rel": worst_e, "H2_max_rel": worst_h2,
                        "worst_rel_terms": checks["worst_rel_terms"],
                        "sigma_min_C": receipt["conditioning"]["sigma_min_C"],
                        "legacy_readout_1_l2": receipt["conditioning"]["legacy_readout_1_l2"],
                        "readout_1_for_M0_l2": receipt["conditioning"]["readout_1_for_M0_l2"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
