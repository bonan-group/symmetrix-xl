#!/usr/bin/env python3
"""Evaluate an official MACELES checkpoint or its Symmetrix conversion."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

SCHEMA = "symmetrix.mace-les-correctness/2"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _json_array(value, name):
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.tolist()


def _runtime_provenance(implementation):
    module_name = "mace" if implementation == "upstream" else "symmetrix"
    distribution = "mace-torch" if implementation == "upstream" else "symmetrix-xl"
    module = importlib.import_module(module_name)
    return {
        "interpreter_path": str(Path(sys.executable).absolute()),
        "python_version": platform.python_version(),
        "package_path": str(Path(module.__file__).resolve()),
        "package_version": importlib.metadata.version(distribution),
    }


def _model_identity(converted_model):
    from symmetrix.mace_jit_artifact_common import mace_les_evaluator_identity

    model = json.loads(Path(converted_model).read_text())
    identity = mace_les_evaluator_identity(model)
    if identity["model_type"] != "MACELES":
        raise ValueError(f"{converted_model} is not a MACELES model")
    return identity


def build_result_document(
    *,
    implementation,
    precision,
    backend,
    mode,
    checkpoint,
    converted_model,
    structure,
    frame,
    atom_count,
    results,
):
    return {
        "schema": SCHEMA,
        **_model_identity(converted_model),
        "implementation": implementation,
        "precision": precision,
        "backend": backend,
        "mode": mode,
        "checkpoint": _file_identity(checkpoint),
        "converted_model": _file_identity(converted_model),
        "structure": {
            **_file_identity(structure),
            "frame": frame,
            "atoms": atom_count,
        },
        "runtime_provenance": _runtime_provenance(implementation),
        "energy": float(results["energy"]),
        "forces": _json_array(results["forces"], "forces"),
        "latent_charges": _json_array(results["latent_charges"], "latent_charges"),
        "stress": _json_array(results["stress"], "stress"),
    }


def _evaluate_upstream(atoms, checkpoint, precision, device):
    from mace.calculators.mace import MACECalculator

    calculator = MACECalculator(
        model_paths=str(checkpoint),
        device=device,
        default_dtype=precision,
    )
    calculator.calculate(atoms, properties=["energy", "forces", "stress"])
    results = calculator.results
    if "latent_charges" not in results:
        raise RuntimeError("upstream MACELES did not publish latent_charges")
    return results


def _evaluate_symmetrix(atoms, converted_model, precision, mode):
    from symmetrix import Symmetrix

    calculator = Symmetrix(
        converted_model,
        dtype=precision,
        streamed_edges="direct" if mode == "direct" else "reference",
        neighbor_skin=0.0,
    )
    calculator.calculate(atoms, properties=["energy", "forces", "stress"])
    if calculator.model_type != "MACELES":
        raise RuntimeError("converted model did not construct a MACELES evaluator")
    required = {"les_energy", "mace_les_energy", "latent_charges"}
    missing = sorted(required - calculator.results.keys())
    if missing:
        raise RuntimeError("MACELES evaluation omitted " + ", ".join(missing))
    return calculator.results


def run(args):
    from ase.io import read

    atoms = read(args.structure, index=args.frame)
    if args.implementation == "upstream":
        results = _evaluate_upstream(
            atoms, args.checkpoint, args.precision, args.device
        )
        backend = args.device.split(":", maxsplit=1)[0]
        mode = "upstream"
    else:
        from symmetrix import selected_backend

        results = _evaluate_symmetrix(
            atoms, args.converted_model, args.precision, args.mode
        )
        descriptor = selected_backend()
        backend = "unknown" if descriptor is None else descriptor["backend"]
        mode = args.mode
    return build_result_document(
        implementation=args.implementation,
        precision=args.precision,
        backend=backend,
        mode=mode,
        checkpoint=args.checkpoint,
        converted_model=args.converted_model,
        structure=args.structure,
        frame=args.frame,
        atom_count=len(atoms),
        results=results,
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("upstream", "symmetrix"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--converted-model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--mode", choices=("reference", "direct"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.frame < 0:
        parser.error("frame must be non-negative")
    document = run(args)
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output is None or args.output == Path("-"):
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
