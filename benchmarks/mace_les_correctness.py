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
# Symmetrix records the volume its stress is divided by and the volume
# mace-torch would divide the same stress by; the two differ for a cell with
# a nonperiodic axis.
STRESS_VOLUME_KEYS = ("stress_volume_A3", "mace_torch_stress_volume_A3")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path, *, materialization="local"):
    if materialization == "recorded":
        if not isinstance(path, dict):
            raise ValueError("recorded identity must be an object")
        result = dict(path)
        result["materialization"] = "recorded"
        if not isinstance(result.get("path"), str) or not result["path"]:
            raise ValueError("recorded identity path is required")
        digest = result.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("recorded identity must contain a lowercase SHA-256")
        if (
            isinstance(result.get("size_bytes"), bool)
            or not isinstance(result.get("size_bytes"), int)
            or result["size_bytes"] <= 0
        ):
            raise ValueError("recorded identity size_bytes is required")
        return result
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "materialization": "local",
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
    checkpoint_identity=None,
):
    return {
        "schema": SCHEMA,
        **_model_identity(converted_model),
        "implementation": implementation,
        "precision": precision,
        "backend": backend,
        "mode": mode,
        "checkpoint": (
            _file_identity(checkpoint_identity, materialization="recorded")
            if checkpoint_identity is not None
            else _file_identity(checkpoint)
        ),
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
        **{
            name: float(results[name]) for name in STRESS_VOLUME_KEYS if name in results
        },
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
    from symmetrix.calculator import _mace_stress_volume, _mace_torch_stress_volume

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
    volumes = (
        _mace_stress_volume(atoms, calculator.cutoff),
        _mace_torch_stress_volume(atoms, calculator.cutoff),
    )
    return {**calculator.results, **dict(zip(STRESS_VOLUME_KEYS, volumes))}


def run(args):
    reference_identity = None
    if args.reference_record is not None and args.checkpoint_identity is not None:
        raise ValueError("choose one portable identity source")
    if args.reference_record is not None:
        record = json.loads(args.reference_record.read_text())
        reference_identity = (
            record.get("checkpoint") if isinstance(record, dict) else None
        )
        if not isinstance(reference_identity, dict):
            raise ValueError("reference record must contain a checkpoint identity")
    if args.implementation == "upstream" and (
        args.reference_record is not None or args.checkpoint_identity is not None
    ):
        raise ValueError("upstream evaluation does not accept portable identities")
    if args.implementation == "upstream" and args.checkpoint is None:
        raise ValueError("upstream evaluation requires --checkpoint")
    if (
        args.implementation == "symmetrix"
        and args.checkpoint is None
        and reference_identity is None
        and args.checkpoint_identity is None
    ):
        raise ValueError(
            "symmetrix evaluation requires --checkpoint or portable identity"
        )
    if (
        args.implementation == "symmetrix"
        and args.checkpoint is not None
        and (reference_identity is not None or args.checkpoint_identity is not None)
    ):
        raise ValueError("choose --checkpoint or one portable identity source")
    if args.checkpoint_identity is not None:
        identity = json.loads(args.checkpoint_identity.read_text())
        if isinstance(identity, dict) and isinstance(identity.get("checkpoint"), dict):
            identity = identity["checkpoint"]
        args.checkpoint_identity = identity

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
        checkpoint_identity=(
            (reference_identity or args.checkpoint_identity)
            if (args.checkpoint_identity is not None or reference_identity is not None)
            else None
        ),
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("upstream", "symmetrix"), required=True
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-identity",
        type=Path,
        help="JSON identity from an immutable upstream record (portable mode)",
    )
    parser.add_argument("--reference-record", type=Path)
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
