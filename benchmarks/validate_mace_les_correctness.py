#!/usr/bin/env python3
"""Validate full-array MACELES checkpoint comparison records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCHEMA = "symmetrix.mace-les-correctness/2"
UPSTREAM_LIMITS = {
    "float32": (3.5e-6, 3.5e-5, 2.5e-6, 1.8e-7),
    "float64": (3.0e-6, 3.1e-5, 2.0e-6, 1.6e-7),
}
DIRECT_LIMITS = {
    "float32": (5.0e-7, 1.0e-5, 2.0e-6, 1.0e-7),
    "float64": (5.0e-7, 1.0e-5, 2.0e-6, 1.0e-7),
}


class CorrectnessError(ValueError):
    pass


def _digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CorrectnessError(f"{name} must be a lowercase SHA-256")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(document, name):
    value = document.get(name)
    if not isinstance(value, dict) or not value.get("path") or not value.get("sha256"):
        raise CorrectnessError(f"{name} identity is incomplete")
    path = Path(value["path"])
    if not path.is_file() or _sha256(path) != value["sha256"]:
        raise CorrectnessError(f"{name} SHA-256 does not match the recorded file")
    size = value.get("size_bytes")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size != path.stat().st_size
    ):
        raise CorrectnessError(f"{name} size does not match the recorded file")
    return value


def _array(document, name):
    try:
        result = np.asarray(document[name], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise CorrectnessError(f"{name} must be a numeric array") from error
    if not np.all(np.isfinite(result)):
        raise CorrectnessError(f"{name} contains non-finite values")
    return result


def _runtime_provenance(document, label):
    value = document.get("runtime_provenance")
    if not isinstance(value, dict):
        raise CorrectnessError(f"{label} runtime provenance must be an object")
    for name in (
        "interpreter_path",
        "python_version",
        "package_path",
        "package_version",
    ):
        if not isinstance(value.get(name), str) or not value[name]:
            raise CorrectnessError(f"{label} runtime provenance {name} is invalid")
    if not Path(value["interpreter_path"]).is_file():
        raise CorrectnessError(f"{label} interpreter path does not exist")
    if not Path(value["package_path"]).is_file():
        raise CorrectnessError(f"{label} package path does not exist")
    return value


def _validate_document(document, label):
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise CorrectnessError(f"{label} has an unsupported schema")
    if document.get("precision") not in UPSTREAM_LIMITS:
        raise CorrectnessError(f"{label} has an unsupported precision")
    if document.get("implementation") not in {"upstream", "symmetrix"}:
        raise CorrectnessError(f"{label} has an unsupported implementation")
    if document.get("mode") not in {"upstream", "reference", "direct"}:
        raise CorrectnessError(f"{label} has an unsupported mode")
    if document.get("backend") not in {"cpu", "cuda", "hip"}:
        raise CorrectnessError(f"{label} has an unsupported backend")
    if document.get("model_type") != "MACELES":
        raise CorrectnessError(f"{label}.model_type must be MACELES")
    for name in (
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
    ):
        _digest(document.get(name), f"{label}.{name}")
    checkpoint = _identity(document, "checkpoint")
    converted = _identity(document, "converted_model")
    structure = _identity(document, "structure")
    atoms = structure.get("atoms")
    if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms <= 0:
        raise CorrectnessError(f"{label} structure atom count must be positive")
    energy = document.get("energy")
    if (
        isinstance(energy, bool)
        or not isinstance(energy, (int, float))
        or not math.isfinite(energy)
    ):
        raise CorrectnessError(f"{label} energy must be finite")
    arrays = {
        name: _array(document, name) for name in ("forces", "latent_charges", "stress")
    }
    if arrays["forces"].shape != (atoms, 3) or arrays["latent_charges"].shape != (
        atoms,
    ):
        raise CorrectnessError(f"{label} atom array shapes are inconsistent")
    if arrays["stress"].shape != (6,):
        raise CorrectnessError(f"{label} stress must have exact Voigt-6 shape")
    runtime = _runtime_provenance(document, label)
    return checkpoint, converted, structure, float(energy), arrays, runtime


def validate_documents(reference, candidate):
    ref = _validate_document(reference, "reference")
    cand = _validate_document(candidate, "candidate")
    if reference["precision"] != candidate["precision"]:
        raise CorrectnessError("precision provenance does not match")
    for name in (
        "model_type",
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
    ):
        if reference[name] != candidate[name]:
            raise CorrectnessError(f"{name} provenance does not match")
    for index, name in enumerate(("checkpoint", "converted model", "structure")):
        key = "sha256"
        if ref[index][key] != cand[index][key]:
            raise CorrectnessError(f"{name} provenance does not match")
    if ref[2].get("frame") != cand[2].get("frame") or ref[2].get("atoms") != cand[
        2
    ].get("atoms"):
        raise CorrectnessError("structure frame provenance does not match")
    pairing = (
        reference["implementation"],
        reference["mode"],
        candidate["implementation"],
        candidate["mode"],
    )
    if pairing == ("symmetrix", "reference", "symmetrix", "direct"):
        if reference["backend"] != candidate["backend"]:
            raise CorrectnessError("direct/reference backends do not match")
        if ref[5] != cand[5]:
            raise CorrectnessError("direct/reference runtime provenance does not match")
        limits = DIRECT_LIMITS[candidate["precision"]]
        comparison = "direct-versus-reference"
    elif pairing == ("upstream", "upstream", "symmetrix", "reference"):
        limits = UPSTREAM_LIMITS[candidate["precision"]]
        comparison = "symmetrix-versus-upstream"
    else:
        raise CorrectnessError("unsupported correctness comparison pairing")
    atom_count = ref[2]["atoms"]
    deviations = {
        "energy_abs_eV_per_atom": abs(cand[3] - ref[3]) / atom_count,
        "force_max_abs_eV_per_A": float(
            np.max(np.abs(cand[4]["forces"] - ref[4]["forces"]))
        ),
        "charge_max_abs": float(
            np.max(np.abs(cand[4]["latent_charges"] - ref[4]["latent_charges"]))
        ),
        "stress_max_abs_eV_per_A3": float(
            np.max(np.abs(cand[4]["stress"] - ref[4]["stress"]))
        ),
    }
    names = tuple(deviations)
    thresholds = dict(zip(names, limits, strict=True))
    failures = [name for name in names if deviations[name] > thresholds[name]]
    if failures:
        raise CorrectnessError(
            "correctness tolerance exceeded for " + ", ".join(failures)
        )
    return {
        "schema": "symmetrix.mace-les-correctness-validation/2",
        "passed": True,
        "comparison": comparison,
        "precision": candidate["precision"],
        "deviations": deviations,
        "thresholds": thresholds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_documents(
            json.loads(args.reference.read_text()),
            json.loads(args.candidate.read_text()),
        )
    except (OSError, json.JSONDecodeError, CorrectnessError) as error:
        parser.error(str(error))
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.json_output == Path("-"):
        sys.stdout.write(rendered)
    elif args.json_output is not None:
        args.json_output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
