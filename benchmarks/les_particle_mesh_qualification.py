"""Qualify periodic LES particle mesh against the explicit reciprocal sum.

The driver emits JSON records and never changes production LES dispatch.  It
uses the private selection seam so a proposed mesh policy can be reviewed from
accuracy evidence before it becomes a public or automatic policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "symmetrix.les-particle-mesh-qualification/2"
ERROR_FIELDS = (
    "energy",
    "atomic_energies",
    "latent_charges",
    "forces",
    "stress",
)
MESH_INADEQUACY_PREFIXES = (
    "LES particle mesh places an active reciprocal vector on an ambiguous Nyquist representative",
    "LES particle mesh aliases active reciprocal vectors",
    "LES particle mesh omits an active reciprocal vector",
)
FINAL_ENVELOPES = {
    "energy": 2.0e-7,
    "atomic_energies": 2.0e-7,
    "latent_charges": 5.0e-7,
    "forces": 5.0e-6,
    "stress": 1.0e-6,
}
EXECUTION_SPACES_BY_BACKEND = {
    "cpu": frozenset(("OpenMP", "Serial")),
    "cuda": frozenset(("Cuda",)),
    "hip": frozenset(("HIP",)),
}
REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "case",
        "algorithm",
        "reference_algorithm",
        "model",
        "model_sha256",
        "model_type",
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
        "mace_architecture",
        "source_provenance",
        "native_extension",
        "native_extension_sha256",
        "backend",
        "kokkos_execution_space",
        "precision",
        "atoms",
        "cell",
        "sigma",
        "dl",
        "mesh",
        "spline_order",
        "directed_edges",
        "translation_phases",
        "active_cutoff_margin",
        "status",
        "inadequacy_reason",
        "errors",
        "envelopes",
        "samples_ns",
        "sample_count",
        "median_us_per_atom",
        "interpreter_path",
        "python_version",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0.0
    )


def _finite_positive(value: object) -> bool:
    return _finite_nonnegative(value) and value > 0.0


def _nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_vector(value: object, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(
            not isinstance(component, bool)
            and isinstance(component, (int, float))
            and math.isfinite(component)
            for component in value
        )
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def example_record_for_testing() -> dict[str, object]:
    """Return a minimal valid record for contract mutation tests."""

    digest = "1" * 64
    return {
        "schema": SCHEMA,
        "case": "orthogonal",
        "algorithm": "particle_mesh",
        "reference_algorithm": "explicit_sum",
        "model": "/qualified/maceles.json",
        "model_sha256": digest,
        "model_type": "MACELES",
        "mace_les_parameters_sha256": digest,
        "mace_les_evaluator_fingerprint": digest,
        "mace_architecture": {
            "mace_num_interactions": 2,
            "mace_num_channels": 128,
            "mace_contract_fingerprints": {
                "R0": f"sha256:{digest}",
                "M0": f"sha256:{digest}",
                "R1": f"sha256:{digest}",
            },
            "atomic_numbers": [1, 8],
            "cutoff": 5.0,
        },
        "source_provenance": {
            "source_commit": "test",
            "source_dirty": False,
            "source_content_sha256": digest,
            "native_source_content_sha256": digest,
        },
        "native_extension": "/qualified/_native_cpu.so",
        "native_extension_sha256": digest,
        "backend": "cpu",
        "kokkos_execution_space": "OpenMP",
        "precision": "float64",
        "atoms": 5,
        "cell": [[5.7, 0.0, 0.0], [0.0, 5.3, 0.0], [0.0, 0.0, 5.5]],
        "sigma": 0.8,
        "dl": 1.35,
        "mesh": [32, 32, 32],
        "spline_order": 6,
        "directed_edges": 800,
        "translation_phases": [[0.0, 0.0, 0.0]],
        "active_cutoff_margin": 0.1,
        "status": "ok",
        "inadequacy_reason": None,
        "errors": {name: 0.0 for name in ERROR_FIELDS},
        "envelopes": dict(FINAL_ENVELOPES),
        "samples_ns": [1000, 1100, 1050],
        "sample_count": 3,
        "median_us_per_atom": 0.21,
        "interpreter_path": "/qualified/venv/bin/python",
        "python_version": "3.12.13",
    }


def validate_record(record: dict[str, object]) -> dict[str, object]:
    """Validate one emitted record and return it unchanged."""

    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        raise ValueError("qualification record is missing: " + ", ".join(missing))
    if record["schema"] != SCHEMA:
        raise ValueError(f"unsupported qualification schema {record['schema']!r}")
    if record["precision"] != "float64":
        raise ValueError("LES particle-mesh qualification precision must be float64")
    if record["model_type"] != "MACELES":
        raise ValueError("qualification model_type must be MACELES")
    if not isinstance(record["mace_architecture"], Mapping):
        raise TypeError("qualification mace_architecture must be an object")
    backend = record["backend"]
    execution_spaces = EXECUTION_SPACES_BY_BACKEND.get(backend)
    if execution_spaces is None:
        raise ValueError(f"unsupported qualification backend {backend!r}")
    if record["kokkos_execution_space"] not in execution_spaces:
        raise ValueError(
            "qualification kokkos_execution_space is inconsistent with backend"
        )
    mesh = record["mesh"]
    if (
        not isinstance(mesh, list)
        or len(mesh) != 3
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 2 for x in mesh)
    ):
        raise ValueError("qualification mesh must contain three integer extents")
    order = record["spline_order"]
    if isinstance(order, bool) or not isinstance(order, int) or not 2 <= order <= 6:
        raise ValueError("qualification spline_order must be between 2 and 6")
    if record["case"] not in {"orthogonal", "triclinic"}:
        raise ValueError("qualification case must be orthogonal or triclinic")
    if record["algorithm"] != "particle_mesh":
        raise ValueError("qualification algorithm must be particle_mesh")
    if record["reference_algorithm"] != "explicit_sum":
        raise ValueError("qualification reference_algorithm must be explicit_sum")
    atoms = record["atoms"]
    if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms < 1:
        raise ValueError("qualification atoms must be positive")
    for name in ("sigma", "dl"):
        if not _finite_positive(record[name]):
            raise ValueError(f"qualification {name} must be finite and positive")
    if not _nonnegative_integer(record["directed_edges"]):
        raise ValueError("qualification directed_edges must be a non-negative integer")
    cell = record["cell"]
    if (
        not isinstance(cell, list)
        or len(cell) != 3
        or not all(_finite_vector(row, 3) for row in cell)
    ):
        raise ValueError("qualification cell must be a finite 3x3 matrix")
    phases = record["translation_phases"]
    if (
        not isinstance(phases, list)
        or not phases
        or not all(_finite_vector(phase, 3) for phase in phases)
    ):
        raise ValueError(
            "qualification translation_phases must contain finite 3-vectors"
        )
    margin = record["active_cutoff_margin"]
    if not _finite_nonnegative(margin) or margin <= 1.0e-3:
        raise ValueError("qualification cutoff margin must exceed 1e-3")
    if record["envelopes"] != FINAL_ENVELOPES:
        raise ValueError("qualification envelopes do not match reviewed limits")
    provenance = record["source_provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("source_provenance must be an object")
    source_commit = provenance.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("source_provenance.source_commit must be a nonempty string")
    if "source_dirty" not in provenance:
        raise ValueError("source_provenance.source_dirty is required")
    if not isinstance(provenance["source_dirty"], bool):
        raise TypeError("source_provenance.source_dirty must be a boolean")
    for name in ("source_content_sha256", "native_source_content_sha256"):
        value = provenance.get(name)
        if not _is_sha256(value):
            raise ValueError(f"source_provenance.{name} must be a lowercase SHA-256")
    for name in (
        "model_sha256",
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
        "native_extension_sha256",
    ):
        value = record[name]
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    status = record["status"]
    errors = record["errors"]
    reason = record["inadequacy_reason"]
    samples = record["samples_ns"]
    sample_count = record["sample_count"]
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 0
        or not isinstance(samples, list)
        or sample_count != len(samples)
    ):
        raise ValueError("qualification sample_count must match samples_ns")
    if status == "ok":
        if (
            not isinstance(samples, list)
            or not samples
            or any(not _finite_nonnegative(value) or value == 0 for value in samples)
        ):
            raise ValueError("qualification samples_ns must contain positive timings")
        if reason is not None:
            raise ValueError(
                "successful qualification cannot have an inadequacy reason"
            )
        if not isinstance(errors, dict) or set(errors) != set(ERROR_FIELDS):
            raise ValueError("successful qualification has incomplete error metrics")
        if any(not _finite_nonnegative(value) for value in errors.values()):
            raise ValueError(
                "qualification error metrics must be finite and non-negative"
            )
        expected_median = statistics.median(samples) / 1000.0 / atoms
        if not math.isclose(
            record["median_us_per_atom"], expected_median, rel_tol=1.0e-12
        ):
            raise ValueError(
                "qualification median_us_per_atom does not match samples and atoms"
            )
    elif status == "inadequate_mesh":
        if samples != [] or record["median_us_per_atom"] is not None:
            raise ValueError("inadequate mesh records cannot contain timings")
        if errors is not None:
            raise ValueError("inadequate mesh records cannot contain error metrics")
        if not isinstance(reason, str) or not reason:
            raise ValueError("inadequate mesh records require an inadequacy reason")
    else:
        raise ValueError(f"unsupported qualification status {status!r}")
    return record


def _mesh_inadequacy_reason(error: ValueError) -> str:
    message = str(error)
    if any(message.startswith(prefix) for prefix in MESH_INADEQUACY_PREFIXES):
        return message
    raise error


def _cases(np):
    fractional = np.array(
        [
            [0.03, 0.11, 0.19],
            [0.27, 0.17, 0.43],
            [0.39, 0.51, 0.23],
            [0.67, 0.37, 0.71],
            [0.83, 0.79, 0.61],
        ],
        dtype=np.float64,
    )
    charges = np.array([0.41, -0.73, 0.28, -0.16, 0.09], dtype=np.float64)
    return (
        ("orthogonal", charges, fractional, np.diag([5.7, 5.3, 5.5])),
        (
            "triclinic",
            charges,
            fractional,
            np.array(
                [[5.71, 0.17, 0.09], [0.11, 5.33, 0.21], [0.07, 0.19, 5.49]],
                dtype=np.float64,
            )
            * 0.98,
        ),
    )


def _active_cutoff_margin(np, cell, dl: float) -> float:
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    counts = [max(1, int(np.linalg.norm(vector) / dl)) for vector in cell]
    axes = [np.arange(-count, count + 1) for count in counts]
    miller = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    norms = np.linalg.norm(miller @ reciprocal, axis=1)
    cutoff = 2.0 * np.pi / dl
    nonzero = norms[norms > 0.0]
    return float(np.min(np.abs(nonzero - cutoff)))


def _identity(native, model_path, model, calculator) -> dict[str, object]:
    extension = Path(native.__file__).resolve()
    build = dict(native._backend_build_info())
    les = model["les"]
    return {
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_type": calculator.model_type,
        "mace_les_parameters_sha256": calculator.mace_les_parameters_sha256,
        "mace_les_evaluator_fingerprint": (calculator.mace_les_evaluator_fingerprint),
        "mace_architecture": {
            "mace_num_interactions": les["mace_num_interactions"],
            "mace_num_channels": les["mace_num_channels"],
            "mace_contract_fingerprints": dict(les["mace_contract_fingerprints"]),
            "atomic_numbers": list(model.get("atomic_numbers", [])),
            "cutoff": float(calculator.cutoff),
        },
        "source_provenance": {
            "source_commit": build.get("source_commit"),
            "source_dirty": build.get("source_dirty"),
            "source_content_sha256": build.get("source_content_sha256"),
            "native_source_content_sha256": build.get("native_source_content_sha256"),
        },
        "native_extension": str(extension),
        "native_extension_sha256": _sha256(extension),
        "backend": build.get("backend"),
        "kokkos_execution_space": native._kokkos_default_execution_space(),
        "precision": "float64",
        "interpreter_path": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
    }


def _selected_periodic_algorithm(calculator) -> str:
    diagnostics = dict(calculator.evaluator._les_particle_mesh_diagnostics_for_testing)
    algorithm = diagnostics.get("periodic_algorithm")
    if algorithm not in {"explicit_sum", "particle_mesh"}:
        raise RuntimeError(
            f"MACELES evaluator reported invalid periodic algorithm {algorithm!r}"
        )
    return algorithm


def run_qualification(
    model,
    *,
    meshes: tuple[int, ...] = (16, 24, 32),
    orders: tuple[int, ...] = (2, 4, 6),
    sample_count: int = 3,
) -> list[dict[str, object]]:
    """Run the fixed convergence matrix and return validated JSON records."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("sample_count must be a positive integer")

    import numpy as np
    from ase import Atoms

    from symmetrix import Symmetrix, backend_loader

    model_path = Path(model).resolve()
    model_data = json.loads(model_path.read_text())
    if model_data.get("model_type") != "MACELES":
        raise ValueError(f"{model_path} is not a MACELES model")
    les = model_data["les"]
    sigma = float(les["ewald"]["sigma"])
    dl = float(les["ewald"]["dl"])
    explicit_calculator = Symmetrix(
        model_path, dtype="float64", streamed_edges="reference", neighbor_skin=0.0
    )
    if explicit_calculator.model_type != "MACELES":
        raise RuntimeError("model did not construct a MACELES evaluator")
    native = backend_loader._native_module
    if native is None:
        raise RuntimeError("MACELES qualification did not load a native backend")
    capability = native._particle_mesh_capability()
    if not capability["available"]:
        raise RuntimeError(capability["reason"] or "particle mesh is unavailable")
    phases = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.137, 0.223, 0.319]),
        np.array([0.499, 0.251, 0.071]),
    )
    configurations = [(mesh, 6) for mesh in meshes]
    configurations.extend((meshes[-1], order) for order in orders)
    configurations = list(dict.fromkeys(configurations))
    identity = _identity(native, model_path, model_data, explicit_calculator)
    atomic_numbers = list(model_data.get("atomic_numbers", ()))
    if not atomic_numbers:
        raise ValueError("MACELES model has no atomic_numbers")
    properties = ["energy", "energies", "forces", "stress"]

    def evaluate(calculator, atoms):
        calculator.calculate(atoms, properties=properties)
        results = calculator.results
        return {
            "energy": float(results["energy"]),
            "atomic_energies": np.asarray(results["energies"]),
            "latent_charges": np.asarray(results["latent_charges"]),
            "forces": np.asarray(results["forces"]),
            "stress": np.asarray(results["stress"]),
        }

    records = []
    for case_name, charges, fractional, cell in _cases(np):
        cutoff_margin = _active_cutoff_margin(np, cell, dl)
        if cutoff_margin <= 1.0e-3:
            raise RuntimeError(
                f"{case_name} reciprocal vector lies too close to the hard cutoff"
            )
        references = []
        numbers = [
            atomic_numbers[index % len(atomic_numbers)] for index in range(charges.size)
        ]
        for phase in phases:
            positions = np.mod(fractional + phase, 1.0) @ cell
            atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
            references.append((atoms, evaluate(explicit_calculator, atoms)))
        if _selected_periodic_algorithm(explicit_calculator) != "explicit_sum":
            raise RuntimeError("reference MACELES evaluator did not use explicit_sum")
        for mesh_extent, order in configurations:
            mesh = [mesh_extent] * 3
            errors = {name: 0.0 for name in ERROR_FIELDS}
            reason = None
            samples_ns = []
            try:
                calculator = Symmetrix(
                    model_path,
                    dtype="float64",
                    streamed_edges="reference",
                    neighbor_skin=0.0,
                )
                calculator.evaluator._configure_les_particle_mesh_for_testing(
                    *mesh, order
                )
                for atoms, expected in references:
                    actual = evaluate(calculator, atoms)
                    if _selected_periodic_algorithm(calculator) != "particle_mesh":
                        raise RuntimeError(
                            "candidate MACELES evaluator did not use particle_mesh"
                        )
                    for name in ERROR_FIELDS:
                        errors[name] = max(
                            errors[name],
                            float(
                                np.max(
                                    np.abs(
                                        np.asarray(actual[name])
                                        - np.asarray(expected[name])
                                    )
                                )
                            ),
                        )
                timing_atoms = references[0][0]
                evaluate(calculator, timing_atoms)
                for _ in range(sample_count):
                    started = time.perf_counter_ns()
                    evaluate(calculator, timing_atoms)
                    samples_ns.append(time.perf_counter_ns() - started)
            except ValueError as error:
                reason = _mesh_inadequacy_reason(error)
                errors = None
                samples_ns = []
            record = {
                "schema": SCHEMA,
                **identity,
                "case": case_name,
                "algorithm": "particle_mesh",
                "reference_algorithm": "explicit_sum",
                "atoms": int(charges.size),
                "cell": np.asarray(cell).tolist(),
                "sigma": sigma,
                "dl": dl,
                "mesh": mesh,
                "spline_order": order,
                "directed_edges": len(calculator._mace_inputs(references[0][0])[3]),
                "translation_phases": [phase.tolist() for phase in phases],
                "active_cutoff_margin": cutoff_margin,
                "status": "inadequate_mesh" if reason is not None else "ok",
                "inadequacy_reason": reason,
                "errors": errors,
                "envelopes": dict(FINAL_ENVELOPES),
                "samples_ns": samples_ns,
                "sample_count": len(samples_ns),
                "median_us_per_atom": (
                    statistics.median(samples_ns) / 1000.0 / charges.size
                    if samples_ns
                    else None
                ),
            }
            records.append(validate_record(record))
    return records


def validate_endpoint(records: list[dict[str, object]]) -> None:
    """Require the 32^3/order-6 endpoint for every cell to meet its envelope."""

    endpoints = [
        record
        for record in records
        if record["mesh"] == [32, 32, 32] and record["spline_order"] == 6
    ]
    cases = {record.get("case") for record in endpoints}
    if cases != {"orthogonal", "triclinic"}:
        raise ValueError("qualification endpoint is incomplete")
    for record in endpoints:
        validate_record(record)
        if record["status"] != "ok":
            raise ValueError(f"{record['case']} endpoint mesh is inadequate")
        for name, limit in FINAL_ENVELOPES.items():
            if record["errors"][name] > limit:
                raise ValueError(
                    f"{record['case']} {name} error {record['errors'][name]} "
                    f"exceeds {limit}"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    records = run_qualification(args.model, sample_count=args.samples)
    validate_endpoint(records)
    print(json.dumps({"schema": SCHEMA, "records": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
