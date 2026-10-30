"""Qualify an exported NequIP contract against the upstream ASE calculator.

The manifest and ``--dry-run`` path intentionally use only the Python standard
library.  NequIP, ASE, NumPy, and the native Symmetrix extension are imported
inside the live comparison path so this driver remains useful for CI metadata
checks on machines without the optional stack.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from benchmarks.oam_release_contract import OAM_RELEASE_TOLERANCES
except ModuleNotFoundError as error:
    if error.name != "benchmarks":
        raise
    from oam_release_contract import OAM_RELEASE_TOLERANCES

SCHEMA = "symmetrix.nequip.oam-qualification"
VERSION = 1
DEFAULT_TOLERANCES = OAM_RELEASE_TOLERANCES["float32"]
FLOAT64_TOLERANCES = OAM_RELEASE_TOLERANCES["float64"]
PACKAGE_NAMES = ("nequip", "ase", "numpy", "torch", "e3nn", "symmetrix")


class Structure:
    __slots__ = ("numbers", "positions", "cell", "pbc")

    def __init__(
        self,
        numbers: tuple[int, ...],
        positions: tuple[tuple[float, float, float], ...],
        cell: tuple[tuple[float, float, float], ...] | None = None,
        pbc: tuple[bool, bool, bool] = (False, False, False),
    ) -> None:
        self.numbers = numbers
        self.positions = positions
        self.cell = cell
        self.pbc = pbc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGE_NAMES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{path} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _vector(value: Any, path: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"{path} must contain three coordinates")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _atomic_number(symbol: str) -> int:
    # The JSON contract normally supplies atomic numbers.  This compact table
    # covers symbols commonly used by OAM structures and the XYZ fallback.
    symbols = (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
    ).split()
    try:
        return symbols.index(symbol) + 1
    except ValueError as error:
        raise ValueError(f"unknown element symbol {symbol!r}") from error


def _structure_from_json(path: Path) -> Structure:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValueError("structure JSON root must be an object")
    positions_raw = data.get("positions")
    if not isinstance(positions_raw, Sequence) or isinstance(
        positions_raw, (str, bytes)
    ):
        raise ValueError("structure.positions must be an array")
    positions = tuple(
        _vector(item, f"structure.positions[{index}]")
        for index, item in enumerate(positions_raw)
    )
    numbers_raw = data.get("atomic_numbers")
    if numbers_raw is None:
        symbols = data.get("symbols")
        if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
            raise ValueError("structure requires atomic_numbers or symbols")
        numbers = tuple(_atomic_number(str(item)) for item in symbols)
    else:
        if not isinstance(numbers_raw, Sequence) or isinstance(
            numbers_raw, (str, bytes)
        ):
            raise ValueError("structure.atomic_numbers must be an array")
        numbers = tuple(int(item) for item in numbers_raw)
    if len(numbers) != len(positions):
        raise ValueError("structure atomic number and position lengths differ")
    cell_raw = data.get("cell")
    cell = None
    if cell_raw is not None:
        if not isinstance(cell_raw, Sequence) or len(cell_raw) != 3:
            raise ValueError("structure.cell must contain three vectors")
        cell = tuple(
            _vector(row, f"structure.cell[{index}]")
            for index, row in enumerate(cell_raw)
        )
    pbc_raw = data.get("pbc", (False, False, False))
    if isinstance(pbc_raw, bool):
        pbc = (pbc_raw, pbc_raw, pbc_raw)
    elif isinstance(pbc_raw, Sequence) and len(pbc_raw) == 3:
        pbc = tuple(bool(value) for value in pbc_raw)  # type: ignore[assignment]
    else:
        raise ValueError("structure.pbc must be a boolean or three booleans")
    if any(pbc) and cell is None:
        raise ValueError("periodic structure JSON requires cell when pbc is enabled")
    return Structure(numbers=numbers, positions=positions, cell=cell, pbc=pbc)


def _structure_from_xyz(path: Path) -> Structure:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("XYZ structure is empty")
    try:
        count = int(lines[0].strip())
    except ValueError as error:
        raise ValueError("XYZ first line must be an atom count") from error
    comment = lines[1] if len(lines) > 1 else ""
    records = lines[2 : 2 + count]
    if len(records) != count:
        raise ValueError("XYZ does not contain the declared atom count")
    numbers = []
    positions = []
    for index, record in enumerate(records):
        fields = record.split()
        if len(fields) < 4:
            raise ValueError(
                f"XYZ atom {index} must contain symbol and xyz coordinates"
            )
        numbers.append(_atomic_number(fields[0]))
        positions.append(
            tuple(_number(item, f"XYZ atom {index}") for item in fields[1:4])
        )
    cell = None
    pbc = (False, False, False)
    lattice_match = re.search(r'Lattice="([^"]+)"', comment)
    if lattice_match:
        lattice_values = [float(value) for value in lattice_match.group(1).split()]
        if len(lattice_values) != 9:
            raise ValueError("Extended-XYZ Lattice must contain nine values")
        cell = tuple(
            tuple(lattice_values[3 * row + column] for column in range(3))
            for row in range(3)
        )
    pbc_match = re.search(r'pbc="([^"]+)"', comment, re.IGNORECASE)
    if pbc_match:
        values = pbc_match.group(1).split()
        if len(values) != 3:
            raise ValueError("Extended-XYZ pbc must contain three values")
        pbc = tuple(value.upper() in {"T", "TRUE", "1"} for value in values)  # type: ignore[assignment]
    return Structure(tuple(numbers), tuple(positions), cell, pbc)


def load_structure(path: Path) -> Structure:
    if path.suffix.lower() == ".json":
        return _structure_from_json(path)
    if path.suffix.lower() in {".xyz", ".extxyz"}:
        return _structure_from_xyz(path)
    try:
        from ase.io import read
    except ImportError as error:
        raise RuntimeError(
            "ASE is required to read this structure format; use JSON or XYZ for dependency-free dry-runs"
        ) from error
    atoms = read(str(path))
    return Structure(
        tuple(int(value) for value in atoms.get_atomic_numbers()),
        tuple(
            tuple(float(component) for component in row)
            for row in atoms.get_positions()
        ),
        tuple(tuple(float(component) for component in row) for row in atoms.cell.array),
        tuple(bool(value) for value in atoms.pbc),
    )


def _cell_shift(
    structure: Structure, shift: tuple[int, int, int]
) -> tuple[float, float, float]:
    if structure.cell is None:
        return (0.0, 0.0, 0.0)
    return tuple(
        sum(shift[row] * structure.cell[row][axis] for row in range(3))
        for axis in range(3)
    )


def directed_edges(structure: Structure, cutoff: float) -> int:
    if cutoff <= 0:
        return 0
    if any(structure.pbc) and structure.cell is None:
        raise ValueError("periodic structure requires a cell")
    ranges = []
    for axis in range(3):
        if not structure.pbc[axis]:
            ranges.append((0,))
            continue
        vector = structure.cell[axis]  # type: ignore[index]
        length = math.sqrt(sum(component * component for component in vector))
        if length <= 0.0:
            raise ValueError("periodic cell vectors must be non-zero")
        radius = max(1, int(math.ceil(cutoff / length)) + 1)
        ranges.append(range(-radius, radius + 1))
    shifts = list(itertools.product(*ranges))
    shifts = [
        shift
        for shift in shifts
        if all(structure.pbc[index] or shift[index] == 0 for index in range(3))
    ]
    total = 0
    cutoff_sq = cutoff * cutoff
    for receiver, origin in enumerate(structure.positions):
        for source, position in enumerate(structure.positions):
            for shift in shifts:
                if receiver == source and shift == (0, 0, 0):
                    continue
                delta = tuple(
                    position[index]
                    + _cell_shift(structure, shift)[index]
                    - origin[index]
                    for index in range(3)
                )
                if sum(value * value for value in delta) <= cutoff_sq + 1.0e-12:
                    total += 1
    return total


def _contract_metadata(
    path: Path,
) -> tuple[str | None, str | None, float | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            contract = json.load(stream)
        if not isinstance(contract, Mapping):
            raise ValueError("contract JSON root must be an object")
        try:
            from symmetrix.nequip_schema import (
                nequip_contract_fingerprint,
                validate_nequip_oam_data,
            )
        except ImportError:
            schema_path = (
                Path(__file__).resolve().parents[1]
                / "symmetrix"
                / "source"
                / "symmetrix"
                / "nequip_schema.py"
            )
            spec = importlib.util.spec_from_file_location(
                "_symmetrix_nequip_schema", schema_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"repository schema module is unavailable at {schema_path}"
                )
            schema_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(schema_module)
            nequip_contract_fingerprint = schema_module.nequip_contract_fingerprint
            validate_nequip_oam_data = schema_module.validate_nequip_oam_data
        validate_nequip_oam_data(contract)
        cutoff = float(contract["r_cut"])
        fingerprint = nequip_contract_fingerprint(contract)
        return fingerprint, cutoff, None, None
    except Exception as error:
        return (
            None,
            None,
            None,
            f"model contract validation failed: {type(error).__name__}: {error}",
        )


def _oam_codegen_metadata(path: Path, precision: str) -> dict[str, Any]:
    """Return generated OAM identity without compiling or loading native code."""

    result: dict[str, Any] = {
        "status": "not-applicable",
        "required": False,
        "plugin_abi": None,
        "plugin_abi_version": None,
        "artifact_id": None,
        "codegen_generation": None,
        "jit_generation_version": None,
        "compiler_identity": None,
        "scalar_size": 4 if precision == "float32" else 8,
        "generic_dispatch_count": None,
        "generic_region_count": None,
    }
    try:
        with path.open("r", encoding="utf-8") as stream:
            contract = json.load(stream)
        if not isinstance(contract, Mapping) or contract.get("architecture") != "oam":
            return result
        # The generated path is a qualification requirement even when the
        # Python metadata helper is unavailable.  Unknown counters remain
        # unknown until a live evaluator supplies diagnostic evidence.
        result["required"] = True
        try:
            from symmetrix.oam_jit_codegen import oam_jit_metadata
        except ImportError:
            schema_path = (
                Path(__file__).resolve().parents[1]
                / "symmetrix"
                / "source"
                / "symmetrix"
            )
            spec = importlib.util.spec_from_file_location(
                "_symmetrix_oam_jit_codegen", schema_path / "oam_jit_codegen.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError("OAM code generator is unavailable")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            oam_jit_metadata = module.oam_jit_metadata
        metadata = oam_jit_metadata(contract, precision)
        result.update(
            {
                "status": "metadata-only",
                "required": True,
                "plugin_abi": metadata["abi"],
                "plugin_abi_version": metadata["abi_version"],
                "artifact_id": metadata["artifact_id"],
                "codegen_generation": metadata["generation_fingerprint"],
                "scalar_size": metadata["scalar_size"],
                "generic_dispatch_count": None,
                "generic_region_count": None,
            }
        )
    except Exception as error:
        result.update(
            {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
        )
    return result


def _diagnostic_count(diagnostic: Any, key: str) -> int | None:
    match = re.search(rf"(?:^|[,; ]){re.escape(key)}=(\d+)(?:[,; ]|$)", str(diagnostic))
    return int(match.group(1)) if match else None


def _artifact_manifest_metadata(path: Any) -> dict[str, Any]:
    """Read compiler identity from the validated generated artifact manifest."""

    if not path:
        return {}
    artifact_path = Path(str(path))
    manifest_path = (
        artifact_path
        if artifact_path.name == "manifest.json"
        else artifact_path.parent / "manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    key_inputs = manifest.get("key_inputs", {})
    build = key_inputs.get("build", {}) if isinstance(key_inputs, Mapping) else {}
    compiler = key_inputs.get("compiler") if isinstance(key_inputs, Mapping) else None
    request = build.get("request", {}) if isinstance(build, Mapping) else {}
    oam = request.get("oam", {}) if isinstance(request, Mapping) else {}
    abi = request.get("abi", {}) if isinstance(request, Mapping) else {}
    result = {
        "compiler_identity": compiler,
        "compiler_command": manifest.get("compiler_command"),
        "jit_generation_version": (
            key_inputs.get("jit_generation_version")
            if isinstance(key_inputs, Mapping)
            else None
        )
        or (
            build.get("jit_generation_version") if isinstance(build, Mapping) else None
        ),
        "manifest_path": str(manifest_path),
    }
    if isinstance(oam, Mapping):
        result.update(
            {
                "artifact_id": oam.get("artifact_id"),
                "codegen_generation": oam.get("generation_fingerprint"),
                "scalar_size": oam.get("scalar_size"),
            }
        )
    if isinstance(abi, Mapping):
        result.update(
            {
                "plugin_abi": abi.get("tag"),
                "plugin_abi_version": abi.get("version"),
            }
        )
    return result


def _properties(properties: Iterable[str]) -> tuple[str, ...]:
    allowed = {"energy", "forces", "stress"}
    result = tuple(
        dict.fromkeys(
            str(item).strip().lower() for item in properties if str(item).strip()
        )
    )
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"unknown properties: {sorted(unknown)}")
    if not result:
        raise ValueError("at least one comparison property is required")
    return result


#: Fields a manifest must carry to claim a completed qualification.  A record
#: whose status is ``passed`` asserts a measured parity result, so the workload
#: it reports must be fully identified.
_QUALIFICATION_IDENTITY_FIELDS = (
    "model_sha256",
    "contract_fingerprint",
    "cutoff",
    "atom_count",
    "directed_edges",
)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = (
        "model_sha256",
        "contract_fingerprint",
        "upstream_package_versions",
        "cutoff",
        "atom_count",
        "directed_edges",
        "tolerances",
        "execution_diagnostics",
        "generated_qualification",
    )
    for key in required:
        if key not in manifest:
            raise ValueError(f"manifest missing required field {key}")
    model_hash = manifest["model_sha256"]
    if model_hash is not None and (
        not isinstance(model_hash, str) or len(model_hash) != 64
    ):
        raise ValueError(
            "manifest model_sha256 must be a 64-character SHA-256 hex digest"
        )
    fingerprint = manifest["contract_fingerprint"]
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise ValueError("manifest contract_fingerprint must be a string or null")
    if not isinstance(manifest["upstream_package_versions"], Mapping):
        raise ValueError("manifest upstream_package_versions must be an object")
    if (
        manifest["cutoff"] is not None
        and _number(manifest["cutoff"], "manifest.cutoff") <= 0
    ):
        raise ValueError("manifest cutoff must be positive")
    for key in ("atom_count", "directed_edges"):
        value = manifest[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"manifest {key} must be a non-negative integer or null")
    tolerances = manifest["tolerances"]
    if not isinstance(tolerances, Mapping):
        raise ValueError("manifest tolerances must be an object")
    for key in DEFAULT_TOLERANCES:
        if (
            key not in tolerances
            or _number(tolerances[key], f"manifest.tolerances.{key}") < 0
        ):
            raise ValueError(f"manifest tolerances missing non-negative {key}")
    if not isinstance(manifest["execution_diagnostics"], Mapping):
        raise ValueError("manifest execution_diagnostics must be an object")
    generated = manifest["generated_qualification"]
    if not isinstance(generated, Mapping):
        raise ValueError("manifest generated_qualification must be an object")
    for key in (
        "plugin_abi",
        "plugin_abi_version",
        "artifact_id",
        "codegen_generation",
        "compiler_identity",
        "scalar_size",
        "generic_dispatch_count",
        "generic_region_count",
    ):
        if key not in generated:
            raise ValueError(
                f"manifest generated_qualification missing required field {key}"
            )
    for key in ("plugin_abi", "artifact_id", "codegen_generation"):
        value = generated[key]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(
                f"manifest generated_qualification.{key} must be a non-empty string or null"
            )
    compiler_identity = generated["compiler_identity"]
    if compiler_identity is not None and not isinstance(
        compiler_identity, (str, Mapping)
    ):
        raise ValueError(
            "manifest generated_qualification.compiler_identity must be a string, object, or null"
        )
    if isinstance(compiler_identity, str) and not compiler_identity:
        raise ValueError(
            "manifest generated_qualification.compiler_identity string must not be empty"
        )
    abi_version = generated["plugin_abi_version"]
    if abi_version is not None and (
        isinstance(abi_version, bool)
        or not isinstance(abi_version, int)
        or abi_version <= 0
    ):
        raise ValueError(
            "manifest generated_qualification.plugin_abi_version must be a positive integer or null"
        )
    scalar_size = generated["scalar_size"]
    if scalar_size is not None and (
        isinstance(scalar_size, bool)
        or not isinstance(scalar_size, int)
        or scalar_size not in (4, 8)
    ):
        raise ValueError(
            "manifest generated_qualification.scalar_size must be 4, 8, or null"
        )
    for key in ("generic_dispatch_count", "generic_region_count"):
        value = generated.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(
                f"manifest generated_qualification.{key} must be non-negative or null"
            )
    # A record only asserts a measured parity result when it is ``passed``.  Its
    # workload identity (model hash, contract fingerprint, cutoff, atom count,
    # directed edge count) is then required: a null there would make the claim
    # unreproducible, which is exactly the record-completeness rule the
    # null-tolerant shape checks above cannot enforce.
    if manifest.get("status") == "passed":
        missing = [
            key for key in _QUALIFICATION_IDENTITY_FIELDS if manifest.get(key) is None
        ]
        if missing:
            raise ValueError(
                "a passed qualification manifest must record " + ", ".join(missing)
            )


def _skip_manifest(reason: str, **values: Any) -> dict[str, Any]:
    values.update({"status": "skipped", "skip_reason": reason})
    return values


def build_manifest(
    *,
    model_json: Path,
    reference_model: Path | None,
    structure: Path | None,
    precision: str,
    backend: str,
    target: str | None = None,
    neighbor_skin: float,
    properties: Iterable[str] = ("energy", "forces"),
    dry_run: bool = False,
) -> dict[str, Any]:
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be float32 or float64")
    if backend not in {"cpu", "cuda", "hip"}:
        raise ValueError("backend must be cpu, cuda, or hip")
    if neighbor_skin < 0 or not math.isfinite(float(neighbor_skin)):
        raise ValueError("neighbor-skin must be finite and non-negative")
    requested = _properties(properties)
    model_json = Path(model_json)
    model_hash = sha256_file(model_json) if model_json.is_file() else None
    fingerprint, cutoff, _, contract_error = (
        _contract_metadata(model_json)
        if model_json.is_file()
        else (None, None, None, "model JSON does not exist")
    )
    atom_count = None
    edge_count = None
    structure_error = None
    if structure is not None:
        try:
            structure_path = Path(structure)
            if dry_run and structure_path.suffix.lower() not in {
                ".json",
                ".xyz",
                ".extxyz",
            }:
                raise RuntimeError(
                    "dependency-free dry-run reads JSON or XYZ structures; "
                    "ASE is required for this structure format"
                )
            loaded = load_structure(structure_path)
            atom_count = len(loaded.numbers)
            if cutoff is not None:
                edge_count = directed_edges(loaded, cutoff + float(neighbor_skin))
        except Exception as error:
            structure_error = f"structure unavailable: {type(error).__name__}: {error}"
    generated = _oam_codegen_metadata(model_json, precision)
    generated["required"] = bool(generated.get("required") and backend != "cpu")
    diagnostics: dict[str, Any] = {
        "backend": backend,
        "requested_backend": backend,
        "selected_backend": None,
        "precision": precision,
        "neighbor_skin_A": float(neighbor_skin),
        "requested_properties": list(requested),
        "reference_adapter": "nequip.integrations.ase.NequIPCalculator",
        "mode": "dry-run" if dry_run else "comparison",
    }
    requested_target = target or ("native" if backend == "cpu" else "generic")
    try:
        from symmetrix.nequip_schema import load_nequip_oam_data

        artifact = (
            {
                "status": "validated",
                "contract_fingerprint": fingerprint,
                "backend": backend,
                "target": requested_target,
            }
            if model_hash is not None
            else None
        )
        if model_hash is not None:
            load_nequip_oam_data(model_json)
    except (ImportError, OSError, TypeError, ValueError) as error:
        artifact = {
            "status": "unavailable",
            "reason": f"OAM contract readiness unavailable: {type(error).__name__}: {error}",
        }
    diagnostics["artifact"] = artifact
    diagnostics["generated_qualification"] = generated
    upstream_versions = package_versions()
    missing_prerequisites = [
        name for name in ("nequip", "ase") if upstream_versions.get(name) is None
    ]
    oam_references: dict[str, dict[str, Any]] = {}
    for size in ("S", "M", "L", "XL"):
        variable = f"SYMMETRIX_NEQUIP_OAM_{size}"
        configured = (
            str(reference_model)
            if size == "S" and reference_model is not None
            else os.environ.get(variable)
        )
        if configured is None:
            status = "skipped"
            reason = (
                f"optional OAM-{size} reference is unavailable ({variable} is unset)"
            )
        elif not Path(configured).is_file():
            status = "skipped"
            reason = f"configured OAM-{size} reference does not exist: {configured}"
        elif missing_prerequisites:
            status = "skipped"
            reason = f"optional OAM-{size} prerequisites are unavailable: {', '.join(missing_prerequisites)}"
        else:
            status = "configured"
            reason = None
        oam_references[size] = {
            "environment": variable,
            "required": size == "S",
            "path": configured,
            "status": status,
            "reason": reason,
        }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "dry-run" if dry_run else "pending",
        "model_json": str(model_json),
        "reference_model": str(reference_model)
        if reference_model is not None
        else None,
        "model_sha256": model_hash,
        "contract_fingerprint": fingerprint,
        "upstream_package_versions": upstream_versions,
        "cutoff": cutoff,
        "neighbor_skin_A": float(neighbor_skin),
        "effective_cutoff": cutoff + float(neighbor_skin)
        if cutoff is not None
        else None,
        "atom_count": atom_count,
        "directed_edges": edge_count,
        "tolerances": dict(
            FLOAT64_TOLERANCES if precision == "float64" else DEFAULT_TOLERANCES
        ),
        "execution_diagnostics": diagnostics,
        "generated_qualification": generated,
        "target": requested_target,
        "requested_properties": list(requested),
        "oam_references": oam_references,
        "property_status": {
            name: {"status": "requested", "reason": None} for name in requested
        },
    }
    errors = [item for item in (contract_error, structure_error) if item]
    if errors:
        manifest["status"] = "failed" if contract_error else "skipped"
        manifest["skip_reason"] = "; ".join(errors)
    elif not dry_run:
        reference = reference_model or (
            Path(os.environ["SYMMETRIX_NEQUIP_OAM_S"])
            if os.environ.get("SYMMETRIX_NEQUIP_OAM_S")
            else None
        )
        if reference is None:
            manifest.update(
                _skip_manifest(
                    "OAM-S reference is unavailable; set SYMMETRIX_NEQUIP_OAM_S or pass --reference-model"
                )
            )
        elif not Path(reference).is_file():
            manifest.update(
                _skip_manifest(f"reference model does not exist: {reference}")
            )
        elif structure is None or atom_count is None:
            manifest.update(
                _skip_manifest(
                    "a readable --structure is required for reference comparison"
                )
            )
        else:
            manifest["reference_model"] = str(reference)
            _run_comparison(
                manifest,
                model_json,
                Path(reference),
                Path(structure),
                precision,
                backend,
                requested,
            )
    validate_manifest(manifest)
    return manifest


def _flatten(value: Any) -> list[float]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if converted is not value:
            return _flatten(converted)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [float(value)]


def _max_abs_difference(left: Any, right: Any) -> float:
    values_left, values_right = _flatten(left), _flatten(right)
    if len(values_left) != len(values_right):
        raise ValueError(
            f"comparison shapes differ: {len(values_left)} != {len(values_right)}"
        )
    return max((abs(a - b) for a, b in zip(values_left, values_right)), default=0.0)


def _mark_comparison_failure(manifest: dict[str, Any], error: Exception) -> None:
    manifest["status"] = "failed"
    manifest["error"] = f"{type(error).__name__}: {error}"


def _backend_mismatch(backend: str, selected: Any, diagnostic: Any) -> str | None:
    requested = backend.lower()
    selected_text = str(selected).lower() if selected is not None else ""
    diagnostic_text = str(diagnostic).lower()
    combined = f"{selected_text} {diagnostic_text}"
    if requested == "cpu" and selected_text in {"cuda", "hip", "rocm"}:
        return f"requested cpu, selected {selected}"
    if requested == "cuda" and selected_text in {
        "hip",
        "rocm",
        "openmp",
        "serial",
        "host",
        "cpu",
    }:
        return f"requested cuda, selected {selected}"
    if requested == "hip" and selected_text in {
        "cuda",
        "openmp",
        "serial",
        "host",
        "cpu",
    }:
        return f"requested hip, selected {selected}"
    if requested == "cpu" and any(
        token in combined for token in ("cuda", "hip", "rocm")
    ):
        return f"requested cpu, selected diagnostic {diagnostic}"
    if requested == "cuda" and any(
        token in combined for token in ("hip", "rocm", "openmp", "serial")
    ):
        return f"requested cuda, selected diagnostic {diagnostic}"
    if requested == "hip" and any(
        token in combined for token in ("cuda", "openmp", "serial")
    ):
        return f"requested hip, selected diagnostic {diagnostic}"
    return None


def _run_comparison(
    manifest: dict[str, Any],
    model_json: Path,
    reference: Path,
    structure: Path,
    precision: str,
    backend: str,
    requested: tuple[str, ...],
) -> None:
    try:
        from ase import Atoms
        from ase.io import read

        try:
            from nequip.integrations.ase import (
                NequIPCalculator as ReferenceCalculator,
            )
        except ImportError:
            from nequip.ase import NequIPCalculator as ReferenceCalculator
    except ImportError as error:
        manifest.update(
            _skip_manifest(
                "reference comparison requires optional packages 'nequip' and 'ase'; "
                f"install them in the qualification environment ({error})"
            )
        )
        return
    try:
        from symmetrix import NequIPCalculator
    except ImportError as error:
        manifest.update(
            _skip_manifest(f"native Symmetrix NequIP evaluator is unavailable: {error}")
        )
        return
    try:
        if structure.suffix.lower() in {".json", ".xyz", ".extxyz"}:
            parsed = load_structure(structure)
            atoms = Atoms(
                numbers=parsed.numbers,
                positions=parsed.positions,
                cell=parsed.cell,
                pbc=parsed.pbc,
            )
        else:
            atoms = read(str(structure))
        device = "cuda" if backend == "cuda" else "cpu"
        # NequIP 0.19's ASE integration accepts an instantiated model.  Its
        # saved-model factory also installs the package's official species and
        # neighbor-list transforms, which are required for .nequip.zip OAM
        # archives (passing the archive path directly is not supported).
        factory = getattr(ReferenceCalculator, "_from_saved_model", None)
        if callable(factory) and str(reference).endswith((".nequip.zip", ".nequip")):
            reference_calculator = factory(
                str(reference),
                device=device,
                chemical_species_to_atom_type_map=True,
            )
        else:
            try:
                reference_calculator = ReferenceCalculator(
                    str(reference), device=device
                )
            except TypeError:
                try:
                    reference_calculator = ReferenceCalculator(
                        model_path=str(reference), device=device
                    )
                except (AttributeError, TypeError):
                    factory = getattr(ReferenceCalculator, "from_deployed_model", None)
                    if not callable(factory):
                        raise
                    reference_calculator = factory(str(reference), device=device)
        # ``--backend cpu`` qualifies the artifact-free reference routing; the
        # accelerator spellings qualify the generated receiver routing.
        requested_routing = "reference" if backend == "cpu" else "direct"
        candidate_calculator = NequIPCalculator(
            str(model_json),
            dtype=precision,
            streamed_edges=requested_routing,
            neighbor_skin=manifest["neighbor_skin_A"],
        )
        evaluator = getattr(candidate_calculator, "evaluator", None)
        if "stress" in requested and not bool(
            getattr(evaluator, "supports_stress", False)
        ):
            reason = "selected native evaluator does not support stress"
            manifest["property_status"]["stress"] = {
                "status": "skipped",
                "reason": reason,
            }
            manifest.update(_skip_manifest(reason))
            return
        generated = manifest["generated_qualification"]
        artifact_path = (
            getattr(evaluator, "cuda_r1_manifest_path", None)
            if backend == "cuda"
            else getattr(evaluator, "jit_oam_host_plugin_path", None)
        )
        generated.update(_artifact_manifest_metadata(artifact_path))
        evaluator_artifact_id = (
            getattr(evaluator, "cuda_r1_artifact_id", None)
            if backend == "cuda"
            else getattr(evaluator, "jit_oam_host_plugin_artifact_id", None)
        )
        if evaluator_artifact_id:
            generated["artifact_id"] = evaluator_artifact_id
            generated["status"] = "validated"
        selected = getattr(candidate_calculator, "backend", None)
        if selected is None:
            selected = getattr(
                getattr(candidate_calculator, "evaluator", None), "backend", None
            )
        if selected is None:
            selected = getattr(
                getattr(candidate_calculator, "evaluator", None),
                "execution_backend",
                None,
            )
            if callable(selected):
                selected = selected()
        diagnostics = manifest["execution_diagnostics"]
        diagnostics["reference_adapter"] = (
            f"{ReferenceCalculator.__module__}.{ReferenceCalculator.__name__}"
        )
        diagnostics["requested_backend"] = backend
        diagnostics["selected_backend"] = selected
        diagnostics["requested_routing"] = requested_routing
        diagnostics["selected_routing"] = candidate_calculator.streamed_edges
        atoms.calc = reference_calculator
        reference_values: dict[str, Any] = {}
        if "energy" in requested:
            reference_values["energy"] = float(atoms.get_potential_energy())
        if "forces" in requested:
            reference_values["forces"] = atoms.get_forces()
        if "stress" in requested:
            reference_values["stress"] = atoms.get_stress()
        atoms.calc = candidate_calculator
        candidate_values: dict[str, Any] = {}
        if "energy" in requested:
            candidate_values["energy"] = float(atoms.get_potential_energy())
        if "forces" in requested:
            candidate_values["forces"] = atoms.get_forces()
        if "stress" in requested:
            candidate_values["stress"] = atoms.get_stress()
        observed: dict[str, float] = {}
        if "energy" in requested:
            observed["energy_abs_eV_per_atom"] = abs(
                candidate_values["energy"] - reference_values["energy"]
            ) / max(len(atoms), 1)
        if "forces" in requested:
            observed["forces_max_abs_eV_per_A"] = _max_abs_difference(
                candidate_values["forces"], reference_values["forces"]
            )
        if "stress" in requested:
            observed["stress_max_abs_eV_per_A3"] = _max_abs_difference(
                candidate_values["stress"], reference_values["stress"]
            )
        manifest["observed_errors"] = observed
        diagnostic = getattr(candidate_calculator, "execution_diagnostic", "unknown")
        generated["generic_dispatch_count"] = _diagnostic_count(
            diagnostic, "edge_generic_dispatches"
        )
        generated["generic_region_count"] = _diagnostic_count(
            diagnostic, "edge_generic_regions"
        )
        if generated.get("required") and (
            generated["generic_dispatch_count"] is None
            or generated["generic_region_count"] is None
        ):
            _mark_comparison_failure(
                manifest,
                RuntimeError(
                    "generated OAM qualification did not report both generic tensor-product counters"
                ),
            )
            return
        if generated.get("required") and (
            generated["generic_dispatch_count"] != 0
            or generated["generic_region_count"] != 0
        ):
            _mark_comparison_failure(
                manifest,
                RuntimeError(
                    "generated OAM qualification observed generic tensor-product dispatch"
                ),
            )
            return
        manifest["execution_diagnostics"]["generated_qualification"] = dict(generated)
        mismatch = _backend_mismatch(backend, selected, diagnostic)
        if backend in {"cuda", "hip"} and str(diagnostic).lower() in {
            "",
            "unknown",
            "uninitialized",
        }:
            mismatch = (
                f"requested {backend}, native evaluator diagnostic is unavailable"
            )
        if mismatch:
            manifest.update(
                _skip_manifest(f"selected evaluator backend mismatch: {mismatch}")
            )
            return
        manifest["execution_diagnostics"].update(
            {
                "candidate": diagnostic,
                "reference": diagnostics["reference_adapter"],
            }
        )
        manifest["status"] = (
            "passed"
            if all(observed[key] <= manifest["tolerances"][key] for key in observed)
            else "failed"
        )
        if manifest["status"] == "failed":
            manifest["error"] = (
                "one or more parity errors exceeded the requested tolerance"
            )
    except Exception as error:
        _mark_comparison_failure(manifest, error)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path)
    parser.add_argument("--structure", type=Path)
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda", "hip"),
        default="cpu",
        help=(
            "candidate execution target; 'cpu' qualifies the artifact-free "
            "reference routing, and the accelerator spellings qualify the "
            "generated receiver routing"
        ),
    )
    parser.add_argument(
        "--target",
        help=(
            "backend architecture target (for example sm90 or gfx942); defaults "
            "to 'native' for --backend cpu and to generic accelerator metadata "
            "otherwise"
        ),
    )
    parser.add_argument("--neighbor-skin", type=float, default=0.5)
    parser.add_argument(
        "--properties",
        default="energy,forces",
        help="comma-separated properties to compare",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write metadata without importing optional runtimes",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = make_parser().parse_args() if args is None else args
    kokkos_was_initialized = False
    symmetrix_module = sys.modules.get("symmetrix")
    load_backend = getattr(symmetrix_module, "load_backend", None)
    if callable(load_backend):
        native = load_backend()
        is_initialized = getattr(native, "_kokkos_is_initialized", None)
        if callable(is_initialized):
            kokkos_was_initialized = bool(is_initialized())
    try:
        manifest = build_manifest(
            model_json=args.model_json,
            reference_model=args.reference_model,
            structure=args.structure,
            precision=args.precision,
            backend=args.backend,
            target=args.target,
            neighbor_skin=args.neighbor_skin,
            properties=args.properties.split(","),
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as error:
        print(f"nequip_oam_equivalence: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "output": str(args.output)}))
    gc.collect()
    symmetrix_module = sys.modules.get("symmetrix")
    load_backend = getattr(symmetrix_module, "load_backend", None)
    if callable(load_backend):
        native = load_backend()
        is_initialized = getattr(native, "_kokkos_is_initialized", None)
        finalize = getattr(native, "_finalize_kokkos", None)
        if (
            not kokkos_was_initialized
            and callable(is_initialized)
            and callable(finalize)
            and is_initialized()
        ):
            finalize()
    return 0 if manifest["status"] in {"dry-run", "passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
