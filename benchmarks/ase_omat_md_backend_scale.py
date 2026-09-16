"""Production-path ASE MD benchmark for the official OMAT-0 medium model."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


SCHEMA_VERSION = 3
BACKENDS = ("kokkos_all", "factorized", "torch_cueq")
ENSEMBLES = ("nve", "npt")
EXPECTED_CHECKPOINT_SHA256 = (
    "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a"
)
EXPECTED_COMPACT_SHA256 = (
    "9f69ea29c0f0f25b0d3af7a485529af06cb93ca0dca7617e401fbcfe70c7baa4"
)
DEFAULT_CHECKPOINT = pathlib.Path(
    os.environ.get("OMAT0_MEDIUM_CHECKPOINT", "mace-omat-0-medium.model")
)
DEFAULT_COMPACT_MODEL = pathlib.Path(
    os.environ.get(
        "OMAT0_MEDIUM_COMPACT_MODEL",
        "mace-omat-0-medium-universal-v2-factorized.json",
    )
)
DEFAULT_REPEATS = (4, 6, 8, 10, 12)
DEFAULT_STEPS = 20
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_TIMESTEP_FS = 1.0
DEFAULT_EXTERNAL_PRESSURE_GPA = 0.0
DEFAULT_THERMOSTAT_TIME_FS = 25.0
DEFAULT_BAROSTAT_TIME_FS = 75.0
DEFAULT_BULK_MODULUS_GPA = 210.0
MODEL_CUTOFF_A = 6.0
SYMMETRIX_NEIGHBOR_SKIN_A = 0.5
DEFAULT_SEED = 20260804


def _bootstrap_explicit_symmetrix() -> None:
    """Load a requested source tree and native extension in a fresh worker."""

    if "symmetrix" in sys.modules:
        return
    source_root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension = os.environ.get("SYMMETRIX_EXTENSION")
    if source_root is None and extension is None:
        return
    if source_root is None or extension is None:
        raise RuntimeError(
            "SYMMETRIX_SOURCE_ROOT and SYMMETRIX_EXTENSION must be set together"
        )
    package_dir = pathlib.Path(source_root).resolve() / "symmetrix/source/symmetrix"
    extension_path = pathlib.Path(extension).resolve()
    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(f"invalid SYMMETRIX_SOURCE_ROOT: {source_root}")
    if not extension_path.is_file():
        raise RuntimeError(f"invalid SYMMETRIX_EXTENSION: {extension}")

    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if type(finder).__name__ != "ScikitBuildRedirectingFinder"
    ]
    package_spec = importlib.util.spec_from_file_location(
        "symmetrix",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    native_spec = importlib.util.spec_from_file_location(
        "symmetrix.symmetrix", extension_path
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError("could not create the explicit Symmetrix package spec")
    if native_spec is None or native_spec.loader is None:
        raise RuntimeError("could not create the explicit Symmetrix extension spec")
    package = importlib.util.module_from_spec(package_spec)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix"] = package
    sys.modules["symmetrix.symmetrix"] = native
    try:
        native_spec.loader.exec_module(native)
        package_spec.loader.exec_module(package)
    except BaseException:
        sys.modules.pop("symmetrix.symmetrix", None)
        sys.modules.pop("symmetrix", None)
        raise


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_artifact(
    path: pathlib.Path, expected_sha256: str, label: str
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    digest = sha256_file(resolved)
    if digest != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest,
    }


def verify_compact_model(
    path: pathlib.Path,
    *,
    expected_sha256: str | None = None,
    require_execution_contracts: bool = True,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"OMAT-0 compact model does not exist: {resolved}")
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            "OMAT-0 compact model SHA-256 mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )
    try:
        with resolved.open(encoding="utf-8") as handle:
            model = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read compact model {resolved}: {exc}") from exc
    if model.get("model_type", "MACE") != "MACE":
        raise RuntimeError("compact model is not ordinary MACE")
    if int(model.get("num_channels", 0)) != 128:
        raise RuntimeError("compact model does not have 128 OMAT-medium channels")
    if not math.isclose(float(model.get("r_cut", 0.0)), 6.0):
        raise RuntimeError("compact model does not have the 6 Angstrom OMAT cutoff")
    contracts = model.get("execution_contracts", {})
    if require_execution_contracts:
        missing = sorted({"M0", "R0", "R1"} - set(contracts))
        if missing:
            raise RuntimeError(
                "compact model lacks current Factorized contracts "
                f"{missing}; regenerate it with the extract subcommand"
            )
        for name in ("M0", "R0", "R1"):
            contract = contracts[name]
            if not contract.get("generation_fingerprint"):
                raise RuntimeError(
                    f"compact model Factorized {name} contract is unfingerprinted"
                )
    provenance_path = resolved.with_suffix(resolved.suffix + ".provenance.json")
    provenance = load_json(provenance_path) if provenance_path.is_file() else None
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest,
        "historical_uncontracted_sha256": EXPECTED_COMPACT_SHA256,
        "execution_contract_fingerprints": {
            name: contracts.get(name, {}).get("generation_fingerprint")
            for name in ("M0", "R0", "R1")
        },
        "provenance": provenance,
    }


def atomic_compact_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def derive_compact_model(
    checkpoint: pathlib.Path, output: pathlib.Path
) -> dict[str, Any]:
    checkpoint_record = verify_artifact(
        checkpoint, EXPECTED_CHECKPOINT_SHA256, "OMAT-0 medium checkpoint"
    )
    _bootstrap_explicit_symmetrix()
    from symmetrix.extract_mace_data import extract_mace_data

    started = time.perf_counter()
    model = extract_mace_data(checkpoint_record["path"], radial_format="compact")
    contracts = model.get("execution_contracts", {})
    missing = sorted({"M0", "R0", "R1"} - set(contracts))
    if missing:
        raise RuntimeError(f"current extractor omitted Factorized contracts: {missing}")
    atomic_compact_json(output, model)
    result = verify_compact_model(output, require_execution_contracts=True)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": checkpoint_record,
        "compact_model": {key: result[key] for key in ("path", "size_bytes", "sha256")},
        "execution_contract_fingerprints": result["execution_contract_fingerprints"],
        "extraction_seconds": time.perf_counter() - started,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "tracked_tree_clean": not bool(
            _git_output("status", "--porcelain", "--untracked-files=no")
        ),
        "packages": package_versions(),
    }
    atomic_json(output.with_suffix(output.suffix + ".provenance.json"), provenance)
    return verify_compact_model(output, require_execution_contracts=True)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], check=False, capture_output=True, text=True, timeout=10
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def build_initial_state(
    repeat: int,
    *,
    seed: int = DEFAULT_SEED,
    temperature_K: float = DEFAULT_TEMPERATURE_K,
) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    from ase.build import bulk
    from ase.md.velocitydistribution import Stationary, thermalize_momenta

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat, repeat, repeat))
    rng_seed = int(seed) + 1_000_003 * int(repeat)
    thermalize_momenta(
        atoms,
        temperature_K=temperature_K,
        rng=np.random.default_rng(rng_seed),
        exact_temperature=False,
    )
    Stationary(atoms, preserve_temperature=True)
    state = {
        "schema_version": SCHEMA_VERSION,
        "repeat": int(repeat),
        "seed": int(seed),
        "rng_seed": rng_seed,
        "requested_temperature_K": float(temperature_K),
        "actual_temperature_K": float(atoms.get_temperature()),
        "numbers": np.asarray(atoms.numbers, dtype=np.int32),
        "masses": np.asarray(atoms.get_masses(), dtype=np.float64),
        "cell_A": np.asarray(atoms.cell.array, dtype=np.float64),
        "pbc": np.asarray(atoms.pbc, dtype=np.bool_),
        "positions_A": np.asarray(atoms.positions, dtype=np.float64),
        "momenta_eV_fs_per_A": np.asarray(atoms.get_momenta(), dtype=np.float64),
    }
    state["state_sha256"] = state_sha256(state)
    return state


def state_sha256(state: dict[str, Any]) -> str:
    return array_sha256(
        np.asarray(state["numbers"], dtype=np.int32),
        np.asarray(state["masses"], dtype=np.float64),
        np.asarray(state["cell_A"], dtype=np.float64),
        np.asarray(state["pbc"], dtype=np.bool_),
        np.asarray(state["positions_A"], dtype=np.float64),
        np.asarray(state["momenta_eV_fs_per_A"], dtype=np.float64),
    )


def save_state(path: pathlib.Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        metadata = {
            key: value
            for key, value in state.items()
            if not isinstance(value, np.ndarray)
        }
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                numbers=state["numbers"],
                masses=state["masses"],
                cell_A=state["cell_A"],
                pbc=state["pbc"],
                positions_A=state["positions_A"],
                momenta_eV_fs_per_A=state["momenta_eV_fs_per_A"],
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_state(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        state = json.loads(str(archive["metadata"]))
        for key in (
            "numbers",
            "masses",
            "cell_A",
            "pbc",
            "positions_A",
            "momenta_eV_fs_per_A",
        ):
            state[key] = np.array(archive[key], copy=True)
    digest = state_sha256(state)
    if digest != state.get("state_sha256"):
        raise RuntimeError(
            f"state SHA-256 mismatch: expected {state.get('state_sha256')}, got {digest}"
        )
    return state


def atoms_from_state(state: dict[str, Any]):
    from ase import Atoms

    atoms = Atoms(
        numbers=state["numbers"],
        positions=state["positions_A"],
        cell=state["cell_A"],
        pbc=state["pbc"],
        masses=state["masses"],
    )
    atoms.set_momenta(state["momenta_eV_fs_per_A"])
    return atoms


class CountingCalculator:
    """ASE calculator proxy that exposes the production evaluation count."""

    def __init__(self, delegate):
        from ase.calculators.calculator import Calculator

        class _Proxy(Calculator):
            implemented_properties = list(delegate.implemented_properties)

            def __init__(self):
                super().__init__()
                self.evaluation_count = 0

            def calculate(self, atoms=None, properties=None, system_changes=None):
                properties = properties or ["energy", "forces"]
                system_changes = system_changes or []
                super().calculate(atoms, properties, system_changes)
                self.evaluation_count += 1
                delegate.calculate(atoms, properties, system_changes)
                self.results = {
                    key: np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in delegate.results.items()
                }

        self.delegate = delegate
        self.proxy = _Proxy()


def timing_summary(samples_s: list[float], atom_count: int) -> dict[str, Any]:
    if not samples_s or any(sample <= 0.0 for sample in samples_s):
        raise ValueError("timing samples must be positive and nonempty")
    samples_ms = [1000.0 * float(sample) for sample in samples_s]
    ordered = sorted(samples_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    mean_ms = statistics.fmean(samples_ms)
    stdev_ms = statistics.pstdev(samples_ms)
    median_ms = statistics.median(samples_ms)
    return {
        "samples_ms": samples_ms,
        "total_ms": sum(samples_ms),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "median_us_per_atom": 1000.0 * median_ms / atom_count,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "stdev_ms": stdev_ms,
        "p95_ms": ordered[p95_index],
        "coefficient_of_variation": stdev_ms / mean_ms if mean_ms else 0.0,
        "median_ms_per_atom": median_ms / atom_count,
        "atom_steps_per_second": atom_count * 1000.0 / median_ms,
    }


def _read_status_memory(pid: int | str = "self") -> dict[str, float | None]:
    result = {"rss_mib": None, "peak_rss_mib": None}
    try:
        with pathlib.Path(f"/proc/{pid}/status").open() as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key == "VmRSS":
                    result["rss_mib"] = int(value.split()[0]) / 1024.0
                elif key == "VmHWM":
                    result["peak_rss_mib"] = int(value.split()[0]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return result


def _read_system_memory() -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "available_mib": None,
        "swap_free_mib": None,
    }
    mapping = {"MemAvailable": "available_mib", "SwapFree": "swap_free_mib"}
    try:
        with pathlib.Path("/proc/meminfo").open() as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in mapping:
                    values[mapping[key]] = int(value.split()[0]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return values


def _run_nvidia_smi(query: str) -> list[list[str]] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-{query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _number(value: str) -> int | float | str | None:
    if value in {"N/A", "[Not Supported]", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def gpu_telemetry() -> dict[str, Any] | None:
    fields = (
        "index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,"
        "clocks.sm,clocks.mem,power.draw"
    )
    rows = _run_nvidia_smi(f"gpu={fields}")
    if not rows:
        return None
    names = fields.split(",")
    devices = [dict(zip(names, map(_number, row))) for row in rows]
    process_rows = _run_nvidia_smi("compute-apps=pid,process_name,used_gpu_memory")
    processes = []
    if process_rows is not None:
        for row in process_rows:
            if len(row) == 3:
                processes.append(
                    {
                        "pid": _number(row[0]),
                        "process_name": row[1],
                        "used_gpu_memory_mib": _number(row[2]),
                    }
                )
    return {"devices": devices, "compute_processes": processes}


KOKKOS_METRICS = (
    "factorized_ready",
    "factorized_workspace_bytes",
    "factorized_workspace_capacity_bytes",
    "factorized_schedule_build_count",
    "factorized_schedule_entries",
    "factorized_schedule_bytes",
    "factorized_graph_generation",
    "factorized_prepared_graph_count",
    "factorized_prepared_evaluation_count",
    "factorized_fallback_evaluation_count",
    "execution_geometry_workspace_bytes",
    "execution_geometry_capacity_edges",
    "execution_geometry_allocation_count",
    "standard_r0_workspace_bytes",
    "factorized_unified_workspace_bytes",
    "factorized_radial_workspace_bytes",
    "factorized_arena_workspace_bytes",
    "factorized_arena_allocation_count",
    "factorized_coupling_workspace_bytes",
    "factorized_coupling_capacity_bytes",
    "factorized_compact_workspace_bytes",
    "execution_prepared_geometry_update_count",
    "execution_geometry_copy_count",
    "factorized_fractional_geometry_preparation_count",
    "factorized_fractional_geometry_initialization_bytes",
    "factorized_cell_update_count",
    "factorized_cell_update_bytes",
    "factorized_geometry_state_allocation_count",
)


def _plain(value: Any) -> Any:
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def capture_telemetry(backend: str, calculator=None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "monotonic_s": time.monotonic(),
        "process": _read_status_memory(),
        "system": _read_system_memory(),
        "gpu": gpu_telemetry(),
    }
    if backend == "torch_cueq":
        try:
            import torch

            record["torch_cuda"] = {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        except (ImportError, RuntimeError):
            record["torch_cuda"] = None
    if backend in {"kokkos_all", "factorized"} and calculator is not None:
        evaluator = getattr(calculator, "evaluator", None)
        record["kokkos"] = {
            name: _plain(getattr(evaluator, name, None)) for name in KOKKOS_METRICS
        }
        record["neighbor_cache"] = {
            name: _plain(getattr(calculator, name, None))
            for name in (
                "neighbor_cache_build_count",
                "neighbor_cache_reuse_count",
                "neighbor_cache_geometry_update_count",
                "neighbor_cache_host_geometry_materialization_count",
            )
        }
    return record


def factorized_identity(calculator) -> dict[str, Any]:
    evaluator = calculator.evaluator
    fields = (
        "factorized_ready",
        "factorized_jit_ready",
        "factorized_jit_artifact_id",
        "factorized_jit_contract_fingerprint",
        "factorized_selected_direct_forward_executor",
        "factorized_selected_direct_reverse_executor",
        "standard_r0_module_ready",
        "standard_r0_module_fallback_reason",
        "standard_r0_module_id",
        "standard_r0_module_revision",
        "standard_r0_model_contract_fingerprint",
        "standard_r0_selected_executor",
        "standard_m0_module_ready",
        "standard_m0_module_fallback_reason",
        "standard_m0_module_id",
        "standard_m0_module_revision",
        "standard_m0_model_structure_fingerprint",
        "standard_m0_selected_executor",
        "factorized_source_strategy",
        "factorized_execution_strategy",
        "factorized_execution_profile",
        "factorized_reverse_cache_policy",
        "factorized_selected_reverse_cache_policy",
        "factorized_planner_budget_bytes",
        "factorized_planned_coupling_workspace_bytes",
    )
    return {
        "jit": {
            "policy": calculator.jit_policy,
            "status": calculator.jit_status,
            "reason": calculator.jit_reason,
            "artifact_id": calculator.jit_artifact_id,
            "variant_id": getattr(calculator, "jit_variant_id", None),
        },
        "evaluator": {name: _plain(getattr(evaluator, name, None)) for name in fields},
    }


def validate_factorized_identity(identity: dict[str, Any]) -> None:
    jit = identity["jit"]
    evaluator = identity["evaluator"]
    if jit["status"] not in {"built", "cached"}:
        raise RuntimeError(f"Factorized JIT specialization is not active: {jit}")
    if not evaluator["factorized_ready"]:
        raise RuntimeError("factorized evaluator is not ready")
    if not evaluator["factorized_jit_ready"]:
        raise RuntimeError("factorized evaluator JIT module is not ready")
    fallback_fields = (
        "standard_r0_module_fallback_reason",
        "standard_m0_module_fallback_reason",
    )
    nonempty = {name: evaluator[name] for name in fallback_fields if evaluator[name]}
    if nonempty:
        raise RuntimeError(f"factorized fallback was selected: {nonempty}")
    required_identity = (
        "factorized_jit_artifact_id",
        "standard_r0_module_id",
        "standard_m0_module_id",
    )
    missing = [name for name in required_identity if not evaluator[name]]
    if missing:
        raise RuntimeError(f"factorized implementation identity is missing: {missing}")


def make_calculator(
    backend: str,
    *,
    checkpoint: pathlib.Path,
    compact_model: pathlib.Path,
    m1_polynomial_policy: str = "retained",
    m1_recompute_tile_channels: int = 32,
):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    if backend == "torch_cueq":
        import torch
        from mace.calculators.mace import MACECalculator

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        calculator = MACECalculator(
            model_paths=checkpoint,
            device="cuda",
            default_dtype="float32",
            enable_cueq=True,
        )
        calculator.models[0].eval()
        return calculator

    _bootstrap_explicit_symmetrix()
    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native_symmetrix

    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space != "Cuda":
        raise RuntimeError(
            f"Kokkos execution space is {execution_space}, expected Cuda"
        )
    streamed_edges = "all_interactions" if backend == "kokkos_all" else "factorized"
    jit = "off" if backend == "kokkos_all" else "required"
    calculator = Symmetrix(
        compact_model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges=streamed_edges,
        jit=jit,
    )
    if getattr(calculator.evaluator, "scalar_size_bytes", None) != 4:
        raise RuntimeError("Kokkos calculator is not float32")
    if calculator.streamed_edges != streamed_edges:
        raise RuntimeError(
            f"requested streamed_edges={streamed_edges}, got {calculator.streamed_edges}"
        )
    calculator.evaluator._set_m1_recompute_tile_channels(m1_recompute_tile_channels)
    calculator.evaluator._set_m1_polynomial_policy(m1_polynomial_policy)
    return calculator


def runtime_identity(backend: str, calculator) -> dict[str, Any]:
    result: dict[str, Any] = {"backend": backend}
    if backend == "torch_cueq":
        import torch

        result.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(),
                "model_class": type(calculator.models[0]).__name__,
                "cueq_enabled": bool(getattr(calculator, "enable_cueq", True)),
            }
        )
    else:
        from symmetrix import symmetrix as native_symmetrix

        extension = pathlib.Path(native_symmetrix.__file__).resolve()
        result.update(
            {
                "execution_space": native_symmetrix._kokkos_default_execution_space(),
                "native_extension": str(extension),
                "native_extension_sha256": sha256_file(extension),
                "streamed_edges": calculator.streamed_edges,
                "model_cutoff_A": MODEL_CUTOFF_A,
                "neighbor_skin_A": calculator.neighbor_skin,
                "effective_neighbor_cutoff_A": (
                    MODEL_CUTOFF_A + calculator.neighbor_skin
                ),
                "m1_polynomial": {
                    "policy": calculator.evaluator.m1_polynomial_policy,
                    "recompute_tile_channels": (
                        calculator.evaluator.m1_recompute_tile_channels
                    ),
                    "recompute_scratch_bytes": (
                        calculator.evaluator.m1_recompute_scratch_bytes
                    ),
                    "poly_values_active_bytes": (
                        calculator.evaluator.m1_poly_values_active_bytes
                    ),
                    "poly_values_capacity_bytes": (
                        calculator.evaluator.m1_poly_values_capacity_bytes
                    ),
                    "poly_adjoints_active_bytes": (
                        calculator.evaluator.m1_poly_adjoints_active_bytes
                    ),
                    "poly_adjoints_capacity_bytes": (
                        calculator.evaluator.m1_poly_adjoints_capacity_bytes
                    ),
                    "recompute_forward_launch_count": (
                        calculator.evaluator.m1_recompute_forward_launch_count
                    ),
                    "recompute_reverse_launch_count": (
                        calculator.evaluator.m1_recompute_reverse_launch_count
                    ),
                },
            }
        )
        model_load_phases = getattr(calculator.evaluator, "model_load_phase_ms", None)
        if model_load_phases is not None:
            result["model_load_phase_ms"] = {
                str(name): float(milliseconds)
                for name, milliseconds in model_load_phases.items()
            }
        if backend == "factorized":
            result["factorized"] = factorized_identity(calculator)
    return result


def _directed_edges(atoms, cutoff: float = MODEL_CUTOFF_A) -> int | None:
    try:
        from matscipy.neighbours import neighbour_list

        return int(len(neighbour_list("i", atoms, cutoff)))
    except (ImportError, MemoryError):
        return None


def validate_evaluation_count(evaluation_count: int, steps: int) -> None:
    expected = steps + 1
    if evaluation_count != expected:
        raise RuntimeError(
            f"expected exactly {expected} calculator evaluations, "
            f"got {evaluation_count}"
        )


def run_md_worker(
    *,
    backend: str,
    state_path: pathlib.Path,
    checkpoint: pathlib.Path,
    compact_model: pathlib.Path,
    steps: int = DEFAULT_STEPS,
    timestep_fs: float = DEFAULT_TIMESTEP_FS,
    ensemble: str = "npt",
    m1_polynomial_policy: str = "retained",
    m1_recompute_tile_channels: int = 32,
    calculator_factory: Callable[..., Any] = make_calculator,
    telemetry_fn: Callable[..., dict[str, Any]] = capture_telemetry,
    identity_fn: Callable[[str, Any], dict[str, Any]] = runtime_identity,
    heartbeat_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    if steps != DEFAULT_STEPS:
        raise ValueError(f"production contract requires exactly {DEFAULT_STEPS} steps")
    if timestep_fs <= 0.0:
        raise ValueError("timestep_fs must be positive")
    if ensemble not in ENSEMBLES:
        raise ValueError(f"unknown ensemble: {ensemble}")
    state = load_state(state_path)
    atoms = atoms_from_state(state)
    initial_positions = atoms.positions.copy()
    initial_cell = atoms.cell.array.copy()
    initial_volume = float(atoms.get_volume())
    atom_count = len(atoms)
    repeat = int(state["repeat"])
    md = {
        "ensemble": ensemble.upper(),
        "integrator": (
            "VelocityVerlet" if ensemble == "nve" else "ASE NPT (Melchionna)"
        ),
        "steps": steps,
        "timestep_fs": float(timestep_fs),
        "requested_temperature_K": state["requested_temperature_K"],
        "initial_temperature_K": state["actual_temperature_K"],
    }
    if ensemble == "npt":
        md.update(
            {
                "external_pressure_GPa": DEFAULT_EXTERNAL_PRESSURE_GPA,
                "thermostat_time_fs": DEFAULT_THERMOSTAT_TIME_FS,
                "barostat_time_fs": DEFAULT_BAROSTAT_TIME_FS,
                "bulk_modulus_GPa": DEFAULT_BULK_MODULUS_GPA,
            }
        )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "backend": backend,
        "pid": os.getpid(),
        "repeat": repeat,
        "atoms": atom_count,
        "state_path": str(state_path.resolve()),
        "state_sha256": state["state_sha256"],
        "md": md,
        "workload": {
            "model_cutoff_A": MODEL_CUTOFF_A,
            "neighbor_skin_A": (
                SYMMETRIX_NEIGHBOR_SKIN_A
                if backend in {"kokkos_all", "factorized"}
                else None
            ),
            "effective_neighbor_cutoff_A": (
                MODEL_CUTOFF_A + SYMMETRIX_NEIGHBOR_SKIN_A
                if backend in {"kokkos_all", "factorized"}
                else MODEL_CUTOFF_A
            ),
            "initial_directed_edges_at_model_cutoff": _directed_edges(
                atoms, MODEL_CUTOFF_A
            ),
            "initial_directed_edges_at_effective_cutoff": _directed_edges(
                atoms,
                MODEL_CUTOFF_A
                + (
                    SYMMETRIX_NEIGHBOR_SKIN_A
                    if backend in {"kokkos_all", "factorized"}
                    else 0.0
                ),
            ),
        },
        "m1_polynomial_request": {
            "policy": m1_polynomial_policy,
            "recompute_tile_channels": m1_recompute_tile_channels,
        },
        "telemetry": [],
    }
    record["telemetry"].append(
        {"phase": "process_start", **telemetry_fn(backend, None)}
    )
    setup_start = time.perf_counter()
    calculator = calculator_factory(
        backend,
        checkpoint=checkpoint,
        compact_model=compact_model,
        m1_polynomial_policy=m1_polynomial_policy,
        m1_recompute_tile_channels=m1_recompute_tile_channels,
    )
    setup_s = time.perf_counter() - setup_start
    counting = CountingCalculator(calculator)
    atoms.calc = counting.proxy
    record["telemetry"].append(
        {"phase": "calculator_ready", **telemetry_fn(backend, calculator)}
    )

    if backend == "torch_cueq":
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    initial_start = time.perf_counter()
    initial_forces = np.asarray(atoms.get_forces(), dtype=float)
    initial_energy = float(atoms.get_potential_energy())
    initial_force_s = time.perf_counter() - initial_start
    if not np.isfinite(initial_energy) or not np.isfinite(initial_forces).all():
        raise FloatingPointError("initial energy or force is nonfinite")
    identity = identity_fn(backend, calculator)
    if backend == "factorized":
        validate_factorized_identity(identity["factorized"])
    record["telemetry"].append(
        {"phase": "initial_force", **telemetry_fn(backend, calculator)}
    )

    from ase import units

    if ensemble == "nve":
        from ase.md.verlet import VelocityVerlet

        dynamics = VelocityVerlet(atoms, timestep=timestep_fs * units.fs, logfile=None)
    else:
        from ase.md.melchionna import MelchionnaNPT

        bulk_modulus = DEFAULT_BULK_MODULUS_GPA * units.GPa
        dynamics = MelchionnaNPT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=DEFAULT_TEMPERATURE_K,
            externalstress=DEFAULT_EXTERNAL_PRESSURE_GPA * units.GPa,
            ttime=DEFAULT_THERMOSTAT_TIME_FS * units.fs,
            pfactor=(DEFAULT_BAROSTAT_TIME_FS * units.fs) ** 2 * bulk_modulus,
            logfile=None,
        )
    samples_s: list[float] = []
    physical: list[dict[str, Any]] = []
    initial_total_energy = initial_energy + float(atoms.get_kinetic_energy())
    for step in range(1, steps + 1):
        start = time.perf_counter()
        dynamics.run(1)
        elapsed = time.perf_counter() - start
        samples_s.append(elapsed)
        results = atoms.calc.results
        forces = np.asarray(results["forces"], dtype=float)
        potential = float(results["energy"])
        kinetic = float(atoms.get_kinetic_energy())
        total = potential + kinetic
        cell = np.asarray(atoms.cell.array, dtype=float)
        volume = float(atoms.get_volume())
        stress = results.get("stress")
        pressure = (
            -float(np.mean(np.asarray(stress, dtype=float)[:3])) / units.GPa
            if stress is not None
            else None
        )
        gibbs = float(dynamics.get_gibbs_free_energy()) if ensemble == "npt" else None
        values = np.concatenate(
            (
                np.asarray(
                    [
                        potential,
                        kinetic,
                        total,
                        atoms.get_temperature(),
                        volume,
                        gibbs if gibbs is not None else 0.0,
                    ]
                ),
                cell.ravel(),
                forces.ravel(),
                atoms.positions.ravel(),
                atoms.get_momenta().ravel(),
            )
        )
        if not np.isfinite(values).all():
            raise FloatingPointError(f"nonfinite physical value after step {step}")
        physical.append(
            {
                "step": step,
                "potential_energy_eV": potential,
                "kinetic_energy_eV": kinetic,
                "total_energy_eV": total,
                "temperature_K": float(atoms.get_temperature()),
                "pressure_GPa": pressure,
                "volume_A3": volume,
                "relative_volume_change": volume / initial_volume - 1.0,
                "cell_A": cell.tolist(),
                "gibbs_free_energy_eV": gibbs,
                "max_abs_force_eV_per_A": float(np.max(np.abs(forces))),
                "max_displacement_A": float(
                    np.max(np.linalg.norm(atoms.positions - initial_positions, axis=1))
                ),
            }
        )
        record["telemetry"].append(
            {"phase": "step", "step": step, **telemetry_fn(backend, calculator)}
        )
        if heartbeat_path is not None:
            atomic_json(
                heartbeat_path,
                {"backend": backend, "repeat": repeat, "last_step": step},
            )

    validate_evaluation_count(counting.proxy.evaluation_count, steps)
    if ensemble == "nve" and not np.array_equal(atoms.cell.array, initial_cell):
        raise RuntimeError("NVE worker changed the fixed cell")
    if ensemble == "npt" and np.array_equal(atoms.cell.array, initial_cell):
        raise RuntimeError("NPT worker did not update the cell")
    if physical[-1]["max_displacement_A"] <= 0.0:
        raise RuntimeError(f"{ensemble.upper()} worker did not move any atom")

    total_energies = [sample["total_energy_eV"] for sample in physical]
    total_energy_drift_per_atom = (
        total_energies[-1] - initial_total_energy
    ) / atom_count
    if ensemble == "npt":
        gibbs_energies = [sample["gibbs_free_energy_eV"] for sample in physical]
        gibbs_drift_per_atom = (gibbs_energies[-1] - gibbs_energies[0]) / atom_count
    else:
        gibbs_drift_per_atom = None
    final_effective_cutoff = (
        MODEL_CUTOFF_A + SYMMETRIX_NEIGHBOR_SKIN_A
        if backend in {"kokkos_all", "factorized"}
        else MODEL_CUTOFF_A
    )
    record.update(
        {
            "status": "success",
            "directed_edges": _directed_edges(atoms, MODEL_CUTOFF_A),
            "directed_edges_at_effective_cutoff": _directed_edges(
                atoms, final_effective_cutoff
            ),
            "timing": timing_summary(samples_s, atom_count),
            "phase_times_s": {
                "calculator_setup": setup_s,
                "initial_force": initial_force_s,
                "measured_steps": sum(samples_s),
            },
            "calculator_evaluations": counting.proxy.evaluation_count,
            "calculator_evaluations_per_step": (
                counting.proxy.evaluation_count / steps
            ),
            "initial": {
                "energy_eV": initial_energy,
                "forces_sha256": array_sha256(initial_forces),
                "max_abs_force_eV_per_A": float(np.max(np.abs(initial_forces))),
                "forces_eV_per_A": initial_forces.tolist() if repeat == 4 else None,
            },
            "physical_samples": physical,
            "physics": {
                "total_energy_drift_eV_per_atom": total_energy_drift_per_atom,
                "abs_total_energy_drift_eV_per_atom": abs(total_energy_drift_per_atom),
                "drift_threshold_eV_per_atom": 1.0e-3,
                "drift_within_threshold": (abs(total_energy_drift_per_atom) <= 1.0e-3),
                "gibbs_drift_eV_per_atom": gibbs_drift_per_atom,
                "abs_gibbs_drift_eV_per_atom": (
                    abs(gibbs_drift_per_atom)
                    if gibbs_drift_per_atom is not None
                    else None
                ),
                "temperature_min_K": min(x["temperature_K"] for x in physical),
                "temperature_max_K": max(x["temperature_K"] for x in physical),
                "volume_min_A3": min(x["volume_A3"] for x in physical),
                "volume_max_A3": max(x["volume_A3"] for x in physical),
                "final_relative_volume_change": (
                    physical[-1]["relative_volume_change"]
                ),
                "pressure_samples_available": sum(
                    x["pressure_GPa"] is not None for x in physical
                ),
            },
            "runtime": identity,
        }
    )
    record["telemetry"].append({"phase": "final", **telemetry_fn(backend, calculator)})
    return record


def classify_failure_text(text: str) -> str:
    text = text.lower()
    if (
        "out of memory" in text
        or "cuda_error_memory_allocation" in text
        or (
            "failed to allocate" in text
            and ("cuda" in text or "kokkos" in text or "device" in text)
        )
    ):
        return "cuda_oom"
    if "cudaerrorillegaladdress" in text or "illegal memory access" in text:
        return "cuda_illegal_address"
    if "std::vector" in text and "max_size" in text:
        return "native_graph_size_error"
    if "nonfinite" in text:
        return "nonfinite_physics"
    if "factorized" in text or "calculator" in text or "artifact" in text:
        return "setup_rejection"
    return "unknown_exception"


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return "host_safety_limit"
    if isinstance(exc, FloatingPointError):
        return "nonfinite_physics"
    return classify_failure_text(f"{type(exc).__name__}: {exc}")


def worker_entry(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    try:
        checkpoint = verify_artifact(
            args.checkpoint, EXPECTED_CHECKPOINT_SHA256, "OMAT-0 medium checkpoint"
        )
        compact = (
            verify_artifact(
                args.compact_model,
                args.compact_sha256,
                "contract-bearing OMAT-0 compact model",
            )
            if args.compact_sha256
            else verify_compact_model(
                args.compact_model, require_execution_contracts=True
            )
        )
        record = run_md_worker(
            backend=args.backend,
            state_path=args.state,
            checkpoint=pathlib.Path(checkpoint["path"]),
            compact_model=pathlib.Path(compact["path"]),
            steps=args.steps,
            timestep_fs=args.timestep_fs,
            ensemble=args.ensemble,
            m1_polynomial_policy=args.m1_polynomial_policy,
            m1_recompute_tile_channels=args.m1_recompute_tile_channels,
            heartbeat_path=args.heartbeat,
        )
        record["artifacts"] = {"checkpoint": checkpoint, "compact_model": compact}
        atomic_json(output, record)
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failure",
            "backend": args.backend,
            "failure": {
                "class": classify_exception(exc),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        atomic_json(output, failure)
        return 1
    finally:
        finalize_runtime()


def finalize_runtime() -> None:
    gc.collect()
    native = sys.modules.get("symmetrix.symmetrix")
    is_initialized = getattr(native, "_kokkos_is_initialized", None)
    finalize = getattr(native, "_finalize_kokkos", None)
    if callable(is_initialized) and callable(finalize) and is_initialized():
        finalize()


@dataclass(frozen=True)
class Attempt:
    backend: str
    repeat: int
    index: int

    @property
    def stem(self) -> str:
        return f"{self.backend}-n{self.repeat}-a{self.index}"


def apply_attempt_identity(record: dict[str, Any], attempt: Attempt) -> dict[str, Any]:
    for name, expected in (("backend", attempt.backend), ("repeat", attempt.repeat)):
        actual = record.get(name)
        if actual is not None and actual != expected:
            raise RuntimeError(
                f"attempt record {name} mismatch: expected {expected}, got {actual}"
            )
        record[name] = expected
    return record


def confirmed_failure(records: list[dict[str, Any]]) -> str | None:
    classes = [
        record.get("failure", {}).get("class")
        for record in records
        if record.get("status") == "failure"
    ]
    counts = Counter(value for value in classes if value)
    for failure_class, count in counts.items():
        if count >= 2:
            return failure_class
    return None


def refinement_repeat(
    last_success: int | None, first_failure: int | None
) -> int | None:
    if last_success is None or first_failure is None:
        return None
    if first_failure - last_success == 2:
        return last_success + 1
    return None


def campaign_repeats(max_repeat: int) -> list[int]:
    if max_repeat < 4:
        raise ValueError("max_repeat must be at least 4")
    candidates = set(DEFAULT_REPEATS) | set(range(14, max_repeat + 1, 2))
    return sorted(repeat for repeat in candidates if repeat <= max_repeat)


def parse_direct_sizes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--sizes must contain comma-separated integers") from exc
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("--sizes must contain positive repeats")
    if sizes != sorted(set(sizes)):
        raise ValueError("--sizes must be unique and strictly increasing")
    return sizes


def parse_scale_backends(value: str) -> list[str]:
    backends = [item.strip() for item in value.split(",") if item.strip()]
    if not backends:
        raise ValueError("--scale-backends cannot be empty")
    invalid = sorted(set(backends) - set(BACKENDS))
    if invalid:
        raise ValueError(f"unknown --scale-backends values: {invalid}")
    if len(backends) != len(set(backends)):
        raise ValueError("--scale-backends values must be unique")
    return backends


def largest_success_repeat(root: pathlib.Path, backend: str) -> int | None:
    repeats = []
    for path in (root / "attempts").glob(f"{backend}-n*-a*.json"):
        record = load_json(path)
        if record.get("status") == "success":
            repeats.append(int(record["repeat"]))
    return max(repeats, default=None)


def has_successful_repeat(root: pathlib.Path, backend: str, repeat: int) -> bool:
    for path in (root / "attempts").glob(f"{backend}-n{repeat}-a*.json"):
        if load_json(path).get("status") == "success":
            return True
    return False


def classify_process_failure(
    *,
    returncode: int | None,
    timed_out: bool,
    host_guard: bool,
    stderr_text: str = "",
) -> str:
    if host_guard:
        return "host_safety_limit"
    if timed_out:
        return "timeout"
    text_class = classify_failure_text(stderr_text)
    if text_class != "unknown_exception":
        return text_class
    lowered = stderr_text.lower()
    if "__n < this->size()" in stderr_text or "vector<_tp" in lowered:
        return "native_bounds_assertion"
    if returncode is not None and returncode < 0:
        return "signal"
    return "unknown_exception"


def _attempt_paths(root: pathlib.Path, attempt: Attempt) -> dict[str, pathlib.Path]:
    return {
        "record": root / "attempts" / f"{attempt.stem}.json",
        "stdout": root / "logs" / f"{attempt.stem}.stdout",
        "stderr": root / "logs" / f"{attempt.stem}.stderr",
        "heartbeat": root / "heartbeats" / f"{attempt.stem}.json",
    }


def _worker_command(
    *,
    python: pathlib.Path,
    script: pathlib.Path,
    attempt: Attempt,
    state_path: pathlib.Path,
    checkpoint: pathlib.Path,
    compact_model: pathlib.Path,
    compact_sha256: str,
    output_path: pathlib.Path,
    heartbeat_path: pathlib.Path,
    timestep_fs: float,
    ensemble: str,
    m1_polynomial_policy: str = "retained",
    m1_recompute_tile_channels: int = 32,
) -> list[str]:
    return [
        str(python),
        str(script),
        "worker",
        "--backend",
        attempt.backend,
        "--state",
        str(state_path),
        "--checkpoint",
        str(checkpoint),
        "--compact-model",
        str(compact_model),
        "--compact-sha256",
        compact_sha256,
        "--output",
        str(output_path),
        "--heartbeat",
        str(heartbeat_path),
        "--steps",
        str(DEFAULT_STEPS),
        "--timestep-fs",
        str(timestep_fs),
        "--ensemble",
        ensemble,
        "--m1-polynomial-policy",
        m1_polynomial_policy,
        "--m1-recompute-tile-channels",
        str(m1_recompute_tile_channels),
    ]


def read_text_if_exists(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def run_subprocess_attempt(
    *,
    root: pathlib.Path,
    attempt: Attempt,
    state_path: pathlib.Path,
    checkpoint: pathlib.Path,
    compact_model: pathlib.Path,
    compact_sha256: str,
    timestep_fs: float,
    ensemble: str,
    timeout_s: float,
    host_reserve_mib: float,
    python: pathlib.Path,
    script: pathlib.Path,
    m1_polynomial_policy: str = "retained",
    m1_recompute_tile_channels: int = 32,
) -> dict[str, Any]:
    paths = _attempt_paths(root, attempt)
    if paths["record"].is_file():
        existing = apply_attempt_identity(load_json(paths["record"]), attempt)
        if existing.get("status") in {"success", "failure"}:
            historical_default = {
                "policy": "retained",
                "recompute_tile_channels": 32,
            }
            requested = {
                "policy": m1_polynomial_policy,
                "recompute_tile_channels": m1_recompute_tile_channels,
            }
            if (
                existing.get("m1_polynomial_request", historical_default) == requested
                and existing.get("md", {}).get("ensemble", "NVE").lower() == ensemble
            ):
                atomic_json(paths["record"], existing)
                return existing
            raise RuntimeError(
                "attempt record uses a different ensemble or M1 polynomial policy; "
                "select a separate output directory"
            )
    for key in ("record", "stdout", "stderr", "heartbeat"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    command = _worker_command(
        python=python,
        script=script,
        attempt=attempt,
        state_path=state_path,
        checkpoint=checkpoint,
        compact_model=compact_model,
        compact_sha256=compact_sha256,
        output_path=paths["record"],
        heartbeat_path=paths["heartbeat"],
        timestep_fs=timestep_fs,
        ensemble=ensemble,
        m1_polynomial_policy=m1_polynomial_policy,
        m1_recompute_tile_channels=m1_recompute_tile_channels,
    )
    started = time.monotonic()
    timed_out = False
    host_guard = False
    with (
        paths["stdout"].open("w", encoding="utf-8") as stdout,
        paths["stderr"].open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, env=os.environ.copy()
        )
        worker_pid = process.pid
        while process.poll() is None:
            elapsed = time.monotonic() - started
            available = _read_system_memory()["available_mib"]
            if available is not None and available < host_reserve_mib:
                host_guard = True
                process.terminate()
                break
            if elapsed > timeout_s:
                timed_out = True
                process.terminate()
                break
            time.sleep(0.25)
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        returncode = process.wait()
    elapsed = time.monotonic() - started
    stderr_text = read_text_if_exists(paths["stderr"])
    if paths["record"].is_file():
        record = apply_attempt_identity(load_json(paths["record"]), attempt)
    else:
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "failure",
            "backend": attempt.backend,
            "repeat": attempt.repeat,
            "failure": {
                "class": classify_process_failure(
                    returncode=returncode,
                    timed_out=timed_out,
                    host_guard=host_guard,
                    stderr_text=stderr_text,
                ),
                "message": "worker exited without a complete record",
                "stderr_tail": stderr_text[-4000:],
            },
        }
        apply_attempt_identity(record, attempt)
    heartbeat = load_json(paths["heartbeat"]) if paths["heartbeat"].is_file() else None
    record["attempt"] = {
        "index": attempt.index,
        "command": command,
        "returncode": returncode,
        "elapsed_s": elapsed,
        "stdout": str(paths["stdout"]),
        "stderr": str(paths["stderr"]),
        "last_heartbeat": heartbeat,
        "pid": worker_pid,
    }
    atomic_json(paths["record"], record)
    return record


def ensure_state(root: pathlib.Path, repeat: int, seed: int) -> pathlib.Path:
    path = root / "states" / f"aln-n{repeat}-seed{seed}.npz"
    if path.is_file():
        state = load_state(path)
        if state["repeat"] != repeat or state["seed"] != seed:
            raise RuntimeError(f"state metadata mismatch: {path}")
        return path
    save_state(path, build_initial_state(repeat, seed=seed))
    return path


def check_initial_parity(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(records) != set(BACKENDS):
        raise RuntimeError("initial parity requires all three backends")
    reference = records["torch_cueq"]
    ref_forces = np.asarray(reference["initial"]["forces_eV_per_A"], dtype=float)
    atom_count = int(reference["atoms"])
    comparisons: dict[str, Any] = {}
    passed = True
    for backend, record in records.items():
        forces = np.asarray(record["initial"]["forces_eV_per_A"], dtype=float)
        energy_error = (
            abs(
                float(record["initial"]["energy_eV"])
                - float(reference["initial"]["energy_eV"])
            )
            / atom_count
        )
        force_error = float(np.max(np.abs(forces - ref_forces)))
        backend_passed = energy_error <= 2.0e-4 and force_error <= 5.0e-4
        comparisons[backend] = {
            "energy_error_eV_per_atom": energy_error,
            "max_force_component_error_eV_per_A": force_error,
            "passed": backend_passed,
        }
        passed &= backend_passed
    return {
        "reference_backend": "torch_cueq",
        "energy_tolerance_eV_per_atom": 2.0e-4,
        "force_tolerance_eV_per_A": 5.0e-4,
        "comparisons": comparisons,
        "passed": passed,
    }


def assemble_report(
    root: pathlib.Path,
    *,
    campaign: dict[str, Any],
    boundaries: dict[str, Any],
    parity: dict[str, Any] | None,
) -> dict[str, Any]:
    attempts = []
    for path in sorted((root / "attempts").glob("*.json")):
        record = load_json(path)
        if record.get("status") == "failure":
            failure = record.get("failure", {})
            evidence = "\n".join(
                str(value)
                for value in (
                    failure.get("type"),
                    failure.get("message"),
                    failure.get("traceback"),
                    read_text_if_exists(
                        pathlib.Path(record.get("attempt", {}).get("stderr", ""))
                    )
                    if record.get("attempt", {}).get("stderr")
                    else "",
                )
                if value
            )
            normalized = classify_failure_text(evidence)
            original = failure.get("class")
            if normalized != "unknown_exception" and normalized != original:
                failure["original_class"] = original
                failure["class"] = normalized
        attempts.append(record)

    boundaries = copy.deepcopy(boundaries)
    for backend, boundary in boundaries.items():
        first_failure = boundary.get("first_failure_repeat")
        if first_failure is None:
            continue
        records = [
            record
            for record in attempts
            if record.get("backend") == backend
            and record.get("repeat") == first_failure
        ]
        normalized = confirmed_failure(records)
        if normalized is not None:
            boundary["failure_class"] = normalized
    successful = [record for record in attempts if record.get("status") == "success"]
    backend_timings = [
        {
            "backend": record["backend"],
            "repeat": int(record["repeat"]),
            "atoms": int(record["atoms"]),
            "median_ms": float(record["timing"]["median_ms"]),
            "median_us_per_atom": float(
                record["timing"].get(
                    "median_us_per_atom",
                    1000.0 * record["timing"]["median_ms"] / record["atoms"],
                )
            ),
        }
        for record in successful
    ]
    speed_rows = []
    by_repeat: dict[int, dict[str, dict[str, Any]]] = {}
    for record in successful:
        by_repeat.setdefault(int(record["repeat"]), {})[record["backend"]] = record
    for repeat, records in sorted(by_repeat.items()):
        if set(records) != set(BACKENDS):
            continue
        medians = {
            backend: records[backend]["timing"]["median_ms"] for backend in BACKENDS
        }
        speed_rows.append(
            {
                "repeat": repeat,
                "atoms": records["kokkos_all"]["atoms"],
                "state_sha256": records["kokkos_all"]["state_sha256"],
                "median_ms": medians,
                "speedup": {
                    "factorized_vs_all": medians["kokkos_all"] / medians["factorized"],
                    "factorized_vs_torch_cueq": medians["torch_cueq"]
                    / medians["factorized"],
                    "all_vs_torch_cueq": medians["torch_cueq"] / medians["kokkos_all"],
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": campaign,
        "parity": parity,
        "boundaries": boundaries,
        "backend_timings": sorted(
            backend_timings, key=lambda row: (row["repeat"], row["backend"])
        ),
        "speed_comparison": speed_rows,
        "attempts": attempts,
    }


def write_csv_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "repeat",
                "atoms",
                "kokkos_all_median_us_per_atom",
                "factorized_median_us_per_atom",
                "torch_cueq_median_us_per_atom",
                "factorized_vs_all",
                "factorized_vs_torch_cueq",
            ]
        )
        for row in report["speed_comparison"]:
            writer.writerow(
                [
                    row["repeat"],
                    row["atoms"],
                    1000.0 * row["median_ms"]["kokkos_all"] / row["atoms"],
                    1000.0 * row["median_ms"]["factorized"] / row["atoms"],
                    1000.0 * row["median_ms"]["torch_cueq"] / row["atoms"],
                    row["speedup"]["factorized_vs_all"],
                    row["speedup"]["factorized_vs_torch_cueq"],
                ]
            )


def write_markdown_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    ensemble = str(report.get("campaign", {}).get("ensemble", "NPT")).upper()
    if ensemble == "NVE":
        timing_description = (
            "End-to-end ASE Velocity-Verlet NVE timings. Each record contains "
            "exactly 20 1 fs steps."
        )
    else:
        timing_description = (
            "End-to-end ASE Melchionna NPT timings. Each record contains exactly "
            "20 1 fs steps at 300 K and 0 GPa."
        )
    lines = [
        "# OMAT-0 Medium ASE MD Backend Speed and Scale",
        "",
        timing_description,
        "",
        "## Backend Timings",
        "",
        "| Backend | Repeat | Atoms | Median (us/atom) |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("backend_timings", ()):
        lines.append(
            f"| {row['backend']} | {row['repeat']} | {row['atoms']} | "
            f"{row['median_us_per_atom']:.3f} |"
        )
    if report["speed_comparison"]:
        lines.extend(
            [
                "",
                "## Speed",
                "",
                "| Repeat | Atoms | Kokkos all (us/atom) | Factorized (us/atom) | Torch/cuEq (us/atom) | Factorized/all | Factorized/cuEq |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["speed_comparison"]:
            lines.append(
                "| {repeat} | {atoms} | {all:.3f} | {factorized:.3f} | "
                "{torch:.3f} | {vs_all:.3f}x | {vs_torch:.3f}x |".format(
                    repeat=row["repeat"],
                    atoms=row["atoms"],
                    all=1000.0 * row["median_ms"]["kokkos_all"] / row["atoms"],
                    factorized=(1000.0 * row["median_ms"]["factorized"] / row["atoms"]),
                    torch=1000.0 * row["median_ms"]["torch_cueq"] / row["atoms"],
                    vs_all=row["speedup"]["factorized_vs_all"],
                    vs_torch=row["speedup"]["factorized_vs_torch_cueq"],
                )
            )
    lines.extend(
        [
            "",
            "## Scale Boundary",
            "",
            "| Backend | Largest success | First confirmed failure | Reason |",
            "|---|---:|---:|---|",
        ]
    )
    for backend in report.get("campaign", {}).get("backends", BACKENDS):
        boundary = report["boundaries"].get(backend, {})
        lines.append(
            f"| {backend} | {boundary.get('largest_success_repeat', '-')} | "
            f"{boundary.get('first_failure_repeat', '-')} | "
            f"{boundary.get('failure_class', boundary.get('status', '-'))} |"
        )
    parity = report.get("parity")
    lines.extend(["", "## Qualification", ""])
    lines.append(
        f"Initial 256-atom parity passed: `{parity['passed']}`."
        if parity is not None
        else "Initial parity: `not requested for a single-backend campaign`."
    )
    lines.append("")
    lines.append(
        "Raw timings, telemetry, model identities, and failures are retained in JSON."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_reports(root: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_json(root / "omat0_medium_ase_md_backend_scale.json", report)
    write_csv_report(root / "omat0_medium_ase_md_backend_scale.csv", report)
    write_markdown_report(root / "omat0_medium_ase_md_backend_scale.md", report)


def campaign_entry(args: argparse.Namespace) -> int:
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = verify_artifact(
        args.checkpoint, EXPECTED_CHECKPOINT_SHA256, "OMAT-0 medium checkpoint"
    )
    compact = verify_compact_model(args.compact_model, require_execution_contracts=True)
    script = pathlib.Path(__file__).resolve()
    python = current_python_executable()
    direct_sizes = parse_direct_sizes(args.sizes)
    active_backends = parse_scale_backends(args.backends)
    scale_backends = parse_scale_backends(args.scale_backends)
    invalid_scale_backends = sorted(set(scale_backends) - set(active_backends))
    if invalid_scale_backends:
        raise ValueError(
            f"--scale-backends must be a subset of --backends: {invalid_scale_backends}"
        )
    requested = direct_sizes or campaign_repeats(args.max_repeat)
    campaign = {
        "id": args.campaign_id,
        "started_unix_s": time.time(),
        "checkpoint": checkpoint,
        "compact_model": compact,
        "steps": DEFAULT_STEPS,
        "timestep_fs": args.timestep_fs,
        "ensemble": args.ensemble.upper(),
        "temperature_K": DEFAULT_TEMPERATURE_K,
        "model_cutoff_A": MODEL_CUTOFF_A,
        "symmetrix_neighbor_skin_A": SYMMETRIX_NEIGHBOR_SKIN_A,
        "symmetrix_effective_neighbor_cutoff_A": (
            MODEL_CUTOFF_A + SYMMETRIX_NEIGHBOR_SKIN_A
        ),
        "seed": args.seed,
        "python": str(python),
        "script": str(script),
        "backends": active_backends,
        "scale_backends": scale_backends,
        "requested_repeats": requested,
        "max_repeat": max(requested),
        "timeout_s": args.timeout_s,
        "host_reserve_mib": args.host_reserve_mib,
        "m1_polynomial": {
            "policy": args.m1_polynomial_policy,
            "recompute_tile_channels": args.m1_recompute_tile_channels,
        },
    }
    if args.ensemble == "npt":
        campaign.update(
            {
                "external_pressure_GPa": DEFAULT_EXTERNAL_PRESSURE_GPA,
                "thermostat_time_fs": DEFAULT_THERMOSTAT_TIME_FS,
                "barostat_time_fs": DEFAULT_BAROSTAT_TIME_FS,
                "bulk_modulus_GPa": DEFAULT_BULK_MODULUS_GPA,
            }
        )
    combined_path = root / "omat0_medium_ase_md_backend_scale.json"
    previous = load_json(combined_path) if combined_path.is_file() else {}
    boundaries: dict[str, Any] = dict(previous.get("boundaries") or {})

    def run_attempt(backend: str, repeat: int, index: int) -> dict[str, Any]:
        state = ensure_state(root, repeat, args.seed)
        record = run_subprocess_attempt(
            root=root,
            attempt=Attempt(backend, repeat, index),
            state_path=state,
            checkpoint=pathlib.Path(checkpoint["path"]),
            compact_model=pathlib.Path(compact["path"]),
            compact_sha256=compact["sha256"],
            timestep_fs=args.timestep_fs,
            ensemble=args.ensemble,
            timeout_s=args.timeout_s,
            host_reserve_mib=args.host_reserve_mib,
            python=python,
            script=script,
            m1_polynomial_policy=args.m1_polynomial_policy,
            m1_recompute_tile_channels=args.m1_recompute_tile_channels,
        )
        report = assemble_report(
            root, campaign=campaign, boundaries=boundaries, parity=None
        )
        publish_reports(root, report)
        return record

    smoke: dict[str, dict[str, Any]] = {}
    for backend in active_backends:
        record = run_attempt(backend, 4, 1)
        if record.get("status") != "success":
            boundaries[backend] = {
                "status": "smoke_failed",
                "first_failure_repeat": 4,
                "failure_class": record.get("failure", {}).get("class"),
            }
            report = assemble_report(
                root, campaign=campaign, boundaries=boundaries, parity=None
            )
            publish_reports(root, report)
            return 2
        smoke[backend] = record
    parity = (
        check_initial_parity(smoke) if set(active_backends) == set(BACKENDS) else None
    )
    if parity is not None and not parity["passed"]:
        report = assemble_report(
            root, campaign=campaign, boundaries=boundaries, parity=parity
        )
        publish_reports(root, report)
        return 3

    for backend in scale_backends:
        previous_boundary = copy.deepcopy(boundaries.get(backend))
        boundaries.pop(backend, None)
        last_success = largest_success_repeat(root, backend) or 4
        initial_last_success = last_success
        first_failure = None
        failure_class = None
        for repeat in requested:
            if direct_sizes is not None:
                if has_successful_repeat(root, backend, repeat):
                    continue
            elif repeat <= last_success:
                continue
            records = [run_attempt(backend, repeat, 1)]
            if records[0].get("status") == "success":
                last_success = max(last_success, repeat)
                continue
            for index in range(2, args.max_attempts + 1):
                records.append(run_attempt(backend, repeat, index))
                failure_class = confirmed_failure(records)
                if failure_class is not None:
                    break
            if failure_class is None:
                boundaries[backend] = {
                    "status": "unstable_failure",
                    "largest_success_repeat": last_success,
                    "first_failure_repeat": repeat,
                }
                break
            first_failure = repeat
            odd = refinement_repeat(last_success, first_failure)
            if odd is not None:
                odd_records = [run_attempt(backend, odd, 1)]
                if odd_records[0].get("status") == "success":
                    last_success = odd
                else:
                    odd_class = None
                    for index in range(2, args.max_attempts + 1):
                        odd_records.append(run_attempt(backend, odd, index))
                        odd_class = confirmed_failure(odd_records)
                        if odd_class is not None:
                            break
                    if odd_class is not None:
                        first_failure = odd
                        failure_class = odd_class
                    else:
                        boundaries[backend] = {
                            "status": "unstable_refinement",
                            "largest_success_repeat": last_success,
                            "first_failure_repeat": odd,
                        }
                        break
            boundaries.setdefault(
                backend,
                {
                    "status": "bounded",
                    "largest_success_repeat": last_success,
                    "first_failure_repeat": first_failure,
                    "failure_class": failure_class,
                },
            )
            break
        else:
            if (
                direct_sizes is not None
                and previous_boundary is not None
                and max(requested) <= initial_last_success
            ):
                boundaries[backend] = previous_boundary
            else:
                boundaries[backend] = {
                    "status": "cap_limited_lower_bound",
                    "largest_success_repeat": last_success,
                    "first_failure_repeat": None,
                }

    report = assemble_report(
        root, campaign=campaign, boundaries=boundaries, parity=parity
    )
    publish_reports(root, report)
    return 0


def report_entry(args: argparse.Namespace) -> int:
    root = args.output_dir.resolve()
    existing = load_json(root / "omat0_medium_ase_md_backend_scale.json")
    report = assemble_report(
        root,
        campaign=existing["campaign"],
        boundaries=existing.get("boundaries", {}),
        parity=existing.get("parity"),
    )
    publish_reports(root, report)
    return 0


def extract_entry(args: argparse.Namespace) -> int:
    result = derive_compact_model(args.checkpoint, args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def package_versions() -> dict[str, str | None]:
    versions = {"python": sys.version.split()[0]}
    for package in (
        "ase",
        "numpy",
        "matscipy",
        "torch",
        "mace-torch",
        "cuequivariance",
        "cuequivariance-torch",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def current_python_executable() -> pathlib.Path:
    """Keep a virtual-environment launcher instead of resolving its base Python."""

    return pathlib.Path(os.path.abspath(sys.executable))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="derive a contract-bearing compact model from the checkpoint"
    )
    extract.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    extract.add_argument("--output", type=pathlib.Path, required=True)
    extract.set_defaults(handler=extract_entry)

    worker = subparsers.add_parser("worker", help="run one fresh-process MD worker")
    worker.add_argument("--backend", choices=BACKENDS, required=True)
    worker.add_argument("--state", type=pathlib.Path, required=True)
    worker.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    worker.add_argument(
        "--compact-model", type=pathlib.Path, default=DEFAULT_COMPACT_MODEL
    )
    worker.add_argument(
        "--compact-sha256",
        help="supervisor-validated compact artifact hash; avoids reparsing JSON",
    )
    worker.add_argument("--output", type=pathlib.Path, required=True)
    worker.add_argument("--heartbeat", type=pathlib.Path)
    worker.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    worker.add_argument("--timestep-fs", type=float, default=DEFAULT_TIMESTEP_FS)
    worker.add_argument("--ensemble", choices=ENSEMBLES, default="npt")
    worker.add_argument(
        "--m1-polynomial-policy",
        choices=("retained", "recompute"),
        default="retained",
    )
    worker.add_argument(
        "--m1-recompute-tile-channels",
        type=int,
        choices=(8, 16, 32),
        default=32,
    )
    worker.set_defaults(handler=worker_entry)

    campaign = subparsers.add_parser(
        "campaign", help="run the restartable scale campaign"
    )
    campaign.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    campaign.add_argument(
        "--compact-model", type=pathlib.Path, default=DEFAULT_COMPACT_MODEL
    )
    campaign.add_argument("--output-dir", type=pathlib.Path, required=True)
    campaign.add_argument("--campaign-id", default="omat0-medium-aln-ase-md")
    campaign.add_argument("--seed", type=int, default=DEFAULT_SEED)
    campaign.add_argument("--timestep-fs", type=float, default=DEFAULT_TIMESTEP_FS)
    campaign.add_argument("--ensemble", choices=ENSEMBLES, default="npt")
    campaign.add_argument("--max-repeat", type=int, default=32)
    campaign.add_argument(
        "--sizes",
        help="explicit strictly increasing cubic repeats; bypasses the default ladder",
    )
    campaign.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help="comma-separated backends to run, including the repeat-4 smoke",
    )
    campaign.add_argument(
        "--scale-backends",
        default=",".join(BACKENDS),
        help="comma-separated backends to extend after the three-backend parity smoke",
    )
    campaign.add_argument("--max-attempts", type=int, default=3)
    campaign.add_argument("--timeout-s", type=float, default=1800.0)
    campaign.add_argument("--host-reserve-mib", type=float, default=8192.0)
    campaign.add_argument(
        "--m1-polynomial-policy",
        choices=("retained", "recompute"),
        default="retained",
    )
    campaign.add_argument(
        "--m1-recompute-tile-channels",
        type=int,
        choices=(8, 16, 32),
        default=32,
    )
    campaign.set_defaults(handler=campaign_entry)

    report = subparsers.add_parser("report", help="regenerate CSV and Markdown")
    report.add_argument("--output-dir", type=pathlib.Path, required=True)
    report.set_defaults(handler=report_entry)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "sizes"):
            parse_direct_sizes(args.sizes)
            active_backends = parse_scale_backends(args.backends)
            scale_backends = parse_scale_backends(args.scale_backends)
            if not set(scale_backends) <= set(active_backends):
                raise ValueError("--scale-backends must be a subset of --backends")
    except ValueError as exc:
        parser.error(str(exc))
    if getattr(args, "steps", DEFAULT_STEPS) != DEFAULT_STEPS:
        parser.error(f"--steps must be exactly {DEFAULT_STEPS}")
    if getattr(args, "timestep_fs", DEFAULT_TIMESTEP_FS) <= 0.0:
        parser.error("--timestep-fs must be positive")
    if hasattr(args, "max_repeat") and args.max_repeat < 4:
        parser.error("--max-repeat must be at least 4")
    if getattr(args, "max_attempts", 2) < 2:
        parser.error("--max-attempts must be at least 2")
    if getattr(args, "timeout_s", 1.0) <= 0.0:
        parser.error("--timeout-s must be positive")
    if getattr(args, "host_reserve_mib", 1.0) <= 0.0:
        parser.error("--host-reserve-mib must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
