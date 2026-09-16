"""Run a small, representative Symmetrix inference benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path


_DEFAULT_MODEL = "mace-omat-0-medium"
_MODEL_ALIASES = {
    "mace-mpa-0-medium": "medium-mpa-0",
    "mace-omat-0-medium": "medium-omat-0",
}
_EXTRACTION_CACHE_VERSION = 1
_CORRECTNESS_TOLERANCES = {
    "energy_abs_eV_per_atom": 1.0e-4,
    "forces_max_abs_eV_per_A": 2.0e-3,
    "stress_max_abs_eV_per_A3": 3.0e-3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(stage: str) -> None:
    print(f"[symmetrix bench] {stage}", file=sys.stderr, flush=True)


def _finalize_loaded_backend(native=None) -> None:
    if native is None:
        from .. import backend_loader

        native = backend_loader._native_module
    gc.collect()
    if native is None:
        return
    is_initialized = getattr(native, "_kokkos_is_initialized", lambda: False)
    if is_initialized():
        native._finalize_kokkos()


def _directed_edge_count(calculator, atoms) -> int:
    """Return graph cardinality without materializing device-resident edges."""
    cache = getattr(calculator, "_neighbor_cache", None)
    if cache is not None and getattr(cache, "device_graph_generation", 0) != 0:
        return int(cache.num_edges)
    return len(calculator._mace_inputs(atoms)[3])


def _model_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    resolved = _cached_model_resolution(value)
    if resolved is not None:
        return resolved
    try:
        # MACE prints library notices to stdout at import; keep --json valid.
        from contextlib import redirect_stdout

        with redirect_stdout(sys.stderr):
            from mace.calculators.foundations_models import (
                download_mace_mp_checkpoint,
            )

            try:
                from mace.calculators.foundations_models import mace_mp_names
            except ImportError:
                # mace-torch 0.3.10 exposes no name registry; let the
                # downloader validate the request instead of failing here.
                mace_mp_names = None
    except ImportError as error:
        raise RuntimeError(
            "downloading a named MACE model requires mace-torch; install it "
            "with `uv pip install mace-torch` or pass a local model path"
        ) from error
    downloader_name = _MODEL_ALIASES.get(value, value)
    if mace_mp_names is not None and downloader_name not in mace_mp_names:
        names = sorted(name for name in mace_mp_names if name is not None)
        raise FileNotFoundError(
            f"model file does not exist and {value!r} is not a recognized MACE "
            f"model name; available names: {', '.join(names)}"
        )
    try:
        # MACE's downloader writes progress to stdout. Keep --json output valid.
        from contextlib import redirect_stdout

        with redirect_stdout(sys.stderr):
            resolved = Path(download_mace_mp_checkpoint(downloader_name)).resolve()
        _record_model_resolution(value, resolved)
        return resolved
    except Exception as error:
        raise RuntimeError(
            f"could not download MACE model {value!r}; pass --model with a local "
            "checkpoint"
        ) from error


def _extraction_cache_root() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "symmetrix" / "bench-models"


def _model_resolution_cache_path() -> Path:
    return _extraction_cache_root() / "resolved-models.json"


def _cached_model_resolution(name: str) -> Path | None:
    try:
        with _model_resolution_cache_path().open(encoding="utf-8") as stream:
            resolved = Path(json.load(stream)[name]).expanduser()
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return resolved.resolve() if resolved.is_file() else None


def _record_model_resolution(name: str, model: Path) -> None:
    cache_path = _model_resolution_cache_path()
    try:
        with cache_path.open(encoding="utf-8") as stream:
            resolutions = json.load(stream)
        if not isinstance(resolutions, dict):
            resolutions = {}
    except (OSError, ValueError):
        resolutions = {}
    resolutions[name] = str(model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=".resolved-models.",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(resolutions, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, cache_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _prepare_benchmark_model(model: Path, species: list[int]) -> tuple[Path, bool]:
    """Return a native JSON model, reusing an atomic extraction cache."""
    if model.suffix.lower() == ".json":
        return model, True
    options = {
        "cache_version": _EXTRACTION_CACHE_VERSION,
        "checkpoint_sha256": _sha256(model),
        "extractor_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "extract_mace_data.py"
        ),
        "species": sorted(set(int(value) for value in species)),
        "num_spline_points": 256,
        "radial_format": "compact",
    }
    key = hashlib.sha256(
        json.dumps(options, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_root = _extraction_cache_root()
    cached = cache_root / f"{key}.json"
    try:
        if cached.is_file() and cached.stat().st_size > 0:
            return cached, True
    except OSError:
        pass

    # MACE prints library notices to stdout at import; keep --json valid.
    from contextlib import redirect_stdout

    with redirect_stdout(sys.stderr):
        from ..extract_mace_data import extract_mace_data

    data = extract_mace_data(
        model,
        species=options["species"],
        num_spline_points=options["num_spline_points"],
        radial_format=options["radial_format"],
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=cache_root, prefix=f".{key}.", delete=False
        ) as stream:
            temporary_name = stream.name
            json.dump(data, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, cached)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return cached, False


def _structure(repeat: int, rattle: float, seed: int):
    import numpy as np
    from ase.spacegroup import crystal

    if repeat < 1:
        raise ValueError("--supercell-repeat must be positive")
    if rattle < 0.0:
        raise ValueError("--rattle must be non-negative")
    atoms = crystal(
        symbols=["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    )
    atoms *= (repeat, repeat, repeat)
    if rattle:
        atoms.positions += np.random.default_rng(seed).normal(
            scale=rattle, size=atoms.positions.shape
        )
    return atoms


def _max_abs_difference(left, right) -> float:
    import numpy as np

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise RuntimeError(
            "MACE-Torch correctness result shape mismatch: "
            f"{left_array.shape} versus {right_array.shape}"
        )
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _validate_against_mace_torch(model, atoms, symmetrix_results, dtype, backend):
    """Run an untimed MACE-Torch parity check when a checkpoint is available."""
    if model.suffix.lower() == ".json":
        return {
            "status": "skipped",
            "reference": "mace-torch",
            "reason": "an extracted JSON model has no MACE-Torch checkpoint",
        }
    if any(importlib.util.find_spec(module) is None for module in ("mace", "torch")):
        return {
            "status": "skipped",
            "reference": "mace-torch",
            "reason": "mace-torch is not installed",
        }

    import numpy as np
    from ase.calculators.calculator import all_changes
    from contextlib import redirect_stdout

    device = "cuda" if backend.get("backend") in {"cuda", "hip"} else "cpu"
    reference_atoms = atoms.copy()
    # MACE prints library notices to stdout; keep --json output valid.
    with redirect_stdout(sys.stderr):
        from mace.calculators import MACECalculator

        reference = MACECalculator(
            model_paths=str(model),
            device=device,
            default_dtype=dtype,
        )
        reference.calculate(
            reference_atoms,
            properties=["energy", "forces", "stress"],
            system_changes=all_changes,
        )
    reference_results = reference.results
    errors = {
        "energy_abs_eV_per_atom": abs(
            float(symmetrix_results["energy"]) - float(reference_results["energy"])
        )
        / len(atoms),
        "forces_max_abs_eV_per_A": _max_abs_difference(
            symmetrix_results["forces"], reference_results["forces"]
        ),
        "stress_max_abs_eV_per_A3": _max_abs_difference(
            symmetrix_results["stress"], reference_results["stress"]
        ),
    }
    finite = all(np.isfinite(value) for value in errors.values())
    failures = [
        name
        for name, value in errors.items()
        if not np.isfinite(value) or value > _CORRECTNESS_TOLERANCES[name]
    ]
    result = {
        "status": "passed" if finite and not failures else "failed",
        "reference": "mace-torch",
        "device": device,
        "errors": errors,
        "tolerances": dict(_CORRECTNESS_TOLERANCES),
        "failed_metrics": failures,
    }
    if failures:
        details = ", ".join(
            f"{name}={errors[name]:.6g} > {_CORRECTNESS_TOLERANCES[name]:.6g}"
            for name in failures
        )
        raise RuntimeError(f"MACE-Torch correctness validation failed: {details}")
    return result


def _is_validation_resource_error(error: BaseException) -> bool:
    """Return whether a reference check failed because resources were exhausted."""
    if isinstance(error, MemoryError):
        return True
    message = str(error).lower()
    return any(
        phrase in message
        for phrase in (
            "out of memory",
            "cuda error: out of memory",
            "hip error: out of memory",
            "memory allocation",
        )
    )


def _run_validation(
    model,
    atoms,
    symmetrix_results,
    dtype,
    backend,
    *,
    skip: bool,
):
    if skip:
        return {
            "status": "skipped",
            "reference": "mace-torch",
            "reason": "disabled by --skip-validation",
        }
    try:
        return _validate_against_mace_torch(
            model, atoms, symmetrix_results, dtype, backend
        )
    except (MemoryError, RuntimeError) as error:
        if not _is_validation_resource_error(error):
            raise
        return {
            "status": "error",
            "reference": "mace-torch",
            "reason": f"reference validation could not allocate enough memory: {error}",
            "exception_type": type(error).__name__,
        }


def run(args: argparse.Namespace) -> dict:
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("--warmups must be non-negative and --repeats positive")
    if args.backend is not None:
        if args.backend == "auto":
            os.environ.pop("SYMMETRIX_BACKEND", None)
        else:
            os.environ["SYMMETRIX_BACKEND"] = args.backend
    for variable in ("KOKKOS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    _progress(f"[1/7] Resolving model: {args.model}")
    model = _model_path(args.model)
    if model.suffix.lower() != ".json" and any(
        importlib.util.find_spec(module) is None for module in ("mace", "torch")
    ):
        raise RuntimeError(
            "loading a MACE checkpoint requires mace-torch; install it with "
            "`uv pip install mace-torch` or pass an extracted JSON model"
        )
    from .. import load_backend, selected_backend

    _progress(f"[2/7] Loading backend: {args.backend or 'automatic'}")
    native = load_backend(args.backend)
    from .. import Symmetrix

    backend = selected_backend() or {}
    _progress(f"Backend ready: {backend.get('selector', 'unknown')}")
    _progress(
        f"[3/7] Building rattled {args.supercell_repeat}x{args.supercell_repeat}x"
        f"{args.supercell_repeat} SrTiO3 structure"
    )
    atoms = _structure(args.supercell_repeat, args.rattle, args.seed)
    species = sorted(set(int(value) for value in atoms.get_atomic_numbers()))
    model_action = (
        "Loading extracted model"
        if model.suffix.lower() == ".json"
        else "Preparing cached model extraction"
    )
    _progress(f"[4/7] {model_action} and preparing direct execution")
    setup_start = time.perf_counter()
    runtime_model, extraction_cache_hit = _prepare_benchmark_model(model, species)
    if model.suffix.lower() != ".json":
        state = "reused" if extraction_cache_hit else "created"
        _progress(f"Extraction cache {state}: {runtime_model}")
    try:
        calculator = Symmetrix(
            runtime_model,
            dtype=args.dtype,
            use_kokkos=True,
            streamed_edges="direct",
            execution_profile=args.profile,
            neighbor_skin=args.neighbor_skin,
        )
    except ModuleNotFoundError as error:
        if model.suffix.lower() != ".json":
            raise RuntimeError(
                "loading a MACE checkpoint requires mace-torch; install it with "
                "`uv pip install mace-torch` or pass an extracted JSON model"
            ) from error
        raise
    model_setup_seconds = time.perf_counter() - setup_start
    _progress(f"Model and direct execution ready in {model_setup_seconds:.2f} s")
    atoms.calc = calculator
    properties = ["energy", "forces", "stress"]
    warmup_label = "evaluation" if args.warmups == 1 else "evaluations"
    _progress(f"[5/7] Running {args.warmups} warmup {warmup_label}")
    warmup_start = time.perf_counter()
    for _ in range(args.warmups):
        calculator.calculate(atoms, properties=properties, system_changes=[])
    warmup_seconds = time.perf_counter() - warmup_start
    sample_label = "evaluation" if args.repeats == 1 else "evaluations"
    _progress(f"[6/7] Measuring {args.repeats} {sample_label}")
    samples_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        calculator.calculate(atoms, properties=properties, system_changes=[])
        samples_ms.append(1000.0 * (time.perf_counter() - start))
    median_ms = statistics.median(samples_ms)
    atoms_count = len(atoms)
    native_path = Path(native.__file__).resolve()
    symmetrix_results = {
        name: calculator.results[name].copy()
        if hasattr(calculator.results[name], "copy")
        else calculator.results[name]
        for name in properties
    }
    report = {
        "model": str(model),
        "model_sha256": _sha256(model),
        "runtime_model": str(runtime_model),
        "extraction_cache_hit": extraction_cache_hit,
        "backend": backend,
        "native_extension": str(native_path),
        "native_extension_sha256": _sha256(native_path),
        "native_build": getattr(native, "_backend_build_info", dict)(),
        "kokkos_execution_space": getattr(
            native, "_kokkos_default_execution_space", lambda: "unknown"
        )(),
        "dtype": args.dtype,
        "streamed_edges": calculator.streamed_edges,
        "execution_profile": args.profile,
        "execution_plan": calculator.execution_plan,
        "atoms": atoms_count,
        "directed_edges": _directed_edge_count(calculator, atoms),
        "supercell_repeat": args.supercell_repeat,
        "rattle_angstrom": args.rattle,
        "neighbor_skin_angstrom": args.neighbor_skin,
        "model_cutoff_angstrom": float(calculator.cutoff),
        "effective_cutoff_angstrom": float(calculator.cutoff + args.neighbor_skin),
        "kokkos_openmp_threads": args.threads,
        "blas_threads": 1,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "model_setup_seconds": model_setup_seconds,
        "warmup_seconds": warmup_seconds,
        "measurement_seconds": sum(samples_ms) / 1000.0,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "median_us_per_atom": median_ms * 1000.0 / atoms_count,
        "atoms_per_second": atoms_count * 1000.0 / median_ms,
        "properties": properties,
    }
    atoms.calc = None
    calculator.reset()
    del calculator
    _finalize_loaded_backend(native)
    _progress(
        "Benchmark results ready before reference validation: "
        f"{report['median_us_per_atom']:.3f} us/atom, "
        f"{report['atoms_per_second']:.1f} atoms/s"
    )
    if getattr(args, "skip_validation", False):
        _progress("[7/7] Skipping MACE-Torch reference validation")
    else:
        _progress("[7/7] Validating energy, forces, and stress against MACE-Torch")
    report["correctness"] = _run_validation(
        model,
        atoms,
        symmetrix_results,
        args.dtype,
        backend,
        skip=getattr(args, "skip_validation", False),
    )
    if report["correctness"]["status"] == "skipped":
        _progress(f"Correctness validation skipped: {report['correctness']['reason']}")
    elif report["correctness"]["status"] == "error":
        _progress(
            f"Correctness validation unavailable: {report['correctness']['reason']}"
        )
    else:
        _progress("MACE-Torch correctness validation passed")
    del atoms
    _progress("Benchmark complete; summary follows")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="symmetrix bench", description=__doc__)
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=(
            "local checkpoint/JSON path or a mace_mp foundation-model name "
            f"(default: {_DEFAULT_MODEL})"
        ),
    )
    parser.add_argument("--backend", help="backend selector; default is automatic")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--profile", choices=("capacity", "speed"), default="capacity")
    parser.add_argument(
        "--supercell-repeat", type=int, default=6, help="SrTiO3 cell repeat"
    )
    parser.add_argument("--rattle", type=float, default=0.02, help="rattle in Angstrom")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--neighbor-skin", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "skip the untimed MACE-Torch energy/force/stress check; useful for "
            "large cells where the reference model would exceed available memory"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        report = run(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        # Preserve the actionable benchmark error if an evaluator still owns
        # Kokkos and teardown reports the secondary lifetime failure.
        try:
            _finalize_loaded_backend()
        except RuntimeError:
            pass
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        backend = report["backend"] or {}
        print(
            f"Model: {Path(report['model']).name}; backend: "
            f"{backend.get('selector', 'automatic')}"
        )
        print(
            f"Structure: {report['atoms']} atoms, {report['directed_edges']} directed edges; "
            f"dtype={report['dtype']}, profile={report['execution_profile']}"
        )
        print(
            f"Cutoff: {report['model_cutoff_angstrom']:.3f} A model + "
            f"{report['neighbor_skin_angstrom']:.3f} A skin = "
            f"{report['effective_cutoff_angstrom']:.3f} A"
        )
        print(
            f"Median: {report['median_us_per_atom']:.3f} us/atom "
            f"({report['atoms_per_second']:.1f} atoms/s)"
        )
        print(
            f"Protocol: warmups={report['warmups']}, samples={report['repeats']}; "
            f"Kokkos/OpenMP threads={report['kokkos_openmp_threads']}, "
            f"BLAS threads={report['blas_threads']}"
        )
        print(
            f"Setup: {report['model_setup_seconds']:.2f} s; "
            f"warmup: {report['warmup_seconds']:.2f} s; "
            f"measurement: {report['measurement_seconds']:.2f} s"
        )
        plan = report["execution_plan"] or {}
        print(f"Execution: {plan.get('selected_id', report['streamed_edges'])}")
        correctness = report.get("correctness")
        if correctness is not None:
            if correctness["status"] == "skipped":
                print(f"Correctness: skipped ({correctness['reason']})")
            elif correctness["status"] == "error":
                print(f"Correctness: unavailable ({correctness['reason']})")
            else:
                errors = correctness["errors"]
                print(
                    "Correctness: passed against MACE-Torch; "
                    f"|dE|/atom={errors['energy_abs_eV_per_atom']:.3e} eV, "
                    f"max|dF|={errors['forces_max_abs_eV_per_A']:.3e} eV/A, "
                    f"max|dstress|={errors['stress_max_abs_eV_per_A3']:.3e} eV/A^3"
                )
    return 0
