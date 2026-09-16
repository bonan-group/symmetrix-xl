#!/usr/bin/env python3
"""Orchestrate the matched Symmetrix/upstream factorized R1 benchmark."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _bootstrap_explicit_symmetrix() -> None:
    """Load the requested source package and extension in fresh subprocesses."""

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


_bootstrap_explicit_symmetrix()


@contextlib.contextmanager
def _nvtx_range(name: str):
    # Use the same NVTX path as the upstream implementation runner. Nsight Systems does
    # not recognize the local libnvtx3interop ctypes path as a capture trigger.
    import torch

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


from symmetrix.factorized_r1_operator_fixture import (  # noqa: E402 - requires the explicit bootstrap above
    ATOL,
    LANE,
    RTOL,
    SCHEMA,
    SCHEMA_VERSION,
    ExecutionR1OperatorFixture,
    FixtureValidationError,
    canonical_contract,
    canonical_conventions,
    graph_sha256,
    load_fixture,
    sha256_file,
    validate_arrays,
    write_fixture,
)

_RESULT_NAMES = {
    "A1": ("A1", "out", "output", "expected_out"),
    "H1_adj": ("H1_adj", "grad_x", "source_adjoint", "expected_grad_x"),
    "edge_force": (
        "edge_force",
        "directed_force",
        "expected_directed_force",
    ),
}
_REFERENCE_NAMES = {
    "A1": "expected_out",
    "H1_adj": "expected_grad_x",
    "edge_force": "expected_directed_force",
}
_MODE_RESULTS = {
    "forward": ("A1",),
    "reverse": ("H1_adj", "edge_force"),
    "forward_backward": ("A1", "H1_adj", "edge_force"),
}
_NATIVE_MODES = {
    "forward": "forward",
    "reverse": "reverse",
    "forward_backward": "forward_reverse",
}
_TIMING_SEMANTICS = {
    "forward": "prepared_graph_forward_only",
    "reverse": "prepared_forward_state_reverse_only",
    "forward_backward": (
        "fresh_forward_plus_fixed_weight_backward_plus_directed_force"
    ),
}
_TIMING_PROTOCOL = {
    "device": "cuda",
    "sample_timer": "cuda_events",
    "stream_scope": "backend_active_cuda_stream",
    "cold_completion": "synchronize_after_cold_iteration",
    "warmup_completion": "single_synchronize_after_warmup_batch",
    "sample_completion": "synchronize_end_event",
    "stopping_rule": ("min_samples_and_min_sample_ms_capped_by_max_samples"),
    "preparation": "outside_timed_samples",
    "validation": "outside_timed_samples",
    "raw_samples_retained": True,
}


@dataclass(frozen=True)
class MeasurementConfig:
    warmup_iterations: int = 10
    warmup_ms: float = 100.0
    min_samples: int = 20
    max_samples: int = 100
    min_sample_ms: float = 1000.0

    def validate(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be nonnegative")
        if self.warmup_ms < 0.0 or self.min_sample_ms < 0.0:
            raise ValueError("warmup_ms and min_sample_ms must be nonnegative")
        if self.min_samples < 1 or self.max_samples < self.min_samples:
            raise ValueError(
                "sample counts must satisfy 1 <= min_samples <= max_samples"
            )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "warmup_iterations": self.warmup_iterations,
            "warmup_ms": self.warmup_ms,
            "min_samples": self.min_samples,
            "max_samples": self.max_samples,
            "min_sample_ms": self.min_sample_ms,
        }


class NativeOperatorAdapter:
    """The only place that knows the private native API spelling."""

    _METHODS = {
        "prepare": "_prepare_factorized_operator_benchmark",
        "run": "_run_factorized_operator_benchmark",
        "measure": "_measure_factorized_operator_benchmark",
        "outputs": "_factorized_operator_benchmark_outputs",
        "export": "_factorized_operator_export",
    }

    def __init__(self, evaluator: Any):
        self.evaluator = evaluator
        missing = [
            name
            for name in self._METHODS.values()
            if not callable(getattr(evaluator, name, None))
        ]
        if missing:
            raise RuntimeError(
                "the Symmetrix extension does not expose the Phase 18 native API: "
                + ", ".join(missing)
            )

    def _method(self, operation: str) -> Any:
        return getattr(self.evaluator, self._METHODS[operation])

    def prepare(self, graph_generation: int, xyz: np.ndarray, r: np.ndarray) -> int:
        return int(self._method("prepare")(graph_generation, xyz, r))

    def export(self, token: int) -> dict[str, np.ndarray]:
        values = self._method("export")(token)
        if not isinstance(values, Mapping):
            raise RuntimeError("native fixture export did not return a mapping")
        return {
            str(name): np.array(value, copy=True, order="C")
            for name, value in values.items()
        }

    def run(self, token: int, mode: str) -> None:
        self._method("run")(token, _NATIVE_MODES[mode], True)

    def outputs(self, token: int) -> dict[str, np.ndarray]:
        values = self._method("outputs")(token)
        if not isinstance(values, Mapping):
            raise RuntimeError("native benchmark outputs did not return a mapping")
        return {
            str(name): np.array(value, copy=True, order="C")
            for name, value in values.items()
        }

    def measure(
        self, token: int, mode: str, config: MeasurementConfig
    ) -> dict[str, Any]:
        values = self._method("measure")(
            token,
            _NATIVE_MODES[mode],
            config.warmup_iterations,
            config.warmup_ms,
            config.min_samples,
            config.max_samples,
            config.min_sample_ms,
        )
        if not isinstance(values, Mapping):
            raise RuntimeError("native timing entry point did not return a mapping")
        result = _json_value(dict(values))
        samples = result.get("samples_ms")
        if not isinstance(samples, list) or not samples:
            raise RuntimeError("native timing report did not retain raw samples_ms")
        return result


@dataclass
class PreparedNativeCase:
    calculator: Any
    adapter: NativeOperatorAdapter
    token: int
    graph_generation: int
    atoms: Any
    system_provenance: dict[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_json(path: pathlib.Path | None, report: Mapping[str, Any]) -> None:
    rendered = (
        json.dumps(
            _json_value(report),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if path is None:
        print(rendered, end="")
        return
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="ascii")


def _repository_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repository_root(),
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _case_atoms(
    structure: pathlib.Path | None, repeat: int
) -> tuple[Any, dict[str, Any]]:
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if structure is not None:
        from ase.io import read

        resolved = structure.resolve()
        atoms = read(resolved)
        provenance = {
            "kind": "ase-structure-file",
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    else:
        from ase.build import bulk

        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(
            (repeat, repeat, repeat)
        )
        provenance = {
            "kind": "AlN-wurtzite",
            "a_angstrom": 3.112,
            "c_angstrom": 4.982,
            "repeat": [repeat, repeat, repeat],
        }
    provenance["atoms"] = len(atoms)
    return atoms, provenance


def _prepare_native_case(
    model: pathlib.Path, structure: pathlib.Path | None, repeat: int
) -> PreparedNativeCase:
    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native_symmetrix

    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space != "Cuda":
        raise RuntimeError(
            "the matched upstream R1 benchmark requires Kokkos CUDA; use "
            "receiver_factorized_rtc_benchmark.py for the CPU experiment"
        )

    model = model.resolve()
    atoms, system_provenance = _case_atoms(structure, repeat)
    calculator = Symmetrix(
        model, use_kokkos=True, dtype="float32", streamed_edges="factorized"
    )
    evaluator = calculator.evaluator
    for name, value in (
        ("_set_factorized_source_strategy", "jit_plugin"),
        ("_set_factorized_direct_forward_executor", "jit_all"),
        ("_set_factorized_direct_reverse_executor", "jit"),
    ):
        method = getattr(evaluator, name, None)
        if callable(method):
            method(value)
    inputs = calculator._mace_inputs(atoms)
    graph_generation = int(evaluator._prepare_factorized_graph(*inputs[:5]))
    if graph_generation < 1:
        raise RuntimeError("native graph preparation did not return a generation token")
    xyz = np.ascontiguousarray(inputs[5], dtype=np.float64)
    r = np.ascontiguousarray(inputs[6], dtype=np.float64)
    # Populate production H1/Y and capture the fixed post-density A1 cotangent.
    # This full-model setup is intentionally outside every operator timing range.
    evaluator._compute_prepared_factorized(graph_generation, xyz.reshape(-1), r)
    adapter = NativeOperatorAdapter(evaluator)
    token = adapter.prepare(graph_generation, xyz, r)
    if token < 1:
        raise RuntimeError("native operator preparation did not return a token")
    return PreparedNativeCase(
        calculator=calculator,
        adapter=adapter,
        token=token,
        graph_generation=graph_generation,
        atoms=atoms,
        system_provenance=system_provenance,
    )


def _native_provenance(
    case: PreparedNativeCase, model: pathlib.Path, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    from symmetrix import symmetrix as native_symmetrix

    evaluator = case.calculator.evaluator
    extension_path = pathlib.Path(native_symmetrix.__file__).resolve()
    optional_fields = (
        "factorized_model_contract_fingerprint",
        "factorized_model_semantic_fingerprint",
        "jit_contract_fingerprint",
        "jit_artifact_id",
        "factorized_selected_direct_forward_executor",
        "factorized_selected_direct_reverse_executor",
    )
    native = {
        name: _json_value(getattr(evaluator, name))
        for name in optional_fields
        if hasattr(evaluator, name)
    }
    execution_space = getattr(native_symmetrix, "_kokkos_default_execution_space", None)
    return {
        "producer": "benchmarks/factorized_operator_benchmark.py",
        "symmetrix_commit": _git_commit(),
        "model_sha256": sha256_file(model),
        "graph_sha256": graph_sha256(arrays["edge_index"]),
        "graph_generation": case.graph_generation,
        "model_path": str(model.resolve()),
        "native_extension": str(extension_path),
        "native_extension_sha256": sha256_file(extension_path),
        "kokkos_execution_space": (
            execution_space() if callable(execution_space) else None
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "process_id": os.getpid(),
        "system": case.system_provenance,
        "native": native,
    }


def export_fixture(
    model: pathlib.Path,
    manifest_path: pathlib.Path,
    *,
    structure: pathlib.Path | None = None,
    repeat: int = 6,
    overwrite: bool = False,
) -> ExecutionR1OperatorFixture:
    case = _prepare_native_case(model, structure, repeat)
    arrays = case.adapter.export(case.token)
    validate_arrays(arrays, require_expected=True)
    provenance = _native_provenance(case, model.resolve(), arrays)
    return write_fixture(manifest_path, arrays, provenance, overwrite=overwrite)


def _select_output(outputs: Mapping[str, np.ndarray], result_name: str) -> np.ndarray:
    found = [name for name in _RESULT_NAMES[result_name] if name in outputs]
    if len(found) != 1:
        raise RuntimeError(
            f"native outputs must contain exactly one alias for {result_name}: "
            + ", ".join(_RESULT_NAMES[result_name])
        )
    return outputs[found[0]]


def _array_comparison(
    actual: np.ndarray, expected: np.ndarray, *, rtol: float = RTOL, atol: float = ATOL
) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise FixtureValidationError(
            f"output shape mismatch: actual {actual.shape}, expected {expected.shape}"
        )
    if actual.dtype != expected.dtype:
        raise FixtureValidationError(
            f"output dtype mismatch: actual {actual.dtype.str}, expected {expected.dtype.str}"
        )
    if not np.isfinite(actual).all():
        raise FixtureValidationError("output contains non-finite values")
    if actual.size == 0:
        return {
            "shape": list(actual.shape),
            "dtype": actual.dtype.str,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "max_scaled_error": 0.0,
            "max_error_index": None,
            "allclose": True,
        }
    error = np.abs(actual - expected)
    threshold = atol + rtol * np.abs(expected)
    flat_index = int(np.argmax(error))
    index = list(np.unravel_index(flat_index, error.shape))
    denominator = np.maximum(np.abs(expected), atol)
    return {
        "shape": list(actual.shape),
        "dtype": actual.dtype.str,
        "max_abs_error": float(np.max(error)),
        "max_rel_error": float(np.max(error / denominator)),
        "max_scaled_error": float(np.max(error / threshold)),
        "max_error_index": index,
        "allclose": bool(np.all(error <= threshold)),
    }


def _validate_outputs(
    fixture: ExecutionR1OperatorFixture,
    outputs: Mapping[str, np.ndarray],
    mode: str,
) -> dict[str, Any]:
    result = {
        "status": "pass",
        "rtol": RTOL,
        "atol": ATOL,
        "outputs": {},
    }
    for result_name in _MODE_RESULTS[mode]:
        reference_name = _REFERENCE_NAMES[result_name]
        comparison = _array_comparison(
            _select_output(outputs, result_name), fixture.arrays[reference_name]
        )
        comparison["reference_array"] = reference_name
        result["outputs"][result_name] = comparison
        if not comparison["allclose"]:
            result["status"] = "fail"
    return result


def _verify_recreated_case(
    fixture: ExecutionR1OperatorFixture, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    validate_arrays(arrays, require_expected=True)
    verification: dict[str, Any] = {"inputs_exact": True, "expected": {}}
    for name in sorted(set(arrays) - set(_REFERENCE_NAMES.values())):
        reference = fixture.arrays[name]
        current = arrays[name]
        if (
            current.dtype != reference.dtype
            or current.shape != reference.shape
            or not np.array_equal(current, reference)
        ):
            raise FixtureValidationError(
                f"recreated native input {name!r} differs from the fixture"
            )
    for result_name, reference_name in _REFERENCE_NAMES.items():
        comparison = _array_comparison(
            arrays[reference_name], fixture.arrays[reference_name]
        )
        comparison["reference_array"] = reference_name
        verification["expected"][result_name] = comparison
        if not comparison["allclose"]:
            raise FixtureValidationError(
                f"recreated native oracle {reference_name!r} differs from the fixture"
            )
    return verification


def _fixture_reference(fixture: ExecutionR1OperatorFixture) -> dict[str, Any]:
    return {
        "manifest": str(fixture.manifest_path),
        "manifest_sha256": fixture.manifest["manifest_sha256"],
        "payload_sha256": fixture.manifest["payload"]["sha256"],
        "model_sha256": fixture.manifest["provenance"]["model_sha256"],
        "graph_sha256": fixture.manifest["provenance"]["graph_sha256"],
        "symmetrix_commit": fixture.manifest["provenance"]["symmetrix_commit"],
        "graph_generation": fixture.manifest["provenance"]["graph_generation"],
        "nodes": fixture.num_nodes,
        "edges": fixture.num_edges,
    }


def run_local(
    model: pathlib.Path,
    fixture_path: pathlib.Path,
    *,
    mode: str = "forward_backward",
    structure: pathlib.Path | None = None,
    repeat: int = 6,
    measurement: MeasurementConfig = MeasurementConfig(),
    validate_only: bool = False,
    nvtx: bool = False,
) -> dict[str, Any]:
    if validate_only and nvtx:
        raise ValueError("validate_only and nvtx are mutually exclusive")
    measurement.validate()
    fixture = load_fixture(fixture_path, require_expected=True)
    if sha256_file(model.resolve()) != fixture.manifest["provenance"]["model_sha256"]:
        raise FixtureValidationError(
            "model SHA-256 differs from the fixture provenance"
        )
    case = _prepare_native_case(model, structure, repeat)
    current_export = case.adapter.export(case.token)
    recreated = _verify_recreated_case(fixture, current_export)
    case.adapter.run(case.token, mode)
    validation = _validate_outputs(fixture, case.adapter.outputs(case.token), mode)
    timing: dict[str, Any]
    nvtx_record: dict[str, Any] | None = None
    if validate_only:
        timing = {"skipped": True, "reason": "validate-only"}
    elif nvtx:
        range_name = f"symmetrix_r1_operator::{mode}"
        with _nvtx_range(range_name):
            case.adapter.run(case.token, mode)
        timing = {"skipped": True, "reason": "nvtx-single-iteration"}
        nvtx_record = {
            "range": range_name,
            "iterations": 1,
            "warmup_iterations": 1,
        }
    else:
        timing = case.adapter.measure(case.token, mode, measurement)
        timing.setdefault("config", measurement.as_dict())
    environment = _native_provenance(case, model.resolve(), current_export)
    report = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "lane": LANE,
        "status": validation["status"],
        "backend": "symmetrix",
        "mode": mode,
        "headline": mode == "forward_backward",
        "timing_semantics": _TIMING_SEMANTICS[mode],
        "timing_protocol": copy.deepcopy(_TIMING_PROTOCOL),
        "measurement": measurement.as_dict(),
        "fixture": _fixture_reference(fixture),
        "contract": copy.deepcopy(fixture.manifest["contract"]),
        "conventions": copy.deepcopy(fixture.manifest["conventions"]),
        "environment": environment,
        "validation": validation,
        "fixture_recreation": recreated,
        "timing": timing,
        "nvtx": nvtx_record,
    }
    return report


def _load_report(path: pathlib.Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureValidationError(f"failed to read result report {path}") from exc
    if not isinstance(report, dict):
        raise FixtureValidationError(f"result report {path} must be a JSON object")
    return report


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureValidationError(f"backend report is missing {description}")
    return value


def _require_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise FixtureValidationError(f"{description} is not a SHA-256 identity")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise FixtureValidationError(f"{description} is not a SHA-256 identity")
    return value


def _validate_measurement_config(value: Any) -> dict[str, Any]:
    measurement = _require_mapping(value, "measurement configuration")
    expected_keys = set(MeasurementConfig().as_dict())
    if set(measurement) != expected_keys:
        raise FixtureValidationError(
            "backend report measurement configuration has different fields"
        )
    for name in ("warmup_iterations", "min_samples", "max_samples"):
        item = measurement[name]
        if not isinstance(item, int) or isinstance(item, bool):
            raise FixtureValidationError(
                f"backend report measurement configuration has invalid {name}"
            )
    for name in ("warmup_ms", "min_sample_ms"):
        item = measurement[name]
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not np.isfinite(item)
        ):
            raise FixtureValidationError(
                f"backend report measurement configuration has invalid {name}"
            )
    try:
        MeasurementConfig(**dict(measurement)).validate()
    except (TypeError, ValueError) as exc:
        raise FixtureValidationError(
            "backend report measurement configuration is invalid"
        ) from exc
    return dict(measurement)


def _validate_backend_provenance(
    report: Mapping[str, Any],
    fixture: ExecutionR1OperatorFixture,
    backend: str,
    *,
    timing_skipped: bool,
) -> None:
    environment = _require_mapping(report.get("environment"), "environment provenance")
    if backend == "symmetrix":
        for key, expected in (
            ("model_sha256", fixture.manifest["provenance"]["model_sha256"]),
            ("graph_sha256", fixture.manifest["provenance"]["graph_sha256"]),
            (
                "symmetrix_commit",
                fixture.manifest["provenance"]["symmetrix_commit"],
            ),
            (
                "graph_generation",
                fixture.manifest["provenance"]["graph_generation"],
            ),
            ("kokkos_execution_space", "Cuda"),
        ):
            if environment.get(key) != expected:
                raise FixtureValidationError(
                    f"Symmetrix environment provenance mismatch for {key}"
                )
        extension_sha256 = _require_sha256(
            environment.get("native_extension_sha256"),
            "Symmetrix native extension",
        )
        fixture_extension = fixture.manifest["provenance"].get(
            "native_extension_sha256"
        )
        if fixture_extension is not None and extension_sha256 != fixture_extension:
            raise FixtureValidationError(
                "Symmetrix native extension differs from fixture provenance"
            )
        native = _require_mapping(
            environment.get("native"), "Symmetrix native execution provenance"
        )
        for key, expected in (
            ("factorized_selected_direct_forward_executor", "jit_all"),
            ("factorized_selected_direct_reverse_executor", "jit"),
        ):
            if native.get(key) != expected:
                raise FixtureValidationError(
                    f"Symmetrix native execution provenance mismatch for {key}"
                )
        for key in (
            "jit_artifact_id",
            "jit_contract_fingerprint",
        ):
            if not isinstance(native.get(key), str) or not native[key]:
                raise FixtureValidationError(
                    f"Symmetrix native execution provenance is missing {key}"
                )
        if not timing_skipped:
            cuda = _require_mapping(
                _require_mapping(report.get("timing"), "timing data").get("cuda"),
                "Symmetrix CUDA timing provenance",
            )
            if not isinstance(cuda.get("name"), str) or not cuda["name"]:
                raise FixtureValidationError("Symmetrix timing is missing the GPU name")
            if not isinstance(cuda.get("compute_capability"), str):
                raise FixtureValidationError(
                    "Symmetrix timing is missing compute capability"
                )
            if not isinstance(cuda.get("runtime_version"), int):
                raise FixtureValidationError(
                    "Symmetrix timing is missing the CUDA runtime version"
                )
        return

    if not isinstance(environment.get("device"), str) or not environment[
        "device"
    ].startswith("cuda"):
        raise FixtureValidationError(
            "upstream implementation did not report a CUDA device"
        )
    for key in ("gpu_name", "compute_capability", "torch", "torch_cuda"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise FixtureValidationError(
                f"upstream implementation environment provenance is missing {key}"
            )
    upstream = _require_mapping(
        report.get("upstream"), "upstream implementation provenance"
    )
    if not isinstance(upstream.get("module"), str) or not upstream["module"]:
        raise FixtureValidationError(
            "upstream implementation provenance has no module name"
        )
    commit = upstream.get("commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise FixtureValidationError(
            "upstream implementation provenance has no exact commit"
        )
    if upstream.get("dirty") is not False:
        raise FixtureValidationError(
            "upstream implementation provenance is not a clean checkout"
        )
    if not isinstance(upstream.get("root"), str) or not upstream["root"]:
        raise FixtureValidationError(
            "upstream implementation provenance has no source root"
        )
    execution = _require_mapping(
        report.get("execution"), "upstream implementation execution provenance"
    )
    if execution.get("gemm") != "ieee":
        raise FixtureValidationError(
            "upstream implementation comparison must use the IEEE GEMM protocol"
        )
    codegen = _require_mapping(
        execution.get("codegen_config"), "upstream implementation codegen provenance"
    )
    if codegen.get("gemm") != execution.get("gemm"):
        raise FixtureValidationError(
            "upstream implementation GEMM and codegen protocols differ"
        )
    if execution.get("parameter_gradients") is not False:
        raise FixtureValidationError(
            "upstream implementation report enabled out-of-scope parameter gradients"
        )


def _validate_backend_report(
    report: Mapping[str, Any],
    fixture: ExecutionR1OperatorFixture,
    *,
    mode: str | None,
    backend: str,
) -> None:
    for key, expected in (
        ("schema", SCHEMA),
        ("version", SCHEMA_VERSION),
        ("lane", LANE),
    ):
        if report.get(key) != expected:
            raise FixtureValidationError(
                f"backend report semantic mismatch for {key}: {report.get(key)!r}"
            )
    if report.get("backend") != backend:
        raise FixtureValidationError(
            f"backend report identity mismatch: expected {backend!r}"
        )
    if mode is not None and report.get("mode") != mode:
        raise FixtureValidationError("backend reports use different measurement modes")
    fixture_ref = report.get("fixture")
    if not isinstance(fixture_ref, Mapping):
        raise FixtureValidationError("backend report is missing fixture identity")
    for key, expected in (
        ("manifest_sha256", fixture.manifest["manifest_sha256"]),
        ("payload_sha256", fixture.manifest["payload"]["sha256"]),
        ("model_sha256", fixture.manifest["provenance"]["model_sha256"]),
        ("graph_sha256", fixture.manifest["provenance"]["graph_sha256"]),
        (
            "symmetrix_commit",
            fixture.manifest["provenance"]["symmetrix_commit"],
        ),
        (
            "graph_generation",
            fixture.manifest["provenance"]["graph_generation"],
        ),
        ("nodes", fixture.num_nodes),
        ("edges", fixture.num_edges),
    ):
        if fixture_ref.get(key) != expected:
            raise FixtureValidationError(
                f"backend report fixture identity mismatch for {key}"
            )
    if report.get("contract") != canonical_contract():
        raise FixtureValidationError(
            "backend report contract does not match the fixture"
        )
    if report.get("conventions") != canonical_conventions():
        raise FixtureValidationError(
            "backend report conventions do not match the fixture"
        )
    validation = report.get("validation")
    if not isinstance(validation, Mapping):
        raise FixtureValidationError("backend report is missing validation results")
    if validation.get("rtol") != RTOL or validation.get("atol") != ATOL:
        raise FixtureValidationError(
            "backend report used different comparison tolerances"
        )
    outputs = validation.get("outputs")
    if not isinstance(outputs, Mapping):
        raise FixtureValidationError("backend report is missing per-output validation")
    expected_results = _MODE_RESULTS[str(report.get("mode"))]
    if set(outputs) != set(expected_results):
        raise FixtureValidationError(
            "backend report validated a different derivative signature"
        )
    for name in expected_results:
        entry = outputs[name]
        if not isinstance(entry, Mapping):
            raise FixtureValidationError(
                f"backend report has invalid {name} validation"
            )
        if entry.get("reference_array") != _REFERENCE_NAMES[name]:
            raise FixtureValidationError(f"backend report changed the {name} reference")
        if entry.get("allclose") is not True:
            raise FixtureValidationError(f"backend report failed {name} comparison")
    if validation.get("status") != "pass" or report.get("status") != "pass":
        raise FixtureValidationError("backend report did not pass validation")
    timing = report.get("timing")
    if not isinstance(timing, Mapping):
        raise FixtureValidationError("backend report is missing timing data")
    skipped = timing.get("skipped", False)
    if not isinstance(skipped, bool):
        raise FixtureValidationError("backend report timing skip state is invalid")
    if skipped:
        if timing.get("reason") not in ("validate-only", "nvtx-single-iteration"):
            raise FixtureValidationError("backend report timing skip reason is invalid")
    else:
        samples = timing.get("samples_ms")
        if not isinstance(samples, list) or not samples:
            raise FixtureValidationError(
                "backend report did not retain raw timing samples"
            )
        if not all(
            isinstance(sample, (int, float))
            and not isinstance(sample, bool)
            and np.isfinite(sample)
            and sample > 0.0
            for sample in samples
        ):
            raise FixtureValidationError(
                "backend report contains invalid timing samples"
            )
        if timing.get("timing_backend") != "cuda_events":
            raise FixtureValidationError(
                "backend report did not use CUDA events for samples"
            )
        expected_stream = (
            "factorized_execution_space.cuda_stream"
            if backend == "symmetrix"
            else "torch_current_cuda_stream"
        )
        if timing.get("execution_stream") != expected_stream:
            raise FixtureValidationError(
                "backend report timed a different CUDA execution stream"
            )
    expected_mode = str(report.get("mode"))
    if report.get("timing_semantics") != _TIMING_SEMANTICS[expected_mode]:
        raise FixtureValidationError(
            "backend report changed the timed operator boundary"
        )
    if report.get("timing_protocol") != _TIMING_PROTOCOL:
        raise FixtureValidationError(
            "backend report changed the timing synchronization protocol"
        )
    _validate_measurement_config(report.get("measurement"))
    _validate_backend_provenance(report, fixture, backend, timing_skipped=skipped)


def _cuda_major_from_runtime_version(value: Any) -> int:
    if not isinstance(value, int) or value < 1000:
        raise FixtureValidationError("invalid Symmetrix CUDA runtime version")
    return value // 1000


def _cuda_major_from_torch_version(value: Any) -> int:
    if not isinstance(value, str):
        raise FixtureValidationError("invalid upstream implementation CUDA version")
    try:
        return int(value.split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise FixtureValidationError(
            "invalid upstream implementation CUDA version"
        ) from exc


def _validate_matched_protocol(
    symmetrix_report: Mapping[str, Any], execution_report: Mapping[str, Any]
) -> dict[str, Any]:
    local_measurement = dict(
        _require_mapping(
            symmetrix_report.get("measurement"), "measurement configuration"
        )
    )
    upstream_measurement = dict(
        _require_mapping(
            execution_report.get("measurement"), "measurement configuration"
        )
    )
    if local_measurement != upstream_measurement:
        raise FixtureValidationError(
            "backend reports used different measurement configurations"
        )
    local_timing = _require_mapping(symmetrix_report.get("timing"), "timing data")
    upstream_timing = _require_mapping(execution_report.get("timing"), "timing data")
    local_skipped = local_timing.get("skipped", False)
    upstream_skipped = upstream_timing.get("skipped", False)
    if local_skipped != upstream_skipped:
        raise FixtureValidationError(
            "backend reports used different timing execution protocols"
        )
    if local_skipped:
        if local_timing.get("reason") != upstream_timing.get("reason"):
            raise FixtureValidationError(
                "backend reports used different timing skip protocols"
            )
        gpu = None
    else:
        local_cuda = _require_mapping(
            local_timing.get("cuda"), "Symmetrix CUDA timing provenance"
        )
        upstream_environment = _require_mapping(
            execution_report.get("environment"),
            "upstream implementation environment provenance",
        )
        for local_key, upstream_key, description in (
            ("name", "gpu_name", "GPU name"),
            ("compute_capability", "compute_capability", "compute capability"),
        ):
            if local_cuda.get(local_key) != upstream_environment.get(upstream_key):
                raise FixtureValidationError(
                    f"backend reports used a different {description}"
                )
        local_cuda_major = _cuda_major_from_runtime_version(
            local_cuda.get("runtime_version")
        )
        upstream_cuda_major = _cuda_major_from_torch_version(
            upstream_environment.get("torch_cuda")
        )
        if local_cuda_major != upstream_cuda_major:
            raise FixtureValidationError(
                "backend reports used different CUDA major-version protocols"
            )
        gpu = {
            "name": local_cuda["name"],
            "compute_capability": local_cuda["compute_capability"],
            "cuda_major": local_cuda_major,
        }
    return {
        "measurement": local_measurement,
        "timing_semantics": symmetrix_report["timing_semantics"],
        "timing_protocol": copy.deepcopy(_TIMING_PROTOCOL),
        "timing_skipped": local_skipped,
        "gpu": gpu,
    }


def compare_reports(
    fixture_path: pathlib.Path,
    symmetrix_report: Mapping[str, Any],
    execution_report: Mapping[str, Any],
) -> dict[str, Any]:
    fixture = load_fixture(fixture_path, require_expected=True)
    mode = symmetrix_report.get("mode")
    if mode not in _MODE_RESULTS:
        raise FixtureValidationError("Symmetrix report has an unsupported mode")
    _validate_backend_report(
        symmetrix_report, fixture, mode=str(mode), backend="symmetrix"
    )
    _validate_backend_report(
        execution_report, fixture, mode=str(mode), backend="execution"
    )
    matched_protocol = _validate_matched_protocol(symmetrix_report, execution_report)
    return {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "lane": LANE,
        "status": "pass",
        "mode": mode,
        "fixture": _fixture_reference(fixture),
        "contract": copy.deepcopy(fixture.manifest["contract"]),
        "conventions": copy.deepcopy(fixture.manifest["conventions"]),
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "matched_protocol": matched_protocol,
        "backends": {
            "symmetrix": copy.deepcopy(dict(symmetrix_report)),
            "execution": copy.deepcopy(dict(execution_report)),
        },
    }


def run_upstream(
    fixture_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    upstream_root: pathlib.Path,
    upstream_module: str,
    mode: str,
    measurement: MeasurementConfig,
    gemm: str = "ieee",
    validate_only: bool = False,
    nvtx: bool = False,
) -> dict[str, Any]:
    runner = (
        pathlib.Path(__file__).resolve().with_name("execution_upstream_r1_runner.py")
    )
    if not runner.is_file():
        raise RuntimeError(f"upstream implementation runner is missing: {runner}")
    command = [
        sys.executable,
        str(runner),
        str(fixture_path.resolve()),
        "--upstream-root",
        str(upstream_root.resolve()),
        "--upstream-module",
        upstream_module,
        "--mode",
        mode,
        "--output",
        str(output_path.resolve()),
        "--gemm",
        gemm,
        "--warmup-iterations",
        str(measurement.warmup_iterations),
        "--warmup-ms",
        str(measurement.warmup_ms),
        "--min-samples",
        str(measurement.min_samples),
        "--max-samples",
        str(measurement.max_samples),
        "--min-sample-ms",
        str(measurement.min_sample_ms),
    ]
    if validate_only:
        command.append("--validate-only")
    if nvtx:
        command.append("--nvtx")
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"upstream implementation runner failed with exit code {result.returncode}: {detail}"
        )
    if not output_path.is_file():
        raise RuntimeError(
            "upstream implementation runner did not write its requested report"
        )
    return _load_report(output_path)


def run_local_subprocess(
    model: pathlib.Path,
    fixture_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    mode: str,
    structure: pathlib.Path | None,
    repeat: int,
    measurement: MeasurementConfig,
    validate_only: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "local",
        str(model.resolve()),
        str(fixture_path.resolve()),
        "--mode",
        mode,
        "--output",
        str(output_path.resolve()),
        "--warmup-iterations",
        str(measurement.warmup_iterations),
        "--warmup-ms",
        str(measurement.warmup_ms),
        "--min-samples",
        str(measurement.min_samples),
        "--max-samples",
        str(measurement.max_samples),
        "--min-sample-ms",
        str(measurement.min_sample_ms),
    ]
    if structure is not None:
        command.extend(("--structure", str(structure.resolve())))
    else:
        command.extend(("--repeat", str(repeat)))
    if validate_only:
        command.append("--validate-only")
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "fresh-process Symmetrix benchmark failed with exit code "
            f"{result.returncode}: {detail}"
        )
    if not output_path.is_file():
        raise RuntimeError("fresh-process Symmetrix benchmark did not write its report")
    return _load_report(output_path)


def _add_system_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--structure", type=pathlib.Path)
    group.add_argument(
        "--repeat",
        type=int,
        default=6,
        help="AlN wurtzite repeat used when --structure is omitted (default: 6)",
    )


def _add_measurement_arguments(
    parser: argparse.ArgumentParser, *, allow_validate_only: bool
) -> None:
    parser.add_argument(
        "--mode", choices=tuple(_MODE_RESULTS), default="forward_backward"
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--warmup-ms", type=float, default=100.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--min-sample-ms", type=float, default=1000.0)
    if allow_validate_only:
        parser.add_argument("--validate-only", action="store_true")


def _measurement_from_args(args: argparse.Namespace) -> MeasurementConfig:
    config = MeasurementConfig(
        warmup_iterations=args.warmup_iterations,
        warmup_ms=args.warmup_ms,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        min_sample_ms=args.min_sample_ms,
    )
    config.validate()
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("model", type=pathlib.Path)
    export_parser.add_argument("fixture", type=pathlib.Path)
    _add_system_arguments(export_parser)
    export_parser.add_argument("--overwrite", action="store_true")

    local_parser = subparsers.add_parser("local")
    local_parser.add_argument("model", type=pathlib.Path)
    local_parser.add_argument("fixture", type=pathlib.Path)
    _add_system_arguments(local_parser)
    _add_measurement_arguments(local_parser, allow_validate_only=True)
    local_parser.add_argument("--nvtx", action="store_true")
    local_parser.add_argument("--output", type=pathlib.Path)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("fixture", type=pathlib.Path)
    compare_parser.add_argument("symmetrix_report", type=pathlib.Path)
    compare_parser.add_argument("execution_report", type=pathlib.Path)
    compare_parser.add_argument("--output", type=pathlib.Path)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("model", type=pathlib.Path)
    all_parser.add_argument("fixture", type=pathlib.Path)
    _add_system_arguments(all_parser)
    _add_measurement_arguments(all_parser, allow_validate_only=True)
    all_parser.add_argument(
        "--upstream-root",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/upstream-r1"),
    )
    all_parser.add_argument(
        "--upstream-module",
        default=os.environ.get("SYMMETRIX_UPSTREAM_R1_MODULE"),
        required="SYMMETRIX_UPSTREAM_R1_MODULE" not in os.environ,
        help=(
            "Python package exposing nn.TPConv and nn.TPConvGraph "
            "(or set SYMMETRIX_UPSTREAM_R1_MODULE)"
        ),
    )
    all_parser.add_argument("--gemm", choices=("ieee", "tf32"), default="ieee")
    all_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    all_parser.add_argument("--output", type=pathlib.Path)
    all_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            fixture = export_fixture(
                args.model,
                args.fixture,
                structure=args.structure,
                repeat=args.repeat,
                overwrite=args.overwrite,
            )
            _write_json(
                None,
                {
                    "status": "pass",
                    "fixture": _fixture_reference(fixture),
                },
            )
            return 0

        if args.command == "local":
            report = run_local(
                args.model,
                args.fixture,
                mode=args.mode,
                structure=args.structure,
                repeat=args.repeat,
                measurement=_measurement_from_args(args),
                validate_only=args.validate_only,
                nvtx=args.nvtx,
            )
            _write_json(args.output, report)
            return int(report["status"] != "pass")

        if args.command == "compare":
            report = compare_reports(
                args.fixture,
                _load_report(args.symmetrix_report),
                _load_report(args.execution_report),
            )
            _write_json(args.output, report)
            return 0

        if args.command == "all":
            measurement = _measurement_from_args(args)
            if args.fixture.is_file() and not args.overwrite:
                fixture = load_fixture(args.fixture, require_expected=True)
            else:
                fixture = export_fixture(
                    args.model,
                    args.fixture,
                    structure=args.structure,
                    repeat=args.repeat,
                    overwrite=args.overwrite,
                )
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            local_path = output_dir / f"symmetrix-{args.mode}.json"
            execution_path = output_dir / f"execution-{args.mode}.json"
            local_report = run_local_subprocess(
                args.model,
                fixture.manifest_path,
                local_path,
                mode=args.mode,
                structure=args.structure,
                repeat=args.repeat,
                measurement=measurement,
                validate_only=args.validate_only,
            )
            execution_report = run_upstream(
                fixture.manifest_path,
                execution_path,
                upstream_root=args.upstream_root,
                upstream_module=args.upstream_module,
                mode=args.mode,
                measurement=measurement,
                gemm=args.gemm,
                validate_only=args.validate_only,
                nvtx=False,
            )
            comparison = compare_reports(
                fixture.manifest_path, local_report, execution_report
            )
            comparison["raw_reports"] = {
                "symmetrix": str(local_path),
                "execution": str(execution_path),
            }
            _write_json(args.output, comparison)
            return 0
    except (
        FixtureValidationError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        gc.collect()
        native = sys.modules.get("symmetrix.symmetrix")
        is_initialized = getattr(native, "_kokkos_is_initialized", None)
        finalize = getattr(native, "_finalize_kokkos", None)
        if callable(is_initialized) and callable(finalize) and is_initialized():
            finalize()
