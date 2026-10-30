"""Run the extracted OMAT R1 operator through a compatible upstream API.

This module intentionally depends only on the versioned fixture, NumPy, PyTorch,
and a caller-selected upstream module. It does not import Symmetrix production
code, so the comparison remains an independent executable boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

FIXTURE_SCHEMA = "symmetrix.execution.r1-operator-benchmark"
FIXTURE_VERSION = 1
FIXTURE_LANE = "extracted_mace_exact_uvw"
REPORT_SCHEMA = FIXTURE_SCHEMA
REPORT_VERSION = 1

CHANNELS = 128
RADIAL_EMBEDDING = 64
INPUT_LM = 4
EDGE_LM = 16
OUTPUT_LM = 16
PATH_COUNT = 10
RADIAL_LINEAR_BLOCK_SIZE = CHANNELS * CHANNELS
WEIGHT_NUMEL = PATH_COUNT * RADIAL_LINEAR_BLOCK_SIZE

INPUT_IRREPS = "128x0e+128x1o"
SH_IRREPS = "1x0e+1x1o+1x2e+1x3o"
OUTPUT_IRREPS = "128x0e+128x1o+128x2e+128x3o"

# Path order is part of the serialized radial projection contract.  For UVW,
# output_elements is eta_count * input multiplicity, so raw=eta_count*128
# gives the local sqrt(2*l_out+1) path normalization exactly.
PATH_TRIPLES = (
    (0, 0, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 2, 1),
    (0, 2, 2),
    (1, 1, 2),
    (1, 3, 2),
    (0, 3, 3),
    (1, 2, 3),
)
ETA_COUNTS = (2, 3, 3, 2)
RAW_PATH_WEIGHTS = tuple(ETA_COUNTS[i_out] * CHANNELS for _, _, i_out in PATH_TRIPLES)
TP_INSTRUCTIONS = tuple(
    (i_in, i_sh, i_out, "uvw", True, float(raw_weight))
    for (i_in, i_sh, i_out), raw_weight in zip(PATH_TRIPLES, RAW_PATH_WEIGHTS)
)

EXPECTED_CONTRACT = {
    "input_irreps": INPUT_IRREPS,
    "sh_irreps": SH_IRREPS,
    "output_irreps": OUTPUT_IRREPS,
    "connection_mode": "uvw",
    "radial_embedding": RADIAL_EMBEDDING,
    "instructions": [
        {
            "index": index,
            "i_in": i_in,
            "i_sh": i_sh,
            "i_out": i_out,
            "raw_path_weight": float(raw_weight),
        }
        for index, ((i_in, i_sh, i_out), raw_weight) in enumerate(
            zip(PATH_TRIPLES, RAW_PATH_WEIGHTS)
        )
    ],
    "radial_linear_layout": (
        "q,path,input_channel,output_channel; trailing dimensions C-flat"
    ),
    "radial_linear_block_size": RADIAL_LINEAR_BLOCK_SIZE,
    "parameter_gradients": False,
    "density_scaling": False,
}

REQUIRED_ARRAYS = (
    "x",
    "edge_index",
    "sh",
    "phi",
    "radial_linear",
    "grad_out",
    "xyz",
    "r",
    "dsh_dxyz",
    "dphi_dr",
    "expected_out",
    "expected_grad_x",
    "expected_directed_force",
)
# These are semantic layout identifiers, not merely shape descriptions.  The
# version-1 lane admits exactly one spelling and axis order for each array.
EXPECTED_LAYOUTS = {
    "x": "node,source_lm,channel",
    "edge_index": "endpoint,edge;endpoint=(source,receiver)",
    "sh": "edge,sh_lm",
    "phi": "edge,q",
    "radial_linear": "q,(path,input_channel,output_channel)-C-flat",
    "grad_out": "node,output_lm,channel",
    "xyz": "edge,coordinate",
    "r": "edge",
    "dsh_dxyz": "edge,coordinate,sh_lm",
    "dphi_dr": "edge,q",
    "expected_out": "node,output_lm,channel",
    "expected_grad_x": "node,source_lm,channel",
    "expected_directed_force": "edge,coordinate",
}

EXPECTED_CONVENTIONS = {
    "edge_index": "edge_index[0]=source;edge_index[1]=receiver",
    "edge_order": "receiver-major;stable-within-receiver",
    "xyz": "source-position-minus-receiver-position",
    "r": "euclidean-norm-of-xyz;native-float64-input",
    "directed_force": "expected_directed_force=-dE/dxyz",
    "atom_force_collection": (
        "source+=expected_directed_force;receiver-=expected_directed_force"
    ),
    "x_layout": "node,source_lm,channel;native-ir_mul",
    "output_layout": "node,output_lm,channel;native-ir_mul",
    "source_adjoint_layout": "node,source_lm,channel;native-ir_mul",
    "sh_lm": "lm=l*l+l+m",
    "coordinate_order": "x,y,z",
}

TIMING_SEMANTICS = {
    "forward": "prepared_graph_forward_only",
    "reverse": "prepared_forward_state_reverse_only",
    "forward_backward": (
        "fresh_forward_plus_fixed_weight_backward_plus_directed_force"
    ),
}
TIMING_PROTOCOL = {
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


class FixtureError(ValueError):
    """The versioned fixture does not satisfy the upstream runner contract."""


@dataclass(frozen=True)
class MeasurementConfig:
    """Exact timing defaults and stopping rule used by the upstream runner."""

    warmup_iterations: int = 10
    warmup_ms: float = 100.0
    min_samples: int = 20
    max_samples: int = 100
    min_sample_ms: float = 1000.0

    def validate(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.warmup_ms < 0.0:
            raise ValueError("warmup_ms must be non-negative")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if self.max_samples < self.min_samples:
            raise ValueError("max_samples must be at least min_samples")
        if self.min_sample_ms < 0.0:
            raise ValueError("min_sample_ms must be non-negative")


@dataclass
class LoadedFixture:
    manifest_path: Path
    payload_path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    manifest_sha256: str
    payload_sha256: str

    @property
    def node_count(self) -> int:
        return int(self.arrays["x"].shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.arrays["edge_index"].shape[1])


def _strip_sha256_prefix(value: str) -> str:
    if not isinstance(value, str):
        raise FixtureError("SHA-256 values must be strings")
    if not value.startswith("sha256:"):
        raise FixtureError("SHA-256 values must start with 'sha256:'")
    result = value.removeprefix("sha256:")
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise FixtureError(f"invalid SHA-256 value: {value!r}")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _manifest_content_sha256(manifest: dict[str, Any]) -> str:
    value = copy.deepcopy(manifest)
    value.pop("manifest_sha256", None)
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_fixture_paths(path: Path) -> tuple[Path, Path, dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.suffix == ".npz":
        payload_path = path
        manifest_path = path.with_suffix(".json")
    else:
        manifest_path = path
        payload_path = Path()
    if not manifest_path.is_file():
        raise FixtureError(f"fixture manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(
            f"cannot read fixture manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise FixtureError("fixture manifest root must be an object")
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise FixtureError("fixture manifest payload must be an object")
    if payload.get("format") != "npz":
        raise FixtureError("fixture payload format must be 'npz'")
    raw_payload_path = payload.get("path")
    if not isinstance(raw_payload_path, str) or not raw_payload_path:
        raise FixtureError("fixture payload.path must be a non-empty string")
    if Path(raw_payload_path).name != raw_payload_path:
        raise FixtureError("fixture payload.path must be a same-directory basename")
    declared_payload_path = (manifest_path.parent / raw_payload_path).resolve()
    if path.suffix != ".npz":
        payload_path = declared_payload_path
    elif payload_path != declared_payload_path:
        raise FixtureError(
            f"NPZ argument {payload_path} does not match payload.path {declared_payload_path}"
        )
    if payload_path.parent != manifest_path.parent:
        raise FixtureError("payload.path must name a file beside the manifest")
    if not payload_path.is_file():
        raise FixtureError(f"fixture payload does not exist: {payload_path}")
    return manifest_path, payload_path, manifest


def _validate_contract(manifest: dict[str, Any]) -> None:
    expected_top_level = {
        "schema",
        "version",
        "lane",
        "payload",
        "dimensions",
        "contract",
        "conventions",
        "arrays",
        "provenance",
        "manifest_sha256",
    }
    if set(manifest) != expected_top_level:
        raise FixtureError(
            "fixture manifest has unexpected keys: "
            f"expected {sorted(expected_top_level)}, got {sorted(manifest)}"
        )
    if manifest.get("schema") != FIXTURE_SCHEMA:
        raise FixtureError(
            f"fixture schema must be {FIXTURE_SCHEMA!r}, got {manifest.get('schema')!r}"
        )
    if manifest.get("version") != FIXTURE_VERSION:
        raise FixtureError(
            f"fixture version must be {FIXTURE_VERSION}, got {manifest.get('version')!r}"
        )
    if manifest.get("lane") != FIXTURE_LANE:
        raise FixtureError(
            f"fixture lane must be {FIXTURE_LANE!r}, got {manifest.get('lane')!r}"
        )

    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise FixtureError("fixture contract must be an object")
    if contract != EXPECTED_CONTRACT:
        raise FixtureError(
            "fixture contract, instructions, or radial path order do not match "
            "the exact folded UVW lane"
        )

    conventions = manifest.get("conventions")
    if not isinstance(conventions, dict):
        raise FixtureError("fixture conventions must be an object")
    if conventions != EXPECTED_CONVENTIONS:
        raise FixtureError(
            "fixture conventions do not match the extracted MACE derivative boundary"
        )

    payload = manifest.get("payload")
    if not isinstance(payload, dict) or set(payload) != {"path", "format", "sha256"}:
        raise FixtureError(
            "fixture payload must contain exactly path, format, and sha256"
        )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise FixtureError("fixture provenance must be an object")
    required_provenance = {
        "producer",
        "symmetrix_commit",
        "model_sha256",
        "graph_sha256",
        "graph_generation",
    }
    missing_provenance = required_provenance - set(provenance)
    if missing_provenance:
        raise FixtureError(
            f"fixture provenance is missing {sorted(missing_provenance)}"
        )
    for name in ("producer", "symmetrix_commit"):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise FixtureError(f"provenance.{name} must be a non-empty string")
    for name in ("model_sha256", "graph_sha256"):
        _strip_sha256_prefix(provenance[name])
    graph_generation = provenance["graph_generation"]
    if (
        not isinstance(graph_generation, int)
        or isinstance(graph_generation, bool)
        or graph_generation < 1
    ):
        raise FixtureError("provenance.graph_generation must be a positive integer")
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FixtureError("fixture provenance is not canonical JSON data") from exc


def _expected_shape(name: str, nodes: int, edges: int) -> tuple[tuple[int, ...], ...]:
    shapes: dict[str, tuple[tuple[int, ...], ...]] = {
        "x": ((nodes, INPUT_LM, CHANNELS),),
        "edge_index": ((2, edges),),
        "sh": ((edges, EDGE_LM),),
        "phi": ((edges, RADIAL_EMBEDDING),),
        "radial_linear": ((RADIAL_EMBEDDING, WEIGHT_NUMEL),),
        "grad_out": ((nodes, OUTPUT_LM, CHANNELS),),
        "xyz": ((edges, 3),),
        "dsh_dxyz": ((edges, 3, EDGE_LM),),
        "dphi_dr": ((edges, RADIAL_EMBEDDING),),
        "r": ((edges,),),
        "expected_out": ((nodes, OUTPUT_LM, CHANNELS),),
        "expected_grad_x": ((nodes, INPUT_LM, CHANNELS),),
        "expected_directed_force": ((edges, 3),),
    }
    return shapes[name]


def _expected_dtype(name: str) -> np.dtype:
    if name == "edge_index":
        return np.dtype(np.int64)
    if name in ("xyz", "r", "expected_directed_force"):
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def load_fixture(path: str | os.PathLike[str]) -> LoadedFixture:
    """Load and fully validate a version-1 NPZ/JSON operator fixture."""

    manifest_path, payload_path, manifest = _resolve_fixture_paths(Path(path))
    _validate_contract(manifest)

    expected_manifest_hash = _strip_sha256_prefix(manifest.get("manifest_sha256"))
    actual_manifest_hash = _manifest_content_sha256(manifest)
    if actual_manifest_hash != expected_manifest_hash:
        raise FixtureError(
            "manifest_sha256 mismatch: "
            f"expected {expected_manifest_hash}, computed {actual_manifest_hash}"
        )

    payload = manifest["payload"]
    expected_payload_hash = _strip_sha256_prefix(payload.get("sha256"))
    actual_payload_hash = _file_sha256(payload_path)
    if actual_payload_hash != expected_payload_hash:
        raise FixtureError(
            "payload SHA-256 mismatch: "
            f"expected {expected_payload_hash}, computed {actual_payload_hash}"
        )

    array_metadata = manifest.get("arrays")
    if not isinstance(array_metadata, dict):
        raise FixtureError("fixture arrays metadata must be an object")
    if set(array_metadata) != set(REQUIRED_ARRAYS):
        raise FixtureError(
            "fixture arrays metadata names do not match the exact comparison lane: "
            f"expected {sorted(REQUIRED_ARRAYS)}, got {sorted(array_metadata)}"
        )

    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            payload_names = set(archive.files)
            if payload_names != set(REQUIRED_ARRAYS):
                raise FixtureError(
                    "fixture payload names do not match the exact comparison lane: "
                    f"expected {sorted(REQUIRED_ARRAYS)}, got {sorted(payload_names)}"
                )
            arrays = {}
            for name in archive.files:
                archived = archive[name]
                if not archived.flags.c_contiguous:
                    raise FixtureError(f"payload array {name} is not C contiguous")
                arrays[name] = np.array(archived, copy=True, order="C")
    except (OSError, ValueError) as exc:
        if isinstance(exc, FixtureError):
            raise
        raise FixtureError(
            f"cannot read fixture payload {payload_path}: {exc}"
        ) from exc

    nodes = int(arrays["x"].shape[0]) if arrays["x"].ndim >= 1 else -1
    edge_index = arrays["edge_index"]
    edges = (
        int(edge_index.shape[1])
        if edge_index.ndim == 2 and edge_index.shape[0] == 2
        else -1
    )
    if nodes < 0 or edges < 0:
        raise FixtureError("x or edge_index has no valid leading extents")

    for name, value in arrays.items():
        metadata = array_metadata.get(name)
        if not isinstance(metadata, dict):
            raise FixtureError(f"arrays.{name} metadata must be an object")
        expected_metadata_keys = {"dtype", "shape", "order", "layout", "sha256"}
        if set(metadata) != expected_metadata_keys:
            raise FixtureError(
                f"arrays.{name} metadata must contain exactly "
                f"{sorted(expected_metadata_keys)}"
            )
        if metadata.get("order") != "C":
            raise FixtureError(f"arrays.{name}.order must be 'C'")
        if not value.flags.c_contiguous:
            raise FixtureError(f"payload array {name} is not C contiguous")
        expected_dtype = _expected_dtype(name)
        if value.dtype != expected_dtype:
            raise FixtureError(
                f"payload array {name} must use {expected_dtype}, got {value.dtype}"
            )
        if metadata.get("dtype") != value.dtype.str:
            raise FixtureError(
                f"arrays.{name}.dtype does not match payload dtype {value.dtype.str!r}"
            )
        if metadata.get("shape") != list(value.shape):
            raise FixtureError(
                f"arrays.{name}.shape does not match payload shape {list(value.shape)}"
            )
        if tuple(value.shape) not in _expected_shape(name, nodes, edges):
            raise FixtureError(
                f"payload array {name} has unsupported shape {value.shape}; "
                f"expected one of {_expected_shape(name, nodes, edges)}"
            )
        layout = metadata.get("layout")
        if layout != EXPECTED_LAYOUTS[name]:
            raise FixtureError(
                f"arrays.{name}.layout must be {EXPECTED_LAYOUTS[name]!r}, "
                f"got {layout!r}"
            )
        expected_hash = _strip_sha256_prefix(metadata.get("sha256"))
        actual_hash = _array_sha256(value)
        if actual_hash != expected_hash:
            raise FixtureError(
                f"array {name} SHA-256 mismatch: expected {expected_hash}, "
                f"computed {actual_hash}"
            )

    dimensions = manifest.get("dimensions")
    if not isinstance(dimensions, dict):
        raise FixtureError("fixture dimensions must be an object")
    scalar_dimensions = {
        "nodes": nodes,
        "edges": edges,
        "channels": CHANNELS,
        "source_components": INPUT_LM,
        "output_components": OUTPUT_LM,
        "radial_embedding": RADIAL_EMBEDDING,
        "paths": PATH_COUNT,
    }
    if set(dimensions) != set(scalar_dimensions):
        raise FixtureError(
            "fixture dimensions keys do not match the version-1 lane: "
            f"expected {sorted(scalar_dimensions)}, got {sorted(dimensions)}"
        )
    for name, expected in scalar_dimensions.items():
        if dimensions[name] != expected:
            raise FixtureError(
                f"dimensions.{name} must be {expected}, got {dimensions[name]!r}"
            )

    if edge_index.size:
        if edge_index.min() < 0 or edge_index.max() >= nodes:
            raise FixtureError("edge_index contains an out-of-range node index")
        receivers = edge_index[1]
        if np.any(receivers[1:] < receivers[:-1]):
            raise FixtureError("edge_index is not receiver-major as declared")
    xyz = arrays["xyz"]
    radii = np.linalg.norm(xyz.astype(np.float64), axis=1)
    if np.any(radii <= 0.0) or not np.all(np.isfinite(radii)):
        raise FixtureError("xyz contains zero-length or non-finite edge vectors")
    if not np.allclose(arrays["r"], radii, rtol=1e-12, atol=1e-12):
        raise FixtureError("r is inconsistent with the Euclidean norm of xyz")
    for name, value in arrays.items():
        if name != "edge_index" and not np.all(np.isfinite(value)):
            raise FixtureError(f"payload array {name} contains non-finite values")

    return LoadedFixture(
        manifest_path=manifest_path,
        payload_path=payload_path,
        manifest=manifest,
        arrays=arrays,
        manifest_sha256=actual_manifest_hash,
        payload_sha256=actual_payload_hash,
    )


def _quantile(ordered: list[float], quantile: float) -> float:
    index = (len(ordered) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _measure_once(callable_: Callable[[], Any], torch: Any, device: Any) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    start_time = time.perf_counter()
    callable_()
    return (time.perf_counter() - start_time) * 1000.0


def measure_callable(
    callable_: Callable[[], Any],
    *,
    torch: Any,
    device: Any,
    edge_count: int,
    config: MeasurementConfig,
) -> dict[str, Any]:
    """Apply the upstream runner's cold/warm/sample timing protocol to a callable."""

    config.validate()
    cold_start = time.perf_counter()
    callable_()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cold_ms = (time.perf_counter() - cold_start) * 1000.0

    warmup_start = time.perf_counter()
    warmup_count = 0
    while (
        warmup_count < config.warmup_iterations
        or (time.perf_counter() - warmup_start) * 1000.0 < config.warmup_ms
    ):
        callable_()
        warmup_count += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples: list[float] = []
    measured_ms = 0.0
    while len(samples) < config.max_samples and (
        len(samples) < config.min_samples or measured_ms < config.min_sample_ms
    ):
        sample = _measure_once(callable_, torch, device)
        samples.append(sample)
        measured_ms += sample

    ordered = sorted(samples)
    median_ms = statistics.median(samples)
    return {
        "cold_ms": cold_ms,
        "samples_ms": samples,
        "median_ms": median_ms,
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "p20_ms": _quantile(ordered, 0.2),
        "p80_ms": _quantile(ordered, 0.8),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "sample_count": len(samples),
        "warmup_count": warmup_count,
        "total_sample_ms": measured_ms,
        "ns_per_edge": median_ms * 1e6 / max(1, edge_count),
        "edges_per_second": edge_count / (median_ms / 1000.0),
        "timing_backend": "cuda_events",
        "execution_stream": "torch_current_cuda_stream",
    }


def _insert_upstream_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"upstream source root does not exist: {resolved}")
    sys.path.insert(0, str(resolved))
    return resolved


def _prepend_interpreter_bin_to_path() -> None:
    """Expose venv build tools to an upstream runtime extension compiler."""

    # Keep the literal executable path: resolving a venv Python symlink would
    # jump to the base interpreter and lose build tools installed in venv/bin.
    interpreter_bin = Path(sys.executable).absolute().parent
    ninja = interpreter_bin / "ninja"
    if not ninja.is_file() or not os.access(ninja, os.X_OK):
        return
    interpreter_bin_text = str(interpreter_bin)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if path_entries and path_entries[0] == interpreter_bin_text:
        return
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_bin_text]
        + [entry for entry in path_entries if entry != interpreter_bin_text]
    )


def _git_provenance(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"root": None, "commit": None, "dirty": None}

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "root": str(root),
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


class UpstreamR1Runner:
    """Prepared upstream execution of one validated fixture."""

    def __init__(
        self,
        fixture: LoadedFixture,
        *,
        torch: Any,
        device: Any,
        gemm: str,
        module_name: str,
        requires_grad: bool,
    ) -> None:
        try:
            upstream_nn = importlib.import_module(f"{module_name}.nn")
            TPConv = upstream_nn.TPConv
            TPConvGraph = upstream_nn.TPConvGraph
        except (AttributeError, ImportError) as error:
            raise RuntimeError(
                f"upstream module {module_name!r} does not expose nn.TPConv "
                "and nn.TPConvGraph"
            ) from error

        if device.type != "cuda":
            raise RuntimeError("the upstream TPConv comparison requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this PyTorch environment")
        self.fixture = fixture
        self.torch = torch
        self.device = device
        self.gemm = gemm
        arrays = fixture.arrays

        def tensor(name: str, dtype: Any) -> Any:
            return (
                torch.from_numpy(arrays[name])
                .to(device=device, dtype=dtype, non_blocking=False)
                .contiguous()
            )

        self.x_value = tensor("x", torch.float32).reshape(fixture.node_count, -1)
        self.edge_index = tensor("edge_index", torch.long)
        self.sh_value = tensor("sh", torch.float32)
        self.phi_value = tensor("phi", torch.float32)
        self.radial_linear = tensor("radial_linear", torch.float32).detach()
        self.grad_out = (
            tensor("grad_out", torch.float32).reshape(fixture.node_count, -1).detach()
        )
        self.xyz = tensor("xyz", torch.float64).detach()
        self.r = tensor("r", torch.float64).detach()
        self.dsh_dxyz = tensor("dsh_dxyz", torch.float32).detach()
        self.dphi_dr = tensor("dphi_dr", torch.float32).detach()
        self.unit_xyz = self.xyz / self.r[:, None]

        self.x = self.x_value.detach().requires_grad_(requires_grad)
        self.sh = self.sh_value.detach().requires_grad_(requires_grad)
        self.phi = self.phi_value.detach().requires_grad_(requires_grad)
        self.module = TPConv(
            INPUT_IRREPS,
            SH_IRREPS,
            OUTPUT_IRREPS,
            phi_dim=RADIAL_EMBEDDING,
            instructions=list(TP_INSTRUCTIONS),
            internal_weights=False,
            gemm=gemm,
            feature_layout="ir_mul",
        )
        self.graph = TPConvGraph(self.edge_index, num_nodes=fixture.node_count)
        if self.module.weight_numel != WEIGHT_NUMEL:
            raise RuntimeError(
                f"upstream TPConv weight_numel is {self.module.weight_numel}, "
                f"expected {WEIGHT_NUMEL}"
            )
        if self.radial_linear.shape != (RADIAL_EMBEDDING, WEIGHT_NUMEL):
            raise RuntimeError("radial_linear has an invalid runtime shape")
        if self.radial_linear.requires_grad:
            raise RuntimeError("fixed radial_linear unexpectedly requires gradients")
        if tuple(self.module.parameters()):
            raise RuntimeError("external-weight TPConv unexpectedly owns parameters")

        problem_paths = self.module.problem.paths
        expected_offsets = tuple(
            index * RADIAL_LINEAR_BLOCK_SIZE for index in range(PATH_COUNT)
        )
        actual_offsets = tuple(path.weight_offset for path in problem_paths)
        if actual_offsets != expected_offsets:
            raise RuntimeError(
                f"upstream TPConv path offsets {actual_offsets} do not match "
                f"the serialized radial order {expected_offsets}"
            )
        expected_normalization = tuple(
            math.sqrt(2 * i_out + 1) for _, _, i_out in PATH_TRIPLES
        )
        actual_normalization = tuple(path.path_weight for path in problem_paths)
        if any(
            not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in zip(actual_normalization, expected_normalization)
        ):
            raise RuntimeError(
                "upstream TPConv path normalization does not match the folded MACE contract"
            )
        self.path_offsets = actual_offsets
        self.path_normalization = actual_normalization
        self._reverse_out = None

    def forward(self) -> dict[str, Any]:
        # Preserve fixed-topology caching while avoiding autograd in inference mode.
        with self.torch.no_grad():
            out = self.module(
                self.x_value,
                self.graph,
                self.sh_value,
                self.phi_value,
                radial_linear=self.radial_linear,
            )
        return {"out": out}

    def _contract_directed_force(self, grad_sh: Any, grad_phi: Any) -> Any:
        angular_gradient = self.torch.einsum("el,ecl->ec", grad_sh, self.dsh_dxyz).to(
            dtype=self.torch.float64
        )
        radial_gradient = (
            self.torch.einsum("eq,eq->e", grad_phi, self.dphi_dr).to(
                dtype=self.torch.float64
            )[:, None]
            * self.unit_xyz
        )
        # Symmetrix stores the directed pair force, -dE/d(source-receiver).
        return -(angular_gradient + radial_gradient)

    def forward_backward(self) -> dict[str, Any]:
        if not (
            self.x.requires_grad and self.sh.requires_grad and self.phi.requires_grad
        ):
            raise RuntimeError(
                "forward_backward requires gradient-enabled x, sh, and phi"
            )
        out = self.module(
            self.x,
            self.graph,
            self.sh,
            self.phi,
            radial_linear=self.radial_linear,
        )
        grad_x, grad_sh, grad_phi = self.torch.autograd.grad(
            out,
            (self.x, self.sh, self.phi),
            self.grad_out,
        )
        directed_force = self._contract_directed_force(grad_sh, grad_phi)
        return {
            "out": out,
            "grad_x": grad_x,
            "grad_sh": grad_sh,
            "grad_phi": grad_phi,
            "directed_force": directed_force,
        }

    def prepare_reverse_diagnostic(self) -> None:
        if not (
            self.x.requires_grad and self.sh.requires_grad and self.phi.requires_grad
        ):
            raise RuntimeError("reverse diagnostic requires gradient-enabled inputs")
        self._reverse_out = self.module(
            self.x,
            self.graph,
            self.sh,
            self.phi,
            radial_linear=self.radial_linear,
        )

    def reverse_diagnostic(self) -> dict[str, Any]:
        if self._reverse_out is None:
            raise RuntimeError("prepare_reverse_diagnostic() must run first")
        grad_x, grad_sh, grad_phi = self.torch.autograd.grad(
            self._reverse_out,
            (self.x, self.sh, self.phi),
            self.grad_out,
            retain_graph=True,
        )
        directed_force = self._contract_directed_force(grad_sh, grad_phi)
        return {
            "grad_x": grad_x,
            "grad_sh": grad_sh,
            "grad_phi": grad_phi,
            "directed_force": directed_force,
        }


def _tensor_summary(value: Any) -> dict[str, Any]:
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": bool(detached.isfinite().all().item()),
        "max_abs": float(detached.abs().max().item()) if detached.numel() else 0.0,
        "l2": float(detached.float().norm().item()),
        "sum": float(detached.double().sum().item()),
    }


def _comparison(
    actual: Any,
    expected: np.ndarray,
    *,
    native_shape: tuple[int, ...] | None = None,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_shape = tuple(actual.shape)
    expected_shape = tuple(expected.shape)
    declared_native_shape = expected_shape if native_shape is None else native_shape
    if actual_shape != declared_native_shape:
        raise FixtureError(
            "upstream implementation output shape drifted before comparison: "
            f"actual {actual_shape}, expected native {declared_native_shape}"
        )
    expected_dtype = f"torch.{expected.dtype.name}"
    actual_dtype = str(actual.dtype)
    if actual_dtype != expected_dtype:
        raise FixtureError(
            "upstream implementation output dtype drifted before comparison: "
            f"actual {actual_dtype}, expected {expected_dtype}"
        )
    canonical_actual = actual.reshape(expected_shape)
    expected_tensor = canonical_actual.new_tensor(expected)
    difference = canonical_actual.detach() - expected_tensor
    absolute = difference.abs()
    denominator = expected_tensor.detach().abs().clamp_min(atol)
    relative = absolute / denominator
    close = canonical_actual.detach().isclose(expected_tensor, rtol=rtol, atol=atol)
    return {
        "shape": list(expected.shape),
        "dtype": expected.dtype.str,
        "allclose": bool(close.all().item()),
        "max_abs_error": float(absolute.max().item()) if absolute.numel() else 0.0,
        "max_rel_error": float(relative.max().item()) if relative.numel() else 0.0,
        "rms_error": float(difference.square().mean().sqrt().item())
        if difference.numel()
        else 0.0,
    }


def validate_outputs(
    outputs: dict[str, Any],
    fixture: LoadedFixture,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    expected_names = {
        "A1": ("out", "expected_out", True),
        "H1_adj": ("grad_x", "expected_grad_x", True),
        "edge_force": ("directed_force", "expected_directed_force", False),
    }
    comparisons = {}
    for boundary_name, (
        result_name,
        fixture_name,
        flattened_native,
    ) in expected_names.items():
        if result_name in outputs and fixture_name in fixture.arrays:
            expected = fixture.arrays[fixture_name]
            native_shape = tuple(expected.shape)
            if flattened_native:
                native_shape = (
                    expected.shape[0],
                    int(np.prod(expected.shape[1:], dtype=np.int64)),
                )
            comparisons[boundary_name] = {
                "reference_array": fixture_name,
                **_comparison(
                    outputs[result_name],
                    expected,
                    native_shape=native_shape,
                    rtol=rtol,
                    atol=atol,
                ),
            }
    passed = bool(comparisons) and all(
        result["allclose"] for result in comparisons.values()
    )
    return {
        "status": "pass" if passed else "fail",
        "rtol": rtol,
        "atol": atol,
        "outputs": comparisons,
    }


def _environment_provenance(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_home": os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
    }


def _write_tensor_output(path: Path, outputs: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {name: value.detach().cpu().numpy() for name, value in outputs.items()}
    np.savez(path, **values)


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="versioned fixture JSON or NPZ")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=(
            Path(os.environ["SYMMETRIX_UPSTREAM_R1_ROOT"])
            if "SYMMETRIX_UPSTREAM_R1_ROOT" in os.environ
            else None
        ),
        help="upstream source root (or set SYMMETRIX_UPSTREAM_R1_ROOT)",
    )
    parser.add_argument(
        "--upstream-module",
        default=os.environ.get("SYMMETRIX_UPSTREAM_R1_MODULE"),
        help=(
            "Python package exposing nn.TPConv and nn.TPConvGraph "
            "(or set SYMMETRIX_UPSTREAM_R1_MODULE)"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "reverse", "forward_backward"),
        default="forward_backward",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gemm", choices=("ieee", "tf32"), default="ieee")
    parser.add_argument("--output", type=Path, help="also write the report JSON")
    parser.add_argument(
        "--tensor-output",
        type=Path,
        help="write the last output/gradients/directed force as NPZ",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run and validate one CUDA iteration, but skip benchmark timing",
    )
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="validate only the fixture contract and hashes; upstream/CUDA are not loaded",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="compile once, then execute exactly one NVTX-marked iteration",
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--warmup-ms", type=float, default=100.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--min-sample-ms", type=float, default=1000.0)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=3e-5)
    args = parser.parse_args(argv)
    if args.rtol < 0.0 or args.atol < 0.0:
        parser.error("--rtol and --atol must be non-negative")
    selected_single_modes = sum((args.validate_only, args.fixture_only, args.nvtx))
    if selected_single_modes > 1:
        parser.error(
            "--validate-only, --fixture-only, and --nvtx are mutually exclusive"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture = load_fixture(args.fixture)
    if args.fixture_only:
        provenance = fixture.manifest["provenance"]
        report = {
            "schema": REPORT_SCHEMA,
            "version": REPORT_VERSION,
            "backend": "execution",
            "status": "pass",
            "lane": FIXTURE_LANE,
            "mode": "fixture_only",
            "contract": EXPECTED_CONTRACT,
            "conventions": EXPECTED_CONVENTIONS,
            "fixture": {
                "manifest": str(fixture.manifest_path),
                "payload": str(fixture.payload_path),
                "manifest_sha256": f"sha256:{fixture.manifest_sha256}",
                "payload_sha256": f"sha256:{fixture.payload_sha256}",
                "model_sha256": provenance["model_sha256"],
                "graph_sha256": provenance["graph_sha256"],
                "symmetrix_commit": provenance["symmetrix_commit"],
                "graph_generation": provenance["graph_generation"],
                "nodes": fixture.node_count,
                "edges": fixture.edge_count,
            },
            "measurement": None,
            "timing": {
                "skipped": True,
                "reason": "fixture-only",
            },
            "validation": {
                "status": "pass",
                "rtol": args.rtol,
                "atol": args.atol,
                "outputs": {},
            },
        }
        if args.output is not None:
            _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    _prepend_interpreter_bin_to_path()
    upstream_root = _insert_upstream_root(args.upstream_root)
    if not args.upstream_module:
        raise RuntimeError(
            "--upstream-module or SYMMETRIX_UPSTREAM_R1_MODULE is required"
        )
    import torch

    device = torch.device(args.device)
    requires_grad = args.mode != "forward"
    runner = UpstreamR1Runner(
        fixture,
        torch=torch,
        device=device,
        gemm=args.gemm,
        module_name=args.upstream_module,
        requires_grad=requires_grad,
    )
    if args.mode == "forward":
        callable_ = runner.forward
    elif args.mode == "reverse":
        runner.prepare_reverse_diagnostic()
        torch.cuda.synchronize(device)
        callable_ = runner.reverse_diagnostic
    else:
        callable_ = runner.forward_backward
    timing_semantics = TIMING_SEMANTICS[args.mode]

    measurement = MeasurementConfig(
        warmup_iterations=args.warmup_iterations,
        warmup_ms=args.warmup_ms,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        min_sample_ms=args.min_sample_ms,
    )
    measurement.validate()
    timing = None
    nvtx = None
    if args.validate_only:
        outputs = callable_()
        torch.cuda.synchronize(device)
        timing = {
            "skipped": True,
            "reason": "validate-only",
        }
    elif args.nvtx:
        callable_()
        torch.cuda.synchronize(device)
        range_name = f"upstream_r1::{args.mode}"
        torch.cuda.nvtx.range_push(range_name)
        try:
            outputs = callable_()
            torch.cuda.synchronize(device)
        finally:
            torch.cuda.nvtx.range_pop()
        nvtx = {"range": range_name, "iterations": 1, "compile_warmup_iterations": 1}
        timing = {
            "skipped": True,
            "reason": "nvtx-single-iteration",
        }
    else:
        timing = measure_callable(
            callable_,
            torch=torch,
            device=device,
            edge_count=fixture.edge_count,
            config=measurement,
        )
        outputs = callable_()
        torch.cuda.synchronize(device)

    validation = validate_outputs(
        outputs,
        fixture,
        rtol=args.rtol,
        atol=args.atol,
    )
    if args.tensor_output is not None:
        _write_tensor_output(args.tensor_output, outputs)

    provenance = fixture.manifest["provenance"]
    report = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "backend": "execution",
        "status": validation["status"],
        "lane": FIXTURE_LANE,
        "contract": EXPECTED_CONTRACT,
        "conventions": EXPECTED_CONVENTIONS,
        "mode": args.mode,
        "headline": args.mode == "forward_backward",
        "timing_semantics": timing_semantics,
        "timing_protocol": TIMING_PROTOCOL,
        "fixture": {
            "manifest": str(fixture.manifest_path),
            "payload": str(fixture.payload_path),
            "manifest_sha256": f"sha256:{fixture.manifest_sha256}",
            "payload_sha256": f"sha256:{fixture.payload_sha256}",
            "schema": fixture.manifest["schema"],
            "version": fixture.manifest["version"],
            "lane": fixture.manifest["lane"],
            "model_sha256": provenance["model_sha256"],
            "graph_sha256": provenance["graph_sha256"],
            "symmetrix_commit": provenance["symmetrix_commit"],
            "graph_generation": provenance["graph_generation"],
            "nodes": fixture.node_count,
            "edges": fixture.edge_count,
            "provenance": provenance,
        },
        "execution": {
            "input_irreps": INPUT_IRREPS,
            "sh_irreps": SH_IRREPS,
            "output_irreps": OUTPUT_IRREPS,
            "connection_mode": "uvw",
            "feature_layout": "ir_mul",
            "radial_embedding": RADIAL_EMBEDDING,
            "weight_numel": WEIGHT_NUMEL,
            "path_offsets": list(runner.path_offsets),
            "path_normalization": list(runner.path_normalization),
            "instructions": [list(value) for value in TP_INSTRUCTIONS],
            "codegen_config": {
                **asdict(runner.module.codegen_config),
                "gemm": args.gemm,
            },
            "gemm": args.gemm,
            "radial_linear_requires_grad": runner.radial_linear.requires_grad,
            "requested_gradients": ["x", "sh", "phi"] if requires_grad else [],
            "parameter_gradients": False,
            "directed_force_sign": "-dE/d(source_minus_receiver)",
        },
        "measurement": asdict(measurement),
        "timing": timing,
        "nvtx": nvtx,
        "validation": validation,
        "outputs": {name: _tensor_summary(value) for name, value in outputs.items()},
        "environment": _environment_provenance(torch, device),
        "upstream": {
            **_git_provenance(upstream_root),
            "module": args.upstream_module,
        },
        "tensor_output": str(args.tensor_output.resolve())
        if args.tensor_output is not None
        else None,
    }
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
