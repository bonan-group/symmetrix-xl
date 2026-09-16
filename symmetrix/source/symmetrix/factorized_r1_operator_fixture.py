"""External fixture format for the matched EXECUTION R1 operator benchmark.

Graph-sized arrays deliberately live in a sidecar NPZ file.  The JSON manifest
fixes the semantic boundary and hashes both the complete payload and every
individual array before either implementation is allowed to run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SCHEMA = "symmetrix.execution.r1-operator-benchmark"
SCHEMA_VERSION = 1
LANE = "extracted_mace_exact_uvw"

CHANNELS = 128
RADIAL_EMBEDDING = 64
SOURCE_COMPONENTS = 4
OUTPUT_COMPONENTS = 16
PATH_COUNT = 10
RADIAL_LINEAR_BLOCK_SIZE = CHANNELS * CHANNELS
RADIAL_LINEAR_WIDTH = PATH_COUNT * RADIAL_LINEAR_BLOCK_SIZE

RTOL = 2.0e-5
ATOL = 3.0e-5

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EXPECTED_ARRAYS = frozenset(
    ("expected_out", "expected_grad_x", "expected_directed_force")
)


class FixtureValidationError(ValueError):
    """Raised when a fixture does not describe the frozen operator boundary."""


@dataclass(frozen=True)
class ExecutionR1OperatorFixture:
    """Validated manifest and in-memory arrays for one benchmark case."""

    manifest_path: pathlib.Path
    payload_path: pathlib.Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def num_nodes(self) -> int:
        return int(self.manifest["dimensions"]["nodes"])

    @property
    def num_edges(self) -> int:
        return int(self.manifest["dimensions"]["edges"])


def canonical_contract() -> dict[str, Any]:
    """Return the exact folded-UVW contract used for the matched lane."""

    path_triples = (
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
    raw_path_weights = (
        256.0,
        256.0,
        384.0,
        384.0,
        384.0,
        384.0,
        384.0,
        384.0,
        256.0,
        256.0,
    )
    return {
        "input_irreps": "128x0e+128x1o",
        "sh_irreps": "1x0e+1x1o+1x2e+1x3o",
        "output_irreps": "128x0e+128x1o+128x2e+128x3o",
        "connection_mode": "uvw",
        "radial_embedding": RADIAL_EMBEDDING,
        "instructions": [
            {
                "index": index,
                "i_in": i_in,
                "i_sh": i_sh,
                "i_out": i_out,
                "raw_path_weight": raw_path_weights[index],
            }
            for index, (i_in, i_sh, i_out) in enumerate(path_triples)
        ],
        "radial_linear_layout": (
            "q,path,input_channel,output_channel; trailing dimensions C-flat"
        ),
        "radial_linear_block_size": RADIAL_LINEAR_BLOCK_SIZE,
        "parameter_gradients": False,
        "density_scaling": False,
    }


def canonical_conventions() -> dict[str, str]:
    """Return graph, layout, derivative, and sign conventions."""

    return {
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


def canonical_array_layouts() -> dict[str, str]:
    """Return exact semantic layout strings for all admitted arrays."""

    return {
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


def _array_specs(
    num_nodes: int, num_edges: int
) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    f32 = np.dtype(np.float32)
    i64 = np.dtype(np.int64)
    return {
        "x": (f32, (num_nodes, SOURCE_COMPONENTS, CHANNELS)),
        "edge_index": (i64, (2, num_edges)),
        "sh": (f32, (num_edges, OUTPUT_COMPONENTS)),
        "phi": (f32, (num_edges, RADIAL_EMBEDDING)),
        "radial_linear": (f32, (RADIAL_EMBEDDING, RADIAL_LINEAR_WIDTH)),
        "grad_out": (f32, (num_nodes, OUTPUT_COMPONENTS, CHANNELS)),
        "xyz": (np.dtype(np.float64), (num_edges, 3)),
        "r": (np.dtype(np.float64), (num_edges,)),
        "dsh_dxyz": (f32, (num_edges, 3, OUTPUT_COMPONENTS)),
        "dphi_dr": (f32, (num_edges, RADIAL_EMBEDDING)),
        "expected_out": (f32, (num_nodes, OUTPUT_COMPONENTS, CHANNELS)),
        "expected_grad_x": (f32, (num_nodes, SOURCE_COMPONENTS, CHANNELS)),
        "expected_directed_force": (np.dtype(np.float64), (num_edges, 3)),
    }


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize JSON canonically for manifest identity hashing."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256_bytes(array.tobytes(order="C"))


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest with the ``manifest_sha256`` member omitted."""

    unhashed = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    return sha256_bytes(canonical_json_bytes(unhashed))


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FixtureValidationError(
            f"{field} must have the form 'sha256:' followed by 64 lowercase hex digits"
        )


def _infer_dimensions(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    if "x" not in arrays or "edge_index" not in arrays:
        raise FixtureValidationError("arrays must include x and edge_index")
    x = np.asarray(arrays["x"])
    edge_index = np.asarray(arrays["edge_index"])
    if x.ndim != 3:
        raise FixtureValidationError("x must have rank 3")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise FixtureValidationError("edge_index must have shape (2, E)")
    return int(x.shape[0]), int(edge_index.shape[1])


def validate_arrays(
    arrays: Mapping[str, np.ndarray], *, require_expected: bool = False
) -> tuple[int, int]:
    """Validate names, exact dtypes/shapes, graph order, and finite values."""

    num_nodes, num_edges = _infer_dimensions(arrays)
    if num_nodes < 1:
        raise FixtureValidationError("the fixture must contain at least one node")
    specs = _array_specs(num_nodes, num_edges)
    required = set(specs) - _EXPECTED_ARRAYS
    names = set(arrays)
    missing = required - names
    unknown = names - set(specs)
    if missing:
        raise FixtureValidationError(
            "missing required arrays: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise FixtureValidationError("unknown arrays: " + ", ".join(sorted(unknown)))
    if require_expected and not _EXPECTED_ARRAYS.issubset(names):
        missing_expected = _EXPECTED_ARRAYS - names
        raise FixtureValidationError(
            "comparison requires expected arrays: "
            + ", ".join(sorted(missing_expected))
        )

    for name in sorted(names):
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise FixtureValidationError(f"array {name!r} must be a numpy.ndarray")
        expected_dtype, expected_shape = specs[name]
        if value.dtype != expected_dtype:
            raise FixtureValidationError(
                f"array {name!r} has dtype {value.dtype.str}; "
                f"expected {expected_dtype.str}"
            )
        if value.shape != expected_shape:
            raise FixtureValidationError(
                f"array {name!r} has shape {value.shape}; expected {expected_shape}"
            )
        if not value.flags.c_contiguous:
            raise FixtureValidationError(f"array {name!r} must be C-contiguous")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise FixtureValidationError(f"array {name!r} contains non-finite values")

    edge_index = arrays["edge_index"]
    if num_edges:
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise FixtureValidationError(
                "edge_index contains an out-of-range node index"
            )
        receivers = edge_index[1]
        if np.any(receivers[1:] < receivers[:-1]):
            raise FixtureValidationError(
                "edge_index must be receiver-major with nondecreasing receivers"
            )
        radii = np.linalg.norm(arrays["xyz"], axis=1)
        if np.any(radii <= 0.0):
            raise FixtureValidationError("xyz contains a zero-length edge vector")
        if not np.allclose(arrays["r"], radii, rtol=1.0e-12, atol=1.0e-12):
            raise FixtureValidationError(
                "r is inconsistent with the float64 norm of xyz"
            )
    return num_nodes, num_edges


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise FixtureValidationError("provenance must be an object")
    result = copy.deepcopy(dict(provenance))
    required = {
        "producer",
        "symmetrix_commit",
        "model_sha256",
        "graph_sha256",
        "graph_generation",
    }
    missing = required - set(result)
    if missing:
        raise FixtureValidationError(
            "missing provenance fields: " + ", ".join(sorted(missing))
        )
    for field in ("producer", "symmetrix_commit"):
        if not isinstance(result[field], str) or not result[field]:
            raise FixtureValidationError(
                f"provenance.{field} must be a nonempty string"
            )
    for field in ("model_sha256", "graph_sha256"):
        _require_sha256(result[field], f"provenance.{field}")
    if (
        not isinstance(result["graph_generation"], int)
        or isinstance(result["graph_generation"], bool)
        or result["graph_generation"] < 1
    ):
        raise FixtureValidationError(
            "provenance.graph_generation must be a positive integer"
        )
    try:
        canonical_json_bytes(result)
    except (TypeError, ValueError) as exc:
        raise FixtureValidationError(
            "provenance must contain canonical JSON values"
        ) from exc
    return result


def build_manifest(
    arrays: Mapping[str, np.ndarray],
    *,
    payload_filename: str,
    payload_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and hash a manifest for already-written NPZ bytes."""

    num_nodes, num_edges = validate_arrays(arrays, require_expected=True)
    payload_name = pathlib.PurePosixPath(payload_filename)
    if (
        payload_name.is_absolute()
        or len(payload_name.parts) != 1
        or payload_name.name != payload_filename
    ):
        raise FixtureValidationError("payload path must be a same-directory basename")
    _require_sha256(payload_sha256, "payload.sha256")
    layouts = canonical_array_layouts()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "lane": LANE,
        "payload": {
            "path": payload_filename,
            "format": "npz",
            "sha256": payload_sha256,
        },
        "dimensions": {
            "nodes": num_nodes,
            "edges": num_edges,
            "channels": CHANNELS,
            "source_components": SOURCE_COMPONENTS,
            "output_components": OUTPUT_COMPONENTS,
            "radial_embedding": RADIAL_EMBEDDING,
            "paths": PATH_COUNT,
        },
        "contract": canonical_contract(),
        "conventions": canonical_conventions(),
        "arrays": {
            name: {
                "dtype": arrays[name].dtype.str,
                "shape": list(arrays[name].shape),
                "order": "C",
                "layout": layouts[name],
                "sha256": sha256_array(arrays[name]),
            }
            for name in sorted(arrays)
        },
        "provenance": _validate_provenance(provenance),
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise FixtureValidationError("manifest must be a JSON object")
    required_keys = {
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
    if set(manifest) != required_keys:
        missing = required_keys - set(manifest)
        unknown = set(manifest) - required_keys
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        raise FixtureValidationError("invalid manifest keys: " + "; ".join(details))
    if manifest["schema"] != SCHEMA or manifest["version"] != SCHEMA_VERSION:
        raise FixtureValidationError(
            f"unsupported fixture schema/version: {manifest['schema']!r} "
            f"version {manifest['version']!r}"
        )
    if manifest["lane"] != LANE:
        raise FixtureValidationError(
            f"semantic lane mismatch: expected {LANE!r}, got {manifest['lane']!r}"
        )
    if manifest["contract"] != canonical_contract():
        raise FixtureValidationError(
            "operator contract does not match the exact UVW lane"
        )
    if manifest["conventions"] != canonical_conventions():
        raise FixtureValidationError("graph/layout/sign conventions do not match")
    _require_sha256(manifest["manifest_sha256"], "manifest_sha256")
    if manifest["manifest_sha256"] != manifest_sha256(manifest):
        raise FixtureValidationError("manifest SHA-256 mismatch")

    payload = manifest["payload"]
    if not isinstance(payload, Mapping) or set(payload) != {"path", "format", "sha256"}:
        raise FixtureValidationError(
            "payload must contain exactly path, format, and sha256"
        )
    if payload["format"] != "npz":
        raise FixtureValidationError("payload format must be 'npz'")
    payload_name = payload["path"]
    if not isinstance(payload_name, str):
        raise FixtureValidationError("payload.path must be a string")
    pure_name = pathlib.PurePosixPath(payload_name)
    if (
        pure_name.is_absolute()
        or len(pure_name.parts) != 1
        or pure_name.name != payload_name
    ):
        raise FixtureValidationError("payload path must be a same-directory basename")
    _require_sha256(payload["sha256"], "payload.sha256")

    dimensions = manifest["dimensions"]
    expected_dimension_keys = {
        "nodes",
        "edges",
        "channels",
        "source_components",
        "output_components",
        "radial_embedding",
        "paths",
    }
    if (
        not isinstance(dimensions, Mapping)
        or set(dimensions) != expected_dimension_keys
    ):
        raise FixtureValidationError("dimensions object has unexpected keys")
    if any(
        not isinstance(dimensions[key], int) or isinstance(dimensions[key], bool)
        for key in dimensions
    ):
        raise FixtureValidationError("all dimensions must be integers")
    fixed_dimensions = {
        "channels": CHANNELS,
        "source_components": SOURCE_COMPONENTS,
        "output_components": OUTPUT_COMPONENTS,
        "radial_embedding": RADIAL_EMBEDDING,
        "paths": PATH_COUNT,
    }
    for key, value in fixed_dimensions.items():
        if dimensions[key] != value:
            raise FixtureValidationError(
                f"dimension {key!r} must be {value}, got {dimensions[key]}"
            )
    if dimensions["nodes"] < 1 or dimensions["edges"] < 0:
        raise FixtureValidationError("nodes must be positive and edges nonnegative")
    _validate_provenance(manifest["provenance"])


def validate_manifest(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None = None
) -> None:
    """Validate the semantic envelope and, when supplied, its arrays."""

    _validate_manifest_structure(manifest)
    if arrays is None:
        return
    num_nodes, num_edges = validate_arrays(arrays, require_expected=True)
    dimensions = manifest["dimensions"]
    if num_nodes != dimensions["nodes"] or num_edges != dimensions["edges"]:
        raise FixtureValidationError("array dimensions do not match the manifest")
    entries = manifest["arrays"]
    if not isinstance(entries, Mapping) or set(entries) != set(arrays):
        raise FixtureValidationError(
            "manifest array names do not match the NPZ payload"
        )
    layouts = canonical_array_layouts()
    for name, array in arrays.items():
        entry = entries[name]
        expected = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "order": "C",
            "layout": layouts[name],
            "sha256": sha256_array(array),
        }
        if entry != expected:
            raise FixtureValidationError(
                f"manifest metadata or SHA-256 mismatch for array {name!r}"
            )


def write_fixture(
    manifest_path: pathlib.Path | str,
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> ExecutionR1OperatorFixture:
    """Write an NPZ sidecar and canonical JSON manifest, then reload them."""

    manifest_path = pathlib.Path(manifest_path).resolve()
    if manifest_path.suffix.lower() != ".json":
        raise FixtureValidationError("fixture manifest path must end in .json")
    payload_path = manifest_path.with_suffix(".npz")
    if not overwrite and (manifest_path.exists() or payload_path.exists()):
        raise FileExistsError(
            f"fixture output already exists: {manifest_path} or {payload_path}"
        )
    validate_arrays(arrays, require_expected=True)
    frozen_arrays = {
        name: np.array(value, copy=True, order="C")
        for name, value in sorted(arrays.items())
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload_temp: pathlib.Path | None = None
    manifest_temp: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=payload_path.name + ".",
            suffix=".tmp",
            dir=payload_path.parent,
            delete=False,
        ) as handle:
            payload_temp = pathlib.Path(handle.name)
            np.savez(handle, **frozen_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        payload_hash = sha256_file(payload_temp)
        manifest = build_manifest(
            frozen_arrays,
            payload_filename=payload_path.name,
            payload_sha256=payload_hash,
            provenance=provenance,
        )
        rendered = (
            json.dumps(
                manifest, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
            )
            + "\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            prefix=manifest_path.name + ".",
            suffix=".tmp",
            dir=manifest_path.parent,
            delete=False,
        ) as handle:
            manifest_temp = pathlib.Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(payload_temp, payload_path)
        payload_temp = None
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
    finally:
        for temporary in (payload_temp, manifest_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return load_fixture(manifest_path)


def load_fixture(
    manifest_path: pathlib.Path | str, *, require_expected: bool = True
) -> ExecutionR1OperatorFixture:
    """Load a fixture only after all semantic and byte hashes validate."""

    manifest_path = pathlib.Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureValidationError(
            f"failed to read fixture manifest {manifest_path}"
        ) from exc
    _validate_manifest_structure(manifest)
    payload_path = manifest_path.parent / manifest["payload"]["path"]
    try:
        payload_hash = sha256_file(payload_path)
    except OSError as exc:
        raise FixtureValidationError(
            f"failed to read NPZ payload {payload_path}"
        ) from exc
    if payload_hash != manifest["payload"]["sha256"]:
        raise FixtureValidationError("NPZ payload SHA-256 mismatch")
    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            arrays = {
                name: np.array(payload[name], copy=True, order="C")
                for name in payload.files
            }
    except (OSError, ValueError, KeyError) as exc:
        raise FixtureValidationError(
            f"failed to load NPZ payload {payload_path}"
        ) from exc
    validate_arrays(arrays, require_expected=require_expected)
    validate_manifest(manifest, arrays)
    return ExecutionR1OperatorFixture(
        manifest_path=manifest_path,
        payload_path=payload_path,
        manifest=manifest,
        arrays=arrays,
    )


def graph_sha256(edge_index: np.ndarray) -> str:
    """Hash the graph topology with dtype and shape domain separation."""

    edge_index = np.ascontiguousarray(edge_index)
    digest = hashlib.sha256()
    digest.update(edge_index.dtype.str.encode("ascii"))
    digest.update(np.asarray(edge_index.shape, dtype=np.int64).tobytes())
    digest.update(edge_index.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


__all__ = [
    "ATOL",
    "CHANNELS",
    "FixtureValidationError",
    "LANE",
    "OUTPUT_COMPONENTS",
    "PATH_COUNT",
    "RADIAL_EMBEDDING",
    "RTOL",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SOURCE_COMPONENTS",
    "ExecutionR1OperatorFixture",
    "build_manifest",
    "canonical_array_layouts",
    "canonical_contract",
    "canonical_conventions",
    "canonical_json_bytes",
    "graph_sha256",
    "load_fixture",
    "manifest_sha256",
    "sha256_array",
    "sha256_file",
    "validate_arrays",
    "validate_manifest",
    "write_fixture",
]
