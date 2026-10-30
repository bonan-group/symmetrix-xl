"""Declare, run, and validate the NequIP-OAM release qualification matrix.

The orchestrator is intentionally evidence preserving.  It can declare the
complete matrix without running it, resumes only records that pass strict
validation, and represents missing models, hardware, records, and baselines as
machine-readable unqualified results.  It never substitutes estimated timing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from benchmarks.oam_release_contract import (
        OAM_HOST_RUNTIME_PAIRS,
        OAM_RELEASE_TOLERANCES,
    )
except ModuleNotFoundError as error:
    if error.name != "benchmarks":
        raise
    from oam_release_contract import OAM_HOST_RUNTIME_PAIRS, OAM_RELEASE_TOLERANCES

MODEL_SIZES = ("S", "M", "L", "XL")
BACKENDS = ("host", "cuda")
PRECISIONS = ("float32", "float64")
HOST_GRAPH_CLASS = "host-100"
CUDA_GRAPH_CLASSES = ("small", "production")
GRAPH_CLASSES = (HOST_GRAPH_CLASS, *CUDA_GRAPH_CLASSES)
HOST_RUNTIME_PAIRS = OAM_HOST_RUNTIME_PAIRS
HOST_THREADS = (1, 4)
GRAPH_COMMON_FIELDS = frozenset(("structure", "seed", "rattle", "skin"))
REQUIRED_BINDING_FIELDS = (
    "operator_id",
    "capability_key",
    "implementation_id",
    "selected_implementation_id",
    "tier",
    "qualified",
    "portable_covered",
    "artifact_required",
)

CELL_SCHEMA = "symmetrix.oam-release-qualification-cell"
REQUEST_SCHEMA = "symmetrix.oam-release-qualification-request"
MANIFEST_SCHEMA = "symmetrix.oam-release-qualification-manifest"
RESULT_SCHEMA = "symmetrix.oam-release-qualification-result"
SCHEMA_VERSION = 3

CORRECTNESS_TOLERANCES = OAM_RELEASE_TOLERANCES

REQUEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "cell_id",
        "kind",
        "model_size",
        "backend",
        "precision",
        "graph_class",
        "profile",
        "threads",
        "model_exists",
        "model_path",
        "model_sha256",
        "python_executable",
        "python_real_executable",
        "driver_path",
        "package_path",
        "output_root",
        "cell_record_path",
        "direct_record_path",
        "reference_record_path",
        "direct_numerical_path",
        "reference_numerical_path",
        "graph_config",
        "graph_input_sha256",
        "host_target",
        "host_floating_point_policy",
        "cuda_target",
        "routes",
        "baseline_id",
        "baseline_sha256",
        "baseline_us_per_atom",
        "source_root",
        "source_exists",
        "source_fingerprint",
        "source_commit",
        "source_dirty",
        "native_path",
        "native_exists",
        "native_sha256",
        "required_bindings",
        "required_reference_bindings",
        "binding_report_version",
        "samples",
        "warmups",
        "edge_variant",
        "generation_policy_request",
        "artifact_evidence_contract",
        "request_fingerprint",
        "jit_cache_path",
    }
)

TIMED_REGION = "compute_prepared+reduce_forces+reduce_stress"
TIMING_EXCLUSIONS = (
    "artifact_compilation",
    "model_preparation",
    "graph_construction",
    "neighbor_construction",
    "warmup",
)


class QualificationError(ValueError):
    """A retained cell is incomplete or does not describe its request."""


def _normalized_cuda_target(value: Any) -> str:
    from symmetrix.cuda_codegen_common import canonical_cuda_target_identity

    try:
        return canonical_cuda_target_identity(value)
    except ValueError as error:
        raise QualificationError("actual CUDA device target is invalid") from error


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{name} must be an object")
    return value


def _require(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for component in path.split("."):
        current = _require_mapping(current, path.rsplit(".", 1)[0] or "record")
        if component not in current:
            raise QualificationError(f"missing required field {path}")
        current = current[component]
    if current is None or current == "":
        raise QualificationError(f"required field {path} is empty")
    return current


def _require_digest(record: Mapping[str, Any], path: str, length: int = 64) -> str:
    value = _require(record, path)
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None
    ):
        raise QualificationError(
            f"{path} must be a hexadecimal {length}-character digest"
        )
    return value


def _finite_number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{path} must be numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise QualificationError(f"{path} must be finite") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise QualificationError(
            f"{path} must be finite" + (" and positive" if positive else "")
        )
    return result


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise QualificationError(f"{path} must be at least {minimum}")
    return value


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's boolean-to-number coercion."""

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return len(left) == len(right) and all(
            key in right and _json_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(
                _json_values_equal(left_value, right_value)
                for left_value, right_value in zip(left, right)
            )
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return (
            isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and left == right
        )
    return type(left) is type(right) and left == right


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _canonical_generation_policy(
    backend: str, precision: str, cuda_target: str | None
) -> dict[str, Any]:
    if backend == "host":
        return {
            "owner_schedule": "whole_owner",
            "channel_tile": 1,
            "scalar_width": 4 if precision == "float32" else 8,
            "vector_policy": "host-simd-lane-major-v1",
            "required_isa": "host-target-bound-v1",
            "full_reverse_reduction_policy": "owner-local-deterministic-v1",
            "resource_policy": "compiler-managed-v1",
        }
    radial_mode = (
        "fp32"
        if precision == "float32" and str(cuda_target).lower() in {"sm120", "sm_120"}
        else "fp64"
    )
    qualified_sm120_fp32 = precision == "float32" and str(cuda_target).lower() in {
        "sm120",
        "sm_120",
    }
    return {
        "forward_live_output_budget": 32 if qualified_sm120_fp32 else None,
        "forward_grouping_mode": (
            "source_first_fit" if qualified_sm120_fp32 else "output_order"
        ),
        "forward_group_execution": ("warp_tasks" if qualified_sm120_fp32 else "serial"),
        "forward_input_mode": "staged" if qualified_sm120_fp32 else "inline",
        "radial_adjoint_mode": radial_mode,
        "path_reuse_mode": "none",
        "reverse_group_execution": "serial",
    }


def _current_package_path() -> str:
    spec = importlib.util.find_spec("symmetrix")
    if spec is None or spec.origin is None:
        return ""
    return str(Path(spec.origin).resolve())


def _probe_current_installed_artifact(request: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the current installed artifact in a fresh exact-native process."""

    script = r"""
import hashlib, importlib.abc, importlib.machinery, importlib.util, json, os, stat, sys
from pathlib import Path
request = json.loads(sys.argv[1])
# Qualification runs in a disposable process, so constrain imports to Python's
# standard finders before binding the complete package namespace. Editable
# install finders otherwise can redirect absolute symmetrix.* imports to source.
sys.meta_path[:] = [
    importlib.machinery.BuiltinImporter,
    importlib.machinery.FrozenImporter,
    importlib.machinery.PathFinder,
]
for loaded_name in tuple(sys.modules):
    if loaded_name == "symmetrix" or loaded_name.startswith("symmetrix."):
        del sys.modules[loaded_name]

def require_directory(path, label):
    path = Path(path)
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be absolute")
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} must be a non-symlinked directory")
    return path.resolve(strict=True)

def require_file_within(path, root, label):
    path = Path(path)
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} is outside the bound package root") from error
    current = root
    for part in relative.parts:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} contains a symlink")
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError(f"{label} must be a regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"{label} resolves outside the bound package root")
    return resolved

package_path = Path(request["package_path"])
package_dir = require_directory(package_path.parent, "bound package root")
package_path = require_file_within(package_path, package_dir, "bound package origin")
extension_path = Path(request["native_path"])
native_package_dir = require_directory(extension_path.parent, "native package root")
extension_path = require_file_within(
    extension_path, native_package_dir, "native extension origin"
)

loader_details = (
    (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES),
    (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES),
    (importlib.machinery.SourcelessFileLoader, importlib.machinery.BYTECODE_SUFFIXES),
)
standard_loader_types = tuple(item[0] for item in loader_details)

def validate_symmetrix_spec(fullname, spec):
    if spec is None or spec.loader is None or spec.origin in {None, "namespace"}:
        raise RuntimeError(f"{fullname} has an invalid namespace or loader")
    if not isinstance(spec.loader, standard_loader_types):
        raise RuntimeError(f"{fullname} has a nonstandard loader")
    origin = require_file_within(spec.origin, package_dir, f"{fullname} origin")
    loader_origin = Path(spec.loader.get_filename(fullname)).resolve(strict=True)
    if loader_origin != origin:
        raise RuntimeError(f"{fullname} loader origin does not match its bound origin")
    locations = spec.submodule_search_locations
    if locations is not None:
        for location in locations:
            resolved = require_directory(location, f"{fullname} namespace path")
            if not resolved.is_relative_to(package_dir):
                raise RuntimeError(f"{fullname} namespace path is outside the bound package root")
    return spec

class BoundSymmetrixFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("symmetrix."):
            return None
        search = [package_dir] if path is None else [Path(item) for item in path]
        for entry in search:
            resolved = require_directory(entry, f"{fullname} namespace path")
            if not resolved.is_relative_to(package_dir):
                raise RuntimeError(
                    f"{fullname} namespace path is outside the bound package root"
                )
            spec = importlib.machinery.FileFinder(
                str(resolved), *loader_details
            ).find_spec(fullname, target)
            if spec is not None:
                return validate_symmetrix_spec(fullname, spec)
        raise ModuleNotFoundError(f"No module named {fullname!r} in bound package")

sys.meta_path.insert(0, BoundSymmetrixFinder())
package_spec = importlib.util.spec_from_file_location(
    "symmetrix", package_path, submodule_search_locations=[str(package_dir)]
)
if package_spec is None or package_spec.loader is None:
    raise RuntimeError("cannot load requested Symmetrix package")
package = importlib.util.module_from_spec(package_spec)
sys.modules["symmetrix"] = package
package_spec.loader.exec_module(package)
validate_symmetrix_spec("symmetrix", package.__spec__)
if [Path(item).resolve() for item in package.__path__] != [package_dir]:
    raise RuntimeError("symmetrix namespace path is outside the bound package root")
module_name = extension_path.name.split(".", 1)[0]
native_package_name = extension_path.parent.name
if native_package_name != "symmetrix":
    native_package_path = require_file_within(
        native_package_dir / "__init__.py",
        native_package_dir,
        "native package origin",
    )
    native_package_spec = importlib.util.spec_from_file_location(
        native_package_name, native_package_path,
        submodule_search_locations=[str(native_package_dir)],
    )
    if native_package_spec is None or native_package_spec.loader is None:
        raise RuntimeError("cannot load requested native package")
    native_package = importlib.util.module_from_spec(native_package_spec)
    sys.modules[native_package_name] = native_package
    native_package_spec.loader.exec_module(native_package)
native_qualified_name = f"{native_package_name}.{module_name}"
native_spec = importlib.util.spec_from_file_location(native_qualified_name, extension_path)
if native_spec is None or native_spec.loader is None:
    raise RuntimeError("cannot load requested native extension")
native = importlib.util.module_from_spec(native_spec)
sys.modules[native_qualified_name] = native
native_spec.loader.exec_module(native)
build_info = native._backend_build_info()

def validate_loaded_symmetrix_namespace():
    for name, module in tuple(sys.modules.items()):
        if name != "symmetrix" and not name.startswith("symmetrix."):
            continue
        spec = getattr(module, "__spec__", None)
        validate_symmetrix_spec(name, spec)

if request["backend"] == "host":
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact
    validate_loaded_symmetrix_namespace()
    artifact = prepare_nequip_oam_receiver_host_artifact(
        request["model_path"], request["precision"], native_module=native,
        installed_prebuilt_required=True, expected_host_target=request["host_target"],
        expected_floating_point_policy=request["host_floating_point_policy"],
    )
    artifact_path = require_file_within(
        artifact.artifact_path, package_dir, "installed artifact origin"
    )
    manifest_path = require_file_within(
        artifact.manifest_path, package_dir, "installed artifact manifest origin"
    )
    result = {
        "status": artifact.status, "origin": artifact.artifact_origin,
        "identity": artifact.artifact_id,
        "artifact_path": str(artifact_path),
        "manifest_path": str(manifest_path),
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "native_build_target": build_info["prebuilt_operator_artifacts"]["target"],
        "active_host_target": artifact.active_host_target,
        "floating_point_policy": artifact.floating_point_policy,
    }
else:
    from symmetrix.oam_cuda_artifact import prepare_nequip_oam_cuda_artifact
    validate_loaded_symmetrix_namespace()
    artifact = prepare_nequip_oam_cuda_artifact(
        request["model_path"], request["precision"], request["cuda_target"],
        native_module=native, installed_prebuilt_required=True,
        expected_target=request["cuda_target"],
    )
    if artifact.status != "installed":
        raise RuntimeError("current CUDA artifact is not an installed prebuilt")
    artifact_path = require_file_within(
        artifact.artifact_path, native_package_dir, "installed artifact origin"
    )
    manifest_path = require_file_within(
        artifact.manifest_path,
        native_package_dir,
        "installed artifact manifest origin",
    )
    result = {
        "status": artifact.status, "origin": artifact.artifact_origin,
        "compiler": artifact.compiler, "identity": artifact.artifact_id,
        "artifact_path": str(artifact_path),
        "manifest_path": str(manifest_path),
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "target": artifact.target,
    }
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        [request["python_executable"], "-c", script, json.dumps(dict(request))],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or ["unknown error"]
        raise QualificationError(
            "retained installed artifact could not be re-resolved: " + detail[0]
        )
    try:
        return json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise QualificationError(
            "retained installed artifact probe returned malformed evidence"
        ) from error


def _revalidate_installed_artifact(
    request: Mapping[str, Any], direct: Mapping[str, Any]
) -> None:
    """Reject retained evidence unless the current catalog resolves identically."""

    artifact = _require_mapping(direct.get("artifact"), "direct.artifact")
    try:
        resolved = _probe_current_installed_artifact(request)
    except QualificationError:
        raise
    except Exception as error:
        raise QualificationError(
            f"retained installed artifact probe failed: {error}"
        ) from error
    expected = {
        "status": artifact["actual_status"],
        "origin": artifact["actual_origin"],
        "identity": artifact["actual_identity"],
        "artifact_path": direct["artifact_path"],
        "sha256": artifact["sha256"],
    }
    if request["backend"] == "host":
        expected.update(
            native_build_target=artifact["native_build_target"],
            active_host_target=artifact["active_host_target"],
            floating_point_policy=artifact["actual_floating_point_policy"],
        )
    else:
        expected.update(compiler=artifact["actual_compiler"], target=artifact["target"])
    for field, value in expected.items():
        matches = _json_values_equal(resolved.get(field), value)
        if field == "target" and request["backend"] == "cuda":
            try:
                matches = _normalized_cuda_target(
                    resolved.get(field)
                ) == _normalized_cuda_target(value)
            except QualificationError:
                matches = False
        if not matches:
            raise QualificationError(
                f"retained artifact.{field} does not match the current installed catalog"
            )
    artifact_path = resolved.get("artifact_path")
    if not isinstance(artifact_path, str) or not Path(artifact_path).is_file():
        raise QualificationError("retained installed artifact is absent")
    if _sha256_bytes(Path(artifact_path).read_bytes()) != artifact["sha256"]:
        raise QualificationError(
            "retained installed artifact bytes do not match artifact.sha256"
        )
    manifest_path = resolved.get("manifest_path")
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_file():
        raise QualificationError("retained installed artifact manifest is absent")


def _select_bindings(
    declaration: Any, cell: Mapping[str, Any], name: str
) -> list[Mapping[str, Any]]:
    if isinstance(declaration, list):
        return declaration
    declaration = _require_mapping(declaration, name)
    selectors = (
        cell["cell_id"],
        "/".join(
            (
                cell["model_size"],
                cell["backend"],
                cell["precision"],
                cell["profile"],
            )
        ),
        "/".join((cell["model_size"], cell["backend"], cell["precision"])),
        "/".join((cell["model_size"], cell["backend"])),
        cell["model_size"],
    )
    for selector in selectors:
        value = declaration.get(selector)
        if isinstance(value, list) and value:
            return value
    raise QualificationError(f"{name} has no binding declaration for {cell['cell_id']}")


@lru_cache(maxsize=8)
def _current_source_provenance(source_root_text: str) -> dict[str, Any]:
    source_root = Path(source_root_text).resolve()
    if not source_root.is_dir():
        return {
            "source_exists": False,
            "source_fingerprint": "0" * 64,
            "source_commit": "0" * 40,
            "source_dirty": True,
        }
    try:
        from tools._symmetrix_build.source_provenance import (
            symmetrix_source_fingerprint,
        )

        fingerprint = symmetrix_source_fingerprint(source_root)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (ImportError, OSError, subprocess.CalledProcessError, ValueError):
        return {
            "source_exists": False,
            "source_fingerprint": "0" * 64,
            "source_commit": "0" * 40,
            "source_dirty": True,
        }
    return {
        "source_exists": True,
        "source_fingerprint": fingerprint,
        "source_commit": commit,
        "source_dirty": dirty,
    }


def _cell_id(
    kind: str,
    model_size: str,
    backend: str,
    precision: str,
    graph_class: str,
    profile: str,
    threads: int,
) -> str:
    return "-".join(
        (
            kind,
            model_size.lower(),
            backend,
            precision,
            graph_class,
            profile,
            f"t{threads}",
        )
    )


def _make_cell(
    *,
    kind: str,
    model_size: str,
    backend: str,
    precision: str,
    graph_class: str,
    profile: str,
    threads: int,
) -> dict[str, Any]:
    return {
        "cell_id": _cell_id(
            kind,
            model_size,
            backend,
            precision,
            graph_class,
            profile,
            threads,
        ),
        "kind": kind,
        "model_size": model_size,
        "backend": backend,
        "precision": precision,
        "graph_class": graph_class,
        "profile": profile,
        "threads": threads,
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    manifest = _require_mapping(manifest, "manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or not isinstance(manifest.get("version"), int)
        or isinstance(manifest.get("version"), bool)
        or manifest.get("version") != SCHEMA_VERSION
    ):
        raise QualificationError("unsupported qualification manifest schema/version")
    models = _require_mapping(manifest.get("models"), "models")
    graphs = _require_mapping(manifest.get("graphs"), "graphs")
    missing_models = [size for size in MODEL_SIZES if size not in models]
    missing_graphs = [name for name in GRAPH_CLASSES if name not in graphs]
    if missing_models:
        raise QualificationError(f"manifest models missing {missing_models}")
    if missing_graphs:
        raise QualificationError(f"manifest graphs missing {missing_graphs}")
    for name in GRAPH_CLASSES:
        graph = _require_mapping(graphs[name], f"graphs.{name}")
        structure = graph.get("structure")
        specific = {"repeat"} if structure == "srtio3" else {"atoms"}
        allowed = GRAPH_COMMON_FIELDS | specific
        if structure not in {"srtio3", "chain"}:
            raise QualificationError(f"graphs.{name}.structure is unsupported")
        unknown = sorted(set(graph) - allowed)
        missing = sorted(allowed - set(graph))
        if unknown or missing:
            raise QualificationError(
                f"graphs.{name} graph fields invalid: missing={missing}, unknown={unknown}"
            )
        count_name = next(iter(specific))
        count = graph[count_name]
        if count_name == "repeat" and isinstance(count, list):
            count_is_valid = len(count) == 3 and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
                for value in count
            )
        else:
            count_is_valid = (
                isinstance(count, int)
                and not isinstance(count, bool)
                and count >= (2 if count_name == "atoms" else 1)
            )
        if not count_is_valid:
            raise QualificationError(f"graphs.{name}.{count_name} is invalid")
        if name == HOST_GRAPH_CLASS and count != [2, 2, 5]:
            raise QualificationError(
                "graphs.host-100.repeat must be the balanced factors [2, 2, 5]"
            )
        seed = graph["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise QualificationError(f"graphs.{name}.seed must be an integer")
        for field in ("rattle", "skin"):
            value = _finite_number(graph[field], f"graphs.{name}.{field}")
            if value < 0:
                raise QualificationError(
                    f"graphs.{name}.{field} must be finite and nonnegative"
                )
    boundaries = manifest.get("capacity_boundaries", [])
    required_boundary = {
        "model_size": "XL",
        "backend": "cuda",
        "precision": "float32",
        "graph_class": "production",
    }
    if not isinstance(boundaries, list) or len(boundaries) != 1:
        raise QualificationError("exactly one capacity boundary is required")
    if boundaries[0] != required_boundary:
        raise QualificationError("capacity boundary must be XL/cuda/float32/production")
    for target_name in ("host_target", "cuda_target"):
        if not isinstance(manifest.get(target_name), str) or not manifest[target_name]:
            raise QualificationError(f"{target_name} must be a non-empty string")
    if manifest["host_target"] != "x86-64-v3":
        raise QualificationError(
            "host_target must be the release-authorized x86-64-v3 target"
        )
    if manifest.get("host_floating_point_policy") not in {"strict", "fast"}:
        raise QualificationError(
            "host_floating_point_policy must be 'strict' or 'fast'"
        )
    for count_name, default in (("samples", 9), ("warmups", 3)):
        _integer(manifest.get(count_name, default), count_name, minimum=1)
    if manifest.get("edge_variant", "full_message") != "full_message":
        raise QualificationError("edge_variant must be full_message")
    provenance = _require_mapping(manifest.get("provenance"), "provenance")
    _require(provenance, "source_root")
    native = _require_mapping(provenance.get("native_paths"), "native_paths")
    for backend in BACKENDS:
        _require(native, backend)
    for bindings_name in ("required_bindings", "required_reference_bindings"):
        bindings = manifest.get(bindings_name)
        if isinstance(bindings, Mapping):
            if not bindings:
                raise QualificationError(f"{bindings_name} must not be empty")
            for selector, value in bindings.items():
                if (
                    not isinstance(selector, str)
                    or not isinstance(value, list)
                    or not value
                ):
                    raise QualificationError(
                        f"{bindings_name}.{selector} must be a non-empty array"
                    )
        elif not isinstance(bindings, list) or not bindings:
            raise QualificationError(
                f"{bindings_name} must be a non-empty array or per-model object"
            )
        binding_groups = (
            bindings.values() if isinstance(bindings, Mapping) else (bindings,)
        )
        for group in binding_groups:
            node_ids = []
            for index, binding in enumerate(group):
                binding = _require_mapping(binding, f"{bindings_name}[{index}]")
                if set(binding) != set(REQUIRED_BINDING_FIELDS):
                    raise QualificationError(
                        f"{bindings_name}[{index}] fields are invalid"
                    )
                for field in REQUIRED_BINDING_FIELDS:
                    _require(binding, field)
                if binding["qualified"] is not True:
                    raise QualificationError("required binding must be qualified")
                node_ids.append(binding["operator_id"])
            if len(set(node_ids)) != len(node_ids):
                raise QualificationError(f"duplicate {bindings_name} node_id")
    baselines = _require_mapping(
        manifest.get("direct_baselines_us_per_atom", {}),
        "direct_baselines_us_per_atom",
    )
    for cell_id, evidence in baselines.items():
        if evidence is None:
            continue
        evidence = _require_mapping(evidence, f"baseline {cell_id}")
        if set(evidence) != {"identity", "sha256", "us_per_atom"}:
            raise QualificationError(f"baseline {cell_id} has unsupported fields")
        _require(evidence, "identity")
        _require_digest(evidence, "sha256")
        _finite_number(
            evidence["us_per_atom"], f"baseline {cell_id} value", positive=True
        )


def _bind_request(
    manifest: Mapping[str, Any], cell: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Bind a cell to every mutable input used to produce its evidence."""

    model_path = _model_path(manifest, cell)
    model_exists = model_path.is_file()
    model_sha256 = _sha256_bytes(model_path.read_bytes()) if model_exists else "0" * 64
    graph = dict(_require_mapping(manifest["graphs"][cell["graph_class"]], "graph"))
    provenance = _require_mapping(manifest["provenance"], "provenance")
    current_source = _current_source_provenance(str(provenance["source_root"]))
    native_paths = _require_mapping(provenance["native_paths"], "native_paths")
    native_path = Path(str(native_paths[cell["backend"]])).resolve()
    native_exists = native_path.is_file()
    native_sha256 = (
        _sha256_bytes(native_path.read_bytes()) if native_exists else "0" * 64
    )
    baseline_evidence = _require_mapping(
        manifest.get("direct_baselines_us_per_atom", {}),
        "direct_baselines_us_per_atom",
    ).get(cell["cell_id"])
    required_bindings = _select_bindings(
        manifest["required_bindings"], cell, "required_bindings"
    )
    required_reference_bindings = _select_bindings(
        manifest["required_reference_bindings"],
        cell,
        "required_reference_bindings",
    )
    baseline_identity = baseline_evidence.get("identity") if baseline_evidence else None
    baseline_sha256 = baseline_evidence.get("sha256") if baseline_evidence else None
    baseline_value = baseline_evidence.get("us_per_atom") if baseline_evidence else None
    model_path = model_path.resolve()
    output_root = output_dir.resolve()
    cell_record_path = output_root / "cells" / f"{cell['cell_id']}.json"
    bound = {
        "schema": REQUEST_SCHEMA,
        "version": SCHEMA_VERSION,
        **cell,
        "model_exists": model_exists,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "python_executable": str(Path(sys.executable).absolute()),
        "python_real_executable": str(Path(sys.executable).resolve()),
        "driver_path": str(Path(__file__).resolve().with_name("oam_cuda_r1_timing.py")),
        "package_path": _current_package_path(),
        "output_root": str(output_root),
        "cell_record_path": str(cell_record_path),
        "direct_record_path": str(cell_record_path.with_suffix(".direct.json")),
        "reference_record_path": str(cell_record_path.with_suffix(".reference.json")),
        "direct_numerical_path": str(
            cell_record_path.with_suffix(".direct.numerics.npz")
        ),
        "reference_numerical_path": str(
            cell_record_path.with_suffix(".reference.numerics.npz")
        ),
        "graph_config": graph,
        "graph_input_sha256": _canonical_sha256(graph),
        "host_target": manifest["host_target"],
        "host_floating_point_policy": manifest["host_floating_point_policy"],
        "cuda_target": manifest.get("cuda_target"),
        "routes": ["direct", "reference"],
        "baseline_id": baseline_identity,
        "baseline_sha256": baseline_sha256,
        "baseline_us_per_atom": baseline_value,
        "source_root": str(Path(str(provenance["source_root"])).resolve()),
        **current_source,
        "native_path": str(native_path),
        "native_exists": native_exists,
        "native_sha256": native_sha256,
        "required_bindings": list(required_bindings),
        "required_reference_bindings": list(required_reference_bindings),
        "binding_report_version": 1,
        "samples": _integer(manifest.get("samples", 9), "samples", minimum=1),
        "warmups": _integer(manifest.get("warmups", 3), "warmups", minimum=1),
        "edge_variant": manifest.get("edge_variant", "full_message"),
        "generation_policy_request": _canonical_generation_policy(
            cell["backend"], cell["precision"], manifest.get("cuda_target")
        ),
        "artifact_evidence_contract": (
            {
                "direct": {
                    "kind": "host-direct",
                    "expected_status": "installed",
                    "expected_origin": "prebuilt",
                    "expected_target": manifest["host_target"],
                    "expected_floating_point_policy": manifest[
                        "host_floating_point_policy"
                    ],
                    "identity": "exact-derived",
                },
                "reference": {"kind": "reference", "origin": "none"},
            }
            if cell["backend"] == "host"
            else {
                "direct": {
                    "kind": "cuda-direct",
                    "expected_target": manifest["cuda_target"],
                    "expected_status": "installed",
                    "expected_origin": "prebuilt",
                    "expected_compiler": "prebuilt",
                    "identity": "exact-derived",
                },
                "reference": {"kind": "reference", "origin": "none"},
            }
        ),
    }
    bound["request_fingerprint"] = _canonical_sha256(bound)
    bound["jit_cache_path"] = str(
        output_root / "jit" / str(cell["cell_id"]) / bound["request_fingerprint"]
    )
    return bound


def build_declared_cells(
    manifest: Mapping[str, Any], output_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Return the stable complete matrix plus scaling and boundary cases."""

    validate_manifest(manifest)
    cells = [
        _make_cell(
            kind="matrix" if threads == 1 else "cpu-scaling",
            model_size=model_size,
            backend="host",
            precision=precision,
            graph_class=HOST_GRAPH_CLASS,
            profile="speed",
            threads=threads,
        )
        for model_size, precision in HOST_RUNTIME_PAIRS
        for threads in HOST_THREADS
    ]
    cells.extend(
        _make_cell(
            kind="matrix",
            model_size=model_size,
            backend="cuda",
            precision=precision,
            graph_class=graph_class,
            profile="speed",
            threads=1,
        )
        for model_size in MODEL_SIZES
        for precision in PRECISIONS
        for graph_class in CUDA_GRAPH_CLASSES
    )
    seen_boundaries: set[tuple[str, str, str, str]] = set()
    for boundary in manifest.get("capacity_boundaries", []):
        boundary = _require_mapping(boundary, "capacity_boundaries[]")
        boundary_fields = {"model_size", "backend", "precision", "graph_class"}
        if set(boundary) != boundary_fields:
            raise QualificationError("capacity boundary fields are invalid")
        model_size = str(boundary.get("model_size"))
        backend = str(boundary.get("backend"))
        precision = str(boundary.get("precision"))
        graph_class = str(boundary.get("graph_class"))
        if model_size not in MODEL_SIZES:
            raise QualificationError(f"invalid capacity model_size {model_size!r}")
        if backend not in BACKENDS:
            raise QualificationError(f"invalid capacity backend {backend!r}")
        if precision not in PRECISIONS:
            raise QualificationError(f"invalid capacity precision {precision!r}")
        if graph_class not in GRAPH_CLASSES:
            raise QualificationError(f"invalid capacity graph_class {graph_class!r}")
        boundary_key = (model_size, backend, precision, graph_class)
        if boundary_key in seen_boundaries:
            raise QualificationError(f"duplicate capacity boundary {boundary_key}")
        seen_boundaries.add(boundary_key)
        cells.append(
            _make_cell(
                kind="capacity-boundary",
                model_size=model_size,
                backend=backend,
                precision=precision,
                graph_class=graph_class,
                profile="capacity",
                threads=1,
            )
        )
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise QualificationError("duplicate generated cell_id")
    baseline_ids = set(
        _require_mapping(
            manifest.get("direct_baselines_us_per_atom", {}),
            "direct_baselines_us_per_atom",
        )
    )
    unknown_baselines = sorted(baseline_ids - {cell["cell_id"] for cell in cells})
    if unknown_baselines:
        raise QualificationError(f"unknown baseline cell IDs {unknown_baselines}")
    if output_dir is None:
        output_dir = Path(str(manifest.get("output_root", "oam-qualification")))
    return [_bind_request(manifest, cell, output_dir) for cell in cells]


def _validate_cell_record(
    request: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    """Fail closed unless a measured cell contains complete matching evidence."""

    if (
        record.get("schema") != CELL_SCHEMA
        or not isinstance(record.get("version"), int)
        or isinstance(record.get("version"), bool)
        or record.get("version") != SCHEMA_VERSION
    ):
        raise QualificationError("unsupported cell schema/version")
    if record.get("status") != "measured":
        raise QualificationError("cell status must be measured")
    if not _json_values_equal(record.get("request"), dict(request)):
        raise QualificationError(
            "request metadata does not exactly match the current cell"
        )
    for field in (
        "cell_id",
        "kind",
        "model_size",
        "backend",
        "precision",
        "graph_class",
        "profile",
        "threads",
    ):
        if record.get(field) != request.get(field):
            raise QualificationError(
                f"{field} mismatch: expected {request.get(field)!r}, got {record.get(field)!r}"
            )
    provenance = _require_mapping(record.get("provenance"), "provenance")
    for field in (
        "source_fingerprint",
        "source_commit",
        "source_dirty",
        "native_sha256",
        "model_sha256",
    ):
        if provenance.get(field) != request.get(field):
            raise QualificationError(
                f"provenance.{field} does not match current request"
            )
    native_fast_math = provenance.get("native_fast_math")
    if request["backend"] == "host":
        if native_fast_math is not (request["host_floating_point_policy"] == "fast"):
            raise QualificationError(
                "provenance.native_fast_math does not match the request"
            )
    elif native_fast_math is not None:
        raise QualificationError("provenance.native_fast_math must be null for CUDA")

    for path in (
        "provenance.source_fingerprint",
        "provenance.source_commit",
        "provenance.source_dirty",
        "provenance.native_sha256",
        "provenance.model_sha256",
        "provenance.graph_sha256",
        "provenance.jit_cache",
        "provenance.command",
        "artifact.actual_identity",
        "artifact.actual_origin",
        "artifact.sha256",
        "execution.plan_id",
        "execution.bindings",
        "execution.schedule",
        "execution.backend",
        "execution.target",
        "execution.fallback_count",
        "graph.atom_count",
        "graph.cutoff",
        "graph.skin",
        "graph.effective_cutoff",
        "graph.directed_edges",
        "memory.active_bytes",
        "memory.retained_bytes",
        "memory.peak_transient_bytes",
        "memory.replacements",
        "reuse.replacement_delta",
        "reuse.completed",
        "timing.unit",
        "timing.direct_us_per_atom",
        "timing.reference_us_per_atom",
        "timing.timed_region",
        "timing.excluded",
        "correctness.passed",
    ):
        _require(record, path)
    for path in (
        "provenance.source_fingerprint",
        "provenance.native_sha256",
        "provenance.model_sha256",
        "provenance.graph_sha256",
        "artifact.sha256",
    ):
        _require_digest(record, path)
    _require_digest(record, "provenance.source_commit", length=40)

    execution = _require_mapping(record["execution"], "execution")
    if execution["backend"] != request["backend"]:
        raise QualificationError(
            "execution.backend does not match the requested backend"
        )
    if execution.get("edge_variant") != request["edge_variant"]:
        raise QualificationError(
            "execution.edge_variant does not match the requested variant"
        )
    expected_schedule = "capacity" if request["profile"] == "capacity" else "throughput"
    if execution["schedule"] != expected_schedule:
        raise QualificationError(
            "execution.schedule does not match the requested profile"
        )
    expected_plan = (
        "oam-direct-capacity"
        if request["profile"] == "capacity"
        else "oam-direct-speed"
    )
    if execution["plan_id"] != expected_plan:
        raise QualificationError(
            f"execution.plan_id must be {expected_plan!r} for this request"
        )
    if (
        _integer(execution["fallback_count"], "execution.fallback_count", minimum=0)
        != 0
    ):
        raise QualificationError("execution.fallback_count must be zero")
    bindings = execution["bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise QualificationError("execution.bindings must be a non-empty array")
    for index, binding in enumerate(bindings):
        binding = _require_mapping(binding, f"execution.bindings[{index}]")
        for field in (
            "operator_id",
            "capability_key",
            "implementation_id",
            "selected_implementation_id",
            "tier",
            "qualified",
            "portable_covered",
            "artifact_required",
            "target",
        ):
            _require(binding, field)
        if binding["qualified"] is not True:
            raise QualificationError(f"execution.bindings[{index}] is not qualified")
        if binding["target"] != request["backend"]:
            raise QualificationError(
                f"execution.bindings[{index}].target does not match the requested backend"
            )
        if (
            binding["capability_key"] == "edge_tensor_product"
            and binding["tier"] != "generated"
        ):
            raise QualificationError(
                f"execution.bindings[{index}] does not bind the generated edge tier"
            )

    graph = _require_mapping(record["graph"], "graph")
    effective_cutoff = _finite_number(
        graph["effective_cutoff"], "graph.effective_cutoff"
    )
    cutoff = _finite_number(graph["cutoff"], "graph.cutoff")
    skin = _finite_number(graph["skin"], "graph.skin")
    if not math.isclose(
        effective_cutoff,
        cutoff + skin,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise QualificationError("graph.effective_cutoff does not equal cutoff + skin")
    _integer(graph["atom_count"], "graph.atom_count", minimum=1)
    _integer(graph["directed_edges"], "graph.directed_edges", minimum=1)

    memory = _require_mapping(record["memory"], "memory")
    retained_bytes = _integer(
        memory["retained_bytes"], "memory.retained_bytes", minimum=0
    )
    active_bytes = _integer(memory["active_bytes"], "memory.active_bytes", minimum=0)
    _integer(memory["peak_transient_bytes"], "memory.peak_transient_bytes", minimum=0)
    _integer(memory["replacements"], "memory.replacements", minimum=0)
    if retained_bytes < active_bytes:
        raise QualificationError("memory.retained_bytes is smaller than active_bytes")
    if (
        _integer(
            record["reuse"]["replacement_delta"], "reuse.replacement_delta", minimum=0
        )
        != 0
    ):
        raise QualificationError("reuse.replacement_delta must be zero")
    if record["reuse"]["completed"] is not True:
        raise QualificationError("reuse.completed must be true")

    timing = _require_mapping(record["timing"], "timing")
    if timing["unit"] != "us/atom":
        raise QualificationError("timing.unit must be us/atom")
    if timing["timed_region"] != TIMED_REGION:
        raise QualificationError(
            "timing.timed_region does not match the qualification contract"
        )
    excluded = set(timing["excluded"])
    missing_exclusions = sorted(set(TIMING_EXCLUSIONS) - excluded)
    if missing_exclusions:
        raise QualificationError(f"timing.excluded missing {missing_exclusions}")
    _finite_number(
        timing["direct_us_per_atom"], "timing.direct_us_per_atom", positive=True
    )
    _finite_number(
        timing["reference_us_per_atom"], "timing.reference_us_per_atom", positive=True
    )

    if request["backend"] == "host":
        cpu = _require_mapping(record.get("cpu"), "cpu")
        for field in (
            "physical_cores",
            "kokkos_threads",
            "observed_kokkos_threads",
            "observed_openmp_threads",
        ):
            value = _require(cpu, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != int(request["threads"])
            ):
                raise QualificationError(
                    f"cpu.{field} does not match requested threads"
                )
        if cpu.get("blas_threads") != 1:
            raise QualificationError("cpu.blas_threads must be one")
        logical = cpu.get("logical_cpu_ids")
        topology = cpu.get("topology")
        if not isinstance(logical, list) or len(logical) != request["threads"]:
            raise QualificationError("cpu.logical_cpu_ids must list every selected CPU")
        if not isinstance(topology, list) or len(topology) != len(logical):
            raise QualificationError("cpu.topology must describe every selected CPU")
        topology_ids = []
        physical = set()
        for row in topology:
            row = _require_mapping(row, "cpu.topology[]")
            topology_ids.append(row.get("logical_cpu"))
            physical.add((row.get("package"), row.get("core")))
        if topology_ids != logical or len(physical) != request["threads"]:
            raise QualificationError(
                "cpu.topology does not describe distinct physical cores"
            )
    if request["kind"] == "capacity-boundary":
        if _require(record, "capacity_boundary.completed") is not True:
            raise QualificationError("capacity_boundary.completed must be true")
    evidence = _require_mapping(record.get("evidence"), "evidence")
    from benchmarks.oam_qualification_record import validate_raw_driver_record

    validate_raw_driver_record(request, evidence.get("direct"), "direct")
    validate_raw_driver_record(request, evidence.get("reference"), "reference")
    direct_evidence = evidence["direct"]
    reference_evidence = evidence["reference"]
    _revalidate_installed_artifact(request, direct_evidence)
    for field in ("artifact", "execution", "graph", "memory", "reuse", "cpu"):
        if not _json_values_equal(record.get(field), direct_evidence.get(field)):
            raise QualificationError(
                f"{field} does not match preserved direct evidence"
            )
    from benchmarks.oam_qualification_record import _canonical_cell_timing

    expected_timing = _canonical_cell_timing(direct_evidence, reference_evidence)
    if not _json_values_equal(timing, expected_timing):
        raise QualificationError("timing does not equal exact raw timing projection")
    direct_timing = expected_timing["direct"]
    reference_timing = expected_timing["reference"]
    if timing["direct_us_per_atom"] != direct_timing["median_us_per_atom"]:
        raise QualificationError("direct timing does not match preserved evidence")
    if timing["reference_us_per_atom"] != reference_timing["median_us_per_atom"]:
        raise QualificationError("reference timing does not match preserved evidence")
    from benchmarks.oam_qualification_record import _correctness_report

    expected_correctness = _correctness_report(
        request, direct_evidence, reference_evidence
    )
    if not _json_values_equal(record.get("correctness"), expected_correctness):
        raise QualificationError(
            "correctness does not equal exact numerical evidence projection"
        )
    if record["correctness"].get("passed") is not True:
        raise QualificationError("correctness parity gate did not pass")
    baseline = _require_mapping(record.get("baseline"), "baseline")
    if baseline.get("identity") != request["baseline_id"]:
        raise QualificationError("baseline.identity does not match the request")
    if baseline.get("sha256") != request["baseline_sha256"]:
        raise QualificationError("baseline.sha256 does not match the request")
    baseline_value = baseline.get("direct_us_per_atom")
    if baseline_value != request["baseline_us_per_atom"]:
        raise QualificationError("baseline value does not match the request")
    if baseline_value is not None and (
        isinstance(baseline_value, bool)
        or not isinstance(baseline_value, (int, float))
        or not math.isfinite(float(baseline_value))
        or float(baseline_value) <= 0.0
    ):
        raise QualificationError(
            "baseline.direct_us_per_atom must be finite and positive"
        )


def validate_cell_record(request: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """Normalize every malformed retained-record failure to QualificationError."""

    try:
        _validate_cell_record(request, record)
    except QualificationError:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise QualificationError(f"malformed qualification cell: {error}") from error


def evaluate_qualification(
    cells: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply all release thresholds without filling gaps or missing baselines."""

    malformed_rows = [
        index for index, record in enumerate(records) if not isinstance(record, Mapping)
    ]
    valid_records = [record for record in records if isinstance(record, Mapping)]
    record_ids = [str(record.get("cell_id")) for record in valid_records]
    duplicate_ids = sorted({item for item in record_ids if record_ids.count(item) > 1})
    cell_ids = [str(cell.get("cell_id")) for cell in cells]
    duplicate_cell_ids = sorted({item for item in cell_ids if cell_ids.count(item) > 1})
    expected_declarations = {
        (
            "host",
            model_size,
            precision,
            HOST_GRAPH_CLASS,
            "matrix" if threads == 1 else "cpu-scaling",
            "speed",
            threads,
        )
        for model_size, precision in HOST_RUNTIME_PAIRS
        for threads in HOST_THREADS
    }
    expected_declarations.update(
        {
            (
                "cuda",
                model_size,
                precision,
                graph_class,
                "matrix",
                "speed",
                1,
            )
            for model_size in MODEL_SIZES
            for precision in PRECISIONS
            for graph_class in CUDA_GRAPH_CLASSES
        }
    )
    expected_declarations.add(
        ("cuda", "XL", "float32", "production", "capacity-boundary", "capacity", 1)
    )
    declared = {
        (
            cell.get("backend"),
            cell.get("model_size"),
            cell.get("precision"),
            cell.get("graph_class"),
            cell.get("kind"),
            cell.get("profile"),
            cell.get("threads"),
        )
        for cell in cells
    }
    declaration_complete = (
        len(cells) == 27
        and len(cell_ids) == len(set(cell_ids))
        and declared == expected_declarations
    )
    unexpected_records = [
        record for record in valid_records if str(record.get("cell_id")) not in cell_ids
    ]
    evidence_set_exact = (
        not malformed_rows
        and not duplicate_ids
        and not duplicate_cell_ids
        and not unexpected_records
        and len(valid_records) == len(cells)
        and set(record_ids) == set(cell_ids)
    )
    by_id = {str(record.get("cell_id")): record for record in valid_records}
    failed_cells: list[str] = []
    unqualified_reasons: list[str] = []
    production_ratios: list[float] = []
    cpu_rows: dict[
        tuple[str, str], dict[int, tuple[float, str, Mapping[str, Any]]]
    ] = {}
    capacity_records: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    speed_records: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    if duplicate_ids:
        unqualified_reasons.append(f"duplicate evidence rows: {duplicate_ids}")
    if duplicate_cell_ids:
        unqualified_reasons.append(f"duplicate declared cells: {duplicate_cell_ids}")
    if malformed_rows:
        unqualified_reasons.append(
            f"evidence row is not an object at indexes {malformed_rows}"
        )
    if not declaration_complete:
        unqualified_reasons.append(
            "declared release matrix is not the exact 10-host + 17-CUDA set"
        )
    for record in unexpected_records:
        schema = record.get("schema")
        version = record.get("version")
        unqualified_reasons.append(
            "unexpected evidence row "
            f"{record.get('cell_id')!r} with schema/version {schema!r}/{version!r}"
        )
    correctness_complete = True
    correctness_passed = True

    for cell in cells:
        cell_id = str(cell["cell_id"])
        record = by_id.get(cell_id)
        if record is None:
            correctness_complete = False
            correctness_passed = False
            unqualified_reasons.append(f"{cell_id}: measured cell evidence is missing")
            continue
        try:
            validate_cell_record(cell, record)
        except QualificationError as error:
            correctness_complete = False
            correctness_passed = False
            failed_cells.append(cell_id)
            unqualified_reasons.append(
                f"{cell_id}: malformed qualification cell: {error}"
            )
            continue
        if record.get("correctness", {}).get("passed") is not True:
            correctness_passed = False
            failed_cells.append(cell_id)
        timing = record["timing"]
        direct = _finite_number(
            timing["direct_us_per_atom"], f"{cell_id}.direct", positive=True
        )
        reference = _finite_number(
            timing["reference_us_per_atom"], f"{cell_id}.reference", positive=True
        )
        if (
            cell["backend"] == "host"
            and cell["kind"] in {"matrix", "cpu-scaling"}
            and cell["profile"] == "speed"
        ):
            if direct > 0.90 * reference:
                failed_cells.append(cell_id)
            key = (str(cell["model_size"]), str(cell["precision"]))
            graph_identity = {
                "graph_sha256": record["provenance"]["graph_sha256"],
                "graph": record["graph"],
                "graph_input_sha256": cell["graph_input_sha256"],
                "graph_config": cell["graph_config"],
            }
            cpu_rows.setdefault(key, {})[int(cell["threads"])] = (
                direct,
                cell_id,
                graph_identity,
            )
        elif cell["kind"] == "matrix":
            if cell["graph_class"] == "production":
                production_ratios.append(reference / direct)
                if direct > 0.90 * reference:
                    failed_cells.append(cell_id)
            elif direct > 1.05 * reference:
                failed_cells.append(cell_id)
        baseline = record.get("baseline")
        baseline_value = (
            baseline.get("direct_us_per_atom")
            if isinstance(baseline, Mapping)
            else None
        )
        if baseline_value is None:
            unqualified_reasons.append(
                f"{cell_id}: approved direct baseline is missing"
            )
        elif direct > 1.05 * float(baseline_value):
            failed_cells.append(cell_id)

        key = (
            cell["model_size"],
            cell["backend"],
            cell["precision"],
            cell["graph_class"],
        )
        if cell["profile"] == "capacity":
            capacity_records[key] = record
        elif cell["profile"] == "speed" and int(cell["threads"]) == 1:
            speed_records[key] = record

    expected_production = sum(
        cell["kind"] == "matrix"
        and cell["backend"] == "cuda"
        and cell["graph_class"] == "production"
        for cell in cells
    )
    production_complete = bool(
        expected_production and len(production_ratios) == expected_production
    )
    geomean = (
        math.exp(
            sum(math.log(value) for value in production_ratios) / len(production_ratios)
        )
        if production_ratios
        else None
    )
    geomean_improvement = geomean - 1.0 if geomean is not None else None
    geomean_passed = bool(
        production_complete
        and geomean_improvement is not None
        and geomean_improvement >= 0.20 - 1e-12
    )
    if production_ratios and not geomean_passed:
        failed_cells.append("production-geomean")
    elif expected_production and not production_complete:
        unqualified_reasons.append(
            f"complete {expected_production}-cell CUDA production matrix is missing"
        )

    expected_cpu_pairs = {
        (str(cell["model_size"]), str(cell["precision"]))
        for cell in cells
        if cell["backend"] == "host"
    }
    cpu_speedups: dict[str, float] = {}
    cpu_scaling_complete = True
    for key in sorted(expected_cpu_pairs):
        rows = cpu_rows.get(key, {})
        if set(rows) != {1, 4}:
            cpu_scaling_complete = False
            unqualified_reasons.append(
                f"CPU scaling {key}: exact 1/4-thread pair is missing"
            )
            continue
        graphs_match = _json_values_equal(rows[1][2], rows[4][2])
        if not graphs_match:
            failed_cells.append(rows[4][1])
            unqualified_reasons.append(
                f"CPU scaling {key}: realized graph differs between t1 and t4"
            )
        speedup = rows[1][0] / rows[4][0]
        selector = "/".join(key)
        cpu_speedups[selector] = speedup
        if speedup <= 1.0:
            failed_cells.append(rows[4][1])
    cpu_scaling_passed = (
        cpu_scaling_complete
        and not any(
            "realized graph differs" in reason for reason in unqualified_reasons
        )
        and all(speedup > 1.0 for speedup in cpu_speedups.values())
    )

    capacity_passed = not duplicate_ids and not duplicate_cell_ids
    expected_capacity = {
        (cell["model_size"], cell["backend"], cell["precision"], cell["graph_class"])
        for cell in cells
        if cell["kind"] == "capacity-boundary"
    }
    if not expected_capacity:
        capacity_passed = False
        unqualified_reasons.append("no capacity boundary is declared")
    for key in sorted(expected_capacity):
        capacity = capacity_records.get(key)
        if capacity is None:
            capacity_passed = False
            unqualified_reasons.append(f"capacity {key}: boundary record is missing")
            continue
        speed = speed_records.get(key)
        if speed is None:
            capacity_passed = False
            unqualified_reasons.append(
                f"capacity {key}: matching speed record is missing"
            )
            continue
        if (
            capacity["provenance"]["graph_sha256"]
            != speed["provenance"]["graph_sha256"]
        ):
            capacity_passed = False
            failed_cells.append(str(capacity["cell_id"]))
            unqualified_reasons.append(f"capacity {key}: actual graph hash mismatch")
        if capacity["graph"] != speed["graph"]:
            capacity_passed = False
            failed_cells.append(str(capacity["cell_id"]))
            unqualified_reasons.append(f"capacity {key}: actual graph facts mismatch")
        if int(capacity["memory"]["retained_bytes"]) > int(
            speed["memory"]["retained_bytes"]
        ):
            capacity_passed = False
            failed_cells.append(str(capacity["cell_id"]))
        if capacity.get("capacity_boundary", {}).get("completed") is not True:
            capacity_passed = False
            failed_cells.append(str(capacity["cell_id"]))

    failed_cells = sorted(set(failed_cells))
    status = (
        "failed" if failed_cells else "unqualified" if unqualified_reasons else "passed"
    )
    return {
        "schema": RESULT_SCHEMA,
        "version": SCHEMA_VERSION,
        "status": status,
        "failed_cells": failed_cells,
        "unqualified_reasons": sorted(set(unqualified_reasons)),
        "gates": {
            "production_geomean": {
                "complete": production_complete,
                "improvement": geomean_improvement,
                "passed": geomean_passed,
            },
            "cpu_scaling": {
                "complete": cpu_scaling_complete,
                "minimum_speedup": min(cpu_speedups.values()) if cpu_speedups else None,
                "passed": cpu_scaling_passed,
                "speedups": cpu_speedups,
            },
            "correctness": {
                "complete": correctness_complete
                and declaration_complete
                and evidence_set_exact,
                "passed": correctness_complete
                and correctness_passed
                and declaration_complete
                and evidence_set_exact,
            },
            "capacity": {"passed": capacity_passed},
        },
    }


def _model_path(manifest: Mapping[str, Any], cell: Mapping[str, Any]) -> Path:
    model = _require_mapping(manifest["models"][cell["model_size"]], "model")
    return Path(str(_require(model, "path")))


def _record_path(output_dir: Path, cell: Mapping[str, Any]) -> Path:
    expected = output_dir.resolve() / "cells" / f"{cell['cell_id']}.json"
    bound = Path(str(_require(cell, "cell_record_path")))
    if bound != expected:
        raise QualificationError("cell record path does not match current output root")
    return bound


def build_driver_command(
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    route: str,
    output_path: Path | None = None,
) -> list[str]:
    if route not in {"direct", "reference"}:
        raise QualificationError("driver route must be direct or reference")
    bound_output = Path(str(_require(cell, f"{route}_record_path")))
    if output_path is not None and output_path.resolve() != bound_output:
        raise QualificationError(f"{route} output path does not match the request")
    graph = _require_mapping(cell["graph_config"], "graph_config")
    command = [
        cell["python_executable"],
        cell["driver_path"],
        "cpu" if cell["backend"] == "host" else "cuda",
        cell["precision"],
        "--contract",
        cell["model_path"],
        "--structure",
        graph["structure"],
    ]
    if graph["structure"] == "srtio3":
        repeat = graph["repeat"]
        command.extend(
            (
                "--repeat",
                ",".join(str(value) for value in repeat)
                if isinstance(repeat, list)
                else str(repeat),
            )
        )
    else:
        command.extend(("--atoms", str(graph["atoms"])))
    command.extend(
        (
            "--rattle",
            str(graph["rattle"]),
            "--seed",
            str(graph["seed"]),
            "--skin",
            str(graph["skin"]),
            "--samples",
            str(cell["samples"]),
            "--warmup",
            str(cell["warmups"]),
            "--threads",
            str(cell["threads"]),
            "--host-target",
            str(cell["host_target"]),
            "--edge-variant",
            str(cell["edge_variant"]),
            "--routing",
            route,
            "--execution-profile",
            str(cell["profile"]),
            "--qualification-cell",
            json.dumps(dict(cell), sort_keys=True),
            "--output",
            str(bound_output),
            "--qualification-arrays",
            str(_require(cell, f"{route}_numerical_path")),
        )
    )
    return command


def validate_driver_qualification_request(
    request: Mapping[str, Any],
    *,
    command: Sequence[str],
    backend: str,
    precision: str,
    contract_path: Path,
    route: str,
    profile: str,
    edge_variant: str,
    actual_target: Any,
    native_path: Path,
    package_path: str,
    python_executable: str,
) -> dict[str, Any]:
    """Validate a complete request against the live driver invocation."""

    request = _require_mapping(request, "qualification request")
    if set(request) != REQUEST_FIELDS:
        raise QualificationError("qualification request fields do not match schema v3")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("version") != SCHEMA_VERSION
    ):
        raise QualificationError("unsupported qualification request schema/version")

    fingerprint = _require_digest(request, "request_fingerprint")
    fingerprint_payload = dict(request)
    fingerprint_payload.pop("request_fingerprint")
    fingerprint_payload.pop("jit_cache_path")
    if _canonical_sha256(fingerprint_payload) != fingerprint:
        raise QualificationError("qualification request fingerprint mismatch")
    expected_cache = (
        Path(str(request["output_root"]))
        / "jit"
        / str(request["cell_id"])
        / fingerprint
    )
    if Path(str(request["jit_cache_path"])) != expected_cache:
        raise QualificationError("qualification request JIT cache fingerprint mismatch")

    if list(command) != build_driver_command({}, request, route):
        raise QualificationError("driver command is not the canonical invocation")
    expected_backend = "host" if backend == "cpu" else backend
    if request["backend"] != expected_backend:
        raise QualificationError("qualification request backend mismatch")
    if request["precision"] != precision:
        raise QualificationError("qualification request precision mismatch")
    if request["profile"] != profile:
        raise QualificationError("qualification request profile mismatch")
    if request["edge_variant"] != edge_variant:
        raise QualificationError("qualification request edge variant mismatch")

    model_path = contract_path.expanduser().resolve()
    if str(model_path) != request["model_path"]:
        raise QualificationError("qualification request model path mismatch")
    try:
        model_sha256 = _sha256_bytes(model_path.read_bytes())
    except OSError as error:
        raise QualificationError(
            f"qualification model is unavailable: {error}"
        ) from error
    if model_sha256 != request["model_sha256"]:
        raise QualificationError("qualification request model hash mismatch")

    native_path = native_path.resolve()
    if str(native_path) != request["native_path"]:
        raise QualificationError("qualification request native path mismatch")
    try:
        native_sha256 = _sha256_bytes(native_path.read_bytes())
    except OSError as error:
        raise QualificationError(
            f"qualification native module is unavailable: {error}"
        ) from error
    if native_sha256 != request["native_sha256"]:
        raise QualificationError("qualification request native hash mismatch")
    if str(Path(package_path).resolve()) != request["package_path"]:
        raise QualificationError("qualification request package path mismatch")
    if python_executable != request["python_executable"]:
        raise QualificationError("qualification request Python executable mismatch")
    if str(Path(python_executable).resolve()) != request["python_real_executable"]:
        raise QualificationError(
            "qualification request real Python executable mismatch"
        )
    if request["driver_path"] != str(
        Path(__file__).resolve().with_name("oam_cuda_r1_timing.py")
    ):
        raise QualificationError("qualification request driver path mismatch")

    current_source = _current_source_provenance(str(request["source_root"]))
    for field, value in current_source.items():
        if request[field] != value:
            raise QualificationError(
                f"qualification request current provenance mismatch: {field}"
            )

    if expected_backend == "cuda":
        cuda_target = _normalized_cuda_target(actual_target)
        if cuda_target != _normalized_cuda_target(request["cuda_target"]):
            raise QualificationError("actual CUDA device target does not match request")
    return dict(request)


def _current_native_source_fingerprint(source_root: str | Path) -> str:
    from tools._symmetrix_build.source_provenance import (
        symmetrix_native_source_fingerprint,
    )

    return symmetrix_native_source_fingerprint(Path(source_root).resolve())


def _validate_private_jit_cache(request: Mapping[str, Any], active: str | None) -> None:
    expected = Path(str(request["jit_cache_path"])).resolve()
    if not active or Path(active).resolve() != expected:
        raise QualificationError(
            "active JIT cache (SYMMETRIX_JIT_CACHE) does not match request"
        )
    try:
        status = expected.stat()
    except OSError as error:
        raise QualificationError(
            f"qualification JIT cache is unavailable: {error}"
        ) from error
    if not expected.is_dir():
        raise QualificationError("qualification JIT cache is not a directory")
    if status.st_uid != os.getuid():
        raise QualificationError("qualification JIT cache has the wrong owner")
    if status.st_mode & 0o777 != 0o700:
        raise QualificationError("qualification JIT cache mode must be 0700")


def _require_live_native_build(
    request: Mapping[str, Any], native_module: Any
) -> dict[str, Any]:
    query = getattr(native_module, "_backend_build_info", None)
    build_info = query() if callable(query) else None
    if not isinstance(build_info, Mapping):
        raise QualificationError("loaded native extension has no build provenance")
    expected_backend = "cuda" if request["backend"] == "cuda" else "cpu"
    expected_target = (
        _normalized_cuda_target(request["cuda_target"])
        if request["backend"] == "cuda"
        else str(request["host_target"])
    )
    native_fingerprint = _current_native_source_fingerprint(request["source_root"])
    expected = {
        "backend": expected_backend,
        "source_commit": request["source_commit"],
        "source_dirty": request["source_dirty"],
        "source_content_sha256": request["source_fingerprint"],
        "native_source_content_sha256": native_fingerprint,
    }
    for field, wanted in expected.items():
        if build_info.get(field) != wanted:
            raise QualificationError(f"native build {field} mismatch")
    architecture = build_info.get("architecture")
    if request["backend"] == "cuda":
        if _normalized_cuda_target(architecture) != expected_target:
            raise QualificationError("native build target mismatch")
    elif (
        not isinstance(architecture, str) or request["host_target"] not in architecture
    ):
        raise QualificationError("native build target mismatch")
    fast_math = build_info.get("fast_math")
    if not isinstance(fast_math, bool):
        raise QualificationError("native build fast_math policy is missing")
    expected_policy = "fast" if fast_math else "strict"
    if (
        request["backend"] == "host"
        and expected_policy != request["host_floating_point_policy"]
    ):
        raise QualificationError("native build floating-point policy mismatch")
    prebuilt = build_info.get("prebuilt_operator_artifacts")
    if not isinstance(prebuilt, Mapping):
        raise QualificationError("native build has no prebuilt catalog provenance")
    if prebuilt.get("mode") not in ("r0", "full"):
        raise QualificationError("native build catalog mode must be r0 or full")
    expected_artifact_count = 14 if prebuilt["mode"] == "r0" else 40
    if prebuilt.get("artifact_count") != expected_artifact_count:
        raise QualificationError(
            "native build catalog artifact_count does not describe a combined "
            f"{prebuilt['mode']} + known OAM catalog"
        )
    installed_bytes = prebuilt.get("installed_bytes")
    if (
        isinstance(installed_bytes, bool)
        or not isinstance(installed_bytes, int)
        or installed_bytes <= 0
    ):
        raise QualificationError("native build catalog installed_bytes is invalid")
    for field in ("compiler_id", "compiler_version"):
        if not isinstance(prebuilt.get(field), str) or not prebuilt[field]:
            raise QualificationError(f"native build catalog {field} is incomplete")
    duration = prebuilt.get("build_duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or duration < 0
    ):
        raise QualificationError(
            "native build catalog build_duration_seconds is invalid"
        )
    catalog_target = prebuilt.get("target")
    if request["backend"] == "cuda":
        catalog_target = _normalized_cuda_target(catalog_target)
    elif catalog_target not in (expected_target, f"host-isa-{expected_target}"):
        raise QualificationError("native build catalog target mismatch")
    if prebuilt.get("backend") != request["backend"]:
        raise QualificationError("native build catalog backend mismatch")
    if catalog_target != expected_target and request["backend"] == "cuda":
        raise QualificationError("native build catalog target mismatch")
    for field, wanted in (
        ("source_content_sha256", request["source_fingerprint"]),
        ("native_source_content_sha256", native_fingerprint),
    ):
        if prebuilt.get(field) != wanted:
            raise QualificationError(f"native build catalog {field} mismatch")
    _require_digest(prebuilt, "catalog_sha256")
    return dict(build_info)


def validate_driver_qualification_preflight(
    request: Mapping[str, Any],
    *,
    native_module: Any,
    active_jit_cache: str | None,
    **validation: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed before qualification artifact or model preparation begins."""

    validated = validate_driver_qualification_request(request, **validation)
    _validate_private_jit_cache(validated, active_jit_cache)
    build_info = _require_live_native_build(validated, native_module)
    return validated, build_info


def build_driver_environment(
    cell: Mapping[str, Any], output_path: Path | None = None
) -> tuple[dict[str, str], Path]:
    fingerprint = _require_digest(cell, "request_fingerprint")
    expected_cache = (
        Path(str(_require(cell, "output_root")))
        / "jit"
        / str(cell["cell_id"])
        / fingerprint
    )
    cache = Path(str(_require(cell, "jit_cache_path")))
    if cache != expected_cache:
        raise QualificationError("JIT cache path does not match the request")
    if output_path is not None:
        valid_outputs = {
            Path(str(cell["cell_record_path"])),
            Path(str(cell["direct_record_path"])),
            Path(str(cell["reference_record_path"])),
        }
        if output_path.resolve() not in valid_outputs:
            raise QualificationError("driver output path does not match the request")
    environment = os.environ.copy()
    environment["SYMMETRIX_JIT_CACHE"] = str(cache)
    return environment, cache


def _default_runner(
    manifest: Mapping[str, Any], cell: Mapping[str, Any], output_path: Path
) -> None:
    from benchmarks.oam_qualification_record import build_cell_record

    if output_path.resolve() != Path(str(cell["cell_record_path"])):
        raise QualificationError("cell output path does not match the request")
    direct_path = Path(str(cell["direct_record_path"]))
    reference_path = Path(str(cell["reference_record_path"]))
    environment, cache = build_driver_environment(cell)
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache.chmod(0o700)
    for route, path in (("direct", direct_path), ("reference", reference_path)):
        subprocess.run(
            build_driver_command(manifest, cell, route),
            check=True,
            env=environment,
        )
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    record = build_cell_record(
        cell,
        direct,
        reference,
        baseline_us_per_atom=cell["baseline_us_per_atom"],
    )
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def orchestrate(
    manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    execute: bool,
    runner: Callable[[Mapping[str, Any], Path], None] | None = None,
) -> dict[str, Any]:
    """Declare or execute cells, reusing only strictly valid retained records."""

    # A long-lived orchestration process may inspect the same manifest after a
    # rebuild. Never let the source-provenance cache authorize stale evidence.
    _current_source_provenance.cache_clear()
    output_dir = output_dir.resolve()
    cells = build_declared_cells(manifest, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cells").mkdir(exist_ok=True)
    states = []
    resumed = 0
    for cell in cells:
        path = _record_path(output_dir, cell)
        state = {**cell, "record": str(path)}
        invalid_retained = False
        if path.is_file():
            try:
                retained = json.loads(path.read_text(encoding="utf-8"))
                validate_cell_record(cell, retained)
            except (OSError, json.JSONDecodeError, QualificationError) as error:
                invalid_retained = True
                state.update(
                    state="unqualified", reason=f"invalid retained record: {error}"
                )
            else:
                resumed += 1
                state.update(state="measured", reason="validated retained record")
                states.append(state)
                continue
        model_path = _model_path(manifest, cell)
        if not cell["model_exists"] or not model_path.is_file():
            state.update(
                state="unqualified", reason=f"model file is missing: {model_path}"
            )
        elif not cell["source_exists"]:
            state.update(
                state="unqualified", reason="current source provenance is unavailable"
            )
        elif not cell["native_exists"]:
            state.update(
                state="unqualified", reason="current native extension is missing"
            )
        elif not execute and invalid_retained:
            pass
        elif not execute:
            state.update(state="pending", reason="execution was not requested")
        else:
            try:
                if runner is None:
                    _default_runner(manifest, cell, path)
                else:
                    runner(cell, path)
                record = json.loads(path.read_text(encoding="utf-8"))
                validate_cell_record(cell, record)
            except (
                OSError,
                subprocess.CalledProcessError,
                json.JSONDecodeError,
                QualificationError,
            ) as error:
                state.update(
                    state="unqualified", reason=f"cell execution unavailable: {error}"
                )
            else:
                state.update(state="measured", reason="new measured record")
        states.append(state)
    measured_records = []
    for cell, state in zip(cells, states, strict=True):
        if state["state"] != "measured":
            continue
        path = _record_path(output_dir, cell)
        measured_records.append(json.loads(path.read_text(encoding="utf-8")))
    qualification = evaluate_qualification(cells, measured_records)
    result = {
        "schema": "symmetrix.oam-release-qualification-orchestration",
        "version": SCHEMA_VERSION,
        "cells": states,
        "resumed_cells": resumed,
        "qualification": qualification,
    }
    (output_dir / "orchestration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "qualification.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = orchestrate(manifest, args.output_dir, execute=args.execute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
