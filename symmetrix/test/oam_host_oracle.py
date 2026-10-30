"""Independent host reference-routing oracle for OAM device tests.

The artifact-free portable Kokkos path now runs in a device process.  Generated
CUDA qualification still uses this child-process host oracle where it needs an
independent backend comparison: the same contract, directed graph, and family
reference routing, evaluated without sharing the CUDA process or artifact.

The oracle is a pure function of its inputs, so results are memoized for the
session.  A child process needs a host backend installed; when none is
available the caller skips with a reason that names the missing prerequisite
rather than silently comparing the device path against itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_CHILD_PROGRAM = r"""
import sys
import numpy as np

payload = np.load(sys.argv[1], allow_pickle=False)
contract = str(payload["contract"])
node_types = payload["node_types"]
counts = payload["counts"]
sources = payload["sources"]
xyz = payload["xyz"]
distances = np.linalg.norm(xyz, axis=1)

from symmetrix import load_backend

native = load_backend("cpu")
native._init_kokkos()
evaluator = native.NequIPOAMKokkos(contract)
evaluator.set_streamed_edges("reference")
evaluator.compute_node_energies_forces(
    len(node_types),
    node_types,
    counts,
    sources,
    node_types[sources],
    xyz.ravel(),
    distances,
)
directed = np.asarray(evaluator.node_forces).reshape(-1, 3)
np.savez(
    sys.argv[2],
    node_energies=np.asarray(evaluator.node_energies),
    directed_forces=directed,
)
"""

_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _cache_key(contract, node_types, counts, sources, xyz) -> str:
    digest = hashlib.sha256()
    for array in (node_types, counts, sources, xyz):
        digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(b"|")
    digest.update(str(Path(contract).resolve()).encode())
    return digest.hexdigest()


def reference_routing(contract, values, counts=None):
    """Return ``(node_energies, per_atom_forces)`` from the host reference path.

    ``values`` is ``(positions, node_types, sources, xyz)``, the same tuple the
    device tests already build.  The directed forces are accumulated into
    per-atom forces exactly as the device tests expect.
    """

    _, node_types, sources, xyz = values
    node_types = np.ascontiguousarray(node_types, dtype=np.int32)
    sources = np.ascontiguousarray(sources, dtype=np.int32)
    xyz = np.ascontiguousarray(xyz, dtype=float)
    if counts is None:
        counts = np.ones(len(node_types), dtype=np.int32)
    counts = np.ascontiguousarray(counts, dtype=np.int32)

    key = _cache_key(contract, node_types, counts, sources, xyz)
    cached = _CACHE.get(key)
    if cached is None:
        cached = _compute(contract, node_types, counts, sources, xyz)
        _CACHE[key] = cached
    node_energies, directed = cached

    forces = np.zeros((len(node_types), 3))
    np.add.at(forces, np.repeat(np.arange(len(node_types)), counts), -directed)
    np.add.at(forces, sources, directed)
    return node_energies.copy(), forces


def _compute(contract, node_types, counts, sources, xyz):
    with tempfile.TemporaryDirectory(prefix="oam-host-oracle-") as work:
        work_path = Path(work)
        payload = work_path / "payload.npz"
        result = work_path / "result.npz"
        np.savez(
            payload,
            contract=np.asarray(str(Path(contract).resolve())),
            node_types=node_types,
            counts=counts,
            sources=sources,
            xyz=xyz,
        )
        process = subprocess.run(
            [sys.executable, "-c", _CHILD_PROGRAM, str(payload), str(result)],
            capture_output=True,
            text=True,
            cwd=work,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip().splitlines()
            message = detail[-1] if detail else "no output"
            if "is not installed" in message or "no compatible CPU backend" in message:
                pytest.skip(
                    "the OAM reference-routing oracle needs a host backend "
                    f"installed: {message}"
                )
            raise AssertionError(
                "the OAM reference-routing oracle failed in a host process: " + message
            )
        with np.load(result, allow_pickle=False) as arrays:
            return (
                np.array(arrays["node_energies"], copy=True),
                np.array(arrays["directed_forces"], copy=True),
            )


def reference_evaluator_results(contract, arguments):
    """Return ``(node_energies, node_forces)`` for a raw evaluator call.

    ``arguments`` is ``(num_nodes, node_types, counts, sources, neighbor_types,
    xyz, distances)``: the positional form the OAM entry point takes, used by
    tests that drive the evaluator directly instead of the graph tuple.  The
    neighbor types are derivable from the sources on both sides, so the oracle
    reports the directed node forces the entry point returns.
    """

    node_types = np.ascontiguousarray(arguments[1], dtype=np.int32)
    counts = np.ascontiguousarray(arguments[2], dtype=np.int32)
    sources = np.ascontiguousarray(arguments[3], dtype=np.int32)
    xyz = np.ascontiguousarray(np.asarray(arguments[5], dtype=float).reshape(-1, 3))

    key = _cache_key(contract, node_types, counts, sources, xyz)
    cached = _CACHE.get(key)
    if cached is None:
        cached = _compute(contract, node_types, counts, sources, xyz)
        _CACHE[key] = cached
    return cached[0].copy(), cached[1].copy()


def require_host_execution_space(execution_space, subject):
    """Skip a host-implementation test when the loaded backend is a device.

    A device build admits neither the artifact-free nor the receiver-host OAM
    path for execution, so a test whose subject is one of those paths cannot run
    there.  The reason names the subject so the skip is not read as a coverage
    gap in the device path.
    """

    if execution_space not in ("OpenMP", "Serial"):
        pytest.skip(
            f"{subject} is a host implementation, and this build executes on "
            f"{execution_space}: every OAM execution on a device build requires "
            "the generated CUDA plugin"
        )


__all__ = [
    "reference_evaluator_results",
    "reference_routing",
    "require_host_execution_space",
]
