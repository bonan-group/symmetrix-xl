#!/usr/bin/env python3
"""Compare experimental CPU receiver factorization with direct streaming."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


def _configure_one_thread() -> dict[str, Any]:
    for name in (
        "KOKKOS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    if affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {affinity[0]})
        affinity = [affinity[0]]
    return {
        "affinity": affinity,
        "environment": {
            name: os.environ[name]
            for name in (
                "KOKKOS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
    }


_THREADS = _configure_one_thread()

import numpy as np  # noqa: E402 - one-thread pinning must precede NumPy import


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_atoms(structure: pathlib.Path | None, repeat: int):
    if structure is not None:
        from ase.io import read

        atoms = read(structure)
        description = {"kind": "file", "path": str(structure.resolve())}
    else:
        from ase.build import bulk

        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
        description = {
            "kind": "wurtzite_AlN",
            "a_A": 3.112,
            "c_A": 4.982,
            "repeat": [repeat, repeat, repeat],
        }
    description["atoms"] = len(atoms)
    return atoms, description


def _prepare_native(model: pathlib.Path, structure: pathlib.Path | None, repeat: int):
    from symmetrix import Symmetrix

    atoms, system = _case_atoms(structure, repeat)
    calculator = Symmetrix(
        model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
        neighbor_skin=0.0,
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
    token = int(evaluator._prepare_factorized_graph(*inputs[:5]))
    xyz = np.ascontiguousarray(inputs[5], dtype=np.float64)
    radius = np.ascontiguousarray(inputs[6], dtype=np.float64)
    evaluator._compute_prepared_factorized(token, xyz.reshape(-1), radius)
    session = int(evaluator._prepare_factorized_operator_benchmark(token, xyz, radius))
    arrays = {
        str(name): np.ascontiguousarray(value)
        for name, value in evaluator._factorized_operator_export(session).items()
    }
    return evaluator, session, arrays, system, float(calculator.cutoff)


def _compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual - expected)
    threshold = 3.0e-5 + 2.0e-5 * np.abs(expected)
    return {
        "max_abs_error": float(difference.max(initial=0.0)),
        "max_scaled_error": float(np.max(difference / threshold, initial=0.0)),
        "allclose": bool(np.all(difference <= threshold)),
    }


def _native_measure(
    evaluator: Any, session: int, mode: str, warmups: int, repeats: int
) -> dict[str, Any]:
    measured = evaluator._measure_factorized_operator_benchmark(
        session, mode, warmups, 0.0, repeats, repeats, 0.0
    )
    return {
        "samples_ms": [float(value) for value in measured["samples_ms"]],
        "median_ms": float(measured["median_ms"]),
        "min_ms": float(measured["min_ms"]),
        "max_ms": float(measured["max_ms"]),
        "timing_backend": measured["timing_backend"],
    }


def run(
    model: pathlib.Path,
    *,
    structure: pathlib.Path | None,
    repeat: int,
    receiver_chunk: int,
    warmups: int,
    repeats: int,
    cache_root: pathlib.Path | None,
) -> dict[str, Any]:
    from symmetrix.receiver_factorized_rtc import ReceiverFactorizedRtc

    model = model.resolve()
    model_data = json.loads(model.read_text(encoding="utf-8"))
    contract = model_data["execution_contracts"]["R1"]
    evaluator, session, arrays, system, model_cutoff = _prepare_native(
        model, structure, repeat
    )
    rtc = ReceiverFactorizedRtc(
        contract, arrays, receiver_chunk=receiver_chunk, cache_root=cache_root
    )
    outputs = rtc.run("forward_reverse")
    validation = {
        "output": _compare(outputs.output, arrays["expected_out"]),
        "source_adjoint": _compare(outputs.source_adjoint, arrays["expected_grad_x"]),
        "directed_force": _compare(
            outputs.directed_force, arrays["expected_directed_force"]
        ),
    }
    status = "pass" if all(item["allclose"] for item in validation.values()) else "fail"
    nodes = arrays["x"].shape[0]
    edges = arrays["edge_index"].shape[1]
    measurements: dict[str, Any] = {}
    for public_mode, native_mode in (
        ("forward", "forward"),
        ("reverse", "reverse"),
        ("forward_reverse", "forward_reverse"),
    ):
        direct = _native_measure(evaluator, session, native_mode, warmups, repeats)
        receiver = rtc.measure(public_mode, warmups=warmups, repeats=repeats)
        for timing in (direct, receiver):
            timing["median_us_per_atom"] = timing["median_ms"] * 1000.0 / nodes
        measurements[public_mode] = {
            "direct_streamed": direct,
            "receiver_factorized": receiver,
            "receiver_over_direct": receiver["median_ms"] / direct["median_ms"],
        }
    result = {
        "schema": "symmetrix.experimental.receiver-factorized-cpu/1",
        "status": status,
        "algorithm_decision": (
            "retain_experimental" if status == "pass" else "reject_incorrect"
        ),
        "model": {
            "path": str(model),
            "sha256": _sha256(model),
            "contract_fingerprint": contract.get("fingerprint"),
            "connection_modes": sorted(
                {path["connection_mode"] for path in contract["paths"]}
            ),
            "channels": contract["channels"],
            "radial_embedding": contract["radial_embedding"],
        },
        "workload": {
            **system,
            "directed_edges": edges,
            "model_cutoff_A": model_cutoff,
            "neighbor_skin_A": 0.0,
            "effective_neighbor_cutoff_A": model_cutoff,
        },
        "threads": _THREADS,
        "implementations": {
            "direct_streamed": {
                "kind": "native_host_RTC",
                "description": "generated edge-direct R1 plus dense A1",
            },
            "receiver_factorized": {
                "kind": "experimental_host_RTC_OpenBLAS",
                "receiver_chunk": receiver_chunk,
                "artifact_status": rtc.build_result.status,
                "artifact_path": str(rtc.build_result.artifact_path),
                "workspace_bytes": int(rtc.workspace.nbytes),
                "description": (
                    "bounded receiver radial-coupling state plus composed projection"
                ),
            },
        },
        "validation": validation,
        "measurements": measurements,
    }
    forward_reverse = measurements["forward_reverse"]
    if status == "pass" and forward_reverse["receiver_over_direct"] <= 1.1:
        result["algorithm_decision"] = "candidate_for_production_integration"
    elif status == "pass":
        result["algorithm_decision"] = "do_not_promote"
    return result


def _write_report(path: pathlib.Path | None, report: Mapping[str, Any]) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    system = parser.add_mutually_exclusive_group()
    system.add_argument("--structure", type=pathlib.Path)
    system.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--receiver-chunk", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(
            args.model,
            structure=args.structure,
            repeat=args.repeat,
            receiver_chunk=args.receiver_chunk,
            warmups=args.warmups,
            repeats=args.repeats,
            cache_root=args.cache_root,
        )
        _write_report(args.output, report)
        return int(report["status"] != "pass")
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
