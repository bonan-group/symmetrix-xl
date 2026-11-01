"""Reusable STO40 CPU/FP32 EF transactions and whole-case deadline control.

No release identities, asset approvals, exports, environment setup or model load
at import. A task-owned launcher supplies verified assets/native-first stack,
retained STO40 geometry, an absolute deadline, and mandatory in-owner route audit.
Both routes time native graph/geometry lifecycle plus EF materialization, not
pure preprepared kernels. No ASE result caching or implicit stress path is used.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
import traceback
from pathlib import Path

FAMILIES = ("mace-omat-0-medium", "mace-mh-1", "oam-s")
PROPERTIES = ("energy", "forces")
BOUNDARY = (
    "retained PYTHON neighbors and fixed xyz; native graph/CSR preparation/rebuild "
    "included (OAM every call), MACE topology reuse verified by generation/skip; "
    "forward+reverse, energy/atom-force reductions and Python EF materialization; "
    "no stress, ASE result cache, neighbor construction, compilation or diagnostics"
)


class Deadline:
    def __init__(self, entered, budget, *, clock=time.monotonic):
        self.end = entered + budget
        self.clock = clock

    def remaining(self):
        remaining = self.end - self.clock()
        if remaining <= 0:
            raise TimeoutError("whole-family deadline expired")
        return remaining


def durable_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def journal(output, stage, **record):
    with (output / "partial.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"stage": stage, "monotonic": time.monotonic(), **record},
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def process_identity(pid):
    """Capture PID start-time, session and process-group ownership facts."""
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    if len(stat) < 6:
        raise RuntimeError(f"cannot read process identity for {pid}")
    return {
        "pid": pid,
        "starttime": stat[21],
        "session": int(stat[5]),
        "pgid": int(stat[4]),
    }


def reservation_matches(identity):
    try:
        current = process_identity(identity["pid"])
        return current == identity and os.getpgid(identity["pid"]) == identity["pgid"]
    except (FileNotFoundError, ProcessLookupError, OSError, ValueError):
        return False


def apply_late_stop(result, output, stop_signal):
    """Persist a pending stop discovered after the primary evidence writes."""
    if stop_signal is None:
        return result
    result["status"] = "failed"
    result["late_stop_signal"] = stop_signal
    result["primary_error"] = (
        result.get("primary_error") or f"late stop: signal {stop_signal}"
    )
    try:
        durable_json(output / "controller-late-stop.json", dict(result))
    except (OSError, ValueError, TypeError) as error:
        result.setdefault("write_errors", []).append(
            f"controller-late-stop.json: {error!r}"
        )
    return result


def supervise(command, output, entered, budget=55.0):
    """Bound one owned worker group, including interrupted spawn/cleanup.

    TERM/INT only set a pending flag, including between Popen and assignment.
    The independent controller polls at <=50ms; it never blocks signals in the
    child or uses preexec_fn. SIGKILL of this supervisor is uncatchable and
    CANNOT guarantee cleanup/reporting. External complete-wall/exit evidence
    is required in addition to this pre-report elapsed stamp.
    """
    result = {
        "status": "failed",
        "budget_seconds": budget,
        "command": command,
        "primary_error": None,
        "cleanup_errors": [],
        "reap_error": None,
        "write_errors": [],
        "returncode": None,
        "worker_reaped": False,
    }
    process, stop_signal, identity = None, None, None
    handlers = {}

    def pending_stop(signum, frame):
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = signum

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, pending_stop)
        if stop_signal is not None:
            result["primary_error"] = (
                f"stop requested before spawn: signal {stop_signal}"
            )
            return result
        if time.monotonic() >= entered + budget - 1.0:
            raise TimeoutError("setup exhausted whole-family budget")
        with (
            (output / "worker.stdout").open("xb") as stdout,
            (output / "worker.stderr").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, start_new_session=True
            )
            identity = process_identity(process.pid)
            while stop_signal is None:
                remaining = entered + budget - 1.0 - time.monotonic()
                if remaining <= 0:
                    break
                info = os.waitid(
                    os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
                if info is not None and info.si_pid == process.pid:
                    result["reserved_exit"] = {
                        "code": info.si_code,
                        "status": info.si_status,
                    }
                    break
                time.sleep(min(0.05, remaining))
            if stop_signal is not None:
                result["primary_error"] = f"stop requested: signal {stop_signal}"
            elif "reserved_exit" not in result:
                result.update(
                    status="timeout",
                    primary_error="whole-family worker deadline expired",
                )
            else:
                worker = (
                    json.loads((output / "worker.json").read_text())
                    if (output / "worker.json").is_file()
                    else {}
                )
                if worker.get("status") == "passed":
                    result["status"] = "passed"
                else:
                    result["primary_error"] = "worker science/cleanup did not pass"
    except TimeoutError as error:
        result.update(status="timeout", primary_error=str(error))
    except BaseException:  # noqa: BLE001 - preserve primary failure while still owning cleanup
        result.update(status="failed", primary_error=traceback.format_exc())
    finally:
        if process is not None:
            if identity is None or not reservation_matches(identity):
                result["cleanup_errors"].append(
                    "worker ownership reservation is stale or unavailable"
                )
            else:
                try:
                    os.killpg(identity["pgid"], signal.SIGKILL)
                    if result["status"] == "passed":
                        result["cleanup_errors"].append(
                            "exited worker left owned descendants"
                        )
                except ProcessLookupError:
                    pass
                except OSError as error:
                    result["cleanup_errors"].append(f"killpg: {error!r}")
            try:
                process.wait(timeout=max(0.01, entered + budget - time.monotonic()))
            except (OSError, subprocess.SubprocessError) as error:
                result["reap_error"] = repr(error)
            result.update(
                returncode=process.returncode,
                worker_reaped=process.returncode is not None,
            )
            if identity is None:
                result["process_group_gone"] = None
                result["cleanup_errors"].append(
                    "no process reservation for group query"
                )
            else:
                try:
                    os.killpg(identity["pgid"], 0)
                    result["process_group_gone"] = False
                    result["cleanup_errors"].append("owned process group still exists")
                except ProcessLookupError:
                    result["process_group_gone"] = True
                except OSError as error:
                    result["process_group_gone"] = None
                    result["cleanup_errors"].append(f"group query: {error!r}")
        result["stop_signal"] = stop_signal
        if stop_signal is not None or result["cleanup_errors"] or result["reap_error"]:
            result["status"] = "failed"
        result["whole_case_pre_report_wall_seconds"] = time.monotonic() - entered
        if result["whole_case_pre_report_wall_seconds"] >= budget:
            result["status"] = "timeout"
        # Keep stop handlers nonraising through BOTH independent evidence writes.
        # If either destination fails, return errors for the external stdout/
        # exit record too. Never overwrite a partial or retry it as a success.
        try:
            durable_json(output / "controller.json", result)
        except (OSError, ValueError, TypeError) as error:
            result["write_errors"].append(f"controller.json: {error!r}")
            result["status"] = "failed"
        try:
            journal(output, "controller-final", result=result)
        except (OSError, ValueError, TypeError) as error:
            result["write_errors"].append(f"partial.jsonl: {error!r}")
            result["status"] = "failed"
        stop_signal = stop_signal or result.get("late_stop_signal")
        apply_late_stop(result, output, stop_signal)
        for signum, handler in handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError) as error:
                result["cleanup_errors"].append(f"restore signal handler: {error!r}")
                result["status"] = "failed"
    return result


def validate_work(before, after):
    if after["work"] <= before["work"]:
        raise RuntimeError("no actual evaluator work advanced (cached result)")
    if before["fallback"] != 0 or after["fallback"] != 0:
        raise RuntimeError("direct execution observed fallback")
    for key, value in before.get("generated", {}).items():
        if after["generated"][key] <= value:
            raise RuntimeError(f"generated forward/reverse work did not advance: {key}")
    if before.get("graph_policy") == "rebuild":
        if after["graph_generation"] != before["graph_generation"] + 1:
            raise RuntimeError("OAM graph generation did not advance exactly once")
        if (
            before["actual_edges"] <= 0
            or after["actual_edges"] != before["actual_edges"]
        ):
            raise RuntimeError("native OAM edge count changed")
    elif before.get("graph_policy") == "retained":
        if after["graph_generation"] <= 0 or (
            before["graph_generation"]
            and before["graph_generation"] != after["graph_generation"]
        ):
            raise RuntimeError("MACE retained graph generation changed")
        if after["skip_count"] != before["skip_count"] + 1:
            raise RuntimeError("MACE topology validation skip did not advance once")
        if after["work"] != before["work"] + 1:
            raise RuntimeError("MACE prepared evaluation count did not advance once")


def transaction(compute, snapshot, deadline):
    deadline.remaining()
    before = snapshot()
    started = time.perf_counter()
    values = compute()
    seconds = time.perf_counter() - started
    after = snapshot()
    validate_work(before, after)
    deadline.remaining()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid sample duration")
    return {"values": values, "seconds": seconds, "before": before, "after": after}


def compare_ef(reference, direct, family):
    for values in (reference, direct):
        forces = values["forces"]
        if len(forces) != 40 or any(len(row) != 3 for row in forces):
            raise ValueError("EF output shape differs from STO40")
        if not all(
            math.isfinite(float(x))
            for x in [values["energy"], *[v for row in forces for v in row]]
        ):
            raise ValueError("nonfinite EF output")
    errors = {
        "energy_abs_eV_per_atom": abs(direct["energy"] - reference["energy"]) / 40,
        "forces_max_abs_eV_per_A": max(
            abs(a - b)
            for ra, rb in zip(reference["forces"], direct["forces"])
            for a, b in zip(ra, rb)
        ),
    }
    # Maintained foundation_model_support_manifest.json FP32 family bounds.
    limits = (1e-5, 5e-4) if family == "oam-s" else (5e-4, 5e-3)
    if any(value > limit for value, limit in zip(errors.values(), limits)):
        raise RuntimeError(f"EF parity failed: {errors}")
    return {"errors": errors, "limits": dict(zip(errors, limits))}


def summarize(samples):
    result = {}
    for route in ("direct", "reference"):
        values = samples[route]
        if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(
                "three positive finite measured samples required per route"
            )
        result[route] = {
            "samples_seconds": values,
            "median_us_per_atom": statistics.median(values) * 1e6 / 40,
        }
    result["symmetrix_reference_over_direct"] = statistics.median(
        samples["reference"]
    ) / statistics.median(samples["direct"])
    return result


def snapshot(evaluator, family, route):
    if family == "oam-s":
        # These native counters are cumulative; no calculator-local call count.
        import re

        diagnostic = str(evaluator.execution_diagnostic)
        counts = {
            key: int(value)
            for key, value in re.findall(r"\b(edge_[a-z_]+)=(\d+)", diagnostic)
        }
        required = {
            "edge_dispatches",
            "edge_generated_dispatches",
            "edge_generated_regions",
            "edge_full_message_launches",
            "edge_generic_dispatches",
            "edge_generic_regions",
        }
        if not required.issubset(counts):
            raise RuntimeError(
                "missing native OAM counter text (no default post-work zeros)"
            )
        work = counts["edge_dispatches"]
        if route == "direct" and (
            counts["edge_generic_dispatches"] or counts["edge_generic_regions"]
        ):
            raise RuntimeError("OAM direct generic fallback")
        generated = (
            {
                key: counts[key]
                for key in (
                    "edge_generated_dispatches",
                    "edge_generated_regions",
                    "edge_full_message_launches",
                )
            }
            if route == "direct"
            else {}
        )
        return {
            "work": work,
            "fallback": int(evaluator.cuda_r1_fallback_count),
            "counters": counts,
            "generated": generated,
            "graph_generation": int(evaluator.graph_generation),
            "graph_policy": "rebuild",
            "actual_edges": int(evaluator.execution_active_edge_count),
        }
    counters = {
        name: int(getattr(evaluator, name))
        for name in (
            "execution_sphericart_launch_count",
            "factorized_prepared_evaluation_count",
            "factorized_fallback_evaluation_count",
        )
    }
    counters["factorized_graph_generation"] = int(evaluator.factorized_graph_generation)
    counters["factorized_topology_validation_skip_count"] = int(
        evaluator.factorized_topology_validation_skip_count
    )
    if route == "direct":
        for name in (
            (
                "execution_mh1_generated_forward_launch_count",
                "execution_mh1_generated_source_reverse_launch_count",
                "execution_mh1_generated_edge_reverse_launch_count",
            )
            if family == "mace-mh-1"
            else (
                "factorized_jit_forward_launch_count",
                "factorized_jit_reverse_launch_count",
            )
        ):
            counters[name] = int(getattr(evaluator, name))
        work = counters["factorized_prepared_evaluation_count"]
    else:
        work = counters["execution_sphericart_launch_count"]
    generated = {
        key: value
        for key, value in counters.items()
        if "generated" in key or "jit_" in key
    }
    return {
        "work": work,
        "fallback": counters["factorized_fallback_evaluation_count"],
        "counters": counters,
        "generated": generated,
        "graph_policy": "retained" if route == "direct" else "reference-whole",
        "graph_generation": counters["factorized_graph_generation"],
        "skip_count": counters["factorized_topology_validation_skip_count"],
    }


def initialize_oam_counters(evaluator, graph_arguments, route):
    """No evaluation: native prepare_graph exposes real initial counters.

    The constructor's state-only diagnostic has no edge_* fields. Never infer
    missing post-work counters as zero. This one untimed setup is within case
    setup; each subsequent timed OAM call STILL includes its own CSR rebuild.
    """
    if int(evaluator.graph_generation) != 0 or int(evaluator.geometry_generation) != 0:
        raise RuntimeError("expected a fresh OAM evaluator before graph setup")
    generation = evaluator.prepare_graph(*graph_arguments)
    initial = snapshot(evaluator, "oam-s", route)
    if (
        initial["graph_generation"] != generation
        or generation != 1
        or initial["work"] != 0
        or initial["fallback"] != 0
        or any(initial["counters"].values())
    ):
        raise RuntimeError("untimed fresh OAM graph setup unexpectedly evaluated work")
    return initial


def require_same_graph(direct, reference):
    for field in (
        "cutoff_A",
        "active_native_input_graph_count",
        "active_native_input_graph_sha256",
    ):
        if direct[field] != reference[field]:
            raise RuntimeError(
                f"routes did not retain the same ordered active native graph: {field}"
            )


def graph_hash(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def graph_identity(values):
    return {"sha256": graph_hash(values), "count": len(values[0]) if values else 0}


def route_case(model_path, family, route, atoms, deadline, output, *, audit_route):
    """Same timed logical work per route; no public calculate/cache/stress path."""
    import numpy as np

    from symmetrix import NequIPCalculator, Symmetrix

    calc = evaluator = compute = None
    try:
        cls = NequIPCalculator if family == "oam-s" else Symmetrix
        options = {
            "dtype": "float32",
            "neighbor_skin": 0.5,
            "streamed_edges": route,
            "execution_profile": "speed",
        }
        if family == "mace-mh-1":
            options["head"] = "omat_pbe"
        calc = cls(model_path, **options)
        evaluator = calc.evaluator
        expected_mode = (
            "generic" if route == "reference" and family != "oam-s" else route
        )
        if evaluator.streamed_edges_mode != expected_mode:
            raise RuntimeError("requested direct/reference route was not selected")
        if family == "oam-s":
            if type(evaluator).__name__ != "NequIPOAMKokkosFloat":
                raise RuntimeError("OAM native evaluator is not FP32")
        elif evaluator.scalar_size_bytes != 4:
            raise RuntimeError("MACE native evaluator is not FP32")
        if family == "oam-s":
            graph = calc._build_graph(atoms)
            types, degrees, sources, neigh_types, receivers, _, xyz, distances = graph
            retained_skin_values = [np.asarray(value).tolist() for value in graph]
            retained_skin_hash = graph_hash(retained_skin_values)
            retained_skin_count = len(sources)
            cutoff = float(evaluator.r_cut)
            active = np.asarray(distances) < cutoff
            receivers = np.asarray(receivers)[active]
            sources = np.asarray(sources)[active]
            neigh_types = np.asarray(neigh_types)[active]
            xyz = np.asarray(xyz).reshape(-1, 3)[active].reshape(-1)
            distances = np.asarray(distances)[active]
            degrees = np.bincount(receivers, minlength=40)
            positions = np.ascontiguousarray(atoms.positions).reshape(-1)
            cell = np.asarray(atoms.cell.array, dtype=float)
            inverse = np.linalg.inv(cell).reshape(-1)
            pbc = np.asarray(atoms.pbc, dtype=np.int32)
            initial_counters = initialize_oam_counters(
                evaluator, (40, types, degrees, sources, neigh_types), route
            )

            def compute():
                if route == "direct":
                    generation = evaluator.prepare_graph(
                        40, types, degrees, sources, neigh_types
                    )
                    evaluator.prepare_geometry(
                        generation,
                        positions,
                        xyz,
                        cell.reshape(-1),
                        inverse,
                        pbc,
                    )
                    evaluator.compute_prepared(generation, xyz, distances)
                    evaluator.reduce_forces(generation)
                    forces = (
                        np.asarray(evaluator.atom_forces, dtype=float)
                        .reshape(40, 3)
                        .copy()
                    )
                else:
                    evaluator.compute_node_energies_forces(
                        40, types, degrees, sources, neigh_types, xyz, distances
                    )
                    pairs = np.asarray(evaluator.node_forces, dtype=float).reshape(
                        -1, 3
                    )
                    forces = np.zeros((40, 3))
                    # Preserve the reference calculator's reduction/order,
                    # rather than introducing a benchmark-only fast baseline.
                    for edge, (receiver, source) in enumerate(zip(receivers, sources)):
                        forces[source] += pairs[edge]
                        forces[receiver] -= pairs[edge]
                return {
                    "energy": float(
                        np.asarray(evaluator.node_energies, dtype=float).sum()
                    ),
                    "forces": forces.tolist(),
                }

            edge_count = len(sources)
        else:
            inputs = calc._mace_inputs(
                atoms, native_geometry=False, allow_device_neighbor_graph=False
            )
            retained_skin_values = [np.asarray(value).tolist() for value in inputs]
            retained_skin_hash = graph_hash(retained_skin_values)
            retained_skin_count = len(inputs[3])
            # Direct uses a skin-padded/clamped graph; reference filters it.
            # Supply the SAME active physical graph to both supported native
            # APIs, excluding only zero-contribution r>=cutoff skin entries.
            n, types, _, sources, neigh_types, xyz, distances, receivers = inputs
            active = np.asarray(distances) < calc.cutoff
            receivers = np.asarray(receivers)[active]
            sources = np.asarray(sources)[active]
            inputs = (
                n,
                types,
                np.bincount(receivers, minlength=n),
                sources,
                np.asarray(neigh_types)[active],
                np.asarray(xyz)[active],
                np.asarray(distances)[active],
                receivers,
            )
            initial_counters = None

            def compute():
                generation = calc._compute_mace(inputs, atoms=atoms)
                values = calc._collect_mace_results(
                    atoms, inputs, PROPERTIES, generation or 0
                )
                return {
                    "energy": float(values["energy"]),
                    "forces": np.asarray(values["forces"]).tolist(),
                }

            edge_count = len(inputs[3])
            cutoff = float(calc.cutoff)
        if route == "reference" and (
            getattr(calc, "jit_artifact_path", None)
            or getattr(evaluator, "jit_oam_host_plugin_ready", False)
            or getattr(evaluator, "factorized_jit_ready", False)
        ):
            raise RuntimeError("reference must be artifact-free")
        if not math.isfinite(cutoff) or cutoff <= 0 or edge_count <= 0:
            raise RuntimeError("invalid actual model cutoff/graph")
        graph_values = (
            [
                np.asarray(value).tolist()
                for value in (
                    types,
                    degrees,
                    sources,
                    neigh_types,
                    receivers,
                    xyz,
                    distances,
                )
            ]
            if family == "oam-s"
            else [np.asarray(value).tolist() for value in inputs]
        )
        graph_sha = graph_hash(graph_values)
        if (
            initial_counters is not None
            and initial_counters["actual_edges"] != edge_count
        ):
            raise RuntimeError(
                "native OAM graph count differs from captured Python graph"
            )
        pre_evidence = audit_route(calc, family, route)
        journal(
            output,
            "route-admitted-before-work",
            route=route,
            evidence=pre_evidence,
            initial_counters=initial_counters,
        )
        rows = []
        for phase, count in (("parity", 1), ("warmup", 3), ("measurement", 3)):
            for index in range(count):
                row = transaction(
                    compute, lambda: snapshot(evaluator, family, route), deadline
                )
                rows.append({"phase": phase, "index": index, **row})
                journal(
                    output, "transaction", route=route, phase=phase, index=index, **row
                )
        if route == "direct":
            counters = rows[-1]["after"]["counters"]
            generated = [
                value
                for key, value in counters.items()
                if "generated" in key or "jit_" in key
            ]
            if not generated or not all(value > 0 for value in generated):
                raise RuntimeError("generated forward/reverse work was not observed")
        after_graph_values = graph_values
        if (
            hashlib.sha256(
                json.dumps(after_graph_values, sort_keys=True).encode()
            ).hexdigest()
            != graph_sha
        ):
            raise RuntimeError(
                "the retained static Python graph changed during native work"
            )
        from ase.neighborlist import neighbor_list

        evidence = audit_route(calc, family, route)
        return {
            "route": route,
            "native_mode": expected_mode,
            "precision": "float32",
            "route_evidence": evidence,
            "route_evidence_before_work": pre_evidence,
            "untimed_native_graph_preparations": 1 if family == "oam-s" else 0,
            "retained_python_graph_sha256": graph_sha,
            "retained_skin_cache_graph_sha256": retained_skin_hash,
            "retained_skin_cache_graph_count": retained_skin_count,
            "active_native_input_graph_sha256": graph_sha,
            "active_native_input_graph_count": edge_count,
            "python_graph_builds": 1,
            "cutoff_A": cutoff,
            "skin_A": 0.5,
            "effective_cutoff_A": cutoff + 0.5,
            "actual_graph_edges": edge_count,
            "physical_directed_edges": len(neighbor_list("i", atoms, cutoff)),
            "modeled_skin_directed_edges": len(neighbor_list("i", atoms, cutoff + 0.5)),
            "transactions": rows,
        }
    finally:
        compute = evaluator = calc = None
        gc.collect()
