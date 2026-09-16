"""Explicit bounded calibration for generated Execution persistent launch grids."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np
from ase.calculators.calculator import all_changes

from .calculator import Symmetrix
from .kernel_launch_tuning import (
    publish_kernel_launch_tuning_record,
    kernel_launch_tuning_identity,
    kernel_launch_workload_identity,
)


@dataclass(frozen=True)
class KernelLaunchCalibrationResult:
    cache_key: str
    published: bool
    decisions: tuple[dict, ...]
    measurements: dict
    diagnostics: dict


def _snapshot(results, properties):
    snapshot = {}
    for name in properties:
        if name not in results:
            continue
        value = results[name]
        snapshot[name] = np.array(value, copy=True) if np.ndim(value) else float(value)
    return snapshot


def _outputs_match(reference, candidate, *, rtol, atol):
    if set(reference) != set(candidate):
        return False
    return all(
        np.allclose(reference[name], candidate[name], rtol=rtol, atol=atol)
        for name in reference
    )


def _evaluate(calculator, atoms, properties):
    start = time.perf_counter()
    calculator.calculate(atoms, properties=list(properties), system_changes=all_changes)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    return elapsed_ms, _snapshot(calculator.results, properties)


def _prepared_evaluator(calculator, atoms, properties):
    mace_inputs = calculator._mace_inputs(atoms)
    if not all(
        callable(getattr(calculator, name, None))
        for name in (
            "_has_native_field_coupling",
            "_compute_mace",
            "_collect_mace_results",
        )
    ):
        return lambda: _evaluate(calculator, atoms, properties), mace_inputs

    def evaluate():
        if calculator._has_native_field_coupling():
            electric_field = calculator._resolve_electric_field()
            start = time.perf_counter()
            calculator._compute_macefield(
                atoms,
                electric_field,
                mace_inputs=mace_inputs,
            )
            response_results = {}
            response_properties = getattr(
                calculator, "_macefield_response_properties", ()
            )
            if any(name in properties for name in response_properties):
                response_results = calculator._calculate_macefield_responses(
                    atoms,
                    electric_field,
                    properties,
                    mace_inputs,
                )
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            results = calculator._collect_mace_results(atoms, mace_inputs)
            results.update(response_results)
        else:
            start = time.perf_counter()
            calculator._compute_mace(mace_inputs)
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            results = calculator._collect_mace_results(atoms, mace_inputs)
        calculator.results = results
        return elapsed_ms, _snapshot(calculator.results, properties)

    return evaluate, mace_inputs


def _candidate_groups(diagnostics, restriction):
    groups = {}
    allowed = None if restriction is None else {int(value) for value in restriction}
    for stage, diagnostic in sorted(diagnostics.items()):
        if not diagnostic.get("calibration_permitted"):
            continue
        candidates = [int(value) for value in diagnostic["calibration_candidates"]]
        if allowed is not None:
            candidates = [value for value in candidates if value in allowed]
        default = int(diagnostic["default_blocks_per_compute_unit"])
        if default not in candidates:
            candidates.append(default)
        candidates = sorted(set(candidates))
        if not candidates:
            continue
        profile_id = str(diagnostic["profile_id"])
        implementation_kind = str(diagnostic["implementation_kind"])
        if implementation_kind == "module":
            kind = "module"
        elif implementation_kind == "artifact":
            kind = "plugin"
        else:
            raise ValueError(
                "unsupported Execution launch implementation kind "
                f"{implementation_kind!r}"
            )
        key = (kind, stage if kind == "plugin" else profile_id)
        if key not in groups:
            groups[key] = {
                "kind": kind,
                "profile_id": profile_id,
                "stages": [],
                "default": default,
                "candidates": candidates,
            }
        group = groups[key]
        group["stages"].append(stage)
        group["candidates"] = sorted(set(group["candidates"]).intersection(candidates))
    return list(groups.values())


def _apply_group(evaluator, group, blocks):
    if group["kind"] == "plugin":
        evaluator._set_jit_device_plugin_launch_override(group["stages"][0], blocks)
    else:
        evaluator._set_kernel_launch_profile_override(group["profile_id"], blocks)


def calibrate_kernel_launches(
    model,
    atoms,
    *,
    precision="float32",
    streamed_edges="direct",
    jit=None,
    properties=("energy", "forces", "stress"),
    warmups=1,
    samples=3,
    confirmation_samples=2,
    time_budget_seconds=120.0,
    minimum_improvement=0.03,
    candidate_restriction=None,
    rtol=2e-5,
    atol=2e-6,
    cache_root=None,
):
    """Calibrate bounded runtime grid multipliers and publish one advisory record."""

    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer of at least two")
    if (
        isinstance(confirmation_samples, bool)
        or not isinstance(confirmation_samples, int)
        or confirmation_samples < 1
    ):
        raise ValueError("confirmation_samples must be a positive integer")
    if time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be positive")
    if minimum_improvement < 0:
        raise ValueError("minimum_improvement must be non-negative")
    properties = tuple(dict.fromkeys(str(value) for value in properties))
    if not properties:
        raise ValueError("properties must not be empty")

    calculator = Symmetrix(
        model,
        dtype=precision,
        use_kokkos=True,
        streamed_edges=streamed_edges,
        jit=jit,
        kernel_launch_policy="static",
    )
    atoms = atoms.copy()
    _, oracle = _evaluate(calculator, atoms, properties)
    evaluate_prepared, mace_inputs = _prepared_evaluator(calculator, atoms, properties)
    baseline_ms, baseline = evaluate_prepared()
    if not _outputs_match(oracle, baseline, rtol=rtol, atol=atol):
        raise RuntimeError(
            "prepared Execution launch baseline does not match ASE output"
        )
    for _ in range(warmups):
        _, repeated = evaluate_prepared()
        if not _outputs_match(baseline, repeated, rtol=rtol, atol=atol):
            raise RuntimeError("static Execution launch baseline is not deterministic")

    diagnostics = calculator.kernel_launch_profile_diagnostics
    groups = _candidate_groups(diagnostics, candidate_restriction)
    if not groups:
        raise RuntimeError("no tuneable persistent Execution launch stage is active")
    calculator.evaluator._set_kernel_launch_policy("automatic")
    deadline = time.monotonic() + float(time_budget_seconds)
    measurements = {
        "baseline_ms": baseline_ms,
        "minimum_improvement": float(minimum_improvement),
        "groups": [],
    }
    decisions = []

    for group in groups:
        default = group["default"]
        samples_by_candidate = {value: [] for value in group["candidates"]}
        rejected = {}
        for round_index in range(samples):
            order = list(group["candidates"])
            if round_index % 2:
                order.reverse()
            for candidate in order:
                if time.monotonic() >= deadline:
                    rejected.setdefault(candidate, "time budget exhausted")
                    continue
                try:
                    _apply_group(calculator.evaluator, group, candidate)
                    elapsed_ms, output = evaluate_prepared()
                except (RuntimeError, ValueError) as error:
                    rejected[candidate] = f"{type(error).__name__}: {error}"
                    continue
                if not _outputs_match(baseline, output, rtol=rtol, atol=atol):
                    rejected[candidate] = "output parity failure"
                    continue
                samples_by_candidate[candidate].append(elapsed_ms)

        valid = {
            candidate: values
            for candidate, values in samples_by_candidate.items()
            if len(values) == samples and candidate not in rejected
        }
        if default not in valid:
            if rejected.get(default) != "time budget exhausted":
                raise RuntimeError(
                    f"static launch candidate {default} did not complete calibration"
                )
            medians = {}
            noise_fraction = None
            provisional = default
            improvement = 0.0
            required = float(minimum_improvement)
            winner = default
        else:
            medians = {
                candidate: statistics.median(values)
                for candidate, values in valid.items()
            }
            default_median = medians[default]
            default_mad = statistics.median(
                abs(value - default_median) for value in valid[default]
            )
            noise_fraction = (
                default_mad / default_median if default_median else math.inf
            )
            provisional = min(medians, key=medians.get)
            improvement = (
                (default_median - medians[provisional]) / default_median
                if default_median
                else 0.0
            )
            required = max(float(minimum_improvement), 2.0 * noise_fraction)
            winner = provisional if improvement > required else default

        confirmation = {default: [], winner: []}
        confirmation_complete = True
        if winner != default:
            for round_index in range(confirmation_samples):
                order = [default, winner]
                if round_index % 2:
                    order.reverse()
                for candidate in order:
                    if time.monotonic() >= deadline:
                        confirmation_complete = False
                        break
                    _apply_group(calculator.evaluator, group, candidate)
                    elapsed_ms, output = evaluate_prepared()
                    if not _outputs_match(baseline, output, rtol=rtol, atol=atol):
                        raise RuntimeError(
                            "provisional Execution launch winner failed confirmation parity"
                        )
                    confirmation[candidate].append(elapsed_ms)
                if not confirmation_complete:
                    break
            if confirmation_complete:
                confirmed_default = statistics.median(confirmation[default])
                confirmed_winner = statistics.median(confirmation[winner])
                confirmed_improvement = (
                    (confirmed_default - confirmed_winner) / confirmed_default
                    if confirmed_default
                    else 0.0
                )
            else:
                confirmed_improvement = None
            if not confirmation_complete or confirmed_improvement <= required:
                winner = default
        _apply_group(calculator.evaluator, group, winner)
        decisions.append(
            {
                "stage": group["stages"][0],
                "profile_id": group["profile_id"],
                "kind": group["kind"],
                "blocks_per_compute_unit": winner,
            }
        )
        measurements["groups"].append(
            {
                "stages": group["stages"],
                "profile_id": group["profile_id"],
                "default": default,
                "winner": winner,
                "samples_ms": {
                    str(candidate): values
                    for candidate, values in samples_by_candidate.items()
                },
                "medians_ms": {
                    str(candidate): value for candidate, value in medians.items()
                },
                "rejected": {
                    str(candidate): reason for candidate, reason in rejected.items()
                },
                "noise_fraction": noise_fraction,
                "required_improvement": required,
                "provisional_improvement": improvement,
                "confirmation_ms": {
                    str(candidate): values for candidate, values in confirmation.items()
                },
                "confirmation_complete": confirmation_complete,
            }
        )

    _, final_output = evaluate_prepared()
    if not _outputs_match(baseline, final_output, rtol=rtol, atol=atol):
        raise RuntimeError("combined Execution launch decision failed final parity")
    num_nodes, _, _, j_list, _, _, _, _ = mace_inputs
    environment = calculator.evaluator.execution_device_execution_environment
    identity = kernel_launch_tuning_identity(
        device_environment=environment,
        implementation_identity=calculator._kernel_launch_implementation_identity(),
        model_identity=calculator._kernel_launch_model_identity(),
        precision=precision,
        workload=kernel_launch_workload_identity(
            int(num_nodes), len(j_list), properties
        ),
    )
    publication = publish_kernel_launch_tuning_record(
        identity,
        decisions,
        measurements,
        cache_root=cache_root,
    )
    return KernelLaunchCalibrationResult(
        cache_key=publication.cache_key,
        published=publication.published,
        decisions=tuple(decisions),
        measurements=measurements,
        diagnostics=diagnostics,
    )


def calibration_result_dict(result):
    return asdict(result)


__all__ = [
    "KernelLaunchCalibrationResult",
    "calibrate_kernel_launches",
    "calibration_result_dict",
]
