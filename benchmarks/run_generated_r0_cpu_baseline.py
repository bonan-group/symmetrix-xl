#!/usr/bin/env python3
"""Run the complete frozen-versus-generated R0 CPU qualification campaign.

The runner deliberately imports NumPy and Symmetrix only after checking the
one-core environment and empty task cache.  Point it at a compact two-layer
standard-MACE JSON containing R1, R0, and M0 execution contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from generated_r0_baseline import (
    PATHS,
    BaselineCase,
    evaluate_retirement_gate,
    measurement_record,
    required_cases,
    skipped_case,
    validate_campaign,
)


_THREAD_ENVIRONMENT = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_HOST_POLICY_ENVIRONMENT = (
    "CXX",
    "SYMMETRIX_JIT_HOST_TARGET",
    "SYMMETRIX_JIT_HOST_FLAGS",
    "SYMMETRIX_JIT_HOST_FP_MODE",
)


def _require_reproducible_cpu_environment(cache_root: Path) -> None:
    incorrect = [name for name in _THREAD_ENVIRONMENT if os.environ.get(name) != "1"]
    if incorrect:
        raise RuntimeError(
            "CPU baseline requires these environment variables to equal 1 before "
            "Python starts: " + ", ".join(incorrect)
        )
    affinity = os.sched_getaffinity(0)
    if len(affinity) != 1:
        raise RuntimeError(
            "CPU baseline requires affinity to exactly one physical CPU before import; "
            f"active CPUs are {sorted(affinity)}"
        )
    if cache_root.exists() and any(cache_root.iterdir()):
        raise ValueError(f"task-specific cache must be empty: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)


def _reproducible_benchmark_command(argv: list[str]) -> str:
    affinity = sorted(os.sched_getaffinity(0))
    environment_names = (
        *_THREAD_ENVIRONMENT,
        "OMP_PROC_BIND",
        "SYMMETRIX_JIT_CACHE",
        *_HOST_POLICY_ENVIRONMENT,
    )
    unset_environment = [
        item
        for name in environment_names
        if name not in os.environ
        for item in ("-u", name)
    ]
    set_environment = [
        f"{name}={os.environ[name]}" for name in environment_names if name in os.environ
    ]
    return shlex.join(
        (
            "taskset",
            "-c",
            ",".join(str(cpu) for cpu in affinity),
            "env",
            *unset_environment,
            *set_environment,
            sys.executable,
            str(Path(__file__).resolve()),
            *argv,
        )
    )


def _host_policy_provenance(
    compiler_command: tuple[str, ...],
    compiler_identity: dict[str, Any],
    host_policy: Any,
) -> dict[str, Any]:
    policy = dict(host_policy.identity)
    requested_fp_mode = os.environ.get("SYMMETRIX_JIT_HOST_FP_MODE", "fast")
    effective_fp_mode = (
        None if policy["selected"] == "explicit-flags" else requested_fp_mode.lower()
    )
    return {
        "compiler_command": list(compiler_command),
        "compiler_identity": dict(compiler_identity),
        "effective_target": policy["effective_target"],
        "flags": list(policy["flags"]),
        "floating_point_mode": effective_fp_mode,
        "requested_floating_point_mode": requested_fp_mode,
        "policy": policy,
        "environment": {
            name: os.environ.get(name) for name in _HOST_POLICY_ENVIRONMENT
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_identity(value: Any, name: str) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        value = value[7:]
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} is not a complete SHA-256 digest: {value!r}")
    return "sha256:" + value


def _git_value(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cpu_identity() -> str:
    model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    cpu = next(iter(os.sched_getaffinity(0)))
    return f"{model}; logical_cpu={cpu}; platform={platform.platform()}"


def _native_inputs(evaluator: Any, repeat: int) -> tuple[Any, ...]:
    import numpy as np
    from ase.build import bulk

    try:
        from matscipy.neighbours import neighbour_list
    except ImportError:
        from ase.neighborlist import neighbor_list as neighbour_list

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    i_list, j_list, distances, displacements = neighbour_list(
        "ijdD", atoms, evaluator.r_cut
    )
    node_types = np.asarray(
        [evaluator.atomic_numbers.index(number) for number in atomic_numbers],
        dtype=np.int32,
    )
    num_neigh = np.asarray(np.bincount(j_list, minlength=len(atoms)), dtype=np.int32)
    neigh_types = np.asarray(
        [evaluator.atomic_numbers.index(atomic_numbers[j]) for j in j_list],
        dtype=np.int32,
    )
    return (
        len(atoms),
        node_types,
        num_neigh,
        np.asarray(j_list, dtype=np.int32),
        neigh_types,
        np.ascontiguousarray(displacements, dtype=np.float64),
        np.ascontiguousarray(distances, dtype=np.float64),
        len(i_list),
    )


def _configure_evaluator(evaluator: Any, precision: str) -> None:
    evaluator.set_streamed_edges("direct")
    evaluator._set_factorized_source_strategy("jit_plugin")
    if precision == "float32":
        evaluator._set_edge_geometry_policy("unit-f32-radius-f64-v1")
    evaluator._set_harmonic_storage_policy("y-only-direct-v1")
    evaluator._set_standard_m0_executor("standard")
    evaluator._set_m1_polynomial_policy("recompute")
    evaluator._set_mh0_state_policy("reuse-adjoints-v1")


def _assert_r0_route(
    evaluator: Any,
    *,
    implementation: str,
    expected_artifact_id: str,
) -> dict[str, Any]:
    expected_schedule = "frozen" if implementation == "frozen" else "host-owner"
    expected_profile = (
        "fixed-weight-coordinate" if implementation == "frozen" else "owner-batched"
    )
    observed = str(getattr(evaluator, "r0_implementation", "missing"))
    if observed != "generated":
        raise RuntimeError(
            f"stage=R0 schedule={expected_schedule}: "
            f"expected generated implementation, observed {observed}"
        )
    route = str(getattr(evaluator, "_r0_loader_implementation", "missing"))
    expected_route = "builtin" if implementation == "frozen" else "host_plugin"
    if route != expected_route:
        raise RuntimeError(
            f"stage=R0 schedule={expected_schedule}: "
            f"expected {expected_route} route, observed {route}"
        )
    if str(getattr(evaluator, "r0_backend", "missing")) != "host":
        raise RuntimeError("stage=R0: CPU qualification requires backend=host")
    origin = getattr(evaluator, "r0_artifact_origin", None)
    expected_origin = None if implementation == "frozen" else "runtime"
    if origin != expected_origin:
        raise RuntimeError(
            f"stage=R0: expected artifact origin {expected_origin!r}, observed {origin!r}"
        )
    schedule = str(getattr(evaluator, "r0_schedule", "missing"))
    profile = str(getattr(evaluator, "r0_profile", "missing"))
    if schedule != expected_schedule:
        raise RuntimeError(
            f"stage=R0: expected native schedule {expected_schedule!r}, observed {schedule!r}"
        )
    if profile != expected_profile:
        raise RuntimeError(
            f"stage=R0: expected native profile {expected_profile!r}, observed {profile!r}"
        )
    selected_id = str(getattr(evaluator, "r0_module_id", "missing"))
    if selected_id != expected_artifact_id:
        raise RuntimeError(
            "stage=R0: selected artifact ID mismatch; "
            f"expected {expected_artifact_id}, observed {selected_id}"
        )
    if implementation == "generated":
        if not bool(getattr(evaluator, "r0_host_plugin_ready", False)):
            raise RuntimeError(
                "stage=R0 schedule=host-owner: generated host plugin is not ready"
            )
        observed_id = str(getattr(evaluator, "r0_host_plugin_artifact_id", "missing"))
        if observed_id != expected_artifact_id:
            raise RuntimeError(
                "stage=R0 schedule=host-owner: generated artifact ID mismatch; "
                f"expected {expected_artifact_id}, observed {observed_id}"
            )
    return {
        "implementation": observed,
        "backend": "host",
        "artifact_origin": origin,
        "schedule": schedule,
        "profile": profile,
        "artifact_id": selected_id,
        "qualification_route": route,
    }


def _prepared_call(
    evaluator: Any,
    inputs: tuple[Any, ...],
    *,
    weighted: bool,
    path: str,
) -> Any:
    import numpy as np

    graph = evaluator._prepare_factorized_graph(*inputs[:5])
    if weighted:
        evaluator._set_alchemical_weights(
            graph,
            np.linspace(0.8, 1.2, inputs[0], dtype=np.float64),
            np.linspace(0.7, 1.1, len(inputs[3]), dtype=np.float64),
        )

    def call() -> Any:
        evaluator._compute_prepared_factorized(graph, inputs[5].reshape(-1), inputs[6])
        pair_forces = np.asarray(evaluator.node_forces, dtype=np.float64).reshape(-1, 3)
        active_forces = pair_forces[: inputs[7]]
        if path == "energy":
            return float(np.sum(evaluator.node_energies, dtype=np.float64))
        if path == "forces":
            return np.array(active_forces, copy=True)
        if path == "stress":
            return -(active_forces.T @ inputs[5])
        if path == "density_scaling":
            if not evaluator.standard_r0_density_scale_fused:
                raise RuntimeError(
                    "stage=R0: density-scaling path did not use fused R0 scaling"
                )
            return np.array(evaluator.node_energies, copy=True)
        if path == "direct_harmonic_gradients":
            if not (
                evaluator.harmonic_gradient_bytes == 0
                and evaluator.execution_direct_harmonic_launch_count > 0
            ):
                raise RuntimeError(
                    "stage=R0: direct-harmonic-gradient path was not active"
                )
            return np.array(active_forces, copy=True)
        if path == "prepared_scale_adjoint":
            if evaluator.standard_m0_input_scale_adjoint_launch_count <= 0:
                raise RuntimeError(
                    "stage=R0: prepared-scale-adjoint path was not active"
                )
            return np.array(evaluator.A0_adj, copy=True)
        raise ValueError(f"unsupported R0 baseline path: {path}")

    return call


def _time_call(call: Any) -> tuple[float, Any]:
    started = time.perf_counter()
    result = call()
    return 1000.0 * (time.perf_counter() - started), result


def _measure_case(
    *,
    make_evaluator: Any,
    implementation: str,
    case: BaselineCase,
    artifact_preparation_ms: float,
    cache_root: Path,
    workload: dict[str, Any],
    provenance: dict[str, Any],
    warmups: int,
    samples: int,
) -> tuple[dict[str, Any], Any]:
    first_call_ms: dict[str, float] = {}
    launch_count = 0
    first_output = None
    for property_name in ("energy", "forces", "stress"):
        evaluator, call = make_evaluator(property_name)
        before = evaluator.standard_r0_module_launch_count
        latency, output = _time_call(call)
        launch_count += evaluator.standard_r0_module_launch_count - before
        first_call_ms[property_name] = latency
        if first_output is None:
            first_output = output

    started = time.perf_counter()
    evaluator, steady_call = make_evaluator(case.path)
    evaluator_preparation_ms = 1000.0 * (time.perf_counter() - started)
    launches_before = evaluator.standard_r0_module_launch_count
    for _ in range(warmups):
        steady_call()
    steady_samples_ms = []
    steady_output = None
    for _ in range(samples):
        latency, steady_output = _time_call(steady_call)
        steady_samples_ms.append(latency)
    launch_count += evaluator.standard_r0_module_launch_count - launches_before

    qualifications = {
        "density_scaling": bool(evaluator.standard_r0_density_scale_fused),
        "direct_harmonic_gradients": bool(
            evaluator.harmonic_gradient_bytes == 0
            and evaluator.execution_direct_harmonic_launch_count > 0
        ),
        "prepared_scale_adjoint": bool(
            evaluator.standard_m0_input_scale_adjoint_launch_count > 0
        ),
    }
    record = measurement_record(
        case,
        implementation=implementation,
        r0_report={
            "implementation": evaluator.r0_implementation,
            "backend": evaluator.r0_backend,
            "artifact_origin": evaluator.r0_artifact_origin,
            "schedule": evaluator.r0_schedule,
            "profile": evaluator.r0_profile,
            "artifact_id": evaluator.r0_module_id,
            "qualification_route": evaluator._r0_loader_implementation,
        },
        preparation_ms=artifact_preparation_ms + evaluator_preparation_ms,
        launch_count=launch_count,
        steady_state_samples_ms=steady_samples_ms,
        first_call_ms=first_call_ms,
        cache={"path": str(cache_root), "empty_before": True},
        workload=workload,
        provenance=provenance,
    )
    record["timing_scope"] = f"native-{case.path}-result-path"
    record["path_qualification"] = qualifications
    record["preparation_breakdown_ms"] = {
        "artifact": artifact_preparation_ms,
        "evaluator_and_graph": evaluator_preparation_ms,
    }
    assert first_output is not None and steady_output is not None
    return record, steady_output


def run_cpu_campaign(
    model_path: Path,
    cache_root: Path,
    *,
    warmups: int = 2,
    samples: int = 7,
    small_repeat: int = 1,
    large_repeat: int = 2,
    benchmark_command: str | None = None,
) -> dict[str, Any]:
    """Measure all mandatory CPU cases and emit explicit accelerator skips."""

    model_path = model_path.resolve()
    cache_root = cache_root.resolve()
    if not model_path.is_file():
        raise ValueError(f"compact model does not exist: {model_path}")
    _require_reproducible_cpu_environment(cache_root)
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

    try:
        import numpy as np
        from symmetrix import prepare_jit_host_artifact
        from symmetrix import symmetrix as native
        from symmetrix.jit import (
            artifact_cache_inputs,
            prepare_jit_artifact,
            probe_compiler_identity,
            resolve_host_jit_policy,
        )
        from symmetrix.mace_jit_operator_codegen import (
            OPERATOR_ARTIFACT_ENVELOPE,
            render_jit_r0_host_plugin,
        )
        from tools._symmetrix_build.source_provenance import (
            symmetrix_source_fingerprint,
        )
    finally:
        sys.path.remove(str(repository_root))

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    execution_space = native._kokkos_default_execution_space()
    if execution_space not in {"OpenMP", "Serial"}:
        raise RuntimeError(
            "CPU baseline requires a Kokkos OpenMP or Serial extension; "
            f"active execution space is {execution_space}"
        )

    model_data = json.loads(model_path.read_text())
    if model_data.get("model_type") != "MACE" or model_data.get("has_field_coupling"):
        raise ValueError(
            "CPU baseline model must be compact standard MACE so both unity and "
            "weighted edge paths are valid"
        )
    contracts = model_data.get("execution_contracts", {})
    missing_contracts = {"R0", "R1", "M0"} - set(contracts)
    if missing_contracts:
        raise ValueError(
            "CPU baseline model is missing execution contracts: "
            + ", ".join(sorted(missing_contracts))
        )

    extension_path = Path(native.__file__).resolve()
    build_info = native._backend_build_info()
    source_content_sha256 = symmetrix_source_fingerprint(repository_root)
    host_compiler_command, host_compiler_identity = probe_compiler_identity()
    host_policy = resolve_host_jit_policy(host_compiler_command)
    command = benchmark_command or _reproducible_benchmark_command(sys.argv[1:])
    common_provenance = {
        "source_commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "source_dirty": bool(_git_value(repository_root, "status", "--porcelain")),
        "source_content_sha256": _sha256_identity(
            source_content_sha256, "source content fingerprint"
        ),
        "native_build_source_content_sha256": _sha256_identity(
            build_info.get("source_content_sha256"),
            "native build source content fingerprint",
        ),
        "extension_sha256": _sha256(extension_path),
        "hardware": _cpu_identity(),
        "threads": {"kokkos": 1, "openmp": 1, "blas": 1},
        "benchmark_command": command,
        "host_jit": _host_policy_provenance(
            host_compiler_command, host_compiler_identity, host_policy
        ),
    }
    model_sha256 = _sha256(model_path)
    records: list[dict[str, Any]] = []
    generated_artifacts: list[dict[str, Any]] = []
    repeats = {"small": small_repeat, "large": large_repeat}

    for precision, class_name in (
        ("float32", "MACEKokkosFloat"),
        ("float64", "MACEKokkos"),
    ):
        r1_artifact = prepare_jit_host_artifact(
            model_path,
            precision=precision,
            cache_root=cache_root / "r1",
        )
        for weighted in (False, True):
            started = time.perf_counter()
            source, metadata = render_jit_r0_host_plugin(
                contracts["R0"],
                precision=precision,
                target=host_policy.effective_target,
                apply_edge_weights=weighted,
            )
            r0_artifact = prepare_jit_artifact(
                source,
                source_content_sha256=source_content_sha256,
                abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
                build={"operator": metadata},
                cache_root=cache_root / "r0",
                cxx=host_compiler_command,
                cxx_flags=None,
                host_policy=host_policy,
                artifact_name=(
                    f"r0_host_{precision}_{'weighted' if weighted else 'unity'}"
                ),
            )
            artifact_preparation_ms = 1000.0 * (time.perf_counter() - started)
            if not r0_artifact.available:
                raise RuntimeError(
                    "stage=R0 schedule=host-owner: generated artifact unavailable: "
                    f"{r0_artifact.reason}; {r0_artifact.diagnostics}"
                )
            artifact_provenance = {
                "artifact_id": metadata["artifact_id"],
                "path": str(r0_artifact.artifact_path.resolve()),
                "sha256": _sha256(r0_artifact.artifact_path),
                "precision": precision,
                "edge_weight_mode": "weighted" if weighted else "unity",
                "stage": metadata["stage"],
                "schedule": metadata["schedule"],
            }
            generated_artifacts.append(artifact_provenance)
            provenance = {
                **common_provenance,
                "generated_r0_artifact": {
                    key: artifact_provenance[key]
                    for key in ("artifact_id", "path", "sha256")
                },
            }

            for graph, repeat in repeats.items():
                seed = getattr(native, class_name)(str(model_path))
                inputs = _native_inputs(seed, repeat)
                workload = {
                    "model": model_path.name,
                    "model_sha256": model_sha256,
                    "atoms": inputs[0],
                    "directed_edges": len(inputs[3]),
                    "cutoff_A": float(seed.r_cut),
                    "skin_A": 0.0,
                    "effective_cutoff_A": float(seed.r_cut),
                }
                template = BaselineCase(
                    "cpu",
                    precision,
                    "weighted" if weighted else "unity",
                    graph,
                    "energy",
                )

                def evaluator_factory(implementation: str, path: str) -> Any:
                    evaluator = getattr(native, class_name)(str(model_path))
                    evaluator._load_jit_host_plugin(str(r1_artifact.artifact_path))
                    if implementation == "generated":
                        evaluator._load_r0_host_plugin(
                            str(r0_artifact.artifact_path),
                            weighted,
                            host_policy.effective_target,
                        )
                        evaluator._set_r0_artifact_origin("runtime")
                    _configure_evaluator(evaluator, precision)
                    _assert_r0_route(
                        evaluator,
                        implementation=implementation,
                        expected_artifact_id=(
                            metadata["artifact_id"]
                            if implementation == "generated"
                            else evaluator.standard_r0_module_id
                        ),
                    )
                    call = _prepared_call(
                        evaluator, inputs, weighted=weighted, path=path
                    )
                    _assert_r0_route(
                        evaluator,
                        implementation=implementation,
                        expected_artifact_id=(
                            metadata["artifact_id"]
                            if implementation == "generated"
                            else evaluator.standard_r0_module_id
                        ),
                    )
                    return evaluator, call

                tolerance = 5.0e-4 if precision == "float32" else 3.0e-10
                for path in PATHS:
                    case = BaselineCase(
                        template.backend,
                        template.precision,
                        template.edge_weight_mode,
                        template.graph,
                        path,
                    )
                    frozen_record, frozen_output = _measure_case(
                        make_evaluator=lambda requested_path: evaluator_factory(
                            "frozen", requested_path
                        ),
                        implementation="frozen",
                        case=case,
                        artifact_preparation_ms=0.0,
                        cache_root=cache_root,
                        workload=workload,
                        provenance=provenance,
                        warmups=warmups,
                        samples=samples,
                    )
                    generated_record, generated_output = _measure_case(
                        make_evaluator=lambda requested_path: evaluator_factory(
                            "generated", requested_path
                        ),
                        implementation="generated",
                        case=case,
                        artifact_preparation_ms=artifact_preparation_ms,
                        cache_root=cache_root,
                        workload=workload,
                        provenance=provenance,
                        warmups=warmups,
                        samples=samples,
                    )
                    np.testing.assert_allclose(
                        generated_output,
                        frozen_output,
                        rtol=0.0,
                        atol=tolerance,
                        err_msg=(
                            f"stage=R0 schedule=host-owner precision={precision} "
                            f"edge_weight_mode={template.edge_weight_mode} "
                            f"graph={graph} path={path}"
                        ),
                    )
                    records.extend((frozen_record, generated_record))

    accelerator_reason = (
        f"active extension is Kokkos {execution_space} on CPU; no architecture-qualified "
        "Kokkos accelerator extension is loaded in this process"
    )
    records.extend(
        skipped_case(case, accelerator_reason)
        for case in required_cases(("cuda", "hip"))
    )
    cpu_records = [record for record in records if record["case"]["backend"] == "cpu"]
    return {
        "records": records,
        "campaign": validate_campaign(records),
        "retirement_gate": evaluate_retirement_gate(cpu_records),
        "provenance": {
            **common_provenance,
            "generated_r0_artifacts": generated_artifacts,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--small-repeat", type=int, default=1)
    parser.add_argument("--large-repeat", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_cpu_campaign(
        args.model,
        args.cache_root,
        warmups=args.warmups,
        samples=args.samples,
        small_repeat=args.small_repeat,
        large_repeat=args.large_repeat,
        benchmark_command=_reproducible_benchmark_command(sys.argv[1:]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    json.dump(
        {
            "output": str(args.output.resolve()),
            "campaign": result["campaign"],
            "retirement_gate": result["retirement_gate"]["status"],
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0 if result["retirement_gate"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
