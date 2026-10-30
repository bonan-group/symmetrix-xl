#!/usr/bin/env python3
"""Run the complete frozen-versus-generated R0 CUDA qualification campaign."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from generated_r0_baseline import (
    PATHS,
    BaselineCase,
    evaluate_retirement_gate,
    validate_campaign,
)
from run_generated_r0_cpu_baseline import (
    _configure_evaluator,
    _git_value,
    _measure_case,
    _native_inputs,
    _prepared_call,
    _sha256,
    _sha256_identity,
)


_THREAD_ENVIRONMENT = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def resolve_r0_device_schedule() -> str:
    """Resolve the same R0 schedule used by production artifact planning."""

    from symmetrix.mace_jit_operator_artifact import (
        resolve_r0_device_schedule as resolve_production_r0_device_schedule,
    )

    return resolve_production_r0_device_schedule()


def _r0_nvrtc_options(metadata: dict[str, Any]) -> tuple[str, ...]:
    policy = metadata.get("floating_point_policy")
    if policy == "cuda-fast-math":
        return ("--split-compile=2", "--use_fast_math")
    if policy == "strict":
        return ("--split-compile=2",)
    raise ValueError(f"unsupported R0 floating-point policy: {policy!r}")


def _require_reproducible_cuda_environment(cache_root: Path) -> None:
    incorrect = [name for name in _THREAD_ENVIRONMENT if os.environ.get(name) != "1"]
    if incorrect:
        raise RuntimeError(
            "CUDA baseline requires host thread variables to equal 1 before "
            "Python starts: " + ", ".join(incorrect)
        )
    if cache_root.exists() and any(cache_root.iterdir()):
        raise ValueError(f"task-specific cache must be empty: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)


def _reproducible_benchmark_command(argv: list[str]) -> str:
    environment_names = (
        *_THREAD_ENVIRONMENT,
        "SYMMETRIX_BACKEND",
        "CUDA_VISIBLE_DEVICES",
        "SYMMETRIX_JIT_CACHE",
        "SYMMETRIX_R0_RTC_SCHEDULE",
    )
    unset_environment = [
        item
        for name in environment_names
        if name not in os.environ
        for item in ("-u", name)
    ]
    environment = [
        f"{name}={os.environ[name]}" for name in environment_names if name in os.environ
    ]
    return shlex.join(
        (
            "env",
            *unset_environment,
            *environment,
            sys.executable,
            str(Path(__file__).resolve()),
            *argv,
        )
    )


def _cuda_identity(environment: dict[str, Any]) -> str:
    return (
        f"{environment['device_name']}; ordinal={environment['device_ordinal']}; "
        f"compute_capability={environment['compute_capability_code']}; "
        f"multiprocessors={environment['multiprocessor_count']}; "
        f"driver={environment['driver_version']}; "
        f"runtime={environment['runtime_version']}"
    )


def _assert_r0_route(
    evaluator: Any,
    *,
    implementation: str,
    expected_artifact_id: str,
    schedule: str,
    persistent_blocks: int,
) -> dict[str, Any]:
    route = str(getattr(evaluator, "_r0_loader_implementation", "missing"))
    expected_route = "builtin" if implementation == "frozen" else "device_module"
    if route != expected_route:
        raise RuntimeError(
            f"stage=R0 schedule={schedule}: expected {expected_route} route, "
            f"observed {route}"
        )
    if str(getattr(evaluator, "r0_implementation", "missing")) != "generated":
        raise RuntimeError("stage=R0: CUDA comparison route is not active")
    if str(getattr(evaluator, "r0_backend", "missing")) != "cuda":
        raise RuntimeError("stage=R0: CUDA qualification requires backend=cuda")
    origin = getattr(evaluator, "r0_artifact_origin", None)
    expected_origin = None if implementation == "frozen" else "runtime"
    if origin != expected_origin:
        raise RuntimeError(
            f"stage=R0: expected artifact origin {expected_origin!r}, "
            f"observed {origin!r}"
        )
    expected_schedule = "frozen" if implementation == "frozen" else schedule
    expected_profile = (
        "fixed-weight-coordinate"
        if implementation == "frozen"
        else f"persistent-{persistent_blocks}"
    )
    if evaluator.r0_schedule != expected_schedule:
        raise RuntimeError(
            f"stage=R0: expected schedule {expected_schedule!r}, "
            f"observed {evaluator.r0_schedule!r}"
        )
    if evaluator.r0_profile != expected_profile:
        raise RuntimeError(
            f"stage=R0: expected profile {expected_profile!r}, "
            f"observed {evaluator.r0_profile!r}"
        )
    if evaluator.r0_module_id != expected_artifact_id:
        raise RuntimeError("stage=R0: selected artifact ID mismatch")
    if implementation == "generated":
        if not evaluator.r0_device_module_ready:
            raise RuntimeError("stage=R0: generated CUDA module is not ready")
        if evaluator.r0_device_module_artifact_id != expected_artifact_id:
            raise RuntimeError("stage=R0: generated CUDA artifact ID mismatch")
    return {
        "implementation": evaluator.r0_implementation,
        "backend": "cuda",
        "artifact_origin": origin,
        "schedule": evaluator.r0_schedule,
        "profile": evaluator.r0_profile,
        "artifact_id": evaluator.r0_module_id,
        "qualification_route": route,
    }


def run_cuda_campaign(
    model_path: Path,
    cache_root: Path,
    *,
    warmups: int = 2,
    samples: int = 7,
    small_repeat: int = 1,
    large_repeat: int = 2,
    benchmark_command: str | None = None,
) -> dict[str, Any]:
    """Measure all mandatory CUDA cases with the active matching extension."""

    model_path = model_path.resolve()
    cache_root = cache_root.resolve()
    if not model_path.is_file():
        raise ValueError(f"compact model does not exist: {model_path}")
    _require_reproducible_cuda_environment(cache_root)
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))
    try:
        import numpy as np
        from symmetrix import prepare_jit_device_artifact
        from symmetrix import symmetrix as native
        from symmetrix.jit import (
            artifact_cache_inputs,
            prepare_nvrtc_jit_artifact,
        )
        from symmetrix.mace_jit_operator_codegen import (
            OPERATOR_ARTIFACT_ENVELOPE,
            render_jit_r0_device_module,
        )
        from tools._symmetrix_build.source_provenance import (
            symmetrix_source_fingerprint,
        )
    finally:
        sys.path.remove(str(repository_root))

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    if native._kokkos_default_execution_space() != "Cuda":
        raise RuntimeError("CUDA baseline requires the Kokkos CUDA extension")
    environment = native._execution_device_execution_environment()
    if not environment.get("available") or environment.get("backend") != "cuda":
        raise RuntimeError("CUDA baseline requires an available CUDA device")
    if environment.get("compute_capability_code") != 120:
        raise RuntimeError("CUDA baseline requires the matching sm120 device")

    model_data = json.loads(model_path.read_text())
    if model_data.get("model_type") != "MACE" or model_data.get("has_field_coupling"):
        raise ValueError("CUDA baseline model must be compact standard MACE")
    contracts = model_data.get("execution_contracts", {})
    missing_contracts = {"R0", "R1", "M0"} - set(contracts)
    if missing_contracts:
        raise ValueError(
            "CUDA baseline model is missing execution contracts: "
            + ", ".join(sorted(missing_contracts))
        )

    extension_path = Path(native.__file__).resolve()
    build_info = native._backend_build_info()
    source_content_sha256 = symmetrix_source_fingerprint(repository_root)
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
        "hardware": _cuda_identity(environment),
        "threads": {"kokkos": 1, "openmp": 1, "blas": 1},
        "benchmark_command": command,
    }
    model_sha256 = _sha256(model_path)
    target = f"sm_{environment['compute_capability_code']}"
    records: list[dict[str, Any]] = []
    generated_artifacts: list[dict[str, Any]] = []

    schedule = resolve_r0_device_schedule()
    for precision, class_name in (
        ("float32", "MACEKokkosFloat"),
        ("float64", "MACEKokkos"),
    ):
        r1_artifact = prepare_jit_device_artifact(
            model_path,
            precision=precision,
            backend="cuda",
            include_low_memory_operators=False,
            cache_root=cache_root / "r1",
        )
        if not r1_artifact.edge_policy:
            raise RuntimeError("CUDA R1 artifact did not report a launch profile")
        r1_blocks = int(r1_artifact.edge_policy["persistent_blocks_per_compute_unit"])
        for weighted in (False, True):
            started = time.perf_counter()
            source, metadata = render_jit_r0_device_module(
                contracts["R0"],
                precision=precision,
                target=target,
                schedule=schedule,
                apply_edge_weights=weighted,
            )
            r0_artifact = prepare_nvrtc_jit_artifact(
                source,
                source_content_sha256=source_content_sha256,
                abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
                build={
                    "generator": "benchmark.generated-r0-cuda",
                    "operator": metadata,
                },
                compute_capability=environment,
                nvrtc_options=_r0_nvrtc_options(metadata),
                target=environment,
                cache_root=cache_root / "r0",
                artifact_name=(
                    f"r0_cuda_{precision}_{'weighted' if weighted else 'unity'}"
                ),
            )
            artifact_preparation_ms = 1000.0 * (time.perf_counter() - started)
            if not r0_artifact.available:
                raise RuntimeError(
                    f"stage=R0 schedule={schedule}: generated artifact unavailable: "
                    f"{r0_artifact.reason}; {r0_artifact.diagnostics}"
                )
            persistent_blocks = int(
                metadata["persistent_profile"]["blocks_per_compute_unit"]
            )
            artifact_provenance = {
                "artifact_id": metadata["artifact_id"],
                "path": str(r0_artifact.artifact_path.resolve()),
                "sha256": _sha256(r0_artifact.artifact_path),
                "precision": precision,
                "edge_weight_mode": "weighted" if weighted else "unity",
                "stage": metadata["stage"],
                "schedule": schedule,
            }
            generated_artifacts.append(artifact_provenance)
            provenance = {
                **common_provenance,
                "generated_r0_artifact": {
                    key: artifact_provenance[key]
                    for key in ("artifact_id", "path", "sha256")
                },
            }

            for graph, repeat in (("small", small_repeat), ("large", large_repeat)):
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

                def evaluator_factory(implementation: str, path: str) -> Any:
                    evaluator = getattr(native, class_name)(str(model_path))
                    evaluator._load_jit_device_plugin(
                        str(r1_artifact.artifact_path), r1_blocks
                    )
                    if implementation == "generated":
                        evaluator._load_r0_device_module(
                            str(r0_artifact.artifact_path),
                            persistent_blocks,
                            weighted,
                            schedule,
                        )
                        evaluator._set_r0_artifact_origin("runtime")
                    _configure_evaluator(evaluator, precision)
                    expected_id = (
                        metadata["artifact_id"]
                        if implementation == "generated"
                        else evaluator.standard_r0_module_id
                    )
                    _assert_r0_route(
                        evaluator,
                        implementation=implementation,
                        expected_artifact_id=expected_id,
                        schedule=schedule,
                        persistent_blocks=persistent_blocks,
                    )
                    return evaluator, _prepared_call(
                        evaluator, inputs, weighted=weighted, path=path
                    )

                tolerance = 5.0e-4 if precision == "float32" else 3.0e-10
                for path in PATHS:
                    case = BaselineCase(
                        "cuda",
                        precision,
                        "weighted" if weighted else "unity",
                        graph,
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
                            f"stage=R0 schedule={schedule} precision={precision} "
                            f"edge_weight_mode={'weighted' if weighted else 'unity'} "
                            f"graph={graph} path={path}"
                        ),
                    )
                    records.extend((frozen_record, generated_record))

    return {
        "records": records,
        "campaign": validate_campaign(records, backends=("cuda",)),
        "retirement_gate": evaluate_retirement_gate(records, backends=("cuda",)),
        "provenance": {
            **common_provenance,
            "generated_r0_artifacts": generated_artifacts,
            "device_environment": environment,
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
    result = run_cuda_campaign(
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
