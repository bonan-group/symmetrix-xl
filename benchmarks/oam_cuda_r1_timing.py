"""Measure OAM generated-R1 energy/forces on a fixed periodic SrTiO3 graph.

Run CPU and CUDA in separate processes/environments. Compilation, graph setup,
module loading and warmup are excluded; force reduction and transfers are timed.
CPU execution pins one logical CPU per distinct physical core before importing
NumPy, BLAS or Kokkos.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

try:
    from benchmarks.structure_workloads import repeat_factors
except ModuleNotFoundError as error:
    if error.name != "benchmarks":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.structure_workloads import repeat_factors

from benchmarks.oam_release_qualification import (
    validate_driver_qualification_preflight,
)


def _parse_repeat(value):
    parts = value.split(",")
    if len(parts) == 1:
        return int(value)
    return repeat_factors(tuple(int(part) for part in parts))


def _select_physical_cpus(available, threads, *, start_cpu=None, topology=None):
    """Select one allowed logical CPU from each distinct physical core."""

    available = list(sorted(int(cpu) for cpu in available))
    if start_cpu is not None:
        if start_cpu not in available:
            raise ValueError(f"CPU {start_cpu} is outside process affinity {available}")
        offset = available.index(start_cpu)
        available = available[offset:] + available[:offset]
    if topology is None:
        topology = {}
        for cpu in available:
            root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            try:
                package = int((root / "physical_package_id").read_text().strip())
                core = int((root / "core_id").read_text().strip())
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"physical topology is unavailable for CPU {cpu}"
                ) from error
            topology[cpu] = (package, core)
    selected = []
    used_cores = set()
    for cpu in available:
        key = topology.get(cpu)
        if key is None:
            raise ValueError(f"physical topology is unavailable for CPU {cpu}")
        if key in used_cores:
            continue
        used_cores.add(key)
        selected.append(cpu)
        if len(selected) == threads:
            return selected
    raise ValueError(
        f"--threads {threads} needs {threads} allowed physical cores; found {len(selected)}"
    )


def _cpu_topology(cpu_ids):
    rows = []
    for cpu in cpu_ids:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((root / "physical_package_id").read_text().strip())
            core = int((root / "core_id").read_text().strip())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"physical topology is unavailable for CPU {cpu}"
            ) from error
        rows.append({"logical_cpu": cpu, "package": package, "core": core})
    return rows


def _observed_cpu_runtime(native):
    query = getattr(native, "_openmp_runtime_info", None)
    if not callable(query):
        return {}
    try:
        info = query()
    except (OSError, RuntimeError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def _load_cuda_profiler():
    candidates = (
        os.environ.get("SYMMETRIX_CUDART_LIBRARY"),
        ctypes.util.find_library("cudart"),
        "libcudart.so",
    )
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        library.cudaProfilerStart.argtypes = ()
        library.cudaProfilerStart.restype = ctypes.c_int
        library.cudaProfilerStop.argtypes = ()
        library.cudaProfilerStop.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("could not load CUDA profiler API: " + "; ".join(errors))


def admit_direct_candidate(evaluator, profile, backend, precision):
    """Pin the declared candidate through the native qualification-control seam."""

    plan_id = "oam-direct-capacity" if profile == "capacity" else "oam-direct-speed"
    candidates = {
        candidate["id"]: candidate
        for candidate in evaluator.execution_plan_report["candidates"]
    }
    candidate = candidates.get(plan_id)
    if candidate is None:
        raise RuntimeError(f"native evaluator does not declare {plan_id}")
    estimate = candidate.get("estimated_bytes")
    if estimate is None:
        raise RuntimeError(f"{plan_id} has no exact planner estimate")
    artifact_identity = evaluator.family_artifact_readiness.get("artifact_id")
    if not artifact_identity:
        raise RuntimeError(f"{plan_id} has no loaded artifact identity")
    evaluator._set_execution_plan_qualification(
        plan_id,
        artifact_identity,
        backend,
        precision,
        "capacity" if profile == "capacity" else "throughput",
        candidate["required_capability"],
        "qualified",
        estimate,
        estimate,
        0,
    )
    selected_id = evaluator.execution_plan_report.get("selected_id")
    if selected_id != plan_id:
        raise RuntimeError(
            f"native evaluator selected {selected_id!r}, expected {plan_id!r}"
        )


def prepare_direct_evaluator(factory, *, profile, variant, precision):
    """Construct and configure a direct evaluator with contextual rejection."""

    if profile not in {"speed", "capacity"}:
        raise ValueError(f"unsupported direct profile {profile!r}")
    if variant != "full_message":
        raise ValueError(f"unsupported direct edge variant {variant!r}")
    if precision not in {"float32", "float64"}:
        raise ValueError(f"unsupported direct precision {precision!r}")
    plan_id = "oam-direct-capacity" if profile == "capacity" else "oam-direct-speed"
    context = {
        "plan_id": plan_id,
        "profile": profile,
        "variant": variant,
        "precision": precision,
    }
    try:
        evaluator = factory(context)
    except ValueError as error:
        raise ValueError(
            f"direct preparation candidate={plan_id} profile={profile} "
            f"variant={variant} precision={precision} rejected: {error}"
        ) from error
    evaluator.set_streamed_edges("direct")
    evaluator._set_execution_plan_request("direct", profile)
    evaluator._set_execution_edge_variant(variant)
    return evaluator


def prepare_diagnostic_host_direct_artifact(preparer, contract, precision):
    """Prepare an ordinary host artifact using the configured JIT/cache policy."""

    artifact = preparer(contract, precision)
    generation_policy = artifact.metadata.get("generation_policy")
    if not isinstance(generation_policy, dict):
        raise RuntimeError("OAM host artifact has no canonical generation policy")
    return artifact, dict(generation_policy)


def prepare_cuda_direct_artifact(
    preparer,
    *,
    contract,
    precision,
    target,
    native,
    compiler,
    threads_per_block,
    generation_policy,
):
    """Prepare ordinary diagnostic CUDA execution with normal JIT precedence."""

    arguments = {
        "compiler": compiler,
        "threads_per_block": threads_per_block,
        "generation_policy": generation_policy,
    }
    return preparer(contract, precision, target, **arguments)


def serialize_diagnostic_artifact_evidence(
    *, backend, route, artifact, artifact_sha256, generation_policy
):
    """Serialize ordinary diagnostic evidence without qualification semantics."""

    if route == "reference":
        if artifact is not None or artifact_sha256 is not None:
            raise RuntimeError("reference execution must be artifact-free")
        return {"kind": "reference", "origin": "none"}
    if route != "direct" or artifact is None or artifact_sha256 is None:
        raise RuntimeError("direct execution requires an exact artifact")
    evidence = {
        "kind": f"{backend}-direct-diagnostic",
        "status": artifact.status,
        "identity": artifact.artifact_id,
        "origin": artifact.artifact_origin,
        "sha256": artifact_sha256,
        "schedule": dict(generation_policy),
    }
    if backend == "cuda":
        evidence.update(
            target=artifact.target,
            compiler=artifact.compiler,
            diagnostics=list(artifact.diagnostics),
            cache_key=artifact.cache_key,
        )
    return evidence


def _qualification_execution_entry(
    request,
    *,
    artifact_preparer,
    contract,
    native,
    target,
    compiler,
    threads_per_block,
    generation_policy,
    **live_validation,
):
    """Preflight, resolve, and serialize installed-only qualification state."""

    validated, build_info = validate_driver_qualification_preflight(
        request, **live_validation
    )
    route = live_validation["route"]
    backend = validated["backend"]
    precision = validated["precision"]
    if route == "reference":
        return {
            "qualification_cell": validated,
            "artifact": None,
            "generation_policy": None,
            "artifact_evidence": {"kind": "reference", "origin": "none"},
        }
    if route != "direct":
        raise RuntimeError(f"unsupported qualification route {route!r}")

    if backend == "host":
        fast_math = build_info.get("fast_math")
        if not isinstance(fast_math, bool):
            raise RuntimeError("native build info has no boolean fast_math policy")
        floating_point_policy = "fast" if fast_math else "strict"
        artifact = artifact_preparer(
            contract,
            precision,
            native_module=native,
            installed_prebuilt_required=True,
            expected_host_target=validated["host_target"],
            expected_floating_point_policy=floating_point_policy,
        )
        policy = artifact.metadata.get("generation_policy")
        if not isinstance(policy, dict):
            raise RuntimeError("OAM host artifact has no canonical generation policy")
        native_target = build_info["prebuilt_operator_artifacts"]["target"]
        active_target = f"host-isa-{validated['host_target']}"
        if (
            artifact.status != "installed"
            or artifact.artifact_origin != "prebuilt"
            or artifact.expected_artifact_origin != "prebuilt"
            or artifact.expected_artifact_id != artifact.artifact_id
            or artifact.expected_host_target != active_target
            or artifact.active_host_target != active_target
            or native_target != active_target
            or artifact.floating_point_policy != floating_point_policy
        ):
            raise RuntimeError("host artifact evidence contradicts the native build")
        evidence = {
            "kind": "host-direct",
            "expected_status": "installed",
            "actual_status": artifact.status,
            "expected_identity": artifact.expected_artifact_id,
            "actual_identity": artifact.artifact_id,
            "expected_origin": artifact.expected_artifact_origin,
            "actual_origin": artifact.artifact_origin,
            "sha256": hashlib.sha256(
                Path(artifact.artifact_path).read_bytes()
            ).hexdigest(),
            "expected_target": validated["host_target"],
            "native_build_target": native_target,
            "active_host_target": artifact.active_host_target,
            "expected_floating_point_policy": floating_point_policy,
            "actual_floating_point_policy": artifact.floating_point_policy,
        }
    elif backend == "cuda":
        from symmetrix.cuda_codegen_common import canonical_cuda_target_identity

        artifact = artifact_preparer(
            contract,
            precision,
            target,
            compiler=compiler,
            threads_per_block=threads_per_block,
            generation_policy=generation_policy,
            native_module=native,
            installed_prebuilt_required=True,
            expected_target=validated["cuda_target"],
        )
        policy = generation_policy
        if policy is None:
            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
            policy = manifest["key_inputs"]["build"]["metadata"]["generation_policy"]
        elif hasattr(policy, "__dataclass_fields__"):
            policy = asdict(policy)
        if (
            artifact.status != "installed"
            or artifact.artifact_origin != "prebuilt"
            or artifact.compiler != "prebuilt"
        ):
            raise RuntimeError(
                "CUDA qualification requires an installed prebuilt artifact"
            )
        if canonical_cuda_target_identity(
            artifact.target
        ) != canonical_cuda_target_identity(validated["cuda_target"]):
            raise RuntimeError("CUDA artifact target contradicts the request")
        evidence = {
            "kind": "cuda-direct",
            "expected_status": "installed",
            "actual_status": artifact.status,
            "expected_identity": artifact.artifact_id,
            "actual_identity": artifact.artifact_id,
            "expected_origin": "prebuilt",
            "actual_origin": artifact.artifact_origin,
            "sha256": hashlib.sha256(
                Path(artifact.artifact_path).read_bytes()
            ).hexdigest(),
            "target": artifact.target,
            "expected_compiler": "prebuilt",
            "actual_compiler": artifact.compiler,
            "schedule": dict(policy),
        }
    else:
        raise RuntimeError(f"unsupported artifact evidence backend {backend!r}")
    return {
        "qualification_cell": validated,
        "artifact": artifact,
        "generation_policy": dict(policy),
        "artifact_evidence": evidence,
    }


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], allow_abbrev=False
    )
    parser.add_argument("backend", choices=("cpu", "cuda"))
    parser.add_argument("precision", choices=("float32", "float64"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--structure", choices=("srtio3", "chain"), default="srtio3")
    parser.add_argument(
        "--atoms", type=int, default=10000, help="Periodic chain atom count"
    )
    parser.add_argument("--repeat", type=_parse_repeat, default=10)
    parser.add_argument("--rattle", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--cpu", type=int)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        choices=(1, 2, 4, 8),
        help="Kokkos/OpenMP use N; BLAS/MKL/BLIS/NumExpr remain at one",
    )
    parser.add_argument("--compiler", choices=("nvrtc", "nvcc"))
    parser.add_argument("--host-target", choices=("x86-64-v3",))
    parser.add_argument(
        "--threads-per-block",
        type=int,
        choices=(32, 64, 128),
        default=128,
        help="Cooperative reverse threads; forward stays at 128",
    )
    parser.add_argument(
        "--forward-live-output-budget",
        type=int,
        choices=(8, 16, 24, 32),
        help="Group generated forward output blocks by this live-component budget",
    )
    parser.add_argument(
        "--forward-input-mode",
        choices=("inline", "staged"),
        default="inline",
        help="Compute harmonics inline or retain one reusable row per edge",
    )
    parser.add_argument(
        "--forward-grouping-mode",
        choices=("output_order", "source_first_fit"),
        default="output_order",
        help="Generated forward path grouping policy",
    )
    parser.add_argument(
        "--forward-group-execution",
        choices=("serial", "warp_tasks"),
        default="serial",
        help="Execute groups serially per receiver or as independent warp tasks",
    )
    parser.add_argument(
        "--radial-adjoint-mode",
        choices=("auto", "fp64", "fp32", "compensated_fp32", "hierarchical_fp32"),
        default="auto",
        help="Generated radial/spline policy; auto uses the qualified target default",
    )
    parser.add_argument(
        "--reverse-group-execution",
        choices=("serial", "source_groups"),
        default="serial",
        help="Use the source-owner baseline or segmented source-irrep partials",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-one",
        action="store_true",
        help="bracket the single measured evaluation with the CUDA profiler API",
    )
    parser.add_argument(
        "--arrays", type=Path, help="Save energies, forces and structure as NPZ"
    )
    parser.add_argument(
        "--qualification-arrays",
        type=Path,
        help="Save compact hash-bound energy/forces/stress qualification evidence",
    )
    parser.add_argument("--routing", choices=("direct", "reference"), default="direct")
    parser.add_argument(
        "--execution-profile", choices=("speed", "capacity"), default="speed"
    )
    parser.add_argument(
        "--edge-variant", choices=("full_message",), default="full_message"
    )
    parser.add_argument(
        "--qualification-cell",
        help="JSON request copied into a qualification record by the orchestrator",
    )
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    qualification_payload = (
        json.loads(args.qualification_cell) if args.qualification_cell else None
    )
    if (
        any(factor < 1 for factor in repeat_factors(args.repeat))
        or args.samples < 1
        or args.warmup < 1
        or args.skin < 0
    ):
        parser.error(
            "repeat, samples, warmup must be positive; skin must be nonnegative"
        )
    if args.atoms < 2:
        parser.error("chain atom count must be at least two")
    if args.backend == "cuda" and args.threads != 1:
        parser.error("--threads applies only to the CPU backend")
    if args.backend == "cpu" and args.host_target != "x86-64-v3":
        parser.error("CPU qualification requires --host-target x86-64-v3")
    if args.profile_one and (args.backend != "cuda" or args.samples != 1):
        parser.error("--profile-one requires the CUDA backend and --samples 1")
    if (
        args.forward_live_output_budget is not None
        or args.forward_grouping_mode != "output_order"
        or args.forward_group_execution != "serial"
        or args.forward_input_mode != "inline"
        or args.radial_adjoint_mode != "auto"
        or args.reverse_group_execution != "serial"
    ) and args.backend != "cuda":
        parser.error("forward generation policy requires the CUDA backend")
    runtime_thread_variables = ("KOKKOS_NUM_THREADS", "OMP_NUM_THREADS")
    blas_thread_variables = (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    thread_variables = runtime_thread_variables + blas_thread_variables
    selected = []
    selected_topology = []
    if args.backend == "cpu":
        available = sorted(os.sched_getaffinity(0))
        try:
            selected = _select_physical_cpus(
                available, args.threads, start_cpu=args.cpu
            )
        except ValueError as error:
            parser.error(str(error))
        os.sched_setaffinity(0, set(selected))
        selected_topology = _cpu_topology(selected)
        for variable in runtime_thread_variables:
            os.environ[variable] = str(args.threads)
        for variable in blas_thread_variables:
            os.environ[variable] = "1"
    os.environ["SYMMETRIX_JIT_POLICY"] = "required"

    import numpy as np
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    from symmetrix.oam_cuda_artifact import prepare_nequip_oam_cuda_artifact
    from symmetrix.oam_cuda_codegen import (
        OAMCudaGenerationPolicy,
    )
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact

    import symmetrix
    from symmetrix import load_backend

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nequip_oam_kokkos_profile import build_srtio3

    native = load_backend("cpu" if args.backend == "cpu" else None)
    native._init_kokkos()
    observed_runtime = _observed_cpu_runtime(native) if args.backend == "cpu" else {}
    worker_sentinel = observed_runtime.get("worker_sentinel", {})
    observed_runtime_evidence = (
        {
            "kokkos_execution_space": observed_runtime.get("kokkos_execution_space"),
            "kokkos_concurrency": observed_runtime.get("kokkos_concurrency"),
            "omp_max_threads": observed_runtime.get("omp_max_threads"),
            "worker_actual_team_size": worker_sentinel.get("actual_team_size"),
            "worker_cpu_observation_supported": worker_sentinel.get(
                "cpu_observation_supported"
            ),
        }
        if args.backend == "cpu"
        else None
    )
    target = native._execution_device_execution_environment()
    if args.backend == "cuda" and target.get("backend") != "cuda":
        raise RuntimeError("CUDA benchmark requires a CUDA native extension")
    native_path = Path(native.__file__).resolve()
    qualification_cell = None
    artifact_evidence = None
    live_validation = None
    if qualification_payload is not None:
        command_arguments = list(sys.argv[1:] if argv is None else argv)
        live_validation = {
            "command": [
                sys.executable,
                str(Path(__file__).resolve()),
                *command_arguments,
            ],
            "backend": args.backend,
            "precision": args.precision,
            "contract_path": args.contract,
            "route": args.routing,
            "profile": args.execution_profile,
            "edge_variant": args.edge_variant,
            "actual_target": target,
            "native_path": native_path,
            "package_path": str(Path(symmetrix.__file__).resolve()),
            "python_executable": sys.executable,
            "native_module": native,
            "active_jit_cache": os.environ.get("SYMMETRIX_JIT_CACHE"),
        }
    class_name = (
        "NequIPOAMKokkosFloat" if args.precision == "float32" else "NequIPOAMKokkos"
    )
    if args.backend == "cuda" and args.routing == "direct":
        if (
            args.forward_live_output_budget is None
            and args.forward_grouping_mode == "output_order"
            and args.forward_group_execution == "serial"
            and args.forward_input_mode == "inline"
            and args.radial_adjoint_mode == "auto"
            and args.reverse_group_execution == "serial"
        ):
            requested_generation_policy = None
            generation_policy = None
        else:
            requested_generation_policy = OAMCudaGenerationPolicy(
                forward_live_output_budget=args.forward_live_output_budget,
                forward_grouping_mode=args.forward_grouping_mode,
                forward_group_execution=args.forward_group_execution,
                forward_input_mode=args.forward_input_mode,
                radial_adjoint_mode=(
                    "fp64"
                    if args.radial_adjoint_mode == "auto"
                    else args.radial_adjoint_mode
                ),
                reverse_group_execution=args.reverse_group_execution,
            )
            generation_policy = requested_generation_policy
        if live_validation is not None:
            qualification = _qualification_execution_entry(
                qualification_payload,
                artifact_preparer=prepare_nequip_oam_cuda_artifact,
                contract=args.contract,
                native=native,
                target=qualification_payload["cuda_target"],
                compiler=args.compiler,
                threads_per_block=args.threads_per_block,
                generation_policy=requested_generation_policy,
                **live_validation,
            )
            qualification_cell = qualification["qualification_cell"]
            artifact_evidence = qualification["artifact_evidence"]
            artifact = qualification["artifact"]
            generation_policy = qualification["generation_policy"]
        else:
            artifact = prepare_cuda_direct_artifact(
                prepare_nequip_oam_cuda_artifact,
                contract=args.contract,
                precision=args.precision,
                target=target,
                native=native,
                compiler=args.compiler,
                threads_per_block=args.threads_per_block,
                generation_policy=requested_generation_policy,
            )
            if generation_policy is None:
                manifest = json.loads(
                    artifact.manifest_path.read_text(encoding="utf-8")
                )
                policy_data = manifest["key_inputs"]["build"]["request"]["oam"][
                    "generation_policy"
                ]
                generation_policy = OAMCudaGenerationPolicy(**policy_data)
        model_start = time.perf_counter()
        evaluator = prepare_direct_evaluator(
            lambda context: getattr(native, class_name)(
                str(args.contract),
                "",
                cuda_plugin_path=str(artifact.artifact_path),
                cuda_artifact_identity={"manifest_path": str(artifact.manifest_path)},
                requested_profile=context["profile"],
                requested_edge_variant=context["variant"],
                requested_precision=context["precision"],
            ),
            profile=args.execution_profile,
            variant=args.edge_variant,
            precision=args.precision,
        )
    elif args.routing == "direct":
        requested_generation_policy = None
        generation_policy = None
        if live_validation is not None:
            qualification = _qualification_execution_entry(
                qualification_payload,
                artifact_preparer=prepare_nequip_oam_receiver_host_artifact,
                contract=args.contract,
                native=native,
                target=args.host_target,
                compiler=None,
                threads_per_block=None,
                generation_policy=None,
                **live_validation,
            )
            qualification_cell = qualification["qualification_cell"]
            artifact_evidence = qualification["artifact_evidence"]
            artifact = qualification["artifact"]
            generation_policy = qualification["generation_policy"]
        else:
            artifact, generation_policy = prepare_diagnostic_host_direct_artifact(
                prepare_nequip_oam_receiver_host_artifact,
                args.contract,
                args.precision,
            )
        model_start = time.perf_counter()
        evaluator = prepare_direct_evaluator(
            lambda context: getattr(native, class_name)(
                str(args.contract),
                str(artifact.artifact_path),
                host_active_target=artifact.active_host_target,
                host_artifact_origin=artifact.artifact_origin,
                requested_profile=context["profile"],
                requested_edge_variant=context["variant"],
                requested_precision=context["precision"],
            ),
            profile=args.execution_profile,
            variant=args.edge_variant,
            precision=args.precision,
        )
    else:
        requested_generation_policy = None
        generation_policy = None
        artifact = None
        if live_validation is not None:
            qualification = _qualification_execution_entry(
                qualification_payload,
                artifact_preparer=None,
                contract=args.contract,
                native=native,
                target=target,
                compiler=None,
                threads_per_block=None,
                generation_policy=None,
                **live_validation,
            )
            qualification_cell = qualification["qualification_cell"]
            artifact_evidence = qualification["artifact_evidence"]
        model_start = time.perf_counter()
        evaluator = getattr(native, class_name)(str(args.contract))
    model_preparation_seconds = time.perf_counter() - model_start
    if args.routing != "direct":
        evaluator.set_streamed_edges(args.routing)
        evaluator._set_execution_plan_request(args.routing, args.execution_profile)
    if args.structure == "srtio3":
        atoms = build_srtio3(repeat=args.repeat, rattle=args.rattle, seed=args.seed)
    else:
        positions = np.zeros((args.atoms, 3))
        positions[:, 0] = 1.3 * np.arange(args.atoms)
        atoms = Atoms(
            numbers=np.full(args.atoms, evaluator.atomic_numbers[0]),
            positions=positions,
            cell=(1.3 * args.atoms, 12.0, 12.0),
            pbc=(True, False, False),
        )
    cutoff = evaluator.r_cut
    receivers, sources, shifts = neighbor_list("ijS", atoms, cutoff + args.skin)
    order = np.argsort(receivers, kind="stable")
    receivers = np.asarray(receivers[order], dtype=np.int32)
    sources = np.asarray(sources[order], dtype=np.int32)
    shifts = np.asarray(shifts[order], dtype=np.int32)
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    xyz = np.ascontiguousarray(
        positions[sources] + shifts @ cell - positions[receivers]
    )
    distances = np.linalg.norm(xyz, axis=1)
    mapping = {int(z): index for index, z in enumerate(evaluator.atomic_numbers)}
    node_types = np.asarray([mapping[int(z)] for z in atoms.numbers], dtype=np.int32)
    generation = evaluator.prepare_graph(
        len(atoms),
        node_types,
        np.bincount(receivers, minlength=len(atoms)).astype(np.int32),
        sources,
        node_types[sources],
    )
    if args.routing == "direct":
        admit_direct_candidate(
            evaluator,
            args.execution_profile,
            "host" if args.backend == "cpu" else "cuda",
            args.precision,
        )
    evaluator.prepare_geometry(
        generation,
        positions.ravel(),
        xyz.ravel(),
        cell.ravel(),
        np.linalg.inv(cell).ravel(),
        np.asarray(atoms.pbc, dtype=np.int32),
    )
    volume = float(abs(np.linalg.det(cell)))

    def evaluate(instance):
        instance.compute_prepared(generation, xyz.ravel(), distances)
        instance.reduce_forces(generation)
        instance.reduce_stress(volume, generation)

    for _ in range(args.warmup):
        evaluate(evaluator)
    replacements_before_samples = int(
        evaluator.execution_memory_report.get("replacements", 0)
    )
    profiler = None
    profiler_library = None
    if args.profile_one:
        profiler, profiler_library = _load_cuda_profiler()
    samples = []
    for _ in range(args.samples):
        if profiler is not None and profiler.cudaProfilerStart() != 0:
            raise RuntimeError("cudaProfilerStart failed")
        start = time.perf_counter()
        try:
            evaluate(evaluator)
        finally:
            samples.append(time.perf_counter() - start)
            if profiler is not None and profiler.cudaProfilerStop() != 0:
                raise RuntimeError("cudaProfilerStop failed")
    energies = np.asarray(evaluator.node_energies, dtype=float)
    forces = np.asarray(evaluator.atom_forces, dtype=float).reshape(-1, 3)
    stress = np.asarray(evaluator.stress_tensor, dtype=float).reshape(3, 3)
    if (
        not np.isfinite(energies).all()
        or not np.isfinite(forces).all()
        or not np.isfinite(stress).all()
    ):
        raise RuntimeError(
            "OAM evaluation produced nonfinite energy, forces, or stress"
        )
    if args.backend == "cuda" and args.routing == "direct":
        assert evaluator.cuda_r1_ready and evaluator.cuda_r1_fallback_count == 0
        assert evaluator.cuda_r1_launch_count == 2 * evaluator.num_layers * (
            args.warmup + args.samples
        )
    graph_hash = hashlib.sha256()
    for array in (positions, cell, node_types, receivers, sources, shifts):
        graph_hash.update(array.tobytes())
    plan = evaluator.execution_plan_report
    memory = evaluator.execution_memory_report
    replacement_delta = int(memory.get("replacements", 0)) - replacements_before_samples
    build_info = getattr(native, "_backend_build_info", lambda: {})()
    source_fingerprint = (
        build_info.get("source_content_sha256")
        or build_info.get("source_fingerprint")
        or "unknown"
    )
    artifact_id = artifact.artifact_id if artifact is not None else "reference"
    artifact_sha256 = (
        hashlib.sha256(Path(artifact.artifact_path).read_bytes()).hexdigest()
        if artifact is not None
        else None
    )
    generation_policy_record = (
        asdict(generation_policy)
        if hasattr(generation_policy, "__dataclass_fields__")
        else generation_policy
    )
    evidence_arguments = {
        "backend": "host" if args.backend == "cpu" else "cuda",
        "route": args.routing,
        "artifact": artifact,
        "artifact_sha256": artifact_sha256,
        "generation_policy": generation_policy_record,
    }
    if artifact_evidence is None:
        artifact_evidence = serialize_diagnostic_artifact_evidence(**evidence_arguments)
    target_name = str(
        target.get("target")
        or target.get("architecture")
        or target.get("backend")
        or ("host" if args.backend == "cpu" else "cuda")
    )
    jit_cache_path = Path(os.environ.get("SYMMETRIX_JIT_CACHE", "")).resolve()
    jit_cache_stat = jit_cache_path.stat()
    if args.backend == "cpu":
        execution_environment = {
            "available": target.get("available"),
            "backend": "host",
            "execution_space": target.get("execution_space"),
            "memory_space": target.get("memory_space"),
            "target": target_name,
        }
    else:
        execution_environment = {
            "available": target.get("available"),
            "backend": "cuda",
            "execution_space": target.get("execution_space"),
            "memory_space": target.get("memory_space"),
            "architecture": target_name,
            "device_ordinal": target.get("device_ordinal"),
            "compute_capability_code": target.get("compute_capability_code"),
        }
    samples_us_per_atom_step = [value * 1e6 / len(atoms) for value in samples]
    canonical_median = statistics.median(samples_us_per_atom_step)
    numerical_evidence = None
    if args.qualification_arrays is not None:
        numerical_path = args.qualification_arrays.resolve()
        numerical_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            numerical_path,
            energy=np.asarray(float(energies.sum()), dtype=np.float64),
            forces=forces,
            stress=stress,
        )
        numerical_evidence = {
            "format": "npz-v1",
            "path": str(numerical_path),
            "sha256": hashlib.sha256(numerical_path.read_bytes()).hexdigest(),
        }
    record = {
        "schema": "symmetrix.oam-qualified-driver-record",
        "version": 3,
        "report_version": 3,
        "route": args.routing,
        "execution_profile": args.execution_profile,
        "qualification_cell": qualification_cell,
        "backend": args.backend,
        "precision": args.precision,
        "structure": args.structure,
        "implementation": (
            "receiver-owned-generated"
            if args.routing == "direct"
            else "portable-oam-kokkos-reference"
        ),
        "model": str(args.contract.resolve()),
        "model_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "provenance": {
            "source_fingerprint": source_fingerprint,
            "source_commit": build_info.get("source_commit", "unknown"),
            "source_dirty": build_info.get("source_dirty"),
            "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            "native_fast_math": (
                build_info.get("fast_math") if args.backend == "cpu" else None
            ),
            "jit_cache": str(jit_cache_path),
            "jit_cache_owner_uid": jit_cache_stat.st_uid,
            "jit_cache_mode": jit_cache_stat.st_mode & 0o777,
        },
        "command": [sys.executable, *sys.argv],
        "python": sys.executable,
        "package": str(Path(symmetrix.__file__).resolve()),
        "native_extension": str(native_path),
        "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "execution_environment": execution_environment,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in thread_variables},
        "observed_runtime": observed_runtime_evidence,
        "atoms": len(atoms),
        "directed_edges": len(sources),
        "model_cutoff": cutoff,
        "skin": args.skin,
        "effective_cutoff": cutoff + args.skin,
        "graph_sha256": graph_hash.hexdigest(),
        "graph": {
            "atom_count": len(atoms),
            "cutoff": cutoff,
            "skin": args.skin,
            "effective_cutoff": cutoff + args.skin,
            "directed_edges": len(sources),
        },
        "repeat": list(args.repeat)
        if not isinstance(args.repeat, int)
        else args.repeat,
        "rattle": args.rattle,
        "seed": args.seed,
        "warmup": args.warmup,
        "sample_count": args.samples,
        "profile_capture": (
            {
                "capture_api": "cudaProfilerApi",
                "library": profiler_library,
                "iterations": 1,
                "warmup_iterations": args.warmup,
            }
            if profiler is not None
            else None
        ),
        "generation_policy": generation_policy_record,
        "generation_policy_selection": (
            "automatic" if requested_generation_policy is None else "explicit"
        ),
        "median_us_per_atom_step": canonical_median,
        "median_us_per_atom": canonical_median,
        "min_us_per_atom_step": min(samples_us_per_atom_step),
        "samples_us_per_atom_step": samples_us_per_atom_step,
        "energy": float(energies.sum()),
        "max_force": float(np.max(np.abs(forces))),
        "numerical_evidence": numerical_evidence,
        "artifact_id": artifact_id,
        "artifact_path": str(artifact.artifact_path) if artifact is not None else None,
        "artifact_cache_status": artifact.status if artifact is not None else None,
        "artifact": artifact_evidence,
        "execution": {
            "plan_id": plan.get("selected_id"),
            "edge_variant": plan.get("edge_variant"),
            "bindings": plan.get("bindings"),
            "schedule": "capacity"
            if args.execution_profile == "capacity"
            else "throughput",
            "backend": "host" if args.backend == "cpu" else "cuda",
            "target": target_name,
            "fallback_count": plan.get("fallback_count", 0),
            "binding_report_version": plan.get("binding_report_version"),
        },
        "memory": {
            "active_bytes": memory.get("active_bytes"),
            "retained_bytes": memory.get("retained_capacity_bytes"),
            "peak_transient_bytes": memory.get("peak_transient_bytes"),
            "replacements": memory.get("replacements"),
        },
        "reuse": {
            "replacement_delta": replacement_delta,
            "completed": replacement_delta == 0,
        },
        "capacity_boundary": {"completed": args.execution_profile == "capacity"},
        "cpu": {
            "physical_cores": len(selected) if args.backend == "cpu" else None,
            "kokkos_threads": args.threads if args.backend == "cpu" else None,
            "blas_threads": 1 if args.backend == "cpu" else None,
            "logical_cpu_ids": selected if args.backend == "cpu" else None,
            "topology": selected_topology if args.backend == "cpu" else None,
            "observed_kokkos_threads": observed_runtime.get("kokkos_concurrency"),
            "observed_openmp_threads": observed_runtime.get("omp_max_threads"),
        },
        "timing_contract": {
            "unit": "us/atom",
            "timed_region": "compute_prepared+reduce_forces+reduce_stress",
            "excluded": [
                "artifact_compilation",
                "model_preparation",
                "graph_construction",
                "neighbor_construction",
                "warmup",
            ],
        },
        "execution_diagnostic": evaluator.execution_diagnostic,
        "workspace_bytes": evaluator.execution_workspace_bytes,
        "topology_bytes": evaluator.execution_topology_bytes,
        "linear_workspace_bytes": evaluator.execution_linear_workspace_bytes,
        "model_preparation_seconds": model_preparation_seconds,
    }
    diagnostic_fields = {
        "conditioned_self_layers": "conditioned_self_layer_count",
        "conditioned_self_forward_launches": "conditioned_self_forward_launch_count",
        "conditioned_self_reverse_launches": "conditioned_self_reverse_launch_count",
    }
    for name, value in re.findall(
        r"\b(conditioned_self_[a-z_]+)=(\d+)", evaluator.execution_diagnostic
    ):
        record[diagnostic_fields.get(name, name)] = int(value)
    if args.backend == "cuda" and args.routing == "direct":
        from symmetrix import oam_cuda_codegen

        record.update(
            compiler=artifact.compiler,
            threads_per_block=args.threads_per_block,
            forward_threads_per_block=getattr(
                oam_cuda_codegen, "FORWARD_THREADS_PER_BLOCK", args.threads_per_block
            ),
            cuda_launch_count=evaluator.cuda_r1_launch_count,
            cuda_fallback_count=evaluator.cuda_r1_fallback_count,
        )
    if args.arrays:
        np.savez(
            args.arrays,
            energies=energies,
            forces=forces,
            stress=stress,
            positions=positions,
            numbers=atoms.numbers,
            cell=cell,
            pbc=atoms.pbc,
        )
    del evaluator
    native._finalize_kokkos()
    text = json.dumps(record, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
