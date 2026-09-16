"""Benchmark native serial or Kokkos MACE-MH-1 execution."""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_THREADS", "1")
BLAS_THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_BLAS_THREADS", "1")
KOKKOS_THREAD_VARIABLES = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
)
BLAS_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _configure_runtime_before_imports():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cpus")
    early_args, _ = parser.parse_known_args()
    if early_args.cpu is not None and early_args.cpus is not None:
        parser.error("--cpu and --cpus are mutually exclusive")

    for variable in KOKKOS_THREAD_VARIABLES:
        os.environ[variable] = THREAD_COUNT
    for variable in BLAS_THREAD_VARIABLES:
        os.environ[variable] = BLAS_THREAD_COUNT
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    requested_cpus = None
    if early_args.cpu is not None:
        requested_cpus = {early_args.cpu}
    elif early_args.cpus is not None:
        try:
            requested_cpus = {int(value) for value in early_args.cpus.split(",")}
        except ValueError:
            parser.error("--cpus must be a comma-separated integer list")
        if not requested_cpus:
            parser.error("--cpus cannot be empty")
    if requested_cpus is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu/--cpus require sched_setaffinity support")
        try:
            os.sched_setaffinity(0, requested_cpus)
        except OSError as error:
            parser.error(f"could not apply requested CPU affinity: {error}")
        if os.sched_getaffinity(0) != requested_cpus:
            parser.error(f"could not enforce CPU affinity {sorted(requested_cpus)}")


def _bootstrap_explicit_symmetrix():
    source_root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension_path = os.environ.get("SYMMETRIX_EXTENSION")
    if not source_root or not extension_path:
        return
    import importlib.util

    package_dir = pathlib.Path(source_root).resolve() / "symmetrix/source/symmetrix"
    extension = pathlib.Path(extension_path).resolve()
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if type(finder).__name__ != "ScikitBuildRedirectingFinder"
    ]
    package_spec = importlib.util.spec_from_file_location(
        "symmetrix",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    extension_module = extension.name.split(".", 1)[0]
    native_module_name = f"symmetrix.{extension_module}"
    native_spec = importlib.util.spec_from_file_location(native_module_name, extension)
    package = importlib.util.module_from_spec(package_spec)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix"] = package
    sys.modules[native_module_name] = native
    if extension_module == "symmetrix":
        sys.modules["symmetrix.symmetrix"] = native
    native_spec.loader.exec_module(native)
    package_spec.loader.exec_module(package)


_configure_runtime_before_imports()
_bootstrap_explicit_symmetrix()


THREAD_VARIABLES = (
    KOKKOS_THREAD_VARIABLES
    + BLAS_THREAD_VARIABLES
    + (
        "OMP_PROC_BIND",
        "OMP_PLACES",
    )
)

import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from symmetrix import Symmetrix  # noqa: E402
from symmetrix import symmetrix as native_symmetrix  # noqa: E402

STREAMED_EDGE_ALIASES = {
    "all_interactions": "generic",
    "factorized": "direct",
    "direct_streamed": "direct",
}
STREAMED_EDGE_MODES = ("materialized", "generic", "direct")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _git_tracked_diff_sha256():
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _cpu_model():
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or None


def _native_build_metadata(extension_path):
    cache = extension_path.parent / "CMakeCache.txt"
    keys = {
        "CMAKE_BUILD_TYPE",
        "SYMMETRIX_KOKKOS",
        "SYMMETRIX_SPHERICART_CUDA",
        "Kokkos_ENABLE_OPENMP",
        "Kokkos_ENABLE_SERIAL",
        "Kokkos_ENABLE_CUDA",
        "Kokkos_ENABLE_HIP",
        "Kokkos_ENABLE_SYCL",
        "Kokkos_ENABLE_OPENMPTARGET",
        "KokkosKernels_ENABLE_TPL_BLAS",
    }
    values = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            declaration, value = line.split("=", 1)
            key = declaration.split(":", 1)[0]
            if key in keys:
                values[key] = value
    return {"cmake_cache": str(cache) if cache.is_file() else None, "values": values}


def _thread_affinities():
    task_directory = pathlib.Path("/proc/self/task")
    if not task_directory.is_dir():
        return None
    affinities = {}
    for task in task_directory.iterdir():
        status = task / "status"
        try:
            lines = status.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("Cpus_allowed_list"):
                affinities[task.name] = line.partition(":")[2].strip()
                break
    return affinities


def _current_rss_bytes():
    statm = pathlib.Path("/proc/self/statm")
    if not statm.is_file():
        raise RuntimeError("lifecycle RSS measurement requires Linux /proc/self/statm")
    resident_pages = int(statm.read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _samples_summary(samples, atom_count):
    return {
        "median_ms": statistics.median(samples),
        "median_ms_per_atom": statistics.median(samples) / atom_count,
        "min_ms": min(samples),
        "min_ms_per_atom": min(samples) / atom_count,
        "max_ms": max(samples),
        "max_ms_per_atom": max(samples) / atom_count,
        "samples_ms": samples,
    }


def _md_timing_summary(samples_ms, atom_count):
    median_ms = statistics.median(samples_ms)
    return {
        "boundary": "one complete ASE VelocityVerlet NVE step",
        "steps": len(samples_ms),
        "samples_ms": samples_ms,
        "total_md_ms": sum(samples_ms),
        "median_ms": median_ms,
        "median_us_per_atom_per_step": 1000.0 * median_ms / atom_count,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "atom_steps_per_second": atom_count * 1000.0 / median_ms,
    }


def _array_sha256(values):
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _peak_rss_bytes():
    status = pathlib.Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _gpu_process_memory_mib():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    used_mib = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            try:
                used_mib += int(fields[1])
            except ValueError:
                return None
    return used_mib


def _jit_metadata(calculator):
    return {
        "jit_requested": getattr(calculator, "jit_policy", None),
        "jit_status": getattr(calculator, "jit_status", None),
        "jit_reason": getattr(calculator, "jit_reason", None),
        "jit_diagnostics": getattr(calculator, "jit_diagnostics", None),
        "jit_cache_key": getattr(calculator, "jit_cache_key", None),
        "jit_artifact_path": getattr(calculator, "jit_artifact_path", None),
        "jit_artifact_id": getattr(calculator, "jit_artifact_id", None),
        "jit_variant_id": getattr(calculator, "jit_variant_id", None),
        "jit_compiler_request": getattr(calculator, "jit_compiler_request", None),
        "jit_compiler_backend": getattr(calculator, "jit_compiler_backend", None),
        "jit_forward_policy": getattr(calculator, "jit_forward_policy", None),
        "jit_source_policy": getattr(calculator, "jit_source_policy", None),
        "jit_edge_policy": getattr(calculator, "jit_edge_policy", None),
    }


def _benchmark(calculator, atoms, warmups, repeats, include_ase):
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], inputs[5].flatten(), inputs[6])
    prepare_graph = getattr(calculator.evaluator, "_prepare_factorized_graph", None)
    compute_prepared = getattr(
        calculator.evaluator, "_compute_prepared_factorized", None
    )
    prepared = (
        getattr(calculator.evaluator, "streamed_edges_mode", None) == "direct"
        and callable(prepare_graph)
        and callable(compute_prepared)
    )
    if prepared:
        graph_generation = prepare_graph(*inputs[:5])

        def evaluate_native():
            compute_prepared(
                graph_generation,
                np.ascontiguousarray(inputs[5]).reshape(-1),
                np.ascontiguousarray(inputs[6]),
            )
    else:

        def evaluate_native():
            calculator.evaluator.compute_node_energies_forces(*native_args)

    rss_before = _current_rss_bytes()
    cuda = (
        calculator.use_kokkos
        and native_symmetrix._kokkos_default_execution_space() == "Cuda"
    )
    gpu_memory_before = _gpu_process_memory_mib() if cuda else None
    synchronize = getattr(calculator.evaluator, "fence", lambda: None)
    for _ in range(warmups):
        synchronize()
        evaluate_native()
        synchronize()
    evaluator_samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        evaluate_native()
        synchronize()
        evaluator_samples.append(1000.0 * (time.perf_counter() - start))
    evaluator_results = calculator._collect_mace_results(atoms, inputs)

    ase_samples = []
    if include_ase:
        for _ in range(warmups):
            synchronize()
            calculator.calculate(atoms.copy(), properties=["energy", "forces"])
            synchronize()
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            calculator.calculate(atoms.copy(), properties=["energy", "forces"])
            synchronize()
            ase_samples.append(1000.0 * (time.perf_counter() - start))

    edge_workspace_bytes = getattr(calculator.evaluator, "edge_workspace_bytes", None)
    selected_e3_linear_backend = getattr(
        calculator.evaluator, "selected_e3_linear_backend", None
    )
    return {
        "atoms": len(atoms),
        "model_cutoff_A": calculator.cutoff,
        "neighbor_skin_A": calculator.neighbor_skin,
        "effective_neighbor_cutoff_A": calculator.cutoff + calculator.neighbor_skin,
        "directed_edges": len(inputs[6]),
        "prepared_graph": prepared,
        **_jit_metadata(calculator),
        "execution_mh1_execution_backend": getattr(
            calculator.evaluator, "execution_mh1_execution_backend", None
        ),
        "jit_mh1_host_plugin_ready": getattr(
            calculator.evaluator, "jit_mh1_host_plugin_ready", None
        ),
        "jit_mh1_host_plugin_v4_ready": getattr(
            calculator.evaluator, "jit_mh1_host_plugin_v4_ready", None
        ),
        "jit_mh1_host_plugin_artifact_id": getattr(
            calculator.evaluator, "jit_mh1_host_plugin_artifact_id", None
        ),
        "jit_mh1_cuda_plugin_ready": getattr(
            calculator.evaluator, "jit_mh1_cuda_plugin_ready", None
        ),
        "jit_mh1_cuda_plugin_v4_ready": getattr(
            calculator.evaluator, "jit_mh1_cuda_plugin_v4_ready", None
        ),
        "jit_mh1_cuda_plugin_artifact_id": getattr(
            calculator.evaluator, "jit_mh1_cuda_plugin_artifact_id", None
        ),
        "factorized_graph_generation": getattr(
            calculator.evaluator, "factorized_graph_generation", None
        ),
        "factorized_prepared_graph_count": getattr(
            calculator.evaluator, "factorized_prepared_graph_count", None
        ),
        "factorized_prepared_evaluation_count": getattr(
            calculator.evaluator, "factorized_prepared_evaluation_count", None
        ),
        "factorized_fallback_evaluation_count": getattr(
            calculator.evaluator, "factorized_fallback_evaluation_count", None
        ),
        "factorized_topology_validation_count": getattr(
            calculator.evaluator, "factorized_topology_validation_count", None
        ),
        "factorized_topology_validation_skip_count": getattr(
            calculator.evaluator,
            "factorized_topology_validation_skip_count",
            None,
        ),
        "factorized_preparation_fence_count": getattr(
            calculator.evaluator, "factorized_preparation_fence_count", None
        ),
        "factorized_evaluation_fence_count": getattr(
            calculator.evaluator, "factorized_evaluation_fence_count", None
        ),
        "factorized_stage_fence_count": getattr(
            calculator.evaluator, "factorized_stage_fence_count", None
        ),
        "factorized_zbl_evaluator_stream_launch_count": getattr(
            calculator.evaluator,
            "factorized_zbl_evaluator_stream_launch_count",
            None,
        ),
        "execution_geometry_capacity_edges": getattr(
            calculator.evaluator, "execution_geometry_capacity_edges", None
        ),
        "execution_geometry_workspace_bytes": getattr(
            calculator.evaluator, "execution_geometry_workspace_bytes", None
        ),
        "execution_geometry_allocation_count": getattr(
            calculator.evaluator, "execution_geometry_allocation_count", None
        ),
        "execution_geometry_copy_count": getattr(
            calculator.evaluator, "execution_geometry_copy_count", None
        ),
        "execution_sphericart_launch_count": getattr(
            calculator.evaluator, "execution_sphericart_launch_count", None
        ),
        "execution_sphericart_async_launch_count": getattr(
            calculator.evaluator, "execution_sphericart_async_launch_count", None
        ),
        "execution_mh1_generated_conditioning_forward_launch_count": getattr(
            calculator.evaluator,
            "execution_mh1_generated_conditioning_forward_launch_count",
            None,
        ),
        "execution_mh1_generated_conditioning_reverse_launch_count": getattr(
            calculator.evaluator,
            "execution_mh1_generated_conditioning_reverse_launch_count",
            None,
        ),
        "execution_mh1_scratch_budget_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_budget_bytes", None
        ),
        "execution_mh1_scratch_minimum_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_minimum_bytes", None
        ),
        "execution_mh1_scratch_planned_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_planned_bytes", None
        ),
        "execution_mh1_scratch_retained_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_retained_bytes", None
        ),
        "execution_mh1_scratch_recomputed_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_recomputed_bytes", None
        ),
        "execution_mh1_scratch_recomputed_layer_count": getattr(
            calculator.evaluator,
            "execution_mh1_scratch_recomputed_layer_count",
            None,
        ),
        "execution_mh1_scratch_budget_satisfied": getattr(
            calculator.evaluator, "execution_mh1_scratch_budget_satisfied", None
        ),
        "execution_mh1_conditioning_recomputation_count": getattr(
            calculator.evaluator,
            "execution_mh1_conditioning_recomputation_count",
            None,
        ),
        "execution_mh1_retained_interaction_output_dimensions": list(
            getattr(
                calculator.evaluator,
                "execution_mh1_retained_interaction_output_dimensions",
                (),
            )
        ),
        "evaluator": _samples_summary(evaluator_samples, len(atoms)),
        "ase": _samples_summary(ase_samples, len(atoms)) if ase_samples else None,
        "rss_before_mib": rss_before / 2**20,
        "rss_after_mib": _current_rss_bytes() / 2**20,
        "peak_rss_mib": (
            _peak_rss_bytes() / 2**20 if _peak_rss_bytes() is not None else None
        ),
        "gpu_process_memory_before_mib": gpu_memory_before,
        "gpu_process_memory_after_mib": _gpu_process_memory_mib() if cuda else None,
        "edge_workspace_rows": getattr(
            calculator.evaluator, "edge_workspace_rows", None
        ),
        "edge_workspace_bytes": edge_workspace_bytes,
        "edge_workspace_mib": (
            edge_workspace_bytes / 2**20 if edge_workspace_bytes is not None else None
        ),
        "conditioned_mlp_workspace_bytes": getattr(
            calculator.evaluator, "conditioned_mlp_workspace_bytes", None
        ),
        "e3_linear_backend": getattr(calculator.evaluator, "e3_linear_backend", None),
        "selected_e3_linear_backend": (
            selected_e3_linear_backend(len(atoms))
            if selected_e3_linear_backend is not None
            else None
        ),
        "tensor_product_backend": getattr(
            calculator.evaluator, "tensor_product_backend", None
        ),
        "tensor_product_execution_backend": getattr(
            calculator.evaluator, "tensor_product_execution_backend", None
        ),
        "tensor_product_channel_team_size": getattr(
            calculator.evaluator, "tensor_product_channel_team_size", None
        ),
        "tensor_product_harmonic_team_size": getattr(
            calculator.evaluator, "tensor_product_harmonic_team_size", None
        ),
        "linear_workspace_bytes": getattr(
            calculator.evaluator, "linear_workspace_bytes", None
        ),
        "tensor_workspace_bytes": getattr(
            calculator.evaluator, "tensor_workspace_bytes", None
        ),
        "product_workspace_bytes": getattr(
            calculator.evaluator, "product_workspace_bytes", None
        ),
        "node_workspace_bytes": getattr(
            calculator.evaluator, "node_workspace_bytes", None
        ),
        "precision_workspace_bytes": getattr(
            calculator.evaluator, "precision_workspace_bytes", edge_workspace_bytes
        ),
        "energy_eV": float(evaluator_results["energy"]),
        "force_l2_eV_per_A": float(np.linalg.norm(evaluator_results["forces"])),
        "force_sum_eV_per_A": np.sum(evaluator_results["forces"], axis=0).tolist(),
    }


def _model_metadata(path):
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _parse_sizes(value, parser, option):
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError:
        parser.error(f"{option} must be a comma-separated integer list")
    if not sizes or any(size < 1 for size in sizes):
        parser.error(f"{option} values must be positive")
    return sizes


def _aln_repeat_from_atom_count(atom_count):
    if atom_count % 4 != 0:
        raise ValueError(
            f"AlN atom count {atom_count} is not a wurtzite 4*n^3 supercell"
        )
    repeat = round((atom_count // 4) ** (1.0 / 3.0))
    if repeat < 1 or 4 * repeat**3 != atom_count:
        raise ValueError(
            f"AlN atom count {atom_count} is not a wurtzite 4*n^3 supercell"
        )
    return repeat


def _lifecycle(calculator, sizes, cycles):
    systems = [bulk("Si", "diamond", a=5.43).repeat((size,) * 3) for size in sizes]
    for _ in range(3):
        for atoms in systems:
            calculator.calculate(
                atoms.copy(), properties=["energy", "forces", "stress"]
            )
    rss_bytes = []
    for _ in range(cycles):
        for atoms in systems:
            calculator.calculate(
                atoms.copy(), properties=["energy", "forces", "stress"]
            )
        rss_bytes.append(_current_rss_bytes())
    return {
        "sizes": sizes,
        "atoms": [len(atoms) for atoms in systems],
        "cycles": cycles,
        "evaluations": cycles * len(systems),
        "rss_mib": [value / 2**20 for value in rss_bytes],
        "min_rss_mib": min(rss_bytes) / 2**20,
        "max_rss_mib": max(rss_bytes) / 2**20,
        "growth_mib": (max(rss_bytes) - min(rss_bytes)) / 2**20,
    }


def _build_md_atoms(repeat, seed, temperature_K, position_noise_std_A):
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(
        scale=position_noise_std_A, size=atoms.positions.shape
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms, preserve_temperature=True)
    return atoms


def _md_runtime_snapshot(calculator):
    evaluator = calculator.evaluator
    cache = getattr(calculator, "_neighbor_cache", None)
    return {
        "neighbor_cache_build_count": int(calculator.neighbor_cache_build_count),
        "neighbor_cache_reuse_count": int(calculator.neighbor_cache_reuse_count),
        "neighbor_cache_geometry_update_count": int(
            calculator.neighbor_cache_geometry_update_count
        ),
        "candidate_directed_edges": (
            len(cache.receivers) if cache is not None else None
        ),
        "active_directed_edges": getattr(
            evaluator, "execution_active_edge_count", None
        ),
        "factorized_prepared_graph_count": getattr(
            evaluator, "factorized_prepared_graph_count", None
        ),
        "factorized_prepared_evaluation_count": getattr(
            evaluator, "factorized_prepared_evaluation_count", None
        ),
        "factorized_fallback_evaluation_count": getattr(
            evaluator, "factorized_fallback_evaluation_count", None
        ),
        "factorized_schedule_build_count": getattr(
            evaluator, "factorized_schedule_build_count", None
        ),
        "factorized_schedule_entries": getattr(
            evaluator, "factorized_schedule_entries", None
        ),
        "edge_workspace_bytes": getattr(evaluator, "edge_workspace_bytes", None),
        "execution_geometry_workspace_bytes": getattr(
            evaluator, "execution_geometry_workspace_bytes", None
        ),
        "node_workspace_bytes": getattr(evaluator, "node_workspace_bytes", None),
        "precision_workspace_bytes": getattr(
            evaluator, "precision_workspace_bytes", None
        ),
    }


def _benchmark_md(
    calculator,
    atoms,
    *,
    steps,
    timestep_fs,
    seed,
    requested_temperature_K,
    position_noise_std_A,
):
    from ase import units
    from ase.md.verlet import VelocityVerlet

    synchronize = getattr(calculator.evaluator, "fence", lambda: None)
    atoms.calc = calculator
    initial_positions = np.asarray(atoms.positions, dtype=float).copy()
    actual_initial_temperature_K = float(atoms.get_temperature())

    synchronize()
    initial_start = time.perf_counter()
    initial_forces = np.asarray(atoms.get_forces(), dtype=float).copy()
    initial_energy = float(atoms.get_potential_energy())
    synchronize()
    initial_evaluation_ms = 1000.0 * (time.perf_counter() - initial_start)
    initial_total_energy = initial_energy + float(atoms.get_kinetic_energy())
    runtime_after_initial = _md_runtime_snapshot(calculator)

    dynamics = VelocityVerlet(atoms, timestep=timestep_fs * units.fs, logfile=None)
    samples_ms = []
    trajectory = []
    for step in range(1, steps + 1):
        synchronize()
        start = time.perf_counter()
        dynamics.run(1)
        synchronize()
        samples_ms.append(1000.0 * (time.perf_counter() - start))

        forces = np.asarray(calculator.results["forces"], dtype=float)
        potential = float(calculator.results["energy"])
        kinetic = float(atoms.get_kinetic_energy())
        runtime = _md_runtime_snapshot(calculator)
        trajectory.append(
            {
                "step": step,
                "potential_energy_eV": potential,
                "kinetic_energy_eV": kinetic,
                "total_energy_eV": potential + kinetic,
                "temperature_K": float(atoms.get_temperature()),
                "max_abs_force_eV_per_A": float(np.max(np.abs(forces))),
                "max_displacement_A": float(
                    np.max(np.linalg.norm(atoms.positions - initial_positions, axis=1))
                ),
                "candidate_directed_edges": runtime["candidate_directed_edges"],
                "active_directed_edges": runtime["active_directed_edges"],
            }
        )

    runtime_final = _md_runtime_snapshot(calculator)
    expected = {
        "neighbor_cache_build_count": 1,
        "neighbor_cache_reuse_count": steps,
        "factorized_prepared_graph_count": 1,
        "factorized_prepared_evaluation_count": steps + 1,
        "factorized_fallback_evaluation_count": 0,
    }
    observed = {name: runtime_final[name] for name in expected}
    mismatches = {
        name: {"expected": expected_value, "observed": observed[name]}
        for name, expected_value in expected.items()
        if observed[name] != expected_value
    }
    if mismatches:
        raise RuntimeError(f"MH-1 MD reuse contract failed: {mismatches}")

    total_energy_final = trajectory[-1]["total_energy_eV"]
    return {
        "atoms": len(atoms),
        "model_cutoff_A": calculator.cutoff,
        "neighbor_skin_A": calculator.neighbor_skin,
        "effective_neighbor_cutoff_A": calculator.cutoff + calculator.neighbor_skin,
        "case": {
            "name": "wurtzite-AlN",
            "repeat": _aln_repeat_from_atom_count(len(atoms)),
            "seed": seed,
            "position_noise_std_A": position_noise_std_A,
            "requested_initial_temperature_K": requested_temperature_K,
            "actual_initial_temperature_K": actual_initial_temperature_K,
        },
        "md": {
            "ensemble": "NVE",
            "integrator": "ASE VelocityVerlet",
            "steps": steps,
            "timestep_fs": timestep_fs,
            "velocity_distribution": "MaxwellBoltzmannDistribution",
            "center_of_mass_removed": True,
        },
        "timing": {
            **_md_timing_summary(samples_ms, len(atoms)),
            "initial_evaluation_ms": initial_evaluation_ms,
            "initial_evaluation_included": False,
        },
        "runtime_after_initial_evaluation": runtime_after_initial,
        "runtime_final": runtime_final,
        "reuse_contract": {
            "expected": expected,
            "observed": observed,
            "passed": True,
        },
        "physics": {
            "initial_energy_eV": initial_energy,
            "initial_forces_sha256": _array_sha256(initial_forces),
            "total_energy_drift_eV_per_atom": (
                total_energy_final - initial_total_energy
            )
            / len(atoms),
            "temperature_min_K": min(x["temperature_K"] for x in trajectory),
            "temperature_max_K": max(x["temperature_K"] for x in trajectory),
            "final_max_displacement_A": trajectory[-1]["max_displacement_A"],
        },
        "trajectory": trajectory,
        "rss_after_mib": _current_rss_bytes() / 2**20,
        "peak_rss_mib": (
            _peak_rss_bytes() / 2**20 if _peak_rss_bytes() is not None else None
        ),
        "gpu_process_memory_after_mib": _gpu_process_memory_mib(),
        **_jit_metadata(calculator),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=pathlib.Path, help="Extracted MACE-MH-1 JSON model"
    )
    parser.add_argument("--backend", choices=("serial", "kokkos"), default="serial")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--reference-model", type=pathlib.Path, help="Optional native MH-0 JSON"
    )
    parser.add_argument("--max-reference-ratio", type=float)
    parser.add_argument(
        "--reference-streamed-edges",
        choices=(
            "auto",
            "materialized",
            "generic",
            "direct",
            "all_interactions",
            "factorized",
            "direct_streamed",
        ),
        default="auto",
    )
    parser.add_argument(
        "--reference-jit",
        dest="reference_jit",
        choices=("auto", "off", "required"),
        default="auto",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="Single CPU to pin (equivalent to a one-entry --cpus list)",
    )
    parser.add_argument(
        "--cpus",
        help="Comma-separated CPU affinity for Kokkos threads",
    )
    parser.add_argument("--sizes", default="1,2,3")
    parser.add_argument(
        "--atom-counts",
        help="Comma-separated exact AlN atom counts from 256,864,4000",
    )
    parser.add_argument(
        "--streamed-edges",
        choices=(
            "materialized",
            "generic",
            "direct",
            "all_interactions",
            "factorized",
            "direct_streamed",
        ),
        default="generic",
    )
    parser.add_argument(
        "--mh1-edge-executor",
        choices=("mlp_reference", "pair_spline_v1"),
        default="pair_spline_v1",
        help="MH-1 edge executor; pair_spline_v1 is the production default",
    )
    parser.add_argument(
        "--neighbor-skin",
        type=float,
        default=0.5,
        help="Neighbor-list skin in Angstrom; use zero for exact-cutoff comparisons",
    )
    parser.add_argument(
        "--factorized-jit",
        dest="jit",
        choices=("auto", "off", "required"),
        help=(
            "Direct JIT specialization policy; defaults to required with "
            "--streamed-edges direct and auto otherwise"
        ),
    )
    parser.add_argument(
        "--factorized-mh1-scratch-budget-bytes",
        dest="execution_mh1_scratch_budget_bytes",
        type=int,
        help=(
            "Maximum generated MH-1 compact-phi workspace; below the reported "
            "minimum, mandatory phase scratch exceeds the requested budget"
        ),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--md-steps",
        type=int,
        default=0,
        help="Run the qualified 20-step ASE NVE protocol instead of static calls",
    )
    parser.add_argument("--md-timestep-fs", type=float, default=1.0)
    parser.add_argument("--md-temperature-K", type=float, default=300.0)
    parser.add_argument("--md-seed", type=int, default=20260808)
    parser.add_argument("--md-position-noise-std-A", type=float, default=0.01)
    parser.add_argument(
        "--e3-linear-backend",
        choices=("auto", "scalar", "packed_gemm"),
        default="auto",
        help="Benchmark-only Kokkos E3-linear execution policy",
    )
    parser.add_argument(
        "--factorized-host-node-backend",
        choices=("auto", "generated", "kokkos"),
        default="auto",
        help=(
            "Benchmark-only MH-1 host node policy; kokkos retains generated "
            "factorized edge execution while using packed Kokkos node operators"
        ),
    )
    parser.add_argument(
        "--execution-profile",
        choices=("speed", "capacity"),
        default="speed",
        help=(
            "Calculator execution profile; capacity selects the public "
            "low-memory policy, while speed honors --mh1-node-state-policy"
        ),
    )
    parser.add_argument(
        "--mh1-node-arena-tile-rows",
        type=int,
        help="Benchmark-only generated MH-1 host node tile row override",
    )
    parser.add_argument(
        "--mh1-node-state-policy",
        choices=(
            "full-retention-v1",
            "recompute-v1",
            "reuse-adjoints-v1",
            "retain-interaction-v1",
        ),
        default="full-retention-v1",
    )
    parser.add_argument(
        "--fused-gate-normalization-reverse",
        choices=("auto", "on", "off"),
        default="auto",
        help="Benchmark-only MH-1 fused node-reverse policy",
    )
    parser.add_argument(
        "--direct-node-tensor-reverse",
        choices=("auto", "on", "off"),
        default="auto",
        help="Benchmark-only MH-1 direct-node tensor-reverse policy",
    )
    parser.add_argument("--evaluator-only", action="store_true")
    parser.add_argument(
        "--allow-generic",
        action="store_true",
        help="Allow the generic nonlinear evaluator for a control measurement",
    )
    parser.add_argument(
        "--lifecycle-sizes",
        help="Comma-separated supercell sizes to alternate for RSS measurement",
    )
    parser.add_argument("--lifecycle-cycles", type=int, default=20)
    parser.add_argument("--max-lifecycle-growth-mib", type=float)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    args.streamed_edges = STREAMED_EDGE_ALIASES.get(
        args.streamed_edges, args.streamed_edges
    )
    args.reference_streamed_edges = STREAMED_EDGE_ALIASES.get(
        args.reference_streamed_edges, args.reference_streamed_edges
    )
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if args.md_steps not in (0, 20):
        parser.error("--md-steps must be 20 when the MD protocol is enabled")
    if args.md_steps:
        if args.neighbor_skin <= 0.0:
            parser.error("--md-steps requires a positive --neighbor-skin")
        if args.md_timestep_fs <= 0.0:
            parser.error("--md-timestep-fs must be positive")
        if args.md_temperature_K <= 0.0:
            parser.error("--md-temperature-K must be positive")
        if args.md_position_noise_std_A < 0.0:
            parser.error("--md-position-noise-std-A must be nonnegative")
        if args.atom_counts is None or len(args.atom_counts.split(",")) != 1:
            parser.error("--md-steps requires exactly one --atom-counts value")
        if args.evaluator_only:
            parser.error("--md-steps is incompatible with --evaluator-only")
        if args.reference_model is not None:
            parser.error("--md-steps is incompatible with --reference-model")
        if args.lifecycle_sizes is not None:
            parser.error("--md-steps is incompatible with --lifecycle-sizes")
    if args.mh1_node_arena_tile_rows is not None and args.mh1_node_arena_tile_rows < 1:
        parser.error("--mh1-node-arena-tile-rows must be positive")
    if not np.isfinite(args.neighbor_skin) or args.neighbor_skin < 0.0:
        parser.error("--neighbor-skin must be a finite non-negative number")
    if args.atom_counts is not None and args.sizes != "1,2,3":
        parser.error("--atom-counts and an explicit --sizes are mutually exclusive")
    if args.lifecycle_cycles < 1:
        parser.error("--lifecycle-cycles must be positive")
    if args.max_lifecycle_growth_mib is not None and args.lifecycle_sizes is None:
        parser.error("--max-lifecycle-growth-mib requires --lifecycle-sizes")
    if args.max_reference_ratio is not None and args.reference_model is None:
        parser.error("--max-reference-ratio requires --reference-model")
    if args.reference_model is None and (
        args.reference_streamed_edges != "auto" or args.reference_jit != "auto"
    ):
        parser.error("reference execution controls require --reference-model")
    if (
        args.execution_mh1_scratch_budget_bytes is not None
        and args.execution_mh1_scratch_budget_bytes < 0
    ):
        parser.error("--factorized-mh1-scratch-budget-bytes must be nonnegative")
    if args.execution_mh1_scratch_budget_bytes is not None and (
        args.backend != "kokkos" or args.streamed_edges != "direct"
    ):
        parser.error(
            "--factorized-mh1-scratch-budget-bytes requires --backend kokkos "
            "and --streamed-edges direct"
        )
    if args.cpu is not None and args.cpus is not None:
        parser.error("--cpu and --cpus are mutually exclusive")
    if args.backend != "kokkos" and (
        args.fused_gate_normalization_reverse != "auto"
        or args.direct_node_tensor_reverse != "auto"
    ):
        parser.error("fused reverse controls require --backend kokkos")
    if args.direct_node_tensor_reverse == "on" and args.streamed_edges != "generic":
        parser.error(
            "--direct-node-tensor-reverse on requires --streamed-edges generic"
        )
    if hasattr(os, "sched_getaffinity"):
        available_cpus = os.sched_getaffinity(0)
        requested_cpus = None
        if args.cpu is not None:
            requested_cpus = {args.cpu}
        elif args.cpus is not None:
            try:
                requested_cpus = {int(value) for value in args.cpus.split(",")}
            except ValueError:
                parser.error("--cpus must be a comma-separated integer list")
            if not requested_cpus:
                parser.error("--cpus cannot be empty")
        if requested_cpus is not None:
            try:
                os.sched_setaffinity(0, requested_cpus)
            except OSError as error:
                parser.error(f"could not apply requested CPU affinity: {error}")
            applied_cpus = os.sched_getaffinity(0)
            if applied_cpus != requested_cpus:
                parser.error(
                    f"requested CPUs {sorted(requested_cpus)} produced affinity "
                    f"{sorted(applied_cpus)}"
                )
        elif len(available_cpus) != int(THREAD_COUNT):
            parser.error(
                "use --cpu/--cpus or launch under taskset with exactly "
                f"{THREAD_COUNT} available CPUs"
            )

    use_kokkos = args.backend == "kokkos"
    jit = args.jit
    if jit is None:
        jit = "required" if args.streamed_edges == "direct" else "auto"
    calculator = Symmetrix(
        args.model,
        use_kokkos=use_kokkos,
        dtype=args.dtype,
        streamed_edges=args.streamed_edges,
        execution_profile=args.execution_profile,
        jit=jit,
        execution_mh1_scratch_budget_bytes=args.execution_mh1_scratch_budget_bytes,
        execution_mh1_node_state_policy=args.mh1_node_state_policy,
        neighbor_skin=args.neighbor_skin,
    )
    if use_kokkos:
        calculator.evaluator.set_mh1_edge_executor(args.mh1_edge_executor)
        calculator.evaluator.set_e3_linear_backend(args.e3_linear_backend)
        calculator.evaluator.set_execution_mh1_host_node_backend(
            args.factorized_host_node_backend
        )
        if args.mh1_node_arena_tile_rows is not None:
            calculator.evaluator._set_execution_mh1_node_arena_tile_rows_for_testing(
                args.mh1_node_arena_tile_rows
            )
        controls = (
            (
                args.fused_gate_normalization_reverse,
                "set_fused_gate_normalization_reverse",
            ),
            (
                args.direct_node_tensor_reverse,
                "set_direct_node_tensor_reverse",
            ),
        )
        for requested, setter_name in controls:
            if requested == "auto":
                continue
            setter = getattr(calculator.evaluator, setter_name, None)
            if setter is None:
                raise RuntimeError(f"Native evaluator does not expose {setter_name}")
            setter(requested == "on")
    if not args.allow_generic and not getattr(
        calculator.evaluator, "uses_mh1_fast_path", False
    ):
        raise RuntimeError(
            f"Model does not match the specialized MACE-MH-1 {args.backend} architecture"
        )
    reference = (
        Symmetrix(
            args.reference_model,
            use_kokkos=use_kokkos,
            dtype=args.dtype,
            streamed_edges=args.reference_streamed_edges,
            jit=args.reference_jit,
        )
        if args.reference_model
        else None
    )

    extension_path = pathlib.Path(native_symmetrix.__file__).resolve()
    device_environment = getattr(
        calculator.evaluator, "execution_device_execution_environment", None
    )
    if device_environment is None:
        device_environment = getattr(
            native_symmetrix, "_execution_device_execution_environment", lambda: None
        )
    device_environment = (
        device_environment() if callable(device_environment) else device_environment
    )
    metadata = {
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "git_tracked_diff_sha256": _git_tracked_diff_sha256(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_affinities": _thread_affinities(),
        "backend": args.backend,
        "dtype": args.dtype,
        "scalar_size_bytes": getattr(calculator.evaluator, "scalar_size_bytes", 8),
        "is_mh1_family": bool(getattr(calculator.evaluator, "is_mh1_family", False)),
        "mh1_node_channels": getattr(calculator.evaluator, "mh1_node_channels", None),
        "mh1_edge_channels": getattr(calculator.evaluator, "mh1_edge_channels", None),
        "mh1_radial_size": getattr(calculator.evaluator, "mh1_radial_size", None),
        "mh1_l_max": getattr(calculator.evaluator, "mh1_l_max", None),
        "mh1_family_rejection_reason": getattr(
            calculator.evaluator, "mh1_family_rejection_reason", None
        ),
        "mh1_uses_compiled_products": getattr(
            calculator.evaluator, "mh1_uses_compiled_products", None
        ),
        "mh1_uses_pair_conditioning": getattr(
            calculator.evaluator, "mh1_uses_pair_conditioning", None
        ),
        "mh1_uses_external_uvu_tensors": getattr(
            calculator.evaluator, "mh1_uses_external_uvu_tensors", None
        ),
        "mh1_fast_path_rejection_reason": getattr(
            calculator.evaluator, "mh1_fast_path_rejection_reason", None
        ),
        "uses_mh1_fast_path": bool(calculator.evaluator.uses_mh1_fast_path),
        "streamed_edges": calculator.streamed_edges,
        "mh1_edge_executor": calculator.evaluator.mh1_edge_executor,
        "execution_mh1_execution_backend": getattr(
            calculator.evaluator, "execution_mh1_execution_backend", None
        ),
        "neighbor_skin_A": args.neighbor_skin,
        "execution_mh1_scratch_budget_requested_bytes": (
            args.execution_mh1_scratch_budget_bytes
        ),
        **_jit_metadata(calculator),
        "e3_linear_backend": getattr(calculator.evaluator, "e3_linear_backend", None),
        "factorized_host_node_backend_requested": args.factorized_host_node_backend,
        "factorized_host_node_backend_selected": getattr(
            calculator.evaluator,
            "selected_execution_mh1_host_node_backend",
            None,
        ),
        "mh1_node_arena_tile_rows_requested": args.mh1_node_arena_tile_rows,
        "mh1_node_arena_tile_rows_selected": getattr(
            calculator.evaluator, "execution_mh1_node_arena_tile_rows", None
        ),
        "execution_profile": calculator.execution_profile,
        "mh1_node_state_policy_requested": args.mh1_node_state_policy,
        "mh1_node_state_policy": calculator.execution_mh1_node_state_policy,
        "tensor_product_execution_backend": getattr(
            calculator.evaluator, "tensor_product_execution_backend", None
        ),
        "tensor_product_channel_team_size": getattr(
            calculator.evaluator, "tensor_product_channel_team_size", None
        ),
        "tensor_product_harmonic_team_size": getattr(
            calculator.evaluator, "tensor_product_harmonic_team_size", None
        ),
        "fused_gate_normalization_reverse_requested": (
            args.fused_gate_normalization_reverse
        ),
        "fused_gate_normalization_reverse_available": getattr(
            calculator.evaluator,
            "fused_gate_normalization_reverse_available",
            None,
        ),
        "uses_fused_gate_normalization_reverse": getattr(
            calculator.evaluator, "uses_fused_gate_normalization_reverse", None
        ),
        "direct_node_tensor_reverse_requested": args.direct_node_tensor_reverse,
        "direct_node_tensor_reverse_available": getattr(
            calculator.evaluator, "direct_node_tensor_reverse_available", None
        ),
        "uses_direct_node_tensor_reverse": getattr(
            calculator.evaluator, "uses_direct_node_tensor_reverse", None
        ),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "md_steps": args.md_steps,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "native_extension": str(extension_path),
        "native_build": _native_build_metadata(extension_path),
        "device_environment": device_environment,
        "model": _model_metadata(args.model),
        "reference_model": _model_metadata(args.reference_model)
        if args.reference_model
        else None,
        "timing_scope": {
            "evaluator": "prebuilt graph; native energy and analytic-force kernel only",
            "ase": "Symmetrix.calculate including graph construction and result collection",
            "md": "one complete ASE VelocityVerlet NVE step",
        },
    }

    if args.atom_counts is None:
        systems = [
            bulk("Si", "diamond", a=5.43).repeat((size,) * 3)
            for size in _parse_sizes(args.sizes, parser, "--sizes")
        ]
    else:
        atom_counts = _parse_sizes(args.atom_counts, parser, "--atom-counts")
        try:
            repeats = [_aln_repeat_from_atom_count(count) for count in atom_counts]
        except ValueError as error:
            parser.error(str(error))
        systems = [
            (
                _build_md_atoms(
                    repeat,
                    args.md_seed,
                    args.md_temperature_K,
                    args.md_position_noise_std_A,
                )
                if args.md_steps
                else bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
            )
            for repeat in repeats
        ]

    results = []
    for atoms in systems:
        if args.md_steps:
            record = _benchmark_md(
                calculator,
                atoms,
                steps=args.md_steps,
                timestep_fs=args.md_timestep_fs,
                seed=args.md_seed,
                requested_temperature_K=args.md_temperature_K,
                position_noise_std_A=args.md_position_noise_std_A,
            )
        else:
            record = _benchmark(
                calculator, atoms, args.warmups, args.repeats, not args.evaluator_only
            )
        if reference is not None:
            reference_record = _benchmark(
                reference, atoms, args.warmups, args.repeats, not args.evaluator_only
            )
            record["reference"] = reference_record
            record["evaluator_reference_ratio"] = (
                record["evaluator"]["median_ms"]
                / reference_record["evaluator"]["median_ms"]
            )
            record["ase_reference_ratio"] = (
                record["ase"]["median_ms"] / reference_record["ase"]["median_ms"]
                if record["ase"] is not None
                else None
            )
            if (
                args.max_reference_ratio is not None
                and record["evaluator_reference_ratio"] > args.max_reference_ratio
            ):
                raise RuntimeError(
                    f"MH-1/MH-0 evaluator ratio {record['evaluator_reference_ratio']:.3f} "
                    f"exceeds {args.max_reference_ratio:.3f} for {len(atoms)} atoms"
                )
        results.append(record)
        print(json.dumps(record), flush=True)

    lifecycle = None
    if args.lifecycle_sizes is not None:
        lifecycle = _lifecycle(
            calculator,
            _parse_sizes(args.lifecycle_sizes, parser, "--lifecycle-sizes"),
            args.lifecycle_cycles,
        )
        print(json.dumps({"lifecycle": lifecycle}), flush=True)
        if (
            args.max_lifecycle_growth_mib is not None
            and lifecycle["growth_mib"] > args.max_lifecycle_growth_mib
        ):
            raise RuntimeError(
                f"lifecycle RSS growth {lifecycle['growth_mib']:.3f} MiB exceeds "
                f"{args.max_lifecycle_growth_mib:.3f} MiB"
            )

    output = {"metadata": metadata, "systems": results, "lifecycle": lifecycle}
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
