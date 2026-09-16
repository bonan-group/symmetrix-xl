"""ASE Calculator for symmetrix implementation of equivariant graph neural
network library

This file was written and publicly released by Dr. Noam Bernstein as part of his
work for the U. S. Government, and is not subject to copyright.
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

try:
    from matscipy.neighbours import neighbour_list as neighbor_list
except ImportError:
    logging.warning("Symmetrix using slow ase.neighborlist.neighbor_list")
    from ase.neighborlist import neighbor_list

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
    compare_atoms,
    equal,
)
from ase.stress import full_3x3_to_voigt_6_stress

from . import symmetrix

_LOGGER = logging.getLogger(__name__)
_FIELD_ADDITIVE_PROPERTIES = ["energy", "free_energy", "energies", "forces", "stress"]
_FIELD_RESPONSE_PROPERTIES = ["polarization", "becs", "polarizability"]
_FIELD_NOT_SET = object()
_LOW_MEMORY_UNSET = object()
_JIT_POLICY_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_POLICY"
_JIT_REQUIRED_POLICY = "required"
_JIT_DEBUG_POLICY = "none"
_EXECUTION_MH0_STATE_POLICIES = frozenset(("full-retention-v1", "reuse-adjoints-v1"))
_EDGE_GEOMETRY_POLICIES = frozenset(("cartesian-f64-v1", "unit-f32-radius-f64-v1"))
_EXECUTION_MH1_NODE_STATE_POLICIES = frozenset(
    (
        "full-retention-v1",
        "recompute-v1",
        "reuse-adjoints-v1",
        "retain-interaction-v1",
    )
)
_EXECUTION_MH1_NODE_ARENA_POLICIES = frozenset(("throughput-v1", "capacity-v1"))
_KERNEL_LAUNCH_POLICIES = frozenset(("automatic", "static"))
_M1_POLYNOMIAL_POLICIES = frozenset(("automatic", "recompute", "retained"))
_EXECUTION_PROFILES = frozenset(("capacity", "speed"))
_DEBUG_EXECUTION_PLANS = frozenset(
    (
        "mh0-direct-speed",
        "mh0-direct-capacity-retained",
        "mh0-direct-capacity-y-only",
        "mh0-single-layer-tiled-v1",
        "mh0-dual-layer-tiled-v1",
    )
)
_JIT_HOST_EXECUTION_SPACES = frozenset(("Serial", "OpenMP"))
_STREAMED_EDGE_ALIASES = {
    "non-compiled": "generic",
    "generic": "generic",
    "all_interactions": "generic",
    "factorized": "direct",
    "direct_streamed": "direct",
}
_STREAMED_EDGE_CANONICAL_MODES = frozenset(("materialized", "generic", "direct"))
_PREPARED_EXECUTION_MODES = frozenset(("direct",))
_FIXED_WORKSPACE_TILED_PLANS = frozenset(
    ("mh0-single-layer-tiled-v1", "mh0-dual-layer-tiled-v1")
)
_NEIGHBOR_EDGE_ORDER_ENVIRONMENT_VARIABLE = "SYMMETRIX_DEBUG_NEIGHBOR_EDGE_ORDER"
_NEIGHBOR_EDGE_ORDERS = frozenset(("receiver-major", "canonical"))
_NEIGHBOR_BACKENDS = frozenset(("automatic", "host", "kokkos"))
_AUTOMATIC_KOKKOS_NEIGHBOR_MIN_ATOMS = 512
_AUTOMATIC_KOKKOS_NEIGHBOR_EXECUTION_SPACES = frozenset(("Cuda", "HIP"))
_INT32_MAX = int(np.iinfo(np.int32).max)


def _validate_int32_graph_cardinality(num_atoms, num_edges):
    num_atoms = int(num_atoms)
    num_edges = int(num_edges)
    if num_atoms < 0 or num_edges < 0:
        raise ValueError(
            "MACE graph atom and directed-edge counts must be non-negative."
        )
    if num_atoms >= _INT32_MAX:
        raise ValueError(
            f"MACE graph atom count {num_atoms} exceeds the 32-bit CSR "
            f"limit {_INT32_MAX - 1}."
        )
    if num_edges > _INT32_MAX:
        raise ValueError(
            f"MACE graph directed-edge count {num_edges} exceeds the 32-bit "
            f"topology limit {_INT32_MAX}."
        )


def _receiver_major_neighbor_arrays(receivers, sources, shifts):
    """Return aligned, contiguous edges grouped by receiver.

    Matscipy and ASE already emit ascending receiver indices. Preserve their
    within-receiver order because direct execution has no source/shift sorting
    requirement. The canonical mode exists only for numerical A/B diagnosis.
    """
    receivers = np.asarray(receivers, dtype=np.int32).reshape(-1)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    shifts = np.asarray(shifts, dtype=np.int32).reshape((-1, 3))
    if len(receivers) != len(sources) or len(receivers) != len(shifts):
        raise ValueError("Neighbor receiver, source, and shift extents must match.")

    order_policy = os.environ.get(
        _NEIGHBOR_EDGE_ORDER_ENVIRONMENT_VARIABLE, "receiver-major"
    )
    if order_policy not in _NEIGHBOR_EDGE_ORDERS:
        choices = ", ".join(sorted(_NEIGHBOR_EDGE_ORDERS))
        raise ValueError(
            f"{_NEIGHBOR_EDGE_ORDER_ENVIRONMENT_VARIABLE} must be one of "
            f"{choices}; got {order_policy!r}."
        )
    if order_policy == "canonical":
        order = np.lexsort(
            (
                shifts[:, 2],
                shifts[:, 1],
                shifts[:, 0],
                sources,
                receivers,
            )
        )
    elif len(receivers) > 1 and np.any(receivers[1:] < receivers[:-1]):
        order = np.argsort(receivers, kind="stable")
    else:
        order = None

    if order is not None:
        receivers = receivers[order]
        sources = sources[order]
        shifts = shifts[order]
    return (
        np.ascontiguousarray(receivers),
        np.ascontiguousarray(sources),
        np.ascontiguousarray(shifts),
    )


def _neighbor_backend_request():
    request = os.environ.get("SYMMETRIX_NEIGHBOR_BACKEND", "automatic").strip().lower()
    if request not in _NEIGHBOR_BACKENDS:
        choices = ", ".join(sorted(_NEIGHBOR_BACKENDS))
        raise ValueError(
            f"SYMMETRIX_NEIGHBOR_BACKEND must be one of {choices}; got {request!r}."
        )
    return request


def _use_device_neighbor_graph(request, eligible, num_atoms, execution_space):
    if not eligible or request == "host":
        return False
    if request == "kokkos":
        return True
    return (
        execution_space in _AUTOMATIC_KOKKOS_NEIGHBOR_EXECUTION_SPACES
        and num_atoms >= _AUTOMATIC_KOKKOS_NEIGHBOR_MIN_ATOMS
    )


def _canonical_streamed_edges_mode(mode):
    if mode == "auto":
        return mode
    return _STREAMED_EDGE_ALIASES.get(mode, mode)


class _JitRequiredError(RuntimeError):
    pass


@dataclass(frozen=True)
class _JitArtifactAttempt:
    compiler: str
    source: str
    prepare: object
    arguments: dict


@dataclass(frozen=True)
class _FactorizedDeviceBackendPlan:
    backend: str
    target: dict
    normalized_target: object
    policy: str
    selected_compiler: str
    attempt_builders: dict
    loader: object
    ready_attribute: str
    artifact_attribute: str
    metadata: dict
    edge_policy: dict | None
    variant_id: str | None

    def build_attempt(self, compiler):
        return self.attempt_builders[compiler]()


@dataclass
class _NeighborCache:
    atomic_numbers: np.ndarray
    pbc: np.ndarray
    topology_cell: np.ndarray
    topology_reference_positions: np.ndarray
    topology_fractional_positions: np.ndarray
    topology_fractional_xyz: np.ndarray
    cell: np.ndarray
    reference_positions: np.ndarray
    receivers: np.ndarray
    sources: np.ndarray
    shifts: np.ndarray
    reference_xyz: np.ndarray
    inverse_cell: np.ndarray
    node_types: np.ndarray
    num_neigh: np.ndarray
    neigh_types: np.ndarray
    native_geometry_eligible: bool
    fractional_geometry_eligible: bool
    generation: int
    geometry_generation: int
    host_geometry_generation: int
    device_graph_generation: int = 0
    device_geometry_generation: int = 0
    num_edges: int = 0


def _to_voigt_stress(stress):
    stress = np.asarray(stress, dtype=float)
    if stress.shape == (3, 3):
        return full_3x3_to_voigt_6_stress(stress)
    if stress.shape == (6,):
        return stress
    raise ValueError("ASE stress must have shape (6,) or (3, 3).")


def _require_matching_jit_generation_version():
    from .jit import JIT_GENERATION_VERSION

    query = getattr(symmetrix, "_required_jit_generation_version", None)
    if not callable(query):
        raise RuntimeError(
            "the native Symmetrix extension predates JIT generation versioning; "
            "rebuild it with the active source tree"
        )
    required = query()
    if isinstance(required, bool) or not isinstance(required, int) or required < 1:
        raise RuntimeError(
            "the native Symmetrix extension reported an invalid required JIT "
            "generation version"
        )
    if required != JIT_GENERATION_VERSION:
        raise RuntimeError(
            "JIT generation version mismatch: native standard modules require "
            f"{required}, but Python generators provide {JIT_GENERATION_VERSION}; "
            "rebuild or reinstall Symmetrix from one source revision"
        )
    return JIT_GENERATION_VERSION


def _factorized_device_backend_plan(backend, target, contract, dtype, evaluator):
    if backend == "cuda":
        from .jit import (
            prepare_execution_cuda_jit_artifact,
            prepare_nvrtc_jit_artifact,
            select_execution_cuda_jit_backend,
            execution_cuda_jit_backend_request,
            execution_cuda_r1_edge_launch_policy,
        )
        from .jit_codegen import (
            render_jit_r1_cuda_module,
            render_jit_r1_cuda_plugin,
            jit_r1_cuda_plugin_metadata,
            factorized_gpu_codegen_identity,
        )

        normalized_target = target.get("compute_capability_code")
        if not isinstance(normalized_target, int):
            dotted = str(target.get("compute_capability", ""))
            major, separator, minor = dotted.partition(".")
            if not separator or not major.isdigit() or not minor.isdigit():
                raise RuntimeError(
                    "the native evaluator reported an invalid CUDA compute capability"
                )
            normalized_target = 10 * int(major) + int(minor)
        policy = execution_cuda_jit_backend_request()
        selected_compiler = select_execution_cuda_jit_backend(policy)
        launch_policy = execution_cuda_r1_edge_launch_policy(
            default_strategy="wave",
            default_logical_subgroup_width=32,
            default_edge_threads_per_block=32,
            default_persistent_blocks_per_compute_unit=8,
        )
        metadata = jit_r1_cuda_plugin_metadata(
            contract,
            normalized_target,
            precision=dtype,
            edge_strategy=launch_policy["strategy"],
            edge_logical_subgroup_width=launch_policy["logical_subgroup_width"],
            edge_threads_per_block=launch_policy["edge_threads_per_block"],
            persistent_blocks_per_compute_unit=launch_policy[
                "persistent_blocks_per_compute_unit"
            ],
        )
        edge_policy = {
            "strategy": metadata["edge_strategy"],
            "logical_subgroup_width": metadata["edge_logical_subgroup_width"],
            "threads_per_block": metadata["edge_threads_per_block"],
            "persistent_blocks_per_compute_unit": metadata[
                "persistent_blocks_per_compute_unit"
            ],
        }
        variant_id = (
            f"{metadata['artifact_id']}-edge-{metadata['edge_strategy']}-w"
            f"{metadata['edge_logical_subgroup_width']}-t"
            f"{metadata['edge_threads_per_block']}-b"
            f"{metadata['persistent_blocks_per_compute_unit']}"
        )

        def build_attempt(compiler):
            is_runtime = compiler == "nvrtc"
            renderer = (
                render_jit_r1_cuda_module if is_runtime else render_jit_r1_cuda_plugin
            )
            source = renderer(
                contract,
                normalized_target,
                precision=dtype,
                edge_strategy=metadata["edge_strategy"],
                edge_logical_subgroup_width=metadata["edge_logical_subgroup_width"],
                edge_threads_per_block=metadata["edge_threads_per_block"],
                persistent_blocks_per_compute_unit=metadata[
                    "persistent_blocks_per_compute_unit"
                ],
            )
            generator = (
                "symmetrix.jit.r1-cuda-module-v2"
                if is_runtime
                else "symmetrix.jit.r1-cuda-v3"
            )
            artifact_name = (
                "factorized_cuda_module" if is_runtime else "factorized_cuda_plugin"
            )
            return _JitArtifactAttempt(
                compiler=compiler,
                source=source,
                prepare=(
                    prepare_nvrtc_jit_artifact
                    if is_runtime
                    else prepare_execution_cuda_jit_artifact
                ),
                arguments={
                    "abi": {
                        "tag": metadata["abi"],
                        "version": metadata["abi_version"],
                    },
                    "build": {
                        "generator": generator,
                        "precision": dtype,
                        "contract": metadata,
                        "gpu_codegen": factorized_gpu_codegen_identity(
                            contract,
                            "cuda",
                            normalized_target,
                            precision=dtype,
                            artifact_kind="module" if is_runtime else "plugin",
                            edge_strategy=metadata["edge_strategy"],
                            edge_logical_subgroup_width=metadata[
                                "edge_logical_subgroup_width"
                            ],
                            edge_threads_per_block=metadata["edge_threads_per_block"],
                            persistent_blocks_per_compute_unit=metadata[
                                "persistent_blocks_per_compute_unit"
                            ],
                        ),
                    },
                    "compute_capability": target,
                    "artifact_name": artifact_name,
                },
            )

        return _FactorizedDeviceBackendPlan(
            backend="cuda",
            target=target,
            normalized_target=normalized_target,
            policy=policy,
            selected_compiler=selected_compiler,
            attempt_builders={
                "nvrtc": lambda: build_attempt("nvrtc"),
                "nvcc": lambda: build_attempt("nvcc"),
            },
            loader=evaluator._load_jit_cuda_plugin,
            ready_attribute="jit_cuda_plugin_ready",
            artifact_attribute="jit_cuda_plugin_artifact_id",
            metadata=metadata,
            edge_policy=edge_policy,
            variant_id=variant_id,
        )

    if backend == "hip":
        from .jit import (
            normalize_execution_hip_target_identity,
            prepare_execution_hip_jit_artifact,
            prepare_hiprtc_jit_artifact,
            select_execution_hip_jit_backend,
            execution_hip_jit_backend_request,
            execution_hip_r1_edge_launch_policy,
        )
        from .jit_codegen import (
            render_jit_r1_hip_module,
            render_jit_r1_hip_plugin,
            factorized_gpu_codegen_identity,
            factorized_hip_plugin_metadata,
        )

        policy = execution_hip_jit_backend_request()
        selected_compiler = select_execution_hip_jit_backend(policy)
        normalized_target = normalize_execution_hip_target_identity(target)
        launch_policy = execution_hip_r1_edge_launch_policy(
            channels=contract.get("channels")
        )
        metadata = factorized_hip_plugin_metadata(
            contract,
            target,
            precision=dtype,
            edge_strategy=launch_policy["strategy"],
            edge_threads_per_block=launch_policy["edge_threads_per_block"],
            persistent_blocks_per_compute_unit=launch_policy[
                "persistent_blocks_per_compute_unit"
            ],
        )
        edge_policy = {
            "strategy": metadata["edge_strategy"],
            "logical_subgroup_width": metadata["edge_logical_subgroup_width"],
            "threads_per_block": metadata["edge_threads_per_block"],
            "persistent_blocks_per_compute_unit": metadata[
                "persistent_blocks_per_compute_unit"
            ],
        }
        variant_id = (
            f"{metadata['artifact_id']}-edge-{metadata['edge_strategy']}-w"
            f"{metadata['edge_logical_subgroup_width']}-t"
            f"{metadata['edge_threads_per_block']}-b"
            f"{metadata['persistent_blocks_per_compute_unit']}"
        )

        def build_attempt(compiler):
            is_runtime = compiler == "hiprtc"
            renderer = (
                render_jit_r1_hip_module if is_runtime else render_jit_r1_hip_plugin
            )
            source = renderer(
                contract,
                target,
                precision=dtype,
                edge_strategy=metadata["edge_strategy"],
                edge_threads_per_block=metadata["edge_threads_per_block"],
                persistent_blocks_per_compute_unit=metadata[
                    "persistent_blocks_per_compute_unit"
                ],
            )
            generator = (
                "symmetrix.jit.r1-hip-module-v2"
                if is_runtime
                else "symmetrix.jit.r1-hip-v2"
            )
            artifact_name = (
                "factorized_hip_module" if is_runtime else "factorized_hip_plugin"
            )
            return _JitArtifactAttempt(
                compiler=compiler,
                source=source,
                prepare=(
                    prepare_hiprtc_jit_artifact
                    if is_runtime
                    else prepare_execution_hip_jit_artifact
                ),
                arguments={
                    "abi": {
                        "tag": metadata["abi"],
                        "version": metadata["abi_version"],
                    },
                    "build": {
                        "generator": generator,
                        "precision": dtype,
                        "contract": metadata,
                        "gpu_codegen": factorized_gpu_codegen_identity(
                            contract,
                            "hip",
                            target,
                            precision=dtype,
                            artifact_kind="module" if is_runtime else "plugin",
                            edge_strategy=metadata["edge_strategy"],
                            edge_threads_per_block=metadata["edge_threads_per_block"],
                            persistent_blocks_per_compute_unit=metadata[
                                "persistent_blocks_per_compute_unit"
                            ],
                        ),
                    },
                    "target": target,
                    "artifact_name": artifact_name,
                },
            )

        return _FactorizedDeviceBackendPlan(
            backend="hip",
            target=target,
            normalized_target=normalized_target,
            policy=policy,
            selected_compiler=selected_compiler,
            attempt_builders={
                "hiprtc": lambda: build_attempt("hiprtc"),
                "hipcc": lambda: build_attempt("hipcc"),
            },
            loader=evaluator._load_jit_hip_plugin,
            ready_attribute="jit_hip_plugin_ready",
            artifact_attribute="jit_hip_plugin_artifact_id",
            metadata=metadata,
            edge_policy=edge_policy,
            variant_id=variant_id,
        )

    raise ValueError(f"unsupported Execution R1 device backend: {backend}")


class Symmetrix(Calculator):
    """ASE calculator for equivariant graph neural-network potentials.

    Parameters
    ----------
    model_file : str or pathlib.Path
        Path to a compact Symmetrix JSON model or a supported MACE
        PyTorch checkpoint. Ordinary-MACE checkpoints are converted on load
        and therefore require ``mace-torch``; compact JSON models do not.
        MACEField checkpoints must be converted to compact JSON before use.
    dtype : {"float32", "float64"}, default="float32"
        Floating-point precision used by the evaluator. Select ``float64``
        explicitly for high-precision calculations. The non-Kokkos serial
        ``MACE_Nonlinear`` evaluator currently requires ``float64``.
    use_kokkos : bool, default=True
        Use the compiled Kokkos evaluator. ``False`` selects the non-Kokkos
        serial CPU evaluator.
    streamed_edges : {"direct", "non-compiled", "materialized"}, default="direct"
        Evaluation algorithm for compact two-layer MACE models. ``direct``
        requires a model-specific RTC artifact and does not fall back.
        ``non-compiled`` is an explicit compiler-free fallback with no
        performance guarantee. ``materialized`` is a frozen legacy mode.
        The old ``generic`` spelling remains accepted as a deprecated alias.
    execution_profile : {"capacity", "speed"}, default="capacity"
        Resource objective for direct Kokkos execution. ``capacity`` chooses
        the fastest qualified direct policy estimated to fit the device; when
        all estimates exceed the advisory budget it attempts the smallest
        qualified policy and lets allocation determine feasibility. ``speed``
        always uses the retained throughput policy. The current native report
        is available as :attr:`execution_plan` after graph preparation.
    allow_fixed_workspace : bool, default=False
        Permit capacity planning to select bounded tiled workspace execution.
        This can increase the maximum graph size for qualified direct Kokkos
        models, but must be enabled explicitly. It does not force a tiled plan.
    dispersion : bool, default=False
        Add a D3 dispersion correction using the optional ``torch-dftd``
        package. The correction is evaluated alongside Symmetrix and added to
        energy, forces, and stress.
    dispersion_damping : str, default="bj"
        D3 damping function (``zero``, ``bj``, ``zerom``, or ``bjm``).
    dispersion_xc : str, default="pbe"
        Exchange-correlation functional parameterization for D3.
    dispersion_cutoff : float, optional
        D3 cutoff in Bohr. Defaults to 40 Bohr, matching MACE's calculator.
    dispersion_device : str, optional
        Torch device for D3. Defaults to CUDA for a CUDA backend, otherwise CPU.
    low_memory : bool, optional
        Deprecated compatibility spelling. ``True`` maps to
        ``execution_profile="capacity"`` and ``False`` maps to ``"speed"``.
        New callers should use ``execution_profile``.
    _debug_execution_plan : str, optional
        Private development pin for a validated MH-0 internal plan. It is
        recorded in :attr:`execution_plan` and never bypasses artifact or
        low-memory eligibility checks.
        The capacity policy recomputes M1/readout state, reuses dead MH-0
        forward buffers for adjoints, uses compact float32 geometry where
        eligible, and reconstructs spherical-harmonic gradients on admitted
        CUDA, HIP, and generated-host paths. Phi1 remains retained. Float64
        retains Cartesian float64 geometry.
        For Float32 CUDA MACE-MH-1, it selects the
        ``retain-interaction-v1`` node-state policy and the bounded
        ``capacity-v1`` node arena. That policy retains the product reverse
        input, replays only pre-gate state, and uses the retained CUDA node
        schedule. HIP keeps its separately qualified conservative schedule.
        Ordinary MACE and MACEField analytical polarizability and Born-charge
        evaluation reconstruct overwritten forward state before response.
    jit : optional
        Deprecated and ignored. Direct execution requires a model-specific JIT
        specialization. Other execution algorithms do not use JIT.
    m1_polynomial_policy : {"automatic", "recompute", "retained"}, default="automatic"
        M1 polynomial workspace policy. ``automatic`` selects recomputation on
        capable Kokkos Serial, OpenMP, CUDA, or HIP evaluators when one of the
        32-, 16-, or 8-channel tiles fits backend team scratch, and otherwise
        retains polynomial values and adjoints with a diagnostic. ``retained``
        is the explicit rollback; ``recompute`` requests recomputation and
        fails with the required and available scratch bytes when unsupported.
        This policy does not alter standard-versus-runtime M0 selection.
    kernel_launch_policy : {"automatic", "static"}, default="automatic"
        Persistent GPU launch-profile policy. ``automatic`` consumes an
        already validated matching calibration record when one exists and
        otherwise immediately uses the resource-selected module or plugin default.
        It never benchmarks. ``static`` ignores calibration records and is the
        exact rollback to the qualified module or plugin descriptor default.
    execution_mh1_scratch_budget_bytes : int or None, default=None
        Maximum compact-phi workspace admitted by the generated MH-1 planner.
        ``None`` preserves retain-all execution. Smaller budgets share a
        phase-local phi buffer and recompute omitted conditioning before reverse.
        Mandatory phase scratch may exceed a budget below the reported minimum.
    execution_mh1_node_state_policy : {"full-retention-v1", "recompute-v1", "reuse-adjoints-v1", "retain-interaction-v1"}, default="full-retention-v1"
        Node-state storage policy for generated MH-1 CUDA and HIP modules.
        ``recompute-v1`` reduces retained node workspace by recomputing the
        required forward state during reverse execution.
        ``reuse-adjoints-v1`` retains the reverse-required forward state but
        overwrites dead forward-message rows with their adjoints after
        evaluating the density derivative. It avoids recomputation while using
        less memory than full retention.
        ``retain-interaction-v1`` retains only the product-reverse input,
        reconstructs pre-gate state exactly from retained upstream state, and
        aliases dead forward-message rows with their adjoints. It is selected
        by ``low_memory=True`` for Float32 CUDA MACE-MH-1.
    execution_mh1_node_arena_policy : {"throughput-v1", "capacity-v1"}, default="throughput-v1"
        Transient node-arena policy for generated MH-1 execution.
        ``throughput-v1`` uses a wider CUDA arena to reduce repeated node-stage
        launches. ``capacity-v1`` retains the bounded 256-row arena used for
        near-capacity systems. This policy is independent of retained node state.
    execution_mh0_state_policy : {"full-retention-v1", "reuse-adjoints-v1"}, default="full-retention-v1"
        State-storage policy for ordinary direct MH-0 inference.
        ``reuse-adjoints-v1`` destructively reuses dead forward tensors for
        their adjoints and requires M1 polynomial recomputation. It is an
        expert policy and is incompatible with execution observers and
        parameter gradients. Prefer ``low_memory=True`` for the supported
        policy bundle.
    edge_geometry_policy : {"cartesian-f64-v1", "unit-f32-radius-f64-v1"}, default="cartesian-f64-v1"
        Active edge-geometry representation for ordinary factorized inference.
        ``unit-f32-radius-f64-v1`` retains float32 unit directions and float64
        radii. It is an opt-in standard MH-0 capacity mode.
    neighbor_skin : float, default=0.5
        Verlet-list skin in Angstrom. Candidate neighbors are built at the
        model cutoff plus this skin and reused until any atom has moved by
        half the skin. With ``streamed_edges="generic"``, current exact-cutoff
        members are compacted from those candidates in native Kokkos code.
        With ``streamed_edges="direct"`` or ``"receiver_factorized"``, inactive candidates are retained at
        the exact compact-radial cutoff, where their radial contribution is
        zero, so the prepared schedule remains stable. Set to zero to rebuild
        the exact neighbor list every call.
    head : str or None, default=None
        Prediction head selected from a multi-head Symmetrix JSON or MACE
        checkpoint. ``None`` uses the JSON's declared default head.

    Attributes
    ----------
    jit_status : str
        Specialization outcome: ``"built"``, ``"cached"``, ``"disabled"``,
        or ``"not_applicable"``. Specialization failures raise during
        construction.
    jit_reason : str or None
        Explanation when specialization is disabled or not applicable.
    jit_artifact_id : str or None
        Contract-derived identity of the active JIT artifact.
    jit_variant_id : str or None
        Model and CUDA execution-policy identity of an active JIT MH-1
        CUDA variant.
    jit_forward_policy, jit_source_policy : dict or None
        Canonical generated MH-1 CUDA partition plans. These remain ``None``
        unless the corresponding plugin compiled and loaded successfully.
    jit_node_state_policy : str or None
        Active generated MH-1 node-state policy. This remains ``None`` unless
        an MH-1 CUDA or HIP module compiled and loaded successfully.
    low_memory_policy : str
        Native whole-bundle selection: ``"disabled"``, ``"pending"``,
        ``"speed"``, ``"capacity-retained"``, or ``"capacity-y-only"``.
        A true ``low_memory`` request may therefore report ``"speed"`` when
        the normal fast path fits current device memory.
    head : str or None
        Selected prediction-head name, or ``None`` for an unnamed legacy JSON.
    available_heads : list[str]
        Prediction heads stored in the JSON.

    Notes
    -----
    Generated Execution specialization supports ordinary MACE and compatible
    MACE-MH-1-family models on Kokkos Serial, OpenMP, and CUDA builds. Dynamic
    compilation requires a POSIX platform; host plugins use a C++20 compiler
    selected by ``CXX`` (or ``c++``). CUDA R1 specialization uses NVRTC by
    default and accepts the runtime from the ``cuda13`` package extra;
    ``SYMMETRIX_JIT_CUDA_JIT_BACKEND=nvcc`` selects the retained toolkit
    compiler. CUDA R1 and nonlinear MH1 automatic specialization both use
    NVRTC. Runtime NVCC selection remains temporarily available as a deprecated
    explicit compatibility path.
    The cache can be relocated with
    ``SYMMETRIX_JIT_CACHE``. The upstream Execution package is neither
    imported nor required.
    """

    implemented_properties = list(_FIELD_ADDITIVE_PROPERTIES)
    _macefield_response_properties = list(_FIELD_RESPONSE_PROPERTIES)
    _macefield_eps0 = 8.8541878128e-12 / 1.602176634e-19 / 1e10

    def __init__(
        self,
        model_file,
        dtype="float32",
        use_kokkos=True,
        streamed_edges="direct",
        execution_profile="capacity",
        allow_fixed_workspace=False,
        low_memory=_LOW_MEMORY_UNSET,
        _debug_execution_plan=None,
        jit=None,
        kernel_launch_policy="automatic",
        m1_polynomial_policy="automatic",
        execution_mh1_scratch_budget_bytes=None,
        execution_mh1_node_state_policy="full-retention-v1",
        execution_mh1_node_arena_policy="throughput-v1",
        execution_mh0_state_policy="full-retention-v1",
        edge_geometry_policy="cartesian-f64-v1",
        neighbor_skin=0.5,
        head=None,
        dispersion=False,
        dispersion_damping="bj",
        dispersion_xc="pbe",
        dispersion_cutoff=None,
        dispersion_device=None,
        **kwargs,
    ):
        self.dispersion = bool(dispersion)
        self._dispersion_calculator = None
        self._dispersion_properties = ()
        Calculator.__init__(self, **kwargs)
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Supported dtypes are 'float64' and 'float32'."
            )
        requested_streamed_edges = streamed_edges
        if requested_streamed_edges in ("generic", "all_interactions"):
            warnings.warn(
                "streamed_edges='generic' is deprecated; use "
                "streamed_edges='non-compiled' for the uncompiled fallback. "
                "This path has no performance guarantee.",
                DeprecationWarning,
                stacklevel=2,
            )
        canonical_streamed_edges = _canonical_streamed_edges_mode(streamed_edges)
        legacy_streamed_alias = requested_streamed_edges in _STREAMED_EDGE_ALIASES
        if canonical_streamed_edges != "auto" and (
            canonical_streamed_edges not in _STREAMED_EDGE_CANONICAL_MODES
        ):
            raise ValueError(
                "streamed_edges must be one of "
                "'auto', 'materialized', 'generic', or 'direct' "
                "(compatibility aliases: "
                "'all_interactions', 'factorized', 'direct_streamed')."
            )
        if execution_profile not in _EXECUTION_PROFILES:
            raise ValueError("execution_profile must be 'capacity' or 'speed'.")
        if not isinstance(allow_fixed_workspace, bool):
            raise ValueError("allow_fixed_workspace must be a bool.")
        if low_memory is not _LOW_MEMORY_UNSET and not isinstance(low_memory, bool):
            raise ValueError("low_memory must be a bool.")
        if _debug_execution_plan is not None:
            if low_memory is not _LOW_MEMORY_UNSET:
                raise ValueError(
                    "_debug_execution_plan cannot be combined with the legacy "
                    "low_memory argument."
                )
            if _debug_execution_plan not in _DEBUG_EXECUTION_PLANS:
                raise ValueError(
                    "_debug_execution_plan must be one of "
                    f"{sorted(_DEBUG_EXECUTION_PLANS)} or None."
                )
            if canonical_streamed_edges != "direct":
                raise ValueError(
                    "_debug_execution_plan requires streamed_edges='direct'."
                )
            required_profile = (
                "speed" if _debug_execution_plan == "mh0-direct-speed" else "capacity"
            )
            if execution_profile != required_profile:
                raise ValueError(
                    f"{_debug_execution_plan} requires execution_profile={required_profile!r}."
                )
        legacy_low_memory_requested = low_memory is True
        # Compatibility mapping for callers of the former public switch. New
        # code should express intent through execution_profile instead.
        if low_memory is not _LOW_MEMORY_UNSET:
            execution_profile = "capacity" if low_memory else "speed"
        elif legacy_streamed_alias and execution_profile == "capacity":
            # Aliases retain the old throughput-oriented behavior while they
            # remain available as compatibility spellings. The literal direct
            # default is the new capacity-oriented public policy.
            execution_profile = "speed"
        if allow_fixed_workspace:
            if not use_kokkos:
                raise ValueError("allow_fixed_workspace=True requires use_kokkos=True.")
            if canonical_streamed_edges != "direct":
                raise ValueError(
                    "allow_fixed_workspace=True requires streamed_edges='direct'."
                )
            if execution_profile != "capacity":
                raise ValueError(
                    "allow_fixed_workspace=True requires execution_profile='capacity'."
                )
        if legacy_low_memory_requested:
            if not use_kokkos:
                raise ValueError("low_memory=True requires use_kokkos=True.")
            if canonical_streamed_edges != "direct":
                raise ValueError("low_memory=True requires streamed_edges='direct'.")
        if execution_mh1_scratch_budget_bytes is not None and (
            isinstance(execution_mh1_scratch_budget_bytes, bool)
            or not isinstance(execution_mh1_scratch_budget_bytes, int)
            or execution_mh1_scratch_budget_bytes < 0
        ):
            raise ValueError(
                "execution_mh1_scratch_budget_bytes must be a non-negative integer or None."
            )
        self.execution_profile = execution_profile
        self.allow_fixed_workspace = allow_fixed_workspace
        self._debug_execution_plan = _debug_execution_plan
        self.low_memory_request = (
            execution_profile == "capacity" and canonical_streamed_edges == "direct"
        )
        self.low_memory = False
        self.low_memory_policy = "disabled"
        self.low_memory_selection_reason = "low_memory=False"
        self.execution_mh1_node_state_policy = (
            self._normalize_execution_mh1_node_state_policy(
                execution_mh1_node_state_policy
            )
        )
        self.execution_mh1_node_arena_policy = (
            self._normalize_execution_mh1_node_arena_policy(
                execution_mh1_node_arena_policy
            )
        )
        self.execution_mh0_state_policy = self._normalize_execution_mh0_state_policy(
            execution_mh0_state_policy
        )
        self.edge_geometry_policy = self._normalize_edge_geometry_policy(
            edge_geometry_policy
        )
        normalized_m1_polynomial_policy = self._normalize_m1_polynomial_policy(
            m1_polynomial_policy
        )
        if (
            isinstance(neighbor_skin, bool)
            or not np.isscalar(neighbor_skin)
            or not np.isfinite(neighbor_skin)
            or neighbor_skin < 0.0
        ):
            raise ValueError("neighbor_skin must be a finite non-negative number.")
        self.neighbor_skin = float(neighbor_skin)
        self._neighbor_cache = None
        self.neighbor_cache_build_count = 0
        self.neighbor_cache_reuse_count = 0
        self.neighbor_cache_geometry_update_count = 0
        self.neighbor_cache_host_geometry_materialization_count = 0
        self.neighbor_graph_backend = "unresolved"
        self._execution_native_geometry_identity = None
        self._execution_native_cell_identity = None
        self._all_interactions_native_geometry_identity = None
        self._all_interactions_graph_identity = None
        self._all_interactions_graph_generation = 0
        self._deprecate_jit_argument(jit)
        self.streamed_edges_requested = requested_streamed_edges
        self.streamed_edges_alias = (
            requested_streamed_edges if legacy_streamed_alias else None
        )
        self.streamed_edges_resolution_reason = None
        self.jit_policy = "unresolved"
        self.kernel_launch_policy = self._normalize_kernel_launch_policy(
            kernel_launch_policy
        )
        self.m1_polynomial_policy_request = normalized_m1_polynomial_policy
        self.m1_polynomial_policy = "retained"
        self.m1_polynomial_policy_reason = None
        self.mh0_state_policy = "full-retention-v1"
        self.mh0_state_policy_reason = None
        self.kernel_launch_tuning_status = (
            "static" if self.kernel_launch_policy == "static" else "unresolved"
        )
        self.kernel_launch_tuning_cache_key = None
        self.kernel_launch_tuning_reason = None
        self._kernel_launch_tuning_applied_key = None
        self._kernel_launch_dtype = dtype
        self._dtype = dtype
        self.jit_status = "unresolved"
        self.jit_cache_key = None
        self.jit_artifact_path = None
        self.jit_artifact_id = None
        self.jit_compiler_backend = None
        self.jit_compiler_request = None
        self.jit_diagnostics = ()
        self.jit_reason = None
        self.jit_forward_policy = None
        self.jit_source_policy = None
        self.jit_edge_policy = None
        self.jit_node_state_policy = None
        self.jit_variant_id = None
        self.jit_operator_modules = {}
        self._macefield_electric_field = None
        self._macefield_response_graph_generation = 0
        self._electric_field = kwargs.get("electric_field", None)
        json_metadata = self._json_metadata(model_file)
        self._model_has_field_coupling = (
            bool(json_metadata.get("has_field_coupling", False))
            if json_metadata is not None
            else False
        )
        self._model_single_layer_readout = (
            bool(json_metadata.get("single_layer_readout", False))
            if json_metadata is not None
            else False
        )

        if use_kokkos and not hasattr(symmetrix, "MACEKokkos"):
            raise RuntimeError("Symmetrix was built without Kokkos support.")
        self.use_kokkos = use_kokkos
        json_model_type = (
            json_metadata.get("model_type", "MACE")
            if json_metadata is not None
            else None
        )
        self._native_model_type = (
            json_model_type if json_model_type is not None else "MACE"
        )
        MACE = self._native_evaluator_class(
            json_model_type if json_model_type is not None else "MACE",
            dtype,
            use_kokkos,
        )
        jit_model_data = None
        try:
            self.evaluator = MACE(str(model_file), "" if head is None else head)
        except RuntimeError as error:
            if not self._is_native_json_parse_error(error):
                raise
            if json_metadata is not None or str(model_file).lower().endswith(".json"):
                raise
            self._raise_if_macefield_checkpoint(model_file)

            # import this here so that torch/mace support isn't needed if file is already symmetrix json
            from .extract_mace_data import extract_mace_data

            kwargs_extract = {
                k: v
                for k, v in kwargs.items()
                if k in ["species", "num_spline_points", "radial_format"]
            }
            if head is not None:
                kwargs_extract["head"] = head
            logging.info(
                f"Converting model from pytorch model to symmetrix dict with {kwargs_extract}"
            )
            data = extract_mace_data(model_file, **kwargs_extract)
            jit_model_data = data
            self._native_model_type = data.get("model_type", "MACE")
            self._model_single_layer_readout = bool(
                data.get("single_layer_readout", False)
            )
            MACE = self._native_evaluator_class(
                data.get("model_type", "MACE"), dtype, use_kokkos
            )
            with NamedTemporaryFile("w") as fout:
                logging.debug(
                    "Loading converted model through temporary JSON file %s",
                    fout.name,
                )
                fout.write(json.dumps(data))
                fout.flush()
                self.evaluator = MACE(fout.name, "" if head is None else head)

        self._model_single_layer_readout = bool(
            getattr(
                self.evaluator,
                "single_layer_readout",
                self._model_single_layer_readout,
            )
        )
        selected_head = getattr(self.evaluator, "selected_head", "")
        self.head = selected_head or None
        self.available_heads = list(getattr(self.evaluator, "available_heads", []))

        if not use_kokkos and canonical_streamed_edges == "direct":
            warnings.warn(
                "use_kokkos=False selects serial generic execution; direct "
                "streamed execution is unavailable and performance will be "
                "substantially lower.",
                RuntimeWarning,
                stacklevel=2,
            )
            canonical_streamed_edges = "generic"
            self.streamed_edges_resolution_reason = (
                "use_kokkos=False selected serial generic execution"
            )
            self.execution_profile = "serial-generic"
            self.low_memory_request = False

        if (
            canonical_streamed_edges != "materialized"
            and hasattr(self.evaluator, "set_streamed_edges")
            and not bool(getattr(self.evaluator, "supports_streamed_edges", True))
        ):
            warnings.warn(
                f"streamed_edges='{canonical_streamed_edges}' is unavailable "
                "for legacy format-v1 pair-spline models; using "
                "streamed_edges='materialized'. Re-export with "
                "radial_format='compact' to enable streamed execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            canonical_streamed_edges = "materialized"
            self.streamed_edges_resolution_reason = (
                "legacy format-v1 model selected materialized execution"
            )
            self.low_memory_request = False

        if (
            self.low_memory_request
            and not legacy_low_memory_requested
            and self._native_model_type == "MACE_Nonlinear"
            and dtype != "float32"
        ):
            warnings.warn(
                "The implied low-memory execution plan for MACE_Nonlinear "
                "models requires dtype 'float32'; keeping the default "
                "execution policies for this evaluator. Pass dtype='float32' "
                "to enable the low-memory direct plan.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.low_memory_request = False
            self.low_memory_selection_reason = (
                "implied low-memory plan unavailable for MACE_Nonlinear "
                f"{dtype}; using default execution policies"
            )

        if self.low_memory_request and self._native_model_type not in (
            "MACE",
            "MACEField",
            "MACE_Nonlinear",
        ):
            raise ValueError(
                "low_memory=True currently supports ordinary MACE, MACEField, "
                "and MACE_Nonlinear models."
            )
        if self.low_memory_request:
            if self._native_model_type == "MACE_Nonlinear":
                if dtype != "float32":
                    raise ValueError(
                        "low_memory=True for MACE_Nonlinear requires dtype 'float32'."
                    )
                self.execution_mh1_node_state_policy = "retain-interaction-v1"
                self.execution_mh1_node_arena_policy = "capacity-v1"
            else:
                self.execution_mh0_state_policy = "reuse-adjoints-v1"
                self.edge_geometry_policy = (
                    "unit-f32-radius-f64-v1"
                    if dtype == "float32"
                    else "cartesian-f64-v1"
                )

        if hasattr(self.evaluator, "set_streamed_edges"):
            if canonical_streamed_edges == "auto":
                supports_direct = bool(use_kokkos) and bool(
                    getattr(self.evaluator, "supports_factorized", False)
                )
                if supports_direct:
                    canonical_streamed_edges = "direct"
                    self.streamed_edges_resolution_reason = (
                        "auto selected direct because the Kokkos evaluator "
                        "admits generated execution"
                    )
                elif bool(getattr(self.evaluator, "supports_streamed_edges", False)):
                    canonical_streamed_edges = "generic"
                    self.streamed_edges_resolution_reason = "auto selected generic because generated execution is unavailable"
                else:
                    canonical_streamed_edges = "materialized"
                    self.streamed_edges_resolution_reason = (
                        "auto selected materialized for a legacy model"
                    )
            self.evaluator.set_streamed_edges(canonical_streamed_edges)
            selected_streamed_edges = self.evaluator.streamed_edges_mode
        elif canonical_streamed_edges not in ("auto", "materialized"):
            raise ValueError(
                "streamed_edges is only supported by compatible compact MACE, "
                "MACEField, or MACE_Nonlinear evaluators."
            )
        else:
            selected_streamed_edges = "materialized"
        self.streamed_edges = selected_streamed_edges
        self.execution_algorithm = selected_streamed_edges
        fixed_workspace_setter = getattr(
            self.evaluator, "_set_allow_fixed_workspace", None
        )
        if self.allow_fixed_workspace:
            if not callable(fixed_workspace_setter):
                raise RuntimeError(
                    "allow_fixed_workspace=True requires a compatible Kokkos evaluator."
                )
            fixed_workspace_setter(True)
        elif callable(fixed_workspace_setter):
            fixed_workspace_setter(False)
        native_plan_request = getattr(
            self.evaluator, "_set_execution_plan_request", None
        )
        native_plan_resolve = getattr(self.evaluator, "_resolve_execution_plan", None)
        self._native_execution_plan_available = callable(
            native_plan_request
        ) and callable(native_plan_resolve)
        if (
            self._native_execution_plan_available
            and self.execution_profile in _EXECUTION_PROFILES
        ):
            native_plan_request(
                self.streamed_edges,
                self.execution_profile,
                ""
                if self._debug_execution_plan is None
                else self._debug_execution_plan,
            )
        self._configure_m1_polynomial_policy()
        launch_policy_setter = getattr(
            self.evaluator, "_set_kernel_launch_policy", None
        )
        if launch_policy_setter is not None:
            launch_policy_setter(self.kernel_launch_policy)
        self._configure_execution_mh1_node_arena_policy()
        self._configure_jit(
            model_file=model_file,
            model_data=jit_model_data,
            dtype=dtype,
            use_kokkos=use_kokkos,
            streamed_edges=self.streamed_edges,
        )
        if (
            self._native_execution_plan_available
            and self.execution_profile in _EXECUTION_PROFILES
        ):
            native_plan_resolve()
            self._sync_low_memory_policy_state()
        elif self.low_memory_request:
            self._configure_low_memory()
        else:
            self._configure_execution_mh0_state_policy()
            self._configure_edge_geometry_policy()
            self.low_memory = bool(getattr(self.evaluator, "low_memory", False))
            self.low_memory_policy = str(
                getattr(self.evaluator, "low_memory_policy", "disabled")
            )
            self.low_memory_selection_reason = str(
                getattr(
                    self.evaluator,
                    "low_memory_selection_reason",
                    "low_memory=False",
                )
            )
        if execution_mh1_scratch_budget_bytes is not None:
            setter = getattr(
                self.evaluator, "_set_execution_mh1_scratch_budget_bytes", None
            )
            if setter is None:
                raise ValueError(
                    "execution_mh1_scratch_budget_bytes requires a Kokkos "
                    "MACE-MH-1 evaluator."
                )
            setter(execution_mh1_scratch_budget_bytes)
        self.cutoff = self.evaluator.r_cut
        self.implemented_properties = list(type(self).implemented_properties)
        if self._has_native_field_coupling():
            self.implemented_properties.append("node_energy")
            self.implemented_properties.extend(self._macefield_response_properties)
        if self.dispersion:
            self._configure_dispersion(
                damping=dispersion_damping,
                xc=dispersion_xc,
                cutoff=dispersion_cutoff,
                device=dispersion_device,
            )

    def _configure_dispersion(self, *, damping, xc, cutoff, device):
        """Attach the optional torch-dftd D3 calculator."""
        try:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        except ImportError as error:
            raise RuntimeError(
                "dispersion=True requires torch-dftd; install it with "
                "`uv pip install torch-dftd`."
            ) from error
        if device is None:
            device = "cuda" if self._native_backend_is_cuda() else "cpu"
        if cutoff is None:
            from ase import units

            cutoff = 40.0 * units.Bohr
        try:
            import torch

            dtype = torch.float32 if self._dtype == "float32" else torch.float64
        except ImportError as error:
            raise RuntimeError("dispersion=True requires torch.") from error
        self._dispersion_calculator = TorchDFTD3Calculator(
            device=device, damping=damping, dtype=dtype, xc=xc, cutoff=cutoff
        )
        self._dispersion_properties = ("energy", "forces", "stress")

    def _native_backend_is_cuda(self):
        return "cuda" in getattr(symmetrix, "__name__", "").lower()

    def _add_dispersion_results(self, properties):
        if getattr(self, "_dispersion_calculator", None) is None:
            return
        requested = [
            prop for prop in ("energy", "forces", "stress") if prop in properties
        ]
        self._dispersion_calculator.calculate(
            self.atoms, properties=requested, system_changes=[]
        )
        for prop in requested:
            value = self._dispersion_calculator.results.get(prop)
            if value is None:
                continue
            if prop == "energy":
                self.results[prop] += float(value)
                if "free_energy" in self.results:
                    self.results["free_energy"] += float(value)
            else:
                self.results[prop] = self.results[prop] + np.asarray(value)

    @staticmethod
    def _deprecate_jit_argument(policy):
        if policy is not None:
            warnings.warn(
                "The jit parameter is deprecated and ignored. Direct execution "
                "and receiver-factorized execution require RTC specialization; "
                "select streamed_edges='non-compiled' for compiler-free execution.",
                FutureWarning,
                stacklevel=3,
            )

    @staticmethod
    def _jit_policy_from_environment():
        selected = os.environ.get(
            _JIT_POLICY_ENVIRONMENT_VARIABLE, _JIT_REQUIRED_POLICY
        )
        if selected not in (_JIT_REQUIRED_POLICY, _JIT_DEBUG_POLICY):
            raise ValueError(
                f"{_JIT_POLICY_ENVIRONMENT_VARIABLE} must be "
                f"'{_JIT_REQUIRED_POLICY}' or '{_JIT_DEBUG_POLICY}'."
            )
        return selected

    @staticmethod
    def _normalize_kernel_launch_policy(policy):
        if policy not in _KERNEL_LAUNCH_POLICIES:
            choices = ", ".join(sorted(_KERNEL_LAUNCH_POLICIES))
            raise ValueError(f"kernel_launch_policy must be one of {choices}.")
        return policy

    @staticmethod
    def _normalize_execution_mh0_state_policy(policy):
        if not isinstance(policy, str) or policy not in _EXECUTION_MH0_STATE_POLICIES:
            choices = ", ".join(
                repr(value) for value in sorted(_EXECUTION_MH0_STATE_POLICIES)
            )
            raise ValueError(f"execution_mh0_state_policy must be one of {choices}.")
        return policy

    @staticmethod
    def _normalize_edge_geometry_policy(policy):
        if not isinstance(policy, str) or policy not in _EDGE_GEOMETRY_POLICIES:
            choices = ", ".join(
                repr(value) for value in sorted(_EDGE_GEOMETRY_POLICIES)
            )
            raise ValueError(f"edge_geometry_policy must be one of {choices}.")
        return policy

    @staticmethod
    def _normalize_execution_mh1_node_state_policy(policy):
        if (
            not isinstance(policy, str)
            or policy not in _EXECUTION_MH1_NODE_STATE_POLICIES
        ):
            choices = ", ".join(
                repr(value) for value in sorted(_EXECUTION_MH1_NODE_STATE_POLICIES)
            )
            raise ValueError(
                f"execution_mh1_node_state_policy must be one of {choices}."
            )
        return policy

    @staticmethod
    def _normalize_execution_mh1_node_arena_policy(policy):
        if (
            not isinstance(policy, str)
            or policy not in _EXECUTION_MH1_NODE_ARENA_POLICIES
        ):
            choices = ", ".join(
                repr(value) for value in sorted(_EXECUTION_MH1_NODE_ARENA_POLICIES)
            )
            raise ValueError(
                f"execution_mh1_node_arena_policy must be one of {choices}."
            )
        return policy

    @staticmethod
    def _normalize_m1_polynomial_policy(policy):
        if policy not in _M1_POLYNOMIAL_POLICIES:
            choices = ", ".join(
                repr(value) for value in sorted(_M1_POLYNOMIAL_POLICIES)
            )
            raise ValueError(f"m1_polynomial_policy must be one of {choices}.")
        return policy

    def _configure_m1_polynomial_policy(self):
        setter = getattr(self.evaluator, "_set_m1_polynomial_policy", None)
        request = self.m1_polynomial_policy_request
        if setter is None:
            if request in ("automatic", "retained"):
                self.m1_polynomial_policy_reason = (
                    "the active evaluator does not expose M1 recomputation"
                )
                return
            raise ValueError(
                "m1_polynomial_policy='recompute' requires a compatible "
                "Kokkos MACE evaluator."
            )
        setter(request)
        self.m1_polynomial_policy = str(
            getattr(self.evaluator, "m1_polynomial_policy", request)
        )
        reason = getattr(self.evaluator, "m1_recompute_fallback_reason", "")
        self.m1_polynomial_policy_reason = str(reason) or None

    def _configure_execution_mh0_state_policy(self):
        setter = getattr(self.evaluator, "_set_mh0_state_policy", None)
        request = self.execution_mh0_state_policy
        if setter is None:
            if request == "full-retention-v1":
                self.mh0_state_policy_reason = (
                    "the active evaluator does not expose MH-0 state reuse"
                )
                return
            raise ValueError(
                "execution_mh0_state_policy='reuse-adjoints-v1' requires a "
                "compatible Kokkos MACE evaluator."
            )
        setter(request)
        self.mh0_state_policy = str(
            getattr(self.evaluator, "mh0_state_policy", request)
        )
        reason = getattr(self.evaluator, "mh0_state_policy_fallback_reason", "")
        self.mh0_state_policy_reason = str(reason) or None

    def _configure_edge_geometry_policy(self):
        setter = getattr(self.evaluator, "_set_edge_geometry_policy", None)
        if setter is None:
            if self.edge_geometry_policy == "cartesian-f64-v1":
                return
            raise ValueError(
                "edge_geometry_policy='unit-f32-radius-f64-v1' requires a "
                "compatible Kokkos MACE evaluator."
            )
        setter(self.edge_geometry_policy)
        self.edge_geometry_policy = str(
            getattr(self.evaluator, "edge_geometry_policy", self.edge_geometry_policy)
        )

    def _configure_execution_mh1_node_arena_policy(self):
        setter = getattr(self.evaluator, "_set_execution_mh1_node_arena_policy", None)
        if setter is None:
            if self.execution_mh1_node_arena_policy == "throughput-v1":
                return
            raise ValueError(
                "execution_mh1_node_arena_policy='capacity-v1' requires a "
                "compatible Kokkos MACE-MH-1 evaluator."
            )
        setter(self.execution_mh1_node_arena_policy)
        self.execution_mh1_node_arena_policy = str(
            getattr(
                self.evaluator,
                "execution_mh1_node_arena_policy",
                self.execution_mh1_node_arena_policy,
            )
        )

    def _configure_low_memory(self):
        if self._native_model_type == "MACE_Nonlinear":
            self.low_memory = True
            self.low_memory_policy = "mh1-retain-interaction-v1"
            self.low_memory_selection_reason = (
                "MACE-MH-1 low-memory selection: retain interaction output, "
                "recompute pre-gate state, and use the capacity node arena"
            )
            return
        setter = getattr(self.evaluator, "_set_low_memory", None)
        if setter is None:
            raise ValueError(
                "low_memory=True requires a compatible Kokkos direct MH-0 evaluator."
            )
        setter(True)
        self.low_memory = bool(
            getattr(
                self.evaluator,
                "low_memory_requested",
                getattr(self.evaluator, "low_memory", False),
            )
        )
        if not self.low_memory:
            raise RuntimeError(
                "the native evaluator did not activate the requested low-memory bundle"
            )
        self.m1_polynomial_policy = str(
            getattr(self.evaluator, "m1_polynomial_policy", "recompute")
        )
        self.m1_polynomial_policy_reason = (
            str(getattr(self.evaluator, "m1_recompute_fallback_reason", "")) or None
        )
        self.mh0_state_policy = str(
            getattr(self.evaluator, "mh0_state_policy", "reuse-adjoints-v1")
        )
        self.mh0_state_policy_reason = (
            str(getattr(self.evaluator, "mh0_state_policy_fallback_reason", "")) or None
        )
        self.edge_geometry_policy = str(
            getattr(
                self.evaluator,
                "edge_geometry_policy",
                "unit-f32-radius-f64-v1",
            )
        )
        self._sync_low_memory_policy_state()

    def _sync_low_memory_policy_state(self):
        if not getattr(self, "low_memory_request", False) and not getattr(
            self, "_native_execution_plan_available", False
        ):
            return
        if self._native_model_type == "MACE_Nonlinear":
            return
        self.low_memory = bool(
            getattr(self.evaluator, "low_memory_requested", self.low_memory)
        )
        self.low_memory_policy = str(
            getattr(self.evaluator, "low_memory_policy", self.low_memory_policy)
        )
        self.low_memory_selection_reason = str(
            getattr(
                self.evaluator,
                "low_memory_selection_reason",
                self.low_memory_selection_reason,
            )
        )
        self.m1_polynomial_policy = str(
            getattr(
                self.evaluator,
                "m1_polynomial_policy",
                self.m1_polynomial_policy,
            )
        )
        self.mh0_state_policy = str(
            getattr(self.evaluator, "mh0_state_policy", self.mh0_state_policy)
        )
        self.edge_geometry_policy = str(
            getattr(
                self.evaluator,
                "edge_geometry_policy",
                self.edge_geometry_policy,
            )
        )

    def _kernel_launch_implementation_identity(self):
        values = []
        for value in (self.jit_cache_key, self.jit_artifact_id):
            if value:
                values.append(str(value))
        for name in (
            "standard_m0_module_id",
            "standard_r0_module_id",
            "factorized_jit_artifact_id",
        ):
            value = getattr(self.evaluator, name, None)
            if callable(value):
                value = value()
            if value:
                values.append(str(value))
        return "+".join(sorted(set(values)))

    def _kernel_launch_model_identity(self):
        values = []
        for name in (
            "standard_m0_model_structure_fingerprint",
            "standard_r0_model_structure_fingerprint",
            "factorized_jit_contract_fingerprint",
        ):
            value = getattr(self.evaluator, name, None)
            if callable(value):
                value = value()
            if value:
                values.append(str(value))
        return (
            "+".join(sorted(set(values)))
            or self._kernel_launch_implementation_identity()
        )

    def _apply_kernel_launch_tuning(self, num_nodes, num_edges, properties):
        if self.kernel_launch_policy == "static":
            self.kernel_launch_tuning_status = "static"
            return False
        environment = getattr(
            self.evaluator, "execution_device_execution_environment", None
        )
        if not isinstance(environment, dict) or not environment.get("available"):
            self.kernel_launch_tuning_status = "ineligible"
            self.kernel_launch_tuning_reason = "device execution is unavailable"
            return False
        backend = environment.get("backend")
        if backend not in ("cuda", "hip"):
            self.kernel_launch_tuning_status = "ineligible"
            self.kernel_launch_tuning_reason = (
                f"Execution launch tuning does not support backend {backend!r}; "
                "CUDA or HIP is required"
            )
            return False
        implementation_identity = self._kernel_launch_implementation_identity()
        if not implementation_identity:
            self.kernel_launch_tuning_status = "ineligible"
            self.kernel_launch_tuning_reason = (
                "no tunable launch implementation is active"
            )
            return False
        from .kernel_launch_tuning import (
            load_kernel_launch_tuning_record,
            kernel_launch_tuning_cache_key,
            kernel_launch_tuning_identity,
            kernel_launch_workload_identity,
        )

        identity = kernel_launch_tuning_identity(
            device_environment=environment,
            implementation_identity=implementation_identity,
            model_identity=self._kernel_launch_model_identity(),
            precision=self._kernel_launch_dtype,
            workload=kernel_launch_workload_identity(
                int(num_nodes), int(num_edges), properties
            ),
        )
        cache_key = kernel_launch_tuning_cache_key(identity)
        self.kernel_launch_tuning_cache_key = cache_key
        if self._kernel_launch_tuning_applied_key == cache_key:
            return False
        clear = getattr(self.evaluator, "_clear_kernel_launch_profile_overrides", None)
        if clear is None:
            self.kernel_launch_tuning_status = "ineligible"
            self.kernel_launch_tuning_reason = (
                "native launch-profile controls are unavailable"
            )
            return False
        lookup = load_kernel_launch_tuning_record(identity)
        if lookup.status != "loaded":
            policy_changed = self.kernel_launch_tuning_status == "calibrated"
            if policy_changed:
                clear()
            self.kernel_launch_tuning_status = (
                "resource_default" if lookup.status == "missing" else "ignored"
            )
            self.kernel_launch_tuning_reason = lookup.reason
            self._kernel_launch_tuning_applied_key = cache_key
            return policy_changed
        clear()
        try:
            for decision in lookup.record["decision"]:
                if decision["kind"] == "module":
                    self.evaluator._set_kernel_launch_profile_override(
                        decision["profile_id"],
                        decision["blocks_per_compute_unit"],
                    )
                else:
                    self.evaluator._set_jit_device_plugin_launch_override(
                        decision["stage"],
                        decision["blocks_per_compute_unit"],
                    )
        except (AttributeError, RuntimeError, ValueError) as error:
            clear()
            self.kernel_launch_tuning_status = "ignored"
            self.kernel_launch_tuning_reason = f"{type(error).__name__}: {error}"
        else:
            self.kernel_launch_tuning_status = "calibrated"
            self.kernel_launch_tuning_reason = None
        self._kernel_launch_tuning_applied_key = cache_key
        return True

    @property
    def kernel_launch_profile_diagnostics(self):
        diagnostics = getattr(self.evaluator, "kernel_launch_profile_diagnostics", {})
        return dict(diagnostics) if isinstance(diagnostics, dict) else {}

    @property
    def execution_plan(self):
        """Current native execution selection and graph-time diagnostics.

        The report is pending until the first prepared graph has been sized.
        Private debug pins are deliberately surfaced as their selection source.
        """
        report = getattr(self.evaluator, "execution_plan_report", None)
        if isinstance(report, dict):
            return dict(report)
        if self.execution_profile == "serial-generic":
            return {
                "requested_algorithm": "direct",
                "requested_profile": "serial-generic",
                "selection_source": "serial_fallback",
                "state": "active",
                "selected_id": "serial-generic",
                "selection_reason": self.streamed_edges_resolution_reason,
                "available_bytes": 0,
                "reserve_bytes": 0,
                "boundary_attempt": False,
                "candidates": [],
            }
        return {
            "requested_algorithm": self.execution_algorithm,
            "requested_profile": self.execution_profile,
            "selection_source": "legacy_adapter",
            "state": "active",
            "selected_id": self.execution_algorithm,
            "selection_reason": "native execution-plan reporting is unavailable",
            "available_bytes": 0,
            "reserve_bytes": 0,
            "boundary_attempt": False,
            "candidates": [],
        }

    @staticmethod
    def _read_json_model(model_file):
        try:
            with Path(model_file).open(encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _jit_failed(self, reason, diagnostics=()):
        self.jit_status = "failed"
        self.jit_reason = str(reason)
        self.jit_diagnostics = tuple(str(value) for value in diagnostics)
        message = f"Execution JIT is unavailable: {self.jit_reason}"
        if self.jit_diagnostics:
            message += "\nExecution JIT diagnostics:\n" + "\n".join(
                self.jit_diagnostics
            )
        message += (
            "\nRTC specialization for direct and receiver-factorized execution "
            "does not fall back. Select "
            "streamed_edges='non-compiled' for compiler-free execution."
        )
        raise _JitRequiredError(message)

    def _configure_receiver_factorized_jit(
        self, *, model_file, model_data, dtype, use_kokkos
    ):
        if not use_kokkos:
            self._jit_failed("use_kokkos is false")
        if dtype != "float32":
            self._jit_failed(
                "receiver-factorized host RTC currently requires dtype 'float32'"
            )
        if model_data is None:
            model_data = self._read_json_model(model_file)
        if model_data is None:
            self._jit_failed("the loaded model JSON is unavailable")
            return
        from .prediction_heads import select_prediction_head

        model_data = select_prediction_head(model_data, self.head)
        model_type = model_data.get("model_type", "MACE")
        if model_type not in ("MACE", "MACEField"):
            self._jit_failed(
                "receiver-factorized RTC currently supports ordinary MACE and "
                "MACEField models"
            )
            return
        contract = model_data.get("execution_contracts", {}).get("R1")
        if not isinstance(contract, dict):
            self._jit_failed("the model does not contain an Execution R1 contract")
            return
        execution_space_query = getattr(
            symmetrix, "_kokkos_default_execution_space", None
        )
        execution_space = (
            execution_space_query() if execution_space_query is not None else None
        )
        if execution_space not in _JIT_HOST_EXECUTION_SPACES:
            self._jit_failed(
                "receiver-factorized RTC currently requires a host OpenMP or "
                "Serial Kokkos backend"
            )
            return
        loader = getattr(self.evaluator, "_load_receiver_factorized_host_plugin", None)
        if not callable(loader):
            self._jit_failed(
                "the native evaluator does not provide the receiver-factorized "
                "host-plugin ABI"
            )
            return
        try:
            from .jit import prepare_jit_artifact
            from .receiver_factorized_rtc import (
                receiver_factorized_host_plugin_metadata,
                render_receiver_factorized_host_source,
            )

            generation_version = _require_matching_jit_generation_version()
            metadata = receiver_factorized_host_plugin_metadata(contract)
            source = render_receiver_factorized_host_source(contract)
            result = prepare_jit_artifact(
                source,
                abi={"tag": metadata["abi"], "version": metadata["abi_version"]},
                build={
                    "generator": "symmetrix.receiver-factorized.host-v1",
                    "precision": dtype,
                    "contract": metadata,
                    "jit_generation_version": generation_version,
                },
                cxx_flags=None,
                artifact_name="receiver_factorized_host_plugin",
            )
            self.jit_cache_key = result.cache_key
            self.jit_diagnostics = (*self.jit_diagnostics, *result.diagnostics)
            if not result.available or result.artifact_path is None:
                self._jit_failed(
                    result.reason
                    or "the receiver-factorized JIT cache did not produce an artifact",
                    self.jit_diagnostics,
                )
                return
            loader(str(result.artifact_path))
            if not bool(
                getattr(self.evaluator, "receiver_factorized_host_plugin_ready", False)
            ):
                raise RuntimeError(
                    "the evaluator did not retain the receiver-factorized host plugin"
                )
            self.jit_status = result.status
            self.jit_artifact_path = str(result.artifact_path)
            self.jit_artifact_id = getattr(
                self.evaluator,
                "receiver_factorized_host_plugin_artifact_id",
                metadata["artifact_id"],
            )
            self.jit_compiler_backend = "host-cxx"
            self.jit_reason = None
        except _JitRequiredError:
            raise
        except Exception as exc:
            diagnostics = (*self.jit_diagnostics, f"{type(exc).__name__}: {exc}")
            self._jit_failed(exc, diagnostics)

    def _configure_low_memory_operator_modules(
        self,
        *,
        model_data,
        dtype,
        backend,
        target,
        jit_generation_version,
        prefer_host_m0_plugin=False,
    ):
        from .jit_operator_artifact import (
            prepare_low_memory_operator_modules,
        )

        modules, diagnostics = prepare_low_memory_operator_modules(
            self.evaluator,
            model_data=model_data,
            precision=dtype,
            backend=backend,
            target=target,
            jit_generation_version=jit_generation_version,
            prefer_host_m0_plugin=prefer_host_m0_plugin,
        )
        self.jit_operator_modules.update(modules)
        self.jit_diagnostics = (*self.jit_diagnostics, *diagnostics)

    def _configure_jit(
        self,
        *,
        model_file,
        model_data,
        dtype,
        use_kokkos,
        streamed_edges,
    ):
        if streamed_edges not in ("direct",):
            self.jit_policy = "not_applicable"
            self.jit_status = "not_applicable"
            self.jit_reason = "streamed_edges does not require RTC"
            return

        self.jit_policy = self._jit_policy_from_environment()
        if self.jit_policy == _JIT_DEBUG_POLICY:
            self._jit_failed(
                f"{_JIT_POLICY_ENVIRONMENT_VARIABLE}=none is incompatible "
                f"with streamed_edges={streamed_edges!r}"
            )

        if not use_kokkos:
            self._jit_failed("use_kokkos is false")

        if model_data is None:
            model_data = self._read_json_model(model_file)
        if model_data is None:
            self._jit_failed("the loaded model JSON is unavailable")
            return
        from .prediction_heads import select_prediction_head

        model_data = select_prediction_head(model_data, self.head)
        model_type = model_data.get("model_type", "MACE")
        if model_type == "MACE" and bool(model_data.get("single_layer_readout")):
            execution_space_query = getattr(
                symmetrix, "_kokkos_default_execution_space", None
            )
            execution_space = (
                execution_space_query() if execution_space_query is not None else None
            )
            environment_query = getattr(
                self.evaluator, "execution_device_execution_environment", None
            )
            device_environment = (
                environment_query()
                if callable(environment_query)
                else environment_query
            )
            if not isinstance(device_environment, dict):
                module_environment_query = getattr(
                    symmetrix, "_execution_device_execution_environment", None
                )
                device_environment = (
                    module_environment_query()
                    if callable(module_environment_query)
                    else {}
                )
            backend = device_environment.get("backend")
            if self.low_memory_request and (
                backend in ("cuda", "hip")
                or execution_space in ("Cuda", "HIP")
                or execution_space in _JIT_HOST_EXECUTION_SPACES
            ):
                if backend == "cuda" or execution_space == "Cuda":
                    resolved_backend = "cuda"
                elif backend == "hip" or execution_space == "HIP":
                    resolved_backend = "hip"
                else:
                    resolved_backend = "host"
                self._configure_low_memory_operator_modules(
                    model_data=model_data,
                    dtype=dtype,
                    backend=resolved_backend,
                    target=device_environment,
                    jit_generation_version=_require_matching_jit_generation_version(),
                )
            self.jit_status = "not_applicable"
            self.jit_reason = (
                "single-layer invariant readout uses prepared R0/M0 execution "
                "without an R1 specialization"
            )
            return
        if model_type in ("MACE", "MACEField"):
            specialization = "r1"
            contract_name = "R1"
        elif model_type == "MACE_Nonlinear":
            specialization = "mh1"
            contract_name = "MH1_UVU"
        else:
            self._jit_failed(
                "Execution JIT currently supports ordinary MACE, MACEField, and "
                "compatible MACE-MH-1-family models"
            )
            return
        contract = model_data.get("execution_contracts", {}).get(contract_name)
        if not isinstance(contract, dict):
            self._jit_failed(
                f"the model does not contain a Execution {contract_name} contract"
            )
            return

        selected_forward_policy = None
        selected_source_policy = None
        selected_edge_policy = None
        selected_node_state_policy = None
        selected_variant_id = None
        mh1_v3_contract = None
        launch_plan = None
        try:
            jit_generation_version = _require_matching_jit_generation_version()
            if specialization == "mh1":
                from .execution_mh1_contract import (
                    project_execution_mh1_v3_contract,
                    validate_execution_mh1_v4_contract_for_model,
                )

                contract = validate_execution_mh1_v4_contract_for_model(
                    model_data, contract
                )
                mh1_v3_contract = project_execution_mh1_v3_contract(contract)
                set_contract_identity = getattr(
                    self.evaluator,
                    "_set_execution_mh1_contract_identity",
                    None,
                )
                if callable(set_contract_identity):
                    set_contract_identity(
                        mh1_v3_contract["tag"],
                        mh1_v3_contract["generation_fingerprint"],
                        mh1_v3_contract["semantic_fingerprint"],
                        mh1_v3_contract["structure_fingerprint"],
                    )
                set_v4_contract_identity = getattr(
                    self.evaluator,
                    "_set_execution_mh1_v4_contract_identity",
                    None,
                )
                if callable(set_v4_contract_identity):
                    set_v4_contract_identity(
                        contract["tag"],
                        contract["generation_fingerprint"],
                        contract["semantic_fingerprint"],
                        contract["structure_fingerprint"],
                        contract["runtime_layout_fingerprint"],
                    )
            execution_space_query = getattr(
                symmetrix, "_kokkos_default_execution_space", None
            )
            execution_space = (
                execution_space_query() if execution_space_query is not None else None
            )
            environment_query = getattr(
                self.evaluator, "execution_device_execution_environment", None
            )
            device_environment = (
                environment_query()
                if callable(environment_query)
                else environment_query
            )
            if not isinstance(device_environment, dict):
                module_environment_query = getattr(
                    symmetrix, "_execution_device_execution_environment", None
                )
                device_environment = (
                    module_environment_query()
                    if callable(module_environment_query)
                    else {}
                )
            execution_backend = device_environment.get("backend")
            if execution_space in _JIT_HOST_EXECUTION_SPACES:
                loader_name = (
                    "_load_jit_mh1_host_plugin_v5"
                    if specialization == "mh1"
                    else "_load_jit_host_plugin"
                )
                if not hasattr(self.evaluator, loader_name):
                    raise RuntimeError(
                        "the native evaluator does not provide the host-plugin ABI"
                    )
                jit_backend = "host"
                cuda_environment = None
            elif execution_backend == "cuda" or execution_space == "Cuda":
                loader_name = (
                    "_load_jit_mh1_cuda_plugin_v4"
                    if specialization == "mh1"
                    else "_load_jit_cuda_plugin"
                )
                if not hasattr(self.evaluator, loader_name):
                    raise RuntimeError(
                        "the native evaluator does not provide the generated "
                        f"{model_type} CUDA-plugin ABI"
                    )
                cuda_environment = device_environment
                if not cuda_environment:
                    cuda_environment = getattr(
                        self.evaluator, "device_cuda_environment", None
                    )
                    if callable(cuda_environment):
                        cuda_environment = cuda_environment()
                if not isinstance(cuda_environment, dict) or not bool(
                    cuda_environment.get("available", False)
                ):
                    raise RuntimeError(
                        "the native evaluator did not report a usable CUDA target"
                    )
                jit_backend = "cuda"
            elif execution_backend == "hip" or execution_space == "HIP":
                loader_name = (
                    "_load_jit_mh1_cuda_plugin_v4"
                    if specialization == "mh1"
                    else "_load_jit_hip_plugin"
                )
                if not hasattr(self.evaluator, loader_name):
                    raise RuntimeError(
                        "the native evaluator does not provide the generated "
                        "Execution R1 HIP-plugin ABI"
                    )
                if not bool(device_environment.get("available", False)):
                    raise RuntimeError(
                        "the native evaluator did not report a usable HIP target"
                    )
                jit_backend = "hip"
            else:
                raise RuntimeError(
                    "the Kokkos default execution space is not supported by Execution JIT"
                )
            if (
                specialization == "mh1"
                and jit_backend == "host"
                and self.execution_mh1_node_state_policy == "reuse-adjoints-v1"
            ):
                raise RuntimeError(
                    "the selected execution_mh1_node_state_policy "
                    "currently requires a generated CUDA or HIP module"
                )

            from .jit import (
                quarantine_jit_artifact,
                remove_jit_quarantine,
            )

            device_plan = None
            if specialization == "r1" and jit_backend in ("cuda", "hip"):
                plan_target = (
                    cuda_environment if jit_backend == "cuda" else device_environment
                )
                device_plan = _factorized_device_backend_plan(
                    jit_backend,
                    plan_target,
                    contract,
                    dtype,
                    self.evaluator,
                )
                self.jit_compiler_request = device_plan.policy
                self.jit_compiler_backend = device_plan.selected_compiler
                metadata = device_plan.metadata
                selected_edge_policy = device_plan.edge_policy
                selected_variant_id = device_plan.variant_id
                ready_attribute = device_plan.ready_attribute
                artifact_attribute = device_plan.artifact_attribute
                load_plugin = device_plan.loader
                attempt = device_plan.build_attempt(self.jit_compiler_backend)
                source = attempt.source
                prepare = attempt.prepare
                prepare_arguments = attempt.arguments
            elif specialization == "mh1" and jit_backend == "hip":
                from .jit import (
                    execution_hip_jit_backend_request,
                    prepare_hiprtc_jit_artifact,
                    select_execution_hip_jit_backend,
                )
                from .mh1_jit_codegen import (
                    MH1_HIP_MODULE_ABI,
                    MH1_HIP_MODULE_ABI_VERSION,
                    execution_mh1_hip_module_v4_launch_plan,
                    execution_mh1_hip_module_v4_metadata,
                    render_execution_mh1_hip_module_v4,
                )

                self.jit_compiler_request = execution_hip_jit_backend_request()
                self.jit_compiler_backend = select_execution_hip_jit_backend(
                    self.jit_compiler_request
                )
                if self.jit_compiler_backend != "hiprtc":
                    raise RuntimeError(
                        "generated MACE-MH-1 HIP specialization requires hipRTC; "
                        "hipcc artifacts are not supported"
                    )
                metadata = execution_mh1_hip_module_v4_metadata(
                    contract,
                    device_environment,
                    node_state_policy=self.execution_mh1_node_state_policy,
                )
                source = render_execution_mh1_hip_module_v4(
                    contract,
                    device_environment,
                    node_state_policy=self.execution_mh1_node_state_policy,
                    precision=dtype,
                )
                launch_plan = execution_mh1_hip_module_v4_launch_plan(
                    contract,
                    device_environment,
                    node_state_policy=self.execution_mh1_node_state_policy,
                    precision=dtype,
                )
                selected_forward_policy = metadata["forward_policy"]
                selected_source_policy = metadata["source_policy"]
                selected_edge_policy = metadata["edge_policy"]
                selected_node_state_policy = metadata["node_state_policy"]
                selected_variant_id = (
                    f"{metadata['artifact_id']}-target-"
                    f"{metadata['target_id'].removeprefix('sha256:')[:12]}"
                    f"-node-{metadata['node_state_policy']}"
                )
                prepare = prepare_hiprtc_jit_artifact
                prepare_arguments = {
                    "abi": {
                        "tag": MH1_HIP_MODULE_ABI,
                        "version": MH1_HIP_MODULE_ABI_VERSION,
                    },
                    "build": {
                        "generator": "symmetrix.jit.mh1-hip-module-v1",
                        "precision": dtype,
                        "contract": metadata,
                        "launch_plan": launch_plan,
                    },
                    "target": device_environment,
                    "artifact_name": "execution_mh1_hip_module",
                }
                ready_attribute = "jit_mh1_cuda_plugin_v4_ready"
                artifact_attribute = "jit_mh1_cuda_plugin_artifact_id"
                load_plugin = getattr(self.evaluator, loader_name)
            elif jit_backend == "cuda":
                from .jit import (
                    prepare_execution_cuda_jit_artifact,
                    prepare_nvrtc_jit_artifact,
                    select_execution_cuda_jit_backend,
                    execution_cuda_jit_backend_request,
                )

                compute_capability = cuda_environment.get("compute_capability_code")
                if not isinstance(compute_capability, int):
                    dotted = str(cuda_environment.get("compute_capability", ""))
                    major, separator, minor = dotted.partition(".")
                    if not separator or not major.isdigit() or not minor.isdigit():
                        raise RuntimeError(
                            "the native evaluator reported an invalid CUDA "
                            "compute capability"
                        )
                    compute_capability = 10 * int(major) + int(minor)
                self.jit_compiler_request = execution_cuda_jit_backend_request()
                self.jit_compiler_backend = select_execution_cuda_jit_backend(
                    self.jit_compiler_request
                )
                if (
                    specialization == "mh1"
                    and dtype != "float32"
                    and self.jit_compiler_backend != "nvrtc"
                ):
                    raise RuntimeError(
                        "generated FP64 MACE-MH-1 CUDA specialization requires NVRTC"
                    )
                if specialization == "mh1":
                    from . import mh1_jit_codegen as jit_codegen

                    metadata_function = getattr(
                        jit_codegen, "jit_mh1_cuda_plugin_v4_metadata", None
                    )
                    render_function = getattr(
                        jit_codegen, "render_jit_mh1_cuda_plugin_v4", None
                    )
                    module_render_function = getattr(
                        jit_codegen, "render_execution_mh1_cuda_module_v4", None
                    )
                    launch_plan_function = getattr(
                        jit_codegen,
                        "execution_mh1_cuda_module_v4_launch_plan",
                        None,
                    )
                    forward_policy_function = getattr(
                        jit_codegen,
                        "resolve_execution_mh1_cuda_forward_policy",
                        None,
                    )
                    source_policy_function = getattr(
                        jit_codegen,
                        "resolve_execution_mh1_cuda_source_policy",
                        None,
                    )
                    edge_policy_function = getattr(
                        jit_codegen,
                        "resolve_execution_mh1_cuda_edge_policy",
                        None,
                    )
                    if (
                        metadata_function is None
                        or render_function is None
                        or module_render_function is None
                        or launch_plan_function is None
                        or forward_policy_function is None
                        or source_policy_function is None
                        or edge_policy_function is None
                    ):
                        raise RuntimeError(
                            "generated MACE-MH-1 CUDA specialization is not "
                            "available in this Symmetrix build"
                        )
                    forward_policy = forward_policy_function(
                        mh1_v3_contract, compute_capability, "auto"
                    )
                    source_policy = source_policy_function(
                        mh1_v3_contract, compute_capability, "auto"
                    )
                    edge_policy = edge_policy_function(
                        mh1_v3_contract, compute_capability, "auto"
                    )
                    metadata = metadata_function(
                        contract,
                        compute_capability,
                        forward_policy=forward_policy,
                        source_policy=source_policy,
                        edge_policy=edge_policy,
                        node_state_policy=self.execution_mh1_node_state_policy,
                    )
                    source = render_function(
                        contract,
                        compute_capability,
                        forward_policy=forward_policy,
                        source_policy=source_policy,
                        edge_policy=edge_policy,
                        node_state_policy=self.execution_mh1_node_state_policy,
                    )
                    if self.jit_compiler_backend == "nvrtc":
                        source = module_render_function(
                            contract,
                            compute_capability,
                            forward_policy=forward_policy,
                            source_policy=source_policy,
                            edge_policy=edge_policy,
                            node_state_policy=self.execution_mh1_node_state_policy,
                            precision=dtype,
                        )
                        launch_plan = launch_plan_function(
                            contract,
                            compute_capability,
                            forward_policy=forward_policy,
                            source_policy=source_policy,
                            edge_policy=edge_policy,
                            node_state_policy=self.execution_mh1_node_state_policy,
                            precision=dtype,
                        )
                    selected_forward_policy = forward_policy
                    selected_source_policy = source_policy
                    selected_edge_policy = edge_policy
                    selected_node_state_policy = metadata["node_state_policy"]
                    selected_variant_id = (
                        f"{metadata['artifact_id']}-fwd-"
                        f"{forward_policy['policy_id'].removeprefix('sha256:')[:12]}"
                        "-src-"
                        f"{source_policy['policy_id'].removeprefix('sha256:')[:12]}"
                        "-edge-"
                        f"{edge_policy['policy_id'].removeprefix('sha256:')[:12]}"
                        f"-node-{metadata['node_state_policy']}"
                    )
                    if self.jit_compiler_backend == "nvrtc":
                        generator = "symmetrix.jit.mh1-cuda-module-v1"
                        artifact_name = "execution_mh1_cuda_module"
                    else:
                        generator = "symmetrix.jit.mh1-cuda-v13"
                        artifact_name = "jit_mh1_cuda_plugin"
                    ready_attribute = "jit_mh1_cuda_plugin_v4_ready"
                    artifact_attribute = "jit_mh1_cuda_plugin_artifact_id"
                prepare = (
                    prepare_nvrtc_jit_artifact
                    if self.jit_compiler_backend == "nvrtc"
                    else prepare_execution_cuda_jit_artifact
                )
                abi_tag = metadata["abi"]
                abi_version = metadata["abi_version"]
                if specialization == "mh1" and launch_plan is not None:
                    abi_tag = jit_codegen.MH1_CUDA_MODULE_ABI
                    abi_version = jit_codegen.MH1_CUDA_MODULE_ABI_VERSION
                prepare_arguments = {
                    "abi": {
                        "tag": abi_tag,
                        "version": abi_version,
                    },
                    "build": {
                        "generator": generator,
                        "precision": dtype,
                        "contract": metadata,
                    },
                    "compute_capability": cuda_environment,
                    "artifact_name": artifact_name,
                }
                if launch_plan is not None:
                    prepare_arguments["build"]["launch_plan"] = launch_plan
                load_plugin = getattr(self.evaluator, loader_name)
            else:
                from .jit import prepare_jit_artifact

                if specialization == "mh1":
                    from . import mh1_jit_codegen as jit_codegen
                    from .mh1_jit_codegen import (
                        render_jit_mh1_host_plugin_v5 as render_function,
                    )
                    from .mh1_jit_codegen import (
                        execution_mh1_node_program_metadata,
                    )

                    def metadata_function(value):
                        return execution_mh1_node_program_metadata(
                            value,
                            backend="host",
                            node_state_policy=self.execution_mh1_node_state_policy,
                        )

                    generator = "symmetrix.jit.mh1-host-v5"
                    artifact_name = "jit_mh1_host_plugin"
                    ready_attribute = "jit_mh1_host_plugin_v5_ready"
                    artifact_attribute = "jit_mh1_host_plugin_artifact_id"
                else:
                    from .jit_codegen import (
                        render_jit_r1_host_plugin as render_function,
                    )
                    from .jit_codegen import (
                        jit_r1_host_plugin_metadata as metadata_function,
                    )

                    generator = "symmetrix.jit.r1-host-v2"
                    artifact_name = "factorized_host_plugin"
                    ready_attribute = "jit_host_plugin_ready"
                    artifact_attribute = "jit_host_plugin_artifact_id"
                if specialization == "r1":
                    metadata = metadata_function(contract, precision=dtype)
                    source = render_function(contract, precision=dtype)
                else:
                    metadata = metadata_function(contract)
                    source = render_function(
                        contract,
                        precision=dtype,
                        node_state_policy=self.execution_mh1_node_state_policy,
                    )
                    if specialization == "mh1":
                        selected_node_state_policy = metadata["node_state_policy"]
                        metadata = {
                            **metadata,
                            "abi": jit_codegen.MH1_HOST_PLUGIN_V5_ABI,
                            "abi_version": jit_codegen.MH1_HOST_PLUGIN_V5_ABI_VERSION,
                        }
                prepare = prepare_jit_artifact
                prepare_arguments = {
                    "abi": {
                        "tag": metadata["abi"],
                        "version": metadata["abi_version"],
                    },
                    "build": {
                        "generator": generator,
                        "precision": dtype,
                        "contract": metadata,
                    },
                    "cxx_flags": None,
                    "artifact_name": artifact_name,
                }
                load_plugin = getattr(self.evaluator, loader_name)

            result = None
            prepare_arguments["build"] = {
                **prepare_arguments["build"],
                "jit_generation_version": jit_generation_version,
            }
            retained_quarantines = []
            failed_load_artifacts = set()
            for _load_attempt in range(4):
                result = prepare(source, **prepare_arguments)
                self.jit_cache_key = result.cache_key
                self.jit_diagnostics = (
                    *self.jit_diagnostics,
                    *result.diagnostics,
                )
                if not result.available or result.artifact_path is None:
                    self._jit_failed(
                        result.reason or "the JIT cache did not produce an artifact",
                        self.jit_diagnostics,
                    )
                    return
                try:
                    if specialization == "mh1" and launch_plan is not None:
                        load_plugin(
                            str(result.artifact_path),
                            json.dumps(
                                launch_plan, sort_keys=True, separators=(",", ":")
                            ),
                        )
                    elif specialization == "r1" and jit_backend == "cuda":
                        load_plugin(
                            str(result.artifact_path),
                            metadata["persistent_blocks_per_compute_unit"],
                        )
                    else:
                        load_plugin(str(result.artifact_path))
                    break
                except Exception as load_error:
                    failed_artifact = (
                        self.jit_compiler_backend,
                        result.cache_key,
                        str(result.artifact_path),
                    )
                    if failed_artifact in failed_load_artifacts:
                        raise
                    failed_load_artifacts.add(failed_artifact)
                    retained_quarantine = quarantine_jit_artifact(
                        result, f"{type(load_error).__name__}: {load_error}"
                    )
                    retained_quarantines.append(retained_quarantine)
                    self.jit_diagnostics = (
                        *self.jit_diagnostics,
                        *retained_quarantine.diagnostics,
                    )
            else:
                raise RuntimeError(
                    "Execution JIT artifact recovery attempts were exhausted"
                )
            assert result is not None
            if hasattr(self.evaluator, ready_attribute) and not bool(
                getattr(self.evaluator, ready_attribute)
            ):
                raise RuntimeError(
                    f"the evaluator did not retain the loaded {jit_backend} plugin"
                )
            for retained_quarantine in retained_quarantines:
                self.jit_diagnostics = (
                    *self.jit_diagnostics,
                    *remove_jit_quarantine(retained_quarantine),
                )
            self.jit_status = result.status
            self.jit_artifact_path = str(result.artifact_path)
            self.jit_artifact_id = getattr(
                self.evaluator, artifact_attribute, metadata["artifact_id"]
            )
            self.jit_forward_policy = selected_forward_policy
            self.jit_source_policy = selected_source_policy
            self.jit_edge_policy = selected_edge_policy
            self.jit_node_state_policy = selected_node_state_policy
            self.jit_variant_id = selected_variant_id
            if specialization == "r1" and self.low_memory_request:
                if jit_backend in ("cuda", "hip"):
                    self._configure_low_memory_operator_modules(
                        model_data=model_data,
                        dtype=dtype,
                        backend=jit_backend,
                        target=(
                            cuda_environment
                            if jit_backend == "cuda"
                            else device_environment
                        ),
                        jit_generation_version=jit_generation_version,
                    )
                elif jit_backend == "host" and callable(
                    getattr(self.evaluator, "_set_low_memory", None)
                ):
                    self._configure_low_memory_operator_modules(
                        model_data=model_data,
                        dtype=dtype,
                        backend="host",
                        target=device_environment,
                        jit_generation_version=jit_generation_version,
                        prefer_host_m0_plugin=True,
                    )
            self.jit_reason = None
        except _JitRequiredError:
            raise
        except Exception as exc:
            diagnostics = (*self.jit_diagnostics, f"{type(exc).__name__}: {exc}")
            self._jit_failed(exc, diagnostics)

    @staticmethod
    def _native_evaluator_class(model_type, dtype, use_kokkos):
        if model_type == "MACE_Nonlinear":
            if use_kokkos:
                evaluator_name = (
                    "MACENonlinearKokkos"
                    if dtype == "float64"
                    else "MACENonlinearKokkosFloat"
                )
                if not hasattr(symmetrix, evaluator_name):
                    raise RuntimeError(
                        "This Symmetrix build does not provide the requested "
                        "native MACE_Nonlinear Kokkos evaluator."
                    )
                if not symmetrix._kokkos_is_initialized():
                    symmetrix._init_kokkos()
                return getattr(symmetrix, evaluator_name)
            if dtype != "float64":
                raise ValueError(
                    "Native serial MACE_Nonlinear models currently require "
                    "dtype 'float64'."
                )
            return symmetrix.MACENonlinear
        if model_type not in ("MACE", "MACEField"):
            raise ValueError(f"Unsupported Symmetrix model_type '{model_type}'.")
        if use_kokkos:
            if not symmetrix._kokkos_is_initialized():
                symmetrix._init_kokkos()
            return (
                symmetrix.MACEKokkos
                if dtype == "float64"
                else symmetrix.MACEKokkosFloat
            )
        return symmetrix.MACE if dtype == "float64" else symmetrix.MACEFloat

    @staticmethod
    def _is_native_json_parse_error(error):
        message = str(error)
        return "[json.exception.parse_error." in message or (
            "not native json" in message.lower()
        )

    @staticmethod
    def _is_native_json_schema_error(error):
        message = str(error)
        return "[json.exception." in message and not (
            Symmetrix._is_native_json_parse_error(error)
        )

    @staticmethod
    def _json_metadata(model_file):
        try:
            model_type, has_field_coupling = symmetrix._model_metadata(str(model_file))
            return {
                "model_type": model_type,
                "has_field_coupling": has_field_coupling,
            }
        except RuntimeError as error:
            if Symmetrix._is_native_json_schema_error(error):
                raise
            return None
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def _raise_if_macefield_checkpoint(self, model_file):
        try:
            import torch
        except ImportError:
            return

        model = torch.load(
            model_file,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        is_macefield = type(model).__name__ == "MACEField" or (
            hasattr(model, "field_feats") and hasattr(model, "field_linear")
        )
        if is_macefield:
            raise RuntimeError(
                "MACEField PyTorch checkpoints cannot be used directly with Symmetrix. "
                "Convert/extract the model to Symmetrix JSON first, then pass the JSON file."
            )

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if (
            self._has_native_field_coupling()
            and not state
            and (
                not hasattr(self, "atoms")
                or not equal(
                    self._macefield_electric_field,
                    self._resolve_electric_field(atoms),
                    atol=tol,
                )
            )
        ):
            state.append("info")
        return state

    def _has_native_field_coupling(self):
        return hasattr(self, "evaluator") and getattr(
            self.evaluator, "has_field_coupling", False
        )

    @property
    def electric_field(self):
        return self._electric_field

    @electric_field.setter
    def electric_field(self, value):
        self._electric_field = value
        self.results.clear()

    def _resolve_electric_field(self, atoms=None):
        if atoms is None:
            atoms = self.atoms

        if self._electric_field is not None:
            field = self._electric_field
        elif "electric_field" in atoms.info:
            field = atoms.info["electric_field"]
        elif "REF_electric_field" in atoms.info:
            field = atoms.info["REF_electric_field"]
        else:
            field = np.zeros(3)

        field = np.asarray(field, dtype=float)
        if field.shape == (3,):
            return field
        if field.shape == (1, 3):
            return field.reshape(3)
        if field.shape == (len(atoms), 3):
            raise ValueError(
                "MACEField ASE electric_field must be a graph-level electric_field "
                "with shape (3,) or (1, 3); per-atom fields are not supported."
            )
        raise ValueError("electric_field must have shape (3,) or (1, 3).")

    @staticmethod
    def _minimum_image_displacements(positions, reference_positions, cell, pbc):
        displacements = np.asarray(positions) - np.asarray(reference_positions)
        if not np.any(pbc):
            return displacements
        try:
            fractional = np.linalg.solve(cell.T, displacements.T).T
        except np.linalg.LinAlgError:
            return None
        fractional[:, pbc] -= np.rint(fractional[:, pbc])
        return fractional @ cell

    def _materialize_neighbor_cache_geometry(self):
        cache = self._neighbor_cache
        if cache is None:
            raise RuntimeError("Neighbor geometry requires a prepared cache.")
        if cache.device_graph_generation != 0:
            raise RuntimeError(
                "A device-resident neighbor graph has no host edge geometry."
            )
        if cache.host_geometry_generation != cache.geometry_generation:
            cache.reference_xyz = np.ascontiguousarray(
                cache.reference_positions[cache.sources]
                - cache.reference_positions[cache.receivers]
                + cache.shifts @ cache.cell,
                dtype=float,
            )
            cache.host_geometry_generation = cache.geometry_generation
            self.neighbor_cache_host_geometry_materialization_count += 1
        return cache

    def _cached_neighbor_geometry(
        self, atoms, native_geometry=False, allow_device_neighbor_graph=True
    ):
        atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        pbc = np.asarray(atoms.pbc, dtype=bool)
        cell = np.asarray(atoms.cell, dtype=float)
        positions = np.asarray(atoms.positions, dtype=float)
        cache = self._neighbor_cache
        displacements = None
        cell_changed = False
        geometry_reference_positions = None
        if (
            cache is not None
            and np.array_equal(atomic_numbers, cache.atomic_numbers)
            and np.array_equal(pbc, cache.pbc)
            and positions.shape == cache.reference_positions.shape
            and (allow_device_neighbor_graph or cache.device_graph_generation == 0)
        ):
            cell_changed = not np.array_equal(cell, cache.cell)
            if np.all(pbc) and cache.fractional_geometry_eligible:
                try:
                    topology_fractional = cache.topology_fractional_positions
                    non_affine = positions - topology_fractional @ cell
                    non_affine_fractional = np.linalg.solve(cell.T, non_affine.T).T
                except np.linalg.LinAlgError:
                    displacements = None
                else:
                    non_affine_fractional[:, pbc] -= np.rint(
                        non_affine_fractional[:, pbc]
                    )
                    non_affine = non_affine_fractional @ cell
                    max_non_affine_displacement = (
                        0.0
                        if len(non_affine) == 0
                        else float(np.linalg.norm(non_affine, axis=1).max())
                    )
                    deformation = np.linalg.solve(cell, cache.topology_cell)
                    contraction = float(np.linalg.norm(deformation, ord=2))
                    topology_valid = (
                        contraction * (self.cutoff + 2.0 * max_non_affine_displacement)
                        <= self.cutoff + self.neighbor_skin
                    )
                    if topology_valid and not cell_changed:
                        displacements = self._minimum_image_displacements(
                            positions, cache.reference_positions, cell, pbc
                        )
                    elif topology_valid:
                        geometry_reference_positions = (
                            topology_fractional @ cell + non_affine
                        )
                        displacements = np.zeros_like(positions)
            else:
                if not cell_changed:
                    displacements = self._minimum_image_displacements(
                        positions, cache.topology_reference_positions, cell, pbc
                    )
                if displacements is not None and len(displacements) != 0:
                    max_displacement = float(
                        np.linalg.norm(displacements, axis=1).max()
                    )
                    if max_displacement > 0.5 * self.neighbor_skin:
                        displacements = None

            if displacements is not None:
                self.neighbor_cache_reuse_count += 1

        if displacements is None:
            native_geometry_eligible = True
            fractional_geometry_eligible = bool(np.all(pbc))
            if np.any(pbc):
                try:
                    inverse_cell = np.linalg.inv(cell)
                except np.linalg.LinAlgError:
                    inverse_cell = np.zeros((3, 3), dtype=float)
                    native_geometry_eligible = False
                    fractional_geometry_eligible = False
            else:
                inverse_cell = np.zeros((3, 3), dtype=float)
                fractional_geometry_eligible = False
            if fractional_geometry_eligible:
                topology_fractional_positions = np.ascontiguousarray(
                    positions @ inverse_cell, dtype=float
                )
            else:
                topology_fractional_positions = np.empty((0, 3), dtype=float)
            type_by_atomic_number = {
                atomic_number: index
                for index, atomic_number in enumerate(self.evaluator.atomic_numbers)
            }
            node_types = np.asarray(
                [type_by_atomic_number[value] for value in atomic_numbers],
                dtype=np.int32,
            )
            device_builder = getattr(
                self.evaluator, "_prepare_periodic_factorized_graph", None
            )
            neighbor_backend = _neighbor_backend_request()
            device_graph_eligible = (
                allow_device_neighbor_graph
                and native_geometry
                and fractional_geometry_eligible
                and getattr(self.evaluator, "streamed_edges_mode", None) == "direct"
                and callable(device_builder)
            )
            if neighbor_backend == "kokkos" and not device_graph_eligible:
                raise RuntimeError(
                    "SYMMETRIX_NEIGHBOR_BACKEND=kokkos requires fully periodic "
                    "direct Kokkos execution with native geometry support."
                )
            use_device_graph = _use_device_neighbor_graph(
                neighbor_backend,
                device_graph_eligible,
                len(atomic_numbers),
                getattr(symmetrix, "_kokkos_default_execution_space", lambda: None)(),
            )
            if use_device_graph:
                device_graph = device_builder(
                    node_types,
                    np.ascontiguousarray(positions, dtype=float).reshape(-1),
                    np.ascontiguousarray(cell, dtype=float).reshape(-1),
                    np.ascontiguousarray(inverse_cell, dtype=float).reshape(-1),
                    self.cutoff + self.neighbor_skin,
                )
                self._sync_low_memory_policy_state()
                num_edges = int(device_graph["num_edges"])
                _validate_int32_graph_cardinality(len(atomic_numbers), num_edges)
                receivers = np.empty(0, dtype=np.int32)
                sources = np.empty(0, dtype=np.int32)
                shifts = np.empty((0, 3), dtype=np.int32)
                num_neigh = np.empty(0, dtype=np.int32)
                neigh_types = np.empty(0, dtype=np.int32)
                device_graph_generation = int(device_graph["generation"])
                topology_fractional_xyz = np.empty((0, 3), dtype=float)
                self.neighbor_graph_backend = "kokkos"
            else:
                receivers, sources, shifts = neighbor_list(
                    "ijS", atoms, self.cutoff + self.neighbor_skin
                )
                _validate_int32_graph_cardinality(len(atomic_numbers), len(receivers))
                receivers, sources, shifts = _receiver_major_neighbor_arrays(
                    receivers, sources, shifts
                )
                num_edges = len(receivers)
                num_neigh = np.asarray(
                    np.bincount(receivers, minlength=len(positions)),
                    dtype=np.int32,
                )
                neigh_types = np.ascontiguousarray(node_types[sources])
                device_graph_generation = 0
                self.neighbor_graph_backend = "host"
                if fractional_geometry_eligible:
                    topology_fractional_xyz = np.ascontiguousarray(
                        topology_fractional_positions[sources]
                        - topology_fractional_positions[receivers]
                        + shifts,
                        dtype=float,
                    )
                else:
                    topology_fractional_xyz = np.empty((0, 3), dtype=float)
            cache = _NeighborCache(
                atomic_numbers=np.array(atomic_numbers, copy=True),
                pbc=np.array(pbc, copy=True),
                topology_cell=np.array(cell, copy=True),
                topology_reference_positions=np.array(positions, copy=True),
                topology_fractional_positions=topology_fractional_positions,
                topology_fractional_xyz=topology_fractional_xyz,
                cell=np.array(cell, copy=True),
                reference_positions=np.array(positions, copy=True),
                receivers=receivers,
                sources=sources,
                shifts=np.ascontiguousarray(shifts),
                reference_xyz=np.empty((0, 3), dtype=float),
                inverse_cell=np.ascontiguousarray(inverse_cell, dtype=float),
                node_types=node_types,
                num_neigh=num_neigh,
                neigh_types=neigh_types,
                native_geometry_eligible=native_geometry_eligible,
                fractional_geometry_eligible=fractional_geometry_eligible,
                generation=self.neighbor_cache_build_count + 1,
                geometry_generation=self.neighbor_cache_geometry_update_count + 1,
                host_geometry_generation=0,
                device_graph_generation=device_graph_generation,
                device_geometry_generation=(
                    self.neighbor_cache_geometry_update_count + 1
                    if device_graph_generation != 0
                    else 0
                ),
                num_edges=num_edges,
            )
            self._neighbor_cache = cache
            self.neighbor_cache_build_count += 1
            self.neighbor_cache_geometry_update_count += 1
            displacements = np.zeros_like(positions)
        elif cell_changed:
            if geometry_reference_positions is None:
                raise RuntimeError(
                    "Cell-aware neighbor reuse requires normalized reference positions."
                )
            try:
                inverse_cell = np.linalg.inv(cell)
            except np.linalg.LinAlgError:
                inverse_cell = np.zeros((3, 3), dtype=float)
                native_geometry_eligible = False
            else:
                native_geometry_eligible = True
            cache.cell = np.array(cell, copy=True)
            cache.reference_positions = np.array(
                geometry_reference_positions, copy=True
            )
            cache.inverse_cell = np.ascontiguousarray(inverse_cell, dtype=float)
            cache.native_geometry_eligible = native_geometry_eligible
            cache.fractional_geometry_eligible = bool(
                native_geometry_eligible and np.all(pbc)
            )
            self.neighbor_cache_geometry_update_count += 1
            cache.geometry_generation = self.neighbor_cache_geometry_update_count
            displacements = np.zeros_like(positions)

        if native_geometry and cache.native_geometry_eligible:
            return (
                np.ascontiguousarray(cache.receivers),
                np.ascontiguousarray(cache.sources),
                None,
                None,
            )

        cache = self._materialize_neighbor_cache_geometry()
        xyz = (
            cache.reference_xyz
            + displacements[cache.sources]
            - displacements[cache.receivers]
        )
        distances = np.linalg.norm(xyz, axis=1)
        if (
            getattr(self.evaluator, "streamed_edges_mode", None)
            in _PREPARED_EXECUTION_MODES
        ):
            inactive = distances >= self.cutoff
            if np.any(inactive):
                xyz = np.array(xyz, copy=True)
                distances = np.array(distances, copy=True)
                xyz[inactive] *= (self.cutoff / distances[inactive])[:, None]
                distances[inactive] = self.cutoff
            return (
                np.ascontiguousarray(cache.receivers),
                np.ascontiguousarray(cache.sources),
                np.ascontiguousarray(distances),
                np.ascontiguousarray(xyz),
            )
        active = distances < self.cutoff
        return (
            np.ascontiguousarray(cache.receivers[active]),
            np.ascontiguousarray(cache.sources[active]),
            np.ascontiguousarray(distances[active]),
            np.ascontiguousarray(xyz[active]),
        )

    def _mace_inputs(
        self, atoms, native_geometry=False, allow_device_neighbor_graph=True
    ):
        ase_atomic_numbers = atoms.get_atomic_numbers().tolist()
        mace_atomic_numbers = self.evaluator.atomic_numbers
        unsupported = sorted(set(ase_atomic_numbers) - set(mace_atomic_numbers))
        if unsupported:
            raise ValueError(
                f"Model does not support atomic numbers {unsupported}. "
                f"Supported atomic numbers are {mace_atomic_numbers}."
            )
        neighbor_backend = _neighbor_backend_request()
        if self.neighbor_skin > 0.0 or native_geometry:
            if neighbor_backend == "kokkos" and self.neighbor_skin <= 0.0:
                raise RuntimeError(
                    "SYMMETRIX_NEIGHBOR_BACKEND=kokkos requires a positive "
                    "neighbor_skin and native prepared geometry."
                )
            i_list, j_list, r, xyz = self._cached_neighbor_geometry(
                atoms,
                native_geometry=native_geometry,
                allow_device_neighbor_graph=(
                    allow_device_neighbor_graph and self.neighbor_skin > 0.0
                ),
            )
        else:
            if neighbor_backend == "kokkos":
                raise RuntimeError(
                    "SYMMETRIX_NEIGHBOR_BACKEND=kokkos requires a positive "
                    "neighbor_skin and native prepared geometry."
                )
            i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, self.cutoff)
            self.neighbor_graph_backend = "host"
        num_nodes = len(atoms)
        cache = getattr(self, "_neighbor_cache", None)
        graph_edges = (
            cache.num_edges
            if cache is not None and cache.device_graph_generation != 0
            else len(i_list)
        )
        _validate_int32_graph_cardinality(num_nodes, graph_edges)
        native_cardinality_guard = getattr(
            self.evaluator, "_validate_graph_cardinality", None
        )
        if callable(native_cardinality_guard):
            native_cardinality_guard(num_nodes, num_nodes, graph_edges)
        if native_geometry and r is None and xyz is None:
            cache = self._neighbor_cache
            return (
                num_nodes,
                cache.node_types,
                cache.num_neigh,
                j_list,
                cache.neigh_types,
                xyz,
                r,
                i_list,
            )
        type_by_atomic_number = {
            atomic_number: index
            for index, atomic_number in enumerate(mace_atomic_numbers)
        }
        node_types = [type_by_atomic_number[value] for value in ase_atomic_numbers]
        num_neigh = np.bincount(i_list, minlength=num_nodes)
        neigh_types = [type_by_atomic_number[ase_atomic_numbers[j]] for j in j_list]
        return num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list

    def _compute_mace(self, mace_inputs, atoms=None):
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, _ = mace_inputs
        if (
            getattr(self.evaluator, "streamed_edges_mode", None) == "generic"
            and xyz is None
            and r is None
        ):
            prepare_graph = getattr(
                self.evaluator, "_prepare_all_interactions_graph", None
            )
            prepare_geometry = getattr(
                self.evaluator, "_prepare_all_interactions_geometry", None
            )
            compute_positions = getattr(
                self.evaluator, "_compute_prepared_all_interactions_positions", None
            )
            cache = self._neighbor_cache
            if (
                atoms is None
                or cache is None
                or not callable(prepare_graph)
                or not callable(prepare_geometry)
                or not callable(compute_positions)
            ):
                raise RuntimeError(
                    "Native streamed all geometry requires atoms and the "
                    "prepared-position API."
                )
            graph_identity = (id(self.evaluator), cache.generation)
            if graph_identity != self._all_interactions_graph_identity:
                graph_generation = prepare_graph(
                    num_nodes, node_types, num_neigh, j_list, neigh_types
                )
                self._sync_low_memory_policy_state()
                self._all_interactions_graph_identity = graph_identity
                self._all_interactions_graph_generation = graph_generation
            else:
                graph_generation = self._all_interactions_graph_generation
            cache = self._materialize_neighbor_cache_geometry()
            geometry_identity = (
                id(self.evaluator),
                graph_generation,
                cache.geometry_generation,
            )
            if geometry_identity != self._all_interactions_native_geometry_identity:
                prepare_geometry(
                    graph_generation,
                    np.ascontiguousarray(
                        cache.reference_positions, dtype=float
                    ).reshape(-1),
                    np.ascontiguousarray(cache.reference_xyz, dtype=float).reshape(-1),
                    np.ascontiguousarray(cache.cell, dtype=float).reshape(-1),
                    np.ascontiguousarray(cache.inverse_cell, dtype=float).reshape(-1),
                    np.ascontiguousarray(cache.pbc, dtype=np.int32).reshape(-1),
                )
                self._all_interactions_native_geometry_identity = geometry_identity
            compute_positions(
                graph_generation,
                np.ascontiguousarray(atoms.positions, dtype=float).reshape(-1),
            )
            return graph_generation

        prepare_graph = getattr(self.evaluator, "_prepare_factorized_graph", None)
        compute_prepared = getattr(self.evaluator, "_compute_prepared_factorized", None)
        if (
            getattr(self.evaluator, "streamed_edges_mode", None)
            in _PREPARED_EXECUTION_MODES
            and callable(prepare_graph)
            and callable(compute_prepared)
        ):
            native_geometry = xyz is None and r is None
            if (xyz is None) != (r is None):
                raise ValueError("MACE geometry must provide both xyz and distances.")
            if native_geometry:
                prepare_geometry = getattr(
                    self.evaluator, "_prepare_factorized_geometry", None
                )
                prepare_fractional_geometry = getattr(
                    self.evaluator,
                    "_prepare_factorized_fractional_geometry",
                    None,
                )
                update_cell = getattr(self.evaluator, "_update_factorized_cell", None)
                compute_positions = getattr(
                    self.evaluator, "_compute_prepared_factorized_positions", None
                )
                cache = self._neighbor_cache
                if atoms is None or cache is None or not callable(compute_positions):
                    raise RuntimeError(
                        "Native Execution geometry requires atoms and the prepared-position API."
                    )
                if cache.device_graph_generation != 0:
                    graph_generation = cache.device_graph_generation
                    if cache.device_geometry_generation != cache.geometry_generation:
                        if not callable(update_cell):
                            raise RuntimeError(
                                "Device-resident neighbor geometry requires the "
                                "fractional cell-update API."
                            )
                        update_cell(
                            graph_generation,
                            np.ascontiguousarray(cache.cell, dtype=float).reshape(-1),
                            np.ascontiguousarray(
                                cache.inverse_cell, dtype=float
                            ).reshape(-1),
                        )
                        cache.device_geometry_generation = cache.geometry_generation
                    compute_positions(
                        graph_generation,
                        np.ascontiguousarray(atoms.positions, dtype=float).reshape(-1),
                    )
                    return graph_generation
                graph_generation = prepare_graph(
                    num_nodes, node_types, num_neigh, j_list, neigh_types
                )
                self._sync_low_memory_policy_state()
                if self._compute_fixed_workspace_shift_geometry(
                    graph_generation, atoms, compute_positions
                ):
                    return graph_generation
                use_fractional_geometry = (
                    self._can_use_fractional_geometry(cache)
                    and callable(prepare_fractional_geometry)
                    and callable(update_cell)
                )
                if use_fractional_geometry:
                    geometry_identity = (
                        id(self.evaluator),
                        graph_generation,
                        cache.generation,
                        "fractional",
                    )
                    cell_identity = (
                        id(self.evaluator),
                        graph_generation,
                        cache.geometry_generation,
                    )
                    if geometry_identity != self._execution_native_geometry_identity:
                        prepare_fractional_geometry(
                            graph_generation,
                            np.ascontiguousarray(
                                cache.topology_fractional_positions, dtype=float
                            ).reshape(-1),
                            np.ascontiguousarray(
                                cache.topology_fractional_xyz, dtype=float
                            ).reshape(-1),
                            np.ascontiguousarray(cache.cell, dtype=float).reshape(-1),
                            np.ascontiguousarray(
                                cache.inverse_cell, dtype=float
                            ).reshape(-1),
                            np.ascontiguousarray(cache.pbc, dtype=np.int32).reshape(-1),
                        )
                        self._execution_native_geometry_identity = geometry_identity
                        self._execution_native_cell_identity = cell_identity
                    elif cell_identity != self._execution_native_cell_identity:
                        update_cell(
                            graph_generation,
                            np.ascontiguousarray(cache.cell, dtype=float).reshape(-1),
                            np.ascontiguousarray(
                                cache.inverse_cell, dtype=float
                            ).reshape(-1),
                        )
                        self._execution_native_cell_identity = cell_identity
                else:
                    if not callable(prepare_geometry):
                        raise RuntimeError(
                            "Native Execution geometry requires the Cartesian "
                            "prepared-geometry API."
                        )
                    cache = self._materialize_neighbor_cache_geometry()
                    geometry_identity = (
                        id(self.evaluator),
                        graph_generation,
                        cache.geometry_generation,
                        "cartesian",
                    )
                    if geometry_identity != self._execution_native_geometry_identity:
                        prepare_geometry(
                            graph_generation,
                            np.ascontiguousarray(
                                cache.reference_positions, dtype=float
                            ).reshape(-1),
                            np.ascontiguousarray(
                                cache.reference_xyz, dtype=float
                            ).reshape(-1),
                            np.ascontiguousarray(cache.cell, dtype=float).reshape(-1),
                            np.ascontiguousarray(
                                cache.inverse_cell, dtype=float
                            ).reshape(-1),
                            np.ascontiguousarray(cache.pbc, dtype=np.int32).reshape(-1),
                        )
                        self._execution_native_geometry_identity = geometry_identity
                        self._execution_native_cell_identity = None
                compute_positions(
                    graph_generation,
                    np.ascontiguousarray(atoms.positions, dtype=float).reshape(-1),
                )
                return graph_generation

            xyz_flat = np.ascontiguousarray(xyz, dtype=float).reshape(-1)
            distances = np.ascontiguousarray(r, dtype=float)
            if (
                not np.isfinite(xyz_flat).all()
                or not np.isfinite(distances).all()
                or not (distances > 0.0).all()
            ):
                raise ValueError(
                    "MACE graph has an invalid distance or displacement vector."
                )
            graph_generation = prepare_graph(
                num_nodes, node_types, num_neigh, j_list, neigh_types
            )
            self._sync_low_memory_policy_state()
            compute_positions = getattr(
                self.evaluator, "_compute_prepared_factorized_positions", None
            )
            if self._compute_fixed_workspace_shift_geometry(
                graph_generation, atoms, compute_positions
            ):
                return graph_generation
            compute_prepared(
                graph_generation,
                xyz_flat,
                distances,
            )
            return graph_generation
        if xyz is None or r is None:
            raise RuntimeError(
                "Native geometry is unavailable for this evaluator path."
            )
        xyz_flat = np.ascontiguousarray(xyz, dtype=float).reshape(-1)
        distances = np.ascontiguousarray(r, dtype=float)
        self.evaluator.compute_node_energies_forces(
            num_nodes,
            node_types,
            num_neigh,
            j_list,
            neigh_types,
            xyz_flat,
            distances,
        )
        return 0

    def _can_use_fractional_geometry(self, cache=None):
        if cache is None:
            cache = self._neighbor_cache
        return (
            getattr(self, "_native_model_type", "MACE") == "MACE"
            and cache is not None
            and cache.fractional_geometry_eligible
            and getattr(self.evaluator, "streamed_edges_mode", None)
            in _PREPARED_EXECUTION_MODES
            and not self._has_native_field_coupling()
            and callable(
                getattr(
                    self.evaluator,
                    "_prepare_factorized_fractional_geometry",
                    None,
                )
            )
            and callable(getattr(self.evaluator, "_update_factorized_cell", None))
        )

    def _fixed_workspace_plan_selected(self):
        native_plan = getattr(self.evaluator, "execution_plan_report", None)
        return (
            isinstance(native_plan, dict)
            and native_plan.get("selected_id") in _FIXED_WORKSPACE_TILED_PLANS
        )

    def _fixed_workspace_geometry_eligible(self):
        return self._model_single_layer_readout or self._fixed_workspace_plan_selected()

    def _compute_fixed_workspace_shift_geometry(
        self, graph_generation, atoms, compute_positions
    ):
        prepare_shift_geometry = getattr(
            self.evaluator, "_prepare_factorized_shift_geometry", None
        )
        if not self._fixed_workspace_plan_selected():
            return False
        cache = getattr(self, "_neighbor_cache", None)
        if (
            not callable(prepare_shift_geometry)
            or not callable(compute_positions)
            or atoms is None
            or cache is None
        ):
            raise RuntimeError(
                "Fixed-workspace tiled execution requires shift geometry, atoms, "
                "and a prepared neighbor cache."
            )
        geometry_identity = (
            id(self.evaluator),
            graph_generation,
            cache.geometry_generation,
            "integer-shifts",
        )
        if geometry_identity != self._execution_native_geometry_identity:
            prepare_shift_geometry(
                graph_generation,
                np.ascontiguousarray(cache.reference_positions, dtype=float).reshape(
                    -1
                ),
                np.ascontiguousarray(cache.shifts, dtype=np.int32).reshape(-1),
                np.ascontiguousarray(cache.cell, dtype=float).reshape(-1),
                np.ascontiguousarray(cache.inverse_cell, dtype=float).reshape(-1),
                np.ascontiguousarray(cache.pbc, dtype=np.int32).reshape(-1),
            )
            self._execution_native_geometry_identity = geometry_identity
            self._execution_native_cell_identity = None
        compute_positions(
            graph_generation,
            np.ascontiguousarray(atoms.positions, dtype=float).reshape(-1),
        )
        return True

    def _can_use_native_geometry(self, properties):
        if self.neighbor_skin <= 0.0 and not self._fixed_workspace_geometry_eligible():
            return False
        if "stress" in properties and not callable(
            getattr(self.evaluator, "_reduce_stress", None)
        ):
            return False
        mode = getattr(self.evaluator, "streamed_edges_mode", None)
        if self._has_native_field_coupling():
            requires_response = any(
                prop in ("becs", "polarizability") for prop in properties
            )
            if requires_response and not callable(
                getattr(
                    self.evaluator,
                    "_compute_prepared_factorized_field_response",
                    None,
                )
            ):
                return False
            return (
                mode in _PREPARED_EXECUTION_MODES
                and callable(getattr(self.evaluator, "_prepare_factorized_graph", None))
                and callable(
                    getattr(self.evaluator, "_prepare_factorized_geometry", None)
                )
                and callable(
                    getattr(
                        self.evaluator,
                        "_compute_prepared_factorized_positions_field",
                        None,
                    )
                )
            )
        if mode == "generic":
            return all(
                callable(getattr(self.evaluator, name, None))
                for name in (
                    "_prepare_all_interactions_graph",
                    "_prepare_all_interactions_geometry",
                    "_compute_prepared_all_interactions_positions",
                )
            )
        return (
            mode in _PREPARED_EXECUTION_MODES
            and callable(getattr(self.evaluator, "_prepare_factorized_graph", None))
            and callable(getattr(self.evaluator, "_prepare_factorized_geometry", None))
            and callable(
                getattr(self.evaluator, "_compute_prepared_factorized_positions", None)
            )
        )

    def _compute_macefield(self, atoms, electric_field, mace_inputs=None):
        if mace_inputs is None:
            mace_inputs = self._mace_inputs(atoms)
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list = (
            mace_inputs
        )
        field = np.ascontiguousarray(electric_field, dtype=float).reshape(-1)
        prepare_graph = getattr(self.evaluator, "_prepare_factorized_graph", None)
        compute_prepared = getattr(
            self.evaluator, "_compute_prepared_factorized_field", None
        )
        compute_positions = getattr(
            self.evaluator, "_compute_prepared_factorized_positions_field", None
        )
        self._macefield_response_graph_generation = 0
        if (
            getattr(self.evaluator, "streamed_edges_mode", None)
            in _PREPARED_EXECUTION_MODES
            and callable(prepare_graph)
            and callable(compute_prepared)
        ):
            native_geometry = xyz is None and r is None
            if (xyz is None) != (r is None):
                raise ValueError(
                    "MACEField geometry must provide both xyz and distances."
                )
            if native_geometry:
                prepare_geometry = getattr(
                    self.evaluator, "_prepare_factorized_geometry", None
                )
                cache = self._neighbor_cache
                if (
                    cache is None
                    or not callable(prepare_geometry)
                    or not callable(compute_positions)
                ):
                    raise RuntimeError(
                        "Native MACEField geometry requires the prepared-position API."
                    )
                if cache.device_graph_generation != 0:
                    graph_generation = cache.device_graph_generation
                    update_cell = getattr(
                        self.evaluator, "_update_factorized_cell", None
                    )
                    if cache.device_geometry_generation != cache.geometry_generation:
                        if not callable(update_cell):
                            raise RuntimeError(
                                "Device-resident MACEField geometry requires the "
                                "fractional cell-update API."
                            )
                        update_cell(
                            graph_generation,
                            np.ascontiguousarray(cache.cell, dtype=float).reshape(-1),
                            np.ascontiguousarray(
                                cache.inverse_cell, dtype=float
                            ).reshape(-1),
                        )
                        cache.device_geometry_generation = cache.geometry_generation
                    compute_positions(
                        graph_generation,
                        np.ascontiguousarray(atoms.positions, dtype=float).reshape(-1),
                        field,
                    )
                    self._macefield_response_graph_generation = graph_generation
                    return mace_inputs
                graph_generation = prepare_graph(
                    num_nodes, node_types, num_neigh, j_list, neigh_types
                )
                self._sync_low_memory_policy_state()
                cache = self._materialize_neighbor_cache_geometry()
                geometry_identity = (
                    id(self.evaluator),
                    graph_generation,
                    cache.geometry_generation,
                    "cartesian-field",
                )
                if geometry_identity != self._execution_native_geometry_identity:
                    prepare_geometry(
                        graph_generation,
                        np.ascontiguousarray(
                            cache.reference_positions, dtype=float
                        ).reshape(-1),
                        np.ascontiguousarray(cache.reference_xyz, dtype=float).reshape(
                            -1
                        ),
                        np.ascontiguousarray(cache.cell, dtype=float).reshape(-1),
                        np.ascontiguousarray(cache.inverse_cell, dtype=float).reshape(
                            -1
                        ),
                        np.ascontiguousarray(cache.pbc, dtype=np.int32).reshape(-1),
                    )
                    self._execution_native_geometry_identity = geometry_identity
                    self._execution_native_cell_identity = None
                compute_positions(
                    graph_generation,
                    np.ascontiguousarray(atoms.positions, dtype=float).reshape(-1),
                    field,
                )
                self._macefield_response_graph_generation = graph_generation
                return mace_inputs

            xyz_flat = np.ascontiguousarray(xyz, dtype=float).reshape(-1)
            distances = np.ascontiguousarray(r, dtype=float)
            if (
                not np.isfinite(xyz_flat).all()
                or not np.isfinite(distances).all()
                or not (distances > 0.0).all()
                or not np.isfinite(field).all()
            ):
                raise ValueError(
                    "MACEField graph has an invalid distance, displacement, or field."
                )
            graph_generation = prepare_graph(
                num_nodes, node_types, num_neigh, j_list, neigh_types
            )
            self._sync_low_memory_policy_state()
            compute_prepared(graph_generation, xyz_flat, distances, field)
            self._macefield_response_graph_generation = graph_generation
            return mace_inputs
        if xyz is None or r is None:
            raise RuntimeError(
                "Native MACEField geometry is unavailable for this evaluator path."
            )
        xyz_flat = np.ascontiguousarray(xyz, dtype=float).reshape(-1)
        distances = np.ascontiguousarray(r, dtype=float)
        self.evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            j_list,
            neigh_types,
            xyz_flat,
            distances,
            field,
        )
        return mace_inputs

    def _calculate_macefield_responses(
        self, atoms, electric_field, properties, mace_inputs
    ):
        volume = atoms.get_volume()
        raw_polarization = -np.asarray(self.evaluator.electric_field_adj, dtype=float)
        if raw_polarization.shape != (3,):
            raise PropertyNotImplementedError(
                "MACEField response properties require a graph-level electric_field with shape (3,)."
            )

        results = {"polarization": np.array(raw_polarization / volume, copy=True)}

        field_derivatives_computed = False
        force_derivatives_computed = False

        def compute_field_derivatives(include_forces=False):
            nonlocal field_derivatives_computed, force_derivatives_computed
            if include_forces and force_derivatives_computed:
                return
            if field_derivatives_computed and not include_forces:
                return
            field = np.asarray(electric_field, dtype=float).flatten()
            compute_prepared_response = getattr(
                self.evaluator,
                "_compute_prepared_factorized_field_response",
                None,
            )
            if (
                callable(compute_prepared_response)
                and self._macefield_response_graph_generation != 0
            ):
                compute_prepared_response(
                    self._macefield_response_graph_generation,
                    field,
                    include_forces,
                )
                field_derivatives_computed = True
                force_derivatives_computed = include_forces
                return

            num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, _ = (
                mace_inputs
            )
            if include_forces:
                compute_current_response = getattr(
                    self.evaluator,
                    "_compute_current_electric_field_force_derivative",
                    None,
                )
                compute_response = (
                    compute_current_response
                    if callable(compute_current_response)
                    else self.evaluator.compute_electric_field_force_derivative
                )
                response_args = (
                    num_nodes,
                    node_types,
                    num_neigh,
                    j_list,
                    neigh_types,
                    xyz.flatten(),
                    r,
                    field,
                )
                if callable(compute_current_response):
                    compute_response(
                        *response_args,
                        self._macefield_response_graph_generation,
                    )
                else:
                    compute_response(*response_args)
                field_derivatives_computed = True
                force_derivatives_computed = True
            else:
                compute_current_response = getattr(
                    self.evaluator,
                    "_compute_current_electric_field_hessian",
                    None,
                )
                compute_response = (
                    compute_current_response
                    if callable(compute_current_response)
                    else self.evaluator.compute_electric_field_hessian
                )
                response_args = (
                    num_nodes,
                    node_types,
                    num_neigh,
                    j_list,
                    neigh_types,
                    xyz.flatten(),
                    r,
                    field,
                )
                if callable(compute_current_response):
                    compute_response(
                        *response_args,
                        self._macefield_response_graph_generation,
                    )
                else:
                    compute_response(*response_args)
                field_derivatives_computed = True

        if "polarizability" in properties:
            compute_field_derivatives(include_forces="becs" in properties)
            polarizability = -np.asarray(
                self.evaluator.electric_field_hessian, dtype=float
            ).reshape(3, 3)
            results["polarizability"] = np.array(
                (polarizability / volume / self._macefield_eps0).reshape(9),
                copy=True,
            )

        if "becs" in properties:
            compute_field_derivatives(include_forces=True)
            num_nodes, _, _, j_list, _, xyz, _, i_list = mace_inputs
            pair_derivatives = np.asarray(
                self.evaluator.electric_field_force_derivative,
                dtype=float,
            ).reshape(3, -1, 3)[:, : len(i_list), :]
            becs = np.zeros((len(atoms), 3, 3))
            for field_component in range(3):
                for cartesian in range(3):
                    becs[:, field_component, cartesian] = np.bincount(
                        j_list,
                        weights=pair_derivatives[field_component, :, cartesian],
                        minlength=num_nodes,
                    ) - np.bincount(
                        i_list,
                        weights=pair_derivatives[field_component, :, cartesian],
                        minlength=num_nodes,
                    )
            results["becs"] = becs.reshape(len(atoms), 9)

        return results

    def _collect_mace_results(
        self,
        atoms,
        mace_inputs,
        properties=None,
        execution_graph_generation=0,
    ):
        if properties is None:
            properties = self.implemented_properties
        num_nodes, node_types, _, j_list, _, xyz, _, i_list = mace_inputs
        # Graph-wide observables accumulate in FP64 even when model arithmetic
        # and the evaluator's materialized outputs use FP32.
        node_energies = np.array(
            self.evaluator.node_energies, dtype=np.float64, copy=True
        )
        total_energy = np.sum(node_energies, dtype=np.float64)
        results = {
            "energy": float(total_energy),
            "free_energy": float(total_energy),
            "energies": node_energies,
        }
        if self._has_native_field_coupling():
            atomic_energies = np.asarray(self.evaluator.atomic_energies, dtype=float)
            results["node_energy"] = (
                node_energies - atomic_energies[np.asarray(node_types, dtype=int)]
            )

        reduce_atom_forces = getattr(self.evaluator, "_reduce_atom_forces", None)
        if callable(reduce_atom_forces):
            atom_forces = np.asarray(
                reduce_atom_forces(
                    num_nodes,
                    i_list,
                    j_list,
                    execution_graph_generation,
                ),
                dtype=float,
            ).reshape((num_nodes, 3))
        else:
            pair_forces = np.asarray(self.evaluator.node_forces, dtype=float).reshape(
                (-1, 3)
            )
            pair_forces = np.array(pair_forces[: len(i_list), :], copy=True)
            atom_forces = np.zeros((num_nodes, 3))
            for component in range(3):
                atom_forces[:, component] = np.bincount(
                    j_list,
                    weights=pair_forces[:, component],
                    minlength=num_nodes,
                ) - np.bincount(
                    i_list,
                    weights=pair_forces[:, component],
                    minlength=num_nodes,
                )
        results["forces"] = atom_forces
        if "stress" in properties:
            reduce_stress = getattr(self.evaluator, "_reduce_stress", None)
            if callable(reduce_stress):
                stress_tensor = np.asarray(
                    reduce_stress(
                        atoms.get_volume(),
                        np.empty(0, dtype=np.float64) if xyz is None else xyz,
                        execution_graph_generation,
                    ),
                    dtype=np.float64,
                ).reshape((3, 3))
            else:
                pair_forces = np.asarray(
                    self.evaluator.node_forces, dtype=np.float64
                ).reshape((-1, 3))
                pair_forces = np.array(pair_forces[: len(i_list), :], copy=True)
                stress_tensor = (-pair_forces.T @ xyz) / atoms.get_volume()
            results["stress"] = full_3x3_to_voigt_6_stress(stress_tensor)
        return results

    def _calculate_macefield_results(
        self, atoms, electric_field, properties, mace_inputs=None
    ):
        if mace_inputs is None:
            mace_inputs = self._mace_inputs(atoms)
        self._compute_macefield(atoms, electric_field, mace_inputs=mace_inputs)
        results = self._collect_mace_results(
            atoms,
            mace_inputs,
            properties,
            self._macefield_response_graph_generation,
        )
        if any(prop in properties for prop in self._macefield_response_properties):
            results.update(
                self._calculate_macefield_responses(
                    atoms,
                    electric_field,
                    properties,
                    mace_inputs,
                )
            )
        return results

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        evaluation_properties = list(dict.fromkeys(properties))
        stress_properties = [*evaluation_properties, "stress"]
        if (
            self.atoms.cell.rank == 3
            and "stress" not in evaluation_properties
            and self._can_use_native_geometry(stress_properties)
        ):
            evaluation_properties.append("stress")

        if self._can_use_native_geometry(evaluation_properties):
            mace_inputs = self._mace_inputs(
                self.atoms,
                native_geometry=True,
                allow_device_neighbor_graph="becs" not in evaluation_properties,
            )
        else:
            mace_inputs = self._mace_inputs(self.atoms)
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list = (
            mace_inputs
        )
        mace_inputs = (
            num_nodes,
            node_types,
            num_neigh,
            j_list,
            neigh_types,
            xyz,
            r,
            i_list,
        )
        graph_edges = (
            self._neighbor_cache.num_edges
            if self._neighbor_cache is not None
            and self._neighbor_cache.device_graph_generation != 0
            else len(j_list)
        )
        launch_policy_changed = self._apply_kernel_launch_tuning(
            num_nodes, graph_edges, evaluation_properties
        )
        if (
            self._neighbor_cache is not None
            and self._neighbor_cache.device_graph_generation != 0
            and launch_policy_changed
        ):
            # Launch-policy changes invalidate native prepared schedules. The
            # first graph supplies the edge cardinality used by the tuning key;
            # rebuild it once after applying that policy.
            self._neighbor_cache = None
            mace_inputs = self._mace_inputs(
                self.atoms,
                native_geometry=True,
                allow_device_neighbor_graph="becs" not in evaluation_properties,
            )
            (
                num_nodes,
                node_types,
                num_neigh,
                j_list,
                neigh_types,
                xyz,
                r,
                i_list,
            ) = mace_inputs
        if self._has_native_field_coupling():
            electric_field = self._resolve_electric_field()
            self.results = self._calculate_macefield_results(
                self.atoms,
                electric_field,
                evaluation_properties,
                mace_inputs=mace_inputs,
            )
            self._macefield_electric_field = np.array(electric_field, copy=True)
        else:
            graph_generation = self._compute_mace(mace_inputs, atoms=self.atoms)
            self.results = self._collect_mace_results(
                self.atoms,
                mace_inputs,
                evaluation_properties,
                graph_generation,
            )
        self._add_dispersion_results(evaluation_properties)


class FieldContributionCalculator(Calculator):
    """Return the exact finite-field contribution of a MACEField calculator.

    Additive properties are evaluated as ``Q(E) - Q(0)``. Electrical response
    properties are returned from the requested-field calculation because the
    zero-field reference is independent of the requested field.
    """

    implemented_properties = _FIELD_ADDITIVE_PROPERTIES + _FIELD_RESPONSE_PROPERTIES

    def __init__(self, field_calculator, **kwargs):
        electric_field = kwargs.pop("electric_field", None)
        Calculator.__init__(self, **kwargs)
        if not isinstance(field_calculator, Symmetrix):
            raise TypeError("field_calculator must be a Symmetrix calculator.")
        if not field_calculator._has_native_field_coupling():
            raise ValueError("field_calculator must use a field-aware MACEField model.")
        self.field_calculator = field_calculator
        if electric_field is not None:
            self.field_calculator.electric_field = electric_field
        self.implemented_properties = list(type(self).implemented_properties)
        self._last_electric_field = None
        self._zero_field_atoms = None
        self._zero_field_results = None

    @property
    def electric_field(self):
        return self.field_calculator.electric_field

    @electric_field.setter
    def electric_field(self, value):
        self.field_calculator.electric_field = value
        self.results.clear()

    def _resolve_electric_field(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        return self.field_calculator._resolve_electric_field(atoms)

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state and (
            self._last_electric_field is None
            or not equal(
                self._last_electric_field,
                self._resolve_electric_field(atoms),
                atol=tol,
            )
        ):
            state.append("info")
        return state

    @staticmethod
    def _zero_results(atoms):
        return {
            "energy": 0.0,
            "free_energy": 0.0,
            "energies": np.zeros(len(atoms)),
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }

    def _zero_reference_is_current(self, atoms):
        if self._zero_field_atoms is None or self._zero_field_results is None:
            return False
        return not compare_atoms(self._zero_field_atoms, atoms)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        electric_field = self._resolve_electric_field(self.atoms)
        needs_response = any(prop in properties for prop in _FIELD_RESPONSE_PROPERTIES)
        needs_additive = any(prop in properties for prop in _FIELD_ADDITIVE_PROPERTIES)
        is_zero_field = np.array_equal(electric_field, np.zeros(3))
        mace_inputs = None

        if is_zero_field:
            results = self._zero_results(self.atoms)
            if needs_response:
                mace_inputs = self.field_calculator._mace_inputs(self.atoms)
                field_results = self.field_calculator._calculate_macefield_results(
                    self.atoms,
                    electric_field,
                    properties,
                    mace_inputs=mace_inputs,
                )
                for prop in _FIELD_RESPONSE_PROPERTIES:
                    if prop in field_results:
                        results[prop] = np.array(field_results[prop], copy=True)
        elif needs_additive:
            mace_inputs = self.field_calculator._mace_inputs(self.atoms)
            additive_properties = list(
                dict.fromkeys([*_FIELD_ADDITIVE_PROPERTIES, *properties])
            )
            if not self._zero_reference_is_current(self.atoms):
                self._zero_field_results = (
                    self.field_calculator._calculate_macefield_results(
                        self.atoms,
                        np.zeros(3),
                        additive_properties,
                        mace_inputs=mace_inputs,
                    )
                )
                self._zero_field_atoms = self.atoms.copy()

            field_results = self.field_calculator._calculate_macefield_results(
                self.atoms,
                electric_field,
                additive_properties,
                mace_inputs=mace_inputs,
            )
            results = {
                prop: np.asarray(field_results[prop])
                - np.asarray(self._zero_field_results[prop])
                for prop in _FIELD_ADDITIVE_PROPERTIES
            }
            results["energy"] = float(results["energy"])
            results["free_energy"] = float(results["free_energy"])
            for prop in _FIELD_RESPONSE_PROPERTIES:
                if prop in field_results:
                    results[prop] = np.array(field_results[prop], copy=True)
        else:
            mace_inputs = self.field_calculator._mace_inputs(self.atoms)
            field_results = self.field_calculator._calculate_macefield_results(
                self.atoms,
                electric_field,
                properties,
                mace_inputs=mace_inputs,
            )
            results = {
                prop: np.array(field_results[prop], copy=True)
                for prop in _FIELD_RESPONSE_PROPERTIES
                if prop in field_results
            }

        self.results = results
        self._last_electric_field = np.array(electric_field, copy=True)
        self.field_calculator.results.clear()


class FieldAwareCalculator(Calculator):
    """Add an exact MACEField contribution to an arbitrary ASE calculator."""

    def __init__(self, base_calculator, field_calculator, **kwargs):
        electric_field = kwargs.pop("electric_field", _FIELD_NOT_SET)
        Calculator.__init__(self, **kwargs)
        self.base_calculator = base_calculator
        if isinstance(field_calculator, FieldContributionCalculator):
            self.field_contribution = field_calculator
        else:
            self.field_contribution = FieldContributionCalculator(field_calculator)
        self.field_calculator = self.field_contribution.field_calculator
        if electric_field is not _FIELD_NOT_SET:
            self.field_contribution.electric_field = electric_field

        additive = [
            prop
            for prop in _FIELD_ADDITIVE_PROPERTIES
            if prop in self.base_calculator.implemented_properties
            and prop in self.field_contribution.implemented_properties
        ]
        responses = [
            prop
            for prop in _FIELD_RESPONSE_PROPERTIES
            if prop in self.field_contribution.implemented_properties
        ]
        self.implemented_properties = additive + responses
        self._last_electric_field = None
        self._base_properties = set()

    @property
    def electric_field(self):
        return self.field_contribution.electric_field

    @electric_field.setter
    def electric_field(self, value):
        self.field_contribution.electric_field = value
        self.results.clear()

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state and self._base_properties:
            base_state = self.base_calculator.check_state(atoms, tol=tol)
            base_results_missing = any(
                prop not in self.base_calculator.results
                for prop in self._base_properties
            )
            if base_state or base_results_missing:
                state.append("calculator")
        if not state and (
            self._last_electric_field is None
            or not equal(
                self._last_electric_field,
                self.field_contribution._resolve_electric_field(atoms),
                atol=tol,
            )
        ):
            state.append("info")
        return state

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        results = {}
        base_properties = set()
        base_atoms = self.atoms.copy()
        base_atoms.calc = None
        for prop in properties:
            field_value = self.field_contribution.get_property(prop, self.atoms)
            if prop in _FIELD_RESPONSE_PROPERTIES:
                results[prop] = field_value
            else:
                base_value = self.base_calculator.get_property(prop, base_atoms)
                if prop == "stress":
                    results[prop] = _to_voigt_stress(base_value) + _to_voigt_stress(
                        field_value
                    )
                else:
                    results[prop] = base_value + field_value
                base_properties.add(prop)
        self.results = results
        self._base_properties = base_properties
        self._last_electric_field = self.field_contribution._resolve_electric_field(
            self.atoms
        ).copy()
