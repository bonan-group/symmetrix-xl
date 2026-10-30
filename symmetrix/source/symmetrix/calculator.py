"""ASE Calculator for symmetrix implementation of equivariant graph neural
network library

This file was written and publicly released by Dr. Noam Bernstein as part of his
work for the U. S. Government, and is not subject to copyright.
"""

import json
import logging
import operator
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
    compare_atoms,
    equal,
)
from ase.stress import full_3x3_to_voigt_6_stress

from . import family_registry, neighbor_graph, symmetrix
from .corrections import CorrectionPipeline, D3Correction
from .mace_alchemical import (
    expand_alchemical_topology,
    normalize_alchemical,
    resolve_alchemical_config,
    validate_alchemical_model,
)
from .neighbor_graph import (
    NeighborTopologyReference,
    build_host_neighbor_graph,
    minimum_image_displacements,
    neighbor_topology_reuse,
    validate_graph_cardinality,
)

_LOGGER = logging.getLogger(__name__)
_FIELD_ADDITIVE_PROPERTIES = ["energy", "free_energy", "energies", "forces", "stress"]
_FIELD_RESPONSE_PROPERTIES = ["polarization", "becs", "polarizability"]
_FIELD_NOT_SET = object()
_LOW_MEMORY_UNSET = object()
_POLAR_OPTION_UNSET = object()


def _positive_runtime_integer(value, option):
    """Normalize an integer-valued runtime policy without lossy coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{option} must be a positive integer")
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{option} must be a positive integer") from error
    if normalized <= 0:
        raise ValueError(f"{option} must be a positive integer")
    return normalized


# ``streamed_edges`` distinguishes an unset default from an explicit
# ``"direct"``: the documented default is ``"direct"``, but a family that
# declares no streamed-edge mode handshake resolves the unset default through
# its declared routings the way ``"auto"`` does, while an explicit request is
# refused by name when the family cannot serve it.
_STREAMED_EDGES_UNSET = object()
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
# The plan ids the MH-0 family declares as its native candidate space.  The
# native debug-plan pin validates a requested id against that same declaration
# and the report lists the declared space, so this Python set is the frontend
# half of one cross-language contract (see
# ``test_declared_plan_space_is_the_cross_language_plan_id_contract``).
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
# Every documented streamed-edge spelling this frontend accepts resolves through
# the execution-mode vocabulary the family registry declares: the values are the
# canonical cross-language strings the native parser accepts plus the
# Python-only ``auto`` selector, and that one declaration is the single
# acceptance source, so a mode is added or retired there and nowhere else.
_MODE_VOCABULARY = family_registry.COMPACT_MODE_VOCABULARY
_PREPARED_EXECUTION_MODES = frozenset(("direct",))
_FIXED_WORKSPACE_TILED_PLANS = frozenset(
    ("mh0-single-layer-tiled-v1", "mh0-dual-layer-tiled-v1")
)
_NEIGHBOR_BACKENDS = frozenset(("automatic", "host", "kokkos"))
_AUTOMATIC_KOKKOS_NEIGHBOR_MIN_ATOMS = 512
_AUTOMATIC_KOKKOS_NEIGHBOR_EXECUTION_SPACES = frozenset(("Cuda", "HIP"))


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


def _factorized_device_backend_plan(
    backend,
    target,
    contract,
    dtype,
    evaluator,
    *,
    apply_edge_weights=False,
    resolve_compiler=True,
):
    if backend == "cuda":
        from .jit import (
            artifact_cache_inputs,
            prepare_execution_cuda_jit_artifact,
            prepare_nvrtc_jit_artifact,
            select_execution_cuda_jit_backend,
            execution_cuda_jit_backend_request,
            execution_cuda_r1_edge_launch_policy,
        )
        from .mace_jit_codegen import (
            R1_ARTIFACT_ENVELOPE,
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
        policy = (
            execution_cuda_jit_backend_request() if resolve_compiler else "deferred"
        )
        selected_compiler = (
            select_execution_cuda_jit_backend(policy) if resolve_compiler else None
        )
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
            apply_edge_weights=apply_edge_weights,
        )
        edge_policy = {
            "strategy": metadata["edge_strategy"],
            "logical_subgroup_width": metadata["edge_logical_subgroup_width"],
            "threads_per_block": metadata["edge_threads_per_block"],
            "persistent_blocks_per_compute_unit": metadata[
                "persistent_blocks_per_compute_unit"
            ],
        }
        abi = artifact_cache_inputs(metadata, R1_ARTIFACT_ENVELOPE)
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
                apply_edge_weights=apply_edge_weights,
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
                    "abi": abi,
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
            artifact_cache_inputs,
            normalize_execution_hip_target_identity,
            prepare_execution_hip_jit_artifact,
            prepare_hiprtc_jit_artifact,
            select_execution_hip_jit_backend,
            execution_hip_jit_backend_request,
            execution_hip_r1_edge_launch_policy,
        )
        from .mace_jit_codegen import (
            R1_ARTIFACT_ENVELOPE,
            render_jit_r1_hip_module,
            render_jit_r1_hip_plugin,
            factorized_gpu_codegen_identity,
            factorized_hip_plugin_metadata,
        )

        policy = execution_hip_jit_backend_request() if resolve_compiler else "deferred"
        selected_compiler = (
            select_execution_hip_jit_backend(policy) if resolve_compiler else None
        )
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
            apply_edge_weights=apply_edge_weights,
        )
        edge_policy = {
            "strategy": metadata["edge_strategy"],
            "logical_subgroup_width": metadata["edge_logical_subgroup_width"],
            "threads_per_block": metadata["edge_threads_per_block"],
            "persistent_blocks_per_compute_unit": metadata[
                "persistent_blocks_per_compute_unit"
            ],
        }
        abi = artifact_cache_inputs(metadata, R1_ARTIFACT_ENVELOPE)
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
                apply_edge_weights=apply_edge_weights,
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
                    "abi": abi,
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


def _factorized_device_installed_plan(
    backend, target, contract, dtype, evaluator, *, apply_edge_weights=False
):
    """Derive exact device R1 identity without selecting or probing a compiler."""

    return _factorized_device_backend_plan(
        backend,
        target,
        contract,
        dtype,
        evaluator,
        apply_edge_weights=apply_edge_weights,
        resolve_compiler=False,
    )


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
        explicitly for high-precision calculations.
    streamed_edges : {"direct", "reference"}, default="direct"
        Execution routing for compact MACE models; ``auto`` resolves from model
        capability, and every documented spelling resolves through one explicit
        alias table onto the family's two routings. ``direct`` runs the
        prepared factorized plan and requires a model-specific RTC artifact,
        which does not fall back. ``reference`` runs the generic streamed
        implementation through the whole-evaluator entry point on the default
        host execution space with no artifact at all. Admitted single-layer
        models use built-in direct M0/R0 execution without an R1 stage,
        although capacity planning may still specialize those operators.
        The ``non-compiled``, ``generic``, ``all_interactions`` and
        ``materialized`` spellings resolve to ``reference``; ``generic`` and
        ``all_interactions`` are deprecated. ``factorized`` and
        ``direct_streamed`` alias ``direct`` and preserve the former throughput
        behavior by changing an otherwise-``capacity`` request to the ``speed``
        profile. Use literal ``direct`` for capacity behavior. A family whose
        evaluator declares no streamed-edge mode handshake (the MACE Polar
        family, whose models do not enter the ordinary streamed-edge stack)
        still declares ``direct`` and ``reference`` routes. Both call the
        family's native ``whole_model`` surface; ``direct`` supplies a
        contract-matched generated backbone artifact and ``reference`` uses
        the artifact-free portable backbone. This routing is distinct from
        ``polar_electrostatics``, which is runtime boundary policy and is not
        serialized in the model JSON.
    execution_profile : {"capacity", "speed"}, default="capacity"
        Resource objective for direct Kokkos execution. ``capacity`` chooses
        the fastest qualified direct policy estimated to fit the device; when
        all estimates exceed the advisory budget it attempts the smallest
        qualified policy and lets allocation determine feasibility. ``speed``
        always uses the retained throughput policy. The current native report
        is available as :attr:`execution_plan` after graph preparation.
    allow_fixed_workspace : bool, default=False
        Permit capacity planning to select bounded tiled workspace execution.
        This can increase the maximum graph size for qualified ordinary MH0
        standard MACE models on CUDA, but must be enabled explicitly. MACEField
        and MACE-MH-1 do not support fixed-workspace plans. The option does not
        force a tiled plan.
    corrections : sequence, optional
        Add native corrections such as :class:`symmetrix.D3Correction` through
        the loaded Kokkos backend.
    dispersion : bool, default=False
        Deprecated compatibility spelling for one native D3 correction. Use
        ``corrections=[D3Correction(...)]`` instead.
    dispersion_damping : str, default="bj"
        D3 damping function (``zero``, ``bj``, ``zerom``, or ``bjm``).
    dispersion_xc : str, default="pbe"
        Exchange-correlation functional parameterization for D3.
    dispersion_cutoff : float, optional
        D3 cutoff in Angstrom. Defaults to 40 Bohr converted to Angstrom.
    dispersion_device : str, optional
        Deprecated compatibility check. Correction backend selection follows
        the loaded Symmetrix backend.
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
        policy bundle. Throughput-qualified direct models apply the
        ``reuse-adjoints-v1`` storage policy under both execution profiles;
        enabling execution observers or parameter gradients forces
        ``full-retention-v1``.
    edge_geometry_policy : {"cartesian-f64-v1", "unit-f32-radius-f64-v1"}, default="cartesian-f64-v1"
        Active edge-geometry representation for ordinary factorized inference.
        ``unit-f32-radius-f64-v1`` retains float32 unit directions and float64
        radii. It is an opt-in standard MH-0 capacity mode.
    neighbor_skin : float, default=0.5
        Verlet-list skin in Angstrom. Candidate neighbors are built at the
        model cutoff plus this skin and reused until any atom has moved by
        half the skin. With ``streamed_edges="non-compiled"``, current
        exact-cutoff members are compacted from those candidates in native
        Kokkos code.
        With ``streamed_edges="direct"``, inactive candidates are retained at
        the exact compact-radial cutoff, where their radial contribution is
        zero, so the prepared schedule remains stable. Set to zero to rebuild
        the exact neighbor list every call.
    polar_electrostatics : {"auto", "realspace", "periodic", "direct_reciprocal", "pme"}, default="auto"
        Runtime POLAR electrostatics policy. Fully non-periodic input resolves
        ``auto`` to real space; any periodic axis resolves it to the Delivery 1
        direct reciprocal implementation. ``periodic`` is the compatibility
        override for molecule-in-box evaluation. Forced ``pme`` is available
        for fully three-dimensional periodic input on qualified CUDA builds
        with an FFT provider; other backends and partial periodicity remain on
        direct reciprocal space. This setting is evaluator policy and is never
        serialized in the model JSON.
    polar_pme_mesh : sequence of 3 int, default=(64, 64, 64)
        Runtime PME mesh dimensions. This evaluator policy is intentionally
        separate from the solver-neutral model artifact.
    polar_pme_order : int, default=6
        Runtime cardinal B-spline interpolation order for forced PME.
    head : str or None, default=None
        Prediction head selected from a multi-head Symmetrix JSON or MACE
        checkpoint. ``None`` uses the JSON's declared default head.
    total_charge, total_spin : float or None, default=None
        The PolarMACE family's graph-level charge and spin targets: the
        values its closed-form Fukui equilibration normalizes the density
        against. ``None`` takes the structure's own ``atoms.info`` entry of the
        same name, then zero. Explicit constructor or post-construction values
        are rejected for a family whose program does not carry this channel.
    fermi_level : float or None, default=None
        The PolarMACE family's graph-level fermi level. ``None`` takes the
        structure's own ``atoms.info["fermi_level"]``, then zero.
    electric_field : array or None, default=None
        Graph-level electric field with shape ``(3,)`` or ``(1, 3)``. It is the
        MACEField coupling's field and the PolarMACE family's external
        field-projection input. ``None`` takes the structure's own
        ``atoms.info["electric_field"]``, then a zero field. Explicit values
        are rejected for families other than MACEField and MACE Polar.

    Attributes
    ----------
    jit_status : str
        Specialization outcome: ``"unresolved"``, ``"loaded"``, ``"built"``,
        ``"cached"``, ``"failed"``, or ``"not_applicable"``. Specialization
        failures raise during construction.
    jit_origin : str or None
        Storage origin of an active generated artifact: ``"prebuilt"``,
        ``"cache"``, or ``"runtime"``. This is not part of artifact identity.
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
    electric_field : array or None
        The graph-level electric field the evaluation was told to use: the
        MACEField coupling's field, and the PolarMACE family's external
        field-projection input. ``None`` means the structure's own
        ``atoms.info["electric_field"]`` (or a zero field) is used.

    Notes
    -----
    The PolarMACE family adapter resolves graph-level scalars before its one
    native whole-model transaction, and a changed value re-evaluates the
    structure. Its dipole output is the family program's ``dipole_readout``
    node and is advertised from the compiled program, so the frontend's output
    surface cannot drift from the program the evaluator carries.

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
    #: Compatibility tombstone for callers checking that the retired Python
    #: scientific-program branch is not configured in production.
    _polar_ops = None
    _polar_contract = None
    _polar_shapes = None
    _polar_composition = None
    _macefield_response_properties = list(_FIELD_RESPONSE_PROPERTIES)
    _macefield_eps0 = 8.8541878128e-12 / 1.602176634e-19 / 1e10

    def __init__(
        self,
        model_file,
        dtype="float32",
        *legacy_positional,
        streamed_edges=_STREAMED_EDGES_UNSET,
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
        polar_electrostatics=_POLAR_OPTION_UNSET,
        polar_pme_mesh=_POLAR_OPTION_UNSET,
        polar_pme_order=_POLAR_OPTION_UNSET,
        head=None,
        total_charge=_POLAR_OPTION_UNSET,
        total_spin=_POLAR_OPTION_UNSET,
        fermi_level=_POLAR_OPTION_UNSET,
        corrections=None,
        dispersion=False,
        dispersion_damping="bj",
        dispersion_xc="pbe",
        dispersion_cutoff=None,
        dispersion_device=None,
        metal=False,
        alchemical=None,
        **kwargs,
    ):
        explicit_polar_options = {
            name
            for name, value in (
                ("polar_electrostatics", polar_electrostatics),
                ("polar_pme_mesh", polar_pme_mesh),
                ("polar_pme_order", polar_pme_order),
                ("total_charge", total_charge),
                ("total_spin", total_spin),
                ("fermi_level", fermi_level),
            )
            if value is not _POLAR_OPTION_UNSET
        }
        if "electric_field" in kwargs:
            explicit_polar_options.add("electric_field")
        if polar_electrostatics is _POLAR_OPTION_UNSET:
            polar_electrostatics = "auto"
        if polar_pme_mesh is _POLAR_OPTION_UNSET:
            polar_pme_mesh = (64, 64, 64)
        if polar_pme_order is _POLAR_OPTION_UNSET:
            polar_pme_order = 6
        if total_charge is _POLAR_OPTION_UNSET:
            total_charge = None
        if total_spin is _POLAR_OPTION_UNSET:
            total_spin = None
        if fermi_level is _POLAR_OPTION_UNSET:
            fermi_level = None
        if polar_electrostatics not in {
            "auto",
            "realspace",
            "periodic",
            "direct_reciprocal",
            "pme",
        }:
            raise ValueError(
                "polar_electrostatics must be one of 'auto', 'realspace', "
                "'periodic', 'direct_reciprocal', or 'pme'."
            )
        try:
            polar_pme_mesh = tuple(
                _positive_runtime_integer(value, "polar_pme_mesh extent")
                for value in polar_pme_mesh
            )
        except TypeError as error:
            raise ValueError("polar_pme_mesh must contain three integers") from error
        if len(polar_pme_mesh) != 3:
            raise ValueError("polar_pme_mesh must contain three positive integers")
        polar_pme_order = _positive_runtime_integer(polar_pme_order, "polar_pme_order")
        # ``use_kokkos`` was the third positional argument and is retired: there
        # is one evaluator stack, and the routing is expressed through
        # ``streamed_edges``.  Both legacy forms are rejected explicitly rather
        # than silently binding the third positional slot to ``streamed_edges``.
        if legacy_positional or "use_kokkos" in kwargs:
            raise TypeError(
                "use_kokkos has been retired: Symmetrix always executes through "
                "the single Kokkos evaluator stack. Pass "
                "streamed_edges='direct' for the generated (T3) routing or "
                "streamed_edges='reference' for single-thread generic execution "
                "on the default host execution space."
            )
        if alchemical is not None:
            alchemical = normalize_alchemical(alchemical)
            if metal:
                raise ValueError("Alchemical execution does not support Metal.")
            if dispersion:
                raise ValueError(
                    "Alchemical execution does not support dispersion corrections."
                )
            if allow_fixed_workspace:
                raise ValueError(
                    "Alchemical execution does not support fixed workspace plans."
                )
        self._alchemical = alchemical
        self._alchemical_evaluated_version = None
        self._alchemical_topology = None
        self._alchemical_topology_generation = None
        self._alchemical_native_weight_identity = None
        self.metal_request = bool(metal)
        self.metal_status = "disabled"
        self.metal_device = None
        self.metal_stages = ()
        if self.metal_request:
            # Fail before any model or kernel setup: Metal is an opt-in Apple
            # GPU backend, and CPU and CUDA selection does not involve it.
            if sys.platform != "darwin":
                raise ValueError(
                    "metal=True is only available on macOS with an Apple GPU; "
                    "on Linux use the CPU or CUDA backend (selected with "
                    "SYMMETRIX_BACKEND) and leave metal unset"
                )
            # Argument errors come before build and device checks.
            if dtype != "float32":
                raise ValueError(
                    "metal=True requires dtype='float32'; Metal has no FP64"
                )
            # Fixed-workspace tiled plans process edges in chunks that the
            # Metal stages do not execute; they are a CUDA capacity feature.
            if (
                allow_fixed_workspace
                or _debug_execution_plan in _FIXED_WORKSPACE_TILED_PLANS
            ):
                raise ValueError(
                    "metal=True does not support fixed-workspace tiled plans; "
                    "leave allow_fixed_workspace unset"
                )
            if not getattr(symmetrix, "_metal_supported", lambda: False)():
                raise RuntimeError(
                    "metal=True requires a build with the Metal layer "
                    "(SYMMETRIX_METAL=ON); see docs/user/metal.md"
                )
            ready, reason = getattr(
                symmetrix, "_metal_device_ready", lambda: (True, "")
            )()
            if not ready:
                raise RuntimeError(f"metal=True has no usable Metal GPU: {reason}")
        explicit_corrections = corrections is not None
        correction_configurations = () if corrections is None else tuple(corrections)
        if alchemical is not None and correction_configurations:
            raise ValueError(
                "Alchemical execution does not support additive corrections."
            )
        if dispersion and explicit_corrections:
            raise ValueError(
                "corrections cannot be combined with legacy dispersion=True"
            )
        if dispersion:
            execution_space = getattr(
                symmetrix, "_kokkos_default_execution_space", lambda: None
            )()
            expected_device = {
                "Cuda": "cuda",
                "HIP": "hip",
            }.get(execution_space, "cpu")
            if dispersion_device is not None and dispersion_device != expected_device:
                raise ValueError(
                    "dispersion_device no longer controls backend selection; "
                    "native corrections follow the loaded Symmetrix backend "
                    f"({expected_device!r})"
                )
            warnings.warn(
                "dispersion=True is deprecated; use "
                "corrections=[D3Correction(...)] instead",
                FutureWarning,
                stacklevel=2,
            )
            correction_arguments = {
                "functional": dispersion_xc,
                "damping": dispersion_damping,
            }
            if dispersion_cutoff is not None:
                correction_arguments["cutoff"] = dispersion_cutoff
                correction_arguments["coordination_cutoff"] = min(
                    dispersion_cutoff, 40.0 * units.Bohr
                )
            correction_configurations = (D3Correction(**correction_arguments),)
        self._correction_configurations = correction_configurations
        self.dispersion = bool(dispersion)
        Calculator.__init__(self, **kwargs)
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Supported dtypes are 'float64' and 'float32'."
            )
        requested_streamed_edges = (
            "direct" if streamed_edges is _STREAMED_EDGES_UNSET else streamed_edges
        )
        streamed_edges_unset = streamed_edges is _STREAMED_EDGES_UNSET
        if requested_streamed_edges in _MODE_VOCABULARY.deprecated:
            warnings.warn(
                "streamed_edges='generic' is deprecated; use "
                "streamed_edges='non-compiled' for the compiler-free debug path. "
                "This path is not optimized and has no performance guarantee.",
                DeprecationWarning,
                stacklevel=2,
            )
        # An unknown spelling passes through unchanged so the constructor
        # reports the documented mode list instead of a table miss.
        canonical_streamed_edges = _MODE_VOCABULARY.canonical_mode(
            requested_streamed_edges
        )
        legacy_streamed_alias = (
            requested_streamed_edges in _MODE_VOCABULARY.alias_spellings
        )
        if canonical_streamed_edges not in _MODE_VOCABULARY.canonical_modes:
            raise ValueError(
                "streamed_edges must be one of "
                "'auto', 'materialized', 'non-compiled', or 'direct' "
                "(compatibility aliases: "
                "'generic', 'all_interactions', 'factorized', 'direct_streamed')."
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
            if canonical_streamed_edges != "direct":
                raise ValueError(
                    "allow_fixed_workspace=True requires streamed_edges='direct'."
                )
            if execution_profile != "capacity":
                raise ValueError(
                    "allow_fixed_workspace=True requires execution_profile='capacity'."
                )
        if legacy_low_memory_requested:
            if canonical_streamed_edges != "direct":
                raise ValueError("low_memory=True requires streamed_edges='direct'.")
        if alchemical is not None and canonical_streamed_edges != "direct":
            raise ValueError("Alchemical execution requires streamed_edges='direct'.")
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
        self.jit_origin = None
        self.jit_cache_key = None
        self.jit_artifact_path = None
        self.jit_artifact_id = None
        self.jit_compiler_backend = None
        self.jit_compiler_request = None
        self.jit_diagnostics = ()
        self.jit_reason = None
        self._direct_artifact_diagnostic = None
        self.jit_forward_policy = None
        self.jit_source_policy = None
        self.jit_edge_policy = None
        self.jit_node_state_policy = None
        self.jit_variant_id = None
        self.jit_operator_modules = {}
        self._macefield_electric_field = None
        self._macefield_response_graph_generation = 0
        self._electric_field = kwargs.get("electric_field", None)
        # The PolarMACE family's graph-level scalar channel: the total charge
        # and spin the Fukui equilibration targets, the fermi level, and the
        # external electric field the field-projection node couples.  They are
        # resolved per evaluation exactly as the MACEField field is (explicit
        # value, then the atoms' own info, then the neutral default), and the
        # last resolution is retained so a changed input invalidates the
        # cached results.
        self._total_charge = total_charge
        self._total_spin = total_spin
        self._fermi_level = fermi_level
        self._polar_graph_inputs = None
        self._explicit_polar_options = frozenset(explicit_polar_options)
        self.polar_electrostatics = polar_electrostatics
        self.polar_pme_mesh = polar_pme_mesh
        self.polar_pme_order = polar_pme_order
        self.polar_electrostatics_diagnostics = {
            "requested_solver": polar_electrostatics,
            "resolved_solver": "unresolved",
            "solver_reason": "not_evaluated",
            "automatic_atom_count": 0,
            "automatic_reciprocal_mode_count": 0,
            "automatic_reciprocal_mode_count_is_lower_bound": False,
            "automatic_atom_mode_pair_count": 0,
            "automatic_atom_mode_pair_threshold": 40_000_000,
            "n_k": 0,
            "reciprocal_allocation_elements": 0,
            "reciprocal_allocation_bytes": 0,
            "reciprocal_table_elements": 0,
            "reciprocal_table_bytes": 0,
            "particle_mesh_active_bytes": 0,
            "particle_mesh_capacity_bytes": 0,
            "particle_mesh_volume": 0,
            "particle_mesh_shape": [0, 0, 0],
            "particle_mesh_rebuilds": 0,
            "particle_mesh_reuses": 0,
            "particle_mesh_forward_plan_builds": 0,
            "particle_mesh_forward_plan_reuses": 0,
            "particle_mesh_inverse_plan_builds": 0,
            "particle_mesh_inverse_plan_reuses": 0,
        }
        identity_model_data = self._read_json_model(model_file)
        json_metadata = self._json_metadata(model_file)
        if alchemical is not None and json_metadata is not None:
            validate_alchemical_model(self._read_json_model(model_file))
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

        if not hasattr(symmetrix, "MACEKokkos"):
            raise RuntimeError("Symmetrix was built without Kokkos support.")
        json_model_type = (
            json_metadata.get("model_type", "MACE")
            if json_metadata is not None
            else None
        )
        self._native_model_type = (
            json_model_type if json_model_type is not None else "MACE"
        )
        self._require_metal_model_family()
        MACE = self._native_evaluator_class(
            json_model_type if json_model_type is not None else "MACE",
            dtype,
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
            identity_model_data = data
            if alchemical is not None:
                validate_alchemical_model(data)
            jit_model_data = data
            self._native_model_type = data.get("model_type", "MACE")
            self._model_single_layer_readout = bool(
                data.get("single_layer_readout", False)
            )
            MACE = self._native_evaluator_class(data.get("model_type", "MACE"), dtype)
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
        if identity_model_data is not None:
            from .prediction_heads import select_prediction_head

            identity_model_data = select_prediction_head(identity_model_data, self.head)

        from .mace_jit_artifact_common import mace_les_evaluator_identity

        model_identity = mace_les_evaluator_identity(
            {} if identity_model_data is None else identity_model_data
        )
        self.model_type = model_identity["model_type"]
        self.mace_les_parameters_sha256 = model_identity["mace_les_parameters_sha256"]
        self.mace_les_evaluator_fingerprint = model_identity[
            "mace_les_evaluator_fingerprint"
        ]
        self._is_mace_les = bool(getattr(self.evaluator, "is_mace_les", False))
        if self._is_mace_les != (self.model_type == "MACELES"):
            raise RuntimeError(
                "native evaluator MACELES identity does not match the selected model"
            )
        self._require_metal_model_family()
        # The family entry is resolved once here, so the prepared-path
        # capability gate on the execution path is an attribute read.
        self._family_entry = family_registry.declared_registry().detect(
            model_type=self._native_model_type
        )
        self._family_frontend_adapter = (
            None
            if self._family_entry is None
            else family_registry.frontend_adapter_of(self._family_entry)
        )
        unsupported_family_options = set(self._explicit_polar_options)
        if self._has_native_field_coupling():
            unsupported_family_options.discard("electric_field")
        if self._family_frontend_adapter is None and unsupported_family_options:
            names = ", ".join(sorted(unsupported_family_options))
            raise ValueError(
                f"Polar-only MACE_Polar option(s) {names} were supplied to "
                f"model family {self._native_model_type!r}"
            )
        if (
            self.low_memory_request
            and not legacy_low_memory_requested
            and not family_registry.supports_surface(
                self._family_entry, self.evaluator, "mode_handshake"
            )
        ):
            # The implied capacity plan is the streamed-edge families' bundle
            # (MH-0's state and edge-geometry policies, MH-1's retained
            # interaction state), selected by the direct default.  A family
            # that declares no mode handshake states its own routing set, so no
            # plan is implied for it; an explicit low_memory request is still
            # refused below.
            self.low_memory_request = False
            self.low_memory_selection_reason = (
                "implied low-memory plan unavailable for family "
                f"{self._family_entry.key!r}; using default execution policies"
            )

        if alchemical is not None:
            # The weight-aware packet surface is the family's own declaration,
            # so a backend that lacks it fails naming the missing method rather
            # than evaluating through an unweighted path.
            family_registry.require_surface(
                self._family_entry, self.evaluator, "alchemical"
            )

        if not bool(
            getattr(self.evaluator, "supports_streamed_edges", True)
        ) and family_registry.supports_surface(
            self._family_entry, self.evaluator, "mode_handshake"
        ):
            # The original Symmetrix pair-spline format (named v1 here) was only
            # ever served by the retired materialized edge representation.  Both
            # routings of the families that declare the streamed-edge handshake
            # stream edges now, so the format is refused with the conversion
            # that enables them instead of a silent downgrade.  A family that
            # declares no handshake (the PolarMACE family, whose reference
            # routing is the whole-evaluator entry point) states its own
            # routing set instead of being refused as a retired format.
            raise ValueError(
                "the original Symmetrix pair-spline format (named v1 here) is no "
                "longer supported: re-export the model with "
                "radial_format='compact' to enable streamed execution."
            )

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
            "MACELES",
            "MACEField",
            "MACE_Nonlinear",
        ):
            raise ValueError(
                "low_memory=True currently supports MACE, MACELES, MACEField, "
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

        if family_registry.supports_surface(
            self._family_entry, self.evaluator, "mode_handshake"
        ):
            if canonical_streamed_edges == "auto":
                if bool(getattr(self.evaluator, "supports_factorized", False)):
                    canonical_streamed_edges = "direct"
                    self.streamed_edges_resolution_reason = (
                        "auto selected direct because the evaluator admits "
                        "generated execution"
                    )
                else:
                    canonical_streamed_edges = "reference"
                    self.streamed_edges_resolution_reason = (
                        "auto selected reference because generated execution is "
                        "unavailable"
                    )
            routing = family_registry.routing_of(
                self._family_entry, canonical_streamed_edges
            )
            if routing is None:
                raise ValueError(
                    f"family {self._family_entry.key!r} serves no "
                    f"streamed_edges={canonical_streamed_edges!r} mode through "
                    "the compact streamed-edge handshake; its declared routings "
                    "are "
                    + ", ".join(routing.key for routing in self._family_entry.routings)
                    + "."
                )
            self.evaluator.set_streamed_edges(routing.native_mode)
        else:
            # A family that declares no streamed-edge handshake still declares
            # both routings. There is no mode to hand to the evaluator: the
            # routing selects whether the frontend loads its generated
            # subprogram, while the evaluator keeps its native portable plan.
            selected_routing = (
                "reference"
                if canonical_streamed_edges == "auto" or streamed_edges_unset
                else canonical_streamed_edges
            )
            routing = family_registry.routing_of(self._family_entry, selected_routing)
            if routing is None:
                raise ValueError(
                    f"family {self._family_entry.key!r} serves no "
                    f"streamed_edges={canonical_streamed_edges!r} routing; its "
                    "declared routings are "
                    + ", ".join(route.key for route in self._family_entry.routings)
                    + "."
                )
            if canonical_streamed_edges == "auto" or streamed_edges_unset:
                self.streamed_edges_resolution_reason = (
                    "the unset default selected reference because the family "
                    "declares no streamed-edge mode handshake"
                    if streamed_edges_unset
                    else "auto selected reference because the family declares no "
                    "streamed-edge mode handshake"
                )
            elif selected_routing == "direct":
                self.streamed_edges_resolution_reason = (
                    "explicit direct selected the family's generated subprogram "
                    "without a streamed-edge mode handshake"
                )
        # The routing is the public vocabulary; the native execution-mode
        # spelling is what the evaluator's own plan and artifact APIs accept, so
        # both are kept.
        self.streamed_edges = routing.key
        self._native_streamed_edges = routing.native_mode
        self.execution_algorithm = routing.key
        self._les_geometry_setter = None
        if self._is_mace_les:
            (self._les_geometry_setter,) = family_registry.require_surface(
                self._family_entry, self.evaluator, "mace_les_geometry"
            )
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
                self._native_streamed_edges,
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
            streamed_edges=self._native_streamed_edges,
        )
        if self.metal_request and self.metal_status != "ready":
            raise RuntimeError(
                "metal=True requires generated direct R1 execution on a host "
                f"backend; JIT status is {self.jit_status!r}"
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
        if self._is_mace_les:
            self.implemented_properties.extend(
                ("les_energy", "mace_les_energy", "latent_charges")
            )
        if self._has_native_field_coupling():
            self.implemented_properties.append("node_energy")
            self.implemented_properties.extend(self._macefield_response_properties)
        if self._family_frontend_adapter is not None:
            self._family_frontend_adapter.configure(self)
        self._correction_pipeline = CorrectionPipeline(
            correction_configurations,
            native=symmetrix,
            dtype=self._dtype,
        )
        self._refresh_diagnostics()

    @property
    def alchemical(self):
        """Immutable virtual topology with explicitly mutable slot coefficients."""
        return self._alchemical

    @staticmethod
    def _deprecate_jit_argument(policy):
        if policy is not None:
            warnings.warn(
                "The jit parameter is deprecated and ignored. Direct execution "
                "requires RTC specialization; "
                "select streamed_edges='non-compiled' for compiler-free debug execution.",
                FutureWarning,
                stacklevel=3,
            )

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

    @staticmethod
    def _apply_default_policy(setter, request):
        """Apply a construction-time default that the bundle may override.

        Kokkos evaluators take a ``pin`` flag on their ``_set_*_policy`` hooks:
        a pinned request keeps its value across the automatic throughput
        bundle, while the defaults applied here must stay overridable. An
        evaluator without the flag (test doubles, legacy evaluators) accepts
        the request alone.
        """

        try:
            setter(request, pin=False)
        except TypeError:
            setter(request)

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
        self._apply_default_policy(setter, request)
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
        self._apply_default_policy(setter, request)
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
        self._apply_default_policy(setter, self.edge_geometry_policy)
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
        values = [self.mace_les_evaluator_fingerprint]
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
    def r0_diagnostics(self):
        """Semantic R0 selection with backend and artifact metadata separated."""

        from .r0_reporting import r0_diagnostics

        return r0_diagnostics(self.evaluator)

    @property
    def execution_plan(self):
        """Current native execution selection and graph-time diagnostics.

        The report is pending until the first prepared graph has been sized.
        Private debug pins are deliberately surfaced as their selection source.
        The candidate list is the family-declared plan space; a family that
        declares none reports the single implicit candidate it selected.
        """
        report = getattr(self.evaluator, "execution_plan_report", None)
        if isinstance(report, dict):
            result = dict(report)
        else:
            result = self._implicit_plan_report(
                requested_algorithm=self.execution_algorithm,
                requested_profile=self.execution_profile,
                selection_source="legacy_adapter",
                selected_id=self.execution_algorithm,
                selection_reason="native execution-plan reporting is unavailable",
            )
        result.update(
            {
                "model_type": self.model_type,
                "mace_les_parameters_sha256": self.mace_les_parameters_sha256,
                "mace_les_evaluator_fingerprint": (self.mace_les_evaluator_fingerprint),
            }
        )
        return result

    @staticmethod
    def _implicit_plan_report(
        *,
        requested_algorithm,
        requested_profile,
        selection_source,
        selected_id,
        selection_reason,
    ):
        """The report of a family that declares no plan space.

        Without a candidate space there is no alternative to qualify and no
        byte estimator to consult, so the selected plan is reported as the one
        implicit candidate, carrying the selection reason as its own.
        """
        return {
            "requested_algorithm": requested_algorithm,
            "requested_profile": requested_profile,
            "selection_source": selection_source,
            "state": "active",
            "selected_id": selected_id,
            "selection_reason": selection_reason,
            "available_bytes": 0,
            "reserve_bytes": 0,
            "boundary_attempt": False,
            "candidates": [
                {
                    "id": selected_id,
                    "qualified": True,
                    "estimated_bytes": 0,
                    "reason": selection_reason,
                }
            ],
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
            "\nRTC specialization for direct execution does not fall back. Select "
            "streamed_edges='non-compiled' for compiler-free debug execution."
        )
        if self._direct_artifact_diagnostic is not None:
            from .jit import direct_artifact_diagnostic_text

            message += direct_artifact_diagnostic_text(self._direct_artifact_diagnostic)
        raise _JitRequiredError(message)

    @staticmethod
    def _direct_artifact_failure_cause(
        *,
        artifact_rejected,
        cache_entry_present,
        artifact_present,
        compiler_available,
        backend="host",
    ):
        """Classify a direct-mode R1 artifact failure into its four documented causes.

        The classification is the execution layer's, so every family's frontend
        reports the same four causes for the same observable signals.
        """

        from .jit import direct_artifact_failure_cause

        return direct_artifact_failure_cause(
            artifact_rejected=artifact_rejected,
            cache_entry_present=cache_entry_present,
            artifact_present=artifact_present,
            compiler_available=compiler_available,
            backend=backend,
        )

    @staticmethod
    def _direct_artifact_compiler_available(backend):
        """Report whether the R1 compiler for ``backend`` can build an artifact."""

        from .jit import direct_artifact_compiler_available

        return direct_artifact_compiler_available(backend)

    def _record_direct_artifact_diagnostic(
        self, *, artifact_rejected, cache_key, artifact_path, backend
    ):
        """Record the four-way R1 artifact cause for a failed direct preparation."""

        from .jit import direct_artifact_cache_entry_present

        self._direct_artifact_diagnostic = self._direct_artifact_failure_cause(
            artifact_rejected=artifact_rejected,
            cache_entry_present=direct_artifact_cache_entry_present(cache_key),
            artifact_present=artifact_path is not None,
            compiler_available=self._direct_artifact_compiler_available(backend),
            backend=backend,
        )
        publisher = getattr(self.evaluator, "_set_direct_artifact_diagnostic", None)
        if callable(publisher):
            publisher(*self._direct_artifact_diagnostic)

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
        from .jit import native_source_content_sha256
        from .mace_jit_operator_artifact import (
            prepare_low_memory_operator_modules,
        )

        modules, diagnostics = prepare_low_memory_operator_modules(
            self.evaluator,
            model_data=model_data,
            precision=dtype,
            backend=backend,
            target=target,
            jit_generation_version=jit_generation_version,
            source_content_sha256=native_source_content_sha256(symmetrix),
            prefer_host_m0_plugin=prefer_host_m0_plugin,
            apply_edge_weights=self.alchemical is not None,
        )
        self.jit_operator_modules.update(modules)
        self.jit_diagnostics = (*self.jit_diagnostics, *diagnostics)

    def _configure_metal_r1(self, model_data):
        """Move R1, the standard R0 first interaction, and M0 onto the Metal GPU."""

        from .metal_codegen import metal_r1_metadata, render_jit_r1_metal_source

        loader = getattr(self.evaluator, "_load_metal_r1_module", None)
        if loader is None:
            raise RuntimeError("the native evaluator does not provide Metal R1 support")
        contract = model_data.get("execution_contracts", {}).get("R1")
        if not contract:
            raise RuntimeError("Metal R1 execution requires an Execution R1 contract")
        metadata = metal_r1_metadata(contract)
        loader(
            render_jit_r1_metal_source(contract),
            metadata["channels"],
            metadata["edge_harmonics"],
            metadata["source_harmonics"],
            metadata["output_components"],
        )
        if not self.evaluator._metal_r1_module_ready():
            raise RuntimeError("the evaluator did not retain the Metal R1 module")
        load_r0 = getattr(self.evaluator, "_load_metal_r0_module", None)
        if load_r0 is not None:
            load_r0()
        m0_contract = model_data.get("execution_contracts", {}).get("M0")
        load_m0 = getattr(self.evaluator, "_load_metal_m0_module", None)
        if m0_contract and load_m0 is not None:
            from .metal_codegen import metal_m0_metadata, render_jit_m0_metal_source

            m0 = metal_m0_metadata(m0_contract)
            load_m0(
                render_jit_m0_metal_source(m0_contract),
                m0["channels"],
                m0["input_components"],
                m0["output_components"],
                m0["term_count"],
            )
        load_m1 = getattr(self.evaluator, "_load_metal_m1_module", None)
        if m0_contract and load_m1 is not None:
            from .metal_codegen import metal_m1_metadata, render_jit_m1_metal_source

            m1 = metal_m1_metadata(m0_contract)
            load_m1(
                render_jit_m1_metal_source(m0_contract),
                m1["channels"],
                m1["input_components"],
                m1["output_components"],
                m1["term_count"],
            )
        # Stages without a loaded module, such as M1 for models whose M1
        # contraction has no standard module, stay on the CPU.
        evaluator = self.evaluator
        self.metal_stages = tuple(
            stage
            for stage, ready in (
                ("R0", getattr(evaluator, "_metal_r0_module_ready", lambda: False)()),
                ("M0", getattr(evaluator, "_metal_m0_module_ready", lambda: False)()),
                ("R1", True),
                (
                    "M1",
                    getattr(evaluator, "_metal_m1_module_ready", lambda: False)()
                    and bool(getattr(evaluator, "standard_m1_module_ready", False)),
                ),
            )
            if ready
        )
        self.metal_status = "ready"
        self.metal_device = self.evaluator._metal_r1_device_name()

    def metal_statistics(self):
        """Return Metal R0/R1 launch counts and GPU/staging seconds."""

        query = getattr(self.evaluator, "_metal_statistics", None)
        return dict(query()) if callable(query) else {}

    def _configure_jit(
        self,
        *,
        model_file,
        model_data,
        dtype,
        streamed_edges,
    ):
        if streamed_edges not in ("direct",):
            self.jit_policy = "not_applicable"
            self.jit_status = "not_applicable"
            self.jit_reason = "streamed_edges does not require RTC"
            return

        from .jit import (
            JIT_DISABLED_POLICY,
            JIT_POLICY_ENVIRONMENT_VARIABLE,
            load_validated_jit_artifact,
            remove_jit_quarantine,
            resolve_jit_policy,
        )

        self.jit_policy = resolve_jit_policy()
        if self.jit_policy == JIT_DISABLED_POLICY:
            self._jit_failed(
                f"{JIT_POLICY_ENVIRONMENT_VARIABLE}=none is incompatible "
                f"with streamed_edges={streamed_edges!r}"
            )

        if model_data is None:
            model_data = self._read_json_model(model_file)
        if model_data is None:
            self._jit_failed("the loaded model JSON is unavailable")
            return
        from .prediction_heads import select_prediction_head

        model_data = select_prediction_head(model_data, self.head)
        model_type = model_data.get("model_type", "MACE")
        family = family_registry.declared_registry().detect(model_type=model_type)
        route = family.generated_contract if family is not None else None
        if route is None or route.json_key is None or route.specialization is None:
            self._jit_failed(
                "Execution JIT currently supports ordinary MACE, MACEField, and "
                "compatible MACE-MH-1-family models"
            )
            return
        if family_registry.variant_of(family, model_data) == "single_layer":
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
        specialization = route.specialization
        contract_name = route.json_key
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
                from .mace_nonlinear_execution_contract import (
                    project_execution_mh1_v3_contract,
                )

                validator_symbol = route.validator
                if validator_symbol is None:
                    raise ValueError(
                        f"the {family.key} family does not declare an MH-1 "
                        "contract validator"
                    )
                validator = family_registry.resolve_symbol(validator_symbol)
                contract = validator(model_data, contract)
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
                        mh1_v3_contract["interaction_count"],
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
                        contract["interaction_count"],
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
                if specialization == "mh1":
                    loader_name = "_load_jit_mh1_host_plugin_v7"
                else:
                    loader_name = "_load_jit_host_plugin"
                if not hasattr(self.evaluator, loader_name):
                    raise RuntimeError(
                        "the native evaluator does not provide the host-plugin ABI"
                    )
                jit_backend = "host"
                cuda_environment = None
            elif execution_backend == "cuda" or execution_space == "Cuda":
                loader_name = (
                    "_load_jit_mh1_cuda_plugin_v5"
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
                    "_load_jit_mh1_cuda_plugin_v5"
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

            from .jit import native_source_content_sha256

            device_plan = None
            installed_resolution = None
            if specialization == "r1" and jit_backend in ("cuda", "hip"):
                plan_target = (
                    cuda_environment if jit_backend == "cuda" else device_environment
                )
                from .installed_operator_artifact import (
                    installed_operator_build_target,
                    resolve_installed_operator_artifact,
                )

                if (
                    installed_operator_build_target(
                        native_module=symmetrix, backend=jit_backend
                    )
                    is not None
                ):
                    device_plan = _factorized_device_installed_plan(
                        jit_backend,
                        plan_target,
                        contract,
                        dtype,
                        self.evaluator,
                        apply_edge_weights=self.alchemical is not None,
                    )
                    installed_loader = device_plan.loader
                    if jit_backend == "cuda":
                        installed_blocks = device_plan.metadata[
                            "persistent_blocks_per_compute_unit"
                        ]

                        def installed_loader(path):
                            device_plan.loader(path, installed_blocks)

                    installed_resolution = resolve_installed_operator_artifact(
                        operator="R1",
                        identity=device_plan.metadata["generated_identity"],
                        abi={
                            "tag": device_plan.metadata["abi"],
                            "version": device_plan.metadata["abi_version"],
                        },
                        required_capabilities=device_plan.metadata["capabilities"],
                        loader=installed_loader,
                        admitted=lambda: bool(
                            getattr(
                                self.evaluator,
                                device_plan.ready_attribute,
                                False,
                            )
                        ),
                        native_module=symmetrix,
                        policy=self.jit_policy,
                    )
                    self.jit_diagnostics = (
                        *self.jit_diagnostics,
                        *installed_resolution.diagnostics,
                    )
                if (
                    installed_resolution is None
                    or installed_resolution.status != "resolved"
                ):
                    device_plan = _factorized_device_backend_plan(
                        jit_backend,
                        plan_target,
                        contract,
                        dtype,
                        self.evaluator,
                        apply_edge_weights=self.alchemical is not None,
                    )
                self.jit_compiler_request = device_plan.policy
                self.jit_compiler_backend = device_plan.selected_compiler
                if (
                    installed_resolution is not None
                    and installed_resolution.status == "resolved"
                ):
                    self.jit_compiler_request = "prebuilt"
                    self.jit_compiler_backend = "prebuilt"
                metadata = device_plan.metadata
                selected_edge_policy = device_plan.edge_policy
                selected_variant_id = device_plan.variant_id
                ready_attribute = device_plan.ready_attribute
                artifact_attribute = device_plan.artifact_attribute
                load_plugin = device_plan.loader
                if (
                    installed_resolution is not None
                    and installed_resolution.status == "resolved"
                ):
                    source = "installed-prebuilt-r1"

                    def prepare(*args, **kwargs):
                        return None

                    prepare_arguments = {
                        "abi": {
                            "tag": metadata["abi"],
                            "version": metadata["abi_version"],
                        },
                        "build": {"generator": "installed-prebuilt-r1"},
                    }
                else:
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
                from .mace_nonlinear_jit_codegen import (
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
                ready_attribute = "jit_mh1_cuda_plugin_v5_ready"
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
                if specialization == "mh1" and self.jit_compiler_backend != "nvrtc":
                    raise RuntimeError(
                        "generation-19 MACE-MH-1 CUDA specialization requires "
                        "the composed NVRTC module; the retired monolithic "
                        "plugin renderer is unavailable"
                    )
                if specialization == "mh1":
                    from . import mace_nonlinear_jit_codegen as jit_codegen

                    metadata_function = getattr(
                        jit_codegen, "jit_mh1_cuda_plugin_v5_metadata", None
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
                    generator = "symmetrix.jit.mh1-cuda-module-v1"
                    artifact_name = "execution_mh1_cuda_module"
                    ready_attribute = "jit_mh1_cuda_plugin_v5_ready"
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
                    from . import mace_nonlinear_jit_codegen as jit_codegen
                    from .mace_nonlinear_jit_codegen import (
                        execution_mh1_node_program_metadata,
                    )

                    from .mace_nonlinear_jit_codegen import (
                        render_jit_mh1_host_plugin_v7 as render_function,
                    )

                    generator = "symmetrix.jit.mh1-host-v7-composed"
                    ready_attribute = "jit_mh1_host_plugin_v7_ready"
                    mh1_abi = (
                        jit_codegen.MH1_HOST_PLUGIN_V7_ABI,
                        jit_codegen.MH1_HOST_PLUGIN_V7_ABI_VERSION,
                    )

                    def metadata_function(value):
                        return execution_mh1_node_program_metadata(
                            value,
                            backend="host",
                            node_state_policy=self.execution_mh1_node_state_policy,
                        )

                    artifact_name = "jit_mh1_host_plugin"
                    artifact_attribute = "jit_mh1_host_plugin_artifact_id"
                else:
                    from .mace_jit_codegen import (
                        render_jit_r1_host_plugin as render_function,
                    )
                    from .mace_jit_codegen import (
                        jit_r1_host_plugin_metadata as metadata_function,
                    )

                    generator = "symmetrix.jit.r1-host-v2"
                    artifact_name = "factorized_host_plugin"
                    ready_attribute = "jit_host_plugin_ready"
                    artifact_attribute = "jit_host_plugin_artifact_id"
                if specialization == "r1":
                    from .installed_operator_artifact import (
                        installed_operator_build_target,
                    )
                    from .jit import probe_compiler_identity, resolve_host_jit_policy

                    installed_r1_target = installed_operator_build_target(
                        native_module=symmetrix, backend="host"
                    )
                    requested_host_target = device_environment.get("cpu_target")
                    if requested_host_target is None:
                        requested_host_target = device_environment.get("host_target")
                    if requested_host_target is not None and (
                        not isinstance(requested_host_target, str)
                        or not requested_host_target
                    ):
                        raise RuntimeError(
                            "the host Execution R1 target must be a nonempty string"
                        )
                    host_compiler_command = None
                    host_policy = None
                    if installed_r1_target is None:
                        host_compiler_command, _ = probe_compiler_identity()
                        host_policy = resolve_host_jit_policy(
                            host_compiler_command, target=requested_host_target
                        )
                    initial_r1_target = (
                        installed_r1_target
                        if host_policy is None
                        else host_policy.effective_target
                    )
                    metadata = metadata_function(
                        contract,
                        precision=dtype,
                        apply_edge_weights=self.alchemical is not None,
                        target=initial_r1_target,
                    )
                    source = render_function(
                        contract,
                        precision=dtype,
                        apply_edge_weights=self.alchemical is not None,
                        target=initial_r1_target,
                    )
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
                            "abi": mh1_abi[0],
                            "abi_version": mh1_abi[1],
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

            if device_plan is None:
                # The device plan above already derived its ABI block from the
                # envelope; every other publication path validates it here, so no
                # published artifact can carry an incomplete identity.
                from .jit import validate_artifact_envelope

                if specialization == "mh1":
                    from .mace_nonlinear_jit_codegen import (
                        NONLINEAR_ARTIFACT_ENVELOPE,
                    )

                    envelope = NONLINEAR_ARTIFACT_ENVELOPE
                else:
                    from .mace_jit_codegen import R1_ARTIFACT_ENVELOPE

                    envelope = R1_ARTIFACT_ENVELOPE
                validate_artifact_envelope(metadata, envelope)

            prepare_arguments["build"] = {
                **prepare_arguments["build"],
                "jit_generation_version": jit_generation_version,
            }
            source_content_sha256 = native_source_content_sha256(symmetrix)
            prepare_arguments["source_content_sha256"] = source_content_sha256
            prepare_arguments["show_progress"] = True
            if jit_backend == "cuda":
                if self.jit_compiler_backend == "nvrtc":
                    prepare_arguments["nvrtc_options"] = ("--split-compile=2",)
                elif self.jit_compiler_backend == "nvcc":
                    prepare_arguments["nvcc_flags"] = ("--split-compile=2",)
            if specialization == "mh1" and launch_plan is not None:
                launch_plan_json = json.dumps(
                    launch_plan, sort_keys=True, separators=(",", ":")
                )

                def load_artifact(path):
                    load_plugin(path, launch_plan_json)

            elif specialization == "r1" and jit_backend == "cuda":
                blocks_per_compute_unit = metadata["persistent_blocks_per_compute_unit"]

                def load_artifact(path):
                    load_plugin(path, blocks_per_compute_unit)

            else:
                load_artifact = load_plugin

            if specialization == "r1" and installed_resolution is None:
                from .installed_operator_artifact import (
                    resolve_installed_operator_artifact,
                )

                installed_resolution = resolve_installed_operator_artifact(
                    operator="R1",
                    identity=metadata.get(
                        "generated_identity",
                        {"artifact_id": metadata["artifact_id"]},
                    ),
                    abi=prepare_arguments["abi"],
                    required_capabilities=metadata.get("capabilities", 0),
                    loader=load_artifact,
                    admitted=lambda: bool(
                        getattr(self.evaluator, ready_attribute, False)
                    ),
                    native_module=symmetrix,
                    policy=self.jit_policy,
                )
                self.jit_diagnostics = (
                    *self.jit_diagnostics,
                    *installed_resolution.diagnostics,
                )

            if (
                specialization == "r1"
                and jit_backend == "host"
                and installed_resolution is not None
                and installed_resolution.status != "resolved"
            ):
                if host_policy is None:
                    host_compiler_command, _ = probe_compiler_identity()
                    host_policy = resolve_host_jit_policy(
                        host_compiler_command, target=requested_host_target
                    )
                runtime_target = host_policy.effective_target
                if runtime_target != initial_r1_target:
                    metadata = metadata_function(
                        contract,
                        precision=dtype,
                        apply_edge_weights=self.alchemical is not None,
                        target=runtime_target,
                    )
                    source = render_function(
                        contract,
                        precision=dtype,
                        apply_edge_weights=self.alchemical is not None,
                        target=runtime_target,
                    )
                    validate_artifact_envelope(metadata, R1_ARTIFACT_ENVELOPE)
                    prepare_arguments["abi"] = {
                        "tag": metadata["abi"],
                        "version": metadata["abi_version"],
                    }
                    prepare_arguments["build"] = {
                        **prepare_arguments["build"],
                        "contract": metadata,
                    }
                prepare_arguments.update(
                    cxx=host_compiler_command,
                    cxx_flags=None,
                    host_policy=host_policy,
                )

            def prepare_r1():
                return prepare(source, **prepare_arguments)

            operator_plan = None
            operator_futures = {}
            r1_future = None
            execution_contracts = model_data.get("execution_contracts", {})
            parallel_operator_backend = None
            product_requests = None
            if specialization == "mh1" and isinstance(
                contract.get("node_program"), dict
            ):
                from .mace_jit_operator_artifact import (
                    _mh1_product_artifact_requests,
                )
                from .mace_nonlinear_execution_contract import (
                    project_execution_mh1_product_contract,
                )

                mh1_product_descriptors = tuple(
                    layer["product"] for layer in contract["node_program"]["layers"]
                )
                retain_last_product_irreps = bool(
                    model_data.get("polar", {}).get("keep_last_layer_irreps", False)
                )
                projected_products = tuple(
                    project_execution_mh1_product_contract(
                        product,
                        final=(
                            index == len(mh1_product_descriptors) - 1
                            and not retain_last_product_irreps
                        ),
                    )
                    for index, product in enumerate(mh1_product_descriptors)
                )
                product_requests = _mh1_product_artifact_requests(projected_products)
                parallel_operator_backend = jit_backend
            elif (
                specialization == "r1"
                and self.low_memory_request
                and jit_backend == "cuda"
            ):
                parallel_operator_backend = "cuda"
            elif (
                specialization == "r1"
                and self.low_memory_request
                and jit_backend == "host"
                and isinstance(execution_contracts.get("M0"), dict)
            ):
                parallel_operator_backend = "host"
            if parallel_operator_backend is not None and (
                installed_resolution is None
                or installed_resolution.status != "resolved"
            ):
                from concurrent.futures import ThreadPoolExecutor

                from .mace_jit_operator_artifact import (
                    _plan_low_memory_operator_modules,
                    _prepare_with_initial_future,
                    _submit_operator_artifact_jobs,
                )

                operator_plan = _plan_low_memory_operator_modules(
                    self.evaluator,
                    model_data=model_data,
                    precision=dtype,
                    backend=parallel_operator_backend,
                    target=(
                        cuda_environment
                        if parallel_operator_backend == "cuda"
                        else device_environment
                    ),
                    jit_generation_version=jit_generation_version,
                    source_content_sha256=source_content_sha256,
                    prefer_host_m0_plugin=parallel_operator_backend == "host",
                    apply_edge_weights=self.alchemical is not None,
                    operator_requests=product_requests,
                )
                if operator_plan.jobs:
                    with ThreadPoolExecutor(
                        max_workers=(
                            1 + len(operator_plan.jobs)
                            if parallel_operator_backend == "cuda"
                            else 2
                        ),
                        thread_name_prefix="symmetrix-jit",
                    ) as executor:
                        r1_future = executor.submit(prepare_r1)
                        operator_futures = _submit_operator_artifact_jobs(
                            executor, operator_plan.jobs
                        )
                    prepare_r1 = _prepare_with_initial_future(prepare_r1, r1_future)

            if (
                installed_resolution is not None
                and installed_resolution.status == "resolved"
            ):
                result = installed_resolution
                retained_quarantines = ()
                self.jit_cache_key = None
                self.jit_status = "loaded"
                self.jit_origin = "prebuilt"
            else:
                attempt = load_validated_jit_artifact(
                    prepare_r1,
                    load_artifact,
                    artifact_key=lambda loaded: (
                        self.jit_compiler_backend,
                        loaded.cache_key,
                        str(loaded.artifact_path),
                    ),
                )
                self.jit_cache_key = attempt.cache_key
                self.jit_diagnostics = (
                    *self.jit_diagnostics,
                    *attempt.diagnostics,
                )
                if attempt.status == "unavailable":
                    self._record_direct_artifact_diagnostic(
                        artifact_rejected=False,
                        cache_key=attempt.cache_key,
                        artifact_path=attempt.result.artifact_path,
                        backend=jit_backend,
                    )
                    self._jit_failed(
                        attempt.result.reason
                        or "the JIT cache did not produce an artifact",
                        self.jit_diagnostics,
                    )
                    return
                if attempt.status == "load-failed":
                    raise attempt.error
                if attempt.status == "exhausted":
                    self._record_direct_artifact_diagnostic(
                        artifact_rejected=True,
                        cache_key=attempt.cache_key,
                        artifact_path=None,
                        backend=jit_backend,
                    )
                    self._jit_failed(
                        "the loaded R1 artifact was rejected as stale",
                        self.jit_diagnostics,
                    )
                result = attempt.result
                retained_quarantines = attempt.quarantines
                self.jit_status = result.status
                self.jit_origin = "cache" if result.status == "cached" else "runtime"
                self.jit_diagnostics = (
                    *self.jit_diagnostics,
                    "origin="
                    + self.jit_origin
                    + " status="
                    + ("hit" if self.jit_origin == "cache" else "compiled"),
                )
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
            self.jit_artifact_path = str(result.artifact_path)
            self.jit_artifact_id = getattr(
                self.evaluator, artifact_attribute, metadata["artifact_id"]
            )
            self.jit_forward_policy = selected_forward_policy
            self.jit_source_policy = selected_source_policy
            self.jit_edge_policy = selected_edge_policy
            self.jit_node_state_policy = selected_node_state_policy
            self.jit_variant_id = selected_variant_id
            if specialization == "mh1" and product_requests is not None:
                from .mace_jit_operator_artifact import (
                    _load_low_memory_operator_modules,
                    _validate_mh1_product_module_records,
                )

                if operator_plan is None:
                    raise RuntimeError("MH-1 product artifact plan is incomplete")
                modules, diagnostics = _load_low_memory_operator_modules(
                    self.evaluator,
                    operator_plan,
                    prepared=operator_futures,
                )
                module_target = next(iter(modules.values())).get("target")
                validated_modules = _validate_mh1_product_module_records(
                    modules,
                    product_requests,
                    backend=jit_backend,
                    precision=dtype,
                    target=module_target,
                )
                if not bool(
                    getattr(
                        self.evaluator,
                        "execution_mh1_product_modules_ready",
                        False,
                    )
                ):
                    raise RuntimeError(
                        "the evaluator did not retain the complete MH-1 product "
                        "artifact set"
                    )
                self.jit_operator_modules.update(validated_modules)
                self.jit_diagnostics = (*self.jit_diagnostics, *diagnostics)
            if specialization == "r1" and jit_backend == "host" and self.metal_request:
                self._configure_metal_r1(model_data)
            if specialization == "r1" and self.low_memory_request:
                if operator_plan is not None:
                    from .mace_jit_operator_artifact import (
                        _load_low_memory_operator_modules,
                    )

                    modules, diagnostics = _load_low_memory_operator_modules(
                        self.evaluator,
                        operator_plan,
                        prepared=operator_futures,
                    )
                    self.jit_operator_modules.update(modules)
                    self.jit_diagnostics = (
                        *self.jit_diagnostics,
                        *diagnostics,
                    )
                elif jit_backend in ("cuda", "hip"):
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
    def _native_evaluator_class(model_type, dtype):
        """Return the native evaluator class the model's family declares."""

        return family_registry.resolve_model_evaluator_class(
            model_type=model_type,
            dtype=dtype,
            native=symmetrix,
        )

    def _resolved_family_entry(self):
        """Return the family entry this calculator's evaluator was resolved from.

        A constructed calculator carries the entry its model type resolved to;
        a calculator assembled directly for a focused test resolves it from its
        model type on first use, so the capability gate reads one declaration
        either way.
        """

        entry = getattr(self, "_family_entry", None)
        if entry is not None:
            return entry
        model_type = getattr(self, "_native_model_type", "MACE")
        entry = family_registry.declared_registry().detect(model_type=model_type)
        if entry is None:
            raise ValueError(f"Unsupported Symmetrix model_type '{model_type}'.")
        return entry

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

    def _require_metal_model_family(self):
        if not self.metal_request:
            return
        if self._native_model_type == "MACE" and not self._model_single_layer_readout:
            # Like the generated direct R1 path on the CPU and CUDA, the Metal
            # stages support edge harmonics up to l_max 3.
            l_max = getattr(getattr(self, "evaluator", None), "l_max", None)
            if l_max is not None and l_max > 3:
                raise ValueError(
                    f"metal=True supports l_max up to 3, as does "
                    f"streamed_edges='direct'; this model has l_max {l_max}. "
                    "Leave metal unset and use streamed_edges='generic'."
                )
            return
        # Only two-interaction standard MACE has been qualified on the Metal
        # stages; other families would reach them untested.
        family = (
            "single-layer MACE"
            if self._native_model_type == "MACE"
            else self._native_model_type
        )
        raise ValueError(
            "metal=True supports two-interaction standard MACE models; "
            f"{family} is not qualified on the Metal backend. Leave metal unset "
            "to evaluate it on the CPU."
        )

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
            self.alchemical is not None
            and self._alchemical_evaluated_version != self.alchemical.version
            and "info" not in state
        ):
            state.append("info")
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
        if (
            not state
            and getattr(self, "_family_frontend_adapter", None) is not None
            and not self._family_frontend_adapter.inputs_unchanged(self, atoms, tol)
        ):
            state.append("info")
        return state

    def _has_native_field_coupling(self):
        return hasattr(self, "evaluator") and getattr(
            self.evaluator, "has_field_coupling", False
        )

    def _validate_family_input_mutation(self, name):
        """Reject a family-specific input after family detection selected none."""

        if not hasattr(self, "_family_frontend_adapter"):
            return
        if self._family_frontend_adapter is not None:
            return
        if name == "electric_field" and self._has_native_field_coupling():
            return
        raise ValueError(
            f"Polar-only MACE_Polar option {name} cannot be assigned to "
            f"model family {self._native_model_type!r}"
        )

    @property
    def polar_electrostatics(self):
        """Runtime electrostatics policy for the MACE Polar family."""

        return self._polar_electrostatics

    @polar_electrostatics.setter
    def polar_electrostatics(self, value):
        if value not in {
            "auto",
            "realspace",
            "periodic",
            "direct_reciprocal",
            "pme",
        }:
            raise ValueError(
                "polar_electrostatics must be one of 'auto', 'realspace', "
                "'periodic', 'direct_reciprocal', or 'pme'"
            )
        self._validate_family_input_mutation("polar_electrostatics")
        self._polar_electrostatics = value
        if hasattr(self, "results"):
            self.results.clear()

    @property
    def polar_pme_mesh(self):
        """Runtime PME mesh dimensions for the MACE Polar family."""

        return self._polar_pme_mesh

    @polar_pme_mesh.setter
    def polar_pme_mesh(self, value):
        try:
            mesh = tuple(
                _positive_runtime_integer(extent, "polar_pme_mesh extent")
                for extent in value
            )
        except TypeError as error:
            raise ValueError("polar_pme_mesh must contain three integers") from error
        if len(mesh) != 3:
            raise ValueError("polar_pme_mesh must contain three positive integers")
        self._validate_family_input_mutation("polar_pme_mesh")
        self._polar_pme_mesh = mesh
        if hasattr(self, "results"):
            self.results.clear()

    @property
    def polar_pme_order(self):
        """Runtime PME cardinal B-spline interpolation order."""

        return self._polar_pme_order

    @polar_pme_order.setter
    def polar_pme_order(self, value):
        order = _positive_runtime_integer(value, "polar_pme_order")
        self._validate_family_input_mutation("polar_pme_order")
        self._polar_pme_order = order
        if hasattr(self, "results"):
            self.results.clear()

    @property
    def total_charge(self):
        """The total charge the PolarMACE equilibration targets, if set."""
        return self._total_charge

    @total_charge.setter
    def total_charge(self, value):
        self._validate_family_input_mutation("total_charge")
        self._total_charge = value
        self.results.clear()

    @property
    def total_spin(self):
        """The total spin the PolarMACE equilibration targets, if set."""
        return self._total_spin

    @total_spin.setter
    def total_spin(self, value):
        self._validate_family_input_mutation("total_spin")
        self._total_spin = value
        self.results.clear()

    @property
    def fermi_level(self):
        """The fermi level the PolarMACE graph carries, if set."""
        return self._fermi_level

    @fermi_level.setter
    def fermi_level(self, value):
        self._validate_family_input_mutation("fermi_level")
        self._fermi_level = value
        self.results.clear()

    def _program_declares_polar_dipole_output(self):
        """Report whether this family's compiled program carries a dipole.

        The dipole is the PolarMACE family's own output (its ``dipole_readout``
        node, fed by the equilibrated density); reading it off the program the
        evaluator compiles is what keeps the frontend's output surface and the
        program from drifting apart.
        """

        if not hasattr(self, "evaluator"):
            return False
        if not family_registry.supports_surface(
            self._resolved_family_entry(), self.evaluator, "polar_graph_inputs"
        ):
            return False
        program = getattr(self.evaluator, "family_program", None)
        if not isinstance(program, dict):
            return False
        return any(
            isinstance(node, dict) and node.get("kind") == "dipole_readout"
            for node in program.get("operators", ())
        )

    def _polar_graph_inputs_unchanged(self, atoms, tol):
        """Report whether the resolved polar graph inputs are the last ones."""

        resolved = self._resolve_polar_graph_inputs(atoms)
        previous = self._polar_graph_inputs
        return bool(equal(previous, resolved, atol=tol))

    def _resolve_polar_graph_inputs(self, atoms=None):
        """Resolve the PolarMACE family's graph-level scalar program inputs.

        Precedence is the MACEField field's: an explicitly set value wins, then
        the structure's own ``atoms.info`` entry, then the neutral default
        (zero charge and spin, zero fermi level, and a zero field).  The result
        is the tuple the family's declared input surface takes.
        """

        if atoms is None:
            atoms = getattr(self, "atoms", None)
        info = getattr(atoms, "info", None) or {}
        return {
            "total_charge": self._resolve_polar_scalar(
                "total_charge", info, self._total_charge, 0.0
            ),
            "total_spin": self._resolve_polar_scalar(
                "total_spin", info, self._total_spin, 0.0
            ),
            "fermi_level": self._resolve_polar_scalar(
                "fermi_level", info, self._fermi_level, 0.0
            ),
            "electric_field": self._resolve_electric_field(atoms),
        }

    @staticmethod
    def _resolve_polar_scalar(name, info, explicit, default):
        if explicit is not None:
            value = explicit
        elif name in info:
            value = info[name]
        else:
            return default
        if isinstance(value, bool) or not np.isscalar(value):
            raise ValueError(
                f"the PolarMACE {name} input must be a graph-level scalar."
            )
        resolved = float(value)
        if not np.isfinite(resolved):
            raise ValueError(f"the PolarMACE {name} input must be finite.")
        return resolved

    def _apply_polar_graph_inputs(self, atoms=None):
        """Hand the family's graph inputs to its declared input surface.

        The channel is the family's own: a family that declares no polar input
        surface (every other family) leaves the evaluation untouched.
        """

        family = self._resolved_family_entry()
        if not family_registry.supports_surface(
            family, self.evaluator, "polar_graph_inputs"
        ):
            return
        (set_polar_graph_inputs,) = family_registry.require_surface(
            family, self.evaluator, "polar_graph_inputs"
        )
        inputs = self._resolve_polar_graph_inputs(atoms)
        set_polar_graph_inputs(
            inputs["total_charge"],
            inputs["total_spin"],
            inputs["fermi_level"],
            np.asarray(inputs["electric_field"], dtype=float).reshape(-1),
        )
        self._polar_graph_inputs = inputs

    def _compose_polar(
        self,
        atoms,
        mace_inputs,
        compute_reverse=False,
        materialize_diagnostics=False,
    ):
        """The family's chain and long-range composition for this evaluation.

        The backbone half is the atomic reference plus the interaction energy
        the evaluator already produced; what the family adds is the oracle's
        own composition of the chain and the global nodes:

            E = e0 + inter + local_electron(chain) + kspace + correction
                + sum_lanes E_lane . mu_lane

        with the chain's equilibrated density feeding the k-space energy, the
        pbc-selected analytic correction, and the dipole readout.  The lanes
        are the family's graph-input channel (``total_charge``, ``total_spin``,
        the electric field in lanes 1..3, and the fermi level in lane 0, which
        the oracle fills and never couples).
        """

        num_nodes = int(mace_inputs[0])
        node_types = np.asarray(mace_inputs[1], dtype=np.int64).reshape(-1)
        native_program = getattr(self.evaluator, "polar_delta_native_program", None)
        if not callable(native_program):
            raise family_registry.MissingCapabilityError(
                "the MACE Polar staged test oracle requires the native "
                "polar_delta_native_program capability; retained-layer "
                "materialization is not a supported compatibility path"
            )
        positions = np.ascontiguousarray(atoms.get_positions(), dtype=np.float64)
        cell = np.ascontiguousarray(np.asarray(atoms.cell, dtype=np.float64)).reshape(
            1, 3, 3
        )
        pbc = np.asarray(atoms.pbc, dtype=bool).reshape(1, 3).astype(np.int32)
        # The composition is one graph per evaluation: the frontend's atoms are
        # one structure, so the node-to-graph index it hands the chain, the
        # cache and every readout is a single all-zero column, and the
        # graph-level reads below (``[0]`` on the energy, the correction, the
        # dipole, and the stress tensor) address that one graph.
        node_graph_index = np.zeros(num_nodes, dtype=np.int32)
        inputs = self._polar_graph_inputs or {}
        total_charge = float(inputs.get("total_charge", 0.0))
        # The channel's own default is the resolution's, so an unset spin means
        # the same thing here as it does in ``_resolve_polar_graph_inputs``.
        total_spin = float(inputs.get("total_spin", 0.0))
        electric_field = np.asarray(
            inputs.get("electric_field", (0.0, 0.0, 0.0)), dtype=np.float64
        ).reshape(3)
        # Lane 0 is the fermi level, which the oracle's external-potential
        # buffer carries as zeros and never couples: the record's fermi-lane
        # channel is bit-identical to its singlet, so the composition passes
        # zeros there too rather than coupling a lane the model ignores.
        external_potential = np.array(
            [[0.0, electric_field[0], electric_field[1], electric_field[2]]],
            dtype=np.float64,
        )
        state = native_program(
            node_types=node_types,
            positions=positions,
            cell=cell,
            graph_pbc=pbc,
            node_graph_index=node_graph_index,
            total_charge=[total_charge],
            total_spin=[total_spin],
            external_potential=external_potential,
            compute_reverse=compute_reverse,
            materialize_outputs=materialize_diagnostics,
            polar_electrostatics=self.polar_electrostatics,
        )
        solver_keys = (
            "requested_solver",
            "resolved_solver",
            "solver_reason",
            "automatic_atom_count",
            "automatic_reciprocal_mode_count",
            "automatic_reciprocal_mode_count_is_lower_bound",
            "automatic_atom_mode_pair_count",
            "automatic_atom_mode_pair_threshold",
            "n_k",
            "reciprocal_allocation_elements",
            "reciprocal_allocation_bytes",
            "reciprocal_table_elements",
            "reciprocal_table_bytes",
            "particle_mesh_active_bytes",
            "particle_mesh_capacity_bytes",
            "particle_mesh_volume",
            "particle_mesh_shape",
            "particle_mesh_rebuilds",
            "particle_mesh_reuses",
            "particle_mesh_forward_plan_builds",
            "particle_mesh_forward_plan_reuses",
            "particle_mesh_inverse_plan_builds",
            "particle_mesh_inverse_plan_reuses",
        )
        # Staged programs can report the selected solver without allocating a
        # particle mesh.  Merge the workspace and call result per field rather
        # than replacing the complete result when one optional PME field is
        # absent.  Explicit zero values prevent diagnostics from a preceding
        # PME evaluation leaking into a later direct-solver evaluation.
        diagnostic_source = {
            **self.evaluator.polar_delta_workspace(),
            **state,
        }
        particle_mesh_defaults = {
            "particle_mesh_active_bytes": 0,
            "particle_mesh_capacity_bytes": 0,
            "particle_mesh_volume": 0,
            "particle_mesh_shape": [0, 0, 0],
            "particle_mesh_rebuilds": 0,
            "particle_mesh_reuses": 0,
            "particle_mesh_forward_plan_builds": 0,
            "particle_mesh_forward_plan_reuses": 0,
            "particle_mesh_inverse_plan_builds": 0,
            "particle_mesh_inverse_plan_reuses": 0,
        }
        solver_diagnostics = {
            key: diagnostic_source.get(key, particle_mesh_defaults[key])
            if key in particle_mesh_defaults
            else diagnostic_source[key]
            for key in solver_keys
        }
        self.polar_electrostatics_diagnostics = solver_diagnostics
        if not materialize_diagnostics:
            return {
                "positions": positions,
                "pbc": pbc,
                "electric_field": electric_field,
                "total_charge": total_charge,
                "total_spin": total_spin,
                "cell": cell,
                "node_graph_index": node_graph_index,
                "external_potential": external_potential,
                "_native_layer_seed_installed": compute_reverse,
                "_native_resident_outputs": True,
                **solver_diagnostics,
            }
        kspace = float(np.asarray(state["electrostatic_energy_kspace"]).reshape(-1)[0])
        correction = float(
            np.asarray(state["electrostatic_energy_correction"]).reshape(-1)[0]
        )
        dipole = np.asarray(state["dipole"], dtype=np.float64).reshape(3)
        local_electron = float(
            np.asarray(state["local_electron_energy"]).reshape(-1)[0]
        )
        field_energy = float(np.asarray(state["external_field_energy"]).reshape(-1)[0])
        result = {
            "positions": positions,
            "pbc": pbc,
            "electric_field": electric_field,
            "energy": float(np.asarray(state["energy"]).reshape(-1)[0]),
            "local_electron_energy": local_electron,
            "electrostatic_energy_kspace": kspace,
            "electrostatic_energy_correction": correction,
            "external_field_energy": field_energy,
            "dipole": dipole,
            "total_charge": total_charge,
            "total_spin": total_spin,
            "cell": cell,
            "node_graph_index": node_graph_index,
            "external_potential": external_potential,
            "_native_layer_seed_installed": compute_reverse,
            **solver_diagnostics,
        }
        if compute_reverse:
            result.update(
                {
                    "positions_adjoint": np.asarray(
                        state["positions_adjoint"], dtype=np.float64
                    ),
                    "strain_adjoint": np.asarray(
                        state["strain_adjoint"], dtype=np.float64
                    ).reshape(-1, 3, 3),
                    "stress": np.asarray(state["stress"], dtype=np.float64).reshape(
                        -1, 3, 3
                    ),
                }
            )
            if "node_features_adjoint" in state:
                result["node_features_adjoint"] = np.asarray(
                    state["node_features_adjoint"], dtype=np.float64
                )
        return result

    def _compose_polar_reverse(self):
        """Run the native reverse of the complete polar energy program.

        Python supplies compact model inputs and receives compact final
        adjoints.  The density, phase-table, geometry, correction, dipole, and
        stress composition stays in libsymmetrix, including its shared device
        accumulators.

        Three products come back, and each has an owner:

        * ``energy_gradient``/``forces`` -- the *fixed-feature* position
          gradient of the polar terms, added to the backbone's;
        * ``layer_product_seed`` -- the chain's seed of the node features it
          was handed, i.e. of the backbone's retained per-layer products,
          split per layer for the reverse tap.  Positions move the products,
          the products move the chain, and only the backbone's own reverse can
          carry that half;
        * ``stress`` -- the complete polar stress, including cell metadata,
          analytic correction, and the position virial.
        """

        composition = self._polar_composition
        # The granular long-range adjoints remain exposed for reference tests,
        # but production's forward state feeds its reverse before the native
        # call returns; Python only reshapes the compact final products.
        position_seed = np.asarray(composition["positions_adjoint"], dtype=np.float64)
        # The chain's seed of its own node features is a seed of the
        # backbone's per-layer products (the features it was handed are the
        # retained products), so it is returned split per layer, in the
        # retention's order and basis: the family adopts it on the backbone's
        # reverse to collect the cross term its composition owns but cannot
        # differentiate -- positions change the products, the products change
        # the chain, and only the backbone's own reverse can carry that.
        layer_product_seed = None
        if not composition.get("_native_layer_seed_installed", False):
            feature_seed = np.asarray(
                composition["node_features_adjoint"], dtype=np.float64
            )
            width = int(self._polar_shapes["node_feature_dimension"])
            layers = int(self._polar_shapes["interaction_count"])
            layer_product_seed = [
                np.ascontiguousarray(
                    feature_seed[:, index * width : (index + 1) * width]
                )
                for index in range(layers)
            ]
            if feature_seed.shape[1] != layers * width:
                raise RuntimeError(
                    "the chain's node-feature seed carries "
                    f"{feature_seed.shape[1]} columns but the contract's "
                    f"{layers} layer(s) of width {width} account for "
                    f"{layers * width}: the seed and the retention disagree on the "
                    "backbone's product layout."
                )
        return {
            "energy_gradient": position_seed,
            "forces": -position_seed,
            "layer_product_seed": layer_product_seed,
            "strain": composition["strain_adjoint"],
            "stress": composition["stress"],
        }

    def _materialize_polar_diagnostics(self):
        """Explicitly materialize private polar state for diagnostic callers.

        Normal energy/force/stress evaluation deliberately keeps this state in
        native memory. This opt-in path recomputes a matching unseeded backbone
        forward, asks the native polar program for its diagnostic outputs, and
        leaves the already served public results unchanged.
        """

        if self._family_frontend_adapter is None or self.atoms is None:
            raise RuntimeError(
                "polar diagnostics require a completed PolarMACE evaluation."
            )
        mace_inputs = self._mace_inputs(self.atoms)
        self.evaluator.clear_layer_product_seed()
        self._compute_mace(mace_inputs, atoms=self.atoms)
        try:
            diagnostic = self._compose_polar(
                self.atoms,
                mace_inputs,
                compute_reverse=True,
                materialize_diagnostics=True,
            )
        finally:
            self.evaluator.clear_layer_product_seed()
        diagnostic["polar_stress"] = np.array(
            diagnostic["stress"], dtype=np.float64, copy=True
        )
        self._polar_composition = diagnostic
        return diagnostic

    def _compose_polar_stress(self, seeds):
        """Return the complete polar stress produced by libsymmetrix."""

        return np.asarray(seeds["stress"], dtype=np.float64).reshape(-1, 3, 3)

    def _polar_dipole_output(self):
        """Return the PolarMACE program's dipole output for this evaluation.

        The composition produces the dipole (the program's ``dipole_readout``
        node fed by the chain's equilibrated density), so this is a readout of
        the evaluation that ran; a dipole requested before any evaluation of
        this calculator is refused by name rather than answered with a number
        the model did not produce.
        """

        composition = self._polar_composition
        if composition is not None:
            return np.asarray(composition["dipole"], dtype=float).reshape(3)
        raise PropertyNotImplementedError(
            "the PolarMACE dipole output is produced by the family's own "
            "composition (the program's dipole_readout node, fed by the "
            "chain's equilibrated density), and no evaluation of this "
            "calculator has run it yet: request the energy and the composition "
            "produces the dipole with it."
        )

    @property
    def electric_field(self):
        return self._electric_field

    @electric_field.setter
    def electric_field(self, value):
        self._validate_family_input_mutation("electric_field")
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
        if cache is not None and (
            allow_device_neighbor_graph or cache.device_graph_generation == 0
        ):
            reference = NeighborTopologyReference(
                atomic_numbers=cache.atomic_numbers,
                pbc=cache.pbc,
                topology_cell=cache.topology_cell,
                topology_reference_positions=cache.topology_reference_positions,
                topology_fractional_positions=cache.topology_fractional_positions,
                cell=cache.cell,
                reference_positions=cache.reference_positions,
            )
            reuse = neighbor_topology_reuse(
                reference=reference,
                atomic_numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff,
                skin=self.neighbor_skin,
            )
            cell_changed = reuse.cell_changed
            geometry_reference_positions = reuse.geometry_reference_positions
            if reuse.reusable:
                if cell_changed:
                    displacements = np.zeros_like(positions)
                else:
                    displacement_reference = (
                        cache.reference_positions
                        if cache.fractional_geometry_eligible
                        else cache.topology_reference_positions
                    )
                    displacements = minimum_image_displacements(
                        positions, displacement_reference, cell, pbc
                    )
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
                validate_graph_cardinality(len(atomic_numbers), num_edges)
                receivers = np.empty(0, dtype=np.int32)
                sources = np.empty(0, dtype=np.int32)
                shifts = np.empty((0, 3), dtype=np.int32)
                num_neigh = np.empty(0, dtype=np.int32)
                neigh_types = np.empty(0, dtype=np.int32)
                device_graph_generation = int(device_graph["generation"])
                topology_fractional_xyz = np.empty((0, 3), dtype=float)
                self.neighbor_graph_backend = "kokkos"
            else:
                graph = build_host_neighbor_graph(
                    atoms, self.cutoff + self.neighbor_skin
                )
                receivers = graph.receivers
                sources = graph.sources
                shifts = graph.shifts
                num_edges = graph.num_edges
                num_neigh = graph.num_neigh
                neigh_types = graph.neighbor_types(node_types)
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
            inverse_cell = reuse.inverse_cell
            if inverse_cell is None:
                raise RuntimeError(
                    "Cell-aware neighbor reuse requires an invertible cell."
                )
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

    def _alchemical_inputs(self, atoms):
        config = self.alchemical
        resolved_config = resolve_alchemical_config(
            config, atoms.numbers, self.evaluator.atomic_numbers
        )
        receivers, sources, distances, xyz = self._cached_neighbor_geometry(
            atoms, native_geometry=False, allow_device_neighbor_graph=False
        )
        cache = self._neighbor_cache
        if self._alchemical_topology_generation != cache.generation:
            self._alchemical_topology = expand_alchemical_topology(
                atomic_numbers=atoms.numbers,
                receivers=receivers,
                sources=sources,
                shifts=cache.shifts,
                config=resolved_config,
            )
            self._alchemical_topology_generation = cache.generation
        topology = self._alchemical_topology
        type_by_number = {
            number: index for index, number in enumerate(self.evaluator.atomic_numbers)
        }
        node_types = np.asarray(
            [type_by_number[number] for number in topology.node_atomic_numbers],
            dtype=np.int32,
        )
        edge_map = topology.edge_to_original_edge
        validate_graph_cardinality(len(node_types), len(edge_map))
        return (
            len(node_types),
            node_types,
            topology.num_neigh,
            topology.edge_sources,
            np.ascontiguousarray(node_types[topology.edge_sources]),
            np.ascontiguousarray(xyz[edge_map]),
            np.ascontiguousarray(distances[edge_map]),
            topology.edge_receivers,
        )

    def _mace_inputs(
        self, atoms, native_geometry=False, allow_device_neighbor_graph=True
    ):
        if self.alchemical is not None:
            return self._alchemical_inputs(atoms)
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
            i_list, j_list, r, xyz = neighbor_graph.host_neighbor_list(
                atoms, self.cutoff, "ijdD"
            )
            self.neighbor_graph_backend = "host"
        num_nodes = len(atoms)
        cache = getattr(self, "_neighbor_cache", None)
        graph_edges = (
            cache.num_edges
            if cache is not None and cache.device_graph_generation != 0
            else len(i_list)
        )
        validate_graph_cardinality(num_nodes, graph_edges)
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
        family = self._resolved_family_entry()
        # The family's graph-level scalar channel is input to whatever routing
        # follows, so it enters here, before the graph is prepared: the same
        # prepared lifecycle the graph itself travels.
        self._apply_polar_graph_inputs(atoms)
        mode = getattr(self.evaluator, "streamed_edges_mode", None)
        if mode == "generic" and xyz is None and r is None:
            prepare_graph, prepare_geometry, compute_positions = (
                family_registry.require_surface(
                    family, self.evaluator, "all_interactions"
                )
            )
            cache = self._neighbor_cache
            if atoms is None or cache is None:
                raise RuntimeError(
                    "Native streamed all geometry requires atoms and a prepared "
                    "neighbor cache."
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

        if mode in _PREPARED_EXECUTION_MODES:
            prepare_graph, compute_prepared = family_registry.require_surface(
                family, self.evaluator, "prepared"
            )
            native_geometry = xyz is None and r is None
            if (xyz is None) != (r is None):
                raise ValueError("MACE geometry must provide both xyz and distances.")
            if native_geometry:
                (compute_positions,) = family_registry.require_surface(
                    family, self.evaluator, "prepared_positions"
                )
                cache = self._neighbor_cache
                if atoms is None or cache is None:
                    raise RuntimeError(
                        "Native Execution geometry requires atoms and a prepared "
                        "neighbor cache."
                    )
                if cache.device_graph_generation != 0:
                    graph_generation = cache.device_graph_generation
                    if cache.device_geometry_generation != cache.geometry_generation:
                        (update_cell,) = family_registry.require_surface(
                            family, self.evaluator, "prepared_cell_update"
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
                    family, graph_generation, atoms
                ):
                    return graph_generation
                if self._can_use_fractional_geometry(cache):
                    (prepare_fractional_geometry,) = family_registry.require_surface(
                        family, self.evaluator, "prepared_fractional_geometry"
                    )
                    (update_cell,) = family_registry.require_surface(
                        family, self.evaluator, "prepared_cell_update"
                    )
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
                    (prepare_geometry,) = family_registry.require_surface(
                        family, self.evaluator, "prepared_geometry"
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
            if self.alchemical is not None:
                weight_identity = (graph_generation, self.alchemical.version)
                if weight_identity != self._alchemical_native_weight_identity:
                    topology = self._alchemical_topology
                    node_weights, edge_weights = topology.resolve_weights(
                        self.alchemical.weights
                    )
                    self.evaluator._set_alchemical_weights(
                        graph_generation, node_weights, edge_weights
                    )
                    self._alchemical_native_weight_identity = weight_identity
            self._sync_low_memory_policy_state()
            if self._compute_fixed_workspace_shift_geometry(
                family, graph_generation, atoms
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
        (compute_reduced,) = family_registry.require_surface(
            family, self.evaluator, "reduced"
        )
        compute_reduced(
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
        family = self._resolved_family_entry()
        return (
            getattr(self, "_native_model_type", "MACE") == "MACE"
            and cache is not None
            and cache.fractional_geometry_eligible
            and getattr(self.evaluator, "streamed_edges_mode", None)
            in _PREPARED_EXECUTION_MODES
            and not self._has_native_field_coupling()
            and family_registry.supports_surface(
                family, self.evaluator, "prepared_fractional_geometry"
            )
            and family_registry.supports_surface(
                family, self.evaluator, "prepared_cell_update"
            )
        )

    def _fixed_workspace_plan_selected(self):
        native_plan = getattr(self.evaluator, "execution_plan_report", None)
        return (
            isinstance(native_plan, dict)
            and native_plan.get("selected_id") in _FIXED_WORKSPACE_TILED_PLANS
        )

    def _fixed_workspace_geometry_eligible(self):
        return self._model_single_layer_readout or self._fixed_workspace_plan_selected()

    def _compute_fixed_workspace_shift_geometry(self, family, graph_generation, atoms):
        if not self._fixed_workspace_plan_selected():
            return False
        cache = getattr(self, "_neighbor_cache", None)
        if atoms is None or cache is None:
            raise RuntimeError(
                "Fixed-workspace tiled execution requires atoms and a prepared "
                "neighbor cache."
            )
        (prepare_shift_geometry,) = family_registry.require_surface(
            family, self.evaluator, "prepared_shift_geometry"
        )
        (compute_positions,) = family_registry.require_surface(
            family, self.evaluator, "prepared_positions"
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
        if self.alchemical is not None:
            return False
        if self.neighbor_skin <= 0.0 and not self._fixed_workspace_geometry_eligible():
            return False
        family = self._resolved_family_entry()
        if "stress" in properties and not family_registry.supports_surface(
            family, self.evaluator, "reduced_stress"
        ):
            return False
        mode = getattr(self.evaluator, "streamed_edges_mode", None)
        if self._has_native_field_coupling():
            requires_response = any(
                prop in ("becs", "polarizability") for prop in properties
            )
            if requires_response and not family_registry.supports_surface(
                family, self.evaluator, "prepared_field_response"
            ):
                return False
            return (
                mode in _PREPARED_EXECUTION_MODES
                and family_registry.supports_surface(
                    family, self.evaluator, "prepared_graph"
                )
                and family_registry.supports_surface(
                    family, self.evaluator, "prepared_geometry"
                )
                and family_registry.supports_surface(
                    family, self.evaluator, "prepared_field_positions"
                )
            )
        if mode == "generic":
            return family_registry.supports_surface(
                family, self.evaluator, "all_interactions"
            )
        return (
            mode in _PREPARED_EXECUTION_MODES
            and family_registry.supports_surface(
                family, self.evaluator, "prepared_graph"
            )
            and family_registry.supports_surface(
                family, self.evaluator, "prepared_geometry"
            )
            and family_registry.supports_surface(
                family, self.evaluator, "prepared_positions"
            )
        )

    def _compute_macefield(self, atoms, electric_field, mace_inputs=None):
        if mace_inputs is None:
            mace_inputs = self._mace_inputs(atoms)
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list = (
            mace_inputs
        )
        field = np.ascontiguousarray(electric_field, dtype=float).reshape(-1)
        family = self._resolved_family_entry()
        mode = getattr(self.evaluator, "streamed_edges_mode", None)
        self._macefield_response_graph_generation = 0
        if mode in _PREPARED_EXECUTION_MODES:
            prepare_graph, compute_prepared = family_registry.require_surface(
                family, self.evaluator, "prepared_field"
            )
            native_geometry = xyz is None and r is None
            if (xyz is None) != (r is None):
                raise ValueError(
                    "MACEField geometry must provide both xyz and distances."
                )
            if native_geometry:
                (compute_positions,) = family_registry.require_surface(
                    family, self.evaluator, "prepared_field_positions"
                )
                (prepare_geometry,) = family_registry.require_surface(
                    family, self.evaluator, "prepared_geometry"
                )
                cache = self._neighbor_cache
                if cache is None:
                    raise RuntimeError(
                        "Native MACEField geometry requires a prepared neighbor cache."
                    )
                if cache.device_graph_generation != 0:
                    graph_generation = cache.device_graph_generation
                    if cache.device_geometry_generation != cache.geometry_generation:
                        (update_cell,) = family_registry.require_surface(
                            family, self.evaluator, "prepared_cell_update"
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
        (compute_reduced_field,) = family_registry.require_surface(
            family, self.evaluator, "reduced_field"
        )
        compute_reduced_field(
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
        force_derivative_bins = False

        def compute_field_derivatives(include_forces=False):
            nonlocal field_derivatives_computed, force_derivatives_computed
            nonlocal force_derivative_bins
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
                    scatter_bec_bins=include_forces,
                )
                field_derivatives_computed = True
                force_derivatives_computed = include_forces
                force_derivative_bins = include_forces
                return

            num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, _ = (
                mace_inputs
            )
            if include_forces:
                compute_current_bins = getattr(
                    self.evaluator,
                    "_compute_current_electric_field_bec_bins",
                    None,
                )
                compute_current_response = getattr(
                    self.evaluator,
                    "_compute_current_electric_field_force_derivative",
                    None,
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
                if callable(compute_current_bins):
                    compute_current_bins(
                        *response_args,
                        self._macefield_response_graph_generation,
                    )
                    force_derivative_bins = True
                elif callable(compute_current_response):
                    compute_current_response(
                        *response_args,
                        self._macefield_response_graph_generation,
                    )
                else:
                    self.evaluator.compute_electric_field_force_derivative(
                        *response_args
                    )
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
            if force_derivative_bins:
                becs = np.array(
                    self.evaluator.electric_field_bec_bins, dtype=float, copy=True
                ).reshape(len(atoms), 3, 3)
            else:
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
        if self.alchemical is not None:
            node_energies = np.bincount(
                self._alchemical_topology.node_to_original_atom,
                weights=node_energies,
                minlength=len(atoms),
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
        if self.alchemical is not None:
            physical_forces = np.zeros((len(atoms), 3), dtype=float)
            np.add.at(
                physical_forces,
                self._alchemical_topology.node_to_original_atom,
                atom_forces,
            )
            atom_forces = physical_forces
        results["forces"] = atom_forces
        if "dipole" in properties:
            results["dipole"] = self._polar_dipole_output()
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

    def _becs_requires_host_neighbor_graph(self, properties):
        """Report whether BEC assembly still needs host receiver/source lists.

        The device-bins reduction owns BEC accumulation natively, so only the
        legacy pair-derivative fallback (evaluators without the bins API)
        consumes host edge geometry.
        """
        if "becs" not in properties:
            return False
        bins = getattr(self.evaluator, "_compute_current_electric_field_bec_bins", None)
        prepared_response = getattr(
            self.evaluator, "_compute_prepared_factorized_field_response", None
        )
        return not (callable(bins) and callable(prepared_response))

    def _refresh_diagnostics(self):
        if self._is_mace_les:
            native_diagnostics = dict(self.evaluator.les_diagnostics)
            native_diagnostics.update(
                {
                    "workspace": dict(self.evaluator.les_charge_workspace_diagnostics),
                }
            )
            self.les_diagnostics = native_diagnostics
        else:
            self.les_diagnostics = None
        self.correction_diagnostics = self._correction_pipeline.diagnostics()

    def _set_current_les_geometry(self, atoms):
        if not self._is_mace_les:
            return
        self._les_geometry_setter(
            np.ascontiguousarray(atoms.positions, dtype=np.float64),
            np.ascontiguousarray(atoms.cell.array, dtype=np.float64),
            np.ascontiguousarray(atoms.pbc, dtype=np.int32),
        )

    def _publish_les_results(self, atoms):
        if not self._is_mace_les:
            return
        les_energy = float(self.evaluator.les_energy)
        if not np.isfinite(les_energy):
            raise ValueError("native LES energy must be finite")
        self.results["les_energy"] = les_energy
        self.results["mace_les_energy"] = float(self.results["energy"])
        latent_charges = np.array(self.evaluator.latent_charges, dtype=float, copy=True)
        if latent_charges.shape != (len(atoms),):
            raise ValueError(
                "native LES latent charges must have shape "
                f"({len(atoms)},), got {latent_charges.shape}"
            )
        if not np.all(np.isfinite(latent_charges)):
            raise ValueError("native LES latent charges must be finite")
        self.results["latent_charges"] = latent_charges
        self._refresh_diagnostics()

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if self._is_mace_les and atoms is not None and not isinstance(atoms, Atoms):
            raise ValueError("MACELES evaluates one ASE Atoms structure at a time")
        Calculator.calculate(self, atoms, properties, system_changes)
        family_frontend_adapter = getattr(self, "_family_frontend_adapter", None)

        requested_properties = list(dict.fromkeys(properties))
        evaluation_properties = list(requested_properties)
        stress_properties = [*evaluation_properties, "stress"]
        if (
            self.atoms.cell.rank == 3
            and "stress" not in evaluation_properties
            and self._can_use_native_geometry(stress_properties)
        ):
            evaluation_properties.append("stress")

        if family_frontend_adapter is not None:
            native_geometry = self._can_use_native_geometry(evaluation_properties)
            mace_inputs = self._mace_inputs(
                self.atoms,
                native_geometry=native_geometry,
                allow_device_neighbor_graph=True,
            )
        elif self._can_use_native_geometry(evaluation_properties):
            mace_inputs = self._mace_inputs(
                self.atoms,
                native_geometry=True,
                allow_device_neighbor_graph=(
                    not self._becs_requires_host_neighbor_graph(evaluation_properties)
                ),
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
                allow_device_neighbor_graph=(
                    not self._becs_requires_host_neighbor_graph(evaluation_properties)
                ),
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
        if family_frontend_adapter is not None:
            self.results = family_frontend_adapter.evaluate(
                self, self.atoms, requested_properties, mace_inputs
            )
        elif self._has_native_field_coupling():
            electric_field = self._resolve_electric_field()
            self.results = self._calculate_macefield_results(
                self.atoms,
                electric_field,
                evaluation_properties,
                mace_inputs=mace_inputs,
            )
            self._macefield_electric_field = np.array(electric_field, copy=True)
        else:
            self._set_current_les_geometry(self.atoms)
            graph_generation = self._compute_mace(mace_inputs, atoms=self.atoms)
            self.results = self._collect_mace_results(
                self.atoms,
                mace_inputs,
                evaluation_properties,
                graph_generation,
            )
            self._publish_les_results(self.atoms)
        if self.alchemical is not None:
            self._alchemical_evaluated_version = self.alchemical.version
        correction_pipeline = getattr(self, "_correction_pipeline", None)
        if correction_pipeline:
            correction_properties = [
                name
                for name in requested_properties
                if name in _FIELD_ADDITIVE_PROPERTIES
            ]
            requested_additive = set(correction_properties)
            for name in _FIELD_ADDITIVE_PROPERTIES:
                if name not in requested_additive:
                    self.results.pop(name, None)
            correction_pipeline.add_results(
                self.atoms, self.results, correction_properties
            )
            if self._is_mace_les and (
                "energy" in requested_properties
                or "free_energy" in requested_properties
            ):
                corrected_energy = self.results.get(
                    "energy", self.results.get("free_energy")
                )
                if corrected_energy is not None:
                    additive = self._correction_configurations
                    if additive and isinstance(additive[0], D3Correction):
                        self.results["d3_energy"] = (
                            float(corrected_energy) - self.results["mace_les_energy"]
                        )
            self._refresh_diagnostics()


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
