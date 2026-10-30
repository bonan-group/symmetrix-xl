"""ASE calculator for explicit native NequIP JSON contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase.stress import full_3x3_to_voigt_6_stress
except ImportError:  # pragma: no cover - exercised by optional-dependency tests
    Calculator = object
    all_changes = ()

from . import family_registry, load_backend
from .neighbor_graph import (
    build_host_neighbor_graph,
    edge_vectors,
    skin_reuse_holds,
)


def _execution_artifact_target(native):
    """Return ``(name, argument)`` for the artifact target this build binds.

    A CUDA build binds its generated CUDA artifact, so the target argument is
    the native execution-device environment that both the CUDA contract
    identity and the shared JIT compiler policy accept.  Every other build — a
    host/OpenMP backend, a HIP backend, or a native build predating the
    environment query — binds the host artifact, whose adapter takes the host
    target as its own name. HIP is rejected explicitly: a host receiver callback
    cannot execute in a HIP Kokkos execution space, and OAM has no HIP artifact.
    """

    query = getattr(native, "_execution_device_execution_environment", None)
    if not callable(query):
        return "host", "host"
    environment = query()
    if not isinstance(environment, dict):
        return "host", "host"
    backend = str(environment.get("backend", "")).lower()
    execution_space = str(environment.get("execution_space", ""))
    if backend == "hip" or "hip" in execution_space.lower():
        raise RuntimeError(
            "NequIP-OAM execution on HIP is not supported in this release; "
            "use a qualified host or CUDA backend"
        )
    if backend != "cuda" and execution_space != "Cuda":
        return "host", "host"
    return "cuda", environment


def _mode_rejection(vocabulary):
    """Return the rejection message of a spelling the vocabulary does not accept.

    The accepted spellings are the detected family's declaration, so the
    message cannot drift from the modes the frontend actually resolves.
    """

    spellings = list(vocabulary.aliases)
    accepted = ", ".join(repr(name) for name in spellings[:-1])
    return f"streamed_edges must be {accepted}, or {spellings[-1]!r}"


class NequIPCalculator(Calculator):
    """Evaluate an exported NequIP contract through the native evaluator.

    The native runtime accepts the versioned JSON contract only.  Upstream
    ``.nequip`` and AOTInductor artifacts must be exported separately.
    """

    implemented_properties = ["energy", "energies", "free_energy", "forces"]

    def __init__(
        self,
        model_file,
        *,
        dtype: str = "float32",
        neighbor_skin: float = 0.5,
        streamed_edges: str = "direct",
        execution_profile: str = "capacity",
        head: str | None = None,
        **kwargs,
    ):
        if "use_kokkos" in kwargs:
            raise TypeError(
                "use_kokkos has been retired: NequIPCalculator always executes "
                "through the single Kokkos evaluator stack. Pass "
                "streamed_edges='direct' for the generated receiver routing or "
                "streamed_edges='reference' for single-thread generic execution "
                "on the default host execution space."
            )
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if execution_profile not in {"capacity", "speed"}:
            raise ValueError("execution_profile must be 'capacity' or 'speed'")
        if head is not None and not isinstance(head, str):
            raise TypeError("head must be a string or None")
        if isinstance(neighbor_skin, bool) or not isinstance(neighbor_skin, Real):
            raise ValueError("neighbor_skin must be a finite non-negative number")
        try:
            skin_value = float(neighbor_skin)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "neighbor_skin must be a finite non-negative number"
            ) from error
        if not np.isfinite(skin_value) or skin_value < 0:
            raise ValueError("neighbor_skin must be a finite non-negative number")
        path = Path(model_file)
        if path.suffix.lower() != ".json":
            raise ValueError(
                "NequIPCalculator requires an exported Symmetrix NequIP JSON file; "
                "export .nequip/.pt2 artifacts first"
            )
        self.model_file = path
        self.dtype = dtype
        self.neighbor_skin = skin_value
        self.execution_profile = execution_profile
        self._configuration_diagnostics: list[str] = []
        self._head = head
        self._graph_key = None
        self._graph = None
        self._reference_positions = None
        self._prepared_key = None
        self._prepared_generation = None
        self._last_execution_diagnostic = "uninitialized"
        self._oam_artifact_diagnostic = ""
        self._oam_cuda_artifact_diagnostic = ""
        super().__init__(**kwargs)

        # The architecture marker is this family's detection key, so the
        # frontend reads it from the contract file itself: the family it
        # detects is what decides which declared schema may load that file.  The
        # contract is then loaded and validated by the family's own schema
        # inside the native evaluator and inside the declared artifact adapters,
        # so the frontend never parses it a second time.
        try:
            with path.open("r", encoding="utf-8") as stream:
                architecture = str(json.load(stream).get("architecture", ""))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read NequIP JSON contract: {path}") from error
        family = family_registry.declared_registry().detect(architecture=architecture)
        if family is None:
            raise ValueError("ordinary NequIP v1 contracts are no longer supported")
        self._family = family
        # The execution-mode vocabulary is the detected family's declaration, so
        # the accepted spellings and the rejection message are the same source
        # and neither can drift from the routings the frontend resolves.
        vocabulary = family.modes
        resolved_mode = vocabulary.canonical_mode(streamed_edges)
        if resolved_mode not in vocabulary.canonical_modes:
            raise ValueError(_mode_rejection(vocabulary))
        routing = family_registry.routing_of(family, resolved_mode)
        if routing is None:
            raise ValueError(
                f"family {family.key!r} declares no {resolved_mode!r} routing"
            )
        self._routing = routing
        self.streamed_edges = routing.key
        self.streamed_edges_requested = streamed_edges
        self.streamed_edges_alias = (
            streamed_edges if streamed_edges in vocabulary.alias_spellings else None
        )
        native = load_backend()
        if not native._kokkos_is_initialized():
            native._init_kokkos()
        direct = routing.key == "direct"
        # Resolve the backend for every routing so an unsupported HIP build is
        # rejected before either the generated or reference evaluator can run.
        target, target_argument = _execution_artifact_target(native)
        adapter = None
        artifact = None
        if direct:
            # The direct routing runs the prepared receiver lifecycle, so it
            # prepares the artifact its family declares for this target; the
            # reference routing reaches the whole-evaluator entry point with no
            # artifact at all.
            adapter, artifact = self._prepare_direct_artifact(
                family, routing, path, target, target_argument, native
            )
        evaluator_type = family_registry.resolve_evaluator_class(
            family,
            model_type=None,
            dtype=self.dtype,
            native=native,
        )
        self.evaluator = self._construct_evaluator(
            evaluator_type, path, adapter, artifact
        )
        if adapter is not None:
            self._record_artifact_diagnostic(adapter, artifact)
        (set_streamed_edges,) = family_registry.require_surface(
            family, self.evaluator, "mode_handshake"
        )
        set_streamed_edges(self._routing.native_mode)
        set_execution_plan_request = getattr(
            self.evaluator, "_set_execution_plan_request", None
        )
        if not callable(set_execution_plan_request):
            raise RuntimeError(
                "the OAM evaluator does not implement the execution-plan request handshake"
            )
        set_execution_plan_request(self._routing.native_mode, execution_profile)
        self.implemented_properties = list(type(self).implemented_properties)
        if self._routing.reduces_stress and bool(
            getattr(self.evaluator, "supports_stress", False)
        ):
            self.implemented_properties.append("stress")

    def _prepare_direct_artifact(
        self, family, routing, path, target, target_argument, native_module
    ):
        """Prepare the artifact the ``direct`` routing executes through.

        The entry point, its result's library attribute, and the keywords the
        evaluator accepts the artifact under are the family's declaration, so
        the frontend prepares and hands over an artifact without knowing any
        adapter's own result type.  A routing that declares no adapter for this
        target cannot be prepared here, so the request fails instead of running
        a different implementation; a preparation that cannot produce a
        qualified artifact fails with the execution layer's four-way artifact
        diagnostic.
        """

        adapter = family_registry.artifact_adapter_of(
            family, routing=routing.key, target=target
        )
        if adapter is None:
            raise RuntimeError(
                f"family {family.key!r} declares no {routing.key!r} artifact "
                f"adapter for target {target!r}; the request cannot be prepared"
            )
        prepare = family_registry.resolve_symbol(adapter.prepare)
        return adapter, prepare(
            path,
            precision=self.dtype,
            target=target_argument,
            native_module=native_module,
        )

    def _construct_evaluator(self, evaluator_type, path, adapter, artifact):
        """Construct the family's evaluator, handing it the prepared artifact.

        An adapter that declares no evaluator keyword prepares the artifact the
        evaluator takes as its second positional argument.  An adapter that
        declares one prepares an artifact the evaluator takes under that
        keyword, together with the identity its manifest publishes, so the
        frontend never reconstructs an artifact's identity itself.
        """

        if adapter is None:
            return evaluator_type(str(path))
        library = str(getattr(artifact, adapter.artifact_path))
        if adapter.evaluator_keyword is None:
            return evaluator_type(
                str(path),
                library,
                host_active_target=getattr(artifact, "active_host_target", ""),
                host_artifact_origin=getattr(artifact, "artifact_origin", ""),
            )
        arguments = {adapter.evaluator_keyword: library}
        if adapter.identity_keyword is not None:
            arguments[adapter.identity_keyword] = {
                "manifest_path": str(getattr(artifact, adapter.manifest_path))
            }
        return evaluator_type(
            str(path),
            "",
            host_artifact_origin=getattr(artifact, "artifact_origin", ""),
            **arguments,
        )

    def _record_artifact_diagnostic(self, adapter, artifact):
        """Record the prepared artifact and the evaluator's own device reason."""

        diagnostic = (
            f"{adapter.target}_artifact={artifact.artifact_id}, "
            f"cache_status={artifact.status}"
        )
        if adapter.target == "cuda":
            self._oam_cuda_artifact_diagnostic = (
                f"{diagnostic}, compiler={artifact.compiler}"
            )
            reason = str(getattr(self.evaluator, "cuda_r1_reason", "") or "")
            if reason:
                self._configuration_diagnostics.append(reason)
            return
        self._oam_artifact_diagnostic = diagnostic

    @property
    def cutoff(self) -> float:
        return float(self.evaluator.r_cut)

    @property
    def atomic_numbers(self):
        return tuple(int(value) for value in self.evaluator.atomic_numbers)

    @property
    def available_heads(self):
        return tuple(self.evaluator.available_heads)

    @property
    def capabilities(self):
        """The capability surface this evaluator's family declares.

        The mapping is the evaluator's own readiness query, read through the
        family's declaration, so a consumer sees which tiers, derivative
        properties, and artifact classes this build serves without probing the
        evaluator itself.  An evaluator that reports no mapping yields an empty
        mapping.
        """

        reported = family_registry.capabilities_of(self._family, self.evaluator)
        return dict(reported) if reported is not None else {}

    @property
    def execution_plan(self):
        """The selected plan and the tier bound to every operator node.

        The report is the native plan report of this evaluator's compiled
        program: which implementation the execution path dispatches, and the
        per-node bindings that state which tier serves each operator and which
        implementation it dispatches.  A report is required, so an evaluator
        that cannot produce one is an error rather than a fabricated plan.
        """

        report = getattr(self.evaluator, "execution_plan_report", None)
        if isinstance(report, Mapping):
            return dict(report)
        raise RuntimeError(
            "the OAM evaluator does not report an execution plan; the native "
            "report is required to state the per-node implementation tiers"
        )

    @property
    def execution_diagnostic(self) -> str:
        return "; ".join(
            item
            for item in (
                *self._configuration_diagnostics,
                self._last_execution_diagnostic,
            )
            if item
        )

    def _build_graph(self, atoms):
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        type_map = {number: index for index, number in enumerate(self.atomic_numbers)}
        unknown = sorted(set(numbers.tolist()) - set(type_map))
        if unknown:
            raise ValueError(
                f"model does not support atomic numbers {unknown}; supported: {list(self.atomic_numbers)}"
            )
        node_types = np.asarray(
            [type_map[int(number)] for number in numbers], dtype=np.int32
        )
        graph = build_host_neighbor_graph(atoms, self.cutoff + self.neighbor_skin)
        cell = np.asarray(atoms.cell.array, dtype=float).reshape((3, 3))
        positions = np.asarray(atoms.get_positions(), dtype=float)
        xyz = graph.edge_vectors(positions, cell)
        distances = np.linalg.norm(xyz, axis=1)
        return (
            node_types,
            graph.num_neigh,
            graph.sources,
            graph.neighbor_types(node_types),
            graph.receivers,
            graph.shifts,
            xyz,
            distances,
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if atoms is None:
            atoms = self.atoms
        super().calculate(atoms, properties, system_changes)
        if "stress" in properties and "stress" not in self.implemented_properties:
            raise RuntimeError("NequIPCalculator evaluator does not support stress")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = np.asarray(atoms.cell.array, dtype=float)
        topology_key = (
            cell.tobytes(),
            np.asarray(atoms.pbc).tobytes(),
            tuple(atoms.get_atomic_numbers()),
        )
        rebuild = self._graph is None or topology_key != self._graph_key
        if not rebuild and self._reference_positions is not None:
            displacement = np.max(
                np.linalg.norm(positions - self._reference_positions, axis=1),
                initial=0.0,
            )
            rebuild = not skin_reuse_holds(displacement, self.neighbor_skin)
        if rebuild:
            self._graph = self._build_graph(atoms)
            self._graph_key = topology_key
            self._reference_positions = np.array(positions, copy=True)
        else:
            node_types, num_neigh, sources, neigh_types, receivers, shifts, _, _ = (
                self._graph
            )
            xyz = edge_vectors(positions, cell, receivers, sources, shifts)
            distances = np.linalg.norm(xyz, axis=1)
            self._graph = (
                node_types,
                num_neigh,
                sources,
                neigh_types,
                receivers,
                shifts,
                xyz,
                distances,
            )
        key = (topology_key, positions.tobytes())
        if key == self._prepared_key and self.results:
            if "stress" in properties and "stress" not in self.results:
                self._reduce_stress(atoms)
            return
        (
            node_types,
            num_neigh,
            sources,
            neigh_types,
            receivers,
            shifts,
            xyz,
            distances,
        ) = self._graph
        n = len(atoms)
        generation = None
        if self._routing.key == "reference":
            (compute_reduced,) = family_registry.require_surface(
                self._family, self.evaluator, self._routing.surface
            )
            compute_reduced(
                n,
                node_types,
                num_neigh,
                sources,
                neigh_types,
                xyz.reshape(-1),
                distances,
            )
            edge_forces = np.asarray(self.evaluator.node_forces, dtype=float).reshape(
                (-1, 3)
            )
            forces = np.zeros((n, 3), dtype=float)
            for edge, (receiver, source) in enumerate(zip(receivers, sources)):
                forces[source] += edge_forces[edge]
                forces[receiver] -= edge_forces[edge]
        else:
            prepare_graph, prepare_geometry, compute_prepared, reduce_forces = (
                family_registry.require_surface(
                    self._family, self.evaluator, self._routing.surface
                )
            )
            generation = prepare_graph(n, node_types, num_neigh, sources, neigh_types)
            periodic = bool(np.any(atoms.pbc))
            if periodic:
                inverse = np.linalg.inv(cell)
                pbc = np.asarray(atoms.pbc, dtype=np.int32)
                cell_arg, inverse_arg = cell.reshape(-1), inverse.reshape(-1)
            else:
                pbc = np.empty(0, dtype=np.int32)
                cell_arg = inverse_arg = np.empty(0, dtype=float)
            prepare_geometry(
                generation,
                positions.reshape(-1),
                xyz.reshape(-1),
                cell_arg,
                inverse_arg,
                pbc,
            )
            compute_prepared(generation, xyz.reshape(-1), distances)
            reduce_forces(generation)
            forces = np.asarray(self.evaluator.atom_forces, dtype=float).reshape((n, 3))
        per_atom = np.asarray(self.evaluator.node_energies, dtype=float).reshape((n,))
        self.results = {
            "energies": per_atom,
            "energy": float(per_atom.sum()),
            "free_energy": float(per_atom.sum()),
            "forces": forces,
        }
        self._prepared_generation = generation
        if "stress" in properties:
            self._reduce_stress(atoms)
        native_diagnostic = str(
            getattr(self.evaluator, "last_execution_diagnostic", "serial")
        )
        self._last_execution_diagnostic = "; ".join(
            item
            for item in (
                self._oam_artifact_diagnostic,
                self._oam_cuda_artifact_diagnostic,
                native_diagnostic,
            )
            if item
        )
        self._prepared_key = key

    def _reduce_stress(self, atoms):
        self.evaluator.reduce_stress(atoms.get_volume(), self._prepared_generation)
        tensor = np.asarray(self.evaluator.stress_tensor, dtype=float).reshape((3, 3))
        self.results["stress"] = full_3x3_to_voigt_6_stress(tensor)

    def get_stress(self, atoms=None):
        if "stress" not in self.implemented_properties:
            raise RuntimeError("NequIPCalculator evaluator does not support stress")
        return super().get_stress(atoms)
