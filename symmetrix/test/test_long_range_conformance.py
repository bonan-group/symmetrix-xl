"""Long-range operator group conformance: the probe-surface admission battery.

The v6 family conformance suite (``test_family_conformance.py``) admits a
model family through its adapter lifecycle.  The long-range operator group
(``libsymmetrix/source/long_range_operator_kinds.hpp``) has no family yet --
the PolarMACE adapter is gated -- so the group enters the conformance
machinery the way the v6 suite admits everything below a family: through the
probe/registry surface, compiler-free, with the committed golden fixtures as
the reference.  One admitted instance per fixture geometry (``mol5`` the
molecule graph, ``pbc4`` the bulk graph), and five proofs per instance:

* **Tier parity per operator kind.**
  ``test_tier_parity_per_operator_kind`` asserts the portable (T1)
  implementation of each of the nine declared kinds against the reference,
  parametrized per geometry and spin channel.  The reference is the golden
  fixture (the frozen oracle stack's own values) wherever the fixtures serve
  the kind; for the pieces they do not serve (the Fukui normalization, the
  dipole readout, the external-field gauge forward) it is the NumPy mirror
  of the upstream formula -- the same reference ``test_long_range_kernels``
  pins, formalized here with the battery's attribution machinery instead of
  ad hoc.
* **Injected-perturbation red evidence.**
  ``test_injected_perturbation_breaks_parity_with_attribution`` injects one
  perturbation at a time (positions, cell, multipoles, basis constants) and
  asserts the parity verdict for every (kind, perturbation) pair against a
  declared sensitivity matrix: a kind that consumes the perturbed buffer
  FAILS parity with the kind named in the failure, and a kind whose declared
  inputs were not touched holds parity -- attribution is specific, not a
  global blowout.  Task 9 grows the matrix by a column family rather than a
  new battery (Task 8's report note): five ``adjoint_seed_*`` columns
  perturb the seeds of the reverse-mode adjoint surfaces, whose rows mirror
  their forward kinds' input sensitivity plus their own seed column -- so a
  red verdict on an adjoint surface names exactly its seed or its declared
  inputs, and no forward kind reacts to any adjoint seed.  The adjoint
  surfaces' numerical correctness is not this battery's concern (it is the
  finite-difference gate of ``test_long_range_adjoints.py``); here they are
  attribution surfaces consuming exactly their declared inputs.
* **Capacity report fields.**  ``test_capacity_report_fields_validate``
  emits the task-3 manifest vocabulary (``n_k``, ``table_bytes``, the
  correction mode, the rebuild trigger) for the fixture instances from the
  cache's own k-point count and asserts the emitted record validates; the
  garbage spellings reject.  The native constructor validates at construction
  (the Task-6 deferred asymmetry, resolved: ``make_long_range_capacity``
  throws on invalid data exactly as the node builder refuses invalid contract
  data; the extended C++ probe exercises the refusal).
* **Layout admission.**  ``k_major`` and ``graph_major`` -- the layout
  orders Task 6 added to the vocabulary -- are exercised on data: the
  k-major buffers realize their leading k extent with rows grouped by graph,
  the graph-major buffers realize their leading graph extent, the cross-axis
  scatter (graph volumes onto k rows) and reduction (k rows into graph
  energies) hold parity, and the multi-graph spelling splits cleanly.  The
  program DAG now declares the long-range nodes, so the layout and shape
  authority is the compiled program's own buffer contracts
  (``symmetrix.polar_long_range_program``) rather than a parallel table: the
  battery reads the declared orders, extents, and adjoint seed columns from
  the program, and the fp32-table measurement that backs the declared table
  dtype is pinned here too.
* **Cache sharing.**  ``test_one_geometry_cache_serves_every_consumer``
  asserts the one-geometry-cache-per-step contract at the binding level the
  probe exposes: one cache build serves both spin channels and the
  feature/energy consumers (the four GEMM consumers), bit-stably, in any
  consumer order.  The probe already expresses sharing -- the cache is an
  explicit first-class argument of every stage method and rides in the build
  result -- so no probe extension was needed; the node-graph spelling of the
  same rule (identically-named identically-shaped cache inputs on the
  features and energy nodes) is pinned by the C++ registry probe.

Tolerance: fp64 parity at 1e-12 (the fixtures are fp64; the observed GEMM
chain noise is ~1e-14), the same gate ``test_long_range_kernels.py`` holds.
"""

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import symmetrix

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

from symmetrix.polar_long_range_kinds import (
    POLAR_LONG_RANGE_CORRECTION_MODES,
    POLAR_LONG_RANGE_KINDS,
    PolarLongRangeContract,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "polar"
GEOMETRIES = ("mol5", "pbc4")
CHANNELS = ("alpha", "beta")
ENERGY_LABELS = ("alpha", "beta", "total")

# fp64 parity against the golden fixtures: relative 1e-12 with an absolute
# floor covering exact zeros (the observed chain noise is ~1e-14).
RTOL = 1e-12
ATOL = 1e-12

# The layout orders the long-range group admits.  The family suite's
# LAYOUT_ORDERS carries the shared-tensor orders (node/lane/edge/block);
# Task 6's declaration adds k_major (one row per half-sphere k-point,
# flattened over graphs) and graph_major (one row per graph) for this group,
# and this battery is their conformance admission.
FAMILY_LAYOUT_ORDERS = frozenset(
    ("node_major", "lane_major", "edge_major", "block_major")
)
LONG_RANGE_LAYOUTS = frozenset(("node_major", "k_major", "graph_major"))

# The correction mode a graph class selects, in the manifest vocabulary.
GRAPH_CLASS_MODES = {
    "molecule": "makov-payne-dabo",
    "slab": "slab-dipole",
    "bulk": "none",
}

# The injected perturbations: one input family each, sized to move outputs far
# beyond the parity tolerance.  The ``adjoint seeds`` family is Task 9's
# column-family growth of the matrix (Task 8's report note): each column
# perturbs the seed of one adjoint surface, so the adjoint rows below break
# exactly on their own seed while every forward row and every other adjoint
# row holds -- attribution extends across the derivative surface.
PERTURBATIONS = (
    "positions",
    "cell",
    "multipoles",
    "constants",
    "adjoint_seed_features",
    "adjoint_seed_energy",
    "adjoint_seed_self_terms",
    "adjoint_seed_external_field",
    "adjoint_seed_dipole",
    "adjoint_seed_stress",
    "adjoint_seed_fukui",
    "adjoint_seed_correction_energy",
    "adjoint_seed_correction_features",
)

ADJOINT_SURFACES = (
    "kspace_coulomb_features_adjoint",
    "kspace_coulomb_energy_adjoint",
    "self_interaction_adjoint",
    "external_field_projection_adjoint",
    "dipole_readout_adjoint",
    "kspace_coulomb_energy_stress",
    # The deferred adjoints of the family-wiring stage: the global
    # scatter-and-broadcast (Fukui) adjoint and the analytic corrections'
    # position adjoints, each with its own seed column.
    "fukui_normalize_adjoint",
    "analytic_correction_energy_adjoint",
    "analytic_correction_features_adjoint",
)

# The reverse-mode adjoint surfaces the battery admits: each mirrors a
# forward kind's input surface (its position gradient and any co-emitted
# seed products), evaluated through the same probe surface convention --
# no evaluator wiring, no new kind declarations.  The stress surface (the
# experimental cell adjoint's assembly) is Task 10's column-family growth:
# it consumes the energy kind's inputs plus its own seed and the
# positions of the position virial.


# The sensitivity matrix the red-evidence battery asserts: the perturbations
# that must BREAK each kind's parity (the kind consumes the perturbed
# buffer).  Every other (kind, perturbation) pair must HOLD parity, which is
# the attribution guarantee: a kind fails exactly when its declared inputs
# were perturbed, so a failure names the right kind.  The adjoint rows
# mirror their forward kinds' input sensitivity plus their own seed column;
# the one deliberate divergence is the self-interaction adjoint, whose
# outputs are constant-linear in the seeds alone -- multipoles move its
# forward but not its adjoint.
PERTURBATION_SENSITIVITY = {
    "kspace_geometry_cache": frozenset(("positions", "cell", "constants")),
    "kspace_coulomb_features": frozenset(
        ("positions", "cell", "multipoles", "constants")
    ),
    "kspace_coulomb_energy": frozenset(
        ("positions", "cell", "multipoles", "constants")
    ),
    "self_interaction_linear": frozenset(("multipoles", "constants")),
    "external_field_projection": frozenset(("positions", "constants")),
    "fukui_normalize": frozenset(("multipoles",)),
    "dipole_readout": frozenset(("positions", "multipoles")),
    "analytic_correction_energy": frozenset(("positions", "cell", "multipoles")),
    "analytic_correction_features": frozenset(
        ("positions", "cell", "multipoles", "constants")
    ),
    "kspace_coulomb_features_adjoint": frozenset(
        ("positions", "cell", "multipoles", "constants", "adjoint_seed_features")
    ),
    "kspace_coulomb_energy_adjoint": frozenset(
        ("positions", "cell", "multipoles", "constants", "adjoint_seed_energy")
    ),
    "self_interaction_adjoint": frozenset(("constants", "adjoint_seed_self_terms")),
    "external_field_projection_adjoint": frozenset(
        ("positions", "constants", "adjoint_seed_external_field")
    ),
    "dipole_readout_adjoint": frozenset(
        ("positions", "multipoles", "adjoint_seed_dipole")
    ),
    # The Fukui adjoint consumes the sources, the l=0 charge channels, the
    # graph-level charge/spin, and its own seeds -- never positions or the
    # cell (the normalization has no geometry dependence).
    "fukui_normalize_adjoint": frozenset(("multipoles", "adjoint_seed_fukui")),
    "analytic_correction_energy_adjoint": frozenset(
        ("positions", "cell", "multipoles", "adjoint_seed_correction_energy")
    ),
    "analytic_correction_features_adjoint": frozenset(
        (
            "positions",
            "cell",
            "multipoles",
            "constants",
            "adjoint_seed_correction_features",
        )
    ),
    "kspace_coulomb_energy_stress": frozenset(
        ("positions", "cell", "multipoles", "constants", "adjoint_seed_stress")
    ),
}


# The channel each kind's perturbation case probes (one is enough: the
# channel axis is data, and the channel-distinct cases below cover the axis).
PROBE_CHANNEL = {
    "kspace_geometry_cache": None,
    "kspace_coulomb_features": "alpha",
    "kspace_coulomb_energy": "total",
    "self_interaction_linear": "alpha",
    "external_field_projection": None,
    "fukui_normalize": "alpha",
    "dipole_readout": "total",
    "analytic_correction_energy": "total",
    "kspace_coulomb_energy_stress": "total",
    "analytic_correction_features": "alpha",
    "kspace_coulomb_features_adjoint": "alpha",
    "kspace_coulomb_energy_adjoint": "alpha",
    "self_interaction_adjoint": "alpha",
    "external_field_projection_adjoint": None,
    "dipole_readout_adjoint": "total",
    "fukui_normalize_adjoint": "alpha",
    "analytic_correction_energy_adjoint": "total",
    "analytic_correction_features_adjoint": "alpha",
}

# Deterministic seeds for the synthetic inputs of the mirror-served kinds
# and the adjoint surfaces' seeds.
_MIRROR_SEEDS = {
    ("mol5", "external_field_projection"): 101,
    ("pbc4", "external_field_projection"): 102,
    ("mol5", "fukui_normalize"): 201,
    ("pbc4", "fukui_normalize"): 202,
    ("mol5", "kspace_coulomb_features_adjoint"): 301,
    ("pbc4", "kspace_coulomb_features_adjoint"): 302,
    ("mol5", "kspace_coulomb_energy_adjoint"): 311,
    ("pbc4", "kspace_coulomb_energy_adjoint"): 312,
    ("mol5", "self_interaction_adjoint"): 321,
    ("pbc4", "self_interaction_adjoint"): 322,
    ("mol5", "external_field_projection_adjoint"): 331,
    ("pbc4", "external_field_projection_adjoint"): 332,
    ("mol5", "dipole_readout_adjoint"): 341,
    ("pbc4", "dipole_readout_adjoint"): 342,
    ("mol5", "kspace_coulomb_energy_stress"): 351,
    ("pbc4", "kspace_coulomb_energy_stress"): 352,
    ("mol5", "fukui_normalize_adjoint"): 361,
    ("pbc4", "fukui_normalize_adjoint"): 362,
    ("mol5", "analytic_correction_energy_adjoint"): 371,
    ("pbc4", "analytic_correction_energy_adjoint"): 372,
    ("mol5", "analytic_correction_features_adjoint"): 381,
    ("pbc4", "analytic_correction_features_adjoint"): 382,
}


# ---------------------------------------------------------------------------
# The admitted instances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LongRangeInstance:
    """One admitted fixture geometry: manifest, arrays, contract, and ops.

    ``parameters`` states the instance's shape (graph class, node and k-point
    counts, basis) so a failure names the instance rather than the group.
    """

    key: str
    graph_class: str
    manifest: dict
    arrays: dict
    contract: PolarLongRangeContract
    ops: symmetrix.PolarLongRangeOps

    @property
    def n_nodes(self) -> int:
        return int(self.manifest["geometry"]["n_nodes"])

    @property
    def n_graphs(self) -> int:
        return int(self.arrays["cell"].shape[0])

    @property
    def periodic(self) -> bool:
        return bool(self.manifest["geometry"]["periodic"])

    def batch(self) -> np.ndarray:
        return self.arrays["batch"].astype(np.int32)

    def pbc(self) -> np.ndarray:
        return self.arrays["pbc"].astype(np.int32)

    def source(self, label: str) -> np.ndarray:
        return self.arrays[f"source_feats_{label}"]


def _load_instance(key: str) -> LongRangeInstance:
    manifest = json.loads((DATA_DIR / f"{key}_manifest.json").read_text())
    arrays = dict(np.load(DATA_DIR / f"{key}_arrays.npz"))
    parameters = manifest["parameters"]
    density_width = (parameters["density_max_l"] + 1) ** 2
    field_width = (parameters["feature_max_l"] + 1) ** 2 * len(
        parameters["feature_smearing_widths"]
    )
    contract = PolarLongRangeContract(
        kspace_cutoff=parameters["kspace_cutoff"],
        density_max_l=parameters["density_max_l"],
        density_smearing_width=parameters["density_smearing_width"],
        feature_max_l=parameters["feature_max_l"],
        feature_smearing_widths=tuple(parameters["feature_smearing_widths"]),
        charges_dimension=2 * density_width,
        field_dimension=field_width,
        potential_dimension=2 * field_width,
    )
    contract.validate()
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=parameters["kspace_cutoff"],
        density_max_l=parameters["density_max_l"],
        density_smearing_width=parameters["density_smearing_width"],
        feature_max_l=parameters["feature_max_l"],
        feature_smearing_widths=list(parameters["feature_smearing_widths"]),
    )
    return LongRangeInstance(
        key=key,
        graph_class=manifest["geometry"]["graph_class"],
        manifest=manifest,
        arrays=arrays,
        contract=contract,
        ops=ops,
    )


@pytest.fixture(params=GEOMETRIES)
def instance(request, instances):
    return instances[request.param]


# ---------------------------------------------------------------------------
# The kind-parity driver: one forward per kind, comparisons against the
# reference, perturbable inputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def instances():
    """One admitted instance per geometry, released before Kokkos finalizes.

    The ops objects own Kokkos allocations (the constant matrices), so the
    module-scoped owners drop them at teardown and collect -- the same
    release-the-native-owners idiom the family suite's ``adapters`` fixture
    uses, so nothing deallocates after the session Kokkos finalizer.
    """
    built = {key: _load_instance(key) for key in GEOMETRIES}
    yield built
    for native_owner in built.values():
        object.__setattr__(native_owner, "ops", None)
    built.clear()
    gc.collect()


def _assert_parity(got, want, context: str) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, f"{context}: shape {got.shape} != {want.shape}"
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL, err_msg=context)


def contract_depth(instance: LongRangeInstance) -> int:
    """The recursion depth the instance's contract declares."""

    return instance.contract.recursion_step_count


def program_slice(instance: LongRangeInstance) -> dict:
    """The long-range half of the PolarMACE program for one fixture instance.

    Compiled from the instance's contract data alone (no model, no backend),
    which is what makes the program's node contracts readable here: the
    buffer contracts, the tiered binding report, and the unrolled schedule are
    the same values the family's evaluator compiles its program with.
    """

    contract = instance.contract
    return symmetrix.polar_long_range_program(
        kspace_cutoff=contract.kspace_cutoff,
        density_max_l=contract.density_max_l,
        density_smearing_width=contract.density_smearing_width,
        feature_max_l=contract.feature_max_l,
        feature_smearing_widths=list(contract.feature_smearing_widths),
        charges_dimension=contract.charges_dimension,
        field_dimension=contract.field_dimension,
        potential_dimension=contract.potential_dimension,
        field_norm_factor=contract.field_norm_factor,
        recursion_step_count=contract.recursion_step_count,
        table_dtype=contract.table_dtype,
    )


def _perturbed_geometry(instance, perturbations):
    """The (positions, cell, ops) inputs with the named perturbations applied.

    ``positions`` shifts every node by a distinct deterministic offset (a
    uniform shift would leave the translation-invariant gauges untouched),
    ``cell`` scales the cells, ``constants`` rebuilds the ops with perturbed
    basis parameters (new analytic Fourier factors and constant matrices).
    """

    arrays = instance.arrays
    positions = np.array(arrays["node_positions"], dtype=np.float64)
    cell = np.array(arrays["cell"], dtype=np.float64)
    if "positions" in perturbations:
        n_nodes = len(positions)
        ramp = 0.05 * np.sin(
            np.arange(3 * n_nodes, dtype=np.float64).reshape(n_nodes, 3) + 1.0
        )
        positions = positions + ramp
    if "cell" in perturbations:
        cell = cell * 1.03
    ops = instance.ops
    if "constants" in perturbations:
        parameters = instance.manifest["parameters"]
        ops = symmetrix.PolarLongRangeOps(
            kspace_cutoff=parameters["kspace_cutoff"],
            density_max_l=parameters["density_max_l"],
            density_smearing_width=parameters["density_smearing_width"] * 1.05,
            feature_max_l=parameters["feature_max_l"],
            feature_smearing_widths=[
                width * 1.05 for width in parameters["feature_smearing_widths"]
            ],
        )
    return positions, cell, ops


def _kind_source(instance, label, perturbations):
    """The per-channel multipoles, perturbed when the case says so."""
    source = np.array(instance.source(label), dtype=np.float64)
    if "multipoles" in perturbations:
        source = source + 0.13
    return source


def _two_pi_cubed() -> float:
    return (2.0 * np.pi) ** 3


def _fukui_reference(
    instance, fukui_sources, charge_l0_alpha, charge_l0_beta, total_charge, total_spin
):
    """The NumPy mirror of the upstream Fukui normalization."""
    graph_of = instance.arrays["batch"]
    norm = np.stack(
        [fukui_sources[graph_of == g].sum(axis=0) for g in range(instance.n_graphs)]
    )
    norm = np.where(norm == 0, 1.0, norm)
    f_normed = fukui_sources / norm[graph_of]
    q_p_s = (total_charge + (total_spin - 1.0))[graph_of]
    q_m_s = (total_charge - (total_spin - 1.0))[graph_of]
    pred_alpha = np.stack(
        [charge_l0_alpha[graph_of == g].sum() for g in range(instance.n_graphs)]
    )[graph_of]
    pred_beta = np.stack(
        [charge_l0_beta[graph_of == g].sum() for g in range(instance.n_graphs)]
    )[graph_of]
    return (
        f_normed[:, 0] * ((q_p_s / 2.0) - pred_alpha),
        f_normed[:, 1] * ((q_m_s / 2.0) - pred_beta),
    )


def _dipole_reference(density, positions, graph_of, n_graphs):
    """The NumPy mirror of the upstream dipole readout (e3nn (y, z, x) order)."""
    charge = np.stack([density[graph_of == g, 0].sum() for g in range(n_graphs)])
    dipole = np.stack(
        [
            (density[graph_of == g, 0, None] * positions[graph_of == g]).sum(axis=0)
            + density[graph_of == g, 1:4].sum(axis=0)[[2, 0, 1]]
            for g in range(n_graphs)
        ]
    )
    return charge, dipole


def _external_field_reference(
    matrix, external_potential, positions, graph_of, n_graphs, norms
):
    """The NumPy mirror of the external-field gauge forward.

    Per-graph barycenter (fp64 scatter mean), V += E.(r - barycenter), the
    [V, E_z, E_x, E_y] permutation, the constant-matrix projection, one half
    per spin channel, then the feature-norm divide (``None`` norms mean the
    unit vector).
    """

    barycenter = np.stack(
        [positions[graph_of == g].mean(axis=0) for g in range(n_graphs)]
    )
    displaced = positions - barycenter[graph_of]
    node_fields = external_potential[graph_of].copy()
    node_fields[:, 0] += np.sum(displaced * node_fields[:, 1:], axis=1)
    permuted = node_fields[:, [0, 3, 1, 2]]
    halved = 0.5 * (permuted @ matrix.T)
    if norms is None:
        return halved
    return halved / norms[None, :]


def _kind_comparisons(instance, kind, channel, perturbations=frozenset()):
    """Run one kind's forward; return ``(name, got, want)`` comparisons.

    ``want`` is always the unperturbed reference (fixture values, or the
    mirror evaluated on the unperturbed inputs); the perturbations apply to
    the forward's inputs only.
    """

    arrays = instance.arrays
    positions, cell, ops = _perturbed_geometry(instance, perturbations)
    batch = instance.batch()
    pbc = instance.pbc()
    build = ops.geometry_cache(positions, cell, batch)
    cache = build["cache"]

    if kind == "kspace_geometry_cache":
        return [
            ("k_vectors", build["k_vectors"], arrays["k_vectors"]),
            ("k_norm2", build["k_norm2"], arrays["k_norm2"]),
            ("k_graph_index", build["k_graph_index"], arrays["k_batch"]),
            ("k0_mask", build["k0_mask"], arrays["k0_mask"]),
            ("volume_per_k", build["volume_per_k"], arrays["volume_per_k"]),
            (
                "k_factor_coulomb",
                build["k_factor_coulomb"],
                arrays["k_factor_coulomb"],
            ),
            ("k_factor_proj", build["k_factor_proj"], arrays["k_factor_proj"]),
            ("cos_table", build["cos_table"], arrays["cosines"]),
            ("sin_table", build["sin_table"], arrays["sines"]),
            (
                "density_fourier_factors",
                build["density_fourier_factors"],
                arrays["density_basis_fs"],
            ),
            (
                "feature_fourier_factors",
                build["feature_fourier_factors"],
                arrays["feature_basis_fs"],
            ),
        ]

    if kind == "kspace_coulomb_features":
        source = _kind_source(instance, channel, perturbations)
        density = ops.assemble_fourier(cache, source)
        potential = ops.apply_coulomb(cache, density)
        features = ops.project_to_features(cache, potential)
        return [
            ("rho", density, arrays[f"rho_{channel}"]),
            ("potential", potential, arrays[f"potential_{channel}"]),
            ("features", features, arrays[f"features_pbc_{channel}"]),
        ]

    if kind == "kspace_coulomb_energy":
        source = _kind_source(instance, channel, perturbations)
        kspace = ops.coulomb_energy(cache, source, None)
        node_energy = ops.coulomb_energy(
            cache, source, arrays[f"energy_self_fields_{channel}"]
        )
        # The oracle decomposes energy_final = kspace - 0.5*sum(source *
        # self_fields) + correction, so the node's self-subtracted output is
        # the final energy minus its graph's analytic correction.
        correction = (
            np.zeros(1)
            if instance.periodic
            else arrays[f"energy_molecule_correction_{channel}"]
        )
        return [
            ("energy_kspace", kspace, arrays[f"energy_kspace_{channel}"]),
            (
                "energy_self_subtracted",
                node_energy,
                arrays[f"energy_final_{channel}"] - correction,
            ),
        ]

    if kind == "self_interaction_linear":
        source = _kind_source(instance, channel, perturbations)
        return [
            (
                "si_feature_constants",
                ops.si_feature_constants(),
                arrays["si_overlap_constants_features"],
            ),
            (
                "si_energy_constants",
                ops.si_energy_constants(),
                arrays["si_overlap_constants_energy"],
            ),
            (
                "feature_self_terms",
                ops.feature_self_terms(source),
                arrays[f"features_self_interaction_{channel}"],
            ),
            (
                "energy_self_fields",
                ops.energy_self_fields(source),
                arrays[f"energy_self_fields_{channel}"],
            ),
        ]

    if kind == "external_field_projection":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        n_graphs = instance.n_graphs
        f_dim = instance.contract.field_dimension
        external_potential = rng.normal(size=(n_graphs, 4))
        norms = rng.uniform(0.5, 2.0, size=f_dim)
        # The reference projects through the fixture's constant matrix; the
        # forward uses the (possibly perturbed) ops' own.
        reference_matrix = arrays["external_field_matrix"]
        want_normed = _external_field_reference(
            reference_matrix,
            external_potential,
            arrays["node_positions"],
            arrays["batch"],
            n_graphs,
            norms,
        )
        want_unnormed = _external_field_reference(
            reference_matrix,
            external_potential,
            arrays["node_positions"],
            arrays["batch"],
            n_graphs,
            None,
        )
        return [
            ("matrix", ops.external_field_matrix(), reference_matrix),
            (
                "matrix_internal",
                ops.external_field_matrix(),
                arrays["internal_field_matrix"],
            ),
            (
                "features",
                ops.external_field_features(
                    external_potential, positions, batch, norms
                ),
                want_normed,
            ),
            (
                "features_unnormed",
                ops.external_field_features(external_potential, positions, batch, None),
                want_unnormed,
            ),
        ]

    if kind == "fukui_normalize":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        n_nodes, n_graphs = instance.n_nodes, instance.n_graphs
        fukui_sources = rng.normal(size=(n_nodes, 2))
        total_charge = rng.normal(size=n_graphs)
        total_spin = rng.normal(size=n_graphs)
        # The l=0 channels come from the multipoles, so the multipoles
        # perturbation flows through them.
        charge_alpha = _kind_source(instance, "alpha", perturbations)[:, 0]
        charge_beta = _kind_source(instance, "beta", perturbations)[:, 0]
        want_alpha, want_beta = _fukui_reference(
            instance,
            fukui_sources,
            instance.source("alpha")[:, 0],
            instance.source("beta")[:, 0],
            total_charge,
            total_spin,
        )
        got = ops.fukui_normalize(
            fukui_sources, charge_alpha, charge_beta, total_charge, total_spin, batch
        )
        comparisons = [
            ("charge_correction_alpha", got["charge_correction_alpha"], want_alpha),
            ("charge_correction_beta", got["charge_correction_beta"], want_beta),
        ]
        return [comparisons[0]] if channel == "alpha" else [comparisons[1]]

    if kind == "dipole_readout":
        density = _kind_source(instance, "total", perturbations)
        want_charge, want_dipole = _dipole_reference(
            instance.source("total"),
            arrays["node_positions"],
            arrays["batch"],
            instance.n_graphs,
        )
        got = ops.dipole_readout(density, positions, batch)
        return [
            ("total_charge", got["total_charge"], want_charge),
            ("total_dipole", got["total_dipole"], want_dipole),
        ]

    if kind == "analytic_correction_energy":
        source = _kind_source(instance, channel, perturbations)
        result = ops.analytic_correction_energy(source, positions, cell, batch, pbc)
        comparisons = [
            (
                "molecule_correction",
                result["molecule_correction"],
                arrays[f"energy_molecule_correction_{channel}"],
            ),
            (
                "slab_correction",
                result["slab_correction"],
                arrays[f"energy_slab_correction_{channel}"],
            ),
        ]
        # Mode selection from the graph's pbc row: molecule graphs take the
        # box term, bulk graphs take nothing.
        if instance.graph_class == "molecule":
            comparisons.append(
                (
                    "selected",
                    result["correction_energy"],
                    result["molecule_correction"],
                )
            )
        else:
            comparisons.append(
                ("selected_zero", result["correction_energy"], np.zeros(1))
            )
        return comparisons

    if kind == "analytic_correction_features":
        # Channel-distinct inputs: each channel's corrective fields follow
        # its own channel density (the correction is linear in it).
        alpha_density = _kind_source(instance, "alpha", perturbations)
        beta_density = _kind_source(instance, "beta", perturbations)
        result = ops.analytic_correction_features(
            alpha_density, beta_density, positions, cell, batch, pbc
        )
        return [
            (
                "correction_fields_alpha",
                result["correction_fields_alpha"],
                arrays["features_correction_alpha"],
            ),
            (
                "correction_fields_beta",
                result["correction_fields_beta"],
                arrays["features_correction_beta"],
            ),
        ]

    # --- the adjoint surfaces ---------------------------------------------
    # want is always the unperturbed adjoint (the same probe surface,
    # unperturbed inputs and seed); the perturbations apply to the got
    # side's inputs and seed.  The composed position gradients add the
    # table seeds of every path before adjoint_geometry fires -- the
    # shared-cache chain rule the FD battery of
    # ``test_long_range_adjoints.py`` pins numerically; here it is the
    # attribution surface.
    if kind == "kspace_coulomb_features_adjoint":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        grad_features = rng.normal(size=arrays["features_pbc_alpha"].shape)
        base_seed = np.array(grad_features)
        if "adjoint_seed_features" in perturbations:
            grad_features = grad_features * 1.5 + 0.2
        source = _kind_source(instance, "alpha", perturbations)

        def features_position_gradient(ops_, positions_, cell_, source_, seed_):
            build_ = ops_.geometry_cache(positions_, cell_, batch)
            cache_ = build_["cache"]
            potential_ = ops_.apply_coulomb(
                cache_, ops_.assemble_fourier(cache_, source_)
            )
            back_ = ops_.adjoint_project(cache_, seed_, potential_)
            grad_density_ = ops_.adjoint_coulomb(cache_, back_["grad_potential"])
            assembled_ = ops_.adjoint_assemble(cache_, grad_density_, source_)
            return ops_.adjoint_geometry(
                cache_,
                back_["grad_cos_table"] + assembled_["grad_cos_table"],
                back_["grad_sin_table"] + assembled_["grad_sin_table"],
            )

        return [
            (
                "grad_positions",
                features_position_gradient(ops, positions, cell, source, grad_features),
                features_position_gradient(
                    instance.ops,
                    arrays["node_positions"],
                    arrays["cell"],
                    instance.source("alpha"),
                    base_seed,
                ),
            )
        ]

    if kind == "kspace_coulomb_energy_adjoint":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        grad_energy = rng.normal(size=instance.n_graphs)
        base_seed = np.array(grad_energy)
        if "adjoint_seed_energy" in perturbations:
            grad_energy = grad_energy * 1.5 + 0.2
        source = _kind_source(instance, "alpha", perturbations)

        def energy_position_gradient(ops_, positions_, cell_, source_, seed_):
            build_ = ops_.geometry_cache(positions_, cell_, batch)
            cache_ = build_["cache"]
            back_ = ops_.adjoint_coulomb_energy(
                cache_, seed_, source_, arrays["energy_self_fields_alpha"]
            )
            return ops_.adjoint_geometry(
                cache_, back_["grad_cos_table"], back_["grad_sin_table"]
            )

        return [
            (
                "grad_positions",
                energy_position_gradient(ops, positions, cell, source, grad_energy),
                energy_position_gradient(
                    instance.ops,
                    arrays["node_positions"],
                    arrays["cell"],
                    instance.source("alpha"),
                    base_seed,
                ),
            )
        ]

    if kind == "self_interaction_adjoint":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        grad_terms = rng.normal(size=arrays["features_self_interaction_alpha"].shape)
        base_terms = np.array(grad_terms)
        grad_fields = rng.normal(size=arrays["energy_self_fields_alpha"].shape)
        base_fields = np.array(grad_fields)
        if "adjoint_seed_self_terms" in perturbations:
            grad_terms = grad_terms * 1.5 + 0.2
            grad_fields = grad_fields * 1.5 + 0.2
        # Constant-linear: the adjoint consumes only the seed and the
        # constants -- the (possibly perturbed) multipoles cannot move it.
        return [
            (
                "grad_multipoles_features",
                ops.adjoint_feature_self_terms(grad_terms),
                instance.ops.adjoint_feature_self_terms(base_terms),
            ),
            (
                "grad_multipoles_energy",
                ops.adjoint_energy_self_fields(grad_fields),
                instance.ops.adjoint_energy_self_fields(base_fields),
            ),
        ]

    if kind == "external_field_projection_adjoint":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        n_graphs = instance.n_graphs
        f_dim = instance.contract.field_dimension
        external_potential = rng.normal(size=(n_graphs, 4))
        norms = rng.uniform(0.5, 2.0, size=f_dim)
        grad_features = rng.normal(size=(arrays["node_positions"].shape[0], f_dim))
        base_seed = np.array(grad_features)
        if "adjoint_seed_external_field" in perturbations:
            grad_features = grad_features * 1.5 + 0.2

        def external_field_adjoint(ops_, positions_, seed_):
            return ops_.adjoint_external_field(
                seed_, external_potential, positions_, batch, norms
            )

        want = external_field_adjoint(instance.ops, arrays["node_positions"], base_seed)
        got = external_field_adjoint(ops, positions, grad_features)
        return [
            ("grad_positions", got["grad_positions"], want["grad_positions"]),
            (
                "grad_external_potential",
                got["grad_external_potential"],
                want["grad_external_potential"],
            ),
        ]

    if kind == "dipole_readout_adjoint":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        n_graphs = instance.n_graphs
        grad_charge = rng.normal(size=n_graphs)
        grad_dipole = rng.normal(size=(n_graphs, 3))
        base_charge = np.array(grad_charge)
        base_dipole = np.array(grad_dipole)
        if "adjoint_seed_dipole" in perturbations:
            grad_charge = grad_charge * 1.5 + 0.2
            grad_dipole = grad_dipole * 1.5 + 0.2
        # The multipoles perturbation flows through the density input.
        density = _kind_source(instance, "total", perturbations)
        got = ops.adjoint_dipole_readout(
            density, positions, batch, grad_charge, grad_dipole
        )
        want = instance.ops.adjoint_dipole_readout(
            instance.source("total"),
            arrays["node_positions"],
            batch,
            base_charge,
            base_dipole,
        )
        return [
            ("grad_positions", got["grad_positions"], want["grad_positions"]),
            (
                "grad_charge_density",
                got["grad_charge_density"],
                want["grad_charge_density"],
            ),
        ]

    if kind == "kspace_coulomb_energy_stress":
        # The stress surface (the composed reciprocal-space virial): the
        # energy seed moves with its own column, and the inputs mirror
        # the energy kind's (the cache's geometry, the charge density,
        # the basis constants, and the positions of the position virial).
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        grad_energy = rng.normal(size=instance.n_graphs)
        base_seed = np.array(grad_energy)
        if "adjoint_seed_stress" in perturbations:
            grad_energy = grad_energy * 1.5 + 0.2
        source = _kind_source(instance, "total", perturbations)

        def energy_stress(ops_, positions_, cell_, source_, seed_):
            build_ = ops_.geometry_cache(positions_, cell_, batch)
            return ops_.coulomb_stress(
                build_["cache"], seed_, None, source_, positions_
            )

        return [
            (
                "stress",
                energy_stress(ops, positions, cell, source, grad_energy),
                energy_stress(
                    instance.ops,
                    arrays["node_positions"],
                    arrays["cell"],
                    instance.source("total"),
                    base_seed,
                ),
            )
        ]
    if kind == "fukui_normalize_adjoint":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        n_nodes, n_graphs = instance.n_nodes, instance.n_graphs
        fukui_sources = rng.normal(size=(n_nodes, 2))
        total_charge = rng.normal(size=n_graphs)
        total_spin = rng.normal(size=n_graphs)
        grad_alpha = rng.normal(size=n_nodes)
        grad_beta = rng.normal(size=n_nodes)
        base_alpha = np.array(grad_alpha)
        base_beta = np.array(grad_beta)
        if "adjoint_seed_fukui" in perturbations:
            grad_alpha = grad_alpha * 1.5 + 0.2
            grad_beta = grad_beta * 1.5 + 0.2
        # The l=0 charge channels come from the multipoles, so the multipoles
        # perturbation flows through them; the sources, the graph-level
        # charge/spin, and the seeds are the surface's own data.  The
        # normalization has no geometry dependence at all.
        charge_alpha = _kind_source(instance, "alpha", perturbations)[:, 0]
        charge_beta = _kind_source(instance, "beta", perturbations)[:, 0]
        got = ops.adjoint_fukui_normalize(
            fukui_sources,
            charge_alpha,
            charge_beta,
            total_charge,
            total_spin,
            batch,
            grad_alpha,
            grad_beta,
        )
        want = instance.ops.adjoint_fukui_normalize(
            fukui_sources,
            instance.source("alpha")[:, 0],
            instance.source("beta")[:, 0],
            total_charge,
            total_spin,
            batch,
            base_alpha,
            base_beta,
        )
        if channel == "alpha":
            comparisons = [
                (
                    "grad_fukui_sources",
                    got["grad_fukui_sources"],
                    want["grad_fukui_sources"],
                ),
                (
                    "grad_charge_l0_alpha",
                    got["grad_charge_l0_alpha"],
                    want["grad_charge_l0_alpha"],
                ),
            ]
        else:
            comparisons = [
                (
                    "grad_fukui_sources",
                    got["grad_fukui_sources"],
                    want["grad_fukui_sources"],
                ),
                (
                    "grad_charge_l0_beta",
                    got["grad_charge_l0_beta"],
                    want["grad_charge_l0_beta"],
                ),
            ]
        # The graph-level seeds are one surface per graph, carried by both
        # channels' cases.
        comparisons.extend(
            (
                (
                    "grad_total_charge",
                    got["grad_total_charge"],
                    want["grad_total_charge"],
                ),
                ("grad_total_spin", got["grad_total_spin"], want["grad_total_spin"]),
            )
        )
        return comparisons

    if kind == "analytic_correction_energy_adjoint":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        grad_energy = rng.normal(size=instance.n_graphs)
        base_seed = np.array(grad_energy)
        if "adjoint_seed_correction_energy" in perturbations:
            grad_energy = grad_energy * 1.5 + 0.2
        source = _kind_source(instance, "total", perturbations)
        got = ops.adjoint_correction_energy(
            source, positions, cell, batch, pbc, grad_energy
        )
        want = instance.ops.adjoint_correction_energy(
            instance.source("total"),
            arrays["node_positions"],
            arrays["cell"],
            batch,
            pbc,
            base_seed,
        )
        return [
            (
                "grad_charge_density",
                got["grad_charge_density"],
                want["grad_charge_density"],
            ),
            ("grad_positions", got["grad_positions"], want["grad_positions"]),
        ]

    if kind == "analytic_correction_features_adjoint":
        seed = _MIRROR_SEEDS[(instance.key, kind)]
        rng = np.random.default_rng(seed)
        f_dim = instance.contract.field_dimension
        grad_alpha = rng.normal(size=(instance.n_nodes, f_dim))
        grad_beta = rng.normal(size=(instance.n_nodes, f_dim))
        base_alpha = np.array(grad_alpha)
        base_beta = np.array(grad_beta)
        if "adjoint_seed_correction_features" in perturbations:
            grad_alpha = grad_alpha * 1.5 + 0.2
            grad_beta = grad_beta * 1.5 + 0.2
        alpha_density = _kind_source(instance, "alpha", perturbations)
        beta_density = _kind_source(instance, "beta", perturbations)
        got = ops.adjoint_correction_features(
            alpha_density,
            beta_density,
            positions,
            cell,
            batch,
            pbc,
            grad_alpha,
            grad_beta,
        )
        want = instance.ops.adjoint_correction_features(
            instance.source("alpha"),
            instance.source("beta"),
            arrays["node_positions"],
            arrays["cell"],
            batch,
            pbc,
            base_alpha,
            base_beta,
        )
        return [
            (
                "grad_charge_density_alpha",
                got["grad_charge_density_alpha"],
                want["grad_charge_density_alpha"],
            ),
            (
                "grad_charge_density_beta",
                got["grad_charge_density_beta"],
                want["grad_charge_density_beta"],
            ),
            ("grad_positions", got["grad_positions"], want["grad_positions"]),
        ]

    raise AssertionError(f"unknown kind {kind!r}")  # pragma: no cover


def _assert_kind_parity(instance, kind, channel, perturbations=frozenset()) -> None:
    label = channel if channel is not None else "-"
    for name, got, want in _kind_comparisons(instance, kind, channel, perturbations):
        _assert_parity(got, want, f"{instance.key}/{kind}/{label}:{name}")


# ---------------------------------------------------------------------------
# 1. Tier parity per operator kind
# ---------------------------------------------------------------------------

# One parity case per (kind, channel/label) the kind serves: the geometry
# cache once, the channel kinds per spin channel, the energy kinds per label.
PER_KIND_PARITY_CASES = (
    ("kspace_geometry_cache", None),
    ("kspace_coulomb_features", "alpha"),
    ("kspace_coulomb_features", "beta"),
    ("kspace_coulomb_energy", "alpha"),
    ("kspace_coulomb_energy", "beta"),
    ("kspace_coulomb_energy", "total"),
    ("self_interaction_linear", "alpha"),
    ("self_interaction_linear", "beta"),
    ("external_field_projection", None),
    ("fukui_normalize", "alpha"),
    ("fukui_normalize", "beta"),
    ("dipole_readout", "total"),
    ("analytic_correction_energy", "alpha"),
    ("analytic_correction_energy", "beta"),
    ("analytic_correction_energy", "total"),
    ("analytic_correction_features", "alpha"),
    ("analytic_correction_features", "beta"),
)

PARITY_CASES = tuple(
    (key, kind, channel)
    for key in GEOMETRIES
    for kind, channel in PER_KIND_PARITY_CASES
)


@pytest.fixture(
    params=PARITY_CASES,
    ids=[f"{key}-{kind}-{channel}" for key, kind, channel in PARITY_CASES],
)
def parity_case(request, instances):
    key, kind, channel = request.param
    return instances[key], kind, channel


def test_tier_parity_per_operator_kind(parity_case):
    """The portable T1 implementation of every kind matches the reference.

    Parametrized per geometry and spin channel, following the suite's
    per-instance coverage pattern: a failure names the instance, the kind,
    the channel, and the buffer.  The six fixture-served kinds compare
    against the golden fixtures; the three mirror-served kinds against the
    NumPy mirror of the upstream formula.
    """
    instance, kind, channel = parity_case
    assert kind in POLAR_LONG_RANGE_KINDS
    _assert_kind_parity(instance, kind, channel)


# ---------------------------------------------------------------------------
# 2. Injected-perturbation red evidence
# ---------------------------------------------------------------------------

PERTURBATION_CASES = tuple(
    (key, kind, perturbation)
    for key in GEOMETRIES
    for kind in (*POLAR_LONG_RANGE_KINDS, *ADJOINT_SURFACES)
    for perturbation in PERTURBATIONS
)


@pytest.fixture(
    params=PERTURBATION_CASES,
    ids=[f"{key}-{kind}-{p}" for key, kind, p in PERTURBATION_CASES],
)
def perturbation_case(request, instances):
    key, kind, perturbation = request.param
    return instances[key], kind, perturbation


def _sensitivity(instance, kind):
    """The perturbations that must break this instance's parity for a kind.

    Instance-aware because the pbc class gates the corrective fields: a bulk
    graph selects none, so its ``analytic_correction_features`` outputs are
    exactly zero whatever the inputs -- insensitive by contract, not by
    accident.  Every other kind's sensitivity is input-surface only.
    """
    if instance.graph_class == "bulk" and kind in (
        "analytic_correction_features",
        "analytic_correction_energy_adjoint",
        "analytic_correction_features_adjoint",
    ):
        return frozenset()
    return PERTURBATION_SENSITIVITY[kind]


def test_injected_perturbation_breaks_parity_with_attribution(perturbation_case):
    """Every (kind, perturbation) verdict matches the sensitivity matrix.

    A kind that consumes the perturbed buffer FAILS parity with the kind
    named in the failure message; a kind that does not consume it holds
    parity.  The verdicts together are the attribution proof: the failure
    lands on exactly the kinds whose declared inputs moved, so a red parity
    check points at the right operator.
    """
    instance, kind, perturbation = perturbation_case
    channel = PROBE_CHANNEL[kind]
    if perturbation in _sensitivity(instance, kind):
        with pytest.raises(AssertionError, match=kind):
            _assert_kind_parity(instance, kind, channel, {perturbation})
    else:
        _assert_kind_parity(instance, kind, channel, {perturbation})


def test_correction_feature_inputs_are_channel_distinct(instance):
    """The corrective fields follow their own channel's density.

    On the molecule instance (where the correction is active): perturbing
    only the beta channel's density breaks the beta parity and leaves the
    alpha parity intact -- the channel-distinct correction input contract,
    covered here as red evidence because the perturbation machinery
    expresses it directly.  On the bulk instance the corrective fields are
    zero by contract and stay exactly zero under the same perturbation.
    """
    arrays = instance.arrays
    result = instance.ops.analytic_correction_features(
        instance.source("alpha"),
        instance.source("beta") + 0.13,
        arrays["node_positions"],
        arrays["cell"],
        instance.batch(),
        instance.pbc(),
    )
    if instance.graph_class == "molecule":
        _assert_parity(
            result["correction_fields_alpha"],
            arrays["features_correction_alpha"],
            f"{instance.key}/analytic_correction_features/alpha:channel_distinct",
        )
        with pytest.raises(AssertionError, match="analytic_correction_features"):
            _assert_parity(
                result["correction_fields_beta"],
                arrays["features_correction_beta"],
                f"{instance.key}/analytic_correction_features/beta:channel_distinct",
            )
    else:
        # Bulk graphs select no corrective fields: exactly zero, and the
        # perturbed beta density cannot change that.
        assert np.all(result["correction_fields_alpha"] == 0.0)
        assert np.all(result["correction_fields_beta"] == 0.0)


# ---------------------------------------------------------------------------
# 3. Capacity report fields
# ---------------------------------------------------------------------------


def test_capacity_report_fields_validate(instance):
    """The emitted capacity record carries the manifest vocabulary.

    ``n_k`` is the cache's own half-sphere k-point count (the capacity
    currency the cache kind owns), ``table_bytes`` the dense-table bytes in
    the contract's table dtype, the correction mode the graph class selects,
    and the rebuild trigger the layer's default.  The record validates on
    emission (the Python side validates inline; the native constructor now
    validates at construction, exercised by the C++ registry probe).
    """
    arrays = instance.arrays
    cache = instance.ops.geometry_cache(
        arrays["node_positions"], arrays["cell"], instance.batch()
    )
    n_k = int(cache["k_vectors"].shape[0])
    assert (
        n_k == instance.manifest["geometry"]["n_k"]
    ), f"{instance.key}: the cache's k-point count is the capacity currency"

    mode = GRAPH_CLASS_MODES[instance.graph_class]
    assert set(GRAPH_CLASS_MODES.values()) <= set(POLAR_LONG_RANGE_CORRECTION_MODES)
    record = instance.contract.capacity_record(
        n_k, instance.n_nodes, mode, rebuild_on_cell_change=True
    )
    assert record["n_k"] == n_k
    assert record["correction_mode"] == mode
    assert record["rebuild_on_cell_change"] is True
    itemsize = 4 if instance.contract.table_dtype == "float32" else 8
    assert record["table_bytes"] == 2 * n_k * instance.n_nodes * itemsize
    assert itemsize == 8
    assert record["table_bytes"] > 0

    # The declared dtype is the delivered one (fp64: the tier-1 kernels carry
    # every table in double, and the fp32 measurement below pins why the
    # declaration cannot say otherwise today); an fp32 delivery would halve
    # the bytes and validates just the same.
    assert instance.contract.table_dtype == "float64"
    fp32 = instance.contract.capacity_record(
        n_k, instance.n_nodes, mode, table_dtype="float32"
    )
    assert fp32["table_bytes"] == record["table_bytes"] // 2


def test_capacity_admission_rejects_garbage(instance):
    """The admission path validates: garbage capacity data has no record.

    Both sides refuse the same three -- a non-positive k-point count, a
    table dtype outside the precision policy, a correction mode outside the
    manifest vocabulary.
    """
    contract = instance.contract
    n_k = instance.manifest["geometry"]["n_k"]
    with pytest.raises(ValueError, match="n_k must be"):
        contract.capacity_record(0, instance.n_nodes, "none")
    with pytest.raises(ValueError, match="correction mode"):
        contract.capacity_record(n_k, instance.n_nodes, "box")
    with pytest.raises(ValueError, match="table_dtype"):
        contract.capacity_record(n_k, instance.n_nodes, "none", "float16")


# ---------------------------------------------------------------------------
# 4. Layout admission: k_major and graph_major on data
# ---------------------------------------------------------------------------


def test_program_node_contracts_are_the_layout_authority(instance):
    """The compiled program's node contracts carry the layout orders.

    The retired set-valued table pinned the C++ node declarations against a
    probe printout; now the program DAG declares the long-range nodes, so the
    contracts themselves are the authority: every kind's buffers declare their
    orders, the orders stay inside the long-range vocabulary, the two new
    orders are genuinely additions to the shared-tensor set, and the program's
    binding report binds every node at the portable tier -- the host admission
    invariant the family's program inherits.
    """
    program = program_slice(instance)
    orders: dict[str, set[str]] = {}
    for node in program["nodes"]:
        kind = node["kind"]
        assert kind not in orders, f"the program declares {kind} once"
        orders[kind] = {
            buffer["layout"] for buffer in (*node["inputs"], *node["outputs"])
        }
    assert set(orders) == set(POLAR_LONG_RANGE_KINDS)
    for kind, declared in orders.items():
        assert declared, f"{instance.key}/{kind} declares at least one buffer"
        assert declared <= LONG_RANGE_LAYOUTS, f"{instance.key}/{kind}"
    assert {"k_major", "graph_major"} <= LONG_RANGE_LAYOUTS
    assert not ({"k_major", "graph_major"} & FAMILY_LAYOUT_ORDERS)

    report = program["binding_report"]
    assert report["admitted"], "every long-range node binds"
    assert report["node_count"] == len(program["nodes"])
    assert report["portable_node_count"] == len(program["nodes"])
    for binding in report["bindings"]:
        assert binding["tier"] == "portable"
        assert binding["portable_covered"]
        assert not binding["artifact_required"]
        assert binding["precision"]["accumulator"] == "float64"


def test_program_node_contracts_realize_their_declared_extents(instance):
    """The declared extents match the contract data and the realized data.

    The shape-validation pass the program assembly runs is a predicate over
    contract data; this is its data-side counterpart.  Every buffer's leading
    extent is the dynamic row count except the fully static constants, the
    channel extents are the contract's basis widths, the self-interaction
    kind's two adjoint seed surfaces carry exactly the columns the battery's
    self-term seeds use, and -- on data -- the buffers the program declares
    k-major lead with the realized k count while the graph-major ones lead
    with the graph count.
    """
    contract = instance.contract
    program = program_slice(instance)
    density_width = (contract.density_max_l + 1) ** 2
    field_width = (contract.feature_max_l + 1) ** 2 * len(
        contract.feature_smearing_widths
    )
    assert contract.field_dimension == field_width
    assert contract.charges_dimension == 2 * density_width

    static_names = {"cell", "field_feature_norms"}
    by_kind = {node["kind"]: node for node in program["nodes"]}
    for kind, node in by_kind.items():
        for buffer in (*node["inputs"], *node["outputs"]):
            shape = buffer["shape"]
            assert shape, f"{instance.key}/{kind}/{buffer['name']} has a shape"
            if buffer["name"] in static_names:
                assert shape[0] > 0, f"{instance.key}/{kind}:{buffer['name']}"
            else:
                assert shape[0] == 0, (
                    f"{instance.key}/{kind}:{buffer['name']} leads on the "
                    "dynamic row count"
                )
            if buffer["name"] in (
                "multipoles_alpha",
                "multipoles_beta",
                "charge_density",
                "charge_density_alpha",
                "charge_density_beta",
                "energy_self_fields",
            ):
                assert shape[-1] == density_width, (
                    f"{instance.key}/{kind}:{buffer['name']} carries the "
                    "density channel width"
                )
    # The self-interaction kind's adjoint seed surfaces: the battery's
    # self-term seeds are generated from these very outputs, so the columns
    # the kernels consume and the columns the contracts declare cannot drift.
    self_terms = by_kind["self_interaction_linear"]
    columns = {buffer["name"]: buffer["shape"][-1] for buffer in self_terms["outputs"]}
    assert columns["feature_self_terms_alpha"] == field_width
    assert columns["feature_self_terms_beta"] == field_width
    assert columns["energy_self_fields"] == density_width
    seed_terms = np.random.default_rng(
        _MIRROR_SEEDS[(instance.key, "self_interaction_adjoint")]
    ).normal(size=(instance.n_nodes, columns["feature_self_terms_alpha"]))
    seed_fields = np.random.default_rng(
        _MIRROR_SEEDS[(instance.key, "self_interaction_adjoint")]
    ).normal(size=(instance.n_nodes, columns["energy_self_fields"]))
    assert instance.ops.adjoint_feature_self_terms(seed_terms).shape == (
        instance.n_nodes,
        density_width,
    )
    assert instance.ops.adjoint_energy_self_fields(seed_fields).shape == (
        instance.n_nodes,
        density_width,
    )

    # On data: the k-major buffers lead with the realized k count and the
    # graph-major ones with the graph count, per the program's own orders.
    arrays = instance.arrays
    cache = instance.ops.geometry_cache(
        arrays["node_positions"], arrays["cell"], instance.batch()
    )
    for buffer in by_kind["kspace_geometry_cache"]["outputs"]:
        rows = cache[buffer["name"]]
        if buffer["layout"] == "k_major":
            assert rows.shape[0] == arrays["k_vectors"].shape[0]
        elif buffer["name"] == "volumes":
            assert rows.shape[0] == instance.n_graphs


def test_program_schedule_is_the_unrolled_chain(instance):
    """The program's schedule is the contract's unrolled chain, pinned.

    The chain is contract data, not orchestration: the depth is the contract's
    recursion depth, the global entries name exactly the operators the program
    declares (each at least once, none absent), the backbone entries stay in
    the region vocabulary, and the native mirror (``symmetrix.polar_schedule``)
    equals the Python builder's schedule entry for entry -- the
    cross-language pin the retired table's role moved to.
    """
    from symmetrix.mace_polar_execution_contract import polar_execution_schedule

    program = program_slice(instance)
    schedule = program["schedule"]
    assert schedule == polar_execution_schedule(contract_depth(instance))
    assert schedule == symmetrix.polar_schedule(contract_depth(instance))

    declared = {node["kind"] for node in program["nodes"]}
    reached = {
        entry["name"]
        for entry in schedule["entries"]
        if entry["kind"] == "global_operator"
    }
    assert reached == declared
    backbone = {
        entry["name"] for entry in schedule["entries"] if entry["kind"] == "backbone"
    }
    assert backbone == {
        "interactions",
        "feature_mixer",
        "field_update",
        "local_electron_energy",
    }
    assert schedule["recursion_step_count"] == contract_depth(instance)
    iterations = [entry["iteration"] for entry in schedule["entries"]]
    assert iterations == sorted(iterations)
    assert iterations[-1] == contract_depth(instance) + 1


def test_fp32_tables_miss_the_declared_parity_gate(instance):
    """The declared table dtype is the one that meets the group's gate.

    The tier-1 kernels deliver fp64 tables and the contract declares fp64.
    This pins the measurement behind that decision: recomputing the fixture's
    own k-space energy from its tables with them rounded to fp32 moves the
    energy by three to four orders of magnitude more than the 1e-12 relative
    gate the group holds -- so an fp32 table delivery is a tolerance-gated
    optimization that would have to move the declaration and the gate with it,
    not a free capacity win.
    """
    arrays = instance.arrays
    two_pi_cubed = _two_pi_cubed()
    field_constant = float(arrays["field_constant"][0])
    source = np.asarray(instance.source("total"), dtype=np.float64)
    k_graph_index = np.asarray(arrays["k_batch"]).astype(int)
    volumes = np.asarray(arrays["volume"], dtype=np.float64)
    factors = np.asarray(arrays["density_basis_fs"], dtype=np.float64)[:, 0]

    def energy(cosines, sines, factors_, coulomb, volume_per_k):
        cosines = np.asarray(cosines, dtype=np.float64)
        sines = np.asarray(sines, dtype=np.float64)
        factors_ = np.asarray(factors_, dtype=np.float64)
        coulomb = np.asarray(coulomb, dtype=np.float64)
        volume_per_k = np.asarray(volume_per_k, dtype=np.float64)
        coeff_cos = cosines @ source
        coeff_sin = sines @ source
        real = np.einsum("km,km->k", coeff_cos, factors_[:, :, 0]) + np.einsum(
            "km,km->k", coeff_sin, factors_[:, :, 1]
        )
        imaginary = np.einsum("km,km->k", coeff_cos, factors_[:, :, 1]) - np.einsum(
            "km,km->k", coeff_sin, factors_[:, :, 0]
        )
        density = (
            np.stack((real, imaginary), axis=1) * two_pi_cubed / volume_per_k[:, None]
        )
        potential = density * coulomb[:, None] * field_constant
        per_k = 2.0 * (
            density[:, 0] * potential[:, 0] + density[:, 1] * potential[:, 1]
        )
        accumulated = np.zeros(len(volumes))
        np.add.at(accumulated, k_graph_index, per_k)
        return (volumes * accumulated * 0.5 / two_pi_cubed**2)[0]

    fp64 = energy(
        arrays["cosines"],
        arrays["sines"],
        factors,
        arrays["k_factor_coulomb"],
        arrays["volume_per_k"],
    )
    reference = float(arrays["energy_kspace_total"][0])
    _assert_parity(fp64, reference, f"{instance.key}/fp32_probe:fp64_mirror")

    def to_fp32(array):
        return np.asarray(array, dtype=np.float32).astype(np.float64)

    fp32 = energy(
        to_fp32(arrays["cosines"]),
        to_fp32(arrays["sines"]),
        to_fp32(factors),
        to_fp32(arrays["k_factor_coulomb"]),
        to_fp32(arrays["volume_per_k"]),
    )
    relative = abs(fp32 - fp64) / abs(fp64)
    assert relative > 100.0 * RTOL, (
        f"{instance.key}: fp32 tables moved the energy by {relative:.3e} "
        f"relative, inside the {RTOL:g} gate -- the fp64 declaration would "
        "then be dishonest"
    )
    assert instance.contract.table_dtype == "float64"


def test_k_major_layout_admission(instance):
    """The k-major buffers realize their declared order on data.

    Every k-indexed buffer of the cache carries the k extent leading, the k
    rows are grouped by graph (the flattened-over-graphs semantics), and the
    graph-major volumes scatter onto the k-major rows -- the cross-layout
    production the graph_major/k_major pair declares.
    """
    arrays = instance.arrays
    result = instance.ops.geometry_cache(
        arrays["node_positions"], arrays["cell"], instance.batch()
    )
    n_k = arrays["k_vectors"].shape[0]
    for name in (
        "k_vectors",
        "k_norm2",
        "k_graph_index",
        "k0_mask",
        "volume_per_k",
        "k_factor_coulomb",
        "k_factor_proj",
    ):
        assert result[name].shape[0] == n_k, f"{instance.key}/{name} leads on k"
    assert result["cos_table"].shape == (n_k, instance.n_nodes)
    assert result["sin_table"].shape == (n_k, instance.n_nodes)
    assert result["density_fourier_factors"].shape[0] == n_k
    assert result["feature_fourier_factors"].shape[0] == n_k

    # Rows grouped by graph: the graph index along the k axis is
    # non-decreasing (every row of graph g precedes every row of graph g+1).
    k_graph_index = np.asarray(result["k_graph_index"])
    assert np.all(
        np.diff(k_graph_index) >= 0
    ), f"{instance.key}: k rows must be grouped by graph"
    # The graph-major volume scattered onto the k-major rows (the fixture's
    # own volume spelling, so the comparison is exact).
    np.testing.assert_array_equal(
        result["volume_per_k"], arrays["volume"][k_graph_index]
    )


def test_graph_major_layout_admission(instance):
    """The graph-major buffers realize their declared order on data.

    The per-graph outputs lead on the graph extent, and the k-major rows
    reduce into them consistently: re-deriving the k-space energy from the
    kernel's own k-major stage outputs (the NumPy mirror of the reduction)
    reproduces the graph-major energy output -- the cross-layout reduction
    the pair declares.
    """
    arrays = instance.arrays
    ops = instance.ops
    batch = instance.batch()
    build = ops.geometry_cache(arrays["node_positions"], arrays["cell"], batch)
    cache = build["cache"]
    n_graphs = instance.n_graphs

    energy = ops.coulomb_energy(cache, instance.source("total"), None)
    corrections = ops.analytic_correction_energy(
        instance.source("total"),
        arrays["node_positions"],
        arrays["cell"],
        batch,
        instance.pbc(),
    )
    dipole = ops.dipole_readout(
        instance.source("total"), arrays["node_positions"], batch
    )
    for name, rows in (
        ("energy_kspace", energy),
        ("molecule_correction", corrections["molecule_correction"]),
        ("slab_correction", corrections["slab_correction"]),
        ("total_charge", dipole["total_charge"]),
    ):
        assert rows.shape[0] == n_graphs, f"{instance.key}/{name} leads on graph"
    assert dipole["total_dipole"].shape == (n_graphs, 3)

    # The k-major -> graph-major reduction mirror: per-graph fp64 sums over
    # that graph's k rows, exactly the kernel's accumulation order.
    density = ops.assemble_fourier(cache, instance.source("total"))
    potential = ops.apply_coulomb(cache, density)
    k_graph_index = np.asarray(build["k_graph_index"])
    volumes = arrays["volume"]
    scale = 0.5 / _two_pi_cubed() ** 2
    per_k = 2.0 * (density[:, 0] * potential[:, 0] + density[:, 1] * potential[:, 1])
    for graph in range(n_graphs):
        mirror = volumes[graph] * scale * np.sum(per_k[k_graph_index == graph])
        _assert_parity(
            energy[graph],
            mirror,
            f"{instance.key}/kspace_coulomb_energy/graph_major_reduction",
        )


def test_multi_graph_spelling_splits_the_k_axis():
    """A genuinely multi-graph batch realizes both orders block-wise.

    The k-major rows split into per-graph blocks (each block bounded by the
    graph's own cutoff sphere), each block's table is the phase GEMM of that
    block's k rows against that graph's nodes (the graph-blocked spelling of
    the k-major table), and the graph-major consumers carry one row per
    graph.  Two graphs with different cells and different pbc classes.
    """
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=2.0,
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0, 1.5],
    )
    cell = np.zeros((2, 3, 3))
    cell[0] = np.diag([6.0, 6.0, 6.0])
    cell[1] = np.diag([4.0, 4.0, 9.0])
    positions = np.array(
        [
            [0.1, 0.2, 0.3],
            [1.0, -0.5, 0.7],
            [2.0, 0.0, 0.1],
            [0.3, 0.4, 0.2],
            [0.9, 0.8, 3.0],
        ]
    )
    graph_of = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    graph_pbc = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.int32)
    result = ops.geometry_cache(positions, cell, graph_of)
    k_graph_index = np.asarray(result["k_graph_index"])
    k_vectors = np.asarray(result["k_vectors"])

    assert np.all(np.diff(k_graph_index) >= 0), "k rows grouped by graph"
    blocks = [k_graph_index == graph for graph in range(2)]
    assert blocks[0].sum() > 0 and blocks[1].sum() > 0
    for graph in range(2):
        # Each block inside its graph's own cutoff sphere.
        norms = np.linalg.norm(k_vectors[blocks[graph]], axis=1)
        assert norms.max() <= 2.0 + 1e-12
        # The graph-blocked spelling of the k-major phase table: the block's
        # rows against that graph's node columns (the table spans every
        # node; the block spelling addresses its own graph's columns).
        nodes = graph_of == graph
        np.testing.assert_allclose(
            result["cos_table"][blocks[graph]][:, nodes],
            np.cos(k_vectors[blocks[graph]] @ positions[nodes].T),
            rtol=0,
            atol=1e-12,
            err_msg=f"graph{graph} cos block",
        )
        np.testing.assert_allclose(
            result["sin_table"][blocks[graph]][:, nodes],
            np.sin(k_vectors[blocks[graph]] @ positions[nodes].T),
            rtol=0,
            atol=1e-12,
            err_msg=f"graph{graph} sin block",
        )
        # The graph-major volume scattered per block (the kernel's cofactor
        # determinant rounds differently from LAPACK's, hence the bound).
        volume = abs(np.linalg.det(cell[graph]))
        np.testing.assert_allclose(
            result["volume_per_k"][blocks[graph]], volume, rtol=1e-12, atol=0
        )

    # The graph-major consumers carry one row per graph.
    density = np.array(
        [
            [0.8, 0.1, -0.2, 0.3],
            [-0.4, 0.0, 0.5, -0.1],
            [0.6, -0.3, 0.2, 0.4],
            [-0.2, 0.7, -0.6, 0.1],
            [0.5, -0.4, 0.3, 0.2],
        ]
    )
    dipole = ops.dipole_readout(density, positions, graph_of)
    assert dipole["total_charge"].shape == (2,)
    assert dipole["total_dipole"].shape == (2, 3)
    corrections = ops.analytic_correction_energy(
        density, positions, cell, graph_of, graph_pbc
    )
    assert corrections["correction_energy"].shape == (2,)
    cache = result["cache"]
    energy = ops.coulomb_energy(cache, density, None)
    assert energy.shape == (2,)


# ---------------------------------------------------------------------------
# 5. Cache sharing at the binding level
# ---------------------------------------------------------------------------

CONSUMER_ORDER = (
    "features_alpha",
    "features_beta",
    "energy_alpha",
    "energy_beta",
    "energy_total",
)


def _consume_shared_cache(
    instance, cache, reverse: bool = False
) -> dict[str, np.ndarray]:
    """Evaluate every consumer from the one shared cache.

    ``reverse`` walks the consumers in the opposite order, so a replay can
    prove no consumer perturbs state another one reads rather than only
    re-proving the first order.
    """
    ops = instance.ops
    outputs = {}
    channels = tuple(reversed(CHANNELS)) if reverse else CHANNELS
    labels = tuple(reversed(ENERGY_LABELS)) if reverse else ENERGY_LABELS
    for channel in channels:
        density = ops.assemble_fourier(cache, instance.source(channel))
        potential = ops.apply_coulomb(cache, density)
        outputs[f"features_{channel}"] = ops.project_to_features(cache, potential)
    for label in labels:
        outputs[f"energy_{label}"] = ops.coulomb_energy(
            cache, instance.source(label), None
        )
    return outputs


def test_one_geometry_cache_serves_every_consumer(instance):
    """One cache build per evaluation serves both channels and both paths.

    The sharing contract the probe expresses: the cache is a first-class
    value every stage method takes (the handle rides in the build result),
    so both spin channels' feature paths and the energy path consume the
    very tables one build produced.  Every consumer stays at parity from the
    single build, re-evaluating the consumers in reverse order reproduces
    every output bit for bit (no consumer perturbs the shared state), and a
    second build of the same geometry is deterministic (the rebuild-every-
    step semantics the capacity trigger records).
    """
    arrays = instance.arrays
    build = instance.ops.geometry_cache(
        arrays["node_positions"], arrays["cell"], instance.batch()
    )
    outputs = _consume_shared_cache(instance, build["cache"])
    assert tuple(outputs) == CONSUMER_ORDER

    references = {
        "features_alpha": arrays["features_pbc_alpha"],
        "features_beta": arrays["features_pbc_beta"],
        "energy_alpha": arrays["energy_kspace_alpha"],
        "energy_beta": arrays["energy_kspace_beta"],
        "energy_total": arrays["energy_kspace_total"],
    }
    for name in CONSUMER_ORDER:
        _assert_parity(
            outputs[name], references[name], f"{instance.key}/shared_cache/{name}"
        )
    # The outputs are non-zero, so the parity above is not vacuous.
    for name in CONSUMER_ORDER:
        assert np.abs(outputs[name]).max() > 0.0, name

    # No consumer perturbs the shared state: a second full pass over the
    # same cache, in reverse consumer order, after every consumer has
    # already consumed it, reproduces every output bit for bit.
    second_pass = _consume_shared_cache(instance, build["cache"], reverse=True)
    for name in CONSUMER_ORDER:
        assert np.array_equal(
            outputs[name], second_pass[name]
        ), f"{instance.key}/{name}: a consumer must not perturb the shared cache"

    # Rebuild determinism: a fresh build of the same geometry reproduces the
    # tables bit for bit (one cache per step, rebuilt to the same values).
    rebuilt = instance.ops.geometry_cache(
        arrays["node_positions"], arrays["cell"], instance.batch()
    )
    np.testing.assert_array_equal(build["cos_table"], rebuilt["cos_table"])
    np.testing.assert_array_equal(build["sin_table"], rebuilt["sin_table"])
    np.testing.assert_array_equal(
        build["feature_fourier_factors"], rebuilt["feature_fourier_factors"]
    )
