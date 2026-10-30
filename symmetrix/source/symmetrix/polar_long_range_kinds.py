"""Contract-data linkage for the long-range operator kind group.

The tiered-registry declarations (``libsymmetrix/source/
long_range_operator_kinds.hpp``) are family-neutral C++ data: operator-kind
identity, buffer contracts, precision policy, capability declarations, and
the implementation descriptors a family folds into its registry.  This
module is the Python side of the same declaration: it maps a POLAR_BACKBONE
contract instance -- or the pinned checkpoint facts that record one -- onto
the kind parameterizations the C++ group binds, validates them against the
same admission rules, and renders capacity records in the manifest's
long-range vocabulary (``benchmarks/foundation_model_support_matrix.py``,
``LONG_RANGE_CAPACITY_VERSION`` 1).

The linkage is additive and read-only: it consumes the backbone contract the
extractor already embeds and never redefines it. See the maintained model
contract in ``docs/user/models.md#mace-polar-1`` for support boundaries; this
module is torch-free by construction -- stdlib plus the contract tag import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .mace_polar_execution_contract import POLAR_CONTRACT_TAG

#: The operator-kind vocabulary, in declaration order.  These are the tokens
#: the C++ kind group registers and the binding report carries; the plan's
#: hyphenated spellings (``kspace-geometry-cache``) map onto them by the
#: enum's snake-case convention, the same convention every existing kind
#: name in a binding report uses.
POLAR_LONG_RANGE_KINDS = (
    "kspace_geometry_cache",
    "kspace_coulomb_features",
    "kspace_coulomb_energy",
    "self_interaction_linear",
    "external_field_projection",
    "fukui_normalize",
    "dipole_readout",
    "analytic_correction_energy",
    "analytic_correction_features",
)

#: Analytic correction modes selected by a graph's pbc class, exactly the
#: long-range capacity vocabulary of the support manifest.  A test pins the
#: equality with ``benchmarks/foundation_model_support_matrix.py`` so the
#: registry declarations and the manifest reader cannot drift apart.
POLAR_LONG_RANGE_CORRECTION_MODES = ("makov-payne-dabo", "slab-dipole", "none")

#: The backbone-delta kind vocabulary, in declaration order, mirroring the C++
#: ``delta_kind_vocabulary`` (``mace_polar_delta_kinds.hpp``).  The deltas are
#: the family's node-local execution layer: the chain's own nodes, which
#: produce the chain state the long-range kinds consume.  A test pins the two
#: sides equal, and pins the node contracts below against the native builder.
POLAR_DELTA_KINDS = (
    "polar_spin_source_heads",
    "polar_feature_mixer",
    "polar_source_map",
    "polar_field_update",
    "polar_charge_state",
    "polar_local_electron_energy",
)

#: The buffers each delta kind consumes and produces, in the C++ builder's
#: declaration order.  The pin exists because these names are the program's
#: linking vocabulary: they are what makes the chain state's producers and the
#: long-range kinds' consumers line up, and a rename on one side alone would
#: silently disconnect the two halves of the program.
POLAR_DELTA_NODE_BUFFERS = {
    "polar_spin_source_heads": {
        "inputs": ("node_features_layers",),
        "outputs": ("spin_charge_density_sources",),
    },
    "polar_feature_mixer": {
        "inputs": ("node_features_layers",),
        "outputs": ("features_mixed",),
    },
    "polar_source_map": {
        "inputs": ("node_features",),
        "outputs": ("fukui_sources",),
    },
    "polar_field_update": {
        "inputs": (
            "features_mixed",
            "potential_features_alpha",
            "potential_features_beta",
            "spin_charge_density",
        ),
        "outputs": ("charge_sources", "fukui_sources"),
    },
    "polar_charge_state": {
        "inputs": (
            "spin_charge_density_sources",
            "charge_sources",
            "charge_correction_alpha",
            "charge_correction_beta",
        ),
        "outputs": (
            "spin_charge_density",
            "multipoles_alpha",
            "multipoles_beta",
            "charge_density",
            "charge_density_alpha",
            "charge_density_beta",
            "charge_l0_alpha",
            "charge_l0_beta",
        ),
    },
    "polar_local_electron_energy": {
        "inputs": (
            "node_features",
            "potential_features_alpha",
            "potential_features_beta",
            "spin_charge_density",
        ),
        "outputs": ("local_electron_energy",),
    },
}

#: The buffers the *program* supplies rather than a node producing them
#: (mirroring ``polar_program_input_buffer``): the prepared graph's geometry
#: and indexing, the polar block's contract constant, and the family's
#: graph-level scalar channel.
POLAR_PROGRAM_INPUT_BUFFERS = frozenset(
    {
        "positions",
        "cell",
        "node_graph_index",
        "graph_pbc",
        "field_feature_norms",
        "external_potential",
        "total_charge",
        "total_spin",
        "node_attrs",
        "node_features_layers",
        "node_features",
    }
)

#: The derivative-mode vocabulary of the group: forward everywhere, the
#: position adjoints of the reverse-mode stage, the normalization adjoint of
#: the global scatter-and-broadcast kind, and the experimental
#: reciprocal-space cell adjoint (the Ewald stress).  The order is the C++
#: ``derivative_mode_vocabulary``'s declaration order, so a kind's mode tuple
#: reads identically on both sides.
POLAR_LONG_RANGE_DERIVATIVE_MODES = (
    "forward",
    "position_adjoint",
    "normalization_adjoint",
    "cell_adjoint",
)

#: The kinds whose forward consumes positions and whose position adjoint is
#: implemented, mirroring the C++ ``LongRangeKindCapabilities::
#: position_adjoint`` flags.  The self-interaction constants and the Fukui
#: normalization carry no position dependence.
POLAR_LONG_RANGE_POSITION_ADJOINT_KINDS = frozenset(
    {
        "kspace_geometry_cache",
        "kspace_coulomb_features",
        "kspace_coulomb_energy",
        "external_field_projection",
        "dipole_readout",
        "analytic_correction_energy",
        "analytic_correction_features",
    }
)

#: The global scatter-and-broadcast kind: its adjoint is the scatter of the
#: broadcast gradient (the Fukui equilibration's own derivative surface).
POLAR_LONG_RANGE_NORMALIZATION_ADJOINT_KINDS = frozenset({"fukui_normalize"})

#: The kinds that declare a cell-adjoint surface, mirroring the C++
#: ``LongRangeKindCapabilities::cell_adjoint`` flags. On the periodic route,
#: the reciprocal geometry, electrostatic feature/energy chains, and analytic
#: correction volume terms contribute cell adjoints. On the fully non-periodic
#: real-space route, fixed-Cartesian cell adjoints are exactly zero. The
#: compatibility kind identifiers retain their historical ``kspace_`` names.
POLAR_LONG_RANGE_CELL_ADJOINT_KINDS = frozenset(
    {
        "kspace_geometry_cache",
        "kspace_coulomb_features",
        "kspace_coulomb_energy",
        "analytic_correction_energy",
        "analytic_correction_features",
    }
)

#: The declaration-time derivative service: every mode of the vocabulary is
#: served by at least one kind, so the declared set is the whole vocabulary.
POLAR_LONG_RANGE_DECLARED_DERIVATIVE_MODES = {
    "forward",
    "position_adjoint",
    "normalization_adjoint",
    "cell_adjoint",
}

#: Fixed unit-convention constants (design note, Kernel inventory): the
#: half-sphere factor-2 weights of the features and energy kernels, the 0.5
#: the external-field gauge carries per spin channel, and the cubic Madelung
#: constant of the molecular box term.
HALF_SPHERE_WEIGHT = 2.0
EXTERNAL_FIELD_SCALE = 0.5
CUBIC_MADELUNG = 2.837297

_TABLE_DTYPES = {"float32": 4, "float64": 8}


def _flag(flags: Any, name: str) -> bool:
    """One boolean self-interaction flag of a contract's flags object.

    The flags object is optional in a trimmed slice (the kind-group linkage
    consumes the long-range slice alone); when it is present the flag must be
    a boolean, so a malformed policy is refused rather than defaulted.
    """

    if flags is None:
        return False
    if not isinstance(flags, Mapping):
        raise ValueError("a polar backbone contract's flags must be a mapping")
    value = flags.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"polar backbone flag {name} must be boolean")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _require_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class PolarLongRangeContract:
    """The long-range slice of one POLAR_BACKBONE contract instance.

    The fields are exactly the contract data the C++ kind group binds: the
    sigma set, both max_l's, the model's own k-space cutoff, the buffer
    dimensions, the recursion depth, and the precision policy.  The
    correction mode is deliberately absent -- it is per-graph runtime data
    selected from pbc, never contract data.
    """

    kspace_cutoff: float
    density_max_l: int
    density_smearing_width: float
    feature_max_l: int
    feature_smearing_widths: tuple[float, ...]
    charges_dimension: int
    field_dimension: int
    potential_dimension: int
    field_norm_factor: float = 1.0
    recursion_step_count: int = 1
    spin_channel_count: int = 2
    # The dense tables' dtype is contract data, and the delivered tier-1
    # implementation's is fp64 (``long_range_kernels.hpp`` carries every table
    # in double).  Measured fp32 quantization of those tables moves the
    # fixture energies by 1.2e-9 (mol5) and 7.8e-9 (pbc4) relative -- three to
    # four orders of magnitude above the group's 1e-12 parity gate -- so the
    # declaration states what is delivered; an fp32 table delivery is a
    # tolerance-gated optimization that would move this field with it.
    table_dtype: str = "float64"
    accumulator_dtype: str = "float64"
    # The model's self-interaction policy (the checkpoint's own fields, the
    # ones the oracle hands to its feature descriptor and its k-space energy):
    # the self-interaction overlaps are subtracted unless
    # `include_electrostatic_self_interaction` holds, and the projected
    # features carry the feature self-terms only when `field_si` holds.
    include_electrostatic_self_interaction: bool = False
    field_si: bool = False

    @classmethod
    def from_backbone_contract(
        cls, contract: Mapping[str, Any]
    ) -> "PolarLongRangeContract":
        """Read the long-range slice of a POLAR_BACKBONE contract instance.

        Reads the contract the extractor embeds; the full contract keeps its
        own validation (``normalize_execution_polar_backbone_contract``), so
        this constructor checks only the tag and the slice it consumes.
        """

        if not isinstance(contract, Mapping):
            raise ValueError("a polar backbone contract must be a mapping")
        tag = contract.get("tag")
        if tag != POLAR_CONTRACT_TAG:
            raise ValueError(
                f"long-range kinds bind {POLAR_CONTRACT_TAG!r} contracts; "
                f"got {tag!r}"
            )
        for section in ("charges", "field", "potential"):
            if not isinstance(contract.get(section), Mapping):
                raise ValueError(
                    f"the polar backbone contract is missing its {section} "
                    "descriptor"
                )
        charges = contract["charges"]
        polar_field = contract["field"]
        return cls(
            kspace_cutoff=_finite_positive(
                polar_field.get("kspace_cutoff"), "field.kspace_cutoff"
            ),
            density_max_l=_require_int(charges.get("max_l"), "charges.max_l", 0),
            density_smearing_width=_finite_positive(
                charges.get("smearing_width"), "charges.smearing_width"
            ),
            feature_max_l=_require_int(polar_field.get("max_l"), "field.max_l", 0),
            feature_smearing_widths=_widths(polar_field.get("smearing_widths")),
            charges_dimension=_require_int(
                charges.get("dimension"), "charges.dimension", 1
            ),
            field_dimension=_require_int(
                polar_field.get("dimension"), "field.dimension", 1
            ),
            potential_dimension=_require_int(
                contract["potential"].get("dimension"),
                "potential.dimension",
                1,
            ),
            field_norm_factor=_finite_nonzero(
                contract.get("field_norm_factor", 1.0), "field_norm_factor"
            ),
            recursion_step_count=_require_int(
                contract.get("recursion_step_count"),
                "recursion_step_count",
                1,
            ),
            include_electrostatic_self_interaction=_flag(
                contract.get("flags"), "include_electrostatic_self_interaction"
            ),
            field_si=_flag(contract.get("flags"), "field_si"),
        )

    @classmethod
    def from_checkpoint_facts(
        cls, entry: Mapping[str, Any]
    ) -> "PolarLongRangeContract":
        """Read the long-range slice of a checkpoint-facts model entry.

        The checkpoint facts (``symmetrix/test/data/polar/
        checkpoint_facts.json``) record the pinned models' long-range
        constants: the sigma set, the smearing width, both max_l's, the
        model's k-space cutoff, the dimensions, and the recursion depth.
        The k-space cutoff factor is provenance the facts also carry; the
        contract binds the cutoff value itself.
        """

        if not isinstance(entry, Mapping):
            raise ValueError("a checkpoint-facts model entry must be a mapping")
        try:
            widths = list(entry["field_feature_widths"])
            contract = cls(
                kspace_cutoff=float(entry["kspace_cutoff"]),
                density_max_l=int(entry["atomic_multipoles_max_l"]),
                density_smearing_width=float(entry["atomic_multipoles_smearing_width"]),
                feature_max_l=int(entry["field_feature_max_l"]),
                feature_smearing_widths=tuple(widths),
                charges_dimension=int(entry["charges_dimension"]),
                field_dimension=int(entry["field_dimension"]),
                potential_dimension=int(entry["potential_dimension"]),
                recursion_step_count=int(entry["num_update"]),
                include_electrostatic_self_interaction=bool(
                    entry.get("include_electrostatic_self_interaction", False)
                ),
                field_si=bool(entry.get("field_si", False)),
            )
        except KeyError as error:
            raise ValueError(
                f"the checkpoint-facts entry is missing {error.args[0]!r}"
            ) from error
        return contract

    def validate(self) -> None:
        """Apply the kind group's admission rules to this slice.

        Mirrors ``validate_long_range_contract`` in the C++ declarative
        layer: the same rules, the same rejections, so a contract that
        parameterizes the registry kinds on one side validates on the other.
        """

        _finite_positive(self.kspace_cutoff, "kspace_cutoff")
        for name, value in (
            ("density_max_l", self.density_max_l),
            ("feature_max_l", self.feature_max_l),
        ):
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be 0 or 1 (the analytic radial Fourier "
                    "factors reject l > 1)"
                )
        _finite_positive(self.density_smearing_width, "density_smearing_width")
        if not self.feature_smearing_widths:
            raise ValueError(
                "feature_smearing_widths must carry at least one smearing width"
            )
        for width in self.feature_smearing_widths:
            _finite_positive(width, "feature_smearing_widths")
        if self.spin_channel_count != 2:
            raise ValueError(
                "spin_channel_count must be 2 (the alpha/beta equilibration)"
            )
        density_width = (self.density_max_l + 1) ** 2
        field_width = (self.feature_max_l + 1) ** 2 * len(self.feature_smearing_widths)
        if self.charges_dimension != self.spin_channel_count * density_width:
            raise ValueError(
                "charges_dimension must be spin_channel_count * "
                f"(density_max_l + 1)^2 = {self.spin_channel_count * density_width}"
            )
        if self.field_dimension != field_width:
            raise ValueError(
                "field_dimension must be (feature_max_l + 1)^2 * the sigma "
                f"count = {field_width}"
            )
        if self.potential_dimension != self.spin_channel_count * self.field_dimension:
            raise ValueError(
                "potential_dimension must be spin_channel_count * "
                f"field_dimension = "
                f"{self.spin_channel_count * self.field_dimension}"
            )
        _finite_nonzero(self.field_norm_factor, "field_norm_factor")
        if self.recursion_step_count < 1:
            raise ValueError("recursion_step_count must be at least 1")
        if self.table_dtype not in _TABLE_DTYPES:
            raise ValueError("table_dtype must be float32 or float64")
        if self.accumulator_dtype != "float64":
            raise ValueError(
                "accumulator_dtype must be float64 (the long-range precision "
                "policy keeps fp64 accumulators)"
            )

    # The static buffer extents the contract fixes; the dynamic extents
    # (nodes, graphs, N_k) are runtime capacity, not contract data.
    @property
    def density_channel_width(self) -> int:
        """Per-spin-channel multipole width: (density_max_l + 1)^2."""

        return (self.density_max_l + 1) ** 2

    @property
    def field_channel_width(self) -> int:
        """Per-spin-channel feature width: (feature_max_l + 1)^2 * sigmas."""

        return (self.feature_max_l + 1) ** 2 * len(self.feature_smearing_widths)

    def kind_parameterizations(self) -> dict[str, dict[str, float]]:
        """The per-kind scalar parameterizations the registry group binds.

        One entry per vocabulary kind, matching the parameters the C++
        ``long_range_operator_nodes`` builder stamps on each node: the
        geometry cache carries the k-space contract (cutoff, max_l's, sigma
        set), the evaluation kernels carry the half-sphere weights and the
        model's include flags, and the fixed-form kinds carry only their
        unit-convention constants.
        """

        parameters: dict[str, dict[str, float]] = {
            "kspace_geometry_cache": {
                "kspace_cutoff": self.kspace_cutoff,
                "density_max_l": float(self.density_max_l),
                "feature_max_l": float(self.feature_max_l),
                "density_smearing_width": self.density_smearing_width,
                "spin_channel_count": float(self.spin_channel_count),
            },
            "kspace_coulomb_features": {
                "half_sphere_weight": HALF_SPHERE_WEIGHT,
                # The chain's multiplicity: the unrolled non-self-consistent
                # iteration evaluates this node once per recursion step.  It is
                # program structure rather than numerics, and it is what makes
                # two recursion depths different programs (the node set itself
                # is depth-independent).
                "evaluation_count": float(self.recursion_step_count),
            },
            "kspace_coulomb_energy": {
                "half_sphere_weight": HALF_SPHERE_WEIGHT,
                # The model's own flag: the oracle's `GTOElectrostaticEnergy`
                # subtracts the self-interaction overlaps unless
                # `include_electrostatic_self_interaction` holds.
                "include_self_interaction": (
                    1.0 if self.include_electrostatic_self_interaction else 0.0
                ),
            },
            "self_interaction_linear": {
                # The feature self-terms follow the feature descriptor's flag
                # (`field_si`), the energy self-fields the k-space energy's.
                "include_feature_self_interaction": (1.0 if self.field_si else 0.0),
                "include_energy_self_interaction": (
                    1.0 if self.include_electrostatic_self_interaction else 0.0
                ),
            },
            "external_field_projection": {
                "external_field_scale": EXTERNAL_FIELD_SCALE,
                "field_norm_factor": self.field_norm_factor,
                "evaluation_count": float(self.recursion_step_count),
            },
            "fukui_normalize": {
                "spin_channel_count": float(self.spin_channel_count),
                # Once before the unrolled loop and once per step: the
                # equilibration count is the chain's own signature.
                "equilibration_count": float(1 + self.recursion_step_count),
            },
            "dipole_readout": {},
            "analytic_correction_energy": {
                "cubic_madelung": CUBIC_MADELUNG,
            },
            "analytic_correction_features": {},
        }
        for index, width in enumerate(self.feature_smearing_widths):
            parameters["kspace_geometry_cache"][f"feature_smearing_width_{index}"] = (
                float(width)
            )
        return parameters

    def capacity_record(
        self,
        n_k: int,
        atoms: int,
        correction_mode: str,
        table_dtype: str | None = None,
        rebuild_on_cell_change: bool = True,
    ) -> dict[str, Any]:
        """Render one capacity record in the manifest long_range vocabulary.

        ``n_k`` is the half-sphere k-point count flattened over graphs,
        ``table_bytes`` the dense cos/sin table bytes (``2 * n_k * atoms *
        itemsize``), the correction mode is recorded per graph class, and
        the rebuild trigger states whether a cell change rebuilds the k-grid
        as well as the tables.
        """

        dtype = table_dtype or self.table_dtype
        if isinstance(n_k, bool) or not isinstance(n_k, int) or n_k < 1:
            raise ValueError("n_k must be a positive k-point count")
        if dtype not in _TABLE_DTYPES:
            raise ValueError("table_dtype must be float32 or float64")
        if correction_mode not in POLAR_LONG_RANGE_CORRECTION_MODES:
            raise ValueError(
                f"unknown long-range correction mode {correction_mode!r}; "
                "the vocabulary is "
                f"{', '.join(POLAR_LONG_RANGE_CORRECTION_MODES)}"
            )
        return {
            "n_k": n_k,
            "table_bytes": 2 * n_k * atoms * _TABLE_DTYPES[dtype],
            "correction_mode": correction_mode,
            "rebuild_on_cell_change": rebuild_on_cell_change,
        }


def _widths(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "field.smearing_widths must be a non-empty list of smearing widths"
        )
    return tuple(_finite_positive(width, "field.smearing_widths") for width in value)


def _finite_nonzero(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonzero number")
    result = float(value)
    if not math.isfinite(result) or result == 0.0:
        raise ValueError(f"{name} must be a finite nonzero number")
    return result


def require_delta_kind(name: str) -> str:
    """Resolve one delta vocabulary token or fail with the vocabulary named."""

    if not isinstance(name, str) or name not in POLAR_DELTA_KINDS:
        raise ValueError(
            f"unknown polar delta kind {name!r}; the vocabulary is "
            f"{', '.join(POLAR_DELTA_KINDS)}"
        )
    return name


def require_long_range_kind(name: str) -> str:
    """Resolve one vocabulary token or fail with the vocabulary named.

    The Python-side mirror of the C++ vocabulary query: an unknown kind is a
    clean failure that lists the vocabulary, never a fallback.
    """

    if not isinstance(name, str) or name not in POLAR_LONG_RANGE_KINDS:
        raise ValueError(
            f"unknown long-range operator kind {name!r}; the vocabulary is "
            f"{', '.join(POLAR_LONG_RANGE_KINDS)}"
        )
    return name


def kind_derivative_modes(name: str) -> tuple[str, ...]:
    """The declaration-time derivative service of one kind.

    Every kind serves forward; the position adjoint is declared on the kinds
    whose forward consumes positions, the normalization adjoint on the global
    scatter-and-broadcast kind, and the cell-adjoint surface on compatibility
    kinds whose periodic kernels carry reciprocal cell dependence. The
    fully non-periodic real-space route returns exact-zero fixed-Cartesian cell
    adjoints. The modes are returned in the vocabulary's declaration order, so
    a kind's tuple reads identically on both sides of the cross-language pin.
    """

    require_long_range_kind(name)
    declared = {"forward"}
    if name in POLAR_LONG_RANGE_POSITION_ADJOINT_KINDS:
        declared.add("position_adjoint")
    if name in POLAR_LONG_RANGE_NORMALIZATION_ADJOINT_KINDS:
        declared.add("normalization_adjoint")
    if name in POLAR_LONG_RANGE_CELL_ADJOINT_KINDS:
        declared.add("cell_adjoint")
    return tuple(mode for mode in POLAR_LONG_RANGE_DERIVATIVE_MODES if mode in declared)
