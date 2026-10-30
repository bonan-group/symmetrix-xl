"""Capture end-to-end MACE-POLAR-1 references from the frozen torch oracle.

The end-to-end tests compare the family's energy, forces, stress, and dipole
against this oracle on molecular, slab, and periodic fixtures, in both spin-channel settings, in
fp32 and fp64.  This runner freezes those references once, out of tree, so the
plain-env suite never imports torch, ``mace``, or ``graph_longrange``.

    python benchmarks/polar_end_to_end_oracle_runner.py \
        --verify-determinism

Like ``benchmarks/polar_oracle_fixture_runner.py``, the oracle uses the pinned
``mace`` and ``graph_electrostatics`` checkouts, fixed inputs and no timestamps
in the fixture content, and a canonical sha256 per record.

See ``docs/user/models.md#mace-polar-1`` for the maintained model contract.
This runner uses two boundary shims and records the evaluated cell:

* **The evaluated cell.**  mace's neighborhood builder extends the cell along
  every non-periodic axis by ``(max|position| + 1) * 5 * cutoff``
  (``mace/data/neighborhood.py``), so a molecule or a slab is evaluated in a
  much larger box than its structure declares, and the volume, the k-grid, the
  k-space energy, and the stress follow that box.  The record carries it as
  ``cell`` and the structure's own box as ``input_cell``; the chain-state
  references record the same evaluated cell.  Before this was recorded the
  end-to-end records carried the structure's box and their k-space energies
  could not be reproduced from their own cells (a molecule's differed by 8.3%,
  a slab's by 16%; the bulk control, fully periodic, agreed -- no extension).

* **Call-signature drift.**  The mace pin invokes ``graph_longrange`` one API
  revision ahead of the package pin (v0.4.4): ``precompute_geometry`` and
  ``coulomb_energy`` take a retired ``force_pbc_evaluator`` mode select, and
  ``forward_dynamic`` takes ``pbc`` plus a size-1 sigma axis.  The shim drops
  the retired mode select and the duplicated ``pbc`` and squeezes the sigma
  axis -- inert for the shipped ``mixed_periodic`` evaluation, which is the
  only path PolarMACE uses.
* **Checkpoint module state.**  The released checkpoints serialize the
  ``graph_longrange`` modules of an earlier revision, whose derived attributes
  (the pbc dispatch, ``GTOSelfInteractionBlock.features_dim``) the pinned
  package expects as instance state.  Both are restorations of derivable state,
  not numerical changes; ``prepare_checkpoint`` restores exactly them.

Only files under ``symmetrix/test/data/polar/`` are written; nothing here is
imported by the package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_SCHEMA = "symmetrix.polar.oracle-end-to-end"
FIXTURE_VERSION = 1

# Oracle pin -- see docs/user/models.md#mace-polar-1 and
# the constants duplicate-pinned in symmetrix/test/test_polar_oracle_fixtures.py.
ORACLE_MACE_SHA = "22f0809735bd4dd1deba80cf8e16f89913a35ff4"
ORACLE_MACE_REF = "v0.3.16-2-g22f08097"
ORACLE_GRAPH_ELECTROSTATICS_SHA = "e0eb8945c21e203656f4e0bbece5d1da488b6753"
ORACLE_GRAPH_LONGRANGE_VERSION = "0.4.4"

MODEL_SIZES = ("S", "M")
PRECISIONS = ("float64", "float32")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "symmetrix/test/data/polar"
DEFAULT_MODEL_DIR = Path("~/.cache/symmetrix/test-models")

#: Structural cross-check of the captured decomposition: the oracle's
#: total energy is atomic + interaction + local-electron + electrostatic +
#: external-field (``extensions.py:931-935``).  fp32 evaluates the identity in
#: its own precision, so the tolerance is dtype-dependent and pinned per
#: record in the manifest.
DECOMPOSITION_TOLERANCE = {"float64": 1.0e-9, "float32": 1.0e-3}

#: Repeat tolerance for the determinism gate.  fp64 is bit-reproducible and is
#: required to be byte-identical.  fp32 is not: the oracle's CPU reductions are
#: order-dependent, so a repeat moves the forces by up to ~1.5e-7 eV/A (measured;
#: the manifest records the observed spread) while energy, dipole, and the
#: equilibrated density repeat exactly.  The bound below is the gate for a
#: repeat, an order of magnitude above the measured spread, and the fp32 parity
#: gates are pinned to the fp64 records with this spread as their floor.
REPEAT_TOLERANCE = {
    "float64": {
        "energy_abs": 0.0,
        "forces_max_abs": 0.0,
        "stress_max_abs": 0.0,
        "dipole_max_abs": 0.0,
        "density_max_abs": 0.0,
    },
    "float32": {
        "energy_abs": 1.0e-3,
        "forces_max_abs": 1.0e-5,
        "stress_max_abs": 1.0e-8,
        "dipole_max_abs": 1.0e-6,
        "density_max_abs": 1.0e-6,
    },
}

#: The k-space energy block's last call, recorded by the drift shim so the
#: capture can split the composed mixed_periodic energy into its pieces.
_ENERGY_CAPTURE: dict[str, Any] = {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-graph-electrostatics",
        type=Path,
        required=True,
        help="Read-only graph_electrostatics checkout (graph_longrange package).",
    )
    parser.add_argument(
        "--oracle-mace",
        type=Path,
        required=True,
        help="Read-only mace checkout (the model side of the oracle).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory holding the released MACE-POLAR-1-{S,M,L}.model checkpoints.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(MODEL_SIZES),
        choices=list(MODEL_SIZES),
        help="Checkpoint sizes to capture (L parity is a follow-up record).",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=list(PRECISIONS),
        choices=list(PRECISIONS),
        help="Oracle default dtypes to capture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Fixture directory (default: symmetrix/test/data/polar).",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Re-run every capture in-process and assert byte-identical values.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Runner-boundary adaptations
# ---------------------------------------------------------------------------


def load_oracle(args: argparse.Namespace) -> Any:
    """Import the pinned oracle stack and install the documented adaptations."""

    for path in (args.oracle_graph_electrostatics, args.oracle_mace):
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise SystemExit(f"Oracle checkout not found: {resolved}")
        sys.path.insert(0, str(resolved))

    import mace  # noqa: F401  (also sets the weights-only env e3nn requires)
    import torch

    from graph_longrange import energy as energy_module
    from graph_longrange import features as features_module
    from graph_longrange import gto_utils as gto_module

    original_precompute = features_module.GTOElectrostaticFeatures.precompute_geometry

    def precompute_geometry(self, **kwargs):
        kwargs.pop("force_pbc_evaluator", None)
        return original_precompute(self, **kwargs)

    features_module.GTOElectrostaticFeatures.precompute_geometry = precompute_geometry

    original_forward = features_module.GTOElectrostaticFeatures.forward_dynamic

    def forward_dynamic(self, cache, source_feats, pbc=None):
        if source_feats.dim() == 3 and source_feats.shape[-2] == 1:
            source_feats = source_feats.squeeze(-2)
        return original_forward(self, cache, source_feats)

    features_module.GTOElectrostaticFeatures.forward_dynamic = forward_dynamic

    original_energy_forward = energy_module.GTOElectrostaticEnergy.forward

    def coulomb_energy(self, **kwargs):
        kwargs.pop("force_pbc_evaluator", None)
        out = original_energy_forward(self, **kwargs)
        # The composed value and its inputs, so the capture can split the
        # mixed_periodic energy into its k-space and correction pieces (the
        # composition the family's two energy nodes must reproduce).
        _ENERGY_CAPTURE["kwargs"] = kwargs
        _ENERGY_CAPTURE["composed"] = out
        return out

    energy_module.GTOElectrostaticEnergy.forward = coulomb_energy

    torch.set_default_dtype(torch.float64)
    return torch, gto_module


def prepare_checkpoint(model: Any, gto_module: Any) -> Any:
    """Restore the derived module state the released checkpoints omit."""

    def restore(module: Any) -> None:
        if isinstance(module, gto_module.GTOSelfInteractionBlock) and not hasattr(
            module, "features_dim"
        ):
            module.features_dim = module.features_irreps.dim
        for child in module.children():
            restore(child)

    restore(model.electric_potential_descriptor)
    restore(model.coulomb_energy)
    # PolarMACE never overrides the default mixed_periodic handling: the design
    # note's evaluation contract (k-space for every graph plus the per-pbc
    # analytic corrections).
    model.electric_potential_descriptor.set_pbc_handling("mixed_periodic")
    model.coulomb_energy.set_pbc_handling("mixed_periodic")
    return model


# ---------------------------------------------------------------------------
# Fixture geometry and channel set
# ---------------------------------------------------------------------------

#: One water monomer, placed so that the fixture is reproducible from the
#: source alone: O-H 0.96 A, H-O-H 104.5 degrees.
_WATER = (
    ("O", (0.000, 0.000, 0.000)),
    ("H", (0.960, 0.000, 0.000)),
    ("H", (-0.240, 0.930, 0.000)),
)


def _water_bulk(seed: int = 20261015) -> tuple[tuple[str, ...], np.ndarray]:
    """Four water molecules on a fixed fcc-like arrangement in a 9.86 A cube."""

    import random

    cell = 9.86
    rng = random.Random(seed)
    centers = (
        (0.00, 0.00, 0.00),
        (0.50, 0.50, 0.00),
        (0.50, 0.00, 0.50),
        (0.00, 0.50, 0.50),
    )
    species: list[str] = []
    positions: list[tuple[float, float, float]] = []
    for center in centers:
        # A deterministic per-molecule rotation keeps the arrangement generic
        # (no symmetry-locked dipoles) while staying reproducible: the fixture
        # records the resulting positions, so only a byte-identical repeat
        # regenerates them.
        angles = [rng.uniform(0.0, 2.0 * np.pi) for _ in range(3)]
        rotation = _rotation(*angles)
        origin = np.array(center) * cell + np.array([0.5, 0.5, 0.5])
        for symbol, offset in _WATER:
            species.append(symbol)
            positions.append(origin + rotation @ np.array(offset))
    return tuple(species), np.array(positions)


def _rotation(alpha: float, beta: float, gamma: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])
    return rz @ ry @ rx


def _water_slab() -> tuple[tuple[str, ...], np.ndarray]:
    """Two water molecules in a 6 x 6 x 18 A cell with vacuum along z."""

    species: list[str] = []
    positions: list[tuple[float, float, float]] = []
    for shift, angle in (
        (np.array([2.4, 2.4, 8.6]), 0.0),
        (np.array([4.4, 4.0, 9.6]), np.pi),
    ):
        rotation = _rotation(angle, 0.4 * angle, 0.0)
        for symbol, offset in _WATER:
            species.append(symbol)
            positions.append(shift + rotation @ np.array(offset))
    return tuple(species), np.array(positions)


def _molecule() -> tuple[tuple[str, ...], np.ndarray]:
    species = tuple(symbol for symbol, _ in _WATER)
    positions = np.array([offset for _, offset in _WATER])
    return species, positions


#: (name, species, positions, cell, pbc, graph class).
def fixture_geometries() -> tuple[dict[str, Any], ...]:
    molecules = _molecule()
    bulk = _water_bulk()
    slab = _water_slab()
    return (
        {
            "name": "mol_h2o",
            "species": list(molecules[0]),
            "positions": molecules[1],
            "cell": 12.0 * np.eye(3),
            "pbc": (False, False, False),
            "graph_class": "molecule",
        },
        {
            "name": "slab_w2",
            "species": list(slab[0]),
            "positions": slab[1],
            "cell": np.diag([6.0, 6.0, 18.0]),
            "pbc": (True, True, False),
            "graph_class": "slab",
        },
        {
            "name": "bulk_w4",
            "species": list(bulk[0]),
            "positions": bulk[1],
            "cell": 9.86 * np.eye(3),
            "pbc": (True, True, True),
            "graph_class": "bulk",
        },
    )


#: The channel settings, as (name, total charge, total spin, external field,
#: fermi level, geometries).  ``spin`` is the checkpoint's own spelling: the
#: alpha channel targets (Q + S - 1) / 2 and the beta channel (Q - S + 1) / 2,
#: so 1.0 is the closed-shell channel setting and 3.0 splits the alpha/beta
#: equilibration targets.  The field setting carries a strong-but-plausible
#: field so its energy term stands well clear of fp32 noise on an energy of
#: this magnitude.
#:
#: ``fermi_lane`` is the targeted probe of the external-potential lane 0: the
#: pinned PolarMACE forward fills it with ``zeros_like(fermi_level)``
#: (extensions.py:643-645), so a nonzero fermi level must leave every recorded
#: quantity unchanged.  It is captured on the molecule geometry only, which is
#: where the lane's l=0 projection would show.
CHANNEL_SETTINGS = (
    ("singlet", 0.0, 1.0, (0.0, 0.0, 0.0), 0.0, None),
    ("charged_triplet", 1.0, 3.0, (0.0, 0.0, 0.0), 0.0, None),
    ("field", 0.0, 1.0, (0.1, -0.04, 0.02), 0.0, None),
    ("fermi_lane", 0.0, 1.0, (0.0, 0.0, 0.0), 7.5, ("mol_h2o",)),
)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _model_path(args: argparse.Namespace, size: str) -> Path:
    path = args.model_dir.expanduser().resolve() / f"MACE-POLAR-1-{size}.model"
    if not path.is_file():
        raise SystemExit(f"Checkpoint not found: {path}")
    return path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _energy_kspace_split(calculator: Any) -> float:
    """The k-space piece of the last mixed_periodic energy evaluation."""

    if not _ENERGY_CAPTURE:
        raise AssertionError(
            "the energy split was not captured: the drift shim's "
            "GTOElectrostaticEnergy.forward was not reached."
        )
    kwargs = _ENERGY_CAPTURE["kwargs"]
    block = calculator.models[0].coulomb_energy
    kspace = block._compute_kspace_energy_common(
        k_vectors=kwargs["k_vectors"],
        k_norm2=kwargs["k_norm2"],
        k_vector_batch=kwargs["k_vector_batch"],
        k0_mask=kwargs["k0_mask"],
        source_feats=kwargs["source_feats"],
        node_positions=kwargs["node_positions"],
        batch=kwargs["batch"],
        volume=kwargs["volume"],
    )
    return float(np.sum(np.asarray(kspace.detach().numpy(), dtype=np.float64)))


def evaluate(
    calculator: Any,
    geometry: dict[str, Any],
    channel: tuple[str, float, float, tuple[float, float, float], float, Any],
) -> dict[str, Any]:
    """Run one oracle evaluation and return its values as plain arrays."""

    from ase import Atoms

    _, charge, spin, field, fermi_level, _geometries = channel
    atoms = Atoms(
        geometry["species"],
        positions=geometry["positions"],
        cell=geometry["cell"],
        pbc=geometry["pbc"],
    )
    atoms.info["charge"] = float(charge)
    atoms.info["spin"] = float(spin)
    atoms.info["external_field"] = np.array(field, dtype=np.float64)
    atoms.info["fermi_level"] = float(fermi_level)
    atoms.calc = calculator

    energy = float(atoms.get_potential_energy())
    forces = np.array(atoms.get_forces())
    stress = np.array(atoms.get_stress())
    results = dict(calculator.results)

    # mace's neighborhood builder extends the cell along every non-periodic
    # axis (``cell[axis] = (max|positions| + 1) * 5 * cutoff``,
    # ``mace/data/neighborhood.py``), so the box a molecule or a slab is
    # evaluated in is not the box its structure declares, and the volume, the
    # k-grid, the k-space energy, and the stress all follow the extended one.
    # The record carries the evaluated cell as ``cell`` and the structure's own
    # box as ``input_cell``.
    evaluation_cell = np.asarray(
        calculator._atoms_to_batch(atoms).to_dict()["cell"].detach().cpu().numpy(),
        dtype=np.float64,
    ).reshape(-1, 3, 3)

    dipole = np.array(results["dipole"], dtype=np.float64)
    # The calculator surfaces the per-node charge density's l=0 channel rather
    # than the model's graph-level total charge; it is the sum the model's
    # ``compute_total_charge_dipole_permuted`` takes.
    charges = np.array(results["charges"], dtype=np.float64)
    total_charge = float(np.sum(charges))
    field_energy = float(np.dot(np.array(field, dtype=np.float64), dipole))
    interaction = float(results["interaction_energy"])
    electron = float(results["electron_energy"])
    electrostatic = float(results["electrostatic_energy"])
    # mixed_periodic composes the k-space energy with the correction the graph's
    # pbc row selects (energy.py: the molecule box term, the slab dipole term,
    # nothing for bulk).  The split is captured so the family's two energy nodes
    # -- kspace_coulomb_energy and analytic_correction_energy -- have separate
    # gates, and so a correction-mode mistake cannot hide inside the sum.
    kspace = _energy_kspace_split(calculator)
    correction = electrostatic - kspace
    # ``energies`` is the model's per-atom output and sums to the backbone half
    # (the atomic reference plus the interaction energy) -- exactly the value
    # the family's backbone-only evaluation returns.  ``node_energy`` is the
    # same array without the atomic reference, so the two give the atomic term.
    backbone = float(np.sum(np.array(results["energies"], dtype=np.float64)))
    node_interaction = np.array(results["node_energy"], dtype=np.float64)

    values = {
        "evaluation_cell": evaluation_cell,
        "energy": energy,
        "backbone_energy": backbone,
        "atomic_energy": backbone - interaction,
        "interaction_energy": interaction,
        "electron_energy": electron,
        "electrostatic_energy": electrostatic,
        "electrostatic_energy_kspace": kspace,
        "electrostatic_energy_correction": correction,
        "field_energy": field_energy,
        "total_charge": total_charge,
        "total_dipole": dipole,
        "forces": forces,
        "stress": stress,
        "charges": charges,
        "spins": np.array(results["spins"], dtype=np.float64),
        "density_coefficients": np.array(
            results["density_coefficients"], dtype=np.float64
        ),
        "spin_charge_density": np.array(
            results["spin_charge_density"], dtype=np.float64
        ),
        "node_interaction_energies": node_interaction,
    }
    # The oracle's own decomposition: the model's total energy is its backbone
    # half plus the long-range chain's two energy kinds plus the external-field
    # term (extensions.py:931-935).  Both sides come from independent outputs,
    # so this is a real cross-check of the captured provenance.
    recomposed = backbone + electron + electrostatic + field_energy
    values["decomposition_residual"] = energy - recomposed
    values["interaction_residual"] = interaction - float(np.sum(node_interaction))
    return values


def capture(
    torch: Any,
    gto_module: Any,
    args: argparse.Namespace,
    size: str,
    precision: str,
    geometry: dict[str, Any],
    channel: tuple[str, float, float, tuple[float, float, float], float, Any],
) -> dict[str, Any]:
    """Load the checkpoint and capture one (size, precision, geometry, channel)."""

    from mace.calculators import MACECalculator

    channel_name, charge, spin, field, fermi_level, _geometries = channel
    model = torch.load(
        str(_model_path(args, size)), map_location="cpu", weights_only=False
    )
    prepare_checkpoint(model, gto_module)
    calculator = MACECalculator(
        models=[model],
        model_type="PolarMACE",
        device="cpu",
        default_dtype=precision,
        # The fermi level is the external-potential lane 0 the oracle zeroes;
        # the channel is carried explicitly so its inertness is recorded.
        info_keys={
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
            "fermi_level": "fermi_level",
        },
    )
    _ENERGY_CAPTURE.clear()
    values = evaluate(calculator, geometry, channel)

    tolerance = DECOMPOSITION_TOLERANCE[precision] * max(1.0, abs(values["energy"]))
    for key in ("decomposition_residual", "interaction_residual"):
        if abs(values[key]) > tolerance:
            raise AssertionError(
                f"{size}/{precision}/{geometry['name']}/{channel_name}: the "
                f"oracle {key} does not close ({values[key]:.3e}, tolerance "
                f"{tolerance:.3e})."
            )

    return {
        "size": size,
        "dtype": precision,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "geometry": geometry["name"],
        "graph_class": geometry["graph_class"],
        "channel": channel_name,
        "charge": float(charge),
        "spin": float(spin),
        "external_field": list(field),
        "fermi_level": float(fermi_level),
        "species": list(geometry["species"]),
        "positions": np.array(geometry["positions"], dtype=np.float64),
        # The evaluated box (mace's extension included) and the structure's own
        # cell, so a consumer never has to guess which one the numbers follow.
        "cell": values.pop("evaluation_cell"),
        "input_cell": np.array(geometry["cell"], dtype=np.float64),
        "pbc": [bool(value) for value in geometry["pbc"]],
        "model_sha256": _sha256_bytes(_model_path(args, size).read_bytes()),
        "decomposition_tolerance": DECOMPOSITION_TOLERANCE[precision],
        **values,
    }


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    """The record's numerical content, canonically ordered for hashing."""

    return {
        key: record[key]
        for key in sorted(record)
        if key
        not in (
            "model_sha256",
            "torch_default_dtype",
        )
    }


def repeat_deltas(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    """Per-quantity distance between two captures of the same record."""

    return {
        "energy_abs": abs(first["energy"] - second["energy"]),
        "forces_max_abs": float(np.abs(first["forces"] - second["forces"]).max()),
        "stress_max_abs": float(np.abs(first["stress"] - second["stress"]).max()),
        "dipole_max_abs": float(
            np.abs(first["total_dipole"] - second["total_dipole"]).max()
        ),
        "density_max_abs": float(
            np.abs(first["spin_charge_density"] - second["spin_charge_density"]).max()
        ),
    }


def assert_repeatable(
    name: str, first: dict[str, Any], second: dict[str, Any]
) -> dict[str, float]:
    """Gate one repeat against the record's dtype repeat tolerance."""

    deltas = repeat_deltas(first, second)
    tolerance = REPEAT_TOLERANCE[first["dtype"]]
    for quantity, delta in deltas.items():
        if delta > tolerance[quantity]:
            raise AssertionError(
                f"{name}: the capture is not reproducible ({quantity} moved by "
                f"{delta:.3e}, tolerance {tolerance[quantity]:.3e})."
            )
    return deltas


def record_name(record: dict[str, Any]) -> str:
    return (
        f"e2e_{record['size']}_{record['geometry']}_"
        f"{record['channel']}_{record['dtype']}.json"
    )


def _dtype_spread(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-quantity fp32-versus-fp64 oracle spread, the fp32 gate's basis."""

    spread: dict[str, Any] = {}
    for key, record in records.items():
        if record["dtype"] != "float32":
            continue
        reference = records.get(key.replace("_float32", "_float64"))
        if reference is None:
            continue
        energy = abs(record["energy"] - reference["energy"])
        forces = float(np.abs(record["forces"] - reference["forces"]).max())
        stress = float(np.abs(record["stress"] - reference["stress"]).max())
        dipole = float(np.abs(record["total_dipole"] - reference["total_dipole"]).max())
        spread[key] = {
            "energy_abs": energy,
            "forces_max_abs": forces,
            "stress_max_abs": stress,
            "dipole_max_abs": dipole,
        }
    return spread


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch, gto_module = load_oracle(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Oracle pin:")
    print(f"  mace:                 {ORACLE_MACE_SHA} ({ORACLE_MACE_REF})")
    print(
        f"  graph_electrostatics: {ORACLE_GRAPH_ELECTROSTATICS_SHA} "
        f"(graph_longrange {ORACLE_GRAPH_LONGRANGE_VERSION})"
    )

    geometries = fixture_geometries()
    records: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    repeat_spread: dict[str, dict[str, float]] = {}
    for size in args.sizes:
        for precision in args.precisions:
            for geometry in geometries:
                for channel in CHANNEL_SETTINGS:
                    if channel[5] is not None and geometry["name"] not in channel[5]:
                        continue
                    record = capture(
                        torch, gto_module, args, size, precision, geometry, channel
                    )
                    name = record_name(record)
                    if args.verify_determinism:
                        repeat = capture(
                            torch,
                            gto_module,
                            args,
                            size,
                            precision,
                            geometry,
                            channel,
                        )
                        spread = assert_repeatable(name, record, repeat)
                        if precision == "float64" and _canonical_json(
                            _payload(record)
                        ) != _canonical_json(_payload(repeat)):
                            raise AssertionError(
                                f"{name}: the fp64 capture is not byte-identical "
                                "across runs."
                            )
                        if any(spread.values()):
                            repeat_spread[name] = spread
                    records[name] = record
                    text = _canonical_json(record)
                    _atomic_write_text(output_dir / name, text)
                    digest = _sha256_bytes(text.encode("utf-8"))
                    entries.append(
                        {
                            "name": name,
                            "path": name,
                            "sha256": digest,
                            "size": size,
                            "dtype": precision,
                            "geometry": geometry["name"],
                            "graph_class": geometry["graph_class"],
                            "channel": channel[0],
                            "n_atoms": len(geometry["species"]),
                            "energy": record["energy"],
                            "model_sha256": record["model_sha256"],
                        }
                    )
                    print(
                        f"  {name}: E={record['energy']:.12f} "
                        f"|F|max={np.abs(record['forces']).max():.6e} "
                        f"|dipole|={np.linalg.norm(record['total_dipole']):.9f} "
                        f"sha256={digest[:12]}..."
                    )

    fingerprint = hashlib.sha256()
    for entry in entries:
        fingerprint.update(entry["sha256"].encode("utf-8"))
    manifest = {
        "schema": FIXTURE_SCHEMA,
        "version": FIXTURE_VERSION,
        "description": (
            "MACE-POLAR-1 end-to-end reference values captured from the frozen "
            "torch oracle stack (docs/user/models.md#mace-polar-1); "
            "the end-to-end parity gates are pinned to these records."
        ),
        "oracle": {
            "mace_sha": ORACLE_MACE_SHA,
            "mace_ref": ORACLE_MACE_REF,
            "graph_electrostatics_sha": ORACLE_GRAPH_ELECTROSTATICS_SHA,
            "graph_longrange_version": ORACLE_GRAPH_LONGRANGE_VERSION,
        },
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "units": {
            "charge": "e",
            "length": "Angstrom",
            "energy": "eV",
            "field": "model units (the checkpoint's external_field channel)",
            "dipole": "e*Angstrom",
            "forces": "eV/Angstrom",
            "stress": "eV/Angstrom^3 (ASE get_stress convention)",
        },
        "geometries": [
            {
                "name": geometry["name"],
                "graph_class": geometry["graph_class"],
                "n_atoms": len(geometry["species"]),
                "pbc": [bool(value) for value in geometry["pbc"]],
            }
            for geometry in geometries
        ],
        "channels": [
            {
                "name": name,
                "charge": charge,
                "spin": spin,
                "external_field": list(field),
                "fermi_level": fermi_level,
                "geometries": list(geometries) if geometries else None,
            }
            for name, charge, spin, field, fermi_level, geometries in CHANNEL_SETTINGS
        ],
        "decomposition_tolerance": DECOMPOSITION_TOLERANCE,
        "repeat_tolerance": REPEAT_TOLERANCE,
        "records": entries,
        "record_fingerprint": fingerprint.hexdigest(),
        "dtype_spread": _dtype_spread(records),
        "repeat_spread": repeat_spread,
        "adaptations": [
            "retired force_pbc_evaluator mode select dropped (inert for mixed_periodic)",
            "duplicated pbc argument dropped and the size-1 sigma axis squeezed in forward_dynamic",
            "derived module state restored after torch.load (pbc dispatch, features_dim)",
            "evaluated cell recorded (mace extends the cell along every non-periodic "
            "axis by (max|position| + 1) * 5 * cutoff; the structure's own box is kept as "
            "input_cell)",
        ],
    }
    manifest_path = output_dir / "e2e_manifest.json"
    _atomic_write_text(manifest_path, _canonical_json(manifest))
    print(f"wrote {manifest_path}")
    print(f"record fingerprint {manifest['record_fingerprint'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
