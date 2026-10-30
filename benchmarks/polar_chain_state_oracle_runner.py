"""Capture the PolarMACE delta chain's inputs and state from the frozen oracle.

The end-to-end runner (``polar_end_to_end_oracle_runner.py``) freezes the
oracle's *outputs*; this runner freezes the chain's own state, so the delta
layer can be gated without a torch environment: the inputs the chain consumes
(the backbone's per-layer node features, the model's node attributes, the
geometry, and the per-graph channel settings) and the state it produces (the
equilibrated spin-resolved density at every equilibration, the per-iteration
potential features, the per-node local-electron energy, and the final
reductions).

    python benchmarks/polar_chain_state_oracle_runner.py

Only files under ``symmetrix/test/data/polar/`` are written; nothing here is
imported by the package.  The runner follows the same oracle pin and the same
two boundary adaptations as the end-to-end runner (see its module docstring).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import importlib.util

REPO_ROOT = Path(__file__).resolve().parent.parent
END_TO_END_RUNNER = REPO_ROOT / "benchmarks" / "polar_end_to_end_oracle_runner.py"
FIXTURE_SCHEMA = "symmetrix.polar.oracle-chain-state"
FIXTURE_VERSION = 1
DEFAULT_OUTPUT_DIR = REPO_ROOT / "symmetrix/test/data/polar"
DEFAULT_MODEL_DIR = Path("~/.cache/symmetrix/test-models")

MODEL_SIZES = ("M",)
PRECISIONS = ("float64", "float32")
#: The chain-state captures: the three geometries' neutral and field channels
#: plus the charge-split channel on the molecule, where the equilibration's
#: closed form moves the most.
RECORDS = (
    ("mol_h2o", "singlet"),
    ("mol_h2o", "charged_triplet"),
    ("mol_h2o", "field"),
    ("slab_w2", "singlet"),
    ("slab_w2", "field"),
    ("bulk_w4", "singlet"),
    ("bulk_w4", "field"),
)


def _load_end_to_end_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "polar_end_to_end_oracle_runner", END_TO_END_RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ChainCapture:
    """The chain's own intermediates, recorded from the oracle's own modules.

    The released model's forward computes the mixed features and each
    iteration's potential features as locals, so the capture installs class
    level recorders on the two modules that consume or produce them
    (``MultiLayerFeatureMixer`` for the mixed features and the field-update
    block for the potential features it is handed).  Class-level patching is
    the same boundary the end-to-end runner uses for the graph_longrange
    call-signature drift.
    """

    def __init__(self) -> None:
        self.mixed: list[np.ndarray] = []
        self.potentials: list[np.ndarray] = []

    def clear(self) -> None:
        self.mixed.clear()
        self.potentials.clear()


_CAPTURE = _ChainCapture()


def install_chain_capture(runner: Any) -> _ChainCapture:
    """Install the recorders (idempotent) and return the capture."""

    from mace.modules import field_blocks

    if getattr(field_blocks, "_symmetrix_chain_capture", None) is _CAPTURE:
        return _CAPTURE
    original_mixer = field_blocks.MultiLayerFeatureMixer.forward
    original_update = field_blocks.AgnosticEmbeddedOneBodyVariableUpdate.forward

    def mixer_forward(self, all_node_feats):
        value = original_mixer(self, all_node_feats)
        _CAPTURE.mixed.append(np.array(value.detach().cpu().numpy(), dtype=np.float64))
        return value

    def update_forward(
        self,
        node_attrs,
        node_feats,
        edge_attrs,
        edge_feats,
        edge_index,
        potential_features,
        local_charges,
    ):
        _CAPTURE.potentials.append(
            np.array(potential_features.detach().cpu().numpy(), dtype=np.float64)
        )
        return original_update(
            self,
            node_attrs,
            node_feats,
            edge_attrs,
            edge_feats,
            edge_index,
            potential_features,
            local_charges,
        )

    field_blocks.MultiLayerFeatureMixer.forward = mixer_forward
    field_blocks.AgnosticEmbeddedOneBodyVariableUpdate.forward = update_forward
    field_blocks._symmetrix_chain_capture = _CAPTURE
    return _CAPTURE


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--sizes", nargs="+", default=list(MODEL_SIZES))
    return parser.parse_args(argv)


def capture(
    runner: Any,
    torch: Any,
    args: argparse.Namespace,
    size: str,
    precision: str,
    geometry: dict[str, Any],
    channel: tuple[Any, ...],
) -> dict[str, np.ndarray]:
    """Run one oracle evaluation and return the chain's inputs and state."""

    from ase import Atoms
    from mace.calculators import MACECalculator
    from graph_longrange import gto_utils as gto_module

    channel_name, charge, spin, field, fermi_level, _geometries = channel
    path = args.model_dir.expanduser() / f"MACE-POLAR-1-{size}.model"
    model = torch.load(str(path), map_location="cpu", weights_only=False)
    runner.prepare_checkpoint(model, gto_module)
    calculator = MACECalculator(
        models=[model],
        model_type="PolarMACE",
        device="cpu",
        default_dtype=precision,
    )
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
    calculator.results.clear()
    capture = install_chain_capture(runner)
    capture.clear()
    atoms.get_potential_energy()

    if precision == "float64":
        # The fp64 reference is the same weights evaluated with fp64
        # accumulation: the released checkpoints' weights are fp32, so the
        # model is cast before the forward (mace logs the conversion).
        model64 = torch.load(str(path), map_location="cpu", weights_only=False)
        runner.prepare_checkpoint(model64, gto_module)
        model64 = model64.to(torch.float64)
        calculator64 = MACECalculator(
            models=[model64],
            model_type="PolarMACE",
            device="cpu",
            default_dtype="float64",
        )
        atoms.calc = calculator64
        calculator64.results.clear()
        capture.clear()
        atoms.get_potential_energy()
        calculator = calculator64

    data = calculator._atoms_to_batch(atoms)
    batch = data.to_dict()
    with torch.no_grad():
        output = model if precision == "float32" else model64
        results = output(batch, training=False, compute_force=False)

    def numpy(value: Any) -> np.ndarray:
        return np.array(value.detach().cpu().numpy(), dtype=np.float64)

    node_feats = numpy(results["node_feats"])
    # A calculator may evaluate the model more than once before the run whose
    # results are read (its compile path probes the graph), so the capture
    # keeps the *last* evaluation's calls: one mixer call and one field-update
    # call per unrolled iteration.
    iterations = len(model.interactions)
    if len(capture.mixed) < 1 or len(capture.potentials) < iterations:
        raise SystemExit(
            "the chain capture expected one mixer call and one field-update "
            "call per iteration, got "
            f"{len(capture.mixed)} and {len(capture.potentials)}"
        )
    arrays = {
        # The chain's own intermediates: the mixed features the update chain
        # consumes and every unrolled iteration's potential features.  They are
        # recorded from the oracle's own modules, so the delta layer's forward
        # is gated on the buffer the end-to-end stage's dipole and energy paths
        # read rather than on its final state alone.
        "features_mixed": capture.mixed[-1],
        "potential_features": np.stack(capture.potentials[-iterations:], axis=0),
        "node_features": node_feats.reshape(node_feats.shape[0], -1),
        "node_attrs": numpy(batch["node_attrs"]),
        "positions": numpy(batch["positions"]),
        "cell": numpy(batch["cell"]).reshape(-1, 3, 3),
        "graph_pbc": np.array(geometry["pbc"], dtype=np.int32).reshape(1, 3),
        "node_graph_index": numpy(batch["batch"]).astype(np.int32),
        "total_charge": np.array([float(charge)], dtype=np.float64),
        "total_spin": np.array([float(spin)], dtype=np.float64),
        "external_potential": np.array(
            [[0.0, float(field[0]), float(field[1]), float(field[2])]], dtype=np.float64
        ),
        "spin_charge_density": numpy(results["spin_charge_density"]).reshape(
            node_feats.shape[0], -1
        ),
        "charge_density": numpy(results["density_coefficients"]),
        "spin_density": numpy(results["spin_density"]),
        # The oracle composes the readout's per-node energies into the graph's
        # total, so the capture carries that sum; the per-node values are
        # zero-guarded by the model's own readout.
        "local_electron_energy": np.asarray(
            results["electron_energy"].detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1),
    }
    return arrays


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = _load_end_to_end_runner()
    torch, _gto = runner.load_oracle(runner._parse_args([]))
    torch.set_default_dtype(torch.float64)
    geometries = {item["name"]: item for item in runner.fixture_geometries()}
    channels = {item[0]: item for item in runner.CHANNEL_SETTINGS}

    manifest_records: list[dict[str, Any]] = []
    for size in args.sizes:
        for geometry_name, channel_name in RECORDS:
            for precision in PRECISIONS:
                arrays = capture(
                    runner,
                    torch,
                    args,
                    size,
                    precision,
                    geometries[geometry_name],
                    channels[channel_name],
                )
                name = f"chain_{size}_{geometry_name}_{channel_name}_{precision}.npz"
                path = args.output_dir / name
                np.savez_compressed(path, **arrays)
                manifest_records.append(
                    {
                        "name": name,
                        "size": size,
                        "geometry": geometry_name,
                        "channel": channel_name,
                        "dtype": precision,
                        "nodes": int(arrays["positions"].shape[0]),
                        "iterations": int(arrays["potential_features"].shape[0]),
                        "bytes": path.stat().st_size,
                    }
                )
                print(f"wrote {name} ({path.stat().st_size} bytes)")
    manifest = {
        "schema": FIXTURE_SCHEMA,
        "version": FIXTURE_VERSION,
        "description": (
            "PolarMACE delta-chain inputs and state captured from the frozen "
            "torch oracle stack (docs/user/models.md#mace-polar-1)."
        ),
        "oracle": {
            "mace_sha": runner.ORACLE_MACE_SHA,
            "mace_ref": runner.ORACLE_MACE_REF,
            "graph_electrostatics_sha": runner.ORACLE_GRAPH_ELECTROSTATICS_SHA,
            "graph_longrange_version": runner.ORACLE_GRAPH_LONGRANGE_VERSION,
        },
        "records": manifest_records,
    }
    (args.output_dir / "chain_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote chain_manifest.json ({len(manifest_records)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
