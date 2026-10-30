"""Capture MACE-POLAR-1 op-level golden fixtures from the frozen oracle stack.

This module runs the ``graph_longrange`` kernels of the pinned
``graph_electrostatics`` checkout (with ``mace.tools.scatter`` from the pinned
``mace`` checkout) on two small deterministic geometries — one molecule in a
vacuum box (nonperiodic graph semantics under ``mixed_periodic``) and one small
periodic cell — and writes float64 golden fixtures under
``symmetrix/test/data/polar/`` for every operator of the long-range layer:

* the half-sphere k-grid with folding metadata (per-graph batch ids, k=0 mask,
  volume-per-k, the masked 1/k^2 Coulomb factor, and the half-sphere
  projection weights — 1.0 with 0.5 at k=0; the factor-2 half-sphere doubling
  itself lives inside the features/energy evaluation kernels, not here),
* the cos/sin geometry cache (k.r phase GEMM, masked cosine/sine tables, GTO
  Fourier factors for the density and feature bases),
* the assembled structure factors rho(k) per spin channel,
* the Coulomb-convolved potentials V(k) = FIELD_CONSTANT/k^2 * rho(k),
* the projected electrostatic features per spin channel (k-space part,
  mixed_periodic correction part, and the combined result) plus the ESP
  reconstruction,
* the Coulomb energy (raw k-space, self-interaction node fields, and the final
  mixed_periodic energy with the Makov-Payne/Dabo and slab correction terms),
* the constant self-interaction overlap vectors, the external-field projection
  matrix, the internal-field projection matrix, and the feature channel
  permutation.

See docs/user/models.md#mace-polar-1 for the model contract. This runner uses
the following kernel conventions: the Coulomb kernel is
``FIELD_CONSTANT/k^2`` (rationalized epsilon_0 convention), not a literal
``4*pi/k^2``; the +-k factor-2 weights live in the features/energy kernels;
the energy path recomputes its own cos/sin tables from the same inputs.

Oracle pin (read-only checkouts, never a hard dependency of the plain-env
test suite; env-gated in-tree tests put these trees on ``sys.path`` and
import from them in torch-equipped environments, skipping cleanly otherwise):

* mace            22f0809735bd4dd1deba80cf8e16f89913a35ff4 (v0.3.16-2-g22f08097)
* graph_electrostatics e0eb8945c21e203656f4e0bbece5d1da488b6753 (v0.4.4)

Environment setup (outside the repository; never install into the repo .venv)::

    uv venv --python 3.12 <ORACLE_VENV>
    uv pip install --python <ORACLE_VENV>/bin/python \
        torch e3nn scipy numpy torch-ema torchmetrics matplotlib pandas
    <ORACLE_VENV>/bin/python benchmarks/polar_oracle_fixture_runner.py \
        --oracle-graph-electrostatics /path/to/graph_electrostatics \
        --oracle-mace /path/to/mace \
        --output-dir symmetrix/test/data/polar

Call-signature drift (design note, "Call-signature drift"): mace HEAD invokes
graph_longrange one API revision ahead of package v0.4.4. This runner calls the
package API directly, i.e. already adapted: no ``force_pbc_evaluator``, no
``pbc`` on ``forward_dynamic``, and 2-D ``[N, m]`` source tensors — all
numerically inert for the default ``mixed_periodic`` evaluation.

Determinism: fixed seeds, fixed inputs, no timestamps inside fixture content;
``--verify-determinism`` re-runs the whole capture in-process and asserts
byte-identical arrays. The payload sha256 recorded in each manifest pins the
committed bytes.
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

FIXTURE_SCHEMA = "symmetrix.polar.oracle-op-fixtures"
FIXTURE_VERSION = 1

# Oracle pin — must match the constants in symmetrix/test/test_polar_oracle_fixtures.py.
# See docs/user/models.md#mace-polar-1 for the model contract.
ORACLE_MACE_SHA = "22f0809735bd4dd1deba80cf8e16f89913a35ff4"
ORACLE_MACE_REF = "v0.3.16-2-g22f08097"
ORACLE_GRAPH_ELECTROSTATICS_SHA = "e0eb8945c21e203656f4e0bbece5d1da488b6753"
ORACLE_GRAPH_LONGRANGE_VERSION = "0.4.4"

# Basis parameters, intentionally divergent from the shipped MACE-POLAR-1
# configs: density width 1.0 with feature widths (1.0, 1.5) and cutoff factor
# 1.5 (k_c ~ 4.155) keep N_k small, while the shipped checkpoints carry sigma
# (1.5, 3.0), density width 1.5, and inferred factor 1.0 (k_c ~ 1.847) —
# digest-pinned in symmetrix/test/data/polar/checkpoint_facts.json.  Op-level
# parity is parameter-agnostic, so the divergence changes no parity contract.
DENSITY_MAX_L = 1
DENSITY_SMEARING_WIDTH = 1.0
FEATURE_MAX_L = 1
FEATURE_WIDTHS = (1.0, 1.5)
KSPACE_CUTOFF_FACTOR = 1.5
INTEGRAL_NORMALIZATION = "receiver"
PBC_HANDLING = "mixed_periodic"
INCLUDE_SELF_INTERACTION = False
SEED = 20261014

# Rationalized-epsilon_0 unit contract of the package (e, Angstrom, eV, V).
FIELD_CONSTANT = 1.0 / (5.526349406e-3)

# Size policy: keep N_k in the tens so committed fixtures stay small.
MAX_K_POINTS = 128

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "symmetrix/test/data/polar"

SPIN_CHANNELS = ("alpha", "beta")
ENERGY_LABELS = ("alpha", "beta", "total")

# Half-sphere weight applied inside the evaluation kernels (2x for +-k
# degeneracy); recorded for provenance only, never stored per-k.
HALF_SPHERE_WEIGHT = 2.0

# (name, n_nodes, periodic) — molecule in a vacuum box; small bulk cell.
GEOMETRIES = (
    ("mol5", 5, False),
    ("pbc4", 4, True),
)


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
        help="Read-only mace checkout (only mace.tools.scatter is imported).",
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
        help="Re-run the full capture in-process and assert byte-identical arrays.",
    )
    return parser.parse_args(argv)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _sha256_bytes(contiguous.tobytes(order="C"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _geometry_cell(torch: Any, name: str):
    if name == "mol5":
        return 4.6 * torch.eye(3, dtype=torch.float64)
    if name == "pbc4":
        return torch.diag(torch.tensor([4.0, 4.6, 5.2], dtype=torch.float64))
    raise ValueError(f"Unknown fixture geometry: {name!r}")


def _deterministic_source(torch: Any, n_nodes: int, generator: Any):
    source = 0.35 * torch.randn((n_nodes, 4), generator=generator, dtype=torch.float64)
    # Give the l=0 channel a small deterministic offset so monopole correction
    # terms see a nonzero total charge without saturating anything.
    source[:, 0] = source[:, 0] + 0.1 * torch.arange(n_nodes, dtype=torch.float64)
    return source


def _build_inputs(torch: Any, name: str, n_nodes: int, periodic: bool):
    generator = torch.Generator().manual_seed(SEED + (0 if periodic else 1))
    cell = _geometry_cell(torch, name).unsqueeze(0)
    lattice_lengths = torch.linalg.norm(cell[0], dim=-1)
    fractions = 0.15 + 0.70 * torch.rand(
        (n_nodes, 3), generator=generator, dtype=torch.float64
    )
    node_positions = fractions * lattice_lengths.unsqueeze(0)
    pbc = torch.tensor([[periodic, periodic, periodic]], dtype=torch.bool)
    return cell, node_positions, pbc


def capture_fixture(torch: Any, name: str, n_nodes: int, periodic: bool):
    """Run every oracle op for one geometry; return ({name: tensor}, summary)."""
    from graph_longrange import energy as energy_mod
    from graph_longrange import features
    from graph_longrange import gto_utils
    from graph_longrange import kspace
    from graph_longrange import slabs

    kspace_cutoff = KSPACE_CUTOFF_FACTOR * gto_utils.gto_basis_kspace_cutoff(
        [DENSITY_SMEARING_WIDTH] + list(FEATURE_WIDTHS), DENSITY_MAX_L
    )

    cell, node_positions, pbc = _build_inputs(torch, name, n_nodes, periodic)
    rcell = 2 * torch.pi * torch.linalg.inv(cell.mT)
    volume = torch.linalg.det(cell).reshape(1)
    batch = torch.zeros(n_nodes, dtype=torch.long)

    # --- k-grid (kspace.py:11-127) ---------------------------------------
    k_vectors, k_norm2, k_batch, k0_mask = kspace.compute_k_vectors_flat(
        kspace_cutoff, cell, rcell
    )
    n_k = int(k_vectors.shape[0])
    if n_k > MAX_K_POINTS:
        raise ValueError(
            f"Fixture {name}: N_k={n_k} exceeds the size policy bound "
            f"{MAX_K_POINTS}; shrink the cell."
        )

    # --- feature blocks (features.py:280-793) ----------------------------
    features_block = features.GTOElectrostaticFeatures(
        density_max_l=DENSITY_MAX_L,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        feature_max_l=FEATURE_MAX_L,
        feature_smearing_widths=list(FEATURE_WIDTHS),
        kspace_cutoff=kspace_cutoff,
        include_self_interaction=INCLUDE_SELF_INTERACTION,
        integral_normalization=INTEGRAL_NORMALIZATION,
    )  # pbc_handling defaults to "mixed_periodic", matching PolarMACE.
    features_block_raw = features.GTOElectrostaticFeatures(
        density_max_l=DENSITY_MAX_L,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        feature_max_l=FEATURE_MAX_L,
        feature_smearing_widths=list(FEATURE_WIDTHS),
        kspace_cutoff=kspace_cutoff,
        include_self_interaction=not INCLUDE_SELF_INTERACTION,
        integral_normalization=INTEGRAL_NORMALIZATION,
        pbc_handling="pbc",
    )  # pure k-space projection: no correction terms, no SI subtraction,
    # so the fixture pins each piece of the mixed_periodic decomposition.

    cache = features_block.precompute_geometry(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_batch,
        k0_mask=k0_mask,
        node_positions=node_positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )

    # k.r phase GEMM — the same matmul the cache builder performs internally
    # (features.py:483); the stored cos/sin tables are its masked images.
    k_dot_r = torch.matmul(k_vectors, node_positions.t())

    # The masked k=0 Coulomb factor must zero the origin mode exactly.
    if torch.any(cache["k_factor_coulomb"][k0_mask > 0.5] != 0.0):
        raise AssertionError(f"Fixture {name}: k=0 Coulomb factor not zeroed.")

    # --- per-spin-channel op chain ---------------------------------------
    per_channel: dict[str, dict[str, Any]] = {}
    channel_seeds = {"alpha": 2, "beta": 3}
    for channel, channel_seed in channel_seeds.items():
        generator = torch.Generator().manual_seed(SEED + channel_seed)
        source = _deterministic_source(torch, n_nodes, generator)
        rho = features.assemble_fourier_series_batch(
            source_feats=source,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )
        potential = features.apply_coulomb_kernel_batch(
            density=rho,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        features_final = features_block.forward_dynamic(
            cache=cache, source_feats=source
        )
        features_pbc = features_block_raw.forward_dynamic(
            cache=cache, source_feats=source
        )
        features_correction = (
            features_block.non_periodic_correction_terms.mixed_periodic(
                source_feats=source,
                node_positions=node_positions,
                batch=batch,
                volumes=cache["volumes"],
                correction_node_masks=cache["correction_node_masks"],
            )
        )
        features_self_interaction = features_block.self_interaction_terms(source)
        esps = features.reconstruct_esps_batch(
            potential=potential,
            cosines=cache["cosines"],
            sines=cache["sines"],
            k0_mask=cache["k0_mask"],
        )
        # Bitwise decomposition check: the mixed_periodic forward is
        # (k-space features with self-interaction subtracted) + correction.
        reconstructed = features_pbc - features_self_interaction + features_correction
        if not torch.allclose(features_final, reconstructed, rtol=0, atol=0):
            raise AssertionError(
                f"Fixture {name}/{channel}: mixed features != "
                "pbc - self_interaction + correction."
            )
        per_channel[channel] = {
            "source": source,
            "rho": rho,
            "potential": potential,
            "features_final": features_final,
            "features_pbc": features_pbc,
            "features_correction": features_correction,
            "features_self_interaction": features_self_interaction,
            "esps": esps,
        }

    # --- energy (energy.py:32-48, 165-209) -------------------------------
    energy_block = energy_mod.GTOElectrostaticEnergy(
        density_max_l=DENSITY_MAX_L,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        kspace_cutoff=kspace_cutoff,
        include_self_interaction=INCLUDE_SELF_INTERACTION,
    )
    energy_block_kspace = energy_mod.GTOElectrostaticEnergy(
        density_max_l=DENSITY_MAX_L,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        kspace_cutoff=kspace_cutoff,
        include_self_interaction=not INCLUDE_SELF_INTERACTION,
        pbc_handling="pbc",
    )  # pure k-space energy: no correction terms, no SI subtraction, so the
    # fixture pins each piece of the mixed_periodic decomposition.

    def _run_energy(block, source):
        return block(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_batch,
            k0_mask=k0_mask,
            source_feats=source,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    source_total = per_channel["alpha"]["source"] + per_channel["beta"]["source"]
    energy_records = {}
    for label in ENERGY_LABELS:
        source = source_total if label == "total" else per_channel[label]["source"]
        record = {
            "energy_kspace": _run_energy(energy_block_kspace, source),
            "energy_final": _run_energy(energy_block, source),
            "energy_self_fields": energy_block.self_interaction_terms(source),
            "energy_molecule_correction": energy_block.monopole_dipole_correction(
                source, node_positions, volume, batch
            ),
            "energy_slab_correction": slabs.slab_dipole_correction_energy(
                source, node_positions, volume, batch
            ),
        }
        # Decomposition check: the mixed_periodic forward is the pure k-space
        # energy minus the self-interaction term plus the correction selected
        # by the graph's pbc pattern (molecule box term here; slab or none
        # otherwise).
        node_self_energies = torch.einsum(
            "nb,nb->n", source, record["energy_self_fields"]
        )
        if periodic:
            selected_correction = torch.zeros_like(record["energy_kspace"])
        else:
            selected_correction = record["energy_molecule_correction"]
        reconstructed = (
            record["energy_kspace"]
            - 0.5 * node_self_energies.sum().view(1)
            + selected_correction
        )
        if not torch.allclose(record["energy_final"], reconstructed, rtol=0, atol=1e-9):
            raise AssertionError(
                f"Fixture {name}/{label}: mixed energy != "
                "kspace - self_interaction + selected correction."
            )
        energy_records[label] = record

    # --- constant linear operators (gto_utils.py:233-427) ----------------
    external_field_block = gto_utils.DisplacedGTOExternalFieldBlock(
        FEATURE_MAX_L, list(FEATURE_WIDTHS), INTEGRAL_NORMALIZATION
    )
    internal_field_block = gto_utils.GTOInternalFieldtoFeaturesBlock(
        FEATURE_MAX_L, list(FEATURE_WIDTHS), INTEGRAL_NORMALIZATION
    )

    arrays: dict[str, Any] = {
        # inputs
        "cell": cell,
        "rcell": rcell,
        "pbc": pbc,
        "node_positions": node_positions,
        "batch": batch,
        "volume": volume,
        "kspace_cutoff": torch.tensor([kspace_cutoff]),
        "field_constant": torch.tensor([FIELD_CONSTANT]),
        "half_sphere_weight": torch.tensor([HALF_SPHERE_WEIGHT]),
        # k-grid and folding metadata
        "k_vectors": k_vectors,
        "k_norm2": k_norm2,
        "k_batch": k_batch,
        "k0_mask": k0_mask,
        "volume_per_k": cache["volume_per_k"],
        "k_factor_coulomb": cache["k_factor_coulomb"],
        "k_factor_proj": cache["k_factor_proj"],
        # geometry cache
        "k_dot_r": k_dot_r,
        "cosines": cache["cosines"],
        "sines": cache["sines"],
        "density_basis_fs": cache["density_basis_fs"],
        "feature_basis_fs": cache["feature_basis_fs"],
        # correction masks (contract data selecting the correction mode)
        "correction_mask_is_molecule_node": cache["correction_node_masks"][
            "is_molecule_node"
        ],
        "correction_mask_is_slab_node": cache["correction_node_masks"]["is_slab_node"],
        # constants (identical across geometries; pinned per fixture anyway)
        "si_overlap_constants_features": (
            features_block.self_interaction_terms.overlap_constants
        ),
        "si_overlap_constants_energy": (
            energy_block.self_interaction_terms.overlap_constants
        ),
        "external_field_matrix": external_field_block.matrix,
        "internal_field_matrix": internal_field_block.matrix,
        "output_permutation": features_block.output_permutation,
    }
    for channel in SPIN_CHANNELS:
        record = per_channel[channel]
        arrays[f"source_feats_{channel}"] = record["source"]
        arrays[f"rho_{channel}"] = record["rho"]
        arrays[f"potential_{channel}"] = record["potential"]
        arrays[f"features_pbc_{channel}"] = record["features_pbc"]
        arrays[f"features_correction_{channel}"] = record["features_correction"]
        arrays[f"features_final_{channel}"] = record["features_final"]
        arrays[f"features_self_interaction_{channel}"] = record[
            "features_self_interaction"
        ]
        arrays[f"esps_{channel}"] = record["esps"]
    arrays["source_feats_total"] = source_total
    for label in ENERGY_LABELS:
        record = energy_records[label]
        arrays[f"energy_kspace_{label}"] = record["energy_kspace"]
        arrays[f"energy_final_{label}"] = record["energy_final"]
        arrays[f"energy_self_fields_{label}"] = record["energy_self_fields"]
        arrays[f"energy_molecule_correction_{label}"] = record[
            "energy_molecule_correction"
        ]
        arrays[f"energy_slab_correction_{label}"] = record["energy_slab_correction"]

    summary = {
        "name": name,
        "n_nodes": n_nodes,
        "n_k": n_k,
        "periodic": periodic,
        "kspace_cutoff": kspace_cutoff,
        "volume": float(volume.reshape(-1)[0]),
    }
    return arrays, summary


def _numpy_arrays(arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for key, value in arrays.items():
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        result[key] = np.asarray(array)
    return result


def _manifest(
    torch: Any,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
    payload_name: str,
    payload_bytes: int,
    payload_sha: str,
) -> dict[str, Any]:
    import e3nn
    import scipy

    return {
        "schema": FIXTURE_SCHEMA,
        "version": FIXTURE_VERSION,
        "description": (
            "MACE-POLAR-1 long-range op-level golden fixture captured from the "
            "frozen graph_electrostatics/mace oracle stack "
            "(docs/user/models.md#mace-polar-1)."
        ),
        "oracle": {
            "mace_sha": ORACLE_MACE_SHA,
            "mace_ref": ORACLE_MACE_REF,
            "graph_electrostatics_sha": ORACLE_GRAPH_ELECTROSTATICS_SHA,
            "graph_longrange_version": ORACLE_GRAPH_LONGRANGE_VERSION,
        },
        "environment": {
            "torch": torch.__version__,
            "e3nn": e3nn.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        },
        "parameters": {
            "density_max_l": DENSITY_MAX_L,
            "density_smearing_width": DENSITY_SMEARING_WIDTH,
            "feature_max_l": FEATURE_MAX_L,
            "feature_smearing_widths": list(FEATURE_WIDTHS),
            "kspace_cutoff_factor": KSPACE_CUTOFF_FACTOR,
            "kspace_cutoff": summary["kspace_cutoff"],
            "integral_normalization": INTEGRAL_NORMALIZATION,
            "pbc_handling": PBC_HANDLING,
            "include_self_interaction": INCLUDE_SELF_INTERACTION,
            "seed": SEED,
        },
        "units": {
            "charge": "e",
            "length": "Angstrom",
            "energy": "eV",
            "potential": "V",
            "field_constant": FIELD_CONSTANT,
            "half_sphere_weight": HALF_SPHERE_WEIGHT,
        },
        "geometry": {
            "name": summary["name"],
            "n_nodes": summary["n_nodes"],
            "n_k": summary["n_k"],
            "periodic": summary["periodic"],
            "pbc": [summary["periodic"]] * 3,
            "volume": summary["volume"],
            "graph_class": "bulk" if summary["periodic"] else "molecule",
        },
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
            }
            for name, array in sorted(arrays.items())
        },
        "payload": {
            "format": "npz",
            "path": payload_name,
            "bytes": payload_bytes,
            "sha256": payload_sha,
        },
    }


def _load_oracle(args: argparse.Namespace):
    for path in (args.oracle_graph_electrostatics, args.oracle_mace):
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise SystemExit(f"Oracle checkout not found: {resolved}")
        sys.path.insert(0, str(resolved))
    import torch

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(SEED)

    import graph_longrange as glr

    return torch, glr


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch, glr = _load_oracle(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Oracle pin:")
    print(f"  mace:                 {ORACLE_MACE_SHA} ({ORACLE_MACE_REF})")
    print(
        f"  graph_electrostatics: {ORACLE_GRAPH_ELECTROSTATICS_SHA} "
        f"(graph_longrange {ORACLE_GRAPH_LONGRANGE_VERSION})"
    )

    for name, n_nodes, periodic in GEOMETRIES:
        arrays, summary = capture_fixture(torch, name, n_nodes, periodic)
        if args.verify_determinism:
            repeat_arrays, repeat_summary = capture_fixture(
                torch, name, n_nodes, periodic
            )
            if repeat_summary != summary:
                raise AssertionError(
                    f"Fixture {name}: capture summary not reproducible."
                )
            first = _numpy_arrays(arrays)
            repeat = _numpy_arrays(repeat_arrays)
            for key in first:
                if _array_sha256(first[key]) != _array_sha256(repeat[key]):
                    raise AssertionError(
                        f"Fixture {name}: array {key!r} not byte-identical "
                        "across runs."
                    )
            print(f"  {name}: determinism verified (byte-identical arrays)")

        numpy_arrays = _numpy_arrays(arrays)
        payload_path = output_dir / f"{name}_arrays.npz"
        _atomic_write_npz(payload_path, numpy_arrays)
        payload_bytes = payload_path.stat().st_size
        payload_sha = _sha256_bytes(payload_path.read_bytes())
        manifest = _manifest(
            torch=torch,
            arrays=numpy_arrays,
            summary=summary,
            payload_name=payload_path.name,
            payload_bytes=payload_bytes,
            payload_sha=payload_sha,
        )
        manifest_path = output_dir / f"{name}_manifest.json"
        _atomic_write_text(manifest_path, _canonical_json(manifest))

        print(
            f"  {name}: N={n_nodes} N_k={summary['n_k']} "
            f"arrays={len(numpy_arrays)} "
            f"payload={payload_bytes} bytes sha256={payload_sha[:12]}..."
        )
        print(f"    wrote {manifest_path}")
        print(f"    wrote {payload_path}")

    print("Provenance: fixed seeds, no timestamps in fixture content; the")
    print("payload sha256 above pins the committed bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
