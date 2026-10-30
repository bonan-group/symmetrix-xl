"""Pin MACE-POLAR-1 S/M/L checkpoint facts into the polar test contract.

Downloads (or accepts local copies of) the shipped MACE-POLAR-1 checkpoints
from the ``mace_polar_1`` release of ACEsuit/mace-foundations, runs the
in-tree PolarMACE backbone extractor over each, and writes the extracted
facts to ``symmetrix/test/data/polar/checkpoint_facts.json``:

* backbone shape facts — interaction count, sigma set, max_l's, field widths
  and norms, k-space cutoff (plus the inferred cutoff factor against the
  oracle heuristic), recursion step count, element coverage, ZBL presence,
  keep-last-layer flag, readout classes, spin-source head / feature-mixer /
  field-update shapes as the POLAR_BACKBONE contract records them;
* the long-range constant matrices carried by the checkpoints — the
  external-field projection matrix and the feature/energy self-interaction
  overlap vectors — as shapes plus float64-content digests;
* the execution-contract outcome per model: the embedded contract keys, the
  POLAR_BACKBONE identity fingerprints, the MH1_UVU outcome, and the exact
  graph-preparation rejection each model hits under the current admission
  (distinct errors for distinct causes, mirroring the v6 fail-fast rule);
* download provenance — release URL, byte size, and SHA-256 per file.

Checkpoints themselves are never committed; only the extracted facts are.

Oracle pin (read-only checkouts, never imported by in-tree code; the
in-tree tests load checkpoints with these trees on ``sys.path``):

* mace            22f0809735bd4dd1deba80cf8e16f89913a35ff4 (v0.3.16-2-g22f08097)
* graph_electrostatics e0eb8945c21e203656f4e0bbece5d1da488b6753 (v0.4.4)

Usage::

    .venv/bin/python benchmarks/polar_checkpoint_facts_runner.py \
        --oracle-mace /path/to/mace \
        --oracle-graph-electrostatics /path/to/graph_electrostatics \
        --checkpoint-dir ~/.cache/symmetrix/test-models \
        --output symmetrix/test/data/polar/checkpoint_facts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

ORACLE_MACE_SHA = "22f0809735bd4dd1deba80cf8e16f89913a35ff4"
ORACLE_MACE_REF = "v0.3.16-2-g22f08097"
ORACLE_GRAPH_ELECTROSTATICS_SHA = "e0eb8945c21e203656f4e0bbece5d1da488b6753"
ORACLE_GRAPH_LONGRANGE_VERSION = "0.4.4"

RELEASE_TAG = "mace_polar_1"
RELEASE_REPOSITORY = "ACEsuit/mace-foundations"
RELEASE_URL_TEMPLATE = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/"
    "mace_polar_1/MACE-POLAR-1-{size}.model"
)
MODEL_SIZES = ("S", "M", "L")

FACTS_SCHEMA = "symmetrix.polar.checkpoint-facts"
FACTS_VERSION = 1

# Expected byte sizes from the release asset listing (api.github.com,
# repos/ACEsuit/mace-foundations/releases, tag mace_polar_1).
RELEASE_ASSET_BYTES = {
    "S": 33375439,
    "M": 68133235,
    "L": 129808715,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_float64(values) -> str:
    import numpy as np

    array = np.asarray(values, dtype="<f8").reshape(-1)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _linear_io(descriptor) -> dict[str, Any]:
    return {
        "input_irreps": descriptor["irreps"]["input"],
        "output_irreps": descriptor["irreps"]["output"],
        "input_dimension": int(descriptor["dimensions"]["input"]),
        "output_dimension": int(descriptor["dimensions"]["output"]),
    }


def _extract_model_facts(model, checkpoint_path: Path, size: str) -> dict[str, Any]:
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data
    from symmetrix.mace_nonlinear_execution_contract import (
        MH1_CONTRACT_KEY,
        POLAR_CONTRACT_KEY,
        make_execution_mh1_v4_contract,
        validate_execution_mh1_contract_for_model,
        validate_execution_polar_backbone_contract_for_model,
    )

    data = extract_mace_nonlinear_data(model)
    polar = data["polar"]
    contracts = data.get("execution_contracts", {})

    polar_contract = validate_execution_polar_backbone_contract_for_model(
        data, contracts[POLAR_CONTRACT_KEY]
    )

    # The interaction-count admission is lifted, so every shipped checkpoint
    # embeds and validates its MH1_UVU identity with the count it declares;
    # the record carries the identity the same way the polar backbone's does.
    mh1_status: dict[str, Any]
    if MH1_CONTRACT_KEY in contracts:
        normalized = validate_execution_mh1_contract_for_model(
            data, contracts[MH1_CONTRACT_KEY]
        )
        mh1_status = {
            "status": "embedded",
            "validated": True,
            "tag": normalized["tag"],
            "version": int(normalized["version"]),
            "interaction_count": int(normalized["interaction_count"]),
            "structure_fingerprint": normalized["structure_fingerprint"],
            "semantic_fingerprint": normalized["semantic_fingerprint"],
            "generation_fingerprint": normalized["generation_fingerprint"],
        }
    else:
        mh1_status = {"status": "absent"}

    # The v4 node program's own support matrix is what calculator JIT
    # validation builds next, so its outcome (built, or the ValueError a real
    # calculator setup observes) is recorded beside the v3 identity.  It is a
    # node-generation rule, not the interaction-count admission: the bias
    # readout the M and L checkpoints read out through is admitted, so those
    # two build their node program, while S's single layer reads out through
    # the plain linear block, which the last-layer rule does not admit.
    try:
        v4 = make_execution_mh1_v4_contract(data)
        mh1_v4_status: dict[str, Any] = {
            "status": "built",
            "tag": v4["tag"],
            "version": int(v4["version"]),
            "interaction_count": int(v4["interaction_count"]),
            "node_program_layer_count": len(v4["node_program"]["layers"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        mh1_v4_status = {"status": "rejected", "reason": str(error)}

    from graph_longrange.gto_utils import gto_basis_kspace_cutoff

    heuristic = gto_basis_kspace_cutoff(
        [polar["atomic_multipoles_smearing_width"]]
        + list(polar["field_feature_widths"]),
        max(polar["atomic_multipoles_max_l"], polar["field_feature_max_l"]),
    )

    external_field = model.external_field_contribution
    feature_self = model.electric_potential_descriptor.self_interaction_terms
    energy_self = model.coulomb_energy.self_interaction_terms

    field = polar_contract["field"]
    charges = polar_contract["charges"]
    heads = polar_contract["spin_source_heads"]
    mixer_linears = polar_contract["feature_mixer"]["linears"]
    updates = polar_contract["runtime_parameter_packs"]["field_update_blocks"]

    return {
        "status": "available",
        "file": {
            "name": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256_file(checkpoint_path),
            "source_url": RELEASE_URL_TEMPLATE.format(size=size),
            "release_asset_bytes": RELEASE_ASSET_BYTES[size],
        },
        "model_class": polar["class"],
        "interaction_count": int(data["num_interactions"]),
        "readout_classes": [readout["class"] for readout in data["readouts"]],
        "sigma_set": [float(width) for width in polar["field_feature_widths"]],
        "atomic_multipoles_max_l": int(polar["atomic_multipoles_max_l"]),
        "atomic_multipoles_smearing_width": float(
            polar["atomic_multipoles_smearing_width"]
        ),
        "field_feature_max_l": int(polar["field_feature_max_l"]),
        "field_feature_widths": [
            float(width) for width in polar["field_feature_widths"]
        ],
        "field_feature_norms": [
            float(value) for value in polar["field_feature_norms"]["values"]
        ],
        "kspace_cutoff": float(polar["kspace_cutoff"]),
        "kspace_cutoff_heuristic": float(heuristic),
        "kspace_cutoff_factor_inferred": float(polar["kspace_cutoff"])
        / float(heuristic),
        "num_update": int(polar["num_recursion_steps"]),
        "element_coverage_count": int(data["num_elements"]),
        "has_zbl": bool(data["has_zbl"]),
        "keep_last_layer_irreps": bool(polar["keep_last_layer_irreps"]),
        # The checkpoint's self-interaction policy: the long-range nodes'
        # include flags are contract data, and these two fields are its
        # source in the model file.
        "include_electrostatic_self_interaction": bool(
            polar["include_electrostatic_self_interaction"]
        ),
        "field_si": bool(polar["field_si"]),
        "r_cut": float(data["r_cut"]),
        "l_max": int(data["l_max"]),
        "charges_irreps": charges["irreps"],
        "charges_dimension": int(charges["dimension"]),
        "field_irreps": field["irreps"],
        "field_dimension": int(field["dimension"]),
        "potential_irreps": polar_contract["potential"]["irreps"],
        "potential_dimension": int(polar_contract["potential"]["dimension"]),
        "node_feats_irreps": polar_contract["node_feats_irreps"],
        "spin_source_heads": {
            "count": len(heads),
            "zero_charges": [bool(head["zero_charges"]) for head in heads],
            "linear": [_linear_io(head["linear"]) for head in heads],
        },
        "feature_mixer": {
            "count": len(mixer_linears),
            "linear": [_linear_io(linear) for linear in mixer_linears],
        },
        "field_update_blocks": {
            "count": int(polar_contract["recursion_step_count"]),
            "runtime_parameter_counts": [
                sum(int(pack[name]["parameter_count"]) for name in pack)
                for pack in updates
            ],
        },
        "constant_matrices": {
            "external_field_projection": {
                "shape": list(external_field.matrix.shape),
                "sha256_float64": _sha256_float64(external_field.matrix),
            },
            "feature_self_interaction_overlaps": {
                "shape": list(feature_self.overlap_constants.shape),
                "sha256_float64": _sha256_float64(feature_self.overlap_constants),
            },
            "energy_self_interaction_overlaps": {
                "shape": list(energy_self.overlap_constants.shape),
                "sha256_float64": _sha256_float64(energy_self.overlap_constants),
            },
        },
        "execution_contracts": {
            "embedded": sorted(contracts),
            "polar_backbone": {
                "status": "validated",
                "tag": polar_contract["tag"],
                "schema": polar_contract["schema"],
                "version": polar_contract["version"],
                "interaction_count": int(polar_contract["interaction_count"]),
                "recursion_step_count": int(polar_contract["recursion_step_count"]),
                "structure_fingerprint": polar_contract["structure_fingerprint"],
                "semantic_fingerprint": polar_contract["semantic_fingerprint"],
                "generation_fingerprint": polar_contract["generation_fingerprint"],
            },
            "mh1_uvu": mh1_status,
            "mh1_uvu_v4": mh1_v4_status,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-mace",
        type=Path,
        required=True,
        help="read-only mace checkout providing the PolarMACE classes",
    )
    parser.add_argument(
        "--oracle-graph-electrostatics",
        type=Path,
        required=True,
        help="read-only graph_electrostatics checkout providing graph_longrange",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path.home() / ".cache/symmetrix/test-models",
        help="directory holding MACE-POLAR-1-<S/M/L>.model files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "symmetrix/test/data/polar/checkpoint_facts.json",
        help="destination facts fixture (committed)",
    )
    args = parser.parse_args()

    for checkout in (args.oracle_mace, args.oracle_graph_electrostatics):
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(REPO_ROOT / "symmetrix/source"))

    import torch

    import mace  # noqa: F401  (also sets the weights-only env e3nn requires)

    models: dict[str, Any] = {}
    for size in MODEL_SIZES:
        checkpoint_path = args.checkpoint_dir / f"MACE-POLAR-1-{size}.model"
        if not checkpoint_path.is_file():
            models[size] = {
                "status": "absent",
                "file": {
                    "name": checkpoint_path.name,
                    "source_url": RELEASE_URL_TEMPLATE.format(size=size),
                },
                "absence_reason": (
                    f"checkpoint not found at {checkpoint_path}; the "
                    f"{RELEASE_TAG} release asset was not pinned"
                ),
            }
            continue
        model = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        ).to(torch.float64)
        models[size] = _extract_model_facts(model, checkpoint_path, size)
        print(f"pinned {size}: {models[size]['interaction_count']} interaction(s)")

    fixture = {
        "schema": FACTS_SCHEMA,
        "version": FACTS_VERSION,
        "source": {
            "release_tag": RELEASE_TAG,
            "repository": RELEASE_REPOSITORY,
            "url_template": RELEASE_URL_TEMPLATE,
        },
        "extraction": {
            "extractor": "symmetrix.extract_mace_nonlinear.extract_mace_nonlinear_data",
            "oracle_mace_sha": ORACLE_MACE_SHA,
            "oracle_mace_ref": ORACLE_MACE_REF,
            "oracle_graph_electrostatics_sha": ORACLE_GRAPH_ELECTROSTATICS_SHA,
            "oracle_graph_longrange_version": ORACLE_GRAPH_LONGRANGE_VERSION,
        },
        "models": models,
    }
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
