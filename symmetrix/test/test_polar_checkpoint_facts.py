"""MACE-POLAR-1 shipped-checkpoint facts pinned against the polar contract.

Two layers of tests live here:

* Committed-data tests (torch-free, always run) validate
  ``symmetrix/test/data/polar/checkpoint_facts.json`` — the extracted facts
  of the shipped MACE-POLAR-1 S/M/L checkpoints from the ``mace_polar_1``
  release — against the POLAR_BACKBONE contract vocabulary of
  ``symmetrix.mace_polar_execution_contract`` and against the
  interaction-count expectations carried by the support plan.

* Env-gated integration tests run the real extraction when a checkpoint
  path is supplied (``SYMMETRIX_POLAR_MODEL_S`` / ``_M`` / ``_L``, following
  the ``SYMMETRIX_MH1_MODEL`` precedent) and compare the freshly extracted
  facts — including contract fingerprints and constant-matrix digests — to
  the committed fixture.  Checkpoints are downloaded to
  ``~/.cache/symmetrix/test-models`` and never committed.

Plan-assumption note: the survey carried "S/M = 2 interactions, L = 3" from
the paper SI.  The shipped checkpoints pin S = 1, M = 2, L = 3 — S carries a
single interaction, per the checkpoint extraction itself (the sole evidence,
with each checkpoint's identity pinned by SHA-256 digest in the fixture).
The fixture records checkpoint-extracted counts, never assumed ones, so S is
pinned at one interaction and its MH1_UVU absence shares the same
interaction-count admission dependency as L (the v6 interaction-count lift),
rather than the full validation the plan expected for it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
from polar_oracle_checkout import optional_checkout

from symmetrix.mace_polar_execution_contract import (
    POLAR_CONTRACT_TAG,
    POLAR_SCHEMA,
    POLAR_SCHEMA_VERSION,
)

ORACLE_MACE_CHECKOUT = optional_checkout("SYMMETRIX_POLAR_ORACLE_MACE")
ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT = optional_checkout(
    "SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE"
)

FACTS_PATH = (
    Path(__file__).resolve().parent / "data" / "polar" / "checkpoint_facts.json"
)
MODEL_SIZES = ("S", "M", "L")

# Interaction counts extracted from the shipped checkpoints; the sole
# evidence is the checkpoint extraction, with each checkpoint's identity
# pinned by SHA-256 digest in the fixture.  The plan's carried paper-SI
# expectation was S/M = 2; the checkpoints pin S = 1.
SHIPPED_INTERACTION_COUNTS = {"S": 1, "M": 2, "L": 3}

# Readout classes the extractor's backbone admission accepts.
SUPPORTED_READOUT_CLASSES = {
    "LinearReadoutBlock",
    "NonLinearReadoutBlock",
    "NonLinearBiasReadoutBlock",
}

_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def facts():
    return json.loads(FACTS_PATH.read_text())


@pytest.fixture(scope="module")
def model_facts(facts):
    return facts["models"]


def test_fixture_schema_and_provenance(facts):
    assert facts["schema"] == "symmetrix.polar.checkpoint-facts"
    assert facts["version"] == 1
    source = facts["source"]
    assert source["release_tag"] == "mace_polar_1"
    assert source["repository"] == "ACEsuit/mace-foundations"
    for size in MODEL_SIZES:
        url = source["url_template"].format(size=size)
        assert url.endswith(f"/MACE-POLAR-1-{size}.model")
        assert url.startswith(
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
        )
    extraction = facts["extraction"]
    # Oracle pins must match the design-note pin.
    assert extraction["oracle_mace_sha"] == ("22f0809735bd4dd1deba80cf8e16f89913a35ff4")
    assert extraction["oracle_graph_electrostatics_sha"] == (
        "e0eb8945c21e203656f4e0bbece5d1da488b6753"
    )
    assert extraction["extractor"] == (
        "symmetrix.extract_mace_polar.extract_mace_polar_data"
    )


def test_all_shipped_models_are_pinned(model_facts):
    assert sorted(model_facts) == sorted(MODEL_SIZES)
    for size in MODEL_SIZES:
        entry = model_facts[size]
        assert entry["status"] == "available", f"model {size} missing from the pin"
        file_info = entry["file"]
        # Byte size must match the release asset listing, tying the pinned
        # digest to the exact published artifact.
        assert file_info["bytes"] == file_info["release_asset_bytes"]
        assert re.fullmatch(r"[0-9a-f]{64}", file_info["sha256"]) is not None


def test_pinned_interaction_counts_match_shipped_checkpoints(model_facts):
    for size in MODEL_SIZES:
        assert (
            model_facts[size]["interaction_count"] == (SHIPPED_INTERACTION_COUNTS[size])
        )
    # The sizes are strictly ordered by depth: S < M < L.
    counts = [model_facts[size]["interaction_count"] for size in MODEL_SIZES]
    assert counts == sorted(counts)


@pytest.mark.parametrize("size", MODEL_SIZES)
def test_facts_align_with_polar_contract_vocabulary(model_facts, size):
    entry = model_facts[size]
    assert entry["model_class"] == "PolarMACE"
    assert entry["keep_last_layer_irreps"] is True
    assert entry["has_zbl"] is False
    assert entry["readout_classes"][0] == "LinearReadoutBlock"
    assert set(entry["readout_classes"]) <= SUPPORTED_READOUT_CLASSES
    # The sigma set is the field smearing-width set.
    assert entry["sigma_set"] == entry["field_feature_widths"]

    polar = entry["execution_contracts"]["polar_backbone"]
    assert polar["status"] == "validated"
    assert polar["tag"] == POLAR_CONTRACT_TAG
    assert polar["schema"] == POLAR_SCHEMA
    assert polar["version"] == POLAR_SCHEMA_VERSION
    for key in (
        "structure_fingerprint",
        "semantic_fingerprint",
        "generation_fingerprint",
    ):
        assert _FINGERPRINT_PATTERN.match(polar[key]) is not None

    interaction_count = entry["interaction_count"]
    assert polar["interaction_count"] == interaction_count
    assert entry["spin_source_heads"]["count"] == interaction_count
    assert entry["feature_mixer"]["count"] == interaction_count
    assert entry["field_update_blocks"]["count"] == entry["num_update"]
    assert polar["recursion_step_count"] == entry["num_update"]
    assert (
        len(entry["field_update_blocks"]["runtime_parameter_counts"])
        == (entry["num_update"])
    )


def test_polar_backbone_fingerprints_distinguish_the_sizes(model_facts):
    fingerprints = [
        model_facts[size]["execution_contracts"]["polar_backbone"][
            "structure_fingerprint"
        ]
        for size in MODEL_SIZES
    ]
    assert len(set(fingerprints)) == len(MODEL_SIZES)


@pytest.mark.parametrize("size", MODEL_SIZES)
def test_constant_matrix_shapes_follow_contract_dimensions(model_facts, size):
    entry = model_facts[size]
    constants = entry["constant_matrices"]
    field_dimension = entry["field_dimension"]
    sigma_count = len(entry["sigma_set"])
    overlap_l = min(entry["atomic_multipoles_max_l"], entry["field_feature_max_l"])
    # GTOInternalFieldtoFeaturesBlock: [projections_dim, 4] with the [V, Ex,
    # Ey, Ez] external-field vector; GTOSelfInteractionBlock: one overlap per
    # receiving (sigma, l, m) channel.
    assert constants["external_field_projection"]["shape"] == [field_dimension, 4]
    assert constants["feature_self_interaction_overlaps"]["shape"] == [
        sigma_count * (overlap_l + 1) ** 2
    ]
    assert constants["energy_self_interaction_overlaps"]["shape"] == [
        (entry["atomic_multipoles_max_l"] + 1) ** 2
    ]
    for matrix in constants.values():
        assert _FINGERPRINT_PATTERN.match(f"sha256:{matrix['sha256_float64']}")


def test_kspace_cutoff_matches_the_oracle_heuristic(model_facts):
    # k_c = factor * 0.75 * (1/min sigma) * (max_l + 1)^0.3 * 3.0 with the
    # density smearing width folded into the sigma list (extensions.py:397).
    for size in MODEL_SIZES:
        entry = model_facts[size]
        assert entry["kspace_cutoff_factor_inferred"] == pytest.approx(1.0, rel=1e-6)


def test_execution_admission_outcomes_are_pinned(model_facts):
    # The interaction-count admission is lifted (the v6 task-20 lift this plan
    # consumes): every shipped checkpoint embeds and validates its MH1_UVU
    # identity with the count it declares, and the polar backbone identity
    # carries the unrolled chain for the model's own recursion depth.
    #
    # The v4 node program is a separate admission from the interaction count,
    # and the released sizes split on it: the bias readout the M and L
    # checkpoints read out through is a node-generation form the program
    # admits, so both build their node program (two and three layers, one per
    # interaction); S's single layer reads out through the plain linear block,
    # which the last-layer rule does not admit, and the refusal is recorded
    # with its own cause.  Whether the *evaluator* admits the backbone is a
    # third, separate rule (the node-program support matrix over the model's
    # irreps), and it is not a contract outcome.
    for size in MODEL_SIZES:
        contracts = model_facts[size]["execution_contracts"]
        interaction_count = model_facts[size]["interaction_count"]
        assert contracts["embedded"] == ["MH1_UVU", "POLAR_BACKBONE"]
        assert contracts["mh1_uvu"]["status"] == "embedded"
        assert contracts["mh1_uvu"]["validated"] is True
        assert contracts["mh1_uvu"]["interaction_count"] == interaction_count
        assert contracts["mh1_uvu"]["version"] == 3
        for key in (
            "structure_fingerprint",
            "semantic_fingerprint",
            "generation_fingerprint",
        ):
            assert _FINGERPRINT_PATTERN.match(contracts["mh1_uvu"][key]) is not None
        v4 = contracts["mh1_uvu_v4"]
        if size == "S":
            assert v4["status"] == "rejected"
            assert v4["reason"] == (
                "MH-1 node generation requires a linear readout for every "
                "layer but the last, and a nonlinear or bias readout for the "
                "last"
            )
        else:
            assert v4["status"] == "built"
            assert v4["tag"] == "symmetrix.execution.mh1_uvu/4"
            assert v4["version"] == 4
            assert v4["interaction_count"] == interaction_count
            assert v4["node_program_layer_count"] == interaction_count
        assert "graph_preparation_rejection" not in contracts


# ---------------------------------------------------------------------------
# Env-gated real-checkpoint extraction comparison
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import guard
    if ORACLE_MACE_CHECKOUT is not None and ORACLE_MACE_CHECKOUT.is_dir():
        sys.path.insert(0, str(ORACLE_MACE_CHECKOUT))
    if (
        ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT is not None
        and ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT.is_dir()
    ):
        sys.path.insert(0, str(ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT))
    import mace  # noqa: F401  (also sets the weights-only env e3nn requires)
    import torch
    from graph_longrange.gto_utils import gto_basis_kspace_cutoff  # noqa: F401

    polar_checkpoint_import_error = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    mace = None
    torch = None
    gto_basis_kspace_cutoff = None
    polar_checkpoint_import_error = exc


def _checkpoint_path(size: str) -> Path:
    if polar_checkpoint_import_error is not None:
        pytest.skip(
            "MACE-POLAR-1 checkpoint extraction requires "
            f"torch/e3nn/mace/graph_longrange: {polar_checkpoint_import_error}"
        )
    path = os.environ.get(f"SYMMETRIX_POLAR_MODEL_{size}")
    if not path:
        pytest.skip(
            f"set SYMMETRIX_POLAR_MODEL_{size} to run the MACE-POLAR-1-{size} "
            "extraction comparison"
        )
    path = Path(path)
    if not path.is_file():
        pytest.skip(f"MACE-POLAR-1-{size} checkpoint does not exist: {path}")
    return path


@pytest.mark.parametrize("size", MODEL_SIZES)
def test_real_checkpoint_extraction_matches_pinned_facts(model_facts, size):
    import hashlib

    import numpy as np

    from symmetrix.extract_mace_polar import extract_mace_polar_data
    from symmetrix.mace_polar_execution_contract import (
        POLAR_CONTRACT_KEY,
        validate_execution_polar_backbone_contract_for_model,
    )

    entry = model_facts[size]
    path = _checkpoint_path(size)
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        file_digest == entry["file"]["sha256"]
    ), f"{path.name} does not match the pinned MACE-POLAR-1-{size} digest"
    model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    model = model.to(torch.float64)
    data = extract_mace_polar_data(model)
    contracts = data["execution_contracts"]
    polar_contract = validate_execution_polar_backbone_contract_for_model(
        data, contracts[POLAR_CONTRACT_KEY]
    )

    assert data["num_interactions"] == entry["interaction_count"]
    assert data["num_elements"] == entry["element_coverage_count"]
    assert data["has_zbl"] == entry["has_zbl"]
    assert data["r_cut"] == entry["r_cut"]
    assert data["l_max"] == entry["l_max"]
    assert [readout["class"] for readout in data["readouts"]] == (
        entry["readout_classes"]
    )
    assert data["polar"]["num_recursion_steps"] == entry["num_update"]
    assert data["polar"]["atomic_multipoles_max_l"] == entry["atomic_multipoles_max_l"]
    assert data["polar"]["field_feature_max_l"] == entry["field_feature_max_l"]
    assert data["polar"]["field_feature_widths"] == entry["field_feature_widths"]
    assert data["polar"]["kspace_cutoff"] == entry["kspace_cutoff"]

    heuristic = gto_basis_kspace_cutoff(
        [data["polar"]["atomic_multipoles_smearing_width"]]
        + list(data["polar"]["field_feature_widths"]),
        max(
            data["polar"]["atomic_multipoles_max_l"],
            data["polar"]["field_feature_max_l"],
        ),
    )
    assert data["polar"]["kspace_cutoff"] / heuristic == pytest.approx(
        entry["kspace_cutoff_factor_inferred"]
    )

    pinned = entry["execution_contracts"]["polar_backbone"]
    assert polar_contract["structure_fingerprint"] == pinned["structure_fingerprint"]

    # The MH-1 identity the same extraction embeds is pinned too, so the
    # recorded admission outcome (embedded and validated with the model's own
    # interaction count) is compared against the real checkpoints rather than
    # assumed from the lift.
    from symmetrix.mace_nonlinear_execution_contract import (
        MH1_CONTRACT_KEY,
        normalize_execution_mh1_contract,
        validate_execution_mh1_contract_for_model,
    )
    from symmetrix.mace_polar_execution_contract import (
        make_execution_polar_mh1_v4_contract,
    )

    mh1 = validate_execution_mh1_contract_for_model(data, contracts[MH1_CONTRACT_KEY])
    assert normalize_execution_mh1_contract(mh1) == mh1
    recorded = entry["execution_contracts"]["mh1_uvu"]
    assert recorded["status"] == "embedded"
    assert recorded["validated"] is True
    assert recorded["tag"] == mh1["tag"]
    assert recorded["version"] == mh1["version"]
    assert recorded["interaction_count"] == mh1["interaction_count"]
    assert recorded["structure_fingerprint"] == mh1["structure_fingerprint"]
    assert recorded["semantic_fingerprint"] == mh1["semantic_fingerprint"]
    assert recorded["generation_fingerprint"] == mh1["generation_fingerprint"]

    v4_record = entry["execution_contracts"]["mh1_uvu_v4"]
    try:
        make_execution_polar_mh1_v4_contract(data)
    except ValueError as error:
        assert v4_record["status"] == "rejected"
        assert v4_record["reason"] == str(error)
    else:
        assert v4_record["status"] == "built"
        assert v4_record["interaction_count"] == entry["interaction_count"]

    def digest(buffer) -> str:
        array = np.asarray(buffer.detach().numpy(), dtype="<f8").reshape(-1)
        return hashlib.sha256(array.tobytes()).hexdigest()

    constants = entry["constant_matrices"]
    assert (
        list(model.external_field_contribution.matrix.shape)
        == (constants["external_field_projection"]["shape"])
    )
    assert (
        digest(model.external_field_contribution.matrix)
        == (constants["external_field_projection"]["sha256_float64"])
    )
    feature_self = model.electric_potential_descriptor.self_interaction_terms
    assert (
        digest(feature_self.overlap_constants)
        == (constants["feature_self_interaction_overlaps"]["sha256_float64"])
    )
    energy_self = model.coulomb_energy.self_interaction_terms
    assert (
        digest(energy_self.overlap_constants)
        == (constants["energy_self_interaction_overlaps"]["sha256_float64"])
    )
