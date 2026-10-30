"""The PolarMACE family's schema over the shared nonlinear backbone JSON.

The polar family's models are MH-1 backbones carrying the polar deltas, so the
two families' schemas share one structural validator and differ in exactly the
spellings the released MACE-POLAR-1 checkpoints use: the family's own model
type, ``scale``/``shift`` serialized as a leading size-1 tensor, and the
atomic-energy vector serialized as ``[num_elements]`` rather than
``[1, num_elements]``.

Two authorities validate one file -- the frontend validator
(:mod:`symmetrix.mace_polar_schema`), which runs before the native parse, and
the native schema (``libsymmetrix/source/mace_polar_schema.cpp``), which is the
one that actually consumes the data -- so this module pins them against the
same mutations: every violation the frontend names is one the native parse
names too, and the constants each side reads the other's file by (the model
type, the contract key, and the contract tag) are pinned behaviorally rather
than by reading either source.  It also pins the other half of the split: the
MH-1 schema still refuses the released spellings, so the polar family's own
rule row is what admits them.

The synthetic model the mutations run over is the backbone builder of
:mod:`test_mace_polar_backbone`; it needs torch, e3nn, and mace, so every test
here skips with the missing prerequisite named when they are unavailable.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from symmetrix.extract_mace_polar import extract_mace_polar_data
from symmetrix.mace_polar_schema import (
    POLAR_CONTRACT_KEY,
    POLAR_CONTRACT_TAG,
    POLAR_FORMAT_VERSION,
    POLAR_MODEL_TYPE,
    validate_mace_polar_model_data,
)
from test_mace_polar_backbone import _build_polar_shaped_model


def _foreign_model_type(data):
    data["model_type"] = "MACE_Nonlinear"


def _unknown_format_version(data):
    data["symmetrix_format_version"] = 4


def _leading_size_two_scale(data):
    data["scale_shift"]["scale"]["shape"] = [2]


def _rank_three_atomic_energies(data):
    data["atomic_energies"]["shape"] = [1, *data["atomic_energies"]["shape"]]


def _missing_polar_block(data):
    del data["polar"]


def _linear_backbone_readout(data):
    data["polar"]["keep_last_layer_irreps"] = False


def _truncated_spin_source_heads(data):
    data["polar"]["spin_source_heads"].pop()


def _truncated_feature_mixer(data):
    data["polar"]["feature_mixer"]["linears"].pop()


def _truncated_field_update_chain(data):
    data["polar"]["field_update_blocks"].pop()


def _empty_charge_irreps(data):
    data["polar"]["charges_irreps"] = ""


def _non_positive_kspace_cutoff(data):
    data["polar"]["kspace_cutoff"] = 0.0


def _missing_polar_contract(data):
    del data["execution_contracts"][POLAR_CONTRACT_KEY]


def _v1_polar_contract(data):
    data["execution_contracts"][POLAR_CONTRACT_KEY]["tag"] = (
        "symmetrix.execution.polar_backbone/1"
    )


def _renamed_polar_contract(data):
    data["execution_contracts"]["POLAR_BACKBONE_V2"] = data["execution_contracts"].pop(
        POLAR_CONTRACT_KEY
    )


#: The violations the two validators must agree on: a name, the mutation that
#: applies the violation to one extracted file, and the token both rejections
#: name -- so a mutation that stopped being a violation, or a validator that
#: stopped naming what it rejected, fails here.
MUTATIONS = (
    ("foreign model type", _foreign_model_type, "model_type"),
    ("unknown format version", _unknown_format_version, "format-version"),
    ("leading size-2 scale", _leading_size_two_scale, "scale"),
    ("rank-three atomic energies", _rank_three_atomic_energies, "atomic"),
    ("missing polar block", _missing_polar_block, "polar"),
    ("linear backbone readout", _linear_backbone_readout, "keep_last_layer_irreps"),
    ("truncated spin-source heads", _truncated_spin_source_heads, "spin_source_heads"),
    ("truncated feature mixer", _truncated_feature_mixer, "feature_mixer"),
    (
        "truncated field-update chain",
        _truncated_field_update_chain,
        "field_update_blocks",
    ),
    ("empty charge irreps", _empty_charge_irreps, "charges_irreps"),
    ("non-positive k-space cutoff", _non_positive_kspace_cutoff, "kspace_cutoff"),
    ("missing polar contract", _missing_polar_contract, POLAR_CONTRACT_KEY),
    ("v1 polar contract", _v1_polar_contract, "polar_backbone"),
    ("renamed polar contract", _renamed_polar_contract, POLAR_CONTRACT_KEY),
)
MUTATION_IDS = [name for name, _, _ in MUTATIONS]


@pytest.fixture(scope="module")
def polar_data() -> dict[str, Any]:
    """The family's file for a synthetic PolarMACE-shaped backbone."""

    return extract_mace_polar_data(_build_polar_shaped_model())


def released_spellings(data: Mapping[str, Any]) -> dict[str, Any]:
    """The released checkpoints' shape spellings for one extracted file.

    A released checkpoint serializes ``scale``/``shift`` as a leading size-1
    tensor and the atomic energies as a vector, where the synthetic backbone
    (and every MH-1 file) carries a scalar and a leading size-1 matrix.
    """

    released = copy.deepcopy(data)
    released["scale_shift"]["scale"]["shape"] = [1]
    released["scale_shift"]["shift"]["shape"] = [1]
    released["atomic_energies"]["shape"] = [len(released["model_atomic_numbers"])]
    return released


def write_model(path: Path, data: Mapping[str, Any]) -> Path:
    """Write one model file and return its path."""

    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


def native_polar_evaluator(path: Path):
    """The native polar evaluator for one model file.

    The class is the family entry's float64 variant, so this is the parse the
    frontend's declared evaluator runs.
    """

    import symmetrix

    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    return symmetrix.MACEPolarKokkos(str(path))


def test_the_extracted_file_carries_the_family_stamp_and_contract(polar_data):
    """The extractor's stamps are the schema's constants, and validate as one.

    The family boundary is the stamp the family's own extractor writes, so the
    two must be the same value: a registry that detects the family by one
    spelling and a schema that validates another would admit no file at all.
    """

    assert polar_data["model_type"] == POLAR_MODEL_TYPE
    assert polar_data["symmetrix_format_version"] == POLAR_FORMAT_VERSION
    assert polar_data["execution_contracts"][POLAR_CONTRACT_KEY]["tag"] == (
        POLAR_CONTRACT_TAG
    )
    assert validate_mace_polar_model_data(polar_data) is polar_data


def test_the_frontend_schema_admits_both_families_spellings(polar_data):
    """The polar row admits the synthetic and the released shape spellings."""

    released = released_spellings(polar_data)

    assert released["scale_shift"]["scale"]["shape"] == [1]
    assert released["atomic_energies"]["shape"] == [
        len(polar_data["model_atomic_numbers"])
    ]
    assert validate_mace_polar_model_data(released) is released


@pytest.mark.parametrize(
    ("mutate", "token"),
    [(mutate, token) for _, mutate, token in MUTATIONS],
    ids=MUTATION_IDS,
)
def test_the_frontend_schema_rejects_every_violation_by_name(polar_data, mutate, token):
    violated = copy.deepcopy(polar_data)
    mutate(violated)

    with pytest.raises(ValueError, match=token):
        validate_mace_polar_model_data(violated)


@pytest.mark.parametrize(
    ("mutate", "token"),
    [(mutate, token) for _, mutate, token in MUTATIONS],
    ids=MUTATION_IDS,
)
def test_the_native_schema_rejects_every_violation_the_frontend_rejects(
    tmp_path, polar_data, mutate, token
):
    """Both authorities report the same violations, so neither can drift.

    The frontend validator reports a polar-specific rejection before the native
    parse; it is not a second authority over the file, so a violation it names
    must be one the native parse names too.  A file that passed one and failed
    the other is exactly the drift the shared backbone validator exists to
    prevent.
    """

    violated = copy.deepcopy(polar_data)
    mutate(violated)

    with pytest.raises(ValueError, match=token):
        validate_mace_polar_model_data(violated)
    path = write_model(tmp_path / "mutated.json", violated)
    with pytest.raises(Exception, match=token):
        native_polar_evaluator(path)


def test_the_native_schema_loads_the_family_file_and_its_released_spellings(
    tmp_path, polar_data
):
    """The released spellings load, and the family stamp is what admits them."""

    evaluator = native_polar_evaluator(write_model(tmp_path / "polar.json", polar_data))
    assert evaluator.has_polar_long_range_contract
    assert (
        evaluator.polar_schedule["recursion_step_count"]
        == (polar_data["polar"]["num_recursion_steps"])
    )

    released = released_spellings(polar_data)
    released_evaluator = native_polar_evaluator(
        write_model(tmp_path / "released.json", released)
    )
    assert released_evaluator.has_polar_long_range_contract
    assert released_evaluator.family_program["family"] == "polar-mace"

    unstamped = copy.deepcopy(polar_data)
    _foreign_model_type(unstamped)
    with pytest.raises(ValueError, match=POLAR_MODEL_TYPE):
        native_polar_evaluator(write_model(tmp_path / "unstamped.json", unstamped))


def test_the_mh1_schema_still_refuses_the_released_polar_spellings(
    tmp_path, polar_data
):
    """The polar spellings are the polar row's, not a loosening of MH-1's.

    The released checkpoints' ``scale_shift`` spelling was the blocker that
    stopped the family's models loading.  The polar family's own rule row
    admits it; an MH-1 file is still held to the scalar rule it was held to
    before, and the polar evaluator is the only class that reads the released
    spelling.
    """

    import symmetrix

    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    released = released_spellings(polar_data)
    mh1_shaped = copy.deepcopy(released)
    _foreign_model_type(mh1_shaped)

    mh1_path = write_model(tmp_path / "released-mh1.json", mh1_shaped)
    with pytest.raises(ValueError, match="scale and shift must be scalar tensors"):
        symmetrix.MACENonlinearKokkos(str(mh1_path))

    assert native_polar_evaluator(
        write_model(tmp_path / "released-polar.json", released)
    ).has_polar_long_range_contract
