"""One artifact identity envelope every family's metadata builder publishes.

The envelope is the artifact-identity subset of a ``*_metadata`` mapping: the
ABI tag and version the compiled descriptor announces, the codegen artifact id,
and the fingerprint triple the native loader checks.  Every builder below is
exercised on the fixtures the rest of the suite uses, against the envelope its
own module declares.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from symmetrix.jit import (
    ArtifactEnvelope,
    JitEnvelopeError,
    artifact_cache_inputs,
    validate_artifact_envelope,
)
from symmetrix.mace_jit_codegen import (
    R1_ARTIFACT_ENVELOPE,
    factorized_hip_plugin_metadata,
    jit_r1_cuda_plugin_metadata,
    jit_r1_host_plugin_metadata,
)
from symmetrix.mace_jit_operator_codegen import (
    OPERATOR_ARTIFACT_ENVELOPE,
    jit_m0_device_metadata,
    jit_m0_host_metadata,
    jit_r0_device_metadata,
)
from symmetrix.mace_nonlinear_execution_contract import (
    make_execution_mh1_contract,
    make_execution_mh1_v4_contract,
)
from symmetrix.mace_nonlinear_jit_codegen import (
    NONLINEAR_ARTIFACT_ENVELOPE,
    execution_mh1_conditioner_layout_metadata,
    execution_mh1_hip_module_v4_metadata,
    execution_mh1_node_program_metadata,
    jit_mh1_cuda_plugin_metadata,
    jit_mh1_cuda_plugin_v4_metadata,
    jit_mh1_host_plugin_metadata,
)
from symmetrix.oam_cuda_codegen import (
    OAM_CUDA_ARTIFACT_ENVELOPE,
    oam_cuda_artifact_metadata,
)
from symmetrix.oam_jit_codegen import OAM_HOST_ARTIFACT_ENVELOPE, oam_jit_metadata
from symmetrix.oam_receiver_codegen import (
    OAM_RECEIVER_ARTIFACT_ENVELOPE,
    oam_receiver_jit_metadata,
)

REPOSITORY = Path(__file__).resolve().parents[2]
CONTRACTS = REPOSITORY / "symmetrix" / "test" / "data" / "execution_contracts"
NEQUIP_DATA = REPOSITORY / "symmetrix" / "test" / "data" / "nequip"

HIP_TARGET = {
    "raw_agent_target": "gfx942:xnack-:sramecc+",
    "target_features": "sramecc+:xnack-",
    "native_subgroup_width": 64,
}
MH1_HIP_TARGET = {
    "architecture": "gfx942",
    "target_features": ["sramecc+", "xnack-"],
    "native_subgroup_width": 64,
    "compute_unit_count": 304,
}


def _contract(name: str) -> dict:
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mh1_v3_contract() -> dict:
    import test_mh1_jit_codegen

    return make_execution_mh1_contract(test_mh1_jit_codegen._definition())


@pytest.fixture(scope="module")
def mh1_v4_contract() -> dict:
    import test_execution_mh1_contract as contract_test

    definition = contract_test.mh1_node_definition.__wrapped__(
        contract_test.mh1_definition.__wrapped__()
    )
    return make_execution_mh1_v4_contract(definition)


# Every published envelope reachable from the Python generators.  The key names
# the builder whose returned metadata mapping is validated.
def _published_envelopes(mh1_v3_contract, mh1_v4_contract):
    r1 = _contract("jit_r1_mace_off23_small_contract.json")
    m0 = _contract("standard_m0_contract.json")
    r0 = _contract("standard_r0_contract.json")
    oam = json.loads(
        (NEQUIP_DATA / "oam_s_v2_minimal.json").read_text(encoding="utf-8")
    )
    return {
        "jit_r1_host_plugin_metadata": (
            jit_r1_host_plugin_metadata(r1, precision="float32"),
            R1_ARTIFACT_ENVELOPE,
        ),
        "jit_r1_host_plugin_metadata/float64": (
            jit_r1_host_plugin_metadata(r1, precision="float64"),
            R1_ARTIFACT_ENVELOPE,
        ),
        "jit_r1_cuda_plugin_metadata": (
            jit_r1_cuda_plugin_metadata(r1, 80),
            R1_ARTIFACT_ENVELOPE,
        ),
        "factorized_hip_plugin_metadata": (
            factorized_hip_plugin_metadata(r1, HIP_TARGET),
            R1_ARTIFACT_ENVELOPE,
        ),
        "jit_m0_device_metadata": (
            jit_m0_device_metadata(m0, precision="float32", target="sm_80"),
            OPERATOR_ARTIFACT_ENVELOPE,
        ),
        "jit_m0_host_metadata": (
            jit_m0_host_metadata(m0, precision="float32"),
            OPERATOR_ARTIFACT_ENVELOPE,
        ),
        "jit_r0_device_metadata": (
            jit_r0_device_metadata(r0, precision="float32", target="sm_80"),
            OPERATOR_ARTIFACT_ENVELOPE,
        ),
        "jit_mh1_host_plugin_metadata": (
            jit_mh1_host_plugin_metadata(mh1_v3_contract),
            NONLINEAR_ARTIFACT_ENVELOPE,
        ),
        "jit_mh1_cuda_plugin_metadata": (
            jit_mh1_cuda_plugin_metadata(mh1_v3_contract, 80),
            NONLINEAR_ARTIFACT_ENVELOPE,
        ),
        "execution_mh1_node_program_metadata/host": (
            execution_mh1_node_program_metadata(mh1_v4_contract, backend="host"),
            NONLINEAR_ARTIFACT_ENVELOPE,
        ),
        "execution_mh1_node_program_metadata/cuda": (
            execution_mh1_node_program_metadata(mh1_v4_contract, backend="cuda"),
            NONLINEAR_ARTIFACT_ENVELOPE,
        ),
        "jit_mh1_cuda_plugin_v4_metadata": (
            jit_mh1_cuda_plugin_v4_metadata(mh1_v4_contract, 80),
            NONLINEAR_ARTIFACT_ENVELOPE,
        ),
        "execution_mh1_hip_module_v4_metadata": (
            execution_mh1_hip_module_v4_metadata(mh1_v4_contract, MH1_HIP_TARGET),
            NONLINEAR_ARTIFACT_ENVELOPE,
        ),
        "oam_jit_metadata": (
            oam_jit_metadata(oam, "float64"),
            OAM_HOST_ARTIFACT_ENVELOPE,
        ),
        "oam_receiver_jit_metadata": (
            oam_receiver_jit_metadata(oam, "float64", host_target="host-isa-x86-64-v3"),
            OAM_RECEIVER_ARTIFACT_ENVELOPE,
        ),
        "oam_cuda_artifact_metadata": (
            oam_cuda_artifact_metadata(oam, "float32", 80),
            OAM_CUDA_ARTIFACT_ENVELOPE,
        ),
    }


def test_every_metadata_builder_publishes_its_envelope(
    mh1_v3_contract, mh1_v4_contract
):
    envelopes = _published_envelopes(mh1_v3_contract, mh1_v4_contract)
    assert len(envelopes) == 16
    for name, (metadata, envelope) in envelopes.items():
        view = validate_artifact_envelope(metadata, envelope)
        assert view["artifact_id"] == metadata["artifact_id"], name
        assert view["abi_version"] == metadata["abi_version"], name
        assert view["structure_fingerprint"] == metadata["structure_fingerprint"], name
        inputs = artifact_cache_inputs(metadata, envelope)
        assert inputs == {"tag": view["abi"], "version": view["abi_version"]}, name


def test_cache_inputs_publish_the_envelope_abi_tag(mh1_v3_contract):
    metadata = jit_mh1_host_plugin_metadata(mh1_v3_contract)
    assert artifact_cache_inputs(metadata, NONLINEAR_ARTIFACT_ENVELOPE) == {
        "tag": metadata["abi"],
        "version": metadata["abi_version"],
    }


def test_cache_inputs_read_the_operator_abi_tag_from_its_published_key():
    metadata = jit_m0_device_metadata(
        _contract("standard_m0_contract.json"), precision="float32", target="sm_80"
    )
    assert "abi" not in metadata
    assert artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE) == {
        "tag": metadata["schema"],
        "version": metadata["abi_version"],
    }


def test_nested_layout_metadata_is_not_an_envelope(mh1_v3_contract):
    # The conditioner layout is payload folded into the non-linear envelope: it
    # carries a generation fingerprint but no artifact id or ABI of its own.
    layout = execution_mh1_conditioner_layout_metadata(mh1_v3_contract)
    assert "artifact_id" not in layout
    assert "abi" not in layout
    with pytest.raises(JitEnvelopeError, match="abi is not an ABI tag"):
        validate_artifact_envelope(layout, NONLINEAR_ARTIFACT_ENVELOPE)


ENVELOPE = ArtifactEnvelope("fixture")
DIGEST = "sha256:" + "a" * 64


def _valid_metadata(**overrides):
    metadata = {
        "abi": "symmetrix.jit.fixture-plugin/1",
        "abi_version": 1,
        "artifact_id": "jit-fixture-gen11-f32-deadbeef",
        "contract_fingerprint": DIGEST,
        "semantic_fingerprint": "sha256:" + "b" * 64,
        "structure_fingerprint": "sha256:" + "c" * 64,
    }
    metadata.update(overrides)
    return metadata


def test_valid_envelope_is_returned_as_the_envelope_view():
    view = validate_artifact_envelope(_valid_metadata(), ENVELOPE)
    assert view == _valid_metadata()


def test_missing_metadata_is_rejected():
    with pytest.raises(JitEnvelopeError, match="metadata is not a mapping"):
        validate_artifact_envelope([("abi", "tag")], ENVELOPE)


@pytest.mark.parametrize("value", [None, "", 7, "not a tag!"])
def test_invalid_abi_tag_is_rejected(value):
    with pytest.raises(JitEnvelopeError, match="abi is not an ABI tag"):
        validate_artifact_envelope(_valid_metadata(abi=value), ENVELOPE)


@pytest.mark.parametrize("value", [None, 0, -1, True, "2"])
def test_invalid_abi_version_is_rejected(value):
    with pytest.raises(JitEnvelopeError, match="abi_version is not a positive integer"):
        validate_artifact_envelope(_valid_metadata(abi_version=value), ENVELOPE)


@pytest.mark.parametrize("value", [None, "", 4, "sub/dir/artifact", "-leading"])
def test_invalid_artifact_id_is_rejected(value):
    with pytest.raises(
        JitEnvelopeError, match="artifact_id is not a safe artifact name"
    ):
        validate_artifact_envelope(_valid_metadata(artifact_id=value), ENVELOPE)


@pytest.mark.parametrize("key", ["contract_fingerprint", "semantic_fingerprint"])
def test_non_canonical_fingerprint_is_rejected(key):
    with pytest.raises(JitEnvelopeError, match=f"{key} is not a canonical digest"):
        validate_artifact_envelope(_valid_metadata(**{key: "generation"}), ENVELOPE)
    with pytest.raises(JitEnvelopeError, match=f"{key} is not a canonical digest"):
        validate_artifact_envelope(_valid_metadata(**{key: "sha256:"}), ENVELOPE)


def test_missing_structure_fingerprint_is_rejected():
    metadata = _valid_metadata()
    del metadata["structure_fingerprint"]
    with pytest.raises(JitEnvelopeError, match="missing structure_fingerprint"):
        validate_artifact_envelope(metadata, ENVELOPE)


def test_missing_identity_fingerprint_is_rejected():
    metadata = _valid_metadata()
    del metadata["contract_fingerprint"]
    with pytest.raises(JitEnvelopeError, match="missing contract_fingerprint"):
        validate_artifact_envelope(metadata, ENVELOPE)


def test_missing_semantic_fingerprint_is_rejected():
    metadata = _valid_metadata()
    del metadata["semantic_fingerprint"]
    with pytest.raises(JitEnvelopeError, match="missing semantic_fingerprint"):
        validate_artifact_envelope(metadata, ENVELOPE)


def test_foreign_generation_fingerprint_is_still_validated():
    # Every declared member of the fingerprint vocabulary is checked, so a
    # family cannot publish a stray value for a vocabulary it never reads.
    with pytest.raises(
        JitEnvelopeError, match="generation_fingerprint is not a canonical digest"
    ):
        validate_artifact_envelope(
            _valid_metadata(generation_fingerprint="v4"), ENVELOPE
        )


def test_unknown_identity_fingerprint_in_the_profile_is_rejected():
    envelope = replace(ENVELOPE, identity_fingerprint="revision_fingerprint")
    with pytest.raises(JitEnvelopeError, match="unknown identity fingerprint"):
        validate_artifact_envelope(_valid_metadata(), envelope)


def test_operator_envelope_needs_only_its_structure_fingerprint():
    envelope = ArtifactEnvelope(
        "fixture operator",
        abi_key="schema",
        identity_fingerprint=None,
        semantic_fingerprint=False,
        capabilities_key="capabilities",
    )
    metadata = {
        "schema": "symmetrix.jit.operator-module/1",
        "abi_version": 1,
        "artifact_id": "jit-m0-gen11-f32-deadbeef",
        "structure_fingerprint": DIGEST,
        "capabilities": 0x1F,
    }
    view = validate_artifact_envelope(metadata, envelope)
    assert view["abi"] == metadata["schema"]
    assert view["capabilities"] == 0x1F


@pytest.mark.parametrize("value", [None, 0, -3, True, "31"])
def test_invalid_capability_word_is_rejected(value):
    envelope = ArtifactEnvelope("fixture operator", capabilities_key="capabilities")
    metadata = _valid_metadata(capabilities=value)
    with pytest.raises(JitEnvelopeError, match="capabilities is not a positive"):
        validate_artifact_envelope(metadata, envelope)


def test_missing_capability_word_is_rejected():
    envelope = ArtifactEnvelope("fixture operator", capabilities_key="capabilities")
    with pytest.raises(JitEnvelopeError, match="capabilities is not a positive"):
        validate_artifact_envelope(_valid_metadata(), envelope)
