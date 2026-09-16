import json
from copy import deepcopy
from pathlib import Path

import pytest

from symmetrix.jit_operator_codegen import (
    M0_CAPABILITIES_V1,
    R0_CAPABILITIES_V1,
    jit_m0_device_metadata,
    jit_r0_device_metadata,
    render_jit_m0_device_module,
    render_jit_m0_host_plugin,
    render_jit_r0_device_module,
)


CONTRACTS = Path(__file__).resolve().parent / "data" / "execution_contracts"


def _contract(name):
    return json.loads((CONTRACTS / name).read_text())


def _without_fingerprints(contract):
    contract = deepcopy(contract)
    for name in (
        "fingerprint",
        "semantic_fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
    ):
        contract.pop(name, None)
    return contract


@pytest.mark.parametrize("schedule", ("chunk32", "table"))
@pytest.mark.parametrize(
    "precision,scalar", (("float32", "float"), ("float64", "double"))
)
def test_m0_device_source_covers_state_free_forward_reverse(
    schedule, precision, scalar
):
    source, metadata = render_jit_m0_device_module(
        _contract("standard_m0_contract.json"),
        precision=precision,
        target="sm_120",
        schedule=schedule,
    )

    assert metadata["capabilities"] == M0_CAPABILITIES_V1
    assert metadata["artifact_id"].endswith(f"-{schedule}-sm_120")
    assert f"using Scalar = {scalar};" in source
    assert "symmetrix_m0_forward_v1" in source
    assert "symmetrix_m0_reverse_v1" in source
    assert "capture_input_scale_adjoint" in source
    assert "atomicAdd(args.input_scale_adjoint+node, scale)" in source
    assert "long long num_nodes;" in source
    assert "constexpr int channel_count = 128;" in source
    assert "const long long node = owner/channel_count;" in source
    assert "owner/args.channels" not in source
    assert 1_000_188 * 40 * 128 > 2**31 - 1
    if schedule == "chunk32":
        assert "m0_fwd_0" in source
        assert "m0_term_components" not in source
    else:
        assert "m0_term_components" in source
        assert "m0_fwd_0" not in source


def test_m0_artifact_identity_is_channel_target_and_topology_specific():
    contract = _without_fingerprints(_contract("standard_m0_contract.json"))
    baseline = jit_m0_device_metadata(
        contract, precision="float32", target="sm_120", schedule="chunk32"
    )

    resized = deepcopy(contract)
    resized["channels"] = 48
    resized["type_count"] = 7
    resized_metadata = jit_m0_device_metadata(
        resized, precision="float32", target="sm_120", schedule="chunk32"
    )
    assert (
        resized_metadata["structure_fingerprint"] == baseline["structure_fingerprint"]
    )
    assert resized_metadata["artifact_id"] != baseline["artifact_id"]

    resized["channels"] = contract["channels"]
    resized_metadata = jit_m0_device_metadata(
        resized, precision="float32", target="sm_120", schedule="chunk32"
    )
    assert resized_metadata["artifact_id"] == baseline["artifact_id"]

    assert (
        jit_m0_device_metadata(
            contract, precision="float32", target="sm_90", schedule="chunk32"
        )["artifact_id"]
        != baseline["artifact_id"]
    )
    assert (
        jit_m0_device_metadata(
            contract, precision="float64", target="sm_120", schedule="chunk32"
        )["artifact_id"]
        != baseline["artifact_id"]
    )
    assert (
        jit_m0_device_metadata(
            contract, precision="float32", target="sm_120", schedule="table"
        )["artifact_id"]
        != baseline["artifact_id"]
    )

    changed = deepcopy(contract)
    changed["monomial_groups"][0]["terms"][0] = [1]
    changed_metadata = jit_m0_device_metadata(
        changed, precision="float32", target="sm_120", schedule="chunk32"
    )
    assert (
        changed_metadata["structure_fingerprint"] != baseline["structure_fingerprint"]
    )
    assert changed_metadata["artifact_id"] != baseline["artifact_id"]


def test_m0_generated_module_accepts_correlation_four():
    contract = _without_fingerprints(_contract("standard_m0_contract.json"))
    contract["correlation"] = 4
    for group in contract["monomial_groups"]:
        group["degree_counts"].append(0)
    group = contract["monomial_groups"][0]
    group["terms"].append([0, 0, 0, 0])
    group["terms"].sort(key=lambda term: (len(term), tuple(term)))
    group["term_count"] = len(group["terms"])
    group["degree_counts"][3] = 1

    metadata = jit_m0_device_metadata(
        contract, precision="float32", target="sm_120", schedule="chunk32"
    )
    assert metadata["correlation"] == 4
    assert metadata["term_count"] == 423

    source, rendered = render_jit_m0_device_module(
        contract, precision="float64", target="sm_120", schedule="chunk32"
    )
    assert rendered["correlation"] == 4
    assert "x[0]*x[0]*x[0]*x[0]" in source

    table_source, table_metadata = render_jit_m0_device_module(
        contract, precision="float32", target="sm_120", schedule="table"
    )
    assert table_metadata["correlation"] == 4
    assert "term_count*correlation" in table_source
    assert "x[m0_term_components[term*correlation+other]]" in table_source


def test_m0_device_module_rejects_correlation_above_generated_capacity():
    contract = _without_fingerprints(_contract("standard_m0_contract.json"))
    contract["correlation"] = 5
    for group in contract["monomial_groups"]:
        group["degree_counts"].extend([0, 0])

    with pytest.raises(ValueError, match="no greater than four"):
        jit_m0_device_metadata(
            contract, precision="float32", target="sm_120", schedule="chunk32"
        )


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_m0_host_source_uses_shared_abi_types(precision):
    source, metadata = render_jit_m0_host_plugin(
        _contract("standard_m0_contract.json"), precision=precision
    )

    assert "const SymmetrixJitM0HostArgsV1* args" in source
    assert "const SymmetrixJitM0HostPluginV1*" in source
    assert "static const SymmetrixJitM0HostPluginV1 descriptor" in source
    assert "sizeof(SymmetrixJitM0HostPluginV1)" in source
    assert "SYMMETRIX_JIT_M0_HOST_PLUGIN_BYTE_ORDER_V1" in source
    assert "struct Args" not in source
    assert "struct Plugin" not in source
    assert "input_raw" not in source
    assert metadata["capabilities"] == 0x1F
    assert "constexpr int host_channel_tile" in source
    assert "owner/((channels+host_channel_tile-1)/host_channel_tile)" in source
    assert "host_capabilities" in source
    assert "descriptor_channel_tile" in source
    assert "SYMMETRIX_M0_ALWAYS_INLINE void m0_fwd_0" in source
    assert "SYMMETRIX_M0_ALWAYS_INLINE void m0_rev_0" in source


@pytest.mark.parametrize(
    "precision,scalar", (("float32", "float"), ("float64", "double"))
)
def test_r0_device_source_covers_density_and_coordinate_reverse(precision, scalar):
    source, metadata = render_jit_r0_device_module(
        _contract("standard_r0_contract.json"),
        precision=precision,
        target="gfx1100",
    )

    assert metadata["capabilities"] == R0_CAPABILITIES_V1
    assert metadata["artifact_id"].endswith("-edge-batch1-gfx1100")
    assert f"using Scalar = {scalar};" in source
    for symbol in (
        "symmetrix_r0_density_prepare_v1",
        "symmetrix_r0_forward_v1",
        "symmetrix_r0_reverse_prepare_v1",
        "symmetrix_r0_coordinate_reverse_v1",
    ):
        assert symbol in source
    assert "coordinates_are_unit" in source
    assert "use_precomputed_scale_adjoint" in source
    assert "unsigned int receiver_base;" in source
    assert "args.edge_receivers[edge])-args.receiver_base" in source
    assert "void direct_harmonic_gradients(" in source
    assert "if (edge_gradients == nullptr)" in source
    assert "args.coordinates_are_unit ? sizeof(float) : sizeof(double)" in source
    assert "edge_gradients += coordinate_offset*harmonic_count" in source
    assert "edge_gradients[axis*harmonic_count+lm]" in source
    assert "gradients[(3*edge+axis)*harmonic_count+lm]" not in source
    assert "long long num_edges;" in source
    assert "for (long long edge =" in source
    assert "static_cast<long long>(edge)*harmonic_count" in source
    assert "bool edge_is_active(" in source
    assert source.count("edge_is_active(") == 4
    assert "if (!edge_is_active(args.cutoff, args.radius[edge])) continue;" in source
    assert 109_020_492 * 3 * 16 > 2**31 - 1


def test_r0_device_module_accepts_runtime_channels_and_rejects_topology_changes():
    contract = _without_fingerprints(_contract("standard_r0_contract.json"))
    baseline = jit_r0_device_metadata(contract, precision="float32", target="sm_120")

    resized = deepcopy(contract)
    resized["channels"] = 48
    for path in resized["paths"]:
        path["path_shape"][0] = 48
    resized_metadata = jit_r0_device_metadata(
        resized, precision="float32", target="sm_120"
    )
    assert (
        resized_metadata["structure_fingerprint"] == baseline["structure_fingerprint"]
    )
    assert resized_metadata["artifact_id"] == baseline["artifact_id"]

    assert (
        jit_r0_device_metadata(contract, precision="float32", target="sm_90")[
            "artifact_id"
        ]
        != baseline["artifact_id"]
    )

    perturbed = deepcopy(contract)
    perturbed["sparse_coupling"]["terms"][1]["coefficient"] = 0.5
    with pytest.raises(ValueError):
        jit_r0_device_metadata(perturbed, precision="float32", target="sm_120")
