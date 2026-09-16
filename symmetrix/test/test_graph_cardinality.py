import json
from pathlib import Path

import numpy as np
import pytest
from compact_r1_model import foundation_r1_test_model
from symmetrix import symmetrix as native_symmetrix


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"
CONTRACT_ROOT = Path(__file__).resolve().parent / "data" / "execution_contracts"


@pytest.fixture(scope="module")
def compact_model_path(tmp_path_factory):
    r1_contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_mace_omat0_small_contract.json").read_text()
    )
    m0_contract = json.loads((CONTRACT_ROOT / "standard_m0_contract.json").read_text())
    m0_contract["output_l_max"] = 0
    m0_contract["output_components"] = 1
    m0_contract["monomial_groups"] = m0_contract["monomial_groups"][:1]
    m0_contract["structure_fingerprint"] = (
        "sha256:3b4ab2d979488798a05c07b5d54c4c96c1026da3ee1eb93069fd2bacec76ac90"
    )
    m0_contract["semantic_fingerprint"] = (
        "sha256:9fe598ceffb71e61d162c0afbdbe5727a0da7997dbca3a531a46b63603d7af85"
    )
    m0_contract["generation_fingerprint"] = (
        "sha256:f974e4af32dfb328e06fb6655b965c4cc5c0cad50a2013223630f9960f59d398"
    )
    r0_contract = json.loads((CONTRACT_ROOT / "standard_r0_contract.json").read_text())
    model = foundation_r1_test_model(
        r1_contract,
        atomic_numbers=(7, 13),
        m0_contract=m0_contract,
        r0_contract=r0_contract,
    )
    model["A0_scaled"] = True
    model["compact_radial"]["networks"]["A0"] = {
        "shape": [r1_contract["radial_embedding"], 1],
        "weights": [[0.01] * r1_contract["radial_embedding"]],
        "activation": "silu",
        "activation_scale": 1.0,
        "postprocess": "tanh-square",
    }
    output = tmp_path_factory.mktemp("graph-cardinality") / "compact.json"
    output.write_text(json.dumps(model, separators=(",", ":")))
    return output


@pytest.mark.parametrize("evaluator_name", ["MACEKokkosFloat", "MACEKokkos"])
def test_native_graph_cardinality_guard_covers_counts_and_model_strides(
    compact_model_path, evaluator_name
):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = getattr(native_symmetrix, evaluator_name)(str(compact_model_path))
    int_max = 2**31 - 1
    channels = json.loads(compact_model_path.read_text())["num_channels"]

    evaluator._validate_graph_cardinality(1_000_188, 1_000_188, 109_020_492)
    maximum_channel_receivers = int_max // channels
    evaluator._validate_graph_cardinality(
        maximum_channel_receivers, maximum_channel_receivers, int_max
    )
    # Flattened edge-channel and harmonic extents may exceed INT32_MAX; only
    # their address arithmetic and byte extent need to remain representable.
    evaluator._validate_graph_cardinality(1, 1, 45_266_828)

    with pytest.raises(ValueError, match="atom count.*32-bit CSR limit"):
        evaluator._validate_graph_cardinality(int_max, int_max, 0)
    with pytest.raises(ValueError, match="directed-edge count.*32-bit topology"):
        evaluator._validate_graph_cardinality(1, 1, int_max + 1)
    with pytest.raises(ValueError, match="receiver-channel launch extent"):
        receiver_count = maximum_channel_receivers + 1
        evaluator._validate_graph_cardinality(receiver_count, receiver_count, 0)
    with pytest.raises(ValueError, match="feature-node-channel launch extent"):
        evaluator._validate_graph_cardinality(1, maximum_channel_receivers + 1, 0)
    with pytest.raises(ValueError, match="smaller than its receiver count"):
        evaluator._validate_graph_cardinality(2, 1, 0)


def test_prepared_factorized_graph_passes_cardinality_admission(compact_model_path):
    evaluator = native_symmetrix.MACEKokkosFloat(str(compact_model_path))
    evaluator.set_streamed_edges("direct")
    token = evaluator._prepare_factorized_graph(
        2,
        np.array([0, 0], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
    )

    assert token > 0
    assert evaluator.factorized_graph_generation == token
    assert evaluator.factorized_schedule_entries == 2


def _function_source(source, start, end):
    return source[source.index(start) : source.index(end)]


def test_graph_entry_points_validate_before_allocation_or_narrowing():
    lifecycle = (SOURCE_ROOT / "mace_kokkos_factorized_lifecycle.cpp").read_text()
    generic_prepare = _function_source(
        lifecycle,
        "std::uint64_t MACEKokkos<Precision>::prepare_all_interactions_graph(",
        "void MACEKokkos<Precision>::prepare_all_interactions_geometry(",
    )
    direct_prepare = _function_source(
        lifecycle,
        "std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph(\n"
        "    const int num_receivers,",
        "std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph_device(",
    )
    schedule = _function_source(
        lifecycle,
        "bool MACEKokkos<Precision>::prepare_factorized_schedule_host(\n"
        "    const int num_receivers,",
        "void MACEKokkos<Precision>::prepare_factorized_model(",
    )

    assert generic_prepare.index("validate_graph_cardinality(") < generic_prepare.index(
        "std::vector<int>("
    )
    assert direct_prepare.index("validate_graph_cardinality(") < direct_prepare.index(
        "std::vector<int>("
    )
    assert schedule.index("validate_graph_cardinality(") < schedule.index(
        "static_cast<int>(schedule_neigh_indices.size())"
    )

    evaluate = (SOURCE_ROOT / "mace_kokkos_evaluate.cpp").read_text()
    ordinary = _function_source(
        evaluate,
        "void MACEKokkos<Precision>::compute_node_energies_forces(",
        "void MACEKokkos<Precision>::compute_node_energies_forces_field(",
    )
    field = _function_source(
        evaluate,
        "void MACEKokkos<Precision>::compute_node_energies_forces_field(",
        "void MACEKokkos<Precision>::begin_factorized_distributed_evaluation(",
    )
    distributed = _function_source(
        evaluate,
        "void MACEKokkos<Precision>::begin_factorized_distributed_evaluation(",
        "void MACEKokkos<Precision>::continue_factorized_distributed_evaluation(",
    )
    for entry_point in (ordinary, field, distributed):
        guard = entry_point.index("validate_graph_cardinality(")
        allocation_markers = (
            "Kokkos::realloc(",
            "ensure_execution_result_capacity(",
        )
        allocation = min(
            entry_point.index(marker)
            for marker in allocation_markers
            if marker in entry_point
        )
        assert guard < allocation
        assert guard < entry_point.index("begin_factorized_production_evaluation()")
