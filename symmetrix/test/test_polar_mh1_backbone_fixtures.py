"""Frozen inputs for the generalized polar MH-1 backbone work."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest


DATA_ROOT = Path(__file__).resolve().parent / "data" / "polar"
CHECKPOINT_FACTS_PATH = DATA_ROOT / "checkpoint_facts.json"
BUILDER_PATH = Path(__file__).resolve().parent / "polar_mh1_backbone_fixture.py"


@lru_cache(maxsize=1)
def _build_topology_template():
    assert BUILDER_PATH.is_file(), "the production-shaped 512/128 builder is missing"
    try:
        from polar_mh1_backbone_fixture import (
            RELEASED_M_CHECKPOINT_SHA256,
            build_polar_mh1_backbone_definition,
        )
    except ImportError as error:
        pytest.skip(f"polar MH-1 fixture requires torch/e3nn/mace: {error}")

    try:
        return build_polar_mh1_backbone_definition(), RELEASED_M_CHECKPOINT_SHA256
    except ImportError as error:
        pytest.skip(f"polar MH-1 fixture requires torch/e3nn/mace: {error}")


def _build_topology():
    topology, checkpoint_sha256 = _build_topology_template()
    return copy.deepcopy(topology), checkpoint_sha256


@lru_cache(maxsize=1)
def _build_scalar_final_topology_template():
    try:
        from polar_mh1_backbone_fixture import (
            build_scalar_final_mixed_edge_mh1_definition,
        )
    except ImportError as error:
        pytest.skip(f"scalar-final MH-1 fixture requires torch/e3nn/mace: {error}")

    try:
        return build_scalar_final_mixed_edge_mh1_definition()
    except ImportError as error:
        pytest.skip(f"scalar-final MH-1 fixture requires torch/e3nn/mace: {error}")


def _build_scalar_final_topology():
    return copy.deepcopy(_build_scalar_final_topology_template())


@lru_cache(maxsize=None)
def _build_uniform_scalar_final_topology_template(edge_channels):
    try:
        from polar_mh1_backbone_fixture import (
            build_scalar_final_uniform_128_edge_mh1_definition,
            build_scalar_final_uniform_512_edge_mh1_definition,
        )
    except ImportError as error:
        pytest.skip(f"uniform-edge MH-1 fixture requires torch/e3nn/mace: {error}")

    builder = {
        128: build_scalar_final_uniform_128_edge_mh1_definition,
        512: build_scalar_final_uniform_512_edge_mh1_definition,
    }[edge_channels]
    try:
        return builder()
    except ImportError as error:
        pytest.skip(f"uniform-edge MH-1 fixture requires torch/e3nn/mace: {error}")


def _build_uniform_scalar_final_topology(edge_channels):
    return copy.deepcopy(_build_uniform_scalar_final_topology_template(edge_channels))


@lru_cache(maxsize=1)
def _build_one_layer_topology_template():
    try:
        from polar_mh1_backbone_fixture import (
            build_one_layer_scalar_final_mh1_definition,
        )
    except ImportError as error:
        pytest.skip(f"one-layer MH-1 fixture requires torch/e3nn/mace: {error}")

    try:
        return build_one_layer_scalar_final_mh1_definition()
    except ImportError as error:
        pytest.skip(f"one-layer MH-1 fixture requires torch/e3nn/mace: {error}")


def _build_one_layer_topology():
    return copy.deepcopy(_build_one_layer_topology_template())


def _scalar_final_mixed_edge_topology():
    """Return a mutable, contract-coherent scalar-final 512/128 definition."""

    return copy.deepcopy(_build_scalar_final_topology())


def _write_topology(path, topology):
    serialized = copy.deepcopy(topology)
    if serialized.get("selected_head") is None:
        serialized["selected_head"] = ""
    if serialized.get("available_heads") is None:
        serialized["available_heads"] = []
    path.write_text(json.dumps(serialized))
    return path


def _program_nodes(program):
    return {node["id"]: node for node in program["operators"]}


def _native_evaluator(native, path, *, polar=False):
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    evaluator = native.MACEPolarKokkos if polar else native.MACENonlinearKokkos
    return evaluator(str(path))


def test_compact_fixture_pins_independent_layer_channels_and_keep_last_output():
    """Catch a real serialized definition collapsing both layers to one width."""

    topology, _ = _build_topology()
    assert topology["symmetrix_format_version"] == 3
    assert topology["model_type"] == "MACE_Polar"
    assert topology["num_interactions"] == 2

    interactions = topology["interactions"]
    assert [item["edge_irreps"] for item in interactions] == [
        "512x0e",
        "128x0e+128x1o",
    ]
    assert [
        [instruction["path_shape"] for instruction in item["conv_tp"]["instructions"]]
        for item in interactions
    ] == [
        [[512, 1]] * 4,
        [[128, 1]] * 10,
    ]
    assert all(
        {
            instruction["connection_mode"]
            for instruction in item["conv_tp"]["instructions"]
        }
        == {"uvu"}
        for item in interactions
    )

    assert topology["polar"]["keep_last_layer_irreps"] is True
    assert topology["products"][1]["symmetric_contractions"]["irreps_out"] == (
        "512x0e+512x1o"
    )
    assert topology["products"][1]["linear"]["irreps_out"] == "512x0e+512x1o"
    assert topology["readouts"][1]["linear_1"]["irreps_in"] == "512x0e+512x1o"
    assert topology["readouts"][1]["linear_2"]["irreps_out"] == "1x0e"
    assert "radial_embedding" in topology


def test_compact_fixture_is_tied_to_the_released_m_checkpoint():
    """Catch accidental replacement of the integration checkpoint identity."""

    _, released_m_sha256 = _build_topology()
    facts = json.loads(CHECKPOINT_FACTS_PATH.read_text())
    released = facts["models"]["M"]["file"]
    assert released_m_sha256 == released["sha256"]


def test_compact_fixture_passes_the_real_mh1_contract_parser():
    """Catch a test-only shape that production contract code cannot consume."""

    topology, _ = _build_topology()
    from symmetrix.mace_nonlinear_execution_contract import (
        normalize_execution_mh1_v4_contract,
    )
    from symmetrix.mace_polar_execution_contract import (
        make_execution_polar_mh1_v4_contract,
        validate_execution_polar_mh1_v4_contract_for_model,
    )

    contract = make_execution_polar_mh1_v4_contract(topology)
    assert [
        interaction["dimensions"]["channels"]
        for interaction in contract["interactions"]
    ] == [512, 128]
    assert contract["node_program"]["layers"][1]["product"]["irreps"]["output"] == (
        "512x0e+512x1o"
    )
    assert normalize_execution_mh1_v4_contract(contract) == contract
    assert (
        validate_execution_polar_mh1_v4_contract_for_model(topology, contract)
        == contract
    )


def test_generated_metadata_selects_the_dedicated_wide_final_schedule():
    """Catch the polar tail silently using the ordinary scalar-final schedule."""

    from symmetrix.mace_nonlinear_execution_contract import (
        make_execution_mh1_v4_contract,
    )
    from symmetrix.mace_polar_execution_contract import (
        make_execution_polar_mh1_v4_contract,
    )
    from symmetrix.mace_nonlinear_jit_codegen import (
        execution_mh1_node_program_metadata,
    )

    polar_contract = make_execution_polar_mh1_v4_contract(_build_topology()[0])
    scalar_contract = make_execution_mh1_v4_contract(_build_scalar_final_topology())

    for backend in ("host", "cuda"):
        polar = execution_mh1_node_program_metadata(polar_contract, backend=backend)[
            "layers"
        ]
        scalar = execution_mh1_node_program_metadata(scalar_contract, backend=backend)[
            "layers"
        ]

        assert polar[0]["final_output_specialization"] == "intermediate_v1"
        assert not polar[0]["uses_wide_final_schedule"]
        assert polar[1]["final_output_specialization"] == ("keep_last_512x0e_512x1o_v1")
        assert polar[1]["uses_wide_final_schedule"]
        assert polar[1]["final_output_fallback_reason"] == ""
        assert scalar[1]["final_output_specialization"] == "scalar_final_v1"
        assert not scalar[1]["uses_wide_final_schedule"]


def test_cuda_wide_final_reverse_keeps_product_seed_outside_the_node_abi():
    """Native composition owns retained outputs and external product seeds."""

    from symmetrix.mace_polar_execution_contract import (
        make_execution_polar_mh1_v4_contract,
    )
    from symmetrix.mace_nonlinear_jit_codegen import (
        execution_mh1_cuda_module_v4_launch_plan,
        render_execution_mh1_cuda_module_v4,
    )

    contract = make_execution_polar_mh1_v4_contract(_build_topology()[0])
    module = render_execution_mh1_cuda_module_v4(contract, 120)
    plan = execution_mh1_cuda_module_v4_launch_plan(contract, 120)

    assert "external_product_seed" not in module
    assert plan["split_phase_composition"] is True

    native = (
        Path(__file__).resolve().parents[2]
        / "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    assert "retain_layer_output(" in native
    assert "add_layer_product_seed(" in native


def test_polar_host_artifact_publishes_the_wide_final_specialization(tmp_path):
    """The standalone adapter serves polar and publishes its selected tail."""

    from symmetrix.mace_nonlinear_jit_artifact import prepare_mh1_host_artifact

    path = _write_topology(
        tmp_path / "polar-host-artifact.json", copy.deepcopy(_build_topology()[0])
    )
    cache_root = tmp_path / "jit-cache"
    cache_root.mkdir(mode=0o700)
    result = prepare_mh1_host_artifact(path, precision="float64", cache_root=cache_root)

    assert result.model_type == "MACE_Polar"
    assert result.final_output_specializations == (
        "intermediate_v1",
        "keep_last_512x0e_512x1o_v1",
    )
    manifest = json.loads(result.manifest_path.read_text())
    layers = manifest["key_inputs"]["build"]["contract"]["layers"]
    assert tuple(layer["final_output_specialization"] for layer in layers) == (
        "intermediate_v1",
        "keep_last_512x0e_512x1o_v1",
    )


def test_native_mh1_admission_accepts_polar_keep_last_output():
    """Catch the explicit polar keep-last policy falling back to scalar-final."""

    topology, _ = _build_topology()
    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert descriptor["compatible"], descriptor["rejection_reason"]
    assert descriptor["first_layer_edge_channels"] == 512
    assert descriptor["layers"][0] == {
        "interaction_index": 0,
        "uvu_channels": 512,
        "edge_blocks": [
            {"multiplicity": 512, "l": 0, "parity": 1},
        ],
        "product_output_blocks": [
            {"multiplicity": 512, "l": 0, "parity": 1},
            {"multiplicity": 512, "l": 1, "parity": -1},
        ],
        "final_output_policy": "not_final",
    }
    assert descriptor["layers"][1] == {
        "interaction_index": 1,
        "uvu_channels": 128,
        "edge_blocks": [
            {"multiplicity": 128, "l": 0, "parity": 1},
            {"multiplicity": 128, "l": 1, "parity": -1},
        ],
        "product_output_blocks": [
            {"multiplicity": 512, "l": 0, "parity": 1},
            {"multiplicity": 512, "l": 1, "parity": -1},
        ],
        "final_output_policy": "keep_last",
    }
    assert len(descriptor["layers"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "false_flag",
        "non_boolean_flag",
        "missing_polar_contract",
        "false_contract_flag",
    ],
)
def test_native_mh1_keep_last_requires_explicit_polar_policy(mutation):
    """Catch ordinary or ambiguously declared models entering keep-last admission."""

    topology = copy.deepcopy(_build_topology()[0])
    if mutation == "false_flag":
        topology["polar"]["keep_last_layer_irreps"] = False
    elif mutation == "non_boolean_flag":
        topology["polar"]["keep_last_layer_irreps"] = 1
    elif mutation == "missing_polar_contract":
        topology["execution_contracts"].pop("POLAR_BACKBONE")
    else:
        topology["execution_contracts"]["POLAR_BACKBONE"]["keep_last_layer_irreps"] = (
            False
        )

    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert not descriptor["compatible"]
    assert descriptor["rejection_reason"].startswith(
        "Polar MH-1 identity validation failed:"
    )


def test_native_mh1_keep_last_rejects_ordinary_family_with_polar_flags():
    """Catch an ordinary nonlinear-family stamp impersonating polar admission."""

    topology = copy.deepcopy(_build_topology()[0])
    topology["model_type"] = "MACE_Nonlinear"
    topology["scale_shift"]["scale"]["shape"] = []
    topology["scale_shift"]["shift"]["shape"] = []
    topology["atomic_energies"]["shape"] = [
        1,
        len(topology["atomic_energies"]["values"]),
    ]

    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert not descriptor["compatible"]
    assert descriptor["layers"][1]["final_output_policy"] == "scalar"
    assert descriptor["rejection_reason"] == (
        "MH-1 layer 1 hidden representation is inconsistent."
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tag", "symmetrix.execution.polar_backbone/1"),
        ("schema", "symmetrix.execution.polar_backbone.stale"),
        ("version", 1),
    ],
)
def test_native_mh1_keep_last_rejects_wrong_polar_contract_identity(field, value):
    """Catch a stale contract identity activating keep-last admission."""

    topology = copy.deepcopy(_build_topology()[0])
    topology["execution_contracts"]["POLAR_BACKBONE"][field] = value

    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert not descriptor["compatible"]
    assert descriptor["rejection_reason"].startswith(
        "Polar MH-1 identity validation failed:"
    )


def test_native_mh1_keep_last_rejects_model_inconsistent_polar_contract():
    """Catch a current-looking contract copied from another backbone topology."""

    topology = copy.deepcopy(_build_topology()[0])
    topology["execution_contracts"]["POLAR_BACKBONE"]["interaction_count"] = 3

    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert not descriptor["compatible"]
    assert descriptor["rejection_reason"].startswith(
        "Polar MH-1 identity validation failed:"
    )
    assert "interaction count" in descriptor["rejection_reason"]


@pytest.mark.parametrize(
    ("mutation", "invariant"),
    [
        ("hidden_higher_l", "hidden representation"),
        ("skip_scalar_only", "skip representation"),
        ("symmetric_higher_l", "product representation"),
        ("retained_node_scalar_only", "product connectivity"),
        ("retained_target_scalar_only", "product connectivity"),
        ("linear_input_scalar_only", "product connectivity"),
        ("linear_output_scalar_only", "product connectivity"),
        ("readout_input_scalar_only", "readout connectivity"),
        ("readout_vector_path", "readout connectivity"),
        ("readout_vector_output", "readout connectivity"),
    ],
)
def test_native_mh1_keep_last_rejects_inconsistent_final_topology(mutation, invariant):
    """Catch keep-last weakening final layout or scalar-energy-readout checks."""

    topology = copy.deepcopy(_build_topology()[0])
    interaction = topology["interactions"][1]
    product = topology["products"][1]
    readout = topology["readouts"][1]
    if mutation == "hidden_higher_l":
        interaction["hidden_irreps"] = "512x0e+512x2e"
    elif mutation == "skip_scalar_only":
        interaction["skip_tp"]["irreps_out"] = "512x0e"
    elif mutation == "symmetric_higher_l":
        product["symmetric_contractions"]["irreps_out"] = "512x0e+512x2e"
    elif mutation == "retained_node_scalar_only":
        product["node_feats_irreps"] = "512x0e"
    elif mutation == "retained_target_scalar_only":
        product["target_irreps"] = "512x0e"
    elif mutation == "linear_input_scalar_only":
        product["linear"]["irreps_in"] = "512x0e"
    elif mutation == "linear_output_scalar_only":
        product["linear"]["irreps_out"] = "512x0e"
    elif mutation == "readout_input_scalar_only":
        readout["linear_1"]["irreps_in"] = "512x0e"
    elif mutation == "readout_vector_path":
        readout["linear_1"]["instructions"].append(
            {
                "i_in": 1,
                "i_out": 0,
                "path_weight": 1.0,
                "path_shape": [512, 2],
            }
        )
    else:
        readout["linear_2"]["irreps_out"] = "1x1o"

    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert not descriptor["compatible"]
    assert descriptor["rejection_reason"] == (
        f"MH-1 layer 1 {invariant} is inconsistent."
    )


def test_native_mh1_admission_accepts_independent_layer_edge_channels():
    """Catch admission deriving layer 1's 128 channels from layer 0's 512."""

    topology = _scalar_final_mixed_edge_topology()
    from symmetrix import symmetrix as native
    from symmetrix.mace_nonlinear_execution_contract import (
        make_execution_mh1_v4_contract,
        validate_execution_mh1_v4_contract_for_model,
    )

    contract = make_execution_mh1_v4_contract(topology)
    assert validate_execution_mh1_v4_contract_for_model(topology, contract) == contract
    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert descriptor["compatible"], descriptor["rejection_reason"]
    assert [layer["uvu_channels"] for layer in descriptor["layers"]] == [512, 128]


def test_native_program_uses_each_layers_edge_channels_and_actual_extents(tmp_path):
    """Catch later operators and harmonics inheriting layer zero's UVU width."""

    topology = copy.deepcopy(_build_topology()[0])
    from symmetrix import symmetrix as native

    evaluator = _native_evaluator(
        native, _write_topology(tmp_path / "mixed-edge.json", topology), polar=True
    )
    nodes = _program_nodes(evaluator.family_program)

    expected_tensor_products = (
        {
            "edge_channels": 512,
            "input_1_dimension": 512,
            "input_2_dimension": 16,
            "output_dimension": 8192,
        },
        {
            "edge_channels": 128,
            "input_1_dimension": 512,
            "input_2_dimension": 16,
            "output_dimension": 5120,
        },
    )
    for layer, expected in enumerate(expected_tensor_products):
        tensor_product = nodes[f"mh1.interaction{layer}.m0"]
        assert {
            name: tensor_product["parameters"][name] for name in expected
        } == expected
        assert (
            nodes[f"mh1.interaction{layer}.node_program"]["parameters"]["edge_channels"]
            == expected["edge_channels"]
        )
        assert nodes[f"mh1.interaction{layer}.harmonics"]["outputs"][0]["shape"] == [
            0,
            16,
        ]
    final_parameters = nodes["mh1.interaction1.node_program"]["parameters"]
    assert {
        name: final_parameters[name]
        for name in (
            "product_output_block_count",
            "product_output_block_0_multiplicity",
            "product_output_block_0_l",
            "product_output_block_0_parity",
            "product_output_block_1_multiplicity",
            "product_output_block_1_l",
            "product_output_block_1_parity",
        )
    } == {
        "product_output_block_count": 2,
        "product_output_block_0_multiplicity": 512,
        "product_output_block_0_l": 0,
        "product_output_block_0_parity": 1,
        "product_output_block_1_multiplicity": 512,
        "product_output_block_1_l": 1,
        "product_output_block_1_parity": -1,
    }


def test_native_program_identity_includes_ordered_layer_edge_topology(tmp_path):
    """Catch a later-layer width change reusing the 512/128 program identity."""

    from symmetrix import symmetrix as native

    definitions = {
        "512-128": _scalar_final_mixed_edge_topology(),
        "512-512": copy.deepcopy(_build_uniform_scalar_final_topology(512)),
        "128-128": copy.deepcopy(_build_uniform_scalar_final_topology(128)),
    }
    programs = {
        name: _native_evaluator(
            native, _write_topology(tmp_path / f"{name}.json", definition)
        ).family_program
        for name, definition in definitions.items()
    }

    assert len({program["fingerprint"] for program in programs.values()}) == 3
    for name, expected in {
        "512-128": [512, 128],
        "512-512": [512, 512],
        "128-128": [128, 128],
    }.items():
        nodes = _program_nodes(programs[name])
        assert [
            nodes[f"mh1.interaction{layer}.m0"]["parameters"]["edge_channels"]
            for layer in range(2)
        ] == expected


def test_generated_identity_includes_ordered_layer_edge_topology():
    """Catch generated artifacts aliasing distinct ordered UVU channel layouts."""

    from symmetrix.mace_nonlinear_execution_contract import (
        make_execution_mh1_v4_contract,
    )
    from symmetrix.mace_nonlinear_jit_codegen import (
        execution_mh1_node_program_metadata,
    )

    definitions = {
        "512-128": _scalar_final_mixed_edge_topology(),
        "512-512": copy.deepcopy(_build_uniform_scalar_final_topology(512)),
        "128-128": copy.deepcopy(_build_uniform_scalar_final_topology(128)),
    }
    contracts = {
        name: make_execution_mh1_v4_contract(definition)
        for name, definition in definitions.items()
    }

    assert {
        name: [
            interaction["dimensions"]["channels"]
            for interaction in contract["interactions"]
        ]
        for name, contract in contracts.items()
    } == {
        "512-128": [512, 128],
        "512-512": [512, 512],
        "128-128": [128, 128],
    }
    for identity in ("generation_fingerprint", "structure_fingerprint"):
        assert len({contract[identity] for contract in contracts.values()}) == 3
    metadata = {
        name: execution_mh1_node_program_metadata(contract, backend="host")
        for name, contract in contracts.items()
    }
    assert len({item["artifact_id"] for item in metadata.values()}) == 3


def test_native_diagnostics_report_portable_executor_and_layer_capabilities(
    tmp_path,
):
    """Catch a global fast-path flag hiding which layer lost a primitive."""

    topology = copy.deepcopy(_build_topology()[0])
    from symmetrix import symmetrix as native

    evaluator = _native_evaluator(
        native, _write_topology(tmp_path / "diagnostics.json", topology), polar=True
    )
    diagnostics = evaluator.mh1_portable_diagnostics

    assert diagnostics["portable_executor"] == evaluator.mh1_edge_executor
    assert diagnostics["layers"] == [
        {
            "interaction_index": 0,
            "uvu_channels": 512,
            "architecture_admitted": True,
            "tensor_product_ready": True,
            "compiled_product_ready": True,
            "pair_conditioning_ready": True,
            "portable_ready": True,
            "reason": "portable MH-1 layer admitted",
        },
        {
            "interaction_index": 1,
            "uvu_channels": 128,
            "architecture_admitted": True,
            "tensor_product_ready": True,
            "compiled_product_ready": True,
            "pair_conditioning_ready": True,
            "portable_ready": True,
            "reason": "portable MH-1 layer admitted",
        },
    ]


def test_polar_reference_selects_artifact_free_wide_final_layer_schedule(tmp_path):
    """Catch the admitted polar topology retaining the generic node tail."""

    topology = copy.deepcopy(_build_topology()[0])
    from symmetrix import symmetrix as native

    evaluator = _native_evaluator(
        native, _write_topology(tmp_path / "portable-final.json", topology), polar=True
    )

    assert evaluator.uses_mh1_fast_path
    assert evaluator.mh1_uses_compiled_products
    assert evaluator.mh1_uses_pair_conditioning
    assert evaluator.mh1_uses_external_uvu_tensors
    assert evaluator.mh1_portable_final_layer_ready
    assert evaluator.mh1_portable_final_layer_enabled
    assert evaluator.mh1_portable_final_layer_kind == "block_scalar_vector_readout_v1"
    assert evaluator.mh1_portable_final_layer_evaluation_count == 0
    assert not evaluator.jit_mh1_host_plugin_ready
    assert not evaluator.jit_mh1_cuda_plugin_ready
    assert evaluator.execution_mh1_generated_forward_launch_count == 0


def test_scalar_final_mh1_keeps_the_existing_generic_readout(tmp_path):
    """Catch the wide-tail schedule leaking into ordinary scalar-final MH-1."""

    topology = _scalar_final_mixed_edge_topology()
    from symmetrix import symmetrix as native

    evaluator = _native_evaluator(
        native, _write_topology(tmp_path / "scalar-final.json", topology)
    )

    assert evaluator.uses_mh1_fast_path
    assert not evaluator.mh1_portable_final_layer_ready
    assert not evaluator.mh1_portable_final_layer_enabled
    assert evaluator.mh1_portable_final_layer_kind == "generic"
    with pytest.raises(ValueError, match="portable MH-1 final-layer schedule"):
        evaluator._set_mh1_portable_final_layer(True)


@pytest.mark.parametrize(
    ("dtype", "value_atol", "force_atol"),
    [("float64", 2e-11, 2e-10), ("float32", 2e-5, 8e-5)],
)
def test_polar_portable_wide_final_layer_matches_generic_retention_and_seed(
    tmp_path, dtype, value_atol, force_atol
):
    """Catch the native wide-tail schedule changing retained or seeded results."""

    from ase import Atoms
    from symmetrix import Symmetrix

    path = _write_topology(
        tmp_path / "portable-final-parity.json", copy.deepcopy(_build_topology()[0])
    )
    calculator = Symmetrix(path, dtype=dtype, streamed_edges="reference")
    evaluator = calculator.evaluator
    assert calculator.streamed_edges == "reference"
    assert evaluator.streamed_edges_mode == "direct"
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.1, 0.2, 0.1]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    inputs = calculator._mace_inputs(atoms)[:-1]
    evaluator.set_retain_layer_features(True)

    def result():
        evaluator.compute_node_energies_forces(*inputs)
        return (
            np.asarray(evaluator.node_energies).copy(),
            np.asarray(evaluator.node_forces).copy(),
            [np.asarray(layer).copy() for layer in evaluator.layer_features()],
        )

    evaluator._set_mh1_portable_final_layer(False)
    generic = result()
    generator = np.random.default_rng(6106)
    seeds = [
        np.ascontiguousarray(generator.normal(size=layer.shape)) for layer in generic[2]
    ]
    evaluator.set_layer_product_seed(seeds)
    generic_seeded = result()

    evaluator._set_mh1_portable_final_layer(True)
    portable_seeded = result()
    evaluator.clear_layer_product_seed()
    portable = result()

    for actual, expected in zip(portable[:2], generic[:2]):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=value_atol)
    for actual, expected in zip(portable[2], generic[2]):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=value_atol)
    np.testing.assert_allclose(
        portable_seeded[0], generic_seeded[0], rtol=0.0, atol=value_atol
    )
    np.testing.assert_allclose(
        portable_seeded[1], generic_seeded[1], rtol=0.0, atol=force_atol
    )
    for actual, expected in zip(portable_seeded[2], generic_seeded[2]):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=value_atol)
    assert evaluator.mh1_portable_final_layer_evaluation_count == 2
    assert evaluator.execution_mh1_generated_forward_launch_count == 0


@pytest.mark.parametrize(
    ("dtype", "value_atol"),
    [("float64", 2e-4), ("float32", 1.5e-3)],
)
def test_polar_generated_backbone_matches_portable_retention_and_seed(
    tmp_path, monkeypatch, dtype, value_atol
):
    """Qualify the generated 512/128 backbone and its wide reverse seeds.

    The production-shaped fixture deliberately contains only the backbone, so
    this drives its native evaluator directly.  Explicit full-width per-layer
    seeds reproduce the polar chain's reverse interface, including the
    specialized 512x0e+512x1o tail.  Comparing retained tensors independently
    catches a forward basis/layout error that could otherwise be hidden by a
    readout.
    """

    from ase import Atoms
    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native

    path = _write_topology(
        tmp_path / f"polar-generated-parity-{dtype}.json",
        copy.deepcopy(_build_topology()[0]),
    )
    cache_root = tmp_path / f"jit-cache-{dtype}"
    cache_root.mkdir(mode=0o700)
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(cache_root))
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.1, 0.2, 0.1]],
        cell=[[8.0, 0.1, 0.0], [0.0, 7.5, 0.2], [0.1, 0.0, 8.5]],
        pbc=True,
    )
    reference = Symmetrix(path, dtype=dtype, streamed_edges="reference")
    reference.evaluator.set_retain_layer_features(True)
    direct = Symmetrix(path, dtype=dtype, streamed_edges="direct")
    direct.evaluator.set_retain_layer_features(True)

    def evaluate(calculator):
        calculator.calculate(atoms.copy(), properties=["energy", "forces"])
        evaluator = calculator.evaluator
        return (
            np.asarray(evaluator.node_energies).copy(),
            np.asarray(evaluator.node_forces).copy(),
            [np.asarray(layer).copy() for layer in evaluator.layer_features()],
        )

    reference_unseeded = evaluate(reference)
    direct_unseeded = evaluate(direct)
    assert len(direct_unseeded[2]) == len(reference_unseeded[2]) == 2
    assert direct_unseeded[2][-1].shape[1] == 2048
    force_atol = 1.6e-3 if dtype == "float32" else value_atol
    np.testing.assert_allclose(
        direct_unseeded[0], reference_unseeded[0], rtol=0.0, atol=value_atol
    )
    np.testing.assert_allclose(
        direct_unseeded[1], reference_unseeded[1], rtol=0.0, atol=force_atol
    )
    for actual_layer, expected_layer in zip(
        direct_unseeded[2], reference_unseeded[2], strict=True
    ):
        np.testing.assert_allclose(
            actual_layer, expected_layer, rtol=0.0, atol=value_atol
        )

    generator = np.random.default_rng(20261017)
    seeds = [
        np.ascontiguousarray(generator.normal(size=layer.shape))
        for layer in reference_unseeded[2]
    ]
    reference.evaluator.set_layer_product_seed(seeds)
    direct.evaluator.set_layer_product_seed(seeds)
    reference_seeded = evaluate(reference)
    direct_seeded = evaluate(direct)
    np.testing.assert_allclose(
        direct_seeded[0], reference_seeded[0], rtol=0.0, atol=value_atol
    )
    np.testing.assert_allclose(
        direct_seeded[1], reference_seeded[1], rtol=0.0, atol=force_atol
    )

    evaluator = direct.evaluator
    assert "shared_products" in evaluator.execution_mh1_execution_backend
    assert all(
        evaluator._execution_mh1_product_module_forward_launch_count(layer) > 0
        for layer in range(2)
    )
    assert all(
        evaluator._execution_mh1_product_module_reverse_launch_count(layer) > 0
        for layer in range(2)
    )
    assert evaluator.factorized_fallback_evaluation_count == 0
    if native._kokkos_default_execution_space() == "Cuda":
        assert evaluator.jit_mh1_cuda_plugin_v5_ready
    else:
        assert evaluator.jit_mh1_host_plugin_v7_ready

    assert reference.jit_status == "not_applicable"
    assert not reference.evaluator.jit_mh1_host_plugin_ready
    assert not reference.evaluator.jit_mh1_cuda_plugin_ready
    assert reference.evaluator.execution_mh1_generated_forward_launch_count == 0
    assert reference.evaluator.execution_mh1_generated_source_reverse_launch_count == 0


def test_retained_feature_layout_describes_portable_and_generated_bases(tmp_path):
    """Catch forward and reverse taps rebuilding different basis rules."""

    from symmetrix import symmetrix as native

    polar = _native_evaluator(
        native,
        _write_topology(
            tmp_path / "retained-layout-polar.json",
            copy.deepcopy(_build_topology()[0]),
        ),
        polar=True,
    )
    assert polar.mh1_layer_count == 2
    for layer in range(polar.mh1_layer_count):
        layout = polar.mh1_layer_retained_feature_layout(layer)
        assert layout == {
            "width": 2048,
            "channels": 512,
            "scalar_offset": 0,
            "scalar_multiplicity": 512,
            "vector_offset": 512,
            "vector_multiplicity": 512,
            "vector_components": 3,
            "retained_from_native_components": [1, 2, 0],
            "retained_from_native_signs": [-1, -1, -1],
            "portable_storage": "mul_ir",
            "generated_storage": "ir_mul",
        }

    scalar = _native_evaluator(
        native,
        _write_topology(
            tmp_path / "retained-layout-scalar.json",
            _scalar_final_mixed_edge_topology(),
        ),
    )
    final_layout = scalar.mh1_layer_retained_feature_layout(1)
    assert final_layout["width"] == 512
    assert final_layout["scalar_offset"] == 0
    assert final_layout["scalar_multiplicity"] == 512
    assert final_layout["vector_offset"] == -1
    assert final_layout["vector_multiplicity"] == 0
    assert final_layout["vector_components"] == 0
    assert final_layout["retained_from_native_components"] == []
    assert final_layout["retained_from_native_signs"] == []


def test_native_diagnostics_localize_a_missing_layer_primitive(tmp_path):
    """Catch one failed conditioner being reported only as a family-wide miss."""

    topology = _scalar_final_mixed_edge_topology()
    first_affine = topology["interactions"][1]["conv_tp_weights"]["layers"][0]
    width = first_affine["weight"]["shape"][1]
    topology["interactions"][1]["conv_tp_weights"]["layers"].insert(
        0,
        {
            "type": "layer_norm",
            "normalized_shape": [width],
            "eps": 1.0e-5,
            "weight": {"shape": [width], "values": [1.0] * width},
            "bias": {"shape": [width], "values": [0.0] * width},
        },
    )
    from symmetrix import symmetrix as native

    evaluator = _native_evaluator(
        native, _write_topology(tmp_path / "layer-1-unconditionable.json", topology)
    )
    layers = evaluator.mh1_portable_diagnostics["layers"]

    assert layers[0]["portable_ready"] is True
    assert layers[0]["reason"] == "portable MH-1 layer admitted"
    assert layers[1]["pair_conditioning_ready"] is False
    assert layers[1]["portable_ready"] is False
    assert layers[1]["reason"] == "Conditioned edge MLP plan is unavailable."


@pytest.mark.parametrize(
    ("mutation", "invariant"),
    [
        ("node", "node representation"),
        ("edge", "edge representation"),
        ("hidden", "hidden representation"),
        ("product", "product representation"),
        ("uvu", "tied-channel UVU layout"),
    ],
)
def test_native_mh1_admission_reports_layer_local_invariant(mutation, invariant):
    """Catch a malformed layer being hidden behind the old combined rejection."""

    topology = _scalar_final_mixed_edge_topology()
    if mutation == "node":
        topology["interactions"][1]["node_feats_irreps"] = "511x0e+512x1o"
    elif mutation == "edge":
        topology["interactions"][1]["edge_irreps"] = "127x0e+127x1o"
    elif mutation == "hidden":
        topology["interactions"][1]["hidden_irreps"] = "511x0e"
    elif mutation == "product":
        topology["products"][1]["symmetric_contractions"]["irreps_out"] = "511x0e"
    else:
        topology["interactions"][1]["conv_tp"]["instructions"][0]["path_shape"] = [
            127,
            1,
        ]

    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert not descriptor["compatible"]
    assert (
        descriptor["rejection_reason"] == f"MH-1 layer 1 {invariant} is inconsistent."
    )


@pytest.mark.parametrize(
    ("mutation", "invariant"),
    [
        ("node", "product connectivity"),
        ("target", "product connectivity"),
        ("linear_input", "product connectivity"),
        ("linear_output", "product connectivity"),
        ("target_irreps", "target representation"),
        ("gate_input", "gate input representation"),
        ("gate_output", "gate output representation"),
        ("readout", "readout class"),
    ],
)
def test_native_mh1_admission_reports_connectivity_and_routing_invariant(
    mutation, invariant
):
    """Catch disconnected products and collapsed target/gate/readout diagnostics."""

    topology = _scalar_final_mixed_edge_topology()
    if mutation == "node":
        topology["products"][1]["node_feats_irreps"] = "511x0e"
    elif mutation == "target":
        topology["products"][1]["target_irreps"] = "511x0e"
    elif mutation == "linear_input":
        topology["products"][1]["linear"]["irreps_in"] = "511x0e"
    elif mutation == "linear_output":
        topology["products"][1]["linear"]["irreps_out"] = "511x0e"
    elif mutation == "target_irreps":
        topology["interactions"][1]["target_irreps"] = "511x0e"
    elif mutation == "gate_input":
        topology["interactions"][1]["gate"]["irreps_in"] = "511x0e"
    elif mutation == "gate_output":
        topology["interactions"][1]["gate"]["irreps_out"] = "511x0e"
    else:
        topology["readouts"][1]["class"] = "LinearReadoutBlock"

    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert not descriptor["compatible"]
    assert (
        descriptor["rejection_reason"] == f"MH-1 layer 1 {invariant} is inconsistent."
    )


def test_native_mh1_admission_preserves_one_layer_hidden_rule():
    """Catch layer zero becoming scalar-hidden merely because it is also final."""

    topology = _build_one_layer_topology()
    from symmetrix import symmetrix as native

    descriptor = native._analyze_mh1_family_architecture(json.dumps(topology))
    assert not descriptor["compatible"]
    assert descriptor["rejection_reason"] == (
        "MH-1 layer 0 hidden representation is inconsistent."
    )
