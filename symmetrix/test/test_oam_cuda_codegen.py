"""Unit tests for the NequIP-OAM CUDA R1 code generator."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from symmetrix.oam_architecture import (
    oam_architecture_fingerprint,
    oam_tensor_product_architecture_fingerprint,
    resolve_oam_architecture,
)
from symmetrix.oam_cuda_codegen import (
    CAPABILITY_DIRECT_GEOMETRY,
    CAPABILITY_GROUPED_REVERSE_PARTIALS,
    CAPABILITY_STAGED_FORWARD_INPUTS,
    OAMCudaGenerationPolicy,
    _forward_path_groups,
    _render_forward_kernel,
    _render_harmonic_helpers,
    _render_descriptor,
    oam_cuda_artifact_metadata,
    recommended_oam_cuda_generation_policy,
    render_oam_cuda_plugin,
)
from symmetrix.oam_receiver_codegen import oam_receiver_jit_metadata

FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_metadata_is_deterministic_and_input_sensitive(contract):
    base = oam_cuda_artifact_metadata(contract, precision="float32", target="sm120")

    again = oam_cuda_artifact_metadata(
        copy.deepcopy(contract), precision="float32", target="sm120"
    )
    assert again == base

    other_precision = oam_cuda_artifact_metadata(
        contract, precision="float64", target="sm120"
    )
    assert other_precision["artifact_id"] != base["artifact_id"]
    assert other_precision["generation_fingerprint"] != base["generation_fingerprint"]

    other_target = oam_cuda_artifact_metadata(
        contract, precision="float32", target="sm90"
    )
    assert other_target["artifact_id"] != base["artifact_id"]

    other_threads = oam_cuda_artifact_metadata(
        contract, precision="float32", target="sm120", threads_per_block=64
    )
    assert other_threads["artifact_id"] != base["artifact_id"]


def test_cuda_metadata_exposes_only_the_full_message_schedule(contract):
    metadata = oam_cuda_artifact_metadata(contract, precision="float32", target="sm120")

    assert metadata["abi_version"] == 3
    assert metadata["execution_variants"] == ["full_message"]
    assert "projected_owner_schedule" not in metadata
    assert "projection_layout" not in metadata

    retrained = copy.deepcopy(contract)
    for layer in retrained["layers"]:
        layer["post_linear"]["weight"]["values"] = [
            value + 0.125 for value in layer["post_linear"]["weight"]["values"]
        ]
    retrained_metadata = oam_cuda_artifact_metadata(
        retrained, precision="float32", target="sm120"
    )
    assert retrained_metadata["artifact_id"] == metadata["artifact_id"]
    assert (
        retrained_metadata["generation_fingerprint"]
        == metadata["generation_fingerprint"]
    )


@pytest.mark.parametrize("mutation", ("mask", "path"))
def test_post_linear_topology_does_not_change_full_message_cuda_artifact(
    contract, mutation
):
    base = oam_cuda_artifact_metadata(contract, target="sm120")
    changed = copy.deepcopy(contract)
    post = changed["layers"][0]["post_linear"]
    if mutation == "mask":
        post["output_mask"]["values"][0] *= -1
    else:
        post["instructions"][0]["path_weight"] *= 2
    other = oam_cuda_artifact_metadata(changed, target="sm120")

    assert other["artifact_id"] == base["artifact_id"]
    assert other["structure_fingerprint"] == base["structure_fingerprint"]


def test_post_linear_bias_does_not_change_full_message_cuda_artifact(
    contract,
):
    architecture = resolve_oam_architecture(contract)
    base = oam_cuda_artifact_metadata(
        contract, target="sm120", architecture=architecture
    )
    changed = copy.deepcopy(contract)
    changed["layers"][0]["post_linear"]["bias"] = {
        "shape": [0],
        "values": [],
    }
    other = oam_cuda_artifact_metadata(
        changed, target="sm120", architecture=architecture
    )

    assert other["artifact_id"] == base["artifact_id"]
    assert other["structure_fingerprint"] == base["structure_fingerprint"]


def test_cuda_source_exports_only_full_message_launches(contract):
    source = render_oam_cuda_plugin(contract, precision="float32", target="sm120")

    assert "symmetrix_oam_cuda_plugin_query_v3" in source
    assert "symmetrix_oam_cuda_plugin_query_v2()" not in source
    assert "symmetrix_oam_projected_forward_kernel_0" not in source
    assert "symmetrix_oam_projected_reverse_kernel_0" not in source
    assert "oam_projected_forward_launch_0" not in source
    assert "oam_projected_reverse_launch_0" not in source
    assert "&oam_reverse_launch_0" in source
    assert '"full_message,projected"' not in source
    assert "symmetrix_oam_cuda_module_full_forward_schedule" in source
    assert "symmetrix_oam_cuda_module_direct_geometry_policy" in source


def test_cuda_full_message_descriptor_has_no_projected_launches(
    contract,
):
    metadata = oam_cuda_artifact_metadata(contract, target="sm120")
    descriptor = _render_descriptor(metadata, "float")

    assert '"full_message"' in descriptor
    assert '"full_message,projected"' not in descriptor
    launch_rows = descriptor.split("static const SymmetrixOAMCudaPluginV3")[0]
    assert "nullptr, nullptr" in launch_rows


def test_cuda_full_message_source_compiles_without_a_gpu(contract, tmp_path):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda-13.3/bin/nvcc"
    if not Path(nvcc).is_file():
        pytest.skip("CUDA compiler is unavailable")
    source_path = tmp_path / "oam_full_message.cu"
    object_path = tmp_path / "oam_full_message.o"
    source_path.write_text(
        render_oam_cuda_plugin(contract, precision="float32", target="sm120"),
        encoding="utf-8",
    )
    subprocess.run(
        [nvcc, "-std=c++17", "-arch=sm_120", "-c", source_path, "-o", object_path],
        check=True,
        capture_output=True,
        text=True,
    )
    assert object_path.is_file()


def test_generation_policy_is_validated_and_identity_bound(contract):
    baseline = oam_cuda_artifact_metadata(
        contract,
        target="sm120",
        generation_policy=OAMCudaGenerationPolicy(),
    )
    policy = OAMCudaGenerationPolicy(forward_live_output_budget=16)
    grouped = oam_cuda_artifact_metadata(
        contract, target="sm120", generation_policy=policy
    )

    assert grouped["generation_policy"] == policy.__dict__
    assert grouped["artifact_id"] != baseline["artifact_id"]
    assert grouped["generation_fingerprint"] != baseline["generation_fingerprint"]
    assert grouped["contract_fingerprint"] == baseline["contract_fingerprint"]
    assert grouped["semantic_fingerprint"] == baseline["semantic_fingerprint"]
    assert grouped["structure_fingerprint"] == baseline["structure_fingerprint"]

    staged = oam_cuda_artifact_metadata(
        contract,
        target="sm120",
        generation_policy=OAMCudaGenerationPolicy(
            forward_live_output_budget=16,
            forward_input_mode="staged",
        ),
    )
    assert staged["required_capabilities"] & CAPABILITY_STAGED_FORWARD_INPUTS
    assert not (baseline["required_capabilities"] & CAPABILITY_STAGED_FORWARD_INPUTS)
    assert staged["artifact_id"] != grouped["artifact_id"]

    with pytest.raises(ValueError, match="live output budget"):
        OAMCudaGenerationPolicy(forward_live_output_budget=12)
    with pytest.raises(ValueError, match="forward input mode"):
        OAMCudaGenerationPolicy(forward_input_mode="global")
    with pytest.raises(ValueError, match="forward grouping mode"):
        OAMCudaGenerationPolicy(forward_grouping_mode="edge")
    with pytest.raises(ValueError, match="forward group execution"):
        OAMCudaGenerationPolicy(forward_group_execution="blocks")
    with pytest.raises(ValueError, match="requires source-first-fit"):
        OAMCudaGenerationPolicy(forward_group_execution="warp_tasks")
    with pytest.raises(ValueError, match="requires a live output budget"):
        OAMCudaGenerationPolicy(forward_grouping_mode="source_first_fit")
    with pytest.raises(ValueError, match="radial adjoint mode"):
        OAMCudaGenerationPolicy(radial_adjoint_mode="mixed")
    with pytest.raises(ValueError, match="path reuse mode"):
        OAMCudaGenerationPolicy(path_reuse_mode="all")
    with pytest.raises(ValueError, match="reverse group execution"):
        OAMCudaGenerationPolicy(reverse_group_execution="warps")
    with pytest.raises(ValueError, match="require float32"):
        oam_cuda_artifact_metadata(
            contract,
            precision="float64",
            target="sm120",
            generation_policy=OAMCudaGenerationPolicy(radial_adjoint_mode="fp32"),
        )


def test_generation_policy_recommends_retained_sm120_fp32_warp_task_path(contract):
    recommended = recommended_oam_cuda_generation_policy("float32", "sm120")

    qualified = OAMCudaGenerationPolicy(
        forward_live_output_budget=32,
        forward_grouping_mode="source_first_fit",
        forward_group_execution="warp_tasks",
        forward_input_mode="staged",
        radial_adjoint_mode="fp32",
    )
    assert recommended == qualified
    l_contract = {
        "layers": [
            {},
            {
                "edge_tensor_product": {
                    "irreps_in1": "128x0e+64x1o+32x2e+32x3o",
                    "instructions": [{}] * 23,
                }
            },
            {},
        ]
    }
    xl_contract = {
        "layers": [
            {},
            {
                "edge_tensor_product": {
                    "irreps_in1": "320x0e+96x1o+64x2e+32x3o+32x4e",
                    "instructions": [{}] * 42,
                }
            },
            {},
        ]
    }
    grouped_policy = OAMCudaGenerationPolicy()
    assert (
        recommended_oam_cuda_generation_policy("float32", "sm120", contract=l_contract)
        == grouped_policy
    )
    assert (
        recommended_oam_cuda_generation_policy("float32", "sm120", contract=xl_contract)
        == grouped_policy
    )
    m_contract = copy.deepcopy(l_contract)
    m_contract["layers"][1]["edge_tensor_product"].update(
        irreps_in1="128x0e+64x1o+32x2e",
        instructions=[{}] * 11,
    )
    assert (
        recommended_oam_cuda_generation_policy("float32", "sm120", contract=m_contract)
        == grouped_policy
    )
    assert (
        recommended_oam_cuda_generation_policy("float64", "sm120")
        == OAMCudaGenerationPolicy()
    )
    assert (
        recommended_oam_cuda_generation_policy("float32", "sm90")
        == OAMCudaGenerationPolicy()
    )

    automatic = oam_cuda_artifact_metadata(
        contract, precision="float32", target="sm120"
    )
    explicit_reference = oam_cuda_artifact_metadata(
        contract,
        precision="float32",
        target="sm120",
        generation_policy=OAMCudaGenerationPolicy(),
    )
    assert automatic["generation_policy"] == {**qualified.__dict__}
    assert automatic["required_capabilities"] & CAPABILITY_STAGED_FORWARD_INPUTS
    assert automatic["spline_coordinate_type"] == "float"
    assert explicit_reference["spline_coordinate_type"] == "double"
    assert automatic["artifact_id"] != explicit_reference["artifact_id"]


def test_automatic_sm120_policy_falls_back_for_multi_path_output_contract():
    from test_nequip_oam_lane_layout import _build_contract

    contract = _build_contract()

    assert (
        recommended_oam_cuda_generation_policy("float32", "sm120", contract=contract)
        == OAMCudaGenerationPolicy()
    )
    metadata = oam_cuda_artifact_metadata(contract, precision="float32", target="sm120")
    assert metadata["generation_policy"] == {**OAMCudaGenerationPolicy().__dict__}


@pytest.mark.parametrize(
    "policy",
    [
        OAMCudaGenerationPolicy(forward_live_output_budget=16),
        OAMCudaGenerationPolicy(forward_input_mode="staged"),
        OAMCudaGenerationPolicy(reverse_group_execution="source_groups"),
    ],
)
def test_qualification_covers_the_entire_cuda_policy_tuple(contract, policy):
    metadata = oam_cuda_artifact_metadata(
        contract, precision="float64", target="sm120", generation_policy=policy
    )
    assert metadata["schedule_qualification"] == "candidate-unqualified-explicit"


def test_unqualified_cuda_cells_keep_conservative_full_message_schedule(contract):
    for precision, target in (
        ("float64", "sm120"),
        ("float32", "sm90"),
        ("float64", "sm90"),
    ):
        policy = recommended_oam_cuda_generation_policy(
            precision, target, contract=contract
        )
        assert policy == OAMCudaGenerationPolicy()


def test_forward_path_groups_preserve_instruction_order_and_weight_offsets():
    layer = {
        "edge_tensor_product": {
            "irreps_out": "1x0e+1x1o+1x2e+1x4e",
            "instructions": [
                {"i_out": 0, "has_weight": True, "path_shape": [2]},
                {"i_out": 1, "has_weight": True, "path_shape": [3]},
                {"i_out": 0, "has_weight": True, "path_shape": [5]},
                {"i_out": 2, "has_weight": False, "path_shape": [1]},
                {"i_out": 3, "has_weight": True, "path_shape": [7]},
            ],
        }
    }

    groups = _forward_path_groups(layer, live_output_budget=8)

    assert groups == [
        {
            "output_blocks": (0, 1),
            "instruction_indices": (0, 1, 2),
            "weight_offsets": (0, 2, 5),
            "live_output_components": 4,
        },
        {
            "output_blocks": (2,),
            "instruction_indices": (3,),
            "weight_offsets": (10,),
            "live_output_components": 5,
        },
        {
            "output_blocks": (3,),
            "instruction_indices": (4,),
            "weight_offsets": (10,),
            "live_output_components": 9,
        },
    ]
    assert _forward_path_groups(layer, live_output_budget=None) == [
        {
            "output_blocks": (0, 1, 2, 3),
            "instruction_indices": (0, 1, 2, 3, 4),
            "weight_offsets": (0, 2, 5, 10, 10),
            "live_output_components": 18,
        }
    ]


def test_source_first_fit_groups_share_inputs_and_pack_largest_outputs_first():
    layer = {
        "edge_tensor_product": {
            "irreps_in1": "4x0e+2x1o",
            "irreps_out": "4x0e+4x1o+4x2e+2x0e+2x1o",
            "instructions": [
                {
                    "i_in1": 0,
                    "i_out": 0,
                    "connection_mode": "uvu",
                    "has_weight": True,
                    "path_shape": [4, 1],
                    "wigner_3j": {"values": [1.0]},
                },
                {
                    "i_in1": 0,
                    "i_out": 1,
                    "connection_mode": "uvu",
                    "has_weight": True,
                    "path_shape": [4, 1],
                    "wigner_3j": {"values": [1.0, 0.0, 2.0]},
                },
                {
                    "i_in1": 0,
                    "i_out": 2,
                    "connection_mode": "uvu",
                    "has_weight": True,
                    "path_shape": [4, 1],
                    "wigner_3j": {"values": [1.0] * 5},
                },
                {
                    "i_in1": 1,
                    "i_out": 3,
                    "connection_mode": "uvu",
                    "has_weight": True,
                    "path_shape": [2, 1],
                    "wigner_3j": {"values": [1.0]},
                },
                {
                    "i_in1": 1,
                    "i_out": 4,
                    "connection_mode": "uvu",
                    "has_weight": True,
                    "path_shape": [2, 1],
                    "wigner_3j": {"values": [1.0] * 3},
                },
            ],
        }
    }

    groups = _forward_path_groups(
        layer, live_output_budget=8, grouping_mode="source_first_fit"
    )

    assert [(group["source_block"], group["output_blocks"]) for group in groups] == [
        (0, (2, 1)),
        (1, (4, 3)),
        (0, (0,)),
    ]
    assert [group["live_output_components"] for group in groups] == [8, 4, 1]
    assert [group["source_multiplicity"] for group in groups] == [4, 2, 4]


def test_source_grouped_warp_tasks_are_identity_bound_and_warp_local(contract):
    policy = OAMCudaGenerationPolicy(
        forward_live_output_budget=16,
        forward_grouping_mode="source_first_fit",
        forward_group_execution="warp_tasks",
        radial_adjoint_mode="fp32",
    )
    metadata = oam_cuda_artifact_metadata(
        contract,
        precision="float32",
        target="sm120",
        generation_policy=policy,
    )
    source = render_oam_cuda_plugin(
        contract,
        "sm120",
        precision="float32",
        generation_policy=policy,
    )

    assert "#pragma once" not in source
    assert metadata["generation_policy"]["forward_group_execution"] == "warp_tasks"
    assert metadata["forward_owner_schedule"] == (
        "receiver-source-group-channel-warp-grid-stride-v1"
    )
    assert all(count > 0 for count in metadata["forward_execution_count"])
    assert "constexpr int forward_warps = 4;" in source
    assert "const int lane = static_cast<int>(threadIdx.x) & 31;" in source
    assert (
        "const std::int64_t task_count = args.num_nodes\n"
        "        * static_cast<std::int64_t>(forward_execution_count);" in source
    )
    assert "static_cast<std::uint64_t>(receiver)" in source
    assert "__syncwarp();" in source
    assert "scratch.source[position] = source[position]" not in source
    forward_source = source.split("symmetrix_oam_forward_kernel_0", 1)[1].split(
        "symmetrix_oam_reverse_kernel_0", 1
    )[0]
    assert "const Scalar* staged_harmonics" not in forward_source

    staged_source = render_oam_cuda_plugin(
        contract,
        "sm120",
        precision="float32",
        generation_policy=OAMCudaGenerationPolicy(
            forward_live_output_budget=16,
            forward_grouping_mode="source_first_fit",
            forward_group_execution="warp_tasks",
            forward_input_mode="staged",
            radial_adjoint_mode="fp32",
        ),
    )
    staged_forward_source = staged_source.split("symmetrix_oam_forward_kernel_0", 1)[
        1
    ].split("symmetrix_oam_reverse_kernel_0", 1)[0]
    assert "const Scalar* staged_harmonics" in staged_forward_source
    assert "const Scalar* edge_vectors" not in staged_forward_source


def test_grouped_reverse_is_identity_bound_and_writes_geometry_partials(contract):
    policy = OAMCudaGenerationPolicy(reverse_group_execution="source_groups")
    metadata = oam_cuda_artifact_metadata(
        contract,
        precision="float32",
        target="sm120",
        generation_policy=policy,
    )
    source = render_oam_cuda_plugin(
        contract,
        "sm120",
        precision="float32",
        generation_policy=policy,
    )

    assert metadata["required_capabilities"] & CAPABILITY_GROUPED_REVERSE_PARTIALS
    assert metadata["reverse_owner_schedule"] == ("source-irrep-specialized-kernel-v1")
    assert metadata["layers"][0]["reverse_group_count"] > 0
    assert metadata["layers"][-1]["reverse_group_count"] == 0
    assert all(row["reverse_group_count"] > 0 for row in metadata["layers"][:-1])
    assert all(row["reverse_group_flags"] == 0 for row in metadata["layers"])
    assert "symmetrix_oam_reverse_kernel_0_group_0" in source
    assert "symmetrix_oam_reverse_kernel_1(" in source
    assert "switch (group)" not in source
    assert "source_adjoint[source" in source
    assert "source_partials[segment" not in source
    assert "geometry_partials" in source


def test_grouped_forward_limits_accumulator_lifetimes():
    irreps = "1x0e+1x1o+1x2e+1x4e"
    instructions = []
    for block, width in enumerate((1, 3, 5, 9)):
        values = [0.0] * (width * width)
        for component in range(width):
            values[component * width + component] = 1.0
        instructions.append(
            {
                "i_in1": block,
                "i_in2": 0,
                "i_out": block,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_shape": [1, 1],
                "path_weight": 1.0,
                "wigner_3j": {
                    "shape": [width, 1, width],
                    "values": values,
                },
            }
        )
    layer = {
        "edge_irreps": "1x0e",
        "tp_output_irreps": irreps,
        "edge_tensor_product": {
            "irreps_in1": irreps,
            "irreps_in2": "1x0e",
            "irreps_out": irreps,
            "instructions": instructions,
        },
    }
    row = {
        "source_dimension": 18,
        "output_dimension": 18,
        "edge_harmonic_count": 1,
        "normalization_factor": 1.0,
        "forward_static_shared_bytes": 96,
    }

    baseline = _render_forward_kernel(layer, row, 0, "float", 128)
    grouped = _render_forward_kernel(layer, row, 0, "float", 128, live_output_budget=8)

    edge_loop = (
        "for (std::int64_t directed_edge = first; directed_edge < last; "
        "++directed_edge)"
    )
    assert baseline.count(edge_loop) == 1
    assert grouped.count(edge_loop) == 3
    assert grouped.count("Scalar accumulator_0[1]") == 1
    assert grouped.count("Scalar accumulator_1[3]") == 1
    assert grouped.count("Scalar accumulator_2[5]") == 1
    assert grouped.count("Scalar accumulator_3[9]") == 1
    assert "static_cast<const Scalar*>(args.edge_harmonics)" not in baseline

    staged = _render_forward_kernel(
        layer,
        row,
        0,
        "float",
        128,
        live_output_budget=8,
        input_mode="staged",
    )
    assert "static_cast<const Scalar*>(args.edge_harmonics)" in staged
    assert "static_cast<const Scalar*>(args.edge_vectors)" not in staged
    assert "oam_compute_harmonics" not in staged
    assert staged.count(edge_loop) == 3


def test_metadata_requires_direct_geometry(contract):
    metadata = oam_cuda_artifact_metadata(contract, precision="float32", target="sm120")
    assert metadata["direct_geometry"] is True
    assert metadata["required_capabilities"] & CAPABILITY_DIRECT_GEOMETRY


@pytest.mark.parametrize(
    ("mode", "scratch_type", "lane_type", "marker"),
    (
        ("fp64", "double radial[4]", "double radial_adjoint = 0.0", None),
        ("fp32", "Scalar radial[4]", "Scalar radial_adjoint = Scalar(0)", None),
        (
            "compensated_fp32",
            "Scalar radial[4]",
            "Scalar radial_adjoint = Scalar(0)",
            "Scalar radial_compensation = Scalar(0)",
        ),
        (
            "hierarchical_fp32",
            "double radial[4]",
            "Scalar radial_adjoint = Scalar(0)",
            "double radial_adjoint_total",
        ),
    ),
)
def test_radial_adjoint_policy_changes_generated_reduction(
    contract, mode, scratch_type, lane_type, marker
):
    source = render_oam_cuda_plugin(
        contract,
        "sm120",
        generation_policy=OAMCudaGenerationPolicy(radial_adjoint_mode=mode),
    )

    assert scratch_type in source
    assert lane_type in source
    coordinate_type = "double" if mode == "fp64" else "Scalar"
    assert f"const {coordinate_type} coordinate)" in source
    if marker is not None:
        assert marker in source


def test_oam_xl_degree_four_harmonics_are_generated():
    source = _render_harmonic_helpers(
        {"layers": [{"edge_harmonic_count": 25}]}, "float"
    )

    assert "raw[24]" in source
    assert "raw_dx[24]" in source
    assert "raw_dy[24]" in source
    assert "raw_dz[24]" in source
    assert source.count("raw[9] =") == 2

    degree_one_source = _render_harmonic_helpers(
        {"layers": [{"edge_harmonic_count": 4}]}, "float"
    )
    assert degree_one_source.count("Scalar raw[4];") == 1


def test_harmonic_generator_rejects_degrees_above_oam_xl():
    with pytest.raises(ValueError, match="l_max <= 4"):
        _render_harmonic_helpers({"layers": [{"edge_harmonic_count": 36}]}, "float")


def test_metadata_contract_fingerprint_matches_validated_contract(contract):
    cuda_metadata = oam_cuda_artifact_metadata(
        contract, precision="float64", target="sm120"
    )
    host_metadata = oam_receiver_jit_metadata(
        contract, "float64", host_target="host-isa-x86-64-v3"
    )
    assert cuda_metadata[
        "contract_fingerprint"
    ] == oam_tensor_product_architecture_fingerprint(contract)
    assert cuda_metadata["contract_fingerprint"] != oam_architecture_fingerprint(
        contract
    )
    assert (
        cuda_metadata["contract_fingerprint"] == host_metadata["contract_fingerprint"]
    )


def test_module_identity_globals_have_c_language_linkage(contract):
    source = render_oam_cuda_plugin(contract, target="sm120")

    assert "#if defined(__CUDACC_RTC__) || defined(SYMMETRIX_OAM_CUDA_MODULE)" in source
    # The driver loader queries these ABI values with cuModuleGetGlobal, so the
    # names need C language linkage.  Do not add a second ``extern`` storage
    # class: CUDA 13 NVRTC rejects ``extern "C" extern __device__`` even though
    # the offline compiler accepts it.
    assert (
        'extern "C" __device__ __constant__ unsigned int\n'
        "symmetrix_oam_cuda_module_abi_version" in source
    )
    assert (
        'extern "C" __device__ __constant__ char\n'
        "symmetrix_oam_cuda_module_artifact_id" in source
    )
    assert 'extern "C" extern __device__' not in source


def test_fingerprints_separate_contract_math_layout_and_schedule(contract):
    base = oam_cuda_artifact_metadata(contract, target="sm120")
    provenance = copy.deepcopy(contract)
    provenance["model_metadata"]["exporter_version"] = "another-exporter"
    changed_contract = oam_cuda_artifact_metadata(provenance, target="sm120")
    for field in (
        "contract_fingerprint",
        "artifact_id",
        "semantic_fingerprint",
        "structure_fingerprint",
        "generation_fingerprint",
    ):
        assert changed_contract[field] == base[field]

    math_variant = copy.deepcopy(contract)
    math_variant["layers"][0]["normalization"]["factor"] *= 2
    changed_math = oam_cuda_artifact_metadata(math_variant, target="sm120")
    assert changed_math["semantic_fingerprint"] != base["semantic_fingerprint"]
    assert changed_math["structure_fingerprint"] == base["structure_fingerprint"]
    assert changed_math["generation_fingerprint"] == base["generation_fingerprint"]

    changed_schedule = oam_cuda_artifact_metadata(
        contract, target="sm120", threads_per_block=64
    )
    assert changed_schedule["generation_fingerprint"] != base["generation_fingerprint"]
    assert changed_schedule["semantic_fingerprint"] == base["semantic_fingerprint"]
    assert changed_schedule["structure_fingerprint"] == base["structure_fingerprint"]

    changed_precision = oam_cuda_artifact_metadata(contract, "float64", "sm120")
    assert changed_precision["generation_fingerprint"] != base["generation_fingerprint"]
    assert changed_precision["semantic_fingerprint"] == base["semantic_fingerprint"]
    assert changed_precision["structure_fingerprint"] == base["structure_fingerprint"]


def test_runtime_annotations_do_not_change_cuda_identity(contract):
    annotated = copy.deepcopy(contract)
    product = annotated["layers"][0]["edge_tensor_product"]
    product["output_mask"]["dtype"] = "float32"
    product["instructions"][0]["wigner_3j"]["device"] = "cuda:0"
    assert oam_cuda_artifact_metadata(annotated, target="sm120") == (
        oam_cuda_artifact_metadata(contract, target="sm120")
    )


def test_metadata_target_accepts_device_environment_mapping(contract):
    metadata = oam_cuda_artifact_metadata(
        contract,
        precision="float32",
        target={"architecture": "sm120", "compute_capability": 120},
    )
    assert metadata["target_architecture"] == "sm120"
    assert metadata["target_compute_capability"] == 120


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_cooperative_kernels_reject_insufficient_shared_memory(contract, precision):
    metadata = oam_cuda_artifact_metadata(contract, precision=precision, target="sm120")
    required = max(
        layer[direction + "_static_shared_bytes"]
        for layer in metadata["layers"]
        for direction in ("forward", "reverse")
    )
    target = {"architecture": "sm120", "max_shared_memory_per_block": required}
    oam_cuda_artifact_metadata(contract, precision=precision, target=target)
    target["max_shared_memory_per_block"] = required - 1
    with pytest.raises(ValueError, match="shared-memory"):
        oam_cuda_artifact_metadata(contract, precision=precision, target=target)


def test_metadata_rejects_invalid_inputs(contract):
    with pytest.raises(ValueError, match="precision"):
        oam_cuda_artifact_metadata(contract, precision="float16", target="sm120")
    with pytest.raises(ValueError, match="threads_per_block"):
        oam_cuda_artifact_metadata(
            contract, precision="float32", target="sm120", threads_per_block=100
        )
    with pytest.raises(ValueError, match="compute capability"):
        oam_cuda_artifact_metadata(contract, precision="float32", target="gpu")


def test_layer_layouts_reject_weight_spline_mismatch(contract):
    # The schema validator already binds both counts to the radial MLP, so
    # this invariant is defense in depth at the layout level.
    from symmetrix.oam_cuda_codegen import _layer_layouts

    broken = copy.deepcopy(contract)
    broken["layers"][0]["kokkos_radial"]["function_count"] = 2
    with pytest.raises(ValueError, match="function count"):
        _layer_layouts(broken)
