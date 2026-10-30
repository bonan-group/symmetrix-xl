from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from symmetrix import mace_nonlinear_jit_codegen as mh1_codegen
from test_execution_mh1_contract import (
    make_execution_mh1_v4_contract,
    mh1_definition,
    mh1_node_definition,
)


def _contract():
    definition = mh1_node_definition.__wrapped__(mh1_definition.__wrapped__())
    return make_execution_mh1_v4_contract(definition)


def test_device_module_publishes_product_free_split_phases():
    contract = _contract()

    source = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)
    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(contract, 80)

    assert metadata["abi"] == "symmetrix.jit.mh1.cuda-plugin/5"
    assert metadata["abi_version"] == 5
    assert plan["schema"] == "symmetrix.jit.mh1.cuda-launch-plan/6"
    assert plan["version"] == 6
    assert plan["module_abi"] == "symmetrix.jit.mh1.cuda-module/5"
    assert plan["module_abi_version"] == 5
    assert plan["split_phase_composition"] is True
    for layer in plan["nodes"]:
        assert [
            bool(layer[phase])
            for phase in (
                "split_pre_forward",
                "split_post_forward",
                "split_post_reverse",
                "split_pre_reverse",
            )
        ] == [True, True, True, True]

    assert "SymmetrixJitMH1CudaNodeForwardArgsV5" in source
    assert "SymmetrixJitMH1CudaNodeReverseArgsV5" in source
    for layer in plan["nodes"]:
        for phase in (
            "pre_forward",
            "split_pre_forward",
            "split_post_forward",
            "split_post_reverse",
            "split_pre_reverse",
            "pre_reverse",
        ):
            assert all(launch["kernel"] in source for launch in layer[phase])


def test_device_launch_plans_bind_the_active_module_abis():
    contract = _contract()
    cuda_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)
    hip_plan = mh1_codegen.execution_mh1_hip_module_v4_launch_plan(
        contract,
        {
            "architecture": "gfx1151",
            "target_features": ["sramecc-", "xnack-"],
            "compiler_offload_target": "gfx1151",
            "native_subgroup_width": 32,
            "compute_unit_count": 40,
        },
    )

    assert (cuda_plan["module_abi"], cuda_plan["module_abi_version"]) == (
        mh1_codegen.MH1_CUDA_MODULE_ABI,
        mh1_codegen.MH1_CUDA_MODULE_ABI_VERSION,
    )
    assert (hip_plan["module_abi"], hip_plan["module_abi_version"]) == (
        mh1_codegen.MH1_HIP_MODULE_ABI,
        mh1_codegen.MH1_HIP_MODULE_ABI_VERSION,
    )


def _surrounding_shape(layer):
    result = {}
    for phase in (
        "split_pre_forward",
        "split_post_forward",
        "split_post_reverse",
        "split_pre_reverse",
    ):
        shape = []
        for launch in layer[phase]:
            kernel = launch["kernel"].split("symmetrix_execution_mh1_split_", 1)[1]
            if "_block_" in kernel:
                kernel = kernel.rsplit("_block_", 1)[0]
            entry = (
                kernel,
                launch["grid"],
                launch.get("tile_nodes"),
                launch["threads"],
            )
            if not shape or shape[-1] != entry:
                shape.append(entry)
        result[phase] = shape
    return result


def test_generation_19_uses_qualified_external_surrounding_schedule():
    """The external boundary stays intact with occupancy-safe linear2 tiles."""

    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(_contract(), 80)
    expected = (
        {
            "split_pre_forward": [
                (
                    "node_grouped_l0_forward_residual_grouped",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                (
                    "node_grouped_l0_forward_message_grouped",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                ("pre_gate_kernel_0", "persistent_nodes", None, 128),
                (
                    "node_grouped_l0_forward_linear2_grouped",
                    "node_channel_tiles",
                    64,
                    256,
                ),
                ("pre_retain_kernel_0", "persistent_nodes", None, 128),
                ("node_grouped_l0_forward_skip_grouped", "node_channel_tiles", 32, 256),
            ],
            "split_post_forward": [
                (
                    "node_tiled_l0_forward_product_linear",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                ("post_finalize_kernel_0", "persistent_nodes", None, 128),
            ],
            "split_post_reverse": [
                ("post_readout_kernel_0", "persistent_nodes", None, 128),
                (
                    "node_tiled_l0_reverse_product_linear",
                    "node_channel_tiles",
                    32,
                    256,
                ),
            ],
            "split_pre_reverse": [
                (
                    "node_grouped_l0_reverse_linear2_grouped",
                    "node_channel_tiles",
                    64,
                    256,
                ),
                ("pre_gate_reverse_kernel_0", "persistent_nodes", None, 128),
                ("pre_scale_kernel_0", "persistent_nodes", None, 128),
                (
                    "node_grouped_l0_reverse_message_grouped",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                ("pre_density_kernel_0", "persistent_nodes", None, 128),
            ],
        },
        {
            "split_pre_forward": [
                (
                    "node_grouped_l1_forward_residual_grouped",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                (
                    "node_grouped_l1_forward_message_grouped",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                ("pre_gate_kernel_1", "persistent_nodes", None, 128),
                (
                    "node_grouped_l1_forward_linear2_grouped",
                    "node_channel_tiles",
                    64,
                    256,
                ),
                ("pre_retain_kernel_1", "persistent_nodes", None, 128),
                ("node_grouped_l1_forward_skip_grouped", "node_channel_tiles", 32, 256),
            ],
            "split_post_forward": [
                (
                    "node_tiled_l1_forward_product_linear",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                ("post_finalize_kernel_1", "persistent_nodes", None, 128),
            ],
            "split_post_reverse": [
                (
                    "node_grouped_l1_reverse_readout1_recompute_grouped",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                ("specialized_readout_middle_kernel_1", "persistent_nodes", None, 128),
                (
                    "node_grouped_l1_reverse_readout1_grouped",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                (
                    "node_grouped_l1_reverse_skip_grouped",
                    "node_channel_tiles",
                    32,
                    256,
                ),
                (
                    "node_tiled_l1_reverse_product_linear",
                    "node_channel_tiles",
                    32,
                    256,
                ),
            ],
            "split_pre_reverse": [
                (
                    "node_grouped_l1_reverse_linear2_grouped",
                    "node_channel_tiles",
                    64,
                    256,
                ),
                ("pre_gate_reverse_kernel_1", "persistent_nodes", None, 128),
                (
                    "node_tiled_l1_reverse_residual",
                    "node_channel_tiles",
                    8,
                    256,
                ),
                ("pre_scale_kernel_1", "persistent_nodes", None, 128),
                (
                    "node_grouped_l1_reverse_message_grouped",
                    "node_channel_tiles",
                    64,
                    256,
                ),
                ("pre_density_kernel_1", "persistent_nodes", None, 128),
            ],
        },
    )
    assert tuple(_surrounding_shape(layer) for layer in plan["nodes"]) == expected
    exact_tiled_extents = (
        {
            "split_pre_forward": [
                (512, 1, 32),
                (512, 1, 32),
                (512, 1, 64),
                (128, 1, 32),
            ],
            "split_post_forward": [(1, 1, 32), (1, 3, 32)],
            "split_post_reverse": [(1, 1, 32), (1, 3, 32)],
            "split_pre_reverse": [(512, 1, 64), (512, 1, 32)],
        },
        {
            "split_pre_forward": [
                (512, 1, 32),
                (512, 1, 32),
                (512, 1, 64),
                (32, 1, 32),
            ],
            "split_post_forward": [(1, 1, 32)],
            "split_post_reverse": [
                (32, 1, 32),
                (32, 1, 32),
                (128, 1, 32),
                (1, 1, 32),
            ],
            "split_pre_reverse": [
                (512, 1, 64),
                (1, 1, 8),
                (1, 3, 8),
                (512, 1, 64),
            ],
        },
    )
    for layer, expected_extents in zip(plan["nodes"], exact_tiled_extents, strict=True):
        for phase, phase_extents in expected_extents.items():
            tiled = [
                launch
                for launch in layer[phase]
                if launch["grid"] == "node_channel_tiles"
            ]
            assert [
                (launch["channels"], launch["components"], launch["tile_nodes"])
                for launch in tiled
            ] == phase_extents
            assert all(
                launch["tile_channels"] == 32 and launch["threads"] == 256
                for launch in tiled
            )
    assert plan["nodes"][1]["pre_forward"][0]["tile_nodes"] == 32
    assert plan["nodes"][1]["pre_reverse"][0]["tile_nodes"] == 32
    source = mh1_codegen.render_execution_mh1_cuda_module_v4(_contract(), 80)
    readout_recompute = source.split(
        "node_grouped_l1_reverse_readout1_recompute_grouped(", 1
    )[1]
    readout_recompute = readout_recompute.split(")", 1)[0]
    assert "SymmetrixJitMH1CudaNodeReverseArgsV5 args" in readout_recompute
    assert "SymmetrixJitMH1CudaNodeForwardArgsV5 args" not in readout_recompute
    readout_middle = source.split("specialized_readout_middle_kernel_1", 1)[1]
    readout_middle = readout_middle.split('extern "C" __global__ void', 1)[0]
    assert (
        "const float energy_scale = static_cast<float>(args.energy_scale);"
        in readout_middle
    )
    assert all(
        sum(launch["grid"] == "node_channel_tiles" for launch in layer[phase]) >= 1
        for layer in plan["nodes"]
        for phase in (
            "split_pre_forward",
            "split_post_forward",
            "split_post_reverse",
            "split_pre_reverse",
        )
    )


@pytest.mark.parametrize(
    ("policy", "schedule"),
    (
        ("full-retention-v1", "tuned-external-surrounding-retained-v1"),
        ("reuse-adjoints-v1", "tuned-external-surrounding-retained-v1"),
        ("retain-interaction-v1", "composed-v5-pre-gate-replay-v1"),
        ("recompute-v1", "composed-v5-full-replay-v1"),
    ),
)
def test_generation_19_schedule_names_truthfully_describe_state_replay(
    policy, schedule
):
    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        _contract(), 80, node_state_policy=policy
    )

    assert plan["node_reverse_schedule"] == schedule
    launch_counts = [
        [
            len(layer[phase])
            for phase in (
                "pre_forward",
                "split_pre_forward",
                "split_post_forward",
                "split_post_reverse",
                "split_pre_reverse",
                "pre_reverse",
            )
        ]
        for layer in plan["nodes"]
    ]
    assert launch_counts[0] == [1, 6, 3, 3, 5, 1]
    assert launch_counts[1] == [
        1,
        6,
        2,
        2 if policy == "recompute-v1" else 5,
        7,
        1,
    ]
    for layer in plan["nodes"]:
        for phase in (
            "pre_forward",
            "split_pre_forward",
            "split_post_forward",
            "split_post_reverse",
            "split_pre_reverse",
            "pre_reverse",
        ):
            for launch in layer[phase]:
                if launch["grid"] != "node_channel_tiles":
                    continue
                assert launch["tile_channels"] == 32
                assert launch["channels"] > 0
                assert launch["components"] > 0
                assert launch["tile_nodes"] in (8, 32, 64, 128)


@pytest.mark.parametrize("policy", ("reuse-adjoints-v1", "retain-interaction-v1"))
def test_message_adjoint_alias_fuses_density_dot_before_overwrite(policy):
    source = mh1_codegen.render_execution_mh1_cuda_module_v4(
        _contract(), 80, node_state_policy=policy
    )
    message_reverse = source.split("node_grouped_l0_reverse_message_grouped", 1)[1]
    message_reverse = message_reverse.split('extern "C" __global__ void', 1)[0]

    density_dot = message_reverse.index("density_dot_0")
    adjoint_write = message_reverse.index("(args.message_adjoint)[")
    assert density_dot < adjoint_write
    assert "atomicAdd((args.node_density_adjoint)" in message_reverse
    assert "args.node_density_adjoint[node] = density_bias_reduction[0]" in source
    assert "* args.node_density_adjoint[node] / normalization" in source

    native = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    assert "state.execution_output_ir_mul.data(),num_nodes,message_width" in native
    assert "?state.execution_output_ir_mul:state.messages" in native


def test_device_split_phases_do_not_own_coefficients():
    contract = _contract()
    source = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)

    split_source = source[source.index("symmetrix_execution_mh1_split_") :]
    assert "product_parameters" not in split_source
    assert "forward_product_tiled" not in split_source
    assert "reverse_product_tiled" not in split_source


def test_generation_19_modules_contain_no_private_product_implementation():
    contract = _contract()
    device = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)
    host = mh1_codegen.render_jit_mh1_host_plugin_v7(contract, precision="float32")
    retired_tokens = (
        "product_parameters",
        "tile_product_forward",
        "tile_product_reverse",
        "_product_forward(",
        "_product_reverse(",
        "NODE_TILE_PRODUCT",
    )
    assert all(token not in device for token in retired_tokens)
    assert all(token not in host for token in retired_tokens)

    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)
    assert plan["version"] == 6
    for node in plan["nodes"]:
        assert node["pre_forward"]
        assert node["split_pre_forward"]
        assert node["post_forward"] == node["split_post_forward"]
        assert node["post_reverse"] == node["split_post_reverse"]
        assert all("product" not in launch["kernel"] for launch in node["pre_forward"])
        assert all("product" not in launch["kernel"] for launch in node["pre_reverse"])


def test_nonlinear_codegen_has_no_private_product_renderer_or_formula():
    source = Path(mh1_codegen.__file__).read_text()
    retired_symbols = (
        "_render_node_product_helpers",
        "render_execution_mh1_node_product_cpp_helpers",
        "render_execution_mh1_node_program_cpp_helpers",
        "render_jit_mh1_host_plugin_v4",
        "render_jit_mh1_host_plugin_v5",
        "render_jit_mh1_host_plugin_v6",
        "render_jit_mh1_cuda_plugin_v4",
        "_cuda_block_product_forward",
        "_cuda_block_product_reverse",
        "_cuda_tiled_product_forward",
        "_cuda_tiled_product_reverse",
        "product_parameters",
        "runtime_coefficient_count",
    )
    assert all(symbol not in source for symbol in retired_symbols)
    assert "monomial_groups" not in source
    assert "coefficient_pointer" not in source

    canonical = (
        Path(mh1_codegen.__file__)
        .with_name("generated_operator_codegen.py")
        .read_text()
    )
    assert "monomial_groups" in canonical
    assert "term_components" in canonical


def test_float64_device_abi_contains_only_split_v5_energy_scale_type():
    source = mh1_codegen.render_execution_mh1_cuda_module_v4(
        _contract(), 80, precision="float64"
    )
    v5 = source.split("typedef struct SymmetrixJitMH1CudaNodeReverseArgsV5", 1)[
        1
    ].split("} SymmetrixJitMH1CudaNodeReverseArgsV5;", 1)[0]

    assert "SymmetrixJitMH1CudaNodeReverseArgsV4" not in source
    assert "double energy_scale;" in v5
    assert (
        "SymmetrixJitMH1CudaNodeReverseArgsV5 energy_scale must remain double" in source
    )


def test_split_phase_full_retention_stores_and_consumes_pre_gate():
    contract = _contract()
    retained = mh1_codegen.render_execution_mh1_cuda_module_v4(
        contract, 80, node_state_policy="full-retention-v1"
    )
    recompute = mh1_codegen.render_execution_mh1_cuda_module_v4(
        contract, 80, node_state_policy="recompute-v1"
    )
    retained_external = retained[retained.index("symmetrix_execution_mh1_split_") :]
    recompute_external = recompute[recompute.index("symmetrix_execution_mh1_split_") :]

    assert "args.retained_pre_gate + node" in retained_external
    assert "retained_pre_gate[column] = pre_gate[column]" in retained_external
    assert "const float value = (retained_pre_gate)[column]" in retained_external
    assert "args.retained_pre_gate + node" not in recompute_external
    assert "retained_pre_gate[column] = arena[column]" not in recompute_external


def test_hip_device_module_uses_the_same_split_boundary():
    contract = _contract()
    target = {
        "architecture": "gfx1151",
        "target_features": ["sramecc-", "xnack-"],
        "compiler_offload_target": "gfx1151",
        "native_subgroup_width": 32,
        "compute_unit_count": 40,
    }

    source = mh1_codegen.render_execution_mh1_hip_module_v4(contract, target)
    plan = mh1_codegen.execution_mh1_hip_module_v4_launch_plan(contract, target)

    assert plan["split_phase_composition"] is True
    assert all(
        launch["kernel"] in source
        for layer in plan["nodes"]
        for phase in ("split_pre_forward", "split_pre_reverse")
        for launch in layer[phase]
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_hip_device_module_compiles_with_hipcc(tmp_path, precision):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is unavailable for the MH-1 split-phase source check")
    contract = _contract()
    target = {
        "architecture": "gfx1151",
        "target_features": ["sramecc-", "xnack-"],
        "compiler_offload_target": "gfx1151",
        "native_subgroup_width": 32,
        "compute_unit_count": 40,
    }
    source_path = tmp_path / f"mh1_split_phase_{precision}.hip"
    source = mh1_codegen.render_execution_mh1_hip_module_v4(
        contract, target, precision=precision
    )
    # hipRTC supplies CUDA-compatible scalar aliases, launch indices, and
    # force-inline spelling without headers.  For the hipcc syntax check use
    # the equivalent runtime header and remove the standalone RTC aliases.
    source = source.replace("typedef signed int int32_t;\n", "")
    source = source.replace("typedef unsigned int uint32_t;\n", "")
    source = source.replace("typedef signed long long int64_t;\n", "")
    source = "#include <hip/hip_runtime.h>\n" + source
    source_path.write_text(source)

    completed = subprocess.run(
        [
            hipcc,
            "-x",
            "hip",
            "-std=c++20",
            "--offload-arch=gfx1151",
            "-fsyntax-only",
            str(source_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_device_composition_preserves_one_stream_without_an_interphase_fence():
    implementation = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    forward_start = implementation.index("if(shared_device_product_program)")
    forward_end = implementation.index("launch_node_forward(", forward_start)
    forward = implementation[forward_start:forward_end]
    reverse_start = implementation.index(
        "if(shared_device_product_program)", forward_end
    )
    reverse_end = implementation.index("launch_node_reverse(", reverse_start)
    reverse = implementation[reverse_start:reverse_end]

    assert forward.index("launch_split_forward") < forward.index(
        "launch_m0_forward_strided"
    )
    assert forward.rindex("launch_split_forward") > forward.index(
        "launch_m1_forward_strided"
    )
    assert reverse.index("launch_split_reverse") < reverse.index(
        "launch_m0_reverse_strided"
    )
    assert reverse.rindex("launch_split_reverse") > reverse.index(
        "launch_m1_reverse_strided"
    )
    assert "Kokkos::fence" not in forward
    assert "Kokkos::fence" not in reverse
    assert forward.count(",stream,") >= 4
    assert reverse.count(",stream,") >= 4


def test_legacy_device_route_does_not_access_common_product_artifacts():
    implementation = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    for marker, end_marker in (
        ("launch_m0_forward_strided", "launch_node_forward("),
        ("launch_m0_reverse_strided", "launch_node_reverse("),
    ):
        split = implementation.index(marker)
        start = implementation.rindex("if(shared_device_product_program)", 0, split)
        end = implementation.index(end_marker, split)
        branch = implementation[start:end]
        assert branch.index("execution_mh1_product_artifacts.at(layer)") > branch.index(
            "if(shared_device_product_program)"
        )
        prefix = implementation[implementation.rfind("#if", 0, start) : start]
        assert "execution_mh1_product_artifacts.at(layer)" not in prefix


def test_product_grid_cap_converts_per_compute_unit_to_total_blocks():
    implementation = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    loader = implementation[
        implementation.index(
            "load_execution_mh1_product_module("
        ) : implementation.index("execution_mh1_product_module_ready(")
    ]
    assert "persistent_blocks_per_compute_unit>32" in loader
    assert (
        "persistent_blocks_per_compute_unit\n"
        "            *execution_device_execution_environment().compute_unit_count"
        in loader
    )
    # RTX 5090 exposes 170 SMs: the default common-product profile is 8/SM.
    assert 8 * 170 == 1360


def test_product_diagnostics_distinguish_cap_from_realized_directional_grids():
    implementation = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    for direction in ("forward", "reverse"):
        assert f'" {direction}_logical_tiles="' in implementation
        assert f'" {direction}_tiles_per_block="' in implementation
        assert f'" {direction}_required_blocks="' in implementation
        assert f'" {direction}_realized_blocks="' in implementation
        assert (
            f"product_artifact.{direction}_realized_blocks=std::min(" in implementation
        )
    # Official 512-channel / 256-node chunk32 case: one logical tile per
    # channel/node pair, 256 tiles per block, below the 8 * 170 SM cap.
    logical_tiles = 512 * 256
    required_blocks = logical_tiles // 256
    assert logical_tiles == 131072
    assert required_blocks == 512
    assert min(required_blocks, 1360) == 512


def test_common_operator_loader_uses_directional_logical_geometry():
    loader = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_jit_operator_module.cpp"
    ).read_text()
    assert "symmetric_launch_blocks" in loader
    assert "state.reverse_node_tile : state.forward_node_tile" in loader
    assert "state.input_components : state.output_components" in loader
    assert "logical_tiles, state.logical_tiles_per_block" in loader


def test_common_operator_loader_rejects_unadvertised_aliasing_before_launch():
    loader = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_jit_operator_module.cpp"
    ).read_text()
    assert "validate_symmetric_aliasing" in loader
    assert "args.output == args.input" in loader
    assert "args.input_adjoint == args.input" in loader
    helper = loader[loader.index("bool validate_symmetric_product_launch") :]
    helper = helper[: helper.index("\n}\n")]
    assert helper.index("if (args.num_nodes == 0)") < helper.index(
        "validate_symmetric_aliasing"
    )
    for launch in (
        "launch_m0_forward_strided",
        "launch_m0_reverse_strided",
        "launch_m1_forward_strided",
        "launch_m1_reverse_strided",
    ):
        body = loader[loader.index(f"void OperatorModule::{launch}") :]
        body = body[: body.index("\n}\n")]
        assert "validate_symmetric_product_launch" in body
        assert body.index("validate_symmetric_product_launch") < body.index(
            "cuLaunchKernel"
        )


def test_common_operator_loader_requires_exact_product_capabilities():
    loader = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_jit_operator_module.cpp"
    ).read_text()
    assert "state.capabilities != expectation.required_capabilities" in loader
    assert "symmetric semantic capabilities do not match" in loader


def test_mh1_common_product_admission_uses_schedule_capabilities_without_alias_bit():
    implementation = (
        Path(__file__).parents[2] / "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    loader = implementation[
        implementation.index(
            "load_execution_mh1_product_module("
        ) : implementation.index("execution_mh1_product_module_ready(")
    ]
    assert (
        "expectation.required_capabilities=\n"
        '            metadata.at("capabilities").get<std::uint32_t>();'
    ) in loader
    assert "SYMMETRIX_JIT_OPERATOR_M0_ALIAS_SAFE_V1" not in loader
    assert "SYMMETRIX_JIT_OPERATOR_M1_ALIAS_SAFE_V1" not in loader
