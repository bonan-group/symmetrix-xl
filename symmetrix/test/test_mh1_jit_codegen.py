import ctypes
import inspect
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import symmetrix.mh1_jit_codegen as mh1_codegen
from symmetrix.execution_mh1_contract import make_execution_mh1_contract
from symmetrix.mh1_jit_codegen import (
    render_execution_mh1_conditioner_cpp_helpers,
    render_jit_mh1_cuda_plugin,
    render_jit_mh1_host_plugin,
    resolve_execution_mh1_cuda_edge_policy,
    resolve_execution_mh1_cuda_forward_policy,
    resolve_execution_mh1_cuda_source_policy,
    execution_mh1_conditioner_layout_metadata,
    jit_mh1_cuda_plugin_metadata,
)


def _tensor(shape, fill=0.0):
    size = 1
    for dimension in shape:
        size *= dimension
    return {"shape": list(shape), "values": [fill] * size}


def _linear(irreps_out):
    return {
        "irreps_in": "2x0e",
        "irreps_out": irreps_out,
        "instructions": [],
        "weight": _tensor([1], 0.25),
        "bias": _tensor([0]),
        "output_mask": _tensor([1], 1.0),
    }


def _density_mlp(prefix_dimension):
    return {
        "layers": [
            {
                "type": "linear",
                "weight": _tensor([prefix_dimension, 9], 0.1),
                "bias": _tensor([prefix_dimension], 0.2),
            },
            {"type": "silu"},
            {
                "type": "linear",
                "weight": _tensor([1, prefix_dimension], 0.3),
                "bias": _tensor([1], 0.4),
            },
        ]
    }


def _layer_norm(dimension, eps=1.0e-5):
    return {
        "type": "layer_norm",
        "normalized_shape": [dimension],
        "eps": eps,
        "weight": _tensor([dimension], 1.0),
        "bias": _tensor([dimension], 0.0),
    }


def _interaction(*, channels=2, phi_dimension=3):
    return {
        "class": "RealAgnosticResidualNonLinearInteractionBlock",
        "source_embedding": _linear("3x0e"),
        "target_embedding": _linear("4x0e"),
        "conv_tp": {
            "irreps_in1": f"{channels}x0e",
            "irreps_in2": "1x0e",
            "irreps_out": f"{channels}x0e",
            "instructions": [
                {
                    "i_in1": 0,
                    "i_in2": 0,
                    "i_out": 0,
                    "connection_mode": "uvu",
                    "has_weight": True,
                    "path_weight": 1.0,
                    "path_shape": [channels, 1],
                    "wigner_3j": _tensor([1, 1, 1], 1.0),
                }
            ],
            "weight": _tensor([0]),
            "output_mask": _tensor([channels], 1.0),
        },
        "conv_tp_weights": {
            "layers": [
                {
                    "type": "linear",
                    "weight": _tensor([phi_dimension, 9], 0.1),
                    "bias": _tensor([phi_dimension], 0.2),
                },
                {"type": "silu"},
                {
                    "type": "linear",
                    "weight": _tensor([channels, phi_dimension], 0.3),
                    "bias": _tensor([channels], 0.4),
                },
            ]
        },
        "density_fn": _density_mlp(phi_dimension),
    }


def _definition(*, channels=2, phi_dimension=3):
    return {
        "model_type": "MACE_Nonlinear",
        "radial_embedding": {
            "apply_cutoff": False,
            "basis": {"type": "bessel", "weights": _tensor([2], 0.5)},
            "cutoff": {"type": "polynomial", "r_max": 6.0, "p": 5},
            "distance_transform": {
                "type": "agnesi",
                "a": 1.1,
                "q": 0.9,
                "p": 4.5,
            },
        },
        "interactions": [
            _interaction(channels=channels, phi_dimension=phi_dimension),
            _interaction(channels=channels, phi_dimension=phi_dimension),
        ],
    }


def _grouped_interaction(*, channels=2, phi_dimension=3):
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    instructions = [
        {
            "i_in1": 0,
            "i_in2": 0,
            "i_out": 0,
            "connection_mode": "uvu",
            "has_weight": True,
            "path_weight": 1.0,
            "path_shape": [channels, 1],
            "wigner_3j": _tensor([1, 1, 1], 1.0),
        },
        {
            "i_in1": 1,
            "i_in2": 0,
            "i_out": 1,
            "connection_mode": "uvu",
            "has_weight": True,
            "path_weight": 1.0,
            "path_shape": [channels, 1],
            "wigner_3j": {"shape": [3, 1, 3], "values": identity},
        },
        {
            "i_in1": 0,
            "i_in2": 1,
            "i_out": 1,
            "connection_mode": "uvu",
            "has_weight": True,
            "path_weight": 1.0,
            "path_shape": [channels, 1],
            "wigner_3j": {"shape": [1, 3, 3], "values": identity},
        },
    ]
    return {
        "class": "RealAgnosticResidualNonLinearInteractionBlock",
        "source_embedding": _linear("3x0e"),
        "target_embedding": _linear("4x0e"),
        "conv_tp": {
            "irreps_in1": f"{channels}x0e+{channels}x1o",
            "irreps_in2": "1x0e+1x1o",
            "irreps_out": f"{channels}x0e+{channels}x1o",
            "instructions": instructions,
            "weight": _tensor([0]),
            "output_mask": _tensor([4 * channels], 1.0),
        },
        "conv_tp_weights": {
            "layers": [
                {
                    "type": "linear",
                    "weight": _tensor([phi_dimension, 9], 0.1),
                    "bias": _tensor([phi_dimension], 0.2),
                },
                {"type": "silu"},
                {
                    "type": "linear",
                    "weight": _tensor([3 * channels, phi_dimension], 0.3),
                    "bias": _tensor([3 * channels], 0.4),
                },
            ]
        },
        "density_fn": _density_mlp(phi_dimension),
    }


def _grouped_definition(*, channels=2, phi_dimension=3):
    definition = _definition(channels=channels, phi_dimension=phi_dimension)
    definition["interactions"] = [
        _grouped_interaction(channels=channels, phi_dimension=phi_dimension),
        _grouped_interaction(channels=channels, phi_dimension=phi_dimension),
    ]
    return definition


def _conditioner_definition():
    definition = _definition(channels=2, phi_dimension=4)
    for interaction in definition["interactions"]:
        interaction["conv_tp_weights"] = {
            "layers": [
                {
                    "type": "linear",
                    "weight": _tensor([4, 9], 0.1),
                    "bias": _tensor([4], 0.2),
                },
                _layer_norm(4, 2.0e-5),
                {"type": "silu"},
                _layer_norm(4, 3.0e-5),
                {
                    "type": "linear",
                    "weight": _tensor([2, 4], 0.3),
                    "bias": _tensor([2], 0.4),
                },
            ]
        }
        interaction["density_fn"] = {
            "layers": [
                {
                    "type": "linear",
                    "weight": _tensor([3, 9], 0.15),
                    "bias": _tensor([3], 0.25),
                },
                _layer_norm(3, 4.0e-5),
                {"type": "silu"},
                {
                    "type": "linear",
                    "weight": _tensor([1, 3], 0.35),
                    "bias": _tensor([1], 0.45),
                },
            ]
        }
    return definition


def _generated_function(source, name):
    begin = source.index(f"void {name}(")
    end = source.index("\n}\n", begin) + 2
    return source[begin:end]


def _reference_conditioner_forward(
    layout, parameters, radial, source_first_layer, target_first_layer
):
    values = list(radial)
    for layer in layout["layers"]:
        if layer["type"] == "linear":
            active_input_dimension = (
                layout["dynamic_input_dimension"]
                if layer["program_index"] == 0
                else layer["input_dimension"]
            )
            result = []
            for row in range(layer["output_dimension"]):
                value = parameters[layer["bias_offset"] + row]
                if layer["program_index"] == 0:
                    value += source_first_layer[row] + target_first_layer[row]
                for column in range(active_input_dimension):
                    value += (
                        parameters[
                            layer["weight_offset"]
                            + row * layer["input_dimension"]
                            + column
                        ]
                        * values[column]
                    )
                result.append(value)
            values = result
        elif layer["type"] == "layer_norm":
            mean = sum(values) / layer["dimension"]
            variance = sum((value - mean) ** 2 for value in values) / layer["dimension"]
            inverse_stddev = 1.0 / math.sqrt(variance + layer["eps"])
            values = [
                (value - mean)
                * inverse_stddev
                * parameters[layer["gamma_offset"] + column]
                + parameters[layer["beta_offset"] + column]
                for column, value in enumerate(values)
            ]
        else:
            values = [value / (1.0 + math.exp(-value)) for value in values]
    return values


@pytest.fixture(scope="module")
def contract():
    return make_execution_mh1_contract(_definition())


def test_typed_mh1_cuda_inputs_preserve_portability_boundaries(contract):
    program, target, schedule = mh1_codegen._mh1_cuda_program_target_schedule(
        contract,
        80,
        abi_version=3,
        forward_policy="auto",
        source_policy="auto",
        edge_policy="auto",
    )
    assert (
        program.contract["generation_fingerprint"] == contract["generation_fingerprint"]
    )
    assert (target.native_subgroup_width, target.logical_edge_width) == (32, 16)
    assert target.channel_tile_width == 32
    thread_counts = (
        schedule.forward_threads,
        schedule.source_threads,
        schedule.edge_threads,
    )
    assert thread_counts == (
        128,
        128,
        128,
    )
    assert (schedule.tile_nodes, schedule.tile_channels, schedule.tile_threads) == (
        8,
        32,
        256,
    )
    assert (
        program.canonical_json()
        == mh1_codegen.MH1Program.from_contract(
            {key: contract[key] for key in reversed(contract)}, abi_version=3
        ).canonical_json()
    )


def test_mh1_neutral_types_own_cuda_compatibility_aliases(contract):
    _, target, schedule = mh1_codegen._mh1_cuda_program_target_schedule(
        contract,
        80,
        abi_version=3,
        forward_policy="auto",
        source_policy="auto",
        edge_policy="auto",
    )

    assert type(schedule) is mh1_codegen.MH1KernelSchedule
    assert mh1_codegen.MH1CudaSchedule is mh1_codegen.MH1KernelSchedule
    assert mh1_codegen.MH1CudaKernelLaunch is mh1_codegen.MH1KernelLaunch
    assert mh1_codegen.MH1NodeCudaProgram is mh1_codegen.MH1NodeProgram
    assert schedule.schedule_id.startswith("sha256:")
    assert target.target_id.startswith("sha256:")
    assert mh1_codegen.MH1_CUDA_FORWARD_POLICY_TAG == mh1_codegen.MH1_FORWARD_POLICY_TAG
    assert mh1_codegen.MH1_CUDA_SOURCE_POLICY_TAG == mh1_codegen.MH1_SOURCE_POLICY_TAG
    assert mh1_codegen.MH1_CUDA_EDGE_POLICY_TAG == mh1_codegen.MH1_EDGE_POLICY_TAG
    assert mh1_codegen.MH1_CUDA_LAUNCH_PLAN_TAG == mh1_codegen.MH1_LAUNCH_PLAN_TAG
    assert schedule.node_state_policy == "full-retention-v1"
    recompute = replace(schedule, node_state_policy="recompute-v1")
    assert recompute.schedule_id != schedule.schedule_id
    reuse = replace(schedule, node_state_policy="reuse-adjoints-v1")
    assert reuse.schedule_id not in (schedule.schedule_id, recompute.schedule_id)
    hybrid = replace(schedule, node_state_policy="retain-interaction-v1")
    assert hybrid.schedule_id not in (
        schedule.schedule_id,
        recompute.schedule_id,
        reuse.schedule_id,
    )
    with pytest.raises(ValueError, match="node-state policy"):
        replace(schedule, node_state_policy="invalid")


def test_cuda_module_reverse_schedule_groups_all_retained_node_linears():
    assert mh1_codegen._cuda_module_node_reverse_schedule("full-retention-v1") == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert mh1_codegen._cuda_module_node_reverse_schedule("recompute-v1") == (
        mh1_codegen._MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE
    )
    assert mh1_codegen._cuda_module_node_reverse_schedule("reuse-adjoints-v1") == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert (
        mh1_codegen._cuda_module_node_reverse_schedule("retain-interaction-v1")
        == mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )


def test_hip_module_reverse_schedule_groups_all_retained_node_linears():
    assert mh1_codegen._hip_module_node_reverse_schedule("full-retention-v1") == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert mh1_codegen._hip_module_node_reverse_schedule("recompute-v1") == (
        "tiled-8x32-message-linear2-32x32-v7"
    )
    assert mh1_codegen._hip_module_node_reverse_schedule("reuse-adjoints-v1") == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert (
        mh1_codegen._hip_module_node_reverse_schedule("retain-interaction-v1")
        == "tiled-8x32-message-linear2-32x32-v7"
    )


def test_grouped_tiled_reverse_rejects_unsupported_node_tiles():
    with pytest.raises(ValueError, match="requires 32, 64, or 128 node tiles"):
        mh1_codegen._cuda_grouped_tiled_linear_reverse(
            {},
            args_type="Args",
            output_adjoint_pointer="args.output_adjoint",
            output_stride=1,
            parameter_pointer="args.parameters",
            input_adjoint_pointer="args.input_adjoint",
            input_stride=1,
            stem="grouped_reverse",
            tile_nodes=16,
        )


def test_grouped_tiled_forward_rejects_unsupported_node_tiles():
    with pytest.raises(ValueError, match="requires 32, 64, or 128 node tiles"):
        mh1_codegen._cuda_grouped_tiled_linear_forward(
            {},
            args_type="Args",
            input_pointer="args.input",
            input_stride=1,
            parameter_pointer="args.parameters",
            output_pointer="args.output",
            output_stride=1,
            stem="grouped_forward",
            tile_nodes=16,
        )


def test_tiled_product_reverse_rejects_unsupported_node_tiles():
    with pytest.raises(ValueError, match="requires 1, 2, or 4 node tiles"):
        mh1_codegen._cuda_tiled_product_reverse(
            {},
            args_type="Args",
            input_pointer="args.input",
            input_stride=1,
            contracted_adjoint_pointer="args.contracted_adjoint",
            contracted_adjoint_stride=1,
            coefficient_pointer="args.coefficients",
            input_adjoint_pointer="args.input_adjoint",
            input_adjoint_stride=1,
            stem="product_reverse",
            tile_nodes=3,
        )


def test_mh1_gpu_target_identity_includes_backend_features_and_geometry():
    cuda = mh1_codegen.MH1GpuTarget(1, "cuda", "sm_120", 32, 16, 32)
    hip = mh1_codegen.MH1GpuTarget(
        1,
        "hip",
        "gfx1151",
        32,
        16,
        32,
        ("sramecc-", "xnack-"),
        "gfx1151",
        40,
    )

    assert cuda.target_id != hip.target_id
    assert json.loads(hip.canonical_json())["target_features"] == [
        "sramecc-",
        "xnack-",
    ]
    assert json.loads(hip.canonical_json())["compute_unit_count"] == 40


def test_mh1_module_and_launch_plan_do_not_parse_completed_device_source():
    module_renderer = inspect.getsource(mh1_codegen.render_execution_mh1_cuda_module_v4)
    plan_renderer = inspect.getsource(
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan
    )
    assert ".index(" not in module_renderer
    assert ".split(" not in module_renderer
    assert ".replace(" not in module_renderer
    assert "re.sub" not in module_renderer
    assert "_CUDA_PLUGIN_ABI_HEADER" not in module_renderer
    shared_renderer = inspect.getsource(mh1_codegen._render_execution_mh1_gpu_module_v4)
    assert "_render_mh1_cuda_device_abi" in shared_renderer
    artifact_renderer = inspect.getsource(
        mh1_codegen._render_execution_mh1_cuda_artifact
    )
    assert "_dialect.device_prelude()" in artifact_renderer
    assert "re." not in plan_renderer
    assert "_node_module_launches" not in plan_renderer


def _nvcc():
    for candidate in (
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.3/bin/nvcc",
    ):
        if candidate is not None and Path(candidate).is_file():
            with tempfile.TemporaryDirectory(
                prefix="symmetrix-nvcc-probe-"
            ) as directory:
                source = Path(directory) / "probe.cu"
                output = Path(directory) / "probe.o"
                source.write_text(
                    "#include <cuda_runtime.h>\n"
                    "__global__ void probe(float* value) { value[0] = sqrtf(value[0]); }\n",
                    encoding="ascii",
                )
                probe = subprocess.run(
                    [candidate, "-std=c++20", "-c", str(source), "-o", str(output)],
                    capture_output=True,
                    text=True,
                )
            if probe.returncode == 0:
                return candidate
            if "exception specification is incompatible" in probe.stderr and (
                'function "rsqrt"' in probe.stderr
                or 'function "rsqrtf"' in probe.stderr
            ):
                pytest.skip(
                    "installed CUDA toolkit is incompatible with system glibc rsqrt declarations"
                )
            pytest.fail(f"nvcc qualification probe failed:\n{probe.stderr}")
    pytest.skip("nvcc is required for MH-1 CUDA codegen coverage")


def test_conditioner_layout_is_deterministic_and_excludes_final_uvu_affine():
    contract = make_execution_mh1_contract(_conditioner_definition())
    first = execution_mh1_conditioner_layout_metadata(contract)
    second = execution_mh1_conditioner_layout_metadata(contract)

    assert first == second
    assert first["tag"] == "symmetrix.execution.mh1.conditioner-layout/1"
    assert first["array_scope"] == "interaction,network"
    assert first["network_order"] == ["conditioned_prefix", "density"]
    assert first["linear_parameter_order"] == ["weight_row_major", "bias"]
    assert first["layer_norm_parameter_order"] == ["gamma", "beta"]
    assert [entry["index"] for entry in first["interactions"]] == [0, 1]

    conditioned = first["interactions"][0]["conditioned_prefix"]
    assert conditioned["layer_count"] == 4
    assert conditioned["dynamic_input_dimension"] == 2
    assert conditioned["conditioned_input_dimension"] == 9
    assert conditioned["first_layer_contribution_dimension"] == 4
    assert conditioned["output_dimension"] == 4
    # 4x9 weight + 4 bias, two 4-wide LayerNorm affine pairs. The
    # final 2x4+2 UVU affine is deliberately absent.
    assert conditioned["parameter_count"] == 56
    assert conditioned["consumed_parameter_count"] == 56
    assert conditioned["runtime_parameter_count"] == 66
    assert conditioned["excluded_parameter_count"] == 10
    assert conditioned["layers"] == [
        {
            "program_index": 0,
            "model_layer_index": 0,
            "type": "linear",
            "parameter_offset": 0,
            "input_dimension": 9,
            "output_dimension": 4,
            "weight_offset": 0,
            "weight_count": 36,
            "bias_offset": 36,
            "bias_count": 4,
            "parameter_count": 40,
        },
        {
            "program_index": 1,
            "model_layer_index": 1,
            "type": "layer_norm",
            "parameter_offset": 40,
            "input_dimension": 4,
            "output_dimension": 4,
            "dimension": 4,
            "eps": 2.0e-5,
            "gamma_offset": 40,
            "gamma_count": 4,
            "beta_offset": 44,
            "beta_count": 4,
            "parameter_count": 8,
        },
        {
            "program_index": 2,
            "model_layer_index": 2,
            "type": "silu",
            "parameter_offset": 48,
            "input_dimension": 4,
            "output_dimension": 4,
            "dimension": 4,
            "parameter_count": 0,
        },
        {
            "program_index": 3,
            "model_layer_index": 3,
            "type": "layer_norm",
            "parameter_offset": 48,
            "input_dimension": 4,
            "output_dimension": 4,
            "dimension": 4,
            "eps": 3.0e-5,
            "gamma_offset": 48,
            "gamma_count": 4,
            "beta_offset": 52,
            "beta_count": 4,
            "parameter_count": 8,
        },
    ]

    density = first["interactions"][0]["density"]
    assert density["layer_count"] == 4
    assert density["output_dimension"] == 1
    assert density["parameter_count"] == 40
    assert density["consumed_parameter_count"] == 40
    assert density["runtime_parameter_count"] == 40
    assert density["excluded_parameter_count"] == 0
    assert density["layers"][-1]["weight_offset"] == 36
    assert density["layers"][-1]["bias_offset"] == 39

    source = render_execution_mh1_conditioner_cpp_helpers(contract)
    assert source.count("execution_mh1_conditioned_prefix_forward_") == 2
    assert source.count("execution_mh1_conditioned_prefix_radial_reverse_") == 2
    assert source.count("execution_mh1_density_forward_") == 2
    assert source.count("execution_mh1_density_radial_reverse_") == 2
    assert "row * 9 + column" in source
    assert "source_first_layer[row] + target_first_layer[row]" in source
    assert "float mean_1 = 0.0f;" in source
    assert "float mean_3 = 0.0f;" in source
    assert "parameter_adjoint" not in source
    assert "source_adjoint" not in source
    assert "target_adjoint" not in source


def test_conditioner_identity_prefix_has_no_runtime_parameters():
    definition = _definition(channels=2, phi_dimension=2)
    for interaction in definition["interactions"]:
        interaction["conv_tp_weights"] = {
            "layers": [
                {
                    "type": "linear",
                    "weight": _tensor([2, 9], 0.1),
                    "bias": _tensor([2], 0.2),
                }
            ]
        }
    contract = make_execution_mh1_contract(definition)
    metadata = execution_mh1_conditioner_layout_metadata(contract)
    prefix = metadata["interactions"][0]["conditioned_prefix"]

    assert prefix["layer_count"] == 0
    assert prefix["parameter_count"] == 0
    assert prefix["runtime_parameter_count"] == 20
    assert prefix["excluded_parameter_count"] == 20
    assert prefix["first_layer_contribution_dimension"] == 0
    assert prefix["output_dimension"] == 2
    source = render_execution_mh1_conditioner_cpp_helpers(
        contract, function_qualifier="inline"
    )
    function = _generated_function(source, "execution_mh1_conditioned_prefix_forward_0")
    assert "output[column] = radial[column];" in function
    assert "(void)parameters;" in function


def test_generated_conditioner_helpers_match_python_forward_and_reverse(
    tmp_path,
):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for conditioner codegen coverage")

    contract = make_execution_mh1_contract(_conditioner_definition())
    metadata = execution_mh1_conditioner_layout_metadata(contract)
    helpers = render_execution_mh1_conditioner_cpp_helpers(
        contract, function_qualifier="inline"
    )
    source_path = tmp_path / "conditioner.cpp"
    library_path = tmp_path / "conditioner.so"
    source_path.write_text(
        f"""#include <cmath>

{helpers}

extern "C" void conditioned_forward(
    const float* radial, const float* source, const float* target,
    const float* parameters, float* output)
{{
    execution_mh1_conditioned_prefix_forward_0(
        radial, source, target, parameters, output);
}}

extern "C" void conditioned_reverse(
    const float* radial, const float* source, const float* target,
    const float* parameters, const float* output_adjoint,
    float* radial_adjoint)
{{
    execution_mh1_conditioned_prefix_radial_reverse_0(
        radial, source, target, parameters, output_adjoint, radial_adjoint);
}}

extern "C" void density_forward(
    const float* radial, const float* source, const float* target,
    const float* parameters, float* output)
{{
    execution_mh1_density_forward_0(radial, source, target, parameters, output);
}}

extern "C" void density_reverse(
    const float* radial, const float* source, const float* target,
    const float* parameters, const float* output_adjoint,
    float* radial_adjoint)
{{
    execution_mh1_density_radial_reverse_0(
        radial, source, target, parameters, output_adjoint, radial_adjoint);
}}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            str(source_path),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.POINTER(ctypes.c_float)
    for name in (
        "conditioned_forward",
        "density_forward",
    ):
        function = getattr(library, name)
        function.argtypes = [pointer, pointer, pointer, pointer, pointer]
        function.restype = None
    for name in (
        "conditioned_reverse",
        "density_reverse",
    ):
        function = getattr(library, name)
        function.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        function.restype = None

    radial_values = [0.35, -0.22]
    radial = (ctypes.c_float * 2)(*radial_values)
    for network, output_adjoint in (
        ("conditioned", [0.6, -0.4, 0.25, 0.1]),
        ("density", [-0.7]),
    ):
        layout = metadata["interactions"][0][
            "conditioned_prefix" if network == "conditioned" else "density"
        ]
        parameter_values = [
            ((index * 7) % 19 - 9) * 0.047 for index in range(layout["parameter_count"])
        ]
        for layer in layout["layers"]:
            if layer["type"] == "layer_norm":
                for column in range(layer["dimension"]):
                    parameter_values[layer["gamma_offset"] + column] = (
                        0.75 + 0.08 * column
                    )
        contribution_dimension = layout["first_layer_contribution_dimension"]
        source_values = [0.03 * (index + 1) for index in range(contribution_dimension)]
        target_values = [
            -0.025 * (index + 2) for index in range(contribution_dimension)
        ]
        parameters = (ctypes.c_float * len(parameter_values))(*parameter_values)
        source = (ctypes.c_float * contribution_dimension)(*source_values)
        target = (ctypes.c_float * contribution_dimension)(*target_values)
        output = (ctypes.c_float * layout["output_dimension"])()
        getattr(library, f"{network}_forward")(
            radial, source, target, parameters, output
        )
        expected_output = _reference_conditioner_forward(
            layout,
            parameter_values,
            radial_values,
            source_values,
            target_values,
        )
        assert list(output) == pytest.approx(expected_output, abs=2.0e-6)

        output_adjoint_array = (ctypes.c_float * len(output_adjoint))(*output_adjoint)
        radial_adjoint = (ctypes.c_float * 2)()
        getattr(library, f"{network}_reverse")(
            radial,
            source,
            target,
            parameters,
            output_adjoint_array,
            radial_adjoint,
        )
        epsilon = 1.0e-4
        expected_adjoint = []
        for column in range(2):
            plus = list(radial_values)
            minus = list(radial_values)
            plus[column] += epsilon
            minus[column] -= epsilon
            plus_output = _reference_conditioner_forward(
                layout, parameter_values, plus, source_values, target_values
            )
            minus_output = _reference_conditioner_forward(
                layout, parameter_values, minus, source_values, target_values
            )
            expected_adjoint.append(
                sum(
                    adjoint * (plus_value - minus_value) / (2.0 * epsilon)
                    for adjoint, plus_value, minus_value in zip(
                        output_adjoint,
                        plus_output,
                        minus_output,
                        strict=True,
                    )
                )
            )
        assert list(radial_adjoint) == pytest.approx(
            expected_adjoint, abs=3.0e-4, rel=3.0e-4
        )


def test_generated_conditioner_device_helpers_compile(tmp_path):
    contract = make_execution_mh1_contract(_conditioner_definition())
    source_path = tmp_path / "conditioner.cu"
    object_path = tmp_path / "conditioner.o"
    source_path.write_text(
        "#include <cuda_runtime.h>\n\n"
        + render_execution_mh1_conditioner_cpp_helpers(contract),
        encoding="ascii",
    )
    subprocess.run(
        [
            _nvcc(),
            "-std=c++20",
            "-O2",
            "-gencode=arch=compute_80,code=sm_80",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cuda_metadata_and_source_cover_both_interactions(contract):
    metadata = jit_mh1_cuda_plugin_metadata(contract, 80)
    source = render_jit_mh1_cuda_plugin(contract, 80)
    split_source = render_jit_mh1_cuda_plugin(contract, 80, edge_policy="split")
    host_source = render_jit_mh1_host_plugin(contract)

    assert metadata["abi"] == "symmetrix.jit.mh1.cuda-plugin/3"
    assert metadata["abi_version"] == 3
    assert metadata["target_compute_capability"] == 80
    assert metadata["forward_policy"]["tag"].endswith("cuda-forward-policy/1")
    assert metadata["forward_policy_id"].startswith("sha256:")
    assert metadata["source_policy"]["tag"].endswith("cuda-source-policy/1")
    assert metadata["source_policy_id"].startswith("sha256:")
    assert metadata["edge_policy"]["tag"].endswith("cuda-edge-policy/2")
    assert metadata["edge_policy_id"].startswith("sha256:")
    assert metadata["edge_reverse_schedule"] == "hybrid-path-tiled-v7"
    assert (
        metadata["edge_policy"]["reverse_schedule"] == metadata["edge_reverse_schedule"]
    )
    assert all(
        interaction["conditioned_prefix"] == "generic-full-state-v1"
        and interaction["density"] == "generic-full-state-v1"
        for interaction in metadata["conditioner_reverse_schedule"]["interactions"]
    )
    assert metadata["conditioner_reverse_schedule_id"].startswith("sha256:")
    assert metadata["conditioner_layout"]["tag"].endswith("conditioner-layout/1")
    assert len(metadata["interactions"]) == 2
    assert [item["multiplicity"] for item in metadata["interactions"]] == [2, 2]
    assert (
        source.count("__global__ void forward_kernel_")
        + source.count("__global__ void __launch_bounds__(128, 4) forward_kernel_")
        == 2
    )
    assert source.count("__global__ void source_reverse_kernel_") == 2
    assert source.count("__global__ void edge_reverse_compact_fused_kernel_") == 2
    assert "work_items / threads_per_block" in source
    assert "work_items + threads_per_block - 1" not in source
    assert "__global__ void edge_reverse_phi_kernel_" not in source
    assert "__global__ void edge_reverse_harmonic_kernel_" not in source
    assert source.count("constexpr std::int64_t edge_phi_edge_batch =\n        4;") == 2
    assert source.count("args->samples / edge_phi_edge_batch") == 2
    assert source.count("__global__ void conditioning_forward_kernel_") == 2
    assert source.count("__global__ void conditioning_reverse_kernel_") == 2
    assert source.count("std::int32_t conditioning_forward_launch_") == 2
    assert source.count("std::int32_t conditioning_reverse_launch_") == 2
    assert host_source.count("static inline void conditioning_forward_") == 2
    for index in range(2):
        assert f"static inline void conditioning_reverse_{index}(" in host_source
    assert (
        source.count(
            "static_cast<std::int64_t>(args.active_receiver_count)\n"
            "            * partition_count * channel_tile_count;"
        )
        == 2
    )
    assert (
        source.count(
            "args.source_owner_count * source_partition_count * channel_tile_count;"
        )
        == 2
    )
    assert source.count("partition_owner % partition_count") == 2
    assert source.count("partition_owner / partition_count") == 2
    assert source.count("source_partition_owner % source_partition_count") == 2
    assert source.count("source_partition_owner / source_partition_count") == 2
    assert "args->output_mask[" not in source
    assert "args->output_mask[" not in split_source
    assert "args->output_mask[" not in host_source
    assert source.count("channel_tile * channel_tile_width + lane") == 4
    assert source.count("threadIdx.x & (channel_tile_width - 1)") == 4
    assert source.count("args.active_receivers[receiver_index]") == 4
    assert source.count("__shared__ float density_reduction[128];") == 2
    assert source.count("receiver_index = blockIdx.x;") == 2
    assert source.count("density_sum += conditioning_forward_edge_") == 2
    assert source.count("args->active_receiver_count > args->num_nodes") == 4
    assert source.count("args->active_receiver_count > args->samples") == 2
    assert "const std::int32_t partition = owner % partition_count" not in source
    assert "for (std::int32_t channel = 0;" not in source
    assert split_source.count("channel = edge_lane;") == 4
    assert split_source.count("channel < 2; channel += 16") == 4
    assert split_source.count("SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(") == 7
    assert "#define SYMMETRIX_JIT_MH1_SHUFFLE_DOWN" in source
    assert "__shfl_down_sync(0xffffffffu" in source
    assert source.count("0xffffffffu") == 1
    assert split_source.count("edge_lane == 0 && edge_active") == 4
    assert split_source.count("edge_base < args.samples;") == 4
    assert split_source.count("local_edge = edge_base + edge_owner;") == 4
    assert split_source.count("edge_active = local_edge < args.samples;") == 4
    assert split_source.count("warps_per_block = blockDim.x / 32;") == 4
    assert source.count("args->samples * 16") == 2
    assert (
        source.count("static_assert(edge_threads >= 32 && edge_threads % 32 == 0,") == 2
    )
    # CUDA consumes a phi-major/channel-contiguous affine matrix. Host
    # generation remains weight-major for its original ABI contract.
    assert "phi * 2 + (0 + channel)" in source
    assert "(0 + channel) * 3 + phi" not in source
    assert "(0 + channel) * 3 + phi" in host_source
    assert host_source.count("float phi_adjoint[3] = {0.0f};") == 2
    assert host_source.count("float harmonic_adjoint[1] = {0.0f};") == 2
    assert host_source.count("float source_adjoint[2] = {0.0f};") == 2
    host_source_reverse = _generated_function(host_source, "source_reverse_0")
    assert host_source_reverse.index(
        "for (std::int32_t scheduled = first_scheduled;"
    ) < host_source_reverse.index("for (std::int32_t channel = 0;")
    assert "phi_adjoint[phi] += weight_adjoint * cutoff" in host_source
    assert (
        "args->edge_phi_adjoint[local_edge * 3 + phi] =\n"
        "            phi_adjoint[phi];" in host_source
    )
    assert "args->edge_phi_adjoint[local_edge * 3 + phi] +=" not in host_source
    assert "harmonic_adjoint[0] +=" in host_source
    assert (
        "local_edge * 1 + harmonic] =\n"
        "                harmonic_adjoint[harmonic];" in host_source
    )
    assert "args->edge_input_2_adjoint[local_edge * 1 + 0] +=" not in host_source
    assert "SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V3" in source
    assert "SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V3" in source
    assert "SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V3" in source
    assert "SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V3" in source
    assert "SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V3" in source
    assert "SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V3" in source
    assert "SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V3" in source
    assert "SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V3" in host_source
    assert "symmetrix_jit_mh1_cuda_plugin_query_v3" in source
    assert "symmetrix_jit_mh1_cuda_plugin_query_v1" not in source
    assert "source_local_edge_indices" not in source
    assert "Kokkos" not in source
    assert "cudaDeviceSynchronize" not in source
    assert "cudaStreamSynchronize" not in source


def test_partial_output_coverage_retains_runtime_mask_load():
    grouped_contract = make_execution_mh1_contract(_grouped_definition())
    interaction = json.loads(json.dumps(grouped_contract["interactions"][0]))
    omitted_offset = max(int(path["output"]["offset"]) for path in interaction["paths"])
    interaction["paths"] = [
        path
        for path in interaction["paths"]
        if int(path["output"]["offset"]) != omitted_offset
    ]

    assert not mh1_codegen._interaction_output_mask_is_identity(interaction)
    source = mh1_codegen._render_host_forward(
        interaction, channel_owner=True, phi_major_linear_weight=True
    )
    assert "args->output_mask[" in source


def test_cuda_interaction_one_source_reverse_pairs_path_weight_reductions():
    grouped_contract = make_execution_mh1_contract(_grouped_definition())
    cuda_source = render_jit_mh1_cuda_plugin(grouped_contract, 80)
    hip_source = mh1_codegen._render_execution_mh1_cuda_artifact(
        grouped_contract,
        80,
        _dialect=mh1_codegen.MH1_HIP_DIALECT,
        _edge_reverse_schedule=mh1_codegen._MH1_HIP_EDGE_REVERSE_SCHEDULE,
    )

    paired = _generated_function(cuda_source, "source_reverse_1_partition_0")
    cuda_untiled = _generated_function(cuda_source, "source_reverse_0_partition_0")
    hip_untiled = _generated_function(hip_source, "source_reverse_1_partition_0")

    assert paired.count("for (std::int32_t phi = 0; phi < 3; ++phi)") == 1
    assert "const float phi_value = args->edge_phi[" in paired
    assert "float weight_0 = has_bias" in paired
    assert "float weight_1 = has_bias" in paired
    assert cuda_untiled.count("for (std::int32_t phi = 0; phi < 3; ++phi)") == 2
    assert "const float phi_value = args->edge_phi[" not in cuda_untiled
    assert hip_untiled.count("for (std::int32_t phi = 0; phi < 3; ++phi)") == 2
    assert "const float phi_value = args->edge_phi[" not in hip_untiled


def test_cuda_forward_and_source_reverse_are_partitioned_by_irrep_block(
    monkeypatch,
):
    monkeypatch.setattr(mh1_codegen, "_CUDA_FORWARD_COMPONENT_BUDGET", 2)
    monkeypatch.setattr(mh1_codegen, "_CUDA_FORWARD_FUSE_COMPONENT_LIMIT", 0)
    grouped_contract = make_execution_mh1_contract(_grouped_definition())
    source = render_jit_mh1_cuda_plugin(grouped_contract, 80, source_policy="grouped")
    host_source = render_jit_mh1_host_plugin(grouped_contract)

    assert (
        source.count("__global__ void forward_kernel_")
        + source.count("__global__ void __launch_bounds__(128, 4) forward_kernel_")
        == 2
    )
    assert source.count("__global__ void source_reverse_kernel_") == 2
    assert source.count("void forward_0_partition_") == 2
    assert source.count("void forward_1_partition_") == 2
    assert source.count("void source_reverse_0_partition_") == 2
    assert source.count("void source_reverse_1_partition_") == 2
    assert source.count("constexpr std::int32_t partition_count = 2;") == 2
    assert source.count("constexpr std::int32_t source_partition_count = 2;") == 2
    assert (
        source.count(
            "static_cast<std::int64_t>(args.active_receiver_count)\n"
            "            * partition_count * channel_tile_count;"
        )
        == 2
    )
    assert (
        source.count(
            "args.source_owner_count * source_partition_count * channel_tile_count;"
        )
        == 2
    )
    assert (
        source.count(
            "static_cast<std::int64_t>(args->active_receiver_count)\n"
            "            * 2 * channel_tiles * 32"
        )
        == 2
    )
    assert (
        source.count(
            "args->source_owner_count * 2\n" "            * source_channel_tiles * 32"
        )
        == 2
    )
    assert source.count("partition_owner % partition_count") == 2
    assert source.count("partition_owner / partition_count") == 2
    assert source.count("source_partition_owner % source_partition_count") == 2
    assert source.count("source_partition_owner / source_partition_count") == 2
    assert source.count("channel_tile * channel_tile_width + lane") == 4
    assert source.count("threadIdx.x & (channel_tile_width - 1)") == 4
    assert source.count("args.active_receivers[receiver_index]") == 4
    assert source.count("__shared__ float density_reduction[128];") == 2
    assert source.count("receiver_index = blockIdx.x;") == 2
    assert "const std::int32_t partition = owner % partition_count" not in source
    assert source.count("source_reverse_kernel_0<<<") == 1
    assert source.count("source_reverse_kernel_1<<<") == 1
    assert source.count("case 0: source_reverse_0_partition_0(") == 1
    assert source.count("case 1: source_reverse_0_partition_1(") == 1
    assert "args->output_mask[" not in source

    forward_scalar = _generated_function(source, "forward_0_partition_0")
    forward_vector = _generated_function(source, "forward_0_partition_1")
    assert "float value_0 = 0.0f;" in forward_scalar
    assert "float value_1 = 0.0f;" not in forward_scalar
    assert "phi * 6 + (0 + channel)" in forward_scalar
    assert "phi * 6 + (2 + channel)" not in forward_scalar
    assert "phi * 6 + (4 + channel)" not in forward_scalar
    assert "float value_2 = 0.0f;" in forward_vector
    assert "float value_3 = 0.0f;" not in forward_vector
    assert "phi * 6 + (0 + channel)" not in forward_vector
    assert "phi * 6 + (2 + channel)" in forward_vector
    assert "phi * 6 + (4 + channel)" in forward_vector

    source_scalar = _generated_function(source, "source_reverse_0_partition_0")
    source_vector = _generated_function(source, "source_reverse_0_partition_1")
    assert "float value_0 = 0.0f;" in source_scalar
    assert "float value_1 = 0.0f;" not in source_scalar
    assert "phi * 6 + (0 + channel)" in source_scalar
    assert "phi * 6 + (2 + channel)" not in source_scalar
    assert "phi * 6 + (4 + channel)" in source_scalar
    assert "float value_2 = 0.0f;" in source_vector
    assert "float value_3 = 0.0f;" not in source_vector
    assert "phi * 6 + (0 + channel)" not in source_vector
    assert "phi * 6 + (2 + channel)" in source_vector
    assert "phi * 6 + (4 + channel)" not in source_vector

    assert "_group_" not in host_source
    assert host_source.count("static void forward_") == 2
    assert host_source.count("static void source_reverse_") == 2


def test_cuda_forward_batches_irrep_groups_under_register_budget():
    interaction = {
        "paths": [
            {"output": {"offset": 0, "components": 1}},
            {"output": {"offset": 128, "components": 1}},
            {"output": {"offset": 256, "components": 3}},
            {"output": {"offset": 640, "components": 3}},
            {"output": {"offset": 1024, "components": 3}},
            {"output": {"offset": 1408, "components": 5}},
            {"output": {"offset": 2048, "components": 5}},
            {"output": {"offset": 2688, "components": 5}},
            {"output": {"offset": 3328, "components": 7}},
            {"output": {"offset": 4224, "components": 7}},
        ]
    }

    selected = mh1_codegen._batch_interaction_paths(
        interaction,
        "output",
        mh1_codegen._CUDA_FORWARD_COMPONENT_BUDGET,
        mh1_codegen._CUDA_FORWARD_FUSE_COMPONENT_LIMIT,
    )
    assert [len(batch) for batch in selected] == [4, 2, 1, 1, 1, 1]

    batches = mh1_codegen._batch_interaction_paths(
        interaction, "output", component_budget=16
    )

    assert [len(batch) for batch in batches] == [6, 2, 2]
    assert [
        sum(
            block[1]
            for block in {
                (path["output"]["offset"], path["output"]["components"])
                for path in batch
            }
        )
        for batch in batches
    ] == [16, 10, 14]

    fused = mh1_codegen._batch_interaction_paths(
        {"paths": interaction["paths"][:6]},
        "output",
        component_budget=10,
        fuse_component_limit=16,
    )
    assert [len(batch) for batch in fused] == [6]


def test_cuda_forward_policy_is_canonical_and_validated(monkeypatch):
    monkeypatch.setattr(mh1_codegen, "_CUDA_FORWARD_COMPONENT_BUDGET", 2)
    monkeypatch.setattr(mh1_codegen, "_CUDA_FORWARD_SHARED_COMPONENT_BUDGET", 3)
    monkeypatch.setattr(mh1_codegen, "_CUDA_FORWARD_FUSE_COMPONENT_LIMIT", 0)
    grouped_contract = make_execution_mh1_contract(_grouped_definition())

    automatic = resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, "auto")
    budget = resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, "budget")
    shared = resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, "shared")
    full = resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, "full")

    assert automatic == budget
    assert [entry["component_counts"] for entry in budget["interactions"]] == [
        [1, 3],
        [1, 3],
    ]
    assert [entry["component_counts"] for entry in shared["interactions"]] == [
        [1, 3],
        [1, 3],
    ]
    assert [entry["component_counts"] for entry in full["interactions"]] == [
        [4],
        [4],
    ]
    assert budget == shared
    assert len({budget["policy_id"], full["policy_id"]}) == 2
    assert (
        jit_mh1_cuda_plugin_metadata(grouped_contract, 80, forward_policy=budget)[
            "forward_policy"
        ]
        == budget
    )

    budget_source = render_jit_mh1_cuda_plugin(
        grouped_contract, 80, forward_policy=budget
    )
    full_source = render_jit_mh1_cuda_plugin(grouped_contract, 80, forward_policy=full)
    assert budget_source.count("void forward_0_partition_") == 2
    assert budget_source.count("void forward_1_partition_") == 2
    assert full_source.count("void forward_0_partition_") == 1
    assert full_source.count("void forward_1_partition_") == 1
    assert (
        _generated_function(full_source, "forward_0_partition_0").count(
            "for (std::int64_t edge = begin; edge < end; ++edge)"
        )
        == 1
    )

    duplicate = json.loads(json.dumps(budget))
    duplicate["interactions"][0]["partitions"][1].append(
        duplicate["interactions"][0]["partitions"][0][0]
    )
    with pytest.raises(ValueError, match="repeats"):
        resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, duplicate)

    omitted = json.loads(json.dumps(budget))
    omitted["interactions"][0]["partitions"].pop()
    with pytest.raises(ValueError, match="omits"):
        resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, omitted)

    stale = json.loads(json.dumps(budget))
    stale["policy_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="stale"):
        resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, stale)

    missing_identity = json.loads(json.dumps(budget))
    missing_identity.pop("policy_id")
    with pytest.raises(ValueError, match="fields"):
        resolve_execution_mh1_cuda_forward_policy(
            grouped_contract, 80, missing_identity
        )

    unknown_field = json.loads(json.dumps(budget))
    unknown_field["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, unknown_field)

    stale_counts = json.loads(json.dumps(budget))
    stale_counts["interactions"][0]["component_counts"][0] += 1
    with pytest.raises(ValueError, match="canonical strategy"):
        resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, stale_counts)

    relabeled = json.loads(json.dumps(budget))
    relabeled["interactions"][0]["strategy"] = "full"
    with pytest.raises(ValueError, match="canonical strategy"):
        resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, relabeled)

    with pytest.raises(ValueError, match="must be"):
        resolve_execution_mh1_cuda_forward_policy(grouped_contract, 80, "unknown")
    with pytest.raises(ValueError, match="target"):
        resolve_execution_mh1_cuda_forward_policy(grouped_contract, 90, budget)


def test_cuda_source_policy_fuses_small_input_blocks_and_is_validated():
    grouped_contract = make_execution_mh1_contract(_grouped_definition())

    automatic = resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, "auto")
    budget = resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, "budget")
    grouped = resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, "grouped")
    full = resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, "full")

    assert automatic == grouped
    assert budget == full
    assert [entry["component_counts"] for entry in full["interactions"]] == [
        [4],
        [4],
    ]
    assert [entry["component_counts"] for entry in grouped["interactions"]] == [
        [1, 3],
        [1, 3],
    ]
    assert full["policy_id"] != grouped["policy_id"]
    metadata = jit_mh1_cuda_plugin_metadata(grouped_contract, 80, source_policy=full)
    assert metadata["source_policy"] == full
    assert metadata["source_policy_id"] == full["policy_id"]

    full_source = render_jit_mh1_cuda_plugin(grouped_contract, 80, source_policy=full)
    grouped_source = render_jit_mh1_cuda_plugin(
        grouped_contract, 80, source_policy=grouped
    )
    assert full_source.count("void source_reverse_0_partition_") == 1
    assert full_source.count("void source_reverse_1_partition_") == 1
    assert grouped_source.count("void source_reverse_0_partition_") == 2
    assert grouped_source.count("void source_reverse_1_partition_") == 2
    fused = _generated_function(full_source, "source_reverse_0_partition_0")
    assert fused.count("for (std::int32_t scheduled = first_scheduled;") == 1
    assert all(f"float value_{index} = 0.0f;" in fused for index in range(4))

    duplicate = json.loads(json.dumps(grouped))
    duplicate["interactions"][0]["partitions"][1].append(
        duplicate["interactions"][0]["partitions"][0][0]
    )
    with pytest.raises(ValueError, match="repeats"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, duplicate)

    omitted = json.loads(json.dumps(grouped))
    omitted["interactions"][0]["partitions"].pop()
    with pytest.raises(ValueError, match="omits"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, omitted)

    stale = json.loads(json.dumps(grouped))
    stale["policy_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="stale"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, stale)

    missing_identity = json.loads(json.dumps(grouped))
    missing_identity.pop("policy_id")
    with pytest.raises(ValueError, match="fields"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, missing_identity)

    stale_counts = json.loads(json.dumps(grouped))
    stale_counts["interactions"][0]["component_counts"][0] += 1
    with pytest.raises(ValueError, match="canonical strategy"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, stale_counts)

    relabeled = json.loads(json.dumps(grouped))
    relabeled["interactions"][1]["strategy"] = "full"
    with pytest.raises(ValueError, match="canonical strategy"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, relabeled)

    wrong_model = json.loads(json.dumps(grouped))
    wrong_model["generation_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="model identity"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, wrong_model)

    with pytest.raises(ValueError, match="must be"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 80, "unknown")
    with pytest.raises(ValueError, match="target"):
        resolve_execution_mh1_cuda_source_policy(grouped_contract, 90, grouped)


def test_cuda_edge_policy_emits_compact_split_fused_and_mixed_programs():
    grouped_contract = make_execution_mh1_contract(_grouped_definition())
    automatic = resolve_execution_mh1_cuda_edge_policy(grouped_contract, 80, "auto")
    compact = resolve_execution_mh1_cuda_edge_policy(
        grouped_contract, 80, "compact_fused"
    )
    split = resolve_execution_mh1_cuda_edge_policy(grouped_contract, 80, "split")
    fused = resolve_execution_mh1_cuda_edge_policy(grouped_contract, 80, "fused")

    assert automatic == compact
    assert [entry["physical_launch_count"] for entry in split["interactions"]] == [
        2,
        2,
    ]
    assert [entry["physical_launch_count"] for entry in fused["interactions"]] == [
        1,
        1,
    ]
    assert [entry["physical_launch_count"] for entry in compact["interactions"]] == [
        1,
        1,
    ]
    assert len({compact["policy_id"], split["policy_id"], fused["policy_id"]}) == 3

    stale_schedule = json.loads(json.dumps(compact))
    stale_schedule["reverse_schedule"] = "hybrid-path-tiled-v6"
    stale_payload = {
        key: value for key, value in stale_schedule.items() if key != "policy_id"
    }
    stale_schedule["policy_id"] = mh1_codegen._policy_digest(stale_payload)
    with pytest.raises(ValueError, match="reverse schedule is unsupported"):
        resolve_execution_mh1_cuda_edge_policy(grouped_contract, 80, stale_schedule)

    mixed = json.loads(json.dumps(split))
    mixed["interactions"][0] = {
        "index": 0,
        "strategy": "fused",
        "physical_launch_count": 1,
    }
    mixed_payload = {key: value for key, value in mixed.items() if key != "policy_id"}
    mixed["policy_id"] = mh1_codegen._policy_digest(mixed_payload)
    assert resolve_execution_mh1_cuda_edge_policy(grouped_contract, 80, mixed) == mixed

    split_source = render_jit_mh1_cuda_plugin(grouped_contract, 80, edge_policy=split)
    compact_source = render_jit_mh1_cuda_plugin(
        grouped_contract, 80, edge_policy=compact
    )
    fused_source = render_jit_mh1_cuda_plugin(grouped_contract, 80, edge_policy=fused)
    mixed_source = render_jit_mh1_cuda_plugin(grouped_contract, 80, edge_policy=mixed)
    assert split_source.count("__global__ void edge_reverse_phi_kernel_") == 2
    assert split_source.count("__global__ void edge_reverse_harmonic_kernel_") == 2
    assert "__global__ void edge_reverse_fused_kernel_" not in split_source
    assert (
        compact_source.count("__global__ void edge_reverse_compact_fused_kernel_") == 2
    )
    assert "float phi_adjoint[3] = {0.0f};" not in compact_source
    assert "float harmonic_adjoint[4] = {0.0f};" not in compact_source
    assert compact_source.count("__shared__ float phi_adjoint[4][3];") == 2
    assert compact_source.count("__shared__ float harmonic_adjoint[4][4];") == 2
    assert compact_source.count("__shared__ float path_weight_adjoint[4][4]") == 2
    for edge_slot in range(4):
        assert f"float cutoff_contribution_{edge_slot} = 0.0f;" in compact_source
        assert f"float weight_{edge_slot} = has_bias" in compact_source
        assert f"float weight_adjoint_{edge_slot} = 0.0f;" in compact_source
        assert f"harmonic_contribution_{edge_slot}_0" in compact_source
    assert "float cutoff_contribution = 0.0f;" not in compact_source
    for index in range(2):
        compact_kernel = _generated_function(
            compact_source, f"edge_reverse_compact_fused_kernel_{index}"
        )
        assert "float weight = has_bias" not in compact_kernel
    assert fused_source.count("__global__ void edge_reverse_fused_kernel_") == 2
    assert "__global__ void edge_reverse_phi_kernel_" not in fused_source
    assert "__global__ void edge_reverse_harmonic_kernel_" not in fused_source
    assert mixed_source.count("__global__ void edge_reverse_fused_kernel_0") == 1
    assert mixed_source.count("__global__ void edge_reverse_phi_kernel_1") == 1

    fused_function = _generated_function(fused_source, "edge_reverse_fused_0")
    assert "float phi_adjoint[3] = {0.0f};" in fused_function
    assert "float harmonic_adjoint[4] = {0.0f};" in fused_function
    assert "float cutoff_adjoint = 0.0f;" in fused_function
    assert fused_function.count("float weight_adjoint = 0.0f;") == 3
    assert "args->edge_phi_adjoint[" in fused_function
    assert "args->edge_input_2_adjoint[" in fused_function
    assert "args->edge_cutoff_scale_adjoint[local_edge]" in fused_function
    assert (
        jit_mh1_cuda_plugin_metadata(grouped_contract, 80, edge_policy=mixed)[
            "edge_policy"
        ]
        == mixed
    )

    stale_count = json.loads(json.dumps(split))
    stale_count["interactions"][0]["physical_launch_count"] = 1
    with pytest.raises(ValueError, match="launch count"):
        resolve_execution_mh1_cuda_edge_policy(grouped_contract, 80, stale_count)

    stale_identity = json.loads(json.dumps(split))
    stale_identity["policy_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="stale"):
        resolve_execution_mh1_cuda_edge_policy(grouped_contract, 80, stale_identity)

    with pytest.raises(ValueError, match="must be"):
        resolve_execution_mh1_cuda_edge_policy(grouped_contract, 80, "unknown")
    with pytest.raises(ValueError, match="target"):
        resolve_execution_mh1_cuda_edge_policy(grouped_contract, 90, split)


def test_cuda_subgroup_edge_reverse_generalizes_to_phi_64():
    wide_contract = make_execution_mh1_contract(
        _definition(channels=19, phi_dimension=64)
    )
    metadata = jit_mh1_cuda_plugin_metadata(wide_contract, 100)
    source = render_jit_mh1_cuda_plugin(wide_contract, 100)

    assert [item["multiplicity"] for item in metadata["interactions"]] == [
        19,
        19,
    ]
    assert [item["phi_dimension"] for item in metadata["interactions"]] == [
        64,
        64,
    ]
    assert "float phi_adjoint[64] = {0.0f};" not in source
    assert source.count("__shared__ float phi_adjoint[4][64];") == 2
    assert source.count("__shared__ float path_weight_adjoint[4][4]") == 2
    assert source.count("path_weight_adjoint[0][3][channel]") == 4
    assert source.count("channel < 19; channel += blockDim.x") == 2
    assert source.count("phi = warp; phi < 64;") == 2
    assert source.count("const float linear_weight = args->linear_weight[") == 4
    assert source.count("local_edge_3 = edge_base + static_cast<std::int64_t>(3);") == 2
    assert (
        "local_edge_3 = edge_base + static_cast<std::int64_t>(3) * gridDim.x"
        not in source
    )
    assert source.count("static_cast<std::int64_t>(blockIdx.x) * 4;") == 2
    assert source.count("edge_base += static_cast<std::int64_t>(gridDim.x) * 4") == 2
    assert source.count("constexpr std::int64_t edge_phi_edge_batch =\n        4;") == 2
    assert source.count("args->samples / edge_phi_edge_batch") == 2
    assert source.count("args->samples % edge_phi_edge_batch != 0 ? 1 : 0") == 2
    assert "phi * 19 + (0 + channel)" in source
    assert source.count("__shared__ float harmonic_adjoint[4][") == 2
    assert "float harmonic_adjoint[4] = {0.0f};" not in source
    assert source.count("float cutoff_contribution_0 = 0.0f;") == 2
    assert source.count("float cutoff_contribution_3 = 0.0f;") == 2
    assert source.count("float weight_0 = has_bias") == 3
    assert source.count("float weight_1 = has_bias") == 2
    assert source.count("float weight_3 = has_bias") == 2
    assert "float cutoff_contribution = 0.0f;" not in source
    assert "harmonic_contribution_3_0" in source


def test_compact_fused_batches_edge_slot_reductions_behind_one_barrier():
    contract = make_execution_mh1_contract(_grouped_definition())

    for interaction in contract["interactions"]:
        kernel = mh1_codegen._render_cuda_edge_reverse_compact_fused_path_tiled_kernel(
            interaction
        )
        path_count = len(interaction["paths"])
        path_batch_count = (path_count + 3) // 4

        assert kernel.count("__syncthreads();") == (
            2 + 2 * path_count + path_batch_count
        )
        barrier_sections = kernel.split("__syncthreads();")
        section = 1
        for batch_start in range(0, path_count, 4):
            for path in interaction["paths"][batch_start : batch_start + 4]:
                producer = barrier_sections[section]
                reduction = barrier_sections[section + 1]
                section += 2
                components = int(path["input_2"]["components"])
                assert producer.count("for (std::int32_t phi = 0;") == 1
                for edge_slot in range(4):
                    assert f"float weight_{edge_slot} = has_bias" in producer
                    assert f"float weight_adjoint_{edge_slot} = 0.0f;" in producer
                    harmonic_store = f"harmonic_warp_partial[{edge_slot}][warp]["
                    harmonic_read = (
                        f"harmonic_warp_partial[{edge_slot}][source_warp][owner]"
                    )
                    cutoff_store = f"cutoff_warp_partial[{edge_slot}][warp] ="
                    cutoff_read = f"cutoff_warp_partial[{edge_slot}][source_warp]"
                    assert producer.count(harmonic_store) == components
                    assert producer.count(cutoff_store) == 1
                    assert harmonic_read not in producer
                    assert cutoff_read not in producer
                    assert reduction.count(harmonic_read) == 1
                    assert reduction.count(cutoff_read) == 1
                    assert harmonic_store not in reduction
                    assert cutoff_store not in reduction
            section += 1
        assert section == len(barrier_sections) - 2
        assert "args->edge_phi_adjoint[" in barrier_sections[section]
        assert "warp_partial" not in barrier_sections[-1]


def test_compact_fused_uses_backend_specific_edge_weight_liveness():
    contract = make_execution_mh1_contract(_grouped_definition())
    interaction = contract["interactions"][0]
    cuda_kernel = mh1_codegen._render_cuda_edge_reverse_compact_fused_path_tiled_kernel(
        interaction, reverse_schedule="hybrid-path-tiled-v7"
    )
    hip_kernel = mh1_codegen._render_cuda_edge_reverse_compact_fused_path_tiled_kernel(
        interaction, reverse_schedule="hybrid-path-tiled-v6"
    )

    producer_loop = "for (std::int32_t channel = threadIdx.x;"
    assert hip_kernel.count(producer_loop) == 4 * cuda_kernel.count(producer_loop)
    assert hip_kernel.count("__syncthreads();") == cuda_kernel.count("__syncthreads();")


def test_mh1_persistent_launch_depth_is_backend_specific():
    repository = Path(__file__).resolve().parents[2]
    source = (
        repository / "libsymmetrix" / "source" / "mace_nonlinear_kokkos_impl.tpp"
    ).read_text(encoding="ascii")
    helper_start = source.index("int execution_mh1_persistent_blocks()")
    helper_end = source.index("\n}\n#endif", helper_start)
    helper = source[helper_start:helper_end]

    assert "#ifdef KOKKOS_ENABLE_HIP" in helper
    assert "blocks_per_compute_unit=8" in helper
    assert "blocks_per_compute_unit=4" in helper


def test_mh1_spline_reverse_wider_depth_is_backend_agnostic():
    repository = Path(__file__).resolve().parents[2]
    source = (
        repository / "libsymmetrix" / "source" / "mace_nonlinear_kokkos_impl.tpp"
    ).read_text(encoding="ascii")
    device_guard = "#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)"
    helper_start = source.index("int execution_mh1_spline_reverse_persistent_blocks()")
    helper_end = source.index("\n}\n#endif", helper_start)
    helper = source[helper_start:helper_end]
    helper_guard_start = source.rfind(device_guard, 0, helper_start)
    helper_guard_end = source.index("\n#endif", helper_end)
    reverse_start = source.index("if(generated_device_spline_reverse)")
    reverse_end = source.index("if(direct_device_spline_reverse)", reverse_start)
    device_reverse = source[reverse_start:reverse_end]

    assert helper_guard_start >= 0
    assert helper_start < helper_guard_end
    assert "blocks_per_compute_unit=8" in helper
    assert "blocks_per_compute_unit=4" not in helper
    assert "#ifdef KOKKOS_ENABLE_CUDA" not in helper
    assert "#ifdef KOKKOS_ENABLE_HIP" not in helper
    assert device_guard in device_reverse
    assert "execution_mh1_spline_reverse_persistent_blocks()" in device_reverse
    assert source.count("execution_mh1_spline_reverse_persistent_blocks()") == 2


def test_mh1_node_arena_policy_decouples_throughput_from_retained_state():
    repository = Path(__file__).resolve().parents[2]
    header = (
        repository / "libsymmetrix" / "source" / "mace_nonlinear_kokkos.hpp"
    ).read_text(encoding="ascii")
    source = (
        repository / "libsymmetrix" / "source" / "mace_nonlinear_kokkos_impl.tpp"
    ).read_text(encoding="ascii")
    loader_start = source.index(
        "void MaceNonlinearKokkosT<Precision>::load_jit_mh1_cuda_plugin_v4("
    )
    loader_end = source.index(
        "\nbool MaceNonlinearKokkosT<Precision>::jit_mh1_cuda_plugin_ready() const",
        loader_start,
    )
    loader = source[loader_start:loader_end]

    assert "static constexpr int capacity_tile_rows=256;" in header
    assert "static constexpr int cuda_throughput_tile_rows=1024;" in header
    assert 'execution_mh1_node_arena_policy_="throughput-v1";' in header
    selector_start = source.index(
        "int MaceNonlinearKokkosT<Precision>::"
        "selected_execution_mh1_node_arena_tile_rows()"
    )
    selector_end = source.index("\n}\n", selector_start)
    selector = source[selector_start:selector_end]
    assert 'execution_mh1_node_arena_policy_=="capacity-v1"' in selector
    assert "ExecutionMH1NodeRuntime::capacity_tile_rows" in selector
    assert "ExecutionMH1NodeRuntime::cuda_throughput_tile_rows" in selector
    assert "retained_pre_gate_dimension" not in selector
    assert "runtime.tile_rows=selected_execution_mh1_node_arena_tile_rows();" in loader
    assert "runtime.retained_pre_gate_dimension>0" not in loader


@pytest.mark.parametrize(
    ("samples", "expected_groups"), ((0, 0), (1, 1), (3, 1), (4, 1), (5, 2))
)
def test_path_tiled_consecutive_edge_groups_cover_each_tail_once(
    samples, expected_groups
):
    edge_batch = 4
    groups = samples // edge_batch + (samples % edge_batch != 0)

    assert groups == expected_groups
    for grid_dim in range(1, max(groups, 1) + 1):
        owned_edges = []
        for block in range(grid_dim):
            edge_base = block * edge_batch
            while edge_base < samples:
                owned_edges.extend(
                    edge_base + slot
                    for slot in range(edge_batch)
                    if edge_base + slot < samples
                )
                edge_base += grid_dim * edge_batch
        assert sorted(owned_edges) == list(range(samples))


@pytest.mark.parametrize(
    ("samples", "path_tiled_blocks", "subgroup_blocks"),
    ((0, 0, 0), (1, 1, 1), (3, 1, 2), (4, 1, 2), (5, 2, 2)),
)
def test_mh1_module_edge_phi_launch_geometry(
    samples, path_tiled_blocks, subgroup_blocks
):
    persistent_blocks = 2

    def blocks(edge_batch, block_multiplier):
        groups = samples // edge_batch + (samples % edge_batch != 0)
        return min(groups, persistent_blocks * block_multiplier)

    assert blocks(4, 4) == path_tiled_blocks
    assert blocks(1, 1) == subgroup_blocks

    repository = Path(__file__).resolve().parents[2]
    adapter = (
        repository / "libsymmetrix" / "source" / "jit_mh1_cuda_plugin.cpp"
    ).read_text(encoding="ascii")
    assert 'strategy=="compact_fused"||phi_schedule=="path_tiled"?4:1' in adapter
    assert "args->samples/plan.edge_phi_edge_batch" in adapter
    assert "args->samples%plan.edge_phi_edge_batch!=0?1:0" in adapter


@pytest.mark.parametrize("compute_capability", (True, 0, -1, 1000, 8.0, "80"))
def test_cuda_codegen_rejects_invalid_compute_capability(contract, compute_capability):
    with pytest.raises(ValueError, match="compute capability"):
        render_jit_mh1_cuda_plugin(contract, compute_capability)


class _Interaction(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("input_1_dimension", ctypes.c_int32),
        ("input_2_dimension", ctypes.c_int32),
        ("output_dimension", ctypes.c_int32),
        ("weight_size", ctypes.c_int32),
        ("phi_dimension", ctypes.c_int32),
        ("multiplicity", ctypes.c_int32),
        ("input_1_angular_dimension", ctypes.c_int32),
        ("instruction_count", ctypes.c_int32),
        ("forward_threads_per_block", ctypes.c_int32),
        ("source_threads_per_block", ctypes.c_int32),
        ("edge_threads_per_block", ctypes.c_int32),
        ("edge_reverse_physical_launch_count", ctypes.c_int32),
    ]


class _Launches(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("interaction", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("reserved_2", ctypes.c_uint32),
        ("forward_launch", ctypes.c_void_p),
        ("reverse_launch", ctypes.c_void_p),
    ]


class _ConditioningLaunches(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("interaction", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("reserved_2", ctypes.c_uint32),
        ("forward_launch", ctypes.c_void_p),
        ("reverse_launch", ctypes.c_void_p),
    ]


class _Plugin(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("pointer_size", ctypes.c_uint32),
        ("byte_order", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("interaction_count", ctypes.c_uint32),
        ("scalar_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("abi_tag", ctypes.c_char_p),
        ("artifact_id", ctypes.c_char_p),
        ("generation_fingerprint", ctypes.c_char_p),
        ("semantic_fingerprint", ctypes.c_char_p),
        ("structure_fingerprint", ctypes.c_char_p),
        ("target_compute_capability", ctypes.c_int32),
        ("reserved_2", ctypes.c_int32),
        ("interactions", _Interaction * 2),
        ("launches", _Launches * 2),
        ("conditioning_launches", _ConditioningLaunches * 2),
    ]


@pytest.mark.parametrize("edge_policy", ("compact_fused", "split", "fused"))
def test_generated_cuda_plugin_compiles_and_exports_descriptor(tmp_path, edge_policy):
    contract = make_execution_mh1_contract(_grouped_definition())
    metadata = jit_mh1_cuda_plugin_metadata(contract, 80, edge_policy=edge_policy)
    source_path = tmp_path / "mh1_plugin.cu"
    artifact_path = tmp_path / "mh1_plugin.so"
    source_path.write_text(
        render_jit_mh1_cuda_plugin(contract, 80, edge_policy=edge_policy),
        encoding="ascii",
    )
    subprocess.run(
        [
            _nvcc(),
            "-std=c++20",
            "-O2",
            "-shared",
            "--cudart=shared",
            "-Xcompiler=-fPIC",
            "-gencode=arch=compute_80,code=sm_80",
            str(source_path),
            "-o",
            str(artifact_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    library = ctypes.CDLL(str(artifact_path))
    with pytest.raises(AttributeError):
        getattr(library, "symmetrix_jit_mh1_cuda_plugin_query_v1")
    query = library.symmetrix_jit_mh1_cuda_plugin_query_v3
    query.restype = ctypes.POINTER(_Plugin)
    plugin = query().contents
    assert plugin.abi_version == 3
    assert plugin.struct_size == ctypes.sizeof(_Plugin)
    assert plugin.capabilities == 0x1FF
    assert plugin.interaction_count == 2
    assert plugin.scalar_size == ctypes.sizeof(ctypes.c_float)
    assert plugin.abi_tag.decode() == metadata["abi"]
    assert plugin.artifact_id.decode() == metadata["artifact_id"]
    assert plugin.target_compute_capability == 80
    assert [item.interaction for item in plugin.launches] == [0, 1]
    expected_edge_launches = 2 if edge_policy == "split" else 1
    assert [
        item.edge_reverse_physical_launch_count for item in plugin.interactions
    ] == [expected_edge_launches, expected_edge_launches]
    assert all(item.forward_launch for item in plugin.launches)
    assert all(item.reverse_launch for item in plugin.launches)
    assert [item.interaction for item in plugin.conditioning_launches] == [0, 1]
    assert all(item.forward_launch for item in plugin.conditioning_launches)
    assert all(item.reverse_launch for item in plugin.conditioning_launches)
