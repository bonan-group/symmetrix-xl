import ctypes
import gc
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from compact_r1_model import foundation_r1_test_model
from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix
from symmetrix.jit import (
    artifact_cache_inputs,
    jit_nvrtc_information,
    prepare_nvrtc_jit_artifact,
)
from symmetrix.mace_execution_contract import (
    make_generated_m1_contract,
    make_standard_m0_contract,
    make_standard_r0_contract,
)
from symmetrix.mace_jit_device_artifact import prepare_jit_device_artifact
from symmetrix.mace_jit_operator_codegen import (
    OPERATOR_ARTIFACT_ENVELOPE,
    render_jit_m0_device_module,
    render_jit_m1_device_module,
    render_jit_r0_device_module,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def _require_cuda_nvrtc():
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("the imported Symmetrix extension was built without Kokkos")
    if not native_symmetrix._kokkos_is_initialized():
        try:
            native_symmetrix._init_kokkos()
        except RuntimeError as error:
            pytest.skip(f"Kokkos Cuda could not be initialized: {error}")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("operator RTC execution requires Kokkos Cuda")
    information = jit_nvrtc_information()
    if not information.get("available", False):
        pytest.skip(f"NVRTC is unavailable: {information.get('reason')}")


def _launch_cuda_product_alias_case(artifact_path, metadata, precision):
    class PackedArgs(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32),
            ("num_nodes", ctypes.c_int64),
            ("channels", ctypes.c_int32),
            ("capture_input_scale_adjoint", ctypes.c_int32),
            ("node_types", ctypes.c_uint64),
            ("input", ctypes.c_uint64),
            ("weights", ctypes.c_uint64),
            ("output_adjoint", ctypes.c_uint64),
            ("output", ctypes.c_uint64),
            ("input_adjoint", ctypes.c_uint64),
            ("input_scale_adjoint", ctypes.c_uint64),
        ]

    class Args(ctypes.Structure):
        _fields_ = [
            ("packed", PackedArgs),
            ("input_node_stride", ctypes.c_int64),
            ("output_node_stride", ctypes.c_int64),
            ("output_adjoint_node_stride", ctypes.c_int64),
            ("input_adjoint_node_stride", ctypes.c_int64),
        ]

    assert ctypes.sizeof(PackedArgs) == 80
    assert ctypes.sizeof(Args) == 112
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]

    def check(status, operation):
        assert status == 0, f"{operation} failed with CUDA status {status}"

    allocations = []

    def upload(values):
        array = np.ascontiguousarray(values)
        pointer = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), array.nbytes), "cuMemAlloc")
        allocations.append(pointer.value)
        check(
            cuda.cuMemcpyHtoD_v2(
                pointer.value, ctypes.c_void_p(array.ctypes.data), array.nbytes
            ),
            "cuMemcpyHtoD",
        )
        return pointer.value, array

    def download(pointer, shape, dtype):
        result = np.empty(shape, dtype=dtype)
        check(
            cuda.cuMemcpyDtoH_v2(
                ctypes.c_void_p(result.ctypes.data), pointer, result.nbytes
            ),
            "cuMemcpyDtoH",
        )
        return result

    dtype = np.float32 if precision == "float32" else np.float64
    nodes = 2
    channels = metadata["contract"]["channels"]
    input_components = metadata["input_components"]
    output_components = metadata["output_components"]
    input_size = nodes * input_components * channels
    output_size = nodes * output_components * channels
    rng = np.random.default_rng(90210)
    host_input = rng.normal(size=input_size).astype(dtype)
    host_weights = rng.normal(size=metadata["term_count"] * channels).astype(dtype)
    host_output_adjoint = rng.normal(size=output_size).astype(dtype)
    module = ctypes.c_void_p()
    try:
        check(
            cuda.cuModuleLoad(ctypes.byref(module), str(artifact_path).encode()),
            "cuModuleLoad",
        )
        functions = {}
        for direction in ("forward", "reverse"):
            function = ctypes.c_void_p()
            check(
                cuda.cuModuleGetFunction(
                    ctypes.byref(function),
                    module,
                    f"symmetrix_{metadata['operator'].lower()}_{direction}_v1".encode(),
                ),
                "cuModuleGetFunction",
            )
            functions[direction] = function
        node_types, _ = upload(np.zeros(nodes, dtype=np.int32))
        weights, _ = upload(host_weights)
        output_adjoint, _ = upload(host_output_adjoint)

        def execute(direction, *, alias):
            input_pointer, _ = upload(host_input)
            output_pointer, _ = upload(np.zeros(output_size, dtype=dtype))
            input_adjoint_pointer, _ = upload(np.zeros(input_size, dtype=dtype))
            if direction == "forward" and alias:
                output_pointer = input_pointer
            if direction == "reverse" and alias:
                input_adjoint_pointer = input_pointer
            args = Args(
                PackedArgs(
                    ctypes.sizeof(PackedArgs),
                    0,
                    nodes,
                    channels,
                    0,
                    node_types,
                    input_pointer,
                    weights,
                    output_adjoint,
                    output_pointer,
                    input_adjoint_pointer,
                    0,
                ),
                input_components * channels,
                output_components * channels,
                output_components * channels,
                input_components * channels,
            )
            argument = ctypes.cast(ctypes.pointer(args), ctypes.c_void_p)
            parameters = (ctypes.c_void_p * 1)(argument)
            owners = nodes * channels
            blocks = (owners + metadata["threads_per_block"] - 1) // metadata[
                "threads_per_block"
            ]
            check(
                cuda.cuLaunchKernel(
                    functions[direction],
                    blocks,
                    1,
                    1,
                    metadata["threads_per_block"],
                    1,
                    1,
                    0,
                    None,
                    parameters,
                    None,
                ),
                "cuLaunchKernel",
            )
            check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
            pointer = (
                output_pointer if direction == "forward" else input_adjoint_pointer
            )
            size = output_size if direction == "forward" else input_size
            return download(pointer, size, dtype)

        for direction in ("forward", "reverse"):
            np.testing.assert_allclose(
                execute(direction, alias=True),
                execute(direction, alias=False),
                rtol=0.0,
                atol=0.0,
            )
    finally:
        for pointer in reversed(allocations):
            cuda.cuMemFree_v2(pointer)
        if module.value:
            cuda.cuModuleUnload(module)


@pytest.mark.parametrize("operator", ("M0", "M1"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_cuda_chunk32_product_metadata_is_disjoint_only(operator, precision):
    target = "sm_120"
    if operator == "M0":
        contract = make_standard_m0_contract(
            channels=33,
            type_count=1,
            input_l_max=1,
            output_l_max=1,
            correlation=2,
            monomials={
                0: [[0], [0, 0]],
                1: [[1], [0, 1]],
                2: [[2], [0, 2]],
                3: [[3], [0, 3]],
            },
        )
        render = render_jit_m0_device_module
    else:
        contract = make_generated_m1_contract(
            channels=33,
            type_count=1,
            input_l_max=1,
            correlation=3,
            monomials=[[0], [1, 1], [0, 2, 3]],
        )
        render = render_jit_m1_device_module
    source, metadata = render(
        contract, precision=precision, target=target, schedule="chunk32"
    )
    assert metadata["schedule_revision"] == (2 if operator == "M0" else 3)
    assert metadata["capabilities"] & (1 << 2) == 0
    assert f"symmetrix_operator_capabilities_v1 = {metadata['capabilities']}u" in source


def _custom_scalar_operator_model():
    contract_path = (
        REPOSITORY
        / "symmetrix/test/data/execution_contracts"
        / "jit_r1_fixture_c4_e3_l0_contract.json"
    )
    r1_contract = json.loads(contract_path.read_text())
    m0_contract = make_standard_m0_contract(
        channels=4,
        type_count=1,
        input_l_max=0,
        output_l_max=0,
        correlation=2,
        monomials={0: [[0], [0, 0]]},
    )
    r0_contract = make_standard_r0_contract(
        channels=4,
        radial_embedding=3,
        edge_l_max=0,
        source_irreps=r1_contract["irreps"]["source"],
        edge_irreps=r1_contract["irreps"]["edge"],
        output_irreps=r1_contract["irreps"]["output"],
        output_l=[0],
        edge_l=[0],
        source_l=[0],
        instructions=[
            {
                "connection_mode": "uvu",
                "has_weight": True,
                "path_shape": [4, 1],
            }
        ],
    )
    model = foundation_r1_test_model(
        r1_contract,
        atomic_numbers=(1,),
        m0_contract=m0_contract,
        r0_contract=r0_contract,
    )
    model["execution_contracts"]["M1"] = make_generated_m1_contract(
        channels=model["num_channels"],
        type_count=model["num_elements"],
        input_l_max=model["l_max"],
        correlation=max(len(term) for term in model["M1_monomials"]),
        monomials=model["M1_monomials"],
    )
    model["A0_scaled"] = True
    model["compact_radial"]["networks"]["A0"] = {
        "shape": [3, 1],
        "weights": [[0.01, -0.02, 0.03]],
        "activation": "silu",
        "activation_scale": 1.0,
        "postprocess": "tanh-square",
    }
    return model


def _standard_r0_operator_model():
    contracts = REPOSITORY / "symmetrix/test/data/execution_contracts"
    r1_contract = json.loads((contracts / "jit_r1_contract.json").read_text())
    m0_contract = json.loads((contracts / "standard_m0_contract.json").read_text())
    r0_contract = json.loads((contracts / "standard_r0_contract.json").read_text())
    model = foundation_r1_test_model(
        r1_contract,
        atomic_numbers=(1, 3),
        m0_contract=m0_contract,
        r0_contract=r0_contract,
    )
    model["A0_scaled"] = True
    model["compact_radial"]["networks"]["A0"] = {
        "shape": [64, 1],
        "weights": [[0.002 * (1 + index % 7) for index in range(64)]],
        "activation": "silu",
        "activation_scale": 1.0,
        "postprocess": "tanh-square",
    }
    return model


def _evaluate(calculator, atoms):
    inputs = calculator._mace_inputs(atoms)
    calculator.evaluator.compute_node_energies_forces(
        *inputs[:5], np.asarray(inputs[5]).reshape(-1), inputs[6]
    )
    result = calculator._collect_mace_results(atoms, inputs)
    return {
        "energy": result["energy"],
        "energies": np.array(result["energies"], copy=True),
        "forces": np.array(result["forces"], copy=True),
        "stress": np.array(result["stress"], copy=True),
    }


def _evaluate_weighted_retained(calculator, atoms):
    evaluator = calculator.evaluator
    evaluator._set_harmonic_storage_policy("retained")
    inputs = calculator._mace_inputs(atoms)
    graph = evaluator._prepare_factorized_graph(*inputs[:5])
    node_weights = np.linspace(0.8, 1.2, inputs[0], dtype=np.float64)
    edge_weights = np.linspace(0.7, 1.1, len(inputs[3]), dtype=np.float64)
    evaluator._set_alchemical_weights(graph, node_weights, edge_weights)
    evaluator._compute_prepared_factorized(
        graph, np.asarray(inputs[5]).reshape(-1), inputs[6]
    )
    result = calculator._collect_mace_results(atoms, inputs)
    return inputs, {
        "energy": result["energy"],
        "energies": np.array(result["energies"], copy=True),
        "forces": np.array(result["forces"], copy=True),
        "stress": np.array(result["stress"], copy=True),
    }


def _launch_cuda_r0_coordinate_slice(artifact_path, *, edge_begin, edge_count):
    class Spline(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("edge_types", ctypes.c_uint32),
            ("intervals", ctypes.c_uint32),
            ("functions", ctypes.c_uint32),
            ("h", ctypes.c_double),
            ("x0", ctypes.c_double),
            ("coefficients", ctypes.c_uint64),
        ]

    class Args(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("receiver_base", ctypes.c_uint32),
            ("num_nodes", ctypes.c_int64),
            ("num_edges", ctypes.c_int64),
            ("active_type_count", ctypes.c_int32),
            ("channels", ctypes.c_int32),
            ("l_max", ctypes.c_int32),
            ("coordinates_are_unit", ctypes.c_int32),
            ("apply_density_scale", ctypes.c_int32),
            ("use_precomputed_scale_adjoint", ctypes.c_int32),
            ("node_types", ctypes.c_uint64),
            ("num_neigh", ctypes.c_uint64),
            ("first_neigh", ctypes.c_uint64),
            ("neigh_types", ctypes.c_uint64),
            ("edge_receivers", ctypes.c_uint64),
            ("type_to_active", ctypes.c_uint64),
            ("coordinates", ctypes.c_uint64),
            ("radius", ctypes.c_uint64),
            ("radial", Spline),
            ("density", Spline),
            ("harmonics", ctypes.c_uint64),
            ("harmonic_gradients", ctypes.c_uint64),
            ("output", ctypes.c_uint64),
            ("output_adjoint", ctypes.c_uint64),
            ("density_state", ctypes.c_uint64),
            ("precomputed_scale_adjoint", ctypes.c_uint64),
            ("directed_forces", ctypes.c_uint64),
            ("cutoff", ctypes.c_double),
            ("edge_weights", ctypes.c_uint64),
        ]

    assert ctypes.sizeof(Args) == 264
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    cuda.cuMemAlloc_v2.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    cuda.cuMemcpyHtoD_v2.argtypes = [
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    cuda.cuMemcpyDtoH_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]

    def check(status, operation):
        assert status == 0, f"{operation} failed with CUDA status {status}"

    allocations = []

    def upload(values):
        array = np.ascontiguousarray(values)
        pointer = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), array.nbytes), "cuMemAlloc")
        allocations.append(pointer.value)
        check(
            cuda.cuMemcpyHtoD_v2(
                pointer.value, ctypes.c_void_p(array.ctypes.data), array.nbytes
            ),
            "cuMemcpyHtoD",
        )
        return pointer.value, array

    module = ctypes.c_void_p()
    try:
        check(
            cuda.cuModuleLoad(ctypes.byref(module), str(artifact_path).encode("utf-8")),
            "cuModuleLoad",
        )
        function = ctypes.c_void_p()
        check(
            cuda.cuModuleGetFunction(
                ctypes.byref(function), module, b"symmetrix_r0_coordinate_reverse_v1"
            ),
            "cuModuleGetFunction",
        )
        node_types, _ = upload(np.array([0], dtype=np.int32))
        neigh_types, _ = upload(np.zeros(4, dtype=np.int32))
        edge_receivers, _ = upload(np.zeros(4, dtype=np.int32))
        type_to_active, _ = upload(np.array([0], dtype=np.int32))
        coordinates, _ = upload(np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), 4))
        radius, _ = upload(np.ones(4, dtype=np.float64))
        coefficients, _ = upload(
            np.array([2.0] * 4 + [0.5] * 4 + [0.0] * 8, dtype=np.float32)
        )
        harmonics, _ = upload(np.ones(4, dtype=np.float32))
        output_adjoint, _ = upload(np.array([0.5] * 4, dtype=np.float32))
        directed_forces, host_forces = upload(np.zeros(12, dtype=np.float64))
        radial = Spline(ctypes.sizeof(Spline), 1, 1, 4, 2.0, 0.0, coefficients)
        args = Args(
            ctypes.sizeof(Args),
            0,
            1,
            edge_count,
            1,
            4,
            0,
            1,
            0,
            0,
            node_types,
            0,
            0,
            neigh_types + edge_begin * np.dtype(np.int32).itemsize,
            edge_receivers + edge_begin * np.dtype(np.int32).itemsize,
            type_to_active,
            coordinates + 3 * edge_begin * np.dtype(np.float32).itemsize,
            radius + edge_begin * np.dtype(np.float64).itemsize,
            radial,
            Spline(),
            harmonics + edge_begin * np.dtype(np.float32).itemsize,
            0,
            0,
            output_adjoint,
            0,
            0,
            directed_forces + 3 * edge_begin * np.dtype(np.float64).itemsize,
            2.0,
            0,
        )
        kernel_argument = ctypes.cast(ctypes.pointer(args), ctypes.c_void_p)
        parameters = (ctypes.c_void_p * 1)(kernel_argument)
        check(
            cuda.cuLaunchKernel(
                function,
                1,
                1,
                1,
                256,
                1,
                1,
                16 * 3 * np.dtype(np.float32).itemsize,
                None,
                parameters,
                None,
            ),
            "cuLaunchKernel",
        )
        check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
        check(
            cuda.cuMemcpyDtoH_v2(
                ctypes.c_void_p(host_forces.ctypes.data),
                directed_forces,
                host_forces.nbytes,
            ),
            "cuMemcpyDtoH",
        )
        return host_forces.reshape(-1, 3)
    finally:
        for pointer in reversed(allocations):
            cuda.cuMemFree_v2(pointer)
        if module.value:
            cuda.cuModuleUnload(module)


def _launch_cuda_r0_l4_forward(artifact_path, metadata):
    class Spline(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("edge_types", ctypes.c_uint32),
            ("intervals", ctypes.c_uint32),
            ("functions", ctypes.c_uint32),
            ("h", ctypes.c_double),
            ("x0", ctypes.c_double),
            ("coefficients", ctypes.c_uint64),
        ]

    class Args(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("receiver_base", ctypes.c_uint32),
            ("num_nodes", ctypes.c_int64),
            ("num_edges", ctypes.c_int64),
            ("active_type_count", ctypes.c_int32),
            ("channels", ctypes.c_int32),
            ("l_max", ctypes.c_int32),
            ("coordinates_are_unit", ctypes.c_int32),
            ("apply_density_scale", ctypes.c_int32),
            ("use_precomputed_scale_adjoint", ctypes.c_int32),
            ("node_types", ctypes.c_uint64),
            ("num_neigh", ctypes.c_uint64),
            ("first_neigh", ctypes.c_uint64),
            ("neigh_types", ctypes.c_uint64),
            ("edge_receivers", ctypes.c_uint64),
            ("type_to_active", ctypes.c_uint64),
            ("coordinates", ctypes.c_uint64),
            ("radius", ctypes.c_uint64),
            ("radial", Spline),
            ("density", Spline),
            ("harmonics", ctypes.c_uint64),
            ("harmonic_gradients", ctypes.c_uint64),
            ("output", ctypes.c_uint64),
            ("output_adjoint", ctypes.c_uint64),
            ("density_state", ctypes.c_uint64),
            ("precomputed_scale_adjoint", ctypes.c_uint64),
            ("directed_forces", ctypes.c_uint64),
            ("cutoff", ctypes.c_double),
            ("edge_weights", ctypes.c_uint64),
        ]

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    allocations = []

    def check(status, operation):
        assert status == 0, f"{operation} failed with CUDA status {status}"

    def upload(values):
        array = np.ascontiguousarray(values)
        pointer = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), array.nbytes), "cuMemAlloc")
        allocations.append(pointer.value)
        check(
            cuda.cuMemcpyHtoD_v2(
                pointer.value, ctypes.c_void_p(array.ctypes.data), array.nbytes
            ),
            "cuMemcpyHtoD",
        )
        return pointer.value, array

    channels = 128
    harmonic_count = 25
    functions = 5 * channels
    module = ctypes.c_void_p()
    try:
        check(
            cuda.cuModuleLoad(ctypes.byref(module), str(artifact_path).encode()), "load"
        )
        function = ctypes.c_void_p()
        check(
            cuda.cuModuleGetFunction(
                ctypes.byref(function), module, b"symmetrix_r0_forward_v1"
            ),
            "get function",
        )
        node_types, _ = upload(np.zeros(1, dtype=np.int32))
        num_neigh, _ = upload(np.ones(1, dtype=np.int32))
        first_neigh, _ = upload(np.zeros(1, dtype=np.int32))
        neigh_types, _ = upload(np.zeros(1, dtype=np.int32))
        type_to_active, _ = upload(np.zeros(1, dtype=np.int32))
        radius, _ = upload(np.ones(1, dtype=np.float64))
        coefficients, _ = upload(
            np.concatenate(
                (
                    np.ones(functions, dtype=np.float32),
                    np.zeros(3 * functions, dtype=np.float32),
                )
            )
        )
        harmonics, host_harmonics = upload(
            np.arange(1, harmonic_count + 1, dtype=np.float32)
        )
        output, host_output = upload(
            np.zeros(harmonic_count * channels, dtype=np.float32)
        )
        args = Args(
            ctypes.sizeof(Args),
            0,
            1,
            1,
            1,
            channels,
            4,
            0,
            0,
            0,
            node_types,
            num_neigh,
            first_neigh,
            neigh_types,
            0,
            type_to_active,
            0,
            radius,
            Spline(ctypes.sizeof(Spline), 1, 1, functions, 2.0, 0.0, coefficients),
            Spline(),
            harmonics,
            0,
            output,
            0,
            0,
            0,
            0,
            2.0,
            0,
        )
        parameters = (ctypes.c_void_p * 1)(
            ctypes.cast(ctypes.pointer(args), ctypes.c_void_p)
        )
        geometry = metadata["launch_geometry"]["forward"]
        blocks = (5 + geometry["owners_per_block"] - 1) // geometry["owners_per_block"]
        check(
            cuda.cuLaunchKernel(
                function,
                blocks,
                1,
                1,
                geometry["threads_per_block"],
                1,
                1,
                0,
                None,
                parameters,
                None,
            ),
            "launch",
        )
        check(cuda.cuCtxSynchronize(), "synchronize")
        check(
            cuda.cuMemcpyDtoH_v2(
                ctypes.c_void_p(host_output.ctypes.data), output, host_output.nbytes
            ),
            "download",
        )
        return host_output.reshape(harmonic_count, channels), host_harmonics
    finally:
        for pointer in reversed(allocations):
            cuda.cuMemFree_v2(pointer)
        if module.value:
            cuda.cuModuleUnload(module)


@pytest.fixture(scope="module")
def _standard_r0_operator_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("standard-r0-operator") / "model.json"
    path.write_text(json.dumps(_standard_r0_operator_model(), separators=(",", ":")))
    return path


@pytest.fixture(scope="module")
def _runtime_specialized_macefield_path(tmp_path_factory, macefield_model_path):
    from symmetrix.extract_mace_data import extract_mace_data

    output_dir = tmp_path_factory.mktemp("rtc-macefield")
    model = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=16,
    )
    zero_component = 0 if 0 in model["M0_monomials"] else "0"
    existing = {tuple(term) for term in model["M0_monomials"][zero_component]}
    input_components = (model["l_max"] + 1) ** 2
    additional = next(
        list(term)
        for degree in range(1, 4)
        for term in combinations_with_replacement(range(input_components), degree)
        if term not in existing
    )
    model["M0_monomials"][zero_component].append(additional)
    for type_weights in model["M0_weights"].values():
        for channel, weights in type_weights[zero_component].items():
            weights.append(1e-4 * (1 + int(channel) % 7))
    monomials = {
        int(component): terms for component, terms in model["M0_monomials"].items()
    }
    model["execution_contracts"]["M0"] = make_standard_m0_contract(
        channels=model["num_channels"],
        type_count=len(model["atomic_numbers"]),
        input_l_max=model["l_max"],
        output_l_max=model["L_max"],
        correlation=max(len(term) for terms in monomials.values() for term in terms),
        monomials=monomials,
    )
    model_path = output_dir / "runtime-specialized-macefield.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    return model_path


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.jit
@pytest.mark.parametrize("schedule", ("table", "chunk32", "tiled128x4-v1"))
@pytest.mark.parametrize("r0_schedule", ("receiver-edge16", "receiver-edge32"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_cuda_low_memory_runtime_specializes_custom_m0_r0(
    tmp_path, monkeypatch, dtype, r0_schedule, schedule
):
    _require_cuda_nvrtc()
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", schedule)
    monkeypatch.setenv("SYMMETRIX_M1_RTC_SCHEDULE", schedule)
    monkeypatch.setenv("SYMMETRIX_R0_RTC_SCHEDULE", r0_schedule)
    model_path = tmp_path / "custom-scalar-operators.json"
    model_path.write_text(
        json.dumps(_custom_scalar_operator_model(), separators=(",", ":"))
    )
    atoms = Atoms(
        "H4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.1],
            [0.3, 1.2, 0.2],
            [1.3, 1.1, 0.4],
        ],
        cell=[7.0, 7.0, 7.0],
        pbc=False,
    )

    retained = Symmetrix(
        model_path,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=False,
    )
    assert retained.evaluator.m0_implementation == "generic"
    assert retained.evaluator.r0_implementation == "generic"
    expected = _evaluate(retained, atoms)

    compact = Symmetrix(
        model_path,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=True,
    )
    evaluator = compact.evaluator
    assert evaluator.m0_implementation == "device_module"
    assert evaluator.r0_implementation == "generated"
    assert evaluator._r0_loader_implementation == "device_module"
    assert evaluator.r0_backend == "cuda"
    assert evaluator.r0_artifact_origin == "runtime"
    assert evaluator.r0_schedule == r0_schedule
    assert evaluator.m0_supports_low_memory
    assert evaluator.r0_supports_low_memory
    target = "sm_" + str(
        evaluator.execution_device_execution_environment["compute_capability_code"]
    )
    assert evaluator.m0_module_id.endswith(f"-{target}")
    assert compact.jit_operator_modules["M0"]["compiler"] == "nvrtc"
    assert compact.jit_operator_modules["R0"]["compiler"] == "nvrtc"
    assert compact.jit_operator_modules["M0"]["schedule"] == schedule
    assert compact.jit_operator_modules["M1"]["schedule"] == schedule
    assert compact.jit_operator_modules["R0"]["schedule"] == r0_schedule
    assert compact.jit_operator_modules["R0"]["schedule_revision"] == 6
    assert compact.jit_operator_modules["R0"]["compatibility_version"] == 2
    assert evaluator.r0_module_id == compact.jit_operator_modules["R0"]["artifact_id"]
    assert evaluator.standard_m0_poly_values_capacity_bytes == 0
    assert evaluator.standard_m0_poly_adjoints_capacity_bytes == 0

    actual = _evaluate(compact, atoms)
    tolerance = 8e-5 if dtype == "float32" else 8e-11
    assert actual["energy"] == pytest.approx(expected["energy"], rel=0.0, abs=tolerance)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=0.0, atol=tolerance
        )
    assert evaluator.m0_module_forward_launch_count > 0
    assert evaluator.m0_module_reverse_launch_count > 0
    assert evaluator.generated_m1_module_forward_launch_count > 0
    assert evaluator.generated_m1_module_reverse_launch_count > 0
    assert evaluator.m0_input_scale_adjoint_launch_count > 0
    assert evaluator.r0_module_launch_count > 0

    first_addresses = evaluator._test_mh0_adjoint_addresses()
    first_allocations = (
        evaluator.mh0_m0_adjoint_allocation_count,
        evaluator.mh0_a0_adjoint_allocation_count,
        evaluator.mh0_a1_adjoint_allocation_count,
    )
    assert first_addresses[0] == first_addresses[1] != 0
    assert first_addresses[2] != first_addresses[3]
    if evaluator.mh0_a1_adjoint_allocation_count:
        assert first_addresses[4] != first_addresses[5]

    repeated = _evaluate(compact, atoms)
    assert repeated["energy"] == pytest.approx(
        expected["energy"], rel=0.0, abs=tolerance
    )
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(
            repeated[name], expected[name], rtol=0.0, atol=tolerance
        )
    assert evaluator._test_mh0_adjoint_addresses() == first_addresses
    assert (
        evaluator.mh0_m0_adjoint_allocation_count,
        evaluator.mh0_a0_adjoint_allocation_count,
        evaluator.mh0_a1_adjoint_allocation_count,
    ) == first_allocations

    for policy in ("full-retention-v1", "reuse-adjoints-v1"):
        evaluator._set_mh0_state_policy(policy)
        transitioned = _evaluate(compact, atoms)
        for name in ("energies", "forces", "stress"):
            np.testing.assert_allclose(
                transitioned[name], expected[name], rtol=0.0, atol=tolerance
            )
        transitioned_addresses = evaluator._test_mh0_adjoint_addresses()
        assert transitioned_addresses[2] == first_addresses[2]
        assert evaluator.mh0_a0_adjoint_allocation_count == first_allocations[1]
        if policy == "full-retention-v1":
            assert transitioned_addresses[0] != transitioned_addresses[1]
            assert transitioned_addresses[4] != transitioned_addresses[5]
        else:
            assert transitioned_addresses[0] == transitioned_addresses[1]
            assert transitioned_addresses[4] != transitioned_addresses[5]
        policy_allocations = (
            evaluator.mh0_m0_adjoint_allocation_count,
            evaluator.mh0_a0_adjoint_allocation_count,
            evaluator.mh0_a1_adjoint_allocation_count,
        )
        _evaluate(compact, atoms)
        assert evaluator._test_mh0_adjoint_addresses() == transitioned_addresses
        assert (
            evaluator.mh0_m0_adjoint_allocation_count,
            evaluator.mh0_a0_adjoint_allocation_count,
            evaluator.mh0_a1_adjoint_allocation_count,
        ) == policy_allocations

    m0_path = compact.jit_operator_modules["M0"]["artifact_path"]
    r0_path = compact.jit_operator_modules["R0"]["artifact_path"]
    r0_blocks = compact.jit_operator_modules["R0"]["persistent_blocks_per_compute_unit"]
    wrong_schedule = "tiled128x4-v1" if schedule == "chunk32" else "chunk32"
    with pytest.raises(RuntimeError, match="identity does not match"):
        evaluator._load_m0_device_module(m0_path, wrong_schedule, 8)
    with pytest.raises(RuntimeError, match="identity does not match"):
        evaluator._load_r0_device_module(m0_path, r0_blocks, False, r0_schedule)
    with pytest.raises(ValueError, match="profile"):
        evaluator._load_r0_device_module(r0_path, r0_blocks + 1, False, r0_schedule)

    if dtype == "float32" and schedule == "chunk32":
        m0_contract = json.loads(model_path.read_text())["execution_contracts"]["M0"]
        m0_source, m0_metadata = render_jit_m0_device_module(
            m0_contract, precision=dtype, target=target, schedule=schedule
        )
        for name, stale_source, message in (
            (
                "old-schedule-revision",
                m0_source.replace(
                    "symmetrix_operator_schedule_revision_v1 = 2;",
                    "symmetrix_operator_schedule_revision_v1 = 1;",
                ),
                "schedule profile|launch geometry",
            ),
            (
                "surplus-alias-capability",
                m0_source.replace(
                    "symmetrix_operator_capabilities_v1 = 27u;",
                    "symmetrix_operator_capabilities_v1 = 31u;",
                ),
                "capabilities do not match",
            ),
        ):
            assert stale_source != m0_source
            stale_artifact = prepare_nvrtc_jit_artifact(
                stale_source,
                abi=artifact_cache_inputs(m0_metadata, OPERATOR_ARTIFACT_ENVELOPE),
                build={"generator": f"test.m0-{name}", "operator": m0_metadata},
                compute_capability=evaluator.execution_device_execution_environment,
                target=evaluator.execution_device_execution_environment,
                cache_root=tmp_path / name,
                artifact_name=f"m0_{name}",
                nvrtc_options=("--split-compile=2",),
            )
            assert stale_artifact.available, (
                stale_artifact.reason,
                stale_artifact.diagnostics,
            )
            with pytest.raises(RuntimeError, match=message):
                evaluator._load_m0_device_module(
                    str(stale_artifact.artifact_path), schedule, 8
                )
    damaged_path = tmp_path / "damaged-operator.cubin"
    damaged_path.write_bytes(Path(r0_path).read_bytes()[:128])
    with pytest.raises(RuntimeError, match="ELF"):
        evaluator._load_r0_device_module(
            str(damaged_path), r0_blocks, False, r0_schedule
        )

    after_rejected_load = _evaluate(compact, atoms)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(
            after_rejected_load[name], actual[name], rtol=0.0, atol=tolerance
        )

    prepared = prepare_jit_device_artifact(
        model_path,
        precision=dtype,
        backend="cuda",
        cache_root=tmp_path / "jit-cache",
    )
    assert prepared.status == "cached"
    assert prepared.operator_modules["M0"]["status"] == "cached"
    assert prepared.operator_modules["R0"]["status"] == "cached"


@pytest.mark.jit
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_host_low_memory_runtime_retains_disjoint_generated_adjoints(
    tmp_path, monkeypatch, dtype
):
    if native_symmetrix._kokkos_default_execution_space() not in (
        "OpenMP",
        "Serial",
    ):
        pytest.skip("generated host lifecycle requires a host Kokkos build")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "chunk32")
    model_path = tmp_path / "custom-scalar-operators.json"
    model_path.write_text(
        json.dumps(_custom_scalar_operator_model(), separators=(",", ":"))
    )
    atoms = Atoms(
        "H4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.1],
            [0.3, 1.2, 0.2],
            [1.3, 1.1, 0.4],
        ],
        cell=[7.0, 7.0, 7.0],
        pbc=False,
    )
    retained = Symmetrix(
        model_path, dtype=dtype, streamed_edges="direct", low_memory=False
    )
    expected = _evaluate(retained, atoms)
    compact = Symmetrix(
        model_path, dtype=dtype, streamed_edges="direct", low_memory=True
    )
    evaluator = compact.evaluator
    assert evaluator.m0_implementation == "host_plugin"
    tolerance = 8e-5 if dtype == "float32" else 8e-11
    for iteration in range(2):
        actual = _evaluate(compact, atoms)
        for name in ("energies", "forces", "stress"):
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=0.0, atol=tolerance
            )
        addresses = evaluator._test_mh0_adjoint_addresses()
        assert addresses[2] != addresses[3]
        if evaluator.mh0_a1_adjoint_allocation_count:
            assert addresses[4] != addresses[5]
        if iteration == 0:
            first_addresses = addresses
            first_allocations = (
                evaluator.mh0_m0_adjoint_allocation_count,
                evaluator.mh0_a0_adjoint_allocation_count,
                evaluator.mh0_a1_adjoint_allocation_count,
            )
    assert evaluator._test_mh0_adjoint_addresses() == first_addresses
    assert (
        evaluator.mh0_m0_adjoint_allocation_count,
        evaluator.mh0_a0_adjoint_allocation_count,
        evaluator.mh0_a1_adjoint_allocation_count,
    ) == first_allocations
    evaluator = None
    compact = None
    retained = None
    gc.collect()


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.jit
@pytest.mark.parametrize(
    ("dtype", "schedule"),
    (("float32", "receiver-edge16"), ("float64", "receiver-edge32")),
)
def test_cuda_standard_r0_medium_weighted_retained_grid(
    _standard_r0_operator_model_path, tmp_path, monkeypatch, dtype, schedule
):
    _require_cuda_nvrtc()
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    atoms = bulk("LiH", "rocksalt", a=3.9, cubic=True).repeat((5, 5, 5))
    control = Symmetrix(
        _standard_r0_operator_model_path,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=False,
    )
    candidate = Symmetrix(
        _standard_r0_operator_model_path,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=False,
    )
    evaluator = candidate.evaluator
    environment = evaluator.execution_device_execution_environment
    target = "sm_" + str(environment["compute_capability_code"])
    contract = json.loads(_standard_r0_operator_model_path.read_text())[
        "execution_contracts"
    ]["R0"]
    source, metadata = render_jit_r0_device_module(
        contract,
        precision=dtype,
        target=target,
        schedule=schedule,
        apply_edge_weights=True,
    )
    artifact = prepare_nvrtc_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"generator": "test.standard-r0-medium", "operator": metadata},
        compute_capability=environment,
        target=environment,
        cache_root=tmp_path / "r0-cache",
        artifact_name=f"r0_{dtype}_{schedule}",
        nvrtc_options=("--split-compile=2", "--use_fast_math"),
    )
    assert artifact.available, (artifact.reason, artifact.diagnostics)
    profile = metadata["persistent_profile"]["blocks_per_compute_unit"]
    evaluator._load_r0_device_module(
        str(artifact.artifact_path), profile, True, schedule
    )
    assert evaluator.r0_implementation == "generated"
    assert evaluator._r0_loader_implementation == "device_module"
    assert metadata["architecture_bounds"] == {
        "channels": 128,
        "l_max": 3,
        "harmonic_count": 16,
    }

    control_inputs, expected = _evaluate_weighted_retained(control, atoms)
    candidate_inputs, actual = _evaluate_weighted_retained(candidate, atoms)
    assert control_inputs[0] == 1000
    assert len(control_inputs[3]) == len(candidate_inputs[3])
    assert np.count_nonzero(np.asarray(candidate_inputs[6]) >= evaluator.r_cut) > 0
    assert evaluator.harmonic_storage_policy == "retained"
    persistent_limit = profile * environment["compute_unit_count"]
    geometry = metadata["launch_geometry"]
    reverse_required = (
        candidate_inputs[0] * geometry["reverse_prepare"]["extent_multiplier"]
        + geometry["reverse_prepare"]["owners_per_block"]
        - 1
    ) // geometry["reverse_prepare"]["owners_per_block"]
    coordinate_required = (
        len(candidate_inputs[3]) * geometry["coordinate_reverse"]["extent_multiplier"]
        + geometry["coordinate_reverse"]["owners_per_block"]
        - 1
    ) // geometry["coordinate_reverse"]["owners_per_block"]
    assert reverse_required > persistent_limit
    assert coordinate_required > persistent_limit

    tolerance = 4e-4 if dtype == "float32" else 4e-10
    assert actual["energy"] == pytest.approx(expected["energy"], rel=0.0, abs=tolerance)
    for name in ("energies", "forces", "stress"):
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=2e-6, atol=tolerance
        )

    with pytest.raises(ValueError, match="profile"):
        evaluator._load_r0_device_module(
            str(artifact.artifact_path), profile + 1, True, schedule
        )
    if dtype == "float32":
        revision_mismatch = source.replace(
            "symmetrix_operator_schedule_revision_v1 = 6;",
            "symmetrix_operator_schedule_revision_v1 = 4;",
        )
        assert revision_mismatch != source
        mismatched_artifact = prepare_nvrtc_jit_artifact(
            revision_mismatch,
            abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
            build={"generator": "test.r0-revision-mismatch", "operator": metadata},
            compute_capability=environment,
            target=environment,
            cache_root=tmp_path / "revision-cache",
            artifact_name="r0_revision_mismatch",
            nvrtc_options=("--split-compile=2", "--use_fast_math"),
        )
        assert mismatched_artifact.available, (
            mismatched_artifact.reason,
            mismatched_artifact.diagnostics,
        )
        with pytest.raises(RuntimeError, match="schedule profile|launch geometry"):
            evaluator._load_r0_device_module(
                str(mismatched_artifact.artifact_path), profile, True, schedule
            )

        for name, stale_source, message in (
            (
                "old-abi-v2",
                source.replace(
                    "symmetrix_operator_abi_version_v1 = 5u;",
                    "symmetrix_operator_abi_version_v1 = 3u;",
                ),
                "ABI",
            ),
            (
                "mutated-compatibility",
                source.replace(
                    "symmetrix_operator_r0_compatibility_version_v1 = 2u;",
                    "symmetrix_operator_r0_compatibility_version_v1 = 1u;",
                ),
                "compatibility_version",
            ),
        ):
            assert stale_source != source
            stale_artifact = prepare_nvrtc_jit_artifact(
                stale_source,
                abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
                build={"generator": f"test.r0-{name}", "operator": metadata},
                compute_capability=environment,
                target=environment,
                cache_root=tmp_path / name,
                artifact_name=name,
                nvrtc_options=("--split-compile=2", "--use_fast_math"),
            )
            assert stale_artifact.available, (
                stale_artifact.reason,
                stale_artifact.diagnostics,
            )
            with pytest.raises(RuntimeError, match=message):
                evaluator._load_r0_device_module(
                    str(stale_artifact.artifact_path), profile, True, schedule
                )


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.jit
def test_cuda_r0_coordinate_reverse_executes_nonzero_pointer_slice(tmp_path):
    _require_cuda_nvrtc()
    environment = native_symmetrix._execution_device_execution_environment()
    target = "sm_" + str(environment["compute_capability_code"])
    contract = _custom_scalar_operator_model()["execution_contracts"]["R0"]
    source, metadata = render_jit_r0_device_module(
        contract,
        precision="float32",
        target=target,
        schedule="receiver-edge16",
    )
    artifact = prepare_nvrtc_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"generator": "test.r0-coordinate-slice", "operator": metadata},
        compute_capability=environment,
        target=environment,
        cache_root=tmp_path / "slice-cache",
        artifact_name="r0_coordinate_slice",
        nvrtc_options=("--split-compile=2", "--use_fast_math"),
    )
    assert artifact.available, (artifact.reason, artifact.diagnostics)

    forces = _launch_cuda_r0_coordinate_slice(
        artifact.artifact_path, edge_begin=1, edge_count=2
    )

    np.testing.assert_allclose(
        forces,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.jit
def test_cuda_r0_nonwide_l4_forward_executes_generic_components(tmp_path):
    _require_cuda_nvrtc()
    environment = native_symmetrix._execution_device_execution_environment()
    contract = make_standard_r0_contract(
        channels=128,
        radial_embedding=64,
        edge_l_max=4,
        source_irreps="128x0e",
        edge_irreps="1x0e+1x1o+1x2e+1x3o+1x4e",
        output_irreps="128x0e+128x1o+128x2e+128x3o+128x4e",
        output_l=range(5),
        edge_l=range(5),
        source_l=(0,) * 5,
        instructions=(
            {"connection_mode": "uvu", "has_weight": True, "path_shape": [128, 1]}
            for _ in range(5)
        ),
    )
    target = "sm_" + str(environment["compute_capability_code"])
    source, metadata = render_jit_r0_device_module(
        contract, precision="float32", target=target, schedule="receiver-edge16"
    )
    assert metadata["schedule_policy"]["forward_owner"] == (
        "receiver-angular-order-team"
    )
    artifact = prepare_nvrtc_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"generator": "test.r0-l4-forward", "operator": metadata},
        compute_capability=environment,
        target=environment,
        cache_root=tmp_path / "l4-forward-cache",
        artifact_name="r0_l4_forward",
        nvrtc_options=("--split-compile=2", "--use_fast_math"),
    )
    assert artifact.available, (artifact.reason, artifact.diagnostics)

    output, harmonics = _launch_cuda_r0_l4_forward(artifact.artifact_path, metadata)
    np.testing.assert_allclose(
        output, np.broadcast_to(harmonics[:, None], output.shape), rtol=0.0, atol=0.0
    )


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.jit
def test_cuda_r0_loader_rejects_opposite_build_floating_policy(
    _standard_r0_operator_model_path, tmp_path
):
    _require_cuda_nvrtc()
    environment = native_symmetrix._execution_device_execution_environment()
    owning_fast_math = native_symmetrix._backend_build_info().get("fast_math", True)
    policy = "strict" if owning_fast_math else "cuda-fast-math"
    target = "sm_" + str(environment["compute_capability_code"])
    contract = json.loads(_standard_r0_operator_model_path.read_text())[
        "execution_contracts"
    ]["R0"]
    source, metadata = render_jit_r0_device_module(
        contract,
        precision="float32",
        target=target,
        schedule="receiver-edge16",
        floating_point_policy=policy,
    )
    options = ["--split-compile=2"]
    if policy == "cuda-fast-math":
        options.append("--use_fast_math")
    artifact = prepare_nvrtc_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"generator": "test.r0-opposite-policy", "operator": metadata},
        compute_capability=environment,
        target=environment,
        cache_root=tmp_path / "opposite-policy-cache",
        artifact_name="r0_opposite_policy",
        nvrtc_options=tuple(options),
    )
    assert artifact.available, (artifact.reason, artifact.diagnostics)
    evaluator = Symmetrix(
        _standard_r0_operator_model_path,
        dtype="float32",
        streamed_edges="direct",
        low_memory=False,
    ).evaluator
    with pytest.raises(RuntimeError, match="artifact"):
        evaluator._load_r0_device_module(
            str(artifact.artifact_path),
            metadata["persistent_profile"]["blocks_per_compute_unit"],
            False,
            "receiver-edge16",
        )


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.jit
@pytest.mark.parametrize(
    "dtype,tolerance",
    (("float32", 1e-3), ("float64", 2e-8)),
)
def test_cuda_runtime_m0_supports_macefield_analytical_response(
    _runtime_specialized_macefield_path,
    tmp_path,
    monkeypatch,
    dtype,
    tolerance,
):
    _require_cuda_nvrtc()
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "chunk32")

    retained = Symmetrix(
        _runtime_specialized_macefield_path,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=False,
    )
    compact = Symmetrix(
        _runtime_specialized_macefield_path,
        dtype=dtype,
        streamed_edges="direct",
        low_memory=True,
    )
    assert retained.evaluator.m0_implementation == "generic"
    assert retained.evaluator.r0_implementation == "generated"
    assert retained.evaluator._r0_loader_implementation == "builtin"
    assert compact.evaluator.m0_implementation == "device_module"
    assert compact.evaluator.r0_implementation == "generated"
    assert compact.evaluator._r0_loader_implementation == "builtin"
    assert compact.jit_operator_modules["M0"]["compiler"] == "nvrtc"
    assert compact.jit_operator_modules["R0"]["status"] == "builtin"

    properties = (
        "energy",
        "node_energy",
        "forces",
        "stress",
        "polarization",
        "polarizability",
        "becs",
    )
    results = {}
    for name, calculator in (("retained", retained), ("compact", compact)):
        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
        atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])
        calculator.calculate(atoms, properties=list(properties))
        results[name] = {
            property_name: np.array(calculator.results[property_name], copy=True)
            for property_name in properties
        }

    assert compact.evaluator.low_memory_policy == "speed"
    # One MH-0 rebuild per response calculation at most: the plan may also
    # serve the request from the prepared primal.
    assert compact.evaluator.macefield_response_primal_reconstruction_count <= 1
    assert compact.evaluator.m0_module_reverse_launch_count > 2
    for property_name in properties:
        np.testing.assert_allclose(
            results["compact"][property_name],
            results["retained"][property_name],
            rtol=0.0,
            atol=tolerance,
        )
