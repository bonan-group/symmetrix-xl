"""Compiler-gated syntax coverage for generated NequIP-OAM CUDA R1 plugins.

This module proves the renderer emits a real CUDA translation unit by handing
it to ``nvcc`` when a usable toolchain is present; otherwise it skips with
the exact missing prerequisite.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from symmetrix.oam_cuda_codegen import (
    ABI_HEADER_NAME,
    OAMCudaGenerationPolicy,
    _render_descriptor,
    _render_module_identity,
    _render_warp_instruction_forward,
    oam_cuda_artifact_metadata,
    render_oam_cuda_plugin,
)

FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"
PLUGIN_QUERY_SYMBOL = "symmetrix_oam_cuda_plugin_query_v3"


def test_native_cuda_identity_parses_and_rejects_full_message_resource_mismatches(
    tmp_path,
):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for CUDA identity validation")
    repository = Path(__file__).resolve().parents[2]
    source = tmp_path / "identity_fields.cpp"
    source.write_text(
        '#include "cuda_r1_identity.hpp"\n#include <algorithm>\n#include <string>\n'
        "int main(){using namespace symmetrix::execution;"
        'nlohmann::json j={{"target_policy","target32"},{"precision_policy","precision32"},'
        '{"schedule_qualification","candidate-unqualified-explicit"}};'
        "auto published=detail::cuda_r1_manifest_identity(j);"
        "CudaR1ArtifactIdentity requested;"
        'requested.target_policy="target16";requested.precision_policy="precision16";'
        'requested.schedule_qualification="qualified";'
        "auto mismatches=cuda_r1_identity_mismatches(requested,published);"
        'for(const char* key:{"target_policy","precision_policy",'
        '"schedule_qualification"})'
        "if(std::find(mismatches.begin(),mismatches.end(),key)==mismatches.end())return 2;"
        "return 0;}",
        encoding="ascii",
    )
    executable = tmp_path / "identity_fields"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(repository / "libsymmetrix" / "source"),
            "-I",
            str(repository / "libsymmetrix" / "external" / "json" / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([executable], check=True, capture_output=True, text=True)
    binding = (
        repository / "symmetrix" / "source" / "cpp" / "oam_kokkos.cpp"
    ).read_text()
    for field in (
        "target_policy",
        "precision_policy",
        "schedule_qualification",
    ):
        assert "identity_field_" in binding and f'"{field}"' in binding


def _oam_xl_degree_four_contract():
    contract = json.loads(FIXTURE.read_text())
    edge_irreps = "1x0e+1x1o+1x2e+1x3o+1x4e"
    for layer in contract["layers"]:
        layer["edge_irreps"] = edge_irreps
        layer["edge_tensor_product"]["irreps_in2"] = edge_irreps
    return contract


def _nvcc():
    """Return a usable ``nvcc`` or skip with the exact missing prerequisite."""

    for candidate in (
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.3/bin/nvcc",
    ):
        if candidate is None or not Path(candidate).is_file():
            continue
        with tempfile.TemporaryDirectory(prefix="symmetrix-nvcc-probe-") as directory:
            source = Path(directory) / "probe.cu"
            output = Path(directory) / "probe.o"
            source.write_text(
                "#include <cuda_runtime.h>\n"
                "__global__ void probe(float* value) { value[0] = sqrtf(value[0]); }\n",
                encoding="ascii",
            )
            probe = subprocess.run(
                [candidate, "-std=c++20", "-c", str(source), "-o", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
        if probe.returncode == 0:
            return candidate
        if "exception specification is incompatible" in probe.stderr and (
            'function "rsqrt"' in probe.stderr or 'function "rsqrtf"' in probe.stderr
        ):
            pytest.skip(
                "installed CUDA toolkit is incompatible with system glibc "
                "rsqrt declarations"
            )
        pytest.fail(f"nvcc qualification probe failed:\n{probe.stderr}")
    pytest.skip(
        f"nvcc is required to compile the generated OAM CUDA plugin "
        f"({ABI_HEADER_NAME} is inlined by the renderer)"
    )


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_generated_oam_cuda_plugin_compiles(precision, tmp_path):
    compiler = _nvcc()
    contract = json.loads(FIXTURE.read_text())
    source = render_oam_cuda_plugin(contract, "sm120", precision=precision)

    translation_unit = tmp_path / f"oam_r1_{precision}.cu"
    translation_unit.write_text(source, encoding="ascii")
    object_file = tmp_path / f"oam_r1_{precision}.o"
    completed = subprocess.run(
        [compiler, "-std=c++20", "-c", str(translation_unit), "-o", str(object_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert object_file.is_file()


def test_generated_oam_xl_degree_four_cuda_plugin_compiles(tmp_path):
    compiler = _nvcc()
    source = render_oam_cuda_plugin(
        _oam_xl_degree_four_contract(), "sm120", precision="float32"
    )

    translation_unit = tmp_path / "oam_r1_l4_float32.cu"
    translation_unit.write_text(source, encoding="ascii")
    object_file = tmp_path / "oam_r1_l4_float32.o"
    completed = subprocess.run(
        [compiler, "-std=c++20", "-c", str(translation_unit), "-o", str(object_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert object_file.is_file()


def test_generated_grouped_staged_cuda_plugin_compiles(tmp_path):
    compiler = _nvcc()
    contract = json.loads(FIXTURE.read_text())
    source = render_oam_cuda_plugin(
        contract,
        "sm120",
        precision="float32",
        generation_policy=OAMCudaGenerationPolicy(
            forward_live_output_budget=8,
            forward_input_mode="staged",
        ),
    )

    translation_unit = tmp_path / "oam_r1_grouped_staged_float32.cu"
    translation_unit.write_text(source, encoding="ascii")
    object_file = tmp_path / "oam_r1_grouped_staged_float32.o"
    completed = subprocess.run(
        [compiler, "-std=c++20", "-c", str(translation_unit), "-o", str(object_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert object_file.is_file()


def test_generated_source_grouped_warp_task_cuda_plugin_compiles(tmp_path):
    compiler = _nvcc()
    contract = json.loads(FIXTURE.read_text())
    source = render_oam_cuda_plugin(
        contract,
        "sm120",
        precision="float32",
        generation_policy=OAMCudaGenerationPolicy(
            forward_live_output_budget=16,
            forward_grouping_mode="source_first_fit",
            forward_group_execution="warp_tasks",
            radial_adjoint_mode="fp32",
        ),
    )

    translation_unit = tmp_path / "oam_r1_source_grouped_warp_tasks_float32.cu"
    translation_unit.write_text(source, encoding="ascii")
    object_file = tmp_path / "oam_r1_source_grouped_warp_tasks_float32.o"
    completed = subprocess.run(
        [compiler, "-std=c++20", "-c", str(translation_unit), "-o", str(object_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert object_file.is_file()


def test_generated_grouped_reverse_cuda_plugin_compiles(tmp_path):
    compiler = _nvcc()
    contract = json.loads(FIXTURE.read_text())
    source = render_oam_cuda_plugin(
        contract,
        "sm120",
        precision="float32",
        generation_policy=OAMCudaGenerationPolicy(
            radial_adjoint_mode="fp32",
            reverse_group_execution="source_groups",
        ),
    )

    translation_unit = tmp_path / "oam_r1_grouped_reverse_float32.cu"
    translation_unit.write_text(source, encoding="ascii")
    object_file = tmp_path / "oam_r1_grouped_reverse_float32.o"
    completed = subprocess.run(
        [compiler, "-std=c++20", "-c", str(translation_unit), "-o", str(object_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert object_file.is_file()


def test_warp_task_instruction_with_multiple_output_components_compiles(tmp_path):
    compiler = _nvcc()
    instruction = {
        "i_in1": 0,
        "i_in2": 0,
        "i_out": 0,
        "connection_mode": "uvu",
        "has_weight": False,
        "path_weight": 1.0,
        "wigner_3j": {
            "shape": [3, 1, 3],
            "values": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
    }
    layer = {
        "edge_irreps": "1x0e",
        "edge_tensor_product": {
            "irreps_in1": "1x1o",
            "irreps_in2": "1x0e",
            "irreps_out": "1x1o",
        },
    }
    body = _render_warp_instruction_forward(layer, instruction, 0, "float")
    source = (
        "#include <cuda_runtime.h>\n"
        "__global__ void instruction(const float* src, const float* harmonics) {\n"
        "  const int a = static_cast<int>(threadIdx.x);\n"
        "  float accumulator_0_0 = 0, accumulator_0_1 = 0, accumulator_0_2 = 0;\n"
        f"{body}\n"
        "  if (a == 0 && accumulator_0_0 + accumulator_0_1 + accumulator_0_2 == 0) {}\n"
        "}\n"
    )
    translation_unit = tmp_path / "warp_instruction_l1.cu"
    translation_unit.write_text(source, encoding="ascii")
    object_file = tmp_path / "warp_instruction_l1.o"
    completed = subprocess.run(
        [compiler, "-std=c++20", "-c", str(translation_unit), "-o", str(object_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert object_file.is_file()


@pytest.mark.parametrize(
    "radial_adjoint_mode", ("fp32", "compensated_fp32", "hierarchical_fp32")
)
def test_generated_narrow_radial_adjoint_cuda_plugin_compiles(
    radial_adjoint_mode, tmp_path
):
    compiler = _nvcc()
    contract = json.loads(FIXTURE.read_text())
    source = render_oam_cuda_plugin(
        contract,
        "sm120",
        precision="float32",
        generation_policy=OAMCudaGenerationPolicy(
            radial_adjoint_mode=radial_adjoint_mode
        ),
    )

    translation_unit = tmp_path / f"oam_r1_{radial_adjoint_mode}.cu"
    translation_unit.write_text(source, encoding="ascii")
    object_file = tmp_path / f"oam_r1_{radial_adjoint_mode}.o"
    completed = subprocess.run(
        [compiler, "-std=c++20", "-c", str(translation_unit), "-o", str(object_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert object_file.is_file()


@pytest.mark.parametrize("representation", ("descriptor", "module"))
def test_generated_identity_rejects_independent_fingerprint_mismatches(
    representation, tmp_path
):
    """Compile emitted identity data and exercise the real native loader."""
    if os.name != "posix":
        pytest.skip("OAM CUDA shared-library loading requires POSIX")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for OAM CUDA identity validation")
    repository = Path(__file__).resolve().parents[2]
    native_source = repository / "libsymmetrix" / "source"
    package_source = repository / "symmetrix" / "source" / "symmetrix"
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    metadata = oam_cuda_artifact_metadata(
        contract,
        target="sm120",
        generation_policy=OAMCudaGenerationPolicy(),
    )
    # Independent fixture identities catch either emitter aliasing the slots.
    metadata["semantic_fingerprint"] = "sha256:generated-mathematics"
    launchers = "\n".join(
        f"int32_t oam_{direction}_launch_{index}("
        "const SymmetrixOAMCudaLayerArgsV2*, void*, int32_t) { return 0; }"
        for index in range(metadata["layer_count"])
        for direction in ("forward", "reverse")
    )
    plugin_text = '#include "oam_cuda_plugin_abi.h"\n' + launchers + "\n"
    if representation == "descriptor":
        plugin_text += _render_descriptor(metadata, "float")
    else:
        # CUDA qualifiers affect storage, not identity contents. Expose those
        # emitted constants through a descriptor to use the same validator.
        plugin_text += "#define __device__\n#define __constant__\n"
        plugin_text += _render_module_identity(metadata, "float")
        plugin_text += _render_descriptor(metadata, "float").replace(
            "return &oam_cuda_plugin;",
            "static auto module = oam_cuda_plugin;\n"
            "    module.contract_fingerprint = "
            "symmetrix_oam_cuda_module_contract_fingerprint;\n"
            "    module.semantic_fingerprint = "
            "symmetrix_oam_cuda_module_semantic_fingerprint;\n"
            "    module.structure_fingerprint = "
            "symmetrix_oam_cuda_module_structure_fingerprint;\n"
            "    return &module;",
        )
    plugin_source = tmp_path / "identity.cpp"
    plugin_source.write_text(plugin_text, encoding="ascii")
    plugin = tmp_path / "identity.so"
    common = ["-std=c++20", "-I", str(native_source), "-I", str(package_source)]
    subprocess.run(
        [compiler, *common, "-shared", "-fPIC", str(plugin_source), "-o", str(plugin)],
        check=True,
        capture_output=True,
        text=True,
    )

    layer_rows = ",".join(
        "{"
        + ",".join(
            str(row[field])
            for field in (
                "source_dimension",
                "edge_dimension",
                "weight_dimension",
                "output_dimension",
                "target_dimension",
                "instruction_count",
                "spline_interval_count",
                "spline_function_count",
                "edge_harmonic_count",
            )
        )
        + "}"
        for row in metadata["layers"]
    )
    harness_source = tmp_path / "loader.cpp"
    harness_source.write_text(
        '#include "oam_cuda_plugin.hpp"\n'
        "#include <exception>\n#include <iostream>\n#include <string>\n"
        "int main(int argc, char** argv) {\n"
        "    if (argc != 3) return 64;\n"
        "    symmetrix::execution::OAMCudaPluginExpectation expected;\n"
        f"    expected.artifact_id = {json.dumps(metadata['artifact_id'])};\n"
        f"    expected.contract_fingerprint = {json.dumps(metadata['contract_fingerprint'])};\n"
        f"    expected.semantic_fingerprint = {json.dumps(metadata['semantic_fingerprint'])};\n"
        f"    expected.structure_fingerprint = {json.dumps(metadata['structure_fingerprint'])};\n"
        '    expected.target_architecture = "sm120";\n'
        "    expected.target_compute_capability = 120;\n"
        f"    expected.max_threads_per_block = {metadata['max_threads_per_block']};\n"
        f"    expected.target_policy = {json.dumps(metadata['target_policy'])};\n"
        f"    expected.precision_policy = {json.dumps(metadata['precision_policy'])};\n"
        f"    expected.schedule_qualification = {json.dumps(metadata['schedule_qualification'])};\n"
        f"    expected.required_capabilities = {metadata['required_capabilities']}u;\n"
        f"    expected.layers = {{{layer_rows}}};\n"
        "    const std::string field = argv[1];\n"
        '    if (field == "architecture") expected.contract_fingerprint = "sha256:wrong";\n'
        '    if (field == "semantic") expected.semantic_fingerprint = "sha256:wrong";\n'
        '    if (field == "target") expected.target_policy = "wrong-target";\n'
        '    if (field == "precision") expected.precision_policy = "wrong-precision";\n'
        '    if (field == "qualification") expected.schedule_qualification = "wrong-qualification";\n'
        "    try {\n"
        "        auto plugin = symmetrix::execution::OAMCudaPlugin::load(argv[2], expected);\n"
        '        return field == "accept" && plugin ? 0 : 1;\n'
        "    } catch (const std::exception& error) {\n"
        "        std::cerr << error.what() << '\\n';\n"
        '        return field != "accept" && std::string(error.what()).find(\n'
        '            field + " fingerprint does not match") != std::string::npos ? 0 : 2;\n'
        "    }\n}\n",
        encoding="ascii",
    )
    harness = tmp_path / "loader"
    subprocess.run(
        [
            compiler,
            *common,
            str(harness_source),
            str(native_source / "oam_cuda_plugin.cpp"),
            "-ldl",
            "-o",
            str(harness),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mismatch_diagnostics = {
        "target": "target policy does not match",
        "precision": "precision policy does not match",
        "qualification": "schedule qualification does not match",
    }
    for mode in (
        "accept",
        "architecture",
        "semantic",
        *mismatch_diagnostics,
    ):
        result = subprocess.run(
            [str(harness), mode, str(plugin)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
            or mismatch_diagnostics.get(mode, "<no mismatch>") in result.stderr
        ), result.stderr
    stale_source = tmp_path / "stale_v2.cpp"
    stale_source.write_text(
        'extern "C" void* symmetrix_oam_cuda_plugin_query_v2(){return nullptr;}',
        encoding="ascii",
    )
    stale_plugin = tmp_path / "stale_v2.so"
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(stale_source), "-o", str(stale_plugin)],
        check=True,
        capture_output=True,
        text=True,
    )
    stale = subprocess.run(
        [str(harness), "stale", str(stale_plugin)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert stale.returncode == 2
    assert "stale CUDA ABI v2 artifact" in stale.stderr
