import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from symmetrix.foundation_operator_manifest import (
    load_foundation_operator_manifest,
    r1_operator_descriptor,
)
from symmetrix.mace_execution_contract import (
    make_generated_m1_contract,
    normalize_jit_r1_contract,
    normalize_generated_m1_contract,
    normalize_standard_m0_contract,
    normalize_standard_r0_contract,
)

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "libsymmetrix" / "source"
CONTRACTS = Path(__file__).resolve().parent / "data" / "execution_contracts"


def test_generated_r0_device_abi_and_generation_revision_are_current():
    abi = (
        ROOT / "symmetrix/source/symmetrix/mace_jit_operator_module_abi.h"
    ).read_text()
    m1_host_abi = (
        ROOT / "symmetrix/source/symmetrix/mace_jit_m1_host_plugin_abi.h"
    ).read_text()
    native_version = (NATIVE / "jit_generation_version.hpp").read_text()
    python_version = (
        ROOT / "symmetrix/source/symmetrix/generated_operator_version.py"
    ).read_text()

    assert "SYMMETRIX_JIT_OPERATOR_R0_ABI_VERSION_V3 4u" in abi
    assert "SYMMETRIX_JIT_OPERATOR_R0_ABI_VERSION_V4 5u" in abi
    assert "SYMMETRIX_JIT_OPERATOR_KIND_M1_V1 3u" in abi
    assert "typedef SymmetrixJitM0ArgsV1 SymmetrixJitM1ArgsV1;" in abi
    assert "SYMMETRIX_JIT_OPERATOR_M1_ACCUMULATE_INPUT_ADJOINT_V1" in abi
    assert '"symmetrix.jit.m1-host-plugin/2"' in m1_host_abi
    assert "symmetrix_jit_m1_host_plugin_query_v2" in m1_host_abi
    assert "required_jit_generation_version = 19" in native_version
    assert "JIT_GENERATION_VERSION = 19" in python_version
    runtime = (NATIVE / "mace_kokkos_runtime.cpp").read_text()
    module_loader = (NATIVE / "mace_jit_operator_module.cpp").read_text()
    assert "&~SYMMETRIX_JIT_OPERATOR_M1_ACCUMULATE_INPUT_ADJOINT_V1" in module_loader
    assert 'backend == "cuda" ? "cuda-fast-math" : "strict"' in runtime


@pytest.mark.parametrize(
    "filename,normalizer,interaction",
    (
        ("standard_m0_contract.json", normalize_standard_m0_contract, "M0"),
        ("standard_r0_contract.json", normalize_standard_r0_contract, "R0"),
        ("jit_r1_contract.json", normalize_jit_r1_contract, "R1"),
        (
            "jit_r1_fixture_c4_e3_l0_contract.json",
            normalize_jit_r1_contract,
            "R1",
        ),
    ),
)
def test_execution_contract_fixtures_are_canonical(filename, normalizer, interaction):
    contract = json.loads((CONTRACTS / filename).read_text())

    assert contract["interaction"] == interaction
    assert normalizer(contract) == contract


def test_generated_m1_contract_is_scalar_and_architecture_sensitive():
    contract = make_generated_m1_contract(
        channels=48,
        type_count=3,
        input_l_max=2,
        correlation=3,
        monomials=[[0], [1, 1], [0, 3, 8]],
    )

    assert normalize_generated_m1_contract(contract) == contract
    assert contract["interaction"] == "M1"
    assert contract["input_components"] == 9
    assert contract["output_components"] == 1
    assert contract["monomial_groups"] == [
        {
            "output_component": 0,
            "term_count": 3,
            "degree_counts": [1, 1, 1],
            "terms": [[0], [1, 1], [0, 3, 8]],
        }
    ]

    changed = make_generated_m1_contract(
        channels=64,
        type_count=3,
        input_l_max=2,
        correlation=3,
        monomials=[[0], [1, 1], [0, 3, 8]],
    )
    assert changed["semantic_fingerprint"] != contract["semantic_fingerprint"]
    assert changed["structure_fingerprint"] == contract["structure_fingerprint"]


def test_foundation_registry_tracks_production_operator_bundles():
    registry = json.loads((CONTRACTS / "foundation_model_registry.json").read_text())
    production = load_foundation_operator_manifest()

    assert registry["schema"] == "symmetrix.foundation-model-registry"
    assert registry["version"] == 2
    assert len(registry["models"]) == 5
    assert registry["aliases"] == production["aliases"]
    production_models = {model["id"]: model for model in production["models"]}
    production_bundles = {bundle["id"]: bundle for bundle in production["bundles"]}
    for model in registry["models"]:
        expected = production_models[model["id"]]
        assert {key: model[key] for key in expected} == expected
        contract_path = CONTRACTS / model["r1_contract"]
        contract = json.loads(contract_path.read_text())
        assert normalize_jit_r1_contract(contract) == contract
        production_r1 = production_bundles[model["bundle"]]["operators"]["R1"]
        assert r1_operator_descriptor(contract) == {
            key: value for key, value in production_r1.items() if key != "contract"
        }
        assert contract == production_r1["contract"]


def test_m0_standard_module_owns_the_fixed_lmax3_topology():
    contract = json.loads((CONTRACTS / "standard_m0_contract.json").read_text())
    source = (NATIVE / "mace_standard_m0.hpp").read_text()

    assert "namespace symmetrix::standard_m0" in source
    assert "deprecated_for_new_specializations = true" in source
    assert "use RTC for new contracts" in source
    assert 'module_id[] = "standard-m0-module-module-v1"' in source
    assert 'scalar_module_id[] = "standard-m0-lmax0-module-v1"' in source
    assert "inline constexpr int module_revision = 2;" in source
    assert "inline constexpr int total_terms = 422;" in source
    assert "inline constexpr int scalar_total_terms = 94;" in source
    assert "inline constexpr int input_l_max = 3;" in source
    assert "inline constexpr int output_l_max = 1;" in source
    assert "inline constexpr int channels = 0;" in source
    assert "inline constexpr int type_count = 0;" in source
    assert "inline constexpr bool uses_runtime_weights = true;" in source
    assert "inline constexpr bool dynamic_channels = true;" in source
    assert "inline constexpr bool dynamic_type_count = true;" in source
    assert "inline constexpr int default_host_channel_tile = 16;" in source
    assert 'std::string_view(value) == "1"' in source
    assert "launch_forward_host_tiled<16>" in source
    assert "launch_reverse_host_tiled<16>" in source
    assert "std::make_index_sequence<scalar_total_terms>" in source
    assert "launch_scalar_forward(" in source
    assert "launch_scalar_reverse(" in source
    assert "{0, 94, 212, 304, 422}" in source
    assert [group["term_count"] for group in contract["monomial_groups"]] == [
        94,
        118,
        92,
        118,
    ]
    assert source.count("StandardM0::") == 6
    assert "_generated" not in source
    assert "artifact_id" not in source


def test_standard_m1_reuses_fixed_m0_terms_with_host_tiling():
    source = (NATIVE / "mace_standard_m1.hpp").read_text()
    model = (NATIVE / "mace_kokkos_model.cpp").read_text()
    runtime = (NATIVE / "mace_kokkos_runtime.cpp").read_text()
    second_interaction = (NATIVE / "mace_kokkos_second_interaction.cpp").read_text()

    assert "standard_m0::term_offsets[1]" in source
    assert "const int channel_count" in source
    assert "const WeightsView weights" in source
    assert "standard_m0::accumulate_forward_terms<Tile>" in source
    assert "standard_m0::accumulate_reverse_terms<Tile>" in source
    assert "StandardM1::reverse_device_direct" in source
    assert "StandardM1::reverse_device_overlap_safe" in source
    assert "StandardM1::reverse_device_scale_adjoint" in source
    assert "std::is_same_v<ExecutionSpace,Kokkos::Cuda>" in source
    assert "views_overlap(input, input_adjoint)" in source
    assert "right_begin-left_begin < left_bytes" in source
    assert "std::make_index_sequence<total_terms>" in source
    assert "InputView::non_const_value_type,float" in source
    assert 'std::getenv("SYMMETRIX_STANDARD_M1_HOST_TILE")' in source
    assert 'std::string_view(value) == "runtime"' in source
    assert "StandardM1::forward_host_tiled" in source
    assert "StandardM1::reverse_host_tiled" in source
    assert "standard_m1::supports_backend" in model
    assert "standard_m1_module_ready = !has_field_coupling" not in model
    assert "(!has_field_coupling || host_backend)" in model
    assert "!has_field_coupling && standard_m1_module_ready" in runtime
    assert (
        """if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (!has_field_coupling && standard_m1_module_ready"""
        in runtime
    )
    assert "canonical_m1_rows[term]" in model
    assert "symmetrix::standard_m1::launch_forward(" in second_interaction
    assert "symmetrix::standard_m1::launch_reverse(" in second_interaction
    assert "standard_m1_module_forward_launch_count += 1" in second_interaction
    assert "standard_m1_module_reverse_launch_count += 1" in second_interaction
    runtime = (NATIVE / "mace_kokkos_runtime.cpp").read_text()
    assert "return generic_reverse/2;" in runtime
    assert "A1.data() != A1_adj.data() || A1.span() != A1_adj.span()" in runtime
    assert "input.data() != input_adjoint.data()" in source
    assert "device_overlap_channel_tile" in source
    assert "artifact_id" not in source


def test_m1_recompute_uses_one_gpu_team_thread_per_channel_wave():
    second_interaction = (NATIVE / "mace_kokkos_second_interaction.cpp").read_text()

    policy = """auto policy = vector_length == 1
            ? Kokkos::TeamPolicy<>(
                execution_space, num_nodes, Kokkos::AUTO, 1)
            : Kokkos::TeamPolicy<>(
                execution_space, num_nodes, 1, vector_length);"""
    assert second_interaction.count(policy) == 2


def test_generated_r1_dispatch_is_prepared_execution_only():
    header = (NATIVE / "mace_kokkos.hpp").read_text()
    evaluator = (NATIVE / "mace_kokkos_evaluate.cpp").read_text()
    bindings = (ROOT / "symmetrix" / "source" / "cpp" / "mace_kokkos.cpp").read_text()

    assert "use_generated_direct_r1_inference" not in header
    assert "use_generated_direct_r1_inference" not in evaluator
    assert "use_generated_direct_r1_inference" not in bindings
    normalized_evaluator = " ".join(evaluator.split())
    assert (
        normalized_evaluator.count(
            "uses_prepared_execution(streamed_edges) "
            "&& use_factorized_direct_inference()"
        )
        == 2
    )


def test_r0_standard_module_owns_scalar_source_execution():
    contract = json.loads((CONTRACTS / "standard_r0_contract.json").read_text())
    source = (NATIVE / "mace_standard_r0.hpp").read_text()

    assert "namespace symmetrix::standard_r0" in source
    assert "deprecated_for_new_specializations = true" in source
    assert "use RTC for new contracts" in source
    assert 'module_id[] = "standard-r0-module-module-v2"' in source
    assert "inline constexpr int module_revision = 4;" in source
    assert "inline constexpr int harmonic_count = 16;" in source
    assert "inline constexpr int l_max = 3;" in source
    assert "inline constexpr int channels = 0;" in source
    assert "inline constexpr bool dynamic_channels = true;" in source
    assert "inline constexpr int forward_host_channel_tile = 8;" in source
    assert "inline constexpr int coordinate_reverse_host_channel_tile = 128;" in source
    assert [path["source_l"] for path in contract["paths"]] == [0, 0, 0, 0]
    assert "StandardR0::" in source
    assert "launch_density_prepare(" in source
    assert "launch_forward(" in source
    assert "harmonic_count*forward_host_channel_tile" in source
    assert "typename Member::scratch_memory_space" in source
    assert "accumulator(6,channel) +=" in source
    assert "launch_coordinate_reverse(" in source
    assert source.count("radius(edge) < cutoff") >= 7
    assert "_generated" not in source
    assert "artifact_id" not in source


def test_accelerator_r0_forward_args_use_shared_device_abi_lowering(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the device ABI syntax probe")

    source = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()
    assert "const SymmetrixJitR0ArgsV1 args{" not in source
    assert "lower_r0_device_arguments(r0_arguments)" in source

    probe = tmp_path / "r0_device_args.cpp"
    probe.write_text(
        '#include "mace_jit_r0_arguments.hpp"\n'
        "int main() {\n"
        "  symmetrix::execution::R0LogicalArguments<double> logical{};\n"
        "  auto args = symmetrix::execution::lower_r0_device_arguments(logical);\n"
        "  return static_cast<int>(args.struct_size);\n"
        "}\n"
    )
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-fsyntax-only",
            str(probe),
            f"-I{NATIVE}",
            f"-I{ROOT / 'symmetrix' / 'source' / 'symmetrix'}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_r0_operator_loader_validates_profile_and_schedule_launch_geometry():
    header = (NATIVE / "mace_jit_operator_module.hpp").read_text()
    loader = (NATIVE / "mace_jit_operator_module.cpp").read_text()
    runtime = (NATIVE / "mace_kokkos_runtime.cpp").read_text()
    normalized_loader = " ".join(loader.split())
    normalized_runtime = " ".join(runtime.split())

    for field in (
        "schedule_revision",
        "persistent_blocks_per_compute_unit",
        "r0_channels",
        "r0_l_max",
        "r0_density_extent_multiplier",
        "r0_density_owners_per_block",
        "r0_density_threads_per_block",
        "r0_density_persistent",
        "r0_forward_extent_multiplier",
        "r0_forward_owners_per_block",
        "r0_forward_threads_per_block",
        "r0_forward_persistent",
        "r0_reverse_prepare_extent_multiplier",
        "r0_reverse_prepare_owners_per_block",
        "r0_reverse_prepare_threads_per_block",
        "r0_reverse_prepare_persistent",
        "r0_coordinate_reverse_extent_multiplier",
        "r0_coordinate_reverse_owners_per_block",
        "r0_coordinate_reverse_threads_per_block",
        "r0_coordinate_reverse_persistent",
    ):
        assert f"std::int32_t {field}" in header
        assert f"state->{field} = read_value<std::int32_t>" in normalized_loader
        assert f"state.{field} != expectation.{field}" in normalized_loader

    assert "std::string r0_schedule" in loader
    assert "state->r0_schedule = read_text" in normalized_loader
    for field in ("r0_harmonic_count", "r0_edge_weight_mode"):
        assert f"std::int32_t {field}" in loader
        assert f"state->{field} = read_value<std::int32_t>" in normalized_loader
    assert "r0_descriptor_mismatch" in loader

    assert (
        "work_items, owners_per_block, persistent ? persistent_blocks : INT32_MAX"
        in normalized_loader
    )
    assert "args.harmonic_gradients == nullptr" in normalized_loader
    assert "gradient_scratch_bytes" in normalized_loader
    for extent, field in (
        ("args.num_nodes", "r0_density_extent_multiplier"),
        ("args.num_nodes", "r0_forward_extent_multiplier"),
        ("args.num_nodes", "r0_reverse_prepare_extent_multiplier"),
        ("args.num_edges", "r0_coordinate_reverse_extent_multiplier"),
    ):
        assert (
            f"static_cast<std::int64_t>({extent}) *state_->{field}" in normalized_loader
        )
    assert "state_->r0_forward_owners_per_block" in normalized_loader
    assert "state_->r0_reverse_prepare_owners_per_block" in normalized_loader
    assert "state_->r0_coordinate_reverse_owners_per_block" in normalized_loader
    assert "expected_r0_persistent_blocks_per_compute_unit" in runtime
    assert (
        "persistent_blocks_per_compute_unit "
        "!= expected_r0_persistent_blocks_per_compute_unit" in normalized_runtime
    )


def test_r0_logical_contract_rejects_each_one_field_mismatch(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the R0 contract probe")

    probe = tmp_path / "r0_logical_contract.cpp"
    probe.write_text(
        r"""
#include "mace_jit_r0_contract.hpp"

#include <array>
#include <cassert>
#include <string_view>

using symmetrix::execution::R0LogicalDescriptor;
using symmetrix::execution::r0_descriptor_mismatch;

int main() {
    const R0LogicalDescriptor expected{
        "artifact", "structure", "target", "receiver-edge16", 2u, 8u,
        0x7fu, 48, 2, 9, 1, 4, 2};
    assert(r0_descriptor_mismatch(expected, expected).empty());

    std::array<std::string_view, 13> mismatches{};
    std::size_t index = 0;
    auto check = [&](R0LogicalDescriptor actual, std::string_view field) {
        mismatches[index++] = r0_descriptor_mismatch(actual, expected);
        assert(mismatches[index-1] == field);
    };
    auto actual = expected;
    actual.artifact_id = "other"; check(actual, "artifact_id");
    actual = expected; actual.structure_fingerprint = "other";
    check(actual, "structure_fingerprint");
    actual = expected; actual.target = "other"; check(actual, "target");
    actual = expected; actual.schedule = "receiver-edge32"; check(actual, "schedule");
    actual = expected; actual.scalar_kind = 1u; check(actual, "scalar_kind");
    actual = expected; actual.scalar_size = 4u; check(actual, "scalar_size");
    actual = expected; actual.capabilities &= ~1u; check(actual, "capabilities");
    actual = expected; actual.channels = 47; check(actual, "channels");
    actual = expected; actual.l_max = 3; check(actual, "l_max");
    actual = expected; actual.harmonic_count = 8; check(actual, "harmonic_count");
    actual = expected; actual.edge_weight_mode = 0; check(actual, "edge_weight_mode");
    actual = expected; actual.schedule_revision = 3;
    check(actual, "schedule_revision");
    actual = expected; actual.compatibility_version = 1;
    check(actual, "compatibility_version");
}
"""
    )
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            str(probe),
            f"-I{NATIVE}",
            f"-I{ROOT / 'symmetrix' / 'source' / 'symmetrix'}",
            "-o",
            str(tmp_path / "r0_logical_contract"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    run = subprocess.run(
        [str(tmp_path / "r0_logical_contract")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr


def test_r0_native_arguments_share_slicing_layout_and_adjoint_lowering(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the R0 argument probe")

    probe = tmp_path / "r0_logical_arguments.cpp"
    probe.write_text(
        r"""
#include "mace_jit_r0_arguments.hpp"

#include <cassert>

int main() {
    int nodes[2]{}, counts[2]{}, first[2]{}, neighbours[11]{}, receivers[11]{};
    int active[2]{};
    double coordinates[33]{}, radius[11]{}, radial_coefficients[7]{};
    double density_coefficients[5]{}, harmonics[99]{}, gradients[297]{};
    double output[18]{}, adjoint[18]{}, density_state[2]{}, scale_adjoint[2]{};
    double forces[33]{}, weights[11]{};
    symmetrix::execution::R0LogicalArguments<double> logical{};
    logical.receiver_base = 4;
    logical.num_nodes = 2;
    logical.num_edges = 3;
    logical.edge_begin = 5;
    logical.active_type_count = 2;
    logical.channels = 2;
    logical.l_max = 2;
    logical.harmonic_count = 9;
    logical.coordinates_are_unit = true;
    logical.apply_density_scale = true;
    logical.use_precomputed_scale_adjoint = true;
    logical.node_types = nodes;
    logical.num_neigh = counts;
    logical.first_neigh = first;
    logical.neigh_types = neighbours;
    logical.edge_receivers = receivers;
    logical.type_to_active = active;
    logical.coordinates = coordinates;
    logical.radius = radius;
    logical.radial = {2, 7, 3, 0.25, -1.0, radial_coefficients};
    logical.density = {1, 5, 1, 0.5, 0.0, density_coefficients};
    logical.harmonics = harmonics;
    logical.harmonic_gradients = gradients;
    logical.output = output;
    logical.output_adjoint = adjoint;
    logical.density_state = density_state;
    logical.precomputed_scale_adjoint = scale_adjoint;
    logical.directed_forces = forces;
    logical.cutoff = 6.0;
    logical.edge_weights = weights;

    const auto host = symmetrix::execution::lower_r0_host_arguments(logical);
    const auto device = symmetrix::execution::lower_r0_device_arguments(logical);
    assert(host.radial.intervals == 7 && device.radial.intervals == 7);
    assert(host.density.intervals == 5 && device.density.intervals == 5);
    assert(host.neigh_types == neighbours+5 && device.neigh_types == neighbours+5);
    assert(host.edge_receivers == receivers+5 && device.edge_receivers == receivers+5);
    assert(host.coordinates == coordinates+15 && device.coordinates == coordinates+15);
    assert(host.radius == radius+5 && device.radius == radius+5);
    assert(host.harmonics == harmonics+45 && device.harmonics == harmonics+45);
    assert(host.harmonic_gradients == gradients+135);
    assert(device.harmonic_gradients == gradients+135);
    assert(host.output_adjoint == adjoint && device.output_adjoint == adjoint);
    assert(host.precomputed_scale_adjoint == scale_adjoint);
    assert(device.precomputed_scale_adjoint == scale_adjoint);
    assert(host.directed_forces == forces+15 && device.directed_forces == forces+15);
    assert(host.edge_weights == weights+5 && device.edge_weights == weights+5);
    assert(host.edge_begin == 5);
}
"""
    )
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            str(probe),
            f"-I{NATIVE}",
            f"-I{ROOT / 'symmetrix' / 'source' / 'symmetrix'}",
            "-o",
            str(tmp_path / "r0_logical_arguments"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    run = subprocess.run(
        [str(tmp_path / "r0_logical_arguments")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr

    first_interaction = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()
    assert "R0LogicalArguments<Precision>" in first_interaction
    assert "lower_r0_host_arguments" in first_interaction
    assert "lower_r0_device_arguments" in first_interaction


def test_r0_builtin_paths_consume_the_shared_logical_arguments():
    first_interaction = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()
    arguments = (NATIVE / "mace_jit_r0_arguments.hpp").read_text()
    adapters = (NATIVE / "mace_standard_r0.hpp").read_text()
    normalized = " ".join(first_interaction.split())

    for launch in (
        "launch_density_prepare<ApplyEdgeWeights>",
        "launch_forward<ApplyEdgeWeights>",
        "launch_forward_all_l<ApplyEdgeWeights>",
        "launch_coordinate_reverse<ApplyEdgeWeights>",
        "launch_coordinate_reverse_edge16<ApplyEdgeWeights>",
        "launch_coordinate_reverse_edge32<ApplyEdgeWeights>",
    ):
        assert f"{launch}( execution_space" in normalized
        assert "r0_arguments);" in normalized
    assert (
        "launch_reverse_prepare( factorized_execution_space, r0_arguments);"
        in normalized
    )
    assert "type_to_active_count" in arguments
    assert first_interaction.count(".type_to_active_count =") == 3
    assert "args.type_to_active_count" in adapters


def test_r0_builtin_nonzero_interval_clips_partial_receiver_rows(tmp_path):
    kokkos_package = ROOT / "symmetrix" / "build" / "cmake_packages" / "Kokkos"
    if not (kokkos_package / "KokkosConfig.cmake").is_file():
        pytest.skip("the CPU development build is required for the builtin R0 probe")

    source = Path(__file__).with_name("cpp") / "r0_builtin_interval.cpp"
    project = tmp_path / "CMakeLists.txt"
    project.write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(r0_builtin_interval LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Kokkos CONFIG REQUIRED PATHS {kokkos_package} NO_DEFAULT_PATH)
add_executable(r0_builtin_interval {source})
target_include_directories(r0_builtin_interval PRIVATE
    {NATIVE}
    {ROOT / "libsymmetrix" / "external" / "sphericart" / "sphericart" / "include"}
    {ROOT / "symmetrix" / "source" / "symmetrix"})
target_link_libraries(r0_builtin_interval PRIVATE Kokkos::kokkos)
"""
    )
    build = tmp_path / "build"
    configured = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert configured.returncode == 0, configured.stderr
    compiled = subprocess.run(
        ["cmake", "--build", str(build), "-j2"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run(
        [str(build / "r0_builtin_interval")],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false"},
    )
    assert run.returncode == 0, run.stderr


def test_r0_host_plugin_launches_owner_ranges_instead_of_individual_owners():
    first_interaction = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()

    for stage in ("forward", "coordinate_reverse", "reverse_prepare"):
        assert f"descriptor.{stage}_range" in first_interaction
    assert "owner(&args, item)" not in first_interaction
    assert "owner(&args, receiver)" not in first_interaction
    assert "owner(&args, node)" not in first_interaction


def test_direct_geometry_assigns_an_exact_inactive_radius_sentinel():
    lifecycle = (NATIVE / "mace_kokkos_factorized_lifecycle.cpp").read_text()

    assert "r(edge) = distance >= cutoff ? cutoff : distance;" in lifecycle
    assert "r(edge) = scale*distance;" not in lifecycle


def test_host_h2_precision_path_is_cpu_bounded():
    header = (NATIVE / "mace_kokkos.hpp").read_text()
    model = (NATIVE / "mace_kokkos_model.cpp").read_text()
    second_interaction = (NATIVE / "mace_kokkos_second_interaction.cpp").read_text()

    assert "using H2WeightPrecision = std::conditional_t<" in header
    assert "Kokkos::View<H2WeightPrecision**" in header
    assert "static_cast<H2WeightPrecision>" in model
    assert "const Precision adjoint =" in second_interaction
    assert "H2_weights_for_H1_reverse" in header
    assert "H2_weights_for_M1_reverse" in header
    assert "H2_weights_for_H1_vec[type][k*num_channels+kp]" in model
    assert "H2_weights_for_M1_vec[k*num_channels+kp]" in model
    assert "H2_weights_for_H1_reverse(" in second_interaction
    assert "node_types(i),kp*num_channels+k" in second_interaction
    assert "H2_weights_for_M1_reverse(" in second_interaction
    assert "typename Kokkos::DefaultExecutionSpace::memory_space" in (
        second_interaction
    )


def test_standard_m0_tracks_input_scale_adjoint_launches():
    header = (NATIVE / "mace_kokkos.hpp").read_text()
    first_interaction = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()
    bindings = (ROOT / "symmetrix" / "source" / "cpp" / "mace_kokkos.cpp").read_text()

    assert "standard_m0_input_scale_adjoint_launch_count = 0" in header
    assert "const bool capture_input_scale_adjoint =" in first_interaction
    assert "standard_m0_input_scale_adjoint_launch_count += 1" in first_interaction
    assert '"standard_m0_input_scale_adjoint_launch_count"' in bindings


def test_host_a1_gates_blas_without_replacing_device_team_gemm():
    source = (NATIVE / "mace_kokkos_second_interaction.cpp").read_text()

    assert '#include "cblas.hpp"' in source
    # One admitted-CBLAS call per direction in the blocked owner kernel and one
    # in the per-node fallback; the device TeamGEMM path is untouched.
    assert source.count("symmetrix_blas_gemm<Precision>(") == 4
    assert source.count("KokkosBatched::TeamGemm<") >= 2
    assert source.count("symmetrix::host_worker_cblas_enabled()") >= 4
    assert source.count("typename Kokkos::DefaultExecutionSpace::memory_space") >= 3


def test_single_worker_host_batches_dense_linears():
    h1 = (NATIVE / "mace_kokkos_h1_phi1.cpp").read_text()
    second_interaction = (NATIVE / "mace_kokkos_second_interaction.cpp").read_text()
    readout = (NATIVE / "host_batched_readout.hpp").read_text()

    # Multi-worker hosts keep the per-node worker GEMMs; one worker batches
    # each lm component (H1) and each node type (H2) into whole GEMMs.
    assert "if (execution_space.concurrency() == 1) {" in h1
    assert "num_nodes, num_channels, num_channels," in h1
    assert second_interaction.count("execution_space.concurrency() == 1") == 2
    assert "symmetrix::host_batched_h2_forward<Precision>(" in second_interaction
    assert "symmetrix::host_batched_h2_reverse<Precision>(" in second_interaction
    assert readout.count("symmetrix_blas_gemm<Precision>(") == 4
    mlp = (NATIVE / "multilayer_perceptron_kokkos.cpp").read_text()
    assert "if (execution_space.concurrency() == 1) {" in mlp
    assert mlp.count("symmetrix_blas_gemm<double>(") == 2


def test_host_dense_backend_uses_qualified_blas_for_parallel_openmp():
    policy = (NATIVE / "host_worker_blas.hpp").read_text()
    dense = (NATIVE / "host_dense_kernels.hpp").read_text()
    h1 = (NATIVE / "mace_kokkos_h1_phi1.cpp").read_text()
    second_interaction = (NATIVE / "mace_kokkos_second_interaction.cpp").read_text()

    assert 'std::getenv("SYMMETRIX_HOST_WORKER_BLAS")' in policy
    assert '#include "cblas.hpp"' in policy
    assert 'std::string_view("automatic")' in policy
    assert 'value == "off"' in policy
    assert 'value == "unsafe"' in policy
    assert "symmetrix_blas_allows_concurrent_cblas()" in policy
    assert 'provider == "mkl"' in policy
    assert 'provider == "openblas"' in policy
    assert "omp_get_max_active_levels() <= 1" in policy
    assert "omp_get_max_threads() <= 1" in policy
    assert "Kokkos::DefaultExecutionSpace().concurrency() <= 1" in policy
    cblas = (NATIVE / "cblas.hpp").read_text()
    assert "symmetrix_blas_symbol_matches_cblas_provider" in cblas
    assert "symmetrix_blas_provider_links_process_openmp_runtime" in cblas
    assert "cblas_dgemm" in cblas
    assert "dli_fbase" in cblas
    assert h1.count("symmetrix::host_worker_cblas_enabled()") == 6
    assert 'std::getenv("SYMMETRIX_HOST_DENSE_BACKEND")' in dense
    assert 'value == "flat"' in dense
    assert 'value == "team"' in dense
    assert "return !host_worker_cblas_enabled();" in dense
    assert 'return "cblas";' in dense
    assert "#pragma omp simd" in dense
    assert h1.count("symmetrix::host_flat_dense_enabled()") == 6
    assert second_interaction.count("symmetrix::host_flat_dense_enabled()") == 4

    # The host owner-computes A1 blocks stay inside the admitted-CBLAS gate, so
    # a pthread OpenBLAS deployment never issues a BLAS call inside a worker.
    assert 'std::getenv("SYMMETRIX_A1_HOST_BLOCKED")' in dense
    assert "inline bool host_a1_blocked_enabled()" in dense
    assert '"SYMMETRIX_A1_HOST_BLOCKED must be' in dense
    # Each blocked launch names its region and its scratch-admission request.
    assert second_interaction.count('"Compute A1 host blocked"') == 2
    assert second_interaction.count('"Reverse A1 host blocked"') == 2
    assert "make_host_a1_block_plan<Precision>(" in second_interaction
    tools = (NATIVE / "tools_kokkos.hpp").read_text()
    assert "std::size_t admitted_team_scratch_limit_bytes(" in tools
    assert "admitted_team_scratch_limit_bytes<>(1)" in second_interaction
    gate = "if (symmetrix::host_worker_cblas_enabled() && !symmetrix::host_dense_backend_overrides_cblas()) {"
    assert " ".join(second_interaction.split()).count(gate) == 2
    blocked_forward = second_interaction.index('"Compute A1 host blocked"')
    per_node_forward = second_interaction.index('"Compute A1 host BLAS"')
    blocked_reverse = second_interaction.index('"Reverse A1 host blocked"')
    per_node_reverse = second_interaction.index('"Reverse A1 host BLAS"')
    assert second_interaction.index("host_worker_cblas_enabled()") < blocked_forward
    assert blocked_forward < per_node_forward < blocked_reverse < per_node_reverse


@pytest.mark.parametrize(
    "provider_openmp_link,expected",
    (("fake", "incompatible"), ("process", "compatible")),
)
def test_openblas_openmp_runtime_must_match_process(
    tmp_path, provider_openmp_link, expected
):
    if sys.platform != "linux":
        pytest.skip("the provider DT_NEEDED regression test requires Linux ELF")
    cc = shutil.which("gcc")
    cxx = shutil.which("g++")
    if cc is None or cxx is None:
        pytest.skip("the provider DT_NEEDED regression test requires GCC")

    fake_openmp = tmp_path / "fake_openmp.c"
    fake_openmp.write_text("int omp_get_max_threads(void) { return 17; }\n")
    subprocess.run(
        [
            cc,
            "-shared",
            "-fPIC",
            str(fake_openmp),
            "-o",
            str(tmp_path / "libfakeomp.so"),
        ],
        check=True,
    )

    provider = tmp_path / "provider.c"
    provider.write_text(
        """
extern int omp_get_max_threads(void);
void cblas_dgemm(void) {}
const char* openblas_get_config(void) {
    return omp_get_max_threads() ? "OpenBLAS fake USE_OPENMP" : "";
}
int openblas_get_parallel(void) { return 2; }
int openblas_get_num_threads(void) { return omp_get_max_threads(); }
void openblas_set_num_threads(int threads) { (void)threads; }
""".lstrip()
    )
    provider_library = tmp_path / f"libprovider_{provider_openmp_link}.so"
    provider_command = [cc, "-shared", "-fPIC", str(provider)]
    if provider_openmp_link == "fake":
        provider_command.extend(
            [
                f"-L{tmp_path}",
                "-Wl,--no-as-needed",
                "-lfakeomp",
                f"-Wl,-rpath,{tmp_path}",
            ]
        )
    else:
        provider_command.append("-fopenmp")
    provider_command.extend(("-o", str(provider_library)))
    subprocess.run(provider_command, check=True)

    probe = tmp_path / "probe.cpp"
    probe.write_text(
        """
#include <omp.h>
#include <iostream>
#include "cblas.hpp"

int main() {
        const bool compatible =
        symmetrix_blas_provider_links_process_openmp_runtime();
    std::cout << (compatible ? "compatible" : "incompatible") << '\\n';
}
""".lstrip()
    )
    executable = tmp_path / f"probe_{provider_openmp_link}"
    subprocess.run(
        [
            cxx,
            "-std=c++20",
            "-fopenmp",
            str(probe),
            f"-I{NATIVE}",
            "-Wl,--no-as-needed",
            "-lgomp",
            f"-L{tmp_path}",
            f"-lprovider_{provider_openmp_link}",
            f"-Wl,-rpath,{tmp_path}",
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
    )
    completed = subprocess.run(
        [executable],
        check=True,
        capture_output=True,
        env={**os.environ, "LD_PRELOAD": str(provider_library)},
        text=True,
    )

    assert completed.stdout.strip() == expected


def test_launch_profiles_are_source_owned_and_module_bounded():
    source = (NATIVE / "kernel_launch_profile.hpp").read_text()

    assert "std::array<LaunchProfile,23>" in source
    assert source.count("{m0_module_id,") == 6
    assert source.count("{m0_scalar_module_id,") == 6
    assert source.count("{r0_module_id,") == 7
    assert source.count("{field_h1_reverse_id,") == 4
    for backend in ("cuda", "hip", "host"):
        for precision in ("float32", "float64"):
            assert (
                f"standard-r0-module-module-v2-{backend}-{precision}-generic" in source
            )
    assert "standard-r0-module-module-v2-hip-float32-gfx1151" in source
    assert "select_launch_profile(" in source
    assert "is_calibration_candidate(" in source
    assert "registry" not in source


def test_model_admission_and_launches_dispatch_selected_modules():
    model = (NATIVE / "mace_kokkos_model.cpp").read_text()
    first_interaction = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()
    response = (NATIVE / "mace_kokkos_response.tpp").read_text()

    assert "symmetrix::standard_m0::matches_structure(" in model
    assert "symmetrix::standard_m0::matches_scalar_structure(" in model
    assert "l_max == symmetrix::standard_m0::input_l_max" in model
    assert "L_max == symmetrix::standard_m0::output_l_max" in model
    assert "l_max == symmetrix::standard_r0::l_max" in model
    assert "symmetrix::standard_r0::structure_fingerprint" in model
    assert "symmetrix::standard_m0::launch_forward(" in first_interaction
    assert "symmetrix::standard_m0::launch_scalar_forward(" in first_interaction
    assert "symmetrix::standard_m0::launch_reverse(" in first_interaction
    assert "symmetrix::standard_m0::launch_scalar_reverse(" in first_interaction
    assert (
        "symmetrix::standard_r0::launch_forward<ApplyEdgeWeights>(" in first_interaction
    )
    assert (
        "symmetrix::standard_r0::launch_forward_all_l<ApplyEdgeWeights>("
        in first_interaction
    )
    assert "use_fused_r0_forward = single_layer_readout" in first_interaction
    assert "launch_M0_module_reverse(" in response
    assert "use_m0_module()" in response
    for source in (model, first_interaction, response):
        assert "execution_artifacts" not in source
        assert "execution_artifact_registry" not in source


def test_generated_r0_loader_binds_the_requested_schedule_identity():
    header = (NATIVE / "mace_kokkos.hpp").read_text()
    runtime = (NATIVE / "mace_kokkos_runtime.cpp").read_text()

    assert "bool apply_edge_weights = false,\n    std::string schedule" in header
    assert 'schedule != "receiver-edge16" && schedule != "receiver-edge32"' in runtime
    assert "constexpr int r0_device_schedule_revision = 6;" in runtime
    assert (
        'apply_edge_weights ? "weighted" : "unity", schedule,\n'
        "            r0_device_schedule_revision);" in runtime
    )


def test_r0_host_plugin_launches_are_compile_time_host_only():
    source = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()

    assert (
        source.count(
            "if constexpr (symmetrix::execution::host_execution_space<\n"
            "                Kokkos::DefaultExecutionSpace>)"
        )
        >= 5
    )


def test_generic_standard_r0_reverse_uses_and_retains_generic_receiver_map():
    first_interaction = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()
    lifecycle = (NATIVE / "mace_kokkos_factorized_lifecycle.cpp").read_text()
    normalized_first_interaction = " ".join(first_interaction.split())

    assert (
        "edge_receivers = streamed_edges == ExecutionMode::generic "
        "? streamed_edge_receivers : execution_edge_receivers;"
        in normalized_first_interaction
    )
    assert ".edge_receivers = edge_receivers.data()," in first_interaction
    assert "args.edge_receivers" in (NATIVE / "mace_standard_r0.hpp").read_text()
    assert (
        """streamed_edges != ExecutionMode::generic
        && !supports_fused_streamed_reverse()"""
        in lifecycle
    )


def test_static_aot_authoring_surface_is_retired():
    for filename in (
        "generate_execution_artifact_registry.py",
        "generate_standard_m0_kernels.py",
        "generate_standard_r0_kernels.py",
        "generate_factorized_kernels.py",
        "generate_factorized_model_bundle.py",
    ):
        assert not (ROOT / "libsymmetrix" / "tools" / filename).exists()
    assert not (NATIVE / "generated").exists()
    assert not (ROOT / "benchmarks" / "execution_generated_stream_fixture.cpp").exists()


_DIRECT_REFERENCE_PARITY_SCRIPT = r"""
import json, os, sys
import numpy as np
from ase import Atoms
from compact_r1_model import foundation_r1_test_model
from symmetrix import Symmetrix
from symmetrix.mace_execution_contract import make_generated_m1_contract

contracts, model_path, dtype = sys.argv[1], sys.argv[2], sys.argv[3]
r1_contract = json.loads(
    open(os.path.join(contracts, "jit_r1_mace_omat0_small_contract.json")).read()
)
# Direct execution runs the generated plan, which is defined against an
# Execution M0 contract and the standard R0 first interaction, so the parity
# model carries both; the three fingerprints are the canonical ones the
# standard M0 contract declares.
m0_contract = json.loads(
    open(os.path.join(contracts, "standard_m0_contract.json")).read()
)
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
r0_contract = json.loads(
    open(os.path.join(contracts, "standard_r0_contract.json")).read()
)
model = foundation_r1_test_model(
    r1_contract,
    atomic_numbers=(7, 13),
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
# The helper's weights are diagonal or zero; random dense, per-type distinct
# weights make transposes, pitches, and type indexing observable.
rng = np.random.default_rng(11)
channels = model["num_channels"]
def randomize(values, scale):
    if isinstance(values, dict):
        return {name: randomize(value, scale) for name, value in values.items()}
    values = np.asarray(values, dtype=float)
    return (scale * rng.standard_normal(values.shape)).tolist()
for key in ("H0_weights", "A0_weights", "H1_weights", "A1_weights",
            "H2_weights_for_H1", "H2_weights_for_M1", "readout_1_weights"):
    model[key] = randomize(model[key], 1.0 / np.sqrt(channels))
for key in ("M0_weights", "M1_weights"):
    model[key] = randomize(model[key], 0.5)
# Density-scaled A0 and A1 select the scaled host paths.
embedding = model["compact_radial"]["networks"]["R0"]["shape"][0]
for name in ("A0", "A1"):
    model["compact_radial"]["networks"][name] = {
        "shape": [embedding, 1],
        "weights": [randomize(np.zeros(embedding), 0.5)],
        "activation": "silu",
        "activation_scale": 1.0,
        "postprocess": "tanh-square",
    }
    model[name + "_scaled"] = True
with open(model_path, "w") as output:
    json.dump(model, output)
atoms = Atoms(
    ["N", "Al", "Al"] * 4,
    positions=rng.uniform(0.0, 6.0, size=(12, 3)),
    cell=[6.0, 6.0, 6.0],
    pbc=True,
)
results = {}
for routing in ("direct", "reference"):
    # The family's two declared routings are the parity axis: ``direct`` runs
    # the generated (T3) implementation through the R1 host plugin this process
    # prepares for the model's contract, and ``reference`` runs the generic
    # implementation on the host execution space with no artifact at all.
    atoms.calc = Symmetrix(model_path, dtype=dtype, streamed_edges=routing)
    evaluator = atoms.calc.evaluator
    results[routing] = {
        "energy": float(atoms.get_potential_energy()),
        "forces": atoms.get_forces().tolist(),
        "stress": atoms.get_stress().tolist(),
        "diagnostics": {
            "routing": atoms.calc.streamed_edges,
            "evaluator_mode": evaluator.streamed_edges_mode,
            "jit_status": atoms.calc.jit_status,
            "jit_origin": atoms.calc.jit_origin,
            "host_plugin_ready": bool(evaluator.jit_host_plugin_ready),
            "forward_launches": int(evaluator.factorized_jit_forward_launch_count),
            "reverse_launches": int(evaluator.factorized_jit_reverse_launch_count),
            "m1_host_plugin_ready": bool(evaluator.m1_host_plugin_ready),
            "m1_forward_launches": int(
                evaluator.generated_m1_module_forward_launch_count
            ),
            "m1_reverse_launches": int(
                evaluator.generated_m1_module_reverse_launch_count
            ),
        },
    }
print("PARITY=" + json.dumps(results))
"""


def _assert_routing_diagnostics(results, routing, mode, artifact_served):
    """Assert one arm ran the routing and implementation it was asked for.

    The numeric comparison below is evidence only while the two arms really are
    two implementations, so each arm reports what it resolved and what it ran:
    the reference arm must show a compiler-free generic evaluation, and the
    direct arm a prepared, loaded, and executed host plugin.
    """

    diagnostics = results[routing]["diagnostics"]
    assert diagnostics["routing"] == routing
    assert diagnostics["evaluator_mode"] == mode
    if artifact_served:
        assert (diagnostics["jit_status"], diagnostics["jit_origin"]) in {
            ("built", "runtime"),
            ("cached", "cache"),
            ("loaded", "prebuilt"),
        }
        assert diagnostics["host_plugin_ready"] is True
        assert diagnostics["forward_launches"] >= 1
        assert diagnostics["reverse_launches"] >= 1
        assert diagnostics["m1_host_plugin_ready"] is True
        assert diagnostics["m1_forward_launches"] >= 1
        assert diagnostics["m1_reverse_launches"] >= 1
    else:
        assert diagnostics["jit_status"] == "not_applicable"
        assert diagnostics["host_plugin_ready"] is False
        assert diagnostics["forward_launches"] == 0
        assert diagnostics["reverse_launches"] == 0
        assert diagnostics["m1_host_plugin_ready"] is False
        assert diagnostics["m1_forward_launches"] == 0
        assert diagnostics["m1_reverse_launches"] == 0


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("threads", [1, 2])
def test_kokkos_host_workers_direct_matches_reference_routing(
    tmp_path, tmp_path_factory, threads, dtype
):
    """The generated routing and the reference routing agree per worker count.

    One worker takes the batched whole-system host paths and two take the
    multi-worker ones, and the Kokkos host execution space serves both routings
    at either count, so the agreement reported here is the parity of two
    implementations per worker count.
    """

    try:
        from symmetrix import symmetrix as native

        space = native._kokkos_default_execution_space()
    except (ImportError, AttributeError, RuntimeError) as error:
        pytest.skip(f"native Kokkos extension unavailable: {error}")
    if space not in ("Serial", "OpenMP"):
        pytest.skip(f"host worker paths do not run on the {space} backend")
    if threads > 1 and space == "Serial":
        pytest.skip("a Serial build always runs one host worker")
    if shutil.which("c++") is None:
        pytest.skip("the direct routing needs a host C++ compiler for its R1 plugin")
    script = tmp_path / "parity.py"
    script.write_text(_DIRECT_REFERENCE_PARITY_SCRIPT)
    shared_cache = tmp_path_factory.getbasetemp() / "standard-routing-jit-cache"
    shared_cache.mkdir(mode=0o700, exist_ok=True)
    environment = dict(
        os.environ,
        KOKKOS_NUM_THREADS=str(threads),
        OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
        # Thread-count cases exercise the same artifact ABI. Reuse one
        # suite-private cache so each precision is compiled once, while the
        # subprocesses still validate fresh-process loading and routing.
        SYMMETRIX_JIT_CACHE=str(shared_cache),
        PYTHONPATH=os.pathsep.join(
            [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
        ),
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(CONTRACTS),
            str(tmp_path / "model.json"),
            dtype,
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    line = next(
        line for line in completed.stdout.splitlines() if line.startswith("PARITY=")
    )
    results = json.loads(line.partition("=")[2])
    _assert_routing_diagnostics(results, "direct", "direct", artifact_served=True)
    _assert_routing_diagnostics(results, "reference", "generic", artifact_served=False)
    direct, reference = results["direct"], results["reference"]
    forces = np.asarray(reference["forces"])
    assert np.abs(forces).max() > 1e-3, "the parity model produced no forces"
    tolerance = 1e-9 if dtype == "float64" else 2e-4
    scale = max(1.0, abs(reference["energy"]))
    assert abs(direct["energy"] - reference["energy"]) <= tolerance * scale
    np.testing.assert_allclose(
        direct["forces"], forces, atol=tolerance * max(1.0, np.abs(forces).max())
    )
    np.testing.assert_allclose(direct["stress"], reference["stress"], atol=tolerance)
