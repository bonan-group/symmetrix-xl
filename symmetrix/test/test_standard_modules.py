import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from symmetrix.execution_contract import (
    normalize_jit_r1_contract,
    normalize_standard_m0_contract,
    normalize_standard_r0_contract,
)

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "libsymmetrix" / "source"
CONTRACTS = Path(__file__).resolve().parent / "data" / "execution_contracts"


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


def test_foundation_registry_tracks_jit_contracts():
    registry = json.loads((CONTRACTS / "foundation_model_registry.json").read_text())

    assert registry["schema"] == "symmetrix.foundation-model-registry"
    assert registry["version"] == 1
    assert len(registry["models"]) == 5
    for model in registry["models"]:
        assert model["runtime_specialization"] == "r1_exact_jit"
        assert model["expected_aot_artifact_ids"] == []
        contract_path = CONTRACTS / model["contract"]
        contract = json.loads(contract_path.read_text())
        assert normalize_jit_r1_contract(contract) == contract


def test_m0_standard_module_owns_the_fixed_lmax3_topology():
    contract = json.loads((CONTRACTS / "standard_m0_contract.json").read_text())
    source = (NATIVE / "standard_m0.hpp").read_text()

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
    source = (NATIVE / "standard_m1.hpp").read_text()
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
            "mace_uses_prepared_execution(streamed_edges) "
            "&& use_factorized_direct_inference()"
        )
        == 2
    )


def test_r0_standard_module_owns_scalar_source_execution():
    contract = json.loads((CONTRACTS / "standard_r0_contract.json").read_text())
    source = (NATIVE / "standard_r0.hpp").read_text()

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
    assert source.count("symmetrix_blas_gemm<Precision>(") == 2
    assert source.count("KokkosBatched::TeamGemm<") >= 2
    assert source.count("symmetrix::host_worker_cblas_enabled()") >= 4
    assert source.count("typename Kokkos::DefaultExecutionSpace::memory_space") >= 3


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
    assert "symmetrix::standard_r0::launch_forward(" in first_interaction
    assert "symmetrix::standard_r0::launch_forward_all_l(" in first_interaction
    assert "use_fused_r0_forward = single_layer_readout" in first_interaction
    assert "launch_M0_module_reverse(" in response
    assert "use_m0_module()" in response
    for source in (model, first_interaction, response):
        assert "execution_artifacts" not in source
        assert "execution_artifact_registry" not in source


def test_generic_standard_r0_reverse_uses_and_retains_generic_receiver_map():
    first_interaction = (NATIVE / "mace_kokkos_first_interaction.cpp").read_text()
    lifecycle = (NATIVE / "mace_kokkos_factorized_lifecycle.cpp").read_text()
    normalized_first_interaction = " ".join(first_interaction.split())

    assert (
        "edge_receivers = streamed_edges == MACEStreamedEdgesMode::generic "
        "? streamed_edge_receivers : execution_edge_receivers;"
        in normalized_first_interaction
    )
    assert first_interaction.count("edge_receivers,") >= 2
    assert (
        """streamed_edges != MACEStreamedEdgesMode::generic
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
