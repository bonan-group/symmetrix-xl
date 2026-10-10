import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "libsymmetrix" / "source"
OWNERS = (
    "mace_kokkos_first_interaction.cpp",
    "mace_kokkos_response.cpp",
    "mace_kokkos_factorized_execution.cpp",
    "mace_kokkos_h1_phi1.cpp",
    "mace_kokkos_model.cpp",
    "mace_kokkos_factorized_lifecycle.cpp",
    "mace_kokkos_second_interaction.cpp",
    "mace_kokkos_factorized_analysis.cpp",
    "mace_kokkos_force_training.cpp",
    "mace_kokkos_training.cpp",
    "mace_kokkos_runtime.cpp",
    "mace_kokkos_evaluate.cpp",
)


def _owner_sources():
    return {name: (SOURCE_ROOT / name).read_text() for name in OWNERS}


def test_cmake_lists_the_semantic_owners_once():
    cmake = (ROOT / "libsymmetrix" / "CMakeLists.txt").read_text()
    listed = tuple(re.findall(r"source/(mace_kokkos_[a-z0-9_]+\.cpp)", cmake))

    assert listed == OWNERS


def test_direct_training_surface_is_part_of_the_public_evaluator():
    cmake = (ROOT / "libsymmetrix" / "CMakeLists.txt").read_text()
    header = (SOURCE_ROOT / "mace_kokkos.hpp").read_text()

    assert "source/mace_kokkos_training.cpp" in cmake
    assert "source/mace_kokkos_force_training.cpp" in cmake
    for declaration in (
        "struct DirectTrainingParameterGroup",
        "struct DirectOptimizerStepResult",
        "struct DirectBatchEnergyLossResult",
        "struct DirectBatchForceLossResult",
        "enum class DirectGradientObjective",
    ):
        assert declaration in header
    assert "source/neighbor_graph_kokkos.cpp" in cmake
    assert "source/jit_operator_module.cpp" in cmake


def test_selector_monolith_is_retired():
    sources = _owner_sources()
    cmake = (ROOT / "libsymmetrix" / "CMakeLists.txt").read_text()

    assert not (SOURCE_ROOT / "mace_kokkos_impl.tpp").exists()
    for source in (*sources.values(), cmake):
        assert "SYMMETRIX_MACE_KOKKOS_PART" not in source
        assert "mace_kokkos_impl.tpp" not in source


def test_each_owner_instantiates_both_precisions_once():
    for name, source in _owner_sources().items():
        instantiations = tuple(
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("template class MACEKokkos")
        )
        if name == "mace_kokkos_force_training.cpp":
            assert "MACEKokkos<float>::compute_prepared_direct_force_loss" in source
            assert "MACEKokkos<double>::compute_prepared_direct_force_loss" in source
        else:
            assert instantiations == (
                "template class MACEKokkos<float>;",
                "template class MACEKokkos<double>;",
            ), name


def test_representative_methods_have_one_semantic_owner():
    sources = _owner_sources()
    expected_owners = {
        "mace_kokkos_runtime.cpp": "MACEKokkos<Precision>::use_standard_m0_module()",
        "mace_kokkos_factorized_lifecycle.cpp": (
            "MACEKokkos<Precision>::prepare_factorized_schedule("
        ),
        "mace_kokkos_factorized_execution.cpp": "MACEKokkos<Precision>::compute_factorized(",
        "mace_kokkos_factorized_analysis.cpp": (
            "MACEKokkos<Precision>::compute_execution_density_parameter_gradients("
        ),
        "mace_kokkos_training.cpp": (
            "MACEKokkos<Precision>::initialize_direct_training_parameters("
        ),
        "mace_kokkos_evaluate.cpp": (
            "MACEKokkos<Precision>::compute_node_energies_forces_field("
        ),
        "mace_kokkos_first_interaction.cpp": (
            "MACEKokkos<Precision>::compute_A0_streamed("
        ),
        "mace_kokkos_h1_phi1.cpp": "MACEKokkos<Precision>::reverse_field_H1(",
        "mace_kokkos_second_interaction.cpp": "MACEKokkos<Precision>::compute_readouts(",
        "mace_kokkos_model.cpp": "MACEKokkos<Precision>::load_from_json(",
    }

    for expected_owner, method in expected_owners.items():
        actual_owners = tuple(
            name for name, source in sources.items() if method in source
        )
        assert actual_owners == (expected_owner,), method

    response_owner = sources["mace_kokkos_response.cpp"]
    response_implementation = (SOURCE_ROOT / "mace_kokkos_response.tpp").read_text()
    assert '#include "mace_kokkos_response.tpp"' in response_owner
    assert (
        "MACEKokkos<Precision>::compute_electric_field_response("
        in response_implementation
    )
