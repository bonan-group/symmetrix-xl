import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "libsymmetrix" / "source"
BINDINGS_ROOT = ROOT / "symmetrix" / "source" / "cpp"
OWNERS = (
    "mace_kokkos_first_interaction.cpp",
    "mace_kokkos_response.cpp",
    "mace_kokkos_factorized_execution.cpp",
    "mace_kokkos_h1_phi1.cpp",
    "mace_kokkos_model.cpp",
    "mace_kokkos_factorized_lifecycle.cpp",
    "mace_kokkos_second_interaction.cpp",
    "mace_kokkos_factorized_analysis.cpp",
    "mace_kokkos_runtime.cpp",
    "mace_kokkos_evaluate.cpp",
)

MACE_BASE_OWNERS = (
    "mace_kokkos.hpp",
    "mace_kokkos_model.cpp",
    "mace_kokkos_evaluate.cpp",
    "mace_kokkos_factorized_lifecycle.cpp",
    "mace_kokkos_runtime.cpp",
)

MACELES_OWNERS = (
    "mace_les_kokkos_model.cpp",
    "mace_les_kokkos_evaluate.cpp",
    "mace_les_kokkos_lifecycle.cpp",
)


def _owner_sources():
    return {name: (SOURCE_ROOT / name).read_text() for name in OWNERS}


def test_cmake_lists_the_semantic_owners_once():
    cmake = (ROOT / "libsymmetrix" / "CMakeLists.txt").read_text()
    listed = tuple(re.findall(r"source/(mace_kokkos_[a-z0-9_]+\.cpp)", cmake))

    assert listed == OWNERS


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
        "mace_kokkos_evaluate.cpp": (
            "MACEKokkos<Precision>::compute_node_energies_forces_field("
        ),
        "mace_kokkos_first_interaction.cpp": (
            "MACEKokkos<Precision>::compute_A0_streamed("
        ),
        "mace_kokkos_h1_phi1.cpp": "MACEKokkos<Precision>::reverse_field_H1(",
        "mace_kokkos_second_interaction.cpp": "MACEKokkos<Precision>::compute_readouts(",
        "mace_kokkos_model.cpp": "MACEKokkos<Precision>::load_mace_backbone(",
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


@pytest.mark.parametrize("name", MACE_BASE_OWNERS)
def test_mace_base_owns_no_les_state_or_algorithm(name):
    source = (SOURCE_ROOT / name).read_text()
    for token in (
        "les_parameters",
        "les_monopole",
        "latent_charges",
        "les_charge_",
        "les_geometry_",
        "les_particle_mesh_",
        "evaluate_les_term",
        "add_les_explicit_",
    ):
        assert token not in source, (name, token)


def test_maceles_owns_les_algorithms_after_legacy_owner_is_retired():
    assert not (SOURCE_ROOT / "mace_kokkos_les.cpp").exists()

    sources = {name: (SOURCE_ROOT / name).read_text() for name in MACELES_OWNERS}
    expected_owners = {
        "mace_les_kokkos_model.cpp": (
            "MACELESKokkos<Precision>::initialize_les_readouts()"
        ),
        "mace_les_kokkos_evaluate.cpp": (
            "MACELESKokkos<Precision>::evaluate_les_term("
        ),
        "mace_les_kokkos_lifecycle.cpp": (
            "MACELESKokkos<Precision>::set_les_geometry("
        ),
    }
    for expected_owner, method in expected_owners.items():
        actual_owners = tuple(
            name for name, source in sources.items() if method in source
        )
        assert actual_owners == (expected_owner,), method

    for name, source in sources.items():
        for precision in ("float", "double"):
            assert "template " in source
            assert f"MACELESKokkos<{precision}>::" in source, (name, precision)


def test_mace_binding_array_helpers_have_one_internal_definition():
    helper_path = BINDINGS_ROOT / "mace_kokkos_binding_helpers.hpp"
    assert helper_path.exists()

    helper = helper_path.read_text()
    owners = {
        name: (BINDINGS_ROOT / name).read_text()
        for name in ("mace_kokkos.cpp", "mace_les_kokkos.cpp")
    }
    definitions = (
        "static py::array_t<T> shaped_array(",
        "static auto view_prefix_1d(",
        "static auto view_prefix_3d(",
    )
    for definition in definitions:
        assert helper.count(definition) == 1, definition
        unqualified_definition = definition.removeprefix("static ")
        for name, source in owners.items():
            assert unqualified_definition not in source, (name, definition)

    for name, source in owners.items():
        assert source.count('#include "mace_kokkos_binding_helpers.hpp"') == 1, name


def test_model_document_parsing_has_one_owner_and_shared_selected_type():
    document_header = (SOURCE_ROOT / "mace_model_document.hpp").read_text()
    document_owner = (SOURCE_ROOT / "mace_model_document.cpp").read_text()
    base_model_owner = (SOURCE_ROOT / "mace_kokkos_model.cpp").read_text()
    base_runtime_owner = (SOURCE_ROOT / "mace_kokkos_runtime.cpp").read_text()
    derived_header = (SOURCE_ROOT / "mace_les_kokkos.hpp").read_text()
    derived_instantiation_owner = (SOURCE_ROOT / "mace_les_kokkos.cpp").read_text()
    derived_model_owner = (SOURCE_ROOT / "mace_les_kokkos_model.cpp").read_text()

    assert "struct SelectedMACEModel" in document_header
    assert "enum class MACEModelKind" in document_header
    assert "read_selected_mace_model(" in document_owner
    assert "nlohmann::json::parse" in document_owner
    for name, source in {
        "mace_kokkos_model.cpp": base_model_owner,
        "mace_kokkos_runtime.cpp": base_runtime_owner,
        "mace_les_kokkos_model.cpp": derived_model_owner,
    }.items():
        assert "nlohmann::json::parse" not in source, name

    assert "const SelectedMACEModel& selected" in base_model_owner
    assert "const SelectedMACEModel& selected" in derived_header
    assert "const SelectedMACEModel& selected" in derived_model_owner
    assert "read_selected_mace_model(" in base_runtime_owner
    assert "read_selected_mace_model(" in derived_model_owner
    for precision in ("float", "double"):
        instantiation = f"template class MACELESKokkos<{precision}>;"
        assert derived_instantiation_owner.count(instantiation) == 1
        assert instantiation not in derived_model_owner

    assert "json_parse_ms" in document_header
    assert "selected.json_parse_ms" in base_model_owner


@pytest.mark.parametrize(
    "token",
    (
        "is_mace_les",
        "prepare_les_charge_readouts",
        "evaluate_les_term",
        "add_les_explicit_forces",
        "add_les_explicit_stress",
        "les_charge_readouts_ready",
        "les_geometry_ready",
    ),
)
def test_base_evaluation_owner_uses_only_neutral_model_hooks(token):
    source = (SOURCE_ROOT / "mace_kokkos_evaluate.cpp").read_text()
    assert token not in source


def test_base_evaluation_preserves_pre_and_post_readout_model_stages():
    source = (SOURCE_ROOT / "mace_kokkos_evaluate.cpp").read_text()
    stage_sequence = re.compile(
        r"prepare_model_readouts\(num_nodes\);\s*"
        r"compute_readouts\(num_nodes, node_types, false, false\);\s*"
        r"evaluate_model_after_readouts\(num_nodes\);"
    )

    assert len(stage_sequence.findall(source)) == 2


@pytest.mark.parametrize(
    ("owner", "token"),
    (
        ("mace_kokkos_evaluate.cpp", "validate_distributed_prepared_model"),
        ("mace_kokkos_factorized_lifecycle.cpp", "is_mace_les"),
        ("mace_kokkos_factorized_lifecycle.cpp", "les_periodic_algorithm"),
        ("mace_kokkos_factorized_lifecycle.cpp", "fence_les_particle_mesh_owner"),
        ("mace_kokkos_factorized_lifecycle.cpp", "les_particle_mesh_owner_valid"),
        ("mace_kokkos_factorized_lifecycle.cpp", "les_geometry_ready"),
        ("mace_kokkos_factorized_lifecycle.cpp", "les_results_ready"),
        ("mace_kokkos_runtime.cpp", "reject_les_artifact_load"),
        ("mace_kokkos_runtime.cpp", "is_mace_les()"),
    ),
)
def test_base_lifecycle_owners_use_only_neutral_model_policy_hooks(owner, token):
    source = (SOURCE_ROOT / owner).read_text()
    assert token not in source


# --- MACE_Nonlinear (MH-1) family partition --------------------------------
# The MH-1 evaluator is partitioned the same way: three translation units
# instantiate the family's template from one implementation body, so an edit to
# a single owner rebuilds that owner and relinks rather than recompiling the
# family.  Its stage description is family-owned and family-prefixed, and it
# carries the interaction-layer count and every layer's region name as data.

MH1_OWNERS = (
    "mace_nonlinear_kokkos_components.cpp",
    "mace_nonlinear_kokkos_runtime.cpp",
    "mace_nonlinear_kokkos_evaluate.cpp",
)

MH1_IMPL = "mace_nonlinear_kokkos_impl.tpp"
MH1_STAGE_HEADER = "mace_nonlinear_stage.hpp"
MH1_RETIRED_STAGE_HEADER = "mh1_stage.hpp"


def test_cmake_lists_the_mh1_family_owners_once():
    cmake = (ROOT / "libsymmetrix" / "CMakeLists.txt").read_text()
    listed = tuple(re.findall(r"source/(mace_nonlinear_kokkos_[a-z0-9_]+\.cpp)", cmake))

    assert listed == MH1_OWNERS


def test_each_mh1_owner_is_one_partition_of_the_single_body():
    for part, name in enumerate(MH1_OWNERS, start=1):
        source = (SOURCE_ROOT / name).read_text()

        assert f"#define SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART {part}" in source, name
        assert f'#include "{MH1_IMPL}"' in source, name


def test_the_mh1_body_is_included_only_by_its_owners():
    includers = tuple(
        sorted(
            path.name
            for path in SOURCE_ROOT.iterdir()
            if path.suffix == ".cpp" and f'#include "{MH1_IMPL}"' in path.read_text()
        )
    )

    # The body has exactly the family's owners as its includers; the CMake list
    # order is a scheduling contract, so the comparison is by set.
    assert includers == tuple(sorted(MH1_OWNERS))


def test_the_mh1_stage_description_is_family_owned_data():
    stage = (SOURCE_ROOT / MH1_STAGE_HEADER).read_text()

    # The description carries the layer count and the region names as data, so
    # an evaluator that admits a different interaction count profiles it through
    # the same value instead of a compiled-in two-layer table.
    assert "StageDescription" in stage
    assert "layer_count" in stage
    assert "canonical" in stage
    assert "symmetrix/mh1/" in stage
    assert not (SOURCE_ROOT / MH1_RETIRED_STAGE_HEADER).exists()
    for name in (*MH1_OWNERS, MH1_IMPL, MH1_STAGE_HEADER):
        assert MH1_RETIRED_STAGE_HEADER not in (SOURCE_ROOT / name).read_text(), name
