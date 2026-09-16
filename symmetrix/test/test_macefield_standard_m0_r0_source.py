from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "libsymmetrix/source"
RUNTIME_SOURCE = (SOURCE_ROOT / "mace_kokkos_runtime.cpp").read_text()
EVALUATE_SOURCE = (SOURCE_ROOT / "mace_kokkos_evaluate.cpp").read_text()
FIRST_INTERACTION_SOURCE = (
    SOURCE_ROOT / "mace_kokkos_first_interaction.cpp"
).read_text()
RESPONSE_SOURCE = (SOURCE_ROOT / "mace_kokkos_response.tpp").read_text()


def test_macefield_uses_selected_m0_module_boundaries():
    predicate = RUNTIME_SOURCE.split(
        "bool MACEKokkos<Precision>::use_m0_module() const", 1
    )[1].split("template <typename Precision>", 1)[0]
    assert "has_field_coupling" not in predicate
    assert EVALUATE_SOURCE.count("compute_M0_module(num_nodes, node_types);") == 2
    assert EVALUATE_SOURCE.count("reverse_M0_module(num_nodes, node_types);") == 2
    assert "if (use_m0_module())" in RESPONSE_SOURCE
    assert "launch_M0_module_reverse(" in RESPONSE_SOURCE


def test_macefield_admits_only_r0_v2_standard_module_route():
    forward = FIRST_INTERACTION_SOURCE.split(
        "void MACEKokkos<Precision>::compute_A0_streamed(", 1
    )[1].split("void MACEKokkos<Precision>::reverse_A0_streamed(", 1)[0]
    scope = forward.split("const bool module_scope =", 1)[1].split(
        "standard_r0_module_active =", 1
    )[0]
    assert "mace_uses_prepared_execution(streamed_edges)" in scope
    assert "factorized_source_strategy" in scope
    assert "FactorizedSourceStrategy::jit_plugin" in scope
    assert "has_field_coupling" not in scope
