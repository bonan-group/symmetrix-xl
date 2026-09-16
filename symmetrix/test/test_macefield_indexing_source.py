import re
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"


def _function_source(source, start, end):
    return source[source.index(start) : source.index(end)]


def test_macefield_response_uses_widened_edge_offsets():
    source = (SOURCE_ROOT / "mace_kokkos_response.tpp").read_text()

    assert 3 * 44_739_243 * 16 > 2**31 - 1
    assert source.count("const std::size_t coordinate_offset = 3*edge;") == 8
    assert source.count("const std::size_t harmonic_offset =") == 5
    assert source.count("const std::size_t gradient_offset =") == 4
    assert source.count("Kokkos::IndexType<std::size_t>") == 5
    assert (
        "response_mode == MACEStreamedEdgesMode::generic\n"
        "            || mace_uses_prepared_execution(response_mode)"
    ) in source

    forbidden = (
        r"\bedge\s*\*\s*lm_count",
        r"3\s*\*\s*edge\s*\*\s*lm_count",
        r"\(\s*3\s*\*\s*edge\s*\+",
        r"xyz\(\s*3\s*\*\s*edge",
        r"response_forces\([^\n]*3\s*\*\s*edge",
    )
    for pattern in forbidden:
        assert re.search(pattern, source) is None, pattern


def test_macefield_field_policies_use_size_t_indices():
    source = (SOURCE_ROOT / "mace_kokkos_h1_phi1.cpp").read_text()
    field_source = _function_source(
        source,
        "void MACEKokkos<Precision>::compute_field_H1(",
        "void MACEKokkos<Precision>::compute_Phi1(",
    )

    assert field_source.count("Kokkos::IndexType<std::size_t>") == 7
    assert "3*static_cast<std::size_t>(num_nodes)" in field_source
    assert "const std::size_t input_count =" in field_source
    assert "static_cast<std::size_t>(num_nodes)*h1_lm*h1_channels" in field_source
    assert '"MACEKokkos::reverse_field_H1_global_features"' in field_source
    assert '"MACEKokkos::reverse_field_H1_global_reduce"' in field_source
    assert "num_nodes*num_channels" not in field_source
    assert field_source.count("const std::size_t field_offset") == 2
    assert "void operator()(const std::size_t index" in source
    assert "{num_nodes,h1_lm,h1_channels}" in field_source


def test_radial_table_rows_widen_layout_right_offsets():
    active = (SOURCE_ROOT / "radial_function_set_kokkos.cpp").read_text()

    assert 4_780_244 * 10 * 48 > 2**31 - 1
    assert (
        "const std::size_t ij = static_cast<std::size_t>(\n"
        "                team_member.league_rank());"
    ) in active
    assert "R(ij,k)" in active
    assert "R_deriv(ij,k)" in active


def test_m1_recompute_scratch_is_portably_capability_admitted():
    source = (SOURCE_ROOT / "mace_kokkos_runtime.cpp").read_text()

    assert "execution_space.cuda_device_prop()" in source
    assert "execution_space.hip_device_prop()" in source
    assert "Kokkos::TeamPolicy<>::scratch_size_max(scratch_level)" in source
    assert "environment.max_shared_memory_per_block" in source
    assert "for (const int candidate : {32, 16, 8})" in source
    assert "m1_recompute_required_scratch_bytes" in source
    assert "std::numeric_limits<std::size_t>::max()" in source
    assert 'environment.backend == "host" ? 1 : 0' in source
    assert '" requires at least "' in source
    assert '"; available limit is "' in source
