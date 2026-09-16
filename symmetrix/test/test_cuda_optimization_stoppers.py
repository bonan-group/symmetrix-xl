from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "libsymmetrix" / "source"


def test_cuda_sphericart_default_preserves_explicit_override():
    source = (ROOT / "symmetrix/CMakeLists.txt").read_text()

    assert (
        """if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_CUDA
    AND NOT DEFINED SYMMETRIX_SPHERICART_CUDA)
  set(SYMMETRIX_SPHERICART_CUDA ON CACHE BOOL \"Enable SpheriCart-CUDA.\")
endif()"""
        in source
    )


def test_gpu_global_field_reverse_remains_bounded_and_wave32_reduced():
    header = (NATIVE / "mace_kokkos.hpp").read_text()
    source = (NATIVE / "mace_kokkos_h1_phi1.cpp").read_text()
    runtime = (NATIVE / "mace_kokkos_runtime.cpp").read_text()

    assert "field_H1_global_partial" in header
    assert "bool use_gpu_field_h1_reverse() const;" in header
    assert "double deterministic_gpu_wave32_sum(double value)" in source
    assert "threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z)" in source
    assert "defined(__HIP_DEVICE_COMPILE__)" in source
    assert '"MACEKokkos::reverse_field_H1_global_features"' in source
    assert '"MACEKokkos::reverse_field_H1_global_reduce"' in source
    assert "symmetrix::execution::field_h1_reverse_id" in source
    assert "execution_space, persistent_blocks, threads_per_block" in source
    assert "partial_count = worker_count/wave_size" in source
    assert "(input_count+wave_size-1)/wave_size" not in source
    assert "if (use_gpu_field_h1_reverse())" in source
    assert "const auto execution_space = use_factorized_async_inference()" in source
    assert 'complete_device_stage("MACEKokkos::reverse_field_H1")' in source
    predicate = runtime[
        runtime.index(
            "bool MACEKokkos<Precision>::use_gpu_field_h1_reverse() const"
        ) : runtime.index(
            "void MACEKokkos<Precision>::complete_device_stage",
            runtime.index(
                "bool MACEKokkos<Precision>::use_gpu_field_h1_reverse() const"
            ),
        )
    ]
    assert "mace_uses_prepared_execution(streamed_edges)" in predicate
    assert "!factorized_observer_enabled" in predicate
    assert "!execution_parameter_gradients_enabled" in predicate


def test_cuda_response_recompute_and_widened_offsets_remain_available():
    header = (NATIVE / "mace_kokkos.hpp").read_text()
    response = (NATIVE / "mace_kokkos_response.tpp").read_text()

    for expected in (
        "macefield_response_m1_recompute_tile_channels() const;",
        "macefield_response_m1_recompute_scratch_bytes() const;",
        "macefield_response_m1_recompute_scratch_limit_bytes() const;",
    ):
        assert expected in header
    assert response.count("Kokkos::IndexType<std::size_t>") >= 5
    assert response.count("const std::size_t coordinate_offset = 3*edge;") >= 6
    assert "macefield_response_m1_recompute_forward_launch_count += 1;" in response
    assert "macefield_response_m1_recompute_reverse_launch_count += 1;" in response


def test_cuda_m1_recompute_admission_distinguishes_adjoint_overlap():
    header = (NATIVE / "mace_kokkos.hpp").read_text()
    runtime = (NATIVE / "mace_kokkos_runtime.cpp").read_text()

    assert "bool require_adjoint_overlap = false" in header
    assert "m1_recompute_effective_scratch_bytes(" in runtime
    assert "const bool require_adjoint_overlap =" in runtime
    assert "channels, require_adjoint_overlap" in runtime
    assert "m1_recompute_tile_channels, true" in runtime
    assert "mh0_state_policy == MH0StatePolicy::reuse_adjoints" in runtime


def test_nvrtc_cache_publication_remains_optimistic_and_no_replace():
    source = (ROOT / "symmetrix/source/symmetrix/jit.py").read_text()

    for expected in (
        "def _create_staging_directory(",
        "def _publish_directory_no_replace(",
        'renameat2 = getattr(library, "renameat2", None)',
        "published = _publish_directory_no_replace(temporary_directory, entry)",
        'diagnostics.append("another process published the JIT artifact first")',
    ):
        assert expected in source
    compile_start = source.index(
        "temporary_directory = _create_staging_directory(staging_root, cache_key)"
    )
    publication = source.index(
        "published = _publish_directory_no_replace(temporary_directory, entry)"
    )
    assert "with _cache_key_lock(" not in source[compile_start:publication]
