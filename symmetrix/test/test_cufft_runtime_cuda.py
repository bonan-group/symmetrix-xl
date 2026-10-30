"""Fresh-process checks of the real lazy CUDA FFT provider boundary."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import symmetrix


@pytest.fixture
def cuda_selector():
    native = symmetrix.load_backend()
    if not hasattr(native, "_cufft_runtime_information"):
        pytest.skip("requires a CUDA backend built with runtime cuFFT dispatch")
    if not Path("/proc/self/maps").is_file():
        pytest.skip("requires Linux process library inspection")
    return symmetrix.selected_backend()["selector"]


def _run(selector, body, **environment):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        env={**os.environ, "SYMMETRIX_BACKEND": selector, **environment},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Failed to call Kokkos" not in result.stderr, result.stderr


@pytest.mark.cuda
def test_missing_cufft_does_not_block_import_or_non_fft_execution(
    cuda_selector, tmp_path
):
    _run(
        cuda_selector,
        """
        import gc
        from pathlib import Path
        import symmetrix

        native = symmetrix.load_backend()
        assert native._backend_build_info()["cufft_linkage"] == "runtime"
        assert native._particle_mesh_capability()["available"]
        assert native._cufft_runtime_information()["state"] == "unloaded"
        assert "libcufft" not in Path("/proc/self/maps").read_text()
        native._init_kokkos()
        assert native._kokkos_device_sentinel()
        storage = native._FourierD3MeshStorage("float64", "layout_right", 2)
        storage.reserve(8, 8, 8, 1)
        domain = native._FourierD3FftPlanDomain("float64")
        try:
            domain.acquire(storage)
        except RuntimeError as error:
            assert "missing-cufft.so" in str(error)
            assert "pip install" in str(error)
        else:
            raise AssertionError("FFT plan unexpectedly accepted a missing runtime")
        assert native._cufft_runtime_information()["state"] == "probe-failed"
        assert "libcufft" not in Path("/proc/self/maps").read_text()
        assert native._kokkos_device_sentinel()
        del domain, storage
        gc.collect()
        assert native._kokkos_live_object_count() == 0
        native._finalize_kokkos()
        """,
        SYMMETRIX_CUFFT_LIBRARY=str(tmp_path / "missing-cufft.so"),
    )


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_external_cufft_loads_on_first_fft_and_matches_numpy(cuda_selector, precision):
    _run(
        cuda_selector,
        """
        import gc
        import os
        from pathlib import Path
        import numpy as np
        import symmetrix

        native = symmetrix.load_backend()
        assert native._cufft_runtime_information()["state"] == "unloaded"
        assert "libcufft" not in Path("/proc/self/maps").read_text()
        native._init_kokkos()
        precision = os.environ["TEST_CUFFT_PRECISION"]
        tolerance = 2e-5 if precision == "float32" else 1e-12
        forward = native._FourierD3ReciprocalForward(precision, 2)
        forward.reserve(8, 10, 12, 1, 2, 2, 2, 4)
        forward.set_cell(np.diag([8., 10., 12.]).reshape(-1))
        forward.set_species(np.array([6], dtype=np.int32), np.array([1.6]))
        forward.set_atoms(np.array([[0., 0., 0.], [1.5, 2.25, 3.]]),
                          np.array([6, 6], dtype=np.int32))
        forward.set_damping(0.4, 2., 1., 0.8)
        forward.set_rank(0, np.array([1., 0.5]))
        forward.set_coefficients(np.zeros((2, 2)))
        field = np.random.default_rng(42).normal(size=(2, 8 * 10 * 12))
        forward.set_field(field)
        forward.transform_forward()
        report = native._cufft_runtime_information()
        assert report["state"] == "loaded" and report["available"], report
        assert Path(report["library"]).is_absolute(), report
        assert report["version"] // 1000 in (11, 12), report
        expected = np.fft.rfftn(field.reshape(2, 8, 10, 12), axes=(1, 2, 3))
        np.testing.assert_allclose(forward.spectrum(), expected,
                                   rtol=tolerance, atol=tolerance)
        forward.transform_inverse()
        np.testing.assert_allclose(forward.field(), 960 * field,
                                   rtol=tolerance, atol=960 * tolerance)
        del forward
        gc.collect()
        assert native._cufft_runtime_information()["state"] == "loaded"
        assert native._kokkos_live_object_count() == 0
        native._finalize_kokkos()
        """,
        TEST_CUFFT_PRECISION=precision,
    )
