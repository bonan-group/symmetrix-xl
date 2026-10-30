"""Routing of NequIP-OAM contracts to the generated CUDA R1 artifact."""

from __future__ import annotations

from pathlib import Path

import pytest
import symmetrix.nequip_calculator as calculator_module

FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"

# The frontend prepares a direct artifact through the entry point the family
# declares for the target it binds, so a stub replaces that declared entry point
# rather than a copy of it on the frontend module.
OAM_CUDA_ADAPTER = "symmetrix.oam_cuda_artifact.prepare_nequip_oam_cuda_artifact"

CUDA_ENVIRONMENT = {
    "backend": "cuda",
    "execution_space": "Cuda",
    "architecture": "sm_120",
    "compute_capability": 120,
    "compute_capability_code": 120,
    "available": True,
}

HIP_ENVIRONMENT = {
    "backend": "hip",
    "execution_space": "HIP",
    "architecture": "gfx1151",
    "available": True,
}


def test_hip_oam_target_is_rejected_instead_of_treated_as_host():
    """A HIP evaluator cannot consume the host receiver callback."""

    class Native:
        @staticmethod
        def _execution_device_execution_environment():
            return dict(HIP_ENVIRONMENT)

    with pytest.raises(RuntimeError, match="NequIP-OAM.*HIP.*not supported"):
        calculator_module._execution_artifact_target(Native)


def test_native_oam_admission_rejects_non_cuda_device_direct_execution():
    source = (
        Path(__file__).resolve().parents[2] / "libsymmetrix/source/oam_kokkos.cpp"
    ).read_text()

    assert 'return "hip";' in source
    assert (
        'mode == "direct" && !execution_space_is_host()\n'
        "            && !execution_space_is_cuda()"
    ) in source
    assert "direct execution is unsupported on target" in source


def test_cuda_oam_artifact_failure_propagates(monkeypatch, tmp_path):
    from symmetrix.oam_cuda_artifact import OAMCudaArtifactError

    class Native:
        NequIPOAMKokkos = object

        @staticmethod
        def _kokkos_is_initialized():
            return True

        @staticmethod
        def _execution_device_execution_environment():
            return dict(CUDA_ENVIRONMENT)

    requests = []

    def unavailable(path, *, precision, target, native_module):
        requests.append((path, precision, target, native_module))
        raise OAMCudaArtifactError("the stub compiler is unavailable")

    monkeypatch.setattr(OAM_CUDA_ADAPTER, unavailable)
    monkeypatch.setattr(calculator_module, "load_backend", lambda: Native)

    with pytest.raises(OAMCudaArtifactError, match="stub compiler"):
        calculator_module.NequIPCalculator(
            FIXTURE, dtype="float64", streamed_edges="direct"
        )

    # The failure proves the CUDA branch only if the frontend resolved the CUDA
    # target and handed the declared adapter the device environment it resolved:
    # the stub is the declared entry point of the CUDA adapter, and the host
    # adapter is never consulted for this request.
    assert requests == [(FIXTURE, "float64", CUDA_ENVIRONMENT, Native)]


def test_debug_v1_is_rejected_at_admission_on_a_device_build(tmp_path, monkeypatch):
    """A host-only debug owner must not be admitted by a device evaluator.

    Loading the legacy host artifact is not enough to make its callbacks safe
    for a device execution space.  Rejecting the mode here keeps admission and
    evaluation consistent instead of accepting it and failing on first use.
    """

    from symmetrix import symmetrix as native
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_host_artifact

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    if native._kokkos_default_execution_space() in ("OpenMP", "Serial"):
        pytest.skip("debug_v1 device admission requires a device Kokkos build")

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    artifact = prepare_nequip_oam_host_artifact(FIXTURE, precision="float64")
    evaluator = native.NequIPOAMKokkos(str(FIXTURE), str(artifact.artifact_path))

    error = None
    try:
        evaluator.set_streamed_edges("debug_v1")
    except ValueError as caught:
        error = caught
    finally:
        # A failing assertion retains its frame until pytest reports it.  Drop
        # the native owner first so its Kokkos guard is released before module
        # teardown even in the expected RED run.
        del evaluator

    assert error is not None
    assert "debug_v1" in str(error)
    assert "host execution space" in str(error)
