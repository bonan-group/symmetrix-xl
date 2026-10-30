"""Cache, identity, and policy behavior of the OAM CUDA artifact adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from symmetrix import jit as jit_module
from symmetrix import oam_cuda_artifact as artifact_module
from symmetrix.oam_architecture import oam_tensor_product_architecture_fingerprint
from symmetrix.oam_cuda_artifact import (
    OAM_CUDA_EXPECTATION_FIELDS,
    OAMCudaArtifactError,
    prepare_nequip_oam_cuda_artifact,
)
from symmetrix.oam_cuda_codegen import OAMCudaGenerationPolicy

FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"
PLAIN_NEQUIP_FIXTURE = Path(__file__).parent / "data" / "nequip" / "tiny_nequip_v1.json"

# ``prepare_nvrtc_jit_artifact`` runs in-process; these stand in for the NVRTC
# library and the device compiler so cache behavior is testable without a GPU.
NVRTC_INFORMATION = {
    "available": True,
    "major": 12,
    "minor": 4,
    "supported_architectures": [90, 120],
    "library": "libnvrtc.so.stub",
}


def _stub_renderer(monkeypatch, rendered=None):
    """Replace the generated-code renderer with a deterministic stub."""

    def render(
        contract,
        target,
        precision="float32",
        threads_per_block=128,
        generation_policy=None,
    ):
        if rendered is not None:
            rendered.append((target, precision, threads_per_block))
        return "// oam cuda stub\n"

    monkeypatch.setattr(artifact_module, "render_oam_cuda_plugin", render)


def _real_nvrtc_cache(monkeypatch):
    """Route NVRTC preparation through the real cache with a stubbed compiler."""

    def prepare(source, **arguments):
        arguments.setdefault("nvrtc_information", dict(NVRTC_INFORMATION))
        arguments.setdefault(
            "compile_source", lambda source, options: (b"\x7fELF-oam-cubin", "")
        )
        return jit_module.prepare_nvrtc_jit_artifact(source, **arguments)

    monkeypatch.setattr(artifact_module, "prepare_nvrtc_jit_artifact", prepare)


def _variant_contract(tmp_path, mutate):
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutate(contract)
    variant = tmp_path / "variant_oam.json"
    variant.write_text(json.dumps(contract), encoding="utf-8")
    return variant


def _bump_first_tensor_product_weight(contract):
    weights = contract["layers"][0]["edge_tensor_product"]["weight"]["values"]
    weights[0] = float(weights[0]) + 0.125


@pytest.fixture
def required_policy(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")


def test_prepare_builds_then_hits_the_content_addressed_cache(
    monkeypatch, tmp_path, required_policy
):
    _stub_renderer(monkeypatch)
    _real_nvrtc_cache(monkeypatch)

    first = prepare_nequip_oam_cuda_artifact(
        FIXTURE, "float32", "sm_90", "nvrtc", cache_root=tmp_path
    )

    assert first.status == "built"
    assert first.available and not first.cache_hit
    assert first.artifact_id.startswith("oam-cuda-r1-")
    assert first.compiler == "nvrtc"
    assert first.target == "sm_90"
    assert first.fallback_reason is None
    assert first.artifact_path.is_file()
    assert first.artifact_path.read_bytes() == b"\x7fELF-oam-cubin"

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["key_inputs"]["build"]["request"]["metadata"]
    bound = manifest["key_inputs"]["build"]["request"]["oam"]
    assert "projections" not in metadata
    assert all("projection" not in layer for layer in metadata["layers"])
    assert bound["artifact_id"] == first.artifact_id
    assert bound["contract_fingerprint"] == oam_tensor_product_architecture_fingerprint(
        json.loads(FIXTURE.read_text(encoding="utf-8"))
    )
    assert bound["semantic_fingerprint"] != bound["contract_fingerprint"]
    assert bound["target_architecture"] == "sm90"
    assert bound["precision"] == "float32"
    assert bound["compiler"] == "nvrtc"
    assert bound["layer_schedule"][0]["layer"] == 0
    assert manifest["key_inputs"]["abi"] == {
        "tag": "symmetrix.jit.oam.cuda-plugin/3",
        "version": 3,
    }
    for field in OAM_CUDA_EXPECTATION_FIELDS:
        assert field in bound

    second = prepare_nequip_oam_cuda_artifact(
        FIXTURE, "float32", "sm_90", "nvrtc", cache_root=tmp_path
    )

    assert second.status == "cached"
    assert second.cache_hit
    assert second.cache_key == first.cache_key
    assert second.artifact_id == first.artifact_id
    assert second.artifact_path == first.artifact_path


def test_installed_cuda_architecture_avoids_runtime_compiler_probe(
    monkeypatch, tmp_path
):
    cubin = tmp_path / "plugin.cubin"
    manifest = tmp_path / "manifest.json"
    cubin.write_bytes(b"installed-cubin")
    manifest.write_text("{}", encoding="utf-8")
    resolved = []

    def resolve(**kwargs):
        resolved.append(kwargs)
        return SimpleNamespace(
            available=True,
            artifact_path=cubin,
            manifest_path=manifest,
            diagnostics=("origin=prebuilt status=loaded family=OAM",),
        )

    monkeypatch.setattr(artifact_module, "resolve_installed_oam_artifact", resolve)
    monkeypatch.setattr(
        artifact_module,
        "_selected_compiler",
        lambda compiler: pytest.fail("runtime compiler must not be probed"),
    )

    result = prepare_nequip_oam_cuda_artifact(
        FIXTURE,
        "float32",
        "sm_120",
        native_module=object(),
    )

    assert result.status == "installed"
    assert result.artifact_path == cubin
    assert result.manifest_path == manifest
    assert result.compiler == "prebuilt"
    assert resolved[0]["architecture"]["layer_count"] == 2
    assert resolved[0]["artifact_id"] == result.artifact_id


@pytest.mark.parametrize("catalog_mode", ("r0", "full", "oam"))
def test_required_installed_cuda_artifact_resolves_from_any_catalog_with_exact_oam(
    monkeypatch, tmp_path, catalog_mode
):
    cubin = tmp_path / "plugin.cubin"
    manifest = tmp_path / "manifest.json"
    cubin.write_bytes(b"installed-cubin")
    manifest.write_text("{}", encoding="utf-8")
    resolved = []

    def resolve(**kwargs):
        resolved.append(kwargs)
        return SimpleNamespace(
            available=True,
            status="resolved",
            artifact_path=cubin,
            manifest_path=manifest,
            diagnostics=("origin=prebuilt status=loaded family=OAM",),
        )

    native = SimpleNamespace(
        _backend_build_info=lambda: {
            "prebuilt_operator_artifacts": {
                "mode": catalog_mode,
                "backend": "cuda",
                "target": "sm_120",
            }
        }
    )
    monkeypatch.setattr(artifact_module, "resolve_installed_oam_artifact", resolve)
    monkeypatch.setattr(
        artifact_module,
        "resolve_jit_policy",
        lambda: pytest.fail("required prebuilt must not resolve JIT policy"),
    )
    monkeypatch.setattr(
        artifact_module,
        "_validated_compiler",
        lambda compiler: pytest.fail("required prebuilt must not validate a compiler"),
    )
    monkeypatch.setattr(
        artifact_module,
        "_selected_compiler",
        lambda compiler: pytest.fail("required prebuilt must not probe a compiler"),
    )

    result = prepare_nequip_oam_cuda_artifact(
        FIXTURE,
        "float32",
        "sm_120",
        native_module=native,
        installed_prebuilt_required=True,
        expected_target="sm120",
    )

    assert result.status == "installed"
    assert result.target == "sm120"
    assert resolved[0]["artifact_id"] == result.artifact_id


@pytest.mark.parametrize("status", ("absent", "rejected"))
def test_required_installed_cuda_artifact_fails_before_compiler_fallback(
    monkeypatch, status
):
    native = SimpleNamespace(
        _backend_build_info=lambda: {
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "cuda",
                "target": "sm120",
            }
        }
    )
    monkeypatch.setattr(
        artifact_module,
        "resolve_installed_oam_artifact",
        lambda **kwargs: SimpleNamespace(
            available=False,
            status=status,
            artifact_path=None,
            manifest_path=None,
            diagnostics=(f"origin=prebuilt status={status} reason=test",),
        ),
    )
    monkeypatch.setattr(
        artifact_module,
        "resolve_jit_policy",
        lambda: pytest.fail("required prebuilt must fail before JIT policy"),
    )
    monkeypatch.setattr(
        artifact_module,
        "_validated_compiler",
        lambda compiler: pytest.fail("required prebuilt must fail before validation"),
    )
    monkeypatch.setattr(
        artifact_module,
        "_selected_compiler",
        lambda compiler: pytest.fail("required prebuilt must fail before compiler"),
    )

    with pytest.raises(OAMCudaArtifactError, match=f"required installed.*{status}"):
        prepare_nequip_oam_cuda_artifact(
            FIXTURE,
            "float32",
            "sm_120",
            native_module=native,
            installed_prebuilt_required=True,
            expected_target="sm120",
        )


def test_required_installed_cuda_artifact_rejects_target_before_lookup(monkeypatch):
    native = SimpleNamespace(
        _backend_build_info=lambda: {
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "cuda",
                "target": "sm90",
            }
        }
    )
    monkeypatch.setattr(
        artifact_module,
        "resolve_installed_oam_artifact",
        lambda **kwargs: pytest.fail("target mismatch must precede catalog lookup"),
    )
    monkeypatch.setattr(
        artifact_module,
        "resolve_jit_policy",
        lambda: pytest.fail("target mismatch must precede JIT policy"),
    )
    monkeypatch.setattr(
        artifact_module,
        "_validated_compiler",
        lambda compiler: pytest.fail("target mismatch must precede validation"),
    )
    monkeypatch.setattr(
        artifact_module,
        "_selected_compiler",
        lambda compiler: pytest.fail("target mismatch must precede compiler"),
    )

    with pytest.raises(OAMCudaArtifactError, match="expected CUDA target.*sm120"):
        prepare_nequip_oam_cuda_artifact(
            FIXTURE,
            "float32",
            "sm_120",
            native_module=native,
            installed_prebuilt_required=True,
            expected_target="sm120",
        )


def test_required_installed_cuda_artifact_rejects_stale_identity_before_lookup(
    monkeypatch,
):
    native = SimpleNamespace(
        _backend_build_info=lambda: {
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "cuda",
                "target": "sm120",
            }
        }
    )
    monkeypatch.setattr(
        artifact_module,
        "resolve_installed_oam_artifact",
        lambda **kwargs: pytest.fail("identity mismatch must precede catalog lookup"),
    )
    monkeypatch.setattr(
        artifact_module,
        "resolve_jit_policy",
        lambda: pytest.fail("identity mismatch must precede JIT policy"),
    )
    monkeypatch.setattr(
        artifact_module,
        "_validated_compiler",
        lambda compiler: pytest.fail("identity mismatch must precede validation"),
    )
    monkeypatch.setattr(
        artifact_module,
        "_selected_compiler",
        lambda compiler: pytest.fail("identity mismatch must precede compiler"),
    )

    with pytest.raises(OAMCudaArtifactError, match="identity mismatch"):
        prepare_nequip_oam_cuda_artifact(
            FIXTURE,
            "float32",
            "sm_120",
            native_module=native,
            installed_prebuilt_required=True,
            expected_target="sm120",
            expected_artifact_id="stale-artifact",
        )


@pytest.mark.parametrize(
    "field",
    (
        "contract_fingerprint",
        "semantic_fingerprint",
        "forward_owner_schedule",
        "forward_static_shared_bytes",
    ),
)
def test_contract_mismatch_is_quarantined_and_rebuilt(
    monkeypatch, tmp_path, required_policy, field
):
    _stub_renderer(monkeypatch)
    _real_nvrtc_cache(monkeypatch)

    first = prepare_nequip_oam_cuda_artifact(
        FIXTURE, "float32", "sm_90", "nvrtc", cache_root=tmp_path
    )
    manifest_path = first.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A published entry bound to another contract must never be reused.
    manifest["key_inputs"]["build"]["request"]["oam"][field] = "stale-identity"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rebuilt = prepare_nequip_oam_cuda_artifact(
        FIXTURE, "float32", "sm_90", "nvrtc", cache_root=tmp_path
    )

    assert rebuilt.status == "built"
    assert rebuilt.cache_key == first.cache_key
    assert any("invalid cached JIT entry" in value for value in rebuilt.diagnostics)
    assert any("quarantined" in value for value in rebuilt.diagnostics)


def test_retrained_weights_reuse_the_same_architecture_artifact(
    monkeypatch, tmp_path, required_policy
):
    _stub_renderer(monkeypatch)
    _real_nvrtc_cache(monkeypatch)
    other = _variant_contract(tmp_path, _bump_first_tensor_product_weight)

    baseline = prepare_nequip_oam_cuda_artifact(
        FIXTURE, "float32", "sm_90", "nvrtc", cache_root=tmp_path
    )
    changed = prepare_nequip_oam_cuda_artifact(
        other, "float32", "sm_90", "nvrtc", cache_root=tmp_path
    )

    assert changed.status == "cached"
    assert changed.artifact_id == baseline.artifact_id
    assert changed.cache_key == baseline.cache_key


def test_generation_policy_uses_a_distinct_cache_identity(
    monkeypatch, tmp_path, required_policy
):
    _stub_renderer(monkeypatch)
    _real_nvrtc_cache(monkeypatch)

    baseline = prepare_nequip_oam_cuda_artifact(
        FIXTURE, "float32", "sm_120", "nvrtc", cache_root=tmp_path
    )
    policy = OAMCudaGenerationPolicy(forward_live_output_budget=16)
    grouped = prepare_nequip_oam_cuda_artifact(
        FIXTURE,
        "float32",
        "sm_120",
        "nvrtc",
        cache_root=tmp_path,
        generation_policy=policy,
    )

    assert grouped.artifact_id != baseline.artifact_id
    assert grouped.cache_key != baseline.cache_key
    manifest = json.loads(grouped.manifest_path.read_text(encoding="utf-8"))
    bound = manifest["key_inputs"]["build"]["request"]["oam"]
    assert bound["generation_policy"] == policy.__dict__
    assert "projected_channel_tile" not in bound
    assert "projected_threads_per_block" not in bound
    assert "projected_grid_policy" not in bound
    assert bound["target_policy"] == "exact-compute-capability-v1"
    assert bound["precision_policy"] == "exact-float32-v1"
    assert bound["schedule_qualification"] == "candidate-unqualified-explicit"


def test_prepare_uses_qualified_sm120_fp32_policy_by_default(
    monkeypatch, tmp_path, required_policy
):
    _stub_renderer(monkeypatch)
    _real_nvrtc_cache(monkeypatch)

    result = prepare_nequip_oam_cuda_artifact(
        FIXTURE, "float32", "sm_120", "nvrtc", cache_root=tmp_path
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    bound = manifest["key_inputs"]["build"]["request"]["oam"]
    assert bound["generation_policy"]["radial_adjoint_mode"] == "fp32"


def test_target_and_precision_change_the_cache_identity(
    monkeypatch, tmp_path, required_policy
):
    _stub_renderer(monkeypatch)
    _real_nvrtc_cache(monkeypatch)

    def prepare(target, precision):
        return prepare_nequip_oam_cuda_artifact(
            FIXTURE, precision, target, "nvrtc", cache_root=tmp_path
        )

    sm90 = prepare("sm_90", "float32")
    sm120 = prepare("sm_120", "float32")
    double = prepare("sm_90", "float64")
    environment = prepare(
        {"architecture": "sm_120", "compute_capability_code": 120}, "float32"
    )

    assert sm90.artifact_id != sm120.artifact_id
    assert len({sm90.cache_key, sm120.cache_key, double.cache_key}) == 3
    # A device-environment mapping names the same target as its string form.
    assert environment.artifact_id == sm120.artifact_id
    assert environment.cache_key == sm120.cache_key


def test_disabled_policy_raises_because_no_fallback_exists(
    monkeypatch, tmp_path, required_policy
):
    _stub_renderer(monkeypatch)
    _real_nvrtc_cache(monkeypatch)
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "none")

    expected = (
        r"(?s)no fallback execution.*"
        r"Direct R1 artifact diagnostic \([^)]+\): .*"
        r"Direct R1 artifact causes: \(a\) .*\(b\) .*\(c\) .*\(d\) "
    )
    with pytest.raises(OAMCudaArtifactError, match=expected):
        prepare_nequip_oam_cuda_artifact(
            FIXTURE, "float32", "sm_90", "nvrtc", cache_root=tmp_path
        )


def test_plain_nequip_contract_is_rejected(tmp_path, required_policy):
    with pytest.raises(OAMCudaArtifactError, match="must be 2"):
        prepare_nequip_oam_cuda_artifact(
            PLAIN_NEQUIP_FIXTURE, "float32", "sm_90", "nvrtc", cache_root=tmp_path
        )


def test_unreadable_contract_raises(tmp_path, required_policy):
    with pytest.raises(OAMCudaArtifactError, match="could not read"):
        prepare_nequip_oam_cuda_artifact(
            tmp_path / "missing.json", "float32", "sm_90", "nvrtc", cache_root=tmp_path
        )


def test_backend_selection_failure_raises(monkeypatch, tmp_path, required_policy):
    def unavailable(request=None):
        raise RuntimeError("no CUDA JIT backend in this environment")

    monkeypatch.setattr(
        artifact_module, "select_execution_cuda_jit_backend", unavailable
    )
    with pytest.raises(OAMCudaArtifactError, match="backend is unavailable"):
        prepare_nequip_oam_cuda_artifact(
            FIXTURE, "float32", "sm_90", cache_root=tmp_path
        )
