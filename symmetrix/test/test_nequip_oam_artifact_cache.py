"""Cache-key invalidation for generated OAM host artifacts.

The prepared artifact is keyed by the rendered source plus the build metadata
(generator, precision, codegen metadata, compiler identity, flags).  The
existing codegen tests assert same-request *reuse* and metadata-fingerprint
sensitivity, which a regression that dropped the semantic identity from the key
would still satisfy.  These tests assert the other half of the contract: a
changed contract, precision, or compiler identity must produce a different
``cache_key``/``artifact_path`` and a rebuild rather than a stale reuse.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import symmetrix.jit as jit
import symmetrix.oam_jit_artifact as artifact_module
from symmetrix.oam_jit_artifact import (
    OAMHostArtifactError,
    prepare_nequip_oam_host_artifact,
    prepare_nequip_oam_receiver_host_artifact,
)

FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"

pytestmark = pytest.mark.skipif(
    shutil.which(os.environ.get("CXX", "c++")) is None,
    reason="a C++ compiler is required for the generated OAM host plugin",
)

PREPARERS = {
    "edge-v1": prepare_nequip_oam_host_artifact,
    "receiver-v2": prepare_nequip_oam_receiver_host_artifact,
}


def _contract() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _write(path: Path, contract: dict) -> Path:
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


@pytest.fixture(params=sorted(PREPARERS), ids=sorted(PREPARERS))
def prepare(request):
    return PREPARERS[request.param]


def test_changed_contract_invalidates_the_cached_artifact(prepare, tmp_path):
    cache = tmp_path / "cache"
    baseline = _write(tmp_path / "baseline.json", _contract())
    changed_contract = copy.deepcopy(_contract())
    changed_contract["layers"][0]["edge_tensor_product"]["instructions"][0][
        "path_weight"
    ] = 2.0
    changed = _write(tmp_path / "changed.json", changed_contract)

    first = prepare(baseline, "float64", cache_dir=cache)
    reused = prepare(baseline, "float64", cache_dir=cache)
    rebuilt = prepare(changed, "float64", cache_dir=cache)

    assert reused.cache_key == first.cache_key
    assert reused.artifact_path == first.artifact_path
    assert reused.cache_hit is True

    assert rebuilt.cache_key != first.cache_key
    assert rebuilt.artifact_path != first.artifact_path
    assert rebuilt.cache_hit is False


def test_installed_receiver_artifact_avoids_runtime_compilation(monkeypatch, tmp_path):
    contract = _write(tmp_path / "model.json", _contract())
    plugin = tmp_path / "installed-oam.so"
    manifest = tmp_path / "installed-oam.json"
    plugin.write_bytes(b"installed")
    manifest.write_text("{}", encoding="utf-8")

    resolved = []

    def resolve(**kwargs):
        resolved.append(kwargs)
        return SimpleNamespace(
            available=True,
            status="resolved",
            artifact_path=plugin,
            manifest_path=manifest,
            diagnostics=("origin=prebuilt status=loaded family=OAM",),
        )

    monkeypatch.setattr(artifact_module, "resolve_installed_oam_artifact", resolve)
    monkeypatch.setattr(
        artifact_module,
        "prepare_jit_artifact",
        lambda *args, **kwargs: pytest.fail("runtime compiler must not be called"),
    )
    monkeypatch.setattr(
        artifact_module,
        "probe_compiler_identity",
        lambda: pytest.fail("installed prebuilt must not probe a compiler"),
    )
    native_module = SimpleNamespace(
        _backend_build_info=lambda: {
            "fast_math": False,
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "host",
                "target": "x86-64-v3",
            },
        }
    )

    result = prepare_nequip_oam_receiver_host_artifact(
        contract,
        "float64",
        cache_dir=tmp_path / "cache",
        native_module=native_module,
        installed_prebuilt_required=True,
        expected_host_target="x86-64-v3",
        expected_floating_point_policy="strict",
    )

    assert result.status == "installed"
    assert result.artifact_path == plugin
    assert result.manifest_path == manifest
    assert result.compiler_command == ()
    assert result.cache_hit is False
    assert result.artifact_origin == "prebuilt"
    assert result.active_host_target == "host-isa-x86-64-v3"
    assert result.expected_host_target == "host-isa-x86-64-v3"
    assert result.expected_artifact_origin == "prebuilt"
    assert result.floating_point_policy == "strict"
    assert resolved[0]["host_target"] == "host-isa-x86-64-v3"
    assert resolved[0]["floating_point_policy"] == "strict"
    assert resolved[0]["artifact_id"] == result.artifact_id
    assert resolved[0]["architecture"]["layer_count"] == 2
    assert "semantic_fingerprint" not in resolved[0]


@pytest.mark.parametrize("resolution_status", ("absent", "rejected"))
def test_required_installed_receiver_fails_before_compiler_fallback(
    monkeypatch, tmp_path, resolution_status
):
    contract = _write(tmp_path / "model.json", _contract())
    native_module = SimpleNamespace(
        _backend_build_info=lambda: {
            "fast_math": True,
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "host",
                "target": "x86-64-v3",
            },
        }
    )
    monkeypatch.setattr(
        artifact_module,
        "resolve_installed_oam_artifact",
        lambda **kwargs: SimpleNamespace(
            available=False,
            status=resolution_status,
            artifact_path=None,
            manifest_path=None,
            diagnostics=(f"origin=prebuilt status={resolution_status} reason=test",),
        ),
    )
    monkeypatch.setattr(
        artifact_module,
        "probe_compiler_identity",
        lambda: pytest.fail("required prebuilt must fail before compiler probing"),
    )

    with pytest.raises(
        OAMHostArtifactError,
        match=f"required installed OAM host artifact.*{resolution_status}",
    ):
        prepare_nequip_oam_receiver_host_artifact(
            contract,
            "float32",
            native_module=native_module,
            installed_prebuilt_required=True,
            expected_host_target="x86-64-v3",
            expected_floating_point_policy="fast",
        )


def test_required_installed_receiver_rejects_wrong_target_before_compiler(
    monkeypatch, tmp_path
):
    contract = _write(tmp_path / "model.json", _contract())
    native_module = SimpleNamespace(
        _backend_build_info=lambda: {
            "fast_math": False,
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "host",
                "target": "x86-64-v2",
            },
        }
    )
    monkeypatch.setattr(
        artifact_module,
        "probe_compiler_identity",
        lambda: pytest.fail("wrong target must fail before compiler probing"),
    )

    with pytest.raises(OAMHostArtifactError, match="expected host target.*x86-64-v3"):
        prepare_nequip_oam_receiver_host_artifact(
            contract,
            native_module=native_module,
            installed_prebuilt_required=True,
            expected_host_target="x86-64-v3",
            expected_floating_point_policy="strict",
        )


def test_required_installed_receiver_rejects_stale_identity_before_compiler(
    monkeypatch, tmp_path
):
    contract = _write(tmp_path / "model.json", _contract())
    native_module = SimpleNamespace(
        _backend_build_info=lambda: {
            "fast_math": False,
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "host",
                "target": "x86-64-v3",
            },
        }
    )
    monkeypatch.setattr(
        artifact_module,
        "probe_compiler_identity",
        lambda: pytest.fail("stale identity must fail before compiler probing"),
    )
    monkeypatch.setattr(
        artifact_module,
        "resolve_installed_oam_artifact",
        lambda **kwargs: pytest.fail("request identity mismatch must precede lookup"),
    )

    with pytest.raises(OAMHostArtifactError, match="identity mismatch.*regenerate"):
        prepare_nequip_oam_receiver_host_artifact(
            contract,
            native_module=native_module,
            installed_prebuilt_required=True,
            expected_host_target="x86-64-v3",
            expected_artifact_id="oam-receiver-v4-float64-stale",
            expected_floating_point_policy="strict",
        )


def test_required_installed_receiver_requires_native_fast_math_evidence(
    monkeypatch, tmp_path
):
    contract = _write(tmp_path / "model.json", _contract())
    native_module = SimpleNamespace(
        _backend_build_info=lambda: {
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "host",
                "target": "x86-64-v3",
            }
        }
    )
    monkeypatch.setattr(
        artifact_module,
        "probe_compiler_identity",
        lambda: pytest.fail("missing build evidence must fail before compiler probing"),
    )

    with pytest.raises(OAMHostArtifactError, match="native fast_math build info"):
        prepare_nequip_oam_receiver_host_artifact(
            contract,
            native_module=native_module,
            installed_prebuilt_required=True,
            expected_host_target="x86-64-v3",
            expected_floating_point_policy="strict",
        )


@pytest.mark.parametrize(
    ("native_fast_math", "expected_policy"),
    ((False, "fast"), (True, "strict")),
)
def test_required_installed_receiver_rejects_strict_fast_mismatch_before_compiler(
    monkeypatch, tmp_path, native_fast_math, expected_policy
):
    contract = _write(tmp_path / "model.json", _contract())
    native_module = SimpleNamespace(
        _backend_build_info=lambda: {
            "fast_math": native_fast_math,
            "prebuilt_operator_artifacts": {
                "mode": "oam",
                "backend": "host",
                "target": "x86-64-v3",
            },
        }
    )
    monkeypatch.setattr(
        artifact_module,
        "probe_compiler_identity",
        lambda: pytest.fail("policy mismatch must fail before compiler probing"),
    )

    with pytest.raises(OAMHostArtifactError, match="policy mismatch"):
        prepare_nequip_oam_receiver_host_artifact(
            contract,
            native_module=native_module,
            installed_prebuilt_required=True,
            expected_host_target="x86-64-v3",
            expected_floating_point_policy=expected_policy,
        )


@pytest.mark.parametrize(
    "argument",
    ("expected_host_target", "expected_artifact_id", "expected_floating_point_policy"),
)
def test_runtime_generation_rejects_installed_only_expectations(
    monkeypatch, tmp_path, argument
):
    contract = _write(tmp_path / "model.json", _contract())
    values = {
        "expected_host_target": "x86-64-v3",
        "expected_artifact_id": "stale-id",
        "expected_floating_point_policy": "fast",
    }
    monkeypatch.setattr(
        artifact_module,
        "probe_compiler_identity",
        lambda: pytest.fail(
            "invalid installed expectation must precede compiler probe"
        ),
    )

    with pytest.raises(OAMHostArtifactError, match="installed-prebuilt-required"):
        prepare_nequip_oam_receiver_host_artifact(
            contract,
            **{argument: values[argument]},
        )


@pytest.mark.parametrize(
    ("host_flags", "expected_policy"),
    (
        (("-O3", "-march=x86-64-v3"), "strict"),
        (("-O3", "-march=x86-64-v3", "-ffast-math"), "fast"),
        (("-ffast-math", "-fno-fast-math", "-march=x86-64-v3"), "strict"),
        (("-fno-fast-math", "-ffast-math", "-march=x86-64-v3"), "fast"),
        (("-Ofast", "-fno-fast-math", "-march=x86-64-v3"), "strict"),
        (("-fno-fast-math", "-Ofast", "-march=x86-64-v3"), "fast"),
    ),
)
def test_runtime_generation_policy_follows_effective_compile_flags(
    tmp_path, host_flags, expected_policy
):
    contract = _write(tmp_path / "model.json", _contract())
    artifact = prepare_nequip_oam_receiver_host_artifact(
        contract,
        cache_dir=tmp_path / expected_policy,
        host_flags=host_flags,
    )

    assert artifact.floating_point_policy == expected_policy
    assert artifact.metadata["floating_point_policy"] == expected_policy


@pytest.mark.parametrize(
    "host_flag",
    (
        "-funsafe-math-optimizations",
        "-ffinite-math-only",
        "-ffp-contract=fast",
        "-ffast-math=yes",
        "/fp:fast",
    ),
)
def test_runtime_generation_rejects_partial_floating_point_flags(tmp_path, host_flag):
    contract = _write(tmp_path / "model.json", _contract())
    with pytest.raises(OAMHostArtifactError, match="floating-point flag"):
        prepare_nequip_oam_receiver_host_artifact(
            contract,
            cache_dir=tmp_path / "rejected",
            host_flags=("-O3", host_flag, "-march=x86-64-v3"),
        )


@pytest.mark.parametrize(
    ("flags", "expected"),
    (
        (("-fno-fast-math", "-ffp-contract=off", "-ffast-math"), "fast"),
        (("-ffast-math", "-fno-fast-math", "-ffp-contract=off"), "strict"),
        (("-Ofast", "-fno-fast-math", "-ffp-contract=off"), "strict"),
    ),
)
def test_floating_point_contract_off_uses_policy_at_its_position(flags, expected):
    assert jit.resolve_host_floating_point_policy(flags) == expected


@pytest.mark.parametrize(
    "flags",
    (
        ("-fno-fast-math", "-ffast-math", "-ffp-contract=off"),
        ("-fno-fast-math", "-Ofast", "-ffp-contract=off"),
        ("-ffp-contract=off", "-fno-fast-math"),
    ),
)
def test_floating_point_contract_off_rejects_non_strict_current_policy(flags):
    with pytest.raises(jit.JitError, match="only supported.*strict selector"):
        jit.resolve_host_floating_point_policy(flags)


def test_changed_precision_invalidates_the_cached_artifact(prepare, tmp_path):
    cache = tmp_path / "cache"
    baseline = _write(tmp_path / "baseline.json", _contract())

    double = prepare(baseline, "float64", cache_dir=cache)
    single = prepare(baseline, "float32", cache_dir=cache)

    assert single.cache_key != double.cache_key
    assert single.artifact_path != double.artifact_path
    assert single.cache_hit is False
    assert single.metadata["scalar_size"] == 4
    assert double.metadata["scalar_size"] == 8


def test_changed_compiler_identity_invalidates_the_cached_artifact(
    prepare, tmp_path, monkeypatch
):
    """A different compiler must not reuse an artifact built by the old one."""

    cache = tmp_path / "cache"
    baseline = _write(tmp_path / "baseline.json", _contract())
    real_probe = jit.probe_compiler_identity

    first = prepare(baseline, "float64", cache_dir=cache)
    assert first.cache_hit is False
    assert prepare(baseline, "float64", cache_dir=cache).cache_hit is True

    def altered_probe(cxx=None, **kwargs):
        command, identity = real_probe(cxx, **kwargs)
        # The real compiler still builds the artifact; only the recorded
        # identity changes, so the key must move and the entry must be rebuilt.
        return command, {**identity, "version": identity["version"] + "\n# other"}

    monkeypatch.setattr(jit, "probe_compiler_identity", altered_probe)
    other = prepare(baseline, "float64", cache_dir=cache)

    assert other.cache_key != first.cache_key
    assert other.artifact_path != first.artifact_path
    assert other.cache_hit is False

    monkeypatch.undo()
    restored = prepare(baseline, "float64", cache_dir=cache)
    assert restored.cache_key == first.cache_key
    assert restored.artifact_path == first.artifact_path
    assert restored.cache_hit is True


def test_loader_rejects_an_artifact_with_another_architecture_identity(tmp_path):
    """The loaded architecture must match the artifact's semantic identity."""

    native = pytest.importorskip("symmetrix.symmetrix")
    evaluator_type = getattr(native, "NequIPOAMKokkos", None)
    if evaluator_type is None:
        pytest.skip("this build has no NequIP OAM Kokkos evaluator")
    # The native evaluator allocates Kokkos state, so the host must initialize
    # the execution space first, exactly as the calculator does.
    if not native._kokkos_is_initialized():
        native._init_kokkos()

    contract = _write(tmp_path / "baseline.json", _contract())
    artifact = prepare_nequip_oam_receiver_host_artifact(
        contract, "float64", cache_dir=tmp_path / "cache"
    )
    manifest = Path(artifact.artifact_path).parent / "manifest.json"
    recorded = json.loads(manifest.read_text(encoding="utf-8"))

    # Positive control: the intact identity is accepted.
    evaluator_type(
        str(contract),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )

    tampered = copy.deepcopy(recorded)
    tampered["key_inputs"]["build"]["metadata"]["semantic_fingerprint"] = (
        "sha256:" + "0" * 64
    )
    manifest.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic fingerprint"):
        evaluator_type(
            str(contract),
            str(artifact.artifact_path),
            host_active_target=artifact.active_host_target,
            host_artifact_origin=artifact.artifact_origin,
        )


def test_loader_reuses_full_message_artifact_across_post_linear_values(tmp_path):
    native = pytest.importorskip("symmetrix.symmetrix")
    evaluator_type = getattr(native, "NequIPOAMKokkos", None)
    if evaluator_type is None:
        pytest.skip("this build has no NequIP OAM Kokkos evaluator")
    if not native._kokkos_is_initialized():
        native._init_kokkos()

    baseline = _write(tmp_path / "baseline.json", _contract())
    artifact = prepare_nequip_oam_receiver_host_artifact(
        baseline, "float64", cache_dir=tmp_path / "cache"
    )
    changed_data = _contract()
    mask = changed_data["layers"][0]["post_linear"]["output_mask"]["values"]
    mask[0] = 0.0 if mask[0] else 1.0
    changed = _write(tmp_path / "changed.json", changed_data)

    evaluator = evaluator_type(
        str(changed),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
    assert evaluator.jit_oam_host_plugin_artifact_id == artifact.artifact_id
    del evaluator


@pytest.mark.parametrize(
    "field",
    (
        "schedule_identity",
        "host_target",
        "forward_owner_schedule",
        "full_reverse_owner_schedule",
        "full_reverse_reduction_policy",
        "tiling_policy",
        "vector_policy",
        "shared_memory_policy",
        "register_budget_policy",
        "direct_geometry_policy",
    ),
)
def test_full_message_v7_loader_rejects_missing_mandatory_metadata(tmp_path, field):
    native = pytest.importorskip("symmetrix.symmetrix")
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    contract = _write(tmp_path / "model.json", _contract())
    artifact = prepare_nequip_oam_receiver_host_artifact(
        contract, "float64", cache_dir=tmp_path / "cache"
    )
    manifest = artifact.manifest_path
    data = json.loads(manifest.read_text(encoding="utf-8"))
    del data["key_inputs"]["build"]["metadata"][field]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="regenerate|invalid identity metadata"):
        native.NequIPOAMKokkos(
            str(contract),
            str(artifact.artifact_path),
            host_active_target=artifact.active_host_target,
            host_artifact_origin=artifact.artifact_origin,
        )


def test_full_message_v7_loader_rejects_incompatible_active_host_target(tmp_path):
    native = pytest.importorskip("symmetrix.symmetrix")
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    contract = _write(tmp_path / "model.json", _contract())
    artifact = prepare_nequip_oam_receiver_host_artifact(
        contract, "float64", cache_dir=tmp_path / "cache"
    )

    with pytest.raises(ValueError, match="incompatible with the active runtime"):
        native.NequIPOAMKokkos(
            str(contract),
            str(artifact.artifact_path),
            host_active_target="host-native-sha256-" + "0" * 64,
            host_artifact_origin=artifact.artifact_origin,
        )


def test_cached_host_artifact_reports_cache_origin(tmp_path):
    native = pytest.importorskip("symmetrix.symmetrix")
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    contract = _write(tmp_path / "model.json", _contract())
    prepare_nequip_oam_receiver_host_artifact(
        contract, "float64", cache_dir=tmp_path / "cache"
    )
    cached = prepare_nequip_oam_receiver_host_artifact(
        contract, "float64", cache_dir=tmp_path / "cache"
    )
    evaluator = native.NequIPOAMKokkos(
        str(contract),
        str(cached.artifact_path),
        host_active_target=cached.active_host_target,
        host_artifact_origin=cached.artifact_origin,
    )

    assert cached.cache_hit is True
    report = evaluator.execution_plan_report
    assert report["artifact_origin"] == "cache"
    assert {
        candidate["artifact_origin"]
        for candidate in report["candidates"]
        if candidate["id"].startswith("oam-direct-")
    } == {"cache"}
