import importlib
import importlib.util
import json
import shutil
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

_PACKAGE_NAME = "_symmetrix_calc_jit_test"
_PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "symmetrix"
_PACKAGE = types.ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_PACKAGE_SOURCE)]
sys.modules[_PACKAGE_NAME] = _PACKAGE
_NATIVE = types.ModuleType(f"{_PACKAGE_NAME}.symmetrix")
sys.modules[f"{_PACKAGE_NAME}.symmetrix"] = _NATIVE
_SPEC = importlib.util.spec_from_file_location(
    f"{_PACKAGE_NAME}.calculator", _PACKAGE_SOURCE / "calculator.py"
)
symmetrix_calc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = symmetrix_calc
_SPEC.loader.exec_module(symmetrix_calc)
Symmetrix = symmetrix_calc.Symmetrix


def _load_unpatched_jit():
    name = f"{_PACKAGE_NAME}.jit_real_compiler"
    spec = importlib.util.spec_from_file_location(name, _PACKAGE_SOURCE / "jit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_METADATA = {
    "abi": "symmetrix.jit.host-plugin/2",
    "abi_version": 2,
    "artifact_id": "jit-r1-fixture",
    "contract_fingerprint": "sha256:generation",
    "semantic_fingerprint": "sha256:semantic",
    "structure_fingerprint": "sha256:structure",
    "channels": 7,
    "embedding": 5,
    "edge_l_max": 2,
    "source_l_max": 2,
    "output_components": 9,
    "precision": "float32",
    "scalar_kind": 1,
    "scalar_size": 4,
}
_CUDA_METADATA = {
    **_METADATA,
    "abi": "symmetrix.jit.cuda-plugin/2",
    "target_compute_capability": 120,
    "forward_threads_per_block": 256,
    "source_threads_per_block": 256,
    "edge_threads_per_block": 32,
    "edge_strategy": "wave",
    "edge_logical_subgroup_width": 32,
    "persistent_blocks_per_compute_unit": 8,
}
_HIP_METADATA = {
    **_METADATA,
    "abi": "symmetrix.jit.hip-plugin/1",
    "abi_version": 1,
    "target_architecture": "gfx1151",
    "target_features": "",
    "native_subgroup_width": 32,
    "forward_threads_per_block": 256,
    "source_threads_per_block": 256,
    "edge_threads_per_block": 64,
    "edge_strategy": "wave",
    "edge_logical_subgroup_width": 32,
    "persistent_blocks_per_compute_unit": 8,
}
_MH1_METADATA = {
    "abi": "symmetrix.jit.mh1.host-plugin/3",
    "abi_version": 3,
    "artifact_id": "jit-mh1-v3-fixture",
    "generation_fingerprint": "sha256:generation",
    "semantic_fingerprint": "sha256:semantic",
    "structure_fingerprint": "sha256:structure",
    "conditioner_layout": {
        "tag": "symmetrix.execution.mh1.conditioner-layout/1",
        "generation_fingerprint": "sha256:generation",
        "interactions": [],
    },
    "interactions": [],
}
_MH1_V4_METADATA = {
    "abi": "symmetrix.jit.mh1.host-plugin/4",
    "abi_version": 4,
    "artifact_id": "jit-mh1-v4-fixture",
    "generation_fingerprint": "sha256:generation-v4",
    "semantic_fingerprint": "sha256:semantic-v4",
    "structure_fingerprint": "sha256:structure-v4",
    "runtime_layout_fingerprint": "sha256:runtime-v4",
    "layers": [],
    "node_state_policy": "full-retention-v1",
}
_MH1_V5_METADATA = {
    **_MH1_V4_METADATA,
    "abi": "symmetrix.jit.mh1.host-plugin/5",
    "abi_version": 5,
}
_MH1_CUDA_METADATA = {
    **_MH1_METADATA,
    "abi": "symmetrix.jit.mh1.cuda-plugin/3",
    "target_compute_capability": 120,
    "forward_threads_per_block": 128,
    "source_threads_per_block": 128,
    "edge_threads_per_block": 128,
}
_MH1_CUDA_FORWARD_POLICY = {
    "tag": "symmetrix.jit.mh1.cuda-forward-policy/1",
    "generation_fingerprint": "sha256:generation",
    "target_compute_capability": 120,
    "interactions": [],
    "policy_id": f"sha256:{'a' * 64}",
}
_MH1_CUDA_SOURCE_POLICY = {
    "tag": "symmetrix.jit.mh1.cuda-source-policy/1",
    "generation_fingerprint": "sha256:generation",
    "target_compute_capability": 120,
    "interactions": [],
    "policy_id": f"sha256:{'b' * 64}",
}
_MH1_CUDA_EDGE_POLICY = {
    "tag": "symmetrix.jit.mh1.cuda-edge-policy/2",
    "generation_fingerprint": "sha256:generation",
    "target_compute_capability": 120,
    "reverse_schedule": "hybrid-path-tiled-v7",
    "interactions": [],
    "policy_id": f"sha256:{'c' * 64}",
}
_MH1_CUDA_METADATA_WITH_POLICY = {
    **_MH1_CUDA_METADATA,
    "forward_policy": _MH1_CUDA_FORWARD_POLICY,
    "forward_policy_id": _MH1_CUDA_FORWARD_POLICY["policy_id"],
    "source_policy": _MH1_CUDA_SOURCE_POLICY,
    "source_policy_id": _MH1_CUDA_SOURCE_POLICY["policy_id"],
    "edge_policy": _MH1_CUDA_EDGE_POLICY,
    "edge_policy_id": _MH1_CUDA_EDGE_POLICY["policy_id"],
}
_MH1_CUDA_V4_METADATA_WITH_POLICY = {
    **_MH1_CUDA_METADATA_WITH_POLICY,
    "abi": "symmetrix.jit.mh1.cuda-plugin/4",
    "abi_version": 4,
    "artifact_id": "jit-mh1-v4-fixture",
    "generation_fingerprint": "sha256:generation-v4",
    "semantic_fingerprint": "sha256:semantic-v4",
    "structure_fingerprint": "sha256:structure-v4",
    "runtime_layout_fingerprint": "sha256:runtime-v4",
    "layers": [],
    "node_state_policy": "full-retention-v1",
}


class _DummyEvaluator:
    r_cut = 5.0

    def __init__(self, events):
        self.events = events
        self.streamed_edges_mode = "materialized"
        self.load_error = None
        self.jit_cuda_plugin_ready = False
        self.jit_cuda_plugin_artifact_id = ""
        self.device_cuda_environment = {
            "available": True,
            "backend": "cuda",
            "device_name": "test GPU",
            "device_ordinal": 0,
            "compute_capability": "12.0",
            "compute_capability_code": 120,
            "multiprocessor_count": 8,
            "warp_width": 32,
            "runtime_version": 13030,
            "driver_version": 13030,
        }
        self.execution_device_execution_environment = dict(self.device_cuda_environment)
        self.jit_hip_plugin_ready = False
        self.jit_hip_plugin_artifact_id = ""
        self.jit_host_plugin_ready = False
        self.jit_host_plugin_artifact_id = ""
        self.receiver_factorized_host_plugin_ready = False
        self.receiver_factorized_host_plugin_artifact_id = ""
        self.jit_mh1_host_plugin_ready = False
        self.jit_mh1_host_plugin_v4_ready = False
        self.jit_mh1_host_plugin_v5_ready = False
        self.jit_mh1_host_plugin_artifact_id = ""
        self.jit_mh1_cuda_plugin_ready = False
        self.jit_mh1_cuda_plugin_v4_ready = False
        self.jit_mh1_cuda_plugin_artifact_id = ""
        self.execution_mh1_scratch_budget_bytes = None
        self.execution_mh1_node_arena_policy = "throughput-v1"
        self.execution_mh1_node_arena_tile_rows = 1024
        self.m1_polynomial_policy = "retained"
        self.m1_recompute_fallback_reason = ""
        self.mh0_state_policy = "full-retention-v1"
        self.mh0_state_policy_fallback_reason = ""
        self.edge_geometry_policy = "cartesian-f64-v1"
        self.low_memory_geometry_policy = "unit-f32-radius-f64-v1"
        self.low_memory = False

    def set_streamed_edges(self, mode):
        self.events.append(("mode", mode))
        self.streamed_edges_mode = mode

    def _set_factorized_source_strategy(self, strategy):
        self.events.append(("strategy", strategy))

    def _set_m1_polynomial_policy(self, policy):
        self.events.append(("m1_polynomial", policy))
        self.m1_polynomial_policy = "recompute" if policy == "recompute" else "retained"

    def _set_mh0_state_policy(self, policy):
        if policy != "full-retention-v1":
            self.events.append(("mh0_state", policy))
        if policy == "reuse-adjoints-v1" and not self.jit_host_plugin_ready:
            raise RuntimeError("MH-0 reuse configured before JIT load")
        self.mh0_state_policy = policy

    def _set_edge_geometry_policy(self, policy):
        if policy != "cartesian-f64-v1":
            self.events.append(("edge_geometry", policy))
        self.edge_geometry_policy = policy

    def _set_low_memory(self, enabled):
        self.events.append(("low_memory", enabled))
        if enabled and not self.jit_host_plugin_ready:
            raise RuntimeError("low-memory bundle configured before JIT load")
        self.low_memory = enabled
        self.m1_polynomial_policy = "recompute" if enabled else "retained"
        self.mh0_state_policy = "reuse-adjoints-v1" if enabled else "full-retention-v1"
        self.edge_geometry_policy = (
            self.low_memory_geometry_policy if enabled else "cartesian-f64-v1"
        )

    def _set_execution_mh1_scratch_budget_bytes(self, budget_bytes):
        self.events.append(("mh1_scratch_budget", budget_bytes))
        self.execution_mh1_scratch_budget_bytes = budget_bytes

    def _set_execution_mh1_node_arena_policy(self, policy):
        if policy != "throughput-v1":
            self.events.append(("mh1_node_arena", policy))
        self.execution_mh1_node_arena_policy = policy
        self.execution_mh1_node_arena_tile_rows = (
            256 if policy == "capacity-v1" else 1024
        )

    def _load_jit_host_plugin(self, path):
        self.events.append(("load", path))
        if self.load_error is not None:
            raise self.load_error
        self.jit_host_plugin_ready = True
        self.jit_host_plugin_artifact_id = "jit-r1-fixture"

    def _load_receiver_factorized_host_plugin(self, path):
        self.events.append(("receiver_load", path))
        if self.load_error is not None:
            raise self.load_error
        self.receiver_factorized_host_plugin_ready = True
        self.receiver_factorized_host_plugin_artifact_id = "receiver-r1-fixture"

    def _load_jit_cuda_plugin(self, path, persistent_blocks):
        self.events.append(("cuda_load", path, persistent_blocks))
        if self.load_error is not None:
            raise self.load_error
        self.jit_cuda_plugin_ready = True
        self.jit_cuda_plugin_artifact_id = "jit-r1-fixture"

    def _load_jit_hip_plugin(self, path):
        self.events.append(("hip_load", path))
        if self.load_error is not None:
            raise self.load_error
        self.jit_hip_plugin_ready = True
        self.jit_hip_plugin_artifact_id = "jit-r1-fixture"

    def _load_jit_mh1_host_plugin(self, path):
        self.events.append(("mh1_load", path))
        if self.load_error is not None:
            raise self.load_error
        self.jit_mh1_host_plugin_ready = True
        self.jit_mh1_host_plugin_artifact_id = "jit-mh1-v3-fixture"

    def _load_jit_mh1_host_plugin_v4(self, path):
        self.events.append(("mh1_v4_load", path))
        if self.load_error is not None:
            raise self.load_error
        self.jit_mh1_host_plugin_v4_ready = True
        self.jit_mh1_host_plugin_artifact_id = "jit-mh1-v4-fixture"

    def _load_jit_mh1_host_plugin_v5(self, path):
        self.events.append(("mh1_v5_load", path))
        if self.load_error is not None:
            raise self.load_error
        self.jit_mh1_host_plugin_v5_ready = True
        self.jit_mh1_host_plugin_artifact_id = "jit-mh1-v4-fixture"

    def _load_jit_mh1_cuda_plugin(self, path):
        self.events.append(("mh1_cuda_load", path))
        if self.load_error is not None:
            raise self.load_error
        self.jit_mh1_cuda_plugin_ready = True
        self.jit_mh1_cuda_plugin_artifact_id = "jit-mh1-v3-fixture"

    def _load_jit_mh1_cuda_plugin_v4(self, path, launch_plan_json=""):
        self.events.append(("mh1_cuda_v4_load", path, launch_plan_json))
        if self.load_error is not None:
            raise self.load_error
        self.jit_mh1_cuda_plugin_v4_ready = True
        self.jit_mh1_cuda_plugin_artifact_id = "jit-mh1-v4-fixture"


def test_m1_polynomial_policy_rejects_unknown_value(jit_harness):
    with pytest.raises(ValueError, match="m1_polynomial_policy must be one of"):
        Symmetrix(
            jit_harness["model_path"],
            use_kokkos=True,
            streamed_edges="materialized",
            m1_polynomial_policy="unknown",
        )


def test_edge_geometry_policy_rejects_unknown_value(jit_harness):
    with pytest.raises(ValueError, match="edge_geometry_policy must be one of"):
        Symmetrix(
            jit_harness["model_path"],
            use_kokkos=True,
            streamed_edges="factorized",
            edge_geometry_policy="unknown",
        )


def test_mh1_node_arena_policy_rejects_unknown_value(jit_harness):
    with pytest.raises(
        ValueError, match="execution_mh1_node_arena_policy must be one of"
    ):
        Symmetrix(
            jit_harness["model_path"],
            use_kokkos=True,
            streamed_edges="materialized",
            execution_mh1_node_arena_policy="unknown",
        )


def test_mh1_node_arena_policy_is_configured_before_jit_load(jit_harness):
    calculator = Symmetrix(
        jit_harness["model_path"],
        use_kokkos=True,
        streamed_edges="factorized",
        execution_mh1_node_arena_policy="capacity-v1",
    )

    arena_index = jit_harness["events"].index(("mh1_node_arena", "capacity-v1"))
    load_index = next(
        index for index, event in enumerate(jit_harness["events"]) if event[0] == "load"
    )
    assert arena_index < load_index
    assert calculator.execution_mh1_node_arena_policy == "capacity-v1"
    assert jit_harness["evaluator"].execution_mh1_node_arena_tile_rows == 256


def test_compact_edge_geometry_is_configured_after_jit_load(jit_harness):
    calculator = Symmetrix(
        jit_harness["model_path"],
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
        edge_geometry_policy="unit-f32-radius-f64-v1",
    )

    assert calculator.edge_geometry_policy == "unit-f32-radius-f64-v1"
    load_index = next(
        index for index, event in enumerate(jit_harness["events"]) if event[0] == "load"
    )
    geometry_index = jit_harness["events"].index(
        ("edge_geometry", "unit-f32-radius-f64-v1")
    )
    assert load_index < geometry_index


@pytest.mark.parametrize("policy", ("automatic", "retained"))
def test_m1_polynomial_policy_falls_back_for_legacy_evaluator(
    jit_harness, monkeypatch, policy
):
    monkeypatch.delattr(_DummyEvaluator, "_set_m1_polynomial_policy")
    calculator = Symmetrix(
        jit_harness["model_path"],
        use_kokkos=True,
        streamed_edges="materialized",
        m1_polynomial_policy=policy,
    )

    assert calculator.m1_polynomial_policy_request == policy
    assert calculator.m1_polynomial_policy == "retained"
    assert (
        calculator.m1_polynomial_policy_reason
        == "the active evaluator does not expose M1 recomputation"
    )


def test_m1_polynomial_policy_explicit_recompute_fails_without_support(
    jit_harness, monkeypatch
):
    monkeypatch.delattr(_DummyEvaluator, "_set_m1_polynomial_policy")
    with pytest.raises(
        ValueError,
        match="m1_polynomial_policy='recompute' requires a compatible",
    ):
        Symmetrix(
            jit_harness["model_path"],
            use_kokkos=True,
            streamed_edges="materialized",
            m1_polynomial_policy="recompute",
        )


@pytest.mark.parametrize(
    ("dtype", "geometry_policy"),
    (
        ("float32", "unit-f32-radius-f64-v1"),
        ("float64", "cartesian-f64-v1"),
    ),
)
def test_low_memory_bundle_is_applied_after_jit_load(
    jit_harness, monkeypatch, dtype, geometry_policy
):
    monkeypatch.setattr(
        Symmetrix,
        "_configure_low_memory_operator_modules",
        lambda _self, **_kwargs: None,
    )
    jit_harness["evaluator"].low_memory_geometry_policy = geometry_policy
    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype=dtype,
        use_kokkos=True,
        streamed_edges="factorized",
        low_memory=True,
        m1_polynomial_policy="retained",
        execution_mh0_state_policy="full-retention-v1",
        edge_geometry_policy="cartesian-f64-v1",
    )

    assert calculator.low_memory_request is True
    assert calculator.low_memory is True
    assert calculator.m1_polynomial_policy_request == "retained"
    assert calculator.m1_polynomial_policy == "recompute"
    assert calculator.execution_mh0_state_policy == "reuse-adjoints-v1"
    assert calculator.mh0_state_policy == "reuse-adjoints-v1"
    assert calculator.edge_geometry_policy == geometry_policy
    load_index = next(
        index for index, event in enumerate(jit_harness["events"]) if event[0] == "load"
    )
    bundle_index = jit_harness["events"].index(("low_memory", True))
    assert load_index < bundle_index


def test_direct_defaults_to_capacity_profile(jit_harness, monkeypatch):
    # The fixture models only the R1 contract. The dedicated operator-artifact
    # path is covered elsewhere; suppress it here to exercise public dispatch.
    monkeypatch.setattr(
        Symmetrix, "_configure_low_memory_operator_modules", lambda self, **kwargs: None
    )
    calculator = Symmetrix(jit_harness["model_path"])

    assert calculator.streamed_edges == "direct"
    assert calculator.execution_profile == "capacity"
    assert calculator._kernel_launch_dtype == "float32"
    assert ("mode", "direct") in jit_harness["events"]
    assert ("low_memory", True) in jit_harness["events"]


def test_speed_profile_preserves_retained_compatibility_path(jit_harness):
    calculator = Symmetrix(jit_harness["model_path"], execution_profile="speed")

    assert calculator.streamed_edges == "direct"
    assert calculator.execution_profile == "speed"
    assert ("low_memory", True) not in jit_harness["events"]


def test_mh0_state_policy_is_configured_after_jit_load(jit_harness):
    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
        m1_polynomial_policy="recompute",
        execution_mh0_state_policy="reuse-adjoints-v1",
    )

    assert calculator.mh0_state_policy == "reuse-adjoints-v1"
    load_index = next(
        index for index, event in enumerate(jit_harness["events"]) if event[0] == "load"
    )
    state_index = jit_harness["events"].index(("mh0_state", "reuse-adjoints-v1"))
    assert load_index < state_index


@pytest.mark.parametrize("low_memory", (None, 0, 1, "true"))
def test_low_memory_rejects_non_boolean_values(jit_harness, low_memory):
    with pytest.raises(ValueError, match="low_memory must be a bool"):
        Symmetrix(jit_harness["model_path"], low_memory=low_memory)


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"dtype": "float32", "use_kokkos": False}, "requires use_kokkos=True"),
        (
            {"dtype": "float32", "streamed_edges": "all_interactions"},
            "requires streamed_edges='direct'",
        ),
        (
            {"dtype": "float32", "streamed_edges": "auto"},
            "requires streamed_edges='direct'",
        ),
    ),
)
def test_low_memory_rejects_unsupported_public_configuration(
    jit_harness, options, message
):
    with pytest.raises(ValueError, match=message):
        Symmetrix(jit_harness["model_path"], low_memory=True, **options)


def test_low_memory_requires_native_bundle_support(jit_harness, monkeypatch):
    monkeypatch.delattr(_DummyEvaluator, "_set_low_memory")
    monkeypatch.setattr(
        Symmetrix,
        "_configure_low_memory_operator_modules",
        lambda self, **kwargs: None,
    )
    with pytest.raises(ValueError, match="compatible Kokkos direct MH-0"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
            low_memory=True,
        )


def test_low_memory_selects_mh1_hybrid_before_cuda_jit_configuration(
    jit_harness, monkeypatch, tmp_path
):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    model_path = tmp_path / "mh1.json"
    contract = {"fixture": "mh1-contract"}
    model_path.write_text(
        json.dumps(
            {
                "model_type": "MACE_Nonlinear",
                "execution_contracts": {"MH1_UVU": contract},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACENonlinearKokkosFloat",
        lambda _path, _head="": jit_harness["evaluator"],
        raising=False,
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(jit, "jit_nvrtc_information", lambda: {"available": True})

    calculator = Symmetrix(
        model_path,
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
        low_memory=True,
    )

    assert calculator.low_memory is True
    assert calculator.low_memory_policy == "mh1-retain-interaction-v1"
    assert calculator.execution_mh1_node_state_policy == "retain-interaction-v1"
    assert calculator.execution_mh1_node_arena_policy == "capacity-v1"
    assert calculator.jit_node_state_policy == "retain-interaction-v1"
    assert jit_harness["evaluator"].execution_mh1_node_arena_tile_rows == 256
    _, _, policies = jit_harness["mh1_cuda_module_render_contracts"][0]
    assert policies["node_state_policy"] == "retain-interaction-v1"


@pytest.fixture
def jit_harness(monkeypatch, tmp_path):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    jit_codegen = importlib.import_module(f"{_PACKAGE_NAME}.jit_codegen")
    receiver_codegen = pytest.importorskip(
        f"{_PACKAGE_NAME}.receiver_factorized_rtc",
        reason=(
            "receiver-factorized RTC execution has not been merged into this branch yet"
        ),
    )
    mh1_jit_codegen = importlib.import_module(f"{_PACKAGE_NAME}.mh1_jit_codegen")
    execution_mh1_contract = importlib.import_module(
        f"{_PACKAGE_NAME}.execution_mh1_contract"
    )

    events = []
    evaluator = _DummyEvaluator(events)
    artifact = tmp_path / "factorized_host_plugin.so"
    artifact.write_bytes(b"plugin")
    state = {
        "events": events,
        "evaluator": evaluator,
        "artifact": artifact,
        "prepare_result": SimpleNamespace(
            status="built",
            cache_key="cache-key",
            artifact_path=artifact,
            diagnostics=("compiler diagnostic",),
            reason=None,
            available=True,
        ),
        "prepare_calls": [],
        "cuda_prepare_calls": [],
        "nvrtc_prepare_calls": [],
        "hip_prepare_calls": [],
        "hiprtc_prepare_calls": [],
        "quarantine_calls": [],
        "quarantine_cleanup_calls": [],
        "render_contracts": [],
        "cuda_render_contracts": [],
        "hip_render_contracts": [],
        "hip_module_render_contracts": [],
        "mh1_render_contracts": [],
        "mh1_cuda_render_contracts": [],
        "mh1_cuda_module_render_contracts": [],
        "mh1_cuda_launch_plan_contracts": [],
        "mh1_cuda_policy_requests": [],
        "mh1_cuda_source_policy_requests": [],
        "mh1_cuda_edge_policy_requests": [],
        "mh1_cuda_metadata_policies": [],
        "construct_paths": [],
        "jit_generation_version": jit.JIT_GENERATION_VERSION,
    }

    def model_metadata(path):
        if str(path).lower().endswith(".json"):
            return "MACE", False
        raise RuntimeError("not native JSON")

    def evaluator_factory(path, head=""):
        assert head == ""
        state["construct_paths"].append(path)
        return evaluator

    def render(contract, **_kwargs):
        state["events"].append(("render", contract))
        state["render_contracts"].append(contract)
        return "generated source"

    def prepare(source, **kwargs):
        state["events"].append(("prepare", source))
        state["prepare_calls"].append((source, kwargs))
        return state["prepare_result"]

    def render_cuda(contract, compute_capability, **_kwargs):
        state["events"].append(("cuda_render", contract, compute_capability))
        state["cuda_render_contracts"].append((contract, compute_capability))
        return "generated CUDA source"

    def render_cuda_module(contract, compute_capability, **_kwargs):
        state["events"].append(("cuda_module_render", contract, compute_capability))
        return "generated CUDA module source"

    def prepare_cuda(source, **kwargs):
        state["events"].append(("cuda_prepare", source))
        state["cuda_prepare_calls"].append((source, kwargs))
        return state["prepare_result"]

    def prepare_nvrtc(source, **kwargs):
        state["events"].append(("nvrtc_prepare", source))
        state["nvrtc_prepare_calls"].append((source, kwargs))
        return state["prepare_result"]

    def render_hip(contract, target, **_kwargs):
        state["events"].append(("hip_render", contract, target))
        state["hip_render_contracts"].append((contract, target))
        return "generated HIP plugin source"

    def render_hip_module(contract, target, **_kwargs):
        state["events"].append(("hip_module_render", contract, target))
        state["hip_module_render_contracts"].append((contract, target))
        return "generated HIP module source"

    def prepare_hip(source, **kwargs):
        state["events"].append(("hip_prepare", source))
        state["hip_prepare_calls"].append((source, kwargs))
        return state["prepare_result"]

    def prepare_hiprtc(source, **kwargs):
        state["events"].append(("hiprtc_prepare", source))
        state["hiprtc_prepare_calls"].append((source, kwargs))
        return state["prepare_result"]

    def render_mh1(contract, *, precision="float32", **_kwargs):
        state["events"].append(("mh1_render", contract))
        state["mh1_render_contracts"].append(contract)
        return f"generated MH1 {precision} source"

    def resolve_mh1_cuda_forward_policy(contract, compute_capability, request):
        state["mh1_cuda_policy_requests"].append(
            (contract, compute_capability, request)
        )
        return _MH1_CUDA_FORWARD_POLICY

    def resolve_mh1_cuda_source_policy(contract, compute_capability, request):
        state["mh1_cuda_source_policy_requests"].append(
            (contract, compute_capability, request)
        )
        return _MH1_CUDA_SOURCE_POLICY

    def resolve_mh1_cuda_edge_policy(contract, compute_capability, request):
        state["mh1_cuda_edge_policy_requests"].append(
            (contract, compute_capability, request)
        )
        return _MH1_CUDA_EDGE_POLICY

    def mh1_cuda_metadata(
        contract,
        compute_capability,
        *,
        forward_policy,
        source_policy,
        edge_policy,
    ):
        state["mh1_cuda_metadata_policies"].append(
            (
                contract,
                compute_capability,
                forward_policy,
                source_policy,
                edge_policy,
            )
        )
        return dict(_MH1_CUDA_METADATA_WITH_POLICY)

    def mh1_cuda_v4_metadata(
        contract,
        compute_capability,
        *,
        forward_policy,
        source_policy,
        edge_policy,
        node_state_policy,
    ):
        state["mh1_cuda_metadata_policies"].append(
            (
                contract,
                compute_capability,
                forward_policy,
                source_policy,
                edge_policy,
            )
        )
        return {
            **_MH1_CUDA_V4_METADATA_WITH_POLICY,
            "node_state_policy": node_state_policy,
        }

    def render_mh1_cuda(
        contract,
        compute_capability,
        *,
        forward_policy,
        source_policy,
        edge_policy,
        node_state_policy,
    ):
        state["events"].append(
            (
                "mh1_cuda_render",
                contract,
                compute_capability,
                forward_policy,
                source_policy,
                edge_policy,
            )
        )
        state["mh1_cuda_render_contracts"].append(
            (
                contract,
                compute_capability,
                forward_policy,
                source_policy,
                edge_policy,
            )
        )
        return "generated MH1 CUDA source"

    def render_mh1_cuda_module(contract, compute_capability, **policies):
        state["mh1_cuda_module_render_contracts"].append(
            (contract, compute_capability, policies)
        )
        return "generated MH1 CUDA module source"

    def mh1_cuda_launch_plan(contract, compute_capability, **policies):
        state["mh1_cuda_launch_plan_contracts"].append(
            (contract, compute_capability, policies)
        )
        return {
            "tag": "symmetrix.jit.mh1.cuda-module-launch-plan/1",
            "forward": [],
            "reverse": [],
        }

    def quarantine(result, reason):
        state["quarantine_calls"].append((result, reason))
        return SimpleNamespace(
            path=tmp_path / ".invalid" / "fixture",
            diagnostics=("quarantined load-broken fixture",),
        )

    def remove_quarantine(quarantine):
        state["quarantine_cleanup_calls"].append(quarantine)
        return ("removed recovered fixture quarantine",)

    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        model_metadata,
        raising=False,
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_is_initialized",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "OpenMP",
        raising=False,
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_required_jit_generation_version",
        lambda: jit.JIT_GENERATION_VERSION,
        raising=False,
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix, "MACEKokkos", evaluator_factory, raising=False
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACEKokkosFloat",
        evaluator_factory,
        raising=False,
    )
    monkeypatch.setattr(
        jit_codegen,
        "jit_r1_host_plugin_metadata",
        lambda contract, precision="float32": {
            **_METADATA,
            "precision": precision,
            "scalar_kind": 1 if precision == "float32" else 2,
            "scalar_size": 4 if precision == "float32" else 8,
        },
    )
    monkeypatch.setattr(jit_codegen, "render_jit_r1_host_plugin", render)
    monkeypatch.setattr(
        receiver_codegen,
        "receiver_factorized_host_plugin_metadata",
        lambda contract: {
            "abi": "symmetrix.receiver-factorized.host-plugin/1",
            "abi_version": 1,
            "artifact_id": "receiver-r1-fixture",
        },
    )
    monkeypatch.setattr(
        receiver_codegen,
        "render_receiver_factorized_host_source",
        render,
    )
    monkeypatch.setattr(
        jit_codegen,
        "factorized_gpu_codegen_identity",
        lambda contract, backend, target, **kwargs: {
            "schema_version": 1,
            "program": {"schema_version": 1},
            "target": {"backend": backend},
            "schedule": {"edge_strategy": kwargs.get("edge_strategy", "serial")},
            "launch_plan": {"artifact_kind": kwargs["artifact_kind"]},
        },
    )
    monkeypatch.setattr(
        jit_codegen,
        "jit_r1_cuda_plugin_metadata",
        lambda contract, compute_capability, precision="float32", **kwargs: {
            **_CUDA_METADATA,
            "precision": precision,
            "scalar_kind": 1 if precision == "float32" else 2,
            "scalar_size": 4 if precision == "float32" else 8,
            "edge_strategy": kwargs.get("edge_strategy", "serial"),
            "edge_logical_subgroup_width": kwargs.get("edge_logical_subgroup_width", 1),
            "edge_threads_per_block": kwargs.get("edge_threads_per_block", 256),
            "persistent_blocks_per_compute_unit": kwargs.get(
                "persistent_blocks_per_compute_unit", 1
            ),
        },
    )
    monkeypatch.setattr(jit_codegen, "render_jit_r1_cuda_plugin", render_cuda)
    monkeypatch.setattr(jit_codegen, "render_jit_r1_cuda_module", render_cuda_module)
    monkeypatch.setattr(
        jit_codegen,
        "factorized_hip_plugin_metadata",
        lambda contract, target, precision="float32", **_kwargs: {
            **_HIP_METADATA,
            "precision": precision,
            "scalar_kind": 1 if precision == "float32" else 2,
            "scalar_size": 4 if precision == "float32" else 8,
        },
    )
    monkeypatch.setattr(jit_codegen, "render_jit_r1_hip_plugin", render_hip)
    monkeypatch.setattr(jit_codegen, "render_jit_r1_hip_module", render_hip_module)
    monkeypatch.setattr(
        mh1_jit_codegen,
        "jit_mh1_host_plugin_metadata",
        lambda contract: dict(_MH1_METADATA),
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "render_jit_mh1_host_plugin",
        render_mh1,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "execution_mh1_node_program_metadata",
        lambda contract, backend, **_kwargs: dict(_MH1_V4_METADATA),
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "render_jit_mh1_host_plugin_v4",
        render_mh1,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "render_jit_mh1_host_plugin_v5",
        render_mh1,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "jit_mh1_cuda_plugin_metadata",
        mh1_cuda_metadata,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "jit_mh1_cuda_plugin_v4_metadata",
        mh1_cuda_v4_metadata,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "resolve_execution_mh1_cuda_forward_policy",
        resolve_mh1_cuda_forward_policy,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "resolve_execution_mh1_cuda_source_policy",
        resolve_mh1_cuda_source_policy,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "resolve_execution_mh1_cuda_edge_policy",
        resolve_mh1_cuda_edge_policy,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "render_jit_mh1_cuda_plugin",
        render_mh1_cuda,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "render_jit_mh1_cuda_plugin_v4",
        render_mh1_cuda,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "render_execution_mh1_cuda_module_v4",
        render_mh1_cuda_module,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "execution_mh1_cuda_module_v4_launch_plan",
        mh1_cuda_launch_plan,
    )
    monkeypatch.setattr(
        execution_mh1_contract,
        "validate_execution_mh1_contract_for_model",
        lambda _model, contract: contract,
    )
    monkeypatch.setattr(
        execution_mh1_contract,
        "validate_execution_mh1_v4_contract_for_model",
        lambda _model, contract: {
            **contract,
            "tag": "symmetrix.execution.mh1_uvu/4",
            "generation_fingerprint": "sha256:generation-v4",
            "semantic_fingerprint": "sha256:semantic-v4",
            "structure_fingerprint": "sha256:structure-v4",
            "runtime_layout_fingerprint": "sha256:runtime-v4",
        },
    )
    monkeypatch.setattr(
        execution_mh1_contract,
        "project_execution_mh1_v3_contract",
        lambda contract: {"fixture": contract["fixture"]},
    )
    monkeypatch.setattr(jit, "prepare_jit_artifact", prepare)
    monkeypatch.setattr(jit, "prepare_execution_cuda_jit_artifact", prepare_cuda)
    monkeypatch.setattr(jit, "prepare_nvrtc_jit_artifact", prepare_nvrtc)
    monkeypatch.setattr(jit, "prepare_execution_hip_jit_artifact", prepare_hip)
    monkeypatch.setattr(jit, "prepare_hiprtc_jit_artifact", prepare_hiprtc)
    monkeypatch.setattr(jit, "quarantine_jit_artifact", quarantine)
    monkeypatch.setattr(jit, "remove_jit_quarantine", remove_quarantine)

    model = {
        "model_type": "MACE",
        "execution_contracts": {"R1": {"fixture": "contract"}},
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    state["model"] = model
    state["model_path"] = model_path
    return state


@pytest.mark.parametrize("cache_status", ["built", "cached"])
def test_jit_miss_builds_or_reuses_and_loads_plugin(jit_harness, cache_status):
    jit_harness["prepare_result"].status = cache_status

    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_status == cache_status
    assert calculator.jit_cache_key == "cache-key"
    assert calculator.jit_artifact_path == str(jit_harness["artifact"])
    assert calculator.jit_artifact_id == "jit-r1-fixture"
    assert calculator.jit_diagnostics == ("compiler diagnostic",)
    assert calculator.jit_reason is None
    assert jit_harness["render_contracts"] == [
        jit_harness["model"]["execution_contracts"]["R1"]
    ]
    source, options = jit_harness["prepare_calls"][0]
    assert source == "generated source"
    assert options["abi"] == {
        "tag": _METADATA["abi"],
        "version": _METADATA["abi_version"],
    }
    assert options["build"]["contract"] == _METADATA
    assert options["build"]["generator"] == "symmetrix.jit.r1-host-v2"
    assert options["build"]["precision"] == "float32"
    assert (
        options["build"]["jit_generation_version"]
        == jit_harness["jit_generation_version"]
    )
    assert options["cxx_flags"] is None
    assert options["artifact_name"] == "factorized_host_plugin"
    assert jit_harness["events"][-1] == (
        "load",
        str(jit_harness["artifact"]),
    )


def test_jit_float64_build_identity_and_loader(jit_harness):
    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float64",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_status == "built"
    source, options = jit_harness["prepare_calls"][0]
    assert source == "generated source"
    assert options["build"]["precision"] == "float64"
    assert options["build"]["contract"]["precision"] == "float64"
    assert options["build"]["contract"]["scalar_kind"] == 2
    assert options["build"]["contract"]["scalar_size"] == 8


def test_low_memory_r1_host_jit_prefers_generated_m0_plugin(jit_harness, monkeypatch):
    calls = []

    def configure_low_memory_operator_modules(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        Symmetrix,
        "_configure_low_memory_operator_modules",
        configure_low_memory_operator_modules,
    )

    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
        low_memory=True,
    )

    assert calculator.low_memory is True
    assert len(calls) == 1
    assert (
        calls[0]["model_data"]["execution_contracts"]
        == jit_harness["model"]["execution_contracts"]
    )
    assert calls[0]["dtype"] == "float32"
    assert calls[0]["backend"] == "host"
    assert calls[0]["target"] == (
        jit_harness["evaluator"].execution_device_execution_environment
    )
    assert calls[0]["jit_generation_version"] == (
        symmetrix_calc._require_matching_jit_generation_version()
    )
    assert calls[0]["prefer_host_m0_plugin"] is True


def test_macefield_uses_prepared_execution_graph_and_field_entry_point():
    import numpy as np

    mace_inputs = (
        2,
        [0, 0],
        np.array([1, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        [0, 0],
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.array([1.0, 1.0]),
        np.array([0, 1], dtype=np.int32),
    )

    class PreparedFieldEvaluator:
        streamed_edges_mode = "direct"

        def __init__(self):
            self.prepared = []
            self.evaluated = []

        def _prepare_factorized_graph(self, *topology):
            self.prepared.append(topology)
            return 23

        def _compute_prepared_factorized_field(
            self, token, xyz, distances, electric_field
        ):
            self.evaluated.append(
                (token, xyz.copy(), distances.copy(), electric_field.copy())
            )

        def compute_node_energies_forces_field(self, *args):
            raise AssertionError("prepared MACEField must not rebuild topology")

    calculator = object.__new__(Symmetrix)
    calculator.evaluator = PreparedFieldEvaluator()
    field = np.array([0.01, -0.02, 0.03])
    returned = calculator._compute_macefield(None, field, mace_inputs)

    assert returned is mace_inputs
    assert len(calculator.evaluator.prepared) == 1
    token, xyz, distances, actual_field = calculator.evaluator.evaluated[0]
    assert token == 23
    assert xyz.flags.c_contiguous and xyz.shape == (6,)
    assert distances.flags.c_contiguous and distances.shape == (2,)
    assert actual_field.flags.c_contiguous
    assert np.array_equal(actual_field, field)

    with pytest.raises(ValueError, match="invalid distance, displacement, or field"):
        calculator._compute_macefield(None, np.array([np.nan, 0.0, 0.0]), mace_inputs)


def test_jit_default_failure_raises(jit_harness):
    jit_harness["prepare_result"] = SimpleNamespace(
        status="fallback",
        cache_key="cache-key",
        artifact_path=None,
        diagnostics=("compiler missing",),
        reason="CXX compiler executable was not found",
        available=False,
    )

    with pytest.raises(
        symmetrix_calc._JitRequiredError,
        match="(?s)compiler missing.*does not fall back",
    ):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )


def test_literal_direct_failure_never_switches_to_generic(jit_harness):
    jit_harness["prepare_result"] = SimpleNamespace(
        status="fallback",
        cache_key="cache-key",
        artifact_path=None,
        diagnostics=("compiler missing",),
        reason="CXX compiler executable was not found",
        available=False,
    )

    with pytest.raises(
        symmetrix_calc._JitRequiredError,
        match="(?s)compiler missing.*does not fall back.*streamed_edges='non-compiled'",
    ):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="direct",
        )

    assert jit_harness["evaluator"].streamed_edges_mode == "direct"


def test_deprecated_jit_fallback_is_ignored(jit_harness):
    jit_harness["prepare_result"] = SimpleNamespace(
        status="fallback",
        cache_key="cache-key",
        artifact_path=None,
        diagnostics=("compiler missing",),
        reason="CXX compiler executable was not found",
        available=False,
    )

    with (
        pytest.warns(FutureWarning, match="jit parameter is deprecated and ignored"),
        pytest.raises(symmetrix_calc._JitRequiredError, match="compiler missing"),
    ):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
            jit="fallback",
        )

    assert not jit_harness["evaluator"].jit_host_plugin_ready


def test_jit_required_failure_raises(jit_harness):
    jit_harness["prepare_result"] = SimpleNamespace(
        status="fallback",
        cache_key=None,
        artifact_path=None,
        diagnostics=("compiler command: c++ -shared plugin.cpp", "fatal error"),
        reason="compiler unavailable",
        available=False,
    )

    with pytest.raises(RuntimeError, match="(?s)compiler command:.*fatal error"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )


def test_jit_loader_failure_is_always_fatal(jit_harness):
    jit_harness["evaluator"].load_error = RuntimeError("descriptor mismatch")

    with (
        pytest.warns(FutureWarning, match="jit parameter is deprecated and ignored"),
        pytest.raises(symmetrix_calc._JitRequiredError, match="descriptor mismatch"),
    ):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
            jit="fallback",
        )

    assert len(jit_harness["prepare_calls"]) == 2
    assert len(jit_harness["quarantine_calls"]) == 1
    assert jit_harness["quarantine_cleanup_calls"] == []
    assert "descriptor mismatch" in jit_harness["quarantine_calls"][0][1]


def test_jit_successful_retry_removes_load_quarantine(jit_harness, monkeypatch):
    attempts = 0

    def load(path):
        nonlocal attempts
        attempts += 1
        jit_harness["events"].append(("load", path))
        if attempts == 1:
            raise RuntimeError("transient descriptor read failure")
        jit_harness["evaluator"].jit_host_plugin_ready = True
        jit_harness["evaluator"].jit_host_plugin_artifact_id = "jit-r1-fixture"

    monkeypatch.setattr(jit_harness["evaluator"], "_load_jit_host_plugin", load)
    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_status == "built"
    assert attempts == 2
    assert len(jit_harness["quarantine_cleanup_calls"]) == 1
    assert calculator.jit_diagnostics[-1] == ("removed recovered fixture quarantine")


def test_jit_required_reports_real_compiler_failure(jit_harness, tmp_path):
    compiler = next(
        (
            shutil.which(name)
            for name in ("c++", "g++", "clang++")
            if shutil.which(name)
        ),
        None,
    )
    if compiler is None:
        pytest.skip("a C++ compiler is required for diagnostic coverage")
    jit = _load_unpatched_jit()
    failed = jit.prepare_jit_artifact(
        "this is not valid C++\n",
        abi="test-abi",
        build="required-diagnostic-test",
        cache_root=tmp_path / "compile-failure-cache",
        cxx=compiler,
        cpu="test-cpu",
    )
    assert not failed.available
    assert any("compiler stderr" in item for item in failed.diagnostics)
    jit_harness["prepare_result"] = failed

    with pytest.raises(RuntimeError, match="compiler stderr"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"streamed_edges": "materialized"}, "streamed_edges"),
    ],
)
def test_jit_is_not_applicable_to_non_factorized_modes(jit_harness, kwargs, reason):
    options = {
        "dtype": "float32",
        "use_kokkos": True,
        "streamed_edges": "factorized",
    }
    options.update(kwargs)

    calculator = Symmetrix(jit_harness["model_path"], **options)

    assert calculator.jit_status == "not_applicable"
    assert reason in calculator.jit_reason
    assert jit_harness["prepare_calls"] == []


@pytest.mark.parametrize(
    ("dtype", "evaluator_name"),
    (("float32", "MACENonlinearKokkosFloat"), ("float64", "MACENonlinearKokkos")),
)
def test_mh1_host_jit_builds_and_loads_generated_plugin(
    jit_harness, monkeypatch, tmp_path, dtype, evaluator_name
):
    contract = {"fixture": "mh1-contract"}
    model_path = tmp_path / "mh1.json"
    model_path.write_text(
        json.dumps(
            {
                "model_type": "MACE_Nonlinear",
                "execution_contracts": {"MH1_UVU": contract},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        evaluator_name,
        lambda path, head="": jit_harness["evaluator"],
        raising=False,
    )

    calculator = Symmetrix(
        model_path,
        dtype=dtype,
        use_kokkos=True,
        streamed_edges="factorized",
    )
    assert calculator.streamed_edges == "direct"
    assert calculator.jit_status == "built"
    assert calculator.jit_artifact_id == "jit-mh1-v4-fixture"
    assert calculator.jit_reason is None
    assert len(jit_harness["mh1_render_contracts"]) == 1
    rendered_contract = jit_harness["mh1_render_contracts"][0]
    assert rendered_contract["fixture"] == contract["fixture"]
    assert rendered_contract["tag"] == "symmetrix.execution.mh1_uvu/4"
    source, options = jit_harness["prepare_calls"][0]
    assert source == f"generated MH1 {dtype} source"
    assert options["abi"] == {
        "tag": _MH1_V5_METADATA["abi"],
        "version": _MH1_V5_METADATA["abi_version"],
    }
    assert options["build"] == {
        "generator": "symmetrix.jit.mh1-host-v5",
        "precision": dtype,
        "contract": _MH1_V5_METADATA,
        "jit_generation_version": jit_harness["jit_generation_version"],
    }
    assert options["artifact_name"] == "jit_mh1_host_plugin"
    assert jit_harness["events"][-1] == (
        "mh1_v5_load",
        str(jit_harness["artifact"]),
    )


def test_mh1_cuda_jit_uses_subgroup_generator_provenance(
    jit_harness, monkeypatch, tmp_path
):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    mh1_jit_codegen = importlib.import_module(f"{_PACKAGE_NAME}.mh1_jit_codegen")
    contract = {"fixture": "mh1-cuda-contract"}
    model_path = tmp_path / "mh1-cuda.json"
    model_path.write_text(
        json.dumps(
            {
                "model_type": "MACE_Nonlinear",
                "execution_contracts": {"MH1_UVU": contract},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACENonlinearKokkosFloat",
        lambda path, head="": jit_harness["evaluator"],
        raising=False,
    )
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True},
    )

    calculator = Symmetrix(
        model_path,
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_status == "built"
    assert calculator.jit_artifact_id == "jit-mh1-v4-fixture"
    assert calculator.jit_forward_policy == _MH1_CUDA_FORWARD_POLICY
    assert calculator.jit_source_policy == _MH1_CUDA_SOURCE_POLICY
    assert calculator.jit_edge_policy == _MH1_CUDA_EDGE_POLICY
    assert calculator.jit_variant_id == (
        "jit-mh1-v4-fixture-fwd-aaaaaaaaaaaa-src-bbbbbbbbbbbb-edge-cccccccccccc"
        "-node-full-retention-v1"
    )
    assert calculator.jit_node_state_policy == "full-retention-v1"
    assert jit_harness["mh1_cuda_policy_requests"] == [(contract, 120, "auto")]
    assert jit_harness["mh1_cuda_source_policy_requests"] == [(contract, 120, "auto")]
    assert jit_harness["mh1_cuda_edge_policy_requests"] == [(contract, 120, "auto")]
    assert jit_harness["mh1_cuda_metadata_policies"] == [
        (
            {
                **contract,
                "tag": "symmetrix.execution.mh1_uvu/4",
                "generation_fingerprint": "sha256:generation-v4",
                "semantic_fingerprint": "sha256:semantic-v4",
                "structure_fingerprint": "sha256:structure-v4",
                "runtime_layout_fingerprint": "sha256:runtime-v4",
            },
            120,
            _MH1_CUDA_FORWARD_POLICY,
            _MH1_CUDA_SOURCE_POLICY,
            _MH1_CUDA_EDGE_POLICY,
        )
    ]
    assert jit_harness["mh1_cuda_render_contracts"] == [
        (
            {
                **contract,
                "tag": "symmetrix.execution.mh1_uvu/4",
                "generation_fingerprint": "sha256:generation-v4",
                "semantic_fingerprint": "sha256:semantic-v4",
                "structure_fingerprint": "sha256:structure-v4",
                "runtime_layout_fingerprint": "sha256:runtime-v4",
            },
            120,
            _MH1_CUDA_FORWARD_POLICY,
            _MH1_CUDA_SOURCE_POLICY,
            _MH1_CUDA_EDGE_POLICY,
        )
    ]
    source, options = jit_harness["nvrtc_prepare_calls"][0]
    assert source == "generated MH1 CUDA module source"
    assert options["abi"] == {
        "tag": mh1_jit_codegen.MH1_CUDA_MODULE_ABI,
        "version": mh1_jit_codegen.MH1_CUDA_MODULE_ABI_VERSION,
    }
    assert options["build"]["generator"] == "symmetrix.jit.mh1-cuda-module-v1"
    assert options["build"]["contract"] == _MH1_CUDA_V4_METADATA_WITH_POLICY
    assert options["build"]["launch_plan"]["tag"] == (
        "symmetrix.jit.mh1.cuda-module-launch-plan/1"
    )
    assert options["artifact_name"] == "execution_mh1_cuda_module"
    assert jit_harness["cuda_prepare_calls"] == []
    event = jit_harness["events"][-1]
    assert event[:2] == ("mh1_cuda_v4_load", str(jit_harness["artifact"]))
    assert json.loads(event[2]) == options["build"]["launch_plan"]


def test_mh1_cuda_explicit_nvrtc_uses_module_and_launch_plan(
    jit_harness, monkeypatch, tmp_path
):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    mh1_jit_codegen = importlib.import_module(f"{_PACKAGE_NAME}.mh1_jit_codegen")
    contract = {"fixture": "mh1-cuda-nvrtc-contract"}
    model_path = tmp_path / "mh1-cuda-nvrtc.json"
    model_path.write_text(
        json.dumps(
            {
                "model_type": "MACE_Nonlinear",
                "execution_contracts": {"MH1_UVU": contract},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACENonlinearKokkosFloat",
        lambda _path, _head="": jit_harness["evaluator"],
        raising=False,
    )
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True},
    )
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvrtc")

    calculator = Symmetrix(
        model_path,
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_compiler_request == "nvrtc"
    assert calculator.jit_compiler_backend == "nvrtc"
    source, options = jit_harness["nvrtc_prepare_calls"][0]
    assert source == "generated MH1 CUDA module source"
    assert options["abi"] == {
        "tag": mh1_jit_codegen.MH1_CUDA_MODULE_ABI,
        "version": mh1_jit_codegen.MH1_CUDA_MODULE_ABI_VERSION,
    }
    assert options["build"]["generator"] == ("symmetrix.jit.mh1-cuda-module-v1")
    assert options["build"]["launch_plan"]["tag"] == (
        "symmetrix.jit.mh1.cuda-module-launch-plan/1"
    )
    assert options["artifact_name"] == "execution_mh1_cuda_module"
    event = jit_harness["events"][-1]
    assert event[:2] == ("mh1_cuda_v4_load", str(jit_harness["artifact"]))
    assert json.loads(event[2]) == options["build"]["launch_plan"]


def test_mh1_cuda_automatic_uses_nvrtc_module_and_launch_plan(
    jit_harness, monkeypatch, tmp_path
):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    contract = {"fixture": "mh1-cuda-load-fallback-contract"}
    model_path = tmp_path / "mh1-cuda-load-fallback.json"
    model_path.write_text(
        json.dumps(
            {
                "model_type": "MACE_Nonlinear",
                "execution_contracts": {"MH1_UVU": contract},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACENonlinearKokkosFloat",
        lambda _path, _head="": jit_harness["evaluator"],
        raising=False,
    )
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True},
    )
    calculator = Symmetrix(
        model_path,
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_compiler_request == "automatic"
    assert calculator.jit_compiler_backend == "nvrtc"
    assert len(jit_harness["nvrtc_prepare_calls"]) == 1
    assert jit_harness["cuda_prepare_calls"] == []
    source, options = jit_harness["nvrtc_prepare_calls"][0]
    assert source == "generated MH1 CUDA module source"
    assert options["build"]["generator"] == "symmetrix.jit.mh1-cuda-module-v1"
    assert "launch_plan" in options["build"]


def test_mh1_cuda_compilation_failure_is_fatal(jit_harness, monkeypatch, tmp_path):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    contract = {"fixture": "mh1-cuda-contract"}
    model_path = tmp_path / "mh1-cuda-fallback.json"
    model_path.write_text(
        json.dumps(
            {
                "model_type": "MACE_Nonlinear",
                "execution_contracts": {"MH1_UVU": contract},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACENonlinearKokkosFloat",
        lambda path, head="": jit_harness["evaluator"],
        raising=False,
    )
    monkeypatch.setattr(jit, "jit_nvrtc_information", lambda: {"available": True})
    jit_harness["prepare_result"] = SimpleNamespace(
        status="fallback",
        cache_key="failed-cache-key",
        artifact_path=None,
        diagnostics=("synthetic compile failure",),
        reason="compiler failed",
        available=False,
    )

    with pytest.raises(symmetrix_calc._JitRequiredError, match="compiler failed"):
        Symmetrix(
            model_path,
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )


def test_mh1_required_rejects_model_without_generation_contract(
    jit_harness, monkeypatch, tmp_path
):
    model_path = tmp_path / "mh1-no-contract.json"
    model_path.write_text(
        json.dumps({"model_type": "MACE_Nonlinear"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACENonlinearKokkosFloat",
        lambda path, head="": jit_harness["evaluator"],
        raising=False,
    )

    with pytest.raises(RuntimeError, match="MH1_UVU contract"):
        Symmetrix(
            model_path,
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )


def test_mh1_required_rejects_contract_not_bound_to_model(
    jit_harness, monkeypatch, tmp_path
):
    contract = {"fixture": "stale-mh1-contract"}
    model_path = tmp_path / "mh1-stale-contract.json"
    model_path.write_text(
        json.dumps(
            {
                "model_type": "MACE_Nonlinear",
                "execution_contracts": {"MH1_UVU": contract},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACENonlinearKokkosFloat",
        lambda path, head="": jit_harness["evaluator"],
        raising=False,
    )
    execution_mh1_contract = importlib.import_module(
        f"{_PACKAGE_NAME}.execution_mh1_contract"
    )

    def reject_stale_contract(_model, _contract):
        raise ValueError(
            "the embedded Execution MH1_UVU contract does not match the model definition"
        )

    monkeypatch.setattr(
        execution_mh1_contract,
        "validate_execution_mh1_v4_contract_for_model",
        reject_stale_contract,
    )
    with pytest.raises(RuntimeError, match="does not match the model definition"):
        Symmetrix(
            model_path,
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )
    assert jit_harness["prepare_calls"] == []


@pytest.mark.parametrize(
    "legacy_value",
    ["auto", "fallback", "jit_only", "off", "required", "none", "sometimes", False],
)
def test_jit_parameter_is_deprecated_and_ignored(jit_harness, legacy_value):
    with pytest.warns(FutureWarning, match="jit parameter is deprecated and ignored"):
        calculator = Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
            jit=legacy_value,
        )

    assert calculator.jit_policy == "required"
    assert calculator.jit_status == "built"


@pytest.mark.parametrize(
    ("mode", "canonical"),
    (("factorized", "direct"), ("receiver_factorized", "receiver_factorized")),
)
def test_jit_none_policy_rejects_rtc_modes_without_fallback(
    jit_harness, monkeypatch, mode, canonical
):
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "none")
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_required_jit_generation_version",
        lambda: (_ for _ in ()).throw(AssertionError("version query must not run")),
    )

    with pytest.raises(
        symmetrix_calc._JitRequiredError,
        match=rf"SYMMETRIX_JIT_POLICY=none is incompatible.*{canonical}",
    ):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges=mode,
        )
    assert jit_harness["prepare_calls"] == []


def test_jit_environment_policy_rejects_unknown_value(jit_harness, monkeypatch):
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "fallback")

    with pytest.raises(ValueError, match="SYMMETRIX_JIT_POLICY must be"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )


def test_execution_mh1_scratch_budget_validation_and_routing(jit_harness):
    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
        execution_mh1_scratch_budget_bytes=12345,
    )
    assert calculator.evaluator.execution_mh1_scratch_budget_bytes == 12345
    assert ("mh1_scratch_budget", 12345) in jit_harness["events"]

    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            Symmetrix(
                jit_harness["model_path"],
                execution_mh1_scratch_budget_bytes=invalid,
            )


@pytest.mark.parametrize("mode", ["materialized", "all_interactions"])
def test_non_factorized_modes_do_not_dispatch_jit(jit_harness, monkeypatch, mode):
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "invalid-but-unused")
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_required_jit_generation_version",
        lambda: (_ for _ in ()).throw(AssertionError("version query must not run")),
    )
    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges=mode,
    )
    assert calculator.jit_status == "not_applicable"
    assert calculator.jit_reason == "streamed_edges does not require RTC"
    assert jit_harness["prepare_calls"] == []


@pytest.mark.parametrize(
    ("requested", "canonical", "alias", "uses_jit"),
    (
        ("materialized", "materialized", None, False),
        ("generic", "generic", "generic", False),
        ("all_interactions", "generic", "all_interactions", False),
        ("direct", "direct", None, True),
        ("factorized", "direct", "factorized", True),
        ("direct_streamed", "direct", "direct_streamed", True),
        ("receiver_factorized", "receiver_factorized", None, True),
    ),
)
def test_execution_mode_aliases_canonicalize_before_native_dispatch(
    jit_harness, requested, canonical, alias, uses_jit
):
    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges=requested,
        execution_profile="speed",
    )

    assert calculator.streamed_edges_requested == requested
    assert calculator.streamed_edges == canonical
    assert calculator.execution_algorithm == canonical
    assert calculator.streamed_edges_alias == alias
    assert ("mode", canonical) in jit_harness["events"]
    assert bool(jit_harness["prepare_calls"]) is uses_jit
    assert calculator.jit_status == ("built" if uses_jit else "not_applicable")


def test_jit_generation_version_mismatch_fails_before_generation(
    jit_harness, monkeypatch
):
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_required_jit_generation_version",
        lambda: 4,
    )

    with pytest.raises(RuntimeError, match="native standard modules require 4"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )

    assert jit_harness["render_contracts"] == []
    assert jit_harness["prepare_calls"] == []


def test_legacy_native_extension_fails_before_jit_generation(jit_harness, monkeypatch):
    monkeypatch.delattr(
        symmetrix_calc.symmetrix,
        "_required_jit_generation_version",
    )

    with pytest.raises(RuntimeError, match="rebuild it with the active source tree"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )

    assert jit_harness["render_contracts"] == []
    assert jit_harness["prepare_calls"] == []


@pytest.mark.parametrize("cache_status", ["built", "cached"])
def test_jit_cuda_miss_builds_and_loads_on_exact_target(
    jit_harness, monkeypatch, cache_status
):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True},
    )
    jit_harness["prepare_result"].status = cache_status

    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_status == cache_status
    assert calculator.jit_artifact_id == "jit-r1-fixture"
    assert calculator.jit_edge_policy == {
        "strategy": "wave",
        "logical_subgroup_width": 32,
        "threads_per_block": 32,
        "persistent_blocks_per_compute_unit": 8,
    }
    assert calculator.jit_variant_id == "jit-r1-fixture-edge-wave-w32-t32-b8"
    assert jit_harness["prepare_calls"] == []
    assert (
        "cuda_module_render",
        jit_harness["model"]["execution_contracts"]["R1"],
        120,
    ) in jit_harness["events"]
    source, options = jit_harness["nvrtc_prepare_calls"][0]
    assert source == "generated CUDA module source"
    assert options["abi"] == {
        "tag": _CUDA_METADATA["abi"],
        "version": _CUDA_METADATA["abi_version"],
    }
    assert options["build"]["generator"] == "symmetrix.jit.r1-cuda-module-v2"
    assert options["build"]["precision"] == "float32"
    assert options["build"]["contract"] == _CUDA_METADATA
    assert options["build"]["gpu_codegen"]["target"]["backend"] == "cuda"
    assert options["build"]["gpu_codegen"]["launch_plan"] == {"artifact_kind": "module"}
    assert options["compute_capability"] == (
        jit_harness["evaluator"].device_cuda_environment
    )
    assert options["artifact_name"] == "factorized_cuda_module"
    assert jit_harness["events"][-1] == (
        "cuda_load",
        str(jit_harness["artifact"]),
        _CUDA_METADATA["persistent_blocks_per_compute_unit"],
    )


def test_jit_cuda_serial_override_preserves_rollback(jit_harness, monkeypatch):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True},
    )
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_R1_EDGE_STRATEGY", "serial")
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_R1_EDGE_LOGICAL_WIDTH", "1")
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_R1_EDGE_THREADS_PER_BLOCK", "256")
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_R1_EDGE_BLOCKS_PER_SM", "1")

    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_edge_policy == {
        "strategy": "serial",
        "logical_subgroup_width": 1,
        "threads_per_block": 256,
        "persistent_blocks_per_compute_unit": 1,
    }
    assert calculator.jit_variant_id.endswith("-edge-serial-w1-t256-b1")
    assert jit_harness["events"][-1] == (
        "cuda_load",
        str(jit_harness["artifact"]),
        1,
    )


@pytest.mark.parametrize("policy", [None, "auto", "fallback", "required"])
def test_jit_cuda_compiler_failure_obeys_policy(jit_harness, monkeypatch, policy):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True},
    )
    jit_harness["prepare_result"] = SimpleNamespace(
        status="fallback",
        cache_key="cuda-cache-key",
        artifact_path=None,
        diagnostics=("NVRTC compilation failed",),
        reason="NVRTC compilation failed",
        available=False,
    )
    options = {
        "dtype": "float32",
        "use_kokkos": True,
        "streamed_edges": "factorized",
        "jit": policy,
    }

    warning = pytest.warns(FutureWarning) if policy is not None else nullcontext()
    with warning, pytest.raises(RuntimeError, match="NVRTC compilation failed"):
        Symmetrix(jit_harness["model_path"], **options)

    assert jit_harness["prepare_calls"] == []
    assert len(jit_harness["nvrtc_prepare_calls"]) == 1
    assert jit_harness["cuda_prepare_calls"] == []
    assert not jit_harness["evaluator"].jit_cuda_plugin_ready


def test_jit_cuda_automatic_compile_failure_never_invokes_nvcc(
    jit_harness, monkeypatch
):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True},
    )
    nvrtc_failure = SimpleNamespace(
        status="fallback",
        cache_key="nvrtc-cache-key",
        artifact_path=None,
        diagnostics=("NVRTC compile failed",),
        reason="invalid generated CUDA",
        available=False,
    )

    def prepare_nvrtc(source, **kwargs):
        jit_harness["nvrtc_prepare_calls"].append((source, kwargs))
        return nvrtc_failure

    monkeypatch.setattr(jit, "prepare_nvrtc_jit_artifact", prepare_nvrtc)

    with pytest.raises(RuntimeError, match="invalid generated CUDA"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )

    assert len(jit_harness["nvrtc_prepare_calls"]) == 1
    assert jit_harness["cuda_prepare_calls"] == []


def test_jit_cuda_automatic_load_failure_never_invokes_nvcc(jit_harness, monkeypatch):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True},
    )
    load_calls = []

    def load(path, *args):
        load_calls.append((path, args))
        raise RuntimeError("invalid cubin fixture")

    jit_harness["evaluator"]._load_jit_cuda_plugin = load

    with pytest.raises(RuntimeError, match="invalid cubin fixture"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )

    assert len(jit_harness["nvrtc_prepare_calls"]) == 2
    assert jit_harness["cuda_prepare_calls"] == []
    assert len(load_calls) == 2
    assert len(jit_harness["quarantine_calls"]) == 1


def _configure_hip_fixture(jit_harness, monkeypatch):
    jit = importlib.import_module(f"{_PACKAGE_NAME}.jit")
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "HIP",
    )
    jit_harness["evaluator"].execution_device_execution_environment = {
        "available": True,
        "backend": "hip",
        "execution_space": "HIP",
        "raw_agent_target": "gfx1151",
        "architecture": "gfx1151",
        "target_features": "",
        "native_subgroup_width": 32,
        "compute_unit_count": 40,
        "runtime_version": "7.14",
    }
    monkeypatch.setattr(
        jit,
        "jit_hiprtc_information",
        lambda: {"available": True, "library": "libhiprtc.so"},
    )
    return jit


def test_mh1_hip_explicit_hiprtc_uses_shared_v4_handle(
    jit_harness, monkeypatch, tmp_path
):
    _configure_hip_fixture(jit_harness, monkeypatch)
    mh1_jit_codegen = importlib.import_module(f"{_PACKAGE_NAME}.mh1_jit_codegen")
    contract = {"fixture": "mh1-hiprtc-contract"}
    model_path = tmp_path / "mh1-hiprtc.json"
    model_path.write_text(
        json.dumps(
            {
                "model_type": "MACE_Nonlinear",
                "execution_contracts": {"MH1_UVU": contract},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_model_metadata",
        lambda _path: ("MACE_Nonlinear", False),
    )
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "MACENonlinearKokkosFloat",
        lambda _path, _head="": jit_harness["evaluator"],
        raising=False,
    )
    metadata = {
        **_MH1_CUDA_V4_METADATA_WITH_POLICY,
        "artifact_id": "jit-mh1-hip-fixture",
        "target_id": "sha256:hip-target-fixture",
        "node_state_policy": "recompute-v1",
    }
    launch_plan = {"schema": "symmetrix.jit.mh1.hip-launch-plan/2"}
    monkeypatch.setattr(
        mh1_jit_codegen,
        "execution_mh1_hip_module_v4_metadata",
        lambda *_args, **_kwargs: metadata,
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "render_execution_mh1_hip_module_v4",
        lambda *_args, **_kwargs: "generated MH1 HIP module source",
    )
    monkeypatch.setattr(
        mh1_jit_codegen,
        "execution_mh1_hip_module_v4_launch_plan",
        lambda *_args, **_kwargs: launch_plan,
    )
    monkeypatch.setenv("SYMMETRIX_JIT_HIP_JIT_BACKEND", "hiprtc")

    calculator = Symmetrix(
        model_path,
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_compiler_backend == "hiprtc"
    assert calculator.evaluator.jit_mh1_cuda_plugin_v4_ready
    assert calculator.jit_artifact_id == "jit-mh1-v4-fixture"
    source, options = jit_harness["hiprtc_prepare_calls"][0]
    assert source == "generated MH1 HIP module source"
    assert options["build"]["generator"] == "symmetrix.jit.mh1-hip-module-v1"
    assert options["build"]["launch_plan"] == launch_plan
    event = jit_harness["events"][-1]
    assert event[:2] == ("mh1_cuda_v4_load", str(jit_harness["artifact"]))


def test_jit_hip_explicit_hiprtc_uses_device_module(jit_harness, monkeypatch):
    _configure_hip_fixture(jit_harness, monkeypatch)
    monkeypatch.setenv("SYMMETRIX_JIT_HIP_JIT_BACKEND", "hiprtc")

    calculator = Symmetrix(
        jit_harness["model_path"],
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calculator.jit_compiler_backend == "hiprtc"
    assert len(jit_harness["hiprtc_prepare_calls"]) == 1
    assert len(jit_harness["hip_prepare_calls"]) == 0
    source, options = jit_harness["hiprtc_prepare_calls"][0]
    assert source == "generated HIP module source"
    assert options["build"]["generator"] == "symmetrix.jit.r1-hip-module-v2"
    assert options["build"]["gpu_codegen"]["target"]["backend"] == "hip"
    assert options["build"]["gpu_codegen"]["launch_plan"] == {"artifact_kind": "module"}
    assert options["artifact_name"] == "factorized_hip_module"


def test_jit_hip_automatic_compile_failure_never_invokes_hipcc(
    jit_harness, monkeypatch
):
    jit = _configure_hip_fixture(jit_harness, monkeypatch)
    hiprtc_failure = SimpleNamespace(
        status="fallback",
        cache_key="hiprtc-cache-key",
        artifact_path=None,
        diagnostics=("hipRTC compile failed",),
        reason="invalid generated HIP",
        available=False,
    )
    monkeypatch.setattr(
        jit,
        "prepare_hiprtc_jit_artifact",
        lambda source, **kwargs: hiprtc_failure,
    )

    with pytest.raises(RuntimeError, match="invalid generated HIP"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )

    assert jit_harness["hip_prepare_calls"] == []


def test_jit_hip_automatic_load_failure_never_invokes_hipcc(jit_harness, monkeypatch):
    _configure_hip_fixture(jit_harness, monkeypatch)
    load_calls = []

    def load(path, *args):
        load_calls.append((path, args))
        raise RuntimeError("invalid hsaco fixture")

    jit_harness["evaluator"]._load_jit_hip_plugin = load
    with pytest.raises(RuntimeError, match="invalid hsaco fixture"):
        Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )

    assert len(jit_harness["hiprtc_prepare_calls"]) == 2
    assert jit_harness["hip_prepare_calls"] == []
    assert len(load_calls) == 2
    assert len(jit_harness["quarantine_calls"]) == 1


def test_jit_explicit_nvcc_quarantines_broken_artifact(jit_harness, monkeypatch):
    monkeypatch.setattr(
        symmetrix_calc.symmetrix,
        "_kokkos_default_execution_space",
        lambda: "Cuda",
    )
    monkeypatch.setenv("SYMMETRIX_JIT_CUDA_JIT_BACKEND", "nvcc")
    load_calls = []

    def load(path, *args):
        load_calls.append((path, args))
        if len(load_calls) == 1:
            raise RuntimeError("broken retained artifact")
        jit_harness["evaluator"].jit_cuda_plugin_ready = True
        jit_harness["evaluator"].jit_cuda_plugin_artifact_id = "jit-r1-fixture"

    jit_harness["evaluator"]._load_jit_cuda_plugin = load

    with pytest.warns(FutureWarning, match="runtime nvcc"):
        calculator = Symmetrix(
            jit_harness["model_path"],
            dtype="float32",
            use_kokkos=True,
            streamed_edges="factorized",
        )

    assert calculator.jit_compiler_backend == "nvcc"
    assert jit_harness["nvrtc_prepare_calls"] == []
    assert len(jit_harness["cuda_prepare_calls"]) == 2
    assert len(load_calls) == 2
    assert len(jit_harness["quarantine_calls"]) == 1
    assert calculator.jit_status == "built"


def test_jit_uses_extracted_checkpoint_data_after_temporary_json_closes(
    jit_harness, monkeypatch, tmp_path
):
    pytest.importorskip("torch")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    extracted = jit_harness["model"]
    calls = []

    def checkpoint_factory(path, head=""):
        assert head == ""
        jit_harness["construct_paths"].append(path)
        if Path(path) == checkpoint:
            raise RuntimeError("not native JSON")
        assert Path(path).is_file()
        calls.append(json.loads(Path(path).read_text(encoding="utf-8")))
        return jit_harness["evaluator"]

    monkeypatch.setattr(symmetrix_calc.symmetrix, "MACEKokkosFloat", checkpoint_factory)
    monkeypatch.setattr(Symmetrix, "_raise_if_macefield_checkpoint", lambda *args: None)
    extract_mace_data = importlib.import_module(f"{_PACKAGE_NAME}.extract_mace_data")
    monkeypatch.setattr(
        extract_mace_data, "extract_mace_data", lambda *args, **kwargs: extracted
    )

    calculator = Symmetrix(
        checkpoint,
        dtype="float32",
        use_kokkos=True,
        streamed_edges="factorized",
    )

    assert calls == [extracted]
    assert calculator.jit_status == "built"
    assert jit_harness["render_contracts"] == [extracted["execution_contracts"]["R1"]]
