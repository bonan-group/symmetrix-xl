from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import symmetrix.oam_jit_artifact as oam_jit_artifact
from ase.atoms import Atoms


FIXTURE = Path(__file__).parent / "data" / "nequip" / "tiny_nequip_v1.json"
OAM_FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"
# The frontend prepares a direct artifact through the entry point the family
# declares, so a stub replaces that declared entry point rather than a copy of
# it on the frontend module.
OAM_HOST_ADAPTER = oam_jit_artifact
OAM_HOST_PREPARER = "prepare_nequip_oam_receiver_host_artifact"


def _calculator_class():
    try:
        from symmetrix import NequIPCalculator
    except Exception as error:
        pytest.skip(f"ASE/native NequIP calculator is unavailable: {error}")
    return NequIPCalculator


def test_non_json_inputs_are_rejected_before_native_loading(tmp_path):
    calculator = _calculator_class()
    with pytest.raises(ValueError, match="exported Symmetrix NequIP JSON"):
        calculator(tmp_path / "model.pt2")


def test_ordinary_nequip_v1_contracts_are_rejected(monkeypatch):
    import symmetrix.nequip_calculator as calculator_module

    class Evaluator:
        def __init__(self, *args):
            self.args = args

    class Native:
        NequIP = Evaluator

    monkeypatch.setattr(calculator_module, "load_backend", lambda: Native)
    with pytest.raises(
        ValueError, match="ordinary NequIP v1 contracts are no longer supported"
    ):
        calculator_module.NequIPCalculator(FIXTURE)


def test_missing_native_backend_reports_the_missing_class(monkeypatch):
    """A native build without the declared class must fail with that class name."""

    import symmetrix.nequip_calculator as calculator_module

    class Artifact:
        artifact_id = "oam-host-fixture"
        artifact_path = Path("/tmp/oam-host-fixture.so")
        status = "cached"

    class Native:
        # An OAM contract requires the OAM Kokkos evaluator.
        NequIP = object

        @staticmethod
        def _kokkos_is_initialized():
            return True

    monkeypatch.setattr(
        OAM_HOST_ADAPTER,
        OAM_HOST_PREPARER,
        lambda *args, **kwargs: Artifact(),
    )
    monkeypatch.setattr(calculator_module, "load_backend", lambda: Native)
    with pytest.raises(RuntimeError, match="NequIPOAMKokkos"):
        calculator_module.NequIPCalculator(OAM_FIXTURE)


def test_oam_direct_routing_prepares_and_passes_the_receiver_plugin(monkeypatch):
    import symmetrix.nequip_calculator as calculator_module

    requests = []

    class Artifact:
        artifact_id = "oam-host-fixture"
        artifact_path = Path("/tmp/oam-host-fixture.so")
        status = "cached"

    class Evaluator:
        r_cut = 4.5
        atomic_numbers = [1, 94]
        available_heads = []

        def __init__(self):
            self.streamed_edges_mode = None

        def set_streamed_edges(self, selected):
            self.streamed_edges_mode = selected

        def _set_execution_plan_request(self, algorithm, profile):
            self.execution_plan_request = (algorithm, profile)

    class Native:
        NequIPOAMKokkos = Evaluator

        @staticmethod
        def _kokkos_is_initialized():
            return True

    def prepare(path, *, precision, target, native_module):
        requests.append((path, precision, target, native_module))
        return Artifact()

    monkeypatch.setattr(OAM_HOST_ADAPTER, OAM_HOST_PREPARER, prepare)
    monkeypatch.setattr(calculator_module, "load_backend", lambda: Native)
    monkeypatch.setattr(
        Native, "NequIPOAMKokkos", lambda path, plugin, **kwargs: Evaluator()
    )

    calculator = calculator_module.NequIPCalculator(
        OAM_FIXTURE, dtype="float64", execution_profile="speed"
    )

    # The artifact is prepared once, through the entry point the family
    # declares for the host target and at the calculator's precision, and the
    # diagnostic it reports must name the artifact the evaluator received.
    assert requests == [(OAM_FIXTURE, "float64", "host", Native)]
    assert calculator.streamed_edges == "direct"
    assert calculator.evaluator.streamed_edges_mode == "direct"
    assert calculator.evaluator.execution_plan_request == ("direct", "speed")
    assert calculator.execution_profile == "speed"
    assert "oam-host-fixture" in calculator._oam_artifact_diagnostic
    assert "cache_status=cached" in calculator._oam_artifact_diagnostic


def test_oam_direct_preparation_failure_is_not_downgraded(monkeypatch):
    import symmetrix.nequip_calculator as calculator_module

    class Native:
        @staticmethod
        def _kokkos_is_initialized():
            return True

    def prepare(*args, **kwargs):
        raise RuntimeError("host plugin compilation failed")

    monkeypatch.setattr(OAM_HOST_ADAPTER, OAM_HOST_PREPARER, prepare)
    monkeypatch.setattr(calculator_module, "load_backend", lambda: Native)

    with pytest.raises(RuntimeError, match="host plugin compilation failed"):
        calculator_module.NequIPCalculator(OAM_FIXTURE)


@pytest.mark.parametrize(
    "environment",
    (
        {"backend": "hip", "execution_space": "HIP"},
        {"backend": "rocm", "execution_space": "Kokkos::HIP"},
    ),
)
def test_oam_rejects_hip_before_artifact_preparation(environment):
    import symmetrix.nequip_calculator as calculator_module

    class Native:
        @staticmethod
        def _execution_device_execution_environment():
            return environment

    with pytest.raises(RuntimeError, match="HIP is not supported"):
        calculator_module._execution_artifact_target(Native)


@pytest.mark.parametrize("profile", (None, "", "latency", 1, True))
def test_oam_execution_profile_rejects_values_outside_the_native_vocabulary(profile):
    import symmetrix.nequip_calculator as calculator_module

    with pytest.raises(
        ValueError, match="execution_profile must be 'capacity' or 'speed'"
    ):
        calculator_module.NequIPCalculator(OAM_FIXTURE, execution_profile=profile)


def test_streamed_edges_rejects_unknown_modes():
    import symmetrix.nequip_calculator as calculator_module

    # The OAM mode set is literal: the compatibility spellings the MACE
    # calculator resolves through its own alias table are not modes here.
    for mode in (
        "auto",
        "generic",
        "non-compiled",
        "all_interactions",
        "materialized",
        "factorized",
        "direct_streamed",
        "",
        "Direct",
    ):
        with pytest.raises(
            ValueError,
            match=(
                "streamed_edges must be 'direct', 'reference', 'debug_v1', or "
                "'mlp_reference'"
            ),
        ):
            calculator_module.NequIPCalculator(OAM_FIXTURE, streamed_edges=mode)


@pytest.mark.parametrize(
    ("mode", "routing"),
    (
        ("direct", "direct"),
        ("reference", "reference"),
        ("debug_v1", "reference"),
        ("mlp_reference", "reference"),
    ),
)
def test_oam_documented_mode_spellings_resolve_onto_a_routing(
    monkeypatch, mode, routing
):
    """Every documented OAM spelling is accepted and reports its routing.

    The compact spellings the MACE calculator resolves through its own alias
    table are not modes here, and the three non-direct spellings resolve onto
    the single reference routing, which reaches the whole-evaluator entry point
    with no artifact at all.
    """

    import symmetrix.nequip_calculator as calculator_module

    prepared = []

    class Artifact:
        artifact_id = "oam-host-fixture"
        artifact_path = Path("/tmp/oam-host-fixture.so")
        status = "cached"

    class Evaluator:
        r_cut = 4.5
        atomic_numbers = [1, 94]
        available_heads = []

        def __init__(self, *args, **kwargs):
            self.args = args
            self.streamed_edges_mode = None

        def set_streamed_edges(self, selected):
            self.streamed_edges_mode = selected

        def _set_execution_plan_request(self, algorithm, profile):
            self.execution_plan_request = (algorithm, profile)

    class Native:
        NequIPOAMKokkos = Evaluator

        @staticmethod
        def _kokkos_is_initialized():
            return True

    def prepare(path, *, precision, target, native_module):
        assert native_module is Native
        prepared.append((precision, target))
        return Artifact()

    monkeypatch.setattr(OAM_HOST_ADAPTER, OAM_HOST_PREPARER, prepare)
    monkeypatch.setattr(calculator_module, "load_backend", lambda: Native)

    calculator = calculator_module.NequIPCalculator(
        OAM_FIXTURE, dtype="float64", streamed_edges=mode
    )

    assert calculator.streamed_edges == routing
    assert calculator.streamed_edges_requested == mode
    assert calculator.streamed_edges_alias == (
        mode if mode in ("debug_v1", "mlp_reference") else None
    )
    assert calculator.implemented_properties
    # The native request preserves the routing distinction: direct requires an
    # admitted generated artifact, while reference owns the portable path.
    assert calculator.evaluator.streamed_edges_mode == routing
    assert calculator.evaluator.execution_plan_request == (routing, "capacity")
    if routing == "direct":
        assert prepared == [("float64", "host")]
        assert len(calculator.evaluator.args) == 2
    else:
        assert prepared == []
        assert calculator.evaluator.args == (str(OAM_FIXTURE),)


def _oam_stub_backend(monkeypatch, evaluator_class, *, prepare=None):
    """Install a stub native backend and artifact preparation for the fixture."""

    import symmetrix.nequip_calculator as calculator_module

    class Artifact:
        artifact_id = "oam-host-fixture"
        artifact_path = Path("/tmp/oam-host-fixture.so")
        status = "cached"

    class Native:
        NequIPOAMKokkos = evaluator_class

        @staticmethod
        def _kokkos_is_initialized():
            return True

    if prepare is None:
        prepare = lambda *args, **kwargs: Artifact()  # noqa: E731 - a stub default
    monkeypatch.setattr(OAM_HOST_ADAPTER, OAM_HOST_PREPARER, prepare)
    monkeypatch.setattr(calculator_module, "load_backend", lambda: Native)
    return calculator_module


def _oam_atoms():
    return Atoms(
        numbers=[1, 94],
        positions=[[0.0, 0.0, 0.0], [1.2, 0.3, -0.2]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )


def test_oam_direct_request_without_the_receiver_surface_raises(monkeypatch):
    """A `direct` request must fail, not silently run the reference path."""

    class Evaluator:
        r_cut = 4.5
        atomic_numbers = [1, 94]
        available_heads = []

        def __init__(self, *args, **kwargs):
            self.streamed_edges_mode = None
            self.reduced = []

        def set_streamed_edges(self, selected):
            self.streamed_edges_mode = selected

        def _set_execution_plan_request(self, algorithm, profile):
            self.execution_plan_request = (algorithm, profile)

        def compute_node_energies_forces(self, *args):
            self.reduced.append(args)

    calculator_module = _oam_stub_backend(monkeypatch, Evaluator)
    calculator = calculator_module.NequIPCalculator(
        OAM_FIXTURE, dtype="float64", streamed_edges="direct"
    )
    assert calculator.streamed_edges == "direct"

    with pytest.raises(
        RuntimeError,
        match=(
            r"family 'OAM' declares the 'receiver' evaluator capability, but the "
            r"evaluator does not provide 'prepare_graph'"
        ),
    ):
        calculator.calculate(_oam_atoms(), properties=("energy",))

    assert calculator.evaluator.reduced == []


def test_oam_reference_routing_requires_the_declared_surface(monkeypatch):
    class Evaluator:
        r_cut = 4.5
        atomic_numbers = [1, 94]
        available_heads = []

        def __init__(self, *args, **kwargs):
            pass

        def set_streamed_edges(self, selected):
            pass

        def _set_execution_plan_request(self, algorithm, profile):
            pass

    calculator_module = _oam_stub_backend(monkeypatch, Evaluator)
    calculator = calculator_module.NequIPCalculator(
        OAM_FIXTURE, dtype="float64", streamed_edges="mlp_reference"
    )
    assert calculator.streamed_edges == "reference"

    with pytest.raises(
        RuntimeError,
        match=(
            r"family 'OAM' declares the 'reference' evaluator capability, but the "
            r"evaluator does not provide 'compute_node_energies_forces'"
        ),
    ):
        calculator.calculate(_oam_atoms(), properties=("energy",))


def test_oam_direct_request_without_the_mode_handshake_raises(monkeypatch):
    class Evaluator:
        r_cut = 4.5
        atomic_numbers = [1, 94]
        available_heads = []

        def __init__(self, *args, **kwargs):
            pass

    calculator_module = _oam_stub_backend(monkeypatch, Evaluator)

    with pytest.raises(
        RuntimeError,
        match=(
            r"family 'OAM' declares the 'mode_handshake' evaluator capability, "
            r"but the evaluator does not provide 'set_streamed_edges'"
        ),
    ):
        calculator_module.NequIPCalculator(
            OAM_FIXTURE, dtype="float64", streamed_edges="direct"
        )


def test_retired_use_kokkos_keyword_is_rejected():
    """The retired serial fallback is a rejection, not a warning.

    ``use_kokkos=False`` used to select serial generic execution and warn that
    the request resolved to ``mlp_reference``; the serial evaluator and its
    fallback are gone, so the constructor rejects the keyword outright.
    """

    import symmetrix.nequip_calculator as calculator_module

    with pytest.raises(TypeError, match="use_kokkos has been retired:"):
        calculator_module.NequIPCalculator(OAM_FIXTURE, use_kokkos=False)


# ---------------------------------------------------------------------------
# The canonical path: readiness, the plan report, and the declared surface
# ---------------------------------------------------------------------------


def _oam_native_module():
    """Return the native backend, skipping when the OAM Kokkos class is absent."""

    try:
        from symmetrix import symmetrix as native
    except Exception as error:  # pragma: no cover - environment without backend
        pytest.skip(f"native Symmetrix package is unavailable: {error}")
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("NequIPOAMKokkos is unavailable in this extension")
    return native


def _oam_native():
    """Return the native backend, skipping when the host OAM path is absent."""

    native = _oam_native_module()
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the OAM host artifact")
    return native


def _oam_routed_artifact_target():
    """The artifact target this build's OAM direct routing binds.

    The frontend's own rule decides the target, so the tests read it there
    rather than assuming a host build: a host Kokkos build binds the
    receiver-owned ABI v2 host artifact and a CUDA build its generated CUDA R1
    plugin.
    """

    from symmetrix.nequip_calculator import _execution_artifact_target

    return _execution_artifact_target(_oam_native_module())[0]


def _supported_atoms():
    """The fixture's two supported species on a small periodic cell."""

    return Atoms(
        numbers=[1, 8],
        positions=[[0.0, 0.0, 0.0], [1.2, 0.3, -0.2]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )


def _edge_bindings(plan):
    """Return the tier bindings of the tensor-product nodes of the program."""

    edges = [
        binding
        for binding in plan["bindings"]
        if binding["capability_key"] == "edge_tensor_product"
    ]
    assert edges, plan["bindings"]
    return edges


def test_oam_reference_request_evaluates_without_any_artifact(tmp_path, monkeypatch):
    """The reference routing is the artifact-free path, end to end.

    No artifact is prepared for it, and the evaluation it produces is the
    whole-evaluator generic entry point, so a host build evaluates with an
    empty cache and no compiler involvement at all.  That is the premise: a
    device build has no compiler-free OAM implementation, so it refuses this
    construction outright and the host build is the only place the behaviour
    exists.
    """

    native = _oam_native()
    if native._kokkos_default_execution_space() not in ("Serial", "OpenMP"):
        pytest.skip(
            "the artifact-free OAM reference path requires a host Kokkos build; a "
            "device build refuses construction without a generated device "
            "tensor-product plugin"
        )
    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    calculator = NequIPCalculator(
        OAM_FIXTURE, dtype="float64", streamed_edges="reference"
    )

    assert calculator.streamed_edges == "reference"
    assert calculator._oam_artifact_diagnostic == ""
    assert calculator._oam_cuda_artifact_diagnostic == ""
    assert calculator.capabilities["generated"] is False

    atoms = _supported_atoms()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert np.isfinite(atoms.get_forces()).all()


def test_json_only_oam_evaluation_does_not_import_upstream_stack(tmp_path):
    """The deployed OAM JSON runtime stays independent of exporter packages."""

    native = _oam_native_module()
    if native._kokkos_default_execution_space() not in ("Serial", "OpenMP"):
        pytest.skip("the independent JSON-runtime import probe uses the host backend")
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import numpy as np
        from ase import Atoms
        from symmetrix import NequIPCalculator

        model = Path(sys.argv[1])
        atoms = Atoms(
            numbers=[1, 8],
            positions=[[0.0, 0.0, 0.0], [1.2, 0.3, -0.2]],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        )
        atoms.calc = NequIPCalculator(
            model, dtype="float64", streamed_edges="reference"
        )
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()
        imported = sorted(
            name for name in sys.modules
            if name.split(".", 1)[0] in {"torch", "nequip", "e3nn"}
        )
        assert imported == [], imported
        """
    )
    environment = os.environ.copy()
    environment["SYMMETRIX_BACKEND"] = "cpu"
    environment["SYMMETRIX_JIT_CACHE"] = str(tmp_path / "jit-cache")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(OAM_FIXTURE)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_oam_direct_request_without_a_qualified_artifact_fails_at_preparation(
    tmp_path, monkeypatch
):
    """A direct request does not fall back; it reports the four-way cause.

    The preparation is the only place that can see why an artifact was not
    produced, so the failure is reported there with the same four-way
    diagnostic every family reports.  The cause is the one this build's direct
    routing depends on, because the direct artifact is the build's: the host
    receiver-owned artifact needs a host C++ compiler and the CUDA R1 plugin
    needs NVRTC, so each branch makes its own artifact compiler unavailable and
    the classification and the enumeration stay the shared factory's.
    """

    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    if _oam_routed_artifact_target() == "cuda":
        # The device artifact compiler has no environment-only failure mode:
        # the native NVRTC probe opens the library once per process and
        # ``SYMMETRIX_JIT_NVRTC_LIBRARY`` can only add a candidate, never
        # remove availability, so the probe itself is what is made
        # unavailable here -- the device counterpart of pointing ``CXX`` at a
        # missing executable for the host artifact below.
        import symmetrix.jit as jit_module

        # The probe is only consulted by the automatic policy, so the request
        # is pinned to it the way the host branch pins the compiler itself.
        monkeypatch.delenv(
            jit_module.CUDA_JIT_BACKEND_ENVIRONMENT_VARIABLE, raising=False
        )
        monkeypatch.setattr(
            jit_module,
            "jit_nvrtc_information",
            lambda: {
                "available": False,
                "reason": "no CUDA JIT compiler is installed in this environment",
            },
        )
        expected = (
            r"(?s)the CUDA JIT backend is unavailable.*"
            r"Direct R1 artifact diagnostic \(compilation-unavailable\): "
            r"no working cuda compiler is available in this environment.*"
            r"\(a\) .*\(b\) .*\(c\) .*\(d\) "
        )
    else:
        monkeypatch.setenv("CXX", str(tmp_path / "missing-compiler"))
        expected = (
            r"(?s)CXX compiler executable was not found.*"
            r"Direct R1 artifact diagnostic \(compilation-unavailable\): "
            r"no working host compiler is available.*"
            r"\(a\) .*\(b\) .*\(c\) .*\(d\) "
        )

    with pytest.raises(RuntimeError, match=expected):
        NequIPCalculator(OAM_FIXTURE, dtype="float64", streamed_edges="direct")


@pytest.mark.parametrize("profile", ("speed", "capacity"))
def test_oam_direct_request_reports_the_plan_and_every_node_tier(
    tmp_path, monkeypatch, profile
):
    """A direct request surfaces the plan and the tier bound to every node.

    The report is the native plan of the compiled program, so the selected
    implementation is the one the execution diagnostic names and every operator
    node carries the tier that serves it.  Both are the build's own tier rather
    than a family choice, so the expected plan is the one this build's direct
    routing binds: a CUDA build dispatches its generated R1 plugin and a host
    build the aligned receiver/source-owned ABI v7 host artifact.
    """

    _oam_native()
    from symmetrix import NequIPCalculator

    target = _oam_routed_artifact_target()
    expected_id = (
        "generated-r1"
        if target == "cuda"
        else "receiver_owned_full_message_oam_host_v7"
    )

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    calculator = NequIPCalculator(
        OAM_FIXTURE,
        dtype="float64",
        streamed_edges="direct",
        execution_profile=profile,
    )
    atoms = _supported_atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()

    plan = calculator.execution_plan
    assert plan["binding_report_version"] == 1
    assert plan["requested_profile"] == profile
    assert plan["selected_id"] == ""
    assert plan["state"] == "unqualified"
    assert [candidate["id"] for candidate in plan["candidates"]] == [
        "oam-direct-speed",
        "oam-direct-capacity",
        "oam-fixed-workspace-v1",
    ]
    assert plan["available_bytes"] is None
    assert plan["reserve_bytes"] is None
    for candidate in plan["candidates"]:
        assert candidate["backend"] == target
        assert candidate["precision"] == "float64"
        assert candidate["required_capability"]
        assert candidate["empirical_qualification"] == "required"
    for candidate in plan["candidates"][:2]:
        assert candidate["estimate_available"] is True
        assert candidate["estimated_bytes"] > 0
        assert "transient" in candidate["estimate_scope"]
    assert plan["candidates"][2]["estimate_available"] is False
    assert plan["candidates"][2]["estimated_bytes"] is None
    assert (
        plan["candidates"][1]["estimated_bytes"]
        <= plan["candidates"][0]["estimated_bytes"]
    )
    assert plan["candidates"][0]["schedule"] == "throughput"
    assert plan["candidates"][1]["schedule"] == "capacity"
    assert plan["candidates"][2]["schedule"] == "fixed_workspace"
    assert "no empirically qualified direct candidate" in plan["selection_reason"]
    assert plan["artifact_origin"] in {"prebuilt", "cache", "runtime"}
    assert plan["loaded_artifact_identity"]
    assert plan["fallback_count"] == 0
    assert "unavailable" in plan["candidates"][2]["reason"]
    # The plan selects a whole-plan profile; each binding names the actual
    # implementation dispatched for its node.
    assert f"implementation={expected_id}" in calculator.execution_diagnostic
    assert calculator.capabilities["generated"] is True

    edges = _edge_bindings(plan)
    assert {binding["tier"] for binding in edges} == {"generated"}
    assert {binding["implementation_id"] for binding in edges} == {expected_id}
    generated_keys = {
        binding["capability_key"]
        for binding in plan["bindings"]
        if binding["tier"] == "generated"
    }
    assert generated_keys == {"edge_tensor_product"}
    assert all(binding["qualified"] for binding in plan["bindings"])
    assert all(binding["target"] == target for binding in plan["bindings"])


def test_oam_capacity_admission_ranks_exact_non_fixed_candidates(tmp_path, monkeypatch):
    _oam_native()
    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    calculator = NequIPCalculator(OAM_FIXTURE, dtype="float64", streamed_edges="direct")
    evaluator = calculator.evaluator
    atoms = _supported_atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    evaluator._set_execution_plan_request("direct", "capacity")

    initial = evaluator.execution_plan_report["candidates"]
    speed_estimate = initial[0]["estimated_bytes"]
    capacity_estimate = initial[1]["estimated_bytes"]
    assert capacity_estimate < speed_estimate

    target = _oam_routed_artifact_target()
    artifact_identity = evaluator.family_artifact_readiness["artifact_id"]
    evaluator._set_execution_plan_qualification(
        "oam-direct-speed",
        artifact_identity,
        target,
        "float64",
        "throughput",
        "generated_edge_tensor_product",
        "qualified",
        speed_estimate,
        speed_estimate + 10,
        0,
    )
    evaluator._set_execution_plan_qualification(
        "oam-direct-capacity",
        artifact_identity,
        target,
        "float64",
        "capacity",
        "exact_capacity_layout",
        "qualified",
        capacity_estimate,
        speed_estimate + 10,
        0,
    )
    assert evaluator.execution_plan_report["selected_id"] == "oam-direct-speed"
    assert evaluator.execution_plan_report["boundary_attempt"] is False

    evaluator._set_execution_plan_qualification(
        "oam-direct-capacity",
        artifact_identity,
        target,
        "float64",
        "capacity",
        "exact_capacity_layout",
        "qualified",
        capacity_estimate,
        capacity_estimate,
        0,
    )
    assert evaluator.execution_plan_report["selected_id"] == "oam-direct-capacity"
    assert evaluator.execution_plan_report["boundary_attempt"] is False

    evaluator._set_execution_plan_qualification(
        "oam-direct-speed",
        artifact_identity,
        target,
        "float64",
        "throughput",
        "generated_edge_tensor_product",
        "qualified",
        speed_estimate,
        capacity_estimate - 1,
        0,
    )
    report = evaluator.execution_plan_report
    assert report["selected_id"] == "oam-direct-capacity"
    assert report["boundary_attempt"] is True
    assert "smallest qualified non-fixed plan" in report["selection_reason"]


def test_oam_exact_layout_drives_estimates_admission_and_arena(tmp_path, monkeypatch):
    _oam_native()
    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    calculator = NequIPCalculator(OAM_FIXTURE, dtype="float64", streamed_edges="direct")
    atoms = _supported_atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    evaluator = calculator.evaluator

    candidates = {
        candidate["id"]: candidate
        for candidate in evaluator.execution_plan_report["candidates"]
    }
    speed = candidates["oam-direct-speed"]["estimated_bytes"]
    capacity = candidates["oam-direct-capacity"]["estimated_bytes"]
    assert speed > 0
    assert 0 < capacity <= speed

    memory = evaluator.execution_memory_report
    assert memory["active_bytes"] == speed
    assert memory["retained_capacity_bytes"] >= memory["active_bytes"]
    assert memory["peak_transient_bytes"] == speed
    assert memory["replacements"] == 1
    assert memory["high_water_reason"] == "initial graph"
    assert memory["runtime_subview_count"] > 0
    assert memory["canary_initializations"] == 1
    assert memory["canaries_intact"] is True
    assert memory["arena_bilinear_workspace_bytes"] >= 0
    assert memory["independent_graph_temporary_bytes"] == 0

    target = _oam_routed_artifact_target()
    artifact_identity = evaluator.family_artifact_readiness["artifact_id"]
    evaluator._set_execution_plan_qualification(
        "oam-direct-speed",
        artifact_identity,
        target,
        "float64",
        "throughput",
        "generated_edge_tensor_product",
        "qualified",
        speed + 64,
        speed * 2,
        0,
    )
    candidate = evaluator.execution_plan_report["candidates"][0]
    assert candidate["qualified"] is False
    assert "planner estimate mismatch" in candidate["reason"]

    evaluator._set_execution_plan_qualification(
        "oam-direct-speed",
        artifact_identity,
        target,
        "float64",
        "throughput",
        "generated_edge_tensor_product",
        "qualified",
        speed,
        speed * 2,
        0,
    )
    candidate = evaluator.execution_plan_report["candidates"][0]
    assert candidate["qualified"] is True
    assert candidate["estimated_bytes"] == memory["active_bytes"]

    evaluator._set_execution_plan_request("direct", "capacity")
    evaluator._set_execution_plan_qualification(
        "oam-direct-capacity",
        artifact_identity,
        target,
        "float64",
        "capacity",
        "exact_capacity_layout",
        "qualified",
        capacity,
        capacity,
        0,
    )
    assert evaluator.execution_plan_report["selected_id"] == "oam-direct-capacity"
    moved = _supported_atoms()
    moved.positions[1, 0] += 0.01
    moved.calc = calculator
    assert np.isfinite(moved.get_potential_energy())
    capacity_memory = evaluator.execution_memory_report
    assert capacity_memory["active_bytes"] == capacity
    assert capacity_memory["active_bytes"] <= speed
    assert capacity_memory["canary_initializations"] == 2
    assert capacity_memory["canaries_intact"] is True


def test_oam_arena_reuses_capacity_for_a_smaller_active_graph(tmp_path, monkeypatch):
    _oam_native()
    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    calculator = NequIPCalculator(OAM_FIXTURE, dtype="float64", streamed_edges="direct")
    large = _supported_atoms().repeat((2, 1, 1))
    large.calc = calculator
    large_energy = large.get_potential_energy()
    large_memory = dict(calculator.evaluator.execution_memory_report)

    small = _supported_atoms()
    small.calc = calculator
    small_energy = small.get_potential_energy()
    small_memory = calculator.evaluator.execution_memory_report

    fresh = NequIPCalculator(OAM_FIXTURE, dtype="float64", streamed_edges="direct")
    reference = _supported_atoms()
    reference.calc = fresh
    reference_energy = reference.get_potential_energy()

    assert np.isfinite(large_energy)
    assert small_energy == pytest.approx(reference_energy, rel=1e-12, abs=1e-12)
    assert small_memory["active_bytes"] < large_memory["active_bytes"]
    assert (
        small_memory["retained_capacity_bytes"]
        == large_memory["retained_capacity_bytes"]
    )
    assert small_memory["replacements"] == large_memory["replacements"]
    assert large_memory["canary_initializations"] == 1
    assert small_memory["canary_initializations"] == 2
    assert small_memory["canaries_intact"] is True


def test_oam_arena_does_not_reinitialize_guards_for_identical_layout(
    tmp_path, monkeypatch
):
    _oam_native()
    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    calculator = NequIPCalculator(OAM_FIXTURE, dtype="float64", streamed_edges="direct")
    atoms = _supported_atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    evaluator = calculator.evaluator
    first = dict(evaluator.execution_memory_report)

    atoms.positions[1, 0] += 0.01
    atoms.get_potential_energy()
    second = evaluator.execution_memory_report

    assert first["canary_initializations"] == 1
    assert second["canary_initializations"] == 1
    assert second["replacements"] == first["replacements"]
    assert second["canaries_intact"] is True


def test_oam_qualification_is_invalidated_by_a_different_graph_layout(
    tmp_path, monkeypatch
):
    _oam_native()
    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    calculator = NequIPCalculator(OAM_FIXTURE, dtype="float64", streamed_edges="direct")
    first = _supported_atoms()
    first.calc = calculator
    first.get_potential_energy()
    evaluator = calculator.evaluator
    candidate = evaluator.execution_plan_report["candidates"][0]
    first_estimate = candidate["estimated_bytes"]
    first_memory = dict(evaluator.execution_memory_report)
    evaluator._set_execution_plan_qualification(
        "oam-direct-speed",
        evaluator.family_artifact_readiness["artifact_id"],
        _oam_routed_artifact_target(),
        "float64",
        "throughput",
        "generated_edge_tensor_product",
        "qualified",
        first_estimate,
        first_estimate * 4,
        0,
    )
    assert evaluator.execution_plan_report["candidates"][0]["qualified"] is True

    second = _supported_atoms().repeat((2, 1, 1))
    second.calc = calculator
    second.get_potential_energy()
    changed = evaluator.execution_plan_report["candidates"][0]
    second_memory = evaluator.execution_memory_report
    assert changed["estimated_bytes"] != first_estimate
    assert changed["qualified"] is False
    assert changed["empirical_qualification"] == "rejected"
    assert "layout" in changed["reason"]
    assert second_memory["replacements"] == first_memory["replacements"] + 1
    assert (
        second_memory["canary_initializations"]
        == first_memory["canary_initializations"] + 1
    )
    assert second_memory["canaries_intact"] is True


def test_oam_real_arena_guard_overwrite_is_reported(tmp_path, monkeypatch):
    native = _oam_native()
    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    calculator = NequIPCalculator(OAM_FIXTURE, dtype="float64", streamed_edges="direct")
    atoms = _supported_atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    evaluator = calculator.evaluator
    assert evaluator.execution_memory_report["canaries_intact"] is True

    native._debug_corrupt_oam_execution_arena_guard(evaluator)

    assert evaluator.execution_memory_report["canaries_intact"] is False
    atoms.positions[1, 0] += 0.01
    atoms.get_potential_energy()
    memory = evaluator.execution_memory_report
    assert memory["canary_initializations"] == 1
    assert memory["canaries_intact"] is False


def test_oam_plan_qualification_rejects_missing_estimate(tmp_path, monkeypatch):
    _oam_native()
    from symmetrix import NequIPCalculator

    target = _oam_routed_artifact_target()
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    evaluator = NequIPCalculator(
        OAM_FIXTURE, dtype="float64", streamed_edges="direct"
    ).evaluator
    evaluator._set_execution_plan_request("direct", "capacity")
    artifact_identity = evaluator.family_artifact_readiness["artifact_id"]
    evaluator._set_execution_plan_qualification(
        "oam-direct-speed",
        artifact_identity,
        target,
        "float64",
        "throughput",
        "generated_edge_tensor_product",
        "qualified",
        None,
        30,
        0,
    )
    report = evaluator.execution_plan_report
    assert report["candidates"][0]["qualified"] is False
    assert report["candidates"][0]["empirical_qualification"] == "rejected"
    assert report["candidates"][0]["estimated_bytes"] > 0
    assert report["selected_id"] == ""
    assert report["boundary_attempt"] is False
    assert "exact estimate is required" in report["candidates"][0]["reason"]


def test_oam_plan_qualification_rejects_mismatched_schedule(tmp_path, monkeypatch):
    _oam_native()
    from symmetrix import NequIPCalculator

    target = _oam_routed_artifact_target()
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    evaluator = NequIPCalculator(
        OAM_FIXTURE, dtype="float64", streamed_edges="direct"
    ).evaluator
    artifact_identity = evaluator.family_artifact_readiness["artifact_id"]
    exact_estimate = evaluator.execution_plan_report["candidates"][0]["estimated_bytes"]
    evaluator._set_execution_plan_qualification(
        "oam-direct-speed",
        artifact_identity,
        target,
        "float64",
        "capacity",
        "generated_edge_tensor_product",
        "qualified",
        exact_estimate,
        100,
        0,
    )
    candidate = evaluator.execution_plan_report["candidates"][0]
    assert candidate["qualified"] is False
    assert candidate["empirical_qualification"] == "rejected"
    assert "schedule mismatch" in candidate["reason"]


def test_oam_plan_qualification_rejects_replaced_artifact_identity(
    tmp_path, monkeypatch
):
    _oam_native()
    from symmetrix import NequIPCalculator

    target = _oam_routed_artifact_target()
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "cache"))
    evaluator = NequIPCalculator(
        OAM_FIXTURE, dtype="float64", streamed_edges="direct"
    ).evaluator
    artifact_identity = evaluator.family_artifact_readiness["artifact_id"]
    exact_estimate = evaluator.execution_plan_report["candidates"][0]["estimated_bytes"]
    arguments = (
        "oam-direct-speed",
        artifact_identity,
        target,
        "float64",
        "throughput",
        "generated_edge_tensor_product",
        "qualified",
        exact_estimate,
        100,
        0,
    )
    evaluator._set_execution_plan_qualification(*arguments)
    assert evaluator.execution_plan_report["candidates"][0]["qualified"] is True

    replaced = list(arguments)
    replaced[1] = artifact_identity + "-replacement"
    evaluator._set_execution_plan_qualification(*replaced)
    candidate = evaluator.execution_plan_report["candidates"][0]
    assert candidate["qualified"] is False
    assert candidate["empirical_qualification"] == "rejected"
    assert candidate["artifact_identity"] == artifact_identity + "-replacement"
    assert "artifact identity mismatch" in candidate["reason"]
