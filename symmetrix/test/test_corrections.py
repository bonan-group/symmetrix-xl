import bisect
import contextlib
import signal
import warnings
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from ase import Atoms, units

import symmetrix as native_symmetrix
import symmetrix.corrections as corrections
from symmetrix.corrections import (
    CorrectionPipeline,
    D2Correction,
    D3Correction,
    FourierD3Correction,
    GraphRequirement,
    build_correction_runtime,
)


class _FakeRuntime:
    properties = frozenset({"energy", "energies", "forces", "stress"})
    graph_requirement = GraphRequirement(
        pair_cutoff=12.0,
        coordination_cutoff=8.0,
        skin=1.0,
    )

    def __init__(self, outputs=None):
        self.outputs = outputs or {
            "energy": -0.5,
            "energies": np.array([-0.2, -0.3]),
            "forces": np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
            "stress": np.arange(6, dtype=float) * 0.1,
        }
        self.requested_properties = []

    def calculate(self, atoms, properties):
        self.requested_properties.append(tuple(properties))
        return {name: self.outputs[name] for name in properties if name in self.outputs}

    def diagnostics(self):
        return {"runtime": "fake", "calls": len(self.requested_properties)}


class _RecordingNativeEvaluator:
    def __init__(self, *_args):
        self.compute_calls = []
        self.result_requests = []
        self.num_nodes = 0

    def prepare_host_graph(
        self,
        atomic_numbers,
        _receiver_offsets,
        _sources,
        _shifts,
        _positions,
        _cell,
        _build_cutoff,
    ):
        self.num_nodes = len(atomic_numbers)

    def compute(self, compute_forces, compute_stress):
        self.compute_calls.append((compute_forces, compute_stress))

    def result(self, properties=None):
        requested = None if properties is None else tuple(properties)
        self.result_requests.append(requested)
        outputs = {
            "energy": -0.5,
            "atomic_energies": np.full(self.num_nodes, -0.5 / self.num_nodes),
            "forces": np.zeros((self.num_nodes, 3)),
            "stress": np.zeros(6),
        }
        return (
            outputs
            if requested is None
            else {name: outputs[name] for name in requested}
        )

    def graph_diagnostics(self):
        return {
            "graph_backend": "host",
            "num_nodes": self.num_nodes,
            "num_edges": 0,
            "generation": 1,
            "build_cutoff": 0.0,
        }


class _RecordingNative:
    D2Kokkos = _RecordingNativeEvaluator
    D2KokkosFloat = _RecordingNativeEvaluator
    D3Kokkos = _RecordingNativeEvaluator
    D3KokkosFloat = _RecordingNativeEvaluator

    @staticmethod
    def _kokkos_default_execution_space():
        return "Serial"


def _install_fake_factory(monkeypatch, runtime, *, seen=None):
    def factory(configuration, *, native, dtype):
        if seen is not None:
            seen.append((configuration, native, dtype))
        return runtime

    monkeypatch.setitem(corrections._RUNTIME_FACTORIES, D3Correction, factory)


def test_correction_configurations_are_immutable_and_use_angstrom_defaults():
    d2 = D2Correction()
    d3 = D3Correction()

    assert d2.cutoff == pytest.approx(40.0 * units.Bohr)
    assert d2.skin == pytest.approx(0.5)
    assert d3.cutoff == pytest.approx(40.0 * units.Bohr)
    assert d3.coordination_cutoff == pytest.approx(40.0 * units.Bohr)
    assert d3.skin == pytest.approx(0.5)
    with pytest.raises(FrozenInstanceError):
        d3.cutoff = 10.0


@pytest.mark.parametrize("damping", ["d2", "becke-johnson", "BJ", ""])
def test_d3_rejects_unknown_damping(damping):
    with pytest.raises(ValueError, match="damping"):
        D3Correction(damping=damping)


@pytest.mark.parametrize(
    ("constructor", "keyword", "value"),
    [
        (D2Correction, "cutoff", 0.0),
        (D2Correction, "cutoff", np.inf),
        (D2Correction, "skin", -0.1),
        (D3Correction, "cutoff", np.nan),
        (D3Correction, "coordination_cutoff", -1.0),
        (D3Correction, "skin", np.inf),
    ],
)
def test_correction_configurations_reject_invalid_lengths(constructor, keyword, value):
    with pytest.raises(ValueError, match=keyword.replace("_", " ")):
        constructor(**{keyword: value})


def test_graph_requirement_validates_lengths_and_topology():
    with pytest.raises(ValueError, match="pair cutoff"):
        GraphRequirement(0.0, None, 0.5)
    with pytest.raises(ValueError, match="coordination cutoff"):
        GraphRequirement(10.0, np.inf, 0.5)
    with pytest.raises(ValueError, match="skin"):
        GraphRequirement(10.0, None, -0.5)
    with pytest.raises(ValueError, match="topology"):
        GraphRequirement(10.0, None, 0.5, topology="mesh")


def test_runtime_construction_normalizes_functional_without_mutating_config(
    monkeypatch,
):
    runtime = _FakeRuntime()
    seen = []
    _install_fake_factory(monkeypatch, runtime, seen=seen)
    configuration = D3Correction(functional="PBE")

    result = build_correction_runtime(configuration, native=object(), dtype=np.float64)

    assert result is runtime
    assert configuration.functional == "PBE"
    assert seen[0][0].functional == "pbe"


@pytest.mark.parametrize("configuration", [True, {"method": "d3"}, object()])
def test_pipeline_rejects_non_configuration_values(configuration):
    with pytest.raises(TypeError, match="correction configuration"):
        CorrectionPipeline([configuration], native=object(), dtype=np.float64)


def test_pipeline_rejects_duplicate_dispersion_corrections(monkeypatch):
    monkeypatch.setitem(
        corrections._RUNTIME_FACTORIES,
        D2Correction,
        lambda configuration, *, native, dtype: _FakeRuntime(),
    )
    _install_fake_factory(monkeypatch, _FakeRuntime())

    with pytest.raises(ValueError, match="duplicate dispersion"):
        CorrectionPipeline(
            [D2Correction(), D3Correction()], native=object(), dtype=np.float64
        )


def test_empty_correction_pipeline_is_inactive():
    assert not CorrectionPipeline([], native=object(), dtype=np.float64)


def test_native_factories_name_the_missing_evaluator():
    with pytest.raises(RuntimeError, match="D2Kokkos native evaluator"):
        build_correction_runtime(D2Correction(), native=object(), dtype=np.float64)
    with pytest.raises(RuntimeError, match="D3Kokkos native evaluator"):
        build_correction_runtime(D3Correction(), native=object(), dtype=np.float64)


@pytest.mark.parametrize(
    "configuration",
    [D2Correction(cutoff=4.0), D3Correction(cutoff=4.0, coordination_cutoff=3.5)],
    ids=("d2", "d3"),
)
@pytest.mark.parametrize(
    ("property_name", "compute_flags", "native_exports"),
    [
        ("energy", (False, False), ("energy",)),
        ("energies", (False, False), ("atomic_energies",)),
        ("forces", (True, False), ("forces",)),
        ("stress", (False, True), ("stress",)),
    ],
)
def test_native_runtime_requests_only_required_compute_and_exports(
    configuration, property_name, compute_flags, native_exports
):
    runtime = build_correction_runtime(
        configuration, native=_RecordingNative(), dtype=np.float64
    )
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.4, 0.2, 0.1]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )

    result = runtime.calculate(atoms, [property_name])

    assert set(result) == {property_name}
    assert runtime.evaluator.compute_calls == [compute_flags]
    assert runtime.evaluator.result_requests == [native_exports]


def test_pipeline_requests_only_needed_correction_properties(monkeypatch):
    runtime = _FakeRuntime()
    _install_fake_factory(monkeypatch, runtime)
    pipeline = CorrectionPipeline([D3Correction()], native=object(), dtype=np.float64)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    correction_results = pipeline.calculate(atoms, ["free_energy", "forces", "dipole"])

    assert runtime.requested_properties == [("energy", "forces")]
    assert correction_results["energy"] == pytest.approx(-0.5)
    np.testing.assert_allclose(correction_results["forces"], runtime.outputs["forces"])
    assert "dipole" not in correction_results


def test_pipeline_skips_native_runtime_without_requested_correction_property(
    monkeypatch,
):
    runtime = _FakeRuntime()
    _install_fake_factory(monkeypatch, runtime)
    pipeline = CorrectionPipeline([D3Correction()], native=object(), dtype=np.float64)

    correction_results = pipeline.calculate(Atoms("H"), ["polarization"])

    assert correction_results == {}
    assert runtime.requested_properties == []


def test_pipeline_adds_correction_results_to_requested_base_results(monkeypatch):
    runtime = _FakeRuntime()
    _install_fake_factory(monkeypatch, runtime)
    pipeline = CorrectionPipeline([D3Correction()], native=object(), dtype=np.float64)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    base = {
        "energy": 10.0,
        "free_energy": 10.0,
        "energies": np.array([4.0, 6.0]),
        "forces": np.zeros((2, 3)),
        "stress": np.zeros(6),
    }

    pipeline.add_results(atoms, base, ["free_energy", "energies", "forces", "stress"])

    assert base["energy"] == pytest.approx(9.5)
    assert base["free_energy"] == pytest.approx(9.5)
    np.testing.assert_allclose(base["energies"], [3.8, 5.7])
    np.testing.assert_allclose(base["forces"], runtime.outputs["forces"])
    np.testing.assert_allclose(base["stress"], runtime.outputs["stress"])
    assert pipeline.diagnostics() == ({"runtime": "fake", "calls": 1},)


@pytest.mark.parametrize(
    ("property_name", "bad_value", "message"),
    [
        ("energy", np.nan, "finite"),
        ("energies", np.array([-0.5]), r"\(2,\)"),
        ("energies", np.array([-0.2, np.inf]), "finite"),
        ("forces", np.zeros((3, 2)), r"\(2, 3\)"),
        ("forces", np.array([[0.0, 0.0, 0.0], [0.0, np.nan, 0.0]]), "finite"),
        ("stress", np.zeros((3, 3)), r"\(6,\)"),
        ("stress", np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.inf]), "finite"),
    ],
)
def test_pipeline_rejects_malformed_runtime_results(
    monkeypatch, property_name, bad_value, message
):
    runtime = _FakeRuntime(outputs={property_name: bad_value})
    _install_fake_factory(monkeypatch, runtime)
    pipeline = CorrectionPipeline([D3Correction()], native=object(), dtype=np.float64)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    with pytest.raises(ValueError, match=message):
        pipeline.calculate(atoms, [property_name])


def test_pipeline_rejects_missing_requested_runtime_result(monkeypatch):
    runtime = _FakeRuntime(outputs={"energy": -0.5})
    _install_fake_factory(monkeypatch, runtime)
    pipeline = CorrectionPipeline([D3Correction()], native=object(), dtype=np.float64)

    with pytest.raises(RuntimeError, match="forces"):
        pipeline.calculate(Atoms("H"), ["forces"])


def test_public_configuration_exports_are_on_the_stable_frontend():
    import symmetrix
    from symmetrix import D2Correction as ExportedD2Correction
    from symmetrix import D3Correction as ExportedD3Correction

    assert ExportedD2Correction is D2Correction
    assert ExportedD3Correction is D3Correction
    assert "D2Correction" in symmetrix.__all__
    assert "D3Correction" in symmetrix.__all__


@pytest.fixture(scope="module", autouse=True)
def _initialize_native_dispersion_backend():
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()


@pytest.mark.parametrize(
    "configuration",
    [
        D2Correction(cutoff=4.0, skin=0.6),
        D3Correction(
            damping="bj",
            cutoff=4.0,
            coordination_cutoff=3.5,
            skin=0.6,
        ),
    ],
    ids=("d2", "d3"),
)
def test_native_pair_correction_runtime_returns_only_requested_outputs(configuration):
    runtime = build_correction_runtime(
        configuration, native=native_symmetrix, dtype="float64"
    )
    atoms = Atoms(
        numbers=[6, 8],
        positions=[[0.0, 0.0, 0.0], [1.4, 0.2, 0.1]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )

    full = runtime.calculate(atoms, ["energy", "energies", "forces", "stress"])
    energy_only = runtime.calculate(atoms, ["energy"])

    assert set(full) == {"energy", "energies", "forces", "stress"}
    assert full["energies"].shape == (2,)
    assert full["forces"].shape == (2, 3)
    assert full["stress"].shape == (6,)
    assert sum(full["energies"]) == pytest.approx(full["energy"])
    assert set(energy_only) == {"energy"}


def test_native_correction_graph_reuses_skin_and_rebuilds_past_half_skin():
    runtime = build_correction_runtime(
        D2Correction(cutoff=3.0, skin=0.4),
        native=native_symmetrix,
        dtype="float64",
    )
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )

    runtime.calculate(atoms, ["energy"])
    atoms.positions[1, 1] += 0.1
    runtime.calculate(atoms, ["energy"])
    reused = runtime.diagnostics()
    atoms.positions[1, 1] += 0.11
    runtime.calculate(atoms, ["energy"])
    rebuilt = runtime.diagnostics()

    assert reused["graph_build_count"] == 1
    assert reused["graph_reuse_count"] == 1
    assert reused["geometry_update_count"] == 2
    assert rebuilt["graph_build_count"] == 2
    assert rebuilt["graph_reuse_count"] == 1


def test_native_correction_reuses_nonperiodic_molecule_without_cell():
    runtime = build_correction_runtime(
        D2Correction(cutoff=3.0, skin=0.4),
        native=native_symmetrix,
        dtype="float64",
    )
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])

    runtime.calculate(atoms, ["energy"])
    atoms.positions[1, 1] += 0.1
    runtime.calculate(atoms, ["energy"])

    diagnostics = runtime.diagnostics()
    assert diagnostics["graph_build_count"] == 1
    assert diagnostics["graph_reuse_count"] == 1


def test_native_correction_graph_reuses_safe_affine_cell_change(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    runtime = build_correction_runtime(
        D3Correction(cutoff=3.0, coordination_cutoff=2.5, skin=0.8),
        native=native_symmetrix,
        dtype="float64",
    )
    atoms = Atoms(
        "CO",
        scaled_positions=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.3]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    runtime.calculate(atoms, ["energy"])
    atoms.set_cell(np.diag([10.1, 10.0, 10.0]), scale_atoms=True)
    runtime.calculate(atoms, ["energy"])

    diagnostics = runtime.diagnostics()
    assert diagnostics["graph_build_count"] == 1
    assert diagnostics["graph_reuse_count"] == 1
    assert diagnostics["geometry_update_count"] == 2


def test_forced_eligible_kokkos_correction_graph_skips_host_builder(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(
        corrections,
        "build_host_neighbor_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("eligible forced Kokkos graph must remain device-side")
        ),
    )
    runtime = build_correction_runtime(
        D2Correction(cutoff=3.0, skin=0.5),
        native=native_symmetrix,
        dtype="float64",
    )
    atoms = Atoms(
        "CO",
        scaled_positions=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.3]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    runtime.calculate(atoms, ["energy"])

    diagnostics = runtime.diagnostics()
    assert diagnostics["graph_backend"] == "kokkos"
    assert diagnostics["graph_build_count"] == 1


def test_float32_native_runtime_selects_float_evaluator_and_promotes_results():
    runtime = build_correction_runtime(
        D2Correction(cutoff=3.0), native=native_symmetrix, dtype="float32"
    )
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])

    results = runtime.calculate(atoms, ["energies", "forces"])

    assert type(runtime.evaluator).__name__ == "D2KokkosFloat"
    assert results["energies"].dtype == np.float64
    assert results["forces"].dtype == np.float64


def test_native_correction_supports_empty_system():
    runtime = build_correction_runtime(
        D2Correction(cutoff=3.0), native=native_symmetrix, dtype="float64"
    )
    atoms = Atoms(cell=[5.0, 5.0, 5.0], pbc=False)

    results = runtime.calculate(atoms, ["energy", "energies", "forces"])

    assert results["energy"] == 0.0
    assert results["energies"].shape == (0,)
    assert results["forces"].shape == (0, 3)


def test_partial_periodic_native_correction_uses_host_graph():
    runtime = build_correction_runtime(
        D3Correction(cutoff=3.0, coordination_cutoff=2.5),
        native=native_symmetrix,
        dtype="float64",
    )
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[5.0, 5.0, 8.0],
        pbc=[True, True, False],
    )

    runtime.calculate(atoms, ["energy"])

    assert runtime.diagnostics()["graph_backend"] == "host"


# ---------------------------------------------------------------------------
# FourierD3 (reciprocal-space) configuration and runtime registration.
#
# The configuration is public; its factory, runtime, and native evaluators remain
# internal implementation seams.
# ---------------------------------------------------------------------------


def test_fourier_d3_configuration_defaults_resolve_to_the_spacing_policy():
    configuration = FourierD3Correction()

    assert configuration.functional == "pbe"
    assert configuration.coordination_cutoff == pytest.approx(40.0 * units.Bohr)
    assert configuration.skin == pytest.approx(0.5)
    assert configuration.mesh is None
    # Naming no mesh policy resolves to the automatic spacing policy, so a constructed
    # configuration always carries exactly one effective policy.
    assert configuration.mesh_spacing == pytest.approx(1.0)
    assert configuration.spline_order == 4
    assert configuration.decomposition_tolerance == pytest.approx(1e-4)
    assert configuration.rank_cap is None
    assert configuration.chunk_size == 512
    with pytest.raises(FrozenInstanceError):
        configuration.spline_order = 2


def test_fourier_d3_configuration_stays_frozen_and_hashable_after_resolution():
    default = FourierD3Correction()
    explicit = FourierD3Correction(mesh=(16, 16, 16))

    assert hash(default) == hash(FourierD3Correction())
    assert len({default, FourierD3Correction(), explicit}) == 2
    with pytest.raises(FrozenInstanceError):
        default.mesh_spacing = 2.0

    replaced = replace(default, functional="PBE")

    assert replaced.functional == "PBE"
    assert replaced.mesh_spacing == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("functional", "", "functional"),
        ("functional", "   ", "functional"),
        ("coordination_cutoff", 0.0, "coordination cutoff"),
        ("coordination_cutoff", -1.0, "coordination cutoff"),
        ("coordination_cutoff", np.inf, "coordination cutoff"),
        ("coordination_cutoff", np.nan, "coordination cutoff"),
        ("skin", -0.1, "skin"),
        ("skin", np.inf, "skin"),
        ("spline_order", 1, "spline order"),
        ("spline_order", 7, "spline order"),
        ("spline_order", 4.0, "spline order"),
        ("spline_order", True, "spline order"),
        ("spline_order", "4", "spline order"),
        ("decomposition_tolerance", 0.0, "decomposition tolerance"),
        ("decomposition_tolerance", -1e-4, "decomposition tolerance"),
        ("decomposition_tolerance", np.inf, "decomposition tolerance"),
        ("decomposition_tolerance", np.nan, "decomposition tolerance"),
        ("rank_cap", 0, "rank cap"),
        ("rank_cap", -3, "rank cap"),
        ("rank_cap", 2.5, "rank cap"),
        ("rank_cap", True, "rank cap"),
        ("chunk_size", 0, "chunk size"),
        ("chunk_size", -1, "chunk size"),
        ("chunk_size", 2.5, "chunk size"),
        ("chunk_size", True, "chunk size"),
        ("mesh_spacing", 0.0, "mesh spacing"),
        ("mesh_spacing", -1.0, "mesh spacing"),
        ("mesh_spacing", np.inf, "mesh spacing"),
        ("mesh_spacing", np.nan, "mesh spacing"),
    ],
)
def test_fourier_d3_configuration_rejects_invalid_values(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        FourierD3Correction(**{keyword: value})


@pytest.mark.parametrize(
    "mesh",
    [
        (16, 16),
        (16, 16, 16, 16),
        (16, 16, 0),
        (16, 16, -8),
        (16, 16.0, 16),
        (16, True, 16),
        (16, "16", 16),
        [16, 16, 16],
        "16,16,16",
        16,
    ],
)
def test_fourier_d3_configuration_rejects_invalid_mesh_shape(mesh):
    with pytest.raises(
        ValueError, match="mesh must be a tuple of three positive integers"
    ):
        FourierD3Correction(mesh=mesh)


@pytest.mark.parametrize(
    "mesh",
    (
        (1 << 63, 1, 1),
        (1 << 32, 1 << 32, 1),
    ),
)
def test_fourier_d3_configuration_rejects_unaddressable_explicit_mesh(mesh):
    with pytest.raises(ValueError, match="64-bit-safe|beyond the supported"):
        FourierD3Correction(mesh=mesh)


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_fourier_d3_configuration_accepts_every_supported_spline_order(order):
    assert FourierD3Correction(spline_order=order).spline_order == order


@pytest.mark.parametrize("rank_cap", [1, 7])
def test_fourier_d3_configuration_accepts_positive_rank_cap(rank_cap):
    assert FourierD3Correction(rank_cap=rank_cap).rank_cap == rank_cap


def test_fourier_d3_configuration_accepts_each_mesh_policy_alone():
    explicit = FourierD3Correction(mesh=(24, 24, 30))
    spaced = FourierD3Correction(mesh_spacing=0.6)

    assert explicit.mesh == (24, 24, 30)
    # An explicit mesh is the effective policy, so no spacing is invented alongside it.
    assert explicit.mesh_spacing is None
    assert spaced.mesh is None
    assert spaced.mesh_spacing == pytest.approx(0.6)


def test_fourier_d3_configuration_rejects_both_mesh_policies():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FourierD3Correction(mesh=(16, 16, 16), mesh_spacing=1.0)


def test_fourier_d3_runtime_builder_normalizes_functional_without_mutating_config(
    monkeypatch,
):
    runtime = _FakeRuntime()
    seen = []
    monkeypatch.setitem(
        corrections._RUNTIME_FACTORIES,
        FourierD3Correction,
        lambda configuration, *, native, dtype: seen.append(configuration) or runtime,
    )
    configuration = FourierD3Correction(functional=" PBE ")

    result = build_correction_runtime(configuration, native=object(), dtype=np.float64)

    assert result is runtime
    assert configuration.functional == " PBE "
    assert seen[0].functional == "pbe"


def test_fourier_d3_factory_registration_is_distinct_from_d3(monkeypatch):
    sentinel = _FakeRuntime()
    fourier_calls = []
    monkeypatch.setitem(
        corrections._RUNTIME_FACTORIES,
        FourierD3Correction,
        lambda configuration, *, native, dtype: fourier_calls.append(configuration)
        or sentinel,
    )

    fourier_runtime = build_correction_runtime(
        FourierD3Correction(), native=object(), dtype=np.float64
    )
    d3_runtime = build_correction_runtime(
        D3Correction(cutoff=4.0), native=_RecordingNative(), dtype=np.float64
    )

    assert fourier_runtime is sentinel
    assert len(fourier_calls) == 1
    assert d3_runtime is not sentinel
    assert type(d3_runtime.evaluator) is _RecordingNativeEvaluator


@pytest.mark.parametrize(
    ("dtype", "class_name"),
    [("float64", "FourierD3Kokkos"), ("float32", "FourierD3KokkosFloat")],
)
def test_fourier_d3_runtime_resolves_the_installed_native_evaluator(dtype, class_name):
    runtime = build_correction_runtime(
        FourierD3Correction(), native=native_symmetrix, dtype=dtype
    )

    # Task 11 publishes the dtype-specific evaluator, so the runtime construction now
    # succeeds and no real-space D3 evaluator stands in for it.
    assert runtime.evaluator is None
    assert runtime.diagnostics()["native_allocation_count"] == 0
    assert runtime._evaluator_class is getattr(native_symmetrix, class_name)


@pytest.mark.parametrize(
    ("dtype", "class_name"),
    [("float64", "FourierD3Kokkos"), ("float32", "FourierD3KokkosFloat")],
)
def test_fourier_d3_runtime_fails_fast_without_a_native_evaluator(dtype, class_name):
    class _WithoutEvaluator:
        """A build whose capability resolved but whose evaluator is absent."""

        @staticmethod
        def _fourier_d3_capability():
            return {"available": True, "provider": "fftw3"}

        @staticmethod
        def _kokkos_default_execution_space():
            return "Serial"

    with pytest.raises(RuntimeError) as error:
        build_correction_runtime(
            FourierD3Correction(), native=_WithoutEvaluator, dtype=dtype
        )

    message = str(error.value)
    assert class_name in message
    assert "not available in this build" in message
    # A real-space D3 evaluator is a different model and is never substituted.
    assert "real-space" in message


def test_fourier_d3_runtime_never_falls_back_to_real_space_d3():
    # A native module exposing only the real-space D3 evaluator must not be enough.
    with pytest.raises(RuntimeError, match="FourierD3"):
        build_correction_runtime(
            FourierD3Correction(), native=_RecordingNative(), dtype=np.float64
        )


def test_fourier_d3_pipeline_accepts_a_single_configuration(monkeypatch):
    runtime = _FakeRuntime()
    monkeypatch.setitem(
        corrections._RUNTIME_FACTORIES,
        FourierD3Correction,
        lambda configuration, *, native, dtype: runtime,
    )
    pipeline = CorrectionPipeline(
        [FourierD3Correction(mesh_spacing=0.8)], native=object(), dtype=np.float64
    )
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    assert pipeline.calculate(atoms, ["energy"])["energy"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "configurations",
    [
        (D2Correction(), D3Correction()),
        (D2Correction(), FourierD3Correction()),
        (D3Correction(), FourierD3Correction()),
        (FourierD3Correction(), FourierD3Correction()),
    ],
    ids=("d2+d3", "d2+fourier", "d3+fourier", "fourier+fourier"),
)
def test_pipeline_rejects_duplicate_dispersion_of_any_mix(configurations):
    with pytest.raises(ValueError, match="duplicate dispersion"):
        CorrectionPipeline(list(configurations), native=object(), dtype=np.float64)


def test_fourier_d3_configuration_is_the_only_public_reciprocal_surface():
    import symmetrix

    assert "FourierD3Correction" in corrections.__all__
    assert "FourierD3Correction" in symmetrix.__all__
    assert "FourierD3Correction" in symmetrix._FRONTEND_EXPORTS
    assert symmetrix.FourierD3Correction is FourierD3Correction
    assert "FourierD3Kokkos" not in symmetrix.__all__
    assert "FourierD3KokkosFloat" not in symmetrix.__all__


# ---------------------------------------------------------------------------
# FourierD3 (reciprocal-space) runtime.
#
# The runtime owns its own short coordination graph and its five independent invalidation
# domains. FFT planning, mesh assignment, native storage, and the numeric kernels belong
# to later tasks, so the runtime is exercised against a recording stand-in for the native
# evaluator binding: the fake records every construction and geometry call, which is what
# makes "no allocation before validation" observable.
# ---------------------------------------------------------------------------


class _FakeReciprocalNative:
    """Recording stand-in for the native FourierD3 evaluator binding.

    The stand-in mirrors the resolved-fields constructor the real seam takes (the
    frozen configuration is unpacked by the runtime, never handed to the native side)
    and records the mesh and decomposition payloads the runtime pushes.
    """

    def __init__(self, *, device_prepare_error=None):
        self.instances = []
        self.host_graphs = []
        self.device_graphs = []
        # The canonical tables are inputs rather than behaviour, so the stand-in
        # publishes the installed ones: the runtime decomposes them exactly as it does
        # with the real evaluator, and only the evaluator itself is faked.
        self.tables = dict(
            native_symmetrix.load_backend()._fourier_d3_canonical_tables()
        )
        self._device_prepare_error = device_prepare_error
        native = self

        class _Evaluator:
            def __init__(
                self,
                functional,
                coordination_cutoff,
                spline_order,
                rank_chunk,
                table_fingerprint,
                radii,
                q_table,
            ):
                self.functional = functional
                self.coordination_cutoff = coordination_cutoff
                self.spline_order = spline_order
                self.rank_chunk = rank_chunk
                self.table_fingerprint = table_fingerprint
                self.radii = np.array(radii, copy=True)
                self.q_table = np.array(q_table, copy=True)
                self.num_nodes = 0
                self.compute_calls = []
                self.result_requests = []
                self.geometry_update_count = 0
                self.graph_backend = None
                self.edge_count = 0
                self.meshes = []
                self.decompositions = []
                self.precision = (
                    "float32"
                    if type(self).__name__ == "FourierD3KokkosFloat"
                    else "float64"
                )
                self.build_cutoff = 0.0
                self.cell = None
                self.positions = None
                self.generation = 0
                # The real seam stages a mesh and commits it with the preparation it
                # belongs to, so a refused preparation leaves the committed mesh - and
                # every counter that describes it - untouched.
                self._pending_mesh = None
                self.mesh_invalidation_count = 0
                self.input_generation = 0
                self.published_generation = 0
                self._timing_enabled = False
                self._stage_timing = None
                native.instances.append(self)

            def set_mesh(self, mesh):
                resolved = tuple(int(size) for size in mesh)
                if self.meshes and self.meshes[-1] == resolved:
                    return
                self._pending_mesh = resolved

            def _commit_mesh(self):
                if self._pending_mesh is None:
                    return
                self.meshes.append(self._pending_mesh)
                self._pending_mesh = None
                self.mesh_invalidation_count += 1
                self.input_generation += 1

            def set_decomposition(
                self,
                species,
                eigenvalues,
                v_q,
                cn_ref,
                valid,
                max_relative_error,
                tolerance_met,
            ):
                self.decompositions.append(
                    {
                        "species": np.array(species, copy=True),
                        "eigenvalues": np.array(eigenvalues, copy=True),
                        "v_q": np.array(v_q, copy=True),
                        "cn_ref": np.array(cn_ref, copy=True),
                        "valid": np.array(valid, copy=True),
                        "max_relative_error": float(max_relative_error),
                        "tolerance_met": bool(tolerance_met),
                    }
                )

            def prepare_host_graph(
                self,
                numbers,
                receiver_offsets,
                sources,
                shifts,
                positions,
                cell,
                build_cutoff,
            ):
                self.num_nodes = len(numbers)
                self.graph_backend = "host"
                self.edge_count = int(len(sources))
                self.build_cutoff = float(build_cutoff)
                self.cell = np.array(cell, copy=True)
                self.positions = np.array(positions, copy=True)
                self.generation += 1
                self._commit_mesh()
                self.input_generation += 1
                native.host_graphs.append(
                    {
                        "numbers": np.array(numbers, copy=True),
                        "receiver_offsets": np.array(receiver_offsets, copy=True),
                        "sources": np.array(sources, copy=True),
                        "shifts": np.array(shifts, copy=True),
                        "positions": np.array(positions, copy=True),
                        "cell": np.array(cell, copy=True),
                        "build_cutoff": float(build_cutoff),
                    }
                )

            def prepare_periodic_graph(
                self, numbers, positions, cell, inverse_cell, build_cutoff
            ):
                if native._device_prepare_error is not None:
                    raise native._device_prepare_error
                self.num_nodes = len(numbers)
                self.graph_backend = "kokkos"
                self.build_cutoff = float(build_cutoff)
                self.cell = np.array(cell, copy=True)
                self.positions = np.array(positions, copy=True)
                self.generation += 1
                self._commit_mesh()
                self.input_generation += 1
                native.device_graphs.append(
                    {
                        "numbers": np.array(numbers, copy=True),
                        "positions": np.array(positions, copy=True),
                        "cell": np.array(cell, copy=True),
                        "inverse_cell": np.array(inverse_cell, copy=True),
                        "build_cutoff": float(build_cutoff),
                    }
                )

            def update_geometry(self, positions, cell, inverse_cell):
                # The real seam returns early when nothing moved and nothing is staged,
                # so an unchanged reuse is not an input update.
                if (
                    self._pending_mesh is None
                    and self.positions is not None
                    and np.array_equal(self.positions, positions)
                    and np.array_equal(self.cell, cell)
                ):
                    return
                self.geometry_update_count += 1
                self.positions = np.array(positions, copy=True)
                self.cell = np.array(cell, copy=True)
                self.generation += 1
                self._commit_mesh()
                self.input_generation += 1

            def compute(self, compute_forces, compute_stress):
                self.compute_calls.append((compute_forces, compute_stress))
                self._commit_mesh()
                self.published_generation = self.input_generation
                if self._timing_enabled:
                    self._stage_timing = {
                        name: {
                            "last_seconds": 1e-6,
                            "cumulative_seconds": 1e-6,
                            "count": 1,
                        }
                        for name in _STAGE_NAMES
                    }

            def set_timing_enabled(self, enabled):
                self._timing_enabled = bool(enabled)

            def timing_enabled(self):
                return self._timing_enabled

            def timing(self):
                return None if not self._timing_enabled else dict(self._stage_timing)

            def result(self, properties=None):
                requested = None if properties is None else tuple(properties)
                self.result_requests.append(requested)
                nodes = self.num_nodes
                outputs = {
                    "energy": -0.5,
                    "atomic_energies": (
                        np.full(nodes, -0.5 / nodes) if nodes else np.zeros(0)
                    ),
                    "forces": np.zeros((nodes, 3)),
                    "stress": np.zeros(6),
                }
                if requested is None:
                    return outputs
                return {name: outputs[name] for name in requested}

            def graph_diagnostics(self):
                return {
                    "graph_backend": self.graph_backend,
                    "generation": self.generation,
                    "geometry_update_count": self.geometry_update_count,
                    "num_nodes": self.num_nodes,
                    "num_edges": self.edge_count,
                    "build_cutoff": self.build_cutoff,
                }

            def diagnostics(self):
                """The evaluator state the real seam reports, from what was recorded.

                The stand-in has no FFT plans, kernels, or workspaces, so those totals
                are zero and their itemizations empty; everything derived from the
                pushed payloads - species, rank, mesh, precision, generations - is the
                real value the runtime handed over.
                """

                payload = self.decompositions[-1] if self.decompositions else None
                mesh = self.meshes[-1] if self.meshes else (0, 0, 0)
                spacing = (0.0, 0.0, 0.0)
                if self.cell is not None:
                    lengths = np.linalg.norm(np.asarray(self.cell), axis=1)
                    spacing = tuple(
                        float(length / size) if size else 0.0
                        for length, size in zip(lengths, mesh)
                    )
                return {
                    "precision": self.precision,
                    "active_species": (
                        [] if payload is None else [int(v) for v in payload["species"]]
                    ),
                    "table_fingerprint": self.table_fingerprint,
                    "retained_rank": 0
                    if payload is None
                    else int(len(payload["eigenvalues"])),
                    "rank_total": 0
                    if payload is None
                    else int(payload["v_q"].shape[0]),
                    "rank_chunk": min(self.rank_chunk, int(len(payload["eigenvalues"])))
                    if payload is not None
                    else 0,
                    "rank_chunk_count": 0,
                    "n_references": 0
                    if payload is None
                    else int(payload["cn_ref"].shape[0]),
                    "decomposition_max_relative_error": (
                        0.0 if payload is None else payload["max_relative_error"]
                    ),
                    "decomposition_tolerance_met": (
                        False if payload is None else payload["tolerance_met"]
                    ),
                    "mesh": list(mesh),
                    "mesh_points": int(np.prod(mesh)),
                    "mesh_spacing": list(spacing),
                    "mesh_invalidation_count": self.mesh_invalidation_count,
                    "active_bytes": 0,
                    "capacity_bytes": 0,
                    "memory_components": [],
                    "plan_retained_bytes": 0,
                    "fft_plans": [],
                    "reservations": 0,
                    "allocations": 0,
                    "plan_builds": 0,
                    "plan_reuses": 0,
                    "plan_releases": 0,
                    "plan_key_rebuilds": 0,
                    "kernel_key_hits": 0,
                    "kernel_key_misses": 0,
                    "kernel_key_rebuilds": 0,
                    "stencil_points": 0,
                    "input_generation": self.input_generation,
                    "published_generation": self.published_generation,
                    "computed": self.published_generation > 0,
                    "result_available": self.published_generation
                    == self.input_generation
                    and self.published_generation > 0,
                    "forces_available": self.published_generation > 0,
                    "stress_available": self.published_generation > 0,
                    "engine_loaded": self.published_generation > 0,
                    "timing_enabled": self._timing_enabled,
                    "stage_timing": self.timing(),
                }

        class FourierD3Kokkos(_Evaluator):
            pass

        class FourierD3KokkosFloat(_Evaluator):
            pass

        self.FourierD3Kokkos = FourierD3Kokkos
        self.FourierD3KokkosFloat = FourierD3KokkosFloat

    def _fourier_d3_canonical_tables(self):
        return self.tables

    @staticmethod
    def _kokkos_default_execution_space():
        return "Serial"


def _fourier_configuration(**overrides):
    arguments = {"coordination_cutoff": 2.5, "skin": 0.4, "mesh": (8, 8, 8)}
    arguments.update(overrides)
    return FourierD3Correction(**arguments)


def _build_fourier_runtime(configuration=None, *, dtype="float64", native=None):
    native = _FakeReciprocalNative() if native is None else native
    runtime = build_correction_runtime(
        _fourier_configuration() if configuration is None else configuration,
        native=native,
        dtype=dtype,
    )
    return runtime, native


def _periodic_pair():
    return Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )


def test_fourier_d3_allocates_the_native_evaluator_lazily_on_first_calculate():
    runtime, native = _build_fourier_runtime()
    atoms = _periodic_pair()

    assert native.instances == []
    assert runtime.evaluator is None
    assert runtime.diagnostics()["native_allocation_count"] == 0

    runtime.calculate(atoms, ["energy"])
    runtime.calculate(atoms, ["energy"])

    assert len(native.instances) == 1
    assert runtime.diagnostics()["native_allocation_count"] == 1
    # The evaluator is constructed from the frozen configuration's resolved fields and
    # the canonical tables, and the runtime pushes the resolved mesh and the
    # active-species decomposition it computed.
    evaluator = runtime.evaluator
    assert evaluator.functional == runtime.configuration.functional
    assert evaluator.coordination_cutoff == pytest.approx(
        runtime.configuration.coordination_cutoff
    )
    assert evaluator.spline_order == runtime.configuration.spline_order
    assert evaluator.rank_chunk == runtime.configuration.chunk_size
    assert evaluator.table_fingerprint == native.tables["table_fingerprint"]
    # Both calculations resolve the same mesh and the same active species; the native
    # side compares an identical payload and skips it, so the push is idempotent.
    assert set(evaluator.meshes) == {(8, 8, 8)}
    assert {
        tuple(payload["species"].tolist()) for payload in evaluator.decompositions
    } == {(6, 8)}


@pytest.mark.parametrize(
    ("cell", "pbc", "message"),
    [
        ([8.0, 8.0, 8.0], [True, True, False], "fully periodic"),
        ([8.0, 8.0, 8.0], [True, False, False], "fully periodic"),
        ([8.0, 8.0, 8.0], False, "fully periodic"),
        ([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]], True, "nonsingular"),
        ([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]], True, "nonsingular"),
        ([[8.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, 8.0]], True, "finite"),
        ([[8.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 8.0]], True, "finite"),
    ],
    ids=("slab", "wire", "molecule", "singular", "rank-deficient", "infinite", "nan"),
)
def test_fourier_d3_rejects_unusable_geometry_before_native_allocation(
    cell, pbc, message
):
    runtime, native = _build_fourier_runtime()
    atoms = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=cell, pbc=pbc
    )

    with pytest.raises(RuntimeError, match=message) as error:
        runtime.calculate(atoms, ["energy"])

    assert "FourierD3 requires" in str(error.value)
    assert native.instances == []
    assert runtime.diagnostics()["native_allocation_count"] == 0


def test_fourier_d3_rejects_an_empty_system_without_a_periodic_cell():
    runtime, native = _build_fourier_runtime()

    with pytest.raises(RuntimeError, match="fully periodic"):
        runtime.calculate(Atoms(), ["energy"])

    assert native.instances == []


def test_fourier_d3_supports_an_empty_fully_periodic_system(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    runtime, native = _build_fourier_runtime()
    atoms = Atoms(cell=[8.0, 8.0, 8.0], pbc=True)

    results = runtime.calculate(atoms, ["energy", "energies", "forces"])

    assert results["energy"] == pytest.approx(-0.5)
    assert results["energies"].shape == (0,)
    assert results["forces"].shape == (0, 3)
    assert len(native.instances) == 1
    assert runtime.decomposition_key == ((), 1e-4, None)


@pytest.mark.parametrize(
    ("dtype", "class_name"),
    [("float64", "FourierD3Kokkos"), ("float32", "FourierD3KokkosFloat")],
)
def test_fourier_d3_selects_the_dtype_specific_evaluator_class(dtype, class_name):
    runtime, native = _build_fourier_runtime(dtype=dtype)

    runtime.calculate(_periodic_pair(), ["energy"])

    assert type(runtime.evaluator).__name__ == class_name
    assert len(native.instances) == 1


def test_fourier_d3_runtime_returns_only_requested_outputs(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    runtime, _ = _build_fourier_runtime()

    full = runtime.calculate(
        _periodic_pair(), ["energy", "energies", "forces", "stress"]
    )
    energy_only = runtime.calculate(_periodic_pair(), ["energy"])

    assert set(full) == {"energy", "energies", "forces", "stress"}
    assert full["energies"].shape == (2,)
    assert full["forces"].shape == (2, 3)
    assert full["stress"].shape == (6,)
    assert sum(full["energies"]) == pytest.approx(full["energy"])
    assert set(energy_only) == {"energy"}
    assert runtime.evaluator.result_requests == [
        ("energy", "atomic_energies", "forces", "stress"),
        ("energy",),
    ]


def test_fourier_d3_short_graph_uses_only_the_coordination_cutoff(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    runtime, native = _build_fourier_runtime()
    atoms = _periodic_pair()

    runtime.calculate(atoms, ["energy"])

    requirement = runtime.graph_requirement
    # A global reciprocal interaction has no real-space pair radius to misuse.
    assert requirement.pair_cutoff is None
    assert requirement.topology == "global"
    assert requirement.coordination_cutoff == pytest.approx(2.5)
    assert requirement.graph_cutoff == pytest.approx(2.5)
    # The graph covers the coordination cutoff plus the skin, and nothing wider.
    graph = native.host_graphs[0]
    assert graph["build_cutoff"] == pytest.approx(2.9)
    assert runtime.diagnostics()["num_edges"] == len(graph["sources"])
    # It is the repository's own graph for these atoms: same receiver-major edges.
    expected = corrections.build_host_neighbor_graph(atoms, 2.9)
    assert np.array_equal(graph["sources"], expected.sources)
    assert np.array_equal(graph["shifts"], expected.shifts)
    assert np.array_equal(graph["numbers"], atoms.numbers.astype(np.int64))
    assert np.array_equal(
        graph["receiver_offsets"],
        np.concatenate([[0], np.cumsum(expected.num_neigh)]),
    )


def test_fourier_d3_graph_reuses_below_half_skin_and_rebuilds_past_it(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    runtime, _ = _build_fourier_runtime()
    atoms = _periodic_pair()

    runtime.calculate(atoms, ["energy"])
    # Exactly half the skin is still covered: the reuse predicate is inclusive.
    atoms.positions[1, 1] += 0.2
    runtime.calculate(atoms, ["energy"])
    reused = runtime.diagnostics()
    atoms.positions[1, 1] += 0.0005
    runtime.calculate(atoms, ["energy"])
    rebuilt = runtime.diagnostics()

    assert reused["graph_build_count"] == 1
    assert reused["graph_reuse_count"] == 1
    assert reused["geometry_update_count"] == 2
    assert rebuilt["graph_build_count"] == 2
    assert rebuilt["graph_reuse_count"] == 1
    assert rebuilt["geometry_update_count"] == 3


def test_fourier_d3_device_graph_follows_the_backend_policy_without_fallback(
    monkeypatch,
):
    def fail_host_builder(*_args, **_kwargs):
        raise AssertionError("device-eligible reciprocal graph used the host builder")

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(corrections, "build_host_neighbor_graph", fail_host_builder)
    runtime, native = _build_fourier_runtime()

    runtime.calculate(_periodic_pair(), ["energy"])

    diagnostics = runtime.diagnostics()
    assert diagnostics["graph_backend"] == "kokkos"
    assert diagnostics["graph_build_count"] == 1
    assert diagnostics["backend_fallback_count"] == 0
    assert native.host_graphs == []
    assert native.device_graphs[0]["build_cutoff"] == pytest.approx(2.9)


def test_fourier_d3_device_prepare_failure_does_not_fall_back(monkeypatch):
    def fail_host_builder(*_args, **_kwargs):
        raise AssertionError("device-eligible reciprocal graph used the host builder")

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(corrections, "build_host_neighbor_graph", fail_host_builder)
    native = _FakeReciprocalNative(
        device_prepare_error=RuntimeError("device build failed")
    )
    runtime, _ = _build_fourier_runtime(native=native)

    with pytest.raises(RuntimeError, match="device build failed"):
        runtime.calculate(_periodic_pair(), ["energy"])

    assert runtime.diagnostics()["backend_fallback_count"] == 0


def test_graph_requirement_global_topology_declares_no_pair_cutoff():
    requirement = GraphRequirement(
        pair_cutoff=None, coordination_cutoff=4.0, skin=0.5, topology="global"
    )

    assert requirement.graph_cutoff == pytest.approx(4.0)

    with pytest.raises(ValueError, match="pair cutoff"):
        GraphRequirement(None, 4.0, 0.5)
    with pytest.raises(ValueError, match="pair cutoff"):
        GraphRequirement(5.0, 4.0, 0.5, topology="global")
    with pytest.raises(ValueError, match="coordination cutoff"):
        GraphRequirement(None, None, 0.5, topology="global")


def test_graph_requirement_graph_cutoff_is_the_widest_declared_radius():
    assert GraphRequirement(12.0, 8.0, 0.5).graph_cutoff == pytest.approx(12.0)
    assert GraphRequirement(12.0, None, 0.5).graph_cutoff == pytest.approx(12.0)
    assert GraphRequirement(3.0, 4.0, 0.5).graph_cutoff == pytest.approx(4.0)


def test_fourier_d3_decomposition_key_follows_the_active_species():
    runtime, _ = _build_fourier_runtime()
    atoms = _periodic_pair()

    runtime.calculate(atoms, ["energy"])
    assert runtime.decomposition_key == ((6, 8), 1e-4, None)
    assert runtime.diagnostics()["decomposition_invalidation_count"] == 1

    # Motion inside the skin changes no species.
    atoms.positions[1, 1] += 0.1
    runtime.calculate(atoms, ["energy"])
    assert runtime.decomposition_key == ((6, 8), 1e-4, None)
    assert runtime.diagnostics()["decomposition_invalidation_count"] == 1

    runtime.calculate(
        Atoms(
            "CON",
            positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [3.0, 3.0, 3.0]],
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        ),
        ["energy"],
    )
    assert runtime.decomposition_key == ((6, 7, 8), 1e-4, None)
    assert runtime.diagnostics()["decomposition_invalidation_count"] == 2

    runtime.calculate(
        Atoms(
            "CN",
            positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        ),
        ["energy"],
    )
    assert runtime.decomposition_key == ((6, 7), 1e-4, None)
    assert runtime.diagnostics()["decomposition_invalidation_count"] == 3


def _scaled_pair(cell):
    return Atoms(
        "CO",
        scaled_positions=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.3]],
        cell=cell,
        pbc=True,
    )


def test_fourier_d3_spacing_mesh_invalidates_only_when_dimensions_change():
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=1.0, skin=0.8)
    )
    atoms = _scaled_pair([10.0, 10.0, 10.0])

    runtime.calculate(atoms, ["energy"])
    assert runtime.resolved_mesh == (10, 10, 10)
    assert runtime.diagnostics()["mesh_invalidation_count"] == 1

    # 9.5 Angstrom still needs ten nodes per axis, so the mesh is unchanged even though
    # the cell, and therefore the reciprocal kernel, changed.
    atoms.set_cell(np.diag([9.5, 10.0, 10.0]), scale_atoms=True)
    runtime.calculate(atoms, ["energy"])
    assert runtime.resolved_mesh == (10, 10, 10)
    diagnostics = runtime.diagnostics()
    assert diagnostics["mesh_invalidation_count"] == 1
    assert diagnostics["reciprocal_kernel_invalidation_count"] == 2

    # 11.0 Angstrom needs eleven nodes, which rounds up to the radix-friendly twelve.
    atoms.set_cell(np.diag([11.0, 10.0, 10.0]), scale_atoms=True)
    runtime.calculate(atoms, ["energy"])
    assert runtime.resolved_mesh == (12, 10, 10)
    assert runtime.diagnostics()["mesh_invalidation_count"] == 2


@pytest.mark.parametrize(
    ("cell", "spacing", "spline_order", "expected"),
    [
        ([10.0, 10.0, 10.0], 1.0, 4, (10, 10, 10)),
        ([11.0, 10.0, 10.0], 1.0, 4, (12, 10, 10)),
        ([10.0, 10.0, 10.0], 2.0, 4, (5, 5, 5)),
        ([10.0, 10.0, 21.0], 1.0, 4, (10, 10, 21)),
        ([3.0, 3.0, 3.0], 1.0, 4, (4, 4, 4)),
        ([3.0, 3.0, 3.0], 1.0, 6, (6, 6, 6)),
    ],
    ids=("exact", "round-up", "coarse", "mixed", "spline-floor-4", "spline-floor-6"),
)
def test_fourier_d3_spacing_resolves_radix_friendly_dimensions(
    cell, spacing, spline_order, expected
):
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(
            mesh=None, mesh_spacing=spacing, spline_order=spline_order
        )
    )

    runtime.calculate(_scaled_pair(cell), ["energy"])

    assert runtime.resolved_mesh == expected


def test_fourier_d3_explicit_mesh_stays_fixed_across_cell_changes():
    runtime, _ = _build_fourier_runtime(_fourier_configuration(skin=0.8))
    atoms = _scaled_pair([10.0, 10.0, 10.0])

    runtime.calculate(atoms, ["energy"])
    assert runtime.resolved_mesh == (8, 8, 8)
    atoms.set_cell(np.diag([11.0, 10.0, 10.0]), scale_atoms=True)
    runtime.calculate(atoms, ["energy"])

    diagnostics = runtime.diagnostics()
    assert runtime.resolved_mesh == (8, 8, 8)
    assert diagnostics["mesh_invalidation_count"] == 1
    assert diagnostics["reciprocal_kernel_invalidation_count"] == 2


def test_fourier_d3_fft_plan_key_follows_its_own_mesh_key():
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=1.0, skin=0.8)
    )
    atoms = _scaled_pair([10.0, 10.0, 10.0])

    runtime.calculate(atoms, ["energy"])
    assert runtime.fft_plan_key == ((10, 10, 10), "float64")
    assert runtime.diagnostics()["fft_plan_invalidation_count"] == 1

    # A cell change with an unchanged resolved mesh reuses the plan.
    atoms.set_cell(np.diag([9.5, 10.0, 10.0]), scale_atoms=True)
    runtime.calculate(atoms, ["energy"])
    assert runtime.fft_plan_key == ((10, 10, 10), "float64")
    assert runtime.diagnostics()["fft_plan_invalidation_count"] == 1
    assert runtime.diagnostics()["reciprocal_kernel_invalidation_count"] == 2

    # A resolved mesh change invalidates it.
    atoms.set_cell(np.diag([11.0, 10.0, 10.0]), scale_atoms=True)
    runtime.calculate(atoms, ["energy"])
    assert runtime.fft_plan_key == ((12, 10, 10), "float64")
    assert runtime.diagnostics()["fft_plan_invalidation_count"] == 2


def test_fourier_d3_graph_rebuild_does_not_invalidate_the_fft_plan():
    runtime, _ = _build_fourier_runtime(_fourier_configuration(skin=0.4))
    atoms = _periodic_pair()

    runtime.calculate(atoms, ["energy"])
    atoms.positions[1, 1] += 0.5
    runtime.calculate(atoms, ["energy"])

    diagnostics = runtime.diagnostics()
    assert diagnostics["graph_build_count"] == 2
    assert diagnostics["fft_plan_invalidation_count"] == 1
    assert diagnostics["mesh_invalidation_count"] == 1


def test_fourier_d3_reuses_the_short_graph_across_a_safe_affine_cell_change(
    monkeypatch,
):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    runtime, _ = _build_fourier_runtime(_fourier_configuration(skin=0.8))
    atoms = _scaled_pair([8.0, 8.0, 8.0])

    runtime.calculate(atoms, ["energy"])
    kernel_key = runtime.reciprocal_kernel_key
    atoms.set_cell(np.diag([8.1, 8.0, 8.0]), scale_atoms=True)
    runtime.calculate(atoms, ["energy"])

    diagnostics = runtime.diagnostics()
    assert diagnostics["graph_build_count"] == 1
    assert diagnostics["graph_reuse_count"] == 1
    assert diagnostics["geometry_update_count"] == 2
    # The same reuse still invalidates the lattice-dependent reciprocal kernel.
    assert runtime.reciprocal_kernel_key != kernel_key
    assert diagnostics["reciprocal_kernel_invalidation_count"] == 2


def test_fourier_d3_pipeline_composes_the_reciprocal_runtime(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    native = _FakeReciprocalNative()
    pipeline = CorrectionPipeline(
        [_fourier_configuration()], native=native, dtype=np.float64
    )

    totals = pipeline.calculate(
        _periodic_pair(), ["energy", "energies", "forces", "stress"]
    )

    assert totals["energy"] == pytest.approx(-0.5)
    assert totals["energies"].shape == (2,)
    assert totals["forces"].shape == (2, 3)
    assert totals["stress"].shape == (6,)
    diagnostics = pipeline.diagnostics()[0]
    assert diagnostics["method"] == "fourier_d3"
    assert diagnostics["topology"] == "global"
    assert diagnostics["pair_cutoff"] is None
    assert diagnostics["graph_build_count"] == 1
    assert diagnostics["resolved_mesh"] == (8, 8, 8)
    assert len(native.instances) == 1


# ---------------------------------------------------------------------------
# FourierD3 mesh policy: supported bounds, resolution order, and clean failure.
#
# The prospective mesh is resolved and validated before the native evaluator exists and
# before any graph or counter changes, so an unsupported mesh policy leaves a runtime that
# has done nothing. The smooth-number search must be affordable for a large target, and
# the whole path must be free of NumPy overflow warnings and raw OverflowError.
# ---------------------------------------------------------------------------

#: Every state a fresh reciprocal runtime must still show after a failed preparation.
_PRISTINE_KEYS = (
    "decomposition_key",
    "reciprocal_kernel_key",
    "resolved_mesh",
    "fft_plan_key",
)

_PRISTINE_COUNTERS = (
    "native_allocation_count",
    "graph_build_count",
    "graph_reuse_count",
    "geometry_update_count",
    "backend_fallback_count",
    "decomposition_invalidation_count",
    "reciprocal_kernel_invalidation_count",
    "mesh_invalidation_count",
    "fft_plan_invalidation_count",
)


def _smooth_numbers_up_to(limit):
    """Every 2/3/5/7-smooth integer at or below ``limit``.

    Generated by merging the scaled streams of the classic smooth-number recurrence, which
    is a different construction from the power-of-two-bounded search the runtime uses.
    """

    values = [1]
    indices = {radix: 0 for radix in (2, 3, 5, 7)}
    following = {radix: radix for radix in (2, 3, 5, 7)}
    while True:
        smallest = min(following.values())
        if smallest > limit:
            return values
        values.append(smallest)
        for radix in (2, 3, 5, 7):
            if following[radix] == smallest:
                indices[radix] += 1
                following[radix] = radix * values[indices[radix]]


@contextlib.contextmanager
def _deadline(seconds):
    """Raise ``TimeoutError`` if the block outlives ``seconds`` of wall clock."""

    def _expire(_signum, _frame):
        raise TimeoutError(f"exceeded the {seconds} second budget")

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _assert_pristine(runtime, native):
    assert native.instances == []
    assert runtime.evaluator is None
    assert runtime._topology_reference is None
    for name in _PRISTINE_KEYS:
        assert getattr(runtime, name) is None, name
    diagnostics = runtime.diagnostics()
    for name in _PRISTINE_COUNTERS:
        assert diagnostics[name] == 0, name


def test_fourier_d3_radix_ceiling_matches_an_independent_enumeration_for_a_huge_target():
    target = (1 << 50) + 1
    smooth = _smooth_numbers_up_to(1 << 51)
    expected = min(value for value in smooth if value >= target)

    # The next smooth number above 2**50 + 1 is 3.8e10 away, so an implementation that
    # walks candidate integers cannot answer inside this budget.
    with _deadline(10.0):
        result = corrections._radix_friendly_ceiling(target)

    assert result == expected
    assert result >= target


def test_fourier_d3_radix_ceiling_matches_an_independent_enumeration_on_a_dense_range():
    smooth = _smooth_numbers_up_to(1 << 16)

    for target in range(1, 5000):
        expected = smooth[bisect.bisect_left(smooth, target)]
        assert corrections._radix_friendly_ceiling(target) == expected, target


def test_fourier_d3_resolves_a_huge_supported_mesh_promptly(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=1.0)
    )
    atoms = Atoms(
        "CO",
        scaled_positions=[[0.25, 0.25, 0.25], [0.3, 0.25, 0.25]],
        cell=[1_500_000.0, 1_500_000.0, 1_500_000.0],
        pbc=True,
    )

    with _deadline(10.0):
        runtime.calculate(atoms, ["energy"])

    # 1_500_000 is itself 2/3/5/7-smooth, so the resolved mesh is exact and the volume
    # stays inside the 64-bit-safe bound.
    assert runtime.resolved_mesh == (1_500_000, 1_500_000, 1_500_000)
    assert runtime.resolved_mesh[0] ** 3 <= corrections._FOURIER_D3_MAX_MESH_VOLUME


@pytest.mark.parametrize(
    ("cell", "spacing"),
    [
        ([8.0, 8.0, 8.0], 5e-324),
        ([[1e308, 0.0, 0.0], [0.0, 1e308, 0.0], [0.0, 0.0, 1e308]], 1e-300),
    ],
    ids=("denormal-spacing", "huge-cell"),
)
def test_fourier_d3_rejects_a_nonfinite_mesh_quotient_without_warning(cell, spacing):
    runtime, native = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=spacing)
    )
    atoms = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=cell, pbc=True
    )

    with warnings.catch_warnings():
        # Any NumPy overflow warning becomes an error here: the failure must be reported,
        # not warned about.
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="mesh spacing") as error:
            runtime.calculate(atoms, ["energy"])

    message = str(error.value)
    assert "FourierD3" in message
    assert "not finite" in message
    assert "cell length" in message
    _assert_pristine(runtime, native)


def test_fourier_d3_rejects_a_mesh_beyond_the_supported_axis_bound():
    runtime, native = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=1.0)
    )
    # A 1e300 Angstrom axis is finite and invertible but demands ~10**300 nodes: the bound
    # has to be checked before the smooth-number search, which could never enumerate that.
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[1e300, 1e300, 1e300],
        pbc=True,
    )

    with _deadline(10.0):
        with pytest.raises(ValueError, match="mesh spacing") as error:
            runtime.calculate(atoms, ["energy"])

    message = str(error.value)
    assert str(corrections._FOURIER_D3_MAX_MESH_AXIS) in message
    assert "axis" in message
    assert "cell length" in message
    _assert_pristine(runtime, native)


def test_fourier_d3_rejects_a_mesh_beyond_the_supported_volume_bound():
    runtime, native = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=1.0)
    )
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[2_000_000_000.0] * 3,
        pbc=True,
    )

    with _deadline(10.0):
        with pytest.raises(ValueError, match="volume") as error:
            runtime.calculate(atoms, ["energy"])

    message = str(error.value)
    assert str(corrections._FOURIER_D3_MAX_MESH_VOLUME) in message
    assert "cell length" in message
    _assert_pristine(runtime, native)


@pytest.mark.parametrize(
    "mesh",
    [(1 << 63, 1, 1), (1 << 62, 1 << 62, 1 << 62)],
    ids=("axis", "volume"),
)
def test_fourier_d3_rejects_an_explicit_mesh_beyond_the_supported_bounds(mesh):
    with pytest.raises(ValueError, match="FourierD3") as error:
        _fourier_configuration(mesh=mesh)

    assert "supported" in str(error.value)


def test_fourier_d3_preparation_failure_commits_no_domain_state(monkeypatch):
    def fail_host_builder(*_args, **_kwargs):
        raise AssertionError("device-eligible reciprocal graph used the host builder")

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(corrections, "build_host_neighbor_graph", fail_host_builder)
    native = _FakeReciprocalNative(
        device_prepare_error=RuntimeError("device build failed")
    )
    runtime, _ = _build_fourier_runtime(native=native)

    with pytest.raises(RuntimeError, match="device build failed"):
        runtime.calculate(_periodic_pair(), ["energy"])

    # The mesh resolved, so the evaluator was allocated, but nothing was committed: the
    # keys and every invalidation counter stay untouched until a preparation succeeds.
    diagnostics = runtime.diagnostics()
    assert diagnostics["native_allocation_count"] == 1
    assert runtime._topology_reference is None
    for name in _PRISTINE_KEYS:
        assert getattr(runtime, name) is None, name
    for name in _PRISTINE_COUNTERS:
        if name != "native_allocation_count":
            assert diagnostics[name] == 0, name

    native._device_prepare_error = None
    runtime.calculate(_periodic_pair(), ["energy"])

    recovered = runtime.diagnostics()
    assert recovered["graph_build_count"] == 1
    assert recovered["native_allocation_count"] == 1
    assert runtime.resolved_mesh == (8, 8, 8)
    for name in (
        "decomposition_invalidation_count",
        "reciprocal_kernel_invalidation_count",
        "mesh_invalidation_count",
        "fft_plan_invalidation_count",
    ):
        assert recovered[name] == 1, name


def test_fourier_d3_rebuild_failure_preserves_the_last_committed_state(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    native = _FakeReciprocalNative()
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=1.0), native=native
    )
    atoms = _periodic_pair()

    runtime.calculate(atoms, ["energy"])
    committed = dict(runtime.diagnostics())
    assert runtime.resolved_mesh == (8, 8, 8)

    # A larger cell plus a displacement past the skin forces a rebuild, and the rebuild
    # fails: the keys and counters of the last successful preparation stand, so the
    # prospective (9, 9, 9) mesh is never installed by a failed step.
    atoms.set_cell(np.diag([9.0, 9.0, 9.0]), scale_atoms=True)
    atoms.positions[1, 1] += 1.0
    native._device_prepare_error = RuntimeError("device rebuild failed")

    with pytest.raises(RuntimeError, match="device rebuild failed"):
        runtime.calculate(atoms, ["energy"])

    assert runtime.resolved_mesh == (8, 8, 8)
    # A refused preparation is reported only through its own failure counter: every
    # field describing committed state stands exactly as it did, including the
    # prospective mesh that the failed step never installed.
    failed = dict(runtime.diagnostics())
    assert failed["failed_preparation_count"] == 1
    assert failed["failed_compute_count"] == 0
    for key in set(committed) - {"failed_preparation_count"}:
        assert failed[key] == committed[key], key
    for name in _PRISTINE_KEYS:
        assert getattr(runtime, name) is not None, name


def test_fourier_d3_skew_cell_resolves_each_axis_from_its_lattice_vector():
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=1.0)
    )
    atoms = Atoms(
        "CO",
        scaled_positions=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.3]],
        cell=[[10.0, 0.0, 0.0], [3.0, 8.0, 0.0], [1.0, 1.0, 7.0]],
        pbc=True,
    )

    runtime.calculate(atoms, ["energy"])

    # Row lengths 10, sqrt(73) = 8.544, sqrt(51) = 7.141 round up to 10, 9, 8.
    assert runtime.resolved_mesh == (10, 9, 8)


# ---------------------------------------------------------------------------
# the merged Task 12 diagnostic record
#
# The runtime owns the counters and the configuration the native evaluator does not
# know; the evaluator owns the realized state.  One record has to carry both without a
# duplicate or a contradictory field, and it has to be readable as state.
# ---------------------------------------------------------------------------

# The stage names the native evaluator times, in the order its record lists them.
_STAGE_NAMES = (
    "preparation",
    "forward_preparation",
    "coordination",
    "assignment_forward_fft",
    "reciprocal_contraction",
    "inverse_fft_gather_adjoint",
    "virial_and_strain",
    "result_reduction",
    "result_export",
)

_CAPABILITY_KEYS = {
    "capability_available",
    "capability_provider",
    "capability_provider_variant",
    "capability_kokkosfft_version",
    "capability_kokkos_version",
    "capability_execution_backend",
    "capability_reason",
}
_DIAGNOSTIC_KEYS = _CAPABILITY_KEYS | {
    "method",
    "topology",
    "graph_backend",
    "graph_cutoff",
    "graph_build_cutoff",
    "coordination_cutoff",
    "skin",
    "pair_cutoff",
    "num_edges",
    "graph_nodes",
    "graph_build_count",
    "graph_reuse_count",
    "geometry_update_count",
    "graph_generation",
    "backend_fallback_count",
    "mesh_policy",
    "mesh_spacing_requested",
    "resolved_mesh",
    "mesh_spacing",
    "mesh_points",
    "spline_order",
    "precision",
    "active_species",
    "table_fingerprint",
    "decomposition_tolerance",
    "rank_cap",
    "retained_rank",
    "rank_total",
    "rank_chunk",
    "rank_chunk_count",
    "n_references",
    "decomposition_max_relative_error",
    "decomposition_tolerance_met",
    "decomposition_rank_capped",
    "decomposition_invalidation_count",
    "reciprocal_kernel_invalidation_count",
    "mesh_invalidation_count",
    "fft_plan_invalidation_count",
    "native_allocation_count",
    "fft_plans",
    "plan_builds",
    "plan_reuses",
    "plan_releases",
    "plan_key_rebuilds",
    "kernel_key_hits",
    "kernel_key_misses",
    "kernel_key_rebuilds",
    "stencil_points",
    "input_generation",
    "published_generation",
    "computed",
    "result_available",
    "forces_available",
    "stress_available",
    "engine_loaded",
    "active_bytes",
    "capacity_bytes",
    "reservations",
    "allocations",
    "plan_retained_bytes",
    "memory_components",
    "stage_timing",
    "timing_enabled",
    "failed_preparation_count",
    "failed_compute_count",
}


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_fourier_d3_diagnostics_record_covers_the_whole_contract(precision):
    runtime, _ = _build_fourier_runtime(dtype=precision)
    runtime.calculate(_periodic_pair(), ["energy", "forces", "stress"])

    report = runtime.diagnostics()

    assert _DIAGNOSTIC_KEYS <= set(report), sorted(_DIAGNOSTIC_KEYS - set(report))
    assert report["method"] == "fourier_d3"
    assert report["topology"] == "global"
    assert report["precision"] == precision
    assert report["pair_cutoff"] is None
    # The exact radius the search must cover, and the radius the built graph covers.
    assert report["graph_cutoff"] == pytest.approx(2.5)
    assert report["graph_build_cutoff"] == pytest.approx(2.9)
    assert report["skin"] == pytest.approx(0.4)
    assert report["coordination_cutoff"] == pytest.approx(2.5)


def test_fourier_d3_diagnostics_report_the_mesh_policy_and_its_outcome():
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(mesh=None, mesh_spacing=1.0)
    )
    runtime.calculate(_periodic_pair(), ["energy"])
    report = runtime.diagnostics()

    assert report["mesh_policy"] == "spacing"
    assert report["mesh_spacing_requested"] == pytest.approx(1.0)
    assert report["resolved_mesh"] == (8, 8, 8)
    assert report["mesh_points"] == 8 * 8 * 8
    assert report["spline_order"] == 4
    assert report["precision"] == "float64"

    explicit, _ = _build_fourier_runtime(_fourier_configuration())
    explicit.calculate(_periodic_pair(), ["energy"])
    explicit_report = explicit.diagnostics()
    assert explicit_report["mesh_policy"] == "explicit"
    assert explicit_report["mesh_spacing_requested"] is None
    assert explicit_report["resolved_mesh"] == (8, 8, 8)


def test_fourier_d3_diagnostics_report_the_requested_and_realized_decomposition():
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(decomposition_tolerance=1e-4, rank_cap=2)
    )
    runtime.calculate(_periodic_pair(), ["energy"])
    report = runtime.diagnostics()

    assert report["decomposition_tolerance"] == pytest.approx(1e-4)
    assert report["rank_cap"] == 2
    assert report["retained_rank"] == 2
    assert report["decomposition_tolerance_met"] is False
    # The cap is named as the reason the tolerance is unmet, and the request is never
    # silently relaxed to a smaller rank or a looser tolerance.
    assert report["decomposition_rank_capped"] is True
    assert report["decomposition_max_relative_error"] > 1e-4


def test_fourier_d3_diagnostics_do_not_claim_a_cap_that_did_not_bind():
    runtime, _ = _build_fourier_runtime(_fourier_configuration(rank_cap=10_000))
    runtime.calculate(_periodic_pair(), ["energy"])
    report = runtime.diagnostics()

    assert report["rank_cap"] == 10_000
    assert report["decomposition_tolerance_met"] is True
    assert report["decomposition_rank_capped"] is False
    assert report["retained_rank"] < 10_000


def test_fourier_d3_diagnostics_do_not_blame_an_unmet_full_rank_tolerance_on_cap():
    runtime, _ = _build_fourier_runtime(
        _fourier_configuration(decomposition_tolerance=1e-30, rank_cap=10)
    )
    runtime.calculate(_periodic_pair(), ["energy"])
    report = runtime.diagnostics()

    assert report["retained_rank"] == 10
    assert report["decomposition_tolerance_met"] is False
    assert report["decomposition_rank_capped"] is False


def test_fourier_d3_diagnostics_merge_the_native_state_without_duplicates():
    runtime, native = _build_fourier_runtime()
    runtime.calculate(_periodic_pair(), ["energy", "forces"])
    report = runtime.diagnostics()
    evaluator = runtime.evaluator
    native_report = evaluator.diagnostics()

    # The graph half of the record is the native graph's own report.
    graph = evaluator.graph_diagnostics()
    assert report["graph_backend"] == graph["graph_backend"]
    assert report["num_edges"] == graph["num_edges"]
    assert report["graph_nodes"] == graph["num_nodes"]
    assert report["graph_generation"] == graph["generation"]
    assert report["graph_build_cutoff"] == pytest.approx(
        native.host_graphs[0]["build_cutoff"]
    )
    # The evaluator half is carried through under names that do not collide with the
    # runtime's own counters, so no consumer has to choose between two values.
    assert report["precision"] == evaluator.precision
    assert report["resolved_mesh"] == evaluator.meshes[-1]
    assert report["active_species"] == list(evaluator.decompositions[-1]["species"])
    assert report["stencil_points"] == native_report["stencil_points"]
    assert report["input_generation"] == native_report["input_generation"]
    assert report["computed"] is True
    assert report["result_available"] is True
    assert report["engine_loaded"] is True
    assert len(set(report)) == len(report)


def test_fourier_d3_diagnostics_report_capability_from_the_build_report():
    native = _FakeReciprocalNative()
    native._fourier_d3_capability = lambda: {
        "available": True,
        "provider": "fftw3",
        "provider_variant": "FFTW_OPENMP",
        "kokkosfft_version": "2.0.0",
        "kokkos_version": "5.0.2",
        "execution_backend": "cpu",
        "reason": "",
    }
    runtime, _ = _build_fourier_runtime(native=native)
    runtime.calculate(_periodic_pair(), ["energy"])
    report = runtime.diagnostics()

    assert report["capability_available"] is True
    assert report["capability_provider"] == "fftw3"
    assert report["capability_provider_variant"] == "FFTW_OPENMP"
    assert report["capability_kokkosfft_version"] == "2.0.0"
    assert report["capability_kokkos_version"] == "5.0.2"
    assert report["capability_execution_backend"] == "cpu"
    assert report["capability_reason"] == ""


def test_fourier_d3_diagnostics_report_an_unreported_capability_truthfully():
    runtime, _ = _build_fourier_runtime()
    runtime.calculate(_periodic_pair(), ["energy"])
    report = runtime.diagnostics()

    # The stand-in reports no capability, so the record says so rather than inventing
    # a provider from the platform or the configuration.
    assert report["capability_available"] is False
    assert report["capability_provider"] == ""
    assert report["capability_reason"] == "unreported"


def test_fourier_d3_failure_counters_never_report_committed_work():
    runtime, _ = _build_fourier_runtime()
    runtime.calculate(_periodic_pair(), ["energy"])
    committed = dict(runtime.diagnostics())
    assert committed["failed_preparation_count"] == 0
    assert committed["failed_compute_count"] == 0

    class _RefusingEvaluator:
        """A stand-in whose compute refuses after a successful preparation."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def compute(self, compute_forces, compute_stress):
            raise RuntimeError("compute refused")

    runtime.evaluator = _RefusingEvaluator(runtime.evaluator)
    with pytest.raises(RuntimeError, match="compute refused"):
        runtime.calculate(_periodic_pair(), ["energy", "forces"])

    failed = dict(runtime.diagnostics())
    assert failed["failed_compute_count"] == 1
    assert failed["failed_preparation_count"] == 0
    # A failed compute publishes nothing: every field that describes the published
    # result, the realized domains, and the inputs is unchanged.  The preparation the
    # call ran did commit - the graph was reused and its geometry pushed - so the two
    # counters that describe preparation may advance, and only those.
    for key in set(committed) - {
        "failed_compute_count",
        "failed_preparation_count",
        "graph_reuse_count",
        "geometry_update_count",
    }:
        assert failed[key] == committed[key], key
    assert failed["graph_reuse_count"] == committed["graph_reuse_count"] + 1
    assert failed["geometry_update_count"] == committed["geometry_update_count"] + 1


def test_fourier_d3_failed_preparation_keeps_the_committed_diagnostics(monkeypatch):
    runtime, _ = _build_fourier_runtime()
    atoms = _periodic_pair()
    runtime.calculate(atoms, ["energy"])
    committed = dict(runtime.diagnostics())

    def refuse(*_args, **_kwargs):
        raise RuntimeError("graph build refused")

    monkeypatch.setattr(corrections, "build_host_neighbor_graph", refuse)
    atoms.positions[1, 1] += 10.0
    with pytest.raises(RuntimeError, match="graph build refused"):
        runtime.calculate(atoms, ["energy"])

    failed = dict(runtime.diagnostics())
    assert failed["failed_preparation_count"] == 1
    assert failed["failed_compute_count"] == 0
    for key in set(committed) - {"failed_preparation_count", "failed_compute_count"}:
        assert failed[key] == committed[key], key


def test_fourier_d3_stage_timing_is_opt_in_and_reported_when_enabled():
    runtime, _ = _build_fourier_runtime()
    atoms = _periodic_pair()
    runtime.calculate(atoms, ["energy"])

    assert runtime.diagnostics()["timing_enabled"] is False
    assert runtime.diagnostics()["stage_timing"] is None

    runtime.enable_stage_timing(True)
    runtime.calculate(atoms, ["energy", "forces", "stress"])
    report = runtime.diagnostics()
    assert report["timing_enabled"] is True
    assert report["stage_timing"]
    for name, stage in report["stage_timing"].items():
        assert stage["count"] >= 1, name
        assert stage["cumulative_seconds"] >= stage["last_seconds"] >= 0.0, name

    runtime.enable_stage_timing(False)
    runtime.calculate(atoms, ["energy"])
    assert runtime.diagnostics()["timing_enabled"] is False
    assert runtime.diagnostics()["stage_timing"] is None


def test_pipeline_forwards_the_stage_timing_opt_in():
    pipeline = CorrectionPipeline(
        [_fourier_configuration()],
        native=_FakeReciprocalNative(),
        dtype=np.float64,
    )
    runtime = pipeline._runtimes[0]

    assert runtime.diagnostics()["timing_enabled"] is False
    pipeline.enable_stage_timing(True)
    assert runtime.diagnostics()["timing_enabled"] is True
    pipeline.calculate(_periodic_pair(), ["energy"])
    assert runtime.diagnostics()["stage_timing"]
    pipeline.enable_stage_timing(False)
    assert runtime.diagnostics()["stage_timing"] is None


def test_pipeline_stage_timing_opt_in_is_a_noop_without_a_timed_runtime():
    pipeline = CorrectionPipeline(
        [D2Correction()], native=_RecordingNative(), dtype=np.float64
    )
    before = pipeline.diagnostics()

    pipeline.enable_stage_timing(True)

    assert pipeline.diagnostics() == before
