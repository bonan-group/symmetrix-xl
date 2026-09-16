from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from symmetrix import Symmetrix, SymmetrixEnsemble
from symmetrix import ensemble as ensemble_module


class DummySymmetrix(Symmetrix):
    def __init__(
        self,
        offset=0.0,
        *,
        field_aware=False,
        cutoff=6.0,
        atomic_numbers=(1, 8),
        dtype="float64",
        use_kokkos=True,
        streamed_edges="direct",
        low_memory=True,
        neighbor_skin=0.5,
    ):
        Calculator.__init__(self)
        self.offset = float(offset)
        self.evaluator = SimpleNamespace(
            has_field_coupling=field_aware,
            r_cut=cutoff,
            atomic_numbers=list(atomic_numbers),
        )
        self._native_model_type = "MACEField" if field_aware else "MACE"
        self._kernel_launch_dtype = dtype
        self.use_kokkos = use_kokkos
        self.streamed_edges = streamed_edges
        self.low_memory = low_memory
        self.neighbor_skin = neighbor_skin
        self._electric_field = None
        self._macefield_electric_field = None
        self.calls = []
        self.implemented_properties = [
            "energy",
            "free_energy",
            "energies",
            "forces",
            "stress",
        ]
        if field_aware:
            self.implemented_properties.extend(
                ["node_energy", "polarization", "becs", "polarizability"]
            )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.calls.append(tuple(properties))
        positions = np.asarray(self.atoms.positions, dtype=float)
        energy = float(np.sum(positions**2) + self.offset)
        available = {
            "energy": energy,
            "free_energy": energy,
            "energies": np.sum(positions**2, axis=1) + self.offset / len(self.atoms),
            "forces": -2.0 * positions + self.offset,
            "stress": np.arange(6, dtype=float) + self.offset,
            "polarization": np.arange(3, dtype=float) + self.offset,
            "becs": np.full((len(self.atoms), 9), self.offset),
            "polarizability": np.arange(9, dtype=float) + self.offset,
        }
        self.results = {
            prop: np.array(available[prop], copy=True) for prop in properties
        }


class ConstructedDummySymmetrix(DummySymmetrix):
    def __init__(self, model_file, **kwargs):
        super().__init__(float(model_file), **kwargs)


@pytest.fixture
def atoms():
    return Atoms(
        "HO",
        positions=[[0.1, 0.2, 0.3], [0.8, 0.4, 0.2]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )


def test_constructor_requires_exactly_one_nonempty_source():
    with pytest.raises(ValueError, match="exactly one"):
        SymmetrixEnsemble()
    with pytest.raises(ValueError, match="exactly one"):
        SymmetrixEnsemble(["model.json"], calculators=[DummySymmetrix()])
    with pytest.raises(ValueError, match="at least one"):
        SymmetrixEnsemble(calculators=[])
    with pytest.raises(TypeError, match="member 0"):
        SymmetrixEnsemble(calculators=[Calculator()])
    with pytest.raises(ValueError, match="cannot be used with calculators"):
        SymmetrixEnsemble(calculators=[DummySymmetrix()], dtype="float32")

    single_member = SymmetrixEnsemble(calculators=DummySymmetrix())
    assert single_member.num_models == 1


def test_model_files_construct_members_with_shared_options(monkeypatch):
    monkeypatch.setattr(ensemble_module, "Symmetrix", ConstructedDummySymmetrix)

    calculator = SymmetrixEnsemble(["1.0", "3.0"], neighbor_skin=0.25, dtype="float32")

    assert calculator.num_models == 2
    assert [member.offset for member in calculator.calculators] == [1.0, 3.0]
    assert all(member.neighbor_skin == 0.25 for member in calculator.calculators)
    assert all(
        member._kernel_launch_dtype == "float32" for member in calculator.calculators
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("field_aware", True, "model type"),
        ("cutoff", 5.0, "cutoff"),
        ("atomic_numbers", (1, 6, 8), "atomic-number mapping"),
        ("dtype", "float32", "precision"),
        ("use_kokkos", False, "Kokkos selection"),
        ("streamed_edges", "generic", "streamed-edge mode"),
        ("low_memory", False, "low-memory policy"),
        ("neighbor_skin", 0.0, "neighbor skin"),
    ],
)
def test_constructor_reports_member_compatibility_mismatch(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        SymmetrixEnsemble(
            calculators=[DummySymmetrix(), DummySymmetrix(**{keyword: value})]
        )


def test_energy_mean_committee_and_population_variance(atoms):
    calculator = SymmetrixEnsemble(
        calculators=[DummySymmetrix(1.0), DummySymmetrix(3.0)]
    )
    atoms.calc = calculator

    variance = calculator.get_property("energy_var", atoms)

    expected_base = np.sum(atoms.positions**2)
    assert calculator.num_models == 2
    assert calculator.results["energy"] == pytest.approx(expected_base + 2.0)
    assert np.allclose(
        calculator.results["energy_comm"], [expected_base + 1.0, expected_base + 3.0]
    )
    assert variance == pytest.approx(1.0)
    assert all(member.calls == [("energy",)] for member in calculator.calculators)


def test_force_and_stress_shapes_and_values(atoms):
    calculator = SymmetrixEnsemble(
        calculators=[DummySymmetrix(0.0), DummySymmetrix(2.0)]
    )
    calculator.calculate(atoms, ["forces", "stress_comm"], all_changes)

    assert calculator.results["forces"].shape == (len(atoms), 3)
    assert calculator.results["forces_comm"].shape == (2, len(atoms), 3)
    assert calculator.results["forces_var"].shape == (len(atoms), 3)
    assert np.allclose(calculator.results["forces_var"], 1.0)
    assert calculator.results["stress"].shape == (6,)
    assert calculator.results["stress_comm"].shape == (2, 6)
    assert calculator.results["stress_var"].shape == (6,)
    assert np.allclose(calculator.results["stress_var"], 1.0)


@pytest.mark.parametrize(
    ("property_name", "shape"),
    [
        ("polarization", (3,)),
        ("becs", (2, 9)),
        ("polarizability", (9,)),
    ],
)
def test_field_response_aggregation(property_name, shape, atoms):
    calculator = SymmetrixEnsemble(
        calculators=[
            DummySymmetrix(1.0, field_aware=True),
            DummySymmetrix(3.0, field_aware=True),
        ]
    )
    calculator.calculate(atoms, [f"{property_name}_comm"], all_changes)

    assert calculator.results[property_name].shape == shape
    assert calculator.results[f"{property_name}_comm"].shape == (2, *shape)
    assert calculator.results[f"{property_name}_var"].shape == shape
    assert np.allclose(calculator.results[f"{property_name}_var"], 1.0)


def test_field_routing_computes_only_requested_response(atoms):
    members = [
        DummySymmetrix(1.0, field_aware=True),
        DummySymmetrix(2.0, field_aware=True),
    ]
    calculator = SymmetrixEnsemble(calculators=members)

    calculator.calculate(atoms, ["polarization_var"], all_changes)

    assert all(member.calls == [("polarization",)] for member in members)
    assert "becs" not in calculator.results
    assert "polarizability" not in calculator.results


def test_single_member_variance_and_result_copy(atoms):
    member = DummySymmetrix(2.0)
    calculator = SymmetrixEnsemble(calculators=[member])
    calculator.calculate(atoms, ["forces"], all_changes)
    expected = calculator.results["forces"].copy()

    member.results["forces"].fill(123.0)

    assert np.allclose(calculator.results["forces"], expected)
    assert np.allclose(calculator.results["forces_var"], 0.0)


def test_ase_attachment_caches_and_invalidates_on_system_change(atoms):
    members = [DummySymmetrix(0.0), DummySymmetrix(1.0)]
    calculator = SymmetrixEnsemble(calculators=members)
    atoms.calc = calculator

    first = atoms.get_potential_energy()
    second = atoms.get_potential_energy()
    atoms.positions[0, 0] += 0.1
    moved = atoms.get_potential_energy()

    assert first == second
    assert moved != first
    assert all(len(member.calls) == 2 for member in members)


def test_additional_property_request_retains_cached_family(atoms):
    members = [DummySymmetrix(0.0), DummySymmetrix(1.0)]
    calculator = SymmetrixEnsemble(calculators=members)
    atoms.calc = calculator

    energy = calculator.get_property("energy", atoms)
    calculator.get_property("forces", atoms)

    assert calculator.results["energy"] == energy
    assert all(member.calls == [("energy",), ("forces",)] for member in members)


def test_electric_field_override_is_forwarded_and_invalidates_cache(atoms):
    members = [
        DummySymmetrix(0.0, field_aware=True),
        DummySymmetrix(1.0, field_aware=True),
    ]
    calculator = SymmetrixEnsemble(calculators=members)
    calculator.results["energy"] = 1.0

    calculator.electric_field = [0.01, -0.02, 0.03]

    assert np.allclose(calculator.electric_field, [0.01, -0.02, 0.03])
    assert all(
        np.allclose(member.electric_field, [0.01, -0.02, 0.03]) for member in members
    )
    assert calculator.results == {}
