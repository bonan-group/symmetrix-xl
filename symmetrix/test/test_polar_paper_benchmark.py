import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


BENCHMARKS = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import polar_paper_benchmark as benchmark  # noqa: E402
from polar_paper_systems import (  # noqa: E402
    build_system,
    default_system_names,
    system_names,
    system_sha256,
)


def test_benchmark_times_only_the_simulation_relevant_efs_path():
    assert benchmark.PROPERTY_SETS == {
        "EFS": ["energy", "forces", "stress"],
    }


def test_mace_torch_cuda_baseline_defaults_to_verified_cueq(tmp_path):
    args = benchmark.parser().parse_args(
        [
            "run",
            "--implementation",
            "mace-torch",
            "--output",
            str(tmp_path / "record.json"),
        ]
    )

    assert args.torch_backend == "cueq"
    assert benchmark._torch_calculator_options(args.torch_backend) == {
        "enable_cueq": True
    }


def test_standard_mace_torch_backend_remains_an_explicit_diagnostic(tmp_path):
    args = benchmark.parser().parse_args(
        [
            "run",
            "--implementation",
            "mace-torch",
            "--torch-backend",
            "e3nn",
            "--output",
            str(tmp_path / "record.json"),
        ]
    )

    assert benchmark._torch_calculator_options(args.torch_backend) == {
        "enable_cueq": False
    }


def test_symmetrix_benchmark_accepts_forced_pme(tmp_path):
    args = benchmark.parser().parse_args(
        [
            "run",
            "--implementation",
            "symmetrix",
            "--polar-electrostatics",
            "pme",
            "--polar-pme-mesh",
            "40",
            "36",
            "32",
            "--polar-pme-order",
            "5",
            "--output",
            str(tmp_path / "record.json"),
        ]
    )

    assert args.polar_electrostatics == "pme"
    assert args.polar_pme_mesh == [40, 36, 32]
    assert args.polar_pme_order == 5


def test_boundary_policy_is_applied_to_converted_calculator_models():
    class LongRangeModule:
        def __init__(self):
            self.pbc_handling = "mixed_periodic"

        def set_pbc_handling(self, value):
            self.pbc_handling = value

    class Model:
        def __init__(self):
            self.electric_potential_descriptor = LongRangeModule()
            self.coulomb_energy = LongRangeModule()

    calculator = SimpleNamespace(models=[Model()])

    status = benchmark._set_calculator_pbc_handling(calculator, "auto")

    assert status == [
        {
            "electric_potential_descriptor": "auto",
            "coulomb_energy": "auto",
        }
    ]
    assert calculator.models[0].electric_potential_descriptor.pbc_handling == "auto"
    assert calculator.models[0].coulomb_energy.pbc_handling == "auto"


@pytest.mark.parametrize(
    "name,atoms,pbc,lengths",
    (
        ("molecule_h2o", 3, (False, False, False), (10.0, 10.0, 10.0)),
        (
            "finite_cluster_384",
            384,
            (False, False, False),
            (20.4, 20.4, 32.8),
        ),
        ("x23_benzene", 48, (True, True, True), (7.39, 9.42, 6.81)),
        ("redox_water64", 192, (True, True, True), (12.4, 12.4, 12.4)),
        (
            "liquid_water333",
            999,
            (True, True, True),
            (21.5383813974, 21.5383813974, 21.5383813974),
        ),
        (
            "liquid_water1000",
            3000,
            (True, True, True),
            (31.07408279771176,) * 3,
        ),
        (
            "liquid_water1728",
            5184,
            (True, True, True),
            (37.28889935725412,) * 3,
        ),
        (
            "liquid_water2744",
            8232,
            (True, True, True),
            (43.50371591679647,) * 3,
        ),
        (
            "liquid_water4096",
            12288,
            (True, True, True),
            (49.71853247633882,) * 3,
        ),
    ),
)
def test_paper_systems_are_deterministic(name, atoms, pbc, lengths):
    first, first_metadata = build_system(name)
    second, second_metadata = build_system(name)

    assert len(first) == atoms
    assert tuple(first.pbc) == pbc
    np.testing.assert_allclose(first.cell.lengths(), lengths, rtol=0.0, atol=1.0e-12)
    assert system_sha256(first) == system_sha256(second)
    assert first_metadata == second_metadata
    assert first.info["charge"] == first.info["total_charge"] == 0.0
    assert first.info["spin"] == first.info["total_spin"] == 1.0


def test_large_scaling_systems_are_opt_in():
    assert "liquid_water1000" in system_names()
    assert "liquid_water1728" in system_names()
    assert "liquid_water2744" in system_names()
    assert "liquid_water4096" in system_names()
    assert "liquid_water1000" not in default_system_names()
    assert "liquid_water1728" not in default_system_names()
    assert "liquid_water2744" not in default_system_names()
    assert "liquid_water4096" not in default_system_names()


def _record(implementation, *, energy_shift=0.0, force_shift=0.0):
    systems = []
    for name in system_names()[:2]:
        atoms, _ = build_system(name)
        forces = np.zeros((len(atoms), 3))
        forces[0, 0] = force_shift
        systems.append(
            {
                "name": name,
                "atoms": len(atoms),
                "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
                "system_sha256": system_sha256(atoms),
                "snapshot": {
                    "energy_eV": 1.0 + energy_shift * len(atoms),
                    "forces_eV_per_A": forces.tolist(),
                    "stress_eV_per_A3_voigt": [0.0] * 6,
                },
                "timings": {
                    label: {"median_ms": 2.0 if implementation == "symmetrix" else 4.0}
                    for label in benchmark.PROPERTY_SETS
                },
                "solver": {},
            }
        )
    return {
        "implementation": implementation,
        "model": {"sha256": "same"},
        "protocol": {
            "device": "cuda",
            "dtype": "float32",
            "warmups": 5,
            "repeats": 20,
            "synchronized": True,
        },
        "systems": systems,
    }


def test_compare_checks_values_and_reports_timing_ratio(tmp_path):
    torch_path = tmp_path / "torch.json"
    symmetrix_path = tmp_path / "symmetrix.json"
    output = tmp_path / "comparison.json"
    torch_path.write_text(json.dumps(_record("mace-torch")))
    symmetrix_path.write_text(
        json.dumps(
            _record(
                "symmetrix",
                energy_shift=1.0e-4,
                force_shift=1.0e-3,
            )
        )
    )
    args = SimpleNamespace(
        torch_record=torch_path,
        symmetrix_record=symmetrix_path,
        output=output,
        energy_atol_per_atom=2.0e-4,
        force_atol=5.0e-3,
        stress_atol=5.0e-4,
        enforce=True,
    )

    assert benchmark.compare(args) == 0
    result = json.loads(output.read_text())
    assert result["passed"]
    assert result["systems"][0]["symmetrix_over_torch_time_ratio"]["EFS"] == 0.5
    assert result["timing_comparability"]["EFS"].startswith("like-for-like")


def test_compare_enforcement_fails_outside_tolerance(tmp_path):
    torch_path = tmp_path / "torch.json"
    symmetrix_path = tmp_path / "symmetrix.json"
    output = tmp_path / "comparison.json"
    torch_path.write_text(json.dumps(_record("mace-torch")))
    symmetrix_path.write_text(json.dumps(_record("symmetrix", force_shift=1.0e-2)))
    args = SimpleNamespace(
        torch_record=torch_path,
        symmetrix_record=symmetrix_path,
        output=output,
        energy_atol_per_atom=2.0e-4,
        force_atol=5.0e-3,
        stress_atol=5.0e-4,
        enforce=True,
    )

    assert benchmark.compare(args) == 1


def test_compare_enforces_zero_symmetrix_stress_for_isolated_systems(tmp_path):
    torch_record = _record("mace-torch")
    symmetrix_record = _record("symmetrix")
    torch_record["systems"][0]["snapshot"]["stress_eV_per_A3_voigt"][0] = 0.1
    torch_path = tmp_path / "torch.json"
    symmetrix_path = tmp_path / "symmetrix.json"
    output = tmp_path / "comparison.json"
    torch_path.write_text(json.dumps(torch_record))
    symmetrix_path.write_text(json.dumps(symmetrix_record))
    args = SimpleNamespace(
        torch_record=torch_path,
        symmetrix_record=symmetrix_path,
        output=output,
        energy_atol_per_atom=2.0e-4,
        force_atol=5.0e-3,
        stress_atol=5.0e-4,
        enforce=True,
    )

    assert benchmark.compare(args) == 0
    result = json.loads(output.read_text())
    assert result["systems"][0]["correctness"]["stress_reference"] == (
        "isolated_exact_zero"
    )

    symmetrix_record["systems"][0]["snapshot"]["stress_eV_per_A3_voigt"][0] = 1.0e-3
    symmetrix_path.write_text(json.dumps(symmetrix_record))
    assert benchmark.compare(args) == 1


def test_cuda_worker_rejects_competing_process_before_loading_calculator(
    tmp_path, monkeypatch
):
    model = tmp_path / "model.pt"
    model.write_bytes(b"not loaded")
    monkeypatch.setattr(
        benchmark,
        "_nvidia_smi",
        lambda: {"gpu": "RTX test", "competing_processes": "123, python, 1024"},
    )
    args = SimpleNamespace(
        model=model,
        systems="molecule_h2o",
        device="cuda",
        allow_busy_gpu=False,
    )

    with pytest.raises(SystemExit, match="active compute processes"):
        benchmark.run(args)
