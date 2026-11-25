import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from compact_r1_model import foundation_r1_test_model

from symmetrix import Symmetrix


@pytest.fixture
def mace_model(tmp_path):
    contract_path = (
        Path(__file__).parent
        / "data/execution_contracts/jit_r1_fixture_c4_e3_l0_contract.json"
    )
    model = foundation_r1_test_model(
        json.loads(contract_path.read_text()), atomic_numbers=(1, 8)
    )
    path = tmp_path / "mace.json"
    path.write_text(json.dumps(model))
    return path


def test_kernel_selection_and_binding_on_reference_route(mace_model):
    options = dict(streamed_edges="reference", dtype="float32")
    standard = Symmetrix(mace_model, **options)
    optimized = Symmetrix(mace_model, kernels="optimized", **options)
    assert standard.kernels == "standard"
    assert optimized.kernels == "optimized"
    assert optimized.set(kernels="optimized") == {}
    with pytest.raises(ValueError, match="requires a new Symmetrix calculator"):
        optimized.set(kernels="standard")

    cell = [[5.2, 0, 0], [3.7, 4.8, 0], [1.1, 1.2, 5.3]]
    cluster = Atoms("HO", positions=[[0, 0, 0], [1.2, 0.4, 0.5]], cell=cell, pbc=True)
    displaced = cluster.copy()
    displaced.positions[1, 0] += 0.08
    isolated = Atoms("H", cell=np.eye(3) * 20.0, pbc=True)
    roundoff = 8 * np.finfo(np.float64).eps
    for atoms in (cluster, displaced, isolated, cluster):
        for calc in (standard, optimized):
            calc.calculate(atoms, ["energy", "forces", "stress"])
        standard_rows = standard.kernel_binding_report()["rows"]
        report = optimized.kernel_binding_report()
        assert (
            report["rows"].keys()
            == standard_rows.keys()
            == {"r1_type_map", "h1_contraction", "a1_contraction", "a1_scale_divide"}
        )
        for name, row in report["rows"].items():
            assert row["refusal"] in {
                "requires a device backend",
                "unsupported device architecture for this row",
                "awaiting device qualification",
                "requires direct execution",
            }, (name, row)
            for direction in ("forward", "reverse"):
                assert row[direction] == standard_rows[name][direction]
                assert row[direction] in {"standard", "not_run"}
                assert row[f"{direction}_launches"] == 0, (name, row)
        for key in ("energy", "forces", "stress"):
            # Atomic edge-to-atom assembly can change the summation order.
            reference = standard.results[key]
            assert np.isfinite(optimized.results[key]).all()
            np.testing.assert_allclose(
                optimized.results[key],
                reference,
                rtol=0,
                atol=roundoff * np.max(np.abs(reference)),
            )

    assert standard.kernel_binding_report()["mode"] == "standard"
    assert report["mode"] == "optimized"
    assert report["operators"]["M0"] == "generic_m0"
    assert report["operators"]["R0"] == "generic_r0"
    assert "M1" in report["operators"]
    assert report["rows"]["a1_contraction"]["forward"] == "standard"
    assert report["rows"]["a1_contraction"]["reverse"] == "standard"
    assert report["rows"]["a1_scale_divide"]["forward"] == "not_run"
    assert report["rows"]["a1_scale_divide"]["reverse"] == "not_run"
    h1 = report["rows"]["h1_contraction"]
    assert h1["forward"] == h1["reverse"] == "standard"


def test_invalid_kernel_selection(mace_model):
    with pytest.raises(ValueError, match="kernels must be"):
        Symmetrix(mace_model, kernels="other", streamed_edges="reference")


@pytest.mark.parametrize("model_type", ["MACE_Polar", "MACE_Nonlinear"])
def test_kernel_selection_report_without_family_surface(model_type):
    calc = object.__new__(Symmetrix)
    calc._native_model_type = model_type
    calc.evaluator = object()
    report = calc.kernel_binding_report()
    assert report == {"mode": "standard", "family": model_type, "rows": {}}
