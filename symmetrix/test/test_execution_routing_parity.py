"""Per-family routing parity: exactly two routings, one of them artifact-free.

Stage 9 consolidated every family onto two routings, declared in the family
registry and read through ``family_registry.routing_of``:

* ``direct`` runs the prepared plan against a contract-bound generated (T3)
  artifact, and
* ``reference`` runs the generic implementation through the whole-evaluator
  entry point on the default host execution space with no artifact at all.

This module drives both routings through the public frontends for every family
that declares a routing case to ``conformance.registry`` and asserts the
properties the consolidation exists for:

* the two routings agree, within tolerances derived from the golden corpus
  captured from the removed CPU-native evaluators before their deletion
  (``.superpowers/.../task-18-golden/serial-vs-reference.json``): the
  consolidated reference matched those evaluators to ``<= 1e-15`` relative for
  the float64 compact model, ``<= 1.6e-12`` relative on the nonlinear family's
  float64 energies and ``<= 4.4e-7`` relative on its directed forces, and
  ``<= 1.2e-6`` relative for the float32 compact model.  The gate below is
  therefore the family's own prepared-path tolerance for float64 and the
  single-precision tolerance for float32; each family's case module records the
  agreement its own paths were measured at;
* a ``reference`` request evaluates with an empty cache and RTC disabled, which is
  the host admission invariant, while a ``direct`` request in the same environment
  fails rather than falling back; and
* the compact frontend refuses its retired keyword in both the positional and the
  keyword form.

The graph convention is the shared one: directed edges indexed by receiver,
``sources`` holding the neighbor node, and ``xyz = positions[sources] -
positions[receivers] + shifts @ cell``.

Each family's model, structure, frontend construction, reference-routing
precondition, and artifact-preparation observer live in that family's own
``test_family_<family>_conformance_cases`` module, so this suite names no family.
"""

from __future__ import annotations

import numpy as np
import pytest
from symmetrix import Symmetrix

from conformance.registry import ROUTING_CASES, crystal

# The prepared-path parity gate per precision, as the cross-family conformance
# suite states it for the same contracts.
ROUTING_TOLERANCE = {
    "float64": dict(rtol=1e-9, atol=1e-8),
    "float32": dict(rtol=1e-3, atol=1e-4),
}

CASE_IDS = [case.key for case in ROUTING_CASES]


@pytest.fixture(scope="session")
def routing_models(tmp_path_factory, request):
    """The session model each declared routing case evaluates.

    Every family builds its own model through its own declaration, so this
    fixture names no family and no model.
    """
    root = tmp_path_factory.mktemp("routing-parity")
    return {case.key: case.build_model(request, root) for case in ROUTING_CASES}


@pytest.fixture(scope="session", autouse=True)
def routing_kokkos_runtime():
    """One Kokkos initialization per session, before any evaluator is built."""

    from symmetrix import symmetrix as native

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    return native


def _calculator(case, model_path, *, routing, dtype):
    """Build the calculator for one case, routing, and precision."""

    atoms = crystal(case.structure)
    calculator = case.calculator(model_path, dtype=dtype, routing=routing)
    atoms.calc = calculator
    return atoms, calculator


def _evaluate(atoms, properties):
    """The requested properties of one structure, in a fixed order."""

    result = {}
    if "energy" in properties:
        result["energy"] = float(atoms.get_potential_energy())
    if "forces" in properties:
        result["forces"] = np.asarray(atoms.get_forces(), dtype=np.float64)
    if "stress" in properties:
        result["stress"] = np.asarray(atoms.get_stress(), dtype=np.float64)
    return result


def _assert_close(label, expected, actual, tolerance):
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
        err_msg=f"{label} does not match within {tolerance}",
    )


@pytest.mark.parametrize("case", ROUTING_CASES, ids=CASE_IDS)
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_the_two_routings_agree_per_family(case, dtype, routing_models):
    """``direct`` and ``reference`` are the same model, evaluated two ways."""

    model_path = routing_models[case.key]
    tolerance = ROUTING_TOLERANCE[dtype]
    direct_atoms, direct = _calculator(case, model_path, routing="direct", dtype=dtype)
    reference_atoms, reference = _calculator(
        case, model_path, routing="reference", dtype=dtype
    )

    assert direct.streamed_edges == "direct"
    assert reference.streamed_edges == "reference"

    # Stress parity holds only where both routings declare it: a reference entry
    # point that leaves no completed generation to reduce against declares
    # `reduces_stress=False` and does not advertise it.
    properties = ["energy", "forces"]
    if "stress" in direct.implemented_properties and (
        "stress" in reference.implemented_properties
    ):
        properties.append("stress")
    expected = _evaluate(direct_atoms, properties)
    actual = _evaluate(reference_atoms, properties)

    label = f"{case.key}/{dtype}"
    _assert_close(f"{label} energy", expected["energy"], actual["energy"], tolerance)
    _assert_close(f"{label} forces", expected["forces"], actual["forces"], tolerance)
    if "stress" in properties:
        _assert_close(
            f"{label} stress", expected["stress"], actual["stress"], tolerance
        )


@pytest.mark.parametrize("case", ROUTING_CASES, ids=CASE_IDS)
def test_reference_routing_evaluates_without_any_artifact(
    case, routing_models, tmp_path, monkeypatch
):
    """A ``reference`` request must not depend on compiled execution.

    The cache is empty and RTC is disabled, so no artifact can be prepared or
    loaded.  The reference routing must still evaluate, and a ``direct`` request
    in the same environment must fail rather than fall back to it.
    """

    model_path = routing_models[case.key]
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "empty-cache"))
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "none")
    prepared = []
    if case.preparer_guard is not None:
        case.preparer_guard(monkeypatch, prepared)

    atoms, calculator = _calculator(
        case, model_path, routing="reference", dtype="float64"
    )
    assert calculator.streamed_edges == "reference"
    assert np.isfinite(float(atoms.get_potential_energy()))
    if case.preparer_guard is not None:
        # A reference request reaches the declared preparer exactly this often.
        assert prepared == list(case.reference_preparer_calls)

    with pytest.raises(RuntimeError):
        _calculator(case, model_path, routing="direct", dtype="float64")
    if case.preparer_guard is not None:
        assert prepared == list(case.direct_preparer_calls)


def test_symmetrix_rejects_use_kokkos_in_both_forms(routing_models):
    """The retired third positional slot and the keyword are both refused."""

    compact = [case for case in ROUTING_CASES if case.compact_frontend]
    assert compact, "no case declares the compact frontend"
    for case in compact:
        model_path = routing_models[case.key]
        for call in (
            lambda: Symmetrix(model_path, "float64", True),
            lambda: Symmetrix(model_path, "float64", False),
            lambda: Symmetrix(model_path, "float64", True, "direct"),
            lambda: Symmetrix(model_path, use_kokkos=True),
            lambda: Symmetrix(model_path, dtype="float64", use_kokkos=False),
        ):
            with pytest.raises(TypeError, match="use_kokkos has been retired"):
                call()
