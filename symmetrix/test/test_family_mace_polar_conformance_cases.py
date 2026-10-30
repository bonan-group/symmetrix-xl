"""MACE-Polar conformance cases for the family-owned whole-model lifecycle.

The shared conformance ``Adapter`` drives the older neutral prepared lifecycle:
``prepare_graph``/``prepare_geometry``/``compute_prepared`` followed by generic
force and stress reductions.  MACE Polar deliberately exposes a different
scientific boundary.  Its thin frontend prepares compact topology and geometry,
then invokes the family-owned ``whole_model`` surface once with charge, spin,
field, PBC, and requested public properties.  That native transaction owns the
backbone forward, Polar chain and electrostatics, seeded backbone reverse, and
public result composition.

The shared ``Adapter`` cannot yet supply those graph-level inputs or consume the
whole-model result, so this module must not fabricate an ``INSTANCE_SPECS``
entry.  Focused assertions below instead pin the actual declaration boundary,
the direct/reference routes, public sibling-evaluator isolation, and the native
single-system admission.  Numerical route parity and early-allocation rejection
remain exercised by ``test_mace_polar_frontend.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from conformance.registry import FamilyConformance, register_family
from conformance.surface import EmptyPathCase, _native
from polar_oracle_checkout import required_checkout

#: The size the family's gates and this module use: the M checkpoint is the one
#: the frozen end-to-end records and the frontend suite pin.
MODEL_SIZE = "M"

#: The pinned oracle checkouts the released checkpoints' own module state refers
#: to: unpickling a checkpoint needs its ``graph_longrange`` classes importable.
_ORACLE_CHECKOUT_ENVIRONMENTS = (
    "SYMMETRIX_POLAR_ORACLE_MACE",
    "SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE",
)

#: The extracted model each size's evaluator is built from, memoized: the
#: readiness case and the declaration test share one extraction per session.
_MODELS: dict[str, Path] = {}


def _checkpoint_path(size: str) -> Path:
    """The MACE-POLAR-1 checkpoint, from the environment or the local cache."""

    configured = os.environ.get(f"SYMMETRIX_POLAR_MODEL_{size}")
    path = (
        Path(configured)
        if configured
        else Path("~/.cache/symmetrix/test-models").expanduser()
        / f"MACE-POLAR-1-{size}.model"
    )
    if not path.is_file():
        pytest.skip(
            f"the polar conformance cases need the MACE-POLAR-1-{size} checkpoint "
            f"(set SYMMETRIX_POLAR_MODEL_{size}, or cache it at {path})."
        )
    return path


def _extracted_model(size: str = MODEL_SIZE) -> Path:
    """Extract the released checkpoint to the JSON the family's evaluator reads."""

    if size in _MODELS:
        return _MODELS[size]
    checkpoint = _checkpoint_path(size)
    for environment in _ORACLE_CHECKOUT_ENVIRONMENTS:
        checkout = required_checkout(environment)
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))
    try:
        import torch
        from symmetrix.extract_mace_polar import extract_mace_polar_data
    except ImportError as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"extracting the polar checkpoint requires torch/e3nn: {error}")

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        data = extract_mace_polar_data(model.to(torch.float64))
    finally:
        torch.set_default_dtype(previous_dtype)
    directory = Path(tempfile.mkdtemp(prefix="polar-conformance-"))
    path = directory / f"MACE-POLAR-1-{size}.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    _MODELS[size] = path
    return path


def _polar_evaluator(size: str = MODEL_SIZE):
    """The family's float64 evaluator, parsed from the released checkpoint."""

    native = _native()
    if not hasattr(native, "MACEPolarKokkos"):
        pytest.skip("MACEPolarKokkos is unavailable in this extension")
    return native.MACEPolarKokkos(str(_extracted_model(size)))


def _polar_readiness_evaluator(request):
    """The polar evaluator the shared empty-path readiness case drives."""

    return _polar_evaluator()


def _build_models(root: Path) -> dict:
    """No session model: the family declares no admitted instance.

    ``build_models`` materializes the models a family's *instances* prepare in,
    so a family with no admitted instance has none.  Nothing is extracted here
    on purpose: the session fixture builds every registered family's models
    before any adapter is built, so a checkpoint extraction that skipped would
    skip the whole shared suite, not just this family's cases.  The readiness
    case resolves its own model instead.
    """

    return {}


#: The framework's representation of "does not qualify": no instance is
#: declared, so no tier is claimed.  See the module docstring for the two facts
#: that close this map and the two framework gaps that keep it closed.
INSTANCE_SPECS: dict[str, dict] = {}


register_family(
    FamilyConformance(
        name="MACE_Polar",
        structures={},
        build_models=_build_models,
        instances=INSTANCE_SPECS,
        readiness_cases=(
            EmptyPathCase(name="mace-polar", build=_polar_readiness_evaluator),
        ),
    )
)


def _minimal_topology() -> tuple:
    """One directed edge over two nodes, in the shared graph convention."""

    return 2, (0, 1), (1, 1), (1, 0), (1, 0)


def test_whole_model_lifecycle_is_declared_outside_the_shared_adapter():
    """Do not pretend the generic prepared adapter drives Polar science."""

    from symmetrix import family_registry

    entry = family_registry.declared_registry().by_key("MACE_Polar")
    adapter = family_registry.frontend_adapter_of(entry)
    whole_model = family_registry.surface_of(entry, "whole_model")

    assert adapter.__module__ == "symmetrix.mace_polar_frontend"
    assert whole_model is not None
    assert whole_model.methods == ("_evaluate_polar_prepared",)
    assert {route.key: route.surface for route in entry.routings} == {
        "direct": "whole_model",
        "reference": "whole_model",
    }
    assert INSTANCE_SPECS == {}, (
        "the shared conformance Adapter cannot drive the family-owned "
        "whole_model lifecycle; add an instance only after that framework "
        "boundary is extended"
    )


def test_polar_is_a_public_sibling_with_single_system_native_admission():
    """Polar owns its native API and rejects batches before its transaction."""

    native = _native()
    polar = native.MACEPolarKokkos
    nonlinear = native.MACENonlinearKokkos

    assert not issubclass(polar, nonlinear)
    assert hasattr(polar, "_evaluate_polar_prepared")
    assert not hasattr(nonlinear, "_evaluate_polar_prepared")

    evaluator = _polar_evaluator()
    assert evaluator.supports_streamed_edges is True
    before = evaluator.polar_whole_model_transaction_count
    with pytest.raises(ValueError, match="exactly one input system"):
        evaluator._evaluate_polar_prepared(
            graph_generation=0,
            node_types=[0],
            positions=[[0.0, 0.0, 0.0]],
            cell=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            graph_pbc=[[0, 0, 0]],
            node_graph_index=[0],
            total_charge=[0.0, 0.0],
            total_spin=[0.0, 0.0],
            external_potential=[[0.0] * 4, [0.0] * 4],
            include_forces=False,
            include_stress=False,
            include_dipole=False,
            polar_electrostatics="auto",
        )
    assert evaluator.polar_whole_model_transaction_count == before
