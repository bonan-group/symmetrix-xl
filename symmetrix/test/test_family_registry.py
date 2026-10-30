"""Focused coverage for the declared family registry and the dispatch it feeds.

The registry replaces the frontend's family branching, so this module checks
both halves of that contract:

* **The declaration.**  Every retained family declares one entry beside its own
  modules, that entry declares the whole family surface, its detection keys are
  disjoint from every other family's, and its evaluator table answers every
  precision and execution space a frontend can request.
* **The dispatch.**  Each family's calculator resolves the class its own
  declaration names for its representative fixture, the declared precedence
  order decides an overlapping key, and an unknown key is still rejected with
  the frontend's own message.

Symbol resolution is part of the declaration check: a declared schema,
extractor, generator, or artifact adapter that no longer exists is a broken
declaration even before a consumer resolves it.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import symmetrix.family_registry as registry_module
from symmetrix import NequIPCalculator, Symmetrix

FIXTURE_DIR = Path(__file__).parent / "data"
OAM_FIXTURE = FIXTURE_DIR / "nequip" / "oam_s_v2_minimal.json"

#: Every shipped family with the module that declares it, in declared
#: precedence order: the compact frontend resolved the nonlinear family first,
#: then the compact model family, then the PolarMACE family.
SHIPPED = (
    ("MACE_Nonlinear", "mace_nonlinear_family"),
    ("MACE", "mace_family"),
    ("MACE_Polar", "mace_polar_family"),
    ("OAM", "oam_family"),
)

_PRECISIONS = frozenset(("float32", "float64"))


def _native():
    from symmetrix import symmetrix as native

    return native


def _native_class(name):
    native = _native()
    evaluator = getattr(native, name, None)
    if evaluator is None:
        pytest.skip(f"{name} is unavailable in this extension")
    return evaluator


class _RecordingNative:
    """A native namespace that provides only the classes it is given."""

    def __init__(self, class_names=()):
        self._class_names = set(class_names)
        self.initialized = False
        self.init_calls = 0

    def _kokkos_is_initialized(self):
        return self.initialized

    def _init_kokkos(self):
        self.init_calls += 1
        self.initialized = True

    def __getattr__(self, name):
        if name in self._class_names:
            return type(name, (), {})
        raise AttributeError(name)


def _synthetic_entry(key, precedence, model_type):
    """Build a minimal entry for the registry's own resolution rules."""

    symbol = registry_module.Symbol("family_registry")
    return registry_module.FamilyEntry(
        key=key,
        precedence=precedence,
        detection=registry_module.Detection(model_types=(model_type,)),
        evaluators=(
            registry_module.EvaluatorVariant(
                kokkos=True, precision="float64", class_name="Synthetic"
            ),
        ),
        modes=registry_module.ModeVocabulary(
            aliases={"direct": "direct"}, primary=("direct",)
        ),
        # The registry validates every entry's routings, so the minimal entry
        # declares the two routings through one surface of its own.
        routings=(
            registry_module.Routing(
                key="direct", native_mode="direct", surface="synthetic"
            ),
            registry_module.Routing(
                key="reference", native_mode="generic", surface="synthetic"
            ),
        ),
        schema=symbol,
        extractor=symbol,
        contract=registry_module.ContractDeclaration(builder=symbol),
        generated_contract=registry_module.ContractDeclaration(builder=symbol),
        codegen=(),
        adapters=(),
        capabilities=registry_module.CapabilityDeclaration(family=key),
        surfaces=(
            registry_module.EvaluatorSurface(key="synthetic", methods=("Synthetic",)),
        ),
    )


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_declared_entries_are_the_shipped_families_in_precedence_order():
    import symmetrix.mace_family as compact_module
    import symmetrix.mace_nonlinear_family as nonlinear_module
    import symmetrix.mace_polar_family as polar_module
    import symmetrix.oam_family as oam_module

    entries = registry_module.declared_entries()

    assert [entry.key for entry in entries] == [key for key, _ in SHIPPED]
    assert {entry.key: entry for entry in entries} == {
        compact_module.FAMILY_ENTRY.key: compact_module.FAMILY_ENTRY,
        nonlinear_module.FAMILY_ENTRY.key: nonlinear_module.FAMILY_ENTRY,
        polar_module.FAMILY_ENTRY.key: polar_module.FAMILY_ENTRY,
        oam_module.FAMILY_ENTRY.key: oam_module.FAMILY_ENTRY,
    }


def test_every_declared_surface_is_implemented_by_the_class_it_names():
    """A declared surface is a promise the family's own evaluators must keep.

    The frontend now fails on a missing prepared-path method instead of falling
    back, so a surface that names a method no evaluator binds would be a
    declaration bug rather than a runtime surprise.  A surface scoped to one
    execution space is checked against that space's evaluators only, so a
    family can scope a surface to the evaluator that implements it.
    """

    native = _native()
    resolved = 0
    for entry in registry_module.declared_entries():
        assert entry.surfaces, entry.key
        keys = [surface.key for surface in entry.surfaces]
        assert len(set(keys)) == len(keys), entry.key
        for variant in entry.evaluators:
            evaluator_class = getattr(native, variant.class_name, None)
            if evaluator_class is None:
                continue
            resolved += 1
            for surface in entry.surfaces:
                if surface.kokkos is not None and surface.kokkos != variant.kokkos:
                    continue
                variant_model_types = variant.model_types or (None,)
                if not any(
                    surface.admits_model_type(model_type)
                    for model_type in variant_model_types
                ):
                    continue
                assert surface.methods, (entry.key, surface.key)
                missing = [
                    name
                    for name in surface.methods
                    if not callable(getattr(evaluator_class, name, None))
                ]
                assert not missing, (
                    entry.key,
                    variant.class_name,
                    surface.key,
                    missing,
                )
    if not resolved:
        pytest.skip("the native backend provides none of the declared evaluators")


def test_every_declared_surface_scope_matches_a_declared_evaluator():
    """A surface scoped to one space must have an evaluator in that space."""

    for entry in registry_module.declared_entries():
        spaces = {variant.kokkos for variant in entry.evaluators}
        for surface in entry.surfaces:
            if surface.kokkos is not None:
                assert surface.kokkos in spaces, (entry.key, surface.key)
            if surface.model_types:
                evaluator_model_types = {
                    model_type
                    for variant in entry.evaluators
                    if surface.kokkos is None or variant.kokkos == surface.kokkos
                    for model_type in variant.model_types
                }
                assert set(surface.model_types) <= evaluator_model_types, (
                    entry.key,
                    surface.key,
                )


def test_declared_surface_readiness_keys_are_declared_capability_keys():
    """A surface may only gate on a capability key the native query reports."""

    for entry in registry_module.declared_entries():
        for surface in entry.surfaces:
            undeclared = sorted(
                set(surface.readiness) - set(registry_module.CAPABILITY_KEYS)
            )
            assert not undeclared, (entry.key, surface.key, undeclared)


def test_every_entry_declares_the_whole_family_surface():
    # The compact extractors import the training framework, as every suite that
    # exercises them imports it first; resolving a declared symbol is the
    # declaration's own promise and nothing more.
    with contextlib.suppress(ImportError):
        import mace.calculators  # noqa: F401

    for entry in registry_module.declared_entries():
        assert entry.detection.declared, entry.key
        assert entry.modes.aliases and entry.modes.primary, entry.key
        assert entry.capabilities.family, entry.key
        assert entry.contract.builder is not None, entry.key
        assert entry.generated_contract.builder is not None, entry.key
        assert entry.codegen, entry.key
        assert entry.adapters, entry.key
        for symbol in (
            entry.schema,
            entry.extractor,
            entry.contract.builder,
            entry.generated_contract.builder,
            *entry.codegen,
            *entry.adapters,
        ):
            assert registry_module.resolve_symbol(symbol) is not None, (
                entry.key,
                symbol,
            )
        # A declared artifact adapter is a promise too: the routing it serves
        # must exist and its entry point must resolve, or a frontend that
        # committed to the routing would fail at preparation for a declaration
        # reason instead of an artifact reason.
        for adapter in entry.artifacts:
            assert adapter.target, (entry.key, adapter)
            assert registry_module.routing_of(entry, adapter.routing) is not None, (
                entry.key,
                adapter.routing,
            )
            assert registry_module.resolve_symbol(adapter.prepare) is not None, (
                entry.key,
                adapter.prepare,
            )


def test_polar_declares_distinct_whole_and_generated_contracts():
    """A polar backbone artifact must not replace its whole-program identity."""

    registry = registry_module.declared_registry()
    polar = registry.detect(model_type="MACE_Polar")
    mh1 = registry.detect(model_type="MACE_Nonlinear")

    assert polar.contract.json_key == "POLAR_BACKBONE"
    assert polar.contract.specialization is None
    assert polar.generated_contract.json_key == "MH1_UVU"
    assert polar.generated_contract.specialization == "mh1"
    assert registry_module.resolve_symbol(polar.contract.builder).__name__ == (
        "make_execution_polar_backbone_contract"
    )
    assert registry_module.resolve_symbol(
        polar.generated_contract.validator
    ).__name__ == ("validate_execution_polar_mh1_v4_contract_for_model")
    assert (
        registry_module.resolve_symbol(polar.generated_contract.builder).__name__
        == "make_execution_mh1_contract"
    )

    assert mh1.contract.json_key == "MH1_UVU"
    assert mh1.contract.specialization == "mh1"
    assert registry_module.resolve_symbol(
        mh1.generated_contract.validator
    ).__name__ == ("validate_execution_mh1_v4_contract_for_model")
    assert mh1.generated_contract == mh1.contract


def test_only_polar_declares_a_lazily_resolved_frontend_adapter():
    """A model-specific frontend lifecycle must be family-owned and lazy."""

    entries = registry_module.declared_entries()
    polar = next(entry for entry in entries if entry.key == "MACE_Polar")

    assert polar.frontend_adapter is not None
    adapter = registry_module.resolve_symbol(polar.frontend_adapter)
    assert adapter.__module__ == "symmetrix.mace_polar_frontend"
    assert callable(getattr(adapter, "evaluate", None))
    assert all(
        entry.frontend_adapter is None for entry in entries if entry.key != "MACE_Polar"
    )


# The two neutralizers the OAM CUDA adapter needs: a contract that reaches the
# adapter's own target check, and a resolved CUDA JIT backend so the adapter
# never probes the toolchain of the machine running the suite.
def _oam_contract(_path):
    """Return a contract the OAM CUDA generator reaches its target check with."""

    return {"architecture": "oam"}


def _resolved_cuda_backend(_request=None):
    """Return a resolved CUDA JIT backend without probing the toolchain."""

    return "nvrtc"


#: The targets every declared artifact adapter is *not* asked to serve, keyed by
#: the ``(family, routing, target)`` it declares, with the module-level entry
#: points a wrong-target call must get past to reach the adapter's own target
#: check.  An adapter that binds its target after it reads its model file would
#: otherwise fail on the absent file instead of on the target the caller asked
#: for, which is why the OAM CUDA adapter's file read is neutralized.  The wrong
#: target is always one another adapter of the same routing serves, so the
#: request is a real routing request and not a typo.
WRONG_TARGET_CALLS = {
    ("MACE_Nonlinear", "direct", "host"): ("cuda", ()),
    ("MACE_Polar", "direct", "host"): ("cuda", ()),
    ("MACE", "direct", "host"): ("cuda", ()),
    ("MACE", "direct", "cuda"): ("host", ()),
    ("MACE", "direct", "hip"): ("host", ()),
    ("OAM", "direct", "host"): ("cuda", ()),
    ("OAM", "direct", "cuda"): (
        "host",
        (
            ("symmetrix.oam_cuda_artifact", "load_nequip_oam_data", _oam_contract),
            (
                "symmetrix.oam_cuda_artifact",
                "select_execution_cuda_jit_backend",
                _resolved_cuda_backend,
            ),
        ),
    ),
}

#: The adapters MACE and MH-1 must declare: their direct routing runs a prepared
#: artifact, so the host target and -- for MH-0 -- both device targets the
#: frontend can resolve are part of the declared surface.
REQUIRED_ADAPTERS = {
    "MACE": {("direct", "host"), ("direct", "cuda"), ("direct", "hip")},
    "MACE_Nonlinear": {("direct", "host")},
    # The polar family's direct routing runs the backbone's generated node
    # program, so it declares the same host entry point the MH-1 family does.
    "MACE_Polar": {("direct", "host")},
}


def _declared_adapter_keys():
    return {
        (entry.key, adapter.routing, adapter.target)
        for entry in registry_module.declared_entries()
        for adapter in entry.artifacts
    }


def test_every_declared_adapter_honours_the_declared_calling_convention(
    monkeypatch,
):
    """``prepare(model_path, precision=..., target=...)`` refuses a foreign target.

    A frontend resolves an adapter by the routing and target it declared and
    then calls the entry point with that same target, so an entry point that
    silently served another target would prepare the wrong artifact.  Every
    declared adapter therefore rejects a target it does not serve, naming it,
    and does so before it touches the model file: no fixture is needed.
    """

    assert _declared_adapter_keys() == set(WRONG_TARGET_CALLS)

    for (key, routing, target), (wrong, patches) in WRONG_TARGET_CALLS.items():
        entry = next(
            declared
            for declared in registry_module.declared_entries()
            if declared.key == key
        )
        adapter = registry_module.artifact_adapter_of(
            entry, routing=routing, target=target
        )
        assert adapter is not None, (key, routing, target)
        for module_path, attribute, replacement in patches:
            monkeypatch.setattr(
                importlib.import_module(module_path), attribute, replacement
            )
        prepare = registry_module.resolve_symbol(adapter.prepare)

        with pytest.raises(RuntimeError, match=re.escape(wrong)):
            prepare("missing-model.json", precision="float64", target=wrong)


def test_mace_and_mh1_declare_their_direct_artifact_adapters():
    """The compact families declare the artifact entry points their routing runs."""

    entries = {entry.key: entry for entry in registry_module.declared_entries()}
    for key, required in REQUIRED_ADAPTERS.items():
        declared = {
            (adapter.routing, adapter.target) for adapter in entries[key].artifacts
        }
        assert required <= declared, (key, sorted(declared))


def test_every_model_type_and_precision_is_answered_by_exactly_one_variant():
    """One evaluator stack answers every detected model type and precision once."""

    for entry in registry_module.declared_entries():
        model_types = entry.detection.model_types or (None,)
        for model_type in model_types:
            answered = [
                variant.precision
                for variant in entry.evaluators
                if variant.admits_model_type(model_type)
            ]
            assert set(answered) == _PRECISIONS, (entry.key, model_type)
            assert len(answered) == len(set(answered)), (entry.key, model_type)
        # The serial evaluators retired with the one-stack contract, so no
        # declared variant names a non-Kokkos execution space.
        assert all(variant.kokkos for variant in entry.evaluators), entry.key


def test_evaluator_variants_resolve_the_class_the_entry_declares():
    for entry in registry_module.declared_entries():
        for variant in entry.evaluators:
            for model_type in variant.model_types or (None,):
                native = _RecordingNative([variant.class_name])
                resolved = registry_module.resolve_evaluator_class(
                    entry,
                    model_type=model_type,
                    dtype=variant.precision,
                    native=native,
                )
                assert resolved.__name__ == variant.class_name
                # There is one evaluator stack, so resolving any variant
                # initializes the execution space before returning its class.
                assert variant.kokkos
                assert native.init_calls == 1


@pytest.mark.parametrize(
    ("model_type", "dtype", "expected"),
    [
        ("MACE", "float64", "MACEKokkos"),
        ("MACE", "float32", "MACEKokkosFloat"),
        ("MACEField", "float64", "MACEKokkos"),
        ("MACEField", "float32", "MACEKokkosFloat"),
        ("MACELES", "float64", "MACELESKokkos"),
        ("MACELES", "float32", "MACELESKokkosFloat"),
    ],
)
def test_compact_family_selects_native_evaluator_by_model_type(
    model_type, dtype, expected
):
    native = SimpleNamespace(
        _kokkos_is_initialized=lambda: True,
        MACEKokkos=type("MACEKokkos", (), {}),
        MACEKokkosFloat=type("MACEKokkosFloat", (), {}),
        MACELESKokkos=type("MACELESKokkos", (), {}),
        MACELESKokkosFloat=type("MACELESKokkosFloat", (), {}),
    )

    resolved = registry_module.resolve_model_evaluator_class(
        model_type=model_type,
        dtype=dtype,
        native=native,
    )

    assert resolved.__name__ == expected


def test_a_missing_class_reports_the_declared_error_before_initializing():
    for entry in registry_module.declared_entries():
        for variant in entry.evaluators:
            model_type = variant.model_types[0] if variant.model_types else None
            if variant.declared_error is None:
                # A variant that declares no build-support failure resolves the
                # class directly, so the missing attribute surfaces unchanged.
                with pytest.raises(AttributeError):
                    registry_module.resolve_evaluator_class(
                        entry,
                        model_type=model_type,
                        dtype=variant.precision,
                        native=_RecordingNative(),
                    )
                continue
            native = _RecordingNative()
            message = variant.declared_error.format(class_name=variant.class_name)
            with pytest.raises(RuntimeError, match=re.escape(message)):
                registry_module.resolve_evaluator_class(
                    entry,
                    model_type=model_type,
                    dtype=variant.precision,
                    native=native,
                )
            # The declared failure is reported before the execution space is
            # initialized, so no runtime is created for a class the backend
            # cannot serve.
            assert native.init_calls == 0


def test_declared_routings_are_the_two_declared_keys():
    """Every family declares exactly ``direct`` and ``reference``."""

    assert registry_module.ROUTING_KEYS == ("direct", "reference")

    for entry in registry_module.declared_entries():
        keys = tuple(routing.key for routing in entry.routings)
        assert set(keys) == set(registry_module.ROUTING_KEYS), entry.key
        assert len(keys) == len(registry_module.ROUTING_KEYS), entry.key


def test_every_declared_routing_names_a_declared_surface():
    """A routing executes through a surface the family declares."""

    for entry in registry_module.declared_entries():
        for routing in entry.routings:
            surface = registry_module.surface_of(entry, routing.surface)
            assert surface is not None, (entry.key, routing.key, routing.surface)
            assert routing.native_mode, (entry.key, routing.key)


def test_routing_of_resolves_each_declared_routing_and_rejects_others():
    for entry in registry_module.declared_entries():
        for routing in entry.routings:
            resolved = registry_module.routing_of(entry, routing.key)
            assert resolved is routing
        # ``auto`` and an unknown spelling are not routings, which leaves the
        # frontend's own rejection message in charge of reporting them.
        assert registry_module.routing_of(entry, "auto") is None
        assert registry_module.routing_of(entry, "materialized") is None
        assert registry_module.routing_of(entry, "generic") is None


def test_each_family_resolves_its_documented_spellings_to_a_declared_routing():
    """The vocabulary's canonical modes are the routing keys a frontend resolves."""

    compact_modes = {"auto", "direct", "reference"}
    for entry in registry_module.declared_entries():
        expected = compact_modes if entry.key != "OAM" else {"direct", "reference"}
        assert entry.modes.canonical_modes == expected, entry.key
        for spelling in entry.modes.aliases:
            canonical = entry.modes.canonical_mode(spelling)
            if canonical == "auto":
                continue
            assert registry_module.routing_of(entry, canonical) is not None, (
                entry.key,
                spelling,
            )


def test_detection_resolves_each_key_to_its_declared_family():
    registry = registry_module.declared_registry()

    assert registry.detect(model_type="MACE").key == "MACE"
    assert registry.detect(model_type="MACEField").key == "MACE"
    assert registry.detect(model_type="MACE_Nonlinear").key == "MACE_Nonlinear"
    assert registry.detect(model_type="MACE_Polar").key == "MACE_Polar"
    assert registry.detect(architecture="oam").key == "OAM"

    # A key the frontend does not serve stays unresolved, which leaves the
    # frontend's own rejection message in charge of reporting it.
    assert registry.detect(model_type="NequIP") is None
    assert registry.detect(architecture="nequip") is None
    assert registry.detect(architecture="") is None

    # No key is declared twice, so no model can change family with the order.
    keys = [key for entry in registry.entries for key in entry.detection.model_types]
    assert len(keys) == len(set(keys))
    architectures = [
        architecture
        for entry in registry.entries
        for architecture in entry.detection.architectures
    ]
    assert len(architectures) == len(set(architectures))


def test_the_first_declared_entry_wins_an_overlapping_detection_key():
    first = _synthetic_entry("first", 0, "shared")
    second = _synthetic_entry("second", 1, "shared")

    # Precedence decides, not the order the entries were constructed in.
    assert (
        registry_module.FamilyRegistry((second, first)).detect(model_type="shared")
        is first
    )
    assert (
        registry_module.FamilyRegistry((first, second)).detect(model_type="shared")
        is first
    )


def test_a_duplicate_declaration_is_rejected():
    with pytest.raises(ValueError, match="duplicate family keys"):
        registry_module.FamilyRegistry(
            (
                _synthetic_entry("same", 0, "first"),
                _synthetic_entry("same", 1, "second"),
            )
        )
    with pytest.raises(ValueError, match="duplicate family precedences"):
        registry_module.FamilyRegistry(
            (
                _synthetic_entry("first", 0, "first"),
                _synthetic_entry("second", 0, "second"),
            )
        )


# ---------------------------------------------------------------------------
# The dispatch
# ---------------------------------------------------------------------------


def _mace_calculator(model_path):
    """Build the compact family's calculator for the shared compact MACE model."""

    _native_class("MACEKokkos")
    return Symmetrix(model_path, dtype="float64")


def _mh1_model_path(tmp_path):
    """Write the generalized MH-1 fixture and return its path."""

    try:
        from test_mh1 import _make_generalized_mh1
    except Exception as error:  # pragma: no cover - missing mace-torch or mace-field
        pytest.skip(f"the MH-1 model builder is unavailable: {error}")

    _model, data = _make_generalized_mh1((4, 2, 4, 2, []))
    model_path = tmp_path / "mh1.json"
    model_path.write_text(json.dumps(data, separators=(",", ":")))
    return model_path


def _mh1_calculator(tmp_path, dtype="float64", model_path=None):
    """Build the nonlinear family's Kokkos calculator from its model builder."""

    if model_path is None:
        model_path = _mh1_model_path(tmp_path)
    _native_class(
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    return Symmetrix(model_path, dtype=dtype)


@pytest.mark.parametrize(
    "option",
    (
        {"polar_electrostatics": "auto"},
        {"polar_pme_mesh": (64, 64, 64)},
        {"polar_pme_order": 6},
        {"total_charge": 0.0},
        {"total_spin": 0.0},
        {"fermi_level": 0.0},
        {"electric_field": (0.0, 0.0, 0.0)},
    ),
)
def test_nonpolar_family_rejects_explicit_polar_options(tmp_path, option):
    """Even an explicit neutral Polar value must not be silently ignored."""

    model_path = _mh1_model_path(tmp_path)
    with pytest.raises(ValueError, match="MACE_Polar|Polar-only"):
        Symmetrix(
            model_path,
            dtype="float64",
            streamed_edges="reference",
            **option,
        )


def test_nonpolar_family_keeps_omitted_polar_defaults_inert(tmp_path):
    """Omitted Polar options must not alter ordinary family construction."""

    calculator = Symmetrix(
        _mh1_model_path(tmp_path),
        dtype="float64",
        streamed_edges="reference",
    )
    assert calculator._family_frontend_adapter is None


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("total_charge", 1.0),
        ("total_spin", 1.0),
        ("fermi_level", -2.0),
        ("polar_electrostatics", "realspace"),
        ("polar_pme_mesh", (32, 32, 32)),
        ("polar_pme_order", 5),
        ("electric_field", (0.0, 0.0, 0.0)),
    ),
)
def test_nonpolar_family_rejects_post_construction_polar_mutation(
    tmp_path, name, value
):
    """A later Polar-only assignment must not become silent inert state."""

    calculator = _mh1_calculator(tmp_path)
    before = getattr(calculator, name)
    calculator.results = {"energy": 3.0}

    with pytest.raises(ValueError, match="MACE_Polar|Polar-only"):
        setattr(calculator, name, value)

    assert getattr(calculator, name) == before
    assert calculator.results == {"energy": 3.0}


def test_macefield_keeps_post_construction_electric_field_mutation():
    """Field coupling, unlike Polar-only inputs, remains valid for MACEField."""

    calculator = object.__new__(Symmetrix)
    calculator._family_frontend_adapter = None
    calculator._native_model_type = "MACEField"
    calculator.evaluator = SimpleNamespace(has_field_coupling=True)
    calculator._electric_field = None
    calculator.results = {"energy": 3.0}

    calculator.electric_field = (0.1, -0.2, 0.3)

    assert calculator.electric_field == pytest.approx((0.1, -0.2, 0.3))
    assert calculator.results == {}


def _oam_calculator():
    """Build the OAM family's Kokkos calculator for the checked-in fixture."""

    _native_class("NequIPOAMKokkos")
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the OAM host artifact")
    return NequIPCalculator(OAM_FIXTURE, dtype="float64")


def test_the_native_capability_key_set_equals_the_declared_key_set(
    tmp_path, compact_mace_model
):
    """Every family's native capability query publishes exactly the declared keys.

    The key set is one cross-language contract: the Python declarations, the
    native bindings, and the generated support matrix all read it, so a key
    added on one side only must fail here instead of silently making the
    matrix's capability rows incomplete.
    """

    calculators = (
        ("MACE", _mace_calculator(compact_mace_model)),
        ("MACE_Nonlinear", _mh1_calculator(tmp_path)),
        ("OAM", _oam_calculator()),
    )

    for key, calculator in calculators:
        entry = registry_module.declared_registry().by_key(key)
        reported = registry_module.capabilities_of(entry, calculator.evaluator)
        assert reported is not None, key
        assert set(reported) == set(registry_module.CAPABILITY_KEYS), key


def test_mace_calculator_constructs_the_class_its_family_declares(
    compact_mace_model,
):
    entry = registry_module.declared_registry().detect(model_type="MACE")
    expected = registry_module.resolve_evaluator_class(
        entry, model_type="MACE", dtype="float64", native=_native()
    )

    calculator = _mace_calculator(compact_mace_model)

    assert isinstance(calculator.evaluator, expected)
    assert (
        registry_module.program_family_of(entry, calculator.evaluator)
        == entry.capabilities.family
    )


def test_macefield_calculator_constructs_the_class_its_family_declares(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    data = extract_mace_data(
        macefield_model_path, species=[7, 13], head="mp-dielectric"
    )
    assert data["model_type"] == "MACEField"
    model_path = tmp_path / "macefield.json"
    model_path.write_text(json.dumps(data))

    entry = registry_module.declared_registry().detect(model_type="MACEField")
    assert entry.key == "MACE"
    expected = registry_module.resolve_evaluator_class(
        entry, model_type="MACEField", dtype="float64", native=_native()
    )

    calculator = Symmetrix(model_path, dtype="float64")

    assert isinstance(calculator.evaluator, expected)
    assert (
        registry_module.program_family_of(entry, calculator.evaluator)
        == entry.capabilities.family
    )
    calculator.results = {"energy": 3.0}
    calculator.electric_field = (0.1, -0.2, 0.3)
    assert calculator.electric_field == pytest.approx((0.1, -0.2, 0.3))
    assert calculator.results == {}


def test_mh1_calculator_constructs_the_class_its_family_declares(tmp_path):
    entry = registry_module.declared_registry().detect(model_type="MACE_Nonlinear")
    model_path = _mh1_model_path(tmp_path)
    for dtype in ("float64", "float32"):
        expected = registry_module.resolve_evaluator_class(
            entry, model_type="MACE_Nonlinear", dtype=dtype, native=_native()
        )
        calculator = _mh1_calculator(tmp_path, dtype=dtype, model_path=model_path)
        assert isinstance(calculator.evaluator, expected)
        assert (
            registry_module.program_family_of(entry, calculator.evaluator)
            == entry.capabilities.family
        )


def test_mh1_node_program_preserves_historical_python_kind(tmp_path):
    """The internal enum split must not change the public program vocabulary."""

    calculator = Symmetrix(
        _mh1_model_path(tmp_path),
        dtype="float64",
        streamed_edges="non-compiled",
    )
    evaluator = calculator.evaluator
    try:
        nodes = [
            node
            for node in evaluator.family_program["operators"]
            if node["capability_key"] == "node_program"
        ]
        bindings = [
            binding
            for binding in evaluator.family_binding_report["bindings"]
            if binding["capability_key"] == "node_program"
        ]

        assert nodes
        assert len(bindings) == len(nodes)
        assert {node["kind"] for node in nodes} == {"custom"}
        assert {binding["kind"] for binding in bindings} == {"custom"}
    finally:
        del evaluator
        del calculator


def test_mh1_declares_both_nonlinear_and_canonical_product_codegen():
    """Dropping common codegen would hide the M0/M1 composition boundary."""
    entry = registry_module.declared_registry().detect(model_type="MACE_Nonlinear")
    assert {(symbol.module, symbol.name) for symbol in entry.codegen} == {
        ("mace_nonlinear_jit_codegen", None),
        ("generated_operator_codegen", None),
    }


def test_oam_calculator_constructs_the_class_its_family_declares():
    entry = registry_module.declared_registry().detect(architecture="oam")
    expected = registry_module.resolve_evaluator_class(
        entry, model_type=None, dtype="float64", native=_native()
    )

    calculator = _oam_calculator()

    assert isinstance(calculator.evaluator, expected)
    assert (
        registry_module.program_family_of(entry, calculator.evaluator)
        == entry.capabilities.family
    )


def test_unknown_model_type_is_rejected_exactly_as_before(tmp_path):
    model_path = tmp_path / "unknown.json"
    model_path.write_text(json.dumps({"model_type": "NequIP"}))

    with pytest.raises(
        ValueError, match=re.escape("Unsupported Symmetrix model_type 'NequIP'.")
    ):
        Symmetrix(model_path)
