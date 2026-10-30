"""Declared registry of the model families the Python frontend serves.

Every retained model family declares exactly one :class:`FamilyEntry` beside
its own modules, in a module whose name ends with :data:`DECLARATION_SUFFIX`
which exposes that entry as :data:`ENTRY_ATTRIBUTE`.  This module owns the data
model, the resolution rules, and the one execution-mode vocabulary a shared
frontend resolves for every family it serves, and nothing else: it names no
family, imports no family module eagerly, and holds no family-specific value,
so one family's declaration cannot leak into another family's consumers.

An entry is pure declaration.  It states the detection keys that admit a model
to its family, every native evaluator class the family admits per precision and
execution space, the execution-mode vocabulary its frontend accepts, the model
variants that route differently, and lazy symbols for the family's model
schema, checkpoint extractor, execution contract, code generators, artifact
adapters, and native capability queries.  A consumer resolves the family once
and asks it, instead of branching on the model-type or architecture string.

Resolution is ordered: the first entry that admits the requested detection keys
wins, so detection cannot depend on import order, and an unknown key resolves to
``None`` and leaves the frontend's own rejection message in charge.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

__all__ = [
    "CAPABILITY_KEYS",
    "ArtifactAdapter",
    "CapabilityDeclaration",
    "COMPACT_MODE_VOCABULARY",
    "ContractDeclaration",
    "DECLARATION_SUFFIX",
    "Detection",
    "ENTRY_ATTRIBUTE",
    "EvaluatorSurface",
    "EvaluatorVariant",
    "FamilyEntry",
    "FamilyRegistry",
    "frontend_adapter_of",
    "MissingCapabilityError",
    "ModeVocabulary",
    "ROUTING_KEYS",
    "Routing",
    "Symbol",
    "Variant",
    "artifact_adapter_of",
    "capabilities_of",
    "declared_entries",
    "declared_registry",
    "program_family_of",
    "require_surface",
    "resolve_evaluator_class",
    "resolve_symbol",
    "routing_of",
    "surface_of",
    "supports_surface",
    "variant_of",
]

#: Suffix of every module that declares a family entry.
DECLARATION_SUFFIX = "_family"
#: Attribute a declaration module exposes its entry as.
ENTRY_ATTRIBUTE = "FAMILY_ENTRY"

#: The capability keys every native family declaration publishes, in the order
#: the bindings report them.  The native query is one cross-language contract,
#: so its key set is declared once here and read by every consumer: the
#: conformance suites, the generated support matrix, and the readiness gate a
#: frontend applies before it commits to a declared surface.
CAPABILITY_KEYS = (
    "target",
    "precision",
    "reference",
    "portable",
    "generated",
    "stress",
    "field",
    "parameter_gradients",
    "host_artifact",
    "device_artifact",
)

#: The two execution routings every family declares, and the only ones a
#: frontend can resolve a documented mode spelling to.  ``direct`` is the
#: generated (T3) implementation; ``reference`` is single-thread execution of
#: the generic implementation on the default host execution space.  A family
#: that declared a third routing would be reintroducing the per-family debug and
#: oracle paths this contract exists to remove, so the key set is fixed here and
#: validated for every entry.
ROUTING_KEYS = ("direct", "reference")


@dataclass(frozen=True)
class Symbol:
    """A module or module attribute resolved on first use.

    ``module`` is relative to the package the declaration lives in, so an entry
    names its own family's modules without repeating the package name, and
    nothing is imported until a consumer resolves the symbol.
    """

    module: str
    name: str | None = None

    def resolve(self, package: str) -> Any:
        """Import the declaring package's module and return this attribute."""

        resolved: ModuleType = importlib.import_module(f".{self.module}", package)
        return resolved if self.name is None else getattr(resolved, self.name)


@dataclass(frozen=True)
class Detection:
    """The model-file keys that admit a model to one family.

    A frontend supplies only the key namespaces it reads, and an entry admits a
    lookup only through namespaces it declares, so a family detected by its
    architecture marker cannot capture a model-type lookup made by a frontend
    that never reads the architecture.
    """

    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()

    @property
    def declared(self) -> tuple[str, ...]:
        """Every detection key this entry declares, in both namespaces."""

        return (*self.model_types, *self.architectures)

    def admits(self, model_type: str | None, architecture: str | None) -> bool:
        """Report whether this entry admits the requested detection keys."""

        if model_type is not None and model_type not in self.model_types:
            return False
        if architecture is not None and architecture not in self.architectures:
            return False
        return bool(self.model_types or self.architectures)


@dataclass(frozen=True)
class EvaluatorVariant:
    """One native evaluator class a family admits.

    ``kokkos`` selects the execution space and ``precision`` the scalar width.
    ``model_types`` scopes the evaluator to model types within a family; an
    empty tuple admits every model type.
    ``declared_error`` is the build-support failure the frontend reports when
    the backend does not provide ``class_name``, and may interpolate
    ``{class_name}``; a variant that declares none is resolved directly, so a
    backend missing that class fails with the attribute error the lookup
    raises.
    """

    kokkos: bool
    precision: str
    class_name: str
    declared_error: str | None = None
    model_types: tuple[str, ...] = ()

    def admits_model_type(self, model_type: str | None) -> bool:
        """Report whether this evaluator serves ``model_type``."""

        return not self.model_types or model_type in self.model_types


@dataclass(frozen=True)
class ModeVocabulary:
    """The execution-mode spellings one family's frontend accepts.

    ``aliases`` is the single acceptance source: a documented spelling resolves
    through it to the canonical cross-language mode, ``primary`` names the
    spellings that name a canonical mode directly, and ``deprecated`` names the
    spellings whose frontend still warns.
    """

    aliases: Mapping[str, str]
    primary: tuple[str, ...]
    deprecated: tuple[str, ...] = ()

    @property
    def canonical_modes(self) -> frozenset[str]:
        """Every canonical mode the vocabulary can resolve to."""

        return frozenset(self.aliases.values())

    @property
    def alias_spellings(self) -> frozenset[str]:
        """Spellings that resolve to a canonical mode instead of naming it."""

        return frozenset(self.aliases) - frozenset(self.primary)

    def canonical_mode(self, spelling: str) -> str:
        """Resolve a documented spelling; an unknown spelling passes through."""

        return self.aliases.get(spelling, spelling)


# The compact frontend resolves one execution-mode vocabulary for every compact
# family it serves, so the vocabulary is declared beside the data model every
# family's entry is written against, and not inside one family's declaration
# that another family would then have to import.  The table is the single
# acceptance source for the documented spellings over the declared routing
# keys: the primary spellings name a routing directly, and the deprecated
# spellings warn.  A spelling is added or retired here and nowhere else.
#
# Every non-direct spelling resolves to the single ``reference`` routing, which
# is why ``materialized`` is an alias rather than a primary spelling: the
# materialized edge representation is no longer a routing of its own, and the
# original Symmetrix pair-spline format that required it is retired.
COMPACT_MODE_VOCABULARY = ModeVocabulary(
    aliases={
        "auto": "auto",
        "direct": "direct",
        "reference": "reference",
        "factorized": "direct",
        "direct_streamed": "direct",
        "materialized": "reference",
        "generic": "reference",
        "non-compiled": "reference",
        "all_interactions": "reference",
    },
    primary=("auto", "direct", "reference"),
    deprecated=("generic", "all_interactions"),
)


@dataclass(frozen=True)
class Variant:
    """A declared model variant of one family, in precedence order."""

    key: str
    matches: Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class ContractDeclaration:
    """The execution contract a family's models carry.

    ``builder`` owns the contract's construction, ``validator`` owns validation
    against a complete model definition, ``json_key`` is the key a model file
    declares it under, and ``specialization`` names the frontend specialization
    that contract selects.  A family whose contract is the whole model file
    declares no key and no specialization.
    """

    builder: Symbol
    validator: Symbol | None = None
    json_key: str | None = None
    specialization: str | None = None


@dataclass(frozen=True)
class CapabilityDeclaration:
    """The capability surface a family's native evaluators expose.

    ``family`` is the identity the native program declares for this family,
    which ties the Python entry to the native family declaration.
    """

    family: str
    query: str = "family_capabilities"
    program_query: str = "family_program"


@dataclass(frozen=True)
class EvaluatorSurface:
    """One evaluator capability a family's implementations declare.

    ``key`` is the neutral capability name a frontend asks for, ``methods`` are
    the evaluator attributes that implement it, in the order the frontend reads
    them, ``readiness`` names the native capability keys any one of which
    admits the surface, and ``kokkos`` scopes the surface to one execution
    space (``None`` when both spaces' evaluators implement it), and
    ``model_types`` scopes a surface to the variants that implement it (an
    empty tuple admits every model type). A declaration is a promise: a
    frontend that requires the surface fails naming the missing method instead
    of running a fallback branch, and a frontend that chooses between two
    admitted policies asks whether the evaluator implements the surface it
    declares.
    """

    key: str
    methods: tuple[str, ...]
    readiness: tuple[str, ...] = ()
    kokkos: bool | None = None
    model_types: tuple[str, ...] = ()

    def admits_model_type(self, model_type: str | None) -> bool:
        """Report whether this surface belongs to ``model_type``."""

        return not self.model_types or model_type in self.model_types


@dataclass(frozen=True)
class Routing:
    """One of the two execution routings every family declares.

    ``key`` is the neutral routing name a frontend resolves a documented mode
    spelling to and reports as the resolved mode, ``native_mode`` is the
    execution-mode spelling the family's native evaluator accepts for it,
    ``surface`` names the declared :class:`EvaluatorSurface` the routing
    executes through, and ``reduces_stress`` states whether the routing leaves
    a completed evaluation the family's stress reduction can be applied to.
    """

    key: str
    native_mode: str
    surface: str
    reduces_stress: bool = True


@dataclass(frozen=True)
class ArtifactAdapter:
    """One entry point that prepares a family's execution artifact.

    A routing that executes generated code runs an artifact, and the artifact
    is prepared per execution target.  ``routing`` names the routing whose
    execution the prepared artifact serves and ``target`` the native target it
    binds, ``prepare`` is the entry point that prepares (and caches) it, and
    ``artifact_path`` names the attribute of the preparation result that holds
    the loadable library, so a frontend hands an evaluator an artifact without
    knowing the adapter's own result type.  ``manifest_path`` names the result
    attribute of the manifest that publishes the artifact's identity envelope,
    and is ``None`` for an adapter whose artifact publishes none.
    ``evaluator_keyword`` is the keyword the family's evaluator accepts the
    prepared library under (``None`` when the evaluator takes it as its second
    positional argument), and ``identity_keyword`` the keyword it accepts the
    artifact's published identity under (``None`` when it derives that identity
    itself).

    ``prepare`` is called as ``prepare(model_path, precision=..., target=...)``:
    the target is the same description the entry was resolved by, so an adapter
    that serves one target rejects a request for another.
    """

    routing: str
    target: str
    prepare: Symbol
    artifact_path: str = "artifact_path"
    manifest_path: str | None = None
    evaluator_keyword: str | None = None
    identity_keyword: str | None = None


class MissingCapabilityError(RuntimeError):
    """An evaluator does not implement a capability its family declares.

    Raised where a frontend has committed to a declared capability, so a
    missing prepared-path method is a named error and never a silent downgrade
    to a slower path.
    """


@dataclass(frozen=True)
class FamilyEntry:
    """The one declaration of a retained model family.

    ``contract`` identifies and validates the whole family program carried by
    the model. ``generated_contract`` identifies the local subprogram a code
    generator consumes; they are equal for families generated as a whole and
    intentionally distinct when only a backbone is generated.
    """

    key: str
    precedence: int
    detection: Detection
    evaluators: tuple[EvaluatorVariant, ...]
    modes: ModeVocabulary
    schema: Symbol
    extractor: Symbol
    contract: ContractDeclaration
    generated_contract: ContractDeclaration
    codegen: tuple[Symbol, ...]
    adapters: tuple[Symbol, ...]
    capabilities: CapabilityDeclaration
    #: Optional family-owned frontend lifecycle adapter.  It is resolved only
    #: after detection selects this entry, so importing the shared registry or
    #: constructing another family never imports model-specific frontend code.
    frontend_adapter: Symbol | None = None
    routings: tuple[Routing, ...] = ()
    variants: tuple[Variant, ...] = ()
    surfaces: tuple[EvaluatorSurface, ...] = ()
    #: The artifact adapters this family's routings prepare their artifacts
    #: through, resolved per execution target.  A family whose frontend still
    #: prepares its artifacts inline, or whose adapter takes generated source
    #: rather than a model file, declares none; a consumer that requires one
    #: then fails at preparation naming the routing and the target instead of
    #: running a different implementation.
    artifacts: tuple[ArtifactAdapter, ...] = ()


class FamilyRegistry:
    """Family entries in declared precedence order; the first admitter wins."""

    def __init__(self, entries: Iterable[FamilyEntry]):
        ordered = tuple(
            sorted(entries, key=lambda entry: (entry.precedence, entry.key))
        )
        if not ordered:
            raise ValueError("the family registry declares no entry")
        keys = [entry.key for entry in ordered]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate family keys: {sorted(keys)}")
        precedences = [entry.precedence for entry in ordered]
        if len(set(precedences)) != len(precedences):
            raise ValueError(f"duplicate family precedences: {sorted(precedences)}")
        for entry in ordered:
            keys = tuple(routing.key for routing in entry.routings)
            if set(keys) != set(ROUTING_KEYS) or len(keys) != len(ROUTING_KEYS):
                raise ValueError(
                    f"family {entry.key!r} must declare exactly the routings "
                    f"{list(ROUTING_KEYS)}, not {list(keys)}"
                )
            for routing in entry.routings:
                if surface_of(entry, routing.surface) is None:
                    raise ValueError(
                        f"family {entry.key!r} routes {routing.key!r} through the "
                        f"undeclared surface {routing.surface!r}"
                    )
        self._entries = ordered

    @property
    def entries(self) -> tuple[FamilyEntry, ...]:
        """Every declared entry, in precedence order."""

        return self._entries

    def by_key(self, key: str) -> FamilyEntry:
        """Return the entry declared under ``key``."""

        for entry in self._entries:
            if entry.key == key:
                return entry
        raise KeyError(f"no family is declared under {key!r}")

    def detect(
        self, *, model_type: str | None = None, architecture: str | None = None
    ) -> FamilyEntry | None:
        """Return the first entry admitting the requested detection keys."""

        for entry in self._entries:
            if entry.detection.admits(model_type, architecture):
                return entry
        return None


def _discover_entries() -> tuple[FamilyEntry, ...]:
    declaring_package = importlib.import_module(__package__)
    entries = []
    for module_info in pkgutil.iter_modules(declaring_package.__path__):
        if not module_info.name.endswith(DECLARATION_SUFFIX):
            continue
        module = importlib.import_module(f".{module_info.name}", __package__)
        entry = getattr(module, ENTRY_ATTRIBUTE, None)
        if not isinstance(entry, FamilyEntry):
            raise ValueError(
                f"{module_info.name} ends with {DECLARATION_SUFFIX!r} but declares "
                f"no {ENTRY_ATTRIBUTE}"
            )
        entries.append(entry)
    if not entries:
        raise ValueError(
            f"no module in {declaring_package.__name__} declares a family entry"
        )
    return tuple(entries)


_DECLARED_ENTRIES: tuple[FamilyEntry, ...] | None = None
_DECLARED_REGISTRY: FamilyRegistry | None = None


def declared_entries() -> tuple[FamilyEntry, ...]:
    """Every declared family entry, in precedence order, discovered once."""

    global _DECLARED_ENTRIES
    if _DECLARED_ENTRIES is None:
        _DECLARED_ENTRIES = FamilyRegistry(_discover_entries()).entries
    return _DECLARED_ENTRIES


def declared_registry() -> FamilyRegistry:
    """The process-wide registry over every declared family entry."""

    global _DECLARED_REGISTRY
    if _DECLARED_REGISTRY is None:
        _DECLARED_REGISTRY = FamilyRegistry(declared_entries())
    return _DECLARED_REGISTRY


def resolve_symbol(symbol: Symbol) -> Any:
    """Resolve a declared symbol against the declaring package."""

    return symbol.resolve(__package__)


def frontend_adapter_of(entry: FamilyEntry) -> Any | None:
    """Resolve the frontend adapter declared by ``entry``, if any."""

    if entry.frontend_adapter is None:
        return None
    return resolve_symbol(entry.frontend_adapter)


def resolve_evaluator_class(
    entry: FamilyEntry,
    *,
    model_type: str | None = None,
    dtype: str,
    native: Any,
) -> Any:
    """Return the evaluator class declared for this model type and precision.

    There is one evaluator stack, so a family declares exactly one matching
    evaluator and the execution space is fixed by the build rather than chosen
    by the caller. A variant that declares a build-support failure is verified
    before the execution space is initialized, so a backend without the class
    reports the declared message instead of initializing a runtime it cannot
    serve.
    """

    matches = tuple(
        variant
        for variant in entry.evaluators
        if variant.precision == dtype and variant.admits_model_type(model_type)
    )
    if len(matches) != 1:
        raise ValueError(
            f"family {entry.key!r} declares {len(matches)} evaluators for "
            f"model_type {model_type!r} and dtype {dtype!r}; expected exactly one"
        )
    variant = matches[0]
    if variant.declared_error is None:
        if not native._kokkos_is_initialized():
            native._init_kokkos()
        return getattr(native, variant.class_name)
    evaluator = getattr(native, variant.class_name, None)
    if evaluator is None:
        raise RuntimeError(variant.declared_error.format(class_name=variant.class_name))
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    return evaluator


def resolve_model_evaluator_class(*, model_type: str, dtype: str, native: Any) -> Any:
    """Resolve the one evaluator declared for ``model_type`` and ``dtype``."""

    entry = declared_registry().detect(model_type=model_type)
    if entry is None:
        raise ValueError(f"Unsupported Symmetrix model_type {model_type!r}.")
    return resolve_evaluator_class(
        entry,
        model_type=model_type,
        dtype=dtype,
        native=native,
    )


def variant_of(entry: FamilyEntry, model_data: Mapping[str, Any]) -> str | None:
    """Return the first declared variant of ``entry`` the model data selects."""

    for variant in entry.variants:
        if variant.matches(model_data):
            return variant.key
    return None


def capabilities_of(entry: FamilyEntry, evaluator: Any) -> Mapping[str, Any] | None:
    """Return the declared capability mapping an evaluator reports, if any."""

    capabilities = getattr(evaluator, entry.capabilities.query, None)
    return capabilities if isinstance(capabilities, Mapping) else None


def program_family_of(entry: FamilyEntry, evaluator: Any) -> str | None:
    """Return the family identity the evaluator's program declares, if any."""

    program = getattr(evaluator, entry.capabilities.program_query, None)
    if not isinstance(program, Mapping):
        return None
    family = program.get("family")
    return family if isinstance(family, str) else None


def surface_of(entry: FamilyEntry, key: str) -> EvaluatorSurface | None:
    """Return the evaluator surface ``entry`` declares under ``key``, if any."""

    for surface in entry.surfaces:
        if surface.key == key:
            return surface
    return None


def routing_of(entry: FamilyEntry, key: str) -> Routing | None:
    """Return the routing ``entry`` declares under ``key``, if any.

    ``key`` is a resolved routing name, so a caller resolves a documented mode
    spelling through the family's vocabulary first: ``auto`` and an unknown
    spelling both resolve to ``None`` here, which is how a frontend keeps its
    own rejection message in charge.
    """

    for routing in entry.routings:
        if routing.key == key:
            return routing
    return None


def artifact_adapter_of(
    entry: FamilyEntry, *, routing: str, target: str
) -> ArtifactAdapter | None:
    """Return the artifact adapter ``entry`` declares for a routing and target.

    A routing that executes generated code runs an artifact, so a frontend that
    has committed to it asks here for the entry point that prepares one.  The
    answer is the family's declaration and nothing more: ``None`` means the
    family has not declared an adapter for that routing and target, so the
    request cannot be prepared and the frontend must fail instead of running a
    different implementation.
    """

    for adapter in entry.artifacts:
        if adapter.routing == routing and adapter.target == target:
            return adapter
    return None


def supports_surface(entry: FamilyEntry, evaluator: Any, key: str) -> bool:
    """Report whether ``evaluator`` implements the surface ``entry`` declares.

    A frontend that chooses between two admitted policies asks this instead of
    probing the evaluator, so renaming or adding a method is a declaration
    change rather than a silent policy flip.  An undeclared surface is not
    supported, which is how a family states that its evaluators do not provide
    the capability at all.
    """

    surface = surface_of(entry, key)
    if surface is None:
        return False
    return all(callable(getattr(evaluator, name, None)) for name in surface.methods)


def require_surface(
    entry: FamilyEntry, evaluator: Any, key: str
) -> tuple[Callable[..., Any], ...]:
    """Return the bound methods of the surface ``entry`` declares under ``key``.

    A frontend that has committed to a declared capability requires it here, so
    a missing method raises :class:`MissingCapabilityError` naming the family,
    the capability, and the method instead of falling through to a slower path.
    When the evaluator reports the native capability mapping, the surface's
    declared readiness keys are checked as well: a surface whose implementation
    tiers are all unavailable is reported the same way.  An evaluator that
    reports no capability mapping is gated on its method surface alone.
    """

    surface = surface_of(entry, key)
    if surface is None:
        raise MissingCapabilityError(
            f"family {entry.key!r} declares no {key!r} evaluator capability"
        )
    for name in surface.methods:
        if not callable(getattr(evaluator, name, None)):
            raise MissingCapabilityError(
                f"family {entry.key!r} declares the {key!r} evaluator "
                f"capability, but the evaluator does not provide {name!r}"
            )
    if surface.readiness:
        capabilities = capabilities_of(entry, evaluator)
        if capabilities is not None and not any(
            bool(capabilities.get(name)) for name in surface.readiness
        ):
            choices = ", ".join(repr(name) for name in surface.readiness)
            raise MissingCapabilityError(
                f"family {entry.key!r} declares the {key!r} evaluator "
                f"capability, but the evaluator reports no ready implementation "
                f"tier ({choices} are all false)"
            )
    return tuple(getattr(evaluator, name) for name in surface.methods)
