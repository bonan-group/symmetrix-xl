"""Registration point for each family's conformance declarations.

One family declares its structures, session models, adapter instances, and
interface cases in one ``test_family_<family>_conformance_cases`` module beside
the shared suites.  Importing this package discovers those modules by name and
collects their declarations here, so a new family is one new file and no edit to
a shared suite.

The merged views below are what the shared suites read; they name no family, and
a family's own module is the only place its names appear.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ase import Atoms

from .surface import EmptyPathCase

#: The directory the shared suites and the family case modules live in.
TEST_DIR = Path(__file__).resolve().parent.parent

#: The name every family case module carries, so discovery names no family.
FAMILY_MODULE_PATTERN = "test_family_*_conformance_cases.py"


@dataclass(frozen=True)
class RoutingParityCase:
    """One family's declaration for the routing-parity suite.

    ``key`` is the parametrization id the suite reports and ``structure`` names
    a structure the same family registered.  ``build_model`` receives the
    requesting fixture and a session root and returns the model path the case
    evaluates; ``calculator`` builds the family's frontend for one routing and
    one precision, so a family whose reference routing needs a particular
    execution space states that requirement in its own calculator.

    ``compact_frontend`` marks the cases whose models the shared compact
    frontend admits, which is what the retired-keyword contract is driven with.
    ``preparer_guard`` optionally installs the refusal stub that makes a
    reference request's artifact preparation observable, and the two call
    tuples are the observed preparer calls after the reference and the refused
    direct request -- empty and non-empty respectively when the guard is
    declared.
    """

    key: str
    structure: str
    build_model: Callable[[Any, Path], Path]
    calculator: Callable[..., Any]
    compact_frontend: bool = False
    preparer_guard: Callable[[Any, list], None] | None = None
    reference_preparer_calls: tuple = ()
    direct_preparer_calls: tuple = ()


@dataclass(frozen=True)
class FamilyConformance:
    """One family's conformance declaration.

    ``structures`` maps the structure key an instance spec names to the factory
    that builds it; ``build_models`` materializes the session models and
    artifact cache that family's adapters prepare in; ``instances`` is the
    family's instance specs, keyed exactly as the shared suite reports them.

    ``field_response_instances`` names the instances whose prepared lifecycle
    serves a field response derivative, and ``host_compiler_gated_instances``
    the instances whose generated tier is a host artifact gated on a working
    C++ compiler -- both are facts about the family, so the family declares them
    and the shared suite drives whatever is declared.  ``routing_parity`` is the
    family's declaration for the routing-parity suite.
    """

    name: str
    structures: Mapping[str, Callable[[], Atoms]]
    build_models: Callable[[Path], Mapping[str, Any]]
    instances: Mapping[str, dict]
    field_response_instances: tuple[str, ...] = ()
    host_compiler_gated_instances: tuple[str, ...] = ()
    readiness_cases: tuple[EmptyPathCase, ...] = ()
    routing_parity: RoutingParityCase | None = None


_FAMILIES: list[FamilyConformance] = []


def register_family(declaration: FamilyConformance) -> FamilyConformance:
    """Register one family's declarations.

    Called by the family's own case module at import; the module is the only
    place its names appear.
    """

    registered = {family.name for family in _FAMILIES}
    if declaration.name in registered:
        raise ValueError(f"duplicate family registration: {declaration.name!r}")
    _FAMILIES.append(declaration)
    return declaration


# The accessors a family module imports while registering read the merged views
# below at call time, so they are defined before discovery runs.
def crystal(name: str) -> Atoms:
    """The periodic structure registered under ``name``, sized so the skin is real."""

    try:
        factory = STRUCTURES[name]
    except KeyError:
        raise ValueError(
            f"unknown structure {name!r}; declared: {sorted(STRUCTURES)}"
        ) from None
    return factory()


def build_models(root: Path) -> dict:
    """Every registered family's session models, built once per session."""

    models: dict = {}
    for family in _FAMILIES:
        models.update(family.build_models(root))
    return models


def _discover_family_modules() -> None:
    """Import every family case module, which registers itself."""

    if str(TEST_DIR) not in sys.path:
        sys.path.insert(0, str(TEST_DIR))
    for path in sorted(TEST_DIR.glob(FAMILY_MODULE_PATTERN)):
        importlib.import_module(path.stem)


def _validate() -> None:
    if not _FAMILIES:
        raise ValueError("no family conformance module was discovered")
    instances: dict[str, str] = {}
    structures: dict[str, str] = {}
    routings: dict[str, str] = {}
    for family in _FAMILIES:
        for key in family.instances:
            if key in instances:
                raise ValueError(
                    f"instance {key!r} is declared by {instances[key]!r} and "
                    f"{family.name!r}"
                )
            instances[key] = family.name
        for key in family.structures:
            if key in structures:
                raise ValueError(
                    f"structure {key!r} is declared by {structures[key]!r} and "
                    f"{family.name!r}"
                )
            structures[key] = family.name
        declared = (
            *family.field_response_instances,
            *family.host_compiler_gated_instances,
        )
        unknown = sorted(set(declared) - set(family.instances))
        if unknown:
            raise ValueError(f"{family.name!r} declares unknown instances: {unknown}")
        if family.routing_parity is not None:
            key = family.routing_parity.key
            if key in routings:
                raise ValueError(
                    f"routing case {key!r} is declared by {routings[key]!r} and "
                    f"{family.name!r}"
                )
            routings[key] = family.name
            unknown_structure = family.routing_parity.structure not in family.structures
            if unknown_structure:
                raise ValueError(
                    f"{family.name!r} declares a routing case on unknown structure "
                    f"{family.routing_parity.structure!r}"
                )


_discover_family_modules()
_validate()


def _merged_instances() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for family in _FAMILIES:
        merged.update(family.instances)
    return merged


#: Every admitted contract instance, keyed as the shared suite reports it.
ADAPTER_SPECS: dict[str, dict] = _merged_instances()

INSTANCES = tuple(ADAPTER_SPECS)

#: Every instance's admitted prepared-graph tiers, one case per (instance, tier).
LIFECYCLE_CASES = tuple(
    (key, tier)
    for key, spec in ADAPTER_SPECS.items()
    for tier in spec["lifecycle_tiers"]
)

#: The periodic structure declared per structure key, as fresh structures.
STRUCTURES: dict[str, Callable[[], Atoms]] = {
    key: factory for family in _FAMILIES for key, factory in family.structures.items()
}

#: The instances whose prepared lifecycle serves a field response derivative.
FIELD_RESPONSE_INSTANCES = tuple(
    key for family in _FAMILIES for key in family.field_response_instances
)

#: The instances whose generated tier is a host artifact gated on a C++ compiler.
HOST_COMPILER_GATED_INSTANCES = tuple(
    key for family in _FAMILIES for key in family.host_compiler_gated_instances
)

#: The evaluator factories the shared readiness case is asserted on.
READINESS_CASES = tuple(case for family in _FAMILIES for case in family.readiness_cases)

#: The routing-parity cases, in registration order, one per declared family.
ROUTING_CASES = tuple(
    family.routing_parity for family in _FAMILIES if family.routing_parity is not None
)
