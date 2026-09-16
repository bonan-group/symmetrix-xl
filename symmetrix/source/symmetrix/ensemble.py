"""Same-process ensemble calculator built from Symmetrix members."""

from os import PathLike

import numpy as np
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from .calculator import Symmetrix


_ENSEMBLE_PROPERTIES = (
    "energy",
    "free_energy",
    "energies",
    "forces",
    "stress",
    "polarization",
    "becs",
    "polarizability",
)
_COMMITTEE_SUFFIX = "_comm"
_VARIANCE_SUFFIX = "_var"


def _as_member_sequence(value, name):
    if isinstance(value, (str, bytes, PathLike)):
        members = (value,)
    else:
        try:
            members = tuple(value)
        except TypeError as error:
            raise TypeError(f"{name} must be a path or an iterable.") from error
    if not members:
        raise ValueError(f"{name} must contain at least one member.")
    return members


def _member_configuration(calculator):
    evaluator = calculator.evaluator
    return {
        "model type": calculator._native_model_type,
        "field coupling": bool(calculator._has_native_field_coupling()),
        "cutoff": float(evaluator.r_cut),
        "atomic-number mapping": tuple(
            int(value) for value in evaluator.atomic_numbers
        ),
        "precision": calculator._kernel_launch_dtype,
        "Kokkos selection": bool(calculator.use_kokkos),
        "streamed-edge mode": calculator.streamed_edges,
        "low-memory policy": bool(calculator.low_memory),
        "neighbor skin": float(calculator.neighbor_skin),
        "implemented properties": frozenset(calculator.implemented_properties),
    }


def _equal_configuration_value(name, expected, actual):
    if name in ("cutoff", "neighbor skin"):
        return np.array_equal(np.asarray(expected), np.asarray(actual))
    return expected == actual


class SymmetrixEnsemble(Calculator):
    """Evaluate a compatible collection of Symmetrix models sequentially.

    Exactly one of ``model_files`` or ``calculators`` must be supplied. Model
    files are passed to :class:`Symmetrix` with ``symmetrix_kwargs``. Existing
    calculators are retained directly and must have compatible execution and
    model interfaces.

    For each base property ``x``, this calculator exposes the member mean as
    ``x``, the stacked member values as ``x_comm``, and the population variance
    as ``x_var``. Stress values use ASE's six-component Voigt ordering, so
    ``stress_comm`` has shape ``(num_models, 6)``.
    """

    def __init__(
        self,
        model_files=None,
        *,
        calculators=None,
        **symmetrix_kwargs,
    ):
        Calculator.__init__(self)
        if (model_files is None) == (calculators is None):
            raise ValueError("Provide exactly one of model_files or calculators.")

        if model_files is not None:
            paths = _as_member_sequence(model_files, "model_files")
            members = tuple(
                Symmetrix(model_file, **symmetrix_kwargs) for model_file in paths
            )
        else:
            if symmetrix_kwargs:
                names = ", ".join(sorted(symmetrix_kwargs))
                raise ValueError(
                    "Symmetrix construction options cannot be used with "
                    f"calculators: {names}."
                )
            members = (
                (calculators,)
                if isinstance(calculators, Symmetrix)
                else _as_member_sequence(calculators, "calculators")
            )

        for index, member in enumerate(members):
            if not isinstance(member, Symmetrix):
                raise TypeError(
                    "Ensemble members must be Symmetrix calculators; "
                    f"member {index} is {type(member).__name__}."
                )

        self.calculators = members
        self.num_models = len(members)
        self._validate_compatibility()

        supported = set(members[0].implemented_properties)
        self.base_properties = tuple(
            prop for prop in _ENSEMBLE_PROPERTIES if prop in supported
        )
        self.implemented_properties = [
            name
            for prop in self.base_properties
            for name in (
                prop,
                f"{prop}{_COMMITTEE_SUFFIX}",
                f"{prop}{_VARIANCE_SUFFIX}",
            )
        ]

    def _validate_compatibility(self):
        reference = _member_configuration(self.calculators[0])
        for index, member in enumerate(self.calculators[1:], start=1):
            candidate = _member_configuration(member)
            for name, expected in reference.items():
                actual = candidate[name]
                if not _equal_configuration_value(name, expected, actual):
                    raise ValueError(
                        f"Ensemble member {index} has incompatible {name}: "
                        f"expected {expected!r}, got {actual!r}."
                    )

    @staticmethod
    def _base_property(name):
        for suffix in (_COMMITTEE_SUFFIX, _VARIANCE_SUFFIX):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @property
    def electric_field(self):
        """Electric-field override shared by all ensemble members."""
        return self.calculators[0].electric_field

    @electric_field.setter
    def electric_field(self, value):
        for member in self.calculators:
            member.electric_field = (
                None if value is None else np.array(value, dtype=float, copy=True)
            )
        self.results.clear()

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state and any(
            member.check_state(atoms, tol=tol) for member in self.calculators
        ):
            state.append("calculator")
        return state

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        retained_results = {} if system_changes else dict(self.results)
        Calculator.calculate(self, atoms, properties, system_changes)
        base_properties = list(
            dict.fromkeys(self._base_property(prop) for prop in properties)
        )
        unsupported = [
            prop for prop in base_properties if prop not in self.base_properties
        ]
        if unsupported:
            raise PropertyNotImplementedError(
                f"Unsupported ensemble property: {unsupported[0]}"
            )

        member_results = {prop: [] for prop in base_properties}
        for member in self.calculators:
            member_changes = member.check_state(self.atoms)
            member.calculate(self.atoms, base_properties, member_changes)
            for prop in base_properties:
                member_results[prop].append(np.array(member.results[prop], copy=True))

        results = retained_results
        for prop, values in member_results.items():
            committee = np.stack(values, axis=0)
            mean = np.mean(committee, axis=0)
            variance = np.var(committee, axis=0, ddof=0)
            if mean.ndim == 0:
                results[prop] = float(mean)
                results[f"{prop}{_VARIANCE_SUFFIX}"] = float(variance)
            else:
                results[prop] = np.array(mean, copy=True)
                results[f"{prop}{_VARIANCE_SUFFIX}"] = np.array(variance, copy=True)
            results[f"{prop}{_COMMITTEE_SUFFIX}"] = np.array(committee, copy=True)
        self.results = results
