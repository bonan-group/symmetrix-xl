"""Stable Python frontend for the selected Symmetrix native backend."""

from __future__ import annotations

import importlib
from typing import Any

from .backend_loader import BackendError as BackendError
from .backend_loader import available_backends as _available_backends
from .backend_loader import load_backend as _load_backend
from .backend_loader import selected_backend as selected_backend

__version__ = "0.2.0"

_FRONTEND_EXPORTS = {
    "AlchemicalConfig": (".mace_alchemical", "AlchemicalConfig"),
    "AlchemicalPair": (".mace_alchemical", "AlchemicalPair"),
    "ElementMixture": (".mace_alchemical", "ElementMixture"),
    "D2Correction": (".corrections", "D2Correction"),
    "D3Correction": (".corrections", "D3Correction"),
    "FourierD3Correction": (".corrections", "FourierD3Correction"),
    "Symmetrix": (".calculator", "Symmetrix"),
    "FieldAwareCalculator": (".calculator", "FieldAwareCalculator"),
    "FieldContributionCalculator": (".calculator", "FieldContributionCalculator"),
    "SymmetrixEnsemble": (".ensemble", "SymmetrixEnsemble"),
    "JitDeviceArtifactError": (".mace_jit_device_artifact", "JitDeviceArtifactError"),
    "JitDeviceArtifactResult": (".mace_jit_device_artifact", "JitDeviceArtifactResult"),
    "prepare_jit_device_artifact": (
        ".mace_jit_device_artifact",
        "prepare_jit_device_artifact",
    ),
    "JitHostArtifactError": (".mace_jit_host_artifact", "JitHostArtifactError"),
    "JitHostArtifactResult": (".mace_jit_host_artifact", "JitHostArtifactResult"),
    "prepare_jit_host_artifact": (
        ".mace_jit_host_artifact",
        "prepare_jit_host_artifact",
    ),
    "extract_nequip_data": (".extract_nequip_data", "extract_nequip_data"),
    "export_nequip_model": (".extract_nequip_data", "export_nequip_model"),
    "NequIPCalculator": (".nequip_calculator", "NequIPCalculator"),
}

# Stable native surface retained for direct imports and ``from symmetrix import *``.
_NATIVE_EXPORTS = (
    "AffineMLP",
    "AffineMLPKokkos",
    "CubicSpline",
    "CubicSplineKokkos",
    "CubicSplineSet",
    "CubicSplineSetKokkos",
    "E3Linear",
    "E3LinearKokkos",
    "E3LinearKokkosFloat",
    "E3ProductBasis",
    "E3ProductBasisKokkos",
    "E3TensorProduct",
    "E3TensorProductKokkos",
    "E3TensorProductKokkosFloat",
    "MACEKokkos",
    "MACEKokkosFloat",
    "MACENonlinearKokkos",
    "MACENonlinearKokkosFloat",
    "MultilayerPerceptron",
    "MultilayerPerceptronKokkos",
    "MultivariatePolynomial",
    "MultivariatePolynomialKokkos",
    "ZBL",
    "ZBLKokkos",
    "real_sph_harm",
    "real_sph_harm_xyz",
    "sph_harm",
    "sph_harm_xyz",
    "sphericart_complex",
    "sphericart_real_sph_harm",
    "sphericart_sph_harm",
)


def available_backends():
    """Return installed descriptors without importing native extensions."""

    return _available_backends(__version__)


def load_backend(request: str | None = None):
    """Load the selected native backend, permanently for this process."""

    return _load_backend(__version__, request)


def __getattr__(name: str) -> Any:
    frontend = _FRONTEND_EXPORTS.get(name)
    if frontend is not None:
        module = importlib.import_module(frontend[0], __name__)
        value = getattr(module, frontend[1])
        globals()[name] = value
        return value
    native = load_backend()
    if name == "symmetrix":
        value = native
    else:
        try:
            value = getattr(native, name)
        except AttributeError as error:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from error
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FRONTEND_EXPORTS) | set(_NATIVE_EXPORTS))


__all__ = [
    "AlchemicalConfig",
    "AlchemicalPair",
    "BackendError",
    "D2Correction",
    "D3Correction",
    "ElementMixture",
    "FieldAwareCalculator",
    "FieldContributionCalculator",
    "FourierD3Correction",
    "NequIPCalculator",
    "Symmetrix",
    "SymmetrixEnsemble",
    "available_backends",
    "export_nequip_model",
    "extract_nequip_data",
    "load_backend",
    "selected_backend",
]
__all__.extend(_NATIVE_EXPORTS)
