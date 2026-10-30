"""Authoritative maintained NequIP-OAM release qualification contract."""

from __future__ import annotations

OAM_RELEASE_TOLERANCES = {
    "float32": {
        "energy_abs_eV_per_atom": 1.0e-5,
        "forces_max_abs_eV_per_A": 5.0e-4,
        "stress_max_abs_eV_per_A3": 5.0e-4,
    },
    "float64": {
        # NequIP 0.19 OAM packages retain float32 feature tensors internally,
        # which sets the established upstream-reference rounding floor.
        "energy_abs_eV_per_atom": 5.0e-6,
        "forces_max_abs_eV_per_A": 1.0e-5,
        "stress_max_abs_eV_per_A3": 1.0e-6,
    },
}

OAM_HOST_RUNTIME_PAIRS = (
    ("S", "float32"),
    ("M", "float32"),
    ("M", "float64"),
    ("L", "float32"),
    ("XL", "float32"),
)
OAM_HOST_AVAILABLE_NOT_RUNTIME_QUALIFIED = (
    ("S", "float64"),
    ("L", "float64"),
    ("XL", "float64"),
)
