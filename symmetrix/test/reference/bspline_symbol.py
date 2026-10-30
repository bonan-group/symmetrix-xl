"""Cardinal B-spline assignment symbols for the FourierD3 deconvolution.

Specification section 5.2 leaves one convention open: dividing the assigned field by the
*continuous* per-axis symbol ``|sinc(m/N)|**p`` (the NVIDIA v0.5.0 choice, and the one v1
follows) or by the *discrete* symbol ``sum_t M_p(t) exp(2 pi i m t / N)``, which is the
exact attenuation of the shipped assignment when the stencil aligns with integer node
offsets.

This module provides both symbols plus a bounded one-dimensional probe of the actual
assignment, so the difference is measured rather than assumed. It contains no mesh, no
transform, and no three-dimensional assignment: :func:`assignment_attenuation` spreads a
single atom on one periodic axis and takes an explicit (non-FFT) Fourier coefficient.
"""

from __future__ import annotations

from math import factorial

import numpy as np


__all__ = [
    "ADOPTED_CONVENTION",
    "assignment_attenuation",
    "cardinal_bspline",
    "cardinal_bspline_values",
    "continuous_symbol",
    "deconvolution_modulus",
    "discrete_symbol",
]


# Specification section 5.2: v1 uses the continuous cardinal B-spline symbol because the
# acceptance criterion is agreement with NVIDIA v0.5.0 fixtures.
ADOPTED_CONVENTION = "continuous"


def cardinal_bspline(order: int, x) -> np.ndarray:
    """Cardinal B-spline ``M_p(x)`` with integer knots and support ``[0, p]``.

    Evaluated by Cox-de Boor recursion, so the values at non-integer arguments are
    available to the assignment probe.
    """
    if order < 1:
        raise ValueError(f"spline order must be at least 1, got {order}")
    x = np.asarray(x, dtype=np.float64)
    knots = np.arange(0, order + 1, dtype=np.float64)

    def recursion(i: int, k: int) -> np.ndarray:
        if k == 1:
            return ((x >= knots[i]) & (x < knots[i + 1])).astype(np.float64)
        total = np.zeros_like(x)
        left = knots[i + k - 1] - knots[i]
        right = knots[i + k] - knots[i + 1]
        if left > 0.0:
            total += (x - knots[i]) / left * recursion(i, k - 1)
        if right > 0.0:
            total += (knots[i + k] - x) / right * recursion(i + 1, k - 1)
        return total

    return recursion(0, order)


def cardinal_bspline_values(order: int) -> tuple[float, ...]:
    """``M_p`` sampled at the integers ``0 .. p-1``.

    These are the Eulerian-number ratios ``A(p-1, t) / (p-1)!``. They are the samples the
    discrete symbol is built from, and are recorded independently in the fixture.
    """
    degree = order - 1
    eulerian = [[0] * (degree + 1) for _ in range(degree + 1)]
    for n in range(degree + 1):
        for k in range(n + 1):
            eulerian[n][k] = (
                1
                if k == 0
                else (n - k) * eulerian[n - 1][k - 1] + (k + 1) * eulerian[n - 1][k]
            )
    return tuple(eulerian[degree][t] / factorial(degree) for t in range(order))


def continuous_symbol(miller, mesh_size: int, order: int) -> np.ndarray:
    """``|sinc(m/N)|**p``: the continuous cardinal B-spline symbol."""
    return np.abs(np.sinc(np.asarray(miller, dtype=np.float64) / mesh_size)) ** order


def discrete_symbol(miller, mesh_size: int, order: int) -> np.ndarray:
    """``|sum_t M_p(t) exp(2 pi i m t / N)|``: the sampled B-spline symbol."""
    samples = np.asarray(cardinal_bspline_values(order), dtype=np.float64)
    t = np.arange(order, dtype=np.float64)
    return np.abs(
        np.sum(
            samples
            * np.exp(
                2j * np.pi * np.asarray(miller)[..., None] * t[None, :] / mesh_size
            ),
            axis=-1,
        )
    )


def deconvolution_modulus(miller, mesh_size: int, order: int) -> np.ndarray:
    """The per-axis modulus v1 divides by (specification section 5.2)."""
    if ADOPTED_CONVENTION != "continuous":
        raise NotImplementedError(
            f"unsupported deconvolution convention {ADOPTED_CONVENTION!r}"
        )
    return continuous_symbol(miller, mesh_size, order)


def assignment_attenuation(
    mesh_size: int, order: int, fraction: float, miller
) -> np.ndarray:
    """Exact per-axis attenuation of the specification section 5.1 assignment.

    Places one atom at fractional coordinate ``fraction`` on a periodic one-dimensional
    mesh, spreads it with the documented cardinal B-spline stencil, and returns the
    magnitude of the explicit discrete Fourier coefficient of the resulting field. This is
    a bounded convention probe, not a mesh implementation: there is no three-dimensional
    grid, no transform library, and no deconvolution.
    """
    if mesh_size < order:
        raise ValueError(
            "every mesh axis must hold at least `order` nodes; "
            f"got mesh_size={mesh_size}, order={order}"
        )
    y = mesh_size * fraction
    base = int(np.floor(y))
    theta = y - base
    start = int(np.floor(theta - (order - 2) / 2))
    field = np.zeros(mesh_size, dtype=np.float64)
    for i in range(order):
        offset = start + i
        field[(base + offset) % mesh_size] += float(
            cardinal_bspline(order, np.array([order / 2.0 + theta - offset]))[0]
        )
    nodes = np.arange(mesh_size, dtype=np.float64)
    return np.abs(
        np.sum(
            field
            * np.exp(
                -2j * np.pi * np.asarray(miller)[..., None] * nodes[None, :] / mesh_size
            ),
            axis=-1,
        )
    )
