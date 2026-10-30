"""Family-neutral code-generation helpers shared by every JIT emitter.

Every generated plugin renders C++/CUDA text, reads irrep block layouts,
inlines certified radial splines, and hashes canonical JSON.  Those pieces are
contract-agnostic and name no model family, so they live here once instead of
once per generator.

Family-owned rendering stays in its family module.  What lives here is the
shared vocabulary: C++ literal emission, canonical JSON and its digest, irrep
block arithmetic, the spline accessors and interval clamping every generated
kernel inlines, and the not-a-knot derivative solve plus certificate that
measures an exported table against its analytic radial network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def cpp_string_literal(value: Any) -> str:
    """Emit a quoted C++ string literal for ``value``."""

    return json.dumps(str(value), ensure_ascii=True)


def cpp_text_literal(value: Any) -> str:
    """Emit a quoted C++ string literal for ``value``."""

    return json.dumps(str(value))


def cpp_scalar_ctor(value: Any, scalar_type: str) -> str:
    """Emit ``scalar_type(value)`` from the value's own repr."""

    return f"{scalar_type}({value!r})"


def cpp_float_literal(value: float) -> str:
    """Emit a 32-bit ``static_cast`` of a full-precision scalar literal."""

    return f"static_cast<float>({float(value)!r})"


def scalar_literal(value: float, scalar: str) -> str:
    """Emit a C++ literal that round-trips exactly in the target scalar."""

    if scalar == "float":
        text = f"{float(value):.9g}"
        if not any(ch in text for ch in ".eEn"):
            text += ".0"
        return text + "f"
    return repr(float(value))


def canonical_json(value: Any) -> str:
    """Return the canonical JSON spelling every identity input is built on."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_digest(value: Any) -> str:
    """Return ``sha256:<hex>`` over :func:`canonical_json`."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def canonical_decimal(value: float) -> str:
    # The native builder evaluates the same network with libm.  Quantize to
    # a fixed decimal grid six orders of magnitude coarser than the observed
    # cross-language difference (a few 1e-14), rounding half to even as
    # std::nearbyint does.  Sub-step values are emitted as a canonical zero:
    # their sign is numerical noise, and both sides must agree on it.
    scaled = round(value * 1.0e6)
    if abs(scaled) < 0.5:
        return "0.000000"
    return f"{scaled / 1.0e6:.6f}"


def irreps_dimension(blocks: Sequence[Mapping[str, Any]]) -> int:
    """Return the flattened element count of a parsed irrep block list."""

    return sum(
        int(block["multiplicity"]) * (2 * int(block["l"]) + 1) for block in blocks
    )


def block_offsets(blocks: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return the node-major start offset of every parsed irrep block."""

    offsets = []
    cursor = 0
    for block in blocks:
        offsets.append(cursor)
        cursor += int(block["multiplicity"]) * (2 * int(block["l"]) + 1)
    return offsets


def lane_column_permutation(blocks: Sequence[Mapping[str, Any]]) -> list[int]:
    """Map every node-major element index to its lane-major index.

    For an irrep block at offset o with multiplicity m and angular width
    d = 2l+1, node-major stores element (channel a, component x) at
    o + a*d + x and lane-major stores it at o + x*m + a.
    """

    permutation: list[int] = []
    for block, offset in zip(blocks, block_offsets(blocks)):
        multiplicity = int(block["multiplicity"])
        width = 2 * int(block["l"]) + 1
        for channel in range(multiplicity):
            for component in range(width):
                permutation.append(offset + component * multiplicity + channel)
    return permutation


def harmonic_indices(blocks: Sequence[Mapping[str, Any]]) -> list[int]:
    """Map layer edge-irrep components onto the shared harmonic layout."""

    indices: list[int] = []
    for block in blocks:
        angular = 2 * int(block["l"]) + 1
        for _ in range(int(block["multiplicity"])):
            for component in range(angular):
                indices.append(int(block["l"]) ** 2 + component)
    return indices


def render_gpu_spline_helpers(
    radial_type: str,
    device_qualifier: str,
    *,
    precision_matched_coordinates: bool,
    include_active_edge_helper: bool = True,
) -> str:
    if precision_matched_coordinates:
        evaluation_scalar_cpp = "Scalar"
        evaluation_point_body = """    const Scalar local_radius =
        static_cast<Scalar>(radius);
    const Scalar x0 = static_cast<Scalar>(radial.x0);
    const Scalar h = static_cast<Scalar>(radial.h);
    std::int32_t interval = static_cast<std::int32_t>(
        floor((local_radius - x0) / h));
    Scalar x = local_radius - x0 - h * interval;
    if (interval < 0) {
        interval = 0;
        x = Scalar(0);
    } else if (interval >= static_cast<std::int32_t>(radial.intervals)) {
        interval = static_cast<std::int32_t>(radial.intervals) - 1;
        x = h;
    }
    const Scalar xx = x * x;
    return {interval, x, xx, xx * x};"""
    else:
        evaluation_scalar_cpp = "double"
        evaluation_point_body = """    std::int32_t interval = static_cast<std::int32_t>(
        floor((radius - radial.x0) / radial.h));
    double x = radius - radial.x0 - radial.h * interval;
    if (interval < 0) {
        interval = 0;
        x = 0.0;
    } else if (interval >= static_cast<std::int32_t>(radial.intervals)) {
        interval = static_cast<std::int32_t>(radial.intervals) - 1;
        x = radial.h;
    }
    const double xx = x * x;
    return {interval, x, xx, xx * x};"""
    active_edge_helper = (
        f"""{device_qualifier} bool edge_is_active(double cutoff, double radius)
{{
    return radius < cutoff;
}}

"""
        if include_active_edge_helper
        else ""
    )
    return f"""struct EvaluationPoint {{
    std::int32_t interval;
    {evaluation_scalar_cpp} x;
    {evaluation_scalar_cpp} xx;
    {evaluation_scalar_cpp} xxx;
}};

{active_edge_helper.rstrip()}
{device_qualifier} EvaluationPoint evaluation_point(
    const {radial_type}& radial, double radius)
{{
{evaluation_point_body}
}}

{device_qualifier} std::size_t radial_index(
    const {radial_type}& radial,
    std::int32_t edge_type,
    std::int32_t interval,
    std::int32_t coefficient,
    std::int32_t function)
{{
    return (((static_cast<std::size_t>(edge_type) * radial.intervals + interval)
        * 4u + coefficient) * radial.functions + function);
}}

{device_qualifier} Scalar evaluate_radial(
    const {radial_type}& radial,
    std::int32_t edge_type,
    const EvaluationPoint& point,
    std::int32_t function)
{{
    const Scalar c0 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 0, function)];
    const Scalar c1 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 1, function)];
    const Scalar c2 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 2, function)];
    const Scalar c3 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 3, function)];
    return c0 + c1 * static_cast<Scalar>(point.x)
        + c2 * static_cast<Scalar>(point.xx)
        + c3 * static_cast<Scalar>(point.xxx);
}}

{device_qualifier} void evaluate_radial(
    const {radial_type}& radial,
    std::int32_t edge_type,
    const EvaluationPoint& point,
    std::int32_t function,
    Scalar& value,
    Scalar& derivative)
{{
    const Scalar c0 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 0, function)];
    const Scalar c1 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 1, function)];
    const Scalar c2 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 2, function)];
    const Scalar c3 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 3, function)];
    value = c0 + c1 * static_cast<Scalar>(point.x)
        + c2 * static_cast<Scalar>(point.xx)
        + c3 * static_cast<Scalar>(point.xxx);
    derivative = c1 + c2 * static_cast<Scalar>(2.0 * point.x)
        + c3 * static_cast<Scalar>(3.0 * point.xx);
}}

{device_qualifier} std::int32_t pair_type(
    std::int32_t left, std::int32_t right, std::int32_t type_count)
{{
    return left <= right
        ? left * (2 * type_count - left - 1) / 2 + right
        : right * (2 * type_count - right - 1) / 2 + left;
}}"""


def render_host_spline_block(scalar: str, weight_dim: int, derivatives: bool) -> str:
    """Inline certified cubic spline evaluation at the packet's edge distance."""

    lines = [
        "const double radius = a->distances[directed_edge];",
        "int interval = static_cast<int>(std::floor(",
        "    (radius - a->spline_x0) / a->spline_h));",
        "double coordinate = radius - a->spline_x0 - a->spline_h * interval;",
        "if (interval < 0) {",
        "    interval = 0;",
        "    coordinate = 0.0;",
        "} else if (interval >= a->spline_interval_count) {",
        "    interval = a->spline_interval_count - 1;",
        "    coordinate = a->spline_h;",
        "}",
        "const double coordinate_squared = coordinate * coordinate;",
        "const double coordinate_cubed = coordinate_squared * coordinate;",
        "const auto* coefficient_base = coefficients",
        "    + static_cast<int64_t>(interval) * 4 * a->spline_function_count;",
        "const auto* coefficient_0 = coefficient_base;",
        "const auto* coefficient_1 = coefficient_base + a->spline_function_count;",
        "const auto* coefficient_2 = coefficient_base + 2 * a->spline_function_count;",
        "const auto* coefficient_3 = coefficient_base + 3 * a->spline_function_count;",
        f"{scalar} weights[{weight_dim}];",
        "#if defined(_OPENMP)",
        "#pragma omp simd",
        "#endif",
        "for (int function = 0; function < a->spline_function_count; ++function)",
        "    weights[function] = coefficient_0[function]",
        f"        + coefficient_1[function] * static_cast<{scalar}>(coordinate)",
        f"        + coefficient_2[function] * static_cast<{scalar}>(coordinate_squared)",
        f"        + coefficient_3[function] * static_cast<{scalar}>(coordinate_cubed);",
    ]
    if derivatives:
        lines += [
            f"{scalar} weight_derivatives[{weight_dim}];",
            "#if defined(_OPENMP)",
            "#pragma omp simd",
            "#endif",
            "for (int function = 0; function < a->spline_function_count; ++function)",
            "    weight_derivatives[function] = coefficient_1[function]",
            f"        + coefficient_2[function] * static_cast<{scalar}>(2.0 * coordinate)",
            f"        + coefficient_3[function] * static_cast<{scalar}>(3.0 * coordinate_squared);",
        ]
    return "\n".join("        " + line for line in lines)


def render_cooperative_spline_helpers(
    *, value_symbol: str, derivative_symbol: str, coordinate_type: str = "double"
) -> str:
    """Render scalar spline accessors for receiver and source owner CTAs."""

    if coordinate_type not in ("double", "Scalar"):
        raise ValueError("spline coordinate type must be double or Scalar")

    return (
        """__device__ inline Scalar {value_symbol}(
    const Scalar* base, const int function_count, const int function,
    const {coordinate_type} coordinate)
{
    const Scalar x = static_cast<Scalar>(coordinate);
    const Scalar x2 = static_cast<Scalar>(coordinate * coordinate);
    const Scalar x3 = static_cast<Scalar>(coordinate * coordinate * coordinate);
    return base[function]
        + base[function_count + function] * x
        + base[2 * function_count + function] * x2
        + base[3 * function_count + function] * x3;
}

__device__ inline Scalar {derivative_symbol}(
    const Scalar* base, const int function_count, const int function,
    const {coordinate_type} coordinate)
{
    const Scalar twice_x = static_cast<Scalar>(2.0 * coordinate);
    const Scalar thrice_x2 = static_cast<Scalar>(3.0 * (coordinate * coordinate));
    return base[function_count + function]
        + base[2 * function_count + function] * twice_x
        + base[3 * function_count + function] * thrice_x2;
}""".replace("{value_symbol}", value_symbol)
        .replace("{derivative_symbol}", derivative_symbol)
        .replace("{coordinate_type}", coordinate_type)
    )


def render_cooperative_spline_block(coordinate_type: str = "double") -> str:
    """Share the interval and double coordinate computed by the edge owner."""

    if coordinate_type not in ("double", "Scalar"):
        raise ValueError("spline coordinate type must be double or Scalar")
    coordinate = (
        "const double coordinate = scratch.coordinate;"
        if coordinate_type == "double"
        else "const Scalar coordinate = static_cast<Scalar>(scratch.coordinate);"
    )

    return """if (threadIdx.x == 0) {
            const double radius = args.distances[directed_edge];
            int interval = static_cast<int>(::floor(
                (radius - args.spline_x0) / args.spline_h));
            double coordinate = radius - args.spline_x0
                - args.spline_h * interval;
            if (interval < 0) {
                interval = 0;
                coordinate = 0.0;
            } else if (interval >= args.spline_interval_count) {
                interval = args.spline_interval_count - 1;
                coordinate = args.spline_h;
            }
            scratch.interval = interval;
            scratch.coordinate = coordinate;
        }
        __syncthreads();
        {coordinate}
        const Scalar* coefficient_base = coefficients
            + static_cast<std::int64_t>(scratch.interval) * 4
                * static_cast<std::int64_t>(args.spline_function_count);""".replace(
        "{coordinate}", coordinate
    )


def spline_nodal_values(
    evaluate,
    function_count: int,
    *,
    num_points: int,
    grid_min: float,
    cutoff: float,
) -> tuple[list[list[float]], float]:
    """Sample an analytic radial network on the uniform spline grid.

    Returns ``(nodal_values, spacing)`` where the row count is the network's
    function count.
    """

    spacing = (cutoff - grid_min) / float(num_points - 1)
    nodal_values = [[0.0] * num_points for _ in range(function_count)]
    for node in range(num_points):
        output = evaluate(grid_min + spacing * node)
        for function, value in enumerate(output):
            nodal_values[function][node] = value
    return nodal_values, spacing


def spline_table_fingerprint_from_nodal(
    *,
    grid_min: float,
    cutoff: float,
    num_points: int,
    function_count: int,
    nodal_values: list[list[float]],
) -> str:
    canonical = (
        ";".join(
            [
                canonical_decimal(grid_min),
                canonical_decimal(cutoff),
                str(num_points),
                str(function_count),
            ]
            + [canonical_decimal(value) for values in nodal_values for value in values]
        )
        + ";"
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def spline_table_fingerprint(
    evaluate,
    function_count: int,
    *,
    num_points: int,
    grid_min: float,
    cutoff: float,
) -> str:
    """Hash the portable grid and nodal MLP outputs used by the native builder."""

    nodal_values, _ = spline_nodal_values(
        evaluate,
        function_count,
        num_points=num_points,
        grid_min=grid_min,
        cutoff=cutoff,
    )
    return spline_table_fingerprint_from_nodal(
        grid_min=grid_min,
        cutoff=cutoff,
        num_points=num_points,
        function_count=len(nodal_values),
        nodal_values=nodal_values,
    )


def spline_derivative_row(values: list[float], h: float) -> list[float]:
    """Reproduce the native not-a-knot cubic derivative solve exactly.

    Mirrors ``CompactRadialModel::spline_derivatives`` in ``compact_radial.cpp``:
    a tridiagonal system for the interior node derivatives, a one-sided
    not-a-knot condition for the first node, and a clamped zero end slope.
    """

    count = len(values)
    unknowns = count - 2
    lower = [0.0] * unknowns
    diagonal = [4.0] * unknowns
    upper = [0.0] * unknowns
    rhs = [0.0] * unknowns
    left_not_a_knot = 2.0 * (-values[0] + 2.0 * values[1] - values[2]) / h
    upper[0] = 2.0
    rhs[0] = 3.0 * (values[2] - values[0]) / h - left_not_a_knot
    for node in range(2, count - 1):
        row = node - 1
        lower[row] = 1.0
        if node + 1 <= count - 2:
            upper[row] = 1.0
        rhs[row] = 3.0 * (values[node + 1] - values[node - 1]) / h
    for row in range(1, unknowns):
        factor = lower[row] / diagonal[row - 1]
        diagonal[row] -= factor * upper[row - 1]
        rhs[row] -= factor * rhs[row - 1]
    solution = [0.0] * unknowns
    solution[-1] = rhs[-1] / diagonal[-1]
    for row in range(unknowns - 2, -1, -1):
        solution[row] = (rhs[row] - upper[row] * solution[row + 1]) / diagonal[row]
    derivatives = [0.0] * count
    for node in range(1, count - 1):
        derivatives[node] = solution[node - 1]
    derivatives[0] = derivatives[2] + left_not_a_knot
    derivatives[-1] = 0.0
    return derivatives


def spline_certification(
    evaluate,
    *,
    num_points: int,
    grid_min: float,
    cutoff: float,
    function_count: int | None = None,
    nodal_values: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Measure an exported table against the analytic radial network.

    The certificate reports real measurements rather than a passed-looking
    constant.  Every interval midpoint is compared, because the leading term of
    the cubic Hermite error ``t**2 (t-h)**2 f'''' / 24`` is maximal there; the
    derivative error uses the same midpoints with a centered difference of the
    analytic network (step ``h/100``, which keeps the difference truncation
    three orders of magnitude below the spline error).  Grid nodes themselves
    are exact by construction.
    """

    h = (cutoff - grid_min) / float(num_points - 1)
    if nodal_values is None:
        if function_count is None:
            function_count = len(evaluate(grid_min))
        nodal_values, h = spline_nodal_values(
            evaluate,
            function_count,
            num_points=num_points,
            grid_min=grid_min,
            cutoff=cutoff,
        )
    function_count = len(nodal_values)
    derivatives = [spline_derivative_row(values, h) for values in nodal_values]
    value_error = 0.0
    derivative_error = 0.0
    step = h / 100.0
    offset = 0.5 * h
    for interval in range(num_points - 1):
        radius = grid_min + h * interval + offset
        analytic = evaluate(radius)
        plus = evaluate(radius + step)
        minus = evaluate(radius - step)
        for function in range(function_count):
            y0 = nodal_values[function][interval]
            y1 = nodal_values[function][interval + 1]
            d0 = derivatives[function][interval]
            d1 = derivatives[function][interval + 1]
            coefficient_2 = (3.0 * (y1 - y0) / h - 2.0 * d0 - d1) / h
            coefficient_3 = (2.0 * (y0 - y1) / h + d0 + d1) / (h * h)
            value_error = max(
                value_error,
                abs(
                    y0
                    + d0 * offset
                    + coefficient_2 * offset * offset
                    + coefficient_3 * offset * offset * offset
                    - analytic[function]
                ),
            )
            derivative_error = max(
                derivative_error,
                abs(
                    d0
                    + 2.0 * coefficient_2 * offset
                    + 3.0 * coefficient_3 * offset * offset
                    - (plus[function] - minus[function]) / (2.0 * step)
                ),
            )
    return {
        "grid_points": num_points,
        "clamped_points": 0,
        "max_abs_value_error": value_error,
        "max_abs_derivative_error": derivative_error,
    }


__all__ = [
    "block_offsets",
    "canonical_decimal",
    "canonical_digest",
    "canonical_json",
    "cpp_float_literal",
    "cpp_scalar_ctor",
    "cpp_string_literal",
    "cpp_text_literal",
    "harmonic_indices",
    "irreps_dimension",
    "lane_column_permutation",
    "render_cooperative_spline_block",
    "render_cooperative_spline_helpers",
    "render_gpu_spline_helpers",
    "render_host_spline_block",
    "scalar_literal",
    "spline_certification",
    "spline_derivative_row",
    "spline_nodal_values",
    "spline_table_fingerprint",
    "spline_table_fingerprint_from_nodal",
]
