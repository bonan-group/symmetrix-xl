#!/usr/bin/env python3
"""Generate native D2/D3 parameter tables from a pinned torch-dftd source."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Iterable
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np

from _torch_dftd_source import (
    EXPECTED_VERSION,
    PARAMETER_SOURCE_PATHS,
    UPSTREAM_COMMIT,
    UPSTREAM_TAG,
    sha256,
    validate_source,
)


# The supplied checkout is an immutable input, including when it is writable.
sys.dont_write_bytecode = True


SCHEMA = "symmetrix.dispersion-parameters/1"
GENERATOR_NAME = "tools/generate_dispersion_parameters.py"
GENERATOR_SCHEMA = "symmetrix.dispersion-parameter-generator/1"
GENERATOR_VERSION = "1"

BOHR_TO_ANGSTROM = 0.52917726
HARTREE_TO_EV = 27.21138505
JOULE_TO_KILOJOULE = 1.0e-3
HARTREE_MOLAR_ENERGY = 2625.4999
BOHR_TO_NANOMETER = 0.052917726
D2_C6_TO_ATOMIC_UNITS = JOULE_TO_KILOJOULE / HARTREE_MOLAR_ENERGY / BOHR_TO_NANOMETER**6
HARTREE_BOHR6_TO_EV_ANGSTROM6 = HARTREE_TO_EV * BOHR_TO_ANGSTROM**6

CONVERSION_CONSTANTS = (
    ("bohr_to_angstrom", BOHR_TO_ANGSTROM, "angstrom / bohr"),
    ("hartree_to_ev", HARTREE_TO_EV, "eV / hartree"),
    ("joule_to_kilojoule", JOULE_TO_KILOJOULE, "kJ / J"),
    (
        "hartree_molar_energy",
        HARTREE_MOLAR_ENERGY,
        "kJ mol^-1 / hartree",
    ),
    ("bohr_to_nanometer", BOHR_TO_NANOMETER, "nm / bohr"),
    (
        "d2_c6_to_atomic_units",
        D2_C6_TO_ATOMIC_UNITS,
        "hartree bohr^6 / (J nm^6 mol^-1)",
    ),
    (
        "hartree_bohr6_to_ev_angstrom6",
        HARTREE_BOHR6_TO_EV_ANGSTROM6,
        "eV angstrom^6 / (hartree bohr^6)",
    ),
)


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load upstream module {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _read_version(path: Path) -> str:
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"cannot read torch-dftd version from {path}")
    return match.group(1)


def _literal_assignment(module_path: Path, name: str) -> list[float]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            continue
        call = statement.value
        if not isinstance(call, ast.Call) or not call.args:
            break
        values = ast.literal_eval(call.args[0])
        return [float(value) for value in values]
    raise RuntimeError(f"cannot extract literal {name} from {module_path}")


def _format_double(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("generated parameter tables must be finite")
    text = format(float(value), ".17g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def _format_array(
    values: np.ndarray,
    indent: str = "    ",
    formatter: Callable[[float | int], str] = _format_double,
) -> str:
    flattened = np.asarray(values).reshape(-1)
    lines = []
    for start in range(0, flattened.size, 6):
        chunk = flattened[start : start + 6]
        lines.append(indent + ", ".join(formatter(value) for value in chunk) + ",")
    return "\n".join(lines)


def _format_int(value: float | int) -> str:
    return str(int(value))


def _d3_reference_metadata(c6ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return int32 counts[95] and float64 coordination[95, 5]."""
    expected_shape = (95, 95, 5, 5, 3)
    if c6ab.shape != expected_shape:
        raise ValueError(
            f"D3 C6 table must have shape {expected_shape}; got {c6ab.shape}"
        )

    coordination = np.zeros((95, 5), dtype=np.float64)
    observed = np.zeros((95, 5), dtype=np.bool_)

    def record(atomic_number: int, reference: int, value: float) -> None:
        if observed[atomic_number, reference]:
            if coordination[atomic_number, reference] != value:
                raise ValueError(
                    "D3 reference coordination is partner-dependent for "
                    f"atomic number {atomic_number}, reference {reference}"
                )
            return
        coordination[atomic_number, reference] = value
        observed[atomic_number, reference] = True

    for receiver_atomic_number in range(95):
        for source_atomic_number in range(95):
            pair = c6ab[receiver_atomic_number, source_atomic_number]
            positive = pair[:, :, 0] > 0.0
            active_receiver = np.any(positive, axis=1)
            active_source = np.any(positive, axis=0)
            rectangular = np.logical_and.outer(active_receiver, active_source)
            if not np.array_equal(positive, rectangular):
                raise ValueError(
                    "D3 positive-C6 reference mask is not rectangular for "
                    f"atomic numbers ({receiver_atomic_number}, "
                    f"{source_atomic_number})"
                )
            for receiver_reference, source_reference in np.argwhere(positive):
                record(
                    receiver_atomic_number,
                    int(receiver_reference),
                    float(pair[receiver_reference, source_reference, 1]),
                )
                record(
                    source_atomic_number,
                    int(source_reference),
                    float(pair[receiver_reference, source_reference, 2]),
                )

    counts = np.zeros(95, dtype=np.int32)
    for atomic_number in range(95):
        active_references = np.flatnonzero(observed[atomic_number])
        count = active_references.size
        if not np.array_equal(active_references, np.arange(count)):
            raise ValueError(
                "D3 active coordination references must be a zero-based "
                f"contiguous prefix for atomic number {atomic_number}"
            )
        counts[atomic_number] = count
    return counts, coordination


def _quoted(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _parameter_rows(
    xc_module: ModuleType, damping_module: ModuleType
) -> tuple[
    list[tuple[str, dict[str, float | None]]], list[tuple[str, str, dict[str, float]]]
]:
    d2: dict[str, dict[str, float | None]] = {}
    d3: dict[tuple[str, str], dict[str, float]] = {}
    for damping, functional, old in damping_module.damping_xc_combination_list:
        values = xc_module.get_dftd3_default_params(damping, functional, old=old)
        if old:
            d2[functional] = values
        else:
            d3[(damping, functional)] = values
    return sorted(d2.items()), [
        (damping, functional, values)
        for (damping, functional), values in sorted(d3.items())
    ]


def _header() -> str:
    return (
        """// Generated by tools/generate_dispersion_parameters.py. Do not edit.
#pragma once

#include <span>
#include <string_view>

namespace symmetrix::dispersion {

struct DampingParameters {
    double s6;
    double rs6;
    double s18;
    double rs18;
    double alp;
};

struct D2ParameterSet {
    std::string_view functional;
    DampingParameters parameters;
};

struct D3ParameterSet {
    std::string_view damping;
    std::string_view functional;
    DampingParameters parameters;
};

inline constexpr int d2_max_atomic_number = 86;
inline constexpr int d3_max_atomic_number = 94;
inline constexpr int d3_max_coordination_references = 5;

"""
        + "inline constexpr double bohr_to_angstrom = "
        + _format_double(BOHR_TO_ANGSTROM)
        + ";\n\n"
        + """const D2ParameterSet& lookup_d2_parameters(std::string_view functional);
const D3ParameterSet& lookup_d3_parameters(
    std::string_view damping, std::string_view functional);

std::span<const double> d2_r0_angstrom();
std::span<const double> d2_c6_ev_angstrom6();
std::span<const double> d3_c6_ev_angstrom6();
std::span<const int> d3_reference_counts();
std::span<const double> d3_reference_coordination();
std::span<const double> d3_r0_angstrom();
std::span<const double> d3_rcov_angstrom();
std::span<const double> d3_r2r4_angstrom();

} // namespace symmetrix::dispersion
"""
    )


def _provenance(hashes: dict[str, str]) -> str:
    rows = "\n".join(
        f"    SourceHash{{{_quoted(name)}, {_quoted(digest)}}},"
        for name, digest in sorted(hashes.items())
    )
    conversion_rows = "\n".join(
        "    ConversionConstant{"
        f"{_quoted(name)}, {_format_double(value)}, {_quoted(units)}"
        "},"
        for name, value, units in CONVERSION_CONSTANTS
    )
    return f"""// Generated by tools/generate_dispersion_parameters.py. Do not edit.
#pragma once

#include <array>
#include <string_view>

namespace symmetrix::dispersion::provenance {{

inline constexpr std::string_view schema = {_quoted(SCHEMA)};
inline constexpr std::string_view generator_name = {_quoted(GENERATOR_NAME)};
inline constexpr std::string_view generator_schema = {_quoted(GENERATOR_SCHEMA)};
inline constexpr std::string_view generator_version = {_quoted(GENERATOR_VERSION)};
inline constexpr std::string_view upstream_version = {_quoted(EXPECTED_VERSION)};
inline constexpr std::string_view upstream_tag = {_quoted(UPSTREAM_TAG)};
inline constexpr std::string_view upstream_commit = {_quoted(UPSTREAM_COMMIT)};
inline constexpr std::string_view license = "MIT";
inline constexpr std::string_view attribution =
    "torch-dftd Copyright (c) 2021 Preferred Networks, Inc.; MIT License";

struct SourceHash {{
    std::string_view path;
    std::string_view sha256;
}};

inline constexpr std::array source_hashes = {{
{rows}
}};

struct ConversionConstant {{
    std::string_view name;
    double value;
    std::string_view units;
}};

inline constexpr std::array conversion_constants = {{
{conversion_rows}
}};

}} // namespace symmetrix::dispersion::provenance
"""


def _parameter_initializer(values: dict[str, float | None]) -> str:
    return ", ".join(
        _format_double(0.0 if values[name] is None else float(values[name]))
        for name in ("s6", "rs6", "s18", "rs18", "alp")
    )


def _source(
    d2_rows: list[tuple[str, dict[str, float | None]]],
    d3_rows: list[tuple[str, str, dict[str, float]]],
    d2_r0: np.ndarray,
    d2_c6: np.ndarray,
    d3_c6: np.ndarray,
    d3_reference_count: np.ndarray,
    d3_reference_coordination: np.ndarray,
    d3_r0: np.ndarray,
    d3_rcov: np.ndarray,
    d3_r2r4: np.ndarray,
) -> str:
    d2_initializers = "\n".join(
        f"    D2ParameterSet{{{_quoted(functional)}, {{{_parameter_initializer(values)}}}}},"
        for functional, values in d2_rows
    )
    d3_initializers = "\n".join(
        f"    D3ParameterSet{{{_quoted(damping)}, {_quoted(functional)}, "
        f"{{{_parameter_initializer(values)}}}}},"
        for damping, functional, values in d3_rows
    )
    arrays = (
        ("double", "d2_r0_data", d2_r0, _format_double),
        ("double", "d2_c6_data", d2_c6, _format_double),
        ("double", "d3_c6_data", d3_c6, _format_double),
        ("int", "d3_reference_count_data", d3_reference_count, _format_int),
        (
            "double",
            "d3_reference_coordination_data",
            d3_reference_coordination,
            _format_double,
        ),
        ("double", "d3_r0_data", d3_r0, _format_double),
        ("double", "d3_rcov_data", d3_rcov, _format_double),
        ("double", "d3_r2r4_data", d3_r2r4, _format_double),
    )
    array_definitions = "\n\n".join(
        f"constexpr std::array<{value_type}, {values.size}> {name} = {{\n"
        f"{_format_array(values, formatter=formatter)}\n}};"
        for value_type, name, values, formatter in arrays
    )
    span_functions = "\n".join(
        f"std::span<const {value_type}> {function}() {{ return {array}; }}"
        for value_type, function, array in (
            ("double", "d2_r0_angstrom", "d2_r0_data"),
            ("double", "d2_c6_ev_angstrom6", "d2_c6_data"),
            ("double", "d3_c6_ev_angstrom6", "d3_c6_data"),
            ("int", "d3_reference_counts", "d3_reference_count_data"),
            (
                "double",
                "d3_reference_coordination",
                "d3_reference_coordination_data",
            ),
            ("double", "d3_r0_angstrom", "d3_r0_data"),
            ("double", "d3_rcov_angstrom", "d3_rcov_data"),
            ("double", "d3_r2r4_angstrom", "d3_r2r4_data"),
        )
    )
    return f"""// Generated by tools/generate_dispersion_parameters.py. Do not edit.
#include "dispersion_parameters.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace symmetrix::dispersion {{
namespace {{

constexpr std::array d2_parameter_sets = {{
{d2_initializers}
}};

constexpr std::array d3_parameter_sets = {{
{d3_initializers}
}};

{array_definitions}

[[noreturn]] void unsupported(
    const std::string_view method,
    const std::string_view damping,
    const std::string_view functional)
{{
    throw std::invalid_argument(
        "unsupported dispersion parameter combination: method="
        +std::string(method)+", damping="+std::string(damping)
        +", functional="+std::string(functional));
}}

}} // namespace

const D2ParameterSet& lookup_d2_parameters(const std::string_view functional)
{{
    for (const auto& parameters : d2_parameter_sets)
        if (parameters.functional == functional)
            return parameters;
    unsupported("d2", "zero", functional);
}}

const D3ParameterSet& lookup_d3_parameters(
    const std::string_view damping, const std::string_view functional)
{{
    for (const auto& parameters : d3_parameter_sets)
        if (parameters.damping == damping && parameters.functional == functional)
            return parameters;
    unsupported("d3", damping, functional);
}}

{span_functions}

}} // namespace symmetrix::dispersion
"""


def generate(source_root: Path, output: Path) -> None:
    version_path = source_root / "torch_dftd" / "_version.py"
    if not version_path.is_file():
        raise RuntimeError(f"missing torch-dftd version file: {version_path}")
    version = _read_version(version_path)
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"parameter generation requires torch-dftd {EXPECTED_VERSION}; got {version}"
        )

    files = validate_source(source_root, PARAMETER_SOURCE_PATHS)
    xc_module = _load_module(
        "_symmetrix_torch_dftd_xc", files["torch_dftd/dftd3_xc_params.py"]
    )
    damping_module = _load_module(
        "_symmetrix_torch_dftd_damping", files["torch_dftd/testing/damping.py"]
    )
    d2_rows, d3_rows = _parameter_rows(xc_module, damping_module)

    d2_source = files["torch_dftd/nn/params/dftd2_params.py"]
    d2_atomic_r0 = np.asarray(_literal_assignment(d2_source, "r0"), dtype=np.float64)
    d2_atomic_c6 = np.asarray(_literal_assignment(d2_source, "c6"), dtype=np.float64)
    d2_r0 = d2_atomic_r0[:, None] + d2_atomic_r0[None, :]
    d2_c6_atomic = (
        np.sqrt(d2_atomic_c6[:, None] * d2_atomic_c6[None, :]) * D2_C6_TO_ATOMIC_UNITS
    )
    d2_c6 = d2_c6_atomic * HARTREE_TO_EV * BOHR_TO_ANGSTROM**6

    with np.load(
        files["torch_dftd/nn/params/dftd3_params.npz"], allow_pickle=False
    ) as archive:
        d3_c6 = np.asarray(archive["c6ab"], dtype=np.float64).copy()
        d3_r0 = np.asarray(archive["r0ab"], dtype=np.float64) * BOHR_TO_ANGSTROM
        d3_rcov = np.asarray(archive["rcov"], dtype=np.float64) * BOHR_TO_ANGSTROM
        d3_r2r4 = np.asarray(archive["r2r4"], dtype=np.float64) * BOHR_TO_ANGSTROM
    d3_reference_count, d3_reference_coordination = _d3_reference_metadata(d3_c6)
    valid_c6 = d3_c6[..., 0] >= 0.0
    d3_c6[..., 0][valid_c6] *= HARTREE_BOHR6_TO_EV_ANGSTROM6

    hashes = {name: sha256(path) for name, path in files.items()}
    output.mkdir(parents=True, exist_ok=True)
    generated = {
        "dispersion_parameters.hpp": _header(),
        "dispersion_parameter_provenance.hpp": _provenance(hashes),
        "dispersion_parameters.cpp": _source(
            d2_rows,
            d3_rows,
            d2_r0,
            d2_c6,
            d3_c6,
            d3_reference_count,
            d3_reference_coordination,
            d3_r0,
            d3_rcov,
            d3_r2r4,
        ),
    }
    for filename, contents in generated.items():
        (output / filename).write_text(contents, encoding="utf-8", newline="\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-dftd-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        generate(arguments.torch_dftd_source.resolve(), arguments.output.resolve())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
