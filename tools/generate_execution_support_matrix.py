"""Generate the execution support matrix from the declared family registry.

The public execution requests, the accepted spellings, the model boundary, the
declared evaluator capabilities, and the execution spaces are read from the
family declarations the frontend dispatches on, so a family that gains a mode
or a capability cannot leave the matrix behind.  Only the documentation facts
that no code declares -- which artifact a request needs, what LAMMPS admits,
and how far a capability is qualified -- live here, and the registry
cross-checks them:

* every canonical mode a family declares has exactly one request row,
* every request row names a declared mode and a declared family,
* every capability row names a key the native query publishes, and the union
  of the rows covers that whole key set,
* a request whose family declares a prepared surface must name the artifact it
  needs, and a rejected request must give its reason.

The generated block is delimited by :data:`BEGIN_MARKER` and :data:`END_MARKER`
in :data:`DOCUMENT`; ``--check`` verifies the document is current and writes
nothing, which is what the drift test runs.

Usage::

    python tools/generate_execution_support_matrix.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = REPOSITORY_ROOT / "docs" / "reference" / "execution_support_matrix.md"

BEGIN_MARKER = "<!-- BEGIN GENERATED EXECUTION SUPPORT MATRIX -->"
END_MARKER = "<!-- END GENERATED EXECUTION SUPPORT MATRIX -->"

#: The execution spaces the shipped Kokkos backends provide.  This is a build
#: fact rather than a family fact, so it is stated once for every family.
KOKKOS_BACKENDS = "Serial, OpenMP, CUDA, HIP"
FAMILY_KOKKOS_BACKENDS = {
    "OAM": "Serial, OpenMP, CUDA; HIP package rejects this family",
}


@dataclass(frozen=True)
class RequestRow:
    """One public execution request of one family."""

    mode: str
    label: str
    admitted: bool
    artifact: str
    lammps: str
    reason: str = ""


@dataclass(frozen=True)
class CapabilityRow:
    """One declared evaluator capability of one family."""

    key: str
    status: str
    scope: str
    limits: str


#: The requests each family publishes, keyed by the family entry's key.  The
#: ``mode`` of every row is a canonical mode of that family's declaration.
FAMILY_REQUESTS: Mapping[str, tuple[RequestRow, ...]] = {
    "MACE_Nonlinear": (
        RequestRow(
            mode="auto",
            label="`auto`",
            admitted=True,
            artifact="As resolved below",
            lammps="Not supported",
        ),
        RequestRow(
            mode="direct",
            label="`direct`",
            admitted=True,
            artifact=(
                "The generated MH-1 node program; a host or device plugin "
                "artifact is required"
            ),
            lammps="MACE-MH-1 is not supported",
        ),
        RequestRow(
            mode="reference",
            label="`reference`",
            admitted=True,
            artifact="None",
            lammps="MACE-MH-1 is not supported",
        ),
    ),
    "MACE": (
        RequestRow(
            mode="auto",
            label="`auto`",
            admitted=True,
            artifact="As resolved below. MACELES evaluates its LES term in both routings",
            lammps=(
                "Format-aware compatibility default for standard "
                "MACE/MACEField; MACELES is not supported in LAMMPS"
            ),
        ),
        RequestRow(
            mode="direct",
            label="`direct`",
            admitted=True,
            artifact=(
                "Two-interaction MACE, MACEField, and MACELES require an R1 "
                "artifact. MACELES does not support fixed-workspace plans; "
                "single-layer standard MACE runs without R1 specialization, "
                "but capacity may specialize M0/R0"
            ),
            lammps=(
                "Standard MACE and MACEField; the R1 artifact is required for "
                "two-interaction models, but not for single-layer models. "
                "MACELES is not supported in LAMMPS"
            ),
        ),
        RequestRow(
            mode="reference",
            label="`reference`",
            admitted=True,
            artifact="None",
            lammps=(
                "The original Symmetrix format (named v1 here) is retired, so "
                "the compatibility spelling resolves only for supported "
                "standard MACE/MACEField controls. MACELES is not supported "
                "in LAMMPS"
            ),
        ),
    ),
    "MACE_Polar": (
        RequestRow(
            mode="auto",
            label="`auto`",
            admitted=True,
            artifact="As resolved below",
            lammps=(
                "`symmetrix/mace/polar/kk`; no artifact uses the portable "
                "`pair_spline_v1` backbone and one optional artifact selects "
                "the generated/direct backbone; the qualified deployment uses "
                "exactly one MPI rank owning the complete system and rejects "
                "domain decomposition"
            ),
        ),
        RequestRow(
            mode="direct",
            label="`direct`",
            admitted=True,
            artifact=(
                "The generated MH-1 backbone subprogram; a host or device "
                "plugin artifact is required, while the long-range polar "
                "program remains native"
            ),
            lammps=(
                "`symmetrix/mace/polar/kk` with one matching generated/direct "
                "artifact for the host or device target; exactly one MPI rank "
                "must own the complete system, and domain decomposition is "
                "rejected"
            ),
        ),
        RequestRow(
            mode="reference",
            label="`reference`",
            admitted=True,
            artifact="None",
            lammps=(
                "`symmetrix/mace/polar/kk` uses the artifact-free portable "
                "`pair_spline_v1` backbone with the same one MPI rank, complete "
                "system admission boundary and rejects domain decomposition"
            ),
        ),
    ),
    "OAM": (
        RequestRow(
            mode="direct",
            label="`direct`",
            admitted=True,
            artifact=(
                "The architecture-bound, full-message-only receiver artifact: "
                "a host ABI v7 spline plugin, a CUDA ABI v3 NVRTC cubin, or an "
                "nvcc shared library"
            ),
            lammps="Not supported",
        ),
        RequestRow(
            mode="reference",
            label="`reference`",
            admitted=True,
            artifact="None",
            lammps="Not supported",
        ),
    ),
}

#: The declared capabilities each family reports, keyed by the family entry's
#: key.  Every ``key`` is one of the native capability keys.
FAMILY_CAPABILITIES: Mapping[str, tuple[CapabilityRow, ...]] = {
    "MACE_Nonlinear": (
        CapabilityRow(
            "target",
            "build-dependent",
            "the default execution space",
            "host, CUDA, or HIP",
        ),
        CapabilityRow("precision", "qualified", "float32 and float64", "none"),
        CapabilityRow(
            "reference",
            "qualified",
            "single-thread generic execution",
            "host builds only",
        ),
        CapabilityRow(
            "portable",
            "qualified",
            "the generic Kokkos prepared path",
            "no artifact required",
        ),
        CapabilityRow(
            "generated",
            "qualified",
            "the generated node program",
            "a prepared artifact is required",
        ),
        CapabilityRow(
            "stress", "qualified", "the prepared-path stress reduction", "none"
        ),
        CapabilityRow(
            "field",
            "qualified",
            "a model property",
            "present only when the model couples a field",
        ),
        CapabilityRow(
            "parameter_gradients", "unsupported", "-", "not implemented for this family"
        ),
        CapabilityRow("host_artifact", "qualified", "every build", "none"),
        CapabilityRow(
            "device_artifact",
            "build-dependent",
            "device execution spaces",
            "a CUDA-capable build is required",
        ),
    ),
    "MACE": (
        CapabilityRow(
            "target",
            "build-dependent",
            "the default execution space",
            "host, CUDA, or HIP",
        ),
        CapabilityRow("precision", "qualified", "float32 and float64", "none"),
        CapabilityRow(
            "reference",
            "qualified",
            "single-thread generic execution",
            "host builds only",
        ),
        CapabilityRow(
            "portable",
            "qualified",
            "the generic Kokkos prepared path",
            "no artifact required",
        ),
        CapabilityRow(
            "generated",
            "qualified",
            "the JIT/RTC artifact or a frozen built-in module",
            "a prepared artifact is required",
        ),
        CapabilityRow(
            "stress", "qualified", "the prepared-path stress reduction", "none"
        ),
        CapabilityRow(
            "field",
            "qualified",
            "MACEField models",
            "ordinary MACE declares no field coupling",
        ),
        CapabilityRow(
            "parameter_gradients", "qualified", "the prepared factorized path", "none"
        ),
        CapabilityRow("host_artifact", "qualified", "every build", "none"),
        CapabilityRow(
            "device_artifact",
            "build-dependent",
            "device execution spaces",
            "a CUDA-capable build is required",
        ),
    ),
    "MACE_Polar": (
        CapabilityRow(
            "target",
            "build-dependent",
            "the default execution space",
            "CPU and CUDA are runtime-qualified; HIP remains unqualified",
        ),
        CapabilityRow("precision", "qualified", "float32 and float64", "none"),
        CapabilityRow(
            "reference",
            "qualified",
            "the artifact-free portable MH-1 backbone and native polar program",
            "CPU and CUDA runtime-qualified; HIP remains unqualified",
        ),
        CapabilityRow(
            "portable",
            "qualified",
            "the layer-local MH-1 fast path and the long-range kind group's portable nodes",
            "independent per-layer UVU widths; supported keep-last layouts only",
        ),
        CapabilityRow(
            "generated",
            "qualified",
            "the generated MH-1 backbone subprogram with retained outputs and external product seeds",
            "a prepared artifact and a recognized keep-last schedule are required; CPU and CUDA are runtime-qualified, while HIP remains unqualified",
        ),
        CapabilityRow(
            "stress",
            "qualified",
            "released M FP32/FP64 EFS: exact-zero isolated stress and composed periodic stress",
            "CPU and CUDA sm120; S/L checkpoints and HIP runtime remain unqualified",
        ),
        CapabilityRow(
            "field",
            "qualified",
            "a program property",
            "declared when the model's compiled program carries the external-field projection node",
        ),
        CapabilityRow(
            "parameter_gradients", "unsupported", "-", "not implemented for this family"
        ),
        CapabilityRow("host_artifact", "qualified", "every build", "none"),
        CapabilityRow(
            "device_artifact",
            "build-dependent",
            "device execution spaces",
            "a CUDA-capable build is required",
        ),
    ),
    "OAM": (
        CapabilityRow(
            "target",
            "build-dependent",
            "the default execution space",
            "host or CUDA; HIP is unsupported",
        ),
        CapabilityRow(
            "precision",
            "build-dependent",
            "float32 and float64 implementations on host and CUDA",
            "current full-message-only artifact release-matrix qualification is pending",
        ),
        CapabilityRow(
            "reference",
            "build-dependent",
            "the artifact-free same-spline Kokkos oracle on host and CUDA",
            "current release-matrix qualification is pending; same-spline parity does not establish upstream-NequIP parity",
        ),
        CapabilityRow(
            "portable",
            "build-dependent",
            "the compiler-free Kokkos edge product",
            "the current runtime matrix is pending; this is the same-spline reference, not the optimized production route",
        ),
        CapabilityRow(
            "generated",
            "build-dependent",
            "the full-message-only host ABI v7 artifact or CUDA ABI v3 receiver cubin",
            "a prepared exact artifact is required; the current CUDA artifact is compile-qualified, but fresh runtime qualification is pending",
        ),
        CapabilityRow(
            "stress",
            "requalification pending",
            "implemented on the current host and CUDA routes",
            "full-message-only ABI-v7/ABI-v3 runtime qualification is pending; HIP is unsupported",
        ),
        CapabilityRow(
            "field", "unsupported", "-", "the OAM contract carries no field coupling"
        ),
        CapabilityRow(
            "parameter_gradients", "unsupported", "-", "not implemented for this family"
        ),
        CapabilityRow(
            "host_artifact",
            "build-dependent",
            "source generation and compilation of the full-message-only ABI-v7 artifact",
            "fresh host runtime qualification and repaired-wheel inclusion gates are pending",
        ),
        CapabilityRow(
            "device_artifact",
            "build-dependent",
            "source generation and compilation of the full-message-only CUDA ABI-v3 artifact",
            "fresh CUDA runtime qualification and repaired-wheel inclusion gates are pending; HIP is unavailable",
        ),
    ),
}


def _registry():
    from symmetrix import family_registry

    return family_registry


def _boundary(entry) -> str:
    detection = entry.detection
    parts = []
    if detection.model_types:
        rendered = "/".join(f"`{name}`" for name in detection.model_types)
        parts.append(f"`model_type` {rendered}")
    if detection.architectures:
        rendered = "/".join(f"`{name}`" for name in detection.architectures)
        parts.append(f"`architecture` {rendered}")
    return "; ".join(parts)


def _evaluator_spaces(entry) -> str:
    """Return evaluator spaces, precisions, and model-type scopes."""

    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for variant in entry.evaluators:
        space = "Kokkos" if variant.kokkos else "serial"
        groups.setdefault((space, variant.model_types), []).append(variant.precision)
    return "; ".join(
        (
            f"{space} ({', '.join(sorted(set(precisions)))}"
            + (
                "; `model_type` "
                + "/".join(f"`{model_type}`" for model_type in model_types)
                if model_types
                else ""
            )
            + ")"
        )
        for (space, model_types), precisions in groups.items()
    )


def _spellings(entry, row: RequestRow) -> str:
    """Return every spelling that resolves to this request's canonical mode."""

    vocabulary = entry.modes
    label = row.label.strip("`")
    others = sorted(
        spelling
        for spelling in vocabulary.aliases
        if spelling != label and vocabulary.canonical_mode(spelling) == row.mode
    )
    if not others:
        return row.label
    rendered = [f"{row.label} (public)"]
    for spelling in others:
        if spelling in vocabulary.deprecated:
            kind = "deprecated"
        elif spelling in vocabulary.primary:
            kind = "primary"
        else:
            kind = "alias"
        rendered.append(f"`{spelling}` ({kind})")
    return ", ".join(rendered)


def _family_heading(entry) -> str:
    return f"`{entry.key}` ({entry.capabilities.family})"


def _request_table(entries: Sequence) -> list[str]:
    lines = [
        "| Family | Public request | Accepted spellings | Model boundary | Declared evaluator spaces | Shipped Kokkos backends | Runtime artifact | LAMMPS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        for row in FAMILY_REQUESTS[entry.key]:
            request = (
                row.label if row.admitted else f"{row.label} (rejected: {row.reason})"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        _family_heading(entry),
                        request,
                        _spellings(entry, row),
                        _boundary(entry),
                        _evaluator_spaces(entry),
                        FAMILY_KOKKOS_BACKENDS.get(entry.key, KOKKOS_BACKENDS),
                        row.artifact,
                        row.lammps,
                    )
                )
                + " |"
            )
    return lines


def _capability_table(entries: Sequence) -> list[str]:
    lines = [
        "| Family | Capability | Declared status | Scope | Limits |",
        "|---|---|---|---|---|",
    ]
    for entry in entries:
        for row in FAMILY_CAPABILITIES[entry.key]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _family_heading(entry),
                        f"`{row.key}`",
                        row.status,
                        row.scope,
                        row.limits,
                    )
                )
                + " |"
            )
    return lines


def render_matrix(entries: Sequence | None = None) -> str:
    """Return the generated block for the declared families."""

    if entries is None:
        entries = _registry().declared_entries()
    validate(entries)
    lines = [
        BEGIN_MARKER,
        "Generated from the declared family registry by",
        "`tools/generate_execution_support_matrix.py`; regenerate it instead of",
        "editing this block.",
        "",
        "## Public execution requests",
        "",
        *_request_table(entries),
        "",
        "## Declared evaluator capabilities",
        "",
        *_capability_table(entries),
        END_MARKER,
    ]
    return "\n".join(lines) + "\n"


def validate(entries: Sequence | None = None) -> None:
    """Reject a declaration the documentation tables cannot describe."""

    registry = _registry()
    if entries is None:
        entries = registry.declared_entries()
    keys = {entry.key for entry in entries}
    undeclared = sorted(set(FAMILY_REQUESTS) ^ keys)
    if undeclared:
        raise ValueError(
            f"the request table does not cover the declared families: {undeclared}"
        )
    undeclared = sorted(set(FAMILY_CAPABILITIES) ^ keys)
    if undeclared:
        raise ValueError(
            f"the capability table does not cover the declared families: {undeclared}"
        )
    documented = set()
    for entry in entries:
        rows = FAMILY_REQUESTS[entry.key]
        modes = [row.mode for row in rows]
        if len(set(modes)) != len(modes):
            raise ValueError(f"family {entry.key!r} documents a mode twice")
        if set(modes) != set(entry.modes.canonical_modes):
            raise ValueError(
                f"family {entry.key!r} declares modes {sorted(entry.modes.canonical_modes)} "
                f"but the request table documents {sorted(modes)}"
            )
        for row in rows:
            if not row.admitted and not row.reason:
                raise ValueError(
                    f"family {entry.key!r} rejects {row.mode!r} without a reason"
                )
            if row.mode == "direct" and not row.artifact and not row.lammps:
                raise ValueError(
                    f"family {entry.key!r} documents no artifact for {row.mode!r}"
                )
        capabilities = FAMILY_CAPABILITIES[entry.key]
        for row in capabilities:
            if row.key not in registry.CAPABILITY_KEYS:
                raise ValueError(
                    f"family {entry.key!r} documents undeclared capability {row.key!r}"
                )
            documented.add(row.key)
    missing = sorted(set(registry.CAPABILITY_KEYS) - documented)
    if missing:
        raise ValueError(f"the capability table omits the declared keys {missing}")


def apply(document: str, generated: str) -> str:
    """Return ``document`` with its generated block replaced."""

    begin = document.find(BEGIN_MARKER)
    end = document.find(END_MARKER)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError(f"the document has no {BEGIN_MARKER} ... {END_MARKER} block")
    if document.count(BEGIN_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise ValueError("the document carries more than one generated block")
    return document[:begin] + generated + document[end + len(END_MARKER) + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the document is current without writing it",
    )
    arguments = parser.parse_args(argv)
    current = DOCUMENT.read_text()
    expected = apply(current, render_matrix())
    if arguments.check:
        if current != expected:
            print(
                f"{DOCUMENT} is not current; run "
                "tools/generate_execution_support_matrix.py",
                file=sys.stderr,
            )
            return 1
        return 0
    if current != expected:
        DOCUMENT.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
