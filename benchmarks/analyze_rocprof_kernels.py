#!/usr/bin/env python3
"""Aggregate rocprofv3 kernel CSVs into auditable semantic categories.

The classifier is intentionally conservative.  A kernel is categorized only when its
name contains a backend-independent Symmetrix operation name or a sufficiently
specific PyTorch/e3nn operation name.  Generic ATen kernels remain ``unclassified``.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "symmetrix.rocprof-kernel-summary"
SCHEMA_VERSION = 2
LES_CATEGORY_ORDER = (
    "preparation",
    "structure_factors",
    "atom_outputs",
    "strain",
    "readout",
    "assemble_maceles",
)
LES_CATEGORY_PATTERNS = {
    "preparation": re.compile(
        r"(?:LES reciprocal table|Count LES reciprocal vectors|LES periodic preparation)",
        re.IGNORECASE,
    ),
    "structure_factors": re.compile(r"LES structure factors", re.IGNORECASE),
    "atom_outputs": re.compile(r"LES (?:periodic )?atom outputs", re.IGNORECASE),
    "strain": re.compile(r"LES .*strain", re.IGNORECASE),
    "readout": re.compile(r"LES .*readout", re.IGNORECASE),
    "assemble_maceles": re.compile(
        r"(?:Compose LES|LES (?:compose|composition))", re.IGNORECASE
    ),
}
DISPERSION_PATTERN = re.compile(r"(?:d3|dispersion)", re.IGNORECASE)


def _les_display_name(category):
    if category == "assemble_maceles":
        return "assemble MACELES"
    return "LES stage: " + category.replace("_", " ")


CATEGORY_ORDER = (
    "common_geometry_harmonics",
    "common_zbl",
    "common_runtime_copy_fill",
    "r0_forward",
    "m0_forward",
    "h1_forward",
    "conditioning_forward",
    "interaction_forward",
    "a1_forward",
    "m1_forward",
    "node_forward",
    "h2_forward",
    "readout_reduction",
    "h2_reverse",
    "m1_reverse",
    "node_reverse",
    "a1_reverse",
    "r1_reverse",
    "source_reverse",
    "edge_reverse_compact_fused",
    "edge_reverse_phi",
    "edge_reverse_harmonic",
    "conditioning_reverse",
    "h1_reverse",
    "m0_reverse",
    "r0_reverse",
    "common_force_assembly",
    "blas",
    "unclassified",
)

# Category IDs are part of the versioned JSON output.  Keep them stable and use
# these labels when presenting stages alongside the standard MACE pipeline.
CATEGORY_DISPLAY_NAMES = {
    "common_geometry_harmonics": "Common: geometry/harmonics",
    "common_force_assembly": "Common: force assembly",
    "common_zbl": "Common: ZBL",
    "common_runtime_copy_fill": "Runtime: copy/fill",
    "conditioning_forward": "R1 edge conditioning forward",
    "conditioning_reverse": "R1 edge conditioning reverse",
    "r0_forward": "R0 forward",
    "m0_forward": "M0 forward",
    "h1_forward": "H1 forward",
    "interaction_forward": "R1 forward",
    "a1_forward": "A1 forward",
    "m1_forward": "M1 forward",
    "source_reverse": "R1 reverse: source",
    "r1_reverse": "R1 reverse",
    "edge_reverse_compact_fused": "R1 reverse: edge (compact fused)",
    "edge_reverse_phi": "R1 reverse: edge phi",
    "edge_reverse_harmonic": "R1 reverse: edge harmonic",
    "node_forward": "M1/node forward",
    "h2_forward": "H2 forward",
    "h2_reverse": "H2 reverse",
    "m1_reverse": "M1 reverse",
    "node_reverse": "M1/node reverse",
    "a1_reverse": "A1 reverse",
    "h1_reverse": "H1 reverse",
    "m0_reverse": "M0 reverse",
    "r0_reverse": "R0 reverse",
    "readout_reduction": "Readout/reduction",
    "blas": "BLAS",
    "unclassified": "Unclassified",
}

MH1_STAGE_ORDER = (
    "r0_forward",
    "m0_forward",
    "r1_forward",
    "m1_forward",
    "m1_reverse",
    "r1_reverse",
    "m0_reverse",
    "r0_reverse",
)
MH1_STAGE_DISPLAY_NAMES = {
    "r0_forward": "R0 forward",
    "m0_forward": "M0 forward",
    "r1_forward": "R1 forward",
    "m1_forward": "M1 forward",
    "m1_reverse": "M1 reverse",
    "r1_reverse": "R1 reverse",
    "m0_reverse": "M0 reverse",
    "r0_reverse": "R0 reverse",
}

_MH1_STAGE_REGION = re.compile(
    r"^symmetrix/mh1/(?P<owner>[RM])(?P<layer>[01])/(?P<direction>forward|reverse)$"
)
_MH1_R_STAGE_KERNEL = re.compile(
    r"^symmetrix_execution_mh1_(?:conditioning_(?P<conditioning_direction>forward|reverse)(?:_[a-z]+)*_kernel_|spline_r_(?P<spline_direction>forward|reverse)_kernel_|(?P<forward>forward_kernel_)|(?P<reverse>(?:source_reverse_kernel_|edge_reverse(?:_[a-z]+)*_kernel_)))(?P<layer>[01])(?:_|$)"
)
_MH1_M_STAGE_KERNEL = re.compile(
    r"^symmetrix_execution_mh1_node_(?:pre_(?P<pre_direction>forward|reverse)_kernel_(?P<pre_layer>[01])|tiled_l(?P<tiled_layer>[01])_(?P<tiled_direction>forward|reverse)(?:_|$))"
)


class AnalysisError(ValueError):
    """Raised when a CSV is not a supported rocprofv3 kernel report."""


@dataclass(frozen=True)
class CategoryRule:
    rule_id: str
    category: str
    pattern: re.Pattern[str]


def _rule(rule_id, category, pattern):
    return CategoryRule(rule_id, category, re.compile(pattern, re.IGNORECASE))


# Keep rules tied to semantic operation names.  In particular, generic names such as
# vectorized_elementwise_kernel, gemm, scatter, and index_add are ambiguous across
# MACE phases and must not be guessed from their implementation primitive.
CATEGORY_RULES = (
    _rule(
        "common.geometry_harmonics",
        "common_geometry_harmonics",
        r"(?:compute_Y|spherical[_ ]?harmonic|radial[_ ]?basis|bessel|compute_edge_vectors)",
    ),
    _rule(
        "common.force_assembly",
        "common_force_assembly",
        r"(?:compute_node_energies_forces|assemble[_ ]?forces)",
    ),
    _rule(
        "common.zbl",
        "common_zbl",
        r"(?:ZBLKokkos|compute_ZBL|(?:^|[:_])zbl(?:$|[:_<(]))",
    ),
    _rule(
        "runtime.copy_fill",
        "common_runtime_copy_fill",
        r"(?:__amd_rocclr_(?:copy|fill)Buffer|hipMemset|hipMemcpy|cudaMemset|cudaMemcpy)",
    ),
    _rule(
        "mh1.conditioning_forward",
        "conditioning_forward",
        r"symmetrix_execution_mh1_conditioning_forward",
    ),
    _rule(
        "mh1.conditioning_reverse",
        "conditioning_reverse",
        r"symmetrix_execution_mh1_conditioning_reverse",
    ),
    _rule(
        "mh1.spline_r0_forward",
        "r0_forward",
        r"symmetrix_execution_mh1_spline_r_forward_kernel_0(?:_|$)",
    ),
    _rule(
        "mh1.spline_r1_forward",
        "interaction_forward",
        r"symmetrix_execution_mh1_spline_r_forward_kernel_1(?:_|$)",
    ),
    _rule(
        "mh1.spline_r0_reverse",
        "r0_reverse",
        r"symmetrix_execution_mh1_spline_r_reverse_kernel_0(?:_|$)",
    ),
    _rule(
        "mh1.spline_r1_reverse",
        "r1_reverse",
        r"symmetrix_execution_mh1_spline_r_reverse_kernel_1(?:_|$)",
    ),
    _rule(
        "symmetrix.interaction_forward",
        "interaction_forward",
        r"(?:symmetrix_execution_mh1_forward_kernel|symmetrix_jit_r1_forward|symmetrix_factorized_forward_v1)",
    ),
    _rule(
        "standard.r0_forward",
        "r0_forward",
        r"standard_r0::launch_(?:density_prepare|forward)",
    ),
    _rule(
        "standard.m0_forward",
        "m0_forward",
        r"standard_m0::launch_forward",
    ),
    _rule(
        "standard.h1_forward",
        "h1_forward",
        r"MACEKokkos<.*>::compute_H1(?:_product|_linear_up)?\(",
    ),
    _rule(
        "standard.a1_forward",
        "a1_forward",
        r"MACEKokkos<.*>::compute_A1(?:_scaled)?\(",
    ),
    _rule(
        "standard.m1_forward",
        "m1_forward",
        r"(?:standard_m1::launch_forward|MACEKokkos<.*>::compute_M1\()",
    ),
    _rule(
        "standard.h2_forward",
        "h2_forward",
        r"MACEKokkos<.*>::compute_H2\(",
    ),
    _rule(
        "symmetrix.source_reverse",
        "source_reverse",
        r"(?:symmetrix_execution_mh1_source_reverse|symmetrix_jit_r1_source)",
    ),
    _rule(
        "standard.r1_reverse",
        "r1_reverse",
        r"symmetrix_factorized_reverse(?:_fused)?_v1",
    ),
    _rule(
        "mh1.edge_reverse_compact_fused",
        "edge_reverse_compact_fused",
        r"symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_",
    ),
    _rule(
        "mh1.edge_reverse_phi",
        "edge_reverse_phi",
        r"symmetrix_execution_mh1_edge_reverse_phi",
    ),
    _rule(
        "mh1.edge_reverse_harmonic",
        "edge_reverse_harmonic",
        r"symmetrix_execution_mh1_edge_reverse_harmonic",
    ),
    _rule(
        "mh1.node_forward",
        "node_forward",
        r"symmetrix_execution_mh1_node_(?:pre_forward|tiled_(?!.*_reverse).*_forward)",
    ),
    _rule(
        "mh1.node_reverse",
        "node_reverse",
        r"symmetrix_execution_mh1_node_(?:pre_reverse|tiled_(?!.*_reverse_readout).*_reverse)",
    ),
    _rule(
        "standard.h2_reverse",
        "h2_reverse",
        r"MACEKokkos<.*>::reverse_H2\(",
    ),
    _rule(
        "standard.m1_reverse",
        "m1_reverse",
        r"(?:standard_m1::launch_reverse|MACEKokkos<.*>::reverse_M1\()",
    ),
    _rule(
        "standard.a1_reverse",
        "a1_reverse",
        r"MACEKokkos<.*>::reverse_A1(?:_from|_scaled)?\(",
    ),
    _rule(
        "standard.h1_reverse",
        "h1_reverse",
        r"MACEKokkos<.*>::reverse_H1(?:_product|_linear_up)?\(",
    ),
    _rule(
        "standard.m0_reverse",
        "m0_reverse",
        r"standard_m0::launch_reverse",
    ),
    _rule(
        "standard.r0_reverse",
        "r0_reverse",
        r"standard_r0::launch_(?:reverse_prepare|coordinate_reverse_edge_owned)",
    ),
    _rule(
        "semantic.readout_reduction",
        "readout_reduction",
        r"(?:compute_readouts|evaluate_gradient|symmetrix_execution_mh1_node_.*_reverse_readout|(?:^|[.:_])readout(?:$|[.:_<(]))",
    ),
    _rule(
        "library.blas",
        "blas",
        r"(?:rocblas|hipblas|cublas|^Cijk_.*ISA[0-9]+)",
    ),
    _rule(
        "e3nn.tensor_product",
        "interaction_forward",
        r"e3nn(?!.*backward).*tensor[_ ]?product",
    ),
    _rule(
        "mace.product_basis",
        "node_forward",
        r"mace.*product[_ ]?basis",
    ),
)


def classify_kernel(name):
    """Return ``(category, rule_id)`` for a kernel name."""
    matches = [rule for rule in CATEGORY_RULES if rule.pattern.search(name)]
    if not matches:
        return "unclassified", None
    categories = {rule.category for rule in matches}
    if len(categories) != 1:
        # Ambiguous names are retained instead of relying on rule ordering.
        return "unclassified", None
    return matches[0].category, matches[0].rule_id


def classify_mh1_stage(name):
    """Return the canonical staged-direct MH-1 owner for a kernel name."""
    match = _MH1_STAGE_REGION.fullmatch(name)
    if match:
        owner = match.group("owner").lower()
        return f"{owner}{match.group('layer')}_{match.group('direction')}"

    match = _MH1_R_STAGE_KERNEL.match(name)
    if match:
        direction = match.group("conditioning_direction") or match.group(
            "spline_direction"
        )
        if direction is None:
            direction = "forward" if match.group("forward") else "reverse"
        return f"r{match.group('layer')}_{direction}"

    match = _MH1_M_STAGE_KERNEL.match(name)
    if match:
        layer = match.group("pre_layer") or match.group("tiled_layer")
        direction = match.group("pre_direction") or match.group("tiled_direction")
        return f"m{layer}_{direction}"
    return None


def _integer(value, label, line_number):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(
            f"line {line_number}: {label} must be an integer"
        ) from error
    if result < 0:
        raise AnalysisError(f"line {line_number}: {label} must be non-negative")
    return result


def _detect_kind(fieldnames):
    fields = set(fieldnames or ())
    if {"Name", "Calls", "TotalDurationNs"} <= fields:
        return "kernel_stats"
    if {"Kernel_Name", "Start_Timestamp", "End_Timestamp"} <= fields:
        return "kernel_trace"
    raise AnalysisError(
        "unsupported CSV columns; expected rocprofv3 kernel stats or kernel trace"
    )


def read_rocprof_csv(path):
    """Read stats or trace rows as name, calls, duration, correlation tuples."""
    path = Path(path)
    try:
        stream = path.open(newline="", encoding="utf-8-sig")
    except OSError as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    with stream:
        reader = csv.DictReader(stream)
        kind = _detect_kind(reader.fieldnames)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if kind == "kernel_stats":
                name = row.get("Name", "")
                calls = _integer(row.get("Calls"), "Calls", line_number)
                duration_ns = _integer(
                    row.get("TotalDurationNs"), "TotalDurationNs", line_number
                )
                correlation_id = None
            else:
                name = row.get("Kernel_Name", "")
                start = _integer(
                    row.get("Start_Timestamp"), "Start_Timestamp", line_number
                )
                end = _integer(row.get("End_Timestamp"), "End_Timestamp", line_number)
                if end < start:
                    raise AnalysisError(
                        f"line {line_number}: End_Timestamp precedes Start_Timestamp"
                    )
                calls = 1
                duration_ns = end - start
                correlation_value = row.get("Correlation_Id")
                correlation_id = (
                    _integer(correlation_value, "Correlation_Id", line_number)
                    if correlation_value not in (None, "")
                    else None
                )
            if not name:
                raise AnalysisError(
                    f"line {line_number}: kernel name must be non-empty"
                )
            rows.append((name, calls, duration_ns, correlation_id))
    if not rows:
        raise AnalysisError("kernel CSV contains no data rows")
    return kind, rows


def read_mh1_stage_correlations(path):
    """Map nested ROCTx range correlations to canonical MH-1 stages."""
    path = Path(path)
    try:
        stream = path.open(newline="", encoding="utf-8-sig")
    except OSError as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    with stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {
            "Domain",
            "Function",
            "Process_Id",
            "Thread_Id",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        }
        if not required <= fields:
            raise AnalysisError(
                "unsupported marker CSV columns; expected a rocprofv3 marker API trace"
            )
        ranges = []
        for line_number, row in enumerate(reader, start=2):
            if row.get("Domain") != "MARKER_CORE_RANGE_API":
                continue
            start = _integer(row.get("Start_Timestamp"), "Start_Timestamp", line_number)
            end = _integer(row.get("End_Timestamp"), "End_Timestamp", line_number)
            if end < start:
                raise AnalysisError(
                    f"line {line_number}: End_Timestamp precedes Start_Timestamp"
                )
            ranges.append(
                {
                    "function": row.get("Function", ""),
                    "process": row.get("Process_Id"),
                    "thread": row.get("Thread_Id"),
                    "correlation_id": _integer(
                        row.get("Correlation_Id"), "Correlation_Id", line_number
                    ),
                    "start": start,
                    "end": end,
                }
            )
    stage_ranges = [
        {**record, "stage": classify_mh1_stage(record["function"])}
        for record in ranges
        if classify_mh1_stage(record["function"]) is not None
    ]
    correlations = {}
    for record in ranges:
        candidates = [
            stage
            for stage in stage_ranges
            if stage["process"] == record["process"]
            and stage["thread"] == record["thread"]
            and stage["start"] <= record["start"]
            and record["end"] <= stage["end"]
        ]
        if not candidates:
            continue
        owner = min(candidates, key=lambda stage: stage["end"] - stage["start"])
        correlations[record["correlation_id"]] = owner["stage"]
    return correlations


def _les_summary(kernel_records, evaluation_count):
    if evaluation_count is None:
        return None
    if (
        isinstance(evaluation_count, bool)
        or not isinstance(evaluation_count, int)
        or evaluation_count <= 0
    ):
        raise AnalysisError("selected evaluation count must be a positive integer")
    dispersion = [
        record["name"]
        for record in kernel_records
        if DISPERSION_PATTERN.search(record["name"])
    ]
    if dispersion:
        raise AnalysisError("selected LES-stage region contains D3/dispersion kernels")
    result = []
    for category in LES_CATEGORY_ORDER:
        matches = [
            record
            for record in kernel_records
            if LES_CATEGORY_PATTERNS[category].search(record["name"])
        ]
        calls = sum(record["calls"] for record in matches)
        duration = sum(record["duration_ns"] for record in matches)
        result.append(
            {
                "category": category,
                "display_name": _les_display_name(category),
                "calls": calls,
                "duration_ns": duration,
                "calls_per_evaluation": calls / evaluation_count,
                "duration_ns_per_evaluation": duration / evaluation_count,
            }
        )
    required = {"structure_factors", "atom_outputs", "strain"}
    missing = [
        record["category"]
        for record in result
        if record["category"] in required and record["calls"] <= 0
    ]
    if missing:
        raise AnalysisError(
            "selected region is missing steady-state LES categories: "
            + ", ".join(missing)
        )
    return result


def summarize(path, marker_trace=None, evaluation_count=None):
    """Return a JSON-serializable summary for a rocprofv3 CSV."""
    kind, rows = read_rocprof_csv(path)
    stage_correlations = (
        read_mh1_stage_correlations(marker_trace) if marker_trace is not None else {}
    )
    kernels = {}
    for name, calls, duration_ns, correlation_id in rows:
        category, rule_id = classify_kernel(name)
        mh1_stage = classify_mh1_stage(name)
        if mh1_stage is None and correlation_id is not None:
            mh1_stage = stage_correlations.get(correlation_id)
        key = (name, category, rule_id, mh1_stage)
        aggregate = kernels.setdefault(key, {"calls": 0, "duration_ns": 0})
        aggregate["calls"] += calls
        aggregate["duration_ns"] += duration_ns

    total_calls = sum(item["calls"] for item in kernels.values())
    total_duration_ns = sum(item["duration_ns"] for item in kernels.values())
    categories = {
        category: {"calls": 0, "duration_ns": 0, "kernel_count": 0}
        for category in CATEGORY_ORDER
    }
    mh1_stages = {
        stage: {"calls": 0, "duration_ns": 0, "kernel_count": 0}
        for stage in MH1_STAGE_ORDER
    }
    mh1_unassigned = {"calls": 0, "duration_ns": 0, "kernel_count": 0}
    kernel_records = []
    for (name, category, rule_id, mh1_stage), aggregate in sorted(
        kernels.items(), key=lambda item: (-item[1]["duration_ns"], item[0][0])
    ):
        category_record = categories[category]
        category_record["calls"] += aggregate["calls"]
        category_record["duration_ns"] += aggregate["duration_ns"]
        category_record["kernel_count"] += 1
        if mh1_stage is None:
            mh1_stage_record = mh1_unassigned
        else:
            mh1_stage_record = mh1_stages[mh1_stage]
        mh1_stage_record["calls"] += aggregate["calls"]
        mh1_stage_record["duration_ns"] += aggregate["duration_ns"]
        mh1_stage_record["kernel_count"] += 1
        kernel_records.append(
            {
                "name": name,
                "category": category,
                "display_name": CATEGORY_DISPLAY_NAMES[category],
                "rule_id": rule_id,
                "mh1_stage": mh1_stage,
                **aggregate,
                "duration_percent": (
                    100.0 * aggregate["duration_ns"] / total_duration_ns
                    if total_duration_ns
                    else 0.0
                ),
            }
        )
    mh1_stage_duration_ns = sum(record["duration_ns"] for record in mh1_stages.values())
    mh1_stage_records = [
        {
            "stage": stage,
            "display_name": MH1_STAGE_DISPLAY_NAMES[stage],
            **mh1_stages[stage],
            "duration_percent": (
                100.0 * mh1_stages[stage]["duration_ns"] / total_duration_ns
                if total_duration_ns
                else 0.0
            ),
        }
        for stage in MH1_STAGE_ORDER
    ]

    category_records = []
    for category in CATEGORY_ORDER:
        record = categories[category]
        category_records.append(
            {
                "category": category,
                "display_name": CATEGORY_DISPLAY_NAMES[category],
                **record,
                "duration_percent": (
                    100.0 * record["duration_ns"] / total_duration_ns
                    if total_duration_ns
                    else 0.0
                ),
            }
        )
    result = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "source": str(Path(path).resolve()),
        "marker_trace": (
            str(Path(marker_trace).resolve()) if marker_trace is not None else None
        ),
        "input_kind": kind,
        "totals": {"calls": total_calls, "duration_ns": total_duration_ns},
        "categories": category_records,
        "mh1_stages": {
            "available": mh1_stage_duration_ns > 0,
            "assigned_duration_ns": mh1_stage_duration_ns,
            "assigned_duration_percent": (
                100.0 * mh1_stage_duration_ns / total_duration_ns
                if total_duration_ns
                else 0.0
            ),
            "stages": mh1_stage_records,
            "unassigned": {
                **mh1_unassigned,
                "duration_percent": (
                    100.0 * mh1_unassigned["duration_ns"] / total_duration_ns
                    if total_duration_ns
                    else 0.0
                ),
            },
        },
        "kernels": kernel_records,
    }
    if evaluation_count is not None:
        result["selected_evaluation_count"] = evaluation_count
        result["les_stage_categories"] = _les_summary(kernel_records, evaluation_count)
    return result


def format_table(summary):
    """Format the category summary as a stable human-readable table."""
    label_width = max(
        len("Stage"),
        *(
            len(record.get("display_name", record["category"]))
            for record in summary["categories"]
        ),
    )
    lines = [
        f"Source: {summary['source']}",
        f"Input:  {summary['input_kind']}",
        "",
        f"{'Stage':<{label_width}} {'Kernels':>7} {'Calls':>10} {'Duration ms':>13} {'Share':>8}",
        f"{'-' * label_width} {'-' * 7} {'-' * 10} {'-' * 13} {'-' * 8}",
    ]
    for record in summary["categories"]:
        display_name = record.get("display_name", record["category"])
        lines.append(
            f"{display_name:<{label_width}} {record['kernel_count']:>7d} "
            f"{record['calls']:>10d} {record['duration_ns'] / 1e6:>13.3f} "
            f"{record['duration_percent']:>7.2f}%"
        )
    lines.extend(
        [
            "",
            f"Total calls: {summary['totals']['calls']}",
            f"Total duration: {summary['totals']['duration_ns'] / 1e6:.3f} ms",
        ]
    )
    if summary["mh1_stages"]["available"]:
        lines.extend(
            [
                "",
                "Canonical MH-1 staged-direct owners:",
                "",
                (
                    f"{'Stage':<12} {'Kernels':>7} {'Calls':>10} "
                    f"{'Duration ms':>13} {'Share':>8}"
                ),
                f"{'-' * 12} {'-' * 7} {'-' * 10} {'-' * 13} {'-' * 8}",
            ]
        )
        for record in summary["mh1_stages"]["stages"]:
            lines.append(
                f"{record['display_name']:<12} {record['kernel_count']:>7d} "
                f"{record['calls']:>10d} {record['duration_ns'] / 1e6:>13.3f} "
                f"{record['duration_percent']:>7.2f}%"
            )
        unassigned = summary["mh1_stages"]["unassigned"]
        lines.append(
            f"{'Outside stages':<12} {unassigned['kernel_count']:>7d} "
            f"{unassigned['calls']:>10d} {unassigned['duration_ns'] / 1e6:>13.3f} "
            f"{unassigned['duration_percent']:>7.2f}%"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="rocprofv3 kernel stats or trace CSV")
    parser.add_argument(
        "--marker-trace",
        type=Path,
        help="rocprofv3 marker API trace used to resolve nested MH-1 stage owners",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="write the machine-readable summary to this path; use '-' for stdout",
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        help="selected MACELES evaluation count used for LES-stage rates",
    )
    arguments = parser.parse_args(argv)
    try:
        summary = summarize(
            arguments.csv, arguments.marker_trace, arguments.evaluations
        )
    except AnalysisError as error:
        parser.error(str(error))
    if arguments.json_output == Path("-"):
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_table(summary))
        if arguments.json_output is not None:
            arguments.json_output.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
