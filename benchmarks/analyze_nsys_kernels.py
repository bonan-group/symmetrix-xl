#!/usr/bin/env python3
"""Summarize Nsight Systems CUDA kernels for the LES stage in MACELES."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCHEMA = "symmetrix.nsys-les-kernel-summary/2"
LES_CATEGORIES = {
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
DISPERSION = re.compile(r"(?:d3|dispersion)", re.IGNORECASE)


def _display_name(category):
    if category == "assemble_maceles":
        return "assemble MACELES"
    return "LES stage: " + category.replace("_", " ")


class AnalysisError(ValueError):
    pass


def _field(row, names, line):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    raise AnalysisError(f"line {line}: missing one of {', '.join(names)}")


def _integer(value, label, line):
    try:
        result = int(str(value).replace(",", ""))
    except ValueError as error:
        raise AnalysisError(f"line {line}: {label} must be an integer") from error
    if result < 0:
        raise AnalysisError(f"line {line}: {label} must be non-negative")
    return result


def read_nsys_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if "Name" not in fields:
            raise AnalysisError("Nsight Systems CSV must contain Name")
        rows = []
        for line, row in enumerate(reader, 2):
            name = row.get("Name", "").strip()
            if not name:
                raise AnalysisError(f"line {line}: kernel name must be non-empty")
            calls = _integer(
                _field(row, ("Instances", "Calls"), line), "instances", line
            )
            duration = _integer(
                _field(row, ("Total Time (ns)", "Total Time", "TotalDurationNs"), line),
                "total duration",
                line,
            )
            rows.append({"name": name, "calls": calls, "duration_ns": duration})
    if not rows:
        raise AnalysisError("Nsight Systems CSV contains no kernel rows")
    return rows


def summarize(path, *, evaluation_count):
    if (
        isinstance(evaluation_count, bool)
        or not isinstance(evaluation_count, int)
        or evaluation_count <= 0
    ):
        raise AnalysisError("selected evaluation count must be a positive integer")
    kernels = read_nsys_csv(path)
    if any(DISPERSION.search(row["name"]) for row in kernels):
        raise AnalysisError("selected LES-stage region contains D3/dispersion kernels")
    categories = []
    for category, pattern in LES_CATEGORIES.items():
        matches = [row for row in kernels if pattern.search(row["name"])]
        calls = sum(row["calls"] for row in matches)
        duration = sum(row["duration_ns"] for row in matches)
        categories.append(
            {
                "category": category,
                "display_name": _display_name(category),
                "calls": calls,
                "duration_ns": duration,
                "calls_per_evaluation": calls / evaluation_count,
                "duration_ns_per_evaluation": duration / evaluation_count,
            }
        )
    required = {"structure_factors", "atom_outputs", "strain"}
    missing = [
        record["category"]
        for record in categories
        if record["category"] in required and record["calls"] <= 0
    ]
    if missing:
        raise AnalysisError(
            "selected region is missing steady-state LES categories: "
            + ", ".join(missing)
        )
    return {
        "schema": SCHEMA,
        "source": str(Path(path).resolve()),
        "selected_evaluation_count": evaluation_count,
        "totals": {
            "calls": sum(row["calls"] for row in kernels),
            "duration_ns": sum(row["duration_ns"] for row in kernels),
        },
        "les_stage_categories": categories,
        "kernels": kernels,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--evaluations", type=int, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.csv, evaluation_count=args.evaluations)
    except AnalysisError as error:
        parser.error(str(error))
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.json_output == Path("-"):
        sys.stdout.write(rendered)
    elif args.json_output is not None:
        args.json_output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
