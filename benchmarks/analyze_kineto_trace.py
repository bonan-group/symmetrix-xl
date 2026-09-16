#!/usr/bin/env python3
"""Summarize a one-step MACE Kineto Chrome trace without double-counting."""

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "symmetrix.kineto-mace-summary"
SCHEMA_VERSION = 1
FORWARD_RANGE = "symmetrix:pytorch_mace:forward"
MODULE_PREFIX = "mace.module:"


class AnalysisError(ValueError):
    """Raised when a Chrome trace cannot be attributed unambiguously."""


def _external_id(event):
    return event.get("args", {}).get("External id")


def _interval(event):
    try:
        start = float(event["ts"])
        duration = float(event["dur"])
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisError("complete events must have numeric ts and dur") from error
    if not math.isfinite(start) or not math.isfinite(duration) or duration < 0.0:
        raise AnalysisError("event timestamps must be finite and duration non-negative")
    return start, start + duration


def _contains(container, event):
    container_start, container_end = _interval(container)
    event_start, event_end = _interval(event)
    return container_start <= event_start and event_end <= container_end


def _record(events):
    return {
        "launches": len(events),
        "device_time_us": sum(float(event["dur"]) for event in events),
    }


def _load_trace(path):
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read Chrome trace {path}: {error}") from error
    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise AnalysisError("Chrome trace must contain a traceEvents array")
    return path, events


def summarize(path):
    """Return a stable summary of MACE pathway and module device work."""
    path, events = _load_trace(path)
    forward_ranges = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name") == FORWARD_RANGE
    ]
    if len(forward_ranges) != 1:
        raise AnalysisError(
            f"expected exactly one CPU {FORWARD_RANGE!r} range, "
            f"found {len(forward_ranges)}"
        )
    forward_range = forward_ranges[0]
    main_process = forward_range.get("pid")
    main_thread = forward_range.get("tid")

    cpu_ops_by_external_id = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        external_id = _external_id(event)
        if external_id is None:
            continue
        cpu_ops_by_external_id.setdefault(external_id, []).append(event)

    module_ranges = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith(MODULE_PREFIX)
    ]
    product_ranges = [
        event
        for event in module_ranges
        if event["name"].startswith(f"{MODULE_PREFIX}products.")
    ]
    interaction_ranges = [
        event
        for event in module_ranges
        if event["name"].startswith(f"{MODULE_PREFIX}interactions.")
    ]
    if not product_ranges or not interaction_ranges:
        raise AnalysisError("trace lacks product or interaction MACE module ranges")

    kernels = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "kernel"
    ]
    if not kernels:
        raise AnalysisError("Chrome trace contains no device kernel events")

    forward = []
    reverse = []
    products = []
    interactions = []
    other_forward = []
    product_copy = []
    product_other = []
    reverse_threads = set()

    for kernel in kernels:
        _interval(kernel)
        external_id = _external_id(kernel)
        if external_id is None:
            raise AnalysisError("device kernel is missing External id")
        cpu_ops = cpu_ops_by_external_id.get(external_id, ())
        if len(cpu_ops) != 1:
            raise AnalysisError(
                f"kernel External id {external_id!r} maps to {len(cpu_ops)} CPU ops; "
                "expected exactly one"
            )
        cpu_op = cpu_ops[0]
        if cpu_op.get("pid") != main_process or not _contains(forward_range, cpu_op):
            raise AnalysisError(
                f"CPU op for kernel External id {external_id!r} is outside the "
                "measured forward range"
            )

        is_main_thread = cpu_op.get("tid") == main_thread
        if is_main_thread:
            forward.append(kernel)
        else:
            reverse.append(kernel)
            reverse_threads.add(cpu_op.get("tid"))

        in_product = any(
            module.get("pid") == cpu_op.get("pid")
            and module.get("tid") == cpu_op.get("tid")
            and _contains(module, cpu_op)
            for module in product_ranges
        )
        in_interaction = any(
            module.get("pid") == cpu_op.get("pid")
            and module.get("tid") == cpu_op.get("tid")
            and _contains(module, cpu_op)
            for module in interaction_ranges
        )
        if in_product and in_interaction:
            raise AnalysisError(
                f"CPU op for kernel External id {external_id!r} is contained by "
                "both product and interaction modules"
            )
        if not is_main_thread and (in_product or in_interaction):
            raise AnalysisError("reverse kernel was attributed to a forward module")
        if is_main_thread:
            if in_product:
                products.append(kernel)
                if cpu_op.get("name") == "aten::copy_":
                    product_copy.append(kernel)
                else:
                    product_other.append(kernel)
            elif in_interaction:
                interactions.append(kernel)
            else:
                other_forward.append(kernel)

    if len(forward) + len(reverse) != len(kernels):
        raise AnalysisError("not every device kernel received one pathway assignment")
    if len(products) + len(interactions) + len(other_forward) != len(forward):
        raise AnalysisError("not every forward kernel received one module assignment")
    if len(product_copy) + len(product_other) != len(products):
        raise AnalysisError("not every product kernel received one operator assignment")

    return {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "source": str(path.resolve()),
        "forward_range": FORWARD_RANGE,
        "threads": {
            "process_id": main_process,
            "main_thread_id": main_thread,
            "autograd_thread_ids": sorted(reverse_threads, key=str),
        },
        "totals": _record(kernels),
        "pathways": {
            "forward": _record(forward),
            "autograd_reverse": _record(reverse),
        },
        "forward_modules": {
            "products": _record(products),
            "interactions": _record(interactions),
            "other": _record(other_forward),
        },
        "product_operators": {
            "aten::copy_": _record(product_copy),
            "other": _record(product_other),
        },
        "invariants": {
            "device_kernels_assigned_once": True,
            "forward_kernels_assigned_once": True,
        },
    }


def format_table(summary):
    """Format the trace summary as a stable human-readable table."""
    rows = [
        ("whole_step", summary["totals"]),
        ("forward", summary["pathways"]["forward"]),
        ("autograd_reverse", summary["pathways"]["autograd_reverse"]),
        ("forward/products", summary["forward_modules"]["products"]),
        ("forward/interactions", summary["forward_modules"]["interactions"]),
        ("forward/other", summary["forward_modules"]["other"]),
        ("products/aten::copy_", summary["product_operators"]["aten::copy_"]),
        ("products/other", summary["product_operators"]["other"]),
    ]
    total_time = summary["totals"]["device_time_us"]
    lines = [
        f"Source: {summary['source']}",
        "",
        f"{'Pathway':<24} {'Launches':>10} {'Device ms':>13} {'Share':>8}",
        f"{'-' * 24} {'-' * 10} {'-' * 13} {'-' * 8}",
    ]
    for name, record in rows:
        share = 100.0 * record["device_time_us"] / total_time if total_time else 0.0
        lines.append(
            f"{name:<24} {record['launches']:>10d} "
            f"{record['device_time_us'] / 1000.0:>13.3f} {share:>7.2f}%"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Kineto Chrome trace JSON")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="write the machine-readable summary to this path; use '-' for stdout",
    )
    arguments = parser.parse_args(argv)
    try:
        summary = summarize(arguments.trace)
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
