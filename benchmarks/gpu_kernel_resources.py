"""Normalize HIP and CUDA compiler resource reports and enforce limits."""

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "symmetrix.gpu-kernel-resources"
SCHEMA_VERSION = 1


class ResourceReportError(ValueError):
    """Raised when compiler resource metadata is malformed or incomplete."""


_HIP_FIELD_MAP = {
    "private_segment_fixed_size": "private_bytes",
    "group_segment_fixed_size": "shared_bytes",
    "vgpr_count": "registers",
    "vgpr_spill_count": "vgpr_spills",
    "sgpr_count": "scalar_registers",
    "sgpr_spill_count": "sgpr_spills",
}


def _nonnegative_integer(value, field, kernel):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ResourceReportError(
            f"{kernel}: {field} must be an integer, got {value!r}"
        ) from error
    if result < 0:
        raise ResourceReportError(f"{kernel}: {field} must be non-negative")
    return result


def parse_hip_metadata(text):
    """Parse the YAML-like AMDGPU metadata printed by ``llvm-readobj --notes``."""
    kernels = []
    current = None
    for line in text.splitlines():
        if re.match(r"\s*-\s*\.args:\s*$", line):
            if current is not None and "name" in current:
                kernels.append(current)
            current = {}
            continue
        match = re.match(r"\s*(-\s*)?\.name:\s*(.+?)\s*$", line)
        if match:
            if match.group(1) and current is not None and "name" in current:
                kernels.append(current)
                current = {}
            elif current is None:
                current = {}
            current["name"] = match.group(2).strip("'\"")
            continue
        if current is None:
            continue
        match = re.match(r"\s*\.([a-z_]+):\s*(\d+)\s*$", line)
        if match and match.group(1) in _HIP_FIELD_MAP:
            field = _HIP_FIELD_MAP[match.group(1)]
            current[field] = _nonnegative_integer(
                match.group(2), field, current.get("name", "<pending kernel>")
            )
    if current is not None and "name" in current:
        kernels.append(current)
    if not kernels:
        raise ResourceReportError("no AMDGPU kernel metadata found")
    required = set(_HIP_FIELD_MAP.values())
    for kernel in kernels:
        missing = sorted(required - kernel.keys())
        if missing:
            raise ResourceReportError(
                f"{kernel['name']}: missing HIP resource fields: {', '.join(missing)}"
            )
        kernel["stack_bytes"] = 0
        kernel["local_bytes"] = kernel["private_bytes"]
    return kernels


def parse_cuda_resources(text):
    """Parse ``cuobjdump --dump-resource-usage`` output."""
    kernels = []
    current = None
    for line in text.splitlines():
        match = re.match(r"\s*Function(?:\s*:\s*|\s+)(.+?)\s*$", line)
        if match:
            if current is not None:
                kernels.append(current)
            current = {"name": match.group(1).strip().removesuffix(":")}
            continue
        if current is None:
            continue
        fields = dict(re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)\b", line))
        if fields:
            current.update(
                registers=_nonnegative_integer(
                    fields.get("REG", 0), "REG", current["name"]
                ),
                stack_bytes=_nonnegative_integer(
                    fields.get("STACK", 0), "STACK", current["name"]
                ),
                shared_bytes=_nonnegative_integer(
                    fields.get("SHARED", 0), "SHARED", current["name"]
                ),
                local_bytes=_nonnegative_integer(
                    fields.get("LOCAL", 0), "LOCAL", current["name"]
                ),
                private_bytes=_nonnegative_integer(
                    fields.get("STACK", 0), "STACK", current["name"]
                )
                + _nonnegative_integer(
                    fields.get("LOCAL", 0), "LOCAL", current["name"]
                ),
                vgpr_spills=None,
                sgpr_spills=None,
                scalar_registers=None,
            )
    if current is not None:
        kernels.append(current)
    if not kernels:
        raise ResourceReportError("no CUDA function resource records found")
    for kernel in kernels:
        if "registers" not in kernel:
            raise ResourceReportError(
                f"{kernel['name']}: missing CUDA resource-usage line"
            )
    return kernels


def select_kernels(kernels, patterns):
    if not patterns:
        return list(kernels)
    compiled = [re.compile(pattern) for pattern in patterns]
    return [
        kernel
        for kernel in kernels
        if any(pattern.search(kernel["name"]) for pattern in compiled)
    ]


def check_limits(kernels, limits):
    violations = []
    for kernel in kernels:
        for field, maximum in limits.items():
            if maximum is None or kernel.get(field) is None:
                continue
            if kernel[field] > maximum:
                violations.append(
                    {
                        "name": kernel["name"],
                        "field": field,
                        "actual": kernel[field],
                        "maximum": maximum,
                    }
                )
    return violations


def summarize(backend, text, patterns=(), limits=None):
    parsers = {"hip": parse_hip_metadata, "cuda": parse_cuda_resources}
    kernels = parsers[backend](text)
    selected = select_kernels(kernels, patterns)
    if patterns and not selected:
        raise ResourceReportError("no kernel matched the requested patterns")
    limits = dict(limits or {})
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "kernel_count": len(kernels),
        "selected_kernel_count": len(selected),
        "patterns": list(patterns),
        "limits": limits,
        "violations": check_limits(selected, limits),
        "kernels": selected,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("hip", "cuda"))
    parser.add_argument("report", type=Path)
    parser.add_argument("--match", action="append", default=[])
    parser.add_argument("--output", type=Path)
    for field in (
        "registers",
        "scalar-registers",
        "private-bytes",
        "stack-bytes",
        "local-bytes",
        "shared-bytes",
        "vgpr-spills",
        "sgpr-spills",
    ):
        parser.add_argument(f"--max-{field}", type=int)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    limits = {
        name.removeprefix("max_"): value
        for name, value in vars(args).items()
        if name.startswith("max_") and value is not None
    }
    try:
        report = summarize(
            args.backend,
            args.report.read_text(encoding="utf-8"),
            args.match,
            limits,
        )
    except (OSError, ResourceReportError, re.error) as error:
        print(f"gpu-kernel-resources: {error}", file=sys.stderr)
        return 2
    output = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(output)
    else:
        args.output.write_text(output, encoding="utf-8")
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
