#!/usr/bin/env python3
"""Measure a clean rebuild of selected MACE Kokkos translation units."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import time


OBJECT_PREFIX = "libsymmetrix/CMakeFiles/symmetrix.dir/source/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--target", default="symmetrix_bindings")
    parser.add_argument(
        "--owner",
        action="append",
        required=True,
        help="MACE owner basename without .cpp; repeat for every clean object",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.05,
        help="Process-tree RSS sampling interval in seconds",
    )
    return parser.parse_args()


def cache_value(cache: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}(?::[^=]+)?=(.*)$")
    for line in cache.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    raise RuntimeError(f"{key} is absent from {cache}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_tree_rss_kib(root_pid: int) -> tuple[int, int, int]:
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            status = (entry / "status").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
        processes[int(entry.name)] = (
            int(fields[1]),
            int(match.group(1)) if match else 0,
        )

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _rss) in processes.items():
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    resident = [processes[pid][1] for pid in descendants if pid in processes]
    total = sum(resident)
    maximum = max(resident, default=0)
    count = len(resident)
    return total, maximum, count


def parse_ninja_log(log_text: str, owners: list[str]) -> tuple[list[dict], list[dict]]:
    records = []
    owner_suffixes = {f"{OBJECT_PREFIX}{owner}.cpp.o": owner for owner in owners}
    for line in log_text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        start_ms, end_ms, _mtime, output = fields[:4]
        record = {
            "output": output,
            "start_seconds": int(start_ms) / 1000,
            "end_seconds": int(end_ms) / 1000,
            "duration_seconds": (int(end_ms) - int(start_ms)) / 1000,
        }
        records.append(record)
        if output in owner_suffixes:
            record["owner"] = owner_suffixes[output]
    records.sort(key=lambda item: (item["start_seconds"], item["output"]))
    owner_records = [record for record in records if "owner" in record]
    return records, owner_records


def write_timeline(path: Path, records: list[dict]) -> None:
    with path.open("w") as handle:
        handle.write("start_seconds\tduration_seconds\tend_seconds\toutput\n")
        for record in records:
            handle.write(
                f'{record["start_seconds"]:.3f}\t'
                f'{record["duration_seconds"]:.3f}\t'
                f'{record["end_seconds"]:.3f}\t{record["output"]}\n'
            )


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive")

    build_dir = args.build_dir.resolve()
    source_dir = args.source_dir.resolve()
    result_dir = (args.output_dir / args.label).resolve()
    result_dir.mkdir(parents=True, exist_ok=False)

    cache = build_dir / "CMakeCache.txt"
    ninja_log = build_dir / ".ninja_log"
    ninja = Path(cache_value(cache, "CMAKE_MAKE_PROGRAM"))
    objects = [f"{OBJECT_PREFIX}{owner}.cpp.o" for owner in args.owner]

    configure_command = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    with (result_dir / "configure.log").open("w") as configure_log:
        subprocess.run(
            configure_command,
            stdout=configure_log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    build_ninja = (build_dir / "build.ninja").read_text()
    missing = [target for target in objects if f"build {target}:" not in build_ninja]
    if missing:
        raise RuntimeError(f"Ninja graph is missing owner objects: {missing}")

    dry_run = subprocess.run(
        [str(ninja), "-C", str(build_dir), "-n", args.target],
        capture_output=True,
        text=True,
        check=True,
    )
    (result_dir / "preflight.log").write_text(dry_run.stdout + dry_run.stderr)
    pending_compile = (
        "Building " in dry_run.stdout
        or "Linking CXX static library" in dry_run.stdout
        or ".cpp.o" in dry_run.stdout
    )
    if "no work to do" not in dry_run.stdout and pending_compile:
        raise RuntimeError(
            "Build target has pending compile or static-library work before "
            "the owner-only clean; "
            f"see {result_dir / 'preflight.log'}"
        )

    if ninja_log.exists():
        shutil.copy2(ninja_log, result_dir / "preexisting.ninja_log")
        ninja_log_offset = ninja_log.stat().st_size
    else:
        ninja_log_offset = 0
    for target in objects:
        (build_dir / target).unlink(missing_ok=False)

    command = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        args.target,
        "-j",
        str(args.jobs),
    ]
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    peak_tree_rss_kib = 0
    peak_single_rss_kib = 0
    peak_process_count = 0
    with (result_dir / "build.log").open("w") as build_log:
        process = subprocess.Popen(
            command,
            stdout=build_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            tree_rss, single_rss, process_count = process_tree_rss_kib(process.pid)
            if tree_rss > peak_tree_rss_kib:
                peak_tree_rss_kib = tree_rss
                peak_process_count = process_count
            peak_single_rss_kib = max(peak_single_rss_kib, single_rss)
            time.sleep(args.sample_interval)
        return_code = process.wait()
    wall_seconds = time.monotonic() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    if not ninja_log.exists():
        raise RuntimeError("Build did not produce a Ninja log")
    shutil.copy2(ninja_log, result_dir / "ninja.log")
    with ninja_log.open() as handle:
        handle.seek(ninja_log_offset)
        current_log = handle.read()
    if not current_log:
        raise RuntimeError("Ninja did not append timing records for the measured build")
    (result_dir / "measured.ninja_log").write_text(current_log)
    records, owner_records = parse_ninja_log(current_log, args.owner)
    write_timeline(result_dir / "timeline.tsv", records)
    write_timeline(result_dir / "owner_timeline.tsv", owner_records)

    end_times = sorted(record["end_seconds"] for record in owner_records)
    idle_tail = end_times[-1] - end_times[-2] if len(end_times) > 1 else 0.0
    summary = {
        "label": args.label,
        "command": command,
        "configure_command": configure_command,
        "target": args.target,
        "jobs": args.jobs,
        "return_code": return_code,
        "wall_seconds": wall_seconds,
        "user_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "system_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "peak_process_tree_rss_kib": peak_tree_rss_kib,
        "peak_single_process_rss_kib": peak_single_rss_kib,
        "process_count_at_peak_rss": peak_process_count,
        "aggregate_owner_compiler_seconds": sum(
            record["duration_seconds"] for record in owner_records
        ),
        "owner_idle_tail_seconds": idle_tail,
        "ninja_completion_seconds": max(
            (record["end_seconds"] for record in records), default=0.0
        ),
        "owners": owner_records,
        "source_sha256": {
            owner: sha256(
                source_dir.parent / "libsymmetrix" / "source" / f"{owner}.cpp"
            )
            for owner in args.owner
        },
        "build_dir": str(build_dir),
        "source_dir": str(source_dir),
        "cmake_build_type": cache_value(cache, "CMAKE_BUILD_TYPE"),
        "cmake_generator": cache_value(cache, "CMAKE_GENERATOR"),
        "ninja": str(ninja),
        "host": platform.uname()._asdict(),
        "cpu_count": os.cpu_count(),
        "sample_interval_seconds": args.sample_interval,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
