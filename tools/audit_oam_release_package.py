#!/usr/bin/env python3
"""Audit an installed repaired wheel's minimal OAM release catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _symmetrix_build.prebuilt_operator_artifacts import (
    audit_installed_oam_release_catalog,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--backend", choices=("host", "cuda"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    probe = json.loads(args.probe_json.read_text(encoding="utf-8"))
    if probe.get("probe_status") != "ok":
        raise ValueError("installed backend probe did not succeed")
    native = probe["probe"]
    native_build_info = native["build_info"]
    build_info = native_build_info["prebuilt_operator_artifacts"]
    extension = Path(native["extension"])
    wheel = args.wheel
    if wheel.is_symlink() or not wheel.is_file():
        raise ValueError("repaired wheel must be a regular file")
    result = audit_installed_oam_release_catalog(
        native_extension_path=extension,
        build_info=build_info,
        backend_descriptor=native["backend"],
        native_build_info=native_build_info,
        reported_validation=native["installed_operator_artifacts"],
        wheel_bytes=wheel.stat().st_size,
        backend=args.backend,
        target=args.target,
    )
    result.update(
        source_content_sha256=build_info["source_content_sha256"],
        native_source_content_sha256=build_info["native_source_content_sha256"],
        wheel=str(wheel.resolve()),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
