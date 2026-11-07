#!/usr/bin/env python3
"""Validate release version sources and an optional publication tag."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def release_metadata(
    repository: Path = REPOSITORY_ROOT, *, require_dated: bool = False
) -> dict[str, str]:
    pyproject = tomllib.loads((repository / "symmetrix/pyproject.toml").read_text())
    version = pyproject["project"]["version"]
    expected = {
        "python": f'__version__ = "{version}"',
        "cmake": f'set(SYMMETRIX_FRONTEND_VERSION "{version}" CACHE STRING',
        "docs": f'release = "{version}"',
    }
    sources = {
        "python": repository / "symmetrix/source/symmetrix/__init__.py",
        "cmake": repository / "symmetrix/CMakeLists.txt",
        "docs": repository / "docs/conf.py",
    }
    for name, marker in expected.items():
        if marker not in sources[name].read_text():
            raise ValueError(f"{name} release version does not match {version}")

    changelog = (repository / "CHANGELOG.md").read_text()
    heading = re.search(
        rf"^## {re.escape(version)} - (Unreleased|\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        re.MULTILINE,
    )
    if heading is None:
        raise ValueError(f"CHANGELOG.md has no valid {version} release heading")
    date = heading.group(1)
    if require_dated and date == "Unreleased":
        raise ValueError(f"CHANGELOG.md has no dated {version} release heading")
    if require_dated:
        current_date = datetime.date.fromisoformat(date)
        prior_dates = [
            datetime.date.fromisoformat(match.group(2))
            for match in re.finditer(
                r"^## (\d+\.\d+\.\d+) - (\d{4}-\d{2}-\d{2})$",
                changelog,
                re.MULTILINE,
            )
            if match.group(1) != version
        ]
        if prior_dates and current_date < max(prior_dates):
            raise ValueError(
                f"CHANGELOG.md release date {date} predates the latest prior release "
                f"{max(prior_dates).isoformat()}"
            )
    if require_dated and re.search(
        r"^## \d+\.\d+\.\d+ - Unreleased$", changelog, re.MULTILINE
    ):
        raise ValueError("CHANGELOG.md still has an Unreleased release section")
    return {"version": version, "date": date, "tag": f"v{version}"}


def validate_publication_ref(
    metadata: dict[str, str], ref_type: str, ref_name: str
) -> None:
    if ref_type != "tag" or ref_name != metadata["tag"]:
        raise ValueError(
            f"publication requires tag {metadata['tag']}; got {ref_type} {ref_name!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tag", action="store_true")
    parser.add_argument("--ref-type", default=os.environ.get("GITHUB_REF_TYPE", ""))
    parser.add_argument("--ref-name", default=os.environ.get("GITHUB_REF_NAME", ""))
    args = parser.parse_args(argv)
    metadata = release_metadata(require_dated=args.require_tag)
    if args.require_tag:
        validate_publication_ref(metadata, args.ref_type, args.ref_name)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
