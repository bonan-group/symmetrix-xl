"""Reject release wheels containing private build data or stray build assets."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath

from .command import BuildError


_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".agents",
        ".codex",
        ".codex-build",
        ".git",
        ".github",
        ".qualification-tmp",
        "__pycache__",
        "build",
        "include",
        "lib",
        "lib64",
        "test",
        "tests",
    }
)
_FORBIDDEN_FILE_RE = re.compile(
    r"(?:^|[_-])(?:agents|harness|plan|qualification)(?:[_-]|[.])", re.IGNORECASE
)
_SENSITIVE_CONTENT = (
    ("Linux user home", b"/home/", re.compile(rb"/home/[^/\x00\s]{1,128}/")),
    ("macOS user home", b"/Users/", re.compile(rb"/Users/[^/\x00\s]{1,128}/")),
    (
        "temporary wheel build",
        b"/tmp/symmetrix-",
        re.compile(rb"/tmp/symmetrix-[^/\x00\s]{1,128}/"),
    ),
    (
        "email address",
        b"@",
        re.compile(
            rb"[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,253}\."
            rb"(?:ai|cn|co|com|de|dev|edu|fr|gov|info|io|jp|me|net|org|tech|uk)",
            re.I,
        ),
    ),
    (
        "private key",
        b"PRIVATE KEY",
        re.compile(rb"-----BEGIN [A-Z ]{0,32}PRIVATE KEY-----"),
    ),
    (
        "GitHub credential",
        b"gh",
        re.compile(rb"(?:ghp|github_pat)_[A-Za-z0-9_]{20,255}"),
    ),
    (
        "PyPI credential",
        b"pypi-",
        re.compile(rb"pypi-[A-Za-z0-9_-]{20,255}"),
    ),
    ("AWS access key", b"AKIA", re.compile(rb"AKIA[0-9A-Z]{16}")),
)


def audit_wheel(path: Path) -> None:
    findings: list[str] = []
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise BuildError(f"cannot inspect wheel {path}: {error}") from error

    with archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            lowered = {part.lower() for part in member.parts}
            forbidden = sorted(lowered & _FORBIDDEN_COMPONENTS)
            forbidden_name = bool(_FORBIDDEN_FILE_RE.search(member.name))
            if forbidden:
                findings.append(
                    f"{info.filename}: forbidden archive component {forbidden[0]!r}"
                )
            if forbidden_name:
                findings.append(f"{info.filename}: forbidden release-only filename")
            if info.is_dir() or forbidden or forbidden_name:
                continue
            content = archive.read(info)
            for label, marker, pattern in _SENSITIVE_CONTENT:
                if marker in content and pattern.search(content):
                    findings.append(f"{info.filename}: contains {label}")

    if findings:
        detail = "\n".join(f"- {finding}" for finding in findings[:40])
        if len(findings) > 40:
            detail += f"\n- ... and {len(findings) - 40} more"
        raise BuildError(f"wheel content audit failed for {path.name}:\n{detail}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for wheel in args.wheel:
        try:
            audit_wheel(wheel)
        except BuildError as error:
            parser.exit(1, f"error: {error}\n")
        print(f"wheel content audit passed: {wheel.name}")
    return 0
