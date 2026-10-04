#!/usr/bin/env python3
"""Populate a Paseo worktree from an instantiated source checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SetupError(RuntimeError):
    """Raised when a local submodule checkout cannot be created safely."""


@dataclass(frozen=True)
class Submodule:
    path: Path
    commit: str


def _run(*args: str | os.PathLike[str], cwd: Path | None = None) -> str:
    command = tuple(os.fspath(value) for value in args)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        raise SetupError(f"command failed: {' '.join(command)}\n{detail}")
    return completed.stdout.rstrip()


def source_submodules(source_checkout: Path) -> tuple[Submodule, ...]:
    output = _run("git", "-C", source_checkout, "submodule", "status", "--recursive")
    submodules: list[Submodule] = []
    for line in output.splitlines():
        if len(line) < 42 or line[0] != " ":
            raise SetupError(
                "source checkout must have clean, initialized recursive submodules; "
                f"unexpected status: {line}"
            )
        commit = line[1:41]
        path = line[42:].rsplit(" (", 1)[0]
        if not path or any(part in {"", ".", ".."} for part in Path(path).parts):
            raise SetupError(f"invalid submodule path reported by Git: {path!r}")
        submodules.append(Submodule(Path(path), commit))
    return tuple(submodules)


def _same_path(first: str, second: Path) -> bool:
    try:
        return Path(first).expanduser().resolve() == second.resolve()
    except OSError:
        return False


def populate_submodules(source_checkout: Path, worktree: Path) -> None:
    source_checkout = source_checkout.expanduser().resolve()
    worktree = worktree.expanduser().resolve()
    if source_checkout == worktree:
        raise SetupError("source checkout and Paseo worktree must be different")
    if not (source_checkout / ".git").exists():
        raise SetupError(f"source checkout is not a Git repository: {source_checkout}")
    if not (worktree / ".git").exists():
        raise SetupError(f"Paseo worktree is not a Git worktree: {worktree}")

    populated: list[Path] = []
    for submodule in source_submodules(source_checkout):
        source = source_checkout / submodule.path
        destination = worktree / submodule.path
        if not source.is_dir():
            raise SetupError(f"source submodule is missing: {source}")

        parent_path = max(
            (path for path in populated if submodule.path.is_relative_to(path)),
            key=lambda path: len(path.parts),
            default=Path(),
        )
        parent = worktree / parent_path
        relative = submodule.path.relative_to(parent_path)
        _run("git", "-C", parent, "submodule", "init", "--", relative)

        destination_git = destination / ".git"
        if destination_git.is_dir():
            origin = _run("git", "-C", destination, "remote", "get-url", "origin")
            if not _same_path(origin, source):
                raise SetupError(
                    f"refusing to reuse submodule with unexpected origin: {destination}"
                )
        else:
            if destination.exists():
                if destination.is_symlink() or any(destination.iterdir()):
                    raise SetupError(
                        "refusing to replace a non-empty submodule directory: "
                        f"{destination}"
                    )
                destination.rmdir()
            destination.parent.mkdir(parents=True, exist_ok=True)
            _run("git", "clone", "--local", "--no-checkout", source, destination)

        _run("git", "-C", destination, "checkout", "--detach", submodule.commit)
        populated.append(submodule.path)

    status = _run("git", "-C", worktree, "submodule", "status", "--recursive")
    invalid = [line for line in status.splitlines() if line and line[0] != " "]
    if invalid:
        raise SetupError("submodule verification failed:\n" + "\n".join(invalid))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create independent submodule worktrees with local hard-linked "
            "copies of the source checkout's Git objects."
        )
    )
    parser.add_argument("source_checkout", type=Path)
    parser.add_argument("--worktree", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        populate_submodules(args.source_checkout, args.worktree)
    except (OSError, SetupError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
