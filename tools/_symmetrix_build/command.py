"""Subprocess helpers for the standalone build frontend."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class BuildError(RuntimeError):
    """A build prerequisite or subprocess failed."""


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


class CommandRunner:
    """Resolve and run external tools without shell interpretation."""

    def which(self, command: str) -> str | None:
        return shutil.which(command)

    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        check: bool = False,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        normalized = tuple(os.fspath(value) for value in args)
        completed = subprocess.run(
            normalized,
            check=False,
            cwd=cwd,
            env=None if env is None else dict(env),
            text=True,
            capture_output=True,
        )
        result = CommandResult(
            normalized,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
        if check and result.returncode != 0:
            rendered = " ".join(normalized)
            detail = result.output.strip() or "no diagnostic output"
            raise BuildError(
                f"command failed ({result.returncode}): {rendered}\n{detail}"
            )
        return result

    def run_logged(
        self,
        args: Sequence[str | os.PathLike[str]],
        log_path: str | os.PathLike[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        append: bool = False,
    ) -> CommandResult:
        """Run a long command while mirroring combined output to a durable log."""
        normalized = tuple(os.fspath(value) for value in args)
        mode = "a" if append else "w"
        with Path(log_path).open(mode) as log:
            if append:
                log.write(f"\n$ {' '.join(normalized)}\n")
            process = subprocess.Popen(
                normalized,
                cwd=cwd,
                env=None if env is None else dict(env),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        return CommandResult(normalized, returncode, "", "")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_executable(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise BuildError(f"executable does not exist: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise BuildError(f"path is not executable: {resolved}")
    return str(resolved)


def validated_executable(path: str | os.PathLike[str]) -> str:
    """Validate an executable while preserving its final symlink basename."""

    absolute = Path(path).expanduser().absolute()
    if not absolute.is_file():
        raise BuildError(f"executable does not exist: {absolute}")
    if not os.access(absolute, os.X_OK):
        raise BuildError(f"path is not executable: {absolute}")
    return str(absolute)
