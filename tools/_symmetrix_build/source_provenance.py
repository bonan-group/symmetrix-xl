"""Content identity and VCS provenance for Symmetrix source builds."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from .command import CommandRunner


def _hash_file(
    digest: Any, root: Path, path: Path, content: bytes | None = None
) -> None:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    if content is None:
        if path.is_symlink():
            content = os.readlink(path).encode("utf-8")
        else:
            content = path.read_bytes()
    _hash_entry(digest, relative, content)


def _hash_entry(digest: Any, relative: bytes, content: bytes) -> None:
    digest.update(len(relative).to_bytes(8, "little"))
    digest.update(relative)
    digest.update(len(content).to_bytes(8, "little"))
    digest.update(content)


def _git_output(directory: Path, *arguments: str) -> bytes | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(directory), *arguments),
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _hash_submodule_identity(digest: Any, repo_root: Path, prefix: str = "") -> None:
    """Hash submodule revisions and local edits without walking clean trees."""

    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.is_file():
        return
    paths: list[Path] = []
    for line in gitmodules.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().endswith(".path"):
            candidate = Path(value.strip())
            if not candidate.is_absolute() and ".." not in candidate.parts:
                paths.append(candidate)
    for relative_path in sorted(set(paths), key=lambda item: item.as_posix()):
        path = repo_root / relative_path
        name = "/".join(item for item in (prefix, relative_path.as_posix()) if item)
        label = f"submodule:{name}".encode()
        head = _git_output(path, "rev-parse", "HEAD")
        if head is None:
            for candidate in sorted(path.rglob("*")):
                if ".git" in candidate.parts or not candidate.is_file():
                    continue
                _hash_file(digest, repo_root, candidate)
            _hash_entry(digest, label + b":unavailable", b"")
            continue
        _hash_entry(digest, label + b":head", head.strip())
        diff = _git_output(path, "diff", "--binary", "HEAD", "--")
        if diff is None:
            _hash_entry(digest, label + b":diff-unavailable", b"")
        else:
            _hash_entry(digest, label + b":diff", diff)
        untracked = _git_output(
            path, "ls-files", "--others", "--exclude-standard", "-z"
        )
        if untracked is None:
            _hash_entry(digest, label + b":untracked", b"")
            continue
        for untracked_name in sorted(item for item in untracked.split(b"\0") if item):
            file_path = path / os.fsdecode(untracked_name)
            if file_path.is_file() and not file_path.is_symlink():
                content = file_path.read_bytes()
            elif file_path.is_symlink():
                content = os.readlink(file_path).encode("utf-8")
            else:
                content = b"<missing>"
            _hash_entry(digest, label + b":untracked:" + untracked_name, content)
        _hash_submodule_identity(digest, path, name)


def _fingerprint(
    repo_root: Path,
    *,
    roots: tuple[Path, ...],
    files: list[Path],
) -> str:
    digest = hashlib.sha256()
    for source_root in roots:
        if source_root.is_dir():
            files.extend(
                path
                for path in sorted(source_root.rglob("*"))
                if (
                    path.is_file()
                    and ".git" not in path.parts
                    and "__pycache__" not in path.parts
                )
            )
    for path in sorted(set(files)):
        if path.is_file():
            _hash_file(digest, repo_root, path)
    _hash_submodule_identity(digest, repo_root)
    return digest.hexdigest()


def symmetrix_source_fingerprint(repo_root: Path) -> str:
    """Hash source inputs whose edits can change a built Symmetrix package."""
    return _fingerprint(
        repo_root,
        roots=(
            repo_root / "tools/_symmetrix_build",
            repo_root / "libsymmetrix/source",
            repo_root / "symmetrix/source",
            repo_root / "libsymmetrix/cmake",
            repo_root / "symmetrix/cmake",
        ),
        files=[
            repo_root / ".gitmodules",
            repo_root / "libsymmetrix/CMakeLists.txt",
            *sorted((repo_root / "libsymmetrix/cmake").rglob("*")),
            repo_root / "symmetrix/CMakeLists.txt",
            repo_root / "symmetrix/pyproject.toml",
            repo_root / "pair_symmetrix/install.sh",
            repo_root / "tools/mpi_gpu_aware_probe.cpp",
            *sorted((repo_root / "pair_symmetrix").glob("*.h")),
            *sorted((repo_root / "pair_symmetrix").glob("*.cpp")),
        ],
    )


def symmetrix_native_source_fingerprint(repo_root: Path) -> str:
    """Hash native inputs while excluding Python-only generators and tooling."""
    return _fingerprint(
        repo_root,
        roots=(
            repo_root / "libsymmetrix/source",
            repo_root / "symmetrix/source/cpp",
        ),
        files=[
            repo_root / ".gitmodules",
            repo_root / "libsymmetrix/CMakeLists.txt",
            repo_root / "symmetrix/CMakeLists.txt",
            *sorted((repo_root / "libsymmetrix/cmake").rglob("*")),
            *sorted((repo_root / "symmetrix/cmake").rglob("*")),
        ],
    )


def symmetrix_source_provenance(
    repo_root: Path, runner: CommandRunner
) -> dict[str, object]:
    """Return the exact content identity plus best-effort Git provenance."""

    revision = ""
    submodules = ""
    dirty: bool | None = None
    git = runner.which("git") if (repo_root / ".git").exists() else None
    if git:
        head = runner.run((git, "-C", str(repo_root), "rev-parse", "HEAD"))
        if head.returncode == 0:
            revision = head.stdout.strip()
        status = runner.run(
            (
                git,
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--ignore-submodules=all",
            )
        )
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())
        status = runner.run(
            (git, "-C", str(repo_root), "submodule", "status", "--recursive")
        )
        if status.returncode == 0:
            submodules = status.stdout.strip()
    return {
        "path": str(repo_root),
        "revision": revision,
        "dirty": dirty,
        "submodules": submodules,
        "content_sha256": symmetrix_source_fingerprint(repo_root),
        "native_content_sha256": symmetrix_native_source_fingerprint(repo_root),
    }
