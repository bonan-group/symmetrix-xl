"""Self-contained source distributions vendored at pinned submodule commits."""

from __future__ import annotations

import json
import re
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from .command import BuildError, CommandRunner, sha256_file

VENDORED_DEPS_SCHEMA = 1
SDIST_PROJECT_TUPLE = "symmetrix_xl"

SUBMODULE_PATHS = (
    "libsymmetrix/external/cblas-prototypes",
    "libsymmetrix/external/json",
    "libsymmetrix/external/kokkos",
    "libsymmetrix/external/kokkos-kernels",
    "libsymmetrix/external/sphericart",
    "symmetrix/external/pybind11",
)

SUPERPROJECT_ARCHIVE_PATHS = (
    "symmetrix",
    "libsymmetrix",
    "tools",
    "LICENSE",
    "README.md",
)

VENDOR_LICENSE_PATTERNS = ("LICENSE*", "LICENCE*", "COPYING*")
VENDOR_NOTICE_OVERRIDES = {
    "libsymmetrix/external/cblas-prototypes": "README.md",
}

REQUIRED_SDIST_MEMBERS = (
    "CMakeLists.txt",
    "pyproject.toml",
    "LICENSE",
    "source/symmetrix/__init__.py",
    "external/pybind11/CMakeLists.txt",
    "libsymmetrix/CMakeLists.txt",
    "libsymmetrix/external/kokkos/CMakeLists.txt",
    "libsymmetrix/external/kokkos-kernels/CMakeLists.txt",
    "libsymmetrix/external/sphericart/sphericart/CMakeLists.txt",
    "libsymmetrix/external/json/CMakeLists.txt",
)


def _git(runner: CommandRunner, repository: Path, *arguments: str) -> str:
    result = runner.run(("git", "-C", str(repository), *arguments))
    if result.returncode != 0:
        detail = result.output.strip() or "no diagnostic output"
        raise BuildError(f"git {' '.join(arguments)} failed in {repository}: {detail}")
    return result.stdout


def head_commit(repo_root: Path, runner: CommandRunner) -> str:
    return _git(runner, repo_root, "rev-parse", "HEAD").strip()


def frontend_version(repo_root: Path) -> str:
    from .python_build import _frontend_version

    return _frontend_version(repo_root / "symmetrix")


def submodule_pinned_commits(
    repo_root: Path,
    runner: CommandRunner,
    paths: tuple[str, ...] = SUBMODULE_PATHS,
) -> dict[str, str]:
    output = _git(runner, repo_root, "ls-tree", "HEAD", *paths)
    found: dict[str, str] = {}
    for line in output.splitlines():
        match = re.fullmatch(r"[0-9]+ (\w+) ([0-9a-f]{40})\t(.+)", line.strip())
        if match is None:
            continue
        kind, commit, path = match.groups()
        if kind == "commit":
            found[path] = commit
    missing = [path for path in paths if path not in found]
    if missing:
        raise BuildError(
            "not every vendored dependency is pinned as a submodule at HEAD: "
            + ", ".join(missing)
        )
    return found


def _extract_git_archive(
    runner: CommandRunner, repository: Path, treeish: str, destination: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / f".{destination.name}.tar"
    result = runner.run(
        (
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            f"--output={archive}",
            treeish,
        )
    )
    if result.returncode != 0:
        detail = result.output.strip() or "no diagnostic output"
        raise BuildError(f"git archive {treeish} failed in {repository}: {detail}")
    try:
        with tarfile.open(archive) as bundle:
            try:
                bundle.extractall(destination, filter="data")
            except TypeError:
                bundle.extractall(destination)
    finally:
        archive.unlink(missing_ok=True)


def prepare_sdist_staging(
    repo_root: Path,
    destination: Path,
    runner: CommandRunner,
    *,
    paths: tuple[str, ...] = SUBMODULE_PATHS,
    archive_paths: tuple[str, ...] = SUPERPROJECT_ARCHIVE_PATHS,
) -> dict[str, str]:
    if destination.exists() and any(destination.iterdir()):
        raise BuildError(f"sdist staging directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    pinned = submodule_pinned_commits(repo_root, runner, paths)
    superproject = destination.parent / f".{destination.name}-superproject"
    if superproject.exists():
        shutil.rmtree(superproject)
    _extract_git_archive(runner, repo_root, "HEAD", superproject)
    try:
        for entry in archive_paths:
            source = superproject / entry
            if not source.exists():
                raise BuildError(f"HEAD archive does not contain {entry}")
            target = destination / entry
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise BuildError(f"staging collision at {target}")
            source.rename(target)
    finally:
        shutil.rmtree(superproject, ignore_errors=True)
    for path, commit in pinned.items():
        _extract_git_archive(runner, repo_root / path, commit, destination / path)
    return pinned


def write_vendored_deps(
    staged_tree: Path, source_commit: str, pinned: dict[str, str]
) -> Path:
    record = {
        "schema_version": VENDORED_DEPS_SCHEMA,
        "source_commit": source_commit,
        "submodules": dict(sorted(pinned.items())),
    }
    path = staged_tree / "symmetrix" / "VENDORED_DEPS.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def verify_vendored_licenses(
    staged_tree: Path, paths: tuple[str, ...] = SUBMODULE_PATHS
) -> dict[str, str]:
    licenses: dict[str, str] = {}
    missing: list[str] = []
    for path in paths:
        root = staged_tree / path
        matches = sorted(
            match.name
            for pattern in VENDOR_LICENSE_PATTERNS
            for match in root.glob(pattern)
        )
        if matches:
            licenses[path] = matches[0]
        elif (notice := VENDOR_NOTICE_OVERRIDES.get(path)) is not None and (
            root / notice
        ).is_file():
            licenses[path] = notice
        else:
            missing.append(path)
    if missing:
        raise BuildError(
            "vendored dependencies must ship their license text or approved "
            "source notice; missing for: " + ", ".join(missing)
        )
    return licenses


def make_sdist_tree_self_contained(staged_tree: Path) -> None:
    """Move the core beside the frontend files used as the sdist root."""
    source = staged_tree / "libsymmetrix"
    destination = staged_tree / "symmetrix" / "libsymmetrix"
    if not source.is_dir():
        raise BuildError(f"staged core source directory is missing: {source}")
    if destination.exists():
        raise BuildError(f"staged core destination already exists: {destination}")
    source.rename(destination)


def verify_sdist_rebuilds_from_itself(
    sdist_path: Path,
    python: str,
    runner: CommandRunner,
    staging_root: Path,
    version: str,
) -> None:
    """Extract the produced sdist and drive its build backend far enough to
    load build configuration and metadata. This catches self-containment
    defects (missing license file, unreadable pyproject) without compiling
    anything: the backend rejects such an sdist during metadata preparation
    even for an sdist-only build."""
    root = f"{SDIST_PROJECT_TUPLE}-{version}"
    extracted = staging_root / "smoke-extracted"
    output = staging_root / "smoke-output"
    for directory in (extracted, output):
        if directory.exists():
            shutil.rmtree(directory)
    extracted.mkdir(parents=True)
    try:
        with tarfile.open(sdist_path, "r:gz") as archive:
            try:
                archive.extractall(extracted, filter="data")
            except TypeError:
                archive.extractall(extracted)
    except (tarfile.TarError, OSError) as error:
        raise BuildError(
            f"cannot extract {sdist_path.name} for the smoke check: {error}"
        ) from error
    source = extracted / root
    if not source.is_dir():
        raise BuildError(
            f"{sdist_path.name} does not contain the expected project root {root}"
        )
    print(f"[sdist] smoke check: rebuilding {root} from itself", flush=True)
    result = runner.run_logged(
        (python, "-m", "build", "--sdist", "--outdir", str(output), str(source)),
        staging_root / "sdist-smoke.log",
    )
    if result.returncode != 0:
        raise BuildError(
            f"smoke check failed: {sdist_path.name} cannot be rebuilt from "
            f"itself; see {staging_root / 'sdist-smoke.log'}"
        )


def sdist_build_invocation(
    python: str, staged_tree: Path, output_dir: Path
) -> tuple[str, ...]:
    return (
        python,
        "-m",
        "build",
        "--sdist",
        "--outdir",
        str(output_dir),
        str(staged_tree / "symmetrix"),
    )


def verify_build_module(python: str, runner: CommandRunner) -> None:
    result = runner.run((python, "-c", "import build"))
    if result.returncode != 0:
        raise BuildError(
            f"building the sdist requires the 'build' package in {python}; "
            "install it with 'pip install build'"
        )


def verify_project_license_copy(repo_root: Path) -> None:
    root_license = repo_root / "LICENSE"
    project_license = repo_root / "symmetrix" / "LICENSE"
    if not root_license.is_file() or not project_license.is_file():
        raise BuildError(
            "the repository root LICENSE and its symmetrix/LICENSE copy must "
            "both exist; the project copy keeps sdists self-contained"
        )
    if project_license.read_text() != root_license.read_text():
        raise BuildError(
            "symmetrix/LICENSE differs from the repository root LICENSE; "
            "re-copy it before building the source distribution"
        )


def verify_clean_worktree(repo_root: Path, runner: CommandRunner) -> None:
    output = _git(runner, repo_root, "status", "--porcelain")
    dirty = [line for line in output.splitlines() if line.strip()]
    if dirty:
        raise BuildError(
            "the working tree has uncommitted changes that the sdist would "
            "exclude because it archives HEAD; commit or stash them first, "
            "or pass --allow-dirty:\n" + "\n".join(dirty)
        )


def verify_sdist_contents(sdist_path: Path, version: str) -> dict[str, Any]:
    root = f"{SDIST_PROJECT_TUPLE}-{version}"
    try:
        with tarfile.open(sdist_path, "r:gz") as archive:
            names = archive.getnames()
    except (tarfile.TarError, OSError) as error:
        raise BuildError(
            f"cannot read source distribution {sdist_path}: {error}"
        ) from error
    members = set(names)
    missing = [
        f"{root}/{member}"
        for member in REQUIRED_SDIST_MEMBERS
        if f"{root}/{member}" not in members
    ]
    if missing:
        raise BuildError(
            f"source distribution {sdist_path.name} is not self-contained; "
            "missing: " + ", ".join(missing)
        )
    if not any(
        name.startswith(f"{root}/libsymmetrix/external/cblas-prototypes/")
        for name in names
    ):
        raise BuildError(
            f"source distribution {sdist_path.name} does not vendor cblas-prototypes"
        )
    if f"{root}/VENDORED_DEPS.json" not in members:
        raise BuildError(
            f"source distribution {sdist_path.name} does not record VENDORED_DEPS.json"
        )
    for name in names:
        if ".git" in PurePosixPath(name).parts:
            raise BuildError(
                f"source distribution {sdist_path.name} contains VCS metadata: {name}"
            )
    return {
        "filename": sdist_path.name,
        "members": len(names),
        "sha256": sha256_file(sdist_path),
        "size": sdist_path.stat().st_size,
    }


def run_sdist_command(
    *,
    repo_root: Path,
    runner: CommandRunner,
    output_directory: Path,
    staging_root: Path | None = None,
    dry_run: bool = False,
    allow_dirty: bool = False,
) -> int:
    from .detect import validate_source_checkout

    python = str(Path(sys.executable).absolute())
    source_commit = head_commit(repo_root, runner)
    pinned = submodule_pinned_commits(repo_root, runner)
    output_dir = output_directory.expanduser().resolve()
    staging = (
        staging_root.expanduser().resolve()
        if staging_root
        else output_dir / ".sdist-staging"
    )
    staged_tree = staging / "tree"
    invocation = sdist_build_invocation(python, staged_tree, output_dir)
    if dry_run:
        print(
            json.dumps(
                {
                    "source_commit": source_commit,
                    "staging": str(staged_tree),
                    "submodules": pinned,
                    "command": list(invocation),
                },
                indent=2,
            )
        )
        return 0
    validate_source_checkout(repo_root, runner)
    verify_project_license_copy(repo_root)
    if not allow_dirty:
        verify_clean_worktree(repo_root, runner)
    if staged_tree.exists() and any(staged_tree.iterdir()):
        raise BuildError(f"sdist staging directory is not empty: {staged_tree}")
    verify_build_module(python, runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    pinned = prepare_sdist_staging(repo_root, staged_tree, runner)
    write_vendored_deps(staged_tree, source_commit, pinned)
    licenses = verify_vendored_licenses(staged_tree)
    make_sdist_tree_self_contained(staged_tree)
    log_path = staging / "sdist-build.log"
    print(f"[sdist] {' '.join(invocation)}", flush=True)
    result = runner.run_logged(invocation, log_path)
    if result.returncode != 0:
        raise BuildError(f"sdist build failed ({result.returncode}); see {log_path}")
    produced = sorted(output_dir.glob(f"{SDIST_PROJECT_TUPLE}-*.tar.gz"))
    if len(produced) != 1:
        raise BuildError(
            f"expected exactly one source distribution in {output_dir}; "
            f"found {len(produced)}"
        )
    version = frontend_version(repo_root)
    verify_sdist_rebuilds_from_itself(produced[0], python, runner, staging, version)
    record = {
        "source_commit": source_commit,
        "submodules": pinned,
        "licenses": licenses,
        "sdist": verify_sdist_contents(produced[0], version),
    }
    (staging / "sdist-record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0
