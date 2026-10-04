import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.setup_paseo_submodules import populate_submodules


def _git(*args, cwd=None):
    return subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip()


def _repository(path: Path, filename: str):
    path.mkdir()
    _git("init", "--initial-branch=main", cwd=path)
    _git("config", "user.name", "Symmetrix test", cwd=path)
    _git("config", "user.email", "symmetrix@example.invalid", cwd=path)
    (path / filename).write_text("test\n")
    _git("add", filename, cwd=path)
    _git("commit", "-m", "Initial commit", cwd=path)


def test_paseo_setup_uses_independent_local_submodule_clone(tmp_path):
    leaf = tmp_path / "leaf"
    dependency = tmp_path / "dependency"
    source = tmp_path / "source"
    worktree = tmp_path / "worktree"
    _repository(leaf, "leaf.txt")
    _repository(dependency, "dependency.txt")
    _git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(leaf),
        "nested/leaf",
        cwd=dependency,
    )
    _git("commit", "-m", "Add nested dependency", cwd=dependency)
    _repository(source, "source.txt")
    _git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(dependency),
        "external/dependency",
        cwd=source,
    )
    _git("commit", "-m", "Add dependency", cwd=source)
    _git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
        cwd=source,
    )
    _git("worktree", "add", "--detach", str(worktree), "HEAD", cwd=source)

    populate_submodules(source, worktree)

    checkout = worktree / "external/dependency"
    assert (checkout / ".git").is_dir()
    alternates = checkout / ".git/objects/info/alternates"
    assert not alternates.exists()
    source_objects = Path(
        _git("rev-parse", "--git-path", "objects", cwd=source / "external/dependency")
    )
    if not source_objects.is_absolute():
        source_objects = source / "external/dependency" / source_objects
    shared_objects = []
    for source_object in source_objects.rglob("*"):
        if not source_object.is_file():
            continue
        destination_object = (
            checkout / ".git/objects" / source_object.relative_to(source_objects)
        )
        if destination_object.is_file():
            source_stat = source_object.stat()
            destination_stat = destination_object.stat()
            if (
                source_stat.st_dev == destination_stat.st_dev
                and source_stat.st_ino == destination_stat.st_ino
            ):
                shared_objects.append(source_object)
    assert shared_objects
    assert (
        Path(_git("remote", "get-url", "origin", cwd=checkout)).resolve()
        == (source / "external/dependency").resolve()
    )
    assert _git("rev-parse", "HEAD", cwd=checkout) == _git(
        "rev-parse", "HEAD", cwd=source / "external/dependency"
    )
    assert (checkout / "nested/leaf/.git").is_dir()
    statuses = _git("submodule", "status", "--recursive", cwd=worktree).splitlines()
    assert len(statuses) == 2
    assert all(status.startswith(" ") for status in statuses)
