"""Vendor discovery must select the backend's package without loading a DSO."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("toolkit", "distribution", "member"),
    [
        ("cuda12", "nvidia-cufft-cu12", "nvidia/cufft/lib/libcufft.so.11"),
        ("cuda13", "nvidia-cufft", "nvidia/cu13/lib/libcufft.so.12"),
    ],
)
def test_packaged_cufft_is_registered_without_loading(
    tmp_path, monkeypatch, toolkit, distribution, member
):
    helper = importlib.import_module("symmetrix.cufft_library")
    library = tmp_path / member
    library.parent.mkdir(parents=True)
    library.write_bytes(b"not a loadable library")
    requests = []

    def installed(name):
        requests.append(name)
        assert name == distribution
        return SimpleNamespace(
            files=[Path(member)], locate_file=lambda path: tmp_path / path
        )

    monkeypatch.setattr(helper.importlib.metadata, "distribution", installed)
    paths = []
    helper.configure_cufft_library(
        SimpleNamespace(_configure_cufft_library=paths.append), toolkit
    )
    assert requests == [distribution]
    assert paths == [str(library.resolve())]


def test_override_is_left_for_first_use_even_if_library_is_missing(monkeypatch):
    helper = importlib.import_module("symmetrix.cufft_library")
    monkeypatch.setenv("SYMMETRIX_CUFFT_LIBRARY", "/nonexistent/user-cufft.so")
    paths = []
    helper.configure_cufft_library(
        SimpleNamespace(_configure_cufft_library=paths.append), "cuda13"
    )
    assert paths == []
