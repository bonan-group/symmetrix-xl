"""Compatibility proxy for the historical native module import path."""

from __future__ import annotations

from . import load_backend

__symmetrix_proxy__ = True


def __getattr__(name):
    return getattr(load_backend(), name)


def __dir__():
    return sorted(vars(load_backend()))
