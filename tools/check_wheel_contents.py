#!/usr/bin/env python3
"""Audit release wheels for private data and unintended build payloads."""

from _symmetrix_build.wheel_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
