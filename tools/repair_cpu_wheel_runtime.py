#!/usr/bin/env python3
"""Restore the build's original dynamic GOMP after CPU auditwheel repair."""

from _symmetrix_build.cpu_wheel_runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
