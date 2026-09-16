import ctypes
import ctypes.util
import json
import os
from pathlib import Path

import pytest

from symmetrix.jit_operator_codegen import (
    render_jit_m0_device_module,
    render_jit_r0_device_module,
)


CONTRACTS = Path(__file__).resolve().parent / "data" / "execution_contracts"


class _HiprtcCompiler:
    def __init__(self, library):
        self.library = library
        self.library.hiprtcCreateProgram.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        )
        self.library.hiprtcCreateProgram.restype = ctypes.c_int
        self.library.hiprtcCompileProgram.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        )
        self.library.hiprtcCompileProgram.restype = ctypes.c_int
        self.library.hiprtcGetProgramLogSize.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        )
        self.library.hiprtcGetProgramLogSize.restype = ctypes.c_int
        self.library.hiprtcGetProgramLog.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
        )
        self.library.hiprtcGetProgramLog.restype = ctypes.c_int
        self.library.hiprtcGetCodeSize.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        )
        self.library.hiprtcGetCodeSize.restype = ctypes.c_int
        self.library.hiprtcDestroyProgram.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
        self.library.hiprtcDestroyProgram.restype = ctypes.c_int

    def _log(self, program):
        size = ctypes.c_size_t()
        if self.library.hiprtcGetProgramLogSize(program, ctypes.byref(size)) != 0:
            return "hipRTC could not retrieve the compiler log"
        if size.value == 0:
            return "hipRTC returned no compiler log"
        buffer = ctypes.create_string_buffer(size.value)
        if self.library.hiprtcGetProgramLog(program, buffer) != 0:
            return "hipRTC could not retrieve the compiler log"
        return buffer.value.decode("utf-8", errors="replace")

    def compile(self, source, *, target):
        program = ctypes.c_void_p()
        status = self.library.hiprtcCreateProgram(
            ctypes.byref(program),
            source.encode(),
            b"symmetrix_jit_operator.hip",
            0,
            None,
            None,
        )
        if status != 0:
            raise RuntimeError(f"hiprtcCreateProgram failed with status {status}")
        try:
            options = (ctypes.c_char_p * 3)(
                b"--std=c++17",
                b"-O3",
                f"--gpu-architecture={target}".encode(),
            )
            status = self.library.hiprtcCompileProgram(program, len(options), options)
            if status != 0:
                raise RuntimeError(self._log(program))
            code_size = ctypes.c_size_t()
            status = self.library.hiprtcGetCodeSize(program, ctypes.byref(code_size))
            if status != 0:
                raise RuntimeError(f"hiprtcGetCodeSize failed with status {status}")
            return code_size.value
        finally:
            self.library.hiprtcDestroyProgram(ctypes.byref(program))


def _hiprtc_compiler():
    candidates = []
    override = os.environ.get("SYMMETRIX_TEST_HIPRTC_LIBRARY")
    if override:
        candidates.append(override)
    discovered = ctypes.util.find_library("hiprtc")
    if discovered:
        candidates.append(discovered)
    rocm_root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    candidates.extend(
        (rocm_root / "lib/libhiprtc.so", rocm_root / "lib64/libhiprtc.so")
    )
    candidates.extend(Path("/opt/rocm").glob("core-*/lib/libhiprtc.so"))
    failures = []
    for candidate in candidates:
        try:
            return _HiprtcCompiler(ctypes.CDLL(str(candidate)))
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    pytest.skip("hipRTC is unavailable: " + "; ".join(failures))


def _contract(name):
    return json.loads((CONTRACTS / name).read_text())


@pytest.mark.device_compile
@pytest.mark.jit
@pytest.mark.parametrize("schedule", ("chunk32", "table"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_m0_operator_module_compiles_with_hiprtc(schedule, precision):
    source, _ = render_jit_m0_device_module(
        _contract("standard_m0_contract.json"),
        precision=precision,
        target="gfx1100",
        schedule=schedule,
    )

    assert _hiprtc_compiler().compile(source, target="gfx1100") > 0


@pytest.mark.device_compile
@pytest.mark.jit
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_r0_operator_module_compiles_with_hiprtc(precision):
    source, _ = render_jit_r0_device_module(
        _contract("standard_r0_contract.json"),
        precision=precision,
        target="gfx1100",
    )

    assert _hiprtc_compiler().compile(source, target="gfx1100") > 0
