import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "libsymmetrix" / "source"


def test_factorized_blas_descriptors_preserve_layout_and_stride_width(tmp_path):
    compiler = next(
        (
            path
            for name in ("c++", "g++", "clang++")
            if (path := shutil.which(name)) is not None
        ),
        None,
    )
    if compiler is None:
        pytest.skip("a C++ compiler is required for Execution BLAS descriptor coverage")

    source = tmp_path / "factorized_blas_descriptor.cpp"
    source.write_text(
        r"""
#include "factorized_blas.hpp"

constexpr auto reverse = make_execution_reverse_gemm_descriptor<float>(
    3, 5, 7, 11, 13, 17);
static_assert(reverse.transpose_a == FactorizedBlasTranspose::none);
static_assert(reverse.transpose_b == FactorizedBlasTranspose::none);
static_assert(reverse.scalar == FactorizedBlasScalar::float32);
static_assert(reverse.m == 11 && reverse.n == 5 && reverse.k == 7);
static_assert(reverse.leading_a == 13);
static_assert(reverse.leading_b == 7);
static_assert(reverse.leading_c == 17);
static_assert(reverse.stride_a == 91);
static_assert(reverse.stride_b == 35);
static_assert(reverse.stride_c == 85);
static_assert(reverse.batch_count == 3);
static_assert(reverse.alpha == 1.0 && reverse.beta == 0.0);

constexpr auto forward = make_execution_forward_gemm_descriptor<double>(
    3, 5, 7, 11, 17);
static_assert(forward.transpose_a == FactorizedBlasTranspose::none);
static_assert(forward.transpose_b == FactorizedBlasTranspose::transpose);
static_assert(forward.scalar == FactorizedBlasScalar::float64);
static_assert(forward.m == 11 && forward.n == 7 && forward.k == 5);
static_assert(forward.leading_a == 17);
static_assert(forward.leading_b == 7);
static_assert(forward.leading_c == 17);
static_assert(forward.stride_a == 85);
static_assert(forward.stride_b == 35);
static_assert(forward.stride_c == 119);
static_assert(forward.batch_count == 3);

constexpr auto wide = make_execution_reverse_gemm_descriptor<double>(
    1, 50000, 50000, 1, 50000, 50000);
static_assert(wide.stride_a == 2500000000LL);
static_assert(wide.stride_b == 2500000000LL);
static_assert(wide.stride_c == 2500000000LL);

int main() {}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            SOURCE,
            source,
            "-o",
            tmp_path / "factorized_blas_descriptor",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
