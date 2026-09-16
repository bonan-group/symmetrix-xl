# Third-Party Software

Symmetrix-XL uses the following source dependencies as Git submodules. Their
license texts remain in their respective source trees and govern those
components:

| Component | Upstream | License |
| --- | --- | --- |
| cblas-prototypes | <https://github.com/wcwitt/cblas-prototypes> | Netlib `cblas.h`; see upstream notice |
| JSON for Modern C++ | <https://github.com/nlohmann/json> | MIT |
| Kokkos | <https://github.com/kokkos/kokkos> | Apache-2.0 with LLVM exception |
| Kokkos Kernels | <https://github.com/kokkos/kokkos-kernels> | Apache-2.0 with LLVM exception |
| pybind11 | <https://github.com/pybind/pybind11> | BSD-3-Clause |
| SpheriCart | <https://github.com/lab-cosmo/sphericart> | MIT |

Binary wheels can also contain or depend on runtime libraries supplied by the
build environment, including OpenBLAS, GCC runtime libraries, CUDA, or ROCm.
Inspect a wheel's bundled libraries and metadata for the exact release artifact.

The main project is MIT licensed. The separate `pair_symmetrix` integration is
GPLv2 licensed; see [pair_symmetrix/LICENSE](pair_symmetrix/LICENSE).
