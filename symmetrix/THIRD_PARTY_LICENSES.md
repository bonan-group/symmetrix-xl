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
| KokkosFFT | <https://github.com/kokkos/kokkos-fft> | MIT OR Apache-2.0 with LLVM exception |
| pybind11 | <https://github.com/pybind/pybind11> | BSD-3-Clause |
| SpheriCart | <https://github.com/lab-cosmo/sphericart> | MIT |

Symmetrix-XL's generated native D2/D3 parameter tables and immutable reference
fixtures are derived from torch-dftd 0.5.3. The following notice applies to
those derived data:

## torch-dftd 0.5.3

MIT License

Copyright (c) 2021 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Binary wheels can also contain or depend on runtime libraries supplied by the
build environment, including OpenBLAS, FFTW3, GCC runtime libraries, CUDA, or
ROCm. A wheel whose build discovered an FFT provider links that provider
dynamically; it is not bundled. Inspect a wheel's bundled libraries and
metadata for the exact release artifact.

The main project is MIT licensed. The separate `pair_symmetrix` integration is
GPLv2 licensed; see [pair_symmetrix/LICENSE](pair_symmetrix/LICENSE).
