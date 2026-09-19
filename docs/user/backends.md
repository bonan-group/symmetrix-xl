# Backends and Diagnostics

The `symmetrix-xl` distribution includes CPU/OpenMP. Detect the local supported
backend first, then install an accelerator extension whose architecture exactly
matches the deployment GPU:

```bash
python tools/symmetrix_build.py detect --backend auto
python tools/symmetrix_build.py install --backend cuda --arch smNN \
    --cuda-root /path/to/cuda
symmetrix backend list
symmetrix doctor --json
```

Replace `smNN` and `/path/to/cuda` with the target architecture and toolkit on
the deployment system. Build one backend per GPU architecture. NVRTC recompiles direct artifacts but
cannot replace the ahead-of-time Kokkos and SpheriCart code in an extension.
`symmetrix doctor` is a fresh-process runtime smoke test: it checks OpenMP
runtime/worker participation on CPU and device selection plus a sentinel kernel
on CUDA/HIP. Use the deployment target's compiler, runtime, and scheduler
documentation when preparing a cluster build.

`symmetrix backend list` marks a usable accelerator or CPU fallback with `*`.
Use `SYMMETRIX_BACKEND=cuda13-sm80` before Python starts to select a particular
compatible installed backend. `symmetrix backend show [selector]` resolves the
automatic or explicit selection; it reports an error for an incompatible GPU.
Add `--probe` to load the selected extension and print its runtime OpenBLAS
configuration, including whether the build is OpenMP-enabled and its active
thread count, OpenMP runtime compatibility, and maximum active levels, just like
`symmetrix doctor`.

To install a published backend wheel from the command line, use:

```bash
symmetrix backend install --arch cuda13-sm120
```

The default is `--arch auto`. Automatic installation requires one visible GPU
architecture and infers the CUDA or ROCm major version. Use an explicit selector
when multiple GPUs are visible or the deployment host has no device. The command
uses `uv` when it is available, otherwise `python -m pip`, and prints the full
resolution details, including the detected architecture, toolkit version and
source, selected package, and Python environment, before running it.
CUDA 13 hosts automatically fall back to a published CUDA 12 wheel for the
same architecture when no CUDA 13 wheel is available; explicit selectors do not
fall back.
On CPU-only hosts, automatic mode explains that the bundled CPU backend needs no
download. If no pre-compiled wheel resolves, it reports the attempted selectors
and a source-build command.

An OpenMP-enabled OpenBLAS build using the same OpenMP runtime as Kokkos is
preferred for CPU Kokkos deployments. Symmetrix-XL selects compatible builds
automatically for host GEMMs and prints a runtime warning when the loaded
OpenBLAS build uses pthreads instead.
