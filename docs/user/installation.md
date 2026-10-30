# Installation

The source-tree package guide remains at `symmetrix/README.md`, and the
top-level `README.md` remains the project overview. Neither is relocated by
the Sphinx documentation system. This guide is the maintained installation
workflow; the retained README files contain source-package detail.

For a normal development or evaluation installation, create a virtual
environment and use the build helper:

```bash
uv venv
source .venv/bin/activate
python tools/symmetrix_build.py install --backend cpu --cpu-target native
```

## CPU BLAS selection

The CPU build looks for an OpenBLAS installation under `/usr/local`, `/usr`,
`/opt/OpenBLAS`, and `/opt/openblas`, and pins a shared-library directory in the
extension's runtime search path when the installation it finds is
OpenMP-threaded. A pthread-threaded OpenBLAS serialises inside this project's
OpenMP regions, which costs roughly 8-13% on the direct CPU path depending on
the library build (measured 34.1 -> 29.5 us/atom for a distro OpenMP variant,
and 25.9 us/atom for a locally built OpenBLAS with native kernels), so the
OpenMP variant is preferred automatically. Debian-family prefixes that ship both
variants under `lib/<triplet>/openblas-{openmp,pthread}` resolve to the OpenMP
one.

Select a prefix explicitly when OpenBLAS lives elsewhere:

```bash
python tools/symmetrix_build.py install --backend cpu --cpu-target native \
    --blas openblas --openblas-root /path/to/openblas
```

The prefix must contain a shared `libopenblas.so` and `cblas.h`. Verify the
deployment with `symmetrix doctor --json`; it should report OpenBLAS threading
as `openmp`, a compatible single OpenMP runtime, and the resolved
`threading_library_path`. A pthread-only (or otherwise non-OpenMP) installation
is left on the system default instead of being pinned, so linking changes only
when an OpenMP build is available.

A pinned prefix is recorded in the build manifest and its library directory in
the extension's `RUNPATH`. Rebuilding from that manifest reuses the recorded
prefix and resolves its variant on the rebuilding host, so a manifest is not a
cross-machine pin. For a wheel meant for redistribution, pass an explicit
prefix (or use the container build flow) so the BLAS that gets vendored is
intentional rather than whatever the build host happened to have under `/usr`.

To force the pthread variant when both are installed — for example to keep the
linkage identical to an existing deployment — point the explicit root at a
prefix whose `lib` directory resolves to that variant. Setting the system-wide
`libopenblas.so.0` alternative is *not* enough: the variant search prefers a
directory named `openmp` inside the prefix, so the alternative link is not
consulted once an OpenMP variant is installed. The variant directories shipped
by Debian-family packages carry the library but not the headers, so an explicit
pthread pin needs a small prefix of symlinks:

```bash
mkdir -p /tmp/openblas-pthread/{lib,include}
ln -s /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so /tmp/openblas-pthread/lib/
ln -s /usr/include/x86_64-linux-gnu/openblas-pthread/cblas.h /tmp/openblas-pthread/include/
python tools/symmetrix_build.py install --backend cpu --cpu-target native \
    --blas openblas --openblas-root /tmp/openblas-pthread
```

### Build OpenBLAS for the deployment host

Distribution packages are compiled with `DYNAMIC_ARCH` for a generically
labelled target and a high thread ceiling, so they are slower than a library
built for the CPU that actually runs the calculation. On the direct CPU path
(OMAT-medium, FP32, `direct`/capacity execution, 320-atom SrTiO3 supercell,
32,896 directed edges, 8 physical cores pinned with `taskset -c 0-7`, one
workstation) that ordering is:

| BLAS | us/atom |
| --- | --- |
| distribution OpenBLAS, pthread | 34.1 |
| distribution OpenBLAS, OpenMP (auto-selected) | 29.5 |
| OpenBLAS built for the host, OpenMP | 25.9 |

Absolute values are machine-specific; the ranking — host-built OpenMP, then
distribution OpenMP, then pthread — is what to expect.

```bash
curl -LO https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.30.tar.gz
tar xf v0.3.30.tar.gz && cd OpenBLAS-0.3.30
make -j"$(nproc)" USE_OPENMP=1 DYNAMIC_ARCH=0 TARGET=ZEN NO_LAPACKE=1 NO_FORTRAN=1 \
    NUM_THREADS=32
make install PREFIX=/opt/openblas-openmp
python tools/symmetrix_build.py install --backend cpu --cpu-target native \
    --blas openblas --openblas-root /opt/openblas-openmp
```

`make -j"$(nproc)" TARGET=... USE_OPENMP=1` takes about a minute on a
workstation. Run it in a fresh tree, or delete `Makefile.conf` before
reconfiguring: OpenBLAS keeps its previous configuration file, so rebuilding in
a tree that already produced a library ignores these flags and can silently give
you a pthread build again. Choose the target for the deployment host (`ZEN`,
`HASWELL`, `SKYLAKEX`, `ARMV8`, `CORTEXA72`, ...; `TargetList.txt` in the source
tree lists the ones that release supports), and keep one prefix per
architecture: a
single-target build is not portable to a different CPU. Set `DYNAMIC_ARCH=1`
instead when one prefix must serve heterogeneous hosts — that is what the
distribution package does, with the corresponding loss above. `NUM_THREADS` is
OpenBLAS's compile-time thread ceiling; the host workers call it with nested
OpenMP teams disabled, so it mainly bounds standalone use.

The host integration requires one OpenMP runtime and no nested teams:
`symmetrix doctor` reports an error when OpenBLAS and the host workers would
nest teams (set `OMP_MAX_ACTIVE_LEVELS=1`). Its `runtime.host_blas` block shows
what was loaded, including the library's own build string in `openblas_config`
(`USE_OPENMP`, target, `MAX_THREADS`).

The helper installs the frontend and selected backend into the target Python
environment. CUDA and HIP backends are separate architecture-qualified
extensions; see {doc}`backends`. Build and release commands are summarized in
{doc}`/reference/build_frontend`.

Source builds default to `--prebuilt-operator-artifacts r0` and
`--prebuilt-oam-artifacts off`. The CPU and CUDA release-wheel workflows request
`--prebuilt-oam-artifacts known`, which installs FP32/FP64 artifacts for the
exact OAM-S/M/L/XL generated architectures; repaired-wheel inclusion remains a
release gate. Catalog presence proves build availability, not current runtime
qualification. Use the same option in a source build to avoid compilation on
first use of those layouts. A custom OAM layout still compiles into the
per-user cache when a matching compiler is available.
For MACE, use `off` for a minimal compiler-dependent build or `full` to opt into
all 32 verified foundation R1/R0/M0/M1 artifacts. Full is not currently a
publication default because no cross-backend size, build-time, and first-call
release budgets have been accepted.

`uv` must be installed before this sequence. Activating `.venv` is required:
the helper intentionally installs into the Python interpreter that invoked it.

Verify the selected backend before running a calculation:

```bash
symmetrix backend list
symmetrix doctor
```
