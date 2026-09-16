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

The helper installs the frontend and selected backend into the target Python
environment. CUDA and HIP backends are separate architecture-qualified
extensions; see {doc}`backends`. Build and release commands are summarized in
{doc}`/reference/build_frontend`.

`uv` must be installed before this sequence. Activating `.venv` is required:
the helper intentionally installs into the Python interpreter that invoked it.

Verify the selected backend before running a calculation:

```bash
symmetrix backend list
symmetrix doctor
```
