# Release runbook

This is the operational procedure for cutting a `symmetrix-xl` release. The
machinery lives in three dispatch workflows:

| Workflow | Artifacts | Publication index |
| --- | --- | --- |
| `release-cpu-wheels.yaml` | `symmetrix-xl` sdist + manylinux x86-64-v3 wheels, CPython 3.10–3.14 | PyPI + TestPyPI |
| `release-backend-wheels.yaml` | `symmetrix-xl-cpu-avx512` add-on wheel | PyPI |
| `release-cuda-wheels.yaml` | `symmetrix-xl-cuda<MAJOR>-<TARGET>` add-on wheels per CUDA major and architecture | PyPI + TestPyPI |

All three open with the same `release-metadata` gate
(`.github/actions/release-metadata`): version agreement across
`symmetrix/pyproject.toml`, `symmetrix/source/symmetrix/__init__.py`,
`symmetrix/CMakeLists.txt` and `docs/conf.py`, plus a dated CHANGELOG heading.
Dry runs (no publish input) only require the version sources to agree;
publication additionally requires a dated heading, no `Unreleased` section,
and dispatch **from the matching tag** (`v<version>`).

## One-time repository setup

* Trusted Publisher on PyPI for `symmetrix-xl` scoped to this repository,
  `release-cpu-wheels.yaml`, and the `pypi` environment, plus the same for
  `testpypi` if the rehearsal is used.
* Trusted Publisher on each CUDA add-on project (`symmetrix-xl-cuda12-*`,
  `symmetrix-xl-cuda13-*`) scoped to `release-cuda-wheels.yaml` and its
  environment: `pypi-cuda<MAJOR>-<TARGET>` for PyPI (for example
  `pypi-cuda13-sm120`) and `testpypi-cuda<MAJOR>-<TARGET>` for the TestPyPI
  rehearsal. The publishing jobs request these environment names in their
  OIDC claims; a publisher scoped to plain `pypi` is rejected.
* GitHub environments `pypi`, `testpypi`, and the per-target CUDA names
  above, with any required approvers.
* A self-hosted CUDA runner registered with labels
  `self-hosted,linux,x64,cuda` for qualification (see below).

## Cutting a release

1. **Close the changelog.** Replace `## X.Y.Z - Unreleased` with
   `## X.Y.Z - YYYY-MM-DD`. The date must not predate the latest prior
   release. No `Unreleased` section may remain anywhere in the file.
2. **Verify the version sources agree** locally:
   `python tools/check_release_metadata.py`.
3. **Run accelerator qualification** (manual gate, before publishing):
   start the self-hosted runner, dispatch *CUDA qualification* and
   *HIP qualification* (once an AMD runner exists) on the release commit,
   confirm green, stop the runner.
4. **Tag**: `git tag -a vX.Y.Z <commit> && git push origin vX.Y.Z`.
5. **TestPyPI rehearsal** (CPU and CUDA workflows): dispatch each workflow
   **on the tag ref** with `publish_testpypi=true`. Inspect the uploaded
   wheels from the TestPyPI project page.
6. **Publish**: dispatch the three workflows on the tag ref with
   `publish_pypi=true`. Publication jobs use Trusted Publishing (OIDC); no
   tokens are stored in the repository.
7. **Open the next section**: add `## X.Y.Z+1 - Unreleased` to the changelog.

### Self-hosted qualification runner (`crucible`)

The RTX 5090 host runs a repo-level runner as the `gh-runner` user. It is
installed but **disabled and stopped by default**; it only receives jobs from
manually dispatched qualification workflows, which are the sole workflows
requesting the `cuda` label.

```bash
sudo systemctl start actions.runner.bonan-group-symmetrix-xl.crucible   # before dispatching
gh workflow run cuda-qualification.yaml --ref <branch-or-tag>
gh run watch
sudo systemctl stop actions.runner.bonan-group-symmetrix-xl.crucible    # after runs finish
```

Never point `self-hosted` labels at workflows that run fork pull requests.

## Notes and known friction

* Pushing any change under `.github/workflows/` requires a credential with
  the `workflow` scope (`gh auth refresh -h github.com -s workflow`), or ask
  a maintainer to push the branch.
* Wheel audits bind the prebuilt-operator catalog to the deterministic plan
  from `symmetrix/source/symmetrix/foundation_operator_manifest.json`
  (`python tools/symmetrix_build.py plan-artifacts --mode r0 --backend cuda
  --target sm120 ...` prints the expected artifact set). Never hardcode
  artifact counts; a contract test rejects bare integer comparisons.
* CUDA and HIP qualification steps are manifest-driven: they run from
  `symmetrix` and pass the `required_nodes` of
  `symmetrix/test/*_required_tests.json` directly to pytest.
* Known issue (vendored gpulite, targeted 0.2.1): the sphericart CUDA
  runtime-compile path resolves NVRTC by prefix-matching loaded modules
  against `libnvrtc`, which also matches `libnvrtc-builtins.so`; when the
  builtins library is mapped before the real NVRTC library, device-graph
  construction fails with `Failed to load function: nvrtcCreateProgram` (an
  `undefined symbol` naming the builtins path). Until the sphericart pin is
  bumped, work around it on CUDA by preloading a real NVRTC library, e.g.
  `LD_PRELOAD=/usr/local/cuda-13.3/lib64/libnvrtc.so.13`.
