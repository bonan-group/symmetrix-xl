# Contributing

Build the documentation extras and render with warnings as errors:

```bash
uv pip install -e "./symmetrix[docs]"
make -C docs html
```

Pull requests that change the documentation build it with warnings treated as
errors. Changes merged to `main` are published to GitHub Pages at
<https://bonan-group.github.io/symmetrix-xl/> by the `Documentation` workflow.

For code, install with the maintained build frontend instead of an editable
install: `python tools/symmetrix_build.py install --backend cpu|cuda|hip` in a
fresh environment and task-specific build root, keeping CPU, CUDA, HIP, and
LAMMPS builds separate. Run focused tests first, then the applicable suite
(`pytest symmetrix/test`), then `uvx pre-commit run --all-files` on the intended
files.

Use MyST Markdown for maintained contracts and guides. Keep reproducible
qualification drivers under `benchmarks/` and raw results outside tracked
documentation. Record the source revision and measured environment with results.
Do not use a result from a retired ABI, artifact generation, or execution
variant as a current support claim; the generated
{doc}`/reference/execution_support_matrix` is the release-facing source of
truth. Changes to an execution or installation contract must update
implementation and documentation together; adding a model family follows
{doc}`adding_a_family`.

The repository-wide build, correctness, and review requirements are maintained
in `AGENTS.md` at the repository root.
