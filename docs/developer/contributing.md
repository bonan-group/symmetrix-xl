# Contributing

Build the documentation extras and render with warnings as errors:

```bash
uv pip install -e "./symmetrix[docs]"
make -C docs html
```

Pull requests that change the documentation build it with warnings treated as
errors. Changes merged to `main` are published to GitHub Pages at
<https://bonan-group.github.io/symmetrix-xl/> by the `Documentation` workflow.

Use MyST Markdown for stable user-facing guides. Keep current qualification
evidence under `benchmarks/` instead of duplicating its data in the user guide,
and prune reports when they are superseded. Changes to an execution or
installation contract must update implementation and documentation together.

The repository-wide build, correctness, and review requirements are maintained
in `AGENTS.md` at the repository root.
