# Contributing to Symmetrix-XL

Bug reports, documentation improvements, tests, and focused implementation
changes are welcome. Open an issue before starting a large behavioral or API
change so its scope and backend impact can be agreed first.

Clone with submodules and create a development environment from the repository
root:

```bash
git clone --recursive https://github.com/bonan-group/symmetrix-xl.git
cd symmetrix-xl
uv venv
source .venv/bin/activate
uv pip install -e "./symmetrix[test,docs]"
```

Run the focused tests for your change, then the applicable suite:

```bash
pytest symmetrix/test
pytest pair_symmetrix/test
uvx pre-commit run --all-files
make -C docs html SPHINXOPTS="-W --keep-going"
```

Accelerator and LAMMPS tests require the corresponding toolchain, hardware, and
runtime. State what you tested, including backend and precision, in the pull
request. Performance changes must include reproducible before/after evidence in
`us/atom` with the workload details required by [AGENTS.md](AGENTS.md).

Keep commits cohesive and use short, imperative subjects. By contributing, you
agree that your contribution is licensed under the license governing the files
you modify: MIT for the main project and GPLv2 for `pair_symmetrix`.

See the [developer documentation](docs/developer/index.md) for architecture,
build, JIT, profiling, and documentation guidance.
