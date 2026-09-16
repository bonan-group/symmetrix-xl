# Benchmarks

This directory contains reproducible performance drivers and concise records of
benchmark decisions. It does not serve as permanent storage for raw profiler
captures, generated kernels, model files, or machine-specific run output.

## Maintained drivers

The Python files directly under this directory are the maintained entry points.
Their names identify the workload or comparison they run. In particular:

- `factorized_operator_benchmark.py` compares the extracted factorized R1
  operator across Symmetrix and the upstream implementation.
- `execution_upstream_r1_runner.py` is the isolated upstream runner used by that
  comparison.
- `standard_mace_streamed_benchmark.py` exercises standard MACE streamed-edge
  execution.
- `streamed_edge_release_benchmark.py` runs the milestone current/develop/
  PyTorch energy, forces, and stress comparison in isolated processes.
- `ase_omat_md_backend_scale.py` and
  `ase_omat_static_graph_backend_scale.py` measure OMAT backend scaling.
- The remaining `*_benchmark.py`, profiling, generation, and validation scripts
  support the workload named by each file.

Drivers may require an existing model checkpoint, compiled extension, GPU, or
external profiler. Consult the corresponding report and command-line help for
the exact environment and invocation.

## Reports

Tracked Markdown files preserve compact methodology, summarized
results, decisions, and reproducibility context. Historical reports are
records, not current API documentation.

Every new performance record must include the atom count and exact model cutoff.
When graph expansion is active, also record the neighbor-list skin, effective
cutoff, and directed-edge count. Include the model, backend, precision, timing,
memory, hardware, software revision, and relevant compiler or profiler context.

## Output and retention

Write generated output beneath `benchmarks/.artifacts/<descriptive-run-name>/`.
The entire `.artifacts` tree is ignored by Git. Raw JSON/CSV logs, profiler
sessions, traces, generated source and binaries, downloaded models, and bulk
per-sample output must not be committed.

Keep raw evidence locally only as long as it is useful for analysis or external
archival. Promote durable conclusions into a small tracked report, including
enough aggregate values and provenance to explain the decision. A raw artifact
may be added to version control only when a focused test consumes it as a stable
fixture; test fixtures belong with that test rather than in `.artifacts`.
