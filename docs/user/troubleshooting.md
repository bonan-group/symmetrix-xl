# Troubleshooting

Use `symmetrix backend list` to identify installed backends and `symmetrix
doctor` to inspect the selected runtime. Run both commands in the exact Python
environment used for evaluation.

```bash
symmetrix backend list
symmetrix backend show
symmetrix doctor --json
```

On CPU/OpenMP, `doctor --json` must report one expected loaded OpenMP runtime
and an actual worker team. CUDA/HIP checks verify the visible device and launch
a sentinel kernel; they do not replace a model-specific numerical test.

For direct execution failures, preserve the execution-plan report, model hash,
backend information, precision, atom and directed-edge counts, cutoff, and
skin. The detailed direct-execution contract is in
{doc}`/streamed_edge_execution`.

For an unsupported or failed direct artifact, retain the error and try the
explicit compiler-free diagnostic mode:

```python
Symmetrix("srtio3-mace.json", streamed_edges="non-compiled")
```

Also record `SYMMETRIX_BACKEND`, `SYMMETRIX_JIT_CACHE`, compiler/toolkit
version, and the output of `backend list`. Generic mode is a diagnostic and
compatibility route, not a direct-performance substitute.
