# Execution Planner

The public defaults are `streamed_edges="direct"` and
`execution_profile="capacity"`, with `allow_fixed_workspace=False`. The Python
frontend owns artifact discovery and compatibility diagnostics; native code
provides contract validation, backend capability, graph-size and memory
information. Fixed-workspace candidates require explicit frontend opt-in.
The current detailed graph-time plan report applies to MH-0 MACE/MACEField.

For standard compact MH-0 models, `speed` is a capability-selected throughput
bundle rather than a synonym for retaining all intermediate state. It retains
harmonic gradients for small graphs and switches to Y-only harmonic storage at
40,960 directed edges when qualified on CPU or CUDA FP32. CUDA FP64 and HIP
retain harmonic gradients. Both variants reuse eligible MH-0 state and
recompute readout state. FP32 uses compact geometry; FP64 retains Cartesian
geometry. Capacity selection uses the same throughput candidate first and may
also select Y-only storage when the estimated device-memory budget requires it.
Debug plan identifiers remain exact even on hosts where no device-memory query
exists.

The complete direct-execution support matrix, internal candidates, and debug
pins are maintained in {doc}`/streamed_edge_execution`.

```{toctree}
:hidden:

/streamed_edge_execution
```
