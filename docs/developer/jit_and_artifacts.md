# JIT and Artifacts

Host artifacts are model-, precision-, ABI-, generation-, and target-specific.
CUDA and HIP direct artifacts are generated at runtime with NVRTC and hipRTC;
the ahead-of-time extension still determines Kokkos and SpheriCart support.
Artifacts must be prepared before LAMMPS production runs.

The direct-execution architecture and cache invariants are described in
{doc}`/streamed_edge_execution`. The implementation in
`symmetrix/source/symmetrix/jit.py` is authoritative for compiler discovery,
cache identity, publication, and validation.

Use a task-specific cache and require direct specialization when qualifying it:

```bash
export SYMMETRIX_JIT_CACHE=/tmp/symmetrix-task-jit-cache
export SYMMETRIX_JIT_POLICY=required
symmetrix_prepare_jit_host_artifact \
    --model srtio3-mace.json --precision float64
```

Prepare device artifacts with `symmetrix_prepare_jit_device_artifact` on the
target GPU. LAMMPS consumes prepared artifacts but never invokes the compiler.
