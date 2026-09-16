# CUDA/HIP Fresh-Process A/B Validation

`validate_cuda_hip_ab.py` is the release gate for portability benchmark pairs.
It validates recorded JSON; it does not run a benchmark or infer missing
resource data from a timing summary.

Run it after baseline and candidate processes have been combined into one
qualification record:

```bash
python benchmarks/validate_cuda_hip_ab.py \
  benchmarks/.artifacts/qualification/r1-gfx1151.json \
  --output benchmarks/.artifacts/qualification/r1-gfx1151-validation.json
```

The input schema is `symmetrix.direct.cuda-hip-ab` version 1. It contains one or
more cases. Each case contains:

- `backend`: `cuda` or `hip`;
- `workload`: `ordinary_r1` or `qualified_macefield` for HIP;
- `process_order_policy`: `randomized` or `balanced_abba`;
- `execution_order`: every baseline and candidate process ID exactly once;
- `baseline` and `candidate` records.

Each side records complete device/model/toolchain provenance, the correctness
validation contract and outcome, generated path and launch evidence, compiler
resources, workspace and peak device memory, and at least three unique fresh
process runs. Every process run requires at least 20 warmups, at least 20 raw
`samples_us_per_atom`, cold compile/load time, and artifact size. The synthetic
records in `test_cuda_hip_ab_validator.py` are executable examples of the full
schema.

The validator fails closed when evidence is absent and enforces:

- unchanged correctness contract, no non-finite results, and successful
  validation on both sides;
- active native backend, exact forward/source/edge launch order, one forward
  and two reverse launches, unchanged synchronization and transfer counts;
- byte-identical explicit workspace and at most 1% peak device-memory growth;
- no local-memory, spill-load, spill-store, or stack growth;
- no register or dynamic-shared-memory growth that reduces resident blocks;
- at most 2% for the deterministic bootstrap upper 95% bound on median
  slowdown;
- unconditional rejection above 5% median or 10% p90 slowdown;
- at most 5% cold compile/load and artifact-size growth;
- the fixed HIP limits of 50 us/atom for ordinary R1 and 45 us/atom for the
  qualified MACEField case.

CUDA and HIP records must use their own same-device baselines. Compiler and
commit identities may differ between A and B; device, architecture, driver,
runtime, Kokkos, and model identities may not. Passing this validator does not
replace numerical runtime tests, profiler/resource collection, or the existing
absolute workflow gates.
