# Command-Line Interface

`symmetrix backend list [--json]` reports installed descriptors and their
usability without loading an extension. `symmetrix backend show [selector]
[--json] [--probe]` resolves the automatic or explicit selection; an
incompatible accelerator selector is an error rather than a usability report.
With `--probe`, the selected extension is loaded and CPU backends report the
same OpenBLAS configuration and threading fields as `symmetrix doctor`.
`symmetrix doctor [--json] [--advisory]` prints runtime diagnostics; `--json`
provides the machine-readable variant for deployment checks.

`symmetrix bench` runs the maintained SrTiO3 MACE-OMAT-0 benchmark and reports
steady-state performance in microseconds per atom. Its options select the
model, backend, precision, execution profile, system size, neighbor skin,
thread count, sampling protocol, and optional MACE-Torch correctness check.

Artifact preparation commands are exposed as `symmetrix_prepare_jit_host_artifact`
and `symmetrix_prepare_jit_device_artifact`. The converter is
`symmetrix_extract_mace --model MODEL [--output PATH]`; it accepts species,
head, compact/pair-spline format, and device-artifact preparation options and
requires the optional `symmetrix-xl[mace]` dependencies. Omitting the species
options creates the preferred universal compact export. Explicit species
selection is limited to restricted format-v2 exports or the original Symmetrix
pair-spline format (named v1 here); format-v3 nonlinear models always retain
the complete checkpoint domain.
`symmetrix_calibrate_kernel_launch` explicitly calibrates bounded GPU launch
profiles for a model and structure; normal `kernel_launch_policy="automatic"`
execution consumes a compatible calibration record but does not benchmark.
