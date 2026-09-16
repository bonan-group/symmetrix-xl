# Command-Line Interface

`symmetrix backend list [--json]` reports installed descriptors and their
usability without loading an extension. `symmetrix backend show [selector]
[--json] [--probe]` resolves the automatic or explicit selection; an
incompatible accelerator selector is an error rather than a usability report.
With `--probe`, the selected extension is loaded and CPU backends report the
same OpenBLAS configuration and threading fields as `symmetrix doctor`.
`symmetrix doctor [--json] [--advisory]` prints runtime diagnostics; `--json`
provides the machine-readable variant for deployment checks.

Artifact preparation commands are exposed as `symmetrix_prepare_jit_host_artifact`
and `symmetrix_prepare_jit_device_artifact`. The converter is
`symmetrix_extract_mace --model MODEL [--output PATH]`; it accepts species,
head, compact/pair-spline format, and device-artifact preparation options.
