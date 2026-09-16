# Models and Properties

- **MACE** supports energy, per-atom energies, forces, and stress.
- **MACEField** additionally supports polarization and qualified analytical
  response properties.
- **MACE-MH-0** uses the ordinary direct execution planner.
- **MACE-MH-1** supports generated direct execution for admitted nonlinear
  contracts. Its matching graph-time execution-plan resolver is still pending.

Compact Symmetrix JSON is the preferred interchange format. Loading an
ordinary MACE checkpoint can require the optional `mace-torch` dependency.
MACEField checkpoints should be exported to compact JSON because the
field-aware upstream modules are not yet present in upstream MACE.

Convert a local ordinary MACE checkpoint after installing the optional
converter dependency:

```bash
uv pip install mace-torch
symmetrix_extract_mace \
    --model mace-omat-0-medium.model \
    --chemical-symbols Sr Ti O \
    --output srtio3-mace.json
```

The command retains all compatible prediction heads. It does not download a
model; obtain a checkpoint from its model provider, then use the resulting JSON
in the calculator. For a GPU LAMMPS run, add
`--prepare-jit-device-artifact` during conversion on the deployment GPU.
`mace-torch` and its PyTorch dependencies are optional and can be a large
download; JSON-only evaluation does not require them.

One-interaction MACE models using the invariant correlation-2 nonlinear
readout are supported with canonical equal-channel hidden irreps. This includes
both the `L_max=1` single-v2 layout and the earlier `L_max=2` single-layer
layout. Their compact JSON contains only the M0 and R0 execution contracts.

Multi-head JSON models retain every head. Select one with `head=` and inspect
`calculator.available_heads`. `SymmetrixEnsemble` provides same-process,
sequential committee evaluation; it returns mean values, member values with
the `_comm` suffix, and population variance with the `_var` suffix.

An ordinary ASE calculation can request the standard properties directly:

```python
atoms.calc = Symmetrix("srtio3-mace.json")  # FP32 by default
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

Use `Symmetrix("srtio3-mace.json", dtype="float64")` when higher numerical
precision is required.

See {doc}`/reference/python_api` for the public calculator and ensemble API.
