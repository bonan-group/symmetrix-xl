# Native Architecture

The semantic native-source ownership map is maintained in
{doc}`/mace_kokkos_source_layout`. The compact MACE pipeline includes radial
stages R0/R1, message stages M0/M1, dense transforms, readouts, and device
force/stress reductions. Generated artifacts specialize eligible R0/M0/R1
contracts while graph shape and learned weights remain runtime values.

```{toctree}
:hidden:

/mace_kokkos_source_layout
```
