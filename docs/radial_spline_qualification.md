# Radial spline qualification

`benchmarks/radial_spline_convergence.py` tests whether the uniform compact
radial tables are sufficiently dense for a particular extracted model. It is
intended to catch the short-distance derivative failure that was exposed by an
unphysical AlN MPI fixture, and to prevent spline-point changes from being made
without model-specific evidence.

The driver changes only `compact_radial.num_spline_points`. For every requested
table size, it evaluates R0 and R1 for every ordered species pair on a combined
linear and logarithmic radius grid. The reference is the exact Bessel basis,
distance transform, cutoff, and extracted PyTorch radial MLP stored in the
compact model. The tested path is the actual Symmetrix-XL runtime spline. A nodal
consistency check first requires the two implementations to agree at every
spline node within `1e-9`; this distinguishes a faulty reference implementation
from interpolation error between nodes.

The JSON report records the worst value and radial-derivative errors, their
radius and output channel, separate bad-radius regions, and any nonmonotonic
convergence. The default gate applies only at radii of at least `1.5 A`; shorter
radii remain in the diagnostic report but do not reject a table intended for
physical AlN configurations. Defaults are `1e-4` for radial values and `1e-2`
for radial derivatives.

Run the direct radial qualification with:

```bash
python benchmarks/radial_spline_convergence.py compact-model.json \
  --spline-points 64,128,256,512,1024 \
  --output spline-convergence.json
```

Supplying the source checkpoint adds an end-to-end comparison against
`MACECalculator`. Use `--structure` with any ASE-readable structure file. When
it is omitted, the driver uses a 32-atom, deterministically perturbed wurtzite
AlN cell. The report records the input file path and SHA-256, or the generated
AlN parameters, together with composition, atom count, periodicity, and minimum
distance. The default end-to-end gates are `1e-3 eV` total energy and
`1e-4 eV/A` maximum force component:

```bash
python benchmarks/radial_spline_convergence.py compact-model.json \
  --checkpoint mace-checkpoint.model --head HEAD --structure structure.extxyz \
  --output spline-convergence.json
```

For MACEField, the driver reads the compact model type and supplies the field
configured by `--electric-field`. Use `--torch-device` to select the PyTorch
device. The command exits unsuccessfully when no tested spline count passes all
enabled gates; `--diagnostic-only` suppresses this failure for exploratory
sweeps.

An initial MH-0 calibration over `0.5-6.0 A` found that 128 points failed the
default physical-range gate, while 256 and 512 points passed. At 256 points the
largest physical-range value and derivative errors were `5.62e-5` and
`7.23e-3`, respectively. The same table had a derivative error of `62.3` near
`0.546 A`, directly identifying why the retired fixture with `0.640 A`
separations was not a valid MPI/PyTorch parity test. These internal radial
errors are diagnostic quantities; the optional PyTorch force gate remains the
acceptance criterion for a production checkpoint.
