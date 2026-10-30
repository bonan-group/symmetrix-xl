# Publications, methods, and software citation

Please cite Symmetrix-XL itself, the method implemented by the model or
correction, and the publication for the checkpoint used in a calculation. A
model package or model card may request additional dataset- or
checkpoint-specific citations; follow those instructions as well.

The lists below distinguish scientific method citations from software used for
conversion, numerical reference data, or validation. The latter packages are
not necessarily runtime dependencies of an installed Symmetrix-XL calculator.

## Symmetrix-XL

L. Fu, Z. Feng, Y. Li, H. Du, X. He, J. Wu, K. Bao, Y. Zhang, Z. Deng,
Z. Lu, and B. Zhu, "Train for Accuracy, Execute at Scale:
Architecture-Preserving Inference for Equivariant Atomistic Foundation
Models," arXiv:2610.01036 (2026). [arXiv](https://arxiv.org/abs/2610.01036)

Symmetrix-XL builds on the original
[Symmetrix](https://github.com/wcwitt/symmetrix) project. Its earliest
published results appear in D. P. Kovács *et al.*, "MACE-OFF: Short-Range
Transferable Machine Learning Force Fields for Organic Molecules," *Journal of
the American Chemical Society* **147**, 17598-17611 (2025).
[DOI](https://doi.org/10.1021/jacs.4c07099)

## MACE model families

For every MACE-family calculation, cite Symmetrix-XL and the original MACE
method:

I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner, and G. Csányi,
"MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and
Accurate Force Fields," *Advances in Neural Information Processing Systems*
**35**, 11423-11436 (2022). [DOI](https://doi.org/10.52202/068431-0830)

Add the citations below when the corresponding model family or checkpoint is
used.

### Foundation and nonlinear MACE models

- For MACE-MP and related foundation checkpoints: I. Batatia *et al.*, "A
  foundation model for atomistic materials chemistry," *The Journal of
  Chemical Physics* **163**, 184110 (2025).
  [DOI](https://doi.org/10.1063/5.0297006)
- For MACE-MH-0 or MACE-MH-1, including the nonlinear interaction and product
  blocks implemented by Symmetrix-XL: I. Batatia, C. Lin, J. Hart, E. Kasoar,
  A. M. Elena, S. W. Norwood, T. Wolf, and G. Csányi, "Cross Learning between
  Electronic Structure Theories for Unifying Molecular, Surface, and
  Inorganic Crystal Foundation Force Fields," arXiv:2510.25380 (2025).
  [DOI](https://doi.org/10.48550/arXiv.2510.25380)
- Cite the checkpoint provider's model card as well. For example, MACE-OFF
  checkpoints request the MACE-OFF paper cited in the Symmetrix-XL section
  above.

### MACELES and Latent Ewald Summation

For `MACELES`, cite both the LES method and its MACE integration:

1. B. Cheng, "Latent Ewald summation for machine learning of long-range
   interactions," *npj Computational Materials* **11**, 80 (2025).
   [DOI](https://doi.org/10.1038/s41524-025-01577-7)
2. D. Kim, X. Wang, S. Vargas, P. Zhong, D. S. King, T. J. Inizan, and
   B. Cheng, "A universal augmentation framework for long-range
   electrostatics in machine learning interatomic potentials," *Journal of
   Chemical Theory and Computation* **21**, 12709-12724 (2025).
   [DOI](https://doi.org/10.1021/acs.jctc.5c01400)

The first paper introduces LES. The second describes its integration with
short-range model families, including MACE, and the MACELES-OFF model.

### MACE-POLAR-1

For `MACE_Polar` and MACE-POLAR-1 checkpoints, cite:

I. Batatia, W. J. Baldwin, D. Kuryla, J. Hart, E. Kasoar, A. M. Elena,
H. Moore, M. J. Gawkowski, B. X. Shi, V. Kapil, P. Kourtis, I.-B. Magdău,
and G. Csányi, "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
for Molecular Chemistry," arXiv:2602.19411 (2026).
[DOI](https://doi.org/10.48550/arXiv.2602.19411)

### MACEField

For `MACEField`, cite the original MACE method above and the model- or
dataset-specific publication supplied with the checkpoint. The field-aware
reference implementation is maintained in the
[MACE-Field repository](https://github.com/mdi-group/mace-field); cite or
archive the exact software revision used for checkpoint conversion or
comparison.

## NequIP-OAM

For NequIP-OAM calculations, cite Symmetrix-XL and these NequIP publications:

1. C. W. Tan, M. L. Descoteaux, M. Kotak, G. de Miranda Nascimento,
   S. R. Kavanagh, L. Zichi, M. Wang, A. Saluja, Y. R. Hu, T. Smidt,
   A. Johansson, W. C. Witt, B. Kozinsky, and A. Musaelian,
   "High-performance training and inference for deep equivariant interatomic
   potentials," *Digital Discovery* **5**, 1558-1567 (2026).
   [DOI](https://doi.org/10.1039/D5DD00423C)
2. S. R. Kavanagh, C. W. Tan, M. Wang, M. L. Descoteaux,
   G. de Miranda Nascimento, U. Unneberg, L. Zichi, F. Libbi, N. Rivano,
   A. Glover, V. Bharadwaj, A. Johansson, W. C. Witt, A. Musaelian, and
   B. Kozinsky, "Fast and Accurate Equivariant Foundation Models for
   Atomistic Simulation," arXiv:2607.28461 (2026).
   [DOI](https://doi.org/10.48550/arXiv.2607.28461)
3. S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa,
   M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky,
   "E(3)-equivariant graph neural networks for data-efficient and accurate
   interatomic potentials," *Nature Communications* **13**, 2453 (2022).
   [DOI](https://doi.org/10.1038/s41467-022-29939-5)

The first paper describes the current NequIP software framework, the second
introduces the OAM foundation-model family, and the third is the original
NequIP method. The official NequIP repository maintains the authoritative
[citation guidance and BibTeX records](https://github.com/mir-group/nequip#references--citing).

## Dispersion corrections

Select citations according to the correction and damping actually used:

- **DFT-D2:** S. Grimme, "Semiempirical GGA-type density functional
  constructed with a long-range dispersion correction," *Journal of
  Computational Chemistry* **27**, 1787-1799 (2006).
  [DOI](https://doi.org/10.1002/jcc.20495)
- **DFT-D3 with zero damping:** S. Grimme, J. Antony, S. Ehrlich, and H. Krieg,
  "A consistent and accurate *ab initio* parametrization of density functional
  dispersion correction (DFT-D) for the 94 elements H-Pu," *The Journal of
  Chemical Physics* **132**, 154104 (2010).
  [DOI](https://doi.org/10.1063/1.3382344)
- **DFT-D3 with Becke-Johnson (`bj`) damping:** S. Grimme, S. Ehrlich, and
  L. Goerigk, "Effect of the damping function in dispersion corrected density
  functional theory," *Journal of Computational Chemistry* **32**, 1456-1465
  (2011). [DOI](https://doi.org/10.1002/jcc.21759)
- **DFT-D3M with `zerom` or `bjm` damping:** D. G. A. Smith, L. A. Burns,
  K. Patkowski, and C. D. Sherrill, "Revised Damping Parameters for the D3
  Dispersion Correction to Density Functional Theory," *The Journal of
  Physical Chemistry Letters* **7**, 2197-2203 (2016).
  [DOI](https://doi.org/10.1021/acs.jpclett.6b00780)
- **FourierD3:** cite the DFT-D3 and Becke-Johnson papers above and
  V. Valeeva, C. H. Ho, M. Geiger, F. Pellegrini, G. Csányi, E. Kucukbenli,
  and C. Ortner, "A fast summation method for the DFT-D3 dispersion
  correction," arXiv:2607.15103 (2026).
  [DOI](https://doi.org/10.48550/arXiv.2607.15103)

Symmetrix-XL implements the two-body D3 term. It does not implement the
Axilrod-Teller-Muto three-body term. `FourierD3Correction` is a modified
particle-mesh model rather than a numerically identical acceleration of
finite-cutoff `D3Correction`; see {doc}`/user/dispersion_corrections` for its
documented boundary.

## Software and numerical reference implementations

These projects contributed source code, checkpoint conversion, formulas,
parameter tables, or numerical oracles. Cite the relevant project or software
paper when that software materially supports published work, and record the
exact version or commit used for reproducibility.

| Project | Role in Symmetrix-XL | Citation or provenance |
| --- | --- | --- |
| [Symmetrix](https://github.com/wcwitt/symmetrix) | Original project and Kokkos MACE implementation | Repository plus the MACE-OFF paper above |
| [MACE](https://github.com/ACEsuit/mace) | Checkpoint conversion and upstream MACE-family parity | Repository plus the MACE method and checkpoint papers above |
| [MACE-Field](https://github.com/mdi-group/mace-field) | Field-aware checkpoint and response reference | Repository revision plus the MACE method |
| [LES](https://github.com/ChengUCB/les) | Scalar-monopole LES formulas and frozen numerical reference fixtures | Repository revision plus the LES papers above |
| [NequIP](https://github.com/mir-group/nequip) and [e3nn](https://github.com/e3nn/e3nn) | Optional OAM export and upstream parity stack | NequIP papers above; M. Geiger and T. Smidt, "e3nn: Euclidean Neural Networks," arXiv:2207.09453 (2022), [DOI](https://doi.org/10.48550/arXiv.2207.09453) |
| [torch-dftd](https://github.com/pfnet-research/torch-dftd) 0.5.3 | Source of generated D2/D3 parameter tables and offline reference fixtures; not a runtime dependency | Repository, D2/D3 papers above, and S. Takamoto *et al.*, "Towards universal neural network potential for material discovery applicable to arbitrary combinations of 45 elements," *Nature Communications* **13**, 2991 (2022), [DOI](https://doi.org/10.1038/s41467-022-30687-9), as requested by the software's [citation guidance](https://github.com/pfnet-research/torch-dftd#citation) |
| [ALCHEMI Toolkit-Ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) 0.5.0 | FourierD3 decomposition and frozen reference-oracle provenance | Repository revision plus the FourierD3 paper above |
| [graph_electrostatics](https://github.com/WillBaldwin0/graph_electrostatics) | Frozen MACE-POLAR-1 long-range operator oracle | Repository revision plus the MACE-POLAR-1 paper above |
| [SpheriCart](https://github.com/lab-cosmo/sphericart) | Bundled spherical-harmonic implementation | F. Bigi, G. Fraux, N. J. Browning, and M. Ceriotti, "Fast evaluation of spherical harmonics with sphericart," *The Journal of Chemical Physics* **159**, 064802 (2023), [DOI](https://doi.org/10.1063/5.0156307) |
| [matscipy](https://github.com/libAtoms/matscipy) | Optional host neighbor-list provider and validation reference | P. Grigorev *et al.*, "matscipy: materials science at the atomic scale with Python," *Journal of Open Source Software* **9**, 5668 (2024), [DOI](https://doi.org/10.21105/joss.05668) |

The exact LES, torch-dftd, FourierD3, and MACE-POLAR-1 oracle revisions are
recorded with their generated fixtures or developer qualification documents.
License attribution is a separate obligation: see the repository
[`THIRD_PARTY_LICENSES.md`](https://github.com/bonan-group/symmetrix-xl/blob/main/THIRD_PARTY_LICENSES.md)
and the license files in each submodule. The table above is not a replacement
for those notices or a complete inventory of ordinary build and runtime
dependencies.
