---
title: "Streamed-Edge Execution for Memory-Efficient MACE Inference"
subtitle: "Symmetrix milestone technical report"
author: "Symmetrix contributors"
date: "22 August 2026"
lang: en
documentclass: article
fontsize: 10pt
papersize: a4
geometry:
  - margin=25mm
linestretch: 1.05
colorlinks: true
numbersections: true
abstract: |
  Equivariant MACE inference is often limited by edge-proportional tensor
  products and their reverse-mode intermediates rather than by model
  parameters. This report describes the streamed-edge execution model in
  Symmetrix, which reassociates receiver aggregation, recomputes bounded local
  state during reverse, and specializes fixed sparse tensor-product programs
  for CPU, CUDA, and HIP execution. It also compares MACE-Torch's per-edge
  radial basis and neural-network evaluation with Symmetrix's active-pair cubic
  spline lowering. The defining practical result is not only faster inference,
  but a substantial increase in the system size that fits on a fixed device.
  On a 32,607 MiB RTX 5090, low-memory execution increased the largest
  demonstrated FP32 system from 219,488 to 389,344 atoms for OMAT-0 and from
  202,612 to 340,736 atoms for MACEField. Fresh OMAT-0 boundary campaigns
  reached 13,500 atoms for
  origin/develop, 19,652 for PyTorch/cuEquivariance, and 4,000 for
  PyTorch/e3nn on the same GPU. On the qualified OMAT-0 workloads, direct
  low-memory execution is also 18.13x faster on one CPU core and 15.05x faster
  on CUDA than the original materialized Symmetrix evaluator. Relative to the
  original PyTorch/e3nn implementation, the measured speedups are 47.96x on
  one CPU core and 24.70x on CUDA; the CUDA path is 7.75x faster than the
  cuEquivariance-accelerated PyTorch reference at 4,000 atoms. Energy, force,
  and stress remain within precision-appropriate agreement tolerances.
---

<!--
PDF build:
pandoc docs/streamed_edge_milestone_report.md --standalone --number-sections \
  --from=markdown+tex_math_dollars --to=typst \
  --output=docs/.streamed_edge_milestone_report.typ
typst compile docs/.streamed_edge_milestone_report.typ \
  streamed_edge_milestone_report.pdf
rm docs/.streamed_edge_milestone_report.typ
-->

**Keywords:** MACE; equivariant neural networks; streamed edges; memory
scaling; runtime compilation; molecular dynamics.

# Introduction

MACE is a model name, not an acronym with an official expansion. MACE [1] is a
higher-order equivariant message-passing interatomic potential. It represents
each local atomic environment as a directed radius graph, applies
$O(3)$-equivariant geometric convolutions, forms higher-correlation polynomial
features through a symmetric product basis, and maps invariant scalar channels
to atomic energies.

Symmetrix changes how that operator is scheduled and stored, not the learned
MACE model. In particular, streamed-edge execution shortens the lifetime of
edge-proportional intermediates, specializes fixed tensor-product programs,
and uses explicit reverse-mode kernels while retaining the same graph,
equivariance, aggregation, product-basis, and energy-derivative semantics.

This addresses capacity as well as speed. Materializing graph-sized forward
and reverse intermediates can make an otherwise valid MACE calculation
impossible when device memory is exhausted. Streamed-edge execution changes
that feasibility boundary: it replaces long-lived edge tensors with bounded
scratch and recomputation, allowing substantially larger atomistic systems on
the same hardware while retaining competitive throughput.

# MACE architecture primer

## Atomic graph and equivariant features

Let the atomic structure contain positions $\mathbf x_i$ and species $z_i$.
Atoms are graph nodes. In the exact model graph, a directed edge $e=(j\to i)$
connects source atom $s(e)=j$ to receiver atom $r(e)=i$ when the pair lies
inside the model cutoff. A simulation may retain a larger skin-expanded
candidate graph;
the prepared direct path masks candidates outside the model cutoff to zero
until the neighbor list is rebuilt. For an active model edge, define

$$
\mathbf r_e=\mathbf x_j-\mathbf x_i,
\qquad
d_e=\lVert\mathbf r_e\rVert,
\qquad
\hat{\mathbf r}_e=\mathbf r_e/d_e.
$$

The reverse edge, when present, is a distinct graph entry. Because the model
uses relative displacements, translations leave its geometric inputs
unchanged.

At interaction $t$, node $i$ carries features organized as a direct sum of
$O(3)$ irreducible representations,

$$
h_i^{(t)}
=
\bigoplus_{\ell,p}h_i^{(t,\ell,p)},
\qquad
h_i^{(t,\ell,p)}\in
\mathbb R^{C_{\ell p}\times(2\ell+1)},
$$

where $\ell$ is angular degree, $p\in\{-1,+1\}$ is parity, and
$C_{\ell p}$ is the learned channel multiplicity. Under $Q\in O(3)$, features
transform by $D^{(\ell,p)}(Q)$. In particular, if $Q=IR$ with
$R\in SO(3)$ and $I$ inversion, then
$D^{(\ell,p)}(Q)=pD^{(\ell)}(R)$; the channel index is unchanged. Only
$(\ell,p)=(0,+1)$ channels are invariant scalars. The $(0,-1)$ representation
is a pseudoscalar, and higher degrees carry equivariant directional
information. Together with translation-invariant relative geometry, this gives
the complete model its $E(3)$, translations combined with $O(3)$, energy
invariance and force equivariance. The
initial state $h_i^{(0)}$ is a learned embedding of species $z_i$.

## One equivariant interaction

Each edge supplies invariant radial features of $d_e$ and real spherical
harmonics $Y_{\ell m}(\hat{\mathbf r}_e)$. A learned equivariant `linear_up`
first maps the source state into the representation consumed by the
interaction. Let $x_j^{(t)}=L_{\mathrm{up},t}h_j^{(t)}$. In schematic
component notation, one allowed tensor-product path $\pi$ contributes

$$
u_{e,\pi c m_o}^{(t)}
=
w_{e,\pi c}^{(t)}
\sum_{m_i,m_y}
C^{\pi}_{m_i m_y m_o}
x_{s(e),c m_i}^{(t)}
Y_{\ell_y m_y}(\hat{\mathbf r}_e),
$$

where $w_{e,\pi c}^{(t)}$ is a learned radial function, $C^{\pi}$ contains the
Clebsch-Gordan coefficient and path normalization, $c$ is a channel, and
$\pi$ identifies compatible input, spherical-harmonic, output, and parity
irreducible representations. The qualified ordinary direct contracts use
channelwise `uvu` paths, so the input and output channel are tied within each
path; the equation is deliberately schematic and does not imply general dense
channel mixing on every edge.

Incoming edge contributions are summed at the receiver and followed by a
learned equivariant interaction linear,

$$
q_i^{(t)}
=
\sum_{e:\,r(e)=i}u_e^{(t)},
\qquad
a_i^{(t)}=L_{\mathrm{int},t}q_i^{(t)}.
$$

This is the edge-conditioned geometric convolution. The receiver sum is
permutation invariant, while the Clebsch-Gordan coupling preserves the
prescribed $O(3)$ transformation law.

## Higher-correlation product basis and update

MACE obtains higher-body information without enumerating neighbor tuples.
Instead it forms products of the receiver-centered equivariant density
$a_i^{(t)}$. A schematic order-$\nu$ product-basis component is

$$
B_{i,\eta,c,LM}^{(t,\nu)}
=
\sum_{\alpha_1,\ldots,\alpha_\nu}
\mathcal U_{\eta,LM;\alpha_1\ldots\alpha_\nu}^{(t,\nu)}
\prod_{\xi=1}^{\nu}a_{i,c,\alpha_\xi}^{(t)},
$$

where each $\alpha_\xi$ denotes an input irrep/angular component,
$\mathcal U$ is a fixed sparse generalized Clebsch-Gordan coupling, $LM$
identifies the output irrep component, and $\eta$ distinguishes independent
valid couplings. The maximum polynomial degree $\nu_{\max}$ is the correlation
order; it is not an interaction-layer index. A species-conditioned learned
contraction combines these basis values without discarding their output irrep,
and the node update is schematically

$$
m_{i,c,LM}^{(t)}
=
\sum_{\nu=1}^{\nu_{\max}}
\sum_{\eta}W_{z_i,t,\nu\eta c}
B_{i,\eta,c,LM}^{(t,\nu)},
\qquad
h_i^{(t+1)}
=
L_{\mathrm{prod},t}m_i^{(t)}
+S_{z_i,t}\!\left(h_i^{(t)}\right).
$$

$L_{\mathrm{prod},t}$ is the learned product linear and $S_{z_i,t}$ is the
optional species-conditioned residual or self connection. The implementation
evaluates the sparse products as a directed acyclic graph (DAG); retaining or
recomputing that DAG is therefore a memory-policy decision, not a change to the
MACE operator. Increasing $\nu_{\max}$ directly raises the correlation order
within one local environment. Its body-order interpretation also depends on
the input density and convention, while stacking interactions compounds the
effective many-body dependence and enlarges the receptive field.

## Readouts, energy, forces, and stress

Each interaction state contributes an atomic energy through its invariant
$(\ell,p)=(0,+1)$ channels. Let $P_0$ select those channels and let $T$ be the
number of interactions. A schematic site energy that includes the terms
supported by the evaluator is

$$
\varepsilon_i
=
E_{\mathrm{ref}}(z_i)
+\varepsilon_{\mathrm{pair},i}
+\sum_{t=1}^{T}\rho_t\!\left(P_0h_i^{(t)}\right),
\qquad
E=\sum_i\varepsilon_i.
$$

$E_{\mathrm{ref}}$ is the isolated-atom reference contribution and
$\varepsilon_{\mathrm{pair},i}$ denotes an optional explicit pair term such as
the Ziegler-Biersack-Littmark (ZBL) repulsion. For the qualified two-interaction
family, the first learned readout is linear in the invariant scalar channels of
$h^{(1)}$ and the final readout is nonlinear in those of $h^{(2)}$.
Position-independent reference terms contribute to energy but not forces.

Forces and stress are derivatives of this same scalar energy,

$$
\mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i},
\qquad
\sigma_{ab}=\frac{1}{V}\frac{\partial E}{\partial\epsilon_{ab}},
$$

where $\epsilon$ is the symmetric infinitesimal strain used by MACE and $V$ is
cell volume. Symmetrix represents coordinate reverse by one directed
contribution $\mathbf f_e$ per edge, with
$\mathbf f_e=-\partial E/\partial\mathbf r_e$. Its atom reduction and
unsymmetrized virial are

$$
\mathbf F_k
=
\sum_{e:\,s(e)=k}\mathbf f_e
-\sum_{e:\,r(e)=k}\mathbf f_e,
\qquad
\widetilde\sigma_{ab}
=
-\frac{1}{V}\sum_e f_{e,a}r_{e,b}.
$$

The derivative with respect to symmetric $\epsilon$ is
$\sigma=(\widetilde\sigma+\widetilde\sigma^T)/2$. For a rotationally invariant
energy the virial is symmetric apart from numerical roundoff. The
implementation reduces the full $3\times3$ virial on the device and only then
converts it to Atomic Simulation Environment (ASE) [8] Voigt order.

# Symmetrix stage map and terminology

The qualified ordinary models contain two interactions. Symmetrix gives their
fused execution boundaries short uppercase implementation names; they need not
be one-to-one stored forms of the conceptual tensors above. Suffix `0` or `1`
identifies the interaction and is not a derivative order. In particular,
`Phi1` represents $q^{(1)}$, `A1` represents $a^{(1)}$, and `M1` represents
$m^{(1)}$ at the second interaction. The diagram shows the ordinary fused
implementation; MACEField splits the first-update state, field transform, and
second-interaction `linear_up` as described below.

```text
species -> H0                         geometry -> Y
              (R0, Y, H0) -> A0 -> A0 scale -> M0 -> H1 -> readout 1
              (R1, Y, H1) -> Phi1r/Phi1 -> A1 -> A1 scale -> M1 -> H2
                                                               -> readout 2
reference energy + optional pair term + readouts -> E -> forces/stress
```

- `H0`: species embedding after the first interaction's source `linear_up`;
  compact extraction usually folds it into first-interaction coefficients.
- `Y`: real spherical harmonics of each directed edge direction; `Y_grad` is
  their Cartesian derivative.
- `R0`: first-interaction radial edge functions. The standard compact module
  also folds in the source species embedding and applicable linear factors.
- `A0`: first-interaction projection/selector of the implicit receiver sum
  `Phi0`, which combines `R0`, `Y`, and the source embedding. Optional
  environment-density scaling follows.
- `M0`: first symmetric-contraction/product-basis polynomial.
- `H1`: native buffer supplying the second interaction. Ordinary extraction
  normally fuses the first product linear with the second interaction's
  `linear_up`. MACEField instead retains the architectural post-product state,
  applies the field transform, and then applies `linear_up`. The first readout
  remains at the architectural first-update boundary.
- `R1`: second-interaction radial edge functions indexed by coupling path and
  channel. This name denotes radial values only.
- `Phi1r`, `Phi1`: raw and Clebsch-Gordan-coupled second-interaction messages
  after receiver reduction. Both are node-sized, not complete edge-message
  arrays.
- `A1`: node tensor $A1=L_{A1}\,Phi1$, where $L_{A1}$ is the learned
  equivariant node-level projection; optional environment-density scaling
  follows.
- `M1`: second symmetric product-basis contraction. Its polynomial DAG may be
  retained or reconstructed in bounded scratch.
- `H2`: final scalar node state combining the `M1` product-linear output with
  the species-conditioned scalar residual from `H1`.
- Readouts: linear scalar-`H1` and nonlinear `H2` energy contributions, summed
  with reference and optional pair energies.

Reverse-mode adjoints use a star or an `_adj` suffix. Radial derivatives such
as `R1_deriv` are ordinary derivatives with respect to distance and must not be
confused with adjoints.

## Radial R1 versus Execution R1

The codebase also uses **Execution R1** as the name of a generated program and
versioned runtime contract. This is broader than the radial array `R1`:
Execution R1 consumes the second-interaction radial representation, `H1`, and
`Y`, produces `Phi1`, and implements the corresponding source-feature and
coordinate reverse. It does **not** include the subsequent node-level `A1`,
`M1`, or `H2` stages. The report uses `R1` alone only for radial values and
uses **Execution R1** for the generated program or artifact.

MACEField [9] uses the same ordinary second-interaction contract. Its forward
order is first product/residual state, field transform, `linear_up`, then
Execution R1. Reverse follows the exact transpose order: Execution R1 reverse,
`linear_up` transpose, then field-transform transpose. Compatible MH-1 models
instead use separate pair-conditioned prefix/density programs and a different
generated contract.

## Scope and public execution modes

This milestone turns Symmetrix's MACE evaluator [1] from a retained edge-tensor
implementation into a family of explicit execution algorithms. The public
names describe behavior rather than project history. Runtime compilation (RTC)
means generating and compiling a model-contract-specific program after the
main library has been built.

| Mode | Meaning |
|---|---|
| `materialized` | Retain the broad edge intermediates used by the original native evaluator. |
| `generic` | Stream both interactions through table-driven Kokkos code without runtime compilation. |
| `direct` | Use a model-contract-specialized Execution R1 program compiled at runtime, with prepared graph/device execution and separate node-level `A1`. |
| `receiver_factorized` | Use the paper-aligned receiver-factorized host RTC algorithm; currently an experimental FP32 Python/Serial/OpenMP selector. |

`low_memory` is an orthogonal execution-policy request for eligible `direct`
execution, not another streamed-edge mode. `low_memory=False` activates no
capacity policy. `low_memory=True` retains the normal speed path when its
model-aware graph estimate fits current device memory; otherwise it selects
bounded M1/readout recomputation, forward/adjoint buffer reuse, Y-only
harmonics, and compact geometry where precision permits. Historical “direct
retained” labels below mean the contemporaneous `low_memory=False` baseline.

`all_interactions`, `factorized`, and `direct_streamed` are compatibility
aliases for `generic`, `direct`, and `direct`. direct execution is the name of the paper
and external reference implementation from which the receiver-factorization
idea is derived [3]. It is not the name of a Symmetrix mode, cache, schema, or
internal code path. The generic and direct implementations use Kokkos [2] for
portable execution and device-resident data ownership.

Below, just-in-time (JIT) and runtime compilation (RTC) refer to specialization
after the main library build. CUDA uses NVIDIA Runtime Compilation (NVRTC), and
HIP uses hipRTC. An application binary interface (ABI) defines the callable
contract of a generated artifact. Basic Linear Algebra Subprograms (BLAS),
including general matrix multiplication (GEMM), implement suitable dense node
operations. A streaming multiprocessor (SM) is a CUDA hardware execution unit;
FP32 and FP64 denote 32- and 64-bit floating-point precision, and VRAM denotes
device memory. GFLOP below counts billions of floating-point operations per
evaluation, not a rate.

The repository uses **ordinary MACE** for the admitted two-interaction
architecture described above. Historical internal records may call this
`MH-0`; `MH-1` denotes a distinct compatible family with pair-conditioned
prefix and density networks. `OMAT-0` is a foundation-model checkpoint/catalog
label rather than an execution stage. None of these labels is expanded into an
invented phrase.

The report compares four distinct deployments:

1. The `streamed-edge` release candidate, primarily `direct` with
   `low_memory=True`.
2. The original Symmetrix `origin/develop` materialized Kokkos evaluator.
3. The original PyTorch [4] MACE/e3nn [5] implementation.
4. The same PyTorch MACE model converted at calculator construction to use
   cuEquivariance CUDA operators [6].

The first two deployments share Symmetrix model extraction and native
numerical layouts.
The two PyTorch rows supply the source-framework operator and autograd
reference. The PyTorch/e3nn evaluation remains the numerical oracle and
original implementation; cuEquivariance represents an optimized framework
deployment of the same
checkpoint. The benchmark results are in
`benchmarks/streamed_edge_milestone_20260822.md`.

The framework rows use the clean editable MACE source revision
`136e4ef040d7c51a5b051a7a609ffb1f29307478`, Torch `2.13.0+cu130` with CUDA
runtime 13.0 and cuDNN 9.2, and the explicitly selected checkpoint head
`default`. Every CUDA row pins Torch intra-op/inter-op and host OpenMP/BLAS
execution to one thread, so CPU graph and result work do not vary between e3nn
and cuEquivariance. The workload uses the OMAT-0 medium checkpoint from the
MACE foundation-model family [7].

# Streaming identity and relationship to direct execution

## Which tensors are avoidable

direct execution [3] separates a mathematical lifetime result from one particular kernel
schedule. Let $\varphi_{eq}$ be a compact invariant edge embedding and let a
learned projection $B$ expand it into tensor-product path weights,

$$
W_{e,\pi a}=\sum_q\varphi_{eq}B_{q,\pi a},
\qquad
m_i=\sum_{e:\,r(e)=i}\sum_{\pi,a}
W_{e,\pi a}T_{\pi a}(h_{s(e)},Y_e).
$$

Here $T_{\pi a}$ is the angular/source-feature tensor-product contribution.
The complete graph-wide edge-weight tensor $W$, complete edge messages, and
their adjoints are not model outputs. Substituting the first equation into the
second permits each expanded value to be consumed immediately during graph
reduction instead of stored as a long-lived edge array. Graph topology,
compact radial/angular inputs, and node features still scale with the physical
workload; streaming removes the wider scheduling intermediates, not all
edge-dependent storage.

Reverse mode applies the transpose of the same reassociation. Receiver-owned
adjoints can be contracted with recomputed edge-local values; source-contiguous
ownership accumulates node-feature adjoints; edge ownership produces radial,
angular, and coordinate derivatives. Symmetrix applies this bounded-lifetime
principle in both `generic` and `direct`.

## direct execution's receiver schedule

direct execution's concrete receiver-factorized implementation forms, for output group
$g$, the bounded receiver state

$$
S_{i,\pi,q,u,m_o}
=
\sum_{e:\,r(e)=i}
\varphi_{eq}\,
C_{\pi,u,m_o}\!\left(h_{s(e)},Y_e\right),
$$

and then applies the learned channel matrix

$$
m^{(g)}_{i,w,m_o}
=
\sum_{\pi\in\mathcal P_g,q,u}
S_{i,\pi,q,u,m_o}A^{(g)}_{q,\pi,u,w}.
$$

With receiver/output-component indices packed as rows, this is $m_g=S_gA_g$.
Backward first forms $\bar S_g=\bar m_gA_g^T$, followed by receiver-, source-,
and edge-owned kernels. The upstream direct execution package realizes this algebra as a
deterministic generated-CUDA PyTorch operator with forward, first backward,
double backward, and parameter gradients for both native `uvu` and `uvw`
connection modes.

In this tensor-product terminology, `uvu` ties input and output multiplicity
channels within each path, whereas `uvw` permits full input/output multiplicity
mixing.

Symmetrix implements the closest equivalent schedule as
`receiver_factorized`. It is currently an explicit FP32 experiment available
through the Python API with Kokkos Serial/OpenMP for ordinary MACE and
MACEField; it is not the production `direct` algorithm and is not implemented
for MH-1.

## Why production direct is different

Symmetrix does not use direct execution's receiver schedule unchanged because its extracted
R1+A1 integration boundary would create a large radial/coupling outer product
and substantially more arithmetic than consuming projected radial splines in
generated Execution R1.

The qualified ordinary MACE contracts use tied-channel `uvu` paths followed by
the learned dense node-level `A1` projection. Production `direct` evaluates
already projected radial splines inside a generated sparse contraction to
produce node-sized `Phi1`, then applies $L_{A1}$ once per node to produce
`A1`. Reverse first applies $L_{A1}^T$, then invokes generated source- and
edge-owned Execution R1 kernels. It removes the complete edge-sized
radial/message tape without constructing direct execution's explicit receiver state $S$.

This distinction is measured rather than stylistic. In the implemented
Symmetrix `receiver_factorized` control, representing the complete extracted
R1+A1 boundary through the receiver schedule forms a 64-wide radial/coupling
outer product for 40 path components and then applies the composed dense A1
projection. Its forward phase alone performs approximately 51.53 billion
floating-point operations (GFLOP) of receiver aggregation and 72.48 GFLOP of
receiver projection, or 124.00 GFLOP per evaluation. On the qualified
864-atom, one-core full-evaluator workload,
`receiver_factorized` measured 3,812.570 us/atom versus 230.820 us/atom for
`direct`, making `direct` 16.5x faster; `receiver_factorized` remained 1.217x
faster than materialized execution. This overhead is a consequence of this
model and chosen R1+A1 integration boundary, not an inherent limitation of
direct execution: direct execution compiles native tied-channel `uvu` as well as dense `uvw`
operators.

Symmetrix also owns a wider inference boundary than the upstream tensor-product
operator. It evaluates active-species-pair radial splines, both MACE
interactions, product bases, residuals, readouts, coordinate forces, device
virial reduction, optional MACEField response, and compatible MH-1 programs.
It targets Serial, OpenMP, CUDA, and HIP and integrates with ASE, LAMMPS, and
MPI without requiring PyTorch at inference. Standard R0, M0, and M1 are
portable build-time modules with runtime parameters; for ordinary MACE, only
model-dependent Execution R1 structure is specialized at runtime. Symmetrix
direct does not
claim direct execution's generated double-backward contract: observers, parameter
gradients, and double backward are outside production direct, while MACEField
higher responses use separate analytical reconstruction.

The relationship is therefore adoption of direct execution's algebraic lifetime result
and ownership principles plus independent full-evaluator systems design. direct execution
is an algorithmic reference and validation oracle, not a renamed Symmetrix
mode, source fork, build dependency, or runtime dependency. The measured
overall capacity advantage over fully materialized baselines also relies on
Symmetrix-specific M1 recomputation, prepared device geometry, and generated
streamed execution. The incremental `low_memory=True` gain over the current
retained `direct` evaluator primarily comes from forward/adjoint buffer reuse
and, in FP32, compact geometry; it must not be attributed to receiver
factorization or M1 recomputation alone.

Floating-point results need not be bitwise equal because reassociation changes
reduction order. The milestone therefore validates total/per-atom energy,
force components, stress components, finite-difference derivatives, repeated
execution, and oracle parity at precision-appropriate tolerances.

# Memory model

Let $N$ be nodes, $E$ directed edges, $C$ channels, $P$
tensor-product paths, $W_R$ the radial width, and $G$ product-basis graph
values. A broad materialized reverse can retain terms proportional to

$$
\mathcal O(ECP)
+\mathcal O(EW_R)
+\mathcal O(NCG).
$$

These terms correspond respectively to edge messages and their adjoints,
radial values and derivatives, and product values and adjoints.

Since $E$ grows with system size and neighbor count, the first two terms are
the capacity bottleneck. PyTorch additionally retains a general autograd graph,
operator temporaries, and allocator reserve. Original Symmetrix native
evaluation removes PyTorch and autograd overhead but retains broad native edge
tensors.

Streaming changes the persistent edge term to graph topology and geometry plus
bounded or compact scratch:

$$
\mathcal O(E)_{\text{graph/geometry}}
+\mathcal O(NC)_{\text{node boundaries}}
+\mathcal O(1)_{\text{bounded tile scratch}},
$$

where the last term is independent of total graph size for a fixed model and
execution tile.

When `low_memory=True` selects the capacity bundle, direct execution further:

- recomputes the M1 product DAG in team scratch during reverse instead of
  retaining $\mathcal O(NCG)$ values and adjoints;
- reuses dead ordinary-MACE forward buffers for adjoints;
- stores FP32 unit directions with FP64 radii for FP32 models, while preserving
  Cartesian FP64 geometry for FP64 models;
- retains the prepared graph and topology-reference geometry on device;
- reconstructs overwritten state only when analytical polarizability or Born
  effective charges are explicitly requested.

This is checkpointing applied to an equivariant graph program: extra local
arithmetic replaces long-lived memory. It is effective because the
reconstructed polynomial work is regular and cheaper than moving/retaining the
full DAG at scale.

# Radial functions and spline lowering

## MACE-Torch radial evaluation

For the qualified compact models, MACE-Torch first evaluates a Bessel basis and
a polynomial cutoff for every directed edge. With transformed distance
$\widetilde d_e=T_{z_i z_j}(d_e)$, basis frequency $\omega_n$, cutoff
$r_c$, and cutoff order $p$, the extracted operator is

$$
b_n(\widetilde d_e)
=\sqrt{\frac{2}{r_c}}\,
\frac{\sin(\omega_n\widetilde d_e)}{\widetilde d_e},
$$

$$
c_p(d_e)=
\left[
1-\frac{(p+1)(p+2)}{2}x^p
+p(p+2)x^{p+1}
-\frac{p(p+1)}{2}x^{p+2}
\right]\mathbf 1_{d_e<r_c},
\qquad x=\frac{d_e}{r_c}.
$$

The radial input and interaction-specific projected output are then

$$
\mathbf z_e=c_p(d_e)
\begin{bmatrix}
b_1(\widetilde d_e)&\cdots&b_{N_b}(\widetilde d_e)
\end{bmatrix}^{\!T},
\qquad
\mathbf R_e^{(t)}=\mathcal N_t(\mathbf z_e),
$$

where $\mathcal N_t$ is the learned normalized-SiLU radial network. For atomic
species $\alpha$ and $\beta$, the optional Agnesi transform used by MACE and
described in the ACEpotentials work [10] is

$$
T_{\alpha\beta}(d)
=
\left[
1+a\frac{(d/r_{0,\alpha\beta})^q}
{1+(d/r_{0,\alpha\beta})^{q-p}}
\right]^{-1},
\qquad
r_{0,\alpha\beta}
=\frac{r_\alpha^{\mathrm{cov}}+r_\beta^{\mathrm{cov}}}{2}.
$$

Without a distance transform, $T(d)=d$. The transform changes the Bessel
argument, while $c_p$ is still evaluated at the physical distance $d_e$.
MACE-Torch executes the sine, powers, transform, cutoff, and radial network as
PyTorch operations on every edge and retains the operations required for
autograd. Coordinate derivatives therefore come from differentiating this
original composition at runtime.

## Symmetrix compact representation

Symmetrix format-v2 preserves the parameters of that same radial operator in
the model artifact: Bessel frequencies and prefactor, polynomial-cutoff
parameters, optional Agnesi parameters and covalent radii, and the R0/R1 radial
network weights. It does not store a full table for every species pair.
When a composition is first prepared, Symmetrix evaluates the radial networks
on a uniform grid only for the unordered species pairs active in that system.
The default extraction grid has 256 nodes from approximately zero to $r_c$.

For one scalar radial output $f(r)$, let
$r_i=r_{\min}+ih$, $y_i=f(r_i)$, and $\tau=(r-r_i)/h$. Let $m_i$ be the
nodal slope produced by the spline boundary-value solve. Symmetrix stores the
equivalent cubic coefficients for the Hermite interpolant

$$
S_i(r)=
(2\tau^3-3\tau^2+1)y_i
+(\tau^3-2\tau^2+\tau)h m_i
+(-2\tau^3+3\tau^2)y_{i+1}
+(\tau^3-\tau^2)h m_{i+1}.
$$

Its radial derivative is evaluated analytically from the same polynomial,

$$
\frac{dS_i}{dr}
=\frac{1}{h}\frac{dS_i}{d\tau},
$$

so Symmetrix forces and stress are exact derivatives of the spline-approximated
energy rather than a separately fitted force model. The nodal derivative solve
uses a not-a-knot condition at the inner boundary and a clamped zero derivative
at $r_c$; together with the MACE cutoff this makes the interpolated radial
function reach the outer cutoff smoothly.

At inference, interval selection and four-coefficient polynomial evaluation
replace per-edge transcendental basis evaluation and the radial MLP. Values and
derivatives are laid out with radial functions contiguous for Kokkos vector or
subgroup traversal. Standard R0 additionally folds source embedding and the
first receiver linear map into its spline coefficients. The radial
representation used for the second interaction depends on the execution
algorithm. Production `direct` uses fully projected `radial_1` splines inside
generated Execution R1. The internal table-driven stateful schedule and public
`receiver_factorized` mode instead use penultimate `execution_radial_1`
splines and retain the final learned radial projection as a runtime
contraction, reducing coefficient storage where that factorization is
advantageous.

This lowering has four practical consequences:

- expensive sine, power, distance-transform, and radial-network work is paid
  once per active species pair rather than once per edge and evaluation;
- explicit spline derivatives avoid constructing a radial autograd tape and
  fit directly into the streamed coordinate reverse;
- a universal checkpoint does not allocate tables for all possible element
  pairs when the current structure contains only a small active subset;
- model parameters remain runtime data and are not embedded into generated
  host, CUDA, or HIP code.

The tradeoff is controlled interpolation error: Symmetrix evaluates a cubic
approximation to MACE-Torch's radial composition, not the transcendental/MLP
expression itself at each edge. FP64 reduces arithmetic and accumulation error
but does not by itself remove spline interpolation error; increasing
`num_spline_points` tightens that approximation at the cost of coefficient
storage and preparation time. Compact format-v2 therefore admits the radial
families it can reconstruct exactly before interpolation: Bessel basis,
applied polynomial cutoff, no transform or the Agnesi transform, and supported
normalized-SiLU networks. Unsupported radial architectures must use the
materialized legacy pair-spline export where available or are rejected rather
than silently changing the model.

The frozen format-v1 path differs mainly in packaging. Extraction evaluates
every requested species-pair radial function through MACE-Torch and serializes
all projected spline values and derivatives into JSON. Format-v2 serializes the
shared radial definition and materializes active-pair tables in native code,
which is substantially better suited to universal models and prepared
simulation workloads.

# Execution algorithms

## Materialized

Materialized execution is the compatibility baseline. It evaluates radial
tables and edge tensor products broadly, retains them for reverse, and uses
general Kokkos kernels. Its value is model coverage and a clear numerical
rollback. Its cost is edge-proportional storage and a reverse schedule with
limited parallel ownership. In the 4,000-atom develop trace, one
`reverse_Phi1` kernel consumes 65.0% of all GPU kernel time and about 83.8 ms
per evaluation.

## Generic

Generic execution removes both interaction radial edge tensors and streams the
path tables through Kokkos. It requires no runtime compiler and supports the
broadest accelerator portability. It is much smaller and faster than develop
on CUDA, but table interpretation, general loop nests, and conservative
ownership make it slower than direct. On one CPU thread it is currently slower
than develop; generic is a portability/reference route rather than the primary
host optimization target.

## Direct

Direct execution normalizes the extracted tensor-product semantics into a
versioned contract. The artifact cache identity includes dimensions, sparse
paths and coefficients, layouts, derivative capabilities, precision, backend,
target architecture, generator version, and compiler options. Learned weights
remain runtime data, so checkpoints with the same architecture can share code
without embedding parameters.

Host Execution R1 code is compiled as a C++20 shared library. CUDA uses
self-contained device source compiled in-process by NVRTC [11] to the active
compute capability; HIP uses hipRTC [12]. A cache miss builds once and later
processes validate and load the content-addressed artifact. Explicit direct
execution fails closed on generation, compilation, fingerprint, ABI, or load
failure. It does not silently become generic.

For ordinary MACE, generated Execution R1 forward lowers the fixed sparse
coupling program and projected radial operations. Its reverse uses cooperative
receiver, source, and edge ownership with subgroup reductions. Built-in R0,
M0, and M1 modules cover common model topology without embedding model
weights:

- standard R0 uses runtime spline/model data but shape-specialized loop and
  launch structure;
- standard M0 lowers the common product topology and avoids retained
  graph-sized polynomial tensors;
- standard M1 supplies specialized product forward/reverse where backend and
  model admission allow it, while recomputation controls its storage.

These built-in modules are compiled with Symmetrix and remain the preferred
zero-cold-start implementations. Direct low-memory CUDA/HIP execution can now
compile compatible unmatched M0 and R0 contracts into separate NVRTC/hipRTC
modules. Their artifact identities contain structural topology, precision,
backend target, ABI, generator, and schedule, while channels, species, learned
weights, nodes, and edges remain runtime quantities. Low-memory admission
therefore depends on state-free, alias-safe operator capabilities rather than a
particular built-in name.

## MH-1

Compatible format-v3 MH-1 contracts generate both interactions' graph programs
and their pair-conditioned prefix/density networks. Graph-wide generated node
phases use a persistent channel-contiguous layout. CUDA dense node linears use
8-node by 32-channel tiles; receiver and source reverse schedules use
contract-derived partitions; edge reverse uses subgroup ownership and split
prefix/harmonic kernels where full fusion would exceed the qualified resource
budget.

MH-1 has separate compact-phi and node-state recomputation policies because
its conditioned networks create different lifetimes from ordinary MACE. These
are composable internal resource policies, but the public low-memory bundle is
currently defined for ordinary MACE/MACEField rather than MH-1. The compatible
MH-1 fast path rejects `materialized`, accepts `generic`, and uses a separate
generated `direct` ABI.

# CPU implementation decisions

The CPU backend is native C++20 with Kokkos Serial/OpenMP and BLAS. Its main
efficiency decisions are:

- generated straight-line/path-group Execution R1 loops remove Python dispatch, e3nn
  object traversal, and runtime sparse-table branches;
- graph-wide dense node operations use GEMM/BLAS where the layout admits it;
- host generated kernels use fixed dimensions, contiguous channel ownership,
  vectorizable inner loops, and bounded scratch;
- receiver/source schedules preserve locality and avoid complete edge-message
  writes and rereads;
- one Kokkos thread and one BLAS thread form the primary optimization baseline,
  avoiding nested parallel oversubscription;
- the packaged core defaults to the native CPU architecture, while
  `SYMMETRIX_HOST_ARCH=x86-64-v3` selects a portable x86-64-v3 build, disables
  Kokkos native targeting, and applies the explicit baseline to bundled Kokkos,
  KokkosKernels, SpheriCart [13], the core, and Python bindings. The milestone
  CPU comparison used that explicit portable override. Host RTC artifacts
  retain target identity in their cache key.

On the 864-atom one-core workload, current direct low-memory reaches
262.962 us/atom, versus 4,767.777 for origin/develop and 12,610.511 for
PyTorch/e3nn: 18.13x and 47.96x speedups. The compiler-free generic path is
5,615.376 us/atom, showing that memory streaming alone is not enough for host
performance; contract specialization and dense-kernel organization matter.

# GPU implementation decisions

The CUDA evaluator keeps topology, geometry, features, adjoints, pair forces,
and stress work on device. Prepared graph tokens validate that a reverse or
response consumes the graph generation that produced its primal state.
Positive skin permits repeated evaluations to update device geometry from
current positions without rebuilding or uploading the complete edge-vector
array.

The optimized path uses:

- one ordered evaluator stream and a terminal completion boundary;
- NVRTC-generated exact-SM Execution R1 kernels with cooperative
  warp/subgroup work;
- persistent launch profiles rather than one block per small logical item;
- built-in R0/M0/M1 kernels for common structures, with runtime-generated
  state-free M0/R0 modules for compatible unmatched low-memory contracts;
- M1 recomputation in shared-memory/team tiles;
- compact FP32 direction/FP64 radius geometry in FP32 low-memory mode;
- graph-wide tiled node linears and KokkosKernels/cuBLAS where profitable;
- device force reduction from directed pair forces;
- nine small device reductions for the 3x3 virial, followed by only a 3x3
  host result transfer.

The last point preserves the exact ASE sign convention
$\sigma_{ab}=-V^{-1}\sum_e f_{e,a}r_{e,b}$ without transferring
$\mathcal O(E)$ pair forces or edge vectors. At 4,000 atoms, device force
reduction costs about 0.044 ms and
all stress-component reductions together about 0.071 ms per evaluation in the
Nsight Systems trace.

The 4,000-atom direct result is 4.008 us/atom at 1,118 MiB process VRAM,
versus 60.326 us/atom and 9,612 MiB for origin/develop, and 98.995 us/atom and
29,248 MiB for PyTorch/e3nn. PyTorch/cuEquivariance reaches 31.052 us/atom at
7,136 MiB on that same exact-cutoff graph. Direct processes 452,342 skin
candidates while the controls process 364,000 exact-cutoff edges.

![OMAT-0 FP32 inference time for energy, forces, and stress. Panel (a) uses one physical CPU core and 864 atoms; panel (b) uses one RTX 5090 and 4,000 atoms. Current direct and generic execution use a 6.0 A model cutoff with a 0.5 A skin, while origin/develop and PyTorch use a 6.0 A exact-cutoff graph. Lower is better; both horizontal axes are logarithmic.](figures/streamed_edge_performance.svg){width=100%}

Nsight Systems shows that the old monolithic reverse bottleneck has been
replaced by a balanced set of generated and standard modules. Direct's largest
kernel, generated Execution R1 reverse, is 21.1% of GPU kernel time at about
3.15 ms.
Nsight Compute identifies its next boundary as cache/dependency latency:
87.44% L2 throughput, 98.58% L2 hit rate, 5.55% DRAM throughput, 128
registers/thread, and 32.34% achieved occupancy. Future work should first test
register-pressure reduction, load/store coalescing, and cache-resident data
reuse against end-to-end timing; raw DRAM bandwidth is not the current limit.

# Capacity and memory scaling

Increasing feasible system size on fixed hardware is a primary objective of
streamed-edge execution, not a side effect of its kernel optimization. Runtime
speed determines how quickly a supported system can be evaluated; peak memory
determines whether that system can be evaluated at all. The capacity campaign
therefore carries equal weight to the throughput comparison in this milestone.

Maximum capacity depends on the model, precision, local neighbor density,
cutoff, graph skin, allocator state, and device memory. The values below are
therefore measured workload boundaries, not universal or theoretical maxima.
The post-merge campaign used FP32 on one 32,607 MiB NVIDIA RTX 5090 with CUDA
13.3 and Kokkos CUDA. Its periodic wurtzite-AlN cells used a 6.0 A model
cutoff, a 0.5 A skin, and therefore a 6.5 A candidate-list cutoff with 109
directed candidate edges per atom. OMAT-0 requested energy, forces, and stress;
MACEField additionally requested polarization, but not Born effective charges
or polarizability.

Each boundary was run in a fresh process. The largest successful size and the
next larger failed size were each reproduced twice. All first failures were
clean CUDA allocator failures after the bundled Sphericart edge-offset fix.

| Model and policy | Largest demonstrated success | Directed edges | Next failed size | Capacity gain |
|---|---:|---:|---:|---:|
| OMAT-0, current direct retained | 219,488 atoms | 23,924,192 | 237,276 atoms | reference |
| OMAT-0, current direct low memory | **389,344 atoms** | 42,438,496 | 415,292 atoms | **1.774x (+77.4%)** |
| MACEField, current direct retained | 202,612 atoms | 22,084,708 | 219,488 atoms | reference |
| MACEField, current direct low memory | **340,736 atoms** | 37,140,224 | 364,500 atoms | **1.682x (+68.2%)** |

Thus, on the tested 32 GB GPU, the largest demonstrated low-memory systems are
389,344 atoms for OMAT-0 and 340,736 atoms for MACEField. These are lower-bound
capacity demonstrations: the next tested cell is the failure bound, but the
discrete cell series does not locate the exact atom at which allocation would
fail. The detailed failure allocations and reproducibility ledger are reported
in `benchmarks/low_memory_capacity.md`.

A separate extreme-scale qualification on one A100-SXM4-80GB reaches
1,000,188 atoms and 109,020,492 directed candidates twice with OMAT-0-medium
FP32 direct low-memory execution. The next `n64` cell, 1,048,576 atoms and
114,294,784 expected candidates, fails twice with a true 2 GiB allocator OOM.
The earlier illegal-address boundary was a signed 32-bit flattened-offset bug,
not a capacity limit. Generated R1 and the maintained SpheriCart patch now use
wide global offsets; details and binary/model hashes are in
`benchmarks/extreme_scale_indexing_20260829.md`.

The retained rows are the current `direct` evaluator with
`low_memory=False`; they are not `origin/develop`. Both current policies use
generated Execution R1 and the same selected M0/R0 implementations, and the
retained public
configuration's
automatic M1 policy already selects recomputation when scratch permits. The
capacity delta therefore primarily measures adjoint-state reuse and, for FP32,
compact edge geometry in addition to the strict policy selection.

At a matched 131,072 atoms and 14,286,848 directed edges, low memory reduces
sampled process GPU memory substantially while preserving steady-state
throughput:

| Model | Current direct retained | Current direct low memory | GPU memory saving | Time change |
|---|---:|---:|---:|---:|
| OMAT-0 | 19,388 MiB; 4.3843 us/atom | 11,322 MiB; 4.3345 us/atom | 8,066 MiB (41.6%) | 1.14% faster |
| MACEField | 20,064 MiB; 5.1904 us/atom | 12,256 MiB; 5.1029 us/atom | 7,808 MiB (38.9%) | 1.69% faster |

The near-boundary OMAT-0 timing is intentionally excluded from speed claims:
with only a small memory reserve, managed-memory paging makes it unstable even
though the evaluation completes.

## Comparison with reference implementations

A fresh reference campaign used the same OMAT-0 FP32 energy/forces/stress
request on the same RTX 5090. Each successful endpoint and immediately larger
failure were observed twice in fresh processes. The references used
deterministically perturbed wurtzite AlN with a 6.0 A exact graph; the current
direct capacity campaign used unperturbed wurtzite AlN with a 6.0 A model
cutoff and 0.5 A skin.

| Implementation | Largest demonstrated success | Directed edges/candidates | First failed size | Largest-success atom-count ratio |
|---|---:|---:|---:|---:|
| Current direct low memory | **389,344 atoms** | 42,438,496 candidates | 415,292 atoms | 1.00x |
| `origin/develop` materialized | 13,500 atoms | 1,228,500 edges | 16,384 atoms | **28.8x** |
| PyTorch/cuEquivariance | 19,652 atoms | 1,788,332 edges | 23,328 atoms | **19.8x** |
| PyTorch/e3nn | 4,000 atoms | 364,000 edges | 5,324 atoms | **97.3x** |

![CUDA capacity comparison on the 32,607 MiB RTX 5090. Panel (a) shows the largest successful current-direct FP32 calculations with retained and low-memory policies; both use 109 skin-expanded candidates per atom. Panel (b) shows the largest successful OMAT-0 size as a circle and the next tested failure as a cross. Current direct uses 109 candidates per atom, while the three reference workloads use 91 exact-cutoff edges per atom; the logarithmic axis communicates the measured scale without implying topology-normalized limits.](figures/streamed_edge_capacity.svg){width=100%}

The first failures are a 128 MiB Kokkos CUDA allocation for origin/develop, a
10.12 GiB cuEquivariance `uniform_1d` reverse allocation, and a 9.24 GiB e3nn
tensor-product allocation. The largest successful reference runs hold
30,850 MiB, 30,878 MiB, and 29,246-29,248 MiB, respectively. Recoverable
allocator warnings occur in the successful PyTorch boundary runs, so their
timings are capacity evidence rather than throughput results.

The earlier milestone comparison remains the matched resident-memory result:

| Implementation | 864 atoms | 4,000 atoms | 4,000-atom memory / direct |
|---|---:|---:|---:|
| Current direct low memory | 858 MiB | **1,118 MiB** | 1.00x |
| `origin/develop` materialized | 2,594 MiB | 9,612 MiB | 8.60x |
| PyTorch/cuEquivariance | 2,136 MiB | 7,136 MiB | 6.38x |
| PyTorch/e3nn | 9,104 MiB | 29,248 MiB | 26.16x |

At 4,000 atoms direct processes 452,342 candidates at a 6.5 A list cutoff,
whereas the three references process 364,000 exact-cutoff edges at 6.0 A.
Direct retains, allocates, and traverses candidates between 6.0 and 6.5 A; its
geometry route clamps their radius to the model cutoff, where the cutoff and
clamped spline derivative make their energy and coordinate contributions zero.
The largest-success atom-count ratios therefore compare bracketed measured
workload boundaries rather than exact, topology-normalized capacity limits.
Notably, the direct workload has 109 candidate edges per atom versus 91 exact
edges per atom for the references. Full commands, hashes, failure classes, and
boundary timings are recorded in
`benchmarks/reference_cuda_capacity_20260823.md`.

## FP64 direct low-memory qualification

`streamed_edges="direct"` with `low_memory=True` supports FP64. When capacity
is selected, it retains Cartesian FP64 edge geometry while enabling M1
recomputation and forward/adjoint state reuse. The historical qualified
32,000-atom, 3,488,000-candidate CUDA comparison forced that capacity policy
before whole-bundle automatic selection was introduced:

| Model | Current direct retained | Current direct low memory | Result |
|---|---:|---:|---|
| OMAT-0 FP64 | 9,624 MiB; 34.667 us/atom | 5,844 MiB; 33.938 us/atom | 39.3% less memory; 2.1% faster |
| MACEField FP64 | 9,842 MiB; 35.897 us/atom | 6,188 MiB; 35.142 us/atom | 37.1% less memory; 2.1% faster |

Low-memory and retained FP64 summaries agree within `7e-15` for the recorded
energies, force summaries, stress, and polarization. A fresh release-binary
smoke test also ran 864 atoms and 97,762 skin-expanded candidates with energy,
forces, and stress: NVRTC built and selected `jit-r1-gen2-f64`, low memory and
M1 recomputation were active, and evaluation completed at 26.371 us/atom with
1,070 MiB resident VRAM. No FP64 maximum-capacity search has yet been
performed, so the maximum-atom boundaries above remain FP32-only results.

# Comparison with PyTorch MACE

PyTorch MACE expresses the model compositionally: radial embedding, equivariant
tensor products, scatter reductions, product bases, readouts, and
autograd-derived forces/stress. The e3nn path is the original
training/source-of-truth implementation. With `enable_cueq=True`, MACE converts
the loaded model through `run_e3nn_to_cueq` and replaces eligible linear,
channelwise tensor-product, symmetric-contraction, and fused-convolution
operators with cuEquivariance implementations. The MACE calculator, graph
construction, result assembly, and autograd-derived coordinate/cell gradient
boundary remain in PyTorch.

Symmetrix changes the execution representation, not the learned model:

| Concern | PyTorch MACE/e3nn | PyTorch MACE/cuEquivariance | Streamed Symmetrix direct |
|---|---|---|---|
| Operator expression | Dynamic module/e3nn composition | MACE model converted to cuEquivariance operators | Extracted normalized contract plus native stages |
| Radial evaluation | Per-edge Bessel/cutoff/network PyTorch operations | Same MACE radial composition in PyTorch | Active-pair cubic splines with analytic derivatives and optional factorized final projection |
| Edge messages | Framework tensors retained as needed by autograd | Framework tensors retained as needed by autograd | Streamed/recomputed; no full broad message tape |
| Reverse | General autograd | General autograd over accelerated operators | Explicit coordinate adjoint program |
| Sparse coupling | Runtime e3nn instructions | cuEquivariance segmented/tensor-product kernels | Generated literal/path-group program |
| Node dense work | Framework linear/einsum operations | Eligible linears replaced during conversion | BLAS or graph-wide tiled native kernels |
| Geometry | Framework tensors constructed per graph | Framework tensors constructed per graph | Prepared device topology and geometry updates |
| Stress | Autograd through displacement/cell | Autograd through displacement/cell | Device pair-force virial reduction |
| Compilation | PyTorch/e3nn kernels | cuEquivariance CUDA operators selected by MACE conversion | C++ build plus contract-specific host RTC/NVRTC/hipRTC |
| Training gradients | Supported | Supported within the converted PyTorch model contract | Parameter gradients and double backward not a production direct contract |

cuEquivariance is therefore a strong accelerated PyTorch reference, not a
different learned operator. It is 2.40x faster than e3nn at 864 atoms and
3.19x faster at 4,000 atoms. Its maximum 4,000-atom differences from e3nn are
`9.77e-7 eV/atom`, `4.69e-6 eV/A` in force components, and
`3.35e-8 eV/A^3` in stress components, consistent with FP32 reassociation.
Direct is still 9.52x faster at 864 atoms and 7.75x faster at 4,000 atoms,
while using 2.49x and 6.38x less sampled VRAM, respectively. The remaining gap
reflects the wider optimization boundary: explicit reverse mode, streamed
state lifetimes, prepared geometry, compact checkpointing, and reduced device
outputs in addition to tensor-product specialization.

# MACEField

MACEField [9] extends the energy model with dependence on an external electric
field $\boldsymbol{\mathcal E}$. Its macroscopic polarization is the field
derivative

$$
\mathbf P=-\frac{1}{V}\frac{\partial E}{\partial\boldsymbol{\mathcal E}},
$$

and further field or mixed field-coordinate derivatives yield polarizability
and Born effective charges (BECs), respectively.

MACEField reuses the ordinary MACE Execution R1 contract. The field transform
conditions the architectural first-update state before the second interaction's
`linear_up`. Reverse first applies Execution R1 reverse, then the `linear_up`
transpose, and only then the field-transform transpose.
Energy, forces, stress, and polarization therefore use the same prepared direct
path and low-memory buffer reuse. Returning polarization is a small reduced
output and does not require the full analytical response reconstruction.

Polarizability and Born effective charges (BECs) are differentiated analytically
through separate response kernels. Low-memory response reconstructs overwritten
A0/A1/M1/H2 state in aliased allocations. BEC still has a larger response
boundary and may copy directed-edge field-force derivatives for atom reduction;
that limitation does not apply to ordinary energy/forces/stress/polarization.

# LAMMPS and MPI

The Kokkos pair style exposes `materialized`, `generic`, and `direct` for
standard MACE/MACEField. Direct requires a prepared artifact because LAMMPS
[14] loads executable artifacts but never invokes a compiler. The Python-only
`receiver_factorized` experiment is intentionally rejected by the pair style.

Domain-decomposed execution communicates owned/ghost feature state at the
interaction boundary. Symmetrix operates only on the rank-local graph and its
ghost atoms; it does not replicate all atoms across GPUs. GPU-aware MPI allows
LAMMPS/Kokkos to pass device buffers directly. Host-staged MPI adds device/host
copies and synchronization, while redundant evaluator-side fences have been
reduced so communication dependencies define the remaining boundaries.

Release qualification is narrower than implementation capability. Same-node
CUDA-aware multi-GPU FP32 standard MACE has been exercised, while CUDA
MACEField, CUDA FP64, HIP MPI, other providers, and cross-node transports remain
explicit qualification targets.

# Correctness and lifecycle strategy

Specialized `direct` and `receiver_factorized` execution fail closed when their
artifact contracts cannot be established; they do not silently change
algorithms. Separately selected `generic` and `materialized` modes serve as
numerical or compatibility oracles. The important gates are:

- exact model/contract/payload fingerprints and versioned artifact ABI;
- FP32/FP64 energy, per-node energy, force, and stress comparisons;
- finite-displacement force and finite-cell stress derivatives;
- MACEField finite-field response and low-memory reconstruction checks;
- empty, one-edge, ragged-degree, reordered, grown, and shrunk graph cases;
- graph-token freshness and stale-token rejection;
- repeated deterministic execution and explicit mode/policy restoration;
- Serial/OpenMP/CUDA runtime tests and HIP build/runtime qualification where
  hardware is available;
- LAMMPS one-rank versus MPI ownership/migration comparisons;
- profiler checks for launch selection, synchronization, and transfer size.

The release qualification found and fixed one CUDA ownership bug: generic
standard-R0 reverse selected an edge executor but passed the receiver view
owned only by prepared/direct execution. Generic now passes and retains its own
receiver map, including above 100,000 edges. A 155,164-edge runtime case
confirms the corrected lifecycle.

# Support and limitations

Functional support is mode-specific:

| Capability | Materialized | Generic | Receiver-factorized | Direct low-memory |
|---|---|---|---|---|
| Ordinary energy/forces/stress | Yes | Yes | Yes, FP32 | Yes, FP32/FP64 |
| MACEField energy/forces/stress/polarization | Yes | Yes | Yes, FP32 | Yes, FP32/FP64 |
| MACEField polarizability/BEC | Analytical path | Analytical path | Analytical path | Reconstructs overwritten state; no generated tangent kernels |
| Compatible MH-1 | Disabled | Yes | No | Ordinary bundle not applicable; generated FP32 direct has separate policies |
| Execution observer | No | No | Opt-in | No |
| Parameter gradients | No | No | Opt-in for admitted ordinary compact models | No |
| Double backward | No | No | No | No |

Backend and deployment support is distinct from functional admission:

| Backend/API | Materialized | Generic | Receiver-factorized | Direct low-memory |
|---|---|---|---|---|
| Kokkos Serial/OpenMP | Yes | Yes | Python API, FP32 | Yes |
| Kokkos CUDA | Yes | Yes | No | Yes |
| Kokkos HIP | Implemented | Implemented | No | Ordinary Execution R1 implemented |
| LAMMPS | Standard MACE/MACEField | Standard MACE/MACEField | No | Standard MACE/MACEField with prepared artifact |

Here “implemented” means that the code and build boundary exist; it does not
claim milestone runtime qualification on physical hardware. The milestone
runtime-qualified Serial/OpenMP and CUDA paths as described above. HIP hardware
qualification remains narrower. Receiver-factorized observers and parameter
gradients require their explicit Python controls, retained Cartesian geometry,
and an admitted ordinary compact model; residual-first parameter gradients are
not supported.

Matched evidence now favors low memory for eligible direct standard MACE and
MACEField: it is faster in the current one-core OpenMP, CUDA FP32, and CUDA
FP64 comparisons while using substantially less memory. This does not justify
removing retained execution. Low-memory admission requires generated Execution R1,
state-free alias-safe M0/R0 implementations, an admitted M1 scratch tile, and
no observer or parameter gradient. MH-1 uses different controls, and analytical MACEField BEC or
polarizability must reconstruct overwritten state. Retained execution also
remains the numerical oracle and operational rollback. An older one-shot CUDA
capacity sample was slower with the complete low-memory bundle, so there is no
unconditional performance guarantee even though the controlled current
comparisons favor it.

A future default should therefore be eligibility-aware rather than the literal
boolean `True`: an automatic state should attempt low memory after `direct`
resolution and fall back to retained execution with a diagnostic; explicit
`True` should remain strict and fail closed; explicit `False` should guarantee
the retained rollback. This tri-state policy requires an API change and broader
Serial/OpenMP/CUDA/HIP and MPI qualification before rollout. Low memory remains
opt-in for this milestone. `legacy` historical terminology in benchmark records
is frozen evidence; maintained APIs and documentation use the canonical names
above.

# Release conclusion

The principal result is a change in feasible problem size, not only a faster
kernel schedule. Streamed-edge execution makes mathematical ownership determine
memory lifetime:

- receiver sums do not require retaining complete edge messages;
- reverse recomputes cheap local values and assigns source/edge owners;
- product DAGs are checkpointed in bounded scratch;
- fixed sparse semantics become compiler-visible through exact contracts;
- graph, geometry, force, and stress data remain on the target device;
- CPU and GPU schedules specialize independently while sharing the same
  extracted operator and correctness oracles.

On the 32,607 MiB RTX 5090 capacity workload, the low-memory policy raises the
largest demonstrated FP32 system by 77.4% for OMAT-0 and 68.2% for MACEField,
while saving 41.6% and 38.9% sampled GPU memory at a matched 131,072 atoms. In
the OMAT-0 boundary campaigns, the 389,344-atom direct largest success is
28.8x the origin/develop largest success, 19.8x the
PyTorch/cuEquivariance largest success, and 97.3x the PyTorch/e3nn largest
success, while processing more candidate edges per atom. These are ratios of
bracketed success points rather than exact capacity thresholds. The gains turn
calculations that exhaust device memory in the retained or framework
implementations into executable workloads on the same GPU.

This larger feasibility envelope is achieved without trading away throughput.
The milestone workloads show 18.13x CPU and 15.05x GPU speedups over the
original develop Kokkos evaluator, and 47.96x CPU and 24.70x GPU speedups over
PyTorch/e3nn. Against PyTorch/cuEquivariance, the measured CUDA speedups are
9.52x at 864 atoms and 7.75x at 4,000 atoms. The remaining direct CUDA
bottleneck is a bounded, cache-resident generated reverse kernel rather than
the former graph-sized materialized reverse. That combination of substantially
greater capacity and high throughput is the intended architectural outcome of
streamed-edge execution.

# References

1. I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner, and G. Csányi,
   "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and
   Accurate Force Fields," *Advances in Neural Information Processing Systems*
   **35**, 11423-11436 (2022).
   [arXiv:2206.07697](https://arxiv.org/abs/2206.07697).

2. C. R. Trott *et al.*, "Kokkos 3: Programming Model Extensions for the
   Exascale Era," *IEEE Transactions on Parallel and Distributed Systems*
   **33**, 805-817 (2022).
   [doi:10.1109/TPDS.2021.3097283](https://doi.org/10.1109/TPDS.2021.3097283).

3. V. Chorošajev and C. Bény, "direct execution: Streaming Equivariant Tensor Product
   Convolutions," arXiv preprint (2026).
   [arXiv:2607.18074](https://arxiv.org/abs/2607.18074).

4. A. Paszke *et al.*, "PyTorch: An Imperative Style, High-Performance Deep
   Learning Library," *Advances in Neural Information Processing Systems*
   **32**, 8024-8035 (2019).
   [paper](https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library).

5. M. Geiger and T. Smidt, "e3nn: Euclidean Neural Networks," arXiv preprint
   (2022). [arXiv:2207.09453](https://arxiv.org/abs/2207.09453).

6. NVIDIA, *cuEquivariance*, version 0.11.1 (2026).
   [software repository](https://github.com/NVIDIA/cuEquivariance/tree/v0.11.1).

7. I. Batatia *et al.*, "A Foundation Model for Atomistic Materials
   Chemistry," *The Journal of Chemical Physics* **163**, 184110 (2025).
   [doi:10.1063/5.0297006](https://doi.org/10.1063/5.0297006).

8. A. H. Larsen *et al.*, "The Atomic Simulation Environment - A Python
   Library for Working with Atoms," *Journal of Physics: Condensed Matter*
   **29**, 273002 (2017).
   [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).

9. B. A. A. Martin, A. M. Ganose, V. Kapil, T. Li, and K. T. Butler,
   "General Learning of the Electric Response of Inorganic Materials," *PRX
   Intelligence* **1**, 013006 (2026).
   [arXiv:2508.17870](https://arxiv.org/abs/2508.17870).

10. W. C. Witt *et al.*, "ACEpotentials.jl: A Julia Implementation of the
    Atomic Cluster Expansion," *The Journal of Chemical Physics* **159**,
    164101 (2023).
    [doi:10.1063/5.0158783](https://doi.org/10.1063/5.0158783).

11. NVIDIA, *CUDA Runtime Compilation API*, CUDA Toolkit documentation
    (accessed 22 August 2026).
    [documentation](https://docs.nvidia.com/cuda/nvrtc/).

12. AMD, *HIP Runtime Compilation (hipRTC)*, ROCm documentation
    (accessed 22 August 2026).
    [documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hiprtc.html).

13. F. Bigi, G. Fraux, N. J. Browning, and M. Ceriotti, "Fast Evaluation of
    Spherical Harmonics with Sphericart," *The Journal of Chemical Physics*
    **159**, 064802 (2023).
    [doi:10.1063/5.0156307](https://doi.org/10.1063/5.0156307).

14. A. P. Thompson *et al.*, "LAMMPS - A Flexible Simulation Tool for
    Particle-Based Materials Modeling at the Atomic, Meso, and Continuum
    Scales," *Computer Physics Communications* **271**, 108171 (2022).
    [doi:10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171).
