# FourierD3 Reciprocal-Space Dispersion — Mathematical and Convention Specification

This maintained specification defines the numerical conventions used by the
FourierD3 implementation and independent oracle fixtures.

This document fixes the model, the reciprocal-space formulation, every unit conversion,
and every sign and normalization that a later independent implementation must reproduce
exactly. It is written against three sources, in this order of authority for numerical
agreement:

1. **NVIDIA ALCHEMI Toolkit-Ops v0.5.0**, `nvalchemiops/interactions/dispersion/`
   (`_fourier_dftd3.py`, `_c6_decomposition.py`) and its NumPy oracle
   `test/interactions/dispersion/_fourier_reference.py`, inspected at tag `v0.5.0`,
   commit `6702138e404d40ea79dec62f68c2fa67e858580e`. Reference fixtures are generated
   from this revision, so where the toolkit and the paper differ, the toolkit is
   normative and the difference is called out explicitly below.
2. **Valeeva et al., "A fast summation method for the DFT-D3 dispersion correction",
   arXiv:2607.15103v2** — the published method. Equations cited as (P.n).
3. **Existing Symmetrix conventions** — `libsymmetrix/source/d3_kokkos_impl.tpp`,
   `libsymmetrix/source/dispersion_parameters.{hpp,cpp}`,
   `libsymmetrix/source/dispersion_parameter_provenance.hpp`,
   `symmetrix/source/symmetrix/corrections.py`, and `docs/user/dispersion_corrections.md:44`.

Where this document says *normative*, an implementation that deviates is wrong even if
its energy looks plausible. Where it says *design choice*, the choice is recorded so
that a deviation is a documented decision rather than an accident.

---

## 1. Scope

FourierD3 evaluates a **modified** two-body DFT-D3(BJ) dispersion correction for fully
three-dimensional periodic systems in `O(N log N)`, with **no real-space cutoff on the
dispersion sum**. The only real-space cutoff is the short coordination-number neighbour
list.

Normative statements about the model:

- Damping is **Becke–Johnson only**. Zero damping is not part of this model.
- The coordination-number function is the **modified** one of §3.2, not the standard D3
  function. This changes the model: FourierD3 is *not* expected to reproduce real-space
  D3, even at a converged mesh (P.II.C, and the toolkit's dispersion guide).
- The dispersion sum is a **lattice sum over all periodic images**, evaluated in
  reciprocal space with **no Ewald splitting parameter** and **no real-space/reciprocal
  split**. There is exactly one sum, plus the self term.
- Version 1 requires a nonsingular, fully periodic cell on all three axes. Open,
  partially periodic, slab-corrected, MPI-distributed, ATM (triplet), and D4 variants are
  out of scope.
- The implementation has **no PyTorch and no NVIDIA runtime dependency**. PyTorch, Warp,
  and `nvalchemi-toolkit-ops` are numerical references and fixture sources only, and must
  never be imported by an installed calculator. There is no host or real-space fallback,
  no public energy-only intermediate state, and no silent reduction in accuracy when an
  FFT provider is absent.

## 2. Units

All quantities are in the Symmetrix working unit system: **length in Å, energy in eV**.
Forces are eV/Å; stress is eV/Å³.

| Quantity | Symbol | Unit | Source |
|---|---|---|---|
| positions, cell, cutoff, `rcov`, `a2`, mesh spacing | — | Å | caller |
| dispersion energy | $E$ | eV | output |
| pair coefficients $C_6$ | $C_6$ | eV·Å⁶ | converted table |
| $q_Z$ (tabulated `r2r4` / `r4r2` / `sqrt_q`) | $q_Z$ | dimensionless | table |
| $a_1$, $s_6$, $s_8$ | — | dimensionless | functional parameters |
| $R_0$, $r$ | — | Å | derived |
| reciprocal vectors $k$ | $k$ | Å⁻¹ | derived |
| virial $\partial E/\partial u$, stress | — | eV, eV/Å³ | output |

### 2.1 Unit conversions (normative)

The DFT-D3 parameter tables are conventionally in atomic units. The existing Symmetrix
tables are already converted, and FourierD3 **must reuse them unchanged** — it must not
introduce a second copy of any table:

- $C_6$: table stored in eV·Å⁶, converted from hartree·bohr⁶ by
  `hartree_bohr6_to_ev_angstrom6` = **0.59752721065076475** eV·Å⁶/(Eh·bohr⁶)
  (`dispersion_parameter_provenance.hpp:47`). Equivalently
  $\text{hartree\_to\_ev} \times \text{bohr\_to\_angstrom}^6$
  = $27.21138505 \times 0.52917726^6$.
- Length: `bohr_to_angstrom` = **0.52917725999999998** Å/bohr.
  Therefore the BJ second parameter converts as $a_2^{(\text{Å})} = 0.52917726\,a_2^{(\text{bohr})}$,
  matching `rs8*bohr_to_angstrom` in `pair_energy_factor` (existing D3).
  $a_1$ is dimensionless and is **not** converted.
- $q_Z$: dimensionless; identical to the value real-space D3 uses.
- `rcov`: stored in Å **with Grimme's 4/3 factor already folded in**
  (`d3_rcov_angstrom`, e.g. H = 0.42666665874797699 = (4/3)·0.32). FourierD3 uses the
  same table as real-space D3 and must not rescale it. The bare radius is
  $\tfrac{3}{4}\times$ the table value and is used only for the transition radius (§3.2).
- Cutoff: **no default**. A value written as `6.0` and meant as 6 Å but interpreted as
  6 bohr is roughly half the intended radius with no error and no symptom.

### 2.2 Damping parameters

BJ parameters are $(a_1, a_2, s_6, s_8)$; PBE-D3(BJ) is
$a_1 = 0.4289$, $a_2 = 4.4407$ bohr, $s_6 = 1$, $s_8 = 0.7875$. $s_6$ is unity for every
common parametrisation but is carried explicitly. $a_2$ is converted once at
configuration construction (§2.1); no conversion is applied at evaluation time.

## 3. The dispersion model

### 3.1 Pair potential

For species $X$ and $Y$ with tabulated $q_X, q_Y$ (dimensionless), define the BJ damping
radius and the pair potential

$$
R_0(X,Y) = a_1\sqrt{3\,q_X q_Y} + a_2, \qquad
\varphi_{XY}(r) = \frac{s_6}{r^6 + R_0^6} + \frac{3\,s_8\,q_X q_Y}{r^8 + R_0^8}.
$$

(P.16) writes $R_0 = \sqrt{3\sqrt{Q_XQ_Y}}$ and $C_8 = 3C_6\sqrt{Q_XQ_Y}$. The two agree
because the tabulated value *is* the square root: $q_Z \equiv \sqrt{Q_Z}$, so
$q_Xq_Y = \sqrt{Q_XQ_Y}$. This identification is normative; the existing real-space D3
code uses the same one (`c8_ratio = 3*pair_r2r4` and
`damping_radius = rs6*sqrt(3*pair_r2r4) + rs8*bohr_to_angstrom`).

$\varphi$ is positive, bounded at $r=0$, and absolutely integrable: that is what makes
both the closed-form transform (§4.4) and the explicit self term (§4.5) possible. Unlike
Coulomb, there is **no $r^{-1}$ singularity**, so there is no splitting parameter and no
"excluded self term" to be neutralised by a background — see §4.5 for what replaces it.

### 3.2 Modified coordination number (model change)

The standard D3 counting function (P.11) tends to a **non-zero constant** as
$r \to \infty$, so truncating it at a neighbour-list radius leaves a discontinuity and the
energy does not converge as the cutoff grows (P.VI.A, Fig. 7). FourierD3 replaces it.

For an ordered pair with separation $r > 0$, table radii $c_i = \text{rcov}(Z_i)$,
$c_j = \text{rcov}(Z_j)$, table sum $C = c_i + c_j$ (which already contains the factor
$\tfrac43$), bare sum $\bar C = \tfrac34 C$, and neighbour-list cutoff $R_{\rm cut}$:

$$
R_{\rm mid} = \tfrac12\bigl(\bar C + R_{\rm cut}\bigr),
\qquad
\sigma(r) =
\begin{cases}
16, & r \le R_{\rm mid},\\[2pt]
16 + \dfrac{(r - R_{\rm mid})^2}{(R_{\rm cut} - r)^2 + \varepsilon}, & r > R_{\rm mid},
\end{cases}
$$

$$
\theta_{ij}(r) = \begin{cases}
\Bigl[1 + \exp\bigl(-\sigma(r)\,\bigl(C/r - 1\bigr)\bigr)\Bigr]^{-1}, & 0 < r < R_{\rm cut},\\[4pt]
0, & r \ge R_{\rm cut},
\end{cases}
\qquad
\theta_i = \sum_{j,\,T}{}' \;\theta_{ij}\bigl(|\mathbf r_{ij} + \mathbf T|\bigr),
$$

with $\varepsilon = 10^{-6}$ in Å² (§3.2.3) and $\sum'$ omitting $j=i,\mathbf T=0$.

**Normative choice — the steepness law.** This is the toolkit v0.5.0 form. P.18 defines
$t_{ij}(r) = 16 + \bigl((r - R_{\rm mid})/(R_{\rm cut} - R_{\rm mid})\bigr)^2$ instead,
whose steepness is bounded by 17 at the cutoff. The two agree exactly for
$r \le R_{\rm mid}$ (both use 16 there) and both cross $\tfrac12$ at $r = C$; they differ
only on $(R_{\rm mid}, R_{\rm cut}]$. Evaluated on $R_{\rm cut} = 6$ Å:

| $\bar C$ (bare sum) | max $\lvert\Delta\theta\rvert$ | max $\lvert\Delta\theta'\rvert$ | $\theta$ at $R_{\rm cut}$: toolkit / P.18 |
|---|---|---|---|
| 0.8534 (H–H) | $1.1\times10^{-6}$ | $1.3\times10^{-6}$ | $0$ / $4.6\times10^{-7}$ |
| 1.125 (H–C) | $7.8\times10^{-6}$ | $9.7\times10^{-6}$ | $0$ / $2.9\times10^{-6}$ |
| 1.425 (C–C) | $2.5\times10^{-5}$ | $3.3\times10^{-5}$ | $0$ / $9.0\times10^{-6}$ |

The paper's literal form therefore leaves a small **jump** at the cutoff (its value just
inside is non-zero, and it is zero just outside), and a corresponding jump in the radial
derivative. The toolkit form is the one that actually delivers the property the
modification exists for — $\theta$ and all its derivatives vanish continuously at the
cutoff — and it is the form the v0.5.0 fixtures encode. **Do not substitute the paper's
literal Eq. (18)**: the difference is small but it is far above round-off, and it would
silently invalidate any fixture comparison tighter than about $10^{-4}$ on a summed
coordination number or on the energy derived from one.

#### 3.2.1 Invariants

- $\theta_{ij} = \tfrac12$ exactly when $r = C$, for any $\sigma$: the argument vanishes
  there. This is the sense in which the modified function "reproduces the standard D3
  behaviour at short range", and it fixes the meaning of the `rcov` table.
- $\theta_{ij} \to 0$ continuously as $r \to R_{\rm cut}^{-}$: the steepness diverges and
  the argument goes to $-\infty$ because $C/r - 1 < 0$ at the cutoff for physical radii.
  In practice $\theta$ underflows to exactly 0 well inside the cutoff (measured: 0 at
  $R_{\rm cut} - 10^{-3}$ Å for a table sum $C = 1.5$ Å, bare sum 1.125 Å).
- The counting function is therefore **independent of the neighbour list** it was summed
  over, provided the list radius equals $R_{\rm cut}$.
- The logistic must be evaluated **branchwise** (e.g. $p = e^{-|x|}$, then
  $1/(1+p)$ or $p/(1+p)$): a plain $1/(1+e^{-x})$ overflows because the argument reaches
  $\pm\infty$.

#### 3.2.2 Radial derivative

With $u = C/r - 1$ and $g = \bigl((r-R_{\rm mid})^2\bigr)/\bigl((R_{\rm cut}-r)^2+\varepsilon\bigr)$
(so $\sigma = 16 + g$):

$$
\frac{\partial\theta_{ij}}{\partial r}
= \theta_{ij}\,(1 - \theta_{ij})\left[\frac{\partial \sigma}{\partial r}\,u
+ \sigma\,\frac{\partial u}{\partial r}\right],
\qquad
\frac{\partial u}{\partial r} = -\frac{C}{r^2},
$$

$$
\frac{\partial g}{\partial r}
= \frac{2\,(r - R_{\rm mid})}{(R_{\rm cut}-r)^2+\varepsilon}
+ \frac{2\,(r - R_{\rm mid})^2 (R_{\rm cut}-r)}{\bigl[(R_{\rm cut}-r)^2+\varepsilon\bigr]^2},
\qquad r > R_{\rm mid},
$$

and $\partial\sigma/\partial r = 0$ for $r \le R_{\rm mid}$. The two terms of
$\partial g/\partial r$ come from the numerator and the denominator respectively; the
second is the one that makes the derivative vanish at the cutoff. For
$r \ge R_{\rm cut}$ the derivative is 0.

#### 3.2.3 The regulariser $\varepsilon$ (design choice)

$\varepsilon$ only prevents a division by zero in the limit $r \to R_{\rm cut}$. It is
**not** unit-converted: whenever $(R_{\rm cut}-r)^2 \lesssim \varepsilon$ the steepness is
already $\gtrsim 10^6$ and $\theta$ is exactly zero in double precision, so the value of
$\varepsilon$ cannot be observed in Å or in bohr. It is recorded here so that the
constant is deliberate; it is not a fitted or physically meaningful quantity.

#### 3.2.4 Neighbour-list requirements (normative)

- The list must contain **both orientations of every pair**. $\theta_i$ is accumulated
  from atom $i$'s own row; the reverse edge is walked by the other atom. A half-filled
  list silently halves every coordination number, shifts the energy by roughly a percent,
  and produces non-conservative forces.
- The list radius must **equal** $R_{\rm cut}$. Only edges with $0 < r < R_{\rm cut}$ are
  counted (strict inequalities).
- Padding atoms (atomic number 0) are skipped.
- These are caller preconditions; they are not checked at runtime (detection costs a
  reduction over the whole list on every step, which measured about a sixth of the
  evaluation at 8,000 atoms, and cannot run under `jax.jit`).

### 3.3 Low-rank separable coefficients

D3's $C_6$ depends on both atoms' coordination numbers in a non-multiplicative way
(P.II.B), which forbids particle-mesh evaluation directly. FourierD3 restores
separability with an **offline** functional tensor decomposition.

#### 3.3.1 Reference weighting (unchanged from D3)

$$
L_i^p = \exp\!\bigl(-k_3\,(\theta_i - \theta^{\rm ref,p}_{Z_i})^2\bigr),
\qquad k_3 = 4,
\qquad
c_{i,\ell}(\theta_i) = \frac{\sum_p v_{\ell,Z_i}(\theta^{\rm ref,p}_{Z_i})\,L_i^p}{\sum_p L_i^p},
$$

so that $C_6^{ij}(\theta_i,\theta_j) = \sum_\ell \lambda_\ell\,c_{i,\ell}c_{j,\ell}$
(P.13, P.22). Reference slots a species does not use are marked with a sentinel
$\theta^{\rm ref} = -1$ and **must be masked**, either by the validity mask in the
coefficient loop or by the zeroed rows §3.3.2 guarantees. The sentinel does **not**
underflow: at a typical $\theta_i \approx 0$ its weight is
$\exp(-k_3(0+1)^2) = e^{-4} \approx 0.018$, an ordinary number. An unmasked slot
therefore inflates the denominator $S = \sum_p w_p$ of §3.3.5 by a few percent (and its
derivative $S'$ by a term proportional to $\theta_i + 1$), rescaling every retained
coefficient of that species. Masking is a correctness requirement, not a numerical
nicety.

#### 3.3.2 Block matrix and eigendecomposition (P.20–21)

For the species set $\mathcal Z$ present, with $n_{\rm ref}$ reference slots per species,
form the symmetric block matrix $M$ of size $(|\mathcal Z|\,n_{\rm ref})^2$ whose
$(\alpha,p),(\beta,q)$ entry is $C^{6,\rm ref}_{Z_\alpha Z_\beta}(\theta^{\rm ref,p}_{Z_\alpha},\theta^{\rm ref,q}_{Z_\beta})$,
zeroed where either slot is invalid. Symmetrise exactly
($M \leftarrow \tfrac12(M + M^{\mathsf T})$) before the decomposition: the shipped tables
are symmetric only up to float32 storage noise, and `eigh` requires an exactly symmetric
operand.

Eigendecompose $M = \sum_\ell \lambda_\ell\,v_\ell v_\ell^{\mathsf T}$ and retain the
first $\ell_{\max}$ terms.

#### 3.3.3 Truncation criterion (normative)

The decomposition error is defined **only over populated entries**:

$$
\epsilon(r) = \max_{\substack{(\alpha,p),(\beta,q)\\ M \neq 0}}
\frac{\bigl|\,[M_r]_{(\alpha,p),(\beta,q)} - M_{(\alpha,p),(\beta,q)}\,\bigr|}{\bigl|M_{(\alpha,p),(\beta,q)}\bigr|},
\qquad [M_r] = V_{:,1:r}\,\mathrm{diag}(\lambda_{1:r})\,V_{:,1:r}^{\mathsf T},
$$

and the retained rank is the **smallest** $r$ with $\epsilon(r) \le \tau$, where the
default tolerance is $\tau = 10^{-4}$ (0.01 %, P.III.A) and an optional hard cap
$\ell_{\max}$ applies. The achieved $\epsilon$ must be reported in diagnostics even when
the cap prevents $\tau$ from being met.

Two normative implementation rules that the paper leaves open:

- **Order by magnitude, not value.** `eigh` returns ascending eigenvalues; reorder by
  $|\lambda_\ell|$ descending before truncating. The energy is a *signed* sum
  (§4.2), so $\lambda$ may be negative and the ordering is by magnitude.
- **Scan, do not bisect.** The error is not monotone in $r$ (adding a term can cancel
  against earlier ones; the toolkit measures the error rising at 44 of 180 steps for
  $Z \le 36$), so a bisection can step over a smaller rank that already meets $\tau$.

#### 3.3.4 Sign gauge (normative; a Symmetrix requirement)

$E$ is invariant under $v_\ell \to -v_\ell$, so the eigenvector signs are a pure gauge.
Raw LAPACK output is not portable across libraries, so for a deterministic cache key,
rank, and hash, each retained eigenvector must be **sign-normalised**: multiply it by
$-1$ if necessary so that the entry of largest magnitude is positive, breaking ties by
the smallest flat index. This is a Symmetrix design decision, not paper behaviour; the
toolkit inherits whatever its LAPACK returns.

#### 3.3.5 Coefficients and their coordination derivative

Per atom, using the same Gaussian weights, in the numerically stable form (subtract the
largest exponent before exponentiating):

$$
c_{i,\ell} = \frac{N_\ell}{S}, \qquad
\frac{\partial c_{i,\ell}}{\partial\theta_i}
= \frac{N'_\ell - c_{i,\ell}\,S'}{S},
$$

$$
N_\ell = \sum_p w_p\,v_{\ell,Z_i}^p,\quad
N'_\ell = \sum_p (-2k_3\,\delta_p)\,w_p\,v_{\ell,Z_i}^p,\quad
S = \sum_p w_p,\quad S' = \sum_p (-2k_3\,\delta_p)\,w_p,\quad
w_p = e^{-k_3\delta_p^2},\quad \delta_p = \theta_i - \theta^{\rm ref,p}_{Z_i}.
$$

An uncovered species (channel index $-1$) yields $c = \partial c/\partial\theta = 0$ and
is dropped from the dispersion sum, while still contributing to its neighbours'
coordination numbers. Coverage is a caller precondition and is **not** checked per step;
the energy comes back finite and wrong if it is violated.

### 3.4 Damping and the coefficient the model needs

The model needs $\varphi$ of §3.1 evaluated for a *pair of species*. Since the rank
decomposition carries species through $c_{i,\ell}$, the reciprocal kernel is evaluated per
species pair $(X,Y)$ (§4.4), and the per-atom channel is $X = Z_i$.

## 4. Reciprocal-space formulation

### 4.1 Lattice and reciprocal lattice (normative convention)

The cell matrix $C$ has the lattice vectors as **rows**, $\Omega = |\det C|$. Positions
are Cartesian row vectors $\mathbf r$. Fractional coordinates are
$\mathbf s = C^{-\mathsf T}\mathbf r$ (equivalently $\mathbf s = \mathbf r C^{-1}$ as a row
vector), so $\mathbf r = \sum_a s_a \mathbf a_a$ and the transformation matrix from
Cartesian to fractional is $C^{-\mathsf T}$ — the "transpose of the inverse cell".

Reciprocal lattice vectors $\mathbf b_i$ satisfy $\mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij}$;
they are the rows of $2\pi\,(C^{-1})^{\mathsf T}$. For integer Miller indices
$\mathbf m \in \mathbb Z^3$ the wave vector is

$$
\boxed{\ \mathbf k = 2\pi\,C^{-1}\mathbf m\ }
\qquad\text{i.e.}\qquad
k_i = 2\pi \sum_j (C^{-1})_{ij} m_j ,
$$

so $k_{\rm matrix} = 2\pi C^{-1}$ and $\mathbf k = k_{\rm matrix}\,\mathbf m$. The phase
factor is $\exp(i\mathbf k\cdot\mathbf r) = \exp(2\pi i\,\mathbf m\cdot\mathbf s)$: the
phase is exactly the fractional-coordinate dot product, which is why the mesh works in
fractional coordinates and integer Miller indices only.

The reciprocal-space cutoff of a mesh $N_x \times N_y \times N_z$ in a cell of side
lengths $L_a$ is $k_{\max,a} = \pi N_a / L_a$ (the Nyquist value); equivalently
$N_a = \lceil k_{\rm cut} L_a / \pi\rceil$ to reach a target $k_{\rm cut}$.

### 4.2 Energy

The Poisson-summation form (P.3, P.4, P.26) is normative:

$$
E_{\rm D3} = V_{\rm self}
- \frac{1}{2\Omega}\sum_\ell \lambda_\ell \sum_{X\in\mathcal Z}\sum_{Y\in\mathcal Z}
  \sum_{\mathbf k \in \Lambda'} \widehat\varphi_{XY}(|\mathbf k|)\,
  S_X^\ell(\mathbf k)\,S_Y^\ell(-\mathbf k),
\qquad
S_X^\ell(\mathbf k) = \sum_{Z_i = X} c_{i,\ell}\,e^{i\mathbf k\cdot\mathbf r_i}.
$$

Equivalently, and this is the form an implementation should compute:

$$
\boxed{\
E_{\rm D3} = V_{\rm self} - \frac{1}{2\Omega}\sum_{\mathbf k}w(\mathbf k)\sum_\ell\lambda_\ell
\sum_{X,Y}\widehat\varphi_{XY}(|\mathbf k|)\;
\mathrm{Re}\!\left[S_X^\ell(\mathbf k)\,\overline{S_Y^\ell(\mathbf k)}\right]\ }
$$

with $w$ the Hermitian multiplicity of §5.3. Because $\widehat\varphi$ is real and even
and the structure factors come from a real field, the $\pm\mathbf k$ pair contributes
twice the real part, and $\mathrm{Re}[S_X\overline{S_Y}]$ is symmetric under $X
\leftrightarrow Y$.

Normative details:

- The $\sum_{X,Y}$ is over **ordered** species pairs. For $X \ne Y$ both $(X,Y)$ and
  $(Y,X)$ are present; the toolkit expresses this as an upper-triangular loop with
  multiplicity 2 and multiplicity 1 on the diagonal. For $X=Y$ there is exactly one
  term.
- The $\mathbf k$ sum runs over the **whole** reciprocal lattice represented by the mesh,
  including $\mathbf k = \mathbf 0$ (§5.5). Nothing is excluded.
- There is **no** Ewald $\alpha$, no real-space dispersion term, and no exclusion of any
  $k$: this is not a split method.
- The primitive energy convention is that the pair sum is over **all ordered pairs
  including $i=j,\mathbf T=0$**, with that term cancelled by $V_{\rm self}$ (§4.5). An
  unprimed sum without the self term would sit a coordination-number-dependent offset
  away and no mesh refinement would close the gap. The $(i=j,\mathbf T\neq\mathbf 0)$
  self-image terms are **retained**, not cancelled — see §4.5.

### 4.3 Structure factors

$S_X^\ell(\mathbf k)$ is the Fourier transform of the coefficient field of species $X$
in rank slot $\ell$. In the mesh evaluation it is obtained by B-spline assignment
(§5.1) and deconvolution (§5.2), not by a direct sum. In the *direct reciprocal oracle*
(independent oracle) it is the explicit sum above.

### 4.4 The BJ Fourier transform (normative closed form)

Define the transform with the convention

$$
\widehat\varphi(\mathbf k) = \int_{\mathbb R^3}\varphi(\mathbf r)\,e^{-i\mathbf k\cdot\mathbf r}\,d^3r,
$$

which is real, even, and exponentially decaying. Then with $x = |\mathbf k| R_0$:

$$
\widehat\varphi_{XY}(|\mathbf k|)
= s_6\,\frac{2\pi^2}{3R_0^3}\,G_6(x)
+ 3\,s_8\,q_Xq_Y\,\left(-\frac{\pi^2}{R_0^5}\right)G_8(x),
$$

$$
G_6(x) = \frac{e^{-x} - 2\,e^{-x/2}\cos\!\bigl(\tfrac{\pi}{3} + \tfrac{\sqrt3}{2}x\bigr)}{x},
$$

$$
G_8(x) = \frac{e^{-x\sin(\pi/8)}\cos\!\bigl(\tfrac{\pi}{4} + x\cos(\pi/8)\bigr)
+ e^{-x\cos(\pi/8)}\cos\!\bigl(\tfrac{3\pi}{4} + x\sin(\pi/8)\bigr)}{x}.
$$

Equivalently $G_6 = N_6(x)/x$ and $G_8 = N_8(x)/x$ with $N_6, N_8$ entire and vanishing
at the origin, which is why both must be evaluated from a **series** for small $x$
(otherwise two cancelling quantities are divided and all precision is lost):

$$
G_6(x) \approx 1 - \frac{x^2}{3} + \frac{x^3}{8} - \frac{x^4}{60} + \frac{x^6}{5040},
\qquad x < 0.1,
$$

$$
G_8(x) \approx -0.54119610014619668 + x^2\left(0.090199350024366076
+ x^2\left(-0.010888024707303151 + x\left(\tfrac{1}{360}\right) - 0.00025923868350721731\,x^2\right)\right),
\qquad x < 0.1 .
$$

The crossover $x = 0.1$ and the coefficients are normative: they are what makes the
$k=0$ bin (§5.5) and the low-$k$ bins accurate. $N_8(0) = 0$ and
$G_8(0) = -\sqrt2\sin(\pi/8) = -0.5411961\ldots$; note $G_8(0) \neq 0$ even though
$N_8(0) = 0$.

The kernel is an **exponentially damped oscillation**: it changes sign, with the first
zero near $x \approx 4.3$. Only its envelope decays monotonically. It must not be
treated as positive, and it must not be truncated by a positivity assumption.

**The radial derivative is also required** (for the virial):

$$
\frac{\partial\widehat\varphi}{\partial|\mathbf k|}
= R_0\left[s_6\frac{2\pi^2}{3R_0^3}G_6'(x) - 3s_8q_Xq_Y\frac{\pi^2}{R_0^5}G_8'(x)\right],
\qquad x = |\mathbf k|R_0,
$$

with $G'(x) = N'(x)/x - N(x)/x^2$ and, on the series branch, the term-by-term derivative
of the polynomials above. The derivative cannot be recovered after $|\mathbf k|$ leaves
scope, so it must be produced by the same kernel evaluation.

### 4.5 Self term and retained self-image terms

The reciprocal sum of §4.2 includes the $(i=j,\mathbf T=0)$ term at zero separation, which
no physical pair sum contains. BJ damping keeps that term **finite** ($\varphi(0) < \infty$),
so it is cancelled exactly (P.25):

$$
\boxed{\
V_{\rm self} = \frac12\sum_i\sum_\ell \lambda_\ell\,c_{i,\ell}^2\,
\varphi_{Z_iZ_i}(0),
\qquad
\varphi_{XX}(0) = \frac{s_6}{R_0^6} + \frac{3\,s_8\,q_X^2}{R_0^8},
\quad R_0 = a_1\sqrt{3q_X^2} + a_2\ }
$$

This is **added** to the reciprocal energy. It is a per-atom quantity, not a lattice
correction: there is no background/neutralising charge, and no "tin-foil" analogue.

**Only the $(i=j,\mathbf T=\mathbf 0)$ term is removed.** Every $(i=j,\mathbf T\neq\mathbf 0)$
term — an atom interacting with its own periodic images — is **retained** by the
reciprocal sum of §4.2 and is part of the model:

$$
E_{\rm self-image}
= -\frac12\sum_i\sum_{\mathbf T\neq\mathbf 0}\sum_\ell \lambda_\ell\,c_{i,\ell}^2\,
\varphi_{Z_iZ_i}(|\mathbf T|),
\qquad
E_{\rm D3} = E_{\rm recip} + V_{\rm self},
$$

with $E_{\rm recip}$ the full reciprocal sum. The paper has the same property (its
$\mathbf k$ sum in Eq. 26 is unrestricted, and its $V_{\rm self}$ is the $r\to0^+$ limit
of a single pair term), and the v0.5.0 NumPy oracle makes it explicit: its only exclusion
is the diagonal at zero translation. Three consequences:

- **An isolated atom in a periodic cell does not have zero dispersion energy.** For a
  single atom the reciprocal sum evaluates to $\varphi(0) + \sum_{\mathbf T\neq\mathbf 0}
  \varphi(|\mathbf T|)$, so
  $E = -\tfrac12\sum_{\mathbf T\neq\mathbf 0}\sum_\ell\lambda_\ell c^2\varphi(|\mathbf T|)$,
  which is negative and depends on the cell. That is the correct behaviour of the periodic
  lattice sum, not an artefact, and it is what makes the result comparable with a converged
  real-space lattice sum.
- **Do not subtract the full lattice self-sum.** Only $\varphi(0)$ is cancelled.
  Subtracting a single-atom lattice sum, a tabulated "self-energy", or any
  $\mathbf T\neq\mathbf 0$ term double-counts the images and biases the energy by a
  cell-dependent amount.
- **The energy is not invariant under a cell change that leaves the atomic configuration
  alone**, such as padding a fully periodic cell with vacuum: the image vectors $\mathbf T$
  change, so the retained terms change. A vacuum-padded cell is therefore still a fully
  interacting three-dimensional periodic system, and the cell must be recorded with any
  reported energy.

Two further consequences that are easy to get wrong:

- The self term is **quadratic in the coefficients**, and the coefficients depend on the
  coordination numbers, so it contributes to $\partial E/\partial c_{i,\ell}$ and thence
  to the forces through the coordination chain rule. It must be folded in **before** the
  chain rule runs; applying it afterwards as a scalar correction silently drops that
  force contribution.
- $V_{\rm self}$ is *not* part of the mesh quadratic form, so it has its own per-atom
  share (§6.4).

## 5. Particle-mesh evaluation

### 5.1 Mesh, coordinates, and B-spline assignment

Mesh $\mathbf N = (N_x,N_y,N_z)$, $n = N_xN_yN_z$ points, node $\mathbf g$ corresponding to
fractional coordinate $s_a = g_a/N_a$, i.e. Cartesian $\mathbf r_{\mathbf g} = \sum_a (g_a/N_a)\mathbf a_a$.

For atom $i$ with fractional coordinate $\mathbf s_i$:

$$
\mathbf y = \mathbf N \circ \mathbf s_i, \qquad
\mathbf{base} = \lfloor \mathbf y \rfloor, \qquad
\boldsymbol\theta = \mathbf y - \mathbf{base} \in [0,1)^3,
$$

$$
\text{offset start} = \bigl\lfloor \theta_a - \tfrac{p-2}{2}\bigr\rfloor, \qquad
\text{offset} = (i_1,i_2,i_3) + \text{offset start}, \quad i_a \in \{0,\dots,p-1\},
$$

$$
u_a = \frac{p}{2} + \theta_a - \text{offset}_a \in [0,p),
\qquad
W_i(\mathbf g) = \prod_a M_p(u_a), \qquad
\mathbf g = (\mathbf{base} + \text{offset}) \bmod \mathbf N .
$$

$M_p$ is the cardinal B-spline of order $p$ (degree $p-1$), supported on $[0,p)$ with
$\int M_p = 1$ and $\sum_{\mathbf g}W_i(\mathbf g) = 1$ exactly. $p=2$ linear, $p=3$
quadratic, $p=4$ cubic, $p=5$ quartic, $p=6$ quintic. $M_p$ has $C^{p-2}$ continuity;
its knots fall at integers, and the stencil is the $p^3$ cube around the atom. The index
wrap is periodic and must handle negative indices ($((i \bmod N)+N)\bmod N$).

The assigned field for rank slot $\ell$ and species channel $X$ is

$$
m_{X,\ell}(\mathbf g) = \sum_{i:\,Z_i=X} c_{i,\ell}\,W_i(\mathbf g).
$$

**Normative:** every stencil point is kept, with **no weight threshold**. A threshold
saves a few atomic adds but removes both the value and its gradient, so the force stops
being the gradient of the energy that was computed; for an atom a thousandth of a cell
from a boundary a quarter of the stencil can fall below any useful threshold, and the
resulting force error survives mesh refinement. The spread and the gather must use
exactly the same stencil set.

### 5.2 Deconvolution

Assignment attenuates each frequency. Dividing the attenuation out is what recovers the
structure factor the mesh stands in for:

$$
M_{\rm mod}(\mathbf m) = \prod_{a} \left|\operatorname{sinc}\!\left(\frac{m_a}{N_a}\right)\right|^{p},
\qquad
\operatorname{sinc}(x) = \frac{\sin \pi x}{\pi x},\quad \operatorname{sinc}(0)=1,
$$

$$
A_{X,\ell}(\mathbf m) = \frac{\widehat m_{X,\ell}(\mathbf m)}{M_{\rm mod}(\mathbf m)},
\qquad
M_{\rm mod} \leftarrow \max(M_{\rm mod}, 10^{-10}) \ \text{for the division.}
$$

Normative: use the **continuous** cardinal B-spline symbol per axis
($|\operatorname{sinc}(m_a/N_a)|^p$), the same convention as the electrostatics PME path
(`compute_bspline_moduli_1d`). v1 uses the continuous symbol **because that is what NVIDIA
v0.5.0 uses**, and numerical acceptance requires agreement with v0.5.0 fixtures; the
choice is a convention decision for fixture agreement, not a claim that it is the more
accurate of the two. Oracle validation may revise this line — and the diagnostic that reports which
symbol was used — before any kernel is written, if the discrete symbol proves materially
better (§5.2 open item below). The modulus is applied **once** per bin — spread and
gather both carry the interpolation, but the energy is formed from the deconvolved
structure factors, and the cotangent of §6.2 re-applies the inverse modulus so that the
adjoint is exact with respect to the *assigned* mesh.

```{important}
**Open validation item (requiring oracle and particle-mesh validation)** The
*discrete* symbol of the shipped assignment is
$D_p(m,N) = \sum_{t=0}^{p-1}M_p(t)\,e^{2\pi i m t/N}$, which is what the assignment
actually produces; $|\operatorname{sinc}(m/N)|^p$ is its continuous analogue. A 1D probe
of the identity
$\widehat m(m) = e^{-2\pi i m f}e^{-i\pi m p/N}D_p(m,N)$ matched to only
$\sim10^{-4}$–$10^{-2}$ for $N \in \{8,12\}$, and a full 3D mesh probe showed a residual
amplitude deficit of $\sim4.5\times10^{-4}$ at the lowest non-zero frequency for a 1 Å
spacing, shrinking with refinement. Energy agreement with the v0.5.0 oracle was still
$5\times10^{-9}$ relative on the checked case, because $|\widehat\varphi|$ decays
exponentially and the deficit is largest where the kernel is negligible. **The independent oracle must
pin the exact per-axis symbol of the implementation's own assignment numerically, and
Particle-mesh validation must demonstrate that the chosen convention's residual is below the
declared tolerance.** Do not assume either convention without that measurement.
```

### 5.3 Discrete energy and Hermitian multiplicity

The forward transform is a real-to-complex transform of a real mesh, so only half the
spectrum is stored. For a full-spectrum sum, each stored bin except the self-conjugate
ones stands in for two:

$$
w(\mathbf m) = \begin{cases}
1, & m_z = 0 \ \text{or}\ (N_z \text{ even and } 2m_z = N_z),\\
2, & \text{otherwise},
\end{cases}
$$

applied along the halved axis only. The energy sum carries this multiplicity; the
cotangent of §6.2 **does not** — the inverse transform reconstructs the unstored half
itself, so weighting it as well would double-count every interior bin.

The discrete energy is then

$$
E_{\rm recip} = -\frac{1}{2\Omega}\sum_{\mathbf m} w(\mathbf m)
\sum_\ell \lambda_\ell \sum_{X\le Y} \mu_{XY}\,
\widehat\varphi_{XY}(|\mathbf k_{\mathbf m}|)\,
\mathrm{Re}\!\left[A_{X,\ell}(\mathbf m)\overline{A_{Y,\ell}(\mathbf m)}\right],
$$

with $\mu_{XY} = 1$ if $X=Y$ and $2$ otherwise (the ordered-pair multiplicity of §4.2),
and $E_{\rm D3} = E_{\rm recip} + V_{\rm self}$.

### 5.4 FFT normalization (normative)

The pair of conventions is normative and must be stated in the implementation, because a
mismatch silently breaks the adjoint relation of §6.2:

- **Forward (unnormalized):** $\widehat m(\mathbf m) = \sum_{\mathbf g} m(\mathbf g)\exp(-2\pi i\,\mathbf m\cdot\mathbf g/\mathbf N)$.
  No $1/n$.
- **Inverse (unnormalized):** the Hermitian-completed inverse,
  $P(\mathbf g) = \sum_{\mathbf m} C(\mathbf m)\exp(+2\pi i\,\mathbf m\cdot\mathbf g/\mathbf N)$,
  with **no** $1/n$. (For a real-to-complex library this is `norm="backward"` forward and
  `norm="forward"` inverse, or the equivalent unscaled pair; an orthonormal or
  $1/n$-normalized pair requires compensating factors everywhere.)

With this pair, and with the cotangent defined in §6.2, the potential read back by the
inverse transform **is** $\partial E/\partial m(\mathbf g)$ exactly — that is the
definition of the cotangent, and it is what makes the gathered force the analytic
gradient of the computed energy. The unscaled forward transform also means the mesh
values, not their normalized transform, are the field the chain rule differentiates.

### 5.5 The $k=0$ bin (normative)

**$\mathbf k = \mathbf 0$ is included.** It is a genuine term of the lattice sum: by
Poisson summation $\sum_{\mathbf T}\varphi(|\mathbf r_{ij}+\mathbf T|)
= \frac1\Omega\sum_{\mathbf k}\widehat\varphi(|\mathbf k|)e^{i\mathbf k\cdot\mathbf r_{ij}}$,
whose $\mathbf k=0$ term is $\widehat\varphi(0)/\Omega \neq 0$. The toolkit computes it
through the same kernel, which is exactly why the small-$x$ series of §4.4 must exist:

$$
\widehat\varphi_{XY}(0) = s_6\frac{2\pi^2}{3R_0^3} - 3s_8q_Xq_Y\frac{\pi^2}{R_0^5}
\bigl(-\sqrt2\sin(\pi/8)\bigr) \neq 0 .
$$

Two things must **not** be carried over from Coulomb Ewald/PME:

- the "tin-foil" convention $G(0) = 0$ (which removes the $k=0$ term by boundary
  condition), and
- any charge-neutrality assumption. There is no background term; the model is not
  neutral by construction, and $A(0) = \sum_i c_{i,\ell}$ is generally non-zero.

For the virial, the $k=0$ bin contributes only the isotropic volume term: its
anisotropic term is zero because $\mathbf k = \mathbf 0$ (guard the $1/|\mathbf k|$).

### 5.6 Mesh selection (design choices)

- Exactly one of `mesh_dimensions` and `mesh_spacing` must be given (unless a prepared
  setup supplies the mesh). There is no accuracy-based default to fall back on, so
  neither and both are errors.
- A mesh derived from a spacing is rounded **up** per axis to the smallest size whose
  prime factors are 2, 3, 5 and 7 (cuFFT/FFTW radix kernels; otherwise Bluestein), and is
  sized from the **largest** cell in a batch. An explicit mesh is used exactly as given,
  including a poor size.
- **Every axis must hold at least $p$ nodes.** The stencil is $p$ wide and wraps
  periodically; a shorter axis makes two stencil points land on the same node and the
  interpolation stops being the B-spline the gather differentiates. Equality is allowed.
- Suggested sizes from P.III.C (a starting point, not a convergence claim):

| system size $N$ | mesh |
|---|---|
| $N \le 200$ | $16^3$ |
| $200 < N \le 2000$ | $32^3$ |
| $2000 < N \le 20000$ | $64^3$ |
| $N > 20000$ | $128^3$ |

- Spline order default 4, allowed 2–6. Accuracy at a fixed mesh improves sharply with
  order (measured against a converged reference: $2.8\times10^{-5}$, $2.5\times10^{-7}$,
  $2.0\times10^{-9}$, $1.4\times10^{-10}$ for $p = 2,3,4,5$ on an 8-atom cell at $96^3$),
  and raising $p$ is usually cheaper than refining the mesh since spread/gather cost
  $p^3$ per atom while the transform cost grows with the mesh volume. Order 3 is
  noticeably noisier than its neighbours; prefer an even order unless measured otherwise.
- **Two independent error sources** must be distinguished in diagnostics and in
  convergence claims: (i) truncation of the reciprocal sum, which is exponentially small
  because $|\widehat\varphi|$ decays exponentially (measured: the mesh-free reciprocal sum
  is converged to $10^{-11}$ relative by $k_{\max}=10$ Å⁻¹ on a small test cell); and
  (ii) interpolation/aliasing, which is what actually sets the mesh requirement and
  decays only algebraically in $N$. P.III.B reports the same qualitative picture
  ("converges algebraically to the Ewald reference"). Do not attribute convergence to
  the exponential decay of the kernel alone.
- Rank chunking (processing a subset of rank slots through spread → transform →
  contraction → gather and summing) is numerically neutral: every reciprocal stage is a
  sum over slots with no coupling between them. It changes only peak memory. The
  self-energy and the coordination chain rule need all slots at once and must run once,
  outside the chunk loop.

## 6. Derivatives

All derivatives below are **analytic** outputs; the energy is not required to be
differentiable by an autodiff framework. Forces and virial are derivatives of the same
$E_{\rm D3}$ that is reported.

### 6.1 Coefficient derivatives

$$
\frac{\partial E}{\partial c_{i,\ell}}
= \underbrace{\sum_{\mathbf g} W_i(\mathbf g)\,P_\ell(\mathbf g)}_{\text{mesh, gathered}}
+ \underbrace{\lambda_\ell\,c_{i,\ell}\,\varphi_{Z_iZ_i}(0)}_{\text{self}},
$$

where $P_\ell(\mathbf g) = \partial E/\partial m_\ell(\mathbf g)$ is the potential of
§5.4 (the inverse transform of the cotangent), and $W_i$ is the same stencil weight as
the spread. The self contribution must be included before §6.2's chain rule.

### 6.2 Forces (normative)

$$
\mathbf F_i = -\frac{\partial E}{\partial \mathbf r_i}
= \mathbf F_i^{\rm mesh} + \mathbf F_i^{\rm coord},
$$

**Direct mesh force.** The mesh depends on the position only through the interpolation
weights (the fractional coordinate is what enters the assignment, and the stencil indices
are piecewise constant, with the weights vanishing continuously at stencil boundaries):

$$
\boxed{\
\mathbf F_i^{\rm mesh}
= -\sum_\ell c_{i,\ell}\sum_{\mathbf g} P_\ell(\mathbf g)\,
\frac{\partial W_i(\mathbf g)}{\partial \mathbf r_i}\ }
$$

$$
\frac{\partial W_i(\mathbf g)}{\partial \mathbf r_i}
= C^{-1}\Bigl(\mathbf N \circ \nabla_{\boldsymbol\theta}W_i(\mathbf g)\Bigr),
\qquad
\bigl(\nabla_{\boldsymbol\theta}W\bigr)_a = \frac{\partial M_p(u_a)}{\partial u_a}
\prod_{b \ne a}M_p(u_b),
$$

with $\partial u_a/\partial\theta_a = +1$ and $\partial\theta_a/\partial s_a = N_a$. In the
boxed expression the vectors are **columns**: the fractional gradient is mapped to the
Cartesian one by $C^{-1}$, because with column vectors $\mathbf s = C^{-\mathsf T}\mathbf r$
and therefore $\nabla_{\mathbf r}W = C^{-1}\nabla_{\mathbf s}W$. Equivalently, in the
row-vector convention of §4.1 ($\mathbf s = \mathbf r C^{-1}$ as a row vector),
$dW/d\mathbf r = (dW/d\mathbf s)\,C^{-\mathsf T}$. These are the same statement written in
the two conventions: the column form left-multiplies the fractional gradient by $C^{-1}$,
the row form right-multiplies it by $C^{-\mathsf T}$, and only one of the two may be used
at a time. The toolkit kernel computes the column form
(`transpose(cell_inv_t) * gradient` with `cell_inv_t = C^{-\mathsf T}`, i.e. $C^{-1}$).
A cubic cell cannot detect a confusion between the two: $C^{-1}$ and $C^{-\mathsf T}$
coincide for a diagonal cell, so this must be tested on a triclinic cell.

**Coordination chain rule.** Define the per-atom coordination sensitivity (the toolkit's
`d_energy_d_cn`; written $\chi_i$ here to avoid colliding with the steepness $\sigma(r)$
of §3.2, the stress $\sigma_{ab}$ of §6.3, and the fractional coordinate $\mathbf s_i$ of
§5.1):

$$
\chi_i = \frac{\partial E}{\partial\theta_i}
= \sum_\ell \frac{\partial E}{\partial c_{i,\ell}}\,
\frac{\partial c_{i,\ell}}{\partial\theta_i} .
$$

For a directed edge from receiver $i$ to source $j$ with
$\boldsymbol\delta = \mathbf r_j + \mathbf T C - \mathbf r_i$, $r = |\boldsymbol\delta|$:

$$
\boxed{\
\mathbf F_i^{\rm coord} \mathrel{+}= \frac{(\chi_i + \chi_j)\,\theta'_{ij}(r)}{r}\,\boldsymbol\delta
\qquad\text{accumulated on the \emph{receiver} only.}\ }
$$

The reverse edge, owned by atom $j$, supplies $-\boldsymbol\delta$ and therefore the
opposite force; **accumulating on both endpoints of a directed edge double counts** the
coordination force by a factor of two. Both orientations must be present in the list, but
each directed edge contributes only to its receiver. This construction makes
$\sum_i \mathbf F_i^{\rm coord} = 0$ exactly, to reduction round-off: it is a sum of
internal pair forces and telescopes.

**The total force sum is not zero (normative).** The mesh force is not translation
invariant, so

$$
\sum_i \mathbf F_i = \sum_i \mathbf F_i^{\rm mesh}
= -\frac{dE}{dt}\Big|_{\text{rigid translation, fixed cell}} \ne 0 .
$$

Partition of unity does not force this to vanish. The assignment is a *compact-support*
interpolation of each atom onto $p$ nodes per axis, and the potential $P_\ell(\mathbf g)$
that multiplies $\partial W_i(\mathbf g)/\partial\mathbf r_i$ in the boxed mesh-force
expression varies from node to node. A rigid translation moves every atom's stencil
together, but it does not move the potential, so the weighted sum of the stencil
derivatives is not the derivative of a constant: the assignment conserves the assigned
charge, not the translational derivative of the energy. Written out,

$$
\sum_i \mathbf F_i
= -\sum_\ell \sum_{\mathbf g} P_\ell(\mathbf g)\,
\frac{\partial}{\partial t}\Bigl(\sum_i c_{i,\ell} W_i(\mathbf g)\Bigr),
$$

and the inner derivative is the change of one bin's assigned charge under a rigid
translation of every atom - non-zero for a B-spline assignment of compact support.

The residual is a property of the discretization, not an error in the force. It vanishes
under refinement, and the analytic mesh force remains the **exact** gradient of the
reported discrete energy $E$ at fixed cell, which is what the finite-difference check
below establishes.

**What an implementation must assert (normative).** The literal obligation
$\sum_i \mathbf F_i = 0$ is replaced by these four checks:

1. $\sum_i \mathbf F_i^{\rm coord} = 0$ to reduction round-off, exactly;
2. $\sum_i \mathbf F_i = \sum_i \mathbf F_i^{\rm mesh}$ to the rounding of the elementwise
   sum, so the coordination adjoint contributes nothing to the total;
3. $\sum_i \mathbf F_i = -dE/dt$ for a rigid translation of every atom at a fixed cell,
   within the same-mesh central-difference band of §9;
4. an absolute bound on $\max_a \bigl|\sum_i F_{i,a}\bigr|$ derived from a **recorded**
   mesh-refinement sequence at the fixture's order and precision.

Item 4 is a recorded measurement, not a formula. No universal $C h^p$ law is asserted:
the meshes a suite evaluates on are not a nested sequence, the assignment error is not
dominated by a single power of $h$ over the whole range, and the observed behaviour is
approximately $h^5$ for the order-4 fixture alone. A measured sequence is evidence; a
fitted exponent presented as a universal law would be a claim the evidence does not carry.

### 6.3 Virial and ASE stress

**Deformation convention (normative).** Deformed coordinates and cells are

$$
\mathbf r' = \mathbf r\,(I + u), \qquad C' = C\,(I + u),
$$

with $u$ not symmetrised (only its symmetric part affects the energy). The virial is

$$
W_{ab} = -\frac{\partial E}{\partial u_{ab}} ,
$$

which is the repository-wide convention (`docs/userguide/about/conventions.md` of the
toolkit and the Symmetrix D3 kernels). **Symmetrix reports**

$$
\boxed{\ \sigma_{ab} = \frac{1}{\Omega}\frac{\partial E}{\partial u_{ab}} = -\frac{W_{ab}}{\Omega}\ }
$$

in eV/Å³, tensile-positive, in the **Voigt order $(xx, yy, zz, yz, xz, xy)$** — the six
independent components of the symmetric tensor, with the off-diagonal entries *not*
doubled (`full_3x3_to_voigt_6_stress` order, `docs/user/dispersion_corrections.md:44`).
The existing real-space D3 path uses the same internal convention
(`stress(component) = virial(component)*inverse_volume` in `StressReductionKernel`), so
both corrections agree without any conversion at the boundary.

**Contributions.** Under this deformation the fractional coordinates are *invariant*
($\mathbf s' = \mathbf r(I+u)(C(I+u))^{-1} = \mathbf s$), so the mesh, the stencil, and
every interpolation weight are strain-invariant. There is therefore **no interpolation
contribution to the virial** — a fact that must not be "fixed" by adding one. The
reciprocal-space virial exists only in the contraction, where $|\mathbf k|$ and $\Omega$
depend on the strain:

$$
\frac{\partial E_{\rm recip}}{\partial u_{ab}}
= \underbrace{-E_{\rm recip}\,\delta_{ab}}_{\Omega^{-1}\ \text{term}}
+ \underbrace{\frac{1}{2\Omega}\sum_{\mathbf m} w(\mathbf m)\sum_\ell\lambda_\ell
\sum_{X,Y}\mu_{XY}\,\widehat\varphi'_{XY}(|\mathbf k|)\,
\frac{k_a k_b}{|\mathbf k|}\,
\mathrm{Re}\!\left[A_X\overline{A_Y}\right]}_{\text{reciprocal-lattice term}},
$$

using $\partial(1/\Omega)/\partial u_{ab} = -\delta_{ab}/\Omega$,
$\partial k_c/\partial u_{ab} = -\delta_{ac}k_b$, hence
$\partial|\mathbf k|/\partial u_{ab} = -k_ak_b/|\mathbf k|$, and
$\widehat\varphi'(|\mathbf k|) = d\widehat\varphi/d|\mathbf k|$ from §4.4. The
$k=0$ bin contributes only the isotropic term.

The coordination part follows from the same pair decomposition used for forces:

$$
\frac{\partial E}{\partial u_{ab}}
\mathrel{+}= \frac12 \sum_{\text{directed edges}} \bigl(\mathbf F^{\rm coord}_{\rm rec}\bigr)_a \boldsymbol\delta_b ,
$$

where the factor $\tfrac12$ accounts for both orientations contributing equally
($-\mathbf F \otimes -\boldsymbol\delta = \mathbf F \otimes \boldsymbol\delta$). Both
accumulations are symmetric in $(a,b)$, so the tensor must come out symmetric; that is a
cheap runtime invariant.

### 6.4 Atomic energy partition (normative; Symmetrix requirement)

The paper and the toolkit return only a total energy. Symmetrix's correction contract
requires per-atom energies that sum to the scalar energy, in the same convention as
real-space D3 ("each atom receives half of every pair it participates in").

Because $E_{\rm recip}$ is homogeneous of degree two in the assigned mesh, Euler's
theorem gives an exact symmetric partition with no extra machinery:

$$
\boxed{\
e_i = \underbrace{\frac12\sum_\ell c_{i,\ell}\sum_{\mathbf g}W_i(\mathbf g)\,P_\ell(\mathbf g)}_{\text{reciprocal share}}
+ \underbrace{\frac12\sum_\ell \lambda_\ell\,c_{i,\ell}^2\,\varphi_{Z_iZ_i}(0)}_{\text{self share}}\ }
$$

using the same potential and the same stencil weights as the force. Then
$\sum_i e_i = E_{\rm D3}$ **exactly** (to reduction round-off), which must be asserted in
tests. This is exactly the "half of each pair" convention of a real-space pair sum: each
pair's energy is split evenly between its two atoms, and the self term — already a
per-atom quantity — is halved the same way. Under rank chunking, each chunk contributes
additively to $e_i$.

The partition is a definition, not a physical decomposition: it is the symmetric
quadratic-form share, and it must be documented as such. It is **not** the same partition
a real-space D3 evaluation would produce for the same geometry (the model differs in the
coordination numbers, and the reciprocal form has no pair decomposition).

## 7. Design choices versus ordinary D3 behaviour

Everything in this table is a place where an implementer might reasonably assume ordinary
D3 semantics. Each row states what is actually required.

| # | Item | Ordinary D3 | FourierD3 (this specification) | Kind |
|---|---|---|---|---|
| 1 | Coordination number | fixed steepness 16, non-zero limit | modified, reaches 0 continuously at $R_{\rm cut}$ | paper model change |
| 2 | Steepness growth law | n/a | $(r-R_{\rm mid})^2/((R_{\rm cut}-r)^2+\varepsilon)$ | **NVIDIA v0.5.0 choice**, deviates from P.18 |
| 3 | $R_{\rm mid}$ | n/a | $\tfrac12(\tfrac34 C_{\rm table} + R_{\rm cut})$ | convention (P.18 uses bare radii) |
| 4 | $\varepsilon$ regulariser | n/a | $10^{-6}$ Å², not unit-converted | implementation detail, unobservable |
| 5 | `rcov` table | has 4/3 folded in | same table, same folding | reuse, no rescale |
| 6 | $C_6$ separability | pairwise interpolation | rank-$\ell$ functional decomposition | paper method |
| 7 | Rank / tolerance | n/a | smallest $r$ with max rel. error over populated entries $\le 10^{-4}$; scan, do not bisect; order by $\lvert\lambda\rvert$ | tolerance from P.III.A; scan/order are toolkit choices |
| 8 | Eigenvector sign | n/a | sign-normalised (largest-magnitude entry positive) | **Symmetrix requirement** (determinism), not paper |
| 9 | Dispersion cutoff | user-chosen real-space cutoff, truncation error $\propto 1/r^3$ | none | paper method |
| 10 | Ewald splitting | n/a | none: single sum, no $\alpha$ | paper method |
| 11 | $k=0$ | n/a | **included** | consequence of the model; do not copy Coulomb tin-foil |
| 12 | Self term | none | $+\tfrac12\sum_i\sum_\ell\lambda_\ell c_{i,\ell}^2\varphi_{Z_iZ_i}(0)$, and it feeds the forces | P.25 |
| 13 | Self-image pairs $(i=j,\mathbf T\neq\mathbf 0)$ | absent: a pair sum runs over distinct atoms, so an atom does not interact with its own images | **retained**: only $(i=j,\mathbf T=\mathbf 0)$ is cancelled, so an isolated atom in a periodic cell has non-zero, cell-dependent energy | **NVIDIA/FourierD3 convention** (§4.5) |
| 14 | Assignment | n/a | B-spline order $p\in[2,6]$, default 4, every stencil point kept | toolkit choice (no threshold) |
| 15 | Deconvolution | n/a | continuous $\lvert\operatorname{sinc}(m/N)\rvert^p$ per axis, applied once, chosen for v0.5.0 fixture agreement | toolkit/PME convention; discrete-symbol alternative unresolved (§5.2) |
| 16 | FFT normalization | n/a | unscaled forward **and** unscaled inverse; cotangent defined so the inverse is $\partial E/\partial m$ | toolkit convention, required for the adjoint |
| 17 | Mesh selection | n/a | exactly one of size/spacing; spacing rounded up to 2,3,5,7-smooth; every axis $\ge p$ nodes | toolkit choices |
| 18 | Per-atom energy | half of each pair | symmetric quadratic-form share (§6.4) | **Symmetrix requirement** |
| 19 | Virial sign | $W = -\partial E/\partial u$ | same; Symmetrix exposes $\sigma = -W/\Omega$ | shared convention |
| 20 | Forces | explicit, not autodiff | explicit, analytic, hand-derived | shared |
| 21 | Damping | zero and BJ | BJ only | scope |
| 22 | Units | table units (bohr/Eh) | Å/eV, conversions fixed in §2.1 | Symmetrix |
| 23 | Boundary conditions | periodic or open | fully periodic, non-singular cell only | scope |

## 8. Constants (normative values)

| Constant | Value | Role |
|---|---|---|
| `CN_STEEPNESS` | 16.0 | base steepness of the counting function |
| `CN_UNSCALE` | 3/4 | converts the table `rcov` (4/3 folded) back to the bare radius |
| `CN_EPSILON` | $10^{-6}$ | regulariser in the steepness denominator (§3.2.3) |
| `CN_GAUSSIAN` ($k_3$) | 4.0 | Gaussian width of the reference weighting |
| series crossover | $x = 0.1$ | series/closed-form switch for $G_6, G_8$ |
| `R6` series | $1,\ 0,\ -\tfrac13,\ \tfrac18,\ -\tfrac1{60},\ 0,\ \tfrac1{5040}$ | $G_6$ for $x<0.1$ |
| `R8` series | $-0.54119610014619668,\ 0,\ 0.090199350024366076,\ 0,\ -0.010888024707303151,\ \tfrac1{360},\ -0.00025923868350721731$ | $G_8$ for $x<0.1$ |
| rank tolerance | $10^{-4}$ | default decomposition tolerance |
| modulus floor | $10^{-10}$ | division guard in the deconvolution |
| default spline order | 4 | allowed 2–6 |
| $|\mathbf k|$ guard | $10^{-12}$ | suppresses the $1/\lvert\mathbf k\rvert$ term at $k=0$ |
| `bohr_to_angstrom` | 0.52917725999999998 | length conversion |
| `hartree_to_ev` | 27.211385050000001 | energy conversion |
| `hartree_bohr6_to_ev_angstrom6` | 0.59752721065076475 | $C_6$ conversion |

## 9. Verification obligations for later tasks

The following were checked while freezing this specification with a throwaway NumPy
harness (not committed; superseded by the permanent oracle). They are recorded
as the evidence that the equations above are self-consistent, and as the checks later
tasks must reproduce:

| Check | Result |
|---|---|
| Mesh energy vs the v0.5.0 NumPy direct-lattice-sum oracle, 4-atom triclinic cell, $20\times18\times22$ mesh, $p=4$ | $4.8\times10^{-9}$ relative |
| Mesh-free direct reciprocal sum (no mesh, no spline) vs the same oracle | $1.4\times10^{-8}$ relative |
| Mesh vs mesh-free direct reciprocal sum | $1.0\times10^{-9}$ relative |
| Analytic forces vs central differences of the same energy (all 12 components) | $1.5\times10^{-10}$ abs, scale $1.2\times10^{-2}$ |
| Analytic $\partial E/\partial u$ vs central differences under affine strain (all 6 Voigt components, positions and cell deformed together) | $6.4\times10^{-11}$ abs, scale $7.2\times10^{-2}$ |
| Virial tensor symmetry | $<10^{-18}$ |
| $\sum_i e_i$ vs $E$ | $4.9\times10^{-17}$ abs |
| Closed-form $G_6, G_8$ vs numerical quadrature of $\int\varphi e^{-i\mathbf k\cdot\mathbf r}d^3r$ | $\le 1.4\times10^{-6}$ relative (quadrature-limited) |
| $\theta_{ij} = \tfrac12$ at $r = C$ | exact |
| $\theta_{ij}$ and $\partial\theta_{ij}/\partial r$ at $R_{\rm cut}^{-}$ | exactly 0 |

Mandatory obligations:

- **Independent oracle**: build the direct-reciprocal oracle (mesh-free, explicit structure factors)
  and the provenance-tracked fixtures; reproduce the oracle agreement above; **pin the
  exact per-axis assignment symbol** (§5.2 open item) and record which convention is used.
- **Counting function**: test the counting function's exact cutoff value and its derivative before
  production code; assert both are exactly zero at $R_{\rm cut}$ and that
  $\theta = \tfrac12$ at $r = C$.
- **Reciprocal energy**: assert the $k=0$ term is present and non-zero, that the self term is added
  with the sign of §4.5, and that the deconvolution uses one modulus per bin.
- **Forces and stress**: central-difference forces and all six strain-derived stress components on
  orthogonal **and triclinic** cells (the transpose in §6.2 is invisible on a cubic cell);
  assert $\sum_i e_i = E$; assert the four force-sum checks of §6.2 — the exact
  coordination sum, the equality of the total and mesh sums, the equality of the total sum
  to $-dE/dt$ under a rigid translation, and an absolute bound taken from a recorded
  mesh-refinement sequence. The literal $\sum_i\mathbf F_i = 0$ is **not** an obligation
  and is not achievable for a compact-support particle mesh.
- **Mesh and backend convergence**: mesh convergence, spline orders 2–6, rank chunking invariance, atom
  reordering invariance, and cross-backend agreement at the tolerances declared in the qualification matrix.

## 10. References

- Valeeva, Ho, Geiger, Pellegrini, Csányi, Kucukbenli, Ortner, *A fast summation method
  for the DFT-D3 dispersion correction*, arXiv:2607.15103v2 (2026).
- NVIDIA ALCHEMI Toolkit-Ops v0.5.0, `nvalchemiops/interactions/dispersion/_fourier_dftd3.py`,
  `_c6_decomposition.py`, `test/interactions/dispersion/_fourier_reference.py`,
  tag `v0.5.0`, commit `6702138e404d40ea79dec62f68c2fa67e858580e`.
- `docs/userguide/about/conventions.md` (toolkit virial/stress/separation conventions).
- `docs/user/dispersion_corrections.md:44` (Symmetrix ASE stress order and units).
- `libsymmetrix/source/d3_kokkos_impl.tpp`, `dispersion_parameters.{hpp,cpp}`,
  `dispersion_parameter_provenance.hpp` (existing D3 model, tables, conversions).
