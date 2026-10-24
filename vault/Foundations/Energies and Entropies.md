#definition #overview

> The vault and the code use "energy" for at least seven different objects and "entropy" for
> five, and they are easy to confuse. This note separates them. The first table is the whole
> point; every later section explains one row, says where it lives in the code, and how it
> relates to its neighbours. Read it before [[Scalar and Multivariate Energy]],
> [[Bethe Free Energy]] or [[Energy-Based Learning]].

> Sources: St Clere Smithe & Perin, *AutoBayes*, arXiv:2503.18608, 2025 (Definitions 20, 22, Theorem 23, Remarks 24, 26); Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. Inf. Theory 2005; LeCun, Chopra, Hadsell, Ranzato & Huang, *A Tutorial on Energy-Based Learning*, 2006; Perrone, *Markov Categories and Entropy*, IEEE Trans. Inf. Theory 70(3), 2024 (Corollary 4.7); the derivative-order table of §7 follows the Gemini review of 2026-10-23 (`Lenticulum.jl-Review/07_Orders_of_Message_Passing_and_Energies.md`), checked here; code: `lib/LenticulumCore.jl/src/energy.jl`, `statistical_game.jl`, `lib/Mycelium.jl/src/free_energy.jl`, `src/gaussian.jl`, `lib/VariationalDiffusion.jl/src/energy_network.jl`
>
> Bibliography: [[Bibliography#^stclere2025autobayes|St Clere Smithe & Perin 2025]] · [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^lecun2006tutorial|LeCun et al. 2006]] · [[Bibliography#^perrone2024entropy|Perrone 2024]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Entropy in a Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Entropy-in-a-Markov-Category) · [Fisher Information Metric](https://mathstruct.org/CategoryTheory-ML-Wiki/Fisher-Information-Metric)

## 1. The table

**Energies.** Rows 1–4 are functions of *points*; rows 5–7 are functionals of *beliefs*.

| # | name here | symbol | evaluated at | value | code | minimised where |
|---|---|---|---|---|---|---|
| 1 | **pointwise energy** | $\mathbf l^c(x, a, y)$, e.g. the residual $r = y - Ax - b$ | a point | vector, signed | `energy`, `pointwise_energyspace` | $r = 0$, the relation |
| 2 | **scalar pointwise energy** | $\sigma(\mathbf l^c)$, e.g. $\tfrac12 r^\top Q^{-1} r$ | a point | number $\ge 0$ | `scalar_energy`, `pointwise_scalarisation` | the relation |
| 3 | **negative log-likelihood** | $-\log p(y \mid x) = \tfrac12 r^\top Q^{-1} r + \tfrac12\log\det(2\pi Q)$ | a point | number | the `fit` + `complexity` integrands | the relation, *and* small $Q$ |
| 4 | **potential** (energy of an energy-based model) | $E_\theta(z) = -\log \tilde p_\theta(z)$ | a point | number, unnormalised | `EnergyNetwork`, `energy(::NoisePredictor{<:EnergyNetwork}, …)`, `implicit_energy` | the modes of the model |
| 5 | **free energy of a factor** | $\mathbf F^c(\pi, y) = \mathbb E_{c'_\pi(y)}[\mathbf l^c] - \mathbf H^c$, graded `(fit, complexity, negentropy)` | a belief | vector (graded) | `free_energy`, `Mycelium.local_free_energy`, `energyspace` | — (a summand) |
| 6 | **scalar free energy** | $F^c = \sigma_c(\mathbf F^c)$ | a belief | number | `scalar_free_energy(f, π, y, …)`, `scalarisation` | — |
| 7 | **Bethe free energy of a graph** | $F_{\mathrm{Bethe}} = \sum_a F_a + \sum_v (1 - d_v)\,(-H_v)$ | all beliefs | number (graded by factor and variable) | `bethe_free_energy`, `scalar_free_energy(store, g, …)` | the fixed points of belief propagation |

**Entropies.**

| # | name here | symbol | of | code |
|---|---|---|---|---|
| A | **game entropy** | $\mathbf H^c(\pi, y)$, the regulariser slot of a statistical game; usually the entropy of the inversion $c'_\pi(y)$ | a factor's inversion | `entropy(f, π, y, …)` |
| B | **factor-belief entropy** | $H(b_a)$, $b_a \propto \psi_a \prod_i \mu_{i\to a}$ | a factor's joint belief | inside `local_free_energy`, as `negentropy = -H` |
| C | **variable entropy** | $H(b_v)$, weighted by the counting number $1 - d_v$ | a variable's marginal | `Mycelium.variable_entropy` |
| D | **Shannon / differential entropy** | $-\sum p \log p$ (discrete) or $-\int p \log p\,dx$ (continuous, relative to Lebesgue) | any belief | `variable_entropy` by belief type |
| E | **categorical entropy** (Perrone) | $H_D(f) = D(\mathrm{copy}\circ f \,\Vert\, (f\otimes f)\circ\mathrm{copy})$ | a morphism | not in the code; theory only |

## 2. Energies on points

**Pointwise energy (row 1).** What a factor charges at one configuration, before any
scalarisation: for the Gaussian, constraint and nonlinear factors the **residual**, a signed
vector that is zero exactly on the relation; for a likelihood factor the negative
log-likelihood as a one-element vector; for a diffusion factor a graded pair
`(clamp, score)`. It is kept a vector because its Jacobian is what Gauss–Newton steps, the
Fisher metric and the implicit function theorem need ([[Scalar and Multivariate Energy]] §6;
[[L-ADR-0002 Energies are vectors, scalarised late]]).

**Scalar pointwise energy (row 2).** The pointwise energy made a number by a scalarisation,
e.g. $\tfrac12 r^\top Q^{-1} r$ with `SquaredNorm(Q⁻¹)`. It is **zero at a perfect fit** by design
([[Scalar and Multivariate Energy]] §8, "Normalised or not").

**Negative log-likelihood (row 3).** Row 2 plus the normaliser $\tfrac12\log\det(2\pi Q)$. The two
differ by a constant while $Q$ is fixed and are different objectives once $Q$ is learned: row 2
alone drives $Q \to 0$ ([[LP-005 Learning the noise covariance]]). In the graded free energy the
two pieces are the `fit` and `complexity` summands.

**Potential (row 4).** In energy-based models and in the diffusion family, "energy" means an
unnormalised negative log-density $E_\theta(z)$ on the joint space, with $p_\theta \propto e^{-E_\theta}$
([[Energy-Based Learning]]). It is a **different kind of object from row 1**: the residual that
vanishes on the relation is its *gradient*, $r_\theta(z) = \nabla_z E_\theta(z)$. Two consequences:

- for the linear-Gaussian factor the two meet: $E(x, y) = \tfrac12 r^\top Q^{-1} r$ (+ const), so
  the potential *is* the scalar pointwise energy of row 2;
- for an energy network they do not: applying `SquaredNorm` to the residual $\nabla_z E_\theta$
  gives $\tfrac12\lVert\nabla_z E_\theta\rVert^2$, a squared gradient norm (the Gauss–Newton merit of
  the stationarity condition), **not** $E_\theta$.

**Not these.** Physical energies in the applications (electrical power, a molecular potential,
mechanical work) are quantities in the model, not any of the rows above, even where a
potential $E_\theta$ is fitted to them.

## 3. Energies on beliefs

**Free energy of a factor (row 5).** AutoBayes Definition 20, in vector form: the expected
pointwise energy under the inversion minus the game entropy,

$$
\mathbf F^c(\pi, y) = \mathop{\mathbb E}_{(x,a)\sim c'_\pi(y)}\bigl[\mathbf l^c(x, a, y)\bigr] - \mathbf H^c(\pi, y).
$$

For the Gaussian factors it is graded `(fit, complexity, negentropy)`: `fit` is
$\mathbb E[\tfrac12 r^\top Q^{-1} r]$, `complexity` the normaliser, `negentropy` $= -H$. In a graph the
same quantity is computed from the incoming messages by `local_free_energy`; the game-level
`free_energy(f, π, y, …)` passes the prior `π` as the message on the unobserved channel and
`y` on the observed one ([[L-SPEC-001 The Factor Interface]] §4).

**Scalar free energy (row 6).** Row 5 scalarised; with the identity on each summand it is
fit + complexity − entropy. The bridge to rows 2–3: at a point, `fit` equals the scalar
pointwise energy; with a covariance it is that energy at the mean plus
$\tfrac12\operatorname{tr}(Q^{-1}\operatorname{Cov} r)$ (tested in `test/runtests.jl`).

**Bethe free energy (row 7).** The free energy of a whole graph: factor free energies minus
the overcounted variable entropies,

$$
F_{\mathrm{Bethe}} = \sum_a F_a + \sum_v (1 - d_v)\,\bigl(-H(b_v)\bigr),
$$

with $d_v$ the degree of variable $v$. **Exact on a tree**, where it equals the variational
free energy, and with the prior factors in place (AutoBayes Remark 24) it equals
$-\log p(\text{data})$ when the beliefs are exact: the project's main correctness test. On a
loopy graph it is an approximation whose stationary points are the fixed points of loopy belief
propagation (Yedidia, Freeman & Weiss). Its negative is the ELBO when the beliefs are a
variational family.

## 4. Entropies

- **Game entropy (A) versus factor-belief entropy (B).** A is a slot in a statistical game, the
  entropy of the *inversion* given a prior; B is the entropy of the factor's *joint* belief in a
  graph. They coincide for a factor whose inversion is the factor belief; in a graph, B is what
  the Bethe formula uses.
- **Variable entropies (C)** enter the Bethe free energy with counting number $1 - d_v$, which is
  negative for $d_v \ge 2$: they *correct* the overcounting of B, they are not regularisers.
- **Discrete versus continuous (D).** Shannon entropy is $\ge 0$ and intrinsic. Differential
  entropy can be **negative**, changes under a change of variables (by the log-Jacobian), and is
  relative to Lebesgue measure. Free energies of continuous beliefs are therefore comparable only
  between models over the same coordinates ([[LP-006 Gaussian entropies are relative to Lebesgue measure]]).
- **Categorical entropy (E).** Perrone's entropy is the divergence of a morphism from being
  deterministic. With the KL divergence it is Shannon entropy for discrete measures and **infinite
  for every continuous one** (Corollary 4.7): the intrinsic reason D needs a reference measure.
- **A Dirac has no entropy** (`variable_entropy(::DiracBelief) = 0`), which is why factors whose
  inversion is a point (the diffusion and equilibrium factors) contribute energy only.
  A particle belief (`SampleBelief`) is an empirical measure with no density, so it has no
  differential entropy, and `variable_entropy` throws for it rather than guess.

**Look-alikes that are not entropies.**

- **Negentropy** is $-H$: the `negentropy` summand is *minus* an entropy, so it is $\le 0$ for discrete
  beliefs and of either sign for continuous ones.
- **Cross-entropy** $\mathbb E_q[-\log p]$ is an *energy* (the energy of a prior game), not an entropy;
  the score-matching term of the diffusion factor is filed this way
  ([[The Implicit Diffusion Factor as a Statistical Game]] §2).
- **Conditional entropy** $H(y \mid x)$ appears when a likelihood factor's output is unobserved: it
  cancels the expected energy, leaving $-H(x)$ (`local_free_energy(::LikelihoodFactor, …)`,
  predictive case).
- **Mutual information** measures the laxness of parallel composition (AutoBayes Remark 26): it
  is a correction between free energies, not the entropy of anything.

## 5. Temperature

All of the above are at temperature 1. With $p \propto e^{-E/T}$ the free energy is
$F = \mathbb E_q[E] - T\,H(q)$. As $T \to 0$ the entropy term vanishes, the optimal belief collapses
to a point, and free-energy minimisation becomes energy minimisation (MAP; sum-product message
passing becomes max-product). Dirac beliefs are this limit; a factor that returns them behaves as
if at $T = 0$ even inside a graph at $T = 1$.

## 6. Checklist of confusions

1. Energy (rows 1–4) versus free energy (rows 5–7): a function of points versus a functional of
   beliefs.
2. Vector versus scalar: `energy` and `free_energy` return vectors; `scalar_energy` and
   `scalar_free_energy` return numbers. Scalarising twice was a real bug (½‖r‖² squared again in
   the DEQ and neural-ODE factors, fixed 2026-10-24).
3. Residual versus potential: for an energy-based model the residual is $\nabla E_\theta$, and its
   squared norm is not $E_\theta$.
4. `fit` versus negative log-likelihood: they differ by `complexity`.
5. Entropy versus `negentropy`: opposite signs.
6. Bethe counting numbers: $(1 - d_v)$ multiplies $-H_v$. The `bethe_free_energy` docstring had
   the sign wrong until 2026-10-24; the code was right.
7. Differential entropies are coordinate-dependent and can be negative.
8. The training loss need not be any free energy above: score matching, a bilevel prediction loss
   or a Lagrangian are separate objectives ([[Training Energy-Based Models]]).

## 7. Derivative orders

Whether a model is written as a residual $r_\theta$ or as a potential $E_\theta$ shifts every
derivative order by one, which matters for cost:

| operation | residual $r_\theta(z) = 0$ | potential $E_\theta(z)$, $r = \nabla_z E_\theta$ |
|---|---|---|
| evaluate the defect | $r_\theta$ | $\nabla E_\theta$ (1st) |
| Gaussian (Laplace) message precision | $\partial_z r_\theta$ (1st) | $\nabla^2_{zz} E_\theta$ (2nd) |
| backward pass by the implicit function theorem | solve with $(\partial_z r_\theta)^\top$ (1st) | solve with $\nabla^2_{zz} E_\theta$ (2nd) |
| differentiate the Laplace precision or entropy | 2nd | 3rd |

For a potential the Hessian is symmetric, so the matrix of the backward solve and the precision
of the Laplace message are the **same matrix** (with the anchor precisions added to both). Caching
it at the end of inference gives the Gaussian message and the backward pass for one Hessian.
Differentiating through $\tfrac12\log\det\nabla^2 E_\theta$ needs third derivatives; training by
score matching or by a point-prediction loss through the implicit function theorem avoids it.

Related: [[Scalar and Multivariate Energy]], [[Bethe Free Energy]], [[Energy-Based Learning]],
[[Energy-Based Factor Graphs]], [[Training Energy-Based Models]],
[[The Implicit Diffusion Factor as a Statistical Game]], [[Factors are Parameterized Statistical Games]],
[[L-SPEC-001 The Factor Interface]]
