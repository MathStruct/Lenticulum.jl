#design #theorem

> **What is the smallest stochastic factor from which a compositional generative learner can be
> built?** Start from a linear constraint $Az = b$ and a probability measure on its solution set.
> A Gaussian there never leaves the Gaussian class, however the factors are composed (§2). The
> smallest addition that does is one **discrete outcome that selects the right-hand side**,
> $Az = b_k$: a measurement of the linear observable $Az$. That is a conditional-Gaussian
> ("switching constraint") factor, closed-form in every polarity, with a closed-form local free
> energy, and composed it represents exactly the finite Gaussian mixtures (§3–§5). The
> quantum-inspired alternative that adds something mixtures cannot is a **Born joint**,
> $p(k, z) \propto \mathcal N(z)\,\tilde z^\top E_k \tilde z$ with a POVM $\{E_k\}$: densities
> with exact zeros and polynomial tilts (§6). Diffusion-like behaviour comes from a chain of
> switching factors with a fresh outcome per step, which is a deep Gaussian mixture (§8).
> Branch `experiment-qm`; nothing here is in the package yet.

> Sources: original to this vault (design and analysis). Definitions from [[Scalar and Multivariate Energy]] (§3 the multivariate statistical game, §4 the chain rule, §8 the two levels); [[Bethe Free Energy]]; Yedidia, Freeman & Weiss, *Constructing free-energy approximations and generalized belief propagation algorithms*, IEEE Trans. Inf. Theory 51(7), 2005 (Bethe stationary points are BP fixed points). Pointers **not yet checked** against Crossref or arXiv and not in `refs.bib`: Lauritzen & Wermuth (1989) and Lauritzen (1992) on conditional-Gaussian distributions; Minka (2001) on expectation propagation; Han et al. (2018) and Cheng et al. (2018) on Born machines; Loconte et al. on squared ("subtractive") mixtures; van den Oord & Schrauwen (2014) and Viroli & McLachlan (2019) on deep Gaussian mixtures; Naimark's dilation theorem. code: `examples/measurement_factor.jl` (the spectral baseline of §7)
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Gaussian Relations](https://mathstruct.org/CategoryTheory-ML-Wiki/Gaussian-Relations) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

Status of every claim: **proved here** (with a proof below it), **known** (a standard result
whose source still has to be checked before it goes into `refs.bib`), or **expected** (a
conjecture, with what would test it).

## 0. Requirements

A factor in this programme should have

| | requirement | in Lenticulum terms |
|---|---|---|
| R1 | a simple definition, linear or nearly so | parameters are matrices and offsets |
| R2 | a stochastic operation that produces samples | a proper conditional to sample from |
| R3 | several admissible polarities | `supported_polarities` with more than one entry |
| R4 | a multivariate statistical game | [[Scalar and Multivariate Energy]] §3: lens, vector energy, entropy, scalarisation |
| R5 | differentiable learning, through the stochastic step | gradients of the free energy in the parameters |
| R6 | composition, message passing, Bethe free energy | `factor_message`, `local_free_energy` |

and **complexity should come from the graph**, not from a network inside a factor.

## 1. The constraint and its polarities

Let $z \in Z = \mathbb R^n$ (the concatenated channels), $A \in \mathbb R^{m \times n}$ of rank
$r$, and $S = \{z : Az = b\} = z_0 + \ker A$, an affine subspace of dimension $n - r$ when $b \in
\operatorname{range} A$. A polarity $p$ splits the coordinates $Z \cong X_p \oplus Y_p \oplus U_p$
(observed, inferred, latent). Clamping $x$ leaves

$$
S_x = \{(y, u) : A_Y y + A_U u = b - A_X x\},
$$

empty, a point, or an affine subspace. A constraint alone selects a *set*; to sample (R2) it
needs a probability measure on that set, and the whole design question is **which measure**.

Two ways to supply one:

- **per polarity**, a separate rule for picking a point of $S_x$ (a sampler, a measurement);
- **once**, a joint distribution $P$ on $S$ (or near it), with every polarity's kernel its
  conditional $P(y \mid x)$, $U$ marginalised.

> **Definition (polarity compatibility).** A family of per-polarity kernels $\{c_p\}$ is
> *compatible* if one joint $P$ on the channels has every $c_p$ as its conditional
> $P(Y_p \mid X_p)$, with $U_p$ marginalised.

**Observation (proved here).** If the kernels are compatible, Bayes' rule under $P$ maps any one
to any other, so the answer to a query does not depend on which polarity's rule produced it, and
a graph built from such factors has one joint. If they are not, two queries that should agree (say
$x \mid y$ computed directly, and by inverting $y \mid x$ against the marginal of $x$) can disagree,
and no single model stands behind the factor. Compatibility of arbitrarily specified conditionals
is a genuine restriction (the theory of conditionally specified distributions; pointer to check:
Arnold, Castillo & Sarabia); the simple way to have it is to define the joint first.

This is the criterion **E** asks for, and it decides most of what follows: every construction
below either defines a joint (and passes) or does not.

## 2. Linear plus Gaussian stays Gaussian

The obvious measure is a Gaussian on $S$: $z = z_0 + N w$ with $N$ a basis of $\ker A$ and
$w \sim \mathcal N(\mu, \Sigma)$, or its soft version, the potential
$\psi(z) = \mathcal N(Az; b, Q)$ times Gaussian priors on the variables. That is
`LinearConstraintFactor` with a prior, and the hard constraint is its limit $Q \to 0$.

> **Proposition 2 (proved here; the categorical form is known: Gaussian relations).** Any factor
> graph of linear-Gaussian potentials and Gaussian priors, in any polarity and any wiring, has a
> Gaussian (possibly degenerate) joint. All its marginals and conditionals are Gaussian, hence
> unimodal; no wiring, depth or choice of query creates a second mode.
>
> *Proof.* The joint density is a product of exponentials of quadratics in $z$, hence an exponential
> of a quadratic: Gaussian when normalisable, degenerate in the hard limit. Marginals and
> conditionals of a Gaussian are Gaussian. $\square$

So the affine-nullspace sampler with a Gaussian density **is** the linear-Gaussian fragment that
Lenticulum already has ([[The Linear Gaussian Chain]], [[Depth in Implicit Learning]] §2, where
latent variables of this fragment are eliminable at no cost). Some non-Gaussian ingredient is
necessary. The question is the smallest one.

### Where the non-Gaussianity can enter

| candidate | what it is | joint exists? | adds capacity? |
|---|---|---|---|
| 1. discrete outcome selecting a linear solution | $Az = b_k$, $k$ drawn | yes (§3) | finite mixtures |
| 2. state-dependent density matrix / POVM | Born weights $\tilde z^\top E_k \tilde z$ | yes, as a Born joint (§6) | polynomial tilts, exact zeros |
| 3. non-Gaussian law on the nullspace | e.g. Laplace, Student-$t$ on $w$ | yes | scale mixtures; needs non-Gaussian beliefs (Gamma) |
| 4. mixture of constraints | $A_k z = b_k$, $k$ drawn | yes (§3, general case) | finite mixtures of subspaces |
| 5. nonlinearity of inference or of the objective | BP, Bethe, EP dynamics | — | **none** by itself (§9) |

Candidates 3 and 4 are special cases of 1 (3 with a continuous latent scale instead of a
discrete $k$). Candidate 5 is the distinction **F** asks for: it changes how the model is fitted
and approximated, not which distributions the model contains.

## 3. The smallest candidate: measuring a linear observable

> **Definition (switching constraint factor).** Channels: continuous channels $z$ as for
> `LinearConstraintFactor`, and one discrete channel $k \in \{1, \dots, K\}$. Potential
> $$
> \psi(k, z) \;=\; w_k\, \mathcal N\bigl(A_k z;\; b_k,\; Q_k\bigr),
> \qquad \textstyle\sum_k w_k = 1,\; Q_k \succ 0 .
> $$
> Minimal form: one shared $A$ and $Q$, $K$ offsets $b_k$ ("$Az$ is measured and the outcome
> is one of $b_1, \dots, b_K$"); general form: per-outcome $A_k$, $Q_k$.

The minimal form is the original construction with the quantum dressing removed: a linear
constraint whose right-hand side is the outcome of a measurement. Its solution sets are $K$
parallel copies of $\ker A$; the general form gives $K$ different subspaces (lines through the
origin at any angles, the case the spectral toy fails, §7).

Like `LinearConstraintFactor`, $\psi$ is a potential, not a normalised density in $z$: it is flat
along $\ker A_k$, and a proper joint comes from the rest of the graph (priors, other factors).

### Messages (all closed form)

Write incoming beliefs $\mu_i = \mathcal N(m_i, S_i)$ on the continuous channels (canonical, possibly
improper, as in `constraint.md` §2) and $\kappa(k)$ on $k$.

- **To $k$** ("the measurement"), with all continuous channels proper:
  $$
  \mu_{f \to k}(k) \;\propto\; w_k\, \mathcal N\bigl(b_k;\; A_k m,\; A_k S A_k^\top + Q_k\bigr),
  $$
  $m, S$ the stacked means and block-diagonal covariances. This is the evidence of outcome $k$:
  the classical Born rule of this factor, and it is a likelihood, not a posterior (L-ADR-0006).
- **To a continuous channel $t$**: a mixture over $k$ of `LinearConstraintFactor`'s message with
  $(A_k, b_k, Q_k)$,
  $$
  \mu_{f \to t} = \sum_k \kappa(k)\, w_k\, Z_k\;
  \mathcal N^{-1}\bigl(A_{k,t}^\top R_k^{-1} d_k,\; A_{k,t}^\top R_k^{-1} A_{k,t}\bigr),
  \quad d_k = b_k - \textstyle\sum_{i \ne t} A_{k,i} m_i,\;
  R_k = Q_k + \sum_{i \ne t} A_{k,i} S_i A_{k,i}^\top,
  $$
  where $Z_k$ is the part of $\int \psi(k, z) \prod_{i \ne t} \mu_i\, dz_{-t}$ that does not depend on
  $z_t$. **The per-outcome constants $Z_k$ must be kept**: they are the relative masses of the
  components, and dropping them is exactly the defect LP-019's reviews found three times in
  `_mixture_product` (constants of different sites do not cancel).

Both are compositions of operations Mycelium already has: Gaussian messages, `CategoricalBelief`,
`MixtureBelief`, and overlaps (`combine_with_overlap`).

## 4. The statistical game (task B)

Fix a polarity $p$ and a proper Gaussian reference on the continuous channels, so that
$P(k, z) \propto \psi(k, z)\,\pi(z)$ is a proper conditional-Gaussian (CG) distribution: Gaussian in
$z$ for each $k$, $P(k, z) = \beta_k \mathcal N(z; m_k, S_k)$.

> **Proposition 3 (proved here).** For every polarity $p = (X, Y, U)$ with $P(x) > 0$:
>
> 1. **Kernel.** $c_p(y \mid x) = \sum_k P(k \mid x)\, \mathcal N\bigl(y;\, m_{k,Y} + S_{k,YX} S_{k,XX}^{-1}(x - m_{k,X}),\, S_{k,YY} - S_{k,YX}S_{k,XX}^{-1}S_{k,XY}\bigr)$
>    with $P(k \mid x) \propto \beta_k\, \mathcal N(x; m_{k,X}, S_{k,XX})$ ($U$ marginalised by dropping
>    its coordinates). If $k$ is itself in $Y$, the kernel is the pair $\bigl(P(k \mid x), P(y \mid x, k)\bigr)$;
>    if $k \in X$, it is the Gaussian $P(y \mid x, k)$.
> 2. **Inversion.** For a prior $\pi_X$ that is a Gaussian or a Gaussian mixture, the exact inversion
>    $c'_{\pi}(y)$ is again CG (a mixture over $k$ and over $\pi$'s components), in closed form.
> 3. **Compatibility.** All kernels are conditionals of $P$, so they are compatible (§1).
>
> *Proof.* For fixed $k$, $(x, y)$ is jointly Gaussian under $P(\cdot \mid k)$, so $P(y \mid x, k)$ is the
> Gaussian conditional; $P(k \mid x) \propto P(k) P(x \mid k)$, with $P(x \mid k)$ the Gaussian marginal.
> For (2), $\pi_X(x) P(y \mid x)$ is, for each pair (component of $\pi$, outcome $k$), a Gaussian in
> $(x, y)$ times a constant, so conditioning on $y$ gives a mixture of Gaussians with closed-form
> weights. (3) is by construction. $\square$

**Vector energy.** On the factor's latent $a = (u, k)$, the pointwise energy is the pair
$$
\mathbf l(x, a, y) \;=\; \bigl(\, r_k(z),\; -\log w_k \,\bigr) \in \mathbb R^m \oplus \mathbb R,
\qquad r_k(z) = A_k z - b_k,
$$
a signed residual plus a selection cost. Its scalarisation
$\sigma_k(r, s) = \tfrac12 r^\top Q_k^{-1} r + s$ depends on the latent outcome through $Q_k$: a
point-dependent scalarisation, which [[L-ADR-0007 Point-dependent scalarisations are localised before use]]
already handles (localise at the outcome, then scalarise). With a shared $Q$ it is an ordinary
`SquaredNorm` plus a linear term. Like every residual factor, the energies do not depend on the
polarity ([[Scalar and Multivariate Energy]] §8, "Per polarity").

**Free energy.** For a factor belief $b(k, z) = \beta_k \mathcal N(z; m_k, S_k)$ (the Bethe belief
$b \propto \psi \prod_i \mu_{i \to f}$ has this form), graded as in `GaussianFactor`:

$$
\begin{aligned}
\mathrm{fit} &= \textstyle\sum_k \beta_k \Bigl[\tfrac12 (A_k m_k - b_k)^\top Q_k^{-1}(A_k m_k - b_k) + \tfrac12\operatorname{tr}\bigl(Q_k^{-1} A_k S_k A_k^\top\bigr)\Bigr], \\
\mathrm{complexity} &= \textstyle\sum_k \beta_k \bigl[\tfrac12 \log\det(2\pi Q_k) - \log w_k\bigr], \\
\mathrm{negentropy} &= -H(b) = -H(\beta) - \textstyle\sum_k \beta_k\, H\bigl(\mathcal N(m_k, S_k)\bigr).
\end{aligned}
$$

> **Proposition 4 (proved here).** $\mathrm{fit} + \mathrm{complexity} + \mathrm{negentropy} =
> \mathrm{KL}\bigl(b \,\|\, \psi\bigr) \ge -\log \sum_k \int \psi(k, z)\, dz$ when the right side is
> finite, i.e. it is the local variational free energy of $\psi$, and every term is in closed form.
>
> *Proof.* $-\log \psi(k, z) = \tfrac12 r_k^\top Q_k^{-1} r_k + \tfrac12 \log\det(2\pi Q_k) - \log w_k$, and
> $\mathbb E_{\mathcal N(m_k, S_k)}[\tfrac12 r_k^\top Q_k^{-1} r_k]$ is the bracket in "fit" (mean plus
> trace term). So the sum is $\mathbb E_b[-\log \psi] - H(b)$. The entropy of a CG belief with $k$
> explicit is $H(\beta) + \sum_k \beta_k H(\cdot \mid k)$, by the chain rule of entropy; both parts
> are closed form. The bound is Gibbs' inequality. $\square$

This is the important difference from a Gaussian-mixture *variable*: because $k$ is a channel,
the factor belief's entropy is exact. The obstacle moves to the **variables**: a continuous
variable whose marginal is a mixture has no closed-form entropy, and the Bethe correction needs it
whenever its degree is at least 2 (§9).

## 5. Learning (task C)

**Parameters** $\theta = (A_k, b_k, Q_k, w_k)$. The free energy above is a smooth function of
$\theta$ and of the beliefs' parameters; at a BP fixed point the gradient of the Bethe free energy
in $\theta$ is the partial derivative of the local terms with beliefs held fixed (the envelope
theorem at a stationary point, [[Bethe Free Energy]]). For this factor that is an expectation of
$\partial_\theta(-\log \psi)$ under the factor belief: the EM M-step, in closed form.

**The discrete outcome needs no gradient estimator.** $k$ has $K$ values; every quantity sums
over them exactly (log-sum-exp), so gradients are exact, not REINFORCE-style. Sampling $k$ is
needed only to **generate**. If one insists on differentiating a sampled path (e.g. a loss on a
generated sample), the score-function estimator is unbiased; Gumbel-softmax relaxations are
biased and should be avoided where exact summation is available.

**Pathwise gradients through $z$.** Given $k$, $z = m_k + L_k \varepsilon$ with
$L_k L_k^\top = S_k$ is reparameterisable, so losses on generated samples have unbiased pathwise
gradients in $(m_k, S_k)$ and hence in $\theta$.

**Rank changes and the hard limit.** The hard constraint $Q_k \to 0$ makes $S$'s dimension, and
the pseudo-inverse in the messages, jump when $\operatorname{rank} A_k$ changes; the free energy is
then relative to Lebesgue measure on subspaces of different dimensions for different $k$, which do
not compare ([[LP-006 Gaussian entropies are relative to Lebesgue measure]]). Recommendation:
learn with $Q_k \succ 0$ (soft constraints; Cholesky or log-precision parameterisation as in
LP-005) and treat hard constraints as a limit, not a parameterisation.

**Repeated eigenvalues** appear only in the spectral construction of §7: an eigenvector's
derivative contains $1/(\lambda_i - \lambda_j)$ and is undefined on a repeated eigenvalue, where
only the spectral projector onto the eigenspace is differentiable. The switching factor and the
POVM of §6 never diagonalise anything, so the problem does not arise.

## 6. The quantum-inspired candidate, done as a joint

The spectral toy reads a density matrix's eigenvectors. What a measurement can contribute that a
mixture cannot appears only once the Born rule is made into a joint distribution.

> **Definition (Born joint).** Homogeneous coordinates $\tilde z = (1, z)$, a POVM
> $E_1, \dots, E_K \succeq 0$ on $\mathbb R^{n+1}$ with $\sum_k E_k = I$, and a Gaussian reference
> $\mathcal N(z; m, S)$:
> $$
> p(k, z) \;=\; \frac{\mathcal N(z; m, S)\; \tilde z^\top E_k \tilde z}{\mathcal Z},
> \qquad \mathcal Z = 1 + \operatorname{tr}(S + m m^\top).
> $$

> **Proposition 5 (proved here).**
> 1. $p(z) = \mathcal N(z)\,\lVert \tilde z \rVert^2 / \mathcal Z$ and $p(k \mid z) = \tilde z^\top E_k \tilde z / \lVert \tilde z \rVert^2$:
>    **the Born rule**, independent of $m$ and $S$.
> 2. Conditioning on part of $z$: $p(k \mid x) = \mathbb E[\tilde z^\top E_k \tilde z \mid x] / \mathbb E[\lVert \tilde z\rVert^2 \mid x]$ under
>    $\mathcal N(y \mid x)$, a ratio of closed-form quadratic moments, and $p(y \mid x, k) \propto
>    \mathcal N(y \mid x)\, \tilde z^\top E_k \tilde z$, a Gaussian times a quadratic: **not Gaussian**.
> 3. For a rank-one element $E_k = v v^\top$, $p(z \mid k)$ vanishes on the hyperplane
>    $v^\top \tilde z = 0$; when the hyperplane passes near the reference mean, it is bimodal across
>    it. No finite Gaussian mixture has an exact zero.
> 4. All polarities' kernels are conditionals of $p$, so they are compatible (§1).
>
> *Proof.* $\sum_k \tilde z^\top E_k \tilde z = \lVert\tilde z\rVert^2$ and
> $\mathbb E_{\mathcal N}\lVert \tilde z\rVert^2 = 1 + \operatorname{tr}(S + mm^\top)$ give (1). (2) is
> Bayes' rule with Gaussian moments $\mathbb E[\tilde z^\top E \tilde z \mid x] = \bar{\tilde z}^\top E \bar{\tilde z} + \operatorname{tr}(E\,\widetilde{\operatorname{Cov}})$.
> (3): $(v^\top \tilde z)^2 = 0$ exactly on the hyperplane, positive elsewhere; a Gaussian mixture is
> positive everywhere. (4) by construction. $\square$

**What it represents.** Composing $L$ Born factors on shared variables multiplies their quadratics:
densities $\mathcal N(z) \prod_l \tilde z^\top E_{l,k_l} \tilde z$, polynomials of degree $2L$ times a
Gaussian, nonnegative by construction (each factor is a square), with normalisers given by Gaussian
moments of degree $2L$ (Isserlis), whose cost grows combinatorially in $L$.

> **Proposition 6 (proved here, for amplitudes).** Densities $P(z)^2\, \mathcal N(z)/\mathcal Z$
> with $P$ a polynomial are dense in $L^1$ among all densities on $\mathbb R^n$.
>
> *Proof.* For a density $p$, $f = \sqrt{p / \mathcal N}$ satisfies
> $\int f^2\, d\mathcal N = \int p = 1$, so $f \in L^2(\mathcal N)$. Hermite polynomials are complete in
> $L^2(\mathcal N)$, so polynomials $P_d \to f$ in $L^2(\mathcal N)$. Then
> $\lVert P_d^2 \mathcal N - p\rVert_{L^1} = \int |P_d^2 - f^2|\, d\mathcal N \le \lVert P_d - f\rVert_{L^2(\mathcal N)} \lVert P_d + f\rVert_{L^2(\mathcal N)} \to 0$,
> and normalising changes the limit by a factor tending to 1. $\square$

The catch is the word *amplitudes*: Proposition 6 needs an arbitrary polynomial $P$ **before**
squaring, i.e. amplitudes that add coherently and are squared once (a Born machine). A factor graph
multiplies *probabilities*; composing Born factors gives products of squares of linear forms, and
summing over outcomes gives sums of such products. Whether those are dense is **expected but not
proved** (in one variable, $x^2 + 1$ is a sum of two squares of linear forms, which suggests yes).
So the genuinely quantum ingredient, interference, needs amplitude-level composition, which
Lenticulum's probability-level products do not provide.

**Parameterising a POVM (R5).** Stack $E_k = V_k^\top V_k$ with $V = [V_1; \dots; V_K]$ an isometry,
$V^\top V = I$: that is Naimark's dilation (a POVM is a projective measurement on a larger space,
compressed). $V$ lives on a Stiefel manifold and is updated by a Riemannian gradient step with a QR
retraction. This is where the Stiefel machinery of the first prototype belongs: as the
parameterisation of a POVM, not as a deterministic projector.

**Entropy.** For rank-one $E_k = vv^\top$, $p(z \mid k)$ is Gaussian across $v^\perp$ and a
quadratically tilted Gaussian along $v$, so $H(z \mid k)$ is a Gaussian entropy plus a
one-dimensional integral (Gauss–Hermite quadrature). Closed form only up to that quadrature.

## 7. Comparison: spectral measurement, POVM, classical (task 5)

| | spectral projective (the toy) | Born joint with POVM (§6) | switching constraint, classical (§3) |
|---|---|---|---|
| state | $\rho = Z^\top Z / \operatorname{tr}$, eigenvectors | Gaussian reference, POVM $\{E_k\}$ | potential $w_k \mathcal N(A_k z; b_k, Q_k)$ |
| joint over $(k, z)$ | **none**: outcome weights $\lambda_k (u_k^x\cdot x)^2$ treat $(x, 0)$ as a state | yes | yes |
| polarity-compatible (§1) | no | yes | yes |
| outcomes | at most $n$, **orthogonal** | any $K$ (Naimark) | any $K$ |
| what it represents | orthogonal line mixtures | polynomial-tilted Gaussians, exact zeros | finite Gaussian mixtures, dense as $K \to \infty$ (known) |
| fitting | closed-form PCA, not maximum likelihood | maximum likelihood, closed-form normaliser | EM or Bethe, closed-form local terms |
| gradients | eigenvector derivatives undefined at repeated eigenvalues | smooth on the Stiefel manifold | exact sums over $k$, pathwise in $z$ |
| entropy | — | 1-D quadrature per outcome | closed form for the factor belief |
| existing machinery | `examples/measurement_factor.jl` | none (new belief type) | Gaussian, categorical and mixture beliefs |

Measured for the spectral column (`examples/measurement_factor.md`, 2026-10-31): exact on two
orthogonal lines (held-out $\log p(y \mid x)$ 0.688 against the true model's 0.691), and 30° wrong on
two lines 60° apart, where it does worse than a single Gaussian. The table says why it cannot be
repaired within the construction: orthogonality is forced by the eigendecomposition, and without a
joint the kernels are not polarity-compatible.

**Recommendation:** the switching constraint factor is the building block (smallest, classical,
everything in closed form, reuses existing beliefs). The Born joint is the one candidate worth an
experiment for what mixtures cannot do (zeros, subtractive structure).

## 8. Composition, and when it is a diffusion (task D and §5 of the brief)

> **Proposition 7 (proved here).** A graph of switching constraint factors with discrete
> channels $k_1, \dots, k_L$ (each of $K_l$ states), any linear-Gaussian factors and Gaussian
> priors, has a joint over the continuous variables that is a finite Gaussian mixture with at most
> $\prod_l K_l$ components. Conversely every $K$-component Gaussian mixture on $\mathbb R^n$ is one
> switching factor with $A_k = I$, $b_k = \mu_k$, $Q_k = \Sigma_k$.
>
> *Proof.* For each configuration $(k_1, \dots, k_L)$ the joint is a product of Gaussian potentials,
> Gaussian by Proposition 2, weighted by $\prod_l w_{l, k_l}$ times its normaliser; summing over
> configurations gives the mixture. The converse is immediate. $\square$

So the class is exactly the finite mixtures; **composition buys parameter efficiency, not a
larger class**: $L$ binary switches give up to $2^L$ components from $O(L)$ parameters, but with
structure (with shared operators and per-switch offsets on one variable, the component means are
sums of offsets: a product or tree structure), so a generic $2^L$-component mixture is not reachable
this way. Which data have that structure is the
expressivity question to test (§10, E3).

### What a diffusion model additionally requires

A single stochastic factor, even a multimodal one, is not a diffusion model. A diffusion model is
a chain $z_0 \to z_1 \to \dots \to z_T$ with

1. a **fixed forward corruption** $q(z_t \mid z_{t-1}) = \mathcal N(\sqrt{1-\beta_t}\, z_{t-1}, \beta_t I)$ whose
   marginals $q(z_t \mid z_0)$ are tractable and whose end $q(z_T)$ is close to a fixed prior;
2. a **learned reverse chain** $p_\theta(z_{t-1} \mid z_t)$ trained to match the time reversal of 1,
   by the evidence bound whose terms are per-step KL divergences (or by score matching);
3. **small steps**, so that each true reverse conditional is close to the simple family used for
   $p_\theta$: the true reverse conditional is nearly Gaussian but with a **mean that is nonlinear in
   $z_t$** (it involves the score $\nabla \log q_t$);
4. **ancestral sampling** from the prior through the reverse chain.

Requirement 1 is free here: it is a chain of `GaussianFactor`s ([[The Linear Gaussian Chain]]).
Requirement 3 is where a linear factor fails: a Gaussian reverse kernel with linear mean is exact
only for Gaussian data. Requirement 2's objective is the chain's variational free energy with the
**forward process as the fixed inversion**, so its entropies are those of the Gaussian forward
process, in closed form; the Bethe variable-entropy obstacle of §4 does not arise (this is an upper
bound, like `VariationalDiffusion`'s `FreeEnergyUpperBound`, not an exact free energy).

> **Proposition 8 (proved here).** If the data are a $K$-component Gaussian mixture, every true
> reverse conditional $q(z_{t-1} \mid z_t)$ of the forward process 1 is a switching linear kernel
> with $K$ outcomes:
> $$
> q(z_{t-1} \mid z_t) = \sum_k q(k \mid z_t)\, \mathcal N\bigl(z_{t-1};\; G_{k,t} z_t + h_{k,t},\; C_{k,t}\bigr),
> \qquad q(k \mid z_t) \propto \pi_k\, \mathcal N\bigl(z_t;\; \alpha_t \mu_k,\; \alpha_t^2 \Sigma_k + \sigma_t^2 I\bigr),
> $$
> with $\alpha_t^2 = \prod_{s \le t}(1-\beta_s)$, $\sigma_t^2 = 1 - \alpha_t^2$.
>
> *Proof.* Given the component $k$, $(z_0, z_{t-1}, z_t)$ is jointly Gaussian (a linear-Gaussian
> chain started at $\mathcal N(\mu_k, \Sigma_k)$), so $q(z_{t-1} \mid z_t, k)$ is Gaussian with mean
> linear in $z_t$. Summing over $k$ with $q(k \mid z_t) \propto \pi_k q(z_t \mid k)$ gives the formula,
> and $q(z_t \mid k) = \mathcal N(\alpha_t \mu_k, \alpha_t^2 \Sigma_k + \sigma_t^2 I)$. $\square$

So the switching factor is the **exact** reverse kernel for mixture data, and its gate
$q(k \mid z_t)$ is the measurement of §3. Two consequences, one deflating, one constructive:

- **For mixture data diffusion is unnecessary.** One switching factor $k \to z_0$ is already the
  exact model, sampled in one shot. Diffusion earns its keep only when the data are not in a
  tractable family.
- **With a fresh outcome per step it is compositional.** Give each reverse step its own $k_t$:
  $p_\theta(z_{t-1} \mid z_t) = \sum_{k_t} p(k_t \mid z_t)\, \mathcal N(G_{k_t} z_t + h_{k_t}, C_{k_t})$.
  Sampling alternates a measurement ($k_t$ from its Born-like gate) and a linear-Gaussian update:
  iterative refinement with no network anywhere. The generated distribution is a mixture over paths
  $(k_1, \dots, k_T)$, i.e. a deep Gaussian mixture or a switching linear dynamical system run
  backwards in time (known model classes; pointers to check). The hypothesis that makes this
  interesting, **expected, not shown**: because small steps make each reverse conditional nearly
  Gaussian with a smooth mean, few outcomes per step ($K_t = 2$ or $3$) suffice even when the data
  need many components, so $T$ steps of $K_t$ outcomes compete with one factor of
  $\prod_t K_t$ components at a fraction of the parameters.

**Learning the reverse chain** needs no discrete gradient estimator either: sample $z_0$ from the
data and $(z_{t-1}, z_t)$ from the fixed forward process (no parameters, so no reparameterisation is
needed), and evaluate the exact local term $\log \sum_{k_t} p(k_t \mid z_t)\, \mathcal N(z_{t-1}; \dots)$
by log-sum-exp. Its gradient is exact per sample and unbiased over samples: denoising score matching
with a mixture-of-linear-experts score in place of a network.

## 9. What the Bethe objective contributes (task F)

1. **Local credit assignment.** On a tree with exact beliefs the Bethe free energy equals
   $-\log p(\text{data})$, and its gradient in a factor's parameters at a BP fixed point involves only
   that factor's local belief (§5). That is how a graph of simple factors learns without a network
   inside any of them: inference (messages) does the credit assignment that backpropagation through
   layers does in a deep net.
2. **What it approximates.** On loopy graphs, Bethe is an approximation to $-\log Z$
   (Yedidia–Freeman–Weiss; [[Loopy Message Passing]]). On hybrid graphs, exact beliefs are mixtures
   whose number of components grows with every switching factor they pass; practical inference
   collapses them (moment matching, EP: known, pointer to check), and learning then fits the model
   under the collapsed objective, which biases it.
3. **Mixture entropies.** A continuous variable of degree $d \ge 2$ with a mixture belief enters with
   counting number $1 - d$ and needs $H$ of a mixture, which has no closed form. Options: the bounds
   $\sum_k w_k H_k \le H \le \sum_k w_k H_k + H(w)$ (`branch_entropy_bounds`), low-dimensional
   quadrature, or keeping the discrete channel inside the region (a Kikuchi region over $(k, z)$, whose
   CG entropy is closed form).
4. **Optimisation versus representation.** Message passing with mixtures is a nonlinear dynamical
   system, and the Bethe objective is nonconvex in the parameters, but neither enlarges the model
   class: that is fixed by the factor families (Gaussian $\to$ Gaussian, switching $\to$ finite
   mixtures, Born $\to$ polynomial-tilted Gaussians; Propositions 2, 7, 5). The one exception is when
   an approximate **sampler** is itself taken as the model (an implicit generative model defined by
   iterating messages): then the generated distribution can differ from the factors' joint, and its
   class is a different question. This note keeps the two apart.

## 10. A minimal experimental plan (task 6)

Each experiment has a closed-form or exactly computable reference, and measures expressive power
at matched parameter counts, not only accuracy on one data set.

| | question | data | models | reference and metric |
|---|---|---|---|---|
| E1 | does the switching factor fix the spectral toy's failure? | lines at 0° and 60° (the toy's case 2) | spectral (toy), switching $K = 2$ (EM), Born $K = 2$ rank-one | true conditional; held-out $\log p(y \mid x)$, line angles |
| E2 | what can only the Born joint do? | $p(z) \propto (v^\top \tilde z)^2 \mathcal N(z)$ (exact zero on a line), an annulus | Born $K = 1$; switching with growing $K$ | exact density; held-out $\log p$ vs number of parameters |
| E3 | where does composition beat a flat mixture? | a $2^L$ grid of modes (product structure); 8 modes on a ring (no product structure) | $L$ binary switching factors on shared $z$; one factor with $2^L$ outcomes | exact mixture likelihood; $\log p$ vs parameters |
| E4 | do many small switching steps act like a diffusion? | (a) a Gaussian mixture; (b) two moons and a ring | reverse chain of $T$ steps, $K_t \in \{1, 2, 3\}$, trained as in §8 | (a) Proposition 8's exact kernels; (b) held-out $\log p$ by exact path sums for small $T$; vs one factor of $\prod_t K_t$ outcomes |
| E5 | are the polarities compatible in practice? | E1 and E3 data | spectral vs switching vs Born | conditional $\log p$ in three query directions against the joint's conditionals |
| E6 | how much does Bethe/EP collapsing bias learning? | small trees where exact mixture marginals are enumerable | exact sum vs collapsed beliefs vs entropy bounds | parameter error and $\log p$ gap |

Order: E1 and E4(a) first (each has an exact oracle and decides whether the building block and the
diffusion reading are right), then E3 and E4(b) (the expressivity questions), then E2 (whether the
Born joint is worth a belief type), E5 and E6.

## 11. Proposed interface (nothing implemented)

`SwitchingConstraintFactor(dims, K, m; shared = true)`: channels as `LinearConstraintFactor` plus
`:k`; `ps = (A, b = [b₁ … b_K], logw, noise)`; messages as in §3 (to `:k` a `CategoricalBelief`, to a
continuous channel a `MixtureBelief` with the per-outcome constants kept); `local_free_energy` as in
Proposition 4, graded `(fit, complexity, negentropy)`; `supported_polarities`: every split of the
continuous channels with one target, with `:k` observed, latent or the target. The reverse-chain
experiment E4 needs no new factor beyond it. The Born joint would need a new belief type
(Gaussian times a nonnegative quadratic) and is deferred until E2 says it is worth one.

## 12. Open

- Whether sums of products of squared linear forms times a Gaussian are dense (§6), i.e. whether
  probability-level composition of Born factors is universal without amplitude-level interference.
- The size of the advantage of $T$ small switching steps over one large mixture (E4(b)): a bound in
  terms of the data's smoothness would turn the diffusion hypothesis of §8 into a theorem.
- The learning bias of collapsed (EP) beliefs in hybrid graphs (E6).
- Hard constraints with outcome-dependent rank: a reference measure under which entropies of
  different-dimensional solution sets compare (§5).

Related: [[Scalar and Multivariate Energy]], [[Bethe Free Energy]], [[Hierarchical Models]], [[The Linear Gaussian Chain]], [[Depth in Implicit Learning]], [[Universal Approximation for Implicit Diffusion Learners]], [[RQ-001 One Potential, Every Polarity]], [[L-SPEC-003 Exposed Parameters and Energy Channels]], [[The Diffusion Family]]
