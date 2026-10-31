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
> with exact zeros and polynomial tilts, but only through observed or weighted outcomes, since with
> the outcome marginalised $p(z)$ does not depend on the POVM (§6). Diffusion-like behaviour comes from a chain of
> switching factors with a fresh outcome per step, which is a deep Gaussian mixture (§8).
> Branch `experiment-qm`; nothing here is in the package yet.

> Sources: original to this vault (design and analysis; numerical checks in `examples/minimal_stochastic_factors_checks.jl`). Definitions from [[Scalar and Multivariate Energy]] (§3 the multivariate statistical game, §4 the chain rule, §8 the two levels); [[Bethe Free Energy]]; Yedidia, Freeman & Weiss, *Constructing free-energy approximations and generalized belief propagation algorithms*, IEEE Trans. Inf. Theory 51(7), 2005 (Bethe stationary points are BP fixed points). Conditional-Gaussian models: Lauritzen & Wermuth, *Graphical Models for Associations between Variables, some of which are Qualitative and some Quantitative*, Ann. Statist. 17(1), 1989; Lauritzen, *Propagation of Probabilities, Means, and Variances in Mixed Graphical Association Models*, JASA 87(420), 1992. Born machines: Han et al., *Unsupervised Generative Modeling Using Matrix Product States*, Phys. Rev. X 8, 2018; Cheng, Chen & Wang, *Information Perspective to Probabilistic Modeling: Boltzmann Machines versus Born Machines*, Entropy 20(8), 2018. Squared models: Loconte et al., *Subtractive Mixture Models via Squaring: Representation and Learning*, arXiv:2310.00724; Marteau-Ferey, Bach & Rudi, *Non-parametric Models for Non-negative Functions*, arXiv:2007.03926; Tsuchida, Ong & Sejdinovic, *Squared Neural Families: A New Class of Tractable Density Models*, arXiv:2305.13552. Deep mixtures: Viroli & McLachlan, *Deep Gaussian Mixture Models*, Stat. Comput. 29(1), 2019. Mixture diffusion: Shah, Chen & Klivans, *Learning Mixtures of Gaussians Using the DDPM Objective*, arXiv:2307.01178. Gaussian sums: Sorenson & Alspach, *Recursive Bayesian Estimation Using Gaussian Sums*, Automatica 7(4), 1971. Smoothing and switching models: Rauch, Tung & Striebel, *Maximum likelihood estimates of linear dynamic systems*, AIAA J. 3(8), 1965; Ghahramani & Hinton, *Variational Learning for Switching State-Space Models*, Neural Comput. 12(4), 2000. Sums of squares: Blekherman, *Nonnegative Polynomials and Sums of Squares*, J. Amer. Math. Soc. 25(3), 2012. Skew-normal closure: Arellano-Valle & Azzalini, *On the Unification of Families of Skew-normal Distributions*, Scand. J. Statist. 33(3), 2006. All of these were checked against Crossref or arXiv (2026-10-31), for existence and metadata only, not for the specific results attributed to them. Pointers **not yet checked**: Minka (2001) on expectation propagation; van den Oord & Schrauwen (2014) on deep Gaussian mixtures; Naimark's dilation theorem; Arnold, Castillo & Sarabia on conditionally specified distributions. code: `examples/measurement_factor.jl` (the spectral baseline of §7)
>
> Bibliography: [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^lauritzen1989graphical|Lauritzen & Wermuth 1989]] · [[Bibliography#^lauritzen1996graphical|Lauritzen 1996]] · [[Bibliography#^lauritzen1992propagation|Lauritzen 1992]] · [[Bibliography#^han2018unsupervised|Han et al. 2018]] · [[Bibliography#^cheng2018information|Cheng et al. 2018]] · [[Bibliography#^loconte2023subtractive|Loconte et al. 2023]] · [[Bibliography#^marteauferey2020nonnegative|Marteau-Ferey et al. 2020]] · [[Bibliography#^tsuchida2023squared|Tsuchida et al. 2023]] · [[Bibliography#^viroli2019deep|Viroli & McLachlan 2019]] · [[Bibliography#^shah2023ddpm|Shah et al. 2023]] · [[Bibliography#^sorenson1971recursive|Sorenson & Alspach 1971]] · [[Bibliography#^rauch1965maximum|Rauch et al. 1965]] · [[Bibliography#^ghahramani2000variational|Ghahramani & Hinton 2000]] · [[Bibliography#^blekherman2012nonnegative|Blekherman 2012]] · [[Bibliography#^arellanovalle2006unification|Arellano-Valle & Azzalini 2006]] · [[Bibliography#^minka2001ep|Minka 2001]]
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
> Gaussian (possibly degenerate) joint, provided the constraints are consistent (a nonempty
> solution set) and the product is normalisable. All its marginals and conditionals are Gaussian,
> hence unimodal; no wiring, depth or choice of query creates a second mode.
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
| 3. non-Gaussian law on the nullspace | e.g. Laplace, Student-$t$ on $w$ | yes | scale mixtures: heavy tails, but unimodal for one factor; needs non-Gaussian beliefs (Gamma) |
| 4. mixture of constraints | $A_k z = b_k$, $k$ drawn | yes (§3, general case) | finite mixtures of subspaces |
| 5. nonlinearity of inference or of the objective | BP, Bethe, EP dynamics | — | **none** by itself (§9) |
| 6. half-space (orthant) constraints | $Az \ge b$: a truncated Gaussian | yes | skewness, truncation, polytopes; closed under conditioning in the unified skew-normal family (Arellano-Valle & Azzalini 2006); messages need $\Phi$ |

Candidate 1 (shared $A$, offsets $b_k$) is a special case of 4 (per-outcome $A_k$), not the
reverse (first version of this note had it backwards). Candidate 3 is a continuous mixture over
covariance scales, the limit of covariance switches $Q_k = Q/\tau_k$; one such factor is unimodal,
though compositions of several can have several modes. Candidate 6 was missing from the first
version (Gemini's review, §13). Among these, discrete outcomes (1, 4) are the minimal addition
that keeps every factor belief and every exact marginal in the finite conditional-Gaussian family,
with no special functions; 3 needs Gamma-type beliefs, 6 needs $\Phi$. That is the precise sense of
"minimal" here, made exact by the following.

> **Lemma (closure, proved here).** Let a factor with a discrete channel $k$ and continuous
> channels $z$ have potentials $\psi(k, \cdot)$. Its messages to the continuous channels, for every
> categorical message on $k$ and every Gaussian message on the others, are finite mixtures of
> (possibly improper, canonical) Gaussians if and only if each $\psi(k, \cdot)$ is a finite sum of
> exponentials of (possibly degenerate) quadratics in $z$.
>
> *Proof.* If so, every product with Gaussians and every Gaussian integral stays in that form, and the
> sum over $k$ is finite. Conversely, with $z$ the only continuous channel and $\kappa = \delta_k$, the
> message is $\psi(k, \cdot)$ itself, which must then be of that form. $\square$

So the switching factor, with one Gaussian site per outcome, is the general closed case up to
splitting outcomes. Three rivals sort themselves by the lemma (Mistral's review, §13): a **folding
factor** $\mathcal N(o;\, |a^\top z - c|, q)$ is piecewise, $\mathcal N(o; u, q)$ on one half-space of $u = a^\top z - c$
and $\mathcal N(o; -u, q)$ on the other, so it is *not* in the family: its integrals against Gaussians
need $\Phi$, as for half-space factors (candidate 6); the smooth symmetric version
$\tfrac12[\mathcal N(o; u, q) + \mathcal N(o; -u, q)]$ is the switching factor with $A_2 = -A_1$, $b_2 = -b_1$; an **unnormalised log-linear gate**
$\psi(k, z) = \exp(a_k^\top \tilde z)$ is a zero-precision Gaussian site, inside the family; a **normalised
softmax gate** $p(k \mid z) = \operatorname{softmax}_k(a_k^\top \tilde z)$ is outside it, because of its normaliser, the
same obstruction as Proposition 3.2. Candidate 5 is the distinction **F** asks for: it changes how the model is fitted
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
parallel copies of $\ker A$, with one shared covariance: only location-shift mixtures. The general
form gives $K$ different subspaces, and is **required** for non-parallel modes (lines through the
origin at different angles, the case the spectral toy fails and E1 tests, §7, §10) and for mixtures
with unequal covariances.

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
  where $\mathcal N^{-1}(\eta, \Lambda)$ denotes the unnormalised $\exp(\eta^\top z_t - \tfrac12 z_t^\top \Lambda z_t)$ and,
  since $\int \mathcal N(A_k z; b_k, Q_k) \prod_{i \ne t} \mathcal N(z_i; m_i, S_i)\, dz_{-t} = \mathcal N(A_{k,t} z_t;\, d_k, R_k)$,
  $$
  Z_k = (2\pi)^{-m/2} \det(R_k)^{-1/2} \exp\bigl(-\tfrac12 d_k^\top R_k^{-1} d_k\bigr).
  $$
  $Z_k$ depends on the **target** $t$ through $d_k$ and $R_k$, so it is recomputed per message, not
  cached per factor. **The per-outcome constants $Z_k$ must be kept**: they are the relative masses of the
  components, and dropping them is exactly the defect LP-019's reviews found three times in
  `_mixture_product` (constants of different sites do not cancel). Normalising each component in
  $z_t$ instead would change the weights by $\det(A_{k,t}^\top R_k^{-1} A_{k,t})^{-1/2}$, which does not
  exist when $A_{k,t}$ is rank-deficient.

  **Improper incoming beliefs.** The formulas assume proper $\mu_i$ for $i \ne t$. A flat direction of
  some $\mu_i$ makes $\int \psi \prod \mu_i$ diverge along it; if the direction is the same for all
  outcomes (it lies in $\ker A_{k,i}$ for every $k$, or the operators share their action on it), the
  divergence is a common factor and cancels in the weights. If it differs between outcomes, the
  relative weights $Z_k$ are undefined. So: proper incoming beliefs, or improper only in directions
  shared by all outcomes. Separately, when $A_{k,t}$ is rank-deficient each component of the outgoing
  message is an improper site: the message is still a well-defined function with well-defined
  coefficients in the unnormalised canonical form, but it cannot be collapsed or given an entropy
  until it is pooled with a proper belief on $z_t$.

Both are compositions of operations Mycelium already has: Gaussian messages, `CategoricalBelief`,
`MixtureBelief`, and overlaps (`combine_with_overlap`).

## 4. The statistical game (task B)

A potential admits two readings, and they have to be kept apart (the first version of this note
did not; ChatGPT's review, §13).

- **Graph reading.** The factor contributes $\psi$ to a product of factors, and the measure on its
  variables comes from the rest of the graph. Messages (§3) and the local Bethe term (Proposition
  4) are closed form. This is how `LinearConstraintFactor` works already, and it is all a graph needs.
- **Game reading, per polarity.** [[Scalar and Multivariate Energy]] §3 asks for an open model
  $c_p : X \multimap Y$, a normalised kernel. $\psi$ is flat along $\ker A_k$, so a kernel needs a
  **reference measure** $\nu$ on $Y \times U$ (a Gaussian):
  $$
  c_p(y, u, k \mid x) \;=\; \frac{\psi(k, z)\, \nu(y, u)}{Z_\nu(x)},
  \qquad Z_\nu(x) = \textstyle\sum_k \int \psi(k, z)\, \nu(dy, du),
  $$
  and $P(x, y, u, k) = P_X(x)\, c_p(y, u, k \mid x)$ is a proper conditional-Gaussian (CG) joint for
  any Gaussian $P_X$: Gaussian in $z$ for each $k$, $P(k, z) = \beta_k \mathcal N(z; m_k, S_k)$.

> **Proposition 3 (proved here; item 2 corrected 2026-10-31).** In the game reading, for every
> polarity $p = (X, Y, U)$ with $P_X(x) > 0$:
>
> 1. **Kernel** (with $S_{k,XX} \succ 0$ for every $k$ with $\beta_k > 0$). $c_p(y \mid x) = \sum_k P(k \mid x)\, \mathcal N\bigl(y;\, m_{k,Y} + S_{k,YX} S_{k,XX}^{-1}(x - m_{k,X}),\, S_{k,YY} - S_{k,YX}S_{k,XX}^{-1}S_{k,XY}\bigr)$
>    with $P(k \mid x) \propto \beta_k\, \mathcal N(x; m_{k,X}, S_{k,XX})$ ($U$ marginalised by dropping
>    its coordinates). If $k \in Y$ the kernel is the pair $\bigl(P(k \mid x), P(y \mid x, k)\bigr)$; if
>    $k \in X$ it is the Gaussian $P(y \mid x, k)$.
> 2. **Inversion.** For a prior $\pi$ on $X$,
>    $c'_\pi(x \mid y) \propto \pi(x)\, c_p(y \mid x) = \dfrac{\pi(x)}{P_X(x)}\, P(x, y)$.
>    It is a CG mixture in closed form when $\pi = P_X$ (it is then $P(x \mid y)$), or when $k \in X$ and
>    $\pi$ is Gaussian or a Gaussian mixture. **For a general Gaussian or mixture prior it is not**: the
>    ratio $\pi / P_X$ of a Gaussian by a mixture is not a finite mixture of Gaussians.
> 3. **Conditional mean.** $\mathbb E[y \mid x] = \sum_k P(k \mid x)\,(G_k x + h_k)$ with
>    $P(k \mid x) = \operatorname{softmax}_k\bigl(\log\beta_k - \tfrac12\lVert x - m_{k,X}\rVert^2_{S_{k,XX}^{-1}} - \tfrac12\log\det S_{k,XX}\bigr)$:
>    **nonlinear in $x$, although every component is linear.** With a shared $S_{XX}$ the gate is a
>    softmax of affine functions of $x$ (multinomial logistic gating).
> 4. **Compatibility.** All kernels are conditionals of $P$, so they are compatible (§1).
>
> *Proof.* (1) For fixed $k$, $(x, y)$ is jointly Gaussian under $P(\cdot \mid k)$, so $P(y \mid x, k)$ is
> the Gaussian conditional, and $P(k \mid x) \propto P(k) P(x \mid k)$. (2) is Bayes' rule with
> $c_p(y \mid x) = P(x, y)/P_X(x)$; for $\pi = P_X$ the ratio is 1, and for $k \in X$ the kernel given $k$ is
> Gaussian, so the product with a Gaussian (mixture) prior is a Gaussian (mixture). For general $\pi$,
> $1/P_X$ with $P_X$ a mixture of two or more Gaussians is not a finite sum of exponentials of
> quadratics. (3) is (1) with $G_k = S_{k,YX} S_{k,XX}^{-1}$, $h_k = m_{k,Y} - G_k m_{k,X}$. (4) by
> construction. $\square$

Measured (`examples/minimal_stochastic_factors_checks.jl`, 2026-10-31; a two-component
reference in one dimension, prior $\mathcal N(0.5, 4)$, $y = 0.7$): the first version's closed form,
which dropped $1/P_X$, is at $L^1$ distance 0.164 from the true inversion; with $\pi = P_X$ the closed
form agrees to $10^{-17}$.

**The two readings agree where it matters.** In a graph, the "prior" a factor is inverted against is
its cavity, and the factor belief is $b \propto \psi \prod_i \mu_{i \to f}$: the potential, never the
normalised kernel, so no $1/Z_\nu(x)$ appears and pooling a mixture prior with $\psi$'s messages is a
closed-form mixture. The non-closure of item 2 belongs to the per-factor game reading with an
arbitrary external prior.

**Typed energies** (the two levels of [[Scalar and Multivariate Energy]] §8):

| level | map | space | scalarisation |
|---|---|---|---|
| pointwise | $\mathbf l : X \times \llbracket c\rrbracket \times Y \to \mathbb R^m \oplus \mathbb R$, $\llbracket c\rrbracket = U \times \{1, \dots, K\}$, $\mathbf l(x, (u, k), y) = (A_k z - b_k,\; -\log w_k)$ | signed residual and selection cost | $\rho_k(r, s) = \tfrac12 r^\top Q_k^{-1} r + s \ge 0$, localised at the outcome ([[L-ADR-0007 Point-dependent scalarisations are localised before use]]) |
| free energy | $\mathbf F : \mathcal P X \times Y \to \mathbb R^3$, graded (fit, complexity, negentropy) | as `GaussianFactor` | identity on each summand (linear, so composition is strict) |

For a polarity, a prior $\pi$ and an observation $y$, with $q = c'_\pi(y)$ the selected inversion (a
distribution on $X \times U \times \{1, \dots, K\}$):

$$
\mathbf F(\pi, y) = \Bigl(\, \mathbb E_q\bigl[\tfrac12 r_k^\top Q_k^{-1} r_k\bigr],\;\;
\mathbb E_q\bigl[\tfrac12 \log\det(2\pi Q_k) - \log w_k\bigr],\;\; -H(q) \Bigr),
$$

so $\mathbf F$'s sum is $\mathbb E_q[\rho_k(\mathbf l)] + \mathbb E_q[\tfrac12\log\det 2\pi Q_k] - H(q)$: built
from the scalarised pointwise energy, as for every residual factor. The one-space reading,
$\sigma(\mathbb E_q[\mathbf l])$, would be wrong here, not merely lax: residuals of different outcomes
would cancel before squaring ($b_1 = 1$, $b_2 = -1$, equal weights, $A m_k = 0$ gives
$\mathbb E[r] = 0$ although neither constraint holds), and the outcome-dependent metric $Q_k^{-1}$ has no
meaning after $k$ is averaged out. The Jensen gap between the two readings contains the
within-outcome variance $\sum_k \beta_k \operatorname{tr}(Q_k^{-1} A_k S_k A_k^\top)$ and a between-outcome
variance of the residual. The normaliser $\tfrac12\log\det(2\pi Q_k)$ is in the complexity summand,
not in $\mathbf l$, as for `GaussianFactor` ([[Scalar and Multivariate Energy]] §8, "Normalised or not"). $H$ is relative to Lebesgue measure
on the continuous coordinates and counting measure on $k$
([[LP-006 Gaussian entropies are relative to Lebesgue measure]]). The energies do not depend on the
polarity ([[Scalar and Multivariate Energy]] §8, "Per polarity"); the inversion does.

**The local term.** For a CG factor belief $b(k, z) = \beta_k \mathcal N(z; m_k, S_k)$ (the Bethe belief
has this form):

$$
\begin{aligned}
\mathrm{fit} &= \textstyle\sum_k \beta_k \Bigl[\tfrac12 (A_k m_k - b_k)^\top Q_k^{-1}(A_k m_k - b_k) + \tfrac12\operatorname{tr}\bigl(Q_k^{-1} A_k S_k A_k^\top\bigr)\Bigr], \\
\mathrm{complexity} &= \textstyle\sum_k \beta_k \bigl[\tfrac12 \log\det(2\pi Q_k) - \log w_k\bigr], \\
\mathrm{negentropy} &= -H(b) = -H(\beta) - \textstyle\sum_k \beta_k\, H\bigl(\mathcal N(m_k, S_k)\bigr).
\end{aligned}
$$

> **Proposition 4 (proved here; statement corrected 2026-10-31).** The sum is the local variational
> functional $\mathcal F_\psi(b) = \mathbb E_b[-\log \psi] - H(b)$, every term in closed form. It is a KL
> divergence only when $\psi$ is normalisable: if $Z_\psi = \sum_k \int \psi\, dz < \infty$ (e.g. every
> $A_k$ invertible), $\mathcal F_\psi(b) = \mathrm{KL}(b \,\|\, \psi / Z_\psi) - \log Z_\psi \ge -\log Z_\psi$.
> For rank-deficient $A_k$, $Z_\psi = \infty$, $\psi$ is a potential and not a density, and
> $\mathcal F_\psi(b)$ is finite whenever $b$ is proper with finite entropy; the normalisation then
> comes from the graph, whose Bethe sum is $-\log Z$ of the whole (proper) model on a tree.
>
> *Proof.* $-\log \psi(k, z) = \tfrac12 r_k^\top Q_k^{-1} r_k + \tfrac12 \log\det(2\pi Q_k) - \log w_k$, and
> $\mathbb E_{\mathcal N(m_k, S_k)}[\tfrac12 r_k^\top Q_k^{-1} r_k]$ is the bracket in "fit" (mean plus
> trace). So the sum is $\mathbb E_b[-\log \psi] - H(b)$, and the entropy of a CG belief with $k$ explicit
> is $H(\beta) + \sum_k \beta_k H(\cdot \mid k)$ (chain rule of entropy). With $Z_\psi < \infty$, write
> $-\log\psi = -\log(\psi/Z_\psi) - \log Z_\psi$ and apply Gibbs' inequality. $\square$

The functional can be negative when $\psi$ is not normalisable: for $n = 2$,
$\psi = \mathcal N(z_1; 0, 1)$ and $b = \mathcal N(0, I_2)$ it is $(\tfrac12 + \tfrac12\log 2\pi) - \log(2\pi e) \approx -1.42$
(Mistral's review). It is a Bethe factor term, not a divergence.

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
gradients in $(m_k, S_k)$ and hence in $\theta$, **conditionally on $k$**. When a loss depends on a
sampled $k$ whose gate depends on sampled continuous variables (generation through the reverse chain
of §8, or a $k$ driven by an upstream sample), the selection step is not reparameterisable and
needs the score-function estimator; the learning rule of §8 avoids this by summing over $k$.

**Conditions on the envelope argument.** It needs a finite, differentiable Bethe free energy at the
fixed point: $Q_k \succ 0$ and proper beliefs (with learnable $Q_k$ and degenerate beliefs the objective
is unbounded below at $S = 0$). On loopy graphs or with collapsed mixture beliefs, BP fixed points are
stationary points of the approximate objective, so the gradients are exact for that objective, which
is biased (§9.2).

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

> **Proposition 5 (proved here; consequence of item 1 added 2026-10-31).**
> 1. $p(z) = \mathcal N(z)\,\lVert \tilde z \rVert^2 / \mathcal Z$ and $p(k \mid z) = \tilde z^\top E_k \tilde z / \lVert \tilde z \rVert^2$:
>    **the Born rule**, independent of $m$ and $S$. **The marginal $p(z)$ does not depend on the
>    POVM at all**: with the outcome unobserved and uncoupled, learning $\{E_k\}$ cannot change the
>    distribution of $z$.
> 2. Conditioning on part of $z$: $p(k \mid x) = \mathbb E[\tilde z^\top E_k \tilde z \mid x] / \mathbb E[\lVert \tilde z\rVert^2 \mid x]$ under
>    $\mathcal N(y \mid x)$, a ratio of closed-form quadratic moments, and $p(y \mid x, k) \propto
>    \mathcal N(y \mid x)\, \tilde z^\top E_k \tilde z$, a Gaussian times a quadratic: **not Gaussian**.
> 3. For a rank-one element $E_k = v v^\top$, $p(z \mid k)$ vanishes on the hyperplane
>    $v^\top \tilde z = 0$; when the hyperplane passes near the reference mean, it is bimodal across
>    it. No finite Gaussian mixture has an exact zero. These zeros are in the **conditionals given an
>    outcome**, not in $p(z)$.
> 3′. A factor on the outcome, weights $w_k \ge 0$, gives $p(z) \propto \mathcal N(z)\, \tilde z^\top M \tilde z$ with
>    $M = \sum_k w_k E_k$: now the POVM matters, but only through one positive semidefinite $M$. If every
>    $w_k > 0$, then $u^\top M u \ge (\min_k w_k)\lVert u\rVert^2$ and $M \succ 0$: **no zeros**. Zeros need some
>    outcome weights to be exactly zero (outcomes eliminated, i.e. conditioning). The zero set is then the
>    affine subspace $\{z : M\tilde z = 0\}$ (a hyperplane for $M = vv^\top$, a point for
>    $M = \operatorname{diag}(0, I)$, giving a ring-shaped density with a zero at the centre, empty for
>    a generic $M$ of higher rank), never a curve such as a circle.
>
>    Since $\sum_k w_k E_k$ ranges over all positive semidefinite $M$ (up to scale) as the POVM and the
>    weights vary, a weighted Born factor is, for the marginal, exactly an **unnormalised quadratic-tilt
>    potential** $\mathcal N(z)\,\tilde z^\top M \tilde z$ with free $M \succeq 0$: the measurement reading
>    adds nothing to $p(z)$ beyond such a tilt.
> 4. All polarities' kernels are conditionals of $p$, so they are compatible (§1).
>
> *Proof.* $\sum_k \tilde z^\top E_k \tilde z = \lVert\tilde z\rVert^2$ and
> $\mathbb E_{\mathcal N}\lVert \tilde z\rVert^2 = 1 + \operatorname{tr}(S + mm^\top)$ give (1). (2) is
> Bayes' rule with Gaussian moments $\mathbb E[\tilde z^\top E \tilde z \mid x] = \bar{\tilde z}^\top E \bar{\tilde z} + \operatorname{tr}(E\,\widetilde{\operatorname{Cov}})$.
> (3): $(v^\top \tilde z)^2 = 0$ exactly on the hyperplane, positive elsewhere; a Gaussian mixture is
> positive everywhere. (4) by construction. $\square$

Measured (`examples/minimal_stochastic_factors_checks.jl`): two different POVMs give the same $p(z)$ to $2\cdot10^{-16}$; with
weights $(1, 0)$ the density at the predicted zero is $-6\cdot10^{-18}$.

**What it represents.** As a marginal generative model the Born joint is therefore *thinner* than
the switching factor: one Gaussian times positive semidefinite quadratic tilts, unimodal or nearly
so, against finite mixtures. Its distinctive capacity is in queries that involve the outcome:
$p(z \mid k)$ with exact zeros, and $p(k \mid x)$. With outcomes unobserved and uncoupled, composing Born factors gives
$\mathcal N(z)\lVert\tilde z\rVert^{2L}$, whatever the POVMs: nothing is learned about $p(z)$. The capacity
comes from **observed, weighted or coupled outcomes**: with weights, $L$ factors on shared variables give
$\mathcal N(z) \prod_l \tilde z^\top M_l \tilde z$, products of positive semidefinite quadratic forms (with independent weights on the outcomes the sum
over configurations factorises, $\sum_{k_1 \dots k_L} \prod_l \tilde z^\top E_{l,k_l}\tilde z\, w_{l,k_l} = \prod_l \tilde z^\top M_l \tilde z$;
positive *sums* of products need coupled outcome weights), i.e.
polynomials of degree $2L$ times a Gaussian, nonnegative by construction, with normalisers given by
Gaussian moments of degree $2L$ (Isserlis), whose cost grows combinatorially in $L$. This differs from
a Born machine, whose learned amplitudes set the data distribution itself.

> **Proposition 6 (proved here, for amplitudes).** Densities $P(z)^2\, \mathcal N(z)/\mathcal Z$
> with $P$ a polynomial are dense in $L^1$ among all densities on $\mathbb R^n$.
>
> *Proof.* For a density $p$, $f = \sqrt{p / \mathcal N}$ satisfies
> $\int f^2\, d\mathcal N = \int p = 1$, so $f \in L^2(\mathcal N)$. Hermite polynomials are complete in
> $L^2(\mathcal N)$, so polynomials $P_d \to f$ in $L^2(\mathcal N)$. Then
> $\lVert P_d^2 \mathcal N - p\rVert_{L^1} = \int |P_d^2 - f^2|\, d\mathcal N \le \lVert P_d - f\rVert_{L^2(\mathcal N)} \lVert P_d + f\rVert_{L^2(\mathcal N)} \to 0$,
> and normalising changes the limit by a factor tending to 1. $\square$

The catch is the word *amplitudes*: Proposition 6 needs an arbitrary polynomial $P$ **before**
squaring, i.e. amplitudes that add coherently and are squared once (a Born machine; squared and
subtractive mixtures are the same idea, Loconte et al.). A factor graph multiplies *probabilities*;
weighted Born factors give products of positive semidefinite quadratic forms, and summing over
outcomes gives positive sums of such products. What that class can do:

> **Proposition 6′ (proved here, from Gemini's review).**
> 1. In one dimension, products of positive semidefinite quadratics $\tilde z^\top E \tilde z$ give every
>    nonnegative polynomial (each factors into nonnegative real quadratics, by the fundamental theorem
>    of algebra), so with Proposition 6 they are dense in $L^1(\mathbb R)$.
> 2. In two or more dimensions, a positive sum of products of positive semidefinite quadratics never
>    vanishes exactly on a curved algebraic set such as the circle $z_1^2 + z_2^2 = 1$, whereas
>    $P(z)^2\mathcal N(z)$ with $P = z_1^2 + z_2^2 - 1$ does.
> 3. One Born factor alone is degree 2 however large $K$ is: $\sum_k w_k \tilde z^\top E_k \tilde z = \tilde z^\top M \tilde z$.
>
> *Proof.* (1) A nonnegative univariate polynomial has real roots of even multiplicity and complex
> roots in conjugate pairs, so it is a product of factors $(z - a)^2 + c^2$, each $\tilde z^\top E \tilde z$
> with $E \succeq 0$. (2) A positive sum vanishes on the circle only if every term does; a product of
> finitely many factors vanishes at infinitely many points of the circle only if one factor does; the
> zero set of $\tilde z^\top M \tilde z$ with $M \succeq 0$, $M \ne 0$ is the affine subspace $\{M\tilde z = 0\}$,
> which meets a circle in at most two points. (3) is linearity in $E$. $\square$

Whether positive sums of products of positive semidefinite quadratics are dense in $L^1(\mathbb R^n)$
for $n \ge 2$ as the number of factors grows remains **open** (the earlier "expected" was not
earned): known gaps between nonnegative polynomials and sums of squares in several variables
(Blekherman 2012) are a reason for caution, though density is weaker than exact representation. (2) only excludes exact
zeros on curves at finite size. Results on positive semidefinite models with nonlinear features
(Marteau-Ferey, Bach & Rudi; Tsuchida, Ong & Sejdinovic) are the place to look, but they use feature
maps, not linear amplitudes, and are not checked for this case. So the genuinely quantum ingredient,
interference, needs amplitude-level composition, which Lenticulum's probability-level products do
not provide.

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
3. in practice, **small steps**, so that each true reverse conditional is close to the simple family
   used for $p_\theta$: the true reverse conditional is nearly Gaussian but with a **mean that is
   nonlinear in $z_t$** (it involves the score $\nabla \log q_t$). This is a design guideline about the
   tightness of the bound, not part of the definition: a few large steps are still a diffusion;
4. **ancestral sampling** from the prior through the reverse chain;
5. in the modern sense, **one time-conditioned model shared across steps**, not free parameters per
   step. Without it, the construction below is a reverse-time switching linear system trained with a
   diffusion-style bound, not a diffusion model in the parameterisation sense (Mistral's review, §13).

Requirement 1 is free here: it is a chain of `GaussianFactor`s ([[The Linear Gaussian Chain]]).
Requirement 3 is where a linear factor fails: a Gaussian reverse kernel with linear mean is exact
only for Gaussian data. Requirement 2's objective is the chain's variational free energy with the
**forward process as the fixed inversion**, so its entropies are those of the Gaussian forward
process, in closed form; the Bethe variable-entropy obstacle of §4 does not arise (this is an upper
bound, like `VariationalDiffusion`'s `FreeEnergyUpperBound`, not an exact free energy).

> **Proposition 8 (elementary, proved here; per component it is Gaussian smoothing as in Rauch, Tung
> & Striebel, and the mixture form is standard for switching linear models).** If the data are a $K$-component Gaussian mixture, every true
> reverse conditional $q(z_{t-1} \mid z_t)$ of the forward process 1 is a switching linear kernel
> with $K$ outcomes:
> $$
> q(z_{t-1} \mid z_t) = \sum_k q(k \mid z_t)\, \mathcal N\bigl(z_{t-1};\; G_{k,t} z_t + h_{k,t},\; C_{k,t}\bigr),
> \qquad q(k \mid z_t) \propto \pi_k\, \mathcal N\bigl(z_t;\; \alpha_t \mu_k,\; \alpha_t^2 \Sigma_k + \sigma_t^2 I\bigr),
> $$
> with $\alpha_t^2 = \prod_{s \le t}(1-\beta_s)$, $\sigma_t^2 = 1 - \alpha_t^2$. (The softmax-gated
> piecewise-linear form of the mixture's reverse kernel and score is also used by Shah, Chen &
> Klivans; not checked in detail.)
>
> *Proof.* Given the component $k$, $(z_0, z_{t-1}, z_t)$ is jointly Gaussian (a linear-Gaussian
> chain started at $\mathcal N(\mu_k, \Sigma_k)$), so $q(z_{t-1} \mid z_t, k)$ is Gaussian with mean
> linear in $z_t$. Summing over $k$ with $q(k \mid z_t) \propto \pi_k q(z_t \mid k)$ gives the formula,
> and $q(z_t \mid k) = \mathcal N(\alpha_t \mu_k, \alpha_t^2 \Sigma_k + \sigma_t^2 I)$. $\square$

So the switching factor is the **exact** reverse kernel for mixture data, and its gate
$q(k \mid z_t)$ is the measurement of §3. Its parameters are closed-form functions of
$(\alpha_t, \sigma_t, \mu_k, \Sigma_k)$, e.g. $\operatorname{Cov}(z_{t-1}, z_t \mid k) = \alpha_{t-1}\alpha_t \Sigma_k + (\alpha_t/\alpha_{t-1})\,\sigma_{t-1}^2 I$,
so a **shared, time-conditioned parameterisation** (learn $\pi_k, \mu_k, \Sigma_k$ once, derive every step's
kernel from the schedule) is available for free and meets requirement 5. Two consequences, one deflating, one constructive:

- **For mixture data diffusion is unnecessary.** One switching factor $k \to z_0$ is already the
  exact model, sampled in one shot. Diffusion earns its keep, in likelihood terms, only when the data
  are not in a tractable family; its value for mixture data lies in optimisation and sample
  complexity (fitting by denoising instead of EM), which this note does not measure.
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

**Directed kernels, not potentials (Gemini's review, §13).** The reverse step must be a
*normalised conditional*,
$$
p_\theta(z_{t-1} \mid z_t) = \frac{\sum_{k} \psi_t(k, z_{t-1}, z_t)}{Z_t(z_t)},
\qquad Z_t(z_t) = \sum_k \int \psi_t(k, z_{t-1}, z_t)\, dz_{t-1},
$$
the game reading of §4. A product of switching *potentials* is a different model. If
$\psi_t$ constrains only $z_{t-1} - G_k z_t$, then $Z_t(z_t) = \sum_k w_k$ is constant and the gate cannot
depend on $z_t$; if it also constrains $z_t$ (so that the gate does), $Z_t$ is a mixture in $z_t$, the
potentials' product couples all steps, and sampling each step from its local conditional does not
sample the product. So the diffusion reading needs the directed kernel, with its $1/Z_t(z_t)$:
ancestral sampling and the per-sample learning objective below stay closed form ($Z_t(z_t)$ is a
Gaussian mixture in $z_t$), while backward messages through the chain, which conditioning on part of
$z_0$ (inpainting) needs, inherit the non-closure of Proposition 3.2. That is the price of the
directed reading, and the trade-off between the two is a design decision for E4.

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
| E1 | does the switching factor fix the spectral toy's failure? | lines at 0° and 60° (the toy's case 2) | spectral (toy), switching $K = 2$ in the general form (per-outcome $A_k$; the shared form gives only parallel lines), Born $K = 2$ rank-one | true conditional; held-out $\log p(y \mid x)$, line angles |
| E2 | what can only the Born joint do? | $p(z) \propto (v^\top \tilde z)^2 \mathcal N(z)$ (exact zero on a line); $\lVert z\rVert^2\mathcal N(z)$ (a ring, zero at the centre); a zero on a circle, which needs degree 4 at the amplitude level | Born with $K = 2$ and outcome weights (not $K = 1$, where $E_1 = I$); switching with growing $K$ | exact density; held-out $\log p$ vs number of parameters |
| E3 | where does composition beat a flat mixture? | a $2^L$ grid of modes (product structure); 8 modes on a ring (no product structure) | $L$ binary switching factors, each a rank-one constraint $a_l^\top z = b_{l,k_l}$ or composed additively ($z = \sum_l u_l$); one factor with $2^L$ outcomes. Full-rank factors on a shared $z$ would suppress disagreeing configurations by their overlap $Z_{k_1 \dots k_L}$ | exact mixture likelihood; $\log p$ vs parameters |
| E4 | do many small switching steps act like a diffusion? | (a) a Gaussian mixture; (b) two moons and a ring | reverse chain of $T$ steps, $K_t \in \{1, 2, 3\}$, trained as in §8 | (a) Proposition 8's exact kernels; (b) held-out $\log p$ by exact path sums for small $T$; vs one factor of $\prod_t K_t$ outcomes |
| E5 | are the polarities compatible in practice? | E1 and E3 data | spectral vs switching vs Born | conditional $\log p$ in three query directions against the joint's conditionals |
| E6 | how much does Bethe/EP collapsing bias learning? | small trees where exact mixture marginals are enumerable | exact sum vs collapsed beliefs vs entropy bounds | parameter error and $\log p$ gap; in particular whether learning with collapsed beliefs on a symmetric bimodal posterior merges the outcomes into one wide mode (mode covering) |

**E1, measured (2026-10-31; `examples/switching_factor_e1.jl`, `.md`).** On lines at 0° and 60° the
general-form switching factor recovers both lines (errors 0.015° and 0.001°, $q_k$ = the noise
variance), and its held-out $\log p(y \mid x)$ is 0.702, equal to the true model's and to a free
two-component Gaussian mixture's (with 6 parameters against 11), where the spectral measurement gets
−0.939 and one Gaussian −0.823. Message passing in a Mycelium graph reproduces the direct conditional
to $5\cdot10^{-15}$, which checks §3's messages with their per-outcome constants. Two of four starts reach
the optimum. The easy case: the truth is in the class.

Order: E1 and E4(a) first (each has an exact oracle and decides whether the building block and the
diffusion reading are right), then E3 and E4(b) (the expressivity questions), then E2 (whether the
Born joint is worth a belief type), E5 and E6.

Conditions for all experiments (Mistral's review, §13):

- **Noise floors.** With learned $Q_k$ on noise-free data the likelihood is unbounded as $Q_k \to 0$; use
  an eigenvalue floor or a prior on $Q_k$ ([[LP-005 Learning the noise covariance]]), else held-out
  $\log p$ is dominated by collapsed components.
- **Local optima.** EM for $K = 2$ can fail for optimisation, not class, reasons: several restarts,
  or an exact grid over the line angles in E1.
- **Controls.** E1 adds an unconstrained Gaussian-mixture EM baseline; E4 adds the ablation shared
  time-conditioned (§8) against free per-step parameters, which is what tests the diffusion claim, and
  uses data with many more components than $\prod_t K_t$ at matched parameter counts; E5 adds a test of
  inversion against an external prior (Proposition 3.2).
- **E2 metric.** Held-out $\log p$ excluding the measure-zero zero set, with near-zero behaviour
  reported separately. E2 needs the Born belief type, so E1's Born arm waits for it too.

## 11. Proposed interface (nothing implemented)

`SwitchingConstraintFactor(dims, K, m; shared = true)`: channels as `LinearConstraintFactor` plus
`:k`; `ps = (A, b = [b₁ … b_K], logw, noise)`; messages as in §3 (to `:k` a `CategoricalBelief`, to a
continuous channel a `MixtureBelief` with the per-outcome constants kept); `local_free_energy` as in
Proposition 4, graded `(fit, complexity, negentropy)`; `supported_polarities`: every split of the
continuous channels with one target, with `:k` observed, latent or the target; `shared = true` only
for location-shift mixtures. The reverse chain of E4 needs a second, **directed** variant, a
switching linear kernel normalised by $Z_t(z_t)$ (§8), not this potential. The Born joint would need a new belief type
(Gaussian times a nonnegative quadratic) and is deferred until E2 says it is worth one.

## 12. Open

- Whether positive sums of products of positive semidefinite quadratics times a Gaussian are dense
  in $L^1(\mathbb R^n)$ for $n \ge 2$ (§6; true for $n = 1$, Proposition 6′.1; exact zeros on curves are
  impossible, 6′.2).
- The size of the advantage of $T$ small switching steps over one large mixture (E4(b)): a bound in
  terms of the data's smoothness would turn the diffusion hypothesis of §8 into a theorem.
- The learning bias of collapsed (EP) beliefs in hybrid graphs (E6).
- Hard constraints with outcome-dependent rank: a reference measure under which entropies of
  different-dimensional solution sets compare (§5).

## 13. Reviews

**ChatGPT, 2026-10-31** (read the branch; could not run code). Claude checked the two main
claims numerically (`examples/minimal_stochastic_factors_checks.jl`) and corrected the note:

1. Proposition 3's inversion claim was too strong: against a prior other than the reference marginal
   the inversion carries $\pi / P_X$ and is not a mixture ($L^1$ error 0.164 of the old closed form).
   Corrected, with the graph reading (§4) where it does not arise.
2. The Born marginal $p(z)$ does not depend on the POVM when the outcome is unobserved, so learning
   the POVM alone changes nothing about $z$, and E2 with $K = 1$ ($E_1 = I$) was vacuous; an annular
   zero set is out of reach of one positive semidefinite quadratic. Corrected in §6 and §10.
3. §4 mixed the pointwise and free-energy levels and called $\mathbb E_b[-\log\psi] - H(b)$ a KL divergence
   although $\psi$ is not normalisable. Retyped (table in §4) and Proposition 4 restated.
4. Its strongest positive point, that nonlinearity arises from conditioning on discrete alternatives
   ($\mathbb E[y \mid x]$ softmax-gated linear), is now Proposition 3.3.

It also noted that the distribution family is established (conditional-Gaussian graphical models,
switching dynamical systems): any novelty lies in arbitrary polarity, compositional statistical games,
the multivariate free energy and efficient message passing, not in the family; and that exponential
inference cost (§9) is the main computational risk.

**Gemini, 2026-10-31** (reviewed the first version, from a self-contained copy; could not run
code). It found independently the inversion error (Proposition 3.2), the vacuous $K = 1$ Born test and
the KL misstatement (all corrected above), and in addition:

1. Chaining switching potentials does not give a directed reverse chain: a $z_t$-dependent gate needs
   the normalised kernel with $1/Z_t(z_t)$, so E4 needs a directed variant of the factor (§8, §11).
2. Positive compositions of Born factors cannot vanish on curved sets, and are dense in one dimension
   (Proposition 6′, replacing the earlier speculation; its suggested $L^1$-density argument for
   $n \ge 2$ did not hold up, since $1 - \lVert z - \mu\rVert^2/2L$ is not positive semidefinite, so that case
   stays open).
3. $Z_k$ made explicit, and the restriction on improper incoming beliefs (§3).
4. Why the one-space reading of the energy fails for this factor (§4).
5. The shared form cannot represent non-parallel modes, so E1 needs the general form (§3, §10).
6. Two missing or mislabelled candidates: half-space factors, and scale mixtures being unimodal
   (§2); and fixes to E3 (overlap penalty of full-rank factors on a shared variable) and E6.

Its references were checked against Crossref and arXiv before use: ten are in `refs.bib` now; two
had wrong details (the Loconte et al. title, and the authors of *Squared Neural Families*), and the
section and theorem numbers it quoted were not checked and are not used.

**Mistral, 2026-10-31** (reviewed the first version; ran its own pure-Python checks: Monte Carlo
and quadrature agreement to 3–4 digits on the messages, Propositions 4, 5 and 8). It found the
inversion error, the Born-marginal degeneracy and the KL misstatement independently, and in
addition: zeros of a weighted Born factor need some outcome weights to be exactly zero, and the
weighted Born factor is, for the marginal, just a quadratic-tilt potential (§6); a closure criterion
for minimality, with the folding and log-linear rivals (§2; its claim that a log-linear gate leaves
the family holds only for the normalised softmax, and its folding factor is piecewise, so not a
switching factor, only its symmetrised version is); $Z_k$ depends on the target (§3; its claim that
improper components make relative weights meaningless does not hold in the unnormalised canonical
form, only collapsing is unavailable); a negative value of the local functional (§4); the
conditionality of pathwise gradients and of the envelope argument (§5); requirement 5, the closed-form
time-conditioned parameterisation, and Proposition 8's provenance (§8); a caution on the
higher-dimensional density question (§6); consistency and regularity hypotheses (Propositions 2, 3);
and conditions and controls for the experiments (§10). Its references were checked: Sorenson &
Alspach, Ghahramani & Hinton and Blekherman are added to `refs.bib` (Rauch, Tung & Striebel was
already there); two of its details were wrong (Sorenson & Alspach is *Automatica* 7(4), not 7(6);
Rauch, Tung & Striebel is *AIAA J.* 3(8), not 3(12)).

Related: [[Scalar and Multivariate Energy]], [[Bethe Free Energy]], [[Hierarchical Models]], [[The Linear Gaussian Chain]], [[Depth in Implicit Learning]], [[Universal Approximation for Implicit Diffusion Learners]], [[RQ-001 One Potential, Every Polarity]], [[L-SPEC-003 Exposed Parameters and Energy Channels]], [[The Diffusion Family]]
