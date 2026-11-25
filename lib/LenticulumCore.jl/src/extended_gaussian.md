#implementation

> `ExtendedGaussian(μ, Σ, D)` (`extended_gaussian.jl`): a Gaussian made flat along a subspace,
> the belief type for linear relations. Three kinds of direction in one object: **flat** (`D`, a
> likelihood that says nothing there), **hard** (`null(Σ)` outside `D`, a linear constraint
> imposed exactly) and **Gaussian**. Pushforward through affine maps and the product are exact;
> `GaussianBelief`, `DiracBelief` and `TrivialBelief` are special cases and convert to it.

> Sources: Stein & Samuelson, *A Category for unifying Gaussian Probability and Nondeterminism*, CALCO 2023 (arXiv:2204.14024: subspaces $D \subseteq X$ admitted as generalised uniform distributions); Stein & Samuelson, *A Categorical Treatment of Open Linear Systems*, LMCS 21(3), 2025 (arXiv:2403.03934: extended Gaussian distributions as Willems' Gaussian linear systems). code: `extended_gaussian.jl`; Mycelium's `combine` methods in `lib/Mycelium.jl/src/gaussian.jl`; first consumer `_improper_split` in `lib/Lenticulum.jl/src/gaussian.jl`. tests: `test/extended_gaussian.jl`, Mycelium `test/runtests.jl` ("combine with extended Gaussians").
>
> Bibliography: [[Bibliography#^stein2023gaussian|Stein & Samuelson 2023]] · [[Bibliography#^stein2025openlinear|Stein & Samuelson 2025]]
>
> Theory (CT-ML wiki): [Gaussian Relations](https://mathstruct.org/CategoryTheory-ML-Wiki/Gaussian-Relations)

## Theory

An extended Gaussian $\mathcal N(\mu, \Sigma) + D$ on $\mathbb R^n$ is the law of $\mu + \varepsilon + d$
with $\varepsilon \sim \mathcal N(0, \Sigma)$ and $d$ "uniform" on the subspace $D$: a Gaussian on the
quotient $\mathbb R^n / D$. It is the limit of the proper Gaussians with covariance
$\Sigma + \tau P_D$ as $\tau \to \infty$ ($P_D$ the orthogonal projector onto $D$). Allowing $\Sigma$ to
be singular adds the other limit: along $\operatorname{null}(\Sigma) \cap D^\perp$ there is no
variance, and the belief imposes $H^\top x = H^\top \mu$ exactly ($H$ a basis of those **hard**
directions).

Every belief the Gaussian code handles is one of these:

| belief | $\Sigma$ | $D$ |
|---|---|---|
| proper `GaussianBelief` | $\Lambda^{-1}$, full rank | $\{0\}$ |
| improper `GaussianBelief` (a likelihood flat along $\operatorname{null}\Lambda$) | $\Lambda^+$ | $\operatorname{null}\Lambda$ |
| `DiracBelief(v)` | $0$ (every direction hard) | $\{0\}$ |
| `TrivialBelief()` | $0$ | $\mathbb R^n$ |
| a hard linear constraint $Cx = c$, nothing else | $0$ on $\operatorname{range} C^\top$ | $\operatorname{null} C$ |

The canonical form $(\eta, \Lambda)$ covers the first two and the zero precision; it cannot hold a
hard direction (infinite precision). The moment form $(\mu, \Sigma)$ covers hard directions and not
flat ones. The extended form holds both, which is what a factor graph of linear relations needs:
a relation that constrains $k$ of $n$ coordinates sends a message flat in $n - k$ directions, and a
noise-free relation imposes its $k$ directions exactly.

**Pushforward.** For $y = Mx + c + \varepsilon'$ with $\varepsilon' \sim \mathcal N(0, Q)$ independent,
the law of $y$ is $\mathcal N(M\mu + c, M\Sigma M^\top + Q) + M D$. The image of a uniform law on
$D$ is uniform on $MD$, so this is exact for every matrix $M$, rectangular or rank-deficient.
Marginalising onto coordinates is the pushforward through the selection matrix.

**Preimage.** For a belief $b$ about $y = Mx + c$, the likelihood $x \mapsto b(Mx + c)$ is again an
extended Gaussian on $x$: hard where $Mx + c$ would have to lie in a hard direction of $b$
($H^\top M x = H^\top(\mu - c)$), Gaussian with precision $M^\top\Sigma^+M$ and information
$M^\top\Sigma^+(\mu - c)$, flat along $\operatorname{null}(M)$ and the preimage of $b$'s flat directions. With
the pushforward it gives the message of every linear relation (`HardConstraintFactor`,
`lib/Lenticulum.jl/src/hard_constraint.md`).

**Product.** Two extended Gaussians as potentials on $\mathbb R^n$: each is supported on an affine
subspace (its hard constraints) and, there, is $\exp(-\tfrac12 (x-\mu)^\top \Sigma^+ (x-\mu))$,
constant along $D$. The product is supported on the intersection: stack the constraints into
$Cx = c$, take a particular solution $x_0$ (refusing the product when none exists: contradictory
constraints) and an orthonormal basis $N$ of $\operatorname{null} C$, and write $x = x_0 + Nz$. The
precisions add, $\Lambda = \Sigma_a^+ + \Sigma_b^+$, $\eta = \Sigma_a^+\mu_a + \Sigma_b^+\mu_b$, and in $z$
the quadratic part is $\Lambda_z = N^\top \Lambda N$, $\eta_z = N^\top(\eta - \Lambda x_0)$. Along
$\operatorname{null}\Lambda_z$ the product is flat; on the rest it is Gaussian with mean $\Lambda_z^+\eta_z$
and covariance $\Lambda_z^+$. Mapped back: $\mu = x_0 + N\Lambda_z^+\eta_z$, $\Sigma = N\Lambda_z^+N^\top$,
$D = N\operatorname{null}(\Lambda_z)$. No component of $\eta_z$ can lie along $\operatorname{null}\Lambda_z$: a vector
$v$ there has $\Lambda_a N v = \Lambda_b N v = 0$ (both are positive semidefinite), so
$\eta_z^\top v = \mu_a^\top\Lambda_a Nv + \mu_b^\top\Lambda_b Nv - x_0^\top\Lambda N v = 0$. The product of two
extended Gaussians is therefore always one, or a contradiction.

## Implementation

- **Normalised representative.** The constructor orthonormalises $D$ (an SVD, dependent columns
  dropped) and projects $\mu$ and $\Sigma$ onto $D^\perp$, since nothing along $D$ is observable.
  Two representatives of one law then agree up to the basis of $D$; the tests compare $P_D$.
  An internal constructor, `ExtendedGaussian(Val(:normalised), μ, Σ, D)`, stores parts that are
  already normalised without checking them: Lenticulum's plate method of `HardConstraintFactor`
  (P6) builds each replica's message from replica 1's $\Sigma$ and $D$ and a mean it projects onto
  $D^\perp$ itself (`hard_constraint.md`, "Plates"). Nothing else uses it.
- **Tolerances.** An eigenvalue or singular value below $16\,n\,\epsilon\,\lambda_{\max}$ counts as zero, the
  rule `_split_moments` used before. So a direction with variance below about $10^{-14}$ of the
  largest is hard: a deliberate choice, since a covariance that small is a constraint in all but
  name, and treating it as Gaussian is what made the rank-deficient messages of LP-024 wrong.
  Contradictory constraints are those whose least-squares residual exceeds $\sqrt\epsilon$ times the
  scale of the means.
- **Conversion.** `extended(::GaussianBelief)` splits by the eigendecomposition of $\Lambda$ (and
  refuses $\Lambda$ with a negative eigenvalue beyond rounding, or $\eta$ along $\operatorname{null}\Lambda$:
  $\exp(\eta^\top x)$ along a flat direction is no limit of Gaussians). `GaussianBelief(::ExtendedGaussian)`
  goes back when there is no hard direction, declaring the result improper when $D \ne \{0\}$.
- **One routine for products and preimages.** `_from_potential(C, c, η, Λ)` builds the extended
  Gaussian of $[Cx = c]\cdot\exp(\eta^\top x - \tfrac12 x^\top\Lambda x)$; the product and `preimage` both reduce
  to it. `_independent_join` stacks independent beliefs (means, block-diagonal covariances and
  flat directions).
- **Damping** (Mycelium's `damp`) is the geometric mixture $a^\alpha b^{1-\alpha}$, the rule canonical
  damping of a `GaussianBelief` follows: `_conjoin` with weights, precisions scaled. Hard constraints
  of the two sides on one direction are not a contradiction there: with `strict = false` their
  values are the weighted least-squares fit, the limit of precisions $\alpha/\varepsilon$ and $(1-\alpha)/\varepsilon$, so a
  hard value moves by the damping weight like any other.
- **Moments.** `belief_mean` and `belief_cov` return $\mu$ and $\Sigma$ when the belief is flat nowhere
  (hard directions have variance zero); along a flat direction there is no mean.
- **Where the product lives.** The algebra is here, beside the type (`_conjoin`); Mycelium's
  `combine` methods call it, for two extended Gaussians or one and a `GaussianBelief`. A
  `DiracBelief` still dominates, as it does against a Gaussian, but is first checked against the
  other side's hard constraints.
- **First consumer.** `_improper_split` in Lenticulum's `gaussian.jl`, which splits a partly flat
  input for the constraint, switching, nonlinear and Gaussian factors, is now `extended(b)`. Its
  outputs are the same up to the basis of the flat directions, which the callers use only through
  its span. The split itself is `_flat_split(η, Λ)`, which returns the parts without building the
  type: Lenticulum's split of a flagged input with a dual $\eta$
  and a precision of plain values calls it directly, and the mean carries the duals (P21). Its
  range check reads $\eta$'s values (`stop_gradient`): `norm` of duals with value exactly zero is
  `NaN`.

## Entropy (2026-11-11)

A belief with hard directions has no density with respect to Lebesgue measure on $\mathbb R^n$, but
it has one on its support, the affine subspace $\mu + \operatorname{range}\Sigma$. `support_entropy(b)`
is the entropy relative to Lebesgue measure there, in orthonormal coordinates:

$$
H(b) \;=\; \tfrac r2 (1 + \log 2\pi) + \tfrac12 \textstyle\sum_{i \le r} \log \lambda_i ,
$$

over the $r$ nonzero eigenvalues of $\Sigma$. It is the finite part of the entropy of
$\mathcal N(\mu, \Sigma + \varepsilon I)$, whose divergent part is $\tfrac{n-r}{2}\log(2\pi e\varepsilon)$.
A point gets $0$, the convention [[Bethe Free Energy]] already uses for clamped variables. With flat
directions the belief is improper and has no entropy; `support_entropy` throws.

This is a convention, and the Bethe sum is only meaningful if every term follows it: each
belief's entropy relative to its own support. A factor with a smooth density whose input has hard
directions works in the input's coordinates along the other directions. `_hard_reduction(b)`
returns $x = x_0 + U s$, with $x_0$ the point fixed along the hard directions, $U$ an orthonormal
basis of the rest, and the Gaussian part in $s$ in canonical form, $(U^\top\eta,\ U^\top\Lambda U)$,
zero along the flat directions. The hard relation's own term is in `HardConstraintFactor`'s note.
On trees the result is $-\log Z$ (Lenticulum `test/hard_free_energy.jl`), which is what makes the
convention the right one there.

## Measured

`test/extended_gaussian.jl`:

- **The product against the limit of proper Gaussians**, 135 random pairs (dimension 2–5,
  0 to $n-1$ flat and up to $n-1$ hard directions between the two, in random bases). The reference
  is the product of the Gaussians with covariance $\Sigma + \varepsilon I + \tau P_D$, $\varepsilon = 10^{-20}$,
  $\tau = 10^{20}$, in 512-bit arithmetic. Off the product's flat directions the means and covariances
  agree to $10^{-9}$; along them the reference variance exceeds $10^{15}$.
- **Laws:** commutativity, associativity, the flat belief as unit, pushforward composing
  (`pushforward(pushforward(b, M₁), M₂) = pushforward(b, M₂M₁)`), `marginalise` as a pushforward.
- **`support_entropy` against the limit**, 60 random beliefs (dimension 1–5, 0 to $n$ hard
  directions): the entropy of $\mathcal N(\mu, \Sigma + \varepsilon I)$ minus $\tfrac h2\log(2\pi e\varepsilon)$ at
  $\varepsilon = 10^{-20}$ in 512-bit arithmetic, with $\Sigma = GWG^\top$ formed there from the
  generator's factors (exactly of rank $n - h$), agrees to $10^{-9}$. **`_hard_reduction`**, 40 random
  beliefs with flat and hard directions: $U$ orthonormal and orthogonal to the hard directions,
  $x_0$ the fixed point along them, the precision $U^\top\Sigma^+U$, zero along the flat directions.
- **Conversions and refusals:** round trips with `GaussianBelief`, the clamp, contradictory
  constraints, $\eta$ outside $\operatorname{range}\Lambda$, a covariance that is not positive semidefinite.

## Derivatives (P4, 2026-11-16)

The type holds any `Real` (`ExtendedGaussian{T<:Real}`), so a ForwardDiff dual or a Zygote-tracked
float passes through every operation. Before, it held only floats, and the coupled replay of a
graph with a `HardConstraintFactor` failed under both (ForwardDiff at the type bound, Zygote at
`_independent_join`, which wrote into zero arrays and took no tuple).

**The rule.** Every subspace (a basis of the flat directions, of their complement, of the hard
directions, of the range of a covariance or precision, of the solution set of stacked constraints)
comes from a decomposition of the values, and its rank is decided there by `_eig_tol`. The
derivative is the one at constant rank. With $X_0$ the value of a matrix $X$ and
$\Delta = X - \texttt{stop\_gradient}(X)$ (zero in value, the tangent in derivative), the bases move
only into each other:

- a positive semidefinite $S$ with range $V_0$ (eigenvalues $\lambda$) and null space $N_0$:
  $V = V_0 + N_0 N_0^\top \Delta V_0 \lambda^{-1}$, $N = N_0 - V_0 \lambda^{-1} V_0^\top \Delta N_0$, and
  $S^+ = V (V_0^\top S V_0)^{-1} V^\top$, whose derivative is the constant-rank one,
  $-S^+\Delta S^+ + N_0N_0^\top\Delta S^{+2} + S^{+2}\Delta N_0N_0^\top$ (`_psd_split`, `_split_pinv`);
- the column space of $M = U_r S V_r^\top$: $B = B_0 + (I - B_0B_0^\top)\Delta V_r S^{-1}$ (`_orth`);
  the complement of the column space of $B$: $W = W_0 - (B^+)^\top\Delta^\top W_0$ (`_complement`);
- the least-squares solution and null space of stacked constraints $Cx = c$, with $P = C^+$ at the
  value: $x_0 = Pc - P\Delta Pc + PP^\top\Delta^\top(I - U_rU_r^\top)c + N_0N_0^\top\Delta^\top P^\top Pc$,
  $N = N_0 - P\Delta N_0$ (`_from_potential`);
- the support entropy uses $\log\det(V_0^\top\Sigma V_0)$, whose first-order change is that of the
  nonzero eigenvalues.

No eigenvector or singular vector is differentiated, so repeated eigenvalues are no difficulty;
these are the formulas of P25 (`LenticulumForwardDiffExt`) and P26, written once for both ADs: the
decompositions are of values (`_value_eigen`, `_value_svd`, marked non-differentiable for
ChainRules), and $\Delta$ carries the derivative through `stop_gradient`, which strips a dual and has
a zero pullback. With floats $\Delta$ is exactly zero and every result is the decomposition's.

**Refusals.** A tangent with $L^\top\Delta R \ne 0$, $L$ and $R$ the left and right null spaces at
the value, changes the rank at first order: the value jumps there and has no derivative.
`_check_constant_rank` (`LenticulumCoreForwardDiffExt`) refuses it under ForwardDiff, with the
tolerance $\sqrt\epsilon\max(\|\Delta_k\|_\infty, \text{spectrum})$ per partial. A Zygote cotangent cannot see
the tangent, so under Zygote the constant-rank derivative is returned without a check.

**Where it is used.** Mycelium's `combine`, `damp` and `belief_distance` of a `GaussianBelief` of
duals and an extended Gaussian *with* hard directions now take the dual side's extended form
(before: an `ArgumentError`, there being no canonical form). Against a soft extended Gaussian they
keep the canonical form, which needs no decomposition and takes a precision whose rank changes at
first order (P25's limit), which the extended form refuses. `HardConstraintFactor`'s free energy
takes $\tfrac12\log\det GG^\top = \sum\log\sigma_i + \operatorname{tr}(G^+\Delta)$ and the null space of $G$ by
the same rule (`hard_constraint.md`).

**Measured.** `test/extended_gaussian_ad.jl`: beliefs $b(t) = \operatorname{pushforward}(b_0, I + tK)$,
which turn their flat, hard and Gaussian directions with $t$ at constant counts (30 random draws,
dimension 2–4); pushforward, preimage, the hard directions' projector, the canonical part, the
product, the support entropy of a product and `extended` of an improper Gaussian whose null space
turns, each through basis-free outputs ($\mu$, $\Sigma$, $DD^\top$): ForwardDiff and Zygote against
central differences, $10^{-6}$ absolute and $10^{-5}$ relative (305 checks). With the bases held fixed
(the moving terms of `_psd_split` multiplied by zero) 182 of them fail. Refusals: a precision and a
covariance that gain rank at first order, and nested duals; a precision whose null space only turns
is accepted. Lenticulum `test/hard_gradient.jl`: the coupled replay on 12 random trees with a hard
relation (some fixing $u$ and $v$ entirely) against the ForwardDiff gradient of the dense $-\log Z$.
A belief with no Gaussian direction (only hard or flat ones) is tested on its own: the loop above
draws at least one, and the review (LP-043) found Zygote failing there.

## Element type (P30, 2026-11-18)

The algebra keeps the element type of its beliefs: `_conjoin` converts Int weights (the default 1) to
the beliefs' float type before `sqrt`, `preimage` its Int offset (default 0), `marginalise` builds
its selection matrix in the belief's type, and `_logsum` and `_scale` start from a zero of the
spectrum's type. Before, each put a Float64 into Float32 messages. `extended(::TrivialBelief, n, T =
Float64)` and `extended(nothing, n, T)` take the type of a belief that carries no information, which
has none of its own. Float weights (damping) are not converted: a Float64 damping weight on Float32
beliefs widens the product, by promotion. Test: Lenticulum `test/p30_element_type.jl`.

## What it does not do

- **Density and log-partition.** A belief with a flat or hard direction has no density with
  respect to Lebesgue measure on $\mathbb R^n$; `belief_logdensity` does not take it. Its entropy is
  relative to its support (above); with flat directions there is none.
- **Factors.** `HardConstraintFactor` emits it; the message store's distance and damping take it
  (Mycelium `gaussian.jl`); every relation factor takes it as input, and `GaussianFactor`,
  `GaussianPrior`, `LinearConstraintFactor`, `SwitchingConstraintFactor` and `HardConstraintFactor`
  take hard inputs in their free energies, the first four also in the frozen beliefs of the
  gradient (P2). `GaussianFactor`, `NonlinearFactor` and `LinearConstraintFactor` emit it for
  improper messages; the `improper` flag of `GaussianBelief` remains, by decision, for AD, plates and mixtures
  ([[Backlog Done 2026-11]], 2026-11-11).
- **Gradients where a rank changes.** A tangent that changes the number of flat, hard or Gaussian
  directions at first order has no derivative (the value jumps); ForwardDiff refuses it, Zygote
  returns the constant-rank derivative without a sign (below). A rank that changes only at second
  order is not detected. Nested duals (second derivatives) are refused.
- **Device arrays.** Host `Matrix` only.

## Difficulties

The constructor first projected $\mu$ and $\Sigma$ with $I - DD^\top$. When $D$ spans everything (every
input of a relation uninformative), that leaves a residue of $10^{-30}$ instead of zero, and the
relative tolerances took it for a tiny variance: a message that should be flat came out hard. It
now projects through an orthonormal basis of the complement of $D$, which is then empty.

Under Zygote, `hard_directions` of a belief with two or more hard directions and no Gaussian one
threw a `DimensionMismatch`: the split had no nonzero eigenvalue, and the pullback of the broadcasts
by the empty $\lambda_0$ could not be projected onto it. `_psd_split` now returns before them when
$\lambda_0$ is empty; at constant rank $S$ stays 0 and $N_0$ spans everything, so nothing moves
([[LP-043 Findings of the review of P4]]).

The first version of the oracle failed in half the cases with covariances of order $10^{12}$. Two
faults, both in the test: with $\tau/\varepsilon = 10^{60}$ two inversions exhaust 256-bit precision,
and the projector $P_D$ formed in `Float64` carries a rounding of $10^{-16}$ that $\tau$ turns into a
variance of $10^4$ in directions that are not flat. The reference now uses $\tau = 10^{20}$, 512 bits,
and forms $P_D$ in `BigFloat`. A test of flatness by $D^\top\Lambda D \approx 0$ failed for the same
reason in a different place (the hard directions have precision $1/\varepsilon$); flatness is tested
by the reference variance along $D$ instead.

## Element types (Q4)

The two-argument constructor takes the empty flat-basis buffer from the mean array.
An empty eigenvalue tolerance has the type of its spectrum. The representation stores dense
CPU vectors and matrices and uses LAPACK for its general spectral factorizations. The
BigFloat limits are exercised by the hard, split and scale-shift cases in Lenticulum's
`test/eltypes.jl`. The convenience conversion of an untyped trivial belief uses Float64
unless the caller supplies a type.

## Device arrays (Q7)

Adapt copies the stored normalised representative without reprojecting it. Covariance and
flat-direction algebra use the representative’s backend and materialise any constant bases
there. The host spectral helper decomposes the small value matrix and returns the selected bases
and eigenvalues or singular values. Products, projections and inverses then use those arrays on
the device.
