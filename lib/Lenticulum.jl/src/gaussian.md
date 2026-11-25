#implementation

> `GaussianFactor` and `GaussianPrior`: the linear-Gaussian conditional
> $p(y \mid x) = \mathcal{N}(y;\, Ax + b,\, Q)$ and the prior factor $\pi : 1 \multimap X$
> of AutoBayes Remark 24. The reference oracle for the entire statistical-game and Bethe
> free-energy interface.

> Sources: code: `gaussian.jl`, `beliefs.jl`, `constraint.jl`; St Clere Smithe & Perin, *AutoBayes*, arXiv:2503.18608, 2025; Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. IT 2005
>
> Bibliography: [[Bibliography#^stclere2025autobayes|St Clere Smithe & Perin 2025]] · [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]]
>
> Theory: [[Factors are Parameterized Statistical Games]] · [[Scalar and Multivariate Energy]] · [[Bethe Free Energy]] · [[Messages are Inversions]]

## Graph free-energy edge directions

`GaussianFactor` and `GaussianPrior` declare `Mycelium.uses_cavity == true`: their local
free energies form joint/posterior Gaussian beliefs from incoming messages.
Graph free-energy evaluation rejects an attached `Emitting()` edge with `ArgumentError`;
use `Bidirectional()` if that channel must also send messages. Missing messages on permitted
edges retain their existing local behaviour. See [[free_energy]] and
[[LP-014 Emitting edges drop the cavity message from the free energy]].

## 1. Role as the reference oracle

`GaussianFactor` is the first factor in the repository that implements the complete
`LenticulumCore` and `Mycelium` interfaces without approximations:

1. **Open model**: `LinearGaussianModel(factor, polarity)` with $\llbracket c \rrbracket \cong 1$.
2. **Exact Bayesian inversion**: `ExactInversion()`, where `likelihood_message` computes the
   cavity-free likelihood message $\mu_{c \to t}$ and `invert` returns the posterior
   $c^\dagger_\pi(y) = \pi \odot \mu_{c \to t}$.
3. **Bidirectional polarity**: both `(x = Observed(), y = Unobserved())` (forward pushforward)
   and `(x = Unobserved(), y = Observed())` (backward likelihood) are supported.
4. **Dual energy spaces**:
   - `pointwise_energyspace(f)` is `EuclideanEnergySpace(nout)` with
     `pointwise_scalarisation(f) = SquaredNorm(Q⁻¹)`, acting on the residual vector
     $r = y - Ax - b$.
   - `energyspace(f)` is `GradedEnergySpace((fit, complexity, negentropy))` with linear
     `IdentityScalarisation()` on each summand.

## 2. Strict vs. lax composition and the Jensen gap

Under the factor belief $b_c(x, y) \propto \mathcal{N}(y;\, Ax+b,\, Q)\,\mu_{x\to c}(x)\,\mu_{y\to c}(y)$,
`residual_statistics(f, msgs, ps)` returns the mean residual $\bar{r}$, the residual covariance
$\operatorname{Cov}(r)$, and the factor belief entropy $H(b_c)$.

The expected quadratic fit term splits into two pieces:

$$
\underbrace{\mathbb{E}_{b_c}\bigl[\tfrac{1}{2} r^\top Q^{-1} r\bigr]}_{\text{exact (`fit` summand)}}
\;=\;
\underbrace{\tfrac{1}{2} \bar{r}^\top Q^{-1} \bar{r}}_{\text{pointwise at mean}}
\;+\;
\underbrace{\tfrac{1}{2} \operatorname{tr}\bigl(Q^{-1} \operatorname{Cov}(r)\bigr)}_{\text{Jensen gap}}
$$

By placing the trace term inside the `fit` summand of `GradedEnergy` and using a linear
`IdentityScalarisation`, `GaussianFactor` makes composition strict while still exposing the
pointwise residual geometry via `pointwise_energyspace` and `pointwise_scalarisation`.

## 3. Opt-in covariance learning (LP-005, 2026-10-26)

`GaussianFactor(...; learn_noise=true)` (also `:full`) adds `ps.noise` with
`log_diagonal` and packed strict `lower` entries of the precision Cholesky:
$Q^{-1}=LL^\top$, $\log\det Q=-2\sum_i\log L_{ii}$ before flooring.
`learn_noise=:diagonal` stores only `noise.log_precision`, the logs of the diagonal
precisions. `noise` supplies the initial covariance, exactly reconstructed at setup;
a full covariance with off-diagonal entries is rejected in diagonal mode.
`GaussianPrior(:x, μ, Σ; learn_noise=...)` uses the same options for `Σ`, keeping `μ`
fixed. `noise_covariance(f, ps)` reads the current covariance; struct fields remain the
initial values. Default parameters and fixed-noise calculations retain their contracts.

`GaussianFactor.energy` stays the residual. Its default pointwise scalarisation is
parameter-dependent and localises to `SquaredNorm` of the current precision.
`scalar_energy` still omits the normaliser. **Train noise with free energy**: fit alone
falls as covariance grows, so it is not a valid noise likelihood objective.
`local_free_energy` and `free_energy` include $\tfrac12\log\det(2\pi Q)$ in
`complexity`. `GaussianPrior.energy` already included its normaliser and continues to
do so for both fixed and learned covariance. Noise hyperpriors enter free energy only.
This preserves [[L-ADR-0003 Pointwise energy and free energy are separate levels]].

Without a noise prior, `eigenvalue_floor` (default `1e-6`) clips precision eigenvalues
at its reciprocal, equivalently flooring covariance eigenvalues. Diagonal mode clips
log-precisions. The initial covariance must satisfy the floor. Flooring is hard and has
a kink: saturated eigenvalues have derivative zero, so optimisation can stick there.
A Wishart prior on full precision (`WishartPrecisionPrior(df, scale)`) or independent
log-normal priors on diagonal variances (`LogNormalVariancePrior(logmean, logstd)`)
replace the floor. Their normalised negative log densities enter `complexity`.
See [[noise]] for density coordinates, dataset multiplicity, formulas and AD rules.

## 4. `likelihood_message` vs. `invert`

In belief propagation, a factor-to-variable message must be a **likelihood** (excluding the
target variable's incoming prior $\pi$), whereas AutoBayes Definition 10 defines the lens
inversion $c^\dagger_\pi(y)$ as the **posterior** (including $\pi$). `GaussianFactor` implements
`LenticulumCore.likelihood_message` for the likelihood $\mu_{c\to t}$ and defines
`LenticulumCore.invert` as `Mycelium.combine(π, μ_{c→t})`, satisfying both contracts cleanly.

## 5. What it does not do

- **No precision-form (`Q = 0`) deterministic limit.** `Q` must be strictly positive definite;
  noiseless linear equality constraints require a non-zero regulariser `Q = εI` (see
  `LinearConstraintFactor` in [[constraint]]).
- **The prior mean is fixed.** Covariance learning does not make `GaussianPrior.μ` a parameter.
- **CPU AD coverage.** ForwardDiff and Zygote are tested; device and higher-order AD are not established.

## 6. Implementation difficulties

- **Improper backward messages.** When $A$ has fewer rows than columns ($n_{\text{out}} < n_{\text{in}}$)
  or is rank-deficient, the backward precision $A^\top (Q + S_y)^{-1} A$ is positive semidefinite
  but singular. The message is declared improper (`GaussianBelief(η, Λ, true)`, `_rank_deficient`):
  left to rounding, a Cholesky factorisation of such a precision sometimes succeeds, and the
  receiver then computes with a covariance of order $1/\varepsilon$. A factor that receives a
  partly flat message does not convert it to moments; see "Improper inputs" below.
- **Clamped channels (`DiracBelief`) in `residual_statistics`.** A `DiracBelief` has infinite
  precision and cannot be converted to canonical parameters `(η, Λ)`. `residual_statistics`
  Therefore branches explicitly on the four combinations of `(xd, yd)` so clamped channels are
  substituted into the affine offset rather than added to the joint precision matrix.

## 7. Game-level entropy and the source lens

`entropy` returns an element of the energy space, `GradedEnergy((fit = 0, complexity = 0, negentropy = H))`;
`free_energy` subtracts it, hence its `negentropy = -H`. The constraint, nonlinear and likelihood
factors follow the same convention. `GaussianPrior` is also a source lens (`assemble`): its
`likelihood_message` is the source Gaussian, `invert` combines it with the cavity, and its pointwise
energy is the graded negative log-density (quadratic fit; the Gaussian normaliser as `complexity`).
Both are checked by `check_factor_conformance` (`lib/LenticulumCore.jl/test/support/factor_conformance.jl`).

Related: [[constraint]], [[nonlinear]], [[likelihood]], [[beliefs]]

## Optional parameter AD (LP-002, 2026-10-26)

For fixed noise, `energy_jacobian(...; wrt=:ps, ad)` and `energy_gradient` differentiate
the residual and its fixed precision norm. For Dirac endpoint beliefs with fixed noise,
`free_energy_gradient` has the same parameter derivative: the normalizer is constant
and entropy is zero. With learned noise the two gradients differ by the normaliser
and optional hyperprior, checked against finite differences in `test/learnable_noise.jl`.
The ImplicitLayers AD tests check these facts with ForwardDiff and Zygote against
`-(x' ⊗ I)`, `-I` and the precision-weighted residual gradients. The top-level suite
also checks their finite-difference reference. Fixed `Q` precision is formed by
`inv(Symmetric(Q))`, avoiding the mutating inverse-of-Cholesky path under Zygote.
`test/learnable_noise.jl` additionally covers learned-covariance local free energies
for all four Dirac/Gaussian endpoint combinations and a Gaussian incoming belief
on `GaussianPrior`. The learned path uses matrix inverses and matrix log determinants
for posterior statistics to avoid mutating inverse-of-Cholesky and unsupported
symmetric factorisation paths in Zygote. Inversion derivatives are not established.

## LP-011 implementation (2026-10-27)

Functional schedule messages use matrix inverses instead of a
mutating inverse of Cholesky; ordinary message passing retains its arithmetic.
Joint block precision assembly now concatenates promoted numeric blocks, permitting
active Gaussian cavity precisions under ForwardDiff/Zygote. The gradient tests compare
full finite-tree replay to closed marginal likelihood derivatives and separately derive
the legacy fixed-cavity local-posterior derivative; the latter is not a true EM M-step. (Since
LP-020 the diagonal default holds the factor beliefs fixed instead, and is the true one.)

Evidence: `lib/MyceliumLearning.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].


## LP-002 batch B replay context (2026-10-27)

The private gradient-message hook now receives the selected AD backend as its
first argument. GaussianFactor, GaussianPrior and GaussianComposite preserve
their existing functional message formulas and forward that constant context.
The first full top-level run found that leaving the old signatures in place
bypassed these methods and traced ordinary inverse-of-Cholesky messages into
LAPACK under Zygote. Updating all three specialisations restores the established
matrix-inverse replay path without changing ordinary inference arithmetic.
`test/gradient_couplings.jl` checks the Gaussian marginal-likelihood gradients
against closed forms and exercises collapsed composites; no dense Gaussian
device factorisation is added.

## Mixture inputs (2026-10-31)

A message whose input is a `MixtureBelief` of Gaussians used to fail with a `MethodError`:
`_moments` had no method for it. Now the mixture is **collapsed to its total mean and covariance**
(`moment_match`) before the Gaussian formula runs. That covers `GaussianFactor`,
`LinearConstraintFactor` and the linearised factors, since all of them read their inputs through
`_moments`. This is assumed-density filtering, not exact message passing. The exact message from a
linear factor with a $K$-component input is itself a $K$-component mixture, and with several mixture
inputs the count multiplies. The collapsed message is exact only for the collapsed input. How much
the approximation costs is measured in experiment E6 ([[Composing Stochastic Factors]]; [`examples/factorial_inference.md`](https://github.com/MathStruct/Lenticulum.jl/blob/experiment-qm/examples/factorial_inference.md) on the branch `experiment-qm`). The first use was
switching factors (binary outcomes) feeding a sum node, where the exact message would have
$2^{L-1}$ components.

## The factor belief, held fixed (LP-020, 2026-11-01)

The graph parameter gradient differentiates each factor's free energy with the factor's
**belief** held fixed, not its incoming messages. That is the envelope gradient of the Bethe free
energy at a belief-propagation fixed point ([[free_energy]], "Fixed beliefs, not fixed messages";
[[LP-020 The fixed-message parameter gradient is not the free-energy gradient]]). Mycelium asks
for two methods, `factor_belief` and `belief_free_energy`. This file implements them for
`GaussianFactor` and `GaussianPrior`. `constraint.jl`, `switching.jl` and `composition.jl` add them
for their factors.

**One construction for all of them.** Each of these factors has an affine residual
$r = \sum_i R_i x_i + r_0$ with weight $W$ ($Q^{-1}$, or the prior's precision):

| factor | blocks $R_i$ | $r_0$ |
|---|---|---|
| `GaussianFactor` | $R_x = -A$, $R_y = I$ | $-b$ |
| `GaussianPrior` | $I$ | $-\mu$ |
| `LinearConstraintFactor` | $A_i$ | $-c$ |
| `SwitchingConstraintFactor` | $A_i$ | $-b_k$, one column per outcome |

`_affine_gaussian_belief` moves the point channels into $r_0$ and builds the joint of the others:
$\Lambda = G^\top W G + \operatorname{blockdiag}(\Lambda_i)$, $\eta = \operatorname{vcat}(\eta_i) - G^\top W r_0$, with $G$ the blocks of the
non-point channels. It returns a `FrozenGaussianBelief` (points, mean, covariance, entropy, and
for the switching factor the outcome posterior). `_frozen_residual` re-evaluates the residual's
mean $G m + r_0$ and covariance $G \Sigma G^\top$ under it with the current parameters. Each
`belief_free_energy` then applies its factor's own `(fit, complexity, negentropy)` formula, and
only `negentropy` stays constant.

**What it does not do.** A mixture message is collapsed to its moments, as in the factors' own
messages, so with mixture inputs the frozen belief is the collapsed one. The prior's mean is not a
parameter, so only a learnable noise moves.

## Improper inputs, split (2026-11-02)

`_split_moments(b, n)` returns an input as `(mean, covariance, N)`: a Gaussian part and the
directions, the columns of `N`, along which it is flat (`nothing` if none). `_moments` is
unchanged and still serves `GaussianFactor`, the composites and the nonlinear factors; only the
relation factors (`LinearConstraintFactor`, `SwitchingConstraintFactor`) read the split. A
proper Gaussian takes the same path as `_moments`, and `isproper` is non-differentiable, so the
differentiated code is unchanged for proper inputs. The theory and the tolerances are in
`constraint.md` §4.1.

Since 2026-11-10 the split of an improper input is the input as a `LenticulumCore.ExtendedGaussian`
(`_improper_split` calls `extended`, which has the tolerances and refusals described there): the
mean and covariance are its Gaussian part, `N` an orthonormal basis of its flat directions. The
relation factors still compute with the tuple; since 2026-11-11 they also take an extended
Gaussian as input (below).

**`GaussianFactor` takes partly flat inputs (2026-11-10).** A `GaussianFactor` is the relation
$-Ax + y = b + \varepsilon$, the two-channel case of `LinearConstraintFactor`. When its input is
informative in some directions and flat in others (`_partly_flat`: a declared-improper or
singular Gaussian with nonzero precision), `_message` calls `_flat_message`, which hands the
relation and the input's split to `_linear_message`. Proper inputs take the old path, with the
same arithmetic. `NonlinearFactor` does the same for a partly flat input on `y`, through its
local linear model. A fully flat input (no message) still gives `TrivialBelief()`.

Before this, a chain with a partially observed vector state (a Kalman smoother with position-only
measurements) threw `belief_mean needs a proper belief`, or, when rounding let the Cholesky
factorisation of the rank-deficient precision through, returned wrong marginals without an error
(for $H = [0.1\ \ 0.3]$ the mean of $x_0$ was off by 0.049). `test/gaussian_oracle.jl` compares
random trees with rectangular coefficients, and that smoother, with the dense joint solve
([[LP-024 Findings of the review of 2026-11-09]]).

**What the checks cost.** Both run on every message. `_partly_flat` is one `isproper` (a symmetry
test and a Cholesky factorisation of the input's precision): 0.19 μs at dimension 2 and 0.41 μs at 6,
against 1.6 and 3.8 μs for a forward message. `_rank_deficient` is one SVD of `A` on every
backward message: 0.71 and 2.6 μs, against 2.4 and 6.6 μs for the whole message. It first called
`_projected_rank(A, A)`, which computes the singular values twice, and was half the cost of a
backward message (`bench/gaussian_message_cost.jl`, 2026-11-10, best of five runs of 20 000
calls, one machine). `A` is a parameter, so the rank could be cached per parameter
tree.

Since 2026-11-12 (I18) it is cached per array. `_rank_deficient(A::Matrix{Float64})` keeps the
answer in a memo keyed by `objectid(A)`. An entry holds a weak reference to the array, a copy of
it and the answer, and it is used only if the reference still points to `A` and the copy still
equals it, so a reused id or a change in place recomputes. The memo is guarded by a lock and is
emptied at 4 096 entries. A hit takes 21 ns instead of the 0.63 μs of the SVD at dimension 2
(Chairmarks, one machine). `bench/gaussian_message_cost.jl` (which reuses one `A`, as an inference
does) now gives 0.95 μs for a forward and 1.06 μs for a backward message at dimension 2, and 2.82
and 2.94 μs at 6, with `_rank_deficient` at 0.02 and 0.03 μs. Other element types (dual numbers, `Float32`) and device arrays take
the SVD every time (`_rank_deficient_svd`). It is a global memo, not one in the graph, because
`factor_message` sees neither the graph nor the factor's position in it. A parameter step that
makes new arrays starts with misses and fills the memo again. The same change made `_moments`
factorise the input's precision once instead of twice (`LenticulumCore._mean_cov`;
LenticulumCore's `gaussian_belief.md`, "Per-message costs"). Tested in `test/i18.jl`: 2 000 calls
against the SVD with changes in place, a full memo, dropped arrays, planted stale entries and
4 000 calls from several threads. Reviewed by Mistral ([[LP-026 Findings of the review of I18]]).

**The constructor accepts a covariance that is symmetric up to rounding** (the tolerance of
`LenticulumCore._nearly_symmetric`) and uses its symmetric part. The collapsed factor of a
composition has the noise $B Q_a B^\top + Q_b$, which is not exactly symmetric; the exact test
refused it for most parameters when $Q_a$ was not diagonal.

## Extended Gaussian inputs (2026-11-11)

Every relation factor takes a `LenticulumCore.ExtendedGaussian` as input, which is what
`HardConstraintFactor` sends and what Mycelium's `combine` makes of it. Before, an extended
Gaussian reaching `GaussianFactor`, `LinearConstraintFactor`, `SwitchingConstraintFactor` or
`NonlinearFactor` threw `MethodError: no method matching _moments(::ExtendedGaussian, ::Int64)`,
so a hard relation could not share a variable with them. Four methods do it:

- `_split_moments(b::ExtendedGaussian, n)` is `(μ, Σ, D)`, its own representation: the tuple was
  this object already. `Σ` may be singular along hard directions; the noise of every relation
  factor is positive definite, so the residual metric $Q + \sum_i A_i S_i A_i^\top$ stays positive
  definite.
- `_moments` takes an extended Gaussian without flat directions (its `μ`, `Σ`), or one flat
  everywhere (no information); one partly flat throws, as no caller reaches it after
  `_partly_flat`.
- `_partly_flat(b::ExtendedGaussian)` is true for **any** flat direction, also when it is flat
  everywhere, so `GaussianFactor` computes that message exactly. With $n_{\text{in}} < n_{\text{out}}$
  an everywhere-flat input on $x$ still fixes $y$ off $\operatorname{range}(A)$, up to the noise.
  The `TrivialBelief()` input (no message) kept the old shortcut, which returned `TrivialBelief()`
  and dropped that information, until P3 ("No input on x" below).
- `_canon(b::ExtendedGaussian, n)`, for the free energy, goes through `GaussianBelief(b)`: zero
  precision along flat directions, and an `ArgumentError` for hard ones, which have no canonical
  form (the entropy row of the backlog). `_mf_split` takes it for mean-field nodes.

Messages were unchanged by this step; the next section changes them.

**Measured** (`test/extended_inputs.jl`): `LinearConstraintFactor` with random extended-Gaussian
inputs that mix flat, hard and Gaussian directions (80 factors, 2–3 channels of dimension 1–3,
1–3 equations; 208 messages) and `GaussianFactor` in both directions (60 factors, 120 messages)
agree with `HardConstraintFactor` with the same positive definite noise, an independent path
(pushforward and preimage), to $10^{-7}$ absolute and $10^{-9}$ relative. For all four relation
factors an input given as a point, proper or improper Gaussian and the same input as `extended(b)`
give the same message (40 trials, 279 checks). A tree of a prior, a `GaussianFactor` and a
`HardConstraintFactor` (20 random instances) matches the dense Gaussian conditioned on the
relation under `infer!` with a tree schedule. Run against the code before this change, the first
testset fails with the `MethodError` above.

**In plates (P4, 2026-11-16).** Extended messages in a plate are stacked as `VectorBlock`s
(`belief_block`), for which `_block_parts` is `nothing`, so the block methods of
`LinearConstraintFactor` and `SwitchingConstraintFactor` take the per-replica path, and
`GaussianFactor`, which has no block method, always does. `test/extended_plates.jl`: a plate of four
replicas, each with an observation, a `GaussianFactor` and a hard relation (10 random instances,
hard messages reaching the `GaussianFactor`s), matches the dense Gaussian of each replica
conditioned on its observation and the relation, to $10^{-7}$ absolute and $10^{-9}$ relative; a
plate of five replicas in which every `LinearConstraintFactor` receives an extended input (8
instances) gives the messages of the same tasks sent one by one to $10^{-8}$, and each replica's
marginals as its own graph to $10^{-10}$. No code changed: the per-replica fallback was already
right; only the test was missing.

Two limits found on the way, `@test_broken` in that file:

- a `SwitchingConstraintFactor` on a variable that a hard relation also reaches: its mixture
  message (or, with `ExpectationPropagationRule`, the EP projection of the cavity) meets the
  extended message in Mycelium's mixture product, which had no rule for an extended component
  (`ArgumentError: no product rule for mixture components`), in a plate and in a single graph
  alike. Done in P27 (2026-11-17): Mycelium `gaussian.md`, "Mixtures meeting extended Gaussians";
  `test/mixture_extended.jl` checks it against enumeration of the outcome;
- the coupled replay (`free_energy_gradient` with `schedule`, `ExactCoupling` edges) through a
  `HardConstraintFactor` under ForwardDiff and Zygote: the hard factor's inputs carry dual numbers
  (a derivative by an upstream $A$), and `ExtendedGaussian` holds only `AbstractFloat`s; under
  Zygote `_independent_join` receives a tuple. Reference differences agree with
  $-\log Z$ of the dense model to $10^{-9}$ (5 random trees, a scratch probe, not a test). This is
  the AD half of P4, still open.

Not checked: extended inputs to `NonlinearFactor` with a partly flat cavity, which `_expansion`
refuses through `_moments`.

## No input on x (P3, 2026-11-17)

**Theory.** With nothing known about $x$, the relation $y = Ax + b + \varepsilon$ still says
something about $y$: $x$ integrated out under a flat prior (the limit $\tau \to \infty$ of
$\mathcal N(0, \tau I)$) leaves $y$ flat along $\operatorname{range}(A)$ and Gaussian off it. With
$U$ an orthonormal basis of $\operatorname{null}(A^\top)$ the message is

$$
\Lambda = U (U^\top Q U)^{-1} U^\top, \qquad \eta = \Lambda b,
$$

of rank $n_{\text{out}} - \operatorname{rank}(A)$: informative whenever $A$ is short of full row
rank, as for every $A$ with $n_{\text{in}} < n_{\text{out}}$. The message backwards with no input
on $y$ is `TrivialBelief()` exactly, since every column of $A$ is reachable.

**Implementation.** Two changes.

- `_message` and `_gradient_message` send a forward message whose input carries no information (no
  message, `TrivialBelief()`, a zero precision, a mixture whose every component carries no
  information, which `_moments` now reports as no input instead of passing it to `moment_match`) through
  `_no_input_forward`, which takes the flat path of the relation factors (`_flat_message`,
  `constraint.md` §4.1; `_split_moments` already made such an input `_flat_part(n)`). A square $A$ of
  full rank is answered `TrivialBelief()` by the memoised rank test, without the flat path. Before,
  these inputs returned `TrivialBelief()`; an everywhere-flat `ExtendedGaussian` already took the flat
  path, so the two forms of "no information" now agree.
- In a graph, Mycelium does not call the factor when no input is informative unless the factor
  supports the resulting polarity (`_make_plan`, `passing.jl`): $x$ `Latent()`, $y$ the target. So
  `GaussianFactor` now supports that polarity (and lists it in `supported_polarities`), as
  `LinearConstraintFactor` supports every polarity with one target. The mirror, $y$ latent and $x$
  the target, stays unsupported: its message is `TrivialBelief()`, which Mycelium sends for it anyway.

**Measured** (`test/p3_no_input.jl`): 80 random factors with $n_{\text{in}}, n_{\text{out}} \in 1{:}4$
and random rank (deficient ones included), seven forms of the empty input (two of them mixtures,
which threw in `moment_match` until the review, [[LP-048 Findings of the review of P3]]),
`factor_message` and `_gradient_message`: 1120 messages, all equal to the closed form above (through `GaussianBelief` of
the extended message) to $10^{-8}$ relative, `TrivialBelief()` exactly when $U$ is empty. A graph
of $x$ with no other factor, a `GaussianFactor` and a prior on $y$ (20 instances,
$n_{\text{in}} < n_{\text{out}}$): the marginal of $y$ under `infer!` with a tree schedule matches the
dense posterior with $x \sim \mathcal N(0, \tau I)$, $\tau = 10^{30}$ in `BigFloat`, to $10^{-8}$
relative. Before the change every one of those 20 marginals was the prior alone.

**What changes for existing schedules.** In the first sweep of a schedule a `GaussianFactor` whose
input has no message yet now sends an informative message to $y$ when $A$ is not of full row rank,
where it sent `TrivialBelief()`. The fixed point of a tree is unchanged (the later message replaces
it); a loopy schedule may take a different path to its fixed point (an expectation, not measured).
Measured on the existing tests: the whole Lenticulum suite passes with the change, and the only
assertion that had to change was the count of `GaussianFactor`'s polarities (2 → 3, `runtests.jl`).

**Does not do.** A `NonlinearFactor` with no input sends this message for its linearisation at
`x0`, and `TrivialBelief()` without `x0` (P28, [[lib/Lenticulum.jl/src/nonlinear|nonlinear]] "No input on x"). A plate of
`GaussianFactor`s with an empty input is not tested (it has no block method, so the replicas go one
by one). The free energy of a `GaussianFactor` whose $x$ has no
information is not covered (no entropy for a flat marginal; the entropy row of the backlog).

## Improper messages as extended Gaussians (2026-11-11)

A likelihood message that the rank test finds flat in some directions is returned as an
`ExtendedGaussian`, flat along $\operatorname{null}(\Lambda)$, instead of a `GaussianBelief` with
`improper = true` (`_likelihood_message`). This covers the backward message of `GaussianFactor` and
of `NonlinearFactor`'s local linear model when $A$ is rank-deficient, and `LinearConstraintFactor`'s
message (also from `MeanField()` nodes, which share `_linear_message`) when the projected $A_t$ is.
Proper messages stay `GaussianBelief`s. The rank test decides, as before, and the conversion is
`LenticulumCore.extended`, so the flat directions are those of $\Lambda$ at its tolerance.

**Where the flag remains.** Four paths still return `GaussianBelief(η, Λ, true)`:

- the ForwardDiff paths, where the values are dual numbers. `_likelihood_message` checks the
  element types of η and Λ (both since P1), since the extended form needs an eigendecomposition, which has no method for
  them;
- `_gradient_message`, the functional message for schedule AD;
- the block messages of plates (`factor_messages`, `GaussianBlock`'s `improper` vector);
- the components of `SwitchingConstraintFactor`'s mixture messages, which Mycelium's mixture
  product weights as likelihoods.

The flag stays (2026-11-11): `ExtendedGaussian` holds only `AbstractFloat` values and finds its flat
directions by eigendecompositions that dual numbers have no method for, plate blocks share one
$\Lambda$ across replicas, and mixture weights rest on the canonical form's constants. The canonical
form with the flag serves AD, batches and mixtures; the extended form is the exact host form, the
only one with hard directions. The crossings between the two are tested since 2026-11-12 (P1,
next section).

**What changed downstream: nothing but three tests.** Run with the change, VariationalDiffusion
passed and the Lenticulum suite failed in exactly two places, both tests that read the old
representation: the smoother test asserted `msg.improper`, and a comparison helper in
`test/extended_inputs.jl` read `η`. AlgebraicFactors, not run then, had a third: its polynomial test
compares with `LinearConstraintFactor`'s message through `η` and `Λ`, and now converts an extended
oracle message with `GaussianBelief` first (found 2026-11-11). Every consumer of these messages already takes extended
Gaussians: `combine`, the store's distance and damping, the relation factors' inputs, and the free
energy. The random trees with rectangular coefficients of `test/gaussian_oracle.jl` still match the
dense joint solve, marginals and free energy, with these messages on their backward edges: in its
first testset 152 of the 336 likelihood messages are extended Gaussians (counted once, 2026-11-11).
`test/extended_inputs.jl`, "improper messages are extended Gaussians": on 60 random factors with
rank-deficient $A$ the backward message is flat exactly along $\operatorname{null}(A)$, and its
precision and $\eta$ off it are the canonical formula's, to $10^{-7}$; a full-rank square $A$ still
gives a proper `GaussianBelief`, and dual numbers keep the flag.

## Where the two forms meet (P1, 2026-11-12)

One edge carries a flat likelihood as an `ExtendedGaussian` on values and as a flagged
`GaussianBelief` under ForwardDiff. `test/forms_crossing.jl` checks the crossings, and Mycelium's
`test/forms_crossing.jl` those of its belief interface:

- **Dual path against value path.** For `GaussianFactor` and `LinearConstraintFactor` with every
  parameter moving along one direction (the coefficients along one that keeps their rank), and for
  all three relation factors with only $\eta$ moving (`b`, `c` or an input's mean; `A` plain
  `Float64`), the dual message is flagged, its values read back by `extended` are the value path's
  message, and its derivatives are central differences of the value path in canonical form
  (100 cases, 761 checks, to $10^{-5}$ with $h = 10^{-4}$; at $h = 10^{-6}$ one case with a
  saturated `tanh`, where the message has a variance of order $10^9$, was off by $7.6 \cdot 10^{-5}$
  through the rounding of the extended-to-canonical round trip, not of the dual path, which agrees
  with central differences of its own values to $3 \cdot 10^{-11}$).
- **One split rule.** `_split_moments` of a flagged belief and of `extended(b)` agree in mean,
  covariance and flat subspace, for proper, flagged, flagged but full-rank and zero precisions
  (100 cases).
- **Through an inference.** `ForwardDiff.derivative` of $x$'s marginal mean by `b`, on a tree
  whose `GaussianFactor` has rank-deficient $A$, equals the closed form $-K b_1$ of the dense joint
  (10 random instances).

**What it found.** `_likelihood_message` decided between the forms by `eltype(Λ)` alone. When only
$\eta$ is dual, which is the case for a derivative by `b`, `c` or an observation, $\Lambda$ is
`Float64` and the extended form of a dual $\eta$ was built, which threw
`MethodError: no method matching svd!(::Matrix{ForwardDiff.Dual...})`. So a ForwardDiff derivative
by `b` through a rank-deficient backward message threw since the extended messages came in
(2026-11-11): against the code before the fix the inference testset errs in all 10 instances
(`GaussianFactor`; the other factors share `_likelihood_message`, not run separately). It
now tests both element types. Mycelium's `damp` was a silent no-op on a mixed pair, and Mistral's
review found that `combine` threw where a variable hears a dual flagged message and a plain
extended one, as on a variable with two rank-deficient factors of which one is differentiated
(Mycelium's `gaussian.md`, "Extended Gaussians"; [[LP-028 Findings of the review of P1]]). The
inference testset covers that case with two factors on $x$ (40 checks).

**Known limits found here**, both closed by P21 (next section): a dual flagged message entering a
relation factor's flat-input path threw a `MethodError`, and so did `NonlinearFactor`'s backward
message at a dual expansion point.

## Dual inputs on the flat path (P21, 2026-11-16)

A flagged input $(\eta, \Lambda)$ enters a relation factor's flat path (`_flat_message`,
`_linear_message`) through `_split_moments`: the mean $\Lambda^+\eta$ and covariance $\Lambda^+$ on
$\operatorname{range}\Lambda$, and an orthonormal basis $N$ of $\operatorname{null}\Lambda$. The
value path builds it as an `ExtendedGaussian`, whose type holds floats only, so a dual $\eta$ threw.

**Theory.** When $\Lambda$ does not depend on the parameter, neither do $N$ and $\Lambda^+$; the
derivative of the split is $(\Lambda^+\dot\eta, 0, 0)$, and every later step of the flat path (the
residual metric $T$, the projected rank) depends on $\Lambda$, the factor's coefficients and the
inputs' covariances only. So with only $\eta$ dual the flat path is linear in the duals, and
carrying them through $d = c - \sum_i A_i m_i$ gives the exact derivative. This is the case of a
derivative by `b`, `c` or an observation (the precision $A^\top R^{-1} A$ of a backward message
does not see them).

**Implementation.** `_improper_split` sends an input with a dual element type to
`_improper_split_dual`, which takes the split on $\Lambda$'s values (`stop_gradient`) through
`LenticulumCore._flat_split`, the rule `extended` now also uses (one rule, one tolerance). It
accepts $\Lambda$ of floats, or of duals whose partials are all zero (as when a whole parameter
vector is made dual), and throws an `ArgumentError` otherwise. `_jacobian` (`nonlinear.jl`)
allocates in the promoted element type and takes its step $h$ from the values, so at a dual
expansion point the Jacobian's partials are central differences of the derivative of $f$; a given
`jacobian` function is converted to the promoted type, not to `Float64`.

**What it does not do.** A flat input whose $\Lambda$ carries a derivative (a dual coefficient
upstream, as in a derivative by $A$) moves its flat directions; P25 handles it (next section). A partial of $\eta$ along $\operatorname{null}\Lambda$ is dropped without an error
(the mean is $\Lambda^+\eta$): such an input leaves $\operatorname{range}\Lambda$ at first order, and the
value path would refuse it at every $t \ne 0$. The factors' own messages do not do this
($\eta = A^\top R^{-1}(m - b)$ stays in $\operatorname{range} A^\top R^{-1} A$ for every $t$); a
hand-built one can (seen in a probe of the review, not tested). Not tested: the `Unscented`
method's own flat-input case.

**Difficulty met.** The range check $\lVert N^\top\eta\rVert \le \sqrt{\epsilon}\,\lVert\eta\rVert$
returned false on duals: a component of $N^\top\eta$ whose value is exactly $0$ but whose partial is
not makes `LinearAlgebra.norm` return `NaN` (its scaling divides $0$ by $0$). The check is now on
$\eta$'s values.

**Tests** (`test/forms_crossing.jl`, the oracle is the value path at $t$ and $t \pm h$, read in
canonical form): a flat source (a rank-deficient backward message, its `b` moving) into
`GaussianFactor` forward and backward and into a three-channel `LinearConstraintFactor`, 90 random
instances, half with $\Lambda$ of zero-partial duals, values to $10^{-7}$ and derivatives to
$10^{-5}$, flat, proper and trivial outcomes all reached (614 checks); a flat input into `SwitchingConstraintFactor`, to its continuous channel (a mixture) and to its
outcome, and two flat inputs into one `LinearConstraintFactor` (271 checks, added after the
review, [[LP-038 Findings of the review of P21]]); `ForwardDiff.derivative` of
a marginal mean through an inference whose flat message passes a relation factor, 15 instances
(30); `NonlinearFactor` at a dual expansion point, `Linearization` and `Unscented`, finite-difference
and given Jacobians, 40 instances (160). Run before the fix, the first two threw
`MethodError` (`svd!` of duals) and the third `Float64(::Dual)` (the P1 `@test_broken` cases).

## Flat directions that move (P25, 2026-11-16)

Under ForwardDiff the flat directions of the flat path move when a flagged input's precision
$\Lambda$ is dual (a derivative by $A$ upstream: the backward message's $A^\top R^{-1}A$), or when the
coefficients or covariances that enter the residual metric are dual. Both need a decomposition of
dual numbers: the eigendecomposition of $\Lambda$ in the split, and the SVD of
$B = L^{-1}G$ in the residual metric (`_residual_metric`, and `_residual_metric_logdet` of the
switching factor, which also needs $\log\operatorname{pdet}(B^\top B)$).

**Theory.** At constant rank these are smooth, and their derivative is found from the
decomposition at the value without differentiating an eigenvector or a singular vector, as P26 does
for a cluster's joint precision ([[lib/Mycelium.jl/src/cluster|cluster]], "Flat directions under a
derivative"). For $\Lambda = U\lambda U^\top$ with null space $N$ and tangent $D$:

$$
U(\varepsilon) = U + \varepsilon NN^\top DU\lambda^{-1}, \qquad
N(\varepsilon) = N - \varepsilon U\lambda^{-1}U^\top DN, \qquad
\Lambda^+(\varepsilon) = U(\varepsilon)\,(U^\top\Lambda U)^{-1}\,U(\varepsilon)^\top ,
$$

whose first-order term is $-\Lambda^+D\Lambda^+ + NN^\top D\Lambda^{+2} + \Lambda^{+2}DNN^\top$, the
derivative of the pseudo-inverse at constant rank. The mean is $\Lambda^+(\varepsilon)\eta$. For
$B = U_rSV_r^\top$ with left complement $W$: $W(\varepsilon) = W - \varepsilon (B^+)^\top D^\top W$ spans
the complement of $\operatorname{range}B(\varepsilon)$ to first order and stays orthonormal
($B^+W = 0$), and $d\log\operatorname{pdet}(B^\top B) = 2\operatorname{tr}(B^+D)$. Only $WW^\top$
enters the metric $T^\top T$, so the choice of basis does not matter. The rank is constant to first
order exactly when $L^\top DR = 0$, with $L$, $R$ the left and right null spaces at the value
($N^\top DN = 0$ for $\Lambda$); otherwise the flat-direction limit drops a different constant on
each side and the message has no derivative.

**Implementation.** `_improper_split_dual` sends a $\Lambda$ with nonzero partials to
`_moving_flat_split`, and `_residual_metric` takes its complement from `_range_complement`; both
have a method for first-order duals of floats in `LenticulumForwardDiffExt`, which decides the rank
on the values by the value path's rule (`_flat_split`'s tolerance, `_range_svd`'s), refuses a
tangent with $\lVert L^\top D_kR\rVert > \sqrt{\epsilon}\max(\lVert D_k\rVert_\infty, s_{\max})$ for any
partial $k$, and builds the formulas above with $D = M - M_0$ (duals of value zero). The fallbacks
(nested duals, other number types) throw an `ArgumentError`; `_range_complement`'s fallback was
missing at first, and nested duals hit a `MethodError` of `svd` ([[LP-042 Findings of the review of P25]]).

**What it does not do.** A tangent that changes the rank only at second order passes the check
and gets the constant-rank derivative while the value is discontinuous there. This is the case of
the backward message itself: $\Lambda = A^\top R^{-1}A$ with $AN = 0$ always has $N^\top DN = 0$, so a
dual $A$ that leaves a rank-deficient value (as $A = A_0 + tE$ from a rank-deficient $A_0$) is not
refused, as in P26 (seen in a scratch probe with $A_0 = 0$, not in the suite). Zygote is not covered (`_complement` stays non-differentiable, P23).
Higher-order derivatives are refused.

**Tests** (`test/forms_crossing.jl`, "P25: …", the oracle is the value path at $t$ and $t \pm h$):
a backward message whose $A(t)$ keeps its rank while its range, its null space or both move, into
`GaussianFactor` forward and backward, a three-channel `LinearConstraintFactor`,
`SwitchingConstraintFactor` to its continuous channel and its outcome, and two such inputs into one
`LinearConstraintFactor`; a fixed flat input with the factor's coefficients dual; flat, proper,
trivial, mixture and categorical messages all reached (281 checks). Refusals and identities: a
tangent with an $N^\top DN$ part refused and one without it accepted, $N(\varepsilon)$ orthonormal
and $\Lambda(\varepsilon)N(\varepsilon) = 0$ to first order, nested duals refused (in the split and in
`_range_complement`), $B$ with two equal
columns separated by the tangent refused and kept parallel accepted, $\log\operatorname{pdet}$
against central differences (200 checks). Run against the code before P25 the first testset throws
(`ArgumentError` of the split, `svd` of duals). The former P21 `@test_broken` is one of its cases.

## Free energy with hard inputs (2026-11-11)

A message from a `HardConstraintFactor` can have hard directions: the relation fixes the variable
along them. The free energies of `GaussianFactor`, `GaussianPrior` and `LinearConstraintFactor`
take such a message. They substitute its hard directions out, as they substitute a point
(`_reduced_relation`, `constraint.jl`): the input becomes $x = x_0 + U s$, with $U$ orthonormal
along its other directions, and the factor's residual becomes $\tilde G s - \tilde c$. The factor
belief's entropy is then taken in $s$, relative to Lebesgue measure on its support, as for the
variables (`lib/LenticulumCore.jl/src/extended_gaussian.md`, "Entropy").

`LinearConstraintFactor.residual_statistics` is this computation for every input
(`_reduced_residual_statistics`). It is the old arithmetic term for term: a point is substituted,
and anything else enters with $U = I$. `GaussianFactor` and `GaussianPrior` keep their own code and
call it only when a message is hard (`_is_hard`). With a learned noise or covariance they refuse a
hard message in the free energy. Checked by the tree oracle of `test/hard_free_energy.jl`, where
hard messages reach both (`hard_constraint.md`, "Measured").

**The rest (P2, 2026-11-17).** `SwitchingConstraintFactor`'s `switch_statistics` is built on
`_reduced_relation` too, so its term takes hard inputs (`switching.md`). The frozen factor belief
of the gradient, `_affine_gaussian_belief` (`GaussianFactor`, `GaussianPrior`,
`LinearConstraintFactor`, `SwitchingConstraintFactor`), substitutes hard directions the same way:
it solves the belief in $s$, takes the entropy there, and lifts the mean and covariance back to
$x$ ($x_0 + U m_s$, $U \Sigma_s U^\top$, singular along the hard directions), so that
`_frozen_residual` reads it unchanged; a channel hard in every direction is held as the point
$x_0$. With a learned noise or covariance `factor_belief` refuses a hard message, as
`local_free_energy` does. The plate paths needed no code: block messages holding extended
Gaussians (`VectorBlock`s) make the batched `factor_beliefs` return `nothing`, and the free energy
and the gradient go replica by replica through the methods above.

Tests (`test/hard_inputs.jl`; every testset fails or errors against the code before this change):
the envelope gradient and the replay with `DiagonalCoupling` on every edge against d$(-\log Z)$ of
the dense model (12 random trees, to $10^{-7}$); a switching factor with an observed outcome beside
a hard relation, $F = -\log Z$ and its gradient by `(A, b, logw)` (15 trees); the switching term
against its softened limit and, with $K = 1$, against `LinearConstraintFactor`'s (60 random
factors); `belief_free_energy ∘ factor_belief` against `local_free_energy` for the four factors;
a plate's free energy and gradient against the sums over its replicas (6 plates of 4); Zygote
against ForwardDiff through the switching term with hard inputs (28 checks, added after the review,
LP-045).

Not handled: the exposed-parameter rules and a learned noise with hard messages. A switching
factor with a latent outcome on a variable a hard relation reaches runs since P27 (2026-11-17):
inference and the parameter gradient; the scalar free energy only under `ExpectationPropagationRule()`
(a mixture marginal has no `variable_entropy`). The free energy and gradient were run on one probe
graph, not checked against an oracle.

## Batched factor beliefs (2026-11-03)

`FrozenGaussianBeliefs` is `FrozenGaussianBelief` for the $n$ replicas of a plate row: points as
$d \times n$ matrices, the free channels' joint means as the columns of `M`, and one covariance and
entropy for all, which holds when every free input is a Gaussian block with one shared precision
(`lib/Mycelium.jl/src/blocks.md`). `_affine_gaussian_beliefs` builds it as
`_affine_gaussian_belief` does, one column per replica; `_frozen_residuals` returns the residual
means as columns and the shared covariance, and `_column_fits` the $n$ fits. `GaussianPrior` and
`LinearConstraintFactor` implement `factor_beliefs`, `belief_free_energies` and
`local_free_energies` with them; other inputs return `nothing` and take the per-replica path.
`GaussianPrior` also has a block `factor_messages`: one shared `GaussianBlock` when the replicas
share mean and covariance. Checked against the same messages per edge in `test/plates.jl`.

## Batched prior messages (2026-11-02)

`Mycelium.factor_messages` for a vector of tied `GaussianPrior`s (a plate) converts the mean and
covariance to canonical form once when the replicas share them, and gives every replica that
message; otherwise each replica is converted on its own. The covariance is read through
`noise_covariance(f, ps)`, so a learned covariance is handled; caching the belief in the factor
would not be. Checked by `test/plates.jl` (the grouped sweep against the sequential one).

## Rank tolerance follows the element type (GH-17, 2026-11-17)

`_rank_deficient_svd` used `eps(Float64)` for singular values computed in Float32, so a Float32
map that is singular up to Float32 rounding (e.g. `Float32[1 2; 2 4]`, second singular value
`6.6e-8`) was called full rank and the no-input forward message dropped the constraint the output
still carries orthogonal to `range(A)`. The threshold now uses `_eps_eltype(eltype(s))`:
`eps(Float32)` for Float32 decompositions, `eps(Float64)` otherwise (duals included). The same
audit fixed `_range_svd` and `_projected_rank` in `constraint.jl` (the projection `Li * At`
promotes to the wider element type, so its tolerance is that type's) and the rank test of
`HardConstraintFactor`'s free energy; `exposure.jl` already promotes to Float64 explicitly and
`LenticulumCore._eig_tol` already uses the element type, so they were left as they are. Tests:
Lenticulum `test/p29_rank_tolerance.jl`, "GH-17" — Float32 deficient against a Float64 oracle that
keeps the same rounding-level singular value, full-rank Float32 maps, and the no-input forward
message's precision along `[-2, 1]/sqrt(5)`.

## Element type (P30, 2026-11-18)

Messages, marginals and free-energy terms are in the type that Julia's promotion gives the values a
computation reads (`A`, `b`, the noise, the prior's `μ` and `Σ`, the incoming messages, the data);
nothing is converted. Float32 parameters, noise and data give a Float32 graph end to end; Float32
parameters with Float64 data give Float64. Before P30 the zero buffers (`zeros(n)`), identities
(`Matrix{Float64}(I, …)`, `Matrix(1.0I, …)`), `log(2π)`, `2π .* Q`, the `0.0` energies and the
`Vector{Float64}` copies in `FrozenGaussianBelief(s)` put Float64 into every message. They now take
the element type of their inputs; `log(2π)` is `LenticulumCore._log2π(T)`; an identically zero energy
or entropy is `false`, as in Mycelium (`free_energy.md`, "Element type in, element type out"), and
its negation stays `false` (`_negentropy`; `-false` is the Int `0`). The default
`eigenvalue_floor` of learnable noise is `one(T) / 1000000` in the noise's float type (the same value
as before for Float64). The rank memo (`_RANK_MEMO`) holds `Matrix{Float32}` as well as
`Matrix{Float64}`, with the same weak-reference and copy checks.

Three Float64 sources sat in LenticulumCore's extended-Gaussian algebra, which every partly flat or
hard message goes through, and were fixed with it: the product `_conjoin` took `sqrt` of its Int
default weights, `preimage` floated its Int default offset, and `extended(::TrivialBelief, n)` had no
element type to take (it now has an optional third argument, used by `HardConstraintFactor`); also
`_logsum`, `_scale` and `marginalise` (`extended_gaussian.md`, "Element type").

Struct changes: `FrozenGaussianBelief{P,M,S,H,Q}` and `FrozenGaussianBeliefs{P,M,S,H}` take their
points' and entropy's types; `LinearConstraintFactor` its fixed output coefficient's (`C`);
`HardConstraintFactor` its noise matrix's (`Q`).

Test: `test/p30_element_type.jl`. Random trees of `GaussianPrior` and `GaussianFactor`s (square and
rectangular `A`, so that extended messages occur; fixed and learnable noise; an observed leaf), built
from one draw in Float32 and in Float64: every message and marginal of the Float32 graph is Float32,
its marginals and free energy agree with the Float64 graph's to `rtol = 1e-3`, and the Float64 graph
agrees with a dense joint solve to `1e-9`; every part of `factor_free_energies` and
`variable_corrections` is Float32 or `false`. The same for two priors joined by a
`GaussianFactor`, a `LinearConstraintFactor` and a `HardConstraintFactor`, node by node and as a
plate of three replicas, against the closed-form conditioning; frozen beliefs, mean-field messages
and energies, the per-replica batched constraint kernel against the per-node message, both noise
hyperpriors, Int inputs, ForwardDiff and Zygote derivatives in Float32, the promotion case, and the
memo (a hit, and a matrix changed in place recomputed).

P31 extends this rule to the remaining Lenticulum factors and the shared hard-constraint plate
kernel; see their implementation notes and `test/p31_element_type.jl`. Upstream cluster inference,
mixture overlap weights, and EM statistics still inject Float64; those limits are recorded there.
Device arrays were not tested with Float32 parameters beyond the existing device tests.

## Element types (Q4)

Covariances and normalisers are evaluated in the type promoted from the parameters and
incoming beliefs. A source prior's outgoing message has only its stored data and any learned
noise parameters to determine its type. Residual and block-diagonal buffers use the arrays
that supply their element type. Scalar and one-row or one-column rank checks can use a
generic norm when LAPACK is unavailable. A general SVD still uses LAPACK. The rank memo is
specialised for CPU Float32 and Float64, and the default Lux bias initializer remains
Float64 because it runs before data are available.
A nested-dual offset leaves a fixed covariance in its stored type. This keeps static flat
subspaces on the existing value path and preserves the circuit Hessian in
`test/p9_relative_tol.jl`.

For first-order duals a precision with zero partials still has the promoted type.
`test/forms_crossing.jl` checks those zero partials and exercises both primal and dual
storage for a constant incoming precision.

## Device arrays (Q7)

The fixed and learned covariances follow the numeric inputs, including in the prior and
collapsed affine games. Dense Cholesky solves, inverses and normalisers use the core kernels.
Rank tests of the affine coefficient use the host spectral helper for their singular values. The
chain test compares inference and scalar free energy with the CPU graph.
