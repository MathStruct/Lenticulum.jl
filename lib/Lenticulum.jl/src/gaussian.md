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
  The `TrivialBelief()` input (no message) keeps the old shortcut, which returns `TrivialBelief()`
  and drops that information. Whether it should also take the exact path is a row of the
  [[Feature Backlog]]: it changes the first sweep of every schedule.
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

Not checked: extended inputs in plates (`_block_parts` returns `nothing` for them, which should
send a plate down the per-replica path; not run), in the AD paths (`_gradient_message`), and to
`NonlinearFactor` with a partly flat cavity, which `_expansion` refuses through `_moments`.

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

**Known limit** (`@test_broken`): a dual flagged message entering a relation factor's flat-input
path (`_flat_message`, `_linear_message` with flat inputs) throws a `MethodError`, since the split
and the residual metric use an eigendecomposition and an SVD. With only $\eta$ dual the flat
directions do not move and a derivative exists; it is a row of the [[Feature Backlog]] (P21). Also
`@test_broken`, and older than P1: `NonlinearFactor`'s backward message at a dual expansion point
throws in `_jacobian`, which writes into a `Float64` matrix.

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
hard messages reach both (`hard_constraint.md`, "Measured"). Not handled: `SwitchingConstraintFactor`
(its `_canon` refuses hard directions), the plate paths, the frozen beliefs of the gradient
(`factor_belief`), and the exposed-parameter rules.

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
