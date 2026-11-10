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
tree; that has not been done.

**The constructor accepts a covariance that is symmetric up to rounding** (the tolerance of
`LenticulumCore._nearly_symmetric`) and uses its symmetric part. The collapsed factor of a
composition has the noise $B Q_a B^\top + Q_b$, which is not exactly symmetric; the exact test
refused it for most parameters when $Q_a$ was not diagonal.

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
