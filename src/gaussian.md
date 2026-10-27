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
  but singular. `GaussianBelief(η, Λ)` stores canonical parameters so the likelihood message can be
  represented without inverting `Λ`; moment conversion (`belief_mean`, `belief_cov`) is deferred
  until a proper prior or cavity message is combined via `Mycelium.combine`.
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
Both are checked by `check_factor_conformance` (`test/factor_conformance.jl`).

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
the legacy fixed-cavity local-posterior derivative; the latter is not a true EM M-step.

Evidence: `lib/Mycelium.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].
