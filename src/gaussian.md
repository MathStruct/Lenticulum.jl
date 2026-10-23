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

## 3. Why `Q` is fixed in v0

The parameters returned by `LuxCore.initialparameters` are `(A, b)`. The noise covariance `Q`
is held fixed on the struct because unconstrained gradient steps on `Q` can leave the positive-definite
cone; learning `Q` requires either a Cholesky/log-Cholesky reparametrisation or a Wishart prior
channel.

## 4. `likelihood_message` vs. `invert`

In belief propagation, a factor-to-variable message must be a **likelihood** (excluding the
target variable's incoming prior $\pi$), whereas AutoBayes Definition 10 defines the lens
inversion $c^\dagger_\pi(y)$ as the **posterior** (including $\pi$). `GaussianFactor` implements
`LenticulumCore.likelihood_message` for the likelihood $\mu_{c\to t}$ and defines
`LenticulumCore.invert` as `Mycelium.combine(π, μ_{c→t})`, satisfying both contracts cleanly.

## 5. What it does not do

- **Observation covariance `Q` is not learnable (`LP-005`).** `LuxCore.initialparameters` returns
  only `(A, b)`; `Q` is fixed on the struct so gradient steps cannot drive it outside the SPD cone.
- **No precision-form (`Q = 0`) deterministic limit.** `Q` must be strictly positive definite;
  noiseless linear equality constraints require a non-zero regulariser `Q = εI` (see
  `LinearConstraintFactor` in [[constraint]]).
- **`GaussianPrior` is non-learnable in v0.** Its `(μ, Σ)` live on the struct rather than in `ps`.

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

Related: [[constraint]], [[nonlinear]], [[likelihood]], [[beliefs]]
