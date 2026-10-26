#implementation

> `LikelihoodFactor`: a Gaussian latent $x$ observed through a **non-Gaussian likelihood** of a
> linear predictor, $p(y \mid x) = L(y \mid Ax + b)$, for counts, dependent counts and ordered
> categories. The message to $x$ stays Gaussian, by the Laplace approximation or by expectation
> propagation; the message to $y$ is the predictive distribution over the discrete outcomes, a
> `CategoricalBelief`.

> Sources: code: `likelihood.jl`, `nonlinear.jl`, `beliefs.jl`; Minka, *Expectation Propagation for approximate Bayesian inference*, UAI 2001; Tierney & Kadane, *Accurate Approximations for Posterior Moments and Marginal Densities*, JASA 1986 (the Laplace approximation); Liu & Pierce, *A Note on Gauss–Hermite Quadrature*, Biometrika 1994 (the adaptive rule); Golub & Welsch, *Calculation of Gauss Quadrature Rules*, Math. Comp. 1969; McCullagh, *Regression Models for Ordinal Data*, JRSS B 1980; Holgate, *Estimation for the Bivariate Poisson Distribution*, Biometrika 1964
>
> Bibliography: [[Bibliography#^minka2001ep|Minka 2001]] · [[Bibliography#^tierney1986laplace|Tierney & Kadane 1986]] · [[Bibliography#^liu1994gausshermite|Liu & Pierce 1994]] · [[Bibliography#^golub1969gauss|Golub & Welsch 1969]] · [[Bibliography#^mccullagh1980ordinal|McCullagh 1980]] · [[Bibliography#^holgate1964bivariate|Holgate 1964]]
>
> Theory: [[Belief Algebra]] §4 · [[Beliefs]] · [[Categorical Belief]] · [[nonlinear]]

## Graph free-energy edge directions

`LikelihoodFactor` declares `Mycelium.uses_cavity == true`: it reads incoming evidence
and input beliefs for its observed and predictive local free energies.
Graph free-energy evaluation rejects an attached `Emitting()` edge with `ArgumentError`;
use `Bidirectional()` if that channel must also send messages. Missing messages on permitted
edges retain their existing local behaviour. See [[free_energy]] and
[[LP-014 Emitting edges drop the cavity message from the free energy]].

## 1. The factor

The likelihood reads $x$ only through the linear predictor $z = Ax + b \in \mathbb R^k$:

$$
p(y \mid x) = L(y \mid Ax + b).
$$

`A` ($k \times n$) and `b` are fixed. That is the generalised linear model's structure, and it is
what keeps the approximations cheap: whatever the dimension of $x$, the non-Gaussian part lives
in $k$ dimensions.

| likelihood | $y$ | $z$ | $L(y \mid z)$ | log-concave |
|---|---|---|---|---|
| `PoissonLikelihood(k)` | $k$ counts | $k$ log-rates | $\prod_j \mathrm{Poisson}(y_j; e^{z_j})$ | yes |
| `BivariatePoissonLikelihood()` | 2 counts | $\log\lambda_{1,2,3}$ | $y_1 = w_1 + w_3$, $y_2 = w_2 + w_3$, $w_i \sim \mathrm{Poisson}(\lambda_i)$ | no |
| `OrdinalProbitLikelihood(c; scale)` | a category $1 \dots K$ | 1 | $\Phi\bigl(\tfrac{c_k - z}{\sigma}\bigr) - \Phi\bigl(\tfrac{c_{k-1} - z}{\sigma}\bigr)$ | yes |

The bivariate Poisson has $\operatorname{Cov}(y_1, y_2) = \lambda_3$, so two counts that share a cause
can be modelled without leaving the family. The ordinal probit with one cutpoint is binary probit.

A new likelihood is a subtype of `AbstractLikelihood` with `npredictors`, `obsdim` and
`loglikelihood`. Derivatives default to central differences and EP moments to quadrature, so
that is enough for every rule; closed-form derivatives and `outcomes` (for the predictive) are
optional.

## 2. The message to $x$: the cavity times the likelihood, made Gaussian

Backwards, the factor receives the **cavity** (the `prior` argument: the product of the other
messages into $x$) and the observed $y$. The exact posterior $\text{cavity}(x)\, L(y \mid Ax+b)$ is
not Gaussian; the message is the Gaussian $q$ that approximates it, divided by the cavity.

**Work in the predictor's coordinates.** The cavity $\mathcal N(m, S)$ gives $z \sim \mathcal N(\mu_z, \Sigma_z)$,
$\Sigma_z = ASA^\top$. On the range of $\Sigma_z$ write $z = \mu_z + U\zeta$, $\zeta \sim \mathcal N(0, D)$ (an eigendecomposition,
so a singular $\Sigma_z$ is fine: a predictor fixed by the offset, or more predictors than
latents). The likelihood depends on $x$ only through $\zeta$, so the exact posterior is the cavity's
conditional $x \mid \zeta$ times a tilted distribution of $\zeta$. A Gaussian site $s(\zeta)$ with
$\mathcal N(\zeta; 0, D)\, s(\zeta) = q(\zeta)$ therefore gives, mapped back through
$\zeta = U^\top(Ax + b - \mu_z)$, a $q(x)$ whose $\zeta$-marginal is $q(\zeta)$ and whose conditional is exact. The
message is that site, a Gaussian on $x$ with precision $A^\top U \Lambda_s U^\top A$: it has rank at most
$k$, informative only along the rows of $A$.

The two methods differ in $q(\zeta)$:

| method | $q(\zeta)$ | how |
|---|---|---|
| `Laplace()` | mode and curvature of the tilted density | Newton with backtracking (Levenberg damping where the density is not log-concave) |
| `ExpectationPropagation()` | its mean and covariance (the KL projection) | adaptive Gauss–Hermite: the tensor rule of `order` points per dimension, centred and scaled at the Laplace approximation; closed form for the ordinal probit |

The adaptive rule matters. A large count under a wide cavity has a likelihood much narrower
than the cavity, and a rule placed on the cavity would put almost no nodes where the mass is.
Centred at the mode, it does: for $y = 50$ under $\mathcal N(0, 4)$ in log-rate, the moments match brute
force to $10^{-7}$ (test suite).

**How good each is** on one Poisson observation, $x \sim \mathcal N(0.5, 0.3)$, $y = 3$, against
brute-force quadrature on 400 001 points (`test/runtests.jl`):

| | posterior mean error | variance error | $\log p(y)$ error | Bethe $F + \log p(y)$ |
|---|---|---|---|---|
| Laplace | $3.5 \times 10^{-2}$ | $1.4 \times 10^{-3}$ | $3.0 \times 10^{-3}$ | $6.2 \times 10^{-3}$ |
| EP | $1 \times 10^{-11}$ | $4 \times 10^{-11}$ | $6 \times 10^{-13}$ | $2.4 \times 10^{-3}$ |

EP's moments and evidence are exact to quadrature accuracy; the remaining gap in the last column
is the variational one, because the Bethe free energy evaluates the Gaussian $q$ in place of the
true posterior, and it is positive, as a bound must be.

**No cavity.** With an absent or improper cavity there is nothing to place the approximation
at. Like `NonlinearFactor`, the factor then sends the second-order expansion of $\log L$ at `x0`
if one was given (one Newton step from there), and an uninformative message otherwise. In a
graph, use `SequentialSchedule(tasks(tree_schedule(g)))` so the message is recomputed once the
cavity has arrived (three sweeps for prior, observation and data).

**Log-concavity.** For a log-concave likelihood, the tilted variance never exceeds the cavity
variance and the curvature is negative, so every message has a positive semidefinite precision.
The bivariate Poisson is not log-concave: its Hessian is $v\, s s^\top - \operatorname{diag}(\lambda)$, with $v$ the
posterior variance of the shared part $w_3$, and a message can have an indefinite precision.
That is legitimate in EP (the posterior stays proper), but a message alone is then not a belief.

## 3. The message to $y$: the predictive over outcomes

Forwards, the factor pushes the belief on $x$ through the likelihood:
$P(y = o) = \mathbb E\, L(o \mid Ax + b)$ for every outcome $o$ in `outcomes(lik)`, a `CategoricalBelief`
labelled by the outcomes, by Gauss–Hermite quadrature over the range of $\operatorname{Cov}(Ax + b)$ (exact
at a point, and in closed form for the ordinal probit, where the predictive is the same formula
with $\sigma^2$ replaced by $\sigma^2 + a^\top S a$).

The count likelihoods' outcomes are the grid $\{0, \dots, N\}^k$ (`maxcount = N`), with the top
value of each coordinate standing for "$N$ or more": the tail is lumped, not dropped, so the
predictive keeps the whole mass and sums to one. For the bivariate Poisson the lumped margins
use $y_1 \sim \mathrm{Poisson}(\lambda_1 + \lambda_3)$, and the joint tail is what remains.

`predictive(f, belief)` is the same computation outside a graph, and `log_evidence(f, cavity, y)`
the log normaliser of the tilted distribution (the Laplace evidence, or EP's quadrature): the
quantity whose gradient fits hyperparameters by evidence.

## 4. The Bethe contribution

`local_free_energy` evaluates the factor's term with the factor belief replaced by $q$:
`fit` $= \mathbb E_q[-\log L(y \mid Ax+b)]$ by quadrature, `negentropy` $= -H(q)$, `complexity` $= 0$. For
a point $x$ it is $-\log L$ exactly. With a Gaussian prior on $x$ the graph's free energy is then
$\mathbb E_q[-\log \pi] + \mathbb E_q[-\log L] - H(q) \ge -\log p(y)$, the variational bound, and the test
suite checks both the bound and its size (table in §2).

## 5. Design decisions

- **One factor, a likelihood argument.** The approximations need only $\ell$, its derivatives in
  $z$, and the outcomes; so the likelihood is a small type and the factor is written once. A
  likelihood with only `loglikelihood` already works (central differences, quadrature), and the
  test suite runs the ordinal probit both ways to check one against the other.
- **Fixed $A$ and $b$.** Learning them is a separate step, as for `NonlinearFactor`.
- **`SpecialFunctions` as a dependency**, for `erfc`, `erfcx` (the normal CDF far in its lower
  tail, where $\log\Phi$ must not underflow), `logfactorial` and the incomplete gamma function (the
  Poisson tail). $\log(\Phi(a) - \Phi(b))$ is computed in whichever tail avoids cancellation.
- **The cavity is never multiplied in.** As in every factor here, the prior argument only places
  the approximation; the message is $q$ divided by it.

## 6. How it is tested

`test/runtests.jl`, against brute-force quadrature on fine grids: Poisson EP moments and
evidence (to $10^{-8}$) and the Laplace mode and curvature; the large-count case; the ordinal
probit's closed-form EP moments and evidence for every category, the quadrature path with
finite-difference derivatives against them, and its closed-form derivatives against central
differences; a finite gradient far in the tail; a two-dimensional latent read through one
predictor (a rank-one message along the row of $A$, and the exact conditional structure of the
posterior); the predictive in closed form and by quadrature, the lumped Poisson tail, and the
exact probabilities at a point; the bivariate Poisson's normalisation, means and covariance
$\lambda_3$, its exact derivatives, its predictive grid, and its EP mean against a two-dimensional
grid (with Laplace at the mode, which differs from the mean by more than 0.05 there); the
messages without a cavity; and in a graph (prior, Poisson observation, data) convergence to
the single-factor answer and the free-energy bound.

## 7. Missing

- **Damping and repeated EP sweeps** for loopy graphs, where several likelihood sites refine
  each other's cavities; today the schedule recomputes the message, with no damping.
- **Many independent observations of one latent** as separate sites. A `PoissonLikelihood(k)`
  with large $k$ works under Laplace but not under EP (the tensor grid has $\texttt{order}^k$ points);
  use one factor per observation, or Laplace.
- **Learnable $A$, $b$, cutpoints and scale**, and gradients of `log_evidence` with respect to them.
- **More likelihoods**: logit links, negative binomial (overdispersed counts), zero inflation,
  censored and truncated observations.
- **Soft observations**: $y$ must be a point (`DiracBelief`).
- **Cubature rules** (sparse grids) for $k > 4$.

## 8. Predictive free energy

With $y$ unobserved, `local_free_energy` keeps both coordinates: `fit` $= \mathbb E_x\,H(Y \mid x)$ and
`negentropy` $= -\mathbb E_x\,H(Y \mid x) - H(x)$. Their sum is $-H(x)$, as before, but a scalarisation
other than the plain sum now sees the true fit and entropy (earlier both were set to 0 because they
cancel). The ordinal entropy sums its categories exactly. The count likelihoods sum their whole
support, not the lumped predictive grid, until less than $10^{-14}$ of the mass is left out. The
expectation over a Gaussian $x$ is by Gauss–Hermite quadrature, an approximation. A likelihood without
a `_conditional_entropy` method throws. Tested in "predictive free energy retains conditional entropy".

Related: [[nonlinear]], [[constraint]], [[Belief Algebra]], [[Categorical Belief]]
