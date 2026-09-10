# reddiff.jl — implementation note

> [RED-Diff](https://arxiv.org/abs/2305.04391) (Mardani et al. 2023) as a proximal operator.
> Variational inference whose posterior is a **point mass** and whose prior gradient is a
> **stop-gradient**, which together make the whole inversion forward-only.

## 1. The objective

Specialising the paper's Eq. 8 to the linear clamp of [[ImplicitREDDiff]]:

$$
E(x_0, x) \;=\;
\underbrace{\mathbb{E}_{t,\varepsilon}\bigl[\omega(t)\|\varepsilon_\theta(\alpha_t x+\sigma_t\varepsilon,\,t)-\varepsilon\|^2\bigr]}_{\text{the learned prior}}
\;+\;
\underbrace{\tfrac12\|P(x_0-x)\|^2}_{\text{data consistency}}
$$

The variational family is $q(x_0\mid y) = \mathcal{N}(\mu,\sigma^2 I)$ with $\sigma\to 0$.
**The posterior is a Dirac by the paper's own construction**, not by a simplification made
here — see `factor.md` §5 for what that costs on a graph.

## 2. Proposition 2 is the whole implementation

$$
\nabla_x\,\mathrm{reg}(x)
\;=\;
\mathbb{E}_{t,\varepsilon}\bigl[\lambda_t\,\bigl(\underbrace{\varepsilon_\theta(x_t,t)}_{\text{stop-gradient}}-\varepsilon\bigr)\bigr],
\qquad
\lambda_t = \frac{\lambda}{\mathrm{SNR}_t} = \frac{\lambda\sigma_t}{\alpha_t}
$$

Two different things are happening in that formula and they are worth separating:

- the **reparametrisation is kept**: $x_t = \alpha_t x + \sigma_t\varepsilon$ is differentiable
  in $x$, and the $\alpha_t$ from that path is absorbed into $\lambda_t$;
- the **denoiser Jacobian is dropped**: the true gradient carries $J_\theta^\top =
  (\partial\varepsilon_\theta/\partial x_t)^\top$ in front of $(\varepsilon_\theta -
  \varepsilon)$, and RED-Diff replaces it with the identity.

So it is a *preconditioner* approximation, not a bias-free estimator. In the vault's
vocabulary ([[abstract_types]]) the sampling path is `PathwiseCoupling` and the network
Jacobian is stopped — which is not exactly any one of the four `AbstractGradientCoupling`
constructors, and is worth noting as a fifth case rather than forced into an existing one.

> [!important] This is why the package has no AD dependency
> The regulariser needs one **forward** pass per Monte-Carlo draw. The clamp's gradient is
> $P^2(x-x_0)$ in closed form. So the entire inversion is forward passes plus arithmetic, and
> `Project.toml` lists `LuxCore`, `Random` and `LinearAlgebra`. The stop-gradient is usually
> sold as a memory optimisation; here it is the difference between a package that depends on
> Zygote and one that does not.

## 3. λ_t grows with t, and that is the mode-seeking

$\lambda_t = \lambda\sigma_t/\alpha_t$ is **unbounded as $t\to1$**: the most heavily noised
times get the most weight. Compare the ELBO weighting $\sigma_t^2/\alpha_t^2$ — wait, that
grows too — but compare the *relative* emphasis: the ELBO's weight is the square of
RED-Diff's, so RED-Diff systematically down-weights the late times relative to a proper
likelihood bound while still leaning on them.

The practical reading is the paper's own: RED-Diff is **mode-seeking**, not
distribution-matching. It finds a point, not a posterior, and the weighting is tuned for
reconstruction quality rather than calibration.

## 4. λ is derivable, not merely tunable — and one λ is not enough

This is the sharpest thing this package has to say about RED-Diff, and both halves are
asserted in the test suite.

### 4.1 For isotropic Gaussian data there is exactly one correct λ

For $x_0\sim\mathcal{N}(0,v_0 I)$ the model is available in closed form,
$\varepsilon_\theta(x,t) = \sigma_t x/D_t$ with $D_t = \alpha_t^2v_0+\sigma_t^2$, and the
expectation collapses:

$$
\mathbb{E}_\varepsilon\bigl[\varepsilon_\theta(\alpha_t x + \sigma_t\varepsilon,t)-\varepsilon\bigr]
= \frac{\sigma_t\alpha_t}{D_t}\,x
\qquad\Longrightarrow\qquad
\nabla_x\mathrm{reg}(x) = \kappa x,
\quad
\kappa = \lambda\!\int_{t_{\min}}^{1}\!\frac{\sigma_t^2}{D_t}\,dt
$$

The true prior gradient is $x/v_0$. So RED-Diff's regulariser **is** a Gaussian prior, of
precision $\kappa$ — and it is the *right* prior only when $\kappa = 1/v_0$.
[`calibrate_lambda`](@ref) returns that $\lambda$ by quadrature.

With it, the RED-Diff fixed point $\kappa x + \rho^2(x-x_0) = 0$ gives
$x = \rho^2x_0/(1/v_0+\rho^2)$, which is **exactly** the Gaussian posterior mean. The test
suite checks this to within Monte-Carlo noise and checks that the noise averages away.

> Tuning $\lambda$ is not a knob on "how much regularisation feels right". It is **choosing
> how strong the learned prior is**, and a wrong value biases every posterior by a computable
> factor. The paper's tuned $\lambda = 0.25$ is a different quantity from
> $\lambda^\star \approx 1.38$ for unit-variance data under the default schedule.

### 4.2 For correlated data no single λ works

Extend to $x_0\sim\mathcal{N}(0,\Sigma)$. Both $\kappa$ and $\Sigma^{-1}$ are functions of
$\Sigma$ and are simultaneously diagonalisable, so per eigenvalue $s$ the requirement is
$\lambda\,s\,I(s) = 1$ with $I(s) = \int\sigma_t^2/(\alpha_t^2 s+\sigma_t^2)\,dt$. That needs
$s\,I(s)$ to be **constant in $s$**. It is not:

| $s$ | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| $s\,I(s)$ | 0.207 | 0.389 | 0.724 | 1.330 | 2.415 |

A factor of ~12 across a 16× spread of eigenvalues. Calibrating at $s=1$ therefore

- **over**-regularises high-variance directions ($s=4$ gets $\approx 3.3\times$ too much), and
- **under**-regularises low-variance ones ($s=0.25$ gets $\approx 0.29\times$).

which is an **over-smoothing bias** — and over-smoothed reconstructions are precisely what
RED-Diff is empirically criticised for. The theory predicts the known artefact. Asserted in
the test suite by pure quadrature, no Monte Carlo.

## 5. Implementation difficulties

### 5.1 The fixed point is only approached, never reached

The gradient is stochastic and does not vanish at the optimum, so constant-step-size descent
hovers in a neighbourhood of radius $\propto \mathrm{lr}\cdot\sqrt{\mathrm{Var}}$. In the
test suite, individual runs land within ~2% of the exact posterior mean and the *average over
seeds* within ~0.4%. Nothing in `REDDiff` implements a decreasing step size, Polyak averaging
or a convergence check — `reddiff_solve` runs exactly `steps` iterations and returns.

**There is no `ConvergenceReport`.** Every other inversion in this project reports whether it
converged; this one cannot, because there is no residual to measure. That is a real gap
against `Mycelium`'s conventions.

### 5.2 Adam is hand-rolled

Five lines, inline. `Mycelium` has `AbstractUpdateRule` with `GradientDescent`, `Momentum`
and `Nesterov`, and reusing them was considered and rejected: those are *lenses for parameter
updates in a graph* ([[Learning Components as Parametric Lenses]]), whereas this loop
optimises a **belief**, not a parameter. Different object, different place. If this file ever
needs a third optimiser, that judgement should be revisited.

### 5.3 `calibrate_lambda` only knows about Gaussians

It is exact for isotropic Gaussian data and meaningless otherwise — there is no calibration
procedure for a real image prior, and §4.2 says a single scalar could not express one anyway.
Its value is diagnostic: it tells you what $\lambda$ *means*, on the one family where the
answer is computable.

### 5.4 The hard clamp is a projection, not a limit

$\rho_{in} = \infty$ cannot go in a gradient, so coordinates with infinite precision are
**overwritten** after every step (`_project!`). That is the correct reading — it is the
categorical *cup* of [[Copiers Cups and Caps]] — but it means the two clamp regimes take
different code paths, and the finite-$\rho$ path never continuously approaches the infinite
one as $\rho$ grows. It just gets stiffer, and eventually ill-conditioned.

### 5.5 `samples = 1` is the paper's setting and is very noisy

Algorithm 1 draws one $(t,\varepsilon)$ per step. That works when you take thousands of steps
on a real image; on the small problems here it needs `samples` in the tens to give a stable
answer in hundreds of steps. The default is 1, matching the paper; the tests raise it.

Related: [[schedule]], [[predictor]], [[factor]], [[RED-Diff as a Statistical Game]],
[[ProxDM and Proximal Alternatives]], [[ImplicitREDDiff]]
