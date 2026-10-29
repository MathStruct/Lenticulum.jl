#implementation

> [RED-Diff](https://arxiv.org/abs/2305.04391) (Mardani et al. 2023) as a proximal operator.
> Variational inference whose posterior is a **point mass** and whose prior gradient is a
> **stop-gradient**, which together make the whole inversion forward-only.

> Sources: code: `reddiff.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Gradient Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses)

## 1. The objective

Specialising the paper's Eq. 8 to the linear clamp of [[ImplicitREDDiff]]:

$$
E(z; z_0) \;=\;
\underbrace{\mathbb{E}_{t,\varepsilon}\bigl[\omega(t)\|\varepsilon_\theta(\alpha_t z+\sigma_t\varepsilon,\,t)-\varepsilon\|^2\bigr]}_{\text{the learned prior}}
\;+\;
\underbrace{\tfrac12\|P(z-z_0)\|^2}_{\text{data consistency}}
$$

The variational family is $q(z\mid z_0) = \mathcal{N}(\mu,\sigma^2 I)$ with $\sigma\to 0$.
**The posterior is a Dirac by the paper's own construction**, not by a simplification made
here — see `factor.md` §5 for what that costs on a graph.

> [!note] Notation
> RED-Diff writes $x_0$ for the clean signal and $y$ for the measurement; [[ImplicitREDDiff]]
> writes $x$ for the state and $x_0$ for the clamp. This vault writes $z \in Z$ for the joint
> state and $z_0$ for the evidence (clamp values and anchors), as fixed in
> [[Channels and Polarity]] §"Notation".

## 2. Proposition 2 is the whole implementation

$$
\nabla_z\,\mathrm{reg}(z)
\;=\;
\mathbb{E}_{t,\varepsilon}\bigl[\lambda_t\,\bigl(\underbrace{\varepsilon_\theta(x_t,t)}_{\text{stop-gradient}}-\varepsilon\bigr)\bigr],
\qquad
\lambda_t = \frac{\lambda}{\sqrt{\mathrm{SNR}_t}} = \frac{\lambda\sigma_t}{\alpha_t}
$$

Two different things are happening in that formula and they are worth separating:

- the **reparametrisation is kept**: $x_t = \alpha_t z + \sigma_t\varepsilon$ is differentiable
  in $x$, and the $\alpha_t$ from that path is absorbed into $\lambda_t$;
- the **denoiser Jacobian is dropped**: the true gradient carries $J_\theta^\top =
  (\partial\varepsilon_\theta/\partial x_t)^\top$ in front of $(\varepsilon_\theta -
  \varepsilon)$, and RED-Diff replaces it with the identity.

So it is a *preconditioner* approximation, not a bias-free estimator. In the vault's
vocabulary ([[abstract_types]]) the sampling path is `PathwiseCoupling` and the network
Jacobian is stopped — which is not exactly any one of the four `AbstractGradientCoupling`
constructors, and is worth noting as a fifth case rather than forced into an existing one.
This is a mathematical comparison, not a coupling argument used by the implementation:
non-default graph annotations are refused at construction (LP-011 option 2). RED-Diff's
closed-form regulariser is separate from the planned per-edge gradient machinery.

> [!important] This is why the package has no AD dependency
> The regulariser needs one **forward** pass per Monte-Carlo draw. The clamp's gradient is
> $P^2(z-z_0)$ in closed form. So the entire inversion is forward passes plus arithmetic, and
> `Project.toml` lists `LuxCore`, `Random` and `LinearAlgebra`. The stop-gradient is usually
> sold as a memory optimisation; here it is the difference between a package that depends on
> Zygote and one that does not.

## 3. The field weight and the likelihood bound

RED-Diff's $\lambda_t=\lambda\sigma_t/\alpha_t=\lambda/\sqrt{\mathrm{SNR}_t}$
grows steeply with noise, but is finite at $t=1$ for the default VP schedule:
$\alpha_1=e^{-5.025}>0$ and $\lambda_1=\lambda\sqrt{e^{10.05}-1}$.
Divergence requires a schedule whose terminal signal vanishes.

With the power convention $\mathrm{SNR}_t=\alpha_t^2/\sigma_t^2$, the continuous
ELBO squared-noise-error coefficient under $dt$ is
$-\tfrac12(d/dt)\log\mathrm{SNR}_t=\beta(t)/(2\sigma_t^2)$ ([[schedule]] §3).
Under $t\sim U[t_{\min},1]$ its coefficient includes $1-t_{\min}$.
RED-Diff instead weights a stop-gradient residual field under that uniform distribution;
it drops the denoiser Jacobian. Comparing these two quantities by squaring weights does
not establish a likelihood bound or explain mode seeking. Its point variational family and
tuned surrogate target reconstruction rather than posterior calibration.
The Gaussian calibration below measures the resulting prior strength directly.

## 4. λ is derivable, not merely tunable — and one λ is not enough

This is the sharpest thing this package has to say about RED-Diff, and both halves are
asserted in the test suite.

### 4.1 For isotropic Gaussian data there is exactly one correct λ

For data $z\sim p_0=\mathcal{N}(0,v_0 I)$ the model is available in closed form,
$\varepsilon_\theta(x,t) = \sigma_t x/D_t$ with $D_t = \alpha_t^2v_0+\sigma_t^2$, and the
expectation collapses:

$$
\mathbb{E}_\varepsilon\bigl[\varepsilon_\theta(\alpha_t z + \sigma_t\varepsilon,t)-\varepsilon\bigr]
= \frac{\sigma_t\alpha_t}{D_t}\,z
\qquad\Longrightarrow\qquad
\nabla_z\mathrm{reg}(z) = \kappa z,
\quad
\kappa = \frac{\lambda}{1-t_{\min}}\!\int_{t_{\min}}^{1}\!\frac{\sigma_t^2}{D_t}\,dt
$$

The true prior gradient is $z/v_0$. So RED-Diff's regulariser **is** a Gaussian prior, of
precision $\kappa$ — and it is the *right* prior only when $\kappa = 1/v_0$.
`calibrate_lambda` returns that $\lambda$ by quadrature.

With it, the RED-Diff fixed point $\kappa z + \rho^2(z-z_0) = 0$ gives
$z = \rho^2z_0/(1/v_0+\rho^2)$, which is **exactly** the Gaussian posterior mean. The test
suite checks this to within Monte-Carlo noise and checks that the noise averages away.

> Tuning $\lambda$ is not a knob on "how much regularisation feels right". It is **choosing
> how strong the learned prior is**, and a wrong value biases every posterior by a computable
> factor. The paper's tuned $\lambda = 0.25$ is a different quantity from
> $\lambda^\star \approx 1.379826$ for unit-variance data under the default schedule.

### 4.2 For correlated data no single λ works

Extend to $p_0=\mathcal{N}(0,\Sigma)$. Both $\kappa$ and $\Sigma^{-1}$ are functions of
$\Sigma$ and are simultaneously diagonalisable, so per eigenvalue $s$ the requirement is
$\lambda\,s\,I(s) = 1$ with $I(s) = \frac{1}{1-t_{\min}}\int_{t_{\min}}^1\sigma_t^2/(\alpha_t^2 s+\sigma_t^2)\,dt$. That needs
$s\,I(s)$ to be **constant in $s$**. It is not:

| $s$ | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| $s\,I(s)$ | 0.206878 | 0.389614 | 0.724729 | 1.331492 | 2.417511 |

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
test suite, individual runs land within 2.184% of the exact posterior mean and the *average over
seeds* within 0.915%. Nothing in `REDDiff` implements a decreasing step size, Polyak averaging
or a convergence check — `reddiff_solve` runs exactly `steps` iterations and returns.

**There is no `ConvergenceReport`.** Every other inversion in this project reports whether it
converged; this one cannot, because there is no residual to measure. That is a real gap
against `Mycelium`'s conventions.

### 5.2 Adam is hand-rolled

Five lines, inline. `Mycelium` has `AbstractUpdateRule` with `GradientDescent`, `Momentum`
and `Nesterov`, and reusing them was considered and rejected: those are *lenses for parameter
updates in a graph* ([Learning Components as Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses)), whereas this loop
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
categorical *cup* of [[Open Models and Latent Channels]] — but it means the two clamp regimes take
different code paths, and the finite-$\rho$ path never continuously approaches the infinite
one as $\rho$ grows. It just gets stiffer, and eventually ill-conditioned.

### 5.5 `samples = 1` is the paper's setting and is very noisy

Algorithm 1 draws one $(t,\varepsilon)$ per step. That works when you take thousands of steps
on a real image; on the small problems here it needs `samples` in the tens to give a stable
answer in hundreds of steps. The default is 1, matching the paper; the tests raise it.

Related: [[schedule]], [[predictor]], [[factor]], [[RED-Diff as a Statistical Game]],
[[ProxDM and Proximal Alternatives]], [[ImplicitREDDiff]]

## LP-018 calibration measurements (2026-10-29)

Time is uniform on `[tmin,1]`; both the shrinkage and `I(s)` above include
`1/(1-tmin)`. `examples/lp018_calibration.jl` reproduces the table (independent
20,000-point midpoint quadrature), the default calibrated value `1.379825576009811`
(previously `1.3812067827926036`), and the six-seed Gaussian posterior measurements.
At clamp precisions 1 and 2, the maximum relative errors are 2.1833% and 1.4030%;
the errors of the six-seed means are 0.9147% and 0.5028%. Settings match the existing
Gaussian posterior regression: 1,500 steps, 64 samples, learning rate 0.1, no Adam.
The corresponding terminal weight at unit lambda is `152.16697028394637`.
