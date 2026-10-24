#implementation

> `NonlinearFactor`: $p(y\mid x) = \mathcal N(y; f(x), Q)$ for a **hand-written** $f$, with
> Gaussian messages by linearisation (the extended Kalman filter) or the unscented transform
> (the unscented Kalman filter). The factor that known physics needs: a measurement model or
> known dynamics, next to a learned relation in the same graph.

> Sources: code: `nonlinear.jl`, `gaussian.jl`; Julier & Uhlmann, *A New Extension of the Kalman Filter to Nonlinear Systems*, 1997; Wan & van der Merwe, *The Unscented Kalman Filter for Nonlinear Estimation*, 2000; Bell & Cathey, *The iterated Kalman filter update as a Gauss–Newton method*, IEEE TAC 1993; García-Fernández, Svensson & Särkkä, *Iterated Posterior Linearization Smoother*, IEEE TAC 2017; the RxInfer delta-node approximations ([[Sharing Types with RxInfer]] §4)
>
> Bibliography: [[Bibliography#^julier1997ukf|Julier & Uhlmann 1997]] · [[Bibliography#^wan2000ukf|Wan & van der Merwe 2000]] · [[Bibliography#^bell1993iekf|Bell & Cathey 1993]] · [[Bibliography#^garciafernandez2017iterated|García-Fernández et al. 2017]]
>
> Theory: [[Belief Algebra]] · [[RxInfer as a Backend]] · [[Open Problems in Implicit Diffusion Learning]]

## 1. One local linear model, then GaussianFactor's algebra

Every message goes through a local model $y \approx Ax + b + \nu$, $\nu \sim \mathcal N(0, Q + \Omega)$,
around an expansion belief $\mathcal N(\bar x, P)$ on $x$ (`linearize`):

| method | $A$ | $b$ | $\Omega$ |
|---|---|---|---|
| `Linearization()` | $\partial f(\bar x)$ | $f(\bar x) - A\bar x$ | 0 |
| `Unscented()` | $C_{xy}^\top P^{-1}$ on the $2n+1$ sigma points | $\bar y - A\bar x$ | $S_{yy} - APA^\top$ |

The unscented row is statistical linear regression: the line that best explains $f$ over the
belief, plus the spread it does not explain. After that the messages are `GaussianFactor`'s for
$(A, b, Q + \Omega)$: forwards $\mathcal N(Am + b, ASA^\top + \Omega + Q)$, backwards the likelihood
$\mathcal N^{-1}(A^\top R^{-1}(m_y - b), A^\top R^{-1}A)$ with $R = Q + \Omega + S_y$.

## 2. Where to linearise

- **Forwards** around the incoming belief on $x$. A point input is exact: $\mathcal N(f(x), Q)$.
- **Backwards** around the **cavity**, the `prior` argument of `factor_message` (the product of
  the other messages into $x$), which Mycelium already passes. It is used only as the expansion
  point, never multiplied in, so nothing is counted twice; and as a schedule sweeps again the
  cavity sharpens and the linearisation improves (the expectation-propagation pattern). With no
  cavity the message is uninformative, unless the factor was given an `x0`. This makes the
  message depend on the schedule: a `TreeSchedule` runs one sweep and may reach the factor before
  the cavity has arrived, so use `SequentialSchedule(tasks(tree_schedule(g)))`, which sweeps until
  the messages stop changing (three sweeps for prior, measurement and data).
- **`invert`** relinearises the posterior `iterations` times: with `Linearization` that is the
  iterated EKF, i.e. Gauss–Newton on the MAP; with `Unscented` it is iterated posterior
  linearisation. On $x \sim \mathcal N(1, 0.5^2)$, $y = x^3 + \mathcal N(0, 0.1^2)$, $y = 2$, thirty
  iterations reach the MAP to $10^{-5}$ and the covariance is the Gauss–Newton curvature there; one
  EKF step is visibly off.

## 3. Design decisions

- **No AD dependency.** The Jacobian is the user's `jacobian(x)` or central differences, as the
  diffusion package's fallback is; pass `x -> ForwardDiff.jacobian(f, x)` for an exact one.
- **Written here, not borrowed.** ReactiveMP has these approximations, but importing it would
  make `Lenticulum` depend on RxInfer's engine for ~150 lines ([[Sharing Types with RxInfer]] §4).
- **Not learnable.** `f` is fixed and `ps` is empty. A parametrised `f(x, θ)` with gradients is a
  separate step.
- **The free energy is the linear model's.** `local_free_energy` builds the `GaussianFactor` of the
  local model around the incoming belief on $x$; exact for affine $f$, the usual EKF/UKF evidence
  approximation otherwise.

## 4. How it is tested

`test/runtests.jl`: for an affine $f$ both methods give `GaussianFactor`'s messages and free
energy exactly, in both directions; for $y = x^2$ the unscented forward message has the exact
mean $m^2 + s^2$ and variance $4m^2s^2 + 2s^4$, the linearised one neither; the finite-difference
Jacobian matches a supplied one; iterated inversion reaches the MAP; and in a graph
(prior, $x^3$ measurement, data) repeated sweeps converge to one EKF update.

## 5. Missing

Learnable parameters in $f$; Gauss–Hermite and cubature rules; CVI; and multi-input functions
$y = f(x_1, \dots, x_k)$ over several channels (today one input channel, which may be a stacked
vector).

Related: [[constraint]], [[likelihood]] (non-Gaussian observations), [[Belief Algebra]], [[Sharing Types with RxInfer]]

The lens's `likelihood_message` performs the same posterior relinearisation loop as its
inversion and returns the final Gaussian site. `invert` combines that site with the original
prior once. The iterated nonlinear regression test checks both Taylor and unscented methods,
including the scalar square-function example with four iterations.
