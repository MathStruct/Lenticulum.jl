# flow.jl — implementation note

> Fixed-step explicit integrators, run forwards and backwards. The whole file exists for one
> line: **`t₁ < t₀` is allowed**, and that is what inverts a NeuralODE.

## 1. The reverse direction is not a feature, it is a theorem

A flow is a diffeomorphism. Integrating $dz/dt = f_\theta(z,t)$ from $t_1$ back to $t_0$
recovers the initial condition, in exact arithmetic, for any $f$ — no invertible architecture
(coupling layers, RealNVP), no learned inverse, no root-find.

So `flow_reverse` is `integrate` with the endpoints swapped, and that is the *entire*
implementation of the reverse polarity. Compare [[deq]], where the reverse direction is a
general nonlinear root-find that may not converge and needs a square residual.

## 2. Why fixed-step, and why not `Tsit5`

An adaptive integrator chooses its steps from the local error estimate, and it chooses
**different steps forwards and backwards**. That does not merely make the round trip
inexact — it makes it inexact in a way that depends on the tolerance and the trajectory, so
`flow_reverse(flow_forward(z))` drifts unpredictably.

Fixed-step RK4 with the same `steps` in both directions is not symplectic and not exactly
reversible either, but its error is deterministic and $O(h^4)$, which is enough for the
round-trip property to be *stated and tested*. The test suite asserts the round trip to
`rtol = 1e-9` and separately asserts that it is **not** bit-exact.

`RK4Integrator` is the default; `EulerIntegrator` is present mostly so the order of accuracy
is testable by comparison (halving the step cuts RK4's error by ~16, and the tests check
that).

## 3. Continuous normalising flows

`integrate_with_divergence` accumulates

$$\frac{d}{dt}\log p(z(t)) = -\operatorname{tr}\frac{\partial f_\theta}{\partial z}
\qquad\Longrightarrow\qquad
\log p(z_1) = \log p(z_0) - \int_{t_0}^{t_1}\!\operatorname{tr}\,\partial_z f_\theta\,dt$$

the instantaneous change of variables of Chen et al. / FFJORD. **This is the piece that would
turn the factor from a map on points into a map on densities** — i.e. the thing that would let
it transport a belief rather than a Dirac.

Checked against a linear vector field $f = Az$, where $\operatorname{tr}\partial_z f = \operatorname{tr}A$
is constant and the integral is $\operatorname{tr}(A)\,\Delta t$ exactly.

## 4. Implementation difficulties

### 4.1 The divergence is computed densely, which is the thing FFJORD exists to avoid

`_divergence` does $n$ forward differences of `vf` per evaluation, and
`integrate_with_divergence` calls it twice per step. So the cost is
$O(n \cdot \text{steps})$ evaluations, against $O(\text{steps})$ for the state alone.

FFJORD's entire contribution is the **Hutchinson trace estimator**,
$\operatorname{tr}J = \mathbb{E}_v[v^\top J v]$, which needs one JVP per sample instead of
$n$. That needs AD. What is here is correct, checkable, and unusable above small $n$ — the
same honest position as [[solve]] §4.1.

### 4.2 The divergence integration is trapezoidal while the state is RK4

Mismatched orders: the state is $O(h^4)$ and the log-det accumulator $O(h^2)$. For the linear
oracle it does not show (the integrand is constant), which means **the test suite does not
actually exercise the divergence integrator's accuracy**. A nonlinear vector field with a
known log-det would; there is no obvious one.

### 4.3 No stiffness, and no warning about it

Explicit RK4 on a stiff system diverges, quietly, and nothing here detects it. Real NeuralODE
work uses adaptive implicit solvers for exactly this reason. Anything with widely separated
timescales will produce nonsense from this file.

That interacts badly with the reverse direction, and the interaction is the known
reverse-mode instability of NeuralODEs: even when the forward solve is fine, re-integrating
backwards can be unstable for dissipative dynamics, because what contracts forwards expands
backwards. The test uses a mildly dissipative $A$ (eigenvalues with negative real part) and
the round trip holds; a strongly dissipative one would not.

### 4.4 `steps` is a magic number

Fifty. There is no error estimate, no adaptivity, and no way for a caller to learn that fifty
was not enough — `integrate` returns a vector and no report, unlike `solve_root`, which
returns a `SolveReport`. That asymmetry is not principled; the integrator should report its
step count and estimated error the same way.

Related: [[neuralode]], [[solve]], [[NeuralODE as an Invertible Factor]],
[[The Equilibrium Family]]
