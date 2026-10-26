#model #derivation

> **Derivation.** The spectral solution of [[Fitting is a Nullspace Problem]] uses the
> algebraic distance. A first-order geometric correction produces a point-dependent
> [[Scalar and Multivariate Energy|scalarisation]] — and shows what the energy space $E$
> *is*: the space the noise lives in.

> Sources: original analysis; Chojnacki, Brooks, van den Hengel & Gawley (2000), *On the fitting of surfaces to data with covariances*, §4.1–§4.2, Fig. 1; Absil, Mahony & Sepulchre (2008), *Optimization Algorithms on Matrix Manifolds*, Chapter 7, Algorithms 10–11; code: `lib/LenticulumCore.jl/src/energy.jl` (`SampsonNorm`, `localise`, `sampson_damping`), `lib/LenticulumCore.jl/test/runtests.jl` (circle closed form), `lib/AlgebraicFactors.jl/src/polynomial.jl`, `polyfit.jl`, `lib/AlgebraicFactors.jl/test/polynomial.jl` (real factor gauge/basis tests and fitting).
>
> Bibliography: [[Bibliography#^chojnacki2000fitting|Chojnacki et al. 2000]] · [[Bibliography#^absil2008optimization|Absil et al. 2008]]
>
> Theory (CT-ML wiki): [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## The two distances

**Algebraic distance.** $\ \sigma_{\mathrm{alg}}(x) = \tfrac12\|r_\Theta(x)\|^2$. Cheap,
quadratic in $\Theta$, hence the closed-form fit. But it is not a distance: it depends on how
the variety is written. Scaling $\Theta$ scales its squared residual. Locally,
$|r|\approx \|\nabla r\|\,\mathrm{dist}(x,V)$ for a hypersurface near a regular point,
so a large residual gradient inflates algebraic distance even at the same geometric distance.

**Geometric distance.** $\ \mathrm{dist}(x, V)^2 = \min_{y \in V}\|x-y\|^2$. The thing you
actually mean. Independent of the equations. Also **very** expensive: the number of complex
critical points of $y \mapsto \|x-y\|^2$ on $V$ is the **Euclidean Distance Degree**
(Draisma–Horobeț–Ottaviani–Sturmfels–Thomas). For a generic hypersurface of degree $d$ in
$\mathbb{C}^n$,

$$\mathrm{EDD} \;=\; d\sum_{i=0}^{n-1}(d-1)^i$$

*Sanity check:* a plane conic, $n=2$, $d=2$: $\ 2(1 + 1) = 4$ — the classical fact that the
foot of the perpendicular to a generic conic solves a quartic. For fixed degree $d>2$,
this formula grows exponentially in $n$. It counts complex critical points, not real
minima or a runtime bound. Exact geometric fitting can therefore require an expensive
projection subproblem; this is not a universal hardness claim (affine models, for example,
have closed-form projection).

## The first-order correction: Sampson distance

Linearise. Near $x$, the variety is approximately the affine subspace
$\{y : r(x) + J(x)(y-x) = 0\}$ with $J = J_r(x)\in\mathbb{R}^{k\times N}$. The minimum-norm
displacement to that subspace, assuming $J$ has full row rank, solves
$\min\|\delta\|^2$ subject to $J\delta = -r$, whose solution is
$\delta = -J^\top(JJ^\top)^{-1}r$, giving

$$\boxed{\;\mathrm{dist}(x,V)^2 \;\approx\; r(x)^\top \bigl(J(x)J(x)^\top\bigr)^{-1} r(x)\;}$$

the **Sampson distance** (Sampson 1982; Taubin 1991). For a single equation ($k=1$) it
reduces to the familiar $r^2/\|\nabla r\|^2$.

## The derivation that matters: $E$ is where the noise lives

Now read the same formula statistically rather than geometrically. Posit the honest
errors-in-variables model: the true point $x^\star$ lies on $V$, and we observe
$x = x^\star + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \Sigma)$ in the **data space**.

Push the noise forward through the residual. To first order,

$$
r(x) = r(x^\star + \epsilon) \approx \underbrace{r(x^\star)}_{=\,0} + J\epsilon = J\epsilon
\qquad\Longrightarrow\qquad
r \;\sim\; \mathcal{N}\bigl(0,\; J\Sigma J^\top\bigr)
$$

For a Gaussian approximation in residual coordinates with invertible $A=J\Sigma J^\top$,
the negative log-density includes a normalisation term:

$$
-\log p_r \;\approx\; \tfrac12 r^\top A^{-1}r
+\tfrac12\log\det A+\tfrac{k}{2}\log(2\pi).
$$

The quadratic term at $\Sigma=I$ is **half** the squared Sampson distance defined
above. `SampsonNorm` implements this energy, not a normalised likelihood. The earlier
formula omitted $\tfrac12\log\det A$ as a “constant”, although $A$ varies with point
and parameters. A density in residual coordinates also changes with the residual gauge;
it is not by itself a likelihood for the observed data. Two consequences:

> [!note] What the energy space is
> The residual space $E = \mathbb{R}^k$ of [[Scalar and Multivariate Energy]] is **the space
> the observation noise lives in after pushforward**, and the scalarisation
> $\sigma_M(e) = \tfrac12 e^\top M e$ is **the noise precision**. Choosing $\sigma$ is
> choosing a noise model, not choosing a convenience.

> [!note] Why the energy must stay multivariate
> $M = (J\Sigma J^\top)^{-1}$ cannot be computed from the scalar energy $\|r\|^2$. It needs
> $r$ **and** its Jacobian — i.e. the vector energy. A factor that scalarises early has
> irrecoverably thrown away its own noise model. This is the concrete instance of
> [[Scalar and Multivariate Energy]] §6's claim that "scalarising early destroys the
> metric".

## The cost: three properties lost

$M = (J\Sigma J^\top)^{-1}$ generally depends on both $x$ and $\Theta$. That breaks three
things that made the algebraic distance attractive:

1. **The objective is no longer quadratic in $\Theta$.**
   $f(\Theta)=\sum_i r_i^\top M(\Theta,x_i)r_i$ does not have the fixed-metric
   nullspace solution of [[Fitting is a Nullspace Problem]]. Freezing the metrics and
   resolving a quadratic subproblem omits derivatives of those metrics. Such
   naive reweighting does not in general minimise the full Sampson objective.
   [[polyfit]] implements its full gradient with an Armijo Riemannian solver,
   the corrected FNS iteration for k=1, and Riemannian trust regions with
   analytic Hessian and truncated CG. `AbstractSampsonMethod`/`sampson_solve`
   make the solver exchangeable; the default is the trust-region method.
2. **There is no single fixed $\sigma$.** A frozen `SquaredNorm(M)` is convex and
   has Jensen gap $\tfrac12\operatorname{tr}(M\operatorname{Cov}(r))$. A field of
   metrics at different points is a different object: the fixed-map Jensen identity
   does not apply. `jensen_gap(SampsonNorm(), ...)` throws, including for empty samples.
   The earlier claim that the varying metric's gap was
   $\tfrac12\operatorname{tr}\operatorname{Cov}$ was wrong.
3. **The inverse fails where the pushed-forward covariance loses rank.**
   In particular this happens when $J$ loses row rank. Vanishing variance in a
   residual direction does not mean unconstrained noise there; the nonsingular
   Gaussian approximation has failed. `SquaredNorm` does validate PSD metrics;
   the earlier claim that it did not was stale. `SampsonNorm` rejects singular or
   numerically deficient covariance rather than silently taking a pseudoinverse.

## Localisation and gauge-compatible damping (implemented)

[[L-ADR-0007 Point-dependent scalarisations are localised before use]] was accepted on 2026-10-25.
Its implementation in [[energy]] adds `AbstractPointScalarisation`, `localise` and
`SampsonNorm`. The pointwise path localises before scalarising; free-energy scalarisation
is unchanged. Graded scalarisations containing point-dependent parts are rejected until
their parts can be given their own factor and arguments.

For $r_\Theta=\Theta v$ and $D=D_zv$, a factor can override `sampson_damping` with
$R=\Theta\Theta^\top$. For any data covariance $\Sigma$ of the right dimension,

$$
J\Sigma J^\top+\lambda\Theta\Theta^\top
=\Theta(D\Sigma D^\top+\lambda I_m)\Theta^\top.
$$

Under $\Theta\mapsto B\Theta$, residual and covariance transform by $r\mapsto Br$
and $A\mapsto BAB^\top$, so $\tfrac12 r^\top A^{-1}r$ is invariant when defined.
For full-row-rank $\Theta$, PSD $\Sigma$ and $\lambda>0$, this covariance is SPD
at rank-deficient $J$ too. Default identity damping lacks this law. The implementation
requires $\lambda_{\min}>k\operatorname{eps}(T)\lambda_{\max}$: badly conditioned gauges
can fail the numerical check even when exact-arithmetic energies agree.

Core tests check the circle closed form $(\rho^2-1)^2/(8\rho^2)$, its ratio tending
to geometric energy $(\rho-1)^2/2$, covariance scaling and singular rejection/damping.
AlgebraicFactors tests use a test-only degree-two residual with two equations in two
variables, seed `2307`, 20 random points and a well-conditioned row transformation.
They check gauge and QR-representative agreement (`rtol=1e-10`), including damping
$\lambda\Theta\Theta^\top$ at a constructed singular point, and demonstrate failures
of identity damping and algebraic distance (difference above `1e-3` at some point).
Those original §2.3 tests remain. The real `PolynomialFactor` tests now check
its exact Jacobian, row gauges at λ=0 and 0.3, anisotropic covariance and a
singular point. Undamped monomial/Chebyshev energies agree. The coefficient
damping ΘΘᵀ is row-gauge compatible but changes under nonorthogonal polynomial
basis conversions; [[polynomial]] states that limitation.

## The honest summary

| distance | fit | statistical interpretation |
|---|---|---|
| algebraic $\|r\|^2$ | global spectral solution under orthonormal-row constraint | depends on residual/feature scaling |
| Sampson $r^\top(JJ^\top)^{-1}r$ | non-convex; full-gradient descent, RTR, or k=1 FNS | first-order geometric correction |
| geometric $\mathrm{dist}(x,V)^2$ | non-convex projection problem | profile errors-in-variables MLE under isotropic Gaussian noise |

`fit_sampson` initialises at `fit_nullspace` by default. The seeded noisy-conic
test lowers the full Sampson cost, checks the gradient and random local
directions, and compares all three methods. RTR reaches tol=1e-7 in four iterations
on the unscaled ellipse, its X/2 variant, and the noisy two-quadric k=2 curve;
[[polyfit]] records costs, gradient norms and budgets. These checks do not
certify a global minimum.
Algebraic and Sampson energies use the same residual, with a fixed metric versus
a field of metrics. Exact geometric projection additionally needs the variety and a
projection solver; it is not in general a function of the residual vector alone.
This refines the reweighting discussion in [[Scalar and Multivariate Energy]] §6.4.

Related: [[Fitting is a Nullspace Problem]], [[Scalar and Multivariate Energy]], [[Branches and the Discriminant]], [[The Algebraic Factor as a Statistical Game]]
