# Algebraic versus Geometric Distance

> **Derivation.** The convexity of [[Fitting is a Nullspace Problem]] is bought by using the
> wrong distance. Correcting it produces, exactly, a
> [[Scalar and Multivariate Energy|scalarisation]] — and shows what the energy space $E$
> *is*: the space the noise lives in.

## The two distances

**Algebraic distance.** $\ \sigma_{\mathrm{alg}}(x) = \tfrac12\|r_\Theta(x)\|^2$. Cheap,
quadratic in $\Theta$, hence the closed-form fit. But it is not a distance: it depends on how
the variety is written. Scaling $\Theta$ scales it; a point far from the variety in a region
where $\|\nabla r\|$ is large scores better than a near point where $\|\nabla r\|$ is small.

**Geometric distance.** $\ \mathrm{dist}(x, V)^2 = \min_{y \in V}\|x-y\|^2$. The thing you
actually mean. Independent of the equations. Also **very** expensive: the number of complex
critical points of $y \mapsto \|x-y\|^2$ on $V$ is the **Euclidean Distance Degree**
(Draisma–Horobeț–Ottaviani–Sturmfels–Thomas). For a generic hypersurface of degree $d$ in
$\mathbb{C}^n$,

$$\mathrm{EDD} \;=\; d\sum_{i=0}^{n-1}(d-1)^i$$

*Sanity check:* a plane conic, $n=2$, $d=2$: $\ 2(1 + 1) = 4$ — the classical fact that the
foot of the perpendicular to a conic solves a quartic. For a general hypersurface the EDD is
exponential in $n$, so **evaluating the exact geometric distance is itself an algebraic
inference problem of the same difficulty as the one we are trying to solve.**

This is the fundamental tension of the family, and it is not resolvable: exact fitting
requires exact projection, which is as hard as inference.

## The first-order correction: Sampson distance

Linearise. Near $x$, the variety is approximately the affine subspace
$\{y : r(x) + J(x)(y-x) = 0\}$ with $J = J_r(x)\in\mathbb{R}^{k\times N}$. The minimum-norm
displacement to that subspace solves
$\min\|\delta\|^2$ subject to $J\delta = -r$, whose solution is the pseudoinverse
$\delta = -J^\top(JJ^\top)^{-1}r$, giving

$$\boxed{\;\mathrm{dist}(x,V)^2 \;\approx\; r(x)^\top \bigl(J(x)J(x)^\top\bigr)^{-1} r(x)\;}$$

the **Sampson distance** (Sampson 1982; Taubin 1991). For a single equation ($k=1$) it
reduces to the familiar $r^2/\|\nabla r\|^2$.

## The derivation that matters: $E$ is where the noise lives

Now read the same formula statistically rather than geometrically. Posit the honest
errors-in-variables model: the true point $x^\star$ lies on $V$, and we observe
$x = x^\star + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \Sigma)$ in the **data space**.

Push the noise forward through the residual. To first order,

$$r(x) = r(x^\star + \epsilon) \approx \underbrace{r(x^\star)}_{=\,0} + J\epsilon = J\epsilon
\qquad\Longrightarrow\qquad
r \;\sim\; \mathcal{N}\bigl(0,\; J\Sigma J^\top\bigr)$$

So the negative log-likelihood of the observation, expressed in residual coordinates, is

$$-\log p \;=\; \tfrac12\, r^\top \bigl(J\Sigma J^\top\bigr)^{-1} r \;+\; \text{const}$$

which for $\Sigma = I$ is **exactly the Sampson distance**. Two conclusions, and they are the
payoff of this note:

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

$M = (J\Sigma J^\top)^{-1}$ depends on $x$, and therefore on $\Theta$. That breaks three
things that made the algebraic distance attractive:

1. **Convexity in $\Theta$ is gone.** $f(\Theta) = \sum_i r_i^\top M(\Theta,x_i) r_i$ is no
   longer quadratic, so [[Fitting is a Nullspace Problem]]'s global optimum does not apply.
   The standard workaround is **iteratively reweighted least squares**: freeze $M$, solve the
   eigenproblem, recompute $M$, repeat. Each step is globally optimal; the alternation is
   not. (This is Taubin's algorithm, and in the conic-fitting literature it is known to work
   well and to have no convergence proof.)
2. **$\sigma$ is not linear**, so by [[Scalar and Multivariate Energy]] §5 the composition of
   games is **lax**: the scalar chain rule and the multivariate one differ by the Jensen gap
   $\tfrac12\operatorname{tr}\operatorname{Cov}$. The algebraic family is therefore a
   *lax* member of the framework, by construction, and the laxness is quantified.
3. **$M$ blows up on the discriminant**, where $J$ loses rank
   ([[Branches and the Discriminant]]). Correctly — the noise really is unconstrained along
   the collapsing direction — but uselessly. The `SquaredNorm(M)` caveat recorded in
   `energy.md` §4 (that $M$ must be PSD and is not validated) is exactly this failure, met
   in the wild.

## The honest summary

| distance | convex in $\Theta$? | statistically correct? | cost |
|---|---|---|---|
| algebraic $\|r\|^2$ | **yes**, closed form | no — biased by $\|\nabla r\|$ | $O(m^3)$ once |
| Sampson $r^\top(JJ^\top)^{-1}r$ | no (IRLS) | first order | $O(m^3)$ per IRLS step |
| geometric $\mathrm{dist}(x,V)^2$ | no | yes (MLE, isotropic noise) | EDD-many roots **per data point** |

The practical recommendation is Sampson via IRLS, initialised at the algebraic solution.
The theoretically interesting statement is that the three rows are three
*scalarisations of the same multivariate energy*, and moving between them is a change of
$\sigma$, not a change of model. That is precisely the reweighting flexibility
[[Scalar and Multivariate Energy]] §6.4 claims the vector energy buys.

Related: [[Fitting is a Nullspace Problem]], [[Scalar and Multivariate Energy]], [[Branches and the Discriminant]], [[The Algebraic Factor as a Statistical Game]]
