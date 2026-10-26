#implementation #derivation #algorithm

> `fit_nullspace` returns an algebraic-distance solution and an explicit rank
> diagnostic. `fit_sampson` minimises the full non-convex Sampson objective by
> exchangeable Riemannian gradient/trust-region solvers, or FNS for one equation.

> Sources: original derivation of the multi-equation gradient; Chojnacki, Brooks, van den Hengel & Gawley (2000), *On the fitting of surfaces to data with covariances*, §4.1 equations (16)–(17), §4.2 and Fig. 1 (FNS); Absil, Mahony & Sepulchre (2008), *Optimization Algorithms on Matrix Manifolds*, Chapter 7, Algorithms 10–11; code: `polyfit.jl`, `grassmann.jl`, `../test/polynomial.jl` (Klein quadric fit_nullspace; Polynomial fitting and full Sampson gradient).
>
> Bibliography: [[Bibliography#^chojnacki2000fitting|Chojnacki et al. 2000]] · [[Bibliography#^absil2008optimization|Absil et al. 2008]]
>
> Theory (CT-ML wiki): [Gradient Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses)

## Algebraic fit and diagnostics

X is N×n with points in columns. V has rows v(xᵢ)ᵀ. No centring, column
equilibration, data normalisation or covariance rescaling is performed. Chebyshev
users should scale their own data and transform the data covariance accordingly.
A basis change preserves polynomial zero sets but changes the noisy algebraic
fit's coefficient normalisation.

The k smallest right singular vectors of V become rows of Θ. This is a global
solution of the **algebraic** objective with orthonormal rows, not the Sampson
objective. Direct SVD avoids squaring the condition number via VᵀV. It retains V
and the decomposition, rather than implementing a streaming scatter accumulator.
When n<m a full right basis is needed: the m−n structural zero singular values
are appended to the returned descending spectrum. `nullity` counts values ≤
`rtol*σmax`, so an all-zero V has nullity m even at zero tolerance.
`numerical_rank=m-nullity` and the absolute threshold are returned too. This
reports the rank decision; it never changes the requested k. Default rtol=1e-12.

## Gradient of the full Sampson objective

Cache vᵢ and Bᵢ=DᵢΣDᵢᵀ+λI. For one point let

$$
r=\Theta v,\quad A=\Theta B\Theta^\top,\quad
u=A^{-1}r,\quad s=\tfrac12 r^\top A^{-1}r.
$$

Using d(A⁻¹)=−A⁻¹(dA)A⁻¹ and symmetric B,

$$
\mathrm ds=u^\top(\mathrm d\Theta)v
-\tfrac12 u^\top[(\mathrm d\Theta)B\Theta^\top+
\Theta B(\mathrm d\Theta)^\top]u,
\qquad
\nabla_\Theta s=uv^\top-uu^\top\Theta B.
$$

The two covariance terms are equal scalars. The second term is essential:
freezing the weights drops it and does not solve the full stationarity equation.
Sum these gradients, then use `project_tangent` with the Frobenius metric on
horizontal Grassmann lifts. The formula also satisfies GΘᵀ=0 in exact arithmetic,
as expected from full row-gauge invariance. Tests independently differentiate the
objective on retracted curves at k=1 and k=2, anisotropic Σ, λ=0 and 0.2.

`:riemannian` retracts −η grad with positive-diagonal thin QR. Armijo accepts
S_new≤S−10⁻⁴η‖grad‖²; trials halve η, starting at 1 then twice the preceding
accepted step capped at 1, with at most 60 trials. Failure returns unconverged.
All built-in methods stop **only** when the summed objective's horizontal Frobenius
gradient norm is ≤tol (default 1e-7); maxiter defaults to 10000. The result gives
Θ, cost, gradient_norm, converged and iterations (completed parameter updates for gradient/FNS; attempted outer steps,
including rejections, for RTR). RTR also reports `hessian_products`.
There is no division by n in either the cost or stopping rule.

Each point covariance is checked using the same relative SPD threshold as core's
SampsonNorm and solved by Cholesky. A singular current or trial point throws an
ArgumentError with its one-based point index; no silent pseudoinverse, rejected
singular trial or automatic damping is used. Positive λ explicitly adds λΘΘᵀ.
The cache uses O(n m²) memory and dense CPU arithmetic; this is for small tables.

## Exact Hessian and Riemannian trust regions

For a horizontal row displacement ξ, differentiate the cached per-point terms:

$$
\dot A=\xi B\Theta^\top+\Theta B\xi^\top,\qquad
\dot u=A^{-1}(\xi v-\dot A u),
$$

$$
\dot G=\dot u v^\top-(\dot u u^\top+u\dot u^\top)\Theta B
-uu^\top\xi B.
$$

Sum over points. With row representatives, the Grassmann connection gives

$$
\operatorname{Hess} S(\Theta)[\xi]
=\Pi_\Theta\bigl(\mathrm DG[\xi]
-\operatorname{sym}(G\Theta^\top)\xi\bigr).
$$

The second term is the curvature (Weingarten) correction. Full invertible
row-gauge invariance gives GΘᵀ=0 in exact arithmetic here; the implementation
retains the correction. No numerical differentiation or Hessian matrix is
used. `Analytic Sampson Riemannian Hessian` checks horizontal output,
self-adjointness, projected central differences of the gradient (h=1e-6),
and second differences of an independent cost oracle (h=2e-4), for k=1,2,
anisotropic Σ and λ=0,0.2. Halving a one-sided retracted step from 0.002 to
0.001 reduces the second-order Taylor remainder by at least a factor of five
(up to 1e-12). QR can introduce a second-order row rotation for k>1; row-gauge
invariance removes its effect on the cost.

`RiemannianTrustRegion(;Δ0=1.0,Δmax=100.0,ρ_accept=0.1,tcg_maxiter=100,κ=0.1,θ=1.0)`
uses the quadratic model ⟨grad,η⟩+⟨η,Hη⟩/2 on the horizontal Frobenius ball.
Steihaug–Toint CG truncates at negative curvature or a boundary crossing;
otherwise its residual threshold is ‖grad‖ min(κ,‖grad‖^θ). The actual/model
reduction ratio quarters the radius below 1/4, doubles it above 3/4 when CG
hits the boundary (capped at Δmax), and accepts steps above ρ_accept. This is
Absil–Mahony–Sepulchre Chapter 7, Algorithms 10–11, checked against the
[publisher's chapter](https://assets.press.princeton.edu/chapters/absil/Absil_Chap7.pdf).
A nonpositive/nonfinite model reduction returns unconverged. There is no
claim of convergence through covariance singularities or global optimality.

## Exchanging the optimiser

`AbstractSampsonMethod` is public. `RiemannianGradient(;c=1e-4,shrink=0.5,max_backtracks=60,initial_step=1.0)`
and `FNS()` reproduce the previous calculations exactly. Symbols `:riemannian`,
`:fns`, `:trust_region` construct default instances. Unknown symbols throw.

Implement one method, `AlgebraicFactors.sampson_solve(method::MyMethod,
problem::SampsonProblem, Θ0; tol, maxiter)`. The problem exposes `manifold`,
`cost_gradient(Θ)` returning cost and Euclidean gradient, and
`hessian_vector(Θ,ξ)` returning the horizontal Hessian action. Geometry is
`project_tangent(problem.manifold,Θ,G)` and `retract(problem.manifold,Θ,ξ)`.
The initial representative and problem must not be mutated. Return Θ, cost,
gradient_norm, converged and iterations as named fields; stop only when the
horizontal gradient norm is ≤tol or the budget is exhausted (a failed search
may stop earlier, unconverged). At maxiter=0 evaluate the initial point.
`FixedSampsonDescent` in the test file demonstrates this hook on a noisy
affine line, reaching the same minimiser as RTR without private cache access.

## FNS, not naive frozen-weight iteration

For k=1 write θ=Θᵀ, aᵢ=θᵀBᵢθ and rᵢ=θᵀvᵢ. FNS forms

$$
H(\theta)=\sum_i\left[\frac{v_i v_i^\top}{a_i}
-\frac{r_i^2 B_i}{a_i^2}\right],\qquad H(\theta)\theta=\nabla_\theta S.
$$

The update is a unit eigenvector of H whose eigenvalue is **closest to zero in
absolute value**, with sign aligned to the previous vector (paper §4.2, Fig. 1).
This is not simply the least eigenvector of the first scatter term. We keep the
same gradient stopping criterion as Riemannian descent rather than the paper's
iterate-distance criterion. FNS need not be monotone; neither method certifies
convergence in general or global optimality. k≠1 is rejected for FNS. Replacing
B by B+λI extends the same formula to the explicitly damped objective.

Metadata were checked against Crossref DOI 10.1109/34.888714: IEEE TPAMI 22(11),
1294–1303 (2000). Crossref reverses the author list; the entry uses the actual
article title-page order, independently checked in the Adelaide repository PDF.
The naive-reweighting fixed-point counterexample is not run here.

## Reproducible results

`Klein quadric fit_nullspace` uses Xoshiro(5106), 400 unnormalised p=z∧w points
with independent Gaussian z,w∈ℝ⁴, affine degree two (28 terms). Both bases have
exactly one singular value below relative threshold 1e-12. Coefficients agree,
up to sign and change of basis, with `plucker_relations(4)` to absolute dot-product
tolerance 2e-14. Relative smallest/next singular values and gap ratios:

| basis | smallest | next | next/smallest |
|---|---:|---:|---:|
| monomial | 4.4408920985e-16 | 0.0848452042553 | 1.91054415134e14 |
| Chebyshev | 4.4408920985e-16 | 0.0540588717743 | 1.21729757389e14 |

These very small values are LAPACK SVD output, not measurements of exact residual
roundoff. Unit-normalised data (expected additional norm equation) remain unverified.

`Polynomial fitting and full Sampson gradient` uses Xoshiro(3003), 200 points on
a rotated and translated ellipse with axes 1.3 and 0.8, coordinate noise σ=0.04,
and Σ=I. Nullspace Sampson cost is 0.16489226427736944. The default gradient
and FNS methods are compared field-for-field (`==`, including Θ) with the
frozen original implementation in `test/legacy_sampson.jl`, at the original
tol=2e-6. Noiseless conic and two-quadric cases also check legacy equality.
The refined conic now uses RTR at tol=1e-7, and retains 25 random unit
horizontal local probes at steps ±1e-4 (allowance 1e-11). This is a finite
local check, not a certificate.

`Exchangeable Sampson solvers and RTR` reports the following runs on Julia
1.13.1, all at tol=1e-7 and maxiter=1000 (RTR is additionally tested with a
20-step budget). Scaled data mean X/2 with Σ=I, so the objective changes by a
factor of four; this is a conditioning comparison, not covariance rescaling.
The noisy two-quadric intersection z=x², x²+y²=1 uses Xoshiro(3010), 200
points and coordinate noise σ=0.02.

| data | method | iterations | cost | gradient norm |
|---|---|---:|---:|---:|
| ellipse | gradient | 103 | 0.16167796980035265 | 9.8000083743e-8 |
| ellipse | FNS | 6 | 0.16167796980035248 | 1.1211849657e-8 |
| ellipse | RTR | 4 | 0.16167796980035246 | 5.0226342224e-12 |
| ellipse / 2 | gradient | 550 | 0.0404194924500884 | 8.5344815395e-8 |
| ellipse / 2 | FNS | 5 | 0.04041949245008824 | 9.0261106662e-8 |
| ellipse / 2 | RTR | 4 | 0.040419492450088135 | 2.8262159520e-12 |
| ellipse × 20 | gradient | 2000 (unconverged) | 64.78412023641039 | 720.236886524822 |
| ellipse × 20 | FNS | 9 | 64.67118792014097 | 2.480934486486978e-8 |
| ellipse × 20 | RTR | 7 | 64.67118792014095 | 3.9735154735114037e-10 |
| noisy two quadrics, k=2 | gradient | 1000 (unconverged) | 0.08014734011235275 | 0.0067665100371 |
| noisy two quadrics, k=2 | RTR | 4 | 0.08014291692590922 | 7.4079661709e-8 |

RTR/FNS ellipse costs agree within 1e-12; RTR is no higher than gradient
within roundoff, including k=2 where FNS is unavailable. RTR uses 19 Hessian
products for each ellipse and 57 for the curve in these runs. The original
noiseless two-quadric fit still checks the exact 2D quadratic nullspace and a
stationary cost below 1e-24. A twisted cubic has three independent quadratic
relations; that was never the two-equation fixture.

The previous stall paragraph relied on a scratch run and overstated the
seeded ellipse failure: this reproducible run reaches tol=1e-7 with gradient
descent, and halving coordinates actually increases its iteration count.
Conditioning remains a real limitation of gradient descent: the ellipse × 20 run
(coordinates of size ~30, so the quadratic Veronese columns are ~10³ times the constant one;
maxiter=2000) ends with gradient norm 720, while FNS and RTR converge in 9 and 7 steps to
the same cost (within 1e-12), and so does the noisy k=2 run.
The default is `RiemannianTrustRegion()` (changed from `RiemannianGradient()` on
2026-10-26, on these measurements): RTR converges on every problem above in at most
7 steps, while gradient descent stalls on two. These are small local problems; this
does not establish universal superiority and wall time is not compared. Singular current or
trial covariances still throw in every method.

No higher-degree ideal quotient or missing-data EM is implemented. Optional
square-system homotopy inference is described in [[homotopy]]. Single-branch
frozen-metric Gaussian messages ([[polynomial]]) do not use this objective's
full varying-metric gradient.
