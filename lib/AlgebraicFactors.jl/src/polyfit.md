#implementation #derivation #algorithm

> `fit_nullspace` returns an algebraic-distance solution and an explicit rank
> diagnostic. `fit_sampson` minimises the full non-convex Sampson objective by
> Riemannian gradient descent, or by FNS for one equation.

> Sources: original derivation of the multi-equation gradient; Chojnacki, Brooks, van den Hengel & Gawley (2000), *On the fitting of surfaces to data with covariances*, §4.1 equations (16)–(17), §4.2 and Fig. 1 (FNS); Absil, Mahony & Sepulchre (2008), *Optimization Algorithms on Matrix Manifolds*; code: `polyfit.jl`, `grassmann.jl`, `../test/polynomial.jl` (Klein quadric fit_nullspace; Polynomial fitting and full Sampson gradient).
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
Both methods stop **only** when the summed objective's horizontal Frobenius
gradient norm is ≤tol (default 1e-7); maxiter defaults to 10000. The result gives
Θ, cost, gradient_norm, converged and iterations (completed parameter updates).
There is no division by n in either the cost or stopping rule.

Each point covariance is checked using the same relative SPD threshold as core's
SampsonNorm and solved by Cholesky. A singular current or trial point throws an
ArgumentError with its one-based point index; no silent pseudoinverse, rejected
singular trial or automatic damping is used. Positive λ explicitly adds λΘΘᵀ.
The cache uses O(n m²) memory and dense CPU arithmetic; this is for small tables.

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
a randomly rotated and translated ellipse with axes 1.3 and 0.8. The noiseless fit
recovers its coefficients up to sign; independent Gaussian coordinate noise has
σ=0.04. With identity Σ (no division by σ²), nullspace Sampson cost is
0.16489226427736944. Riemannian descent at tol=2e-6 gives 0.1616779698003989,
gradient norm 1.7759602467593341e-6 in 82 updates. FNS gives
0.1616779698003528, gradient 2.2989028737805305e-7 in 5 updates. Costs agree within
1e-11. Twenty-five random unit horizontal directions at steps ±1e-4 do not
lower the final cost (allowance 1e-11); this is a finite local-minimum check,
not a proof or a global claim. Central-difference gradient checks use h=1e-6,
rtol=2e-5,atol=1e-4, including k=2 and non-isotropic covariance.

A second seeded curve in ℝ³, z=x² and x²+y²=1, has exactly two degree-two null
directions. The fitted projector matches the two known quadrics to atol=2e-13;
Sampson fitting from that solution is already stationary (cost below 1e-24).
A twisted cubic would have **three**, not two, independent quadratic generators;
the complete two-quadric intersection avoids that ambiguity. No higher-degree
ideal quotient, missing-data EM or homotopy is implemented. Single-branch
frozen-metric Gaussian messages are implemented separately in [[polynomial]];
they do not use the full varying-metric gradient derived here.

**Conditioning (a limitation, found in review).** `:riemannian` is first-order
gradient descent, so it is slow when the Veronese columns have very different scales.
On unscaled data with coordinates of a few units it can stall with a gradient norm
around `1e-6` and return `converged = false` at the default `tol = 1e-7` after
`maxiter` updates; this is why the conic test above uses `tol = 2e-6`. FNS (for
`k = 1`) and the same descent on data scaled into `[-1, 1]` reached the default
tolerance in a scratch check during review. Scale the data (and Σ with it) before
fitting, or prefer `:fns` when `k = 1`. A second-order or trust-region method on
the Grassmannian is not implemented.
