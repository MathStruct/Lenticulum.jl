#design

> **Status: partly implemented (2026-10-25): §0, §1.1–§1.7, §2.1–§2.3, §3(a) and §5; §2.4, §3(b) messages and §4 remain planned.** Originally planned on Daniel's request (2026-10-25, "Could you plan for that?"), following Gemini's suggestion to begin with toric models and the Grassmannian. Related: [[Toric Models as Factors]], [[Grassmann Mixing and Plücker Coordinates]], [[The Parameter is a Grassmannian]], [[Algebraic Implicit Learners]], [[Open Problems in Algebraic Implicit Learning]], [[L-SPEC-001 The Factor Interface]], [[L-SPEC-003 Exposed Parameters and Energy Channels]]. Code: `lib/AlgebraicFactors.jl/src/toric.jl`, `fit.jl`, `grassmann.jl`, `plucker.jl`, `veronese.jl`, `polynomial.jl` and `polyfit.jl`, tested in `lib/AlgebraicFactors.jl/test/runtests.jl`. §2.3 also uses `lib/LenticulumCore.jl/src/energy.jl` and its core tests. These steps needed no Mycelium changes.

## Scope

The first code for the algebraic family ([[Algebraic Implicit Learners]]) implements finite
toric factors, Grassmannian parameter updates and forward Plücker transport; polynomial residuals and complete-data fitting are implemented, with polynomial messages still planned. Gemini's review (Proposal 7 of `05_Architectural_Design_Proposals.md` in
the review folder) sketches a whole package `lib/AlgebraicFactors.jl`; this specification orders
it so that the cases where everything is exact or cheap come first:

1. toric factors ([[Toric Models as Factors]]): exact discrete factors with known implicit
   equations;
2. Grassmannian parameters ([[The Parameter is a Grassmannian]]): projection and retraction for
   parameters defined up to a gauge;
3. the general polynomial factor, on top of both;
4. multi-branch inference by homotopy continuation, last.

Plücker coordinates ([[Grassmann Mixing and Plücker Coordinates]]) enter as a small
deterministic factor and as test data. The contract below is a proposal; each part becomes
binding when it is implemented with its tests.

## Today

| capability | state |
|---|---|
| a factor over several discrete variables | `AlgebraicFactors.ToricFactor`: exact finite sum-product messages and joint factor free energy; tree marginals and Bethe free energy checked against enumeration |
| toric fitting and model checks | `fit_mle`: gauge-reduced Newton or cyclic binary-feature IPF; separate moment convergence and existence statuses; positive `inmodel` and supplied `binomial_residuals`, checked against closed forms and independence minors |
| parameters on a manifold | `Grassmann`, horizontal projection, positive-diagonal QR retraction and `Retracted` for Mycelium's `GradientDescent` and `Momentum`, including velocity transport and `OptimiserFactor` integration; Nesterov remains planned |
| Plücker map and quadrics | `plucker`, closed-form Jacobian, lexicographic coordinate pairs and sparse relations; forward-only `PluckerFactor` with exact Dirac/paired-particle transport and independent-Gaussian moments; singular Gaussian messages require explicitly requested regularisation |
| gauge-invariant residual scalarisation | core `SampsonNorm` localises to a metric; checked on the real polynomial factor with ΘΘ′ damping, a singular Jacobian and undamped polynomial basis changes |
| polynomial basis tables, residual and fitting (§3a) | `ExponentTable`, exact monomial/Chebyshev values and Jacobians, affine `basis_change`, `PolynomialFactor` with full-coordinate Sampson scalarisation, `fit_nullspace` rank diagnostics, full-gradient Riemannian `fit_sampson` and k=1 FNS; no messages |
| multi-branch roots | none; `MixtureBelief` exists to carry them |

## The proposed contract

**§0 Package.** `lib/AlgebraicFactors.jl`, depending on LenticulumCore, Mycelium and
LinearAlgebra, plus LuxCore and Random for the parameter/state interface, until §4 adds a package extension. Tests and docs are wired like the other
packages under `lib/`.

**§1 Toric factors.**
- §1.1 `ToricFactor(names::Tuple, alphabets::Tuple, A::AbstractMatrix{<:Integer}; normalised = false)`:
  one channel per name, of the alphabet's size. The joint states are ordered column-major over the
  alphabets (documented), and column $x$ of `A` is $a_x$; the constructor checks
  `size(A, 2) == prod(alphabets)`. Constructors build `A` for common cases:
  `independence(names, alphabets)` and `hierarchical(names, alphabets, generators)`
  (marginal-cell indicators per generator).
- §1.2 Parameters `ps = (η = …,)` in $\mathbb R^d$. $p_\eta = p_{\eta'}$ iff $A^\top(\eta-\eta') \in \operatorname{span}\{\mathbf 1\}$ (the docstring
  says so); an optional reduction to independent logit differences is a chart, not a change of
  model. Statements that need $\mathbf 1$ in the row space of $A$ use $\bar A = \binom{\mathbf 1^\top}{A}$ internally
  ([[Toric Models as Factors]]).
- §1.3 `energy(f, x, …)` is $-\langle\eta, a_x\rangle$, plus $\Lambda(\eta)$ when `normalised`; $x$ is a tuple of labels. Scalar.
- §1.4 `factor_message(f, target, polarity, inputs, prior, ps, st) -> (belief, st)`: the exact
  sum-product likelihood message to `target`, a `CategoricalBelief`. Observed inputs may be Diracs
  on labels or categorical beliefs; latent channels (missing or uninformative inputs, which
  Mycelium resolves to `Latent()`) are summed over with unit weight, so ordinary BP initialisation
  works. The message excludes the target's `prior`. The cost, $\prod$ alphabets per message, is stated.
- §1.5 `local_free_energy(f, msgs, ps, st)`: $\mathbb E_{b_f}E - H(b_f)$ for the factor belief
  $b_f \propto \psi_\eta \prod_u \mu_{u\to f}$, exact, in the form `PriorFactor` returns it (a scalar). With
  `normalised = true`, the game-level `free_energy` and `entropy` of [[L-SPEC-001 The Factor Interface]] §4
  are exact as well.
- §1.6 `fit_mle(f, counts; method = :newton | :ipf)`, returning $\eta$, a convergence status and,
  separately, an existence status. The stopping test is moment matching, $\lVert A\hat p - Au/n\rVert$
  (Birch), which can also approach zero at an extended MLE on the boundary. Without an existence
  certificate (§Open), a diverging $\eta$ is reported as non-convergence or suspected boundary
  convergence, never as proven non-existence, and never silently. Newton removes the gauge
  directions first.
- §1.7 `inmodel(f, p; atol)`: for positive $p$, whether $\log p$ lies in the row space of $\bar A$ (least
  squares). `binomial_residuals(f, p, moves)` evaluates $p^{u^+} - p^{u^-}$ for moves the user supplies.
  No Markov-basis computation.

**§2 Grassmannian parameters.**
- §2.1 `Grassmann(k, m)`, represented by $\Theta \in \mathbb R^{k\times m}$ with $\Theta\Theta^\top = I_k$:
  `project_tangent(M, Θ, G)` $= G - G\Theta^\top\Theta$; `retract(M, Θ, Δ)` orthonormalises the rows of
  $\Theta + \Delta$ by a thin QR of the transpose, with signs chosen so that $\operatorname{diag}(R) > 0$, so that
  $R_\Theta(0) = \Theta$ (in exact arithmetic; tested with `≈`); `rand(rng, M)` and `ispoint(M, Θ; atol)`. The projection is the horizontal lift for Gr, not the full Stiefel tangent projection. Non-finite and numerically rank-deficient QR inputs throw (threshold documented in [[grassmann]]). (Edelman, Arias & Smith 1998; Absil, Mahony & Sepulchre 2008.)
- §2.2 `Retracted(rule, M) <: AbstractUpdateRule` for Mycelium's optimiser lens, at first for
  `GradientDescent` and `Momentum` only, with `rule_get(...) = Θ`. `rule_put` projects the gradient,
  applies `rule`, converts the candidate point it returns into a displacement $p' - \Theta$, projects
  that onto the tangent space and retracts. A rule's state (momentum) is projected onto the new
  tangent space, which is a vector transport for an embedded manifold (Absil, Mahony & Sepulchre
  2008). Mycelium has no rule-state initializer: callers supply `nothing` for gradient descent or a plain `zero(Θ)` velocity matrix for momentum. Its recurrence is $s'=\gamma s-\eta g$ (corrected in [[LP-015 Momentum had the wrong sign on its decay]]), which the wrapper preserves. Nesterov waits: its look-ahead $p + \gamma s$ leaves the manifold, and a retracted look-ahead
  with a consistent transport of the gradient has to be specified first. This is the lens $G^*(\Theta, \Delta) = \mathrm{qf}(\Theta + \Pi_\Theta\Delta)$ of
  [[The Parameter is a Grassmannian]].
- §2.3 Gauge invariance as a test obligation (implemented with a test-only residual,
  not the §3 factor): a row-space relation uses the same Sampson scalar energy for
  $\Theta$ and $B\Theta$ when the covariance is numerically invertible.
  Following [[L-ADR-0007 Point-dependent scalarisations are localised before use]]
  (accepted 2026-10-25), core `SampsonNorm` is localised at the factor and point
  to `SquaredNorm((JΣJᵀ + λR)⁻¹)`. Unlocalised application and `jensen_gap` throw.
  Default damping is identity; a factor with a gauge overrides `sampson_damping`
  with $R\mapsto BRB^\top$, here $R=\Theta\Theta^\top$. This keeps damped energy
  invariant even at rank-deficient $J$, provided the covariance passes the SPD check.
  Numerically singular covariance throws; there is no silent pseudoinverse.
  Graded scalarisations with point-dependent parts are rejected until per-part
  factor/argument routing is specified. [[energy]] gives the numerical threshold,
  reference Jacobian and frozen-metric Gauss–Newton convention.
- §2.4 With [[L-SPEC-003 Exposed Parameters and Energy Channels]], an exposed Grassmannian
  parameter is a channel whose space is a manifold; that spec's "keep open" list already allows it.

**§3 The polynomial factor. Part (a) implemented; part (b) messages planned.**

- `ExponentTable(N,d; basis=:monomial|:chebyshev, homogeneous=false)` stores
  N×m indices in increasing total degree, descending lexicographic order.
  `veronese`, `veronese_jacobian` use polynomial arithmetic/three-term recurrences.
  No data scaling is implicit; Chebyshev is intended for [-1,1] without enforcement.
  `basis_change(mon,cheb)` is exact for affine tables; degree-exact Chebyshev
  products generally leave the homogeneous monomial span, which throws.
- `PolynomialFactor(dims,tab,k;Σ=I,λ=0)` has seeded Grassmann rows and empty state.
  Raw energy accepts named values split between x/y, concatenated in dims order.
  Sampson energy/Jacobian uses x=nothing, packed y=z, wrt=:y, all coordinates.
  `sampson_damping=ΘΘ′`; λ≥0 preserves row-gauge invariance. This coefficient
  metric does not survive nonorthogonal polynomial basis changes; undamped
  energy does. No message polarities are declared. See [[polynomial]].
- `fit_nullspace(tab,X,k;rtol=1e-12)` uses the bottom right singular vectors,
  returns Θ, the full spectrum (structural zeros included), numerical_rank,
  nullity and threshold; there is no column scaling or normalisation.
- `fit_sampson(tab,X,k;Σ=I,Θ0=nothing,method=:riemannian,tol=1e-7,
  maxiter=10000,λ=0)` starts at the nullspace fit by default. Full analytic
  gradient, horizontal projection, QR retraction and Armijo backtracking
  minimise the non-convex summed half squared Sampson objective. The stopping
  rule is horizontal Frobenius gradient norm ≤tol; results report cost,
  gradient_norm, converged and completed iterations. Singular current/trial
  covariances throw with point index; optional λ adds gauge-compatible damping.
  `method=:fns` is the published corrected iteration for k=1 (throws otherwise),
  with the same gradient stopping criterion. Naive frozen-weight IRLS is not a
  substitute for this objective's gradient. See [[polyfit]] for derivation,
  verified citation, numerical checks and limitations. No global optimum claimed.
- Part (b): single-branch messages by Newton with a Laplace Gaussian remain
  planned ([[Inference as Root Finding]]). No inference or homotopy is added here.

**§4 Multi-branch inference** (last): a HomotopyContinuation.jl package extension finding all
isolated real roots and returning a `MixtureBelief` over branches
([[Branches and the Discriminant]]).

**§5 Plücker (implemented).** `plucker_pairs(r)` lists increasing pairs in lexicographic
order. `plucker(z,w)` evaluates $p_{ij}=z_iw_j-z_jw_i$; `plucker_jacobian(z,w)` has
columns in `vcat(z,w)` order. `plucker_relations(r)` returns $\binom r4$ records in
lexicographic quadruple order, each with `indices=(i,j,k,l)`, three `monomials`
(pairs of indices into `p`) and `coefficients=(1,-1,1)`, representing
$p_{ij}p_{kl}-p_{ik}p_{jl}+p_{il}p_{jk}$. `plucker_residuals(r,p)` evaluates them.

`PluckerFactor(r; names=(:z,:w,:p), covariance_jitter=0)` has empty parameters/state,
and exactly the forward polarity (observed `z,w`, unobserved `p`). The inverse of a
nonzero decomposable bivector is an $SL_2$-orbit of pairs, not a built belief;
$p=0$ is degenerate. Dirac pairs give a Dirac; two sample clouds are paired by index,
with equal positive counts and matching normalised weights (`rtol=1e-12, atol=0`).
The first cloud's weights survive; a Dirac may accompany either cloud.
Separate proper Gaussian inputs are assumed independent; the output `GaussianBelief`
matches the exact first two moments, not the generally non-Gaussian law.
`plucker_moments(μ,S,ν,T)` exposes those moments with zero covariance for a Dirac.
[[plucker]] derives the full bilinear covariance, including the term absent from
Jacobian-only propagation. Core Gaussian beliefs cannot represent singular covariance:
by default a covariance with $\lambda_{\min}\le n\operatorname{eps}(T)\lambda_{\max}$
is rejected; explicit positive `covariance_jitter` adds that absolute variance to
each output coordinate, an approximation. Exact singular moments remain available
without jitter through `plucker_moments`. No core or Mycelium changes were needed.

Scalar pointwise energy and graph-local free energy are zero, following `GeneratorFactor`.
Game-level quantities and an assembled lens remain absent (LP-009); the full common
conformance checker requires game quantities and therefore does not apply.
The 400-point Klein-quadric check now exercises §3(a)
`fit_nullspace` using this map and `plucker_relations` as its independent oracle.

## Checked by

Implemented §0 and §1.1–§1.5: `lib/AlgebraicFactors.jl/test/runtests.jl` checks hand-written
independence columns, state order, energies, messages with labelled/Dirac/categorical/latent
inputs, local and game quantities, and five-variable trees with three toric factors against
independent enumeration. The conformance checker
runs its explicit direct-message path over every supported polarity of the test factors at
arities one, two and three, using independent site and pooled-posterior fixtures. Existing
factors retain the lens checks. No assembled lens or joint latent posterior API is supplied. See [[toric]]
for label rules, game argument conventions and the Bethe entropy accounting.

Implemented §1.6–§1.7: the `Toric fitting and membership` testset checks positive
2×3 and 3×4 independence against the outer product of margins (cell tolerance `3e-10`),
and a positive 2×3×2 chain against $\hat p_{AB}\hat p_{BC}/\hat p_B$ (`5e-10`), by both
Newton and IPF. A seeded random integer design checks moment matching (`1e-10`) and
$Au/n-Ap_\eta$ against central likelihood differences. Redundant rows and an added
intercept give the same probabilities; an independent pseudoinverse checks the minimum-norm
gauge, and normalised energies reproduce the returned probabilities. An empty-row table
and its reversed table approach the extended MLE within `3e-10`, with finite parameters,
moment convergence and `existence == :suspected_boundary`. Zero cells with positive margins
can instead give `:unknown`: only all-positive counts are currently certified `:exists`.
Positive membership agrees with all independence minors on seeded random points in and
out of the model. Invalid inputs, exact augmented-kernel checks and degenerate uniform
models are tested too. See [[fit]] for all numerical choices and the heuristic thresholds.
The earlier planned requirement that an empty-row fit must not be "converged" was wrong:
§1.6 distinguishes moment convergence at an extended MLE from finite-MLE existence.

Implemented §2.1–§2.2: `Grassmann geometry` and `Retracted rules and eigenplane descent`
in the same test file check orthonormality, zero-step identity, positive QR diagonal,
horizontal projection and idempotence, invalid dimensions, shape/rank/non-finite failures,
and Float32 retraction. For the row-space-invariant trace objective, the projected
gradient is horizontal and its step of `1e-6` decreases the objective by
$\eta\|\operatorname{grad}F\|_F^2$ within relative tolerance `2e-5`.
Both wrapped rules use a seeded symmetric 6×6 matrix with spectrum `(-2,-1,1,2,3,5)`,
`k=2`, step size `η=0.04 < 1/(2*opnorm(S))=0.1` and **400 iterations** (momentum
`γ=0.6`, heavy-ball). From the seeded random start, the
largest principal angle to the two smallest eigenvectors is below `1e-6` radians;
the objective matches their eigenvalue sum to absolute tolerance `2e-12`.
`OptimiserFactor` / `optimiser_step` gives identical iterates to direct `rule_put`,
and momentum is horizontal after every step. Nesterov and another unsupported rule
are rejected at construction. This demonstrates those fixed runs, not a general
momentum convergence theorem. See [[grassmann]] for state layout and numerical choices.

Implemented §2.3: core's `Point-dependent scalarisation localisation` testset checks
fixed scalarisations, validation, state handling and Float32 metric types. A circle
residual agrees with $(ρ^2-1)^2/(8ρ^2)$ (`rtol=2e-10`); its ratio to geometric energy
$(ρ-1)^2/2$ approaches one from both sides at distances `1e-2` through `1e-5`, with
error below `1.1*abs(ρ-1)` and finally `1.1e-5`. It also checks covariance scaling,
singular rejection, damping and the localised Gauss–Newton pull-back.
AlgebraicFactors' `Sampson gauge obligation (§2.3)` uses a test-only two-equation,
six-monomial residual: exact Jacobians agree with the finite-difference default
(`rtol=2e-10, atol=2e-9`). Seed `2307`, 20 points, a row transformation with singular
values `0.7, 1.8`, damping `0, 0.3` and identity/non-isotropic data covariance check
invariance (`rtol=1e-10`). QR-orthonormalised representatives agree at both damping
values. Identity damping and algebraic distance each differ by more than `1e-3` at
some tested point. Constructed quadratics at the origin have rank-zero $J$:
zero damping throws, and $ΘΘ^\top$ damping is finite and gauge invariant. These are
finite numerical checks, not arbitrary-condition-number guarantees or §3 implementation.

Implemented §5: `Plücker coordinates and relations`, `PluckerFactor forward beliefs`
and `Plücker exact moments and Gaussian messages` in the same test file check hand cases
at $r=2,3,4$, antisymmetry, determinant scaling, Lagrange's identity, rectangular
exterior-power equivariance against independent minors, central Jacobian differences
(`h=1e-5`, absolute tolerance `2e-10`), relation counts/order and vanishing at $r=4,5,6$
(`3e-14` absolute), and nondecomposable violations. Polarity tests enumerate all 27
observed/latent/unobserved assignments. Dirac and paired/weighted sample messages,
custom names, target exclusion, independent proper Gaussian messages and a Mycelium
DAG with two Gaussian priors are checked. Exact moments at $r=2,3,4,5$ agree with
an independent expansion of second moments, including Dirac limits. Monte Carlo
uses 200000 independent pairs, `Xoshiro(5104)`, at $r=4$: normalised mean error
`0.00307141 < 0.015`, covariance Frobenius relative error `0.00487539 < 0.025`.
Singular rejection, rank $r-1$ for fixed nonzero input, and explicitly regularised
messages are checked; no silent jitter or claim of exact Gaussian law is made.

Implemented §3(a): `lib/AlgebraicFactors.jl/test/polynomial.jl`, included from
`runtests.jl`, checks basis counts/order/invalid inputs, homogeneous indices,
hand-written monomials, exact rational evaluation, Float32 type preservation,
Chebyshev cosine references and central Jacobian differences through degree six,
and exact affine basis changes. The real factor's exact Jacobian agrees with
core's finite-difference default (atol=2e-9,rtol=2e-10). Seed 3002, 20 points,
row gauges with singular values 0.8 and 1.6, identity/anisotropic Σ and λ=0,0.3
check row-gauge invariance (rtol=1e-10); undamped energies also agree between
monomials and Chebyshev. Singular J throws at λ=0 and is finite and row-gauge
invariant at λ=0.3. Full Sampson gradients at k=1 and 2, anisotropic Σ and
λ=0,0.2 agree with central differences along retracted curves
(h=1e-6,rtol=2e-5,atol=1e-4).

`Klein quadric fit_nullspace` uses 400 unnormalised Gaussian-pair points,
Xoshiro(5106), N=6,d=2,m=28. **Both real fitting APIs** (monomial and Chebyshev
tables) find exactly one singular value below relative rtol=1e-12 and recover
`plucker_relations(4)` coefficients, up to sign and basis change, within 2e-14
in absolute normalised dot product. Smallest/next relative values are
4.4408920985e-16 / 0.0848452042553 (monomial, gap 1.91054415134e14),
4.4408920985e-16 / 0.0540588717743 (Chebyshev, gap 1.21729757389e14).
Unit-normalised data remain unverified; see [[polyfit]].

The conic test uses Xoshiro(3003), 200 points on a random rotated/translated ellipse
with axes 1.3 and 0.8; noiseless fitting recovers its coefficients up to sign.
Coordinate noise σ=0.04 gives nullspace Sampson cost 0.16489226427736944;
Riemannian fitting gives 0.1616779698003989, gradient 1.7759602467593341e-6,
82 updates at tol=2e-6. FNS gives 0.1616779698003528, gradient
2.2989028737805305e-7, 5 updates. Costs agree within 1e-11. Twenty-five random
horizontal unit directions at steps ±1e-4 do not reduce the final cost
(allowance 1e-11), a finite local check rather than a global certificate.
The seeded intersection z=x², x²+y²=1 in ℝ³ has exactly two quadratic null
directions; the fitted projector agrees with the known equations (atol=2e-13)
and k=2 Sampson fitting starts stationary (cost below 1e-24). Underdetermined
and zero evaluation matrices, invalid arguments and indexed singular failures
are checked too. No naive-reweighting counterexample is claimed.

Still planned: §2.4 exposed manifold channels, §3(b) Newton/Laplace messages,
and §4 homotopy. No LenticulumCore or Mycelium changes or dependencies were needed.

## Open

- General MLE existence certification: the implemented sufficient certificate is all-positive
  counts. Deciding relative-interior membership of $(1,Au/n)$ in the convex hull of the
  columns of $\bar A$ (Fienberg & Rinaldo 2012) can be formulated as a linear program;
  whether to depend on an LP solver. The boundary diagnostic is only a heuristic.
- Markov bases need external software (4ti2 and similar); whether to depend on it.
- Large tables: messages are exponential in arity. Whether `hierarchical` should split into one
  factor per generator automatically is a junction-tree question
  ([[LP-010 Where the graph of a problem comes from]]).
- Beliefs on the Grassmannian (rather than points on it) are not planned.
- A causal Grassmann-mixing block ([[Grassmann Mixing and Plücker Coordinates]] §"What Lenticulum
  can take") is an experiment for LenticulumLM, not part of this package.
