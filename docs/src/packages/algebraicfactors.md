# [AlgebraicFactors](@id algebraicfactors)

```@meta
CurrentModule = AlgebraicFactors
```

Finite toric potentials over named discrete channels, implementing L-SPEC-004 §0 and
§1.1–§1.7, plus Grassmannian parameters (§2.1–§2.2) and the Plücker map (§5).
Toric messages, energies and entropies use exact enumeration; fitting uses
Newton or binary-feature IPF. Polynomial tables, residuals and complete-data fitting (§3a) are implemented.
Polynomial messages and homotopy remain planned. Sampson scalarisation comes
from LenticulumCore and is checked on the real polynomial factor too.

```julia
using AlgebraicFactors, LenticulumCore, Mycelium, Random, LinearAlgebra

names, alphabets = (:a, :b), (2, 3)
A = independence(names, alphabets)   # a design matrix, not a factor
f = ToricFactor(names, alphabets, A)
ps, st = setup(Random.Xoshiro(1), f)
p = Polarity(a = Unobserved(), b = Observed())
site, st = factor_message(f, :a, p, (b = DiracBelief(2),), nothing, ps, st)
```

Joint states are ordered with the first variable fastest, as in `CartesianIndices`.
Default labels are `1:alphabet`; custom labels are a tuple of unique-label vectors.
Diracs match actual labels. Unlabelled categoricals are positional; labelled categoricals
are matched and reordered by label. Shared graph variables need consistent canonical label
order because Mycelium combines outgoing categorical messages positionally.

`hierarchical` builds a marginal-cell indicator block for each generator, named by tuples
of channels or indices. Both matrix builders retain redundant statistics. Parameters are
`(η = …,)`, initially zero; the factor retains redundant statistics. Fitting returns
the minimum Euclidean norm representative modulo constant logits and null directions.

`factor_message` excludes the target prior and target input. Latent channels enter with
unit weight, and the factor supports every observed/latent combination around one target.
This is a direct-message factor: it provides no assembled lens, joint latent posterior
or forward model. The common conformance checker uses independent site and pooled-posterior
fixtures through its direct-message path.

`local_free_energy` computes the joint factor expectation minus joint entropy from the
incoming cavity messages. Mycelium supplies the variable entropy counting correction.
Set `normalised = true` for standalone game quantities: `free_energy(f, π, y, ps, st)`
and `LenticulumCore.entropy(f, π, y, ps, st)`. Use disjoint NamedTuples of prior beliefs
and observations, or one prior belief and observations naming every other channel.

`fit_mle(f, counts; method = :newton, tol = 1e-10, maxiter = 1000)` returns
`(η, p, converged, iterations, residual, existence)`. Counts can be a joint-state vector
or an array of size `alphabets`. `p` is a vector; the parameter can be used with
`ps = (η = fit.η,)`.
Newton removes the gauge by an SVD of centred logits; `:ipf` requires a 0/1 design and
cycles over individual rows. At target moments zero or one, IPF uses finite log-odds
steps of -32 or +32 to approximate the infinite endpoint update.

`converged` means the infinity norm of the moment residual is at most `tol`.
`existence == :exists` certifies an ordinary MLE only when every count is positive.
Otherwise it is `:suspected_boundary` when the centred logit norm has grown from zero
to at least 10 and `length(p)*minimum(p) <= 1e-8`, or `:unknown`. The boundary status
is a heuristic, not proof of non-existence; it can accompany moment convergence.

`inmodel(f, p; atol = 1e-8)` requires strictly positive probabilities and tests the
Euclidean least-squares residual of log probabilities in the augmented row space
`[1'; A]`. `binomial_residuals(f, p, moves)` permits zero probabilities and checks
supplied integer moves in the exact kernel of `[1'; A]` before evaluating their
binomials. No Markov basis is computed. Both check probability normalisation with
`atol=1e-12, rtol=1e-8`, separately from the membership tolerance.

Positive independence tables and a decomposable chain are checked against their
closed forms by both methods; the tests also cover an arbitrary integer design,
likelihood finite differences, the gauge, empty-row extended MLEs and independence
minors. The implementation note `lib/AlgebraicFactors.jl/src/fit.md` gives the
numerical thresholds and limitations.

`Grassmann(k, m)` represents a row space by a real `k × m` matrix `Θ` with
`Θ * Θ' ≈ I`. `project_tangent(M, Θ, G)` returns its horizontal gradient,
`retract(M, Θ, Δ)` orthonormalises the rows with positive-diagonal thin QR,
and `ispoint(M, Θ; atol=1e-8)` checks the representation. Retraction rejects
non-finite or numerically rank-deficient input (see its docstring).

```julia
M = Grassmann(2, 6)
Θ = rand(Random.Xoshiro(2403), M)
r = Retracted(Momentum(0.04, 0.6), M)  # also accepts GradientDescent
state = zero(Θ)                       # use nothing for GradientDescent
of = OptimiserFactor(:θ, r)
S = Diagonal([-2.0, -1.0, 1.0, 2.0, 3.0, 5.0]) # using LinearAlgebra
state, Θ = optimiser_step(of, state, Θ, 2Θ * S)
```

`Retracted` preserves Mycelium's momentum recurrence `s′ = γ*s - η*grad`
and transports the velocity by projection onto the new horizontal space.
Mycelium has no rule-state initializer; initialise it as above. Factor message
emission reads `(p=Θ, state=state)`. Nesterov and other rules are rejected:
Nesterov's look-ahead leaves the manifold. The geometry tests check first-order
descent and both optimisers against a known eigenplane, using `η=0.04` for
400 iterations; they check factor-driven updates against direct `rule_put`
at every step. The implementation note `lib/AlgebraicFactors.jl/src/grassmann.md`
records the exact problem, tolerances and limitations. This uses the canonical
Grassmann metric; it does not implement a statistical model's Fisher metric.

`plucker(z,w)` returns the bivector in lexicographic `(i,j)` order (`i<j`),
listed by `plucker_pairs(r)`. `plucker_jacobian` differentiates with respect to
`vcat(z,w)`. `plucker_relations(r)` returns sparse quadric records with increasing
quadruples, monomial coordinate-index pairs and coefficients `(1,-1,1)`;
`plucker_residuals(r,p)` evaluates them.

```julia
f = PluckerFactor(4)  # names default to (:z, :w, :p)
pol = only(supported_polarities(f))
site, st = factor_message(f, :p, pol,
    (z = DiracBelief([1.,2,3,4]), w = DiracBelief([5.,6,7,8])),
    nothing, NamedTuple(), NamedTuple())
# site.value == [-4,-8,-12,-4,-8,-4]
```

The only polarity observes both inputs and emits on `p`; inverse fibres are not
represented. Parameters and state are empty. Diracs transport exactly; sample
beliefs pair particles by index, requiring equal positive counts and matching
normalised weights (`rtol=1e-12, atol=0`). The first cloud's weights survive.
A Dirac may accompany a cloud. Samples are column matrices or vectors of vectors.

Separate proper Gaussian inputs are independent. Their exact output mean and
covariance are returned in a moment-matched `GaussianBelief`; the bilinear law
is generally non-Gaussian. `plucker_moments(μ,S,ν,T)` exposes the exact moments,
including singular covariance (zero input covariance represents a Dirac).
For example, Dirac/Gaussian output has rank `r-1` for nonzero fixed input.
Core Gaussian beliefs cannot represent singular covariance. By default the factor
rejects it, using `λmin <= n*eps(T)*λmax` as the numerical rank threshold;
it never adds jitter automatically. Explicit `covariance_jitter=δ > 0` adds
absolute variance `δ*I`, giving an approximate covariance. The implementation
note `lib/AlgebraicFactors.jl/src/plucker.md` derives the formula and explains
this distinction from singular precision.

Pointwise and graph-local energy are zero, as for `GeneratorFactor`.
Game-level free energy/entropy and an assembled lens are absent (LP-009), so the
full conformance checker does not apply. Tests exercise the supported interface
and a Mycelium DAG with two Gaussian priors, and compare moments to an independent
second-moment expansion and seeded Monte Carlo. The 400-point Klein-quadric
test now exercises `fit_nullspace` in monomial and Chebyshev bases.

Enumeration costs grow with the product of alphabet sizes. This implementation uses CPU
arrays and makes no GPU or automatic differentiation guarantee.


The §2.3 `Sampson gauge obligation (§2.3)` test uses a **test-only** two-equation
residual `r = Θ*v(z)` with the six degree-at-most-two monomials in two variables.
It checks an exact Jacobian against the core finite-difference default and Sampson
energy invariance under a well-conditioned invertible row transformation and QR
row orthonormalisation (`rtol=1e-10`). Seed `2307`, 20 random points, damping
`λ=0, 0.3`, and identity/non-isotropic data covariances are used for the row-transform
test. Gauge-compatible damping `Θ*Θ'` retains invariance at a constructed rank-zero
Jacobian, where zero damping throws. Default identity damping at `λ=0.3` and plain
`SquaredNorm()` each change energy by more than `1e-3` at some tested point.
The implementation note `lib/LenticulumCore.jl/src/energy.md` derives the covariance law and documents the numerical SPD threshold,
localisation and frozen Gauss–Newton metric. That original test remains alongside the §3(a) real polynomial-factor tests.
Polynomial messages (§3b) and homotopy (§4) remain unimplemented.


`ExponentTable`, `veronese`, `veronese_jacobian` and `basis_change` implement
monomial and Chebyshev tables (see `lib/AlgebraicFactors.jl/src/veronese.md`). Ordering is increasing degree,
then descending lexicographic exponents. There is no implicit data scaling;
Chebyshev is intended for [-1,1] without enforcing it. Degree-exact Chebyshev
products need not be homogeneous; incompatible square basis conversions throw.

`PolynomialFactor(dims,tab,k;Σ=I,λ=0)` initialises seeded Grassmann rows and empty
state. Raw energies accept named channel values split between x/y. For Sampson
scalarisation pack all values in dims order into y=z, with x=nothing, wrt=:y.
The exact Jacobian is ΘDv for **all** data coordinates. No message polarities
are declared. ΘΘ′ damping preserves row-gauge invariance; undamped energy is
also invariant under polynomial basis conversion, while coefficient-identity
damping is basis dependent. See the implementation note `polynomial.md`.

`fit_nullspace(tab,X,k;rtol=1e-12)` takes N×n points in columns and reports Θ,
all m singular values, numerical_rank, nullity and threshold. No centring,
normalisation or column scaling is performed. `fit_sampson` minimises the summed
half squared Sampson cost with the full analytic gradient, horizontal projection,
QR retraction and Armijo backtracking; method=:fns uses the corrected published
one-equation FNS update. Both stop at the horizontal Frobenius gradient norm,
and return Θ,cost,gradient_norm,converged,iterations. Singular covariance throws
with point index; explicit λ adds λΘΘ′. No global optimality is claimed.
The implementation note `polyfit.md` derives the gradient and records the
seeded Klein/conic/two-equation checks and their numerical results.

```@index
Modules = [AlgebraicFactors]
```

```@autodocs
Modules = [AlgebraicFactors]
```
