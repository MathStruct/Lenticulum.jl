#implementation

> Fixed scalarisations act on energy vectors; point-dependent scalarisations first localise
> to a metric at a factor and point. The free-energy level retains its fixed scalarisations.

> Sources: code: `energy.jl`, `../test/runtests.jl`, `../../AlgebraicFactors.jl/test/runtests.jl`
>
> Theory (CT-ML wiki): [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Bicategory](https://mathstruct.org/CategoryTheory-ML-Wiki/Bicategory)

Implements: [[Scalar and Multivariate Energy]] — the two-energy design, and the algebra that
makes AutoBayes' Definition 22 a *corollary* of ours rather than a competitor.

## What is implemented

| concept | type |
|---|---|
| $(E, K)$ energy space | `ScalarEnergySpace`, `EuclideanEnergySpace`, `GradedEnergySpace` |
| $E_{dc} = E_c \oplus E_d$ | `oplus` / `⊕`, `GradedEnergySpace` |
| an element of $E_G$ | `GradedEnergy` (a `NamedTuple` keyed by factor) |
| $\sigma : E \to \mathbb{R}$ | `IdentityScalarisation`, `WeightedSum`, `SquaredNorm`, `GradedScalarisation` |
| a field of pointwise metrics | `AbstractPointScalarisation`, `SampsonNorm`, `localise`, `sampson_damping` |
| linear vs nonlinear (fixed scalarisations) | `islinear` |
| the Jensen gap | `jensen_gap` |
| the pointwise level | `pointwise_energyspace`, `pointwise_scalarisation`, `scalar_energy(...; polarity, σ)` |

`GradedEnergy` carries `+`, `-`, scalar `*` and `zero` because
[Definition 22](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#composition-energies-add-entropies-chain-definition-22)'s entropy law takes an **expectation** of
energies, which a bare monoid cannot support.

## The two theorems, in code

**Strict for linear $\sigma$.** `jensen_gap(σ, samples) == 0.0` short-circuits on
`islinear(σ)`, and the test asserts
`scalarise(σ, mean(Fc)) ≈ mean(scalarise.(Ref(σ), Fc))`.

**Lax for convex $\sigma$, by exactly a variance.** For `SquaredNorm()` the test asserts

$$\texttt{jensen\_gap} \;=\; \tfrac12 \operatorname{tr}\operatorname{Cov}(\mathbf{F}^c)$$

against `Statistics.cov(...; corrected=false)`, and

$$F^{dc}_{\text{scalar}} \;=\; \sigma\bigl(\mathbf{F}^{dc}\bigr) \;+\; \texttt{chain\_rule\_defect}$$

with $\sigma(\mathbf{F}^{dc}) \le F^{dc}$. These are the two claims the whole design rests
on, so they are tested numerically rather than argued.

## Recovering the paper exactly

```julia
energyspace(f)    = ScalarEnergySpace()
scalarisation(f)  = IdentityScalarisation()
```

are the defaults on `AbstractLenticulumFactor`. A factor that ignores the vector machinery
behaves exactly as AutoBayes Definition 20 specifies. **The generalisation is opt-in**,
which is the only acceptable way to ship it — a reader of the paper must be able to write a
factor without learning our extension first.

## Two levels: pointwise and free energy

`energy` returns the **pointwise** energy, `free_energy` (and a graph's `local_free_energy`) the
**free** energy. AutoBayes has them in one space, and so does every factor by default:

```julia
pointwise_energyspace(f)   = energyspace(f)
pointwise_scalarisation(f) = scalarisation(f)
```

A residual factor overrides the pointwise pair: its `energy` is the signed residual in
`EuclideanEnergySpace(k)`, scalarised by `SquaredNorm(Q⁻¹)`, while its free energy stays graded
`(fit, complexity, negentropy)` with the identity on each summand. The expectation of the
scalarised residual is the `fit` summand, which the tests check at a point for
`GaussianFactor`, `LinearConstraintFactor`, `NonlinearFactor` and `LikelihoodFactor` (the last
has the negative log-likelihood as a one-element vector, summed by `WeightedSum([1.0])`).

Before this split, `energyspace`/`scalarisation` were used for both levels, and for a residual
factor they described only the free energy: `scalar_energy` applied a graded scalarisation to a
residual vector and threw a `MethodError` (issue #8). The graph's scalar free energy was right
all along, which is why the suite did not notice.

The scalarisation is a parameter: `scalar_energy(f, x, a, y, ps, st; σ = SquaredNorm())` and
`scalar_free_energy(f, π, y, ps, st; σ)` override the factor's default for one call. All four
space/scalarisation functions take an optional polarity and fall back to the polarity-free
method; no factor yet defines a polarity-dependent one. The choices this leaves open (one space
or two, cone or gauge, which σ and metric, normalised or not, per polarity) are
[[Scalar and Multivariate Energy]] §8.

## Localising a field of metrics

Following [[L-ADR-0007 Point-dependent scalarisations are localised before use]]
(accepted 2026-10-25), a point scalarisation is a field $p \mapsto \sigma_p$, not a single
map on energy space. `localise(σ, f, x, a, y, ps, st)` returns an ordinary scalarisation.
Every fixed scalarisation returns itself (`===`). `scalar_energy` localises using the
incoming state, then evaluates the residual and returns the energy's outgoing state.
`scalarise` on an unlocalised point scalarisation throws; `islinear` is false.

For `SampsonNorm(wrt=:y, Σ=nothing, λ=0)`, let $u$ be the selected `:x` or `:y` data
coordinates and $J=D_u r$. Localisation returns `SquaredNorm(M)` with

$$
A=J\Sigma J^\top+\lambda R,\qquad M=A^{-1},\qquad
\sigma_p(r)=\tfrac12 r^\top M r.
$$

`Σ=nothing` means identity in data space; a supplied real covariance is checked for
finiteness, shape, symmetry and PSD with the `SquaredNorm` checking convention.
Nonzero damping also checks the factor's `R`. `λ` must be finite and nonnegative.
The numerical SPD policy is $\lambda_{\min}(A)>n\operatorname{eps}(T)\lambda_{\max}(A)$,
where $n$ is the residual dimension and $T$ the floating element type. Non-finite,
empty, singular or numerically deficient covariance throws `ArgumentError`; there
is no pseudoinverse or automatic damping. The verified covariance is inverted and
wrapped with `SquaredNorm(...; check=false)`. Dense real CPU arrays are the supported
Sampson path; it preserves Float32 when the Jacobian and supplied coefficients do.

The default `energy_jacobian` uses central finite differences. Factors can override
it exactly. `gauss_newton_metric(f, ...; σ=SampsonNorm())` localises first and forms
$J_{\mathrm{pull}}^\top M J_{\mathrm{pull}}$. This **freezes the metric at the point**:
it ignores derivatives of $M$ and is not the Hessian of the varying-metric energy.
The pull-back's `wrt` is independent of `SampsonNorm.wrt`, which selects the noise space.
Finite-difference localisation does not provide AD through the metric.

A graded scalarisation can contain metrics for different factors. Passing the whole
composite context to every part is unsound: `TensorFactor` and `ComposedFactor` parts
need their own factor, arguments, parameters and state. Until that routing is specified,
`localise` rejects any graded scalarisation containing point-dependent parts, recursively.
Callers can localise parts in their own contexts and build an ordinary graded scalarisation.
Fixed graded scalarisations still return themselves.

`jensen_gap` rejects point-dependent scalarisations (also inside gradings, even with
empty samples). Different points use different maps; the fixed-map Jensen identity
cannot describe their expectation. One may apply `jensen_gap` to one frozen metric,
which measures a different quantity. Free-energy scalarisation in `scalar_free_energy`
and Mycelium's Bethe sum is unchanged: it acts on free energies, not point residuals.

### Gauge-compatible damping

For a residual $r_\Theta(z)=\Theta v(z)$, write $D=D_zv$, so $J=\Theta D$.
Under $\Theta\mapsto B\Theta$, both $r\mapsto Br$ and $J\mapsto BJ$.
To retain invariance when damping, a factor overrides `sampson_damping` with a
PSD matrix satisfying $R\mapsto BRB^\top$. In particular $R=\Theta\Theta^\top$ gives

$$
J\Sigma J^\top+\lambda\Theta\Theta^\top
=\Theta(D\Sigma D^\top+\lambda I_m)\Theta^\top.
$$

Here $m$ is the number of basis functions. This identity holds for **any** data
covariance $\Sigma$ of the right size, not just $\Sigma=I$; in that case its middle
factor is $DD^\top+\lambda I_m$. For full-row-rank $\Theta$, PSD $\Sigma$ and
$\lambda>0$, the result is SPD even if $J$ loses rank. Since $A\mapsto BAB^\top$,
$(Br)^\top(BAB^\top)^{-1}(Br)=r^\top A^{-1}r$. Identity damping generally fails
this transformation law. Exact gauge invariance does not imply numerical acceptance
for arbitrarily ill-conditioned $B$: the SPD threshold depends on the representative.

### Checked examples and limits

Core's `Point-dependent scalarisation localisation` testset uses the test-only circle
$r(y)=y_1^2+y_2^2-1$. At radius $\rho>0$, its energy is
$(\rho^2-1)^2/(8\rho^2)$ (relative tolerance `2e-10`). The ratio to geometric
energy $(\rho-1)^2/2$ tends to one: for both signs of $\rho-1$, with magnitudes
`1e-2, 1e-3, 1e-4, 1e-5`, the ratio error decreases and is below
`1.1*abs(ρ-1)`, finally below `1.1e-5`. Covariance $s^2I$ scales energy by $1/s^2$.
At the centre, zero damping throws and positive damping gives a finite value.
Tests also check identity localisation of fixed scalarisations, old scalar energies,
input-state use, validation, Float32 metric types, inversion overflow rejection,
large finite covariances, and distinct noise/pull-back coordinates in the frozen
Gauss–Newton formula.

AlgebraicFactors' `Sampson gauge obligation (§2.3)` testset checks a test-only
six-monomial, two-equation residual. With `Xoshiro(2307)`, 20 random points and
$B$ with singular values `0.7, 1.8`, energy agrees to `rtol=1e-10` under $B$ and
QR row orthonormalisation, for damping `0, 0.3`. The $B$ test covers identity and
non-isotropic covariance. Exact Jacobians agree with the finite-difference default
(`rtol=2e-10, atol=2e-9`). Identity damping and plain algebraic distance each differ
by more than `1e-3` at some tested point. Two quadratics at the origin have
rank-zero $J$: zero damping throws and $\Theta\Theta^\top$ damping remains finite
and gauge invariant. This is the §2.3 obligation, **not** a §3 polynomial factor,
fit, IRLS, inference method or a general numerical invariance guarantee.

## Implementation difficulties

### 1. Eager accumulation is only valid for linear $\sigma$

This is the difficulty that motivates `islinear` existing at all. Walking a graph and
adding up scalar losses as you go computes $\mathbb{E}[\sigma(\mathbf{F})]$. The
multivariate chain rule asks for $\sigma(\mathbb{E}[\mathbf{F}])$. For linear $\sigma$ these
coincide and eager accumulation is free; for convex $\sigma$ they do not, and the vector
loss must be assembled first.

There is no way to detect this at runtime after the fact — both are finite numbers, both
look plausible, and for convex scalarisations, eager scalarisation gives the *larger*
value by Jensen’s inequality. Hence the trait.

### 2. `oplus` on unnamed operands invents `:left` / `:right`

Folding a binary `⊕` gives nested rather than flat gradings, so associativity holds only up
to renaming. Acceptable (composition of open models is likewise bicategorical) but a trap.
**Build $E_G$ in one step from a node-keyed `NamedTuple` once the graph layer exists**;
treat binary `oplus` as a test-and-prototyping convenience.

### 3. The entropy has to be told *which coordinate* it regularises

$\mathbf{H}^c$ is typed into $K_c$, but entropy is naturally a scalar. The idiom is to
return `h * u` for a fixed direction `u ∈ K_c`. This is not busywork — it is what makes it
possible to regularise one block of a factor and not another, which is the whole point of
per-coordinate precisions. But it *is* an extra decision imposed on every factor author,
and a `ScalarEntropy` wrapper that broadcasts against a default direction would be worth
adding once there is evidence about what direction people actually pick.

### 4. Metric validation

`SquaredNorm(M)` checks symmetry (Hermitian symmetry for complex metrics) and positive
semidefiniteness, including structured and non-strided matrices. A `Diagonal` metric only
needs real, nonnegative diagonal entries; other representations use Cholesky with an
eigenvalue fallback for semidefinite metrics. `check = false` skips numerical validation
for AD/GPU callers that already know the metric is valid. The strided matrix path keeps
its existing numeric-type guard and operations, including its treatment of AD element types. The validation test in `test/runtests.jl` covers indefinite and
semidefinite diagonal, symmetric, Hermitian and indexed-view metrics.

### 5. `scalarise` on `GradedScalarisation` requires matching names

`scalarise(σ::GradedScalarisation{names}, e::GradedEnergy{names})` is constrained to
identical `names`. A mismatch is a `MethodError` — which is the right failure (loud, at the
right place), but the message will be unhelpful. A custom error explaining "the
scalarisation's grading does not match the energy's" is worth adding. Issue #8 was this
error in another form: a graded scalarisation applied to a plain vector.

### 6. Parameter derivatives and optional AD (LP-002, 2026-10-26)

`energy_jacobian(...; wrt=:ps, ad)` returns the pointwise vector energy's parameter
Jacobian. Extending the existing function keeps input and parameter pull-backs on the
same API; `wrt=:x` remains the compatible default. `wrt=:y` is also supported.
`gauss_newton_metric` accepts the same backend and coordinates. With no backend,
central differences remain the reference/default.

`energy_gradient` differentiates `scalar_energy`; `free_energy_gradient` differentiates
`scalar_free_energy`. They return the same numeric tree structure as `ps`, including
empty subtrees, scalar leaves and array shapes. `parameter_gradient(loss, ps; ad)`
provides the same mechanism for a user scalar loss. Incoming state is held fixed;
derivative evaluations do not commit state. A family using randomness must itself
provide an appropriate differentiable expectation/estimator; these APIs do not invent one.

Load `DifferentiationInterface` and the backend package, then pass an ADTypes object,
for example `ad=AutoForwardDiff()` or `ad=AutoZygote()`. DI is a weak dependency;
passing a backend without its extension gives a loading hint. Backend compatibility
still depends on the model's operations. `_check_factor_ad(ad, f, st)` is an optional
compatibility hook called before tracing factor Jacobians and energy/free-energy
gradients. Its default delegates to `_check_factor_ad(ad, f)`, preserving existing
two-argument overrides and callers. ImplicitLayers uses the state-aware hook to
reject SciML AD with nonempty Lux state, while primal evaluation and reference
differences remain available, and the backend hook to require runtime activity for Enzyme Euler/RK4;
see [[neuralode]] and its parameter AD tests. Core's `test/parameter_ad.jl` checks both
backends, Enzyme and finite differences against linear residual Jacobians and quadratic gradients.
Its Enzyme Jacobians use `AutoEnzyme(; mode = Enzyme.Reverse)`: DI's default
`AutoEnzyme()` Jacobian runs forward mode, which with Enzyme 0.13.216 returns a NaN
entry in about 5% of calls on `u .- A*x` with constant `A`, `x`. The defect is
upstream ([Enzyme.jl#3884](https://github.com/EnzymeAD/Enzyme.jl/issues/3884));
`examples/lp002_enzyme_endpoint_limit.jl` reproduces it without Lenticulum.
The top-level `test/runtests.jl` also checks GaussianFactor's parameter Jacobian,
precision-weighted gradient and Gauss–Newton metric against their closed forms.

Graded scalarisation uses an explicit empty case rather than the tested stack's
unsupported Zygote `sum(::Tuple; init)` rule; constant objectives rebuild zero trees
even when a backend returns `nothing` for an inactive derivative. Both cases have
analytic gradient tests.

Coordinate selection is dispatched before the AD trace, keeping active and constant
arrays separate for Enzyme static activity analysis. Existing exact Jacobian overrides
can still omit the `ad` keyword on calls without a backend. After adding weak extensions
to an already-instantiated checkout, run `Pkg.resolve()` in that environment to refresh
its manifest extension metadata.

The shared tree implementation is [[parameter_tree]]. Backend AD through a
finite-difference Sampson localisation is not supported; this addition does not supply
higher-order rules for that metric. Gauss–Newton freezes the local metric as before.

Related: [[Scalar and Multivariate Energy]], [[Factors are Parameterized Statistical Games]], [[statistical_game]]


## Device parameter derivatives (2026-10-27)

The GPUArraysCore/DI optional extension supports Float32/Float64 numeric array leaves
with Zygote and ForwardDiff; evidence is `lib/ImplicitLayers.jl/test/device_arrays.jl`.
`energy_gradient`, `free_energy_gradient` and `parameter_gradient` keep gradient
arrays on the parameter device. ForwardDiff's native vector API scalar-indexes its
seed arrays on JLArrays. The Core adapter instead seeds whole directional vectors
by broadcast through `DI.pushforward`, assembling Jacobian columns or scalar-gradient
entries without host scalar indexing. Cost is one evaluation per coordinate; this
is intended for small reference problems. Direct `ForwardDiff.gradient/jacobian`
on device vectors still has the upstream limitation. CPU DI dispatch is unchanged.

Device finite differences and Enzyme/Mooncake parameter AD give ArgumentErrors;
this batch does not add native device rules for those backends. Real numeric array
leaves are tested; scalar parameter leaves in a device tree are rejected (use
one-element device arrays). Higher-order nesting is untested.


Device functional flattening also handles empty parameter subtrees and all-empty
array leaves; see [[parameter_tree]]. The DEQ nested-solve tests preserve storage
and precision through both supported backends. CPU concatenation is unchanged.


## LP-002 batch C: differentiating localisation (2026-10-28)

History: `b4d77ab0` introduced finite-difference Sampson localisation on
2026-10-25; LP-002's `6e0bbbb6` Remaining list named its AD follow-up on
2026-10-26. This means the **total** derivative of the pointwise scalar energy,
including the local covariance $A=J\Sigma J^\top+\lambda R$, $J=D_u r$.
It does not mean changing the deliberately frozen-metric Gauss–Newton pullback.
For $M=A^{-1}$ and $w=Mr$,

$$
d\left(\tfrac12r^\top Mr\right)=w^\top dr-\tfrac12w^\top(dA)w.
$$

The second term requires mixed derivatives of $r$ and any derivative of the
factor's damping covariance. `energy_jacobian(...; ad=...)` does not automatically
change the Jacobian that `SampsonNorm.localise` constructs: the latter still calls
the default central-difference reference unless the factor overrides it.
The generic reference mutates perturbations and Jacobian columns; numerical SPD
validation uses a spectral test before inversion. Wrapping it in outer AD is not
a supported higher-order metric interface. A successful scalar endpoint gradient
or `J' * M * J` test says nothing about this total derivative.

`lib/ImplicitLayers.jl/examples/lp002_metric_divergence_limits.jl` reproduces the
localisation path on $r=a y^2-b$, with $y=1.3$ and no damping. Its independent
closed form is $(a y^2-b)^2/(8a^2y^2)$. The batch C section of LP-002 records primal,
reference-difference and backend observations separately. Nested reference
differences can amplify the inner Jacobian's cancellation error; they do not
constitute an AD implementation. An opt-in functional Jacobian/metric path or
an analytic factor-specific localisation still needs its own checked derivative
contract, including rank/acceptance boundaries. No metric or SPD default changes
in this batch.
