#implementation

> `DEQFactor`: the fixed-point condition $z = g_\theta(z,x)$ kept as a **residual**, so the
> polarity decides which channel to solve for.

> Sources: original residual/adjoint derivations; code: `deq.jl`, `../test/solver_numerics.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## Graph free-energy edge directions

`DEQFactor` declares `Mycelium.uses_cavity == true`: its local free energy evaluates the
residual at incoming points, even though it returns zero if a point is absent.
Graph free-energy evaluation rejects an attached `Emitting()` edge with `ArgumentError`;
use `Bidirectional()` if that channel must also send messages. Missing messages on permitted
edges retain their existing local behaviour. See [[free_energy]] and
[[LP-014 Emitting edges drop the cavity message from the free energy]].

## 1. What SciML gives you, and what it keeps

`DeepEquilibriumNetworks.jl` builds a Lux layer. You hand it a cell and a
`NonlinearSolve.jl` solver, and you get `x ↦ z★(x)` — the relation solved in one direction,
with the solve sealed inside a function and the solve's diagnostics tucked into `st` as a
`DeepEquilibriumSolution`.

Everything in that sentence is good engineering and none of it is a relation any more. The
object that *is* a relation is the residual:

$$r(x,z) = z - g_\theta(z,x), \qquad R_\theta = \{(x,z) : r(x,z) = 0\}$$

which is [[README]]'s definition of an implicit learner, verbatim. `DEQFactor` keeps it, and
therefore gets both directions out of one solver:

| polarity | solves | who can do this |
|---|---|---|
| `x` observed, `z` unobserved | $z = g(z,x)$ for $z$ | SciML's DEQ, and this |
| `z` observed, `x` unobserved | $z = g(z,x)$ for $x$ | **only this** |

Same residual, same `BroydenSolver`, different variable held fixed. The test suite checks
both against the closed form for a linear cell, and checks that they really are inverse to
each other.

## 2. The energy is a proper vector energy

`pointwise_energyspace` is `EuclideanEnergySpace(dim z)` and
`pointwise_scalarisation` is `SquaredNorm` — the
residual is returned as an $\mathbb{R}^n$-valued object, not collapsed to a number.

This is what [[Scalar and Multivariate Energy]] asks every implicit factor to do and what
`VariationalDiffusion.jl`'s factor could not deliver (its `predictor.md` §4.3 records the
inconsistency). Here it comes for free, and it is exactly what the IFT needs — that note's
§6.3 is about needing the Jacobian of the *vector* residual.

The game energy instead has `ScalarEnergySpace()` and `IdentityScalarisation()`:
it contains the squared residual, not the residual vector.

## 3. The reverse direction needs a square residual

Solving $z = g(z,x)$ for `x` is $\dim z$ equations in $\dim x$ unknowns. Unless
$\dim x = \dim z$ it is over- or under-determined, and a root-find is not asking a well-posed
question.

`supports_polarity` refuses that polarity when the dimensions differ, and
`supported_polarities` returns one element rather than two — so a non-square DEQ correctly
reports `isunidirectional == true`, i.e. *it really is just a Lux layer*. Asserted in the
tests.

> The honest reading: the bidirectionality of a DEQ factor is conditional on a shape
> coincidence. A DEQ whose input and state have different widths — which is the common case
> in practice, since the input is usually an embedding — is unidirectional and gains nothing
> from this package over `LuxFactor`. What would help is a least-squares solve for the
> over-determined case and a regularised one for the under-determined case; neither is
> implemented.

## 4. Implementation difficulties

### 4.1 Every message is a `DiracBelief`

A root-find returns a point. Propagating a *distribution* would need the linearisation
`deq_sensitivity` already computes — push a covariance through $\partial z^\ast/\partial x$ —
and somewhere to put it, i.e. a `GaussianBelief`.

**`GaussianBelief` used to live in the top-level `Lenticulum` package**, out of reach of every
`lib/` package. It has moved to `LenticulumCore` (October 2026, [[The Equilibrium Family]] §5),
and `VariationalDiffusion` already returns Laplace Gaussians. For this factor the same is
possible, $\Sigma_y = (\partial z^\ast/\partial x)\,\Sigma_x\,(\partial z^\ast/\partial x)^\top$ from `deq_sensitivity`,
but not built.

### 4.2 The prior warm-starts the solve

The `prior` argument supplies the solver's initial guess. The returned Dirac site does not
condition on that prior; `likelihood_message` therefore uses the default point-lens path.
A conflicting hard prior cannot be combined with that site. The lens game below records
that incompatibility as infinite anchor cost; it does not change the solve into Bayesian
conditioning. No distributional posterior is supplied.

### 4.3 Point-belief entropy convention

`local_free_energy` returns $\tfrac12\|r\|^2$ without an entropy term. The game adopts the
same zero point-belief entropy convention as Mycelium's counting correction; this is not
the differential entropy of a Dirac relative to Lebesgue measure. The graph method can also
read a Gaussian's mean, but supplies no joint distributional factor entropy. Consequently,
mixing these residual diagnostics with Gaussian factors does not establish a normalized
marginal-likelihood interpretation for the resulting Bethe objective.

### 4.4 The initial guess is zeros

`solve_state` starts from $z_0 = 0$ unless warm-started. `SkipDeepEquilibriumNetwork` exists
precisely because a *learned* initial guess $z_0 = h_\phi(x)$ converges much faster and
regularises training. That is a second parameter tree on the inversion — the
`AmortisedInversion` situation of [[Inversions and Bayesian Lenses]] — and implementing it would make this the
first factor in the project with genuinely two parametrised halves. Not done.

### 4.5 Sensitivities and training

`deq_sensitivity(f, x, zstar, ps, st; wrt=:x, ad=nothing)` defaults to central
differences; `wrt=:ps` selects flattened parameters. Supplying an ADTypes backend
uses DI residual derivatives. `solve_state(...; ad)` also has optional ChainRulesCore,
ForwardDiff, EnzymeCore and Mooncake extensions implementing the IFT through the
solved state; see [[solve]].
`DEQFactor(...; ift=DenseIFT())` keeps dense LU as the default.
`ift=KrylovIFT(; rtol, atol, itmax, memory, precond)` selects matrix-free GMRES in all
solved-state AD rules (load Krylov and DI); failure to converge throws.
`deq_sensitivity` remains a dense reference regardless of this option. See [[solve]]
for the operator products, storage, backend choices and 200-dimensional oracle.
Its incoming state is fixed throughout iterations and advanced once at the root.

For $z^*=(I-W)^{-1}(Ux+b)$, a perturbation obeys

$$
\delta z^*=(I-W)^{-1}(\delta W z^*+\delta Ux+\delta b).
$$

`test/parameter_ad.jl` checks this parameter Jacobian against ForwardDiff, Zygote,
Enzyme, Mooncake and central differences, and checks a scalar loss on the solved state. `energy_gradient`
and `free_energy_gradient` instead score fixed incoming endpoints, so their DEQ
parameter derivatives need no root solve. Graph `parameter_step` uses this latter
objective with fixed messages; it does not backpropagate through message schedules.

Related: [[solve]], [[luxfactor]], [[neuralode]], [[DEQ as a Relation]],
[[The Equilibrium Family]], [[Scalar and Multivariate Energy]], [[Implicit Learners]]

## Incoming factor game and solved-point lens game (LP-009)

`free_energy(f, π, y, ps, st)` follows `GaussianFactor`'s fixed channel order:
`π` is the incoming point on the first channel (`x`), `y` the point on the second (`z`).
This corresponds to the reverse split, with `x` unobserved and `z` observed; the factor
method takes no polarity and does not reinterpret the arguments for a forward solve.
Both must be `DiracBelief`s. The method passes them to `local_free_energy`; entropy is
scalar zero. Other beliefs throw because no distributional residual expectation is defined.

`lens = first(assemble(f, p, ps, st))` restricts the relation to polarity `p`.
`free_energy(lens, π, inputs, ps, st)` runs that inversion and evaluates the residual at its
returned point, plus the prior cost on the unobserved channel. Here `inputs` is the named
Dirac observation (or that Dirac directly). A trivial prior costs zero, a Dirac costs zero
only at the same point and infinity elsewhere, and a proper Gaussian costs its quadratic
penalty relative to its mean. No normalizing constant is charged. This is where Definition
20's expectation under the inversion lives, with zero point-belief entropy. A finite channel
precision does not soften a Dirac prior; soft observations and singular Gaussian anchors
throw. The solver still uses the prior only as a warm start.

For residuals affine in the unknown `u`, with root `u*`,

$$
\tfrac12\|r(x,u_0)\|^2 = \tfrac12\|J_u(u_0-u^*)\|^2.
$$

Thus the incoming factor energy equals the solved residual energy (zero at a root) plus
an anchor of precision `J_u'J_u`. `energy_jacobian` supplies this metric; the caller chooses
it, and the lens game does not invent a prior. For `g(z,x)=z/2+x`, at `x=1,z₀=3`, the root
is `z*=2`, `Jz=1/2`, so both sides are `½(1/2)²=1/8`. In reverse at `z=3,x₀=1`, the root
is `x*=3/2`, `Jx=-1`, giving `½(1/2)²=1/8` again.

`test/conformance.jl` checks both full factor games without `game_gap`, solved-point lens
games, hard-prior infinity, and the exact linear identity. Its nonlinear cell has
`r= z+z²/4-x`, root `z*=2` at `x=3`, and `Jz=2`. At `z₀=2+h` the incoming energy is
`2h²+h³/2+h⁴/32`; the Gaussian anchor costs `2h²`. The gap is `h³/2+h⁴/32`, checked at
four halving step sizes. The residual approximation is first order (Gauss–Newton), with
cubic absolute energy error and, generically, first-order relative energy error near the root.

Scoring an arbitrary point uses `energy`/`scalar_energy`, the pointwise level of
[[L-ADR-0003 Pointwise energy and free energy are separate levels]], not a keyword on
`free_energy`. Shared implementation and limitations: [[point_game]].

Solved-state IFT AD accepts `AutoForwardDiff()`, `AutoZygote()`, `AutoEnzyme()` and
`AutoMooncake()`. Plain `AutoEnzyme()` works for these DEQ rules without runtime
activity; the built-in ODE stage fold requires an explicit runtime-activity mode
(see [[neuralode]]). Load DI, the backend and (for Enzyme/Mooncake) ForwardDiff; missing
extensions and other solve backends are rejected before solving. Pass the backend to
`solve_state` and the outer derivative. Enzyme outer closures that capture arrays
need DI constant contexts or `function_annotation=Enzyme.Const`; see [[solve]] for
the limitation and reproducer. Hardware GPU execution remains untested; the JLArrays subset is described below. Second order uses the dedicated dense `deq_hvp` API below.


## Solver numerics and the other polarity (2026-10-27)

`LimitedBroydenSolver(; memory=20, maxiters=100, tol=1e-10, init_scale=1.0)`
is an opt-in primal with $O(mn)$ storage. The default stays dense `BroydenSolver()`.
No eviction reproduces its iterates up to roundoff; finite memory discards the
oldest rank-one correction. The first step is Picard at unit `init_scale`, and
damped Picard otherwise. [[solve]] gives the compact update and its limits.

LP-002's “reverse-input” item means `solve_input(...; ad)`: hold $z$ fixed and
solve for $x$, then differentiate that solution with respect to observed $z$ and
parameters. The input derivative of `solve_state` was already covered. For the
linear cell:

$$
x^*=U^{-1}((I-W)z-b),\qquad
\delta x^*=U^{-1}((I-W)\delta z-\delta Wz-\delta Ux^*-\delta b).
$$

ForwardDiff, ChainRules/Zygote, Enzyme forward/reverse and Mooncake now implement
this first-order IFT through the shared solve primitive, including dense and
preconditioned Krylov solves. An internal problem adapter changes which residual
coordinate is unknown. Warm starts, reports and returned Lux state have no tangent.
The AD equation holds incoming state fixed and advances it once at the root.
The existing no-AD reverse solve retains its legacy state threading, so stateful
AD and no-AD solves need not agree. Branch selection remains the solver's job;
the derivative is local to its converged, nonsingular branch. Nonsquare systems,
nonconvergence and singular residual Jacobians remain errors in AD.

Krylov accepts a left operator or a problem factory returning `(; left, right)`;
`ldiv=false` applies inverse preconditioners with `mul!`, and `ldiv=true` uses
`ldiv!`. Factories receive a `transposed` flag and must honor it. Built-in
`JacobiIFT()` probes the current operator diagonal in $n$ products and stores
$n$ entries. `on_solve` records each RHS's iteration count and actual residual.
The full contract, setup costs and final residual policy are in [[solve]].

For a scalar loss depending only on the forward fixed point, use
`deq_hvp(loss, f, x, ps, st, v; wrt=:ps, ad=AutoForwardDiff())`.
`:x` selects the input Hessian block and `:joint` includes input/parameter mixed
blocks in input-then-parameter order. The result is a flat vector in the selected
coordinates. A dedicated second-order adjoint differentiates residuals and loss
twice at one root; no solver trajectory is differentiated. Only dense CPU real
ForwardDiff residual/ loss derivatives are supported, with quadratic storage in
the combined coordinates. The reverse polarity uses `solve_input_hvp` below.
Krylov HVPs and arbitrary AD nesting remain unsupported. Unsupported backend selections and detected
nested primitive derivatives throw `ArgumentError`; use the dedicated API.

`test/solver_numerics.jl` compares reverse derivatives to the closed form above,
a nonlinear selected branch and reference differences. HVP tests compare to the
analytic linear solve (including mixed blocks), scalar analytic second derivatives,
and finite differences of first-order IFT gradients. See LP-002 for run measurements.


## Batch A: reverse-input second order (2026-10-27)

`solve_input_hvp(loss, f, z, ps, st, v; wrt=:ps, ad=AutoForwardDiff(), x₀=...)`
uses the same dense second-order adjoint through the input-problem adapter.
`:z` selects observed state; `:joint` orders observed state before flattened
parameters. The loss is a function of solved input only. Incoming state is fixed;
branch selection uses `x₀`. Square dimensions, a converged root and nonsingular
input residual Jacobian are required. Backend, dense-storage and nesting limits
are identical to `deq_hvp`.

In the derivation in [[solve]], replace the unknown by x and observed coordinate
by z; the equations and signs are unchanged. `test/lp002_batch_a.jl` compares
linear mixed Hessian blocks to the closed-form inverse and finite differences of
first-order IFT gradients. For `x=sqrt(z-a*z²-b-bias)` it checks the analytic
second state derivative and the full joint Hessian product. No root iterations
are differentiated.


## Device first-order rules (2026-10-27)

`test/device_arrays.jl` covers Picard, dense Broyden and limited Broyden roots in
both polarities with JLArrays scalar indexing disabled. Device default warm starts
follow the observed array's storage and precision; explicit guesses still select
their own storage. CPU defaults are unchanged. Both observed-channel and parameter
IFT gradients match linear closed forms with opt-in KrylovIFT, including retained
Broyden and Jacobi preconditioning. Active warm starts have zero derivative.
DenseIFT has no working LU on JLArrays and gives a clear error; no host fallback
is supplied. Enzyme/Mooncake device rules, device HVPs and complex rules are not
supported. See [[solve]] for storage, tolerances and ForwardDiff directional costs.


Coupled Mycelium replay passes its selected AD backend to DEQ message solves via
the DI extension's private gradient-message hook. Both message directions now use
IFT rules, checked against nonzero downstream linear losses on JLArrays. Raw
ChainRules/Zygote inference replays the same primitive. Use the graph gradient
entry point with explicit `ad` for ForwardDiff; raw implicit `infer!` is not an
automatic backend-selection interface. Reverse-input message replay with nonempty
Lux state throws, since legacy primal state threading and fixed-state IFT define
different equations. Normal inference/default diagonal training are unchanged.

Raw ForwardDiff DEQ messages now reject dual-valued unannotated solves with an
ArgumentError pointing to the explicit graph gradient driver, preventing accidental
root-iteration differentiation. Ordinary primal messages are unchanged.
