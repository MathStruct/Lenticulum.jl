#implementation

> One root-finder serves both directions of a DEQ, because both directions are the *same
> residual* solved for a different variable. That is the entire reason a DEQ is worth
> expressing as a factor.

> Sources: code: `solve.jl`, `deq.jl`, `../ext/ImplicitLayersEnzymeCoreExt.jl`, `../ext/ImplicitLayersMooncakeExt.jl`, `../test/parameter_ad.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## 1. Two solvers, and the difference is the caveat

$$
\text{Picard: } u \leftarrow u - \beta F(u)
\qquad\qquad
\text{Broyden: } u \leftarrow u - H F(u),\quad H \approx J^{-1}
$$

For the DEQ residual $F(z) = z - g(z,x)$, Picard *is* "keep applying the layer":
$z \leftarrow (1-\beta)z + \beta g(z,x)$. It converges **iff the iteration is a
contraction**, which is [[Implicit Learners]]'s recorded caveat —
*"this only works if the iteration converges, and unconstrained DEQs need not"* — and the
test suite makes it concrete: a linear cell with $\rho(W) = 1.6$ has a perfectly good unique
fixed point that Picard cannot reach and Broyden reaches in a handful of steps.

Broyden is initialised with $H_0 = I$, which makes its first step

$$u_1 = u_0 - I\cdot F(u_0) = u_0 - (u_0 - g(u_0)) = g(u_0)$$

**exactly a Picard step**, with everything after it the correction Picard never makes.
Asserted in the tests, and worth knowing because it means Broyden is never worse than Picard
on the first iteration.

Both root algorithms are derivative-free, so inference has no hard AD dependency.
Optional AD of a solved state is a separate IFT calculation. These are also what
`DeepEquilibriumNetworks.jl` does — Broyden and limited-memory Broyden are its workhorses.

## 2. `SolveReport`, and why nothing throws

A non-convergent solve returns `converged = false` and the caller decides. That is
[[Inversions and Bayesian Lenses]]'s position quoted directly: *a solver that stopped early is simply an
inexact inversion, and the loss records the cost.* Throwing would make a legal-but-poor
inversion into an error.

`nfe` is reported alongside `iterations` because iteration count is meaningless when a
Broyden step and a Picard step cost differently — `DeepEquilibriumNetworks.jl` reports it in
its `DeepEquilibriumSolution` for the same reason.

## 3. The implicit function theorem

$$
(I - \partial_z g)\frac{\partial z^\ast}{\partial x} = \partial_x g
\qquad\Longrightarrow\qquad
\frac{\partial z^\ast}{\partial x} = (I - \partial_z g)^{-1}\partial_x g
$$

**One linear solve, no unrolling** — the property that makes implicit layers $O(1)$ in memory
and the reason [[Implicit Learners]] lists the IFT as this family's backward pass. See
[[Backpropagation by the Implicit Function Theorem]].

`ift_sensitivity` throws `SingularException` when $I - \partial_z g$ is singular, and that
deserves emphasis:

> [!important] A singular $I - \partial_z g$ prevents applying the IFT
> It means the IFT does not establish a locally differentiable inverse; the relation may branch
> there, but uniqueness alone is not ruled out. The algebraic family calls that locus the discriminant
> ([[Branches and the Discriminant]]); it is literally the same phenomenon in a different
> model class, and it produces the same failure. Two of the three
> [[Implicit Learners]] families therefore fail in the same place for the same reason, which
> the vault had not previously connected.

## 4. Implementation difficulties

### 4.1 Optional AD uses the implicit function theorem

`fd_jacobian` remains a forward-difference reference. `deq_sensitivity(...; wrt=:ps,
ad)` and the differentiable `solve_state(...; ad)` use optional DI derivatives instead.
For a residual $r(z,u)=0$ they solve

$$
D_u z^* = -(D_z r)^{-1} D_u r.
$$

The ChainRulesCore weak extension supplies an `rrule` for the internal solved-state
primitive. A cotangent $v$ gives $\lambda$ by $(D_z r)^\top\lambda=v$, then
$\bar u=-(D_u r)^\top\lambda$ via a backend VJP. The ForwardDiff weak extension
strips duals before the root solve, obtains residual JVPs in their seed directions,
and solves the same linear system to rebuild the root's duals. Neither differentiates
Broyden/Picard iterations. Both the input and parameters are active; the initial guess,
solve report and returned Lux state are nondifferentiated bookkeeping.

AD requires a converged root and a nonsingular residual Jacobian. A singular Jacobian
means the IFT gives no local inverse; it does **not** prove nonuniqueness (for example
$r(z)=z^3$ has a unique root and zero derivative there). The solve is dense: a reverse
backend VJP in the ChainRules rule can avoid parameter-Jacobian storage, but the
Enzyme/Mooncake rules use a dense ForwardDiff VJP. The state Jacobian and its
factorisation cost quadratic storage and cubic factorisation work. A matrix-free
Krylov version remains future work. These are first-order rules; higher derivatives
through the primitive, the reverse `solve_input`, GPU and additional backends are untested.

The EnzymeCore extension provides native `EnzymeRules.forward`, `augmented_primal`
and `reverse` rules on `_implicit_state`. DI's default `AutoEnzyme()` Jacobian uses
batched forward mode; its scalar gradient uses reverse mode. The Mooncake extension
registers a native reverse `rrule!!` with `@is_primitive`; it accumulates the joint
vector cotangent in Mooncake's mutable forward data. Neither uses `@from_rrule`.
Both extensions also require ForwardDiff to be loaded. They use `AutoForwardDiff()`
through DI for the residual state Jacobian, JVPs and VJPs: these evaluate only
`_joint_residual`, never the solve primitive. `deq_sensitivity` uses this same inner
backend for Enzyme/Mooncake. ForwardDiff VJPs here form a dense residual Jacobian;
this is a small-system reference, not a matrix-free implementation. The reverse
rules factor the transposed state Jacobian once per primal solve.

Only the joint vector is active. `_solve_metadata` makes the backend, model,
parameter template, state and initial guess constant to Enzyme. The template supplies
shapes, not parameter values. `_joint_vector` copies inputs and flattened parameters
separately: tracing `vcat(constant_input, active_parameters)` merges their activities
in Julia's copy loop and fails Enzyme's static activity analysis. ChainRules supplies
the corresponding slicing pullback for this packing helper. Array parameter leaves
are materialised before packing to avoid array/view union types. ForwardDiff strips
duals from initial guesses even when only the initial guess is active.

`test/parameter_ad.jl` checks all four backends against the linear-cell closed form
and central differences, including active warm starts, a zero warm-start-only
Jacobian and a nonzero downstream scalar loss. Nonlinear scalar parameters,
nonempty Lux state, endpoint gradients and graph parameter steps have separate checks.

### 4.2 Broyden stores a dense inverse Jacobian

$O(n^2)$ memory, which is fine at test scale and hopeless for a real DEQ. That is exactly
why `DeepEquilibriumNetworks.jl` ships *limited-memory* Broyden: keep the last $m$ secant
pairs instead of the matrix. Not implemented; it is a self-contained addition and the obvious
next one.

Anderson acceleration is likewise absent, and is the other solver DEQ implementations
commonly use.

### 4.3 The tolerance is absolute

`tol` compares against $\|F(u)\|_2$ with no relative component and no scaling by $\|u\|$. On
a badly scaled problem that is either unreachable or trivially satisfied. A real solver takes
`abstol` and `reltol`; `NonlinearSolve.jl` does, and if this package ever grows a dependency
it should be that one.

### 4.4 Root state is fixed during `solve_state`

A fixed-point solve must evaluate one fixed residual, including its incoming Lux state.
`solve_state` now holds that state fixed through the root iterations and advances it once
at the final root. This also defines the equation used by the IFT. State is not an
additional implicit variable. A cell that mutates arrays in its incoming state in place
violates this functional contract. The reverse `solve_input` still has the legacy `Ref`
threading and has no AD rule.

Related: [[deq]], [[flow]], [[neuralode]], [[The Equilibrium Family]],
[[Backpropagation by the Implicit Function Theorem]], [[Branches and the Discriminant]]

Solved-state IFT AD accepts `AutoForwardDiff()`, `AutoZygote()`, `AutoEnzyme()` and
`AutoMooncake()`. Plain `AutoEnzyme()` works for these DEQ rules without runtime
activity; the built-in ODE stage fold requires an explicit runtime-activity mode
(see [[neuralode]]). Other solve backends are rejected explicitly. Missing Enzyme/Mooncake
extensions also raise an error before the solve; load DI, the backend and ForwardDiff.
Pass the backend to `solve_state` as well as the outer derivative.

For Enzyme outer functions that capture arrays, use DI constant contexts (as the tests
do), or explicitly set `AutoEnzyme(function_annotation=Enzyme.Const)`. Automatic
closure activity can raise `EnzymeMutabilityException` on this stack; it is outside
the solve rule. This reproduces with the linear-cell fixture from `test/runtests.jl`:

```julia
using Enzyme, ADTypes, DifferentiationInterface, ForwardDiff
function closure_probe(f, x, ps, st)
    θ, off = LenticulumCore._flatten(ps), LenticulumCore.offset_tree(ps)
    ad = AutoEnzyme()
    loss = θ -> first(solve_state(f, x,
        LenticulumCore._unflatten(θ, off, ps), st; ad))
    DifferentiationInterface.jacobian(loss, ad, θ) # EnzymeMutabilityException
end
# f = DEQFactor(CELL, (x=2,z=2)); ps, st = LuxCore.setup(Xoshiro(3), f)
# closure_probe(f, [0.7, -0.3], ps, st)
```

Validation including the runtime-activity flow follow-up (2026-10-26, Julia 1.13.1, CPU):
`julia --project=<package> -e 'using Pkg; @time Pkg.test()'` passed with the following
counts (sum of top-level test summaries) and `Pkg.test()` wall times:

| Package | Passed | Time |
|---|---:|---:|
| ImplicitLayers | 995 | 255.1 s |
| LenticulumCore | 316 | 46.8 s |
| Mycelium | 494 | 11.2 s |

The implicit environment resolved DI 0.7.21, Enzyme 0.13.216, EnzymeCore 0.8.26,
Mooncake 0.5.65 and ForwardDiff 1.4.6. The solve tests also exercise explicitly
selected Enzyme forward gradients and reverse Jacobians, besides DI's default modes.
DEQ graph learning holds messages fixed; it differentiates endpoint residuals, not
message schedules. Its final `solve_state` check is a primal validation of the learned
parameter, separate from the solved-state derivative oracles.

Both runtime-activity Enzyme modes pass the Euler/RK4 and ODE graph learning checks;
plain `AutoEnzyme()` still passes DEQ and scored LuxFactor checks. The separate
activity extension requires only DI and EnzymeCore. The vault build passes with
246 HTML pages and no KaTeX errors; `julia --project=docs docs/make.jl` passes in
435.9 s (including tutorial execution and the vault). It reports size warnings for
the core API page and search index. Both builds use a disposable `QUARTZ_DIR` copy
with its checkout/clean EXIT hook removed.
