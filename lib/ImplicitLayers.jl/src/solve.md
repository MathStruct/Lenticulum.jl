#implementation

> One root-finder serves both directions of a DEQ, because both directions are the *same
> residual* solved for a different variable. That is the entire reason a DEQ is worth
> expressing as a factor.

> Sources: Montoison and Orban (2023), *Krylov.jl*, p. 1 (operator products); Saad and Schultz (1986), GMRES;  code: `solve.jl`, `deq.jl`, `../ext/ImplicitLayersEnzymeCoreExt.jl`, `../ext/ImplicitLayersMooncakeExt.jl`, `../test/parameter_ad.jl`, `../test/solver_numerics.jl`
>
> Bibliography: [[Bibliography#^montoison2023krylov|Montoison & Orban 2023]] · [[Bibliography#^saad1986gmres|Saad & Schultz 1986]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## 1. Root solvers and the convergence caveat

$$
\text{Picard: } u \leftarrow u - \beta F(u)
\qquad\qquad
\text{Broyden: } u \leftarrow u - H F(u),\quad H \approx J^{-1}
$$

For the DEQ residual $F(z) = z - g(z,x)$, Picard *is* "keep applying the layer":
$z \leftarrow (1-\beta)z + \beta g(z,x)$. A contraction is **sufficient for convergence**, which is [[Implicit Learners]]'s recorded caveat —
*"this only works if the iteration converges, and unconstrained DEQs need not"* — and the
test suite makes it concrete: a linear cell with $\rho(W) = 1.6$ has a perfectly good unique
fixed point that Picard cannot reach and Broyden reaches in a handful of steps.

Broyden is initialised with $H_0 = I$, which makes its first step

$$u_1 = u_0 - I\cdot F(u_0) = u_0 - (u_0 - g(u_0)) = g(u_0)$$

**exactly a Picard step**, with everything after it the correction Picard never makes.
Asserted in the tests when `init_scale=1`; other scales give damped Picard.
This equality says nothing about later convergence.

All three root algorithms are derivative-free, so inference has no hard AD dependency.
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
$r(z)=z^3$ has a unique root and zero derivative there). The default `DenseIFT()` solve is dense: a reverse
backend VJP in the ChainRules rule can avoid parameter-Jacobian storage, but the
Enzyme/Mooncake rules use a dense ForwardDiff VJP. The state Jacobian and its
factorisation cost quadratic storage and cubic factorisation work. An opt-in `KrylovIFT()` path is described below. These primitive rules are first order. `solve_input(...; ad)` now reuses them in
the other polarity. A dedicated dense second-order adjoint is described below;
GPU and additional backends are untested.

The EnzymeCore extension provides native `EnzymeRules.forward`, `augmented_primal`
and `reverse` rules on `_implicit_state`. DI's default `AutoEnzyme()` Jacobian uses
batched forward mode; its scalar gradient uses reverse mode. The Mooncake extension
registers a native reverse `rrule!!` with `@is_primitive`; it accumulates the joint
vector cotangent in Mooncake's mutable forward data. Neither uses `@from_rrule`.
Both extensions also require ForwardDiff to be loaded. In dense mode they use `AutoForwardDiff()`
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

### 4.2 Dense and limited-memory Broyden

The default `BroydenSolver` still stores an $O(n^2)$ inverse. Opt-in
`LimitedBroydenSolver(memory=m)` stores transformed secant pairs $(a_i,b_i)$:

$$
H_k=\alpha I+\sum_i a_i b_i^\top,\qquad
 a_k=\frac{s_k-H_k y_k}{s_k^\top H_k y_k},\qquad b_k=H_k^\top s_k.
$$

Here $s_k=u_{k+1}-u_k$, $y_k=F(u_{k+1})-F(u_k)$ and $\alpha$ is `init_scale`.
This is the same good inverse Broyden update as the dense solver. Products with
$H_k$ and $H_k^\top$ cost $O(mn)$; no dense inverse is allocated. The absolute
`1e-14` denominator guard matches the default solver. At capacity the oldest
rank-one correction is removed, without recomputing the retained corrections.
These are transformed secant pairs, not a multisecant interpolation of the last
raw $(s,y)$ pairs: eviction need not preserve the latest secant equation and can
slow convergence. There is no line search or global convergence guarantee.

Without eviction, iterates agree with dense Broyden up to roundoff; arithmetic
order differs, so bitwise equality is not claimed. The first step is
$u_1=u_0-\alpha F(u_0)$, exactly damped Picard (ordinary Picard for $\alpha=1$).
`SolveReport` and nonconvergence behavior match the existing solvers. Tests cover
full iteration traces, finite memory, degenerate updates and nonfinite residuals.

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
violates this functional contract. `solve_input(...; ad)` uses this same contract;
without `ad`, it retains legacy `Ref` state threading to preserve the existing API.
For stateful cells these are deliberately different equations, so the legacy
trajectory is not a derivative oracle for the AD solve.

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

## Matrix-free first-order IFT (LP-002 follow-up)

```julia
using Krylov, DifferentiationInterface
f = DEQFactor(cell, (x=200, z=200); solver=PicardSolver(tol=1e-12),
              ift=KrylovIFT(rtol=1e-10, atol=1e-12, itmax=400, memory=20))
z, report, stnew = solve_state(f, x, ps, st; ad)
```

The primal root solver and its convergence policy are unchanged. The IFT option
selects only the linear solve inside the AD rules. For $r(z,u)=0$, forward rules
solve $D_z r\,\delta z=-D_u r\,\delta u$ using residual JVPs. Reverse rules solve
$(D_z r)^\top v=\bar z$ with residual VJPs and return $-(D_u r)^\top v$.
ChainRules uses the supplied reverse backend; the native Enzyme and Mooncake
rules use their own reverse backend in Krylov mode. Enzyme's inner VJPs enable
runtime activity and a constant function annotation to handle mixed constant/active
cell inputs; users can still select plain `AutoEnzyme()` for the outer DEQ rule. ForwardDiff remains the inner
JVP backend for the Enzyme forward rule. Dense mode retains its existing
ForwardDiff linearisation for native Enzyme/Mooncake rules.

Krylov.jl is a weak dependency instead of the broader LinearSolve interface:
GMRES needs only a typed `size`/`eltype`/`mul!` operator, so no operator package or
matrix materialisation is needed. The operator has no indexing or conversion to a
matrix. Prepared DI products are reused across GMRES iterations. Restarted GMRES
stores `O(memory)` basis vectors (plus its small Hessenberg workspace).
Preconditioning is opt-in; see below. `itmax=0` means Krylov's default, twice
the state dimension. Nonconvergence throws `ArgumentError`, including when the
explicit final residual exceeds `atol + rtol*norm(rhs)` despite an optimistic
recursive estimate. Zero right-hand sides still undergo the residual check.

`test/lp002_followups.jl` checks all four existing backends plus explicit Enzyme
forward/reverse modes on the two-dimensional linear oracle, including forced
nonconvergence. A 200-dimensional random contractive cell checks scalar gradients
against $(I-W)^{-\top}z^*$ and overrides `_root_jacobian` to throw if called.
This guards the state-Jacobian entry point independently of AD chunk size.
`deq_sensitivity` and `ift_sensitivity` remain dense reference APIs; selecting
Krylov on the factor changes differentiable `solve_state`, not those APIs.
Select Picard or LimitedBroydenSolver for a primal without a dense inverse.
GPU and complex parameters remain untested; second order has the dedicated scope below.


## Solver numerics follow-up (2026-10-27)

### Preconditioning and measurement

`KrylovIFT(precond=...)` accepts a left operator or a factory returning
`(; left=M, right=N)`; omitted sides are identity. A factory receives the current
`operator`, `transposed`, `factor`, `solved` (channel name), `joint`, `root`, `template`, `offsets`, and
incoming `state`. The joint vector is observed channel then parameters, in either
polarity. Operators must implement Krylov's `mul!` protocol, or `ldiv!` when
`ldiv=true`. With the default `ldiv=false`, supply approximate inverses: GMRES
operates on $MAN$ and recovers the unknown through $N$. Factories are responsible
for orientation, including transposing their preconditioners for reverse solves.
Preconditioner construction is bookkeeping, without parameter derivatives.

`JacobiIFT()` supplies an inverse diagonal on the left. It probes the current
operator with each coordinate basis vector: $n$ JVPs in forward mode or $n$ VJPs
in reverse mode, $O(n)$ storage, no matrix. This setup can be expensive and is
not a stochastic diagonal estimator. Nonfinite or small diagonal entries throw;
`ldiv=true` with this inverse preconditioner also throws. No forward Broyden
history is reused: doing so would require retaining solver workspace in the
primitive and defining its orientation for both polarities.

`on_solve = report -> ...` receives iterations, actual residual and solved status
for each RHS. Convergence still requires the original unpreconditioned residual
bound, not just Krylov's recursive estimate. A preconditioner can change the
recursive stopping scale; the explicit final check can therefore reject an
apparently converged preconditioned solve. There is no automatic refinement.
The defaults remain `DenseIFT()` and `BroydenSolver()`.

### Reverse-input first-order IFT

LP-002's history and [[deq]] describe “different variable held fixed.” We interpret
“reverse-input solve rules” as differentiating `solve_input` with observed $z$,
not as the input cotangent of `solve_state` (which was already tested).
For $r(x,z,\theta)=0$ with square nonsingular $r_x$:

$$
\delta x=-r_x^{-1}(r_z\delta z+r_\theta\delta\theta),\qquad
r_x^\top\lambda=\bar x,\qquad
(\bar z,\bar\theta)=-(r_z,r_\theta)^\top\lambda.
$$

An internal problem adapter swaps the residual coordinate solved by the existing
primitive. Thus ForwardDiff, ChainRules/Zygote, native Enzyme forward/reverse and
Mooncake use the same rules, dense or Krylov. No duplicated AD rules trace the
root iterations. The initial guess has zero derivative, including an active
warm start. Nonconverged roots and singular residual Jacobians fail as in
`solve_state`. Square shape alone does not ensure local invertibility.

### Dedicated second-order adjoint

`deq_hvp(loss, f, x, ps, st, direction; wrt=:ps, ad=AutoForwardDiff())`
returns a flat Hessian-vector product for `loss(zstar)`. `:x` selects the input
block; `:joint` selects input followed by flattened parameters and includes mixed
blocks. The loss has no direct input/parameter dependence. `DenseIFT()` and CPU
real floating-point arrays are required. Other backend selections are rejected
with `ArgumentError`. This is an explicit second-order API, not an arbitrary
nested derivative rule for the solve primitive.

Write $u=(x,\theta)$, $J=r_z$, $B=r_u$, $Jt=-Bv$ and
$J^\top\lambda=L_z$. At fixed $\lambda$ compute

$$
c=\nabla^2_{(z,u)}(\lambda^\top r)\begin{pmatrix}t\\v\end{pmatrix},\qquad
J^\top\dot\lambda=L_{zz}t-c_z,\qquad
\nabla_u^2 L(z^*(u))v=-B^\top\dot\lambda-c_u.
$$

ForwardDiff builds the residual Jacobian, the scalar contracted residual Hessian
and the loss Hessian, evaluated at one converged root. Only residuals and loss
are differentiated twice; the root finder is never traced. This reference uses
quadratic storage in state plus input plus parameter dimension and dense solves;
it is unsuitable for large parameter trees. State and warm start are fixed and
carry no tangent. The HVP API itself has no higher-order rule.

ForwardDiff nesting of the solve primitive and ForwardDiff over the other three
solve backends are rejected. The ChainRules pullback also has an explicit nested
reverse guard. Use the dedicated adjoint instead of composing backend tapes;
other native mixed/nested Enzyme/Mooncake tapes are outside the supported API.

`test/solver_numerics.jl` records iteration counts and checks linear full-trace
Broyden agreement, reverse-input closed forms and nonlinear branches, fixed state,
left/right and inverse/solve preconditioner conventions, and HVPs against an
analytic linear solution, scalar second derivatives and central differences of
first-order IFT gradients. Measurements and full verification are recorded in
[[LP-002 The Jacobian side is not implemented]].


## Batch A: retained Broyden history (2026-10-27)

`KrylovIFT(precond=BroydenIFT())` opts into the actual forward
`LimitedBroydenSolver` inverse history. Each solve owns its correction vectors;
there is no global cache and no repeat solve. A private report carries history
across the AD primitive; the public return remains `SolveReport`. Every backend
passes that history into linear preparation. Repeated pullbacks only read it.

For `H = scale*I + sum(aᵢ*bᵢ')`, the left inverse preconditioner applies H in
forward solves and H' in transposed solves (swap the two vector lists). This
also handles reverse polarity: history approximates the inverse of the residual
Jacobian in the actually solved coordinate, including its sign. It requires
`LimitedBroydenSolver` and `ldiv=false`; unsupported choices throw. A root warm
start may yield no secants, leaving just the initial scaled identity. Eviction
or poor secant information can reduce quality; no speedup is guaranteed. The
unpreconditioned final residual check remains mandatory. If the retained inverse
causes recursive residual convergence before the original tolerance is met, up
to two residual-correction solves refine the result. They share any explicit
`itmax` budget; `on_solve.iterations` includes every correction iteration. Failure
after refinement still throws. This refinement is opt-in through BroydenIFT and
leaves other preconditioners/defaults unchanged.

`test/lp002_batch_a.jl` measures GMRES counts on a rotated, nonsymmetric,
ill-conditioned linear problem, with no preconditioner, Jacobi and retained
Broyden. It checks gradients against DenseIFT on four backends, explicitly
checks H/H' application, and checks both solve polarities. Counts exclude
forward root work and Jacobi setup; they do not establish a wall-time speedup.

### Outer Enzyme closure boundary

`examples/lp002_enzyme_limits.jl` reproduces `EnzymeMutabilityException` for a
DI outer closure capturing arrays, then succeeds with
`AutoEnzyme(function_annotation=Enzyme.Const)`. The implicit rule runs only after
Enzyme has selected the outer function's activity. It cannot retroactively
annotate that closure. Automatically changing arbitrary DI calls would require
intercepting methods owned by DI/Enzyme, and treating every captured array as
constant could discard intentionally active closure data. No such global rule
is installed. Library-owned derivatives already put their callable/context in
`DI.Constant`; callers of arbitrary outer DI functions must choose annotation
or explicit constant contexts themselves. This remains an upstream/API boundary,
with the working explicit annotation tested in `test/lp002_batch_a.jl`.


## Device arrays: LP-002 batch B (2026-10-27)

The optional GPUArraysCore extension keeps primal vectors, dense Broyden's inverse,
and limited Broyden's secant vectors on the input device. Its scales/damping are
converted to the vector element type. CPU allocations/scales retain their previous
behavior, including their Float64 promotion. No root algorithm or denominator guard
changes. A missing device initial guess is allocated with `similar(observed, n)`;
CPU defaults remain Float64 zeros. Retained history is owned by the same solve.

Jacobi setup forms basis vectors by broadcast and reads diagonal slices without
host scalar indexing. GMRES tolerances are converted to its RHS real element type:
the previous Float64 keywords caused a type error with Float32 RHSs. User left/right
inverse operators, left `ldiv=true`, Jacobi and retained Broyden preconditioning are
checked on nonsymmetric two-dimensional residuals. The original residual check and
convergence failures remain in force. Very tight default tolerances need loosening
for Float32; the API does not silently change a user's convergence criteria.

`test/device_arrays.jl` uses JLArrays with `allowscalar(false)`, Float32/Float64,
closed-form linear roots and IFT products. Both solve polarities are covered with
Zygote and ForwardDiff. ForwardDiff dual reconstruction broadcasts over whole
columns instead of scalar-indexing each root component. Core's device ForwardDiff
entry points assemble directional DI pushforwards, avoiding native vector seeding.
This costs one direction per active coordinate and is a small-problem reference,
not a claim of scalable forward parameter differentiation.

JLArrays has no device LU on the tested stack. DenseIFT attempts the backend
factorisation and reports an ArgumentError pointing to KrylovIFT on failure; it
never copies a Jacobian to the CPU. DenseIFT success on hardware GPUs is untested.
Enzyme/Mooncake device solve/parameter entry points are explicitly unsupported.
Second-order adjoints and finite-difference reference derivatives require CPU
arrays. No CUDA/ROCm execution or performance result is claimed.


### Typed retained history found during batch B verification

A full CPU regression run exposed an abstract inferred return type when retained
Broyden history crossed the native Enzyme custom rule. Reading the history from
`Ref{Any}` could leave a UnionAll for the report/preconditioner. Enzyme rejected
that ABI contract. Explicit rule-return annotations repaired reverse mode but
batched forward mode triggered an Enzyme LLVM assertion; those annotations were
removed. The solver now returns the same correction vectors as a typed third
result on its private retained-history path. Every exit uses the same result
helper. The public solve still returns `(root, SolveReport)`, including the
existing optional history-holder keyword; residual evaluations, update arithmetic
and iteration counts are unchanged. No extra root or residual evaluation occurs.

The existing non-diagonal history oracles in `test/lp002_batch_a.jl` check reverse
and batched forward derivatives, both orientations, eviction and warm starts.
The focused reproduction also uses explicit bounds checking. This adds no
Enzyme device or nested-AD rule.

## Finite-difference steps and residual norms (2026-10-29, LP-019 items 4 and 10)

`fd_jacobian` added $h = \varepsilon\max(1, |u_j|)$ to $u_j$ in the element type and divided by
the unrounded $h$. In Float32 the probe $1 + 10^{-7}$ rounds to $1 + 2^{-23}$, so the identity's
Jacobian came out as 1.1920929; in Float16 the probe vanished and the Jacobian was 0. Even in
Float64 the identity gave $1 + 5.8\cdot10^{-10}$. The quotient now uses the increment the rounded
probe actually makes (`_fd_step`, shared with `flow.jl`'s `_divergence`), and the default
relative step is $\max(10^{-7}, \sqrt{\varepsilon_T})$, so Float64 keeps $10^{-7}$.

`_norm` was `sqrt(sum(abs2, v))`, which underflows: a constant residual `[1e-200]` with
`tol = 1e-210` was reported converged at iteration 0 by Picard, Broyden and limited Broyden,
although there is no root. It is now `LinearAlgebra.norm`, which rescales. Tested in
`test/lp019.jl`.

After review (2026-10-30): the default step uses `eps` of the element type when it is defined,
which for a ForwardDiff dual is its primal's (Float16 duals had kept $10^{-7}$ and a zero
increment); a step that does not change $x$ throws. `_norm` returns the norm in the residual's
own precision: converting to Float64 underflowed a BigFloat residual of $10^{-400}$.
