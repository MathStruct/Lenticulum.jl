#design #implementation

> The design of P14: one package, `lib/ImplicitSolvers.jl`, holding what `ImplicitLayers`,
> `VariationalDiffusion` and MyceliumLearning's graph adjoint each implement for themselves today: one
> tolerance rule, one solver report with a primal residual, the linear solves (dense and GMRES), the
> fixed-point and quasi-Newton root solvers, and the linear solve of the implicit-function adjoint.
> Element types come from the problem, never from a `Float64` field. Newton and trust-region steps
> stay in `VariationalDiffusion` for now (phase 2). Status: design, 2026-11-22; W1 implemented the same day.

> Sources: original to this vault (design). Walker & Ni, *Anderson acceleration for fixed-point iterations*, SIAM J. Numer. Anal. 49(4), 2011 (Anderson mixing; `walker2011anderson`); Saad & Schultz, *GMRES*, SIAM J. Sci. Stat. Comput. 7(3), 1986 (restarted GMRES; `saad1986gmres`); Bai, Kolter & Koltun, *Deep Equilibrium Models*, NeurIPS 2019 (the adjoint by the implicit function theorem; `bai2019deq`); Montoison & Orban, *Krylov.jl: A Julia basket of hand-picked Krylov methods*, JOSS 8(89), 2023 (the alternative; `montoison2023krylov`); code: `lib/ImplicitLayers.jl/src/solve.jl`, `lib/VariationalDiffusion.jl/src/implicit.jl`, `krylov.jl`, `proxdm.jl`, `many.jl`, `lib/MyceliumLearning.jl/src/graph_adjoint.jl`, `lib/Mycelium.jl/src/passing.jl`, `bench/krylov_allocations.jl`
>
> Bibliography: [[Bibliography#^walker2011anderson|Walker & Ni 2011]] · [[Bibliography#^saad1986gmres|Saad & Schultz 1986]] · [[Bibliography#^bai2019deq|Bai et al. 2019]] · [[Bibliography#^montoison2023krylov|Montoison & Orban 2023]]

> Theory (CT-ML wiki): —

## The problem

An inventory of the code at `af2e6311` (2026-11-22; file:line as of that commit):

| what | where | convention |
|---|---|---|
| Picard, Broyden, limited Broyden | `ImplicitLayers/src/solve.jl` 73, 100, 117 | absolute, $\lVert F\rVert \le$ `tol`, `tol::Float64 = 1e-10`; `1e-14` guards (161, 229) not cast |
| `SolveReport` | `solve.jl` 36 | `residual::Float64` |
| dense and Krylov IFT | `solve.jl` 302–375, `ext/ImplicitLayersKrylovExt.jl` 76, 91 | `Matrix{Float64}` (304); Krylov.jl GMRES, `rtol = 1e-10`, `atol = 1e-12` as `Float64` fields |
| Newton with Armijo, trust region | `VariationalDiffusion/src/implicit.jl` 436, 490 | relative, $\lVert r\rVert \le \tau(1+\lVert z\rVert)$, $\tau = \max(10^{-9}, 30\epsilon)$ (`default_tolerance`, 371) |
| the same, batched | `many.jl` 126, 184 | a copy of the two loops |
| damped fixed point on a prox map | `proxdm.jl` 229 | relative, `tol = 1e-9` |
| `ImplicitSolution`, `QueryReport` | `implicit.jl` 347, `report.jl` 39 | `residual::T` |
| restarted GMRES, CG, Lanczos | `krylov.jl` 80, 135, 161 | hand-written; `KrylovSolve.rtol::Float64 = 1e-8` |
| adjoint solves | `implicit.jl` 627 (dense), 643 (GMRES on $J^\top$); `proxdm.jl` 401, 406 | — |
| graph adjoint $(I - D^\top)\lambda = r$ | `MyceliumLearning/src/graph_adjoint.jl` 124–137, `ext/MyceliumLearningKrylovExt.jl` 7 | `linear_tol = 1e-10`, Krylov.jl GMRES |
| message passing | `Mycelium/src/passing.jl` 872, `ConvergenceReport` 753 | $\text{change} \le \text{tol} + \text{rtol}\cdot s$ per parameter group (P9); primal residual (I23) |
| finite-difference Jacobians | `solve.jl` 254 (step $\max(10^{-7}, \sqrt\epsilon)$), `proxdm.jl` 289 ($h = 10^{-6}$) | — |

Four report types, three GMRES (one hand-written, two through Krylov.jl), two fixed-point loops with
opposite tolerance conventions, two adjoint linear solves of the same form. `ImplicitLayers`' defaults
are absolute and fixed for `Float64`: under `Float32` the default `1e-10` is below rounding and never met;
under dual numbers `SolveReport` cannot hold the residual.

## Decisions

**D1. A package of its own.** `lib/ImplicitSolvers.jl` uses `LinearAlgebra` and has a
`ForwardDiff` weak dependency. `Random` is used by tests. The consumers are
`ImplicitLayers`, `VariationalDiffusion` and `MyceliumLearning`. Mycelium message passing
keeps its own criterion (D2). The maintainer chose a standalone package on 2026-11-22.

**D2. One tolerance rule.** A solver has converged when
$$
\lVert r\rVert \le \text{atol} + \text{rtol}\cdot s ,
$$
with $r$ the solver's residual and $s$ its scale, stated by each solver ($\lVert u\rVert$ for a root
solver, $\lVert b\rVert$ for a linear solve). This is Mycelium's P9 criterion with one group, and
`VariationalDiffusion`'s $\tau(1+\lVert z\rVert)$ is the case $\text{atol} = \text{rtol} = \tau$.
- `Tolerance(; atol = nothing, rtol = nothing)`; the fields hold what was given (any `Real`, or
  `nothing`), and the comparison is made in the residual's primal type.
- A `nothing` resolves from the element type $T$ of the problem:
  $\text{atol} = \text{rtol} = \tau(T) = \max(10^{-9}, 30\,\epsilon(T))$, the rule `VariationalDiffusion`
  measured (`implicit.md` §11): $10^{-9}$ for `Float64`, $3.6\cdot10^{-6}$ for `Float32`. A
  `BigFloat` gets $10^{-9}$ too; a caller that wants more passes it.
- Guards against division by small numbers (the `1e-14` of the Broyden updates) are `sqrt(eps(T))`
  times the scale of the quantity divided, not literals.
- Mycelium's `propagate!` keeps its keywords and per-group scales: it is not a root solver of a vector,
  and its defaults are covered by many tests. It shares the semantics, not the code.

**D3. One report.** `SolverReport{R<:Real}` with `converged::Bool`, `iterations::Int`, `residual::R`,
`nfe::Int` and `reason::Symbol` (`:converged`, `:maxiters`, `:nonfinite`, `:stalled`). `residual` is
always the primal value (`ImplicitSolvers.primal`, the identity, with a method for `ForwardDiff.Dual` in
the extension), and the primal iteration stops on that value (Mycelium's I23 rule,
[[LP-053 Findings of the review of I23]]). After primal convergence GMRES and MINRES can add
tangent corrections and count their work in the same report (P38, `lib/ImplicitSolvers.jl/src/linsolve.md`).
- `ImplicitLayers.SolveReport` becomes `const SolveReport = ImplicitSolvers.SolverReport` (same field
  names, plus `reason`); its `show` moves along.
- `VariationalDiffusion.ImplicitSolution` keeps its fields (`z`, `residual`, `iters`, `converged`,
  `stable`); `residual` becomes the primal value. `QueryReport` is a report about a query, not a solve,
  and stays.
- `Mycelium.ConvergenceReport` stays (it has `exact` and a `reason::String` read by tests).

**D4. Linear solves.** `linsolve(A, b, alg; tol, left, right) -> (x, report)`, with `A` a
matrix or an operator action $v \mapsto Av$. `DenseSolve()` uses LU or Cholesky for
a matrix asserted to be symmetric positive definite. `GMRES(; restart = 40,
maxiters = 200)` is matrix-free and generic in the element type. It comes from
`VariationalDiffusion`'s restarted GMRES. `MINRES(; maxiters = 200)` uses short Lanczos
recurrences for Hermitian systems, including indefinite ones, and checks the true residual. The scale is $s = \lVert b\rVert$. CG,
Lanczos and the log-determinant stay in `VariationalDiffusion`, their only user.
- W5 removes the Krylov.jl extensions of `ImplicitLayers` and `MyceliumLearning`.
  `lib/ImplicitSolvers.jl/test/krylov_oracle.jl` retains the direct test dependency and
  checks agreement on generated systems.
- Why Krylov.jl was dropped, what the in-house solvers lack, and the open choice: "Why not Krylov.jl" below (P40).

**D5. Root solvers.** `solve_root(F, u0, alg; tol = Tolerance(), maxiters) -> (u, report)` for
$F(u) = 0$, $u$ an `AbstractVector`. Algorithms: `Picard(; damping = 1)` ($u \leftarrow u - \alpha F(u)$),
`Broyden(; init_scale = 1)`, `LimitedBroyden(; memory = 20)` (with the history kept for the adjoint
when asked, as `ImplicitLayers`' `BroydenIFT` needs). These are `ImplicitLayers`' solvers moved, with D2
and D3. `ImplicitLayers`' constructor names (`PicardSolver`, `BroydenSolver`, `LimitedBroydenSolver`)
stay as its own names for the moved types. A fixed point $u = T(u)$ is the root of $F(u) = u - T(u)$;
`VariationalDiffusion`'s `prox_infer` uses `Picard` with its damping.
- Newton with Armijo backtracking and the trust region are **not** in phase 1: `implicit_infer` checks
  stability, runs on devices and has a batched copy (`many.jl`); moving them is phase 2 (a row of its
  own), together with removing that copy.

**D6. The adjoint linear solve.** `adjoint_solve(vjp, g, alg; tol, left, right)` solves
$J^*\lambda = g$ given $v \mapsto J^* v$ (`vjp`). The same keywords on
`fixed_point_adjoint` solve $(I - J^*)\lambda = g$ through D4. Each package keeps
the rules that know its parameters and cotangents. The shared solve and tolerance
serve `ImplicitLayers`' `DenseIFT` and `KrylovIFT`, `VariationalDiffusion`'s
`_adjoint_solve` and `_prox_adjoint`, and MyceliumLearning's dense and GMRES graph
adjoints. `JacobiIFT` and `BroydenIFT` stay in `ImplicitLayers`.

**D7. One finite-difference Jacobian.** `fd_jacobian(F, u; step = nothing)`, central differences, the
step $\sqrt[3]{\epsilon(T)}\max(1, \lvert u_i\rvert)$ per coordinate. It replaces both copies; their
users are tests and diagnostics.

## What must not change

- Every existing test of the consumers passes with its present tolerances, except where a deliberate
  change below says otherwise.
- Exported names of `ImplicitLayers` and `VariationalDiffusion` keep working (aliases where the type
  moved).
- Gradients through the rules: the same values (tolerance of the solves) on the existing tests.
- No consumer's load time grows by more than 10 % (`@time using`, measured before and after).

## Deliberate changes

- `ImplicitLayers`' default tolerance: absolute $10^{-10}$ becomes D2's default, $10^{-9}(1 + \lVert u\rVert)$
  for `Float64`. A test that needs the old bound passes `tol` explicitly; each such test is listed in the
  work package's report.
- Reports hold the primal residual (D3).
- `prox_infer` uses D2 to test the fixed-point residual at the returned iterate. The W3 baseline
  comparison measures the change against the old bound. The finite-difference fallback serves
  production prox adjoints and keeps its previous step.
- Preconditioned GMRES decides convergence from the true residual against D2
  with scale $\lVert b\rVert$. It checks that residual after each cycle.
- GMRES accepts complex systems using conjugating inner products and unitary
  Givens rotations.
- MyceliumLearning's `:krylov` is an alias of `:gmres` for existing callers.

## Work packages

Each one is a branch, reviewed by a model that did not write it.

1. **W1, the package.** *Done 2026-11-22* (Codex; GLM's review and its three fixes in [[LP-078 Findings of the reviews of the Mycelium coverage tests and the Anderson and block AD fixes|LP-078]]). Departures: none of D2–D7; a root solve started exactly at the root returns the initial guess's tangent under ForwardDiff, not the implicit derivative (three `@test_broken`; consumers' rules supply the implicit derivative). D2–D7 with their tests, written from this note: the tolerance rule at its edges
   (zero residual, `atol = 0`, `rtol = 0`, both `nothing` for `Float32`, `Float64`, `BigFloat`, duals);
   `linsolve` dense against GMRES on random well-conditioned, nonsymmetric and symmetric positive
   definite systems (sizes 1, 2, 50), as functions and as matrices, `Float32` and `Float64`; GMRES
   reports `:maxiters` on a system it cannot solve in the budget; root solvers on closed-form problems
   (a linear $F$, $u - \cos u$ componentwise, a contraction with a known fixed point), the iteration
   count unchanged under duals with the tangent of the root equal to the implicit-function theorem's;
   the adjoint solve against `(J')\g` and `(I - J')\g`. Aqua. A CI entry and a docs page.
2. **W2, `ImplicitLayers` onto it.** *Done 2026-11-23* (Codex; reviewed by Mistral,
   [[LP-079 Findings of the reviews of P14 W2 and W3|LP-079]]). Root solvers, the report, finite
   differences, CPU dense vector solves and real unpreconditioned GMRES use the shared package; device
   LU, `JacobiIFT` and preconditioned, complex or `ldiv` Krylov solves keep their own paths. No test
   needed an explicit `tol` (every one was already explicit); three assertions changed to the shared
   API's behaviour (LP-079). Its tests need Enzyme ≥ 0.13.221.
3. **W3, `VariationalDiffusion` onto it.** *Done 2026-11-23* (Codex; reviewed by Mistral,
   [[LP-079 Findings of the reviews of P14 W2 and W3|LP-079]]). Shared Picard, GMRES and adjoints
   implement D4–D6, and `ImplicitSolution.residual` is primal. The package suite passes; the eight
   VariationalDiffusion smoke examples printed the same numbers before and after (Codex's run). D7's
   step left all three diagnostic verdicts unchanged; the production fallback (`prox_jacobian`, used
   by the prox adjoints without an AD backend) kept its `1e-6` step. Newly refused: a negative or
   NaN `tol`, a negative `maxiters`, a nonfinite dense adjoint system (LP-079).
4. **W4, MyceliumLearning's graph adjoint onto D6.** *Done 2026-11-23* (Codex).
   The dense reference solve and the new matrix-free `:gmres` use ImplicitSolvers.
   The graph retains its original residual check for all three solvers, including
   the Krylov.jl extension. Across the 18 graph-adjoint cases in `scratch/w4/compare.jl`,
   all 52 dense and 27 Krylov gradients are bitwise equal to their baselines.
   The new GMRES gradients differ from the old dense gradients by at most
   $2.019\cdot10^{-10}$ absolute and $8.304\cdot10^{-10}$ relative in the infinity norm.
5. **W5, Krylov.jl out.** *Done 2026-11-23* (Codex). In
   `lib/ImplicitSolvers.jl/test/krylov_oracle.jl`, both solvers passed the true
   residual and conditioning bounds on 648 generated real and complex systems.
   The one-step budget case reports nonconvergence in both. The two extensions
   are removed. In `scratch/w5/compare.jl`, real unpreconditioned `KrylovIFT`
   results and all 52 recorded graph `:gmres` gradients are bitwise equal to the
   baseline. The preconditioned, `ldiv` and complex IFT results meet the fixture
   tolerances. For the old graph `:krylov` path versus the new alias, the largest
   differences are $3.028\cdot10^{-10}$ absolute and $1.246\cdot10^{-9}$ relative.
   The full comparison and load times are in `IMPL-p14-w5.md`. Device checks use
   JLArrays. The Lenticulum suite and hardware GPU execution are outside this run.

W1 first; W2, W3 and W4 are independent after it; W5 last. Phase 2 (Newton, the trust region and
`many.jl`'s copy) gets its own row when phase 1 is done.

## Phase 2: Newton and the trust region (P39)

Design, 2026-11-23, after phase 1 was done; implemented the same day (Codex, reviewed by Mistral, [[LP-085 Findings of the review of P39|LP-085]]). Deliberate change found by the review: a trial point with a non-finite residual counts as ratio 0 and shrinks the radius (before, the NaN ratio kept the radius and the same step was retried until `maxiters`); no recorded call reaches it. The `problem` named tuple has three optional members beyond D8 (`trial_residual`, `residual_norms`, `round_steps`), which keep the old evaluation order and rounding bitwise; `many.jl`'s trust-region `scale` callback also evaluates the batch Jacobian that `stable` and `model` read, which relies on the driver calling `scale` on all unfinished columns first in each iteration. The inventory at `077e8f5a`:

| what | where |
|---|---|
| Newton with Armijo backtracking where $\operatorname{sym}J_{FF} \succ 0$, a field step otherwise | `VariationalDiffusion/src/implicit.jl` 442 (`implicit_infer`, the loop from 460) |
| the trust region: Moré–Sorensen step from an eigendecomposition, Steihaug's truncated CG, the ratio rule with a noise floor | `implicit.jl` 496 (`_infer_trust_region`), 538–610 (`_tr_model`, `_tr_step`, `_to_boundary`) |
| the same two loops over the columns of a matrix | `many.jl` 126 (`implicit_infer_many`), 186 (`_infer_many_trust_region`) |
| what one iteration needs from the relation | `_newton_system` (620, 626), `_tr_model`, `_stable` (361), `_lambda_min`, `_query_energy` (`answers.jl` 156), `_arith_eps` (380) |

The two loops exist twice because the batched copy evaluates all unfinished queries in one network call
per noise level; the steps, the radii and the acceptance are per query and are the same code written
again. Other Newton iterations in the repository (`proxdm.jl`'s modified Newton, `Lenticulum/src/likelihood.jl`
495, `AlgebraicFactors/src/fit.jl` 36) are not in this phase (open questions).

**D8. One driver, over columns.** ImplicitSolvers gets the two iterations as algorithms over a batch
of queries, a matrix $U$ whose columns are the free coordinates of the queries; one query is a batch of
one column. The relation supplies the evaluations through callbacks on the unfinished columns (`cols`):

- `residual(U, cols) -> R` (the residual of each column, the root to find);
- `scale(U, cols) -> s` (D2's scale per column; `VariationalDiffusion` passes $\lVert z\rVert$ of the whole
  state, clamps included);
- for `NewtonDescent`: `system(U, cols) -> (stable, solve)`, a `Bool` and a function $r \mapsto J^{-1}r$ per
  column;
- for `TrustRegion`: `merit(U, R, cols) -> f` and `model(U, R, cols) -> (g, H)` per column, `H` a
  symmetric matrix (the exact step) or a function $v \mapsto Hv$ (Steihaug); `stable(U, cols)` for the
  report.

`solve_columns(problem, U0, alg; tol::Tolerance, maxiters) -> (U, reports::Vector{SolverReport}, stable::Vector{Bool})`.
A column leaves the batch when it converges or stops; the callbacks see only the unfinished ones, as
`many.jl` does today. The driver is generic in the array type (no scalar indexing; a device matrix stays
on the device, the per-column steps work on what the callbacks return).

**D9. The algorithms.** `NewtonDescent(; step = 0.05, armijo = 1e-4, min_damping = 1e-8)`: today's
iteration, Newton with backtracking on $\lVert r\rVert$ where `stable`, $-\texttt{step}\cdot r$ otherwise.
`TrustRegion(; radius = nothing, cg_maxiters = 200, cg_rtol = 1e-8, precision = nothing)`: today's Algorithm
4.1 with its constants (shrink below ratio $1/4$, grow above $3/4$ at the boundary, accept above
$10^{-4}$), the noise rule (a predicted decrease below $100\,\epsilon_a\max(\lvert f\rvert,1)$ is decided
by the residual) and the radius floor $10\,\epsilon_a(1+s)$, with $\epsilon_a$ = `precision` (the
arithmetic's epsilon, `_arith_eps` in the caller; `eps` of the element type when `nothing`). `radius =
nothing` is $0.1\sqrt{\max(n_F,1)}$. The step routines `_tr_step` (both) and `_to_boundary` move with them.
The relation decides which merit it has (its energy, or $\tfrac12\lVert r\rVert^2$ with the Gauss–Newton
model): that stays in `VariationalDiffusion`'s `model` callback.

**D10. D2 and D3 for these solves.** `VariationalDiffusion`'s `tol::Real` keyword stays and becomes
`Tolerance(atol = tol, rtol = tol)` with the scale above, so $\lVert r\rVert \le \tau(1+\lVert z\rVert)$ is
computed as $\tau + \tau\lVert z\rVert$ (a deliberate change at the last bit). `ImplicitSolution` keeps
its fields, filled from the report (`iters = iterations`) and `stable`; continuation stays in
`VariationalDiffusion`, around the driver.

**D11. What stays in `VariationalDiffusion`.** The field, its Jacobians and operators, the energies, the
merit choice, `_stable` and `_lambda_min`, `default_tolerance`, continuation, `implicit_classify`, the
pullback. `implicit_infer` and `implicit_infer_many` keep their signatures and results; they build the
callbacks (one column, with `prior_jacobian` or the matrix-free operators; many columns, with
`_jacobian_many`, `_field_many`, `_energy_many`) and call the driver. The two copies in `many.jl` go.

**Must not change (phase 2).** Each test of `VariationalDiffusion` passes with its tolerances; on every
call of `implicit_infer` and `implicit_infer_many` in its suite and examples, $z$, `iters`, `converged`
and `stable` equal the values before the change bitwise, except where the rounding of D10 flips a
comparison (each such case listed with its margin). The trust-region and Newton steps of one query and
of the same query in a batch keep the relation of today (they use different Jacobian routes and are not
bitwise equal; the tests that compare them stay as they are). Load time of `VariationalDiffusion` within
10 %; `bench/minres_energy.jl` and `examples/gpu_energy.jl` (smoke) not slower beyond noise.

**Work packages.**

6. **W6, the driver in ImplicitSolvers.** D8–D9 with tests written from this note: a strictly convex
   quadratic (one Newton step; the trust region accepts every step and ends at the minimum); a merit
   with a saddle and a minimum, started near the saddle (the trust region ends at the minimum;
   Newton-descent takes field steps while $\operatorname{sym}J$ is indefinite); the hard case of the exact step ($g \perp$ the lowest eigenvector); Steihaug against the exact
   step inside the ball; Rosenbrock's gradient as a root (convergence from $(-1.2, 1)$); a residual with a
   minimum that is not a root under the Gauss–Newton merit (stops, not converged); a batch of mixed
   columns gives, column by column, bitwise what each column gives alone; `maxiters = 0`; an empty batch;
   a non-finite residual (`:nonfinite`); `Float32` with the noise rule; a JLArrays batch with scalar
   indexing disallowed. Aqua, the docs page, `roots.md` or a new note beside the source.
7. **W7, `VariationalDiffusion` onto it.** D10–D11. Before the change, record $z$, `iters`, `converged`,
   `stable` of every `implicit_infer` and `implicit_infer_many` call in the suite and the examples (a
   script under the worktree's `scratch/`), compare after; load times; the benchmark and the smoke
   examples. `implicit.md` §7 and §11 say where the loops live now. VD's tests of `_tr_step` (runtests.jl
   about 1924) move to ImplicitSolvers or call the moved function.

W6 and W7 go on one branch (two commits): the interface is only right if it fits its one consumer, and
the review sees both.

## Why not Krylov.jl (record, 2026-11-24)

Written after the fact, at the maintainer's request: P14 replaced Krylov.jl without anyone recording why
or weighing the two options, and the result has gaps that Krylov.jl does not have. Open as P40.

**The reason given.** The W5 brief and D4 give only consolidation: "one solver, one tolerance rule, one
report". At the time Krylov.jl was used only by two optional extensions (`KrylovIFT` with a
preconditioner, `ldiv` or a complex right-hand side; MyceliumLearning's `:krylov`), and
`VariationalDiffusion` already had its own restarted GMRES. W5 extended that GMRES and removed the
extensions. Nobody checked whether Krylov.jl, wrapped, could have met D2 and D3. P37 then wrote MINRES
in-house after measuring with Krylov.jl's `minres` as a stand-in (LP-083).

**What the in-house solvers have.** The four points below are what they do, not reasons anyone gave
before W5:
- convergence on the true residual. In the 2×2 fixtures of `lib/ImplicitSolvers.jl/test/krylov_oracle.jl`
  with a nonuniform left Jacobi preconditioner, Krylov.jl stopped on its preconditioned estimate above
  the D2 bound (LP-082);
- the D2 rule, with defaults from the problem's element type, in place of `Float64` fields;
- under ForwardDiff, stopping on primal values and correcting tangents afterwards (P38, LP-084),
  including complex dual numbers (LP-082). The tangent correction runs after a converged solve and
  calls `linsolve` again. It does not depend on which Krylov code produced the primal solution;
- a `SolverReport` with `nfe`.

Whether Krylov.jl's callback and its handling of dual element types could give the same was not
checked.

**What they lack** (the maintainer's review, 2026-11-24, checked against `linsolve.jl` at `ce4b0282`):
- *Allocation-free use.* `linsolve` has no in-place form and no workspace. On each call, GMRES
  allocates `x0`, `r` and the product `ax`. On each Arnoldi step it allocates the operator's result,
  its `copy` and the normalised basis vector, pushed onto a growing `Vector`. On each cycle it
  allocates `H`, `c`, `s` and `g`, and on each restart `xnew`. The operator contract is
  $v \mapsto Av$ returning a new vector, not `mul!(y, A, v)`. Measured with `bench/krylov_allocations.jl`
  (Julia 1.13.1, one thread, dense $A = I + \text{randn}/(4\sqrt n)$, restart 40, tolerance $10^{-10}$,
  17 iterations): the shared GMRES allocates 112 kB at $n = 200$ and 948 kB at $n = 2000$, about 70 and
  59 vectors. Krylov.jl's `gmres!` with a `GmresWorkspace` allocates 0 bytes. Its time is 0.057 against
  0.073 ms at $n = 200$, and 1.81 against 2.14 ms at $n = 2000$. Time is close at these sizes; the
  allocations matter for repeated solves (every implicit adjoint, every Newton step) and on devices.
- *Types for dispatch.* `GMRES` and `MINRES` hold only `Int` fields. Nothing is parametrised by the
  element type or the storage type, unlike a Krylov.jl workspace (`{T, FC, S}`). The operator is
  told apart only as `AbstractMatrix` versus anything else, and genericity is by duck typing on
  broadcasts.
- *Array types and GPUs.* The device tests (`lib/ImplicitSolvers.jl/test/device.jl`) are JLArrays on
  2×2 systems, which catches scalar indexing and nothing else. No CUDA or other hardware GPU run, no
  device size, and no other array types (views, `ComponentArray`s, distributed arrays) are tested.
  GMRES keeps `H` and `g` on the host and calls `dot` twice per basis vector per step, one device
  synchronisation each; this was not measured. The open question below (no device benchmark planned)
  is still open.
- *Preconditioners.* GMRES takes `left` and `right` only as functions $v \mapsto Mv$. There is no
  `ldiv!` object protocol, and no in-place form. The package provides no preconditioner itself:
  Jacobi appears only in tests. MINRES takes none. CG and Lanczos are a third copy in
  `lib/VariationalDiffusion.jl/src/krylov.jl` (`_cg`, `_lanczos`), with their own `rtol`.

Krylov.jl provides the missing pieces according to its documentation (Montoison and Orban 2023):
in-place methods on preallocated workspaces, `mul!` operators, the preconditioners `M` and `N`
including `ldiv!` objects, and GPU arrays. None of these was tried here.

**The open choice (P40).**
(a) Krylov.jl as the backend of `linsolve` for GMRES, MINRES and CG. `Tolerance`,
`SolverReport`, a final true-residual check and the tangent correction remain a wrapper. The oracle test
then tests the wrapper against `DenseSolve`.
(b) Keep the in-house solvers and add what they lack: workspaces, `mul!` operators, `ldiv!`
preconditioners, and a hardware GPU test and benchmark.

Before (a), check three things: Krylov.jl on `ForwardDiff.Dual` and complex `Dual` element types, on
JLArrays with `allowscalar(false)`, and whether its callback can stop on the true residual under
left preconditioning. (b) is the larger amount of code to own.

The maintainer's decision (2026-11-24): Krylov.jl comes back as an optional package extension of
ImplicitSolvers (P41), beside the in-house solvers, which stay the default. P40 now decides only
whether the default changes, on what running both backends on the same cases shows.

P41 is done (2026-11-24, `KrylovJL`, [[LP-086 Findings of the review of P41|LP-086]]). What it showed for
P40: both backends meet D2 on the oracle families; Krylov.jl's GMRES with a left preconditioner does stop on
its estimate above the bound (LP-082's case), which the wrapper's true-residual check catches; Krylov.jl
0.10.12 refuses ForwardDiff dual element types, so the backend is not differentiable by ForwardDiff until
P42; and a workspace built per call allocates 781 kB per solve at $n = 2000$ against 948 kB in-house
(`bench/krylov_allocations.md`), so the allocation-free use waits for Q11's `init`/`solve!`.

## Open questions

- Whether `Anderson` belongs in the package as a root solver (Mycelium's `AndersonSchedule` mixes
  messages, not a vector; nothing else uses Anderson today). Not in phase 1.
- A device (GPU) path for GMRES: the moved code must not scalar-index (tested with JLArrays and
  `allowscalar(false)`), but no device benchmark is planned.
