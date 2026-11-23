#design #implementation

> The design of P14: one package, `lib/ImplicitSolvers.jl`, holding what `ImplicitLayers`,
> `VariationalDiffusion` and MyceliumLearning's graph adjoint each implement for themselves today: one
> tolerance rule, one solver report with a primal residual, the linear solves (dense and GMRES), the
> fixed-point and quasi-Newton root solvers, and the linear solve of the implicit-function adjoint.
> Element types come from the problem, never from a `Float64` field. Newton and trust-region steps
> stay in `VariationalDiffusion` for now (phase 2). Status: design, 2026-11-22; W1 implemented the same day.

> Sources: original to this vault (design). Walker & Ni, *Anderson acceleration for fixed-point iterations*, SIAM J. Numer. Anal. 49(4), 2011 (Anderson mixing; `walker2011anderson`); Saad & Schultz, *GMRES*, SIAM J. Sci. Stat. Comput. 7(3), 1986 (restarted GMRES; `saad1986gmres`); Bai, Kolter & Koltun, *Deep Equilibrium Models*, NeurIPS 2019 (the adjoint by the implicit function theorem; `bai2019deq`); code: `lib/ImplicitLayers.jl/src/solve.jl`, `lib/VariationalDiffusion.jl/src/implicit.jl`, `krylov.jl`, `proxdm.jl`, `many.jl`, `lib/MyceliumLearning.jl/src/graph_adjoint.jl`, `lib/Mycelium.jl/src/passing.jl`
>
> Bibliography: [[Bibliography#^walker2011anderson|Walker & Ni 2011]] · [[Bibliography#^saad1986gmres|Saad & Schultz 1986]] · [[Bibliography#^bai2019deq|Bai et al. 2019]]

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
the extension), and the stopping decision is made on it, so tangents never change the iteration count
(Mycelium's I23 rule, [[LP-053 Findings of the review of I23]]).
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

## Open questions

- Whether `Anderson` belongs in the package as a root solver (Mycelium's `AndersonSchedule` mixes
  messages, not a vector; nothing else uses Anderson today). Not in phase 1.
- A device (GPU) path for GMRES: the moved code must not scalar-index (tested with JLArrays and
  `allowscalar(false)`), but no device benchmark is planned.
