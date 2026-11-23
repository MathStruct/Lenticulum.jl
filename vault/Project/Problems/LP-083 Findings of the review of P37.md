#problem

> **Status: closed (2026-11-23).** P37 (MINRES in ImplicitSolvers, used by VariationalDiffusion's
> `KrylovSolve` for an energy's symmetric free block; Codex) and what its review found. One point of
> the design was wrong and was corrected before the review (the symmetry predicate); Mistral's probes
> found no defect of the change and one limit it shares with GMRES, recorded as `@test_broken` and
> row P38. Related: [[Shared Solver Layer]], [[LP-082 Findings of the reviews of P14 W5|LP-082]],
> [[Feature Backlog]] (P37, P38).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; brief and answer in
> `../Lenticulum.jl-Review-2026-11-23/p37-minres/` (`IMPL-p37-minres.md`; the brief is
> `~/.cache/lenticulum/codex-queue/p37-brief.md`), its scripts in `scratch/p37/` of `../Lenticulum.jl-codex`).
> Mistral Large (`vibe --agent review`, \$0, ten probes, finished answer file:
> `../Lenticulum.jl-Review-2026-11-23/p37-minres/ANSWER-mistral.md`, probes in `scratch/p37r/` of
> `../Lenticulum.jl-mistral`). code: `lib/ImplicitSolvers.jl/src/linsolve.jl` (`MINRES`, `_minres_cycle`),
> `lib/VariationalDiffusion.jl/src/krylov.jl` (`_symmetric_block`, `_krylov_solver`), `implicit.jl`
> (`_newton_system`, `_adjoint_solve`). tests: `lib/ImplicitSolvers.jl/test/linsolve.jl`,
> `lib/ImplicitSolvers.jl/test/krylov_oracle.jl`, `lib/ImplicitSolvers.jl/test/device.jl`, `lib/VariationalDiffusion.jl/test/p37_minres.jl`.

## The design point Codex corrected

The brief restricted MINRES to an energy whose free coordinates share one noise time at every node,
and called `_free_operators`' one-product symmetric part (`S = J` for an energy) a latent defect under
unequal block times, from the remark in `energy_network.md` that the noise Jacobian
$\operatorname{diag}(\sigma)\nabla^2E$ need not be symmetric then (LP-018). Codex's first test of that
"defect" failed the other way: the field's Jacobian was symmetric to $1.5\cdot10^{-19}$. The reason is in
`_node_coefs`: with block levels the field weights coordinate $i$ by $c\,\alpha_i/\sigma_i$, which cancels
the predictor's row scale $\sigma_i$, so each node adds $c\operatorname{diag}(\alpha)H\operatorname{diag}(\alpha)$.
Codex implemented the brief's conservative predicate and reported the finding; the session made the
predicate "an `ImplicitDiffusion` whose predictor is an energy" (`EnergyNetwork` or `GaussianMixtureEps`,
at any levels), put `S` back to one product, and replaced the test by one that checks the symmetry at
unequal levels for the quadratic energy and for the mixture ring
(`test/p37_minres.jl`, "P37 an energy's field is symmetric at unequal block times"). Mistral's `probe4.jl`
looked for a counterexample (two and three blocks, unequal block sizes, six level patterns, noisy and
noise-free nodes, batched or not, clamps) and found none: symmetric to about $10^{-15}$ relative.

## What P37 measured (Codex, the session)

On the code before the change (`scratch/p37/record.jl`, `compare.jl`): every ε-model and `:gmres` case
bitwise equal; `:auto` on the energy MLP within $3\cdot10^{-17}$ (MINRES), on the mixture ring within
$9\cdot10^{-16}$ (the ring counts as an energy). Krylov.jl's `minres` as the oracle: 90 comparisons, the
same iteration counts. `@time using VariationalDiffusion`: median 38.4 → 37.9 ms. The benchmark is in
`krylov.md` §4.

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| — | no defect: MINRES against the dense solve and Krylov.jl's `minres` on definite, indefinite, singular, eigenvector right-hand sides, scaled $10^{\pm8}$, `Float32`, complex functions; honest reports on nearly symmetric and non-Hermitian operators; `nfe` and budgets; JLArrays; duals generic; VariationalDiffusion end to end against `DenseSolve` and `:gmres` (probes 1–10) | — | — |
| F1 | at an exact Lanczos breakdown ($b$ an eigenvector of $A(0)$) the ForwardDiff tangent of $x$ is $[0,0]$, the dense solve's $[0,-1/4]$ (`probe2.jl`, `probe3.jl`) | right; the session's `scratch/p37r/session_f1.jl` shows GMRES gives the same $[0,0]$, so it predates P37 | recorded: `@test_broken` for both solvers in `lib/ImplicitSolvers.jl/test/linsolve.jl`, "Krylov tangents at an exact breakdown"; `linsolve.md` "Known limit"; row P38 |
| O1 | $b$ with zero primal and nonzero tangent: MINRES and GMRES return a zero tangent, `DenseSolve` the right one (`probe10.jl`) | right, the same limit as F1, shared with GMRES | in the `linsolve.md` paragraph and P38; no separate test |
| O2 | `KrylovSolve` against `DenseSolve` at a runaway root ($z \approx -3\cdot10^{6}$, $J_{FF} \approx 3\cdot10^{-20}$): `rtol` is relative to $\lVert b\rVert$ and the adjoint's $\lambda$ differs from the dense one by $2\cdot10^{3}$ while `:converged` (`probe5.jl`, `probe6.jl`) | right as a description; MINRES and GMRES agree bitwise there, so not a defect of P37: the relative tolerance does what it says on a near-singular block, and the dense path refuses such blocks by its condition check | not changed; no row (a condition check for the Krylov path would cost a Lanczos run per pullback) |
| O3 | the brief calls the bound `max(atol, rtol‖b‖)`; the code (D2) uses `atol + rtol‖b‖` | right: the brief was wrong, the code and its tests follow D2 | none |

Not run: the device tests of VariationalDiffusion with a physical GPU, the Reactant suite, the docs
builds after the last edits (Codex ran both builds on its version), the full smoke runner (the new
benchmark's smoke run passed).
