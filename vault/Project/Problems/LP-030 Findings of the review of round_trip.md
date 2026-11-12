#open-problem

> **Status: resolved (2026-11-12).** One defect: `round_trip` threw on a precision vector of element
> type `Any`, which `implicit_infer` accepts; fixed, with a regression test. The other finding was a
> list of untested edge cases, all of which Mistral ran and found working; three of them are now
> tests. Related: [[Feature Backlog]] (F3), [[LP-029 Findings of the review of P17 and P18]].

> Sources: a review of the uncommitted `round_trip` (F3) by Mistral (`vibe --agent review`), run with
> Julia in the helper worktree `Lenticulum.jl-mistral` (2026-11-12). Its answer is in
> `../Lenticulum.jl-Review-2026-11-12/REVIEW-f3-from-mistral.md` (the first run failed on a server
> error of the API before its first turn); its scripts are `scratch/f3/any_rho.jl`,
> `scratch/f3/probe.jl`, `scratch/f3/probe2.jl` and `scratch/f3/probe3.jl` in that worktree.
> code: `lib/VariationalDiffusion.jl/src/answers.jl` (`round_trip`); test:
> `lib/VariationalDiffusion.jl/test/runtests.jl`, "round trip: answers on the relation stay,
> projections walk back".

**Problem.** `round_trip(m, sol, z₀, ρ, ps, st)` reverses a query (frees the hard clamps, clamps the
rest at the answer, solves back from the answer) and returns how far the clamps move, `Inf` when the
reversed solve is not a converged stable root. It must accept every query `implicit_infer` accepts.

Mistral ran the testset (10 of 10 at the time) and the whole VariationalDiffusion suite (899 of 899,
Aqua 11 of 11). It probed element types (Float16, Float32, Float64, Int, Real, Number, a union, `Any`),
views, an all-clamped query, soft evidence, a `ProductRelation`, a one-dimensional relation, two hard
coordinates, a non-energy MLP relation on the Newton path, and `kwargs` passed through.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | `round_trip` built the reversed precision with `zero(eltype(ρ))`, a `MethodError` for `ρ::Vector{Any}` and views of one | right: reproduced (the extended testset failed with that `MethodError` before the fix) | the reversed precision takes the float type of the answer's state | `Any[Inf, 0.0]`, a view and `Float32` in the testset |
| F2 | The testset had no case for an all-clamped query, soft evidence, a `ProductRelation`, Float32, integer `ρ`, views | right as a gap; Mistral found the code working on each | soft evidence, `Any`, a view and Float32 added. Checking soft evidence showed that the evidence pulls the answer off the relation, and the round trip measures that pull (0.011, 0.12, 0.23 for precisions 0.1, 0.5, 2 at $x = 0.6$); the reversed answer lies on the mode circle. This is now asserted and stated in the docstring | in the testset |
| F3 | (checked, no defect) the `Inf` case of the testset does reach the stability branch: the reversed solve from the centre converges at iteration 0 with `stable = false` | — | nothing | — |

**Leaning / resolution.** F1 fixed, F2 partly turned into tests. Not tested in the suite: the
`ProductRelation` and the Krylov path through `round_trip` (Mistral ran them in its scratch scripts
only), and device arrays (no device-specific code in `round_trip`).
