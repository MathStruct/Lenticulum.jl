#open-problem

> **Status: resolved (2026-11-13).** No wrong result, crash or missing check: Mistral's probes
> recomputed every field of `ridge_residual` independently (a 3-D Gaussian, the ring, an MLP with a
> non-symmetric Jacobian, a product relation, Float32, device arrays) and found them equal. Four
> findings were missing tests: mutants that drop the symmetrisation, test positivity on every
> eigenvalue, or never flag an `Inf` distance passed the testsets. Each now has a test that fails
> on its mutant. Related: [[Feature Backlog]] (F3), [[LP-031 Findings of the review of the query report]],
> [[LP-030 Findings of the review of round_trip]].

> Sources: a review of the uncommitted change by Mistral (`vibe --agent review`), run with Julia in
> the helper worktree `Lenticulum.jl-mistral` (2026-11-12), its answer written probe by probe as the
> brief asked (`../Lenticulum.jl-Review-2026-11-12/ridge-from-mistral.md`; probes in
> `scratch/ridge/` of that worktree). The worktree was at `7277fe4f`; I6, the commit between it and
> the reviewed base, changed neither VariationalDiffusion nor LenticulumCore.
> code: `lib/VariationalDiffusion.jl/src/answers.jl` (`ridge_residual`), `src/report.jl`; tests:
> `lib/VariationalDiffusion.jl/test/runtests.jl`, "ridge residual: …" and "query report: …".

**Problem.** `ridge_residual` is the first-order distance from an answer to the ridge of the
smoothed density, from the field and the symmetric part of its Jacobian at the answer
([[implicit]] §5); `query_report` reports it and flags `:off_ridge`.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | the symmetrisation $S = \operatorname{sym} J$ is never exercised: the fixtures' Jacobians are symmetric to $10^{-12}$, and a mutant taking `eigen(Symmetric(J))` passed all 54 assertions | right (the code was correct on an MLP with a non-symmetric Jacobian) | a linear test field $g = (S + K)z$ with $K$ antisymmetric; the expected values come from $S$ | the mutant fails 6 assertions |
| F2 | positivity is checked on the normal curvatures only, and no fixture has a mixed-sign Jacobian: a mutant checking every eigenvalue passed | right | the same field with $S = \operatorname{diag}(-1, 3)$, a saddle: finite at codim 1, `Inf` at codim 2 | the mutant fails 1 |
| F3 | `:off_ridge` on an `Inf` distance is reached (the ring's centre given as the answer) but never asserted | right | asserted, also with `ridge_tol = 1e9` | a mutant flagging only by tolerance fails 1 |
| F4 | the default codimension of the `ImplicitSolution` method is only exercised with one free and one clamped coordinate | right | a 3-D field with 0, 1, 2 and 3 free coordinates (precisions as `Vector{Any}` too), against the explicit `codim` | — |
| F5 | notes stale: `report.md` still said "no ridge residual", no note on `ridge_residual`, Feature Backlog rows | true of the diff it saw; the notes were written after the diff was sent (`implicit.md` §5, `report.md`, the backlog) | — | — |

Observations, not defects: `gap = -Inf` at `codim = 0` (a convention); at tied eigenvalues the
distance depends on the eigenbasis (documented: a small `gap` means the distance is not determined);
the returned fields mix element types for a Float32 state (`distance` is promoted).

**Leaning / resolution.** F1–F4 closed by tests that fail on their mutants. The review did not run
the whole suite nor `examples/calibration.jl` (the maintainer's session ran both).
