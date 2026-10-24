#open-problem

> **Status: closed (2026-10-25; first closed 2026-10-23 on `review-argon`, but that fix skipped structured matrices).** Related: [[energy]] §"Implementation difficulties" 4, [[Scalar and Multivariate Energy]] §5.

**Problem.** `SquaredNorm(M)` computes $\tfrac12 e^\top M e$ and assumes $M$ symmetric positive semidefinite; nothing checks it. With an indefinite $M$ the scalarisation is not convex, `islinear` still says `false`, and `jensen_gap` can be negative, which silently breaks the claim that the multivariate composite is a lower bound. [[energy]] calls it the sharpest outstanding correctness gap in the file.

**Options.**
1. Validate $M$ at construction (a Cholesky attempt).
2. Store a factor $L$ with $M = L^\top L$, so positive semidefiniteness holds by construction.
3. Document the precondition only (the current state).

**Resolution.** Option 1 implemented in `lib/LenticulumCore.jl/src/energy.jl`: `SquaredNorm(M; check = true)` verifies that `M` is square, symmetric, and positive semidefinite (attempting `cholesky` first for strictly PD matrices and falling back to `eigvals` for rank-deficient PSD matrices on standard float/integer strided matrices). Factors whose `M = inv(Symmetric(f.Q))` is derived from an already-validated SPD `Q` pass `check = false`.

**Reopened and closed again (2026-10-25, [[LP-013 Defects found in the review of master]]).** The check ran only for `StridedMatrix`, so `SquaredNorm(Diagonal([1, -1]))` and its `Symmetric` form were accepted and returned $-\tfrac12$ at $(0, 1)$. Now a `Diagonal` metric needs a real, nonnegative diagonal, and every other representation (symmetric, Hermitian, views) goes through the Cholesky/eigenvalue check (3254b1f8).
