#open-problem

> **Status: open (2026-10-22).** Related: [[energy]] §"Implementation difficulties" 4, [[Scalar and Multivariate Energy]] §5.

**Problem.** `SquaredNorm(M)` computes $\tfrac12 e^\top M e$ and assumes $M$ symmetric positive semidefinite; nothing checks it. With an indefinite $M$ the scalarisation is not convex, `islinear` still says `false`, and `jensen_gap` can be negative, which silently breaks the claim that the multivariate composite is a lower bound. [[energy]] calls it the sharpest outstanding correctness gap in the file.

**Options.**
1. Validate $M$ at construction (a Cholesky attempt).
2. Store a factor $L$ with $M = L^\top L$, so positive semidefiniteness holds by construction.
3. Document the precondition only (the current state).

**Leaning.** Option 1 for user-supplied matrices; factors that derive $M = Q^{-1}$ from a checked $Q$ can skip it.
