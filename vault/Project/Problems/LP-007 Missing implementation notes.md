#open-problem

> **Status: closed (2026-10-23, `review-argon`).** Related: [[Start Here]] §"Adding to it", [[Map of Content]] §4.

**Problem.** Every Julia file is meant to have an implementation note beside it. `lib/Lenticulum.jl/src/gaussian.jl` and `lib/Lenticulum.jl/src/beliefs.jl` have none (`gaussian.md`, `beliefs.md`), although their docstrings cite `beliefs.md`, `gaussian.md` (§2, §3) and a vault note *The Gaussian Factor*, which does not exist either. Those references point nowhere; `lib/Lenticulum.jl/src/constraint.md` already flags this.

**Resolution.** Wrote `lib/Lenticulum.jl/src/gaussian.md` and `lib/Lenticulum.jl/src/beliefs.md` covering the theory, implementation, what they do not do (`LP-005`), and implementation difficulties, and updated `lib/Lenticulum.jl/src/gaussian.jl` to reference `Scalar and Multivariate Energy.md`.
