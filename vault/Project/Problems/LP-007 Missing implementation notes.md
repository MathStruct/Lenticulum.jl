#open-problem

> **Status: open (2026-10-22).** Related: [[Start Here]] §"Adding to it", [[Map of Content]] §4.

**Problem.** Every Julia file is meant to have an implementation note beside it. `src/gaussian.jl` and `src/beliefs.jl` have none (`gaussian.md`, `beliefs.md`), although their docstrings cite `beliefs.md`, `gaussian.md` (§2, §3) and a vault note *The Gaussian Factor*, which does not exist either. Those references point nowhere; `src/constraint.md` already flags this.

**Resolution.** Write the two implementation notes, and either write *The Gaussian Factor* or point the docstrings elsewhere. The notes need the theory, the implementation, what they do not do, and the difficulties met. `gaussian.md` should cover the residual and its two scalarisations ([[L-ADR-0003 Pointwise energy and free energy are separate levels]]) and the deferred learnable $Q$ ([[LP-005 Learning the noise covariance]]).
