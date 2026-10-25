#open-problem

> **Status: closed (2026-10-25).** Found while reviewing `Retracted` ([[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §2.2). Related: [[factors]], [[Everything is a Factor]], [[grassmann]]. Code: `lib/Mycelium.jl/src/factors.jl` (`Momentum`, `Nesterov`), `lib/Mycelium.jl/test/runtests.jl`.

**Problem.** Mycelium's `Momentum(η, γ)` and the `put` of `Nesterov(η, γ)` computed
$s' = -\gamma s - \eta\bar p$, then $p + s'$. Cruttwell et al., *Categorical Foundations of
Gradient-Based Learning*, Example 3.16 (arXiv:2103.01931v1), define momentum by
$v' = \gamma v + \epsilon p'$ and $p \mapsto p - v'$; with Mycelium's state $s = -v$ this is
$s' = \gamma s - \eta\bar p$. Example 3.17's Nesterov look-ahead $p - \gamma v = p + \gamma s$ was
already consistent with $s = -v$; only the update had the wrong sign. With $\gamma > 0$ the old
rule damped each step by the previous one instead of accelerating it. The Mycelium test
hard-coded the wrong value, and two tables (`factors.md`, [[Everything is a Factor]]) repeated
it.

**Resolution.** The sign is corrected in both rules, the test now expects
$0.9\cdot(-0.2) - 0.2 = -0.38$, and the tables are fixed. The Grassmannian descent test of
`lib/AlgebraicFactors.jl` runs `Retracted(Momentum(0.04, 0.6), M)` with the corrected rule.
