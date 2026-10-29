#open-problem

> **Status: resolved (2026-10-29) by Claude; not yet reviewed.** Related: [[LP-018 Defects found in the review of the older packages]], [[Messages are Inversions]], [[DEQ as a Relation]], [[The Equilibrium Family]], [[Bethe Free Energy]].

**Problem.** Codex ran a read-only review (2026-10-29) of Mycelium, LenticulumCore (with its extensions), ImplicitLayers and the vault notes on factor graphs and the equilibrium family. It reported 12 defects. Claude reran every reproducer before recording it here and got the same numbers. The reproducers are scratch scripts outside the repository (`~/.julia/lenticulum-runs/review-core/`). Each fix must therefore add a regression test with the failing input below.

Code defects, most severe first:

1. **A relay changes the Bethe free energy** (`lib/Mycelium.jl/src/factors.jl`, `local_free_energy(::RelayFactor, …)`). A categorical prior `[0.3, 0.7]` on one variable has $F = 0$. Relaying it to a second variable leaves the model and the marginals unchanged but gives $F = 0.6108643$, which is $H(p)$. The relay is an equality factor: its energy is zero, but its joint belief $p(a)\,\delta(a = b)$ has entropy $H(p)$, and the zero contribution leaves the extra degree of the relayed variable uncancelled. The contribution should be $-H(q)$, with $q$ the relayed belief (relative to the measure on the diagonal).
2. **Pooling a mixture with an improper Gaussian depends on operand order** (`src/messages.jl`, `_mixture_product`; `src/gaussian.jl`, `combine_with_overlap`). Components `N(-1, 1)`, `N(1, 1)` with equal weights and a site `η = [0.3]`, `Λ = [0]` give weights `[0.3543, 0.6457]` mixture-first. Site-first throws "requires a proper first Gaussian belief". The order follows the factor insertion order, so the same graph built in a different order fails.
3. **Gaussian validation reads one triangle** (`lib/LenticulumCore.jl/src/gaussian_belief.jl`, `_chol`, `_precision_isproper`). Covariance `[1 0.2; 10 1]` is accepted and becomes `[1 0.2; 0.2 1]`. As a precision it passes `isproper`, although its symmetric part has eigenvalues −4.1 and 6.1. LP-018 item 7 fixed the same defect in the top-level constructors.
4. **Finite differences divide by the requested step, not the step taken** (`lib/ImplicitLayers.jl/src/solve.jl`, `fd_jacobian`; `src/flow.jl`, `_divergence`). The increment is rounded to the element type, but the quotient uses the unrounded `h`. `fd_jacobian(identity, Float32[1])` returns 1.1920929, and the RK4 log-determinant of $u' = u$ on $[0, 1]$ (50 steps) is 0.9758 instead of 1. In Float16 both return 0.
5. **A damped tree schedule claims exact convergence** (`lib/Mycelium.jl/src/passing.jl`, `propagate!`). Warm-start a unary Gaussian source at mean 0, change its parameter to mean 10, and propagate with `damping = 0.5`. The marginal mean is 5, but the report says converged, residual 0, "one sweep is exact". The fixed point is 10.
6. **Gaussian overlaps cancel catastrophically** (`src/gaussian.jl`, `combine_with_overlap`). The log-partition difference loses everything at large means: two identical `N(1e8, 1)` give −4 instead of $-\tfrac12\log 4\pi = -1.2655$. Shifting a two-component mixture and its evidence by `1e8` changes the product weights from `[0.5312, 0.4688]` to `[0.5, 0.5]`; at a shift of `1e6` they are still correct.
7. **The weight-normalisation pullback overflows** (`lib/LenticulumCore.jl/ext/LenticulumCoreChainRulesCoreExt.jl`). At equal weights the gradient of `sum(CategoricalBelief(w).logp)` is exactly 0. At `w = [1e-308, 1e-308]` it is `[-Inf, -Inf]`, at `[1e-320, 1e-320]` `[NaN, NaN]`; `1e-307` is still fine.
8. **Scalar particles come back from `SampleList` as zero-dimensional arrays** (`ext/LenticulumCoreExponentialFamilyExt.jl`, `as_belief(::SampleList)`). `SampleBelief([1, 3], [0.5, 0.5])` round-trips to samples `[fill(1), fill(3)]`. Pooling with categorical probabilities `[0.2, 0.3, 0.5]` gives `[2/7, 5/7]` before and throws after.
9. **A residual schedule with no budget certifies convergence** (`src/passing.jl`, `sweep!(…, ::ResidualSchedule, …)`). With `maxtasks = 0` or `-1` and a unary `DataFactor(:x, 7)`, no message is computed; the marginal is `TrivialBelief()` and the report says converged, residual 0. The expected marginal is `DiracBelief(7)`.
10. **Squared residuals underflow in the root solvers** (`lib/ImplicitLayers.jl/src/solve.jl`, `_norm`). `sqrt(sum(abs2, v))` returns 0 for `v = [1e-200]`. With a constant residual `[1e-200]` and `tol = 1e-210`, Picard, Broyden and limited Broyden all report convergence at iteration 0, and there is no root.

Wrong claims in notes:

11. **"A factor → variable message is the inversion $c'_\pi$"** ([[Messages are Inversions]]; also [[DEQ as a Relation]] §5). Mycelium's messages are likelihood sites; the inversion is the posterior. For a prior $\mathcal N(0, 1)$ and an observation $y = 2$ with unit noise the outgoing message is $\mathcal N(2, 1)$ and `invert` returns $\mathcal N(1, 0.5)$. Pooling the note's message with the prior gives $\mathcal N(2/3, 1/3)$, which is wrong.
12. **"A singular $I - \partial_z g$ means the relation branches"** ([[DEQ as a Relation]] §4, [[The Equilibrium Family]] §4). For $g(z, x) = z - z^3 + x$ the relation is $z^3 = x$, with the unique real root $\sqrt[3]{x}$. At $x = z = 0$, $I - \partial_z g = 0$ and `ift_sensitivity` throws. A singular Jacobian means the implicit function theorem's sufficient condition fails (here: the root is not differentiable in $x$); branching has to be shown separately.

Found while verifying the fixes (Claude, 2026-10-29):

13. **LP-018 item 7 rejects covariances that are symmetric up to rounding** (`src/constraint.jl`, `src/nonlinear.jl`). The LP-018 fix tests `issymmetric(Q)`, which is exact. AlgebraicFactors builds $Q = J S J^\top + \lambda\Theta\Theta^\top$, which rounds asymmetrically, and its "Polynomial degree-one messages and lens contract" testset errors with "noise covariance must be symmetric". LP-018's verification ran five suites, not AlgebraicFactors, so this reached `master` with commit 5513f457.

**Checked and found correct** (Codex, 2026-10-29):
- Mycelium: 546 existing assertions; 458 new ones over six Gaussian trees (different roots; tree, flooding and residual schedules) agreeing with dense references in means, variances, cavity exclusion, counting numbers and Bethe free energy; dense and Krylov implicit graph gradients.
- LenticulumCore: 234 assertions on energy algebra, scalarisation, Sampson localisation, belief construction and lazy polarity enumeration.
- ImplicitLayers: 544 assertions on linear DEQ solves in both directions, sensitivities, games, anchors, energy-error checks and Euler/RK4 convergence orders.

Documented limits (dropped categorical labels, legacy state threading) were excluded as defects.

**Options.** Fix each item test-first, as in LP-018. Items 11–12 change only notes.

**Leaning / resolution.** Daniel asked Claude to implement the fixes (2026-10-29: "You may implement it yourself"); no commits requested.

## Resolution (2026-10-29, Claude)

Each item got a regression test with the input above, run and seen to fail before its fix:
`lib/Mycelium.jl/test/lp019.jl` (items 1, 2, 5, 6, 9), `lib/LenticulumCore.jl/test/lp019.jl`
(item 3), `lib/ImplicitLayers.jl/test/lp019.jl` (items 4, 10) and the top-level
`test/lp019_interop.jl` (items 7, 8, 13; the ChainRulesCore and ExponentialFamily extensions do
not load in LenticulumCore's own test environment).

1. `RelayFactor` contributes $-H(q)$, $q$ the pooled incoming messages, and `uses_cavity` is
   `true`. Relaying a categorical or Gaussian prior now leaves $F = 0$. **Behaviour change:** a
   directed relay (`Absorbing()` in, `Emitting()` out) now raises the LP-014 error in graph
   free-energy evaluation; the LP-014 test now uses `Bidirectional()` relay edges (still
   $F = 4.5$) and asserts the error for the directed graph.
2. `_mixture_product` puts a proper Gaussian first whichever side it came from; both orders give
   `[0.3543, 0.6457]`.
3. `Gaussian`, the precision factorisation and `isproper` check
   $\max|M - M^\top| \le \sqrt{\varepsilon_{\text{Float32}}}\max|M|$ (`_nearly_symmetric`).
4. `_fd_step` (shared by `fd_jacobian` and `_divergence`) divides by the increment the rounded
   probe makes; the default relative step is $\max(10^{-7}, \sqrt{\varepsilon_T})$ for IEEE types,
   so Float64 keeps $10^{-7}$, and stays $10^{-7}$ for dual numbers. The identity's Jacobian is
   exactly 1 in Float64, Float32 and Float16 (it was $1 + 5.8\cdot10^{-10}$ even in Float64).
   The first version converted to Float64 and broke the stiff SciML CNF testset, which runs
   ForwardDiff duals through `_divergence`; fixed before the final runs.
5. Damped tree schedules are iterated and checked; undamped ones keep the one-sweep shortcut.
6. Proper overlaps use $\log\mathcal N(\mu_c - \mu_d; 0, \Sigma_c + \Sigma_d)$, improper ones an
   expansion about $\mu_c$ in the same convention as before. Exact to rounding at shifts $0$,
   $10^6$, $10^8$; beyond $\sim 10^{12}$ the canonical form itself loses the mean.
7. The pullback is $(D_j - p_j\sum D)/w_j$; zero at equal weights down to `1e-320`.
8. `as_belief(::SampleList)` unwraps zero-dimensional samples.
9. `ResidualSchedule` rejects `maxtasks < 1` at construction; the LP-002 adjoint test that
   expected the gradient driver to reject it now expects the constructor to.
10. `_norm` is `LinearAlgebra.norm`.
11. [[Messages are Inversions]] states $c'_\pi(y) = \pi \odot \mu_{f\to v}$ with the worked case,
    and `passing.md` and [[Backward Signatures]] name `likelihood_message`. [[DEQ as a Relation]]
    §5 now says the DEQ message is the fixed point, with the prior only a warm start (the old
    "posterior rather than a likelihood" was wrong).
12. [[DEQ as a Relation]] §4 and [[The Equilibrium Family]] §4 say a singular $I - \partial_z g$
    is the failure of the IFT's sufficient condition, with $z^3 = x$ as the counterexample.
13. `LinearConstraintFactor` and `NonlinearFactor` use `_nearly_symmetric` instead of exact
    `issymmetric`.

Implementation notes updated: Mycelium `factors.md`, `passing.md`, `schedules.md`,
`gaussian.md`; LenticulumCore `gaussian_belief.md`, `interop.md`, `open_model.md`;
ImplicitLayers `solve.md`, `flow.md`.

**Verification (2026-10-29).** All seven suites pass with `Pkg.test()`: LenticulumCore,
Mycelium and ImplicitLayers (both with `LENTICULUM_DEVICE_TESTS=true`), the top-level
package, AlgebraicFactors, Adversarial and VariationalDiffusion. The vault builds (258 pages)
and `docs/make.jl` builds. Logs: `~/.julia/lenticulum-runs/lp018-claude/lp019/`.

