#open-problem

> **Status: resolved (2026-10-29) by Claude; reviewed by Codex five times (2026-10-30); all follow-ups fixed by Claude, limitations recorded.** Related: [[LP-018 Defects found in the review of the older packages]], [[Messages are Inversions]], [[DEQ as a Relation]], [[The Equilibrium Family]], [[Bethe Free Energy]].

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

## Review of the resolution (2026-10-30, Codex)

Codex reviewed commit 29b2b18b read-only and confirmed items 1, 2, 5, 8, 9, 12 and the
mathematics of every fix (the relay's $-H(q)$ and the LP-014 consequence; sixty dense checks of
the overlap formulas; the pullback formula; the AlgebraicFactors caller of
`combine_with_overlap`). It found nine regressions or gaps in the fixes; Claude reproduced each
with Codex's scripts (`~/.julia/lenticulum-runs/review-lp019/`), added a regression test, and
fixed it:

1. Item 6: the proper overlap called `inv` on a Cholesky factor, which Zygote cannot
   differentiate (`dpotri` foreigncall); expected derivative 0.25. Now uses the
   `belief_cov`/`_precision_*` helpers that have rules.
2. Item 6: the same code scalar-indexed diagonal device Gaussians (JLArrays,
   `allowscalar(false)`). The helpers keep the diagonal device path.
3. Item 6: an improper quadratic site far from the origin ($s = 10^8$) still gave weights
   `[1.0, 0.368]`, summing to 1.37: the log masses were $O(s^2)$ and the normalisation added its
   correction to a number of size $10^{16}$. Mixture products now re-centre each improper site at
   a reference point shared by all its pairs (`_overlap(c, d, ref)`; the constant dropped depends
   on the site alone), and normalise by subtracting the maximum first.
4. Items 3 and 13: the tolerance $\sqrt{\varepsilon_{\text{Float32}}}$ admitted
   `[1 0.9999; 1.0002 1]`, whose upper triangle is definite but whose symmetric part is
   indefinite. The tolerance is now $\sqrt\varepsilon$ of the element type, and definiteness is
   checked on, and factorisations use, $M/2 + M^\top/2$ (not $(M + M^\top)/2$, which overflows near
   floatmax); the constraint and nonlinear constructors store that symmetric part.
5. Item 7: for `w = [1e-200, 1e200]` the new pullback lost $-1/\sum w = -10^{-200}$ because
   $p_1$ underflowed. Where $p_j$ underflows it now uses $D_j/w_j - \sum D/\sum w$.
6. Item 6: two `Gaussian(0, 1e308)` gave $-\infty$; $\Sigma_c + \Sigma_d$ is now scaled by its
   largest diagonal entry.
7. Item 4: Float16 ForwardDiff duals still got the $10^{-7}$ step (a zero increment, NaN);
   the step now uses `eps` of the dual (its primal's), and a step that does not change $x$
   throws instead of dividing 0 by 0.
8. Item 10: `Float64(norm(v))` underflowed a BigFloat residual of $10^{-400}$ to a false root;
   the norm now stays in the residual's precision.
9. Item 11: [[DEQ as a Relation]] §5 said a point message cannot carry a likelihood; it is the
   likelihood of an exact constraint, $\delta(z - z^\ast)$. The note now says it cannot carry a
   finite-noise likelihood.

## Second review (2026-10-30, Codex)

Codex reran its fifteen reproducers (the nine above and six on the exposure code of
[[L-SPEC-003 Exposed Parameters and Energy Channels]]) against the working tree: fourteen fixed,
the exposure polarity list still incomplete above its size cap. It found further defects, all
reproduced and fixed by Claude with regression tests:

- **Re-centring changed relative masses between different improper sites** (introduced by the
  fix of the first review's item 3): the dropped constant $\eta^\top r - r^\top\Lambda r/2$ is a
  constant of each site, so it cancels only when the other side has exactly one improper site.
  Weights for two sites against $\mathcal N([2,0], I)$ were `[0.731, 0.269]`; the closed form is
  $\propto (e^{-1}, e^{2})$. A reference is now used only in the single-site case.
- **One global scale for the proper overlap underflowed tiny variances**
  ($\mathcal N(0, \operatorname{diag}(10^{200}, 10^{-200}))$ threw); it is now a congruence scaling
  by $\sqrt{\max(\Sigma_{c,ii}, \Sigma_{d,ii})}$ per coordinate, diagonal-preserving.
- **Zygote through the diagonal device overlap** scalar-indexed in the pullback of
  `Diagonal / α`; now broadcasts on `diag`.
- **The √eps(T) symmetry tolerance** changed under ForwardDiff duals and rejected LP-002's
  finite-difference adjoint perturbations (the Mycelium suite's "LP-002 C correlated dense
  Gaussian message coordinates", ~1.3e-6 asymmetry). The tolerance is back to √eps(Float32) for
  every type: with definiteness checked on the symmetric part, it only has to refuse gross
  asymmetry.
- **The symmetric part $M/2 + M^\top/2$ rounded a subnormal diagonal to zero**; it is now
  $M + (M^\top - M)/2$, exact for symmetric $M$ and free of overflow.
- **The pullback's shared term $\sum D/(m s)$ overflowed** for `[1e-200, 1e308, 1e308]`; now
  $(\sum D/m)/s$.
- **The projected-moments tolerance** rejected a valid Float32 rank-one belief and accepted a
  flat direction read with weight $10^{-11}$; it now finds flat directions by an eigendecomposition
  in Float64 and requires $C$ not to see them beyond $64\varepsilon\lVert C\rVert$.
- **The channel/parameter shadow rule rejected valid clamped exposure**; such factors now keep
  the clamped path and refuse only the belief path.
- **The exposed polarity list**: complete enumeration compiles one `Polarity` type per
  assignment (21.9 s for six channels). Above five channels in all, `supported_polarities` now
  throws, as L-SPEC-001 permits, and `hasnopolarity`/`isunidirectional` answer lazily.

Recorded, not fixed: `SolveReport.residual` is a `Float64` field, so a BigFloat residual of
$10^{-400}$ is reported as `0.0` (convergence is decided in full precision; pre-existing).

## Third review (2026-10-30, Codex)

All fifteen original reproducers and the second round's reproducers pass. Nine further
findings, reproduced by Codex's scripts (`~/.julia/lenticulum-runs/review-followups2/`) and fixed by
Claude with regression tests:

- A mixture with proper components beside one improper site still got the reference shift
  (weights `[0.101, 0.899]`, closed form `[0.015, 0.985]`): a reference is now used only when
  the other side is that single site and nothing else.
- Exposure: an information vector with zero precision (a linear site $e^{\eta^\top z}$) was taken
  for no information; it is now refused where the factor reads it. The flat-direction test had
  an absolute floor that let a read of weight $10^{-16}$ through; it is now relative.
- The covariance pullback returned an asymmetric precision cotangent, inconsistent with the
  symmetric-part forward map (Zygote $-0.548$/$-0.025$ against ForwardDiff $-0.286$ for both
  off-diagonals); it is symmetrised.
- The weight pullback overflowed $\sum D$ for cotangents near floatmax; $D$ is scaled first.
- The shadow rule refused channels named like *unexposed* parameters (`(:x, :noise)` with
  learned noise); the generic rule now concerns exposed blocks only, and `GaussianFactor`
  refuses only channels named `:A` or `:b` on its belief path.
- `isunidirectional` above the enumeration cap traversed $3^6$ assignments (24 s, 1.6 GB);
  both predicates now answer structurally, assuming a factor that accepts a block target with
  latent channels also accepts it with everything observed.
- The proper overlap overflowed for means near $\pm$floatmax; it now scales before subtracting
  and halves before squaring.
- `show` printed an improper belief (indefinite symmetric part) as a proper Gaussian.

Still not supported, as documented: mixtures with improper components on both sides
(the overlap rule needs a proper first operand), and `SolveReport.residual`'s Float64 field.

## Fourth review (2026-10-30, Codex)

Every earlier counterexample passes. Three further findings, fixed by Claude with regression
tests (scripts in `~/.julia/lenticulum-runs/review-followups3/`):

- An improper site with a skew part in $\Lambda$ (`[1 1e-4; -1e-4 0]`) gave weights `[0.378, 0.622]`
  where its symmetric part `diag(1, 0)` gives `[0.5, 0.5]`: the expansion used the raw $\Lambda$
  while the factorisations used the symmetric part. It now uses the symmetric part throughout.
- The third round's "scale before subtracting" rounded each Float32 mean separately; means
  $10^{-6}$ apart with variances $10^{-12}$ lost 0.027 in the log overlap. The difference is now
  taken first, with the scaled form only as a fallback when $\mu_c - \mu_d$ overflows.
- The structural `hasnopolarity`/`isunidirectional` of `ExposedFactor` disagreed with the
  accepted assignments for custom hooks; both are exact again, by the lazy search with early
  stopping (exponential when it finds none or exactly one).

Recorded, not fixed (extremes only): cotangent rescaling can give `[-1, 0]` where the exact
pullback is zero after underflow, subnormal Float32 weights can give infinite gradients, and
the symmetrised covariance adjoint can overflow where the reference is $-10^{308}$.

## Fifth review (2026-10-30, Codex)

Every earlier counterexample passes. Two further findings were fixed by Claude with
regression tests (scripts in `~/.julia/lenticulum-runs/review-followups4/`):

- AlgebraicFactors' `branch_entropy_bounds` still mirrored the upper triangle
  (`Symmetric(c.Λ)`), so after item 3 its bounds excluded the entropy Core and Mycelium report
  for a slightly asymmetric precision (2.98178 against 2.98168), and threw for one that Core
  accepts. It now uses `_symmetric_part`.
- The flat-direction test in exposure's `_projected_moments` used a cut-off relative to the
  largest eigenvalue, so a precision of $10^{16}$ on an unread coordinate made a read coordinate
  of precision 1 look flat. The precision is now equilibrated, $D\Lambda D$ with
  $D = \operatorname{diag}(\Lambda_{ii}^{-1/2})$, before the cut-off.

Accepted, not changed: exact `hasnopolarity`/`isunidirectional` on `ExposedFactor` visit all
$3^n$ assignments when they find none or one (Codex measured 27 s for six channels on first
call); the round-four structural shortcut was wrong for custom hooks, and exactness was kept.
A fast path would need factor-provided support summaries. The extreme-value limitations above
are unchanged.

