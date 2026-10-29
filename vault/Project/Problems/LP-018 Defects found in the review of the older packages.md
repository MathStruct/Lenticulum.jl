#open-problem

> **Status: resolved (2026-10-29), reviewed by Claude.** Related: [[LP-013 Defects found in the review of master]], [[L-SPEC-001 The Factor Interface]], [[src/likelihood|likelihood]], [[reddiff]], [[proxdm]], [[schedule]], [[krylov]], [[src/beliefs|beliefs]].

**Problem.** Codex ran a read-only review (2026-10-29) of the packages the LP-002/LP-011 work did not touch: VariationalDiffusion, the top-level `Lenticulum`, AlgebraicFactors and Adversarial. It reported 13 defects. Claude reproduced or re-derived each one before it was recorded here. The reproducers are scratch scripts outside the repository (`~/.julia/lenticulum-runs/review-older/`). Each fix must therefore add a regression test with the failing input below.

Code defects, most severe first:

1. **Poisson predictive free energy can hang** (`src/likelihood.jl`, `_poisson_entropy`). The loop stops only when `1 - mass < 1e-14`, so termination depends on the rounding of the accumulated mass. At $\lambda = 500$ (`x = log(500)`, `LikelihoodFactor(PoissonLikelihood(), 1)`, no observation) the mass stalls at `0.9999999999998282`. Once the later terms have underflowed, the loop never returns; a 60-second timeout killed Claude's rerun. The expected entropy is ≈ 4.5260757487. The bivariate `_conditional_entropy` has the same stopping rule and doubles `N` each time round, so it would allocate ever larger arrays instead.
2. **RED-Diff calibration omits the time-distribution normalisation** (`lib/VariationalDiffusion.jl/src/reddiff.jl`, `calibrate_lambda`). `regulariser_gradient` averages over $t \sim U[t_{\min}, 1]$, so the shrinkage is $\kappa = \frac{\lambda}{1 - t_{\min}}\int_{t_{\min}}^1 \frac{\sigma_t^2}{\alpha_t^2 v_0 + \sigma_t^2}\,dt$, not the integral alone. The returned $\lambda$ is too large by the factor $1/(1 - t_{\min})$ (direction corrected during resolution). That is 0.1% at the default `tmin = 1e-3`; at `tmin = 0.5` the gradient doubles (`2x` instead of `x`), and a unit-Gaussian posterior moves from 1/2 to 1/3. The quadrature reference in the tests repeats the missing factor.
3. **Energy-network Jacobian blocks use the first column's noise scale** (`energy_network.jl`, `epsilon_jacobian_block` via `_sigma_rows`, which reads `t[owner[idx], 1]`). It scales the weighted sum of Hessians by one σ per row, instead of weighting each column's Hessian by its own σ. Test case: $E(x) = \|x\|^2/2$, times `[0.1 0.2 0.3; 0.4 0.5 0.6]`, weights `[0.2, 0.3, 0.5]`. The diagonal comes out as `(0.32205, 0.89692)` against `(0.62824, 0.96069)`. `ImplicitDiffusion` passes per-node times here, so the implicit solve can reach this code.
4. **The generic batched input VJP passes the whole time matrix to each column** (`analytic.jl`). For $\varepsilon(X, T) = X \odot T$ with two noise blocks and three columns it throws `DimensionMismatch: times of size (2, 3) for 2 blocks and 1 columns`; the expected VJP is $T$. A scalar time fails too, after expansion.
5. **Pointwise energy ignores violated hard clamps** (`factor.jl`, `LenticulumCore.energy(::DiffusionFactor, …)`). With `(x = Observed(), y = Unobserved())`, reference `[0, 0]` and state `[1, 0]`, it returns `clamp = 0` for a violated hard coordinate. Finite precisions give costs that grow without bound as the precision increases, but an infinite precision gives zero.
6. **`MixtureProx` can return a strict local maximum** (`proxdm.jl`, `proximal(::MixtureProx, …)`). The residual test runs before any curvature check. Equal mixture means `[-2, 2]`, width 0.2, `v = 0`, `t = 0`, `λ = 1`: it returns `u = 0`, with objective 49.3095 and second derivative −2474. At `u ≈ 1.923` the objective is 1.9257. The docstring promises "the local minimiser reached from `v`". The failure needs a stationary starting point, so it is an edge case.
7. **Constraint and nonlinear constructors accept invalid noise covariances** (`src/constraint.jl`, `src/nonlinear.jl`). They test `isposdef(Symmetric(Q))`, which reads only one triangle. So `[1 0.2; 10 1]` is accepted, although its symmetric part has eigenvalues −4.1 and 6.1. A residual of dimension 2 also accepts `noise = [1.0]`, which leaves a one-dimensional energy space. `GaussianFactor` validates both shape and symmetry.
8. **Narrow ordinal categories lose all precision** (`src/likelihood.jl`, `OrdinalProbitLikelihood`). $\Phi(b) - \Phi(a)$ cancels. Cutpoints `[-1e-20, 1e-20]`, category 2, predictor 0 give log-likelihood `-Inf` with NaN derivatives; the true value is $\log(2\cdot10^{-20}) - \tfrac12\log 2\pi = -46.2775$. Narrow intervals far in a tail fail the same way.
9. **Finite particle weights can overflow during normalisation** (`lib/Adversarial.jl/src/ratio.jl`, `src/beliefs.jl`). Particles `{1, 3}` with weights `[1e308, 1e308]` give mean 0, covariance 0 and ESS `Inf`, instead of 2, 1 and 2. The ratio energy raises a false disjoint-support error, and `moment_match` reports a singular covariance. Dividing by the largest weight first avoids all three.

Wrong claims in notes and docstrings:

10. **The ELBO weighting** (`lib/VariationalDiffusion.jl/src/schedule.md` §3; also `reddiff.md` §3 and *The VP-SDE*). The weight on $\|\hat\varepsilon - \varepsilon\|^2$ in the continuous likelihood bound is $-\tfrac12\,\frac{d}{dt}\log\mathrm{SNR}_t$ with $\mathrm{SNR}_t = \alpha_t^2/\sigma_t^2$. For the VP-SDE that is $\beta(t)/(2\sigma_t^2)$, not $\sigma_t^2/\alpha_t^2$. Gaussian KL steps give 10.0754 at $t = 0.1$, where the claim gives 0.1157; at $t = 0.9$ they give 9.008 against 3460.6. The table also uses SNR inconsistently (`1/SNR²` in one row, `σ/α = 1/SNR` in the next), and `reddiff.md` §3 still contains a "— wait, that grows too —" from drafting.
11. **"RED-Diff's weight grows without bound as $t \to 1$"** (`reddiff.md` §3, `reddiff_weight` docstring, the vault). For the default VP-SDE, $\alpha_1 = e^{-5.025} > 0$, so $\lambda\sigma_t/\alpha_t \to 152.17\,\lambda$, which is finite. It grows steeply, but it stays bounded unless the schedule's terminal signal vanishes.
12. **"Exact at small sizes" for the log-determinant** (`krylov.md`, `KrylovSolve` docstring). Unit-vector probes remove the trace-sampling error but not the Lanczos truncation error. With `S = [2 1; 1 2]`, `lanczos = 1`, `probes = 2` it returns $2\log 2 = 1.386$ instead of $\log 3 = 1.099$. Exactness needs `lanczos ≥ n` as well, which the defaults (30 ≥ 8) happen to satisfy.
13. **"Pruning changes the moments by at most the pruned weight"** (`src/beliefs.md`, `reduce_mixture` docstring). Unit components at 0 and `1e12`, with weights `1 - 1e-10` and `1e-10`: pruning removes mass `1e-10` and moves the mean from 100 to 0. The pruned mass bounds the total-variation change, not the change in moments.

**Checked and found correct** (Codex, 2026-10-29):
- dense Gaussian conditioning, constraint equivalence and the three graded free-energy coordinates against an independent dense reference;
- Poisson, bivariate Poisson and ordinal derivatives against finite differences;
- the LP-013 mixture-anchor and prior-cotangent fixes;
- AlgebraicFactors, where no defect was found: toric enumeration and MLE, Plücker moments, polynomial Jacobians, Sampson gradients, and 1,510 existing game assertions;
- Adversarial ordinary-weight reweighting, moments and graph energy.

Analytic-mixture batching and the constraint polarity shortlist were excluded as defects because their notes document them as limits.

**Options.** Fix each item test-first, one commit each, as in LP-013. Items 10–13 change only notes and docstrings. Item 2 changes the value `calibrate_lambda` returns, so it also changes any recorded result that depends on it.

**Leaning / resolution.** Daniel authorised all thirteen fixes, test-first, on 2026-10-29; no commits requested. See the dated resolution below.

## Resolution (2026-10-29, Codex)

The pre-existing LP-002, LP-018 and Problem Log edits were retained. No checkout, clean,
reset, stash, commit, push or merge was run. Each item has an owning-suite regression in
`test/lp018.jl` (top-level), `lib/VariationalDiffusion.jl/test/lp018.jl` (VD), or
`lib/Adversarial.jl/test/lp018.jl`, with the particle-distance copy covered in
`lib/Mycelium.jl/test/lp018.jl`; the interop regressions are in top-level
`test/lp018_interop.jl`. Logs are under `~/.julia/lenticulum-runs/lp018/`.

Before fixing code, each reported code defect produced a failed assertion or error
in its regression test, as recorded in `top-before.log`, `diffusion-before.log` and
`adversarial-before.log`. For the documentation-only items 10, 11 and 13, the
independent arithmetic already passed, while assertions on the documented contract failed.
Item 1 calls with an explicit work budget: the old implementation rejects that keyword
rather than hanging the suite. The unbounded old call's hang is the original review's
reproduction, not a timeout hidden in this suite. Additional shifted-interval and EP
regressions failed before those paths were repaired; the optional interop/pullback tests
also failed before their fixes. On resumption, the Mycelium particle-distance
copy failed its two assertions in `Mycelium-final.log` before its normaliser was fixed;
`mycelium-regression-after.log` records the focused passing rerun; the subsequent
vector/tuple checks pass in `mycelium-containers-after.log`.

1. **Poisson entropy.** `src/likelihood.jl` and `src/likelihood.md`: sum relative pmfs
   about the mode, normalise afterwards, and stop using geometric mass/entropy tail
   bounds, with a throwing iteration cap. Bivariate entropy uses geometric marginal
   bounds, separate rectangular supports and a throwing grid-size cap. Regression:
   “LP-018.1 bounded Poisson predictive entropy”, independent 256-bit sums over
   32 rates from 1e-3 to 1e4 (including 500) and a three-Poisson convolution. The
   full top-level run measured a maximum absolute error of `8.881784197001252e-16`
   and entropy `4.526075748658402` at lambda 500.
2. **Calibration.** `reddiff.jl`, `reddiff.md`, `schedule.md`, VD `test/runtests.jl`,
   `examples/lp018_calibration.jl`, and the vault's *RED-Diff as a Statistical Game*:
   include the uniform-time density in shrinkage and its test reference; lambda is
   multiplied by `1-tmin`. The original report had the direction of lambda's error
   backwards, despite the correct doubled-gradient example. Regression: “LP-018.2
   calibration under uniform time measure”, independent midpoint integration at
   `tmin=0.001` and `0.5`, with posterior target 0.5 and no Monte Carlo.
3. **Energy Jacobian.** `energy_network.jl` and its implementation note: compute
   `sum_b w_b diag(sigma_b) H_b` by batched Hessian-vector products, including each
   reference diagonal. Regression: “LP-018.3 energy block Jacobians weight each
   column”, using explicit per-column quadratic Hessians, reference energies, cross
   terms, and subset/reordered rows; input VJP/JVP checked against the same independent
   matrices. Mixture methods already selected the right column's times.
4. **Generic VJP.** `analytic.jl` and `analytic.md`: select the corresponding time
   column before calling the single-point Jacobian. Regression: “LP-018.4 generic
   batched VJP column times”, `epsilon = X .* T`, both matrix and scalar time.
5. **Hard clamps.** `factor.jl` and `factor.md`: `clamp=Inf` on a violated hard
   coordinate and zero contribution on exact agreement; finite-precision costs unchanged.
   Regression: “LP-018.5 hard clamp pointwise energy”, also checks the projected local
   free energy. L-SPEC-001 §3 requires the same graded pointwise space, not finiteness
   on infeasible points; `Inf` matches other factors' infeasibility convention. The VD
   conformance fixture now uses a feasible point for its finite-value check and
   separately asserts `Inf` for the violated hard observation; its old `0.5`
   expectation codified the defect (`diffusion-conformance-before.log`).
6. **Mixture prox.** `proxdm.jl` and `proxdm.md`: check curvature before accepting a
   stationary point, backtrack along negative curvature to escape a maximum, and throw
   on solver failure. Regression: “LP-018.6 symmetric mixture prox escapes a maximum”,
   independent two-normal objective and curvature; the existing ProxDM testsets retain
   the one-Gaussian closed forms, samplers and inference checks.
7. **Covariances.** `src/constraint.jl`, `src/nonlinear.jl` and their implementation
   notes: require exact residual dimension and symmetry before positive definiteness,
   with clear `ArgumentError`s. Regression: “LP-018.7 residual covariance validation”,
   the reported asymmetric covariance and short noise vector, plus rectangular,
   indefinite and valid SPD cases for both constructors.
8. **Ordinal intervals.** `src/likelihood.jl` and `src/likelihood.md`: symmetric tail
   log differences with `expm1`, a midpoint expansion for narrow intervals, and its
   matching analytic gradient/Hessian and EP moments. Preserve cutpoint width before
   shifting by the predictor. Regression: “LP-018.8 ordinal narrow intervals and both
   tails”, independent 256-bit references at the failing category, `[8,9]`, `[-9,-8]`
   and narrow tail intervals, at predictors 0 and 0.7; a vanishing-width EP update is
   checked against a Gaussian conditional with mean 0.35 and variance 0.5.
9. **Weights.** Adversarial `ratio.jl`, top-level `beliefs.jl`, VD `analytic.jl`, Core's
   ExponentialFamily and ChainRules extensions, and the corresponding `ratio.md`,
   `beliefs.md`, `analytic.md`, `interop.md` and `open_model.md`, plus Mycelium's
   `messages.jl` and `messages.md`: divide finite
   weights by the maximum before summation, including the normalisation pullback's
   denominator. Regressions: “LP-018.9 finite overflowing particle weights”, “LP-018.9
   finite particle weights in moment matching”, “LP-018.9 mixture weights avoid finite
   overflow”, and “LP-018.9 finite particle weights across interop and AD”. The
   `{1,3}` example now has mean 2, variance 1, ESS 2; constant log ratio 2 gives energy
   -2, SampleList preserves the mean, and the uniform log-weight cotangent is zero.
   Mycelium’s “LP-018.9 particle distance normalises finite overflowing weights”
   also checks distances 0 and 1/6 against independent total variation. Adversarial and
   Mycelium scaling remains non-mutating so tuple weights still work.
   Other normalisers already subtract the largest log weight (Core/Mycelium reweighting/sampler)
   or divide by the maximum (AlgebraicFactors); those required no change.
10. **ELBO and SNR.** `schedule.jl`'s docstring, `schedule.md`, `sampler.jl`'s
    log-grid comment and `sampler.md`, `reddiff.jl`'s
    docstrings, `reddiff.md`, and vault *The VP-SDE* and *RED-Diff as a Statistical
    Game*: derive `-1/2 d(log SNR)/dt = beta/(2sigma^2)` from Gaussian KL steps,
    use power SNR, state the `dt` measure and the `1-tmin` conversion to a uniform
    expectation. The historical `snr` function still returns the amplitude ratio.
    Remove the drafting text and the invalid comparison by squaring field weights.
    Regression: “LP-018.10 ELBO epsilon weight from Gaussian KL”, finite Gaussian KL
    differences plus an assertion on the corrected documented formula.
11. **Terminal weight.** `reddiff.jl`, `reddiff.md`, and vault *RED-Diff as a
    Statistical Game*: the default
    VP terminal signal is nonzero, so the unit-lambda terminal weight is finite,
    `152.16697028394637`; divergence requires a vanishing terminal signal. Regression:
    “LP-018.11 terminal RED-Diff weight is finite”, closed-form endpoint and docstring.
12. **Krylov exactness.** `krylov.jl` and `krylov.md`: enforce full Lanczos depth on
    the unit-vector branch, rather than merely weakening the claim. Other branches
    retain the requested depth and all defaults are unchanged. Regression: “LP-018.12
    exact unit-vector logdet uses full depth”, the recorded 2x2 case, `log(3)`.
13. **Pruning.** `src/beliefs.jl`'s docstring and `src/beliefs.md`: removed mass bounds
    total variation after renormalisation; moments need bounds on locations/covariances.
    Regression: “LP-018.13 pruning bounds mass, not moments”, the recorded mean change
    100 to 0 on removing mass 1e-10, and the corrected documented contract.

### Downstream calibration numbers

Repository-wide searches for `calibrate_lambda`, `shrink_integral`, and the recorded
values found two numeric tables/measurements, in `reddiff.md` and *RED-Diff as a
Statistical Game*. Both were updated from
`examples/lp018_calibration.jl` (rerun on 2026-10-29):

- Default lambda: `1.3812067827926036` → `1.379825576009811` (the old displayed 1.38
  rounded both values; the notes now give 1.379826).
- The old unnormalised `s I(s)` values, for `s=(0.25,0.5,1,2,4)`, were
  `(0.206671051,0.389224512,0.724004556,1.330160041,2.415093770)`;
  the uniform-time averages are
  `(0.206877929,0.389614126,0.724729285,1.331491532,2.417511281)`.
  Both notes now print these to six decimals. Ratios of relative prior strengths
  remain the same because the calibration and time density factors cancel.
- The previous unbacked “~2% / ~0.4%” posterior measurement in `reddiff.md` was replaced:
  at clamp precisions 1 and 2, maximum relative errors are 2.1833% and 1.4030%, and
  six-seed mean errors are 0.9147% and 0.5028%, using the existing test's settings.
  Both notes name the script and run. Test tolerances and exact posterior targets
  are unchanged. No examples or benchmarks outside this new script record a numeric
  `calibrate_lambda` result; its other callers are covered by the full VD suite.


### Final package verification

All package test entrypoints were run sequentially, with one Julia job at a time,
using the package test environments. The top-level environment in
`~/.julia/lenticulum-runs/lp018/root-test/` contains the root project's test extras
and local source paths, including the optional ExponentialFamily and Zygote checks.
`verification.json` records wall-clock times and log paths:

| suite | passing assertions | wall time (seconds) |
|---|---:|---:|
| VariationalDiffusion | 1452 | 350.4 |
| Mycelium | 1073 | 734.1 |
| Lenticulum | 1302 | 366.2 |
| Adversarial | 382 | 9.9 |
| LenticulumCore | 314 | 48.3 |

Mycelium's final focused file also passes all four vector/tuple assertions after the
non-mutating-container check was added; two of those extend the 1,073-assertion full
run. Device suites gated by `LENTICULUM_DEVICE_TESTS` were not enabled. No commit is
being made.

The required vault and API builds use `QUARTZ_DIR` pointing to the disposable copy
under the same run directory. `vault-build.log`, `docs-build.log` and `builds.json`
record their outcomes and wall times. The copy's restoration hook was removed so no
checkout or clean runs inside Quartz; the copy is deleted after the final build.

## Review of the resolution (2026-10-29, Claude)

Checked independently of Codex's logs:

- Every source change matches its item. The narrow-interval expansion
  $\int_{-h}^{h}\varphi(m+u)\,du = 2h\,\varphi(m)\bigl[1+\tfrac{m^2-1}{6}h^2+\tfrac{m^4-6m^2+3}{120}h^4\bigr]$,
  its first two derivatives in $m$ and the EP moments were re-derived by hand. The
  ELBO coefficient $-\tfrac12\frac{d}{dt}\log\mathrm{SNR}_t=\beta/(2\sigma_t^2)$ follows from
  $\alpha_t^2=e^{-B_t}$. The corrected direction of item 2 is right: the old $\lambda$ was too
  large by $1/(1-t_{\min})$.
- Against 2048-bit references: `_poisson_entropy` at $\lambda\in\{10^{-3},0.3,1,7.5,40,500,3000\}$
  agrees to relative error $\le 5.7\cdot10^{-15}$ (the largest at $\lambda=10^{-3}$, where
  $H\approx0.0079$), and $\lambda=500$ takes 13 µs; `_logΦdiff` on narrow intervals at
  $m\in\{0,\pm30,2.3,-7,0.4\}$ agrees to $\le 2.1\cdot10^{-16}$. `calibrate_lambda` agrees with
  an independent 200 000-node midpoint rule to $10^{-8}$ at $v_0\in\{0.25,1,4\}$
  (the code uses a 200-node trapezoid). The terminal weight is $152.16697\ldots$.
- The new regression files, run against HEAD (940ca211) in a separate worktree: every
  LP-018 testset fails or errors (top level 12 failed + 3 errors of 22; VariationalDiffusion
  25 failed + 2 errors of 44).
- With the fixes, run sequentially: VariationalDiffusion, Lenticulum, Adversarial,
  LenticulumCore and Mycelium all pass (387 s, 418 s, 11 s, 51 s, 738 s); the vault builds
  (256 pages). Logs in `~/.julia/lenticulum-runs/lp018-claude/`.

Behaviour changes to be aware of: `MixtureProx` now throws when it does not reach a local
minimiser (it used to return the last iterate), and a violated hard clamp has energy `Inf`.
