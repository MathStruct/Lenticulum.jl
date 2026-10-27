#open-problem

> **Status: resolved within the agreed bounds + exact-cases scope (2026-10-27); general games outside that scope remain explicitly unsupported. The 2026-10-23 resolution listed methods that did not exist.** Related: [[L-SPEC-001 The Factor Interface]] §4–§5, [[statistical_game]], [[abstract_types]], [[factors]].

**Problem.** `LenticulumCore` declares the game-level `entropy(f, π, y, ps, st)` and `free_energy(f, π, y, ps, st)` (AutoBayes Definition 20), and `abstract_types.jl` lists `entropy` among the methods a factor must provide. At the time this problem was opened, no factor implemented either. Every factor that is used in a graph implements Mycelium's `local_free_energy(f, msgs, ps, st)` instead, which takes all incoming messages because a graph has no distinguished prior/observation split. Then `scalar_free_energy(f, π, y, ...)` failed for every factor in the repository, the same kind of gap as issue #8.

**Options.**
1. Derive the game-level free energy from `local_free_energy`, by passing the prior as the message on the unobserved channel and a Dirac belief on the observed one.
2. Implement `free_energy` and `entropy` per factor.
3. Drop them from the required list, and document `local_free_energy` as the free energy of a factor.

**Resolution so far.** Option 1 for the five `Lenticulum` factors (`GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `NonlinearFactor`, `LikelihoodFactor`): `free_energy` passes the prior and the observation to `Mycelium.local_free_energy`, and since 4a49ca5c `entropy` returns an element of the energy space, not a bare number. `PriorFactor` has exact game-level methods for Dirac and categorical beliefs (bb3af9e9); for a Gaussian belief with an arbitrary `nlogp` they throw, because the expectation has no closed form. The 2026-10-23 resolution also listed a generic `LenticulumCore` fallback and methods for `EqualityFactor`, `DeterministicFactor`, the equilibrium and neural-ODE factors; none of those methods existed at review ([[LP-013 Defects found in the review of master]]). **Correction (2026-10-26):** `EqualityFactor` and `DeterministicFactor` are not types in this
monorepo (`rg` over the Julia sources finds no occurrence); they are not outstanding
implementations. DEQFactor and NeuralODEFactor do exist; the incoming/solved distinction is now
implemented at factor and assembled-lens levels, as recorded below.


**Algebraic family (L-SPEC-004; history through 2026-10-25, superseded below).** `ToricFactor` has exact game-level quantities
when normalised; `PluckerFactor` and `PolynomialFactor` do not. Polynomial
single-branch messages now supply an assembled local lens and Gaussian/GN sites,
but freezing a different metric at each target/iteration does not by itself define
a normalised joint branch model or exact entropy. No game or graph-local free
energy is claimed for it. The full common conformance checker therefore remains
inapplicable; all supported degree-one polarities have independent lens/site/posterior
checks in `lib/AlgebraicFactors.jl/test/polynomial.jl` ([[polynomial]]).


## Decisions and checks, 2026-10-26

- DataFactor and RelayFactor: zero free energy and entropy, matching their declared
  clamped-source/identity accounting. Their pointwise methods now expose the same zero.
- PriorFactor: existing exact Dirac/categorical games retained; pointwise `nlogp(x)` exposed.
- LossFactor: terminal game, all channels supplied as named point/Dirac observations `y`;
  energy is the loss and entropy is zero with the loss's scalar/vector shape. No inversion
  or polarity is claimed.
- OptimiserFactor: stateful reparametrisation/update rule; statistical game quantities are
  undefined and now throw clear `ArgumentError`s.
- LuxFactor: conditional point inversion, constant-zero energy and zero point-belief entropy.
- GeneratorFactor: same zero pair for a Dirac latent observation `y`; NoiseSource: same pair
  conditioned on a fixed Dirac draw `π` with `nsamples=1`, without resampling. Particle entropy is not supplied;
  unsupported game domains throw `ArgumentError`.
- RatioFactor: with a Dirac prior, reweighting preserves the point, so energy is `-logratio`
  and entropy zero. Particle game priors still throw because no entropy reference measure
  has been chosen (LP-006). LP-016 now supplies an extrinsic likelihood potential and
  averages the graph energy contribution under the reweighted factor belief.
- PluckerFactor: trivially extendable forward point game, zero energy/entropy for named Dirac
  input observations. Its existing direct message path passes the common checker without a
  lens. Nonpoint game observations throw. No inverse fibre law is supplied.

Every implemented domain above is tested against `local_free_energy` with option-1 messages
(the unary prior/fixed draw, or prior plus Dirac observation; terminal Loss has all observations
and a trivial prior). Shared checker fixtures use closed forms; package tests also assert the
actual energy/entropy values, including nonzero Prior, Loss and Ratio energies.

## Item 1 resolved: incoming factor and solved-point lens games (2026-10-26)

The maintainer chose two levels. `DEQFactor` and `NeuralODEFactor` now implement option 1:
`π` is a Dirac on the first channel, `y` a Dirac on the second, matching `GaussianFactor`'s
fixed argument order. This is the first-unobserved/second-observed split regardless of the
polarity used for inversion. Factor free energy delegates to `local_free_energy`; entropy
is zero in `ScalarEnergySpace()`. Other beliefs throw with a reason.

For `lens = first(assemble(f, p, ps, st))`, `free_energy(lens, π, inputs, ps, st)` instead
scores the point returned by that polarity's inversion: residual energy plus the prior's
anchor cost. This is Definition 20's expectation under the inversion, with zero point-belief
entropy. A trivial prior has no cost. A hard Dirac prior at a different point costs infinity.
A proper Gaussian anchor costs `½(u*-μ)'Λ(u*-μ)`, relative to its mean (no normalization
constant). A finite channel precision does not soften a Dirac prior. Soft observations,
singular Gaussian anchors and other priors throw. The existing solve/flow inversion is
unchanged; scoring a prior does not turn it into a prior-conditioned posterior.

For a residual affine in the unknown `u`, vanishing at `u*`, the incoming energy equals
the solved residual energy plus the anchor centered at `u₀`, with precision `J_u'J_u`.
`energy_jacobian` computes that precision in `lib/ImplicitLayers.jl/test/conformance.jl`.
For `g=z/2+x`, `x=1,z₀=3` gives root `2`, Jacobian `1/2`, and `½(3-2)²/4=1/8`;
reverse `z=3,x₀=1` gives root `3/2`, Jacobian `-1`, and `½(1-3/2)²=1/8`.
For `dz/dt=0`, endpoints `1,3` give `½(3-1)²=2` in either direction.
The nonlinear test derives a cubic energy gap under the first-order Gauss–Newton
approximation; see [[deq]] and [[point_game]]. These replace the old disagreement assertions
and remove both factors' `game_gap` exemptions. Arbitrary-point scoring remains
`energy`/`scalar_energy` ([[L-ADR-0003 Pointwise energy and free energy are separate levels]]),
not a keyword on `free_energy`.

**Remaining, with reasons (original item numbers retained):**

2. DiffusionFactor: graph `(clamp,score)` uses a denoising surrogate, not posterior entropy;
   its missing-channel heuristic also reconstructs a polarity with a trivial prior. An
   exact Definition 20 entropy/free energy cannot be recovered by wrapping that code.
   Fixed-node quadrature removes randomness but does not supply the missing semantics.
3. Ratio's non-Dirac priors and Generator/NoiseSource/Plucker distributional games need a
   density/entropy law; no new approximation is licensed here.
4. PolynomialFactor has no normalized joint branch model or entropy, and no graph-local
   free energy. Existing independent lens/site/posterior tests remain unchanged.
5. ComposedFactor and TensorFactor have no channels, inversion or evaluated energy/game
   laws. Core tests verify that the common checker cannot treat them as wired factors.
   Composition requires pushforward/downstream expectation; tensor requires joint input and
   belief conventions. Value-level composition helpers do not implement these operations.

The problem remains open for these precise domains. No methods are claimed for fictitious
types, and no arbitrary expectation or posterior-entropy approximation has been added.

## Items 2–5: bounds plus exact cases (2026-10-27)

The maintainer's decision of 2026-10-26 is implemented within the following explicit
scope. This section supersedes the earlier remaining-item descriptions without claiming
the general distributional games are solved.

2. **Diffusion:** `free_energy_upper_bound` evaluates the negative ELBO of an explicitly
   normalised finite VP reverse chain on full joint point evidence. Terminal $N(0,I)$,
   all reverse Gaussian variances, decoder and normalisers are specified by the API;
   the variational law is the forward chain conditioned on evidence. A single analytic
   Gaussian predictor has an exact expectation and returns `FreeEnergyUpperBound(value)`.
   General predictors use `estimate_free_energy_upper_bound`, returning a different
   `FreeEnergyBoundEstimate(value,standard_error,nsamples)`; a sampled estimate need not
   itself lie above negative log evidence. Neither type is a number/ordinary energy.
   `lib/VariationalDiffusion.jl/test/bound.jl` compares with exact Gaussian likelihoods:
   the stationary analytic score saturates the bound, and the gap decreases as a displaced
   score approaches it. The graph `(clamp,score)` surrogate is not a normalised ELBO;
   exact `free_energy`/`entropy` and partial-evidence bounds still throw. See [[bound]].
3. **Generators and source:** `AffineGenerator` with a proper Gaussian latent law and a
   square nonsingular map implements exact Lebesgue entropy
   $H(g_*p)=H(p)+\log|\det A|$. The unconditional forward game has zero map energy and
   free energy $-H(g_*p)$; output conditioning is rejected. `InvertibleGenerator` carries
   a caller-certified global bijection and exact log-Jacobian callback; nonlinear
   Gaussian-source expectations are separately labelled `EntropyEstimate`s, never exact
   game entropies. Dirac/finite supported laws are Lebesgue-singular even if a Jacobian
   expectation can be a finite exact sum. Non-invertible maps throw. The built-in
   standard-normal NoiseSource has exact unconditional entropy without resampling;
   custom samplers still supply no density law. Tests: `lib/Adversarial.jl/test/transport.jl`,
   including Gaussian closed forms, a nonlinear bijection with analytic Jacobian expectation,
   and common conformance for the affine Gaussian game. See [[transport]] and [[generator]].
   **Ratio:** non-Dirac games remain unsupported: arbitrary ratio networks have no analytic
   normaliser/reweighted entropy contract, and empirical weights are not a Lebesgue density.
   **Plucker:** an exact distributional case does exist at $r=2$, with one fixed nonzero
   vector and one proper Gaussian: its scalar image is Gaussian. Entropy/free energy and
   direct-message conformance are checked in `lib/AlgebraicFactors.jl/test/game_exact.jl`.
   Higher-dimensional fixed-vector images are singular; two Gaussian inputs are generally
   non-Gaussian. Jitter is an approximation and is refused by the exact game. See [[plucker]].
4. **Polynomial:** degree one with `branches=:single`, a square nonsingular last-channel
   block, SPD constant residual covariance and a proper pooled posterior defines an exact
   normalised linear-Gaussian conditional. Its normaliser includes the output log-Jacobian.
   Game/graph quantities use the Gaussian grading; `π` is on the first channel and `y`
   names all others. `lib/AlgebraicFactors.jl/test/game_exact.jl` derives the closed form,
   checks both polarities with the common checker, and checks basis/gauge/damping cases.
   Other degrees, branch mixtures and unsolvable normalised output blocks throw. See
   [[polynomial]]; no local metric is promoted to a nonlinear density.
5. **Composition and tensor:** a pair of GaussianFactors integrates out its shared channel,
   giving $A=BA_1$, $b=Ba_1+b_2$, $Q=BQ_1B^\top+Q_2$. Endpoint messages, games, entropy
   and graph contributions match that collapsed Gaussian factor. Both endpoint polarities
   pass common conformance and independent posterior/entropy tests. Independent TensorFactor
   games accept products explicitly as NamedTuples keyed by `parts`; each entry describes
   the corresponding complete channel game. Branch-graded results scalarise to the sum.
   Correlated joint beliefs and undeclared/nonlinear components throw. Tensor graph wiring,
   nested composed chains and composition with noise hyperpriors remain unsupported.
   Tests: `test/game_composition.jl`. See [[composition]] and [[statistical_game]].

The interface amendment is appended to [[L-SPEC-001 The Factor Interface]], not made into
an ADR. The original conditional point games and their conformance coverage remain in place.
The unsupported domains above are deliberate boundaries of this resolution, not hidden
approximations. Verification results for the seven suites and documentation builds are
recorded below after the runs complete.

### Verification, 2026-10-27

All seven package suites pass with the reviewed LP-002/LP-005 work retained. Commands were
`julia --project=<package> -e 'using Pkg; Pkg.test()'` (`--project=.` for Lenticulum), on
Julia 1.13.1, with `OPENBLAS_NUM_THREADS=1` and `JULIA_NUM_PRECOMPILE_TASKS=2`.
Counts sum the top-level test summaries without double-counting their nested rows.
Elapsed times include package setup and precompilation; the top-level and Adversarial
rows are their final reruns.

| package | passing assertions | elapsed seconds |
|---|---:|---:|
| Lenticulum | 994 | 219.49 |
| LenticulumCore | 316 | 46.69 |
| Mycelium | 494 | 11.08 |
| ImplicitLayers | 1195 | 459.73 |
| Adversarial | 315 | 9.37 |
| AlgebraicFactors | 15102 | 163.12 |
| VariationalDiffusion | 1201 | 428.01 |

Total: **19617 passing assertions**, no failures, errors or broken tests. The new
LP-009 testsets contribute 100 (Gaussian composition/product games), 61 (transport),
659 (polynomial), 44 (Plucker) and 58 (diffusion bounds) assertions, respectively.
The first algebraic iteration exposed roundoff accepting an unanchored singular precision;
the final eigenvalue guard is regression-tested. The first top-level iteration exposed a
`pushforward` import collision in the new test; calls now qualify LenticulumCore's binding.

`julia --project=docs docs/make.jl` passed, executing all ten Literate tutorials and producing
26 API HTML pages, with an initial elapsed time of 450.86 seconds. Its vault build also
passed. The standalone `docs/site/build.sh` run produced 251 HTML pages in 8.04 seconds
(including disposable Quartz-copy setup), without KaTeX errors. Both used an explicit
`QUARTZ_DIR` under `/tmp`, with the copy's Git restoration hook removed and its npm cache
kept in that copy; no checkout/clean/reset/stash/commit command was run. The original
`docs/quartz` was not used as a build target. Final rebuild timings are recorded below.

`git diff --check` passes. The initial diff snapshot was compared with the final tree:
all 38 unrelated reviewed tracked-file diffs are unchanged; overlapping loader/test/log
files retain their existing changes. No commit, push or merge was made.

Rebuilds after the API coverage corrections and verification-table update also passed:
API documentation **379.92 seconds**, 26 HTML pages; standalone vault **7.48 seconds**,
251 HTML pages, no KaTeX errors. Documenter reported only its page/search-index size
warnings and skipped deployment in the local environment.
