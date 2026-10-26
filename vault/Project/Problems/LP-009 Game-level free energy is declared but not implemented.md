#open-problem

> **Status: open, item 1 resolved; other semantics remain (2026-10-26). The 2026-10-23 resolution listed methods that do not exist.** Related: [[L-SPEC-001 The Factor Interface]] §4–§5, [[statistical_game]], [[abstract_types]], [[factors]].

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
