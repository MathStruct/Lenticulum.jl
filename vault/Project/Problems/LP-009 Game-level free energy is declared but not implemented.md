#open-problem

> **Status: open, remaining semantics identified (2026-10-26). The 2026-10-23 resolution listed methods that do not exist.** Related: [[L-SPEC-001 The Factor Interface]] §4–§5, [[statistical_game]], [[abstract_types]], [[factors]].

**Problem.** `LenticulumCore` declares the game-level `entropy(f, π, y, ps, st)` and `free_energy(f, π, y, ps, st)` (AutoBayes Definition 20), and `abstract_types.jl` lists `entropy` among the methods a factor must provide. At the time this problem was opened, no factor implemented either. Every factor that is used in a graph implements Mycelium's `local_free_energy(f, msgs, ps, st)` instead, which takes all incoming messages because a graph has no distinguished prior/observation split. Then `scalar_free_energy(f, π, y, ...)` failed for every factor in the repository, the same kind of gap as issue #8.

**Options.**
1. Derive the game-level free energy from `local_free_energy`, by passing the prior as the message on the unobserved channel and a Dirac belief on the observed one.
2. Implement `free_energy` and `entropy` per factor.
3. Drop them from the required list, and document `local_free_energy` as the free energy of a factor.

**Resolution so far.** Option 1 for the five `Lenticulum` factors (`GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `NonlinearFactor`, `LikelihoodFactor`): `free_energy` passes the prior and the observation to `Mycelium.local_free_energy`, and since 4a49ca5c `entropy` returns an element of the energy space, not a bare number. `PriorFactor` has exact game-level methods for Dirac and categorical beliefs (bb3af9e9); for a Gaussian belief with an arbitrary `nlogp` they throw, because the expectation has no closed form. The 2026-10-23 resolution also listed a generic `LenticulumCore` fallback and methods for `EqualityFactor`, `DeterministicFactor`, the equilibrium and neural-ODE factors; none of these exist ([[LP-013 Defects found in the review of master]]). **Correction (2026-10-26):** `EqualityFactor` and `DeterministicFactor` are not types in this
monorepo (`rg` over the Julia sources finds no occurrence); they are not outstanding
implementations. DEQFactor and NeuralODEFactor do exist, but have the graph/game conflict
recorded below.


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
  conditioned on a fixed Dirac draw `π` with `nsamples=1` with `nsamples=1`, without resampling. Particle entropy is not supplied;
  unsupported game domains throw `ArgumentError`.
- RatioFactor: with a Dirac prior, reweighting preserves the point, so energy is `-logratio`
  and entropy zero. Particle game priors throw: the local incoming-cloud expectation is
  different from the posterior expectation, demonstrated independently in LP-016.
- PluckerFactor: trivially extendable forward point game, zero energy/entropy for named Dirac
  input observations. Its existing direct message path passes the common checker without a
  lens. Nonpoint game observations throw. No inverse fibre law is supplied.

Every implemented domain above is tested against `local_free_energy` with option-1 messages
(the unary prior/fixed draw, or prior plus Dirac observation; terminal Loss has all observations
and a trivial prior). Shared checker fixtures use closed forms; package tests also assert the
actual energy/entropy values, including nonzero Prior, Loss and Ratio energies.

**Remaining, with reasons:**

1. DEQFactor and NeuralODEFactor: their graph methods evaluate residuals at **incoming**
   endpoints, whereas inversion returns a new solved point. For z=z/2+x at x=1,z=3 the
   incoming energy is 1/8 and solved energy zero; for dz/dt=0 at endpoints 1,3 they are 2
   and zero. Both directions are tested; the disagreement is asserted, not marked broken.
   No game methods were added because the requested Dirac-solution interpretation would
   contradict the graph. Entropy zero alone cannot repair the mismatch.
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
