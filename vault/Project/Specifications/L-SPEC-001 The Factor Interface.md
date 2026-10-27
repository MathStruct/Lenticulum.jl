#design

> **Status: draft (2026-10-22, Claude; updated 2026-10-23 on `review-argon`; coverage updated 2026-10-27).** Related: [[abstract_types]], [[channels]], [[energy]], [[statistical_game]], [[factors]], [[L-ADR-0002 Energies are vectors, scalarised late]], [[L-ADR-0003 Pointwise energy and free energy are separate levels]], [[L-ADR-0006 Separate likelihood_message from invert]]. Code: `lib/LenticulumCore.jl/src/abstract_types.jl`, `channels.jl`, `energy.jl`, `statistical_game.jl`; `lib/Mycelium.jl/src/passing.jl`, `factors.jl`.

## Scope

What a new factor type must implement to be used on its own (as a parameterized statistical game) and in a factor graph (by Mycelium's message passing). It fixes the names, arguments and return types. It does not fix how a factor computes its messages or energies, and it does not cover beliefs (see [[Beliefs]], [[Belief Algebra]]).

## The contract

A factor is a subtype of `AbstractLenticulumFactor`. Like a Lux layer, it is a *description*: parameters `ps` and states `st` are separate values, and every function below takes and returns them.

**§1 Parameters and states** (from LuxCore, unchanged):
- `initialparameters(rng, f)`, `initialstates(rng, f)` — required;
- `parameterlength(f)`, `statelength(f)` — optional.

**§2 Channels and polarities** (`channels.jl`, `bayesian_lens.jl`):
- `channels(f)` — the named ports, each a `Channel(name, dim)`; required;
- `supports_polarity(f, p::Polarity)` — which splits into observed, unobserved and latent channels the factor can answer; required;
- `supported_polarities(f)` — the complete list of accepted default-precision Observed/Unobserved/Latent assignments when listable; a factor may throw an informative error above a documented cap;
- `polarities(f)` — lazy traversal filtered by the authoritative `supports_polarity` predicate; heterogeneous assignment types make it type-unstable;
- `hasnopolarity(f)`, `isunidirectional(f)` — structural zero/one-support predicates; defaults inspect the list, capped factors must override both. Mycelium's `issink` delegates to the former. Polarity resolution constructs each per-target assignment from available messages and checks the predicate, without enumeration ([[Polarity Resolution]]);
- `assemble(f, p, ps, st) -> (lens, st)` — the Bayesian lens for a polarity; required unless §5's `factor_message` is implemented directly;
- `invert(lens, π, inputs, ps, st) -> (posterior, st)` — the Bayesian inversion $c^\dagger_\pi(y)$ of AutoBayes Definition 10 (posterior including the prior $\pi$);
- `likelihood_message(lens, π, inputs, ps, st) -> (msg, st)` — the cavity-free BP likelihood message $\mu_{c\to\text{target}}$ (using $\pi$ at most as an expansion point; defaults to `invert` for point/likelihood lenses, [[L-ADR-0006 Separate likelihood_message from invert]]).

**§3 Energies** (`energy.jl`; two levels, [[L-ADR-0003 Pointwise energy and free energy are separate levels]]):
- `energy(f, x, a, y, ps, st) -> (𝐥, st)` — the pointwise vector energy; required for `scalar_energy`;
- `energyspace(f[, p])`, `scalarisation(f[, p])` — the space and default scalarisation of the free energy; default `ScalarEnergySpace()` and `IdentityScalarisation()`;
- `pointwise_energyspace(f[, p])`, `pointwise_scalarisation(f[, p])` — those of `energy`; default: the free-energy pair;
- the dimension of the value returned by `energy` equals `dimension(pointwise_energyspace(f))`;
- `energy_jacobian(f, x, a, y, ps, st; wrt, ε)` and `gauss_newton_metric(f, x, a, y, ps, st; wrt, polarity, σ, ε)` provide the finite-difference Jacobian $J$ of `energy` and the Gauss–Newton pullback metric $J^\top M J$ when `pointwise_scalarisation` is a `SquaredNorm(M)`.

**§4 Entropies and free energy** (`statistical_game.jl`):
- `entropy(f, π, y, ps, st) -> (𝐇, st)` and `free_energy(f, π, y, ps, st) -> (𝐅, st)` are the game-level functions of AutoBayes Definition 20 (both in `energyspace(f)`: `entropy` is a vector too, e.g. `(fit = 0, complexity = 0, negentropy = H)` in the graded space, and `free_energy` subtracts it). Implemented for the five `Lenticulum` factors and exact Dirac/categorical `PriorFactor`
  games; also Data/Relay/Lux constant-zero games, Loss terminal point observations, fixed-Dirac
  Generator/NoiseSource/Plucker games, and Dirac-prior Ratio games. Optimiser explicitly throws.
  DEQ/NeuralODE Dirac factor games score incoming points (`π` on the first channel, `y` on
  the second), with scalar zero entropy, matching graph energy. Their assembled-lens games
  score the inversion output plus a hard Dirac or proper Gaussian anchor cost, or no cost
  for a trivial prior; this is their Definition 20 expectation. Gaussian cost is relative
  to its mean; soft observations and singular anchors are unsupported. `assemble(f,p)`
  restricts the factor relation to one polarity. Arbitrary points use `energy`/`scalar_energy`,
  not a keyword on `free_energy` (L-ADR-0003). Diffusion's score surrogate has no posterior
  entropy; the exact polynomial/container cases are specified in the 2026-10-27 amendment below ([[LP-009 Game-level free energy is declared but not implemented]]);
- `scalar_free_energy(f, π, y, ps, st; polarity, σ)` and `scalar_energy(f, x, a, y, ps, st; polarity, σ)` are provided; `σ` overrides the default scalarisation for one call.

**§5 In a factor graph** (Mycelium):
- `Mycelium.factor_message(f, target, polarity, inputs, prior, ps, st) -> (belief, st)` — the outgoing likelihood belief on `target`. The generic method assembles the lens (§2) and calls `LenticulumCore.likelihood_message`; structural factors implement it directly;
- `Mycelium.local_free_energy(f, msgs, ps, st) -> (𝐅, st)` — the factor's Bethe contribution in `energyspace(f)`, from the *incoming messages*, not the marginals. Required: the generic method throws, because every factor in a graph must report its free energy ([[factors]]);
- the graph's scalar free energy applies each factor's `scalarisation(f)` to its own summand.

**§6 Learnability** (optional): `islearnable(f)` says whether `ps` are fitted. The parameters of a container factor are a `NamedTuple` keyed by its sub-factors, as in Lux (`AbstractLenticulumContainerFactor`).

## Checked by

- `test/factor_conformance.jl`: `check_factor_conformance` ([[LP-008 No conformance test for the factor interface]]), run from `test/runtests.jl` over every supported polarity of `GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `NonlinearFactor`, and `LikelihoodFactor`); "GaussianFactor: the factor interface, nothing stubbed" (§1–§4 for the reference factor); "pointwise energy and free energy: one contract" (§3, for four factors);
- `lib/LenticulumCore.jl/test/runtests.jl` (the energy algebra, `SquaredNorm` validation, `energy_jacobian`, and `gauss_newton_metric` of §3–§4);
- `lib/Mycelium.jl/test/runtests.jl` (§5 on structural factors, including non-zero `PriorFactor` `local_free_energy`);
- `lib/ImplicitLayers.jl/test/conformance.jl`: common default lens path for Lux forward and
  DEQ/NeuralODE both directions, including full incoming game checks, solved-point lens
  games, linear identity and nonlinear Gauss–Newton error tests.
- `lib/Mycelium.jl/test/conformance.jl`: common direct-message path on Data, Prior, Optimiser
  unary and Relay both directions; Loss sink path, with explicit unsupported-section reasons.
- `lib/Adversarial.jl/test/conformance.jl`: common default lens path on NoiseSource unary,
  Generator forward, Ratio unary; conditional point games, fixed draw and LP-016 likelihood/particle graph checks.
- `lib/VariationalDiffusion.jl/test/conformance.jl`: common default lens path on all five binary
  assignments and unary, with fixed-node Gaussian/point fixtures and an explicit game gap.
  "Diffusion polarity enumeration and cap (LP-017)" and "Joint diffusion messages (LP-017)"
  check complete support and joint point/Gaussian/mixture messages; the four LP-017 broken
  assertions now pass.
- `lib/AlgebraicFactors.jl/test/runtests.jl`: direct-message common checker for Toric at arities
  one to three and Plucker forward point games at r=2,4. Degree-one Polynomial additionally
  uses the common lens/game checker with independent Gaussian fixtures in `test/game_exact.jl`; nonlinear branch laws remain unsupported.
- `test/game_composition.jl`: common checker for collapsed Gaussian composition in both
  endpoint polarities and soft Gaussian games, plus independent/nested tensor product games.
  Core "Container conformance scope (LP-008/009)" verifies that unsupported component laws
  still reject games and do not invent channels or assembled inversions.

The checker is included from the single repository `test/factor_conformance.jl` file.
Its default checks are retained. Nonempty `pointwise_gap`, `game_gap`, `message_gap`, `polarity_gap` reasons
omit only the named section, and `p=nothing` represents a sink. Game/graph checks compare
energy coordinates as well as scalarisations. A passing partial run is not a claim of a
missing lens, distributional entropy or general BP law.

## Open

- Polarity-dependent energies are allowed by §3 but no factor has one.
- Optional DifferentiationInterface parameter Jacobians/gradients and parameter pull-back metrics are implemented; finite differences remain the reference. See [[LP-002 The Jacobian side is not implemented]] for tested paths and remaining scope.

## Bounds and exact cases, 2026-10-27 (LP-009 amendment to §4–§5)

This dated amendment supersedes the coverage statements above for the domains it names.
The maintainer's 2026-10-26 decision is **bounds plus exact cases**. New bounds do not
implement exact `free_energy` and must not flow through `scalar_free_energy` or `scalarise`.

- `FreeEnergyUpperBound(value)` represents an analytically evaluated scalar upper bound
  on a specified normalised model's negative log evidence. It is neither `Real` nor an
  energy-space element; reading `.value` is explicit.
- `FreeEnergyBoundEstimate(value,standard_error,nsamples)` is a Monte Carlo estimate of
  that expectation. Its sampled value is not itself an upper bound. The producing API
  specifies the model, variational law, evidence domain and reference measure.
- `free_energy_upper_bound(f,x,ps,st;times)` and
  `estimate_free_energy_upper_bound(f,x,ps,st;times,rng,nsamples)` are separate extension
  points. DiffusionFactor implements a finite VP Gaussian reverse chain with terminal
  $N(0,I)$, reverse variance $\beta_i I$ (including its decoder), the standard epsilon
  reverse mean, and forward-chain variational law conditioned on full joint evidence.
  Exact expectation requires its analytic single-Gaussian predictor. Arbitrary predictors
  return estimates only. Partial evidence and exact prox/game entropy remain unsupported.
  The existing graph surrogate is not a bound on this model; [[bound]] defines the model.

Continuous entropy is relative to Lebesgue coordinates (LP-006). GeneratorFactor now has
an exact Gaussian forward game for an `AffineGenerator` with identity input adapter and
square nonsingular matrix: `y` is the latent Gaussian, `π` is absent/trivial, entropy is
$H(p)+\log|\det A|$, free energy its negative. Rectangular/singular maps throw. Nonlinear
caller-certified bijections expose only `EntropyEstimate` through
`estimate_pushforward_entropy`; no estimated value enters exact game APIs. Dirac and finite
supported laws are Lebesgue-singular, so an exact finite Jacobian sum does not supply their
differential entropy. Conditional point conventions are unchanged. The built-in
standard-normal NoiseSource has exact unconditional entropy for `π=TrivialBelief()` and
no observation. Custom samplers supply no density law. [[transport]] states the full domain.

PluckerFactor has an additional exact forward game at $r=2$ with one Dirac and one proper
Gaussian input, positive scalar output variance, no output conditioning and zero covariance
jitter. Two Gaussian inputs and higher-dimensional fixed-vector images remain unsupported.
Ratio non-Dirac game priors remain unsupported: arbitrary ratio networks supply neither an
analytic normaliser nor reweighted entropy. Empirical Shannon entropy is not substituted.

PolynomialFactor has an exact degree-one, single-branch game when the last-channel
coefficient block is square nonsingular, residual covariance is positive definite and the
pooled joint posterior is proper. The last channel specifies the normalised conditional;
its output log-Jacobian enters the normaliser. The game uses `π` on the first channel and
a NamedTuple `y` for all remaining channel beliefs. Nonlinear or singular conditional
cases throw with reasons. Game and graph quantities share the Gaussian grading.

ComposedFactor of two GaussianFactors integrates out the shared channel. Its endpoint
channels, energy space, entropy and game/graph free energy are those of the collapsed
linear-Gaussian factor with $A=BA_1$, $b=Ba_1+b_2$ and $Q=BQ_1B^\top+Q_2$.
This marginal game differs from the generic `compose_*` path-energy pairs. Dimensions and
shared names must match; component noise hyperpriors are unsupported. Other compositions,
including nested chains, throw. TensorFactor accepts independent products only: `π` and `y`
are NamedTuples keyed by `parts`, describing each component's complete channels. It returns
branch-graded game energies/entropies and component states; scalarisation sums contributions.
Only declared exact linear-Gaussian components (and their independent products) are accepted.
Correlated joint beliefs are rejected. Tensor graph wiring/inversion remains absent.

Checked by `test/game_composition.jl`, `lib/VariationalDiffusion.jl/test/bound.jl`,
`lib/Adversarial.jl/test/transport.jl`, and `lib/AlgebraicFactors.jl/test/game_exact.jl`.
The common checker covers Gaussian composition in both directions, affine Gaussian
forward generators, the exact scalar Plucker image and degree-one polynomial polarities.
The common checker's `scope=:bound` checks evaluated bounds against independent exact
Gaussian reverse-chain likelihoods, separately from the graph surrogate. `scope=:game`
checks tensor product games and unconditional standard-normal NoiseSource entropy against
independent energy/entropy fixtures, without claiming wiring or repeatable stochastic
messages. Both scopes require their fixtures explicitly. The default `scope=:wired` retains
all existing checks; optional expected game coordinates also check independent closed forms.
The audited rollout table and deliberately retained exemptions are in [[LP-008 No conformance test for the factor interface]].

## LP-011 factor addition (2026-10-27)

`Mycelium.StraightThroughFactor` is a new parameterless directed structural factor.
Its pointwise energy, game free energy and entropy are zero, and its forward message
is a Dirac at the forward callable's value. Its separate surrogate derivative is
specified in [[L-SPEC-002 Stop Gradient]]. The shared checker now exercises this
factor in `lib/Mycelium.jl/test/conformance.jl`, with independent round-value and
zero-cost fixtures. This extends the earlier LP-008 audit without changing its history.
