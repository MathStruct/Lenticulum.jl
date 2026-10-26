#design

> **Status: draft (2026-10-22, Claude; updated 2026-10-23 on `review-argon`; coverage updated 2026-10-26).** Related: [[abstract_types]], [[channels]], [[energy]], [[statistical_game]], [[factors]], [[L-ADR-0002 Energies are vectors, scalarised late]], [[L-ADR-0003 Pointwise energy and free energy are separate levels]], [[L-ADR-0006 Separate likelihood_message from invert]]. Code: `lib/LenticulumCore.jl/src/abstract_types.jl`, `channels.jl`, `energy.jl`, `statistical_game.jl`; `lib/Mycelium.jl/src/passing.jl`, `factors.jl`.

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
- `supported_polarities(f)` — the list of them; required for graph use, since polarity resolution enumerates it ([[Polarity Resolution]]);
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
  DEQ/NeuralODE graph residuals disagree with solved-point energy; diffusion's score surrogate
  has no posterior entropy; polynomial and container games remain undefined ([[LP-009 Game-level free energy is declared but not implemented]]);
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
  DEQ/NeuralODE both directions; explicit game gaps and broken conflict tests.
- `lib/Mycelium.jl/test/conformance.jl`: common direct-message path on Data, Prior, Optimiser
  unary and Relay both directions; Loss sink path, with explicit unsupported-section reasons.
- `lib/Adversarial.jl/test/conformance.jl`: common default lens path on NoiseSource unary,
  Generator forward, Ratio unary; conditional point games, fixed draw and LP-016 broken tests.
- `lib/VariationalDiffusion.jl/test/conformance.jl`: common default lens path on both enumerated
  binary polarities, the two latent/single-target cases and the multiple-target case, plus
  unary: fixed-node Gaussian/point fixtures, explicit game/enumeration/joint-message gaps
  and four broken LP-017 assertions.
- `lib/AlgebraicFactors.jl/test/runtests.jl`: direct-message common checker for Toric at arities
  one to three and Plucker forward point games at r=2,4. Polynomial keeps its independent
  per-polarity subset tests; full conformance is not claimed.
- Core "Container conformance scope (LP-008/009)" checks ComposedFactor/TensorFactor's
  missing channel/game APIs; the common checker is inapplicable.

The checker is included from the single repository `test/factor_conformance.jl` file.
Its default checks are retained. Nonempty `pointwise_gap`, `game_gap`, `message_gap`, `polarity_gap` reasons
omit only the named section, and `p=nothing` represents a sink. Game/graph checks compare
energy coordinates as well as scalarisations. A passing partial run is not a claim of a
missing lens, distributional entropy or general BP law.

## Open

- Polarity-dependent energies are allowed by §3 but no factor has one.
- AD package extensions for parameter-space Jacobians $D_\theta \mathbf F$ and closed-form factor Jacobian overrides remain open ([[LP-002 The Jacobian side is not implemented]]).
