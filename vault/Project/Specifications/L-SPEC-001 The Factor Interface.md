#design

> **Status: draft (2026-10-22, Claude; describes the code as it is).** Related: [[abstract_types]], [[channels]], [[energy]], [[statistical_game]], [[factors]], [[L-ADR-0002 Energies are vectors, scalarised late]], [[L-ADR-0003 Pointwise energy and free energy are separate levels]]. Code: `lib/LenticulumCore.jl/src/abstract_types.jl`, `channels.jl`, `energy.jl`, `statistical_game.jl`; `lib/Mycelium.jl/src/passing.jl`, `factors.jl`.

## Scope

What a new factor type must implement to be used on its own (as a parameterized statistical game) and in a factor graph (by Mycelium's message passing). It fixes the names, arguments and return types. It does not fix how a factor computes its messages or energies, and it does not cover beliefs (see [[Beliefs]], [[Belief Algebra]]).

## The contract

A factor is a subtype of `AbstractLenticulumFactor`. Like a Lux layer, it is a *description*: parameters `ps` and states `st` are separate values, and every function below takes and returns them.

**§1 Parameters and states** (from LuxCore, unchanged):
- `initialparameters(rng, f)`, `initialstates(rng, f)` — required;
- `parameterlength(f)`, `statelength(f)` — optional.

**§2 Channels and polarities** (`channels.jl`):
- `channels(f)` — the named ports, each a `Channel(name, dim)`; required;
- `supports_polarity(f, p::Polarity)` — which splits into observed, unobserved and latent channels the factor can answer; required;
- `supported_polarities(f)` — the list of them; required for graph use, since polarity resolution enumerates it ([[Polarity Resolution]]);
- `assemble(f, p, ps, st) -> (lens, st)` — the Bayesian lens for a polarity; required unless §5's `factor_message` is implemented directly.

**§3 Energies** (`energy.jl`; two levels, [[L-ADR-0003 Pointwise energy and free energy are separate levels]]):
- `energy(f, x, a, y, ps, st) -> (𝐥, st)` — the pointwise vector energy; required for `scalar_energy`;
- `energyspace(f[, p])`, `scalarisation(f[, p])` — the space and default scalarisation of the free energy; default `ScalarEnergySpace()` and `IdentityScalarisation()`;
- `pointwise_energyspace(f[, p])`, `pointwise_scalarisation(f[, p])` — those of `energy`; default: the free-energy pair;
- the dimension of the value returned by `energy` equals `dimension(pointwise_energyspace(f))`.

**§4 Entropies and free energy** (`statistical_game.jl`):
- `entropy(f, π, y, ps, st) -> (𝐇, st)` and `free_energy(f, π, y, ps, st) -> (𝐅, st)`, both in `energyspace(f)`, are the game-level functions of AutoBayes Definition 20. They are **declared but implemented by no factor yet**: in a graph the free energy comes from §5's `local_free_energy`, which takes all the incoming messages instead of a prior/observation split. `abstract_types.jl` still lists `entropy` as required;
- `scalar_free_energy(f, π, y, ps, st; polarity, σ)` and `scalar_energy(f, x, a, y, ps, st; polarity, σ)` are provided; `σ` overrides the default scalarisation for one call. `scalar_free_energy` therefore works only for a factor that defines `free_energy`.

**§5 In a factor graph** (Mycelium):
- `Mycelium.factor_message(f, target, polarity, inputs, prior, ps, st) -> (belief, st)` — the outgoing belief on `target`. The generic method assembles the lens (§2) and inverts it; structural factors implement it directly;
- `Mycelium.local_free_energy(f, msgs, ps, st) -> (𝐅, st)` — the factor's Bethe contribution in `energyspace(f)`, from the *incoming messages*, not the marginals. Required: the generic method throws, because every factor in a graph must report its free energy ([[factors]]);
- the graph's scalar free energy applies each factor's `scalarisation(f)` to its own summand.

**§6 Learnability** (optional): `islearnable(f)` says whether `ps` are fitted. The parameters of a container factor are a `NamedTuple` keyed by its sub-factors, as in Lux (`AbstractLenticulumContainerFactor`).

## Checked by

- `test/runtests.jl`: "GaussianFactor: the factor interface, nothing stubbed" (§1–§4 for the reference factor); "pointwise energy and free energy: one contract" (§3, for four factors);
- `lib/LenticulumCore.jl/test/runtests.jl` (the energy algebra of §3–§4);
- `lib/Mycelium.jl/test/runtests.jl` (§5 on structural factors).

No single test yet runs §1–§5 against every factor type; [[LP-008 No conformance test for the factor interface]] tracks it.

## Open

- Polarity-dependent energies are allowed by §3 but no factor has one.
- §4 versus §5: whether the game-level `free_energy` should be implemented per factor, derived from `local_free_energy`, or dropped from the required list ([[LP-009 Game-level free energy is declared but not implemented]]).
- The Jacobian of the vector energy is not part of the interface yet ([[LP-002 The Jacobian side is not implemented]]).
