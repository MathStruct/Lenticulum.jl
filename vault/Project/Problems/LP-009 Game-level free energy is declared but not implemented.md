#open-problem

> **Status: open, partly addressed (2026-10-25). The 2026-10-23 resolution listed methods that do not exist.** Related: [[L-SPEC-001 The Factor Interface]] §4–§5, [[statistical_game]], [[abstract_types]], [[factors]].

**Problem.** `LenticulumCore` declares the game-level `entropy(f, π, y, ps, st)` and `free_energy(f, π, y, ps, st)` (AutoBayes Definition 20), and `abstract_types.jl` lists `entropy` among the methods a factor must provide. No factor implements either. Every factor that is used in a graph implements Mycelium's `local_free_energy(f, msgs, ps, st)` instead, which takes all incoming messages because a graph has no distinguished prior/observation split. So `scalar_free_energy(f, π, y, ...)` fails for every factor in the repository, the same kind of gap as issue #8.

**Options.**
1. Derive the game-level free energy from `local_free_energy`, by passing the prior as the message on the unobserved channel and a Dirac belief on the observed one.
2. Implement `free_energy` and `entropy` per factor.
3. Drop them from the required list, and document `local_free_energy` as the free energy of a factor.

**Resolution so far.** Option 1 for the five `Lenticulum` factors (`GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `NonlinearFactor`, `LikelihoodFactor`): `free_energy` passes the prior and the observation to `Mycelium.local_free_energy`, and since 4a49ca5c `entropy` returns an element of the energy space, not a bare number. `PriorFactor` has exact game-level methods for Dirac and categorical beliefs (bb3af9e9); for a Gaussian belief with an arbitrary `nlogp` they throw, because the expectation has no closed form. The 2026-10-23 resolution also listed a generic `LenticulumCore` fallback and methods for `EqualityFactor`, `DeterministicFactor`, the equilibrium and neural-ODE factors; none of these exist ([[LP-013 Defects found in the review of master]]). **Open:** those factors. The DEQ and neural-ODE factors solve for a point, so an exact distributional free energy needs a choice (a Dirac belief, or a stated approximation).
