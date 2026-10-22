#open-problem

> **Status: open (2026-10-22).** Related: [[L-SPEC-001 The Factor Interface]] §4–§5, [[statistical_game]], [[abstract_types]], [[factors]].

**Problem.** `LenticulumCore` declares the game-level `entropy(f, π, y, ps, st)` and `free_energy(f, π, y, ps, st)` (AutoBayes Definition 20), and `abstract_types.jl` lists `entropy` among the methods a factor must provide. No factor implements either. Every factor that is used in a graph implements Mycelium's `local_free_energy(f, msgs, ps, st)` instead, which takes all incoming messages because a graph has no distinguished prior/observation split. So `scalar_free_energy(f, π, y, ...)` fails for every factor in the repository, the same kind of gap as issue #8.

**Options.**
1. Derive the game-level free energy from `local_free_energy`, by passing the prior as the message on the unobserved channel and a Dirac belief on the observed one.
2. Implement `free_energy` and `entropy` per factor.
3. Drop them from the required list, and document `local_free_energy` as the free energy of a factor.

**Leaning.** Option 1 as a generic method, which makes the game-level interface work everywhere `local_free_energy` does, with a test that it agrees with AutoBayes' chain rule on the linear-Gaussian chain.
