#open-problem

> **Status: open (2026-10-25).** Found while checking `AlgebraicFactors.ToricFactor` ([[L-SPEC-004 Algebraic Factors Toric and Grassmannian]]). Related: [[free_energy]], [[toric]], [[L-SPEC-001 The Factor Interface]]. Code: `lib/Mycelium.jl/src/free_energy.jl` (`messages_into`, `factor_free_energies`), `lib/Mycelium.jl/src/factors.jl` (`PriorFactor`).

**Problem.** The Bethe factor belief is $b_a \propto f_a \prod_{i\in a}\mu_{i\to a}$. `messages_into` reads
$\mu_{i\to a}$ from the store, and an `Emitting()` edge carries no variable-to-factor message, so
it returns `nothing` for that channel. `PriorFactor.local_free_energy` then uses its own belief
alone as the factor belief, instead of the prior combined with the cavity message. The marginals
are unaffected (the outgoing message is right), but the Bethe free energy is wrong, silently.

On a tree of `ToricFactor`s with a categorical `PriorFactor` on one leaf, the free energy
matches $-\log Z$ by enumeration when the prior's edge is `Bidirectional()`, and differs by an
amount of order one when it is `Emitting()`; the testset "tree BP and Bethe partition function"
in `lib/AlgebraicFactors.jl/test/runtests.jl` records both (the second as `@test_broken`). A
`DataFactor` on an `Emitting()` edge is not affected: its belief is a Dirac either way.

Connecting a prior with `Emitting()` is the natural way to write it, and nothing warns.

**Options.**
1. Let Mycelium send variable-to-factor messages along every edge for the free-energy
   computation, whatever its direction (the direction then constrains scheduling only).
2. Throw (or warn) in `factor_free_energies` when a factor whose free energy depends on the
   cavity sits on an edge without a variable-to-factor message.
3. Document that free energies need `Bidirectional()` edges for every non-Dirac factor.

**Leaning.** Option 2 now, so the error is not silent; option 1 if direction is meant to be a
scheduling hint only.
