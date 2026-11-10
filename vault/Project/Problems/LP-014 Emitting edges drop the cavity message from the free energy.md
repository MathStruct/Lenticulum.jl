#open-problem

> **Status: closed (2026-10-26), option 2.** Found while checking `AlgebraicFactors.ToricFactor` ([[L-SPEC-004 Algebraic Factors Toric and Grassmannian]]). Related: [[free_energy]], [[toric]], [[L-SPEC-001 The Factor Interface]]. Code: `lib/Mycelium.jl/src/free_energy.jl` (`messages_into`, `factor_free_energies`), `lib/Mycelium.jl/src/factors.jl` (`PriorFactor`).

> Sources: original analysis; code: `lib/Mycelium.jl/src/factors.jl`, `free_energy.jl`,
> `lib/Mycelium.jl/test/runtests.jl`, `lib/AlgebraicFactors.jl/test/runtests.jl`.
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Closure (2026-10-26)

Option 2 is implemented with the public trait `Mycelium.uses_cavity(factor)`, default `false`.
Factors whose `local_free_energy` reads incoming messages override it as `true` in their own
package. `factor_free_energies` checks their edges with `absorbs(edge.direction)` and throws
`ArgumentError` for an `Emitting()` edge, naming the factor, channel and variable and asking
for `Bidirectional()`. Both graph `scalar_free_energy` and `bethe_free_energy` use that path.
A stored `nothing` on a permitted edge is unchanged. Message passing and schedules are unchanged.

Mycelium's "LP-014: cavity messages and edge directions" checks the trait values, all three
graph entry points, diagnostic, permitted missing messages, and a directed data/relay/loss
graph's unchanged free energy. AlgebraicFactors' "LP-014: a prior on an Emitting edge" replaces
the original `@test_broken` with `@test_throws ArgumentError`, keeps exact marginals checked
against enumeration in both directions, and checks the Bidirectional free energy against
$-\log Z$. Its "tree BP and Bethe partition function" retains the larger tree checks.

## History: original report (2026-10-25)

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
