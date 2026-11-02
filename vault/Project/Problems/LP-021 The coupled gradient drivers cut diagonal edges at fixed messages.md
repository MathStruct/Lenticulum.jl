#open-problem

> **Status: resolved (2026-11-02), by option 1 below (Daniel's choice).** Found while testing plates ([[L-SPEC-006 Plates]]); measured and now checked in `test/lp021.jl`. Related: [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]], [[LP-011 Gradient couplings and isfrozen are declared but unused]], [[graph_adjoint]], [[gradient_couplings]], [[free_energy]].

> Sources: original repository analysis; code: `lib/Mycelium.jl/src/graph_adjoint.jl` (`_graph_update`, `_graph_loss`), `lib/Mycelium.jl/src/gradient_couplings.jl` (`_coupled_gradient`, `_replay`), `lib/Mycelium.jl/src/free_energy.jl` (`free_energy_gradient`); tests: `test/lp021.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

**Problem.** LP-020 changed what a `DiagonalCoupling` edge means in the default gradient driver:
each factor's term is differentiated at its fixed **belief**, the envelope gradient of the Bethe
free energy. The coupled drivers, which run when any edge is non-diagonal or with
`adjoint = :implicit`, were not changed. There a diagonal edge is still a constant cut of its
**message**, so a factor whose edges are all cut contributes the fixed-message derivative that
LP-020 showed is not the free-energy gradient.

Measured in `test/lp021.jl` on a tree: $z \sim \mathcal N(0, I_2)$, $x = W z + c + \varepsilon$ with $x \in \mathbb R^3$
observed, every parameter at its default initialisation. Against central differences of the
converged free energy:

| driver | edges | max error |
|---|---|---|
| default (`adjoint = :unrolled`, all diagonal) | diagonal | below $10^{-8}$ |
| `adjoint = :implicit` | diagonal | 0.58 |
| `adjoint = :implicit` | `ExactCoupling()` | below $10^{-8}$ |

So `adjoint = :implicit` on a graph with default couplings silently returns a different gradient
from the default driver. The same applies in mixed graphs: one `ExactCoupling` edge sends the
whole graph to the coupled driver, and every factor elsewhere whose edges are all diagonal then
switches from the fixed-belief to the fixed-message derivative. That last case is reasoned from
the code (`_replay` evaluates `_local_scalar` at the replayed messages), not measured.

**Options.**

1. In the coupled drivers, differentiate a factor whose edges are all cut (diagonal or detached)
   at its fixed belief, as the default driver does; factors with a derivative path keep the
   current treatment. A graph with every edge diagonal then gives the default driver's gradient
   under every adjoint. Mixed factors (some edges cut, some not) still need a definition.
2. Refuse `adjoint = :implicit` (and non-diagonal replays) when no edge carries a derivative, and
   document that a diagonal edge in a coupled graph means a stop-gradient on the message.
3. Leave the semantics and document them: in the coupled drivers a cut is a stop-gradient on the
   message, which is a different quantity from the default driver's.

**Resolution.** Option 1. In both coupled drivers, a belief-propagation factor whose edges are all
cut (diagonal or detached) has its term differentiated at its fixed belief: `_frozen_beliefs`
evaluates `factor_belief` at the reference messages (the realised replay's final messages, or the
converged messages of the implicit adjoint) outside AD, and `_graph_scalar` uses
`belief_free_energy` at that belief for those factors. Such a factor's messages are constants in
both drivers (a cut row of the implicit update map, a reference belief in the replay), so this
changes its own term only; the variable terms and the factors with a derivative path are as
before. A factor without `factor_belief` keeps the fixed-message term.

Checked in `test/lp021.jl` against central differences of the converged free energy, all below
$10^{-8}$: `adjoint = :implicit` with every edge diagonal (0.58 before), the same graph with one
`ExactCoupling()` edge (the data edge) under the unrolled replay and the implicit adjoint, and
every edge exact. The existing coupled-gradient tests pass unchanged.

**Still open.** A factor with some edges cut and some not (mixed) keeps the current treatment: its
term is differentiated at its replayed messages, with the cut ones constant. Whether such a cut
should mean "fixed message" or something else was not decided.
