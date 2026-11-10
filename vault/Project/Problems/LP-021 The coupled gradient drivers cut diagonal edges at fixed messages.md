#open-problem

> **Status: resolved (2026-11-02), by option 1 below (Daniel's choice); the mixed case resolved 2026-11-04 by option 3 of the follow-up (Daniel's choice).** Found while testing plates ([[L-SPEC-006 Plates]]); measured and now checked in `lib/Lenticulum.jl/test/lp021.jl`. Related: [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]], [[LP-011 Gradient couplings and isfrozen are declared but unused]], [[graph_adjoint]], [[gradient_couplings]], [[free_energy]].

> Sources: original repository analysis; code: `lib/MyceliumLearning.jl/src/graph_adjoint.jl` (`_graph_update`, `_graph_loss`), `lib/MyceliumLearning.jl/src/gradient_couplings.jl` (`_coupled_gradient`, `_replay`), `lib/Mycelium.jl/src/free_energy.jl` (`free_energy_gradient`); tests: `lib/Lenticulum.jl/test/lp021.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

**Problem.** LP-020 changed what a `DiagonalCoupling` edge means in the default gradient driver:
each factor's term is differentiated at its fixed **belief**, the envelope gradient of the Bethe
free energy. The coupled drivers, which run when any edge is non-diagonal or with
`adjoint = :implicit`, were not changed. There a diagonal edge is still a constant cut of its
**message**, so a factor whose edges are all cut contributes the fixed-message derivative that
LP-020 showed is not the free-energy gradient.

Measured in `lib/Lenticulum.jl/test/lp021.jl` on a tree: $z \sim \mathcal N(0, I_2)$, $x = W z + c + \varepsilon$ with $x \in \mathbb R^3$
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

Checked in `lib/Lenticulum.jl/test/lp021.jl` against central differences of the converged free energy, all below
$10^{-8}$: `adjoint = :implicit` with every edge diagonal (0.58 before), the same graph with one
`ExactCoupling()` edge (the data edge) under the unrolled replay and the implicit adjoint, and
every edge exact. The existing coupled-gradient tests pass unchanged.

**The mixed case (2026-11-04).** A factor with some edges cut and some differentiated. Measured on
the graph above and on a chain $z_1 \to z_2 \to x$ (scratch scripts, then checked in `lib/Lenticulum.jl/test/lp021.jl`),
maximum error against central differences, gradients of size 1.1–1.5:

| graph | fixed cut messages (the replay) | any cut factor at its belief |
|---|---|---|
| `lik`: $z$ exact, $x$ cut; prior's $z$ edge cut | 0.58 | 0.38 |
| `lik`: $z$ exact, $x$ cut; prior's $z$ edge exact | $5 \cdot 10^{-10}$ | 0.58 |
| `lik`: $z$ cut, $x$ exact; data edge exact | 0.58 | $5 \cdot 10^{-10}$ |
| chain: $z_1$'s edges cut, $z_2$'s exact | 0.29 | 0.68 |
| chain: $z_1$'s exact, $z_2$'s cut | 0.18 | 0.29 |

Neither local rule is right: each is exact only where the mixed factor is not really mixed (its cut
or its differentiated edge carries a message that does not depend on the parameters, here the
observed $x$). The free-energy gradient is either every factor at its fixed belief or every message
differentiated; at a fixed point both are the envelope gradient, and a mixture keeps terms that
cancel only when all messages are differentiated.

Options: (1) keep fixed messages and document a cut as a stop-gradient; (2) hold any factor with a
cut at its belief (ruled out by the table); (3) let the couplings not change the free-energy
gradient at a fixed point, so that they matter only in a replay that differentiates a truncated
computation, and for losses other than the free energy; (4) refuse mixed graphs.

**Resolution of the mixed case: option 3** (Daniel's choice). `free_energy_gradient` without
`schedule` gives the envelope gradient at the store whatever the couplings (diagonal, exact,
pathwise; before, non-diagonal edges required a replay). `adjoint = :implicit` on a mixed graph
returns the envelope gradient at its converged fixed point (`_mixed_cuts`: a cut and a
differentiated edge, both carrying parameter-dependent messages; edges of observed variables do not
count). A replay (`schedule`) still differentiates the replayed computation, and on a mixed graph it
warns that this is not the free-energy gradient. Score-function couplings and straight-through
factors keep their replay semantics. Checked in `lib/Lenticulum.jl/test/lp021.jl` ("mixed cuts give the envelope
gradient"): three mixed chains, below $10^{-8}$ without `schedule` and with `adjoint = :implicit`,
the replay warned and more than 0.1 off.
