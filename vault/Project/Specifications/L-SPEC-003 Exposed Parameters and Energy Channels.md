#design

> **Status: §1 partly implemented (2026-10-30): §1.1 generically, §1.2 exactly for `GaussianFactor` (`A`, `b`) with a point input, §1.3 by wiring; §2–§3 planned.** Planned 2026-10-25 (Daniel: "If the library is extensible in this direction one should plan for it"). Related: [[Unusual Wirings]] (the theory), [[L-SPEC-001 The Factor Interface]], [[L-SPEC-002 Stop Gradient]], [[Everything is a Factor]], [[The Inferencer and the Optimizer]] §7, [[RQ-007 Risk Scalarisations and Local Decomposition]], [[LP-005 Learning the noise covariance]]. Code: `lib/LenticulumCore.jl/src/channels.jl`, `energy.jl`, `statistical_game.jl`; `lib/Mycelium.jl/src/factors.jl` (`OptimiserFactor`, `PriorFactor`), `graph.jl`, `free_energy.jl`.

## Scope

Three extensions of [[L-SPEC-001 The Factor Interface]] that let a graph wire an output into a
weight, an energy into an input, and keep beliefs over weights and energies ([[Unusual Wirings]]).
None is built. This note records the intended shape, so that work on the existing interface does
not close the door on them; the contract below is a proposal, and each part becomes binding only
when it is implemented with its tests.

## Today

| capability | state |
|---|---|
| a parameter as a variable | only by writing a factor whose weight *is* a channel; `OptimiserFactor` and `PriorFactor` then attach to it. No generic way to expose a block of `ps` |
| sharing a weight between factors | no tying interface: `ps` is per factor ([[The Inferencer and the Optimizer]] §7.1); two entries can alias one array, but shared gradients and updates are not handled |
| an energy as a value another factor reads | no generic tap: a custom factor can call `energy`/`scalar_energy` of another, but no wiring exposes an existing factor's energy as a channel |
| a belief over an energy | no standard API: factors return evaluated, expected or estimated quantities (Gaussian `fit` is an expectation, DEQ energies are evaluated at points, the diffusion factor's score term is Monte Carlo), never the distribution |
| scalarisation of an energy belief | no: `scalarise(σ, e)` takes a value in an energy space |

## The proposed contract

**§1 Exposure.** `expose(f, names::Tuple)` returns a factor whose channels are `channels(f)` plus
one channel per parameter block in `names` (its space the block's shape), and whose `ps` lack those
blocks.
- §1.1 With every exposed channel **hard-clamped** (a `DiracBelief`), it behaves exactly as `f` with those blocks
  substituted: messages, energies and free energies are `f`'s. (Testable against `f` for every
  factor; this part needs nothing from `f`.) An observed channel with finite precision is an
  uncertain weight and falls under §1.2.
- §1.2 With an exposed channel **unobserved**, the factor needs the message to that channel. A
  factor provides it by an optional method `parameter_message(f, name, polarity, msgs, ps, st)`, which
  returns a likelihood message (the prior on the weight is a separate factor, as for any variable,
  [[L-ADR-0006 Separate likelihood_message from invert]]); for the linear-Gaussian factor in $A$ with a
  point $x$ it is an exact, possibly improper, Gaussian kernel (the factor is linear in $A$). Without the method,
  a generic Laplace message by AD may be used, and its docstring and the polarity say it is an
  approximation. Beliefs on both $x$ and $A$ make the factor bilinear: mean-field (VMP) or EP, stated.
- §1.3 Two exposed channels connected to one variable share the weight; this replaces
  name-keyed `ps` for weight tying.

**§1 as implemented (2026-10-30).** `Mycelium.expose(f, names)` returns an `ExposedFactor`
(`lib/Mycelium.jl/src/exposure.jl`): a wrapper, which settles the first Open question below.
A block must be an array; its channel carries `vec(block)`. The optional methods are
`supports_parameter_polarity(f, polarity)`, `parameter_message(f, target, polarity, msgs, ps, st)`
and `parameter_free_energy(f, msgs, ps, st)`, with polarities and messages keyed by `f`'s
channels and the parameter names. Two changes from the text above: `parameter_message` also
sends to `f`'s own channels when a block carries a belief (the posterior predictive needs it),
and §1.2 needed a free-energy counterpart, `parameter_free_energy`, for the Bethe sum. No
generic Laplace message is provided: without the methods, a belief on an exposed block throws.
`GaussianFactor` implements them exactly for `A` and/or `b` with a point input ``x_0``
(`src/exposure.jl`): ``r = y - Ax_0 - b`` is linear in ``(\operatorname{vec}A, b, y)`` through
``x_0^\top \otimes I``; a belief on ``x`` as well is bilinear and throws. Game-level methods
(`assemble`, `energy`, `free_energy`) are not forwarded. A clamped exposed channel should be
wired `Absorbing()`, like data: a message back to it may be bilinear.

**§2 Energy taps.** `tap(f)` returns a factor with an extra channel `:energy` in
`pointwise_energyspace(f)`.
- §2.1 Unobserved: the message is the energy's pushforward under the factor belief; exact for a
  Dirac belief, otherwise by samples or a stated approximation.
- §2.2 Observed with finite precision $\tau$: the tap owns a quadratic penalty $\tfrac\tau2\lVert E_f - e\rVert^2$, a
  term of its own in the free energy beside $f$'s; with infinite precision, a level-set constraint,
  supported only where a disintegration is available (as for `LinearConstraintFactor`), whose
  reference measure and entropy treatment the implementation must state.
- §2.3 Gradients: by default no cotangent flows back through `:energy` into `f` (a stop-gradient,
  [[L-SPEC-002 Stop Gradient]]); a user may switch it on.
- §2.4 Bethe accounting: the tap contributes `f`'s free-energy term once, plus §2.2's penalty when
  `:energy` is observed. A factor that reads `:energy` adds its own term; the sum must not contain
  $E_f$ twice unless the reader says it reweights it. That an unobserved, unread tap leaves the Bethe
  free energy unchanged is a property to prove (the deterministic $e = E_f(x)$ makes the joint
  singular with respect to Lebesgue measure, so the entropy convention matters), and to test.

**§3 Energy beliefs and risk scalarisations.**
- §3.1 Two different objects, not to be conflated: the **pushforward** of the factor belief under
  the pointwise energy (a belief over `pointwise_energyspace(f)`, e.g. the residual $r$, whose mean is
  not `fit`: for $r \sim \mathcal N(0, 1)$, $\mathbb E r = 0$ but `fit` $= \tfrac12$), and the law of the **scalar loss** whose
  expectation is a free-energy coordinate (e.g. $\tfrac12 r^\top Q^{-1} r$, mean `fit`). Entropy coordinates
  are functionals of the belief, not means of a random energy, and have no such law. An optional
  `energy_belief(f, msgs, ps, st)` returns the first; risk scalarisations act on the second.
- §3.2 Scalarisations that act on such a belief: `Expectation()` (today's semantics),
  `EntropicRisk(β)`, `CVaR(α)`. Each documents when it is finite (the entropic risk needs a finite
  positive exponential moment, which an energy with a heavier-than-exponential right tail lacks) and how it composes over factors
  ([[RQ-007 Risk Scalarisations and Local Decomposition]]).

## Keep open meanwhile

Changes to the current interface should not make these harder:

- a `Channel`'s space may be matrix-valued or an energy space, not only a vector length;
- `scalarise` stays a generic function that can gain methods on beliefs;
- Mycelium does not assume parameters and variables are disjoint kinds (an `OptimiserFactor`
  already attaches to a variable that is a parameter);
- `local_free_energy` keeps returning coordinates that are expectations of named terms, so that
  §3.1 can return their distribution without changing their meaning.

## Checked by

§1, in `test/exposure.jl` (2026-10-30):
- §1.1: clamped exposure of `GaussianFactor` (`A`, `b`, both, renamed) in both polarities, and of
  `LinearConstraintFactor`'s constant, give the original messages and graded local free energy;
  a whole graph gives the same marginal and Bethe free energy. Not yet run over every factor in
  the conformance checker.
- §1.2–§1.3: seven observations tying one exposed weight under a `GaussianPrior` give the
  closed-form Bayesian linear regression posterior, the posterior predictive at a new input,
  and Bethe free energy $= -\log p(y)$; a $2\times3$ weight is checked against a design matrix
  built entry by entry; uninformative blocks give exactly `TrivialBelief()`.
- A hierarchy over weights: random slopes ([[Hierarchical Models]] §3) rebuilt with one
  regression factor per observation and the group weight exposed and tied agrees with the
  dense reference (means to rtol $10^{-9}$, covariances, $-\log p(y)$ to rtol $10^{-10}$).

Still to test: §2.4 by a Bethe sum with a tap, compared with the sum without it; §3.2 against
closed forms for Gaussian energies (entropic risk of $\mathcal N(m, s^2)$ is $m + \beta s^2/2$).

## Open

- ~~Whether exposure is a wrapper or a property each factor declares~~: a wrapper, with optional per-factor methods (2026-10-30).
- A generic approximate `parameter_message` (Laplace by AD) for nonlinear factors, and VMP/EP for beliefs on both inputs and weights.
- How exposed weights meet `LuxCore`'s parameter handling and `Optimisers.jl` setups.
- Whether energy taps need §2.3's stop-gradient to be a per-wire option, which depends on
  [[L-SPEC-002 Stop Gradient]].
- Which risk scalarisations admit message passing: [[RQ-007 Risk Scalarisations and Local Decomposition]].
