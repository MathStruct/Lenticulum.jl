#design

> **Status: planned (2026-10-22, Daniel: "one spec for later would be stop gradient. That seems kinda important").** Related: [[L-SPEC-001 The Factor Interface]], [[Factors are Parameterized Statistical Games]] §"Parameters, and the gradient the paper asks for", [[Backward Signatures]], [[RED-Diff as a Statistical Game]] §3, [[LP-011 Gradient couplings and isfrozen are declared but unused]]. Code: `lib/LenticulumCore.jl/src/abstract_types.jl` (`AbstractGradientCoupling`), `statistical_game.jl` (`isfrozen`), `lib/Mycelium.jl/src/graph.jl` (`EdgeDirection`, `connect!`).

> Sources: code: `lib/LenticulumCore.jl/src/abstract_types.jl`, `statistical_game.jl`, `lib/Mycelium.jl/src/graph.jl`, `polarity_resolution.jl`; tests: both packages' `test/runtests.jl`, `LP-011` testsets.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## Scope

A **stop gradient** lets values pass forward but no gradient flow back: JAX's `stop_gradient`, PyTorch's `detach`, `ChainRulesCore.ignore_derivatives` in Julia. In Lenticulum there are several things one could mean by it, and the repository already has a piece of each. This specification is to say which ones are part of the interface, how a user asks for them, and what the backward pass then computes. It is not written yet; this note records the scope.

## What exists

| meaning | where | state |
|---|---|---|
| **Detach a value.** No cotangent crosses a wire (frozen targets, EM-style alternation, straight-through estimators) | nothing Lenticulum-specific; `ignore_derivatives` works inside a factor's own code | — |
| **Freeze parameters.** A factor's `ps` are not updated, but cotangents still flow *through* it to upstream factors | `isfrozen(factor)`; planned distinction from the optimiser-list trait `islearnable` | not implemented; `build` and `validate` reject true, even for non-learnable factors (LP-011 option 2, 2026-10-26) |
| **Drop coupling terms.** Keep only the block-diagonal of the composite Jacobian (AutoBayes Definition 29): stop the gradient through the sampling path and the pushforward prior | `DiagonalCoupling()` per edge (`connect!(…; coupling)`), the default; alternatives `PathwiseCoupling`, `ScoreFunctionCoupling`, `ExactCoupling` | no gradient implementation; only `DiagonalCoupling()` accepted by `connect!`, `Edge` and `ComposedFactor`; other values throw (LP-011 option 2) |
| **One-way messages.** A wire carries messages in one direction only (the message-passing analogue) | `Emitting()`, `Absorbing()` edge directions | implemented: the schedules prune forbidden messages |
| **A stop-gradient as the method.** RED-Diff's regulariser gradient treats the network output as constant | [[RED-Diff as a Statistical Game]] §3; [[Implicit Diffusion Learners]] proves the resulting field conservative | implemented in VariationalDiffusion.jl |

The default coupling annotation preserves existing behavior; it does not insert AD stops.
Rejecting `isfrozen` also for non-learnable factors is deliberate: an optimiser-list filter
is not a backward rule preserving input cotangents. These are construction/validation checks
only, covered by the LenticulumCore and Mycelium `LP-011` testsets. Option 1 remains planned.

## Open (for the specification)

- Per wire, per factor, or per parameter subset? Stopping with respect to some parameters only (stop the gradient into the encoder, not the decoder) is the common case in amortised inference.
- How it composes with the [[Backward Signatures]] design: a stopped wire returns `NoTangent()` in the factor-level `rrule`; a frozen factor still returns input cotangents.
- Whether `DiagonalCoupling` should stay the default once couplings are read: it is the cheapest choice and also a biased one.
- Whether a straight-through estimator (custom backward pass) belongs here or is a separate factor.
- One rule per meaning, a test per rule, and how ADEV-style estimators (the CT-ML wiki's [Automatic Differentiation of Expected Values](https://mathstruct.org/CategoryTheory-ML-Wiki/Automatic-Differentiation-of-Expected-Values)) supply the dropped coupling terms.
