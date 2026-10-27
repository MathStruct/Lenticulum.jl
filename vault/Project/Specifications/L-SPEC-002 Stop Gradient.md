#design

> **Status: implemented for the documented CPU/static-schedule domains (2026-10-27); unsupported combinations throw.** Related: [[L-SPEC-001 The Factor Interface]], [[Factors are Parameterized Statistical Games]] §"Parameters, and the gradient the paper asks for", [[Backward Signatures]], [[RED-Diff as a Statistical Game]] §3, [[LP-011 Gradient couplings and isfrozen are declared but unused]]. Code: `lib/LenticulumCore.jl/src/abstract_types.jl` (`AbstractGradientCoupling`), `statistical_game.jl` (`isfrozen`), `lib/Mycelium.jl/src/graph.jl` (`EdgeDirection`, `connect!`).

> Sources: code: `lib/LenticulumCore.jl/src/abstract_types.jl`, `statistical_game.jl`, `lib/Mycelium.jl/src/graph.jl`, `polarity_resolution.jl`; tests: both packages' `test/runtests.jl`, `LP-011` testsets.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## Initial scope (2026-10-22–26; retained history)

The following scope and option-2 account describe the state before the dated implementation below.
Initial status (2026-10-22): planned. Daniel: "one spec for later would be stop gradient.
That seems kinda important".

### Scope

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

## Maintainer decisions and implementation (2026-10-27)

The maintainer selected option 1 of [[LP-011 Gradient couplings and isfrozen are declared but unused]].
These rules supersede the historical construction guards above:

1. **Detach a wire:** `connect!(...; detach=true)` passes messages unchanged and suppresses
   derivatives crossing that attachment in both directions. It overrides its coupling.
2. **Freeze parameters:** `factor!(...; freeze=(:encoder,))`, a tuple of paths, or `:all`
   pins complete subtrees. Frozen coordinates have zero gradient/update and no optimiser
   state. Input derivatives still flow through the factor. `isfrozen=true` pins all
   parameters for this run, while `islearnable=false` removes the factor from the intrinsic
   candidate list. Frozen learnable factors remain in `learnable_factors`.
3. **Choose gradient blocks:** diagonal preserves the existing fixed-cavity driver;
   exact differentiates a finite static inference unroll; pathwise propagates Dirac and
   analytic Gaussian moment derivatives; score supplies an explicit terminal sampling
   estimator with standard errors. Non-diagonal gradients require `schedule` and a finite
   `nsweeps` (default 1). Score additionally requires `rng` and `nsamples >= 2`.
4. **Compose mixed blocks:** expand the scheduled computation's Jacobian into paths.
   A path survives only if every crossed attachment permits it. A diagonal/detached
   boundary cuts that path; exact/pathwise boundaries keep their local Jacobian.
   Terminal score nodes add the probability derivative of their sampled point costs.
5. **Straight-through:** `StraightThroughFactor(forward,surrogate)` is a separate
   parameterless directed factor. Forward values come from the first callable;
   ForwardDiff partials and ChainRules reverse pullbacks use the second callable.

See [[gradient_couplings]] for the exact API, estimator objective, mixed semantics,
complexity and domain restrictions. Static sequential/tree/forward-backward schedules,
reference differences and DI ForwardDiff/Zygote are tested; reference differences do not
implement straight-through and are rejected there. Score sampling is currently terminal,
Gaussian/categorical, one sweep, with a fixed scalar baseline. Bidirectional sampling,
flooding/residual replay, other schedule backends and other informative pathwise families
throw explanatory `ArgumentError`s. General solver stochastic estimators, GPU and an
implicit graph fixed-point adjoint remain outside this implementation.

**Compatibility correction.** The requested description of today's diagonal gradient as
an EM M-step was inaccurate: it holds cavity messages fixed while Gaussian factors
recompute their local posteriors. This existing derivative is preserved. At a true exact
posterior, fixed-posterior EM and the marginal-likelihood gradient coincide; requiring
these two to differ would be an incorrect oracle. `test/gradient_couplings.jl` derives
all three quantities independently. `ComposedFactor`'s Gaussian endpoint game already
collapses its internal channel, so its legacy diagonal, analytic pathwise and exact
endpoint gradients coincide. Its score rule samples the exact internal posterior and
returns an estimate of the same endpoint derivative, including a baseline control variate.

**Evidence:** `lib/Mycelium.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`; final suite counts/timings are recorded in the LP-011 note.
No new paper citation was introduced; the ADEV wiki note was read for estimator semantics.
