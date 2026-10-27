#implementation

> The factor and gradient-coupling type hierarchy; all four built-in couplings have
> supported implementations, with explicit domain restrictions (LP-011).

> Sources: code: `abstract_types.jl`
>
> Theory (CT-ML wiki): [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Functor)

Implements: the type hierarchy for [[Open Models and Latent Channels]] (Def. 1), [[Inversions and Bayesian Lenses]] (Def. 9),
[[Factors are Parameterized Statistical Games]] (Def. 20), [[Factors are Parameterized Statistical Games]] (Def. 27) and
[Composition of Gradients](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#parameterized-statistical-games-and-their-gradients) (Def. 29).

## The three families

```
AbstractLenticulumFactor                      ← a parameterized statistical game (Def. 27)
├─ AbstractLenticulumContainerFactor{factors} ← ps/st nested under field names
└─ AbstractLenticulumWrapperFactor{factor}    ← ps/st passed through unwrapped

AbstractOpenModel        ← Def. 1:  c : X ⇝ ⟦c⟧ × Y
AbstractBelief           ← an element of 𝒫X
AbstractInversion        ← Def. 9:  c' : 𝒫X → {Y ⇝ X × ⟦c⟧}
AbstractBayesianLens     ← Def. 9:  the pair (c, c')

AbstractEnergySpace      ← E_c        (ours, not the paper's)
AbstractScalarisation    ← σ : E → ℝ  (ours)

AbstractGradientCoupling ← Def. 29's laxness, made a per-edge choice
```

The first family mirrors `LuxCore` name for name — `AbstractLuxContainerLayer{layers}` and
`AbstractLuxWrapperLayer{layer}` become `...ContainerFactor{factors}` and
`...WrapperFactor{factor}`, with the same `{names}`-in-the-type-parameter trick for deriving
the parameter tree from field names.

## Why factors are not a subtype of `AbstractLuxLayer`

It would be tempting, and it is wrong. `LuxCore.apply(layer, x, ps, st)` presumes a
direction: `x` in, `y` out. A factor has no direction until a [[Channels and Polarity|polarity]]
is chosen, and offering an `apply` that silently picks one would be a trap. Subtyping would
also inherit `outputsize`, which is meaningless for a relation.

Instead we **extend the functions** (`initialparameters`, `initialstates`,
`parameterlength`, `statelength`) without subtyping the abstract type. That gets the
interop — a factor may contain Lux layers, and setup works uniformly — without inheriting
the directional assumptions. This is the single design decision most likely to be
questioned later, so it is worth stating why: *the parameter interface is directionless and
worth sharing; the application interface is directional and must not be.*

## `AbstractGradientCoupling` — the paper's "semantics functors", as a type

[Definition 29](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#parameterized-statistical-games-and-their-gradients) composes gradients block-diagonally, and the
paper is explicit that the result is only lax and that "this can be accounted for
mechanistically by an implementation". Its closing paragraph then predicts that "the Laplace
method, the delta rule, and sampling schemes of various kinds — will correspond to different
**semantics functors**".

`DiagonalCoupling` / `PathwiseCoupling` / `ScoreFunctionCoupling` / `ExactCoupling` select
per-boundary derivative rules; see [[gradient_couplings]] for the supported domains.

Historical state, 2026-10-26 (superseded by the implementation below): a shared construction-time guard accepts
only `DiagonalCoupling()` in `ComposedFactor`, `connect!` and `Edge`; all other values raise
`ArgumentError` identifying LP-011 and L-SPEC-002. The types and fields remain for the future
gradient implementation. Accepting the default preserves existing behavior; it does not
insert AD stop-gradients. The LenticulumCore and Mycelium `LP-011` testsets check the three
non-default built-in families (including score-function baselines), a custom subtype,
constructor bypasses, and default construction. No message-passing dispatch changes.

## Implementation difficulties

### 1. Beliefs are the hardest abstraction and are deliberately under-specified

`AbstractBelief` has no interface here beyond `isexact`. It should eventually support
something like `logdensity`, `sample`, `mean`, `marginal`, `project(family)` — but the right
set depends on which of the three regimes ([[Implicit Learners]]) is in play, and choosing
too early would bake in a representation. LuxCore's precedent is instructive: it has no
array interface at all, and is better for it.

The one commitment made: `DiracBelief`, `SampleBelief`, `TrivialBelief` exist so that
downstream code has something to dispatch on, and so that the important degenerate cases —
a clamped channel, a particle set, the unit space — are nameable from day one.

### 2. `isexact` defaults to `false`, on purpose

Silence should not imply exactness. An approximate pushforward that forgot to declare itself
would otherwise be treated as ground truth by any future diagnostic. Defaulting to `false`
makes the failure mode "we report more approximation than there is", which is the safe
direction.

### 3. `supports_polarity` defaults to `false`

Same reasoning inverted: a factor supports no polarity until it says so. The alternative —
default `true` — would let a graph compile and then fail at message-passing time with an
error far from its cause.

Related: [[LenticulumCore]], [[channels]], [[statistical_game]]

## LP-011 implementation (2026-10-27)

All four built-in gradient couplings are accepted; unknown subtypes still throw.
Graph parameter gradients dispatch per boundary, with explicit finite replay for
non-diagonal edges. The old option-2 guards described above are historical.

Evidence: `lib/Mycelium.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].
