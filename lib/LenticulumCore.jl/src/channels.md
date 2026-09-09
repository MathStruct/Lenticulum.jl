# channels.jl — implementation note

Implements: the $X / \llbracket c \rrbracket / Y$ trichotomy of [[Open Model|Definition 1]]
as a runtime-selectable **polarity**, unifying it with the
$P_{in} + P_{out} + P_{latent} = \mathrm{Id}$ of [[ImplicitREDDiff]].

Theory: [[Channels and Polarity]].

## The mapping, restated

| code | AutoBayes | ImplicitREDDiff | meaning |
|---|---|---|---|
| `Observed()` | $Y$ | $P_{in}$ | clamped to data |
| `Unobserved()` | $X$ | $P_{out}$ | inferred; the posterior ranges over it |
| `Latent()` | $\llbracket c \rrbracket$ | $P_{latent}$ | internal scratch |

**The names cross.** "Unobserved" is what inference *outputs*. This is documented on the
`Unobserved` docstring with a `!!! warning`, because it is the mistake everyone makes once.

## Why the polarity is in the type

`Polarity{names,T,R}` wraps two `NamedTuple`s. `names` being a type parameter means

- `select(p, Observed)` folds at compile time to a tuple of `Symbol`s;
- `assemble(factor, polarity)` can return a concretely-typed lens, so the assembled forward
  kernel specialises rather than dispatching through a `Dict` on every message.

A `Dict{Symbol,ChannelPolarity}` would be simpler and would cost a dynamic dispatch per
channel per message. Given that message passing is the inner loop, the type-domain version
is the right default. The cost is compile time when a graph has many distinct polarities —
watch for it.

## Precisions: $P = \rho_{in}P_{in} + \rho_{out}P_{out} + \rho_{latent}P_{latent}$

`Polarity` carries a `precisions` `NamedTuple` alongside the assignment, defaulting via
`default_precision`: `Inf` for `Observed`, `1.0` otherwise.

`Inf` is the *hard* clamp — the categorical [[Copiers Cups and Caps|cup]], "this channel is
exactly the data". A **finite** `Observed` precision is a soft clamp, and having it be
expressible is the point: noisy observations, annealed conditioning, and the guidance
strength of a diffusion sampler are all "an observation I only partly believe". A framework
that only had hard clamps would need a separate mechanism for each.

`ispartition` is currently trivially `true` — the `NamedTuple` representation makes it
impossible for a channel to have two polarities or none. It is kept as a named predicate
because the corresponding *graph-wide* consistency check (every variable node has exactly
one factor asserting it as `Unobserved` per message, no contradictory clamps) is not
trivial, and will want the same name.

## Implementation difficulties

### 1. `Channel` shadows `Base.Channel`

Defining `struct Channel` inside a module is legal (Julia only errors if the binding has
already been resolved to `Base.Channel` in that scope), and it was verified to load. But
**exporting** it would inflict an ambiguity on every downstream `using LenticulumCore`.

Decision: define, do not export. Access as `LenticulumCore.Channel` or import explicitly.
The alternative names considered — `Port`, `FactorChannel`, `Chan` — all lose the
[[README]]'s vocabulary, and the vocabulary is worth more than the convenience of an export.

### 2. `precision` was renamed to `channel_precision`

`Base.precision(::AbstractFloat)` exists, so exporting our own `precision` produced
`UndefVarError: precision not defined` with an ambiguity hint — the test suite caught it on
first run. Unlike `Channel`, `precision` has no strong claim to the concept here, so
renaming was the cheap fix. Recorded because it will look like an arbitrary name otherwise.

### 3. `supports_polarity` is a predicate, not a computation — for now

A factor whose forward kernel is an invertible map could in principle *derive* which
polarities it supports. A factor built from a residual $r_\theta$ could derive it from the
Jacobian's rank structure. Neither is attempted: the predicate is declared by the factor.

The reason to keep it declarative is that legality is not purely a property of the maths —
a polarity may be *mathematically* invertible but computationally hopeless, and the factor
author is the one who knows. When solver-backed factors exist, expect a
`supports_polarity(f, p) = rank_condition(...)` helper, not a change to the interface.

### 4. Latent channels are not yet distinguished from revealed ones

[[Composition of Open Models]] describes `reveal` as a free retyping promoting
$\llbracket c \rrbracket$ into $Y$. There is currently no `reveal` in the code, and
`Latent()` conflates "hidden and will be marginalised" with "hidden but retrievable". They
have very different costs. When `reveal` lands, `Latent` should probably split into
`Latent()` and `Revealed()`, or gain a flag.

Related: [[Channels and Polarity]], [[Open Model]], [[Copiers Cups and Caps]], [[abstract_types]]
