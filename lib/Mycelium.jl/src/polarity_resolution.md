# polarity_resolution.jl — implementation note

Implements: the step with no counterpart in Lux — deciding, per message, which way to run a
factor. Theory: [[Polarity Resolution]], [[Channels and Polarity]].

## The rule

```
target channel                              -> Unobserved()
channels with a message (excluding target)  -> Observed()
all other declared channels                 -> Latent()
```

then `LenticulumCore.assemble(factor, polarity)` gives a lens and the message is an inversion
([[Messages are Inversions]]).

## Two legality checks that are genuinely different

- **`check_legal`** on the *edge directions*: an `Emitting` edge may not be `Observed()`, an
  `Absorbing` edge may not be `Unobserved()`. A property of the **wiring**.
- **`supports_polarity`** on the *factor*: can it be run this way at all? A property of the
  **factor**.

The direction check fires first and names the channel, which matters for error quality:
"edge is `Absorbing`, so channel `:pred` cannot be `Unobserved`" says where the wiring is wrong;
`supports_polarity` returning `false` says only that something is.

Together they replace Lux's acyclicity restriction: a Lux `Chain` is well-formed iff the wiring
is a DAG; a Mycelium graph is well-formed iff every scheduled message passes both.

## `validate`

Pre-flight checks: every edge's channel is declared by its factor, no isolated factors, no
isolated variables. Cheap, and it converts a class of wiring mistakes from confusing runtime
`PolarityError`s into one clear message at build time.

## Implementation difficulties

### 1. The `channels` call is wrapped in a bare `try`

`validate` does `try … catch; nothing; end` around `LenticulumCore.channels(fn.factor)`, so a
factor that has not implemented `channels` is silently skipped rather than rejected. That
swallows real errors too — a `channels` method that throws for a genuine reason looks identical
to one that does not exist.

The right fix is `hasmethod(LenticulumCore.channels, (typeof(f),))` and an explicit error
otherwise. Not done, because `channels` has no default and requiring it would break the
`AbstractLuxWrapperFactor` pattern where channels might be derived. Recorded as a real wart.

### 2. Exactly one `Unobserved()` per message

Deliberate v0 restriction. When a factor's inversion couples several unobserved channels,
sending them separately **discards the correlation** — which is the mean-field laxness of
[[Composition of Bayesian Lenses|Remark 16]] appearing at the message level, measured by the
same mutual information.

The generalisation is a joint message over a *set* of channels delivered to a composite
variable. Not implemented, and not an oversight: fixing it means deciding where to pay for joint
belief representations, which is the decision
[[Composition of Bayesian Lenses]] says should be exposed at the graph level rather than hidden.

### 3. `NamedTuple{decl}(vals)` is type-unstable

`decl` comes from `LenticulumCore.channels(factor)` at runtime, so the polarity's type is not
known to the compiler at the call site even though it *is* concrete once built. Every message
therefore pays a dynamic dispatch to construct its polarity.

The fix is for `channels` to be a compile-time property of the factor's type (a type parameter,
as `AbstractLenticulumContainerFactor{factors}` already does for its sub-factors) rather than a
runtime field. That is a `LenticulumCore` change and it is probably right, since the whole
argument for putting polarity in the type domain ([[channels]]) is undermined if the names
arrive dynamically.

### 4. `check_legal`'s error picks an arbitrary channel

When `supports_polarity` returns `false` the error names
`first(unobserved_channels(p))` — the target — which is usually right but is a guess. The factor
knows *why* it refused and has no way to say so. A `supports_polarity` returning a reason
(`nothing` for yes, a `String` for no) would be strictly better and is a small interface change.

Related: [[Polarity Resolution]], [[Channels and Polarity]], [[messages]]
