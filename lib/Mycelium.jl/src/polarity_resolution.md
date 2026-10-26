#implementation

> Structural validation and message polarity resolution, including rejection of unsupported frozen factors.

> Sources: code: `polarity_resolution.jl`
>
> Theory (CT-ML wiki): [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Functor)

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

Pre-flight checks: declared channels, no isolated factors or variables, and a warning for
multiple emitting edges with no reader. Structural validation is optional and separate from
`build`. Both `validate` and `build` reject `isfrozen == true` with an `ArgumentError` naming
the factor, even if it is non-learnable (LP-011 option 2, 2026-10-26). No frozen backward
semantics exist; excluding a node from `learnable_factors` does not supply them. The
`LP-011 graph construction guards` testset checks both learnability values and both entry
points independently. The check is outside message passing.

## Implementation difficulties

### 1. Missing `channels` methods are skipped

`validate` uses `hasmethod` before consulting `LenticulumCore.channels(fn.factor)`.
If no method exists, the declared-channel check is skipped; exceptions from an existing
method propagate. The earlier note claiming a bare `try` swallowed these errors was stale.
Requiring a channel declaration for every factor remains a separate interface decision.

### 2. Exactly one `Unobserved()` per message

Deliberate v0 restriction. When a factor's inversion couples several unobserved channels,
sending them separately **discards the correlation** — which is the mean-field laxness of
[Remark 16](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens#parallel-composition-is-lax) appearing at the message level, measured by the
same mutual information.

The generalisation is a joint message over a *set* of channels delivered to a composite
variable. Not implemented, and not an oversight: fixing it means deciding where to pay for joint
belief representations, which is the decision
[Composition of Bayesian Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens#the-chain-rule-bayesian-inversion-is-functorial) says should be exposed at the graph level rather than hidden.

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
