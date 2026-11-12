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
`build`. Both accept frozen factors: the parameter driver pins selected coordinates
and preserves input derivatives. The old option-2 rejection is recorded in LP-011's
historical section. Freeze paths are checked against actual parameter trees at use.

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

**Cached per graph (2026-11-12).** A profile of a sweep on `bench/state_scaling.jl`'s chain
(800 steps, 2 401 factors, 50 sweeps, 8 267 samples) put about a quarter of the sweep in
`resolve_polarity`. Building `NamedTuple{decl}` with run-time names took 997 samples in that line
alone, and rebuilding `decl` from `channels(factor)` took 909. `FactorGraph` now keeps, per
factor, a `Dict` from `(target, available)` to the polarity, behind a lock. The graph and its
factors are immutable, so an entry never goes stale. Only a successful resolution is stored, so
an invalid key throws on every call. After the change `resolve_polarity` takes about 100 of
6 296 samples, and the sweep takes 14.0 ms instead of 18.9 (best of 5,
one machine). The dynamic dispatch on the polarity's type in `_rule_message` remains; the type-level fix
above would remove it too. Test: "polarity resolution" in `test/runtests.jl`, which compares the
cached polarity with a fresh `_resolve_polarity` for every target and every subset of the other
channels, given as a tuple, a vector and a set. Reviewed by Mistral (no defects; it also checked
the graph-edit paths, eight tasks resolving at once, and Zygote and ForwardDiff through `step!`).

**Planned per graph (2026-11-12, the rest of I18).** After the cache, `_step!` still paid per message
for what the cache does not hold: `available_channels` built a `Tuple` from a `Vector`, the cache key
was a tuple of names, `check_legal` ran (a dispatch on the polarity's type, a loop over the edges and
`supports_polarity`), `_inputs` looked each input's edge up by name, and every field of the
`FactorNode` was read through a dynamic `getproperty`. None of it depends on more than the graph, the
edge and which inputs are informative. `_step!` now keeps, in `g.emit_plans` and behind the same lock,
an `_EmitPlan` per (edge, bit mask of the factor's edges with an informative input): the factor, the
rule, the checked polarity and the input names in its type, and the input edges. A message computes
the mask (no allocation), looks the plan up, and `_emit` calls the rule with everything but the
beliefs known to the compiler. A plan of `nothing` is the `TrivialBelief()` case; an illegal key is
not stored and throws on every call; a factor with more than 64 edges plans each message afresh.
Mean-field nodes keep the old path. This is not the type-level fix above: the beliefs in the store
are still type-erased, so one dynamic dispatch into `factor_message` per message remains, and
`channels` is still a run-time property.

Measured with `bench/sweep_profile.jl` (T = 800, 2 401 factors, best of 5, one machine): 9.94 ms per
sweep instead of 12.02, 13.2 MB allocated instead of 14.6; `_step!` holds 1 367 profile samples
instead of 1 686. What is left around the factor's own work is reading the type-erased messages
(`getproperty` on `Message`, about 100 samples) and the dispatch into the factor. A new set of
informative inputs is a new `NamedTuple` type and compiles once on either path; a scratch comparison
(not kept in the repository) on a 20-channel factor gave the same compile time per new set, about
70 ms, before and after. Test: `test/emit_plan.jl`, which compares `step!` with the old sequence
(resolve, check, gather by name, call the rule) on random wirings, edge directions, store contents
and rules, for factors of 1 to 70 channels; checks that an illegal plan is not stored; and runs 8
tasks with their own stores on one graph. Reviewed by Mistral: no defect
([[LP-027 Findings of the review of the I18 plans]]).

### 4. `check_legal`'s error picks an arbitrary channel

When `supports_polarity` returns `false` the error names
`first(unobserved_channels(p))` — the target — which is usually right but is a guess. The factor
knows *why* it refused and has no way to say so. A `supports_polarity` returning a reason
(`nothing` for yes, a `String` for no) would be strictly better and is a small interface change.

Related: [[Polarity Resolution]], [[Channels and Polarity]], [[messages]]

## LP-011 implementation (2026-10-27)

Frozen factors are now legal in both build and validate;
parameter freezing lives in the parameter/gradient driver and preserves input
cotangents. The 2026-10-26 frozen-factor rejection described above is historical.
Polarity legality and direction checks remain unchanged.

Evidence: `lib/MyceliumLearning.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].
