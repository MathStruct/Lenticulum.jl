# Polarity Resolution

> The step with no counterpart in Lux: deciding, per message, which way to run a factor.

## Why it exists

A Lux layer knows its direction; `Dense(3 => 5)` cannot be run backwards. A Lenticulum factor
is a relation and has no direction until one is chosen. So a factor cannot *store* a
`(get, put)` pair — it must be able to **assemble** one on demand:

$$\text{factor} \;+\; \text{polarity} \;\longrightarrow\; \text{parametric lens} \;\longrightarrow\; \text{message}$$

That is [[README]]'s "our program must extract or assemble parametric Lenses before they can be
used", made into a function call.

## The resolution rule

Given a target channel and the set of channels with available incoming messages:

```
target                         -> Unobserved()
available (excluding target)   -> Observed()
all other declared channels    -> Latent()
```

Exactly one channel is `Unobserved()` per message. That is a v0 restriction; see §"Joint
messages" below.

## Two independent legality checks

A resolved polarity must pass **both**, and they are genuinely different things:

| check | a property of | what it means |
|---|---|---|
| **edge direction** | the *wiring* | an `Emitting` edge may not be `Observed()`; an `Absorbing` edge may not be `Unobserved()` |
| **`supports_polarity`** | the *factor* | can this factor be run this way at all? |

Together they replace Lux's acyclicity restriction:

> A Lux `Chain` is well-formed iff the wiring is a DAG.
> A Mycelium graph is well-formed iff **every scheduled message passes both checks.**

The direction check fires first and names the offending channel, which matters for error
quality: "edge is `Absorbing`, so channel `:pred` cannot be `Unobserved`" tells you where the
wiring is wrong, whereas `supports_polarity` returning `false` tells you only that something is.

## Legality is computable for some families and declared for others

[[Channels and Polarity]] notes that `supports_polarity` is *declared* by the factor author
rather than derived, and gives the reason: a polarity may be mathematically invertible but
computationally hopeless, and only the author knows.

There is one family where it is genuinely decidable — the algebraic one. There the condition is
a Jacobian rank ([[Inference as Root Finding]]), and in the differential case it is
**structural identifiability**, computed by differential elimination
([[Differential Algebra and DAE Factors]]). So `supports_polarity` is a declaration in general
and a theorem in the cases where a theorem exists.

## Three ways a factor answers a polarity

Which one applies is the factor's business, not the graph's — which is the point of the
abstraction:

| mechanism | family | cost |
|---|---|---|
| closed form | conjugate, invertible maps, flows | cheap |
| root-finding on $r_\theta = 0$ | [[Implicit Learners|algebraic, DEQ, NeuralODE]] | Newton / homotopy |
| proximal step on an energy | [[ImplicitREDDiff|diffusion]] | iterative |

## Joint messages: the known limitation

One channel is `Unobserved()` per message, so a factor always reports about one variable at a
time. When a factor's inversion genuinely couples several unobserved channels, sending them
separately **discards the correlation between them** — which is precisely the mean-field
laxness of [[Composition of Bayesian Lenses|Remark 16]], reappearing at the message level and
measured by the same mutual information.

The correct generalisation is a joint message over a *set* of channels, delivered to a
composite variable. It is not implemented. Recording it here rather than in a TODO because it
is not an oversight: it is the message-passing form of a laxness the framework has at every
level, and "fixing" it means deciding where to pay for joint representations — the same
decision [[Composition of Bayesian Lenses]] says should be exposed at the graph level rather
than hidden.

Related: [[Channels and Polarity]], [[Messages are Inversions]], [[polarity_resolution]]
