# factors.jl — implementation note

Implements: the structural factors — data, priors, losses, optimisers, relays.
Theory: [[Everything is a Factor]].

## Why these are factors at all

Forced by AutoBayes Remark 24: the composite loss is the true variational free energy only if
the **prior is a separate factor** with trivial inversion, energy $-\log p_\pi$, and zero
entropy. Once that is true of priors it is true of data (a Dirac prior), of losses (energy, no
inversion), and — once parameters can be exposed as channels — of optimisers.

| type | edges | learnable | polarities | energy |
|---|---|---|---|---|
| `DataFactor` | `Emitting` | no | 1 | $0$ |
| `PriorFactor` | `Emitting` | optional | 1 | $-\log p_\pi(x)$ |
| `LossFactor` | `Absorbing` | no | **0** (a sink) | $f(\ldots)$ |
| `RelayFactor` | `Bidirectional` | no | 2 | $0$ |
| `OptimiserFactor` | `Bidirectional` | no (holds *state*) | 1 | $0$ |

`LossFactor` having **zero** polarities is the third case beyond unidirectional and
bidirectional: a **sink**, corresponding exactly to Cruttwell et al.'s learning-rate cap, the
lens $(L,L')\to(1,1)$ that terminates a wire.

## The optimiser rules are Cruttwell's table verbatim

| rule | `get` $U(s,p)$ | `put` $U^*(s,p,\bar p)$ |
|---|---|---|
| `GradientDescent(η)` | $p$ | $(s,\ p - \eta\bar p)$ |
| `Momentum(η,γ)` | $p$ | $s' = -\gamma s - \eta\bar p$; $(s',\ p+s')$ |
| `Nesterov(η,γ)` | $p + \gamma s$ | as momentum |

**Nesterov is the one with a non-trivial `get`** — the reason optimisers must be lenses and not
functions. The test suite asserts `rule_get(nesterov, s, p) != p` while
`rule_get(gd, s, p) == p`, because that inequality *is* the content of
[[Learning Components as Parametric Lenses]] §3.4.

## Implementation difficulties

### 1. `NamedTuple{(f.channel,)}` everywhere is type-unstable

`DataFactor`, `PriorFactor` and `OptimiserFactor` store their channel name as a **field**, so
every `channels`, `supported_polarities` and `supports_polarity` call builds a `NamedTuple` from
a runtime `Symbol`. The compiler cannot specialise, and the polarity's type is unknown at the
call site even though it is concrete once built.

Making the channel a **type parameter** (`DataFactor{ch,T}`) would fix it and would make the
whole polarity-in-the-type argument of [[channels]] actually pay off. Not done because it would
propagate a type parameter through every structural factor and the ergonomics get worse. This is
the same complaint as [[polarity_resolution]] §3 and they share one fix.

### 2. `local_free_energy` is a second free-energy entry point

`LenticulumCore.free_energy(factor, π, y, ps, st)` splits prior from observation;
`Mycelium.local_free_energy(factor, beliefs, ps, st)` takes all channels at once. Two functions
for one concept.

The justification is real — in a graph there is no distinguished split until a polarity is
chosen, and the free energy is a property of the factor, not of any one message — but two entry
points is still a smell, and a caller could implement one and not the other and get a confusing
`ArgumentError` rather than a missing-method error. A generic
`local_free_energy` deriving from `free_energy` for a chosen polarity would be better and is not
obviously well-defined.

### 3. `point` refuses non-Dirac beliefs, which blocks every real energy

Energies are pointwise ([[Statistical Game|Definition 20]]), so they need a sample. `point`
extracts one from a `DiracBelief` and **throws** for anything else, rather than silently taking
a mean.

Throwing is right — a mean is not a sample, and $\mathbb{E}[l(x)] \ne l(\mathbb{E}[x])$ for any
nonlinear $l$, which is the same Jensen gap [[Scalar and Multivariate Energy]] §5 quantifies.
But it means **no structural factor can compute an energy under a non-Dirac belief**, which is
most of the interesting cases. The fix is a `sample(belief, rng)` and Monte-Carlo evaluation, or
an energy interface that accepts distributions. Neither exists.

### 4. `OptimiserFactor.factor_message` reads `st.p` and `st.state` by convention

There is no declared state schema. A state tree missing `:p` yields `nothing`, which then flows
into `rule_get` and produces a `DiracBelief(nothing)` — a wrong answer with no error. This is
the most fragile thing in the file and wants either a typed state struct or an explicit check.

### 5. `optimiser_step` is separate from message passing, deliberately

A parameter update happens once per **training** step; a message happens once per **inference**
sweep. Merging them (updating parameters inside a belief sweep) is a classic source of silent
bugs, so `optimiser_step` is not reachable from `step!` at all. The cost is that a training loop
must interleave the two schedules by hand — there is no `train!` yet, and writing one is the
natural next task.

### 6. `PriorFactor.learnable` is a `Bool` field, not a trait dispatch

`islearnable(f) = f.learnable`, so it is a runtime value. Every other factor answers from its
type. Inconsistent, and it means `islearnable` cannot be constant-folded for priors. It exists
because a prior parameterised by its natural parameters is the single most likely learnable
structural factor ([[Examples from the Paper|Example 1]]'s mixing probabilities), and forcing a
new type for it seemed worse. Worth revisiting.

Related: [[Everything is a Factor]], [[Learning Components as Parametric Lenses]], [[passing]], [[Statistical Game]]
