#implementation

> Structural factors and their local free-energy contract, including whether they read cavity messages.

> Sources: code: `factors.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Gradient Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses)

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
| `PriorFactor` | `Bidirectional` for graph free energy | optional | 1 | $-\log p_\pi(x)$ |
| `LossFactor` | `Absorbing` | no | **0** (a sink) | $f(\ldots)$ |
| `RelayFactor` | `Bidirectional` for graph free energy | no | 2 | $0$ (free energy $-H(q)$) |
| `OptimiserFactor` | `Bidirectional` | no (holds *state*) | 1 | $0$ |

`LossFactor` having **zero** polarities is the third case beyond unidirectional and
bidirectional: a **sink**, corresponding exactly to Cruttwell et al.'s learning-rate cap, the
lens $(L,L')\to(1,1)$ that terminates a wire.

`issink(f)` delegates to `LenticulumCore.hasnopolarity(f)`. Its default still checks
whether `supported_polarities(f)` is empty. Factors with prohibitive lists override the
core predicate, so structural sink checks need no enumeration (LP-017).

## Cavity-message trait

`uses_cavity` defaults to `false`. `PriorFactor` overrides it with `true` because its factor
belief combines the prior with the incoming cavity. `LossFactor` also returns `true`: it
reads incoming points to evaluate the loss. `DataFactor`, `RelayFactor` and `OptimiserFactor`
ignore incoming messages in `local_free_energy` and return zero, so retain `false`.

For factors with `uses_cavity == true`, graph free-energy evaluation rejects `Emitting()`
edges with `ArgumentError`; use `Bidirectional()` when the factor must send as well.
This is independent of supported message polarities: a prior still only emits its own belief.
`nothing` on a permitted edge remains valid input to the local method. The Mycelium testset
"LP-014: cavity messages and edge directions" checks the trait values and guard; see
[[free_energy]] and [[LP-014 Emitting edges drop the cavity message from the free energy]].

## The optimiser rules are Cruttwell's table verbatim

| rule | `get` $U(s,p)$ | `put` $U^*(s,p,\bar p)$ |
|---|---|---|
| `GradientDescent(η)` | $p$ | $(s,\ p - \eta\bar p)$ |
| `Momentum(η,γ)` | $p$ | $s' = \gamma s - \eta\bar p$; $(s',\ p+s')$ |
| `Nesterov(η,γ)` | $p + \gamma s$ | as momentum |

**Nesterov is the one with a non-trivial `get`** — the reason optimisers must be lenses and not
functions. The test suite asserts `rule_get(nesterov, s, p) != p` while
`rule_get(gd, s, p) == p`, because that inequality *is* the content of
[Learning Components as Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses) §3.4.

## Exact game quantities for a prior

`PriorFactor` has game-level `free_energy` and `entropy` for Dirac and categorical beliefs, both
scalar: the expectation of `nlogp` (a finite sum) minus the entropy, and the entropy. For a Gaussian
belief with an arbitrary `nlogp` there is no exact expectation, so they throw rather than reuse the
approximation in `local_free_energy`. Tested in "PriorFactor exact game free energy and entropy".

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

### 3. `LossFactor` requires Dirac beliefs

Energies are pointwise ([Definition 20](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)), so they need a sample. `point`
extracts one from a `DiracBelief` and **throws** for anything else, rather than silently taking
a mean.

Throwing is right — a mean is not a sample, and $\mathbb{E}[l(x)] \ne l(\mathbb{E}[x])$ for any
nonlinear $l$, which is the same Jensen gap [[Scalar and Multivariate Energy]] §5 quantifies.
This limits `LossFactor` to incoming Dirac beliefs. `PriorFactor` handles categorical
expectations directly and Gaussian expectations with the qualification in its docstring.

### 4. `OptimiserFactor.factor_message` reads `st.p` and `st.state` by convention

There is no declared state schema. `factor_message` explicitly checks that both fields exist
and throws `ArgumentError` when either is missing; the state values themselves remain untyped.

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
structural factor ([[AutoBayes Examples as Factor Graphs|Example 1]]'s mixing probabilities), and forcing a
new type for it seemed worse. Worth revisiting.

Related: [[Everything is a Factor]], [Learning Components as Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses), [[passing]], [[Factors are Parameterized Statistical Games]]

## Game quantities and conformance (2026-10-26)

`DataFactor` is a clamped source: its game energy and entropy are zero, matching its
constant graph contribution. `RelayFactor` is the identity game with the same zero pair.
Their pointwise energies now expose that convention. `PriorFactor.energy` evaluates
`nlogp(x)`; its existing exact Dirac/categorical game methods are unchanged.

`LossFactor` is a terminal game with no inferred variables. Pass all observations in a
`NamedTuple` as `y`, either points or Dirac beliefs. `free_energy` evaluates the loss and
`entropy` returns zero with the same shape (scalar or vector). There is no assembled lens,
polarity or canonical pointwise `x/a/y` split. Nonpoint beliefs are refused by `point`.
`OptimiserFactor` is a stateful update rule rather than an inference game: both game methods
throw `ArgumentError`. Its graph contribution remains zero and it has no pointwise energy.

`test/conformance.jl` includes the single repository checker by relative path; no production
dependency or duplicate checker is needed. Data, Prior and Optimiser use its direct-message
path at their unary polarity, Relay at both polarities. Optimiser explicitly opts out of
pointwise and game checks. Loss uses `p = nothing` (sink) with pointwise/message opt-outs;
its scalar and vector terminal games still check game/graph agreement. Fixtures use fixed
clamps, independently normalised categorical probabilities and elementary losses. This
checks the declared accounting conventions, not distributional entropy of a data source.


## Device optimiser coefficients (2026-10-27)

GPUArraysCore methods for GradientDescent, Momentum and Nesterov cast learning
rate/momentum coefficients to the parameter element type. This prevents Float64
configuration scalars from widening Float32 device parameters. CPU methods and
all public rule signatures are unchanged. JLArrays tests compare get/put with
the CPU formula and assert device/element-type preservation; graph tests also
exercise the default learning rate. See `lib/ImplicitLayers.jl/test/device_arrays.jl`.

## The relay's entropy (2026-10-29, LP-019 item 1)

A relay is an equality factor. Its energy is zero, but its Bethe belief $q(a)\,\delta(a = b)$,
with $q$ the pooled incoming messages, has the entropy of $q$ relative to the measure on the
diagonal. The graph free energy counts the relayed variable with one more degree, and the
relay's $-H(q)$ is what cancels it. With the zero contribution used before, a categorical
prior `[0.3, 0.7]` relayed to a second variable gave $F = H(p) = 0.6109$ instead of $0$; a
Gaussian prior gave its differential entropy. `local_free_energy(::RelayFactor)` now returns
$-H(q)$ and `uses_cavity` is `true`, so graph free-energy evaluation needs its edges to carry
variable-to-factor messages: a directed relay (`Absorbing()` in, `Emitting()` out), accepted
before with a zero contribution, now raises the LP-014 error asking for `Bidirectional()`. The game quantities (`energy`, the identity game) are unchanged.
Tested in `test/lp019.jl`.
