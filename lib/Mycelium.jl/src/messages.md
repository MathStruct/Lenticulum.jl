#implementation

> The message store keeps both directions on every edge. Pooling and public
> overlap rules support categorical and mixture beliefs, with explicit exclusions.
> Likelihood potentials add log-densities and reweight explicit particles.

> Sources: code: `messages.jl`, `gaussian.jl`, `../test/overlap.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

Implements: the message store, belief pooling, and **both** exclusion principles.
Theory: [[Messages are Inversions]].

## The store keeps both directions per edge

`to_variable[e]` and `to_factor[e]`, plus `nothing` for "not yet computed" — which is distinct
from "computed and uninformative" (`TrivialBelief()`). Keeping both directions separately is the
alternative to *dividing out*, which a density representation would do and which is not
available for general beliefs.

`excluded_marginal` **recomputes** the product over the other edges rather than dividing the
full marginal by the skipped message. Division needs densities and is numerically fragile;
recomputation costs $O(d_v)$ per message and is always defined.

## Both exclusions

1. **Variable side** — `excluded_marginal`: $\mu_{v\to f}$ omits $\mu_{f\to v}$. Without it a
   factor's own previous claim returns to it as independent evidence and the belief becomes
   exponentially over-confident.
2. **Factor side** — in [[passing|`step!`]]: when computing $\mu_{f\to v}$, the incoming message
   on that same edge is excluded from the observed channels.

The second was **not** in the design; the test suite found it. See [[Mycelium]] §1.

## Implementation difficulties

### 1. `combine` is deliberately partial (sample–potential pooling is resolved)

| case | result |
|---|---|
| `TrivialBelief` with anything | the other one |
| agreeing `DiracBelief`s | that Dirac |
| **disagreeing** `DiracBelief`s | **throws** |
| `DiracBelief` with anything | the Dirac |
| `CategoricalBelief` with `CategoricalBelief` | elementwise product, renormalised; disjoint supports throw |
| `SampleBelief` with a belief that has `belief_logdensity` | the samples, reweighted (warns if the effective sample size collapses) |
| potential with potential | `LogPotentialBelief` whose log-density is the sum |
| `SampleBelief` with a potential | importance reweighting via `belief_logdensity`, without resampling |
| Gaussian, categorical (with or without labels), or mixture with a potential | throws: no exact arbitrary-callable product rule; supply particles or a richer potential type |
| `MixtureBelief` with another supported belief | componentwise products, weights scaled by each overlap (`combine_with_overlap` rules: categorical here, Gaussian in Mycelium’s `gaussian.jl`) |
| two `SampleBelief`s | throws: neither has a density |
| anything else | throws informatively |

The sample and mixture rules live in the catch-all method, not in new two-argument methods:
any `(Specific, Abstract)` method next to the existing `(Abstract, Dirac)` and `(Abstract, Trivial)`
ones would create dispatch ambiguities (§1b), whereas the catch-all is only reached when no
specific rule applies.

Disagreeing Diracs throw rather than pick one because two hard clamps in contradiction is a
*wiring* error, not a numerical one, and the error message says how to express the intended
thing instead (a finite `Observed` precision — a soft clamp, per [[Channels and Polarity]]).

Two `SampleBelief`s still cannot be pooled: neither has a density. A sample belief *can* now
be pooled with anything that has one (Gaussian, categorical, mixture) by importance
reweighting, and `moment_match` (in `Lenticulum`) projects samples onto a Gaussian, after which
two particle sets can meet. The two concerns of §1c below are handled as follows: the
effective sample size is computed on every reweighting and a collapse **warns** (below 10
for large sample sets, below 1 % of the samples for medium ones, below 1.5 for tiny ones); and the missing idempotence is the correct
behaviour of a product (pooling the same evidence twice counts it twice), so it is
documented, not prevented. See [[Belief Algebra]], [[Categorical Belief]], [[Mixture Belief]].

### 1b. The unit law was unreachable for the one type that can be pooled

The three unit-law methods were originally

```julia
combine(::TrivialBelief, b) = b
combine(a, ::TrivialBelief) = a
combine(a::TrivialBelief, ::TrivialBelief) = a
```

with the untyped second argument covering "anything". Against the catch-all at the bottom of
the file,

```julia
combine(a::AbstractBelief, b::AbstractBelief) = throw(ArgumentError(...))
```

the pair `(TrivialBelief, GaussianBelief)` matches both `(TrivialBelief, Any)` and
`(AbstractBelief, AbstractBelief)`, and **neither is more specific than the other**: the first
is narrower on the left, the second on the right. Julia reports a `MethodError: combine(...) is
ambiguous`, thrown from inside `excluded_marginal` on the first sweep — so pooling a message
with the accumulator's `TrivialBelief` seed failed for `GaussianBelief`, the only belief type
in the project that `combine` can actually pool.

The fix is to restate the unit law one rung down, at `AbstractBelief`:

```julia
combine(::TrivialBelief, b::AbstractBelief) = b
combine(a::AbstractBelief, ::TrivialBelief) = a
```

which *is* strictly more specific than the catch-all, so the unit always wins. The untyped
methods stay, for payloads that are not `AbstractBelief`s at all.

> [!warning] This failed loudly only by luck
> The catch-all throws. Had it returned something — a `TrivialBelief`, say, or a "best effort"
> pooling — the ambiguity would have been resolved silently in whichever order the methods
> happened to be defined, and `marginal` would have quietly dropped every message pooled
> against the seed. **A catch-all that throws is what turned a specificity bug into a
> stack trace.**
>
> The general lesson for this file: every rule in the table above is stated at two different
> levels of the type hierarchy (`Any` for foreign payloads, `AbstractBelief` for beliefs), and
> a rule stated at only one of them is a latent ambiguity waiting for the first downstream
> belief type. Adding a belief type is not what exposes it — adding a belief type that gets
> *pooled* is.

Caught by [[The Linear Gaussian Chain]] and by the two-factor model in `Lenticulum`'s test
suite, not by this package's own tests at the time: those tests used `DiracBelief` and
`TrivialBelief` only, and every one of those pairs had an explicit method. The current
suite also checks structured beliefs, particle weighting, and callable potentials. **A package whose
central operation is "combine two beliefs" cannot test that operation with one belief type.**

### 1c. A density RATIO is enough, and a classifier estimates one

The oldest sample-pooling blocker is resolved when one operand supplies a potential.
`Adversarial.RatioFactor` emits `LogPotentialBelief(x -> log r(x))`; the existing
`belief_logdensity` reweighting path applies it to a particle cavity. Two such messages
add their log-potentials. No density of the particle cloud is required, and no density
estimator is invented for two clouds.

Mycelium reports severe ESS collapse. Reapplying the same evidence multiplies it twice;
products are not idempotent, and this does not violate associativity or commutativity.
`test/runtests.jl` checks both orders and parenthesisations for two ratio potentials on
one particle variable, including matrix and vector-of-vectors particles. Adversarial's
`test/conformance.jl` checks the same algebra through a graph with two RatioFactors.

An arbitrary callable does not specify an exact Gaussian or mixture posterior. This
implementation also refuses categorical products, including labeled ones, rather than
assuming how labels map to evaluation points. Richer likelihood subtypes can supply
those rules later.

### 1d. Stated as a typing problem

Partiality is the shape of the difficulty, not an incidental feature of it. The set of pairs
`combine` accepts is a **relation on types**, checked dynamically by method lookup and an error
branch — and the ambiguity bug recorded in §1b is what happens when symmetric multiple dispatch
is asked to resolve a partial binary operation on a type lattice.

A discipline that indexed beliefs by what they support would turn
`combine(SampleBelief, SampleBelief)` from a runtime throw into a compile-time refusal, and
would make the conditions on a *new* belief type explicit rather than discovered by the first
crash. See [[The Type Discipline of a Factor Graph]] §3.1.

### 1e. There is a Julia package that already does this

`IncrementalInference.jl` (with `Caesar.jl` and `KernelDensityEstimate.jl`) performs
**nonparametric belief propagation**: beliefs as kernel density estimates, with the *product*
of belief functions estimated by multiscale Gibbs sampling, over a Bayes/junction tree.

That is this gap, closed, in production robotics — by a third route, neither the
density-ratio one of §1c nor the NCE one of §1d. Anything built here should be measured
against it first. See [[Related Julia Projects]] §6.

### 2. `combine` is not associative-by-construction, and the fold assumes it is

`marginal` folds `combine` left to right over the incident edges. Product-of-densities is
associative and commutative, so this is fine *in principle*; but the implemented `DiracBelief`
dominance rule is only associative because Dirac-vs-Dirac throws on disagreement. Add a
belief type whose combination is order-sensitive (a moment-matched Gaussian projection, say)
and `marginal` silently becomes order-dependent.

The LP-016 tests check particle/potential associativity and commutativity against explicit
normalised weights; these are algebraic checks on the finite support, not guarantees about
importance-sampling accuracy.

### 3. `belief_distance` returns `Inf`, not `0`, for unknown pairs

Deliberate. An unverifiable residual makes [[passing|`propagate!`]] run to `maxsweeps` and
report that convergence could not be checked. Returning `0` would be more convenient and would
produce silent false "converged" reports. A wasted sweep is much cheaper than a wrong answer.

`LogPotentialBelief` returns distance zero only when the two wrapped callables are identical
(`===`), and `Inf` otherwise. Other potential subtypes retain the conservative fallback.
This does not attempt to recognise functions equal up to a constant, and assumes captured
mutable data is not changed while a message is in use.

### 4. Damping is often a no-op

`damp` returns the new message unchanged when `can_damp` is false. This includes potentials:
normalisation is unknown, so a convex mixture has no chosen interpretation. Schedules can
inspect `can_damp`, but currently do not report unsupported damping automatically. Earlier
notes overstated that reporting. Numeric Diracs, categorical
beliefs on the same support, samples on identical particles, Gaussians, and mixtures with
matching components have their existing damping rules.

### 5. `Vector{Any}` for the store

Messages are heterogeneous (different belief types per edge), so the store is `Vector{Any}` and
every access is a dynamic dispatch. Correct and simple; the alternative — a per-edge
type-parameterised store — would make `MessageStore` depend on the whole graph's belief types,
which are not known until the first sweep. Probably the right trade at this scale, but it means
message passing will not be fast, and pretending otherwise would be misleading.

**Measured**, on the same access pattern with and without concrete types: `Vector{Any}` is
**130× slower** and turns an allocation-free loop into 19 MB. And a second, separate finding
sits next to it — a sweep is **quadratic in the number of factors**, because `ps` is a
`NamedTuple` looked up with a runtime `Symbol`, which is linear in its field count. Both are in
[[Parallelism and Compilation]] §2, with the benchmark in `bench/scaling.jl`.

Related: [[Messages are Inversions]], [[passing]], [[Loopy Message Passing]],
[[Parallelism and Compilation]],
[[The Linear Gaussian Chain]]

## Public overlap rule

`combine_with_overlap(c,d)` returns the pooled belief and log ∫p_c p_d.
Categorical pairs use a log-sum-exp; disjoint support returns `(c,-Inf)`
as a placeholder, and mixture pooling drops that zero-mass pair. Gaussian
pairs are implemented in Mycelium's `gaussian.jl`, require proper c, and
allow improper d up to an additive constant depending on d alone. There is
no extra `log_overlap` wrapper: callers needing only the mass use `last`.
`test/overlap.jl` checks categorical sums, Gaussian convolution, and improper
Gaussian overlap differences against closed forms. Homotopy branch weighting
uses this public rule, as does mixture pooling's availability check.

## LP-011 implementation (2026-10-27)

Read-only message operations accept an internal abstract store,
allowing immutable replay tuples alongside the ordinary mutable MessageStore. Marginal
and excluded-marginal reads enforce detach even on externally populated stores.
A Gaussian belief covariance ChainRules rule uses the inverse-matrix derivative,
avoiding differentiation through a mutating Cholesky inverse.

Evidence: `lib/Mycelium.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].
