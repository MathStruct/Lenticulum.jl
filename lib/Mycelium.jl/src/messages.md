# messages.jl — implementation note

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

### 1. `combine` is deliberately partial, and this is the package's main gap

| case | result |
|---|---|
| `TrivialBelief` with anything | the other one |
| agreeing `DiracBelief`s | that Dirac |
| **disagreeing** `DiracBelief`s | **throws** |
| `DiracBelief` with anything | the Dirac |
| anything else | throws informatively |

Disagreeing Diracs throw rather than pick one because two hard clamps in contradiction is a
*wiring* error, not a numerical one, and the error message says how to express the intended
thing instead (a finite `Observed` precision — a soft clamp, per [[Channels and Polarity]]).

Two `SampleBelief`s cannot be pooled without importance reweighting, which needs
`belief_logdensity`, which no belief type implements. **Every downstream feature — particle
messages, conjugate messages, moment matching — is blocked on this**, and it is the same open
question [[open_model]] §4 records.

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
suite, not by this package's own tests: `Mycelium`'s tests use `DiracBelief` and
`TrivialBelief` only, and every one of those pairs has an explicit method. **A package whose
central operation is "combine two beliefs" cannot test that operation with one belief type.**

### 1c. A density RATIO is enough, and a classifier estimates one

The blocker above is stated as needing `belief_logdensity`. It does not — importance
reweighting needs only a **ratio** $p(x)/q(x)$, and Mohamed & Lakshminarayanan's identity says
the logit of an optimal classifier separating $p$ from $q$ *is* $\log p - \log q$, estimated
from samples alone with neither density.

`Adversarial.RatioFactor` implements it and `Adversarial.reweight` performs the pooling; the
test suite recovers a Gaussian's moments by reweighting samples of a wider one. So the
operation exists. It is still not wired into `combine`, for two reasons:

1. its **quality is unreported** — `effective_sample_size` can collapse to a handful of
   particles with nothing raised, and a silently degenerating `combine` is worse than one that
   throws;
2. it is **not idempotent** — reweighting twice by the same ratio squares the weights, so the
   operation violates §2 below unless the caller tracks which ratios have been applied.

And there is a stronger version. In a GAN the comparison distribution $q$ is the generator and
unknown, so a ratio is all you get. In **noise-contrastive estimation** $q$ is *chosen*, so

$$\log p(x) = \operatorname{logit} D^\ast(x) + \log q(x)$$

and you recover the normalised log-density — which is `belief_logdensity` itself, not a
special-cased substitute for it. See [[Training Energy-Based Models]] §2.2.

Recorded here because this file has named the gap since the beginning and these are the first
concrete routes around it. See [[Implicit Generative Models]] §5 and [[ratio]] §4.

### 2. `combine` is not associative-by-construction, and the fold assumes it is

`marginal` folds `combine` left to right over the incident edges. Product-of-densities is
associative and commutative, so this is fine *in principle*; but the implemented `DiracBelief`
dominance rule is only associative because Dirac-vs-Dirac throws on disagreement. Add a
belief type whose combination is order-sensitive (a moment-matched Gaussian projection, say)
and `marginal` silently becomes order-dependent.

There is no test for associativity because there is nothing yet to test it on. When a second
non-trivial belief type lands, that test must land with it.

### 3. `belief_distance` returns `Inf`, not `0`, for unknown pairs

Deliberate. An unverifiable residual makes [[passing|`propagate!`]] run to `maxsweeps` and
report that convergence could not be checked. Returning `0` would be more convenient and would
produce silent false "converged" reports. A wasted sweep is much cheaper than a wrong answer.

### 4. Damping is often a silent no-op

`damp` falls back to returning the new message unchanged when `can_damp` is false, which is the
case for everything except numeric-payload Diracs. A caller passing `damping = 0.3` to
[[passing|`propagate!`]] on a graph of particle beliefs gets no damping at all.

`can_damp` exists so this is *inspectable*, and `ConvergenceReport.reason` mentions it — but it
is not currently checked automatically, so the report can say "hit maxsweeps" without saying
"and your damping did nothing". A pre-flight check in `propagate!` would be a genuine
improvement.

### 5. `Vector{Any}` for the store

Messages are heterogeneous (different belief types per edge), so the store is `Vector{Any}` and
every access is a dynamic dispatch. Correct and simple; the alternative — a per-edge
type-parameterised store — would make `MessageStore` depend on the whole graph's belief types,
which are not known until the first sweep. Probably the right trade at this scale, but it means
message passing will not be fast, and pretending otherwise would be misleading.

Related: [[Messages are Inversions]], [[passing]], [[Loopy Message Passing]],
[[The Linear Gaussian Chain]]
