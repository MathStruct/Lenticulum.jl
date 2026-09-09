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

Related: [[Messages are Inversions]], [[passing]], [[Loopy Message Passing]]
