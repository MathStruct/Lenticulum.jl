# Mycelium.jl — implementation note

Implements: factor graphs and message passing for Lenticulum. This is the layer with **no
counterpart in Lux**, because in Lux the wiring is a DAG and the message order is implied by
it.

Concept notes: [[Factor Graphs]], [[Everything is a Factor]], [[Messages are Inversions]],
[[Polarity Resolution]], [[Schedules]], [[Bethe Free Energy]], [[Loopy Message Passing]].

## File map

| file | note | content |
|---|---|---|
| `graph.jl` | [[graph]] | the bipartite structure, edge directions, `istree` / `isdag` |
| `polarity_resolution.jl` | [[polarity_resolution]] | target + available messages → `Polarity`; legality |
| `messages.jl` | [[messages]] | the store, `combine`, both exclusion principles |
| `schedules.jl` | [[schedules]] | tree / flooding / forward-backward / residual |
| `passing.jl` | [[passing]] | executing a schedule; convergence reporting |
| `free_energy.jl` | [[free_energy]] | counting numbers, Bethe, the graded energy |
| `factors.jl` | [[factors]] | data, priors, losses, optimisers, relays |

## Dependencies

`LenticulumCore` (the factor interface and the belief/energy types), `LuxCore` (parameter
trees), `Random`, `DispatchDoctor`. **No graph library**: the adjacency structure needed here is
small, bipartite and channel-labelled, and every graph package would need adapting rather than
using. No solver, no AD, no distributions — same discipline as `LenticulumCore`.

## Status

Working and tested (107 assertions): the graph structure and its two acyclicity notions,
polarity resolution and legality, both exclusion principles, all schedule generators, the
message loop with damping and convergence reporting, the counting-number arithmetic, the
graded/Bethe free energy, and the five structural factors including the three optimiser rules.

Stubbed: `belief_logdensity` (and therefore a general `combine`), `ResidualSchedule`'s
execution path, joint messages over several channels.

## Three amendments made to `LenticulumCore` for this

These belong in the interface package, not downstream, so they were added there:

- `supported_polarities(factor)` — the *listable* form of `supports_polarity`. A scheduler
  needs to enumerate, not just test.
- `isunidirectional(factor)` — exactly one supported polarity.
- `islearnable(factor)` / `isfrozen(factor)` — the user's "non-learnable factors" made a trait.
  `islearnable` defaults to `parameterlength > 0` but is overridable, because a factor with
  pinned parameters (a pretrained encoder, a physical constant) is not learnable even though it
  has parameters. `isfrozen` is the *other* case, and confusing the two silently detaches a
  subgraph: a frozen factor still propagates cotangents to its inputs, a non-learnable one has
  no parameter wire at all.

`Mycelium` adds a fourth, locally: `issink(factor)` — **zero** supported polarities. Losses and
monitors are sinks, and this is a third case beyond unidirectional and bidirectional.

## Implementation difficulties

### 1. Two bugs the tests found that the design did not anticipate

Both produce plausible-looking wrong answers rather than errors, so both are worth recording.

**Tree schedules must be pruned by edge direction.** The first version generated all $2|E|$
messages and threw a `PolarityError` at the loss node: an `Absorbing` edge cannot carry a
factor → variable message. The fix filters, and `TreeSchedule.pruned` counts what was dropped —
surfaced rather than swallowed, so a genuinely mis-wired graph is visible. See [[schedules]] §1.

**The exclusion principle has to be enforced twice.** The variable-side exclusion
($\mu_{v\to f}$ omits $\mu_{f\to v}$) was designed in; the **factor-side** one was not. When
computing $\mu_{f\to v}$, the incoming message *on that same edge* must not be counted among
the observed channels, or the target resolves as both `Observed` and `Unobserved`.

That it *threw* rather than silently miscomputing is luck attributable to a design decision
made for a different reason: `Polarity{names}` carries channel names in its **type**, so a
double assignment is a construction error. With a `Dict` one would have overwritten the other
and the factor would have quietly conditioned on its own prediction. The performance argument
for the type-domain representation ([[channels]] §"Why the polarity is in the type") turned out
to be the lesser reason for it.

### 2. Two name collisions, again

`Mycelium.scalar_free_energy` clashed with `LenticulumCore.scalar_free_energy`. Resolution:
**extend** the existing function rather than define a new one — the scalar free energy of a
graph and of a factor are the same concept at two scales, and two exported bindings with that
name would force every user to disambiguate.

After that, an exhaustive scan of `names(Mycelium)` against `Base` and `LenticulumCore` shows
zero remaining collisions. Worth doing once per package rather than discovering them one test at
a time — this is the third and fourth such collision in the project (`Channel`, `precision`
were the first two).

### 3. `combine` is the real gap

Pooling two general beliefs needs densities. Implemented: `TrivialBelief` as unit,
`DiracBelief` idempotent, `DiracBelief` dominating (the $\rho_{in}=\infty$ clamp), and
contradictory Diracs **throwing** — two hard clamps in disagreement is a wiring error, not a
numerical one. Everything else throws informatively.

This is the same open question `open_model.md` §4 records: the belief representation cannot be
designed before there is one working factor to design it against. Every downstream feature
(particle messages, conjugate messages, moment matching) is blocked on it, and that is the
single most important thing to unblock next.

### 4. Convergence cannot always be verified

`belief_distance` returns `Inf` for pairs it cannot compare, which makes `propagate!` run to
`maxsweeps` and report that convergence is unverifiable. Returning `0` would have been more
convenient and would have produced silent false "converged" reports. **A wasted sweep is much
cheaper than a wrong answer**, so `Inf` it is; the reason string says so explicitly.

### 5. Parameters: hidden or exposed, and the graph decides

A factor's parameters may live in `ps` (Lux style, hidden) or be **exposed as a channel** and
thus become a variable — which is what makes an optimiser an ordinary factor
([[Everything is a Factor]]). Both are supported, but nothing yet *checks* that a graph is
consistent about it: exposing a parameter and also threading it through `ps` would double-count
it silently. A validation pass for this is missing.

### 6. What is not implemented, and why not

- **Joint messages** over several channels at once. A message addresses one variable, so a
  factor coupling several unobserved channels discards the correlation between them — the
  mean-field laxness of [[Composition of Bayesian Lenses|Remark 16]] at the message level. Not
  an oversight; fixing it means deciding where to pay for joint representations.
- **`ResidualSchedule` execution.** The type and its documentation exist; the priority loop does
  not, because it is useless without a working `belief_distance`.
- **Cotangent flow.** Deliberately *not* messages. Gradients are accumulated per factor under
  each edge's `AbstractGradientCoupling`, on a different schedule (once per training step, not
  once per inference sweep). `optimiser_step` is a separate function from `step!` for exactly
  this reason. Conflating the two schedules is a classic source of silent bugs.

Related: [[LenticulumCore]], [[AutoBayes to Lenticulum]], [[Factor Graphs]]
