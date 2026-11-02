#implementation

> Executes belief schedules, with optional recording of committed residual tasks
> for the separately defined realised-schedule derivative.

> Sources: Riegler, Kirkelund, Manchon, Badiu & Fleury, *Merging Belief Propagation and the Mean Field Approximation: A Free Energy Approach*, IEEE Trans. Inf. Theory 59(1), 2013 (mean-field nodes). code: `passing.jl`
>
> Bibliography: [[Bibliography#^riegler2013merging|Riegler et al. 2013]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

Implements: executing a schedule. `step!` → `sweep!` → `propagate!` → `infer!`.
Theory: [[Messages are Inversions]], [[Loopy Message Passing]].

## `step!` is where the identity lives

For a `:to_variable` task:

```julia
chans  = available_channels(s, g, fid; skip = t.edge)   # ← factor-side exclusion
pol    = resolve_polarity(g, fid, e.channel, chans)
check_legal(g, fid, pol)
prior  = excluded_marginal(s, g, e.variable, t.edge)    # ← variable-side exclusion
belief = factor_message(factor, e.channel, pol, inputs, prior, ps, st)
```

Five lines, and every one of them is a named thing from the paper: the polarity is the
input/output split, the prior is the $\pi$ of $c'_\pi$ (the cavity), the inputs are the $y$,
and `factor_message` is the inversion **with $\pi$ divided out**: the likelihood site
$\mu_{f\to v}$ with $c'_\pi(y) = \pi \odot \mu_{f\to v}$, not the posterior itself
(`Messages are Inversions.md`, LP-019 item 11).

`factor_message`'s generic implementation is `assemble` then `likelihood_message`, which
defaults to `invert` for lenses whose inversion is already extrinsic (L-ADR-0006). Structural factors override it directly, because
for them the lens is trivial and building one would be ceremony.

## Mean-field nodes (2026-11-01)

Every factor node carries a message rule, chosen with `factor!(b, name, f; rule)`:
`BeliefPropagation()` (the default, everything above) or `MeanField()`. For a mean-field node
`step!` replaces the two cavity lines by the other variables' *full* marginals:

```julia
inputs = _marginal_inputs(s, g, fid, t.edge)          # marginal(s, g, v) for every other channel
pol    = resolve_polarity(g, fid, e.channel, keys(inputs))
belief = mean_field_message(factor, e.channel, pol, inputs, ps, st)   # exp E_q[log f]
```

Nothing else changes. Belief-propagation nodes still receive cavities, and these include the
mean-field nodes' messages. So a graph with both kinds runs the hybrid of Riegler, Kirkelund,
Manchon, Badiu & Fleury (*Merging Belief Propagation and the Mean Field Approximation: A Free
Energy Approach*, IEEE Trans. Inf. Theory 59(1), 2013). Its stationary points are exact within
the belief-propagation part and factorised across the mean-field nodes. A typical use is a dense
coupling factor, such as a sum node over many inputs, on mean field, with structured factors on
belief propagation around it.

- **Iterate a sequential schedule.** A mean-field node's message to one neighbour changes the
  marginals its other messages read, so one tree sweep is not a fixed point. In-place sweeps give
  coordinate ascent.
- **`mean_field_message` has no generic method.** The expectation needs the factor's form, so a
  factor opts in. Lenticulum implements it for `LinearConstraintFactor` (the BP message with each
  input at its mean, since $\log f$ is quadratic) and for `SwitchingConstraintFactor`.
- **The graph free energy is the merged one** (2026-11-02, [[L-SPEC-005 The Hybrid Free Energy]]).
  Riegler et al. (2013), eq. (17): a mean-field node contributes `mean_field_energy`, its average
  energy under the product of its variables' marginals with no entropy, and the variables'
  counting numbers count belief-propagation neighbours only. `free_energy_gradient` and
  `parameter_step` train such graphs at fixed beliefs and marginals. Before this the free energy
  refused them, and learning was EM outside the graph.
- **Convergence is measured on the messages**, as for loopy BP. Mixtures of very narrow components
  have a rounding floor in `belief_distance`, about $10^{-10}$ to $10^{-8}$ at width $10^{-8}$, so set
  `tol` above it.

Tested in `test/message_rules.jl`: the default rule; a recording factor that sees the marginal
(prior pooled with its own earlier message), not the cavity; the refusal for a factor without `mean_field_energy`; and the
missing-method error. The end-to-end check, a hybrid graph against an independent structured mean field,
is in Lenticulum's `test/switching_constraint.jl`.

## `sweep!` executors have different semantics

- **Generic** (sequential, tree, forward-backward): executes **in place**. Later tasks see
  earlier results *within the same sweep*, so information crosses a whole path per sweep.
- **`FloodingSchedule`**: **double buffered**. Every task is computed with `commit = false`
  against the previous sweep's messages, and all results are written afterwards. Information
  crosses one edge per sweep.
- **`ResidualSchedule`**: probes every legal task, commits the largest residual,
  then repeats up to its task budget. Probe state updates are discarded; committed
  updates advance state. Ties prefer the oldest message, then task order.

The trade is not just speed. On a loopy graph the sequential order *is* a modelling choice made
without noticing. Flooding removes message-order dependence for stateless factors;
Lux state still threads in task order, so stateful evaluations can depend on that order.

## Implementation difficulties

### 1. `_inputs` is $O(d_f)$ per channel, so `step!` is $O(d_f^2)$

`_inputs` does a `findfirst` over the factor's edges for each requested channel. For a factor of
degree $d$ that is $O(d^2)$ per message and $O(d^3)$ per factor per sweep. Irrelevant for
degree 2–5, wrong for a factor with a hundred channels. A precomputed `channel → edge` map per
factor in `FactorGraph` would fix it and is the obvious next optimisation.

### 2. `ps`/`st` threading is by name and is not validated

`_subtree(ps, name)` returns `NamedTuple()` when the key is absent, so a **typo in a factor's
name silently yields empty parameters** rather than an error. That is the single most likely
way to get a confusingly wrong answer from this file: a factor trains on nothing and nobody
says so.

`_setsubtree` also `merge`s into the state tree on every message, allocating a fresh
`NamedTuple` each time. Correct (states must not be mutated in place if the caller holds the old
one) but wasteful in the inner loop.

A `validate_parameters(g, ps)` pre-flight check — every learnable factor has an entry, no
entries without a factor — is missing and would be cheap.

### 3. Damping is applied *after* the message is computed, and only to `:to_variable`

`step!` damps the outgoing belief against the previous one on that edge. Variable → factor
messages are never damped, which is the usual convention (they are deterministic pools of
already-damped inputs) but is a convention, not a theorem.

Damping is implemented for numeric Diracs, Gaussian canonical parameters,
categoricals and compatible fixed-component mixtures/samples ([[messages]]).
It remains a **silent no-op** for unsupported belief pairs. `propagate!` does not check `can_damp` before promising damping, so a
caller can pass `damping = 0.3` and get none.

### 4. `propagate!` special-cases `TreeSchedule` by type

```julia
if sched isa TreeSchedule && iszero(damping)
    ... return ConvergenceReport(true, 1, 0.0, "tree schedule: one sweep is exact")
```

With damping the shortcut is wrong even on a tree: a damped sweep keeps part of each stored
message, so from a warm start one sweep is not the fixed point. A unary source moved from mean
0 to 10 and swept once with `damping = 0.5` gave mean 5 and the report "exact" (LP-019
item 5). Damped tree schedules are now iterated and checked like any other schedule.

The claim "one sweep is exact" is true only if the graph really is a tree — which
`tree_schedule` enforced at construction. So the invariant is carried by the *type*, which is
sound but implicit: a hand-constructed `TreeSchedule(inward, outward)` over a loopy graph would
get a false "exact" report. A `TreeSchedule` carrying the graph it was built for, or a
construction-only inner constructor, would close that hole.

### 5. `infer!` returns four things

`(marginals, report, st, store)`. The `store` is returned because
[[free_energy|`bethe_free_energy`]] needs it, but a four-tuple is a poor interface. An
`InferenceResult` struct with named fields is the obvious improvement and was skipped only to
avoid inventing a type before knowing what else belongs in it.

### 6. Cotangents are not here, deliberately

Gradients with respect to parameters are **not** belief messages. They are computed
once per parameter step by the separate [[gradient_couplings]] driver, with finite
schedule replay or an opt-in converged-message adjoint for non-diagonal boundaries. `optimiser_step` remains separate from
`step!`; parameters are not updated inside an inference sweep.

Related: [[Messages are Inversions]], [[schedules]], [[free_energy]], [[Loopy Message Passing]]

## LP-011 implementation (2026-10-27)

Values crossing detached edges use the core stop primitive, in both
directions. Factor parameters are pinned according to graph freezing and traits before
ordinary message evaluation. The optional ChainRules infer! rule replays a fixed realised
static sweep count without damping; it returns no state/report cotangents. It does not
differentiate convergence decisions. The former statement that all coupling gradients
are unimplemented is superseded by the separate parameter driver.

Evidence: `lib/Mycelium.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].


## Batch C recording and replay (2026-10-28)

The optional `record` sink on residual `sweep!` and `propagate!` appends only
committed tasks. With no sink, task choices and arithmetic are unchanged.
The ChainRules `infer!` rule records that reference pass and replays its fixed
choices; flooding replays double buffering and all schedules include outgoing
damping. No report, graph or state cotangent is returned. [[gradient_couplings]]
specifies the piecewise derivative and deterministic/functional-factor requirement.
The separate implicit driver verifies every synchronous BP row before its adjoint
solve, rather than trusting TreeSchedule's ordinary one-sweep report.

Evidence: `test/graph_adjoint.jl`, `test/graph_adjoint_devices.jl`.

## Parameter keys and batched messages (2026-11-02)

`step!` reads and writes `ps[key]` and `st[key]` with the node's parameter key (`plates.md`),
which is its name unless tied, so difficulty 2 above now reads "by key". `factor_messages(factors,
target, polarity, inputs, priors, ps, st)` is the batched form of `factor_message` for tied
replicas; the default calls `factor_message` for each, and a plate sweep (`PlateSchedule`) calls
it for each group it can batch.
