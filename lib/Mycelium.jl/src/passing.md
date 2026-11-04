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

## Convergence on marginals (2026-11-03)

`propagate!(...; criterion = :marginals)` compares every variable's marginal before and after a
sweep by moments (`_moment_distance` in `messages.jl`: means and covariances of proper Gaussians,
mixture weights and components, total variation for categoricals, `belief_distance` otherwise).
The message residual of a mixture of components of width $10^{-8}$ stops falling near $10^{-8}$ because it
is measured in natural parameters, whose entries are of order $10^8$; the moments have no such floor,
so `test/switching_constraint.jl` converges its hybrid at `tol = 1e-10`. It is the default for a
`HybridSchedule` and for plate schedules lifted from one; it costs one marginal per variable per
sweep.

## `factor_messages` on blocks (2026-11-03)

A second method of `factor_messages` takes `inputs` as a `NamedTuple` of `BeliefBlock`s and
returns one block (`blocks.md`). The default takes each replica's column, calls the method on
vectors and packs the result with `belief_block`, so a factor type adds block arithmetic by adding
a method of that form and nothing else changes. `DataFactor` (in `factors.jl`) returns its data as
one point block.

## Expectation propagation and mixture reduction (2026-11-03)

Two more rules (Extending Mycelium §4, [[L-SPEC-008 Expectation Propagation and Mixture Reduction]]).
Both compute the belief-propagation message first, from the same cavities, and change what is
sent. `step!` calls `_rule_message(rule, ...)`, which is `factor_message` for belief propagation.

**`ExpectationPropagationRule()`.** With $m_t$ the belief-propagation message to channel $t$ and
$\pi_t$ the target's cavity (the excluded marginal, already the `prior` argument), the node sends

$$
m_t^{\mathrm{EP}} = \operatorname{proj}\bigl[\pi_t\, m_t\bigr] / \pi_t ,
$$

the projection matching moments (Minka 2001). Three hooks, all in Mycelium: `ep_message(m, π)`
(a message that is not a mixture is in its family and passes unchanged; a point cavity, a pinned
variable, makes the message irrelevant and it passes too), `ep_project` (a mixture of Gaussians:
its total mean and covariance) and `ep_divide` (Gaussians: subtract canonical parameters). A site
whose precision is not positive definite, possibly indefinite, is declared improper. An update
is skipped, keeping the edge's message, when the cavity is a Gaussian that is not proper, when a
component of the tilted belief is not proper, or when an input to the factor has a negative
direction (left by an earlier negative site). Skipping keeps the iteration defined; it is a common
remedy, not a rule we quote from Minka's paper. The
first skip on a run warns. On a Gaussian graph nothing leaves the family and the rule is belief
propagation (tested to rounding).

The name is not `ExpectationPropagation`: Lenticulum already exports that, the moment-matching
inversion method of a `LikelihoodFactor` (`likelihood.jl`, Gauss–Hermite quadrature inside one
factor). The rule generalises it to any factor whose message leaves the family.

**`MixtureReduction(M)`.** The message is the exact one with every mixture input expanded
(`mixture_message`, a factor hook; belief propagation collapses those inputs to moments), then
reduced to at most `M` components (`reduce_message`; Lenticulum's method is Runnalls' merge,
`reduce_mixture`). Lenticulum implements `mixture_message` for `LinearConstraintFactor` and
`SwitchingConstraintFactor` (`switching.md` §9). A factor without it refuses a mixture input on
such a node; one with only Gaussian inputs takes `factor_message`.

**One tree sweep.** `propagate!` used to stop a `TreeSchedule` after one undamped sweep whatever
the rules. That is a fixed point when every message depends on upstream messages only: belief
propagation, and `MixtureReduction`, whose messages are exact for reduced inputs. An EP message
reads the target's cavity and a mean-field message the marginals, so a graph with either is now
iterated (`_tree_exact`). Before, a mean-field graph swept with a `TreeSchedule` was reported
exact after one sweep; the hybrid tests use `HybridSchedule`, so no test met it.

**Free energy.** An EP node counts like a belief-propagation node: its local free energy at its
incoming messages, and an entropy at each of its variables (`free_energy.md`). A
`MixtureReduction` node is refused there: no free energy we know of has its fixed points as
stationary points, and the factors' local free energies would collapse the mixtures they were
sent.

Tests: Mycelium's own (`lib/Mycelium.jl/test/message_rules.jl`, last testset) run a prior and three
likelihood mixtures without Lenticulum: belief propagation pools them into 8 components in one
sweep, the EP graph is iterated and each site times its cavity has the tilted moments to $10^{-10}$,
EP edges count in `bp_degree`, and the refusals. The end-to-end ones are Lenticulum's
(`test/mixture_rules.jl`):

- Clutter problem, $D = 1$, $w = 0.5$, prior variance 100, clutter variance 10, data drawn from
  the model (`Xoshiro(3)`). Belief propagation is exact on this star: its marginal holds all
  $2^N$ components and matches the enumerated posterior to $10^{-12}$ ($N = 8, 12$). The EP rule reaches
  the fixed point of a hand-written Minka loop to $10^{-10}$ on a tree and on a sequential schedule.
  At $N = 12$ its mean is within $6 \cdot 10^{-4}$ of the exact one and its variance within 2%; at $N = 8$
  the mean is off by 0.10. At $N = 4$ the undamped iteration oscillates, as the hand loop does, and
  the rule follows that loop step for step.
- A location-mixture likelihood ($y_i = x - b_{k_i} + \varepsilon$, six points, tied switching factors on EP):
  the gradient of the free energy at fixed messages equals the derivative of the free energy at
  the re-converged fixed point, to $10^{-7}$ relative, for an offset and a weight. So the free energy
  of EP nodes is stationary in the messages at a fixed point, which is what lets the
  fixed-belief gradient train such graphs. This is a numerical check, not a theorem quoted.
- A switching linear chain, $T = 5$, $K = 2$ ($x_{t+1} = x_t + \beta_{k_t} + \varepsilon$, $y_t = x_t + \delta$), against
  enumeration of the 16 switch sequences: the largest error of a marginal mean or variance
  falls with the budget, 0.16, 0.011, $9.6 \cdot 10^{-4}$, $1.0 \cdot 10^{-4}$ for $M = 1, 2, 4, 8$, and is $1.6 \cdot 10^{-15}$
  at $M = 16$, where nothing is merged. Belief propagation gives the $M = 2$ result: it collapses
  the input and sends $K = 2$ components.

Difficulties:

1. **The exact reference needed a new product.** Belief propagation on the clutter star multiplies
   likelihood messages, mixtures whose components are unnormalised exponentials
   (`GaussianBelief(η, Λ, true)`). `combine_with_overlap` needed a proper first operand, so the
   reference threw. The product of two unnormalised exponentials is exact in canonical form
   (`gaussian.md`, "Products of improper components").
2. **The first EP sweep has no cavities.** On a tree schedule's inward pass a likelihood node can
   fire before the prior's message has reached the variable, so its cavity is uninformative and
   its tilted belief, the likelihood mixture itself, has improper components. Such an update is
   skipped; with no previous message the edge stays empty, and the outward pass fills it.
3. **A message to a pointed variable stays a mixture.** On the location-mixture graph the messages
   to the observed $y_i$ are not projected: their cavities are points. That is harmless (a point
   dominates), but an assertion that "every message is Gaussian" is wrong there; the test checks
   the marginal of $x$.

## Skipped updates under flooding and residual schedules (2026-11-04, LP-023)

An expectation-propagation update without a proper cavity is skipped: the edge keeps its message.
`step!` with `commit = true` already did that, but the two schedules that compute without
committing did not. `FloodingSchedule` stored the returned `TrivialBelief()` on an edge that had no
message, and `ResidualSchedule` selected the skipped task (residual Inf) and could not commit it.
`_step!` now also returns whether the update was skipped; flooding leaves such an edge as it is, and
the residual schedule never selects it. `step!` is unchanged. Test: `test/mixture_rules.jl`.

## Compilation per structure (2026-11-04)

`ps` and `st` are graph-wide trees with a key per parameter key, so their type changes with every
new graph structure. `step!`, `_step!`, every `sweep!` and the plate group steps
(`_step_planned!`, `_step_blocks!`) read them only through `_subtree` by the node's key, and are
now `@nospecialize` in them; the factor's own methods still specialise on its small subtree.
Measured with SnoopCompile (a scratch script; `examples/structure_edits.jl`'s graphs, 200 points):
a structure with three switches, after one with two had compiled, spent 9.06 s inferring its first
two sweeps, 6.93 s of it in `_step!` for a tree of 201 keys; with `@nospecialize`, 1.14 s. Its first
sweeps took 8.77 s before and 0.70 s after. The example's greedy search fell from 192–200 s to 163.8 s
with gradient fits and from 177 s to 143 s with EM, the same paths and scores. A ForwardDiff
gradient still compiles per structure (3.7 s), since the dual numbers' chunk follows the parameter
count; EM does not use it.

Extended 2026-11-05 to the graph-wide free-energy entry points (`scalar_free_energy`,
`bethe_free_energy`, `factor_free_energies`, `_factor_terms`, `_factor_terms_rows`), the cluster
code outside AD (`_cluster_joint`, the cluster's `factor_message` and `factor_belief`, the batched
`factor_messages`, `factor_beliefs`, `_batch_joints`, `_joint_plan`, `_cluster_batchable`) and EM
(`moment_statistics`, `em_step`, the statistics' `ClusterForm`). The functions AD differentiates
stay specialised (their dual-number types change per structure anyway). Measured in a scratch
script on `examples/structure_edits.jl`'s graphs: the free energy, `em_step` and a five-step
`fit_em` on a new structure took 1.8 s, now 1.14 s (0.31 s warm); the warm per-call times are
unchanged. The structure search with EM went from 34.9 s to 31.3 s, the same path and scores.
