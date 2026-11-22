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
- **`ResidualSchedule`**: commits the task of largest residual, then repeats up to its task
  budget; a priority queue re-probes only the tasks that read the committed message (I4, "The
  residual schedule's queue" below). Probe state updates are discarded; committed
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

`_setsubtree` `merge`s into the state tree, allocating a fresh `NamedTuple`. Correct (states
must not be mutated in place if the caller holds the old one); since 2026-11-11 it runs only when a
factor changed its state ("The state tree" below).

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

A tree is also not enough: one sweep is the fixed point, but whether it is exact depends on the
message rules. That part is checked at run time since P13 (next section).

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

Evidence: `lib/MyceliumLearning.jl/test/gradient_couplings.jl` and
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

**Two quiet sweeps (I22, 2026-11-17).** A marginal reads only the messages to its variable, so one
sweep within the bound does not show that the messages to factors are settled. Under flooding from
an empty store the second sweep is such a sweep: the priors have reached their variables, but the
factors that pass them on have not yet seen them, so every marginal stays at its prior. With one
quiet sweep, `:marginals` stopped there on the loopy divider of `test/p9_relative_tol.jl` (relative
error 1, absolute and relative tolerance alike). `_iterate!` now counts consecutive sweeps within
the bound and stops `:marginals` at two; `:messages` still stops at one.

Why two suffice under flooding: write $V_t$ and $F_t$ for the messages to variables and to factors
after sweep $t$; flooding computes $V_t = f(F_{t-1})$ and $F_t = g(V_{t-1})$. If sweeps $t$ and
$t+1$ leave $V$ unchanged ($V_{t-1} = V_t = V_{t+1}$), then $F_{t+1} = g(V_t) = g(V_{t-1}) = F_t$,
and from there $V_{t+2} = f(F_{t+1}) = f(F_t) = V_{t+1}$: a fixed point. The argument assumes that
an unchanged marginal means unchanged messages to it, which a change that cancels in the product
would break; that is not excluded, only unlikely. In-place schedules can lag the same way (a message
to a factor computed after the one it feeds in a sweep), and the same count covers one sweep of lag
there. The cost is one more sweep for every `:marginals` run, including the hybrid schedules.
**Partial residual sweeps (I24, 2026-11-17).** A `ResidualSchedule` whose `maxtasks` is below the
number of tasks (also the default `maxtasks = 10_000` on a larger graph) commits only part of a sweep,
possibly only messages to factors, so two quiet partial sweeps say nothing: on the divider it stopped
after 2 sweeps with every marginal trivial (LP-052 F1). Such a run now also needs every pending
message residual within the bound when it stops, which makes the stop a fixed point of the messages
too, in the units of the criterion (absolute or relative). The sweep's residual is a cheap filter: it
is at least the sweep's first update, which is the largest pending residual at the start of the
sweep. It is not enough on its own: the sweep's later updates can raise other residuals, and the
review measured up to 2.7 times the bound at the report when only that was checked (LP-054 F1). So
when the marginals have been quiet twice and the filter passes, one scan of every task (evaluated,
not committed) decides; if a residual is above the bound the run goes on. The report gives the
largest of the residuals and says "every pending message residual" in its `reason`. A sweep that
covers every task is unchanged, and so is `:messages`. A partial run whose messages have no
`belief_distance` (residual `Inf`) now never reports convergence under `:marginals` (by the code;
not run). Measured on the divider (`maxtasks` 1 and 3, damping 0 and 0.3, `tol` from 1e-11 to 1e-9):
every pending residual within the bound after the report; a warm start stops after 4–15 sweeps
against 515–1106 cold. Test: `test/p9_relative_tol.jl`, the I24 lines (also the relative criterion,
against the dense solve and a scan of the pending residuals).

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
reduced to at most `M` components (`reduce_message`; for Gaussian mixtures Runnalls' merge,
`reduce_mixture`, in `gaussian.jl` since P17, 2026-11-12, before in Lenticulum). Lenticulum implements `mixture_message` for `LinearConstraintFactor` and
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
`factor_messages`, `factor_beliefs`, `_batch_joint`, `_joint_plan`, `_cluster_batchable`) and EM
(`moment_statistics`, `em_step`, the statistics' `ClusterForm`). The functions AD differentiates
stay specialised (their dual-number types change per structure anyway). Measured in a scratch
script on `examples/structure_edits.jl`'s graphs: the free energy, `em_step` and a five-step
`fit_em` on a new structure took 1.8 s, now 1.14 s (0.31 s warm); the warm per-call times are
unchanged. The structure search with EM went from 34.9 s to 31.3 s, the same path and scores.

**The AD paths (I1, 2026-11-22).** Measured first, on `bench/kalman_chain.jl`'s chain with every
edge `ExactCoupling()` and the tree schedule, the first call of `free_energy_gradient` on a new
structure after one of 16 factors had compiled (a scratch script; `~/.cache/lenticulum/i1/`). The
envelope driver differentiates per key and compiled 0.1 s, EM 0.01 s; the graph drivers compiled
per structure:

| factors | replay, ForwardDiff | replay, Zygote | implicit, Zygote (first − warm) | Zygote of `scalar_free_energy` |
|---|---|---|---|---|
| 31 | 10.7 s | 7.1 s | 15 s | 0.9 s |
| 61 | 11.2 s | 13.7 s | 26 s | 1.3 s |

SnoopCompile put ForwardDiff's 11.5 s of inference in the factors' own code (Gaussian messages,
`LinearAlgebra` on duals), compiled again because the dual type was new: DifferentiationInterface
tags duals with the type of the differentiated closure, which holds `ps` and `st`. The graph
drivers now call `_graph_gradient`, `_graph_jacobian` and `_graph_parameter_gradient`
(`LenticulumCore/src/energy.jl`), whose backend comes from `_graph_ad`: for `AutoForwardDiff()`
without a tag of the caller's, the tag `Tag{_GraphTag, V}` ($V$ the coordinates' element type)
and a chunk of 12, or the length when shorter (`LenticulumCoreForwardDiffDIExt`). The number of
chunks is ForwardDiff's own; only the last one may be padded. ForwardDiff orders nested tags by
their creation, and a fixed tag is created once, so a caller's tag created later would be taken
for the inner one: the fixed tag is used only when no dual is reachable from the contexts (a walk
over the arguments, `FactorGraph` included, 0.7 ms at 61 factors and 3 ms at 241, 1–3 % of a
gradient), and the default otherwise. Forcing the fixed tag under a caller's dual makes ForwardDiff
refuse the call ("Cannot determine ordering of Dual tags"), it does not give a wrong value (a
scratch check); `test/structure_compile.jl` in Lenticulum and `test/i1.jl` in LenticulumCore check
the guard, the nesting through a factor's constant and the equal backend types.

Zygote's 7.4 s at 31 factors went mostly to its generic adjoints over the trees' tuples (`∇map`,
`indexed_iterate`, `_pullback`): `_rebuild` (θ back into `ps`) mapped over them. A profile at 61
factors after the rule below put 2.4 s more in `offset_tree`, which recursed on the tuple's tail,
inferring one tuple type per step, O(keys²). `_unflatten` has a rule (`LenticulumCoreChainRulesCoreExt`) that adds each leaf's
cotangent into its slice of one vector, on the top-level call only (a rule on `_rebuild` would apply
at every level and give a cotangent as long as θ per subtree), and `offset_tree` is a loop compiled
once for a tree wider than 32 fields (narrower ones keep the recursion, which Enzyme needs:
`parameter_tree.md`). The rule's buffer takes θ's element type promoted with the cotangents' (LP-076:
ForwardDiff over a Zygote gradient gives dual cotangents). After both:

| factors | replay, ForwardDiff | replay, Zygote |
|---|---|---|
| 31 | 0.81 s | 2.5 s |
| 61 | 1.5 s | 5.3 s |
| 121 | 3.6 s | 13.4 s |

Warm times are unchanged (best of five at 61 factors: 23 ms for ForwardDiff with the fixed tag or a
fixed chunk alone). **Left:** what remains grows with the structure. For Zygote it is the
inference of its `Pullback` types, whose parameters hold `ps` and `st` in every argument of every
function it transforms (`indexed_iterate` on them was 1.5 s of 5.7 s at 61 factors); ForwardDiff's
remainder was not profiled. Removing it needs the graph-wide trees kept out of every
differentiated signature, a type-erased tree through the replay and the implicit adjoint (a
backlog row). The implicit adjoint was not measured after the change; its dense solve dominates its
first call. Zygote of `scalar_free_energy` itself (`bench/zygote_scaling.jl`) is unchanged: 2.5,
7.4, 25.8 s at 121, 241, 481 factors.

## The state tree (2026-11-11)

LP-024 measured the free energy of a Kalman chain at 7.7, 25.8, 102 ms for 601, 1 201, 2 401
factors, quadratic, and put it down to `_setsubtree`, which `merge`s the graph-wide state tree
once per message. A profile of the free energy at 2 401 factors (20 calls, 1 699 samples) put the
time elsewhere: 830 samples in `_check_tied_state`, which counted the factors of every parameter
key (O(keys × factors)) on every call; 484 in `_subtree`, the lookups of `ps` and `st` by name; and
242 in `_setsubtree`. A lookup in a `NamedTuple` by a run-time `Symbol` scans its names, so every
lookup is O(keys), and a sweep or a free energy does O(factors) of them.

Two changes. `_check_tied_state` counts in one pass. `_writeback(st, key, old, new)` writes a
node's state only when `new !== old`, `old` being what the caller read with `_stored`; a stateless
factor hands back the empty `NamedTuple` it was given, so a stateless graph's tree is never
rebuilt. `_stored` returns `nothing` for a missing key, which no returned state is `===`, so a
missing key is still added: callers read `st[name]` after `infer!` (`lib/Lenticulum.jl/test/lp020.jl`
failed when a first version left missing keys out). Every write-back site goes through it:
`step!`, the factor terms of the free energy (with and without blocks), the plate paths.

**Tied nodes.** Several factors that share a parameter key read one state. In the free energy's
factor terms they all read the original tree, so a factor whose `params` is not its own name
always writes back there, changed or not: the last factor's state is kept, as before. Skipping
there kept the last *changed* state instead, which Mistral's second review reproduced
(`test/state_tree.jl`, "tied factors that create state"). Sweeps read the current tree at every
step and needed nothing. `plates.md` excludes stateful tied factors anyway; `_check_tied_state`
checks the input state only.

`bench/state_scaling.jl` (best of 5 runs; position-velocity chain with three factors per step):

| factors | sweep before (ms) | sweep after (ms) | free energy before (ms) | free energy after (ms) |
|---|---|---|---|---|
| 301 | 3.50 | 3.35 | 2.63 | 1.74 |
| 601 | 8.93 | 8.24 | 8.28 | 4.50 |
| 1 201 | 24.70 | 23.35 | 26.74 | 13.12 |
| 2 401 | 77.46 | 74.76 | 95.15 | 42.88 |
| 4 801 | 258.70 | 261.55 | 352.59 | 150.48 |

The free energy is 1.5 times faster at 301 factors and 2.3 times at 4 801; the sweep is unchanged
within the noise (−1 to 8%). Both are still quadratic: what is left is
the lookup by name, which only the integer-slot store removes ([[Feature Backlog]], I5). Removed since: "The trees by position" below.
`test/state_tree.jl` times `_check_tied_state` at 2 000 and 8 000 factors (at 500 and 2 000 the
old code took 17 times as long; under 8 is required). Its other tests check that a stateless sweep
and free energy return the same tree, that a stateful factor's state is still written, that
missing keys are added, and that tied nodes with state are refused. The first of these would pass
on the old code too: immutable `NamedTuple`s compare by content under `===`, so a `merge` of the
same values is `===` the original. The plate paths with a stateful factor are not tested here.

**Review (Mistral, by reading; it could not run Julia).** (1) The tied-node difference above:
right, recorded, not changed. (2) Missing tests for absent keys, a stateful free energy and the
tied-state error: right for absent keys and the error, both added; a stateful factor through the
free energy is tested only in MyceliumLearning (`test/gradient_couplings.jl`). Plate paths: not
added. (3) The timing test at tens of microseconds could be flaky: plausible, the sizes are four
times larger now.

**Second review (Mistral, with Julia, in its worktree).** It compared old and new semantics on
the same scenarios by redefining `_writeback` as the old unconditional write: identical for
stateless and stateful graphs, missing keys, plates; Zygote and ForwardDiff gradients of the free
energy with a stateful factor match central differences. (1) The tied-node difference: reproduced,
fixed as above, its scenario kept as a test. (2) Zygote through `scalar_free_energy` on a plate
with block rows fails in `_row_scalar` (`sum(...; init = 0.0)` has no rrule): the same before
this change, recorded in the [[Feature Backlog]] (P20). (3) A factor returning `nothing` as its
state would no longer be stored for a missing key: no factor does; not reproduced.

## Where a sweep spends its time after I5 (2026-11-12)

A profile of 50 sweeps on `bench/state_scaling.jl`'s chain at 800 steps (2 401 factors, 7 202
tasks; `T=800 julia --project=bench bench/sweep_profile.jl`, one sample every 0.1 ms). Sample
counts vary from run to run; the shares are what to compare. Of 8 267 samples in `sweep!`:

- `belief_distance` took 145 (1.8%), or about 330 (4%) with the dynamic dispatch at its two call
  sites in `_step!`. The earlier profile had it at 106 of 725, before I5 made the sweep four times
  faster. It is no longer worth work.
- `resolve_polarity` took about 2 100 (25%): it built a `NamedTuple` type from run-time channel
  names on every message. It is now cached per graph ([[polarity_resolution]], difficulty 3); the
  sweep went from 18.9 to 14.0 ms.

The sweep column of `bench/state_scaling.jl` after the cache (best of 5; "before" is the I5 table
below). The free energy does not resolve polarities and is unchanged:

| factors | sweep before (ms) | sweep (ms) |
|---|---|---|
| 301 | 2.34 | 1.77 |
| 601 | 4.76 | 3.56 |
| 1 201 | 9.46 | 7.20 |
| 2 401 | 19.21 | 14.57 |
| 4 801 | 39.08 | 29.79 |

What is left after the cache (6 296 samples): `factor_message` of `GaussianFactor` 2 500, of which
`_rank_deficient` (an SVD of the weight matrix on every backward message, `gaussian.jl`) 477;
`excluded_marginal` 475; `_improper_check` 427 (a Cholesky factor and a symmetry check per
`belief_cov`); `belief_distance` 193; `check_legal` about 270.

### After I18 (2026-11-12)

I18 cached the rank test per weight array and factorised each input's precision once instead of
twice (Lenticulum's `gaussian.md`, "What the checks cost"; LenticulumCore's `gaussian_belief.md`,
"Per-message costs"). The sweep column of `bench/state_scaling.jl` (best of 5; "before" is the
polarity-cache table above):

| factors | sweep before (ms) | sweep (ms) |
|---|---|---|
| 301 | 1.77 | 1.51 |
| 601 | 3.56 | 3.03 |
| 1 201 | 7.20 | 6.03 |
| 2 401 | 14.57 | 12.43 |
| 4 801 | 29.79 | 25.45 |

A sweep at 2 401 factors allocates 14.6 MB instead of 22.7 (`bench/sweep_profile.jl`). Its profile
(1 686 samples in `_step!`): `factor_message` of `GaussianFactor` 422, of which `_moments` 142;
`excluded_marginal` 159; `_rank_deficient` no longer appears. About 290 samples are dynamic
`getproperty` and dispatch in `_step!` and `_rule_message`. That is the part of I18 still open: the
polarity's type is not known to the compiler ([[polarity_resolution]], difficulty 3).

### After the plans (2026-11-12, the rest of I18)

`_step!` keeps the polarity, its legality and the input edges per (edge, informative inputs)
([[polarity_resolution]], difficulty 3, "Planned per graph"). The sweep column of
`bench/state_scaling.jl` (best of 5; "before" is the table above):

| factors | sweep before (ms) | sweep (ms) |
|---|---|---|
| 301 | 1.51 | 1.23 |
| 601 | 3.03 | 2.45 |
| 1 201 | 6.03 | 4.90 |
| 2 401 | 12.43 | 9.96 |
| 4 801 | 25.45 | 20.59 |

About 19% less at every size. A sweep at 2 401 factors allocates 13.2 MB instead of 14.6
(`bench/sweep_profile.jl`), and its profile has 1 367 samples in `_step!` instead of 1 686. What is
left outside the factor's own work is reading the type-erased messages of the store and one dynamic
dispatch into the factor per message.

## The trees by position (2026-11-12, I5)

**The cost.** `ps` and `st` stay `NamedTuple`s for the caller. Reading one entry by a run-time
`Symbol` (`haskey`, `getfield(nt, name)`) scans the names, so it is O(keys), and each factor needs
one or two such reads. Reading by position, `getfield(nt, i::Int)`, takes constant time. A scratch
measurement put one read per entry at 1.0, 4.2, 17.2 and 65 ms in total for 600, 1 200, 2 400 and
4 800 entries. Building a `Dict{Symbol,Int}` of the positions once took 0.04 to 0.30 ms.

**Where the positions live.** They are kept per call, not in the graph. A loop over the factors
first wraps its trees:

- `_IndexedTree(tree, pos)` is used for reading. `_subtree` and `_stored` look the key up in
  `pos` and read by position; a missing key gives `NamedTuple()` and `nothing`, as before.
- `_StateBuffer` is the state a sweep threads. `_writeback` writes in place, and a missing key
  is appended, as `merge` appended it. The values are copied out of the tree at the first write
  only, so a stateless sweep returns the very tree it was given (`===`). `_tree` turns the buffer
  back into a `NamedTuple` once, at the end of the sweep.
- `_ReadOnlyState` is used for the residual schedule's probes: they compute an update and
  discard it, so their writes are dropped. The old code discarded the state those probes returned.

The helpers on these types (`_subtree`, `_stored`, `_writeback`) keep their names, so the
per-factor code (`_step!`, the plate paths) is unchanged; only the loops wrap and unwrap. Here
"the loops" means each `sweep!` method (sequential and tree, flooding, residual, plates; Anderson
delegates to its inner schedule), `_factor_terms`, `_factor_terms_rows`, `_check_tied_state`,
MyceliumLearning's `_envelope_gradient` and `parameter_step`. `step!` called directly with a
`NamedTuple` still reads by name: a loop of the caller's own over `step!` stays quadratic.

Storing a slot in each `FactorNode` when the graph is built was considered and left out. A slot is
a position in `parameter_keys(g)`, not in the caller's tree, which may order its keys differently
or lack some, so each call would still have to check the tree. A slot left stale by a graph
transformation would read another factor's parameters without an error. And it saves only one
hash per read, under 1% of a message. O5 and I2, which need integer slots anyway, can add them.

**Zygote.** The free energy without block rows is differentiated by Zygote, which refuses array
mutation even on the state, whose gradient nobody asks for. There the state is only read through
an `_IndexedTree`. The write-backs are collected afterwards (`_state_writes`: per key, the factor
whose state is kept, in the order of the first write) and applied with one `merge`
(`_written_states`). The rule for tied nodes is unchanged: the last write wins. `test/state_tree.jl`
compares this with the old write-backs, one at a time, on 50 random graphs: keys missing and
shuffled, keys tied, factors with and without state. `_keypositions`, `_slot`, `_state_writes` and
`_key_groups` are `@non_differentiable` (`MyceliumChainRulesCoreExt`). A scratch test checked that
Zygote and ForwardDiff differentiate through `getfield(tree, i)` with a run-time `i` and through a
`Dict` lookup marked that way. The block-row path mutated (`covered .= true`), so Zygote could
not run through it; it uses the buffer, and since P20 Zygote runs through a row by an `rrule`
([[free_energy]] "Zygote on block rows").

**The gradient drivers.** `_envelope_gradient` called `tied_factors(g, key)` for every key, a
scan of all factors, and `_energy_rows(s, g)` for every key. `parameter_step` read `ps`, the
optimiser state and the gradient by name, and called `_key_node` for every key. Both now take the
groups from `_key_groups(g)` (one pass) and read the trees by position.

**Measured.** `bench/state_scaling.jl`, best of 5 runs (3 for the gradient columns). "Before" for
the sweep and the free energy comes from the table above, after P7. "Before" for the gradient and
`parameter_step` comes from the same measurement run on the unchanged code before the benchmark
had those columns.

| factors | sweep before (ms) | sweep (ms) | F before (ms) | F (ms) | gradient before (ms) | gradient (ms) | `parameter_step` before (ms) | `parameter_step` (ms) |
|---|---|---|---|---|---|---|---|---|
| 301 | 3.35 | 2.34 | 1.74 | 1.22 | 10.79 | 9.11 | 18.40 | 12.71 |
| 601 | 8.24 | 4.76 | 4.50 | 2.37 | 24.67 | 18.08 | 52.34 | 27.41 |
| 1 201 | 23.35 | 9.46 | 13.12 | 4.79 | 67.07 | 36.37 | 121.14 | 54.60 |
| 2 401 | 74.76 | 19.21 | 42.88 | 9.88 | 191.88 | 75.45 | 343.49 | 98.17 |
| 4 801 | 261.55 | 39.08 | 150.48 | 19.91 | — | 151.79 | — | 216.33 |

Every column now grows by a factor of 1.8 to 2.2 per doubling. The time per sweep task stays at
2.6–2.7 μs from 301 to 4 801 factors. At 4 801 factors the sweep is 6.7 times faster and the free
energy 7.6 times; at 2 401 factors the gradient is 2.5 times faster and `parameter_step` 3.5 times.
The gradient columns were not measured "before" at 4 801 factors.

**Tests** (`test/state_tree.jl`):
- *Linear in the number of factors.* At 4 times the factors, the sweep and the free energy must
  take less than 8 times as long, with and without state (500 and 2 000 variables). On the code
  before this change, at 1 000 and 4 000 variables, they took 12 to 15 times as long and all four
  checks failed.
- *Write-backs by position match the write-backs by name.* The comparison above, plus each
  sequential sweep against `step!` on the `NamedTuple`, task by task.
- *The residual schedule's probes do not write state.* The counter of a counting source equals the
  number of its committed tasks.

**What it does not do.**
- The coupled gradient drivers (`gradient_couplings.jl`: `_setsubtree` per task, `ps[name]` and
  `_key_node` per key) and the graph adjoint still read by name. Done in I17 ([[gradient_couplings]],
  "Reverse mode by message versions").
- The cold paths (`cluster`, `cluster_batch`, `sampling`, `em`, Lenticulum's `switching.jl`)
  still read by name too.
- Zygote's backward pass through the free energy stayed quadratic; measured and fixed in I16
  ("The backward pass by key" below).

**Difficulties.**
- The timing test is slow to compile: a graph with 4 000 keys gives `NamedTuple` types of 4 000
  fields. On the old code at 4 000 variables the run took two hours, almost all of it compilation,
  so the test sizes are 500 and 2 000.
- A Lenticulum test calls `_row_items` with its old signature, which is kept as a method.

**Review (Mistral, with Julia, in its worktree; [[LP-025 Findings of the review of I5]]).** Its
checks of the semantics, of plates with state and of AD all passed when run again. It found one
defect: `parameter_step` refused an optimiser state given as a `Dict`, which the old code accepted.
That is fixed, with a test. Its suspicion of a per-key scan of the energy rows was not reproduced.

## The backward pass by key (I16, 2026-11-20)

**The cost.** Zygote's pullback of a read from a container builds a cotangent with the container's
shape: reading one subtree of `ps` (by name or by position) gives a `NamedTuple` with an entry for
every key, and reading one message of the store gives a vector as long as the store. The free energy
read `ps` once per factor, and the store once per factor and once per variable, so the cotangents
were added up as O(factors × keys) and O(nodes × edges). The I5 positions did not help: they make the
forward read cheap, not its cotangent. A profile at 481 factors put half the backward time in the
pullback of `messages_into`, not of `ps`; the fix of the `ps` reads alone (the first version)
was 1.3× faster at 121 factors and still grew 3.4–3.9× per doubling.

**What is done.** Every container is read once, for all nodes, and each node's term is computed in
a `map` over the per-node pieces, so Zygote's map pullback hands each node's cotangent back without
widening it:

- `_subtrees(ps, nodes)` (`passing.jl`) gives each node's parameter subtree. Its `rrule`
  (`MyceliumChainRulesCoreExt`) adds the nodes' cotangents per key (tied nodes share one) and builds
  the tree's cotangent once, as a `Tangent` of the caller's `NamedTuple` (its key order); a key no
  node read gets `ZeroTangent`.
- `_states(st, nodes)` does the same for the state, with the same rule.
- `_local_stores(s, reads)` (`free_energy.jl`) gives each node a `_LocalStore`: the messages it reads
  (`_EdgeSlice`, the edges' global numbers and their messages) and the pins of its variables. A
  belief-propagation node reads its incoming messages, a mean-field node the to-variable messages of
  its variables, a variable its own (none for a variable of degree 1, whose term is zero). The per-node
  code is unchanged (`messages_into`, `marginals_into`, `marginal`, `_hard_overlap` read
  `s.to_variable[ei]` and `_pin` on either store). The `rrule` scatters each node's cotangent of its
  messages back to their edges and returns one cotangent of the store, so the gradient through the
  messages of a store built inside AD is kept (scratch check against ForwardDiff, scaled messages of a
  4-step chain). Only the messages go through the rule (`_local_messages`); the pins are read by plain
  code (`_local_pins`), since Zygote keeps a `Dict`'s cotangent in its own context, out of an `rrule`'s
  reach: a pin set inside AD lost its gradient when the rule read them ([[LP-073 Findings of the review of I16]]).
- `scalar_free_energy` adds the factor terms and the variable terms with `_sum_onto(total, map(...))`,
  from the left as the loops did (a pairwise `sum` changed Float32 values, LP-073), with a rule that
  hands every term the sum's cotangent: a loop that indexes the vector of terms gave a cotangent as long
  as the vector per index.

Moving `_hard_overlap`'s reads of `g` outside AD did not help (more allocation, not less); it was
reverted.

**Measured.** `bench/zygote_scaling.jl` (Zygote's gradient of the free energy at fixed messages on
the chain of `state_scaling.jl`; best of 3; the first call is the compilation of the pullback for a
new type of `ps`, one field per factor). Before and after, the same script on the same machine, the
final code after the review's fixes (`~/.cache/lenticulum/i16/bench-{old,final}.log`):

| factors | F before (ms) | F (ms) | gradient before (ms) | gradient (ms) | allocated before (MB) | allocated (MB) | first call before (s) | first call (s) |
|---|---|---|---|---|---|---|---|---|
| 61 | 0.37 | 0.36 | 34.1 | 17.9 | 15.3 | 8.4 | 36.2 | 25.8 |
| 121 | 0.63 | 0.63 | 87.2 | 35.1 | 40.8 | 17.2 | 23.0 | 2.5 |
| 241 | 1.27 | 1.18 | 252.9 | 68.3 | 124.3 | 36.6 | 75.7 | 7.2 |
| 481 | 2.33 | 2.22 | 851.0 | 135.4 | 417.8 | 82.2 | 388.6 | 25.1 |
| 961 | 4.63 | 4.26 | 3110.6 | 286.4 | 1561.6 | 202.1 | 2725.0 | 66.2 |

The gradient's time now grows 1.9–2.1× per doubling, against 2.6–3.7× before; at 961 factors it is 10.9
times faster, and its compilation 41 times. The free energy without AD is not slower: 3.72 against 3.66 ms at 961 factors and 19.1
against 19.1 ms at 4 801 (a scratch script, best of 20, `../Lenticulum.jl-Review-2026-11-20/i16/ftime.jl`).
Three things had made it slower on the way: a pin vector built per node (25%; a store without pins now
gives every node one empty vector), `map` in place of typed comprehensions (1.8×: vectors of varying
element types; `map` is kept only where no rule applies, since Zygote cannot go through a typed
comprehension), and `_subtrees` specialised on the type of `ps`, a `NamedTuple` of one field per factor
(2×; it is `@nospecialize` again). The first row's first
call includes loading Zygote. A scratch run before the review's fixes, at 1 921 factors, took 651 ms and
557 MB against 10.1 s and 4.7 GB for the first version, which fixed only `ps`. The allocations still
grow 2.0–2.5× per doubling (2.7× to 1 921 factors): an allocation profile at 481 and 961 factors put the
rest in Zygote's conversions of the cotangent of `ps` at each function boundary (`unbuffer`,
`wrap_chainrules_output`, a `map` over a tuple with one entry per key), once per call, not per node; not
examined further.

**Tests.** `lib/Lenticulum.jl/test/zygote_keys.jl`, written by Codex from a specification without the
implementation, and the regression tests of the review (LP-073: a pin set inside AD, summation order in
Float32 and under ForwardDiff, a state with one field per factor, the cotangent forms the rule takes): Zygote against ForwardDiff and a finite difference on 20 random chains and trees,
tied keys as the sum of the untied ones, extra, empty and frozen keys, shuffled key order, a missing
key, one factor, plate block rows, and the allocations of one gradient at 32 and 128 steps of the chain
(less than 8× at 4× the steps). On the code before, the last failed (9.5×); now 4.6×.

**What it does not do.**
- At a variable of high degree $d$ a node's reads of its own $d$ messages give cotangents of length $d$
  each, O(d²) (it was O(d × edges)); mean-field factors at a hub each compute the hub's marginal, O(d²)
  already in the forward pass (LP-073, finding 4: 11.2 → 40.7 MB backward for a hub of degree 512 → 1 024).
- The store with plate block rows (`_factor_terms_rows`) still reads `ps` per node for the nodes outside
  its rows, through a buffer; its rows go through the `rrule` of `_row_scalar_energy`. Not measured.
- The compilation of the pullback still grows with the number of keys (66 s at 961 factors): every new
  type of `ps` compiles anew ([[Feature Backlog]], I1).
- The coupled gradient drivers and the graph adjoint (I17) still read by name; done in I17
  ([[gradient_couplings]], "Reverse mode by message versions"). `factor_free_energies`
  shares `_factor_terms` and so its factor reads; its graded result was not measured under Zygote.
- Found in I17: the `rrule` of `_subtrees` read `getfield(ps, i)` in code specialised on the type of `ps`,
  O(keys) per read, so its forward pass was O(nodes × keys) (10.7 MB at 241 keys and 722 nodes). It
  reads through a `@noinline @nospecialize` helper (`_read_slots`) now. The table above was measured
  before; the free energy's gradient at 961 factors was not measured again.

## Exactness checked at run time (P13, 2026-11-16)

**Problem.** `propagate!` reported "tree schedule: one sweep is exact" for every undamped tree of
belief-propagation and `MixtureReduction` nodes. One sweep is then the fixed point, but not always
the exact marginals: a factor that receives a mixture on one edge and sends on another collapses it
to its moments (`Bethe Free Energy`, "What exact requires"), `NonlinearFactor` sends the messages of
a local linear model, and a `MixtureReduction` node merges components. On a switching chain under
belief propagation the report said "exact" while the marginals differ from the enumerated ones by
up to 0.008 in a mean and 0.011 in a variance (`MR_CHAIN`, Lenticulum `test/mixture_rules.jl`; checked
in `test/exact_report.jl`).

**Implementation.** `ConvergenceReport` has a fifth field, `exact` (the four-argument constructor
sets it `false`, so every iterated schedule says not exact). After the tree sweep
`_inexact_reasons(store, g)` lists, for each node that sends a message (an emitting edge): a
`MixtureReduction` rule; a factor type that does not declare `exact_messages(f) = true`; and each
edge on which it received a `MixtureBelief` while sending on another edge. The list empty, the
report is exact; otherwise `reason` reads "one sweep is the fixed point, not exact: …" with every
item. Mixtures that meet at a variable are multiplied exactly and do not count, and neither does a
mixture into a factor that sends only on that edge (a prior), so a switching factor whose mixture
reaches a leaf stays exact.

`exact_messages` is opt-in: `false` by default, declared for Mycelium's `DataFactor`,
`PriorFactor`, `RelayFactor`, `TableFactor` and `ClusterFactor`, and for Lenticulum's
`GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `HardConstraintFactor` and
`SwitchingConstraintFactor`, each tested against a closed form. A factor nobody declared (a
`LuxFactor`, a diffusion factor, `NonlinearFactor`, `LikelihoodFactor`, `RegimeFactor` outside a
cluster) makes the report say so, rather than call its messages exact.

**What it does not do.** It is cautious with `MixtureReduction(M)`: with a budget no message
exceeds, nothing is merged and the marginals are exact, but the report says not exact (the store
does not show whether a reduction happened). It reads the rule and the message types, not the
messages' values, so a factor declared exact but given a belief type it handles approximately other
than a mixture would pass unnoticed; the declared factors' other inputs (point, categorical,
Gaussian, flat) are exact. The loopy-graph hole of the section above (a hand-built `TreeSchedule`)
is still open.

**Tests.** Lenticulum `test/exact_report.jl` (99 checks): the switching chain under belief
propagation, not exact and its marginals more than 0.005 from enumeration; `MixtureReduction(1)` and
`(16)`, not exact; the same chain in one cluster node, exact and the enumerated means; the
additive wiring, collapsed at the sum factor and exact once clustered; 20 random priors times a
switching factor's mixture, exact and equal to the closed-form mixture moments and weights;
`NonlinearFactor`, not declared; sequential and damped schedules, never exact. The 40 random
Gaussian trees of `test/gaussian_oracle.jl` and Mycelium's chain (data, relay, absorbing loss)
assert `rep.exact`.

## A relative convergence criterion (P9, 2026-11-17)

**Why.** The default criterion, a sweep's largest `belief_distance` at most `tol`, is absolute in
the natural parameters, so it depends on the units of the model. The loopy voltage divider of
Lenticulum's `runtests.jl`, written in units $c$ (every value times $c$, every variance times $c^2$,
so every precision over $c^2$), was run under flooding (Lenticulum `test/p9_relative_tol.jl`, measured
2026-11-17). At $c = 10^{-4}$ (precisions about $10^8$) `tol = 1e-11` is below the rounding of the
messages and is never met in 3000 sweeps. At $c = 10^4$ `tol = 1e-8` is met after 205 sweeps with the
means still $3.4\cdot10^{-7}$ off, relative to the dense joint solve. At $c = 1$ the same `tol` gives
$9\cdot10^{-12}$.

**What.** `propagate!(...; rtol)` with `rtol > 0`: a message has converged when each group of its
parameters changed by at most `tol + rtol·s`, with `s` the larger $\|\cdot\|_\infty$ of that group on
the two sides. The groups are $\eta$ and $\Lambda$ of a `GaussianBelief`; $\mu$, $\Sigma$ and the
flat-direction projector (scale 1) of an `ExtendedGaussian`; the point of a `DiracBelief` (Euclidean
norms); the weights (scale 1) and the components of a mixture. Every other pair gets its
`belief_distance` at scale 1 (`relative_residual` in `messages.jl`). Under `criterion = :marginals`
the groups are the means and the covariances (`_moment_residual`). The residual is reported in units
of the bound and convergence means at most 1. On the divider with `tol = 0, rtol = 1e-10`, flooding
takes 337 sweeps at every $c$ from $10^{-4}$ to $10^4$, with relative error $4.6\cdot10^{-11}$. The
spanning-tree, Anderson and residual schedules and `:marginals` also take the same number of sweeps
at every scale, within two. `rtol = 0`, the default, keeps the absolute criterion unchanged; `tol` is
the floor for groups near zero.

**How it reaches the sweeps.** The tolerance is a `ScopedValue` (`_RELATIVE_TOL`) that
`propagate!` sets around its loop. `_step!`, the plate sweeps (`_block_residual`, with column-wise
reductions for Gaussian and point blocks in `blocks.jl`) and through them the residual and Anderson
schedules read it. So no keyword is threaded through every `sweep!` method, and a `sweep!` called
outside `propagate!` still reports absolute distances. The residual schedule's priorities are then
relative too, which is the point: an edge with large precisions no longer outranks every other edge
because of its units.

**What it does not do.** A group's scale is its own largest entry, so a Gaussian whose mean is near
zero has $\eta \approx 0$ while its rounding is of order $\epsilon\|\Lambda\|\|\mu\|$; there `tol` is
the floor, and `tol = 0` may not be met. The scale of $\eta$ is not $\|\Lambda\|\|\mu\|$, because
improper messages have no mean. Sample and potential beliefs have no scale (scale 1). The default stays
absolute: making `rtol` the default would change every schedule's stopping point. Dual numbers: the
residual math and the tree schedule differentiate under `rtol > 0`. Whether a change is zero is
decided on the primal value (`_primal`), since a dual's comparisons break ties by the partials
(LP-051). An iterated `infer!` with dual parameters used to throw in `ConvergenceReport` whatever
`rtol` was; since I23 the report holds the primal residual, `_iterate!` decides convergence on it,
and the derivative of the fixed point's means matches the dense solve on flooding, spanning-tree and
residual schedules, under both criteria (`test/p9_relative_tol.jl`, "I23"). With duals the sweep
count can differ by one from the float run, since dual matrices take the generic linear algebra and
round differently. The residual schedule chooses its next task on the primal residual too, so the
tangents do not change the order of the updates (LP-053). `AndersonSchedule` runs without its memory
under duals (`_anderson_state` mixes only `Float64` messages): its derivatives are right, but it takes
the inner schedule's sweeps. The derivative is that of the iterates at the stopping sweep, not the implicit
derivative of the fixed point; it is close to it only when the iteration has converged. Not tested: device blocks under the relative path (the block methods use reductions and
no scalar indexing, but were not run on JLArrays).

**Found on the way.** Under flooding, `criterion = :marginals` can stop too early: flooding moves
information one hop per sweep, so on the divider its second sweep changes messages but no marginal,
and `propagate!` reports convergence after 2 sweeps with every mean still at the prior (relative
error 1), under the absolute criterion as well. The spanning-tree, Anderson and residual schedules
are not affected. Fixed by I22 (two quiet sweeps, "Convergence on marginals" above); the
`@test_broken` in `test/p9_relative_tol.jl` is now its regression test.

## The tree shortcut and cavity-reading messages (GH-15, 2026-11-17)

The one-sweep shortcut of `propagate!` on a `TreeSchedule` checked only the inference rule. A BP
`NonlinearFactor` linearises its backward message at the target's cavity, so one collect/distribute
pass need not be a fixed point, yet the report claimed `converged` after one sweep. The shortcut is
now restricted to messages guaranteed independent of the target cavity: the new trait
`message_uses_cavity(factor)` (default `false`, declared `true` by `NonlinearFactor`) is part of
`_tree_exact`, and a tree with a cavity-reading message is iterated with its residual checked like
any other graph, so the report distinguishes actual convergence from the shortcut. Follow-ups, not
done: `DEQFactor` and `DiffusionFactor` (ImplicitLayers, VariationalDiffusion — off the reviewed
paths) also read the `prior` in their messages and should declare the trait when those packages are
reviewed. Tests: Lenticulum `test/gh15_nonlinear_tree.jl` — rooted at `:prior` another sweep moves the
marginal, rooted at `:data` one sweep is the fixed point and the second changes nothing; the report no longer claims a one-sweep fixed point, and a linear-Gaussian tree
keeps the one-sweep exact report.

## The residual schedule's queue (I4, 2026-11-20)

A `ResidualSchedule` sweep used to probe every legal task (`_step!` with `commit = false`) before each
of its commits: $T^2$ message evaluations for a sweep of $T$ commits, 2.1 times flooding's on the
divider to convergence and out of reach on a grid of a few thousand tasks. It now keeps the residuals
in an indexed binary heap (`_TaskHeap`). A sweep probes every task once, then pops the task of largest
residual, commits it, and probes again only the tasks whose message can read the committed one
(`_dependents!`):

- a message to factor $f$ on edge $e$ is an input of $f$'s messages on its other edges (not of a
  `MeanField` node's, which read marginals);
- a message to variable $v$ on edge $e$ (unless $v$ is pinned: its cavity and marginal are the pin) is
  in $v$'s cavity on its other edges: the messages to their
  factors read it, and so do the messages *from* those factors if they read the target's cavity
  (`_reads_cavity`);
- it is also in $v$'s marginal, which a `MeanField` node next to $v$ reads for its messages on edges
  other than one to $v$;
- a damped message reads its own previous value, so it is probed again after its commit;
- a commit that changed a factor's state can change every message of the factors sharing its
  parameter key, which are all probed again.

A committed message $b$ that is in none of these sets would equal itself if probed again, so its key is
the distance of $b$ to itself, computed without calling the factor: 0, but `Inf` for a belief without a
distance and NaN for one with a NaN (LP-072 F1). Nothing is probed after the budget's last commit. The keys
are the residuals' primal values as `Float64`, or as `Real` once a residual of another type appears
(`BigFloat` residuals rounded to `Float64` tied or became 0, LP-072 F2). The heap orders by residual, then the older message, then the position in
`all_tasks(g)`, which is the order the scan chose by; a skipped update or a NaN residual gets the key
$-1$ and is never popped (the scan never chose them either). So the residuals in the heap are the ones
the scan would compute, and **the queue makes the same commits in the same order, bit for bit**. The
scan is kept as `_residual_sweep_scan!`, the reference of `test/residual_queue.jl` (Lenticulum): 40
random loopy graphs × damping 0 and 0.3 × `maxtasks` all and 3, eight sweeps each, with mean-field,
expectation-propagation, nonlinear, cavity-reading and stateful tied nodes; record, store and state
compared with `==`. Each dependency rule was removed in turn and the test failed each time
(5 to 79 of its comparisons). The review (Codex, [[LP-072 Findings of the review of I4]]) found three
differences, now fixed with regression tests in the same file, and one accepted: on a graph that throws
(contradictory hard clamps) both throw, at different commits, since an exact message builds its cavity
without reading it and the queue does not probe it again.

**Which messages read the cavity.** Every rule but `BeliefPropagation` and `MixtureReduction` reads it.
Under those two, `message_uses_cavity` is not a guarantee: it defaults to `false`, and `DEQFactor` and
`DiffusionFactor` read `prior` without declaring it (GH-15 above). So a factor counts as not reading the
cavity only when it declares `exact_messages` (an exact sum-product message does not depend on the
target's cavity) and not `message_uses_cavity`. A wrong `true` costs probes; a wrong `false` would change
the commits silently. The factors that declare `exact_messages` were read for a use of `prior` in their
messages (none found).

**Measured** (`bench/residual_queue.jl`, Gaussian $n \times n$ grids of `bp_grid.jl`, one machine (8 cores),
Julia on one thread, 2026-11-20, after the review's fixes). One sweep from an empty store, the scan against the queue, the commits checked
equal:

| grid | tasks $T$ | scan | queue | faster | evaluations, scan ($T^2$) | queue |
|---|---|---|---|---|---|---|
| 3 × 3 | 66 | 0.0081 s | 0.0004 s | 19× | 4 356 | 180 |
| 6 × 6 | 312 | 0.197 s | 0.0022 s | 89× | 97 344 | 966 |
| 10 × 10 | 920 | 1.74 s | 0.0067 s | 261× | 846 400 | 2 966 |
| 32 × 32 | 9 984 | not run | 0.078 s | | | 33 414 |

The queue's evaluations are about $3.3\,T$ per sweep: $T$ for the first probe of every task and about
2.3 per commit. To a message residual of $10^{-10}$, against flooding:

| grid | prior variance | flooding: sweeps, evaluations, time | residual: sweeps, evaluations, time |
|---|---|---|---|
| 4 × 4 | 1 | 87, 11 136, 0.029 s | 30, 8 803, 0.048 s |
| 4 × 4 | 100 | 4 289, 548 992, 1.50 s | 862, 251 463, 0.95 s |
| 8 × 8 | 100 | 6 305, 3 631 680, 10.5 s | 1 500, 2 034 646, 8.1 s |
| 16 × 16 | 1 | 113, 274 816, 0.85 s | 37, 216 156, 0.87 s |
| 16 × 16 | 100 | 7 063, 17 177 216, 52.2 s | 1 583, 9 165 915, 37.7 s |

The residual schedule now needs 0.46–0.89 of flooding's evaluations and 0.64–1.66 of its time (the heap
and the probes' bookkeeping; the 1.66 is the 4 × 4 grid's 0.05 s run, 1.07 in the run before the
review's fixes, so timing noise at that size; the large runs 0.72–0.77); before, the same commits cost $T$ times flooding's evaluations per sweep
(on the 16 × 16 grid at prior variance 100, about $9\cdot10^{9}$). Its means are further from the dense
solve at the same message tolerance ($4\cdot10^{-9}$ against $10^{-10}$ at prior variance 100): the stop
is decided on the sweep's committed residuals, and the schedule leaves smaller residuals pending in
the slow modes. That is the schedule's, not the queue's (the commits are the scan's); not examined further.

**Not done.** The first probe of every task in each sweep is a third of the cost; keeping the heap
between sweeps would save it but needs the store to say when it was changed outside a sweep (a `pin!`,
an edit, a user's write). `_max_pending` (I24's check at the stop of a partial sweep) still scans every
task once. Plates (`PlateSchedule`) have no residual schedule.
