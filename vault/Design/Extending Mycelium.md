#design

> A plan for growing Mycelium from a belief-propagation engine into a general inference engine for
> hybrid (discrete and continuous) factor graphs. It has seven parts:
>
> - per-node message rules, each with its own free energy;
> - learning through graphs with mean-field nodes;
> - expectation propagation and mixture reduction as rules;
> - cluster nodes with exact inference inside;
> - plates with shared parameters;
> - structure edits on a live graph;
> - schedules that know the rules.
>
> The first step, the message rule, is done ([[L-ADR-0009 Factor nodes carry a message rule]]), and
> so are the free energy of the hybrid (§1, [[L-SPEC-005 The Hybrid Free Energy]]), plates (§2,
> [[L-SPEC-006 Plates]]) and rule-aware schedules (§3, [[L-SPEC-007 Hybrid Schedules]]). The rest is
> proposed, ordered by what it unblocks.

> Sources: original to this vault (design). Hybrid belief propagation and mean field: Riegler, Kirkelund, Manchon, Badiu & Fleury, *Merging Belief Propagation and the Mean Field Approximation: A Free Energy Approach*, IEEE Trans. Inf. Theory 59(1), 2013. Region free energies and generalised belief propagation: Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. Inf. Theory 51(7), 2005. Expectation propagation: Minka, UAI 2001. Gaussian-mixture reduction: Runnalls, IEEE Trans. Aerosp. Electron. Syst. 2007. Monte Carlo EM: Wei & Tanner, *A Monte Carlo Implementation of the EM Algorithm and the Poor Man's Data Augmentation Algorithms*, JASA 85(411), 1990. Stochastic-approximation EM: Delyon, Lavielle & Moulines, *Convergence of a Stochastic Approximation Version of the EM Algorithm*, Ann. Statist. 27(1), 1999. code: `lib/Mycelium.jl/src/graph.jl`, `passing.jl`, `free_energy.jl`, `schedules.jl`, `exposure.jl`
>
> Bibliography: [[Bibliography#^riegler2013merging|Riegler et al. 2013]] · [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^minka2001ep|Minka 2001]] · [[Bibliography#^runnalls2007kl|Runnalls 2007]] · [[Bibliography#^wei1990montecarlo|Wei & Tanner 1990]] · [[Bibliography#^delyon1999saem|Delyon et al. 1999]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

## Where Mycelium stands

Mycelium builds a bipartite graph once (`GraphBuilder`, `build`), passes messages by a schedule
(`step!`, `sweep!`, `propagate!`), scores the result by the Bethe free energy (`bethe_free_energy`,
counting numbers $1 - d_v$), and trains parameters by its gradient at fixed factor beliefs (`parameter_step`, LP-020),
with schedule adjoints for coupled edges. Each factor node now carries a message rule.

What the experiments behind [[Composing Stochastic Factors]] ran into, in order of cost:

| gap | where it showed | consequence |
|---|---|---|
| only belief propagation | E6: a sum node fed by switches sees the other switches' prior moments | biased learning, 0.5–0.8 nats; fixed for one node by `MeanField()` |
| no free energy for hybrid graphs | E7 in the graph | `parameter_step` refused; the M-step was written by hand (resolved by §1, 2026-11-02) |
| one graph per data point | EM through the hybrid graph | 14 times slower than the vectorised code (28 s against 2 s); plates and tying added by §2 (2026-11-02), the per-message overhead remains |
| no mixture entropies at variables | the Bethe sum with a mixture marginal of degree $\ge 2$ | no free energy for most hybrid graphs |
| mixtures only collapse to moments | E6, E7 | the only policy is assumed-density filtering |
| frozen graphs | structure search over switch wirings | structure moves rebuild models outside Mycelium |
| blocks built by hand | E7: a block of switches is a hand-made factor with $2^B$ outcomes | each block is new code |

## 1. A free energy for every rule (first)

**Status: implemented for `MeanField()` (2026-11-02)**, [[L-SPEC-005 The Hybrid Free Energy]]. The
counting numbers are those of Riegler et al. (2013), §III, eqs. (16)–(17), checked against the
paper: $c_i = 1 - |N_{BP}(i)|$, mean-field nodes with an average energy and no entropy. The hook is
`mean_field_energy(factor, marginals, ps, st)` rather than a rule argument to `local_free_energy`.
Tests: naive mean field in closed form, Monte Carlo for both average energies, a bound against
exact enumeration on a sum of switches, the gradient against central differences, and descent
under coordinate updates for naive mean field. The E7 block-graph descent and closed-form M-step
checks below were not run as written; the gradient is checked against central differences of the
converged free energy instead.

**What.** Each message rule declares the free energy it is the stationary point of, and the graph sums
them. For belief propagation that is Bethe. For the hybrid it is the free energy of Riegler et al.
(2013), in which:

- belief-propagation factors contribute their Bethe terms;
- mean-field factors contribute only an average energy $\mathbb E_{\prod_i q_i}[-\log f]$, with no entropy of their
  own;
- the variable entropies are counted from the belief-propagation neighbours.

The exact counting numbers are to be taken from the paper's statement before implementing; this
sketch is not a substitute for it.

**Before it: LP-020 (resolved 2026-11-01).** The graph gradient held messages, not beliefs, fixed,
and was not the gradient of the Bethe free energy. It now holds the beliefs fixed
(`factor_belief`, `belief_free_energy`). A free energy for the hybrid needs the same for its
mean-field nodes, whose beliefs are products of marginals.

**Why first.** It turns the refusal of [[L-ADR-0009 Factor nodes carry a message rule]] into an
answer, and with it `parameter_step` trains hybrid graphs: EM becomes coordinate descent on one
objective, inside Mycelium.

**How.**

- `local_free_energy` gets a rule argument. The mean-field term needs only the factor's expected
  energy under a product of marginals, which every residual factor can provide from means and
  covariances.
- `variable_corrections` counts belief-propagation degrees.
- `_check_energy_edges` then refuses only rules without a free energy.

**Test** (expected properties, each with an oracle):

- with every node on belief propagation it is Bethe, unchanged;
- with every node on mean field it is the naive mean-field free energy, checked by Monte Carlo;
- on the block graph of E7, sequential updates never increase it;
- at a fixed point its parameter gradient equals the closed-form M-step.

## 2. Plates: one graph, many data points, shared parameters

**Status: implemented (2026-11-02)**, [[L-SPEC-006 Plates]], `lib/Mycelium.jl/src/plates.md`. Tying is
general (`factor!(...; params)`), the plate builder unrolls the template with tied factors, and
`plate_schedule` runs each template task for all replicas at once through `factor_messages`.
Checked against the loop of $N$ graphs (marginals, free energy, gradients with three backends) and
closed forms, including a shared latent variable and a plate of hybrid graphs. The speed target
was not met: for factor analysis the grouped plate is about 6.3 times faster than a loop of
prebuilt graphs and 170–240 times slower than the hand-written E-step, after the fixes of a
review (graph types that grew with the graph, per-replica work in batched groups; `plates.md`). The comparison with E7's vectorised code was not rerun (E7 lives on
`experiment-qm`). A plate is unrolled rather than batched; "factors inside see batched inputs"
holds only through `factor_messages`.

**What.** A graph template replicated over $N$ observations with tied parameters, run as one batched
graph: beliefs with a batch dimension, messages computed for all replicas at once, and gradients
summed over replicas.

**Why.** Learning through a graph is the point of the library. Today it means $N$ graphs and a loop,
and the docs list "no parameter sharing between factor nodes" as a gap. [[L-SPEC-003 Exposed Parameters and Energy Channels]]
already ties parameters by exposing them as channels on a shared variable. A plate is the
batched, structured form of the same thing.

**How.**

- `Plate(name, n)` in the builder, with variables and factors declared inside or outside it.
- Factors inside see batched inputs. A factor without a batched method falls back to a loop over
  replicas, so every existing factor works and the fast ones opt in.
- Parameters outside the plate are shared, and their gradient is the sum over replicas (as for exposed
  parameters).

**Test.** A batched graph equals the loop of $N$ graphs to rounding, in marginals, free energy and
gradients. Speed is measured against the vectorised E-step of E7, with the target of being within a
small factor of hand-written code.

## 3. Schedules that know the rules

**Status: implemented (2026-11-03)**, [[L-SPEC-007 Hybrid Schedules]]. `hybrid_schedule(g)` as
described, `propagate!(...; criterion = :marginals)` comparing marginals by their moments (the
default for hybrid schedules), and `plate_schedule(g)` for hybrid templates. It reproduces the
hand-ordered hybrid of `test/switching_constraint.jl` to $10^{-9}$ at a tolerance of $10^{-10}$ (the hand
schedule needed $10^{-6}$ on messages), and the free energy does not increase along it with
components of width $10^{-3}$. Tutorial 11 uses it.

**What.** `hybrid_schedule(g)`: belief-propagation subtrees swept inward and outward around each
mean-field node, and mean-field nodes updated in turn. It replaces the hand-written task lists of the
current tests. Convergence would be measured on beliefs (marginals), not only on messages, because
mixtures of very narrow components have a rounding floor in `belief_distance`.

**Test.** It reproduces the hand-ordered hybrid of `test/switching_constraint.jl`, and the free energy
of §1 decreases along it.

## 4. Expectation propagation and mixture reduction as rules

**What.** Two more rules:

- `ExpectationPropagation()` projects the tilted belief (cavity times factor) onto the target's
  family and divides the cavity out. Gaussian division is subtraction in canonical form, so it is
  defined even for improper sites.
- `Reduce(maxcomponents)` lets a node send mixtures reduced to a budget (Runnalls' merge,
  `reduce_mixture` exists) instead of collapsing them to moments, which is the only policy today.

**Why.**

- E6 measured Gaussian EP on the additive wiring as a per-point loop: better pair moments at $L = 8$
  (0.16–0.40 nats), unstable at $L = 12$. As a rule it can be tried on any graph, with damping.
- Mixture reduction is the standard middle ground between exact mixtures, which grow, and
  assumed-density filtering, which collapses.

**Test.**

- EP on Minka's clutter problem against the exact posterior in small cases.
- `Reduce` on a switching linear chain against exact enumeration, as the budget grows.

## 5. Cluster nodes: exact inference inside, messages outside

**What.** A node that groups several factors and variables and runs exact inference inside
(enumeration of its discrete states, Gaussian elimination of its continuous ones). Outside it looks
like one factor, with messages and a closed-form local free energy. This is the region of
generalised belief propagation (Yedidia, Freeman & Weiss 2005), restricted to regions small enough to
solve exactly.

**Why.** E7's remedy, exact blocks of overlapping switches, is a cluster, but today each block is a
hand-built `SwitchingConstraintFactor` with $2^B$ outcomes. A cluster node would make "group these
switches" a graph operation. Grouping by overlap (largest $|\cos|$ of the offsets in the noise metric)
becomes a transformation of the graph. A cluster over $(k, z)$ also gives the closed-form entropy that
a mixture *variable* lacks, which is the cleanest answer to the mixture-entropy gap.

**Test.** A cluster of $B$ binary switches equals the hand-built block factor in messages and free
energy. With every factor in one cluster, the graph is exact.

## 6. Structure edits on a live graph

**What.** Adding and removing variables, factors and edges on a built graph, keeping and
warm-starting the messages that remain valid ([[Dynamic Graphs]] has the analysis). The free energy
of §1 scores the edits.

**Why.** Greedy structure search over switch wirings worked on the branch (four moves scored by BIC,
generating structures recovered), but it rebuilt models outside Mycelium, with exact scores.

**Test.** The moves of `structure_switches.jl`, done as edits, reproduce its search paths and scores
where exact inference is available.

## 6b. Sampling, and what an average must keep

**Status: implemented (2026-11-01)** in `lib/Mycelium.jl/src/sampling.jl` ([[sampling]]): pins, the
Gibbs sampler, the buffer with window and decay filters and a horizon, and the Monte Carlo EM gradient
(one inference per sample, then the fixed-belief gradient of LP-020). Building it exposed [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]].

**What.** Passing points instead of beliefs: a Gibbs sampler over chosen variables. The switches are
sampled; the continuous variables between them are integrated out exactly by belief propagation
with the sampled ones pinned (Rao-Blackwellisation). Beliefs then never grow, each update is local,
and learning is Monte Carlo EM (Wei & Tanner 1990): the parameter gradient is the average, over
posterior samples, of the pinned graph's free-energy gradient. With all sampled variables pinned
and the rest a tree, that free energy is $-\log p(\text{data}, \text{pins})$.

**Requirement: averaging a structured energy.** A vector energy averaged the wrong way loses exactly
the structure it was made to keep. Every sampling or point-passing rule, and every report built on
samples, must satisfy the following.

1. **Scalarise each sample, then average, one graded component at a time.** Never average a raw
   residual vector across samples or outcomes before its scalarisation: residuals of different
   samples cancel (two outcomes with right-hand sides $\pm 1$ average to a zero residual although
   neither constraint holds). This is L-ADR-0003's two levels, applied to samples.
2. **Report the mean and the full covariance of each factor's residual vector.** For a squared-norm
   scalarisation they are exact sufficient statistics,
   $\mathbb E\lVert r\rVert^2_W = \lVert \mathbb E r\rVert^2_W + \operatorname{tr}(W \operatorname{Cov} r)$,
   so the trace term, the Jensen gap, stays visible and checkable. When a sample carries a Gaussian
   part (Rao-Blackwellisation), the covariance is combined by the law of total covariance: the
   average of the within-sample covariances plus the covariance of the sample means.
3. **A diagonal covariance is a declared reduction, valid only for a diagonal weighting.** With a
   full noise matrix the off-diagonal covariances enter the trace.
4. **Nonlinear scalarisations need the samples themselves.** Mean and covariance are not sufficient
   for robust or point-dependent scalarisations (L-ADR-0007), so the samples are kept.

**Interface: samples are kept, filtered, and forgotten.**

- A **sample buffer** keeps each sample's points, its graded local energies, the residual moments
  of every factor that has them, and the parameter step at which it was drawn.
- A **filter** turns the buffer into weights. A window keeps the last $n$ samples with equal
  weight. An exponential filter weighs a sample $a$ draws old by $\lambda^a$; with a decreasing step size this
  is stochastic-approximation EM (Delyon, Lavielle & Moulines 1999).
- **Forgetting:** after `horizon` parameter steps a sample is dropped, since samples drawn under old
  parameters bias the gradient.
- Averages, residual moments and gradients are always computed from the buffer under the filter's
  weights. The gradient re-runs belief propagation at the *current* parameters with each stored
  sample pinned.

**Test.**

- The Gibbs conditional is exact (frequencies against enumeration).
- With the buffer holding every configuration at its exact posterior weight, the gradient equals
  the gradient of the exact $-\log p$, and the residual moments equal the enumerated ones.
- The filters' weights and the forgetting are checked deterministically.

## 7. Factors that follow from this (Lenticulum)

These belong to Lenticulum, not to Mycelium, but the plan depends on them:

- the **general switching factor** (per-outcome $A_k$, $Q_k$; [[Switching Constraint Factors]] §8);
- the **directed switching kernel** for chains;
- **mean-field rules** for the remaining residual factors, `GaussianFactor` and `NonlinearFactor`
  (linearised).

## Order

1. The free energy of every rule (§1). It unblocks learning through hybrid graphs.
2. Plates (§2). They make that learning affordable.
3. Rule-aware schedules (§3). They are small, and §1's tests need them.
4. EP and reduction rules (§4), then cluster nodes (§5). They widen what the graph can approximate.
5. Structure edits (§6). They need §1's score.

Sampling (§6b) is independent of the others and is implemented first, by Daniel's choice
(2026-11-01): it needs no new free energy, because the pinned graph's Bethe free energy is exact.

Each part gets an LP or an L-SPEC when it starts, with its tests written against the oracles named
above.

Related: [[L-ADR-0009 Factor nodes carry a message rule]], [[Composing Stochastic Factors]], [[Switching Constraint Factors]], [[Bethe Free Energy]], [[Loopy Message Passing]], [[Dynamic Graphs]], [[Schedules]]
