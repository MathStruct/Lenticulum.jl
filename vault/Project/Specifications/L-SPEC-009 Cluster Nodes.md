#design

> **Status: draft (2026-11-03, Claude), implemented as described.** Related: [[Extending Mycelium]] §5, [[L-SPEC-008 Expectation Propagation and Mixture Reduction]], [[Switching Constraint Factors]], [[Composing Stochastic Factors]] (E7). Code: `lib/Mycelium.jl/src/cluster.jl`; Lenticulum `src/switching.jl` (`cluster_form` methods).

> Sources: Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. Inf. Theory 51(7), 2005, the region free energy (here regions are solved exactly, not approximated). code: `lib/Mycelium.jl/src/cluster.jl`, `src/switching.jl`; tests: `test/cluster.jl`, `lib/Mycelium.jl/test/cluster.jl`
>
> Bibliography: [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Scope

A graph operation that replaces a set of factor nodes by one node with exact inference inside,
for conditionally linear-Gaussian factors. Automatic grouping, approximate regions (generalised
belief propagation with overlapping regions) and clusters inside plates are out of scope.

## The contract

**§1. The transformation.** `cluster(g, name, factors)` returns a new graph in which the named
factor nodes are one `ClusterFactor` node `name`. A variable whose every edge is to a clustered
factor becomes internal; every other variable a clustered factor touches becomes a channel of the
node, named after the variable, on a bidirectional diagonal edge. The other nodes and edges are
copied unchanged. A variable is discrete when a clustered factor switches on it
(`cluster_switch`); every variable has an `Int` dimension. Refused: plates, a missing or taken
name, a factor without `cluster_form`, a factor not on `BeliefPropagation()`, frozen parameter
paths, a parameter key shared with a node outside, a clustered edge that is not bidirectional,
diagonal and attached.

**§2. The factor form.** `cluster_form(factor, ps)` returns `ClusterForm(A, C, Q, logw, switch)`:
$-\log f(x, k) = \tfrac12 r^\top Q^{-1} r + \tfrac12 \log\det 2\pi Q - \log w_k$ with $r = \sum_c A_c x_c - C_{:,k}$; `switch` the
discrete channel $k$ is read from (`nothing`: one column). $A$ and $Q$ do not depend on $k$.

**§3. Messages.** To a channel, given the messages on the others: the exact message of the
product of the clustered factors, with internal variables integrated and summed out. Discrete
states are enumerated (at most $2^{16}$ joint states); continuous variables are eliminated in
canonical form; points are substituted; a Gaussian-mixture input is collapsed to its moments. To
a continuous channel the message is a mixture over the joint states, or a `GaussianBelief` for one;
to a discrete channel a `CategoricalBelief`. Continuous variables not determined by the factors and
the messages are refused.

**§4. Free energy.** The node's local free energy is $\mathbb E_b[-\log \prod_f f] - H(b)$ for its exact
belief $b \propto \prod_f f \prod_v \mu_v$, graded `(fit, complexity, negentropy)`; `factor_belief` and
`belief_free_energy` give its parameter gradient at a fixed belief. Internal variables do not
appear in the graph's counting numbers.

**§6. Several clusters, tied, and grouping by overlap** (planned and implemented 2026-11-03).
`cluster(g, groups::Vector{Pair}; params)` makes one node per group; with `params` the nodes read
one entry (tied), and a key shared between groups is allowed only then. `overlap_groups(C,
maxsize)` is E7's greedy grouping; Lenticulum's `switch_overlaps(g, ps, sum, switches)` gives the
switches' overlaps as they feed one linear factor (largest $|\cos|$ of their offset directions in the
sum's total-noise metric). Grouping does not rewire by itself: without E7's partial sum per block a
cluster of switches meeting only at the sum gains nothing (`cluster.md`). Lenticulum's
`partial_sums(g, ps, sum, switches, groups; noise)` makes that edit (added 2026-11-03,
`switching.md` §14) and returns the groups to cluster.

**§7. Joint outcomes and discrete tables** (added 2026-11-03). A `ClusterForm`'s `switch` may be a
tuple of discrete channels, the outcome their joint state in column-major order. `TableFactor(dims,
logtable)` is a fixed factor over discrete variables with such a form; its `-Inf` entries tie
variables together. The node drops the assignments of weight zero before its Gaussian part, and a
zero-weight outcome contributes nothing to the free energy ($0 \log 0 = 0$).

**§8. Over a plate** (added 2026-11-04). `cluster(g, plate, name, factors; params)` makes one tied
node per replica of the plate's template factors and keeps the plate; inside, variables carry
template names. `as_plate(g, name, n)` records the nodes named `x[i]` as a plate. For tied nodes
with the same factors and inputs of one shape, messages and free energies are batched
(`factor_messages`, `factor_beliefs` → `FrozenClusterBeliefs`, `belief_free_energies`); otherwise
per node. Values equal the unplated graph's.

**§9. Per-outcome coefficients and noise** (added 2026-11-04). An entry of a `ClusterForm`'s `A`
may be an $m \times d \times K$ array and its `Q` an $m \times m \times K$ array. Assignments are grouped into
precision classes (one per combination of the varying factors' outcomes), each eliminated with its
own precision; a message to a continuous channel is then a mixture whose components may differ in
precision, and is refused if improper components differ. Such nodes are not batched over a plate.
`as_cluster(f)` is one factor as a cluster node, for a factor with only a `cluster_form`; Lenticulum's
`RegimeFactor` ($0 = \sum_i A_{i,k} x_i - b_k + \varepsilon_k$, $\varepsilon_k \sim \mathcal N(0, Q_k)$) is one.

**§5. Parameters and marginals.** The node's parameters are keyed by the clustered factors'
parameter keys; `cluster_parameters(gc, name, ps)` converts a parameter tree of the original graph.
`cluster_marginal(store, gc, name, v, ps)` is the marginal of a cluster variable under the node's
belief.

## Checked by

`test/cluster.jl`: a switching chain in one cluster against enumeration (free energy $= -\log p(y)$ to
$10^{-12}$, marginal moments to $10^{-12}$, the gradient in tied offsets against central differences);
three switches in one cluster against E7's block factor (messages to $10^{-12}$, marginals of the
same loopy schedule and local free energies to $10^{-10}$); the cluster of the sum and the switches
has $-\log p(z)$ where the plain graph has no free energy; the refusals. `lib/Mycelium.jl/test/cluster.jl`:
the same on Mycelium's own prior factors.

Grouping (`test/cluster.jl`, `bench/cluster_overlap.jl`): `switch_overlaps` equals E7's normalised
Gram to $10^{-12}$ and `overlap_groups` E7's `overlap_blocks` for block sizes 2–4; on eight switches in
overlapping pairs, blocks of 4 by overlap (with partial sums, made by `partial_sums`) cut the mean error of the switch
posteriors from 0.092 to 0.031, blocks in index order to 0.092.

Tables (`lib/Mycelium.jl/test/cluster.jl`): three categorical variables and two tables with $-\infty$
entries in one cluster against enumeration ($F = -\log Z$ and marginals to $10^{-12}$), and belief
propagation through the tables on the same tree.

Per-outcome forms (Lenticulum `test/regime.jl`): a mixture of factor analysers with three
precisions in one cluster against the closed form ($F = -\log p(z)$, the regime posterior and the
posteriors of $x$ to $10^{-12}$; the gradient in $A$, $b$, $\log w$ and the noise against central differences
to $10^{-6}$, ForwardDiff and Zygote), the one-factor cluster's messages against the closed form, and
a plate of four points.

## Open

- Flat continuous directions inside a cluster; batching per-outcome forms over a plate.
- Done 2026-11-03/04: clusters in plates (§8), E7's partial sums from the groups (Lenticulum's
  `partial_sums`), merging repeated components of a message, pruned enumeration, per-outcome
  coefficients and noise (§9).
