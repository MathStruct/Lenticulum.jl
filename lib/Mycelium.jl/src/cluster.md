#implementation

> Cluster nodes: `cluster(g, name, factors)` replaces a set of factor nodes by one node that runs
> exact inference inside (enumeration of the discrete states, Gaussian elimination of the
> continuous ones) and looks like one factor outside, with exact messages and a closed-form local
> free energy. A whole switching chain in one cluster has the free energy $-\log p(y)$ to
> $10^{-12}$; three switches in one cluster send the messages of E7's hand-built block factor.

> Sources: Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. Inf. Theory 51(7), 2005 (regions; here a region is solved exactly); de Jong, *The Diffuse Kalman Filter*, Ann. Statist. 19(2), 1991 (the diffuse likelihood: flat directions). code: `cluster.jl`; Lenticulum `src/switching.jl` (`cluster_form` for the constraint factors and `GaussianPrior`); tests: `../test/cluster.jl`, Lenticulum `test/cluster.jl`
>
> Bibliography: [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^dejong1991diffuse|de Jong 1991]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

Implements: [[L-SPEC-009 Cluster Nodes]], [[Extending Mycelium]] §5.

## What a cluster computes

Every factor in a cluster states itself through `cluster_form(factor, ps)` as a conditionally
linear-Gaussian term: a residual $r = \sum_c A_c x_c - C_{:,k}$ with noise $\mathcal N(0, Q)$ and weight $w_k$,
the column $k$ read from one discrete channel or fixed, so that
$-\log f = \tfrac12 r^\top Q^{-1} r + \tfrac12\log\det 2\pi Q - \log w_k$. A discrete-only factor (a categorical prior)
has no rows. Mycelium implements the form for `PriorFactor` (categorical or proper Gaussian
belief), Lenticulum for `LinearConstraintFactor`, `SwitchingConstraintFactor` and `GaussianPrior`.

For a joint assignment $a$ of the cluster's discrete variables, the product of the factors and the
incoming messages is $\exp(h_a^\top x - \tfrac12 x^\top J x + c_a)$ over the free continuous variables, with
$J = \sum_f G_f^\top Q_f^{-1} G_f + \sum_v \Lambda_v$ the same for every assignment (the forms share $A$ and $Q$ across
outcomes), $h_a = \sum_f G_f^\top Q_f^{-1} c_{f,a} + \sum_v \eta_v$, and $c_a$ collecting the quadratic constants, the
normalisers, the log-weights and the incoming discrete log-probabilities. Points (data, pins) are
substituted into the constants; a Gaussian-mixture input is collapsed to its moments.

- **Message to a continuous channel $t$.** Eliminating the other continuous variables (a Cholesky
  of $J_{rr}$, shared) leaves per assignment $\exp(\tilde\eta_a^\top x_t - \tfrac12 x_t^\top \tilde\Lambda x_t)$ with
  $\tilde\Lambda = J_{tt} - J_{tr} J_{rr}^{-1} J_{rt}$ and the mass $c_a + \tfrac12 h_r^\top J_{rr}^{-1} h_r - \tfrac12\log\det J_{rr} + \tfrac{n_r}2 \log 2\pi$. The message
  is the mixture over the assignments, normalised components when $\tilde\Lambda$ is positive definite,
  unnormalised exponentials otherwise, as the switching factor's.
- **Message to a discrete channel.** $P(k) \propto \sum_{a: a_d = k} \exp(c_a + \log\!\int \exp(h_a^\top x - \tfrac12 x^\top J x)\,dx)$.
- **Local free energy.** The belief $b(x, a)$ is, given $a$, the Gaussian $\mathcal N(J^{-1}h_a, J^{-1})$, and $q_a \propto e^{\log Z_a}$.
  So $H(b) = H(q) + H(x \mid a)$ is closed form, and $\mathbb E_b[-\log \prod_f f]$ sums each factor's expected
  quadratic and its normaliser. Graded as `(fit, complexity, negentropy)`, as the switching
  factor's local free energy, so a cluster of switches and its block factor agree term by term.
- **Gradient.** `factor_belief` freezes $q$, the means per assignment and the covariance;
  `belief_free_energy` re-evaluates every factor's form at `ps` (LP-020). At an exact belief this is
  the gradient of $-\log p$. The bookkeeping of which outcome and which channels each factor reads
  is `_cluster_layout`, non-differentiable.

`cluster(g, name, factors)` builds the new graph: the variables touched only by the clustered
factors become internal, the others are the node's channels (named after the variables, on
bidirectional edges); a variable is discrete when a clustered factor switches on it
(`cluster_switch`). The node's parameters are a `NamedTuple` keyed by the clustered factors'
parameter keys, so tied factors inside keep one entry (`cluster_parameters` rearranges a parameter
tree). `cluster_marginal` reads the marginal of any cluster variable, internal ones included.

## Tests

Lenticulum `test/cluster.jl`:

- the switching chain of `test/mixture_rules.jl` ($T = 5$) with every factor but the data in one
  cluster: the free energy equals $-\log p(y)$ by enumeration to $10^{-12}$ (relative), every
  $x_t$'s marginal mean and variance match the enumerated posterior to $10^{-12}$, and the gradient
  in the tied offsets matches central differences of $-\log p(y)$ (reference differences $10^{-6}$,
  ForwardDiff and Zygote $10^{-7}$);
- E7's additive wiring ($z = u_1 + u_2 + u_3 + \varepsilon$, one two-outcome switch per $u_l$): the cluster of the three
  switches sends the message of the hand-built 8-outcome block factor (log-densities to $10^{-12}$),
  the same loopy schedule gives the same marginals on both graphs to $10^{-10}$, and the two nodes'
  local free energies agree term by term to $10^{-10}$; a cluster of one switch is that switch;
- the same wiring with the sum and the switches in one cluster: the plain graph has no free
  energy (mixture marginals at the $u_l$), the clustered one has $-\log p(z)$ to $10^{-12}$;
- refusals: a factor without a form, a missing name, a taken name, a key tied outside, a plate.

Mycelium `test/cluster.jl`: priors only, the message is the product of the Gaussian priors, an
internal categorical keeps its prior, and the refusals.

## Grouping by overlap and several clusters at once (2026-11-03)

`cluster(g, [name₁ => factors₁, name₂ => factors₂, …]; params)` makes several cluster nodes in one
call. With `params` they read one parameter entry, tied: one cluster per data point of one model
(`examples/structure_edits.jl`). Without it a parameter key shared between groups is refused,
since each node would hold its own copy. `cluster_parameters(gc, ps)` converts for every node.

`overlap_groups(C, maxsize)` is E7's greedy grouping (`Composing Stochastic Factors`): start from
singletons, merge the two groups whose largest pairwise overlap is largest while the merged group
fits and the overlap is positive. Lenticulum's `switch_overlaps(g, ps, sum, switches)` gives the
overlaps of switching factors feeding one linear factor: each switch's offset directions mapped
through the sum's coefficients, $|\cos|$ in the metric of the sum's total noise
$R = Q_{\mathrm{sum}} + \sum_l A_l Q_l A_l^\top$. For binary switches this is E7's $|G_{lm}|/\sqrt{G_{ll}G_{mm}}$, $G = W^\top R^{-1} W$: `test/cluster.jl`
compares both to $10^{-12}$ and the groups with E7's own function, copied as the oracle, for group
sizes 2, 3 and 4; two planted overlapping pairs end up in the same groups.

Whether it helps, `bench/cluster_overlap.jl` ($d = 8$, eight binary switches in four overlapping pairs,
interleaved so that index order splits them, 100 points from the model, 50 loopy sweeps), the mean
over points of the mean $\lvert\Delta P(k_l = 2)\rvert$ against the exact posterior (every factor in one cluster):

| E-step | mean $\lvert\Delta P\rvert$ | max |
|---|---|---|
| one switching factor per switch | 0.0922 | 0.2376 |
| clusters of the overlap groups, the switches only | 0.0922 | 0.2376 |
| E7's blocks of 4 in index order (partial sum + switches per cluster) | 0.0917 | 0.2433 |
| E7's blocks of 4 grouped by overlap | 0.0309 | 0.1446 |

Grouping by overlap cuts the error to a third, as in E7, but only with E7's wiring: each block's
switches feed a partial sum $s_b = \sum_{l \in b} u_l$ inside the block's cluster, and the blocks' sums feed
the sum node. Clustering the switches alone changes nothing, to four digits: they meet only at the
sum node, which collapses each $u_l$'s mixture to its moments separately, so the cluster's joint
over its switches never reaches the coupling. Blocks that ignore overlap help as little.

## Discrete tables, joint outcomes and impossible assignments (2026-11-03)

A `ClusterForm`'s `switch` may be a tuple of discrete channels; the outcome is then their joint
state, in column-major order (the first channel fastest, `LinearIndices`). `TableFactor(dims,
logtable)` uses this: a fixed factor $f(k_1, \dots, k_n) = \exp(\texttt{logtable}[k_1, \dots, k_n])$ over
discrete variables, a discrete-only term ($m = 0$) inside a cluster. Outside one it sends exact
belief-propagation messages (the table summed against the other channels' categorical messages,
or sliced at an observed state); it has no local free energy there.

Its entries may be $-\infty$, so a table can tie discrete variables together. That is how
`examples/structure_clusters.jl` writes E3b's switch over sums: each switch reads an outcome
$q_l = (\text{bit}, j)$ with $2J$ states, and a table over $(j, q_l)$ allows only the $q_l$ of cluster $j$.
Enumerating every joint state would cost $J (2J)^L$ assignments where $J 2^L$ are possible, so
`_cluster_joint` now adds every factor's log-weights first and drops the assignments of weight
zero before the Gaussian part (the precisions and solves). The beliefs keep the surviving
assignments only. Since 2026-11-04 the enumeration itself is pruned (`_enumerate_assignments`): the
assignments are built one discrete variable at a time, a factor's log-weights are added once all
its switch channels are assigned, and a partial assignment of weight zero is dropped there. The 2^16
cap now counts the possible assignments, not the full product: 18 binary variables tied equal by
tables (2^18 joint states, 2 possible) form one cluster (`test/cluster.jl`, "pruned enumeration";
before, the cap refused it). It does not speed up the plated example: at $(4, 3)$, 2 048 joint states
of which 32 are possible, a fitting step took 0.155 s against 0.158 s before (a scratch timing), since
the batched path builds one joint per plate group.

A cluster's message merges equal components: assignments the target does not see (an internal
discrete variable that does not move it) gave repeated components with split weights; they are now
one component with the summed weight, and a message whose components are all equal is one
Gaussian (`test/cluster.jl`, Mycelium's side). In `belief_free_energy` a $-\infty$ log-weight met a weight of exactly 0 and
gave `0 · (-Inf) = NaN`; such entries are set to 0 before the product (`_finite_logw`), which is the
$0 \log 0 = 0$ convention, and no AD backend sees the infinity.

Tests (`test/cluster.jl`): three categorical variables with two tables (one $-\infty$ entry each) in
one cluster give $F = -\log Z$ and the marginals of enumeration to $10^{-12}$, and belief propagation
through the tables (the graph is a tree) the same marginals; the observed-state message and the
column order. In Lenticulum, the switch over sums with $J = 2$, $L = 2$ in one cluster gives
$F = -\log p(z)$ to $10^{-14}$ (a scratch check), and the example checks it at $(3, 2)$ on 200 points
($9 \cdot 10^{-13}$) along with the gradient against central differences (`examples/structure_clusters.md`).

## Cluster nodes over a plate (2026-11-04)

`cluster(g, plate, name, factors; params)` clusters the template factors `factors` of every replica
of `plate` into one tied node per replica, and the graph keeps the plate. Inside each node the
variables carry their template names (`:z`, not `z[3]`), so every replica has the same channels,
which a plate group needs. `as_plate(g, name, n)` (`plates.jl`) records the nodes named `x[i]` of an
unplated graph as a plate; structure edits refuse plates, so an edited graph gets its plate back
this way.

Tied clusters differ only in what their replica's messages bring. When every replica has the
same factors (compared field by field) and its inputs the same shape (which channels are points,
the input precisions, the discrete inputs), replica $i$'s joint is replica 1's with
$H_i = H_1 + (h_i - h_1)\mathbf 1^\top$ and $c_i = c_1 + (\delta_i - \delta_1)$: $h$ collects the Gaussian inputs'
$\eta$ and the points' shifts $-G_f^\top Q_f^{-1} s_f$, $\delta$ the terms $s_f^\top Q_f^{-1} C_{f,a} - \tfrac12 s_f^\top Q_f^{-1} s_f$. So
`factor_messages` builds one joint, its precision's factorisations and the surviving assignments
once (`cluster_batch.jl`), and `factor_beliefs` returns `FrozenClusterBeliefs`, whose
`belief_free_energies` evaluates all replicas as a few stacked matrix products: what the parameter
gradient differentiates. Replicas that differ fall back to the per-node methods.

Tests (`test/cluster.jl`, "Cluster nodes over a plate"): the switch over sums ($J = 2$, a table) on 12
points, plated against the same graph without the plate record: free energy equal to $10^{-10}$ and
to $-\sum_i \log p(z_i)$ by enumeration, the batched beliefs taken, the gradient equal to $10^{-8}$
with ForwardDiff and Zygote, batched messages equal to one by one, the fallback chosen when one
replica's input is a Gaussian instead of a point; `as_plate`'s refusals. On 50 points at $(3, 2)$ a
scratch timing gave sweeps plus free energy 3.7 times and the gradient 6 times faster than the
unplated clusters, at equal values ($9 \cdot 10^{-13}$, $10^{-11}$). The examples' timings are in
`examples/structure_edits.md` and `examples/structure_clusters.md`.

## Per-outcome coefficients and noise (2026-11-04)

A `ClusterForm` may give an entry of $A$ as an $m \times d \times K$ array and $Q$ as an $m \times m \times K$ array:
the factor's coefficients $A_{c,k}$ and noise $Q_k$ then depend on its outcome $k$, as in Lenticulum's
`RegimeFactor` (`src/regime.md`). Given the assignment $a$ of the cluster's discrete variables the
continuous ones are still jointly Gaussian, but the precision

$$
J_a = J_0 + \sum_{f \text{ varying}} G_{f,k_f(a)}^\top Q_{f,k_f(a)}^{-1} G_{f,k_f(a)}
$$

now depends on $a$ through the outcomes $k_f(a)$ of the varying factors only ($J_0$ holds the incoming
messages and the factors whose $A$ and $Q$ do not vary). The assignments are grouped into
**precision classes**, one per combination of those outcomes; each class is factorised once
(`_precision_classes`, the factorisation cache is keyed by class). Without a varying factor there
is one class and every computation is the one above.

- **Messages.** To a discrete channel nothing changes but the class-wise log-integrals. To a
  continuous channel each class gives its own precision $\Lambda_\kappa = J_{tt} - J_{tr} J_{rr}^{-1} J_{rt}$, so the
  message is a mixture whose components need not share a precision. Components are merged when
  both their precision and their natural mean are equal. A proper component is normalised and its
  mass moved into the weight; an improper one is the unnormalised exponential
  $\exp(\eta^\top x - x^\top \Lambda x / 2)$ its weight multiplies, which is how Mycelium's mixture products read it
  (`gaussian.jl`, `_improper_overlap`). So improper components with different precisions are
  weighted consistently. (The first version refused them, after `switching.md` §5; that concern is
  about a factor that projects flat inputs out, not about an exact message.)
- **Belief and free energy.** `FrozenClusterBelief` holds one covariance $\Sigma_\kappa$ and one entropy
  $H(x \mid \kappa)$ per class and the class of each assignment. In the free energy a factor whose $A$ and $Q$
  vary is evaluated per outcome group: its residual means with $A_{c,k}$, its trace term
  $\sum_\kappa q_{k,\kappa} \operatorname{tr}(Q_k^{-1} G_k \Sigma_\kappa G_k^\top)$ with $q_{k,\kappa}$ the belief's mass on the assignments of outcome $k$ in
  class $\kappa$, and its $\tfrac12 \log\det 2\pi Q_k$ weighted by the mass on outcome $k$. The negentropy is
  $-H(q) - \sum_a q_a H(x \mid \kappa(a))$.
- **Plates.** Replicas with a varying factor are not batched (`_cluster_batchable` is false): the
  data terms $G_k^\top Q_k^{-1} s$ would differ per class, and the batched path assumes one precision. They
  take the per-replica path, with the same results.
- **One factor outside a cluster.** `as_cluster(f)` is a factor as a one-factor cluster over its
  own channels. A factor stated only by a `cluster_form` gets its messages and Bethe free energy
  from it (Lenticulum's `RegimeFactor` does).

Tests: Lenticulum `test/regime.jl`. A mixture of factor analysers ($x \sim \mathcal N(0, I)$, $z = W_k x + \mu_k + \varepsilon_k$,
three regimes with different $W_k$ and $Q_k$) in one cluster gives $F = -\log p(z)$ to $10^{-12}$, the regime
posterior and the three posteriors of $x$ (each with its own covariance) to $10^{-12}$; the gradient in
$A$, $b$, $\log w$ and the noise's Cholesky factors agrees with central differences of $-\log p(z)$ to $10^{-6}$
(ForwardDiff and Zygote); over a plate of four points the per-replica path is taken and is exact.

## Flat directions (2026-11-04)

A direction of the continuous variables that no factor in the cluster and no incoming message
determines is **flat**: the precision ($J_\kappa$, or $J_{rr}$ when the target's message eliminates the other
variables) is singular there. Until 2026-11-04 the node refused it. It now integrates such
directions out in the **diffuse limit**: a vague prior $\mathcal N(0, s^2 I)$, $s \to \infty$, with the factor $(2\pi s^2)^{k/2}$
of the $k$ flat directions dropped. On the range of $J$ (an orthonormal basis $U$, eigenvalues $\lambda$),

$$
\log \int \exp\bigl(h^\top x - \tfrac12 x^\top J x\bigr)\, dx \;\to\; \tfrac12 h^\top J^{+} h - \tfrac12 \log \operatorname{pdet} J + \tfrac{n - k}{2} \log 2\pi ,
$$

which needs $h \perp \ker J$: every quadratic term vanishes on $\ker J$ ($J$ is a sum of positive
semidefinite terms), so only a linear term of an incoming message can reach a flat direction, and
that integral diverges; it is refused. The dropped factor is the same for all assignments only if
they leave equally many directions flat; otherwise the node refuses (in the vague limit the
assignments with more flat directions would take all the weight, which is rarely what a model
means). This is the diffuse treatment of state-space models (de Jong 1991, the diffuse likelihood) and what
the constraint factors do with a flat input (`constraint.md` §4.1).

- **Where it matters.** A boundary channel with no incoming message yet (the first sweep of a
  loopy schedule), a factor outside a cluster with a flat input (`as_cluster`, Lenticulum's
  `RegimeFactor`), and combinations of variables the model never determines (only $u - w$ enters,
  so $u + w$ is flat).
- **Belief and free energy.** The frozen belief holds $M = J^+ H$ and $\Sigma = J^+$ per class, the flat
  directions at mean 0 and dropped; $H(x \mid \kappa)$ is the entropy on the range. Every factor's $G$ annihilates
  $\ker J$, so the fit terms are the limits of the vague ones, and their gradient in $\theta$ at the frozen belief
  is the vague gradient's limit (a term $G N N^\top / \varepsilon$ with $GN = 0$ vanishes).
- **Marginals.** `cluster_marginal` of a variable in a flat class is the information form of the
  other variables eliminated; a variable with a flat direction gets an improper component (zero
  precision there) rather than a covariance with zeros.
- **Deciding flatness.** An eigenvalue at most $10^{-10}$ times the largest is flat. Cholesky can
  succeed on a singular $J$ (rounding leaves a pivot of $10^{-15}$, and the first test then gave variances
  of $6 \cdot 10^{14}$ and a free energy 18.7 off), so its smallest pivot is checked against the same threshold.
  A precision that weak but meant (a prior of standard deviation $10^5$ beside factors of precision 1)
  is taken as flat, which shifts the free energy by the dropped constant.

Tests: Lenticulum `test/cluster_flat.jl`. With only $u - w$ determined, the free energy equals the
vague graph's ($s = 10^4$) less $\tfrac12 \log 2\pi s^2$ to $10^{-6}$ and the closed form $-\log p(z) + \tfrac12 \log 2$ to $10^{-12}$; the
switch posterior and the gradient (ForwardDiff, Zygote) equal the vague ones, and $u$'s marginal is
improper of zero precision. A `RegimeFactor` with $x \in \mathbb R^3$ seen through $2 \times 3$ maps sends the message to
the regime of the vague input to $10^{-7}$ and the closed form $w_k / \sqrt{\det W_k W_k^\top}$ to $10^{-12}$; a linear term along
a flat direction and regimes with different numbers of flat directions are refused; improper
components of rank one with different precisions have the closed-form $\Lambda_k$, $\eta_k$ and weights.

## What it does not do

- **Factors outside the conditionally linear-Gaussian class.** A factor needs `cluster_form`
  (its $A$ and $Q$ may depend on its outcome since 2026-11-04, above). `DataFactor` stays outside:
  data reaches the cluster as point messages.
- **Flat directions with unequal counts, or reached by a linear term.** Refused (above). Flat
  directions in equal number are integrated out in the diffuse limit.
- **Enumeration is exponential.** At most $2^{16}$ joint discrete states of nonzero weight (pruned as
  they are built, above); equal components of a message are merged, but nothing reduces the
  number of distinct ones. Precision classes add one factorisation each.
- **Plates, frozen parameters, other rules, other couplings.** `cluster(g, groups)` refuses graphs
  with plates (`cluster(g, plate, name, factors)` clusters a plate's factors, on a graph with one
  plate), frozen parameter paths, nodes that are not on belief propagation, a parameter key tied
  to a node outside, and edges that are not bidirectional, diagonal and attached. The cluster node
  itself is a belief-propagation node; differentiating through inference (`adjoint = :implicit`)
  is not tested with it.
- **Grouping rewires nothing.** `overlap_groups` chooses the groups; putting a partial sum per
  group into the graph, which is what makes the groups pay (below), is an edit of the graph. For
  switches feeding a Lenticulum linear factor, Lenticulum's `partial_sums` makes it
  (`src/switching.md` §14); Mycelium itself knows no sum factor.

## Difficulties

1. **Tied parameters.** The first version keyed the node's parameters by factor name and refused
   tied factors, which refused the switching chain at once (its observation and dynamics factors
   are tied). The node now keys them by parameter key, and refuses only a key shared with a node
   outside the cluster.
2. **A comparison the test could not make.** A cluster of the three switches and the block factor
   were first compared by the graph free energy; neither graph has one, since the $u_l$ have
   mixture marginals and Mycelium no mixture entropy. The test compares the nodes' local free
   energies instead, and a separate test shows that clustering the $u_l$ in removes the gap.
3. **Review (Gemini via `agy`, 2026-11-03).** Three findings were fixed: a factor channel left
   unconnected inside a cluster gave a `KeyError` (now refused by `cluster`, tested); a discrete
   input with no possible state gave an empty-collection error instead of an `ArgumentError`; the
   batched mixture product did not check the replica count. Two were not bugs: `_indefinite`
   depends on the precision only, so a shared one is one test; and Zygote through
   `belief_free_energy` works (the chain's gradient is tested with it, and `_cluster_layout` is
   non-differentiable in the ChainRulesCore extension, which the reviewed diff did not show).
4. **Review of the grouping (Gemini via `agy`, 2026-11-03).** Fixed: a cluster's parameter key could
   equal the key of a node left outside, and `cluster_parameters` would then overwrite that node's
   entry (refused now, tested); tied nodes compared their parameter keys in order, so the same
   factors listed in another order were refused (compared as sets now); `switch_overlaps` took the
   directions of a switch with three or more outcomes from the first outcome only (every pair
   now). Documented: the metric counts the noise of the sum and of the listed switches only.
