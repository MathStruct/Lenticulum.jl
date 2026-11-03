#implementation

> Cluster nodes: `cluster(g, name, factors)` replaces a set of factor nodes by one node that runs
> exact inference inside (enumeration of the discrete states, Gaussian elimination of the
> continuous ones) and looks like one factor outside, with exact messages and a closed-form local
> free energy. A whole switching chain in one cluster has the free energy $-\log p(y)$ to
> $10^{-12}$; three switches in one cluster send the messages of E7's hand-built block factor.

> Sources: Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. Inf. Theory 51(7), 2005 (regions; here a region is solved exactly). code: `cluster.jl`; Lenticulum `src/switching.jl` (`cluster_form` for the constraint factors and `GaussianPrior`); tests: `../test/cluster.jl`, Lenticulum `test/cluster.jl`
>
> Bibliography: [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]]
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
assignments only. In `belief_free_energy` a $-\infty$ log-weight met a weight of exactly 0 and
gave `0 · (-Inf) = NaN`; such entries are set to 0 before the product (`_finite_logw`), which is the
$0 \log 0 = 0$ convention, and no AD backend sees the infinity.

Tests (`test/cluster.jl`): three categorical variables with two tables (one $-\infty$ entry each) in
one cluster give $F = -\log Z$ and the marginals of enumeration to $10^{-12}$, and belief propagation
through the tables (the graph is a tree) the same marginals; the observed-state message and the
column order. In Lenticulum, the switch over sums with $J = 2$, $L = 2$ in one cluster gives
$F = -\log p(z)$ to $10^{-14}$ (a scratch check), and the example checks it at $(3, 2)$ on 200 points
($9 \cdot 10^{-13}$) along with the gradient against central differences (`examples/structure_clusters.md`).

## What it does not do

- **Factors outside the conditionally linear-Gaussian class.** A factor needs `cluster_form`, and
  its $A$ and $Q$ must not depend on the outcome (as the switching factor's do not). `DataFactor`
  stays outside: data reaches the cluster as point messages.
- **Undetermined continuous variables.** Elimination needs $J_{rr}$ (or $J$) positive definite: every
  continuous variable must be pinned down by the cluster's factors and the incoming messages. A
  flat direction is refused rather than integrated under a flat prior, as the constraint factors
  do it.
- **Enumeration is exponential.** At most $2^{16}$ joint discrete states; components are not merged,
  so a message to a variable that some switches do not touch repeats components.
- **Plates, frozen parameters, other rules, other couplings.** `cluster` refuses graphs with
  plates, frozen parameter paths, nodes that are not on belief propagation, a parameter key tied
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
