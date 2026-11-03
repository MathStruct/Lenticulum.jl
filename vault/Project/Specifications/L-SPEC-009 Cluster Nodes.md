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

## Open

- Factors whose $A$ or $Q$ depend on the outcome, flat continuous directions inside a cluster,
  clusters in plates, automatic grouping by overlap (E7's $|\cos|$ criterion).
- Merging repeated components of a message (assignments the target does not see).
