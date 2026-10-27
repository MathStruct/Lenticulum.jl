#design #open-problem

> **Graphs that change while they run**, as SLAM needs: poses and measurements arrive, old
> poses are marginalised, false loop closures are switched off. Lenticulum's graphs are built
> once and frozen, so none of this exists yet. This note collects what it would take: the data
> structure, what adding and deleting mean for the beliefs, the Bethe free energy as the
> criterion for structural changes, and a way to add and remove edges of a learned factor
> without touching its parameters.

> Sources: original to this vault (design and analysis); Kaess et al., *iSAM2: Incremental smoothing and mapping using the Bayes tree*, IJRR 2012; *IncrementalInference.jl*; Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. Inf. Theory 2005; Sünderhauf & Protzel, *Switchable Constraints for Robust Pose Graph SLAM*, IROS 2012; Agarwal et al., *Robust Map Optimization Using Dynamic Covariance Scaling*, ICRA 2013; Olson & Agarwal, *Inference on Networks of Mixtures for Robust Robot Mapping*, RSS 2012; Yang et al., *Graduated Non-Convexity for Robust Spatial Perception*, RA-L 2020; Chen, Goodfellow & Shlens, *Net2Net: Accelerating Learning via Knowledge Transfer*, ICLR 2016; Zaheer et al., *Deep Sets*, NeurIPS 2017; code: `graph.jl`, `messages.jl`, `free_energy.jl`
>
> Bibliography: [[Bibliography#^kaess2012isam2|Kaess et al. 2012]] · [[Bibliography#^incrementalinference|IncrementalInference.jl: Non-parametric ]] · [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^sunderhauf2012switchable|Sünderhauf & Protzel 2012]] · [[Bibliography#^agarwal2013dcs|Agarwal et al. 2013]] · [[Bibliography#^olson2012maxmixture|Olson & Agarwal 2012]] · [[Bibliography#^yang2020gnc|Yang et al. 2020]] · [[Bibliography#^chen2016net2net|Chen et al. 2016]] · [[Bibliography#^zaheer2017deepsets|Zaheer et al. 2017]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## 1. What exists: graphs are frozen

A `FactorGraph` is made by a `GraphBuilder` (`variable!`, `factor!`, `connect!`) and frozen by
`build` into an immutable structure with adjacency indices ([[graph]]). There is no operation that
adds or removes a variable, factor or edge of a built graph. The `MessageStore` holds one
message per edge in each direction, indexed by the edge's position in the edge list
([[messages]]).

The cheapest step towards dynamic graphs follows from that indexing, and is not implemented:
**append-only growth**. A builder that only appends variables, factors and edges keeps every
existing edge index. After `build`, the old store's messages can be copied into a new, longer
store, the new edges starting empty. Belief propagation then continues from where it was, with
no recompilation of anything. Deletion breaks the indexing, so it needs masks (§2) or a
re-indexing step that maps old edges to new ones.

## 2. The data structure the design already chose

[[Parallelism and Compilation]] §5 resolves the tension between compiled kernels and changing
graphs: compile one batched kernel per factor *type*, and let the graph enter as index arrays.
Adding a pose and an odometry factor then appends to an index array, and deleting sets a mask.
Capacity grows by doubling, so shapes change $O(\log n)$ times in a run. §7 of that note adds the
junction (Bayes) tree, whose clique recycling re-solves only the cliques a new measurement
touches. That is iSAM2's method, and IncrementalInference.jl does it with non-Gaussian beliefs.
Anything built here should be measured against those two. A new factor *type* arriving mid-run
still needs compiling; on another thread that stalls the running graph only through the garbage
collector, about 20 ms per burst, and not at all with the GC held off
([[Parallelism and Compilation]] §5).

## 3. What adding and deleting mean for the beliefs

The four operations are not symmetric:

| operation | belief propagation | cost |
|---|---|---|
| add a factor | its messages start uniform, the others keep theirs; iterate from there | local at first, then spreads through loops |
| add a variable | a new node with no messages, attached by new factors | as adding its factors |
| remove a factor | drop its messages; the neighbours' beliefs lose its information | local at first |
| remove a variable | **marginalise** it: replace it and its factors by one factor on its neighbours | creates fill-in |

Belief propagation is anytime: a fixed point of the old graph is a good start for the new one,
so additions warm-start naturally. Removing a variable by just deleting it throws its information
away. Marginalising keeps it: in the Gaussian case the Schur complement turns the variable and
its factors into one prior factor on its Markov blanket. That is fixed-lag smoothing, a SLAM
sliding window. The price is fill-in: the new factor joins all the neighbours, and repeated
marginalisation makes the graph dense. Choosing what to marginalise, and when to sparsify the
result, is part of the structural question of §4.

For a learned factor there is no Schur complement. Marginalising a variable of a diffusion
relation means a relation on the remaining channels, and with noise blocks that is the same
relation with the variable's block held at level 1 (§5).

## 4. The Bethe free energy as the criterion

The Bethe free energy is local and additive ([[Bethe Free Energy]]),

$$
F_{\mathrm{Bethe}} = \sum_a F_a + \sum_i (d_i - 1)\,H_i ,
$$

and `factor_free_energies` returns it graded, one term per factor. So the effect of a structural
change can be read off locally: adding or removing factor $a$ changes $F_a$ and the counting
corrections of its variables, before the messages re-equilibrate. Since $F$ approximates
$-\log Z$, the difference between the graph with and without a candidate factor is a Bethe
estimate of the change in log evidence. That gives criteria for the decisions SLAM makes all the
time:

- **Accepting a loop closure or a data association**: add the candidate factor if it lowers the
  free energy by more than a threshold, i.e. if the evidence prefers the graph with it.
- **Rejecting outliers**: a factor whose energy stays high after the messages converge is in
  disagreement with the rest. The graded energy names it, and the multivariate energy
  ([[Scalar and Multivariate Energy]]) names its channel.
- **Pruning**: a factor whose removal barely changes $F$ contributes little to the beliefs.

Robust SLAM already does versions of this from residuals: switchable constraints, dynamic
covariance scaling, max-mixtures and graduated non-convexity all switch factors on and off, or
down-weight them, by how badly they fit. Read through the free energy, each is a rule for
changing the graph's structure from its energy.

**Caveats.** On a loopy graph the Bethe free energy approximates $-\log Z$ and is not a bound in
general. Its locality holds only to first order: after a change the messages move, and on a
loopy graph the effect travels. Comparing graphs with different variables needs care: a removed
variable must be marginalised, not dropped, for the two numbers to mean the same thing, and an
added variable brings its prior with it. The Gaussian structure-search example now measures
a with/without comparison ([[Finding a Factor Graph]], 2026-10-27); dynamic graph mutation
and a SLAM benchmark are still absent. That example also corrects the earlier minus sign in
the displayed formula: $F_a=U_a-H_a$, so the variable entropy correction is positive. For
**fitted** parameters, state free energy supplies no automatic parameter-count penalty;
a normalised data likelihood needs its data-energy term as well. BIC helps in the example
but does not remove the measured loopy normaliser bias. This differs from integrating
parameters to obtain model evidence ([[Factor Graphs and Schemes]] §1).

## 5. Edges of learned factors

Whole factors are easy for parameters: each factor owns its Lux parameters, and the graph's
parameters are a `NamedTuple` keyed by factor name. Adding or removing a factor adds or removes
an entry, and every other entry stays identical.

An edge that changes a factor's arity is hard: the factor's network changes its input width.
Lux has no operation that grows a model while keeping its parameters on the old entries. Four
ways around it:

1. **Fixed arity.** A new connection is a new factor, not a new edge of an old one. Pairwise
   SLAM factors already work this way.
2. **Function-preserving widening.** Append zero columns to the first `Dense` weight for the new
   inputs (Net2Net's idea). The network computes exactly the same function, and the old
   parameters are untouched. But for an implicit learner a new channel changes the joint space:
   the relation must still learn how the new variable relates to the others.
3. **Capacity and masks with set-invariant layers.** A factor whose network reads a set of
   channels (Deep Sets, attention over the neighbours) takes any number of them with the same
   parameters; absent channels are masked. This is the usual machine-learning answer.
4. **Noise blocks as the mask, for diffusion factors.** With noise blocks (UniDiffuser), a block
   held at level $t = 1$ is pure noise and is marginalised away; a block held at $t_{\min}$ is
   clamped. A factor trained on block times for all its channels, as the MNIST model
   ([[MNIST in Both Directions]]) was, can therefore treat an edge as **absent** (its block at
   1) or **present** (clamped, or annealed with the query) without any change to its
   parameters. The trained network is a factor with preallocated capacity, and the mask is a
   noise level. It needs the factor to be trained with every subset of channels in play, which
   `block_times` provides.

The fourth is specific to this project and already within reach: `ImplicitDiffusion`'s
`levels` hold a block at any level per query. What is missing is the graph-level operation that
sets them when an edge is added or removed.

## 6. Open

- **Append-only growth with message reuse** (§1): the first operation to build, and the test of
  whether warm-started belief propagation converges faster than a fresh start.
- **Deletion**: masks in the index-array design, and marginalisation, Gaussian (Schur
  complement with fill-in control) and learned (noise blocks at level 1).
- **The Bethe criterion** (§4): the Gaussian with/without example is implemented and measured
  in [[Finding a Factor Graph]]; a loop-closure test against robust-SLAM methods on a standard
  pose graph remains open.
- **Edges as noise levels** (§5): a graph operation that adds or removes an edge of a diffusion
  factor by setting its block's level, and a test that a factor trained with all its channels
  answers correctly with some of them marginalised.
- **The junction tree** ([[Parallelism and Compilation]] §7), which makes incremental updates
  exact and local on loopy graphs, and is what iSAM2 and IncrementalInference.jl are measured by.

Related: [[SLAM and Sensor Fusion]], [[Parallelism and Compilation]], [[Bethe Free Energy]],
[[Loopy Message Passing]], [[Related Julia Projects]], [[Scalar and Multivariate Energy]],
[[Open Problems in Implicit Diffusion Learning]]
