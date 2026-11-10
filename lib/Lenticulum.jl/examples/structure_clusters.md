# Structure search with split and merge moves, as edits of one graph

Run `julia -t auto --project=docs lib/Lenticulum.jl/examples/structure_clusters.jl` (about 18 minutes, the greedy search 16 of them; `STRUCTURE_GREEDY=false` runs the reference alone; `STRUCTURE_FIT=em` fits by EM, `STRUCTURE_BEAM=k` runs a beam search; the docs environment
has ForwardDiff). This extends [`structure_edits.md`](structure_edits.md) from one sum node ($J = 1$) to
E3b's switch over sums, and belongs to [`Extending Mycelium`](../vault/Design/Extending%20Mycelium.md)
§6 and [L-SPEC-010](../vault/Project/Specifications/L-SPEC-010%20Structure%20Edits.md).

## Question

`structure_edits.jl` ported the add and remove moves of `experiment-qm`'s
`examples/structure_switches.jl` to edits of one Mycelium graph. The branch has two more, split and
merge a cluster, for the switch over sums with $J$ clusters. Can those run on the graph too, scored
exactly?

## Setup

- **Model.** $z = c_j + \sum_l k_l W_{jl} + \varepsilon$ in 2-d, $j \sim \pi$, $k_l \mid j \sim \mathrm{Bernoulli}(p_{jl})$, noise variance 0.1
  (fixed), switch noise $10^{-3}$. It has $p = J(3L + 3) - 1$ free parameters.
- **As a graph**, per point: the label $j$ ($J$ states), read by a switch that sets $u_0 = c_j$ (log-weights
  $\log \pi$); per switch $l$ an outcome $q_l = (\text{bit}, j)$ with $2J$ states, read by a switch setting
  $u_l \in \{0, W_{jl}\}$ with log-weights $\log p(\text{bit} \mid j) - \log J$; a `TableFactor` over $(j, q_l)$
  that is $\log J$ where $q_l$ belongs to cluster $j$ and $-\infty$ elsewhere; the sum $z - u_0 - \sum_l u_l$; the
  data. The table ties each $q_l$ to $j$, and with its $\log J$ the joint is exactly the model's.
- **Exact scores.** Each point is one cluster node, tied (`params = :mix`), so the graph's free energy
  is $-\sum_i \log p(z_i)$. A cluster enumerates $J(2J)^L$ joint states and keeps the $J 2^L$ the tables allow
  (`lib/Mycelium.jl/src/cluster.md`).
- **Moves**, each an edit of the unclustered graph (`GraphBuilder(g)`), with the branch's warm starts:

  | move | edit | warm start |
  |---|---|---|
  | add a switch | every sum node gains a channel (`replace_factor!`); a switch, a table and their variables are added | every cluster gains an offset $2\sigma\,\mathcal N(0, I)$, $p = 1/2$ (4 draws) |
  | remove a switch | its factors and variables are removed; the sum nodes lose the channel | the least used switch (mean $\min(p, 1 - p)$) is dropped |
  | split a cluster | $j$ and every $q_l$ are re-added with more states, their switches and tables replaced | the heaviest cluster is duplicated, centres perturbed by $\sigma\,\mathcal N(0, I)$ (4 draws) |
  | merge two clusters | the same with fewer states | the two closest centres are averaged by weight |

  Switches are named by a running id, so removing one renames nothing. `transfer` carries the
  messages of the unchanged edges (a cluster node's edge to its $z$).
- **Fit.** 300 Momentum steps ($\eta = 0.02$, $\gamma = 0.9$) of the free-energy gradient at the exact beliefs
  (ForwardDiff), in $\log \pi$, $c$, $W$ and the logits of $p$ (the chain rule through the log-weights by hand).
- **Score.** $\mathrm{BIC} = 2F + p \log N$. **Search:** greedy from $(1, 0)$, the best improving move until none
  improves, $J \le 4$, $L \le 3$.
- **Data.** 200 points of the branch's tree (`hierarchy_e3b.jl`'s `tree_data`): 3 clusters of rotated
  2 × 2 grids, so the generating structure is $(3, 2)$; covariance $[0.12\ 0.04;\ 0.04\ 0.09]$, `Xoshiro(5)`.

Checked before the run (a scratch script, not in the repository): the graph reached by edits
(two adds, two splits) has the same nodes and edges as one built directly at $(3, 2)$; its free
energy equals $-\sum_i \log p(z_i)$ from the mixture formula to $9 \cdot 10^{-13}$; the gradient in $\theta$ agrees
with central differences of the re-converged free energy to $1.8 \cdot 10^{-6}$ (gradients of size $10^3$).

## Measured (2026-11-03)

The branch's four moves (`STRUCTURE_COMPOUND=false`; times with the clusters as a plate, 2026-11-04):

| step | move | $(J, L)$ | BIC | time |
|---|---|---|---|---|
| start | | (1, 0) | 46 415.38 | |
| 1 | add a switch | (1, 1) | 17 218.10 | 46 s |
| 2 | split a cluster | (2, 1) | 2 637.50 | 105 s |
| 3 | add a switch | (2, 2) | 1 659.24 | 163 s |
| 4 | add a switch | (2, 3) | 1 355.84 | 300 s |
| stop | no split, merge or removal improves; $L \le 3$ | (2, 3) | | 441 s, 44 fits |

The final fit has weights 0.594 and 0.406 and centres $(1.86, 4.99)$ and $(4.23, -2.76)$: one cluster
holds two of the three generating clusters, its switches spanning both.

Reference, the generating structure $(3, 2)$ (graph built by the same edits), after that greedy
search in the same run:

| start | BIC |
|---|---|
| the generating parameters | **1 299.15** |
| random starts 1–4 | 2 490.53, 2 119.74, **1 457.61**, 1 594.49 |

**With compound moves and data-driven splits** (the default since 2026-11-04). Two compound moves,
a split followed by a removal and a merge followed by an addition (no single move leads from $(2, 3)$
to $(3, 2)$), and a data-driven warm start for splits: the heaviest cluster's points (responsibility
above 1/2, read from each point's cluster node with `cluster_marginal`) are split by 2-means, each
half's centre its mean minus the switches' expected offsets. A plain split gets it as a fifth draw;
the compound split uses it and tries removing each switch in turn.

| step | move | $(J, L)$ | BIC | time |
|---|---|---|---|---|
| start | | (1, 0) | 46 415.38 | |
| 1 | add a switch | (1, 1) | 17 218.10 | 49 s |
| 2 | split a cluster | (2, 1) | 2 637.50 | 119 s |
| 3 | add a switch | (2, 2) | 1 659.24 | 215 s |
| 4 | add a switch | (2, 3) | 1 355.84 | 425 s |
| 5 | **split, remove a switch** | **(3, 2)** | **1 299.17** | 657 s |
| stop | | (3, 2) | | 970 s, 81 fits |

At the stop the best candidates were (3, 3) at 1 341.68 (add), (4, 2) at 1 343.67 (split), (2, 2) at
1 688.21 (merge), (3, 1) at 1 864.61 (remove), (4, 1) at 1 637.54 (split and remove) and (2, 3) at
1 655.55 (merge and add). The final fit has weights 0.38, 0.21, 0.41 and centres $(-2.83, -1.55)$,
$(4.17, -2.64)$, $(1.83, 5.01)$; the generating centres (the grids' corners) are
$(0, 5)$, $(-4.33, -2.5)$ and $(4.33, -2.5)$ with weights 0.4, 0.35, 0.25 (the fitted centres are the
"all switches off" corners of rotated grids, so they need not coincide). The greedy score is
0.02 above the fit from the generating parameters (1 299.15). The reference's random starts in this
run reach 1 420.65 at best.

An intermediate version (same day) had the compound split use a random split and drop only the
least used switch; it reached $(3, 2)$ only at BIC 1 661.92 and the search stopped at $(2, 3)$. The
move set was enough; the warm start was not.

**Fitted by EM** (`STRUCTURE_FIT=em`, 2026-11-04). Two M-steps were tried, both SQUAREM-accelerated
(`MyceliumLearning.fit_em`) until $F$ falls by less than $10^{-4}$:

- **Per factor** (each switch's offsets from the moments of its $u$, `MyceliumLearning.moment_statistics`,
  within this parameterisation): 1143 s, and the search stopped at $(2, 3)$ with BIC 1 359.22; the
  compound split reached $(3, 2)$ only at 1 677.42. A switch of noise $10^{-3}$ pins its $u$ to the current
  offset, so each EM step moves the offsets by little and $\Delta F < 10^{-4}$ comes long before the optimum
  (`lib/MyceliumLearning.jl/src/em.md` §2).
- **Collapsed** (the default of `em_θ` now): given a point's assignment the $u$'s integrate out,
  $z \sim \mathcal N(c_j + \sum_l \mathrm{bit}_l W_{jl},\ (0.1 + (L + 1) \cdot 10^{-3}) I)$, so the M-step for $(c_j, W_{j\cdot})$ is one weighted least-squares
  fit per cluster over the assignments' posterior weights, read from each point's cluster belief.
  EM on the same likelihood with fewer latent variables.

  | step | move | $(J, L)$ | BIC | time |
  |---|---|---|---|---|
  | start | | (1, 0) | 46 415.38 | |
  | 1 | split a cluster | (2, 0) | 17 383.50 | 6 s |
  | 2 | add a switch | (2, 1) | 2 637.50 | 11 s |
  | 3 | add a switch | (2, 2) | 1 780.52 | 25 s |
  | 4 | split a cluster | (3, 2) | 1 343.89 | 69 s |
  | stop | | (3, 2) | | 181 s, 74 fits |

  Reference in the same run: 1 299.14 from the generating parameters, 1 412.51, 1 732.46, 1 448.49 and
  1 861.76 from the random starts. The search is six times faster than with gradient steps (1 121 s
  that day) and reaches the generating structure, but by another path (a split before any switch,
  since fully converged fits make the split score better at $(1, 0)$) and into a worse local optimum
  of it (1 343.89 against 1 299.17): two of its centres sit near the same grid. The fits themselves are
  as good or better (1 299.14 from the generating parameters against 1 299.15; the best random start
  1 412.51 against 1 420.65).
- **Collapsed, with restarts** (the default of `fit_by_em` now: `STRUCTURE_RESTARTS = 4`, the warm
  start and three perturbed copies, the best kept). At $(1, 0)$ without restarts three of four
  add-a-switch draws ended at BIC 31 942.95 by EM and one by gradient steps; EM settles on stationary
  points that gradient steps leave. With restarts the search follows the gradient search's path to
  $(2, 3)$ (1 355.83 against 1 355.84), then takes a plain split to $(3, 3)$ at **1 343.74** where the gradient
  search's compound move reached $(3, 2)$ at 1 299.17: 74 fits in 670 s. Reference: 1 299.14 from the
  generating parameters; 1 826.59, 1 379.88, 1 661.42, 1 449.49 from the random starts.

  Why (an instrumented scratch copy, at that step): from EM's $(2, 3)$ fit, the compound candidates
  reach 2 720.80, 1 593.59 and 1 683.27 by EM, and 2 472.62, 1 509.55 and 1 769.61 by gradient steps from
  the same starts; gradient steps from EM's results do not move them, and EM to $\Delta F < 10^{-9}$ changes
  little. So the fit method does not decide it: the two $(2, 3)$ fits score alike but are different
  optima, and only the gradient search's leads on to $(3, 2)$. Splitting every cluster instead of the
  heaviest (tried, not kept) did not help: $(3, 3)$ at 1 344.20 in 784 s.

**Beam search** (`STRUCTURE_BEAM=k`, with EM fits, 2026-11-04). Each kept structure is expanded by
every move; the best fit per structure $(J, L)$ not expanded before is kept, and the $k$ best of those
form the next beam; the search stops when no candidate improves on the best score seen.

| $k$ | beams | result | fits | time |
|---|---|---|---|---|
| 1 (greedy) | | $(3, 3)$, 1 343.74 | 74 | 670 s |
| 2 | (1, 1) (2, 0) → (2, 1) (1, 2) → (2, 2) (3, 1) → (3, 2) (2, 3) | **$(3, 2)$, 1 328.91** | 123 | 654 s |
| 3 | … → (3, 2) (2, 3) (4, 1) | $(3, 2)$, 1 335.11 | 156 | 946 s |

Both beams reach the generating structure, by the path $(1, 0) 	o (2, 0) 	o (2, 1) 	o (3, 1) 	o (3, 2)$, which greedy
EM misses. They fit it less well than the gradient search's path did (1 299.17; 1 299.14 from the
generating parameters): the $(3, 2)$ fits reached from these warm starts are local optima at
1 329–1 335, and the beams' random draws differ (they are taken in another order), which is why $k = 3$
is not better than $k = 2$. The reference random starts in these runs reach 1 389.95 and 1 408.89 at best.
So with EM the beam fixes the structure, not the fit. **A final refit** of the chosen structure from 32
starts (`STRUCTURE_REFIT`, default 32 with EM: 16 perturbations of its fit and 16 random starts with
centres at random data points) closes the gap: with $k = 2$ the best refit, from a random start,
reaches **BIC 1 299.14**, the fit from the generating parameters, in 29 s (median 1 328.91). In all
$(3, 2)$ at 1 299.14 in 692 s, against $(3, 2)$ at 1 299.17 in 1 121 s for the greedy search with gradient
steps the same day.

## What it shows

1. **All four moves run on one graph.** Splits and merges change the number of states of the
   label and of every switch outcome; they are edits (re-added variables, replaced factors) like
   adding a switch, and each candidate is scored by the exact free energy of the edited graph.
   The search took the moves one expects at first: a switch, a split, then switches.
2. **With the four moves it stops at a worse structure.** Greedy ends at $(2, 3)$, BIC 1 355.84, while
   $(3, 2)$ reaches 1 299.15 from the generating parameters. From $(2, 3)$ no single move reaches
   $(3, 2)$, the branch's finding on the 16-mode grid. A compound move (split and remove) with a
   data-driven split finds it, at 1 299.17; with a random split it did not. The warm start
   decides, as in E3: random starts of $(3, 2)$ itself reach 1 420.65 at best.

3. **It is slow**: 441 s for 44 fits (970 s for 81 with the compound moves), against the branch's 3.6–13 s with closed-form EM and random
   restarts. Fitting dominates (300 gradient steps through 200 cluster nodes per candidate). The
   clusters are a plate (`as_plate`, `cluster(g, plate, ...)`), so their messages and free energies
   are batched; the first run without the plate took 2 314 s for the same path and scores.

## Caveats

- One data set, one seed, 200 points (the branch used 2 000, and found $(3, 2)$ on this tree), and no
  random restarts per candidate (the branch added 32). Either may explain the different path. The noise covariance is fixed at $0.1 I$, not learned.
- $L$ is shared by the clusters, as on the branch.
- The impossible assignments are still enumerated before they are dropped; $J(2J)^L$ grows fast.
- The sum node's coefficients are frozen (`freeze = (:A,)`, 2026-11-04). Rerun with the freeze: the
  same path, scores, stop candidates and fitted centres to the printed digits, 1121 s for the
  greedy search (970 s above, from an earlier run). The reference random starts in that run gave
  1420.82, 1420.65, 1584.92 and 2291.43.

