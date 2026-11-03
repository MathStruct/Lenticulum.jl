# Structure search as edits of one graph

Run `julia -t auto --project=docs examples/structure_edits.jl` (about 15 minutes; the docs environment
has ForwardDiff). This belongs to [`Extending Mycelium`](../vault/Design/Extending%20Mycelium.md) §6
and [L-SPEC-010](../vault/Project/Specifications/L-SPEC-010%20Structure%20Edits.md).

## Question

On `experiment-qm`, `examples/structure_switches.jl` searched switch wirings greedily, but every
candidate was a new model refitted outside Mycelium. Can the same search run on one Mycelium
graph, each move an edit of it and each candidate scored by the graph's own free energy?

## Setup

- **Model.** E3's additive wiring with $L$ binary switches in 2-d,
  $z = c + \sum_l u_l + \varepsilon$ with $u_l \in \{b_{l1}, b_{l2}\}$, noise variance 0.1, switch noise $10^{-3}$.
- **Data.** 200 points from three switches (E3's 8-mode grid), $W = 1.5\,[2\ 0\ 1;\ 0\ 2\ -1]$, `Xoshiro(3)`.
- **Graph.** One graph holds all points. Point $i$ has its data, its sum node and its switches.
  Each point is one cluster node (`cluster(...; params = :mix)`), and the clusters are tied, so the
  graph's free energy is exactly $-\sum_i \log p(z_i)$.
- **Moves.** Both are edits of the unclustered graph (`GraphBuilder(g)`):
  - *add a switch*: every sum node is replaced by one with another channel (`replace_factor!`),
    and a switch and its variables are added;
  - *remove the least used switch* (the most unequal weights): its factor and variables are
    removed (`remove_factor!`, `remove_variable!`).
- **Fit.** 300 Momentum steps ($\eta = 0.02$, $\gamma = 0.9$) of the free-energy gradient at the exact beliefs
  (generalised EM, ForwardDiff), on the shared offset $c$, the switches' outcomes $b$ and their
  log-weights. The sum node's coefficients are not updated.
- **Warm starts.** The current fit plus a new switch of small random offset (4 draws); for a
  removal, the remaining switches. The message store is carried over with `transfer`.
- **Score.** $\mathrm{BIC} = 2F + p\log N$ with $p = d + L(d + 1)$: a shared offset, one direction and one weight per
  switch. The two outcomes of a switch carry one redundant offset, so this counts the identified
  parameters.
- **Search.** Greedy from $L = 0$, taking the best improving move. The reference fits every
  $L \le 4$ from 4 random starts.

## Measured (2026-11-03)

| step | move | $L$ | BIC |
|---|---|---|---|
| start | | 0 | 10 923.84 |
| 1 | add | 1 | 6 493.69 |
| 2 | add | 2 | 2 863.19 |
| 3 | add | 3 | 1 067.60 |
| stop | (add: 1 087.9; remove worse) | 3 | |

Reference, best BIC per $L$ from random starts: 10 923.84, 6 493.69, 2 863.19, 1 067.60, 1 087.91 for
$L = 0, \dots, 4$. The greedy search took 734 s.

## What it shows

1. **The search runs on one graph.** Greedy search finds the generating structure, $L = 3$, by
   adding a switch three times. That is the path `structure_switches.jl` took for the 8-mode grid
   with closed-form EM on the branch, and the reference agrees. Each move was an edit, and each
   score the free energy of the edited graph, exact because every point is one cluster node.
2. **It is slow.** 734 s against 3.6–13 s on the branch. Each candidate takes 300 gradient steps
   through 200 cluster nodes; the branch's M-step was closed form. The graph spends its time in
   fitting, not in the edits, which take milliseconds.
3. **The score is exact only because inference is.** With approximate inference the free energy
   that scores the moves is biased (E6). The cluster nodes make it exact while $2^L$ stays small.

## Caveats

- One family (additive wiring, $J = 1$), one data set. The branch's split and merge moves (the switch
  over sums, $J > 1$) are not ported.
- The sum node's coefficients are held fixed by hand in the fitting loop. A cluster node cannot
  carry frozen parameter paths (`cluster.md`).
- How much `transfer` saves is not measured here: each component's tree is solved again in two
  sweeps. `test/structure_edits.jl` checks the warm start on a loopy graph.
