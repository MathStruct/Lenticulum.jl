# Structure learning for switch wirings

Run `julia -t auto --project=. examples/structure_switches.jl` (about 14 minutes on 8 threads, almost
all of it in the exhaustive reference); the test is `test/structure_switches.jl`. This belongs to the
design note [`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§10)
and to [`Dynamic Graphs`](../vault/Factor%20Graphs/Dynamic%20Graphs.md) (structural changes).

## Question

Mycelium has no rule that changes a graph's structure. Its graphs are frozen once built, and the
only structure search in the repository adds Gaussian edges (`examples/gaussian_structure_search.jl`).
For switch wirings there are natural structural moves. Can a greedy search with them find the
structure of the data, scored exactly?

## Setup

The model family is E3b's switch over sums with structure $(J, L)$: $J$ clusters, each an additive model
with $L$ binary switches. $L = 0$ is a flat $J$-mixture with a shared covariance, and $J = 1$ is E3's single
sum node. It has $J(3L + 3) + 2$ parameters. There are four moves, each with a warm start built from the
current fit:

| move | change | warm start |
|---|---|---|
| add a switch | $L \to L + 1$ | every cluster gains a small random offset, $p = 1/2$ (4 draws) |
| remove a switch | $L \to L - 1$ | the least used switch is dropped |
| split a cluster | $J \to J + 1$ | the heaviest cluster is duplicated with perturbed centres (4 draws) |
| merge two clusters | $J \to J - 1$ | the two closest clusters are averaged |

Each candidate is refitted by exact EM from its warm starts plus 32 random starts, and scored by BIC
on 2 000 training points. Greedy search starts at $(1, 0)$ and takes the best improving move until no
move improves. The reference is exhaustive search over $J \le 9$, $L \le 4$ (45 structures), with 32 random
starts each.

The data are E3's 8- and 16-mode grids, E3b's tree (3 clusters of rotated 2 × 2 grids) and E3's ring
of 8 modes. The "true" structure is the one the data were generated from. It is not always the only
exact representation: the octagon is two squares rotated by 45°, which is exactly $(2, 2)$ with 20
parameters, fewer than $(8, 0)$'s 26.

## Measured (2026-10-31)

| data | generated as | greedy finds | fits | exhaustive optimum | BIC: greedy / generating structure | held-out gap of the greedy model |
|---|---|---|---|---|---|---|
| 8-mode grid | (1, 3) | **(1, 3)** | 12 | (1, 3) | 10 047.3 / 10 047.3 | 0.002 |
| 16-mode grid | (1, 4) | (2, 3) | 18 | (1, 4) | 12 570.7 / 12 513.8 | 0.004 |
| tree | (3, 2) | **(3, 2)** | 18 | (3, 2) | 11 664.7 / 11 664.7 | 0.005 |
| ring | (8, 0) | **(2, 2)** | 14 | (4, 1) | 11 299.3 / 11 343.0 | 0.001 |

Greedy paths:
- 8-mode grid: add a switch three times.
- 16-mode grid: split, then add a switch three times.
- Tree: add a switch, split, add a switch, split.
- Ring: split, then add a switch twice.

Fitted from random starts only, as in the exhaustive search, the ring's $(2, 2)$ reaches a BIC of
13 933.4, a local optimum. Its warm start in the greedy search reaches 11 299.3. $(4, 1)$ and $(8, 0)$ both
reach 11 343.0 (same likelihood, same parameter count).

The greedy search takes 3.6–13 s, against about 200 s for the exhaustive search.

## What it shows

1. **Greedy structure search works for this family.** In all four cases the structure it finds
   represents the data to within 0.005 nats. In two it is the generating structure, found by the
   moves one would expect: add switches for a grid, add switches and split clusters for a tree.
2. **It can stop at an equivalent but larger structure.** On the 16-mode grid it split first and
   then grew three switches per cluster. $(2, 3)$ contains $(1, 4)$'s grid, but costs 9 more parameters
   (26 against 17; BIC 57 worse), and no single move leads from $(2, 3)$ to $(1, 4)$. A combined move (merge and add a
   switch) or a second pass from the result would be needed.
3. **Warm starts matter more than exhaustiveness.** On the ring, greedy finds the most compact exact
   representation, $(2, 2)$, which random starts miss. So it beats the exhaustive search's BIC with 14
   fits instead of 45. The structure search is only as good as the fits it compares, which is the
   lesson of E3's local optima again.
4. **This is a structure-changing learning rule, but outside Mycelium.** Each candidate is a new
   model, refitted from scratch apart from its warm start. It is scored exactly, because the exact
   posterior is small here. In a Mycelium graph a move would add or remove a switch's variable and
   factors. Doing that without rebuilding needs the append-only growth and deletion masks of
   [`Dynamic Graphs`](../vault/Factor%20Graphs/Dynamic%20Graphs.md), and once the posterior is too
   large it needs a score that approximate inference can deliver (E6 shows how biased that can be).

## Caveats

- One seed and one data draw per data set.
- BIC is the only score. The penalty is right for regular models, but mixtures are singular, so
  BIC is a heuristic here (as `Finding a Factor Graph` notes).
- $L$ is shared by all clusters, so per-cluster switch counts would need more moves.
- The exhaustive reference uses random starts only, which makes it weaker than the greedy search
  on the ring.
