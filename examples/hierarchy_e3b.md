# E3b: a switch over sums

Run `julia -t auto --project=. examples/hierarchy_e3b.jl` (under a minute on 8 threads); the test
is `test/hierarchy_e3b.jl`. This follows E3 in the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§8, §10).

## Question

E3 showed that one sum node fed by binary switches wins on product grids. One level deeper, a
switch chooses which sum node generates the point. The components are then arranged as a tree:
clusters, each with its own grid. Does the deeper wiring win on data with that arrangement, and
does exact EM still train it?

## Setup

The data have 12 modes: 3 clusters at the corners of a triangle with circumradius 5 and weights 0.4,
0.35 and 0.25. Each cluster is a 2 × 2 grid with offsets of length 1.8 and 1.4, rotated by its own
angle (0, 0.5 and 1.1 rad), with its own bit probabilities. All modes share E3's covariance. The
modes are at least 1.40 apart. Because the local grids are rotated differently, the data are not a
single product grid.

All models are fitted by exact EM, the composed ones from 256 random starts:

- **hierarchical**, $J = 3$, $L = 2$, 29 parameters. A top switch $j$ chooses the cluster:
  $p(z) = \sum_j \pi_j \sum_k \prod_l p_{jl}^{k_l} (1 - p_{jl})^{1-k_l}\, \mathcal N(z;\, c_j + \sum_l k_l \delta_{jl},\, \Sigma)$.
  The posterior over $(j, k)$ is exact, and the M-step is a weighted least squares per cluster
  (`em_hier`).
- **additive**, one sum node with $L \in \{2, 3, 4\}$ switches (E3's `em_additive`).
- **flat**, $K \in \{3, 6, 12\}$ components, with 10 EM restarts.

## Measured (2026-10-31)

The truth scores −2.886 on 2 000 held-out points. Gaps are in nats; smaller is better.

| model | parameters | gap, $n = 150$ | gap, $n = 2000$ | starts at the best ($n = 150$ / $2000$) |
|---|---|---|---|---|
| **hierarchical, $J = 3$, $L = 2$** | **29** | **0.091** | **0.008** | 58 / 57 of 256 |
| additive, $L = 2$ | 11 | 0.803 | 0.798 | 40 / 41 |
| additive, $L = 3$ | 14 | 0.758 | 0.703 | 16 / 14 |
| additive, $L = 4$ | 17 | 0.693 | 0.635 | 3 / 1 |
| flat, $K = 3$ | 17 | 0.653 | 0.643 | — |
| flat, $K = 6$ | 35 | 0.456 | 0.367 | — |
| flat, $K = 12$ | 71 | 0.996 | 0.016 | — |

The hierarchical model's EM log-likelihood never decreased, in any start.

## What it shows

1. **The deeper wiring wins on tree-structured data.** With 29 parameters the hierarchical model
   is within 0.008 nats of the truth at $n = 2000$. The flat mixture needs all 12 components
   (71 parameters) to get within 0.016, and at $n = 150$ it overfits (0.996), where the
   hierarchical model is at 0.091.
2. **One sum node cannot express it.** With up to 16 components, the single additive model stays
   0.64 to 0.80 away. Its offsets are shared by every cluster, so it cannot rotate the local grids.
3. **Exact EM scales to the deeper wiring.** The posterior over $(j, k)$ has $J \cdot 2^L$ states and
   stays exact. A quarter of the random starts reach the optimum, more than for E3's flat
   additive grids.
4. **The pattern from E3 holds one level up:** composition is a prior on how the components are
   arranged. A switch over sums is the right prior for clustered grids, as one sum is for one grid.

## Caveats

- These are synthetic data with exactly the hierarchical model's structure.
- There is one seed.
- The exact posterior grows as $J \cdot 2^L$, so wider or deeper trees will need approximate
  inference (messages in a Mycelium graph), which this experiment does not test.
