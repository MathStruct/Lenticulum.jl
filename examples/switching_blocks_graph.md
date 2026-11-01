# Blocks of switches as factors in the graph

Run `julia -t auto --project=. examples/switching_blocks_graph.jl` (about 3 minutes on 8
threads); the test is `test/switching_blocks_graph.jl`. This uses the new
`SwitchingConstraintFactor` (`src/switching.jl`, `src/switching.md`) and belongs to the design
note [`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§10, after
E6 and E7).

## Question

E6 (`factorial_inference.md`) found that Mycelium's star graph for the additive wiring
$z = c + \sum_l k_l \delta_l + \varepsilon$ learns biased parameters once the switches overlap, because a
factorised posterior gets the pair moments $\mathbb E[k_l k_m]$ wrong. E7 (`hierarchy_inference.md`)
removed most of the bias with exact blocks of overlapping switches, inside a structured
mean-field iteration. With a switching factor, a block is a single factor in the graph. Is
changing the factor enough, or does the graph also have to iterate between blocks the way E7 did?

## The graph

A block $b$ of $B$ switches becomes one `SwitchingConstraintFactor` with $2^B$ outcomes. The outcome is
the configuration of the block's bits, the right-hand sides are $-\sum_{l \in b} k_l \delta_l$, and the
log-weights are the configuration priors:

```text
data ── z ── sum node ── u′_1 ── block factor 1 ── k_1   (2^B states)
                    ├─── u′_2 ── block factor 2 ── k_2
                    └─── …
```

The sum node is E6's `LinearConstraintFactor` $z + \sum_b u'_b = c$. The graph is a tree, so one pass
of `infer!` is its fixed point. The marginal of $k_b$ is then the exact posterior over the block's
configurations, *given* a Gaussian stand-in for the other blocks: the sum node collapses their
mixture messages to their prior moments (assumed-density filtering across blocks). Within a block
the pair moments are exact; across blocks the M-step uses $q_l q_m$.

Two checks, on every seed and $L$:

- Mycelium's `infer!` on the block graph agrees with the vectorised E-step to
  $5.2 \cdot 10^{-13}$–$4.9 \cdot 10^{-12}$ on 10 held-out points;
- with blocks of one switch the E-step is E6's ADF, to $1.4$–$2.2 \cdot 10^{-15}$.

The truths, data, M-step and scores are E6's. There are five seeds and $L \in \{8, 12\}$ switches in 8
dimensions, with 2 000 training and 2 000 held-out points. Every E-step is run from the true
parameters to its EM fixed point. The control is E7's structured mean field with the same blocks
(recomputed by overlap at every E-step) on the same data: the same within-block posterior, but
iterated between blocks with the other blocks' data-dependent means.

## Measured (2026-11-01)

Gap of each fixed point to the truth's held-out log-likelihood, in nats, for seeds 1–5:

| E-step | $L = 8$ | $L = 12$ |
|---|---|---|
| exact | 0.032 / 0.036 / 0.032 / 0.032 / 0.030 | 0.039 / 0.049 / 0.043 / 0.028 / 0.051 |
| graph, blocks of 1 (E6's ADF) | 0.642 / 0.672 / 0.665 / 0.478 / 0.608 | 0.600 / 0.792 / 0.572 / 0.528 / 0.475 |
| graph, fixed blocks of 4 | 0.519 / 0.332 / 0.545 / 0.284 / 0.528 | 0.589 / 0.564 / 0.556 / 0.485 / 0.462 |
| graph, overlap blocks of ≤ 4 | 0.214 / 0.328 / 0.389 / 0.327 / 0.216 | 0.554 / 0.571 / 0.525 / 0.433 / 0.358 |
| graph, overlap blocks of ≤ 6 | | 0.490 / 0.456 / 0.480 / 0.414 / 0.349 |
| mean field (E6) | 0.923 / 0.492 / 0.588 / 0.538 / 0.674 | 0.597 / 0.813 / 0.564 / 0.537 / 0.460 |
| structured MF, overlap blocks of ≤ 4 (E7) | 0.058 / 0.144 / 0.276 / 0.127 / 0.085 | 0.554 / 0.171 / 0.329 / 0.167 / 0.190 |
| structured MF, overlap blocks of ≤ 6 (E7) | | 0.216 / 0.114 / 0.175 / 0.106 / 0.206 |

Means over the seeds:

| E-step | $L = 8$ | $L = 12$ |
|---|---|---|
| graph, blocks of 1 | 0.613 | 0.593 |
| graph, overlap blocks ($\le 4$ at $L = 8$, $\le 6$ at $L = 12$) | 0.295 | 0.438 |
| structured MF, the same blocks | 0.138 | 0.163 |
| exact | 0.032 | 0.042 |

## What it shows

1. **The factor puts E7's blocks into the graph unchanged.** A block of switches is one
   `SwitchingConstraintFactor`, and Mycelium runs it with the generic message passing; the
   graph agrees with the hand-written E-step to $10^{-12}$.
2. **Blocks alone, in one pass, remove part of the bias.** Overlap blocks cut the graph's gap
   from 0.61 to 0.30 nats on average at $L = 8$. At $L = 12$ the cut is from 0.59 to 0.44, with blocks
   of 6. Fixed blocks help less, though on two seeds at $L = 8$ they do as well as overlap blocks.
3. **The rest needs iteration between blocks.** With the same blocks, E7's structured mean field
   is 2.1 times closer to the truth at $L = 8$ (0.14) and 2.7 times closer at $L = 12$ (0.16). Iteration
   without blocks does not help: mean field over single switches is as biased as the one-pass
   graph (0.46–0.92). What removes the bias is exact blocks *and* the other blocks' data-dependent
   means. The sum node on belief propagation gives neither: it collapses the other blocks to their
   prior moments.
4. **So the missing piece was the message at the sum node, not the factor.** It is now in Mycelium
   (next section).

## The hybrid graph: the sum node on mean field (added 2026-11-01)

Mycelium nodes now take a message rule (`factor!(b, name, f; rule = MeanField())`, Mycelium's
`passing.md`, "Mean-field nodes"). A mean-field node sends $\exp \mathbb E_q[\log f]$ over the other
variables' current marginals, and Lenticulum's `LinearConstraintFactor` implements it: its
belief-propagation message with each input at its mean. With the sum node on mean field and the
block factors on belief propagation, the graph is the hybrid of Riegler et al. (2013). Each block's
posterior is exact given the other blocks' current means, which is E7's structured mean field.
`hybrid_posteriors` runs it by sequential sweeps (the sum node's message to each block in turn,
then the blocks' outcomes) from the blocks' prior mixtures, E7's starting point.

Measured on the same truths (10 held-out points per seed and $L$, the true parameters, overlap
blocks of $\le 4$, block factors of width $10^{-8}$):

- the block means and second moments agree with E7's structured mean field to at most
  $8.9 \cdot 10^{-7}$ (seed 4, $L = 12$), and to $9 \cdot 10^{-9}$–$1.2 \cdot 10^{-7}$ on the other nine cases;
- the iteration converges (message residual below $10^{-6}$) in 3–57 sweeps;
- EM with the hybrid graph as its E-step (`fit_hybrid`, one graph per point) reaches the parameters
  E7's code reaches: seed 1, $L = 8$, 300 points, 30 iterations, $|\Delta W| = 3.2 \cdot 10^{-8}$, held-out gaps
  0.3291328 against 0.3291328. It took 28 s, against 2 s for E7's vectorised code.

So the structured-mean-field numbers of the table (0.14 at $L = 8$, 0.16 at $L = 12$) are what learning
through the Mycelium graph gives, with the generic message passing and two rules, one per factor
type. Two things are still missing. The graph has no free energy for such hybrids, so
`parameter_step` refuses them and the M-step is done outside. And building one graph per data point
is slow.

## Caveats

- Five random truths per $L$, one data draw each, fixed points from the true parameters only (no
  random starts).
- The overlap criterion (greedy merging by $|\cos|$ in the noise metric) is a heuristic. At $L = 12$
  with blocks of $\le 4$, seed 1 is as bad under structured mean field as under the one-pass graph
  (0.554 both); not investigated.
- The structured mean field comes from E7's code with one cluster. Its M-step is algebraically E6's
  closed form, but it is a separate implementation.
- The block factors' mixture components have width $10^{-4}$, as in E6. The E-step is exact in the
  limit; at this width the difference was not measured.
