# Structured inference inside the hierarchy (E7)

Run `julia -t auto --project=. examples/hierarchy_inference.jl` (about 75 minutes on 8 threads,
most of it in the random starts); the test is `test/hierarchy_inference.jl`. This belongs to the
design note [`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (E7 of
§10). The second round of reviews asked for it (Grok; Gemini called it "E7").

## Question

E6 showed that factorised posteriors over the switches bias EM, and that the bias sits in the pair
moments $\mathbb E[k_l k_m]$, not in the marginals. The remedy E6 proposed is to keep the pair moments
exact where they matter: exact inference inside small blocks of switches, factorised across
blocks. Does that work inside E3b's hierarchy (a top switch over clusters, each cluster an additive
model), and does it matter which switches share a block?

## Model and methods

The model is E3b's switch over sums, in $d = 8$ dimensions:

$$
j \sim \text{Categorical}(w), \quad k_l \mid j \sim \text{Bernoulli}(p_{jl}), \quad z = c_j + \sum_{l=1}^L k_l \delta_{jl} + \varepsilon, \quad \varepsilon \sim \mathcal N(0, \Sigma_j),
$$

with $J = 3$ clusters and $L \in \{6, 8\}$ switches per cluster, so $J \cdot 2^L$ = 192 or 768 components.
Each truth has cluster centres 6 apart in random directions, offsets of length 2.5 in random
directions, $p_{jl} \in [0.3, 0.7]$ and $\Sigma_j = s_j I$ with $s_j \in [0.2, 0.3]$. There are three truth
seeds, each with 4 000 training and 2 000 held-out points.

Every E-step keeps the top switch exact, $q(j) \propto \exp(\text{ELBO}_j)$, and approximates $q(k \mid j)$
by a product over blocks of switches. Within a block the posterior and its pair moments are exact;
across blocks $\mathbb E[k_l k_m] = q_l q_m$. The block updates are coordinate ascent on the evidence bound
(structured mean field, as in Ghahramani & Jordan 1997). The M-step is closed form per cluster.
Four blockings, with $B = L/2$:

- **exact**: one block of all $L$ switches (the oracle);
- **mean field**: blocks of one switch;
- **fixed blocks of $B$**: consecutive switches, whatever their geometry;
- **overlap blocks of at most $B$**: chosen per cluster at every E-step by greedily merging the
  switches whose offsets overlap most (largest $|\cos|$ in the metric $\Sigma_j^{-1}$).

The exact E-step agrees with brute-force enumeration to $10^{-15}$, and the bounds are ordered
mean field ≤ blocks ≤ exact (the test checks both). Every fitted model is scored by its exact
held-out log-likelihood and by the energy distance of 1 000 generated samples against held-out
points. Every method is run from the true parameters, to locate its own fixed point; for seed 1 it
is also run from 16 random starts, keeping the best training likelihood.

## Measured (2026-11-01)

Gap of the fixed point reached from the true parameters to the truth's held-out log-likelihood, in
nats, for seeds 1 / 2 / 3:

| method | $L = 6$ (192 components) | $L = 8$ (768 components) |
|---|---|---|
| exact | 0.026 / 0.043 / 0.043 | 0.043 / 0.058 / 0.039 |
| mean field | 0.457 / 0.360 / 0.253 | 0.587 / 0.643 / 0.681 |
| fixed blocks of $L/2$ | 0.384 / 0.151 / 0.399 | 0.419 / 0.352 / 0.483 |
| overlap blocks of $\le L/2$ | 0.028 / 0.123 / 0.062 | 0.217 / 0.190 / 0.327 |

Seed 1 from 16 random starts (best training likelihood), with the wall time of the 16 fits:

| method | $L = 6$ | $L = 8$ |
|---|---|---|
| exact | 0.501 (359 s) | 0.558 (528 s) |
| mean field | 0.656 (605 s) | 0.751 (735 s) |
| fixed blocks | 0.526 (403 s) | 0.731 (454 s) |
| overlap blocks | 0.515 (400 s) | 0.538 (462 s) |

Energy distances (0.007–0.032, against sampling floors of 0.008–0.021) stay within 0.013 of the
floor and are not ordered like the gaps: on seed 1 at $L = 8$, exact EM scores 0.027 and fixed blocks
0.026. They do not separate the methods here; the log-likelihood gap does.

## What it shows

1. **The E6 bias reappears one level down.** With the top switch exact, mean field over the switches
   inside a cluster is 0.25–0.46 nats off at $L = 6$ and 0.59–0.68 at $L = 8$, against 0.03–0.06 for
   exact EM: the same size as in E6's flat additive model.
2. **Exact blocks remove most of it if the blocks follow the overlaps.** Blocks chosen by overlap
   cut the mean-field gap to 6–34 % of it at $L = 6$ and to 30–48 % at $L = 8$. On seed 1 at $L = 6$ the
   overlap blocks are as good as exact (0.028 against 0.026).
3. **Blocks that ignore the geometry help little.** Consecutive blocks of the same size are 0.15–0.40
   nats off at $L = 6$ (on seed 3 worse than mean field) and 0.35–0.48 at $L = 8$. What matters is which
   pairs of switches the block keeps exact, as E6's diagnosis (the bias is in the pair moments of
   overlapping switches) predicts.
4. **From random starts, local optima dominate.** Seed 1's best of 16 random starts is 0.50–0.56 nats
   off for exact EM, so at this size initialisation costs more than the E-step. Overlap blocks
   match exact there (0.515 and 0.538 against 0.501 and 0.558); mean field and, at $L = 8$, fixed
   blocks stay worse.
5. **This size does not show the saving.** At $L \le 8$ exact enumeration (256 states per cluster) is
   cheaper than the block coordinate ascent (5 s against 27 s per fit at $L = 8$). The blocked E-step
   costs $L/B \cdot 2^B$ per sweep instead of $2^L$, which pays off only for larger $L$; not measured.

## Caveats

- Three random truths per $L$, one data draw each; random starts on one seed only.
- The overlap criterion is a heuristic (greedy merging by $|\cos|$, at most $B = L/2$ per block), and
  one block size was tried. With $B = L/2$ there are only two blocks, so the factorised pair
  moments are those across the two halves.
- The M-step treats the structured posterior as exact (variational EM). Fits stop after 200 EM
  iterations or when the parameters move by less than $10^{-6}$.
- E7 has an exact oracle by construction; the cases where blocks are needed ($L \gtrsim 15$) are not
  tested.
