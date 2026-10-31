# Factorial inference: when the exact posterior is too large

Run `julia -t auto --project=. examples/factorial_inference.jl` (about 10 minutes on 8 threads,
most of it in exact EM's random starts at $L = 12$); the test is `test/factorial_inference.jl`. This
belongs to the design note [`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md)
(§9, E6 of §10).

## Question

E3 and E3b trained the additive wiring by exact EM over all $2^L$ switch configurations, which stops
working around $L \approx 15$. Message passing in a factor graph is the scalable alternative. How
much does it cost in accuracy, and does Mycelium run it?

## Model and methods

The model is $z = c + \sum_l k_l \delta_l + \varepsilon$ in $d = 8$ dimensions, with $\varepsilon \sim \mathcal N(0, \Sigma)$
and independent bits $k_l \sim \text{Bernoulli}(p_l)$. The truth has random offset directions of
length 2.5, $p_l \in [0.3, 0.7]$, $\Sigma = 0.25 I$, and $L \in \{4, 8, 12\}$ switches. There are 2 000 training and
2 000 held-out points.

Three E-steps, each followed by the same closed-form M-step:

- **exact**: enumerates the $2^L$ configurations.
- **graph (collapsed BP)**: runs in Mycelium on the star graph shown below. The data clamp $z$; one
  `LinearConstraintFactor` encodes $z = c + \sum_l u_l + \varepsilon$; each $u_l$ has a `PriorFactor`
  emitting the switch's two-component mixture. The sum node's message to $u_l$ collapses the other
  switches' mixtures to their first two moments, via the new `_moments` method for
  `MixtureBelief` (`src/gaussian.md`, "Mixture inputs"). The graph is a tree, so one pass gives the
  fixed point. The E-step itself uses a vectorised copy of these messages, which agrees with
  Mycelium's `infer!` to $7 \cdot 10^{-13}$ on 20 test points for every $L$.
- **mean field**: a factorised $q(k) = \prod_l q_l$, updated by coordinate ascent on the evidence bound
  until it converges.

```text
data ── z ── sum node ── u_1 ── switch_1
                    ├─── u_2 ── switch_2
                    └─── …
```

The approximate posteriors are factorised, so the M-step takes $\mathbb E[k_l k_m] = q_l q_m$ for $l \ne m$.
Every fitted model is scored by its exact held-out log-likelihood. Each method is run twice:

- **from the true parameters**, to find where its own fixed point lies (its bias);
- **from 64 random starts**, keeping the best training likelihood, to see what learning finds in
  practice.

## Measured (2026-10-31)

The gap to the truth's held-out log-likelihood, in nats:

| $L$ (components) | method | from the truth | best of 64 random starts | wall time (random starts) |
|---|---|---|---|---|
| 4 (16) | exact | 0.025 | 0.025 | 2.0 s |
| | graph (collapsed BP) | 0.041 | 0.041 | 1.3 s |
| | mean field | 0.026 | 0.025 | 6.5 s |
| 8 (256) | exact | 0.032 | 0.282 | 35.6 s |
| | graph (collapsed BP) | 0.642 | 0.643 | 3.4 s |
| | mean field | 0.923 | 0.641 | 25.3 s |
| 12 (4 096) | exact | 0.039 | 0.215 | 473.5 s |
| | graph (collapsed BP) | 0.600 | 0.599 | 5.0 s |
| | mean field | 0.597 | 0.595 | 13.7 s |

The mean error of the approximate bit posteriors, measured at the true parameters against the exact
marginals:

| $L$ | graph | mean field |
|---|---|---|
| 4 | 0.010 | 0.002 |
| 8 | 0.085 | 0.088 |
| 12 | 0.128 | 0.172 |

## What it shows

1. **Mycelium runs the switch-and-sum wiring.** One `LinearConstraintFactor`, one `PriorFactor`
   per switch, and the collapsed mixture input are enough, and the graph agrees with its
   vectorised copy to rounding. The E-step is linear in $L$: 5 s for 64 fits at $L = 12$, where exact
   EM takes 474 s.
2. **With few switches the factorised posteriors are good.** At $L = 4$ all three methods end within
   0.04 nats of the truth.
3. **With overlapping switches they are biased, and the bias is large.** At $L = 8$ and $12$ in 8
   dimensions the offsets overlap, so the bits explain each other away: the evidence for one bit
   depends on the others. A factorised posterior cannot represent that. Both approximations then
   have fixed points 0.6 to 0.9 nats from the truth, even when they start *at* the truth. Exact EM's
   fixed point is within 0.04.
4. **That bias is larger than exact EM's local optima.** From random starts, exact EM stops 0.22 to
   0.28 nats from the truth, which is still better than either approximation.
5. **So Mycelium needs structured beliefs, not just more message passing.** This is E6's question,
   answered for this model: learning under collapsed or factorised beliefs is measurably biased once
   the latent causes interact. The candidates are:
   - beliefs over groups of switches (regions, as in Kikuchi or generalised BP);
   - expectation propagation with cavity-corrected messages instead of the one-pass collapse;
   - exact inference within small blocks, combined with message passing between them.

   None of these is implemented.

## Caveats

- One random truth per $L$, one data draw.
- The collapsed BP here is the simplest graph version. Expectation propagation on the same graph
  could do better.
- The M-step treats the approximate posteriors as if they were exact (variational EM for mean field;
  for the collapsed BP there is no bound behind it).
