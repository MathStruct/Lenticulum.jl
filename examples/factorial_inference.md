# Factorial inference: when the exact posterior is too large

Run `julia -t auto --project=. examples/factorial_inference.jl` (about two hours on 8
threads, most of it in EP); the test is `test/factorial_inference.jl`. This belongs to the design
note [`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§9, E6 of
§10). The second round of reviews (note §13) asked for more seeds, the right name for the graph
E-step, and controls that separate the posterior's marginals from its pair moments; this version
has them.

## Question

E3 and E3b trained the additive wiring by exact EM over all $2^L$ switch configurations, which stops
working around $L \approx 15$. Message passing in a factor graph is the scalable alternative. How
much does it cost in accuracy, and does Mycelium run it?

## Model and methods

The model is $z = c + \sum_l k_l \delta_l + \varepsilon$ in $d = 8$ dimensions, with $\varepsilon \sim \mathcal N(0, \Sigma)$
and independent bits $k_l \sim \text{Bernoulli}(p_l)$. Each truth has random offset directions of
length 2.5, $p_l \in [0.3, 0.7]$, $\Sigma = 0.25 I$, and $L \in \{4, 8, 12\}$ switches. There are five truth seeds,
each with its own 2 000 training and 2 000 held-out points.

Six E-steps, each followed by the same closed-form M-step:

- **exact**: enumerates the $2^L$ configurations.
- **exact marginals, factorised pairs** (a diagnostic): the exact $q_l = P(k_l = 1 \mid z)$, but the
  M-step takes $\mathbb E[k_l k_m] = q_l q_m$ for $l \ne m$, as the factorised methods must.
- **graph (ADF: one pass, prior moments)**: runs in Mycelium on the star graph shown below. The
  data clamp $z$; one `LinearConstraintFactor` encodes $z = c + \sum_l u_l + \varepsilon$; each $u_l$ has a
  `PriorFactor` emitting the switch's two-component mixture. The sum node's message to $u_l$
  collapses the other switches' *prior* mixtures to their first two moments, via the `_moments`
  method for `MixtureBelief` (`src/gaussian.md`, "Mixture inputs"). That is one pass of
  assumed-density filtering, not a fixed point of an iterative method: the other switches are never
  updated by the data (Gemini and Mistral, second round). The E-step uses a vectorised copy of these
  messages, which agrees with Mycelium's `infer!` to $1.3 \cdot 10^{-12}$ on 20 test points in every case.
- **mean field**: a factorised $q(k) = \prod_l q_l$, updated by coordinate ascent on the evidence bound
  until it converges.
- **mean field + linear response** (Welling & Teh 2004; Gemini's suggestion): the pair moments
  $q_l q_m + C_{lm}$ with $C = (D^{-1} + G_\text{off})^{-1}$, $D = \operatorname{diag}(q_l(1 - q_l))$ and $G_\text{off}$ the
  off-diagonal part of $\Delta^\top \Sigma^{-1} \Delta$. The diagonal of $G$ is not included, because $k_l^2 = k_l$
  puts it into the bits' own natural parameters.
- **EP (joint Gaussian pairs)**: Gaussian expectation propagation on the same star graph (Minka
  2001), with the Gaussian sites on $u_l$ refined against cavities until they converge (damping 0.5,
  at most 30 sweeps). The bit marginals come from the tilted distributions and the pair moments from
  the joint Gaussian over $(u_1, \dots, u_L)$. Not in Mycelium: it is a separate per-point loop.

```text
data ── z ── sum node ── u_1 ── switch_1
                    ├─── u_2 ── switch_2
                    └─── …
```

Every fitted model is scored by its exact held-out log-likelihood. Every method is run from the true
parameters, to find where its own fixed point lies (its bias). For seed 1, the fast methods are also
run from 64 random starts, keeping the best training likelihood. EP was run on seeds 1–3 only (about
10 minutes per fit at $L \ge 8$).

## Measured (2026-10-31 and 11-01)

The gap of each fixed point to the truth's held-out log-likelihood, in nats, started from the true
parameters: the mean over seeds, with the range.

| method | $L = 4$ (16 components) | $L = 8$ (256) | $L = 12$ (4 096) |
|---|---|---|---|
| exact | 0.023 (0.012–0.031) | 0.032 (0.030–0.036) | 0.042 (0.028–0.051) |
| exact marginals, factorised pairs | 0.077 (0.023–0.156) | 0.513 (0.416–0.636) | 0.578 (0.462–0.753) |
| graph (ADF) | 0.168 (0.033–0.381) | 0.613 (0.478–0.672) | 0.593 (0.475–0.792) |
| mean field | 0.124 (0.021–0.293) | 0.643 (0.492–0.923) | 0.594 (0.460–0.813) |
| mean field + linear response | 0.159 (0.021–0.465) | 0.891 (0.720–1.123) | 0.802 (0.624–1.086) |
| EP, seeds 1–3 | 0.046 (0.033–0.056) | 0.278 (0.157–0.402) | 0.558 (0.266–1.080) |

The largest |cosine| between two offsets is 0.31–0.67 at $L = 4$, 0.60–0.77 at $L = 8$ and 0.69–0.84 at
$L = 12$. The mean error of the bit marginals at the true parameters, against the exact ones (range
over seeds):

| $L$ | graph (ADF) | mean field | EP |
|---|---|---|---|
| 4 | 0.007–0.070 | 0.001–0.101 | 0.008–0.037 |
| 8 | 0.085–0.109 | 0.088–0.139 | 0.033–0.051 |
| 12 | 0.118–0.128 | 0.140–0.214 | 0.056–0.074 |

Seed 1 from 64 random starts, best training likelihood, with the wall time of the 64 fits:

| $L$ | exact | graph (ADF) | mean field | mean field + LR |
|---|---|---|---|---|
| 4 | 0.025 (1.6 s) | 0.041 (1.1 s) | 0.025 (5.5 s) | 0.338 (11.7 s) |
| 8 | 0.282 (30.7 s) | 0.643 (2.9 s) | 0.641 (21.0 s) | 0.847 (23.7 s) |
| 12 | 0.215 (436.5 s) | 0.599 (4.3 s) | 0.595 (11.5 s) | 0.754 (34.0 s) |

## What it shows

1. **Mycelium runs the switch-and-sum wiring.** One `LinearConstraintFactor`, one `PriorFactor`
   per switch, and the collapsed mixture input are enough, and the graph agrees with its
   vectorised copy to rounding. The E-step is linear in $L$: 4 s for 64 fits at $L = 12$, where exact
   EM takes 437 s.
2. **Where the switches barely overlap, factorised posteriors are good; where they overlap, they are
   biased, on every seed.** At $L = 4$ the factorised methods are near exact on seeds with small
   overlaps (cosine 0.31–0.48) and 0.15–0.47 nats off on seeds with cosine 0.67. At $L = 8$ and $12$
   every factorised method has fixed points 0.42 to 1.12 nats from the truth, against 0.03–0.05 for
   exact EM. That bias is larger than exact EM's local optima from random starts (0.22–0.28).
3. **The bias is in the pair moments, not in the marginals.** With the *exact* marginals and
   factorised pair moments, EM is still 0.42–0.75 nats off at $L \ge 8$, nearly as far as ADF and mean
   field. At the true parameters, $\mathbb E_z[q_l q_m] = p_l p_m - \mathbb E_z \operatorname{Cov}(k_l, k_m \mid z)$, so the
   M-step's normal equations are off by the explaining-away covariance and the truth is not a
   fixed point (Gemini's argument, second round; confirmed here). Better marginals alone cannot fix
   this.
4. **Pair moments from a joint Gaussian help, but not reliably.** EP's marginals are the best of the
   approximations (0.03–0.07), and its fixed points are much closer at $L = 8$ (0.16–0.40). At $L = 12$
   it is 0.27 and 0.33 on two seeds and 1.08 on the third, worse than mean field. It costs three to four
   orders of magnitude more than the ADF E-step here (230–1 075 s per fit against 0.1–0.2 s).
5. **Linear response makes it worse.** The linear-response pair moments move the fixed points
   *further* from the truth (0.72–1.12 at $L = 8$) and break random starts at $L = 4$. Either the
   correction is too crude for these binary posteriors, or it is applied in the wrong place (the
   mean-field fixed point it linearises is itself biased). Not investigated further.
6. **So scaling the additive wiring needs pair moments that are right, not more passes.** The
   options that give them exactly are exact inference within small blocks of switches (with message
   passing between blocks) and region beliefs over groups of switches (Kikuchi, generalised BP).
   Neither is implemented. Gaussian EP is a partial answer that is costly and unstable at $L = 12$.

## Caveats

- Five random truths per $L$, one data draw each; EP on three of them.
- One EP variant (Gaussian sites on $u_l$, fixed damping, updates that fail to factorise skipped); the
  $L = 12$ outlier on seed 2 was not traced.
- The M-step treats the approximate posteriors as if they were exact (variational EM for mean
  field; ADF and EP have no bound behind them).
- E6 tests moment collapse and factorised or Gaussian pair moments, not the Bethe free energy
  itself (ChatGPT, second round).
