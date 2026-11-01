# A neural reverse kernel as a reference for E4(b)

Run `julia -t auto --project=. examples/neural_reference.jl` (about 5 minutes on 8 threads); the
test is `test/neural_reference.jl`. This belongs to the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§10). The second
round of reviews asked for it (Grok): how much does E4(b) lose by staying inside the
switching-linear family, against the standard neural reverse kernel trained the standard way?

## Setup

The data, forward process, prior and scores are E4(b)'s (`examples/switching_chain_e4b.md`): two
moons and a ring, 4 000 training and 1 000 held-out points, $\alpha_T^2 = 0.8^{20}$ spread evenly over $T$
steps, a Gaussian prior fitted to the noised training points, and the importance-weighted bound
with $S = 64$ forward paths per test point. The bound is a lower bound on $\log p(z_0)$, so every gap
below is an upper bound on the model's true gap, up to Monte Carlo error.

The reverse kernel is that of Ho, Jain & Abbeel (2020), $p(z_{t-1} \mid z_t) = \mathcal N(\mu_\theta(z_t, t), \sigma_t^2 I)$:

- the mean comes from a noise predictor, $\mu = (z_t - \beta_t \hat\varepsilon_\theta(z_t, t) / \sqrt{1 - \bar\alpha_t}) / \sqrt{1 - \beta_t}$;
- $\hat\varepsilon_\theta$ is one MLP shared by all steps. Its inputs are $z_t$ and 16 sinusoidal features of $t/T$,
  with two hidden SiLU layers of width $H$. It is trained on Ho et al.'s simple loss $\mathbb E\|\varepsilon - \hat\varepsilon\|^2$
  by Adam (batch 512, learning rate $2 \cdot 10^{-3}$ with cosine decay, 30 000 steps), using
  hand-written backpropagation, which agrees with finite differences to $10^{-9}$ (in the test);
- $\sigma_t^2$ is the bound's optimum given the mean, $\mathbb E\|z_{t-1} - \mu(z_t)\|^2 / 2$, on one fresh pair per
  training point. Proposition 9 makes each step's term separate.

Parameters counted: the network's, plus $T$ variances, plus the prior's 5.

## Measured (2026-11-01)

Gap of the importance-weighted bound to the truth's held-out $\log p$ (−0.992 on moons, −1.764 on
the ring), in nats, with the single-path bound ($S = 1$, the ELBO) and the energy distance of 1 500
samples:

| $T$, $H$ | parameters | moons: IW / ELBO / energy distance | ring: IW / ELBO / energy distance |
|---|---|---|---|
| 20, 64 | 5 531 | 0.804 / 0.885 / 0.0099 | 0.573 / 0.785 / 0.0095 |
| 100, 64 | 5 611 | 0.549 / 0.628 / 0.0080 | 0.114 / 0.289 / 0.0042 |
| 100, 128 | 19 307 | 0.494 / 0.592 / 0.0079 | 0.097 / 0.285 / 0.0040 |
| 100, 128, 100 000 Adam steps | 19 307 | 0.412 / 0.556 / 0.0072 | 0.085 / 0.281 / 0.0037 |
| 1 000, 128 | 20 207 | 0.315 / 0.461 / 0.0044 | **0.009** / 0.183 / 0.0041 |

E4(b)'s references on the same data and scores (gaps; energy distances on moons / ring):

| model | parameters | moons | ring |
|---|---|---|---|
| flat mixture, $K = 16$ | 95 | 0.013 (0.0026) | 0.019 (0.0034) |
| switching chain, $K = 4$, $T = 20$ | 1 085 | 0.341 (0.0054) | 0.309 (0.0052) |
| data against data (energy-distance floor) | | (0.0008) | (0.0038) |

## What it shows

1. **With small steps the neural kernel closes the gap on the ring, but not on moons.** At $T = 1000$ the
   ring's bound is 0.009 nats from the truth, better than the best flat mixture (0.019) and every
   switching chain (0.31–0.32 at $K = 4$). On moons it is 0.315, no better than a 4-outcome chain
   and far behind the flat 16-mixture (0.013). Its samples are near the floor on the ring and
   5.5 times the floor on moons.
2. **The fixed-forward bound is not what pins E4(b)'s chains, given a rich enough kernel and small
   enough steps.** On the ring the same objective that leaves $K$-outcome chains at a $K$-mixture's
   score (Proposition 9, E4(b)) trains a neural Gaussian kernel to within 0.01 nats. That matches
   Proposition 9's per-step reading. Each step's term is the KL from the true reverse kernel to the
   model's; a time-conditioned network can follow the true kernel as $\beta \to 0$, but a kernel
   with $K$ outcomes and a linear gate cannot. So the objective control of §10 (exact likelihood
   rescues the chains) and this reference are consistent with one statement: the fixed-forward bound
   needs per-step kernels close to the true reverse kernel.
3. **The neural kernel needs many steps and many parameters.** At $T = 20$ it is worse than E4(b)'s
   4-outcome chains at the same depth on both data sets. At its best it uses 20 207 parameters against 95 for
   the flat mixture. Moons' residual falls with $T$ (0.80, 0.49, 0.32 at $T$ = 20, 100, 1 000) and a
   little with training (0.49 to 0.41 at $T = 100$), so it is not converged. Moons' thin curves at
   noise 0.1 are where a smooth $\hat\varepsilon$ is weakest.
4. **Whether 64 paths saturate the bound here is not known.** The single-path and 64-path bounds
   differ by 0.08–0.15 nats on moons and by 0.17–0.20 on the ring at $T \ge 100$. That is comparable
   to E4(b)'s 4-outcome chains (0.24 at $T = 20$), for which E4(b)'s sweep found no further gain beyond
   64 paths. The sweep was not repeated for the neural chain, so its true gaps may be smaller than
   the table shows.

## Caveats

- One architecture and one training recipe, without tuning. No EMA of the weights, no learned
  variances, no data-dependent noise schedule, and one training seed.
- $T = 1000$ was run only at $H = 128$ with 30 000 steps.
- The IW bound at $S = 64$ was not checked for saturation on the neural chain (point 4).
- This is a reference, not a candidate: it puts a network inside the factor, which the brief
  excludes.
