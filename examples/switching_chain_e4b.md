# E4(b): switching chains on data that are not a mixture

Run `julia -t auto --project=. examples/switching_chain_e4b.jl` (about 95 minutes on 8 threads,
most of it in the 30 EM restarts of the large flat mixtures); the test is
`test/switching_chain_e4b.jl`. Experiment E4(b) of the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§8, §10).

## Question

A chain of $T$ directed switching steps with $K$ outcomes each generates a mixture over $K^T$ paths
with input-dependent gates. On data that no finite mixture contains, does this beat one flat
Gaussian mixture? E4(a) and the small-step test used mixture data, where the answer is bounded by
Proposition 8. Here no model is exact.

## Setup

There are two data sets, each a uniform measure on curves convolved with isotropic Gaussian noise.
Their densities are computed exactly by a one-dimensional quadrature over the curve parameter, and
integrate to one to $10^{-6}$ on a grid (in the test).

- **two moons:** the usual two half-circles, with noise 0.1;
- **ring:** a circle of radius 1.5, with noise 0.15.

Every model is trained on the same 4 000 points and scored on 1 000 held-out points.

- **flat** ($6K - 1$ parameters): a $K$-component mixture fitted by EM with 30 restarts,
  scored by its exact log-likelihood.
- **chain** ($T(15K - 1) + 5$ parameters): $T$ reverse steps with $\alpha_T^2 = 0.0115$, as in E4(a).
  - Each step is a directed switching kernel with $K$ outcomes, fitted by conditional maximum
    likelihood (`SwitchingSmallSteps.fit_pairs`: joint EM, then 120 Adam steps).
  - Each step's pairs are built from the training points with fresh forward noise, which is the
    usual diffusion training data.
  - The prior at $T$ is a Gaussian fitted to the noised training points.
  - The chain's marginal has no closed form, so it is scored by the importance-weighted bound
    with $S = 64$ forward paths per test point. That is a lower bound on $\log p(z_0)$ in expectation,
    so a chain that beats a flat mixture beats it for real, up to Monte Carlo error.

## Measured (2026-10-31)

The table gives the gap to the truth's held-out $\log p$, in nats; smaller is better. The truth
scores −0.992 on moons and −1.764 on the ring. The chain's figure is the gap of its
importance-weighted bound.

| model | parameters | moons | ring |
|---|---|---|---|
| flat, $K = 2$ | 11 | 0.733 | 1.034 |
| chain, $K = 2$, $T = 5 / 10 / 20$ | 150 / 295 / 585 | 0.734 / 0.732 / 0.715 | 1.030 / 1.015 / 0.938 |
| flat, $K = 3$ | 17 | 0.498 | 0.621 |
| chain, $K = 3$, $T = 5 / 10 / 20$ | 225 / 445 / 885 | 0.499 / 0.499 / 0.492 | 0.622 / 0.608 / 0.580 |
| flat, $K = 4$ | 23 | 0.335 | 0.318 |
| chain, $K = 4$, $T = 5 / 10 / 20$ | 300 / 595 / 1 185 | 0.338 / 0.340 / 0.341 | 0.322 / 0.323 / 0.309 |
| flat, $K = 8$ | 47 | 0.038 | 0.048 |
| flat, $K = 16$ | 95 | **0.013** | **0.019** |
| flat, $K = 32$ | 191 | 0.029 | 0.024 |
| flat, $K = 64$ | 383 | 0.031 | 0.033 |

Energy distances of 1 500 generated samples against the held-out data:

| model | moons | ring |
|---|---|---|
| data against data (the floor) | 0.0008 | 0.0038 |
| flat, $K = 4$ | 0.0039 | 0.0062 |
| flat, $K = 16$ | 0.0026 | 0.0034 |
| chain, $K = 4$, $T = 20$ | 0.0054 | 0.0052 |

At $T = 20$, the importance-weighted bound is above the plain ELBO by 0.08 to 0.25 nats, so the
bound is still loose there. At $T = 5$ the two are within 0.06 of each other.

One flat fit at first misled. With 3 EM restarts, the flat $K = 3$ mixture on moons stopped at a
local optimum with gap 0.612, which made the chains look 0.11 nats better. With 30 restarts it reaches
0.498, and the difference disappears. The table uses 30 restarts throughout.

## What it shows

1. **A chain with $K$ outcomes per step scores like a flat mixture with $K$ components.** For
   every $K$, the chain's gap is within 0.006 of the flat $K$-mixture's at $T = 5$. At $T = 10$ it
   is within 0.005 everywhere except the ring with $K = 2$ and $K = 3$, which are 0.019 and 0.013
   better. The $K^T$ paths buy almost nothing.
2. **More steps help a little, and only with few outcomes.** At $T = 20$ the chain is up to 0.10
   nats better than the flat $K$-mixture on the ring with $K = 2$, and 0.04 better with $K = 3$.
   That is still far from what a few more components give.
3. **Per parameter, the chain loses by orders of magnitude.** The best chain is 0.31 to 0.34 nats
   from the truth, with 300 parameters on moons and 1 185 on the ring. A flat mixture with 16 components gets within 0.02 with
   95 parameters.
4. **The explanation is the objective, not the class** (found after this run; Propositions 9
   and 10 of the design note, checked in `examples/chain_objective.md`). Chains of 2-outcome
   kernels contain $2^L$-component mixtures (Proposition 10). But with a fixed forward process the
   evidence bound decomposes into per-step KLs to the data's own reverse conditionals
   (Proposition 9). The per-step fits used here are therefore the bound's optimum, and they must
   follow the data's noised marginals with $K$ linear pieces per step. The mean of a $K$-outcome
   kernel has the form of a $K$-component mixture's score, $\sum_k \gamma_k(z)\, \Sigma_k^{-1}(\mu_k - z)$, so
   the best such projection behaves like the reverse of a $K$-component mixture. That last step is
   still observed rather than proved.
5. **Together with the small-step test and E3: under the diffusion objective, depth in time does
   not substitute for outcomes.** Composition pays when the components have a product arrangement,
   the wiring adds, and the inference side is learned (E3).

## Caveats

- The chain's score is a lower bound. With $T = 20$ it is visibly loose, so the chains could be
  somewhat better than shown. At $T = 5$ the bound is tight, and the chains still match the flat
  $K$-mixture there.
- The step kernels are fitted separately. A shared, time-conditioned parameterisation would have far
  fewer parameters, but by the explanation above it would face the same capacity limit.
- One seed for each setting; 120 Adam steps per kernel.
