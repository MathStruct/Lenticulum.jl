# E4(a): a diffusion as a chain of directed switching kernels

Run `julia -t auto --project=. examples/switching_reverse_chain_e4a.jl` (about 9 minutes on 8
threads); the test is `test/switching_reverse_chain_e4a.jl`. Experiment E4(a) of the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§8, §10).

## Setup

Data: a three-component Gaussian mixture in two dimensions (weights 0.5, 0.3, 0.2; elongated,
rotated covariances); 4 000 training and 1 000 held-out points. Forward process: VP, $T$ steps
$z_t = \sqrt{1-\beta}\, z_{t-1} + \sqrt\beta\, \varepsilon$. Reverse kernels are **directed** switching
kernels (the note's §8: the normalised reading, not a product of potentials),

$$
p(z_{t-1} \mid z_t) = \sum_k g_k(z_t)\, \mathcal N(z_{t-1};\, G_k z_t + h_k,\, C_k),
\qquad g_k(z_t) \propto w_k\, \mathcal N(z_t; m_k, S_k).
$$

Models:

- **exact**: Proposition 8's kernels and the exact $q(z_T)$;
- **shared, time-conditioned**: $\pi_k, \mu_k, \Sigma_k$ learned once (17 parameters) by the per-step
  denoising objective (the chain's evidence bound over forward paths; Adam with ForwardDiff),
  every step's kernel derived from the schedule by Proposition 8;
- **free per step**: each step's kernel on its own, a $K_t$-component Gaussian mixture on the pairs
  $(z_{t-1}, z_t)$ fitted by EM and then conditioned (the joint maximum-likelihood fit, not the
  conditional one), $K_t \in \{3, 2, 1\}$; prior at $T$ one fitted Gaussian;
- **one-shot**: a three-component mixture fitted to $z_0$ by EM.

Scores: the held-out evidence bound per path (a lower bound on the model's $\log p(z_0)$; the same
forward draws for every model), and the energy distance between 1 500 generated samples and the
held-out data (against the distance between two data samples).

## Measured (2026-10-31)

$T = 20$, $\beta = 0.2$ ($\alpha_T^2 = 0.0115$):

| model | parameters | held-out bound | energy distance |
|---|---|---|---|
| truth, $\log q_0(z_0)$ | — | −2.108 | data vs data 0.0063 |
| exact kernels | — | $= \log q_0(z_0)$ on every path, max error $1.6\cdot10^{-14}$ | 0.0045 |
| one-shot mixture (EM) | 17 | −2.108 (exact log-likelihood) | — |
| **shared, time-conditioned** | **17** | **−2.123** | 0.0117 |
| free per step, $K = 3$ | 880 | −2.201 | 0.0042 |
| free per step, $K = 2$ | 580 | −2.544 | 0.0334 |
| free per step, $K = 1$ (linear-Gaussian) | 280 | −3.581 | 0.1112 |

The shared fit recovers the data mixture: $\max|\hat\pi - \pi| = 0.024$, $\max\lVert\hat\mu - \mu\rVert = 0.064$,
$\max\lVert\hat\Sigma - \Sigma\rVert = 0.042$.

$T = 100$, $\beta = 0.0436$ (the same $\alpha_T$), free per step only:

| model | parameters | held-out bound | energy distance |
|---|---|---|---|
| exact kernels | — | max error $8.2\cdot10^{-14}$ | 0.0074 |
| free per step, $K = 2$ | 2 900 | −2.842 | 0.0194 |
| free per step, $K = 1$ | 1 400 | −3.689 | 0.1146 |

## What it shows

1. **Proposition 8 holds exactly**: the exact reverse kernels make the chain's joint equal the
   forward process's, so every path's bound is $\log q_0(z_0)$ to rounding.
2. **The shared, time-conditioned parameterisation works**: 17 parameters, trained only by the
   per-step denoising objective, come within 0.015 nats of the truth's bound and recover the
   mixture. Free per-step kernels need 50 times the parameters for a worse bound.
3. **Linear-Gaussian steps fail at any step size** (Proposition 2: the chain stays Gaussian).
4. **The "few outcomes per small step" hypothesis is not established.** With two outcomes per
   step for three-component data, smaller steps improve the samples (energy distance 0.033 → 0.019,
   against a floor of 0.006) but not to the floor, and the bound gets worse (−2.54 → −2.84): with
   free per-step kernels the per-step errors accumulate over more steps. Weak, mixed evidence; a
   fair test needs a conditional (not joint) fit per step and the shared time-conditioned form
   with fewer outcomes than components.

## Caveats

The free per-step kernels are fitted as joint mixtures on pairs, which is not the conditional
maximum likelihood the evidence bound rewards; the energy distance has sampling noise of the
order of its floor (the exact kernels score 0.0045 and 0.0074 in the two runs); one data set.
