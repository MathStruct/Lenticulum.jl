# The objective, not the class: Propositions 9 and 10

Run `julia -t auto --project=. examples/chain_objective.jl` (about 23 minutes on 8 threads); the
test is `test/chain_objective.jl`. This checks Propositions 9 and 10 of the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§8).

## Why

In E4(b), on two moons and a ring, a chain with $K$ outcomes per step scored like a flat
$K$-component mixture, at every depth tried ($T \le 20$). The obvious explanation, that such chains can represent
only about $K$ components, is false.

- **Proposition 10:** a chain of $L$ two-outcome kernels with constant gates, identity experts
  and shifts $k\,\delta_l$ *is* the additive model of E3. It has $2^L$ components.
- **Proposition 9:** with a fixed forward process, the evidence bound decomposes exactly into
  per-step KLs to the data's own reverse conditionals. So the bound-trained chain must follow the
  data's noised marginals step by step, and with $K$ linear pieces per step it cannot.

## Setup

The data are E3's 16-mode product grid (`grid4`), centred and scaled to unit average variance so
that the variance-preserving forward process of E4 suits it. There are 4 000 training points and
1 000 held-out points.

1. **The additive model, fitted by exact EM** (256 starts, `CompositionE3.em_additive`). It is then
   rewritten as a 4-step chain of 2-outcome kernels with constant gates (`additive_chain`,
   $\varepsilon = 10^{-4}$). The chain's marginal is computed exactly by enumerating its 16 paths
   (`constant_gate_logpdf`) and compared with the additive density.
2. **Chains trained by the bound,** as in E4(b): per-step conditional maximum likelihood,
   $T \in \{5, 10, 20\}$, $K \in \{2, 4\}$, scored by the importance-weighted bound with 64 paths.
   They are compared with flat mixtures fitted by EM.
3. **The objective control** (`objective_control`, added after the second round of reviews, §13 of
   the note). Both objectives work in one family, chains whose gates do not depend on $z_t$:
   - exact marginal likelihood: the additive model by EM, written as a chain, as in 1;
   - the bound with the fixed forward process: by Proposition 9 its optimum is a separate
     conditional maximum-likelihood fit for each step, here EM for a mixture of $K$ linear
     regressions with constant weights (`fit_constant_gate_step`, 8 starts), for
     $(T, K) \in \{(4, 2), (10, 2), (5, 4)\}$.

   That family contains the additive chain, and its marginals are finite Gaussian mixtures, so every
   chain is scored by its exact held-out likelihood (path enumeration), not only by the bound. This
   is the control ChatGPT and Mistral asked for: same family, same scoring, only the objective
   differs.

## Measured (2026-10-31)

The truth scores −1.547. Gaps are in nats; smaller is better.

| model | parameters | gap to the truth |
|---|---|---|
| **additive by EM = chain with $T = 4$, $K = 2$, constant gates** | **17** | **0.001** (exact) |
| flat, $K = 2$ | 11 | 0.976 |
| flat, $K = 4$ | 23 | 0.778 |
| flat, $K = 16$ | 95 | 0.043 |
| bound-trained chain, $K = 2$, $T = 5 / 10 / 20$ | 125 / 245 / 485 | 1.087 / 1.080 / 1.035 |
| bound-trained chain, $K = 4$, $T = 5 / 10 / 20$ | 275 / 545 / 1 085 | 0.780 / 0.786 / 0.780 |

The chain's marginal by path enumeration equals the additive density to $7.1 \cdot 10^{-15}$ on every
test point.

The objective control (2026-11-01), chains with constant gates, all scored exactly:

| objective | $T$ | $K$ | exact held-out gap | IW bound gap ($S = 64$) |
|---|---|---|---|---|
| exact marginal likelihood (EM) | 4 | 2 | **0.001** | — |
| bound, fixed forward process | 4 | 2 | 1.124 | 1.126 |
| bound, fixed forward process | 10 | 2 | 1.158 | 1.160 |
| bound, fixed forward process | 5 | 4 | 0.933 | 0.939 |

## What it shows

1. **The class is not the limit.** A chain of four 2-outcome kernels represents the 16-mode grid to
   0.001 nats with 17 parameters (Proposition 10, checked to rounding).
2. **The objective is.** Trained by the bound, chains of the same kind, and much deeper ones,
   score like flat mixtures with as many components as they have outcomes per step: with $K = 4$
   they are within 0.008 of the flat 4-component mixture at every $T$, and with $K = 2$ they are
   0.06 to 0.11 below the flat 2-component one (the bound may be loose there). This is
   Proposition 9 in action. The optimum projects each true reverse conditional onto $K$-outcome
   kernels, and for this data the true conditional at small noise needs 16 pieces.
3. **The control isolates the objective.** In one family, with exact scores, the bound's optimum is
   0.93 to 1.16 nats from the truth, while exact likelihood finds a member of the same family 0.001
   away. The bound does not fail to find a good chain because of the scoring or the family: it
   prefers a worse one. (Its exact gaps are also *worse* than the gated chains' bounds
   above, since constant gates cannot follow $z_t$.) For these chains the IW bound with 64 paths is
   within 0.006 of the exact value.
4. **So a fixed forward process is the wrong objective for compositional chains.** The additive
   chain does not follow the forward process's noised marginals. It reaches the data by another
   route, which a fixed forward process cannot reward. Exact marginal likelihood, by EM over the
   paths as in E3, does train it. That makes it a hierarchical latent-variable model rather than a
   diffusion. Whether a *learned* forward process (a learned inference side) would also find it is
   not tested. The first version said it needs one; the measurement only shows that this fixed
   one is not enough (Mistral, second round).

## Caveats

- The gated chains of setup item 2 are scored by a lower bound. Their marginals are not finite
  mixtures, so they cannot be enumerated. The constant-gate chains of item 3 are scored exactly and
  show the same pinning.
- One data set and one seed. The constant-gate step fits use 8 EM starts each. The grid is the
  additive model's best case (E3 shows where it fails).
