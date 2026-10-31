# The objective, not the class: Propositions 9 and 10

Run `julia -t auto --project=. examples/chain_objective.jl` (about 18 minutes on 8 threads); the
test is `test/chain_objective.jl`. This checks Propositions 9 and 10 of the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§8).

## Why

In E4(b), on two moons and a ring, a chain with $K$ outcomes per step scored like a flat
$K$-component mixture, whatever its depth. The obvious explanation, that such chains can represent
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

## Measured (2026-10-31)

The truth scores −1.547. Gaps are in nats; smaller is better.

| model | parameters | gap to the truth |
|---|---|---|
| **additive by EM = chain with $T = 4$, $K = 2$, constant gates** | **17** | **0.001** (exact) |
| flat, $K = 2$ | 11 | 0.976 |
| flat, $K = 4$ | 23 | 0.778 |
| flat, $K = 16$ | 95 | 0.043 |
| bound-trained chain, $K = 2$, $T = 5 / 10 / 20$ | 150 / 295 / 585 | 1.087 / 1.080 / 1.035 |
| bound-trained chain, $K = 4$, $T = 5 / 10 / 20$ | 300 / 595 / 1 185 | 0.780 / 0.786 / 0.780 |

The chain's marginal by path enumeration equals the additive density to $7.1 \cdot 10^{-15}$ on every
test point.

## What it shows

1. **The class is not the limit.** A chain of four 2-outcome kernels represents the 16-mode grid to
   0.001 nats with 17 parameters (Proposition 10, checked to rounding).
2. **The objective is.** Trained by the bound, chains of the same kind, and much deeper ones,
   score like flat mixtures with as many components as they have outcomes per step: with $K = 4$
   they are within 0.008 of the flat 4-component mixture at every $T$, and with $K = 2$ they are
   0.06 to 0.11 below the flat 2-component one (the bound may be loose there). This is
   Proposition 9 in action. The optimum projects each true reverse conditional onto $K$-outcome
   kernels, and for this data the true conditional at small noise needs 16 pieces.
3. **So the switching chain is a good generative model only with a learned inference side.** The
   additive chain does not follow the forward process's noised marginals. It reaches the data by
   another route, which a fixed forward process cannot reward. Fitting its configuration posterior
   exactly (EM, as in E3) is what trains it. That is a hierarchical latent-variable model, a
   discrete-switch relative of a VAE, rather than a diffusion.

## Caveats

- The chains are scored by a lower bound, which is loose for deep chains (E4(b)). The pinning
  is also visible at $T = 5$, where the bound is tighter.
- One data set and one seed. The grid is the additive model's best case (E3 shows where it fails).
