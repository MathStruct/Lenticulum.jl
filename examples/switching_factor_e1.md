# E1: the switching constraint factor on non-orthogonal lines

Run `julia --project=. examples/switching_factor_e1.jl`; the test is `test/switching_factor_e1.jl`.
Experiment E1 of the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§10): does the
switching constraint factor fix the failure of the spectral measurement
([`measurement_factor.md`](measurement_factor.md)) on two lines 60° apart?

## Setup

Two noisy lines through the origin at 0° and 60°, equal weights, noise σ = 0.05; 20 000 training
and 5 000 held-out points (`MeasurementFactors.lines_data`). The switching factor is the general
form of the note's §3, $\psi(k, z) = w_k\, \mathcal N(a_k^\top z; 0, q_k)$ with $a_k$ the normal of the line at
angle $\varphi_k$, and a Gaussian reference $\mathcal N(0, s^2 I)$, the graph's prior. Outcome $k$ of the joint is
$\mathcal N(0, C_k)$, $C_k^{-1} = I/s^2 + a_k a_k^\top / q_k$, weighted by $w_k \mathcal N(0; 0, q_k + s^2)$; with
$s^2 = 1 + \sigma^2$ and $q_k$ matching the across-line variance the true model is in the class. It is
fitted by maximum likelihood (exact log-likelihood, Adam with ForwardDiff gradients, 600 steps,
four starting angle pairs, a floor $q_k \ge 10^{-6}$). Six parameters: two angles, two $q_k$, one
weight, $s^2$.

The conditional $p(y \mid x)$ of the fitted factor is computed twice: by conditioning each outcome's
Gaussian (Proposition 3.1), and by message passing in a Mycelium graph (`GaussianPrior`s on $x$ and
$y$, a `SwitchingConstraintFactor` defined in the example, a `DataFactor` clamping $x$). The
factor's message is the mixture of §3, with the per-outcome constant $Z_k = \mathcal N(d_k; 0, R_k)$ and,
because Mycelium's `MixtureBelief` holds normalised components, each weight multiplied by its
component's mass $\sqrt{2\pi/\Lambda_k}\, e^{\eta_k^2 / 2\Lambda_k}$.

## Measured (2026-10-31)

| model | lines | held-out mean $\log p(y \mid x)$ | held-out mean $\log p(z)$ | parameters |
|---|---|---|---|---|
| switching factor, general form | 0.02°, 60.00° | **0.702** | **−0.459** | 6 |
| free two-component Gaussian mixture (EM, 4 restarts) | — | 0.702 | −0.459 | 11 |
| spectral measurement (the toy) | 30.03°, 120.03° | −0.939 | — | — |
| one Gaussian | — | −0.823 | — | — |
| true model | 0°, 60° | 0.702 | −0.459 | — |

The fitted $q_k$ are 0.00251 and 0.00249 (the noise variance is 0.0025), $s^2 = 1.001$ ($1 + \sigma^2$),
$w_1 = 0.498$. Graph message passing and the direct conditional agree to $5.4\cdot10^{-15}$ at four
inputs (weights, means and variances).

Two of the four starts reach the optimum (training log-likelihood −0.4625); the other two stop
at −0.4647 and −0.4649 within 600 steps. Restarts matter, as the review predicted.

## What it shows and what it does not

The failure of the spectral measurement was the eigendecomposition, not linearity: one discrete
outcome selecting between two linear constraints represents non-orthogonal lines exactly, with
fewer parameters than a free mixture and the same fit. The graph check confirms the message
formulas of §3, including the per-outcome constants whose omission was LP-019's recurring defect.

This is the easy case: the true model is in the class, and $K$ equals the number of lines. It says
nothing about composition (E3, E4), and the fit is by direct maximum likelihood, not by Bethe or EM
through the graph.
