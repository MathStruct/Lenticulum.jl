# The small-step test: few outcomes per step

Run `julia -t auto --project=. examples/switching_small_steps.jl` (about 57 minutes on 8 threads);
the test is `test/switching_small_steps.jl`. It is the fair version of the question E4(a) left open
in the design note [`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md)
(§8, §10): can $T$ small switching steps with few outcomes each replace a mixture with many
components?

## Prediction

As $\beta \to 0$ each true reverse conditional $q(z_{t-1} \mid z_t)$ becomes unimodal, so a step's
outcomes are needed only to represent its mean, which is the score field $\nabla \log q_t$ written as a
softmax-gated piecewise-linear map. For a $K$-component mixture that map has $K$ pieces
(Proposition 8). So, stated before the run:

- **$K = 3$ outcomes per step**, for three-component data: the bound should approach the truth as $T$ grows.
- **$K = 2$**: the bound should level off below the truth at any step size.

## Setup

These choices separate capacity from the noise that E4(a)'s free per-step fits mixed in:

- **Data:** the three-component mixture of E4(a).
- **Fresh pairs:** every step is fitted on its own 6 000 pairs $(z_{t-1}, z_t)$, drawn from the true
  mixture and the forward process.
- **Conditional fit:** each step's directed switching kernel is fitted by conditional maximum
  likelihood of $z_{t-1}$ given $z_t$. It starts from a joint EM fit on the pairs and is refined by
  120 Adam steps with ForwardDiff.
- **Exact prior:** the prior at $T$ is the exact $q(z_T)$, so only the kernels differ.
- **Same endpoint:** $T \in \{10, 25, 50, 100\}$, all with the same $\alpha_T$.

The scores are the same as in E4(a): the held-out evidence bound per path (1 000 test points),
and the energy distance of 1 500 generated samples.

## Measured (2026-10-31)

The truth's held-out $\log q_0(z_0)$ is −2.108. The data-vs-data energy distance (the floor) is 0.0063.

| $T$ | $K$ | held-out bound | gap to truth | estimation cost $T d / 2n$ | gap − cost | energy distance |
|---|---|---|---|---|---|---|
| 10 | 2 | −2.608 | 0.500 | — | — | 0.0414 |
| 25 | 2 | −2.630 | 0.522 | — | — | 0.0180 |
| 50 | 2 | −2.668 | 0.560 | — | — | 0.0205 |
| 100 | 2 | −2.786 | 0.678 | — | — | 0.0201 |
| 10 | 3 | −2.156 | 0.048 | 0.033 | 0.015 | 0.0088 |
| 25 | 3 | −2.212 | 0.104 | 0.081 | 0.023 | 0.0036 |
| 50 | 3 | −2.294 | 0.186 | 0.163 | 0.023 | 0.0034 |
| 100 | 3 | −2.568 | 0.460 | 0.325 | 0.135 | 0.0049 |

The gap is the **bound's** gap. With the exact prior, Proposition 9 makes it exactly the sum of the
$T$ per-step conditional KLs. The estimation cost is the asymptotic loss of maximum likelihood: about
$d/(2n)$ nats per step, with $n = 6000$ pairs and $d = 15K - 6$ identified parameters. That is 9 per
affine expert and 6 per gate log-odds against a reference outcome: a quadratic common to all gates
cancels in the softmax.

The run printed the cost with the first version's count, $15K - 1$. The second round of reviews
corrected it (design note §13). The two cost columns above are arithmetic with the corrected count,
which the script now uses.

The formula is the regular, well-specified asymptotic one:
- it is exact as $n \to \infty$ for $K = 3$, whose true reverse kernel is in the family (Proposition 8);
- it degrades where the gates become nearly flat, at large $t$, where the Fisher information
  degenerates (mixtures are not regular models; Chen 1995);
- it is not the right first-order form for the misspecified $K = 2$, so no cost is shown there. The conditional refit improves on the joint EM start in every case
(total gain 0.010 to 0.148).

## What it shows

1. **The bound does not approach the truth as $T$ grows, and most of the growth is estimation.**
   With $K = 3$ the gap grows with $T$, and most of it is the expected cost of fitting $T$ separate
   kernels on $n$ pairs each. What remains after subtracting it is 0.015, 0.023 and 0.023 for $T$ = 10,
   25 and 50. At $T = 100$ the remainder is 0.135. The refit gain is largest there (0.148), so the 120
   Adam steps may not have converged. The regular cost formula is also least reliable there. The two
   explanations are confounded. A tied (shared) parameterisation, whose cost does not grow with $T$,
   would separate them; Mistral's review suggested it.
2. **Two outcomes per step have a capacity gap that small steps do not close.** The raw gap for
   $K = 2$ is 0.50 to 0.68 nats and does not shrink with $T$. Even the full cost estimate would
   explain at most 0.24 of it at $T = 100$. The capacity conclusion rests on the raw gaps, on this one
   three-component data set.
3. **The samples tell the same story.** For $T \ge 25$, $K = 3$ is at the floor (0.0034 to 0.0049
   against 0.0063). $K = 2$ levels off at about 0.02, three times the floor.
4. **So the prediction holds on capacity:** as many outcomes as the score field has linear pieces are
   enough; fewer are not, whatever the step size. The diffusion hypothesis of §8 in its strong form,
   that small steps let *fewer* outcomes than components suffice, is **refuted** on this data.
5. **Free per-step kernels pay for every step.** The estimation cost is linear in $T$, so a chain
   of free kernels gets worse with smaller steps. This is why E4(a)'s shared, time-conditioned chain
   (17 parameters, whose cost does not grow with $T$) did so much better: it is the only chain here
   whose bound has a reason to improve as steps shrink.

## Caveats

- One data set, one seed per step.
- The estimation cost is an asymptotic estimate, not a measured quantity. Its first version
  overcounted the parameters by 5 per kernel.
- The energy distance has sampling noise of the order of its floor.
- The $T = 100$, $K = 3$ fits are probably under-optimised.
- The run allocated heavily (47 % of its time in garbage collection): the per-step objective is
  evaluated through ForwardDiff on all 6 000 pairs.
