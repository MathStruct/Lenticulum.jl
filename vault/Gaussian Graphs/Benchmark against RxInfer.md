#benchmark #comparison

> Lenticulum against RxInfer, and both against exact references, on two models: a Kalman
> smoother and a mixture of two linear regressions. On the smoother both are exact to $10^{-10}$
> and agree with each other to about $10^{-10}$. Lenticulum is within a factor 1.7 of RxInfer up
> to $T = 100$ and 4.3–5.8 times slower at $T = 1000$–$2000$, because its cost grows faster than
> linearly in the size of the graph. On the mixture, Lenticulum's `fit_em` is exactly
> maximum-likelihood EM; RxInfer's variational estimates are slightly worse in likelihood but 9
> to 10 times faster. Hand-written closed-form EM is about 2 000 and 4 900 times faster than
> Lenticulum and about 220 and 490 times faster than RxInfer.

> Sources: Bagaev, Podusenko & de Vries, *RxInfer: A Julia package for reactive real-time Bayesian inference*, JOSS 2023 (the library compared, version 6.1.3). code: `bench/rxinfer_baseline.jl`.
>
> Bibliography: [[Bibliography#^bagaev2023rxinfer|Bagaev et al. 2023]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Setup

Run `julia -t auto --project=bench bench/rxinfer_baseline.jl` (`BENCH_PART=ssm` or `mix` for one part).
Measured 2026-11-11 on one machine (8 threads, `-t auto`; Julia 1.13.1, RxInfer 6.1.3), both parts
in one run; times are warm (after one run), best of five. The script checks, and stops otherwise,
that `infer!` converged and that `fit_em`'s free energy at its estimate is the score below.

## 1. A linear-Gaussian state-space model

A constant-velocity state in $\mathbb R^2$, position-only measurements, $T$ steps
(`Xoshiro(T)` data). Both libraries run belief propagation on a tree, which is exact, so their
free energies must equal $-\log p(y)$ and their posteriors the conditional marginals. The
references are a Kalman filter's prediction-error decomposition in 256-bit arithmetic for
$-\log p(y)$, and the dense joint solve in information form for the marginals.

| $T$ | $\lvert F_L - F^\ast\rvert$ | $\lvert F_R - F^\ast\rvert$ | $\lvert F_L - F_R\rvert$ | mean error L / R | L build | L infer | L free energy | L total | R | R without $F$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 3.6e-15 | 5.7e-14 | 5.3e-14 | 8.9e-15 / 3.5e-13 | 0.2 | 0.7 | 0.4 | 1.3 | 1.0 | 0.7 |
| 100 | 1.7e-13 | 2.8e-13 | 1.1e-13 | 9.9e-14 / 3.3e-13 | 2.6 | 7.8 | 4.6 | 15.3 | 9.2 | 6.7 |
| 1000 | 5.8e-11 | 3.5e-11 | 2.3e-11 | 6.4e-12 / 4.1e-12 | 127.5 | 164.6 | 167.6 | 458.3 | 106.9 | 76.2 |
| 2000 | 8.8e-11 | 3.2e-11 | 1.2e-10 | 7.3e-12 / 6.4e-12 | 470.6 | 493.7 | 599.9 | 1583.0 | 273.9 | 188.1 |

Times in ms. L build: the graph and its parameters; L infer: `tree_schedule` and `infer!`; L free
energy: `scalar_free_energy`. R: RxInfer's `infer` with `free_energy = true`, model creation
included. Errors cover every marginal, $x_0$ to $x_T$. The largest covariance error is $5 \cdot 10^{-16}$–$9 \cdot 10^{-16}$ for Lenticulum and
$1.8 \cdot 10^{-14}$ for RxInfer at every $T$.

**What it shows.**

1. **Both are exact**, to $10^{-10}$ in $-\log p(y)$ and $10^{-11}$ in the means, and agree with
   each other to $1.2 \cdot 10^{-10}$.
2. **Lenticulum's cost grows faster than linearly.** From $T = 1000$ to $2000$ the graph build
   takes 3.7 times as long, the free energy 3.6 times and `infer!` 3.0 times; RxInfer takes 2.6
   (with $F$) and 2.5 times. The quadratic part is the one the [[Feature Backlog]] records for
   parameter keys looked up by name (LP-024: the free energy of 601, 1 201 and 2 401 factors in
   7.7, 25.8 and 102 ms). Up to $T = 100$ the two are within a factor 1.7.
3. **RxInfer needs an option beyond $T = 1000$.** At $T = 2000$ its default run overflowed the
   stack; its documented remedy, `limit_stack_depth = 100`, is used there.

**Getting the reference right.** Two dense formulas for $-\log p(y)$ were wrong at $T = 1000$
while the libraries agreed with each other to $2 \cdot 10^{-11}$. The covariance form was off by
$2.8 \cdot 10^{-7}$: the prior covariance of a constant-velocity state grows like $T^3$. The
information form was off by $1.2 \cdot 10^{-5}$: it is a difference of terms of order $10^3$–$10^4$.
The Kalman filter in BigFloat is exact.

## 2. A mixture of two linear regressions

300 or 3 000 points, $x \sim U(-3, 3)$, line $k$ with weights $(0.5, 0.5)$,
$y = a_k x + c_k + \sigma_k\varepsilon$, $a = (1, -1)$, $c = (0, 0.5)$, $\sigma = (0.2, 0.5)$ (`Xoshiro(11)`, the data of
`lib/Lenticulum.jl/examples/regime_lines.md`). One start for all: $a = (0.5, -0.5)$, $c = 0$,
$\sigma = 1$, $w = (0.5, 0.5)$.

- **EM (hand)**: closed-form EM for a mixture of regressions, the reference.
- **Lenticulum**: one `RegimeFactor` per point with tied parameters, `fit_em` (the closed-form
  M-steps of `src/m_steps.jl`), as in `test/em.jl`. Graph building not timed.
- **RxInfer**: priors $\theta_k \sim \mathcal N(0, 100 I)$, $\tau_k \sim \Gamma(1, 1)$, $w \sim \operatorname{Dir}(1, 1)$, the means
  `dot(X[i], θk)`, a `NormalMixture` likelihood, mean-field VMP over all variables. Point
  estimates are posterior means, $\sigma_k = 1/\sqrt{\mathbb E\tau_k}$. Run until its free energy changes by less than
  $10^{-8}$ per iteration.

The estimators differ: EM gives the maximum-likelihood parameters, VMP a variational posterior
under priors. The common score is $-\log p(y \mid x, \hat\theta)$ at each estimate, labels matched.

| $N$ | method | $a$ | $c$ | $\sigma$ | $w$ | $-\log p(y \mid x, \hat\theta)$ | iterations | time (ms) |
|---|---|---|---|---|---|---|---|---|
| 300 | EM (hand) | (0.9972, −0.9941) | (−0.0092, 0.4784) | (0.1934, 0.5023) | (0.519, 0.481) | 253.389349 | 11 | 0.6 |
| 300 | Lenticulum | (0.9972, −0.9941) | (−0.0092, 0.4784) | (0.1934, 0.5023) | (0.519, 0.481) | 253.389349 | 11 | 1 196.3 |
| 300 | RxInfer | (0.9970, −0.9940) | (−0.0092, 0.4805) | (0.2284, 0.5162) | (0.522, 0.478) | 256.859526 | 32 | 132.6 |
| 3000 | EM (hand) | (1.0003, −1.0064) | (0.0021, 0.4979) | (0.1972, 0.4915) | (0.500, 0.500) | 2 603.684235 | 11 | 6.5 |
| 3000 | Lenticulum | (1.0003, −1.0064) | (0.0021, 0.4979) | (0.1972, 0.4915) | (0.500, 0.500) | 2 603.684235 | 11 | 31 635.8 |
| 3000 | RxInfer | (1.0003, −1.0065) | (0.0022, 0.4978) | (0.2009, 0.4927) | (0.501, 0.499) | 2 604.168160 | 40 | 3 197.4 |

Lenticulum's time excludes building its graph, 2.7 and 28.0 ms; RxInfer's includes creating its
model. RxInfer's own free energy at convergence is 305.24 and 2 666.56: a bound under its priors,
not comparable with the column above.

**What it shows.**

1. **Lenticulum's EM is EM**: the same iterates and the same maximum to the printed digits, as
   `test/em.jl` asserts step for step.
2. **It is slow.** 1.2 s against 0.6 ms for the closed form at $N = 300$, 32 s against 6.5 ms at
   $N = 3000$, and 9.0 and 9.9 times RxInfer. Ten times the data costs 26 times the time. The regime
   factors run point by point, and per-outcome forms are not batched over a plate (the row
   "batched cluster nodes" of the [[Feature Backlog]]); the cost of parameter keys above comes on
   top.
3. **RxInfer's estimates are close but not maximum likelihood.** At $N = 300$ its $\hat\sigma_1$ is 0.228
   against 0.193, which costs 3.5 nats in $-\log p(y \mid x, \hat\theta)$. At $N = 3000$ it is 0.48 nats.
   Partly that is the estimator: $1/\sqrt{\mathbb E\tau}$ is not the maximiser, and the posterior is
   mean-field. Which part is which was not separated.

## Caveats

- One machine, one data set per size, one start. Both EMs kept the start's label order;
  RxInfer's came out swapped and were matched by the slopes.
- RxInfer was run as its documentation describes, not tuned: no custom schedules,
  `limit_stack_depth` only where the default failed. Lenticulum's graph-building time is excluded
  in part 2 (reported beside it) and included in part 1's total.
- Not measured: compilation time (first run), memory, streaming use, graphs with loops.

## Review (2026-11-11)

Mistral (`vibe`) reviewed the script and this note before the first commit, given the
requirement and the diff. Its ten points:

| # | point | right? | done |
|---|---|---|---|
| 1 | `rx_converged` counts one iteration too many | no: a test on a constructed trace returns the first iteration whose change is below the tolerance (`findfirst` returns a position in `2:length(F)`) | nothing |
| 2 | part 2 times RxInfer's model creation but not Lenticulum's graph build | yes | the build is timed and reported (2.7 and 28 ms) |
| 3, 4 | "9–10 times" and "210–420 times" did not cover 10.3 and 424 | yes | the ranges, from the final run |
| 5 | "agree as closely" as $10^{-10}$ while $\lvert F_L - F_R\rvert$ is $1.2 \cdot 10^{-10}$ | yes | stated as measured |
| 6 | the run command did not request the 8 threads used | yes | `-t auto` in the command |
| 7 | RxInfer's $x_0$ marginal was not compared | yes | compared |
| 8 | `infer!`'s convergence report was discarded | yes | checked |
| 9 | only RxInfer's labels were matched | yes | Lenticulum's too |
| 10 | `fit_em`'s free energy was not checked against the score | yes | checked |

Points 2–10 changed no result but the timing description and the ranges.
