# Two crossing lines with one noise level each, as one graph of regime factors

Run `julia -t auto --project=docs examples/regime_lines.jl` (about a minute; the docs
environment has ForwardDiff). This belongs to `src/regime.md` and
[L-SPEC-009](../vault/Project/Specifications/L-SPEC-009%20Cluster%20Nodes.md) §9.

## Question

Experiment E1 (branch `experiment-qm`) fitted two crossing lines, a mixture of regressions. Each
line is a factor $0 = y - a_k x - c_k + \varepsilon_k$ on the channels $y$ and $x$: the coefficients depend on the line,
and here the noise too ($\sigma_1 \neq \sigma_2$). `SwitchingConstraintFactor` cannot state this (`switching.md`
§5). Does `RegimeFactor` fit it on a Mycelium graph, by the free-energy gradient, to the maximum
likelihood that closed-form EM reaches?

## Setup

- **Data.** 300 points, $x \sim U(-3, 3)$, line $k$ with weights $(0.5, 0.5)$, $y = a_k x + c_k + \sigma_k \varepsilon$,
  $a = (1, -1)$, $c = (0, 0.5)$, $\sigma = (0.2, 0.5)$, `Xoshiro(11)`.
- **Graph.** Point $i$: its data on $x_i$ and $y_i$, its outcome $k_i$, one `RegimeFactor((y = 1, x = 1), 2, 1;
  learn_noise = true, output = :y)`, tied over the points. The graph is a forest and $k_i$ has degree one, so the
  Bethe free energy is exactly $-\sum_i \log p(y_i \mid x_i)$; no cluster node is needed. Two sequential
  sweeps solve it.
- **Fit.** 400 Adam steps ($\eta = 0.05$) of the free-energy gradient (ForwardDiff) in $a$, $c$, $\log w$ and $\log \sigma$
  (the parameter `L`). $y$ is the factor's output, so its coefficient is fixed at 1 and is not a
  parameter. With it learnable, scaling it and $\sigma$ together lowers $F$ without bound
  ([[LP-022 Constraint factors are normalised in their residual, not their channels]]). The first
  version, on one channel $z = (x, y)$, held that coefficient at 1 by hand.
- **Reference.** EM for a mixture of linear regressions (weighted least squares per line, the
  weighted residual variance per line) from the same start, $a = (0.5, -0.5)$, $c = 0$, $\sigma = 1$, $w = (0.5, 0.5)$,
  until $-\log p$ changes by less than $10^{-12}$. Also EM with one $\sigma$ for both lines, what a
  `SwitchingConstraintFactor` could state.

## Measured (2026-11-04)

At the start the graph's free energy equals the closed form, 555.7111638001, to the printed digits.

| fit | $a$ | $c$ | $\sigma$ | $w$ | $-\log p$ | time |
|---|---|---|---|---|---|---|
| generating parameters | (1, −1) | (0, 0.5) | (0.2, 0.5) | (0.5, 0.5) | 254.1073 | |
| graph, 400 Adam steps | (0.997, −0.994) | (−0.009, 0.478) | (0.193, 0.502) | (0.519, 0.481) | **253.3893** | 50.5 s |
| EM | (0.997, −0.994) | (−0.009, 0.478) | (0.193, 0.502) | (0.519, 0.481) | **253.3893** | 0.54 s, 12 iterations |
| EM, one $\sigma$ | (0.996, −0.995) | (−0.001, 0.487) | (0.377, 0.377) | (0.532, 0.468) | 306.8969 | |

The graph's fit and EM differ by $2.0 \cdot 10^{-8}$ in $-\log p$ and by at most $3.2 \cdot 10^{-6}$ in any parameter
(log-weights normalised; `logw` is determined up to a constant).

## What it shows

1. **Per-line coefficients and noise fit on the graph.** The free-energy gradient through the
   regime factors reaches EM's maximum likelihood, to the parameters' sixth digit.
2. **The noise per line matters here.** With one $\sigma$ (the location-shift case) the best
   $-\log p$ is 306.90, 53.5 worse; the lines are found either way, the noise levels are not.
3. **It is slow against closed form**, 50.5 s against 0.54 s: 400 gradient steps, each through
   300 one-factor clusters replica by replica (per-outcome forms are not batched over a plate).

## Caveats

- One data set, one start, well separated lines; EM's label order is the start's, and so is the
  graph's.
- Timings: 21.7 s in the first run (one channel $z$, the message to $z$ left out of the schedule),
  29.4 s with that message, 50.5 s with separate channels $x$, $y$ and two data factors per point
  (2026-11-04). The values were the same in all three.
