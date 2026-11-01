#implementation

> Passing points instead of beliefs. A Rao-Blackwellised Gibbs sampler draws chosen variables
> (typically discrete switches) from their exact conditionals, integrating the rest out by belief
> propagation with the drawn ones pinned. A sample buffer keeps the samples, weighs them by a
> filter, forgets old ones, and averages energies, residual moments and Monte Carlo EM gradients
> without losing the structure of a multivariate energy.

> Sources: Wei & Tanner, *A Monte Carlo Implementation of the EM Algorithm and the Poor Man's Data Augmentation Algorithms*, JASA 85(411), 1990 (Monte Carlo EM); Delyon, Lavielle & Moulines, *Convergence of a Stochastic Approximation Version of the EM Algorithm*, Ann. Statist. 27(1), 1999 (the decaying filter). code: `sampling.jl`, `messages.jl` (pins), `free_energy.jl`; tests: `test/sampling.jl`, Lenticulum's `test/switching_constraint.jl` and `test/lp020.jl`
>
> Bibliography: [[Bibliography#^wei1990montecarlo|Wei & Tanner 1990]] · [[Bibliography#^delyon1999saem|Delyon et al. 1999]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

Implements §6b of [[Extending Mycelium]].

## Pins

`pin!(store, g, :v, value)` makes variable `v` observed for the time being: its marginal, and every
message it sends to a factor, is `DiracBelief(value)`, whatever the factors send it. Pins live in
the `MessageStore`, so a sampler moves them between updates without rebuilding the graph. `reset!`
clears messages and keeps pins. Only `marginal` and `excluded_marginal` read them, and every reader
of beliefs goes through those two (variable → factor messages, the Bethe variable terms, the
read-out). The free-energy functions see a pinned variable as clamped data.

## The sampler

`GibbsSampler(g, sampled)` names the variables to draw. One update of variable $v$:

1. pin every other sampled variable at its current point and release $v$;
2. clear the messages and run the schedule (a tree schedule by default);
3. draw $v$ from its marginal (`sample_belief`: Dirac, categorical, proper Gaussian, mixture).

The marginal of $v$ is its exact conditional given the pins whenever the graph without the pinned
variables is a tree with closed-form messages. In a conditional-Gaussian graph with its discrete
variables sampled, that is the case, and the continuous variables are integrated out exactly,
never sampled (Rao-Blackwellisation). Blocks are sampled jointly by making them one variable, e.g.
one switching factor with $2^B$ outcomes for $B$ overlapping switches. That is the mixing remedy for
switches that explain each other away, the same block structure that fixes the pair-moment bias
of factorised beliefs.

`gibbs_init` draws the first points in order, each from its marginal with the earlier ones pinned.
For the later ones that marginal is only approximate; it starts the chain and nothing more. Each
update reruns the whole schedule, which is simple and costs one inference per variable.

## The buffer: kept, filtered, forgotten

A `SampleBuffer(filter; horizon)` stores, per sample:

- the pins;
- the graded local free energies of every factor and the variable corrections;
- the total (the pinned graph's scalar free energy);
- every factor's residual mean and covariance (`factor_residual_moments`, implemented by
  Lenticulum's constraint factors);
- an optional weight;
- the parameter step at which it was drawn.

The **filter** weighs the samples by age:

- `WindowFilter(n)`: the last $n$, equal weights;
- `DecayFilter(λ)`: $\lambda^{\text{age}}$, which with $\lambda$ moved towards 1 is stochastic-approximation EM.

`advance!` counts a parameter step and forgets samples older than `horizon` steps, since samples
from old parameters bias the gradient.

The queries obey the averaging requirement of [[Extending Mycelium]] §6b:

- `energy_moments`: the weighted mean and variance of every graded component, of each sample's
  *already scalarised* local energies. Nothing is averaged before its scalarisation.
- `residual_moments(buffer, factor)`: the mean and the full covariance of the factor's residual
  vector, by the law of total covariance (mean of within-sample covariances plus covariance of
  within-sample means). For a squared-norm scalarisation these are exact sufficient statistics,
  $\mathbb E\lVert r\rVert^2_W = \lVert \bar r\rVert^2_W + \operatorname{tr}(W \operatorname{Cov} r)$.
- `free_energy_mean`: the weighted mean of the totals, each $-\log p(\text{data}, \text{pins})$ on an exact
  pinned graph.
- `sample_gradient`: the weighted mean of each sample's gradient of its total at the current
  parameters.

## The gradient

`sample_gradient` pins each stored sample, reruns inference at the current parameters, and
takes the gradient of the total ($-\log p(\text{data}, \text{pins})$) at those parameters. With exact posterior weights it equals the gradient
of the exact $-\log p(\text{data})$ (Lenticulum's test, against finite differences of the enumerated
likelihood). With posterior samples it is the Monte Carlo EM gradient.

By default (since 2026-11-01) each sample's pinned inference is followed by
`free_energy_gradient`, which holds the factor beliefs fixed
([[LP-020 The fixed-message parameter gradient is not the free-energy gradient]]). That is one
inference per sample, and exact once the inference has converged, as on a pinned tree. The first
version differentiated through the inference instead: one inference per sample and, with
reference differences, per parameter coordinate too. It stays as `through_inference = true`,
needs no factor hook, and is what the default is checked against.

## Tested

- Mycelium's `test/sampling.jl`: pins (marginal, free energy, `reset!` keeps them, `unpin!`);
  `sample_belief` against its distributions; the sampler on a categorical prior; window and decay
  weights; per-component energy moments; forgetting by horizon; the errors.
- Lenticulum's `test/switching_constraint.jl`, on two overlapping switches into a sum node:
  - the sampler's frequencies against the enumerated posterior;
  - with every configuration at its exact weight, each total is $-\log p(x, k)$, the residual
    moments match Gaussian conditioning, and the gradient matches finite differences of
    $-\log p(x)$ in `c`, `b` and `logw`. The default and `through_inference = true` agree to $10^{-6}$.

## Not done

- The cost of one full inference per update. A sampler should rerun only the part of the schedule
  that a pin change affects.
- No convergence diagnostics (effective sample size, several chains) and no tempering.
- Sampling of continuous variables works through `sample_belief`, but a pinned continuous
  variable makes its neighbours' messages Dirac-driven, and nothing here checks that this mixes.
- The buffer re-runs inference per stored sample for the gradient; caching the pinned stores would
  trade memory for time.
- No `parameter_step`-style driver: the caller applies the gradient with an optimiser.
