# Hierarchical models

Run `julia --project=. examples/hierarchical_models.jl` from the repository root. The
explanation is the vault note
[`Hierarchical Models`](../vault/Factor%20Graphs/Hierarchical%20Models.md); the script is
included by `test/hierarchical_models.jl` without running `main()`.

Three linear-Gaussian hierarchies, built from `GaussianPrior`, `GaussianFactor`,
`LinearConstraintFactor` and `DataFactor`. No package source or API changes were needed. Every
result is compared with a dense Gaussian conditioned by plain linear algebra
(`schools_dense`, `slopes_dense`, `crossed_dense`), which shares no code with Lenticulum.

| example | graph | schedule | checked |
|---|---|---|---|
| eight schools (Rubin 1981), random intercepts | tree (a star around μ) | `tree_schedule` | means, variances, −log p(y∣τ); a ninth school without data |
| random slopes, 5 groups × 6 points | tree | `tree_schedule` | means, 2×2 covariances, −log p(y) |
| crossed effects, 4 rows × 3 columns | 6 independent loops | `flooding_schedule` | means exact; variances not |

## Design

- **The measurement is two pieces.** As in the localisation tutorial, each observation is a
  noise factor to an auxiliary variable plus a `DataFactor`, so removing the reading turns
  the same graph into a predictor.
- **Random slopes put the weights on the graph.** The design matrix $X_j$ is the parameter
  `A` of the observation `GaussianFactor`, and the group's weights $\beta_j$ are its *input
  channel*. A regression whose weights were `ps` could not take a population factor; this one
  can, because nothing about the hierarchy touches a factor's parameters.
- **Crossed effects use the n-ary constraint.** Each observation is
  `LinearConstraintFactor((a = 1, b = 1), 1)` with the reading as its constant $c$: the residual
  $a_r + b_c - y_{rc}$ joins two variables without an auxiliary one.
- **τ is a parameter, not a variable.** The population variance is the noise of the
  population factor, so the example scans it: the free energy at each τ is
  $-\log p(y \mid \tau)$, and a uniform prior on a grid turns the scan into weights for a mixture
  over τ, computed outside the graph. Making τ a variable would need a non-Gaussian
  (Gamma or inverse-Gamma) belief, which LenticulumCore does not have.

## Measured (2026-10-29)

- Eight schools: graph vs dense within $1.2\cdot10^{-13}$ (means, variances, free energy) at
  τ ∈ {0.5, 5, 10, 25}, and within $6.4\cdot10^{-14}$ in −log p(y∣τ) over 160 grid points.
- The marginal likelihood is largest at the grid's lower end (τ = 0.25): type-II maximum
  likelihood pools the schools completely. With a uniform prior on τ ∈ [0.25, 40], E[τ∣y] = 6.61,
  and school A's effect is 11.41 ± 8.34 (y = 28).
- Random slopes: graph vs dense within $2.2\cdot10^{-10}$.
- Crossed effects: converged; means within $1.3\cdot10^{-13}$ of the exact posterior; loopy
  variances are 0.599 to 0.646 of the exact ones (overconfident).

## Difficulties

- A ninth school without data makes its population factor send a trivial message towards μ;
  Mycelium warns about it (`no informative inputs`), which is correct and harmless here.
- `τ = 0` is not allowed (the noise covariance must be positive definite); the scan starts at
  0.25 and the test of complete pooling uses τ = 10⁻³.
- μ's prior is N(0, 100²), not flat, so that the dense reference and the graph have the same
  proper prior; the pooled mean in the test includes its precision $10^{-4}$.
