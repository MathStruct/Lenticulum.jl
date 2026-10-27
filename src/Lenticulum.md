#implementation

> The public Lenticulum module loads Gaussian, constraint, nonlinear and likelihood
> factors, along with their belief utilities and optional covariance-learning API.

> Sources: code: `Lenticulum.jl`, `gaussian.jl`, `noise.jl`, `test/learnable_noise.jl`.
>
> Theory (CT-ML wiki): [Statistical Games](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Games).

`noise.jl` loads before `gaussian.jl` so the Gaussian constructors can share covariance
parameterisation and prior-density helpers. The module exports `noise_covariance`,
`WishartPrecisionPrior` and `LogNormalVariancePrior`; the spectral AD rules load only
when ForwardDiff or ChainRulesCore is present. Their theory and limits are recorded
in [[noise]] and [[gaussian]]. Existing factor exports retain their meanings.

## Exact composition (2026-10-27, LP-009)

The module now loads `composition.jl` after the Gaussian implementation. It extends the
core ComposedFactor for Gaussian pairs, including exact collapse, messages, games and
pushforwards, and declares the supported exact tensor-game components. See [[composition]]
and `test/game_composition.jl`. Other composition laws are explicitly unsupported.
