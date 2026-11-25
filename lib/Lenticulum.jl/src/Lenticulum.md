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

## AD backend matrix (Q5)

Measured 2026-11-24 with `lib/Lenticulum.jl/test/ad_matrix.jl`.

| Entry point | Zygote | ForwardDiff | Enzyme | Mooncake |
|---|---|---|---|---|
| `energy/GaussianFactor` | pass | pass | pass | pass |
| `factor_message/GaussianFactor` | pass | pass | broken: other | broken: mutation/unsupported construct |
| `local_free_energy/GaussianFactor` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `energy/GaussianPrior` | pass | pass | pass | pass |
| `factor_message/GaussianPrior` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `local_free_energy/GaussianPrior` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `energy/NonlinearFactor` | pass | pass | pass | pass |
| `factor_message/NonlinearFactor` | broken: mutation/unsupported construct | pass | broken: other | broken: mutation/unsupported construct |
| `local_free_energy/NonlinearFactor` | broken: mutation/unsupported construct | pass | broken: other | broken: mutation/unsupported construct |
| `energy/LinearConstraintFactor` | pass | pass | broken: mutation/unsupported construct | pass |
| `factor_message/LinearConstraintFactor` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `local_free_energy/LinearConstraintFactor` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `factor_message/HardConstraintFactor` | pass | pass | broken: other | broken: mutation/unsupported construct |
| `local_free_energy/HardConstraintFactor` | pass | pass | broken: other | broken: mutation/unsupported construct |
| `factor_message/SplitFactor` | broken: mutation/unsupported construct | pass | broken: other | broken: mutation/unsupported construct |
| `local_free_energy/SplitFactor` | broken: mutation/unsupported construct | pass | broken: other | broken: mutation/unsupported construct |
| `factor_message/ScaleShiftFactor` | pass | pass | skip | broken: mutation/unsupported construct |
| `local_free_energy/ScaleShiftFactor` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `energy/SwitchingConstraintFactor` | pass | pass | broken: other | broken: mutation/unsupported construct |
| `factor_message/SwitchingConstraintFactor` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `local_free_energy/SwitchingConstraintFactor` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `factor_message/RegimeFactor` | broken: mutation/unsupported construct | pass | broken: mutation/unsupported construct | pass |
| `local_free_energy/RegimeFactor` | pass | pass | broken: mutation/unsupported construct | pass |
| `energy/LikelihoodFactor` | pass | pass | pass | pass |
| `factor_message/LikelihoodFactor` | broken: mutation/unsupported construct | pass | broken: other | broken: mutation/unsupported construct |
| `local_free_energy/LikelihoodFactor` | pass | pass | broken: other | pass |
| `free_energy after infer!/Kalman` | pass | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `moment_statistics/regime` | broken: mutation/unsupported construct | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `energy/HardConstraintFactor` | skip | skip | skip | skip |
| `energy/SplitFactor` | skip | skip | skip | skip |
| `energy/ScaleShiftFactor` | skip | skip | skip | skip |
| `energy/RegimeFactor` | skip | skip | skip | skip |

NonlinearFactor and LikelihoodFactor use continuous inputs, and SplitFactor uses an incoming belief, because those fixtures have empty parameter trees. HardConstraintFactor, SplitFactor, ScaleShiftFactor and RegimeFactor have graph energy methods and no pointwise `energy` method. The isolated Enzyme reverse probe for the ScaleShiftFactor message reached its 185-second limit.

The first default matrix took 4m52s. The default tier now covers pointwise energy and the Gaussian transition message. The revised slow matrix, including both graph cases, took 14m27s.

## Element types (Q4)

The Gaussian, relation and likelihood implementations use LenticulumCore's numeric-type
and zero-buffer helpers. These helpers inspect numeric input storage and leave discrete labels,
polarity and structural flags separate from the computation.
