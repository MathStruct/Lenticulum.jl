"""
    VariationalDiffusion

A diffusion model as a **statistical game**: the third of the three
`Implicit Learners.md` families, and the only one whose Bayesian inversion is neither exact
nor a root-find but a *proximal solve*.

## The chain

1. `schedule.jl` — the **VP-SDE** of Song et al. 2021 (arXiv:2011.13456), presented through
   its perturbation kernel ``p_{0t}(x_t\\mid x_0) = \\mathcal{N}(\\alpha_t x_0, \\sigma_t^2 I)``
   with ``\\alpha_t^2 + \\sigma_t^2 = 1``.
2. `predictor.jl` — [`NoisePredictor`](@ref) wraps any **Lux model** as
   ``\\varepsilon_\\theta(x,t)`` and derives from it the score ``-\\varepsilon_\\theta/\\sigma_t``
   and Tweedie's denoiser ``(x - \\sigma_t\\varepsilon_\\theta)/\\alpha_t``.
3. `reddiff.jl` — [`REDDiff`](@ref), the proximal operator of Mardani et al. 2023
   (arXiv:2305.04391): variational inference with a point-mass posterior, whose regulariser
   gradient is ``\\mathbb{E}_{t,\\varepsilon}[\\lambda_t(\\varepsilon_\\theta(x_t,t)-\\varepsilon)]``
   under a stop-gradient.
4. `factor.jl` — [`DiffusionFactor`](@ref), where `ImplicitREDDiff.md`'s selection matrices
   ``P_{in} + P_{out} + P_{latent} = \\mathrm{Id}`` become a `LenticulumCore.Polarity` and

   ```math
   E(x_0, x) = \\underbrace{\\mathbb{E}_{t,\\varepsilon}\\bigl[\\omega(t)\\|\\varepsilon_\\theta(\\alpha_t x + \\sigma_t\\varepsilon,t)-\\varepsilon\\|^2\\bigr]}_{\\text{entropy } \\mathbf{H}^c}
   \\;+\\; \\underbrace{\\tfrac12\\|P(x_0-x)\\|^2}_{\\text{energy } \\mathbf{l}^c}
   ```

## Two things worth knowing before reading the code

**No automatic differentiation anywhere.** RED-Diff's stop-gradient means the denoiser's
Jacobian is never formed, and the clamp's gradient is ``P^2(x-x_0)`` in closed form. So the
entire inversion is forward passes plus arithmetic, and this package's dependencies are
`LuxCore`, `Random` and `LinearAlgebra`. **Lux itself is not a dependency** — any Lux model
is an `AbstractLuxLayer`, which is all the wrapper needs.

**The inversion returns a `DiracBelief`,** because RED-Diff's variational family is
``\\mathcal{N}(\\mu,\\sigma^2 I)`` with ``\\sigma\\to 0``. That is the paper's own choice, not
a simplification made here, and it has consequences on a graph — see `factor.md` §5.

Concept notes are in `markdown/Diffusion/`; per-file implementation notes sit next to each
source file, per `Start here.md`.
"""
module VariationalDiffusion

using DispatchDoctor: @stable
using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG, randn, rand
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore
using Mycelium: Mycelium

include("schedule.jl")
include("predictor.jl")
include("reddiff.jl")
include("factor.jl")

# --- Schedules -------------------------------------------------------------
export AbstractNoiseSchedule, VPSDE
export beta, integrated_beta, alpha, sigma, snr, drift, diffusion
export perturb, sample_time, marginal_variance

# --- The noise predictor ---------------------------------------------------
export NoisePredictor, noise_schedule, epsilon, score, denoise, denoising_loss

# --- RED-Diff --------------------------------------------------------------
export REDDiff, reddiff_weight, regulariser_gradient, reddiff_solve, calibrate_lambda

# --- The factor ------------------------------------------------------------
export DiffusionFactor, DiffusionModel
export statedim, blockranges, precision_vector, assemble_state

end # module
