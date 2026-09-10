# ---------------------------------------------------------------------------
# ε_θ(x, t) — a Lux model, wrapped so that the rest of the package can ask it for a score
# or a denoised estimate without knowing anything about its architecture.
#
# Three quantities, one network:
#
#     ε_θ(x,t)                                   the noise prediction        (the network)
#     s_θ(x,t) = -ε_θ(x,t)/σ_t                   the score  ∇ₓ log p_t(x)    (Song, §3.4)
#     x̂₀(x,t)  = (x - σ_t ε_θ(x,t))/α_t          the denoiser E[x₀|x_t]      (Tweedie)
#
# The second and third are *derived*, not learned, and that they are derived is what makes a
# noise predictor usable as a prior over x₀ rather than merely as a sampler.
#
# See `predictor.md`.
# ---------------------------------------------------------------------------

"""
    NoisePredictor(model, schedule; input = default_input)

``\\varepsilon_\\theta(x,t)`` — a wrapper making any Lux model into a diffusion prior.

`model` is any `LuxCore.AbstractLuxLayer`; parameters and state are **delegated** to it
untouched, via `AbstractLuxWrapperLayer`, so `ps` for a `NoisePredictor` *is* `ps` for the
model. Wrapping adds no parameters of its own.

`input` adapts the `(x, t)` pair to whatever the model expects, defaulting to the tuple
`(x, t)`. A model taking a concatenated time channel, a sinusoidal embedding or a
`NamedTuple` is accommodated by passing a different `input`; the wrapper deliberately knows
nothing about time embeddings, because that is the model's business.

```julia
pred = NoisePredictor(my_unet, VPSDE())
ps, st = LuxCore.setup(rng, pred)          # == LuxCore.setup(rng, my_unet)
ε, st  = epsilon(pred, x, 0.3, ps, st)
```

!!! note "LuxCore, not Lux"
    This package depends on **LuxCore**, the interface package, exactly as `Lenticulum` does.
    Any Lux model is an `AbstractLuxLayer`, so `Lux` itself is never needed here — see
    `predictor.md` §1.
"""
struct NoisePredictor{L,S<:AbstractNoiseSchedule,F} <: LuxCore.AbstractLuxWrapperLayer{:model}
    model::L
    schedule::S
    input::F
end

default_input(x, t) = (x, t)

NoisePredictor(model, sched::AbstractNoiseSchedule; input = default_input) =
    NoisePredictor(model, sched, input)

"""
    noise_schedule(p::NoisePredictor) -> AbstractNoiseSchedule

The schedule the predictor was built with.

Named `noise_schedule` rather than `schedule` because the latter is `Base.schedule` (for
`Task`s), and shadowing it would be the sixth name collision in this project — see
`Mycelium.md` §2 for the others.
"""
noise_schedule(p::NoisePredictor) = p.schedule

"""
    epsilon(p::NoisePredictor, x, t, ps, st) -> (ε̂, st)

One forward pass of the wrapped network: ``\\varepsilon_\\theta(x, t)``.

**This is the only place the network is evaluated, and it is only ever evaluated forwards.**
RED-Diff's stop-gradient means no reverse pass through `model` is required anywhere in this
package, which is why it has no automatic-differentiation dependency at all
(`reddiff.md` §2).
"""
epsilon(p::NoisePredictor, x, t, ps, st) = LuxCore.apply(p.model, p.input(x, t), ps, st)

"""
    score(p::NoisePredictor, x, t, ps, st) -> (s, st)

``s_\\theta(x,t) = -\\varepsilon_\\theta(x,t)/\\sigma_t \\approx \\nabla_x \\log p_t(x)``.

The identity is exact for the perturbation kernel: with
``x_t = \\alpha_t x_0 + \\sigma_t\\varepsilon``,

```math
\\nabla_{x_t}\\log p_{0t}(x_t\\mid x_0)
 = -\\frac{x_t - \\alpha_t x_0}{\\sigma_t^2}
 = -\\frac{\\varepsilon}{\\sigma_t}
```

so a perfect noise predictor is a perfect score. The ``1/\\sigma_t`` is why
[`VPSDE`](@ref) carries a `tmin`.
"""
function score(p::NoisePredictor, x, t, ps, st)
    ε̂, st = epsilon(p, x, t, ps, st)
    return (-ε̂ ./ sigma(p.schedule, t), st)
end

"""
    denoise(p::NoisePredictor, x, t, ps, st) -> (x̂₀, st)

Tweedie's formula: ``\\hat x_0 = (x - \\sigma_t\\varepsilon_\\theta(x,t))/\\alpha_t``, the
posterior mean ``\\mathbb{E}[x_0 \\mid x_t = x]``.

> [!note] This is exact, not heuristic
> For a Gaussian data distribution the test suite checks `denoise` against the closed-form
> linear-Gaussian posterior mean and they agree to floating point. Tweedie's formula turns a
> noise predictor into an **MMSE denoiser**, and that is the sense in which a diffusion model
> is a prior: it is the object the RED-Diff regulariser scores against.
"""
function denoise(p::NoisePredictor, x, t, ps, st)
    ε̂, st = epsilon(p, x, t, ps, st)
    s = p.schedule
    return ((x .- sigma(s, t) .* ε̂) ./ alpha(s, t), st)
end

"""
    denoising_loss(p, x₀, t, ε, ps, st) -> (ℝ, st)

``\\|\\varepsilon_\\theta(\\alpha_t x_0 + \\sigma_t\\varepsilon, t) - \\varepsilon\\|^2``, one
Monte-Carlo sample of the score-matching objective at a single `(t, ε)`.

Unweighted: the weighting ``\\omega(t)`` belongs to whoever is taking the expectation, and
different weightings mean different things — the ELBO, the perceptual-quality objective, and
RED-Diff's regulariser are the same integrand with three different ``\\omega``. See
`schedule.md` §3.
"""
function denoising_loss(p::NoisePredictor, x₀, t, ε, ps, st)
    x_t = perturb(p.schedule, x₀, t, ε)
    ε̂, st = epsilon(p, x_t, t, ps, st)
    d = ε̂ .- ε
    return (sum(abs2, d), st)
end
