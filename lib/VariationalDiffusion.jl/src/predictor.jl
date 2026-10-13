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

`device = (to = gpu_device(), back = cpu_device())` runs the network on an accelerator while
the solvers keep their state on the CPU: arguments go to the device on every call, derivatives
are taken there, results come back. The parameters must be moved to the device by the caller
(`ps = gpu_device()(ps)`); their cotangents stay there. MLDataDevices' `gpu_device()` converts
`Float64` arrays to `Float32`, which a consumer GPU needs for speed. `input` then receives
device arrays and must build any time embedding on the same device (`predictor.md` §5).

!!! note "LuxCore, not Lux"
    This package depends on **LuxCore**, the interface package, exactly as `Lenticulum` does.
    Any Lux model is an `AbstractLuxLayer`, so `Lux` itself is never needed here — see
    `predictor.md` §1.
"""
struct NoisePredictor{L,S<:AbstractNoiseSchedule,F,A,D} <: LuxCore.AbstractLuxWrapperLayer{:model}
    model::L
    schedule::S
    input::F
    ad::A
    device::D
end

default_input(x, t) = (x, t)

"""
    DeviceTransfer(to, back)

Where a [`NoisePredictor`](@ref) runs: `to` moves arrays to the device, `back` brings results
home. Built from the keyword `device = (to = …, back = …)`.
"""
struct DeviceTransfer{T,B}
    to::T
    back::B
end
_device(::Nothing) = nothing
_device(d::DeviceTransfer) = d
_device(d::NamedTuple) = DeviceTransfer(d.to, d.back)

NoisePredictor(model, sched::AbstractNoiseSchedule; input = default_input, ad = nothing, device = nothing) =
    NoisePredictor(model, sched, input, ad, _device(device))

# --- running on a device -------------------------------------------------------------
# With `device = (to, back)` every evaluation and derivative of the network takes its array
# arguments to the device once, runs there, with derivatives taken there too (dual numbers
# are made on the device, never moved), and brings plain arrays back. The solvers' state, the
# field's noise draws and the small linear algebra stay where they are. Inside, a copy of the
# predictor without a device does the work, so the hooks see device arrays and nothing else.
_local(p::NoisePredictor) = NoisePredictor(p.model, p.schedule, p.input, p.ad, nothing)
_ondevice(p::NoisePredictor) = p.device !== nothing
_to(p::NoisePredictor, x::AbstractArray) = p.device.to(x)
_to(p::NoisePredictor, x) = x
_back(p::NoisePredictor, y::AbstractArray) = p.device.back(y)
_back(p::NoisePredictor, y::Number) = y
_back(p::NoisePredictor, y::Union{NamedTuple,Tuple}) = y      # parameter cotangents stay with the parameters


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

**This is the only place the network is evaluated forwards.** RED-Diff's stop-gradient means
sampling and RED-Diff need no reverse pass through `model`. Only the implicit learner's
backward pass does (a parameter VJP and an input Jacobian), and it goes through the predictor's
`ad` backend, provided by a package extension (`predictor.md` §5).
"""
function epsilon(p::NoisePredictor, x, t, ps, st)
    _ondevice(p) && return _epsilon_device(p, x, t, ps, st)
    return _apply(p.ad, p, x, t, ps, st)
end

function _epsilon_device(p, x, t, ps, st)
    e, st = epsilon(_local(p), _to(p, x), _to(p, t), ps, st)
    return (_back(p, e), st)
end

# The forward hook. Plain Lux by default; an extension may compile it for its backend
# (`VariationalDiffusionReactantExt` does, for `ad = AutoReactant()`). See `predictor.md` §5.
_apply(ad, p::NoisePredictor, x, t, ps, st) = LuxCore.apply(p.model, p.input(x, t), ps, st)

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
