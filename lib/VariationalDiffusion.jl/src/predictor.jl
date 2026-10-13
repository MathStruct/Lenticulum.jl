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
struct NoisePredictor{L,S<:AbstractNoiseSchedule,F,A,D,B} <: LuxCore.AbstractLuxWrapperLayer{:model}
    model::L
    schedule::S
    input::F
    ad::A
    device::D
    blocks::B
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

NoisePredictor(model, sched::AbstractNoiseSchedule; input = default_input, ad = nothing, device = nothing,
               blocks = nothing) =
    NoisePredictor(model, sched, input, ad, _device(device), _noiseblocks(blocks))

# --- a noise level per block of coordinates --------------------------------------------
# With `blocks`, the coordinates are partitioned into blocks that may sit at different noise
# levels (UniDiffuser): `t` is then a number (every block at t), a vector with one time per
# block, or a (#blocks × B) matrix with one column per column of x. The network's `input`
# always receives the matrix; α and σ act coordinate by coordinate. See `predictor.md` §6.

"""
    NoiseBlocks(blocks)

A partition of the coordinates `1:n` into blocks that carry their own noise level, built from
the keyword `blocks` of [`NoisePredictor`](@ref) (a vector of index ranges or vectors).
"""
struct NoiseBlocks
    blocks::Vector{Vector{Int}}
    owner::Vector{Int}                                # the block of each coordinate
    function NoiseBlocks(blocks::AbstractVector)
        bs = [collect(Int, b) for b in blocks]
        length(bs) ≥ 2 || throw(ArgumentError("blocks: give at least two blocks"))
        n = sum(length, bs)
        owner = zeros(Int, n)
        for (k, b) in enumerate(bs), i in b
            1 ≤ i ≤ n && owner[i] == 0 || throw(ArgumentError("blocks must partition 1:$n; coordinate $i is out of range or repeated"))
            owner[i] = k
        end
        return new(bs, owner)
    end
end
_noiseblocks(::Nothing) = nothing
_noiseblocks(b::NoiseBlocks) = b
_noiseblocks(b) = NoiseBlocks(b)

"""
    nblocks(p::NoisePredictor) -> Int

The number of noise blocks (1 without `blocks`).
"""
nblocks(p::NoisePredictor) = p.blocks === nothing ? 1 : length(p.blocks.blocks)

_ncols(x::AbstractVector) = 1
_ncols(x::AbstractArray) = prod(size(x)[2:end])

# the times the network sees: unchanged without blocks, a (#blocks × B) matrix with them
_times(p::NoisePredictor, x, t) = p.blocks === nothing ? t : _block_times(length(p.blocks.blocks), t, _ncols(x))
_block_times(nb, t::Number, B) = fill(float(t), nb, B)
function _block_times(nb, t::AbstractVector, B)
    length(t) == nb || throw(DimensionMismatch("one time per noise block: $(length(t)) for $nb blocks"))
    return repeat(reshape(float.(t), nb, 1), 1, B)
end
function _block_times(nb, t::AbstractMatrix, B)
    size(t, 1) == nb && size(t, 2) == B && return t
    size(t, 1) == 1 && return _block_times(nb, nb == 1 ? t : repeat(t, nb, 1), B)    # a shared time per column
    size(t, 1) == nb && size(t, 2) == 1 && return repeat(t, 1, B)
    throw(DimensionMismatch("times of size $(size(t)) for $nb blocks and $B columns"))
end

# f(schedule, t) coordinate by coordinate, shaped like x (a scalar without blocks and a scalar t)
function _coordwise(p::NoisePredictor, f, t, x)
    s = p.schedule
    if p.blocks === nothing
        return t isa Number ? f(s, t) : f.(Ref(s), t)
    end
    T = float(eltype(x))                               # a Float32 batch on a device stays Float32
    F = T.(f.(Ref(s), _times(p, x, t)))[p.blocks.owner, :]
    return x isa AbstractVector ? vec(F) : reshape(F, size(x))
end
_alphas(p::NoisePredictor, t, x) = _coordwise(p, alpha, t, x)
_sigmas(p::NoisePredictor, t, x) = _coordwise(p, sigma, t, x)

# --- running on a device -------------------------------------------------------------
# With `device = (to, back)` every evaluation and derivative of the network takes its array
# arguments to the device once, runs there, with derivatives taken there too (dual numbers
# are made on the device, never moved), and brings plain arrays back. The solvers' state, the
# field's noise draws and the small linear algebra stay where they are. Inside, a copy of the
# predictor without a device does the work, so the hooks see device arrays and nothing else.
_local(p::NoisePredictor) = NoisePredictor(p.model, p.schedule, p.input, p.ad, nothing, p.blocks)
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
    t = _times(p, x, t)
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
    return (-ε̂ ./ _sigmas(p, t, x), st)
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
    return ((x .- _sigmas(p, t, x) .* ε̂) ./ _alphas(p, t, x), st)
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
    x_t = _alphas(p, t, x₀) .* x₀ .+ _sigmas(p, t, x₀) .* ε
    ε̂, st = epsilon(p, x_t, t, ps, st)
    d = ε̂ .- ε
    return (sum(abs2, d), st)
end
