# ---------------------------------------------------------------------------
# Energy-parametrised diffusion models: the network outputs a scalar E_θ(x, t), and the noise
# prediction is its gradient,
#
#     ε_θ(x, t) = σ_t ∇ₓ E_θ(x, t),    so the score is  s_θ = -∇ₓ E_θ  exactly.
#
# A noise predictor that outputs ε directly is a *direct* model of the score and need not be a
# gradient of anything (its Jacobian is not symmetric). With an energy it is conservative by
# construction, and the implicit learner's field g becomes the exact gradient of a scalar U,
# which gives the learned relation a real energy (`implicit_energy`).
#
# The price is second derivatives: ε needs ∇ₓE, the input Jacobian is σ_t times the Hessian,
# and the parameter VJP is the mixed derivative ∂_θ⟨∇ₓE, w⟩. All three come from the predictor's
# `ad` backend through a package extension; the core stays AD-free. See `energy_network.md`.
# ---------------------------------------------------------------------------

"""
    EnergyNetwork(model)

Marks `model` as a **scalar energy** ``E_\\theta(x, t)`` rather than a noise predictor. Wrapped in
a [`NoisePredictor`](@ref), it defines

```math
\\varepsilon_\\theta(x, t) = \\sigma_t\\,\\nabla_x E_\\theta(x, t), \\qquad s_\\theta(x, t) = -\\nabla_x E_\\theta(x, t),
```

so the score is conservative by construction:

```julia
pred = NoisePredictor(EnergyNetwork(net), VPSDE(); input, ad = AutoZygote())
```

`net` sees `input(x, t)` and returns one value per column of `x` (a `1 × B` output, or a
scalar for a single point). Parameters are the network's own. Every function of the package
that takes a `NoisePredictor` accepts this one; the implicit learner additionally gains a
scalar energy, [`implicit_energy`](@ref).

Derivatives need an AD backend in `ad`, and second derivatives at that: the input Jacobian is
a Hessian and the parameter VJP a mixed derivative. `AutoZygote()` and
`DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoZygote())` work with Lux layers
(load DifferentiationInterface and the backend), as does `SecondOrder(AutoMooncakeForward(), AutoMooncake())`;
Enzyme's forward-over-reverse on CPU does not yet, but `ad = AutoReactant()` (with Reactant
loaded) runs it inside the compiled program (`energy_network.md` §4, §7).
"""
struct EnergyNetwork{L} <: LuxCore.AbstractLuxWrapperLayer{:model}
    model::L
end

"""
    energy(p::NoisePredictor{<:EnergyNetwork}, x, t, ps, st) -> (E, st)

The network's energy ``E_\\theta(x, t)``: a scalar for a single point, the sum over columns for
a batch. One forward pass; no derivative needed.
"""
function energy(p::NoisePredictor{<:EnergyNetwork}, x, t, ps, st)
    t = _times(p, x, t)
    _ondevice(p) && return energy(_local(p), _to(p, x), _to(p, t), ps, st)        # a scalar comes back as it is
    E, st = LuxCore.apply(p.model.model, p.input(x, t), ps, st)
    return (sum(E), st)
end

_sigmas(s, t) = t isa Number ? sigma(s, t) : sigma.(Ref(s), t)

# σ of the coordinates `idx` (column 1); a block's columns share their times in every caller
_sigma_rows(p::NoisePredictor, t, X, idx) = p.blocks === nothing ? sigma(p.schedule, t) :
    Array(sigma.(Ref(p.schedule), Array(t)[p.blocks.owner[idx], 1]))

function epsilon(p::NoisePredictor{<:EnergyNetwork}, x, t, ps, st)
    t = _times(p, x, t)
    _ondevice(p) && return _epsilon_device(p, x, t, ps, st)
    p.ad === nothing && _need_ad()
    return (_sigmas(p, t, x) .* reshape(_energy_grad(p.ad, p, x, t, ps, st), size(x)), st)
end

# σ ⊙ ∇²E: the Hessian, its rows scaled by σ (symmetric for one noise level)
function epsilon_jacobian(p::NoisePredictor{<:EnergyNetwork}, x, t, ps, st)
    t = _times(p, x, t)
    _ondevice(p) && return _back(p, epsilon_jacobian(_local(p), _to(p, x), _to(p, t), ps, st))
    p.ad === nothing && _need_ad()
    H = _energy_hessian(p.ad, p, x, t, ps, st)
    return p.blocks === nothing ? sigma(p.schedule, t) .* H : _sigma_rows(p, t, x, eachindex(x)) .* Array(H)
end

# the free-coordinate block: σ times the Hessian of Σ_b w_b E(x_b + P u) in u
function epsilon_jacobian_block(p::NoisePredictor{<:EnergyNetwork}, X, t, ps, st, idx, w)
    t = _times(p, X, t)
    _ondevice(p) && return _back(p, epsilon_jacobian_block(_local(p), _to(p, X), _to(p, t), ps, st, idx, _to(p, w)))
    p.ad === nothing && _need_ad()
    return _sigma_rows(p, t, X, idx) .* _energy_hessian_block(p.ad, p, X, t, ps, st, idx, w)
end

# Jᵀw = ∇²E (σ ⊙ w) and J v = σ ⊙ ∇²E v, column by column: one Hessian-vector product of the
# summed energy (the same product for one noise level, where σ is a number)
function epsilon_vjp_input(p::NoisePredictor{<:EnergyNetwork}, X, t, ps, st, W)
    t = _times(p, X, t)
    _ondevice(p) && return _back(p, epsilon_vjp_input(_local(p), _to(p, X), _to(p, t), ps, st, _to(p, W)))
    p.ad === nothing && _need_ad()
    return _energy_hvp(p.ad, p, X, t, ps, st, _sigmas(p, t, X) .* reshape(W, size(X)))
end

function epsilon_jvp_input(p::NoisePredictor{<:EnergyNetwork}, X, t, ps, st, V)
    t = _times(p, X, t)
    _ondevice(p) && return _back(p, epsilon_jvp_input(_local(p), _to(p, X), _to(p, t), ps, st, _to(p, V)))
    p.ad === nothing && _need_ad()
    return _sigmas(p, t, X) .* _energy_hvp(p.ad, p, X, t, ps, st, V)
end

# ∂_θ⟨σ ⊙ ∇ₓE, w⟩ = ∂_θ⟨∇ₓE, σ ⊙ w⟩, with σ per column (and per coordinate with noise blocks)
function epsilon_vjp_params(p::NoisePredictor{<:EnergyNetwork}, x, t, ps, st, w)
    t = _times(p, x, t)
    _ondevice(p) && return epsilon_vjp_params(_local(p), _to(p, x), _to(p, t), ps, st, _to(p, w))
    p.ad === nothing && _need_ad()
    return _energy_mixed(p.ad, p, x, t, ps, st, reshape(w, size(x)) .* _sigmas(p, t, x))
end

_need_ad() = throw(ArgumentError(
    "an EnergyNetwork needs an AD backend for ε = σ∇E: NoisePredictor(EnergyNetwork(net), schedule; ad = AutoZygote()), " *
    "with DifferentiationInterface and Zygote loaded"))

# the hooks the DifferentiationInterface extension implements
_energy_grad(ad, p, x, t, ps, st) = _no_energy_ad(ad)
_energy_hessian(ad, p, x, t, ps, st) = _no_energy_ad(ad)
_energy_mixed(ad, p, x, t, ps, st, w) = _no_energy_ad(ad)
_energy_hessian_block(ad, p, X, t, ps, st, idx, w) = _no_energy_ad(ad)
_energy_hvp(ad, p, X, t, ps, st, W) = _no_energy_ad(ad)
_no_energy_ad(ad) = throw(ArgumentError(
    "no energy derivatives for the backend $(typeof(ad)): load DifferentiationInterface and the backend package"))

"""
    implicit_energy(m::ImplicitDiffusion, z, ps, st) -> (U, st)

The scalar whose gradient is the implicit learner's field, for an energy-parametrised
predictor:

```math
U(z) = \\sum_k w_k\\,\\lambda_{t_k}\\Bigl[\\tfrac{\\sigma_k}{\\alpha_k}\\,E_\\theta(\\alpha_k z + \\sigma_k\\varepsilon_k,\\ t_k)
       - \\varepsilon_k^\\top z\\Bigr], \\qquad \\nabla U = g = \\texttt{prior\\_field}(m, z).
```

So the learned relation has an energy, and a query has a loss: ``U(z) + \\tfrac12\\lVert P(z - z_0)\\rVert^2``
on the free coordinates. For a noise predictor that outputs ε directly no such ``U`` exists;
see `energy_network.md` §3. A closed-form [`GaussianMixtureEps`](@ref) (and so a kernel density
estimate) has one too, with ``E = -\\log p_t`` in closed form.
"""
function implicit_energy(m::ImplicitDiffusion{<:NoisePredictor{<:EnergyNetwork}}, z, ps, st)
    U = zero(float(eltype(z)))
    for k in eachindex(m.nodes.t)
        t = m.nodes.t[k]
        E, st = energy(m.predictor, _node_input(m, z, k), _node_times(m, t), ps, st)
        U += m.nodes.w[k] * _node_energy(m, t, E, view(m.nodes.ε, :, k), z)
    end
    return (U, st)
end

# one node's term c_t E - εᵀ(λ ⊙ z), c_t = λσ_t²/α_t²: its z-gradient is the node's part of the field
function _node_energy(m::ImplicitDiffusion, t, E, ε, z)
    s = m.predictor.schedule
    _, _, λc = _node_coefs(m, t)
    return m.λ * sigma(s, t)^2 / alpha(s, t)^2 * E - dot(ε, λc .* z)
end

# The closed-form mixture is an exact score, ε* = -σ ∇ log p_t, so its energy is E = -log p_t:
# the same formula, in closed form. This gives kernel and mixture models an energy too.
function implicit_energy(m::ImplicitDiffusion{<:NoisePredictor{<:GaussianMixtureEps}}, z, ps, st)
    U = zero(float(eltype(z)))
    for k in eachindex(m.nodes.t)
        t = m.nodes.t[k]
        E = -mixture_logdensity(m.predictor.model, _node_input(m, z, k), _mixture_times(m, t), ps)
        U += m.nodes.w[k] * _node_energy(m, t, E, view(m.nodes.ε, :, k), z)
    end
    return (U, st)
end

# a mixture's time per coordinate at a node of level t
_mixture_times(m::ImplicitDiffusion, t) = m.levels === nothing ? t : _node_times(m, t)[m.predictor.blocks.owner]

"""
    denoising_gradient(p::NoisePredictor, x₀, t, ε, ps, st; weights = nothing) -> (loss, ps̄, st)

The denoising loss ``\\frac1B\\sum_i w_i(\\hat\\varepsilon_i - \\varepsilon_i)^2`` over a batch (columns of `x₀`)
and its gradient with respect to the parameters, through [`epsilon_vjp_params`](@ref). `t` is a
number, a `1 × B` row, or with noise blocks a `(#blocks × B)` matrix ([`block_times`](@ref)): each
coordinate is noised at its block's level, ``x_t = \\alpha \\odot x_0 + \\sigma \\odot \\varepsilon``.
`weights` (one per coordinate, default all 1) weights the loss coordinate by coordinate: a small
block such as a label needs a large weight to count next to a large one.

For an [`EnergyNetwork`](@ref) this is the training step: the loss already contains a derivative
of the network, and this computes the loss gradient as one mixed second derivative, with no
nested AD in the training loop. Apply `ps̄` with any optimiser.
"""
function denoising_gradient(p::NoisePredictor, x₀, t, ε, ps, st; weights = nothing)
    t = _times(p, x₀, t)
    xt = _alphas(p, t, x₀) .* x₀ .+ _sigmas(p, t, x₀) .* ε
    ε̂, st = epsilon(p, xt, t, ps, st)
    B = _ncols(x₀)
    r = ε̂ .- ε
    wr = weights === nothing ? r : weights .* r
    return (sum(wr .* r) / B, epsilon_vjp_params(p, xt, t, ps, st, (2 / B) .* wr), st)
end

"""
    block_times(rng, p::NoisePredictor, B; joint = 0.0) -> Matrix

Training times for a predictor with noise blocks: a `(#blocks × B)` matrix, each entry drawn
uniformly from `[tmin, 1]` independently (UniDiffuser), so the network learns every block at
every level: a block at `tmin` is a clean condition, a block at 1 is marginalised away. With
`joint` the given fraction of the columns has one time shared by all blocks instead, the joint
model's training distribution.
"""
function block_times(rng::AbstractRNG, p::NoisePredictor, B::Integer; joint::Real = 0.0)
    s = p.schedule
    nb = nblocks(p)
    T = s.tmin .+ (1 - s.tmin) .* rand(rng, nb, B)
    shared = rand(rng, B) .< joint
    T[:, shared] .= T[1:1, shared]
    return T
end

# the energy of every column: one network call (with a device, there), no derivative
function energy_columns(p::NoisePredictor{<:EnergyNetwork}, X, t, ps, st)
    t = _times(p, X, t)
    if _ondevice(p)
        E, st = energy_columns(_local(p), _to(p, X), _to(p, t), ps, st)
        return (_back(p, E), st)
    end
    E, st = LuxCore.apply(p.model.model, p.input(X, t), ps, st)
    return (vec(E), st)
end
