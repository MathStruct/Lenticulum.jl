# ---------------------------------------------------------------------------
# AD-backend-agnostic derivatives of ε_θ, through DifferentiationInterface.
#
# Loaded when DifferentiationInterface is. The backend is whatever ADTypes object the user put
# in `NoisePredictor(...; ad = ...)`: AutoZygote, AutoEnzyme, AutoMooncake, AutoForwardDiff, …
# The backend package itself (Zygote, Enzyme, …) must be loaded by the user, as with any DI call.
#
# Lux parameter trees are nested NamedTuples, which not every backend accepts as a
# differentiation variable, so they are flattened to one vector and rebuilt — with the vector's
# element type, so forward-mode duals pass through. Model, input and time travel as DI
# `Constant` contexts rather than in a closure, which is what Enzyme wants.
# ---------------------------------------------------------------------------
module VariationalDiffusionDifferentiationInterfaceExt

using VariationalDiffusion: VariationalDiffusion, NoisePredictor, EnergyNetwork, ProxNetwork, epsilon
using LuxCore: LuxCore
using DifferentiationInterface: DifferentiationInterface as DI
const ADTypes = DI.ADTypes

# --- flatten / rebuild a parameter tree ---------------------------------------
_flatlen(x::AbstractArray{<:Number}) = length(x)
_flatlen(x::Number) = 1
_flatlen(x::Union{NamedTuple,Tuple}) = sum(_flatlen, values(x); init = 0)

_flatfill!(v, i, x::AbstractArray{<:Number}) = (n = length(x); copyto!(v, i, vec(x), 1, n); i + n)
_flatfill!(v, i, x::Number) = (v[i] = x; i + 1)
function _flatfill!(v, i, x::Union{NamedTuple,Tuple})
    for y in values(x)
        i = _flatfill!(v, i, y)
    end
    return i
end
_eltype(x::AbstractArray{<:Number}) = eltype(x)
_eltype(x::Number) = typeof(x)
_eltype(x::Union{NamedTuple,Tuple}) = isempty(x) ? Float64 : promote_type(map(_eltype, values(x))...)

function _flatten(ps)
    v = Vector{_eltype(ps)}(undef, _flatlen(ps))
    _flatfill!(v, 1, ps)
    return v
end

# The offsets of every leaf are computed once, outside the differentiated function (they do not
# depend on θ); rebuilding then only slices and reshapes θ, which every backend differentiates.
_offsets(x::AbstractArray{<:Number}, i) = (i, i + length(x))
_offsets(x::Number, i) = (i, i + 1)
function _offsets(x::Union{NamedTuple,Tuple}, i)
    out = Any[]
    for y in values(x)
        o, i = _offsets(y, i)
        push!(out, o)
    end
    r = Tuple(out)
    return (x isa NamedTuple ? NamedTuple{keys(x)}(r) : r, i)
end
offset_tree(ps) = first(_offsets(ps, 1))

_rebuild(θ, o::Int, x::AbstractArray{<:Number}) = reshape(θ[o:(o + length(x) - 1)], size(x))
_rebuild(θ, o::Int, x::Number) = θ[o]
_rebuild(θ, o::NamedTuple, x::NamedTuple) = NamedTuple{keys(x)}(map((oi, xi) -> _rebuild(θ, oi, xi), values(o), values(x)))
_rebuild(θ, o::Tuple, x::Tuple) = map((oi, xi) -> _rebuild(θ, oi, xi), o, x)
_unflatten(θ, offs, like) = _rebuild(θ, offs, like)
_unflatten(θ, like) = _rebuild(θ, offset_tree(like), like)

# --- the two operations ---------------------------------------------------------
# Outputs are flattened: a model may return a column (n×1) for a vector input, and the
# cotangent `w` must then match the output's shape, not the input's.
_eps_of_params(θ, pred, x, t, like, offs, st) = vec(first(epsilon(pred, x, t, _unflatten(θ, offs, like), st)))
_eps_of_input(z, pred, t, ps, st) = vec(first(epsilon(pred, z, t, ps, st)))

function VariationalDiffusion._ad_vjp_params(ad::ADTypes.AbstractADType, pred::NoisePredictor,
                                             x, t, ps, st, w)
    θ = _flatten(ps)
    offs = offset_tree(ps)
    tθ = DI.pullback(_eps_of_params, ad, θ, (vec(w),),
                     DI.Constant(pred), DI.Constant(x), DI.Constant(t), DI.Constant(ps), DI.Constant(offs),
                     DI.Constant(st))
    return _unflatten(only(tθ), ps)
end

function VariationalDiffusion._ad_jacobian(ad::ADTypes.AbstractADType, pred::NoisePredictor, x, t, ps, st)
    return DI.jacobian(_eps_of_input, ad, collect(float.(x)),
                       DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
end

# --- the free-coordinate block, and input VJPs ------------------------------------------
# All columns of X move together with z (x_b = α z + σ ε_b), so the block is the Jacobian of
# u ↦ (Σ_b w_b ε(x_b + P u))_I at u = 0, a map ℝ^|I| → ℝ^|I|: |I| JVPs in forward mode, |I| VJPs
# in reverse mode. P embeds u into the coordinates I.
_embedding(T, n, idx) = (P = zeros(T, n, length(idx)); foreach(((a, j),) -> P[j, a] = 1, enumerate(idx)); P)

function _eps_block(u, pred, X, t, ps, st, P, w, idx)
    ε̂ = first(epsilon(pred, X .+ P * u, t, ps, st))
    return (reshape(ε̂, size(X, 1), :) * w)[idx]
end

function VariationalDiffusion._ad_jacobian_block(ad::ADTypes.AbstractADType, pred::NoisePredictor,
                                                 X, t, ps, st, idx, w)
    T = float(eltype(X))
    return DI.jacobian(_eps_block, ad, zeros(T, length(idx)), DI.Constant(pred), DI.Constant(collect(T.(X))),
                       DI.Constant(t), DI.Constant(ps), DI.Constant(st), DI.Constant(_embedding(T, size(X, 1), idx)),
                       DI.Constant(collect(T.(w))), DI.Constant(idx))
end

_eps_of_inputs(X, pred, t, ps, st) = vec(first(epsilon(pred, X, t, ps, st)))

function VariationalDiffusion._ad_vjp_input(ad::ADTypes.AbstractADType, pred::NoisePredictor, X, t, ps, st, W)
    X̄ = DI.pullback(_eps_of_inputs, ad, collect(float.(X)), (collect(float.(vec(W))),),
                    DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
    return reshape(only(X̄), size(X))
end

# a forward-mode JVP where the backend has one; reverse-only backends (Zygote, Mooncake) would pay
# one pullback per coordinate for it, so they take the directional difference instead
function VariationalDiffusion._ad_jvp_input(ad::ADTypes.AbstractADType, pred::NoisePredictor, X, t, ps, st, V)
    ADTypes.mode(ad) isa ADTypes.ReverseMode && return VariationalDiffusion._fd_jvp_input(pred, X, t, ps, st, V)
    Ẋ = DI.pushforward(_eps_of_inputs, ad, collect(float.(X)), (collect(float.(reshape(V, size(X)))),),
                       DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
    return reshape(only(Ẋ), size(X))
end

# --- energy-parametrised predictors: first and second derivatives of a scalar -----------
# E is summed over the columns of a batch, so its x-gradient is the per-column gradients.
# The mixed derivative ∂_θ⟨∇ₓE, w⟩ is the θ-block of one Hessian-vector product on the joint
# vector u = [x; θ] in the direction [w; 0]; its x-block would be ∇²ₓE·w.
_energy_x(x, pred, t, ps, st) = sum(first(LuxCore.apply(pred.model.model, pred.input(x, t), ps, st)))

function _energy_joint(u, pred, t, xshape, like, offs, st)
    nx = prod(xshape)
    x = reshape(u[1:nx], xshape)
    return _energy_x(x, pred, t, _unflatten(u[(nx + 1):end], offs, like), st)
end

_first_order(ad) = ad isa DI.SecondOrder ? DI.inner(ad) : ad

function VariationalDiffusion._energy_grad(ad::ADTypes.AbstractADType, pred::NoisePredictor{<:EnergyNetwork},
                                           x, t, ps, st)
    return DI.gradient(_energy_x, _first_order(ad), collect(float.(x)),
                       DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
end

function VariationalDiffusion._energy_hessian(ad::ADTypes.AbstractADType, pred::NoisePredictor{<:EnergyNetwork},
                                              x, t, ps, st)
    return DI.hessian(_energy_x, ad, collect(float.(vec(x))),
                      DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
end

# the block: the Hessian in u of Σ_b w_b E(x_b + P u), |I| Hessian-vector products
function _energy_block(u, pred, X, t, ps, st, P, w)
    E = first(LuxCore.apply(pred.model.model, pred.input(X .+ P * u, t), ps, st))
    return E isa Number ? E * only(w) : sum(vec(E) .* w)
end

function VariationalDiffusion._energy_hessian_block(ad::ADTypes.AbstractADType, pred::NoisePredictor{<:EnergyNetwork},
                                                    X, t, ps, st, idx, w)
    T = float(eltype(X))
    return DI.hessian(_energy_block, ad, zeros(T, length(idx)), DI.Constant(pred), DI.Constant(collect(T.(X))),
                      DI.Constant(t), DI.Constant(ps), DI.Constant(st), DI.Constant(_embedding(T, size(X, 1), idx)),
                      DI.Constant(collect(T.(w))))
end

# ∇²E W column by column: the energy is summed over columns, so its Hessian is block diagonal
function VariationalDiffusion._energy_hvp(ad::ADTypes.AbstractADType, pred::NoisePredictor{<:EnergyNetwork},
                                          X, t, ps, st, W)
    hv = DI.hvp(_energy_x, ad, collect(float.(X)), (collect(float.(reshape(W, size(X)))),),
                DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
    return reshape(only(hv), size(X))
end

function VariationalDiffusion._energy_mixed(ad::ADTypes.AbstractADType, pred::NoisePredictor{<:EnergyNetwork},
                                            x, t, ps, st, w)
    θ = _flatten(ps)
    xv = collect(float.(vec(x)))
    T = promote_type(eltype(xv), eltype(θ))
    u = vcat(T.(xv), T.(θ))
    v = vcat(T.(vec(w)), zeros(T, length(θ)))
    hv = only(DI.hvp(_energy_joint, ad, u, (v,), DI.Constant(pred), DI.Constant(t), DI.Constant(size(x)),
                     DI.Constant(ps), DI.Constant(offset_tree(ps)), DI.Constant(st)))
    return _unflatten(hv[(length(xv) + 1):end], ps)
end

# --- ProxDM: the network's residual ε̂(v; t, λ), for the adjoint of `prox_infer` ------------
_prox_eps(v, p, t, λ, ps, st) = vec(first(LuxCore.apply(p.model, p.input(v, t, λ), ps, st)))
_prox_eps_params(θ, p, v, t, λ, like, offs, st) = _prox_eps(v, p, t, λ, _unflatten(θ, offs, like), st)

function VariationalDiffusion._prox_ad_vjp_input(ad::ADTypes.AbstractADType, p::ProxNetwork, v, t, λ, ps, st, y)
    v̄ = DI.pullback(_prox_eps, ad, collect(float.(v)), (collect(float.(vec(y))),),
                    DI.Constant(p), DI.Constant(t), DI.Constant(λ), DI.Constant(ps), DI.Constant(st))
    return reshape(only(v̄), size(v))
end

function VariationalDiffusion._prox_ad_vjp_params(ad::ADTypes.AbstractADType, p::ProxNetwork, v, t, λ, ps, st, y)
    tθ = DI.pullback(_prox_eps_params, ad, _flatten(ps), (collect(float.(vec(y))),),
                     DI.Constant(p), DI.Constant(collect(float.(v))), DI.Constant(t), DI.Constant(λ), DI.Constant(ps),
                     DI.Constant(offset_tree(ps)), DI.Constant(st))
    return _unflatten(only(tθ), ps)
end

function VariationalDiffusion._prox_ad_jacobian(ad::ADTypes.AbstractADType, p::ProxNetwork, v, t, λ, ps, st)
    return DI.jacobian(_prox_eps, ad, collect(float.(v)),
                       DI.Constant(p), DI.Constant(t), DI.Constant(λ), DI.Constant(ps), DI.Constant(st))
end

end # module
