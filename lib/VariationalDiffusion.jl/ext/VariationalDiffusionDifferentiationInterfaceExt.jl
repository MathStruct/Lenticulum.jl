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
using LenticulumCore: _flatten, offset_tree, _unflatten

# --- arrays where the caller's arrays are ----------------------------------------------
# A dense floating-point copy on the same device (`collect` would bring a GPU array home), and
# a CPU-built constant moved next to `X`.
_dense(x::AbstractArray) = copyto!(similar(x, float(eltype(x)), size(x)), x)
_dense(x::Number) = float(x)
_like(X, A::AbstractArray) = X isa Array ? A : copyto!(similar(X, eltype(A), size(A)), A)
_onehot(X, T, n, a) = _like(X, (e = zeros(T, n); e[a] = 1; e))

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
    return DI.jacobian(_eps_of_input, ad, _dense(x),
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
    Xd, P, wd = _dense(X), _like(X, _embedding(T, size(X, 1), idx)), _like(X, T.(collect(w)))
    ctx = (DI.Constant(pred), DI.Constant(Xd), DI.Constant(t), DI.Constant(ps), DI.Constant(st), DI.Constant(P),
           DI.Constant(wd), DI.Constant(idx))
    X isa Array && return DI.jacobian(_eps_block, ad, zeros(T, length(idx)), ctx...)
    # on a device: one product per coordinate, with basis vectors made there
    k = length(idx)
    u = _like(X, zeros(T, k))
    J = zeros(T, k, k)
    if ADTypes.mode(ad) isa ADTypes.ReverseMode
        for a in 1:k                                  # row a
            J[a, :] .= Array(only(DI.pullback(_eps_block, ad, u, (_onehot(X, T, k, a),), ctx...)))
        end
    else
        for a in 1:k                                  # column a
            J[:, a] .= Array(only(DI.pushforward(_eps_block, ad, u, (_onehot(X, T, k, a),), ctx...)))
        end
    end
    return J
end

_eps_of_inputs(X, pred, t, ps, st) = vec(first(epsilon(pred, X, t, ps, st)))

function VariationalDiffusion._ad_vjp_input(ad::ADTypes.AbstractADType, pred::NoisePredictor, X, t, ps, st, W)
    X̄ = DI.pullback(_eps_of_inputs, ad, _dense(X), (_dense(vec(W)),),
                    DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
    return reshape(only(X̄), size(X))
end

# a forward-mode JVP where the backend has one; reverse-only backends (Zygote, Mooncake) would pay
# one pullback per coordinate for it, so they take the directional difference instead
function VariationalDiffusion._ad_jvp_input(ad::ADTypes.AbstractADType, pred::NoisePredictor, X, t, ps, st, V)
    ADTypes.mode(ad) isa ADTypes.ReverseMode && return VariationalDiffusion._fd_jvp_input(pred, X, t, ps, st, V)
    Ẋ = DI.pushforward(_eps_of_inputs, ad, _dense(X), (_dense(reshape(V, size(X))),),
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
    return DI.gradient(_energy_x, _first_order(ad), _dense(x),
                       DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
end

function VariationalDiffusion._energy_hessian(ad::ADTypes.AbstractADType, pred::NoisePredictor{<:EnergyNetwork},
                                              x, t, ps, st)
    return DI.hessian(_energy_x, ad, _dense(vec(x)),
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
    Xd, P, wd = _dense(X), _like(X, _embedding(T, size(X, 1), idx)), _like(X, T.(collect(w)))
    ctx = (DI.Constant(pred), DI.Constant(Xd), DI.Constant(t), DI.Constant(ps), DI.Constant(st), DI.Constant(P),
           DI.Constant(wd))
    X isa Array && return DI.hessian(_energy_block, ad, zeros(T, length(idx)), ctx...)
    # on a device: one Hessian-vector product per coordinate
    k = length(idx)
    u = _like(X, zeros(T, k))
    H = zeros(T, k, k)
    for a in 1:k
        H[:, a] .= Array(only(DI.hvp(_energy_block, ad, u, (_onehot(X, T, k, a),), ctx...)))
    end
    return (H .+ H') ./ 2
end

# ∇²E W column by column: the energy is summed over columns, so its Hessian is block diagonal
function VariationalDiffusion._energy_hvp(ad::ADTypes.AbstractADType, pred::NoisePredictor{<:EnergyNetwork},
                                          X, t, ps, st, W)
    hv = DI.hvp(_energy_x, ad, _dense(X), (_dense(reshape(W, size(X))),),
                DI.Constant(pred), DI.Constant(t), DI.Constant(ps), DI.Constant(st))
    return reshape(only(hv), size(X))
end

# With a forward outer backend the mixed derivative is the derivative in h of the parameter
# gradient at x + h w: ∂_h ∇_θ E(x + h w) = ∇_θ⟨∇ₓE, w⟩. Only the input carries dual numbers,
# the parameters stay where they are (on a GPU), and no joint [x; θ] vector is formed.
_energy_theta(θ, pred, x, t, like, offs, st) = _energy_x(x, pred, t, _unflatten(θ, offs, like), st)
_mixed_at(h, pred, x, w, t, θ, inner, like, offs, st) =
    DI.gradient(_energy_theta, inner, θ, DI.Constant(pred), DI.Constant(x .+ h .* w), DI.Constant(t),
                DI.Constant(like), DI.Constant(offs), DI.Constant(st))

function VariationalDiffusion._energy_mixed(ad::DI.SecondOrder{<:ADTypes.AutoForwardDiff}, pred::NoisePredictor{<:EnergyNetwork},
                                            x, t, ps, st, w)
    θ = _flatten(ps)
    xv = float.(x)
    g = DI.derivative(_mixed_at, DI.outer(ad), zero(eltype(xv)), DI.Constant(pred), DI.Constant(xv),
                      DI.Constant(reshape(float.(w), size(xv))), DI.Constant(t), DI.Constant(θ), DI.Constant(DI.inner(ad)),
                      DI.Constant(ps), DI.Constant(offset_tree(ps)), DI.Constant(st))
    return _unflatten(g, ps)
end

function VariationalDiffusion._energy_mixed(ad::ADTypes.AbstractADType, pred::NoisePredictor{<:EnergyNetwork},
                                            x, t, ps, st, w)
    θ = _flatten(ps)
    xv = _dense(vec(x))
    T = promote_type(eltype(xv), eltype(θ))
    u = vcat(T.(xv), T.(θ))
    v = vcat(T.(vec(w)), fill!(similar(θ, T, length(θ)), 0))
    hv = only(DI.hvp(_energy_joint, ad, u, (v,), DI.Constant(pred), DI.Constant(t), DI.Constant(size(x)),
                     DI.Constant(ps), DI.Constant(offset_tree(ps)), DI.Constant(st)))
    return _unflatten(hv[(length(xv) + 1):end], ps)
end

# --- ProxDM: the network's residual ε̂(v; t, λ), for the adjoint of `prox_infer` ------------
_prox_eps(v, p, t, λ, ps, st) = vec(first(LuxCore.apply(p.model, p.input(v, t, λ), ps, st)))
_prox_eps_params(θ, p, v, t, λ, like, offs, st) = _prox_eps(v, p, t, λ, _unflatten(θ, offs, like), st)

function VariationalDiffusion._prox_ad_vjp_input(ad::ADTypes.AbstractADType, p::ProxNetwork, v, t, λ, ps, st, y)
    v̄ = DI.pullback(_prox_eps, ad, _dense(v), (_dense(vec(y)),),
                    DI.Constant(p), DI.Constant(t), DI.Constant(λ), DI.Constant(ps), DI.Constant(st))
    return reshape(only(v̄), size(v))
end

function VariationalDiffusion._prox_ad_vjp_params(ad::ADTypes.AbstractADType, p::ProxNetwork, v, t, λ, ps, st, y)
    tθ = DI.pullback(_prox_eps_params, ad, _flatten(ps), (_dense(vec(y)),),
                     DI.Constant(p), DI.Constant(_dense(v)), DI.Constant(t), DI.Constant(λ), DI.Constant(ps),
                     DI.Constant(offset_tree(ps)), DI.Constant(st))
    return _unflatten(only(tθ), ps)
end

function VariationalDiffusion._prox_ad_jacobian(ad::ADTypes.AbstractADType, p::ProxNetwork, v, t, λ, ps, st)
    return DI.jacobian(_prox_eps, ad, _dense(v),
                       DI.Constant(p), DI.Constant(t), DI.Constant(λ), DI.Constant(ps), DI.Constant(st))
end

end # module
