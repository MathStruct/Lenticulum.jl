# ---------------------------------------------------------------------------
# `NoisePredictor(model, schedule; ad = AutoReactant())`: the network's forward pass, its
# parameter VJP and its input Jacobian, each compiled once by Reactant (XLA) and cached per
# argument shape, with Enzyme as the differentiator inside the compiled program. For an
# `EnergyNetwork`, the gradient, Hessian-vector products and the mixed derivative likewise
# (nested Enzyme, forward over reverse).
#
# DifferentiationInterface has no Reactant backend, hence this separate extension. Inputs and
# outputs stay ordinary Julia arrays: the solvers in this package are CPU code that calls the
# network a few times per iteration, so each call moves its arguments to the device and the
# result back. That is cheap for the small models this package is about and keeps the solvers
# unchanged; time `t` is passed as a traced number, so a new `t` does not recompile.
# See `backends.md`.
# ---------------------------------------------------------------------------
module VariationalDiffusionReactantExt

using VariationalDiffusion: VariationalDiffusion, NoisePredictor, EnergyNetwork
using LuxCore: LuxCore
using ADTypes: AutoReactant
using Reactant: Reactant
const Enzyme = Reactant.Enzyme

const COMPILED = Dict{Any,Any}()

_dev(a) = Reactant.to_rarray(a)
_host(a::AbstractArray) = Array(a)
_host(a::Union{NamedTuple,Tuple}) = map(_host, a)
_host(a) = a
_shape(a::AbstractArray) = (typeof(a), size(a))
_shape(a) = typeof(a)

_fwd(pred, x, t, ps, st) = first(LuxCore.apply(pred.model, pred.input(x, t), ps, st))
_dot(pred, x, t, ps, st, w) = sum(vec(_fwd(pred, x, t, ps, st)) .* vec(w))   # shapes may differ (n×1 vs n)

function _vjp_ps(pred, x, t, ps, st, w)
    f(ps) = _dot(pred, x, t, ps, st, w)
    return only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), ps))
end

function _vjp_x(pred, x, t, ps, st, w)
    f(x) = _dot(pred, x, t, ps, st, w)
    return only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x))
end

# compile `f(pred, args...)` once per (function, predictor, argument shapes)
function _run(f, pred, args...)
    dargs = map(_dev, args)
    g = get!(COMPILED, (f, pred, map(_shape, dargs))) do
        Reactant.@compile f(pred, dargs...)
    end
    return g(pred, dargs...)
end

_args(x, t, ps, st) = (x, Reactant.ConcreteRNumber(float(eltype(x))(t)), ps, st)

function VariationalDiffusion._apply(::AutoReactant, pred::NoisePredictor, x, t, ps, st)
    return (Array(_run(_fwd, pred, _args(x, t, ps, st)...)), st)
end

function VariationalDiffusion._ad_vjp_params(::AutoReactant, pred::NoisePredictor, x, t, ps, st, w)
    return _host(_run(_vjp_ps, pred, _args(x, t, ps, st)..., w))
end

# one compiled input-VJP, called once per output coordinate; dim Z is small here
function VariationalDiffusion._ad_jacobian(ad::AutoReactant, pred::NoisePredictor, x, t, ps, st)
    T = float(eltype(x))
    x = collect(T.(x))
    m = length(first(VariationalDiffusion._apply(ad, pred, x, t, ps, st)))
    J = zeros(T, m, length(x))
    for i in 1:m
        w = zeros(T, m)
        w[i] = 1
        J[i, :] = vec(Array(_run(_vjp_x, pred, _args(x, t, ps, st)..., w)))
    end
    return J
end

# the free-coordinate block, one compiled input-VJP per free coordinate: row i of
# Σ_b w_b ∂ε(x_b)/∂x_b is the VJP with cotangent e_i w' (every column of X at once)
function VariationalDiffusion._ad_jacobian_block(ad::AutoReactant, pred::NoisePredictor, X, t, ps, st, idx, w)
    T = float(eltype(X))
    X = collect(T.(X))
    n = size(X, 1)
    J = zeros(T, length(idx), length(idx))
    for (a, i) in enumerate(idx)
        W = zeros(T, size(X))
        reshape(W, n, :)[i, :] .= w
        X̄ = Array(_run(_vjp_x, pred, _args(X, t, ps, st)..., W))
        J[a, :] = vec(sum(reshape(X̄, n, :); dims = 2))[idx]
    end
    return J
end

function VariationalDiffusion._ad_vjp_input(ad::AutoReactant, pred::NoisePredictor, X, t, ps, st, W)
    T = float(eltype(X))
    return reshape(Array(_run(_vjp_x, pred, _args(collect(T.(X)), t, ps, st)..., collect(T.(reshape(W, size(X)))))), size(X))
end


# --- energy-parametrised predictors: nested Enzyme inside the compiled program -------------
# Enzyme's forward-over-reverse fails on Julia code (energy_network.md §4) but is exact on the
# traced program, so every second derivative is forward mode over a compiled reverse gradient.
# The energy is weighted per column, Σ_b w_b E(x_b), so one program serves the plain gradient
# (w = 1) and the free-coordinate block (w the field's weights).
_ew(pred, x, t, ps, st, w) = sum(vec(first(LuxCore.apply(pred.model.model, pred.input(x, t), ps, st))) .* vec(w))
_ew_ps(ps, pred, x, t, st, w) = _ew(pred, x, t, ps, st, w)

_egrad(pred, x, t, ps, st, w) =
    Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(_ew), Enzyme.Const(pred), x, Enzyme.Const(t), Enzyme.Const(ps),
                    Enzyme.Const(st), Enzyme.Const(w))[2]
_egrad_ps(pred, x, t, ps, st, w) =
    first(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(_ew_ps), ps, Enzyme.Const(pred), Enzyme.Const(x), Enzyme.Const(t),
                          Enzyme.Const(st), Enzyme.Const(w)))

# ∇²E v, column by column (x-direction v), and ∂_θ⟨∇ₓE, v⟩: the forward derivative in x of
# the x-gradient and of the θ-gradient
_ehvp(pred, x, t, ps, st, w, v) =
    only(Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(_egrad), Enzyme.Const(pred), Enzyme.Duplicated(x, v), Enzyme.Const(t),
                         Enzyme.Const(ps), Enzyme.Const(st), Enzyme.Const(w)))
_emixed(pred, x, t, ps, st, w, v) =
    only(Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(_egrad_ps), Enzyme.Const(pred), Enzyme.Duplicated(x, v),
                         Enzyme.Const(t), Enzyme.Const(ps), Enzyme.Const(st), Enzyme.Const(w)))

_ncols(x) = x isa AbstractVector ? 1 : prod(size(x)[2:end])
_ones(x, T) = ones(T, _ncols(x))
_dense(T, x) = collect(T.(x))

function VariationalDiffusion._energy_grad(::AutoReactant, pred::NoisePredictor{<:EnergyNetwork}, x, t, ps, st)
    T = float(eltype(x))
    return reshape(Array(_run(_egrad, pred, _args(_dense(T, x), t, ps, st)..., _ones(x, T))), size(x))
end

function VariationalDiffusion._energy_hvp(::AutoReactant, pred::NoisePredictor{<:EnergyNetwork}, X, t, ps, st, W)
    T = float(eltype(X))
    V = _dense(T, reshape(W, size(X)))
    return reshape(Array(_run(_ehvp, pred, _args(_dense(T, X), t, ps, st)..., _ones(X, T), V)), size(X))
end

# one compiled Hessian-vector product per coordinate; dim Z is small here
function VariationalDiffusion._energy_hessian(ad::AutoReactant, pred::NoisePredictor{<:EnergyNetwork}, x, t, ps, st)
    T = float(eltype(x))
    x = _dense(T, vec(x))
    n = length(x)
    H = zeros(T, n, n)
    for i in 1:n
        H[:, i] = VariationalDiffusion._energy_hvp(ad, pred, x, t, ps, st, T.(1:n .== i))
    end
    return (H .+ H') ./ 2
end

# the Hessian in u of Σ_b w_b E(x_b + P u): column a is Σ_b w_b ∇²E(x_b) e_{idx[a]} restricted to idx,
# one weighted Hessian-vector product with the direction e_{idx[a]} in every column
function VariationalDiffusion._energy_hessian_block(::AutoReactant, pred::NoisePredictor{<:EnergyNetwork},
                                                    X, t, ps, st, idx, w)
    T = float(eltype(X))
    X = _dense(T, X)
    n, k = size(X, 1), length(idx)
    wd = _dense(T, vec(w))
    H = zeros(T, k, k)
    for (a, i) in enumerate(idx)
        V = zeros(T, size(X))
        reshape(V, n, :)[i, :] .= 1
        HV = Array(_run(_ehvp, pred, _args(X, t, ps, st)..., wd, V))
        H[:, a] = vec(sum(reshape(HV, n, :); dims = 2))[idx]
    end
    return (H .+ H') ./ 2
end

function VariationalDiffusion._energy_mixed(::AutoReactant, pred::NoisePredictor{<:EnergyNetwork}, x, t, ps, st, w)
    T = float(eltype(x))
    V = _dense(T, reshape(w, size(x)))
    return _host(_run(_emixed, pred, _args(_dense(T, x), t, ps, st)..., _ones(x, T), V))
end

end
