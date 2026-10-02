# ---------------------------------------------------------------------------
# `NoisePredictor(model, schedule; ad = AutoReactant())`: the network's forward pass, its
# parameter VJP and its input Jacobian, each compiled once by Reactant (XLA) and cached per
# argument shape, with Enzyme as the differentiator inside the compiled program.
#
# DifferentiationInterface has no Reactant backend, hence this separate extension. Inputs and
# outputs stay ordinary Julia arrays: the solvers in this package are CPU code that calls the
# network a few times per iteration, so each call moves its arguments to the device and the
# result back. That is cheap for the small models this package is about and keeps the solvers
# unchanged; time `t` is passed as a traced number, so a new `t` does not recompile.
# See `backends.md`.
# ---------------------------------------------------------------------------
module VariationalDiffusionReactantExt

using VariationalDiffusion: VariationalDiffusion, NoisePredictor
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

end
