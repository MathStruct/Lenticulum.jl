# y = a*x + b + N(0,q). The test supplies the parameter-channel rule; expose supplies routing.
struct P36Affine{Beliefs} <: AbstractLenticulumFactor
    q::Float64
end
LuxCore.parameterlength(::P36Affine) = 2
LenticulumCore.channels(::P36Affine) = (LenticulumCore.Channel(:x, 1), LenticulumCore.Channel(:y, 1))
LenticulumCore.supported_polarities(::P36Affine) =
    (Polarity(x = Observed(), y = Unobserved()), Polarity(x = Unobserved(), y = Observed()))
LenticulumCore.supports_polarity(f::P36Affine, p::Polarity) = p in supported_polarities(f)
LuxCore.initialparameters(::AbstractRNG, ::P36Affine) = (a = [1.2], b = [0.3])
LuxCore.initialstates(::AbstractRNG, ::P36Affine) = (;)
LenticulumCore.energyspace(::P36Affine) = ScalarEnergySpace()
LenticulumCore.scalarisation(::P36Affine) = IdentityScalarisation()
LenticulumCore.pointwise_energyspace(::P36Affine) = ScalarEnergySpace()
LenticulumCore.pointwise_scalarisation(::P36Affine) = IdentityScalarisation()
function Mycelium.factor_message(f::P36Affine, target::Symbol, pol, inputs, prior, ps, st)
    a, b = only(ps.a), only(ps.b)
    if target === :y
        x = only(inputs.x.value)
        return Gaussian(a*x + b, f.q), st
    end
    y = only(inputs.y.value)
    return GaussianBelief([a*(y-b)/f.q], fill(a^2/f.q, 1, 1)), st
end
Mycelium.local_free_energy(f::P36Affine, msgs, ps, st) =
    (((only(msgs.y.value)-only(ps.a)*only(msgs.x.value)-only(ps.b))^2/f.q + log(2π*f.q))/2, st)
Mycelium.supports_parameter_polarity(::P36Affine{true}, p) =
    p[:a] isa Unobserved && p[:x] isa Observed && p[:y] isa Observed
function Mycelium.parameter_message(f::P36Affine{true}, target, p, msgs, ps, st)
    x, y, b = only(msgs.x.value), only(msgs.y.value), only(ps.b)
    return GaussianBelief([x*(y-b)/f.q], fill(x^2/f.q, 1, 1)), st
end
function Mycelium.parameter_free_energy(f::P36Affine{true}, msgs, ps, st)
    x, y, b = only(msgs.x.value), only(msgs.y.value), only(ps.b)
    μ, v = only(belief_mean(msgs.a)), only(belief_cov(msgs.a))
    return ((y-b-x*μ)^2 + x^2*v)/ (2f.q) + log(2π*f.q)/2, st
end
