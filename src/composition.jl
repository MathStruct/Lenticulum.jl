# Exact marginalisation of the internal Gaussian channel. This is a collapsed game,
# not the graded path game of compose_free_energy (which retains the internal channel).
const GaussianComposite = LenticulumCore.ComposedFactor{<:GaussianFactor,<:GaussianFactor}

function _check_gaussian_composite(f::GaussianComposite)
    a, b = f.first, f.second
    a.nout == b.nin || throw(DimensionMismatch("Gaussian composition has incompatible intermediate dimensions"))
    a.out_channel === b.in_channel || throw(ArgumentError("Gaussian composition must share the intermediate channel name"))
    a.in_channel !== b.out_channel || throw(ArgumentError("Gaussian composite endpoint channel names must differ"))
    for part in (a, b)
        part.noise_learning === nothing || part.noise_learning.prior === nothing ||
            throw(ArgumentError("collapsed Gaussian games do not marginalise parameter hyperpriors; use components without noise_prior"))
    end
    return nothing
end

function _collapsed_gaussian(f::GaussianComposite, ps)
    _check_gaussian_composite(f)
    a, b = f.first, f.second
    A, B = ps.first.A, ps.second.A
    Q = B * noise_covariance(a, ps.first) * B' + noise_covariance(b, ps.second)
    g = GaussianFactor(a.nin => b.nout; noise=Q, channels=(a.in_channel,b.out_channel))
    return g, (A=B*A, b=B*ps.first.b+ps.second.b)
end

LenticulumCore.channels(f::GaussianComposite) = (_check_gaussian_composite(f);
    (first(LenticulumCore.channels(f.first)), last(LenticulumCore.channels(f.second))))
LenticulumCore.energyspace(f::GaussianComposite) = LenticulumCore.energyspace(f.first)
LenticulumCore.scalarisation(f::GaussianComposite) = LenticulumCore.scalarisation(f.first)
LenticulumCore.pointwise_energyspace(f::GaussianComposite) = LenticulumCore.EuclideanEnergySpace(f.second.nout)
# Q depends on parameters, so scalarisation is localised at the call.
struct ComposedGaussianNorm <: LenticulumCore.AbstractPointScalarisation end
LenticulumCore.pointwise_scalarisation(::GaussianComposite) = ComposedGaussianNorm()
function LenticulumCore.localise(::ComposedGaussianNorm, f::GaussianComposite, x, a, y, ps, st)
    g, _ = _collapsed_gaussian(f, ps)
    return LenticulumCore.pointwise_scalarisation(g)
end
function LenticulumCore.supported_polarities(f::GaussianComposite)
    names = Tuple(LenticulumCore.channelname(c) for c in LenticulumCore.channels(f))
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    return (LenticulumCore.Polarity(NamedTuple{names}((O,U))),
            LenticulumCore.Polarity(NamedTuple{names}((U,O))))
end
function LenticulumCore.supports_polarity(f::GaussianComposite, p::LenticulumCore.Polarity)
    names = Tuple(LenticulumCore.channelname(c) for c in LenticulumCore.channels(f))
    keys(p) == names || return false
    x,y = p[names[1]],p[names[2]]
    O,U = LenticulumCore.Observed,LenticulumCore.Unobserved
    return (x isa O && y isa U) || (x isa U && y isa O)
end

struct ComposedGaussianModel{F,P} <: LenticulumCore.AbstractOpenModel
    factor::F
    polarity::P
end
LenticulumCore.ispure(::ComposedGaussianModel) = true
LenticulumCore.latentspace(::ComposedGaussianModel) = nothing
function LenticulumCore.assemble(f::GaussianComposite, p::LenticulumCore.Polarity, ps, st)
    LenticulumCore.supports_polarity(f,p) || throw(ArgumentError("unsupported Gaussian composite polarity"))
    _collapsed_gaussian(f, ps)
    return LenticulumCore.BayesianLens(ComposedGaussianModel(f,p), LenticulumCore.ExactInversion()), st
end
for op in (:invert, :likelihood_message)
    @eval function LenticulumCore.$op(lens::LenticulumCore.BayesianLens{<:ComposedGaussianModel}, π, inputs, ps, st)
        g, gp = _collapsed_gaussian(lens.model.factor, ps)
        gl, _ = LenticulumCore.assemble(g, lens.model.polarity, gp, (;))
        result, _ = LenticulumCore.$op(gl, π, inputs, gp, (;))
        return result, st
    end
end
function Mycelium.factor_message(f::GaussianComposite, target::Symbol, p, inputs, prior, ps, st)
    g, gp = _collapsed_gaussian(f, ps)
    result, _ = Mycelium.factor_message(g,target,p,inputs,prior,gp,(;))
    return result, st
end
function Mycelium.local_free_energy(f::GaussianComposite, msgs, ps, st)
    g, gp = _collapsed_gaussian(f, ps)
    result, _ = Mycelium.local_free_energy(g,msgs,gp,(;))
    return result, st
end
Mycelium.uses_cavity(::GaussianComposite) = true
for op in (:free_energy, :entropy)
    @eval function LenticulumCore.$op(f::GaussianComposite, π, y, ps, st)
        g, gp = _collapsed_gaussian(f, ps)
        result, _ = LenticulumCore.$op(g,π,y,gp,(;))
        return result, st
    end
end
function LenticulumCore.energy(f::GaussianComposite, x, a, y, ps, st)
    g, gp = _collapsed_gaussian(f, ps)
    result, _ = LenticulumCore.energy(g,x,a,y,gp,(;))
    return result, st
end
function LenticulumCore.pushforward(f::GaussianFactor, π::Union{GaussianBelief,LenticulumCore.DiracBelief}, ps, st)
    m, S = _moments(π,f.nin)
    length(m) == f.nin || throw(DimensionMismatch("Gaussian pushforward input dimension"))
    return Gaussian(ps.A*m+ps.b, ps.A*S*ps.A'+noise_covariance(f,ps)), st
end
LenticulumCore.pushforward(::GaussianFactor, π, ps, st) = throw(ArgumentError(
    "exact Gaussian pushforward requires a proper Gaussian or Dirac input"))
function LenticulumCore.pushforward(f::GaussianComposite, π, ps, st)
    g, gp = _collapsed_gaussian(f,ps)
    result, _ = LenticulumCore.pushforward(g,π,gp,(;))
    return result, st
end
LenticulumCore.exact_product_game(::Union{GaussianFactor,GaussianPrior,GaussianComposite}) = true
