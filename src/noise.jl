# Opt-in covariance parameters shared by the Gaussian conditional and source.

"""
    WishartPrecisionPrior(df, scale)

Wishart density on the precision, with degrees of freedom `df` and SPD scale matrix
`scale` (mean precision `df * scale`). Its negative log density is charged once per
local free-energy evaluation, with no parameter-coordinate Jacobian. Thus MAP refers
to density in precision coordinates. Requires `df > dimension - 1`.
"""
struct WishartPrecisionPrior{T,M}
    df::T
    scale::M
    function WishartPrecisionPrior(df::Real, scale::AbstractMatrix)
        V = Matrix(float.(scale))
        d = size(V, 1)
        size(V, 2) == d && issymmetric(V) && isposdef(Symmetric(V)) ||
            throw(ArgumentError("Wishart scale must be symmetric positive definite"))
        isfinite(df) && df > d - 1 || throw(ArgumentError("Wishart df must exceed dimension - 1"))
        new{typeof(float(df)),typeof(V)}(float(df), V)
    end
end

"""
    LogNormalVariancePrior(logmean, logstd)

Independent log-normal densities on diagonal variances. Arguments are scalars or
vectors of the noise dimension; `logstd` must be positive. The negative log density
(including the log-variance term) is charged in free energy. MAP is in variance
coordinates, without a Jacobian for the trainable log-precision coordinates.
"""
struct LogNormalVariancePrior{M,S}
    logmean::M
    logstd::S
    function LogNormalVariancePrior(logmean, logstd)
        all(isfinite, logmean) && all(x -> isfinite(x) && x > 0, logstd) ||
            throw(ArgumentError("log-normal parameters must be finite, with logstd > 0"))
        new{typeof(logmean),typeof(logstd)}(logmean, logstd)
    end
end

struct NoiseLearning{P,T}
    diagonal::Bool
    prior::P
    floor::T
end

function _noise_learning(Q, mode, prior, floor)
    if mode === false
        prior === nothing || throw(ArgumentError("noise_prior requires learn_noise"))
        return nothing
    end
    (mode === true || mode === :full || mode === :diagonal) ||
        throw(ArgumentError("learn_noise must be false, true, :full or :diagonal"))
    isfinite(floor) && floor > 0 || throw(ArgumentError("eigenvalue_floor must be finite and positive"))
    diagonal = mode === :diagonal
    d = size(Q, 1)
    diagonal && !isdiag(Q) && throw(ArgumentError("diagonal learning requires a diagonal initial covariance"))
    if prior isa WishartPrecisionPrior
        size(prior.scale) == size(Q) || throw(DimensionMismatch("Wishart scale and covariance"))
        diagonal && throw(ArgumentError("use LogNormalVariancePrior for diagonal noise"))
    elseif prior isa LogNormalVariancePrior
        diagonal || throw(ArgumentError("LogNormalVariancePrior requires diagonal noise"))
        for p in (prior.logmean, prior.logstd)
            p isa Real || length(p) == d || throw(DimensionMismatch("log-normal prior and covariance"))
        end
    elseif prior !== nothing
        throw(ArgumentError("noise_prior must be WishartPrecisionPrior or LogNormalVariancePrior"))
    end
    prior === nothing && eigmin(Symmetric(Q)) < floor &&
        throw(ArgumentError("initial covariance must respect eigenvalue_floor"))
    return NoiseLearning(diagonal, prior, float(floor))
end

_noise_parameters(::Nothing, Q) = (;)
function _noise_parameters(c::NoiseLearning, Q)
    if c.diagonal
        return (noise = (log_precision = -log.(diag(Q)),),)
    end
    L = cholesky(Symmetric(inv(Symmetric(Q)))).L
    d = size(Q, 1)
    return (noise = (log_diagonal = log.(diag(L)),
                     lower = [L[i, j] for j in 1:d for i in (j + 1):d]),)
end
_noise_parameterlength(::Nothing, d) = 0
_noise_parameterlength(c::NoiseLearning, d) = c.diagonal ? d : d * (d + 1) ÷ 2

# Spectral clipping acts on precision: covariance eigenvalues are at least floor.
# The optional AD extensions apply its divided-difference Frechet derivative.
function _clip_precision(P, cap)
    E = eigen(Symmetric(P))
    maximum(E.values) < cap && return P
    return E.vectors * Diagonal(min.(E.values, cap)) * E.vectors'
end

function _clip_frechet(values, cap)
    f = min.(values, cap)
    return [values[i] == values[j] ? (values[i] < cap ? one(cap) : zero(cap)) :
            (f[i] - f[j]) / (values[i] - values[j]) for i in eachindex(values), j in eachindex(values)]
end

# Fixed-noise callers keep their original normaliser calculation; the second
# component is an unused placeholder on this branch.
_noise_metrics(::Nothing, Q, ps) = (inv(Symmetric(Q)), zero(eltype(Q)))
function _noise_metrics(c::NoiseLearning, Q, ps)
    p = ps.noise
    if c.diagonal
        logs = c.prior === nothing ? min.(p.log_precision, -log(c.floor)) : p.log_precision
        return Diagonal(exp.(logs)), -sum(logs)
    end
    d = length(p.log_diagonal)
    # Packed strict lower triangle, column-major. No mutation: both AD paths see it.
    L = [i == j ? exp(p.log_diagonal[i]) : i > j ?
         p.lower[(j - 1) * d - j * (j - 1) ÷ 2 + i - j] : zero(eltype(p.log_diagonal))
         for i in 1:d, j in 1:d]
    raw = L * L'
    if c.prior === nothing
        P = _clip_precision(raw, inv(c.floor))
        return P, P === raw ? -2sum(p.log_diagonal) : -logdet(Matrix(P))
    end
    return raw, -2sum(p.log_diagonal)
end

_noise_penalty(::Nothing, P, logdetQ) = zero(logdetQ)
function _noise_penalty(p::WishartPrecisionPrior, P, logdetQ)
    d, ν = size(P, 1), p.df
    loggamma_d = d * (d - 1) / 4 * log(π) +
                 sum(SpecialFunctions.loggamma((ν + 1 - j) / 2) for j in 1:d)
    return (tr(p.scale \ P) + (ν - d - 1) * logdetQ + ν * d * log(2) +
            ν * logdet(p.scale)) / 2 + loggamma_d
end
function _noise_penalty(p::LogNormalVariancePrior, P, logdetQ)
    logvar = -log.(diag(P))
    return sum(logvar .+ ((logvar .- p.logmean) ./ p.logstd).^2 ./ 2 .+
               log.(p.logstd) .+ log(2π) / 2)
end

struct LearnedPrecisionNorm <: LenticulumCore.AbstractPointScalarisation end
function LenticulumCore.localise(::LearnedPrecisionNorm, f, x, a, y, ps, st)
    P, _ = _noise_metrics(f.noise_learning, f.Q, ps)
    return LenticulumCore.SquaredNorm(P; check = false)
end
