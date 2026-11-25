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
    WishartPrecisionPrior(::Val{:checked}, df::T, scale::M) where {T,M} = new{T,M}(df, scale)
    function WishartPrecisionPrior(df::Real, scale::AbstractMatrix)
        V = Matrix(float.(scale))
        d = size(V, 1)
        size(V, 2) == d && issymmetric(V) && isposdef(Symmetric(V)) ||
            throw(ArgumentError("Wishart scale must be symmetric positive definite"))
        isfinite(df) && df > d - 1 || throw(ArgumentError("Wishart df must exceed dimension - 1"))
        # an Int df takes the scale's float type, so it does not widen a Float32 prior (P30 review)
        dfv = df isa Integer ? convert(eltype(V), df) : float(df)
        new{typeof(dfv),typeof(V)}(dfv, V)
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
        return (noise = (log_precision = -log.(_device_diag(Q)),),)
    end
    C = _symmetric_cholesky(_symmetric_inv(Q))
    L = C.L
    d = size(Q, 1)
    if _device_leaf(Q) === nothing
        return (noise = (log_diagonal = log.(diag(L)),
                         lower = [L[i, j] for j in 1:d for i in (j + 1):d]),)
    end
    indices = [(j-1)*d+i for j in 1:d for i in (j+1):d]
    return (noise = (log_diagonal = log.(_device_diag(L)), lower = vec(L)[indices]),)
end
_noise_parameterlength(::Nothing, d) = 0
_noise_parameterlength(c::NoiseLearning, d) = c.diagonal ? d : d * (d + 1) ÷ 2

# Spectral clipping acts on precision: covariance eigenvalues are at least floor.
# The optional AD extensions apply its divided-difference Frechet derivative.
function _clip_precision(P, cap)
    E = _eigen(Symmetric(P))
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
_noise_metrics(::Nothing, Q, ps) = (_symmetric_inv(Q), zero(eltype(Q)))
function _noise_metrics(c::NoiseLearning, Q, ps)
    p = ps.noise
    T = _compute_eltype(eltype(Q), Q, p)
    V = typeof(LenticulumCore.stop_gradient(zero(T)))
    floor = V(LenticulumCore.stop_gradient(c.floor))
    if c.diagonal
        raw = _at_eltype(T, p.log_precision)
        logs = c.prior === nothing ? min.(raw, -log(floor)) : raw
        return Diagonal(exp.(logs)), -sum(logs)
    end
    log_diagonal = _at_eltype(T, p.log_diagonal)
    lower = _at_eltype(T, p.lower)
    d = length(log_diagonal)
    # Packed strict lower triangle, column-major. No mutation: both AD paths see it.
    if _device_leaf(log_diagonal) === nothing
        L = [i == j ? exp(log_diagonal[i]) : i > j ?
             lower[(j - 1) * d - j * (j - 1) ÷ 2 + i - j] : zero(T)
             for i in 1:d, j in 1:d]
    else
        L = _matrix(Diagonal(exp.(log_diagonal)))
        indices = [(j-1)*d+i for j in 1:d for i in (j+1):d]
        vec(L)[indices] .= lower
    end
    raw = L * L'
    if c.prior === nothing
        P = _clip_precision(raw, inv(floor))
        return P, P === raw ? -2sum(log_diagonal) : -_symmetric_logdet(_matrix(P))
    end
    return raw, -2sum(log_diagonal)
end

_noise_penalty(::Nothing, P, logdetQ) = zero(logdetQ)
function _noise_penalty(p::WishartPrecisionPrior, P, logdetQ)
    T = typeof(logdetQ)
    d, ν = size(P, 1), T(p.df)
    scale = _on_device(_device_leaf(P), _at_eltype(T, p.scale))
    loggamma_d = d * (d - 1) * log(oftype(ν, π)) / 4 +
                 sum(SpecialFunctions.loggamma((ν + 1 - j) / 2) for j in 1:d)
    return (tr(_symmetric_solve(scale, P)) + (ν - d - 1) * logdetQ + ν * d * log(oftype(ν, 2)) +
            ν * _symmetric_logdet(scale)) / 2 + loggamma_d
end
function _noise_penalty(p::LogNormalVariancePrior, P, logdetQ)
    T = typeof(logdetQ)
    logmean = p.logmean isa Real ? T(p.logmean) : T.(p.logmean)
    logstd = p.logstd isa Real ? T(p.logstd) : T.(p.logstd)
    logvar = -log.(diag(P))
    return sum(logvar .+ ((logvar .- logmean) ./ logstd).^2 ./ 2 .+
               log.(logstd) .+ LenticulumCore._log2π(T) / 2)
end

struct LearnedPrecisionNorm <: LenticulumCore.AbstractPointScalarisation end
function LenticulumCore.localise(::LearnedPrecisionNorm, f, x, a, y, ps, st)
    P, _ = _noise_metrics(f.noise_learning, _covariance_at(f.Q, ps, x, y), ps)
    return LenticulumCore.SquaredNorm(P; check = false)
end

Adapt.@adapt_structure LogNormalVariancePrior
Adapt.@adapt_structure NoiseLearning

Adapt.adapt_structure(to, p::WishartPrecisionPrior) = WishartPrecisionPrior(Val(:checked), p.df, Adapt.adapt(to, p.scale))
