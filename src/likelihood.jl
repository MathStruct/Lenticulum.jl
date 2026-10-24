# ---------------------------------------------------------------------------
# A latent Gaussian observed through a non-Gaussian likelihood:  p(y | x) = L(y | Ax + b)
#
# Counts, ordered categories, binary outcomes: y is not Gaussian, and its likelihood is not a
# Gaussian in x, so the message to x is approximated by a Gaussian. Both rules work on the
# cavity (the other messages into x) and on the linear predictor z = Ax + b, which is where the
# likelihood lives:
#
#   Laplace()                  the mode and curvature of cavity × likelihood          (Laplace)
#   ExpectationPropagation()   its mean and covariance, by adaptive Gauss–Hermite      (EP)
#
# and the message is the Gaussian site "approximate posterior ÷ cavity". Forwards, the factor
# pushes a Gaussian belief on x through the likelihood into a CategoricalBelief over a grid of
# outcomes (the predictive distribution).
#
# See `likelihood.md` and `Belief Algebra.md` §4.
# ---------------------------------------------------------------------------

# --- Likelihoods ----------------------------------------------------------------------

"""
    AbstractLikelihood

A likelihood ``L(y \\mid z)`` of an observation `y` (a vector of length [`obsdim`](@ref)) given
a linear predictor `z` (a vector of length [`npredictors`](@ref)). A new likelihood defines

- `npredictors(L)`, `obsdim(L)`;
- `loglikelihood(L, y, z)`;
- optionally `loglikelihood_derivatives(L, y, z) -> (ℓ, ∇ℓ, ∇²ℓ)` in `z` (default: central
  differences of `loglikelihood`);
- for the predictive (the forward message), `outcomes(L)`, the finite list of values of `y`,
  and optionally `outcome_logprobs(L, z)`, their log-probabilities given `z` (default:
  `loglikelihood` at each outcome).

Every factor rule then works for it: see [`LikelihoodFactor`](@ref).
"""
abstract type AbstractLikelihood end

"""
    npredictors(L) -> Int

The length of the linear predictor `z` the likelihood reads.
"""
function npredictors end

"""
    obsdim(L) -> Int

The length of an observation `y`.
"""
function obsdim end

"""
    loglikelihood(L, y, z) -> Real

``\\log L(y \\mid z)``.
"""
function loglikelihood end

"""
    loglikelihood_derivatives(L, y, z) -> (ℓ, g, H)

``\\ell = \\log L(y \\mid z)``, its gradient ``g`` and Hessian ``H`` in `z`. The default uses central
differences of [`loglikelihood`](@ref); the likelihoods here give them in closed form.
"""
function loglikelihood_derivatives(L::AbstractLikelihood, y, z)
    k = length(z)
    ℓ0 = loglikelihood(L, y, z)
    grad(z) = begin
        g = zeros(k)
        for i in 1:k
            h = 1e-5 * (1 + abs(z[i]))
            e = zeros(k); e[i] = h
            g[i] = (loglikelihood(L, y, z .+ e) - loglikelihood(L, y, z .- e)) / (2h)
        end
        g
    end
    H = zeros(k, k)
    for i in 1:k
        h = 1e-4 * (1 + abs(z[i]))
        e = zeros(k); e[i] = h
        H[:, i] = (grad(z .+ e) .- grad(z .- e)) ./ (2h)
    end
    return ℓ0, grad(z), (H .+ H') ./ 2
end

"""
    outcomes(L) -> Vector

The values of `y` the predictive is a distribution over, in order. For count likelihoods the
largest value of each coordinate stands for "that many or more", so the predictive keeps the
whole mass.
"""
function outcomes end

"""
    outcome_logprobs(L, z) -> Vector

``\\log P(y = o \\mid z)`` for each `o` in [`outcomes`](@ref)`(L)`.
"""
outcome_logprobs(L::AbstractLikelihood, z) = [loglikelihood(L, o, z) for o in outcomes(L)]

_logfact(n) = SpecialFunctions.logfactorial(n)
_count(y) = (isinteger(y) && y ≥ 0) ? Int(y) :
    throw(ArgumentError("a count must be a non-negative integer, got $y"))
_mul_count(k::Integer, z::Real) = k == 0 ? zero(float(z)) : k * z

# log P(Y = j) for j = 0 … N-1 and log P(Y ≥ N), Y ~ Poisson(e^z)
function _poisson_lumped(z, N)
    λ = exp(z)
    lp = [j * z - λ - _logfact(j) for j in 0:(N - 1)]
    tail = 1 - sum(exp, lp; init = 0.0)
    # the upper tail directly when it is small, so it is not lost to cancellation
    t = tail > 1e-3 ? log(tail) : log(max(SpecialFunctions.gamma_inc(float(N), λ)[1], floatmin()))
    return vcat(lp, t)
end

"""
    PoissonLikelihood(k = 1; maxcount = 20)

``k`` independent counts, ``y_j \\sim \\mathrm{Poisson}(e^{z_j})``: the log link, one predictor per
count. Log-concave in `z`, so its Laplace and EP messages always have a positive semidefinite
precision. The predictive is over ``\\{0, \\dots, \\texttt{maxcount}\\}^k``, the top value meaning
"`maxcount` or more".
"""
struct PoissonLikelihood <: AbstractLikelihood
    k::Int
    maxcount::Int
end
function PoissonLikelihood(k::Integer = 1; maxcount::Integer = 20)
    k ≥ 1 || throw(ArgumentError("need at least one count"))
    maxcount ≥ 1 || throw(ArgumentError("maxcount must be at least 1"))
    return PoissonLikelihood(k, maxcount)
end
npredictors(L::PoissonLikelihood) = L.k
obsdim(L::PoissonLikelihood) = L.k
loglikelihood(L::PoissonLikelihood, y, z) =
    sum(_mul_count(_count(y[j]), z[j]) - exp(z[j]) - _logfact(_count(y[j])) for j in 1:L.k)
function loglikelihood_derivatives(L::PoissonLikelihood, y, z)
    λ = exp.(z)
    n = _count.(y)
    return sum(_mul_count.(n, z) .- λ .- _logfact.(n)), n .- λ, Matrix(Diagonal(-λ))
end
outcomes(L::PoissonLikelihood) =
    vec([collect(float.(Tuple(c)) .- 1) for c in CartesianIndices(ntuple(_ -> L.maxcount + 1, L.k))])
function outcome_logprobs(L::PoissonLikelihood, z)
    per = [_poisson_lumped(z[j], L.maxcount) for j in 1:L.k]
    return vec([sum(per[j][c[j]] for j in 1:L.k) for c in CartesianIndices(ntuple(_ -> L.maxcount + 1, L.k))])
end

"""
    BivariatePoissonLikelihood(; maxcount = 20)

Two dependent counts by a common shock: ``y_1 = w_1 + w_3``, ``y_2 = w_2 + w_3`` with
``w_i \\sim \\mathrm{Poisson}(\\lambda_i)`` independent, so ``\\operatorname{Cov}(y_1, y_2) = \\lambda_3``. The predictor is
``z = (\\log\\lambda_1, \\log\\lambda_2, \\log\\lambda_3)``. Derivatives are exact, through the posterior of
the shared part ``w_3`` given ``y``. Not log-concave in general, so a message's precision can be
indefinite. The predictive is over ``\\{0, \\dots, \\texttt{maxcount}\\}^2``, the top value meaning "or
more".
"""
struct BivariatePoissonLikelihood <: AbstractLikelihood
    maxcount::Int
end
function BivariatePoissonLikelihood(; maxcount::Integer = 20)
    maxcount ≥ 1 || throw(ArgumentError("maxcount must be at least 1"))
    return BivariatePoissonLikelihood(maxcount)
end
npredictors(::BivariatePoissonLikelihood) = 3
obsdim(::BivariatePoissonLikelihood) = 2

# log-weights of w₃ = j given y, without the common factor e^{-(λ₁+λ₂+λ₃)}
_bp_terms(n1, n2, z) = [_mul_count(n1 - j, z[1]) + _mul_count(n2 - j, z[2]) + _mul_count(j, z[3]) -
                        _logfact(n1 - j) - _logfact(n2 - j) - _logfact(j) for j in 0:min(n1, n2)]

function _logsumexp(v)
    m = maximum(v)
    return m + log(sum(x -> exp(x - m), v))
end

loglikelihood(::BivariatePoissonLikelihood, y, z) =
    _logsumexp(_bp_terms(_count(y[1]), _count(y[2]), z)) - sum(exp, z)
function loglikelihood_derivatives(::BivariatePoissonLikelihood, y, z)
    n1, n2 = _count(y[1]), _count(y[2])
    t = _bp_terms(n1, n2, z)
    lse = _logsumexp(t)
    p = exp.(t .- lse)                          # posterior of w₃ = 0 … min(y)
    j = 0:min(n1, n2)
    e3 = sum(p .* j)
    v = sum(p .* (j .- e3) .^ 2)
    λ = exp.(z)
    Ew = [n1 - e3, n2 - e3, e3]                 # E[w | y]; ∂ℓ/∂zᵢ = E[wᵢ | y] - λᵢ
    s = [-1.0, -1.0, 1.0]                       # w = (n1, n2, 0) + s·w₃, so Cov(w | y) = v s sᵀ
    return lse - sum(λ), Ew .- λ, v .* (s * s') .- Diagonal(λ)
end
outcomes(L::BivariatePoissonLikelihood) =
    vec([[float(c[1] - 1), float(c[2] - 1)] for c in CartesianIndices((L.maxcount + 1, L.maxcount + 1))])
function outcome_logprobs(L::BivariatePoissonLikelihood, z)
    N = L.maxcount
    P = [exp(loglikelihood(L, (i, j), z)) for i in 0:(N - 1), j in 0:(N - 1)]
    m1 = exp.(_poisson_lumped(log(exp(z[1]) + exp(z[3])), N))   # y₁ ~ Poisson(λ₁ + λ₃)
    m2 = exp.(_poisson_lumped(log(exp(z[2]) + exp(z[3])), N))
    Q = zeros(N + 1, N + 1)
    Q[1:N, 1:N] .= P
    Q[1:N, N + 1] .= max.(m1[1:N] .- vec(sum(P; dims = 2)), 0.0)   # y₁ = i, y₂ ≥ N
    Q[N + 1, 1:N] .= max.(m2[1:N] .- vec(sum(P; dims = 1)), 0.0)   # y₁ ≥ N, y₂ = j
    Q[N + 1, N + 1] = max(1 - sum(Q), 0.0)
    return vec(log.(Q))
end

"""
    OrdinalProbitLikelihood(cutpoints; scale = 1)

An ordered category ``y \\in \\{1, \\dots, K\\}`` from one predictor: ``y = k`` when
``z + \\varepsilon \\in (c_{k-1}, c_k]``, ``\\varepsilon \\sim \\mathcal N(0, \\sigma^2)``, with ``c_0 = -\\infty`` and
``c_K = +\\infty``:

```math
P(y = k \\mid z) = \\Phi\\Bigl(\\frac{c_k - z}{\\sigma}\\Bigr) - \\Phi\\Bigl(\\frac{c_{k-1} - z}{\\sigma}\\Bigr).
```

`cutpoints` are ``c_1 < \\dots < c_{K-1}``; one cutpoint is binary probit. Log-concave. The EP moments
and the predictive are in closed form.
"""
struct OrdinalProbitLikelihood <: AbstractLikelihood
    cutpoints::Vector{Float64}
    scale::Float64
end
function OrdinalProbitLikelihood(cutpoints::AbstractVector{<:Real}; scale::Real = 1.0)
    isempty(cutpoints) && throw(ArgumentError("need at least one cutpoint"))
    issorted(cutpoints; lt = ≤) || throw(ArgumentError("cutpoints must be strictly increasing"))
    scale > 0 || throw(ArgumentError("scale must be positive"))
    return OrdinalProbitLikelihood(collect(float.(cutpoints)), float(scale))
end
npredictors(::OrdinalProbitLikelihood) = 1
obsdim(::OrdinalProbitLikelihood) = 1
outcomes(L::OrdinalProbitLikelihood) = [[float(k)] for k in 1:(length(L.cutpoints) + 1)]

function _category(L::OrdinalProbitLikelihood, y)
    k = y[1]
    K = length(L.cutpoints) + 1
    (isinteger(k) && 1 ≤ k ≤ K) || throw(ArgumentError("an ordinal observation must be one of 1, …, $K, got $k"))
    return Int(k)
end
_upper(L::OrdinalProbitLikelihood, k) = k > length(L.cutpoints) ? Inf : L.cutpoints[k]
_lower(L::OrdinalProbitLikelihood, k) = k == 1 ? -Inf : L.cutpoints[k - 1]

const _SQRT2 = sqrt(2.0)
_logϕ(u) = isinf(u) ? -Inf : -u^2 / 2 - log(2π) / 2
# log Φ(u), stable far into the lower tail
_logΦ(u) = u == Inf ? 0.0 : u == -Inf ? -Inf :
    u < -5 ? log(SpecialFunctions.erfcx(-u / _SQRT2) / 2) - u^2 / 2 : log(SpecialFunctions.erfc(-u / _SQRT2) / 2)
# log(Φ(a) - Φ(b)) for a > b, without cancellation in either tail
function _logΦdiff(a, b)
    a > b || throw(ArgumentError("need a > b"))
    a ≤ 0 && return _logΦ(a) + log1p(-exp(_logΦ(b) - _logΦ(a)))
    b ≥ 0 && return _logΦ(-b) + log1p(-exp(_logΦ(-a) - _logΦ(-b)))
    return log(SpecialFunctions.erfc(-a / _SQRT2) / 2 - SpecialFunctions.erfc(-b / _SQRT2) / 2)
end
_uϕ(u, logZ) = isinf(u) ? 0.0 : u * exp(_logϕ(u) - logZ)   # u φ(u) / Z, 0 at ±∞

function loglikelihood(L::OrdinalProbitLikelihood, y, z)
    k = _category(L, y)
    σ = L.scale
    return _logΦdiff((_upper(L, k) - z[1]) / σ, (_lower(L, k) - z[1]) / σ)
end
function loglikelihood_derivatives(L::OrdinalProbitLikelihood, y, z)
    k = _category(L, y)
    σ = L.scale
    a, b = (_upper(L, k) - z[1]) / σ, (_lower(L, k) - z[1]) / σ
    lZ = _logΦdiff(a, b)
    ra, rb = exp(_logϕ(a) - lZ), exp(_logϕ(b) - lZ)
    g = -(ra - rb) / σ
    h = -(_uϕ(a, lZ) - _uϕ(b, lZ)) / σ^2 - g^2
    return lZ, [g], fill(h, 1, 1)
end
outcome_logprobs(L::OrdinalProbitLikelihood, z) = _ordinal_logprobs(L, z[1], L.scale)
_ordinal_logprobs(L, μ, t) =
    [_logΦdiff((_upper(L, k) - μ) / t, (_lower(L, k) - μ) / t) for k in 1:(length(L.cutpoints) + 1)]

# --- The approximation methods --------------------------------------------------------

"""
    Laplace(; iterations = 100, tol = 1e-10)

Approximate the posterior ``\\propto`` cavity × likelihood by the Gaussian at its mode with the
curvature there (Newton's method, at most `iterations` steps, stopping when a step is below
`tol`). The message is that Gaussian divided by the cavity: precision ``-A^\\top\\nabla^2\\ell\\,A`` at the
mode. Works for any predictor dimension.
"""
Base.@kwdef struct Laplace
    iterations::Int = 100
    tol::Float64 = 1e-10
end

"""
    ExpectationPropagation(; order = 20, iterations = 100, tol = 1e-10)

Approximate the posterior ``\\propto`` cavity × likelihood by the Gaussian with its mean and
covariance (moment matching: the minimiser of ``\\mathrm{KL}(p \\Vert q)``, the projection step of EP).
The moments are computed by adaptive Gauss–Hermite quadrature, `order` points per predictor
dimension, centred and scaled at the Laplace approximation (found as for [`Laplace`](@ref)
with `iterations` and `tol`); in closed form for [`OrdinalProbitLikelihood`](@ref). The tensor grid has
``\\texttt{order}^k`` points, so this is for few predictors (``k \\le 4``).
"""
Base.@kwdef struct ExpectationPropagation
    order::Int = 20
    iterations::Int = 100
    tol::Float64 = 1e-10
end

# probabilists' Gauss–Hermite: nodes and weights for E f(ξ), ξ ~ N(0, 1) (Golub–Welsch)
function _gauss_hermite(n::Integer)
    n == 1 && return [0.0], [1.0]
    J = LinearAlgebra.SymTridiagonal(zeros(n), sqrt.(1.0:(n - 1)))
    E = LinearAlgebra.eigen(J)
    return E.values, E.vectors[1, :] .^ 2
end

# tensor grid in r dimensions: (r × N nodes, N weights)
function _gh_grid(order, r)
    order^r ≤ 10^6 || throw(ArgumentError("a Gauss–Hermite grid of $order^$r points is too large; " *
                                         "use Laplace(), or a smaller order"))
    ξ, w = _gauss_hermite(order)
    idx = CartesianIndices(ntuple(_ -> order, r))
    X = reduce(hcat, ([ξ[c[i]] for i in 1:r] for c in vec(collect(idx))); init = zeros(r, 0))
    W = [prod((w[c[i]] for i in 1:r); init = 1.0) for c in vec(collect(idx))]
    return X, W
end

# --- The factor ---------------------------------------------------------------------

"""
    LikelihoodFactor(lik, nin; A = nothing, b = nothing, channels = (:x, :y),
                     method = Laplace(), x0 = nothing)

A Gaussian latent ``x \\in \\mathbb R^{n}`` observed through a non-Gaussian likelihood of the linear
predictor:

```math
p(y \\mid x) = L(y \\mid Ax + b),
```

`lik` an [`AbstractLikelihood`](@ref) ([`PoissonLikelihood`](@ref), [`BivariatePoissonLikelihood`](@ref),
[`OrdinalProbitLikelihood`](@ref), or one of your own). `A` is `npredictors(lik) × nin` (default
the identity when the sizes agree) and `b` the offset (default zero); both fixed.

- **Backwards** (target ``x``, ``y`` observed as a `DiracBelief`): the Gaussian message
  ``q / \\text{cavity}``, where the cavity is the `prior` argument (the other messages into ``x``)
  and ``q`` its product with the likelihood, approximated by `method`: [`Laplace`](@ref) or
  [`ExpectationPropagation`](@ref). With no proper cavity, the second-order expansion of
  ``\\log L`` at `x0` (one Newton step from there), or an uninformative message without `x0`.
- **Forwards** (target ``y``): the predictive ``P(y) = \\mathbb E[L(y \\mid Ax+b)]`` under the belief on
  ``x``, a `CategoricalBelief` over [`outcomes`](@ref)`(lik)` (labelled by them), by Gauss–Hermite
  quadrature (`method.order` points per dimension for EP, 20 for Laplace).

No parameters (`A`, `b` are fixed). See `likelihood.md`.
"""
struct LikelihoodFactor{L<:AbstractLikelihood,M,X} <: LenticulumCore.AbstractLenticulumFactor
    lik::L
    nin::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    in_channel::Symbol
    out_channel::Symbol
    method::M
    x0::X
end

function LikelihoodFactor(lik::AbstractLikelihood, nin::Integer; A = nothing, b = nothing,
                          channels::Tuple{Symbol,Symbol} = (:x, :y), method = Laplace(), x0 = nothing)
    k = npredictors(lik)
    if A === nothing
        nin == k || throw(ArgumentError("give A ($k × $nin): the likelihood reads $k predictors from $nin latents"))
        Am = Matrix{Float64}(I, k, k)
    else
        Am = A isa Real ? fill(float(A), 1, 1) : Matrix{Float64}(A)
    end
    size(Am) == (k, nin) || throw(DimensionMismatch("A must be $k × $nin, got $(size(Am))"))
    bv = b === nothing ? zeros(k) : _vec(b)
    length(bv) == k || throw(DimensionMismatch("b must have length $k"))
    method isa Union{Laplace,ExpectationPropagation} ||
        throw(ArgumentError("method must be Laplace() or ExpectationPropagation(), got $(typeof(method))"))
    x0 === nothing || length(x0) == nin || throw(DimensionMismatch("x0 must have length $nin"))
    return LikelihoodFactor(lik, Int(nin), Am, bv, channels[1], channels[2], method,
                            x0 === nothing ? nothing : _vec(x0))
end

LuxCore.initialparameters(::AbstractRNG, ::LikelihoodFactor) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::LikelihoodFactor) = NamedTuple()
LuxCore.parameterlength(::LikelihoodFactor) = 0

LenticulumCore.channels(f::LikelihoodFactor) = (
    LenticulumCore.Channel(f.in_channel, f.nin),
    LenticulumCore.Channel(f.out_channel, obsdim(f.lik)),
)
function LenticulumCore.supported_polarities(f::LikelihoodFactor)
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    ks = (f.in_channel, f.out_channel)
    return (LenticulumCore.Polarity(NamedTuple{ks}((O, U))),
            LenticulumCore.Polarity(NamedTuple{ks}((U, O))))
end
function LenticulumCore.supports_polarity(f::LikelihoodFactor, p::LenticulumCore.Polarity)
    keys(p) == (f.in_channel, f.out_channel) || return false
    pi_, po = p[f.in_channel], p[f.out_channel]
    return (pi_ isa LenticulumCore.Observed && po isa LenticulumCore.Unobserved) ||
           (pi_ isa LenticulumCore.Unobserved && po isa LenticulumCore.Observed)
end
LenticulumCore.islearnable(::LikelihoodFactor) = false
LenticulumCore.energyspace(::LikelihoodFactor) = LenticulumCore.energyspace(GaussianFactor(1 => 1))
LenticulumCore.scalarisation(::LikelihoodFactor) = LenticulumCore.scalarisation(GaussianFactor(1 => 1))

_predictor(f::LikelihoodFactor, x) = f.A * _vec(x) .+ f.b

"""
    LenticulumCore.energy(f::LikelihoodFactor, x, a, y, ps, st)

The negative log-likelihood ``-\\log L(y \\mid Ax + b)``, as a one-element vector.
"""
LenticulumCore.energy(f::LikelihoodFactor, x, a, y, ps, st) =
    ([-loglikelihood(f.lik, _vec(y), _predictor(f, x))], st)
# pointwise: that one-element vector, summed as it is (the likelihood is normalised in y, so there
# is no separate complexity); its expectation is the `fit` summand of the free energy
LenticulumCore.pointwise_energyspace(::LikelihoodFactor) = LenticulumCore.EuclideanEnergySpace(1)
LenticulumCore.pointwise_scalarisation(::LikelihoodFactor) = LenticulumCore.WeightedSum([1.0])

# --- The tilted distribution in the predictor's coordinates -------------------------------
#
# The cavity N(m, S) on x gives z = Ax + b ~ N(μz, Σz), Σz = A S Aᵀ, possibly singular. On the
# range of Σz, z = μz + U ζ with ζ ~ N(0, D); the likelihood depends on x only through ζ, so the
# exact tilted distribution is the cavity's conditional x | ζ times a tilted distribution of ζ,
# and a Gaussian approximation of the latter, written as a site s(ζ) with cavity × s = q, is
# the Gaussian approximation of the whole. The message on x is s(Uᵀ(Ax + b - μz)).

struct _Tilted
    μz::Vector{Float64}
    U::Matrix{Float64}
    D::Vector{Float64}
end

function _tilted(f::LikelihoodFactor, m, S)
    μz = _predictor(f, m)
    Σz = Symmetric(f.A * S * f.A')
    E = LinearAlgebra.eigen(Σz)
    keep = E.values .> 1e-12 * max(maximum(E.values; init = 0.0), floatmin())
    return _Tilted(μz, E.vectors[:, keep], E.values[keep])
end

# log N(ζ; 0, D) + ℓ(y, z(ζ)), with its gradient and Hessian in ζ
function _tilted_derivatives(f, t::_Tilted, y, ζ)
    ℓ, g, H = loglikelihood_derivatives(f.lik, y, t.μz .+ t.U * ζ)
    lp = -sum(ζ .^ 2 ./ t.D) / 2 - sum(log, 2π .* t.D) / 2
    return lp + ℓ, t.U' * g .- ζ ./ t.D, t.U' * H * t.U .- Diagonal(1 ./ t.D)
end

# Newton's method with backtracking for the mode of the tilted density in ζ
function _tilted_mode(f, t::_Tilted, y, iterations, tol)
    ζ = zeros(length(t.D))
    φ, g, H = _tilted_derivatives(f, t, y, ζ)
    for _ in 1:iterations
        C = -Symmetric(H)
        all(isfinite, C) || break
        δ = 0.0
        cmax = maximum(abs, C; init = 0.0)
        while !isposdef(C + δ * I)                 # Levenberg damping where the density is not log-concave
            δ = max(2δ, 1e-8 * (1 + cmax))
            isfinite(δ) || break
        end
        isfinite(δ) || break
        Δ = (C + δ * I) \ g
        s = 1.0
        φn, gn, Hn = _tilted_derivatives(f, t, y, ζ .+ Δ)
        while !(isfinite(φn) && all(isfinite, gn) && all(isfinite, Hn) && φn ≥ φ - 1e-12 * abs(φ)) && s > 1e-10
            s /= 2
            φn, gn, Hn = _tilted_derivatives(f, t, y, ζ .+ s .* Δ)
        end
        (isfinite(φn) && all(isfinite, gn) && all(isfinite, Hn)) || break
        ζ = ζ .+ s .* Δ
        φ, g, H = φn, gn, Hn
        maximum(abs, s .* Δ; init = 0.0) < tol && break
    end
    isposdef(-Symmetric(H)) || throw(ArgumentError(
        "Laplace: the cavity × likelihood has no strict local maximum (its Hessian is not negative definite)"))
    return ζ, φ, Matrix(-Symmetric(H))
end

# the Gaussian approximation of the tilted ζ-distribution: (mean, covariance, log normaliser)
function _approximate(::Laplace, f, t, y)
    ζ̂, φ, C = _tilted_mode(f, t, y, f.method.iterations, f.method.tol)
    r = length(ζ̂)
    return ζ̂, inv(_chol(C, "the Laplace curvature")), φ + r * log(2π) / 2 - logdet(_chol(C, "C")) / 2
end

function _approximate(ep::ExpectationPropagation, f, t, y)
    f.lik isa OrdinalProbitLikelihood && length(t.D) == 1 && return _ordinal_moments(f.lik, t, y)
    ζ̂, _, C = _tilted_mode(f, t, y, ep.iterations, ep.tol)
    r = length(ζ̂)
    Lc = cholesky(Symmetric(inv(_chol(C, "the Laplace curvature")))).L
    X, W = _gh_grid(ep.order, r)
    Z = ζ̂ .+ Lc * X
    logv = [_tilted_derivatives(f, t, y, Z[:, i])[1] + sum(abs2, X[:, i]) / 2 + log(W[i]) for i in axes(Z, 2)]
    lse = _logsumexp(logv)
    w = exp.(logv .- lse)
    μ = Z * w
    D = Z .- μ
    Σ = (D .* w') * D'
    logZ = lse + logdet(Lc) + r * log(2π) / 2
    return μ, Matrix(Symmetric(Σ)), logZ
end

# closed-form tilted moments of N(z; μ, s²) × [Φ((c_k - z)/σ) - Φ((c_{k-1} - z)/σ)]
function _ordinal_moments(L::OrdinalProbitLikelihood, t::_Tilted, y)
    k = _category(L, y)
    μ, s², u = t.μz[1], t.D[1], t.U[1, 1]          # z = μ + u ζ, u = ±1
    τ = sqrt(L.scale^2 + s²)
    a, b = (_upper(L, k) - μ) / τ, (_lower(L, k) - μ) / τ
    lZ = _logΦdiff(a, b)
    ra, rb = exp(_logϕ(a) - lZ), exp(_logϕ(b) - lZ)
    mz = μ - s² / τ * (ra - rb)
    vz = s² - s²^2 / τ^2 * ((_uϕ(a, lZ) - _uϕ(b, lZ)) + (ra - rb)^2)
    return [(mz - μ) / u], fill(vz, 1, 1), lZ
end

# the site s(ζ) = q(ζ) / N(ζ; 0, D), mapped to x through ζ = Uᵀ(Ax + b - μz)
function _site_message(f, t::_Tilted, μζ, Σζ)
    Λq = inv(_chol(Matrix(Symmetric(Σζ)), "the approximate posterior covariance"))
    Λs = Λq .- Diagonal(1 ./ t.D)
    ηs = Λq * μζ
    B = t.U' * f.A
    c = t.U' * (f.b .- t.μz)
    Λ = B' * Λs * B
    return GaussianBelief(B' * (ηs .- Λs * c), Matrix(Symmetric(Λ)))
end

# with no cavity: the second-order expansion of ℓ at z₀ = A x₀ + b
function _expansion_message(f, y, x0)
    z0 = _predictor(f, x0)
    _, g, H = loglikelihood_derivatives(f.lik, y, z0)
    Λz = -H
    ηz = g .+ Λz * z0
    return GaussianBelief(f.A' * (ηz .- Λz * f.b), Matrix(Symmetric(f.A' * Λz * f.A)))
end

_observation(::Nothing, f) = nothing
_observation(::LenticulumCore.TrivialBelief, f) = nothing
_observation(b::LenticulumCore.DiracBelief, f) = _vec(b.value)
_observation(b, f) = throw(ArgumentError(
    "LikelihoodFactor: the observation :$(f.out_channel) must be a DiracBelief, got $(typeof(b))"))

# --- Messages -----------------------------------------------------------------------

"""
    Mycelium.factor_message(f::LikelihoodFactor, target, polarity, inputs, prior, ps, st)

Backwards (target ``x``): the Gaussian site of [`LikelihoodFactor`](@ref)'s `method` at the cavity
`prior`. The cavity is used to place the approximation, never multiplied in, so nothing is
counted twice. Forwards (target ``y``): the predictive `CategoricalBelief`.
"""
Mycelium.factor_message(f::LikelihoodFactor, target::Symbol, polarity, inputs, prior, ps, st) =
    (_lik_message(f, target, inputs, prior), st)

function _lik_message(f::LikelihoodFactor, target::Symbol, inputs, prior)
    if target === f.in_channel
        y = _observation(_get(inputs, f.out_channel), f)
        y === nothing && return LenticulumCore.TrivialBelief()
        m, S = _cavity_moments(prior, f.nin)
        if m === nothing
            f.x0 === nothing && return LenticulumCore.TrivialBelief()
            return _expansion_message(f, y, f.x0)
        end
        t = _tilted(f, m, S)
        isempty(t.D) && return uninformative(f.nin)   # the cavity fixes z: y says nothing more about x
        μζ, Σζ, _ = _approximate(f.method, f, t, y)
        return _site_message(f, t, μζ, Σζ)
    elseif target === f.out_channel
        m, S = _moments(_get(inputs, f.in_channel), f.nin)
        m === nothing && return LenticulumCore.TrivialBelief()
        return predictive(f, _gaussian_or_point(m, S))
    end
    throw(ArgumentError("channel :$target is not a channel of this LikelihoodFactor"))
end

_gaussian_or_point(m, S) = all(iszero, S) ? LenticulumCore.DiracBelief(m) : Gaussian(m, S)

# a cavity is usable when it is proper; an improper or absent one counts as none
function _cavity_moments(b, n)
    b isa GaussianBelief && !isproper(b) && return (nothing, nothing)
    return _moments(b, n)
end

"""
    predictive(f::LikelihoodFactor, belief) -> CategoricalBelief

``P(y = o) = \\mathbb E\\,L(o \\mid Ax + b)`` for each `o` in `outcomes(f.lik)`, under a `GaussianBelief`
or a `DiracBelief` on ``x``, labelled by the outcomes. Gauss–Hermite quadrature over the range of
``\\operatorname{Cov}(Ax+b)``; in closed form for [`OrdinalProbitLikelihood`](@ref) and for a point.
"""
function predictive(f::LikelihoodFactor, belief)
    m, S = _moments(belief, f.nin)
    m === nothing && throw(ArgumentError("predictive: the belief on x carries no information"))
    labels = outcomes(f.lik)
    t = _tilted(f, m, S)
    if isempty(t.D)
        lp = outcome_logprobs(f.lik, t.μz)
    elseif f.lik isa OrdinalProbitLikelihood
        lp = _ordinal_logprobs(f.lik, t.μz[1], sqrt(f.lik.scale^2 + t.D[1]))
    else
        order = f.method isa ExpectationPropagation ? f.method.order : 20
        X, W = _gh_grid(order, length(t.D))
        Zs = t.μz .+ t.U * (sqrt.(t.D) .* X)
        LP = reduce(hcat, (outcome_logprobs(f.lik, Zs[:, i]) for i in axes(Zs, 2)))
        lp = [_logsumexp(LP[j, :] .+ log.(W)) for j in axes(LP, 1)]
    end
    return LenticulumCore.CategoricalBelief(Val(:log), lp .- _logsumexp(lp), labels)
end

"""
    log_evidence(f::LikelihoodFactor, cavity, y) -> Real

``\\log \\int \\text{cavity}(x)\\, L(y \\mid Ax+b)\\, dx``, the log normaliser of the tilted
distribution, as the factor's `method` approximates it (the Laplace evidence, or the quadrature
of EP). The quantity whose gradient fits hyperparameters by evidence.
"""
function log_evidence(f::LikelihoodFactor, cavity, y)
    m, S = _cavity_moments(cavity, f.nin)
    m === nothing && throw(ArgumentError("log_evidence needs a proper cavity belief"))
    t = _tilted(f, m, S)
    isempty(t.D) && return loglikelihood(f.lik, _vec(y), t.μz)
    return _approximate(f.method, f, t, _vec(y))[3]
end

# --- The lens and the Bethe contribution -----------------------------------------------

"""
    LikelihoodModel(factor, polarity)

The open model of a [`LikelihoodFactor`](@ref) in the direction the polarity selects. Pure.
"""
struct LikelihoodModel{F<:LikelihoodFactor,P} <: LenticulumCore.AbstractOpenModel
    factor::F
    polarity::P
end
LenticulumCore.ispure(::LikelihoodModel) = true
LenticulumCore.latentspace(::LikelihoodModel) = nothing

LenticulumCore.assemble(f::LikelihoodFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(LikelihoodModel(f, p), LenticulumCore.SolverInversion(f.method)), st)

"""
    LenticulumCore.invert(lens, π, inputs, ps, st) -> (belief, st)

The approximate posterior ``\\pi \\cdot \\mu_{c\\to\\text{target}}``: backwards, the Laplace or EP
approximation of ``\\pi(x) L(y \\mid Ax+b)``; forwards, the predictive pooled with ``\\pi``.
"""
function LenticulumCore.likelihood_message(
    lens::LenticulumCore.BayesianLens{<:LikelihoodModel,<:LenticulumCore.SolverInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    target = only(LenticulumCore.unobserved_channels(lens.model.polarity))
    return (_lik_message(f, target, inputs, π), st)
end

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:LikelihoodModel,<:LenticulumCore.SolverInversion},
    π, inputs, ps, st,
)
    msg, st = LenticulumCore.likelihood_message(lens, π, inputs, ps, st)
    return (Mycelium.combine(π, msg), st)
end

# Only likelihoods with an exact conditional entropy may use predictive free energy.
# The count predictors lump an infinite tail; that grid is not their likelihood's support.
_conditional_entropy(L::AbstractLikelihood, z) = throw(ArgumentError(
    "exact conditional entropy is not implemented for $(typeof(L)); predictive outcome grids may lump tails"))
function _conditional_entropy(L::OrdinalProbitLikelihood, z)
    lp = outcome_logprobs(L, z)
    return -sum(l == -Inf ? zero(l) : exp(l) * l for l in lp)
end

"""
    Mycelium.local_free_energy(f::LikelihoodFactor, msgs, ps, st)

The factor's Bethe contribution, graded `(fit, complexity, negentropy)`, with the factor belief
replaced by its Gaussian approximation ``q`` (cavity = the incoming message on ``x``, times the
site): `fit` is ``\\mathbb E_q[-\\log L(y \\mid Ax+b)]`` by Gauss–Hermite quadrature, `negentropy` is
``-H(q)``, `complexity` is 0 (the likelihood is normalised in ``y``). For a point ``x`` it is
``-\\log L(y \\mid Ax+b)`` exactly. Approximate as the messages are.

With `y` unobserved, `fit = E_x[H(L(⋅|Ax+b))]` and
`negentropy = -fit - H(x)`. The conditional entropy sums all ordinal categories exactly;
its expectation over a Gaussian input uses Gauss–Hermite quadrature (an approximation,
`method.order` for EP, 20 for Laplace). Unsupported conditional entropies, including the
infinite-support count likelihoods, throw rather than use the tail-lumped predictive grid.
"""
function Mycelium.local_free_energy(f::LikelihoodFactor, msgs, ps, st)
    mx = _get(msgs, f.in_channel)
    y = _observation(_get(msgs, f.out_channel), f)
    if y === nothing
        m, S = _moments(mx, f.nin)
        m === nothing && throw(ArgumentError(
            "predictive local_free_energy needs a proper belief on :$(f.in_channel)"))
        t = _tilted(f, m, S)
        if isempty(t.D)
            fit = _conditional_entropy(f.lik, t.μz)
        else
            order = f.method isa ExpectationPropagation ? f.method.order : 20
            X, W = _gh_grid(order, length(t.D))
            fit = sum(W[i] * _conditional_entropy(f.lik,
                      t.μz .+ t.U * (sqrt.(t.D) .* X[:, i])) for i in eachindex(W))
        end
        Hx = Mycelium.variable_entropy(mx)
        return LenticulumCore.GradedEnergy((fit = fit, complexity = 0.0, negentropy = -fit - Hx)), st
    end
    if mx isa LenticulumCore.DiracBelief
        fit = -loglikelihood(f.lik, y, _predictor(f, mx.value))
        return LenticulumCore.GradedEnergy((fit = fit, complexity = 0.0, negentropy = 0.0)), st
    end
    q = Mycelium.combine(mx, _lik_message(f, f.in_channel, msgs, mx))
    q isa GaussianBelief && isproper(q) ||
        throw(ArgumentError("local_free_energy: no proper belief on :$(f.in_channel) to evaluate at"))
    m, S = belief_mean(q), belief_cov(q)
    t = _tilted(f, m, S)
    order = f.method isa ExpectationPropagation ? f.method.order : 20
    X, W = _gh_grid(order, length(t.D))
    fit = -sum(W[i] * loglikelihood(f.lik, y, t.μz .+ t.U * (sqrt.(t.D) .* X[:, i])) for i in eachindex(W))
    H = Mycelium.variable_entropy(q)
    return LenticulumCore.GradedEnergy((fit = fit, complexity = 0.0, negentropy = -H)), st
end

function LenticulumCore.free_energy(f::LikelihoodFactor, π, y, ps, st)
    return Mycelium.local_free_energy(f, _game_messages(f, π, y), ps, st)
end

function LenticulumCore.entropy(f::LikelihoodFactor, π, y, ps, st)
    F, st = LenticulumCore.free_energy(f, π, y, ps, st)
    return LenticulumCore.GradedEnergy((fit = 0.0, complexity = 0.0, negentropy = -F[:negentropy])), st
end
