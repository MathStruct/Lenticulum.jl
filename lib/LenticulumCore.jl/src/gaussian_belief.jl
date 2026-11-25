# ---------------------------------------------------------------------------
# Gaussian beliefs in CANONICAL (information / natural-parameter) form.
#
#     N⁻¹(η, Λ)  ∝  exp(-½ xᵀΛx + ηᵀx),      Λ = Σ⁻¹,   η = Λμ
#
# The reason for the canonical form is one line:
#
#     combine(N⁻¹(η₁,Λ₁), N⁻¹(η₂,Λ₂))  =  N⁻¹(η₁+η₂, Λ₁+Λ₂)
#
# Pooling beliefs is ADDITION: exact, associative, commutative, and total. `TrivialBelief` is
# (0,0), the additive identity, and a `DiracBelief` is the Λ → ∞ limit — the ρ_in = ∞ hard clamp
# of `Channels and Polarity.md`.
#
# The type lives here, beside the other belief types, so that every package (the diffusion
# factors and implicit layers included) can emit Gaussians; its message rules (`combine`,
# densities, entropy, damping, mixture reduction `reduce_mixture`) are in Mycelium's
# `gaussian.jl`, its projection `moment_match` in Lenticulum's `beliefs.jl`. The moments of a
# `SampleBelief` are at the end of this file (P18).
#
# See `Gaussian Belief.md` and `Beliefs.md` in vault/Factor Graphs/.
# ---------------------------------------------------------------------------

"""
    GaussianBelief(η, Λ)
    GaussianBelief(η, Λ, improper::Bool)

A Gaussian in canonical form: information vector `η = Λμ` and precision `Λ = Σ⁻¹`.

`Λ` may be **singular or zero**, and that is not an error: a factor → variable message is a
*likelihood*, not a distribution, and a likelihood that constrains only some directions has a
rank-deficient precision. The moment form cannot represent this at all, which is the second
reason for the canonical parametrisation.

Whether a belief is proper is otherwise read off `Λ` ([`isproper`](@ref)), and a precision of
rank `r < n` is positive definite after rounding often enough. A factor that knows its message
is rank-deficient says so with `improper = true`: [`isproper`](@ref) is then `false` whatever the
rounding, and the mixture product weights it as a likelihood. The flag survives pooling only
with an uninformative belief, since a sum of improper sites can be proper.

Use [`Gaussian`](@ref) to build one from a mean and covariance.
"""
struct GaussianBelief{V<:AbstractVector,M<:AbstractMatrix} <: LenticulumCore.AbstractBelief
    η::V
    Λ::M
    improper::Bool
    function GaussianBelief(η::V, Λ::M, improper::Bool = false) where {V<:AbstractVector,M<:AbstractMatrix}
        size(Λ, 1) == size(Λ, 2) == length(η) ||
            throw(DimensionMismatch("η has length $(length(η)) but Λ is $(size(Λ))"))
        return new{V,M}(η, Λ, improper)
    end
end

"""
    Gaussian(μ, Σ)
    Gaussian(μ::Real, σ²::Real)

A Gaussian from its **moments**. Converts to canonical form immediately; `Σ` must be positive
definite.
"""
function Gaussian(μ::AbstractVector, Σ::AbstractMatrix)
    prototype = _first_device((μ, Σ))
    Λ = _spd_inv(_on_device(prototype, Σ), "Σ")
    return GaussianBelief(Λ * _on_device(prototype, μ), Λ)
end
Gaussian(μ::Real, σ²::Real) = Gaussian([float(μ)], fill(float(σ²), 1, 1))

"""
    uninformative(n)
    uninformative(::Type{T}, n)

The improper uniform belief on ``\\mathbb{R}^n``: `η = 0`, `Λ = 0`. The identity for
`Mycelium.combine`, and the canonical-form spelling of
`LenticulumCore.TrivialBelief`.
"""
uninformative(::Type{T}, n::Int) where {T<:Real} = GaussianBelief(zeros(T, n), zeros(T, n, n))
uninformative(n::Int) = uninformative(Float64, n)

"""
    isuninformative(b::AbstractBelief) -> Bool

Whether `b` carries zero precision / information (`TrivialBelief()` or a `GaussianBelief`
with `Λ == 0` and `η == 0`). Used by polarity resolution to distinguish informative
incoming messages from identity placeholders.
"""
isuninformative(::TrivialBelief) = true
isuninformative(b::GaussianBelief) = all(iszero, b.Λ) && all(iszero, b.η)
isuninformative(::AbstractBelief) = false

dimension(b::GaussianBelief) = length(b.η)

# `Symmetric` reads one triangle, so an asymmetric matrix would be validated and used as its
# upper triangle mirrored (LP-019 item 3). Definiteness is checked on, and factorisations use,
# the symmetric part, so an asymmetry cannot hide an indefinite form behind a definite
# triangle (LP-019 review). The tolerance, √eps(Float32) relative to max|M| for every element
# type, then only refuses gross asymmetry: it admits rounding (A Λ Aᵀ) and the single-entry
# perturbations of LP-002's finite-difference adjoints (~1e-6), and does not change under
# ForwardDiff duals.
function _nearly_symmetric(M)
    A = Matrix(M)
    return maximum(abs, A - A'; init = 0.0) <= sqrt(eps(Float32)) * maximum(abs, A; init = 0.0)
end
# The same test without the two copies, for a dense float matrix; it runs on every message (I18).
# A NaN anywhere, or an infinite entry (∞ − ∞ is NaN in `A - A'`), makes the method above false
# through `maximum`, which returns NaN; the loop returns false for the same entries.
function _nearly_symmetric(A::Matrix{T}) where {T<:Union{Float32,Float64}}
    n, m = size(A)
    n == m || return invoke(_nearly_symmetric, Tuple{Any}, A)   # its DimensionMismatch (LP-026 F1)
    asym, amax = 0.0, 0.0
    @inbounds for j in 1:n, i in 1:n
        d = abs(A[i, j] - A[j, i])               # in T, as `A - A'` computes it
        isnan(d) && return false
        asym = max(asym, Float64(d)); amax = max(amax, Float64(abs(A[i, j])))
    end
    return asym <= sqrt(eps(Float32)) * amax
end
# A + (Aᵀ − A)/2: exact for a symmetric A (a subnormal diagonal survives, unlike A/2 + Aᵀ/2),
# and no overflow near floatmax (unlike (A + Aᵀ)/2).
# One fused broadcast: the same operations per element as `A + (A' - A) / 2`, one array (I18).
_symmetric_part(M) = (A = Matrix(M); Symmetric(A .+ (A' .- A) ./ 2))
_check_symmetric(M, name) = _nearly_symmetric(M) ||
    throw(ArgumentError("$name is not symmetric: $(M)"))

function _chol(M, name)
    _check_symmetric(M, name)
    return try
        cholesky(_symmetric_part(M))
    catch
        throw(ArgumentError("$name is not positive definite: $(M)"))
    end
end

# The inverse of a symmetric positive definite matrix through its Cholesky factor, with `_chol`'s
# errors. It has an rrule (`LenticulumCoreChainRulesCoreExt`): Zygote has no rule for
# `inv(::Cholesky)` (P23).
_spd_inv(M, name) = inv(_chol(M, name))

"""
    isproper(b::GaussianBelief) -> Bool

Whether the belief is a normalisable distribution rather than a likelihood: `false` if it was
built with `improper = true`, otherwise whether `Λ` is positive definite. `belief_mean`,
`belief_cov`, `variable_entropy` and `belief_logdensity` all require this.
"""
isproper(b::GaussianBelief) = !b.improper && _precision_isproper(b.Λ)
_precision_isproper(M) = _nearly_symmetric(M) && isposdef(_symmetric_part(M))

"""
    belief_mean(b) -> Vector

``\\mu = \\Lambda^{-1}\\eta``. Errors on an improper belief, rather than returning a
least-squares pseudo-mean that would be silently wrong.

Named `belief_mean` rather than extending `Statistics.mean` to avoid a fifth name collision in
this project — see `Mycelium.md` §2.
"""
function belief_mean(b::GaussianBelief)
    C = _improper_check(b, "belief_mean")
    return _precision_solve(C, b.η)
end

"""
    belief_cov(b) -> Matrix

``\\Sigma = \\Lambda^{-1}``.
"""
belief_cov(b::GaussianBelief) = inv(_improper_check(b, "belief_cov"))

"""
    belief_mean(b::SampleBelief) -> Vector
    belief_cov(b::SampleBelief) -> Matrix

Self-normalised importance-sampling estimates of the first two moments: the weighted mean and
the weighted covariance ``\\sum_i w_i (x_i - \\bar x)(x_i - \\bar x)^\\top`` with the weights
normalised to sum to one (uniform without weights). Samples are the columns of a matrix or the
elements of a vector (vectors or scalars). Defined here, beside the type (P18; they were in
AdversarialFactors, `ratio.jl`, and existed only with it loaded). See `open_model.md`, "Sample moments".
"""
function belief_mean(b::SampleBelief)
    w = _moment_weights(b)
    return vec(_samples_matrix(b) * w)
end

function belief_cov(b::SampleBelief)
    w = _moment_weights(b)
    S = _samples_matrix(b)
    m = vec(S * w)
    X = S .- m
    return (X .* w') * X'
end

# the samples as a matrix, columns one sample: a matrix already is, a vector of scalars is one
# row (a reshape), a vector of vectors one `reduce(hcat, …)` (ChainRules pulls it back; a splatted
# `hcat` compiles a method per sample count)
_samples_matrix(b::SampleBelief) = _samples_matrix(b.samples)
_samples_matrix(S::AbstractMatrix) = S
_samples_matrix(v::AbstractVector{<:Number}) = reshape(v, 1, :)
_samples_matrix(v::AbstractVector) = reduce(hcat, v)

_sample_points(b::SampleBelief) = b.samples isa AbstractMatrix ? eachcol(b.samples) : b.samples
# divided by the largest weight first, so that weights near floatmax do not overflow the sum.
# Not Mycelium's `_sample_weights` (messages.jl), which returns uniform weights for all-zero ones;
# the constructor refuses those, so the difference is not reached (LP-029, F4).
# The moments are pure reductions (a matrix-vector product and a centred cross-product), so they
# differentiate under ForwardDiff and Zygote and keep the samples' scalar type: unweighted
# Float32 samples give Float32 moments, integer samples are floated (GH-21).
function _moment_weights(b::SampleBelief)
    pts = _sample_points(b)
    device = _device_leaf(b)
    if device !== nothing
        S = _samples_matrix(b)
        T = float(eltype(S))
        n = size(S, 2)
        b.weights === nothing && return fill!(similar(S, T, n), one(T) / n)
        w = _on_device(device, b.weights isa AbstractVector ? float.(b.weights) : collect(float.(b.weights)))
        w = w ./ maximum(w)
        return w ./ sum(w)
    end
    n = length(pts)
    p0 = first(pts)
    T = p0 isa AbstractArray ? eltype(p0) : typeof(p0)
    F = float(T)
    b.weights === nothing && return fill(one(F) / n, n)
    w = float.(collect(b.weights))   # weights may be a tuple (AdversarialFactors' particle weights)
    m = maximum(w)
    m > 0 || throw(ArgumentError("all sample weights are zero"))
    w = w ./ m
    return w ./ sum(w)
end

# `(belief_mean(b), belief_cov(b))` from one factorisation of Λ instead of two: the relation
# factors need both for every message (I18). Only for beliefs on the CPU (the device methods of
# the two functions stay in charge there). Reverse mode differentiates the two functions
# separately (the rule in the ChainRulesCore extension), so its gradients are theirs.
_mean_cov(b::GaussianBelief) = (belief_mean(b), belief_cov(b))
function _mean_cov(b::GaussianBelief{<:Array,<:Array})
    C = _improper_check(b, "belief_mean")   # the error `_moments` gave, from `belief_mean` (LP-026 F2)
    return (_precision_solve(C, b.η), inv(C))
end

function _improper_check(b::GaussianBelief, who)
    b.improper && throw(ArgumentError(
        "$who needs a proper belief, but this one was declared improper: a likelihood message, \
         not a distribution. Combine it with a prior first."))
    try
        return _precision_factor(b.Λ)
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError(
            "$who needs a proper belief, but Λ is singular. This belief is a likelihood \
             message, not a distribution — combine it with a prior first."))
    end
end

"""
    logpartition(b) -> Real

``\\log \\int \\exp(-\\tfrac12 x^\\top\\Lambda x + \\eta^\\top x)\\,dx
 = \\tfrac12\\eta^\\top\\Lambda^{-1}\\eta - \\tfrac12\\log\\det\\Lambda + \\tfrac n2\\log 2\\pi``.
"""
function logpartition(b::GaussianBelief)
    C = _improper_check(b, "logpartition")
    n = dimension(b)
    ld = _precision_logdet(C)
    return (b.η' * _precision_solve(C, b.η)) / 2 - ld / 2 + n * _log2π(typeof(ld)) / 2
end

function Base.show(io::IO, b::GaussianBelief)
    C = cholesky(_symmetric_part(b.Λ); check = false)
    if isproper(b) && issuccess(C)
        μ = C \ b.η
        Σ = inv(C)
        print(io, "Gaussian(μ=", round.(μ; digits = 4),
            ", Σ=", round.(Σ; digits = 4), ")")
    else
        print(io, "GaussianBelief(η=", round.(b.η; digits = 4),
            ", Λ=", round.(b.Λ; digits = 4), "; improper)")
    end
end

function _precision_factor(M)
    _check_symmetric(M, "Λ")
    return cholesky(_symmetric_part(M))
end

_precision_logdet(C) = logdet(C)

# log(2π) in the number type `T` of the value it is added to: the Float64 constant `log(2π)` widens
# a Float32 energy (P8). `T` is a type, not a value, so reverse-mode AD sees a constant.
_log2π(::Type{T}) where {T<:Real} = log(2 * convert(T, π))     # floats and duals
_log2π(::Type{<:Integer}) = log(2π)

_precision_solve(C, η) = C \ η


# Numeric input leaves, with discrete labels and empty structural containers neutral.
_numeric_eltype(x) = Union{}
_numeric_eltype(x::Real) = typeof(x)
_numeric_eltype(x::Integer) = Union{}
_numeric_eltype(x::AbstractArray{<:Real}) = eltype(x) <: Integer ? Union{} : eltype(x)
_numeric_eltype(x::Union{Tuple,NamedTuple,AbstractArray}) =
    mapreduce(_numeric_eltype, promote_type, x; init = Union{})
_numeric_eltype(b::DiracBelief) = _numeric_eltype(b.value)
_numeric_eltype(b::GaussianBelief) = promote_type(_numeric_eltype(b.η), _numeric_eltype(b.Λ))
_numeric_eltype(b::CategoricalBelief) = eltype(b.logp)
_numeric_eltype(b::MixtureBelief) = promote_type(eltype(b.logw), _numeric_eltype(b.components))
_numeric_eltype(b::SampleBelief) = promote_type(_numeric_eltype(b.samples), _numeric_eltype(b.weights))

function _compute_eltype(default::Type{D}, inputs::Vararg{Any,N}) where {D,N}   # specialised (LP-092)
    T = _numeric_eltype(inputs)
    return T === Union{} ? default : float(T)
end

_at_eltype(::Type{T}, x::AbstractArray{T}) where {T} = x
_at_eltype(::Type{T}, x::AbstractArray) where {T} = T.(x)
_zeros_like(x::Array, ::Type{T}, dims::Int...) where {T} = zeros(T, dims)
_zeros_like(x, ::Type{T}, dims::Int...) where {T} = fill!(similar(x, T, dims), zero(T))

Adapt.@adapt_structure GaussianBelief
_device_leaf(b::GaussianBelief) = _first_device((b.η, b.Λ))
