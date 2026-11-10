"""
    AffineGenerator(A, b)

A Lux-compatible affine generator `z -> ps.A*z + ps.b`. Unlike an arbitrary network,
this layer supplies an exact Gaussian pushforward and a certified constant Jacobian.
Differential entropy requires square nonsingular `ps.A`; particle transport does not.
"""
struct AffineGenerator{M,V} <: LuxCore.AbstractLuxLayer
    A::M
    b::V
    function AffineGenerator(A::AbstractMatrix, b::AbstractVector)
        size(A,1) == length(b) || throw(DimensionMismatch("affine output and bias"))
        new{typeof(A),typeof(b)}(A,b)
    end
end
LuxCore.initialparameters(::AbstractRNG, f::AffineGenerator) = (A=copy(f.A),b=copy(f.b))
LuxCore.initialstates(::AbstractRNG, ::AffineGenerator) = (;)
(f::AffineGenerator)(z,ps,st) = (ps.A*z+ps.b,st)

"""
    InvertibleGenerator(net, logabsdet)

An explicit assertion by the caller that `net` (including the GeneratorFactor's input
adapter) is a global differentiable bijection of equal-dimensional Euclidean spaces.
`logabsdet(z, ps, st)` supplies its exact log absolute Jacobian determinant. Local
nonsingularity alone cannot certify global invertibility (e.g. folding maps).
Used only by `estimate_pushforward_entropy`, never to claim an exact nonlinear integral.
"""
struct InvertibleGenerator{N,J} <: LuxCore.AbstractLuxWrapperLayer{:net}
    net::N
    logabsdet::J
end
(f::InvertibleGenerator)(z,ps,st) = LuxCore.apply(f.net,z,ps,st)

"""
    EntropyEstimate(value, standard_error, nsamples)

A Monte Carlo differential entropy estimate, relative to Lebesgue coordinates.
It is deliberately neither a number nor an exact game entropy.
"""
struct EntropyEstimate{T,S}
    value::T
    standard_error::S
    nsamples::Int
end

function _source_gaussian_entropy(b::LenticulumCore.GaussianBelief, n)
    length(b.η) == n || throw(DimensionMismatch("latent Gaussian dimension"))
    all(isfinite,b.η) && all(isfinite,b.Λ) && LinearAlgebra.issymmetric(b.Λ) ||
        throw(ArgumentError("source Gaussian must have finite information and symmetric precision"))
    LenticulumCore.isproper(b) || throw(ArgumentError("entropy requires a proper Gaussian source"))
    return (n*(1+log((2 * Base.MathConstants.pi)))-LinearAlgebra.logdet(LinearAlgebra.Symmetric(b.Λ)))/2
end
function _affine_logdet(f::GeneratorFactor, ps)
    f.net isa AffineGenerator && f.input === identity || throw(ArgumentError(
        "exact generator entropy requires AffineGenerator with identity input; arbitrary networks have no certified invertible law"))
    latentdim(f) == sampledim(f) || throw(ArgumentError(
        "Lebesgue change-of-variables entropy requires a square invertible map; rectangular generators are unsupported"))
    size(ps.A) == (sampledim(f),latentdim(f)) && length(ps.b) == sampledim(f) ||
        throw(DimensionMismatch("affine parameters must match generator channels"))
    all(isfinite,ps.A) && all(isfinite,ps.b) || throw(ArgumentError("affine parameters must be finite"))
    ld, sign = LinearAlgebra.logabsdet(ps.A)
    sign != 0 && isfinite(ld) || throw(ArgumentError(
        "singular generator has no output density relative to full-dimensional Lebesgue measure"))
    return ld
end

"""
    pushforward_entropy(f::GeneratorFactor, source::GaussianBelief, ps, st)

Exact `H(g#source) = H(source) + log|det A|` for a square nonsingular
`AffineGenerator`. Differential entropy is relative to Lebesgue (LP-006).
Empirical and Dirac laws have no finite differential entropy; their conditional point
convention is available separately through `entropy(f, π, DiracBelief(z), ps, st)`.
"""
function pushforward_entropy(f::GeneratorFactor, source::LenticulumCore.GaussianBelief, ps, st)
    ld = _affine_logdet(f,ps)
    return _source_gaussian_entropy(source,latentdim(f))+ld, st
end
pushforward_entropy(::GeneratorFactor, source, ps, st) = throw(ArgumentError(
    "exact Lebesgue pushforward entropy requires a proper Gaussian source and an affine bijection; Dirac/finite empirical laws are singular"))

function LenticulumCore.pushforward(f::GeneratorFactor, source::LenticulumCore.GaussianBelief, ps, st)
    _affine_logdet(f,ps)
    _source_gaussian_entropy(source,latentdim(f))
    μ, Σ = LenticulumCore.belief_mean(source), LenticulumCore.belief_cov(source)
    return LenticulumCore.Gaussian(ps.A*μ+ps.b, ps.A*Σ*ps.A'), st
end

"""
    estimate_pushforward_entropy(f, source::GaussianBelief, ps, st; nsamples=1024, rng)

For a caller-certified nonlinear `InvertibleGenerator`, estimate only the Jacobian
expectation using independent draws from the known Gaussian source; add its exact entropy.
Returns `EntropyEstimate`. The callback and network must describe the same global bijection.
Finite-support quadrature could integrate a Jacobian exactly, but would not turn a singular
source into a Lebesgue density; this API therefore refuses sample/Dirac sources.
"""
function estimate_pushforward_entropy(f::GeneratorFactor, source::LenticulumCore.GaussianBelief, ps, st;
                                      nsamples::Int=1024, rng::AbstractRNG)
    f.net isa InvertibleGenerator || throw(ArgumentError("nonlinear entropy requires an explicit InvertibleGenerator law"))
    latentdim(f) == sampledim(f) || throw(ArgumentError("nonlinear entropy requires equal input/output dimensions"))
    nsamples >= 2 || throw(ArgumentError("nsamples must be at least two"))
    H = _source_gaussian_entropy(source,latentdim(f))
    μ = LenticulumCore.belief_mean(source)
    L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(LenticulumCore.belief_cov(source))).L
    vals = map(1:nsamples) do _
        z = μ + L*randn(rng,length(μ))
        value = f.net.logabsdet(z,ps,st)
        value isa Real && isfinite(value) || throw(ArgumentError("Jacobian must be nonsingular with finite real logabsdet"))
        value
    end
    m = sum(vals)/nsamples
    se = sqrt(sum(abs2,vals .- m)/(nsamples*(nsamples-1)))
    return EntropyEstimate(H+m,se,nsamples), st
end
estimate_pushforward_entropy(::GeneratorFactor, source, ps, st; kwargs...) = throw(ArgumentError(
    "nonlinear Lebesgue entropy estimates require a proper Gaussian source; empirical beliefs do not supply a density"))

# Distributional forward game: y is the latent law, π must be absent/trivial.
# Conditioning the output on another belief would require a separate inversion/density law.
function LenticulumCore.entropy(f::GeneratorFactor, π, y::LenticulumCore.GaussianBelief, ps, st)
    π === nothing || π isa LenticulumCore.TrivialBelief || throw(ArgumentError(
        "generator distributional game requires no output conditioning (π=nothing or TrivialBelief)"))
    return pushforward_entropy(f,y,ps,st)
end
function LenticulumCore.free_energy(f::GeneratorFactor, π, y::LenticulumCore.GaussianBelief, ps, st)
    H, st = LenticulumCore.entropy(f,π,y,ps,st)
    return -H, st
end

# The built-in source's law is known exactly, regardless of particle count. Custom samplers
# still have no density contract. π=TrivialBelief selects the unconditional source game.
function LenticulumCore.entropy(f::NoiseSource, π::LenticulumCore.TrivialBelief, y, ps, st)
    f.dist === _std_normal || throw(ArgumentError("custom NoiseSource sampler supplies no exact density/entropy law"))
    y === nothing || y isa LenticulumCore.TrivialBelief || throw(ArgumentError("unconditional NoiseSource entropy requires no observations"))
    return f.dim*(1+log((2 * Base.MathConstants.pi)))/2, st
end
function LenticulumCore.free_energy(f::NoiseSource, π::LenticulumCore.TrivialBelief, y, ps, st)
    H, st = LenticulumCore.entropy(f,π,y,ps,st)
    return -H, st
end
