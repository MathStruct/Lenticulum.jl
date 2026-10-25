"""
    plucker_pairs(r)

Coordinate indices for `r ≥ 2`, in lexicographic order:
`(1,2), (1,3), …, (1,r), (2,3), …, (r-1,r)`. There are `binomial(r,2)`.
"""
function plucker_pairs(r::Integer)
    r >= 2 || throw(ArgumentError("Plücker coordinates need r ≥ 2"))
    return [(i, j) for i in 1:r for j in (i+1):r]
end

function _plucker_vectors(z::AbstractVector, w::AbstractVector)
    length(z) == length(w) || throw(DimensionMismatch("z and w must have equal lengths"))
    length(z) >= 2 || throw(ArgumentError("z and w need length ≥ 2"))
    Base.require_one_based_indexing(z, w)
    return length(z)
end

"""
    plucker(z, w)

The bivector `z ∧ w`: `z[i]*w[j] - z[j]*w[i]` in [`plucker_pairs`](@ref)
order. Inputs must be vectors of equal length ≥ 2. Preserves generic arithmetic
(including integers, rationals and complex numbers); no floating-point conversion.
"""
function plucker(z::AbstractVector, w::AbstractVector)
    r = _plucker_vectors(z, w)
    return [z[i]*w[j] - z[j]*w[i] for (i, j) in plucker_pairs(r)]
end

"""
    plucker_jacobian(z, w)

Closed-form `binomial(r,2) × 2r` Jacobian. Columns differentiate with respect to
`vcat(z, w)`: all `z` coordinates first, then all `w` coordinates.
"""
function plucker_jacobian(z::AbstractVector, w::AbstractVector)
    r = _plucker_vectors(z, w)
    J = zeros(promote_type(eltype(z), eltype(w)), binomial(r, 2), 2r)
    for (a, (i, j)) in enumerate(plucker_pairs(r))
        J[a,i] = w[j]; J[a,j] = -w[i]
        J[a,r+i] = -z[j]; J[a,r+j] = z[i]
    end
    return J
end

"""
    plucker_relations(r)

A vector of relation records in lexicographic quadruple order. Each record has
`indices = (i,j,k,l)` and `monomials = ((ij,kl), (ik,jl), (il,jk))`, where each
integer in a monomial indexes `p` in `plucker_pairs(r)` order, and
`coefficients = (1,-1,1)`. Thus each represents
`p[ij]*p[kl] - p[ik]*p[jl] + p[il]*p[jk]`. There are `binomial(r,4)` records;
for `r=2,3` the vector is empty. This is sparse polynomial test data, not fitting.
"""
function plucker_relations(r::Integer)
    lookup = Dict(pair => a for (a, pair) in enumerate(plucker_pairs(r)))
    return [(indices = (i,j,k,l),
             monomials = ((lookup[(i,j)], lookup[(k,l)]),
                          (lookup[(i,k)], lookup[(j,l)]),
                          (lookup[(i,l)], lookup[(j,k)])),
             coefficients = (1,-1,1))
            for i in 1:r for j in (i+1):r for k in (j+1):r for l in (k+1):r]
end

"""
    plucker_residuals(r, p)

Evaluate all [`plucker_relations`](@ref) at a vector `p` of length `binomial(r,2)`.
"""
function plucker_residuals(r::Integer, p::AbstractVector)
    relations = plucker_relations(r)
    length(p) == binomial(r, 2) || throw(DimensionMismatch("wrong Plücker coordinate count"))
    Base.require_one_based_indexing(p)
    return [sum(c * p[a] * p[b] for (c, (a,b)) in zip(rel.coefficients, rel.monomials))
            for rel in relations]
end

"""
    plucker_moments(μ, S, ν, T) -> (mean, cov)

Exact first two moments of `z ∧ w` for **independent** real random vectors with
means `μ, ν` and covariances `S, T`. Gaussianity is unnecessary for this identity.
Zero covariance represents a Dirac. Covariances are assumed valid; only shapes are
checked. Returns the exact covariance even when singular; no jitter is added.
See `plucker.md` for the derivation.
"""
function plucker_moments(μ::AbstractVector{<:Real}, S::AbstractMatrix{<:Real},
                         ν::AbstractVector{<:Real}, T::AbstractMatrix{<:Real})
    r = _plucker_vectors(μ, ν)
    size(S) == size(T) == (r,r) || throw(DimensionMismatch("covariances must be r × r"))
    Base.require_one_based_indexing(S, T)
    J = plucker_jacobian(μ, ν)
    C = J[:,1:r] * S * transpose(J[:,1:r]) + J[:,r+1:2r] * T * transpose(J[:,r+1:2r])
    pairs = plucker_pairs(r)
    for (a, (i,j)) in enumerate(pairs), (b, (k,l)) in enumerate(pairs)
        C[a,b] += S[i,k]*T[j,l] - S[i,l]*T[j,k] - S[j,k]*T[i,l] + S[j,l]*T[i,k]
    end
    return (mean = plucker(μ, ν), cov = C)
end

"""
    PluckerFactor(r; names = (:z, :w, :p), covariance_jitter = 0)

Parameter-free deterministic map `(z,w) → z ∧ w`. Only the forward polarity is
supported: the fibre of a nonzero decomposable bivector is an `SL₂`-orbit of pairs;
no belief on that fibre is built, and `p=0` is degenerate.

Dirac pairs give a Dirac. Sample pairs are transported by **matching particle index**,
with equal nonzero counts and matching normalised weights (`rtol=1e-12, atol=0`);
these are a joint empirical belief, not a Cartesian product of independent clouds.
The first cloud's weights are preserved. A Dirac may accompany either sample cloud.
Matrices store particles in columns; vectors store vector-valued particles.

Separate Gaussian inputs are assumed independent. Gaussian/Gaussian and Dirac/Gaussian
messages match exact moments, but the bilinear output law is generally not Gaussian.
`GaussianBelief` cannot represent singular covariance. Without explicit regularisation,
a covariance with `λmin ≤ n*eps(T)*λmax` throws; `plucker_moments` still exposes its
exact moments. Set `covariance_jitter > 0` to explicitly add that absolute variance
to every output coordinate before constructing the belief (then covariance is approximate).
No jitter is ever added by default. Improper Gaussians and other belief pairs throw.

Pointwise energy and graph-local free energy return zero, following `GeneratorFactor`.
Game-level free energy/entropy and an assembled Bayesian lens are not supplied (LP-009).
"""
struct PluckerFactor{N,J<:Real} <: LenticulumCore.AbstractLenticulumFactor
    r::Int
    names::N
    covariance_jitter::J
    function PluckerFactor(r::Integer; names = (:z, :w, :p), covariance_jitter::Real = 0)
        r >= 2 || throw(ArgumentError("PluckerFactor needs r ≥ 2"))
        names isa Tuple{Symbol,Symbol,Symbol} && length(unique(names)) == 3 ||
            throw(ArgumentError("names must be three distinct Symbols in (z,w,p) order"))
        isfinite(covariance_jitter) && covariance_jitter >= 0 ||
            throw(ArgumentError("covariance_jitter must be finite and nonnegative"))
        return new{typeof(names),typeof(covariance_jitter)}(Int(r), names, covariance_jitter)
    end
end

LenticulumCore.channels(f::PluckerFactor) =
    map((n,d) -> LenticulumCore.Channel(n,d), f.names, (f.r, f.r, binomial(f.r,2)))
LenticulumCore.islearnable(::PluckerFactor) = false
LuxCore.initialparameters(::AbstractRNG, ::PluckerFactor) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::PluckerFactor) = NamedTuple()
LuxCore.parameterlength(::PluckerFactor) = 0
LuxCore.statelength(::PluckerFactor) = 0
LenticulumCore.supported_polarities(f::PluckerFactor) =
    (LenticulumCore.Polarity(NamedTuple{f.names}((LenticulumCore.Observed(),
        LenticulumCore.Observed(), LenticulumCore.Unobserved()))),)
LenticulumCore.supports_polarity(f::PluckerFactor, p::LenticulumCore.Polarity) =
    keys(p) == f.names && p[f.names[1]] isa LenticulumCore.Observed &&
    p[f.names[2]] isa LenticulumCore.Observed && p[f.names[3]] isa LenticulumCore.Unobserved
LenticulumCore.energyspace(::PluckerFactor) = LenticulumCore.ScalarEnergySpace()
LenticulumCore.energy(::PluckerFactor, x, a, y, ps, st) = (0.0, st)
Mycelium.local_free_energy(::PluckerFactor, _, ps, st) = (0.0, st)

function _plucker_point(f, x)
    x isa AbstractVector{<:Real} || throw(ArgumentError("PluckerFactor inputs must be real vectors"))
    length(x) == f.r || throw(DimensionMismatch("PluckerFactor input needs length $(f.r)"))
    return x
end
_plucker_points(b::LenticulumCore.SampleBelief) =
    b.samples isa AbstractMatrix ? collect(eachcol(b.samples)) : b.samples
function _plucker_weights(b, n)
    b.weights === nothing && return fill(1/n, n)
    m = maximum(b.weights)
    m > 0 || throw(ArgumentError("sample weights must have positive total mass"))
    w = b.weights / m
    return w / sum(w)
end
function _plucker_input_moments(f, b::LenticulumCore.DiracBelief)
    μ = _plucker_point(f, b.value)
    return μ, zeros(eltype(μ), f.r, f.r)
end
function _plucker_input_moments(f, b::LenticulumCore.GaussianBelief)
    μ = _plucker_point(f, LenticulumCore.belief_mean(b))
    return μ, LenticulumCore.belief_cov(b)
end

"""
    LenticulumCore.pushforward(f::PluckerFactor, (zbelief, wbelief), ps, st)

Push a pair of beliefs through the forward map, using the conventions of
[`PluckerFactor`](@ref). State is returned unchanged; no parameters are used.
"""
function LenticulumCore.pushforward(f::PluckerFactor, beliefs::Tuple{Any,Any}, ps, st)
    z, w = beliefs
    D, S, G = LenticulumCore.DiracBelief, LenticulumCore.SampleBelief, LenticulumCore.GaussianBelief
    if z isa D && w isa D
        return (D(plucker(_plucker_point(f, z.value), _plucker_point(f, w.value))), st)
    elseif (z isa S && w isa Union{S,D}) || (w isa S && z isa D)
        cloud = z isa S ? z : w
        zs = z isa S ? _plucker_points(z) : nothing
        ws = w isa S ? _plucker_points(w) : nothing
        n = length(_plucker_points(cloud))
        n > 0 || throw(ArgumentError("sample beliefs must be nonempty"))
        weights = _plucker_weights(cloud, n)
        if z isa S && w isa S
            length(ws) == n || throw(ArgumentError("paired sample beliefs need equal sample counts"))
            isapprox(weights, _plucker_weights(w, n); rtol = 1e-12, atol = 0) ||
                throw(ArgumentError("paired sample beliefs need matching normalised weights"))
        end
        out = [plucker(_plucker_point(f, z isa S ? zs[a] : z.value),
                       _plucker_point(f, w isa S ? ws[a] : w.value)) for a in 1:n]
        return (S(out, cloud.weights), st)
    elseif z isa Union{G,D} && w isa Union{G,D}
        μ, A = _plucker_input_moments(f, z)
        ν, B = _plucker_input_moments(f, w)
        moments = plucker_moments(μ, A, ν, B)
        C = float.(moments.cov)
        if f.covariance_jitter > 0
            C += convert(eltype(C), f.covariance_jitter) * LinearAlgebra.I
        end
        λ = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(C))
        minimum(λ) > length(λ)*eps(eltype(C))*maximum(λ) || throw(ArgumentError(
            "Plücker covariance is singular or numerically rank deficient; GaussianBelief cannot " *
            "represent singular covariance. Use plucker_moments for exact moments, or explicitly " *
            "set covariance_jitter > 0 for regularised Gaussian messages."))
        return (LenticulumCore.Gaussian(moments.mean, C), st)
    end
    throw(ArgumentError("unsupported PluckerFactor belief pair: $(typeof(z)), $(typeof(w)); " *
                        "use Dirac/Dirac, paired samples, Dirac/samples, or proper Gaussian/Dirac inputs"))
end

function Mycelium.factor_message(f::PluckerFactor, target::Symbol, polarity, inputs, prior, ps, st)
    target == f.names[3] || throw(ArgumentError("PluckerFactor emits only on :$(f.names[3]); inverse fibres are not implemented"))
    if polarity !== nothing
        polarity isa LenticulumCore.Polarity && LenticulumCore.supports_polarity(f, polarity) ||
            throw(ArgumentError("PluckerFactor supports only its forward polarity"))
    end
    all(n -> haskey(inputs, n), f.names[1:2]) ||
        throw(ArgumentError("PluckerFactor needs beliefs on both input channels $(f.names[1:2])"))
    return LenticulumCore.pushforward(f, (inputs[f.names[1]], inputs[f.names[2]]), ps, st)
end
