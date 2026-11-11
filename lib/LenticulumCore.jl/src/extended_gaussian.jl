# ---------------------------------------------------------------------------
# Extended Gaussians: a Gaussian together with a subspace along which it is flat.
#
#     N(μ, Σ) + D        μ ∈ ℝⁿ,  Σ ⪰ 0,  D ⊆ ℝⁿ a subspace (orthonormal basis, columns of D)
#
# the law of μ + ε + d with ε ~ N(0, Σ) and d "uniform" on D (Stein & Samuelson's extended
# Gaussian distributions). Three kinds of direction in one object: flat (D: a likelihood that
# says nothing there), hard (null(Σ) outside D: no variance at all, a linear constraint) and
# Gaussian (the rest). `GaussianBelief` holds the first and third; `DiracBelief` is the case
# with every direction hard; `TrivialBelief` the case with every direction flat.
#
# The representative is normalised: D orthonormal, and μ and Σ projected onto D⊥, since
# anything along D is invisible. Pushforward through an affine map is exact in this form; the
# product (`_conjoin`, Mycelium's `combine`) stacks the hard constraints and adds precisions on
# what they leave free. See `extended_gaussian.md`.
# ---------------------------------------------------------------------------

"""
    ExtendedGaussian(μ, Σ, D)
    ExtendedGaussian(μ, Σ)

A Gaussian `N(μ, Σ)` made flat along the subspace spanned by the columns of `D`: the law of
`μ + ε + d` with `ε ~ N(0, Σ)` and `d` uniform on `span(D)` (an *extended Gaussian
distribution*, Stein & Samuelson). `Σ` may be singular: a direction in `null(Σ)` and not in
`span(D)` is **hard**, a linear constraint the belief imposes exactly.

The constructor normalises the representative: the columns of `D` are made orthonormal (and
dependent ones dropped), and `μ` and `Σ` are projected onto the complement of `span(D)`. `Σ`
must be positive semidefinite up to rounding.

Operations: [`pushforward`](@ref)`(b, M, c, Q)` (exact through affine maps, with optional
added noise), [`preimage`](@ref)`(b, M, c)` (the likelihood of `x` when `b` is about `Mx + c`),
[`marginalise`](@ref), conversion from and to `GaussianBelief` ([`extended`](@ref),
`GaussianBelief(b)`), [`flat_directions`](@ref), [`hard_directions`](@ref). The product of two
beliefs is Mycelium's `combine`. There is no density, entropy or log-partition: a belief with
flat or hard directions has none in closed form.
"""
struct ExtendedGaussian{T<:AbstractFloat} <: AbstractBelief
    μ::Vector{T}
    Σ::Matrix{T}
    D::Matrix{T}
    function ExtendedGaussian(μ::AbstractVector, Σ::AbstractMatrix, D::AbstractMatrix)
        n = length(μ)
        size(Σ) == (n, n) || throw(DimensionMismatch("μ has length $n but Σ is $(size(Σ))"))
        size(D, 1) == n || throw(DimensionMismatch("μ has length $n but D has $(size(D, 1)) rows"))
        T = float(promote_type(eltype(μ), eltype(Σ), eltype(D)))
        _nearly_symmetric(Σ) || throw(ArgumentError("Σ must be symmetric"))
        B = _orth(Matrix{T}(D))
        # Project onto the complement of span(D) through an orthonormal basis U of it, not with
        # I − BBᵀ: when D spans everything U is empty and Σ is exactly zero, where I − BBᵀ would
        # leave a residue of 1e-30 that the relative tolerances below take for a variance.
        U = _complement(B)
        Σu = Matrix(_symmetric_part(U' * Matrix{T}(Σ) * U))
        λ = isempty(Σu) ? T[] : eigvals(Symmetric(Σu))
        isempty(λ) || minimum(λ) >= -_eig_tol(λ) || throw(ArgumentError(
            "Σ must be positive semidefinite; it has the eigenvalue $(minimum(λ))"))
        S = U * Σu * U'
        return new{T}(U * (U' * Vector{T}(μ)), (S + S') / 2, B)
    end
end
ExtendedGaussian(μ::AbstractVector, Σ::AbstractMatrix) = ExtendedGaussian(μ, Σ, zeros(eltype(μ), length(μ), 0))

dimension(b::ExtendedGaussian) = length(b.μ)
isuninformative(b::ExtendedGaussian) = size(b.D, 2) == length(b.μ)

# Eigenvalues (or singular values) below this count as zero: the tolerance of `_split_moments`.
_eig_tol(λ) = isempty(λ) ? 0.0 : 16 * length(λ) * eps(float(eltype(λ))) * maximum(abs, λ)

# An orthonormal basis of the column space of M (columns below the tolerance dropped).
function _orth(M::AbstractMatrix{T}) where {T}
    size(M, 2) == 0 && return zeros(T, size(M, 1), 0)
    F = svd(M)
    return F.U[:, F.S .> _eig_tol(F.S)]
end

# An orthonormal basis of the orthogonal complement of the column space of B (B orthonormal).
function _complement(B::AbstractMatrix{T}) where {T}
    n = size(B, 1)
    size(B, 2) == 0 && return Matrix{T}(I, n, n)
    F = svd(B; full = true)
    r = count(F.S .> _eig_tol(F.S))
    return F.U[:, (r + 1):n]
end

"""
    flat_directions(b::ExtendedGaussian) -> Matrix

An orthonormal basis of the subspace along which `b` is flat (says nothing).
"""
flat_directions(b::ExtendedGaussian) = b.D

"""
    hard_directions(b::ExtendedGaussian) -> Matrix

An orthonormal basis of the directions in which `b` has no variance and is not flat: the linear
constraints `Hᵀx = Hᵀμ` it imposes exactly.
"""
function hard_directions(b::ExtendedGaussian{T}) where {T}
    n = length(b.μ)
    U = _complement(b.D)                       # D⊥
    size(U, 2) == 0 && return zeros(T, n, 0)
    E = eigen(Symmetric(U' * b.Σ * U))
    return U * E.vectors[:, E.values .<= _eig_tol(E.values)]
end

# The quadratic part on the Gaussian directions: Λ = Σ⁺ (zero on the flat and hard directions)
# and η = Λμ.
function _canonical_part(b::ExtendedGaussian{T}) where {T}
    E = eigen(Symmetric(b.Σ))
    keep = E.values .> _eig_tol(E.values)
    V = E.vectors[:, keep]
    Λ = V * Diagonal(1 ./ E.values[keep]) * V'
    Λ = (Λ + Λ') / 2
    return Λ * b.μ, Λ
end

"""
    extended(b) -> ExtendedGaussian
    extended(b, n)

The belief as an [`ExtendedGaussian`](@ref):

- a proper `GaussianBelief`: its mean and covariance, nowhere flat;
- an improper or singular one: flat along `null(Λ)`, with mean `Λ⁺η` and covariance `Λ⁺` on
  `range(Λ)` (`η` must lie in `range(Λ)`: a component along `null(Λ)` is `exp(ηᵀx)` there, which
  no limit of Gaussians gives; `Λ` must be positive semidefinite to rounding);
- a `DiracBelief`: its value, every direction hard;
- `TrivialBelief()` (needs the dimension `n`): flat everywhere.
"""
function extended(b::GaussianBelief)
    n = dimension(b)
    T = float(eltype(b.Λ))
    all(iszero, b.Λ) && return ExtendedGaussian(zeros(T, n), zeros(T, n, n), Matrix{T}(I, n, n))
    E = eigen(_symmetric_part(b.Λ))
    tol = _eig_tol(E.values)
    minimum(E.values) < -tol && throw(ArgumentError(
        "a precision must be positive semidefinite; it has the eigenvalue $(minimum(E.values))"))
    keep = E.values .> tol
    V, N = E.vectors[:, keep], E.vectors[:, .!keep]
    norm(N' * b.η) <= sqrt(eps(T)) * norm(b.η) || throw(ArgumentError(
        "an improper Gaussian has an information vector outside the range of its precision; it is \
         not a limit of Gaussians"))
    S = V * Diagonal(1 ./ E.values[keep]) * V'
    return ExtendedGaussian(S * b.η, (S + S') / 2, N)
end
function extended(b::DiracBelief)
    v = b.value isa AbstractVector ? float.(b.value) : [float(b.value)]
    return ExtendedGaussian(v, zeros(eltype(v), length(v), length(v)))
end
extended(b::ExtendedGaussian) = b
extended(b, n::Integer) = extended(b)
extended(::TrivialBelief, n::Integer) = ExtendedGaussian(zeros(n), zeros(n, n), Matrix(1.0I, n, n))
extended(::Nothing, n::Integer) = extended(TrivialBelief(), n)

"""
    GaussianBelief(b::ExtendedGaussian)

The canonical form of an extended Gaussian without hard directions: `Λ = Σ⁺`, `η = Λμ`, declared
improper when `b` is flat anywhere. A hard direction has infinite precision and no canonical
form: that is an `ArgumentError` (keep the `ExtendedGaussian`, or use a `DiracBelief` when every
direction is hard).
"""
function GaussianBelief(b::ExtendedGaussian)
    size(hard_directions(b), 2) == 0 || throw(ArgumentError(
        "an extended Gaussian with hard directions (no variance) has no canonical form"))
    η, Λ = _canonical_part(b)
    return GaussianBelief(η, Λ, size(b.D, 2) > 0)
end

"""
    pushforward(b::ExtendedGaussian, M, c = 0, Q = 0) -> ExtendedGaussian

The law of `M x + c + ε` for `x ~ b` and independent `ε ~ N(0, Q)`: mean `Mμ + c`, covariance
`MΣMᵀ + Q`, flat along `M·span(D)`. Exact for any matrix `M`, also rank-deficient or rectangular.
"""
function pushforward(b::ExtendedGaussian, M::AbstractMatrix, c = 0, Q = 0)
    size(M, 2) == length(b.μ) || throw(DimensionMismatch("M has $(size(M, 2)) columns, the belief dimension $(length(b.μ))"))
    m = size(M, 1)
    cv = c isa Number ? fill(float(c), m) : c
    Qm = Q isa Number ? Q * I(m) : Q
    return ExtendedGaussian(M * b.μ .+ cv, M * b.Σ * M' + Qm, M * b.D)
end

"""
    marginalise(b::ExtendedGaussian, idx) -> ExtendedGaussian

The law of the coordinates `idx` (a vector of indices or a range): the pushforward through the
selection matrix.
"""
function marginalise(b::ExtendedGaussian{T}, idx) where {T}
    n = length(b.μ)
    S = zeros(T, length(idx), n)
    for (r, j) in enumerate(idx)
        S[r, j] = one(T)
    end
    return pushforward(b, S)
end

# The product of two extended Gaussians as potentials on ℝⁿ: the intersection of their supports
# (each a hard constraint Hᵀx = Hᵀμ) with the precisions added on what is left free. Written
# x = x₀ + N z with N an orthonormal basis of the free directions, the quadratic part in z is
# NᵀΛN; its null space is flat in the product. Contradictory constraints are an ArgumentError.
#
# With weights, it is the geometric mixture a^wa · b^wb (damping, wa + wb = 1): precisions scale
# by the weights, and with `strict = false` the hard values are the weighted least-squares fit of
# both sides' constraints, the limit of precisions wa/ε and wb/ε, instead of a contradiction.
function _conjoin(a::ExtendedGaussian, b::ExtendedGaussian; wa::Real = 1, wb::Real = 1, strict::Bool = true)
    n = length(a.μ)
    n == length(b.μ) || throw(DimensionMismatch("beliefs of dimension $n and $(length(b.μ))"))
    Ha, Hb = hard_directions(a), hard_directions(b)
    C = vcat(sqrt(wa) .* Ha', sqrt(wb) .* Hb')
    c = vcat(sqrt(wa) .* (Ha' * a.μ), sqrt(wb) .* (Hb' * b.μ))
    ηa, Λa = _canonical_part(a)
    ηb, Λb = _canonical_part(b)
    return _from_potential(C, c, wa .* ηa .+ wb .* ηb, wa .* Λa .+ wb .* Λb;
                           strict, scale = max(norm(a.μ), norm(b.μ)))
end

# The extended Gaussian of the potential  [Cx = c] · exp(ηᵀx − xᵀΛx/2)  on ℝⁿ: hard where C
# constrains, Gaussian where Λ is positive on what C leaves free, flat elsewhere. With `strict`,
# constraints without a common solution are an ArgumentError; otherwise x₀ is their
# least-squares solution. `η` must have no component along the flat directions (true for every
# potential built from extended Gaussians: see extended_gaussian.md, "Product").
function _from_potential(C::AbstractMatrix, c::AbstractVector, η::AbstractVector, Λ::AbstractMatrix;
                         strict::Bool = true, scale::Real = 0)
    n = length(η)
    T = float(promote_type(eltype(C), eltype(c), eltype(η), eltype(Λ)))
    if size(C, 1) == 0
        x0, N = zeros(T, n), Matrix{T}(I, n, n)
    else
        F = svd(C; full = true)
        r = count(F.S .> _eig_tol(F.S))
        Ur, Sr, Vr = F.U[:, 1:r], F.S[1:r], F.V[:, 1:r]
        x0 = Vr * ((Ur' * c) ./ Sr)
        tol = sqrt(eps(T)) * max(norm(c), scale, one(T))
        !strict || norm(C * x0 - c) <= tol || throw(ArgumentError(
            "contradictory hard constraints: they fix a direction to different values \
             (residual $(norm(C * x0 - c)))"))
        N = F.V[:, (r + 1):n]
    end
    size(N, 2) == 0 && return ExtendedGaussian(x0, zeros(T, n, n))
    Λz = Symmetric(N' * Λ * N)
    ηz = N' * (η - Λ * x0)
    E = eigen(Λz)
    keep = E.values .> _eig_tol(E.values)
    V, W = E.vectors[:, keep], E.vectors[:, .!keep]
    Sz = V * Diagonal(1 ./ E.values[keep]) * V'
    μ = x0 + N * (Sz * ηz)
    return ExtendedGaussian(μ, N * ((Sz + Sz') / 2) * N', N * W)
end

"""
    preimage(b::ExtendedGaussian, M, c = 0) -> ExtendedGaussian

The likelihood ``x \\mapsto b(Mx + c)`` as an extended Gaussian on the domain of `M`: what `b`
says about `x` when it is a belief about `y = Mx + c`. Hard where `Mx + c` would have to lie in a
hard direction of `b`, Gaussian with precision ``M^\\top \\Sigma^+ M`` where `b` is Gaussian, flat along
`null(M)` and along the preimage of `b`'s flat directions. Exact for any `M`. With
[`pushforward`](@ref) it gives the message of a linear relation: `preimage(pushforward(...), Aₜ)`.
"""
function preimage(b::ExtendedGaussian, M::AbstractMatrix, c = 0)
    size(M, 1) == length(b.μ) || throw(DimensionMismatch("M has $(size(M, 1)) rows, the belief dimension $(length(b.μ))"))
    cv = c isa Number ? fill(float(c), size(M, 1)) : c
    H = hard_directions(b)
    η, Λ = _canonical_part(b)
    return _from_potential(H' * M, H' * (b.μ .- cv), M' * (η .- Λ * cv), M' * Λ * M; scale = norm(b.μ))
end

# The joint of independent extended Gaussians: means stacked, covariances and flat directions
# block-diagonal.
function _independent_join(bs::AbstractVector{<:ExtendedGaussian})
    T = mapreduce(b -> eltype(b.μ), promote_type, bs)
    ns = map(b -> length(b.μ), bs); ks = map(b -> size(b.D, 2), bs)
    n, k = sum(ns), sum(ks)
    μ = zeros(T, n); Σ = zeros(T, n, n); D = zeros(T, n, k)
    o = ok = 0
    for (b, ni, ki) in zip(bs, ns, ks)
        μ[o+1:o+ni] = b.μ; Σ[o+1:o+ni, o+1:o+ni] = b.Σ; D[o+1:o+ni, ok+1:ok+ki] = b.D
        o += ni; ok += ki
    end
    return ExtendedGaussian(μ, Σ, D)
end

# Damping (Mycelium's `damp`): the geometric mixture new^α · old^(1-α), the rule canonical damping
# of a GaussianBelief follows. Precisions are mixed; a direction flat in one side carries the
# other's precision scaled by its weight, so it becomes informative gradually; hard directions take
# the weighted average of the two sides' values.
_damp_extended(a::ExtendedGaussian, b::ExtendedGaussian, α::Real) =
    _conjoin(a, b; wa = α, wb = 1 - α, strict = false)

"""
    belief_mean(b::ExtendedGaussian)
    belief_cov(b::ExtendedGaussian)

The mean `μ` and covariance `Σ` of an extended Gaussian that is flat nowhere (hard directions
allowed: they have variance zero). Along a flat direction there is no mean, which is an
`ArgumentError`.
"""
function belief_mean(b::ExtendedGaussian)
    size(b.D, 2) == 0 || throw(ArgumentError("belief_mean: the extended Gaussian is flat along $(size(b.D, 2)) direction(s)"))
    return b.μ
end
function belief_cov(b::ExtendedGaussian)
    size(b.D, 2) == 0 || throw(ArgumentError("belief_cov: the extended Gaussian is flat along $(size(b.D, 2)) direction(s)"))
    return b.Σ
end
