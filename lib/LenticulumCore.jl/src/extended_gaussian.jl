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
`GaussianBelief(b)`), [`flat_directions`](@ref), [`hard_directions`](@ref),
[`support_entropy`](@ref). The product of two beliefs is Mycelium's `combine`. A belief with
flat directions has no density and no entropy; one with hard directions has them only relative
to the measure on its support (`support_entropy`).
"""
struct ExtendedGaussian{T<:Real} <: AbstractBelief
    μ::Vector{T}
    Σ::Matrix{T}
    D::Matrix{T}
    function ExtendedGaussian(μ::AbstractVector, Σ::AbstractMatrix, D::AbstractMatrix)
        n = length(μ)
        size(Σ) == (n, n) || throw(DimensionMismatch("μ has length $n but Σ is $(size(Σ))"))
        size(D, 1) == n || throw(DimensionMismatch("μ has length $n but D has $(size(D, 1)) rows"))
        T = float(promote_type(eltype(μ), eltype(Σ), eltype(D)))
        _nearly_symmetric(stop_gradient(Σ)) || throw(ArgumentError("Σ must be symmetric"))
        B = _orth(Matrix{T}(D))
        # Project onto the complement of span(D) through an orthonormal basis U of it, not with
        # I − BBᵀ: when D spans everything U is empty and Σ is exactly zero, where I − BBᵀ would
        # leave a residue of 1e-30 that the relative tolerances below take for a variance.
        U = _complement(B)
        Σu = Matrix(_symmetric_part(U' * Matrix{T}(Σ) * U))
        _check_psd(Σu, "Σ must be positive semidefinite")
        S = U * Σu * U'
        return new{T}(U * (U' * Vector{T}(μ)), (S + S') / 2, B)
    end
end
ExtendedGaussian(μ::AbstractVector, Σ::AbstractMatrix) = ExtendedGaussian(μ, Σ, zeros(eltype(μ), length(μ), 0))

dimension(b::ExtendedGaussian) = length(b.μ)
isuninformative(b::ExtendedGaussian) = size(b.D, 2) == length(b.μ)

# Eigenvalues (or singular values) below this count as zero: the tolerance of `_split_moments`.
_eig_tol(λ) = isempty(λ) ? 0.0 : 16 * length(λ) * eps(float(eltype(λ))) * maximum(abs, λ)

# --- Subspaces, and their derivatives at constant rank (P4) -------------------------------------
#
# Every subspace an extended Gaussian holds or computes (its flat and hard directions, the range of
# a covariance or a precision, the solution set of stacked constraints) comes from a decomposition
# of the *values*, and its rank is decided there by `_eig_tol`. Under automatic differentiation
# (first-order ForwardDiff duals, Zygote) the bases then move by the constant-rank formulas, written
# with the tangent Δ = X − stop_gradient(X), which is zero in value: a basis of a subspace and one
# of its complement turn only into each other, to first order, and no eigenvector or singular
# vector is differentiated, so repeated eigenvalues are no difficulty (the formulas of P25 and P26).
# With floats Δ is exactly zero and every result is the decomposition's. A tangent that changes a
# rank to first order has no derivative (the value jumps there): under ForwardDiff it is an
# ArgumentError (`_check_constant_rank`, LenticulumCoreForwardDiffExt); a Zygote cotangent cannot
# see it, and the constant-rank derivative is returned. See `extended_gaussian.md`, "Derivatives".

_tangent(X) = X .- stop_gradient(X)

# The values of X as a matrix of floats. Nested duals (a Hessian) are refused: their values are
# duals, which the decompositions below take no derivative of.
function _values(X)
    X0 = stop_gradient(X)
    eltype(X0) <: AbstractFloat || throw(ArgumentError(
        "extended Gaussians take floats and first-order ForwardDiff duals of floats; got $(eltype(X))"))
    return X0 isa AbstractMatrix ? Matrix(X0) : X0
end

# The decompositions, of values only (not differentiated: LenticulumCoreChainRulesCoreExt). The
# eigendecomposition of a symmetric matrix split at `_eig_tol` (range V with eigenvalues λ, null
# space N, all eigenvalues), and the full SVD with its rank.
function _value_eigen(S)
    E = eigen(_symmetric_part(_values(S)))
    keep = E.values .> _eig_tol(E.values)
    return E.vectors[:, keep], E.values[keep], E.vectors[:, .!keep], E.values
end
function _value_svd(M)
    F = svd(_values(M); full = true)
    return F.U, F.S, F.V, count(F.S .> _eig_tol(F.S))
end

# A symmetric matrix that must be positive semidefinite up to rounding, checked on its values.
function _check_psd(S, what)
    isempty(S) && return nothing
    λ = _value_eigen(S)[4]
    minimum(λ) >= -_eig_tol(λ) || throw(ArgumentError("$what; it has the eigenvalue $(minimum(λ))"))
    return nothing
end

# Refuses a tangent that changes a rank to first order: X = Lᵀ Δ R with L, R the left and right
# null spaces at the value must vanish. Floats and Zygote: nothing to check (ForwardDiff: the ext).
_check_constant_rank(X, Δ, scale, what) = nothing
# the size of a spectrum of values (not differentiated: Zygote has no rule for `maximum` with `init`)
_scale(λ) = maximum(abs, λ; init = 0.0)
_logsum(λ) = sum(log, λ; init = 0.0)

# A symmetric positive semidefinite S split at constant rank: V a basis of its range and N of its
# null space, at the value V₀, N₀ with eigenvalues λ, moved to first order by
#     V = V₀ + N₀N₀ᵀΔV₀λ⁻¹,   N = N₀ − V₀λ⁻¹V₀ᵀΔN₀,
# and K = V₀ᵀSV₀, positive definite, whose first-order change V₀ᵀΔV₀ is that of the nonzero
# eigenvalues. Returns (V, K, N).
function _psd_split(S, what)
    V0, λ0, N0, _ = _value_eigen(S)
    Δ = _tangent(S)
    _check_constant_rank(N0' * Δ * N0, Δ, _scale(λ0), what)
    # S = 0 (no nonzero eigenvalue): at constant rank it stays 0, N₀ spans everything and does
    # not move. Returned before the broadcasts by λ₀, whose Zygote pullback fails on an empty λ₀.
    isempty(λ0) && return V0, zeros(eltype(S), 0, 0), N0
    V = V0 .+ N0 * ((N0' * Δ * V0) ./ transpose(λ0))
    N = N0 .- V0 * ((V0' * Δ * N0) ./ λ0)
    K = V0' * S * V0
    return V, (K + K') / 2, N
end

# The pseudo-inverse V K⁻¹ Vᵀ of a split: its derivative is that of S⁺ at constant rank,
# −S⁺ΔS⁺ + NNᵀΔS⁺² + S⁺²ΔNNᵀ.
function _split_pinv(V, K)
    n = size(V, 1)
    size(K, 1) == 0 && return zeros(eltype(V), n, n)
    S = V * _spd_inv(K, "the range part of a covariance") * V'
    return (S + S') / 2
end

# An orthonormal basis of the column space of M (columns below the tolerance dropped). With
# M = U S Vᵀ at the value and B₀ the first r columns of U: B = B₀ + (I − B₀B₀ᵀ) Δ Vᵣ S⁻¹.
function _orth(M::AbstractMatrix)
    T = float(eltype(M))
    size(M, 2) == 0 && return zeros(T, size(M, 1), 0)
    U0, S0, V0, r = _value_svd(M)
    Δ = _tangent(M)
    _check_constant_rank(U0[:, (r + 1):end]' * Δ * V0[:, (r + 1):end], Δ, _scale(S0),
                         "the span of an extended Gaussian's flat directions")
    B0 = U0[:, 1:r]
    X = Δ * (V0[:, 1:r] ./ transpose(S0[1:r]))
    return B0 .+ (X .- B0 * (B0' * X))
end

# An orthonormal basis of the orthogonal complement of the column space of B (B of full column
# rank; orthonormal where it is called). With W₀ the complement at the value and B⁺ its
# pseudo-inverse there: W = W₀ − (B⁺)ᵀΔᵀW₀.
function _complement(B::AbstractMatrix)
    n = size(B, 1)
    T = float(eltype(B))
    size(B, 2) == 0 && return Matrix{T}(I, n, n)
    U0, S0, V0, r = _value_svd(B)
    Δ = _tangent(B)
    W0 = U0[:, (r + 1):n]
    _check_constant_rank(W0' * Δ * V0[:, (r + 1):end], Δ, _scale(S0),
                         "the complement of a subspace")
    Bp = V0[:, 1:r] * (U0[:, 1:r]' ./ S0[1:r])
    return W0 .- Bp' * (Δ' * W0)
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
    S = U' * b.Σ * U
    _, _, N = _psd_split((S + S') / 2, "the hard directions of an extended Gaussian")
    return U * N
end

"""
    support_entropy(b::ExtendedGaussian) -> Real

The differential entropy of `b` relative to Lebesgue measure on its support, the affine subspace
``\\mu + \\operatorname{range}(\\Sigma)``, in orthonormal coordinates:

```math
H \\;=\\; \\tfrac r2\\,(1 + \\log 2\\pi) \\;+\\; \\tfrac12 \\textstyle\\sum_{i=1}^{r} \\log \\lambda_i ,
```

over the ``r`` nonzero eigenvalues of ``\\Sigma``. Without hard directions it is the Gaussian
entropy; a point (every direction hard) has entropy `0`, the convention for clamped variables.
It is the finite part of the entropy of ``\\mathcal N(\\mu, \\Sigma + \\varepsilon I)`` as
``\\varepsilon \\to 0``, after removing ``\\tfrac{n-r}{2}\\log(2\\pi e\\varepsilon)``; the Bethe free
energy of a graph with hard constraints is finite with it (`extended_gaussian.md`, "Entropy").
A belief with flat directions has no entropy (it is improper): `ArgumentError`.
"""
function support_entropy(b::ExtendedGaussian{T}) where {T}
    size(b.D, 2) == 0 || throw(ArgumentError(
        "an extended Gaussian with flat directions is improper and has no entropy"))
    isempty(b.Σ) && return zero(T)
    _, K, _ = _psd_split(b.Σ, "the support of an extended Gaussian")
    r = size(K, 1)
    r == 0 && return zero(T)
    # log det K = Σ log λᵢ: K is diagonal at the value, up to rounding
    return r * (1 + log(2π)) / 2 + logdet(_chol(K, "the covariance on the support")) / 2
end

# The belief in coordinates along its non-hard directions: x = x0 + U s, with x0 the point it
# fixes along its hard directions H (x0 = HHᵀμ) and U an orthonormal basis of H⊥ (its Gaussian
# and flat directions). In s it is the Gaussian part in canonical form, (η_s, Λ_s) = (Uᵀη, UᵀΛU)
# with (η, Λ) = `_canonical_part(b)`: zero precision along the flat directions. Substituting the
# hard directions out is how a factor with a smooth density takes a hard input, as it takes a
# point (`extended_gaussian.md`, "Entropy").
function _hard_reduction(b::ExtendedGaussian)
    H = hard_directions(b)
    U = _complement(H)
    η, Λ = _canonical_part(b)
    Λs = U' * Λ * U
    return H * (H' * b.μ), U, U' * η, (Λs + Λs') / 2
end

# The quadratic part on the Gaussian directions: Λ = Σ⁺ (zero on the flat and hard directions)
# and η = Λμ.
function _canonical_part(b::ExtendedGaussian)
    V, K, _ = _psd_split(b.Σ, "the covariance of an extended Gaussian")
    Λ = _split_pinv(V, K)
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

`η` and `Λ` may be first-order ForwardDiff duals (or tracked by Zygote): the split is taken at the
value and moves at constant rank (`extended_gaussian.md`, "Derivatives").
"""
function extended(b::GaussianBelief)
    n = dimension(b)
    T = float(promote_type(eltype(b.η), eltype(b.Λ)))
    if all(iszero, _values(b.Λ))
        Δ = _tangent(b.Λ)
        _check_constant_rank(Δ, Δ, 0.0, "a zero precision")
        return ExtendedGaussian(zeros(T, n), zeros(T, n, n), Matrix{T}(I, n, n))
    end
    return ExtendedGaussian(_flat_split(b.η, b.Λ)...)
end

# The parts of an improper Gaussian (η, Λ) as an extended Gaussian: the mean Λ⁺η, the covariance
# Λ⁺ and an orthonormal basis N of null(Λ), with the tolerances and refusals of `extended`. η and
# Λ may carry derivatives (P21: only η dual, as in Lenticulum's `_improper_split`; P4: Λ too).
function _flat_split(η::AbstractVector, Λ::AbstractMatrix)
    _check_psd(Λ, "a precision must be positive semidefinite")
    V, K, N = _psd_split(Λ, "a flat precision")
    # on η's values: `norm` of duals whose value is exactly zero is NaN (its scaling divides 0 by
    # 0), which a component N'η that rounds to 0 reached (P21)
    ηv, Nv = stop_gradient(η), stop_gradient(N)
    Tv = eltype(_values(Λ))
    norm(Nv' * ηv) <= sqrt(eps(float(Tv))) * norm(ηv) || throw(ArgumentError(
        "an improper Gaussian has an information vector outside the range of its precision; it is \
         not a limit of Gaussians"))
    S = _split_pinv(V, K)
    return (S * η, S, N)
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
    S = M * b.Σ * M'
    # a scalar noise as `S + Q*I`, not `+ Q * I(m)`: Zygote's pullback of the `Diagonal` fails (P4)
    Σy = Q isa Number ? (iszero(Q) ? S : S + Q * I) : S + Q
    return ExtendedGaussian(M * b.μ .+ cv, Σy, M * b.D)
end

"""
    marginalise(b::ExtendedGaussian, idx) -> ExtendedGaussian

The law of the coordinates `idx` (a vector of indices or a range): the pushforward through the
selection matrix.
"""
function marginalise(b::ExtendedGaussian, idx)
    n = length(b.μ)
    S = zeros(length(idx), n)
    for (r, j) in enumerate(idx)
        S[r, j] = 1.0
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
                           strict, scale = max(norm(stop_gradient(a.μ)), norm(stop_gradient(b.μ))))
end

# The extended Gaussian of the potential  [Cx = c] · exp(ηᵀx − xᵀΛx/2)  on ℝⁿ: hard where C
# constrains, Gaussian where Λ is positive on what C leaves free, flat elsewhere. With `strict`,
# constraints without a common solution are an ArgumentError; otherwise x₀ is their
# least-squares solution. `η` must have no component along the flat directions (true for every
# potential built from extended Gaussians: see extended_gaussian.md, "Product").
#
# x₀ = C⁺c, the least-squares solution of least norm; with C = UᵣSVᵣᵀ at the value, P = C⁺ there,
# N₀ = null(C) and Δ the tangent of C, at constant rank (`extended_gaussian.md`, "Derivatives"):
#     x₀ = Pc − PΔPc + PPᵀΔᵀ(I − UᵣUᵣᵀ)c + N₀N₀ᵀΔᵀPᵀPc,     N = N₀ − PΔN₀.
function _from_potential(C::AbstractMatrix, c::AbstractVector, η::AbstractVector, Λ::AbstractMatrix;
                         strict::Bool = true, scale::Real = 0)
    n = length(η)
    T = float(promote_type(eltype(C), eltype(c), eltype(η), eltype(Λ)))
    if size(C, 1) == 0
        x0, N = zeros(T, n), Matrix{T}(I, n, n)
    else
        U0, S0, V0, r = _value_svd(C)
        Δ = _tangent(C)
        Ur, N0 = U0[:, 1:r], V0[:, (r + 1):n]
        _check_constant_rank(U0[:, (r + 1):end]' * Δ * N0, Δ, _scale(S0),
                             "the hard constraints of a product")
        P = V0[:, 1:r] * (Ur' ./ S0[1:r])
        Pc = P * c
        x0 = Pc .- P * (Δ * Pc) .+ P * (P' * (Δ' * (c .- Ur * (Ur' * c)))) .+ N0 * (N0' * (Δ' * (P' * Pc)))
        Tv = eltype(S0)
        cv, xv = stop_gradient(c), P * stop_gradient(c)
        res = norm(_values(C) * xv - cv)
        tol = sqrt(eps(Tv)) * max(norm(cv), scale, one(Tv))
        !strict || res <= tol || throw(ArgumentError(
            "contradictory hard constraints: they fix a direction to different values \
             (residual $res)"))
        N = N0 .- P * (Δ * N0)
    end
    size(N, 2) == 0 && return ExtendedGaussian(x0, zeros(T, n, n))
    Λz = N' * Λ * N
    ηz = N' * (η - Λ * x0)
    V, K, W = _psd_split((Λz + Λz') / 2, "the precision of a product")
    Sz = _split_pinv(V, K)
    μ = x0 + N * (Sz * ηz)
    return ExtendedGaussian(μ, N * Sz * N', N * W)
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
    return _from_potential(H' * M, H' * (b.μ .- cv), M' * (η .- Λ * cv), M' * Λ * M;
                           scale = norm(stop_gradient(b.μ)))
end

# The joint of independent extended Gaussians (a vector or a tuple): means stacked, covariances
# and flat directions block-diagonal. Built by `vcat` and `cat`, not by writes into zero arrays:
# Zygote cannot differentiate those (P4).
function _independent_join(bs::Union{Tuple,AbstractVector})
    μ = reduce(vcat, map(b -> b.μ, bs))
    Σ = _block_diagonal(map(b -> b.Σ, bs))
    D = _block_diagonal(map(b -> b.D, bs))
    return ExtendedGaussian(μ, Σ, D)
end
_block_diagonal(blocks) = length(blocks) == 1 ? Matrix(first(blocks)) : Matrix(cat(blocks...; dims = (1, 2)))

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
