"""
    Grassmann(k, m)

The real Grassmannian of `k`-dimensional subspaces of `ℝ^m`, with `1 ≤ k ≤ m`.
A point is represented by a real `k × m` matrix `Θ` with orthonormal rows.
Representatives `Θ` and `O * Θ` for orthogonal `O` describe the same point.
Geometry routines assume orthonormal input; use [`ispoint`](@ref) to check it.
"""
struct Grassmann
    k::Int
    m::Int
    function Grassmann(k::Integer, m::Integer)
        1 <= k <= m || throw(ArgumentError("Grassmann requires 1 ≤ k ≤ m"))
        return new(k, m)
    end
end

function _grassmann_size(M::Grassmann, A::AbstractMatrix)
    size(A) == (M.k, M.m) || throw(DimensionMismatch(
        "expected a $(M.k) × $(M.m) matrix, got $(size(A))"))
    return nothing
end

"""
    ispoint(M::Grassmann, Θ; atol = 1e-8)

Check shape, finiteness and `Θ * Θ' ≈ I` using absolute tolerance `atol`
(and zero relative tolerance). Negative or non-finite tolerances throw.
"""
function ispoint(M::Grassmann, Θ::AbstractMatrix{<:Real}; atol::Real = 1e-8)
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and non-negative"))
    return size(Θ) == (M.k, M.m) && all(isfinite, Θ) &&
           isapprox(Θ * Θ', LinearAlgebra.I; atol, rtol = 0)
end

"""
    project_tangent(M::Grassmann, Θ, G)

Return `G - (G * Θ') * Θ`. This is the horizontal lift of the Grassmann
tangent space at the orthonormal representative `Θ`: `Δ * Θ' = 0`.
It removes the vertical (row-basis rotation) directions from the Stiefel
tangent space; it is not the full Stiefel tangent projection.
Uses the Frobenius metric on horizontal lifts, with cost `O(k²m)`.
"""
function project_tangent(M::Grassmann, Θ::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real})
    _grassmann_size(M, Θ)
    _grassmann_size(M, G)
    return G - (G * Θ') * Θ
end

# A thin Q, without materialising an m × m orthogonal matrix. Fix the QR
# ambiguity by multiplying Q's columns by sign(diag(R)), so the new R is positive.
function _grassmann_qrows(M::Grassmann, A::AbstractMatrix{<:Real})
    all(isfinite, A) || throw(ArgumentError("QR input must be finite"))
    F = LinearAlgebra.qr(float.(A))
    d = LinearAlgebra.diag(F.R)
    threshold = max(M.k, M.m) * eps(eltype(d)) * LinearAlgebra.norm(F.R, Inf)
    all(x -> abs(x) > threshold, d) || throw(ArgumentError(
        "QR input is numerically rank-deficient (diagonal tolerance $threshold)"))
    Q = Matrix(F.Q * Matrix{eltype(d)}(LinearAlgebra.I, M.m, M.k))
    return Matrix(transpose(Q .* reshape(sign.(d), 1, :)))
end

"""
    retract(M::Grassmann, Θ, Δ)

Orthonormalise the rows of `Θ + Δ` by thin QR of its transpose, fixing signs
so `diag(R) > 0`. Thus `retract(M, Θ, zero(Θ)) ≈ Θ`. For a retraction step,
`Θ` must be orthonormal and `Δ` horizontal (see [`project_tangent`](@ref));
arbitrary displacements are also accepted as row orthonormalisation.

Throws `ArgumentError` on non-finite or numerically rank-deficient input:
any `abs(R[i,i]) ≤ max(k,m) * eps(T) * norm(R, Inf)` is rejected, where
`T` is the QR element type. This is a floating-point QR diagonal criterion,
not an exact rank certificate. Dense real CPU arrays are the supported use.
"""
function retract(M::Grassmann, Θ::AbstractMatrix{<:Real}, Δ::AbstractMatrix{<:Real})
    _grassmann_size(M, Θ)
    _grassmann_size(M, Δ)
    return _grassmann_qrows(M, transpose(Θ + Δ))
end

"""
    rand(rng::AbstractRNG, M::Grassmann)
    rand(M::Grassmann)

Draw a Haar-random Grassmann point, represented by orthonormal rows, using
thin QR of an `m × k` standard Gaussian matrix with positive-diagonal R.
Returns a dense Float64 matrix.
"""
Random.rand(rng::AbstractRNG, M::Grassmann) =
    _grassmann_qrows(M, Random.randn(rng, M.m, M.k))
Random.rand(M::Grassmann) = Random.rand(Random.default_rng(), M)

"""
    Retracted(rule, M::Grassmann) <: Mycelium.AbstractUpdateRule

Wrap `Mycelium.GradientDescent` or `Mycelium.Momentum` with horizontal
projection and QR retraction. All other rules throw `ArgumentError`.
Nesterov is excluded because its look-ahead leaves the manifold; a retracted
look-ahead and consistent gradient transport would need a separate contract.

`rule_get(r, state, Θ) = Θ`. `rule_put` projects the gradient, calls the inner
rule, projects its candidate displacement, and retracts. Momentum's new state
is the inner rule's plain `k × m` velocity matrix, transported by horizontal
projection at the new representative. It uses Mycelium's recurrence
`s′ = γ*s - η*grad` (heavy-ball momentum).

Mycelium has no rule-state initializer: supply `nothing` for gradient descent
(its state is passed through unchanged), or `zero(Θ)` for momentum. Works with
`Mycelium.OptimiserFactor(:θ, r)` and `Mycelium.optimiser_step` directly;
message emission uses `st = (p = Θ, state = state)` as in Mycelium.
"""
struct Retracted{R<:Mycelium.AbstractUpdateRule} <: Mycelium.AbstractUpdateRule
    rule::R
    manifold::Grassmann
    function Retracted(rule::R, M::Grassmann) where {R<:Mycelium.AbstractUpdateRule}
        rule isa Union{Mycelium.GradientDescent,Mycelium.Momentum} ||
            throw(ArgumentError("Retracted supports only GradientDescent and Momentum; Nesterov's look-ahead leaves the manifold"))
        return new{R}(rule, M)
    end
end

Mycelium.rule_get(::Retracted, state, Θ) = Θ

function Mycelium.rule_put(r::Retracted, state, Θ, G)
    grad = project_tangent(r.manifold, Θ, G)
    state′, candidate = Mycelium.rule_put(r.rule, state, Θ, grad)
    Δ = project_tangent(r.manifold, Θ, candidate - Θ)
    Θ′ = retract(r.manifold, Θ, Δ)
    if r.rule isa Mycelium.Momentum
        state′ = project_tangent(r.manifold, Θ′, state′)
    end
    return state′, Θ′
end
