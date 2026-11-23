# ---------------------------------------------------------------------------
# Matrix-free linear algebra for the implicit solver: the four places it meets J_FF.
#
#     Newton step       J_FF d = -r_F              GMRES on JVPs or MINRES on VJPs
#     adjoint           J_FFᵀ λ = -z̄_F             GMRES or MINRES on VJPs
#     stability         λ_min(sym J_FF) > 0        Lanczos on sym J_FF
#     Laplace           (sym J_FF)⁻¹ on a few coordinates, log det sym J_FF
#                                                  conjugate gradients, stochastic Lanczos quadrature
#
# None of them forms J_FF, so the cost of an iteration grows with the number of Krylov steps,
# not with the number of free coordinates. Every product is one JVP or VJP per noise level.
# GMRES and MINRES live in ImplicitSolvers. See `krylov.md`.
# ---------------------------------------------------------------------------

"""
    AbstractLinearSolve

How the implicit solver handles its linear algebra on the free coordinates:
[`DenseSolve`](@ref) (form the block ``J_{FF}``) or [`KrylovSolve`](@ref) (matrix-free).
"""
abstract type AbstractLinearSolve end

"""
    DenseSolve()

Form the free-coordinate block ``J_{FF}`` ([`prior_jacobian`](@ref) with `coords = free`) and use
dense factorisations: exact, and the right choice while few coordinates are free (one derivative
pass per free coordinate and noise level). The default.
"""
struct DenseSolve <: AbstractLinearSolve end

"""
    KrylovSolve(; rtol = 1e-8, maxiters = 200, restart = 40, lanczos = 30, probes = 8, seed = 0, method = :auto)

Matrix-free linear algebra, for queries with many free coordinates (generating an image):

- the Newton step and the adjoint solve ``J_{FF}d = -r_F`` and ``J_{FF}^\\top\\lambda = -\\bar z_F`` by
  GMRES or MINRES (relative tolerance `rtol`, at most `maxiters` Krylov steps). The default
  `method = :auto` selects MINRES where ``J_{FF}`` is symmetric, for an energy (an
  [`EnergyNetwork`](@ref) or a closed-form mixture, at any block times), and GMRES otherwise. `method = :minres` requires that symmetry and throws `ArgumentError`
  otherwise. `method = :gmres` always uses GMRES and `restart` applies only to GMRES.
  MINRES uses vector-Jacobian products ([`prior_vjp`](@ref)) for both solves. GMRES uses
  Jacobian-vector products ([`prior_jvp`](@ref)) for Newton and VJPs for the adjoint;
- stability (``\\operatorname{sym}J_{FF}\\succ0``) is decided by the smallest Ritz value of
  `lanczos` Lanczos steps on the symmetric part;
- the Laplace approximation solves for the covariance on the requested coordinates only, by
  conjugate gradients, and estimates ``\\log\\det`` by stochastic Lanczos quadrature with `probes`
  Rademacher probes of `lanczos` steps each.

Lanczos and the probes draw from `Xoshiro(seed)`, so the answers are deterministic. With at most
`lanczos` free coordinates the stability test is exact, and with at most `probes` the log
determinant is too (the probes are then the unit vectors, each with full Lanczos depth
regardless of the `lanczos` setting). Otherwise both are estimates: the
smallest Ritz value approaches ``\\lambda_{\\min}`` from above, so a barely unstable point can be
reported stable (`krylov.md` §3).
"""
Base.@kwdef struct KrylovSolve{T<:Real} <: AbstractLinearSolve
    rtol::T = 1e-8
    maxiters::Int = 200
    restart::Int = 40
    lanczos::Int = 30
    probes::Int = 8
    seed::Int = 0
    method::Symbol = :auto

    function KrylovSolve(rtol::T, maxiters, restart, lanczos, probes, seed, method) where {T<:Real}
        method in (:auto, :gmres, :minres) ||
            throw(ArgumentError("KrylovSolve method must be :auto, :gmres or :minres, got $method"))
        return new{T}(rtol, maxiters, restart, lanczos, probes, seed, method)
    end
end

KrylovSolve(rtol::Real, maxiters, restart, lanczos, probes, seed) =
    KrylovSolve(rtol, maxiters, restart, lanczos, probes, seed, :auto)

_shared_solver(::DenseSolve) = ImplicitSolvers.DenseSolve()
_shared_solver(ls::KrylovSolve) = ImplicitSolvers.GMRES(; ls.restart, ls.maxiters)
_linear_tolerance(ls::KrylovSolve) = ImplicitSolvers.Tolerance(; atol = 0, ls.rtol)

function _krylov_solver(ls::KrylovSolve, m, ρ)
    ls.method === :gmres && return _shared_solver(ls)
    _symmetric_block(m) && return ImplicitSolvers.MINRES(; ls.maxiters)
    ls.method === :minres && throw(ArgumentError(
        "KrylovSolve(method = :minres) requires a symmetric free block: an ImplicitDiffusion whose predictor is an energy (EnergyNetwork or GaussianMixtureEps)"))
    return _shared_solver(ls)
end

function _dense_adjoint_solve(ls::DenseSolve, At, b)
    x, report = ImplicitSolvers.adjoint_solve(At, b, _shared_solver(ls))
    if !isempty(b) && report.iterations == 0
        report.reason === :nonfinite && throw(ArgumentError("nonfinite dense adjoint system"))
        throw(LinearAlgebra.SingularException(0))
    end
    return x
end

# --- the operators on the free coordinates --------------------------------------------

# Is J_FF symmetric? For an energy (an `EnergyNetwork`, or a mixture's exact ε) the node's noise
# Jacobian is diag(σ) H diag(α) and the field weights coordinate i by c α_i/σ_i (`_node_coefs`),
# so each node adds c diag(α) H diag(α): symmetric at any block times (P37; checked in
# test/p37_minres.jl). The predictor's Jacobian alone is not symmetric under unequal times (LP-018).
_symmetric_block(m) = m isa ImplicitDiffusion && _has_energy(m)

# x ↦ J_FF x, x ↦ J_FFᵀ x and the symmetric part, all with the precision diagonal, at z
function _free_operators(m, z, ρ, ps, st)
    free = .!isinf.(ρ)
    ρ²F = ρ[free] .^ 2
    T = eltype(float(z))
    embed(v) = (u = zeros(T, length(z)); u[free] .= v; u)
    J(v) = prior_jvp(m, z, embed(v), ps, st)[free] .+ ρ²F .* v
    Jt(v) = prior_vjp(m, z, embed(v), ps, st)[free] .+ ρ²F .* v
    S(v) = _symmetric_block(m) ? J(v) : (J(v) .+ Jt(v)) ./ 2      # an energy's field is symmetric
    return (J = J, Jt = Jt, S = S, n = count(free))
end

# --- conjugate gradients ---------------------------------------------------------------------

function _cg(A, b; rtol = 1e-8, maxiters = 200)
    T = eltype(float(b))
    x = zeros(T, length(b))
    r = copy(float.(b)); p = copy(r)
    rr = dot(r, r)
    bn = sqrt(rr)
    bn == 0 && return (x, true)
    for _ in 1:maxiters
        Ap = A(p)
        pAp = dot(p, Ap)
        pAp > 0 || return (x, false)                   # not positive definite on this direction
        a = rr / pAp
        x .+= a .* p
        r .-= a .* Ap
        rr₊ = dot(r, r)
        sqrt(rr₊) ≤ rtol * bn && return (x, true)
        p .= r .+ (rr₊ / rr) .* p
        rr = rr₊
    end
    return (x, false)
end

# --- Lanczos ----------------------------------------------------------------------------------

# k steps of Lanczos on the symmetric S from q, with full reorthogonalisation: the tridiagonal
# matrix's diagonal and off-diagonal (shorter if an invariant subspace is found first)
function _lanczos(S, q, k)
    T = eltype(float(q))
    n = length(q)
    k = min(k, n)
    Q = zeros(T, n, k)
    α = T[]; β = T[]
    nq = norm(q)
    nq == 0 && return (α, β)
    q = q ./ nq
    for j in 1:k
        Q[:, j] .= q
        w = S(q)
        push!(α, dot(q, w))
        for _ in 1:2
            Qj = view(Q, :, 1:j)
            w .-= Qj * (Qj' * w)
        end
        b = norm(w)
        (j == k || b ≤ sqrt(eps(T)) * (abs(α[end]) + norm(α) / sqrt(j))) && break
        push!(β, b)
        q = w ./ b
    end
    return (α, β)
end

# the smallest Ritz value of sym J_FF: exact once the Krylov space is the whole space
function _lambda_min(S, n, ls::KrylovSolve)
    n == 0 && return Inf
    α, β = _lanczos(S, randn(Random.Xoshiro(ls.seed), n), ls.lanczos)
    return LinearAlgebra.eigmin(LinearAlgebra.SymTridiagonal(α, β))
end

# log det S by stochastic Lanczos quadrature: E[vᵀ log(S) v] over Rademacher probes, or the exact
# trace over unit vectors when there are no more coordinates than probes
function _logdet(S, n, ls::KrylovSolve)
    n == 0 && return 0.0
    rng = Random.Xoshiro(ls.seed + 1)
    exact = n ≤ ls.probes
    probes = exact ? [(e = zeros(n); e[i] = 1; e) for i in 1:n] : [rand(rng, (-1.0, 1.0), n) for _ in 1:ls.probes]
    total = 0.0
    for v in probes
        α, β = _lanczos(S, v, exact ? n : ls.lanczos)
        θ, U = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(α, β))
        all(>(0), θ) || throw(ArgumentError("log det: the symmetric part of J_FF is not positive definite"))
        total += dot(v, v) * sum(U[1, :] .^ 2 .* log.(θ))
    end
    return exact ? total : total / ls.probes             # Hutchinson: E[v vᵀ] = I
end
