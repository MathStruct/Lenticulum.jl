"""
    ExponentTable(N, d; basis=:monomial, homogeneous=false)

Columns of `exponents::Matrix{Int}` are all N-variable multi-indices of total
degree ≤ d (exactly d when homogeneous). Order: increasing total degree, then
descending lexicographic order, e.g. `(1,x,y,x²,xy,y²)` for N=2,d=2.
Chebyshev entries are products of first-kind T polynomials, intended for data
scaled to [-1,1]; the domain is not enforced. With `homogeneous=true` the
*indices* have degree d; Chebyshev polynomials need not be homogeneous.
"""
struct ExponentTable
    exponents::Matrix{Int}
    basis::Symbol
    degree::Int
    homogeneous::Bool
    function ExponentTable(N::Integer, d::Integer; basis::Symbol=:monomial, homogeneous::Bool=false)
        N > 0 && !(N isa Bool) || throw(ArgumentError("N must be a positive integer"))
        d >= 0 && !(d isa Bool) || throw(ArgumentError("d must be a nonnegative integer"))
        basis in (:monomial, :chebyshev) || throw(ArgumentError("unknown polynomial basis $basis"))
        cols = Vector{Int}[]
        α = zeros(Int, N)
        function visit(i, left)
            if i == N
                α[i] = left
                push!(cols, copy(α))
            else
                for a in left:-1:0
                    α[i] = a
                    visit(i+1, left-a)
                end
            end
        end
        for δ in (homogeneous ? (d:d) : (0:d))
            visit(1, δ)
        end
        return new(reduce(hcat, cols), basis, d, homogeneous)
    end
end

function _basis_values(tab::ExponentTable, x::AbstractVector)
    N, d = size(tab.exponents, 1), tab.degree
    length(x) == N || throw(DimensionMismatch("expected $N coordinates"))
    T = eltype(x)
    P, Q = ones(T, d+1, N), zeros(T, d+1, N)
    for j in 1:N
        if d >= 1
            P[2,j], Q[2,j] = x[j], one(T)
        end
        # U₀=1, U₁=2x; Q[n+1]=n U_{n-1} for Chebyshev.
        u0, u1 = one(T), x[j]+x[j]
        for n in 2:d
            if tab.basis === :monomial
                P[n+1,j] = P[n,j]*x[j]
                Q[n+1,j] = n*P[n,j]
            else
                P[n+1,j] = 2*x[j]*P[n,j]-P[n-1,j]
                Q[n+1,j] = n*u1
                u0, u1 = u1, 2*x[j]*u1-u0
            end
        end
    end
    return P, Q
end

"""    veronese(tab, x)

Evaluate the ordered polynomial basis. Monomial arithmetic preserves the input
element type (including integers and rationals); there is no scaling.
"""
function veronese(tab::ExponentTable, x::AbstractVector)
    P, _ = _basis_values(tab, x)
    E = tab.exponents
    return [prod(P[E[j,a]+1,j] for j in axes(E,1)) for a in axes(E,2)]
end

"""    veronese_jacobian(tab, x)

Exact polynomial Jacobian, m × N. Products avoid division by coordinates, so
zero coordinates and Chebyshev endpoints are supported.
"""
function veronese_jacobian(tab::ExponentTable, x::AbstractVector)
    P, Q = _basis_values(tab, x)
    E = tab.exponents
    return [Q[E[j,a]+1,j]*prod((P[E[l,a]+1,l] for l in axes(E,1) if l != j); init=one(eltype(P)))
            for a in axes(E,2), j in axes(E,1)]
end

"""
    basis_change(mon, cheb)

Exact BigInt matrix C with `veronese(cheb,x) == C*veronese(mon,x)` and
`Θ_mon = Θ_cheb*C`. Tables must have identical indices and the indicated bases.
Degree-exact Chebyshev tables generally contain lower-degree terms outside the
homogeneous monomial span; those incompatible tables throw (d≥2).
"""
function basis_change(mon::ExponentTable, cheb::ExponentTable)
    mon.basis === :monomial && cheb.basis === :chebyshev ||
        throw(ArgumentError("expected monomial and Chebyshev tables, in that order"))
    mon.exponents == cheb.exponents || throw(ArgumentError("tables must have identical indices"))
    d = cheb.degree
    polys = [zeros(BigInt,d+1) for _ in 0:d]
    polys[1][1] = 1
    d >= 1 && (polys[2][2] = 1)
    for n in 2:d
        polys[n+1][2:end] .= 2 .* polys[n][1:end-1]
        polys[n+1] .-= polys[n-1]
    end
    E = mon.exponents
    lookup = Dict(Tuple(E[:,a])=>a for a in axes(E,2))
    C = zeros(BigInt,size(E,2),size(E,2))
    for a in axes(E,2)
        for β in Iterators.product((0:E[j,a] for j in axes(E,1))...)
            c = prod(polys[E[j,a]+1][β[j]+1] for j in axes(E,1))
            iszero(c) && continue
            haskey(lookup,β) || throw(ArgumentError("Chebyshev polynomial leaves the homogeneous monomial span"))
            C[a,lookup[β]] = c
        end
    end
    return C
end
