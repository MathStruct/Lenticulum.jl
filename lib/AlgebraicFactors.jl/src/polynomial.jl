"""
    PolynomialFactor(dims::NamedTuple, tab::ExponentTable, k; Σ=I, λ=0)

Soft relation `Θ*veronese(tab,z) ≈ 0` over named channels, concatenated in
`keys(dims)` order. Parameters `(Θ=...,)` are seeded Haar Grassmann(k,m) rows;
state is empty. Σ is a data-space PSD covariance, λ≥0.

`energy(f,x,a,y,ps,st)` accepts disjoint NamedTuples in x/y (or nothing), like
LinearConstraintFactor. Scalars represent one-dimensional channels. For Sampson
scalarisation and Jacobians use the packed convention `x=nothing, y=z::Vector`:
`wrt=:y` differentiates **all** data coordinates. Named splits only support the
raw residual; pack them before scalarising. No messages/polarities, inference,
normalised likelihood or game-level free energy are supplied.

Damping is λΘΘ′, invariant under invertible row gauges. Nonorthogonal basis
changes preserve undamped energy, but do not preserve this coefficient metric.
"""
struct PolynomialFactor{D,C,L} <: LenticulumCore.AbstractLenticulumFactor
    dims::D
    table::ExponentTable
    k::Int
    Σ::C
    λ::L
end

function PolynomialFactor(dims::NamedTuple, tab::ExponentTable, k::Integer;
                          Σ=LinearAlgebra.I, λ::Real=0)
    !isempty(dims) && all(d->d isa Integer && !(d isa Bool) && d>0, values(dims)) ||
        throw(ArgumentError("channel dimensions must be positive integers"))
    sum(values(dims)) == size(tab.exponents,1) || throw(DimensionMismatch("dims must sum to table dimension"))
    k isa Bool && throw(ArgumentError("k must be an integer dimension"))
    Grassmann(k,size(tab.exponents,2))
    cov = _polynomial_covariance(Σ, size(tab.exponents,1))
    LenticulumCore.SampsonNorm(;wrt=:y,Σ=cov,λ) # validation shared with core
    return PolynomialFactor(dims,tab,Int(k),cov,λ)
end

function _polynomial_covariance(Σ, N)
    cov = Σ isa LinearAlgebra.UniformScaling ? Matrix{typeof(float(Σ.λ))}(Σ,N,N) : Σ
    if cov !== nothing
        cov isa AbstractMatrix{<:Real} || throw(ArgumentError("Σ must be a real PSD covariance"))
        size(cov) == (N,N) || throw(DimensionMismatch("Σ must have size ($N,$N)"))
        LenticulumCore.SampsonNorm(;Σ=cov)
    end
    return cov
end

LuxCore.initialparameters(rng::AbstractRNG,f::PolynomialFactor) = (Θ=rand(rng,Grassmann(f.k,size(f.table.exponents,2))),)
LuxCore.initialstates(::AbstractRNG,::PolynomialFactor) = NamedTuple()
LuxCore.parameterlength(f::PolynomialFactor) = f.k*size(f.table.exponents,2)
LuxCore.statelength(::PolynomialFactor) = 0
LenticulumCore.islearnable(::PolynomialFactor) = true
LenticulumCore.channels(f::PolynomialFactor) = map((n,d)->LenticulumCore.Channel(n,d),keys(f.dims),values(f.dims))
LenticulumCore.pointwise_energyspace(f::PolynomialFactor) = LenticulumCore.EuclideanEnergySpace(f.k)
LenticulumCore.pointwise_scalarisation(f::PolynomialFactor) = LenticulumCore.SampsonNorm(;wrt=:y,Σ=f.Σ,λ=f.λ)

function _polynomial_data(f, x, y)
    if x === nothing && y isa AbstractVector
        y isa AbstractVector{<:Real} || throw(ArgumentError("packed data must be a real vector"))
        return y
    end
    x === nothing || x isa NamedTuple || throw(ArgumentError("x must be a NamedTuple or nothing"))
    y === nothing || y isa NamedTuple || throw(ArgumentError("y must be a NamedTuple or packed vector"))
    xs, ys = x === nothing ? (;) : x, y === nothing ? (;) : y
    isempty(intersect(keys(xs),keys(ys))) || throw(ArgumentError("channel values in x and y must be disjoint"))
    vals = merge(xs,ys)
    Set(keys(vals)) == Set(keys(f.dims)) || throw(ArgumentError("supply exactly the named channels $(keys(f.dims))"))
    parts = map(keys(f.dims)) do n
        v = vals[n] isa Real ? [vals[n]] : vals[n]
        v isa AbstractVector{<:Real} || throw(ArgumentError("channel :$n must be a real scalar/vector"))
        length(v) == f.dims[n] || throw(DimensionMismatch("wrong dimension for channel :$n"))
        v
    end
    return reduce(vcat,parts)
end

function LenticulumCore.energy(f::PolynomialFactor,x,a,y,ps,st)
    size(ps.Θ) == (f.k,size(f.table.exponents,2)) || throw(DimensionMismatch("wrong Θ shape"))
    return ps.Θ*veronese(f.table,_polynomial_data(f,x,y)),st
end

function LenticulumCore.energy_jacobian(f::PolynomialFactor,x,a,y,ps,st;wrt::Symbol=:y,ε::Real=1e-6)
    wrt === :y && x === nothing && y isa AbstractVector || throw(ArgumentError(
        "PolynomialFactor Jacobian needs x=nothing, packed y=z, wrt=:y (all coordinates)"))
    return ps.Θ*veronese_jacobian(f.table,y)
end
LenticulumCore.sampson_damping(::PolynomialFactor,x,a,y,ps,st) = ps.Θ*ps.Θ'
