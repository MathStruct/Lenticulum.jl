"""
    PolynomialFactor(dims::NamedTuple, tab::ExponentTable, k; Σ=I, λ=0, iterations=1, branches=:single)

Soft relation `Θ*veronese(tab,z) ≈ 0` over named channels, concatenated in
`keys(dims)` order. Parameters `(Θ=...,)` are seeded Haar Grassmann(k,m) rows;
state is empty. Σ is a data-space PSD covariance, λ≥0.

`energy(f,x,a,y,ps,st)` accepts disjoint NamedTuples in x/y (or nothing), like
LinearConstraintFactor. Scalars represent one-dimensional channels. For Sampson
scalarisation and Jacobians use the packed convention `x=nothing, y=z::Vector`:
`wrt=:y` differentiates **all** data coordinates. Named splits only support the
raw residual; pack them before scalarising.

One polarity per target channel supplies a single-branch Gaussian likelihood site.
Observed beliefs must be Dirac or proper Gaussian; the target expansion point is
its proper Gaussian/Dirac cavity mean, or zero without a cavity. `iterations ≥ 1`
relinearises at the current posterior mode (frozen-metric Gauss–Newton). The
returned site excludes the prior; `invert` pools it with the prior once.
Singular target precision is retained; singular residual covariance throws.
Set `branches=HomotopyBranches()` and load HomotopyContinuation for all isolated
roots of square systems; see [`polynomial_branches`](@ref). No game-level free
energy is supplied. `iterations` applies only to `branches=:single`.

Damping is λΘΘ′, invariant under invertible row gauges. Nonorthogonal basis
changes preserve undamped energy, but do not preserve this coefficient metric.
"""
struct PolynomialFactor{D,C,L,M} <: LenticulumCore.AbstractLenticulumFactor
    dims::D
    table::ExponentTable
    k::Int
    Σ::C
    λ::L
    iterations::Int
    branches::M
end

function PolynomialFactor(dims::NamedTuple, tab::ExponentTable, k::Integer;
                          Σ=LinearAlgebra.I, λ::Real=0, iterations::Integer=1, branches=:single)
    !isempty(dims) && all(d->d isa Integer && !(d isa Bool) && d>0, values(dims)) ||
        throw(ArgumentError("channel dimensions must be positive integers"))
    sum(values(dims)) == size(tab.exponents,1) || throw(DimensionMismatch("dims must sum to table dimension"))
    k isa Bool && throw(ArgumentError("k must be an integer dimension"))
    Grassmann(k,size(tab.exponents,2))
    cov = _polynomial_covariance(Σ, size(tab.exponents,1))
    LenticulumCore.SampsonNorm(;wrt=:y,Σ=cov,λ) # validation shared with core
    iterations >= 1 && !(iterations isa Bool) || throw(ArgumentError("iterations must be a positive integer"))
    branches === :single || branches isa HomotopyBranches || throw(ArgumentError("branches must be :single or HomotopyBranches()"))
    return PolynomialFactor(dims,tab,Int(k),cov,λ,Int(iterations),branches)
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

# A polarity declares an algorithm, not a rank condition at an unknown point.
function LenticulumCore.supported_polarities(f::PolynomialFactor)
    names = keys(f.dims)
    return map(names) do t
        LenticulumCore.Polarity(NamedTuple{names}(map(names) do n
            n === t ? LenticulumCore.Unobserved() : LenticulumCore.Observed()
        end))
    end
end
function LenticulumCore.supports_polarity(f::PolynomialFactor, p::LenticulumCore.Polarity)
    keys(p) == keys(f.dims) || return false
    return length(LenticulumCore.unobserved_channels(p)) == 1 &&
        length(LenticulumCore.observed_channels(p)) == length(f.dims)-1
end

"""
    PolynomialModel(factor, polarity)

Single-target Gaussian model of a polynomial relation. By default it is local;
`HomotopyBranches()` supplies a mixture over regular real isolated roots. Its solver uses
exact polynomial Jacobians and frozen residual covariances, with Gauss–Newton
curvature (not the full Laplace Hessian away from a zero residual). The single-branch default does not enumerate roots. Neither inference mode
differentiates the solve by the implicit function theorem.
"""
struct PolynomialModel{F<:PolynomialFactor,P} <: LenticulumCore.AbstractOpenModel
    factor::F
    polarity::P
end
LenticulumCore.ispure(::PolynomialModel) = true
LenticulumCore.latentspace(::PolynomialModel) = nothing
function LenticulumCore.assemble(f::PolynomialFactor, p::LenticulumCore.Polarity, ps, st)
    LenticulumCore.supports_polarity(f,p) || throw(ArgumentError("unsupported PolynomialFactor polarity"))
    return LenticulumCore.BayesianLens(PolynomialModel(f,p),
        LenticulumCore.SolverInversion((iterations=f.iterations,))), st
end

function _polynomial_vector(v, n)
    x = v isa Real ? [float(v)] : v
    x isa AbstractVector{<:Real} && all(isfinite,x) ||
        throw(ArgumentError("channel values must be finite real scalars/vectors"))
    length(x) == n || throw(DimensionMismatch("channel needs $n coordinates"))
    return float.(x)
end

function _polynomial_moments(b, n)
    if b isa LenticulumCore.DiracBelief
        return _polynomial_vector(b.value,n), zeros(n,n)
    elseif b isa LenticulumCore.GaussianBelief
        length(b.η) == n || throw(DimensionMismatch("Gaussian channel needs $n coordinates"))
        LenticulumCore.isproper(b) || throw(ArgumentError("observed Gaussian must be proper"))
        return _polynomial_vector(LenticulumCore.belief_mean(b),n), LenticulumCore.belief_cov(b)
    end
    throw(ArgumentError("observed channels require DiracBelief or proper GaussianBelief"))
end

# The site is defined even when it cannot itself be normalised. Never invert Λ.
function _polynomial_linearisation(f, target, inputs, x0, ps)
    size(ps.Θ) == (f.k,size(f.table.exponents,2)) || throw(DimensionMismatch("wrong Θ shape"))
    ps.Θ isa AbstractMatrix{<:Real} && all(isfinite,ps.Θ) ||
        throw(ArgumentError("Θ must be a finite real matrix"))
    names = keys(f.dims)
    ranges = let offset = 0
        map(values(f.dims)) do n
            r = offset+1:offset+n
            offset += n
            r
        end
    end
    blocks = NamedTuple{names}(ranges)
    parts = map(names) do n
        n === target ? (x0,zeros(length(x0),length(x0))) :
            _polynomial_moments(get(inputs,n,nothing),f.dims[n])
    end
    z = reduce(vcat,first.(parts))
    r = ps.Θ*veronese(f.table,z)
    J = ps.Θ*veronese_jacobian(f.table,z)
    A = f.Σ === nothing ? J*J' : J*f.Σ*J'
    for (n,(_,C)) in zip(names,parts)
        n === target && continue
        Jo = J[:,blocks[n]]
        A += Jo*C*Jo'
    end
    A += f.λ*(ps.Θ*ps.Θ')
    all(isfinite,A) && all(isfinite,r) && all(isfinite,J) ||
        throw(ArgumentError("polynomial linearisation must be finite"))
    return r, J[:,blocks[target]], A
end

function _polynomial_site(f, target, inputs, x0, ps)
    r, Ju, A = _polynomial_linearisation(f,target,inputs,x0,ps)
    ev = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(A))
    threshold = f.k*eps(eltype(ev))*maximum(ev)
    minimum(ev)>threshold || throw(ArgumentError(
        "residual covariance A is numerically singular: cannot form a Gaussian residual likelihood; use positive λ with full-row-rank Θ (no pseudoinverse)"))
    chol = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A))
    numerical_rank = LinearAlgebra.rank(Ju)
    Λ = Matrix(LinearAlgebra.Symmetric(Ju'*(chol\Ju)))
    h = Λ*x0-Ju'*(chol\r)
    return LenticulumCore.GaussianBelief(h,Λ), numerical_rank
end

function _polynomial_message(f, target, polarity, inputs, prior, ps, st)
    target in keys(f.dims) || throw(ArgumentError("unknown PolynomialFactor target :$target"))
    polarity isa LenticulumCore.Polarity && LenticulumCore.supports_polarity(f,polarity) &&
        only(LenticulumCore.unobserved_channels(polarity)) === target ||
        throw(ArgumentError("polarity must observe all channels except :$target"))
    f.branches === :single || return _homotopy_message(f.branches,f,target,inputs,prior,ps,st)
    q = f.dims[target]
    π = prior === nothing ? LenticulumCore.TrivialBelief() : prior
    if π isa LenticulumCore.GaussianBelief
        length(π.η) == q || throw(DimensionMismatch("target prior needs $q coordinates"))
    end
    x0 = π isa LenticulumCore.TrivialBelief ||
         (π isa LenticulumCore.GaussianBelief && LenticulumCore.isuninformative(π)) ?
         zeros(q) : first(_polynomial_moments(π,q))
    msg, numerical_rank = _polynomial_site(f,target,inputs,x0,ps)
    for _ in 2:f.iterations
        # No mode exists on a flat fibre without a proper cavity. Keep the site,
        # including rank-zero sites, rather than choosing a pseudoinverse mode.
        numerical_rank < q && !(π isa LenticulumCore.GaussianBelief && LenticulumCore.isproper(π)) &&
            !(π isa LenticulumCore.DiracBelief) && break
        post = Mycelium.combine(π,msg)
        x0 = first(_polynomial_moments(post,q))
        msg, numerical_rank = _polynomial_site(f,target,inputs,x0,ps)
    end
    return msg, st
end

"""
    Mycelium.factor_message(f::PolynomialFactor, target, polarity, inputs, prior, ps, st)

Single-branch extrinsic Gaussian site at the final expansion point `x0`:
`Λ = J_u'*(A\\J_u)`, `h = Λ*x0 - J_u'*(A\\r)`, with
`A = J*Σ*J' + sum(J_o*C_o*J_o') + λ*Θ*Θ'` (all data coordinates in Σ).
The Sampson metric is frozen per iteration; refreshing it is an approximation,
not differentiation of the full varying-metric Sampson objective. Rank-deficient
`J_u` gives an improper site, including zero precision at rank zero. Without a
proper posterior no further relinearisation is attempted. No damping or
backtracking is automatic, and iterations have no general descent guarantee.
The target input is ignored; the cavity affects expansion but is excluded from
the returned canonical parameters. `invert` combines it exactly once.
"""
Mycelium.factor_message(f::PolynomialFactor,target::Symbol,p,inputs,prior,ps,st) =
    _polynomial_message(f,target,p,inputs,prior,ps,st)

function LenticulumCore.likelihood_message(
    lens::LenticulumCore.BayesianLens{<:PolynomialModel,<:LenticulumCore.SolverInversion},
    π,inputs,ps,st,
)
    p = lens.model.polarity
    return _polynomial_message(lens.model.factor,only(LenticulumCore.unobserved_channels(p)),
                              p,inputs,π,ps,st)
end
function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:PolynomialModel,<:LenticulumCore.SolverInversion},
    π,inputs,ps,st,
)
    msg,st = LenticulumCore.likelihood_message(lens,π,inputs,ps,st)
    return Mycelium.combine(π === nothing ? LenticulumCore.TrivialBelief() : π,msg),st
end
