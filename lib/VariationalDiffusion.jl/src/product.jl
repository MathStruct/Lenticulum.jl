# ---------------------------------------------------------------------------
# Several learned relations pooled: the product of their densities.
#
# Each implicit learner's field g_i is (for an ideal model) the gradient of a smoothed negative
# log-density, so the weighted sum Σ β_i g_i is the field of the product Π p_i^{β_i}: its stable
# roots are where all relations hold at once, e.g. the intersection points of a learned circle
# and a learned line. Every query function works on the product unchanged.
#
# A factor may see a subset of the variables (`relation => coords`); the joint is then the
# factor-graph product p_A(x,y) p_B(y,z) / p_S(y), with the separator p_S a relation on the
# shared coordinates and weight -1, so that the shared marginal is not counted twice.
#
# See `product.md` and `Composing Diffusion Factors` in the vault.
# ---------------------------------------------------------------------------

"""
    ProductRelation(relations...; weights = ones, dim = nothing)

Several implicit relations pooled: the field is the weighted sum ``\\sum_i \\beta_i\\,g_i`` of the
relations' fields, i.e. the field of the product of their densities ``\\prod_i p_i^{\\beta_i}``.
Its stable roots are the configurations that satisfy all relations together, e.g. the
intersection points of a learned circle and a learned line. The weights ``\\beta_i`` set how much
each relation is trusted (a tempering exponent).

**Variables.** A plain relation sees the whole joint space. A pair `relation => coords` sees
only the coordinates `coords` of the joint state, in that order, so relations on different but
overlapping variables compose:

```julia
# p(x, y, z) = p_A(x, y) p_B(y, z) / p_S(y)
ProductRelation(mA => [1, 2], mB => [2, 3], mS => [2]; weights = [1, 1, -1])
```

A **separator** is a relation with a negative weight, normally ``-1`` on the shared coordinates:
without it the shared marginal is counted once per factor that sees it (``p_S^2`` instead of
``p_S`` above), which sharpens it and reweights its modes. Every coordinate must be seen by a
relation, and the weights of the relations that see a coordinate must sum to a positive number.
`dim` is the dimension of the joint state; by default the largest coordinate used (or the
relations' own dimension when none has `coords`).

Parameters and states are **tuples**, one entry per relation, in order: `ps = (ps₁, ps₂)`. Every
query function accepts a product: [`implicit_infer`](@ref), [`implicit_roots`](@ref),
[`implicit_laplace`](@ref), [`implicit_mixture`](@ref), [`implicit_pullback`](@ref) (which returns
a tuple of parameter cotangents), and [`implicit_sample`](@ref) when all relations share a noise
schedule. Products may be nested.

Pooling smoothed densities is exact only at zero noise: at the field's noise levels the
product of the smoothed densities is not the smoothed product, so intersections are found
with the same kind of smoothing bias as single relations, and sampling needs correction steps
(`product.md` §3, §5).
"""
struct ProductRelation{M<:Tuple,W<:AbstractVector} <: AbstractImplicitRelation
    relations::M
    weights::W
    coords::Vector{Vector{Int}}        # the joint coordinates each relation sees, in its order
    dim::Int
end

_ondevice(m::ProductRelation) = any(_ondevice, m.relations)

_split_factor(r::AbstractImplicitRelation) = (r, nothing)
_split_factor(p::Pair{<:AbstractImplicitRelation}) = (p.first, collect(Int, p.second))

# the dimension of a relation's state, where it is known
_dim(m::ImplicitDiffusion) = size(m.nodes.ε, 1)
_dim(m::ProductRelation) = m.dim
_dim(::AbstractImplicitRelation) = nothing

function ProductRelation(factors::Union{AbstractImplicitRelation,Pair{<:AbstractImplicitRelation}}...;
                         weights = ones(length(factors)), dim::Union{Nothing,Integer} = nothing)
    isempty(factors) && throw(ArgumentError("a product needs at least one relation"))
    length(weights) == length(factors) || throw(ArgumentError("need one weight per relation"))
    all(!iszero, weights) || throw(ArgumentError("weights must be nonzero"))
    rels = Tuple(first(_split_factor(f)) for f in factors)
    given = [last(_split_factor(f)) for f in factors]
    n = if dim !== nothing
        Int(dim)
    elseif any(!isnothing, given)
        maximum(maximum(c) for c in given if c !== nothing && !isempty(c))
    else
        ds = filter(!isnothing, [_dim(r) for r in rels])
        isempty(ds) ? throw(ArgumentError("cannot infer the joint dimension; pass `dim`")) : first(ds)
    end
    coords = [c === nothing ? collect(1:n) : c for c in given]
    for (r, c) in zip(rels, coords)
        allunique(c) && all(i -> 1 ≤ i ≤ n, c) ||
            throw(ArgumentError("coordinates $c are not distinct indices into a $n-dimensional state"))
        d = _dim(r)
        d === nothing || d == length(c) ||
            throw(DimensionMismatch("a relation of dimension $d is given $(length(c)) coordinates"))
    end
    β = float.(collect(weights))
    for i in 1:n
        seen = [k for k in eachindex(coords) if i in coords[k]]
        isempty(seen) && throw(ArgumentError("coordinate $i is seen by no relation"))
        sum(β[seen]) > 0 || throw(ArgumentError(
            "the weights of the relations that see coordinate $i sum to $(sum(β[seen])); a separator must not outweigh the factors"))
    end
    return ProductRelation(rels, β, coords, n)
end

_check_tuple(m::ProductRelation, x, what) = (x isa Tuple && length(x) == length(m.relations)) ||
    throw(ArgumentError("a ProductRelation of $(length(m.relations)) relations needs `$what` as a tuple with one entry per relation"))

function _check_args(m::ProductRelation, z, ps, st)
    _check_tuple(m, ps, "ps"); _check_tuple(m, st, "st")
    length(z) == m.dim || throw(DimensionMismatch("the product's state has $(m.dim) coordinates, got $(length(z))"))
    return nothing
end

function prior_field(m::ProductRelation, z, ps, st)
    _check_args(m, z, ps, st)
    g = zero(float.(z))
    sts = map(eachindex(m.relations)) do i
        c = m.coords[i]
        gi, sti = prior_field(m.relations[i], z[c], ps[i], st[i])
        g[c] .+= m.weights[i] .* gi
        sti
    end
    return (g, Tuple(sts))
end

# the block on `coords` of the joint: each relation contributes the block on the coordinates it
# sees among them, at their positions
function prior_jacobian(m::ProductRelation, z, ps, st; coords = Colon())
    _check_args(m, z, ps, st)
    sel = _coords(coords, length(z))
    pos = zeros(Int, length(z)); pos[sel] .= eachindex(sel)
    J = zeros(eltype(float(z)), length(sel), length(sel))
    for i in eachindex(m.relations)
        c = m.coords[i]
        loc = findall(j -> pos[j] > 0, c)                # the relation's own indices that are selected
        isempty(loc) && continue
        p = pos[c[loc]]
        J[p, p] .+= m.weights[i] .* prior_jacobian(m.relations[i], z[c], ps[i], st[i]; coords = loc)
    end
    return J
end

function prior_vjp(m::ProductRelation, z, w, ps, st)
    _check_args(m, z, ps, st)
    v = zero(float.(z))
    for i in eachindex(m.relations)
        c = m.coords[i]
        v[c] .+= m.weights[i] .* prior_vjp(m.relations[i], z[c], w[c], ps[i], st[i])
    end
    return v
end

function prior_jvp(m::ProductRelation, z, u, ps, st)
    _check_args(m, z, ps, st)
    v = zero(float.(z))
    for i in eachindex(m.relations)
        c = m.coords[i]
        v[c] .+= m.weights[i] .* prior_jvp(m.relations[i], z[c], u[c], ps[i], st[i])
    end
    return v
end

_params_cotangent(m::ProductRelation, z, λ, ps, st) =
    Tuple(_params_cotangent(m.relations[i], z[m.coords[i]], m.weights[i] .* λ[m.coords[i]], ps[i], st[i])
          for i in eachindex(m.relations))

# a product of energy-parametrised relations has the weighted sum of their energies
_has_energy(m::ProductRelation) = all(_has_energy, m.relations)

function implicit_energy(m::ProductRelation, z, ps, st)
    _has_energy(m) || throw(ArgumentError("implicit_energy needs every relation of the product to be energy-parametrised"))
    _check_args(m, z, ps, st)
    U = zero(float(eltype(z)))
    sts = map(eachindex(m.relations)) do i
        Ui, sti = implicit_energy(m.relations[i], z[m.coords[i]], ps[i], st[i])
        U += m.weights[i] * Ui
        sti
    end
    return (U, Tuple(sts))
end

# --- sampling: the summed noise prediction ε_Π = Σ β_i ε_i, i.e. the summed score ---------------

function _schedule(m::ProductRelation)
    ss = map(_schedule, m.relations)
    all(==(first(ss)), ss) || throw(ArgumentError("sampling a product needs one noise schedule for all relations"))
    return first(ss)
end

function _eps_columns(m::ProductRelation, X, t, ps, st)
    _check_tuple(m, ps, "ps"); _check_tuple(m, st, "st")
    E = zero(float.(X))
    sts = map(eachindex(m.relations)) do i
        c = m.coords[i]
        Ei, sti = _eps_columns(m.relations[i], X[c, :], t, ps[i], st[i])
        E[c, :] .+= m.weights[i] .* Ei
        sti
    end
    return (E, Tuple(sts))
end

function _logdensity_columns(m::ProductRelation, X, t, ps, st)
    L = zeros(eltype(float(X)), size(X, 2))
    sts = map(eachindex(m.relations)) do i
        Li, sti = _logdensity_columns(m.relations[i], X[m.coords[i], :], t, ps[i], st[i])
        L .+= m.weights[i] .* Li
        sti
    end
    return (L, Tuple(sts))
end
