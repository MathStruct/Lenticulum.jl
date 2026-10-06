# ---------------------------------------------------------------------------
# Several learned relations on the same variables, pooled: the product of their densities.
#
# Each implicit learner's field g_i is (for an ideal model) the gradient of a smoothed negative
# log-density, so the weighted sum Σ β_i g_i is the field of the product Π p_i^{β_i}: its stable
# roots are where all relations hold at once, e.g. the intersection points of a learned circle
# and a learned line. Every query function works on the product unchanged.
#
# See `product.md` and `Composing Diffusion Factors` in the vault.
# ---------------------------------------------------------------------------

"""
    ProductRelation(relations...; weights = ones)

Several implicit relations on the **same** joint space, pooled: the field is the weighted sum
``\\sum_i \\beta_i\\,g_i`` of the relations' fields, i.e. the field of the product of their densities
``\\prod_i p_i^{\\beta_i}``. Its stable roots are the configurations that satisfy all relations
together, e.g. the intersection points of a learned circle and a learned line. The weights
``\\beta_i`` set how much each relation is trusted (a tempering exponent).

Parameters and states are **tuples**, one entry per relation, in order: `ps = (ps₁, ps₂)`. Every
query function accepts a product: [`implicit_infer`](@ref), [`implicit_roots`](@ref),
[`implicit_laplace`](@ref), and [`implicit_pullback`](@ref) (which returns a tuple of parameter
cotangents). Products may be nested.

Pooling smoothed densities is exact only at zero noise: at the field's noise levels the
product of the smoothed densities is not the smoothed product, so intersections are found
with the same kind of smoothing bias as single relations (`product.md` §3).
"""
struct ProductRelation{M<:Tuple,W<:AbstractVector} <: AbstractImplicitRelation
    relations::M
    weights::W
end
function ProductRelation(relations::AbstractImplicitRelation...; weights = ones(length(relations)))
    isempty(relations) && throw(ArgumentError("a product needs at least one relation"))
    length(weights) == length(relations) || throw(ArgumentError("need one weight per relation"))
    all(>(0), weights) || throw(ArgumentError("weights must be positive"))
    return ProductRelation(relations, float.(collect(weights)))
end

_check_tuple(m::ProductRelation, x, what) = (x isa Tuple && length(x) == length(m.relations)) ||
    throw(ArgumentError("a ProductRelation of $(length(m.relations)) relations needs `$what` as a tuple with one entry per relation"))

function prior_field(m::ProductRelation, z, ps, st)
    _check_tuple(m, ps, "ps"); _check_tuple(m, st, "st")
    g = zero(float.(z))
    sts = map(eachindex(m.relations)) do i
        gi, sti = prior_field(m.relations[i], z, ps[i], st[i])
        g = g .+ m.weights[i] .* gi
        sti
    end
    return (g, Tuple(sts))
end

function prior_jacobian(m::ProductRelation, z, ps, st)
    _check_tuple(m, ps, "ps"); _check_tuple(m, st, "st")
    return sum(m.weights[i] .* prior_jacobian(m.relations[i], z, ps[i], st[i]) for i in eachindex(m.relations))
end

_params_cotangent(m::ProductRelation, z, λ, ps, st) =
    Tuple(_params_cotangent(m.relations[i], z, m.weights[i] .* λ, ps[i], st[i]) for i in eachindex(m.relations))

# a product of energy-parametrised relations has the weighted sum of their energies
_has_energy(m::ProductRelation) = all(_has_energy, m.relations)

function implicit_energy(m::ProductRelation, z, ps, st)
    _has_energy(m) || throw(ArgumentError("implicit_energy needs every relation of the product to be energy-parametrised"))
    U = zero(float(eltype(z)))
    sts = map(eachindex(m.relations)) do i
        Ui, sti = implicit_energy(m.relations[i], z, ps[i], st[i])
        U += m.weights[i] * Ui
        sti
    end
    return (U, Tuple(sts))
end
