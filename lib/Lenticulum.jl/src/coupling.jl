# ---------------------------------------------------------------------------
# Coupling strengths of the factors, as default weights for `spanning_tree_schedule` (I25).
#
# A linear-Gaussian factor contributes the precision J_f = Cᵀ Q⁻¹ C to the joint over its channels
# (C the stacked coefficients, Q the noise). Its weight is the strongest coupling it adds between two
# of its variables, measured in the variables' own precision: max ‖D_a^{-1/2} J_ab D_b^{-1/2}‖₂, where
# D_v sums the diagonal blocks every factor puts on variable v. See `coupling.md`.
# ---------------------------------------------------------------------------

"""
    coupling_precision(f, ps) -> (blocks::NamedTuple, J::Matrix) | :hard | nothing

The precision a factor adds to the joint of its continuous channels: `J` stacked in the order of
`blocks`, a `NamedTuple` of channel names to dimensions. `:hard` for an exact relation (its precision
is infinite along the directions it fixes), `nothing` when the factor has no fixed linear-Gaussian
form (a nonlinear or non-Gaussian factor; its precision depends on where it is linearised).

Methods: `GaussianFactor` (``[-A\\; I]^\\top Q^{-1} [-A\\; I]``), `GaussianPrior` (``Σ^{-1}``),
`LinearConstraintFactor` and `SwitchingConstraintFactor` (``A^\\top Q^{-1} A``; the switch only shifts
the constant), `RegimeFactor` (``\\sum_k w_k A_k^\\top Q_k^{-1} A_k`` with the regime weights
``w = \\operatorname{softmax}(\\text{logw})``), `HardConstraintFactor` (`:hard`).
"""
coupling_precision(f, ps) = nothing

function _stacked_precision(C::NamedTuple, dims::NamedTuple, P)
    names = keys(dims)
    Cs = reduce(hcat, (Matrix{Float64}(C[n]) for n in names))
    return (dims, Matrix(Symmetric(Cs' * P * Cs)))
end
_precision(Q) = Matrix(inv(Symmetric(Matrix{Float64}(Q))))

coupling_precision(f::GaussianFactor, ps) =
    _stacked_precision(_gaussian_blocks(f, ps), _gaussian_dims(f), _precision(noise_covariance(f, ps)))
coupling_precision(f::GaussianPrior, ps) =
    (NamedTuple{(f.channel,)}((length(f.μ),)), _precision(noise_covariance(f, ps)))
coupling_precision(f::LinearConstraintFactor, ps) =
    _stacked_precision(_coeffs(f, ps), f.dims, _precision(f.Q))
coupling_precision(f::SwitchingConstraintFactor, ps) =
    _stacked_precision(_coeffs(f, ps), f.dims, _precision(f.Q))
coupling_precision(::HardConstraintFactor, ps) = :hard
function coupling_precision(f::RegimeFactor, ps)
    A = _coeffs(f, ps); Q = regime_covariances(f, ps); w = exp.(_logweights(ps))
    J = sum(w[k] * last(_stacked_precision(map(a -> a[:, :, k], A), f.dims, _precision(Q[:, :, k])))
            for k in 1:f.K)
    return (f.dims, J)
end

# J^{-1/2} on the range of a positive semidefinite D (pseudo-inverse square root).
function _inv_sqrt_psd(D)
    λ, V = eigen(Symmetric(D))
    tol = LenticulumCore._eig_tol(λ)
    return V * Diagonal([l > tol ? inv(sqrt(l)) : 0.0 for l in λ]) * V'
end

_block_ranges(dims::NamedTuple) =
    NamedTuple{keys(dims)}(Tuple(range(s - d + 1; length = d) for (s, d) in zip(cumsum(collect(values(dims))), values(dims))))

"""
    coupling_weights(g, ps; normalize = true, default = 0.0, hard = nothing) -> Dict{Symbol,Float64}

A weight per factor of `g`, keyed by factor name, for
`spanning_tree_schedule(g; weights = coupling_weights(g, ps))`: the strongest coupling the factor adds
between two different variables, from [`coupling_precision`](@ref) with the parameters `ps` (the
graph's parameter tree, as given to `infer!`).

With `normalize = true` the coupling between channels `a` and `b` (on variables ``u \\ne v``) is
``\\lVert D_u^{-1/2} J_{ab} D_v^{-1/2} \\rVert_2``, where ``D_v`` sums the diagonal blocks of every
factor with a linear-Gaussian form on ``v``; it lies in ``[0, 1]`` and does not change when a variable
is rescaled. With `normalize = false` it is ``\\lVert J_{ab} \\rVert_2``, in the variables' units: for
`LinearConstraintFactor((a = 1, b = 1), 1; noise = q)` with ``A = (1, -1)`` that is ``1/q``, the weight
measured in `bench/bp_trees.jl`.

A factor with fewer than two connected variables gets weight 0 (its edges belong to every spanning
forest anyway). A `HardConstraintFactor` gets `hard`, by default 1 (normalised) or the largest
finite weight in the graph plus one (not normalised); it adds nothing to ``D``. A factor without a
linear-Gaussian form (`coupling_precision` returns `nothing`: `NonlinearFactor`, `LikelihoodFactor`,
factors of other packages) gets `default` and adds nothing to ``D``.

Parameters are read as `Float64`; this is a choice of schedule, not differentiated.
"""
function coupling_weights(g::Mycelium.FactorGraph, ps; normalize::Bool = true, default::Real = 0.0,
                          hard::Union{Nothing,Real} = nothing)
    isfinite(default) || throw(ArgumentError("default must be finite; got $default"))
    hard === nothing || isfinite(hard) || throw(ArgumentError("hard must be finite; got $hard"))
    nf = length(g.factors)
    forms = Vector{Any}(undef, nf)
    for (fi, node) in enumerate(g.factors)
        psf = Mycelium._factor_parameters(node, Mycelium._subtree(ps, node.params))
        forms[fi] = coupling_precision(node.factor, psf)
    end
    # the variable of each connected channel, per factor
    chans = [Dict(g.edges[ei].channel => g.edges[ei].variable for ei in g.edges_of_factor[fi]) for fi in 1:nf]
    D = Dict{Int,Matrix{Float64}}()
    if normalize
        for fi in 1:nf
            form = forms[fi]
            form isa Tuple || continue
            dims, J = form; r = _block_ranges(dims)
            for (c, v) in chans[fi]
                haskey(r, c) || continue
                blk = J[r[c], r[c]]
                D[v] = haskey(D, v) ? D[v] + blk : copy(blk)
            end
        end
    end
    S = Dict(v => _inv_sqrt_psd(Dv) for (v, Dv) in D)
    w = Vector{Float64}(undef, nf)
    ishard = falses(nf)
    for fi in 1:nf
        form = forms[fi]
        if form === :hard
            ishard[fi] = true; w[fi] = 0.0; continue
        elseif form === nothing
            w[fi] = float(default); continue
        end
        dims, J = form; r = _block_ranges(dims)
        cs = [(c, v) for (c, v) in chans[fi] if haskey(r, c)]
        best = 0.0
        for (i, (a, u)) in enumerate(cs), (b, v) in cs[i+1:end]
            u == v && continue
            Jab = J[r[a], r[b]]
            Jab = normalize ? S[u] * Jab * S[v] : Jab
            best = max(best, LinearAlgebra.opnorm(Jab))
        end
        w[fi] = best
    end
    if any(ishard)
        h = hard !== nothing ? float(hard) : normalize ? 1.0 :
            maximum((w[fi] for fi in 1:nf if !ishard[fi]); init = 0.0) + 1.0
        w[ishard] .= h
    end
    all(isfinite, w) || throw(ArgumentError(
        "a coupling weight is not finite (factors $(join((g.factors[fi].name for fi in 1:nf if !isfinite(w[fi])), ", ")))"))
    return Dict(g.factors[fi].name => w[fi] for fi in 1:nf)
end
