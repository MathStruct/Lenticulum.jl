# ---------------------------------------------------------------------------
# Cluster nodes over a plate: the batched messages and free energies (L-SPEC-009 §8).
#
# Tied cluster nodes of one plate (`cluster(g, plate, name, factors)`) differ only in what their
# replica's messages bring: points (data) and the means of Gaussian inputs. When every replica has
# the same factors and its inputs the same shape (which channels are points, the input precisions,
# the discrete inputs), the joints share their precision J, the assignments that survive and the
# factors' forms; replica i's joint is replica 1's with H shifted by one vector and the constants
# by one vector over the assignments. The messages then share their Cholesky factors, and the
# free energy of all replicas is evaluated as a few stacked matrix products, which is what the
# parameter gradient differentiates. Otherwise every method falls back to replica by replica.
#
# See `cluster.md`.
# ---------------------------------------------------------------------------

# Fieldwise equality of two factors (the factors of two replicas are separate objects).
_same_struct(x, y) = x === y || (typeof(x) === typeof(y) && all(f -> isequal(getfield(x, f), getfield(y, f)), fieldnames(typeof(x))))
_same_cluster(a::ClusterFactor, b::ClusterFactor) = a === b || (a.wiring == b.wiring && a.keys == b.keys &&
    a.variables == b.variables && a.dims == b.dims && a.discrete == b.discrete && a.boundary == b.boundary &&
    all(_same_struct(x, y) for (x, y) in zip(a.factors, b.factors)))

# What of one replica's incoming messages fixes its joint's structure (`_cluster_joint`).
function _input_shape(c::ClusterFactor, msgs, skip)
    return map(c.boundary) do v
        b = _incoming(msgs, v, skip)
        b isa LenticulumCore.DiracBelief && return (:point,)
        if v in c.discrete
            return b isa LenticulumCore.CategoricalBelief ? (:cat, b.logp) : (:none,)
        end
        b === nothing && return (:none,)
        bb = _cluster_input(b)
        return bb isa GaussianBelief ? (:gauss, bb.Λ) : (:none,)
    end
end
function _cluster_batchable(cs, msgs, skip)
    length(cs) >= 2 || return false
    c1, k1 = cs[1], _input_shape(cs[1], msgs[1], skip)
    return all(i -> _same_cluster(cs[i], c1) && isequal(_input_shape(cs[i], msgs[i], skip), k1), 2:length(cs))
end

# The per-factor pieces of a joint that do not depend on the replica's data.
function _joint_plan(c::ClusterFactor, jt::_ClusterJoint, ps)
    n = size(jt.J, 1)
    plan = Any[]
    for (i, f) in enumerate(c.factors)
        F = cluster_form(f, _subtree(ps, c.keys[i]))
        m = size(F.C, 1)
        m == 0 && continue
        w = Dict(c.wiring[i])
        G = zeros(m, n)
        for (ch, Ach) in pairs(F.A)
            v = w[ch]
            haskey(jt.offset, v) && (G[:, jt.offset[v]] .+= Ach)
        end
        CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(F.Q)))
        push!(plan, (F, w, G, CQ, F.C[:, _switch_columns(c, F, w, jt.assignments)]))
    end
    return plan
end

# A replica's data terms: H = H_data + h 1ᵀ + (shared), c = c_data + δ + (shared), and its points.
function _data_terms(c::ClusterFactor, jt::_ClusterJoint, plan, msgs, skip)
    points = Dict{Symbol,Vector{Float64}}()
    for v in c.continuous
        b = _incoming(msgs, v, skip)
        b isa LenticulumCore.DiracBelief && (points[v] = collect(float.(b.value isa Real ? [b.value] : b.value)))
    end
    h = zeros(size(jt.J, 1)); δ = zeros(length(jt.assignments))
    for v in jt.free
        b = _cluster_input(_incoming(msgs, v, skip))
        b isa GaussianBelief && (h[jt.offset[v]] .+= b.η)
    end
    for (F, w, G, CQ, Ck) in plan
        s = zeros(size(G, 1))
        for (ch, Ach) in pairs(F.A)
            haskey(points, w[ch]) && (s .+= Ach * points[w[ch]])
        end
        any(!iszero, s) || continue
        Qs = CQ \ s
        h .-= G' * Qs
        δ .+= vec(Qs' * Ck) .- LinearAlgebra.dot(s, Qs) / 2
    end
    return h, δ, points
end

# Every replica's joint, from replica 1's and the data terms.
function _batch_joints(cs, msgs, ps, skip)
    jt1 = _cluster_joint(cs[1], msgs[1], ps, skip)
    plan = _joint_plan(cs[1], jt1, ps)
    h1, δ1, _ = _data_terms(cs[1], jt1, plan, msgs[1], skip)
    return map(eachindex(cs)) do i
        i == 1 && return jt1
        h, δ, pts = _data_terms(cs[i], jt1, plan, msgs[i], skip)
        _ClusterJoint(jt1.free, jt1.offset, pts, jt1.assignments, jt1.J, jt1.H .+ (h .- h1), jt1.c .+ (δ .- δ1))
    end
end

function factor_messages(cs::AbstractVector{<:ClusterFactor}, target::Symbol, polarity, inputs::AbstractVector,
                         priors::AbstractVector, ps, st)
    if !_cluster_batchable(cs, inputs, target)
        out = Vector{Any}(undef, length(cs))
        for i in eachindex(cs)
            out[i], st = factor_message(cs[i], target, polarity, inputs[i], priors[i], ps, st)
        end
        return out, st
    end
    target in cs[1].boundary || throw(ArgumentError("channel :$target is not a channel of this cluster (has $(cs[1].boundary))"))
    chol = Dict{Symbol,Any}()
    return Any[_joint_message(cs[i], jt, target, chol) for (i, jt) in enumerate(_batch_joints(cs, inputs, ps, target))], st
end

"""
    FrozenClusterBeliefs

The beliefs of tied cluster nodes over a plate whose joints share their structure, held fixed as
one object: `q` (assignments × replicas), the means `M` (free coordinates × (assignments ·
replicas), replica-major), one covariance `Σ` and conditional entropy `H`, and the points of each
substituted variable as columns. Returned by `factor_beliefs` for a vector of `ClusterFactor`s;
`belief_free_energies` evaluates all replicas in stacked products.
"""
struct FrozenClusterBeliefs
    q::Matrix{Float64}
    M::Matrix{Float64}
    Σ::Matrix{Float64}
    H::Float64
    free::Vector{Symbol}
    offset::Dict{Symbol,UnitRange{Int}}
    points::Dict{Symbol,Matrix{Float64}}
    assignments::Vector{Vector{Int}}
end

function factor_beliefs(cs::AbstractVector{<:ClusterFactor}, msgs::NamedTuple, ps, st)
    cols = [_column(msgs, i) for i in eachindex(cs)]
    _cluster_batchable(cs, cols, nothing) || return [factor_belief(cs[i], cols[i], ps, st) for i in eachindex(cs)]
    chol = Dict{Symbol,Any}()
    bs = [_joint_belief(jt, chol) for jt in _batch_joints(cs, cols, ps, nothing)]
    b1 = bs[1]
    pts = Dict(v => reduce(hcat, [b.points[v] for b in bs]) for v in keys(b1.points))
    return FrozenClusterBeliefs(reduce(hcat, [b.q for b in bs]), reduce(hcat, [b.M for b in bs]), b1.Σ, b1.H,
                                b1.free, b1.offset, pts, b1.assignments)
end

# The structure of factor i's term over the batch (outside AD, `@non_differentiable` in the
# ChainRulesCore extension): its outcome columns, the channels feeding each free variable, and
# the channels holding points.
function _batch_layout(c::ClusterFactor, i::Int, F::ClusterForm, B::FrozenClusterBeliefs)
    w = Dict(c.wiring[i])
    kcol = _switch_columns(c, F, w, B.assignments)
    feeds = [[ch for ch in keys(F.A) if w[ch] === v] for v in B.free]
    pts = [(ch, w[ch]) for ch in keys(F.A) if haskey(B.points, w[ch])]
    return kcol, feeds, pts
end

function belief_free_energies(cs::AbstractVector{<:ClusterFactor}, B::FrozenClusterBeliefs, ps, st)
    c = cs[1]
    nA, N = size(B.q)
    n = size(B.Σ, 1)
    terms = map(eachindex(c.factors)) do i
        F = cluster_form(c.factors[i], _subtree(ps, c.keys[i]))
        kcol, feeds, pts = _batch_layout(c, i, F, B)
        cx = -vec(sum(B.q .* _finite_logw(F.logw[kcol]); dims = 1))
        m = size(F.C, 1)
        m == 0 && return (zeros(N), cx)
        R = repeat(-F.C[:, kcol], 1, N)
        for (ch, v) in pts
            R = R .+ repeat(getfield(F.A, ch) * B.points[v]; inner = (1, nA))
        end
        Cr = zeros(m, m)
        if n > 0
            G = reduce(hcat, [isempty(chs) ? zeros(m, length(B.offset[v])) : reduce(+, [getfield(F.A, ch) for ch in chs])
                              for (v, chs) in zip(B.free, feeds)])
            R = R .+ G * B.M
            Cr = G * B.Σ * G'
        end
        CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(F.Q)))
        quad = reshape(vec(sum(R .* (CQ \ R); dims = 1)), nA, N)
        return ((vec(sum(B.q .* quad; dims = 1)) .+ LinearAlgebra.tr(CQ \ Cr)) ./ 2, cx .+ (logdet(CQ) + m * log(2π)) / 2)
    end
    fit, complexity = sum(first, terms), sum(last, terms)
    Hq = vec(-sum(ifelse.(B.q .> 0, B.q .* log.(max.(B.q, floatmin(Float64))), 0.0); dims = 1))
    return [LenticulumCore.GradedEnergy((fit = fit[r], complexity = complexity[r], negentropy = -Hq[r] - B.H)) for r in 1:N], st
end

function local_free_energies(cs::AbstractVector{<:ClusterFactor}, msgs::NamedTuple, ps, st)
    B = factor_beliefs(cs, msgs, ps, st)
    B isa FrozenClusterBeliefs && return belief_free_energies(cs, B, ps, st)
    return [first(belief_free_energy(cs[i], B[i], ps, st)) for i in eachindex(cs)], st
end
