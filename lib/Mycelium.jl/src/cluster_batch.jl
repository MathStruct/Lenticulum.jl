# ---------------------------------------------------------------------------
# Cluster nodes over a plate: the batched messages and free energies (L-SPEC-009 §8).
#
# Tied cluster nodes of one plate (`cluster(g, plate, name, factors)`) differ only in what their
# replica's messages bring: points (data) and the means of Gaussian inputs. When every replica has
# the same factors and its inputs the same shape (which channels are points, the input precisions,
# the discrete inputs), the joints share their precisions J_κ (one per precision class), the
# assignments that survive and the factors' forms; replica i's joint is replica 1's with H shifted
# by one column per assignment (one vector for all of them unless a factor's coefficients depend
# on its outcome) and the constants by one vector over the assignments. All replicas' joints are
# then one stacked joint (`_batch_joint`), whose messages and beliefs take one set of solves for
# every replica, and the free energy of all replicas is evaluated as a few stacked matrix
# products, which is what the parameter gradient differentiates. Otherwise every method falls back to replica by replica.
#
# See `cluster.md`.
# ---------------------------------------------------------------------------

# Fieldwise equality of two factors (the factors of two replicas are separate objects), unrolled
# over the fields so that each comparison is typed.
_same_struct(x, y) = x === y || (typeof(x) === typeof(y) && _same_fields(x, y))
@generated function _same_fields(x::T, y::T) where {T}
    fieldcount(T) == 0 && return :(true)
    return foldr((a, b) -> :($a && $b), [:(isequal(getfield(x, $i), getfield(y, $i))) for i in 1:fieldcount(T)])
end
_same_cluster(a::ClusterFactor, b::ClusterFactor) = a === b || (a.wiring == b.wiring && a.keys == b.keys &&
    a.variables == b.variables && a.dims == b.dims && a.discrete == b.discrete && a.boundary == b.boundary &&
    all(_same_struct(x, y) for (x, y) in zip(a.factors, b.factors)))

# A plate row of factors that have only a `cluster_form` (`RegimeFactor`) as one-factor clusters:
# one cluster shared by all replicas when their factors are equal fieldwise (the usual case,
# replicas of one factor), which `_same_cluster` then accepts at once; one per replica otherwise.
function _as_clusters(fs::AbstractVector)
    isempty(fs) && return ClusterFactor[]
    f1 = first(fs)
    all(f -> _same_struct(f, f1), fs) && return fill(as_cluster(f1), length(fs))
    return ClusterFactor[as_cluster(f) for f in fs]
end

# What of one replica's incoming messages fixes its joint's structure (`_cluster_joint`).
function _input_shape(c::ClusterFactor, msgs, skip)
    return map(c.boundary) do v
        b = _incoming(msgs, v, skip)
        # an observed discrete state fixes which assignments survive, so it is part of the shape
        # (a continuous point only shifts h and the constants, `_batch_joint`)
        b isa LenticulumCore.DiracBelief && return v in c.discrete ? (:state, b.value) : (:point,)
        if v in c.discrete
            # the labels too: `_cluster_joint` checks replica 1's only (LP-032)
            return b isa LenticulumCore.CategoricalBelief ? (:cat, b.logp, b.labels) : (:none,)
        end
        b === nothing && return (:none,)
        bb = _cluster_input(b)
        return bb isa GaussianBelief ? (:gauss, bb.Λ) : (:none,)
    end
end
# The stacked joint is built in `Float64`; parameters of another number type (dual numbers under
# ForwardDiff, P24) go replica by replica, whose joint is generic.
_plain(@nospecialize(ps)) = _numtype(ps) === Float64
function _cluster_batchable(cs, msgs, skip, @nospecialize(ps))
    length(cs) >= 2 && _plain(ps) || return false
    c1, k1 = cs[1], _input_shape(cs[1], msgs[1], skip)
    return all(i -> _same_cluster(cs[i], c1) && isequal(_input_shape(cs[i], msgs[i], skip), k1), 2:length(cs))
end

# The per-factor pieces of a joint that do not depend on the replica's data: per outcome group
# (the assignments reading one outcome's A and Q, all of them when these do not depend on the
# outcome; `k = 0` then) the coefficients on the free coordinates, Q's factorisation and the
# offset columns.
function _joint_plan(c::ClusterFactor, jt::_ClusterJoint, @nospecialize(ps))
    n = size(jt.H, 1)
    plan = Any[]
    for (i, f) in enumerate(c.factors)
        F = cluster_form(f, _subtree(ps, c.keys[i]))
        m = size(F.C, 1)
        m == 0 && continue
        w = Dict(c.wiring[i])
        kcol = _switch_columns(c, F, w, jt.assignments)
        groups = map(_varies(F) ? unique(kcol) : [0]) do k
            idx = k == 0 ? collect(eachindex(kcol)) : findall(==(k), kcol)
            G = zeros(m, n)
            for (ch, Ach) in pairs(F.A)
                v = w[ch]
                haskey(jt.offset, v) && (G[:, jt.offset[v]] .+= _at(Ach, k))
            end
            CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(_at(F.Q, k))))
            (k, idx, G, CQ, F.C[:, kcol[idx]])
        end
        push!(plan, (F, w, groups))
    end
    return plan
end

_point_vector(b) = collect(float.(b.value isa Real ? [b.value] : b.value))

# All replicas' joints as one stacked joint: replica 1's structure (free variables, assignments
# held once, precisions J_κ), and H (n × nA·N) and the constants (nA·N) replica-major, each
# replica's columns being replica 1's shifted by the difference of their data terms. The data
# terms of all replicas are computed together: the points of a channel are the columns of one
# matrix, and per factor and outcome group one solve with Q serves every replica. With
# per-outcome coefficients a point's shift A_k x differs per outcome, so the data term has one
# column per assignment. Returns the stacked joint, replica 1's precision classes (nA), the points
# of each substituted variable as columns, and nA.
function _batch_joint(cs, msgs, @nospecialize(ps), skip)
    c, N = cs[1], length(cs)
    jt1 = _cluster_joint(c, _batch_first(msgs), ps, skip)
    plan = _joint_plan(c, jt1, ps)
    n, nA = size(jt1.H)
    # which continuous channels hold points is the same in every replica (`_input_shape`)
    pts = Dict{Symbol,Matrix{Float64}}(v => _batch_points(msgs, v, skip) for v in keys(jt1.points))
    Hd = zeros(n, nA, N); δ = zeros(nA, N)
    for v in jt1.free
        _batch_eta!(Hd, jt1.offset[v], msgs, v, skip)
    end
    for (F, w, groups) in plan, (k, idx, G, CQ, Ck) in groups
        chs = [ch for ch in keys(F.A) if haskey(pts, w[ch])]
        isempty(chs) && continue
        S = reduce(+, [_at(getfield(F.A, ch), k) * pts[w[ch]] for ch in chs])     # m × N
        QS = CQ \ S
        GQ = G' * QS                                                             # n × N
        for a in idx
            @views Hd[:, a, :] .-= GQ
        end
        δ[idx, :] .+= Ck' * QS .- sum(S .* QS; dims = 1) ./ 2
    end
    H = Matrix{Float64}(undef, n, nA * N); cst = Vector{Float64}(undef, nA * N)
    for r in 1:N
        cols = ((r - 1) * nA + 1):(r * nA)
        @views H[:, cols] .= jt1.H .+ (Hd[:, :, r] .- Hd[:, :, 1])
        @views cst[cols] .= jt1.c .+ (δ[:, r] .- δ[:, 1])
    end
    jt = _ClusterJoint(jt1.free, jt1.offset, Dict{Symbol,Vector{Float64}}(), jt1.assignments, jt1.J,
                       repeat(jt1.class, N), H, cst)
    return jt, jt1.class, pts, nA
end

# The inputs of `_batch_joint`: a vector of per-replica NamedTuples, or (a plate sweep) one
# NamedTuple of blocks (`blocks.jl`) that `_block_batchable` accepted.
_batch_first(msgs::AbstractVector) = msgs[1]
_batch_first(msgs::NamedTuple) = _column(msgs, 1)
_batch_points(msgs::AbstractVector, v, skip) = reduce(hcat, [_point_vector(_incoming(m, v, skip)) for m in msgs])
_batch_points(msgs::NamedTuple, v, skip) = Matrix{Float64}(float.(_incoming(msgs, v, skip).X))
function _batch_eta!(Hd, off, msgs::AbstractVector, v, skip)
    for r in eachindex(msgs)
        b = _cluster_input(_incoming(msgs[r], v, skip))
        b isa GaussianBelief && (Hd[off, :, r] .+= b.η)
    end
end
function _batch_eta!(Hd, off, msgs::NamedTuple, v, skip)
    b = _incoming(msgs, v, skip)
    b isa GaussianBlock && (@views Hd[off, :, :] .+= reshape(b.η, length(off), 1, size(b.η, 2)))
    return nothing
end

# Whether blocks of inputs give every replica the shape (`_input_shape`) of replica 1, read off the
# blocks: points (on a discrete channel, the same state in every column), Gaussians of one
# precision, categorical inputs to a discrete channel equal in every column, or nothing. Mixture
# and vector blocks go column by column (`_cluster_batchable` then decides).
function _block_batchable(cs, msgs::NamedTuple, skip)
    length(cs) >= 2 || return false
    c1 = cs[1]
    all(i -> _same_cluster(cs[i], c1), 2:length(cs)) || return false
    return all(v -> _uniform_shape(_incoming(msgs, v, skip), v in c1.discrete), c1.boundary)
end
_uniform_shape(::Union{Nothing,TrivialBlock}, discrete) = true
_uniform_shape(b::PointBlock, discrete) = !discrete || _equal_columns(b.X)
_uniform_shape(b::CategoricalBlock, discrete) = !discrete || _equal_columns(b.logp)
_uniform_shape(b::GaussianBlock, discrete) = discrete || _shared(b) || all(L -> isequal(L, b.Λ[1]), b.Λ)
_uniform_shape(b, discrete) = false
_equal_columns(X) = all(i -> isequal(view(X, :, i), view(X, :, 1)), axes(X, 2))

# Block inputs (a plate sweep): the stacked joint read from the blocks, and the messages returned
# as one block (`_discrete_block`, `_continuous_block`) without a belief per replica. Inputs the
# blocks cannot vouch for go column by column, as the default for blocks does.
function factor_messages(cs::AbstractVector{<:ClusterFactor}, target::Symbol, polarity, inputs::NamedTuple,
                         priors::AbstractVector, @nospecialize(ps), @nospecialize(st))
    if !(_plain(ps) && _block_batchable(cs, inputs, target))
        out, st = factor_messages(cs, target, polarity, [_column(inputs, i) for i in eachindex(cs)], priors, ps, st)
        return belief_block(out), st
    end
    c, N = cs[1], length(cs)
    target in c.boundary || throw(ArgumentError("channel :$target is not a channel of this cluster (has $(c.boundary))"))
    chol = Dict{Any,Any}()
    jt, _, _, nA = _batch_joint(cs, inputs, ps, target)
    target in c.discrete && return _discrete_block(target, _discrete_parts(c, jt, target, chol)..., nA, N), st
    return _continuous_block(_continuous_parts(jt, target, chol)..., jt.class, nA, N), st
end

# `_discrete_message` of every replica, as the columns of one block.
function _discrete_block(target, l, sel, nA, N)
    L = reshape(l, nA, N)
    logp = Matrix{Float64}(undef, length(sel), N)
    lk = Vector{Float64}(undef, length(sel))
    for r in 1:N
        for (k, s) in enumerate(sel)
            lk[k] = _lse(L[s, r])
        end
        all(==(-Inf), lk) && throw(ArgumentError("cluster: every state of :$target has zero probability"))
        logp[:, r] .= lk .- _lse(lk)
    end
    return CategoricalBlock(logp, nothing)
end

# `_continuous_message` of every replica as one block. When no assignment has weight zero, no
# replica has two equal components (which `_continuous_message` would merge) and the target is
# not trivial, the assignments are the components: of a `MixtureBlock` when all classes have one
# precision (a `GaussianBlock` for one assignment), else of a `ComponentMixtureBlock`, component
# a holding its class's precision (I21). Otherwise each replica's message is built and stacked.
function _continuous_block(E, l, Λs, proper, class, nA, N)
    ids = _precision_ids(Λs)
    d = size(E, 1)
    if !(length(Λs) == 1 && all(iszero, Λs[1])) && all(>(-Inf), l) && !_merging(E, nA, N)
        if nA == 1                                         # one component: `_continuous_message` returns it alone
            κ = ids[class[1]]
            return GaussianBlock(copy(E), Λs[κ], fill(!proper[κ], N))
        end
        W = reshape(l, nA, N)
        logw = similar(W)
        for r in 1:N
            w = W[:, r]
            logw[:, r] .= w .- _lse(w)
        end
        all(==(1), ids) && return MixtureBlock(reshape(copy(E), d, nA, N), Λs[1], logw, !proper[1])
        κs = [ids[class[a]] for a in 1:nA]                # replica 1's assignments; every replica's are these
        return ComponentMixtureBlock(reshape(copy(E), d, nA, N), [Λs[κ] for κ in κs], logw, [!proper[κ] for κ in κs])
    end
    return belief_block(Any[_continuous_message(E, l, Λs, proper, class, ((r - 1) * nA + 1):(r * nA), ids) for r in 1:N])
end
# Whether some replica has two assignments with equal columns of E.
function _merging(E, nA, N)
    for r in 1:N, a in 1:nA, b in (a + 1):nA
        isequal(view(E, :, (r - 1) * nA + a), view(E, :, (r - 1) * nA + b)) && return true
    end
    return false
end

function factor_messages(cs::AbstractVector{<:ClusterFactor}, target::Symbol, polarity, inputs::AbstractVector,
                         priors::AbstractVector, @nospecialize(ps), @nospecialize(st))
    if !_cluster_batchable(cs, inputs, target, ps)
        out = Vector{Any}(undef, length(cs))
        for i in eachindex(cs)
            out[i], st = factor_message(cs[i], target, polarity, inputs[i], priors[i], ps, st)
        end
        return out, st
    end
    c = cs[1]
    target in c.boundary || throw(ArgumentError("channel :$target is not a channel of this cluster (has $(c.boundary))"))
    chol = Dict{Any,Any}()
    jt, _, _, nA = _batch_joint(cs, inputs, ps, target)
    cols(r) = ((r - 1) * nA + 1):(r * nA)
    if target in c.discrete
        l, sel = _discrete_parts(c, jt, target, chol)
        return Any[_discrete_message(c, target, l, sel, cols(r)) for r in eachindex(cs)], st
    end
    E, l, Λs, proper = _continuous_parts(jt, target, chol)
    return Any[_continuous_message(E, l, Λs, proper, jt.class, cols(r)) for r in eachindex(cs)], st
end

"""
    FrozenClusterBeliefs

The beliefs of tied cluster nodes over a plate whose joints share their structure, held fixed as
one object: `q` (assignments × replicas), the means `M` (free coordinates × (assignments ·
replicas), replica-major), the covariance `Σ[κ]` and conditional entropy `H[κ]` of each precision
class (`class[a]` for assignment a; one class unless a factor's coefficients or noise depend on
its outcome), and the points of each substituted variable as columns. Returned by
`factor_beliefs` for a vector of `ClusterFactor`s; `belief_free_energies` evaluates all replicas
in stacked products.
"""
struct FrozenClusterBeliefs
    q::Matrix{Float64}
    M::Matrix{Float64}
    Σ::Vector{Matrix{Float64}}
    H::Vector{Float64}
    class::Vector{Int}
    free::Vector{Symbol}
    offset::Dict{Symbol,UnitRange{Int}}
    points::Dict{Symbol,Matrix{Float64}}
    assignments::Vector{Vector{Int}}
end

function factor_beliefs(cs::AbstractVector{<:ClusterFactor}, msgs::NamedTuple, @nospecialize(ps), @nospecialize(st))
    if _plain(ps) && _block_batchable(cs, msgs, nothing)    # the blocks read directly, without columns
        jt, class, pts, nA = _batch_joint(cs, msgs, ps, nothing)
    else
        cols = [_column(msgs, i) for i in eachindex(cs)]
        _cluster_batchable(cs, cols, nothing, ps) || return [factor_belief(cs[i], cols[i], ps, st) for i in eachindex(cs)]
        jt, class, pts, nA = _batch_joint(cs, cols, ps, nothing)
    end
    q, M, Σ, H = _belief_parts(jt, Dict{Any,Any}(), nA)
    return FrozenClusterBeliefs(q, M, Σ, H, class, jt.free, jt.offset, pts, jt.assignments)
end

# The structure of factor i's term over the batch (outside AD, `@non_differentiable` in the
# ChainRulesCore extension): its outcome columns; per outcome group (the assignments reading one
# outcome's A and Q; one group, `k = 0`, when these do not depend on the outcome) its
# assignments, their columns of `M` in every replica (replica-major) and their precision classes;
# the channels feeding each free variable; the channels holding points.
function _batch_layout(c::ClusterFactor, i::Int, F::ClusterForm, B::FrozenClusterBeliefs)
    w = Dict(c.wiring[i])
    kcol = _switch_columns(c, F, w, B.assignments)
    nA, N = size(B.q)
    groups = map(_varies(F) ? unique(kcol) : [0]) do k
        idx = k == 0 ? collect(1:nA) : findall(==(k), kcol)
        (k, idx, vec([(r - 1) * nA + a for a in idx, r in 1:N]), B.class[idx])
    end
    feeds = [[ch for ch in keys(F.A) if w[ch] === v] for v in B.free]
    pts = [(ch, w[ch]) for ch in keys(F.A) if haskey(B.points, w[ch])]
    return kcol, groups, feeds, pts
end

function belief_free_energies(cs::AbstractVector{<:ClusterFactor}, B::FrozenClusterBeliefs, ps, st)
    c = cs[1]
    nA, N = size(B.q)
    n = size(B.M, 1)
    terms = map(eachindex(c.factors)) do i
        F = cluster_form(c.factors[i], _subtree(ps, c.keys[i]))
        kcol, groups, feeds, pts = _batch_layout(c, i, F, B)
        cx = -vec(sum(B.q .* _finite_logw(F.logw[kcol]); dims = 1))
        m = size(F.C, 1)
        m == 0 && return (zeros(N), cx)
        parts = map(groups) do (k, idx, cols, cls)
            na = length(idx)
            qk = B.q[idx, :]
            R = repeat(-F.C[:, kcol[idx]], 1, N)
            for (ch, v) in pts
                R = R .+ repeat(_at(getfield(F.A, ch), k) * B.points[v]; inner = (1, na))
            end
            CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(_at(F.Q, k))))
            tr = zeros(N)
            if n > 0
                G = reduce(hcat, [isempty(chs) ? zeros(m, length(B.offset[v])) : reduce(+, [_at(getfield(F.A, ch), k) for ch in chs])
                                  for (v, chs) in zip(B.free, feeds)])
                R = R .+ G * B.M[:, cols]
                T = map(Σ -> LinearAlgebra.tr(CQ \ (G * Σ * G')), B.Σ)
                tr = vec(sum(qk .* T[cls]; dims = 1))
            end
            quad = reshape(vec(sum(R .* (CQ \ R); dims = 1)), na, N)
            ((vec(sum(qk .* quad; dims = 1)) .+ tr) ./ 2, vec(sum(qk; dims = 1)) .* ((logdet(CQ) + m * log(2π)) / 2))
        end
        return (reduce(+, map(first, parts)), cx .+ reduce(+, map(last, parts)))
    end
    fit, complexity = reduce(+, map(first, terms)), reduce(+, map(last, terms))
    Hq = vec(-sum(ifelse.(B.q .> 0, B.q .* log.(max.(B.q, floatmin(Float64))), 0.0); dims = 1))
    Hx = vec(sum(B.q .* B.H[B.class]; dims = 1))
    return [LenticulumCore.GradedEnergy((fit = fit[r], complexity = complexity[r], negentropy = -Hq[r] - Hx[r])) for r in 1:N], st
end

function local_free_energies(cs::AbstractVector{<:ClusterFactor}, msgs::NamedTuple, ps, st)
    B = factor_beliefs(cs, msgs, ps, st)
    B isa FrozenClusterBeliefs && return belief_free_energies(cs, B, ps, st)
    return [first(belief_free_energy(cs[i], B[i], ps, st)) for i in eachindex(cs)], st
end
