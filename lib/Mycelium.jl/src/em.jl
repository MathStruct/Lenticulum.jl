# ---------------------------------------------------------------------------
# Closed-form M-steps from cluster beliefs (expectation maximisation).
#
# At a fixed belief b the free energy's parameter-dependent part is Σ_f E_b[−log f(θ_f)], which
# separates over parameter keys. For a factor stated by a `cluster_form` (a conditionally linear-
# Gaussian term), E_b[−log f] depends on b only through the moment statistics of its continuous
# channels per outcome: n_k = Σ_a q_a [k(a) = k], m_k = Σ_a q_a E[x | a] [k(a) = k] and
# S_k = Σ_a q_a E[x xᵀ | a] [k(a) = k], x the factor's continuous channels stacked. `em_step`
# collects them from every cluster node's exact belief (summed over tied nodes and replicas) and
# hands them to the factor's `m_step`, which returns its maximiser. With exact beliefs (every
# factor in a cluster node) one `em_step` after converged inference is one step of EM, so the
# free energy, −log p of the data, does not increase.
#
# See `em.md`.
# ---------------------------------------------------------------------------

"""
    MomentStats

The moment statistics of one factor's continuous channels under a belief, per outcome `k` of the
factor (one outcome without a switch): `n[k]` the mass, `m[:, k]` the sum of the means, `S[:, :, k]`
the sum of the second moments ``\\mathbb E[x x^\\top]``, with `x` the channels `channels` (dimensions
`dims`) stacked in that order. Sums over the assignments of a cluster node's belief, the tied
nodes and the replicas of a plate. Read by [`m_step`](@ref).
"""
struct MomentStats
    channels::Vector{Symbol}
    dims::Vector{Int}
    n::Vector{Float64}
    m::Matrix{Float64}
    S::Array{Float64,3}
end
MomentStats(chs, dims, K) = (d = sum(dims; init = 0); MomentStats(chs, dims, zeros(K), zeros(d, K), zeros(d, d, K)))

function _add!(a::MomentStats, b::MomentStats)
    (a.channels == b.channels && a.dims == b.dims && length(a.n) == length(b.n)) || throw(ArgumentError(
        "em_step: tied factors with different channels or outcomes cannot pool their statistics"))
    a.n .+= b.n; a.m .+= b.m; a.S .+= b.S
    return a
end

"""
    m_step(factor, stats::MomentStats, ps, frozen) -> parameters or nothing

The parameters of `factor` that minimise ``\\mathbb E_b[-\\log f]`` given its moment statistics,
leaving the paths in `frozen` (`:all`, or a tuple of parameter paths relative to the factor) at
their values in `ps`. `nothing` (the default) means no closed form: [`em_step`](@ref) keeps `ps`.
Lenticulum implements it for its constraint factors.
"""
m_step(factor, stats, ps, frozen) = nothing
_has_m_step(f) = which(m_step, Tuple{typeof(f),MomentStats,Any,Any}) !== which(m_step, Tuple{Any,Any,Any,Any})

# The statistics of factor i of cluster node c under its belief b.
function _moment_stats(c::ClusterFactor, i::Int, F::ClusterForm, b::FrozenClusterBelief)
    w = Dict(c.wiring[i])
    K = size(F.C, 2)
    chs = collect(keys(F.A))
    dims = [size(getfield(F.A, ch), 2) for ch in chs]
    st = MomentStats(chs, dims, K)
    kcol = _switch_columns(c, F, w, b.assignments)
    offs = cumsum([0; dims])
    # per class: the covariance of x (free channels only; points have none)
    Σx = map(b.Σ) do Σ
        X = zeros(sum(dims; init = 0), sum(dims; init = 0))
        for (p, cp) in enumerate(chs), (r, cr) in enumerate(chs)
            vp, vr = w[cp], w[cr]
            (haskey(b.offset, vp) && haskey(b.offset, vr)) || continue
            X[(offs[p] + 1):offs[p + 1], (offs[r] + 1):offs[r + 1]] .= Σ[b.offset[vp], b.offset[vr]]
        end
        X
    end
    for a in eachindex(b.q)
        qa = b.q[a]
        qa > 0 || continue
        μ = isempty(chs) ? Float64[] : reduce(vcat, [haskey(b.points, w[ch]) ? b.points[w[ch]] : b.M[b.offset[w[ch]], a] for ch in chs])
        k = kcol[a]
        st.n[k] += qa
        st.m[:, k] .+= qa .* μ
        st.S[:, :, k] .+= qa .* (μ * μ' .+ Σx[b.class[a]])
    end
    return st
end

# a cluster node's freeze, for the factor under key k: :all, or its paths with k stripped
function _inner_freeze(freeze, k::Symbol)
    paths = LenticulumCore._freeze_paths(freeze)
    paths === :all && return :all
    any(p -> p == (k,), paths) && return :all
    return Tuple(Base.tail(p) for p in paths if length(p) > 1 && p[1] === k)
end

"""
    em_step(store, g, ps, st) -> ps′

One M-step: from every cluster node's exact belief under the store's messages, the moment
statistics of each clustered factor (pooled over tied nodes and plate replicas), and each factor's
closed-form [`m_step`](@ref); a factor without one keeps its parameters, as do frozen paths. A
node outside a cluster takes part when its belief is a [`FrozenClusterBelief`](@ref) (a factor
computed as a one-factor cluster, `as_cluster`). Run inference to convergence first: with exact
beliefs this is one step of EM and does not increase the free energy (`em.md`).
"""
function em_step(s::AbstractMessageStore, g::FactorGraph, ps, st)
    acc = Dict{Tuple{Symbol,Union{Symbol,Nothing}},Any}()      # (node key, factor key) => (factor, stats, frozen)
    for fn in g.factors
        LenticulumCore.islearnable(fn.factor) || continue
        fn.freeze === :all && continue
        psn, stn = _subtree(ps, fn.params), _subtree(st, fn.params)
        if fn.factor isa ClusterFactor
            c = fn.factor
            b = factor_belief(c, messages_into(s, g, fn.id), psn, stn)
            for (i, f) in enumerate(c.factors)
                LenticulumCore.islearnable(f) && _has_m_step(f) || continue
                k = c.keys[i]
                F = cluster_form(f, _subtree(psn, k))
                _push_stats!(acc, (fn.params, k), f, _moment_stats(c, i, F, b), _inner_freeze(fn.freeze, k))
            end
        elseif _has_m_step(fn.factor)
            b = factor_belief(fn.factor, messages_into(s, g, fn.id), psn, stn)
            b isa FrozenClusterBelief || continue
            c = as_cluster(fn.factor)
            F = cluster_form(fn.factor, psn)
            _push_stats!(acc, (fn.params, nothing), fn.factor, _moment_stats(c, 1, F, b),
                         LenticulumCore._freeze_paths(fn.freeze))
        end
    end
    out = ps
    for ((nk, fk), (f, stats, frozen)) in acc
        frozen === :all && continue
        old = fk === nothing ? _subtree(out, nk) : _subtree(_subtree(out, nk), fk)
        new = m_step(f, stats, old, frozen)
        new === nothing && continue
        node = fk === nothing ? new : merge(_subtree(out, nk), NamedTuple{(fk,)}((new,)))
        out = merge(out, NamedTuple{(nk,)}((node,)))
    end
    return out
end
_dense(x::NamedTuple) = map(_dense, x)
_dense(x::Tuple) = map(_dense, x)
_dense(x::AbstractArray) = Array(x)
_dense(x) = x
function _push_stats!(acc, key, f, stats, frozen)
    if haskey(acc, key)
        _add!(acc[key][2], stats)
    else
        acc[key] = (f, stats, frozen)
    end
end

"""
    fit_em(g, ps, st; schedule, sweeps = 2, iters = 100, tol = 1e-8, store = nothing, accelerate = false)
        -> (ps, Fs, store)

Expectation maximisation on `g`: `sweeps` sweeps of `schedule` (the E-step; exact when every
learned factor is in a cluster node on a forest), the free energy, then [`em_step`](@ref), until
the free energy falls by less than `tol` or `iters` steps. Returns the parameters, the free energy
before each M-step and, last, at the returned parameters (non-increasing for exact beliefs), and the
store, holding the messages at the returned parameters. A warm `store` is reused.

With `accelerate = true` each step is a SQUAREM cycle (Varadhan & Roland 2008, scheme S3): two EM
steps θ₁ = M(θ₀), θ₂ = M(θ₁), the extrapolation θ′ = θ₀ − 2αr + α²v with r = θ₁ − θ₀, v = θ₂ − 2θ₁ + θ₀,
α = −‖r‖/‖v‖ (at most −1), then one EM step from θ′; a cycle whose free energy exceeds θ₀'s falls back
to θ₂. EM converges slowly when a latent variable is almost determined by the parameters (a switch
with small noise feeding a sum: each M-step moves the offsets by little), and the extrapolation
takes the long steps EM will not (`em.md`). `Fs` then holds one entry per cycle.
"""
function fit_em(g::FactorGraph, ps, st; schedule, sweeps::Int = 2, iters::Int = 100, tol::Real = 1e-8,
                store = nothing, accelerate::Bool = false)
    s = store === nothing ? MessageStore(g) : store
    estep(θ) = (for _ in 1:sweeps; sweep!(s, g, schedule, θ, st); end; first(LenticulumCore.scalar_free_energy(s, g, θ, st)))
    # extrapolate only the subtrees of learnable nodes: flattening the whole tree (one entry per
    # data node, say) compiles code the size of the graph for every new tree type
    lkeys = Tuple(unique(fn.params for fn in g.factors if LenticulumCore.islearnable(fn.factor) && fn.freeze !== :all && haskey(ps, fn.params)))
    learned(p) = NamedTuple{lkeys}(map(k -> getfield(p, k), lkeys))
    flat(p) = LenticulumCore._flatten_active(learned(p))
    Fs = Float64[]
    converged = false
    for _ in 1:iters
        F = estep(ps)
        push!(Fs, F)
        length(Fs) >= 2 && Fs[end - 1] - F < tol && (converged = true; break)
        if !accelerate
            ps = em_step(s, g, ps, st)
            continue
        end
        θ0 = flat(ps)
        p1 = em_step(s, g, ps, st); estep(p1)
        p2 = em_step(s, g, p1, st)
        r = flat(p1) .- θ0
        v = flat(p2) .- 2 .* flat(p1) .+ θ0
        nv = LinearAlgebra.norm(v)
        if nv == 0
            ps = p2
            continue
        end
        α = min(-LinearAlgebra.norm(r) / nv, -1.0)
        # plain arrays, not views of the flat vector: a new parameter type recompiles inference
        pext = merge(ps, _dense(LenticulumCore._unflatten(θ0 .- 2α .* r .+ α^2 .* v, learned(ps))))
        Fext = try estep(pext) catch; Inf end
        if isfinite(Fext)
            pnew = em_step(s, g, pext, st)
            Fnew = try estep(pnew) catch; Inf end
            Fnew <= F && (ps = pnew; continue)
        end
        ps = p2                                   # the extrapolation went uphill: plain EM
    end
    converged || push!(Fs, estep(ps))             # the returned store and `Fs[end]` belong to `ps`
    return ps, Fs, s
end
