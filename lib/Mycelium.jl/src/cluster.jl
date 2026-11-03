# ---------------------------------------------------------------------------
# Cluster nodes: exact inference inside, messages outside (Extending Mycelium §5, L-SPEC-009).
#
# `cluster(g, name, factors)` replaces a set of factor nodes by one `ClusterFactor`. The variables
# that only those factors touch become internal; the others are its channels. Inside, inference is
# exact for conditionally linear-Gaussian factors: every factor states, through `cluster_form`, a
# residual r = Σ A_c x_c − C[:, k] with noise N(0, Q) and a weight w_k, the column k chosen by one
# discrete channel (or fixed). For each joint assignment of the cluster's discrete variables the
# continuous ones are jointly Gaussian in canonical form, with one precision J for every
# assignment, and are eliminated exactly; the assignments are enumerated.
#
# See `cluster.md`.
# ---------------------------------------------------------------------------

"""
    ClusterForm(A, C, Q, logw, switch)

A factor as a cluster node reads it: the residual ``r = \\sum_c A_c x_c - C_{:,k}`` with noise
``\\mathcal N(0, Q)`` and log-weight ``\\log w_k``, so that
``-\\log f(x, k) = \\tfrac12 r^\\top Q^{-1} r + \\tfrac12 \\log\\det 2\\pi Q - \\log w_k``. `A` is a `NamedTuple` of
matrices keyed by the continuous channels, `C` has one column per outcome, `logw` one entry
per outcome, and `switch` names the discrete channel the outcome is read from (`nothing` with one
outcome), or a tuple of discrete channels whose joint state is the outcome (column-major: the
first channel varies fastest, as `LinearIndices`). A discrete-only factor has `m = 0` rows.
Returned by [`cluster_form`](@ref).
"""
struct ClusterForm{A<:NamedTuple,C<:AbstractMatrix,Q<:AbstractMatrix,W<:AbstractVector}
    A::A
    C::C
    Q::Q
    logw::W
    switch::Union{Symbol,Nothing,Tuple{Vararg{Symbol}}}
end

"""
    cluster_form(factor, ps) -> ClusterForm

The factor as a conditionally linear-Gaussian term, for [`cluster`](@ref) nodes, and the discrete
channel it switches on (structural, without parameters). Mycelium implements them for
`PriorFactor`s whose belief is a `CategoricalBelief` (a discrete term) or a proper
`GaussianBelief`; Lenticulum for `LinearConstraintFactor`, `SwitchingConstraintFactor` and
`GaussianPrior` without learnable noise. A factor without a method cannot be put in a cluster.
"""
function cluster_form(factor, ps)
    throw(ArgumentError("$(nameof(typeof(factor))) has no `Mycelium.cluster_form`, so it cannot be put in a cluster node"))
end
"""
    cluster_switch(factor) -> Union{Symbol,Nothing,Tuple}

The discrete channel (or tuple of channels) a factor switches on, read without parameters: what
makes a variable discrete inside a [`cluster`](@ref) node. `nothing` by default.
"""
cluster_switch(factor) = nothing

cluster_switch(f::PriorFactor) = f.belief isa LenticulumCore.CategoricalBelief ? f.channel : nothing
function cluster_form(f::PriorFactor, ps)
    b = f.belief
    if b isa LenticulumCore.CategoricalBelief
        return ClusterForm(NamedTuple(), zeros(0, length(b.logp)), zeros(0, 0), collect(b.logp), f.channel)
    elseif b isa GaussianBelief && LenticulumCore.isproper(b)
        n = dimension(b)
        A = NamedTuple{(f.channel,)}((Matrix(1.0LinearAlgebra.I, n, n),))
        return ClusterForm(A, reshape(LenticulumCore.belief_mean(b), n, 1), Matrix(LenticulumCore.belief_cov(b)), [0.0], nothing)
    end
    throw(ArgumentError("a PriorFactor in a cluster needs a categorical or proper Gaussian belief"))
end

"""
    TableFactor(dims::NamedTuple, logtable::AbstractArray)

A fixed factor over several discrete variables: ``f(k_1, …, k_n) = \exp(\texttt{logtable}[k_1, …, k_n])``,
with channel `c` of `dims[c]` states (`size(logtable) == values(dims)`). Entries may be `-Inf`
(impossible combinations), so a table can tie discrete variables together, e.g. make a switch's
outcome agree with a cluster label. Without parameters.

Inside a [`cluster`](@ref) node it is a discrete term on the joint state of its channels
([`cluster_form`](@ref) with a tuple of switch channels). Outside one it sends exact
belief-propagation messages (sums over the other channels' incoming categorical messages, or
their observed states); it has no free energy of its own outside a cluster.
"""
struct TableFactor{names,T<:AbstractArray{<:Real}} <: LenticulumCore.AbstractLenticulumFactor
    dims::NamedTuple{names}
    logtable::T
    function TableFactor(dims::NamedTuple{names}, logtable::AbstractArray{<:Real}) where {names}
        isempty(dims) && throw(ArgumentError("TableFactor: needs at least one channel"))
        all(d -> d isa Integer && d >= 1, values(dims)) || throw(ArgumentError("TableFactor: dimensions must be positive Ints; got $dims"))
        size(logtable) == Tuple(values(dims)) || throw(DimensionMismatch(
            "TableFactor: the table has size $(size(logtable)), the channels $(Tuple(values(dims)))"))
        any(isnan, logtable) && throw(ArgumentError("TableFactor: the table has NaN entries"))
        all(==(-Inf), logtable) && throw(ArgumentError("TableFactor: every combination is impossible"))
        return new{names,typeof(logtable)}(dims, logtable)
    end
end
LenticulumCore.channels(f::TableFactor{names}) where {names} =
    Tuple(LenticulumCore.Channel(n, Int(f.dims[n])) for n in names)
function LenticulumCore.supported_polarities(f::TableFactor{names}) where {names}
    out = LenticulumCore.Polarity[]
    for t in eachindex(names), rest in Iterators.product(ntuple(_ -> (LenticulumCore.Observed(), LenticulumCore.Latent()), length(names) - 1)...)
        vals = Any[rest...]
        insert!(vals, t, LenticulumCore.Unobserved())
        push!(out, _polarity(names, Tuple(vals)))
    end
    return Tuple(out)
end
LenticulumCore.supports_polarity(f::TableFactor{names}, p::LenticulumCore.Polarity) where {names} =
    Set(keys(p)) == Set(names) && length(LenticulumCore.unobserved_channels(p)) == 1
LenticulumCore.islearnable(::TableFactor) = false
LuxCore.initialparameters(::AbstractRNG, ::TableFactor) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::TableFactor) = NamedTuple()
cluster_switch(::TableFactor{names}) where {names} = names
cluster_form(f::TableFactor{names}, ps) where {names} =
    ClusterForm(NamedTuple(), zeros(0, length(f.logtable)), zeros(0, 0), vec(collect(Float64, f.logtable)), names)

# the log-weights a channel's incoming message puts on its states (an observed state is one-hot)
function _table_input(b, K::Int, name)
    b === nothing && return zeros(K)
    b isa LenticulumCore.TrivialBelief && return zeros(K)
    b isa LenticulumCore.CategoricalBelief && length(b.logp) == K && return collect(Float64, b.logp)
    if b isa LenticulumCore.DiracBelief
        v = b.value isa AbstractArray ? only(b.value) : b.value
        v isa Integer && 1 <= v <= K || throw(ArgumentError("TableFactor: an observed :$name must be a state in 1:$K; got $v"))
        l = fill(-Inf, K); l[v] = 0.0
        return l
    end
    throw(ArgumentError("TableFactor: the message into :$name must be categorical with $K states; got $(typeof(b))"))
end
function factor_message(f::TableFactor{names}, target::Symbol, _, inputs, _, ps, st) where {names}
    t = findfirst(==(target), names)
    t === nothing && throw(ArgumentError("TableFactor: no channel :$target"))
    T = collect(Float64, f.logtable)
    for (j, n) in enumerate(names)
        j == t && continue
        l = _table_input(haskey(inputs, n) ? getfield(inputs, n) : nothing, Int(f.dims[n]), n)
        T = T .+ reshape(l, ntuple(i -> i == j ? length(l) : 1, length(names)))
    end
    lk = [_lse(vec(selectdim(T, t, k))) for k in 1:Int(f.dims[target])]
    all(==(-Inf), lk) && throw(ArgumentError("TableFactor: every state of :$target has zero probability"))
    return LenticulumCore.CategoricalBelief(Val(:log), lk .- _lse(lk), nothing), st
end

"""
    ClusterFactor

A cluster node, built by [`cluster`](@ref): the factors it replaces (`names`, `factors`, each
with its channels wired to cluster variables, and the parameter key it read, `keys`), its
variables and their dimensions, which of them are discrete, and the channels it exposes
(`boundary`, one per variable shared with the rest of the graph, named after it). Its parameters
are a `NamedTuple` keyed by those parameter keys, so factors tied inside the cluster still share
one entry ([`cluster_parameters`](@ref)).
"""
struct ClusterFactor <: LenticulumCore.AbstractLenticulumFactor
    names::Vector{Symbol}
    keys::Vector{Symbol}
    factors::Vector{Any}
    wiring::Vector{Vector{Pair{Symbol,Symbol}}}    # per factor: its channel => cluster variable
    variables::Vector{Symbol}
    dims::Dict{Symbol,Int}
    discrete::Vector{Symbol}
    continuous::Vector{Symbol}
    boundary::Vector{Symbol}
end

LenticulumCore.channels(c::ClusterFactor) = Tuple(LenticulumCore.Channel(v, c.dims[v]) for v in c.boundary)
function LenticulumCore.supported_polarities(c::ClusterFactor)
    names = Tuple(c.boundary)
    out = LenticulumCore.Polarity[]
    for t in eachindex(names), rest in Iterators.product(ntuple(_ -> (LenticulumCore.Observed(), LenticulumCore.Latent()), length(names) - 1)...)
        vals = Any[rest...]
        insert!(vals, t, LenticulumCore.Unobserved())
        push!(out, _polarity(names, Tuple(vals)))
    end
    return Tuple(out)
end
function LenticulumCore.supports_polarity(c::ClusterFactor, p::LenticulumCore.Polarity)
    Set(keys(p)) == Set(c.boundary) || return false
    return length(LenticulumCore.unobserved_channels(p)) == 1
end
LenticulumCore.hasnopolarity(::ClusterFactor) = false
LenticulumCore.isunidirectional(::ClusterFactor) = false
LenticulumCore.islearnable(c::ClusterFactor) = any(LenticulumCore.islearnable, c.factors)
LenticulumCore.energyspace(::ClusterFactor) = LenticulumCore.GradedEnergySpace((
    fit = LenticulumCore.ScalarEnergySpace(), complexity = LenticulumCore.ScalarEnergySpace(),
    negentropy = LenticulumCore.ScalarEnergySpace()))
LenticulumCore.scalarisation(::ClusterFactor) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(), complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation()))
_unique_keys(c::ClusterFactor) = unique(c.keys)
LuxCore.initialparameters(rng::AbstractRNG, c::ClusterFactor) = NamedTuple{Tuple(_unique_keys(c))}(
    Tuple(LuxCore.initialparameters(rng, c.factors[findfirst(==(k), c.keys)]) for k in _unique_keys(c)))
LuxCore.initialstates(rng::AbstractRNG, c::ClusterFactor) = NamedTuple{Tuple(_unique_keys(c))}(
    Tuple(LuxCore.initialstates(rng, c.factors[findfirst(==(k), c.keys)]) for k in _unique_keys(c)))
uses_cavity(::ClusterFactor) = true

# --- The graph transformation -------------------------------------------------------

"""
    cluster(g::FactorGraph, name::Symbol, factors) -> FactorGraph
    cluster(g::FactorGraph, groups::AbstractVector{<:Pair}; params = nothing) -> FactorGraph

The graph with the factor nodes named in `factors` replaced by one [`ClusterFactor`](@ref) node
`name`; the second form makes one node per `name => factors` pair, at once. Variables touched
only by a group's factors become internal to its node; every other variable they touch becomes
one of its channels, named after the variable, on a bidirectional edge. Inside the node inference
is exact (`cluster.md`): each factor must have a [`cluster_form`](@ref), a discrete variable is one
some factor switches on ([`cluster_switch`](@ref)), and every variable needs an `Int` dimension (its
number of states, if discrete).

A cluster node reads the parameter entry `name`, a `NamedTuple` keyed by its factors' parameter
keys ([`cluster_parameters`](@ref)). With `params`, every node reads that one entry instead: the
nodes are tied, which is how replicas of one model (one cluster per data point) share their
parameters, and which requires every node to hold the same parameter keys.

Refused: graphs with plates; groups that overlap; factors that are not belief-propagation nodes,
have frozen parameter paths, or share a parameter key with a node outside every group (or, without
`params`, with a node in another group); unconnected channels; edges of the clustered factors
that are not bidirectional, diagonal and attached.
"""
cluster(g::FactorGraph, name::Symbol, factors) = cluster(g, [name => factors])

function cluster(g::FactorGraph, groups::AbstractVector{<:Pair}; params::Union{Symbol,Nothing} = nothing)
    isempty(g.plates) || throw(ArgumentError("cluster: graphs with plates are not supported"))
    isempty(groups) && throw(ArgumentError("cluster: no groups"))
    names = Symbol[first(gr) for gr in groups]
    allunique(names) || throw(ArgumentError("cluster: the cluster names must differ"))
    fidss = Vector{Int}[]
    for (name, factors) in groups
        fids = Int[]
        for fname in factors
            haskey(g.factor_index, fname) || throw(ArgumentError("cluster: no factor :$fname"))
            push!(fids, g.factor_index[fname])
        end
        allunique(fids) && !isempty(fids) || throw(ArgumentError("cluster: give each factor once, and at least one"))
        push!(fidss, fids)
    end
    allunique(reduce(vcat, fidss)) || throw(ArgumentError("cluster: the groups overlap"))
    union_ = Set(reduce(vcat, fidss))
    for (name, fids) in zip(names, fidss)
        haskey(g.factor_index, name) && !(g.factor_index[name] in union_) && throw(ArgumentError("cluster: a factor :$name exists already"))
    end
    nodes = [_cluster_node(g, fids, Set(fids), union_, params !== nothing) for fids in fidss]
    if params !== nothing
        all(c -> Set(_unique_keys(c)) == Set(_unique_keys(first(nodes))), nodes) || throw(ArgumentError(
            "cluster: tied cluster nodes (params = :$params) must hold the same parameter keys"))
    end
    # a cluster's parameter key must not be one that a node left outside reads
    outside_keys = Set(f.params for f in g.factors if !(f.id in union_))
    for k in (params === nothing ? names : [params])
        k in outside_keys && throw(ArgumentError(
            "cluster: the parameter key :$k is read by a node outside the clusters; choose another name or `params`"))
    end
    owner = Dict(f => k for (k, fids) in enumerate(fidss) for f in fids)
    internal = union(Set{Int}(), (_internal_variables(g, fids) for fids in fidss)...)
    b = GraphBuilder()
    for v in g.variables
        v.id in internal || variable!(b, v.name, v.space)
    end
    placed = falses(length(groups))
    for f in g.factors
        k = get(owner, f.id, 0)
        if k == 0
            factor!(b, f.name, f.factor; freeze = f.freeze, rule = f.rule, params = f.params)
        elseif !placed[k]
            factor!(b, names[k], nodes[k]; params = something(params, names[k])); placed[k] = true
        end
    end
    for e in g.edges
        haskey(owner, e.factor) && continue
        connect!(b, g.factors[e.factor].name, e.channel, g.variables[e.variable].name;
                 direction = e.direction, coupling = e.coupling, detach = e.detach)
    end
    for (name, node) in zip(names, nodes), v in node.boundary
        connect!(b, name, v, v)
    end
    return build(b)
end

_internal_variables(g, fids) = (inside = Set(fids); vids = unique(g.edges[ei].variable for fid in fids for ei in g.edges_of_factor[fid]);
    Set(v for v in vids if all(ei -> g.edges[ei].factor in inside, g.edges_of_variable[v])))

# One group's node, after the checks of `cluster`.
function _cluster_node(g, fids, inside, union_, tied)
    for fid in fids
        f = g.factors[fid]
        f.rule isa BeliefPropagation || throw(ArgumentError("cluster: :$(f.name) is not a BeliefPropagation() node"))
        which(cluster_form, Tuple{typeof(f.factor),Any}) === which(cluster_form, Tuple{Any,Any}) && throw(ArgumentError(
            "cluster: :$(f.name) ($(nameof(typeof(f.factor)))) has no `Mycelium.cluster_form`; keep it outside the cluster"))
        isempty(f.freeze) || throw(ArgumentError("cluster: :$(f.name) has frozen parameter paths"))
        all(h -> h.params !== f.params || h.id in (tied ? union_ : inside), g.factors) || throw(ArgumentError(
            "cluster: :$(f.name) shares its parameter key :$(f.params) with a node outside the cluster" *
            (tied ? "s" : " (tie the clusters with `params` to share keys between them)")))
        wired = Set(g.edges[ei].channel for ei in g.edges_of_factor[fid])
        for ch in LenticulumCore.channels(f.factor)
            LenticulumCore.channelname(ch) in wired || throw(ArgumentError(
                "cluster: channel :$(LenticulumCore.channelname(ch)) of :$(f.name) is not connected; inside a cluster every channel needs a variable"))
        end
        for ei in g.edges_of_factor[fid]
            e = g.edges[ei]
            e.direction isa Bidirectional && e.coupling isa DiagonalCoupling && !e.detach || throw(ArgumentError(
                "cluster: the edge of :$(f.name) on :$(e.channel) must be bidirectional, diagonal and attached"))
        end
    end
    vids = unique(g.edges[ei].variable for fid in fids for ei in g.edges_of_factor[fid])
    internal = _internal_variables(g, fids)
    vnames = [g.variables[v].name for v in vids]
    dims = Dict{Symbol,Int}()
    for v in vids
        sp = g.variables[v].space
        sp isa Integer || throw(ArgumentError("cluster: variable :$(g.variables[v].name) needs an Int dimension"))
        dims[g.variables[v].name] = Int(sp)
    end
    wiring = [[g.edges[ei].channel => g.variables[g.edges[ei].variable].name for ei in g.edges_of_factor[fid]] for fid in fids]
    discrete = Symbol[]
    for (fid, w) in zip(fids, wiring)
        sws = cluster_switch(g.factors[fid].factor)
        sws === nothing && continue
        for sw in (sws isa Symbol ? (sws,) : sws)
            k = findfirst(p -> p.first === sw, w)
            k === nothing || push!(discrete, w[k].second)
        end
    end
    discrete = unique(discrete)
    continuous = [v for v in vnames if !(v in discrete)]
    boundary = [g.variables[v].name for v in vids if !(v in internal)]
    return ClusterFactor([g.factors[f].name for f in fids], [g.factors[f].params for f in fids],
                         Any[g.factors[f].factor for f in fids], wiring, vnames, dims, discrete, continuous, boundary)
end

"""
    cluster_parameters(gc, ps) -> NamedTuple
    cluster_parameters(gc, name, ps) -> NamedTuple

A parameter (or state) tree of the original graph as the clustered graph `gc` reads it: each
cluster node's factors' entries are moved under the node's parameter key (its name, or the shared
`params` of tied nodes), keyed by their parameter keys. The second form checks that `name` is a
cluster node and converts for all of them.
"""
function cluster_parameters(gc::FactorGraph, ps::NamedTuple)
    nodes = [f for f in gc.factors if f.factor isa ClusterFactor]
    moved = Dict{Symbol,Any}()
    inner_keys = Set{Symbol}()
    for f in nodes
        ks = _unique_keys(f.factor)
        union!(inner_keys, ks)
        haskey(moved, f.params) || (moved[f.params] = NamedTuple{Tuple(ks)}(Tuple(_subtree(ps, k) for k in ks)))
    end
    rest = Tuple(k for k in keys(ps) if !(k in inner_keys) && !haskey(moved, k))
    outer = NamedTuple{rest}(Tuple(getfield(ps, k) for k in rest))
    added = Tuple(unique(f.params for f in nodes))
    return merge(outer, NamedTuple{added}(Tuple(moved[k] for k in added)))
end
function cluster_parameters(gc::FactorGraph, name::Symbol, ps::NamedTuple)
    factornode(gc, gc.factor_index[name]).factor isa ClusterFactor ||
        throw(ArgumentError("cluster_parameters: :$name is not a cluster node"))
    return cluster_parameters(gc, ps)
end

"""
    overlap_groups(C::AbstractMatrix, maxsize) -> Vector{Vector{Int}}

Groups of at most `maxsize` items by their pairwise overlaps `C` (symmetric, larger is more
overlap), E7's rule (`Composing Stochastic Factors`): starting from singletons, merge the two
groups whose largest overlap between members is largest, as long as the merged group fits and
that overlap is positive. Lenticulum's `switch_overlaps` gives the overlaps of switching factors
feeding one sum; the groups then go to [`cluster`](@ref).
"""
function overlap_groups(C::AbstractMatrix, maxsize::Integer)
    L = size(C, 1)
    size(C, 2) == L || throw(DimensionMismatch("the overlaps must be a square matrix"))
    maxsize >= 1 || throw(ArgumentError("maxsize must be at least 1"))
    groups = [[l] for l in 1:L]
    while true
        best, pair = 0.0, nothing
        for a in eachindex(groups), b in (a + 1):length(groups)
            length(groups[a]) + length(groups[b]) <= maxsize || continue
            o = maximum(C[l, m] for l in groups[a] for m in groups[b])
            o > best && ((best, pair) = (o, (a, b)))
        end
        pair === nothing && break
        a, b = pair
        groups[a] = sort(vcat(groups[a], groups[b])); deleteat!(groups, b)
    end
    return groups
end

# The outcome column each assignment reads: the state of the factor's switch channel, or the
# column-major index of the joint state of its switch channels.
function _switch_columns(c::ClusterFactor, F::ClusterForm, w, assignments)
    F.switch === nothing && return fill(1, length(assignments))
    if F.switch isa Symbol
        j = findfirst(==(w[F.switch]), c.discrete)
        return [a[j] for a in assignments]
    end
    js = [findfirst(==(w[ch]), c.discrete) for ch in F.switch]
    L = LinearIndices(Tuple(c.dims[w[ch]] for ch in F.switch))
    return [L[(a[j] for j in js)...] for a in assignments]
end

# --- Exact inference inside -----------------------------------------------------

# The cluster's joint in canonical form, given its incoming messages (`msgs`, keyed by boundary
# variable; `skip` excluded): the free continuous variables and their offsets, the points
# substituted, the assignments of the discrete variables, J, one h and one constant per assignment.
struct _ClusterJoint
    free::Vector{Symbol}
    offset::Dict{Symbol,UnitRange{Int}}
    points::Dict{Symbol,Vector{Float64}}
    assignments::Vector{Vector{Int}}           # per assignment: the state of each discrete variable
    J::Matrix{Float64}
    H::Matrix{Float64}                         # n × nA
    c::Vector{Float64}                         # nA
end

_incoming(msgs, v, skip) = v === skip ? nothing : (haskey(msgs, v) ? getfield(msgs, v) : nothing)
_cluster_input(b) = b
_cluster_input(b::LenticulumCore.MixtureBelief{<:GaussianBelief}) = ep_project(b) === nothing ?
    throw(ArgumentError("a cluster node collapses a mixture input to its moments, which needs proper components")) : ep_project(b)

function _cluster_joint(c::ClusterFactor, msgs, ps, skip)
    points = Dict{Symbol,Vector{Float64}}()
    for v in c.continuous
        b = _incoming(msgs, v, skip)
        b isa LenticulumCore.DiracBelief && (points[v] = collect(float.(b.value isa Real ? [b.value] : b.value)))
    end
    free = [v for v in c.continuous if !haskey(points, v)]
    offset = Dict{Symbol,UnitRange{Int}}()
    n = 0
    for v in free
        offset[v] = (n + 1):(n + c.dims[v]); n += c.dims[v]
    end
    # the discrete states allowed and their incoming log-probabilities
    states, lps = Vector{Vector{Int}}(), Vector{Vector{Float64}}()
    for v in c.discrete
        K = c.dims[v]
        b = _incoming(msgs, v, skip)
        if b isa LenticulumCore.DiracBelief
            k = Int(only(b.value isa Real ? [b.value] : b.value))
            push!(states, [k]); push!(lps, [0.0])
        elseif b isa LenticulumCore.CategoricalBelief
            length(b.logp) == K || throw(DimensionMismatch("message to :$v has $(length(b.logp)) states, the variable $K"))
            ks = [k for k in 1:K if b.logp[k] > -Inf]
            isempty(ks) && throw(ArgumentError("cluster: the message to :$v gives every state zero probability"))
            push!(states, ks); push!(lps, b.logp[ks])
        else
            push!(states, collect(1:K)); push!(lps, zeros(K))
        end
    end
    combos = vec(collect(Iterators.product(eachindex.(states)...)))
    nA = length(combos)
    nA <= 2^16 || throw(ArgumentError("cluster: $nA joint discrete states; at most 2^16 are enumerated"))
    assignments = [[states[j][ix[j]] for j in eachindex(states)] for ix in combos]
    cst = [sum(lps[j][ix[j]] for j in eachindex(states); init = 0.0) for ix in combos]
    forms = [cluster_form(f, _subtree(ps, c.keys[i])) for (i, f) in enumerate(c.factors)]
    wirings = [Dict(w) for w in c.wiring]
    for (F, w) in zip(forms, wirings)
        cst .+= F.logw[_switch_columns(c, F, w, assignments)]
    end
    # assignments of weight zero (a -Inf log-weight, e.g. a `TableFactor` tying discrete variables)
    # are dropped before the Gaussian part, which is the expensive one
    keep = findall(>(-Inf), cst)
    isempty(keep) && throw(ArgumentError("cluster: every joint discrete state has zero probability"))
    if length(keep) < length(cst)
        assignments, cst = assignments[keep], cst[keep]
    end
    nA = length(assignments)
    J = zeros(n, n); H = zeros(n, nA)
    # incoming Gaussian messages on free boundary variables
    for v in free
        b = _cluster_input(_incoming(msgs, v, skip))
        b isa GaussianBelief || continue
        o = offset[v]
        J[o, o] .+= b.Λ
        H[o, :] .+= b.η
    end
    for (F, w) in zip(forms, wirings)
        m = size(F.C, 1)
        m == 0 && continue
        kcol = _switch_columns(c, F, w, assignments)
        G = zeros(m, n)
        shift = zeros(m)
        for (ch, Ach) in pairs(F.A)
            v = w[ch]
            haskey(points, v) ? (shift .+= Ach * points[v]) : (G[:, offset[v]] .+= Ach)
        end
        CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(F.Q)))
        Cc = F.C[:, kcol] .- shift                       # r = G x − c_a
        QiC = CQ \ Cc
        QiG = CQ \ G
        J .+= G' * QiG
        H .+= G' * QiC
        cst .+= -vec(sum(Cc .* QiC; dims = 1)) ./ 2 .- (logdet(CQ) + m * log(2π)) / 2
    end
    return _ClusterJoint(free, offset, points, assignments, (J + J') / 2, H, cst)
end

# log ∫ exp(hᵀx − xᵀJx/2) dx for each column of H, with J positive definite.
function _log_integrals(J, H, what)
    size(J, 1) == 0 && return zeros(size(H, 2))
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(J); check = false)
    LinearAlgebra.issuccess(C) || throw(ArgumentError(
        "cluster: $what are not determined by the cluster's factors and incoming messages (singular precision)"))
    return vec(sum(H .* (C \ H); dims = 1)) ./ 2 .- logdet(C) / 2 .+ size(J, 1) * log(2π) / 2
end

_lse(v) = (m = maximum(v); m == -Inf ? -Inf : m + log(sum(exp.(v .- m))))

"""
    Mycelium.factor_message(c::ClusterFactor, target, polarity, inputs, prior, ps, st)

The exact message of the cluster to one of its channels, given the others' messages: every joint
assignment of the discrete variables is enumerated, and the continuous variables other than the
target are eliminated from that assignment's Gaussian (one precision for all). To a continuous
channel the message is a mixture over the assignments (a `GaussianBelief` if there is one); to a
discrete channel a `CategoricalBelief`. Points are substituted; a Gaussian-mixture input is
collapsed to its moments.
"""
function factor_message(c::ClusterFactor, target::Symbol, polarity, inputs, prior, ps, st)
    target in c.boundary || throw(ArgumentError("channel :$target is not a channel of this cluster (has $(c.boundary))"))
    jt = _cluster_joint(c, inputs, ps, target)
    if target in c.discrete
        l = jt.c .+ _log_integrals(jt.J, jt.H, "its continuous variables")
        j = findfirst(==(target), c.discrete)
        K = c.dims[target]
        lk = [_lse([l[a] for a in eachindex(l) if jt.assignments[a][j] == k]; ) for k in 1:K]
        all(==(-Inf), lk) && throw(ArgumentError("cluster: every state of :$target has zero probability"))
        return LenticulumCore.CategoricalBelief(Val(:log), lk .- _lse(lk), nothing), st
    end
    ot = jt.offset[target]
    rs = setdiff(1:size(jt.J, 1), ot)
    Jrr, Jtr, Jtt = jt.J[rs, rs], jt.J[ot, rs], jt.J[ot, ot]
    Hr, Ht = jt.H[rs, :], jt.H[ot, :]
    if isempty(rs)
        Λ, E, l = Jtt, Ht, copy(jt.c)
    else
        Cr = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Jrr); check = false)
        LinearAlgebra.issuccess(Cr) || throw(ArgumentError(
            "cluster: the continuous variables other than :$target are not determined by the cluster's factors and incoming messages"))
        X = Cr \ Hr
        Λ = Jtt - Jtr * (Cr \ Jtr')
        E = Ht - Jtr * X
        l = jt.c .+ vec(sum(Hr .* X; dims = 1)) ./ 2 .- logdet(Cr) / 2 .+ length(rs) * log(2π) / 2
    end
    Λ = (Λ + Λ') / 2
    all(iszero, Λ) && all(iszero, E) && return LenticulumCore.TrivialBelief(), st
    d = length(ot)
    CΛ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Λ); check = false)
    proper = LinearAlgebra.issuccess(CΛ)
    if proper                                   # normalised components: their mass into the weight
        l = l .+ vec(sum(E .* (CΛ \ E); dims = 1)) ./ 2 .- logdet(CΛ) / 2 .+ d * log(2π) / 2
    end
    keep = findall(>(-Inf), l)
    isempty(keep) && throw(ArgumentError("cluster: every assignment has zero probability"))
    comps = [GaussianBelief(E[:, a], Λ, !proper) for a in keep]
    length(comps) == 1 && return only(comps), st
    lk = l[keep]
    return LenticulumCore.MixtureBelief(Val(:log), comps, lk .- _lse(lk)), st
end

# --- The free energy --------------------------------------------------------------

"""
    FrozenClusterBelief

The cluster's belief ``b(x, a) \\propto \\prod_f f(x, a) \\prod_v \\mu_v``, held fixed: the posterior `q` over
the discrete assignments, the means of the free continuous variables per assignment (columns of
`M`), their covariance `Σ` (one for all assignments), the conditional entropy ``H(x \\mid a)``, the
substituted points, and the assignments.
"""
struct FrozenClusterBelief{Q<:AbstractVector,M<:AbstractMatrix,S<:AbstractMatrix,H<:Real}
    q::Q
    M::M
    Σ::S
    H::H
    free::Vector{Symbol}
    offset::Dict{Symbol,UnitRange{Int}}
    points::Dict{Symbol,Vector{Float64}}
    assignments::Vector{Vector{Int}}
end

function factor_belief(c::ClusterFactor, msgs, ps, st)
    jt = _cluster_joint(c, msgs, ps, nothing)
    n = size(jt.J, 1)
    l = jt.c .+ _log_integrals(jt.J, jt.H, "its continuous variables")
    all(==(-Inf), l) && throw(ArgumentError("cluster: every assignment has zero probability"))
    q = exp.(l .- _lse(l))
    if n == 0
        M, Σ, H = zeros(0, length(q)), zeros(0, 0), 0.0
    else
        C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(jt.J))
        M, Σ = C \ jt.H, Matrix(inv(C))
        H = n * (1 + log(2π)) / 2 - logdet(C) / 2
    end
    return FrozenClusterBelief(q, M, Σ, H, jt.free, jt.offset, jt.points, jt.assignments)
end

"""
    Mycelium.local_free_energy(c::ClusterFactor, msgs, ps, st)
    Mycelium.belief_free_energy(c::ClusterFactor, b::FrozenClusterBelief, ps, st)

``\\mathbb E_b[-\\log \\prod_f f] - H(b)`` for the cluster's exact belief, graded as `(fit, complexity,
negentropy)`: `fit` is ``\\sum_f \\sum_a q_a \\tfrac12 \\mathbb E[r_f^\\top Q_f^{-1} r_f \\mid a]``, `complexity`
``\\sum_f (\\tfrac12 \\log\\det 2\\pi Q_f - \\sum_a q_a \\log w_{f,a})``, and `negentropy` ``-H(q) - H(x \\mid a)``, closed form
since given the assignment the continuous part is one Gaussian. `belief_free_energy` re-evaluates
the factors' terms at `ps` with the belief fixed (the parameter gradient, LP-020).
"""
local_free_energy(c::ClusterFactor, msgs, ps, st) = belief_free_energy(c, factor_belief(c, msgs, ps, st), ps, st)

# Which outcome column each assignment reads, which channels feed each free variable, and which
# channels hold points: structure only, computed outside AD (`@non_differentiable` in the
# ChainRulesCore extension).
function _cluster_layout(c::ClusterFactor, i::Int, F::ClusterForm, b::FrozenClusterBelief)
    w = Dict(c.wiring[i])
    kcol = _switch_columns(c, F, w, b.assignments)
    feeds = [[ch for ch in keys(F.A) if w[ch] === v] for v in b.free]
    pts = [(ch, b.points[w[ch]]) for ch in keys(F.A) if haskey(b.points, w[ch])]
    return kcol, feeds, pts
end

_finite_logw(lw) = map(x -> x == -Inf ? zero(x) : x, lw)

function belief_free_energy(c::ClusterFactor, b::FrozenClusterBelief, ps, st)
    n = size(b.M, 1)
    terms = map(eachindex(c.factors)) do i
        F = cluster_form(c.factors[i], _subtree(ps, c.keys[i]))
        kcol, feeds, pts = _cluster_layout(c, i, F, b)
        # an impossible outcome (log-weight -Inf) has q = 0 exactly, and its 0·log 0 term is 0; the
        # entry is replaced before the product so that no AD backend meets 0 · (-Inf)
        cx = -sum(b.q .* _finite_logw(F.logw[kcol]))
        m = size(F.C, 1)
        m == 0 && return (0.0, cx)
        R = -F.C[:, kcol]
        for (ch, x) in pts
            R = R .+ getfield(F.A, ch) * x
        end
        Cr = zeros(m, m)
        if n > 0
            G = reduce(hcat, [isempty(chs) ? zeros(m, length(b.offset[v])) : reduce(+, [getfield(F.A, ch) for ch in chs])
                              for (v, chs) in zip(b.free, feeds)])
            R = R .+ G * b.M
            Cr = G * b.Σ * G'
        end
        CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(F.Q)))
        quad = vec(sum(R .* (CQ \ R); dims = 1))
        return ((sum(b.q .* quad) + LinearAlgebra.tr(CQ \ Cr)) / 2, cx + (logdet(CQ) + m * log(2π)) / 2)
    end
    fit, complexity = sum(first, terms), sum(last, terms)
    Hq = -sum(ifelse.(b.q .> 0, b.q .* log.(max.(b.q, floatmin(Float64))), 0.0))
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -Hq - b.H)), st
end

"""
    cluster_marginal(store, gc, name, v, ps, st = NamedTuple()) -> AbstractBelief

The marginal of variable `v` of cluster node `name` (internal or a channel) under the cluster's
exact belief given its current incoming messages: a mixture over the discrete assignments of the
Gaussian of `v` given each (a `GaussianBelief` if there is one assignment), a categorical for a
discrete variable, the point for a substituted one.
"""
function cluster_marginal(s::AbstractMessageStore, gc::FactorGraph, name::Symbol, v::Symbol, ps, st = NamedTuple())
    fid = gc.factor_index[name]
    c = factornode(gc, fid).factor
    c isa ClusterFactor || throw(ArgumentError("cluster_marginal: :$name is not a cluster node"))
    v in c.variables || throw(ArgumentError("cluster_marginal: :$v is not a variable of cluster :$name"))
    b = factor_belief(c, messages_into(s, gc, fid), _subtree(ps, factornode(gc, fid).params), _subtree(st, name))
    if v in c.discrete
        j = findfirst(==(v), c.discrete)
        p = zeros(c.dims[v])
        for (a, qa) in zip(b.assignments, b.q)
            p[a[j]] += qa
        end
        return LenticulumCore.CategoricalBelief(p)
    end
    haskey(b.points, v) && return LenticulumCore.DiracBelief(b.points[v])
    o = b.offset[v]
    S = Matrix(LinearAlgebra.Symmetric(b.Σ[o, o]))
    comps = [LenticulumCore.Gaussian(b.M[o, a], S) for a in eachindex(b.q) if b.q[a] > 0]
    length(comps) == 1 && return only(comps)
    return LenticulumCore.MixtureBelief(comps, b.q[b.q .> 0])
end
