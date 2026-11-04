# ---------------------------------------------------------------------------
# Cluster nodes: exact inference inside, messages outside (Extending Mycelium §5, L-SPEC-009).
#
# `cluster(g, name, factors)` replaces a set of factor nodes by one `ClusterFactor`. The variables
# that only those factors touch become internal; the others are its channels. Inside, inference is
# exact for conditionally linear-Gaussian factors: every factor states, through `cluster_form`, a
# residual r = Σ A_c x_c − C[:, k] with noise N(0, Q) and a weight w_k, the column k chosen by one
# discrete channel (or fixed); A and Q may depend on the outcome too. For each joint assignment
# of the cluster's discrete variables the continuous ones are jointly Gaussian in canonical form
# and are eliminated exactly; the assignments are enumerated. Assignments that give the same
# precision J (the same outcomes of every factor whose A or Q depends on its outcome) form one
# precision class and share its factorisations; without such factors there is one class.
# Directions that nothing determines are integrated out in the diffuse limit (pseudo-inverses).
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

The coefficients and the noise may depend on the outcome: an entry of `A` may be an
`m × d × K` array (``A_{c,k}`` = `A[c][:, :, k]`) and `Q` an `m × m × K` array (``Q_k``), with `K` the number
of columns of `C`. The cluster then has one precision per combination of such outcomes
(`cluster.md`, "Per-outcome coefficients and noise").
Returned by [`cluster_form`](@ref).
"""
struct ClusterForm{A<:NamedTuple,C<:AbstractMatrix,Q<:AbstractArray,W<:AbstractVector}
    A::A
    C::C
    Q::Q
    logw::W
    switch::Union{Symbol,Nothing,Tuple{Vararg{Symbol}}}
    function ClusterForm(A::NamedTuple, C::AbstractMatrix, Q::AbstractArray, logw::AbstractVector,
                         switch::Union{Symbol,Nothing,Tuple{Vararg{Symbol}}})
        m, K = size(C)
        ndims(Q) in (2, 3) && size(Q, 1) == size(Q, 2) == m && (ndims(Q) == 2 || size(Q, 3) == K) ||
            throw(DimensionMismatch("ClusterForm: Q has size $(size(Q)); needs ($m, $m) or ($m, $m, $K)"))
        for (ch, a) in pairs(A)
            ndims(a) in (2, 3) && size(a, 1) == m && (ndims(a) == 2 || size(a, 3) == K) ||
                throw(DimensionMismatch("ClusterForm: A.$ch has size $(size(a)); needs ($m, d) or ($m, d, $K)"))
        end
        length(logw) == K || throw(DimensionMismatch("ClusterForm: $(length(logw)) log-weights for $K outcomes"))
        return new{typeof(A),typeof(C),typeof(Q),typeof(logw)}(A, C, Q, logw, switch)
    end
end

# Whether a form's coefficients or noise depend on its outcome, and their values at outcome k.
_varies(F::ClusterForm) = size(F.C, 1) > 0 && (ndims(F.Q) == 3 || any(a -> ndims(a) == 3, values(F.A)))
_at(a::AbstractMatrix, _) = a
_at(a::AbstractArray{<:Any,3}, k) = a[:, :, k]

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

# A categorical message read by position: its labels, if any, must be the states 1:K in order
# (a factor indexes its outcomes 1:K; labels in another order would be read as the wrong states).
function _check_positional(b::LenticulumCore.CategoricalBelief, K::Int, what)
    b.labels === nothing || isequal(collect(b.labels), collect(1:K)) || throw(ArgumentError(
        "$what: a categorical message with labels $(b.labels) cannot be read as the states 1:$K; drop or reorder its labels"))
    return b
end

# the log-weights a channel's incoming message puts on its states (an observed state is one-hot)
function _table_input(b, K::Int, name)
    b === nothing && return zeros(K)
    b isa LenticulumCore.TrivialBelief && return zeros(K)
    b isa LenticulumCore.CategoricalBelief && length(b.logp) == K && return collect(Float64, _check_positional(b, K, "TableFactor").logp)
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

"""
    as_cluster(factor) -> ClusterFactor

One factor as a cluster node over its own channels (variables named after them, parameters
under the key `:factor`): its exact belief-propagation messages and Bethe free energy outside a
cluster, for a factor that has only a [`cluster_form`](@ref). Call the result with parameters
`(factor = ps,)` and inputs keyed by the factor's channel names.
"""
function as_cluster(f)
    names = [LenticulumCore.channelname(ch) for ch in LenticulumCore.channels(f)]
    dims = Dict(LenticulumCore.channelname(ch) => Int(LenticulumCore.channelspace(ch)) for ch in LenticulumCore.channels(f))
    sws = cluster_switch(f)
    discrete = sws === nothing ? Symbol[] : collect(Symbol, sws isa Symbol ? (sws,) : sws)
    return ClusterFactor([:factor], [:factor], Any[f], [[n => n for n in names]], names, dims, discrete,
                         [n for n in names if !(n in discrete)], names)
end

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

Frozen parameter paths carry over: a clustered node's `freeze` becomes the cluster node's paths
under its parameter key (`(key, path...)`, or `(key,)` when frozen whole).

Refused: graphs with plates; groups that overlap; factors that are not belief-propagation nodes or
share a parameter key with a node outside every group (or, without `params`, with a node in
another group); tied groups that freeze different paths; unconnected channels; edges of the
clustered factors that are not bidirectional, diagonal and attached.
"""
cluster(g::FactorGraph, name::Symbol, factors) = cluster(g, [name => factors])

function cluster(g::FactorGraph, groups::AbstractVector{<:Pair}; params::Union{Symbol,Nothing} = nothing)
    isempty(g.plates) || throw(ArgumentError("cluster: on a graph with plates, cluster a plate's factors with `cluster(g, plate, name, factors)`"))
    return _cluster(g, groups, params, nothing)
end

"""
    cluster(g, plate::Symbol, name::Symbol, factors; params = name) -> FactorGraph

Cluster the plate factors `factors` (template names) of every replica of `plate` into one node
per replica, `replica_name(name, i)`, tied (all read `params`). The new graph keeps the plate,
with the cluster nodes in it, so plate schedules group their messages and the batched cluster
methods (`factor_messages`, `factor_beliefs`) share the work that does not depend on the replica.
Inside each node the variables carry their template names (`:z`, not `Symbol("z[3]")`): those are
its channels, the same in every replica, and the names `cluster_marginal` takes. The graph must
have this one plate only.
"""
function cluster(g::FactorGraph, plate::Symbol, name::Symbol, factors::AbstractVector{Symbol}; params::Symbol = name)
    pl = _plate(g, plate)
    length(g.plates) == 1 || throw(ArgumentError("cluster: a graph with several plates is not supported"))
    renames = map(1:pl.n) do i
        Dict(g.variables[pl.variables[r, i]].name => Symbol(replace(String(g.variables[pl.variables[r, i]].name), r"\[\d+\]$" => ""))
             for r in axes(pl.variables, 1))
    end
    groups = [replica_name(name, i) => [replica_name(f, i) for f in factors] for i in 1:pl.n]
    gc = _cluster(build(_copy_builder(g)), groups, params, renames)
    return as_plate(gc, plate, pl.n)
end

# `renames[k]`: graph variable name => the name inside cluster k (its channel), or `nothing`.
function _cluster(g::FactorGraph, groups, params, renames)
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
    renames === nothing || (nodes = [_renamed(c, m) for (c, m) in zip(nodes, renames)])
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
    freezes = [_cluster_freeze(g, fids) for fids in fidss]
    params === nothing || all(==(first(freezes)), freezes) || throw(ArgumentError(
        "cluster: tied cluster nodes (params = :$params) must freeze the same parameter paths"))
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
            factor!(b, names[k], nodes[k]; params = something(params, names[k]), freeze = freezes[k]); placed[k] = true
        end
    end
    for e in g.edges
        haskey(owner, e.factor) && continue
        connect!(b, g.factors[e.factor].name, e.channel, g.variables[e.variable].name;
                 direction = e.direction, coupling = e.coupling, detach = e.detach)
    end
    for (k, (name, node)) in enumerate(zip(names, nodes))
        inv = renames === nothing ? Dict{Symbol,Symbol}() : Dict(w => v for (v, w) in renames[k])
        for v in node.boundary
            connect!(b, name, v, get(inv, v, v))
        end
    end
    return build(b)
end

# The node with its variables renamed by `m` (graph name => name inside); others keep theirs.
function _renamed(c::ClusterFactor, m)
    r(v) = get(m, v, v)
    return ClusterFactor(c.names, c.keys, c.factors, [[ch => r(v) for (ch, v) in w] for w in c.wiring],
                         r.(c.variables), Dict(r(v) => k for (v, k) in c.dims), r.(c.discrete), r.(c.continuous), r.(c.boundary))
end

_internal_variables(g, fids) = (inside = Set(fids); vids = unique(g.edges[ei].variable for fid in fids for ei in g.edges_of_factor[fid]);
    Set(v for v in vids if all(ei -> g.edges[ei].factor in inside, g.edges_of_variable[v])))

# One group's node, after the checks of `cluster`.
# The cluster node's frozen paths: each clustered node's paths under its parameter key (the node's
# parameters are keyed by those keys), `(key,)` for a node frozen whole. Tied nodes share a key
# and agree on their freeze (`build` checks), so each key is read once.
function _cluster_freeze(g, fids)
    paths = Tuple[]
    seen = Set{Symbol}()
    for fid in fids
        f = g.factors[fid]
        f.params in seen && continue
        push!(seen, f.params)
        fr = LenticulumCore._freeze_paths(f.freeze)
        if fr === :all || LenticulumCore.isfrozen(f.factor)
            push!(paths, (f.params,))
        else
            for q in fr
                push!(paths, (f.params, q...))
            end
        end
    end
    return Tuple(paths)
end

function _cluster_node(g, fids, inside, union_, tied)
    for fid in fids
        f = g.factors[fid]
        f.rule isa BeliefPropagation || throw(ArgumentError("cluster: :$(f.name) is not a BeliefPropagation() node"))
        which(cluster_form, Tuple{typeof(f.factor),Any}) === which(cluster_form, Tuple{Any,Any}) && throw(ArgumentError(
            "cluster: :$(f.name) ($(nameof(typeof(f.factor)))) has no `Mycelium.cluster_form`; keep it outside the cluster"))
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
# substituted, the assignments of the discrete variables, one h and one constant per assignment,
# and the precision of each precision class (`J[class[a]]` for assignment a; one class unless a
# factor's A or Q depends on its outcome).
struct _ClusterJoint
    free::Vector{Symbol}
    offset::Dict{Symbol,UnitRange{Int}}
    points::Dict{Symbol,Vector{Float64}}
    assignments::Vector{Vector{Int}}           # per assignment: the state of each discrete variable
    J::Vector{Matrix{Float64}}                 # per precision class
    class::Vector{Int}                         # per assignment: its precision class
    H::Matrix{Float64}                         # n × nA
    c::Vector{Float64}                         # nA
end
_members(jt, κ) = findall(==(κ), jt.class)

_incoming(msgs, v, skip) = v === skip ? nothing : (haskey(msgs, v) ? getfield(msgs, v) : nothing)
_cluster_input(b) = b
_cluster_input(b::LenticulumCore.MixtureBelief{<:GaussianBelief}) = ep_project(b) === nothing ?
    throw(ArgumentError("a cluster node collapses a mixture input to its moments, which needs proper components")) : ep_project(b)

# The joint discrete assignments of nonzero weight and their log-weights (incoming messages and
# every factor's outcome weights). Built one variable at a time: a factor's log-weights are added
# as soon as all its switch channels are assigned, and partial assignments of weight zero (a -Inf,
# e.g. a `TableFactor` tying variables) are dropped there, so the impossible ones are never
# enumerated in full.
function _enumerate_assignments(c::ClusterFactor, forms, wirings, states, lps)
    nd = length(c.discrete)
    chans(F) = F.switch === nothing ? () : (F.switch isa Symbol ? (F.switch,) : F.switch)
    pos = [[findfirst(==(w[ch]), c.discrete) for ch in chans(F)] for (F, w) in zip(forms, wirings)]
    done = [isempty(p) ? 0 : maximum(p) for p in pos]
    partial = [Int[]]
    cst = [sum((F.logw[1] for (F, d) in zip(forms, done) if d == 0); init = 0.0)]
    for j in 1:nd
        np, nc = Vector{Vector{Int}}(), Float64[]
        for (a, ca) in zip(partial, cst), (t, k) in enumerate(states[j])
            push!(np, [a; k]); push!(nc, ca + lps[j][t])
        end
        for (F, p, d) in zip(forms, pos, done)
            d == j || continue
            L = LinearIndices(Tuple(c.dims[c.discrete[q]] for q in p))
            nc .+= F.logw[[L[(a[q] for q in p)...] for a in np]]
        end
        keep = findall(>(-Inf), nc)
        isempty(keep) && throw(ArgumentError("cluster: every joint discrete state has zero probability"))
        partial, cst = np[keep], nc[keep]
        length(partial) <= 2^16 || throw(ArgumentError("cluster: more than 2^16 joint discrete states of nonzero weight"))
    end
    return partial, cst
end

function _cluster_joint(c::ClusterFactor, msgs, @nospecialize(ps), skip)
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
            _check_positional(b, K, "cluster")
            ks = [k for k in 1:K if b.logp[k] > -Inf]
            isempty(ks) && throw(ArgumentError("cluster: the message to :$v gives every state zero probability"))
            push!(states, ks); push!(lps, b.logp[ks])
        else
            push!(states, collect(1:K)); push!(lps, zeros(K))
        end
    end
    forms = [cluster_form(f, _subtree(ps, c.keys[i])) for (i, f) in enumerate(c.factors)]
    wirings = [Dict(w) for w in c.wiring]
    assignments, cst = _enumerate_assignments(c, forms, wirings, states, lps)
    nA = length(assignments)
    kcols = [_switch_columns(c, F, w, assignments) for (F, w) in zip(forms, wirings)]
    class, keys_ = _precision_classes(forms, kcols, nA)
    J = zeros(n, n); H = zeros(n, nA)
    Js = [zeros(n, n) for _ in keys_]
    # incoming Gaussian messages on free boundary variables
    for v in free
        b = _cluster_input(_incoming(msgs, v, skip))
        b isa GaussianBelief || continue
        o = offset[v]
        J[o, o] .+= b.Λ
        H[o, :] .+= b.η
    end
    vf = 0
    for (F, w, kcol) in zip(forms, wirings, kcols)
        m = size(F.C, 1)
        m == 0 && continue
        varies = _varies(F)
        varies && (vf += 1)
        for k in (varies ? unique(kcol) : [0])
            idx = varies ? findall(==(k), kcol) : collect(1:nA)
            G = zeros(m, n)
            shift = zeros(m)
            for (ch, Ach) in pairs(F.A)
                v = w[ch]
                Ak = _at(Ach, k)
                haskey(points, v) ? (shift .+= Ak * points[v]) : (G[:, offset[v]] .+= Ak)
            end
            CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(_at(F.Q, k))))
            Cc = F.C[:, kcol[idx]] .- shift                 # r = G x − c_a
            QiC = CQ \ Cc
            GQG = G' * (CQ \ G)
            if varies                                       # into the classes with this outcome
                for (κ, key) in enumerate(keys_)
                    key[vf] == k && (Js[κ] .+= GQG)
                end
            else
                J .+= GQG
            end
            H[:, idx] .+= G' * QiC
            cst[idx] .+= -vec(sum(Cc .* QiC; dims = 1)) ./ 2 .- (logdet(CQ) + m * log(2π)) / 2
        end
    end
    Js = [(Jκ + J + (Jκ + J)') / 2 for Jκ in Js]
    return _ClusterJoint(free, offset, points, assignments, Js, class, H, cst)
end

# The precision class of each assignment: the outcomes it gives the factors whose A or Q depends
# on their outcome (`kcols[f][a]`, the column factor f reads), numbered in order of appearance.
function _precision_classes(forms, kcols, nA)
    vary = [i for (i, F) in enumerate(forms) if _varies(F)]
    keys_ = Vector{Vector{Int}}()
    index = Dict{Vector{Int},Int}()
    class = Vector{Int}(undef, nA)
    for a in 1:nA
        key = [kcols[i][a] for i in vary]
        class[a] = get!(index, key) do
            push!(keys_, key); length(keys_)
        end
    end
    isempty(keys_) && push!(keys_, Int[])
    return class, keys_
end

# log ∫ exp(hᵀx − xᵀJx/2) dx for each column of H, with J positive definite.
# A value computed once per joint precision: `chol` is `nothing` (no cache) or a `Dict`.
_cached(::Nothing, _, f) = f()
_cached(chol::AbstractDict, k, f) = get!(f, chol, k)

# --- Flat directions: the diffuse limit -------------------------------------------
#
# A precision J (positive semidefinite) is factorised by Cholesky when it is definite, and
# otherwise split into its range and its null space (`_Flat`): the directions no factor and no
# incoming message determines. Integrals over x then mean the limit of a vague prior N(0, s²I),
# s → ∞, with the factor (2πs²)^{k/2} of the k flat directions dropped: pseudo-inverse and
# pseudo-determinant on the range (`cluster.md`, "Flat directions"). The dropped factor is common
# to all assignments only if they leave equally many directions flat, which is checked
# (`_same_flatness`); a linear term reaching a flat direction makes the integral diverge and is
# refused (`_check_range`).
struct _Flat
    U::Matrix{Float64}          # an orthonormal basis of the range
    λ::Vector{Float64}          # J's eigenvalues on it
    N::Matrix{Float64}          # an orthonormal basis of the null space
end
const _FLAT_RTOL = 1e-10
# Flat means an eigenvalue at most _FLAT_RTOL times the largest. A Cholesky factorisation can
# succeed on a singular J (rounding leaves a tiny positive pivot), so its smallest pivot is checked
# against the same threshold, and the eigendecomposition decides when it is small.
function _psd_factor(J::AbstractMatrix)
    scale = maximum(abs, LinearAlgebra.diag(J); init = 0.0)
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(J); check = false)
    LinearAlgebra.issuccess(C) && minimum(abs2, LinearAlgebra.diag(C.U); init = Inf) > _FLAT_RTOL * scale && return C
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Matrix(J)))
    λmax = maximum(abs, E.values; init = 0.0)
    any(E.values .< -sqrt(eps()) * max(λmax, 1.0)) && throw(ArgumentError("cluster: an indefinite joint precision"))
    keep = E.values .> _FLAT_RTOL * λmax
    all(keep) && LinearAlgebra.issuccess(C) && return C
    return _Flat(E.vectors[:, keep], E.values[keep], E.vectors[:, .!keep])
end
# Whether a precision Λ (of a message or marginal, eliminated from a joint of diagonal scale
# `scale`) is definite: its Cholesky pivots above the flat threshold.
function _definite(Λ, scale)
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Λ); check = false)
    ok = LinearAlgebra.issuccess(C) && minimum(abs2, LinearAlgebra.diag(C.U); init = Inf) > _FLAT_RTOL * scale
    return ok, C
end
_psd_solve(C::LinearAlgebra.Cholesky, B) = C \ B
_psd_solve(F::_Flat, B) = F.U * ((F.U' * B) ./ F.λ)
_psd_logdet(C::LinearAlgebra.Cholesky) = logdet(C)
_psd_logdet(F::_Flat) = sum(log, F.λ; init = 0.0)
_psd_rank(C::LinearAlgebra.Cholesky) = size(C, 1)
_psd_rank(F::_Flat) = length(F.λ)
_psd_inv(C::LinearAlgebra.Cholesky) = Matrix(inv(C))
_psd_inv(F::_Flat) = F.U * LinearAlgebra.Diagonal(1 ./ F.λ) * F.U'
_check_range(::LinearAlgebra.Cholesky, _, _) = nothing
function _check_range(F::_Flat, H, what)
    isempty(H) && return nothing
    tol = sqrt(eps()) * max(maximum(abs, H), 1.0) * size(F.N, 1)
    maximum(abs, F.N' * H; init = 0.0) <= tol || throw(ArgumentError(
        "cluster: $what have a flat direction that a linear term reaches, so their integral diverges " *
        "(an incoming message or factor offset along a direction nothing determines)"))
    return nothing
end
_same_flatness(fs, what) = length(unique(_psd_rank.(fs))) <= 1 || throw(ArgumentError(
    "cluster: the assignments leave different numbers of directions of $what flat, so their weights " *
    "are not comparable in the diffuse limit (cluster.md, \"Flat directions\")"))

# log ∫ exp(hᵀx − xᵀJ_κx/2) dx for each assignment (column of H), J_κ its class's precision; in
# the diffuse limit if J_κ is singular. Factorisations are cached per class.
function _log_integrals(jt::_ClusterJoint, what, chol = nothing)
    n = size(jt.H, 1)
    n == 0 && return zeros(size(jt.H, 2))
    out = zeros(size(jt.H, 2))
    fs = [_cached(chol, (:CJ, κ), () -> _psd_factor(jt.J[κ])) for κ in eachindex(jt.J)]
    _same_flatness(fs, what)
    for κ in eachindex(jt.J)
        idx = _members(jt, κ)
        C = fs[κ]
        Hκ = jt.H[:, idx]
        _check_range(C, Hκ, what)
        out[idx] .= vec(sum(Hκ .* _psd_solve(C, Hκ); dims = 1)) ./ 2 .- _psd_logdet(C) / 2 .+ _psd_rank(C) * log(2π) / 2
    end
    return out
end

_lse(v) = (m = maximum(v); m == -Inf ? -Inf : m + log(sum(exp.(v .- m))))

"""
    Mycelium.factor_message(c::ClusterFactor, target, polarity, inputs, prior, ps, st)

The exact message of the cluster to one of its channels, given the others' messages: every joint
assignment of the discrete variables is enumerated, and the continuous variables other than the
target are eliminated from that assignment's Gaussian. To a continuous channel the message is a
mixture over the assignments (a `GaussianBelief` if there is one); its components have one
precision per precision class; an improper component is the unnormalised exponential its weight
multiplies. To a discrete channel a `CategoricalBelief`. Points are substituted; a
Gaussian-mixture input is collapsed to its moments. Directions that nothing determines (a channel
without a message, say) are integrated out in the diffuse limit (`cluster.md`, "Flat directions").
"""
function factor_message(c::ClusterFactor, target::Symbol, polarity, inputs, prior, @nospecialize(ps), @nospecialize(st))
    target in c.boundary || throw(ArgumentError("channel :$target is not a channel of this cluster (has $(c.boundary))"))
    jt = _cluster_joint(c, inputs, ps, target)
    return _joint_message(c, jt, target, nothing), st
end

# The message to `target` from a joint. `chol` caches the factorisations, which depend on the
# classes' precisions only: `nothing`, or a `Dict` filled on first use (tied replicas sharing them
# reuse it, `cluster_batch.jl`).
function _joint_message(c::ClusterFactor, jt::_ClusterJoint, target::Symbol, chol)
    if target in c.discrete
        l = jt.c .+ _log_integrals(jt, "its continuous variables", chol)
        j = findfirst(==(target), c.discrete)
        K = c.dims[target]
        lk = [_lse([l[a] for a in eachindex(l) if jt.assignments[a][j] == k]; ) for k in 1:K]
        all(==(-Inf), lk) && throw(ArgumentError("cluster: every state of :$target has zero probability"))
        return LenticulumCore.CategoricalBelief(Val(:log), lk .- _lse(lk), nothing)
    end
    ot = jt.offset[target]
    rs = setdiff(1:size(jt.H, 1), ot)
    d = length(ot)
    nA = size(jt.H, 2)
    E = zeros(d, nA); l = copy(jt.c)
    Λs = Matrix{Float64}[]; proper = Bool[]
    isempty(rs) || _same_flatness([_cached(chol, (:Cr, κ), () -> _psd_factor(jt.J[κ][rs, rs])) for κ in eachindex(jt.J)],
                                  "the continuous variables other than :$target")
    for κ in eachindex(jt.J)
        idx = _members(jt, κ)
        Jκ = jt.J[κ]
        Jtt = Jκ[ot, ot]
        Hr, Ht = jt.H[rs, idx], jt.H[ot, idx]
        if isempty(rs)
            Λ = Jtt
            E[:, idx] .= Ht
        else
            Jtr = Jκ[ot, rs]
            Cr = _cached(chol, (:Cr, κ), () -> _psd_factor(Jκ[rs, rs]))
            _check_range(Cr, Hr, "the continuous variables other than :$target")
            X = _psd_solve(Cr, Hr)
            Λ = _cached(chol, (:Λ, κ), () -> Jtt - Jtr * _psd_solve(Cr, Jtr'))
            E[:, idx] .= Ht - Jtr * X
            l[idx] .+= vec(sum(Hr .* X; dims = 1)) ./ 2 .- _psd_logdet(Cr) / 2 .+ _psd_rank(Cr) * log(2π) / 2
        end
        Λ = (Λ + Λ') / 2
        ok, CΛ = _cached(chol, (:CΛ, κ), () -> _definite(Λ, maximum(abs, LinearAlgebra.diag(Jκ); init = 0.0)))
        if ok                                   # normalised components: their mass into the weight
            Eκ = E[:, idx]
            l[idx] .+= vec(sum(Eκ .* (CΛ \ Eκ); dims = 1)) ./ 2 .- logdet(CΛ) / 2 .+ d * log(2π) / 2
        end
        push!(Λs, Λ); push!(proper, ok)
    end
    keep = findall(>(-Inf), l)
    isempty(keep) && throw(ArgumentError("cluster: every assignment has zero probability"))
    length(Λs) == 1 && all(iszero, Λs[1]) && all(iszero, E) && return LenticulumCore.TrivialBelief()
    # assignments the target does not see (internal discrete variables that do not move it) give
    # equal components; they are merged, their weights summed
    groups = Dict{Tuple{Int,Vector{Float64}},Vector{Int}}()
    order = Vector{Tuple{Int,Vector{Float64}}}()
    for a in keep
        key = (findfirst(==(Λs[jt.class[a]]), Λs), E[:, a])  # equal precisions count as one
        haskey(groups, key) || (groups[key] = Int[]; push!(order, key))
        push!(groups[key], a)
    end
    comp((κ, col)) = GaussianBelief(col, Λs[κ], !proper[κ])
    length(order) == 1 && return comp(only(order))
    lk = [_lse(l[groups[key]]) for key in order]
    return LenticulumCore.MixtureBelief(Val(:log), [comp(key) for key in order], lk .- _lse(lk))
end

# --- The free energy --------------------------------------------------------------

"""
    FrozenClusterBelief

The cluster's belief ``b(x, a) \\propto \\prod_f f(x, a) \\prod_v \\mu_v``, held fixed: the posterior `q` over
the discrete assignments, the means of the free continuous variables per assignment (columns of
`M`), their covariance per precision class (`Σ[class[a]]`) and its conditional entropy
``H(x \\mid a)`` (`H[class[a]]`), the substituted points, and the assignments.
"""
struct FrozenClusterBelief{Q<:AbstractVector,M<:AbstractMatrix,S<:AbstractVector,H<:AbstractVector}
    q::Q
    M::M
    Σ::S
    H::H
    class::Vector{Int}
    free::Vector{Symbol}
    offset::Dict{Symbol,UnitRange{Int}}
    points::Dict{Symbol,Vector{Float64}}
    assignments::Vector{Vector{Int}}
end

factor_belief(c::ClusterFactor, msgs, @nospecialize(ps), @nospecialize(st)) = _joint_belief(_cluster_joint(c, msgs, ps, nothing), nothing)

function _joint_belief(jt::_ClusterJoint, chol)
    n = size(jt.H, 1)
    l = jt.c .+ _log_integrals(jt, "its continuous variables", chol)
    all(==(-Inf), l) && throw(ArgumentError("cluster: every assignment has zero probability"))
    q = exp.(l .- _lse(l))
    M = zeros(n, length(q))
    Σs = Matrix{Float64}[]; Hs = Float64[]
    for κ in eachindex(jt.J)
        if n == 0
            push!(Σs, zeros(0, 0)); push!(Hs, 0.0)
            continue
        end
        C = _cached(chol, (:CJ, κ), () -> _psd_factor(jt.J[κ]))
        idx = _members(jt, κ)
        M[:, idx] .= _psd_solve(C, jt.H[:, idx])
        # on the range of J_κ in the diffuse limit: the flat directions have mean 0 and drop out
        Σ, H = _cached(chol, (:ΣH, κ), () -> (_psd_inv(C), _psd_rank(C) * (1 + log(2π)) / 2 - _psd_logdet(C) / 2))
        push!(Σs, Σ); push!(Hs, H)
    end
    return FrozenClusterBelief(q, M, Σs, Hs, jt.class, jt.free, jt.offset, jt.points, jt.assignments)
end

"""
    Mycelium.local_free_energy(c::ClusterFactor, msgs, ps, st)
    Mycelium.belief_free_energy(c::ClusterFactor, b::FrozenClusterBelief, ps, st)

``\\mathbb E_b[-\\log \\prod_f f] - H(b)`` for the cluster's exact belief, graded as `(fit, complexity,
negentropy)`: `fit` is ``\\sum_f \\sum_a q_a \\tfrac12 \\mathbb E[r_f^\\top Q_{f,a}^{-1} r_f \\mid a]``, `complexity`
``\\sum_f \\sum_a q_a (\\tfrac12 \\log\\det 2\\pi Q_{f,a} - \\log w_{f,a})``, and `negentropy`
``-H(q) - \\sum_a q_a H(x \\mid a)``, closed form since given the assignment the continuous part is
one Gaussian. `belief_free_energy` re-evaluates the factors' terms at `ps` with the belief fixed
(the parameter gradient, LP-020).
"""
local_free_energy(c::ClusterFactor, msgs, ps, st) = belief_free_energy(c, factor_belief(c, msgs, ps, st), ps, st)

# Which outcome column each assignment reads, the groups of assignments that read one outcome's
# A and Q (one group if they do not depend on the outcome) with the belief's weight on each
# precision class within the group, which channels feed each free variable, and which channels
# hold points: structure only, computed outside AD (`@non_differentiable` in the ChainRulesCore
# extension).
function _cluster_layout(c::ClusterFactor, i::Int, F::ClusterForm, b::FrozenClusterBelief)
    w = Dict(c.wiring[i])
    kcol = _switch_columns(c, F, w, b.assignments)
    groups = map(_varies(F) ? unique(kcol) : [0]) do k
        idx = _varies(F) ? findall(==(k), kcol) : collect(eachindex(kcol))
        (k, idx, [sum(b.q[a] for a in idx if b.class[a] == κ; init = 0.0) for κ in eachindex(b.Σ)])
    end
    feeds = [[ch for ch in keys(F.A) if w[ch] === v] for v in b.free]
    pts = [(ch, b.points[w[ch]]) for ch in keys(F.A) if haskey(b.points, w[ch])]
    return kcol, groups, feeds, pts
end

_finite_logw(lw) = map(x -> x == -Inf ? zero(x) : x, lw)

function belief_free_energy(c::ClusterFactor, b::FrozenClusterBelief, ps, st)
    n = size(b.M, 1)
    terms = map(eachindex(c.factors)) do i
        F = cluster_form(c.factors[i], _subtree(ps, c.keys[i]))
        kcol, groups, feeds, pts = _cluster_layout(c, i, F, b)
        # an impossible outcome (log-weight -Inf) has q = 0 exactly, and its 0·log 0 term is 0; the
        # entry is replaced before the product so that no AD backend meets 0 · (-Inf)
        cx = -sum(b.q .* _finite_logw(F.logw[kcol]))
        m = size(F.C, 1)
        m == 0 && return (0.0, cx)
        parts = map(groups) do (k, idx, wκ)
            R = -F.C[:, kcol[idx]]
            for (ch, x) in pts
                R = R .+ _at(getfield(F.A, ch), k) * x
            end
            CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(_at(F.Q, k))))
            tr = 0.0
            if n > 0
                G = reduce(hcat, [isempty(chs) ? zeros(m, length(b.offset[v])) : reduce(+, [_at(getfield(F.A, ch), k) for ch in chs])
                                  for (v, chs) in zip(b.free, feeds)])
                R = R .+ G * b.M[:, idx]
                tr = sum(map(κ -> wκ[κ] * LinearAlgebra.tr(CQ \ (G * b.Σ[κ] * G')), eachindex(wκ)))
            end
            quad = vec(sum(R .* (CQ \ R); dims = 1))
            ((sum(b.q[idx] .* quad) + tr) / 2, sum(wκ) * (logdet(CQ) + m * log(2π)) / 2)
        end
        return (sum(first, parts), cx + sum(last, parts))
    end
    fit, complexity = sum(first, terms), sum(last, terms)
    Hq = -sum(ifelse.(b.q .> 0, b.q .* log.(max.(b.q, floatmin(Float64))), 0.0))
    Hx = sum(b.q .* b.H[b.class])
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -Hq - Hx)), st
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
    jt = _cluster_joint(c, messages_into(s, gc, fid), _subtree(ps, factornode(gc, fid).params), nothing)
    chol = Dict{Any,Any}()
    b = _joint_belief(jt, chol)
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
    rs = setdiff(1:size(jt.H, 1), o)
    comp = map(eachindex(b.Σ)) do κ
        chol[(:CJ, κ)] isa _Flat || return a -> LenticulumCore.Gaussian(b.M[o, a], Matrix(LinearAlgebra.Symmetric(b.Σ[κ][o, o])))
        # a flat class: v's marginal in information form (the other variables eliminated in the
        # diffuse limit), improper if v itself has a flat direction
        J = jt.J[κ]
        Cr = isempty(rs) ? nothing : _psd_factor(J[rs, rs])
        Λ = isempty(rs) ? J[o, o] : J[o, o] - J[o, rs] * _psd_solve(Cr, J[rs, o])
        Λ = (Λ + Λ') / 2
        proper = first(_definite(Λ, maximum(abs, LinearAlgebra.diag(J); init = 0.0)))
        a -> GaussianBelief(isempty(rs) ? jt.H[o, a] : jt.H[o, a] - J[o, rs] * _psd_solve(Cr, jt.H[rs, a]), Λ, !proper)
    end
    comps = [comp[b.class[a]](a) for a in eachindex(b.q) if b.q[a] > 0]
    length(comps) == 1 && return only(comps)
    return LenticulumCore.MixtureBelief(comps, b.q[b.q .> 0])
end
