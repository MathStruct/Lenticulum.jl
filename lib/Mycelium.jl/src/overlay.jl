# ---------------------------------------------------------------------------
# Overlaid factors: several factors on the same variables, each with its own parameters, as one
# node (Feature Backlog "mirrored factors", RQ-008).
#
# The overlay is the product of its members, so its energy is the sum of theirs. For members stated
# by a `cluster_form` the sum is again one: the members' residuals stacked, their noises block
# diagonal, their outcomes the joint outcome of all their switches, their log-weights added. With
# `normalise = (:y, …)` the overlay is instead, for each joint outcome, the product of its members
# normalised as a density of the channels `y` given the others: a conditional Gaussian with the
# members' precisions on `y` added (a product of experts), and the outcomes a mixture over it.
#
# See `overlay.md`.
# ---------------------------------------------------------------------------

"""
    OverlayFactor(:a => f₁, :b => f₂, …; normalise = nothing)

Factors overlaid on the same variables, as one node: the product ``\\prod_i f_i``, each member with its
own parameters (the overlay's parameters are a `NamedTuple` keyed by the members' names). The
channels are the union of the members' channels; two members with a channel of one name share that
variable, a switch channel included. Members must have a [`cluster_form`](@ref).

With `normalise = (:y, …)` (continuous channels) the overlay is, for each joint outcome ``k`` of the
members' switches, the product normalised as a density of ``y`` given the other channels:

```math
f(y, x, k) = w_k\\, \\mathcal N\\bigl(y;\\ P_k^{-1} h_k(x),\\ P_k^{-1}\\bigr), \\qquad P_k = \\textstyle\\sum_i A_{y,i}^\\top Q_i^{-1} A_{y,i},
```

with ``w_k`` the product of the members' weights. This is a mixture of experts normalised in ``y``: it
cannot satisfy observed ``y`` by collapsing (LP-022), since ``P_k`` must be invertible and a common
scaling of the coefficients and the noise leaves it unchanged. Members that do not involve ``y``
cancel in the normalisation.

Inside a [`cluster`](@ref) node the overlay is its `cluster_form`; outside one, its messages and free
energy are those of a one-factor cluster ([`as_cluster`](@ref)). See `overlay.md`.
"""
struct OverlayFactor{F<:Tuple} <: LenticulumCore.AbstractLenticulumFactor
    names::Tuple{Vararg{Symbol}}
    members::F
    normalise::Union{Nothing,Tuple{Vararg{Symbol}}}
    channels::Vector{Pair{Symbol,Int}}       # channel => dimension (continuous) or states (discrete), in order
    switches::Vector{Tuple{Vararg{Symbol}}}  # per member: its switch channels
end

function OverlayFactor(members::Pair{Symbol}...; normalise = nothing)
    isempty(members) && throw(ArgumentError("an overlay needs at least one member"))
    names = Tuple(first.(members)); fs = Tuple(last.(members))
    allunique(names) || throw(ArgumentError("overlay members need distinct names; got $names"))
    chans = Pair{Symbol,Int}[]
    for f in fs, ch in LenticulumCore.channels(f)
        n, d = LenticulumCore.channelname(ch), Int(LenticulumCore.channelspace(ch))
        k = findfirst(p -> p.first === n, chans)
        k === nothing ? push!(chans, n => d) : (chans[k].second == d || throw(DimensionMismatch(
            "overlay: channel :$n has dimension $(chans[k].second) in one member and $d in another")))
    end
    sws = map(fs) do f
        s = cluster_switch(f)
        s === nothing ? () : (s isa Symbol ? (s,) : Tuple(s))
    end
    nz = normalise === nothing ? nothing : (normalise isa Symbol ? (normalise,) : Tuple(normalise))
    if nz !== nothing
        allsw = reduce(vcat, collect.(sws); init = Symbol[])
        for y in nz
            any(p -> p.first === y, chans) || throw(ArgumentError("overlay: normalise channel :$y is not a channel"))
            y in allsw && throw(ArgumentError("overlay: normalise channel :$y is discrete"))
        end
    end
    return OverlayFactor(names, fs, nz, chans, Tuple{Vararg{Symbol}}[s for s in sws])
end

LenticulumCore.channels(o::OverlayFactor) = Tuple(LenticulumCore.Channel(n, d) for (n, d) in o.channels)
function LenticulumCore.supported_polarities(o::OverlayFactor)
    names = Tuple(first.(o.channels))
    out = LenticulumCore.Polarity[]
    for t in eachindex(names), rest in Iterators.product(ntuple(_ -> (LenticulumCore.Observed(), LenticulumCore.Latent()), length(names) - 1)...)
        vals = Any[rest...]
        insert!(vals, t, LenticulumCore.Unobserved())
        push!(out, _polarity(names, Tuple(vals)))
    end
    return Tuple(out)
end
LenticulumCore.supports_polarity(o::OverlayFactor, p::LenticulumCore.Polarity) =
    Set(keys(p)) == Set(first.(o.channels)) && length(LenticulumCore.unobserved_channels(p)) == 1
LenticulumCore.hasnopolarity(::OverlayFactor) = false
LenticulumCore.isunidirectional(::OverlayFactor) = false
LenticulumCore.islearnable(o::OverlayFactor) = any(LenticulumCore.islearnable, o.members)
LenticulumCore.energyspace(::OverlayFactor) = LenticulumCore.GradedEnergySpace((
    fit = LenticulumCore.ScalarEnergySpace(), complexity = LenticulumCore.ScalarEnergySpace(),
    negentropy = LenticulumCore.ScalarEnergySpace()))
LenticulumCore.scalarisation(::OverlayFactor) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(), complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation()))
LuxCore.initialparameters(rng::AbstractRNG, o::OverlayFactor) =
    NamedTuple{o.names}(map(f -> LuxCore.initialparameters(rng, f), o.members))
LuxCore.initialstates(rng::AbstractRNG, o::OverlayFactor) =
    NamedTuple{o.names}(map(f -> LuxCore.initialstates(rng, f), o.members))
uses_cavity(::OverlayFactor) = true

function cluster_switch(o::OverlayFactor)
    allsw = Tuple(reduce(vcat, collect.(o.switches); init = Symbol[]))
    return isempty(allsw) ? nothing : allsw
end

# The members' outcome (column) for each joint outcome j of the overlay's switch channels.
function _overlay_columns(o::OverlayFactor)
    dims = Dict(o.channels)
    allsw = reduce(vcat, collect.(o.switches); init = Symbol[])
    K = prod(dims[s] for s in allsw; init = 1)
    pos = Int[]; offs = Vector{UnitRange{Int}}()
    for sw in o.switches
        push!(offs, (length(pos) + 1):(length(pos) + length(sw)))
        append!(pos, 1:length(sw))
    end
    C = CartesianIndices(Tuple(dims[s] for s in allsw))
    return K, [map(enumerate(o.switches)) do (i, sw)
        isempty(sw) && return 1
        st = Tuple(C[j])[offs[i]]
        LinearIndices(Tuple(dims[s] for s in sw))[st...]
    end for j in 1:K]
end

_blockdiag(Qs) = isempty(Qs) ? zeros(0, 0) : cat(Qs...; dims = (1, 2))

function cluster_form(o::OverlayFactor, ps)
    forms = [cluster_form(f, getfield(ps, n)) for (n, f) in zip(o.names, o.members)]
    K, cols = _overlay_columns(o)
    allsw = reduce(vcat, collect.(o.switches); init = Symbol[])
    cont = [(n, d) for (n, d) in o.channels if !(n in allsw)]
    varies = any(_varies, forms)
    # the stacked residual of joint outcome j: rows of every member with a continuous part
    stacked(j) = begin
        ks = cols[j]
        rows = [(F, k) for (F, k) in zip(forms, ks) if size(F.C, 1) > 0]
        A = NamedTuple{Tuple(first.(cont))}(Tuple(isempty(rows) ? zeros(0, d) :
            reduce(vcat, [haskey(F.A, n) ? _at(getfield(F.A, n), k) : zeros(size(F.C, 1), d) for (F, k) in rows]) for (n, d) in cont))
        C = isempty(rows) ? zeros(0) : reduce(vcat, [F.C[:, k] for (F, k) in rows])
        Q = _blockdiag([Matrix(_at(F.Q, k)) for (F, k) in rows])
        lw = sum(F.logw[k] for (F, k) in zip(forms, ks); init = 0.0)
        (A, C, Q, lw)
    end
    parts = [stacked(j) for j in 1:K]
    sw = isempty(allsw) ? nothing : Tuple(allsw)
    logw = [p[4] for p in parts]
    if o.normalise === nothing
        A1, C1, Q1, _ = parts[1]
        A = varies ? map(n -> cat((getfield(p[1], n) for p in parts)...; dims = 3), NamedTuple{keys(A1)}(keys(A1))) : A1
        Cm = reduce(hcat, [p[2] for p in parts])
        Q = varies ? cat((p[3] for p in parts)...; dims = 3) : Q1
        return ClusterForm(A, Cm, Q, logw, sw)
    end
    # normalised in y: per joint outcome, N(y; P⁻¹ A_yᵀQ⁻¹(C − A_x x), P⁻¹), P = A_yᵀQ⁻¹A_y
    ys = o.normalise
    xs = Tuple(n for (n, _) in cont if !(n in ys))
    normed(p) = begin
        A, C, Q, _ = p
        AY = reduce(hcat, [getfield(A, y) for y in ys])
        CQ = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Q))
        P = LinearAlgebra.Symmetric(AY' * (CQ \ AY))
        CP = LinearAlgebra.cholesky(P; check = false)
        LinearAlgebra.issuccess(CP) || throw(ArgumentError(
            "overlay: the members do not determine $(ys) (A_yᵀQ⁻¹A_y is singular), so the overlay cannot be normalised in it"))
        G = CP \ (AY' * (CQ \ Matrix(1.0LinearAlgebra.I, size(Q, 1), size(Q, 1))))   # P⁻¹ A_yᵀ Q⁻¹
        dy = size(AY, 2)
        offs = cumsum([0; [size(getfield(A, y), 2) for y in ys]])
        Ay = [Matrix(1.0LinearAlgebra.I, dy, dy)[:, (offs[i] + 1):offs[i + 1]] for i in eachindex(ys)]
        Anew = NamedTuple{(ys..., xs...)}((Ay..., (G * getfield(A, x) for x in xs)...))
        (Anew, G * C, Matrix(inv(CP)))
    end
    nparts = [normed(p) for p in parts]
    A1 = nparts[1][1]
    A = varies ? map(n -> cat((getfield(p[1], n) for p in nparts)...; dims = 3), NamedTuple{keys(A1)}(keys(A1))) : A1
    Cm = reduce(hcat, [p[2] for p in nparts])
    Q = varies ? cat((p[3] for p in nparts)...; dims = 3) : nparts[1][3]
    return ClusterForm(A, Cm, Q, logw, sw)
end

# Outside a cluster: a one-factor cluster over the overlay's channels.
factor_message(o::OverlayFactor, target::Symbol, polarity, inputs, prior, ps, st) =
    factor_message(as_cluster(o), target, polarity, inputs, prior, (factor = ps,), st)
factor_belief(o::OverlayFactor, msgs, ps, st) = factor_belief(as_cluster(o), msgs, (factor = ps,), st)
belief_free_energy(o::OverlayFactor, b::FrozenClusterBelief, ps, st) = belief_free_energy(as_cluster(o), b, (factor = ps,), st)
local_free_energy(o::OverlayFactor, msgs, ps, st) = belief_free_energy(o, factor_belief(o, msgs, ps, st), ps, st)

# Unnormalised, the overlay collapses where its members do; normalised in its data channels it
# is a conditional density there (LP-022).
function learns_unnormalised(o::OverlayFactor, observed)
    o.normalise === nothing && return any(f -> learns_unnormalised(f, observed), o.members)
    return false
end
