# ---------------------------------------------------------------------------
# Message blocks: the messages of one template edge of a plate, for all replicas at once
# (L-ADR-0010).
#
# A plate is unrolled, so every replica's edge is an ordinary edge. What a block adds is the
# storage: the edges of one template position ("a block row") can hold one `BeliefBlock` per
# direction instead of one `Message` each, such as the information vectors of all replicas as a
# d × n matrix with one shared precision. Plate sweeps read and write whole blocks; every other
# reader indexes an edge as before and receives a view of its column (`EdgeMessages`). Writing a
# single edge of a row held as a block first turns the row back into per-edge messages.
#
# See `blocks.md`.
# ---------------------------------------------------------------------------

"""
    BeliefBlock

The messages of `n` replicas on one template edge, as one value. Column `i` is replica `i`'s
belief, read with [`replica`](@ref). Implemented:

| block | holds | column `i` |
|---|---|---|
| `TrivialBlock(n)` | nothing | `TrivialBelief()` |
| `PointBlock(X)` | points as the columns of a `d × n` matrix | `DiracBelief(X[:, i])` |
| `GaussianBlock(η, Λ, improper)` | information vectors `η` (`d × n`), precisions `Λ` | `GaussianBelief(η[:, i], Λᵢ, improper[i])` |
| `VectorBlock(beliefs)` | any beliefs, one per replica | `beliefs[i]` |

`Λ` of a `GaussianBlock` is one matrix shared by every replica, or a vector of `n` matrices.
Build a block from per-replica beliefs with [`belief_block`](@ref).
"""
abstract type BeliefBlock end

struct TrivialBlock <: BeliefBlock
    n::Int
end
struct PointBlock{M<:AbstractMatrix} <: BeliefBlock
    X::M
end
struct GaussianBlock{E<:AbstractMatrix,L} <: BeliefBlock
    η::E
    Λ::L                      # an AbstractMatrix (shared) or an AbstractVector of them
    improper::Vector{Bool}
    function GaussianBlock(η::E, Λ::L, improper::AbstractVector{Bool}) where {E<:AbstractMatrix,L}
        length(improper) == size(η, 2) || throw(DimensionMismatch("$(size(η, 2)) replicas but $(length(improper)) flags"))
        Λ isa AbstractMatrix || length(Λ) == size(η, 2) ||
            throw(DimensionMismatch("$(size(η, 2)) replicas but $(length(Λ)) precisions"))
        return new{E,L}(η, Λ, collect(Bool, improper))
    end
end
struct VectorBlock <: BeliefBlock
    beliefs::Vector{Any}
end

"""
    nreplicas(block) -> Int

The number of replicas in a [`BeliefBlock`](@ref).
"""
nreplicas(b::TrivialBlock) = b.n
nreplicas(b::PointBlock) = size(b.X, 2)
nreplicas(b::GaussianBlock) = size(b.η, 2)
nreplicas(b::VectorBlock) = length(b.beliefs)

"""
    replica(block, i) -> AbstractBelief
    replicas(block) -> Vector{Any}

Replica `i`'s belief in a [`BeliefBlock`](@ref) (a view built on demand: a new belief holding
column `i`), and all of them.
"""
function replica end
replica(::TrivialBlock, i::Int) = LenticulumCore.TrivialBelief()
replica(b::PointBlock, i::Int) = LenticulumCore.DiracBelief(b.X[:, i])
replica(b::GaussianBlock, i::Int) = GaussianBelief(b.η[:, i], _precision(b, i), b.improper[i])
replica(b::VectorBlock, i::Int) = b.beliefs[i]

_shared(b::GaussianBlock) = b.Λ isa AbstractMatrix
_precision(b::GaussianBlock, i) = _shared(b) ? b.Λ : b.Λ[i]
replicas(b::BeliefBlock) = Any[replica(b, i) for i in 1:nreplicas(b)]

"""
    belief_block(beliefs::AbstractVector) -> BeliefBlock

The block of per-replica beliefs: a `TrivialBlock` when all are trivial, a `PointBlock` when all
are points that are real vectors of one length, a `GaussianBlock` when all are Gaussians of one
dimension (with a shared precision when every belief holds the same precision object), and a
`VectorBlock` otherwise. [`replica`](@ref) of the result returns beliefs equal to the inputs.
"""
function belief_block(bs::AbstractVector)
    n = length(bs)
    n == 0 && return VectorBlock(Any[])
    all(b -> b isa LenticulumCore.TrivialBelief, bs) && return TrivialBlock(n)
    b1 = first(bs)
    if b1 isa LenticulumCore.DiracBelief && b1.value isa AbstractVector{<:Real}
        d = length(b1.value)
        all(b -> b isa LenticulumCore.DiracBelief && b.value isa AbstractVector{<:Real} && length(b.value) == d, bs) &&
            return PointBlock(reduce(hcat, [b.value for b in bs]))
    elseif b1 isa GaussianBelief
        d = dimension(b1)
        if all(b -> b isa GaussianBelief && dimension(b) == d, bs)
            η = reduce(hcat, [b.η for b in bs])
            Λ = all(b -> b.Λ === b1.Λ, bs) ? b1.Λ : [b.Λ for b in bs]
            return GaussianBlock(η, Λ, [b.improper for b in bs])
        end
    end
    return VectorBlock(Any[b for b in bs])
end

# --- Block operations, column for column as the per-belief rules ------------------

# `combine` of each column. Gaussians add canonical parameters, a shared precision staying shared
# when both are; a point dominates a Gaussian; the unit is the trivial block. Anything else is
# combined column by column, with the per-belief rule and its errors.
combine_blocks(::TrivialBlock, b::BeliefBlock) = b
combine_blocks(a::BeliefBlock, ::TrivialBlock) = a
combine_blocks(a::TrivialBlock, ::TrivialBlock) = a
combine_blocks(a::PointBlock, ::GaussianBlock) = a
combine_blocks(::GaussianBlock, b::PointBlock) = b
function combine_blocks(a::PointBlock, b::PointBlock)
    isequal(a.X, b.X) && return a
    return _combine_columns(a, b)                      # throws: contradictory clamps
end
function combine_blocks(a::GaussianBlock, b::GaussianBlock)
    size(a.η, 1) == size(b.η, 1) || throw(DimensionMismatch(
        "cannot combine beliefs of dimension $(size(a.η, 1)) and $(size(b.η, 1))"))
    n = nreplicas(a)
    if _shared(a) && _shared(b)
        Λ = a.Λ + b.Λ
        za, zb = all(iszero, a.Λ), all(iszero, b.Λ)
        improper = [(a.improper[i] && zb) || (b.improper[i] && za) for i in 1:n]
    else
        Λ = [_precision(a, i) + _precision(b, i) for i in 1:n]
        improper = [(a.improper[i] && all(iszero, _precision(b, i))) ||
                    (b.improper[i] && all(iszero, _precision(a, i))) for i in 1:n]
    end
    return GaussianBlock(a.η + b.η, Λ, improper)
end
combine_blocks(a::BeliefBlock, b::BeliefBlock) = _combine_columns(a, b)
_combine_columns(a, b) = belief_block([combine(replica(a, i), replica(b, i)) for i in 1:nreplicas(a)])

# The largest `belief_distance` over the columns; `Inf` against a missing block.
block_distance(::Nothing, ::BeliefBlock) = Inf
block_distance(::TrivialBlock, ::TrivialBlock) = 0.0
function block_distance(a::GaussianBlock, b::GaussianBlock)
    size(a.η) == size(b.η) || return Inf
    dη = _maxabsdiff(a.η, b.η)
    dΛ = _shared(a) && _shared(b) ? _maxabsdiff(a.Λ, b.Λ) :
         maximum(i -> _maxabsdiff(_precision(a, i), _precision(b, i)), 1:nreplicas(a); init = 0.0)
    return max(dη, dΛ)
end
function block_distance(a::PointBlock, b::PointBlock)
    size(a.X) == size(b.X) || return Inf
    a.X === b.X && return 0.0
    d = 0.0
    for i in axes(a.X, 2)
        s = 0.0
        for k in axes(a.X, 1)
            s += abs2(a.X[k, i] - b.X[k, i])
        end
        d = max(d, sqrt(s))        # `belief_distance` of two points: 0 when equal, else the norm
    end
    return d
end
function _maxabsdiff(x::AbstractArray, y::AbstractArray)
    x === y && return 0.0
    size(x) == size(y) || return Inf
    d = 0.0
    for (u, v) in zip(x, y)
        d = max(d, abs(u - v))
    end
    return d
end
function block_distance(a::BeliefBlock, b::BeliefBlock)
    nreplicas(a) == nreplicas(b) || return Inf
    return maximum(i -> belief_distance(replica(a, i), replica(b, i)), 1:nreplicas(a); init = 0.0)
end

# Whether every column is uninformative (`true`), none is (`false`), or some are (`missing`).
_uninformative(::TrivialBlock) = true
_uninformative(::PointBlock) = false
function _uninformative(b::GaussianBlock)       # isuninformative: Λᵢ = 0 and ηᵢ = 0
    _shared(b) && !all(iszero, b.Λ) && return false
    flags = [all(iszero, _precision(b, i)) && all(iszero, view(b.η, :, i)) for i in 1:nreplicas(b)]
    return all(flags) ? true : any(flags) ? missing : false
end
function _uninformative(b::BeliefBlock)
    flags = [LenticulumCore.isuninformative(replica(b, i)) for i in 1:nreplicas(b)]
    return all(flags) ? true : any(flags) ? missing : false
end

# --- Edge storage with block rows ---------------------------------------------------

"""
    EdgeMessages

The messages of one direction, one per edge: what `store.to_variable` and `store.to_factor` are.
Indexing returns the edge's `Message` or `nothing`, as a `Vector{Any}` would. The edges of a
plate's template position form a *block row*; a row can hold one `Message` whose belief is a
[`BeliefBlock`](@ref), and an edge of that row then reads as a view, its column. Assigning to one
edge of a row held as a block first writes every column back as a per-edge message.
"""
mutable struct EdgeMessages <: AbstractVector{Any}
    msgs::Vector{Any}
    row::Vector{Int}                  # the block row of each edge (0: none)
    col::Vector{Int}                  # the replica of each edge in its row
    rows::Vector{Vector{Int}}         # the edges of each row, in replica order
    blocks::Vector{Any}               # per row: `nothing`, or a Message holding a BeliefBlock
end
EdgeMessages(msgs::AbstractVector) = EdgeMessages(Vector{Any}(msgs), zeros(Int, length(msgs)), zeros(Int, length(msgs)),
                                                  Vector{Int}[], Any[])
function EdgeMessages(n::Int, rows::Vector{Vector{Int}})
    row, col = zeros(Int, n), zeros(Int, n)
    for (r, es) in enumerate(rows), (i, e) in enumerate(es)
        row[e] == 0 || throw(ArgumentError("edge $e is in two block rows"))
        row[e], col[e] = r, i
    end
    return EdgeMessages(Vector{Any}(nothing, n), row, col, rows, Any[nothing for _ in rows])
end

Base.size(m::EdgeMessages) = size(m.msgs)
Base.IndexStyle(::Type{EdgeMessages}) = IndexLinear()
function Base.getindex(m::EdgeMessages, e::Int)
    r = m.row[e]
    if r != 0
        b = m.blocks[r]
        b === nothing || return Message(replica(b.belief, m.col[e]), b.iteration)
    end
    return m.msgs[e]
end
function Base.setindex!(m::EdgeMessages, v, e::Int)
    r = m.row[e]
    r != 0 && m.blocks[r] !== nothing && _unblock!(m, r)
    m.msgs[e] = v
    return m
end
function Base.fill!(m::EdgeMessages, v)
    fill!(m.msgs, v)
    fill!(m.blocks, nothing)
    return m
end

_isset(v::AbstractVector, e::Int) = v[e] !== nothing
function _isset(m::EdgeMessages, e::Int)
    r = m.row[e]
    return (r != 0 && m.blocks[r] !== nothing) || m.msgs[e] !== nothing
end

function _unblock!(m::EdgeMessages, r::Int)
    b = m.blocks[r]
    m.blocks[r] = nothing
    for (i, e) in enumerate(m.rows[r])
        m.msgs[e] = Message(replica(b.belief, i), b.iteration)
    end
    return m
end

# Row `r` as a block: the stored block, or one built from the per-edge messages. `nothing` when
# no edge of the row has a message, `missing` when some have and some do not.
function _row_block(m::EdgeMessages, r::Int)
    b = m.blocks[r]
    b === nothing || return b.belief
    es = m.rows[r]
    k = count(e -> m.msgs[e] !== nothing, es)
    k == 0 && return nothing
    k < length(es) && return missing
    return belief_block([m.msgs[e].belief for e in es])
end

function _set_block!(m::EdgeMessages, r::Int, b::BeliefBlock, iteration::Int)
    m.blocks[r] = Message(b, iteration)
    return m
end

# The block rows of a graph: one per template edge of each plate, the edges in replica order.
_block_rows(g) = Vector{Int}[collect(pl.edges[k, :]) for pl in g.plates for k in axes(pl.edges, 1)]
