# Shared numeric Lux trees: depth-first tuple order, column-major array order.
_flatlen(x::AbstractArray{<:Number}) = length(x)
_flatlen(x::Number) = 1
_flatlen(x::Union{NamedTuple,Tuple}) = sum(_flatlen, values(x); init = 0)

_flatfill!(v, i, x::AbstractArray{<:Number}) = (n = length(x); copyto!(v, i, vec(x), 1, n); i + n)
_flatfill!(v, i, x::Number) = (fill!(view(v, i:i), x); i + 1)
function _flatfill!(v, i, x::Union{NamedTuple,Tuple})
    for y in values(x)
        i = _flatfill!(v, i, y)
    end
    return i
end
_eltype(x::AbstractArray{<:Number}) = eltype(x)
_eltype(x::Number) = typeof(x)
# A splat, not a loop: inferred to a constant, which Enzyme needs where it differentiates `_flatten`
# (a loop under `@nospecialize` broke ImplicitLayers' Enzyme tests, I1).
_eltype(x::Union{NamedTuple,Tuple}) = isempty(x) ? Union{} : promote_type(map(_eltype, values(x))...)

# the vector lives where the parameters do (a GPU array for parameters on a GPU)
_firstleaf(x::AbstractArray{<:Number}) = x
_firstleaf(x::Number) = nothing
function _firstleaf(x::Union{NamedTuple,Tuple})
    for y in values(x)
        l = _firstleaf(y)
        l === nothing || return l
    end
    return nothing
end

function _flatten(ps)
    leaf_type, n = _eltype(ps), _flatlen(ps)
    T = leaf_type === Union{} ? Float64 : leaf_type
    leaf = _firstleaf(ps)
    v = leaf === nothing ? Vector{T}(undef, n) : similar(leaf, T, n)
    _flatfill!(v, 1, ps)
    return v
end

# The offsets of every leaf are computed once, outside the differentiated function (they do not
# depend on θ); rebuilding then only slices and reshapes θ, which every backend differentiates.
_offsets(x::AbstractArray{<:Number}, i) = (i, i + length(x))
_offsets(x::Number, i) = (i, i + 1)
# Two forms (I1). The recursion on the tuple's tail is inferred to a constant type, which Enzyme
# needs where it differentiates through `offset_tree` (ImplicitLayers' solves); but it infers one
# tuple type per step, O(keys²) of compilation for a graph-wide tree on every new structure (2.4 s
# of a Zygote replay's first call on a 61-factor chain). A tree wider than `_WIDE` fields takes a
# loop compiled once instead.
const _WIDE = 32
_offset_sequence(::Tuple{}, i) = ((), i)
function _offset_sequence(xs::Tuple, i)
    o, j = _offsets(first(xs), i)
    rest, k = _offset_sequence(Base.tail(xs), j)
    return (o, rest...), k
end
function _offsets(x::Union{NamedTuple,Tuple}, i)
    length(x) > _WIDE && return _wide_offsets(x, i)
    r, j = _offset_sequence(values(x), i)
    return (x isa NamedTuple ? NamedTuple{keys(x)}(r) : r), j
end
function _wide_offsets(@nospecialize(x::Union{NamedTuple,Tuple}), i)
    r = Vector{Any}(undef, length(x))
    for (k, y) in enumerate(values(x))
        r[k], i = _offsets(y, i)
    end
    t = Tuple(r)
    return (x isa NamedTuple ? NamedTuple{keys(x)}(t) : t), i
end
offset_tree(ps) = first(_offsets(ps, 1))

_rebuild(θ, o::Int, x::AbstractArray{<:Number}) = reshape(view(θ, o:(o + length(x) - 1)), size(x))
_rebuild(θ, o::Int, x::Number) = only(view(θ, o:o))
_rebuild(θ, o::NamedTuple, x::NamedTuple) = NamedTuple{keys(x)}(map((oi, xi) -> _rebuild(θ, oi, xi), values(o), values(x)))
_rebuild(θ, o::Tuple, x::Tuple) = map((oi, xi) -> _rebuild(θ, oi, xi), o, x)
_unflatten(θ, offs, like) = _rebuild(θ, offs, like)
_unflatten(θ, like) = _rebuild(θ, offset_tree(like), like)

# I30: a box for the graph-wide trees inside a differentiated region. Its type does not depend on
# the structure of the tree it holds, so Zygote's `Pullback` types (which hold the types of every
# argument of every function it transforms) are the same for every graph structure, and the glue
# is compiled once. Every use of the tree is explicit: `_flatten`, `offset_tree`, `_unflatten`,
# `keys` and `isempty` are deliberately not defined for `_Erased`, so a path that forgets to unbox
# fails loudly instead of silently differentiating through `getfield`.
struct _Erased
    tree::Any
end

# A rule's pullback, boxed the same way. Zygote's `Pullback` types hold the run-time types of the
# pullbacks they store, and a closure's type holds the types of what it captures: a rule whose
# pullback captures the unboxed tree made the enclosing `Pullback` type, and so the backward pass,
# depend on the structure again (a SnoopCompile profile, I30). The rules on boxed trees return
# their pullback in this box.
struct _ErasedPullback
    back::Any
end
(p::_ErasedPullback)(Δ) = p.back(Δ)

# θ rebuilt into a boxed tree (I30, step 2): the rule of `_unflatten` (the ChainRulesCore
# extension) does the work on the unboxed trees; this rule only unboxes the cotangent.
_unflatten_erased(θ, offs::_Erased, like::_Erased) = _Erased(_unflatten(θ, offs.tree, like.tree))


# Materialise views so concatenation does not merge array/view union activities.
_flatten_active(x::AbstractArray{<:Number}) = vec(copy(x))
_flatten_active(x::Number) = [x]
function _flatten_active(x::Union{NamedTuple,Tuple})
    return _flatten_active_tree(_firstleaf(x), x)
end
function _flatten_active_tree(leaf, x)
    parts = filter(!isempty, map(_flatten_active, values(x)))
    return isempty(parts) ? Float64[] : reduce(vcat, parts)
end

# A tuple of symbols is one path; a tuple of tuples denotes several paths.
function _freeze_paths(freeze)
    freeze === :all && return :all
    freeze isa Tuple || throw(ArgumentError("freeze must be :all, a tuple path, or a tuple of paths"))
    isempty(freeze) && return ()
    all(x -> x isa Union{Symbol,Int}, freeze) && return (freeze,)
    all(x -> x isa Tuple && !isempty(x) && all(k -> k isa Union{Symbol,Int}, x), freeze) ||
        throw(ArgumentError("invalid parameter freeze paths: $freeze"))
    return freeze
end
function _trainable_indices(ps, freeze)
    paths = _freeze_paths(freeze)
    paths === :all && return Int[]
    frozen = Int[]
    for path in paths
        # The start of the frozen range is accumulated while walking the path: the flat length of
        # every sibling before the key at each level. Reading it off the subtree's first leaf is
        # wrong when that subtree begins with an empty entry (a parameterless first layer).
        x, start = ps, 1
        for key in path
            valid = x isa NamedTuple ? key in keys(x) : x isa Tuple && key isa Int && 1 <= key <= length(x)
            valid || throw(ArgumentError("freeze path $path does not exist in the parameter tree"))
            for (k, y) in pairs(x)
                k == key && break
                start += _flatlen(y)
            end
            x = x[key]
        end
        append!(frozen, start:(start + _flatlen(x) - 1))
    end
    return setdiff(1:_flatlen(ps), frozen)
end

"""
    stop_gradient(x)

Identity on values, zero derivative. ChainRules and ForwardDiff extensions implement
the AD rule. Reference differences require a fixed reference value at the boundary.
"""
stop_gradient(x) = x
