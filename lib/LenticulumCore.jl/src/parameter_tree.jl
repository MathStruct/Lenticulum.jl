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
_offset_sequence(::Tuple{}, i) = ((), i)
function _offset_sequence(xs::Tuple, i)
    o, j = _offsets(first(xs), i)
    rest, k = _offset_sequence(Base.tail(xs), j)
    return (o, rest...), k
end
function _offsets(x::Union{NamedTuple,Tuple}, i)
    r, j = _offset_sequence(values(x), i)
    return (x isa NamedTuple ? NamedTuple{keys(x)}(r) : r), j
end
offset_tree(ps) = first(_offsets(ps, 1))

_rebuild(θ, o::Int, x::AbstractArray{<:Number}) = reshape(view(θ, o:(o + length(x) - 1)), size(x))
_rebuild(θ, o::Int, x::Number) = only(view(θ, o:o))
_rebuild(θ, o::NamedTuple, x::NamedTuple) = NamedTuple{keys(x)}(map((oi, xi) -> _rebuild(θ, oi, xi), values(o), values(x)))
_rebuild(θ, o::Tuple, x::Tuple) = map((oi, xi) -> _rebuild(θ, oi, xi), o, x)
_unflatten(θ, offs, like) = _rebuild(θ, offs, like)
_unflatten(θ, like) = _rebuild(θ, offset_tree(like), like)


# Materialise views so concatenation does not merge array/view union activities.
_flatten_active(x::AbstractArray{<:Number}) = vec(copy(x))
_flatten_active(x::Number) = [x]
function _flatten_active(x::Union{NamedTuple,Tuple})
    return _flatten_active_tree(_firstleaf(x), x)
end
function _flatten_active_tree(leaf, x)
    parts = map(_flatten_active, values(x))
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
    offsets = offset_tree(ps)
    for path in paths
        x, o = ps, offsets
        for key in path
            valid = x isa NamedTuple ? key in keys(x) : x isa Tuple && key isa Int && 1 <= key <= length(x)
            valid || throw(ArgumentError("freeze path $path does not exist in the parameter tree"))
            x, o = x[key], o[key]
        end
        start = o isa Int ? o : _firstoffset(o)
        append!(frozen, start:(start + _flatlen(x) - 1))
    end
    return setdiff(1:_flatlen(ps), frozen)
end
_firstoffset(o::Int) = o
_firstoffset(o::Union{NamedTuple,Tuple}) = isempty(o) ? 1 : _firstoffset(first(o))

"""
    stop_gradient(x)

Identity on values, zero derivative. ChainRules and ForwardDiff extensions implement
the AD rule. Reference differences require a fixed reference value at the boundary.
"""
stop_gradient(x) = x
