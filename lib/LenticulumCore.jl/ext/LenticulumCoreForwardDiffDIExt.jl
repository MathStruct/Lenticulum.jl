module LenticulumCoreForwardDiffDIExt
# One dual type for the graph drivers' ForwardDiff calls, whatever the structure (I1). Without a
# tag, DifferentiationInterface tags the duals with the type of the differentiated closure, which
# holds the graph-wide parameter and state trees (a `NamedTuple` with a field per factor): every new
# structure gave a new dual type, and every factor's message and energy code compiled again for it
# (11 s per structure on a 31-factor chain, `passing.md`). Here the tag is `Tag{_GraphTag, V}`,
# `V` the element type of the coordinates, and the chunk is fixed: 12, or the length when shorter.
# The number of chunks is ForwardDiff's own (`pickchunksize` keeps it, choosing a smaller chunk);
# only the last chunk may be padded.
#
# Nesting. ForwardDiff orders nested tags by creation, and a fixed tag is created once. So the
# fixed tag is used only where nothing in the contexts is a dual already (a user's outer
# derivative through a factor's constants, the state, or the coordinates themselves); then no other
# tag can meet it, and a nested call inside (none exists today) would take its default tag, created
# later. Otherwise the backend is left as it was.
using LenticulumCore: LenticulumCore
using ForwardDiff: ForwardDiff
using DifferentiationInterface: DifferentiationInterface as DI
const ADTypes = DI.ADTypes

struct _GraphTag end

const _CHUNK = 12

function LenticulumCore._graph_ad(ad::ADTypes.AutoForwardDiff{C,Nothing}, u, args) where {C}
    eltype(u) <: Union{Float32,Float64} || return ad
    _has_dual(args, IdDict{Any,Nothing}()) && return ad
    chunk = C === nothing ? clamp(length(u), 1, _CHUNK) : C
    return ADTypes.AutoForwardDiff(; chunksize = chunk, tag = ForwardDiff.Tag(_GraphTag(), eltype(u)))
end

# Whether a dual number is reachable from `x`: by type where the type says it (bits types, arrays
# of bits types), else through the fields and elements. `@nospecialize`, so that the walk compiles
# once and not per structure.
_type_has_dual(@nospecialize(T)) = T <: ForwardDiff.Dual ||
    (isconcretetype(T) && isbitstype(T) && any(_type_has_dual, fieldtypes(T)))
function _has_dual(@nospecialize(x), seen)
    x isa ForwardDiff.Dual && return true
    x isa Union{Symbol,AbstractString,Module,Type,Task,Base.AbstractLock,IO,Ptr,Core.MethodInstance,Method} && return false
    T = typeof(x)
    isbitstype(T) && return _type_has_dual(T)
    if ismutable(x)
        haskey(seen, x) && return false
        seen[x] = nothing
    end
    if x isa Union{AbstractArray,Memory}
        E = eltype(x)
        isconcretetype(E) && isbitstype(E) && return _type_has_dual(E)
        x isa Union{Array,Memory} || return _walk_fields(x, seen)
        for i in eachindex(x)
            isassigned(x, i) && _has_dual(x[i], seen) && return true
        end
        return false
    end
    return _walk_fields(x, seen)
end
function _walk_fields(@nospecialize(x), seen)
    for i in 1:nfields(x)
        isdefined(x, i) && _has_dual(getfield(x, i), seen) && return true
    end
    return false
end

end
