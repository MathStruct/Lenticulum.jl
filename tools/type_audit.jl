# Fields of a package's struct types whose declared type is not concrete, or is a container with
# an abstract element type (`Vector{Any}`, `Dict{K,Any}`, `Ref{Any}`), found by reflection.
# Fields typed by a type parameter are fine and not listed. Closure types (names starting with `#`)
# are listed apart: an `Any` there is a captured variable whose type inference lost.
#
#   julia --project=lib/Mycelium.jl tools/type_audit.jl Mycelium
#
# Used for the baseline of Backlog Q (vault/Project/Feature Backlog/Backlog Q Quality and Engineering.md).
const PKG = Symbol(only(ARGS))
@eval using $PKG
const MOD = getfield(Main, PKG)

hasvar(t) = t isa TypeVar || t isa UnionAll ||
    (t isa DataType && any(hasvar, t.parameters)) || (t isa Union && (hasvar(t.a) || hasvar(t.b)))

concrete_or_small_union(p) = isconcretetype(p) || (p isa Union && all(isconcretetype, Base.uniontypes(p)))

# A concrete container whose parameters are abstract types: `Vector{Any}`, `Dict{Int,Any}`.
abstract_parameter(t) = t isa DataType && !(t <: Type) && !(t <: Function) &&
    any(p -> p isa Type && !hasvar(p) && (!concrete_or_small_union(p) || abstract_parameter(p)), t.parameters)

function audit(mod)
    fields, closures, ntypes = String[], String[], 0
    for n in names(mod; all = true)
        isdefined(mod, n) || continue
        T = getfield(mod, n)
        T isa Type || continue
        D = Base.unwrap_unionall(T)
        D isa DataType || continue
        (parentmodule(D) === mod && !isabstracttype(D) && isstructtype(D) && D.name.name === n) || continue
        ntypes += 1
        for (f, ft) in zip(fieldnames(D), fieldtypes(D))
            hasvar(ft) && continue
            isconcretetype(ft) && !abstract_parameter(ft) && continue
            push!(startswith(string(n), '#') ? closures : fields, "$(D.name.name).$f::$ft")
        end
    end
    return ntypes, fields, closures
end

ntypes, fields, closures = audit(MOD)
println(PKG, ": ", ntypes, " struct types (closures included); ", length(fields),
        " fields with a non-concrete or abstract-element type; ", length(closures), " in closures")
foreach(r -> println("  ", r), sort(fields))
isempty(closures) || println("closures:")
foreach(r -> println("  ", r), sort(closures))
