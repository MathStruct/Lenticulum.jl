# Zygote through a cluster node's free energy at fixed messages (P24).
#
# The cluster's joint and belief are built by enumeration with `push!` and in-place writes, which
# Zygote cannot take; they are generic in the number type, so ForwardDiff can. The reverse rule of
# `_cluster_local_energy` therefore takes its derivative in `ps` by ForwardDiff over the flattened
# parameter tree: the derivative of the energy with the belief a function of `ps`, as a ForwardDiff
# gradient of the whole free energy gives it (not the fixed-belief gradient of
# `free_energy_gradient`, LP-020). A cluster node has few parameters, so the forward passes are
# cheap. The messages get no tangent: inside AD they would come from the cluster's messages, which
# Zygote cannot take either (`cluster.md`, "Derivatives").
module MyceliumForwardDiffExt
using Mycelium, LenticulumCore, ChainRulesCore, ForwardDiff

_weight(x::Number) = x
_weight(_) = false                                  # a zero tangent (`nothing`, `ZeroTangent`)
_part(Δparts, name) = Δparts isa ChainRulesCore.AbstractZero || Δparts === nothing ? false : _weight(getproperty(Δparts, name))

_materialise(x::AbstractArray) = copy(x)
_materialise(x::Number) = x
_materialise(x::NamedTuple) = map(_materialise, x)
_materialise(x::Tuple) = map(_materialise, x)

function ChainRulesCore.rrule(::typeof(Mycelium._cluster_local_energy), c::Mycelium.ClusterFactor, msgs, ps, st)
    y = Mycelium._cluster_local_energy(c, msgs, ps, st)
    nt = ChainRulesCore.NoTangent()
    dmsgs = ChainRulesCore.@not_implemented("the tangent of a cluster node's free energy in its incoming messages (P24)")
    function cluster_energy_pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        dE = Δ isa ChainRulesCore.AbstractZero ? Δ : ChainRulesCore.unthunk(Δ[1])
        (dE isa ChainRulesCore.AbstractZero || dE === nothing) && return (nt, nt, dmsgs, ChainRulesCore.ZeroTangent(), nt)
        E0 = first(y)
        Δparts = dE isa Number ? nothing : ChainRulesCore.unthunk(getproperty(dE, :parts))
        names = keys(E0.parts)
        ws = map(n -> _part(Δparts, n), names)
        θ0 = LenticulumCore._flatten(ps)
        isempty(θ0) && return (nt, nt, dmsgs, ChainRulesCore.ZeroTangent(), nt)
        offs = LenticulumCore.offset_tree(ps)
        g = ForwardDiff.gradient(θ0) do θ
            E = first(Mycelium._cluster_local_energy(c, msgs, LenticulumCore._unflatten(θ, offs, ps), st))
            sum(w * E[n] for (w, n) in zip(ws, names))
        end
        return (nt, nt, dmsgs, _materialise(LenticulumCore._unflatten(g, offs, ps)), nt)
    end
    return y, cluster_energy_pullback
end
end
