module ImplicitLayersMooncakeExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using Mooncake: Mooncake as MC
using LinearAlgebra: lu

IL._implicit_backend(::DI.ADTypes.AutoMooncake) = DI.ADTypes.AutoForwardDiff()

MC.@is_primitive MC.DefaultCtx MC.ReverseMode Tuple{typeof(IL._implicit_state), Any, Any, Any, Any, Any, Any, Any}
function MC.rrule!!(::MC.CoDual{typeof(IL._implicit_state)}, ad::MC.CoDual,
                    f::MC.CoDual, u::MC.CoDual, like::MC.CoDual, offs::MC.CoDual,
                    st::MC.CoDual, z0::MC.CoDual)
    a, model, primal, template, offsets, state, initial =
        map(MC.primal, (ad, f, u, like, offs, st, z0))
    out = IL._implicit_state(a, model, primal, template, offsets, state, initial)
    inner = DI.ADTypes.AutoForwardDiff()
    fac = lu(IL._root_jacobian(inner, model, primal, out[1], template, offsets, state)')
    result = MC.zero_fcodual(out)
    dz = MC.tangent(result)[1]
    du = MC.tangent(u)
    saved = copy(primal)
    zeros = map(x -> MC.zero_rdata(MC.primal(x)), (ad, f, u, like, offs, st, z0))
    function back(_)
        v = fac \ dz
        du .-= only(DI.pullback(IL._joint_residual, inner, saved, (v,),
            DI.Constant(model), DI.Constant(out[1]), DI.Constant(template),
            DI.Constant(offsets), DI.Constant(state)))
        fill!(dz, 0)
        return (MC.NoRData(), zeros...)
    end
    return result, back
end
end
