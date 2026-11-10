module ImplicitLayersMooncakeExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using Mooncake: Mooncake as MC

IL._implicit_backend(::DI.ADTypes.AutoMooncake) = DI.ADTypes.AutoForwardDiff()

MC.@is_primitive MC.DefaultCtx MC.ReverseMode Tuple{typeof(IL._implicit_state), Any, Any, Any, Any, Any, Any, Any}
function MC.rrule!!(::MC.CoDual{typeof(IL._implicit_state)}, ad::MC.CoDual,
                    f::MC.CoDual, u::MC.CoDual, like::MC.CoDual, offs::MC.CoDual,
                    st::MC.CoDual, z0::MC.CoDual)
    a, model, primal, template, offsets, state, initial =
        map(MC.primal, (ad, f, u, like, offs, st, z0))
    out = IL._implicit_state(a, model, primal, template, offsets, state, initial)
    inner = model.ift isa IL.KrylovIFT ? DI.ADTypes.AutoMooncake() : DI.ADTypes.AutoForwardDiff()
    saved, z = copy(primal), copy(out[1])
    fac = IL._ift_prepare(inner, model, saved, z, template, offsets, state, true, IL._solve_history(out[2]))
    result = MC.zero_fcodual(out)
    dz = MC.tangent(result)[1]
    du = MC.tangent(u)
    zeros = map(x -> MC.zero_rdata(MC.primal(x)), (ad, f, u, like, offs, st, z0))
    function back(_)
        v = IL._ift_solve(fac, dz)
        du .-= only(DI.pullback(IL._joint_residual, inner, saved, (v,),
            DI.Constant(model), DI.Constant(z), DI.Constant(template),
            DI.Constant(offsets), DI.Constant(state)))
        fill!(dz, 0)
        return (MC.NoRData(), zeros...)
    end
    return result, back
end
end
