module ImplicitLayersChainRulesCoreExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using ChainRulesCore: ChainRulesCore as CRC
using LinearAlgebra: lu
function CRC.rrule(::typeof(IL._implicit_state), ad, f, u, like, offs, st, z0)
    out = IL._implicit_state(ad, f, u, like, offs, st, z0)
    z = out[1]
    # No solver iterations on the tape. One dense residual linearisation is kept;
    # the parameter/input cotangent is a backend VJP after the transposed solve.
    Jz = IL._root_jacobian(ad, f, u, z, like, offs, st)
    fac = lu(Jz')
    function back(dout)
        dout = CRC.unthunk(dout)
        dz = dout isa CRC.AbstractZero ? dout : CRC.unthunk(dout[1])
        du = if dz isa CRC.AbstractZero
            zero(u)
        else
            v = fac \ dz
            -only(DI.pullback(IL._joint_residual, ad, u, (v,), DI.Constant(f),
                DI.Constant(z), DI.Constant(like), DI.Constant(offs), DI.Constant(st)))
        end
        return CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), du,
               CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), CRC.ZeroTangent()
    end
    return out, back
end
end
