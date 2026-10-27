module ImplicitLayersChainRulesCoreExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using ChainRulesCore: ChainRulesCore as CRC

function _solve_pullback(ad, fac, f, u, z, like, offs, st, dz)
    v = IL._ift_solve(fac, dz)
    return -only(DI.pullback(IL._joint_residual, ad, u, (v,), DI.Constant(f),
        DI.Constant(z), DI.Constant(like), DI.Constant(offs), DI.Constant(st)))
end
function CRC.rrule(::typeof(_solve_pullback), args...)
    throw(ArgumentError("Nested DEQ reverse AD is unsupported; use deq_hvp(...; ad=AutoForwardDiff()) for second order."))
end
function CRC.rrule(::typeof(IL._joint_vector), x, θ)
    u = IL._joint_vector(x, θ)
    function back(du)
        du = CRC.unthunk(du)
        du isa CRC.AbstractZero && return (CRC.NoTangent(), du, du)
        return CRC.NoTangent(), du[1:length(x)], du[(length(x)+1):end]
    end
    return u, back
end
function CRC.rrule(::typeof(IL._implicit_state), ad, f, u, like, offs, st, z0)
    out = IL._implicit_state(ad, f, u, like, offs, st, z0)
    z = out[1]
    # No solver iterations on the tape. Keep either a dense LU or a residual
    # product operator; the input/parameter cotangent follows the transposed solve.
    fac = IL._ift_prepare(ad, f, u, z, like, offs, st, true)
    function back(dout)
        dout = CRC.unthunk(dout)
        dz = dout isa CRC.AbstractZero ? dout : CRC.unthunk(dout[1])
        du = if dz isa CRC.AbstractZero
            zero(u)
        else
            _solve_pullback(ad, fac, f, u, z, like, offs, st, dz)
        end
        return CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), du,
               CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), CRC.ZeroTangent()
    end
    return out, back
end
end
