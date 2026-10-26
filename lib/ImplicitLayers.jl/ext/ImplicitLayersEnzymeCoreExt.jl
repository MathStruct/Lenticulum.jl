module ImplicitLayersEnzymeCoreExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using EnzymeCore: EnzymeCore as EC
using EnzymeCore: EnzymeRules as ER
using LinearAlgebra: lu

IL._implicit_backend(::DI.ADTypes.AutoEnzyme) = DI.ADTypes.AutoForwardDiff()

ER.inactive(::typeof(IL._solve_metadata), args...) = true

_shadows(x::EC.Const, n) = ntuple(_ -> zero(x.val), Val(n))
_shadows(x::EC.Duplicated, n) = (x.dval,)
_shadows(x::EC.BatchDuplicated, n) = x.dval

function ER.forward(config::ER.FwdConfig, ::EC.Const{typeof(IL._implicit_state)},
                    ::Type{<:EC.Annotation}, ad::EC.Const, f::EC.Const,
                    u::EC.Annotation, like::EC.Const, offs::EC.Const, st::EC.Const,
                    z0::EC.Annotation)
    out = IL._implicit_state(ad.val, f.val, u.val, like.val, offs.val, st.val, z0.val)
    ER.needs_shadow(config) || return ER.needs_primal(config) ? out : nothing
    inner = DI.ADTypes.AutoForwardDiff()
    Jz = IL._root_jacobian(inner, f.val, u.val, out[1], like.val, offs.val, st.val)
    rhs = DI.pushforward(IL._joint_residual, inner, u.val, _shadows(u, ER.width(config)),
        DI.Constant(f.val), DI.Constant(out[1]), DI.Constant(like.val),
        DI.Constant(offs.val), DI.Constant(st.val))
    dz = -(Jz \ hcat(rhs...))
    shadows = ntuple(Val(ER.width(config))) do k
        (dz[:, k], EC.make_zero(out[2]), EC.make_zero(out[3]))
    end
    shadow = ER.width(config) == 1 ? only(shadows) : shadows
    ER.needs_primal(config) || return shadow
    return ER.width(config) == 1 ? EC.Duplicated(out, shadow) : EC.BatchDuplicated(out, shadow)
end

function ER.augmented_primal(config::ER.RevConfig, ::EC.Const{typeof(IL._implicit_state)},
                             ::Type{<:EC.Annotation}, ad::EC.Const, f::EC.Const,
                             u::EC.Annotation, like::EC.Const, offs::EC.Const, st::EC.Const,
                             z0::EC.Annotation)
    out = IL._implicit_state(ad.val, f.val, u.val, like.val, offs.val, st.val, z0.val)
    inner = DI.ADTypes.AutoForwardDiff()
    fac = lu(IL._root_jacobian(inner, f.val, u.val, out[1], like.val, offs.val, st.val)')
    shadows = ntuple(_ -> EC.make_zero(out), Val(ER.width(config)))
    shadow = ER.width(config) == 1 ? only(shadows) : shadows
    # Save the active values in case their storage is overwritten before reverse.
    tape = (out[1], copy(u.val), fac, shadows)
    return ER.AugmentedReturn(ER.needs_primal(config) ? out : nothing,
        ER.needs_shadow(config) ? shadow : nothing, tape)
end

function ER.reverse(config::ER.RevConfig, ::EC.Const{typeof(IL._implicit_state)},
                    ::Type{<:EC.Annotation}, tape, ad::EC.Const, f::EC.Const,
                    u::EC.Annotation, like::EC.Const, offs::EC.Const, st::EC.Const,
                    z0::EC.Annotation)
    z, primal, fac, shadows = tape
    for (du, dout) in zip(_shadows(u, ER.width(config)), shadows)
        v = fac \ dout[1]
        du .-= only(DI.pullback(IL._joint_residual, DI.ADTypes.AutoForwardDiff(), primal,
            (v,), DI.Constant(f.val), DI.Constant(z), DI.Constant(like.val),
            DI.Constant(offs.val), DI.Constant(st.val)))
        fill!(dout[1], 0)
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

end
