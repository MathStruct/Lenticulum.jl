module ImplicitLayersForwardDiffExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff: ForwardDiff as FD
function IL._implicit_state(ad, f, u::AbstractVector{FD.Dual{Tag,V,N}}, like, offs, st, z0) where {Tag,V,N}
    primal = FD.value.(u)
    z, rep, stnew = IL._implicit_state(ad, f, primal, like, offs, st, FD.value.(z0))
    Jz = IL._root_jacobian(ad, f, primal, z, like, offs, st)
    # A JVP for each dual direction, followed by one multi-RHS linear solve.
    dirs = ntuple(k -> map(v -> FD.partials(v)[k], u), N)
    rhs = DI.pushforward(IL._joint_residual, ad, primal, dirs, DI.Constant(f),
        DI.Constant(z), DI.Constant(like), DI.Constant(offs), DI.Constant(st))
    dz = -(Jz \ hcat(rhs...))
    dualz = map(eachindex(z)) do i
        FD.Dual{Tag}(z[i], FD.Partials(ntuple(k -> dz[i, k], N)))
    end
    return dualz, rep, stnew
end
end
