module ImplicitLayersForwardDiffExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff: ForwardDiff as FD
IL._solve_metadata(x::AbstractVector{<:FD.Dual}) = FD.value.(x)

function IL._implicit_state(ad, f, u::AbstractVector{FD.Dual{Tag,V,N}}, like, offs, st, z0) where {Tag,V,N}
    primal = FD.value.(u)
    z, rep, stnew = IL._implicit_state(ad, f, primal, like, offs, st, FD.value.(z0))
    Jz = IL._ift_prepare(ad, f, primal, z, like, offs, st, false)
    # A residual JVP and IFT solve for each dual direction.
    dirs = ntuple(k -> map(v -> FD.partials(v)[k], u), N)
    rhs = DI.pushforward(IL._joint_residual, ad, primal, dirs, DI.Constant(f),
        DI.Constant(z), DI.Constant(like), DI.Constant(offs), DI.Constant(st))
    dz = -hcat(map(b -> IL._ift_solve(Jz, b), rhs)...)
    dualz = map(eachindex(z)) do i
        FD.Dual{Tag}(z[i], FD.Partials(ntuple(k -> dz[i, k], N)))
    end
    return dualz, rep, stnew
end
end
