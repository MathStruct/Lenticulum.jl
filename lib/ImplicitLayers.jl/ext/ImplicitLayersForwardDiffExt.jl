module ImplicitLayersForwardDiffExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff: ForwardDiff as FD
using LenticulumCore: LenticulumCore as LC
using LinearAlgebra: dot
IL._solve_metadata(x::AbstractVector{<:FD.Dual}) = FD.value.(x)

function IL._implicit_state(ad, f, u::AbstractVector{FD.Dual{Tag,V,N}}, like, offs, st, z0) where {Tag,V,N}
    V <: AbstractFloat && ad isa DI.ADTypes.AutoForwardDiff || throw(ArgumentError(
        "Nested DEQ solve AD is unsupported; use deq_hvp(...; ad=AutoForwardDiff()) for second order."))
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

function IL._deq_hvp(::DI.ADTypes.AutoForwardDiff, loss, f, x, ps, st, direction, wrt, z0)
    θ, offs = LC._flatten_active(ps), LC.offset_tree(ps)
    u = IL._joint_vector(x, θ)
    eltype(u) <: AbstractFloat && eltype(direction) <: AbstractFloat ||
        throw(ArgumentError("deq_hvp requires real floating-point primal vectors; AD nesting is unsupported"))
    nx = length(x)
    inds = wrt === :ps ? (nx+1:length(u)) : wrt === :x ? (1:nx) : (1:length(u))
    length(direction) == length(inds) || throw(ArgumentError("deq_hvp direction has the wrong length for wrt=$wrt"))
    v = zero(u)
    v[inds] .= direction
    z, _, _ = IL._implicit_state(DI.ADTypes.AutoForwardDiff(), f, u, ps, offs, st, z0)
    n = length(z)
    R(w) = IL._joint_residual(w[(n+1):end], f, w[1:n], ps, offs, st)
    w = vcat(z, u)
    D = FD.jacobian(R, w)
    J, B = D[:,1:n], D[:,(n+1):end]
    tangent = -(J \ (B*v))
    λ = J' \ FD.gradient(loss, z)
    # Contract the residual Hessians with the adjoint before differentiating.
    curvature = FD.hessian(w -> dot(λ, R(w)), w) * vcat(tangent, v)
    dλ = J' \ (FD.hessian(loss, z)*tangent - curvature[1:n])
    hv = -B'*dλ - curvature[(n+1):end]
    return hv[inds]
end
end
