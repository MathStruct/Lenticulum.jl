module ImplicitLayersKrylovExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using Krylov: gmres
using LinearAlgebra: LinearAlgebra, norm

# Minimal operator protocol: no indexing and no matrix conversion.
struct ResidualOperator{T,F}
    n::Int
    product::F
end
Base.size(A::ResidualOperator) = (A.n, A.n)
Base.size(A::ResidualOperator, k::Integer) = k <= 2 ? A.n : 1
Base.eltype(::ResidualOperator{T}) where {T} = T
LinearAlgebra.mul!(y, A::ResidualOperator, x) = copyto!(y, A.product(x))
struct KrylovSystem{A,K}
    operator::A
    options::K
end
_state_residual(z, u, f, like, offs, st) = IL._joint_residual(u, f, z, like, offs, st)
function IL._ift_prepare(k::IL.KrylovIFT, ad, f, u, z, like, offs, st, transposed)
    ctx = map(DI.Constant, (u, f, like, offs, st))
    product = if transposed
        prep = DI.prepare_pullback(_state_residual, ad, z, (zero(z),), ctx...)
        v -> only(DI.pullback(_state_residual, prep, ad, z, (v,), ctx...))
    else
        prep = DI.prepare_pushforward(_state_residual, ad, z, (zero(z),), ctx...)
        v -> only(DI.pushforward(_state_residual, prep, ad, z, (v,), ctx...))
    end
    return KrylovSystem(ResidualOperator{eltype(z),typeof(product)}(length(z), product), k)
end
function IL._ift_solve(s::KrylovSystem, b)
    k = s.options
    x, stats = gmres(s.operator, b; k.rtol, k.atol, k.itmax, k.memory, restart=true)
    # Check the actual residual too, not only GMRES's recursive estimate.
    r = similar(b)
    LinearAlgebra.mul!(r, s.operator, x)
    residual = norm(r - b)
    stats.solved && isfinite(residual) && residual <= k.atol + k.rtol * norm(b) ||
        throw(ArgumentError("DEQ Krylov IFT did not converge: $(stats.status); residual=$residual"))
    return x
end
end
