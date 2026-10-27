module ImplicitLayersKrylovExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using Krylov: gmres
using LinearAlgebra: LinearAlgebra, norm, I, Diagonal

# Minimal operator protocol: no indexing and no matrix conversion.
struct ResidualOperator{T,F}
    n::Int
    product::F
end
Base.size(A::ResidualOperator) = (A.n, A.n)
Base.size(A::ResidualOperator, k::Integer) = k <= 2 ? A.n : 1
Base.eltype(::ResidualOperator{T}) where {T} = T
LinearAlgebra.mul!(y, A::ResidualOperator, x) = copyto!(y, A.product(x))
struct KrylovSystem{A,K,M,N}
    operator::A
    options::K
    left::M
    right::N
end
_preconditioners(::Nothing, ctx, k) = (I, I)
function _preconditioners(p::IL.JacobiIFT, ctx, k)
    k.ldiv && throw(ArgumentError("JacobiIFT supplies an inverse; use ldiv=false"))
    z, A = ctx.root, ctx.operator
    diagonal, e, out = similar(z), zero(z), similar(z)
    for j in eachindex(z)
        fill!(e, 0)
        e[j] = 1
        LinearAlgebra.mul!(out, A, e)
        d = out[j]
        isfinite(d) && abs(d) > p.min_diagonal || throw(ArgumentError(
            "JacobiIFT requires finite nonzero diagonal entries; entry $j is $d"))
        diagonal[j] = inv(d)
    end
    return Diagonal(diagonal), I
end
function _preconditioners(p, ctx, k)
    # A callable is a factory receiving this solve's problem and orientation.
    pair = applicable(p, ctx) ? p(ctx) : p
    if pair isa NamedTuple
        all(key -> key in (:left, :right), keys(pair)) ||
            throw(ArgumentError("precond factory must return an operator or (; left, right)"))
        return get(pair, :left, I), get(pair, :right, I)
    end
    return pair, I
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
    A = ResidualOperator{eltype(z),typeof(product)}(length(z), product)
    factor = f isa IL._InputProblem ? f.factor : f
    solved = f isa IL._InputProblem ? IL.inchannel(factor) : IL.statechannel(factor)
    problem = (; operator=A, transposed, factor, solved, joint=u, root=z, template=like, offsets=offs, state=st)
    M, N = _preconditioners(k.precond, problem, k)
    return KrylovSystem(A, k, M, N)
end
function IL._ift_solve(s::KrylovSystem, b)
    k = s.options
    x, stats = gmres(s.operator, b; k.rtol, k.atol, k.itmax, k.memory,
        M=s.left, N=s.right, k.ldiv, restart=true)
    # Check the actual residual too, not only GMRES's recursive estimate.
    r = similar(b)
    LinearAlgebra.mul!(r, s.operator, x)
    residual = norm(r - b)
    solved = stats.solved && isfinite(residual) && residual <= k.atol + k.rtol * norm(b)
    k.on_solve === nothing || k.on_solve((; iterations=stats.niter, residual, solved))
    solved ||
        throw(ArgumentError("DEQ Krylov IFT did not converge: $(stats.status); residual=$residual"))
    return x
end
end
