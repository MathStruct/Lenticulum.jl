import Krylov

# Krylov.jl's in-place matrix-free protocol, with the same action passed to linsolve.
struct OracleAction{T,F}
    n::Int
    apply::F
end
Base.size(A::OracleAction) = (A.n, A.n)
Base.size(A::OracleAction, d::Integer) = d <= 2 ? A.n : 1
Base.eltype(::OracleAction{T}) where {T} = T
LinearAlgebra.mul!(y::AbstractVector, A::OracleAction, x::AbstractVector) =
    copyto!(y, A.apply(x))

function krylov_oracle_system(rng, T, n, family)
    R = real(T)
    if family === :nonsymmetric
        A = Matrix{T}(I, n, n) + (R(0.4) / sqrt(R(n))) * randn(rng, T, n, n)
        return A, randn(rng, T, n)
    elseif family === :spd
        W = randn(rng, T, n, n) / sqrt(R(n))
        # In two dimensions a one-vector restart can stop on either side of
        # D2 when the left operator is nonuniform. Equalize the diagonal here
        # so the oracle's relative residual has the same scale as D2.
        n == 2 && (W ./= sqrt.(sum(abs2, W; dims=1)))
        A = Matrix{T}(I, n, n) + W' * W
        return A, randn(rng, T, n)
    end
    # Prescribe A's SVD. Cluster the good singular values so restarted solves
    # can make progress within the budget even with a condition number of 10⁶.
    U = Matrix(qr(randn(rng, T, n, n)).Q)
    σ = if n == 1
        ones(R, 1)
    elseif n == 2
        # Equal diagonal entries keep Jacobi a scalar at restart length one.
        phases = diag(U) ./ abs.(diag(U))
        U = Diagonal(phases) * T[1 1; -1 1] / sqrt(R(2))
        R[1e-6, 1]
    else
        vcat(R(1e-6), range(R(0.8), one(R); length=n-1))
    end
    A = U * Diagonal(σ) * U'
    z = randn(rng, T, n)
    n == 2 && (z[1] *= R(1e-4)) # The one-vector restart has only one good direction.
    return A, U * (σ .* z)
end

@testset "W5 Krylov.jl GMRES oracle" begin
    rng = Xoshiro(1405)
    for T in (Float32, Float64, ComplexF64), family in (:nonsymmetric, :spd, :ill_conditioned),
            n in (1, 2, 10, 50, 200)
        A, b = krylov_oracle_system(rng, T, n, family)
        R = real(T)
        atol, rtol = zero(R), T === Float32 ? R(2e-5) : R(2e-9)
        tol = Tolerance(; atol, rtol)
        diagonal = Diagonal(diag(A))
        jacobi = inv(diagonal)
        linop = v -> A * v
        action = OracleAction{T,typeof(linop)}(n, linop)
        Awide = widen.(A)
        bwide = widen.(b)
        oracle_type = T <: Real ? Float64 : ComplexF64
        singular_values = svdvals(Matrix{oracle_type}(A))
        inverse_norm = inv(last(singular_values))
        if family === :ill_conditioned && n > 1
            @test first(singular_values) * inverse_norm ≈ 1e6 rtol=0.08
        end
        bound = atol + rtol * norm(bwide)
        memories = n == 1 ? (1,) : (min(n-1, 12), n)
        for memory in memories, use_function in (false, true), precond in (:none, :left, :right, :ldiv)
            left = precond === :left ? v -> jacobi * v :
                precond === :ldiv ? v -> diagonal \ v : nothing
            right = precond === :right ? v -> jacobi * v : nothing
            M = precond === :left ? jacobi : precond === :ldiv ? diagonal : I
            N = precond === :right ? jacobi : I
            itmax = 4n + 20
            x, report = linsolve(use_function ? linop : A, b, GMRES(restart=memory, maxiters=itmax);
                tol, left, right)
            y, stats = Krylov.gmres(use_function ? action : A, b;
                atol, rtol, memory, restart=true, itmax, M, N, ldiv=precond === :ldiv)
            @testset "$T $family n=$n memory=$memory function=$use_function $precond" begin
                # Recompute residuals and their conditioning bound in wider arithmetic.
                xwide, ywide = widen.(x), widen.(y)
                rx = norm(bwide - Awide * xwide)
                ry = norm(bwide - Awide * ywide)
                @test report.converged
                @test rx <= bound
                @test report.residual == norm(b - A*x)
                @test norm(xwide - ywide) <= inverse_norm * (rx + ry) * (1 + 1e-6)
                if stats.solved
                    @test ry <= bound
                else
                    @info "Krylov.jl did not report convergence" T family n memory use_function precond stats.status
                end
                @debug "GMRES oracle iterations" T family n memory use_function precond shared=report.iterations krylov=stats.niter krylov_solved=stats.solved
            end
        end
    end

    A = [1.0 0; 0 2]
    b = ones(2)
    tol = Tolerance(atol=0, rtol=1e-12)
    _, report = linsolve(A, b, GMRES(restart=1, maxiters=1); tol)
    _, stats = Krylov.gmres(A, b; atol=0.0, rtol=1e-12, memory=1, restart=true, itmax=1)
    @test report.reason === :maxiters
    @test !report.converged
    @test !stats.solved
    @info "GMRES oracle iterations (exhausted budget)" shared=report.iterations krylov=stats.niter
end
