@testset "D4 and D5 JLArrays without scalar indexing" begin
    JLArrays.allowscalar(false)
    for T in (Float32, Float64)
        A = T[1.25 0.125; 0.125 1.5]
        b = T[0.5, -0.25]
        dA, db = JLArray(A), JLArray(b)
        bound = max(T(1e-9), 30 * eps(T))
        xref = A \ b
        for alg in (DenseSolve(), DenseSolve(spd=true), GMRES()), op in (dA, v -> dA * v)
            x, report = linsolve(op, db, alg)
            @test x isa JLArray{T,1}
            @test report.converged
            @test norm(Array(x) - xref) <= 2 * bound
            @test norm(b - A * Array(x)) <= bound * (1 + norm(b))
            @test report.residual isa T
        end

        for alg in (Picard(damping=0.5), Broyden(), LimitedBroyden(memory=2))
            x, report = solve_root(v -> dA * v .- db, JLArray(zeros(T, 2)), alg)
            @test x isa JLArray{T,1}
            @test report.converged
            @test norm(Array(x) - xref) <= 2 * bound
            @test norm(A * Array(x) - b) <= bound * (1 + norm(Array(x)))
            @test report.residual isa T
        end
    end

    # D6 on device arrays (GLM's review of W1, F3)
    for T in (Float32, Float64)
        J = T[0.25 0.125; -0.125 0.5]; g = T[1, -2]
        dg = JLArray(g)
        bound = max(T(1e-9), 30 * eps(T))
        for alg in (DenseSolve(), GMRES()), op in (JLArray(Matrix(J')), v -> JLArray(Matrix(J')) * v)
            λ, report = adjoint_solve(op, dg, alg)
            @test λ isa JLArray{T,1} && report.converged
            @test norm(Array(λ) - J' \ g) <= 4 * bound * (1 + norm(g))
            μ, report = fixed_point_adjoint(op, dg, alg)
            @test μ isa JLArray{T,1} && report.converged
            @test norm(Array(μ) - (I - J') \ g) <= 4 * bound * (1 + norm(g))
        end
    end
end
