@testset "D6 residual and fixed-point adjoints" begin
    J = [0.5 0.125 -0.25; 0.0 0.25 -0.125; 0.125 0.0 0.375]
    g = [1.0, -3.0, 2.0]
    tol = Tolerance(atol=1e-12, rtol=1e-12)
    for alg in (DenseSolve(), GMRES(restart=2)), vjp in (J', v -> J' * v)
        lam, report = adjoint_solve(vjp, g, alg; tol)
        @test report.converged
        @test lam ≈ J' \ g atol=4e-11 rtol=0
        @test report.residual ≈ norm(g - J' * lam)

        lam, report = fixed_point_adjoint(vjp, g, alg; tol)
        @test report.converged
        @test lam ≈ (I - J') \ g atol=4e-11 rtol=0
        @test report.residual <= 1e-12 * (1 + norm(g))
    end
end

@testset "D7 central finite differences" begin
    for T in (Float32, Float64, BigFloat)
        F(u) = [u[1]^2 + 3u[2], u[1] * u[2], 2u[1] - u[2]^2]
        u = T[2, -3]
        J = fd_jacobian(F, u)
        @test eltype(J) === T
        @test J ≈ T[4 3; -3 2; 2 6] rtol=30 * eps(T)^(T(2) / 3)

        # A third-degree polynomial shows the symmetric stencil and the default step.
        for step in (nothing, T(1 // 100))
            x = T(3 // 2)
            h = (step === nothing ? cbrt(eps(T)) : step) * x
            J = fd_jacobian(v -> v .^ 3, [x]; step)
            @test only(J) ≈ 3x^2 + h^2 rtol=50 * eps(T)^(T(2) / 3)
        end

        probes = T[]
        x = T(8)
        fd_jacobian(u -> (push!(probes, only(u)); u), [x])
        h = cbrt(eps(T)) * x
        @test probes == [x + h, x - h]
    end

    # These probes round to unequal displacements around 1f0. The identity
    # derivative is one when the quotient uses their actual separation.
    @test only(fd_jacobian(identity, Float32[1]; step=7f-8)) === 1f0
    @test_throws ArgumentError fd_jacobian(identity, Float32[1]; step=1f-12)
    @test_throws ArgumentError fd_jacobian(identity, [1.0]; step=0)
    @test fd_jacobian(_ -> Float32[2, 3], Float32[]) == zeros(Float32, 2, 0)
end

@testset "W5 preconditioners apply to the adjoint system" begin
    J = ComplexF64[0.4+0.1im 0.15; 0.1im 0.5-0.1im]
    g = ComplexF64[1-im, -2im]
    for (solve, A) in ((adjoint_solve, J'), (fixed_point_adjoint, I - J'))
        N = Diagonal(inv.(diag(A)))
        x, report = solve(J', g, GMRES(restart=1); left=v -> N*v, right=v -> N*v)
        @test report.converged
        @test report.residual ≈ norm(g - A*x) atol=1e-15
        @test x ≈ A \ g atol=1e-8
        @test_throws ArgumentError solve(J', g, DenseSolve(); right=identity)
    end
end
