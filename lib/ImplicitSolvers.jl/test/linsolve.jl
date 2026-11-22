@testset "D4 linear systems against backslash" begin
    rng = Xoshiro(2810)
    for T in (Float32, Float64), n in (1, 2, 50), spd in (false, true)
        R = randn(rng, T, n, n)
        R = spd ? R' * R : R
        A = Matrix{T}(I, n, n) + R / (3 * norm(R))
        b = randn(rng, T, n)
        xref = A \ b
        bound = max(T(1e-9), 30 * eps(T)) * (1 + norm(b))
        error_bound = opnorm(inv(A)) * bound + 8 * eps(T) * norm(xref)
        for (solver, operator) in Iterators.product(
            (DenseSolve(; spd), GMRES(restart=3)), (A, v -> A * v))
            x, report = linsolve(operator, b, solver)
            @test report.converged
            @test norm(x - xref) <= error_bound
            @test norm(b - A * x) <= bound
            @test report.residual ≈ norm(b - A * x)
            @test report.residual isa T
            @test eltype(x) === T
        end
    end

    A = [1.0 0; 0 2]
    b = ones(2)
    tol = Tolerance(atol=0, rtol=1e-12)
    calls = Ref(0)
    apply = v -> (calls[] += 1; A * v)
    x, report = linsolve(apply, b, GMRES(maxiters=1); tol)
    check_report(report, :maxiters, 1, 3)
    @test calls[] == report.nfe
    # Minimize ||b - α A b|| over the first Krylov space.
    @test x ≈ [3 / 5, 3 / 5]
    @test report.residual ≈ sqrt(1 / 5)
    x, report = linsolve(A, b, GMRES(restart=1); tol)
    @test report.iterations > 1
    @test report.residual <= 1e-12 * sqrt(2)
    @test x ≈ [1, 1 / 2] atol=2e-12

    x, report = linsolve(zeros(2, 2), b, GMRES())
    check_report(report, :stalled, 1, 3)
    @test x == zeros(2)
    @test report.residual == sqrt(2)
    x, report = linsolve(A, b, GMRES(maxiters=0))
    check_report(report, :maxiters, 0, 1)
    @test x == zeros(2)
    @test report.residual == sqrt(2)
    x, report = linsolve(A, zeros(2), GMRES())
    check_report(report, :converged, 0, 1)
    @test x == zeros(2)
    @test report.residual === 0.0

    for (rtol, reason) in ((1.0, :converged), (prevfloat(1.0), :maxiters))
        # At x=0 the residual equals the RHS scale exactly.
        _, report = linsolve(identity, [2.0], GMRES(maxiters=0);
                             tol=Tolerance(; atol=0, rtol))
        check_report(report, reason, 0, 1)
        @test report.residual === 2.0
    end

    # Scaling A and b together must not trigger an absolute Arnoldi breakdown test.
    for magnitude in (1e-80, 1e80)
        x, report = linsolve(magnitude .* A, magnitude .* b, GMRES(); tol)
        @test report.converged
        @test x ≈ [1.0, 0.5] atol=2e-12
        @test report.residual <= 1e-12 * magnitude * sqrt(2)
    end

    for solver in (DenseSolve(), DenseSolve(spd=true), GMRES())
        x, report = linsolve(zeros(Float32, 0, 0), Float32[], solver)
        check_report(report, :converged, 0, 0)
        @test isempty(x)
        @test report.residual === 0.0f0
        x, report = linsolve(fill(NaN, 1, 1), [1.0], solver)
        @test report.reason === :nonfinite
        @test isnan(report.residual)
        @test_throws DimensionMismatch linsolve(ones(2, 1), b, solver)
    end

    for solver in (DenseSolve(), DenseSolve(spd=true), GMRES())
        rhs = [0.5, 0.25]
        p = ForwardDiff.Dual{:p}(rhs[1], 1.0)
        x, report = linsolve(A, [p, rhs[2]], solver; tol)
        _, reference = linsolve(A, rhs, solver; tol)
        check_report(report, :converged, solver isa GMRES ? 2 : 1, solver isa GMRES ? 4 : 1)
        @test report.iterations == reference.iterations
        @test report.residual isa Float64
        @test primal.(x) ≈ [0.5, 0.125]
        @test ForwardDiff.partials.(x, 1) ≈ [1.0, 0.0]
    end

    for solver in (DenseSolve(), GMRES())
        x, report = linsolve(BigFloat[4 1; 1 3], BigFloat[2, 1], solver;
                             tol=Tolerance(atol=big"1e-65", rtol=0))
        @test report.converged
        @test report.residual isa BigFloat
        @test x ≈ BigFloat[5 // 11, 2 // 11] atol=big"1e-65" rtol=0
    end
end

@testset "GMRES refuses complex systems" begin
    # GLM's review of W1, F2: the real Givens rotations diverged on a complex system
    A = diagm(ComplexF64[2 + 3im, 1 + 0.5im]); b = ComplexF64[1 - im, 2]
    @test_throws ArgumentError linsolve(A, b, GMRES())
    @test_throws ArgumentError linsolve(A, real.(b), GMRES())
    x, report = linsolve(A, b, DenseSolve())
    @test report.converged && x ≈ A \ b
end
