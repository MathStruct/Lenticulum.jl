function check_report(report, reason, iterations, nfe)
    @test report.reason === reason
    @test report.converged == (reason === :converged)
    @test report.iterations == iterations
    @test report.nfe == nfe
end

@testset "D2 tolerance and D3 primal report" begin
    # GLM's review of W1, F1: a negative or NaN bound made GMRES divide 0/0 after an exact solve
    for bad in (-1e-3, NaN, -0.5f0)
        @test_throws ArgumentError Tolerance(atol=bad)
        @test_throws ArgumentError Tolerance(rtol=bad)
    end
    @test_throws ArgumentError Tolerance(atol=ForwardDiff.Dual(-1.0, 1.0))
    @test Tolerance(atol=0, rtol=0) isa Tolerance
    exact = Tolerance(atol=0, rtol=0)
    _, report = solve_root(identity, [0.0], Picard(); tol=exact, maxiters=0)
    check_report(report, :converged, 0, 1)
    @test report.residual === 0.0
    _, report = solve_root(identity, [eps()], Picard(); tol=exact, maxiters=0)
    check_report(report, :maxiters, 0, 1)
    @test report.residual === eps()

    for (tol, u, bound) in ((Tolerance(atol=0, rtol=0.5), [4.0], 2.0),
                            (Tolerance(atol=0.5, rtol=0), [40.0], 0.5),
                            (Tolerance(atol=0.5, rtol=0.25), [4.0], 1.5))
        for (r, reason) in ((bound, :converged), (nextfloat(bound), :maxiters))
            _, report = solve_root(_ -> [r], u, Picard(); tol, maxiters=0)
            check_report(report, reason, 0, 1)
            @test report.residual === r
        end
    end

    for T in (Float32, Float64, BigFloat)
        default = max(T(1e-9), 30 * eps(T))
        for (tol, bound) in ((Tolerance(), 3 * default),
                             (Tolerance(rtol=0), default),
                             (Tolerance(atol=0), 2 * default))
            for (r, accepted) in ((bound, true), (nextfloat(bound), false))
                _, report = solve_root(_ -> [r], T[2], Picard(); tol, maxiters=0)
                @test report.converged == accepted
                @test report.residual isa T
                @test report.residual == r
            end
        end
    end

    # Casting the supplied bound to Float32 rounds it to the actual residual.
    tol = Tolerance(atol=0.1, rtol=0)
    @test tol.atol === 0.1
    _, report = solve_root(identity, Float32[0.1], Picard(); tol, maxiters=0)
    @test report.converged

    p = ForwardDiff.Dual{:p}(0.1f0, 1.0f0)
    outer = ForwardDiff.Dual{:outer}(p, one(p))
    @test primal(outer) === 0.1f0
    _, report = solve_root(_ -> [outer], Float32[0], Picard(); tol, maxiters=0)
    check_report(report, :converged, 0, 1)
    @test report.residual === 0.1f0
    @test SolverReport(true, 0, outer, 1, :converged).residual === 0.1f0

    # Rounding near an exact root must not consult even a nonfinite tangent.
    pzero = ForwardDiff.Dual{:p}(0.0, NaN)
    _, report = solve_root(identity, [pzero], Picard(); tol=exact)
    check_report(report, :converged, 0, 1)
    @test report.residual === 0.0
end
