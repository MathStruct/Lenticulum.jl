# Independent interval bisection on [0, 1], where x - cos(x) is strictly increasing.
function find_cosine_root()
    lo, hi = 0.0, 1.0
    for _ in 1:53
        mid = (lo + hi) / 2
        if mid > cos(mid)
            hi = mid
        else
            lo = mid
        end
    end
    return (lo + hi) / 2
end

@testset "D5 root solvers" begin
    algs = (Picard(damping=0.8), Broyden(init_scale=0.8),
            LimitedBroyden(memory=3, init_scale=0.8))
    tol = Tolerance(atol=1e-12, rtol=1e-12)
    A = [0.75 0.125; -0.25 1.0]
    b = [0.5, 0.25]
    C = [0.25 0.125; 0.125 0.25]
    for alg in algs
        for (F, root) in ((u -> A * u - b, A \ b),
                          (u -> u .- (C * u .+ b), (I - C) \ b),
                          (u -> u .- cos.(u), fill(find_cosine_root(), 2)))
            count = Ref(0)
            tracked = u -> (count[] += 1; F(u))
            u, report = solve_root(tracked, [0.0, 0.25], alg; tol)
            @test report.converged
            @test u ≈ root atol=6e-12 rtol=0
            @test report.residual ≈ norm(F(u))
            @test report.residual <= 1e-12 * (1 + norm(u))
            @test report.nfe == count[] == report.iterations + 1
        end

        _, report = solve_root(identity, Float32[], alg)
        check_report(report, :converged, 0, 1)
        @test report.residual === 0.0f0
        _, report = solve_root(_ -> [NaN], [0.0], alg)
        check_report(report, :nonfinite, 0, 1)
        @test isnan(report.residual)
        _, report = solve_root(u -> [iszero(only(u)) ? -1.0 : NaN], [0.0], alg)
        check_report(report, :nonfinite, 1, 2)
        @test isnan(report.residual)
        _, report = solve_root(_ -> [1.0], [1e30], alg)
        check_report(report, :converged, 0, 1)
        # Explicitly use an absolute criterion to exercise rounding-stall detection.
        _, report = solve_root(_ -> [1.0], [1e30], alg; tol=Tolerance(atol=1e-12, rtol=0))
        check_report(report, :stalled, 1, 2)
        @test report.residual === 1.0
    end

    u, report = solve_root(u -> u .- 1, [0.0], Picard(damping=0))
    check_report(report, :stalled, 1, 2)
    @test u == [0.0]
    @test report.residual === 1.0
end

@testset "D5 dual trajectories and implicit tangents" begin
    tol = Tolerance(atol=1e-12, rtol=0)
    for alg in (Picard(damping=0.5), Broyden(init_scale=0.5), LimitedBroyden(init_scale=0.5))
        # Choose p so the nonlinear root is known independently.
        root = 0.625
        p = root - cos(root)
        residual(a) = u -> u .- cos.(u) .- a
        u, reference = solve_root(residual(p), [0.125], alg; tol)
        dual, report = solve_root(residual(ForwardDiff.Dual{:p}(p, 1.0)), [0.125], alg; tol)
        @test reference.converged && report.converged
        @test report.reason === reference.reason
        @test report.iterations == reference.iterations
        @test report.nfe == reference.nfe
        @test report.residual isa Float64
        @test only(u) ≈ root atol=2e-12
        @test primal(only(dual)) ≈ root atol=2e-12
        @test ForwardDiff.partials(only(dual), 1) ≈ inv(1 + sin(root)) atol=5e-8

        A = [0.5 0.125; 0.125 0.75]
        b = [1.0, 2.0]
        closed(p) = A \ (p .* b)
        run(p) = first(solve_root(u -> A * u .- p .* b, zeros(2), alg; tol))
        @test run(1.0) ≈ closed(1.0) atol=6e-12
        @test ForwardDiff.derivative(run, 1.0) ≈ ForwardDiff.derivative(closed, 1.0) atol=5e-8
    end

    # A zero-step warm start preserves the input tangent even when IFT differs.
    exact = Tolerance(atol=0, rtol=0)
    for alg in (Picard(), Broyden(), LimitedBroyden())
        u, report = solve_root(u -> u .- ForwardDiff.Dual{:p}(1.0, 1.0), [1.0], alg; tol=exact)
        check_report(report, :converged, 0, 1)
        @test primal(only(u)) === 1.0
        warm_start(p) = only(first(solve_root(u -> u .- p, [1.0], alg; tol=exact)))
        tangent = ForwardDiff.derivative(warm_start, 1.0)
        @test tangent == 0
        @test_broken tangent == 1
    end
end

@testset "D5 retained inverse secants" begin
    A = [2.0 1.0; 0.5 3.0]
    v = [1.0, -2.0]
    F = u -> A * u .- v
    # Compute two updates with the original dense inverse-secant formula.
    u0 = zeros(2)
    H0 = Matrix{Float64}(I, 2, 2)
    s1 = v
    y1 = A * s1
    H1 = H0 + ((s1 - y1) * s1') / dot(s1, y1)
    s2 = -H1 * F(s1)
    y2 = H1 * A * s2
    ΔH = ((s2 - y2) * (s2' * H1)) / dot(s2, y2)
    for memory in (1, 2)
        hist = Ref{Any}()
        tol = Tolerance(atol=0, rtol=0)
        u, report = solve_root(F, u0, LimitedBroyden(; memory); tol, maxiters=2, history=hist)
        check_report(report, :maxiters, 2, 3)
        @test u ≈ s1 + s2
        a, b, scale = hist[]
        @test length(a) == length(b) == memory
        @test scale == 1
        @test scale .* v + sum(ai .* dot(bi, v) for (ai, bi) in zip(a, b)) ≈
            ((memory == 1 ? H0 : H1) + ΔH) * v
        @test scale .* v + sum(bi .* dot(ai, v) for (ai, bi) in zip(a, b)) ≈
            ((memory == 1 ? H0 : H1) + ΔH)' * v
    end

    hist = Ref{Any}()
    u, report = solve_root(identity, [0.0], LimitedBroyden(init_scale=2); history=hist)
    check_report(report, :converged, 0, 1)
    @test isempty(hist[].a) && isempty(hist[].b)
    @test hist[].scale === 2.0

    for alg in (Broyden(), LimitedBroyden())
        u, report = solve_root(u -> 2 .* u, BigFloat[big"1e-400"], alg;
                               tol=Tolerance(atol=0, rtol=0))
        check_report(report, :converged, 2, 3)
        @test iszero(only(u))
    end
end
