# Quadratic, quartic saddle, and Rosenbrock fixtures from the W6 design.
function descent_problem(field, energy, hessian)
    return (residual = (U, cols) -> stack(map(field, eachcol(U))),
            scale = (U, cols) -> map(norm, eachcol(U)),
            system = (U, cols) -> begin
                H = map(hessian, eachcol(U))
                (map(isposdef, H), map(A -> r -> A \ r, H))
            end,
            merit = (U, R, cols) -> map(energy, eachcol(U)),
            model = (U, R, cols) -> [(R[:, b], hessian(U[:, b])) for b in axes(U, 2)],
            stable = (U, cols) -> map(isposdef ∘ hessian, eachcol(U)))
end

@testset "P39 column solvers" begin
    tol = Tolerance(atol=1e-10, rtol=0)
    H = Symmetric([2.0 1.0; 1.0 3.0])
    root = [1.0, -2.0]
    r(u) = H * (u - root)
    E(u) = dot(u - root, r(u)) / 2
    start = reshape([0.0, 0.0], :, 1)
    p = descent_problem(r, E, _ -> H)

    U, reports, flags = solve_columns(p, start, NewtonDescent(); tol, maxiters=1)
    @test vec(U) ≈ root atol=1e-14
    check_report(only(reports), :converged, 1, 3)
    @test only(flags)
    @test iszero(start)

    visits = Float64[]
    tracked = merge(p, (merit = (U, R, cols) -> begin
        values = p.merit(U, R, cols)
        append!(visits, values)
        values
    end,))
    U, reports, flags = solve_columns(tracked, start, TrustRegion(); tol, maxiters=40)
    @test vec(U) ≈ root atol=1e-10
    @test only(flags) && only(reports).converged
    @test length(visits) == only(reports).iterations + 1 == only(reports).nfe
    @test all(<(0), diff(visits))

    # The saddle has x=0, with two minima at x=±1 on its unstable axis.
    quartic(u) = (u[1]^2 - 1)^2 / 4 + u[2]^2 / 2
    quartic_gradient(u) = [u[1]^3 - u[1], u[2]]
    quartic_hessian(u) = Symmetric([3u[1]^2 - 1 0; 0 1])
    saddle = descent_problem(quartic_gradient, quartic, quartic_hessian)
    near = reshape([0.05, 0.1], :, 1)
    U, reports, flags = solve_columns(saddle, near, NewtonDescent(); tol, maxiters=3)
    expected = vec(near)
    for _ in 1:3
        @test !isposdef(quartic_hessian(expected))
        expected = expected - 0.05 * quartic_gradient(expected)
    end
    @test vec(U) == expected
    @test !only(flags)
    check_report(only(reports), :maxiters, 3, 4)
    U, reports, flags = solve_columns(saddle, near, TrustRegion(); tol, maxiters=40)
    @test vec(U) ≈ [1.0, 0.0] atol=1e-10
    @test only(flags) && only(reports).converged

    rosen(u) = (1 - u[1])^2 + 100(u[2] - u[1]^2)^2
    rosen_gradient(u) = [2(u[1] - 1) - 400u[1] * (u[2] - u[1]^2), 200(u[2] - u[1]^2)]
    rosen_hessian(u) = Symmetric([2 - 400u[2] + 1200u[1]^2 -400u[1]; -400u[1] 200])
    problem = descent_problem(rosen_gradient, rosen, rosen_hessian)
    U, reports, flags = solve_columns(problem, reshape([-1.2, 1.0], :, 1), TrustRegion();
                                      tol, maxiters=200)
    @test vec(U) ≈ [1.0, 1.0] atol=1e-9
    @test only(reports).converged && only(flags)

    # F(x,y)=[x, 1] has a Gauss–Newton minimum at x=0 but has no root.
    least_squares = (residual = (U, cols) -> vcat(U[1:1, :], ones(1, size(U, 2))),
        scale = (U, cols) -> map(norm, eachcol(U)),
        merit = (U, R, cols) -> map(r -> sum(abs2, r) / 2, eachcol(R)),
        model = (U, R, cols) -> [(vcat(r[1], 0.0), v -> [v[1], 0.0]) for r in eachcol(R)],
        stable = (U, cols) -> fill(false, length(cols)))
    U, reports, flags = solve_columns(least_squares, reshape([1.0, 2.0], :, 1),
                                      TrustRegion(radius=2.0); tol, maxiters=40)
    @test vec(U) == [0.0, 2.0]
    check_report(only(reports), :stalled, 2, 3)
    @test only(reports).residual == 1.0

    for alg in (NewtonDescent(), TrustRegion())
        starts = [1.0 0.05 -0.7 0.2 NaN; 0.0 0.1 2.0 0.0 0.0]
        evaluated = Vector{Vector{Int}}()
        tracked = merge(saddle, (residual = (U, cols) -> begin
            push!(evaluated, copy(cols))
            saddle.residual(U, cols)
        end,))
        U, reports, flags = solve_columns(tracked, starts, alg; tol, maxiters=30)
        @test first(evaluated) == collect(1:5)
        @test all(cols -> 1 ∉ cols && 5 ∉ cols, evaluated[2:end])
        for b in axes(starts, 2)
            u, rs, f = solve_columns(saddle, starts[:, b:b], alg; tol, maxiters=30)
            @test reinterpret(UInt64, U[:, b]) == reinterpret(UInt64, vec(u))
            actual, expected = reports[b], only(rs)
            @test actual.reason === expected.reason
            @test actual.converged == expected.converged
            @test actual.iterations == expected.iterations
            @test actual.nfe == expected.nfe
            @test actual.nfe == count(cols -> b in cols, evaluated)
            @test reinterpret(UInt64, actual.residual) == reinterpret(UInt64, expected.residual)
            @test flags[b] == only(f)
        end
        @test reports[1].converged && reports[1].iterations == 0
        check_report(reports[5], :nonfinite, 0, 1)

        U, reports, flags = solve_columns(p, start, alg; tol, maxiters=0)
        @test U == start
        check_report(only(reports), :maxiters, 0, 1)
        @test only(flags)
        U, reports, flags = solve_columns(p, reshape(root, :, 1), alg; tol, maxiters=0)
        check_report(only(reports), :converged, 0, 1)
        U, reports, flags = solve_columns(NamedTuple(), zeros(Float32, 2, 0), alg; tol, maxiters=40)
        @test U isa Matrix{Float32} && size(U) == (2, 0)
        @test reports isa Vector{SolverReport} && isempty(reports)
        @test flags isa Vector{Bool} && isempty(flags)
    end

    # The merit rounds to 1 at both iterates; the residual alone accepts the step.
    small = reshape(Float32[1e-4], :, 1)
    energy32(u) = 1f0 + sum(abs2, u) / 2
    @test energy32(vec(small)) === 1f0
    p32 = descent_problem(identity, energy32, _ -> Symmetric(ones(Float32, 1, 1)))
    U, reports, _ = solve_columns(p32, small, TrustRegion();
                                  tol=Tolerance(atol=0, rtol=0), maxiters=1)
    @test iszero(U) && eltype(U) === Float32
    @test only(reports).residual === 0f0
    check_report(only(reports), :converged, 1, 2)

    # A lower-precision field can still supply a full-precision descent update.
    # The batch adapter can opt into rounding the direction in its residual buffer.
    mixed = (residual = (U, cols) -> -Float32.(U),
             scale = (U, cols) -> map(norm, eachcol(U)),
             system = (U, cols) -> ([false], (identity,)))
    raw = reshape([0.7], :, 1)
    rounded = merge(mixed, (round_steps = true,))
    for (p, delta) in ((mixed, 0.05 * 0.7f0), (rounded, Float32(0.05 * 0.7f0)))
        U, _, _ = solve_columns(p, raw, NewtonDescent(); tol, maxiters=1)
        @test only(U) === 0.7 + delta
    end

    # LP-085: a trial point with a non-finite residual counts as ratio 0, so the radius shrinks.
    # Before P39, VariationalDiffusion's loop got a NaN ratio, kept the radius and retried the
    # same step until `maxiters`. Here every point below 0.5 is NaN and the root is at 0.
    nantrial = (residual = (U, cols) -> ifelse.(U .> 0.5, U, NaN),
                scale = (U, cols) -> map(norm, eachcol(U)),
                merit = (U, R, cols) -> vec(sum(abs2, R; dims=1)) ./ 2,
                model = (U, R, cols) -> [(R[:, b], Symmetric(ones(1, 1))) for b in axes(U, 2)],
                stable = (U, cols) -> fill(true, size(U, 2)))
    U, reports, _ = solve_columns(nantrial, reshape([2.0], :, 1), TrustRegion(radius=4.0);
                                  tol=Tolerance(atol=1e-14, rtol=0), maxiters=200)
    @test only(reports).reason === :stalled
    @test only(reports).iterations < 200
    @test 0.5 < only(U) < 0.5 + 1e-6
end

@testset "P39 trust-region subproblem" begin
    # KKT: λ=2, p₂=-1/3, and ‖p‖=1. Either sign of p₁ minimizes the model.
    H = Symmetric([-2.0 0.0; 0.0 1.0])
    g = [0.0, 1.0]
    p = ImplicitSolvers._tr_step(TrustRegion(), g, H, 1.0)
    @test norm(p) ≈ 1
    @test abs(p[1]) ≈ sqrt(8 / 9)
    @test p[2] ≈ -1 / 3 atol=2e-8
    @test (H + 2I) * p + g ≈ zeros(2) atol=3sqrt(eps(Float64))

    A = Symmetric([3.0 0.5; 0.5 2.0])
    g = [1.0, -2.0]
    exact = -(A \ g)
    alg = TrustRegion(cg_rtol=1e-12)
    p = ImplicitSolvers._tr_step(alg, g, A, 10.0)
    q = ImplicitSolvers._tr_step(alg, g, v -> A * v, 10.0)
    @test p ≈ exact atol=1e-12
    @test q ≈ p atol=1e-12
    @test norm(ImplicitSolvers._tr_step(alg, [1.0, 0.0], v -> H * v, 0.25)) ≈ 0.25
end
