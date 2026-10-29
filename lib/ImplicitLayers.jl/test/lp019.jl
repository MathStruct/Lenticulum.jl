# Regression tests for LP-019 (Codex's review of the core packages).

@testset "LP-019.4 finite differences divide by the step actually taken" begin
    for T in (Float64, Float32, Float16)
        u = T[1]
        # For the identity F(u + h) − F(u) is exactly the rounded step, so the quotient is 1.
        @test fd_jacobian(identity, u) == ones(1, 1)        # 1.1920929 in Float32, 0 in Float16
        @test ImplicitLayers._divergence((u, t) -> u, u, T(0)) == 1
        @test fd_jacobian(x -> 3 .* x, T[2.5])[1] ≈ 3 rtol = sqrt(eps(T))
        for itg in (EulerIntegrator(steps = 1), RK4Integrator(steps = 50))
            # u′ = u: the divergence is 1, so Δ log det = t₁ − t₀ = 1
            _, Δ = integrate_with_divergence((u, t) -> u, u, T(0), T(1), itg)
            @test Δ ≈ 1 rtol = 10eps(T)                  # was 0.9758 (Float32 RK4), 0 (Float16)
        end
    end
end

@testset "LP-019.10 residual norms do not underflow" begin
    for solver in (PicardSolver(tol = 1e-210), BroydenSolver(tol = 1e-210), LimitedBroydenSolver(tol = 1e-210))
        _, rep = solve_root(u -> [1e-200], [0.0], solver)
        @test !rep.converged                                # was converged at iteration 0
        @test rep.residual ≈ 1e-200
    end
    @test ImplicitLayers._norm([3e-200, 4e-200]) ≈ 5e-200
    @test ImplicitLayers._norm([3e200, 4e200]) ≈ 5e200   # and the other end does not overflow
end
