include("../examples/flow_roundtrip.jl")

@testset "Explicit inversion: discrete one-step maps and convergence" begin
    for Integrator in (EulerIntegrator, RK4Integrator)
        @test !isreversible(Integrator())
        @test inversion_kind(Integrator(), 0.0, 1.0) isa SolverInversion
    end
    for a in (0.5, -5.0)
        h = 0.1
        vf = (u,t) -> a .* u
        @test only(integrate(vf, [1.0], 0.0, h, EulerIntegrator(1))) ≈ 1 + a*h
        @test only(integrate(vf, [1.0], h, 0.0, EulerIntegrator(1))) ≈ 1 - a*h
        @test (1 - a*h) != inv(1 + a*h)
        R(x) = 1 + x + x^2/2 + x^3/6 + x^4/24
        @test only(integrate(vf, [1.0], 0.0, h, RK4Integrator(1))) ≈ R(a*h)
        @test R(a*h)*R(-a*h) ≈ 1 + (a*h)^6/72 + (a*h)^8/576
    end
    setprecision(256) do
        for a in (big"0.5", big"-5")
            # Euler: N >= 400 keeps |a|²/N <= 0.0625, giving the first-order regime.
            # 4% allows the finite-step correction; BigFloat removes the roundoff floor.
            euler = [FlowRoundTripMeasurements.measure(EulerIntegrator, a, n).residual
                     for n in (400, 800, 1600)]
            @test all(isapprox.(euler[1:2] ./ euler[2:3], 2; rtol = 0.04))
            rk = [FlowRoundTripMeasurements.measure(RK4Integrator, a, n)
                  for n in (100, 200, 400)]
            # R(x)R(-x) = 1 + x⁶/72 + x⁸/576: N pairs accumulate O(N⁻⁵), not N⁻⁴.
            # |a|h <= 0.05 makes the omitted terms small; allow 2% for the round trip.
            @test all(isapprox.([rk[i].residual/rk[i+1].residual for i in 1:2], 32; rtol=0.02))
            # The one-direction global error still has RK4's fourth order (4% finite-h margin).
            @test all(isapprox.([rk[i].forward_error/rk[i+1].forward_error for i in 1:2], 16; rtol=0.04))
            # Check every table row against closed-form discrete multipliers.
            for n in (1, 25, 50, 100, 200), Integrator in (EulerIntegrator, RK4Integrator)
                x = a/n
                R(x) = Integrator === EulerIntegrator ? 1+x : 1+x+x^2/2+x^3/6+x^4/24
                row = FlowRoundTripMeasurements.measure(Integrator, a, n)
                @test row.residual ≈ abs((R(x)*R(-x))^n - 1) rtol=big"1e-50" atol=big"1e-70"
            end
        end
    end
    # Preserve Float64 regressions for both the default grid and a disastrously coarse grid.
    for (Integrator, expected) in ((EulerIntegrator, 1 - 0.99^50),
                                   (RK4Integrator, Float64((1 + big"0.1"^6/72 + big"0.1"^8/576)^50 - 1)))
        vf = (u,t) -> -5 .* u
        z0 = integrate(vf, [3.0], 1.0, 0.0, Integrator(50))
        r = abs(only(integrate(vf, z0, 0.0, 1.0, Integrator(50))) - 3)/3
        @test r ≈ expected rtol=1e-6
    end
    vf = (u,t) -> -5 .* u
    z0 = integrate(vf, [3.0], 1.0, 0.0, RK4Integrator(1))
    @test abs(only(integrate(vf, z0, 0.0, 1.0, RK4Integrator(1))) - 3)/3 ≈ 895.1822916666667
end
