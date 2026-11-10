using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using MyceliumLearning

# LP-021: the coupled gradient drivers (non-diagonal couplings, `adjoint = :implicit`) cut a
# `DiagonalCoupling` edge at its fixed message, the derivative LP-020 replaced in the default
# driver by the fixed-belief one; a factor whose edges were all cut contributed the fixed-message
# derivative (0.58 off here). Fixed by option 1: such a factor is differentiated at its fixed
# belief in the coupled drivers too. Oracle: central differences of the converged free energy.

@testset "LP-021: the coupled drivers hold fully cut factors at their beliefs" begin
    d, k = 3, 2
    x = randn(Xoshiro(3), d)
    function graph(coupling; data = coupling)
        b = GraphBuilder()
        variable!(b, :z, k); variable!(b, :x, d)
        factor!(b, :prior, GaussianPrior(:z, zeros(k), Matrix(1.0I, k, k)))
        factor!(b, :lik, LinearConstraintFactor((x = d, z = k), d; noise = 0.7))
        factor!(b, :data, DataFactor(:x, x))
        connect!(b, :prior, :z, :z; coupling); connect!(b, :lik, :z, :z; coupling)
        connect!(b, :lik, :x, :x; direction = Absorbing(), coupling)
        connect!(b, :data, :x, :x; direction = Emitting(), coupling = data)
        return build(b)
    end
    g, ge = graph(DiagonalCoupling()), graph(ExactCoupling())
    gm = graph(DiagonalCoupling(); data = ExactCoupling())   # one exact edge: the coupled drivers run
    ps = LuxCore.initialparameters(Xoshiro(0), g)
    st = LuxCore.initialstates(Xoshiro(0), g)
    fl(p) = LenticulumCore._flatten(p)
    θ = fl(ps)
    function F(t)
        p = LenticulumCore._unflatten(t, ps)
        _, _, st2, s2 = infer!(g, tree_schedule(g), p, st)
        return first(LenticulumCore.scalar_free_energy(s2, g, p, st2))
    end
    e(i) = 1e-6 .* (1:length(θ) .== i)
    fd = [(F(θ .+ e(i)) - F(θ .- e(i))) / 2e-6 for i in eachindex(θ)]
    _, _, st1, s = infer!(g, tree_schedule(g), ps, st)
    _, _, ste, se = infer!(ge, tree_schedule(ge), ps, st)
    diagonal = fl(LenticulumCore.free_energy_gradient(s, g, ps, st1))
    implicit_diag = fl(LenticulumCore.free_energy_gradient(s, g, ps, st1; adjoint = :implicit, schedule = tree_schedule(g)))
    implicit_exact = fl(LenticulumCore.free_energy_gradient(se, ge, ps, ste; adjoint = :implicit,
                                                           schedule = tree_schedule(ge), linear_tol = 1e-7))
    _, _, stm, sm = infer!(gm, tree_schedule(gm), ps, st)
    mixed_unrolled = fl(LenticulumCore.free_energy_gradient(sm, gm, ps, stm; schedule = tree_schedule(gm), nsweeps = 1))
    mixed_implicit = fl(LenticulumCore.free_energy_gradient(sm, gm, ps, stm; adjoint = :implicit,
                                                           schedule = tree_schedule(gm), linear_tol = 1e-7))
    @test maximum(abs, diagonal .- fd) < 1e-8                 # the default driver: fixed beliefs (LP-020)
    @test maximum(abs, implicit_exact .- fd) < 1e-8           # differentiating through every message
    @test maximum(abs, implicit_diag .- fd) < 1e-8            # was 0.58 before the fix
    @test maximum(abs, mixed_unrolled .- fd) < 1e-8           # one exact edge no longer changes the rest
    @test maximum(abs, mixed_implicit .- fd) < 1e-8
end

# LP-021's mixed case, resolved by option 3 (2026-11-04): a factor with cut and differentiated edges
# whose messages depend on the parameters. The replay (with `schedule`) differentiates the truncated
# computation and is off even at convergence (measured here, and warned); without `schedule` and with
# `adjoint = :implicit` the result is the envelope gradient. Oracle: central differences of the
# converged free energy. A chain with two latent variables: z1 ~ N(0, I), z2 = A z1 + ε, x = W z2 + ε.
@testset "LP-021: mixed cuts give the envelope gradient (option 3)" begin
    k1, k2, d = 2, 2, 3
    x = randn(Xoshiro(3), d)
    function chain(c1p, c1l, c2l, c2k)
        b = GraphBuilder()
        variable!(b, :z1, k1); variable!(b, :z2, k2); variable!(b, :x, d)
        factor!(b, :prior, GaussianPrior(:z1, zeros(k1), Matrix(1.0I, k1, k1)))
        factor!(b, :lin, LinearConstraintFactor((z2 = k2, z1 = k1), k2; noise = 0.5))
        factor!(b, :lik, LinearConstraintFactor((x = d, z2 = k2), d; noise = 0.7))
        factor!(b, :data, DataFactor(:x, x))
        connect!(b, :prior, :z1, :z1; coupling = c1p); connect!(b, :lin, :z1, :z1; coupling = c1l)
        connect!(b, :lin, :z2, :z2; coupling = c2l); connect!(b, :lik, :z2, :z2; coupling = c2k)
        connect!(b, :lik, :x, :x; direction = Absorbing())
        connect!(b, :data, :x, :x; direction = Emitting())
        return build(b)
    end
    D, X = DiagonalCoupling(), ExactCoupling()
    g0 = chain(D, D, D, D)
    ps = LuxCore.initialparameters(Xoshiro(0), g0); st = LuxCore.initialstates(Xoshiro(0), g0)
    fl(p) = LenticulumCore._flatten(p); θ = fl(ps)
    function F(t)
        p = LenticulumCore._unflatten(t, ps)
        _, _, st2, s2 = infer!(g0, tree_schedule(g0), p, st)
        return first(LenticulumCore.scalar_free_energy(s2, g0, p, st2))
    end
    e(i) = 1e-6 .* (1:length(θ) .== i)
    fd = [(F(θ .+ e(i)) - F(θ .- e(i))) / 2e-6 for i in eachindex(θ)]
    for g in (chain(D, D, X, X), chain(X, X, D, D), chain(D, D, X, D))       # lin, or z2's edges, mixed
        @test MyceliumLearning._mixed_cuts(g)
        _, _, s1, s = infer!(g, tree_schedule(g), ps, st)
        @test maximum(abs, fl(LenticulumCore.free_energy_gradient(s, g, ps, s1)) .- fd) < 1e-8            # no schedule
        @test maximum(abs, fl(LenticulumCore.free_energy_gradient(s, g, ps, s1; adjoint = :implicit,
                                                                  schedule = tree_schedule(g), linear_tol = 1e-8)) .- fd) < 1e-8
        replay = @test_logs (:warn, r"LP-021") match_mode = :any fl(LenticulumCore.free_energy_gradient(s, g, ps, s1;
                                                                     schedule = tree_schedule(g), nsweeps = 1))
        @test maximum(abs, replay .- fd) > 0.1       # the truncated computation's derivative (0.18–0.73 measured)
    end
    # consistent graphs are not mixed: all exact, and cuts only next to the observed x
    @test !MyceliumLearning._mixed_cuts(chain(X, X, X, X)) && !MyceliumLearning._mixed_cuts(g0)
end
