using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random

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
