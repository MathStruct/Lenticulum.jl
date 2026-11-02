using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random

# LP-021: the coupled gradient drivers (non-diagonal couplings, `adjoint = :implicit`) cut a
# `DiagonalCoupling` edge at its fixed message, the derivative LP-020 replaced in the default
# driver by the fixed-belief one. On a tree with every edge diagonal, `adjoint = :implicit` returns
# the fixed-message derivative, not the gradient of the converged free energy. With every edge on
# `ExactCoupling` it agrees. Oracle: central differences of the converged free energy.

@testset "LP-021: the implicit adjoint cuts diagonal edges at fixed messages" begin
    d, k = 3, 2
    x = randn(Xoshiro(3), d)
    function graph(coupling)
        b = GraphBuilder()
        variable!(b, :z, k); variable!(b, :x, d)
        factor!(b, :prior, GaussianPrior(:z, zeros(k), Matrix(1.0I, k, k)))
        factor!(b, :lik, LinearConstraintFactor((x = d, z = k), d; noise = 0.7))
        factor!(b, :data, DataFactor(:x, x))
        connect!(b, :prior, :z, :z; coupling); connect!(b, :lik, :z, :z; coupling)
        connect!(b, :lik, :x, :x; direction = Absorbing(), coupling)
        connect!(b, :data, :x, :x; direction = Emitting(), coupling)
        return build(b)
    end
    g, ge = graph(DiagonalCoupling()), graph(ExactCoupling())
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
    @test maximum(abs, diagonal .- fd) < 1e-8                 # the default driver: fixed beliefs (LP-020)
    @test maximum(abs, implicit_exact .- fd) < 1e-8           # differentiating through every message
    gap = maximum(abs, implicit_diag .- fd)
    @info "LP-021: implicit adjoint on diagonal edges, max |error| against central differences" gap
    @test gap > 0.1                                           # the defect, as measured (0.58 here)
    @test_broken maximum(abs, implicit_diag .- fd) < 1e-8
end
