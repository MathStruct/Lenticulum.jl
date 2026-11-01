using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random

# LP-020: the default graph parameter gradient holds the incoming messages fixed, not the factor
# beliefs, so it is not the gradient of the free energy, and alternating inference with it does not
# converge to maximum likelihood. Model: u ~ N(0, 1), z = a u + c + ε, ε ~ N(0, q), z observed.
@testset "LP-020: the fixed-message parameter gradient is not the free-energy gradient" begin
    q, z = 0.3, [1.7]
    b = GraphBuilder()
    variable!(b, :u, 1); variable!(b, :z, 1)
    factor!(b, :pu, GaussianPrior(:u, 0.0, 1.0))
    factor!(b, :lc, LinearConstraintFactor((z = 1, u = 1), 1; noise = q))
    factor!(b, :data, DataFactor(:z, z))
    connect!(b, :pu, :u, :u); connect!(b, :lc, :u, :u); connect!(b, :lc, :z, :z; direction = Absorbing())
    connect!(b, :data, :z, :z; direction = Emitting())
    g = build(b)
    ps0 = LuxCore.initialparameters(Xoshiro(0), g)
    ps(a, c) = merge(ps0, (lc = (A = (z = fill(1.0, 1, 1), u = fill(-a, 1, 1)), c = [c]),))
    nlp(a, c) = ((z[1] - c)^2 / (a^2 + q) + log(2π * (a^2 + q))) / 2           # −log p(z)
    h = 1e-6
    exact(a, c) = (A_u = -(nlp(a + h, c) - nlp(a - h, c)) / 2h, c = (nlp(a, c + h) - nlp(a, c - h)) / 2h)
    fixed(a, c) = (p = ps(a, c); (_, _, st, s) = infer!(g, tree_schedule(g), p, NamedTuple());
                   gr = LenticulumCore.free_energy_gradient(s, g, p, st; ad = nothing); (A_u = gr.lc.A.u[1], c = gr.lc.c[1]))

    a, c = 0.8, 0.4
    _, _, st, s = infer!(g, tree_schedule(g), ps(a, c), NamedTuple())
    @test first(LenticulumCore.scalar_free_energy(s, g, ps(a, c), st)) ≈ nlp(a, c)   # the Bethe free energy is exact
    # differentiating through inference (Mycelium's sample_gradient, here with nothing sampled) is exact
    sampler = GibbsSampler(g, ())
    buf = SampleBuffer(WindowFilter(1))
    record!(buf, sampler, NamedTuple(), ps(a, c), NamedTuple())
    gi = sample_gradient(buf, sampler, ps(a, c), NamedTuple())
    @test gi.lc.A.u[1] ≈ exact(a, c).A_u rtol = 1e-6
    @test gi.lc.c[1] ≈ exact(a, c).c rtol = 1e-6
    # the default fixed-message gradient is not (0.8, 0.4: −0.146 against 0.679, −0.441 against −1.383)
    @test_broken fixed(a, c).A_u ≈ exact(a, c).A_u rtol = 1e-3
    @test_broken fixed(a, c).c ≈ exact(a, c).c rtol = 1e-3
    # at the maximum-likelihood a it does not vanish, and alternating inference with it stops elsewhere
    aml = sqrt((z[1] - c)^2 - q)
    @test abs(exact(aml, c).A_u) < 1e-6
    @test abs(fixed(aml, c).A_u) > 0.5
    at = 0.8
    for _ in 1:2000
        at += 0.05 * fixed(at, c).A_u                                               # gradient descent on A_u = −a
    end
    @test abs(at - aml) > 0.4                                                       # 0.754 against 1.179
end
