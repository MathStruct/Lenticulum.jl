using Lenticulum
using LenticulumCore
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved
using Mycelium
using LinearAlgebra
using Test

# ---------------------------------------------------------------------------
# The reference model:   x ~ N(μ₀, Σ₀),   y | x ~ N(Ax + b, Q),   y clamped to y₀.
#
# Everything below is checked against the closed form
#     p(y₀) = N(y₀;  Aμ₀ + b,  AΣ₀Aᵀ + Q)
# which the framework must reproduce without ever being told it.
# ---------------------------------------------------------------------------
const μ₀ = [1.0]
const Σ₀ = fill(2.0, 1, 1)
const A  = fill(3.0, 1, 1)
const bb = [0.5]
const Q  = fill(0.25, 1, 1)
const y₀ = [4.0]

log_marginal(y) = begin
    m = A * μ₀ .+ bb
    S = A * Σ₀ * A' + Q
    d = y .- m
    -length(y) * log(2π) / 2 - logdet(S) / 2 - (d' * (S \ d)) / 2
end

# exact posterior over x, by hand
const Λpost = inv(Σ₀) + A' * (Q \ A)
const ηpost = (Σ₀ \ μ₀) + A' * (Q \ (y₀ .- bb))
const mpost = Λpost \ ηpost
const Σpost = inv(Λpost)

function reference_graph()
    b = GraphBuilder()
    variable!(b, :x, 1); variable!(b, :y, 1)
    factor!(b, :prior, GaussianPrior(:x, μ₀, Σ₀))
    factor!(b, :gauss, GaussianFactor(1 => 1; noise = Q, channels = (:x, :y)))
    factor!(b, :data,  DataFactor(:y, y₀))
    # the prior's edge is Bidirectional: it emits its belief AND must hear the likelihood,
    # because its Bethe belief is b_π ∝ f_π · μ_{x→π}
    connect!(b, :prior, :x, :x; direction = Bidirectional())
    connect!(b, :gauss, :x, :x; direction = Bidirectional())
    connect!(b, :gauss, :y, :y; direction = Bidirectional())
    connect!(b, :data,  :y, :y; direction = Emitting())
    return validate(build(b))
end
const PS = (prior = NamedTuple(), gauss = (A = A, b = bb), data = NamedTuple())
const ST = (prior = NamedTuple(), gauss = NamedTuple(), data = NamedTuple())

@testset "Lenticulum" begin

@testset "GaussianBelief: canonical form makes pooling addition" begin
    a = Gaussian([0.0], fill(1.0, 1, 1))
    c = Gaussian([2.0], fill(0.5, 1, 1))
    ab = Mycelium.combine(a, c)
    # canonical parameters add
    @test ab.η ≈ a.η .+ c.η
    @test ab.Λ ≈ a.Λ .+ c.Λ
    # and that is the product of densities: precision 1 + 2 = 3, mean (0·1 + 2·2)/3
    @test belief_cov(ab)[1] ≈ 1 / 3
    @test belief_mean(ab)[1] ≈ 4 / 3

    # associative and commutative, unlike a moment-form implementation would be
    d = Gaussian([-1.0], fill(3.0, 1, 1))
    @test Mycelium.combine(Mycelium.combine(a, c), d).η ≈
          Mycelium.combine(a, Mycelium.combine(c, d)).η
    @test Mycelium.combine(a, c).Λ ≈ Mycelium.combine(c, a).Λ

    # TrivialBelief is the additive identity; a Dirac still dominates (ρ_in = ∞)
    @test Mycelium.combine(TrivialBelief(), a) === a
    @test Mycelium.combine(uninformative(1), a).η ≈ a.η
    @test Mycelium.combine(DiracBelief([7.0]), a) === DiracBelief([7.0])

    # improper beliefs are representable and are NOT an error
    imp = GaussianBelief([0.0], zeros(1, 1))
    @test !isproper(imp)
    @test isproper(a)
    @test_throws ArgumentError belief_mean(imp)

    # entropy, logdensity, KL against closed forms
    g = Gaussian([0.0], fill(4.0, 1, 1))
    @test Mycelium.variable_entropy(g) ≈ (log(2π * ℯ * 4)) / 2
    @test Mycelium.belief_logdensity(g, [0.0]) ≈ -log(2π * 4) / 2
    @test kl_divergence(g, g) ≈ 0 atol = 1e-12
    @test kl_divergence(Gaussian([1.0], fill(1.0, 1, 1)), Gaussian([0.0], fill(1.0, 1, 1))) ≈ 0.5

    # damping is a convex combination in canonical parameters
    @test Mycelium.can_damp(a, c)
    @test Mycelium.damp(a, c, 0.5).Λ ≈ (a.Λ .+ c.Λ) ./ 2
end

@testset "GaussianFactor: the factor interface, nothing stubbed" begin
    f = GaussianFactor(2 => 3; noise = 0.5)
    ps, st = LuxCore.setup(Random.default_rng(), f)
    @test size(ps.A) == (3, 2) && size(ps.b) == (3,)
    @test LuxCore.parameterlength(f) == 6 + 3
    @test LenticulumCore.islearnable(f)
    @test !LenticulumCore.isunidirectional(f)      # bidirectional: it is a relation
    @test length(LenticulumCore.supported_polarities(f)) == 2
    @test !Mycelium.issink(f)

    # both polarities are supported; neither is privileged
    fwd = LenticulumCore.Polarity(; x = Observed(), y = Unobserved())
    bwd = LenticulumCore.Polarity(; x = Unobserved(), y = Observed())
    @test LenticulumCore.supports_polarity(f, fwd)
    @test LenticulumCore.supports_polarity(f, bwd)

    # assemble really produces a lens (the operation with no Lux counterpart)
    lens, _ = LenticulumCore.assemble(f, fwd, ps, st)
    @test lens isa LenticulumCore.BayesianLens
    @test lens.inversion isa LenticulumCore.ExactInversion
    @test LenticulumCore.ispure(lens.model)

    # the pointwise vector energy is the residual, in the noise space
    x, y = [1.0, 2.0], [0.0, 0.0, 0.0]
    r, _ = LenticulumCore.energy(f, x, nothing, y, ps, st)
    @test r ≈ y .- ps.A * x .- ps.b
    @test length(r) == 3
end

@testset "messages are exact in both directions" begin
    f = GaussianFactor(1 => 1; noise = Q)
    ps, st = (A = A, b = bb), NamedTuple()
    fwd = LenticulumCore.Polarity(; x = Observed(), y = Unobserved())
    bwd = LenticulumCore.Polarity(; x = Unobserved(), y = Observed())

    # forward: pushforward of a Gaussian prior,  N(Aμ+b, AΣAᵀ+Q)
    μf, _ = Mycelium.factor_message(f, :y, fwd, (; x = Gaussian(μ₀, Σ₀)), TrivialBelief(), ps, st)
    @test belief_mean(μf) ≈ A * μ₀ .+ bb
    @test belief_cov(μf) ≈ A * Σ₀ * A' + Q

    # backward: the LIKELIHOOD  N⁻¹(AᵀQ⁻¹(y-b), AᵀQ⁻¹A) — not the posterior
    μb, _ = Mycelium.factor_message(f, :x, bwd, (; y = DiracBelief(y₀)), TrivialBelief(), ps, st)
    @test μb.Λ ≈ A' * (Q \ A)
    @test μb.η ≈ A' * (Q \ (y₀ .- bb))

    # ...and combining it with the prior gives exactly the posterior.
    # This is the precise relation between BP's message and AutoBayes's c†_π.
    post = Mycelium.combine(Gaussian(μ₀, Σ₀), μb)
    @test post.Λ ≈ Λpost
    @test belief_mean(post) ≈ mpost

    # LenticulumCore.invert returns the POSTERIOR (AutoBayes semantics)
    lens, _ = LenticulumCore.assemble(f, bwd, ps, st)
    inv_post, _ = LenticulumCore.invert(lens, Gaussian(μ₀, Σ₀), (; y = DiracBelief(y₀)), ps, st)
    @test inv_post.Λ ≈ Λpost
    @test belief_mean(inv_post) ≈ mpost

    # a message from an uninformative input carries no information
    μ0, _ = Mycelium.factor_message(f, :y, fwd, (; x = TrivialBelief()), TrivialBelief(), ps, st)
    @test μ0 isa TrivialBelief
end

@testset "message passing recovers the exact posterior" begin
    g = reference_graph()
    @test istree(g)
    @test !isdag(g)
    sched = tree_schedule(g)
    marg, report, st, store = infer!(g, sched, PS, ST)
    @test report.converged

    @test marg.x isa GaussianBelief
    @test marg.x.Λ ≈ Λpost
    @test belief_mean(marg.x) ≈ mpost
    @test belief_cov(marg.x) ≈ Σpost
    @test marg.y === DiracBelief(y₀)          # the data clamp dominates
end

@testset "AutoBayes Remark 24: F^{cπ}(∗,y) = −log p_{c*π}(y)" begin
    # THE test. The graph is never told the marginal likelihood; it composes local
    # energies and entropies and the counting correction, and the answer comes out.
    g = reference_graph()
    marg, report, st, store = infer!(g, tree_schedule(g), PS, ST)
    total, st = LenticulumCore.scalar_free_energy(store, g, PS, st)
    @test total ≈ -log_marginal(y₀)[1]

    # and it is a real number, not something that vanishes trivially
    @test total > 2.0
end

@testset "the graded energy exhibits fit / complexity / entropy" begin
    g = reference_graph()
    _, _, st, store = infer!(g, tree_schedule(g), PS, ST)
    F, st = factor_free_energies(store, g, PS, st)

    @test sort(collect(keys(F))) == [:data, :gauss, :prior]
    @test F[:data] == 0.0
    for n in (:prior, :gauss)
        @test sort(collect(keys(F[n]))) == [:complexity, :fit, :negentropy]
    end
    # the complexity summands are exactly the log-normalisers, independent of the data
    @test F[:gauss][:complexity] ≈ logdet(2π .* Q) / 2
    @test F[:prior][:complexity] ≈ logdet(2π .* Σ₀) / 2
    # both factors see the same posterior, hence the same entropy
    Hq = log(2π * ℯ * Σpost[1]) / 2
    @test F[:gauss][:negentropy] ≈ -Hq
    @test F[:prior][:negentropy] ≈ -Hq

    # the counting correction adds ONE copy of H back (x has degree 2)
    V = variable_corrections(store, g)
    @test counting_numbers(g).x == -1
    @test V[:x] ≈ Hq                     # (1 - d_x)·(−H_x) = (−1)(−H) = +H
    @test V[:y] ≈ 0.0                    # y is clamped
    @test total_counting_number(g) == euler_characteristic(g) == 1
end

@testset "the two energies differ by exactly ½·tr(Q⁻¹ Cov r)" begin
    # Scalar and Multivariate Energy.md §5, in closed form on a real model.
    g = reference_graph()
    _, _, st, store = infer!(g, tree_schedule(g), PS, ST)
    f = factornode(g, :gauss).factor
    msgs = messages_into(store, g, g.factor_index[:gauss])
    r̄, Cr, H = residual_statistics(f, msgs, PS.gauss)

    F, _ = factor_free_energies(store, g, PS, st)
    exact = F[:gauss][:fit]                        # E[½ rᵀQ⁻¹r]  — mean SQUARED residual
    lax   = (r̄' * (Q \ r̄)) / 2                     # ½ E[r]ᵀQ⁻¹E[r] — SQUARED MEAN residual
    gap   = tr(Q \ Cr) / 2                         # ½ tr(Q⁻¹ Cov r)

    @test exact ≈ lax + gap
    @test gap > 0                                  # Jensen: the multivariate one is a lower bound
    @test lax <= exact
    # the covariance of the residual is A Σ_post Aᵀ — posterior uncertainty in x, pushed through
    @test Cr ≈ A * Σpost * A'
    @test gap ≈ tr(Q \ (A * Σpost * A')) / 2

    # the default scalarisation is LINEAR, so composition is strict (§5 case 1)
    @test LenticulumCore.islinear(LenticulumCore.scalarisation(f))
end

@testset "the same factor, run the other way" begin
    # compact closure: swap which end is clamped and the factor still works,
    # with no change to the factor itself.
    b = GraphBuilder()
    variable!(b, :x, 1); variable!(b, :y, 1)
    factor!(b, :dx,    DataFactor(:x, [2.0]))
    factor!(b, :gauss, GaussianFactor(1 => 1; noise = Q, channels = (:x, :y)))
    connect!(b, :dx, :x, :x; direction = Emitting())
    connect!(b, :gauss, :x, :x; direction = Bidirectional())
    connect!(b, :gauss, :y, :y; direction = Bidirectional())
    g = validate(build(b))
    ps = (dx = NamedTuple(), gauss = (A = A, b = bb))
    st = (dx = NamedTuple(), gauss = NamedTuple())
    marg, report, _, _ = infer!(g, tree_schedule(g), ps, st)
    # x clamped ⇒ y is the pushforward N(A·2 + b, Q)
    @test belief_mean(marg.y) ≈ A * [2.0] .+ bb
    @test belief_cov(marg.y) ≈ Q
end

end
