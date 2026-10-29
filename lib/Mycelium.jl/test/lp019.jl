using Mycelium, LenticulumCore, LinearAlgebra, Test
using LenticulumCore: CategoricalBelief, MixtureBelief, DiracBelief, TrivialBelief, Unobserved, Polarity

# Regression tests for LP-019 (Codex's review of the core packages); each uses the failing
# input recorded in the problem note.

function _relay_free_energy(prior::PriorFactor, relay::Bool)
    b = GraphBuilder()
    variable!(b, :x)
    factor!(b, :prior, prior)
    connect!(b, :prior, prior.channel, :x)
    if relay
        variable!(b, :y)
        factor!(b, :relay, RelayFactor(:a, :b))
        connect!(b, :relay, :a, :x)
        connect!(b, :relay, :b, :y)
    end
    g = build(b)
    m, _, st, s = infer!(g, tree_schedule(g), NamedTuple(), NamedTuple())
    return first(LenticulumCore.scalar_free_energy(s, g, NamedTuple(), st)), m
end

@testset "LP-019.1 a relay leaves the Bethe free energy unchanged" begin
    p = [0.3, 0.7]
    cat = PriorFactor(:out, CategoricalBelief(p), i -> -log(p[i]))
    F0, _ = _relay_free_energy(cat, false)
    F1, m = _relay_free_energy(cat, true)
    @test F0 ≈ 0 atol = 1e-14
    @test F1 ≈ 0 atol = 1e-14                       # was H(p) = 0.6108643
    @test probabilities(m.y) ≈ p
    # Gaussian: the prior's nlogp is its own negative log-density, so F = 0 exactly.
    μ, σ² = 1.5, 2.0
    gau = PriorFactor(:out, Gaussian(μ, σ²), x -> (x[1] - μ)^2 / (2σ²) + log(2π * σ²) / 2)
    @test first(_relay_free_energy(gau, false)) ≈ 0 atol = 1e-12
    @test first(_relay_free_energy(gau, true)) ≈ 0 atol = 1e-12
end

@testset "LP-019.2 mixture pooling does not depend on operand order" begin
    mix = MixtureBelief([Gaussian(-1.0, 1.0), Gaussian(1.0, 1.0)], [0.5, 0.5])
    site = GaussianBelief([0.3], zeros(1, 1))
    # The site exp(0.3x) reweights N(±1, 1) by exp(±0.3), so the weights are ∝ (1, e^0.6).
    w = [1, exp(0.6)] ./ (1 + exp(0.6))
    @test mixture_weights(combine(mix, site)) ≈ w
    @test mixture_weights(combine(site, mix)) ≈ w   # threw "requires a proper first Gaussian belief"
    @test belief_mean(combine(site, mix).components[2]) ≈ belief_mean(combine(mix, site).components[2])
end

# A unary Gaussian source whose mean is a parameter, so a warm start can be stale.
struct LP019MovingSource <: AbstractLenticulumFactor end
LenticulumCore.islearnable(::LP019MovingSource) = true
LenticulumCore.channels(::LP019MovingSource) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supported_polarities(::LP019MovingSource) = (Polarity(x = Unobserved()),)
LenticulumCore.supports_polarity(::LP019MovingSource, p::Polarity) = p[:x] isa Unobserved
Mycelium.factor_message(::LP019MovingSource, ::Symbol, _, _, _, ps, st) = (Gaussian(ps.mu, 1.0), st)

@testset "LP-019.5 a damped tree schedule iterates to its fixed point" begin
    b = GraphBuilder(); variable!(b, :x); factor!(b, :p, LP019MovingSource()); connect!(b, :p, :x, :x)
    g = build(b); s = MessageStore(g); sched = tree_schedule(g)
    propagate!(s, g, sched, (p = (mu = 0.0,),), (p = (;),))
    r, _ = propagate!(s, g, sched, (p = (mu = 10.0,),), (p = (;),); damping = 0.5, tol = 1e-10)
    @test r.converged
    @test belief_mean(marginal(s, g, 1)) ≈ [10.0] atol = 1e-8   # was 5, reported as exact
    # A single damped sweep from a warm start is not exact, and the report must not say so.
    propagate!(s, g, sched, (p = (mu = 0.0,),), (p = (;),))
    r1, _ = propagate!(s, g, sched, (p = (mu = 10.0,),), (p = (;),); damping = 0.5, maxsweeps = 1)
    @test !r1.converged
    # Undamped, one sweep is still exact.
    r0, _ = propagate!(s, g, sched, (p = (mu = 3.0,),), (p = (;),))
    @test r0.converged && r0.sweeps == 1 && belief_mean(marginal(s, g, 1)) ≈ [3.0]
end

@testset "LP-019.6 Gaussian overlaps are translation invariant" begin
    # 1e8 is the reviewed case; far beyond it the canonical form itself loses the mean
    # (μ = η/Λ rounds to ~1e-16·|μ|), which no overlap formula can recover.
    for shift in (0.0, 1e6, 1e8)
        _, logZ = combine_with_overlap(Gaussian(shift + 1, 1.0), Gaussian(shift + 1, 1.0))
        @test logZ ≈ -log(4π) / 2                    # was −4 at 1e8
        # log N(μc − μd; 0, Σc + Σd) for two different proper Gaussians
        _, logZ2 = combine_with_overlap(Gaussian(shift + 1, 2.0), Gaussian(shift - 1, 0.5))
        @test logZ2 ≈ -(2.0)^2 / (2 * 2.5) - log(2π * 2.5) / 2
        # a mixture against proper evidence: weights ∝ wᵢ N(μᵢ − y; 0, 1 + s²)
        mix = MixtureBelief([Gaussian(shift - 1, 1.0), Gaussian(shift + 2, 1.0)], [0.5, 0.5])
        ev = Gaussian(shift + 0.25, 3.0)
        l = [-(m - 0.25)^2 / (2 * 4.0) for m in (-1.0, 2.0)]
        @test mixture_weights(combine(mix, ev)) ≈ exp.(l) ./ sum(exp.(l))
        # an improper site: weights ∝ exp(0.3 μᵢ), shifted out
        site = GaussianBelief([0.3], zeros(1, 1))
        @test mixture_weights(combine(mix, site)) ≈ [1, exp(0.9)] ./ (1 + exp(0.9)) rtol = 1e-6
    end
    # multivariate, with a rank-one improper site: against the dense formula at no shift
    c = Gaussian([1.0, -1.0], [2.0 0.3; 0.3 1.0])
    d = GaussianBelief([0.5, 0.0], [1.0 0.0; 0.0 0.0])
    @test last(combine_with_overlap(c, d)) ≈ logpartition(combine(c, d)) - logpartition(c)
end

@testset "LP-019.9 a residual schedule needs a positive task budget" begin
    @test_throws ArgumentError ResidualSchedule(maxtasks = 0)
    @test_throws ArgumentError ResidualSchedule(maxtasks = -1)
    @test_throws ArgumentError ResidualSchedule(0)
    b = GraphBuilder(); variable!(b, :x); factor!(b, :d, DataFactor(:x, 7.0)); connect!(b, :d, :x, :x)
    g = build(b)
    m, r, _, _ = infer!(g, ResidualSchedule(maxtasks = 1), NamedTuple(), NamedTuple())
    @test r.converged && m.x isa DiracBelief && m.x.value == 7.0
end
