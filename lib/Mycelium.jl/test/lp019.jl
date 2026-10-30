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

# Codex's review of the LP-019 fixes (2026-10-30).
import Zygote, ForwardDiff
@testset "LP-019 review: overlaps differentiate, scale and stay relative" begin
    # Zygote through the proper overlap: d/dq log ∫ N(x; 0, 1/q) N(x; 0, 1) dx = 1/(4) at q = 1
    d1 = Gaussian(0.0, 1.0)                  # built outside: only the overlap is differentiated
    f(q) = last(combine_with_overlap(GaussianBelief([0.0], fill(q, 1, 1)), d1))
    @test Zygote.gradient(f, 1.0)[1] ≈ 0.25
    @test ForwardDiff.derivative(f, 1.0) ≈ 0.25
    # finite covariances whose sum overflows: −½ log(2π · 2e308)
    @test last(combine_with_overlap(Gaussian(0.0, 1e308), Gaussian(0.0, 1e308))) ≈ -(log(2π) + log(2.0) + log(1e308)) / 2
    # an improper quadratic site far from the origin: weights ∝ wᵢ ∫ N(x; μᵢ, I) e^{-(x₁-s)²/2},
    # i.e. ∝ N(μᵢ₁ − s; 0, 2)
    for s in (0.0, 1e8)
        mix = MixtureBelief([Gaussian([s - 1, 0.0], Matrix(1.0I, 2, 2)), Gaussian([s + 2, 0.0], Matrix(1.0I, 2, 2))], [0.5, 0.5])
        site = GaussianBelief([s, 0.0], [1.0 0.0; 0.0 0.0])
        l = [-(1.0)^2 / 4, -(2.0)^2 / 4]
        w = mixture_weights(combine(mix, site))
        @test w ≈ exp.(l) ./ sum(exp.(l))                  # was [1.0, 0.368] at s = 1e8
        @test sum(w) ≈ 1
        @test mixture_weights(combine(site, mix)) ≈ w
    end
end

if get(ENV, "LENTICULUM_DEVICE_TESTS", "false") == "true"
    using JLArrays
    @testset "LP-019 review: proper overlaps on diagonal device Gaussians" begin
        JLArrays.allowscalar(false)
        for T in (Float32, Float64)
            c = GaussianBelief(JLArray(T[1, 1]), Diagonal(JLArray(T[1, 1 / 2])))      # N([1,2], diag(1,2)), η = Λμ
            d = GaussianBelief(JLArray(T[1, 1]), Diagonal(JLArray(T[1 / 2, 1 / 3])))  # N([2,3], diag(2,3))
            _, logZ = combine_with_overlap(c, d)
            # N([1,2] − [2,3]; 0, diag(3,5))
            @test logZ ≈ -(log(2π * 3) + log(2π * 5) + 1 / 3 + 1 / 5) / 2 rtol = sqrt(eps(T))
        end
    end
end

@testset "LP-019 second review: several sites, extreme diagonals" begin
    # two different improper sites against one Gaussian: log masses (2 + η)²/4 − 2
    c = Gaussian([2.0, 0.0], Matrix(1.0I, 2, 2))
    sites = MixtureBelief([GaussianBelief([0.0, 0.0], [1.0 0.0; 0.0 0.0]), GaussianBelief([2.0, 0.0], [1.0 0.0; 0.0 0.0])], [0.5, 0.5])
    l = [(2 + 0.0)^2 / 4 - 2, (2 + 2.0)^2 / 4 - 2]
    @test mixture_weights(combine(c, sites)) ≈ exp.(l) ./ sum(exp.(l))     # was [0.731, 0.269]
    @test mixture_weights(combine(sites, c)) ≈ exp.(l) ./ sum(exp.(l))
    # huge and tiny variances in one belief: −½ log det(2π · 2Σ) = −(log 2π + log 2)
    g = Gaussian(zeros(2), [1e200 0.0; 0.0 1e-200])
    @test last(combine_with_overlap(g, g)) ≈ -(log(2π) + log(2.0))
end

if get(ENV, "LENTICULUM_DEVICE_TESTS", "false") == "true"
    @testset "LP-019 second review: Zygote through a diagonal device overlap" begin
        JLArrays.allowscalar(false)
        d = GaussianBelief(JLArray([1.0, 1.0]), Diagonal(JLArray([1 / 2, 1 / 3])))   # N([2,3], diag(2,3))
        f(η, λ) = last(combine_with_overlap(GaussianBelief(η, Diagonal(λ)), d))
        gη, gλ = Zygote.gradient(f, JLArray([1.0, 1.0]), JLArray([1.0, 0.5]))
        # c = N(η/λ, 1/λ) at μ = [1, 2], Σ = [1, 2]; ∂/∂η = −(μ − μ_d)/(Σ + Σ_d)/λ, and so on
        @test Array(gη) ≈ [1 / 3, 0.4]
        @test Array(gλ) ≈ [-2 / 9, -0.48]
    end
end

@testset "LP-019 third review: a site beside proper components, means near floatmax" begin
    # c against the mixture ½N([2,0], I) + ½ site(η = [2,0], Λ = diag(1,0)):
    # proper pair log N([2,0] − [2,0]; 0, 2I) = −log 4π; site pair (2 + 2)²/4 − 2 − ½ log 2
    # (∫ N(x₁; 2, 1) e^{2x₁ − x₁²/2} dx₁ = e^{(2+η)²/4 − 2}/√2, and x₂ integrates to 1)
    c = Gaussian([2.0, 0.0], Matrix(1.0I, 2, 2))
    mix = MixtureBelief([c, GaussianBelief([2.0, 0.0], [1.0 0.0; 0.0 0.0])], [0.5, 0.5])
    l = [-log(4π), 4.0^2 / 4 - 2 - log(2) / 2]
    @test mixture_weights(combine(c, mix)) ≈ exp.(l) ./ sum(exp.(l))      # was [0.101, 0.899]
    # N(1e308, 1e308) against N(−1e308, 1e308): −(2e308)²/(2·2e308) − ½ log(2π·2e308)
    _, logZ = combine_with_overlap(Gaussian(1e308, 1e308), Gaussian(-1e308, 1e308))
    @test isfinite(logZ) && logZ ≈ -1e308
end

@testset "LP-019 fourth review: skew parts of sites, Float32 mean differences" begin
    # a site whose Λ has a skew part reads like its symmetric part diag(1, 0): two components
    # that agree on the first coordinate get equal weights
    site = GaussianBelief([1e4, 0.0], [1.0 1e-4; -1e-4 0.0])
    mix = MixtureBelief([Gaussian([1e4, 0.0], [1.0 0.0; 0.0 1.0]), Gaussian([1e4, 0.0], [1.0 0.0; 0.0 2.0])], [0.5, 0.5])
    @test mixture_weights(combine(mix, site)) ≈ [0.5, 0.5]                 # was [0.378, 0.622]
    # Float32 means 1e-6 apart with variances 1e-12: scalar convolution of the actual moments
    c = Gaussian(1.0f0, 1.0f-12)
    comps = [Gaussian(Float32(1 + k * 1e-6), 1.0f-12) for k in 0:2]
    μc, vc = only(belief_mean(c)), only(belief_cov(c))
    ref = [(m = only(belief_mean(g)); v = only(belief_cov(g)); -(Float64(μc) - Float64(m))^2 / (2 * (Float64(vc) + Float64(v))) - log(2π * (Float64(vc) + Float64(v))) / 2) for g in comps]
    got = [last(combine_with_overlap(c, g)) for g in comps]
    @test maximum(abs, got .- ref) < 1e-3                                  # was 0.027
end
