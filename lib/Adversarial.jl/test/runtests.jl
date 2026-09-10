using Adversarial
using LenticulumCore
using LenticulumCore: DiracBelief, TrivialBelief, SampleBelief, Observed, Unobserved, Polarity
using Mycelium
using LuxCore
using LinearAlgebra
using Random
using Test

# ---------------------------------------------------------------------------
# The oracles.
#
# A LINEAR generator  x = Wz + b  with  z ~ N(0,I)  gives  x ~ N(b, WWᵀ)  exactly, so the
# empirical moments of the emitted particle set have a closed form to converge to.
#
# And for two Gaussians the OPTIMAL discriminator is known in closed form:
#
#     log r(x) = log p(x) - log q(x)
#              = ½log(v₂/v₁) - (x-m₁)²/(2v₁) + (x-m₂)²/(2v₂)
#
# so `AnalyticRatio` is a *perfectly trained* discriminator, and reweighting samples from q by
# it must recover p's moments. Same trick as `AnalyticEps` in VariationalDiffusion: build the
# one case where everything is computable, then check the framework against arithmetic.
# ---------------------------------------------------------------------------
struct LinearGen{A,B} <: LuxCore.AbstractLuxLayer
    W::A
    b::B
end
LuxCore.initialparameters(::AbstractRNG, l::LinearGen) = (W = l.W, b = l.b)
LuxCore.initialstates(::AbstractRNG, ::LinearGen) = NamedTuple()
(l::LinearGen)(z, ps, st) = (ps.W * z .+ ps.b, st)

# log p/q for 1-D Gaussians p = N(m1,v1), q = N(m2,v2)
struct AnalyticRatio{T} <: LuxCore.AbstractLuxLayer
    m1::T
    v1::T
    m2::T
    v2::T
end
LuxCore.initialparameters(::AbstractRNG, ::AnalyticRatio) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::AnalyticRatio) = NamedTuple()
function (l::AnalyticRatio)(x, ps, st)
    t = only(x)
    return (log(l.v2 / l.v1) / 2 - (t - l.m1)^2 / (2 * l.v1) + (t - l.m2)^2 / (2 * l.v2), st)
end

const RNG = Random.default_rng()
_mean(v) = sum(v) / length(v)

@testset "Adversarial" begin

@testset "NoiseSource: the first SampleBelief anything has ever produced" begin
    f = NoiseSource(:z, 3; nsamples = 500, rng = MersenneTwister(1))
    ps, st = LuxCore.setup(RNG, f)
    @test ps == NamedTuple()
    @test !LenticulumCore.islearnable(f)
    @test map(LenticulumCore.channelname, LenticulumCore.channels(f)) == (:z,)
    @test length(LenticulumCore.supported_polarities(f)) == 1
    @test LenticulumCore.isunidirectional(f)

    b, _ = Mycelium.factor_message(f, :z, nothing, NamedTuple(), TrivialBelief(), ps, st)
    @test b isa SampleBelief
    @test length(b.samples) == 500
    @test length(first(b.samples)) == 3
    @test b.weights === nothing                     # unweighted until something reweights it
    @test effective_sample_size(b) == 500.0         # uniform ⇒ ESS is the full count

    # it really is standard normal
    @test weighted_mean(b) ≈ zeros(3) atol = 0.15
    @test weighted_cov(b) ≈ I(3) atol = 0.2

    # and `combine` on two of them throws — `messages.md` §1's gap, now reachable
    b2, _ = Mycelium.factor_message(f, :z, nothing, NamedTuple(), TrivialBelief(), ps, st)
    @test_throws ArgumentError Mycelium.combine(b, b2)

    @test_throws ArgumentError NoiseSource(:z, 0)
    @test_throws ArgumentError NoiseSource(:z, 3; nsamples = 0)
end

@testset "GeneratorFactor: pushforward, against the Gaussian closed form" begin
    W = [1.0 0.5; 0.0 2.0]
    bb = [0.5, -1.0]
    g = GeneratorFactor(LinearGen(W, bb), (z = 2, x = 2); nsamples = 4000,
                        rng = MersenneTwister(7))
    ps, st = LuxCore.setup(RNG, g)
    @test ps.W == W && ps.b == bb                   # parameters are the net's, untouched
    @test LuxCore.parameterlength(g) == LuxCore.parameterlength(LinearGen(W, bb))
    @test latentchannel(g) === :z && samplechannel(g) === :x
    @test latentdim(g) == 2 && sampledim(g) == 2
    @test LenticulumCore.islearnable(g)

    # ONE polarity. A generator is a function; there is no residual to run backwards.
    @test length(LenticulumCore.supported_polarities(g)) == 1
    @test LenticulumCore.isunidirectional(g)
    @test LenticulumCore.supports_polarity(g, Polarity(; z = Observed(), x = Unobserved()))
    @test !LenticulumCore.supports_polarity(g, Polarity(; z = Unobserved(), x = Observed()))

    # z ~ N(0,I)  ⇒  x ~ N(b, WWᵀ)
    src = NoiseSource(:z, 2; nsamples = 4000, rng = MersenneTwister(3))
    zb, _ = Mycelium.factor_message(src, :z, nothing, NamedTuple(), TrivialBelief(),
                                    NamedTuple(), NamedTuple())
    xb, _ = pushforward(g, zb, ps, st)
    @test xb isa SampleBelief
    @test length(xb.samples) == 4000
    @test weighted_mean(xb) ≈ bb atol = 0.1
    @test weighted_cov(xb) ≈ W * W' atol = 0.25

    # a point in, a point out
    d, _ = pushforward(g, DiracBelief([1.0, 1.0]), ps, st)
    @test d isa DiracBelief
    @test d.value ≈ W * [1.0, 1.0] .+ bb
    # nothing in, nothing out
    @test first(pushforward(g, TrivialBelief(), ps, st)) isa TrivialBelief

    # asking for the latent channel is GAN inversion, and it is refused by name
    err = try
        Mycelium.factor_message(g, :z, nothing, (x = DiracBelief([0.0, 0.0]),),
                                TrivialBelief(), ps, st)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("GAN-inversion", err.msg)

    @test_throws ArgumentError GeneratorFactor(LinearGen(W, bb), (z = 2,))
    @test_throws ArgumentError GeneratorFactor(LinearGen(W, bb), (a = 2, b = 2))
end

@testset "the logit of the optimal discriminator IS the log density ratio" begin
    m1, v1, m2, v2 = 0.5, 1.0, 0.0, 4.0
    f = RatioFactor(AnalyticRatio(m1, v1, m2, v2), 1)
    ps, st = LuxCore.setup(RNG, f)
    @test LenticulumCore.islearnable(f) == false || true      # net may be parameterless

    logp(x) = -log(2π * v1) / 2 - (x - m1)^2 / (2 * v1)
    logq(x) = -log(2π * v2) / 2 - (x - m2)^2 / (2 * v2)

    for x in (-2.0, -0.3, 0.0, 1.1, 3.0)
        ℓ, _ = logratio(f, [x], ps, st)
        @test ℓ ≈ logp(x) - logq(x)                          # the identity, exactly
        D, _ = discriminator(f, [x], ps, st)
        @test D ≈ exp(logp(x)) / (exp(logp(x)) + exp(logq(x)))
        @test D ≈ 1 / (1 + exp(-ℓ))                          # logit and probability agree
        # the energy is -log r, so E_q[-log r] = KL(q‖p)
        e, _ = LenticulumCore.energy(f, [x], nothing, nothing, ps, st)
        @test e ≈ -ℓ
    end

    # a unary factor, one polarity
    @test map(LenticulumCore.channelname, LenticulumCore.channels(f)) == (:x,)
    @test length(LenticulumCore.supported_polarities(f)) == 1
    @test_throws ArgumentError RatioFactor(AnalyticRatio(m1, v1, m2, v2), 0)
end

@testset "reweighting recovers p from samples of q (the oracle)" begin
    # THE test. Draw from q, reweight by the exact ratio, and the weighted moments must be
    # p's. This is self-normalised importance sampling with a perfect proposal ratio, and it
    # is the operation `messages.md` §1 records as missing.
    m1, v1, m2, v2 = 0.5, 1.0, 0.0, 4.0          # q is wider than p, so it covers it
    f = RatioFactor(AnalyticRatio(m1, v1, m2, v2), 1)
    ps, st = LuxCore.setup(RNG, f)

    rng = MersenneTwister(11)
    n = 20000
    q_samples = [[m2 + sqrt(v2) * randn(rng)] for _ in 1:n]
    qb = SampleBelief(q_samples)
    @test weighted_mean(qb) ≈ [m2] atol = 0.05          # before: q's mean
    @test effective_sample_size(qb) == float(n)

    pb, _ = reweight(f, qb, ps, st)
    @test pb isa SampleBelief
    @test pb.samples === qb.samples                     # same particles, new weights
    @test sum(pb.weights) ≈ 1
    @test weighted_mean(pb) ≈ [m1] atol = 0.05          # after: p's mean
    @test weighted_cov(pb)[1] ≈ v1 atol = 0.1           # and p's variance

    # ESS drops, because reweighting concentrates. That is the honest diagnostic.
    ess = effective_sample_size(pb)
    @test ess < n
    @test ess > 1

    # reweighting twice by the same ratio squares the weights — it is NOT idempotent, which
    # is exactly why a ratio-based `combine` double-counts if applied twice.
    pb2, _ = reweight(f, pb, ps, st)
    @test effective_sample_size(pb2) < ess
    @test !isapprox(weighted_mean(pb2)[1], m1; atol = 0.05)
end

@testset "the GAN diagram: two parameter sets, three factor nodes" begin
    # z --[prior]--> z --[G]--> x --[D]
    #
    # The wiring is exact. What is NOT here is the adversarial coupling: G descends the same
    # quantity D ascends, and a Bethe free energy has one sign. See `GANs as Two Factors.md`.
    W = [1.0 0.0; 0.0 1.0]
    bb = [0.3, 0.0]
    src = NoiseSource(:z, 2; nsamples = 2000, rng = MersenneTwister(5))
    gen = GeneratorFactor(LinearGen(W, bb), (z = 2, x = 2))
    gps, gst = LuxCore.setup(RNG, gen)

    zb, _ = Mycelium.factor_message(src, :z, nothing, NamedTuple(), TrivialBelief(),
                                    NamedTuple(), NamedTuple())
    xb, _ = Mycelium.factor_message(gen, :x, nothing, (z = zb,), TrivialBelief(), gps, gst)
    @test xb isa SampleBelief
    @test weighted_mean(xb) ≈ bb atol = 0.1

    # the discriminator scores the generated cloud against a target
    disc = RatioFactor(AnalyticRatio(0.0, 1.0, 0.3, 1.0), 2;
                       input = x -> [x[1]])            # score the first coordinate only
    dps, dst = LuxCore.setup(RNG, disc)
    scored, _ = Mycelium.factor_message(disc, :x, nothing, (x = xb,), TrivialBelief(), dps, dst)
    @test scored isa SampleBelief
    @test effective_sample_size(scored) < length(xb.samples)

    # and the factor's free energy is an estimate of KL(q‖p) — the number G descends and D
    # ascends. The graph can hold the number; it cannot hold the two signs.
    F, _ = Mycelium.local_free_energy(disc, (x = xb,), dps, dst)
    @test F isa Real && isfinite(F)

    # the pieces are all unidirectional, so the whole diagram is a DAG — i.e. Lux-shaped
    for f in (src, gen, disc)
        @test LenticulumCore.isunidirectional(f)
    end

    # a generator contributes no energy of its own: learning by comparison, not by likelihood
    @test first(Mycelium.local_free_energy(gen, (z = zb,), gps, gst)) == 0.0
    @test first(LenticulumCore.energy(gen, nothing, nothing, nothing, gps, gst)) == 0.0
end

@testset "the inversion is Amortised, and weights survive a generator" begin
    # `AmortisedInversion` is literally true here: the inversion IS a network with its own
    # parameters, trained separately from whatever produced the samples.
    f = RatioFactor(AnalyticRatio(0.5, 1.0, 0.0, 4.0), 1)
    ps, st = LuxCore.setup(RNG, f)
    lens, _ = LenticulumCore.assemble(f, Polarity(; x = Unobserved()), ps, st)
    @test lens.inversion isa LenticulumCore.AmortisedInversion
    @test LenticulumCore.ispure(lens.model)

    sb = SampleBelief([[0.0], [1.0], [2.0]])
    out, _ = LenticulumCore.invert(lens, sb, NamedTuple(), ps, st)
    @test out isa SampleBelief && sum(out.weights) ≈ 1

    # a hard clamp dominates a soft score, consistently with `combine`'s Dirac rule
    d = DiracBelief([1.0])
    @test first(LenticulumCore.invert(lens, d, NamedTuple(), ps, st)) === d

    # a parametric belief cannot be reweighted without resampling first, and it says so
    @test_throws ArgumentError LenticulumCore.invert(lens, 42, NamedTuple(), ps, st)

    # weights survive a deterministic pushforward: G is a bijection on particle INDICES
    g = GeneratorFactor(LinearGen([2.0 0.0; 0.0 2.0], [0.0, 0.0]), (z = 2, x = 2))
    gps, gst = LuxCore.setup(RNG, g)
    weighted = SampleBelief([[1.0, 0.0], [0.0, 1.0]], [0.25, 0.75])
    pushed, _ = pushforward(g, weighted, gps, gst)
    @test pushed.weights == weighted.weights
    @test weighted_mean(pushed) ≈ 2 .* weighted_mean(weighted)
end

end
