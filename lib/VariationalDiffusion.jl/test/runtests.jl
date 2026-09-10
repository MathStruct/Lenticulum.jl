using VariationalDiffusion
using LenticulumCore
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent, Polarity
using Mycelium
using LuxCore
using LinearAlgebra
using Random
using Test

# ---------------------------------------------------------------------------
# The oracle.
#
# For Gaussian data x₀ ~ N(0, Σ) the whole diffusion model is available in closed form:
#
#     p_t = N(0, α_t²Σ + σ_t²I),   s(x,t) = -(α_t²Σ + σ_t²I)⁻¹x,   ε_θ = σ_t(α_t²Σ+σ_t²I)⁻¹x
#
# so `AnalyticEps` is a *perfectly trained* ε_θ. That makes every claim in this package
# checkable against arithmetic rather than against itself — the same trick `GaussianFactor`
# plays in the parent package, and the only way to test a diffusion model at all.
# ---------------------------------------------------------------------------
struct AnalyticEps{S,M} <: LuxCore.AbstractLuxLayer
    schedule::S
    Σ::M            # prior covariance; a scalar means isotropic
end
LuxCore.initialparameters(::AbstractRNG, ::AnalyticEps) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::AnalyticEps) = NamedTuple()

function (l::AnalyticEps{S,<:Real})(inp, ps, st) where {S}
    x, t = inp
    return (sigma(l.schedule, t) .* x ./ marginal_variance(l.schedule, t, l.Σ), st)
end
function (l::AnalyticEps{S,<:AbstractMatrix})(inp, ps, st) where {S}
    x, t = inp
    s = l.schedule
    D = alpha(s, t)^2 * l.Σ + sigma(s, t)^2 * I
    return (sigma(s, t) .* (D \ x), st)
end

const SCHED = VPSDE()
const V₀ = 1.0
const PRED = NoisePredictor(AnalyticEps(SCHED, V₀), SCHED)
const PS, ST = LuxCore.setup(Random.default_rng(), PRED)

# ∫ σ_t²/(α_t² s + σ_t²) dt by the same quadrature `calibrate_lambda` uses
function shrink_integral(s::VPSDE, v; nodes = 20000)
    a, b, h = s.tmin, 1.0, (1.0 - s.tmin) / nodes
    f(t) = sigma(s, t)^2 / marginal_variance(s, t, v)
    return ((f(a) + f(b)) / 2 + sum(f(a + i * h) for i in 1:(nodes - 1))) * h
end

_mean(v) = sum(v) / length(v)

@testset "VariationalDiffusion" begin

@testset "VP-SDE: the perturbation kernel preserves variance" begin
    s = VPSDE()
    @test s.βmin == 0.1 && s.βmax == 20.0        # Song et al.'s defaults

    # THE defining property: α_t² + σ_t² = 1, for every t. Everything else follows.
    for t in (0.0, 1e-4, 0.01, 0.137, 0.5, 0.863, 1.0)
        @test alpha(s, t)^2 + sigma(s, t)^2 ≈ 1
    end
    @test alpha(s, 0.0) == 1.0 && sigma(s, 0.0) == 0.0
    @test marginal_variance(s, 0.42, 1.0) ≈ 1     # unit data stays unit — "variance preserving"
    @test marginal_variance(s, 0.42, 4.0) > 1

    # α decreasing, σ increasing, SNR decreasing: the process only ever adds noise
    ts = 0.0:0.05:1.0
    @test issorted([alpha(s, t) for t in ts]; rev = true)
    @test issorted([sigma(s, t) for t in ts])
    @test issorted([snr(s, t) for t in ts[2:end]]; rev = true)
    @test alpha(s, 1.0) < 0.01 && sigma(s, 1.0) > 0.99   # ≈ pure noise at t=1

    # B(t) = ∫β against quadrature, and β linear in t
    for t in (0.2, 0.7, 1.0)
        n = 20000
        h = t / n
        q = h * ((beta(s, 0.0) + beta(s, t)) / 2 + sum(beta(s, i * h) for i in 1:(n - 1)))
        @test integrated_beta(s, t) ≈ q rtol = 1e-8
    end
    @test beta(s, 0.5) ≈ (s.βmin + s.βmax) / 2

    # σ is computed via expm1; the naive sqrt(1-α²) loses half its digits near 0
    @test sigma(s, 1e-8) > 0
    @test sigma(s, 1e-8) ≈ sqrt(integrated_beta(s, 1e-8)) rtol = 1e-6   # σ ≈ √B for small B

    # the SDE coefficients, for completeness
    @test drift(s, [2.0], 0.3) ≈ [-beta(s, 0.3)]
    @test diffusion(s, 0.3) ≈ sqrt(beta(s, 0.3))

    # perturb is the reparametrisation, hence linear in x₀
    @test perturb(s, [1.0], 0.3, [0.5]) ≈ alpha(s, 0.3) .* [1.0] .+ sigma(s, 0.3) .* [0.5]

    # times are drawn from [tmin, 1], never 0
    rng = MersenneTwister(1)
    smp = [sample_time(rng, s) for _ in 1:2000]
    @test all(t -> s.tmin <= t <= 1, smp)
    @test minimum(smp) < 0.05 && maximum(smp) > 0.95

    @test_throws ArgumentError VPSDE(βmin = -1.0)
    @test_throws ArgumentError VPSDE(βmin = 5.0, βmax = 1.0)
    @test_throws ArgumentError VPSDE(tmin = 1.5)
end

@testset "NoisePredictor: score and Tweedie are exact, not heuristic" begin
    # wrapping adds no parameters — ps IS the model's ps
    @test PS == LuxCore.initialparameters(Random.default_rng(), PRED.model)
    @test LuxCore.parameterlength(PRED) == 0
    @test noise_schedule(PRED) === SCHED

    for t in (0.05, 0.3, 0.77), x in ([1.3], [-2.0], [0.0])
        D = marginal_variance(SCHED, t, V₀)

        # ε_θ is what the wrapped model says
        ε̂, _ = epsilon(PRED, x, t, PS, ST)
        @test ε̂ ≈ sigma(SCHED, t) .* x ./ D

        # score = -ε_θ/σ_t, and for this data distribution that is exactly ∇log p_t
        sc, _ = score(PRED, x, t, PS, ST)
        @test sc ≈ -ε̂ ./ sigma(SCHED, t)
        @test sc ≈ -x ./ D

        # Tweedie == the closed-form linear-Gaussian posterior mean E[x₀|x_t]
        x̂₀, _ = denoise(PRED, x, t, PS, ST)
        @test x̂₀ ≈ (x .- sigma(SCHED, t) .* ε̂) ./ alpha(SCHED, t)
        @test x̂₀ ≈ alpha(SCHED, t) * V₀ / D .* x
    end

    # a perfectly trained ε_θ still has nonzero denoising loss — it predicts E[ε|x_t], not ε
    l, _ = denoising_loss(PRED, [1.0], 0.5, [0.7], PS, ST)
    @test l > 0
    @test l isa Real
end

@testset "RED-Diff: the regulariser gradient is Proposition 2" begin
    cfg = REDDiff(λ = 0.25)
    @test cfg.λ == 0.25 && cfg.adam                       # the paper's tuned λ, and Adam
    @test_throws ArgumentError REDDiff(steps = 0)
    @test_throws ArgumentError REDDiff(samples = -1)

    # λ_t = λ/SNR_t = λσ_t/α_t, and it GROWS with t — late, noisy times weigh most
    for t in (0.1, 0.5, 0.9)
        @test reddiff_weight(cfg, SCHED, t) ≈ cfg.λ * sigma(SCHED, t) / alpha(SCHED, t)
        @test reddiff_weight(cfg, SCHED, t) ≈ cfg.λ / snr(SCHED, t)
    end
    @test reddiff_weight(cfg, SCHED, 0.9) > reddiff_weight(cfg, SCHED, 0.1)

    # For Gaussian data the expectation collapses to a linear shrinkage κx, with
    #     κ = λ ∫ σ_t²/(α_t²v₀ + σ_t²) dt
    # because E_ε[ε_θ(α x + σ ε) - ε] = σ_t α_t x/D_t. Checked by heavy Monte Carlo.
    λ = 0.25
    κ = λ * shrink_integral(SCHED, V₀)
    big = REDDiff(λ = λ, samples = 200_000, rng = MersenneTwister(7))
    for x in ([1.7], [-0.5])
        g, _ = regulariser_gradient(PRED, x, big, PS, ST)
        @test g ≈ κ .* x rtol = 0.02
    end
    # and it is genuinely a gradient of a prior: it points back toward the mean
    g, _ = regulariser_gradient(PRED, [3.0], big, PS, ST)
    @test g[1] > 0
end

@testset "calibrate_lambda: λ is derivable, not merely tunable" begin
    # The unique λ making RED-Diff's implied prior equal the true Gaussian prior:
    # κ = λ∫σ²/D dt must equal 1/v₀.
    for v₀ in (0.5, 1.0, 3.0)
        λ★ = calibrate_lambda(SCHED, v₀)
        @test λ★ > 0
        @test λ★ * shrink_integral(SCHED, v₀) ≈ 1 / v₀ rtol = 1e-4
    end
    # tighter data ⇒ stronger regularisation
    @test calibrate_lambda(SCHED, 0.5) > calibrate_lambda(SCHED, 3.0)
    # the paper's 0.25 is a different quantity entirely, and this says by how much
    @test !isapprox(calibrate_lambda(SCHED, 1.0), 0.25; rtol = 0.5)
end

@testset "RED-Diff recovers the exact Gaussian posterior (the oracle)" begin
    # A soft clamp of precision ρ on a Gaussian prior N(0, v₀) has posterior mean
    #     ρ²x₀/(1/v₀ + ρ²)
    # and the RED-Diff fixed point solves  κx + ρ²(x - x₀) = 0  ⟹  x = ρ²x₀/(κ + ρ²).
    # With the calibrated λ, κ = 1/v₀ and the two coincide EXACTLY. That identity is the
    # strongest statement available about this inversion.
    λ★ = calibrate_lambda(SCHED, V₀)
    x₀ = [2.0]

    for ρ in (1.0, 2.0)
        want = ρ^2 * x₀[1] / (1 / V₀ + ρ^2)
        got = Float64[]
        for seed in 1:6
            f = DiffusionFactor((x = 1,), PRED;
                prox = REDDiff(λ = λ★, steps = 1500, samples = 64, lr = 0.1,
                               adam = false, rng = MersenneTwister(seed)))
            lens, _ = LenticulumCore.assemble(f, Polarity((x = Unobserved(),), (x = ρ,)), PS, ST)
            b, _ = LenticulumCore.invert(lens, DiracBelief(x₀), NamedTuple(), PS, ST)
            @test b isa DiracBelief          # RED-Diff's q is a point mass, by construction
            push!(got, b.value[1])
        end
        # each run is within a couple of percent; the AVERAGE is far closer, which is the
        # claim that matters — the residual is Monte-Carlo noise, not bias.
        @test all(g -> isapprox(g, want; rtol = 0.05), got)
        @test _mean(got) ≈ want rtol = 0.015
    end
end

@testset "one λ cannot fit a correlated Gaussian prior" begin
    # κ_j = λ·I(s_j) must equal 1/s_j for every eigenvalue s_j of Σ, i.e. s_j·I(s_j) must be
    # CONSTANT across j. It is not — it varies by an order of magnitude over a 16× spread —
    # so a single scalar λ can calibrate exactly one eigendirection.
    vals = [(s, s * shrink_integral(SCHED, s)) for s in (0.25, 0.5, 1.0, 2.0, 4.0)]
    prods = last.(vals)
    @test maximum(prods) / minimum(prods) > 10
    @test issorted(prods)                     # s·I(s) increases with s

    # Calibrating at s = 1 therefore OVER-regularises high-variance directions and
    # UNDER-regularises low-variance ones — an over-smoothing bias, which is exactly the
    # artefact RED-Diff is empirically criticised for.
    λ★ = calibrate_lambda(SCHED, 1.0)
    implied(s) = λ★ * shrink_integral(SCHED, s) * s        # == 1 iff correctly weighted
    @test implied(1.0) ≈ 1 rtol = 1e-4
    @test implied(4.0) > 3                                  # far too strong
    @test implied(0.25) < 0.35                              # far too weak

    # and it really is visible in the model: the anisotropic ε_θ is not a multiple of x
    Σ = [4.0 0.0; 0.0 0.25]
    pred2 = NoisePredictor(AnalyticEps(SCHED, Σ), SCHED)
    ps2, st2 = LuxCore.setup(Random.default_rng(), pred2)
    x = [1.0, 1.0]
    ε̂, _ = epsilon(pred2, x, 0.5, ps2, st2)
    @test !isapprox(ε̂[1], ε̂[2])
    # Tweedie still matches the exact correlated posterior mean, though
    x̂₀, _ = denoise(pred2, x, 0.5, ps2, st2)
    D = alpha(SCHED, 0.5)^2 * Σ + sigma(SCHED, 0.5)^2 * I
    @test x̂₀ ≈ alpha(SCHED, 0.5) .* (Σ * (D \ x))
end

@testset "DiffusionFactor: selection matrices are the polarity" begin
    f = DiffusionFactor((obs = 2, hidden = 3), PRED)
    @test statedim(f) == 5
    @test blockranges(f) == (obs = 1:2, hidden = 3:5)
    @test map(LenticulumCore.channelname, LenticulumCore.channels(f)) == (:obs, :hidden)
    @test LenticulumCore.islearnable(f)
    @test_throws ArgumentError DiffusionFactor((obs = 0,), PRED)
    @test_throws ArgumentError DiffusionFactor(NamedTuple(), PRED)

    # P = ρ_in P_in + ρ_out P_out + ρ_latent P_latent, read straight off the polarity.
    # Observed defaults to ρ = Inf, so it lands in the hard mask — the categorical cup.
    p = Polarity(; obs = Observed(), hidden = Unobserved())
    ρ, hard = precision_vector(f, p)
    @test hard == BitVector([1, 1, 0, 0, 0])
    @test ρ[3:5] == [1.0, 1.0, 1.0]                 # default_precision(Unobserved()) == 1
    @test ρ[1:2] == [0.0, 0.0]                      # infinite entries are not stored as Inf

    # a SOFT clamp is what a finite Observed precision means
    psoft = Polarity((obs = Observed(), hidden = Unobserved()), (obs = 3.0, hidden = 0.5))
    ρs, hs = precision_vector(f, psoft)
    @test !any(hs)
    @test ρs == [3.0, 3.0, 0.5, 0.5, 0.5]

    # latent channels are AutoBayes' ⟦c⟧ and the model is therefore not pure
    plat = Polarity(; obs = Observed(), hidden = Latent())
    @test LenticulumCore.supports_polarity(f, plat) == false   # nothing unobserved to infer
    p3 = DiffusionFactor((a = 1, b = 1, c = 1), PRED)
    @test LenticulumCore.supports_polarity(p3,
        Polarity(; a = Unobserved(), b = Observed(), c = Latent()))
    lens, _ = LenticulumCore.assemble(p3,
        Polarity(; a = Unobserved(), b = Observed(), c = Latent()), PS, ST)
    @test !LenticulumCore.ispure(lens.model)
    @test lens.inversion isa LenticulumCore.ProximalInversion
    @test LenticulumCore.latentspace(lens.model) == (:c,)

    # enumerated polarities: one per channel, and the predicate is strictly more permissive
    pols = LenticulumCore.supported_polarities(f)
    @test length(pols) == 2
    @test all(p -> LenticulumCore.supports_polarity(f, p), pols)
    @test !LenticulumCore.supports_polarity(f, Polarity(; obs = Observed()))

    # assemble_state lays each channel's message into its own block
    x₀ = assemble_state(f, (obs = DiracBelief([1.0, 2.0]),), TrivialBelief())
    @test x₀ == [1.0, 2.0, 0.0, 0.0, 0.0]
    @test_throws DimensionMismatch assemble_state(f, (obs = DiracBelief([1.0]),), TrivialBelief())
end

@testset "the hard clamp is exact, and inpainting fills the rest" begin
    # ρ_in = ∞ must be honoured by PROJECTION, not by a large penalty: the observed block
    # comes back bit-identical, no matter what the prox does to the rest.
    f = DiffusionFactor((obs = 1, hidden = 1), PRED;
        prox = REDDiff(λ = calibrate_lambda(SCHED, V₀), steps = 300, samples = 16,
                       lr = 0.05, adam = false, rng = MersenneTwister(5)))
    p = Polarity(; obs = Observed(), hidden = Unobserved())
    obs = [1.5]

    msg, _ = Mycelium.factor_message(f, :hidden, p, (obs = DiracBelief(obs),),
                                     TrivialBelief(), PS, ST)
    @test msg isa DiracBelief
    @test length(msg.value) == 1

    # under this ISOTROPIC prior the two coordinates are independent, so the observation is
    # uninformative about `hidden` and the reconstruction must sit at the prior mean, 0.
    @test abs(msg.value[1]) < 0.15

    # the clamped coordinate is untouched — check via the internal state the prox returns
    x, _ = VariationalDiffusion._run_prox(f, p, (obs = DiracBelief(obs),), TrivialBelief(), PS, ST)
    @test x[1] == obs[1]                       # exactly, not approximately

    # asking for a channel the factor does not have is an error, not a silent zero
    @test_throws ArgumentError Mycelium.factor_message(
        f, :nope, p, (obs = DiracBelief(obs),), TrivialBelief(), PS, ST)
end

@testset "the graded energy splits clamp from score" begin
    f = DiffusionFactor((obs = 1, hidden = 1), PRED;
        prox = REDDiff(steps = 20, samples = 4, rng = MersenneTwister(2)))
    p = Polarity((obs = Observed(), hidden = Unobserved()), (obs = 2.0, hidden = 1.0))
    @test sort(collect(keys(LenticulumCore.energyspace(f).parts))) == [:clamp, :score]

    x = [1.0, 0.5]
    y = [1.5, 0.0]
    E, _ = LenticulumCore.energy(f, x, p, y, PS, ST)
    @test sort(collect(keys(E))) == [:clamp, :score]
    # the clamp summand is ½‖P(y-x)‖², in closed form
    ρ, _ = precision_vector(f, p)
    @test E[:clamp] ≈ sum(abs2, ρ .* (y .- x)) / 2
    @test E[:clamp] ≈ (2.0^2 * 0.25 + 1.0^2 * 0.25) / 2
    @test E[:score] > 0        # a learned prior always charges something

    # the scalarisation is linear on each summand, so composition stays strict
    @test LenticulumCore.islinear(LenticulumCore.scalarisation(f).parts.clamp)

    # a Bethe contribution can be produced at all, and it is Monte-Carlo noisy by nature
    F, _ = Mycelium.local_free_energy(f, (obs = DiracBelief([1.5]),), PS, ST)
    @test sort(collect(keys(F))) == [:clamp, :score]
    @test F[:score] > 0
end

end
