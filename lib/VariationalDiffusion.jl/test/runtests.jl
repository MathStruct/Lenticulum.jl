using VariationalDiffusion
using LenticulumCore
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent, Polarity
using Mycelium
using LuxCore
using LinearAlgebra
using Random
using Test
using Statistics: mean, std
using Lux: Lux, Chain, Dense, swish
using DifferentiationInterface: DifferentiationInterface
using ADTypes: AutoZygote, AutoForwardDiff, AutoEnzyme
import Zygote, ForwardDiff, Enzyme

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

# a ring of narrow Gaussians: a circle relation with an exact score (implicit-inference tests)
const RING_J = 48
const RING_μ = vcat(cos.(range(0, 2π; length = RING_J + 1)[1:RING_J])', sin.(range(0, 2π; length = RING_J + 1)[1:RING_J])')
const RING = NoisePredictor(GaussianMixtureEps(SCHED, RING_μ; s = 0.05), SCHED)
const RPS, RST = LuxCore.setup(Random.default_rng(), RING)
const HARDX = [Inf, 0.0]                      # x clamped, y free
_fd(f, x, i; h = 1e-6) = (f(x .+ h .* (eachindex(x) .== i)) - f(x .- h .* (eachindex(x) .== i))) / (2h)

# a small Lux denoiser (MLP on [x; time embedding]) for the AD-extension tests
const FREQ = [0.5, 1.0, 2.0, 4.0]
_emb(t) = vcat(sin.((2π * t) .* FREQ), cos.((2π * t) .* FREQ))
_mlp_input(x, t) = vcat(x, _emb(t))
const MLP = Chain(Dense(10 => 32, swish), Dense(32 => 32, swish), Dense(32 => 2))
_f64(x::AbstractArray) = Float64.(x); _f64(x::NamedTuple) = map(_f64, x); _f64(x) = x

# -log p_t of a Gaussian mixture as an AD-friendly Lux layer: an energy network whose
# ε = σ∇E must reproduce GaussianMixtureEps exactly
struct MixEnergy{M} <: LuxCore.AbstractLuxLayer
    μ₀::M
    s::Float64
end
LuxCore.initialparameters(::AbstractRNG, l::MixEnergy) = (μ = copy(l.μ₀),)
function (l::MixEnergy)(inp, ps, st)
    x, t = inp
    a, sg = alpha(SCHED, t), sigma(SCHED, t)
    v = a^2 * l.s^2 + sg^2
    ℓ = vec(-sum(abs2, x .- a .* ps.μ; dims = 1)) ./ (2v)
    m = maximum(ℓ)
    return (-(m + log(sum(exp.(ℓ .- m)) / length(ℓ)) - length(x) / 2 * log(2π * v)), st)
end
_batch_input(x, t) = vcat(x, vcat(sin.((2π .* t) .* FREQ), cos.((2π .* t) .* FREQ)) .* ones(eltype(x), 1, size(x, 2)))
const ENERGY_MLP = Chain(Dense(10 => 32, swish), Dense(32 => 32, swish), Dense(32 => 1))

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


# ---------------------------------------------------------------------------
# Implicit inference and its backward pass, against closed-form oracles.
# See `implicit.md` and `Implicit Diffusion Learners.md`.
# ---------------------------------------------------------------------------

@testset "closed-form mixture: Jacobian and parameter VJP are exact" begin
    x, t = [0.3, -0.7], 0.02
    Jε = epsilon_jacobian(RING, x, t, RPS, RST)
    @test Jε ≈ Jε'                                         # −σ_t × a Hessian: symmetric
    for i in 1:2
        @test Jε[:, i] ≈ [_fd(z -> first(epsilon(RING, z, t, RPS, RST))[r], x, i) for r in 1:2] rtol = 1e-6
    end
    w = [0.4, -1.1]
    G = epsilon_vjp_params(RING, x, t, RPS, RST, w).μ
    for (a, b) in ((1, 1), (2, 7), (1, 30))
        E = zeros(size(RING_μ)); E[a, b] = 1e-6
        f(δ) = dot(w, first(epsilon(RING, x, t, (μ = RPS.μ .+ δ .* E ./ 1e-6,), RST)))
        @test G[a, b] ≈ (f(1e-6) - f(-1e-6)) / 2e-6 rtol = 1e-5 atol = 1e-10
    end
end

@testset "the deterministic field is the gradient of a smoothed log-density" begin
    m = ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2; samples = 2))
    Φ(z) = -sum(m.nodes.w[k] * VariationalDiffusion._weight(m, m.nodes.t[k]) * sigma(SCHED, m.nodes.t[k]) /
                alpha(SCHED, m.nodes.t[k]) * mixture_logdensity(RING.model, VariationalDiffusion._node_input(m, z, k), m.nodes.t[k], RPS)
                for k in eachindex(m.nodes.t))      # antithetic nodes: no linear tilt
    z = [0.3, 0.8]
    g = first(prior_field(m, z, RPS, RST))
    @test g ≈ [_fd(Φ, z, i) for i in 1:2] rtol = 1e-6
    Jg = prior_jacobian(m, z, RPS, RST)
    @test Jg ≈ Jg'
end

@testset "implicit inference recovers both branches of the circle" begin
    m = ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2; samples = 8))
    up, _ = implicit_infer(m, [0.6, 0.5], HARDX, RPS, RST)
    dn, _ = implicit_infer(m, [0.6, -0.5], HARDX, RPS, RST)
    @test up.converged && up.stable && dn.converged && dn.stable
    @test up.z[2] ≈ 0.8 atol = 0.03
    @test dn.z[2] ≈ -0.8 atol = 0.03
    @test up.z[1] == 0.6                                   # the hard clamp is exact
    # near the branch point the smoothed relation has ONE root: both starts land on it
    a, _ = implicit_infer(m, [0.99, 0.5], HARDX, RPS, RST)
    b, _ = implicit_infer(m, [0.99, -0.5], HARDX, RPS, RST)
    @test a.converged && b.converged
    @test a.z[2] ≈ b.z[2] atol = 1e-6
    # outside the support inference still answers, with a point near the ridge ("closest point")
    o, _ = implicit_infer(m, [1.05, 0.0], HARDX, RPS, RST)
    @test o.converged && o.stable
    @test abs(o.z[2]) < 0.2
end

@testset "smoothing bias: the relation is the ridge of a smoothed density" begin
    rad(thi) = abs(first(implicit_infer(ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2;
                   levels = range(0.002, thi; length = 6), samples = 4)), [0.0, 0.9], HARDX, RPS, RST)).z[2])
    r = [rad(t) for t in (0.01, 0.05, 0.2, 1.0)]
    @test issorted(r; rev = true)                          # more smoothing, more shrinkage
    @test r[1] > 0.99 && r[end] < 0.05                     # RED-Diff's full range collapses the circle
    # single noise-free level: the mode radius is R − (s² + σ²/α²)/(2R) to first order
    t = 0.03
    m = ImplicitDiffusion(RING, noisefree_nodes(2, t))
    sol, _ = implicit_infer(m, [0.0, 0.9], HARDX, RPS, RST)
    @test sol.z[2] ≈ 1 - (0.05^2 + (sigma(SCHED, t) / alpha(SCHED, t))^2) / 2 atol = 2e-3
end

@testset "deterministic relaxation: a fixed point of the Tweedie denoiser (a DEQ)" begin
    for t in (0.005, 0.03)
        m = ImplicitDiffusion(RING, noisefree_nodes(2, t))
        sol, _ = implicit_infer(m, [0.6, 0.7], HARDX, RPS, RST)
        x̂, _ = denoise(RING, alpha(SCHED, t) .* sol.z, t, RPS, RST)
        @test sol.converged
        @test x̂[2] ≈ sol.z[2] atol = 1e-8
    end
end

@testset "Gaussian prior: the root and its derivative in closed form" begin
    # N(0, v₀) data, antithetic nodes: g(z) = κ z exactly, so with a soft anchor ρ on y the root is
    # y★ = ρ² y₀ / (κ + ρ²) and dy★/dy₀ = ρ²/(κ + ρ²).
    m = ImplicitDiffusion(PRED, field_nodes(Xoshiro(2), 1; samples = 3))
    κ = sum(m.nodes.w[k] * VariationalDiffusion._weight(m, m.nodes.t[k]) * alpha(SCHED, m.nodes.t[k]) *
            sigma(SCHED, m.nodes.t[k]) / marginal_variance(SCHED, m.nodes.t[k], V₀) for k in eachindex(m.nodes.t))
    ρ, y₀ = [0.7], [1.3]
    sol, _ = implicit_infer(m, y₀, ρ, PS, ST)
    @test sol.z[1] ≈ ρ[1]^2 * y₀[1] / (κ + ρ[1]^2) rtol = 1e-8
    b = implicit_pullback(m, sol, y₀, ρ, [1.0], PS, ST)
    @test b.z₀[1] ≈ ρ[1]^2 / (κ + ρ[1]^2) rtol = 1e-6
end

@testset "the adjoint gradient equals finite differences of the solve" begin
    m = ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2; samples = 2))
    ℓ(z) = (z[2] - 0.7)^2 / 2
    solve(z₀, ρ, ps) = first(implicit_infer(m, z₀, ρ, ps, RST; tol = 1e-12)).z
    # hard clamp on x: gradients w.r.t. the clamped input and a mixture mean
    z₀ = [0.6, 0.5]
    sol, _ = implicit_infer(m, z₀, HARDX, RPS, RST)
    b = implicit_pullback(m, sol, z₀, HARDX, [0.0, sol.z[2] - 0.7], RPS, RST)
    @test b.z₀[1] ≈ _fd(q -> ℓ(solve(q, HARDX, RPS)), z₀, 1) rtol = 1e-5
    E = zeros(size(RING_μ)); E[2, 8] = 1.0
    fμ(δ) = ℓ(solve(z₀, HARDX, (μ = RPS.μ .+ δ .* E,)))
    @test b.ps.μ[2, 8] ≈ (fμ(1e-6) - fμ(-1e-6)) / 2e-6 rtol = 1e-5
    # soft clamps: gradients w.r.t. the soft targets and the precisions themselves
    ρs = [3.0, 0.5]
    sol2, _ = implicit_infer(m, z₀, ρs, RPS, RST)
    b2 = implicit_pullback(m, sol2, z₀, ρs, [0.0, sol2.z[2] - 0.7], RPS, RST)
    for i in 1:2
        @test b2.z₀[i] ≈ _fd(q -> ℓ(solve(q, ρs, RPS)), z₀, i) rtol = 1e-5
        @test b2.ρ[i] ≈ _fd(q -> ℓ(solve(z₀, q, RPS)), ρs, i) rtol = 1e-4 atol = 1e-10
    end
    # no IFT gradient at a point that is not a root
    nc, _ = implicit_infer(m, [0.99, 0.05], HARDX, RPS, RST; maxiters = 3)
    @test_throws ArgumentError implicit_pullback(m, nc, [0.99, 0.05], HARDX, [0.0, 1.0], RPS, RST)
end

@testset "learning a relation by backpropagating through inference" begin
    # start from the circle; move the mixture means so that inference x ↦ y reproduces y = x² − 0.5
    m = ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2; samples = 2))
    target(x) = x^2 - 0.5
    xs = collect(range(-0.8, 0.8; length = 9))
    μ = copy(RPS.μ); yw = fill(-0.5, length(xs)); mo = zero(μ); v = zero(μ)
    loss(μμ) = sum((first(implicit_infer(m, [x, -0.5], HARDX, (μ = μμ,), RST)).z[2] - target(x))^2 for x in xs) / length(xs)
    L0 = loss(μ)
    for ep in 1:60
        G = zero(μ)
        for (i, x) in enumerate(xs)
            sol, _ = implicit_infer(m, [x, yw[i]], HARDX, (μ = μ,), RST)
            (sol.converged && sol.stable) || continue
            yw[i] = sol.z[2]
            G .+= implicit_pullback(m, sol, [x, yw[i]], HARDX, [0.0, sol.z[2] - target(x)], (μ = μ,), RST).ps.μ
        end
        mo = 0.9 .* mo .+ 0.1 .* G; v = 0.999 .* v .+ 0.001 .* G .^ 2
        μ .-= 0.01 .* (mo ./ (1 - 0.9^ep)) ./ (sqrt.(v ./ (1 - 0.999^ep)) .+ 1e-8)
    end
    @test loss(μ) < L0 / 50
end


@testset "DiffusionFactor with the implicit solver as its inversion" begin
    prox = ImplicitProx(field_nodes(Xoshiro(1), 2; samples = 8))
    f = DiffusionFactor((x = 1, y = 1), RING; prox = prox)
    @test_throws DimensionMismatch DiffusionFactor((x = 1, y = 2), RING; prox = prox)
    p = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.0))   # pure conditional
    lens, _ = LenticulumCore.assemble(f, p, RPS, RST)
    # the prior point is the warm start, and it selects the branch
    up, _ = LenticulumCore.invert(lens, DiracBelief([0.5]), (x = DiracBelief([0.6]),), RPS, RST)
    dn, _ = LenticulumCore.invert(lens, DiracBelief([-0.5]), (x = DiracBelief([0.6]),), RPS, RST)
    @test up isa DiracBelief
    @test up.value[1] ≈ 0.78 atol = 0.03
    @test dn.value[1] ≈ -0.78 atol = 0.03
    # same answer as the bare solver, and a report
    sol, _ = implicit_solution(f, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), RPS, RST)
    @test sol.converged && sol.stable
    @test sol.z[2] == up.value[1]
    ref, _ = implicit_infer(ImplicitDiffusion(RING, prox.nodes), [0.6, 0.5], [Inf, 0.0], RPS, RST)
    @test ref.z ≈ sol.z
    # the message is the same Dirac
    msg, _ = Mycelium.factor_message(f, :y, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), RPS, RST)
    @test msg.value ≈ up.value
    # the free energy is deterministic now (RED-Diff's is Monte-Carlo noisy)
    F1, _ = Mycelium.local_free_energy(f, (x = DiracBelief([0.6]),), RPS, RST)
    F2, _ = Mycelium.local_free_energy(f, (x = DiracBelief([0.6]),), RPS, RST)
    @test F1[:clamp] == F2[:clamp] && F1[:score] == F2[:score]
    # the factor-level pullback agrees with finite differences of the factor's inversion
    ℓ(xin, ρy) = begin
        q = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = ρy))
        b, _ = LenticulumCore.invert(first(LenticulumCore.assemble(f, q, RPS, RST)), DiracBelief([0.5]),
                                     (x = DiracBelief([xin]),), RPS, RST)
        (b.value[1] - 0.7)^2 / 2
    end
    psoft = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.3))
    sol2, _ = implicit_solution(f, psoft, (x = DiracBelief([0.6]),), DiracBelief([0.5]), RPS, RST)
    g = implicit_factor_pullback(f, psoft, (x = DiracBelief([0.6]),), DiracBelief([0.5]),
                                 (y = [sol2.z[2] - 0.7],), RPS, RST)
    h = 1e-6
    @test g.inputs.x[1] ≈ (ℓ(0.6 + h, 0.3) - ℓ(0.6 - h, 0.3)) / 2h rtol = 1e-5
    @test g.precisions.y ≈ (ℓ(0.6, 0.3 + h) - ℓ(0.6, 0.3 - h)) / 2h rtol = 1e-5
    @test g.precisions.x == 0                   # a hard clamp has no differentiable precision
    @test size(g.ps.μ) == size(RING_μ)
    # a RED-Diff factor has no implicit solution to report
    @test_throws ArgumentError implicit_solution(DiffusionFactor((x = 1, y = 1), RING), p,
                                                 (x = DiracBelief([0.6]),), DiracBelief([0.5]), RPS, RST)
end


# ---------------------------------------------------------------------------
# AD-backend-agnostic derivatives for a learned (Lux) predictor, via DifferentiationInterface.
# ---------------------------------------------------------------------------

@testset "a Lux predictor: parameter VJP and input Jacobian through any AD backend" begin
    ps0, st = Lux.setup(Xoshiro(0), MLP)
    ps = _f64(ps0)
    x, t, w, h = [0.3, -0.7], 0.03, [0.4, -1.1], 1e-6
    # without a backend: a clear error, and finite differences for the Jacobian
    plain = NoisePredictor(MLP, SCHED; input = _mlp_input)
    @test_throws ArgumentError epsilon_vjp_params(plain, x, t, ps, st, w)
    Jfd = epsilon_jacobian(plain, x, t, ps, st)
    bumped(δ) = merge(ps, (layer_1 = (weight = (W = copy(ps.layer_1.weight); W[7] += δ; W), bias = ps.layer_1.bias),))
    for ad in (AutoZygote(), AutoForwardDiff(), AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
        pred = NoisePredictor(MLP, SCHED; input = _mlp_input, ad = ad)
        g = epsilon_vjp_params(pred, x, t, ps, st, w)
        f(δ) = dot(w, first(epsilon(pred, x, t, bumped(δ), st)))
        @test g.layer_1.weight[7] ≈ (f(h) - f(-h)) / 2h rtol = 1e-5
        @test size(g.layer_3.weight) == size(ps.layer_3.weight)        # shaped like ps
        @test epsilon_jacobian(pred, x, t, ps, st) ≈ Jfd rtol = 1e-5
        # a batch-style input returns a 2×1 column for a vector x; w stays a vector
        col = NoisePredictor(MLP, SCHED; input = (x, t) -> reshape(_mlp_input(x, t), :, 1), ad = ad)
        @test epsilon_vjp_params(col, x, t, ps, st, w).layer_1.weight[7] ≈ g.layer_1.weight[7] rtol = 1e-10
    end
end

@testset "implicit inference and its adjoint through a Lux network" begin
    ps0, st = Lux.setup(Xoshiro(1), MLP)
    ps = _f64(ps0)
    pred = NoisePredictor(MLP, SCHED; input = _mlp_input, ad = AutoZygote())
    m = ImplicitDiffusion(pred, field_nodes(Xoshiro(1), 2; samples = 2))
    ρ = [Inf, 1.0]                     # an untrained net: anchor y so the root is well posed
    J = prior_jacobian(m, [0.6, 0.5], ps, st)
    @test norm(J - J') > 1e-6          # a learned field is not a gradient: the transpose matters
    sol, _ = implicit_infer(m, [0.6, 0.5], ρ, ps, st; tol = 1e-12)
    @test sol.converged
    b = implicit_pullback(m, sol, [0.6, 0.5], ρ, [0.0, sol.z[2] - 0.7], ps, st)
    L(z₀, p) = (first(implicit_infer(m, z₀, ρ, p, st; tol = 1e-12)).z[2] - 0.7)^2 / 2
    h = 1e-6
    @test b.z₀[1] ≈ (L([0.6 + h, 0.5], ps) - L([0.6 - h, 0.5], ps)) / 2h rtol = 1e-5
    W(δ) = merge(ps, (layer_2 = (weight = (A = copy(ps.layer_2.weight); A[5] += δ; A), bias = ps.layer_2.bias),))
    @test b.ps.layer_2.weight[5] ≈ (L([0.6, 0.5], W(h)) - L([0.6, 0.5], W(-h))) / 2h rtol = 1e-4 atol = 1e-10
end

# ---------------------------------------------------------------------------
# Proximal diffusion models (ProxDM), against the exact proximal operator of a mixture.
# ---------------------------------------------------------------------------
@testset "kernel baseline: a Gaussian KDE as a relation" begin
    rng = Xoshiro(3)
    data = randn(rng, 2, 400)
    h = kde_bandwidth(data)
    @test 0.15 < h < 0.7                     # the scale of Silverman's rule for 400 Gaussian samples in 2-D
    kde = kde_predictor(SCHED, data; bandwidth = h)
    pk, sk = LuxCore.setup(Xoshiro(0), kde)
    @test pk.μ == data
    ref = NoisePredictor(GaussianMixtureEps(SCHED, data; s = h), SCHED)
    x = [0.3, -0.2]
    @test first(epsilon(kde, x, 0.2, pk, sk)) == first(epsilon(ref, x, 0.2, pk, sk))

    # noisy samples of the unit circle: the KDE ridge answers "y given x" with both branches
    φ = 2π .* rand(rng, 300)
    circ = vcat(cos.(φ)', sin.(φ)') .+ 0.02 .* randn(rng, 2, 300)
    hc = kde_bandwidth(circ)
    @test hc < 0.2
    kc = kde_predictor(SCHED, circ; bandwidth = hc)
    pc, sc = LuxCore.setup(Xoshiro(0), kc)
    roots, _ = implicit_roots(ImplicitDiffusion(kc, field_nodes(Xoshiro(1), 2; samples = 4)), [0.6, 0.0], [Inf, 0.0], pc, sc)
    @test length(roots) == 2
    @test sort([r.z[2] for r in roots]) ≈ [-0.8, 0.8] atol = 0.06
end

@testset "weighted mixtures and an online, forgetting kernel model" begin
    # weights: a zero weight removes a component; equal weights are the default
    μ = [0.0 3.0; 0.0 0.0]
    one = GaussianMixtureEps(SCHED, μ[:, 1:1]; s = 0.4)
    two = GaussianMixtureEps(SCHED, μ; s = 0.4, weights = [1.0, 0.0])
    x = [0.3, -0.2]
    @test mixture_logdensity(two, x, 0.3, (μ = μ,)) ≈ mixture_logdensity(one, x, 0.3, (μ = μ[:, 1:1],)) atol = 1e-12
    @test two((x, 0.3), (μ = μ,), NamedTuple())[1] ≈ one((x, 0.3), (μ = μ[:, 1:1],), NamedTuple())[1] atol = 1e-12
    @test GaussianMixtureEps(SCHED, μ).logw ≈ log.([0.5, 0.5])
    @test_throws ArgumentError GaussianMixtureEps(SCHED, μ; weights = [1.0])

    # no forgetting and no merging: exactly the batch KDE
    data = randn(Xoshiro(5), 2, 40)
    ok = observe!(OnlineKDE(2; bandwidth = 0.3, merge_radius = 0.0), data)
    a, b = kde_predictor(SCHED, ok), kde_predictor(SCHED, data; bandwidth = 0.3)
    pa, sa = LuxCore.setup(Xoshiro(0), a); pb, sb = LuxCore.setup(Xoshiro(0), b)
    @test first(epsilon(a, x, 0.3, pa, sa)) ≈ first(epsilon(b, x, 0.3, pb, sb)) atol = 1e-14

    # forgetting is geometric; merging keeps the weighted mean; the budget merges, never drops mass
    f = observe!(OnlineKDE(1; bandwidth = 0.1, forget = 0.5), [0.0 5.0 10.0])
    @test f.weights ≈ [0.25, 0.5, 1.0]
    g = observe!(OnlineKDE(1; bandwidth = 0.1, merge_radius = 0.2), [1.0 1.1])
    @test size(g.centres, 2) == 1 && g.centres[1] ≈ 1.05 && g.weights == [2.0]
    h = observe!(OnlineKDE(2; bandwidth = 0.05, budget = 25), randn(Xoshiro(6), 2, 200))
    @test size(h.centres, 2) == 25 && sum(h.weights) ≈ 200

    # a drifting relation: a circle whose radius grows from 1.0 to 1.5 over the stream
    K, rng = 1200, Xoshiro(4)
    stream = [(φ = 2π * rand(rng); (1 + 0.5k / K) .* [cos(φ), sin(φ)] .+ 0.02 .* randn(rng, 2)) for k in 1:K]
    nodes = field_nodes(Xoshiro(1), 2; samples = 4)
    answer(kde) = begin
        pred = kde_predictor(SCHED, kde); p, s = LuxCore.setup(Xoshiro(0), pred)
        roots, _ = implicit_roots(ImplicitDiffusion(pred, nodes), [0.0, 0.0], [Inf, 0.0], p, s; nstarts = 12, spread = 1.5)
        maximum(r.z[2] for r in roots)
    end
    remember = OnlineKDE(2; bandwidth = 0.05, budget = 300)
    forget = OnlineKDE(2; bandwidth = 0.05, forget = 0.99, budget = 300)
    foreach(z -> (observe!(remember, z); observe!(forget, z)), stream)
    @test answer(forget) ≈ 1.5 atol = 0.08       # tracks the current radius
    @test 1.1 < answer(remember) < 1.35          # averages the whole history
end

@testset "all answers of a query, and their Laplace uncertainty" begin
    # Gaussian data: the curvature is RED-Diff's κ, so the Laplace covariance has a closed form
    v0 = 0.5
    G = NoisePredictor(GaussianMixtureEps(SCHED, zeros(2, 1); s = sqrt(v0)), SCHED)
    pg, sg = LuxCore.setup(Xoshiro(0), G)
    nodes = field_nodes(Xoshiro(2), 2; samples = 3)
    mg = ImplicitDiffusion(G, nodes; λ = 0.7)
    κ = sum(nodes.w[k] * 0.7 * sigma(SCHED, t)^2 / (alpha(SCHED, t)^2 * v0 + sigma(SCHED, t)^2) for (k, t) in enumerate(nodes.t))
    ρ, z₀ = [Inf, 2.0], [0.3, 1.0]
    sol, _ = implicit_infer(mg, z₀, ρ, pg, sg; tol = 1e-12)
    L = implicit_laplace(mg, sol, ρ, pg, sg)
    @test L.mean == sol.z
    @test L.cov[2, 2] ≈ 1 / (κ + 4) rtol = 1e-10
    @test L.cov[1, 1] == 0 && L.free == [false, true]
    @test sol.z[2] ≈ 4 * z₀[2] / (κ + 4) rtol = 1e-10
    @test length(first(implicit_roots(mg, z₀, ρ, pg, sg))) == 1          # a Gaussian has one answer

    # the circle: both branches of "y given x", and none on the unstable centre
    θ = range(0, 2π; length = 49)[1:48]
    C = NoisePredictor(GaussianMixtureEps(SCHED, vcat(cos.(θ)', sin.(θ)'); s = 0.05), SCHED)
    pc, sc = LuxCore.setup(Xoshiro(0), C)
    cn = field_nodes(Xoshiro(1), 2; samples = 8)
    mc = ImplicitDiffusion(C, cn; λ = density_lambda(SCHED, cn))
    roots, _ = implicit_roots(mc, [0.6, 0.0], [Inf, 0.0], pc, sc)
    @test length(roots) == 2
    @test sort([r.z[2] for r in roots]) ≈ [-0.78, 0.77] atol = 0.02
    @test all(r -> r.converged && r.stable, roots)
    sds = [sqrt(implicit_laplace(mc, r, [Inf, 0.0], pc, sc).cov[2, 2]) for r in roots]
    @test all(0.1 .< sds .< 0.25)                                         # about the smoothed ring's width
    # λ rescales the curvature, not the answers, for hard / free queries
    r1, _ = implicit_roots(ImplicitDiffusion(C, cn), [0.6, 0.0], [Inf, 0.0], pc, sc)
    @test sort([r.z[2] for r in r1]) ≈ sort([r.z[2] for r in roots]) atol = 1e-6
    # an unstable answer has no Laplace approximation
    bad = ImplicitSolution([0.0, 0.0], 0.0, 1, true, false)
    @test_throws ArgumentError implicit_laplace(mc, bad, [Inf, 0.0], pc, sc)
end

@testset "energy-parametrised predictor: exact against the mixture, conservative, trainable" begin
    μ = [1.0 -0.5 0.2; 0.3 0.8 -1.0]
    exact = NoisePredictor(GaussianMixtureEps(SCHED, μ; s = 0.3), SCHED)
    ps, st = (μ = copy(μ),), NamedTuple()
    x, t, w = [0.4, -0.2], 0.35, [0.7, -1.3]
    @test_throws ArgumentError epsilon(NoisePredictor(EnergyNetwork(MixEnergy(μ, 0.3)), SCHED), x, t, ps, st)
    for ad in (AutoZygote(), DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoZygote()))
        en = NoisePredictor(EnergyNetwork(MixEnergy(μ, 0.3)), SCHED; ad)
        @test first(epsilon(en, x, t, ps, st)) ≈ first(epsilon(exact, x, t, ps, st)) atol = 1e-12
        @test epsilon_jacobian(en, x, t, ps, st) ≈ epsilon_jacobian(exact, x, t, ps, st) atol = 1e-12
        @test epsilon_vjp_params(en, x, t, ps, st, w).μ ≈ epsilon_vjp_params(exact, x, t, ps, st, w).μ atol = 1e-12
    end

    # a Lux energy network: a symmetric field, with a scalar energy for the relation
    ps0, st = Lux.setup(Xoshiro(0), ENERGY_MLP)
    ps = _f64(ps0)
    pred = NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = _batch_input, ad = AutoZygote())
    J = epsilon_jacobian(pred, x, t, ps, st)
    @test J ≈ J' atol = 1e-12
    m = ImplicitDiffusion(pred, field_nodes(Xoshiro(1), 2; samples = 2))
    z, h = [0.6, 0.4], 1e-6
    U(z) = first(implicit_energy(m, z, ps, st))
    @test [(U(z .+ h .* (1:2 .== i)) - U(z .- h .* (1:2 .== i))) / 2h for i in 1:2] ≈ first(prior_field(m, z, ps, st)) atol = 1e-8
    Jg = prior_jacobian(m, z, ps, st)
    @test Jg ≈ Jg' atol = 1e-12

    # inference and the adjoint, unchanged
    ρ = [Inf, 1.0]
    sol, _ = implicit_infer(m, [0.6, 0.5], ρ, ps, st; tol = 1e-12)
    @test sol.converged
    b = implicit_pullback(m, sol, [0.6, 0.5], ρ, [0.0, 1.0], ps, st)
    L(z₀, p) = first(implicit_infer(m, z₀, ρ, p, st; tol = 1e-12)).z[2]
    W(δ) = merge(ps, (layer_2 = (weight = (A = copy(ps.layer_2.weight); A[5] += δ; A), bias = ps.layer_2.bias),))
    @test b.ps.layer_2.weight[5] ≈ (L([0.6, 0.5], W(h)) - L([0.6, 0.5], W(-h))) / 2h rtol = 1e-5

    # the training gradient on a batch with one t per column
    x₀, tb, εb = randn(Xoshiro(2), 2, 5), reshape([0.1, 0.2, 0.3, 0.4, 0.5], 1, 5), randn(Xoshiro(3), 2, 5)
    @test first(epsilon(pred, x₀, tb, ps, st))[:, 2] ≈ first(epsilon(pred, x₀[:, 2], 0.2, ps, st)) atol = 1e-14
    _, g, _ = denoising_gradient(pred, x₀, tb, εb, ps, st)
    Lb(p) = first(denoising_gradient(pred, x₀, tb, εb, p, st))
    V(δ) = merge(ps, (layer_1 = (weight = (A = copy(ps.layer_1.weight); A[3] += δ; A), bias = ps.layer_1.bias),))
    @test g.layer_1.weight[3] ≈ (Lb(V(h)) - Lb(V(-h))) / 2h rtol = 1e-6
end

@testset "ProxDM: the exact prox, the sampler, proximal inference, the loss" begin
    m, s = [0.5, -1.0], 0.3
    G = GaussianMixtureEps(SCHED, reshape(m, 2, 1); s = s)
    P = MixtureProx(G); ps1 = (μ = reshape(m, 2, 1),)
    # one Gaussian: prox(v) = (v_t v + λ α m)/(v_t + λ), valid at t = 0 too
    for (t, λ) in ((0.0, 0.1), (0.3, 0.5), (0.8, 1.5))
        v = [1.0, 2.0]; a = alpha(SCHED, t); vt = a^2 * s^2 + sigma(SCHED, t)^2
        @test first(proximal(P, v, t, λ, ps1, NamedTuple())) ≈ (vt .* v .+ λ * a .* m) ./ (vt + λ) atol = 1e-10
    end
    # the sampler recovers the moments; its spread converges as the steps increase (first order)
    for hybrid in (false, true)
        rng = Xoshiro(3)
        X = reduce(hcat, [first(proxdm_sample(P, SCHED, randn(rng, 2), ps1, NamedTuple(); steps = 100, hybrid, rng))
                          for _ in 1:1500])
        @test vec(mean(X; dims = 2)) ≈ m atol = 0.03
        @test mean(std(X; dims = 2)) ≈ s atol = 0.03
    end
    @test_throws ArgumentError proxdm_sample(P, SCHED, randn(2), ps1, NamedTuple(); steps = 5)   # γ ≥ 2
    # proximal inference with a hard clamp finds both branches of the circle, exactly symmetric
    Pc = MixtureProx(RING.model)
    up, _ = prox_infer(Pc, [0.6, 0.5], HARDX, 0.005, RPS, RST; λ = 0.01)
    dn, _ = prox_infer(Pc, [0.6, -0.5], HARDX, 0.005, RPS, RST; λ = 0.01)
    @test up.converged && dn.converged
    @test up.z[1] == 0.6
    @test up.z[2] ≈ 0.8 atol = 0.03
    @test dn.z[2] ≈ -up.z[2] atol = 1e-8
    # the proximal-matching loss is in [0, 1) and zero exactly when the residual is predicted
    pn = ProxNetwork(MLP, SCHED; input = (v, t, λ) -> vcat(v, _emb(t) .* λ))
    pps, pst = Lux.setup(Xoshiro(0), MLP)
    ℓ, _ = proximal_matching_loss(pn, [0.1, 0.2], 0.1, 0.2, [0.3, -0.4], 0.5, pps, pst)
    @test 0 ≤ ℓ < 1
end

end
