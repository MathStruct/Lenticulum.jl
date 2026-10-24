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
using JLArrays: JLArrays, JLArray
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

# the reference energy written out in the network, differentiated by AD: the oracle for the
# closed-form terms of `EnergyNetwork(net; reference)`
struct RefExplicit{L,F,O} <: LuxCore.AbstractLuxWrapperLayer{:net}
    net::L
    input::F
    μ::Vector{Float64}
    v::Vector{Float64}
    owner::O
end
function (l::RefExplicit)(inp, ps, st)
    x, t = inp
    E, st = l.net(l.input(x, t), ps, st)
    a, s = Zygote.@ignore (alpha.(Ref(SCHED), t), sigma.(Ref(SCHED), t))
    A, S = l.owner === nothing ? (a, s) : (a[l.owner, :], s[l.owner, :])
    return (E .+ sum((x .- A .* l.μ) .^ 2 ./ (2 .* (A .^ 2 .* l.v .+ S .^ 2)); dims = 1), st)
end

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


@testset "answers as beliefs: Laplace Gaussians, mixtures, Gaussian messages" begin
    θs = range(0, 2π; length = 129)[1:128]
    C = reduce(hcat, [[cos(θ), sin(θ)] for θ in θs])
    nodes = field_nodes(Xoshiro(1), 2; samples = 8)
    λ = density_lambda(SCHED, nodes)
    function circle_model(w)
        p = NoisePredictor(GaussianMixtureEps(SCHED, C; s = 0.05, weights = w), SCHED)
        ps, st = LuxCore.setup(Xoshiro(0), p)
        return ImplicitDiffusion(p, nodes; λ), p, ps, st
    end
    m, p, ps, st = circle_model(nothing)
    # a closed-form mixture has an energy, and its gradient is the field
    z = [0.3, 0.7]; h = 1e-6
    U(z) = first(implicit_energy(m, z, ps, st))
    @test [(U(z .+ h .* e) - U(z .- h .* e)) / 2h for e in ([1.0, 0.0], [0.0, 1.0])] ≈ first(prior_field(m, z, ps, st)) rtol = 1e-6
    # laplace_belief is implicit_laplace's block as a GaussianBelief
    sol, _ = implicit_infer(m, [0.3, 0.5], [Inf, 0.0], ps, st)
    b = laplace_belief(m, sol, [Inf, 0.0], ps, st)
    L = implicit_laplace(m, sol, [Inf, 0.0], ps, st)
    @test b isa GaussianBelief && dimension(b) == 1
    @test belief_mean(b) ≈ sol.z[2:2] && belief_cov(b) ≈ L.cov[2:2, 2:2]
    @test_throws ArgumentError laplace_belief(m, sol, [Inf, 0.0], ps, st; coords = [1])
    # implicit_mixture: both branches, with weights that track the data's mass on each
    mix0, _ = implicit_mixture(m, [0.3, 0.0], [Inf, 0.0], ps, st)
    m3, _, ps3, st3 = circle_model([sin(θ) > 0 ? 3.0 : 1.0 for θ in θs])   # upper half 3× heavier
    mix3, _ = implicit_mixture(m3, [0.3, 0.0], [Inf, 0.0], ps3, st3)
    for mix in (mix0, mix3)
        @test mix isa MixtureBelief && length(mix.components) == 2
        @test sort([belief_mean(c)[1] for c in mix.components]) ≈ [-0.954, 0.954] atol = 0.04
    end
    up(mix) = mixture_weights(mix)[argmax([belief_mean(c)[1] for c in mix.components])]
    odds(q) = q / (1 - q)
    @test abs(up(mix0) - 0.5) < 0.1                       # symmetric data: about even
    @test odds(up(mix3)) / odds(up(mix0)) ≈ 3 rtol = 0.05  # the reweighting, recovered
    # a DiffusionFactor that sends Gaussian messages
    @test_throws ArgumentError ImplicitProx(nodes; message = :laplace)
    f = DiffusionFactor((x = 1, y = 1), p; prox = ImplicitProx(nodes; λ, message = :gaussian))
    q = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.5))   # a soft anchor on y
    lens, _ = LenticulumCore.assemble(f, q, ps, st)
    post, _ = LenticulumCore.invert(lens, DiracBelief([0.5]), (x = DiracBelief([0.6]),), ps, st)
    msg, _ = Mycelium.factor_message(f, :y, q, (x = DiracBelief([0.6]),), DiracBelief([0.5]), ps, st)
    @test post isa GaussianBelief && msg isa GaussianBelief
    @test belief_mean(post)[1] ≈ 0.78 atol = 0.05
    # the message times the anchor N(0.5, 1/0.5²) is the posterior: the anchor is not counted twice
    @test combine(msg, Gaussian([0.5], fill(1 / 0.25, 1, 1))).Λ ≈ post.Λ
    @test belief_mean(combine(msg, Gaussian([0.5], fill(1 / 0.25, 1, 1)))) ≈ belief_mean(post)
    # with no anchor (a pure conditional) message and posterior coincide
    q0 = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.0))
    msg0, _ = Mycelium.factor_message(f, :y, q0, (x = DiracBelief([0.6]),), DiracBelief([0.5]), ps, st)
    post0, _ = LenticulumCore.invert(first(LenticulumCore.assemble(f, q0, ps, st)), DiracBelief([0.5]),
                                     (x = DiracBelief([0.6]),), ps, st)
    @test msg0.Λ ≈ post0.Λ && msg0.η ≈ post0.η
    # a Gaussian prior's mean is now used as the anchor and warm start (it was ignored before)
    postg, _ = LenticulumCore.invert(lens, Gaussian([-0.5], fill(1.0, 1, 1)), (x = DiracBelief([0.6]),), ps, st)
    @test belief_mean(postg)[1] ≈ -0.78 atol = 0.05
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

@testset "free-coordinate Jacobian blocks, input VJPs, batched nodes" begin
    z, w = [0.6, 0.5], [0.3, -0.8]
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    # the closed-form mixture: the block is the slice, the VJP is Jᵀw
    m = ImplicitDiffusion(RING, nodes)
    J = prior_jacobian(m, z, RPS, RST)
    @test prior_jacobian(m, z, RPS, RST; coords = [2]) ≈ J[[2], [2]] atol = 1e-12
    @test prior_jacobian(m, z, RPS, RST; coords = [false, true]) ≈ J[[2], [2]] atol = 1e-12
    @test size(prior_jacobian(m, z, RPS, RST; coords = Int[])) == (0, 0)
    @test prior_vjp(m, z, w, RPS, RST) ≈ J' * w atol = 1e-12
    @test_throws DimensionMismatch prior_jacobian(m, z, RPS, RST; coords = [true])

    ps0, st = Lux.setup(Xoshiro(0), MLP)
    ps = _f64(ps0)
    # no backend: finite differences along the free coordinates only
    plain = ImplicitDiffusion(NoisePredictor(MLP, SCHED; input = _mlp_input), nodes)
    Jp = prior_jacobian(plain, z, ps, st)
    @test prior_jacobian(plain, z, ps, st; coords = [1]) ≈ Jp[[1], [1]] rtol = 1e-6
    @test prior_vjp(plain, z, w, ps, st) ≈ Jp' * w rtol = 1e-6

    # every AD backend, one node per call and one noise level per call
    for ad in (AutoZygote(), AutoForwardDiff(), AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
        pred = NoisePredictor(MLP, SCHED; input = _batch_input, ad = ad)
        single, batched = ImplicitDiffusion(pred, nodes), ImplicitDiffusion(pred, nodes; batched = true)
        J1 = prior_jacobian(single, z, ps, st)
        @test J1 ≈ Jp rtol = 1e-5
        @test prior_jacobian(single, z, ps, st; coords = [2]) ≈ J1[[2], [2]] rtol = 1e-10
        @test prior_vjp(single, z, w, ps, st) ≈ J1' * w rtol = 1e-10
        @test first(prior_field(batched, z, ps, st)) ≈ first(prior_field(single, z, ps, st)) rtol = 1e-12
        @test prior_jacobian(batched, z, ps, st) ≈ J1 rtol = 1e-10
        @test prior_jacobian(batched, z, ps, st; coords = [2]) ≈ J1[[2], [2]] rtol = 1e-10
        @test prior_vjp(batched, z, w, ps, st) ≈ J1' * w rtol = 1e-10
    end

    # inference and the adjoint do not depend on how the nodes are batched
    pred = NoisePredictor(MLP, SCHED; input = _batch_input, ad = AutoZygote())
    single, batched = ImplicitDiffusion(pred, nodes), ImplicitDiffusion(pred, nodes; batched = true)
    ρ = [Inf, 1.0]
    s1, _ = implicit_infer(single, z, ρ, ps, st; tol = 1e-12)
    s2, _ = implicit_infer(batched, z, ρ, ps, st; tol = 1e-12)
    @test s1.converged && s2.converged
    @test s2.z ≈ s1.z atol = 1e-10
    b1 = implicit_pullback(single, s1, z, ρ, [0.0, 1.0], ps, st)
    b2 = implicit_pullback(batched, s2, z, ρ, [0.0, 1.0], ps, st)
    @test b2.z₀ ≈ b1.z₀ rtol = 1e-8
    @test b2.ps.layer_2.weight ≈ b1.ps.layer_2.weight rtol = 1e-8 atol = 1e-12

    # energy networks: the block is a Hessian block, the VJP a Hessian-vector product
    eps0, est = Lux.setup(Xoshiro(0), ENERGY_MLP)
    eps_ = _f64(eps0)
    epred = NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = _batch_input, ad = AutoZygote())
    es, eb = ImplicitDiffusion(epred, nodes), ImplicitDiffusion(epred, nodes; batched = true)
    Je = prior_jacobian(es, z, eps_, est)
    @test prior_jacobian(es, z, eps_, est; coords = [1]) ≈ Je[[1], [1]] rtol = 1e-10
    @test prior_jacobian(eb, z, eps_, est) ≈ Je rtol = 1e-10
    @test prior_vjp(eb, z, w, eps_, est) ≈ Je' * w rtol = 1e-10

    # products: the block and the VJP are the weighted sums
    pr = ProductRelation(m, m; weights = [1.0, 2.0])
    @test prior_jacobian(pr, z, (RPS, RPS), (RST, RST); coords = [2]) ≈ 3 .* J[[2], [2]] atol = 1e-12
    @test prior_vjp(pr, z, w, (RPS, RPS), (RST, RST)) ≈ 3 .* (J' * w) atol = 1e-12
end

@testset "matrix-free solves: GMRES, Lanczos, Krylov inference and adjoint" begin
    VD = VariationalDiffusion
    # the linear algebra, against dense factorisations
    rng = Xoshiro(3)
    A = randn(rng, 30, 30) + 8I
    b = randn(rng, 30)
    x, ok, _ = VD._gmres(v -> A * v, b; rtol = 1e-12, restart = 7)          # restarts several times
    @test ok
    @test x ≈ A \ b rtol = 1e-9
    Ssym = Symmetric(A + A') / 2
    x, ok = VD._cg(v -> Ssym * v, b; rtol = 1e-12)
    @test ok
    @test x ≈ Ssym \ b rtol = 1e-9
    ls = KrylovSolve(lanczos = 30, probes = 40)
    @test VD._lambda_min(v -> Ssym * v, 30, ls) ≈ eigmin(Ssym) rtol = 1e-8   # exact: a full Krylov space
    @test VD._logdet(v -> Ssym * v, 30, ls) ≈ logdet(Ssym) rtol = 1e-8       # exact: unit-vector probes
    est = VD._logdet(v -> Ssym * v, 30, KrylovSolve(lanczos = 15, probes = 8))
    @test est ≈ logdet(Ssym) rtol = 0.05                                     # a stochastic estimate
    @test VD._lambda_min(v -> -(Ssym * v), 30, ls) < 0                       # an indefinite operator is caught

    # prior_jvp is J v: closed form, finite differences, every AD backend, energy, product
    z, v = [0.6, 0.5], [0.3, -0.8]
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    m = ImplicitDiffusion(RING, nodes)
    J = prior_jacobian(m, z, RPS, RST)
    @test prior_jvp(m, z, v, RPS, RST) ≈ J * v atol = 1e-12
    @test prior_jvp(ProductRelation(m, m; weights = [1.0, 2.0]), z, v, (RPS, RPS), (RST, RST)) ≈ 3 .* (J * v) atol = 1e-12
    ps0, st = Lux.setup(Xoshiro(0), MLP)
    ps = _f64(ps0)
    plain = ImplicitDiffusion(NoisePredictor(MLP, SCHED; input = _mlp_input), nodes)
    @test prior_jvp(plain, z, v, ps, st) ≈ prior_jacobian(plain, z, ps, st) * v rtol = 1e-6
    for ad in (AutoZygote(), AutoForwardDiff(), AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward)))
        pred = NoisePredictor(MLP, SCHED; input = _batch_input, ad = ad)
        for batched in (false, true)
            mm = ImplicitDiffusion(pred, nodes; batched)
            @test prior_jvp(mm, z, v, ps, st) ≈ prior_jacobian(mm, z, ps, st) * v rtol = 1e-7
        end
    end
    eps0, est_ = Lux.setup(Xoshiro(0), ENERGY_MLP)
    eps_ = _f64(eps0)
    em = ImplicitDiffusion(NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = _batch_input, ad = AutoZygote()), nodes)
    @test prior_jvp(em, z, v, eps_, est_) ≈ prior_jacobian(em, z, eps_, est_) * v rtol = 1e-10

    # Krylov inference, adjoint and Laplace agree with the dense path on the circle
    K = KrylovSolve()
    for ρ in ([Inf, 0.0], [1.0, 0.0], [0.5, 0.5])
        s1, _ = implicit_infer(m, z, ρ, RPS, RST; method = :newton)       # GMRES Newton steps = dense ones
        s2, _ = implicit_infer(m, z, ρ, RPS, RST; linsolve = K, method = :newton)
        @test s2.converged && s2.stable
        @test s2.z ≈ s1.z atol = 1e-10
        t1, _ = implicit_infer(m, z, ρ, RPS, RST)                         # trust region: exact steps against Steihaug
        t2, _ = implicit_infer(m, z, ρ, RPS, RST; linsolve = K)
        @test t2.converged && t2.stable && t2.z ≈ t1.z && t1.z ≈ s1.z
        b1 = implicit_pullback(m, s1, z, ρ, [0.2, 1.0], RPS, RST)
        b2 = implicit_pullback(m, s2, z, ρ, [0.2, 1.0], RPS, RST; linsolve = K)
        @test b2.z₀ ≈ b1.z₀ atol = 1e-8
        @test b2.ρ ≈ b1.ρ atol = 1e-8
        @test b2.ps.μ ≈ b1.ps.μ atol = 1e-8
        g1 = laplace_belief(m, s1, ρ, RPS, RST)
        g2 = laplace_belief(m, s2, ρ, RPS, RST; linsolve = K)
        @test belief_mean(g2) ≈ belief_mean(g1)
        @test belief_cov(g2) ≈ belief_cov(g1) rtol = 1e-7
    end
    # the centre of the circle is a maximum of the energy: unstable in both
    c1, _ = implicit_infer(m, [0.0, 0.0], [Inf, 0.0], RPS, RST; maxiters = 0)
    c2, _ = implicit_infer(m, [0.0, 0.0], [Inf, 0.0], RPS, RST; maxiters = 0, linsolve = K)
    @test !c1.stable && !c2.stable
    # mixtures: the same roots, covariances and (closed-form energy) weights
    mx1, _ = implicit_mixture(m, [0.3, 0.0], [Inf, 0.0], RPS, RST)
    mx2, _ = implicit_mixture(m, [0.3, 0.0], [Inf, 0.0], RPS, RST; linsolve = K)
    @test length(mx2.components) == length(mx1.components) == 2
    @test belief_mean.(mx2.components) ≈ belief_mean.(mx1.components) atol = 1e-8
    @test mx2.logw ≈ mx1.logw atol = 1e-8

    # a Lux network: Zygote's JVPs are differences, the adjoint's VJPs exact
    pred = NoisePredictor(MLP, SCHED; input = _batch_input, ad = AutoZygote())
    mm = ImplicitDiffusion(pred, nodes; batched = true)
    ρ = [Inf, 1.0]
    s1, _ = implicit_infer(mm, z, ρ, ps, st; tol = 1e-12)
    s2, _ = implicit_infer(mm, z, ρ, ps, st; tol = 1e-12, linsolve = K)
    @test s2.converged
    @test s2.z ≈ s1.z atol = 1e-10
    b1 = implicit_pullback(mm, s1, z, ρ, [0.0, 1.0], ps, st)
    b2 = implicit_pullback(mm, s2, z, ρ, [0.0, 1.0], ps, st; linsolve = K)
    @test b2.z₀ ≈ b1.z₀ rtol = 1e-8
    @test b2.ps.layer_2.weight ≈ b1.ps.layer_2.weight rtol = 1e-8 atol = 1e-12

    # a factor configured with batched nodes and Krylov solves gives the same inversion
    fd = DiffusionFactor((x = 1, y = 1), RING; prox = ImplicitProx(field_nodes(Xoshiro(1), 2; samples = 8)))
    fk = DiffusionFactor((x = 1, y = 1), RING; prox = ImplicitProx(field_nodes(Xoshiro(1), 2; samples = 8);
                                                                    linsolve = K, message = :gaussian))
    p = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.3))
    sd, _ = implicit_solution(fd, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), RPS, RST)
    sk, _ = implicit_solution(fk, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), RPS, RST)
    @test sk.z ≈ sd.z atol = 1e-10
    gd = implicit_factor_pullback(fd, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), (y = [1.0],), RPS, RST)
    gk = implicit_factor_pullback(fk, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), (y = [1.0],), RPS, RST)
    @test gk.inputs.x ≈ gd.inputs.x atol = 1e-8
    @test ImplicitProx(nodes; batched = true).batched
end

@testset "Zygote differentiates through implicit_infer (the rrule)" begin
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    m = ImplicitDiffusion(RING, nodes)
    z₀, ρ = [0.6, 0.5], [Inf, 0.3]
    loss(z₀, ρ, ps) = (first(implicit_infer(m, z₀, ρ, ps, RST)).z[2] - 0.7)^2 / 2
    gz, gρ, gp = Zygote.gradient(loss, z₀, ρ, RPS)
    sol, _ = implicit_infer(m, z₀, ρ, RPS, RST)
    b = implicit_pullback(m, sol, z₀, ρ, [0.0, sol.z[2] - 0.7], RPS, RST)
    @test gz ≈ b.z₀ && gρ ≈ b.ρ && gp.μ ≈ b.ps.μ
    h = 1e-6
    @test gz[1] ≈ (loss([0.6 + h, 0.5], ρ, RPS) - loss([0.6 - h, 0.5], ρ, RPS)) / 2h rtol = 1e-5
    @test gρ[2] ≈ (loss(z₀, [Inf, 0.3 + h], RPS) - loss(z₀, [Inf, 0.3 - h], RPS)) / 2h rtol = 1e-5
    # a loss on the solver's report only has no gradient; Krylov solves give the same one
    @test Zygote.gradient(ps -> Float64(first(implicit_infer(m, z₀, ρ, ps, RST)).iters), RPS) == (nothing,)
    gk = Zygote.gradient(ps -> (first(implicit_infer(m, z₀, ρ, ps, RST; linsolve = KrylovSolve())).z[2] - 0.7)^2 / 2, RPS)
    @test gk[1].μ ≈ b.ps.μ atol = 1e-8
    # a Lux network inside a larger Zygote computation: training through inference
    ps0, st = Lux.setup(Xoshiro(1), MLP)
    ps = _f64(ps0)
    mm = ImplicitDiffusion(NoisePredictor(MLP, SCHED; input = _batch_input, ad = AutoZygote()), nodes; batched = true)
    L(p) = sum(abs2, first(implicit_infer(mm, [0.6, 0.5], [Inf, 1.0], p, st; tol = 1e-12)).z .- [0.6, 0.7]) / 2
    g = only(Zygote.gradient(L, ps))
    W(δ) = merge(ps, (layer_2 = (weight = (A = copy(ps.layer_2.weight); A[5] += δ; A), bias = ps.layer_2.bias),))
    @test g.layer_2.weight[5] ≈ (L(W(h)) - L(W(-h))) / 2h rtol = 1e-4 atol = 1e-10
    @test size(g.layer_1.weight) == size(ps.layer_1.weight)
end

@testset "clamped sampling: branch frequencies and spread match the data" begin
    θs = range(0, 2π; length = 129)[1:128]
    C = reduce(hcat, [[cos(θ), sin(θ)] for θ in θs])
    w = [sin(θ) > 0 ? 3.0 : 1.0 for θ in θs]                       # upper half 3× heavier
    p = NoisePredictor(GaussianMixtureEps(SCHED, C; s = 0.05, weights = w), SCHED)
    ps, st = LuxCore.setup(Xoshiro(0), p)
    m = ImplicitDiffusion(p, field_nodes(Xoshiro(1), 2; samples = 8))
    # unconditional: the ring's radius and width
    Z, _ = implicit_sample(m, [0.0, 0.0], [0.0, 0.0], ps, st; n = 2000, rng = Xoshiro(1))
    r = vec(sqrt.(sum(abs2, Z; dims = 1)))
    @test mean(r) ≈ 1 atol = 0.01
    @test std(r) ≈ 0.05 rtol = 0.1
    # x clamped at 0.3: the exact conditional puts 0.75 on the upper branch, |y| = 0.952 ± 0.052
    Z, _ = implicit_sample(m, [0.3, 0.0], [Inf, 0.0], ps, st; n = 2000, resample = 3, rng = Xoshiro(2))
    @test all(==(0.3), Z[1, :])                                     # the clamp is exact
    @test mean(Z[2, :] .> 0) ≈ 0.75 atol = 0.03
    @test mean(abs.(Z[2, :])) ≈ 0.952 atol = 0.01
    @test std(abs.(Z[2, :])) ≈ 0.052 rtol = 0.15
    @test_throws ArgumentError implicit_sample(m, [0.3, 0.0], [1.0, 0.0], ps, st)   # soft evidence
    # batched and unbatched chains are the same draws
    ps0, stm = Lux.setup(Xoshiro(0), MLP)
    psm = _f64(ps0)
    pred = NoisePredictor(MLP, SCHED; input = _batch_input)
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    Z1, _ = implicit_sample(ImplicitDiffusion(pred, nodes), [0.3, 0.0], [Inf, 0.0], psm, stm; n = 5, steps = 10, rng = Xoshiro(3))
    Z2, _ = implicit_sample(ImplicitDiffusion(pred, nodes; batched = true), [0.3, 0.0], [Inf, 0.0], psm, stm;
                            n = 5, steps = 10, rng = Xoshiro(3))
    @test Z1 ≈ Z2 rtol = 1e-12
end

@testset "coverage: answers off the data are flagged, typical ones are not" begin
    draw(rng, N) = RING_μ[:, rand(rng, 1:RING_J, N)] .+ 0.05 .* randn(rng, 2, N)   # the ring's data
    m = ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2; samples = 8))
    data, cal = draw(Xoshiro(2), 2000), draw(Xoshiro(3), 500)
    ref = coverage_reference(m, data, RPS, RST; calibration = cal)
    # valid: about 5% of fresh in-distribution points have p < 0.05, for every score
    fresh = coverage(ref, draw(Xoshiro(4), 400))
    for f in (:p_distance, :p_reconstruction, :p_energy, :p)
        @test mean(c -> getfield(c, f) < 0.05, fresh) < 0.08
    end
    # points off the ring are beyond every reference point
    for z in ([0.0, 0.0], [1.3, 0.0], [0.0, 0.8])
        c = coverage(ref, z)
        @test c.p_distance == c.p_reconstruction == 1 / 501
        @test c.p < 0.01
    end
    # an answer on the relation passes; a clamp off it (x = 1.3) still converges, and is flagged
    ok, _ = implicit_infer(m, [0.6, 0.5], HARDX, RPS, RST)
    off, _ = implicit_infer(m, [1.3, 0.5], HARDX, RPS, RST)
    @test ok.converged && off.converged
    @test coverage(ref, ok).p > 0.1
    @test coverage(ref, off).p < 0.01
    @test coverage(ref, ok) == coverage(ref, ok.z)                  # fixed noise draws: deterministic
    # default calibration: columns of the data, each left out of its own nearest-neighbour search
    ref2 = coverage_reference(m, data, RPS, RST; ncal = 300)
    @test length(ref2.distance) == 300 && all(>(0), ref2.distance)
    @test coverage(ref2, ok).p > 0.1 && coverage(ref2, off).p < 0.01
    @test_throws ArgumentError coverage_reference(m, data, RPS, RST; scale = [1.0, -1.0])
    @test coverage_reference(m, data, RPS, RST; calibration = cal[:, 1:5], scale = [1.0, 2.0]).distance ≈
          [minimum(norm(([1.0, 0.5]) .* (cal[:, j] .- data[:, k])) for k in axes(data, 2)) for j in 1:5]
    # an energy network's energy score is its own E: here the mixture's -log p_t
    en = NoisePredictor(EnergyNetwork(MixEnergy(RING_μ, 0.05)), SCHED; ad = AutoZygote())
    refe = coverage_reference(ImplicitDiffusion(en, m.nodes), data, (μ = copy(RING_μ),), NamedTuple(); calibration = cal[:, 1:50])
    refm = coverage_reference(m, data, RPS, RST; calibration = cal[:, 1:50])
    @test refe.energy ≈ refm.energy rtol = 1e-10
    @test refe.reconstruction ≈ refm.reconstruction rtol = 1e-8
    # a plain ε-network has no energy: two scores, and batched calls give the same ones
    ps0, stm = Lux.setup(Xoshiro(0), MLP)
    psm = _f64(ps0)
    pred = NoisePredictor(MLP, SCHED; input = _batch_input)
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    refp = coverage_reference(ImplicitDiffusion(pred, nodes), data, psm, stm; calibration = cal[:, 1:50], draws = 4)
    refb = coverage_reference(ImplicitDiffusion(pred, nodes; batched = true), data, psm, stm; calibration = cal[:, 1:50], draws = 4)
    @test refp.energy === nothing
    @test refb.reconstruction ≈ refp.reconstruction rtol = 1e-12
    c = coverage(refp, [0.6, 0.8])
    @test c.energy === nothing && c.p_energy === nothing
    @test c.p == min(1, 2 * min(c.p_distance, c.p_reconstruction))
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

@testset "products of relations: intersections" begin
    curve(f, n) = reduce(hcat, [collect(f(θ)) for θ in range(0, 2π; length = n + 1)[1:n]])
    nodes = field_nodes(Xoshiro(1), 2; samples = 8)
    function model(c)
        p = NoisePredictor(GaussianMixtureEps(SCHED, c; s = 0.05), SCHED)
        return ImplicitDiffusion(p, nodes), LuxCore.setup(Xoshiro(0), p)...
    end
    m1, p1, s1 = model(curve(θ -> (cos(θ), sin(θ)), 64))                       # unit circle
    m2, p2, s2 = model(reduce(hcat, [[u, u] for u in range(-1.6, 1.6; length = 65)]))   # line y = x
    m3, p3, s3 = model(curve(θ -> (1.5cos(θ), 0.6sin(θ)), 96))                 # ellipse
    # the field of a product is the weighted sum of the fields
    prod = ProductRelation(m1, m2; weights = [1.0, 2.0])
    z = [0.3, -0.2]
    @test first(prior_field(prod, z, (p1, p2), (s1, s2))) ≈
          first(prior_field(m1, z, p1, s1)) .+ 2 .* first(prior_field(m2, z, p2, s2))
    @test_throws ArgumentError prior_field(prod, z, p1, s1)                    # ps must be a tuple
    # circle ∩ line: two answers, circle ∩ ellipse: four, near the exact intersections
    for (other, ps, st, exact) in ((m2, p2, s2, [[0.7071, 0.7071], [-0.7071, -0.7071]]),
                                   (m3, p3, s3, [[sx * 0.873, sy * 0.488] for sx in (1, -1) for sy in (1, -1)]))
        roots, _ = implicit_roots(ProductRelation(m1, other), [0.0, 0.0], [0.0, 0.0], (p1, ps), (s1, st);
                                  nstarts = 40, spread = 1.5, maxiters = 400)
        @test length(roots) == length(exact)
        @test all(r -> minimum(norm(r.z .- e) for e in exact) < 0.05, roots)
    end
    # the adjoint through a product: w.r.t. the clamped input and a parameter of the second factor
    pe = ProductRelation(m1, m3); ρ = [Inf, 0.0]; h = 1e-6
    sol, _ = implicit_infer(pe, [0.95, 0.4], ρ, (p1, p3), (s1, s3); tol = 1e-12)
    b = implicit_pullback(pe, sol, [0.95, 0.4], ρ, [0.0, 1.0], (p1, p3), (s1, s3))
    y(x, μ) = first(implicit_infer(pe, [x, 0.4], ρ, (p1, (μ = μ,)), (s1, s3); tol = 1e-12)).z[2]
    @test b.z₀[1] ≈ (y(0.95 + h, p3.μ) - y(0.95 - h, p3.μ)) / 2h rtol = 1e-6
    μ₊ = copy(p3.μ); μ₊[2, 10] += h; μ₋ = copy(p3.μ); μ₋[2, 10] -= h
    @test b.ps[2].μ[2, 10] ≈ (y(0.95, μ₊) - y(0.95, μ₋)) / 2h rtol = 1e-6
    @test b.ps isa Tuple && length(b.ps) == 2
end

@testset "products over different variables: separators" begin
    # p(x, y, z) = p_A(x, y) p_B(y, z) / p_S(y): two blobs, weights 0.8 and 0.2
    w = [0.8, 0.2]
    n2 = field_nodes(Xoshiro(1), 2; samples = 8); n1 = field_nodes(Xoshiro(2), 1; samples = 8)
    function model(μ, nodes)
        p = NoisePredictor(GaussianMixtureEps(SCHED, μ; s = 0.05, weights = w), SCHED)
        return ImplicitDiffusion(p, nodes; λ = density_lambda(SCHED, nodes)), LuxCore.setup(Xoshiro(0), p)...
    end
    mA, pA, sA = model([-1.0 1.0; -1.0 1.0], n2)              # (x, y)
    mB, pB, sB = model([-1.0 1.0; -2.0 2.0], n2)              # (y, z)
    mS, pS, sS = model([-1.0 1.0], n1)                        # y
    sep = ProductRelation(mA => [1, 2], mB => [2, 3], mS => [2]; weights = [1, 1, -1])
    ps, st = (pA, pB, pS), (sA, sB, sS)
    # the field: each relation's field on its own coordinates
    z = [-0.8, -1.1, -1.7]
    g = zeros(3)
    g[1:2] .+= first(prior_field(mA, z[1:2], pA, sA))
    g[2:3] .+= first(prior_field(mB, z[2:3], pB, sB))
    g[2:2] .-= first(prior_field(mS, z[2:2], pS, sS))
    @test first(prior_field(sep, z, ps, st)) ≈ g
    # Jacobian blocks, JVP and VJP agree with central differences of the field
    h = 1e-6
    Jfd = reduce(hcat, [(first(prior_field(sep, z .+ h .* e, ps, st)) .- first(prior_field(sep, z .- h .* e, ps, st))) ./ 2h
                        for e in eachcol(Matrix(1.0I, 3, 3))])
    J = prior_jacobian(sep, z, ps, st)
    @test J ≈ Jfd rtol = 1e-6
    @test prior_jacobian(sep, z, ps, st; coords = [false, true, true]) ≈ J[2:3, 2:3]
    @test prior_jacobian(sep, z, ps, st; coords = [1, 3]) ≈ J[[1, 3], [1, 3]]
    v = [0.3, -1.0, 0.5]
    @test prior_jvp(sep, z, v, ps, st) ≈ J * v
    @test prior_vjp(sep, z, v, ps, st) ≈ J' * v
    # with the separator the answers carry the data's weights; without it the shared marginal
    # counts twice, 0.8² : 0.2²
    for (rel, p, s, want) in ((sep, ps, st, 0.8), (ProductRelation(mA => [1, 2], mB => [2, 3]), (pA, pB), (sA, sB), 0.64 / 0.68))
        mix, _ = implicit_mixture(rel, zeros(3), zeros(3), p, s; nstarts = 30, spread = 1.5)
        ws = exp.(mix.logw) ./ sum(exp.(mix.logw))
        i = argmax(ws)
        @test belief_mean(mix.components[i]) ≈ [-1, -1, -2] atol = 0.01
        @test ws[i] ≈ want atol = 0.005
    end
    # the adjoint through a chain of curves, y = x/2 and z = y², with x clamped: d z★/d x and a
    # parameter of the second relation against central differences
    v = collect(range(-1.6, 1.6; length = 65))
    curve(μ, nodes) = (p = NoisePredictor(GaussianMixtureEps(SCHED, μ; s = 0.05), SCHED);
                       (ImplicitDiffusion(p, nodes), LuxCore.setup(Xoshiro(0), p)...))
    cA, qA, tA = curve(vcat(v', v' ./ 2), n2)
    cB, qB, tB = curve(vcat(v', v' .^ 2), n2)
    cS, qS, tS = curve(v' ./ 2, n1)
    ch = ProductRelation(cA => [1, 2], cB => [2, 3], cS => [2]; weights = [1, 1, -1])
    ρ = [Inf, 0.0, 0.0]
    zstar(x, μB) = first(implicit_infer(ch, [x, 0.3, 0.1], ρ, (qA, (μ = μB,), qS), (tA, tB, tS); tol = 1e-12))
    sol = zstar(0.6, qB.μ)
    @test sol.converged && sol.stable
    @test sol.z[2:3] ≈ [0.3, 0.09] atol = 0.03         # the smoothing bias
    b = implicit_pullback(ch, sol, [0.6, 0.3, 0.1], ρ, [0.0, 0.0, 1.0], (qA, qB, qS), (tA, tB, tS))
    @test b.z₀[1] ≈ (zstar(0.6 + h, qB.μ).z[3] - zstar(0.6 - h, qB.μ).z[3]) / 2h rtol = 1e-5
    k = argmin(abs.(v .- 0.3))
    μ₊ = copy(qB.μ); μ₊[2, k] += h; μ₋ = copy(qB.μ); μ₋[2, k] -= h
    @test b.ps[2].μ[2, k] ≈ (zstar(0.6, μ₊).z[3] - zstar(0.6, μ₋).z[3]) / 2h rtol = 1e-5
    @test abs(b.ps[2].μ[2, k]) > 0.01
    @test b.ps isa Tuple && length(b.ps) == 3
    # the energy is the weighted sum of the relations' energies on their coordinates
    @test first(implicit_energy(sep, z, ps, st)) ≈ first(implicit_energy(mA, z[1:2], pA, sA)) +
          first(implicit_energy(mB, z[2:3], pB, sB)) - first(implicit_energy(mS, z[2:2], pS, sS))
    # three relations in a chain, the outer two sharing no variable: a bit passed along three
    # links (blobs at (±1, ±1), weights π(a) T(a, b)), x₀ = +1 held. The answers are the 8 paths of
    # the bit with their probabilities, and P(x₃ = +1) is the (+1, +1) entry of T³ (examples/chain.jl)
    T = [0.8 0.2; 0.3 0.7]
    B4 = [1.0 1.0 -1.0 -1.0; 1.0 -1.0 1.0 -1.0]
    st₊(v) = v > 0 ? 1 : 2
    πs = [[0.5, 0.5]' * T^k for k in 0:3]
    tel = map(1:3) do k
        p = NoisePredictor(GaussianMixtureEps(SCHED, B4; s = 0.05, weights = [πs[k][st₊(a)] * T[st₊(a), st₊(b)] for (a, b) in eachcol(B4)]), SCHED)
        (ImplicitDiffusion(p, n2; λ = density_lambda(SCHED, n2)), LuxCore.setup(Xoshiro(0), p)...)
    end
    tsep = map(2:3) do k
        p = NoisePredictor(GaussianMixtureEps(SCHED, [1.0 -1.0]; s = 0.05, weights = vec(πs[k])), SCHED)
        (ImplicitDiffusion(p, n1; λ = density_lambda(SCHED, n1)), LuxCore.setup(Xoshiro(0), p)...)
    end
    three = ProductRelation(tel[1][1] => [1, 2], tel[2][1] => [2, 3], tel[3][1] => [3, 4], tsep[1][1] => [2], tsep[2][1] => [3];
                            weights = [1, 1, 1, -1, -1])
    tp, ts = Tuple(r[2] for r in [tel; tsep]), Tuple(r[3] for r in [tel; tsep])
    mix, _ = implicit_mixture(three, [1.0, 0, 0, 0], [Inf, 0, 0, 0], tp, ts; nstarts = 320, spread = 1.5)
    @test length(mix.components) == 8
    ws = exp.(mix.logw) ./ sum(exp.(mix.logw))
    paths = [sign.(belief_mean(c)) for c in mix.components]
    @test all(i -> isapprox(ws[i], prod(T[st₊(a), st₊(b)] for (a, b) in zip([1.0; paths[i][1:2]], paths[i])); atol = 1e-3), eachindex(ws))
    @test sum(ws[i] for i in eachindex(ws) if paths[i][3] > 0) ≈ (T^3)[1, 1] atol = 1e-3
    # construction errors
    @test_throws ArgumentError ProductRelation(mA => [1, 2], mB => [2, 3], mS => [2]; weights = [1, 1, -2])   # separator outweighs
    @test_throws ArgumentError ProductRelation(mA => [1, 2], mB => [3, 4]; dim = 5)                # coordinate 5 unseen
    @test_throws DimensionMismatch ProductRelation(mA => [1, 2, 3])                                # dimension 2, three coordinates
    @test_throws ArgumentError ProductRelation(mA => [1, 1])                                       # repeated coordinate
    @test_throws DimensionMismatch prior_field(sep, zeros(2), ps, st)
end

@testset "sampling a product: corrector steps and reweighting" begin
    # circle (upper half 3× heavier) ∩ line y = x: the exact product puts 0.75 on the upper
    # intersection, both blobs of width 0.05/√2·√2 = 0.05 along and across the line
    θ = range(0, 2π; length = 129)[1:128]
    mk(μ; w = nothing) = (p = NoisePredictor(GaussianMixtureEps(SCHED, μ; s = 0.05, weights = w), SCHED);
                          (ImplicitDiffusion(p, noisefree_nodes(2, 0.01)), LuxCore.setup(Xoshiro(0), p)...))
    mC, pC, sC = mk(vcat(cos.(θ)', sin.(θ)'); w = [sin(t) > 0 ? 3.0 : 1.0 for t in θ])
    u = range(-1.6, 1.6; length = 65)
    mL, pL, sL = mk(vcat(u', u'))
    prod = ProductRelation(mC, mL); ps, st = (pC, pL), (sC, sL)
    stats(Z) = (up = mean(Z[1, :] .+ Z[2, :] .> 0),
                stray = mean((abs.(vec(sqrt.(sum(abs2, Z; dims = 1))) .- 1) .> 0.15) .| (abs.(Z[1, :] .- Z[2, :]) .> 0.2)),
                across = std((Z[1, Z[1, :] .+ Z[2, :] .> 0] .- Z[2, Z[1, :] .+ Z[2, :] .> 0]) ./ sqrt(2)))
    plain = stats(first(implicit_sample(prod, zeros(2), zeros(2), ps, st; n = 2000, rng = Xoshiro(1))))
    @test plain.stray > 0.05                                        # summed scores alone: T9
    @test plain.up < 0.7
    lang = stats(first(implicit_sample(prod, zeros(2), zeros(2), ps, st; n = 2000, rng = Xoshiro(1),
                                       corrector = :langevin, corrector_steps = 10, corrector_step = 0.5)))
    @test lang.stray < 0.01
    @test lang.across ≈ 0.05 atol = 0.004
    smc = stats(first(implicit_sample(prod, zeros(2), zeros(2), ps, st; n = 2000, rng = Xoshiro(1),
                                      smc = true, corrector = :mala, corrector_steps = 3, corrector_step = 0.5)))
    @test smc.up ≈ 0.75 atol = 0.04                                 # the mass split, which MCMC alone misses
    @test smc.stray < 0.015
    # a single model with clamps: the conditional's branch weights
    m1 = ImplicitDiffusion(mC.predictor, noisefree_nodes(2, 0.01))
    Z, _ = implicit_sample(m1, [0.3, 0.0], [Inf, 0.0], pC, sC; n = 2000, smc = true, rng = Xoshiro(2))
    @test all(==(0.3), Z[1, :])
    @test mean(Z[2, :] .> 0) ≈ 0.75 atol = 0.04
    @test mean(abs.(Z[2, :])) ≈ 0.952 atol = 0.01
    # misuse
    @test_throws ArgumentError implicit_sample(prod, zeros(2), zeros(2), ps, st; smc = true, corrector = :langevin)
    @test_throws ArgumentError implicit_sample(prod, zeros(2), zeros(2), ps, st; smc = true, resample = 1)
    @test_throws ArgumentError implicit_sample(prod, zeros(2), zeros(2), ps, st; corrector = :hmc)
    S2 = VPSDE(; βmax = 10.0)
    p2 = NoisePredictor(GaussianMixtureEps(S2, vcat(u', u'); s = 0.05), S2)
    @test_throws ArgumentError implicit_sample(ProductRelation(mC, ImplicitDiffusion(p2, noisefree_nodes(2, 0.01))),
                                               zeros(2), zeros(2), (pC, pL), (sC, sL))
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

@testset "categorical answers: classification as a query" begin
    # three classes in the plane, each a ring of 8 components plus its centre, with the label as
    # a one-hot code: the exact posterior p(class | x) is a ratio of Gaussian sums
    s = 0.15
    centres = [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.5]]
    pts = [(k, m .+ r .* [cos(a), sin(a)]) for (k, m) in enumerate(centres) for (r, a) in
           [(0.0, 0.0); [(0.4, a) for a in range(0, 2π; length = 9)[1:8]]]]
    μ = reduce(hcat, [[x; 1:3 .== k] for (k, x) in pts])
    p = NoisePredictor(GaussianMixtureEps(SCHED, float.(μ); s), SCHED)
    ps, st = LuxCore.setup(Xoshiro(0), p)
    nodes = field_nodes(Xoshiro(1), 5; samples = 8, levels = range(0.002, 0.02; length = 6))
    m = ImplicitDiffusion(p, nodes; λ = density_lambda(SCHED, nodes))
    codes = Matrix(1.0I, 3, 3)
    exact(x) = (l = [sum(exp(-sum(abs2, x .- y) / (2s^2)) for (k, y) in pts if k == c) for c in 1:3]; l ./ sum(l))
    ρ = [Inf, Inf, 0, 0, 0]
    for x in ([-1.0, 0.0], [0.0, 0.0], [-0.02, 0.0], [0.04, 0.0], [0.0, 0.8])
        a, _ = implicit_classify(m, [x; zeros(3)], ρ, ps, st; label = 3:5, classes = codes, labels = [:a, :b, :c])
        @test a.unassigned < 1e-8
        @test maximum(abs, probabilities(a.belief) .- exact(x)) < 0.04
        @test a.belief.labels == [:a, :b, :c]
        @test a.mixture isa MixtureBelief
    end
    # the boundary point: two answers of equal weight, the third class negligible
    a, _ = implicit_classify(m, [0.0, 0.0, 0, 0, 0], ρ, ps, st; label = [3, 4, 5], classes = codes)
    @test probabilities(a.belief) ≈ [0.5, 0.5, 0.0] atol = 1e-3
    # reading a mixture: assignment, summing, unassigned mass reported, not dropped
    G(μ) = Gaussian(μ, 0.01 * Matrix(1.0I, 3, 3))
    mix = MixtureBelief([G([0.9, 0.1, 0]), G([0, 1.05, 0]), G([0.95, 0, 0.05]), G([0.5, 0.5, 0.0])], [0.2, 0.3, 0.1, 0.4])
    r = categorical_answer(mix, codes)
    @test r.assignment == [1, 2, 1, 0]
    @test r.unassigned ≈ 0.4
    @test probabilities(r.belief) ≈ [0.5, 0.5, 0.0]
    @test categorical_answer(mix, codes; radius = 1.0).unassigned ≈ 0 atol = 1e-12
    @test categorical_answer(MixtureBelief([G([0.5, 0.5, 0.0])]), codes).belief === nothing
    @test_throws DimensionMismatch categorical_answer(mix, Matrix(1.0I, 2, 2))
    @test_throws ArgumentError implicit_classify(m, zeros(5), [Inf, Inf, Inf, 0, 0], ps, st; label = 3:5, classes = codes)
end

@testset "many queries at once: the same answers, shared network calls" begin
    # the circle: y given x for many x, against implicit_infer query by query
    m = ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2; samples = 8))
    xs = range(-0.9, 0.9; length = 7)
    Z₀ = vcat(collect(xs)', fill(0.5, 1, 7))
    sols, _ = implicit_infer_many(m, Z₀, HARDX, RPS, RST; tol = 1e-12, method = :newton)
    for (b, x) in enumerate(xs)
        ref, _ = implicit_infer(m, [x, 0.5], HARDX, RPS, RST; tol = 1e-12, method = :newton)   # the same iteration
        @test sols[b].z ≈ ref.z atol = 1e-10
        @test sols[b].converged && sols[b].stable && sols[b].iters == ref.iters
    end
    # the trust region (the default for this relation, which has an energy), from hard starts too:
    # each query takes the steps, radii and decisions of the single-query method
    # (not y = 0 at x = 0: that start is itself a root, the saddle between the branches, and any
    # solver stops there at once and reports it unstable)
    Zh = vcat(repeat(collect(xs)', 1, 4), repeat([0.5 0.01 6.0 -0.3], 1, 7)[:, 1:28])
    sols, _ = implicit_infer_many(m, Zh, HARDX, RPS, RST; maxiters = 50)
    for b in axes(Zh, 2)
        ref, _ = implicit_infer(m, Zh[:, b], HARDX, RPS, RST; maxiters = 50, method = :trust_region)
        @test sols[b].z ≈ ref.z atol = 1e-8
        @test sols[b].iters == ref.iters && sols[b].converged == ref.converged
    end
    @test all(s -> s.converged && s.stable, sols)
    @test count(s -> s.converged && s.stable,
                first(implicit_infer_many(m, Zh, HARDX, RPS, RST; maxiters = 50, method = :newton))) < size(Zh, 2)
    @test_throws ArgumentError implicit_infer_many(m, Z₀, HARDX, RPS, RST; method = :gauss)
    # starts on the other branch, and soft targets
    Zi = copy(Z₀); Zi[2, :] .= -0.5
    sols, _ = implicit_infer_many(m, Z₀, HARDX, RPS, RST; Z_init = Zi, tol = 1e-12)
    @test all(s -> s.z[2] < 0, sols)
    ρs = [Inf, 0.5]
    sols, _ = implicit_infer_many(m, Z₀, ρs, RPS, RST; tol = 1e-12)
    @test all(b -> sols[b].z ≈ first(implicit_infer(m, Z₀[:, b], ρs, RPS, RST; tol = 1e-12)).z, 1:7)
    @test_throws DimensionMismatch implicit_infer_many(m, Z₀, [Inf, 0.0, 0.0], RPS, RST)

    # a Lux network, batched (one call per noise level for all queries) and unbatched, with an
    # AD backend (the Jacobian blocks from VJPs) and without (from finite-difference JVPs)
    ps0, stm = Lux.setup(Xoshiro(0), MLP)
    psm = _f64(ps0)
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    Zq = [0.3 -0.2 0.6; 0.1 0.4 -0.3]
    for ad in (nothing, AutoForwardDiff(), AutoZygote()), batched in (false, true)
        pred = NoisePredictor(MLP, SCHED; input = _batch_input, ad)
        mm = ImplicitDiffusion(pred, nodes; batched)
        sols, _ = implicit_infer_many(mm, Zq, HARDX, psm, stm; tol = 1e-10, maxiters = 100)
        for b in 1:3
            ref, _ = implicit_infer(mm, Zq[:, b], HARDX, psm, stm; tol = 1e-10, maxiters = 100)
            @test sols[b].z ≈ ref.z rtol = 1e-6
            @test sols[b].converged == ref.converged
        end
    end

    # classification of many queries: the single-query answers
    s = 0.15
    centres = [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.5]]
    pts = [(k, c .+ r .* [cos(a), sin(a)]) for (k, c) in enumerate(centres) for (r, a) in
           [(0.0, 0.0); [(0.4, a) for a in range(0, 2π; length = 9)[1:8]]]]
    p = NoisePredictor(GaussianMixtureEps(SCHED, float.(reduce(hcat, [[x; 1:3 .== k] for (k, x) in pts])); s), SCHED)
    pp, sp = LuxCore.setup(Xoshiro(0), p)
    nodes5 = field_nodes(Xoshiro(1), 5; samples = 8, levels = range(0.002, 0.02; length = 6))
    mc = ImplicitDiffusion(p, nodes5; λ = density_lambda(SCHED, nodes5))
    X = [-1.0 -0.02 0.0 0.04 0.0 0.9; 0.0 0.0 0.0 0.0 0.8 0.1]
    Zc = vcat(X, zeros(3, size(X, 2)))
    ρ = [Inf, Inf, 0, 0, 0]
    many, _ = implicit_classify(mc, Zc, ρ, pp, sp; label = 3:5, classes = Matrix(1.0I, 3, 3))
    @test length(many) == size(X, 2)
    for b in axes(X, 2)
        one, _ = implicit_classify(mc, Zc[:, b], ρ, pp, sp; label = 3:5, classes = Matrix(1.0I, 3, 3))
        @test probabilities(many[b].belief) ≈ probabilities(one.belief) atol = 1e-6
        @test length(many[b].mixture.components) == length(one.mixture.components)
    end
end

@testset "noise blocks: a level per block, conditional fields, held clamps" begin
    # x and y as two blocks; the mixture stays exact with a noise level per coordinate
    blk = NoisePredictor(GaussianMixtureEps(SCHED, RING_μ; s = 0.05), SCHED; blocks = [[1], [2]])
    @test_throws ArgumentError NoisePredictor(MLP, SCHED; blocks = [[1], [1, 2]])
    @test nblocks(blk) == 2 && nblocks(RING) == 1
    x, h = [0.5, -0.7], 1e-6
    # every block at one level: the predictor without blocks
    @test first(epsilon(blk, x, 0.2, RPS, RST)) ≈ first(epsilon(RING, x, 0.2, RPS, RST)) atol = 1e-14
    @test epsilon_jacobian(blk, x, 0.2, RPS, RST) ≈ epsilon_jacobian(RING, x, 0.2, RPS, RST) atol = 1e-12
    # two levels: ε = -σ ⊙ ∇ log p_τ, both per coordinate
    τ = [0.02, 0.3]
    logp(x) = mixture_logdensity(blk.model, x, τ, RPS)
    @test first(epsilon(blk, x, τ, RPS, RST)) ≈ -sigma.(Ref(SCHED), τ) .* [_fd(logp, x, i) for i in 1:2] atol = 1e-7
    ε(x) = first(epsilon(blk, x, τ, RPS, RST))
    J = epsilon_jacobian(blk, x, τ, RPS, RST)
    @test J ≈ hcat([_fd(x -> ε(x), x, i) for i in 1:2]...) atol = 1e-6
    @test norm(J - J') > 1e-3                                            # σ scales the rows
    X, W, wb = [0.5 0.1 -0.9; -0.7 0.8 0.2], [0.3 -1.0 0.4; 0.6 0.2 -0.5], [0.2, 0.5, 0.3]
    Js = [epsilon_jacobian(blk, X[:, b], τ, RPS, RST) for b in 1:3]
    @test epsilon_jacobian_block(blk, X, τ, RPS, RST, [1, 2], wb) ≈ sum(wb .* Js) atol = 1e-12
    @test epsilon_jacobian_block(blk, X, τ, RPS, RST, [2], wb) ≈ sum(wb .* Js)[[2], [2]] atol = 1e-12
    @test epsilon_vjp_input(blk, X, τ, RPS, RST, W) ≈ hcat([Js[b]' * W[:, b] for b in 1:3]...) atol = 1e-12
    @test epsilon_jvp_input(blk, X, τ, RPS, RST, W) ≈ hcat([Js[b] * W[:, b] for b in 1:3]...) atol = 1e-12
    G = epsilon_vjp_params(blk, x, τ, RPS, RST, W[:, 1]).μ
    μp(δ) = (μ = (M = copy(RPS.μ); M[2, 7] += δ; M),)
    @test G[2, 7] ≈ (dot(first(epsilon(blk, x, τ, μp(h), RST)), W[:, 1]) - dot(first(epsilon(blk, x, τ, μp(-h), RST)), W[:, 1])) / 2h atol = 1e-6

    # the relation: annealed blocks are the relation without blocks; a held block conditions
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    z = [0.6, 0.5]
    @test first(prior_field(ImplicitDiffusion(blk, nodes; levels = [nothing, nothing]), z, RPS, RST)) ≈
          first(prior_field(ImplicitDiffusion(RING, nodes), z, RPS, RST)) atol = 1e-12
    @test conditional_levels(blk, HARDX) == [SCHED.tmin, nothing]
    @test_throws ArgumentError ImplicitDiffusion(RING, nodes; levels = [nothing, 0.1])
    @test_throws DimensionMismatch ImplicitDiffusion(blk, nodes; levels = [nothing])
    m = ImplicitDiffusion(blk, nodes; levels = conditional_levels(blk, HARDX))
    g(z) = first(prior_field(m, z, RPS, RST))
    U(z) = first(implicit_energy(m, z, RPS, RST))
    @test g(z) ≈ [_fd(U, z, i) for i in 1:2] atol = 1e-6                    # still the gradient of an energy
    Jm = prior_jacobian(m, z, RPS, RST)
    @test Jm ≈ hcat([_fd(z -> g(z), z, i) for i in 1:2]...) atol = 1e-5
    @test prior_jacobian(m, z, RPS, RST; coords = [2]) ≈ Jm[[2], [2]] atol = 1e-12
    @test prior_vjp(m, z, [0.3, -0.8], RPS, RST) ≈ Jm' * [0.3, -0.8] atol = 1e-12
    @test prior_jvp(m, z, [0.3, -0.8], RPS, RST) ≈ Jm * [0.3, -0.8] atol = 1e-12
    sol, _ = implicit_infer(m, z, HARDX, RPS, RST)
    @test sol.converged && sol.stable
    b = implicit_pullback(m, sol, z, HARDX, [0.0, 1.0], RPS, RST)
    yof(z₀, p = RPS) = first(implicit_infer(m, z₀, HARDX, p, RST; z_init = sol.z)).z[2]
    @test b.z₀[1] ≈ (yof([0.6 + h, 0.5]) - yof([0.6 - h, 0.5])) / 2h rtol = 1e-5
    @test b.ps.μ[2, 7] ≈ (yof(z, μp(h)) - yof(z, μp(-h))) / 2h atol = 1e-6
    # conditioning: with x held clean only y is smoothed, and the answer on the unit circle moves
    # towards the exact 0.8 (T2: smoothing x as well pulls the root inwards)
    joint, _ = implicit_infer(ImplicitDiffusion(RING, nodes), z, HARDX, RPS, RST)
    @test abs(sol.z[2] - 0.8) < abs(joint.z[2] - 0.8)

    # a Lux energy network over two blocks of one coordinate: ε = σ ⊙ ∇E with σ per block
    function blk_input(x, t)
        te = Zygote.@ignore begin
            f = vcat(sin.((2π .* kron(Array(t), FREQ))), cos.((2π .* kron(Array(t), FREQ))))
            x isa Array ? f : copyto!(similar(x, eltype(x), size(f)), f)
        end
        return vcat(x, te)
    end
    net = Chain(Dense(18 => 32, swish), Dense(32 => 32, swish), Dense(32 => 1))
    ps = _f64(first(Lux.setup(Xoshiro(0), net))); st = last(Lux.setup(Xoshiro(0), net))
    ad = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoZygote())
    en = NoisePredictor(EnergyNetwork(net), SCHED; input = blk_input, ad, blocks = [[1], [2]])
    E(x, τ, p = ps) = first(VariationalDiffusion.energy(en, x, τ, p, st))
    @test first(epsilon(en, x, τ, ps, st)) ≈ sigma.(Ref(SCHED), τ) .* [_fd(x -> E(x, τ), x, i) for i in 1:2] atol = 1e-7
    T2 = [0.02 0.3 0.1; 0.3 0.05 0.1]
    EX = first(epsilon(en, X, T2, ps, st))
    @test all(b -> EX[:, b] ≈ first(epsilon(en, X[:, b], T2[:, b], ps, st)), 1:3)
    Je = epsilon_jacobian(en, x, τ, ps, st)
    @test Je ≈ hcat([_fd(x -> first(epsilon(en, x, τ, ps, st)), x, i) for i in 1:2]...) atol = 1e-6
    Jes = [epsilon_jacobian(en, X[:, b], τ, ps, st) for b in 1:3]
    @test epsilon_vjp_input(en, X, τ, ps, st, W) ≈ hcat([Jes[b]' * W[:, b] for b in 1:3]...) atol = 1e-10
    @test epsilon_jvp_input(en, X, τ, ps, st, W) ≈ hcat([Jes[b] * W[:, b] for b in 1:3]...) atol = 1e-10
    @test epsilon_jacobian_block(en, X, τ, ps, st, [2], wb) ≈ sum(wb .* Jes)[[2], [2]] atol = 1e-10
    # weighted training on block times
    @test size(block_times(Xoshiro(1), en, 7)) == (2, 7)
    Tj = block_times(Xoshiro(1), en, 200; joint = 1.0)
    @test Tj[1, :] == Tj[2, :] && all(SCHED.tmin .≤ Tj .≤ 1)
    Tc = block_times(Xoshiro(1), en, 1000; clean = 1.0)               # one block held at tmin, either one
    @test all(==(1), vec(sum(Tc .== SCHED.tmin; dims = 1)))
    @test 400 < count(==(SCHED.tmin), Tc[1, :]) < 600
    @test 200 < count(c -> any(==(SCHED.tmin), c), eachcol(block_times(Xoshiro(2), en, 1000; clean = 0.3))) < 400
    Tb = block_times(Xoshiro(2), en, 5)
    x₀, εb, wts = randn(Xoshiro(3), 2, 5), randn(Xoshiro(4), 2, 5), [1.0, 7.0]
    L, gr, _ = denoising_gradient(en, x₀, Tb, εb, ps, st; weights = wts)
    Lw(p) = first(denoising_gradient(en, x₀, Tb, εb, p, st; weights = wts))
    V(δ) = merge(ps, (layer_1 = (weight = (A = copy(ps.layer_1.weight); A[3] += δ; A), bias = ps.layer_1.bias),))
    @test gr.layer_1.weight[3] ≈ (Lw(V(h)) - Lw(V(-h))) / 2h rtol = 1e-6
    xt = alpha.(Ref(SCHED), Tb) .* x₀ .+ sigma.(Ref(SCHED), Tb) .* εb             # one coordinate per block
    @test L ≈ sum(wts .* (first(epsilon(en, xt, Tb, ps, st)) .- εb) .^ 2) / 5 atol = 1e-12
    # held relation, batched and not, one query and many: the same field, block and answers
    mh = ImplicitDiffusion(en, nodes; levels = conditional_levels(en, HARDX))
    mb = ImplicitDiffusion(en, nodes; levels = conditional_levels(en, HARDX), batched = true)
    @test first(prior_field(mb, z, ps, st)) ≈ first(prior_field(mh, z, ps, st)) atol = 1e-12
    @test prior_jacobian(mb, z, ps, st; coords = [2]) ≈ prior_jacobian(mh, z, ps, st; coords = [2]) atol = 1e-10
    Ue(z) = first(implicit_energy(mh, z, ps, st))
    @test first(prior_field(mh, z, ps, st)) ≈ [_fd(Ue, z, i) for i in 1:2] atol = 1e-7
    Zq = [0.3 -0.2 0.6; 0.1 0.4 -0.3]
    for method in (:newton, :trust_region)
        sm, _ = implicit_infer_many(mb, Zq, HARDX, ps, st; tol = 1e-10, method)
        @test all(q -> sm[q].z ≈ first(implicit_infer(mh, Zq[:, q], HARDX, ps, st; tol = 1e-10, method)).z, 1:3)
    end

    # the same on a device (JLArrays): blocks' times move with x, σ is spread on the device
    JLArrays.allowscalar(false)
    to(a::AbstractArray) = JLArray(Float64.(a)); to(a) = a
    pd = Lux.fmap(a -> a isa AbstractArray ? JLArray(a) : a, ps)
    dv = NoisePredictor(EnergyNetwork(net), SCHED; input = blk_input, ad, blocks = [[1], [2]], device = (to = to, back = Array))
    @test first(epsilon(dv, X, T2, pd, st)) ≈ EX atol = 1e-12
    @test epsilon_vjp_input(dv, X, τ, pd, st, W) ≈ epsilon_vjp_input(en, X, τ, ps, st, W) atol = 1e-12
    @test epsilon_jvp_input(dv, X, τ, pd, st, W) ≈ epsilon_jvp_input(en, X, τ, ps, st, W) atol = 1e-12
    @test epsilon_jacobian_block(dv, X, τ, pd, st, [2], wb) ≈ epsilon_jacobian_block(en, X, τ, ps, st, [2], wb) atol = 1e-12
    _, gd, _ = denoising_gradient(dv, x₀, Tb, εb, pd, st; weights = wts)
    @test Array(gd.layer_1.weight) ≈ gr.layer_1.weight atol = 1e-12
    # in Float32 on the device (as gpu_device() converts): σ per coordinate must not widen to Float64
    to32(a::AbstractArray) = JLArray(Float32.(a)); to32(a) = a
    p32 = Lux.fmap(a -> a isa AbstractArray ? JLArray(Float32.(a)) : a, ps)
    d32 = NoisePredictor(EnergyNetwork(net), SCHED; input = blk_input, ad, blocks = [[1], [2]], device = (to = to32, back = Array))
    @test eltype(epsilon_vjp_input(d32, X, τ, p32, st, W)) == Float32
    @test epsilon_vjp_input(d32, X, τ, p32, st, W) ≈ epsilon_vjp_input(en, X, τ, ps, st, W) rtol = 1e-4

    # sampling with the clamp held clean: two narrow components, x clamped at one of them
    two = NoisePredictor(GaussianMixtureEps(SCHED, [-1.0 1.0; -1.0 1.0]; s = 0.1), SCHED; blocks = [[1], [2]])
    tps, tst = LuxCore.setup(Xoshiro(0), two)
    Y, _ = implicit_sample(ImplicitDiffusion(two, nodes), [1.0, 0.0], HARDX, tps, tst; n = 200, steps = 100,
                           clamp_level = SCHED.tmin, rng = Xoshiro(5))
    @test all(Y[1, :] .== 1.0)
    @test abs(mean(Y[2, :]) - 1) < 0.05 && 0.07 < std(Y[2, :]) < 0.13
    Yg, _ = implicit_sample(ImplicitDiffusion(two, nodes), [1.0, 0.0], HARDX, tps, tst; n = 200, steps = 100,
                            clamp_level = SCHED.tmin, guidance = 2.0, rng = Xoshiro(5))
    @test abs(mean(Yg[2, :]) - 1) < 0.05
    @test_throws ArgumentError implicit_sample(ImplicitDiffusion(RING, nodes), z, HARDX, RPS, RST; clamp_level = 0.001)
    @test_throws ArgumentError implicit_sample(ImplicitDiffusion(two, nodes), z, HARDX, tps, tst; guidance = 1.0)
    @test_throws ArgumentError implicit_sample(ImplicitDiffusion(two, nodes), z, HARDX, tps, tst; clamp_level = 0.001,
                                               corrector = :langevin)
end

@testset "a Gaussian reference energy: closed-form terms, exact against AD" begin
    Xd = [0.5 -1.0 0.2 0.9 0.0; 3.0 3.0 3.0 3.0 3.0]
    r = GaussianReference(Xd)
    @test r.μ ≈ vec(mean(Xd; dims = 2)) && r.v ≈ [std(Xd[1, :])^2, 1e-4]           # a constant coordinate: the floor
    @test_throws ArgumentError GaussianReference([0.0], [0.0])
    μ, v = [0.3, -0.4], [0.5, 0.02]
    ad = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoZygote())
    function blk_input(x, t)
        te = Zygote.@ignore begin
            f = vcat(sin.((2π .* kron(Array(t), FREQ))), cos.((2π .* kron(Array(t), FREQ))))
            x isa Array ? f : copyto!(similar(x, eltype(x), size(f)), f)
        end
        return vcat(x, te)
    end
    function dev_input(x, t)
        te = Zygote.@ignore begin
            tt = t isa Number ? fill(Float64(t), 1, size(x, 2)) : Array(t)
            f = vcat(sin.((2π .* tt) .* FREQ), cos.((2π .* tt) .* FREQ))
            x isa Array ? f : copyto!(similar(x, eltype(x), size(f)), f)
        end
        return vcat(x, te)
    end
    X, W, wb, h = [0.5 0.1 -0.9; -0.7 0.8 0.2], [0.3 -1.0 0.4; 0.6 0.2 -0.5], [0.2, 0.5, 0.3], 1e-6
    for (net, input, blocks, τ, Tb) in
        ((ENERGY_MLP, dev_input, nothing, 0.2, reshape([0.1, 0.5, 0.9], 1, 3)),
         (Chain(Dense(18 => 32, swish), Dense(32 => 32, swish), Dense(32 => 1)), blk_input, [[1], [2]], [0.02, 0.6],
          [0.02 0.3 0.9; 0.7 0.05 0.001]))
        ps = _f64(first(Lux.setup(Xoshiro(0), net))); st = last(Lux.setup(Xoshiro(0), net))
        cf = NoisePredictor(EnergyNetwork(net; reference = (μ, v)), SCHED; input, ad, blocks)
        ex = NoisePredictor(EnergyNetwork(RefExplicit(net, input, μ, v, blocks === nothing ? nothing : [1, 2])), SCHED;
                            input = (x, t) -> (x, t), ad, blocks)
        x = X[:, 1]
        @test first(VariationalDiffusion.energy(cf, x, τ, ps, st)) ≈ first(VariationalDiffusion.energy(ex, x, τ, ps, st)) atol = 1e-12
        @test first(VariationalDiffusion.energy_columns(cf, X, Tb, ps, st)) ≈ first(VariationalDiffusion.energy_columns(ex, X, Tb, ps, st)) atol = 1e-12
        @test first(epsilon(cf, X, Tb, ps, st)) ≈ first(epsilon(ex, X, Tb, ps, st)) atol = 1e-12
        @test first(epsilon(cf, x, τ, ps, st)) ≈ first(epsilon(ex, x, τ, ps, st)) atol = 1e-12
        @test epsilon_jacobian(cf, x, τ, ps, st) ≈ epsilon_jacobian(ex, x, τ, ps, st) atol = 1e-10
        @test epsilon_jacobian_block(cf, X, τ, ps, st, [2], wb) ≈ epsilon_jacobian_block(ex, X, τ, ps, st, [2], wb) atol = 1e-10
        @test epsilon_jacobian_block(cf, X, τ, ps, st, [1, 2], wb) ≈ epsilon_jacobian_block(ex, X, τ, ps, st, [1, 2], wb) atol = 1e-10
        @test epsilon_vjp_input(cf, X, τ, ps, st, W) ≈ epsilon_vjp_input(ex, X, τ, ps, st, W) atol = 1e-10
        @test epsilon_jvp_input(cf, X, τ, ps, st, W) ≈ epsilon_jvp_input(ex, X, τ, ps, st, W) atol = 1e-10
        εb = randn(Xoshiro(4), 2, 3)
        Lc, gc, _ = denoising_gradient(cf, X, Tb, εb, ps, st)
        Le, ge, _ = denoising_gradient(ex, X, Tb, εb, ps, st)
        @test Lc ≈ Le atol = 1e-12
        @test gc.layer_1.weight ≈ ge.layer_1.weight atol = 1e-10
        # the field of the relation, and its energy
        nodes = field_nodes(Xoshiro(1), 2; samples = 2)
        levels = blocks === nothing ? nothing : conditional_levels(cf, HARDX)
        mc, me = ImplicitDiffusion(cf, nodes; levels), ImplicitDiffusion(ex, nodes; levels)
        z = [0.6, 0.5]
        @test first(prior_field(mc, z, ps, st)) ≈ first(prior_field(me, z, ps, st)) atol = 1e-10
        @test first(implicit_energy(mc, z, ps, st)) ≈ first(implicit_energy(me, z, ps, st)) atol = 1e-10
        # on a device, in Float64 and in Float32
        JLArrays.allowscalar(false)
        to(a::AbstractArray) = JLArray(Float64.(a)); to(a) = a
        to32(a::AbstractArray) = JLArray(Float32.(a)); to32(a) = a
        for (dev, P, tol) in ((to, Lux.fmap(a -> a isa AbstractArray ? JLArray(a) : a, ps), 1e-12),
                              (to32, Lux.fmap(a -> a isa AbstractArray ? JLArray(Float32.(a)) : a, ps), 1e-4))
            dc = NoisePredictor(EnergyNetwork(net; reference = (μ, v)), SCHED; input,
                                ad, blocks, device = (to = dev, back = Array))
            @test first(epsilon(dc, X, Tb, P, st)) ≈ first(epsilon(cf, X, Tb, ps, st)) rtol = tol
            @test epsilon_vjp_input(dc, X, τ, P, st, W) ≈ epsilon_vjp_input(cf, X, τ, ps, st, W) rtol = tol
            @test epsilon_jacobian_block(dc, X, τ, P, st, [1, 2], wb) ≈ epsilon_jacobian_block(cf, X, τ, ps, st, [1, 2], wb) rtol = tol
            @test first(VariationalDiffusion.energy(dc, X, τ, P, st)) ≈ first(VariationalDiffusion.energy(cf, X, τ, ps, st)) rtol = tol
            dev === to32 && @test eltype(epsilon_vjp_input(dc, X, τ, P, st, W)) == Float32
        end
    end
end

@testset "single-root answers: code posteriors, calibration, flags, tolerances, continuation" begin
    classes = 2.0 .* Matrix(1.0I, 3, 3)
    G(μ, s) = LenticulumCore.Gaussian(μ, s^2 .* Matrix(1.0I, 3, 3))
    one_root(μ; s = 0.1) = LenticulumCore.MixtureBelief([G(μ, s)], [1.0])
    # a root on a code is that class; a root halfway between two codes splits evenly
    q = code_posterior(G([2.0, 0, 0], 0.1), classes, 0.2)
    @test sum(q) ≈ 1 && q[1] > 0.999
    a = categorical_answer(one_root([1.0, 1.0, 0.0]), classes; temperature = 0.5)
    p = probabilities(a.belief)
    @test p[1] ≈ p[2] atol = 1e-12
    @test p[3] < p[1]
    @test a.unassigned == 1.0                          # still reported: the root is between the codes
    # without a temperature the same root is not a class at all
    @test categorical_answer(one_root([1.0, 1.0, 0.0]), classes).belief === nothing
    # a wider root (larger Σ) or a higher temperature gives flatter probabilities
    pa = probabilities(categorical_answer(one_root([1.6, 0.3, 0.1]), classes; temperature = 0.3).belief)
    pb = probabilities(categorical_answer(one_root([1.6, 0.3, 0.1]; s = 0.8), classes; temperature = 0.3).belief)
    @test pb[1] < pa[1]
    # calibration recovers the spread of answers around their codes: for μ ~ N(c_k, s²I), the
    # exact posterior of k is the code posterior with τ = s
    rng = Xoshiro(3)
    truth = rand(rng, 1:3, 2000)
    answers = [one_root(classes[:, k] .+ 0.7 .* randn(rng, 3); s = 1e-4) for k in truth]
    cal = calibrate_temperature(answers, classes, truth; scale_bounds = (1, 1))
    @test abs(cal.temperature - 0.7) < 0.05 && cal.cov_scale == 1
    @test cal.nll < calibrate_temperature(answers, classes, truth; bounds = (3.0, 10.0), scale_bounds = (1, 1)).nll
    # a Laplace covariance four times too wide (the answers' spread is 0.7, Σ says 1.4): κ ≈ 1/4
    wide = [one_root(classes[:, k] .+ 0.7 .* randn(rng, 3); s = 1.4) for k in truth]
    cw = calibrate_temperature(wide, classes, truth; bounds = (1e-3, 0.1))
    @test abs(cw.cov_scale - 0.25) < 0.03
    @test cw.nll < calibrate_temperature(wide, classes, truth; bounds = (1e-3, 0.1), scale_bounds = (1, 1)).nll
    @test probabilities(categorical_answer(wide[1], classes; temperature = cw.temperature, cov_scale = cw.cov_scale).belief) ≈
          code_posterior(wide[1].components[1], classes, cw.temperature; cov_scale = cw.cov_scale) atol = 1e-12
    # classification passes the temperature on
    ring3 = NoisePredictor(GaussianMixtureEps(SCHED, [1.0 -1.0; 0.0 0.0]; s = 0.3), SCHED)
    m3 = ImplicitDiffusion(ring3, field_nodes(Xoshiro(1), 2; samples = 2); λ = density_lambda(SCHED, field_nodes(Xoshiro(1), 2; samples = 2)))
    rps, rst = LuxCore.setup(Xoshiro(0), ring3)
    ans, _ = implicit_classify(m3, [0.0, 0.0], [0.0, Inf], rps, rst; label = [1], classes = [1.0 -1.0], temperature = 0.5)
    @test sum(probabilities(ans.belief)) ≈ 1

    # coverage carries the unassigned mass and flags it
    data = RING_μ[:, rand(Xoshiro(1), 1:RING_J, 300)] .+ 0.05 .* randn(Xoshiro(2), 2, 300)
    ref = coverage_reference(ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2)), data, RPS, RST; ncal = 100)
    c0, c1 = coverage(ref, [1.0, 0.0]), coverage(ref, [1.0, 0.0]; unassigned = 0.8)
    @test !c0.flagged && c1.flagged && c1.unassigned == 0.8 && c1.p == c0.p
    @test coverage(ref, [3.0, 3.0]).flagged
    @test [c.flagged for c in coverage(ref, [1.0 1.0; 0.0 0.0]; unassigned = [0.0, 0.9])] == [false, true]

    # the default tolerance follows the precision the field is computed in
    mr = ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2; samples = 2))
    @test default_tolerance(mr, [0.0, 0.0], RPS) == 1e-9
    ps32, st32 = Lux.setup(Xoshiro(0), ENERGY_MLP)
    nod = field_nodes(Xoshiro(1), 2; samples = 2)
    cpu32 = ImplicitDiffusion(NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = _batch_input, ad = AutoZygote()), nod)
    dev32 = ImplicitDiffusion(NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = _batch_input, ad = AutoZygote(),
                                             device = (to = a -> JLArray(Float32.(a)), back = Array)), nod)
    @test default_tolerance(cpu32, [0.0, 0.0], ps32) == 1e-9               # Float32 weights, Float64 arithmetic
    @test default_tolerance(cpu32, Float32[0, 0], ps32) ≈ 30 * eps(Float32)
    @test default_tolerance(dev32, [0.0, 0.0], ps32) ≈ 30 * eps(Float32)   # on the device the inputs become Float32
    @test default_tolerance(ProductRelation(dev32, cpu32), [0.0, 0.0], (ps32, ps32)) ≈ 30 * eps(Float32)
    @test default_tolerance(cpu32, [0.0, 0.0], (a = ps32, b = _f64(ps32))) == 1e-9

    # continuation: smoother relations first, the iterations counted; from a start in the basin it
    # returns the same answer. On the circle a high level smooths the two branches into one
    # blob, and the continuation stops at the symmetric point between them (implicit.md §10).
    plain, _ = implicit_infer(mr, [0.0, 0.5], HARDX, RPS, RST)
    cont, _ = implicit_infer(mr, [0.0, 0.5], HARDX, RPS, RST; continuation = (3, 1.5))
    @test cont.converged && cont.stable
    @test cont.z ≈ plain.z atol = 1e-8
    @test cont.iters > plain.iters
    @test VariationalDiffusion._rescaled(mr, 30).nodes.t == min.(30 .* mr.nodes.t, 1.0)
    @test_throws ArgumentError implicit_infer(ProductRelation(mr, mr), [0.0, 0.5], HARDX,
                                              (RPS, RPS), (RST, RST); continuation = (2,))
    cs, _ = implicit_infer(mr, [0.6, 3.0], HARDX, RPS, RST; continuation = (8, 3), maxiters = 30, method = :newton)
    @test abs(cs.z[2]) < 0.1                           # Newton-or-descent: the symmetric point, not a branch
    ct, _ = implicit_infer(mr, [0.6, 3.0], HARDX, RPS, RST; continuation = (8, 3), maxiters = 30)
    @test ct.converged && ct.stable && abs(ct.z[2]) > 0.7   # the trust region walks off it, onto a branch
end

@testset "trust region: the exact step, robust convergence, both merits" begin
    VD = VariationalDiffusion
    # the dense step minimises the model within the ball: against a brute-force search on a grid
    model(g, H, p) = dot(g, p) + dot(p, H * p) / 2
    function brute(g, H, Δ)
        best, bp = Inf, zeros(2)
        for r in range(0, Δ; length = 401), θ in range(0, 2π; length = 721)
            p = r .* [cos(θ), sin(θ)]
            v = model(g, H, p)
            v < best && (best = v; bp = p)
        end
        return best
    end
    for (g, H, Δ) in (([1.0, 0.5], [2.0 0.3; 0.3 1.0], 5.0),          # interior Newton step
                      ([1.0, 0.5], [2.0 0.3; 0.3 1.0], 0.2),          # on the boundary
                      ([0.3, -1.0], [-1.0 0.2; 0.2 0.5], 0.7),        # indefinite
                      ([0.0, 1.0], [-1.0 0.0; 0.0 2.0], 0.9))         # hard case: g ⊥ lowest eigenvector
        p = VD._tr_step(DenseSolve(), g, Symmetric(H), Δ)
        @test norm(p) ≤ Δ * (1 + 1e-8)
        @test model(g, H, p) ≤ brute(g, H, Δ) + 1e-4
        q = VD._tr_step(KrylovSolve(), g, v -> H * v, Δ)         # Steihaug: within the ball, a decrease
        @test norm(q) ≤ Δ * (1 + 1e-8) && model(g, H, q) < 0
    end
    @test VD._tr_step(DenseSolve(), [1.0, 0.5], Symmetric([2.0 0.3; 0.3 1.0]), 5.0) ≈ -([2.0 0.3; 0.3 1.0] \ [1.0, 0.5])

    # on the circle every start reaches a branch, where Newton-or-descent stalls in a quarter
    m = ImplicitDiffusion(RING, field_nodes(Xoshiro(1), 2; samples = 2))
    starts = [[x, y] for x in range(-0.95, 0.95; length = 10) for y in (-6.0, -0.3, 0.0, 0.02, 1.0, 6.0)]
    tr = [first(implicit_infer(m, s, HARDX, RPS, RST; maxiters = 50, method = :trust_region)) for s in starts]
    nw = [first(implicit_infer(m, s, HARDX, RPS, RST; maxiters = 50, method = :newton)) for s in starts]
    @test all(i -> first(implicit_infer(m, starts[i], HARDX, RPS, RST; maxiters = 50)).z == tr[i].z, 1:6)   # the default
    @test all(s -> s.converged && s.stable, tr)
    @test count(s -> s.converged && s.stable, nw) < length(starts)
    @test all(s -> 0.95 < norm(s.z) < 1.0, tr)                          # on the (smoothed) circle
    @test all(i -> !(nw[i].converged && nw[i].stable) || abs(abs(nw[i].z[2]) - abs(tr[i].z[2])) < 1e-6, eachindex(starts))
    @test sum(s -> s.iters, tr) < sum(s -> s.iters, nw)
    kr = [first(implicit_infer(m, s, HARDX, RPS, RST; maxiters = 50, method = :trust_region, linsolve = KrylovSolve())) for s in starts[1:12]]
    @test all(i -> kr[i].z ≈ tr[i].z, 1:12)
    # with continuation, and a given starting radius
    c, _ = implicit_infer(m, [0.0, 3.0], HARDX, RPS, RST; method = :trust_region, continuation = (3,), radius = 1.0)
    @test c.converged && c.stable && abs(abs(c.z[2]) - 0.98) < 0.01
    @test_throws ArgumentError implicit_infer(m, [0.0, 3.0], HARDX, RPS, RST; method = :gauss)

    # without an energy (Gaussian oracle, correlated): ½‖r‖² with the Gauss–Newton model, same root
    pg = NoisePredictor(AnalyticEps(SCHED, [1.0 0.8; 0.8 1.0]), SCHED)
    gps, gst = LuxCore.setup(Random.default_rng(), pg)
    mg = ImplicitDiffusion(pg, field_nodes(Xoshiro(1), 2; samples = 2))
    @test !VD._has_energy(mg)
    a, _ = implicit_infer(mg, [0.7, 20.0], HARDX, gps, gst; method = :trust_region)
    b, _ = implicit_infer(mg, [0.7, 0.0], HARDX, gps, gst)
    @test a.converged && a.stable
    @test a.z ≈ b.z atol = 1e-8
end

@testset "cost model: pass counts, measured units, the break-even" begin
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)                 # 8 levels × 4 = 32 nodes
    m = ImplicitDiffusion(RING, nodes)
    u = (epsilon = 1.0, energy = 0.5, derivative = 3.0)
    N = length(nodes.t)
    # (I+1) N (|F| c_d + c_ε) + (I+1) N c_E for the trust region; Newton adds I N b c_ε instead
    @test root_cost(m, 2, u; iterations = 4) ≈ 5N * (2 * 3.0 + 1.0) + 5N * 0.5
    @test root_cost(m, 2, u; iterations = 4, method = :newton, backtracks = 2) ≈ 5N * (2 * 3.0 + 1.0) + 4N * 2 * 1.0
    @test root_cost(m, 2, u; iterations = 4, starts = 10) ≈ 10 * root_cost(m, 2, u; iterations = 4)
    @test classifier_cost(10, 32, u) == 320
    K = break_even_classes(m, 2, u; draws = 32, iterations = 4)
    @test classifier_cost(1, 32, u) * K ≈ root_cost(m, 2, u; iterations = 4)
    @test_throws ArgumentError root_cost(m, 2, (epsilon = 1.0, energy = nothing, derivative = 1.0); iterations = 3,
                                         method = :trust_region)
    # measured units: positive, the energy only for an energy network
    ps0, stm = Lux.setup(Xoshiro(0), MLP)
    eps_pred = NoisePredictor(MLP, SCHED; input = _batch_input, ad = AutoZygote())
    ue = pass_costs(eps_pred, randn(Xoshiro(2), 2, 20), 0.1, _f64(ps0), stm; directions = 2, reps = 1)
    @test ue.epsilon > 0 && ue.derivative > 0 && ue.energy === nothing
    pe, ste = Lux.setup(Xoshiro(0), ENERGY_MLP)
    en = NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = _batch_input, ad = AutoZygote())
    uE = pass_costs(en, randn(Xoshiro(2), 2, 20), 0.1, _f64(pe), ste; reps = 1)
    @test uE.energy > 0
end

@testset "a predictor on a device: every hook through device arrays" begin
    # JLArrays is the reference GPU array, on the CPU and without scalar indexing: what passes
    # here moves to CUDA unchanged. Time features are made on the CPU and placed beside x.
    JLArrays.allowscalar(false)
    to(x::AbstractArray) = JLArray(Float64.(x)); to(x) = x
    tojl(ps) = Lux.fmap(a -> a isa AbstractArray ? JLArray(a) : a, ps)
    function dev_input(x, t)
        te = Zygote.@ignore begin
            tt = t isa Number ? fill(Float64(t), 1, size(x, 2)) : Array(t)
            f = vcat(sin.((2π .* tt) .* FREQ), cos.((2π .* tt) .* FREQ))
            x isa Array ? f : copyto!(similar(x, Float64, size(f)), f)
        end
        return vcat(x, te)
    end
    X, W = randn(Xoshiro(1), 2, 6), randn(Xoshiro(2), 2, 6)
    t, w = 0.05, fill(1 / 6, 6)
    rel(a, b) = norm(Array(a) .- b) / norm(b)
    ps0, st = Lux.setup(Xoshiro(0), ENERGY_MLP)
    ps = _f64(ps0)
    ad = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoZygote())
    host = NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = dev_input, ad)
    dev = NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = dev_input, ad, device = (to = to, back = Array))
    pd = tojl(ps)
    @test first(epsilon(dev, X, t, pd, st)) isa Matrix
    @test rel(first(epsilon(dev, X, t, pd, st)), first(epsilon(host, X, t, ps, st))) < 1e-12
    @test rel(epsilon_vjp_input(dev, X, t, pd, st, W), epsilon_vjp_input(host, X, t, ps, st, W)) < 1e-12
    @test rel(epsilon_jacobian_block(dev, X, t, pd, st, [2], w), epsilon_jacobian_block(host, X, t, ps, st, [2], w)) < 1e-10
    @test first(VariationalDiffusion.energy(dev, X, t, pd, st)) ≈ first(VariationalDiffusion.energy(host, X, t, ps, st)) rtol = 1e-12
    # the training gradient: the cotangent stays with the parameters, on the device
    x₀, tb, εb = randn(Xoshiro(3), 2, 5), reshape([0.1, 0.2, 0.3, 0.4, 0.5], 1, 5), randn(Xoshiro(4), 2, 5)
    Lh, gh, _ = denoising_gradient(host, x₀, tb, εb, ps, st)
    Ld, gd, _ = denoising_gradient(dev, x₀, tb, εb, pd, st)
    @test Ld ≈ Lh rtol = 1e-12
    @test gd.layer_1.weight isa JLArray
    @test rel(gd.layer_1.weight, gh.layer_1.weight) < 1e-10
    # batched solves and classification-shaped queries give the host's answers
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    mh, md = ImplicitDiffusion(host, nodes; batched = true), ImplicitDiffusion(dev, nodes; batched = true)
    Zq = [0.3 -0.2 0.6; 0.1 0.4 -0.3]
    sh, _ = implicit_infer_many(mh, Zq, HARDX, ps, st; tol = 1e-10, maxiters = 50)
    sd, _ = implicit_infer_many(md, Zq, HARDX, pd, st; tol = 1e-10, maxiters = 50)
    @test all(b -> sd[b].z ≈ sh[b].z, 1:3)
    @test first(implicit_energy(md, Zq[:, 1], pd, st)) ≈ first(implicit_energy(mh, Zq[:, 1], ps, st)) rtol = 1e-12
    # an ε-network, with a reverse and a forward backend (the Jacobian block by rows and by columns)
    psm = _f64(first(Lux.setup(Xoshiro(0), MLP))); stm = last(Lux.setup(Xoshiro(0), MLP))
    for ad in (AutoZygote(), AutoForwardDiff())
        h = NoisePredictor(MLP, SCHED; input = dev_input, ad)
        g = NoisePredictor(MLP, SCHED; input = dev_input, ad, device = (to = to, back = Array))
        @test rel(epsilon_jacobian_block(g, X, t, tojl(psm), stm, [1, 2], w), epsilon_jacobian_block(h, X, t, psm, stm, [1, 2], w)) < 1e-10
        @test rel(epsilon_vjp_params(g, X[:, 1], t, tojl(psm), stm, W[:, 1]).layer_2.weight,
                  epsilon_vjp_params(h, X[:, 1], t, psm, stm, W[:, 1]).layer_2.weight) < 1e-10
    end
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
    # the trapezoidal scheme is second order: at 20 steps PDA's spread is 0.05 short, its is not
    sd(scheme) = (rng = Xoshiro(4);
                  X = reduce(hcat, [first(proxdm_sample(P, SCHED, randn(rng, 2), ps1, NamedTuple(); steps = 20, scheme, rng))
                                    for _ in 1:2000]);
                  (vec(mean(X; dims = 2)), mean(std(X; dims = 2))))
    (mp, sp), (mt, stz) = sd(:pda), sd(:trapezoidal)
    @test sp < s - 0.04
    @test abs(stz - s) < 0.015
    @test mt ≈ m atol = 0.03
    # on the ring (radial width 0.05) the same: 50 trapezoidal steps beat 100 of PDA
    radial(scheme, steps) = (rng = Xoshiro(5);
                             std([norm(first(proxdm_sample(MixtureProx(RING.model), SCHED, randn(rng, 2), RPS, RST; steps, scheme, rng)))
                                  for _ in 1:1000]))
    @test radial(:trapezoidal, 50) > radial(:pda, 100) + 0.003
    @test radial(:trapezoidal, 50) ≈ 0.05 atol = 0.01
    @test first(proxdm_sample(P, SCHED, [0.1, 0.2], ps1, NamedTuple(); steps = 10, hybrid = true, rng = Xoshiro(6))) ==
          first(proxdm_sample(P, SCHED, [0.1, 0.2], ps1, NamedTuple(); steps = 10, scheme = :hybrid, rng = Xoshiro(6)))
    @test_throws ArgumentError proxdm_sample(P, SCHED, randn(2), ps1, NamedTuple(); scheme = :heun)
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

@testset "ProxDM adjoint: prox_pullback against finite differences of prox_infer" begin
    P = MixtureProx(RING.model; tol = 1e-13)
    t, λ, c = 0.005, 0.01, [0.3, -0.7]
    _e(i, n) = float.(1:n .== i)
    # the prox's own derivatives: the closed-form Jacobian against differences
    v = [0.62, 0.71]
    @test prox_jacobian(P, v, t, λ, RPS, RST) ≈ invoke(prox_jacobian, NTuple{6,Any}, P, v, t, λ, RPS, RST) atol = 1e-6
    # hard clamp; soft target; soft clamp with a free coordinate; all coordinates soft
    for ρ in ([Inf, 0.0], [Inf, 2.0], [3.0, 0.0], [3.0, 1.0])
        z₀ = [0.6, 0.5]
        sol, _ = prox_infer(P, z₀, ρ, t, RPS, RST; λ, tol = 1e-12, maxiters = 5000)
        @test sol.converged
        # the reference runs a fixed 1000 sweeps (tol = 0): where the splitting contracts slowly, a
        # stopping rule leaves an error of tol/(1-κ), which a difference quotient amplifies by 1/h
        L(z0, r, ps) = dot(c, first(prox_infer(P, z0, r, t, ps, RST; λ, z_init = sol.z, tol = 0.0, maxiters = 1000)).z)
        b = prox_pullback(P, sol, z₀, ρ, t, c, RPS, RST; λ)
        h = 1e-4
        for i in 1:2
            @test b.z₀[i] ≈ (L(z₀ .+ h .* _e(i, 2), ρ, RPS) - L(z₀ .- h .* _e(i, 2), ρ, RPS)) / 2h rtol = 1e-4
            isinf(ρ[i]) || @test b.ρ[i] ≈ (L(z₀, ρ .+ h .* _e(i, 2), RPS) - L(z₀, ρ .- h .* _e(i, 2), RPS)) / 2h rtol = 1e-4
        end
        k = argmin(vec(sum(abs2, RING_μ .- sol.z; dims = 1)))           # the component nearest the answer
        for i in 1:2
            E = _e(i, 2) * _e(k, RING_J)'
            @test b.ps.μ[i, k] ≈ (L(z₀, ρ, (μ = RING_μ .+ h .* E,)) - L(z₀, ρ, (μ = RING_μ .- h .* E,))) / 2h rtol = 1e-4
        end
        # matrix-free: GMRES on prox VJPs gives the same cotangents
        bk = prox_pullback(P, sol, z₀, ρ, t, c, RPS, RST; λ, linsolve = KrylovSolve(rtol = 1e-12))
        @test bk.z₀ ≈ b.z₀ atol = 1e-12
        @test bk.ρ ≈ b.ρ atol = 1e-12
        @test bk.ps.μ ≈ b.ps.μ atol = 1e-12
    end
    bad, _ = prox_infer(P, [0.6, 0.5], HARDX, t, RPS, RST; λ, maxiters = 2)
    @test_throws ArgumentError prox_pullback(P, bad, [0.6, 0.5], HARDX, t, c, RPS, RST; λ)
    # Zygote differentiates through prox_infer with the rrule, which is the pullback
    sol, _ = prox_infer(P, [0.6, 0.5], HARDX, t, RPS, RST; λ)
    gz = Zygote.gradient(z -> dot(c, first(prox_infer(P, z, HARDX, t, RPS, RST; λ)).z), [0.6, 0.5])[1]
    @test gz == prox_pullback(P, sol, [0.6, 0.5], HARDX, t, c, RPS, RST; λ).z₀

    # a learned prox: a ProxNetwork with an AD backend
    input(v, t, λ) = vcat(v, [t, λ])
    net = Chain(Dense(4 => 16, tanh), Dense(16 => 2))
    pp0, pst = Lux.setup(Xoshiro(0), net)
    pp = _f64(pp0)
    pn = ProxNetwork(net, SCHED; input, ad = AutoZygote())
    pn0 = ProxNetwork(net, SCHED; input)                            # no backend: differences, and no parameter VJP
    y = [0.4, -0.9]
    @test prox_vjp_input(pn, v, 0.1, 0.2, pp, pst, y) ≈ prox_vjp_input(pn0, v, 0.1, 0.2, pp, pst, y) atol = 1e-8
    @test prox_jacobian(pn, v, 0.1, 0.2, pp, pst) ≈ prox_jacobian(pn0, v, 0.1, 0.2, pp, pst) atol = 1e-8
    @test_throws ArgumentError prox_vjp_params(pn0, v, 0.1, 0.2, pp, pst, y)
    sol, _ = prox_infer(pn, [0.6, 0.5], HARDX, 0.1, pp, pst; λ = 0.2, tol = 1e-12, maxiters = 5000)
    @test sol.converged
    b = prox_pullback(pn, sol, [0.6, 0.5], HARDX, 0.1, c, pp, pst; λ = 0.2)
    function Lw(δ)
        q = deepcopy(pp)
        q.layer_1.weight[3, 1] += δ
        return dot(c, first(prox_infer(pn, [0.6, 0.5], HARDX, 0.1, q, pst; λ = 0.2, z_init = sol.z, tol = 0.0, maxiters = 1000)).z)
    end
    @test b.ps.layer_1.weight[3, 1] ≈ (Lw(1e-3) - Lw(-1e-3)) / 2e-3 rtol = 1e-4
end

@testset "DiffusionFactor with ProxDM as its inversion: two parameter trees" begin
    P = MixtureProx(RING.model; tol = 1e-13)
    f = DiffusionFactor((x = 1, y = 1), RING; prox = ProxDM(P))
    @test f.prox.nodes isa FieldNodes && size(f.prox.nodes.ε, 1) == 2   # score nodes filled in
    @test_throws DimensionMismatch DiffusionFactor((x = 1, y = 1), RING;
                                                   prox = ProxDM(P; nodes = field_nodes(Xoshiro(1), 3)))
    ps, st = LuxCore.setup(Xoshiro(0), f)
    @test keys(ps) == (:predictor, :prox) && keys(st) == (:predictor, :prox)
    @test ps.prox.μ == RING_μ && ps.predictor.μ == RING_μ
    @test LuxCore.parameterlength(f) == 2 * length(RING_μ)
    p = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.0))
    lens, _ = LenticulumCore.assemble(f, p, ps, st)
    # the prior point is the warm start, and it selects the branch
    up, _ = LenticulumCore.invert(lens, DiracBelief([0.5]), (x = DiracBelief([0.6]),), ps, st)
    dn, _ = LenticulumCore.invert(lens, DiracBelief([-0.5]), (x = DiracBelief([0.6]),), ps, st)
    @test up isa DiracBelief
    @test up.value[1] ≈ 0.79 atol = 0.02
    @test dn.value[1] ≈ -up.value[1] atol = 1e-8
    # the same answer as the bare solver, with a report
    sol, _ = implicit_solution(f, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), ps, st)
    ref, _ = prox_infer(P, [0.6, 0.5], [Inf, 0.0], 0.005, ps.prox, st.prox; λ = 0.01)
    @test sol.converged && sol.z == ref.z
    msg, _ = Mycelium.factor_message(f, :y, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), ps, st)
    @test msg.value == up.value
    # the inversion reads the prox's tree only: moving the prior's means changes nothing
    ps_moved = (predictor = (μ = RING_μ .+ 0.3,), prox = ps.prox)
    @test first(LenticulumCore.invert(lens, DiracBelief([0.5]), (x = DiracBelief([0.6]),), ps_moved, st)).value == up.value
    # the free energy is deterministic, and its score summand reads the prior's tree
    F1, _ = Mycelium.local_free_energy(f, (x = DiracBelief([0.6]),), ps, st)
    F2, _ = Mycelium.local_free_energy(f, (x = DiracBelief([0.6]),), ps, st)
    F3, _ = Mycelium.local_free_energy(f, (x = DiracBelief([0.6]),), ps_moved, st)
    @test F1[:clamp] == F2[:clamp] && F1[:score] == F2[:score]
    @test F3[:score] != F1[:score]
    # the factor-level pullback agrees with finite differences of the factor's inversion
    ℓ(xin, ρy) = begin
        q = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = ρy))
        b, _ = LenticulumCore.invert(first(LenticulumCore.assemble(f, q, ps, st)), DiracBelief([0.5]),
                                     (x = DiracBelief([xin]),), ps, st)
        (b.value[1] - 0.7)^2 / 2
    end
    psoft = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.3))
    sol2, _ = implicit_solution(f, psoft, (x = DiracBelief([0.6]),), DiracBelief([0.5]), ps, st)
    g = implicit_factor_pullback(f, psoft, (x = DiracBelief([0.6]),), DiracBelief([0.5]),
                                 (y = [sol2.z[2] - 0.7],), ps, st)
    h = 1e-4
    @test g.inputs.x[1] ≈ (ℓ(0.6 + h, 0.3) - ℓ(0.6 - h, 0.3)) / 2h rtol = 1e-4
    @test g.precisions.y ≈ (ℓ(0.6, 0.3 + h) - ℓ(0.6, 0.3 - h)) / 2h rtol = 1e-4
    @test g.precisions.x == 0
    @test all(iszero, g.ps.predictor.μ)                   # the inversion does not read the prior
    @test size(g.ps.prox.μ) == size(RING_μ) && any(!iszero, g.ps.prox.μ)

    # a learned prox: a ProxNetwork's parameters are the factor's `prox` tree
    net = Chain(Dense(4 => 16, tanh), Dense(16 => 2))
    pn = ProxNetwork(net, SCHED; input = (v, t, λ) -> vcat(v, [t, λ]), ad = AutoZygote())
    fn = DiffusionFactor((x = 1, y = 1), RING; prox = ProxDM(pn; t = 0.1, λ = 0.2, tol = 1e-12, maxiters = 5000))
    psn, stn = LuxCore.setup(Xoshiro(0), fn)
    psn = (predictor = psn.predictor, prox = _f64(psn.prox))
    @test keys(psn.prox) == (:layer_1, :layer_2)
    gn = implicit_factor_pullback(fn, p, (x = DiracBelief([0.6]),), DiracBelief([0.5]), (y = [1.0],), psn, stn)
    @test size(gn.ps.prox.layer_1.weight) == (16, 4)
    function ℓw(δ)
        q = deepcopy(psn.prox)
        q.layer_1.weight[3, 1] += δ
        b, _ = LenticulumCore.invert(first(LenticulumCore.assemble(fn, p, psn, stn)), DiracBelief([0.5]),
                                     (x = DiracBelief([0.6]),), (predictor = psn.predictor, prox = q), stn)
        b.value[1]
    end
    @test gn.ps.prox.layer_1.weight[3, 1] ≈ (ℓw(1e-3) - ℓw(-1e-3)) / 2e-3 rtol = 1e-3
end

@testset "Phase 2 fixes: 3-channel assemble_state, prior cotangent, mixture message, orthogonal nodes" begin
    # 1. Orthogonal field_nodes preserves radial norms and orthogonalises columns within each block
    nodes_orth = field_nodes(Xoshiro(42), 3; levels = [0.01], samples = 3, antithetic = false, orthogonal = true)
    @test size(nodes_orth.ε) == (3, 3)
    G = nodes_orth.ε ./ [norm(nodes_orth.ε[:, j]) for j in 1:3]'
    @test G' * G ≈ I(3) atol = 1e-12
    # ... and its directions are unbiased without antithetic pairs (Householder QR fixes the
    # sign of each column unless R's diagonal signs are taken out; uncorrected, the first
    # direction's first coordinate averages -1/2)
    many = field_nodes(Xoshiro(7), 3; levels = [0.01], samples = 3 * 4000, antithetic = false, orthogonal = true)
    dirs = many.ε ./ [norm(c) for c in eachcol(many.ε)]'
    @test maximum(abs, sum(dirs[:, 1:3:end]; dims = 2) ./ 4000) < 0.05

    # 2. 3-channel assemble_state with polarity: π applies only to unobserved target channels
    f3 = DiffusionFactor((x = 1, y = 1, z = 1), NoisePredictor(GaussianMixtureEps(SCHED, randn(Xoshiro(1), 3, 4); s = 0.2), SCHED))
    p3 = Polarity((x = Observed(), y = Unobserved(), z = Unobserved()), (x = Inf, y = 0.5, z = 0.0))
    st_vec = VariationalDiffusion.assemble_state(f3, (x = DiracBelief([1.0]), z = DiracBelief([3.0])), DiracBelief([2.0]), p3)
    @test st_vec == [1.0, 2.0, 3.0]

    # 3. implicit_factor_pullback returns prior cotangent
    nodes2 = field_nodes(Xoshiro(1), 2; levels = [0.01], samples = 2)
    f2 = DiffusionFactor((x = 1, y = 1), RING; prox = ImplicitProx(nodes2))
    ps2, st2 = LuxCore.setup(Xoshiro(0), f2)
    psoft = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.4))
    g2 = implicit_factor_pullback(f2, psoft, (x = DiracBelief([0.6]),), DiracBelief([0.5]), (y = [1.0],), ps2, st2)
    @test haskey(g2, :prior) && length(g2.prior) == 1

    # 4. ImplicitProx with message = :mixture returns MixtureBelief on multimodal ring query
    f_mix = DiffusionFactor((x = 1, y = 1), RING; prox = ImplicitProx(nodes2; message = :mixture))
    lens_mix, _ = LenticulumCore.assemble(f_mix, Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.0)), ps2, st2)
    bmix, _ = LenticulumCore.invert(lens_mix, DiracBelief([0.0]), (x = DiracBelief([0.0]),), ps2, st2)
    @test bmix isa LenticulumCore.MixtureBelief
    @test length(bmix.components) >= 2
end

end
