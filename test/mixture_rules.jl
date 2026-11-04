# Extending Mycelium §4: expectation propagation and mixture reduction as message rules.
# Every result is checked against an oracle written out here: exact enumeration, a grid-free
# closed form, or Minka's EP loop by hand.
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, ForwardDiff
using LenticulumCore: MixtureBelief, DiracBelief, TrivialBelief, CategoricalBelief, Unobserved, Observed, Polarity

mr_lognorm(y, m, v) = -(y - m)^2 / (2v) - log(2π * v) / 2
mr_lognormal(x, μ, Σ) = -(dot(x - μ, Σ \ (x - μ)) + logdet(2π * Σ)) / 2
mr_lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))

# --- Minka's clutter problem (D = 1): x ~ N(0, 100), y | x ~ (1 − w) N(x, 1) + w N(0, 10) ----
const MR_W, MR_A, MR_V0 = 0.5, 10.0, 100.0

# The likelihood of one point as a message to x, a mixture of unnormalised exponentials:
# exp(yx − x²/2) with the rest of N(y; x, 1) in the weight, and the flat clutter term.
function mr_clutter_message(y)
    comps = [GaussianBelief([y], fill(1.0, 1, 1), true), GaussianBelief([0.0], zeros(1, 1), true)]
    l = [log(1 - MR_W) - y^2 / 2 - log(2π) / 2, log(MR_W) + mr_lognorm(y, 0, MR_A)]
    return MixtureBelief(Val(:log), comps, l .- mr_lse(l))
end

function mr_clutter_graph(ys, rule)
    b = GraphBuilder(); variable!(b, :x, 1)
    factor!(b, :prior, GaussianPrior(:x, 0.0, MR_V0)); connect!(b, :prior, :x, :x)
    for (i, y) in enumerate(ys)
        factor!(b, Symbol(:l, i), PriorFactor(:x, mr_clutter_message(y), x -> 0.0); rule)
        connect!(b, Symbol(:l, i), :x, :x)
    end
    return build(b)
end

# The exact posterior moments, by enumerating which points are signal.
function mr_clutter_exact(ys)
    N = length(ys); lws, ms, vs = Float64[], Float64[], Float64[]
    for S in 0:(2^N - 1)
        sig = [((S >> (i - 1)) & 1) == 1 for i in 1:N]
        J = 1 / MR_V0 + count(sig)
        ysg = ys[sig]; C = Matrix(1.0I, length(ysg), length(ysg)) .+ MR_V0
        le = isempty(ysg) ? 0.0 : -(dot(ysg, C \ ysg) + logdet(2π * C)) / 2
        push!(lws, le + count(sig) * log(1 - MR_W) + sum(log(MR_W) + mr_lognorm(y, 0, MR_A) for y in ys[.!sig]; init = 0.0))
        push!(ms, sum(ysg; init = 0.0) / J); push!(vs, 1 / J)
    end
    W = exp.(lws .- maximum(lws)); W ./= sum(W)
    μ = dot(W, ms)
    return μ, dot(W, vs .+ (ms .- μ) .^ 2)
end

# Minka's EP for this problem, by hand: sites (τᵢ, νᵢ) in natural parameters, an update skipped
# when its cavity precision is not positive.
function mr_clutter_ep(ys; sweeps = 2000, tol = 1e-13)
    N = length(ys); τ = zeros(N); ν = zeros(N)
    for _ in 1:sweeps
        δ = 0.0
        for i in 1:N
            τc = 1 / MR_V0 + sum(τ) - τ[i]; νc = sum(ν) - ν[i]
            τc > 0 || continue
            mc, vc = νc / τc, 1 / τc
            l1 = log(1 - MR_W) + mr_lognorm(ys[i], mc, vc + 1); l2 = log(MR_W) + mr_lognorm(ys[i], 0, MR_A)
            ρ = 1 / (1 + exp(l2 - l1))
            v1 = 1 / (1 / vc + 1); m1 = v1 * (mc / vc + ys[i])
            m = ρ * m1 + (1 - ρ) * mc
            v = ρ * (v1 + m1^2) + (1 - ρ) * (vc + mc^2) - m^2
            τn, νn = 1 / v - τc, m / v - νc
            δ = max(δ, abs(τn - τ[i]), abs(νn - ν[i])); τ[i], ν[i] = τn, νn
        end
        δ < tol && break
    end
    J = 1 / MR_V0 + sum(τ)
    return sum(ν) / J, 1 / J
end

@testset "Expectation propagation on the clutter problem" begin
    rng = Xoshiro(3)
    sets = [[rand(rng) < MR_W ? sqrt(MR_A) * randn(rng) : 2.0 + randn(rng) for _ in 1:N] for N in (4, 8, 12)]
    for ys in sets[2:3]
        μe, ve = mr_clutter_exact(ys)
        # Belief propagation is exact on this star: the marginal holds all 2ᴺ components. That needs
        # the product of two improper components (the likelihoods of two points).
        g = mr_clutter_graph(ys, BeliefPropagation())
        m, rep, _, _ = infer!(g, tree_schedule(g), NamedTuple(), NamedTuple())
        q = moment_match(m.x)
        @test length(m.x.components) == 2^length(ys)
        @test only(belief_mean(q)) ≈ μe atol = 1e-12
        @test only(belief_cov(q)) ≈ ve rtol = 1e-12
        # The rule reaches the fixed point of Minka's EP loop, on either schedule.
        μh, vh = mr_clutter_ep(ys)
        g = mr_clutter_graph(ys, ExpectationPropagationRule())
        for sched in (tree_schedule(g), SequentialSchedule(all_tasks(g)))
            m, rep, _, _ = infer!(g, sched, NamedTuple(), NamedTuple(); maxsweeps = 2000, tol = 1e-12)
            @test rep.converged
            @test m.x isa GaussianBelief
            @test only(belief_mean(m.x)) ≈ μh atol = 1e-10
            @test only(belief_cov(m.x)) ≈ vh rtol = 1e-10
        end
    end
    # With 12 points EP is close to the exact posterior (mean to 6e-4, variance to 2%); with 8 it is
    # not (mean off by 0.10), the regime Minka's paper also shows for few points.
    ys = sets[3]
    μe, ve = mr_clutter_exact(ys)
    m, _, _, _ = infer!(mr_clutter_graph(ys, ExpectationPropagationRule()), tree_schedule(mr_clutter_graph(ys, BeliefPropagation())),
                        NamedTuple(), NamedTuple(); maxsweeps = 2000, tol = 1e-12)
    @test abs(only(belief_mean(m.x)) - μe) < 1e-3
    @test abs(only(belief_cov(m.x)) - ve) < 0.02 * ve
    # With 4 points the undamped iteration oscillates, as Minka's loop does: the rule follows the
    # hand-written loop step for step on the same sequential order.
    ys = sets[1]
    g = mr_clutter_graph(ys, ExpectationPropagationRule())
    m, rep, _, _ = infer!(g, SequentialSchedule(all_tasks(g)), NamedTuple(), NamedTuple(); maxsweeps = 300, tol = 1e-12)
    μh, vh = mr_clutter_ep(ys; sweeps = 300, tol = 0.0)
    @test !rep.converged
    @test only(belief_mean(m.x)) ≈ μh atol = 1e-8
end

@testset "Expectation propagation: the rule's pieces" begin
    # A message in its family is sent as it is; an improper Gaussian cavity skips the update.
    gb = Gaussian([1.0], fill(2.0, 1, 1))
    @test ep_message(gb, Gaussian([0.0], fill(1.0, 1, 1))) === gb
    mix = mr_clutter_message(1.0)
    @test ep_message(mix, GaussianBelief([0.0], fill(-1.0, 1, 1), true)) === nothing
    @test ep_message(mix, DiracBelief([0.5])) === mix
    # The site times the cavity has the tilted moments.
    cav = Gaussian([0.3], fill(4.0, 1, 1))
    site = ep_message(mix, cav)
    tilted = Mycelium.combine(cav, mix)
    q = Mycelium.combine(cav, site)
    @test belief_mean(q) ≈ belief_mean(moment_match(tilted)) && belief_cov(q) ≈ belief_cov(moment_match(tilted))
    # Division can leave an indefinite site, declared improper.
    s = ep_divide(Gaussian([0.0], fill(2.0, 1, 1)), Gaussian([0.0], fill(1.0, 1, 1)))
    @test s.Λ ≈ fill(-0.5, 1, 1) && !isproper(s)
    @test ep_project(MixtureBelief([GaussianBelief([0.0], zeros(1, 1), true), gb], [0.5, 0.5])) === nothing
    # the sign test runs on values, so inference with ForwardDiff parameters does not hit eigvals of duals
    @test ForwardDiff.derivative(t -> (Mycelium._indefinite(GaussianBelief([t], fill(-t, 1, 1))) ? 1.0 : 0.0) * t, 1.0) == 1.0
    # On a Gaussian graph the rule is belief propagation.
    function lg(rule)
        b = GraphBuilder(); variable!(b, :x, 1); variable!(b, :y, 1)
        factor!(b, :prior, GaussianPrior(:x, 0.2, 1.5)); connect!(b, :prior, :x, :x)
        factor!(b, :lik, LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.3); rule)
        factor!(b, :d, DataFactor(:y, [0.9])); connect!(b, :d, :y, :y; direction = Emitting())
        connect!(b, :lik, :x, :x); connect!(b, :lik, :y, :y)
        return build(b)
    end
    lps = (prior = NamedTuple(), d = NamedTuple(), lik = (A = (x = fill(1.0, 1, 1), y = fill(-1.0, 1, 1)), c = [0.0]))
    mb, _, _, _ = infer!(lg(BeliefPropagation()), tree_schedule(lg(BeliefPropagation())), lps, NamedTuple())
    me, _, _, _ = infer!(lg(ExpectationPropagationRule()), tree_schedule(lg(BeliefPropagation())), lps, NamedTuple(); tol = 1e-14)
    @test me.x.η ≈ mb.x.η && me.x.Λ ≈ mb.x.Λ
    @test_throws ArgumentError ep_project(CategoricalBelief([0.5, 0.5]) |> b -> MixtureBelief([b, b], [0.5, 0.5]))
end

# --- A location-mixture likelihood on EP nodes: the free energy and its gradient --------------
const MR_YS = [0.3, 2.8, 3.4, -0.4, 3.1, 0.9]
function mr_location_graph(rule)
    b = GraphBuilder(); variable!(b, :x, 1)
    factor!(b, :prior, GaussianPrior(:x, 0.0, 25.0)); connect!(b, :prior, :x, :x)
    for (i, y) in enumerate(MR_YS)
        variable!(b, Symbol(:y, i), 1); variable!(b, Symbol(:k, i), 2)
        factor!(b, Symbol(:d, i), DataFactor(:y, [y]))
        factor!(b, Symbol(:s, i), SwitchingConstraintFactor((x = 1, y = 1), 2, 1; noise = 0.5); rule, params = :lik)
        connect!(b, Symbol(:d, i), :y, Symbol(:y, i); direction = Emitting())
        connect!(b, Symbol(:s, i), :y, Symbol(:y, i)); connect!(b, Symbol(:s, i), :x, :x)
        connect!(b, Symbol(:s, i), :k, Symbol(:k, i))
    end
    return build(b)
end
mr_location_ps(θ) = merge((prior = NamedTuple(),),
    NamedTuple{Tuple(Symbol(:d, i) for i in eachindex(MR_YS))}(ntuple(_ -> NamedTuple(), length(MR_YS))), (lik = θ,))

@testset "Expectation propagation: the free energy is stationary at a fixed point" begin
    g = mr_location_graph(ExpectationPropagationRule())
    θ0 = (A = (x = fill(1.0, 1, 1), y = fill(-1.0, 1, 1)), b = [0.0 -3.0], logw = [0.2, -0.1])
    function F(θ)
        s = MessageStore(g)
        rep, _ = propagate!(s, g, tree_schedule(g), mr_location_ps(θ), NamedTuple(); maxsweeps = 1000, tol = 1e-13)
        @test rep.converged
        return first(scalar_free_energy(s, g, mr_location_ps(θ), NamedTuple())), s
    end
    _, s0 = F(θ0)
    # x gets Gaussian sites; the messages to the observed y stay mixtures, irrelevant beside the point.
    @test Mycelium.marginal(s0, g, g.variable_index[:x]) isa GaussianBelief
    # The gradient at fixed messages is the total derivative: the free energy of EP nodes is
    # stationary in the messages at a fixed point (envelope theorem).
    gr = free_energy_gradient(s0, g, mr_location_ps(θ0), NamedTuple())
    h = 1e-5
    for (fld, idx) in ((:b, 2), (:logw, 1), (:b, 1))
        vp = copy(getfield(θ0, fld)); vp[idx] += h
        vm = copy(getfield(θ0, fld)); vm[idx] -= h
        fd = (first(F(merge(θ0, NamedTuple{(fld,)}((vp,))))) - first(F(merge(θ0, NamedTuple{(fld,)}((vm,)))))) / 2h
        @test getfield(gr.lik, fld)[idx] ≈ fd rtol = 1e-7
    end
    # Paths that differentiate through inference are written for belief propagation and refuse it.
    @test_throws ArgumentError free_energy_gradient(s0, g, mr_location_ps(θ0), NamedTuple(); adjoint = :implicit)
end

@testset "mixture_message: mixture inputs expanded" begin
    rng = Xoshiro(5)
    f = SwitchingConstraintFactor((x = 2, u = 2), 3, 2; noise = [0.5 0.1; 0.1 0.4])
    ps, st = LuxCore.setup(rng, f)
    cs = [Gaussian(randn(rng, 2), (B = randn(rng, 2, 2); B * B' + 0.3I)) for _ in 1:3]
    vw = [0.5, 0.3, 0.2]
    umix = MixtureBelief(cs, vw)
    kb = CategoricalBelief([0.2, 0.5, 0.3])
    w = exp.(ps.logw .- mr_lse(ps.logw))
    R(j) = f.Q + ps.A.u * belief_cov(cs[j]) * ps.A.u'
    xs = [randn(rng, 2) for _ in 1:4]
    # to x: log m(x) = log Σⱼ vⱼ Σₖ πₖ wₖ N(Aₓx + A_u μⱼ − bₖ; 0, Q + A_u Sⱼ A_uᵀ), up to a constant
    pol = Polarity(; x = Unobserved(), u = Observed(), k = Observed())
    m, _ = mixture_message(f, :x, pol, (u = umix, k = kb), nothing, ps, st)
    logm(x) = mr_lse([log(vw[j]) + kb.logp[k] + log(w[k]) + mr_lognormal(ps.A.x * x + ps.A.u * belief_mean(cs[j]), ps.b[:, k], R(j))
                      for j in 1:3 for k in 1:3])
    @test length(m.components) == 9
    for x in xs[2:end]
        @test Mycelium.belief_logdensity(m, x) - Mycelium.belief_logdensity(m, xs[1]) ≈ logm(x) - logm(xs[1]) atol = 1e-12
    end
    # to the outcome
    mk, _ = mixture_message(f, :k, Polarity(; x = Observed(), u = Observed(), k = Unobserved()), (x = DiracBelief(xs[1]), u = umix), nothing, ps, st)
    lk = [mr_lse([log(vw[j]) + log(w[k]) + mr_lognormal(ps.A.x * xs[1] + ps.A.u * belief_mean(cs[j]), ps.b[:, k], R(j)) for j in 1:3]) for k in 1:3]
    @test mk.logp ≈ lk .- mr_lse(lk) atol = 1e-12
    # without mixture inputs it is factor_message
    m1, _ = mixture_message(f, :x, pol, (u = cs[1], k = kb), nothing, ps, st)
    m2, _ = Mycelium.factor_message(f, :x, pol, (u = cs[1], k = kb), nothing, ps, st)
    @test exp.(m1.logw) ≈ exp.(m2.logw) atol = 1e-14
    @test all(c -> m1.components[c].η ≈ m2.components[c].η && m1.components[c].Λ ≈ m2.components[c].Λ, 1:3)
    # a flat input beside the mixture: the limit of a wide prior N(0, τI) on it
    f3 = SwitchingConstraintFactor((x = 2, u = 2, z = 1), 2, 2; noise = [0.5 0.1; 0.1 0.4])
    p3, _ = LuxCore.setup(Xoshiro(6), f3)
    m3, _ = mixture_message(f3, :x, Polarity(; x = Unobserved(), u = Observed(), z = Unobserved(), k = Unobserved()),
                            (u = umix,), nothing, p3, st)
    τ = 1e8
    w3 = exp.(p3.logw .- mr_lse(p3.logw))
    logm3(x) = mr_lse([log(vw[j]) + log(w3[k]) + mr_lognormal(p3.A.x * x + p3.A.u * belief_mean(cs[j]), p3.b[:, k],
                       f3.Q + p3.A.u * belief_cov(cs[j]) * p3.A.u' + τ * p3.A.z * p3.A.z') for j in 1:3 for k in 1:2])
    # the projected residual has one direction, so the components are improper (unnormalised
    # exponentials), and the message is read as one
    @test all(c -> !isproper(c), m3.components)
    unnorm(m, x) = mr_lse(m.logw .+ [dot(c.η, x) - dot(x, c.Λ * x) / 2 for c in m.components])
    for x in xs[2:end]
        @test unnorm(m3, x) - unnorm(m3, xs[1]) ≈ logm3(x) - logm3(xs[1]) atol = 1e-6
    end
    # LinearConstraintFactor
    lf = LinearConstraintFactor((x = 2, u = 2), 2; noise = [0.5 0.1; 0.1 0.4])
    lps, _ = LuxCore.setup(rng, lf)
    ml, _ = mixture_message(lf, :x, Polarity(; x = Unobserved(), u = Observed()), (u = umix,), nothing, lps, NamedTuple())
    logl(x) = mr_lse([log(vw[j]) + mr_lognormal(lps.A.x * x + lps.A.u * belief_mean(cs[j]), lps.c, lf.Q + lps.A.u * belief_cov(cs[j]) * lps.A.u') for j in 1:3])
    for x in xs[2:end]
        @test Mycelium.belief_logdensity(ml, x) - Mycelium.belief_logdensity(ml, xs[1]) ≈ logl(x) - logl(xs[1]) atol = 1e-12
    end
    # refusals: an improper mixture component, and a factor without the method
    bad = MixtureBelief([GaussianBelief(zeros(2), zeros(2, 2), true), cs[1]], [0.5, 0.5])
    @test_throws ArgumentError mixture_message(f, :x, pol, (u = bad, k = kb), nothing, ps, st)
    # a target the factor cannot reach, with every outcome impossible: refused, as by factor_message
    f0 = SwitchingConstraintFactor((x = 1, u = 2), 2, 2; noise = 0.5)
    p0 = (A = (x = zeros(2, 1), u = Matrix(1.0I, 2, 2)), b = zeros(2, 2), logw = zeros(2))
    @test_throws ArgumentError mixture_message(f0, :x, nothing, (u = umix, k = CategoricalBelief(Val(:log), [-Inf, -Inf], nothing)), nothing, p0, st)
    @test_throws ArgumentError Mycelium.factor_message(f0, :x, nothing, (u = cs[1], k = CategoricalBelief(Val(:log), [-Inf, -Inf], nothing)), nothing, p0, st)
    # an outcome of weight zero stays impossible (not NaN) in the message to the outcome
    pz = merge(ps, (logw = [0.0, -Inf, 0.0],))
    mz, _ = mixture_message(f, :k, Polarity(; x = Observed(), u = Observed(), k = Unobserved()), (x = DiracBelief(xs[1]), u = umix), nothing, pz, st)
    @test mz.logp[2] == -Inf && all(isfinite, mz.logp[[1, 3]]) && sum(exp, mz.logp) ≈ 1
    gf = GaussianFactor(1 => 1; noise = 0.3)
    @test_throws ArgumentError mixture_message(gf, :y, nothing, (x = umix,), nothing, NamedTuple(), NamedTuple())
end

# --- A switching linear chain: x_{t+1} = x_t + β_{k_t} + ε, y_t = x_t + δ, independent k_t ----
const MR_T = 5
const MR_CHAIN = (m0 = 0.0, v0 = 1.0, q = 0.1, r = 0.2, offs = [0.0, 1.5], wk = [0.6, 0.4], ys = [0.1, 1.4, 1.7, 3.1, 3.0])

function mr_chain(rule)
    c = MR_CHAIN; b = GraphBuilder()
    for t in 1:MR_T
        variable!(b, Symbol(:x, t), 1); variable!(b, Symbol(:y, t), 1)
        factor!(b, Symbol(:d, t), DataFactor(:y, [c.ys[t]])); connect!(b, Symbol(:d, t), :y, Symbol(:y, t); direction = Emitting())
        factor!(b, Symbol(:o, t), LinearConstraintFactor((x = 1, y = 1), 1; noise = c.r); params = :obs)
        connect!(b, Symbol(:o, t), :x, Symbol(:x, t)); connect!(b, Symbol(:o, t), :y, Symbol(:y, t))
    end
    factor!(b, :prior, GaussianPrior(:x1, c.m0, c.v0)); connect!(b, :prior, :x1, :x1)
    for t in 1:(MR_T - 1)
        variable!(b, Symbol(:k, t), 2)
        factor!(b, Symbol(:f, t), SwitchingConstraintFactor((a = 1, b = 1), 2, 1; noise = c.q); rule, params = :dyn)
        connect!(b, Symbol(:f, t), :a, Symbol(:x, t)); connect!(b, Symbol(:f, t), :b, Symbol(:x, t + 1))
        connect!(b, Symbol(:f, t), :k, Symbol(:k, t))
    end
    return build(b)
end
const MR_CHAIN_PS = merge(
    (obs = (A = (x = fill(1.0, 1, 1), y = fill(-1.0, 1, 1)), c = [0.0]),
     dyn = (A = (a = fill(-1.0, 1, 1), b = fill(1.0, 1, 1)), b = reshape(MR_CHAIN.offs, 1, 2), logw = log.(MR_CHAIN.wk)),
     prior = NamedTuple()),
    NamedTuple{Tuple(Symbol(:d, t) for t in 1:MR_T)}(ntuple(_ -> NamedTuple(), MR_T)))

# Exact marginal moments, by enumerating the switches: given them the chain is Gaussian, with one
# precision J for every sequence; the sequence's weight is wₛ exp(hᵀJ⁻¹h/2 − c/2).
function mr_chain_exact()
    c = MR_CHAIN; T = MR_T
    J = zeros(T, T); J[1, 1] += 1 / c.v0
    for t in 1:T; J[t, t] += 1 / c.r; end
    for t in 1:(T - 1)
        J[t, t] += 1 / c.q; J[t + 1, t + 1] += 1 / c.q; J[t, t + 1] -= 1 / c.q; J[t + 1, t] -= 1 / c.q
    end
    Σ = inv(J); lw = Float64[]; M = Vector{Float64}[]
    for seq in Iterators.product(ntuple(_ -> 1:2, T - 1)...)
        h = zeros(T); h[1] += c.m0 / c.v0; h .+= c.ys ./ c.r; κ = 0.0; l = 0.0
        for t in 1:(T - 1)
            β = c.offs[seq[t]]; h[t + 1] += β / c.q; h[t] -= β / c.q; κ += β^2 / c.q; l += log(c.wk[seq[t]])
        end
        μ = Σ * h; push!(M, μ); push!(lw, l - κ / 2 + dot(h, μ) / 2)
    end
    W = exp.(lw .- maximum(lw)); W ./= sum(W)
    μ = sum(W .* M)
    return μ, [sum(W[i] * (Σ[t, t] + (M[i][t] - μ[t])^2) for i in eachindex(W)) for t in 1:T]
end

@testset "MixtureReduction on a switching linear chain against enumeration" begin
    μe, ve = mr_chain_exact()
    function chain_error(rule)
        g = mr_chain(rule)
        mg, rep, _, _ = infer!(g, tree_schedule(g), MR_CHAIN_PS, NamedTuple())
        @test rep.converged
        qs = [moment_match(getfield(mg, Symbol(:x, t))) for t in 1:MR_T]
        return maximum(t -> max(abs(only(belief_mean(qs[t])) - μe[t]), abs(only(belief_cov(qs[t])) - ve[t])), 1:MR_T)
    end
    errs = [chain_error(MixtureReduction(M)) for M in (1, 2, 4, 8, 16)]
    @test issorted(errs; rev = true)
    @test errs[1] > 0.1                     # one component: assumed-density filtering on each message
    @test errs[end] < 1e-12                 # 16 = 2⁴ components: nothing merged, exact
    # Belief propagation collapses mixture inputs and sends K = 2 components: here the budget-2 result.
    @test chain_error(BeliefPropagation()) ≈ errs[2] rtol = 1e-8
    # The free energy is not defined for these nodes, and is refused.
    g = mr_chain(MixtureReduction(4))
    _, _, _, s = infer!(g, tree_schedule(g), MR_CHAIN_PS, NamedTuple())
    @test_throws ArgumentError scalar_free_energy(s, g, MR_CHAIN_PS, NamedTuple())
    @test_throws ArgumentError MixtureReduction(0)
    # Runnalls' merge needs moments: an improper mixture over the budget is refused.
    @test_throws ArgumentError reduce_message(mr_clutter_message(1.0), 1)
    @test reduce_message(mr_clutter_message(1.0), 2) isa MixtureBelief
    @test reduce_message(MixtureBelief([Gaussian(0.0, 1.0), Gaussian(1.0, 2.0)], [0.5, 0.5]), 1) isa GaussianBelief
end

@testset "The product of two improper mixture components" begin
    # exp(η₁x − Λ₁x²/2)·exp(η₂x − Λ₂x²/2), proper: normalised, its mass in logZ
    c = GaussianBelief([0.7], fill(1.0, 1, 1), true); d = GaussianBelief([-0.2], fill(0.5, 1, 1), true)
    cd, logZ = Mycelium.combine_with_overlap(c, d)
    η, Λ = 0.5, 1.5
    @test isproper(cd) && cd.η ≈ [η] && cd.Λ ≈ fill(Λ, 1, 1)
    @test logZ ≈ η^2 / (2Λ) + log(2π / Λ) / 2
    # flat times a likelihood with a null direction: still improper, logZ = 0
    e, lz = Mycelium.combine_with_overlap(GaussianBelief(zeros(2), zeros(2, 2), true), GaussianBelief([1.0, 0.0], [1.0 0.0; 0.0 0.0], true))
    @test !isproper(e) && lz == 0
end

@testset "Expectation propagation: a skipped update is not stored by flooding or residual schedules" begin
    # No prior on x: the first EP update of each point has no proper cavity and is skipped. A
    # skipped update must leave the edge without a message (review of 2026-11-04, ChatGPT P1).
    b = GraphBuilder(); variable!(b, :x, 1)
    for (i, y) in enumerate((0.5, 2.0))
        factor!(b, Symbol(:l, i), PriorFactor(:x, mr_clutter_message(y), x -> 0.0); rule = ExpectationPropagationRule())
        connect!(b, Symbol(:l, i), :x, :x)
    end
    g = build(b)
    ts = [t for t in all_tasks(g) if t.kind === :to_variable]
    s = MessageStore(g)
    @test_logs (:warn, r"skipped an expectation-propagation update") match_mode = :any sweep!(s, g, FloodingSchedule(ts), NamedTuple(), NamedTuple())
    @test all(t -> s.to_variable[t.edge] === nothing, ts)
    # the residual schedule never selects a task it cannot commit
    rec = MessageTask[]
    s2 = MessageStore(g)
    sweep!(s2, g, ResidualSchedule(4), NamedTuple(), NamedTuple(); record = rec)
    @test !any(t -> t in rec, ts) && all(t -> s2.to_variable[t.edge] === nothing, ts)
end
