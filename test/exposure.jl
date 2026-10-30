using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, Test
import LuxCore
using LenticulumCore: DiracBelief, Observed, Unobserved, Latent, Polarity
isdefined(@__MODULE__, :HierarchicalModels) ||
    include(joinpath(@__DIR__, "..", "examples", "hierarchical_models.jl"))

# L-SPEC-003 §1: exposed parameters. Every reference below is dense linear algebra that
# shares no code with the factors.

_F(store, g, ps, st) = first(LenticulumCore.scalar_free_energy(store, g, ps, st))

@testset "L-SPEC-003 §1.1 a clamped exposed block is the original factor" begin
    A, bias = [0.7 -0.3; 0.2 0.5; 1.1 0.4], [0.1, -0.2, 0.3]
    f = GaussianFactor(2 => 3; noise = [0.5, 0.2, 0.3])
    yobs = [1.0, -0.5, 2.0]
    function graph(exposed)
        b = GraphBuilder()
        variable!(b, :x, 2); variable!(b, :y, 3)
        factor!(b, :prior, GaussianPrior(:x, [0.3, -0.1], [1.0 0.2; 0.2 2.0]))
        factor!(b, :data, DataFactor(:y, yobs))
        connect!(b, :prior, :x, :x)
        connect!(b, :data, :y, :y; direction = Emitting())
        if exposed
            factor!(b, :lik, expose(f, (:A => :weight, :b)))
            variable!(b, :weight, 6); variable!(b, :bias, 3)
            factor!(b, :wdata, DataFactor(:weight, vec(A)))
            factor!(b, :bdata, DataFactor(:b, bias))
            # clamped weights are data: absorb only (a message back to them would be bilinear in x)
            connect!(b, :lik, :weight, :weight; direction = Absorbing())
            connect!(b, :lik, :b, :bias; direction = Absorbing())
            connect!(b, :wdata, :weight, :weight; direction = Emitting())
            connect!(b, :bdata, :b, :bias; direction = Emitting())
        else
            factor!(b, :lik, f)
        end
        connect!(b, :lik, :x, :x); connect!(b, :lik, :y, :y)
        g = build(b)
        ps = merge(LuxCore.initialparameters(Xoshiro(1), g), exposed ? (lik = NamedTuple(),) : (lik = (A = A, b = bias),))
        return g, ps, LuxCore.initialstates(Xoshiro(1), g)
    end
    g0, ps0, st0 = graph(false)
    g1, ps1, st1 = graph(true)
    @test keys(ps1.lik) == ()                                    # both blocks moved out of ps
    m0, r0, st0, s0 = infer!(g0, tree_schedule(g0), ps0, st0)
    m1, r1, st1, s1 = infer!(g1, tree_schedule(g1), ps1, st1)
    @test r0.converged && r1.converged
    @test belief_mean(m1.x) ≈ belief_mean(m0.x) && belief_cov(m1.x) ≈ belief_cov(m0.x)
    @test _F(s1, g1, ps1, st1) ≈ _F(s0, g0, ps0, st0)
    # the posterior on x, densely: y = A x + b + ε
    Σ0, Q = [1.0 0.2; 0.2 2.0], Diagonal([0.5, 0.2, 0.3])
    S = A * Σ0 * A' + Q
    @test belief_mean(m1.x) ≈ [0.3, -0.1] + Σ0 * A' * (S \ (yobs - A * [0.3, -0.1] - bias))
    # a constraint factor exposes its constant the same way
    c = LinearConstraintFactor((a = 1, b = 1), 1; noise = 0.1)
    e = expose(c, (:c => :level,))
    @test keys(LuxCore.initialparameters(Xoshiro(0), e)) == (:A,)
    @test LenticulumCore.Channel(:level, 1) in LenticulumCore.channels(e)
end

# §1.1 directly: with the exposed blocks given as points, messages and the local free energy
# are the original factor's with those blocks substituted.
function _check_clamped(f, names, target, pol, inputs, prior, ps)
    e = expose(f, names)
    chans = map(n -> n isa Pair ? last(n) : n, names)
    params = map(n -> n isa Pair ? first(n) : n, names)
    pts = NamedTuple{chans}(map(p -> DiracBelief(vec(getfield(ps, p))), params))
    keep = Tuple(k for k in keys(ps) if !(k in params))
    pse = NamedTuple{keep}(map(k -> getfield(ps, k), keep))
    pe = Polarity(merge(pol.assignment, NamedTuple{chans}(map(_ -> Observed(), chans))))
    @test LenticulumCore.supports_polarity(e, pe)
    m0, _ = Mycelium.factor_message(f, target, pol, inputs, prior, ps, NamedTuple())
    m1, _ = Mycelium.factor_message(e, target, pe, merge(inputs, pts), prior, pse, NamedTuple())
    @test m1.η ≈ m0.η && m1.Λ ≈ m0.Λ
    msgs = merge(inputs, NamedTuple{(target,)}((prior,)))
    F0, _ = Mycelium.local_free_energy(f, msgs, ps, NamedTuple())
    F1, _ = Mycelium.local_free_energy(e, merge(msgs, pts), pse, NamedTuple())
    @test all(F1[k] ≈ F0[k] for k in keys(F0))
end

@testset "L-SPEC-003 §1.1 clamped exposure, message by message" begin
    f = GaussianFactor(2 => 3; noise = [0.5, 0.2, 0.3])
    ps = (A = [0.7 -0.3; 0.2 0.5; 1.1 0.4], b = [0.1, -0.2, 0.3])
    gx, gy = Gaussian([0.3, -0.1], [1.0 0.2; 0.2 2.0]), Gaussian([1.0, -0.5, 2.0], Matrix(0.5I, 3, 3))
    for names in ((:A,), (:b,), (:A => :w, :b => :c))
        _check_clamped(f, names, :y, Polarity(x = Observed(), y = Unobserved()), (x = gx,), gy, ps)
        _check_clamped(f, names, :x, Polarity(x = Unobserved(), y = Observed()), (y = gy,), gx, ps)
    end
    c = LinearConstraintFactor((a = 1, b = 1), 1; noise = 0.1)
    psc = LuxCore.initialparameters(Xoshiro(3), c)
    _check_clamped(c, (:c => :level,), :a, Polarity(a = Unobserved(), b = Observed()),
        (b = Gaussian(0.5, 2.0),), Gaussian(-0.2, 1.5), psc)
end

@testset "L-SPEC-003 §1 interface errors" begin
    f = GaussianFactor(2 => 1)
    @test_throws ArgumentError expose(f, (:Q,))                 # no such block
    @test_throws ArgumentError expose(f, (:A => :x,))           # clashes with a channel
    @test_throws ArgumentError expose(f, (:A, :A))
    @test_throws ArgumentError expose(LinearConstraintFactor((a = 1, b = 1), 1), (:A,))  # not an array
    # a belief on both x and A is bilinear: refused, not approximated
    e = expose(f, (:A,))
    msgs = (x = Gaussian([0.0, 0.0], Matrix(1.0I, 2, 2)), y = DiracBelief([1.0]), A = Gaussian(zeros(2), Matrix(1.0I, 2, 2)))
    @test_throws ArgumentError Mycelium.factor_message(e, :A, Polarity(x = Observed(), y = Observed(), A = Unobserved()),
        msgs, nothing, (b = [0.0],), NamedTuple())
    @test_throws ArgumentError Mycelium.factor_message(e, :x, Polarity(x = Unobserved(), y = Observed(), A = Observed()),
        (y = DiracBelief([1.0]), A = msgs.A), nothing, (b = [0.0],), NamedTuple())
    @test !LenticulumCore.supports_polarity(e, Polarity(x = Latent(), y = Observed(), A = Unobserved()))
    @test LenticulumCore.supports_polarity(e, Polarity(x = Observed(), y = Observed(), A = Unobserved()))
    @test Polarity(x = Observed(), y = Observed(), A = Unobserved()) in LenticulumCore.supported_polarities(e)
end

# Bayesian linear regression: n exposed factors tie one weight (§1.3), a GaussianPrior on it.
function _regression_graph(X, y, m0, S0, σ², bias; xnew = nothing)
    n, d = size(X)
    b = GraphBuilder()
    variable!(b, :w, d)
    factor!(b, :prior, GaussianPrior(:w, m0, S0))
    connect!(b, :prior, :w, :w)
    ps = Pair{Symbol,Any}[:prior => NamedTuple()]
    rows = xnew === nothing ? (1:n) : (1:n+1)
    for i in rows
        xi = i <= n ? X[i, :] : xnew
        fx, fy, fo = Symbol(:x, i), Symbol(:y, i), Symbol(:obs, i)
        variable!(b, fx, d); variable!(b, fy, 1)
        factor!(b, fo, expose(GaussianFactor(d => 1; noise = σ²), (:A => :w,)))
        factor!(b, Symbol(:dx, i), DataFactor(fx, xi))
        connect!(b, Symbol(:dx, i), fx, fx; direction = Emitting())
        connect!(b, fo, :x, fx; direction = Absorbing())
        connect!(b, fo, :w, :w)
        if i <= n
            factor!(b, Symbol(:dy, i), DataFactor(fy, [y[i]]))
            connect!(b, Symbol(:dy, i), fy, fy; direction = Emitting())
            connect!(b, fo, :y, fy; direction = Absorbing())
            push!(ps, Symbol(:dy, i) => NamedTuple())
        else
            connect!(b, fo, :y, fy)    # the prediction; a leaf, so −log p(y) is unchanged
        end
        push!(ps, fo => (b = [bias],), Symbol(:dx, i) => NamedTuple())
    end
    g = validate(build(b))
    return g, NamedTuple(ps), LuxCore.initialstates(Xoshiro(0), g)
end

@testset "L-SPEC-003 §1.2–1.3 a tied exposed weight is Bayesian linear regression" begin
    rng = Xoshiro(2026)
    n, d = 7, 3
    X = randn(rng, n, d)
    m0, S0, σ², bias = [0.5, -1.0, 0.0], [1.0 0.3 0.0; 0.3 2.0 0.1; 0.0 0.1 0.5], 0.4, 0.25
    y = X * [1.0, 0.5, -2.0] .+ bias .+ sqrt(σ²) .* randn(rng, n)
    xnew = [0.3, -1.2, 0.8]
    g, ps, st = _regression_graph(X, y, m0, S0, σ², bias; xnew)
    @test istree(g)
    marg, rep, st, store = infer!(g, tree_schedule(g), ps, st)
    @test rep.converged
    # closed form: Λ = S0⁻¹ + XᵀX/σ², η = S0⁻¹m0 + Xᵀ(y − b)/σ²
    Λ = inv(S0) + X' * X / σ²
    μ = Λ \ (S0 \ m0 + X' * (y .- bias) / σ²)
    @test belief_mean(marg.w) ≈ μ
    @test belief_cov(marg.w) ≈ inv(Λ)
    # the posterior predictive at a new input
    ynew = only(marg[Symbol(:y, n + 1)] |> belief_mean)
    @test ynew ≈ dot(xnew, μ) + bias
    @test only(belief_cov(marg[Symbol(:y, n + 1)])) ≈ dot(xnew, Λ \ xnew) + σ²
    # Bethe free energy on the tree = −log p(y) (the prediction factor adds nothing)
    Sy = X * S0 * X' + σ² * I
    r = y .- X * m0 .- bias
    @test _F(store, g, ps, st) ≈ (n * log(2π) + logdet(Sy) + dot(r, Sy \ r)) / 2
end

@testset "L-SPEC-003 §1.2 a matrix weight: vec ordering against an explicit sum" begin
    # y = A x + b + ε with A (2×3) exposed, vec(A) ~ N(0, I). yⱼ = Σₖ A[j,k] xₖ + bⱼ, so the
    # coefficient of vec index (k-1)·2 + j is xₖ; build that design by loops.
    x0, b0, Q, yobs = [0.5, -1.0, 2.0], [0.1, -0.1], [0.3 0.05; 0.05 0.2], [1.0, -2.0]
    H = zeros(2, 6)
    for j in 1:2, k in 1:3
        H[j, (k-1)*2+j] = x0[k]
    end
    Λ = I + H' * (Q \ H)
    μ = Λ \ (H' * (Q \ (yobs - b0)))
    e = expose(GaussianFactor(3 => 2; noise = Q), (:A,))
    msg, _ = Mycelium.factor_message(e, :A, Polarity(x = Observed(), y = Observed(), A = Unobserved()),
        (x = DiracBelief(x0), y = DiracBelief(yobs)), nothing, (b = b0,), NamedTuple())
    post = combine(Gaussian(zeros(6), Matrix(1.0I, 6, 6)), msg)
    @test belief_mean(post) ≈ μ && belief_cov(post) ≈ inv(Λ)
    @test rank(msg.Λ) == 2                                       # a likelihood in 2 of 6 directions
    # a Gaussian, not a point, on y: R = Q + S_y
    Sy = [0.1 0.0; 0.0 0.4]
    msg2, _ = Mycelium.factor_message(e, :A, Polarity(x = Observed(), y = Observed(), A = Unobserved()),
        (x = DiracBelief(x0), y = Gaussian(yobs, Sy)), nothing, (b = b0,), NamedTuple())
    @test msg2.Λ ≈ H' * ((Q + Sy) \ H)
    # the bias exposed with A as a point: an ordinary Gaussian likelihood of b
    A0 = [1.0 0.0 0.5; -0.5 1.0 0.0]
    eb = expose(GaussianFactor(3 => 2; noise = Q), (:b,))
    mb, _ = Mycelium.factor_message(eb, :b, Polarity(x = Observed(), y = Observed(), b = Unobserved()),
        (x = DiracBelief(x0), y = DiracBelief(yobs)), nothing, (A = A0,), NamedTuple())
    @test belief_mean(mb) ≈ yobs - A0 * x0 && belief_cov(mb) ≈ Q
    # no information on the other exposed block: exactly uninformative
    eab = expose(GaussianFactor(3 => 2; noise = Q), (:A, :b))
    mt, _ = Mycelium.factor_message(eab, :A, Polarity(x = Observed(), y = Observed(), A = Unobserved(), b = Latent()),
        (x = DiracBelief(x0), y = DiracBelief(yobs)), nothing, NamedTuple(), NamedTuple())
    @test mt isa LenticulumCore.TrivialBelief
end

@testset "L-SPEC-003 a hierarchy over weights: random slopes with exposed weights" begin
    # The hierarchical-models example writes random slopes with the weights as input
    # channels. Exposure gives the same model the usual way round: one regression factor per
    # observation, its weight Aⱼ (1×2) exposed and tied within group j, βⱼ ~ N(β̄, Σβ).
    d = HierarchicalModels.slopes_data()
    ref = HierarchicalModels.slopes_dense(d)
    b = GraphBuilder()
    variable!(b, :betabar, 2)
    factor!(b, :prior, GaussianPrior(:betabar, zeros(2), Matrix(HierarchicalModels.V0 * I, 2, 2)))
    connect!(b, :prior, :betabar, :betabar)
    ps = Pair{Symbol,Any}[:prior => NamedTuple()]
    for j in eachindex(d.X)
        β, pop = Symbol(:beta, j), Symbol(:pop, j)
        variable!(b, β, 2)
        factor!(b, pop, GaussianFactor(2 => 2; noise = d.Σβ))
        connect!(b, pop, :x, :betabar); connect!(b, pop, :y, β)
        push!(ps, pop => (A = Matrix(1.0I, 2, 2), b = zeros(2)))
        for i in axes(d.X[j], 1)
            o, xv, yv = Symbol(:obs, j, :_, i), Symbol(:x, j, :_, i), Symbol(:y, j, :_, i)
            variable!(b, xv, 2); variable!(b, yv, 1)
            factor!(b, o, expose(GaussianFactor(2 => 1; noise = d.σ^2), (:A => :w,)))
            factor!(b, Symbol(:d, xv), DataFactor(xv, d.X[j][i, :]))
            factor!(b, Symbol(:d, yv), DataFactor(yv, [d.y[j][i]]))
            connect!(b, Symbol(:d, xv), xv, xv; direction = Emitting())
            connect!(b, Symbol(:d, yv), yv, yv; direction = Emitting())
            connect!(b, o, :x, xv; direction = Absorbing())
            connect!(b, o, :y, yv; direction = Absorbing())
            connect!(b, o, :w, β)
            push!(ps, o => (b = [0.0],), Symbol(:d, xv) => NamedTuple(), Symbol(:d, yv) => NamedTuple())
        end
    end
    g = validate(build(b))
    ps = NamedTuple(ps); st = LuxCore.initialstates(Xoshiro(0), g)
    @test istree(g)
    marg, rep, st, store = infer!(g, tree_schedule(g), ps, st)
    names = [:betabar; [Symbol(:beta, j) for j in eachindex(d.X)]]
    @test reduce(vcat, [belief_mean(marg[k]) for k in names]) ≈ ref.mean rtol = 1e-9
    @test all(belief_cov(marg[k]) ≈ C for (k, C) in zip(names, ref.cov))
    @test _F(store, g, ps, st) ≈ ref.F rtol = 1e-10
end

# Codex's review of the exposure code (2026-10-30).
struct _FrozenPair <: LenticulumCore.AbstractLenticulumFactor end
LuxCore.initialparameters(::AbstractRNG, ::_FrozenPair) = (a = [1.0], b = [2.0])
LenticulumCore.channels(::_FrozenPair) = (LenticulumCore.Channel(:z, 1),)
LenticulumCore.isfrozen(::_FrozenPair) = true

@testset "L-SPEC-003 review: names, traits, enumeration, improper weights" begin
    # a channel named like a parameter block: the clamped path still works (separate
    # containers), the belief path is refused (one namespace)
    es = expose(GaussianFactor(1 => 1; noise = 0.5, channels = (:x, :b)), (:A => :weight,))
    ms, _ = Mycelium.factor_message(es, :b, Polarity(x = Observed(), b = Unobserved(), weight = Observed()),
        (x = DiracBelief([1.0]), weight = DiracBelief([2.0])), nothing, (b = [3.0],), NamedTuple())
    @test belief_mean(ms) ≈ [5.0] && belief_cov(ms) ≈ fill(0.5, 1, 1)
    @test_throws ArgumentError Mycelium.factor_message(es, :b, Polarity(x = Observed(), b = Unobserved(), weight = Observed()),
        (x = DiracBelief([1.0]), weight = Gaussian(2.0, 1.0)), nothing, (b = [3.0],), NamedTuple())
    @test !LenticulumCore.supports_polarity(es, Polarity(x = Observed(), b = Observed(), weight = Unobserved()))
    # an unexposed parameter named like a channel is no ambiguity: channels (:x, :noise)
    en = expose(GaussianFactor(1 => 1; noise = 0.5, learn_noise = :diagonal, channels = (:x, :noise)), (:A,))
    psn = LuxCore.initialparameters(Xoshiro(0), en)
    mn, _ = Mycelium.factor_message(en, :noise, Polarity(x = Observed(), noise = Unobserved(), A = Observed()),
        (x = DiracBelief([1.0]), A = Gaussian(2.0, 1.0)), nothing, merge(psn, (b = [3.0],)), NamedTuple())
    @test belief_mean(mn) ≈ [5.0] && belief_cov(mn) ≈ fill(1.5, 1, 1)
    # traits and counts of the remaining parameters
    e = expose(_FrozenPair(), (:a,))
    @test LenticulumCore.isfrozen(e)
    @test LuxCore.parameterlength(expose(GaussianFactor(1 => 1), (:A,))) == 1
    # every accepted polarity is listed, latent ones included
    eab = expose(GaussianFactor(2 => 1), (:A, :b))
    listed = LenticulumCore.supported_polarities(eab)
    O, U, L = Observed(), Unobserved(), Latent()
    all_ = [Polarity(NamedTuple{(:x, :y, :A, :b)}(a)) for a in Iterators.product(ntuple(_ -> (O, U, L), 4)...)]
    @test Set(listed) == Set(filter(p -> LenticulumCore.supports_polarity(eab, p), all_))
    @test Polarity(x = O, y = O, A = U, b = L) in listed
    # above the cap the list throws, as L-SPEC-001 permits; the predicates answer lazily
    big = expose(LinearConstraintFactor(NamedTuple{Tuple(Symbol(:v, i) for i in 1:5)}(ntuple(_ -> 1, 5)), 1; noise = 0.1), (:c,))
    @test_throws ArgumentError LenticulumCore.supported_polarities(big)
    @test !LenticulumCore.hasnopolarity(big) && !LenticulumCore.isunidirectional(big)
    # a weight belief improper only in directions the factor does not see: x = [1, 0] reads
    # only A[1], whose information is N(2, 1); the predictive is N(2, 1 + 0.5)
    ea = expose(GaussianFactor(2 => 1; noise = 0.5), (:A,))
    m, _ = Mycelium.factor_message(ea, :y, Polarity(x = O, y = U, A = O),
        (x = DiracBelief([1.0, 0.0]), A = GaussianBelief([2.0, 0.0], [1.0 0.0; 0.0 0.0])), nothing, (b = [0.0],), NamedTuple())
    @test belief_mean(m) ≈ [2.0] && belief_cov(m) ≈ fill(1.5, 1, 1)
    # Float32 rank-one information along (1, 1), read by x = [1, 1]: z₁ + z₂ ~ N(2, 1)
    m32, _ = Mycelium.factor_message(ea, :y, Polarity(x = O, y = U, A = O),
        (x = DiracBelief([1.0, 1.0]), A = GaussianBelief(Float32[2, 2], Float32[1 1; 1 1])), nothing, (b = [0.0],), NamedTuple())
    @test belief_mean(m32) ≈ [2.0] && belief_cov(m32) ≈ fill(1.5, 1, 1)
    # a flat direction read with weight 1e-11 is still read: refused
    @test_throws ArgumentError Mycelium.factor_message(ea, :y, Polarity(x = O, y = U, A = O),
        (x = DiracBelief([1.0, 1e-11]), A = GaussianBelief([2.0, 0.0], [1.0 0.0; 0.0 0.0])), nothing, (b = [0.0],), NamedTuple())
    # tiny but nonzero reads of a flat direction are reads: refused (no absolute floor)
    @test_throws ArgumentError Mycelium.factor_message(ea, :y, Polarity(x = O, y = U, A = O),
        (x = DiracBelief([0.0, 1e-16]), A = GaussianBelief([2.0, 0.0], [1.0 0.0; 0.0 0.0])), nothing, (b = [0.0],), NamedTuple())
    # Λ = 0 with η ≠ 0 is a linear site exp(a), not no information: refused, not dropped
    e11 = expose(GaussianFactor(1 => 1; noise = 0.5), (:A,))
    @test_throws ArgumentError Mycelium.factor_message(e11, :y, Polarity(x = O, y = U, A = O),
        (x = DiracBelief([1.0]), A = GaussianBelief([1.0], zeros(1, 1))), nothing, (b = [0.0],), NamedTuple())
    # ... but improper where it reads: refused
    @test_throws ArgumentError Mycelium.factor_message(ea, :y, Polarity(x = O, y = U, A = O),
        (x = DiracBelief([0.0, 1.0]), A = GaussianBelief([2.0, 0.0], [1.0 0.0; 0.0 0.0])), nothing, (b = [0.0],), NamedTuple())
    # x₀ = 0: the weight is unseen and the message to it has rank 0
    e1 = expose(GaussianFactor(1 => 2; noise = 0.5), (:A,))
    m0, _ = Mycelium.factor_message(e1, :A, Polarity(x = O, y = O, A = U),
        (x = DiracBelief([0.0]), y = DiracBelief([1.0, 2.0])), nothing, (b = [0.0, 0.0],), NamedTuple())
    @test rank(m0.Λ) == 0
end

@testset "L-SPEC-003 fifth review: the flatness test is unit-free" begin
    # x = [0, 1, 0] reads only A[2] ~ N(2, 1); a precision of 1e16 on the unread first
    # coordinate must not make A[2] look flat. Predictive N(2, 1 + 0.5).
    e = expose(GaussianFactor(3 => 1; noise = 0.5), (:A,))
    m, _ = Mycelium.factor_message(e, :y, Polarity(x = Observed(), y = Unobserved(), A = Observed()),
        (x = DiracBelief([0.0, 1.0, 0.0]), A = GaussianBelief([0.0, 2.0, 0.0], Diagonal([1e16, 1.0, 0.0]))),
        nothing, (b = [0.0],), NamedTuple())
    @test belief_mean(m) ≈ [2.0] && belief_cov(m) ≈ fill(1.5, 1, 1)
end
