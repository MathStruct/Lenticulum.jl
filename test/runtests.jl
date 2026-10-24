using Lenticulum
using LenticulumCore
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent
using Mycelium
using LinearAlgebra
using LuxCore
using Random
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

# ---------------------------------------------------------------------------
# The linear-Gaussian chain — the example every factor-graph library opens with.
#
# GTSAM's `OdometryExample`, with one GPS-style measurement bolted onto the last pose:
#
#     π ── x₁ ──[odo₁]── x₂ ──[odo₂]── x₃ ──[gps]── z ──[data]
#
#     x₁ ~ N(μ₀, Σ₀),   x_{k+1} = x_k + u_k + ε_k,  ε_k ~ N(0, Q_k),
#     z  = x₃ + δ,      δ ~ N(0, R),                z clamped to z₀.
#
# Every "between" factor is the SAME `GaussianFactor(1 => 1)` with A = I and b = u_k; the
# unary measurement factor is that factor again, plus a `DataFactor` clamp on an auxiliary
# variable (`Everything is a Factor.md`: data is evidence, and evidence is a factor).
#
# The oracle is what GTSAM actually computes: stack the variables, sum the factors'
# contributions into one information matrix, and solve. Belief propagation on a tree must
# reproduce its marginals exactly — not approximately.
#
# See `The Linear Gaussian Chain.md`.
# ---------------------------------------------------------------------------

"""
Assemble the joint information form of a linear-Gaussian factor graph and solve it.

Each factor is given as `(G, Q, c)`, contributing energy ½‖Gξ − c‖²_{Q⁻¹} on the stacked
variable ξ; so Λ = Σ GᵀQ⁻¹G and η = Σ GᵀQ⁻¹c. This is the sparse Hessian GTSAM builds by
linearisation, dense and unapologetic.
"""
function joint_solve(blocks)
    n = size(first(blocks)[1], 2)
    Λ, η = zeros(n, n), zeros(n)
    for (G, Qm, c) in blocks
        Λ .+= G' * (Qm \ G)
        η .+= G' * (Qm \ c)
    end
    return Λ \ η, inv(Λ), Λ
end

# selector: the 1×n row picking out variable `i` of `n`
_sel(i, n) = (r = zeros(1, n); r[1, i] = 1.0; r)

const CHμ₀ = [0.0]
const CHΣ₀ = fill(1.0, 1, 1)
const U    = ([2.0], [2.0])                        # odometry measurements
const QK   = (fill(0.25, 1, 1), fill(0.25, 1, 1))  # odometry noise
const R    = fill(0.5, 1, 1)                       # GPS noise
const z₀   = [4.5]                                 # the GPS reading
const Id1  = fill(1.0, 1, 1)

"""
An `n`-pose odometry chain with a GPS measurement on the last pose. Returns `(g, ps, st)`.
Poses are `:x1 … :xn`; the auxiliary measurement variable is `:z`.
"""
function chain_graph(n::Int, u, Qs, Rm, z)
    b = GraphBuilder()
    for k in 1:n
        variable!(b, Symbol(:x, k), 1)
    end
    variable!(b, :z, 1)

    factor!(b, :prior, GaussianPrior(Symbol(:x, 1), CHμ₀, CHΣ₀))
    connect!(b, :prior, Symbol(:x, 1), Symbol(:x, 1); direction = Bidirectional())

    for k in 1:(n - 1)
        name = Symbol(:odo, k)
        factor!(b, name, GaussianFactor(1 => 1; noise = Qs[k], channels = (:x, :y)))
        # :x is the earlier pose, :y the later one — the factor is run forwards along the
        # chain when the schedule sweeps up and backwards when it sweeps down, with no
        # change to the factor. That is the whole point of a bidirectional factor.
        connect!(b, name, :x, Symbol(:x, k); direction = Bidirectional())
        connect!(b, name, :y, Symbol(:x, k + 1); direction = Bidirectional())
    end

    factor!(b, :gps, GaussianFactor(1 => 1; noise = Rm, channels = (:x, :y)))
    connect!(b, :gps, :x, Symbol(:x, n); direction = Bidirectional())
    connect!(b, :gps, :y, :z; direction = Bidirectional())
    factor!(b, :data, DataFactor(:z, z))
    connect!(b, :data, :z, :z; direction = Emitting())

    g = validate(build(b))
    odo = NamedTuple(Symbol(:odo, k) => (A = Id1, b = u[k]) for k in 1:(n - 1))
    ps = merge((prior = NamedTuple(),), odo,
               (gps = (A = Id1, b = [0.0]), data = NamedTuple()))
    st = merge((prior = NamedTuple(),),
               NamedTuple(Symbol(:odo, k) => NamedTuple() for k in 1:(n - 1)),
               (gps = NamedTuple(), data = NamedTuple()))
    return g, ps, st
end

# The same chain, as blocks for `joint_solve`. Variable order is (x₁ … xₙ); `z` is clamped
# and therefore eliminated by hand, exactly as GTSAM eliminates a measurement.
function chain_blocks(n::Int, u, Qs, Rm, z)
    blocks = Any[(_sel(1, n), CHΣ₀, CHμ₀)]
    for k in 1:(n - 1)
        push!(blocks, (_sel(k + 1, n) .- _sel(k, n), Qs[k], u[k]))
    end
    push!(blocks, (_sel(n, n), Rm, z))
    return blocks
end

# −log p(z₀): the chain's pushforward prior on xₙ is N(μ₀ + Σuₖ, Σ₀ + ΣQₖ), and z adds R.
function chain_log_evidence(n, u, Qs, Rm, z)
    m = CHμ₀ .+ sum(u[1:(n - 1)])
    S = CHΣ₀ .+ sum(Qs[1:(n - 1)]) .+ Rm
    d = z .- m
    return (log(2π) + logdet(S) + d' * (S \ d))[1] / 2
end

# ---------------------------------------------------------------------------
# Data reconciliation: a conservation law with noisy measurements on each branch.
#
#     i₁ + i₂ + i₃ = 0    (Kirchhoff — ModelingToolkit's connector equation)
#     ĩ_k ~ N(i_k, s_k)   (a flow meter on each branch)
#
# Three measurements that do NOT satisfy the law, and one equation that says they must.
# This is a real industrial problem, and it is the smallest graph in which a factor is used
# in three different directions within a single sweep.
# ---------------------------------------------------------------------------
const KCLq  = fill(1e-4, 1, 1)                                  # how hard the law is
const KCLm  = ([1.0], [2.0], [-2.7])                            # the readings; they don't balance
const KCLs  = (fill(0.10, 1, 1), fill(0.20, 1, 1), fill(0.05, 1, 1))
const Ones1 = ones(1, 1)

# ---------------------------------------------------------------------------
# A resistive divider — the smallest acausal model that is NOT a tree.
#
#        Vs ──[R₁]── v ──[R₂]── gnd            v + R₁i₁ = Vs,   v − R₂i₂ = 0,   i₁ − i₂ = 0
#
# Three equations, three unknowns, and the factor graph has a cycle
# (v — ohm₁ — i₁ — kcl — i₂ — ohm₂ — v). Circuits are loopy; that is the whole difficulty
# of acausal modelling, and it is why ModelingToolkit solves rather than propagates.
#
# See `ModelingToolkit as an Acausal Relation.md` §6.
# ---------------------------------------------------------------------------
const R₁, R₂, Vs = 2.0, 3.0, 10.0
const CircQ = fill(1e-3, 1, 1)          # each law holds only to within this variance

"""
The divider as a factor graph. `pv` is the variance of the weak regularising prior every
variable carries; without one the loop cannot start (see `constraint.md` §4).
"""
function divider_graph(pv::Float64)
    wp = fill(pv, 1, 1)
    b = GraphBuilder()
    for v in (:v, :i1, :i2)
        variable!(b, v, 1)
    end
    factor!(b, :ohm1, LinearConstraintFactor((v = 1, i1 = 1), 1; noise = CircQ))
    factor!(b, :ohm2, LinearConstraintFactor((v = 1, i2 = 1), 1; noise = CircQ))
    factor!(b, :kcl,  LinearConstraintFactor((i1 = 1, i2 = 1), 1; noise = CircQ))
    connect!(b, :ohm1, :v, :v;   direction = Bidirectional())
    connect!(b, :ohm1, :i1, :i1; direction = Bidirectional())
    connect!(b, :ohm2, :v, :v;   direction = Bidirectional())
    connect!(b, :ohm2, :i2, :i2; direction = Bidirectional())
    connect!(b, :kcl,  :i1, :i1; direction = Bidirectional())
    connect!(b, :kcl,  :i2, :i2; direction = Bidirectional())
    for v in (:v, :i1, :i2)
        factor!(b, Symbol(:reg, v), GaussianPrior(v, [0.0], wp))
        connect!(b, Symbol(:reg, v), v, v; direction = Bidirectional())
    end
    g = validate(build(b))
    ps = merge((ohm1 = (A = (v = Ones1, i1 = fill(R₁, 1, 1)), c = [Vs]),
                ohm2 = (A = (v = Ones1, i2 = fill(-R₂, 1, 1)), c = [0.0]),
                kcl  = (A = (i1 = Ones1, i2 = -Ones1), c = [0.0])),
               NamedTuple(Symbol(:reg, v) => NamedTuple() for v in (:v, :i1, :i2)))
    st = merge((ohm1 = NamedTuple(), ohm2 = NamedTuple(), kcl = NamedTuple()),
               NamedTuple(Symbol(:reg, v) => NamedTuple() for v in (:v, :i1, :i2)))
    return g, ps, st
end

function divider_blocks(pv::Float64)
    wp = fill(pv, 1, 1)
    blocks = Any[([1.0 R₁ 0.0], CircQ, [Vs]),
                 ([1.0 0.0 -R₂], CircQ, [0.0]),
                 ([0.0 1.0 -1.0], CircQ, [0.0])]
    for k in 1:3
        push!(blocks, (_sel(k, 3), wp, [0.0]))
    end
    return blocks
end

# a likelihood with only `loglikelihood`: the derivatives fall back to finite differences and
# the EP moments to quadrature, so it checks both against the closed forms
struct FDOrdinal <: AbstractLikelihood
    inner::OrdinalProbitLikelihood
end
Lenticulum.npredictors(::FDOrdinal) = 1
Lenticulum.obsdim(::FDOrdinal) = 1
Lenticulum.loglikelihood(L::FDOrdinal, y, z) = loglikelihood(L.inner, y, z)
Lenticulum.outcomes(L::FDOrdinal) = outcomes(L.inner)

# moments and log normaliser of N(z; μ, s²) e^{ℓ(z)} on a fine grid: the reference
function brute_tilted(ℓ, μ, s²; lo = μ - 12sqrt(s²), hi = μ + 12sqrt(s²), n = 400001)
    zs = range(lo, hi; length = n)
    lw = [-(z - μ)^2 / (2s²) - log(2π * s²) / 2 + ℓ(z) for z in zs]
    mx = maximum(lw)
    w = exp.(lw .- mx)
    Z = sum(w)
    m = sum(w .* zs) / Z
    return m, sum(w .* (zs .- m) .^ 2) / Z, mx + log(Z * step(zs)), zs[argmax(lw)]
end


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
    hard = DiracBelief([7.0])
    @test Mycelium.combine(hard, a) === hard
    @test Mycelium.combine(a, hard) === hard

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

@testset "pointwise energy and free energy: one contract (issue #8)" begin
    # Two levels: `energy` is the pointwise energy (for a residual factor the signed residual,
    # scalarised by a norm), the free energy is graded (fit, complexity, negentropy); at a point
    # the scalar pointwise energy is the `fit` summand.
    NT = NamedTuple()
    at(f, vals, ps) = first(Mycelium.local_free_energy(f, map(DiracBelief, vals), ps, NT))[:fit]

    gf = GaussianFactor(1 => 1; noise = 0.1)
    gps = (A = ones(1, 1), b = zeros(1))
    e, _ = scalar_energy(gf, [1.0], nothing, [2.0], gps, NT)
    @test e ≈ 0.5 * (2.0 - 1.0)^2 / 0.1
    @test e ≈ at(gf, (x = [1.0], y = [2.0]), gps)
    @test LenticulumCore.pointwise_scalarisation(gf) isa SquaredNorm
    @test LenticulumCore.scalarisation(gf) isa GradedScalarisation      # the free energy's, unchanged

    # the scalarisation is a parameter with a default, like the p of a norm
    @test first(scalar_energy(gf, [1.0], nothing, [2.0], gps, NT; σ = SquaredNorm())) ≈ 0.5
    # per polarity: every polarity of this factor has the same residual, so the same answer
    for p in LenticulumCore.supported_polarities(gf)
        @test first(scalar_energy(gf, [1.0], nothing, [2.0], gps, NT; polarity = p)) ≈ e
    end

    cf = LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.5)
    cps = (A = (x = fill(2.0, 1, 1), y = fill(-1.0, 1, 1)), c = [0.5])
    ec, _ = scalar_energy(cf, (x = [1.0],), nothing, (y = [3.0],), cps, NT)
    @test ec ≈ 0.5 * (2.0 - 3.0 - 0.5)^2 / 0.5
    @test ec ≈ at(cf, (x = [1.0], y = [3.0]), cps)

    nf = NonlinearFactor(x -> x .^ 2, 1 => 1; noise = 0.25)
    en, _ = scalar_energy(nf, [2.0], nothing, [3.0], NT, NT)
    @test en ≈ 0.5 * (3.0 - 4.0)^2 / 0.25
    @test en ≈ at(nf, (x = [2.0], y = [3.0]), NT)

    lf = LikelihoodFactor(PoissonLikelihood(), 1)
    el, _ = scalar_energy(lf, [log(2.0)], nothing, [3.0], NT, NT)
    @test el ≈ -log(exp(-2.0) * 8 / 6)
    @test el ≈ at(lf, (x = [log(2.0)], y = [3.0]), NT)

    # the energy lands in the space the factor declares for it
    for (f, x, y, ps) in ((gf, [1.0], [2.0], gps), (cf, (x = [1.0],), (y = [3.0],), cps),
                          (nf, [2.0], [3.0], NT), (lf, [log(2.0)], [3.0], NT))
        r, _ = LenticulumCore.energy(f, x, nothing, y, ps, NT)
        @test length(r) == LenticulumCore.dimension(LenticulumCore.pointwise_energyspace(f))
    end
end

@testset "the linear-Gaussian chain (GTSAM's OdometryExample)" begin
    g, ps, st = chain_graph(3, U, QK, R, z₀)

    # A chain with unary factors hanging off it is a tree, so BP is exact and two sweeps
    # suffice — no damping, no iteration count to tune.
    @test istree(g)
    @test euler_characteristic(g) == 1
    @test nvariables(g) == 4 && nfactors(g) == 5

    marg, report, st1, store = infer!(g, tree_schedule(g), ps, st)
    @test report.converged

    mjoint, Σjoint, _ = joint_solve(chain_blocks(3, U, QK, R, z₀))

    # THE assertion: every pose marginal matches the joint solve, mean and variance.
    for (i, v) in enumerate((:x1, :x2, :x3))
        @test marg[v] isa GaussianBelief
        @test belief_mean(marg[v])[1] ≈ mjoint[i]
        @test belief_cov(marg[v])[1] ≈ Σjoint[i, i]
    end
    @test marg.z === DiracBelief(z₀)

    # These are SMOOTHED marginals, not filtered ones: x₁'s variance is below its prior
    # because the GPS reading travelled back down the chain. A forward-only sweep could not
    # produce this, and it is what makes the two-pass tree schedule worth having.
    @test belief_cov(marg.x1)[1] < CHΣ₀[1]
    @test belief_cov(marg.x1)[1] ≈ 0.5
    @test belief_mean(marg.x1)[1] ≈ 0.25
    @test belief_mean(marg.x3)[1] ≈ 4.375

    # ...and the marginal likelihood of the reading falls out of the Bethe free energy,
    # which is Remark 24 again, now on a graph the paper never mentions.
    total, st1 = LenticulumCore.scalar_free_energy(store, g, ps, st1)
    @test total ≈ chain_log_evidence(3, U, QK, R, z₀)

    # every non-clamped variable in the chain has degree 2, so each contributes one −H
    F, st1 = factor_free_energies(store, g, ps, st1)
    @test sort(collect(keys(F))) == [:data, :gps, :odo1, :odo2, :prior]
    @test F[:data] == 0.0
    for v in (:x1, :x2, :x3, :z)
        @test counting_numbers(g)[v] == -1
    end
    @test total_counting_number(g) == 1
end

@testset "the chain is exact at every length, not just three" begin
    # n = 3 could be a coincidence; n = 2 … 8 is not. The oracle is rebuilt at each length,
    # so this compares two independent computations rather than a stored answer.
    for n in 2:8
        u = ntuple(k -> [2.0], n)
        Q = ntuple(k -> fill(0.1 + 0.05k, 1, 1), n)   # deliberately inhomogeneous
        g, ps, st = chain_graph(n, u, Q, R, z₀)
        @test istree(g)
        marg, report, st1, store = infer!(g, tree_schedule(g), ps, st)
        @test report.converged

        mjoint, Σjoint, _ = joint_solve(chain_blocks(n, u, Q, R, z₀))
        for k in 1:n
            v = Symbol(:x, k)
            @test belief_mean(marg[v])[1] ≈ mjoint[k]
            @test belief_cov(marg[v])[1] ≈ Σjoint[k, k]
        end

        total, _ = LenticulumCore.scalar_free_energy(store, g, ps, st1)
        @test total ≈ chain_log_evidence(n, u, Q, R, z₀)
    end
end

@testset "unclamping the measurement turns estimation into prediction" begin
    # `The Linear Gaussian Chain.md` §5: drop the DataFactor and the SAME graph predicts
    # instead of estimating. No new code path — Polarity Resolution works out the direction
    # each factor must run in from which messages have arrived.
    b = GraphBuilder()
    for v in (:x1, :x2, :x3, :z)
        variable!(b, v, 1)
    end
    factor!(b, :prior, GaussianPrior(:x1, CHμ₀, CHΣ₀))
    factor!(b, :odo1, GaussianFactor(1 => 1; noise = QK[1], channels = (:x, :y)))
    factor!(b, :odo2, GaussianFactor(1 => 1; noise = QK[2], channels = (:x, :y)))
    factor!(b, :gps,  GaussianFactor(1 => 1; noise = R,     channels = (:x, :y)))
    connect!(b, :prior, :x1, :x1; direction = Bidirectional())
    connect!(b, :odo1, :x, :x1; direction = Bidirectional())
    connect!(b, :odo1, :y, :x2; direction = Bidirectional())
    connect!(b, :odo2, :x, :x2; direction = Bidirectional())
    connect!(b, :odo2, :y, :x3; direction = Bidirectional())
    connect!(b, :gps,  :x, :x3; direction = Bidirectional())
    connect!(b, :gps,  :y, :z;  direction = Bidirectional())
    g = validate(build(b))
    ps = (prior = NamedTuple(), odo1 = (A = Id1, b = U[1]), odo2 = (A = Id1, b = U[2]),
          gps = (A = Id1, b = [0.0]))
    st = (prior = NamedTuple(), odo1 = NamedTuple(), odo2 = NamedTuple(), gps = NamedTuple())
    marg, report, _, _ = infer!(g, tree_schedule(g), ps, st)
    @test report.converged

    # z is now the PREDICTIVE distribution: the prior pushed along the whole chain, with
    # noise accumulated at every step.
    @test belief_mean(marg.z) ≈ CHμ₀ .+ U[1] .+ U[2]
    @test belief_cov(marg.z) ≈ CHΣ₀ .+ QK[1] .+ QK[2] .+ R
    @test belief_cov(marg.x3) ≈ CHΣ₀ .+ QK[1] .+ QK[2]

    # ...and with no evidence anywhere downstream, x₁ falls back to exactly its prior —
    # nothing flows backwards, because there is nothing to flow.
    @test belief_mean(marg.x1) ≈ CHμ₀
    @test belief_cov(marg.x1) ≈ CHΣ₀
end

@testset "LinearConstraintFactor: the acausal factor interface" begin
    f = LinearConstraintFactor((a = 2, b = 3, c = 1), 2; noise = 0.5)
    ps, st = LuxCore.setup(Random.default_rng(), f)
    @test size(ps.A.a) == (2, 2) && size(ps.A.b) == (2, 3) && size(ps.A.c) == (2, 1)
    @test length(ps.c) == 2
    @test LuxCore.parameterlength(f) == 2 * (2 + 3 + 1) + 2
    @test LenticulumCore.islearnable(f)
    @test map(LenticulumCore.channelname, LenticulumCore.channels(f)) == (:a, :b, :c)

    # n channels ⇒ n polarities, and NONE of them is the factor's "real" direction.
    # GaussianFactor has 2 no matter what; this is where "acausal" becomes a number.
    pols = LenticulumCore.supported_polarities(f)
    @test length(pols) == 3
    @test !LenticulumCore.isunidirectional(f)
    for p in pols
        @test LenticulumCore.supports_polarity(f, p)
        @test length(LenticulumCore.unobserved_channels(p)) == 1
    end
    @test Set(only(LenticulumCore.unobserved_channels(p)) for p in pols) == Set((:a, :b, :c))

    # a Latent channel is a legal modelling choice (marginalise it out)...
    @test LenticulumCore.supports_polarity(f,
        LenticulumCore.Polarity(; a = Unobserved(), b = Observed(), c = Latent()))
    # ...but two unobserved channels is a JOINT message, which v0 does not do
    @test !LenticulumCore.supports_polarity(f,
        LenticulumCore.Polarity(; a = Unobserved(), b = Unobserved(), c = Observed()))
    # and a polarity over the wrong channel set is not this factor's
    @test !LenticulumCore.supports_polarity(f,
        LenticulumCore.Polarity(; a = Unobserved(), b = Observed()))

    # the residual takes ALL channels at once — there is no (x, a, y) split to give it
    r = residual(f, (a = [1.0, 0.0], b = [0.0, 0.0, 0.0], c = [1.0]), ps)
    @test r ≈ ps.A.a * [1.0, 0.0] .+ ps.A.c * [1.0] .- ps.c
    @test length(r) == 2
    @test_throws ArgumentError residual(f, (a = [1.0, 0.0], b = zeros(3)), ps)
end

@testset "the acausal factor subsumes GaussianFactor exactly" begin
    # GaussianFactor(A, b) is  y = Ax + b + ε,  i.e.  0 = (−A)x + Iy − b + ε.
    # Same model, same arithmetic — the only difference is that one of them calls `x` an
    # input. If these two ever disagree, one of them is wrong.
    gf = GaussianFactor(1 => 1; noise = Q, channels = (:x, :y))
    cf = LinearConstraintFactor((x = 1, y = 1), 1; noise = Q)
    gps, cps = (A = A, b = bb), (A = (x = -A, y = fill(1.0, 1, 1)), c = bb)
    NT = NamedTuple()
    fwd = LenticulumCore.Polarity(; x = Observed(), y = Unobserved())
    bwd = LenticulumCore.Polarity(; x = Unobserved(), y = Observed())

    for (target, pol, inputs) in (
        (:y, fwd, (; x = Gaussian(μ₀, Σ₀))),        # forward: a pushforward
        (:x, bwd, (; y = DiracBelief(y₀))),          # backward: a likelihood
        (:x, bwd, (; y = Gaussian(y₀, Q))),          # backward from a soft observation
    )
        μg, _ = Mycelium.factor_message(gf, target, pol, inputs, TrivialBelief(), gps, NT)
        μc, _ = Mycelium.factor_message(cf, target, pol, inputs, TrivialBelief(), cps, NT)
        @test μc.η ≈ μg.η
        @test μc.Λ ≈ μg.Λ
    end

    # the pointwise energies agree
    rg, _ = LenticulumCore.energy(gf, [2.0], nothing, [7.0], gps, NT)
    @test residual(cf, (x = [2.0], y = [7.0]), cps) ≈ rg

    # ...and so does the whole graded Bethe contribution, summand by summand
    msgs = (x = Gaussian(μ₀, Σ₀), y = DiracBelief(y₀))
    Fg, _ = Mycelium.local_free_energy(gf, msgs, gps, NT)
    Fc, _ = Mycelium.local_free_energy(cf, msgs, cps, NT)
    @test keys(Fc) == keys(Fg)
    for k in keys(Fg)
        @test Fc[k] ≈ Fg[k]
    end

    # an uninformative input makes the equation vacuous, in both spellings
    μ0, _ = Mycelium.factor_message(cf, :y, fwd, (; x = TrivialBelief()), TrivialBelief(), cps, NT)
    @test μ0 isa TrivialBelief
end

@testset "acausal: a conservation law reconciles inconsistent measurements" begin
    b = GraphBuilder()
    for v in (:i1, :i2, :i3)
        variable!(b, v, 1)
    end
    factor!(b, :kcl, LinearConstraintFactor((i1 = 1, i2 = 1, i3 = 1), 1; noise = KCLq))
    for (k, v) in enumerate((:i1, :i2, :i3))
        connect!(b, :kcl, v, v; direction = Bidirectional())
        factor!(b, Symbol(:meas, k), GaussianPrior(v, KCLm[k], KCLs[k]))
        connect!(b, Symbol(:meas, k), v, v; direction = Bidirectional())
    end
    g = validate(build(b))
    ps = (kcl = (A = (i1 = Ones1, i2 = Ones1, i3 = Ones1), c = [0.0]),
          meas1 = NamedTuple(), meas2 = NamedTuple(), meas3 = NamedTuple())
    st = (kcl = NamedTuple(), meas1 = NamedTuple(), meas2 = NamedTuple(), meas3 = NamedTuple())

    # A star is a tree, so this is exact — a 3-ary factor does not spoil that.
    @test istree(g)
    @test LenticulumCore.isunidirectional(factornode(g, :kcl).factor) == false
    marg, report, st1, store = infer!(g, tree_schedule(g), ps, st)
    @test report.converged

    blocks = Any[(ones(1, 3), KCLq, [0.0])]
    for k in 1:3
        push!(blocks, (_sel(k, 3), KCLs[k], KCLm[k]))
    end
    mjoint, Σjoint, _ = joint_solve(blocks)
    for (k, v) in enumerate((:i1, :i2, :i3))
        @test belief_mean(marg[v])[1] ≈ mjoint[k]
        @test belief_cov(marg[v])[1] ≈ Σjoint[k, k]
    end

    # the reconciled currents balance to within the law's own tolerance, though the raw
    # readings miss by 0.3 — the whole point of the exercise
    @test abs(sum(KCLm[k][1] for k in 1:3)) ≈ 0.3
    @test abs(sum(belief_mean(marg[v])[1] for v in (:i1, :i2, :i3))) < 1e-3

    # each branch is pulled toward balance in proportion to how badly it is measured:
    # i₃ has the tightest meter (s = 0.05) and moves least, i₂ the loosest and moves most
    moved = [abs(belief_mean(marg[v])[1] - KCLm[k][1]) for (k, v) in enumerate((:i1, :i2, :i3))]
    @test moved[3] < moved[1] < moved[2]

    # every posterior variance is below its meter's, because the law is evidence too
    for (k, v) in enumerate((:i1, :i2, :i3))
        @test belief_cov(marg[v])[1] < KCLs[k][1]
    end
end

@testset "acausal and loopy: exact means, wrong variances" begin
    # Weiss & Freeman (2001): when Gaussian loopy BP converges, the MEANS are exact and the
    # variances are not. Both halves are asserted here, because the second half is the one
    # that gets forgotten.
    g, ps, st = divider_graph(1e4)
    @test !istree(g)
    @test euler_characteristic(g) == 0          # one independent cycle
    @test_throws Exception tree_schedule(g)     # the exact schedule is simply unavailable

    marg, report, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps = 2000, tol = 1e-13)
    @test report.converged

    mjoint, Σjoint, _ = joint_solve(divider_blocks(1e4))
    bp_mean = [belief_mean(marg[v])[1] for v in (:v, :i1, :i2)]
    bp_var  = [belief_cov(marg[v])[1] for v in (:v, :i1, :i2)]

    # 1. the means are exact, to machine precision, despite the cycle
    @test bp_mean ≈ mjoint atol = 1e-9
    # and they are the physics: i = Vs/(R₁+R₂), v = Vs·R₂/(R₁+R₂)
    @test bp_mean[1] ≈ Vs * R₂ / (R₁ + R₂) atol = 1e-4
    @test bp_mean[2] ≈ Vs / (R₁ + R₂) atol = 1e-4
    @test bp_mean[2] ≈ bp_mean[3] atol = 1e-6        # KCL holds

    # 2. the variances are NOT exact — here inflated, by a common factor across the loop
    ratio = bp_var ./ [Σjoint[k, k] for k in 1:3]
    @test all(>(1.5), ratio)
    @test maximum(ratio) - minimum(ratio) < 1e-3     # one cycle, one correction factor

    # 3. and the error is a property of the CYCLE, not of the factor: strengthen the priors
    #    so the loop carries less of the inference, and the discrepancy shrinks toward 1.
    g2, ps2, st2 = divider_graph(1e-2)
    m2, r2, _, _ = infer!(g2, flooding_schedule(g2), ps2, st2; maxsweeps = 2000, tol = 1e-13)
    @test r2.converged
    m2joint, Σ2, _ = joint_solve(divider_blocks(1e-2))
    ratio2 = [belief_cov(m2[v])[1] for v in (:v, :i1, :i2)] ./ [Σ2[k, k] for k in 1:3]
    @test maximum(ratio2) < minimum(ratio)
    # the means stay exact throughout — that is the half of the theorem that is robust
    @test [belief_mean(m2[v])[1] for v in (:v, :i1, :i2)] ≈ m2joint atol = 1e-9
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

@testset "belief algebra: Gaussian mixtures, moment matching, reduction" begin
    dens(b, x) = exp(Mycelium.belief_logdensity(b, [x]))
    xs = range(-6, 8; length = 20001); dx = step(xs)
    m = LenticulumCore.MixtureBelief([Gaussian(-1.0, 0.3), Gaussian(2.0, 0.5)], [0.3, 0.7])
    g = Gaussian(1.0, 1.0)
    # the product of a mixture and a Gaussian, against numerical integration
    pm = Mycelium.combine(m, g)
    num = [dens(m, x) * dens(g, x) for x in xs]; num ./= sum(num) * dx
    @test maximum(abs, num .- [dens(pm, x) for x in xs]) < 1e-10
    # an uninformative (improper) message leaves the mixture unchanged
    pu = Mycelium.combine(m, uninformative(1))
    @test LenticulumCore.mixture_weights(pu) ≈ [0.3, 0.7]
    @test belief_mean(pu.components[1]) ≈ [-1.0]
    # mixture × mixture
    m2 = LenticulumCore.MixtureBelief([Gaussian(0.0, 1.0), Gaussian(3.0, 0.2)], [0.5, 0.5])
    pmm = Mycelium.combine(m, m2)
    num2 = [dens(m, x) * dens(m2, x) for x in xs]; num2 ./= sum(num2) * dx
    @test length(pmm.components) == 4
    @test maximum(abs, num2 .- [dens(pmm, x) for x in xs]) < 1e-10
    # moment matching: the analytic mean and variance of the mixture
    mm = moment_match(m)
    @test belief_mean(mm) ≈ [0.3 * -1 + 0.7 * 2]
    @test belief_cov(mm) ≈ fill(0.3 * 0.3 + 0.7 * 0.5 + 0.3 * 2.1^2 + 0.7 * 0.9^2, 1, 1)
    # reduction merges while preserving the overall mean and covariance
    r = reduce_mixture(pmm; max_components = 2)
    @test length(r.components) == 2
    @test belief_mean(moment_match(r)) ≈ belief_mean(moment_match(pmm))
    @test belief_cov(moment_match(r)) ≈ belief_cov(moment_match(pmm))
    # samples reweighted by a Gaussian, then projected: the conjugate posterior mean
    s = LenticulumCore.SampleBelief(2 .* randn(Xoshiro(1), 1, 200_000))      # prior N(0, 4)
    post = moment_match(Mycelium.combine(s, g))
    @test belief_mean(post)[1] ≈ (1 / 1) / (1 / 4 + 1) atol = 0.02
    @test belief_cov(post)[1] ≈ 1 / (1 / 4 + 1) atol = 0.02
end

# ---------------------------------------------------------------------------
# Interop with the BayesBase / ExponentialFamily / RxInfer ecosystem (LenticulumCore's
# extension). Loaded last, so that everything above runs without it.
# ---------------------------------------------------------------------------
using ExponentialFamily: ExponentialFamily, MvNormalWeightedMeanPrecision, NormalWeightedMeanPrecision,
    MvNormalMeanCovariance, NormalMeanVariance
using BayesBase: BayesBase, PointMass, SampleList, GenericProd
using Distributions: Distributions, Categorical, Bernoulli, MixtureModel
import RxInfer
using RxInfer: GraphPPL   # the @model macro expands to calls into GraphPPL

RxInfer.@model function onestep(y)
    x ~ RxInfer.NormalMeanVariance(0.0, 4.0)
    y ~ RxInfer.NormalMeanVariance(2.0 * x + 1.0, 0.5)
end

@testset "interop: beliefs as RxInfer distributions and back" begin
    g = Gaussian([1.0, -2.0], [2.0 0.5; 0.5 1.0])
    d = as_distribution(g)
    @test d isa MvNormalWeightedMeanPrecision
    @test BayesBase.weightedmean_precision(d)[1] == g.η && BayesBase.weightedmean_precision(d)[2] == g.Λ
    @test Distributions.mean(d) ≈ belief_mean(g) && Distributions.cov(d) ≈ belief_cov(g)
    @test as_belief(d).η ≈ g.η && as_belief(d).Λ ≈ g.Λ                         # exact round trip
    @test as_distribution(Gaussian(0.5, 2.0); univariate = true) isa NormalWeightedMeanPrecision
    @test belief_cov(as_belief(NormalMeanVariance(0.5, 2.0))) ≈ fill(2.0, 1, 1)  # any parametrisation
    @test belief_mean(as_belief(MvNormalMeanCovariance([1.0, 0.0], [1.0 0.0; 0.0 4.0]))) ≈ [1.0, 0.0]
    @test belief_mean(as_belief(Distributions.MvNormal([1.0, 2.0], [1.0 0.0; 0.0 1.0]))) ≈ [1.0, 2.0]
    # an improper Gaussian (a likelihood on one direction) converts too
    lik = GaussianBelief([1.0, 0.0], [1.0 0.0; 0.0 0.0])
    @test as_belief(as_distribution(lik)).Λ == lik.Λ
    # pooling agrees with RxInfer's product
    h = Gaussian([0.0, 1.0], [1.0 0.0; 0.0 3.0])
    p = prod(GenericProd(), as_distribution(g), as_distribution(h))
    @test as_belief(p).η ≈ combine(g, h).η && as_belief(p).Λ ≈ combine(g, h).Λ
    # the discrete and particle beliefs
    c = LenticulumCore.CategoricalBelief([0.2, 0.3, 0.5])
    @test as_distribution(c) isa Categorical && Distributions.probs(as_distribution(c)) ≈ [0.2, 0.3, 0.5]
    @test probabilities(as_belief(as_distribution(c))) ≈ [0.2, 0.3, 0.5]
    @test probabilities(as_belief(Bernoulli(0.3))) ≈ [0.7, 0.3]
    c2 = LenticulumCore.CategoricalBelief([0.5, 0.25, 0.25])
    @test probabilities(combine(c, c2)) ≈ Distributions.probs(prod(GenericProd(), as_distribution(c), as_distribution(c2)))
    x = as_distribution(DiracBelief([1.0, 2.0]))
    @test x isa PointMass && as_belief(x).value == [1.0, 2.0]
    m = LenticulumCore.MixtureBelief([Gaussian(-1.0, 0.5), Gaussian(2.0, 1.0)], [0.25, 0.75])
    md = as_distribution(m)
    @test md isa MixtureModel && Distributions.probs(md) ≈ [0.25, 0.75]
    @test mixture_weights(as_belief(md)) ≈ [0.25, 0.75]
    @test belief_mean(as_belief(md).components[2]) ≈ [2.0]
    s = LenticulumCore.SampleBelief([[1.0], [2.0], [4.0]], [1.0, 1.0, 2.0])
    sl = as_distribution(s)
    @test sl isa SampleList && Distributions.mean(sl) ≈ [2.75]
    @test as_belief(sl).weights ≈ [0.25, 0.25, 0.5]
    @test_throws ArgumentError as_distribution(TrivialBelief())
    # the same conjugate model solved by RxInfer and by Lenticulum's Gaussian factors
    rx = RxInfer.infer(model = onestep(), data = (y = 3.0,))
    post_rx = as_belief(rx.posteriors[:x])
    prior = Gaussian(0.0, 4.0)
    f = GaussianFactor(1 => 1; noise = 0.5)
    ps = (A = fill(2.0, 1, 1), b = [1.0])
    bwd = LenticulumCore.Polarity(; x = Unobserved(), y = Observed())
    msg, _ = Mycelium.factor_message(f, :x, bwd, (; y = DiracBelief([3.0])), TrivialBelief(), ps, NamedTuple())
    post = combine(prior, msg)
    @test belief_mean(post_rx) ≈ belief_mean(post) && belief_cov(post_rx) ≈ belief_cov(post)
end


@testset "NonlinearFactor: EKF and UKF messages, iterated to the MAP" begin
    fwd = LenticulumCore.Polarity(; x = Observed(), y = Unobserved())
    bwd = LenticulumCore.Polarity(; x = Unobserved(), y = Observed())
    # an affine f: both methods reproduce GaussianFactor exactly, in both directions
    Am, bm = [1.0 2.0; -0.5 0.3; 0.0 1.0], [0.1, -0.2, 0.4]
    Qm = Diagonal([0.2, 0.3, 0.1])
    g = GaussianFactor(2 => 3; noise = Matrix(Qm))
    gps = (A = Am, b = bm)
    px = Gaussian([0.3, -0.4], [0.5 0.1; 0.1 0.4])
    py = Gaussian([1.0, 0.0, 2.0], Matrix(0.2I, 3, 3))
    for method in (Linearization(), Unscented())
        f = NonlinearFactor(x -> Am * x .+ bm, 2 => 3; noise = Matrix(Qm), method)
        ps, st = LuxCore.setup(Random.default_rng(), f)
        @test ps == NamedTuple() && !LenticulumCore.islearnable(f)
        for (target, pol, inputs) in ((:y, fwd, (; x = px)), (:x, bwd, (; y = py)), (:x, bwd, (; y = DiracBelief([1.0, 0.0, 2.0]))))
            mn, _ = Mycelium.factor_message(f, target, pol, inputs, px, ps, st)
            mg, _ = Mycelium.factor_message(g, target, pol, inputs, px, gps, NamedTuple())
            @test mn.η ≈ mg.η && mn.Λ ≈ mg.Λ
        end
        Fn, _ = Mycelium.local_free_energy(f, (; x = px, y = DiracBelief([1.0, 0.0, 2.0])), ps, st)
        Fg, _ = Mycelium.local_free_energy(g, (; x = px, y = DiracBelief([1.0, 0.0, 2.0])), gps, NamedTuple())
        @test LenticulumCore.scalarise(LenticulumCore.scalarisation(f), Fn) ≈ LenticulumCore.scalarise(LenticulumCore.scalarisation(g), Fg)
    end
    # a point input is exact: N(f(x), Q)
    sq = NonlinearFactor(x -> x .^ 2, 1 => 1; noise = 0.01)
    m1, _ = Mycelium.factor_message(sq, :y, fwd, (; x = DiracBelief([1.5])), TrivialBelief(), NamedTuple(), NamedTuple())
    @test belief_mean(m1) ≈ [2.25] && belief_cov(m1) ≈ fill(0.01, 1, 1)
    # y = x² for x ~ N(m, s²): E y = m² + s², Var y = 4m²s² + 2s⁴. The UKF gets both, the EKF neither
    m, s² = 1.0, 0.25
    u, _ = Mycelium.factor_message(NonlinearFactor(x -> x .^ 2, 1 => 1; noise = 1e-8, method = Unscented()),
                                   :y, fwd, (; x = Gaussian([m], fill(s², 1, 1))), TrivialBelief(), NamedTuple(), NamedTuple())
    @test belief_mean(u)[1] ≈ m^2 + s² rtol = 1e-6
    @test belief_cov(u)[1] ≈ 4m^2 * s² + 2s²^2 rtol = 1e-6
    e, _ = Mycelium.factor_message(sq, :y, fwd, (; x = Gaussian([m], fill(s², 1, 1))), TrivialBelief(), NamedTuple(), NamedTuple())
    @test belief_mean(e)[1] ≈ m^2 && belief_cov(e)[1] ≈ 4m^2 * s² + 0.01
    # finite-difference and supplied Jacobians agree
    cube(x) = [x[1]^3 + x[2], sin(x[2])]
    fd = NonlinearFactor(cube, 2 => 2)
    ex = NonlinearFactor(cube, 2 => 2; jacobian = x -> [3x[1]^2 1.0; 0.0 cos(x[2])])
    @test linearize(fd, [0.7, -0.2], zeros(2, 2))[1] ≈ linearize(ex, [0.7, -0.2], zeros(2, 2))[1] rtol = 1e-8
    # backwards, iterated: the posterior mean is the MAP of N(x; 1, 0.5²) N(2; x³, 0.1²)
    f3 = NonlinearFactor(x -> x .^ 3, 1 => 1; noise = 0.01, method = Linearization(iterations = 30))
    prior = Gaussian([1.0], fill(0.25, 1, 1))
    lens, _ = LenticulumCore.assemble(f3, bwd, NamedTuple(), NamedTuple())
    post, _ = LenticulumCore.invert(lens, prior, (; y = DiracBelief([2.0])), NamedTuple(), NamedTuple())
    xs = range(0.5, 1.8; length = 200001)
    logp(x) = -(x - 1)^2 / (2 * 0.25) - (2 - x^3)^2 / (2 * 0.01)
    xmap = xs[argmax(logp.(xs))]
    @test belief_mean(post)[1] ≈ xmap atol = 1e-5
    @test belief_cov(post)[1] ≈ inv(1 / 0.25 + (3xmap^2)^2 / 0.01) rtol = 1e-4   # Gauss–Newton curvature
    one_step, _ = LenticulumCore.invert(first(LenticulumCore.assemble(NonlinearFactor(x -> x .^ 3, 1 => 1; noise = 0.01), bwd,
                                              NamedTuple(), NamedTuple())), prior, (; y = DiracBelief([2.0])), NamedTuple(), NamedTuple())
    @test abs(belief_mean(one_step)[1] - xmap) > 10 * abs(belief_mean(post)[1] - xmap)   # one EKF step is not the MAP
    # no expansion point: uninformative, unless x0 is given
    b0, _ = Mycelium.factor_message(f3, :x, bwd, (; y = DiracBelief([2.0])), TrivialBelief(), NamedTuple(), NamedTuple())
    @test b0 isa TrivialBelief
    f30 = NonlinearFactor(x -> x .^ 3, 1 => 1; noise = 0.01, x0 = [1.0])
    @test Mycelium.factor_message(f30, :x, bwd, (; y = DiracBelief([2.0])), TrivialBelief(), NamedTuple(), NamedTuple())[1] isa GaussianBelief
    # in a graph: prior → x ─[x³]─ y ← data. The backward message needs its cavity (the prior's
    # message), which a single tree sweep sends too late; repeated sweeps give one EKF update
    gb = GraphBuilder()
    variable!(gb, :x, 1); variable!(gb, :y, 1)
    factor!(gb, :prior, GaussianPrior(:x, 1.0, 0.25))
    factor!(gb, :meas, NonlinearFactor(x -> x .^ 3, 1 => 1; noise = 0.01))
    factor!(gb, :data, DataFactor(:y, [2.0]))
    connect!(gb, :prior, :x, :x; direction = Bidirectional())
    connect!(gb, :meas, :x, :x; direction = Bidirectional())
    connect!(gb, :meas, :y, :y; direction = Bidirectional())
    connect!(gb, :data, :y, :y; direction = Emitting())
    G = validate(build(gb))
    pz = (prior = NamedTuple(), meas = NamedTuple(), data = NamedTuple())
    marg, report, _, _ = infer!(G, SequentialSchedule(tasks(tree_schedule(G))), pz, pz)
    @test report.converged
    @test belief_mean(marg.x) ≈ belief_mean(one_step) && belief_cov(marg.x) ≈ belief_cov(one_step)
    @test_throws ArgumentError NonlinearFactor(identity, 1 => 1; method = :ukf)
end

@testset "LikelihoodFactor: Laplace and EP messages, the predictive over outcomes" begin
    bwd = LenticulumCore.Polarity(; x = Unobserved(), y = Observed())
    fwd = LenticulumCore.Polarity(; x = Observed(), y = Unobserved())
    msg(f, cav, y) = Mycelium.factor_message(f, :x, bwd, (; y = DiracBelief(y)), cav, NamedTuple(), NamedTuple())[1]
    cav = Gaussian([0.5], fill(0.3, 1, 1))

    # Poisson: EP is the tilted mean and variance, Laplace the mode and curvature
    ref = brute_tilted(z -> 3z - exp(z) - log(6), 0.5, 0.3)
    fe = LikelihoodFactor(PoissonLikelihood(), 1; method = ExpectationPropagation())
    q = Mycelium.combine(cav, msg(fe, cav, [3.0]))
    @test belief_mean(q)[1] ≈ ref[1] rtol = 1e-8
    @test belief_cov(q)[1] ≈ ref[2] rtol = 1e-8
    @test log_evidence(fe, cav, [3.0]) ≈ ref[3] rtol = 1e-8
    fl = LikelihoodFactor(PoissonLikelihood(), 1; method = Laplace())
    ql = Mycelium.combine(cav, msg(fl, cav, [3.0]))
    @test belief_mean(ql)[1] ≈ ref[4] atol = 1e-4
    @test belief_cov(ql)[1] ≈ inv(1 / 0.3 + exp(belief_mean(ql)[1]))
    @test msg(fl, cav, [3.0]).Λ[1] > 0                       # log-concave: a positive precision
    # a large count under a wide cavity: the adaptive grid still gets the moments
    ref50 = brute_tilted(z -> 50z - exp(z) - Lenticulum._logfact(50), 0.0, 4.0)
    q50 = Mycelium.combine(Gaussian([0.0], fill(4.0, 1, 1)), msg(fe, Gaussian([0.0], fill(4.0, 1, 1)), [50.0]))
    @test belief_mean(q50)[1] ≈ ref50[1] rtol = 1e-7
    @test belief_cov(q50)[1] ≈ ref50[2] rtol = 1e-6

    # ordinal probit: the closed-form EP moments, the quadrature path and the brute force agree
    op = OrdinalProbitLikelihood([-1.0, 0.5, 2.0]; scale = 0.7)
    cz = Gaussian([0.2], fill(1.5, 1, 1))
    for k in 1:4
        r = brute_tilted(z -> loglikelihood(op, [k], [z]), 0.2, 1.5)
        qc = Mycelium.combine(cz, msg(LikelihoodFactor(op, 1; method = ExpectationPropagation()), cz, [float(k)]))
        qq = Mycelium.combine(cz, msg(LikelihoodFactor(FDOrdinal(op), 1; method = ExpectationPropagation(order = 40)), cz, [float(k)]))
        @test belief_mean(qc)[1] ≈ r[1] rtol = 1e-7
        @test belief_cov(qc)[1] ≈ r[2] rtol = 1e-6
        @test belief_mean(qq)[1] ≈ r[1] rtol = 1e-5
        @test belief_cov(qq)[1] ≈ r[2] rtol = 1e-4
        @test log_evidence(LikelihoodFactor(op, 1; method = ExpectationPropagation()), cz, [k]) ≈ r[3] rtol = 1e-7
        ℓ, g, H = loglikelihood_derivatives(op, [k], [0.3])
        _, gd, Hd = invoke(loglikelihood_derivatives, Tuple{AbstractLikelihood,Any,Any}, op, [k], [0.3])
        @test g ≈ gd rtol = 1e-6
        @test H ≈ Hd rtol = 1e-4
    end
    # far in a tail the log-probabilities stay finite and the derivatives exact
    ℓt, gt, _ = loglikelihood_derivatives(op, [1], [40.0])
    @test isfinite(ℓt)
    @test gt[1] ≈ -(40.0 + 1.0) / 0.7^2 rtol = 0.01   # the Mills ratio: g ≈ -(z - c₁)/σ²

    # a two-dimensional latent read through one predictor: the message lives on the row of A,
    # and the posterior is the exact tilted mean and covariance in x
    A2 = [1.0 -1.0]
    f2 = LikelihoodFactor(op, 2; A = A2, b = [0.1], method = ExpectationPropagation())
    c2 = Gaussian([0.3, -0.2], [1.0 0.3; 0.3 0.6])
    m2 = msg(f2, c2, [3.0])
    @test rank(m2.Λ; atol = 1e-10) == 1 && m2.Λ ≈ m2.Λ[1, 1] .* [1 -1; -1 1]
    q2 = Mycelium.combine(c2, m2)
    S2 = belief_cov(c2); μz = (A2 * belief_mean(c2))[1] + 0.1; sz = (A2 * S2 * A2')[1]
    r2 = brute_tilted(z -> loglikelihood(op, [3], [z]), μz, sz)
    K = S2 * A2' / sz                               # x | z is the cavity's conditional
    @test belief_mean(q2) ≈ belief_mean(c2) .+ vec(K) .* (r2[1] - μz) rtol = 1e-6
    @test belief_cov(q2) ≈ S2 .- K * K' .* (sz - r2[2]) rtol = 1e-5

    # the predictive: ordinal in closed form, against quadrature over z
    p, _ = Mycelium.factor_message(f2, :y, fwd, (; x = c2), TrivialBelief(), NamedTuple(), NamedTuple())
    @test p isa LenticulumCore.CategoricalBelief && p.labels == [[1.0], [2.0], [3.0], [4.0]]
    @test probabilities(p) ≈ [exp(brute_tilted(z -> loglikelihood(op, [k], [z]), μz, sz)[3]) for k in 1:4] rtol = 1e-7
    # Poisson predictive with the tail lumped into the top count: sums to one, matches the mixture
    fp = LikelihoodFactor(PoissonLikelihood(; maxcount = 8), 1; method = ExpectationPropagation(order = 40))
    pp = probabilities(predictive(fp, cav))
    @test sum(pp) ≈ 1
    @test pp[1:8] ≈ [exp(brute_tilted(z -> j * z - exp(z) - Lenticulum._logfact(j), 0.5, 0.3)[3]) for j in 0:7] rtol = 1e-8
    # at a point: exactly the likelihood's probabilities
    pd = probabilities(predictive(fp, DiracBelief([log(2.0)])))
    @test pd[1:8] ≈ [exp(-2.0) * 2.0^j / factorial(j) for j in 0:7]
    @test pd[9] ≈ 1 - sum(pd[1:8])

    # bivariate Poisson: a proper pmf with Cov(y₁, y₂) = λ₃, exact derivatives
    bp = BivariatePoissonLikelihood(; maxcount = 40)
    zb = log.([1.2, 0.7, 0.4])
    P = [exp(loglikelihood(bp, [i, j], zb)) for i in 0:39, j in 0:39]
    @test sum(P) ≈ 1 atol = 1e-12
    E1 = sum(P .* (0:39)); E2 = sum(P .* (0:39)')
    @test sum(P .* ((0:39) .* (0:39)')) - E1 * E2 ≈ 0.4 rtol = 1e-8
    @test E1 ≈ 1.6 && E2 ≈ 1.1
    for y in ([0, 0], [3, 1], [2, 5])
        _, g, H = loglikelihood_derivatives(bp, y, zb)
        _, gd, Hd = invoke(loglikelihood_derivatives, Tuple{AbstractLikelihood,Any,Any}, bp, y, zb)
        @test g ≈ gd rtol = 1e-6
        @test H ≈ Hd rtol = 1e-4 atol = 1e-6
    end
    small = BivariatePoissonLikelihood(; maxcount = 5)
    pb = predictive(LikelihoodFactor(small, 3), DiracBelief(zb))
    @test sum(probabilities(pb)) ≈ 1 && length(pb.labels) == 36
    @test probabilities(pb)[1:5] ≈ P[1:5, 1]                  # y₂ = 0, y₁ = 0 … 4
    # two latents, three predictors (λ₃ fixed by the offset): on a 2-d grid, EP's posterior mean
    # is the tilted mean and Laplace's is the mode
    Ab = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    cb = Gaussian([0.2, -0.1], [0.4 0.1; 0.1 0.3])
    fb = LikelihoodFactor(bp, 2; A = Ab, b = [0.0, 0.0, log(0.4)], method = ExpectationPropagation(order = 30))
    qb = Mycelium.combine(cb, msg(fb, cb, [3.0, 1.0]))
    xs = range(-3, 3; length = 601)
    Sb = belief_cov(cb); mb = belief_mean(cb)
    lw = [-(([u, v] .- mb)' * (Sb \ ([u, v] .- mb))) / 2 + loglikelihood(bp, [3, 1], Ab * [u, v] .+ [0, 0, log(0.4)]) for u in xs, v in xs]
    w = exp.(lw .- maximum(lw)); w ./= sum(w)
    @test belief_mean(qb) ≈ [sum(w .* xs), sum(w .* xs')] rtol = 1e-5
    ql2 = Mycelium.combine(cb, msg(LikelihoodFactor(bp, 2; A = Ab, b = [0.0, 0.0, log(0.4)]), cb, [3.0, 1.0]))
    imax = argmax(lw)                               # Laplace: the mode, not the mean
    @test belief_mean(ql2) ≈ [xs[imax[1]], xs[imax[2]]] atol = 0.006
    @test norm(belief_mean(ql2) .- belief_mean(qb)) > 0.05   # the two differ: the posterior is skewed

    # no cavity: uninformative, or the expansion at x0
    @test msg(fl, TrivialBelief(), [3.0]) isa TrivialBelief
    @test msg(fl, uninformative(1), [3.0]) isa TrivialBelief
    e0 = msg(LikelihoodFactor(PoissonLikelihood(), 1; x0 = [0.0]), TrivialBelief(), [3.0])
    @test e0.Λ ≈ fill(1.0, 1, 1) && e0.η ≈ [2.0]             # ℓ ≈ 2x - x²/2 at 0
    @test_throws ArgumentError msg(fl, cav, [2.5])            # not a count
    @test_throws ArgumentError msg(fl, cav, [-1.0])
    @test_throws ArgumentError LikelihoodFactor(PoissonLikelihood(2), 3)    # A is needed
    @test_throws ArgumentError OrdinalProbitLikelihood([1.0, 0.0])

    # in a graph: prior → x ─[Poisson]─ y ← data. The Bethe free energy of the EP posterior is a
    # variational bound on −log p(y), and close to it
    for method in (Laplace(), ExpectationPropagation())
        gb = GraphBuilder()
        variable!(gb, :x, 1); variable!(gb, :y, 1)
        factor!(gb, :prior, GaussianPrior(:x, 0.5, 0.3))
        factor!(gb, :obs, LikelihoodFactor(PoissonLikelihood(), 1; method))
        factor!(gb, :data, DataFactor(:y, [3.0]))
        connect!(gb, :prior, :x, :x; direction = Bidirectional())
        connect!(gb, :obs, :x, :x; direction = Bidirectional())
        connect!(gb, :obs, :y, :y; direction = Bidirectional())
        connect!(gb, :data, :y, :y; direction = Emitting())
        G = validate(build(gb))
        pz = (prior = NamedTuple(), obs = NamedTuple(), data = NamedTuple())
        marg, report, st, store = infer!(G, SequentialSchedule(tasks(tree_schedule(G))), pz, pz)
        @test report.converged
        qx = method isa Laplace ? ql : q
        @test belief_mean(marg.x) ≈ belief_mean(qx) && belief_cov(marg.x) ≈ belief_cov(qx)
        F, _ = LenticulumCore.scalar_free_energy(store, G, pz, st)
        @test F ≥ -ref[3] - 1e-9
        @test F + ref[3] < 0.01
    end
    # at a point the factor's energy is the negative log-likelihood
    Fd, _ = Mycelium.local_free_energy(fl, (; x = DiracBelief([log(2.0)]), y = DiracBelief([3.0])), NamedTuple(), NamedTuple())
    @test LenticulumCore.scalarise(LenticulumCore.scalarisation(fl), Fd) ≈ -log(exp(-2.0) * 8 / 6)

    # The count predictive grid lumps the tail: it cannot give exact conditional entropy.
    @test_throws ArgumentError Mycelium.local_free_energy(fl, (; x = cav, y = nothing),
                                                         NamedTuple(), NamedTuple())

    # BivariatePoissonLikelihood handles z = -Inf without 0 * (-Inf) == NaN
    @test isfinite(loglikelihood(bp, [2, 1], [0.0, 0.0, -Inf]))
end

@testset "Factor interface conformance (L-SPEC-001 / LP-008) & game-level free_energy (LP-009)" begin
    rng = Random.default_rng()
    factors = [
        GaussianFactor(2 => 2; noise = 0.2),
        GaussianPrior(:x, [0.0, 0.0], Matrix(1.0I, 2, 2)),
        LinearConstraintFactor((x = 2, y = 2), 2; noise = 0.2),
        NonlinearFactor(x -> x .^ 2, 2 => 2; noise = 0.2),
        LikelihoodFactor(PoissonLikelihood(1), 1),
    ]
    for f in factors
        ps, st = LuxCore.setup(rng, f)
        chans = LenticulumCore.channels(f)
        @test !isempty(chans)
        pols = LenticulumCore.supported_polarities(f)
        @test !isempty(pols)
        for p in pols
            @test LenticulumCore.supports_polarity(f, p)
            @test LenticulumCore.ispartition(p)
        end
        @test LenticulumCore.energyspace(f) isa LenticulumCore.AbstractEnergySpace
        @test LenticulumCore.scalarisation(f) isa LenticulumCore.AbstractScalarisation
        @test LenticulumCore.pointwise_energyspace(f) isa LenticulumCore.AbstractEnergySpace
        @test LenticulumCore.pointwise_scalarisation(f) isa LenticulumCore.AbstractScalarisation
    end

    # Game-level free_energy and entropy (LP-009)
    gf = GaussianFactor(1 => 1; noise = Q)
    gps = (A = A, b = bb)
    px = Gaussian(μ₀, Σ₀)
    Fg_game, _ = LenticulumCore.free_energy(gf, px, y₀, gps, NamedTuple())
    Hg_game, _ = LenticulumCore.entropy(gf, px, y₀, gps, NamedTuple())
    @test Fg_game[:negentropy] ≈ -Hg_game[:negentropy]
    sfe, _ = LenticulumCore.scalar_free_energy(gf, px, y₀, gps, NamedTuple())
    @test isfinite(sfe)
end

@testset "game entropy occupies its free-energy coordinate" begin
    nt = NamedTuple()
    πx = Gaussian([1.0], fill(1.0, 1, 1))
    cases = (
        (GaussianFactor(1 => 1; noise = 0.1), (A = fill(2.0, 1, 1), b = [0.0]), [4.0]),
        (GaussianPrior(:x, 0.0, 1.0), nt, nothing),
        (LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.1),
         (A = (x = fill(-2.0, 1, 1), y = fill(1.0, 1, 1)), c = [0.0]), (y = DiracBelief([4.0]),)),
        (NonlinearFactor(x -> x .^ 2, 1 => 1; noise = 0.1), nt, [4.0]),
        (LikelihoodFactor(OrdinalProbitLikelihood([-1.0, 1.0]), 1), nt, [2.0]),
    )
    for (f, ps, y) in cases
        F, _ = LenticulumCore.free_energy(f, πx, y, ps, nt)
        H, _ = LenticulumCore.entropy(f, πx, y, ps, nt)
        @test H isa GradedEnergy
        if H isa GradedEnergy
            @test keys(H) == keys(F)
            @test H[:fit] == H[:complexity] == 0
            @test H[:negentropy] ≈ -F[:negentropy]
            @test scalarise(scalarisation(f), H) ≈ -F[:negentropy]
            @test compose_entropy([H, H], H)[:first][:negentropy] ≈ H[:negentropy]
        end
    end
end

@testset "iterated nonlinear likelihood is the final posterior site" begin
    nt = NamedTuple()
    for method in (Linearization(iterations = 4), Unscented(iterations = 4, α = 1.0))
        f = NonlinearFactor(x -> x .^ 2, 1 => 1; noise = 0.1, method)
        πx = Gaussian([1.0], fill(1.0, 1, 1))
        p = Polarity(x = Unobserved(), y = Observed())
        lens, _ = assemble(f, p, nt, nt)
        inputs = (y = DiracBelief([4.0]),)
        msg, _ = likelihood_message(lens, πx, inputs, nt, nt)
        post, _ = invert(lens, πx, inputs, nt, nt)
        pooled = combine(πx, msg)
        @test belief_mean(pooled) ≈ belief_mean(post)
        @test belief_cov(pooled) ≈ belief_cov(post)
        method isa Linearization && @test belief_mean(post)[1] ≈ 1.993762 atol = 1e-6
    end
end

@testset "predictive free energy retains conditional entropy" begin
    nt = NamedTuple()
    f = LikelihoodFactor(OrdinalProbitLikelihood([-1.0, 1.0]), 1)
    x = DiracBelief([0.0])
    F, _ = Mycelium.local_free_energy(f, (x = x, y = nothing), nt, nt)
    H = Mycelium.variable_entropy(predictive(f, x))
    @test H ≈ 0.844768 atol = 1e-6
    @test F[:fit] ≈ H
    @test F[:negentropy] ≈ -H
    σ = GradedScalarisation((fit = WeightedSum(2.0), complexity = IdentityScalarisation(),
                             negentropy = IdentityScalarisation()))
    @test scalarise(σ, F) ≈ H
    @test scalarise(scalarisation(f), F) ≈ 0 atol = 1e-14
    q = Gaussian([0.2], fill(0.3, 1, 1))
    Fq, _ = Mycelium.local_free_energy(f, (x = q, y = nothing), nt, nt)
    # Independent higher-order quadrature of the exact finite conditional entropy.
    X, W = Lenticulum._gh_grid(80, 1)
    expected = sum(W[i] * Mycelium.variable_entropy(predictive(f,
                   DiracBelief([0.2 + sqrt(0.3) * X[1, i]]))) for i in eachindex(W))
    @test Fq[:fit] ≈ expected rtol = 1e-7
    @test Fq[:negentropy] ≈ -expected - Mycelium.variable_entropy(q) rtol = 1e-7
    for L in (PoissonLikelihood(1), BivariatePoissonLikelihood())
        fc = LikelihoodFactor(L, npredictors(L))
        @test_throws ArgumentError Mycelium.local_free_energy(fc,
            (x = DiracBelief(zeros(npredictors(L))), y = nothing), nt, nt)
    end
    @test_throws ArgumentError Mycelium.local_free_energy(f, (x = nothing, y = nothing), nt, nt)
end
