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
