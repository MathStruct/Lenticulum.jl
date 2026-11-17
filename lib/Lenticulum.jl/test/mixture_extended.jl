# P27 (Feature Backlog): a switching factor's mixture message meeting a hard relation's extended
# message on one variable. Mycelium's mixture product takes an extended component
# (`combine_with_overlap` of a Gaussian and an `ExtendedGaussian`, `Mycelium/src/gaussian.jl`), and
# the relation factors collapse a mixture input of extended components to its moments.
#
# Oracle: enumeration of the outcome k. The graph is a tree in which the switching factor is a leaf
# on u, so the marginals of u and k are exact: for each k the replica is linear-Gaussian on the
# support of the hard relation, solved densely there (z = z₀ + N t), with log evidence
# log π_k + the Gaussian integral over t. The marginal of k is the softmax of these, and the marginal
# of u is the mixture of the u-blocks of the conditional laws, with those weights. The messages
# leaving u (to the relation factors) collapse the mixture, an assumed-density approximation:
# the marginals of v and y are not exact and are only checked to exist.
#
# With the expectation-propagation rule on the switching node the site is the projection of the
# tilted mixture (cavity × message, extended components) divided by the extended cavity. On this
# tree the cavity does not depend on the site, so at the fixed point the marginal of u is the
# moment projection of the exact mixture (an extended Gaussian with its total mean and covariance)
# and the marginal of k is still exact.

using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random

module MixtureExtended
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# replica i: w = W[:, i] observed through LinearConstraintFactor (w, u); v ~ N(μv, Σv); the hard
# relation A_u u + A_v v = c; y ~ N(μy, Σy) with LinearConstraintFactor (u, y); a switching factor
# on u with outcome k.
function template!(p, wi, d, fx, rule)
    nw, nu, nv, ny, m, K = d
    variable!(p, :w, nw); variable!(p, :u, nu); variable!(p, :v, nv); variable!(p, :y, ny); variable!(p, :k, K)
    factor!(p, :data, DataFactor(:x, wi)); connect!(p, :data, :x, :w; direction = Emitting())
    factor!(p, :obs, LinearConstraintFactor((w = nw, u = nu), nw; noise = fx.Qw))
    connect!(p, :obs, :w, :w; direction = Absorbing()); connect!(p, :obs, :u, :u)
    factor!(p, :pv, GaussianPrior(:x, fx.μv, fx.Σv)); connect!(p, :pv, :x, :v)
    factor!(p, :rel, HardConstraintFactor((u = nu, v = nv), m)); connect!(p, :rel, :u, :u); connect!(p, :rel, :v, :v)
    factor!(p, :py, GaussianPrior(:x, fx.μy, fx.Σy)); connect!(p, :py, :x, :y)
    factor!(p, :lin, LinearConstraintFactor((u = nu, y = ny), ny; noise = fx.Qy)); connect!(p, :lin, :u, :u); connect!(p, :lin, :y, :y)
    factor!(p, :sw, SwitchingConstraintFactor((u = nu,), K, nu; noise = fx.Qs); rule); connect!(p, :sw, :u, :u); connect!(p, :sw, :k, :k)
end

function plate_graph(W, d, fx, rule = BeliefPropagation())
    b = GraphBuilder()
    plate!((p, i) -> template!(p, W[:, i], d, fx, rule), b, :r, size(W, 2))
    return validate(build(b))
end

# the exact log evidence of each outcome and the conditional law of u under it, for replica wi
function oracle(ps, wi, d, fx)
    nw, nu, nv, ny, m, K = d
    nz = nu + nv + ny
    Eu = [Matrix(1.0I, nu, nu) zeros(nu, nv + ny)]
    Ev = [zeros(nv, nu) Matrix(1.0I, nv, nv) zeros(nv, ny)]
    Ey = [zeros(ny, nu + nv) Matrix(1.0I, ny, ny)]
    soft = [(ps.obs.A.u * Eu, ps.obs.A.w * wi - ps.obs.c, inv(fx.Qw)),
            (Ev, -fx.μv, inv(fx.Σv)), (Ey, -fx.μy, inv(fx.Σy)),
            (ps.lin.A.u * Eu + ps.lin.A.y * Ey, -ps.lin.c, inv(fx.Qy))]
    C = ps.rel.A.u * Eu + ps.rel.A.v * Ev
    z0 = pinv(C) * ps.rel.c
    N = nullspace(C)
    logπ = ps.sw.logw .- log(sum(exp.(ps.sw.logw)))
    out = map(1:K) do k
        terms = vcat(soft, [(ps.sw.A.u * Eu, -ps.sw.b[:, k], inv(fx.Qs))])
        P = sum(R' * Wt * R for (R, _, Wt) in terms)
        h = -sum(R' * Wt * r for (R, r, Wt) in terms)
        c0 = -sum(r' * Wt * r for (_, r, Wt) in terms) / 2 - logdet(fx.Qs) / 2
        Pt = Symmetric(N' * P * N); ht = N' * (h - P * z0)
        logZ = logπ[k] + c0 + dot(h, z0) - dot(z0, P * z0) / 2 + dot(ht, Pt \ ht) / 2 - logdet(Pt) / 2
        μ = z0 + N * (Pt \ ht); Σ = N * (Pt \ N')
        (logZ, μ[1:nu], Matrix(Symmetric(Σ[1:nu, 1:nu])))
    end
    l = first.(out)
    return exp.(l .- maximum(l)) ./ sum(exp.(l .- maximum(l))), [o[2] for o in out], [o[3] for o in out]
end
end

@testset "P27: a mixture message meets a hard relation (oracle: enumeration of the outcome)" begin
    ME = MixtureExtended
    rng = Xoshiro(27)
    worst = 0.0
    for trial in 1:12, rule in (BeliefPropagation(), ExpectationPropagationRule())
        nu, nv = rand(rng, 1:3), rand(rng, 1:2)
        m = rand(rng, 1:(nu + nv - 1))                # the relation leaves a support of dimension ≥ 1
        d = (rand(rng, 1:2), nu, nv, rand(rng, 1:2), m, rand(rng, 2:3))
        nw, _, _, ny, _, K = d
        fx = (Qw = ME.spd(rng, nw), μv = randn(rng, nv), Σv = ME.spd(rng, nv), μy = randn(rng, ny),
              Σy = ME.spd(rng, ny), Qy = ME.spd(rng, ny), Qs = ME.spd(rng, nu))
        R = 3
        W = randn(rng, nw, R)
        g = ME.plate_graph(W, d, fx, rule)
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        ps = merge(ps, (sw = merge(ps.sw, (logw = randn(rng, K), b = 2 .* randn(rng, nu, K))),))
        for sched in (plate_schedule(g), SequentialSchedule(tasks(plate_schedule(g))))
            s = MessageStore(g)
            for _ in 1:4
                sweep!(s, g, sched, ps, st)
            end
            for i in 1:R
                q, μs, Σs = ME.oracle(ps, W[:, i], d, fx)
                mk = marginal(s, g, g.variable_index[replica_name(:k, i)])
                mu = marginal(s, g, g.variable_index[replica_name(:u, i)])
                @test probabilities(mk) ≈ q rtol = 1e-8
                if rule isa BeliefPropagation
                    @test mu isa MixtureBelief
                    @test LenticulumCore.mixture_weights(mu) ≈ q rtol = 1e-8
                    for k in 1:K
                        c = mu.components[k]
                        @test belief_mean(c) ≈ μs[k] rtol = 1e-8 atol = 1e-10
                        @test belief_cov(c) ≈ Σs[k] rtol = 1e-8 atol = 1e-10
                        worst = max(worst, norm(belief_mean(c) - μs[k]))
                    end
                else
                    μ = sum(q[k] .* μs[k] for k in 1:K)
                    Σ = sum(q[k] .* (Σs[k] .+ (μs[k] .- μ) * (μs[k] .- μ)') for k in 1:K)
                    @test mu isa ExtendedGaussian
                    @test belief_mean(mu) ≈ μ rtol = 1e-8 atol = 1e-10
                    @test belief_cov(mu) ≈ Σ rtol = 1e-8 atol = 1e-10
                    worst = max(worst, norm(belief_mean(mu) - μ))
                end
                # the messages that leave u collapse the mixture: v and y have a marginal
                for v in (:v, :y)
                    b = marginal(s, g, g.variable_index[replica_name(v, i)])
                    @test all(isfinite, belief_mean(b isa ExtendedGaussian ? b : LenticulumCore.extended(b)))
                end
            end
        end
    end
    @info "P27 oracle: largest error of a component mean" worst
end
