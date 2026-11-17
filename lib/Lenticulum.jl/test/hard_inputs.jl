# P2 (Feature Backlog): hard inputs (extended Gaussians with hard directions, from a
# `HardConstraintFactor`) in the free energy of `SwitchingConstraintFactor`, in the frozen factor
# beliefs of the parameter gradient (`factor_belief`, LP-020), and in plates.
#
# Oracles:
# 1. On a tree the Bethe free energy is −log Z (`hard_free_energy.jl`), so the envelope gradient
#    (the default driver, frozen beliefs) and the replay with diagonal couplings everywhere are
#    d(−log Z) of the dense closed form Z = N(c; Cμ, CΣCᵀ), differentiated by ForwardDiff.
# 2. A switching factor with an observed outcome k beside a hard relation, in a tree: −log Z is
#    −log w_k plus the negative log density of (P z + ε, C z) at (b_k, c), z the Gaussian prior.
# 3. Factor by factor, the switching factor's term with a hard input is the limit of its term
#    with variance ε along the hard directions, after the divergent entropy
#    −(h/2) log(2πeε) is removed (h hard directions); with K = 1 it is LinearConstraintFactor's
#    term (oracle-tested in `hard_free_energy.jl`).
# 4. The frozen belief agrees with the local free energy where it was evaluated (LP-020), for
#    GaussianFactor, GaussianPrior, LinearConstraintFactor and SwitchingConstraintFactor.
# 5. Zygote through the switching term with hard inputs, against ForwardDiff.
# 6. Plates: the free energy and the envelope gradient of a plate with hard relations are the
#    sums over its replicas taken as their own graphs.

using Test, Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, LuxCore, Random
using ForwardDiff
import Zygote

module HardInputs
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, ForwardDiff

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# x ~ N(μ₀, Σ₀); u = A₁x + b₁ + ε₁; w = A₂u + b₂ + ε₂; v ~ N(μᵥ, Σᵥ); Cᵤu + Cᵥv = c exactly
function model(rng, n, nu, nv, nw, m; coupling = nothing)
    gb = GraphBuilder()
    kw = coupling === nothing ? (;) : (; coupling)
    variable!(gb, :x, n); variable!(gb, :u, nu); variable!(gb, :v, nv); variable!(gb, :w, nw)
    μ0, Σ0, μv, Σv = randn(rng, n), spd(rng, n), randn(rng, nv), spd(rng, nv)
    Q1, Q2 = spd(rng, nu), spd(rng, nw)
    factor!(gb, :px, GaussianPrior(:x, μ0, Σ0)); connect!(gb, :px, :x, :x; kw...)
    factor!(gb, :pv, GaussianPrior(:x, μv, Σv)); connect!(gb, :pv, :x, :v; kw...)
    factor!(gb, :g1, GaussianFactor(n => nu; noise = Q1))
    connect!(gb, :g1, :x, :x; kw...); connect!(gb, :g1, :y, :u; kw...)
    factor!(gb, :g2, GaussianFactor(nu => nw; noise = Q2))
    connect!(gb, :g2, :x, :u; kw...); connect!(gb, :g2, :y, :w; kw...)
    factor!(gb, :rel, HardConstraintFactor((u = nu, v = nv), m))
    connect!(gb, :rel, :u, :u; kw...); connect!(gb, :rel, :v, :v; kw...)
    return validate(build(gb)), (; μ0, Σ0, μv, Σv, Q1, Q2)
end

# −log Z of that model as a function of (A₁, b₁, A₂, b₂), for any element type
function nlogZ(p, rel, A1, b1, A2, b2)
    n, nu, nv, nw = size(A1, 2), size(A1, 1), length(p.μv), size(A2, 1)
    Z(r, c) = zeros(r, c)
    L = [I(n) Z(n, nu) Z(n, nv) Z(n, nw)
         A1 I(nu) Z(nu, nv) Z(nu, nw)
         Z(nv, n) Z(nv, nu) I(nv) Z(nv, nw)
         A2 * A1 A2 Z(nw, nv) I(nw)]
    o = vcat(zeros(n), b1, zeros(nv), A2 * b1 + b2)
    S = cat(p.Σ0, p.Q1, p.Σv, p.Q2; dims = (1, 2))
    μ = L * vcat(p.μ0, zeros(nu), p.μv, zeros(nw)) + o
    C = hcat(zeros(size(rel.c, 1), n), rel.A.u, rel.A.v, zeros(size(rel.c, 1), nw))
    Sc = C * L * S * L' * C'
    Sc = (Sc + Sc') / 2
    d = rel.c - C * μ
    return (d' * (Sc \ d) + logdet(2π * Sc)) / 2
end

flat(G) = vcat(vec(G.g1.A), G.g1.b, vec(G.g2.A), G.g2.b)
function dense_gradient(p, ps)
    sz = (size(ps.g1.A), length(ps.g1.b), size(ps.g2.A), length(ps.g2.b))
    function f(θ)
        i = 0
        take(k) = (r = θ[(i + 1):(i + k)]; i += k; r)
        A1 = reshape(take(prod(sz[1])), sz[1]); b1 = take(sz[2])
        A2 = reshape(take(prod(sz[3])), sz[3]); b2 = take(sz[4])
        return nlogZ(p, ps.rel, A1, b1, A2, b2)
    end
    return ForwardDiff.gradient(f, flat(ps))
end

# u ~ N(μu, Σu), v ~ N(μv, Σv); Cᵤu + Cᵥv = c exactly; a switching factor on (u, y) with
# y ~ N(μy, Σy) and the outcome k observed
function switch_model(rng, nu, nv, ny, m, ms, K, k)
    gb = GraphBuilder()
    variable!(gb, :u, nu); variable!(gb, :v, nv); variable!(gb, :y, ny); variable!(gb, :k, K)
    fx = (μu = randn(rng, nu), Σu = spd(rng, nu), μv = randn(rng, nv), Σv = spd(rng, nv),
          μy = randn(rng, ny), Σy = spd(rng, ny), Qs = spd(rng, ms))
    factor!(gb, :pu, GaussianPrior(:x, fx.μu, fx.Σu)); connect!(gb, :pu, :x, :u)
    factor!(gb, :pv, GaussianPrior(:x, fx.μv, fx.Σv)); connect!(gb, :pv, :x, :v)
    factor!(gb, :py, GaussianPrior(:x, fx.μy, fx.Σy)); connect!(gb, :py, :x, :y)
    factor!(gb, :rel, HardConstraintFactor((u = nu, v = nv), m)); connect!(gb, :rel, :u, :u); connect!(gb, :rel, :v, :v)
    factor!(gb, :sw, SwitchingConstraintFactor((u = nu, y = ny), K, ms; noise = fx.Qs))
    connect!(gb, :sw, :u, :u); connect!(gb, :sw, :y, :y); connect!(gb, :sw, :k, :k)
    factor!(gb, :dk, DataFactor(:k, k)); connect!(gb, :dk, :k, :k; direction = Emitting())
    return validate(build(gb)), fx
end

# −log Z: z = (u, v, y) ~ N(μ, Σ); (P z + ε, C z) observed at (b_k, c), ε ~ N(0, Qs); w_k
function switch_nlogZ(fx, ps, k)
    nu, nv, ny = length(fx.μu), length(fx.μv), length(fx.μy)
    μ = vcat(fx.μu, fx.μv, fx.μy)
    Σ = cat(fx.Σu, fx.Σv, fx.Σy; dims = (1, 2))
    # the switching residual A_u u + A_y y − b_k = −ε
    P = hcat(ps.sw.A.u, zeros(size(ps.sw.A.u, 1), nv), ps.sw.A.y)
    C = hcat(ps.rel.A.u, ps.rel.A.v, zeros(length(ps.rel.c), ny))
    J = vcat(P, C)
    S = J * Σ * J'
    S[1:size(P, 1), 1:size(P, 1)] += fx.Qs
    S = Symmetric((S + S') / 2)
    d = vcat(ps.sw.b[:, k], ps.rel.c) - J * μ
    lw = ps.sw.logw .- log(sum(exp, ps.sw.logw))
    return (d' * (S \ d) + logdet(2π * S)) / 2 - lw[k]
end

# a random extended Gaussian on ℝⁿ: kf flat and kh hard directions, the rest Gaussian
function random_eg(rng, n; kf = rand(rng, 0:n), kh = rand(rng, 0:(n - kf)))
    B = Matrix(qr(randn(rng, n, n)).Q)
    V = B[:, (kf + kh + 1):n]
    Σ = V * spd(rng, size(V, 2)) * V'
    return ExtendedGaussian(randn(rng, n), (Σ + Σ') / 2, B[:, 1:kf])
end
# the same with variance ε along its hard directions, and their number
function softened(b::ExtendedGaussian, ε)
    H = hard_directions(b)
    Σ = b.Σ + ε * H * H'
    return ExtendedGaussian(b.μ, (Σ + Σ') / 2, b.D), size(H, 2)
end
softened(b, ε) = (b, 0)
end # module

@testset "P2: the envelope gradient and the diagonal replay through hard messages, against d(−log Z)" begin
    HI = HardInputs
    rng = Xoshiro(211)
    nhard = 0
    for trial in 1:12
        n, nu, nv, nw = rand(rng, 1:3), rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:2)
        m = rand(rng, 1:(nu + nv))
        r = copy(rng)
        g, p = HI.model(rng, n, nu, nv, nw, m)
        gd, _ = HI.model(r, n, nu, nv, nw, m; coupling = DiagonalCoupling())   # the same draws
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        ps = merge(ps, (g1 = (A = randn(rng, nu, n), b = randn(rng, nu)), g2 = (A = randn(rng, nw, nu), b = randn(rng, nw)),
                        rel = merge(ps.rel, (c = randn(rng, m),))))
        dθ = HI.dense_gradient(p, ps)
        _, rep, st1, s = infer!(g, tree_schedule(g), ps, st)
        @test rep.converged
        nhard += size(hard_directions(extended(marginal(s, g, g.variable_index[:u]), nu)), 2) > 0
        G = LenticulumCore.free_energy_gradient(s, g, ps, st1)
        @test HI.flat(G) ≈ dθ atol = 1e-7 rtol = 1e-7
        std = LuxCore.initialstates(Xoshiro(trial), gd)
        _, _, std, sd = infer!(gd, tree_schedule(gd), ps, std)
        Gd = LenticulumCore.free_energy_gradient(sd, gd, ps, std; schedule = tree_schedule(gd), nsweeps = 1)
        @test HI.flat(Gd) ≈ dθ atol = 1e-7 rtol = 1e-7
    end
    @test nhard >= 6
end

@testset "P2: a switching factor beside a hard relation, outcome observed: F = −log Z, and its gradient" begin
    HI = HardInputs
    rng = Xoshiro(307)
    nhard = 0
    for trial in 1:15
        nu, nv, ny = rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:2)
        m, ms, K = rand(rng, 1:(nu + nv)), rand(rng, 1:2), rand(rng, 2:3)
        k = rand(rng, 1:K)
        g, fx = HI.switch_model(rng, nu, nv, ny, m, ms, K, k)
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        ps = merge(ps, (sw = merge(ps.sw, (logw = randn(rng, K),)), rel = merge(ps.rel, (c = randn(rng, m),))))
        _, rep, st, s = infer!(g, tree_schedule(g), ps, st)
        @test rep.converged
        ins = Mycelium.messages_into(s, g, g.factor_index[:sw])
        nhard += ins.u isa ExtendedGaussian && size(hard_directions(ins.u), 2) > 0
        F = first(LenticulumCore.scalar_free_energy(s, g, ps, st))
        nlz = HI.switch_nlogZ(fx, ps, k)
        @test F ≈ nlz atol = 1e-7 * (1 + abs(nlz))
        # the envelope gradient by the switching factor's parameters, against the dense one
        G = LenticulumCore.free_energy_gradient(s, g, ps, st)
        θ = vcat(vec(ps.sw.A.u), vec(ps.sw.A.y), vec(ps.sw.b), ps.sw.logw)
        function f(θ)
            i = 0
            take(n) = (r = θ[(i + 1):(i + n)]; i += n; r)
            Au = reshape(take(length(ps.sw.A.u)), size(ps.sw.A.u)); Ay = reshape(take(length(ps.sw.A.y)), size(ps.sw.A.y))
            b = reshape(take(length(ps.sw.b)), size(ps.sw.b)); lw = take(K)
            return HI.switch_nlogZ(fx, merge(ps, (sw = (A = (u = Au, y = Ay), b, logw = lw),)), k)
        end
        @test vcat(vec(G.sw.A.u), vec(G.sw.A.y), vec(G.sw.b), G.sw.logw) ≈ ForwardDiff.gradient(f, θ) atol = 1e-7 rtol = 1e-7
    end
    @test nhard >= 8          # hard messages reached the switching factor in 11 of 15 (seed 307)
end

@testset "P2: the switching factor's term with hard inputs is the limit of its softened term" begin
    HI = HardInputs
    rng = Xoshiro(419)
    ε = 1e-9
    nh = 0
    for trial in 1:60
        nu, ny, m, K = rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:3), rand(rng, 1:3)
        f = SwitchingConstraintFactor((u = nu, y = ny), K, m; noise = HI.spd(rng, m))
        ps = merge(LuxCore.initialparameters(Xoshiro(trial), f), (logw = randn(rng, K),))
        u = HI.random_eg(rng, nu)
        y = rand(rng) < 0.3 ? DiracBelief(randn(rng, ny)) : HI.random_eg(rng, ny; kf = 0)
        kmsg = rand(rng) < 0.5 ? nothing : CategoricalBelief(normalize(rand(rng, K) .+ 0.1, 1))
        msgs = (u = u, y = y, k = kmsg)
        # the factor belief must be proper: no flat direction may be left unconstrained
        F = try
            Mycelium.local_free_energy(f, msgs, ps, NamedTuple())[1]
        catch e
            e isa Union{ArgumentError,LinearAlgebra.PosDefException} || rethrow()
            continue
        end
        (us, hu), (ys, hy) = HI.softened(u, ε), HI.softened(y, ε)
        h = hu + hy
        h == 0 && continue
        nh += 1
        Fε = Mycelium.local_free_energy(f, (u = us, y = ys, k = kmsg), ps, NamedTuple())[1]
        Fs, Fεs = LenticulumCore.scalarise(LenticulumCore.scalarisation(f), F), LenticulumCore.scalarise(LenticulumCore.scalarisation(f), Fε)
        @test Fs ≈ Fεs + h / 2 * log(2π * ℯ * ε) atol = 1e-5 * (1 + abs(Fs))
        @test F[:fit] ≈ Fε[:fit] atol = 1e-5 * (1 + abs(F[:fit]))
        # LP-020: the frozen belief gives the same term
        b = Mycelium.factor_belief(f, msgs, ps, NamedTuple())
        Fb = Mycelium.belief_free_energy(f, b, ps, NamedTuple())[1]
        for c in (:fit, :complexity, :negentropy)
            @test Fb[c] ≈ F[c] atol = 1e-9 * (1 + abs(F[c]))
        end
        # with one outcome it is LinearConstraintFactor's term
        if K == 1
            l = LinearConstraintFactor((u = nu, y = ny), m; noise = f.Q)
            Fl = Mycelium.local_free_energy(l, (u = u, y = y), (A = ps.A, c = ps.b[:, 1]), NamedTuple())[1]
            for c in (:fit, :complexity, :negentropy)
                @test Fl[c] ≈ F[c] atol = 1e-9 * (1 + abs(F[c]))
            end
        end
    end
    @test nh >= 25
end

@testset "P2: frozen beliefs with hard inputs agree with the local free energy (LP-020)" begin
    HI = HardInputs
    rng = Xoshiro(521)
    checked = 0
    agree(f, msgs, ps) = begin
        F = Mycelium.local_free_energy(f, msgs, ps, NamedTuple())[1]
        Fb = Mycelium.belief_free_energy(f, Mycelium.factor_belief(f, msgs, ps, NamedTuple()), ps, NamedTuple())[1]
        all(c -> isapprox(Fb[c], F[c]; atol = 1e-9 * (1 + abs(F[c]))), (:fit, :complexity, :negentropy))
    end
    for trial in 1:40
        nx, ny, m = rand(rng, 1:3), rand(rng, 1:3), rand(rng, 1:3)
        try
            gf = GaussianFactor(nx => ny; noise = HI.spd(rng, ny))
            pg = LuxCore.initialparameters(Xoshiro(trial), gf)
            @test agree(gf, (x = HI.random_eg(rng, nx; kf = 0), y = HI.random_eg(rng, ny; kf = 0)), pg)
            gp = GaussianPrior(:x, randn(rng, nx), HI.spd(rng, nx))
            @test agree(gp, (x = HI.random_eg(rng, nx),), NamedTuple())
            lc = LinearConstraintFactor((a = nx, b = ny), m; noise = HI.spd(rng, m))
            pl = LuxCore.initialparameters(Xoshiro(trial), lc)
            @test agree(lc, (a = HI.random_eg(rng, nx; kf = 0), b = HI.random_eg(rng, ny; kf = 0)), pl)
            checked += 1
        catch e
            e isa Union{ArgumentError,LinearAlgebra.PosDefException} || rethrow()
        end
    end
    @test checked >= 30
    # a learned noise still refuses hard messages, in the frozen belief as in the local term
    gf = GaussianFactor(2 => 2; noise = 0.5, learn_noise = true)
    pg = LuxCore.initialparameters(Xoshiro(1), gf)
    hard = ExtendedGaussian([0.3, -0.2], [0.5 0.5; 0.5 0.5], zeros(2, 0))
    msgs = (x = Gaussian(zeros(2), Matrix(1.0I, 2, 2)), y = hard)
    @test_throws ArgumentError Mycelium.local_free_energy(gf, msgs, pg, NamedTuple())
    @test_throws ArgumentError Mycelium.factor_belief(gf, msgs, pg, NamedTuple())
end

@testset "P2: a plate with hard relations: free energy and envelope gradient are the sums over replicas" begin
    HI = HardInputs
    rng = Xoshiro(613)
    R = 4
    # replica i: w observed through obs; a hard relation on (u, v); lin links u and y
    function tpl!(p, wi, d, fx)
        nw, nu, nv, ny, m = d
        variable!(p, :w, nw); variable!(p, :u, nu); variable!(p, :v, nv); variable!(p, :y, ny)
        factor!(p, :data, DataFactor(:x, wi)); connect!(p, :data, :x, :w; direction = Emitting())
        factor!(p, :pu, GaussianPrior(:x, fx.μu, fx.Σu)); connect!(p, :pu, :x, :u)
        factor!(p, :obs, GaussianFactor(nu => nw; noise = fx.Qw)); connect!(p, :obs, :x, :u)
        connect!(p, :obs, :y, :w; direction = Absorbing())
        factor!(p, :pv, GaussianPrior(:x, fx.μv, fx.Σv)); connect!(p, :pv, :x, :v)
        factor!(p, :rel, HardConstraintFactor((u = nu, v = nv), m)); connect!(p, :rel, :u, :u); connect!(p, :rel, :v, :v)
        factor!(p, :py, GaussianPrior(:x, fx.μy, fx.Σy)); connect!(p, :py, :x, :y)
        factor!(p, :lin, LinearConstraintFactor((u = nu, y = ny), ny; noise = fx.Qy)); connect!(p, :lin, :u, :u); connect!(p, :lin, :y, :y)
    end
    function plate(W, d, fx)
        b = GraphBuilder()
        plate!((p, i) -> tpl!(p, W[:, i], d, fx), b, :r, size(W, 2))
        return validate(build(b))
    end
    for trial in 1:6
        nw, nu, nv, ny = rand(rng, 1:2), rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:2)
        d = (nw, nu, nv, ny, rand(rng, 1:(nu + nv)))
        fx = (μu = randn(rng, nu), Σu = HI.spd(rng, nu), Qw = HI.spd(rng, nw), μv = randn(rng, nv), Σv = HI.spd(rng, nv),
              μy = randn(rng, ny), Σy = HI.spd(rng, ny), Qy = HI.spd(rng, ny))
        W = randn(rng, nw, R)
        g = plate(W, d, fx)
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        ps = merge(ps, (rel = merge(ps.rel, (c = randn(rng, d[5]),)),))
        _, rep, st, s = infer!(g, plate_schedule(g), ps, st; maxsweeps = 30)
        @test rep.converged
        F = first(LenticulumCore.scalar_free_energy(s, g, ps, st))
        G = LenticulumCore.free_energy_gradient(s, g, ps, st)
        Fsum, Gsum = 0.0, nothing
        for i in 1:R
            gi = plate(W[:, i:i], d, fx)
            sti = LuxCore.initialstates(Xoshiro(trial), gi)
            _, _, sti, si = infer!(gi, tree_schedule(gi), ps, sti)
            Fsum += first(LenticulumCore.scalar_free_energy(si, gi, ps, sti))
            Gi = LenticulumCore.free_energy_gradient(si, gi, ps, sti)
            Gsum = Gsum === nothing ? LenticulumCore._flatten(Gi) : Gsum .+ LenticulumCore._flatten(Gi)
        end
        @test F ≈ Fsum atol = 1e-7 * (1 + abs(Fsum))
        @test LenticulumCore._flatten(G) ≈ Gsum atol = 1e-7 rtol = 1e-7
    end
end

@testset "P2: Zygote through the switching term with hard inputs, against ForwardDiff" begin
    HI = HardInputs
    rng = Xoshiro(5)
    n = 0
    for t in 1:40
        nu, ny, m, K = rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:3), rand(rng, 2:3)
        f = SwitchingConstraintFactor((u = nu, y = ny), K, m; noise = HI.spd(rng, m))
        ps0 = LuxCore.initialparameters(Xoshiro(t), f)
        u = HI.random_eg(rng, nu; kf = 0)
        size(hard_directions(u), 2) == 0 && continue
        y = rand(rng) < 0.3 ? DiracBelief(randn(rng, ny)) : HI.random_eg(rng, ny; kf = 0)
        msgs = (u = u, y = y, k = CategoricalBelief(normalize(rand(rng, K) .+ 0.1, 1)))
        θ = vcat(vec(ps0.A.u), vec(ps0.A.y), vec(ps0.b), ps0.logw)
        unpack(θ) = (A = (u = reshape(θ[1:(nu * m)], m, nu), y = reshape(θ[(nu * m + 1):((nu + ny) * m)], m, ny)),
                     b = reshape(θ[((nu + ny) * m + 1):((nu + ny + K) * m)], m, K), logw = θ[(end - K + 1):end])
        F(θ) = (E = first(Mycelium.local_free_energy(f, msgs, unpack(θ), NamedTuple())); E[:fit] + E[:complexity] + E[:negentropy])
        @test Zygote.gradient(F, θ)[1] ≈ ForwardDiff.gradient(F, θ) atol = 1e-10 rtol = 1e-10
        n += 1
    end
    @test n >= 20
end
