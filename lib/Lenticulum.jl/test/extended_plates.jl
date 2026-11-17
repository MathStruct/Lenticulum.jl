# P4 (Feature Backlog), plates: extended-Gaussian messages inside a plate. A hard relation
# (`HardConstraintFactor`) sends extended Gaussians, with hard and flat directions, to the
# relation factors beside it; in a plate they arrive as `VectorBlock`s, which the block methods
# hand to the per-replica path (`_block_parts` is `nothing` for them; `gaussian.md`, "Extended
# Gaussian inputs").
#
# Oracles: the dense Gaussian of one replica conditioned on its observation and on the relation
# (for a plate of GaussianFactors), the same tasks sent one by one (`SequentialSchedule`), and each
# replica as its own graph (for LinearConstraintFactor's block method). Known limits are
# `@test_broken`. A mixture message meeting an extended one was one until P27 (`mixture_extended.jl`
# has its oracle). The coupled replay through a hard
# relation under ForwardDiff and Zygote, broken until P4's AD half, is tested here against
# reference differences and in `hard_gradient.jl` against the dense −log Z.

using Test, Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, LuxCore, Random
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

module ExtendedPlates
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# Replica i: x ~ N(μ0, Σ0); w = H x + h + ε_w, observed (w = W[:, i]); u = A x + b + ε
# (GaussianFactor); v ~ N(μv, Σv); Cu u + Cv v = c exactly. Every parameter is tied.
function gauss_template!(p, wi, d, fx)
    n, nw, nu, nv, m = d
    variable!(p, :x, n); variable!(p, :w, nw); variable!(p, :u, nu); variable!(p, :v, nv)
    factor!(p, :px, GaussianPrior(:x, fx.μ0, fx.Σ0)); connect!(p, :px, :x, :x)
    factor!(p, :pv, GaussianPrior(:x, fx.μv, fx.Σv)); connect!(p, :pv, :x, :v)
    factor!(p, :obs, GaussianFactor(n => nw; noise = fx.Qw)); connect!(p, :obs, :x, :x)
    connect!(p, :obs, :y, :w; direction = Absorbing())
    factor!(p, :data, DataFactor(:x, wi)); connect!(p, :data, :x, :w; direction = Emitting())
    factor!(p, :g, GaussianFactor(n => nu; noise = fx.Q)); connect!(p, :g, :x, :x); connect!(p, :g, :y, :u)
    factor!(p, :rel, HardConstraintFactor((u = nu, v = nv), m)); connect!(p, :rel, :u, :u); connect!(p, :rel, :v, :v)
end

# the dense joint of (x, w, u, v) of one replica, conditioned on w = wi and on the relation
function gauss_oracle(ps, wi, d, fx)
    n, nw, nu, nv, m = d
    H, A = ps.obs.A, ps.g.A
    L = vcat(Matrix(1.0I, n, n), H, A)
    μ = vcat(fx.μ0, H * fx.μ0 + ps.obs.b, A * fx.μ0 + ps.g.b, fx.μv)
    N = n + nw + nu + nv
    iw, iu, iv = (n + 1):(n + nw), (n + nw + 1):(n + nw + nu), (n + nw + nu + 1):N
    Σ = zeros(N, N)
    Σ[1:(n + nw + nu), 1:(n + nw + nu)] = L * fx.Σ0 * L'
    Σ[iw, iw] += fx.Qw
    Σ[iu, iu] += fx.Q
    Σ[iv, iv] = fx.Σv
    C = zeros(nw + m, N)
    C[1:nw, iw] = I(nw)
    C[(nw + 1):end, iu] = ps.rel.A.u
    C[(nw + 1):end, iv] = ps.rel.A.v
    K = Σ * C' * pinv(C * Σ * C')
    μp = μ - K * (C * μ - vcat(wi, ps.rel.c))
    return μp, Σ - K * C * Σ, (x = 1:n, u = iu, v = iv)
end

# Replica i: w observed; obs links w and u; v ~ prior; a hard relation on (u, v); lin links u and
# y (y ~ prior); with `switching`, a SwitchingConstraintFactor on u.
function lin_template!(p, wi, d, fx)
    nw, nu, nv, ny, m = d
    variable!(p, :w, nw); variable!(p, :u, nu); variable!(p, :v, nv); variable!(p, :y, ny)
    factor!(p, :data, DataFactor(:x, wi)); connect!(p, :data, :x, :w; direction = Emitting())
    factor!(p, :obs, LinearConstraintFactor((w = nw, u = nu), nw; noise = fx.Qw))
    connect!(p, :obs, :w, :w; direction = Absorbing()); connect!(p, :obs, :u, :u)
    factor!(p, :pv, GaussianPrior(:x, fx.μv, fx.Σv)); connect!(p, :pv, :x, :v)
    factor!(p, :rel, HardConstraintFactor((u = nu, v = nv), m)); connect!(p, :rel, :u, :u); connect!(p, :rel, :v, :v)
    factor!(p, :py, GaussianPrior(:x, fx.μy, fx.Σy)); connect!(p, :py, :x, :y)
    factor!(p, :lin, LinearConstraintFactor((u = nu, y = ny), ny; noise = fx.Qy)); connect!(p, :lin, :u, :u); connect!(p, :lin, :y, :y)
    if fx.switching
        variable!(p, :k, 2)
        factor!(p, :sw, SwitchingConstraintFactor((u = nu,), 2, nu; noise = fx.Qs)); connect!(p, :sw, :u, :u); connect!(p, :sw, :k, :k)
    end
end

function plate_graph(tpl!, W, d, fx)
    b = GraphBuilder()
    plate!((p, i) -> tpl!(p, W[:, i], d, fx), b, :r, size(W, 2))
    return validate(build(b))
end

msg(m) = m === nothing ? nothing : m.belief
function dist(a, b)
    (a === nothing) != (b === nothing) && return Inf
    return a === nothing ? 0.0 : belief_distance(a, b)
end
# the extended Gaussians among the messages sent to channel `ch` of the factors named `name[i]`
function extended_inputs(s, g, name, ch)
    k = 0
    for (i, e) in enumerate(g.edges)
        e.channel === ch && startswith(string(g.factors[e.factor].name), string(name, "[")) || continue
        (b = msg(s.to_factor[i])) === nothing && continue
        bs = b isa Mycelium.BeliefBlock ? [Mycelium.replica(b, j) for j in 1:Mycelium.nreplicas(b)] : [b]
        k += count(x -> x isa ExtendedGaussian && size(x.D, 2) < length(x.μ), bs)
    end
    return k
end
end # module

@testset "P4: a plate of GaussianFactors under hard relations, against the conditioned Gaussian" begin
    XP = ExtendedPlates
    rng = Xoshiro(11)
    R, reached = 4, 0
    for trial in 1:10
        n, nw, nu, nv = rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:3), rand(rng, 1:2)
        d = (n, nw, nu, nv, rand(rng, 1:(nu + nv)))
        fx = (μ0 = randn(rng, n), Σ0 = XP.spd(rng, n), μv = randn(rng, nv), Σv = XP.spd(rng, nv),
              Q = XP.spd(rng, nu), Qw = XP.spd(rng, nw))
        W = randn(rng, nw, R)
        g = XP.plate_graph(XP.gauss_template!, W, d, fx)
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        s = MessageStore(g)
        _, rep = propagate!(s, g, plate_schedule(g), ps, st; maxsweeps = 20)
        reached += XP.extended_inputs(s, g, :g, :y)
        for i in 1:R
            μp, Σp, idx = XP.gauss_oracle(ps, W[:, i], d, fx)
            for v in (:x, :u, :v)
                r = idx[v]
                e = extended(marginal(s, g, g.variable_index[replica_name(v, i)]), length(r))
                @test size(e.D, 2) == 0
                @test e.μ ≈ μp[r] atol = 1e-7 rtol = 1e-9
                @test e.Σ ≈ Σp[r, r] atol = 1e-7 rtol = 1e-9
            end
        end
    end
    @test reached > 0   # extended messages reached the GaussianFactors' :y ports
end

@testset "P4: extended inputs into LinearConstraintFactor's block method" begin
    XP = ExtendedPlates
    rng = Xoshiro(23)
    R = 5
    for trial in 1:8
        nw, nu, nv, ny = rand(rng, 1:2), rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:2)
        d = (nw, nu, nv, ny, rand(rng, 1:(nu + nv)))
        fx = (Qw = XP.spd(rng, nw), μv = randn(rng, nv), Σv = XP.spd(rng, nv), μy = randn(rng, ny),
              Σy = XP.spd(rng, ny), Qy = XP.spd(rng, ny), Qs = XP.spd(rng, nu), switching = false)
        W = randn(rng, nw, R)
        g = XP.plate_graph(XP.lin_template!, W, d, fx)
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        sched = plate_schedule(g)
        s1, s2 = MessageStore(g), MessageStore(g)
        for _ in 1:15
            sweep!(s1, g, sched, ps, st); sweep!(s2, g, SequentialSchedule(tasks(sched)), ps, st)
        end
        @test XP.extended_inputs(s1, g, :lin, :u) == R
        for i in 1:nedges(g)
            @test XP.dist(XP.msg(s1.to_variable[i]), XP.msg(s2.to_variable[i])) < 1e-8
        end
        for i in 1:R
            gi = XP.plate_graph(XP.lin_template!, W[:, i:i], d, fx)
            psi = LuxCore.initialparameters(Xoshiro(trial), gi)
            psi = NamedTuple{keys(psi)}(map(k -> getfield(ps, k), keys(psi)))
            sti = LuxCore.initialstates(Xoshiro(trial), gi)
            si = MessageStore(gi)
            for _ in 1:15
                sweep!(si, gi, SequentialSchedule(tasks(plate_schedule(gi))), psi, sti)
            end
            for v in (:u, :v, :y)
                @test XP.dist(marginal(s1, g, g.variable_index[replica_name(v, i)]),
                              marginal(si, gi, gi.variable_index[replica_name(v, 1)])) < 1e-10
            end
        end
    end
end

@testset "P4: known limits (broken), and the replay through a hard relation" begin
    XP = ExtendedPlates
    rng = Xoshiro(5)
    # a switching factor's mixture message meets the hard relation's extended one on u: broken until
    # P27 (Mycelium's mixture product had no rule for an extended component); the values are checked
    # against enumeration in `mixture_extended.jl`
    d = (1, 2, 1, 1, 2)
    fx = (Qw = XP.spd(rng, 1), μv = randn(rng, 1), Σv = XP.spd(rng, 1), μy = randn(rng, 1),
          Σy = XP.spd(rng, 1), Qy = XP.spd(rng, 1), Qs = XP.spd(rng, 2), switching = true)
    g = XP.plate_graph(XP.lin_template!, randn(rng, 1, 2), d, fx)
    ps = LuxCore.initialparameters(Xoshiro(1), g); st = LuxCore.initialstates(Xoshiro(1), g)
    mixed = try
        s = MessageStore(g); sweep!(s, g, plate_schedule(g), ps, st); sweep!(s, g, plate_schedule(g), ps, st); true
    catch e
        e isa ArgumentError || rethrow()
        false
    end
    @test mixed
    # the coupled replay through a hard relation: broken before P4's AD half (ExtendedGaussian
    # held only floats under ForwardDiff, and its join took no tuple under Zygote)
    gb = GraphBuilder()
    variable!(gb, :x, 2); variable!(gb, :u, 2); variable!(gb, :v, 1)
    c = ExactCoupling()
    factor!(gb, :px, GaussianPrior(:x, zeros(2), Matrix(1.0I, 2, 2))); connect!(gb, :px, :x, :x; coupling = c)
    factor!(gb, :pv, GaussianPrior(:x, zeros(1), Matrix(1.0I, 1, 1))); connect!(gb, :pv, :x, :v; coupling = c)
    factor!(gb, :g, GaussianFactor(2 => 2; noise = 0.5)); connect!(gb, :g, :x, :x; coupling = c); connect!(gb, :g, :y, :u; coupling = c)
    factor!(gb, :rel, HardConstraintFactor((u = 2, v = 1), 2)); connect!(gb, :rel, :u, :u; coupling = c); connect!(gb, :rel, :v, :v; coupling = c)
    g = validate(build(gb))
    ps = LuxCore.initialparameters(Xoshiro(2), g); st = LuxCore.initialstates(Xoshiro(2), g)
    _, _, st, s = infer!(g, tree_schedule(g), ps, st)
    Gref = LenticulumCore.free_energy_gradient(s, g, ps, st; schedule = tree_schedule(g), nsweeps = 1)
    for ad in (AutoForwardDiff(), AutoZygote())
        G = LenticulumCore.free_energy_gradient(s, g, ps, st; ad, schedule = tree_schedule(g), nsweeps = 1)
        @test LenticulumCore._flatten(G.g) ≈ LenticulumCore._flatten(Gref.g) atol = 1e-6
    end
end
