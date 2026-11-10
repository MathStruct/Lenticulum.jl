# The E-step of factor analysis through Mycelium, one graph per point against one plate.
# Recorded in `lib/Mycelium.jl/src/plates.md` ("Speed") and L-SPEC-006.
# Run:  julia --project=bench bench/plates.jl
#
# Model: z ~ N(0, I_k), x = W z + c + ε, ε ~ N(0, σ² I_d); the E-step is the posterior of every
# z_i. Five ways: a graph built and run per point (as `fit_hybrid` did on `experiment-qm`), the
# same graphs built once, one plate on a sequential schedule, the plate on its grouped schedule
# (`plate_schedule`), and the closed form vectorised by hand: P = I + WᵀΣ⁻¹W,
# M = P⁻¹WᵀΣ⁻¹(X − c). Each timing is the minimum over five runs after a warm-up, and times the
# message passing only (a fresh store and one sweep); reading the posterior means out of the
# stores is checked against the closed form outside the timings.
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Printf
using MyceliumLearning

const D, K, σ2 = 3, 2, 0.7

function template!(p, x)
    variable!(p, :z, K); variable!(p, :x, D)
    factor!(p, :prior, GaussianPrior(:z, zeros(K), Matrix(1.0I, K, K)))
    factor!(p, :lik, LinearConstraintFactor((x = D, z = K), D; noise = σ2))
    factor!(p, :data, DataFactor(:x, x))
    connect!(p, :prior, :z, :z); connect!(p, :lik, :z, :z)
    connect!(p, :lik, :x, :x; direction = Absorbing()); connect!(p, :data, :x, :x; direction = Emitting())
end
function plate_graph(X)
    b = GraphBuilder()
    plate!((p, i) -> template!(p, X[:, i]), b, :obs, size(X, 2))
    return build(b)
end
setparams(g, W, c) = (ps = LuxCore.initialparameters(Xoshiro(0), g);
    merge(ps, (lik = merge(ps.lik, (A = (x = Matrix(1.0I, D, D), z = -W), c = c)),)))
# the posterior means of every z, read through the plate record's ids (outside the timings)
zmeans(s, g) = (pl = only(plates(g)); reduce(hcat, [belief_mean(marginal(s, g, pl.variables[1, i])) for i in 1:pl.n]))

function loop_built(X, W, c)
    map(axes(X, 2)) do i
        g = plate_graph(X[:, i:i])
        s = MessageStore(g)
        sweep!(s, g, tree_schedule(g), setparams(g, W, c), NamedTuple())
        s
    end
end
function loop_prebuilt(gs, pss, scheds)
    map(eachindex(gs)) do i
        s = MessageStore(gs[i])
        sweep!(s, gs[i], scheds[i], pss[i], NamedTuple())
        s
    end
end
function plate_run(g, ps, sched)
    s = MessageStore(g)
    sweep!(s, g, sched, ps, NamedTuple())
    s
end
function vectorised(X, W, c)
    Wt = W' / σ2
    return (I + Wt * W) \ (Wt * (X .- c))
end

best(f) = (f(); minimum(@elapsed(f()) for _ in 1:5))

@printf("%6s %12s %12s %12s %12s %12s %10s\n", "N", "loop+build", "loop", "plate seq", "plate grp", "by hand", "max |Δμ|")
for N in (100, 300, 1000)
    rng = Xoshiro(N)
    X = randn(rng, D, N); W = randn(rng, D, K); c = randn(rng, D)
    gs = [plate_graph(X[:, i:i]) for i in 1:N]
    pss = [setparams(g, W, c) for g in gs]; scheds = [tree_schedule(g) for g in gs]
    g = plate_graph(X); ps = setparams(g, W, c)
    grouped = plate_schedule(g); seq = SequentialSchedule(tasks(grouped))
    M = vectorised(X, W, c)
    Δ = max(maximum(abs, zmeans(plate_run(g, ps, grouped), g) .- M), maximum(abs, zmeans(plate_run(g, ps, seq), g) .- M))
    @assert maximum(abs, reduce(hcat, [zmeans(s, gi) for (s, gi) in zip(loop_prebuilt(gs, pss, scheds), gs)]) .- M) < 1e-10
    @printf("%6d %10.2f ms %10.2f ms %10.2f ms %10.2f ms %10.4f ms %10.1e\n", N,
            1e3best(() -> loop_built(X, W, c)), 1e3best(() -> loop_prebuilt(gs, pss, scheds)),
            1e3best(() -> plate_run(g, ps, seq)), 1e3best(() -> plate_run(g, ps, grouped)),
            1e3best(() -> vectorised(X, W, c)), Δ)
end

# Large plates: building the graph, the first (compiling) and a later call of the scalar free
# energy, after one grouped sweep. The first N in the loop pays the method's compilation.
println()
@printf("%6s %10s %16s %16s\n", "N", "build", "F, first call", "F, later call")
for N in (100, 1000, 3000)
    rng = Xoshiro(N)
    X = randn(rng, D, N); W = randn(rng, D, K); c = randn(rng, D)
    tb = @elapsed (g = plate_graph(X))
    ps = setparams(g, W, c)
    s = plate_run(g, ps, plate_schedule(g))
    t1 = @elapsed LenticulumCore.scalar_free_energy(s, g, ps, NamedTuple())
    t2 = minimum(@elapsed(LenticulumCore.scalar_free_energy(s, g, ps, NamedTuple())) for _ in 1:3)
    @printf("%6d %8.2f s %14.3f s %13.1f ms\n", N, tb, t1, 1e3t2)
end

# Energies on message blocks (L-ADR-0010 step 2): the scalar free energy and its parameter gradient
# (reference differences, the default backend) after one grouped sweep, with the messages held as
# blocks against the same messages held per edge (a store built from the edge vectors).
println()
@printf("%6s %14s %14s %16s %16s\n", "N", "F, blocks", "F, per edge", "∇F, blocks", "∇F, per edge")
for N in (100, 1000, 3000)
    rng = Xoshiro(N)
    X = randn(rng, D, N); W = randn(rng, D, K); c = randn(rng, D)
    g = plate_graph(X); ps = setparams(g, W, c)
    s = plate_run(g, ps, plate_schedule(g))
    se = MessageStore(copy(s.to_variable), copy(s.to_factor), s.iteration)
    F(st) = first(LenticulumCore.scalar_free_energy(st, g, ps, NamedTuple()))
    @assert abs(F(s) - F(se)) <= 1e-10 * abs(F(se))
    tF(st) = (F(st); minimum(@elapsed(F(st)) for _ in 1:3))
    tG(st) = (LenticulumCore.free_energy_gradient(st, g, ps, NamedTuple());
              minimum(@elapsed(LenticulumCore.free_energy_gradient(st, g, ps, NamedTuple())) for _ in 1:3))
    @printf("%6d %11.2f ms %11.2f ms %13.2f ms %13.2f ms\n", N, 1e3tF(s), 1e3tF(se), 1e3tG(s), 1e3tG(se))
end
