# How the cost of a sweep and of the free energy grows with the number of factors, on a
# position-velocity Kalman chain (three factors per step). LP-024 measured the free energy at
# 7.7, 25.8, 102 ms for 601, 1 201, 2 401 factors: quadratic, from rebuilding the graph-wide state
# `NamedTuple` once per factor (`_setsubtree`). Recorded in `lib/Mycelium.jl/src/passing.md`,
# "The state tree" and "The trees by position" (I5, which also added the gradient columns).
# Run:  julia --project=bench bench/state_scaling.jl
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Random, LuxCore, Printf

function chain(T; nx = 2, H = [1.0 0.0; 0.0 1.0], A = [1.0 1.0; 0.0 1.0])
    b = GraphBuilder()
    for t in 0:T; variable!(b, Symbol(:x, t), nx); end
    factor!(b, :prior, GaussianPrior(:x, zeros(nx), Matrix(1.0I, nx, nx))); connect!(b, :prior, :x, :x0)
    for t in 1:T
        factor!(b, Symbol(:f, t), GaussianFactor(nx => nx; noise = 0.1, channels = (:a, :b)))
        connect!(b, Symbol(:f, t), :a, Symbol(:x, t - 1)); connect!(b, Symbol(:f, t), :b, Symbol(:x, t))
        variable!(b, Symbol(:y, t), size(H, 1))
        factor!(b, Symbol(:h, t), GaussianFactor(nx => size(H, 1); noise = 0.5, channels = (:a, :b)))
        connect!(b, Symbol(:h, t), :a, Symbol(:x, t)); connect!(b, Symbol(:h, t), :b, Symbol(:y, t))
        factor!(b, Symbol(:d, t), DataFactor(:v, fill(0.3t, size(H, 1))))
        connect!(b, Symbol(:d, t), :v, Symbol(:y, t); direction = Emitting())
    end
    g = build(b)
    rng = Xoshiro(1)
    ps = LuxCore.initialparameters(rng, g); st = LuxCore.initialstates(rng, g)
    kv = Dict{Symbol,Any}(pairs(ps))
    for t in 1:T
        kv[Symbol(:f, t)] = (A = A, b = zeros(nx))
        kv[Symbol(:h, t)] = (A = H, b = zeros(size(H, 1)))
    end
    ks = keys(ps)
    return g, NamedTuple{ks}(Tuple(kv[k] for k in ks)), st
end

best(f; n = 5) = minimum(_ -> (GC.gc(); @elapsed f()), 1:n)

@printf("%6s %8s %8s %16s %12s %10s %14s %16s\n", "T", "factors", "tasks", "sweep/task (μs)", "sweep (ms)", "F (ms)",
        "gradient (ms)", "param step (ms)")
for T in (100, 200, 400, 800, 1600)
    g, ps, st = chain(T)
    sched = SequentialSchedule(tasks(tree_schedule(g)))
    s = MessageStore(g)
    sweep!(s, g, sched, ps, st)                                      # compile, fill the store
    LenticulumCore.scalar_free_energy(s, g, ps, st)
    nt = length(tasks(sched))
    ts = best(() -> sweep!(s, g, sched, ps, st))
    tf = best(() -> LenticulumCore.scalar_free_energy(s, g, ps, st))
    LenticulumCore.free_energy_gradient(s, g, ps, st); MyceliumLearning.parameter_step(s, g, ps, st)
    tg = best(() -> LenticulumCore.free_energy_gradient(s, g, ps, st); n = 3)
    tp = best(() -> MyceliumLearning.parameter_step(s, g, ps, st); n = 3)
    @printf("%6d %8d %8d %16.2f %12.2f %10.2f %14.2f %16.2f\n", T, length(g.factors), nt, ts / nt * 1e6,
            ts * 1e3, tf * 1e3, tg * 1e3, tp * 1e3)
end
