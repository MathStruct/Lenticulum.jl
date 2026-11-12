# How the cost of a sweep and of the free energy grows with the number of factors, on a
# position-velocity Kalman chain (three factors per step). LP-024 measured the free energy at
# 7.7, 25.8, 102 ms for 601, 1 201, 2 401 factors: quadratic, from rebuilding the graph-wide state
# `NamedTuple` once per factor (`_setsubtree`). Recorded in `lib/Mycelium.jl/src/passing.md`,
# "The state tree" and "The trees by position" (I5, which also added the gradient columns).
# Run:  julia --project=bench bench/state_scaling.jl
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Random, LuxCore, Printf

include(joinpath(@__DIR__, "kalman_chain.jl"))

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
