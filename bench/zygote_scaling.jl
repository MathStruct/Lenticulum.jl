# How the cost of Zygote's gradient of the free energy in `ps`, at fixed messages, grows with the
# number of factors, on the position-velocity Kalman chain of `state_scaling.jl` (three factors per
# step, one parameter key per factor). Before I16 every node's read of `ps` and of the message store
# gave a cotangent as wide as the tree or the store: O(factors × keys). Recorded in
# `lib/Mycelium.jl/src/passing.md`, "The backward pass by key".
# Run:  julia --project=bench bench/zygote_scaling.jl          (TS="20 40" to choose the steps)
# Smoke: LENTICULUM_SMOKE=true (bench/smoke.jl) runs it with 2–4 chain steps and 1 timed run.
#
# Zygote compiles its pullback once per type of `ps`, a `NamedTuple` with one field per factor, so
# every size compiles anew; the "first call" column is that compilation and grows with the keys.

@isdefined(provenance) || include(joinpath(@__DIR__, "provenance.jl"))
abspath(PROGRAM_FILE) == (@__FILE__) && provenance()

@isdefined(SMOKE) || const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Random, LuxCore, Printf, Zygote

include(joinpath(@__DIR__, "kalman_chain.jl"))

best(f; n = (SMOKE ? 1 : 3)) = minimum(_ -> (GC.gc(); @elapsed f()), 1:n)
steps = SMOKE ? (2, 4) : Tuple(parse.(Int, split(get(ENV, "TS", "20 40 80 160 320"))))

@printf("%6s %8s %10s %16s %16s %14s\n", "T", "factors", "F (ms)", "first call (s)", "gradient (ms)", "allocated (MB)")
for T in steps
    g, ps, st = chain(T)
    s = MessageStore(g)
    sweep!(s, g, SequentialSchedule(tasks(tree_schedule(g))), ps, st)
    F(p) = first(LenticulumCore.scalar_free_energy(s, g, p, st))
    tf = best(() -> F(ps))
    tc = @elapsed Zygote.gradient(F, ps)
    tz = best(() -> Zygote.gradient(F, ps))
    bytes = @allocated Zygote.gradient(F, ps)
    @printf("%6d %8d %10.2f %16.1f %16.2f %14.1f\n", T, length(g.factors), tf * 1e3, tc, tz * 1e3, bytes / 2^20)
    flush(stdout)
end
