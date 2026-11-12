# Where a sequential sweep spends its time, on `state_scaling.jl`'s Kalman chain. Prints the sweep
# time (best of 5) and the allocations, profiles 50 sweeps (one sample every 0.1 ms) and writes a
# flat and a tree profile to `~/.cache/lenticulum/sweep_profile/` (or `$OUT`). Recorded in
# `lib/Mycelium.jl/src/passing.md`, "Where a sweep spends its time after I5".
# Run:  T=800 julia --project=bench bench/sweep_profile.jl
include(joinpath(@__DIR__, "kalman_chain.jl"))
using Profile

T = parse(Int, get(ENV, "T", "800"))
out = get(ENV, "OUT", joinpath(homedir(), ".cache", "lenticulum", "sweep_profile")); mkpath(out)
g, ps, st = chain(T)
sched = SequentialSchedule(tasks(tree_schedule(g)))
s = MessageStore(g)
sweep!(s, g, sched, ps, st)                                          # compile, fill the store
nt = length(tasks(sched))
ts = minimum(_ -> (GC.gc(); @elapsed sweep!(s, g, sched, ps, st)), 1:5)
println("T = $T, factors = $(length(g.factors)), tasks = $nt: sweep $(round(ts * 1e3; digits = 2)) ms, ",
        "$(round(ts / nt * 1e6; digits = 2)) μs per task")
a = @allocated sweep!(s, g, sched, ps, st)
println("allocated per sweep: $a bytes, $(round(a / nt; digits = 1)) per task")
Profile.clear(); Profile.init(n = 10^7, delay = 0.0001)
@profile for _ in 1:50; sweep!(s, g, sched, ps, st); end
open(io -> Profile.print(IOContext(io, :displaysize => (10000, 250)); format = :flat, sortedby = :count, mincount = 20),
     joinpath(out, "flat.txt"), "w")
open(io -> Profile.print(IOContext(io, :displaysize => (10000, 250)); mincount = 30, noisefloor = 2),
     joinpath(out, "tree.txt"), "w")
println("profiles in $out")
