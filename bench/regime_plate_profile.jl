# Where `fit_em` spends its time on a plate of `RegimeFactor`s: the mixture of two regressions of
# `rxinfer_baseline.jl` (part 2) as a plate of N points (Feature Backlog I19). Prints the time of
# the whole fit (warm, best of 3) and of one EM iteration with its allocations, profiles one fit
# (one sample per ms) and writes a flat and a tree profile to `~/.cache/lenticulum/regime_plate_profile/`
# (or `$OUT`). Recorded in `lib/Mycelium.jl/src/cluster.md`, "Batching in one stacked joint".
# Run:  N=3000 julia -t auto --project=bench bench/regime_plate_profile.jl
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, LuxCore, Random, Profile

const START = (a = [0.5, -0.5], c = [0.0, 0.0], σ = [1.0, 1.0], w = [0.5, 0.5])
function mix_data(N; seed = 11)            # as in rxinfer_baseline.jl
    rng = Xoshiro(seed)
    x = 6 .* rand(rng, N) .- 3
    k = [rand(rng) < 0.5 ? 1 : 2 for _ in 1:N]
    y = [1.0, -1.0][k] .* x .+ [0.0, 0.5][k] .+ [0.2, 0.5][k] .* randn(rng, N)
    return x, y
end
function mix_plate(x, y)
    b = GraphBuilder()
    for i in eachindex(x)
        r(s) = replica_name(s, i)
        variable!(b, r(:x), 1); variable!(b, r(:y), 1); variable!(b, r(:k), 2)
        factor!(b, r(:dx), DataFactor(:x, [x[i]])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
        factor!(b, r(:dy), DataFactor(:y, [y[i]])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
        factor!(b, r(:reg), RegimeFactor((y = 1, x = 1), 2, 1; learn_noise = true, output = :y); params = :reg)
        connect!(b, r(:reg), :y, r(:y)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
    end
    return as_plate(build(b), :pts, length(x))
end
mix_ps(N) = merge(NamedTuple(s => NamedTuple() for i in 1:N for s in (replica_name(:dx, i), replica_name(:dy, i))),
    (reg = (A = (x = reshape(-START.a, 1, 1, 2),), b = reshape(START.c, 1, 2), logw = log.(START.w),
            L = reshape(log.(START.σ), 1, 1, 2)),))

N = parse(Int, get(ENV, "N", "3000"))
out = get(ENV, "OUT", joinpath(homedir(), ".cache", "lenticulum", "regime_plate_profile")); mkpath(out)
x, y = mix_data(N)
g = mix_plate(x, y)
ps, sched = mix_ps(N), plate_schedule(g)
fit(iters) = fit_em(g, ps, NamedTuple(); schedule = sched, iters, tol = 1e-10)
_, Fs, _ = fit(10_000)                                               # compile
tf = minimum(_ -> (GC.gc(); @elapsed fit(10_000)), 1:3)
t1 = minimum(_ -> (GC.gc(); @elapsed fit(1)), 1:3)
a1 = @allocated fit(1)
println("N = $N: fit_em $(round(tf; digits = 3)) s, $(length(Fs) - 1) iterations, F = $(last(Fs)); ",
        "one iteration $(round(t1 * 1e3; digits = 1)) ms, $(round(a1 / 2^20; digits = 1)) MiB")
Profile.clear(); Profile.init(n = 10^7, delay = 0.001)
@profile fit(10_000)
open(io -> Profile.print(IOContext(io, :displaysize => (10000, 250)); format = :flat, sortedby = :count, mincount = 20),
     joinpath(out, "flat.txt"), "w")
open(io -> Profile.print(IOContext(io, :displaysize => (10000, 250)); maxdepth = 40, mincount = 30, noisefloor = 2),
     joinpath(out, "tree.txt"), "w")
println("profiles in $out")
