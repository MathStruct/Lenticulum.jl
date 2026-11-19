# The speed of Float32 cluster plates (P33): `fit_em` on two plates of `RegimeFactor` clusters, in
# Float64 and in Float32, on the stacked path (`cluster_batch.jl`: one joint for all replicas) and
# forced replica by replica (`Mycelium._cluster_batchable` redefined to `false`, which is what a
# Float32 plate did before P33). Recorded in `lib/Mycelium.jl/src/cluster.md`, "Float32 plates".
#
# Plates:
#   :reg1  the mixture of two regressions of `rxinfer_baseline.jl` (part 2): scalar x, y, two
#          regimes, one-factor clusters (`_as_clusters`), as in `regime_plate_profile.jl`; `noise = T(0.1)`:
#          the default `noise = 0.1` would make the output coefficient (`output_coeff = I`) Float64
#          and widen the whole Float32 plate (the initial noise itself comes from `L`);
#   :reg3  the plate of `lib/Lenticulum.jl/test/p31_element_type.jl` ("P32: Float32 EM"): z ∈ R³,
#          x ∈ R² with a Gaussian prior, three regimes, one two-factor cluster per point.
# Every `fit_em` runs a fixed number of iterations (`tol = 0`); a time is the minimum over three runs
# after a warm-up, the allocation that of one more run. The last free energy of each run is printed,
# so the paths can be compared (stacked against per replica in one type agree to rounding).
# Run:  julia -t auto --project=bench bench/float32_cluster_plates.jl
#       (NS = "300,3000", ITERS = 10; output to stdout)
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, LuxCore, Random, Printf

const NT = NamedTuple()

function reg1_plate(T, N)
    rng = Xoshiro(11)                                            # as in rxinfer_baseline.jl
    x = 6 .* rand(rng, N) .- 3
    k = [rand(rng) < 0.5 ? 1 : 2 for _ in 1:N]
    y = [1.0, -1.0][k] .* x .+ [0.0, 0.5][k] .+ [0.2, 0.5][k] .* randn(rng, N)
    b = GraphBuilder()
    for i in 1:N
        r(s) = replica_name(s, i)
        variable!(b, r(:x), 1); variable!(b, r(:y), 1); variable!(b, r(:k), 2)
        factor!(b, r(:dx), DataFactor(:x, T[x[i]])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
        factor!(b, r(:dy), DataFactor(:y, T[y[i]])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
        factor!(b, r(:reg), RegimeFactor((y = 1, x = 1), 2, 1; learn_noise = true, output = :y, noise = T(0.1)); params = :reg)
        connect!(b, r(:reg), :y, r(:y)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
    end
    g = as_plate(build(b), :pts, N)
    ps = merge(NamedTuple(s => NT for i in 1:N for s in (replica_name(:dx, i), replica_name(:dy, i))),
               (reg = (A = (x = reshape(T[-0.5, 0.5], 1, 1, 2),), b = zeros(T, 1, 2), logw = log.(T[0.5, 0.5]),
                       L = zeros(T, 1, 1, 2)),))
    return g, ps
end

function reg3_plate(T, N)                                        # as p32_em_plate in p31_element_type.jl
    rng = Xoshiro(4)
    Wt = [randn(rng, 3, 2) for _ in 1:3]
    μt = [[0.0, 3.0, 0.0], [3.0, -2.0, 1.0], [-3.0, 0.0, 2.0]]
    Z = reduce(hcat, [(k = mod1(i, 3); Wt[k] * randn(rng, 2) .+ μt[k] .+ 0.2 .* randn(rng, 3)) for i in 1:N])
    b = GraphBuilder()
    for i in 1:N
        r(s) = replica_name(s, i)
        variable!(b, r(:z), 3); variable!(b, r(:x), 2); variable!(b, r(:k), 3)
        factor!(b, r(:d), DataFactor(:z, T.(Z[:, i]))); connect!(b, r(:d), :z, r(:z); direction = Emitting())
        factor!(b, r(:pr), GaussianPrior(:x, zeros(T, 2), Matrix{T}(I, 2, 2)); params = :pr); connect!(b, r(:pr), :x, r(:x))
        factor!(b, r(:reg), RegimeFactor((z = 3, x = 2), 3, 3; learn_noise = true, output = :z, noise = T(0.5)); params = :reg)
        connect!(b, r(:reg), :z, r(:z)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
    end
    g = cluster(as_plate(build(b), :pts, N), :pts, :c, [:pr, :reg]; params = :mix)
    f = RegimeFactor((z = 3, x = 2), 3, 3; learn_noise = true, output = :z, noise = T(0.5))
    p0 = map(x -> x isa AbstractArray ? T.(x) : x, merge(LuxCore.initialparameters(Xoshiro(2), f), (b = Z[:, 1:3],)))
    p0 = merge(p0, (A = map(a -> T.(a), p0.A),))
    ps = merge(NamedTuple(replica_name(:d, i) => NT for i in 1:N), (mix = (pr = NT, reg = p0),))
    return g, ps
end

const NS = parse.(Int, split(get(ENV, "NS", "300,3000"), ","))
const ITERS = parse(Int, get(ENV, "ITERS", "10"))
const PLATES = (reg1 = reg1_plate, reg3 = reg3_plate)

function measure(make, T, N)
    g, ps = make(T, N)
    sched = plate_schedule(g)
    fit() = fit_em(g, ps, NT; schedule = sched, iters = ITERS, tol = 0)
    _, Fs, _ = fit()                                             # compile
    t = minimum(_ -> (GC.gc(); @elapsed fit()), 1:3)
    a = @allocated fit()
    return (; t, a, F = last(Fs), n = length(Fs) - 1, FT = eltype(Fs))
end

function run_all(path)
    res = Dict{Tuple{Symbol,DataType,Int},Any}()
    for (name, make) in pairs(PLATES), N in NS, T in (Float64, Float32)
        r = measure(make, T, N)
        res[(name, T, N)] = r
        @printf("%-8s %-5s N = %5d  %-7s: %8.3f s  %9.1f MiB  %d iterations  F = %s (%s)\n",
                path, name, N, T, r.t, r.a / 2^20, r.n, repr(r.F), r.FT)
        flush(stdout)
    end
    return res
end

println("threads = $(Threads.nthreads()), ITERS = $ITERS, Julia $(VERSION)")
stacked = run_all("stacked")
# Every cluster plate now goes replica by replica (P33 left this path for duals and mixed types).
@eval Mycelium begin
    _cluster_batchable(cs, msgs, skip, @nospecialize(ps)) = false
    _cluster_batchable(cs, msgs::NamedTuple, skip, @nospecialize(ps)) = false
end
perrep = run_all("replica")

println("\nratios (time): replica / stacked, and Float32 / Float64 on each path")
for (name, _) in pairs(PLATES), N in NS
    s64, s32 = stacked[(name, Float64, N)], stacked[(name, Float32, N)]
    r64, r32 = perrep[(name, Float64, N)], perrep[(name, Float32, N)]
    @printf("%-5s N = %5d  replica/stacked: F64 %5.2f  F32 %5.2f (F32 before P33)   F32/F64: stacked %5.2f  replica %5.2f   |F32 - F64|/|F64| stacked %.1e   |ΔF| replica - stacked: F64 %.1e  F32 %.1e\n",
            name, N, r64.t / s64.t, r32.t / s32.t, s32.t / s64.t, r32.t / r64.t,
            abs(s32.F - s64.F) / abs(s64.F), abs(r64.F - s64.F), abs(r32.F - s32.F))
end
