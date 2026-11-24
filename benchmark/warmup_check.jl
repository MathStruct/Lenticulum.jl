# LP-088 F1: a case with fixed `evals` must have filled the graph's caches before its first timed sample.
# `tune!` does nothing for such a case and `run` does not warm up, so `worker.jl` calls
# `BenchmarkTools.warmup` after `tune!`. This repeats the worker's sequence on Mycelium's chain.
# Run by `benchmark/smoke.jl` with `--project=benchmark`.
include(joinpath(@__DIR__, "common.jl"))
using BenchmarkTools

suite = load_suite("Mycelium")
g = first(Base.invokelatest(suite.kalman))
bench = Base.invokelatest(getproperty, suite, :SUITE)["Mycelium"]["kalman"]["tree_infer"]
filled(g) = sum(length, g.polarities) > 0 && !isempty(g.emit_plans)
@assert !filled(g) "the fixture's graph should start with empty caches"
worker = read(joinpath(@__DIR__, "worker.jl"), String)
@assert occursin("BenchmarkTools.warmup(benchmark", worker) "worker.jl no longer warms up before sampling"
tune!(bench; seconds=0.15)
BenchmarkTools.warmup(bench; verbose=false)
@assert filled(g) "the graph's caches are still empty after tune! and warmup"
println("Benchmark warm-up check passed")
