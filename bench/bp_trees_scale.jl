# Does `coupling_weights` need `normalize = true`? The grids of bp_trees.jl with variables stated in
# mixed units (y_v = x_v / s_v): Gaussian BP is equivariant under such a rescaling, so a schedule that
# picks the same tree does the same work; the raw weight s_u s_v / q changes with the units, the
# normalised one does not. Stopping is relative (tol = 0, rtol = 1e-10), so it is unit-free too.
# Scales: none; "blocks" (the right half of the grid in units 1e-3, i.e. numbers 1000 times larger);
# "random" (s_v = 10^U(-3, 3), seeded). Errors are in the original units, against the dense solve.
# Recorded in lib/Lenticulum.jl/src/coupling.md, "Normalised or raw".
# Run: julia --project=bench bench/bp_trees_scale.jl
# Smoke: LENTICULUM_SMOKE=true (bench/smoke.jl) runs it with 2 by 2 grids and 20 sweeps, all noise and unit variants.

@isdefined(provenance) || include(joinpath(@__DIR__, "provenance.jl"))
abspath(PROGRAM_FILE) == (@__FILE__) && provenance()

@isdefined(SMOKE) || const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
include("bp_trees.jl")

function scales(n, kind; seed = 20261119)
    kind === :none && return ones(n*n)
    kind === :blocks && return [j > n ÷ 2 ? 1e-3 : 1.0 for j in 1:n for i in 1:n]
    kind === :random && return 10.0 .^ (-3 .+ 6 .* rand(Xoshiro(seed), n*n))
    error("unknown scale $kind")
end

function run_scaled(n, mode, kind)
    s = scales(n, kind)
    g, ps, st, exact, qs = trees_grid(n, mode; scale = s)
    precision = Dict(k => 1/v for (k,v) in qs)
    specs = [("BFS", (;)), ("precision", (; weights=precision)),
             ("coupling", (; weights=coupling_weights(g, ps))),
             ("coupling raw", (; weights=coupling_weights(g, ps; normalize=false)))]
    println("\n$n × $n, $mode noise, $kind scale, pv=100")
    println("| schedule | sweeps | evaluations | seconds | error (x units) | converged | tree edges shared with precision |")
    println("|---|---:|---:|---:|---:|---|---:|")
    ref = only(Mycelium._spanning_forests(g; weights=precision))
    for (label, kw) in specs, accelerated in (false, true)
        inner = spanning_tree_schedule(g; kw...)
        forest = only(Mycelium._spanning_forests(g; kw...))
        make_schedule() = accelerated ? AndersonSchedule(inner; m=3) : inner
        run() = infer!(g, make_schedule(), ps, st; maxsweeps=(SMOKE ? 20 : 20_000), tol=0.0, rtol=1e-10)
        run()
        elapsed = @elapsed (marg, rep, _, _) = run()
        err = maximum(s[i] * abs(only(belief_mean(marg[g.variables[i].name])) - exact[i]) for i in 1:nvariables(g))
        @printf("| %s | %d | %d | %.3f | %.3e | %s | %d/%d |\n", label * (accelerated ? "+A(3)" : ""),
                rep.sweeps, rep.sweeps*length(tasks(inner)), elapsed, err, rep.converged,
                count(forest .& ref), count(ref))
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    for mode in (:isotropic, :horizontal, :random), kind in (:none, :blocks, :random)
        run_scaled(SMOKE ? 2 : 10, mode, kind)
    end
end
