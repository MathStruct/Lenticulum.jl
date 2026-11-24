# Which spanning tree, and do alternating trees repay their extra message evaluations?
# Gaussian grids with pv = 100 and isotropic, anisotropic, or seeded log-uniform edge noise.
# Each mean is checked against the dense precision solve; second run timed after warm-up.
# Recorded in lib/Mycelium.jl/src/schedules.md, "Choice of tree and alternating trees".
# "coupling" is `coupling_weights(g, ps)` (I25, normalised); "coupling raw" its `normalize = false`.
# Run: julia --project=bench bench/bp_trees.jl
# Smoke: LENTICULUM_SMOKE=true (bench/smoke.jl) runs it with 2 by 2 grids and 20 sweeps, all schedules and noise variants.

@isdefined(provenance) || include(joinpath(@__DIR__, "provenance.jl"))
abspath(PROGRAM_FILE) == (@__FILE__) && provenance()

@isdefined(SMOKE) || const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
include("divider.jl")
using Random

# `scale` (length n², default ones) states variable v in units s_v: y_v = x_v / s_v, so the prior is
# N(m/s, pv/s²), a constraint reads s_u y_u - s_v y_v, and `exact` is in the y units.
function trees_grid(n, mode; seed = 20261118, scale = ones(n*n))
    rng = Xoshiro(seed); pv = 100.0; sc = scale
    m = randn(rng, n*n)
    name(i,j) = Symbol(:x, i, :_, j)
    idx(i,j) = (j-1)*n+i
    b = GraphBuilder(); pd = Dict{Symbol,Any}(); qs = Dict{Symbol,Float64}()
    J = Matrix(1.0I, n*n, n*n) / pv
    for j in 1:n, i in 1:n
        v, f = name(i,j), Symbol(:p, i, :_, j)
        variable!(b, v, 1)
        factor!(b, f, GaussianPrior(:x, [m[idx(i,j)] / sc[idx(i,j)]], fill(pv / sc[idx(i,j)]^2, 1, 1)))
        connect!(b, f, :x, v); pd[f] = NamedTuple(); qs[f] = 1.0
    end
    for j in 1:n, i in 1:n, (di,dj) in ((1,0), (0,1))
        (i+di <= n && j+dj <= n) || continue
        q = mode === :random ? 10.0^(-2 + 3rand(rng)) :
            mode === :horizontal ? (di == 1 ? 0.01 : 1.0) :
            mode === :vertical ? (dj == 1 ? 0.01 : 1.0) : 1.0
        f = Symbol(:s, i, :_, j, :_, di)
        factor!(b, f, LinearConstraintFactor((a=1, b=1), 1; noise=fill(q,1,1)))
        connect!(b, f, :a, name(i,j)); connect!(b, f, :b, name(i+di,j+dj))
        a,c = idx(i,j),idx(i+di,j+dj)
        pd[f] = (A=(a=fill(sc[a],1,1),b=fill(-sc[c],1,1)), c=[0.0]); qs[f] = q
        J[a,a] += 1/q; J[c,c] += 1/q; J[a,c] -= 1/q; J[c,a] -= 1/q
    end
    g = validate(build(b)); ps = NamedTuple(pd)
    st = NamedTuple(k => NamedTuple() for k in keys(ps))
    exact = (J \ (m/pv)) ./ sc
    return g, ps, st, exact, qs
end

function run_trees(n, mode)
    g, ps, st, exact, qs = trees_grid(n, mode)
    precision = Dict(k => 1/v for (k,v) in qs)
    specs = [("flooding", nothing), ("BFS", (;)), ("precision", (; weights=precision)),
             ("q control", (; weights=qs)), ("BFS 2", (; ntrees=2)),
             ("BFS cover", (; ntrees=:cover)), ("precision 2", (; weights=precision, ntrees=2)),
             ("precision cover", (; weights=precision, ntrees=:cover)),
             ("coupling", (; weights=coupling_weights(g, ps))),
             ("coupling raw", (; weights=coupling_weights(g, ps; normalize=false)))]
    println("\n$n × $n, $mode, pv=100, seed=20261118")
    println("| schedule | sweeps | evaluations | trees | seconds | mean error | converged |")
    println("|---|---:|---:|---:|---:|---:|---|")
    start = time()
    for (label, kw) in specs, accelerated in (false, true)
        inner = kw === nothing ? flooding_schedule(g) : spanning_tree_schedule(g; kw...)
        k = kw === nothing ? 0 : length(Mycelium._spanning_forests(g; kw...))
        make_schedule() = accelerated ? AndersonSchedule(inner; m=3) : inner
        infer!(g, make_schedule(), ps, st; maxsweeps=(SMOKE ? 20 : 20_000), tol=1e-11)
        elapsed = @elapsed (marg, rep, _, _) = infer!(g, make_schedule(), ps, st; maxsweeps=(SMOKE ? 20 : 20_000), tol=1e-11)
        err = maximum(abs(only(belief_mean(marg[g.variables[i].name])) - exact[i]) for i in 1:nvariables(g))
        @printf("| %s | %d | %d | %d | %.3f | %.3e | %s |\n", label * (accelerated ? "+A(3)" : ""),
                rep.sweeps, rep.sweeps*length(tasks(inner)), k, elapsed, err, rep.converged)
        flush(stdout)
    end
    @printf("Setting total: %.1f seconds\n", time()-start)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for (n,mode) in ((SMOKE ? 2 : 10, :isotropic),(SMOKE ? 2 : 10, :horizontal),(SMOKE ? 2 : 10, :vertical),(SMOKE ? 2 : 10, :random))
        run_trees(n, mode)
    end
end
