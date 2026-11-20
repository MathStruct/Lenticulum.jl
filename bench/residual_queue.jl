# I4: the residual schedule's priority queue against the full scan it replaced, on Gaussian n × n
# grids (`bp_grid.jl`'s model: scalar variables, priors N(m_k, pv), a smoothness constraint on every
# grid edge; Gaussian BP's means are exact at a fixed point, checked against the dense solve).
#
# 1. One sweep from an empty store: the full scan (`Mycelium._residual_sweep_scan!`, every task
#    probed before each commit, O(T²) evaluations) against the queue (`sweep!` with a
#    `ResidualSchedule`: every task probed once, then only the dependents of each commit). Both must
#    commit the same tasks in the same order; the time per sweep and its ratio.
# 2. To convergence (`propagate!`, message residual 1e-10, at most 20 000 sweeps): flooding against
#    the residual schedule, in sweeps, message evaluations and wall time. The queue's evaluations
#    are counted from its record: T per sweep and, per commit, the dependents re-probed.
#
# Recorded in `vault/Gaussian Graphs/Composing Optimisers.md` §4.2 and `lib/Mycelium.jl/src/passing.md`,
# "The residual schedule's queue".
# Run:  julia --project=bench bench/residual_queue.jl
# Smoke: LENTICULUM_SMOKE=true (bench/smoke.jl) runs 2 × 2 and 3 × 3 grids, at most 50 sweeps.

@isdefined(SMOKE) || const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
include("divider.jl")
using Random

function grid_graph(n, pv; q = 1.0, seed = 1)
    rng = Xoshiro(seed)
    m = randn(rng, n * n)
    name(i, j) = Symbol(:x, i, :_, j)
    idx(i, j) = (j - 1) * n + i
    b = GraphBuilder(); psd = Dict{Symbol,Any}(); L = zeros(n * n, n * n)
    for j in 1:n, i in 1:n
        variable!(b, name(i, j), 1)
        u = Symbol(:u, i, :_, j)
        factor!(b, u, GaussianPrior(:x, [m[idx(i, j)]], fill(pv, 1, 1)))
        connect!(b, u, :x, name(i, j); direction = Bidirectional())
        psd[u] = NamedTuple()
    end
    for j in 1:n, i in 1:n, (di, dj) in ((1, 0), (0, 1))
        (i + di <= n && j + dj <= n) || continue
        f = Symbol(:s, i, :_, j, :_, di)
        factor!(b, f, LinearConstraintFactor((a = 1, b = 1), 1; noise = fill(q, 1, 1)))
        connect!(b, f, :a, name(i, j); direction = Bidirectional())
        connect!(b, f, :b, name(i + di, j + dj); direction = Bidirectional())
        psd[f] = (A = (a = Ones1, b = -Ones1), c = [0.0])
        a, c = idx(i, j), idx(i + di, j + dj)
        L[a, a] += 1; L[c, c] += 1; L[a, c] -= 1; L[c, a] -= 1
    end
    g = validate(build(b))
    ps = NamedTuple(psd); st = NamedTuple(k => NamedTuple() for k in keys(ps))
    μ = (I / pv + L / q) \ (m ./ pv)
    exact = Dict(name(i, j) => μ[idx(i, j)] for j in 1:n, i in 1:n)
    return g, ps, st, exact
end

mean_error(marg, exact) = maximum(abs(belief_mean(marg[k])[1] - v) for (k, v) in exact)

# the queue's evaluations: T per sweep, and per commit the distinct dependents it re-probes
function queue_evals(g, rec, sweeps)
    ts = all_tasks(g); vix, fix = zeros(Int, nedges(g)), zeros(Int, nedges(g))
    for (i, t) in enumerate(ts); t.kind === :to_factor ? (fix[t.edge] = i) : (vix[t.edge] = i); end
    d = Int[]
    s0 = MessageStore(g)       # no pins: only `_dependents!`'s pin check reads it
    # every sweep commits T tasks (none reaches residual 0 here), with no probe after its last
    return sweeps * length(ts) + sum(((k, t),) -> k % length(ts) == 0 ? 0 :
                                     (empty!(d); length(unique!(Mycelium._dependents!(d, s0, g, t, vix, fix)))),
                                     enumerate(rec); init = 0)
end

function one_sweep(n)
    g, ps, st, _ = grid_graph(n, 10.0)
    T = length(all_tasks(g)); sched = ResidualSchedule(; maxtasks = T)
    run(f) = (s = MessageStore(g); rec = MessageTask[]; t = @elapsed f(s, g, sched, ps, st; record = rec); (t, rec))
    run(sweep!)                                                         # compile
    tq, rq = minimum(first, (run(sweep!) for _ in 1:3)), last(run(sweep!))
    scan = T <= (SMOKE ? 100 : 1_500)
    if scan
        run(Mycelium._residual_sweep_scan!)
        ts, rs = run(Mycelium._residual_sweep_scan!)
        @assert rs == rq "the queue's commits differ from the scan's"
    end
    @printf("  %3d × %-3d tasks %6d  scan %10s  queue %9.4f s  (%s)  evaluations: scan %9s, queue %7d\n", n, n, T,
            scan ? @sprintf("%9.4f s", ts) : "–", tq, scan ? @sprintf("%7.1f× faster", ts / tq) : "scan not run",
            scan ? string(T^2) : "–", queue_evals(g, rq, 1))
end

function to_convergence(n, pv)
    g, ps, st, exact = grid_graph(n, pv)
    T = length(all_tasks(g)); maxsweeps = SMOKE ? 50 : 20_000
    for (label, sched) in (("flooding", flooding_schedule(g)), ("residual", ResidualSchedule()))
        rec = sched isa ResidualSchedule ? MessageTask[] : nothing
        kw = sched isa ResidualSchedule ? (; record = rec) : (;)
        propagate!(MessageStore(g), g, sched, ps, st; maxsweeps = 2, tol = 1e-10, kw...)          # compile
        rec === nothing || empty!(rec)
        s = MessageStore(g)
        t = @elapsed (rep = first(propagate!(s, g, sched, ps, st; maxsweeps, tol = 1e-10, kw...)))
        marg = NamedTuple{Tuple(v.name for v in g.variables)}(Tuple(marginal(s, g, v.id) for v in g.variables))
        ev = rec === nothing ? rep.sweeps * T : queue_evals(g, rec, rep.sweeps)
        @printf("  %3d × %-3d pv %6.1f  %-8s  %s  sweeps %6d  evaluations %9d  %8.3f s  mean error %.1e\n", n, n, pv, label,
                rep.converged ? "converged" : "NOT conv.", rep.sweeps, ev, t, mean_error(marg, exact))
    end
end

println("One sweep from an empty store (time: best of 3; evaluations: the scan's T², the queue's counted):")
for n in (SMOKE ? (2, 3) : (3, 4, 6, 8, 10, 16, 24, 32)); one_sweep(n); end
println("\nTo a message residual of 1e-10:")
for n in (SMOKE ? (2, 3) : (4, 8, 16)), pv in (1.0, 100.0); to_convergence(n, pv); end
