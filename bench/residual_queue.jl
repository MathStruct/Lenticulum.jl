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

@isdefined(provenance) || include(joinpath(@__DIR__, "provenance.jl"))
abspath(PROGRAM_FILE) == (@__FILE__) && provenance()

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

# Record actual sweep boundaries, including early stops before the task budget.
struct GridQueueRecord
    store::MessageStore
    commits::Vector{Tuple{Int,MessageTask}}
end
GridQueueRecord(s) = GridQueueRecord(s, Tuple{Int,MessageTask}[])
Base.push!(r::GridQueueRecord, t::MessageTask) = push!(r.commits, (r.store.iteration, t))

# On these undamped, stateless grids the only input changes are the recorded commits.
function queue_counts(g, record, sweeps; reuse, from = 1)
    ts = all_tasks(g)
    vix, fix = Mycelium._residual_task_indices(g, ts)
    rec = record.commits
    @assert rec[1][1] == 1 && rec[end][1] == sweeps
    probes = 0
    commits = 0
    d = Int[]
    for (k, (sweep, task)) in enumerate(rec)
        if k == 1 || rec[k-1][1] != sweep
            @assert k == 1 || rec[k-1][1] == sweep - 1
            if sweep >= from && (!reuse || k == 1)
                probes += length(ts)
            end
        end
        sweep >= from && (commits += 1)
        k == length(rec) && continue
        next_sweep = rec[k+1][1]
        next_sweep >= from || continue
        if next_sweep == sweep || reuse
            empty!(d)
            Mycelium._dependents!(d, record.store, g, task, vix, fix)
            if next_sweep != sweep
                push!(d, task.kind === :to_factor ? fix[task.edge] : vix[task.edge])
            end
            probes += length(unique!(d))
        end
    end
    return probes, probes + commits
end

function one_sweep(n; warm = false)
    g, ps, st, _ = grid_graph(n, 10.0)
    T = length(all_tasks(g))
    function run(f, reuse)
        s = MessageStore(g)
        sched = ResidualSchedule(; maxtasks = T, reuse)
        rec = GridQueueRecord(s)
        warm && sweep!(s, g, sched, ps, st; record = rec)
        t = @elapsed f(s, g, sched, ps, st; record = rec)
        return t, rec
    end
    run(sweep!, false)
    run(sweep!, true)
    off = argmin(first, [run(sweep!, false) for _ in 1:3])
    on = argmin(first, [run(sweep!, true) for _ in 1:3])
    @assert off[2].commits == on[2].commits
    sweep = warm ? 2 : 1
    scan = !warm && T <= (SMOKE ? 100 : 1_500)
    if scan
        run(Mycelium._residual_sweep_scan!, false)
        ts, rs = run(Mycelium._residual_sweep_scan!, false)
        @assert rs.commits == off[2].commits "the queue's commits differ from the scan's"
    end
    p0, e0 = queue_counts(g, off[2], sweep; reuse = false, from = sweep)
    p1, e1 = queue_counts(g, on[2], sweep; reuse = true, from = sweep)
    @printf("  %2d × %-2d tasks %5d  scan %21s  off %9.6f s %6d / %6d   on %9.6f s %6d / %6d\n",
            n, n, T, scan ? @sprintf("%.6f s %d / %d", ts, T^2, T^2 + T) : "–",
            off[1], p0, e0, on[1], p1, e1)
    flush(stdout)
end

function to_convergence(n, pv)
    g, ps, st, exact = grid_graph(n, pv)
    T = length(all_tasks(g)); maxsweeps = SMOKE ? 50 : 20_000
    prior = nothing
    for (label, sched) in (("flooding", flooding_schedule(g)), ("queue off", ResidualSchedule(; reuse = false)),
                           ("queue on", ResidualSchedule()))
        s = MessageStore(g)
        rec = sched isa ResidualSchedule ? GridQueueRecord(s) : nothing
        kw = sched isa ResidualSchedule ? (; record = rec) : (;)
        warm = MessageStore(g)
        warm_kw = rec === nothing ? (;) : (; record = GridQueueRecord(warm))
        propagate!(warm, g, sched, ps, st; maxsweeps = 2, tol = 1e-10, warm_kw...)
        t = @elapsed (rep = first(propagate!(s, g, sched, ps, st; maxsweeps, tol = 1e-10, kw...)))
        marg = NamedTuple{Tuple(v.name for v in g.variables)}(Tuple(marginal(s, g, v.id) for v in g.variables))
        if rec === nothing
            p, ev = 0, rep.sweeps * T
        else
            p, ev = queue_counts(g, rec, rep.sweeps; reuse = sched.reuse)
            if sched.reuse
                @assert rec.commits == prior.commits "heap reuse changed the commits"
            else
                prior = rec
            end
        end
        @printf("  %2d × %-2d pv %6.1f  %-9s  %s  sweeps %5d  %9d / %9d  %8.3f s  mean error %.1e\n",
                n, n, pv, label, rep.converged ? "converged" : "NOT conv.",
                rep.sweeps, p, ev, t, mean_error(marg, exact))
        flush(stdout)
    end
end

println("One sweep from an empty store (best of 3, probes / total evaluations):")
for n in (SMOKE ? (2, 3) : (3, 4, 6, 8, 10, 16, 24, 32)); one_sweep(n); end
println("\nSecond sweep on the same store (best of 3, probes / total evaluations):")
for n in (SMOKE ? (2, 3) : (3, 4, 6, 8, 10, 16, 24, 32)); one_sweep(n; warm = true); end
println("\nTo a message residual of 1e-10 (probes / total evaluations):")
for n in (SMOKE ? (2, 3) : (4, 8, 16)), pv in (1.0, 100.0); to_convergence(n, pv); end
