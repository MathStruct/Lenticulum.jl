# Experiment 2 of `vault/Composing Optimisers.md` §5: on the loopy divider, is the speed-up of the
# residual schedule (experiment 1, §4.1) in the ORDER of the messages, and can it be had at
# flooding's cost per sweep?
#
# Five ways to run the same 18 messages, all compared by MESSAGE EVALUATIONS (calls of the update
# rule, committed or not) to a relative error of 1e-10 against the fixed point:
#
#   flooding    double buffered, 18 evaluations per sweep (the baseline)
#   sequential  in place, in `all_tasks` order (edge by edge)
#   loop        in place: the messages off the loop, then once round the loop each way
#   spanning    in place: the messages off a spanning tree, then the tree's inward and outward
#               sweeps (`_tree_sweeps` from a variable); needs no knowledge of the loop
#   residual    `ResidualSchedule`: every candidate evaluated before each commit (18² per sweep)
#   queue       residual BP with a priority queue (Elidan, McGraw & Koller): after a commit only
#               the messages that read the committed one are re-evaluated
#
# The queue's dependencies are those of belief propagation (a message reads the messages into
# its source node on the other edges); mean-field and EP nodes read more, and are not covered.
#
# Recorded in `vault/Control/Composing Optimisers.md` §4.2.
# Run:  julia --project=. bench/bp_schedules.jl
include("divider.jl")

relerr(x, xstar, ηi, Λi, sη, sΛ) = any(isnan, x) ? Inf :
    max(maximum(abs, x[ηi] .- xstar[ηi]) / sη, maximum(abs, x[Λi] .- xstar[Λi]) / sΛ)

# one cycle of the factor graph, as a closed walk of edges (depth-first search from variable 1)
function find_cycle(g)
    nodes(kind, i) = kind === :variable ?
        [(ei, (:factor, g.edges[ei].factor)) for ei in g.edges_of_variable[i]] :
        [(ei, (:variable, g.edges[ei].variable)) for ei in g.edges_of_factor[i]]
    parent = Dict{Tuple{Symbol,Int},Tuple{Int,Tuple{Symbol,Int}}}()   # node => (edge, from)
    root = (:variable, 1); seen = Set([root]); stack = [root]
    while !isempty(stack)
        n = pop!(stack)
        for (ei, m) in nodes(n...)
            haskey(parent, n) && parent[n][1] == ei && continue
            if m in seen    # closing edge: walk both ends back to their common ancestor
                path(x) = (p = [x]; while haskey(parent, p[end]); push!(p, parent[p[end]][2]); end; p)
                pn, pm = path(n), path(m)
                common = first(x for x in pn if x in pm)
                edges_up(p) = [parent[x][1] for x in p[1:findfirst(==(common), p)-1]]
                # walk: common → … → m, closing edge m → n, n → … → common
                walk = vcat(reverse(edges_up(pm)), [ei], edges_up(pn))
                start = common
                return walk, start
            end
            push!(seen, m); parent[m] = (ei, n); push!(stack, m)
        end
    end
    error("no cycle")
end

# the messages that carry information along `walk` from `start`
function along(g, walk, start)
    out = MessageTask[]; at = start
    for ei in walk
        e = g.edges[ei]
        if at[1] === :variable
            push!(out, MessageTask(ei, :to_factor)); at = (:factor, e.factor)
        else
            push!(out, MessageTask(ei, :to_variable)); at = (:variable, e.variable)
        end
    end
    return out
end

function loop_order(g)
    walk, start = find_cycle(g)
    cw = along(g, walk, start)
    ccw = along(g, reverse(walk), start)
    return SequentialSchedule(vcat(setdiff(Mycelium.all_tasks(g), cw, ccw), cw, ccw))
end

function spanning_order(g)
    ts = Mycelium._tree_sweeps(g, (:variable, 1))
    tree = vcat(ts.inward, ts.outward)
    return SequentialSchedule(vcat(setdiff(Mycelium.all_tasks(g), tree), tree))
end

# BP dependencies: the messages that read task t's message
function dependents(g, t)
    e = g.edges[t.edge]
    t.kind === :to_factor ?
        [MessageTask(ej, :to_variable) for ej in g.edges_of_factor[e.factor] if ej != t.edge && Mycelium.can_emit(g, ej)] :
        [MessageTask(ej, :to_factor) for ej in g.edges_of_variable[e.variable] if ej != t.edge && Mycelium.can_absorb(g, ej)]
end

# a static schedule: evaluations and sweeps to an error of 1e-10, and the rate per sweep
function run_static(sched, g, ps, st, xstar, err; cap = 400_000)
    s = MessageStore(g); e = Float64[]
    for k in 1:cap
        sweep!(s, g, sched, ps, st)
        push!(e, err(getstate(s)))
        e[end] < 1e-12 && break
    end
    n = length(tasks(sched isa ResidualSchedule ? flooding_schedule(g) : sched))
    k10 = findfirst(<(1e-10), e)
    b = findfirst(<(1e-12), e)
    rate = (b === nothing || b <= 7) ? NaN : (e[b] / e[b-6])^(1 / 6)
    evals = sched isa ResidualSchedule ? n^2 : n
    return (; sweeps = k10, evals = k10 === nothing ? nothing : k10 * evals, rate)
end

# residual BP with a priority queue; one "sweep" here is as many commits as there are tasks
function run_queue(g, ps, st, xstar, err; cap = 10_000_000)
    s = MessageStore(g)
    ts = Mycelium.all_tasks(g); n = length(ts)
    res = Dict{MessageTask,Float64}(); evals = 0
    function probe(t)
        evals += 1
        _, r, _, skip = Mycelium._step!(s, g, t, ps, st; commit = false)
        return skip ? -1.0 : r
    end
    for t in ts; res[t] = probe(t); end
    commits = 0
    while evals < cap
        t = argmax(t -> (res[t], -Mycelium._task_age(s, t)), ts)
        s.iteration += 1
        step!(s, g, t, ps, st); evals += 1; commits += 1
        res[t] = 0.0
        for d in dependents(g, t); res[d] = probe(d); end
        err(getstate(s)) < 1e-10 && return (; commits, sweeps = cld(commits, n), evals)
    end
    return (; commits, sweeps = nothing, evals = nothing)
end

function study(R₁, R₂; cap = 400_000, slow = true)
    g, ps, st = divider_graph(R₁, R₂, 1e4)
    # the fixed point, by the loop order run to its rounding floor
    s = MessageStore(g); lo = loop_order(g)
    best, kbest = Inf, 0
    for k in 1:cap
        r, _ = sweep!(s, g, lo, ps, st)
        r < best && ((best, kbest) = (r, k))
        (r == 0 || k - kbest > 200) && break      # no progress for 200 sweeps: the floor
    end
    xstar = getstate(s)
    ηi, Λi = 1:2:length(xstar), 2:2:length(xstar)
    sη, sΛ = maximum(abs, xstar[ηi]), maximum(abs, xstar[Λi])
    err(x) = relerr(x, xstar, ηi, Λi, sη, sΛ)
    @printf("\nR₁ = %.0f, R₂ = %.0f (%d tasks; the loop: %s)\n", R₁, R₂, length(tasks(lo)),
            join(("$(t.kind === :to_factor ? "v→f" : "f→v") e$(t.edge)" for t in tasks(lo)[end-11:end-6]), ", "))
    scheds = [("flooding", flooding_schedule(g)), ("sequential", SequentialSchedule(Mycelium.all_tasks(g))),
              ("loop", lo), ("spanning", spanning_order(g))]
    slow && push!(scheds, ("residual", ResidualSchedule(; maxtasks = length(tasks(lo)))))
    base = nothing
    for (name, sc) in scheds
        r = run_static(sc, g, ps, st, xstar, err; cap)
        name == "flooding" && (base = r.evals)
        @printf("  %-10s sweeps %7s  evaluations %8s  (%5s × flooding)  rate per sweep %.4f\n", name,
                something(r.sweeps, "–"), something(r.evals, "–"),
                r.evals === nothing || base === nothing ? "–" : @sprintf("%.2f", r.evals / base), r.rate)
    end
    q = run_queue(g, ps, st, xstar, err)
    @printf("  %-10s commits %6s  evaluations %8s  (%5s × flooding)\n", "queue", something(q.commits, "–"),
            something(q.evals, "–"), q.evals === nothing || base === nothing ? "–" : @sprintf("%.2f", q.evals / base))
end

study(2.0, 3.0)                 # the test's divider
study(1.0, 3.0)
study(3.0, 2.0)
study(2.0, 2.0; slow = false)   # equal resistors: ρ⁶ = 0.99905 for flooding
