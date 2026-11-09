# Experiment 3 of `vault/Control/Composing Optimisers.md` §5: Anderson acceleration of a BP sweep on
# the loopy divider, around flooding and around the spanning-tree order (§4.2).
#
# The sweep is a map G on the message store written as one vector x (every message's (η, Λ),
# `divider.jl`). Anderson with memory m (Walker & Ni 2011, type II, no damping) keeps the last m + 1
# iterates and residuals f = G(x) − x and steps to
#
#     x⁺ = G(x_k) − ΔG γ,   γ = argmin ‖f_k − ΔF γ‖₂,
#
# ΔF, ΔG the differences of consecutive residuals and images. It keeps G's fixed points (a fixed
# point has f = 0, so γ = 0). The least squares are taken in coordinates scaled by each block's
# largest entry (precisions are about 1e3, means about 1), with a ridge of 1e-14 of the Gram
# matrix's trace against the near-singular Gram matrices near convergence. A step that would make a
# precision non-positive is replaced by the plain sweep and the memory cleared (counted as a
# restart). Cost: one sweep (18 message evaluations) per iteration; the least squares on 72 numbers
# are not counted.
#
# Recorded in `vault/Control/Composing Optimisers.md` §4.3.
# Run:  julia --project=. bench/bp_anderson.jl
include("divider.jl")

# runs the sweep with Anderson(m) (m = 0: the plain sweep); evaluations to an error < 1e-10, or,
# with `iterations`, that many iterations and the point reached with its scaled residual
function anderson(g, ps, st, sched, xstar, m; cap = 400_000, iterations = nothing)
    ηi, Λi = 1:2:length(xstar), 2:2:length(xstar)
    sη, sΛ = maximum(abs, xstar[ηi]), maximum(abs, xstar[Λi])
    w = zeros(length(xstar)); w[ηi] .= 1 / sη; w[Λi] .= 1 / sΛ
    err(x) = any(isnan, x) ? Inf : max(maximum(abs, x[ηi] .- xstar[ηi]) / sη, maximum(abs, x[Λi] .- xstar[Λi]) / sΛ)
    s = MessageStore(g); evals = 0; n = length(tasks(sched))
    # a cold store has no messages yet: plain sweeps until every message is Gaussian
    while any(isnan, getstate(s))
        sweep!(s, g, sched, ps, st); evals += n
    end
    x = getstate(s)
    G(x) = (t = setstate!(deepcopy(s), x); sweep!(t, g, sched, ps, st); evals += n; getstate(t))
    Xs = Vector{Float64}[]; Gs = Vector{Float64}[]; restarts = 0
    for _ in 1:something(iterations, cap)
        iterations === nothing && err(x) < 1e-10 && return (; evals, restarts)
        gx = G(x)
        push!(Xs, w .* x); push!(Gs, w .* gx)
        length(Xs) > m + 1 && (popfirst!(Xs); popfirst!(Gs))
        if m == 0 || length(Xs) < 2
            x = gx; continue
        end
        F = [Gs[i] .- Xs[i] for i in eachindex(Xs)]
        ΔF = reduce(hcat, [F[i+1] .- F[i] for i in 1:length(F)-1])
        ΔG = reduce(hcat, [Gs[i+1] .- Gs[i] for i in 1:length(Gs)-1])
        A = ΔF' * ΔF
        γ = (A + 1e-14 * max(tr(A), eps()) * I) \ (ΔF' * F[end])
        xn = (Gs[end] .- ΔG * γ) ./ w
        if any(<=(0), xn[Λi]) || any(!isfinite, xn)
            restarts += 1; empty!(Xs); empty!(Gs); x = gx
        else
            x = xn
        end
    end
    iterations === nothing || return (; x, residual = maximum(abs, w .* (G(x) .- x)), restarts)
    return (; evals = nothing, restarts)
end

function study(R₁, R₂; ms = (0, 1, 2, 3, 5, 10))
    g, ps, st = divider_graph(R₁, R₂, 1e4)
    sp = spanning_tree_schedule(g)
    xstar = fixed_point(g, ps, st, sp)
    # the reference is the plain sweep's rounding floor; check it against Anderson(3) run far past it
    ηi, Λi = 1:2:length(xstar), 2:2:length(xstar)
    sη, sΛ = maximum(abs, xstar[ηi]), maximum(abs, xstar[Λi])
    a = anderson(g, ps, st, sp, xstar, 3; iterations = 300)
    @printf("\nR₁ = %.0f, R₂ = %.0f: the reference against Anderson(3) after 300 iterations (scaled residual %.1e): %.1e\n",
            R₁, R₂, a.residual, max(maximum(abs, a.x[ηi] .- xstar[ηi]) / sη, maximum(abs, a.x[Λi] .- xstar[Λi]) / sΛ))
    @printf("R₁ = %.0f, R₂ = %.0f: message evaluations to an error < 1e-10 (restarts)\n", R₁, R₂)
    base = nothing
    for (name, sc) in (("flooding", flooding_schedule(g)), ("spanning", sp))
        print(@sprintf("  %-9s", name))
        for m in ms
            r = anderson(g, ps, st, sc, xstar, m)
            name == "flooding" && m == 0 && (base = r.evals)
            print(@sprintf("  m=%-2d %7s (%4s, %d)", m, something(r.evals, "–"),
                           r.evals === nothing ? "–" : @sprintf("%.2f", r.evals / base), r.restarts))
        end
        println()
    end
end

study(2.0, 3.0)
study(1.0, 3.0)
study(3.0, 2.0)
study(2.0, 2.0)

# The library's `AndersonSchedule` (Mycelium, `anderson.jl`), run by `infer!` to its message residual
# of 1e-11 as the tests do: sweeps, restarts, and the largest difference of the means from the plain
# spanning-tree order's
println("\nThe library's AndersonSchedule, by infer! (tol = 1e-11 on the message residual):")
for (R₁, R₂) in ((2.0, 3.0), (1.0, 3.0), (3.0, 2.0), (2.0, 2.0))
    g, ps, st = divider_graph(R₁, R₂, 1e4)
    sp = spanning_tree_schedule(g)
    mp, rp, _, _ = infer!(g, sp, ps, st; maxsweeps = 400_000, tol = 1e-11)
    mf, rf, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps = 400_000, tol = 1e-11)
    @printf("  R₁ = %.0f, R₂ = %.0f: flooding %d sweeps, spanning %d", R₁, R₂, rf.sweeps, rp.sweeps)
    for (name, sc) in (("spanning+A(2)", AndersonSchedule(sp; m = 2)), ("spanning+A(3)", AndersonSchedule(sp; m = 3)),
                       ("flooding+A(3)", AndersonSchedule(flooding_schedule(g); m = 3)))
        m, r, _, s = infer!(g, sc, ps, st; maxsweeps = 20_000, tol = 1e-11)
        d = maximum(abs(belief_mean(m[v])[1] - belief_mean(mp[v])[1]) for v in (:v, :i1, :i2))
        @printf(", %s %s (%d restarts, |Δμ| %.0e)", name, r.converged ? string(r.sweeps) : "> 20000",
                anderson_restarts(sc, s), d)
    end
    println()
end
