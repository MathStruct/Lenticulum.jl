# The spectrum of the linearised flooding sweep on the resistive divider (the loopy graph of
# `test/runtests.jl`, `divider_graph(1e4)`), and what it predicts for damping and heavy ball.
#
# The message store is written as one vector x of every message's (η, Λ); one flooding sweep is a
# map x ↦ Φ(x), and its Jacobian at the fixed point x* is taken by central differences. Gaussian BP's
# precisions do not depend on the means, so the Jacobian is block triangular and its spectrum is the
# union of the mean block's and the precision block's. Each modifier then acts on each eigenvalue λ:
# relaxation of every message maps λ ↦ 1 − α + αλ (Mycelium's `damping` relaxes only half of them, below); heavy ball turns λ into the roots of
# μ² − (λ + β)μ + β. The predictions are checked against real runs.
#
# Recorded in `vault/Control/Composing Optimisers.md` §4 (experiment 1 of §5).
# Run:  julia --project=. bench/bp_spectrum.jl

include("divider.jl")

heavyball(λ, β) = (d = sqrt(complex((λ + β)^2 - 4β)); max(abs((λ + β + d) / 2), abs((λ + β - d) / 2)))
relaxed(λs, α) = maximum(l -> abs(1 - α + α * l), λs)

function study(R₁, R₂; maxsweeps = 5000, runs = true)
    g, ps, st = divider_graph(R₁, R₂, 1e4)
    sched = flooding_schedule(g)
    s = MessageStore(g)
    nsw, r = 0, Inf
    for k in 1:maxsweeps
        r, _ = sweep!(s, g, sched, ps, st); nsw = k
        r <= 1e-13 && break
    end
    xstar = getstate(s)
    Φ(x) = (t = setstate!(deepcopy(s), x); sweep!(t, g, sched, ps, st); getstate(t))
    n = length(xstar); J = zeros(n, n)
    for j in 1:n
        h = 1e-6 * max(1.0, abs(xstar[j]))
        xp = copy(xstar); xp[j] += h; xm = copy(xstar); xm[j] -= h
        J[:, j] = (Φ(xp) .- Φ(xm)) ./ (2h)
    end
    ηi, Λi = 1:2:n, 2:2:n
    λη = filter(l -> abs(l) > 1e-9, eigvals(J[ηi, ηi]))
    λΛ = filter(l -> abs(l) > 1e-9, eigvals(J[Λi, Λi]))
    ρη, ρΛ = maximum(abs, λη), maximum(abs, λΛ)
    args(λs, ρ) = sort(unique(round.(Int, rad2deg.(angle.(filter(l -> abs(l) > 0.999ρ, λs))))))
    @printf("\nR₁ = %.0f, R₂ = %.0f: flooding, %d sweeps (residual %.1e; 1e-13 is at the rounding floor; the test uses 1e-11); ‖Φ(x*) − x*‖∞ = %.1e, max|∂Λ'/∂η| = %.1e\n",
            R₁, R₂, nsw, r, maximum(abs, Φ(xstar) .- xstar), maximum(abs, J[Λi, ηi]))
    @printf("  means:      %2d nonzero eigenvalues, all |λ| = %.6f (spread %.0e), arg ∈ %s°; ρ^6 = %.6f\n",
            length(λη), ρη, ρη - minimum(abs, λη), args(λη, ρη), ρη^6)
    @printf("  precisions: %2d nonzero eigenvalues, all |λ| = %.6f (spread %.0e), arg ∈ %s°; ρ^6 = %.6f\n",
            length(λΛ), ρΛ, ρΛ - minimum(abs, λΛ), args(λΛ, ρΛ), ρΛ^6)
    allλ = vcat(λη, λΛ)
    αs = 0.05:0.05:1.5
    αb = αs[argmin([relaxed(allλ, α) for α in αs])]
    @printf("  predicted: relaxation best α = %.2f → %.4f (α = 1: %.4f, 0.5: %.4f, 0.2: %.4f)\n",
            αb, relaxed(allλ, αb), relaxed(allλ, 1.0), relaxed(allλ, 0.5), relaxed(allλ, 0.2))
    @printf("  predicted: heavy ball β = 0.05, 0.06, 0.07, 0.1, 0.3 → %s\n",
            join((@sprintf("%.4f", maximum(l -> heavyball(l, β), allλ)) for β in (0.05, 0.06, 0.07, 0.1, 0.3)), ", "))
    # Mycelium's `damping` relaxes only the factor-to-variable messages (passing.jl, `_step!`), in
    # natural parameters: its Jacobian is D J + (I − D), D = (1 − damping) on those entries, 1 elsewhere.
    tv = [k === :to_variable for (k, _) in _slots(s) for _ in 1:2]
    mycelium(d) = maximum(abs, eigvals(Diagonal(ifelse.(tv, 1 - d, 1.0)) * J + Diagonal(ifelse.(tv, d, 0.0))))
    ds = 0.0:0.05:0.95
    db = ds[argmin(mycelium.(ds))]
    @printf("  predicted: Mycelium damping best = %.2f → %.4f (0: %.4f, 0.25: %.4f, 0.5: %.4f, 0.8: %.4f)\n",
            db, mycelium(db), mycelium(0.0), mycelium(0.25), mycelium(0.5), mycelium(0.8))
    runs || return
    # measured: the mean and precision errors to x*, each relative to its block's largest entry
    sη, sΛ = maximum(abs, xstar[ηi]), maximum(abs, xstar[Λi])
    floodk = 0
    for d in (0.0, 0.25, 0.5, 0.8)
        t = MessageStore(g); eη = Float64[]; eΛ = Float64[]
        for k in 1:20000
            sweep!(t, g, sched, ps, st; damping = d)
            x = getstate(t)
            push!(eη, any(isnan, x) ? Inf : maximum(abs, x[ηi] .- xstar[ηi]) / sη)
            push!(eΛ, any(isnan, x) ? Inf : maximum(abs, x[Λi] .- xstar[Λi]) / sΛ)
            max(eη[end], eΛ[end]) < 1e-13 && break
        end
        k10 = findfirst(k -> max(eη[k], eΛ[k]) < 1e-10, eachindex(eη))
        d == 0 && (floodk = k10)
        # the asymptotic rate: the larger error's decay over the last 36 sweeps (six trips round the
        # loop) before it falls below 1e-12
        e = max.(eη, eΛ)
        b = findfirst(<(1e-12), e)
        rate = (b === nothing || b <= 37) ? NaN : (e[b] / e[b-36])^(1 / 36)
        @printf("  measured: Mycelium damping = %.2f: predicted rate %.4f, measured %.4f, sweeps to both errors < 1e-10: %s\n",
                d, mycelium(d), rate, k10 === nothing ? "> 20000" : string(k10))
    end
    # the schedule as a modifier: a sequential sweep (in place) can carry a message round the loop in
    # one sweep, where flooding needs six. Both in flooding's task order and with as many updates per sweep
    # the order the residual schedule chooses in its 20th sweep (warm), replayed as a plain sequential
    # sweep: the same order at flooding's cost per sweep
    rs = ResidualSchedule(; maxtasks = length(tasks(sched)))
    t = MessageStore(g); chosen = MessageTask[]
    for k in 1:20
        sweep!(t, g, rs, ps, st; record = k == 20 ? chosen : nothing)
    end
    @printf("  the residual schedule's warm sweep: %d updates, %d distinct tasks of %d\n",
            length(chosen), length(unique(chosen)), length(tasks(sched)))
    for (name, sc) in (("sequential", SequentialSchedule(tasks(sched))), ("residual", rs),
                       ("replayed", SequentialSchedule(vcat(chosen, setdiff(tasks(sched), chosen)))))
        t = MessageStore(g); k10 = nothing; e = Float64[]
        for k in 1:20000
            sweep!(t, g, sc, ps, st)
            x = getstate(t)
            push!(e, any(isnan, x) ? Inf : max(maximum(abs, x[ηi] .- xstar[ηi]) / sη, maximum(abs, x[Λi] .- xstar[Λi]) / sΛ))
            e[end] < 1e-12 && break
        end
        k10 = findfirst(<(1e-10), e)
        b = findfirst(<(1e-12), e)
        rate = (b === nothing || b <= 7) ? NaN : (e[b] / e[b-6])^(1 / 6)
        # message evaluations per sweep: the residual schedule evaluates every task's candidate before
        # each of its commits (passing.jl), the others evaluate each of their tasks once
        evals = sc isa ResidualSchedule ? length(tasks(sched))^2 : length(tasks(sc))
        @printf("  measured: %-10s schedule: rate %.4f per sweep (flooding ρ = %.4f, ρ^6 = %.4f), sweeps to both errors < 1e-10: %s, message evaluations: %s (flooding: %d)\n",
                name, rate, ρη, ρη^6, k10 === nothing ? "> 20000" : string(k10),
                k10 === nothing ? "–" : string(k10 * evals), floodk * length(tasks(sched)))
    end
end

study(2.0, 3.0)                                   # the test's divider
study(1.0, 3.0)
study(3.0, 2.0)
study(2.0, 2.0; maxsweeps = 400_000, runs = false) # equal resistors: ρ ≈ 1, spectrum only
