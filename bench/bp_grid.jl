# Two questions left open by `vault/Control/Composing Optimisers.md` §4.2–4.3:
#
# 1. Why does Anderson with a memory of 2–3 suffice around the spanning-tree order? The spectrum
#    of the linearised spanning-tree sweep against flooding's, on the divider and on small grids.
# 2. Do the spanning-tree order and Anderson still pay on a graph with MANY loops? A Gaussian
#    n × n grid: scalar variables x_k with priors N(m_k, pv) and a smoothness constraint
#    x_a − x_b ~ N(0, q) on every grid edge, so the exact means solve J μ = h with
#    J = I/pv + L/q (L the grid Laplacian) and h = m/pv. Gaussian BP's means are exact at a fixed
#    point, so every run is checked against J \ h. Large pv makes the priors weak and the
#    smoothing long-range, the slow regime.
#
# Spectra by central differences of one sweep at the fixed point (`bp_spectrum.jl`'s method);
# sweeps by `infer!` to a message residual of 1e-11, at most 20 000.
#
# Recorded in `vault/Control/Composing Optimisers.md` §4.4.
# Run:  julia --project=. bench/bp_grid.jl
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
    J = I / pv + L / q
    μ = J \ (m ./ pv)
    exact = Dict(name(i, j) => μ[idx(i, j)] for j in 1:n, i in 1:n)
    D = Diagonal(1 ./ sqrt.(diag(J)))
    ρR = maximum(abs, eigvals(Symmetric(Matrix(I - D * J * D))))   # of the normalised couplings
    return g, ps, st, exact, ρR
end

# the nonzero eigenvalues of the linearised sweep's mean and precision blocks at the fixed point
function sweep_spectrum(g, ps, st, sched, s0)
    xstar = getstate(s0)
    Φ(x) = (t = setstate!(deepcopy(s0), x); sweep!(t, g, sched, ps, st); getstate(t))
    n = length(xstar); J = zeros(n, n)
    for j in 1:n
        h = 1e-6 * max(1.0, abs(xstar[j]))
        xp = copy(xstar); xp[j] += h; xm = copy(xstar); xm[j] -= h
        J[:, j] = (Φ(xp) .- Φ(xm)) ./ (2h)
    end
    ηi, Λi = 1:2:n, 2:2:n
    λη = sort(filter(l -> abs(l) > 1e-7, eigvals(J[ηi, ηi])); by = abs, rev = true)
    λΛ = sort(filter(l -> abs(l) > 1e-7, eigvals(J[Λi, Λi])); by = abs, rev = true)
    return λη, λΛ
end

function show_spectrum(label, λη, λΛ; k = 6)
    distinct(λs) = length(unique(round.(abs.(λs); digits = 6)))
    @printf("  %-9s means: %3d nonzero, %3d distinct moduli, ρ = %.6f; largest: %s\n", label, length(λη),
            distinct(λη), isempty(λη) ? 0.0 : abs(λη[1]),
            join((@sprintf("%.4f∠%.0f°", abs(l), rad2deg(angle(l))) for l in λη[1:min(k, end)]), ", "))
    @printf("  %-9s precisions: %3d nonzero, ρ = %.6f\n", "", length(λΛ), isempty(λΛ) ? 0.0 : abs(λΛ[1]))
end

println("1. The spectrum of one sweep at the fixed point")
for (R₁, R₂) in ((2.0, 3.0), (2.0, 2.0))
    g, ps, st = divider_graph(R₁, R₂, 1e4)
    sp = spanning_tree_schedule(g)
    s0 = fixed_store(g, ps, st, sp)
    @printf("\nthe divider, R₁ = %.0f, R₂ = %.0f (%d messages)\n", R₁, R₂, length(tasks(sp)))
    show_spectrum("flooding", sweep_spectrum(g, ps, st, flooding_schedule(g), s0)...)
    show_spectrum("spanning", sweep_spectrum(g, ps, st, sp, s0)...)
end
for (n, pv) in ((3, 10.0), (4, 100.0))
    g, ps, st, _, ρR = grid_graph(n, pv)
    sp = spanning_tree_schedule(g)
    s0 = fixed_store(g, ps, st, sp)
    @printf("\na %d × %d grid, pv = %.0f (%d messages; ρ of the normalised couplings %.4f)\n", n, n, pv, length(tasks(sp)), ρR)
    show_spectrum("flooding", sweep_spectrum(g, ps, st, flooding_schedule(g), s0)...)
    show_spectrum("spanning", sweep_spectrum(g, ps, st, sp, s0)...)
end

println("\n2. Sweeps to a message residual of 1e-11 (at most 20 000), and the largest error of a mean against J \\ h")
for (n, pv) in ((5, 1.0), (5, 100.0), (10, 1.0), (10, 100.0), (10, 10_000.0), (20, 100.0))
    g, ps, st, exact, ρR = grid_graph(n, pv)
    sp = spanning_tree_schedule(g)
    @printf("\n%2d × %-2d grid, pv = %-6.0f (%d messages per sweep; ρ of the normalised couplings %.6f)\n", n, n, pv,
            length(tasks(sp)), ρR)
    for (name, sc) in (("flooding", flooding_schedule(g)), ("spanning", sp),
                       ("spanning+A(2)", AndersonSchedule(sp; m = 2)), ("spanning+A(3)", AndersonSchedule(sp; m = 3)),
                       ("spanning+A(5)", AndersonSchedule(sp; m = 5)), ("flooding+A(3)", AndersonSchedule(flooding_schedule(g); m = 3)))
        t = @elapsed (marg, r, _, s) = infer!(g, sc, ps, st; maxsweeps = 20_000, tol = 1e-11)
        err = maximum(abs(belief_mean(marg[v])[1] - exact[v]) for v in keys(exact))
        restarts = sc isa AndersonSchedule ? anderson_restarts(sc, s) : 0
        @printf("  %-14s %s sweeps, error %.1e%s, %.2f s\n", name, r.converged ? @sprintf("%6d", r.sweeps) : "> 20000",
                err, sc isa AndersonSchedule ? " ($restarts restarts)" : "", t)
    end
end
