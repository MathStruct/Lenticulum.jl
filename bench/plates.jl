# The E-step of factor analysis through Mycelium, one graph per point against one plate.
# Recorded in `lib/Mycelium.jl/src/plates.md` ("Speed") and L-SPEC-006.
# Run:  julia --project=. bench/plates.jl
#
# Model: z ~ N(0, I_k), x = W z + c + ε, ε ~ N(0, σ² I_d); the E-step is the posterior of every
# z_i. Five ways: a graph built and run per point (as `fit_hybrid` did on `experiment-qm`), the
# same graphs built once, one plate on a sequential schedule, the plate on its grouped schedule
# (`plate_schedule`), and the closed form vectorised by hand: P = I + WᵀΣ⁻¹W,
# M = P⁻¹WᵀΣ⁻¹(X − c). Each timing is the minimum over five runs after a warm-up.
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Printf

const D, K, σ2 = 3, 2, 0.7

function template!(p, x)
    variable!(p, :z, K); variable!(p, :x, D)
    factor!(p, :prior, GaussianPrior(:z, zeros(K), Matrix(1.0I, K, K)))
    factor!(p, :lik, LinearConstraintFactor((x = D, z = K), D; noise = σ2))
    factor!(p, :data, DataFactor(:x, x))
    connect!(p, :prior, :z, :z); connect!(p, :lik, :z, :z)
    connect!(p, :lik, :x, :x; direction = Absorbing()); connect!(p, :data, :x, :x; direction = Emitting())
end
function plate_graph(X)
    b = GraphBuilder()
    plate!((p, i) -> template!(p, X[:, i]), b, :obs, size(X, 2))
    return build(b)
end
setparams(g, W, c) = (ps = LuxCore.initialparameters(Xoshiro(0), g);
    merge(ps, (lik = merge(ps.lik, (A = (x = Matrix(1.0I, D, D), z = -W), c = c)),)))
zmean(s, g, name) = belief_mean(marginal(s, g, g.variable_index[name]))

function loop_built(X, W, c)
    map(axes(X, 2)) do i
        g = plate_graph(X[:, i:i])
        s = MessageStore(g)
        sweep!(s, g, tree_schedule(g), setparams(g, W, c), NamedTuple())
        zmean(s, g, Symbol("z[1]"))
    end
end
function loop_prebuilt(gs, pss, scheds)
    map(eachindex(gs)) do i
        s = MessageStore(gs[i])
        sweep!(s, gs[i], scheds[i], pss[i], NamedTuple())
        zmean(s, gs[i], Symbol("z[1]"))
    end
end
function plate_run(g, ps, sched, n)
    s = MessageStore(g)
    sweep!(s, g, sched, ps, NamedTuple())
    [zmean(s, g, replica_name(:z, i)) for i in 1:n]
end
function vectorised(X, W, c)
    Wt = W' / σ2
    return (I + Wt * W) \ (Wt * (X .- c))
end

best(f) = (f(); minimum(@elapsed(f()) for _ in 1:5))

@printf("%6s %12s %12s %12s %12s %12s %10s\n", "N", "loop+build", "loop", "plate seq", "plate grp", "by hand", "max |Δμ|")
for N in (100, 300, 1000)
    rng = Xoshiro(N)
    X = randn(rng, D, N); W = randn(rng, D, K); c = randn(rng, D)
    gs = [plate_graph(X[:, i:i]) for i in 1:N]
    pss = [setparams(g, W, c) for g in gs]; scheds = [tree_schedule(g) for g in gs]
    g = plate_graph(X); ps = setparams(g, W, c)
    grouped = plate_schedule(g); seq = SequentialSchedule(tasks(grouped))
    M = vectorised(X, W, c)
    Δ = maximum(abs, reduce(hcat, plate_run(g, ps, grouped, N)) .- M)
    @assert maximum(abs, reduce(hcat, loop_prebuilt(gs, pss, scheds)) .- M) < 1e-10
    @printf("%6d %10.2f ms %10.2f ms %10.2f ms %10.2f ms %10.4f ms %10.1e\n", N,
            1e3best(() -> loop_built(X, W, c)), 1e3best(() -> loop_prebuilt(gs, pss, scheds)),
            1e3best(() -> plate_run(g, ps, seq, N)), 1e3best(() -> plate_run(g, ps, grouped, N)),
            1e3best(() -> vectorised(X, W, c)), Δ)
end
