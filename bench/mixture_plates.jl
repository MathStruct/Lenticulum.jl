# The E-step of a Gaussian mixture through a Mycelium plate of switching factors, against the
# responsibilities by hand, and the free energy and its gradient on message blocks against the
# same messages per edge. Recorded in `src/switching.md` §10 and L-SPEC-006 §7.
# Run:  julia --project=. bench/mixture_plates.jl
#
# Model: x_i = b_{k_i} + ε, ε ~ N(0, Q), P(k) = softmax(logw); one replica per point: the data, a
# `SwitchingConstraintFactor((z = d,), K, d)` and the outcome k_i. The E-step is P(k_i | x_i).
# Each timing is the minimum over five runs after a warm-up; a sweep times a fresh store and one
# sweep, sequential (edge by edge, as before message blocks) or grouped (`plate_schedule`), and the responsibilities read from it are checked against
# the hand-written ones outside the timings.
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Printf

const D, K = 2, 3
const Q = [0.3 0.05; 0.05 0.2]

# `direction`: Absorbing() for the E-step alone; Bidirectional() also sends each replica's mixture
# message back to its (observed) z, which goes replica by replica (no mixture blocks yet).
function template!(p, x, direction)
    variable!(p, :z, D); variable!(p, :k, K)
    factor!(p, :data, DataFactor(:z, x))
    factor!(p, :sw, SwitchingConstraintFactor((z = D,), K, D; noise = Q))
    connect!(p, :data, :z, :z; direction = Emitting()); connect!(p, :sw, :z, :z; direction); connect!(p, :sw, :k, :k)
end
function plate_graph(X, direction = Absorbing())
    b = GraphBuilder()
    plate!((p, i) -> template!(p, X[:, i], direction), b, :obs, size(X, 2))
    return build(b)
end
function run(g, ps, sched)
    s = MessageStore(g)
    sweep!(s, g, sched, ps, NamedTuple())
    return s
end
function by_hand(X, B, logw)
    L = cholesky(Q).L
    lw = logw .- log(sum(exp.(logw)))
    R = zeros(K, size(X, 2))
    for k in 1:K
        R[k, :] .= lw[k] .- vec(sum(abs2, L \ (X .- B[:, k]); dims = 1)) ./ 2
    end
    R .-= maximum(R; dims = 1)
    P = exp.(R)
    return P ./ sum(P; dims = 1)
end
best(f) = (f(); minimum(@elapsed(f()) for _ in 1:5))

@printf("%6s %12s %12s %12s %12s %10s %12s %12s %12s %12s\n", "N", "sequential", "plate sweep", "both ways", "by hand", "max |Δp|",
        "F blocks", "F per edge", "∇F blocks", "∇F per edge")
for N in (100, 1000, 3000)
    X = randn(Xoshiro(N), D, N) .* 2
    g = plate_graph(X)
    ps0 = LuxCore.initialparameters(Xoshiro(0), g)
    ps = merge(ps0, (sw = merge(ps0.sw, (A = (z = Matrix(1.0I, D, D),),)),))
    sched = plate_schedule(g)
    s = run(g, ps, sched)
    pl = only(plates(g))
    P = reduce(hcat, [exp.(marginal(s, g, pl.variables[2, i]).logp) for i in 1:N])
    Δ = maximum(abs, P .- by_hand(X, ps.sw.b, ps.sw.logw))
    se = MessageStore(copy(s.to_variable), copy(s.to_factor), s.iteration)
    g2 = plate_graph(X, Bidirectional()); s2 = plate_schedule(g2)
    @printf("%6d %10.3f ms %10.3f ms %10.3f ms %10.4f ms %10.1e %10.3f ms %10.3f ms %10.3f ms %10.3f ms\n", N,
            1e3best(() -> run(g, ps, SequentialSchedule(tasks(sched)))), 1e3best(() -> run(g, ps, sched)), 1e3best(() -> run(g2, ps, s2)), 1e3best(() -> by_hand(X, ps.sw.b, ps.sw.logw)), Δ,
            1e3best(() -> scalar_free_energy(s, g, ps, NamedTuple())), 1e3best(() -> scalar_free_energy(se, g, ps, NamedTuple())),
            1e3best(() -> free_energy_gradient(s, g, ps, NamedTuple())), 1e3best(() -> free_energy_gradient(se, g, ps, NamedTuple())))
end
